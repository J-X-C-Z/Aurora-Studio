//! PF90021 (`LipStickLutMakeup`) same-frame producer inputs.
//!
//! The recovered `MakeupEnhance/effect.json` row is a lip makeup enhance LUT,
//! not a skin-translucency effect: positive `Paths=[0000.png,LipsLUT.png]`,
//! negative `[0000.png,LipDodgeLUT.png]`, `FilterType=71`, `Opacity=80`,
//! `ROIRect=[350,800,300,120]`, `LipMaskType=3`, `DefaultDegree=50`,
//! `ParamRange=[0,0.5,1]`, positive `EffectRange=[0,0,1]` and negative
//! `EffectRange=[1,0,0]`.
//!
//! Both 512x512 vendor atlases decode through
//! [`VolumeLut64::from_atlas_rgb8`] into 64^3 float volumes. Their paths come
//! from [`PF90021_LUT_POSITIVE_ENV`] / [`PF90021_LUT_NEGATIVE_ENV`]; a missing
//! or undecodable file is cached and falls back to the synthetic volume so the
//! two-pass contract still runs without inventing vendor bytes.
//!
//! The effect region must be the recovered `LipMaskType=3` lip mask. This tree
//! has no lip-layer producer: FaceParsingV2 has 17 planes but no recovered lip
//! label, and the suit owner-mask map explicitly refuses classes 1/3/6/8. The
//! producer therefore reports `PF90021_LIP_MASK` instead of silently reusing
//! the class-1 skin plane.

use std::sync::OnceLock;

use aurora_portrait::cuda_pf90021::{
    Direction as Pf90021Direction, RuntimeScalars as Pf90021RuntimeScalars, VolumeLut64, ATLAS_EDGE,
    LUT_EDGE, LUT_FLOATS,
};
use aurora_portrait::Refusal;

use crate::C1FaceDetection;
use crate::portrait_stage::{
    Pf90021DirectionReceipt, Pf90021MaskSource, Pf90021ProducerInputs, Pf90021RuntimeReceipt,
    Pf90021SameJobBinding, PortraitFrameReceipt,
};

const MASK_CHANNEL: u8 = 0;
const FNV1A64_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV1A64_PRIME: u64 = 0x100000001b3;

/// Environment override for the positive `LipsLUT` atlas (PNG or JPEG).
pub const PF90021_LUT_POSITIVE_ENV: &str = "AURORA_PF90021_LUT_POSITIVE";
/// Environment override for the negative `LipDodgeLUT` atlas (PNG or JPEG).
pub const PF90021_LUT_NEGATIVE_ENV: &str = "AURORA_PF90021_LUT_NEGATIVE";
/// Named refusal: the recovered `LipMaskType=3` lip layer has no producer yet.
pub const PF90021_LIP_MASK: &str = "PF90021_LIP_MASK";

/// Recovered `Opacity: 80` (percent, not 0..1).
pub const PF90021_OPACITY_PERCENT: f32 = 80.0;
/// Recovered `ROIRect: [350, 800, 300, 120]`.
pub const PF90021_ROI_XYWH: [f32; 4] = [350.0, 800.0, 300.0, 120.0];
/// Recovered `ParamRange`.
pub const PF90021_PARAM_RANGE: [f32; 3] = [0.0, 0.5, 1.0];
/// Recovered positive row `EffectRange`.
pub const PF90021_POSITIVE_EFFECT_RANGE: [f32; 3] = [0.0, 0.0, 1.0];
/// Recovered negative row `EffectRange`.
pub const PF90021_NEGATIVE_EFFECT_RANGE: [f32; 3] = [1.0, 0.0, 0.0];
/// `ShelterType: 0` binds no shelter channel on the recovered main-only filter
/// object; the runtime shelter factor is therefore the neutral pass-through.
pub const PF90021_SHELTER01: f32 = 1.0;
/// The recovered region factor is folded into the lip mask. The config
/// `ROIRect` is recorded separately; its source-frame mapping is still open.
pub const PF90021_REGION01: f32 = 1.0;

fn fnv1a64(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    hash
}

fn fnv_u64(hash: u64, value: u64) -> u64 {
    fnv1a64(hash, &value.to_le_bytes())
}

/// `pc:` control is the recovered 0..1 degree amount (`DegreeRange [0,100]`);
/// non-finite is refused so `resolve_pf90021` never sees a NaN runtime scalar.
fn pf90021_degree01(control: f32) -> Result<f32, Refusal> {
    if !control.is_finite() {
        return Err(Refusal::ValueOutOfRange("PF90021_CONTROL"));
    }
    Ok(control.clamp(0.0, 1.0))
}

/// Selects the recovered effect row from the control sign: values below the
/// neutral midpoint use the `EffectRange=[1,0,0]` negative atlas, values above
/// use `[0,0,1]`, and the midpoint is a real no-launch bypass.
fn pf90021_direction_for_control(control01: f32) -> Pf90021Direction {
    if control01 > 0.5 {
        Pf90021Direction::Positive
    } else if control01 < 0.5 {
        Pf90021Direction::Negative
    } else {
        Pf90021Direction::Neutral
    }
}

/// Returns the selected row, its `input01` (the config parameter) and the
/// remapped `output01` degree. The remap is `portrait_stage`'s recovered
/// `ParamRange [0,0.5,1]` -> `EffectRange` piecewise map, so the receipt and
/// the stage agree bit-stably.
fn pf90021_direction_and_degree(
    control: f32,
) -> Result<(Pf90021Direction, f32, f32), Refusal> {
    let input01 = pf90021_degree01(control)?;
    let direction = pf90021_direction_for_control(input01);
    let output01 = match direction {
        Pf90021Direction::Neutral => 0.0,
        Pf90021Direction::Positive | Pf90021Direction::Negative => {
            crate::portrait_stage::pf90021_degree_remap(direction, input01)
        }
    };
    Ok((direction, input01, output01))
}

fn pf90021_effect_range(direction: Pf90021Direction) -> [f32; 3] {
    match direction {
        Pf90021Direction::Neutral => [0.0; 3],
        Pf90021Direction::Positive => PF90021_POSITIVE_EFFECT_RANGE,
        Pf90021Direction::Negative => PF90021_NEGATIVE_EFFECT_RANGE,
    }
}

/// Decodes one vendor 512x512 LUT atlas into the typed 64^3 volume. The tile
/// origin math lives in `VolumeLut64::from_atlas_rgb8`; alpha is dropped by the
/// explicit `to_rgb8()`.
fn pf90021_decode_lut_atlas(bytes: &[u8]) -> Result<VolumeLut64, Refusal> {
    let decoded = image::load_from_memory(bytes)
        .map_err(|_| Refusal::ShapeMismatch("PF90021_LUT_ATLAS_IMAGE"))?;
    if decoded.width() != ATLAS_EDGE as u32 || decoded.height() != ATLAS_EDGE as u32 {
        return Err(Refusal::ShapeMismatch("PF90021_LUT_ATLAS_IMAGE"));
    }
    let rgb = decoded.to_rgb8();
    VolumeLut64::from_atlas_rgb8(rgb.as_raw())
}

/// Reads and decodes one env-rooted atlas. Every failure path returns `None`
/// after a warning; callers cache that `None` so a bad path is read once.
fn pf90021_lut_from_env_value(env: &str) -> Option<VolumeLut64> {
    let path = std::env::var_os(env)?;
    if path.is_empty() {
        return None;
    }
    let path = std::path::PathBuf::from(path);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) => {
            log::warn!("PF90021 {env}: cannot read {}: {error}", path.display());
            return None;
        }
    };
    match pf90021_decode_lut_atlas(&bytes) {
        Ok(volume) => Some(volume),
        Err(refusal) => {
            log::warn!(
                "PF90021 {env}: {} is not a 512x512 LUT atlas ({refusal:?})",
                path.display()
            );
            None
        }
    }
}

/// Caches the env-rooted atlas, including a failed/missing lookup.
fn pf90021_env_volume(positive: bool) -> Option<&'static VolumeLut64> {
    static POS_VOLUME: OnceLock<Option<VolumeLut64>> = OnceLock::new();
    static NEG_VOLUME: OnceLock<Option<VolumeLut64>> = OnceLock::new();
    let cell: &'static OnceLock<Option<VolumeLut64>> = if positive {
        &POS_VOLUME
    } else {
        &NEG_VOLUME
    };
    let env = if positive {
        PF90021_LUT_POSITIVE_ENV
    } else {
        PF90021_LUT_NEGATIVE_ENV
    };
    cell.get_or_init(|| pf90021_lut_from_env_value(env))
        .as_ref()
}

/// Picks the real atlas when present, otherwise the synthetic fallback.
fn pf90021_select_volume<'a>(
    loaded: Option<&'a VolumeLut64>,
    positive: bool,
) -> &'a VolumeLut64 {
    loaded.unwrap_or_else(|| pf90021_synthetic_volume(positive))
}

fn pf90021_volume(positive: bool) -> &'static VolumeLut64 {
    pf90021_select_volume(pf90021_env_volume(positive), positive)
}

/// Synthetic fallback used only when the vendor atlas is absent. It is a mild
/// lip midtone lift, not the vendor transfer; the receipt still records the
/// real config row, so the substitution is visible.
fn pf90021_synthetic_volume(positive: bool) -> &'static VolumeLut64 {
    static POS_VOLUME: OnceLock<VolumeLut64> = OnceLock::new();
    static NEG_VOLUME: OnceLock<VolumeLut64> = OnceLock::new();
    let cell = if positive { &POS_VOLUME } else { &NEG_VOLUME };
    cell.get_or_init(|| {
        let denominator = (LUT_EDGE - 1) as f32;
        let strength = if positive { 0.14f32 } else { -0.10f32 };
        let mut nodes = vec![0.0f32; LUT_FLOATS];
        for z in 0..LUT_EDGE {
            let b = z as f32 / denominator;
            for y in 0..LUT_EDGE {
                let g = y as f32 / denominator;
                for x in 0..LUT_EDGE {
                    let r = x as f32 / denominator;
                    let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                    let mid_weight = 4.0 * luma * (1.0 - luma);
                    let lift = strength * mid_weight;
                    let sat_pull = if positive { 0.08 * mid_weight } else { 0.0 };
                    let at = ((z * LUT_EDGE + y) * LUT_EDGE + x) * 3;
                    nodes[at] = (r + lift + (luma - r) * sat_pull).clamp(0.0, 1.0);
                    nodes[at + 1] = (g + lift * 0.96 + (luma - g) * sat_pull).clamp(0.0, 1.0);
                    nodes[at + 2] = (b + lift * 1.04 + (luma - b) * sat_pull).clamp(0.0, 1.0);
                }
            }
        }
        VolumeLut64::from_nodes(nodes).expect("PF90021 64^3 synthetic volume")
    })
}

/// Identity 64^3 volume: `volume[z][y][x] = (x, y, z) / 63`.
#[allow(dead_code)]
fn pf90021_identity_volume() -> &'static VolumeLut64 {
    static VOLUME: OnceLock<VolumeLut64> = OnceLock::new();
    VOLUME.get_or_init(|| {
        let denominator = (LUT_EDGE - 1) as f32;
        let mut nodes = vec![0.0f32; LUT_FLOATS];
        for z in 0..LUT_EDGE {
            for y in 0..LUT_EDGE {
                for x in 0..LUT_EDGE {
                    let at = ((z * LUT_EDGE + y) * LUT_EDGE + x) * 3;
                    nodes[at] = x as f32 / denominator;
                    nodes[at + 1] = y as f32 / denominator;
                    nodes[at + 2] = z as f32 / denominator;
                }
            }
        }
        VolumeLut64::from_nodes(nodes).expect("identity PF90021 64^3 volume")
    })
}

/// The recovered `LipMaskType=3` main mask is a lip plane. No producer exists
/// in this tree (FaceParsingV2 has no recovered lip label), so refuse by name
/// rather than substitute the class-1 skin plane.
fn pf90021_lip_mask() -> Result<Vec<u8>, Refusal> {
    Err(Refusal::MissingInput(PF90021_LIP_MASK))
}

/// Stable face identity for the same-job receipt. `C1FaceDetection` has no
/// numeric slot id, so the accepted detection's landmarks and score are hashed;
/// a non-finite detection reports 0 (single-face producer default).
fn pf90021_face_id(detection: &C1FaceDetection) -> i32 {
    let points = detection.landmarks_source;
    if !detection.score.is_finite() || points.iter().flatten().any(|value| !value.is_finite()) {
        return 0;
    }
    let mut hash = fnv1a64(FNV1A64_OFFSET_BASIS, b"aurora-pf90021-face-v1");
    hash = fnv_u64(hash, u64::from(detection.score.to_bits()));
    for point in points {
        for axis in point {
            hash = fnv_u64(hash, u64::from(axis.to_bits()));
        }
    }
    (hash & 0x7fff_ffff) as i32
}

fn pf90021_mask_texture_id(job_id: u64, frame: PortraitFrameReceipt, mask_r: &[u8]) -> u64 {
    let mut hash = fnv1a64(FNV1A64_OFFSET_BASIS, b"aurora-pf90021-mask-v1");
    hash = fnv_u64(hash, job_id);
    hash = fnv_u64(hash, frame.fingerprint());
    hash = fnv_u64(hash, u64::from(MASK_CHANNEL));
    fnv1a64(hash, mask_r).max(1)
}

impl crate::portrait_face_input_stage::FaceInputBundle {
    pub fn attach_pf90021_inputs(
        &self,
        source: &aurora_portrait::Image,
        control: f32,
        producer: &mut crate::portrait_stage::ProducerInputs,
    ) -> Result<(), aurora_portrait::Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        let (direction, input01, output01) = pf90021_direction_and_degree(control)?;
        let frame =
            PortraitFrameReceipt::from_renderer_frame(&source.pixels, source.width, source.height)?;
        let runtime = Pf90021RuntimeScalars {
            degree01: output01,
            opacity01: PF90021_OPACITY_PERCENT / 100.0,
            shelter: PF90021_SHELTER01,
            region: PF90021_REGION01,
        };
        runtime.u_alpha()?;
        // Load both real atlases (or their cached fallback) before the mask
        // gate so the env wiring is exercised on the same call path.
        let pos_lut = pf90021_volume(true);
        let neg_lut = pf90021_volume(false);
        let mask_r = match direction {
            Pf90021Direction::Neutral => Vec::new(),
            Pf90021Direction::Positive | Pf90021Direction::Negative => pf90021_lip_mask()?,
        };
        let same_job = Pf90021SameJobBinding {
            job_id: self.job_id,
            face_id: pf90021_face_id(&self.detection),
            ticket: self.job_id,
            frame,
            mask_texture_id: pf90021_mask_texture_id(self.job_id, frame, &mask_r),
            mask_channel: MASK_CHANNEL,
            mask_source: Pf90021MaskSource::LipMaskType3ChannelR,
            roi_bits: PF90021_ROI_XYWH.map(f32::to_bits),
        };
        producer.pf90021 = Some(Pf90021ProducerInputs {
            direction,
            positive_lut: VolumeLut64::clone(pos_lut),
            negative_lut: VolumeLut64::clone(neg_lut),
            mask_r,
            runtime,
            same_job,
            direction_receipt: Pf90021DirectionReceipt {
                effect_range: pf90021_effect_range(direction),
                param_range: PF90021_PARAM_RANGE,
                input01,
                output01,
            },
            runtime_receipt: Pf90021RuntimeReceipt {
                degree01: runtime.degree01,
                opacity01: runtime.opacity01,
                shelter01: runtime.shelter,
                region01: runtime.region,
            },
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn direction_mapping_follows_the_two_effect_rows() {
        for (control, direction, expected_input, expected_output) in [
            (0.0, Pf90021Direction::Negative, 0.0, 1.0),
            (0.25, Pf90021Direction::Negative, 0.25, 0.5),
            (0.5, Pf90021Direction::Neutral, 0.5, 0.0),
            (0.75, Pf90021Direction::Positive, 0.75, 0.5),
            (1.0, Pf90021Direction::Positive, 1.0, 1.0),
            (2.0, Pf90021Direction::Positive, 1.0, 1.0),
            (-1.0, Pf90021Direction::Negative, 0.0, 1.0),
        ] {
            let (actual_direction, input01, output01) =
                pf90021_direction_and_degree(control).unwrap();
            assert_eq!(actual_direction, direction, "{control}");
            assert_eq!(input01.to_bits(), (expected_input as f32).to_bits(), "{control}");
            assert_eq!(output01.to_bits(), (expected_output as f32).to_bits(), "{control}");
        }
        assert_eq!(
            pf90021_direction_and_degree(f32::NAN),
            Err(Refusal::ValueOutOfRange("PF90021_CONTROL"))
        );
    }

    #[test]
    fn direction_selects_the_recovered_effect_range() {
        assert_eq!(
            pf90021_effect_range(Pf90021Direction::Positive),
            PF90021_POSITIVE_EFFECT_RANGE
        );
        assert_eq!(
            pf90021_effect_range(Pf90021Direction::Negative),
            PF90021_NEGATIVE_EFFECT_RANGE
        );
        assert_eq!(
            pf90021_effect_range(Pf90021Direction::Neutral),
            [0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn config_opacity_and_roi_are_used_verbatim() {
        assert_eq!(PF90021_OPACITY_PERCENT / 100.0, 0.8);
        assert_eq!(PF90021_ROI_XYWH, [350.0, 800.0, 300.0, 120.0]);
        assert_eq!(PF90021_PARAM_RANGE, [0.0, 0.5, 1.0]);
        assert_eq!(PF90021_SHELTER01, 1.0);
        assert_eq!(PF90021_REGION01, 1.0);
    }

    #[test]
    fn vendor_atlas_decodes_to_the_64_cube_float_volume() {
        let (x, y, z) = (11usize, 7usize, 9usize);
        let mut raw = vec![0u8; ATLAS_EDGE * ATLAS_EDGE * 3];
        let at = (((z >> 3) * LUT_EDGE + y) * ATLAS_EDGE + (z & 7) * LUT_EDGE + x) * 3;
        raw[at..at + 3].copy_from_slice(&[10, 20, 30]);
        let atlas = image::RgbImage::from_raw(ATLAS_EDGE as u32, ATLAS_EDGE as u32, raw).unwrap();
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(atlas)
            .write_to(
                &mut std::io::Cursor::new(&mut png),
                image::ImageFormat::Png,
            )
            .unwrap();

        let volume = pf90021_decode_lut_atlas(&png).unwrap();
        assert_eq!(volume.nodes().len(), LUT_FLOATS);
        let node = ((z * LUT_EDGE + y) * LUT_EDGE + x) * 3;
        assert_eq!(
            &volume.nodes()[node..node + 3],
            &[10.0 / 255.0, 20.0 / 255.0, 30.0 / 255.0]
        );
        assert_eq!(
            pf90021_decode_lut_atlas(&[0, 1, 2]).unwrap_err(),
            Refusal::ShapeMismatch("PF90021_LUT_ATLAS_IMAGE")
        );
    }

    #[test]
    fn missing_env_lut_selects_the_cached_synthetic_fallback() {
        let env = "AURORA_PF90021_LUT_POSITIVE_TEST_MISSING_9C1D";
        assert!(std::env::var_os(env).is_none());
        assert!(pf90021_lut_from_env_value(env).is_none());

        let synthetic = pf90021_synthetic_volume(true);
        assert!(std::ptr::eq(
            pf90021_select_volume(None, true),
            synthetic
        ));
        assert_ne!(
            pf90021_synthetic_volume(true).nodes(),
            pf90021_synthetic_volume(false).nodes(),
            "the fallback must not collapse both directions into identity"
        );
    }

    #[test]
    fn volume_loader_returns_a_typed_64_cube_for_both_directions() {
        assert_eq!(pf90021_volume(true).nodes().len(), LUT_FLOATS);
        assert_eq!(pf90021_volume(false).nodes().len(), LUT_FLOATS);
        // The second call hits the cached slot instead of re-reading the env.
        assert!(std::ptr::eq(pf90021_volume(true), pf90021_volume(true)));
    }

    #[test]
    fn non_neutral_controls_refuse_the_missing_lip_layer() {
        assert_eq!(
            pf90021_lip_mask(),
            Err(Refusal::MissingInput(PF90021_LIP_MASK))
        );
    }

    #[test]
    fn identity_volume_is_the_neutral_transfer() {
        let volume = pf90021_identity_volume();
        for (x, y, z) in [(0usize, 0, 0), (17, 33, 63)] {
            let at = ((z * LUT_EDGE + y) * LUT_EDGE + x) * 3;
            let expected = [x as f32 / 63.0, y as f32 / 63.0, z as f32 / 63.0];
            assert_eq!(&volume.nodes()[at..at + 3], &expected[..]);
        }
    }
}
