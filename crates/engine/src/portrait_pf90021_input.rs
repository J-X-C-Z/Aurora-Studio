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
//! from [`PF90021_LUT_POSITIVE_ENV`] / [`PF90021_LUT_NEGATIVE_ENV`] or the
//! decoded vendor effect root. Non-neutral effects require both vendor atlases.
//!
//! The verified renderer input is `LipMaskType=3`. The optional FaceParsingV2
//! class-8 route is separately identified because the original live mask
//! texture and projection are still unverified.

use std::path::{Path, PathBuf};
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
pub const PF90021_LIP_MASK: &str = "PF90021_LIP_MASK";
pub const PF90021_FACE_PARSING_LIP_ENV: &str = "AURORA_PF90021_FACE_PARSING_LIP";

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

fn pf90021_lut_path(override_path: Option<PathBuf>, root: Option<&Path>, positive: bool) -> Option<PathBuf> {
    override_path.filter(|path| !path.as_os_str().is_empty()).or_else(|| {
        let name = if positive { "LipsLUT.png" } else { "LipDodgeLUT.png" };
        root.map(|root| root.join("MakeupEnhance").join(name))
    })
}

fn pf90021_read_lut(path: &Path) -> Option<VolumeLut64> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) => {
            log::warn!("PF90021: cannot read {}: {error}", path.display());
            return None;
        }
    };
    match pf90021_decode_lut_atlas(&bytes) {
        Ok(volume) => Some(volume),
        Err(refusal) => {
            log::warn!(
                "PF90021: {} is not a 512x512 LUT atlas ({refusal:?})",
                path.display()
            );
            None
        }
    }
}

/// Caches the env-rooted atlas, including a failed/missing lookup.
fn pf90021_vendor_volume(positive: bool) -> Option<&'static VolumeLut64> {
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
    cell.get_or_init(|| {
        let explicit = std::env::var_os(env).map(PathBuf::from);
        let root = crate::portrait_vendor_config::root_from_env()
            .or_else(crate::portrait_makeup_wiring::makeup_asset_root);
        let path = pf90021_lut_path(explicit, root.as_deref(), positive)?;
        pf90021_read_lut(&path)
    }).as_ref()
}

fn pf90021_volumes(
    direction: Pf90021Direction,
    load: impl Fn(bool) -> Option<&'static VolumeLut64>,
) -> Result<(&'static VolumeLut64, &'static VolumeLut64), Refusal> {
    if direction == Pf90021Direction::Neutral {
        let neutral = pf90021_identity_volume();
        return Ok((neutral, neutral));
    }
    let positive = load(true).ok_or(Refusal::MissingInput("PF90021_POSITIVE_LUT64"))?;
    let negative = load(false).ok_or(Refusal::MissingInput("PF90021_NEGATIVE_LUT64"))?;
    Ok((positive, negative))
}

/// Identity 64^3 volume: `volume[z][y][x] = (x, y, z) / 63`.
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

fn pf90021_face_parsing_lip_enabled(env_value: Option<&std::ffi::OsStr>) -> bool {
    !env_value.is_some_and(|v| v == "0" || v.eq_ignore_ascii_case("false"))
}

fn pf90021_lip_mask(
    enabled: bool,
    parsing: Option<&crate::face_parsing::FaceParsingProducerInputs>,
    job_id: u64,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, Refusal> {
    let parsing = parsing.filter(|_| enabled).ok_or(Refusal::MissingInput(PF90021_LIP_MASK))?;
    crate::face_parsing::project_face_parsing_lip_mask(parsing, job_id, width, height)
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
        let (pos_lut, neg_lut) = pf90021_volumes(direction, pf90021_vendor_volume)?;
        let (mask_r, mask_source) = match direction {
            Pf90021Direction::Neutral => (Vec::new(), Pf90021MaskSource::LipMaskType3ChannelR),
            Pf90021Direction::Positive | Pf90021Direction::Negative => {
                let enabled = pf90021_face_parsing_lip_enabled(
                    std::env::var_os(PF90021_FACE_PARSING_LIP_ENV).as_deref(),
                );
                let parsing = self.model_inputs.as_ref().and_then(|inputs| inputs.parsing.as_ref());
                (pf90021_lip_mask(enabled, parsing, self.job_id, source.width, source.height)?,
                    Pf90021MaskSource::FaceParsingV2Class8Candidate)
            }
        };
        let same_job = Pf90021SameJobBinding {
            job_id: self.job_id,
            face_id: pf90021_face_id(&self.detection),
            ticket: self.job_id,
            frame,
            mask_texture_id: pf90021_mask_texture_id(self.job_id, frame, &mask_r),
            mask_channel: MASK_CHANNEL,
            mask_source,
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
    fn vendor_root_resolves_both_atlases_and_preserves_explicit_override() {
        let root = Path::new("decoded/config");
        assert_eq!(pf90021_lut_path(None, Some(root), true), Some(root.join("MakeupEnhance").join("LipsLUT.png")));
        assert_eq!(pf90021_lut_path(None, Some(root), false), Some(root.join("MakeupEnhance").join("LipDodgeLUT.png")));
        let override_path = PathBuf::from("custom/vendor.png");
        assert_eq!(pf90021_lut_path(Some(override_path.clone()), Some(root), false), Some(override_path));
        assert_eq!(pf90021_lut_path(None, None, true), None);
    }

    #[test]
    fn non_neutral_requires_both_vendor_atlases() {
        for direction in [Pf90021Direction::Positive, Pf90021Direction::Negative] {
            assert!(matches!(
                pf90021_volumes(direction, |_| None),
                Err(Refusal::MissingInput("PF90021_POSITIVE_LUT64"))
            ));
            assert!(matches!(
                pf90021_volumes(direction, |positive| positive.then(pf90021_identity_volume)),
                Err(Refusal::MissingInput("PF90021_NEGATIVE_LUT64"))
            ));
        }
        assert!(pf90021_volumes(Pf90021Direction::Neutral, |_| panic!("neutral should not load a LUT")).is_ok());
    }

    #[test]
    fn non_neutral_controls_refuse_the_missing_lip_layer() {
        assert_eq!(
            pf90021_lip_mask(true, None, 17, 2, 1),
            Err(Refusal::MissingInput(PF90021_LIP_MASK))
        );
    }

    #[test]
    fn class8_lip_candidate_requires_opt_in_and_matching_frame_ticket() {
        use crate::face_parsing::{FaceParsingProducerInputs, FaceParsingRoi, FACE_PARSING_V2_CLASSES,
            FACE_PARSING_V2_MODEL_SHA256, FACE_PARSING_V2_SIDE};
        let pixels = (FACE_PARSING_V2_SIDE as usize).pow(2);
        let mut masks = vec![0; FACE_PARSING_V2_CLASSES * pixels];
        masks[7 * pixels..8 * pixels].fill(255);
        masks[8 * pixels..9 * pixels].fill(201);
        masks[9 * pixels..10 * pixels].fill(255);
        let parsing = FaceParsingProducerInputs {
            job_id: 17,
            width: FACE_PARSING_V2_SIDE,
            height: FACE_PARSING_V2_SIDE,
            channels: FACE_PARSING_V2_CLASSES as u8,
            model_sha256: FACE_PARSING_V2_MODEL_SHA256,
            source_to_crop: FaceParsingRoi { x: 0.0, y: 0.0, width: 2.0, height: 1.0, roll_radians: 0.0 }
                .source_to_crop().unwrap(),
            masks,
            mask_provider: "cpu",
        };
        assert_eq!(pf90021_lip_mask(false, Some(&parsing), 17, 2, 1), Err(Refusal::MissingInput(PF90021_LIP_MASK)));
        assert_eq!(pf90021_lip_mask(true, Some(&parsing), 18, 2, 1), Err(Refusal::ValueOutOfRange("FaceParsingV2 lip job ticket")));
        assert_eq!(pf90021_lip_mask(true, Some(&parsing), 17, 2, 1).unwrap(), vec![201, 201]);
        let mut stale = parsing;
        stale.model_sha256 = [0; 32];
        assert!(matches!(pf90021_lip_mask(true, Some(&stale), 17, 2, 1), Err(Refusal::ValueOutOfRange(_))));
    }

    #[test]
    fn face_parsing_lip_env_defaults_enabled_and_honors_explicit_disable() {
        use std::ffi::OsStr;
        assert!(pf90021_face_parsing_lip_enabled(None));
        assert!(pf90021_face_parsing_lip_enabled(Some(OsStr::new("1"))));
        assert!(pf90021_face_parsing_lip_enabled(Some(OsStr::new("true"))));
        assert!(!pf90021_face_parsing_lip_enabled(Some(OsStr::new("0"))));
        assert!(!pf90021_face_parsing_lip_enabled(Some(OsStr::new("false"))));
        assert!(!pf90021_face_parsing_lip_enabled(Some(OsStr::new("FALSE"))));
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
