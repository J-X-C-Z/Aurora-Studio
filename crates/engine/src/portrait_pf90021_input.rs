//! PF90021 (肤色透亮 / skin translucency) same-frame producer inputs.
//!
//! The recovered stage needs a full-frame class-1 skin raster, both 64^3
//! volumes, runtime scalars and receipts bound to the exact pre-effect frame.
//! No PF90021 64^3 LUT asset exists in this tree yet, so both directions fall
//! back to the identity volume built through `VolumeLut64::from_nodes`; the real
//! two-pass contract still runs, but the transfer is neutral until vendor LUT
//! bytes are vendored.

use std::sync::OnceLock;

use aurora_portrait::cuda_pf90021::{
    Direction as Pf90021Direction, RuntimeScalars as Pf90021RuntimeScalars, VolumeLut64, LUT_EDGE,
    LUT_FLOATS,
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

/// `pc:` control is the recovered 0..1 degree amount; non-finite is refused so
/// `resolve_pf90021` never sees a NaN runtime scalar.
fn pf90021_degree01(control: f32) -> Result<f32, Refusal> {
    if !control.is_finite() {
        return Err(Refusal::ValueOutOfRange("PF90021_CONTROL"));
    }
    Ok(control.clamp(0.0, 1.0))
}

/// Mirrors the recovered `param_range` `[0, 0.5, 1]` positive remap
/// `(param - 0.5) / 0.5`; `0.5 + degree * 0.5` is its exact inverse, so
/// `resolve_pf90021` sees `output01 == runtime.degree01` bit-stably.
fn pf90021_param01(degree01: f32) -> f32 {
    0.5 + degree01 * 0.5
}

/// Identity 64^3 volume: `volume[z][y][x] = (x, y, z) / 63`. Both directions
/// share it until real positive/negative vendor volumes are vendored.
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
        let degree01 = pf90021_degree01(control)?;
        let parsing = self
            .model_inputs
            .as_ref()
            .and_then(|inputs| inputs.parsing.as_ref())
            .ok_or(Refusal::MissingInput("FaceParsingV2 parsing planes"))?;
        let mask_r = crate::face_parsing::project_face_parsing_skin_mask(
            parsing,
            source.width,
            source.height,
        )?;
        if self.roi_xywh.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("PF90021_ROI_BOUNDS"));
        }
        let frame =
            PortraitFrameReceipt::from_renderer_frame(&source.pixels, source.width, source.height)?;
        let runtime = Pf90021RuntimeScalars {
            degree01,
            opacity01: 1.0,
            shelter: 1.0,
            region: 1.0,
        };
        runtime.u_alpha()?;
        let same_job = Pf90021SameJobBinding {
            job_id: self.job_id,
            face_id: pf90021_face_id(&self.detection),
            ticket: self.job_id,
            frame,
            mask_texture_id: pf90021_mask_texture_id(self.job_id, frame, &mask_r),
            mask_channel: MASK_CHANNEL,
            mask_source: Pf90021MaskSource::FaceParsingChannel0Rasterized,
            roi_bits: self.roi_xywh.map(|value| value.round().to_bits()),
        };
        let identity = pf90021_identity_volume();
        producer.pf90021 = Some(Pf90021ProducerInputs {
            direction: Pf90021Direction::Positive,
            positive_lut: VolumeLut64::clone(identity),
            negative_lut: VolumeLut64::clone(identity),
            mask_r,
            runtime,
            same_job,
            direction_receipt: Pf90021DirectionReceipt {
                effect_range: [0.0, 0.0, 1.0],
                param_range: [0.0, 0.5, 1.0],
                input01: pf90021_param01(degree01),
                output01: degree01,
            },
            runtime_receipt: Pf90021RuntimeReceipt {
                degree01,
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
    use crate::face_parsing::{
        FaceParsingProducerInputs, FACE_PARSING_V2_CLASSES, FACE_PARSING_V2_MODEL_SHA256,
        FACE_PARSING_V2_SIDE, FACE_PARSING_V2_SKIN_CLASS,
    };

    fn parsing_fixture(skin: u8) -> FaceParsingProducerInputs {
        let side = FACE_PARSING_V2_SIDE as usize;
        let plane = side * side;
        let mut masks = vec![0u8; FACE_PARSING_V2_CLASSES * plane];
        let start = FACE_PARSING_V2_SKIN_CLASS * plane;
        masks[start..start + plane].fill(skin);
        FaceParsingProducerInputs {
            job_id: 7,
            width: FACE_PARSING_V2_SIDE,
            height: FACE_PARSING_V2_SIDE,
            channels: FACE_PARSING_V2_CLASSES as u8,
            model_sha256: FACE_PARSING_V2_MODEL_SHA256,
            source_to_crop: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            masks,
            mask_provider: "cpu",
        }
    }

    #[test]
    fn class1_skin_projects_onto_the_full_source_plane() {
        let mask = crate::face_parsing::project_face_parsing_skin_mask(
            &parsing_fixture(200),
            FACE_PARSING_V2_SIDE,
            FACE_PARSING_V2_SIDE,
        )
        .unwrap();
        assert_eq!(mask.len(), (FACE_PARSING_V2_SIDE * FACE_PARSING_V2_SIDE) as usize);
        assert!(mask.iter().all(|&value| value == 200));
    }

    #[test]
    fn control_remaps_into_the_recovered_degree_range() {
        for (control, degree, param) in [
            (0.0, 0.0, 0.5),
            (0.25, 0.25, 0.625),
            (0.5, 0.5, 0.75),
            (1.0, 1.0, 1.0),
            (2.0, 1.0, 1.0),
            (-1.0, 0.0, 0.5),
        ] {
            let actual = pf90021_degree01(control).unwrap();
            assert!((actual - degree).abs() <= f32::EPSILON, "{control}");
            let input = pf90021_param01(actual);
            assert!((input - param).abs() <= 1e-6, "{control}");
            let remapped = (input - 0.5) / 0.5;
            assert!((remapped - actual).abs() <= 1e-6, "{control}");
        }
        assert_eq!(
            pf90021_degree01(f32::NAN),
            Err(Refusal::ValueOutOfRange("PF90021_CONTROL"))
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
