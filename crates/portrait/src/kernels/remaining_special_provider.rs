//! Typed provider boundary for the remaining special rows.
//!
//! This module does not add vendor pixel laws. It exposes the same-job inputs
//! that are already understood, delegates to existing recovered kernels where
//! one exists, and refuses before image mutation where the static PE has no
//! final pixel consumer.

use crate::{Image, Refusal};

use super::{body_geometry, skin_adjust};

pub const PF90001_PIXEL_LAW_OPEN: &str =
    "PF90001 requires the SkinDB service response and final model-to-source blend";
pub const PF90003_PIXEL_LAW_OPEN: &str =
    "PF90003 TradBodySkinTone curve, LUT, shader and writeback are END_STATIC";
pub const PF101_LIVE_ONLY: &str =
    "PF101 requires the live Silkworm mask, LUT, blur pass and final shader binding; H3 PE candidates RVA 0xB825/0xBAB6/0xC27A are only `cmp al, 0x65` immediates with no semantic xref, program, FBO or readback";
pub const PF30209_END_STATIC: &str =
    "PF30209 stage +0x3A0/+0x3B8 is not forwarded by the fixed PE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderStatus {
    TypedProducer,
    EndStatic,
    LiveOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemainingSpecialContract {
    pub pf: u32,
    pub status: ProviderStatus,
    pub required_same_job_inputs: &'static [&'static str],
    pub reason: &'static str,
}

pub const CONTRACTS: [RemainingSpecialContract; 7] = [
    RemainingSpecialContract {
        pf: 90001,
        status: ProviderStatus::LiveOnly,
        required_same_job_inputs: &["source", "ui", "per_face_engine", "SkinDB response"],
        reason: PF90001_PIXEL_LAW_OPEN,
    },
    RemainingSpecialContract {
        pf: 90003,
        status: ProviderStatus::EndStatic,
        required_same_job_inputs: &[
            "source_rgb",
            "body_mask",
            "roi_mask",
            "native 0x15F93",
            "TradBodySkinTone curve/LUT",
        ],
        reason: PF90003_PIXEL_LAW_OPEN,
    },
    RemainingSpecialContract {
        pf: 403,
        status: ProviderStatus::TypedProducer,
        required_same_job_inputs: &["source_mask", "radius01"],
        reason: "PF403 uses the recovered Dilate -> Gaussian -> MeanBlur mask kernel",
    },
    RemainingSpecialContract {
        pf: 101,
        status: ProviderStatus::LiveOnly,
        required_same_job_inputs: &[
            "eye_mask",
            "eye_mask_type",
            "face_blur_factor",
            "lut",
            "program/attribute/FBO/readback",
        ],
        reason: PF101_LIVE_ONLY,
    },
    RemainingSpecialContract {
        pf: 30188,
        status: ProviderStatus::TypedProducer,
        required_same_job_inputs: &["native PF90034", "native PF30188"],
        reason: "recovered producer is PF90034 * PF30188; final model output is open",
    },
    RemainingSpecialContract {
        pf: 30189,
        status: ProviderStatus::TypedProducer,
        required_same_job_inputs: &["native PF90034", "native PF30189"],
        reason: "recovered producer is PF90034 * PF30189; final shadow composite is open",
    },
    RemainingSpecialContract {
        pf: 30209,
        status: ProviderStatus::EndStatic,
        required_same_job_inputs: &[
            "face_194_packet",
            "jawline_indices",
            "mask",
            "model_output",
        ],
        reason: PF30209_END_STATIC,
    },
];

pub fn contract(pf: u32) -> Option<RemainingSpecialContract> {
    CONTRACTS.iter().copied().find(|row| row.pf == pf)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoubleChinWeights {
    pub geometry: f32,
    pub shadow: f32,
}

/// Static PF30188/30189 producer. These are already-native values; no UI scale
/// or clamp is inferred here. The caller still owns the model and ROI stages.
pub fn double_chin_weights(
    native_pf90034: f32,
    native_pf30188: f32,
    native_pf30189: f32,
) -> Result<DoubleChinWeights, Refusal> {
    if !native_pf90034.is_finite()
        || !native_pf30188.is_finite()
        || !native_pf30189.is_finite()
    {
        return Err(Refusal::ValueOutOfRange("double chin native values"));
    }
    Ok(DoubleChinWeights {
        geometry: native_pf90034 * native_pf30188,
        shadow: native_pf90034 * native_pf30189,
    })
}

/// PF403 delegates to the recovered source-mask kernel.
pub fn enhance_breast_mask(source_mask: &Image, radius01: f32) -> Result<Image, Refusal> {
    body_geometry::enhance_breast_mask(source_mask, radius01)
}

/// PF90001 delegates only to the existing gate. An active non-zero request
/// remains refused until the SkinDB response has a recovered blend law.
pub fn texture_smooth(
    source: &Image,
    ui: f32,
    per_face_engine: &[f32],
) -> Result<Image, Refusal> {
    skin_adjust::texture_smooth_apply(source, ui, per_face_engine)
}

/// PF90003 validates the same-job shape and permits only the zero-control
/// identity. It deliberately does not call the old explicit-target shortcut.
pub fn body_skin_uniform_contract(
    source_rgb: &[u8],
    width: u32,
    height: u32,
    body_mask: &[u8],
    roi_mask: &[u8],
    ui: f32,
    native_0x15f93: f32,
) -> Result<Vec<u8>, Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("source_rgb"));
    }
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("source_rgb"))?;
    if source_rgb.len() != pixels * 3 {
        return Err(Refusal::ShapeMismatch("source_rgb"));
    }
    if body_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("body_mask"));
    }
    if roi_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("roi_mask"));
    }
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if ui == 0.0 {
        return Ok(source_rgb.to_vec());
    }
    if !native_0x15f93.is_finite() {
        return Err(Refusal::ValueOutOfRange("native 0x15F93"));
    }
    Err(Refusal::MissingInput(PF90003_PIXEL_LAW_OPEN))
}

#[derive(Clone, Copy, Debug)]
pub struct Pf101SameJobInputs<'a> {
    pub eye_mask: &'a [u8],
    pub eye_mask_type: u32,
    pub face_blur_factor: f32,
    pub lut: &'a [u8],
}

/// PF101 remains live-only: even structurally complete inputs cannot select a
/// final program/pass sequence from the fixed PE.
pub fn silkworm(inputs: Pf101SameJobInputs<'_>) -> Result<Image, Refusal> {
    if inputs.eye_mask.is_empty() {
        return Err(Refusal::MissingInput("PF101 eye_mask"));
    }
    if inputs.lut.is_empty() {
        return Err(Refusal::MissingInput("PF101 lut"));
    }
    if !inputs.face_blur_factor.is_finite() {
        return Err(Refusal::ValueOutOfRange("PF101 face_blur_factor"));
    }
    let _eye_mask_type = inputs.eye_mask_type;
    Err(Refusal::MissingInput(PF101_LIVE_ONLY))
}

#[derive(Clone, Copy, Debug)]
pub struct Pf30209SameJobInputs<'a> {
    pub face_194_packet: &'a [f32],
    pub jawline_indices: &'a [u16],
    pub mask: &'a [u8],
    pub model_output: &'a [f32],
}

/// PF30209 is END_STATIC at the fixed PE boundary. Inputs are checked only so
/// callers cannot mistake an empty placeholder for a live-only residual.
pub fn jawline(inputs: Pf30209SameJobInputs<'_>) -> Result<Image, Refusal> {
    if inputs.face_194_packet.is_empty() {
        return Err(Refusal::MissingInput("PF30209 face_194_packet"));
    }
    if inputs.jawline_indices.is_empty() {
        return Err(Refusal::MissingInput("PF30209 jawline_indices"));
    }
    if inputs.mask.is_empty() {
        return Err(Refusal::MissingInput("PF30209 mask"));
    }
    if inputs.model_output.is_empty() {
        return Err(Refusal::MissingInput("PF30209 model_output"));
    }
    if inputs.face_194_packet.iter().any(|value| !value.is_finite())
        || inputs.model_output.iter().any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("PF30209 finite input"));
    }
    Err(Refusal::MissingInput(PF30209_END_STATIC))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contracts_cover_each_row_once() {
        let mut pfs: Vec<u32> = CONTRACTS.iter().map(|row| row.pf).collect();
        pfs.sort_unstable();
        assert_eq!(pfs, [101, 403, 30188, 30189, 30209, 90001, 90003]);
        for row in CONTRACTS {
            assert_eq!(contract(row.pf), Some(row));
        }
    }

    #[test]
    fn double_chin_weights_preserve_recovered_product() {
        assert_eq!(
            double_chin_weights(0.5, 0.4, 0.2),
            Ok(DoubleChinWeights {
                geometry: 0.2,
                shadow: 0.1,
            })
        );
        assert_eq!(
            double_chin_weights(f32::NAN, 0.4, 0.2),
            Err(Refusal::ValueOutOfRange("double chin native values"))
        );
    }

    #[test]
    fn pf403_uses_the_recovered_mask_kernel() {
        let mask = Image::solid(4, 4, [32, 64, 128, 255]);
        let out = enhance_breast_mask(&mask, 0.0).unwrap();
        assert_eq!((out.width, out.height), (4, 4));
        assert_eq!(out.pixels.len(), mask.pixels.len());
    }

    #[test]
    fn pf90003_does_not_use_the_explicit_target_shortcut() {
        let source = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let mask = vec![255; 4];
        assert_eq!(
            body_skin_uniform_contract(&source, 2, 2, &mask, &mask, 50.0, 0.5),
            Err(Refusal::MissingInput(PF90003_PIXEL_LAW_OPEN))
        );
        assert_eq!(
            body_skin_uniform_contract(&source, 2, 2, &mask, &mask, 0.0, f32::NAN),
            Ok(source)
        );
    }

    #[test]
    fn live_and_end_static_rows_refuse_after_shape_checks() {
        assert!(matches!(
            silkworm(Pf101SameJobInputs {
                eye_mask: &[1],
                eye_mask_type: 0,
                face_blur_factor: 1.0,
                lut: &[1],
            }),
            Err(Refusal::MissingInput(PF101_LIVE_ONLY))
        ));
        assert!(matches!(
            jawline(Pf30209SameJobInputs {
                face_194_packet: &[0.0],
                jawline_indices: &[0],
                mask: &[1],
                model_output: &[0.0],
            }),
            Err(Refusal::MissingInput(PF30209_END_STATIC))
        ));
    }
}
