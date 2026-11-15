//! PF30187 hand-neutral-gray final consumer on NVIDIA.
//!
//! Same pixel law as `kernels::hair_body::hand_neutral_gray`:
//! `w = hand_mask/255 * model_alpha/255 * ui/100`,
//! `rgb = round(src*(1-w) + neutral*w)`, alpha copied. Neutral is either an
//! explicit same-job BGRA plane or an explicit normalized BGR model residual.
//! An active frame with a missing producer is a typed refusal, never a
//! synthesized mask, alpha or target.

use crate::kernels::hair_body::{hand_neutral_gray, NeutralTarget};
use crate::{Image, Refusal};

#[cfg(windows)]
use crate::cuda::{CudaError, Dispatch, Nvidia};

pub const KERNEL: &str = "hand_neutral_gray_kernel";
pub const PF: u32 = 30187;
const BLOCK: u32 = 16;
const TARGET_BGRA: i32 = 0;
const TARGET_DIFF: i32 = 1;
const PTX: &str = include_str!("../native/cuda/hand_neutral_gray.ptx");

/// 16-byte argument block, field-for-field with the C header.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: i32,
    pub height: i32,
    pub degree: f32,
    pub target_kind: i32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 16);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    Identity,
    Active,
}

pub struct Frame<'a> {
    pub source: &'a Image,
    pub ui: f32,
    pub hand_mask: Option<&'a [u8]>,
    pub model_alpha: Option<&'a [u8]>,
    pub target: Option<NeutralTarget<'a>>,
}

fn validate_target(source: &Image, target: &NeutralTarget<'_>) -> Result<(), Refusal> {
    match target {
        NeutralTarget::Bgra(neutral) => {
            neutral
                .validate()
                .map_err(|_| Refusal::ShapeMismatch("PF30187_MODEL_NEUTRAL"))?;
            if neutral.width != source.width || neutral.height != source.height {
                return Err(Refusal::ShapeMismatch("PF30187_MODEL_NEUTRAL"));
            }
        }
        NeutralTarget::Diff(diff) => {
            let expected = source
                .pixel_count()
                .checked_mul(3)
                .ok_or(Refusal::ShapeMismatch("PF30187_MODEL_DIFF"))?;
            if diff.len() != expected {
                return Err(Refusal::ShapeMismatch("PF30187_MODEL_DIFF"));
            }
            if diff.iter().any(|value| !value.is_finite()) {
                return Err(Refusal::ValueOutOfRange("PF30187_MODEL_DIFF"));
            }
        }
    }
    Ok(())
}

/// Validates the producer contract before any CPU or GPU work.
pub fn gate(frame: &Frame<'_>) -> Result<Gate, Refusal> {
    frame
        .source
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("source"))?;
    if !frame.ui.is_finite() || !(0.0..=100.0).contains(&frame.ui) {
        return Err(Refusal::ValueOutOfRange("PF30187_UI"));
    }
    if frame.ui == 0.0 {
        return Ok(Gate::Identity);
    }

    let hand_mask = frame
        .hand_mask
        .ok_or(Refusal::MissingInput("pf30187_hand_mask_producer"))?;
    let model_alpha = frame
        .model_alpha
        .ok_or(Refusal::MissingInput("pf30187_model_alpha_producer"))?;
    let target = frame
        .target
        .as_ref()
        .ok_or(Refusal::MissingInput("pf30187_model_neutral_producer"))?;

    let pixels = frame.source.pixel_count();
    if hand_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("PF30187_HAND_MASK"));
    }
    if model_alpha.len() != pixels {
        return Err(Refusal::ShapeMismatch("PF30187_MODEL_ALPHA"));
    }
    validate_target(frame.source, target)?;
    if i32::try_from(frame.source.width).is_err() || i32::try_from(frame.source.height).is_err() {
        return Err(Refusal::ShapeMismatch("source"));
    }
    Ok(Gate::Active)
}

/// CPU reference for the exact validated frame.
pub fn reference(frame: &Frame<'_>) -> Result<Image, Refusal> {
    if gate(frame)? == Gate::Identity {
        return Ok(frame.source.clone());
    }
    let target = match frame.target.as_ref().expect("active frame validated") {
        NeutralTarget::Bgra(image) => NeutralTarget::Bgra(image),
        NeutralTarget::Diff(diff) => NeutralTarget::Diff(diff),
    };
    hand_neutral_gray(
        frame.source,
        frame.ui,
        frame.hand_mask.expect("active frame validated"),
        frame.model_alpha.expect("active frame validated"),
        target,
    )
}

fn as_bytes<T>(value: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            value.as_ptr() as *const u8,
            std::mem::size_of_val(value),
        )
    }
}

#[cfg(windows)]
pub struct Blend {
    gpu: Nvidia,
}

#[cfg(windows)]
impl Blend {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self {
            gpu: Nvidia::open_ptx(PTX)?,
        })
    }

    /// Runs the final consumer. `None` dispatch means the UI-zero identity
    /// gate returned the source without touching the GPU.
    pub fn apply(&self, frame: &Frame<'_>) -> Result<(Image, Option<Dispatch>), CudaError> {
        if gate(frame)? == Gate::Identity {
            return Ok((frame.source.clone(), None));
        }

        let hand_mask = frame.hand_mask.expect("active frame validated");
        let model_alpha = frame.model_alpha.expect("active frame validated");
        let target = frame.target.as_ref().expect("active frame validated");
        let uniforms = Uniforms {
            width: frame.source.width as i32,
            height: frame.source.height as i32,
            degree: frame.ui / 100.0,
            target_kind: match target {
                NeutralTarget::Bgra(_) => TARGET_BGRA,
                NeutralTarget::Diff(_) => TARGET_DIFF,
            },
        };
        let grid = (
            frame.source.width.div_ceil(BLOCK).max(1),
            frame.source.height.div_ceil(BLOCK).max(1),
        );
        let dummy = [0u8; 8];
        let (pixels, dispatch) = match target {
            NeutralTarget::Bgra(neutral) => self.gpu.run_u8(
                KERNEL,
                grid,
                &[
                    &frame.source.pixels,
                    hand_mask,
                    model_alpha,
                    &neutral.pixels,
                    &dummy,
                ],
                as_bytes(std::slice::from_ref(&uniforms)),
            )?,
            NeutralTarget::Diff(diff) => self.gpu.run_u8(
                KERNEL,
                grid,
                &[
                    &frame.source.pixels,
                    hand_mask,
                    model_alpha,
                    &dummy,
                    as_bytes(diff),
                ],
                as_bytes(std::slice::from_ref(&uniforms)),
            )?,
        };
        let image = Image::new(frame.source.width, frame.source.height, pixels)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, Some(dispatch)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_ui_is_identity_without_producers() {
        let source = Image::solid(1, 1, [1, 2, 3, 4]);
        let frame = Frame {
            source: &source,
            ui: 0.0,
            hand_mask: None,
            model_alpha: None,
            target: None,
        };
        assert_eq!(gate(&frame), Ok(Gate::Identity));
        assert_eq!(reference(&frame).unwrap(), source);
    }

    #[test]
    fn active_frame_names_each_missing_producer() {
        let source = Image::solid(1, 1, [1, 2, 3, 4]);
        let neutral = Image::solid(1, 1, [9, 9, 9, 4]);
        let hand = [255];
        let alpha = [128];
        assert_eq!(
            gate(&Frame {
                source: &source,
                ui: 50.0,
                hand_mask: None,
                model_alpha: Some(&alpha),
                target: Some(NeutralTarget::Bgra(&neutral)),
            }),
            Err(Refusal::MissingInput("pf30187_hand_mask_producer"))
        );
        assert_eq!(
            gate(&Frame {
                source: &source,
                ui: 50.0,
                hand_mask: Some(&hand),
                model_alpha: None,
                target: Some(NeutralTarget::Bgra(&neutral)),
            }),
            Err(Refusal::MissingInput("pf30187_model_alpha_producer"))
        );
        assert_eq!(
            gate(&Frame {
                source: &source,
                ui: 50.0,
                hand_mask: Some(&hand),
                model_alpha: Some(&alpha),
                target: None,
            }),
            Err(Refusal::MissingInput("pf30187_model_neutral_producer"))
        );
    }

    #[test]
    fn active_frame_refuses_bad_geometry_and_values() {
        let source = Image::solid(2, 1, [1, 2, 3, 4]);
        let neutral = Image::solid(2, 1, [9, 9, 9, 4]);
        let hand = [255, 255];
        let alpha = [255, 255];
        assert_eq!(
            gate(&Frame {
                source: &source,
                ui: 50.0,
                hand_mask: Some(&hand[..1]),
                model_alpha: Some(&alpha),
                target: Some(NeutralTarget::Bgra(&neutral)),
            }),
            Err(Refusal::ShapeMismatch("PF30187_HAND_MASK"))
        );
        assert_eq!(
            gate(&Frame {
                source: &source,
                ui: 50.0,
                hand_mask: Some(&hand),
                model_alpha: Some(&alpha),
                target: Some(NeutralTarget::Diff(&[
                    0.0, 0.0, 0.0, 0.0, 0.0, f32::NAN,
                ])),
            }),
            Err(Refusal::ValueOutOfRange("PF30187_MODEL_DIFF"))
        );
        assert_eq!(
            gate(&Frame {
                source: &source,
                ui: 101.0,
                hand_mask: Some(&hand),
                model_alpha: Some(&alpha),
                target: Some(NeutralTarget::Bgra(&neutral)),
            }),
            Err(Refusal::ValueOutOfRange("PF30187_UI"))
        );
    }

    #[test]
    fn kernel_block_matches_the_c_header() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 16);
        assert_eq!(std::mem::align_of::<Uniforms>(), 4);
        let signature = PTX
            .split(&format!(".entry {KERNEL}("))
            .nth(1)
            .expect("PTX kernel entry")
            .split(')')
            .next()
            .unwrap();
        let params: Vec<_> = signature
            .lines()
            .filter(|line| line.contains(".param"))
            .collect();
        assert_eq!(params.len(), 7);
        assert!(params[6].contains(".b8") && params[6].contains("[16]"));
    }
}
