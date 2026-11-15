//! PF90080 final-consumer CUDA candidate.
//!
//! The producer must supply a full-frame packed RGB effect plane and a
//! one-byte mask already aligned to `source`. This module does not choose model
//! lanes, polarity, ROI crop, or upscale. An active frame with either producer
//! missing is a typed refusal; an inactive native degree is an identity.

use crate::kernels::skin_color::oil_shine_mask_blend;
use crate::{Image, Refusal};

#[cfg(windows)]
use crate::cuda::{CudaError, Dispatch, Nvidia};

pub const KERNEL: &str = "oil_shine_mask_blend_kernel";
pub const PF: u32 = 90080;
pub const ACTIVE_GATE: f32 = 0.001;
const BLOCK: u32 = 16;
const PTX: &str = include_str!("../native/cuda/oil_shine_blend.ptx");

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: i32,
    pub height: i32,
    pub degree: f32,
    pub reserved: i32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 16);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate {
    Identity,
    Active,
}

pub struct Frame<'a> {
    pub source: &'a Image,
    pub effect_rgb: Option<&'a [u8]>,
    pub mask: Option<&'a [u8]>,
    pub degree: f32,
}

/// Validates the consumer contract and applies the recovered native `> 0.001`
/// gate. Active frames require both same-job producer planes.
pub fn gate(frame: &Frame<'_>) -> Result<Gate, Refusal> {
    frame
        .source
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("source"))?;
    if !frame.degree.is_finite() || !(0.0..=1.0).contains(&frame.degree) {
        return Err(Refusal::ValueOutOfRange("degree"));
    }
    if frame.degree <= ACTIVE_GATE {
        return Ok(Gate::Identity);
    }

    let effect = frame
        .effect_rgb
        .ok_or(Refusal::MissingInput("pf90080_effect_rgb_producer"))?;
    let mask = frame
        .mask
        .ok_or(Refusal::MissingInput("pf90080_mask_producer"))?;
    let pixels = frame.source.pixel_count();
    let rgb_bytes = pixels
        .checked_mul(3)
        .ok_or(Refusal::ShapeMismatch("effect_rgb"))?;
    if effect.len() != rgb_bytes {
        return Err(Refusal::ShapeMismatch("effect_rgb"));
    }
    if mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("mask"));
    }
    if i32::try_from(frame.source.width).is_err() || i32::try_from(frame.source.height).is_err() {
        return Err(Refusal::ShapeMismatch("source"));
    }
    Ok(Gate::Active)
}

/// Explicit CPU reference for a validated producer frame.
pub fn reference(frame: &Frame<'_>) -> Result<Image, Refusal> {
    if gate(frame)? == Gate::Identity {
        return Ok(frame.source.clone());
    }
    oil_shine_mask_blend(
        frame.source,
        frame.effect_rgb.expect("active frame validated"),
        frame.mask.expect("active frame validated"),
        frame.degree,
    )
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

    /// Runs one full-frame candidate consumer. `None` dispatch means the native
    /// degree gate returned the source without touching the GPU.
    pub fn apply(&self, frame: &Frame<'_>) -> Result<(Image, Option<Dispatch>), CudaError> {
        if gate(frame)? == Gate::Identity {
            return Ok((frame.source.clone(), None));
        }
        let effect = frame.effect_rgb.expect("active frame validated");
        let mask = frame.mask.expect("active frame validated");
        let uniforms = Uniforms {
            width: frame.source.width as i32,
            height: frame.source.height as i32,
            degree: frame.degree,
            reserved: 0,
        };
        let grid = (
            frame.source.width.div_ceil(BLOCK).max(1),
            frame.source.height.div_ceil(BLOCK).max(1),
        );
        let (pixels, dispatch) = self.gpu.run_u8(
            KERNEL,
            grid,
            &[&frame.source.pixels, effect, mask],
            unsafe {
                std::slice::from_raw_parts(
                    &uniforms as *const Uniforms as *const u8,
                    std::mem::size_of::<Uniforms>(),
                )
            },
        )?;
        let image = Image::new(frame.source.width, frame.source.height, pixels)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, Some(dispatch)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inactive_gate_needs_no_producer() {
        let source = Image::solid(1, 1, [1, 2, 3, 4]);
        let frame = Frame {
            source: &source,
            effect_rgb: None,
            mask: None,
            degree: ACTIVE_GATE,
        };
        assert_eq!(gate(&frame), Ok(Gate::Identity));
        assert_eq!(reference(&frame).unwrap(), source);
    }

    #[test]
    fn active_gate_names_each_missing_producer() {
        let source = Image::solid(1, 1, [1, 2, 3, 4]);
        let effect = [9, 9, 9];
        let mask = [255];
        assert_eq!(
            gate(&Frame {
                source: &source,
                effect_rgb: None,
                mask: Some(&mask),
                degree: 0.5,
            }),
            Err(Refusal::MissingInput("pf90080_effect_rgb_producer"))
        );
        assert_eq!(
            gate(&Frame {
                source: &source,
                effect_rgb: Some(&effect),
                mask: None,
                degree: 0.5,
            }),
            Err(Refusal::MissingInput("pf90080_mask_producer"))
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
        assert_eq!(params.len(), 5);
        assert!(params[4].contains(".b8") && params[4].contains("[16]"));
    }

    #[test]
    fn reusing_texture_blend_would_change_half_ties() {
        use crate::kernels::texture_mask_blend::{self, NormalBlend};

        let original = [0, 2, 10, 77];
        let effect = [1, 3, 10, 0];
        assert_eq!(
            texture_mask_blend::pixel(
                original,
                effect,
                255,
                NormalBlend {
                    degree: 0.5,
                    reverse_alpha: false,
                }
            )
            .unwrap(),
            [1, 3, 10, 77]
        );

        let source = Image::solid(1, 1, original);
        assert_eq!(
            reference(&Frame {
                source: &source,
                effect_rgb: Some(&[1, 3, 10]),
                mask: Some(&[255]),
                degree: 0.5,
            })
            .unwrap()
            .pixels,
            [0, 2, 10, 77]
        );
    }
}
