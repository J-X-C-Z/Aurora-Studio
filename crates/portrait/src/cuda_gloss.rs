//! PF8068 SKIN_GLOSS on NVIDIA.
//!
//! Same law as `kernels::skin_models::gloss_apply`: three passes over the image
//! (positive1, positive2 with `positive = max(2u-1, 0)`, negative with
//! `negative = max(1-2u, 0)`), each blending towards the caller-supplied 256-entry
//! table by `weight = (mask/255) * (extra_mask/255)`. RGB is rewritten, alpha is
//! copied. A pass whose degree is exactly zero never launches, which is the
//! CPU's strict `if degree == 0.0 { return; }` gate.
//!
//! The driver, context and the 222-only PTX retry live in [`crate::cuda`]; this
//! module validates inputs, builds the ABI and calls the shared `run_u8` path.

use crate::cuda::{CudaError, Dispatch};
use crate::{Image, Refusal};

/// Recovered kernel entry point.
pub const KERNEL: &str = "gloss_pass_kernel";
/// Kernel launch shape, shared with the P15 stages.
const BLOCK: u32 = 16;

const PTX: &str = include_str!("../native/cuda/gloss_lut.ptx");

/// 32-byte kernel argument block, matching `native/cuda/gloss_lut.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub mask_row_bytes: u32,
    pub pass: u32,
    pub degree: f32,
    pub reserved0: u32,
    pub reserved1: u32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 32);

impl Uniforms {
    pub fn new(width: u32, height: u32, pass: u32, degree: f32) -> Self {
        Self {
            width,
            height,
            stride: width * 4,
            mask_row_bytes: width,
            pass,
            degree,
            reserved0: 0,
            reserved1: 0,
        }
    }
}

fn as_bytes(value: &Uniforms) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            value as *const Uniforms as *const u8,
            std::mem::size_of::<Uniforms>(),
        )
    }
}

/// Refusals shared with the CPU lane, checked before any driver call.
fn check(
    source: &Image,
    masks_r: [&[u8]; 3],
    extra_masks_r: [&[u8]; 3],
    native_percent: f32,
) -> Result<(), Refusal> {
    source.validate().map_err(|_| Refusal::ShapeMismatch("source"))?;
    if !native_percent.is_finite() || !(0.0..=100.0).contains(&native_percent) {
        return Err(Refusal::ValueOutOfRange("native_percent"));
    }
    let pixels = source.pixel_count();
    for mask in masks_r.iter().chain(extra_masks_r.iter()) {
        if mask.len() != pixels {
            return Err(Refusal::ShapeMismatch("mask_r"));
        }
    }
    Ok(())
}

/// `(pass index, degree)` in the CPU's fixed order.
fn passes(native_percent: f32) -> [(u32, f32); 3] {
    let normalized = native_percent / 100.0;
    let positive = (2.0 * normalized - 1.0).max(0.0);
    let negative = (1.0 - 2.0 * normalized).max(0.0);
    [(0, positive), (1, positive), (2, negative)]
}

#[cfg(windows)]
pub struct Gloss(crate::cuda::Nvidia);

#[cfg(windows)]
impl Gloss {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self(crate::cuda::Nvidia::open_ptx(PTX)?))
    }

    /// Compatibility entry: active work refuses missing LUTs; neutral returns
    /// identity with no dispatch. Use apply_with_luts for caller-owned curves.
    pub fn apply(
        &self,
        source: &Image,
        masks_r: [&[u8]; 3],
        extra_masks_r: [&[u8]; 3],
        native_percent: f32,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        self.apply_with_luts(source, masks_r, extra_masks_r, native_percent, None)
    }

    /// Explicit caller-owned tables; None refuses active work without dispatch.
    pub fn apply_with_luts(
        &self,
        source: &Image,
        masks_r: [&[u8]; 3],
        extra_masks_r: [&[u8]; 3],
        native_percent: f32,
        luts: Option<[&[u8; 256]; 3]>,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        check(source, masks_r, extra_masks_r, native_percent)?;
        if passes(native_percent).iter().all(|&(_, degree)| degree == 0.0) {
            return Ok((source.clone(), Vec::new()));
        }
        let luts = luts.ok_or(Refusal::MissingInput("gloss_luts"))?;
        let grid = (
            source.width.div_ceil(BLOCK).max(1),
            source.height.div_ceil(BLOCK).max(1),
        );
        let mut current = source.pixels.clone();
        let mut dispatches = Vec::new();
        for (pass, degree) in passes(native_percent) {
            if degree == 0.0 {
                continue;
            }
            let uniforms = Uniforms::new(source.width, source.height, pass, degree);
            let (next, dispatch) = self.0.run_u8(
                KERNEL,
                grid,
                &[&current, masks_r[pass as usize], extra_masks_r[pass as usize], luts[pass as usize]],
                as_bytes(&uniforms),
            )?;
            current = next;
            dispatches.push(dispatch);
        }
        let image = Image::new(source.width, source.height, current)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, dispatches))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_block_matches_the_c_header() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 32);
        assert_eq!(std::mem::align_of::<Uniforms>(), 4);
        let uniforms = Uniforms::new(17, 13, 2, 0.25);
        assert_eq!((uniforms.stride, uniforms.mask_row_bytes), (17 * 4, 17));
        assert_eq!(as_bytes(&uniforms).len(), 32);
    }

    #[test]
    fn pass_degrees_follow_the_captured_config_percent() {
        assert_eq!(passes(0.0), [(0, 0.0), (1, 0.0), (2, 1.0)]);
        assert_eq!(passes(50.0), [(0, 0.0), (1, 0.0), (2, 0.0)]);
        assert_eq!(passes(75.0), [(0, 0.5), (1, 0.5), (2, 0.0)]);
        assert_eq!(passes(100.0), [(0, 1.0), (1, 1.0), (2, 0.0)]);
    }
}
