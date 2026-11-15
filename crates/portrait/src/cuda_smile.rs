//! PF100007 SmileV1 and PF8033 dense smile warps on NVIDIA.
//!
//! Dedicated kernel, not the type-143 UV path: the CPU consumer works in pixel
//! space with a byte-space bilinear tap, truncating store and copied alpha, and
//! this kernel reproduces that order exactly. No UV normalisation is involved,
//! so there is no rounding gap between the two.
//!
//! Driver and the 222-only PTX retry live in [`crate::cuda`]; the call goes
//! through the shared `run_u8` path.

use crate::cuda::{CudaError, Dispatch};
use crate::kernels::smile_warp::Rect;
use crate::Refusal;

/// Recovered kernel entry point.
pub const KERNEL: &str = "smile_warp_kernel";
/// Mask-only entry for PF8033: same body without the constant mouth plane.
pub const MASK_KERNEL: &str = "smile_warp_mask_kernel";
const BLOCK: u32 = 16;

const PTX: &str = include_str!("../native/cuda/smile_warp.ptx");

/// 48-byte kernel argument block, matching `native/cuda/smile_warp.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub degree: f32,
    pub roi_x: u32,
    pub roi_y: u32,
    pub roi_w: u32,
    pub roi_h: u32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u32,
    pub reserved3: u32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 48);

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) }
}

/// Entry checks shared with `pf100007_apply`, plus the overflow guards the CPU
/// lane is adding: zero size, size overflow, non-finite displacement, ROI wrap.
fn check(
    source_bgra: &[u8],
    width: u32,
    height: u32,
    degree: f32,
    lip_mask: &[u8],
    mouth_mask: &[u8],
    model_dx: &[f32],
    model_dy: &[f32],
    roi: Rect,
) -> Result<usize, Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("pf100007 bgra"));
    }
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("pf100007 bgra"))?;
    let bytes = pixels.checked_mul(4).ok_or(Refusal::ShapeMismatch("pf100007 bgra"))?;
    if source_bgra.len() != bytes {
        return Err(Refusal::ShapeMismatch("pf100007 bgra"));
    }
    for (len, name) in [
        (lip_mask.len(), "pf100007 lip_mask"),
        (mouth_mask.len(), "pf100007 mouth_mask"),
        (model_dx.len(), "pf100007 model_dx"),
        (model_dy.len(), "pf100007 model_dy"),
    ] {
        if len != pixels {
            return Err(Refusal::MissingInput(name));
        }
    }
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("pf100007 degree"));
    }
    if model_dx.iter().chain(model_dy).any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("pf100007 displacement"));
    }
    // Subtraction form: no u32 addition to overflow.
    if roi.x > width
        || roi.y > height
        || roi.width > width - roi.x
        || roi.height > height - roi.y
    {
        return Err(Refusal::ValueOutOfRange("pf100007 roi"));
    }
    Ok(pixels)
}

#[cfg(windows)]
pub struct Smile {
    gpu: crate::cuda::Nvidia,
}

#[cfg(windows)]
impl Smile {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self { gpu: crate::cuda::Nvidia::open_ptx(PTX)? })
    }

    /// PF8033 dense warp with an explicit displacement field and mask.
    ///
    /// Uses the mask-only entry: `support = mask/255` is bit-identical to the
    /// shared kernel's `(mask/255) * (255/255)`, so no temporary 255 plane is
    /// built, allocated or uploaded. Sampling, truncation and alpha are the
    /// same code as the PF100007 entry.
    pub fn apply_pf8033(
        &self,
        source_bgra: &[u8],
        width: u32,
        height: u32,
        dx: &[f32],
        dy: &[f32],
        mask: &[u8],
        degree01: f32,
    ) -> Result<(Vec<u8>, Vec<Dispatch>), CudaError> {
        if crate::Image::byte_len(width, height) != Some(source_bgra.len()) {
            return Err(Refusal::ShapeMismatch("pf8033 source_bgra").into());
        }
        let pixels = source_bgra.len() / 4;
        if dx.len() != pixels || dy.len() != pixels || mask.len() != pixels {
            return Err(Refusal::MissingInput("pf8033 dx/dy/mask").into());
        }
        if !degree01.is_finite() || !(0.0..=1.0).contains(&degree01) {
            return Err(Refusal::ValueOutOfRange("pf8033 degree01").into());
        }
        if dx.iter().chain(dy).any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("pf8033 displacement").into());
        }
        if degree01 <= 0.0 {
            return Ok((source_bgra.to_vec(), Vec::new()));
        }
        let uniforms = Uniforms {
            width,
            height,
            stride: width * 4,
            degree: degree01,
            roi_x: 0,
            roi_y: 0,
            roi_w: width,
            roi_h: height,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
        };
        let grid = (width.div_ceil(BLOCK).max(1), height.div_ceil(BLOCK).max(1));
        let (out, dispatch) = self.gpu.run_u8(
            MASK_KERNEL,
            grid,
            &[source_bgra, mask, f32_bytes(dx), f32_bytes(dy)],
            as_bytes(&uniforms),
        )?;
        Ok((out, vec![dispatch]))
    }

    /// GPU twin of `smile_warp::pf100007_apply`. Returns the warped BGRA bytes.
    pub fn apply(
        &self,
        source_bgra: &[u8],
        width: u32,
        height: u32,
        degree: f32,
        lip_mask: &[u8],
        mouth_mask: &[u8],
        model_dx: &[f32],
        model_dy: &[f32],
        roi: Rect,
    ) -> Result<(Vec<u8>, Vec<Dispatch>), CudaError> {
        check(source_bgra, width, height, degree, lip_mask, mouth_mask, model_dx, model_dy, roi)?;
        if roi.width == 0 || roi.height == 0 || degree <= 0.0 {
            return Ok((source_bgra.to_vec(), Vec::new()));
        }
        let uniforms = Uniforms {
            width,
            height,
            stride: width * 4,
            degree,
            roi_x: roi.x,
            roi_y: roi.y,
            roi_w: roi.width,
            roi_h: roi.height,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
        };
        // Full-image dispatch: the kernel copies every pixel and only warps
        // inside the ROI, so out-of-ROI bytes match the source.
        let grid = (width.div_ceil(BLOCK).max(1), height.div_ceil(BLOCK).max(1));
        let (out, dispatch) = self.gpu.run_u8(
            KERNEL,
            grid,
            &[source_bgra, lip_mask, mouth_mask, f32_bytes(model_dx), f32_bytes(model_dy)],
            as_bytes(&uniforms),
        )?;
        Ok((out, vec![dispatch]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_block_matches_the_c_header() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 48);
        assert_eq!(std::mem::align_of::<Uniforms>(), 4);
    }

    #[test]
    fn entry_checks_cover_size_model_and_roi() {
        let source = vec![0u8; 4 * 4 * 4];
        let plane = vec![255u8; 16];
        let dx = vec![0.0f32; 16];
        let dy = vec![0.0f32; 16];
        let roi = Rect { x: 0, y: 0, width: 4, height: 4 };
        assert_eq!(check(&source, 4, 4, 1.0, &plane, &plane, &dx, &dy, roi), Ok(16));
        assert_eq!(
            check(&source, 0, 4, 1.0, &plane, &plane, &dx, &dy, roi),
            Err(Refusal::ShapeMismatch("pf100007 bgra"))
        );
        let mut bad = dx.clone();
        bad[3] = f32::NAN;
        assert_eq!(
            check(&source, 4, 4, 1.0, &plane, &plane, &bad, &dy, roi),
            Err(Refusal::ValueOutOfRange("pf100007 displacement"))
        );
        assert_eq!(
            check(&source, 4, 4, 1.0, &plane, &plane, &dx, &dy, Rect { x: u32::MAX, y: 0, width: 4, height: 4 }),
            Err(Refusal::ValueOutOfRange("pf100007 roi"))
        );
    }
}
