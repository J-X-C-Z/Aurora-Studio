//! PF90116 eye-bag composite on NVIDIA.
//!
//! The CPU `kernels::eye_bag` final stage is the reference for
//! `eye_bag_compose_kernel`:
//! ```text
//! q = saturate_u8(round(255 * response)) // nearest, ties to even (cvRound)
//! m = q * (1/255) * degree
//! out[c] = clamp(trunc(color[c]*m + src[c]*(1-m)), 0, 255)
//! m == 0 or NaN -> copy; alpha follows src
//! ```
//! The native color target is float RGB, so it stays float until composition.

use crate::cuda::{CudaError, Dispatch};
use crate::kernels::eye_bag::{self, PF90116_ACTIVE_GATE};
use crate::{Image, Refusal};

/// Recovered kernel entry point.
pub const KERNEL: &str = "eye_bag_compose_kernel";
const BLOCK: u32 = 16;

const PTX: &str = include_str!("../native/cuda/eye_bag.ptx");

/// 32-byte kernel argument block, matching `native/cuda/eye_bag.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub degree: f32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u32,
    pub reserved3: u32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 32);

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) }
}

fn check(source: &Image, color: &Image, response: &[f32], degree: f32) -> Result<usize, Refusal> {
    source.validate()?;
    color.validate()?;
    if (source.width, source.height) != (color.width, color.height) {
        return Err(Refusal::ShapeMismatch("color"));
    }
    if response.len() != source.pixel_count() {
        return Err(Refusal::ShapeMismatch("response"));
    }
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("degree"));
    }
    Ok(source.pixel_count())
}

/// Image convenience form of the production CPU reference.
pub fn reference(
    source: &Image,
    color: &Image,
    response: &[f32],
    degree: f32,
) -> Result<Image, Refusal> {
    check(source, color, response, degree)?;
    eye_bag::compose_image_pf90116(source, color, response, degree, eye_bag::BlendMode::Vendor)
}

#[cfg(windows)]
pub struct EyeBag {
    gpu: crate::cuda::Nvidia,
}

#[cfg(windows)]
impl EyeBag {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self { gpu: crate::cuda::Nvidia::open_ptx(PTX)? })
    }

    /// GPU twin of [`reference`]. Uploads source, color and response once.
    pub fn compose(
        &self,
        source: &Image,
        color: &Image,
        response: &[f32],
        degree: f32,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        check(source, color, response, degree).map_err(CudaError::Refused)?;
        let field: Vec<f32> = color.pixels.chunks_exact(4)
            .flat_map(|pixel| pixel[..3].iter().map(|&value| value as f32))
            .collect();
        self.compose_field(source, &field, response, degree)
    }

    /// Retains the model's float color target through the final truncation.
    pub fn compose_field(
        &self,
        source: &Image,
        color3: &[f32],
        response: &[f32],
        degree: f32,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        source.validate().map_err(CudaError::Refused)?;
        eye_bag::check_pf90116(source.pixel_count(), color3, response, degree)
            .map_err(CudaError::Refused)?;
        if degree < PF90116_ACTIVE_GATE {
            return Ok((source.clone(), Vec::new()));
        }
        let uniforms = Uniforms {
            width: source.width,
            height: source.height,
            stride: source.width * 4,
            degree,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
        };
        let grid = (
            source.width.div_ceil(BLOCK).max(1),
            source.height.div_ceil(BLOCK).max(1),
        );
        let (pixels, dispatch) = self.gpu.run_u8(
            KERNEL,
            grid,
            &[&source.pixels, f32_bytes(color3), f32_bytes(response)],
            as_bytes(&uniforms),
        )?;
        let image = Image::new(source.width, source.height, pixels)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, vec![dispatch]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_block_matches_the_c_header() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 32);
        assert_eq!(std::mem::align_of::<Uniforms>(), 4);
    }

    #[test]
    fn reference_skips_zero_and_nan_and_truncates() {
        let source = Image::solid(1, 1, [10, 200, 255, 42]);
        let color = Image::solid(1, 1, [250, 0, 0, 7]);
        assert_eq!(reference(&source, &color, &[0.0], 1.0).unwrap(), source);
        assert_eq!(reference(&source, &color, &[f32::NAN], 1.0).unwrap(), source);
        assert_eq!(reference(&source, &color, &[1.0], 0.0).unwrap(), source);
        // q=255 -> m=1 -> pure color, and alpha stays the source's.
        assert_eq!(reference(&source, &color, &[1.0], 1.0).unwrap().pixel(0, 0), [250, 0, 0, 42]);
        // q=128 -> m=128/255; truncation, not rounding.
        let half = reference(&source, &color, &[128.0 / 255.0], 1.0).unwrap();
        let m = 128.0f32 / 255.0;
        let expected = ((250.0f32 * m + 10.0 * (1.0 - m)) as i32) as u8;
        assert_eq!(half.pixel(0, 0)[0], expected);
    }
}
