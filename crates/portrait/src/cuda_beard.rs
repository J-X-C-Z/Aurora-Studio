//! PF21026 stubble removal on NVIDIA.
//!
//! Same contract as `kernels::beard::apply`: the host applies the strength gate
//! and the `0.75` small-frame rule, the kernel performs `w = min(mask_a,
//! mask_b)`, the byte lerp, nearest-even rounding and alpha copy.

use crate::cuda::CudaError;
#[cfg(windows)]
use crate::cuda::Nvidia;
pub use crate::kernels::beard::scaled_strength;
use crate::{Image, Refusal};

pub const KERNEL: &str = "beard_blend_kernel";
const BLOCK: u32 = 16;
const PTX: &str = include_str!("../native/cuda/beard.ptx");

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub strength: f32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u32,
    pub reserved3: u32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 32);

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn check(
    source: &Image,
    repair: &Image,
    mask_a: &[u8],
    mask_b: &[u8],
) -> Result<usize, Refusal> {
    source.validate()?;
    repair.validate().map_err(|_| Refusal::ShapeMismatch("repair"))?;
    if (source.width, source.height) != (repair.width, repair.height) {
        return Err(Refusal::ShapeMismatch("repair"));
    }
    if mask_a.len() != source.pixel_count() || mask_b.len() != source.pixel_count() {
        return Err(Refusal::ShapeMismatch("mask"));
    }
    Ok(source.pixel_count())
}

#[cfg(windows)]
pub struct Beard {
    gpu: Nvidia,
}

#[cfg(windows)]
impl Beard {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self { gpu: Nvidia::open_ptx(PTX)? })
    }

    pub fn apply(
        &self,
        source: &Image,
        repair: &Image,
        mask_a: &[u8],
        mask_b: &[u8],
        strength: f32,
    ) -> Result<(Image, Vec<crate::cuda::Dispatch>), CudaError> {
        check(source, repair, mask_a, mask_b).map_err(CudaError::Refused)?;
        let Some(strength) = scaled_strength(source.width, source.height, strength)
            .map_err(CudaError::Refused)?
        else {
            return Ok((source.clone(), Vec::new()));
        };
        let uniforms = Uniforms {
            width: source.width,
            height: source.height,
            stride: source.width * 4,
            strength,
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
            &[&source.pixels, &repair.pixels, mask_a, mask_b],
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
    fn host_strength_matches_the_cpu_rule() {
        assert_eq!(scaled_strength(400, 100, 0.005).unwrap(), None);
        assert_eq!(scaled_strength(400, 100, 1.0).unwrap(), Some(1.0));
        assert_eq!(scaled_strength(200, 100, 1.0).unwrap(), Some(0.75));
        assert_eq!(
            scaled_strength(400, 100, -0.1),
            Err(Refusal::ValueOutOfRange("strength"))
        );
    }
}
