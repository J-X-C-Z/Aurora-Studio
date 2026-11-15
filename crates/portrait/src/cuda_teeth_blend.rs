//! PF30026 final teeth blend on NVIDIA.
//!
//! Same law as `kernels::teeth_blend::apply`: `m = mask*0x3B807BBC`,
//! `out = trunc(m*effect + original*(1-m))`, the `mask == 255` fast path, and
//! alpha always copied from the original. The uniform block is passed by value.

use crate::cuda::{CudaError, Dispatch};
use crate::{Image, Refusal};

/// Recovered kernel entry point.
pub const KERNEL: &str = "teeth_blend_kernel";
const BLOCK: u32 = 16;

const PTX: &str = include_str!("../native/cuda/teeth_blend.ptx");

/// 32-byte kernel argument block, matching `native/cuda/teeth_blend.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u32,
    pub reserved3: u32,
    pub reserved4: u32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 32);

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

/// Mirrors `teeth_blend::apply`'s entry checks.
fn check(original: &Image, effect: &Image, mask: &[u8]) -> Result<(), Refusal> {
    original.validate().map_err(|_| Refusal::ShapeMismatch("teeth blend original"))?;
    effect.validate().map_err(|_| Refusal::ShapeMismatch("teeth blend effect"))?;
    if original.width != effect.width || original.height != effect.height {
        return Err(Refusal::ShapeMismatch("teeth blend effect"));
    }
    if mask.len() != original.pixel_count() {
        return Err(Refusal::ShapeMismatch("teeth blend mask"));
    }
    Ok(())
}

#[cfg(windows)]
pub struct TeethBlend {
    gpu: crate::cuda::Nvidia,
}

#[cfg(windows)]
impl TeethBlend {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self { gpu: crate::cuda::Nvidia::open_ptx(PTX)? })
    }

    /// GPU twin of `teeth_blend::apply`.
    pub fn apply(
        &self,
        original: &Image,
        effect: &Image,
        mask: &[u8],
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        check(original, effect, mask).map_err(CudaError::Refused)?;
        let uniforms = Uniforms {
            width: original.width,
            height: original.height,
            stride: original.width * 4,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
            reserved4: 0,
        };
        let grid = (
            original.width.div_ceil(BLOCK).max(1),
            original.height.div_ceil(BLOCK).max(1),
        );
        let (pixels, dispatch) = self.gpu.run_u8(
            KERNEL,
            grid,
            &[&original.pixels, &effect.pixels, mask],
            as_bytes(&uniforms),
        )?;
        let image = Image::new(original.width, original.height, pixels)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("teeth blend output")))?;
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
    fn checks_match_the_cpu_lane() {
        let image = Image::solid(2, 2, [10, 20, 30, 40]);
        assert_eq!(check(&image, &image, &[0; 4]), Ok(()));
        assert_eq!(
            check(&image, &image, &[0; 3]),
            Err(Refusal::ShapeMismatch("teeth blend mask"))
        );
        let other = Image::solid(1, 1, [0; 4]);
        assert_eq!(
            check(&image, &other, &[0; 4]),
            Err(Refusal::ShapeMismatch("teeth blend effect"))
        );
    }
}
