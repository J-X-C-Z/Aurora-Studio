//! PF90118 lip-line compositor on NVIDIA.
//!
//! PF90118 and PF21026 share the native compositor at `0x82AA2E0`. Reuse the
//! existing beard PTX with unit strength, but do not apply PF21026's `0.005`
//! gate or its `0.75` small-frame scale. `source`/`repair` are caller-provided
//! BGRA8 tiles; `mask_a`/`mask_b` are explicit same-job u8 planes. The 11 lip
//! points select the producer ROI and never participate in the pixel equation.

#[cfg(windows)]
use crate::cuda::Nvidia;
use crate::cuda::{CudaError, Dispatch};
use crate::cuda_beard::{Uniforms, KERNEL};
use crate::{Image, Refusal};

const BLOCK: u32 = 16;
const PTX: &str = include_str!("../native/cuda/beard.ptx");

fn check(source: &Image, repair: &Image, mask_a: &[u8], mask_b: &[u8]) -> Result<usize, Refusal> {
    source.validate()?;
    repair
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("repair"))?;
    if (source.width, source.height) != (repair.width, repair.height) {
        return Err(Refusal::ShapeMismatch("repair"));
    }
    let pixels = source.pixel_count();
    if mask_a.len() != pixels {
        return Err(Refusal::ShapeMismatch("mask_a"));
    }
    if mask_b.len() != pixels {
        return Err(Refusal::ShapeMismatch("mask_b"));
    }
    Ok(pixels)
}

#[cfg(windows)]
pub struct LipLineGpu {
    gpu: Nvidia,
}

#[cfg(windows)]
impl LipLineGpu {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self {
            gpu: Nvidia::open_ptx(PTX)?,
        })
    }

    pub fn apply(
        &self,
        source: &Image,
        repair: &Image,
        mask_a: &[u8],
        mask_b: &[u8],
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        check(source, repair, mask_a, mask_b).map_err(CudaError::Refused)?;
        let uniforms = Uniforms {
            width: source.width,
            height: source.height,
            stride: source.width * 4,
            strength: 1.0,
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
            unsafe {
                std::slice::from_raw_parts(
                    &uniforms as *const Uniforms as *const u8,
                    std::mem::size_of::<Uniforms>(),
                )
            },
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
    fn shape_refusals_name_the_missing_plane() {
        let image = Image::solid(2, 2, [0; 4]);
        assert_eq!(
            check(&image, &image, &[0; 3], &[0; 4]),
            Err(Refusal::ShapeMismatch("mask_a"))
        );
        assert_eq!(
            check(&image, &image, &[0; 4], &[0; 3]),
            Err(Refusal::ShapeMismatch("mask_b"))
        );
    }
}
