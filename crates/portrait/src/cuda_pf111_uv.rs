//! Minimal NVIDIA adapter for PF111's typed per-pixel UV producer.
//!
//! The existing `mesh_attribute` PTX already implements the CPU reference's
//! coverage and interpolation. PF111 only supplies `u16` EBO indices; this
//! adapter widens them and preserves the typed PF111 result.

use crate::cuda::{CudaError, Dispatch};
#[cfg(windows)]
use crate::cuda_mesh_attribute::MeshAttribute;
use crate::kernels::pf111_uv::{Pf111PixelUv, Pf111RasterInputs, from_dense, raster_args};

#[cfg(windows)]
pub struct Pf111UvRaster {
    gpu: MeshAttribute,
}

#[cfg(windows)]
impl Pf111UvRaster {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self {
            gpu: MeshAttribute::open()?,
        })
    }

    pub fn rasterize(
        &self,
        inputs: Pf111RasterInputs<'_>,
    ) -> Result<(Pf111PixelUv, Vec<Dispatch>), CudaError> {
        let args = raster_args(inputs)?;
        let (dense, dispatches) = self.gpu.rasterize(
            args.width,
            args.height,
            args.positions,
            args.normal_eye_uv,
            &args.faces,
        )?;
        Ok((from_dense(args.side, dense), dispatches))
    }
}
