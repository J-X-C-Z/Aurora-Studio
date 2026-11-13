//! PF8033 attribute rasterizer on NVIDIA.
//!
//! Two passes over the shared CUDA driver helpers:
//!   select   one thread per triangle, records the highest covering triangle
//!            index and the covering count per pixel
//!   resolve  one thread per pixel, re-evaluates that winning triangle
//!
//! "Highest index wins" is exactly the CPU's triangle-order last-writer rule, so
//! the pixels and the write counts match `kernels::mesh_attribute::rasterize`
//! bit for bit. Concurrent writes are never raced: the winner is chosen by an
//! atomic max on the index, and the value is written once by the pixel thread.

use crate::cuda::{CudaError, Dispatch};
#[cfg(windows)]
use crate::cuda::Nvidia;
use crate::kernels::mesh_attribute::DenseAttribute;
use crate::Refusal;

pub const SELECT_KERNEL: &str = "mesh_attribute_select_kernel";
pub const RESOLVE_KERNEL: &str = "mesh_attribute_resolve_kernel";
const BLOCK: u32 = 128;

const PTX: &str = include_str!("../native/cuda/mesh_attribute.ptx");

/// 32-byte kernel argument block, matching `native/cuda/mesh_attribute.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub vertex_count: u32,
    pub triangle_count: u32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u32,
    pub reserved3: u32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 32);

fn f32_pairs_bytes(values: &[[f32; 2]]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) }
}

fn u32_bytes(values: &[u32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) }
}

/// Mirrors `mesh_attribute::rasterize`'s entry checks.
fn check(
    width: u32,
    height: u32,
    dest: &[[f32; 2]],
    attr: &[[f32; 2]],
    faces: &[u32],
) -> Result<usize, Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("mesh attribute size"));
    }
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("mesh attribute size"))?;
    if dest.is_empty() || dest.len() != attr.len() {
        return Err(Refusal::ShapeMismatch("mesh attribute vertices"));
    }
    if faces.is_empty() || faces.len() % 3 != 0 {
        return Err(Refusal::ShapeMismatch("mesh attribute faces"));
    }
    if dest
        .iter()
        .chain(attr)
        .flat_map(|value| value.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("mesh attribute vertex"));
    }
    if faces.iter().any(|index| *index as usize >= dest.len()) {
        return Err(Refusal::ValueOutOfRange("mesh attribute face index"));
    }
    Ok(pixels)
}

#[cfg(windows)]
pub struct MeshAttribute {
    gpu: Nvidia,
}

#[cfg(windows)]
impl MeshAttribute {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self { gpu: Nvidia::open_ptx(PTX)? })
    }

    /// GPU twin of `mesh_attribute::rasterize`.
    pub fn rasterize(
        &self,
        width: u32,
        height: u32,
        dest: &[[f32; 2]],
        attr: &[[f32; 2]],
        faces: &[u32],
    ) -> Result<(DenseAttribute, Vec<Dispatch>), CudaError> {
        let pixels = check(width, height, dest, attr, faces)?;
        let triangles = faces.len() / 3;
        let uniforms = Uniforms {
            width,
            height,
            vertex_count: dest.len() as u32,
            triangle_count: triangles as u32,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
            reserved3: 0,
        };

        let dest_buf = self.gpu.alloc_buffer(std::mem::size_of_val(dest))?;
        let attr_buf = self.gpu.alloc_buffer(std::mem::size_of_val(attr))?;
        let faces_buf = self.gpu.alloc_buffer(std::mem::size_of_val(faces))?;
        let last_buf = self.gpu.alloc_buffer(pixels * 4)?;
        let count_buf = self.gpu.alloc_buffer(pixels * 4)?;
        let values_buf = self.gpu.alloc_buffer(pixels * 8)?;
        self.gpu.upload_buffer(&values_buf, &vec![0u8; pixels * 8])?;
        self.gpu.upload_buffer(&dest_buf, f32_pairs_bytes(dest))?;
        self.gpu.upload_buffer(&attr_buf, f32_pairs_bytes(attr))?;
        self.gpu.upload_buffer(&faces_buf, u32_bytes(faces))?;
        self.gpu.upload_buffer(&last_buf, &vec![0u8; pixels * 4])?;
        self.gpu.upload_buffer(&count_buf, &vec![0u8; pixels * 4])?;

        let mut dest_arg = dest_buf.handle();
        let mut attr_arg = attr_buf.handle();
        let mut faces_arg = faces_buf.handle();
        let mut last_arg = last_buf.handle();
        let mut count_arg = count_buf.handle();
        let mut values_arg = values_buf.handle();
        let mut uniforms_arg = uniforms;

        let mut select_params: [*mut std::ffi::c_void; 5] = [
            &mut dest_arg as *mut u64 as *mut std::ffi::c_void,
            &mut faces_arg as *mut u64 as *mut std::ffi::c_void,
            &mut last_arg as *mut u64 as *mut std::ffi::c_void,
            &mut count_arg as *mut u64 as *mut std::ffi::c_void,
            &mut uniforms_arg as *mut Uniforms as *mut std::ffi::c_void,
        ];
        let select_grid = (triangles.div_ceil(BLOCK as usize) as u32, 1);
        let select_dispatch =
            self.gpu.launch_raw(SELECT_KERNEL, select_grid, (BLOCK, 1), &mut select_params)?;

        let mut resolve_params: [*mut std::ffi::c_void; 6] = [
            &mut dest_arg as *mut u64 as *mut std::ffi::c_void,
            &mut attr_arg as *mut u64 as *mut std::ffi::c_void,
            &mut faces_arg as *mut u64 as *mut std::ffi::c_void,
            &mut last_arg as *mut u64 as *mut std::ffi::c_void,
            &mut values_arg as *mut u64 as *mut std::ffi::c_void,
            &mut uniforms_arg as *mut Uniforms as *mut std::ffi::c_void,
        ];
        let resolve_grid = (pixels.div_ceil(BLOCK as usize) as u32, 1);
        let resolve_dispatch =
            self.gpu.launch_raw(RESOLVE_KERNEL, resolve_grid, (BLOCK, 1), &mut resolve_params)?;

        let mut raw_values = vec![0u8; pixels * 8];
        let mut raw_counts = vec![0u8; pixels * 4];
        self.gpu.download_buffer(&values_buf, &mut raw_values)?;
        self.gpu.download_buffer(&count_buf, &mut raw_counts)?;
        let values: Vec<[f32; 2]> = raw_values
            .chunks_exact(8)
            .map(|chunk| {
                [
                    f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                    f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
                ]
            })
            .collect();
        // The CPU saturates each increment, so the total is min(sum, u16::MAX).
        let writes: Vec<u16> = raw_counts
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).min(u16::MAX as u32) as u16)
            .collect();
        Ok((DenseAttribute { values, writes }, vec![select_dispatch, resolve_dispatch]))
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
    fn entry_checks_match_the_cpu_lane() {
        let dest = [[0.0f32, 0.0], [4.0, 0.0], [0.0, 4.0]];
        let attr = [[0.0f32, 0.0]; 3];
        assert_eq!(
            check(0, 4, &dest, &attr, &[0, 1, 2]),
            Err(Refusal::ShapeMismatch("mesh attribute size"))
        );
        assert_eq!(
            check(4, 4, &dest, &attr, &[0, 1]),
            Err(Refusal::ShapeMismatch("mesh attribute faces"))
        );
        assert_eq!(
            check(4, 4, &dest, &attr, &[0, 1, 3]),
            Err(Refusal::ValueOutOfRange("mesh attribute face index"))
        );
    }
}
