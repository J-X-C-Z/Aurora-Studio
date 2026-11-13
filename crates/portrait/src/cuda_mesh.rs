//! NVIDIA execution of the O3b mesh rasterizer (`native/cuda/mesh_raster.*`).
//!
//! Two kernels, run in order against one caller-owned scratch pair:
//!   select  one thread per triangle, bbox scan, per-pixel resolve
//!   emit    one thread per pixel, re-evaluate, bilinear sample
//!
//! Both are 1D over triangles/pixels with a 128-thread block, which is the
//! shape the shipped PTX was compiled for; the P15 16x16 tiling does not apply
//! and would repeat coverage per triangle. `Dispatch` records the real geometry.
//!
//! Scratch starts at `selected = u32::MAX`, `coverage = 0`; `out` is prefilled
//! with the source bytes, so uncovered pixels keep the source exactly.
//!
//! Driver, context, allocation and the 222-only PTX retry live in
//! [`crate::cuda`]; this module validates inputs, builds the ABI and launches.

use crate::cuda::{CudaError, Dispatch};
use crate::raster::{Cull, Edge, Mesh, Overlap, Rules, Sampler, Winding, Wrap};
use crate::{Image, Refusal};

/// Kernel block: the shipped PTX assumes `blockIdx.x * blockDim.x + threadIdx.x`.
const BLOCK: u32 = 128;

const PTX: &str = include_str!("../native/cuda/mesh_raster.ptx");

/// 16-byte rule block, matching `MeshRasterRules` in `mesh_raster.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RulesAbi {
    winding: i32,
    cull: i32,
    edge: i32,
    overlap: i32,
}

const _: () = assert!(std::mem::size_of::<RulesAbi>() == 16);

impl RulesAbi {
    fn new(rules: Rules) -> Self {
        Self {
            winding: match rules.winding {
                Winding::PositiveAreaIsFront => 1,
                Winding::NegativeAreaIsFront => 2,
            },
            cull: match rules.cull {
                Cull::None => 1,
                Cull::Front => 2,
                Cull::Back => 3,
            },
            edge: match rules.edge {
                Edge::Exclusive => 1,
                Edge::Inclusive => 2,
            },
            overlap: match rules.overlap {
                Overlap::Reject => 1,
                Overlap::First => 2,
                Overlap::Last => 3,
            },
        }
    }
}

/// The PTX keeps both kernels in an anonymous namespace, so the entry names are
/// mangled with a build hash. Resolve them from the shipped image instead of
/// hardcoding that hash.
fn entry(fragment: &str) -> Option<&'static str> {
    PTX.lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            trimmed.strip_prefix(".visible .entry ").or_else(|| trimmed.strip_prefix(".entry "))
        })
        .filter_map(|rest| rest.split('(').next())
        .find(|symbol| symbol.contains(fragment))
}

fn bytes_of<T>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) }
}

fn bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, std::mem::size_of_val(values))
    }
}

fn grid(items: usize) -> (u32, u32) {
    ((items.div_ceil(BLOCK as usize) as u32).max(1), 1)
}

/// Output of one raster pass pair, with the scratch it resolved.
pub struct Output {
    pub image: Image,
    pub selected: Vec<u32>,
    pub coverage: Vec<u32>,
    pub select_dispatch: Dispatch,
    pub emit_dispatch: Dispatch,
}

#[cfg(windows)]
pub struct MeshRaster {
    gpu: crate::cuda::Nvidia,
    select: &'static str,
    emit: &'static str,
}

#[cfg(windows)]
impl MeshRaster {
    pub fn open() -> Result<Self, CudaError> {
        let select = entry("mesh_raster_select").ok_or(CudaError::Unavailable("mesh_raster_select"))?;
        let emit = entry("mesh_raster_emit").ok_or(CudaError::Unavailable("mesh_raster_emit"))?;
        Ok(Self { gpu: crate::cuda::Nvidia::open_ptx(PTX)?, select, emit })
    }

    /// GPU twin of `raster::rasterize`.
    pub fn rasterize(
        &self,
        source: &Image,
        mesh: Mesh<'_>,
        rules: Rules,
        sampler: Sampler,
    ) -> Result<Output, CudaError> {
        validate(source, &mesh)?;
        let width = source.width as usize;
        let height = source.height as usize;
        let pixels = width * height;
        let triangles = mesh.ebo.len() / 3;
        let rules_abi = RulesAbi::new(rules);
        let wrap = match sampler.wrap {
            Wrap::ClampToEdge => 0i32,
            Wrap::Repeat => 1i32,
        };

        let dest_buf = self.gpu.alloc_buffer(std::mem::size_of_val(mesh.dest))?;
        let uv_buf = self.gpu.alloc_buffer(std::mem::size_of_val(mesh.source_uv))?;
        let ebo_buf = self.gpu.alloc_buffer(std::mem::size_of_val(mesh.ebo))?;
        let source_buf = self.gpu.alloc_buffer(source.pixels.len())?;
        let out_buf = self.gpu.alloc_buffer(source.pixels.len())?;
        let selected_buf = self.gpu.alloc_buffer(pixels * 4)?;
        let coverage_buf = self.gpu.alloc_buffer(pixels * 4)?;
        self.gpu.upload_buffer(&dest_buf, bytes_of(mesh.dest))?;
        self.gpu.upload_buffer(&uv_buf, bytes_of(mesh.source_uv))?;
        self.gpu.upload_buffer(&ebo_buf, bytes_of(mesh.ebo))?;
        self.gpu.upload_buffer(&source_buf, &source.pixels)?;
        self.gpu.upload_buffer(&out_buf, &source.pixels)?;
        // Last uses atomicMax, so its sentinel must be below every triangle id.
        let selected_fill = if rules.overlap == Overlap::Last { 0u8 } else { 0xFFu8 };
        self.gpu.upload_buffer(&selected_buf, &vec![selected_fill; pixels * 4])?;
        self.gpu.upload_buffer(&coverage_buf, &vec![0u8; pixels * 4])?;

        let mut width_arg = width as i32;
        let mut height_arg = height as i32;
        let mut triangle_arg = triangles as i32;
        let mut rules_arg = rules_abi;
        let mut dest_arg = dest_buf.handle();
        let mut ebo_arg = ebo_buf.handle();
        let mut selected_arg = selected_buf.handle();
        let mut coverage_arg = coverage_buf.handle();
        let mut select_params: [*mut std::ffi::c_void; 8] = [
            &mut dest_arg as *mut u64 as *mut std::ffi::c_void,
            &mut ebo_arg as *mut u64 as *mut std::ffi::c_void,
            &mut width_arg as *mut i32 as *mut std::ffi::c_void,
            &mut height_arg as *mut i32 as *mut std::ffi::c_void,
            &mut triangle_arg as *mut i32 as *mut std::ffi::c_void,
            &mut rules_arg as *mut RulesAbi as *mut std::ffi::c_void,
            &mut selected_arg as *mut u64 as *mut std::ffi::c_void,
            &mut coverage_arg as *mut u64 as *mut std::ffi::c_void,
        ];
        let select_dispatch =
            self.gpu.launch_raw(self.select, grid(triangles), (BLOCK, 1), &mut select_params)?;

        let mut uv_arg = uv_buf.handle();
        let mut source_arg = source_buf.handle();
        let mut out_arg = out_buf.handle();
        let mut wrap_arg = wrap;
        let mut emit_params: [*mut std::ffi::c_void; 11] = [
            &mut dest_arg as *mut u64 as *mut std::ffi::c_void,
            &mut uv_arg as *mut u64 as *mut std::ffi::c_void,
            &mut ebo_arg as *mut u64 as *mut std::ffi::c_void,
            &mut source_arg as *mut u64 as *mut std::ffi::c_void,
            &mut out_arg as *mut u64 as *mut std::ffi::c_void,
            &mut width_arg as *mut i32 as *mut std::ffi::c_void,
            &mut height_arg as *mut i32 as *mut std::ffi::c_void,
            &mut wrap_arg as *mut i32 as *mut std::ffi::c_void,
            &mut rules_arg as *mut RulesAbi as *mut std::ffi::c_void,
            &mut selected_arg as *mut u64 as *mut std::ffi::c_void,
            &mut coverage_arg as *mut u64 as *mut std::ffi::c_void,
        ];
        let emit_dispatch = self.gpu.launch_raw(self.emit, grid(pixels), (BLOCK, 1), &mut emit_params)?;

        let mut pixels_out = vec![0u8; source.pixels.len()];
        let mut selected = vec![0u32; pixels];
        let mut coverage = vec![0u32; pixels];
        self.gpu.download_buffer(&out_buf, &mut pixels_out)?;
        self.gpu.download_buffer(&selected_buf, bytes_mut(&mut selected))?;
        self.gpu.download_buffer(&coverage_buf, bytes_mut(&mut coverage))?;
        let image = Image::new(source.width, source.height, pixels_out)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok(Output { image, selected, coverage, select_dispatch, emit_dispatch })
    }
}

/// Mirrors `raster::validate` and adds the native index bounds the kernels
/// compute in 32-bit ints, so an unrepresentable shape is refused before launch.
fn validate(source: &Image, mesh: &Mesh<'_>) -> Result<(), Refusal> {
    source.validate()?;
    if mesh.dest.len() != mesh.source_uv.len() {
        return Err(Refusal::ShapeMismatch("source_uv"));
    }
    if mesh.dest.len() < 3 {
        return Err(Refusal::ShapeMismatch("dest"));
    }
    if mesh.ebo.is_empty() {
        return Err(Refusal::MissingInput("ebo"));
    }
    if mesh.ebo.len() % 3 != 0 {
        return Err(Refusal::ShapeMismatch("ebo"));
    }
    if mesh.dest.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("dest"));
    }
    if mesh.source_uv.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("source_uv"));
    }
    if mesh.ebo.iter().any(|&index| index as usize >= mesh.dest.len()) {
        return Err(Refusal::ValueOutOfRange("ebo index"));
    }
    let pixels = source.width as u64 * source.height as u64;
    if pixels > i32::MAX as u64 {
        return Err(Refusal::ValueOutOfRange("shape"));
    }
    if mesh.ebo.len() as u64 > i32::MAX as u64 {
        return Err(Refusal::ValueOutOfRange("ebo"));
    }
    if mesh.dest.len() as u64 > i32::MAX as u64 {
        return Err(Refusal::ValueOutOfRange("dest"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rule_block_matches_the_c_header() {
        assert_eq!(std::mem::size_of::<RulesAbi>(), 16);
        let abi = RulesAbi::new(Rules {
            winding: Winding::NegativeAreaIsFront,
            cull: Cull::Front,
            edge: Edge::Inclusive,
            overlap: Overlap::Last,
        });
        assert_eq!(abi, RulesAbi { winding: 2, cull: 2, edge: 2, overlap: 3 });
    }

    #[test]
    fn both_kernel_entries_resolve_from_the_shipped_image() {
        let select = entry("mesh_raster_select").expect("select entry");
        let emit = entry("mesh_raster_emit").expect("emit entry");
        assert_ne!(select, emit);
        assert!(PTX.contains(".target sm_"));
    }

    #[test]
    fn malformed_meshes_are_refused_before_launch() {
        let source = Image::solid(2, 2, [0, 0, 0, 0]);
        let dest = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mesh = Mesh { dest: &dest, source_uv: &uv, ebo: &[0, 1, 2] };
        assert_eq!(validate(&source, &mesh), Ok(()));
        let short = Mesh { dest: &dest, source_uv: &uv, ebo: &[0, 1, 7] };
        assert_eq!(validate(&source, &short), Err(Refusal::ValueOutOfRange("ebo index")));
        let ragged = Mesh { dest: &dest, source_uv: &uv, ebo: &[0, 1] };
        assert_eq!(validate(&source, &ragged), Err(Refusal::ShapeMismatch("ebo")));
    }
}
