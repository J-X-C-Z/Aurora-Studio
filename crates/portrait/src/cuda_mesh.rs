//! NVIDIA execution of the O3b mesh rasterizer (`native/cuda/mesh_raster.*`).
//!
//! Two kernels, run in order against one caller-owned scratch pair:
//!   select  one thread per (triangle, scanline), bbox scan, per-pixel resolve
//!   emit    one thread per pixel, re-evaluate, bilinear sample
//!
//! Both are 1D with a 128-thread block, which is the shape the shipped PTX was
//! compiled for; the P15 16x16 tiling does not apply and would repeat coverage
//! per triangle. `Dispatch` records the real geometry.
//!
//! Scratch starts at `selected = u32::MAX`, `coverage = 0`; `out` is prefilled
//! with the source bytes, so uncovered pixels keep the source exactly.
//!
//! Driver, context, allocation and the 222-only PTX retry live in
//! [`crate::cuda`]; this module validates inputs, builds the ABI and launches.

use crate::cuda::{CudaError, Dispatch};
use crate::raster::{Cull, Edge, Mesh, Overlap, Rules, Sampler, Tolerance, Winding, Wrap};
use crate::raster_fragment::{Clamp as FragmentClamp, Defines as FragmentDefines};
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

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct ClampAbi {
    factor: f32,
    protect_x: i32,
    protect_y: i32,
}

const _: () = assert!(std::mem::size_of::<ClampAbi>() == 12);

impl ClampAbi {
    fn new(clamp: FragmentClamp) -> Self {
        Self {
            factor: clamp.factor,
            protect_x: clamp.protect_x as i32,
            protect_y: clamp.protect_y as i32,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DefinesAbi {
    clamp_edge: i32,
    uv_map: i32,
    uv_map_ba: i32,
    catmull_rom: i32,
}

const _: () = assert!(std::mem::size_of::<DefinesAbi>() == 16);

impl DefinesAbi {
    fn new(defines: FragmentDefines) -> Self {
        Self {
            clamp_edge: i32::from(defines.clamp_edge),
            uv_map: i32::from(defines.uv_map),
            uv_map_ba: i32::from(defines.uv_map_ba),
            catmull_rom: i32::from(defines.catmull_rom),
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
            trimmed
                .strip_prefix(".visible .entry ")
                .or_else(|| trimmed.strip_prefix(".entry "))
        })
        .filter_map(|rest| rest.split('(').next())
        .find(|symbol| symbol.contains(fragment))
}

fn bytes_of<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

fn bytes_mut<T>(values: &mut [T]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(
            values.as_mut_ptr() as *mut u8,
            std::mem::size_of_val(values),
        )
    }
}

const MAX_GRID_X: usize = i32::MAX as usize;

fn checked_grid(items: usize, label: &'static str) -> Result<u32, CudaError> {
    let block = BLOCK as usize;
    let blocks = items / block + usize::from(items % block != 0);
    if blocks == 0 || blocks > MAX_GRID_X {
        return Err(CudaError::Refused(Refusal::ValueOutOfRange(label)));
    }
    Ok(blocks as u32)
}

/// Launch grid for the scanline-expanded select pass.
pub fn select_grid(triangles: usize, height: usize) -> Result<u32, CudaError> {
    let items = triangles
        .checked_mul(height)
        .ok_or(CudaError::Refused(Refusal::ValueOutOfRange("select grid")))?;
    checked_grid(items, "select grid")
}

/// Output of one raster pass pair, with the scratch it resolved.
pub struct Output {
    pub image: Image,
    pub selected: Vec<u32>,
    pub coverage: Vec<u32>,
    pub select_dispatch: Dispatch,
    pub emit_dispatch: Dispatch,
}

/// Scratch and launch evidence from a write into an existing RGBA8 image.
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
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
    fragment_emit: &'static str,
}

#[cfg(windows)]
impl MeshRaster {
    pub fn open() -> Result<Self, CudaError> {
        let select =
            entry("mesh_raster_select").ok_or(CudaError::Unavailable("mesh_raster_select"))?;
        let emit = entry("mesh_raster_emit").ok_or(CudaError::Unavailable("mesh_raster_emit"))?;
        let fragment_emit = entry("mesh_raster_fragment_emit")
            .ok_or(CudaError::Unavailable("mesh_raster_fragment_emit"))?;
        Ok(Self {
            gpu: crate::cuda::Nvidia::open_ptx(PTX)?,
            select,
            emit,
            fragment_emit,
        })
    }

    /// GPU twin of `raster::rasterize`, writing the complete RGBA8 result into
    /// `out`. Uncovered pixels are reset to the source bytes before launch.
    pub fn rasterize_into(
        &self,
        source: &Image,
        mesh: Mesh<'_>,
        rules: Rules,
        sampler: Sampler,
        out: &mut Image,
    ) -> Result<Trace, CudaError> {
        validate(source, &mesh)?;
        if out.width != source.width || out.height != source.height {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("out")));
        }
        out.validate()?;
        out.pixels.copy_from_slice(&source.pixels);
        let width = source.width as usize;
        let height = source.height as usize;
        let pixels = width * height;
        let triangles = mesh.ebo.len() / 3;
        let select_grid = select_grid(triangles, height)?;
        let emit_grid = checked_grid(pixels, "emit grid")?;
        let rules_abi = RulesAbi::new(rules);
        let wrap = match sampler.wrap {
            Wrap::ClampToEdge => 0i32,
            Wrap::Repeat => 1i32,
        };

        self.gpu.with_reusable_buffers(
            "mesh.raster",
            &[
                std::mem::size_of_val(mesh.dest),
                std::mem::size_of_val(mesh.source_uv),
                std::mem::size_of_val(mesh.ebo),
                source.pixels.len(),
                source.pixels.len(),
                pixels * 4,
                pixels * 4,
            ],
            |buffers| {
                buffers.upload(0, bytes_of(mesh.dest))?;
                buffers.upload(1, bytes_of(mesh.source_uv))?;
                buffers.upload(2, bytes_of(mesh.ebo))?;
                buffers.upload(3, &source.pixels)?;
                buffers.upload(4, &out.pixels)?;
                // Last uses atomicMax, so its sentinel must be below every triangle id.
                let selected_fill = if rules.overlap == Overlap::Last {
                    0u8
                } else {
                    0xFFu8
                };
                buffers.upload(5, &vec![selected_fill; pixels * 4])?;
                buffers.upload(6, &vec![0u8; pixels * 4])?;

                let mut width_arg = width as i32;
                let mut height_arg = height as i32;
                let mut triangle_arg = triangles as i32;
                let mut rules_arg = rules_abi;
                let mut degenerate_area2 = Tolerance::TILE.degenerate_area2;
                let mut edge_epsilon = Tolerance::TILE.edge;
                let mut dest_arg = buffers.handle(0);
                let mut ebo_arg = buffers.handle(2);
                let mut selected_arg = buffers.handle(5);
                let mut coverage_arg = buffers.handle(6);
                let mut select_params: [*mut std::ffi::c_void; 10] = [
                    &mut dest_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut ebo_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut width_arg as *mut i32 as *mut std::ffi::c_void,
                    &mut height_arg as *mut i32 as *mut std::ffi::c_void,
                    &mut triangle_arg as *mut i32 as *mut std::ffi::c_void,
                    &mut rules_arg as *mut RulesAbi as *mut std::ffi::c_void,
                    &mut degenerate_area2 as *mut f32 as *mut std::ffi::c_void,
                    &mut edge_epsilon as *mut f32 as *mut std::ffi::c_void,
                    &mut selected_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut coverage_arg as *mut u64 as *mut std::ffi::c_void,
                ];
                let select_dispatch = buffers.launch(
                    self.select,
                    (select_grid, 1),
                    (BLOCK, 1),
                    &mut select_params,
                )?;

                let mut uv_arg = buffers.handle(1);
                let mut source_arg = buffers.handle(3);
                let mut out_arg = buffers.handle(4);
                let mut wrap_arg = wrap;
                let mut emit_params: [*mut std::ffi::c_void; 13] = [
                    &mut dest_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut uv_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut ebo_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut source_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut out_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut width_arg as *mut i32 as *mut std::ffi::c_void,
                    &mut height_arg as *mut i32 as *mut std::ffi::c_void,
                    &mut wrap_arg as *mut i32 as *mut std::ffi::c_void,
                    &mut rules_arg as *mut RulesAbi as *mut std::ffi::c_void,
                    &mut degenerate_area2 as *mut f32 as *mut std::ffi::c_void,
                    &mut edge_epsilon as *mut f32 as *mut std::ffi::c_void,
                    &mut selected_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut coverage_arg as *mut u64 as *mut std::ffi::c_void,
                ];
                let emit_dispatch =
                    buffers.launch(self.emit, (emit_grid, 1), (BLOCK, 1), &mut emit_params)?;

                let mut selected = vec![0u32; pixels];
                let mut coverage = vec![0u32; pixels];
                buffers.download(4, &mut out.pixels)?;
                buffers.download(5, bytes_mut(&mut selected))?;
                buffers.download(6, bytes_mut(&mut coverage))?;
                buffers.mark_consumed();
                Ok(Trace {
                    selected,
                    coverage,
                    select_dispatch,
                    emit_dispatch,
                })
            },
        )
    }

    /// GPU twin of `raster::rasterize`.
    pub fn rasterize(
        &self,
        source: &Image,
        mesh: Mesh<'_>,
        rules: Rules,
        sampler: Sampler,
    ) -> Result<Output, CudaError> {
        let mut image = source.clone();
        let trace = self.rasterize_into(source, mesh, rules, sampler, &mut image)?;
        Ok(Output {
            image,
            selected: trace.selected,
            coverage: trace.coverage,
            select_dispatch: trace.select_dispatch,
            emit_dispatch: trace.emit_dispatch,
        })
    }

    /// GPU twin of `raster::rasterize_with_fragment`.
    #[allow(clippy::too_many_arguments)]
    pub fn rasterize_fragment(
        &self,
        source: &Image,
        mesh: Mesh<'_>,
        rules: Rules,
        sampler: Sampler,
        edge: crate::raster_fragment::Edge,
        defines: FragmentDefines,
        clamp: FragmentClamp,
        tolerance: Tolerance,
    ) -> Result<Output, CudaError> {
        self.rasterize_fragment_into(
            source, source, mesh, rules, sampler, edge, defines, clamp, tolerance,
        )
    }

    /// Fragment writer with independent source and target images.
    #[allow(clippy::too_many_arguments)]
    pub fn rasterize_fragment_into(
        &self,
        source: &Image,
        target: &Image,
        mesh: Mesh<'_>,
        rules: Rules,
        sampler: Sampler,
        edge: crate::raster_fragment::Edge,
        defines: FragmentDefines,
        clamp: FragmentClamp,
        tolerance: Tolerance,
    ) -> Result<Output, CudaError> {
        validate(source, &mesh)?;
        target.validate().map_err(CudaError::Refused)?;
        if target.width != source.width || target.height != source.height {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("output")));
        }
        if sampler.wrap == Wrap::Repeat {
            return Err(CudaError::Refused(Refusal::MissingInput(
                "fragment repeat wrap",
            )));
        }
        if !tolerance.edge.is_finite()
            || !tolerance.degenerate_area2.is_finite()
            || tolerance.degenerate_area2 < 0.0
        {
            return Err(CudaError::Refused(Refusal::ValueOutOfRange(
                "fragment tolerance",
            )));
        }

        let width = source.width as usize;
        let height = source.height as usize;
        let pixels = width * height;
        let triangles = mesh.ebo.len() / 3;
        let select_grid = select_grid(triangles, height)?;
        let emit_grid = checked_grid(pixels, "emit grid")?;
        let rules_abi = RulesAbi::new(rules);
        let fragment_edge = match edge {
            crate::raster_fragment::Edge::ReshapeNormal => 0i32,
            crate::raster_fragment::Edge::AdjClampEdge => 1i32,
        };
        let clamp_abi = ClampAbi::new(clamp);
        let defines_abi = DefinesAbi::new(defines);

        self.gpu.with_reusable_buffers(
            "mesh.raster-fragment",
            &[
                std::mem::size_of_val(mesh.dest),
                std::mem::size_of_val(mesh.source_uv),
                std::mem::size_of_val(mesh.ebo),
                source.pixels.len(),
                source.pixels.len(),
                pixels * 4,
                pixels * 4,
            ],
            |buffers| {
                buffers.upload(0, bytes_of(mesh.dest))?;
                buffers.upload(1, bytes_of(mesh.source_uv))?;
                buffers.upload(2, bytes_of(mesh.ebo))?;
                buffers.upload(3, &source.pixels)?;
                buffers.upload(4, &target.pixels)?;
                let selected_fill = if rules.overlap == Overlap::Last {
                    0u8
                } else {
                    0xFFu8
                };
                buffers.upload(5, &vec![selected_fill; pixels * 4])?;
                buffers.upload(6, &vec![0u8; pixels * 4])?;

                let mut width_arg = width as i32;
                let mut height_arg = height as i32;
                let mut triangle_arg = triangles as i32;
                let mut rules_arg = rules_abi;
                let mut dest_arg = buffers.handle(0);
                let mut ebo_arg = buffers.handle(2);
                let mut selected_arg = buffers.handle(5);
                let mut coverage_arg = buffers.handle(6);
                let mut degenerate_area2 = tolerance.degenerate_area2;
                let mut edge_epsilon = tolerance.edge;
                let mut select_params: [*mut std::ffi::c_void; 10] = [
                    &mut dest_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut ebo_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut width_arg as *mut i32 as *mut std::ffi::c_void,
                    &mut height_arg as *mut i32 as *mut std::ffi::c_void,
                    &mut triangle_arg as *mut i32 as *mut std::ffi::c_void,
                    &mut rules_arg as *mut RulesAbi as *mut std::ffi::c_void,
                    &mut degenerate_area2 as *mut f32 as *mut std::ffi::c_void,
                    &mut edge_epsilon as *mut f32 as *mut std::ffi::c_void,
                    &mut selected_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut coverage_arg as *mut u64 as *mut std::ffi::c_void,
                ];
                let select_dispatch = buffers.launch(
                    self.select,
                    (select_grid, 1),
                    (BLOCK, 1),
                    &mut select_params,
                )?;

                let mut uv_arg = buffers.handle(1);
                let mut source_arg = buffers.handle(3);
                let mut out_arg = buffers.handle(4);
                let mut fragment_edge_arg = fragment_edge;
                let mut clamp_arg = clamp_abi;
                let mut defines_arg = defines_abi;
                let mut catmull_width = 0.0f32;
                let mut catmull_height = 0.0f32;
                let mut emit_params: [*mut std::ffi::c_void; 17] = [
                    &mut dest_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut uv_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut ebo_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut source_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut out_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut width_arg as *mut i32 as *mut std::ffi::c_void,
                    &mut height_arg as *mut i32 as *mut std::ffi::c_void,
                    &mut rules_arg as *mut RulesAbi as *mut std::ffi::c_void,
                    &mut degenerate_area2 as *mut f32 as *mut std::ffi::c_void,
                    &mut edge_epsilon as *mut f32 as *mut std::ffi::c_void,
                    &mut selected_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut coverage_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut fragment_edge_arg as *mut i32 as *mut std::ffi::c_void,
                    &mut clamp_arg as *mut ClampAbi as *mut std::ffi::c_void,
                    &mut defines_arg as *mut DefinesAbi as *mut std::ffi::c_void,
                    &mut catmull_width as *mut f32 as *mut std::ffi::c_void,
                    &mut catmull_height as *mut f32 as *mut std::ffi::c_void,
                ];
                let emit_dispatch = buffers.launch(
                    self.fragment_emit,
                    (emit_grid, 1),
                    (BLOCK, 1),
                    &mut emit_params,
                )?;

                let mut pixels_out = vec![0u8; source.pixels.len()];
                let mut selected = vec![0u32; pixels];
                let mut coverage = vec![0u32; pixels];
                buffers.download(4, &mut pixels_out)?;
                buffers.download(5, bytes_mut(&mut selected))?;
                buffers.download(6, bytes_mut(&mut coverage))?;
                buffers.mark_consumed();
                let image = Image::new(source.width, source.height, pixels_out)
                    .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
                Ok(Output {
                    image,
                    selected,
                    coverage,
                    select_dispatch,
                    emit_dispatch,
                })
            },
        )
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
    if mesh
        .source_uv
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("source_uv"));
    }
    if mesh
        .ebo
        .iter()
        .any(|&index| index as usize >= mesh.dest.len())
    {
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

    fn ptx_param_count(name: &str) -> usize {
        let mut inside = false;
        let mut count = 0;
        for line in PTX.lines() {
            let line = line.trim_start();
            if line.starts_with(".entry ") || line.starts_with(".visible .entry ") {
                inside = line.contains(name);
                continue;
            }
            if !inside {
                continue;
            }
            if line.starts_with(".param ") {
                count += 1;
            } else if line.starts_with('{') {
                break;
            }
        }
        count
    }

    #[test]
    fn rule_block_matches_the_c_header() {
        assert_eq!(std::mem::size_of::<RulesAbi>(), 16);
        assert_eq!(std::mem::size_of::<ClampAbi>(), 12);
        assert_eq!(std::mem::size_of::<DefinesAbi>(), 16);
        let abi = RulesAbi::new(Rules {
            winding: Winding::NegativeAreaIsFront,
            cull: Cull::Front,
            edge: Edge::Inclusive,
            overlap: Overlap::Last,
        });
        assert_eq!(
            abi,
            RulesAbi {
                winding: 2,
                cull: 2,
                edge: 2,
                overlap: 3
            }
        );
    }

    #[test]
    fn all_kernel_entries_resolve_from_the_shipped_image() {
        let select = entry("mesh_raster_select").expect("select entry");
        let emit = entry("mesh_raster_emit").expect("emit entry");
        let fragment = entry("mesh_raster_fragment_emit").expect("fragment emit entry");
        assert_ne!(select, emit);
        assert_ne!(select, fragment);
        assert_ne!(emit, fragment);
        assert!(PTX.contains(".target sm_"));
    }

    #[test]
    fn shipped_ptx_parameter_counts_match_the_launchers() {
        assert_eq!(ptx_param_count("mesh_raster_select"), 10);
        assert_eq!(ptx_param_count("mesh_raster_emit"), 13);
        assert_eq!(ptx_param_count("mesh_raster_fragment_emit"), 17);
    }

    #[test]
    fn select_grid_expands_scanlines_and_refuses_overflow() {
        assert_eq!(select_grid(233, 512).unwrap(), 932);
        assert_eq!(select_grid(1, 64).unwrap(), 1);
        assert_eq!(select_grid(5_000, 2_048).unwrap(), 80_000);
        assert!(select_grid(usize::MAX, 2).is_err());
        let max_items = MAX_GRID_X * BLOCK as usize;
        assert_eq!(select_grid(max_items / 2, 2).unwrap(), i32::MAX as u32);
        assert!(select_grid(max_items / 2 + 1, 2).is_err());
    }

    #[test]
    fn malformed_meshes_are_refused_before_launch() {
        let source = Image::solid(2, 2, [0, 0, 0, 0]);
        let dest = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mesh = Mesh {
            dest: &dest,
            source_uv: &uv,
            ebo: &[0, 1, 2],
        };
        assert_eq!(validate(&source, &mesh), Ok(()));
        let short = Mesh {
            dest: &dest,
            source_uv: &uv,
            ebo: &[0, 1, 7],
        };
        assert_eq!(
            validate(&source, &short),
            Err(Refusal::ValueOutOfRange("ebo index"))
        );
        let ragged = Mesh {
            dest: &dest,
            source_uv: &uv,
            ebo: &[0, 1],
        };
        assert_eq!(
            validate(&source, &ragged),
            Err(Refusal::ShapeMismatch("ebo"))
        );
    }
}
