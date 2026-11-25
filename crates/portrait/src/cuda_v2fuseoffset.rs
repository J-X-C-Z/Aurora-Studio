//! PF8025/PF8026 V2FuseOffset -> V2Warp candidate chain.
//!
//! The stage consumes the same-job 255x171 `Grid_2005` RGBA8 texture, the
//! previous 280x280 encoded target, the 194 landmarks that identify the face
//! handoff, and the exact 526/2982 mesh consumed by V2Warp. Missing or mixed
//! job inputs are typed refusals; static fixture bytes are not live evidence.

use crate::cuda::{CudaError, Dispatch};
use crate::cuda_v2warp as v2warp;
use crate::cuda_v2warp::{Mesh, OffsetTexture};
#[cfg(windows)]
use crate::cuda_v2warp::V2Warp;
use crate::raster::Rules;
use crate::{Image, Refusal};

pub const OFFSET_WIDTH: u32 = 280;
pub const OFFSET_HEIGHT: u32 = 280;
pub const GRID_WIDTH: u32 = 255;
pub const GRID_HEIGHT: u32 = 171;
pub const RECT_X: u32 = 12;
pub const RECT_Y: u32 = 86;
pub const RECT_WIDTH: u32 = 255;
pub const RECT_HEIGHT: u32 = 171;
pub const POINT_COUNT: usize = 194;
pub const FUSE_KERNEL: &str = "v2fuseoffset_kernel";
pub const PATH: &str = "v2fuseoffset_to_forward_mesh_candidate";

const BLOCK: u32 = 128;
const PTX: &str = include_str!("../native/cuda/v2fuseoffset.ptx");
const INV_WIDTH: f32 = 1.0 / 280.0;
const INV_GRID_WIDTH: f32 = 1.0 / 255.0;
const INV_GRID_HEIGHT: f32 = 1.0 / 171.0;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Uniforms {
    left_degree: f32,
    right_degree: f32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 8);

#[derive(Clone, Copy)]
pub struct FuseInput<'a> {
    /// Nonzero identifier shared by every live input in this job.
    pub ticket: u64,
    /// Decoded Grid_2005 RGBA8 bytes, row-major.
    pub grid: &'a [u8],
    /// Previous encoded 280x280 target, or a zero-filled first target.
    pub base: &'a [[f32; 4]],
    pub left_degree: f32,
    pub right_degree: f32,
}

#[derive(Clone, Copy)]
pub struct PipelineInput<'a> {
    pub ticket: u64,
    pub source: &'a Image,
    pub grid: &'a [u8],
    pub base: &'a [[f32; 4]],
    pub left_degree: f32,
    pub right_degree: f32,
    pub points194: &'a [[f32; 2]],
    pub mesh: Mesh<'a>,
    pub rules: Rules,
    pub mat2: [f32; 4],
    /// Shader `F/O`; this adapter accepts only the source texture dimensions.
    pub f_o: [f32; 2],
}

pub struct FuseOutput {
    pub texture: Vec<[f32; 4]>,
    pub dispatch: Dispatch,
}

pub struct GpuOutput {
    pub texture: Vec<[f32; 4]>,
    pub fuse_dispatch: Dispatch,
    pub warp: v2warp::Output,
    pub path: &'static str,
}

pub struct CpuOutput {
    pub texture: Vec<[f32; 4]>,
    pub image: Image,
    pub path: &'static str,
}

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

fn grid(items: usize) -> (u32, u32) {
    ((items.div_ceil(BLOCK as usize) as u32).max(1), 1)
}

fn validate_common(
    ticket: u64,
    grid_pixels: &[u8],
    base: &[[f32; 4]],
    left_degree: f32,
    right_degree: f32,
) -> Result<(), Refusal> {
    if ticket == 0 {
        return Err(Refusal::MissingInput("same-job ticket"));
    }
    if grid_pixels.len() != (GRID_WIDTH * GRID_HEIGHT * 4) as usize {
        return Err(Refusal::ShapeMismatch("Grid_2005 RGBA8 255x171"));
    }
    if base.len() != (OFFSET_WIDTH * OFFSET_HEIGHT) as usize {
        return Err(Refusal::ShapeMismatch(
            "V2FuseOffset base 280x280 RGBA32F",
        ));
    }
    if base.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange(
            "V2FuseOffset base 280x280 RGBA32F",
        ));
    }
    if [left_degree, right_degree]
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.5).contains(value))
    {
        return Err(Refusal::ValueOutOfRange("u_degree2"));
    }
    Ok(())
}

pub fn validate(input: &FuseInput<'_>) -> Result<(), Refusal> {
    validate_common(
        input.ticket,
        input.grid,
        input.base,
        input.left_degree,
        input.right_degree,
    )
}

fn validate_pipeline(input: &PipelineInput<'_>) -> Result<(), Refusal> {
    validate_common(
        input.ticket,
        input.grid,
        input.base,
        input.left_degree,
        input.right_degree,
    )?;
    input.source.validate()?;
    if input.points194.len() != POINT_COUNT {
        return Err(Refusal::ShapeMismatch("points194"));
    }
    if input.points194.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("points194"));
    }
    if input.f_o != [input.source.width as f32, input.source.height as f32] {
        return Err(Refusal::ValueOutOfRange("V2Warp F/O != source size"));
    }
    if input.mat2.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("V2Warp mat2"));
    }
    v2warp::validate_mesh(input.mesh)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn sample_u8(
    source: &[u8],
    width: usize,
    height: usize,
    u: f32,
    v: f32,
) -> [f32; 4] {
    let w = width as f32;
    let h = height as f32;
    let x = (u * w - 0.5).clamp(0.0, w - 1.0);
    let y = (v * h - 0.5).clamp(0.0, h - 1.0);
    let x0 = x.floor();
    let y0 = y.floor();
    let x1 = (x0 + 1.0).min(w - 1.0);
    let y1 = (y0 + 1.0).min(h - 1.0);
    let fx = x - x0;
    let fy = y - y0;
    let x0 = x0 as usize;
    let y0 = y0 as usize;
    let x1 = x1 as usize;
    let y1 = y1 as usize;
    let texel = |x: usize, y: usize, channel: usize| {
        source[(y * width + x) * 4 + channel] as f32 / 255.0
    };
    let mut out = [0.0f32; 4];
    for channel in 0..4 {
        let top = lerp(texel(x0, y0, channel), texel(x1, y0, channel), fx);
        let bottom = lerp(texel(x0, y1, channel), texel(x1, y1, channel), fx);
        out[channel] = lerp(top, bottom, fy);
    }
    out
}

fn sample_f32(source: &[[f32; 4]], u: f32, v: f32) -> [f32; 4] {
    let w = OFFSET_WIDTH as f32;
    let h = OFFSET_HEIGHT as f32;
    let x = (u * w - 0.5).clamp(0.0, w - 1.0);
    let y = (v * h - 0.5).clamp(0.0, h - 1.0);
    let x0 = x.floor();
    let y0 = y.floor();
    let x1 = (x0 + 1.0).min(w - 1.0);
    let y1 = (y0 + 1.0).min(h - 1.0);
    let fx = x - x0;
    let fy = y - y0;
    let x0 = x0 as usize;
    let y0 = y0 as usize;
    let x1 = x1 as usize;
    let y1 = y1 as usize;
    let mut out = [0.0f32; 4];
    for channel in 0..4 {
        let top = lerp(
            source[y0 * OFFSET_WIDTH as usize + x0][channel],
            source[y0 * OFFSET_WIDTH as usize + x1][channel],
            fx,
        );
        let bottom = lerp(
            source[y1 * OFFSET_WIDTH as usize + x0][channel],
            source[y1 * OFFSET_WIDTH as usize + x1][channel],
            fx,
        );
        out[channel] = lerp(top, bottom, fy);
    }
    out
}

pub fn fuse_cpu(input: &FuseInput<'_>) -> Result<Vec<[f32; 4]>, Refusal> {
    validate(input)?;
    let mut out = input.base.to_vec();
    for py in RECT_Y..RECT_Y + RECT_HEIGHT {
        for px in RECT_X..RECT_X + RECT_WIDTH {
            let target_u = (px as f32 + 0.5) * INV_WIDTH;
            let target_v = (py as f32 + 0.5) * INV_WIDTH;
            let grid_u = ((px - RECT_X) as f32 + 0.5) * INV_GRID_WIDTH;
            let grid_v = ((py - RECT_Y) as f32 + 0.5) * INV_GRID_HEIGHT;
            let base = sample_f32(input.base, target_u, target_v);
            let atlas = sample_u8(
                input.grid,
                GRID_WIDTH as usize,
                GRID_HEIGHT as usize,
                grid_u,
                grid_v,
            );
            let factor = if grid_u >= 0.5 {
                input.right_degree
            } else {
                input.left_degree
            };
            let dx = (base[0] - base[1]) * 0.25 + (atlas[0] - atlas[1]) * 0.25 * factor;
            let dy = (base[2] - base[3]) * 0.25 + (atlas[2] - atlas[3]) * 0.25 * factor;
            out[(py * OFFSET_WIDTH + px) as usize] = [
                dx.max(0.0) * 4.0,
                (-dx).max(0.0) * 4.0,
                dy.max(0.0) * 4.0,
                (-dy).max(0.0) * 4.0,
            ];
        }
    }
    Ok(out)
}

pub fn apply_cpu(input: &PipelineInput<'_>) -> Result<CpuOutput, Refusal> {
    validate_pipeline(input)?;
    let texture = fuse_cpu(&FuseInput {
        ticket: input.ticket,
        grid: input.grid,
        base: input.base,
        left_degree: input.left_degree,
        right_degree: input.right_degree,
    })?;
    let image = v2warp::apply_cpu(
        input.source,
        OffsetTexture {
            width: OFFSET_WIDTH,
            height: OFFSET_HEIGHT,
            pixels: &texture,
        },
        input.mesh,
        input.rules,
        input.mat2,
        input.f_o,
    )?;
    Ok(CpuOutput {
        texture,
        image,
        path: PATH,
    })
}

#[cfg(windows)]
pub struct FuseOffset {
    gpu: crate::cuda::Nvidia,
    kernel: &'static str,
}

#[cfg(windows)]
impl FuseOffset {
    pub fn open() -> Result<Self, CudaError> {
        let kernel =
            entry("v2fuseoffset_kernel").ok_or(CudaError::Unavailable(FUSE_KERNEL))?;
        Ok(Self {
            gpu: crate::cuda::Nvidia::open_ptx(PTX)?,
            kernel,
        })
    }

    pub fn render(&self, input: &FuseInput<'_>) -> Result<FuseOutput, CudaError> {
        validate(input)?;
        let base_buf = self.gpu.alloc_buffer(std::mem::size_of_val(input.base))?;
        let grid_buf = self.gpu.alloc_buffer(input.grid.len())?;
        let out_buf = self
            .gpu
            .alloc_buffer((OFFSET_WIDTH * OFFSET_HEIGHT) as usize * 16)?;
        self.gpu.upload_buffer(&base_buf, bytes_of(input.base))?;
        self.gpu.upload_buffer(&grid_buf, input.grid)?;
        self.gpu.upload_buffer(&out_buf, bytes_of(input.base))?;

        let mut base_arg = base_buf.handle();
        let mut grid_arg = grid_buf.handle();
        let mut out_arg = out_buf.handle();
        let mut uniforms = Uniforms {
            left_degree: input.left_degree,
            right_degree: input.right_degree,
        };
        let mut params: [*mut std::ffi::c_void; 4] = [
            &mut base_arg as *mut u64 as *mut std::ffi::c_void,
            &mut grid_arg as *mut u64 as *mut std::ffi::c_void,
            &mut out_arg as *mut u64 as *mut std::ffi::c_void,
            &mut uniforms as *mut Uniforms as *mut std::ffi::c_void,
        ];
        let dispatch = self.gpu.launch_raw(
            self.kernel,
            grid((OFFSET_WIDTH * OFFSET_HEIGHT) as usize),
            (BLOCK, 1),
            &mut params,
        )?;
        let mut texture = vec![[0.0f32; 4]; (OFFSET_WIDTH * OFFSET_HEIGHT) as usize];
        self.gpu.download_buffer(&out_buf, bytes_mut(&mut texture))?;
        Ok(FuseOutput { texture, dispatch })
    }
}

#[cfg(windows)]
pub fn apply_cuda(input: &PipelineInput<'_>) -> Result<GpuOutput, CudaError> {
    validate_pipeline(input)?;
    let fuse = {
        let gpu = FuseOffset::open()?;
        gpu.render(&FuseInput {
            ticket: input.ticket,
            grid: input.grid,
            base: input.base,
            left_degree: input.left_degree,
            right_degree: input.right_degree,
        })?
    };
    let warp = V2Warp::open()?;
    let warp = warp.render(
        input.source,
        OffsetTexture {
            width: OFFSET_WIDTH,
            height: OFFSET_HEIGHT,
            pixels: &fuse.texture,
        },
        input.mesh,
        input.rules,
        input.mat2,
        input.f_o,
    )?;
    Ok(GpuOutput {
        texture: fuse.texture,
        fuse_dispatch: fuse.dispatch,
        warp,
        path: PATH,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_v2warp::{INDEX_COUNT, VERTEX_COUNT};
    use crate::raster::{Cull, Edge, Overlap, Winding};

    const RULES: Rules = Rules {
        winding: Winding::NegativeAreaIsFront,
        cull: Cull::None,
        edge: Edge::Inclusive,
        overlap: Overlap::Last,
    };

    fn grid() -> Vec<u8> {
        let mut out = Vec::with_capacity((GRID_WIDTH * GRID_HEIGHT * 4) as usize);
        for y in 0..GRID_HEIGHT {
            for x in 0..GRID_WIDTH {
                out.extend_from_slice(&[
                    ((x + y * 3) % 31) as u8,
                    ((x * 2 + y) % 29) as u8,
                    ((x + y * 5) % 11) as u8,
                    0,
                ]);
            }
        }
        out
    }

    fn base() -> Vec<[f32; 4]> {
        (0..(OFFSET_WIDTH * OFFSET_HEIGHT) as usize)
            .map(|index| {
                let x = index % OFFSET_WIDTH as usize;
                let y = index / OFFSET_WIDTH as usize;
                [
                    ((x + y) % 7) as f32,
                    (x % 5) as f32,
                    ((x * 2 + y) % 9) as f32,
                    (y % 3) as f32,
                ]
            })
            .collect()
    }

    fn read_pairs(path: &std::path::Path) -> Vec<[f32; 2]> {
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(bytes.len() % 8, 0);
        bytes
            .chunks_exact(8)
            .map(|chunk| {
                [
                    f32::from_le_bytes(chunk[0..4].try_into().unwrap()),
                    f32::from_le_bytes(chunk[4..8].try_into().unwrap()),
                ]
            })
            .collect()
    }

    fn fixture_grid_base() -> Option<(Vec<u8>, Vec<[f32; 4]>)> {
        let root = std::env::var_os("V2FUSE_FIXTURE_ROOT")?;
        let root = std::path::PathBuf::from(root);
        let grid = std::fs::read(root.join("Grid_2005.rgba8")).unwrap();
        let base_bytes = std::fs::read(root.join("v2fuseoffset.base.280x280.rgba32f")).unwrap();
        assert_eq!(base_bytes.len() % 16, 0);
        let base = base_bytes
            .chunks_exact(16)
            .map(|chunk| {
                [
                    f32::from_le_bytes(chunk[0..4].try_into().unwrap()),
                    f32::from_le_bytes(chunk[4..8].try_into().unwrap()),
                    f32::from_le_bytes(chunk[8..12].try_into().unwrap()),
                    f32::from_le_bytes(chunk[12..16].try_into().unwrap()),
                ]
            })
            .collect();
        Some((grid, base))
    }

    fn write_texture(path: &std::path::Path, texture: &[[f32; 4]]) {
        let bytes: Vec<u8> = texture
            .iter()
            .flatten()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(path, bytes).unwrap();
    }

    fn source() -> Image {
        let mut pixels = Vec::with_capacity(17 * 13 * 4);
        for y in 0..13u32 {
            for x in 0..17u32 {
                pixels.extend_from_slice(&[
                    (x * 11 + y * 3) as u8,
                    (y * 13 + x * 2) as u8,
                    (x * 5 + y * 7) as u8,
                    255,
                ]);
            }
        }
        Image::new(17, 13, pixels).unwrap()
    }

    fn mesh() -> (
        Vec<[f32; 2]>,
        Vec<[f32; 2]>,
        Vec<[f32; 2]>,
        Vec<[f32; 2]>,
        Vec<u16>,
    ) {
        let mut positions = vec![[0.0; 2]; VERTEX_COUNT];
        let mut texcoord = vec![[0.0; 2]; VERTEX_COUNT];
        let offset_texcoord = vec![[0.5; 2]; VERTEX_COUNT];
        let face_scales = vec![[1.0; 2]; VERTEX_COUNT];
        positions[0] = [0.0, 0.0];
        positions[1] = [17.0, 0.0];
        positions[2] = [0.0, 13.0];
        texcoord[0] = [0.0, 0.0];
        texcoord[1] = [1.0, 0.0];
        texcoord[2] = [0.0, 1.0];
        let mut ebo = vec![0u16; INDEX_COUNT];
        ebo[0] = 0;
        ebo[1] = 1;
        ebo[2] = 2;
        (
            positions,
            texcoord,
            offset_texcoord,
            face_scales,
            ebo,
        )
    }

    fn pipeline_parts() -> (
        Image,
        Vec<u8>,
        Vec<[f32; 4]>,
        Vec<[f32; 2]>,
        Vec<[f32; 2]>,
        Vec<[f32; 2]>,
        Vec<[f32; 2]>,
        Vec<[f32; 2]>,
        Vec<u16>,
    ) {
        let (p, t, o, s, e) = mesh();
        (
            source(),
            grid(),
            base(),
            vec![[0.25, 0.75]; POINT_COUNT],
            p,
            t,
            o,
            s,
            e,
        )
    }

    #[test]
    fn exact_same_job_and_mesh_contracts_are_required() {
        let (source, grid, base, points, p, t, o, s, e) = pipeline_parts();
        let mesh = Mesh {
            framebuffer_positions_w1: &p,
            texcoord: &t,
            offset_texcoord: &o,
            face_scales: &s,
            ebo: &e,
        };
        let mut input = PipelineInput {
            ticket: 7,
            source: &source,
            grid: &grid,
            base: &base,
            left_degree: 1.2,
            right_degree: 0.6,
            points194: &points,
            mesh,
            rules: RULES,
            mat2: [1.0, 0.0, 0.0, 1.0],
            f_o: [17.0, 13.0],
        };
        assert_eq!(validate_pipeline(&input), Ok(()));

        input.ticket = 0;
        assert_eq!(
            validate_pipeline(&input),
            Err(Refusal::MissingInput("same-job ticket"))
        );
        input.ticket = 7;

        input.points194 = &points[..193];
        assert_eq!(
            validate_pipeline(&input),
            Err(Refusal::ShapeMismatch("points194"))
        );
        input.points194 = &points;

        input.grid = &grid[..grid.len() - 1];
        assert_eq!(
            validate_pipeline(&input),
            Err(Refusal::ShapeMismatch("Grid_2005 RGBA8 255x171"))
        );
        input.grid = &grid;

        input.base = &base[..base.len() - 1];
        assert_eq!(
            validate_pipeline(&input),
            Err(Refusal::ShapeMismatch(
                "V2FuseOffset base 280x280 RGBA32F"
            ))
        );
        input.base = &base;

        input.f_o = [16.0, 13.0];
        assert_eq!(
            validate_pipeline(&input),
            Err(Refusal::ValueOutOfRange("V2Warp F/O != source size"))
        );
        input.f_o = [17.0, 13.0];

        input.mesh.ebo = &e[..INDEX_COUNT - 1];
        assert_eq!(
            validate_pipeline(&input),
            Err(Refusal::ShapeMismatch("topology2982"))
        );
    }

    #[test]
    fn cpu_preserves_base_outside_rectangle_and_encodes_inside() {
        let grid = grid();
        let base = vec![[0.25, 0.125, 0.5, 0.25]; (OFFSET_WIDTH * OFFSET_HEIGHT) as usize];
        let out = fuse_cpu(&FuseInput {
            ticket: 1,
            grid: &grid,
            base: &base,
            left_degree: 1.2,
            right_degree: 0.6,
        })
        .unwrap();
        assert_eq!(out[0], base[0]);
        let inside = ((RECT_Y + 10) * OFFSET_WIDTH + RECT_X + 10) as usize;
        assert_ne!(out[inside], base[inside]);
        assert!(out[inside].iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn cpu_full_chain_consumes_fuse_output() {
        let (source, grid, base, points, p, t, o, s, e) = pipeline_parts();
        let input = PipelineInput {
            ticket: 1,
            source: &source,
            grid: &grid,
            base: &base,
            left_degree: 1.2,
            right_degree: 0.6,
            points194: &points,
            mesh: Mesh {
                framebuffer_positions_w1: &p,
                texcoord: &t,
                offset_texcoord: &o,
                face_scales: &s,
                ebo: &e,
            },
            rules: RULES,
            mat2: [1.0, 0.0, 0.0, 1.0],
            f_o: [17.0, 13.0],
        };
        let out = apply_cpu(&input).unwrap();
        assert_eq!(out.texture.len(), (OFFSET_WIDTH * OFFSET_HEIGHT) as usize);
        assert_eq!(out.image.width, 17);
        assert_eq!(out.image.height, 13);
        assert_eq!(out.path, PATH);
    }

    #[test]
    fn archived_grid_and_mesh_fixture_contract() {
        let Some(root) = std::env::var_os("V2FUSE_FIXTURE_ROOT") else {
            eprintln!("SKIP_NO_V2FUSE_FIXTURE_ROOT");
            return;
        };
        let root = std::path::PathBuf::from(root);
        let points = read_pairs(&root.join("photo-a_face0.points194.f32le"));
        let positions = read_pairs(&root.join("photo-a_face0.positions526.f32le"));
        let texcoord = read_pairs(&root.join("photo-a_face0.texcoord526.f32le"));
        let offset_texcoord = read_pairs(&root.join("photo-a_face0.offset_texcoord526.f32le"));
        let face_scales = read_pairs(&root.join("photo-a_face0.face_scales526.f32le"));
        let ebo_bytes = std::fs::read(root.join("pf8025.topology2982.u16le")).unwrap();
        let ebo: Vec<u16> = ebo_bytes
            .chunks_exact(2)
            .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
            .collect();
        assert_eq!(points.len(), POINT_COUNT);
        assert_eq!(positions.len(), VERTEX_COUNT);
        assert_eq!(texcoord.len(), VERTEX_COUNT);
        assert_eq!(offset_texcoord.len(), VERTEX_COUNT);
        assert_eq!(face_scales.len(), VERTEX_COUNT);
        assert_eq!(ebo.len(), INDEX_COUNT);
        crate::cuda_v2warp::validate_mesh(Mesh {
            framebuffer_positions_w1: &positions,
            texcoord: &texcoord,
            offset_texcoord: &offset_texcoord,
            face_scales: &face_scales,
            ebo: &ebo,
        })
        .unwrap();

        let (grid, base) = fixture_grid_base().unwrap();
        let texture = fuse_cpu(&FuseInput {
            ticket: 1,
            grid: &grid,
            base: &base,
            left_degree: 1.2,
            right_degree: 0.6,
        })
        .unwrap();
        assert_eq!(texture[0], base[0]);
        let inside = ((RECT_Y + 10) * OFFSET_WIDTH + RECT_X + 10) as usize;
        assert_ne!(texture[inside], base[inside]);
        if let Some(path) = std::env::var_os("V2FUSE_CPU_FIXTURE_OUT") {
            write_texture(std::path::Path::new(&path), &texture);
        }
    }

    #[cfg(windows)]
    #[test]
    fn gpu_fuse_matches_cpu_on_full_280_target() {
        let gpu = match FuseOffset::open() {
            Ok(gpu) => gpu,
            Err(CudaError::Unavailable(reason)) => {
                eprintln!("SKIP_NO_NVIDIA_DRIVER: {reason}");
                assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "CUDA required but unavailable: {reason}"
                );
                return;
            }
            Err(broken) => panic!("NVIDIA path present but broken: {broken:?}"),
        };
        let (grid, base) = fixture_grid_base().unwrap_or_else(|| (grid(), base()));
        let input = FuseInput {
            ticket: 1,
            grid: &grid,
            base: &base,
            left_degree: 1.2,
            right_degree: 0.6,
        };
        let expected = fuse_cpu(&input).unwrap();
        let actual = gpu.render(&input).expect("real GPU dispatch");
        let max_diff = expected
            .iter()
            .flatten()
            .zip(actual.texture.iter().flatten())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "V2FUSE_GPU device={} kernel={} grid={}x{} block={}x{} max_diff={}",
            actual.dispatch.device,
            actual.dispatch.kernel,
            actual.dispatch.grid.0,
            actual.dispatch.grid.1,
            actual.dispatch.block.0,
            actual.dispatch.block.1,
            max_diff
        );
        assert_eq!(actual.dispatch.kernel, FUSE_KERNEL);
        assert!(max_diff <= 1e-6, "GPU/CPU max diff {max_diff}");
        if let Some(path) = std::env::var_os("V2FUSE_GPU_FIXTURE_OUT") {
            write_texture(std::path::Path::new(&path), &actual.texture);
        }
    }

    #[cfg(windows)]
    #[test]
    fn gpu_full_chain_reaches_v2warp() {
        let (source, grid, base, points, p, t, o, s, e) = pipeline_parts();
        let input = PipelineInput {
            ticket: 1,
            source: &source,
            grid: &grid,
            base: &base,
            left_degree: 1.2,
            right_degree: 0.6,
            points194: &points,
            mesh: Mesh {
                framebuffer_positions_w1: &p,
                texcoord: &t,
                offset_texcoord: &o,
                face_scales: &s,
                ebo: &e,
            },
            rules: RULES,
            mat2: [1.0, 0.0, 0.0, 1.0],
            f_o: [17.0, 13.0],
        };
        let expected = apply_cpu(&input).unwrap();
        let actual = apply_cuda(&input).expect("real GPU chain");
        let max_diff = expected
            .image
            .pixels
            .iter()
            .zip(&actual.warp.image.pixels)
            .map(|(left, right)| left.abs_diff(*right))
            .max()
            .unwrap();
        let covered = actual
            .warp
            .coverage
            .iter()
            .filter(|value| **value > 0)
            .count();
        eprintln!(
            "V2FUSE_WARP_GPU device={} fuse={} select={} eval={} covered={} max_diff={}",
            actual.fuse_dispatch.device,
            actual.fuse_dispatch.kernel,
            actual.warp.select_dispatch.kernel,
            actual.warp.eval_dispatch.kernel,
            covered,
            max_diff
        );
        assert!(covered > 0);
        assert!(max_diff <= 1, "GPU/CPU max diff {max_diff}");
        assert_eq!(actual.path, PATH);
    }
}
