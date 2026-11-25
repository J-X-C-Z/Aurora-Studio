//! PF8025/PF8026 V2Warp forward-mesh candidate.
//!
//! The pinned V2 matrix reduces `a_position` to framebuffer pixels with `w=1`.
//! This adapter keeps the forward topology: one thread per triangle selects
//! coverage/winding/overlap, then one thread per covered pixel evaluates the
//! recovered fragment order from four independent 526-vertex attributes.
//!
//! It consumes `texture2D` samples from the encoded target produced by
//! [`crate::cuda_v2fuseoffset`]. The f32 exchange buffer is not a claim that
//! the vendor target is RGBA32F; effective storage precision remains live
//! state. This is not a vendor-parity claim.

use crate::cuda::{CudaError, Dispatch};
use crate::raster::{Cull, Edge, Overlap, Rules, Winding};
use crate::{Image, Refusal};
use std::sync::{Arc, Mutex, OnceLock};

pub const VERTEX_COUNT: usize = 526;
pub const INDEX_COUNT: usize = 2982;
pub const TRIANGLE_COUNT: usize = 994;
pub const OFFSET_WIDTH: u32 = 280;
pub const OFFSET_HEIGHT: u32 = 280;
pub const SELECT_KERNEL: &str = "v2warp_select_kernel";
pub const EVAL_KERNEL: &str = "v2warp_eval_kernel";
pub const PATH: &str = "forward_mesh_candidate";

const BLOCK: u32 = 128;
const PTX: &str = include_str!("../native/cuda/v2warp.ptx");

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
pub struct Uniforms {
    pub width: i32,
    pub height: i32,
    pub offset_width: i32,
    pub offset_height: i32,
    pub triangle_count: i32,
    pub overlap: i32,
    pub edge: i32,
    /// GL upload order after `shufps D8`: raw0, raw2, raw1, raw3.
    pub mat2: [f32; 4],
    /// Shader `F/O`; the adapter requires it to equal the source texture size.
    pub filter_size: [f32; 2],
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 52);

impl Uniforms {
    pub fn new(
        source: &Image,
        offset: OffsetTexture<'_>,
        rules: Rules,
        mat2: [f32; 4],
        filter_size: [f32; 2],
    ) -> Self {
        Self {
            width: source.width as i32,
            height: source.height as i32,
            offset_width: offset.width as i32,
            offset_height: offset.height as i32,
            triangle_count: TRIANGLE_COUNT as i32,
            overlap: match rules.overlap {
                Overlap::Reject => 1,
                Overlap::First => 2,
                Overlap::Last => 3,
            },
            edge: match rules.edge {
                Edge::Exclusive => 1,
                Edge::Inclusive => 2,
            },
            mat2,
            filter_size,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct OffsetTexture<'a> {
    pub width: u32,
    pub height: u32,
    /// `texture2D` samples represented as f32. The source texture's effective
    /// internal format and component precision remain caller/live state.
    pub pixels: &'a [[f32; 4]],
}

#[derive(Clone, Copy, Debug)]
pub struct Mesh<'a> {
    /// V2 `a_position`, projected with the pinned `w=1` matrix to framebuffer pixels.
    pub framebuffer_positions_w1: &'a [[f32; 2]],
    /// V2 `a_texcoord`, per-face source UV.
    pub texcoord: &'a [[f32; 2]],
    /// V2 `a_offset_texcoord`, the neutral offset-texture UV.
    pub offset_texcoord: &'a [[f32; 2]],
    /// V2 `a_face_scales`, componentwise displacement scale.
    pub face_scales: &'a [[f32; 2]],
    /// 2982 uint16 indices = 994 triangles.
    pub ebo: &'a [u16],
}

pub struct Output {
    pub image: Image,
    pub selected: Vec<u32>,
    pub coverage: Vec<u32>,
    pub select_dispatch: Dispatch,
    pub eval_dispatch: Dispatch,
    /// This candidate never reports the vendor GL path as executed.
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

pub(crate) fn validate_mesh(mesh: Mesh<'_>) -> Result<(), Refusal> {
    for (name, len) in [
        ("a_position_w1", mesh.framebuffer_positions_w1.len()),
        ("a_texcoord", mesh.texcoord.len()),
        ("a_offset_texcoord", mesh.offset_texcoord.len()),
        ("a_face_scales", mesh.face_scales.len()),
    ] {
        if len != VERTEX_COUNT {
            return Err(Refusal::ShapeMismatch(name));
        }
    }
    if mesh.ebo.len() != INDEX_COUNT {
        return Err(Refusal::ShapeMismatch("topology2982"));
    }
    if mesh.ebo.iter().any(|&index| index as usize >= VERTEX_COUNT) {
        return Err(Refusal::ValueOutOfRange("topology2982 index"));
    }
    if mesh
        .framebuffer_positions_w1
        .iter()
        .chain(mesh.texcoord)
        .chain(mesh.offset_texcoord)
        .chain(mesh.face_scales)
        .flat_map(|value| value.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("V2 mesh attribute"));
    }
    Ok(())
}

fn validate(
    source: &Image,
    offset: OffsetTexture<'_>,
    mesh: Mesh<'_>,
    mat2: [f32; 4],
    filter_size: [f32; 2],
) -> Result<(), Refusal> {
    source.validate()?;
    let offset_len = (offset.width as usize)
        .checked_mul(offset.height as usize)
        .ok_or(Refusal::ShapeMismatch("V2 offset RGBA32F 280x280"))?;
    if offset.width != OFFSET_WIDTH
        || offset.height != OFFSET_HEIGHT
        || offset_len != offset.pixels.len()
    {
        return Err(Refusal::ShapeMismatch("V2 offset RGBA32F 280x280"));
    }
    validate_mesh(mesh)?;
    if mat2
        .iter()
        .chain(filter_size.iter())
        .any(|value| !value.is_finite())
        || filter_size.iter().any(|value| *value <= 0.0)
    {
        return Err(Refusal::ValueOutOfRange("V2Warp uniforms"));
    }
    if filter_size != [source.width as f32, source.height as f32] {
        return Err(Refusal::ValueOutOfRange("V2Warp F/O != source size"));
    }
    if offset
        .pixels
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("V2 offset RGBA32F"));
    }
    if source.pixel_count() > i32::MAX as usize || offset_len > i32::MAX as usize {
        return Err(Refusal::ValueOutOfRange("V2Warp size"));
    }
    Ok(())
}

#[cfg(windows)]
pub struct V2Warp {
    gpu: crate::cuda::Nvidia,
    select: &'static str,
    eval: &'static str,
}

#[cfg(windows)]
impl V2Warp {
    pub fn open() -> Result<Self, CudaError> {
        let select =
            entry("v2warp_select").ok_or(CudaError::Unavailable("v2warp_select_kernel"))?;
        let eval = entry("v2warp_eval").ok_or(CudaError::Unavailable("v2warp_eval_kernel"))?;
        Ok(Self {
            gpu: crate::cuda::Nvidia::open_ptx(PTX)?,
            select,
            eval,
        })
    }

    pub fn render(
        &self,
        source: &Image,
        offset: OffsetTexture<'_>,
        mesh: Mesh<'_>,
        rules: Rules,
        mat2: [f32; 4],
        filter_size: [f32; 2],
    ) -> Result<Output, CudaError> {
        validate(source, offset, mesh, mat2, filter_size)?;
        let width = source.width as usize;
        let height = source.height as usize;
        let pixels = width * height;
        let rules_abi = RulesAbi::new(rules);
        let uniforms = Uniforms::new(source, offset, rules, mat2, filter_size);

        let source_buf = self.gpu.alloc_buffer(source.pixels.len())?;
        let offset_buf = self
            .gpu
            .alloc_buffer(std::mem::size_of_val(offset.pixels))?;
        let out_buf = self.gpu.alloc_buffer(source.pixels.len())?;
        let positions_buf = self
            .gpu
            .alloc_buffer(std::mem::size_of_val(mesh.framebuffer_positions_w1))?;
        let texcoord_buf = self
            .gpu
            .alloc_buffer(std::mem::size_of_val(mesh.texcoord))?;
        let offset_uv_buf = self
            .gpu
            .alloc_buffer(std::mem::size_of_val(mesh.offset_texcoord))?;
        let scales_buf = self
            .gpu
            .alloc_buffer(std::mem::size_of_val(mesh.face_scales))?;
        let ebo_buf = self.gpu.alloc_buffer(std::mem::size_of_val(mesh.ebo))?;
        let selected_buf = self.gpu.alloc_buffer(pixels * 4)?;
        let coverage_buf = self.gpu.alloc_buffer(pixels * 4)?;

        self.gpu.upload_buffer(&source_buf, &source.pixels)?;
        self.gpu
            .upload_buffer(&offset_buf, bytes_of(offset.pixels))?;
        self.gpu.upload_buffer(&out_buf, &source.pixels)?;
        self.gpu
            .upload_buffer(&positions_buf, bytes_of(mesh.framebuffer_positions_w1))?;
        self.gpu
            .upload_buffer(&texcoord_buf, bytes_of(mesh.texcoord))?;
        self.gpu
            .upload_buffer(&offset_uv_buf, bytes_of(mesh.offset_texcoord))?;
        self.gpu
            .upload_buffer(&scales_buf, bytes_of(mesh.face_scales))?;
        self.gpu.upload_buffer(&ebo_buf, bytes_of(mesh.ebo))?;
        let selected_fill = if rules.overlap == Overlap::Last {
            0u8
        } else {
            0xFFu8
        };
        self.gpu
            .upload_buffer(&selected_buf, &vec![selected_fill; pixels * 4])?;
        self.gpu
            .upload_buffer(&coverage_buf, &vec![0u8; pixels * 4])?;

        let mut width_arg = source.width as i32;
        let mut height_arg = source.height as i32;
        let mut triangles_arg = TRIANGLE_COUNT as i32;
        let mut rules_arg = rules_abi;
        let mut positions_arg = positions_buf.handle();
        let mut ebo_arg = ebo_buf.handle();
        let mut selected_arg = selected_buf.handle();
        let mut coverage_arg = coverage_buf.handle();
        let mut select_params: [*mut std::ffi::c_void; 8] = [
            &mut positions_arg as *mut u64 as *mut std::ffi::c_void,
            &mut ebo_arg as *mut u64 as *mut std::ffi::c_void,
            &mut width_arg as *mut i32 as *mut std::ffi::c_void,
            &mut height_arg as *mut i32 as *mut std::ffi::c_void,
            &mut triangles_arg as *mut i32 as *mut std::ffi::c_void,
            &mut rules_arg as *mut RulesAbi as *mut std::ffi::c_void,
            &mut selected_arg as *mut u64 as *mut std::ffi::c_void,
            &mut coverage_arg as *mut u64 as *mut std::ffi::c_void,
        ];
        let select_dispatch = self.gpu.launch_raw(
            self.select,
            grid(TRIANGLE_COUNT),
            (BLOCK, 1),
            &mut select_params,
        )?;

        let mut source_arg = source_buf.handle();
        let mut offset_arg = offset_buf.handle();
        let mut out_arg = out_buf.handle();
        let mut texcoord_arg = texcoord_buf.handle();
        let mut offset_uv_arg = offset_uv_buf.handle();
        let mut scales_arg = scales_buf.handle();
        let mut uniforms_arg = uniforms;
        let mut eval_params: [*mut std::ffi::c_void; 11] = [
            &mut source_arg as *mut u64 as *mut std::ffi::c_void,
            &mut offset_arg as *mut u64 as *mut std::ffi::c_void,
            &mut out_arg as *mut u64 as *mut std::ffi::c_void,
            &mut positions_arg as *mut u64 as *mut std::ffi::c_void,
            &mut texcoord_arg as *mut u64 as *mut std::ffi::c_void,
            &mut offset_uv_arg as *mut u64 as *mut std::ffi::c_void,
            &mut scales_arg as *mut u64 as *mut std::ffi::c_void,
            &mut ebo_arg as *mut u64 as *mut std::ffi::c_void,
            &mut selected_arg as *mut u64 as *mut std::ffi::c_void,
            &mut coverage_arg as *mut u64 as *mut std::ffi::c_void,
            &mut uniforms_arg as *mut Uniforms as *mut std::ffi::c_void,
        ];
        let eval_dispatch =
            self.gpu
                .launch_raw(self.eval, grid(pixels), (BLOCK, 1), &mut eval_params)?;

        let mut pixels_out = vec![0u8; source.pixels.len()];
        let mut selected = vec![0u32; pixels];
        let mut coverage = vec![0u32; pixels];
        self.gpu.download_buffer(&out_buf, &mut pixels_out)?;
        self.gpu
            .download_buffer(&selected_buf, bytes_mut(&mut selected))?;
        self.gpu
            .download_buffer(&coverage_buf, bytes_mut(&mut coverage))?;
        let image = Image::new(source.width, source.height, pixels_out)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("V2Warp output")))?;
        Ok(Output {
            image,
            selected,
            coverage,
            select_dispatch,
            eval_dispatch,
            path: PATH,
        })
    }
}

/// CPU twin of the recovered forward-mesh stage.
pub fn apply_cpu(
    source: &Image,
    offset: OffsetTexture<'_>,
    mesh: Mesh<'_>,
    rules: Rules,
    mat2: [f32; 4],
    filter_size: [f32; 2],
) -> Result<Image, Refusal> {
    validate(source, offset, mesh, mat2, filter_size)?;
    let width = source.width as usize;
    let height = source.height as usize;
    let (selected, _coverage, covered) = selection_pair(width, height, &mesh, rules);

    let mut pixels = source.pixels.clone();
    let workers = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .clamp(1, 8);
    let chunk = covered.len().div_ceil(workers).max(1);
    let pixels_ptr = pixels.as_mut_ptr() as usize;
    std::thread::scope(|scope| {
        for chunk in covered.chunks(chunk) {
            let ptr = pixels_ptr;
            let selected = Arc::clone(&selected);
            scope.spawn(move || {
                for &pixel in chunk {
                let pixel = pixel as usize;
                let py = pixel / width;
                let px = pixel % width;
                let triangle = selected[pixel] as usize;
                let i = [
                    mesh.ebo[triangle * 3] as usize,
                    mesh.ebo[triangle * 3 + 1] as usize,
                    mesh.ebo[triangle * 3 + 2] as usize,
                ];
                let p = [
                    mesh.framebuffer_positions_w1[i[0]],
                    mesh.framebuffer_positions_w1[i[1]],
                    mesh.framebuffer_positions_w1[i[2]],
                ];
                let area2 = cross(sub(p[1], p[0]), sub(p[2], p[0]));
                let Some(b) = weights(p, area2, px as f32 + 0.5, py as f32 + 0.5, rules.edge)
                else {
                    continue;
                };
                let uv = bary(
                    mesh.texcoord[i[0]],
                    mesh.texcoord[i[1]],
                    mesh.texcoord[i[2]],
                    b,
                );
                let offset_uv = bary(
                    mesh.offset_texcoord[i[0]],
                    mesh.offset_texcoord[i[1]],
                    mesh.offset_texcoord[i[2]],
                    b,
                );
                let scales = bary(
                    mesh.face_scales[i[0]],
                    mesh.face_scales[i[1]],
                    mesh.face_scales[i[2]],
                    b,
                );
                let mut d = sample_offset(offset, offset_uv);
                d[0] *= scales[0];
                d[1] *= scales[1];
                let input = d;
                d[0] = input[0] * mat2[0] + input[1] * mat2[2];
                d[1] = input[0] * mat2[1] + input[1] * mat2[3];
                let edge = edge(uv);
                let sample_uv = [uv[0] + d[0] * edge, uv[1] + d[1] * edge];
                let color = crate::raster_fragment::sample_catmull_rom(
                    source,
                    sample_uv[0],
                    sample_uv[1],
                    (filter_size[0], filter_size[1]),
                );
                let stored = crate::raster_fragment::to_rgba8(color);
                // SAFETY: `covered` holds each pixel exactly once, so these
                // writes are disjoint; `pixels` outlives this scope.
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        stored.as_ptr(),
                        (ptr as *mut u8).add(pixel * 4),
                        4,
                    );
                }
                }
            });
        }
    });
    Image::new(source.width, source.height, pixels).ok_or(Refusal::ShapeMismatch("V2Warp output"))
}

struct SelectionCache {
    width: usize,
    height: usize,
    rules: Rules,
    positions: Vec<[f32; 2]>,
    ebo: Vec<u16>,
    selected: Arc<Vec<u32>>,
    coverage: Arc<Vec<u32>>,
    covered: Arc<Vec<u32>>,
}

fn selection_cache() -> &'static Mutex<Option<SelectionCache>> {
    static CACHE: OnceLock<Mutex<Option<SelectionCache>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

/// Triangle selection depends on the mesh and raster rules, not on the
/// control-driven offset texture. A drag reuses the same mesh every tick.
fn selection_pair(
    width: usize,
    height: usize,
    mesh: &Mesh<'_>,
    rules: Rules,
) -> (Arc<Vec<u32>>, Arc<Vec<u32>>, Arc<Vec<u32>>) {
    let mut cache = selection_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(entry) = cache.as_ref() {
        if entry.width == width
            && entry.height == height
            && entry.rules == rules
            && entry.positions.as_slice() == mesh.framebuffer_positions_w1
            && entry.ebo.as_slice() == mesh.ebo
        {
            return (
                Arc::clone(&entry.selected),
                Arc::clone(&entry.coverage),
                Arc::clone(&entry.covered),
            );
        }
    }
    let (selected, coverage, covered) = select_impl(width, height, mesh, rules);
    let entry = SelectionCache {
        width,
        height,
        rules,
        positions: mesh.framebuffer_positions_w1.to_vec(),
        ebo: mesh.ebo.to_vec(),
        selected: Arc::new(selected),
        coverage: Arc::new(coverage),
        covered: Arc::new(covered),
    };
    let pair = (
        Arc::clone(&entry.selected),
        Arc::clone(&entry.coverage),
        Arc::clone(&entry.covered),
    );
    *cache = Some(entry);
    pair
}

fn select_impl(
    width: usize,
    height: usize,
    mesh: &Mesh<'_>,
    rules: Rules,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut selected = if rules.overlap == Overlap::Last {
        vec![0u32; width * height]
    } else {
        vec![u32::MAX; width * height]
    };
    let mut coverage = vec![0u32; width * height];
    for triangle in 0..TRIANGLE_COUNT {
        let i = [
            mesh.ebo[triangle * 3] as usize,
            mesh.ebo[triangle * 3 + 1] as usize,
            mesh.ebo[triangle * 3 + 2] as usize,
        ];
        if i[0] == i[1] || i[1] == i[2] || i[0] == i[2] {
            continue;
        }
        let p = [
            mesh.framebuffer_positions_w1[i[0]],
            mesh.framebuffer_positions_w1[i[1]],
            mesh.framebuffer_positions_w1[i[2]],
        ];
        let area2 = cross(sub(p[1], p[0]), sub(p[2], p[0]));
        if area2.abs() <= 1e-12 {
            continue;
        }
        let front = match rules.winding {
            Winding::PositiveAreaIsFront => area2 > 0.0,
            Winding::NegativeAreaIsFront => area2 < 0.0,
        };
        if (rules.cull == Cull::Front && front) || (rules.cull == Cull::Back && !front) {
            continue;
        }
        let min_x = p[0][0].min(p[1][0]).min(p[2][0]).floor().max(0.0) as usize;
        let max_x = p[0][0]
            .max(p[1][0])
            .max(p[2][0])
            .ceil()
            .min(width.saturating_sub(1) as f32)
            .max(0.0) as usize;
        let min_y = p[0][1].min(p[1][1]).min(p[2][1]).floor().max(0.0) as usize;
        let max_y = p[0][1]
            .max(p[1][1])
            .max(p[2][1])
            .ceil()
            .min(height.saturating_sub(1) as f32)
            .max(0.0) as usize;
        if min_x > max_x || min_y > max_y {
            continue;
        }
        for py in min_y..=max_y {
            for px in min_x..=max_x {
                if weights(p, area2, px as f32 + 0.5, py as f32 + 0.5, rules.edge).is_none() {
                    continue;
                }
                let pixel = py * width + px;
                coverage[pixel] += 1;
                selected[pixel] = match rules.overlap {
                    Overlap::Last => selected[pixel].max(triangle as u32),
                    Overlap::First | Overlap::Reject => selected[pixel].min(triangle as u32),
                };
            }
        }
    }
    // Row-major covered list. Only these pixels can be written, so the eval
    // pass never rescans the frame.
    let covered = coverage
        .iter()
        .enumerate()
        .filter(|&(_, &count)| count > 0 && (count == 1 || rules.overlap != Overlap::Reject))
        .map(|(pixel, _)| pixel as u32)
        .collect();
    (selected, coverage, covered)
}

fn sub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn cross(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[1] - a[1] * b[0]
}

fn weights(p: [[f32; 2]; 3], area2: f32, sx: f32, sy: f32, edge: Edge) -> Option<[f32; 3]> {
    let s = [sx, sy];
    let mut w = [
        cross(sub(p[1], s), sub(p[2], s)) / area2,
        cross(sub(p[2], s), sub(p[0], s)) / area2,
        cross(sub(p[0], s), sub(p[1], s)) / area2,
    ];
    let inside = match edge {
        Edge::Inclusive => w.iter().all(|value| *value >= 0.0),
        Edge::Exclusive => w.iter().all(|value| *value > 0.0),
    };
    if !inside {
        return None;
    }
    for value in &mut w {
        *value = value.max(0.0);
    }
    let sum = w[0] + w[1] + w[2];
    (sum > 0.0).then(|| [w[0] / sum, w[1] / sum, w[2] / sum])
}

fn bary(a: [f32; 2], b: [f32; 2], c: [f32; 2], w: [f32; 3]) -> [f32; 2] {
    [
        w[0] * a[0] + w[1] * b[0] + w[2] * c[0],
        w[0] * a[1] + w[1] * b[1] + w[2] * c[1],
    ]
}

fn sample_offset(offset: OffsetTexture<'_>, uv: [f32; 2]) -> [f32; 2] {
    let w = offset.width as f32;
    let h = offset.height as f32;
    let x = (uv[0] * w - 0.5).clamp(0.0, w - 1.0);
    let y = (uv[1] * h - 0.5).clamp(0.0, h - 1.0);
    let x0 = x.floor();
    let y0 = y.floor();
    let x1 = (x0 + 1.0).min(w - 1.0);
    let y1 = (y0 + 1.0).min(h - 1.0);
    let fx = x - x0;
    let fy = y - y0;
    let mut rgba = [0.0f32; 4];
    let texel =
        |x: usize, y: usize, channel: usize| offset.pixels[y * offset.width as usize + x][channel];
    for channel in 0..4 {
        let top = texel(x0 as usize, y0 as usize, channel)
            + (texel(x1 as usize, y0 as usize, channel) - texel(x0 as usize, y0 as usize, channel))
                * fx;
        let bottom = texel(x0 as usize, y1 as usize, channel)
            + (texel(x1 as usize, y1 as usize, channel) - texel(x0 as usize, y1 as usize, channel))
                * fx;
        rgba[channel] = top + (bottom - top) * fy;
    }
    [(rgba[0] - rgba[1]) * 0.25, (rgba[2] - rgba[3]) * 0.25]
}

fn edge(uv: [f32; 2]) -> f32 {
    let x = 2.49 - (4.98 * (uv[0] - 0.5)).abs();
    let y = 2.49 - (4.98 * (uv[1] - 0.5)).abs();
    x.min(y).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RULES: Rules = Rules {
        winding: Winding::NegativeAreaIsFront,
        cull: Cull::None,
        edge: Edge::Inclusive,
        overlap: Overlap::Last,
    };

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

    fn offset_pixels() -> Vec<[f32; 4]> {
        vec![[4.0, 0.0, 0.0, 0.0]; (OFFSET_WIDTH * OFFSET_HEIGHT) as usize]
    }

    fn one_triangle() -> (
        Vec<[f32; 2]>,
        Vec<[f32; 2]>,
        Vec<[f32; 2]>,
        Vec<[f32; 2]>,
        Vec<u16>,
    ) {
        let mut positions = vec![[0.0; 2]; VERTEX_COUNT];
        let mut texcoord = vec![[0.0; 2]; VERTEX_COUNT];
        let offset_uv = vec![[0.5; 2]; VERTEX_COUNT];
        let scales = vec![[1.0; 2]; VERTEX_COUNT];
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
        // Repeating degenerate triangles keeps the fixed ABI without adding geometry.
        (positions, texcoord, offset_uv, scales, ebo)
    }

    #[test]
    fn exact_526_2982_contract_is_required() {
        let source = source();
        let offset_pixels = offset_pixels();
        let offset = OffsetTexture {
            width: OFFSET_WIDTH,
            height: OFFSET_HEIGHT,
            pixels: &offset_pixels,
        };
        let (p, t, o, s, e) = one_triangle();
        let mesh = Mesh {
            framebuffer_positions_w1: &p,
            texcoord: &t,
            offset_texcoord: &o,
            face_scales: &s,
            ebo: &e,
        };
        assert_eq!(
            validate(&source, offset, mesh, [1.0, 0.0, 0.0, 1.0], [17.0, 13.0]),
            Ok(())
        );
        assert_eq!(
            validate(
                &source,
                offset,
                Mesh {
                    framebuffer_positions_w1: &p[..525],
                    ..mesh
                },
                [1.0; 4],
                [17.0, 13.0]
            ),
            Err(Refusal::ShapeMismatch("a_position_w1"))
        );
        assert_eq!(
            validate(
                &source,
                offset,
                Mesh {
                    ebo: &e[..2981],
                    ..mesh
                },
                [1.0; 4],
                [17.0, 13.0]
            ),
            Err(Refusal::ShapeMismatch("topology2982"))
        );
        assert_eq!(
            validate(&source, offset, mesh, [1.0; 4], [16.0, 13.0]),
            Err(Refusal::ValueOutOfRange("V2Warp F/O != source size"))
        );
    }

    #[test]
    fn cpu_forward_formula_covers_pixels() {
        let source = source();
        let offset_pixels = offset_pixels();
        let offset = OffsetTexture {
            width: OFFSET_WIDTH,
            height: OFFSET_HEIGHT,
            pixels: &offset_pixels,
        };
        let (p, t, o, s, e) = one_triangle();
        let mesh = Mesh {
            framebuffer_positions_w1: &p,
            texcoord: &t,
            offset_texcoord: &o,
            face_scales: &s,
            ebo: &e,
        };
        let out = apply_cpu(
            &source,
            offset,
            mesh,
            RULES,
            [1.0, 0.0, 0.0, 1.0],
            [17.0, 13.0],
        )
        .unwrap();
        assert_ne!(out.pixels, source.pixels);
        assert_eq!(out.pixels[3], 255);
    }

    #[cfg(windows)]
    #[test]
    fn gpu_forward_candidate_matches_cpu_oracle() {
        let gpu = match V2Warp::open() {
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
        let source = source();
        let offset_pixels = offset_pixels();
        let offset = OffsetTexture {
            width: OFFSET_WIDTH,
            height: OFFSET_HEIGHT,
            pixels: &offset_pixels,
        };
        let (p, t, o, s, e) = one_triangle();
        let mesh = Mesh {
            framebuffer_positions_w1: &p,
            texcoord: &t,
            offset_texcoord: &o,
            face_scales: &s,
            ebo: &e,
        };
        let expected = apply_cpu(
            &source,
            offset,
            mesh,
            RULES,
            [1.0, 0.0, 0.0, 1.0],
            [17.0, 13.0],
        )
        .unwrap();
        let out = gpu
            .render(
                &source,
                offset,
                mesh,
                RULES,
                [1.0, 0.0, 0.0, 1.0],
                [17.0, 13.0],
            )
            .expect("real GPU dispatch");
        let max_diff = out
            .image
            .pixels
            .iter()
            .zip(&expected.pixels)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        let changed = out.coverage.iter().filter(|value| **value > 0).count();
        eprintln!(
            "V2WARP_GPU path={} device={} select={} eval={} covered={} max_diff={}",
            out.path,
            out.select_dispatch.device,
            out.select_dispatch.kernel,
            out.eval_dispatch.kernel,
            changed,
            max_diff
        );
        assert!(changed > 0);
        assert!(max_diff <= 1, "GPU/CPU max diff {max_diff}");
        assert_eq!(out.path, PATH);
        assert_eq!(out.select_dispatch.kernel, SELECT_KERNEL);
        assert_eq!(out.eval_dispatch.kernel, EVAL_KERNEL);
    }
}

#[cfg(test)]
mod selection_cache_tests {
    use super::*;

    #[test]
    fn selection_cache_reuses_the_same_mesh_and_matches_a_fresh_compute() {
        let mut positions = vec![[0.0f32, 0.0]; VERTEX_COUNT];
        positions[0] = [2.0, 2.0];
        positions[1] = [12.0, 2.0];
        positions[2] = [2.0, 12.0];
        let texcoord = vec![[0.0f32, 0.0]; VERTEX_COUNT];
        let face_scales = vec![[1.0f32, 1.0]; VERTEX_COUNT];
        let mut ebo = vec![0u16; INDEX_COUNT];
        ebo[0] = 0;
        ebo[1] = 1;
        ebo[2] = 2;
        let mesh = Mesh {
            framebuffer_positions_w1: &positions,
            texcoord: &texcoord,
            offset_texcoord: &texcoord,
            face_scales: &face_scales,
            ebo: &ebo,
        };
        let rules = Rules {
            winding: Winding::PositiveAreaIsFront,
            cull: Cull::None,
            edge: Edge::Inclusive,
            overlap: Overlap::Last,
        };
        *selection_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        let (selected, coverage, covered) = selection_pair(16, 16, &mesh, rules);
        let (serial_selected, serial_coverage, serial_covered) =
            select_impl(16, 16, &mesh, rules);
        assert_eq!(*selected, serial_selected);
        assert_eq!(*coverage, serial_coverage);
        assert_eq!(*covered, serial_covered);
        let (again, _, _) = selection_pair(16, 16, &mesh, rules);
        assert!(Arc::ptr_eq(&selected, &again));
        *selection_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
    }
}
