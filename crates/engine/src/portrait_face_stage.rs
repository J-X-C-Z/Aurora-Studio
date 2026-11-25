use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use aurora_portrait::cuda::Dispatch;
use aurora_portrait::cuda_v2warp::{self, OffsetTexture};
use aurora_portrait::cuda_vface::{self, Row};
use aurora_portrait::kernels::{eye_geometry, face_detail, face_mesh, smile_warp};
use aurora_portrait::{Image, Refusal, raster, raster_fragment};

use super::{PortraitBackend, SlimProducerInputs, sha256_32, smile_refusal};
use super::portrait_v2_mesh::VFaceAtlas;
use crate::NormRect;

#[cfg(windows)]
use aurora_portrait::cuda_mesh::MeshRaster;

/// One draw produced for the exact requested control pair and input frame.
/// Arrays for the V2 vertex/offset draws borrow the existing Slim bundle.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceGeometryDraw {
    pub pf: u32,
    pub controls: [f32; 2],
    pub source_sha256: [u8; 32],
    pub geometry: FaceGeometry,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FaceGeometry {
    /// Position replacements in the captured V2 vertex index space.
    V2Positions(Vec<(u16, [f32; 2])>),
    /// Exact atlas bindings and final encoded field after the active rows
    /// have been fused in producer order.
    VFace {
        rows: Vec<Row>,
        encoded_offset: Vec<[f32; 4]>,
        atlases: Vec<VFaceAtlas>,
    },
    /// Type1011 prepared100 source points and its six vendor weights. The EBO
    /// is generated at dispatch time from these prepared points.
    Type1011 {
        prepared: Vec<[f32; 2]>,
        param_list: Vec<f32>,
        center: [f32; 2],
    },
    /// Prepared eye/brow draw, with framebuffer-pixel positions, normalized
    /// UVs and the captured EBO and fragment state.
    Eye {
        dest: Vec<[f32; 2]>,
        uv: Vec<[f32; 2]>,
        ebo: Vec<u16>,
        rules: raster::Rules,
        edge: raster_fragment::Edge,
        defines: raster_fragment::Defines,
        clamp: raster_fragment::Clamp,
    },
    /// Unit-strength PF8033 displacement and mask in full-frame pixel order.
    Smile {
        dx: Vec<f32>,
        dy: Vec<f32>,
        mask: Vec<u8>,
        host_split_receipt: Option<SmileHostSplitReceipt>,
    },
    /// PF202 Type1012 draw from the recovered 5x27 producer layout. The EBO
    /// is the captured 136/705 topology, not a tessellation of source points.
    HairLine {
        dest: Vec<[f32; 2]>,
        uv: Vec<[f32; 2]>,
        ebo: Vec<u16>,
        rules: raster::Rules,
        edge: raster_fragment::Edge,
        defines: raster_fragment::Defines,
        clamp: raster_fragment::Clamp,
    },
    /// PF232 radial contour from points 192..206 and the fifteen recovered
    /// factors. This stage has no EBO because the producer is a radial warp.
    Skull {
        anchor: [f32; 2],
        source: Vec<[f32; 2]>,
        warped: Vec<[f32; 2]>,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct FaceGeometryInputs {
    pub job_id: u64,
    pub width: u32,
    pub height: u32,
    pub draws: Vec<FaceGeometryDraw>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmileHostSplitReceipt {
    pub job_id: u64,
    pub model_output_id: u32,
    pub model_output_sha256: [u8; 32],
    pub dx_sha256: [u8; 32],
    pub dy_sha256: [u8; 32],
    pub mask_sha256: [u8; 32],
}

impl SmileHostSplitReceipt {
    /// Bind the exact model tensor and the host-split planes derived from it.
    pub fn from_model_output(
        job_id: u64,
        model_output: &super::SmileRgbReconstructionField,
        dx: &[f32],
        dy: &[f32],
        mask: &[u8],
    ) -> Result<Self, Refusal> {
        if job_id == 0 {
            return Err(Refusal::ValueOutOfRange("PF8033 host split ticket"));
        }
        let pixels = super::smile_model_pixels(
            model_output.width,
            model_output.height,
            "PF8033 780929437 shape",
        )?;
        if model_output.values.len() != pixels.saturating_mul(3)
            || model_output.values.iter().any(|value| !value.is_finite())
        {
            return Err(Refusal::ShapeMismatch("PF8033 780929437 values"));
        }
        if dx.len() != pixels || dy.len() != pixels || mask.len() != pixels {
            return Err(Refusal::ShapeMismatch("PF8033 host split planes"));
        }
        if dx.iter().chain(dy).any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("PF8033 displacement"));
        }
        let model_bytes = model_output
            .values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        Ok(Self {
            job_id,
            model_output_id: super::SmileRgbReconstructionField::OUTPUT_ID,
            model_output_sha256: sha256_32(&model_bytes),
            dx_sha256: hash_f32(dx),
            dy_sha256: hash_f32(dy),
            mask_sha256: sha256_32(mask),
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct FaceGeometryRun {
    pub image: Option<Image>,
    pub backend: PortraitBackend,
    pub dispatches: Vec<Dispatch>,
    pub fallback_reason: Option<String>,
}

impl FaceGeometryRun {
    fn skipped() -> Self {
        Self {
            image: None,
            backend: PortraitBackend::Skipped,
            dispatches: Vec::new(),
            fallback_reason: None,
        }
    }

    fn cpu(image: Option<Image>) -> Self {
        Self {
            image,
            backend: PortraitBackend::Cpu,
            dispatches: Vec::new(),
            fallback_reason: None,
        }
    }
}

impl FaceGeometryInputs {
    pub(super) fn fingerprint(&self) -> u64 {
        let mut h = DefaultHasher::new();
        (self.job_id, self.width, self.height).hash(&mut h);
        self.draws.len().hash(&mut h);
        for draw in &self.draws {
            (draw.pf, draw.controls.map(f32::to_bits), draw.source_sha256).hash(&mut h);
            std::mem::discriminant(&draw.geometry).hash(&mut h);
            match &draw.geometry {
                FaceGeometry::V2Positions(points) => {
                    points.len().hash(&mut h);
                    for &(i, p) in points {
                        (i, p.map(f32::to_bits)).hash(&mut h);
                    }
                }
                FaceGeometry::VFace {
                    rows,
                    encoded_offset,
                    atlases,
                } => {
                    rows.len().hash(&mut h);
                    for row in rows {
                        (*row as u8).hash(&mut h);
                    }
                    hash_points(encoded_offset, &mut h);
                    for atlas in atlases {
                        (atlas.row as u8, atlas.width, atlas.height).hash(&mut h);
                        atlas.pixels.hash(&mut h);
                    }
                }
                FaceGeometry::Type1011 {
                    prepared,
                    param_list,
                    center,
                } => {
                    hash_points(prepared, &mut h);
                    param_list.len().hash(&mut h);
                    for value in param_list {
                        value.to_bits().hash(&mut h);
                    }
                    center.map(f32::to_bits).hash(&mut h);
                }
                FaceGeometry::Eye {
                    dest,
                    uv,
                    ebo,
                    rules,
                    edge,
                    defines,
                    clamp,
                } => {
                    hash_points(dest, &mut h);
                    hash_points(uv, &mut h);
                    ebo.hash(&mut h);
                    (
                        rules.winding as u8,
                        rules.cull as u8,
                        rules.edge as u8,
                        rules.overlap as u8,
                        *edge as u8,
                        defines.clamp_edge,
                        defines.uv_map,
                        defines.uv_map_ba,
                        defines.catmull_rom,
                        clamp.factor.to_bits(),
                        clamp.protect_x,
                        clamp.protect_y,
                    )
                        .hash(&mut h);
                }
                FaceGeometry::Smile { dx, dy, mask, .. } => {
                    for plane in [dx, dy] {
                        plane.len().hash(&mut h);
                        for value in plane {
                            value.to_bits().hash(&mut h);
                        }
                    }
                    mask.hash(&mut h);
                }
                FaceGeometry::HairLine {
                    dest,
                    uv,
                    ebo,
                    rules,
                    edge,
                    defines,
                    clamp,
                } => {
                    hash_points(dest, &mut h);
                    hash_points(uv, &mut h);
                    ebo.hash(&mut h);
                    (
                        rules.winding as u8,
                        rules.cull as u8,
                        rules.edge as u8,
                        rules.overlap as u8,
                        *edge as u8,
                        defines.clamp_edge,
                        defines.uv_map,
                        defines.uv_map_ba,
                        defines.catmull_rom,
                        clamp.factor.to_bits(),
                        clamp.protect_x,
                        clamp.protect_y,
                    )
                        .hash(&mut h);
                }
                FaceGeometry::Skull {
                    anchor,
                    source,
                    warped,
                } => {
                    anchor.map(f32::to_bits).hash(&mut h);
                    hash_points(source, &mut h);
                    hash_points(warped, &mut h);
                }
            }
        }
        h.finish()
    }
}

fn hash_points<const N: usize>(points: &[[f32; N]], h: &mut DefaultHasher) {
    points.len().hash(h);
    for point in points {
        point.map(f32::to_bits).hash(h);
    }
}

pub(super) fn hash_f32(values: &[f32]) -> [u8; 32] {
    sha256_32(
        &values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

pub(super) fn pair(pf: u32) -> Option<(u32, u32)> {
    match pf {
        202 => Some((202, 202)),
        232 => Some((232, 232)),
        242 | 243 => Some((242, 243)),
        247 | 248 => Some((247, 248)),
        30073 | 30079 => Some((30073, 30079)),
        30080 | 30081 => Some((30080, 30081)),
        30082 | 30083 => Some((30082, 30083)),
        44124 | 48019 => Some((44124, 48019)),
        49221 | 41922 => Some((49221, 41922)),
        8001..=8024 | 8027..=8030 if !matches!(pf, 8003 | 8004 | 8007 | 8008) => {
            let left = pf - (pf + 1) % 2;
            Some((left, left + 1))
        }
        8033 => Some((8033, 8033)),
        _ => None,
    }
}

fn missing_draw(pf: u32) -> &'static str {
    match producer_gap(pf) {
        Some(gap) => gap,
        None => draw_gap(pf),
    }
}

/// Registry rows whose CPU geometry producer is callable but has no stage
/// route at all. The refusal names the inputs that producer documents, so a
/// missing producer stays distinguishable from a row with no kernel.
pub(super) fn producer_gap(pf: u32) -> Option<&'static str> {
    let _ = pf;
    None
}

fn draw_gap(pf: u32) -> &'static str {
    match pf {
        30080..=30083 => "PF30080..PF30083 Type1012 texture/V2 handoff",
        30073 | 30079 => "PF30073/PF30079 Type1011 EBO 0x60F2B60 and frame draw",
        44124 | 48019 | 49221 | 41922 => {
            "PF44124/PF48019/PF49221/PF41922 Type1018 EBO 0x7080940 and frame draw"
        }
        // Every brow/topology row names the exact index buffer the kernel
        // refuses without, so a refusal states the missing contract instead
        // of a generic same-job label.
        _ => eye_geometry::ebo_rva(pf).unwrap_or("same-job face geometry"),
    }
}

fn is_eye_draw(pf: u32) -> bool {
    matches!(
        pf,
        8001..=8022 | 44124 | 48019 | 49221 | 41922
    )
}

fn remaining_geometry(pf: u32) -> bool {
    matches!(
        pf,
        30073 | 30079 | 44124 | 48019 | 49221 | 41922
    )
}

/// Rows the CUDA fragment kernel may serve. The fragment emit path takes the
/// caller's index buffer without any topology assumption (`cuda_mesh` only
/// validates length and index range), so the recovered eye and brow family
/// rides the same contract as the WL rows once a same-job EBO is supplied.
fn prefer_cuda(pf: u32) -> bool {
    remaining_geometry(pf) || matches!(pf, 8001..=8022)
}

fn eye_draw(
    source: &Image,
    pf: u32,
    dest: &[[f32; 2]],
    uv: &[[f32; 2]],
    ebo: &[u16],
    rules: raster::Rules,
    edge: raster_fragment::Edge,
    defines: raster_fragment::Defines,
    clamp: raster_fragment::Clamp,
) -> Result<Image, Refusal> {
    eye_geometry::accept_indices(pf, Some(ebo), dest.len())
        .map_err(|_| Refusal::ValueOutOfRange("eye EBO"))?;
    if defines.uv_map || defines.uv_map_ba || !clamp.factor.is_finite() {
        return Err(Refusal::ValueOutOfRange("eye pixel fragment state"));
    }
    raster::rasterize_with_fragment(
        source,
        raster::Mesh {
            dest,
            source_uv: uv,
            ebo,
        },
        rules,
        raster::Sampler {
            wrap: raster::Wrap::ClampToEdge,
        },
        edge,
        defines,
        clamp,
        raster::Tolerance::RESHAPE,
    )
}

fn apply_eye(
    source: &Image,
    pf: u32,
    dest: &[[f32; 2]],
    uv: &[[f32; 2]],
    ebo: &[u16],
    rules: raster::Rules,
    edge: raster_fragment::Edge,
    defines: raster_fragment::Defines,
    clamp: raster_fragment::Clamp,
    prefer_cuda: bool,
) -> Result<FaceGeometryRun, Refusal> {
    #[cfg(windows)]
    {
        if !prefer_cuda {
            let output = eye_draw(source, pf, dest, uv, ebo, rules, edge, defines, clamp)?;
            let changed = output != *source;
            return Ok(FaceGeometryRun::cpu(changed.then_some(output)));
        }
        let cuda = match super::forced_cuda_failure() {
            Some(error) => Err(error),
            None => MeshRaster::open().and_then(|gpu| {
                gpu.rasterize_fragment(
                    source,
                    raster::Mesh {
                        dest,
                        source_uv: uv,
                        ebo,
                    },
                    rules,
                    raster::Sampler {
                        wrap: raster::Wrap::ClampToEdge,
                    },
                    edge,
                    defines,
                    clamp,
                    raster::Tolerance::RESHAPE,
                )
            }),
        };
        match cuda {
            Ok(output) => {
                let changed = output.image != *source;
                return Ok(FaceGeometryRun {
                    image: changed.then_some(output.image),
                    backend: PortraitBackend::Cuda,
                    dispatches: vec![output.select_dispatch, output.emit_dispatch],
                    fallback_reason: None,
                });
            }
            Err(error) => {
                let fallback = format!("CUDA face geometry fallback: {error:?}");
                let output = eye_draw(source, pf, dest, uv, ebo, rules, edge, defines, clamp)?;
                let changed = output != *source;
                return Ok(FaceGeometryRun {
                    image: changed.then_some(output),
                    backend: PortraitBackend::Cpu,
                    dispatches: Vec::new(),
                    fallback_reason: Some(fallback),
                });
            }
        }
    }
    #[cfg(not(windows))]
    {
        let output = eye_draw(source, pf, dest, uv, ebo, rules, edge, defines, clamp)?;
        let changed = output != *source;
        Ok(FaceGeometryRun::cpu(changed.then_some(output)))
    }
}

fn render_v2(
    source: &Image,
    offset: OffsetTexture<'_>,
    mesh: cuda_v2warp::Mesh<'_>,
    rules: raster::Rules,
    mat2: [f32; 4],
    filter_size: [f32; 2],
) -> Result<FaceGeometryRun, Refusal> {
    #[cfg(windows)]
    {
        // Per-call CUDA buffers cost more than the face-local covered-pixel
        // CPU pass at preview sizes.
        let preview_sized =
            u64::from(source.width) * u64::from(source.height) <= 4_000_000;
        if preview_sized && v2_mesh_is_face_local(source, &mesh) {
            let image =
                cuda_v2warp::apply_cpu(source, offset, mesh, rules, mat2, filter_size)?;
            let changed = image != *source;
            return Ok(FaceGeometryRun {
                image: changed.then_some(image),
                backend: PortraitBackend::Cpu,
                dispatches: Vec::new(),
                fallback_reason: Some(
                    "face-local V2 warp: CPU covered-pixel path".to_string(),
                ),
            });
        }
        let cuda = match super::forced_cuda_failure() {
            Some(error) => Err(error),
            None => cuda_v2warp::V2Warp::open()
                .and_then(|gpu| gpu.render(source, offset, mesh, rules, mat2, filter_size)),
        };
        match cuda {
            Ok(output) => {
                let changed = output.image != *source;
                return Ok(FaceGeometryRun {
                    image: changed.then_some(output.image),
                    backend: PortraitBackend::Cuda,
                    dispatches: vec![output.select_dispatch, output.eval_dispatch],
                    fallback_reason: None,
                });
            }
            Err(error) => {
                let fallback = format!("CUDA V2Warp fallback: {error:?}");
                let image = cuda_v2warp::apply_cpu(source, offset, mesh, rules, mat2, filter_size)?;
                let changed = image != *source;
                return Ok(FaceGeometryRun {
                    image: changed.then_some(image),
                    backend: PortraitBackend::Cpu,
                    dispatches: Vec::new(),
                    fallback_reason: Some(fallback),
                });
            }
        }
    }
    #[cfg(not(windows))]
    {
        let image = cuda_v2warp::apply_cpu(source, offset, mesh, rules, mat2, filter_size)?;
        let changed = image != *source;
        Ok(FaceGeometryRun::cpu(changed.then_some(image)))
    }
}

fn v2_mesh_is_face_local(source: &Image, mesh: &cuda_v2warp::Mesh<'_>) -> bool {
    let (mut min_x, mut max_x) = (f32::INFINITY, f32::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f32::INFINITY, f32::NEG_INFINITY);
    for point in mesh.framebuffer_positions_w1 {
        if !point[0].is_finite() || !point[1].is_finite() {
            return false;
        }
        min_x = min_x.min(point[0]);
        max_x = max_x.max(point[0]);
        min_y = min_y.min(point[1]);
        max_y = max_y.max(point[1]);
    }
    if !min_x.is_finite() {
        return false;
    }
    let (width, height) = (source.width as f32, source.height as f32);
    let bbox = (max_x - min_x).max(0.0).min(width) * (max_y - min_y).max(0.0).min(height);
    bbox <= width * height * 0.5
}

fn render_mesh(
    source: &Image,
    dest: &[[f32; 2]],
    uv: &[[f32; 2]],
    ebo: &[u16],
    rules: raster::Rules,
    edge: raster_fragment::Edge,
    defines: raster_fragment::Defines,
    clamp: raster_fragment::Clamp,
) -> Result<FaceGeometryRun, Refusal> {
    if dest.is_empty()
        || dest.len() != uv.len()
        || ebo.is_empty()
        || ebo.len() % 3 != 0
        || ebo.iter().any(|&index| index as usize >= dest.len())
        || dest.iter().chain(uv).flatten().any(|value| !value.is_finite())
    {
        return Err(Refusal::ShapeMismatch("PF202 Type1012 mesh"));
    }
    let mesh = raster::Mesh {
        dest,
        source_uv: uv,
        ebo,
    };
    let sampler = raster::Sampler {
        wrap: raster::Wrap::ClampToEdge,
    };
    #[cfg(windows)]
    {
        let cuda = match super::forced_cuda_failure() {
            Some(error) => Err(error),
            None => MeshRaster::open().and_then(|gpu| {
                gpu.rasterize_fragment(
                    source,
                    mesh,
                    rules,
                    sampler,
                    edge,
                    defines,
                    clamp,
                    raster::Tolerance::RESHAPE,
                )
            }),
        };
        match cuda {
            Ok(output) => {
                let changed = output.image != *source;
                return Ok(FaceGeometryRun {
                    image: changed.then_some(output.image),
                    backend: PortraitBackend::Cuda,
                    dispatches: vec![output.select_dispatch, output.emit_dispatch],
                    fallback_reason: None,
                });
            }
            Err(error) => {
                let fallback = format!("CUDA PF202 fallback: {error:?}");
                let image = raster::rasterize_with_fragment(
                    source,
                    mesh,
                    rules,
                    sampler,
                    edge,
                    defines,
                    clamp,
                    raster::Tolerance::RESHAPE,
                )?;
                let changed = image != *source;
                return Ok(FaceGeometryRun {
                    image: changed.then_some(image),
                    backend: PortraitBackend::Cpu,
                    dispatches: Vec::new(),
                    fallback_reason: Some(fallback),
                });
            }
        }
    }
    #[cfg(not(windows))]
    {
        let image = raster::rasterize_with_fragment(
            source,
            mesh,
            rules,
            sampler,
            edge,
            defines,
            clamp,
            raster::Tolerance::RESHAPE,
        )?;
        let changed = image != *source;
        Ok(FaceGeometryRun::cpu(changed.then_some(image)))
    }
}

fn sample_bilinear(source: &Image, x: f32, y: f32) -> [u8; 4] {
    let x = x.clamp(0.0, source.width.saturating_sub(1) as f32);
    let y = y.clamp(0.0, source.height.saturating_sub(1) as f32);
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(source.width - 1);
    let y1 = (y0 + 1).min(source.height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let pixel = |px: u32, py: u32| -> [u8; 4] {
        let at = (py as usize * source.width as usize + px as usize) * 4;
        source.pixels[at..at + 4].try_into().unwrap()
    };
    let a = pixel(x0, y0);
    let b = pixel(x1, y0);
    let c = pixel(x0, y1);
    let d = pixel(x1, y1);
    let mut out = [0u8; 4];
    for channel in 0..3 {
        let top = a[channel] as f32 * (1.0 - fx) + b[channel] as f32 * fx;
        let bottom = c[channel] as f32 * (1.0 - fx) + d[channel] as f32 * fx;
        out[channel] = (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
    }
    out[3] = source.pixels[(y0 as usize * source.width as usize + x0 as usize) * 4 + 3];
    out
}

fn render_skull(
    source: &Image,
    anchor: [f32; 2],
    contour_source: &[[f32; 2]],
    contour_warped: &[[f32; 2]],
) -> Result<FaceGeometryRun, Refusal> {
    if contour_source.len() != 15
        || contour_warped.len() != 15
        || contour_source
            .iter()
            .chain(contour_warped)
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(Refusal::ShapeMismatch("PF232 15-point radial contour"));
    }
    let mut min_x = anchor[0];
    let mut min_y = anchor[1];
    let mut max_x = anchor[0];
    let mut max_y = anchor[1];
    for point in contour_source.iter().chain(contour_warped) {
        min_x = min_x.min(point[0]);
        min_y = min_y.min(point[1]);
        max_x = max_x.max(point[0]);
        max_y = max_y.max(point[1]);
    }
    let x0 = (min_x.floor() as i32 - 5).clamp(0, source.width as i32) as u32;
    let y0 = (min_y.floor() as i32 - 5).clamp(0, source.height as i32) as u32;
    let x1 = (max_x.ceil() as i32 + 5).clamp(x0 as i32, source.width as i32) as u32;
    let y1 = (max_y.ceil() as i32 + 5).clamp(y0 as i32, source.height as i32) as u32;
    let mut output = source.clone();
    for y in y0..y1 {
        for x in x0..x1 {
            let pixel = [x as f32 + 0.5, y as f32 + 0.5];
            let dx = pixel[0] - anchor[0];
            let dy = pixel[1] - anchor[1];
            let radius = (dx * dx + dy * dy).sqrt();
            if radius < 1e-6 {
                continue;
            }
            let direction = [dx / radius, dy / radius];
            let mut best = 0usize;
            let mut best_dot = f32::MIN;
            for (index, (warped, original)) in contour_warped
                .iter()
                .zip(contour_source)
                .enumerate()
            {
                let vx = (warped[0] + original[0]) * 0.5 - anchor[0];
                let vy = (warped[1] + original[1]) * 0.5 - anchor[1];
                let norm = (vx * vx + vy * vy).sqrt();
                let dot = if norm > 1e-6 {
                    direction[0] * vx / norm + direction[1] * vy / norm
                } else {
                    f32::MIN
                };
                if dot > best_dot {
                    best_dot = dot;
                    best = index;
                }
            }
            let contour_radius = |point: [f32; 2]| {
                ((point[0] - anchor[0]) * direction[0]
                    + (point[1] - anchor[1]) * direction[1])
                    .max(1e-4)
            };
            let warped_radius = contour_radius(contour_warped[best]);
            let source_radius = contour_radius(contour_source[best]);
            let mapped_radius = radius * source_radius / warped_radius;
            let sx = anchor[0] + direction[0] * mapped_radius;
            let sy = anchor[1] + direction[1] * mapped_radius;
            let at = (y as usize * source.width as usize + x as usize) * 4;
            output.pixels[at..at + 4].copy_from_slice(&sample_bilinear(source, sx, sy));
        }
    }
    let changed = output != *source;
    Ok(FaceGeometryRun::cpu(changed.then_some(output)))
}

fn apply_type1011(
    source: &Image,
    pf: u32,
    controls: [f32; 2],
    shared: Option<&SlimProducerInputs>,
    prepared: &[[f32; 2]],
    param_list: &[f32],
    center: [f32; 2],
) -> Result<FaceGeometryRun, Refusal> {
    let shared = shared.ok_or(Refusal::MissingInput("shared face landmarks"))?;
    if shared.points194.len() != 194
        || shared
            .points194
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
    {
        return Err(Refusal::ShapeMismatch(
            "Type1011 same-job 194 landmarks",
        ));
    }
    if source.width == 0 || source.height == 0 || prepared.len() != 100 {
        return Err(Refusal::ShapeMismatch("Type1011 prepared100 frame"));
    }
    let prepared_points: Vec<eye_geometry::Point> = prepared
        .iter()
        .map(|point| eye_geometry::Point::new(point[0], point[1]))
        .collect();
    let indices = eye_geometry::type1011_indices(&prepared_points)
        .map_err(|_| Refusal::ShapeMismatch("Type1011 prepared100 EBO"))?;
    let output = eye_geometry::row19_lower_eyelid(
        &prepared_points,
        Some(param_list),
        eye_geometry::Point::new(center[0], center[1]),
        controls[0],
        controls[1],
        Some(&indices),
    )
    .map_err(|_| Refusal::ShapeMismatch("Type1011 prepared100 EBO"))?;
    let source_uv = prepared
        .iter()
        .map(|point| {
            [
                point[0] / source.width as f32,
                point[1] / source.height as f32,
            ]
        })
        .collect();
    let positions = output
        .target
        .iter()
        .map(|point| [point.x, point.y])
        .collect();
    let part = match pf {
        30073 => 6,
        30079 => 7,
        _ => return Err(Refusal::ValueOutOfRange("Type1011 PF")),
    };
    let draw = face_mesh::Draw::from_type1011(
        part,
        pf,
        controls,
        sha256_32(&source.pixels),
        prepared,
        positions,
        source_uv,
        indices,
    )?;
    let draws = [&draw];

    #[cfg(windows)]
    {
        if !prefer_cuda(pf) {
            let output = face_mesh::rasterize(source, &draws)?;
            let changed = output != *source;
            return Ok(FaceGeometryRun::cpu(changed.then_some(output)));
        }
        let cuda = match super::forced_cuda_failure() {
            Some(error) => Err(error),
            None => MeshRaster::open()
                .and_then(|gpu| face_mesh::rasterize_cuda_with(&gpu, source, &draws)),
        };
        match cuda {
            Ok(output) => {
                let changed = output.image != *source;
                return Ok(FaceGeometryRun {
                    image: changed.then_some(output.image),
                    backend: PortraitBackend::Cuda,
                    dispatches: output.dispatches,
                    fallback_reason: None,
                });
            }
            Err(error) => {
                let fallback = format!("CUDA Type1011 face mesh fallback: {error:?}");
                let output = face_mesh::rasterize(source, &draws)?;
                let changed = output != *source;
                return Ok(FaceGeometryRun {
                    image: changed.then_some(output),
                    backend: PortraitBackend::Cpu,
                    dispatches: Vec::new(),
                    fallback_reason: Some(fallback),
                });
            }
        }
    }
    #[cfg(not(windows))]
    {
        let output = face_mesh::rasterize(source, &draws)?;
        let changed = output != *source;
        Ok(FaceGeometryRun::cpu(changed.then_some(output)))
    }
}

pub(super) fn apply(
    source: &Image,
    region: NormRect,
    job_id: Option<u64>,
    input: Option<&FaceGeometryInputs>,
    shared: Option<&SlimProducerInputs>,
    pf: u32,
    controls: [f32; 2],
) -> Result<FaceGeometryRun, Refusal> {
    apply_with_mode(source, region, job_id, input, shared, pf, controls, false, None)
}

pub(super) fn apply_with_stage_sha(
    source: &Image,
    region: NormRect,
    job_id: Option<u64>,
    input: Option<&FaceGeometryInputs>,
    shared: Option<&SlimProducerInputs>,
    pf: u32,
    controls: [f32; 2],
    stage_sha256: Option<[u8; 32]>,
) -> Result<FaceGeometryRun, Refusal> {
    apply_with_mode(
        source,
        region,
        job_id,
        input,
        shared,
        pf,
        controls,
        false,
        stage_sha256,
    )
}

pub(super) fn apply_cpu(
    source: &Image,
    region: NormRect,
    job_id: Option<u64>,
    input: Option<&FaceGeometryInputs>,
    shared: Option<&SlimProducerInputs>,
    pf: u32,
    controls: [f32; 2],
) -> Result<FaceGeometryRun, Refusal> {
    apply_with_mode(source, region, job_id, input, shared, pf, controls, true, None)
}

fn apply_with_mode(
    source: &Image,
    region: NormRect,
    job_id: Option<u64>,
    input: Option<&FaceGeometryInputs>,
    shared: Option<&SlimProducerInputs>,
    pf: u32,
    controls: [f32; 2],
    cpu_only: bool,
    stage_sha256: Option<[u8; 32]>,
) -> Result<FaceGeometryRun, Refusal> {
    let min = if pf == 8033 { 0.0 } else { -1.0 };
    if controls.iter().any(|value| !(min..=1.0).contains(value)) {
        return Err(Refusal::ValueOutOfRange("face geometry UI"));
    }
    if controls == [0.0; 2] {
        return Ok(FaceGeometryRun::skipped());
    }
    let digest = stage_sha256.unwrap_or_else(|| sha256_32(&source.pixels));
    let input = input.ok_or(Refusal::MissingInput(missing_draw(pf)))?;
    if input.job_id == 0 || Some(input.job_id) != job_id {
        return Err(Refusal::ValueOutOfRange("face geometry job ticket"));
    }
    if region != NormRect::FULL || (input.width, input.height) != (source.width, source.height) {
        return Err(Refusal::ShapeMismatch("face geometry frame/region"));
    }
    if matches!(pf, 30080..=30083) {
        return Err(Refusal::MissingInput(missing_draw(pf)));
    }
    let mut draws = input.draws.iter().filter(|draw| draw.pf == pf);
    let draw = draws
        .next()
        .ok_or(Refusal::MissingInput(missing_draw(pf)))?;
    if draws.next().is_some() || draw.controls != controls {
        return Err(Refusal::ValueOutOfRange("face geometry draw controls"));
    }
    if draw.source_sha256 != digest {
        return Err(Refusal::ValueOutOfRange("face geometry frame digest"));
    }
    let v2 = || -> Result<&SlimProducerInputs, Refusal> {
        let shared = shared.ok_or(Refusal::MissingInput("shared Slim V2 inputs"))?;
        shared.validate_ticket(source, region, job_id)?;
        Ok(shared)
    };
    let output = match (&draw.geometry, pf) {
        (FaceGeometry::V2Positions(replacements), 242 | 8023 | 8027 | 8029) => {
            let shared = v2()?;
            let mut positions = shared.framebuffer_positions_w1.clone();
            for &(index, point) in replacements {
                *positions
                    .get_mut(usize::from(index))
                    .ok_or(Refusal::ValueOutOfRange("face V2 vertex index"))? = point;
            }
            let mesh = cuda_v2warp::Mesh {
                framebuffer_positions_w1: &positions,
                ..shared.mesh()
            };
            render_v2(
                source,
                OffsetTexture {
                    width: cuda_vface::OFFSET_EDGE,
                    height: cuda_vface::OFFSET_EDGE,
                    pixels: &shared.base,
                },
                mesh,
                shared.rules,
                shared.mat2,
                shared.f_o,
            )?
        }
        (
            FaceGeometry::VFace {
                rows,
                encoded_offset,
                atlases,
            },
            247,
        ) => {
            let shared = v2()?;
            let mut active = 0;
            for row in cuda_vface::ROWS {
                let degree = cuda_vface::degree_for(row, controls[0] * 100.0, controls[1] * 100.0)?;
                if degree == 0.0 {
                    continue;
                }
                active += 1;
                if !rows.contains(&row) {
                    return Err(Refusal::MissingInput("V-face active atlas"));
                }
            }
            if rows.len() != active {
                return Err(Refusal::ShapeMismatch("V-face active atlas bindings"));
            }
            let fused = if atlases.is_empty() {
                encoded_offset.clone()
            } else {
                let mut fused = shared.base.clone();
                for row in cuda_vface::ROWS {
                    let degree =
                        cuda_vface::degree_for(row, controls[0] * 100.0, controls[1] * 100.0)?;
                    if degree == 0.0 {
                        continue;
                    }
                    let atlas = atlases
                        .iter()
                        .find(|atlas| atlas.row == row)
                        .ok_or(Refusal::MissingInput("V-face active atlas bytes"))?;
                    atlas.validate()?;
                    fused = cuda_vface::fuse_offset_f32_cpu(
                        &fused,
                        cuda_vface::Atlas {
                            row: atlas.row,
                            width: atlas.width,
                            height: atlas.height,
                            pixels: &atlas.pixels,
                        },
                        degree,
                    )?;
                }
                if !encoded_offset.is_empty() && encoded_offset != &fused {
                    return Err(Refusal::ValueOutOfRange(
                        "V-face atlas/encoded offset binding",
                    ));
                }
                fused
            };
            render_v2(
                source,
                OffsetTexture {
                    width: cuda_vface::OFFSET_EDGE,
                    height: cuda_vface::OFFSET_EDGE,
                    pixels: &fused,
                },
                shared.mesh(),
                shared.rules,
                shared.mat2,
                shared.f_o,
            )?
        }
        (
            FaceGeometry::Type1011 {
                prepared,
                param_list,
                center,
            },
            30073 | 30079,
        ) => apply_type1011(
            source,
            pf,
            controls,
            shared,
            prepared,
            param_list,
            *center,
        )?,
        (
            FaceGeometry::Eye {
                dest,
                uv,
                ebo,
                rules,
                edge,
                defines,
                clamp,
            },
            _,
        ) if is_eye_draw(pf) => apply_eye(
                source,
            pf,
                    dest,
            uv,
                    ebo,
                *rules,
                *edge,
                *defines,
                *clamp,
            prefer_cuda(pf),
        )?,
        (
            FaceGeometry::HairLine {
                dest,
                uv,
                ebo,
                rules,
                edge,
                defines,
                clamp,
            },
            202,
        ) => render_mesh(
            source, dest, uv, ebo, *rules, *edge, *defines, *clamp,
        )?,
        (
            FaceGeometry::Skull {
                anchor,
                source: original,
                warped,
            },
            232,
        ) => render_skull(source, *anchor, original, warped)?,
        (
            FaceGeometry::Smile {
                dx,
                dy,
                mask,
                host_split_receipt,
            },
            8033,
        ) => {
            let receipt = host_split_receipt
                .as_ref()
                .ok_or(Refusal::MissingInput("PF8033 host split receipt"))?;
            if receipt.job_id != input.job_id
                || receipt.model_output_id != super::SmileRgbReconstructionField::OUTPUT_ID
                || receipt.dx_sha256 != hash_f32(dx)
                || receipt.dy_sha256 != hash_f32(dy)
                || receipt.mask_sha256 != sha256_32(mask)
            {
                return Err(Refusal::ValueOutOfRange("PF8033 host split receipt"));
            }
            // The warp samples each color lane independently and copies alpha,
            // so the BGRA-named adapter also preserves RGBA channel order.
            let cpu = || {
                let pixels = smile_warp::pf8033_dense_warp(
                    &source.pixels,
                    source.width,
                    source.height,
                    dx,
                    dy,
                    mask,
                    controls[0],
                )
                .map_err(smile_refusal)?;
                Image::new(source.width, source.height, pixels)
                    .map(|image| FaceGeometryRun::cpu(Some(image)))
                    .ok_or(Refusal::ShapeMismatch("PF8033 output"))
            };
            if cpu_only {
                cpu()?
            } else {
                #[cfg(windows)]
                {
                    let cuda = match super::forced_cuda_failure() {
                        Some(error) => Err(error),
                        None => aurora_portrait::cuda_smile::Smile::open().and_then(|gpu| {
                            gpu.apply_pf8033(
                                &source.pixels,
                                source.width,
                                source.height,
                                dx,
                                dy,
                                mask,
                                controls[0],
                            )
                        }),
                    };
                    match cuda {
                        Ok((pixels, dispatches)) => {
                            let image = Image::new(source.width, source.height, pixels)
                                .ok_or(Refusal::ShapeMismatch("PF8033 output"))?;
                            let changed = image != *source;
                            FaceGeometryRun {
                                image: changed.then_some(image),
                                backend: PortraitBackend::Cuda,
                                dispatches,
                                fallback_reason: None,
                            }
                        }
                        Err(error) => {
                            let mut run = cpu()?;
                            run.backend = PortraitBackend::Cpu;
                            run.fallback_reason = Some(format!("CUDA PF8033 fallback: {error:?}"));
                            run
                        }
                    }
                }
                #[cfg(not(windows))]
                {
                    cpu()?
                }
            }
        }
        _ => return Err(Refusal::ValueOutOfRange("face geometry draw kind")),
    };
    if let Some(image) = output.image.as_ref() {
        if *image != *source {
            return Ok(output);
        }
    }
    Ok(FaceGeometryRun {
        image: None,
        ..output
    })
}

#[cfg(test)]
#[path = "portrait_face_stage_tests.rs"]
mod tests;
