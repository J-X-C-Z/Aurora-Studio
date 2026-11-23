use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use aurora_portrait::cuda::Dispatch;
use aurora_portrait::cuda_v2warp::{self, OffsetTexture};
use aurora_portrait::cuda_vface::{self, Row};
use aurora_portrait::kernels::{eye_geometry, smile_warp};
use aurora_portrait::{Image, Refusal, raster, raster_fragment};

use super::{PortraitBackend, SlimProducerInputs, sha256_32, smile_refusal};
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
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct FaceGeometryInputs {
    pub job_id: u64,
    pub width: u32,
    pub height: u32,
    pub draws: Vec<FaceGeometryDraw>,
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
                } => {
                    rows.len().hash(&mut h);
                    for row in rows {
                        (*row as u8).hash(&mut h);
                    }
                    hash_points(encoded_offset, &mut h);
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
                FaceGeometry::Smile { dx, dy, mask } => {
                    for plane in [dx, dy] {
                        plane.len().hash(&mut h);
                        for value in plane {
                            value.to_bits().hash(&mut h);
                        }
                    }
                    mask.hash(&mut h);
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

pub(super) fn pair(pf: u32) -> Option<(u32, u32)> {
    match pf {
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
    match pf {
        202 => Some("PF202 hairLine producer: 194 landmarks + 135 weights + control slots"),
        232 => Some("PF232 skullEnhance producer: landmarks 192..206 + 15 factors"),
        _ => None,
    }
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
        8001..=8022 | 30073 | 30079 | 44124 | 48019 | 49221 | 41922
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

pub(super) fn apply(
    source: &Image,
    region: NormRect,
    job_id: Option<u64>,
    input: Option<&FaceGeometryInputs>,
    shared: Option<&SlimProducerInputs>,
    pf: u32,
    controls: [f32; 2],
) -> Result<FaceGeometryRun, Refusal> {
    let min = if pf == 8033 { 0.0 } else { -1.0 };
    if controls.iter().any(|value| !(min..=1.0).contains(value)) {
        return Err(Refusal::ValueOutOfRange("face geometry UI"));
    }
    if controls == [0.0; 2] {
        return Ok(FaceGeometryRun::skipped());
    }
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
    if draw.source_sha256 != sha256_32(&source.pixels) {
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
            FaceGeometryRun::cpu(Some(cuda_v2warp::apply_cpu(
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
            )?))
        }
        (
            FaceGeometry::VFace {
                rows,
                encoded_offset,
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
            FaceGeometryRun::cpu(Some(cuda_v2warp::apply_cpu(
                source,
                OffsetTexture {
                    width: cuda_vface::OFFSET_EDGE,
                    height: cuda_vface::OFFSET_EDGE,
                    pixels: encoded_offset,
                },
                shared.mesh(),
                shared.rules,
                shared.mat2,
                shared.f_o,
            )?))
        }
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
        (FaceGeometry::Smile { dx, dy, mask }, 8033) => {
            // The warp samples each color lane independently and copies alpha,
            // so the BGRA-named adapter also preserves RGBA channel order.
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
            FaceGeometryRun::cpu(Some(
            Image::new(source.width, source.height, pixels)
                    .ok_or(Refusal::ShapeMismatch("PF8033 output"))?,
            ))
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
