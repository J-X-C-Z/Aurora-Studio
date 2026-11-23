use std::collections::{BTreeMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

use aurora_portrait::kernels::body_geometry::{
    self as geometry, Brush, Mesh, Pass, PassDefines, Vec2,
};
use aurora_portrait::kernels::hair_body::{self, PackedShiftMap};
use aurora_portrait::kernels::{leg_refine, leg_refine_uv::LegColumn};
use aurora_portrait::{Image, Refusal};

use super::sha256_32;
use crate::NormRect;

/// Producer payload for the renderer ticket returned for these exact controls,
/// source, region and output dimensions. All fields use the output-frame space.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyGeometryJob {
    pub job_id: u64,
    pub frame_size: (u32, u32),
    /// SHA-256 of the exact RGBA8 source bytes for this ticket.
    pub source_sha256: [u8; 32],
    pub effects: BTreeMap<u32, BodyGeometryInput>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SixPointWarp {
    pub source_points: [Vec2; 6],
    pub destination_points: [Vec2; 6],
    pub degrees: [f32; 6],
    /// Native strength for PF400, native scale for PF404.
    pub amount: f32,
    pub inverse_aspect: f32,
    pub valid_point_count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BodyShiftInput {
    pub mask: Vec<u8>,
    pub map_size: (u32, u32),
    pub rgba8: Vec<u8>,
    pub ratio: f32,
    pub shift_size: (u32, u32),
}

/// Buffers and native uniforms are prepared for the requested control value.
/// Mesh passes must be final-color draws in submission order.
#[derive(Clone, Debug, PartialEq)]
pub enum BodyGeometryInput {
    SixPoint(SixPointWarp),
    Mesh {
        mesh: Mesh,
        passes: Vec<Pass>,
    },
    BodySlim {
        mask_r: Vec<u8>,
        source_uv: Vec<Vec2>,
        target_uv: Vec<Vec2>,
    },
    Leg {
        columns: [LegColumn; 5],
        catmull_size: Option<(f32, f32)>,
    },
    Height {
        source_uv: Vec<Vec2>,
        destination_uv: Vec<Vec2>,
        defines: PassDefines,
    },
    Waist {
        warped_uv: Vec<Vec2>,
        mask_r: Vec<f32>,
    },
    BreastLift {
        degree01: f32,
        brushes: Vec<Brush>,
        density: Vec<f32>,
        density_size: (u32, u32),
        freeze_mask: Vec<u8>,
        warp_direction: f32,
        edge_margin: f32,
        push_left: bool,
    },
    Shift(BodyShiftInput),
}

/// The provider never chooses between body and face landmarks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LandmarkDomain {
    Body,
    Face,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyVboLayout {
    /// destination x/y then source UV x/y, little-endian f32.
    DestinationUvF32x4Le,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyGeometryEvidence<'a> {
    pub landmark_domain: Option<LandmarkDomain>,
    pub landmarks: Option<&'a [Vec2; 6]>,
    pub body_mask: Option<&'a [u8]>,
    pub mesh: Option<&'a Mesh>,
    pub passes: Option<&'a [Pass]>,
    pub vbo_layout: Option<BodyVboLayout>,
    pub vbo: Option<&'a [u8]>,
    pub ebo: Option<&'a [u8]>,
    pub brushes: Option<&'a [Brush]>,
    pub density: Option<&'a [f32]>,
}

pub struct BodyGeometryBuildRow<'a> {
    pub pf: u32,
    pub ticket: u64,
    pub frame_size: (u32, u32),
    pub source_sha256: [u8; 32],
    pub region: Option<NormRect>,
    pub input: Option<&'a BodyGeometryInput>,
    pub evidence: BodyGeometryEvidence<'a>,
}

pub struct BodyGeometryBuildFrame<'a> {
    pub ticket: u64,
    pub frame_size: (u32, u32),
    pub source_sha256: [u8; 32],
    pub region: NormRect,
    pub rows: &'a [BodyGeometryBuildRow<'a>],
}

impl BodyGeometryJob {
    /// Build the producer payload from explicit same-ticket, same-frame rows.
    ///
    /// This function never creates a mask, landmark, mesh, VBO, EBO or pass.
    pub fn build(source: &Image, frame: &BodyGeometryBuildFrame<'_>) -> Result<Self, Refusal> {
        if frame.ticket == 0 {
            return Err(Refusal::ValueOutOfRange("body geometry provider ticket"));
        }
        if frame.frame_size != (source.width, source.height) {
            return Err(Refusal::ShapeMismatch("body geometry provider frame"));
        }
        if frame.source_sha256.iter().all(|byte| *byte == 0) {
            return Err(Refusal::MissingInput(
                "body geometry provider frame SHA-256",
            ));
        }
        if frame.source_sha256 != sha256_32(&source.pixels) {
            return Err(Refusal::ValueOutOfRange(
                "body geometry provider frame digest",
            ));
        }
        if frame.region != NormRect::FULL {
            return Err(Refusal::ValueOutOfRange("body geometry provider ROI"));
        }
        if frame.rows.is_empty() {
            return Err(Refusal::MissingInput("body geometry provider rows"));
        }

        let pixel_count = usize::try_from(source.width)
            .ok()
            .and_then(|width| {
                usize::try_from(source.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or(Refusal::ShapeMismatch("body geometry provider frame"))?;
        let mut effects = BTreeMap::new();
        for row in frame.rows {
            if row.ticket != frame.ticket {
                return Err(Refusal::ValueOutOfRange(
                    "body geometry provider row ticket",
                ));
            }
            if row.frame_size != frame.frame_size {
                return Err(Refusal::ShapeMismatch("body geometry provider row frame"));
            }
            if row.source_sha256 != frame.source_sha256 {
                return Err(Refusal::ValueOutOfRange(
                    "body geometry provider row digest",
                ));
            }
            if row.region != Some(NormRect::FULL) {
                return Err(Refusal::ValueOutOfRange("body geometry provider row ROI"));
            }
            if !supports(row.pf) {
                return Err(Refusal::ValueOutOfRange("body geometry provider PF"));
            }
            if effects.contains_key(&row.pf) {
                return Err(Refusal::ValueOutOfRange(
                    "body geometry provider duplicate PF",
                ));
            }
            let input = row
                .input
                .ok_or(Refusal::MissingInput("body geometry provider input"))?;
            effects.insert(
                row.pf,
                build_row(row.pf, input, &row.evidence, pixel_count)?,
            );
        }

        Ok(Self {
            job_id: frame.ticket,
            frame_size: frame.frame_size,
            source_sha256: frame.source_sha256,
            effects,
        })
    }

    pub(super) fn fingerprint(&self) -> u64 {
        let mut hash = DefaultHasher::new();
        self.job_id.hash(&mut hash);
        self.frame_size.hash(&mut hash);
        self.source_sha256.hash(&mut hash);
        self.effects.len().hash(&mut hash);
        for (pf, input) in &self.effects {
            pf.hash(&mut hash);
            std::mem::discriminant(input).hash(&mut hash);
            match input {
                BodyGeometryInput::SixPoint(warp) => {
                    uv_hash(&mut hash, &warp.source_points);
                    uv_hash(&mut hash, &warp.destination_points);
                    float_hash(&mut hash, &warp.degrees);
                    float_hash(&mut hash, &[warp.amount, warp.inverse_aspect]);
                    warp.valid_point_count.hash(&mut hash);
                }
                BodyGeometryInput::Mesh { mesh, passes } => {
                    uv_hash(&mut hash, &mesh.source_uv);
                    uv_hash(&mut hash, &mesh.destination);
                    mesh.indices.hash(&mut hash);
                    passes.len().hash(&mut hash);
                    for pass in passes {
                        (pass.edge as u8, pass.clamp.protect_x, pass.clamp.protect_y)
                            .hash(&mut hash);
                        defines_hash(&mut hash, pass.defines);
                        float_hash(&mut hash, &pass.matrix);
                        pass.clamp.factor.to_bits().hash(&mut hash);
                    }
                }
                BodyGeometryInput::BodySlim {
                    mask_r,
                    source_uv,
                    target_uv,
                } => {
                    mask_r.hash(&mut hash);
                    uv_hash(&mut hash, source_uv);
                    uv_hash(&mut hash, target_uv);
                }
                BodyGeometryInput::Leg {
                    columns,
                    catmull_size,
                } => {
                    for column in columns {
                        float_hash(&mut hash, &[column.sample_u, column.clip_x]);
                    }
                    catmull_size
                        .map(|(x, y)| (x.to_bits(), y.to_bits()))
                        .hash(&mut hash);
                }
                BodyGeometryInput::Height {
                    source_uv,
                    destination_uv,
                    defines,
                } => {
                    uv_hash(&mut hash, source_uv);
                    uv_hash(&mut hash, destination_uv);
                    defines_hash(&mut hash, *defines);
                }
                BodyGeometryInput::Waist { warped_uv, mask_r } => {
                    uv_hash(&mut hash, warped_uv);
                    float_hash(&mut hash, mask_r);
                }
                BodyGeometryInput::BreastLift {
                    degree01,
                    brushes,
                    density,
                    density_size,
                    freeze_mask,
                    warp_direction,
                    edge_margin,
                    push_left,
                } => {
                    float_hash(&mut hash, &[*degree01, *warp_direction, *edge_margin]);
                    brushes.len().hash(&mut hash);
                    for brush in brushes {
                        uv_hash(&mut hash, &[brush.position, brush.direction]);
                        float_hash(&mut hash, &[brush.radius, brush.pressure]);
                    }
                    float_hash(&mut hash, density);
                    density_size.hash(&mut hash);
                    freeze_mask.hash(&mut hash);
                    push_left.hash(&mut hash);
                }
                BodyGeometryInput::Shift(shift) => {
                    shift.mask.hash(&mut hash);
                    shift.map_size.hash(&mut hash);
                    shift.rgba8.hash(&mut hash);
                    shift.ratio.to_bits().hash(&mut hash);
                    shift.shift_size.hash(&mut hash);
                }
            }
        }
        hash.finish()
    }
}

fn build_row(
    pf: u32,
    input: &BodyGeometryInput,
    evidence: &BodyGeometryEvidence<'_>,
    pixel_count: usize,
) -> Result<BodyGeometryInput, Refusal> {
    use BodyGeometryInput::*;
    match (pf, input) {
        (400 | 404, SixPoint(warp)) => {
            evidence
                .landmark_domain
                .ok_or(Refusal::MissingInput("body geometry landmark domain"))?;
            let landmarks = evidence
                .landmarks
                .ok_or(Refusal::MissingInput("body geometry landmarks"))?;
            if warp.source_points != *landmarks {
                return Err(Refusal::ValueOutOfRange(
                    "body geometry landmark source points",
                ));
            }
            if warp.valid_point_count == 0 || warp.valid_point_count > 6 {
                return Err(Refusal::ValueOutOfRange("body geometry valid point count"));
            }
            if !warp.amount.is_finite() || !warp.inverse_aspect.is_finite() {
                return Err(Refusal::ValueOutOfRange("body geometry six point uniforms"));
            }
            if warp
                .destination_points
                .iter()
                .any(|point| !point.x.is_finite() || !point.y.is_finite())
                || warp.degrees.iter().any(|degree| !degree.is_finite())
            {
                return Err(Refusal::ValueOutOfRange("body geometry six point data"));
            }
            Ok(SixPoint(warp.clone()))
        }
        (401 | 406 | 410 | 415, Mesh { mesh, passes }) => {
            let evidence_mesh = evidence
                .mesh
                .ok_or(Refusal::MissingInput("body geometry mesh"))?;
            let evidence_passes = evidence
                .passes
                .ok_or(Refusal::MissingInput("body geometry passes"))?;
            if evidence_mesh != mesh || evidence_passes != passes.as_slice() {
                return Err(Refusal::ValueOutOfRange(
                    "body geometry mesh/passes evidence",
                ));
            }
            if evidence.vbo_layout != Some(BodyVboLayout::DestinationUvF32x4Le) {
                return Err(Refusal::MissingInput("body geometry VBO layout"));
            }
            let vbo = evidence
                .vbo
                .ok_or(Refusal::MissingInput("body geometry VBO"))?;
            let ebo = evidence
                .ebo
                .ok_or(Refusal::MissingInput("body geometry EBO"))?;
            let built = mesh_from_vbo_ebo(vbo, ebo)?;
            if &built != evidence_mesh || &built != mesh {
                return Err(Refusal::ValueOutOfRange("body geometry VBO/EBO"));
            }
            if passes.is_empty() || passes.iter().any(|pass| pass.defines.uv_map) {
                return Err(Refusal::ValueOutOfRange("body geometry color passes"));
            }
            Ok(Mesh {
                mesh: built,
                passes: passes.clone(),
            })
        }
        (
            405,
            BodySlim {
                mask_r,
                source_uv,
                target_uv,
            },
        ) => {
            let body_mask = evidence
                .body_mask
                .ok_or(Refusal::MissingInput("body geometry body mask"))?;
            if body_mask != mask_r.as_slice()
                || source_uv.len() != pixel_count
                || target_uv.len() != pixel_count
            {
                return Err(Refusal::ShapeMismatch("body geometry body slim inputs"));
            }
            Ok(BodySlim {
                mask_r: mask_r.clone(),
                source_uv: source_uv.clone(),
                target_uv: target_uv.clone(),
            })
        }
        (
            407,
            Leg {
                columns,
                catmull_size,
            },
        ) => Ok(Leg {
            columns: columns.clone(),
            catmull_size: *catmull_size,
        }),
        (
            408,
            Height {
                source_uv,
                destination_uv,
                defines,
            },
        ) => {
            if defines.uv_map {
                return Err(Refusal::ValueOutOfRange("body geometry color passes"));
            }
            if source_uv.len() != pixel_count || destination_uv.len() != pixel_count {
                return Err(Refusal::ShapeMismatch("body geometry height UV"));
            }
            Ok(Height {
                source_uv: source_uv.clone(),
                destination_uv: destination_uv.clone(),
                defines: *defines,
            })
        }
        (416, Waist { warped_uv, mask_r }) => {
            let body_mask = evidence
                .body_mask
                .ok_or(Refusal::MissingInput("body geometry body mask"))?;
            if body_mask.len() != pixel_count
                || warped_uv.len() != pixel_count
                || mask_r.len() != pixel_count
            {
                return Err(Refusal::ShapeMismatch("body geometry waist inputs"));
            }
            Ok(Waist {
                warped_uv: warped_uv.clone(),
                mask_r: mask_r.clone(),
            })
        }
        (
            38556,
            BreastLift {
                degree01,
                brushes,
                density,
                density_size,
                freeze_mask,
                warp_direction,
                edge_margin,
                push_left,
            },
        ) => {
            let evidence_brushes = evidence
                .brushes
                .ok_or(Refusal::MissingInput("body geometry brushes"))?;
            let evidence_density = evidence
                .density
                .ok_or(Refusal::MissingInput("body geometry density"))?;
            if evidence_brushes != brushes.as_slice() || evidence_density != density.as_slice() {
                return Err(Refusal::ValueOutOfRange(
                    "body geometry brush/density evidence",
                ));
            }
            let density_pixels = usize::try_from(density_size.0)
                .ok()
                .and_then(|width| {
                    usize::try_from(density_size.1)
                        .ok()
                        .and_then(|height| width.checked_mul(height))
                })
                .ok_or(Refusal::ShapeMismatch("body geometry density"))?;
            if brushes.is_empty()
                || density.len() != density_pixels
                || freeze_mask.len() != pixel_count
                || !degree01.is_finite()
                || !warp_direction.is_finite()
                || !edge_margin.is_finite()
            {
                return Err(Refusal::ShapeMismatch("body geometry breast lift inputs"));
            }
            Ok(BreastLift {
                degree01: *degree01,
                brushes: brushes.clone(),
                density: density.clone(),
                density_size: *density_size,
                freeze_mask: freeze_mask.clone(),
                warp_direction: *warp_direction,
                edge_margin: *edge_margin,
                push_left: *push_left,
            })
        }
        (90041, Shift(shift)) => {
            let body_mask = evidence
                .body_mask
                .ok_or(Refusal::MissingInput("body geometry body mask"))?;
            if body_mask != shift.mask.as_slice()
                || shift.map_size.0 == 0
                || shift.map_size.1 == 0
                || shift.shift_size.0 == 0
                || shift.shift_size.1 == 0
                || !shift.ratio.is_finite()
            {
                return Err(Refusal::ShapeMismatch("body geometry shift inputs"));
            }
            let map_pixels = usize::try_from(shift.map_size.0)
                .ok()
                .and_then(|width| {
                    usize::try_from(shift.map_size.1)
                        .ok()
                        .and_then(|height| width.checked_mul(height))
                })
                .and_then(|pixels| pixels.checked_mul(4))
                .ok_or(Refusal::ShapeMismatch("body geometry shift inputs"))?;
            if body_mask.len() != pixel_count || shift.rgba8.len() != map_pixels {
                return Err(Refusal::ShapeMismatch("body geometry shift inputs"));
            }
            Ok(Shift(shift.clone()))
        }
        _ => Err(Refusal::ShapeMismatch("body geometry provider kind")),
    }
}

fn mesh_from_vbo_ebo(vbo: &[u8], ebo: &[u8]) -> Result<Mesh, Refusal> {
    if vbo.is_empty() {
        return Err(Refusal::MissingInput("body geometry VBO"));
    }
    if vbo.len() % 16 != 0 {
        return Err(Refusal::ShapeMismatch("body geometry VBO"));
    }
    if ebo.is_empty() {
        return Err(Refusal::MissingInput("body geometry EBO"));
    }
    if ebo.len() % 2 != 0 {
        return Err(Refusal::ShapeMismatch("body geometry EBO"));
    }

    let mut destination = Vec::with_capacity(vbo.len() / 16);
    let mut source_uv = Vec::with_capacity(vbo.len() / 16);
    for vertex in vbo.chunks_exact(16) {
        let value = |offset: usize| {
            f32::from_le_bytes(
                vertex[offset..offset + 4]
                    .try_into()
                    .expect("fixed VBO field"),
            )
        };
        let values = [value(0), value(4), value(8), value(12)];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("body geometry VBO"));
        }
        destination.push(Vec2::new(values[0], values[1]));
        source_uv.push(Vec2::new(values[2], values[3]));
    }
    let indices = ebo
        .chunks_exact(2)
        .map(|index| u16::from_le_bytes([index[0], index[1]]))
        .collect();
    Mesh::new(destination, source_uv, indices)
        .map_err(|_| Refusal::ValueOutOfRange("body geometry VBO/EBO"))
}

fn float_hash(hash: &mut DefaultHasher, values: &[f32]) {
    values.len().hash(hash);
    for value in values {
        value.to_bits().hash(hash);
    }
}

fn uv_hash(hash: &mut DefaultHasher, values: &[Vec2]) {
    values.len().hash(hash);
    for value in values {
        value.x.to_bits().hash(hash);
        value.y.to_bits().hash(hash);
    }
}

fn defines_hash(hash: &mut DefaultHasher, defines: PassDefines) {
    (
        defines.clamp_edge,
        defines.uv_map,
        defines.uv_map_ba,
        defines.catmull_rom,
    )
        .hash(hash);
}

pub(super) fn supports(pf: u32) -> bool {
    matches!(pf, 400 | 401 | 404..=408 | 410 | 415 | 416 | 38556 | 90041)
}

/// Payload kind a routed row accepts. Mirrors the dispatch in [`apply`] so a job
/// whose payload kind disagrees with its row is refused when it is written.
/// Kind-level only: every kernel still validates its own streams at dispatch,
/// and nothing here synthesizes a missing mask, mesh, UV or control point.
pub(super) fn accepts(pf: u32, input: &BodyGeometryInput) -> bool {
    use BodyGeometryInput::*;
    matches!(
        (pf, input),
        (400 | 404, SixPoint(_))
            | (401 | 406 | 408 | 410 | 415, Mesh { .. })
            | (405, BodySlim { .. })
            | (407, Leg { .. })
            | (408, Height { .. })
            | (416, Waist { .. })
            | (38556, BreastLift { .. })
            | (90041, Shift(_))
    )
}

pub(super) fn apply(
    source: &Image,
    pf: u32,
    ui: f32,
    job_id: Option<u64>,
    job: Option<&BodyGeometryJob>,
) -> Result<Option<Image>, Refusal> {
    if !ui.is_finite() {
        return Err(Refusal::ValueOutOfRange("body geometry control"));
    }
    if ui == 0.0 {
        return Ok(None);
    }
    let job = job.ok_or(Refusal::MissingInput("body geometry job"))?;
    if job.job_id == 0 || job_id != Some(job.job_id) {
        return Err(Refusal::ValueOutOfRange("body geometry job id"));
    }
    if job.frame_size != (source.width, source.height) {
        return Err(Refusal::ShapeMismatch("body geometry frame"));
    }
    if job.source_sha256.iter().all(|byte| *byte == 0) {
        return Err(Refusal::MissingInput("body geometry frame SHA-256"));
    }
    if job.source_sha256 != sha256_32(&source.pixels) {
        return Err(Refusal::ValueOutOfRange("body geometry frame digest"));
    }
    let input = job
        .effects
        .get(&pf)
        .ok_or(Refusal::MissingInput("body geometry effect"))?;
    use BodyGeometryInput::*;
    match (pf, input) {
        (400 | 404, SixPoint(warp)) => {
            let kernel = if pf == 400 {
                geometry::slim_arm
            } else {
                geometry::hip_enhancement
            };
            kernel(
                source,
                &warp.source_points,
                &warp.destination_points,
                &warp.degrees,
                warp.amount,
                warp.inverse_aspect,
                warp.valid_point_count,
            )
        }
        (401 | 406 | 410 | 415, Mesh { mesh, passes }) => {
            if passes.iter().any(|pass| pass.defines.uv_map) {
                return Err(Refusal::ValueOutOfRange(
                    "body geometry requires color passes",
                ));
            }
            geometry::reshape_mesh(source, mesh, passes)
        }
        (
            405,
            BodySlim {
                mask_r,
                source_uv,
                target_uv,
            },
        ) => hair_body::body_slim_warp_field(source, mask_r, source_uv, target_uv),
        (
            407,
            Leg {
                columns,
                catmull_size,
            },
        ) => leg_refine::apply(source, columns, *catmull_size),
        (
            408,
            Height {
                source_uv,
                destination_uv,
                defines,
            },
        ) => {
            if defines.uv_map {
                return Err(Refusal::ValueOutOfRange(
                    "body geometry requires color passes",
                ));
            }
            geometry::height_increase(source, source_uv, destination_uv, *defines)
        }
        (416, Waist { warped_uv, mask_r }) => geometry::waist_length(source, warped_uv, mask_r),
        (
            38556,
            BreastLift {
                degree01,
                brushes,
                density,
                density_size,
                freeze_mask,
                warp_direction,
                edge_margin,
                push_left,
            },
        ) => {
            if *degree01 == 0.0 {
                return Ok(None);
            }
            geometry::breast_lift(
                source,
                *degree01,
                brushes,
                density,
                *density_size,
                freeze_mask,
                *warp_direction,
                *edge_margin,
                *push_left,
            )
        }
        (90041, Shift(shift)) => {
            if !(-100.0..=100.0).contains(&ui) || ui.fract() != 0.0 {
                return Err(Refusal::ValueOutOfRange("PF90041_UI"));
            }
            hair_body::body_shift(
                source,
                ui as i32,
                &shift.mask,
                &PackedShiftMap {
                    width: shift.map_size.0,
                    height: shift.map_size.1,
                    rgba8: &shift.rgba8,
                    ratio: shift.ratio,
                    shift_width: shift.shift_size.0,
                    shift_height: shift.shift_size.1,
                },
            )
        }
        _ => Err(Refusal::ShapeMismatch("body geometry effect kind")),
    }
    .map(Some)
}
