use std::collections::{BTreeMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

use aurora_portrait::kernels::body_geometry::{
    self as geometry, Brush, Mesh, Pass, PassDefines, Vec2,
};
use aurora_portrait::kernels::hair_body::{self, PackedShiftMap};
use aurora_portrait::kernels::{leg_refine, leg_refine_uv::LegColumn};
use aurora_portrait::{Image, Refusal};

/// Producer payload for the renderer ticket returned for these exact controls,
/// source, region and output dimensions. All fields use the output-frame space.
#[derive(Clone, Debug, PartialEq)]
pub struct BodyGeometryJob {
    pub job_id: u64,
    pub frame_size: (u32, u32),
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

impl BodyGeometryJob {
    pub(super) fn fingerprint(&self) -> u64 {
        let mut hash = DefaultHasher::new();
        self.job_id.hash(&mut hash);
        self.frame_size.hash(&mut hash);
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
        (401 | 406 | 408 | 410 | 415, Mesh { mesh, passes }) => {
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
