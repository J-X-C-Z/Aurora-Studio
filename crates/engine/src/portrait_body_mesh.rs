//! The 13 Mesh-row geometry generators (module B, `pr-B`).
//!
//! `generate` is the single entry: a row number, the curve-mapped `degree`
//! (relative to neutral, shoulder width keeps its real sign), the validated frame
//! size, this photo's semantic anchors and the frame-sized foreground weight go in;
//! one [`BodyGeometryInput::Mesh`] comes out.
//!
//! Rows: `401` waist, `403` bust, `406` head zoom, `410` neck length, `415` upper
//! proportion, `411`/`412` shoulder square (left/right), `413`/`414` swan neck
//! (left/right), `417`/`418` shoulder width (left/right), `30074` thigh, `30075` calf.
//!
//! ## Reused instead of re-invented
//!
//! * `body_geometry::{Vec2, Mesh, Pass, PassDefines, Clamp, EdgeMode, grid_indices}`
//!   for the 11×11 stream pair and the recovered two-triangle cell order.
//! * `body_region_geometry::{TorsoFrame, RegionBand, WAIST_BAND, BREAST_BAND,
//!   NECK_BAND, band_weight, tube_weight, split_left_right}` for the torso frame,
//!   the calibrated bands and the one-sided split.
//! * `body_exact_vendor` owns the vendor `DegreesScale`/`RadiusScale` mapping
//!   (`shoulder_degree_sets`); that mapping stays with the wiring layer. This module
//!   consumes the already-mapped signed `degree` and never claims to reproduce the
//!   vendor curve.
//!
//! ## What is an engineering approximation
//!
//! Every magnitude constant below is `[D]`: chosen so the field is local,
//! boundary-preserving and monotone in `degree`. They are **not** OEM parity
//! numbers and no such claim is made here.
//!
//! ## Units and orientation
//!
//! `source_uv` and `destination` are both normalised `[0,1]` frame coordinates with
//! the origin top-left and **y growing downward** — the space
//! `body_geometry::height_increase` builds (`(x + 0.5) / width`) and the space
//! `rasterize` inverts through [`Pass::normalized_to_clip`]. `grid_indices` walks
//! rows downward and columns rightward, which is that same order, so the lattice
//! index is `row * MESH_SIDE + column`.
//!
//! The outer ring of the lattice is pinned to identity: a row never opens a hole or
//! a seam at its box edge, and only the interior moves.

use aurora_portrait::kernels::body_geometry::{
    grid_indices, Clamp, EdgeMode, Mesh, Pass, PassDefines, Vec2,
};
use aurora_portrait::kernels::body_pose_adapter::{Anchor, AnchorSet, ObservedKeypoint};
use aurora_portrait::kernels::body_region_geometry::{
    band_weight, split_left_right, tube_weight, RegionBand, TorsoFrame, BREAST_BAND, NECK_BAND,
    WAIST_BAND,
};
use aurora_portrait::Refusal;

use super::portrait_body_stage::BodyGeometryInput;

/// Lattice side. 11×11 = 121 vertices, 200 triangles, 600 indices: the recovered
/// PF401 cell order, reused for every Mesh row here.
pub(super) const MESH_SIDE: usize = 11;

/// Torso-band feather, in torso units (a twelfth of the band height).
const BAND_FEATHER: f32 = 0.08;

/// Below this foreground weight a vertex is left exactly where it was.
const MASK_FLOOR: f32 = 0.02;

// Per-row magnitudes, all `[D]`, all fractions of the region's own size so a 4K
// frame and a proxy frame produce the same shape.
const WAIST_SLIM: f32 = 0.20;
const BUST_OUT: f32 = 0.14;
const BUST_UP: f32 = 0.06;
const HEAD_ZOOM: f32 = 0.20;
const NECK_STRETCH: f32 = 0.14;
const TORSO_PROPORTION: f32 = 0.12;
const SHOULDER_SQUARE: f32 = 0.12;
const SWAN_NECK: f32 = 0.16;
const SHOULDER_WIDTH: f32 = 0.22;
const LIMB_SLIM: f32 = 0.18;

/// Which side of the body a row acts on. From the anchor names, never from
/// `x < 0.5`: a subject facing away still has their own left on the left anchor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Side {
    Left,
    Right,
}

impl Side {
    fn shoulder(self) -> Anchor {
        match self {
            Self::Left => Anchor::LeftShoulder,
            Self::Right => Anchor::RightShoulder,
        }
    }

    /// Weight for the person's own side at torso coordinate `u` (`u < 0` is left).
    fn weight(self, u: f32) -> f32 {
        let (left, right) = split_left_right(u, 0.15);
        match self {
            Self::Left => left,
            Self::Right => right,
        }
    }

    /// Unit sign of "outward" in torso-`u` terms: left is -u, right is +u.
    fn outward(self) -> f32 {
        match self {
            Self::Left => -1.0,
            Self::Right => 1.0,
        }
    }
}

/// The three torso rows, kept apart so each really has its own field.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TorsoRow {
    Waist,
    Bust,
    Proportion,
}

/// What a row displaces.
enum Region {
    Torso { frame: TorsoFrame, band: RegionBand, row: TorsoRow },
    /// The head, as a disc between the nose and the vertex.
    Head { centre: Vec2, radius: f32 },
    /// The neck band, whole (`410`) or one side of it (`413`/`414`).
    Neck { frame: TorsoFrame, band: RegionBand, side: Option<Side> },
    /// One shoulder corner, as a radial field around that anchor.
    Shoulder { frame: TorsoFrame, side: Side, corner: Vec2, radius: f32, width_row: bool },
    /// Both legs, each with its own axis: a row never crosses the gap.
    Limb { segments: [(Vec2, Vec2); 2], radius: f32 },
}

struct Plan {
    region: Region,
    min: Vec2,
    max: Vec2,
}

fn point(observed: ObservedKeypoint) -> Vec2 {
    Vec2::new(observed.x, observed.y)
}

fn anchor_at(anchors: &AnchorSet, anchor: Anchor) -> Result<Vec2, Refusal> {
    anchors.require(anchor).map(point)
}

fn torso_frame(anchors: &AnchorSet) -> Result<TorsoFrame, Refusal> {
    TorsoFrame::new(
        anchor_at(anchors, Anchor::LeftShoulder)?,
        anchor_at(anchors, Anchor::RightShoulder)?,
        anchor_at(anchors, Anchor::LeftHip)?,
        anchor_at(anchors, Anchor::RightHip)?,
    )
}

/// Frame-UV box of a torso band, widened by `u_pad` half-shoulder widths.
fn band_box(frame: &TorsoFrame, band: RegionBand, u_pad: f32) -> Result<(Vec2, Vec2), Refusal> {
    let mut min = Vec2::new(f32::INFINITY, f32::INFINITY);
    let mut max = Vec2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
    for t in [band.t0, band.t1] {
        for u in [band.u0 - u_pad, band.u1 + u_pad] {
            let corner = frame.to_uv(t, u)?;
            min = Vec2::new(min.x.min(corner.x), min.y.min(corner.y));
            max = Vec2::new(max.x.max(corner.x), max.y.max(corner.y));
        }
    }
    if !(max.x - min.x > 1e-3) || !(max.y - min.y > 1e-3) {
        return Err(Refusal::ValueOutOfRange("body mesh band box"));
    }
    Ok((min, max))
}

fn side_of(pf: u32) -> Option<Side> {
    match pf {
        411 | 413 | 417 => Some(Side::Left),
        412 | 414 | 418 => Some(Side::Right),
        _ => None,
    }
}

fn limb_length(a: Vec2, b: Vec2) -> f32 {
    ((b.x - a.x).powi(2) + (b.y - a.y).powi(2)).sqrt()
}

/// Resolve one row's region and lattice box, requiring exactly the anchors that row
/// needs. A missing anchor becomes that anchor's own `MissingInput` label.
fn plan(pf: u32, anchors: &AnchorSet) -> Result<Plan, Refusal> {
    match pf {
        401 | 403 | 415 => {
            let frame = torso_frame(anchors)?;
            let (row, band, pad) = match pf {
                401 => (TorsoRow::Waist, WAIST_BAND, 0.15),
                403 => (TorsoRow::Bust, BREAST_BAND, 0.20),
                _ => (TorsoRow::Proportion, RegionBand::new(0.0, 1.0, -1.2, 1.2)?, 0.10),
            };
            let (min, max) = band_box(&frame, band, pad)?;
            Ok(Plan {
                region: Region::Torso { frame, band, row },
                min,
                max,
            })
        }
        406 => {
            let nose = anchor_at(anchors, Anchor::Nose)?;
            let head_top = anchor_at(anchors, Anchor::HeadTop)?;
            let neck = anchor_at(anchors, Anchor::NeckBase)?;
            let centre = Vec2::new((nose.x + head_top.x) * 0.5, (nose.y + head_top.y) * 0.5);
            let radius = limb_length(head_top, neck);
            if !(radius > 1e-3) {
                return Err(Refusal::ValueOutOfRange("body mesh head radius"));
            }
            let span = radius * 1.15;
            Ok(Plan {
                region: Region::Head { centre, radius },
                min: Vec2::new(centre.x - span, centre.y - span),
                max: Vec2::new(centre.x + span, centre.y + span),
            })
        }
        410 | 413 | 414 => {
            let frame = torso_frame(anchors)?;
            let side = side_of(pf);
            if let Some(side) = side {
                // The sided rows still need that side's own shoulder to exist.
                anchor_at(anchors, side.shoulder())?;
            }
            let (min, max) = band_box(&frame, NECK_BAND, 0.10)?;
            Ok(Plan {
                region: Region::Neck {
                    frame,
                    band: NECK_BAND,
                    side,
                },
                min,
                max,
            })
        }
        411 | 412 | 417 | 418 => {
            let frame = torso_frame(anchors)?;
            let side = side_of(pf).expect("shoulder rows are sided");
            let corner = anchor_at(anchors, side.shoulder())?;
            let axes = frame.axes()?;
            let width_row = matches!(pf, 417 | 418);
            if width_row {
                let band = RegionBand::new(-0.05, 0.30, -1.2, 1.2)?;
                let (min, max) = band_box(&frame, band, 0.15)?;
                Ok(Plan {
                    region: Region::Shoulder {
                        frame,
                        side,
                        corner,
                        radius: axes.shoulder_width * 0.55,
                        width_row,
                    },
                    min,
                    max,
                })
            } else {
                let half = axes.shoulder_width * 0.45;
                let up = axes.torso_height * 0.32;
                Ok(Plan {
                    region: Region::Shoulder {
                        frame,
                        side,
                        corner,
                        radius: half,
                        width_row,
                    },
                    min: Vec2::new(corner.x - half, corner.y - up),
                    max: Vec2::new(corner.x + half, corner.y + up),
                })
            }
        }
        30074 | 30075 => {
            // Both legs, each its own segment: a missing side is an error, not a
            // silent one-leg result.
            let (left, right) = match pf {
                30074 => (
                    (
                        anchor_at(anchors, Anchor::LeftHip)?,
                        anchor_at(anchors, Anchor::LeftKnee)?,
                    ),
                    (
                        anchor_at(anchors, Anchor::RightHip)?,
                        anchor_at(anchors, Anchor::RightKnee)?,
                    ),
                ),
                _ => (
                    (
                        anchor_at(anchors, Anchor::LeftKnee)?,
                        anchor_at(anchors, Anchor::LeftAnkle)?,
                    ),
                    (
                        anchor_at(anchors, Anchor::RightKnee)?,
                        anchor_at(anchors, Anchor::RightAnkle)?,
                    ),
                ),
            };
            let radius = limb_length(left.0, left.1).min(limb_length(right.0, right.1)) * 0.22;
            if !(radius > 1e-3) {
                return Err(Refusal::ValueOutOfRange("body mesh limb axis"));
            }
            let mut min = Vec2::new(left.0.x.min(right.0.x), left.0.y.min(right.0.y));
            let mut max = Vec2::new(left.0.x.max(right.0.x), left.0.y.max(right.0.y));
            for q in [left.1, right.1] {
                min = Vec2::new(min.x.min(q.x), min.y.min(q.y));
                max = Vec2::new(max.x.max(q.x), max.y.max(q.y));
            }
            Ok(Plan {
                region: Region::Limb {
                    segments: [left, right],
                    radius,
                },
                min: Vec2::new(min.x - radius, min.y - radius * 0.25),
                max: Vec2::new(max.x + radius, max.y + radius * 0.25),
            })
        }
        _ => Err(Refusal::UnknownEffect(pf)),
    }
}

/// Bilinear foreground weight at a normalised point, clamped to the frame.
fn mask_at(mask: &[f32], width: u32, height: u32, p: Vec2) -> f32 {
    let w = width as f32;
    let h = height as f32;
    let x = (p.x.clamp(0.0, 1.0) * w - 0.5).clamp(0.0, (w - 1.0).max(0.0));
    let y = (p.y.clamp(0.0, 1.0) * h - 0.5).clamp(0.0, (h - 1.0).max(0.0));
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width as usize - 1);
    let y1 = (y0 + 1).min(height as usize - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let at = |ix: usize, iy: usize| mask[iy * width as usize + ix].clamp(0.0, 1.0);
    let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
    let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
    (top * (1.0 - fy) + bottom * fy).clamp(0.0, 1.0)
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Displacement for one lattice vertex. `weight` already folds the foreground mask.
///
/// `degree` is signed and both signs are meaningful: a negative value runs the same
/// field the other way (narrow↔widen, slim↔full, lengthen↔shorten). Only zero is a
/// no-op, so a neutral slider produces an exactly still mesh.
fn displacement(plan: &Plan, p: Vec2, degree: f32, weight: f32) -> Result<Vec2, Refusal> {
    let zero = Vec2::new(0.0, 0.0);
    if weight <= 0.0 || degree == 0.0 {
        return Ok(zero);
    }
    match &plan.region {
        Region::Torso { frame, band, row } => {
            let axes = frame.axes()?;
            let cover = band_weight(p, frame, *band, BAND_FEATHER)?;
            let strength = weight * cover * degree;
            if strength <= 0.0 {
                return Ok(zero);
            }
            let (t, u) = frame.to_torso(p)?;
            let half = axes.shoulder_width * 0.5;
            let (lateral, vertical) = match row {
                // Toward the torso axis.
                TorsoRow::Waist => (-u * half * WAIST_SLIM, 0.0),
                // Outward and slightly up.
                TorsoRow::Bust => (u * half * BUST_OUT, -axes.torso_height * BUST_UP),
                // Lengthen above the waist line, shorten below it.
                TorsoRow::Proportion => (
                    0.0,
                    (t - 0.5) * axes.torso_height * TORSO_PROPORTION,
                ),
            };
            let lateral = lateral * strength;
            let vertical = vertical * strength;
            Ok(Vec2::new(
                axes.right.x * lateral + axes.down.x * vertical,
                axes.right.y * lateral + axes.down.y * vertical,
            ))
        }
        Region::Head { centre, radius } => {
            let cover = 1.0 - smoothstep(radius * 0.55, *radius, limb_length(*centre, p));
            let strength = weight * cover * degree * HEAD_ZOOM;
            Ok(Vec2::new(
                (centre.x - p.x) * strength,
                (centre.y - p.y) * strength,
            ))
        }
        Region::Neck { frame, band, side } => {
            let axes = frame.axes()?;
            let cover = band_weight(p, frame, *band, BAND_FEATHER)?;
            match side {
                // 410: lift the whole band, so the neck gets longer.
                None => {
                    let strength = weight * cover * degree * NECK_STRETCH * axes.torso_height;
                    Ok(Vec2::new(-axes.down.x * strength, -axes.down.y * strength))
                }
                // 413/414: pull one side toward the neck axis.
                Some(side) => {
                    let (_, u) = frame.to_torso(p)?;
                    let one_side = side.weight(u);
                    let strength = weight * cover * one_side * degree * SWAN_NECK;
                    let lateral = -u * axes.shoulder_width * 0.5 * strength;
                    Ok(Vec2::new(axes.right.x * lateral, axes.right.y * lateral))
                }
            }
        }
        Region::Shoulder {
            frame,
            side,
            corner,
            radius,
            width_row,
        } => {
            let axes = frame.axes()?;
            let (_, u) = frame.to_torso(p)?;
            let one_side = side.weight(u);
            // Radial cover around that corner keeps the field off the other shoulder.
            let cover = 1.0 - smoothstep(radius * 0.45, *radius, limb_length(*corner, p));
            let strength = weight * cover * one_side * degree;
            if *width_row {
                // Positive degree widens that shoulder outward, on its own side only.
                let lateral = side.outward() * u.abs() * axes.shoulder_width * 0.5
                    * SHOULDER_WIDTH
                    * strength;
                Ok(Vec2::new(axes.right.x * lateral, axes.right.y * lateral))
            } else {
                // Square-off: raise that corner along the torso's up direction.
                let vertical = -axes.torso_height * SHOULDER_SQUARE * strength;
                Ok(Vec2::new(axes.down.x * vertical, axes.down.y * vertical))
            }
        }
        Region::Limb { segments, radius } => {
            let mut cover = 0.0f32;
            let mut pull = Vec2::new(0.0, 0.0);
            for (a, b) in segments {
                let tube = tube_weight(p, *a, *b, *radius, *radius * 0.6)?;
                if tube <= cover {
                    continue;
                }
                let ab = Vec2::new(b.x - a.x, b.y - a.y);
                let len2 = ab.x * ab.x + ab.y * ab.y;
                if len2 <= 1e-12 {
                    return Err(Refusal::ValueOutOfRange("body mesh limb axis"));
                }
                let t = (((p.x - a.x) * ab.x + (p.y - a.y) * ab.y) / len2).clamp(0.0, 1.0);
                cover = tube;
                pull = Vec2::new(a.x + ab.x * t - p.x, a.y + ab.y * t - p.y);
            }
            let strength = weight * cover * degree * LIMB_SLIM;
            Ok(Vec2::new(pull.x * strength, pull.y * strength))
        }
    }
}

/// Build one Mesh row.
///
/// * `pf` — one of the 13 rows; anything else is `UnknownEffect`.
/// * `degree` — already curve-mapped, signed, relative to neutral. Magnitudes above
///   1 are clamped; `NaN`/`inf` are refused. Zero leaves the mesh exactly still (the
///   neutral slider); a negative value runs the same field the other way.
/// * `width`/`height` — the frame the mask belongs to; both non-zero.
/// * `anchors` — this photo's semantic anchors; each row requires its own, and a
///   missing one is that anchor's own `MissingInput`.
/// * `mask` — frame-sized foreground weight in `0..=1`. It multiplies every
///   displacement, so a vertex over background never moves.
pub(super) fn generate(
    pf: u32,
    degree: f32,
    width: u32,
    height: u32,
    anchors: &AnchorSet,
    mask: &[f32],
) -> Result<BodyGeometryInput, Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("body mesh frame"));
    }
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("body mesh mask"))?;
    if mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("body mesh mask"));
    }
    if !degree.is_finite() {
        return Err(Refusal::ValueOutOfRange("body mesh degree"));
    }
    let degree = degree.clamp(-1.0, 1.0);
    let plan = plan(pf, anchors)?;

    let mut source_uv = Vec::with_capacity(MESH_SIDE * MESH_SIDE);
    let mut destination = Vec::with_capacity(MESH_SIDE * MESH_SIDE);
    for row in 0..MESH_SIDE {
        for column in 0..MESH_SIDE {
            let fu = column as f32 / (MESH_SIDE - 1) as f32;
            let fv = row as f32 / (MESH_SIDE - 1) as f32;
            let p = Vec2::new(
                plan.min.x + (plan.max.x - plan.min.x) * fu,
                plan.min.y + (plan.max.y - plan.min.y) * fv,
            );
            source_uv.push(p);
            let on_ring =
                row == 0 || row == MESH_SIDE - 1 || column == 0 || column == MESH_SIDE - 1;
            let moved = if on_ring {
                p
            } else {
                let weight = mask_at(mask, width, height, p);
                if weight <= MASK_FLOOR {
                    p
                } else {
                    let d = displacement(&plan, p, degree, weight)?;
                    Vec2::new(p.x + d.x, p.y + d.y)
                }
            };
            destination.push(moved);
        }
    }

    let mesh = Mesh::new(source_uv, destination, grid_indices(MESH_SIDE, MESH_SIDE)?)?;
    // One recovered-type pass over the mesh. `uv_map` must stay false: the stage
    // refuses colour passes on these rows.
    let passes = vec![Pass {
        edge: EdgeMode::ReshapeNormal,
        defines: PassDefines::plain(),
        clamp: Clamp::disabled(),
        matrix: Pass::normalized_to_clip(),
    }];
    Ok(BodyGeometryInput::Mesh { mesh, passes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_portrait::kernels::body_pose_adapter::{adopt, ModelLayout, ObservedKeypoint};

    /// A Halpe-26 observation with the anchors these rows need, at the layout's own
    /// indices (5/6 shoulders, 11/12 hips, 13/14 knees, 15/16 ankles, 17 vertex,
    /// 18 neck base).
    fn halpe_anchors() -> AnchorSet {
        let mut points: Vec<ObservedKeypoint> = (0..26)
            .map(|_| ObservedKeypoint {
                x: 0.5,
                y: 0.5,
                score: 1.0,
                present: true,
            })
            .collect();
        let put = |points: &mut Vec<ObservedKeypoint>, index: usize, x: f32, y: f32| {
            points[index] = ObservedKeypoint {
                x,
                y,
                score: 1.0,
                present: true,
            };
        };
        put(&mut points, 0, 0.50, 0.16); // nose
        put(&mut points, 5, 0.38, 0.30); // left shoulder
        put(&mut points, 6, 0.62, 0.30); // right shoulder
        put(&mut points, 11, 0.42, 0.62); // left hip
        put(&mut points, 12, 0.58, 0.62); // right hip
        put(&mut points, 13, 0.42, 0.78); // left knee
        put(&mut points, 14, 0.58, 0.78); // right knee
        put(&mut points, 15, 0.42, 0.92); // left ankle
        put(&mut points, 16, 0.58, 0.92); // right ankle
        put(&mut points, 17, 0.50, 0.06); // head top
        put(&mut points, 18, 0.50, 0.24); // neck base
        adopt(ModelLayout::Halpe26, &points, 0.3).expect("halpe fixture")
    }

    fn full_mask(width: u32, height: u32) -> Vec<f32> {
        vec![1.0; (width * height) as usize]
    }

    fn mesh_of(input: &BodyGeometryInput) -> (&Mesh, &[Pass]) {
        match input {
            BodyGeometryInput::Mesh { mesh, passes } => (mesh, passes),
            other => panic!("expected a Mesh row, got {other:?}"),
        }
    }

    #[test]
    fn the_lattice_is_pinned_and_the_streams_match_the_rasterizer() {
        let anchors = halpe_anchors();
        let (width, height) = (320, 480);
        let mask = full_mask(width, height);
        let input = generate(401, 0.8, width, height, &anchors, &mask).expect("waist row");
        let (mesh, passes) = mesh_of(&input);

        assert_eq!(mesh.source_uv.len(), MESH_SIDE * MESH_SIDE);
        assert_eq!(mesh.destination.len(), mesh.source_uv.len());
        assert_eq!(
            mesh.indices.len(),
            (MESH_SIDE - 1) * (MESH_SIDE - 1) * 6,
            "the recovered cell order, reused"
        );
        assert_eq!(passes.len(), 1);
        assert_eq!(passes[0].edge, EdgeMode::ReshapeNormal);
        assert!(!passes[0].defines.uv_map, "the stage refuses colour passes");
        assert_eq!(passes[0].matrix, Pass::normalized_to_clip());

        // Lattice orientation: index 0 is the top-left corner, index 10 the next
        // column (x grows), index 11 the next row (y grows). This is the order
        // `grid_indices` walks, so a swapped row/column would show up here.
        assert!(mesh.source_uv[1].x > mesh.source_uv[0].x);
        assert_eq!(mesh.source_uv[1].y, mesh.source_uv[0].y);
        assert!(mesh.source_uv[MESH_SIDE].y > mesh.source_uv[0].y);
        assert_eq!(mesh.source_uv[MESH_SIDE].x, mesh.source_uv[0].x);

        let mut moved_inside = 0;
        for row in 0..MESH_SIDE {
            for column in 0..MESH_SIDE {
                let index = row * MESH_SIDE + column;
                let s = mesh.source_uv[index];
                let d = mesh.destination[index];
                let on_ring =
                    row == 0 || row == MESH_SIDE - 1 || column == 0 || column == MESH_SIDE - 1;
                if on_ring {
                    assert_eq!(s, d, "ring vertex ({row},{column}) must not move");
                } else if s != d {
                    moved_inside += 1;
                }
            }
        }
        assert!(moved_inside > 0, "a non-zero waist degree must move interior");
        assert!(mesh
            .destination
            .iter()
            .all(|p| p.x.is_finite() && p.y.is_finite()));
    }

    #[test]
    fn sided_rows_stay_on_their_own_side_and_missing_anchors_refuse() {
        let anchors = halpe_anchors();
        let (width, height) = (320, 480);
        let mask = full_mask(width, height);

        let left = generate(417, 0.9, width, height, &anchors, &mask).expect("left width");
        let right = generate(418, 0.9, width, height, &anchors, &mask).expect("right width");
        let axis = TorsoFrame::new(
            anchor_at(&anchors, Anchor::LeftShoulder).unwrap(),
            anchor_at(&anchors, Anchor::RightShoulder).unwrap(),
            anchor_at(&anchors, Anchor::LeftHip).unwrap(),
            anchor_at(&anchors, Anchor::RightHip).unwrap(),
        )
        .unwrap();
        let lateral = |input: &BodyGeometryInput| -> f32 {
            let (mesh, _) = mesh_of(input);
            mesh.source_uv
                .iter()
                .zip(mesh.destination.iter())
                .map(|(s, d)| {
                    let before = axis.to_torso(*s).unwrap().1;
                    let after = axis.to_torso(*d).unwrap().1;
                    after - before
                })
                .sum()
        };
        let left_shift = lateral(&left);
        let right_shift = lateral(&right);
        assert!(
            left_shift < -1e-4,
            "PF417 must push the person's left shoulder outward (-u), got {left_shift}"
        );
        assert!(
            right_shift > 1e-4,
            "PF418 must push the person's right shoulder outward (+u), got {right_shift}"
        );

        // Each side only moves its own half. The 0.15 split transition is excluded,
        // so "the other side" means the region where its own-side weight is exactly
        // zero rather than merely small.
        let opposite = |input: &BodyGeometryInput, sign: f32| -> f32 {
            let (mesh, _) = mesh_of(input);
            mesh.source_uv
                .iter()
                .zip(mesh.destination.iter())
                .filter(|(s, _)| axis.to_torso(**s).unwrap().1 * sign > 0.25)
                .map(|(s, d)| {
                    (axis.to_torso(*d).unwrap().1 - axis.to_torso(*s).unwrap().1).abs()
                })
                .fold(0.0f32, f32::max)
        };
        assert_eq!(opposite(&left, 1.0), 0.0, "PF417 must not touch the right side");
        assert_eq!(opposite(&right, -1.0), 0.0, "PF418 must not touch the left side");

        // No anchors at all: the row refuses with the anchor's own label.
        let empty = AnchorSet::empty();
        match generate(417, 0.9, width, height, &empty, &mask) {
            Err(aurora_portrait::Refusal::MissingInput(label)) => {
                assert!(!label.is_empty(), "the refusal must name the anchor")
            }
            other => panic!("a missing anchor must be refused, got {other:?}"),
        }

        // A background vertex never moves, even at full degree.
        let background = vec![0.0f32; (width * height) as usize];
        let still = generate(401, 1.0, width, height, &anchors, &background).expect("masked row");
        let (mesh, _) = mesh_of(&still);
        assert!(mesh
            .source_uv
            .iter()
            .zip(mesh.destination.iter())
            .all(|(s, d)| s == d));
    }
}
