//! Storage-independent color sampling of generated body geometry.
//!
//! Both output depths see the same pixel-center coordinates and subject
//! coverage, but the wide input never passes through RGBA8 or an 8-bit delta.
//! Legacy externally supplied body jobs retain their existing consumer.

use aurora_portrait::kernels::{
    body_geometry::{self as g, Clamp, EdgeMode, Pass, Vec2},
    hair_body::PackedShiftMap,
    leg_refine,
};
use aurora_portrait::Refusal;

use super::portrait_body_stage::{BodyGeometryInput, BodyGeometryJob};

/// Colors of one bounded subject rectangle, in the destination frame's grid.
pub(super) struct Tile {
    pub narrow: Vec<u8>,
    pub wide: Option<Vec<u16>>,
}

impl Tile {
    fn pass(
        &mut self,
        size: (u32, u32),
        visit: impl FnOnce(&mut dyn FnMut(usize, Vec2, f32)) -> Result<(), Refusal>,
    ) -> Result<(), Refusal> {
        let src8 = &self.narrow;
        let src16 = self.wide.as_deref();
        let mut out8 = src8.clone();
        let mut out16 = self.wide.clone();
        let mut invalid = false;
        visit(&mut |i, uv, mix| {
            if !uv.x.is_finite() || !uv.y.is_finite() || !mix.is_finite() {
                invalid = true;
                return;
            }
            let origin = center(i, size);
            if mix <= 0.0 || ((uv.x - origin.x).abs() < 1e-7 && (uv.y - origin.y).abs() < 1e-7) {
                return;
            }
            let sample = Footprint::new(uv, size);
            let mix = mix.min(1.0);
            let base = i * 4;
            // GEOMETRY changes RGB; output alpha stays destination-owned.
            for c in 0..3 {
                let from = src8[base + c] as f32;
                out8[base + c] = lerp(from, sample.rgb(src8, c), mix)
                    .clamp(0.0, 255.0).round_ties_even() as u8;
                if let (Some(src), Some(dst)) = (src16, out16.as_deref_mut()) {
                    let from = src[base + c] as f32;
                    dst[base + c] = lerp(from, sample.rgb(src, c), mix)
                        .clamp(0.0, 65535.0).round_ties_even() as u16;
                }
            }
        })?;
        if invalid { return Err(Refusal::ValueOutOfRange("body sampling coordinates")); }
        self.narrow = out8;
        self.wide = out16;
        Ok(())
    }
}

fn center(i: usize, (w, h): (u32, u32)) -> Vec2 {
    Vec2::new(((i % w as usize) as f32 + 0.5) / w as f32,
              ((i / w as usize) as f32 + 0.5) / h as f32)
}
fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }
fn mix_uv(a: Vec2, b: Vec2, t: f32) -> Vec2 {
    Vec2::new(lerp(a.x, b.x, t), lerp(a.y, b.y, t))
}
struct Footprint { taps: [usize; 4], tx: f32, ty: f32 }
impl Footprint {
    fn new(uv: Vec2, (w, h): (u32, u32)) -> Self {
        let axis = |u: f32, n: u32| {
            let p = (u * n as f32 - 0.5).clamp(0.0, (n - 1) as f32);
            let lo = p.floor() as usize;
            (lo, (lo + 1).min(n as usize - 1), p - lo as f32)
        };
        let (x0, x1, tx) = axis(uv.x, w);
        let (y0, y1, ty) = axis(uv.y, h);
        let w = w as usize;
        Self { taps: [y0*w+x0, y0*w+x1, y1*w+x0, y1*w+x1], tx, ty }
    }
    fn rgb<T: Copy + Into<f32>>(&self, pixels: &[T], channel: usize) -> f32 {
        let [a, b, c, d] = self.taps.map(|i| pixels[4*i+channel].into());
        lerp(lerp(a, b, self.tx), lerp(c, d, self.tx), self.ty)
    }
}

/// One generated ticket is bound to the exact tile it is about to resample.
pub(super) fn apply(
    tile: &mut Tile, pf: u32, job_id: u64, job: &BodyGeometryJob, mask: &[f32],
) -> Result<(), Refusal> {
    job.validate_frame(job.frame_size, &tile.narrow, Some(job_id))?;
    let size @ (w, h) = job.frame_size;
    let n = (w as usize).checked_mul(h as usize).ok_or(Refusal::ShapeMismatch("body tile"))?;
    if n.checked_mul(4) != Some(tile.narrow.len()) || n == 0 || mask.len() != n
        || tile.wide.as_ref().is_some_and(|p| p.len() != tile.narrow.len())
    {
        return Err(Refusal::ShapeMismatch("body tile"));
    }
    let input = job.effects.get(&pf).ok_or(Refusal::MissingInput("body geometry effect"))?;
    use BodyGeometryInput::*;
    match input {
        SixPoint(s) => {
            if !(1..=6).contains(&s.valid_point_count) || !s.inverse_aspect.is_finite() || s.inverse_aspect <= 0.0
                || !s.amount.is_finite() || s.degrees.iter().any(|x| !x.is_finite())
            {
                return Err(Refusal::ValueOutOfRange("body six point"));
            }
            let warp = if pf == 400 { g::body_warp_uv } else { g::body_liquid_uv };
            tile.pass(size, |set| {
                for (i, &m) in mask.iter().enumerate() {
                    if m > 0.0 {
                        let uv = warp(center(i, size), &s.source_points, &s.destination_points,
                                      &s.degrees, s.amount, s.inverse_aspect, s.valid_point_count);
                        set(i, uv, m);
                    }
                }
                Ok(())
            })
        }
        Mesh { mesh, passes } => {
            if passes.is_empty() { return Err(Refusal::MissingInput("body mesh passes")); }
            for pass in passes {
                tile.pass(size, |set| g::mesh_color_samples(size, mesh, pass, |i, uv| set(i, uv, mask[i])))?;
            }
            Ok(())
        }
        BodySlim { mask_r, source_uv, target_uv } => {
            if mask_r.len() != n || source_uv.len() != n || target_uv.len() != n {
                return Err(Refusal::ShapeMismatch("body slim UV"));
            }
            tile.pass(size, |set| {
                for (i, &m) in mask_r.iter().enumerate() {
                    if m < 253 { set(i, mix_uv(source_uv[i], target_uv[i], 1.0-m as f32/255.0), 1.0); }
                }
                Ok(())
            })
        }
        Leg { columns, catmull_size } => {
            if catmull_size.is_some() { return Err(Refusal::ValueOutOfRange("generated leg sampling")); }
            // The existing strip gives the inverse mapping of its main axis.
            // A leg length control stretches vertically; the 1px minor axis
            // saves allocating a second full-size offset field.
            let field = leg_refine::offset_field((h, 1), columns)?;
            tile.pass(size, |set| {
                for (i, &m) in mask.iter().enumerate() {
                    let uv = center(i, size);
                    set(i, Vec2::new(uv.x, uv.y + field[i / w as usize][0]), m);
                }
                Ok(())
            })
        }
        Height { source_uv, destination_uv, defines } => {
            if source_uv.len() != n || destination_uv.len() != n || defines.uv_map || defines.catmull_rom {
                return Err(Refusal::ShapeMismatch("body height UV"));
            }
            let pass = Pass { edge: EdgeMode::ReshapeNormal, defines: *defines,
                              clamp: Clamp::disabled(), matrix: Pass::normalized_to_clip() };
            tile.pass(size, |set| {
                for (i, (&source, &dest)) in source_uv.iter().zip(destination_uv).enumerate() {
                    let uv = center(i, size);
                    if (uv.x-dest.x).abs() > 1e-6 || (uv.y-dest.y).abs() > 1e-6 {
                        return Err(Refusal::ValueOutOfRange("body height destination grid"));
                    }
                    set(i, g::color_sample_uv(source, dest, &pass), mask[i]);
                }
                Ok(())
            })
        }
        Waist { warped_uv, mask_r } => {
            if warped_uv.len() != n || mask_r.len() != n { return Err(Refusal::ShapeMismatch("body waist UV")); }
            tile.pass(size, |set| {
                for (i, (&warp, &m)) in warped_uv.iter().zip(mask_r).enumerate() {
                    if !m.is_finite() || !(0.0..=1.0).contains(&m) {
                        return Err(Refusal::ValueOutOfRange("body waist mask"));
                    }
                    if m < 0.99 { set(i, mix_uv(center(i, size), warp, 1.0-m), 1.0); }
                }
                Ok(())
            })
        }
        BreastLift { degree01, brushes, density, density_size: (dw, dh), freeze_mask,
                     warp_direction, edge_margin, push_left } => {
            if !(0.0..=1.0).contains(degree01) || *dw == 0 || *dh == 0
                || (*dw as usize).checked_mul(*dh as usize) != Some(density.len())
                || freeze_mask.len() != n || brushes.is_empty() || !warp_direction.is_finite()
                || !edge_margin.is_finite() || *edge_margin <= 0.0
            {
                return Err(Refusal::ShapeMismatch("body lift inputs"));
            }
            for brush in brushes {
                if !brush.radius.is_finite() || brush.radius <= 0.0 || !brush.pressure.is_finite() {
                    return Err(Refusal::ValueOutOfRange("body lift brush"));
                }
                tile.pass(size, |set| {
                    for (i, &m) in mask.iter().enumerate() {
                        if m > 0.0 {
                            let uv = center(i, size);
                            let (u, v) = g::brush_sample_uv(uv.x, uv.y, w as f32, h as f32, brush,
                                *warp_direction, density, *dw as usize, *dh as usize, freeze_mask, *edge_margin, *push_left);
                            set(i, Vec2::new(u, v), m);
                        }
                    }
                    Ok(())
                })?;
            }
            Ok(())
        }
        Shift(p) => {
            if p.mask.len() != n { return Err(Refusal::ShapeMismatch("body shift mask")); }
            let shift = PackedShiftMap { width: p.map_size.0, height: p.map_size.1,
                rgba8: &p.rgba8, ratio: p.ratio, shift_width: p.shift_size.0, shift_height: p.shift_size.1 };
            shift.validate()?;
            tile.pass(size, |set| {
                for (i, &m) in p.mask.iter().enumerate().filter(|(_, m)| **m > 0) {
                    let uv = center(i, size);
                    let (dx, dy) = shift.displacement(uv.x, uv.y);
                    set(i, Vec2::new(uv.x+dx, uv.y+dy), m as f32/255.0);
                }
                Ok(())
            })
        }
    }
}
