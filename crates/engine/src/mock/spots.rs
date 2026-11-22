//! Spot removal for a developed RGBA frame: dust (median) and heal (Laplacian
//! diffusion of the surrounding ring plus high-frequency detail from a mirrored
//! patch just outside the spot).
//! Clone copies a source patch instead of synthesising one, feathering its rim.
//! Approximation only — no seam matching or source-point search, and the ring
//! is sampled at 32 angles.
//! Limits: radii follow the spot radius fraction of the oriented image's long side.

use crate::{AdjustmentSet, CoverKind, CoverPatch, Geometry, Spot, SpotKind};

/// What a spot does to the pixels it covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpotMode {
    /// Diffusion fill from the surrounding ring.
    Heal,
    /// 3x3 median filter.
    Dust,
    /// Copy the patch at the spot's source offset.
    Clone,
}

const TAU: f32 = std::f32::consts::TAU;
const RING_SAMPLES: usize = 32;
const FEATHER_RATIO: f32 = 0.35;
/// C1 feathers a heal over the two pixels just outside the spot radius.
const HEAL_FEATHER_PX: f32 = 2.0;
const DIFFUSE_ITERATIONS: usize = 240;

/// Applies every spot in `adj` to a developed frame of `width` x `height`.
///
/// `source_size` is the size of the source image the frame was rendered from,
/// in source pixels (before the quarter turn). `Spot` coordinates live in the
/// oriented, uncropped image, so they are mapped through `adj.geometry` first.
pub(crate) fn apply_spots(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    adj: &AdjustmentSet,
    source_size: (u32, u32),
) {
    if adj.spots.is_empty() || width == 0 || height == 0 {
        return;
    }
    let tasks: Vec<(SpotMode, Spot)> =
        adj.spots.iter().map(|s| (mode_of(s.kind), s.clone())).collect();
    apply_spots_with(rgba, width, height, &adj.geometry, &tasks, source_size);
}

/// Applies spots with explicit modes.
pub(crate) fn apply_spots_with(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    geometry: &Geometry,
    tasks: &[(SpotMode, Spot)],
    source_size: (u32, u32),
) {
    if tasks.is_empty() || width == 0 || height == 0 {
        return;
    }
    for (mode, spot) in tasks {
        let Some(frame) = spot_geometry(spot, geometry, source_size, width, height) else {
            continue;
        };
        match mode {
            SpotMode::Heal => heal(rgba, width, height, frame.cx, frame.cy, frame.radius),
            SpotMode::Dust => dust(rgba, width, height, frame.cx, frame.cy, frame.radius),
            SpotMode::Clone => {
                if let Some(src) = frame.src {
                    clone_patch(rgba, width, height, frame.cx, frame.cy, frame.radius, src);
                }
            }
        }
    }
}

/// Applies the manual spots in `adj` plus AI-detected dust, each area once.
///
/// A detected spot that overlaps a manual spot is dropped: the manual edit wins,
/// so heal/dust never runs twice on the same pixels. Nothing calls this
/// automatically — whether detected dust is applied is the caller's decision.
#[allow(dead_code)] // engine-side API; the interface decides when to apply it
pub(crate) fn apply_detected_spots(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    adj: &AdjustmentSet,
    geometry: &Geometry,
    dust: &[Spot],
    source_size: (u32, u32),
) {
    if width == 0 || height == 0 || (adj.spots.is_empty() && dust.is_empty()) {
        return;
    }
    let mut tasks: Vec<(SpotMode, Spot)> =
        adj.spots.iter().map(|s| (mode_of(s.kind), s.clone())).collect();
    for spot in dust {
        if adj.spots.iter().any(|manual| overlaps(manual, spot)) {
            continue;
        }
        tasks.push((SpotMode::Dust, spot.clone()));
    }
    apply_spots_with(rgba, width, height, geometry, &tasks, source_size);
}

/// Normalised centres closer than half the summed radii count as the same spot.
#[allow(dead_code)]
fn overlaps(a: &Spot, b: &Spot) -> bool {
    let reach = 0.5 * (a.radius.max(0.0) + b.radius.max(0.0));
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt() <= reach
}

fn mode_of(kind: SpotKind) -> SpotMode {
    match kind {
        // C1's ZSPOTS types 1 (Black), 2 (Spot) and 3 (Dust) all inpaint from
        // the surrounding ring; only Clone differs. The 3x3 median is left for
        // AI-detected dust, which is small and fixed-size.
        SpotKind::Dust | SpotKind::Spot => SpotMode::Heal,
        SpotKind::Clone { .. } => SpotMode::Clone,
    }
}

/// A spot in developed-frame pixels.
struct FrameSpot {
    cx: f32,
    cy: f32,
    radius: f32,
    /// Clone source offset in frame pixels; `None` for heal and dust.
    src: Option<(f32, f32)>,
}

/// Maps a spot to frame pixels: centre in the developed frame, its radius, and
/// the clone source offset.
fn spot_geometry(
    spot: &Spot,
    geometry: &Geometry,
    source_size: (u32, u32),
    width: u32,
    height: u32,
) -> Option<FrameSpot> {
    if !spot.x.is_finite() || !spot.y.is_finite() || !spot.radius.is_finite() {
        return None;
    }
    let (cx, cy, scale) = oriented_to_frame([spot.x, spot.y], geometry, source_size, width, height)?;
    let (sw, sh) = source_size;
    let (ow, oh) = if geometry.orientation % 2 == 1 {
        (sh as f32, sw as f32)
    } else {
        (sw as f32, sh as f32)
    };
    // Radius is a fraction of the image's long side; convert to frame pixels.
    let long_side = ow.max(oh);
    let radius_px = spot.radius * long_side * scale;
    if !radius_px.is_finite() || radius_px <= 0.0 {
        return None;
    }
    // The source offset lives in the oriented, uncropped image, so it takes the
    // same quarter turn and crop scale as the centre.
    let src = match spot.kind {
        SpotKind::Clone { src_offset } => {
            let rot = geometry.rotation.to_radians();
            let (sin, cos) = rot.sin_cos();
            let (dx, dy) = (src_offset.0 as f32, src_offset.1 as f32);
            Some(((cos * dx - sin * dy) * scale, (sin * dx + cos * dy) * scale))
        }
        _ => None,
    };
    Some(FrameSpot { cx, cy, radius: radius_px, src })
}

/// Normalized oriented point → frame pixels, plus the frame-pixels-per-oriented-
/// pixel scale. The spot crop/fine-rotation path; keystone is not modelled.
fn oriented_to_frame(
    p: [f32; 2],
    geometry: &Geometry,
    source_size: (u32, u32),
    width: u32,
    height: u32,
) -> Option<(f32, f32, f32)> {
    let (sw, sh) = source_size;
    if sw == 0 || sh == 0 || width == 0 || height == 0 {
        return None;
    }
    if !p[0].is_finite() || !p[1].is_finite() {
        return None;
    }
    let (ow, oh) = if geometry.orientation % 2 == 1 {
        (sh as f32, sw as f32)
    } else {
        (sw as f32, sh as f32)
    };
    let crop = super::geometry::crop_for_aspect(geometry, source_size);
    if crop.width <= 0.0 || crop.height <= 0.0 {
        return None;
    }
    // The point sits in the oriented image before crop and fine rotation; the
    // frame shows the crop rotated about the oriented centre, so undo that
    // rotation first and then undo the crop.
    let (qx, qy) = (p[0] * ow, p[1] * oh);
    let (cx, cy) = (0.5 * ow, 0.5 * oh);
    let rot = geometry.rotation.to_radians();
    let (sin, cos) = rot.sin_cos();
    let (dx, dy) = (qx - cx, qy - cy);
    let px = cx + cos * dx - sin * dy;
    let py = cy + sin * dx + cos * dy;
    let fx = (px / ow - crop.x) / crop.width;
    let fy = (py / oh - crop.y) / crop.height;
    if !fx.is_finite() || !fy.is_finite() {
        return None;
    }
    let scale = (width as f32 / (crop.width * ow)).min(height as f32 / (crop.height * oh));
    Some((fx * width as f32, fy * height as f32, scale))
}

/// Applies the `.comask` Cover arrows of one repair layer.
///
/// Each arrow replaces a rectangle at `dst` with the pixels at `src` and feathers
/// the rim over 2 px outside it, like a spot. Heal adds
/// `mean(src region) - mean(dst ring)` to every copied pixel: C1's colour
/// correction for a repair lifted from a differently lit area.
pub(crate) fn apply_cover_patches(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    geometry: &Geometry,
    patches: &[CoverPatch],
    source_size: (u32, u32),
) {
    let (sw, sh) = source_size;
    if patches.is_empty() || width == 0 || height == 0 || sw == 0 || sh == 0 {
        return;
    }
    let snapshot = rgba.to_vec();
    let (ow, oh) = if geometry.orientation % 2 == 1 {
        (sh as f32, sw as f32)
    } else {
        (sw as f32, sh as f32)
    };
    for patch in patches {
        let Some((dx, dy, scale)) =
            oriented_to_frame(patch.dst, geometry, source_size, width, height)
        else {
            continue;
        };
        let Some((sx, sy, _)) =
            oriented_to_frame(patch.src, geometry, source_size, width, height)
        else {
            continue;
        };
        let hw = patch.half[0] * ow * scale;
        let hh = patch.half[1] * oh * scale;
        if !hw.is_finite() || !hh.is_finite() || hw <= 0.0 || hh <= 0.0 {
            continue;
        }
        let rect = (dx - hw, dy - hh, dx + hw, dy + hh);
        let offset = (sx - dx, sy - dy);
        let shift = match patch.kind {
            CoverKind::Clone => [0.0; 3],
            CoverKind::Heal => colour_shift(&snapshot, width, height, rect, offset),
        };
        cover_rect(rgba, &snapshot, width, height, rect, offset, shift);
    }
}

/// Distance from a pixel index to a half-open rectangle `[x0, x1) x [y0, y1)`;
/// zero inside, one at the first pixel past the edge.
fn rect_distance(x: f32, y: f32, rect: (f32, f32, f32, f32)) -> f32 {
    let (x0, y0, x1, y1) = rect;
    let dx = (x0 - x).max(x - (x1 - 1.0)).max(0.0);
    let dy = (y0 - y).max(y - (y1 - 1.0)).max(0.0);
    (dx * dx + dy * dy).sqrt()
}

/// `mean(src) - mean(ring around dst)` per channel, over the 2 px feather band.
fn colour_shift(
    snapshot: &[u8],
    width: u32,
    height: u32,
    rect: (f32, f32, f32, f32),
    offset: (f32, f32),
) -> [f32; 3] {
    let (x0, x1) = (rect.0 - HEAL_FEATHER_PX, rect.2 + HEAL_FEATHER_PX);
    let (y0, y1) = (rect.1 - HEAL_FEATHER_PX, rect.3 + HEAL_FEATHER_PX);
    let mut src = [0f32; 3];
    let mut ring = [0f32; 3];
    let (mut n_src, mut n_ring) = (0f32, 0f32);
    for py in y0.floor() as i64..=y1.ceil() as i64 {
        for px in x0.floor() as i64..=x1.ceil() as i64 {
            let (x, y) = (px as f32, py as f32);
            let d = rect_distance(x, y, rect);
            if d <= 0.0 {
                if let Some(c) = sample(snapshot, width, height, x + offset.0, y + offset.1) {
                    for k in 0..3 {
                        src[k] += c[k];
                    }
                    n_src += 1.0;
                }
            } else if d <= HEAL_FEATHER_PX {
                if let Some(c) = sample(snapshot, width, height, x, y) {
                    for k in 0..3 {
                        ring[k] += c[k];
                    }
                    n_ring += 1.0;
                }
            }
        }
    }
    if n_src == 0.0 || n_ring == 0.0 {
        return [0.0; 3];
    }
    let mut shift = [0f32; 3];
    for k in 0..3 {
        shift[k] = src[k] / n_src - ring[k] / n_ring;
    }
    shift
}

/// Copies the source rectangle over `rect`, feathering the 2 px outside it.
fn cover_rect(
    rgba: &mut [u8],
    snapshot: &[u8],
    width: u32,
    height: u32,
    rect: (f32, f32, f32, f32),
    offset: (f32, f32),
    shift: [f32; 3],
) {
    let feather = HEAL_FEATHER_PX;
    let x0 = (rect.0 - feather).floor().max(0.0) as i64;
    let x1 = (rect.2 + feather).ceil().min(width as f32 - 1.0) as i64;
    let y0 = (rect.1 - feather).floor().max(0.0) as i64;
    let y1 = (rect.3 + feather).ceil().min(height as f32 - 1.0) as i64;
    for py in y0..=y1 {
        for px in x0..=x1 {
            let (x, y) = (px as f32, py as f32);
            let d = rect_distance(x, y, rect);
            if d > feather {
                continue;
            }
            let weight = if d <= 0.0 { 1.0 } else { 1.0 - smoothstep(0.0, feather, d) };
            if weight <= 0.0 {
                continue;
            }
            let p = (py as usize * width as usize + px as usize) * 4;
            if p + 3 >= rgba.len() {
                continue;
            }
            let Some(c) = sample(snapshot, width, height, x + offset.0, y + offset.1) else {
                continue;
            };
            for k in 0..3 {
                let repaired = c[k] + shift[k];
                let orig = snapshot[p + k] as f32;
                rgba[p + k] = (orig * (1.0 - weight) + repaired * weight).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Clone: copies the source patch into the disc, feathering its rim into the
/// pixels it replaces. The source is read from a snapshot, so an overlapping
/// source cannot feed on the clone itself.
fn clone_patch(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    cx: f32,
    cy: f32,
    radius: f32,
    src: (f32, f32),
) {
    let radius = radius.max(1.0);
    let feather = (radius * FEATHER_RATIO).max(1.0);
    let outer = radius + feather;
    let x0 = (cx - outer).floor().max(0.0) as i64;
    let x1 = (cx + outer).ceil().min(width as f32 - 1.0) as i64;
    let y0 = (cy - outer).floor().max(0.0) as i64;
    let y1 = (cy + outer).ceil().min(height as f32 - 1.0) as i64;
    let bw = (x1 - x0 + 1).max(0) as usize;
    let bh = (y1 - y0 + 1).max(0) as usize;
    if bw == 0 || bh == 0 {
        return;
    }

    // Capture only the destination RGB and sampled source colours needed by
    // this clone. Both are read from the pre-write frame, so an overlapping
    // source cannot feed on pixels already replaced by this operation.
    let mut original = vec![[0u8; 3]; bw * bh];
    let mut sampled = vec![[0.0f32; 3]; bw * bh];
    for gy in 0..bh {
        for gx in 0..bw {
            let px = x0 + gx as i64;
            let py = y0 + gy as i64;
            let p = (py as usize * width as usize + px as usize) * 4;
            if p + 3 >= rgba.len() {
                continue;
            }
            let i = gy * bw + gx;
            original[i] = [rgba[p], rgba[p + 1], rgba[p + 2]];
            sampled[i] =
                sample(rgba, width, height, px as f32 + src.0, py as f32 + src.1).unwrap_or([0.0; 3]);
        }
    }

    for py in y0..=y1 {
        for px in x0..=x1 {
            let d = ((px as f32 - cx).powi(2) + (py as f32 - cy).powi(2)).sqrt();
            if d > outer {
                continue;
            }
            let w = if d <= radius { 1.0 } else { 1.0 - smoothstep(radius, outer, d) };
            if w <= 0.0 {
                continue;
            }
            let p = (py as usize * width as usize + px as usize) * 4;
            if p + 3 >= rgba.len() {
                continue;
            }
            let i = (py - y0) as usize * bw + (px - x0) as usize;
            let c = sampled[i];
            for k in 0..3 {
                let orig = original[i][k] as f32;
                rgba[p + k] = (orig * (1.0 - w) + c[k] * w).clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Heal: fills the disc with a diffusion of the surrounding ring.
fn heal(rgba: &mut [u8], width: u32, height: u32, cx: f32, cy: f32, radius: f32) {
    let radius = radius.max(1.0);
    let feather = HEAL_FEATHER_PX;
    let outer = radius + feather;
    let ring_r = (radius * 1.4).clamp(radius + 0.5, radius * 1.6);

    // Ring statistics: mean colour, and the mean deviation that carries texture.
    let mut sum = [0f32; 3];
    let mut count = 0f32;
    let mut samples: Vec<[f32; 3]> = Vec::with_capacity(RING_SAMPLES);
    for i in 0..RING_SAMPLES {
        let a = i as f32 / RING_SAMPLES as f32 * TAU;
        let sx = cx + ring_r * a.cos();
        let sy = cy + ring_r * a.sin();
        if let Some(c) = sample(rgba, width, height, sx, sy) {
            for k in 0..3 {
                sum[k] += c[k];
            }
            count += 1.0;
            samples.push(c);
        }
    }
    if count == 0.0 {
        return;
    }
    let mean = [sum[0] / count, sum[1] / count, sum[2] / count];
    let mut dev = [0f32; 3];
    for c in &samples {
        for k in 0..3 {
            dev[k] += c[k] - mean[k];
        }
    }
    for k in 0..3 {
        dev[k] /= count;
    }

    // Diffusion grid: everything within `outer` is unknown, the ring beyond it
    // holds the original pixels as fixed boundary values.
    let x0 = (cx - outer - 2.0).floor().max(0.0) as i64;
    let x1 = (cx + outer + 2.0).ceil().min(width as f32 - 1.0) as i64;
    let y0 = (cy - outer - 2.0).floor().max(0.0) as i64;
    let y1 = (cy + outer + 2.0).ceil().min(height as f32 - 1.0) as i64;
    if x1 < x0 || y1 < y0 {
        return;
    }
    let gw = (x1 - x0 + 1) as usize;
    let gh = (y1 - y0 + 1) as usize;
    let n = gw * gh;
    let mut orig = vec![[0f32; 3]; n];
    let mut vals = vec![[0f32; 3]; n];
    let mut unknown = vec![false; n];
    let mut dist = vec![0f32; n];
    for gy in 0..gh {
        for gx in 0..gw {
            let idx = gy * gw + gx;
            let px = (x0 + gx as i64) as f32;
            let py = (y0 + gy as i64) as f32;
            let d = ((px - cx).powi(2) + (py - cy).powi(2)).sqrt();
            dist[idx] = d;
            orig[idx] = sample(rgba, width, height, px, py).unwrap_or(mean);
            unknown[idx] = d <= outer;
            vals[idx] = if unknown[idx] { mean } else { orig[idx] };
        }
    }

    // Gauss-Seidel on the unknown region, fixed boundary outside it.
    for _ in 0..DIFFUSE_ITERATIONS {
        for gy in 0..gh {
            for gx in 0..gw {
                let idx = gy * gw + gx;
                if !unknown[idx] {
                    continue;
                }
                let mut acc = [0f32; 3];
                let mut cnt = 0.0f32;
                for (nx, ny) in [
                    (gx as i64 - 1, gy as i64),
                    (gx as i64 + 1, gy as i64),
                    (gx as i64, gy as i64 - 1),
                    (gx as i64, gy as i64 + 1),
                ] {
                    if nx < 0 || ny < 0 || nx >= gw as i64 || ny >= gh as i64 {
                        continue;
                    }
                    let nidx = ny as usize * gw + nx as usize;
                    let v = if unknown[nidx] { vals[nidx] } else { orig[nidx] };
                    for k in 0..3 {
                        acc[k] += v[k];
                    }
                    cnt += 1.0;
                }
                if cnt > 0.0 {
                    for k in 0..3 {
                        vals[idx][k] = acc[k] / cnt;
                    }
                }
            }
        }
    }

    // Brightness compensation: the ring mean says how bright the area should be.
    let mut inner_sum = [0f32; 3];
    let mut inner_count = 0f32;
    for idx in 0..n {
        if unknown[idx] && dist[idx] <= radius {
            for k in 0..3 {
                inner_sum[k] += vals[idx][k];
            }
            inner_count += 1.0;
        }
    }
    let offset = if inner_count > 0.0 {
        [
            mean[0] - inner_sum[0] / inner_count,
            mean[1] - inner_sum[1] / inner_count,
            mean[2] - inner_sum[2] / inner_count,
        ]
    } else {
        [0.0; 3]
    };

    // A mirrored strip beside the patch keeps the source outside the damaged
    // area. Diffusion supplies its low frequency; this only carries fine detail.
    let source_right = cx + 3.0 * outer + 2.0 <= width as f32;
    let source_axis = if source_right {
        (cx + outer).ceil()
    } else {
        (cx - outer).floor()
    };
    let source_x = |px: f32| 2.0 * source_axis - px;

    for gy in 0..gh {
        for gx in 0..gw {
            let idx = gy * gw + gx;
            if !unknown[idx] {
                continue;
            }
            let d = dist[idx];
            let w = if d <= radius { 1.0 } else { 1.0 - smoothstep(radius, outer, d) };
            if w <= 0.0 {
                continue;
            }
            let px = (x0 + gx as i64) as usize;
            let py = (y0 + gy as i64) as usize;
            let p = (py * width as usize + px) * 4;
            if p + 3 >= rgba.len() {
                continue;
            }
            let detail = high_pass(rgba, width, height, source_x(px as f32), py as f32);
            for k in 0..3 {
                let healed = vals[idx][k] + dev[k] + offset[k] + detail[k];
                let v = orig[idx][k] * (1.0 - w) + healed * w;
                rgba[p + k] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Sample minus a four-neighbour average: the fine detail of the source patch.
fn high_pass(rgba: &[u8], width: u32, height: u32, x: f32, y: f32) -> [f32; 3] {
    let Some(centre) = sample(rgba, width, height, x, y) else {
        return [0.0; 3];
    };
    let mut around = [0f32; 3];
    let mut n = 0.0;
    for (dx, dy) in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0)] {
        if let Some(c) = sample(rgba, width, height, x + dx, y + dy) {
            for k in 0..3 {
                around[k] += c[k];
            }
            n += 1.0;
        }
    }
    if n == 0.0 {
        return [0.0; 3];
    }
    let mut out = [0f32; 3];
    for k in 0..3 {
        out[k] = centre[k] - around[k] / n;
    }
    out
}

/// Dust: 3x3 median filter over the disc.
fn dust(rgba: &mut [u8], width: u32, height: u32, cx: f32, cy: f32, radius: f32) {
    let radius = radius.max(1.0);
    let src: Vec<u8> = rgba.to_vec();
    let x0 = (cx - radius).floor().max(0.0) as i64;
    let x1 = (cx + radius).ceil().min(width as f32 - 1.0) as i64;
    let y0 = (cy - radius).floor().max(0.0) as i64;
    let y1 = (cy + radius).ceil().min(height as f32 - 1.0) as i64;
    for py in y0..=y1 {
        for px in x0..=x1 {
            let d = ((px as f32 - cx).powi(2) + (py as f32 - cy).powi(2)).sqrt();
            if d > radius {
                continue;
            }
            let mut window = [[0f32; 3]; 9];
            let mut count = 0usize;
            for wy in -1i64..=1 {
                for wx in -1i64..=1 {
                    let sx = px + wx;
                    let sy = py + wy;
                    if sx < 0 || sy < 0 || sx >= width as i64 || sy >= height as i64 {
                        continue;
                    }
                    let s = (sy as usize * width as usize + sx as usize) * 4;
                    if s + 3 >= src.len() {
                        continue;
                    }
                    window[count] = [src[s] as f32, src[s + 1] as f32, src[s + 2] as f32];
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            let mut median = [0f32; 3];
            for k in 0..3 {
                let mut v: Vec<f32> = window[..count].iter().map(|c| c[k]).collect();
                v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                median[k] = v[count / 2];
            }
            let p = (py as usize * width as usize + px as usize) * 4;
            if p + 3 >= rgba.len() {
                continue;
            }
            for k in 0..3 {
                rgba[p + k] = median[k].clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Bilinear sample with edge clamping; `None` only for an empty image.
fn sample(rgba: &[u8], width: u32, height: u32, x: f32, y: f32) -> Option<[f32; 3]> {
    if width == 0 || height == 0 {
        return None;
    }
    let fx = x.clamp(0.0, width as f32 - 1.0);
    let fy = y.clamp(0.0, height as f32 - 1.0);
    let x0 = fx.floor().clamp(0.0, width as f32 - 1.0) as usize;
    let y0 = fy.floor().clamp(0.0, height as f32 - 1.0) as usize;
    let x1 = (x0 + 1).min(width as usize - 1);
    let y1 = (y0 + 1).min(height as usize - 1);
    let tx = fx - fx.floor();
    let ty = fy - fy.floor();
    let get = |xx: usize, yy: usize| -> [f32; 3] {
        let i = (yy * width as usize + xx) * 4;
        if i + 3 >= rgba.len() {
            return [0.0; 3];
        }
        [rgba[i] as f32, rgba[i + 1] as f32, rgba[i + 2] as f32]
    };
    let p00 = get(x0, y0);
    let p10 = get(x1, y0);
    let p01 = get(x0, y1);
    let p11 = get(x1, y1);
    let mut out = [0f32; 3];
    for k in 0..3 {
        let top = p00[k] * (1.0 - tx) + p10[k] * tx;
        let bottom = p01[k] * (1.0 - tx) + p11[k] * tx;
        out[k] = top * (1.0 - ty) + bottom * ty;
    }
    Some(out)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x >= edge1 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(w: u32, h: u32, v: u8) -> Vec<u8> {
        let mut buf = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            buf.extend_from_slice(&[v, v, v, 255]);
        }
        buf
    }

    fn spot(kind: SpotKind, x: f32, y: f32, radius: f32) -> Spot {
        Spot { id: 1, kind, x, y, radius }
    }

    #[test]
    fn empty_spots_leave_bytes_untouched() {
        let mut buf = gray(16, 16, 128);
        let before = buf.clone();
        let adj = AdjustmentSet::default();
        apply_spots(&mut buf, 16, 16, &adj, (16, 16));
        assert_eq!(buf, before);
    }

    #[test]
    fn heal_removes_a_black_dot_from_a_flat_field() {
        let mut buf = gray(100, 100, 200);
        for y in 49..=51 {
            for x in 49..=51 {
                let p = (y * 100 + x) * 4;
                buf[p] = 0;
                buf[p + 1] = 0;
                buf[p + 2] = 0;
            }
        }
        let mut adj = AdjustmentSet::default();
        adj.spots.push(spot(SpotKind::Spot, 0.5, 0.5, 0.05));
        apply_spots(&mut buf, 100, 100, &adj, (100, 100));
        let p = (50 * 100 + 50) * 4;
        assert!(buf[p] > 180, "healed centre is {}", buf[p]);
        assert!(buf[p] < 220, "healed centre is {}", buf[p]);
    }

    #[test]
    fn heal_removes_a_bright_spot_from_a_smooth_gradient() {
        let (w, h) = (100u32, 100u32);
        let base = |x: u32, y: u32| 90.0 + x as f32 * 0.6 + y as f32 * 0.2;
        let mut buf = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = base(x, y).round() as u8;
                buf.extend_from_slice(&[v, v, v, 255]);
            }
        }
        // A bright blemish inside the 5 px heal radius.
        for y in 46..=54 {
            for x in 46..=54 {
                if ((x as f32 - 50.0).powi(2) + (y as f32 - 50.0).powi(2)).sqrt() <= 4.0 {
                    let p = ((y * 100 + x) * 4) as usize;
                    let v = (base(x, y) + 70.0).min(255.0) as u8;
                    buf[p..p + 3].fill(v);
                }
            }
        }
        let before = buf.clone();
        let mut adj = AdjustmentSet::default();
        adj.spots.push(spot(SpotKind::Spot, 0.5, 0.5, 0.05));
        apply_spots(&mut buf, w, h, &adj, (w, h));

        // Surrounding mean from an untouched ring 8 px out; the gradient is linear,
        // so this is also the value the centre should have had without the blemish.
        let mut sum = 0f32;
        let mut n = 0f32;
        for i in 0..64 {
            let a = i as f32 / 64.0 * std::f32::consts::TAU;
            let x = (50.0 + 8.0 * a.cos()).round() as u32;
            let y = (50.0 + 8.0 * a.sin()).round() as u32;
            sum += before[((y * 100 + x) * 4) as usize] as f32;
            n += 1.0;
        }
        let mean = sum / n;
        let centre = buf[(50 * 100 + 50) * 4] as f32;
        assert!(
            (centre - mean).abs() < 2.0,
            "healed centre {centre} differs from surrounding mean {mean}"
        );

        // Nothing beyond 3 px past the heal radius may move.
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f32 - 50.0).powi(2) + (y as f32 - 50.0).powi(2)).sqrt();
                if d > 5.0 + 3.0 {
                    let p = ((y * 100 + x) * 4) as usize;
                    assert_eq!(&buf[p..p + 3], &before[p..p + 3], "pixel ({x},{y}) outside the patch changed");
                }
            }
        }
    }

    #[test]
    fn heal_keeps_fine_texture_from_the_source_patch() {
        let (w, h) = (100u32, 100u32);
        let mut buf = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let v = (128 + if (x + y) % 2 == 0 { 12 } else { -12 }) as u8;
                buf.extend_from_slice(&[v, v, v, 255]);
            }
        }
        for y in 48..=52 {
            for x in 48..=52 {
                let p = ((y * 100 + x) * 4) as usize;
                buf[p..p + 3].fill(220);
            }
        }
        let mut adj = AdjustmentSet::default();
        adj.spots.push(spot(SpotKind::Spot, 0.5, 0.5, 0.05));
        apply_spots(&mut buf, w, h, &adj, (w, h));

        let energy = |b: &[u8], cx: u32, cy: u32| -> f32 {
            let mut sum = 0f32;
            let mut n = 0f32;
            for y in cy - 2..=cy + 2 {
                for x in cx - 2..=cx + 2 {
                    let p = ((y * 100 + x) * 4) as usize;
                    let around = [-1i32, 0, 1, 0]
                        .into_iter()
                        .zip([0i32, -1, 0, 1])
                        .map(|(dx, dy)| b[(((y as i32 + dy) * 100 + x as i32 + dx) * 4) as usize] as f32)
                        .sum::<f32>()
                        / 4.0;
                    sum += (b[p] as f32 - around).abs();
                    n += 1.0;
                }
            }
            sum / n
        };
        let control = energy(&buf, 25, 25);
        let healed = energy(&buf, 50, 50);
        assert!(
            healed > control * 0.5,
            "texture energy in the healed patch {healed} is far below a clean patch {control}"
        );
    }

    #[test]
    fn out_of_range_spots_are_ignored() {
        let mut buf = gray(32, 32, 77);
        let before = buf.clone();
        let mut adj = AdjustmentSet::default();
        adj.spots.push(spot(SpotKind::Spot, 5.0, -3.0, 0.1));
        adj.spots.push(spot(SpotKind::Dust, 0.5, 0.5, 0.0));
        apply_spots(&mut buf, 32, 32, &adj, (32, 32));
        assert_eq!(buf, before);
    }

    #[test]
    fn empty_image_does_not_panic() {
        let mut buf: Vec<u8> = Vec::new();
        let mut adj = AdjustmentSet::default();
        adj.spots.push(spot(SpotKind::Spot, 0.5, 0.5, 0.05));
        apply_spots(&mut buf, 0, 0, &adj, (0, 0));
    }

    #[test]
    fn detected_spots_at_a_manual_position_are_applied_once() {
        let mut manual_only = gray(100, 100, 200);
        for y in 49..=51 {
            for x in 49..=51 {
                let p = (y * 100 + x) * 4;
                manual_only[p..p + 3].fill(0);
            }
        }
        let mut combined = manual_only.clone();
        let mut adj = AdjustmentSet::default();
        adj.spots.push(spot(SpotKind::Spot, 0.5, 0.5, 0.05));
        apply_spots(&mut manual_only, 100, 100, &adj, (100, 100));

        let dust = [spot(SpotKind::Dust, 0.5, 0.5, 0.05)];
        apply_detected_spots(&mut combined, 100, 100, &adj, &Geometry::default(), &dust, (100, 100));
        assert_eq!(combined, manual_only, "the detected duplicate ran a second time");
    }

    #[test]
    fn detected_spots_are_applied_where_no_manual_spot_exists() {
        let mut buf = gray(100, 100, 200);
        // One pixel: the 3×3 median's surroundings must stay clean to restore it.
        let p = (50 * 100 + 30) * 4;
        buf[p..p + 3].fill(0);
        let adj = AdjustmentSet::default();
        let dust = [spot(SpotKind::Dust, 0.3, 0.5, 0.05)];
        apply_detected_spots(&mut buf, 100, 100, &adj, &Geometry::default(), &dust, (100, 100));
        assert!(buf[p] > 150, "dust was not removed, centre is {}", buf[p]);
    }

    #[test]
    fn out_of_range_detected_spots_are_ignored() {
        let mut buf = gray(32, 32, 77);
        let before = buf.clone();
        let adj = AdjustmentSet::default();
        let dust = [
            spot(SpotKind::Dust, -1.0, 2.0, 0.1),
            spot(SpotKind::Dust, 0.5, 0.5, 0.0),
        ];
        apply_detected_spots(&mut buf, 32, 32, &adj, &Geometry::default(), &dust, (32, 32));
        assert_eq!(buf, before);
    }

    fn pattern(w: u32, h: u32) -> Vec<u8> {
        let mut buf = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let px = [
                    ((x * 3 + y * 7) % 251) as u8,
                    ((x * 11 + y * 5) % 241) as u8,
                    ((x * 13 + y * 17) % 233) as u8,
                    255,
                ];
                buf.extend_from_slice(&px);
            }
        }
        buf
    }

    fn clone_patch_full_snapshot_reference(
        rgba: &mut [u8],
        width: u32,
        height: u32,
        cx: f32,
        cy: f32,
        radius: f32,
        src: (f32, f32),
    ) {
        let radius = radius.max(1.0);
        let feather = (radius * FEATHER_RATIO).max(1.0);
        let outer = radius + feather;
        let source = rgba.to_vec();
        let x0 = (cx - outer).floor().max(0.0) as i64;
        let x1 = (cx + outer).ceil().min(width as f32 - 1.0) as i64;
        let y0 = (cy - outer).floor().max(0.0) as i64;
        let y1 = (cy + outer).ceil().min(height as f32 - 1.0) as i64;
        for py in y0..=y1 {
            for px in x0..=x1 {
                let d = ((px as f32 - cx).powi(2) + (py as f32 - cy).powi(2)).sqrt();
                if d > outer {
                    continue;
                }
                let w = if d <= radius { 1.0 } else { 1.0 - smoothstep(radius, outer, d) };
                if w <= 0.0 {
                    continue;
                }
                let p = (py as usize * width as usize + px as usize) * 4;
                if p + 3 >= rgba.len() {
                    continue;
                }
                let c = sample(
                    &source,
                    width,
                    height,
                    px as f32 + src.0,
                    py as f32 + src.1,
                )
                .unwrap();
                for k in 0..3 {
                    let orig = source[p + k] as f32;
                    rgba[p + k] = (orig * (1.0 - w) + c[k] * w).clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    #[test]
    fn clone_roi_snapshot_matches_full_frame_reference() {
        let (w, h) = (61u32, 47u32);
        let base = pattern(w, h);
        let args = (30.3, 23.7, 12.2, (17.25, -8.5));
        let mut expected = base.clone();
        clone_patch_full_snapshot_reference(&mut expected, w, h, args.0, args.1, args.2, args.3);
        let mut actual = base;
        clone_patch(&mut actual, w, h, args.0, args.1, args.2, args.3);
        assert_eq!(actual, expected);
    }

    #[test]
    fn clone_copies_the_source_point() {
        let (w, h) = (64u32, 64u32);
        let mut buf = pattern(w, h);
        let before = buf.clone();
        let mut adj = AdjustmentSet::default();
        // Centre (16,16), source (48,16): offset (32,0) in source pixels.
        adj.spots.push(spot(SpotKind::Clone { src_offset: (32, 0) }, 0.25, 0.25, 0.1));
        apply_spots(&mut buf, w, h, &adj, (w, h));
        for (dx, dy) in [(0i32, 0i32), (2, 1), (-1, -2)] {
            let dp = (((16 + dy) * 64 + (16 + dx)) * 4) as usize;
            let sp = (((16 + dy) * 64 + (48 + dx)) * 4) as usize;
            for k in 0..3 {
                let diff = buf[dp + k] as i32 - before[sp + k] as i32;
                assert!(
                    diff.abs() <= 1,
                    "channel {k} at ({},{}) is {} but the source has {}",
                    dx,
                    dy,
                    buf[dp + k],
                    before[sp + k]
                );
            }
        }
    }

    #[test]
    fn clone_scales_the_source_offset_to_the_frame() {
        let (w, h) = (128u32, 128u32);
        let mut buf = pattern(w, h);
        let before = buf.clone();
        let mut adj = AdjustmentSet::default();
        // A 64 px source frame shown at 128 px: the 16 px offset doubles.
        adj.spots.push(spot(SpotKind::Clone { src_offset: (16, 0) }, 0.25, 0.25, 0.1));
        apply_spots(&mut buf, w, h, &adj, (64, 64));
        let dp = (32 * 128 + 32) * 4;
        let sp = (32 * 128 + 64) * 4;
        for k in 0..3 {
            let diff = buf[dp + k] as i32 - before[sp + k] as i32;
            assert!(diff.abs() <= 1, "channel {k}: {} vs source {}", buf[dp + k], before[sp + k]);
        }
    }

    #[test]
    fn clone_feathers_its_edge() {
        let (w, h) = (96u32, 96u32);
        let mut buf = gray(w, h, 100);
        for y in 0..h {
            for x in 50..90 {
                let p = ((y * 96 + x) * 4) as usize;
                buf[p..p + 3].fill(220);
            }
        }
        let before = buf.clone();
        let mut adj = AdjustmentSet::default();
        adj.spots.push(spot(SpotKind::Clone { src_offset: (40, 0) }, 0.3, 0.5, 10.0 / 96.0));
        apply_spots(&mut buf, w, h, &adj, (w, h));

        // The inner disc is the source patch; nothing 4 px past the feather moves.
        let centre = ((48 * 96 + 29) * 4) as usize;
        assert_eq!(buf[centre], 220, "clone centre is {}", buf[centre]);
        let mut partial = 0;
        for y in 0..h {
            for x in 0..w {
                let d = ((x as f32 - 28.8).powi(2) + (y as f32 - 48.0).powi(2)).sqrt();
                let p = ((y * 96 + x) * 4) as usize;
                if d > 10.0 + 10.0 * 0.35 + 4.0 {
                    assert_eq!(&buf[p..p + 3], &before[p..p + 3], "pixel ({x},{y}) outside moved");
                }
                if buf[p] > 105 && buf[p] < 215 {
                    partial += 1;
                }
                // Steps are judged inside the clone only: the source stripe's own
                // edge at x = 50 is a hard step in the input.
                if (1..45).contains(&x) {
                    let step = (buf[p] as i32 - buf[p - 4] as i32).abs();
                    assert!(step <= 60, "hard seam at ({x},{y}): step {step}");
                }
            }
        }
        assert!(partial > 10, "only {partial} feathered pixels, the edge looks cut");
    }

    #[test]
    fn heal_output_matches_the_pre_clone_golden() {
        let (w, h) = (100u32, 100u32);
        let base = |x: u32, y: u32| 90.0 + x as f32 * 0.6 + y as f32 * 0.2;
        let mut buf = Vec::new();
        for y in 0..h {
            for x in 0..w {
                let v = base(x, y).round() as u8;
                buf.extend_from_slice(&[v, v, v, 255]);
            }
        }
        for y in 40..=60 {
            for x in 40..=60 {
                let d = ((x as f32 - 50.0).powi(2) + (y as f32 - 50.0).powi(2)).sqrt();
                if d <= 4.0 {
                    let p = ((y * 100 + x) * 4) as usize;
                    let v = (base(x, y) + 70.0).min(255.0) as u8;
                    buf[p..p + 3].fill(v);
                }
            }
        }
        let mut adj = AdjustmentSet::default();
        adj.spots.push(spot(SpotKind::Spot, 0.5, 0.5, 0.05));
        apply_spots(&mut buf, w, h, &adj, (w, h));
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for b in &buf {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
        assert_eq!(hash, 0x51ee_30b0_e822_fad4, "heal output changed: {hash:016x}");
    }
}
