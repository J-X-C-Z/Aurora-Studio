//! Spot removal for a developed RGBA frame: dust (median) and heal (Laplacian
//! diffusion of the surrounding ring).
//! Approximation only — no texture synthesis, no seam matching, and the ring
//! is sampled at 32 angles, so healed areas lose fine detail.
//! Limits: radii follow the spot radius fraction of the oriented image's long side.

use crate::{AdjustmentSet, Geometry, Spot, SpotKind};

/// What a spot does to the pixels it covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SpotMode {
    /// Diffusion fill from the surrounding ring.
    Heal,
    /// 3x3 median filter.
    Dust,
}

const TAU: f32 = std::f32::consts::TAU;
const RING_SAMPLES: usize = 32;
const FEATHER_RATIO: f32 = 0.35;
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
        let Some((cx, cy, radius_px)) = spot_geometry(spot, geometry, source_size, width, height)
        else {
            continue;
        };
        match mode {
            SpotMode::Heal => heal(rgba, width, height, cx, cy, radius_px),
            SpotMode::Dust => dust(rgba, width, height, cx, cy, radius_px),
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
        SpotKind::Dust => SpotMode::Dust,
        // `Spot` is the interface's "heal" spot; there is no clone spot kind.
        SpotKind::Spot => SpotMode::Heal,
    }
}

/// Maps a spot to frame pixels: centre in the developed frame and its radius.
fn spot_geometry(
    spot: &Spot,
    geometry: &Geometry,
    source_size: (u32, u32),
    width: u32,
    height: u32,
) -> Option<(f32, f32, f32)> {
    let (sw, sh) = source_size;
    if sw == 0 || sh == 0 || width == 0 || height == 0 {
        return None;
    }
    if !spot.x.is_finite() || !spot.y.is_finite() || !spot.radius.is_finite() {
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
    // The spot sits in the oriented image before crop and fine rotation; the
    // frame shows the crop rotated about the oriented centre, so undo that
    // rotation first and then undo the crop.
    let qx = spot.x * ow;
    let qy = spot.y * oh;
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
    // Radius is a fraction of the image's long side; convert to frame pixels.
    let long_side = ow.max(oh);
    let scale = (width as f32 / (crop.width * ow)).min(height as f32 / (crop.height * oh));
    let radius_px = spot.radius * long_side * scale;
    if !radius_px.is_finite() || radius_px <= 0.0 {
        return None;
    }
    Some((fx * width as f32, fy * height as f32, radius_px))
}

/// Heal: fills the disc with a diffusion of the surrounding ring.
fn heal(rgba: &mut [u8], width: u32, height: u32, cx: f32, cy: f32, radius: f32) {
    let radius = radius.max(1.0);
    let feather = (radius * FEATHER_RATIO).max(1.0);
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
            for k in 0..3 {
                let healed = vals[idx][k] + dev[k] + offset[k];
                let v = orig[idx][k] * (1.0 - w) + healed * w;
                rgba[p + k] = v.clamp(0.0, 255.0) as u8;
            }
        }
    }
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
}
