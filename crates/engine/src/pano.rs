//! Horizontal panorama stitch (translation only, no rotation, perspective or
//! deghosting).
//!
//! Each neighbouring pair is aligned by a 1-D search over column offsets; the
//! shift with the smallest mean absolute difference over the overlap wins. The
//! second frame is also searched +/-10 % of the frame height for vertical
//! misalignment. Per-channel gains computed from the overlap means match the
//! second frame's exposure and white balance to the first. The frames are then
//! pasted left to right (and up/down by the found offset) and the overlap is
//! cross-faded with a linear ramp, so the seam has no hard edge.

use anyhow::{bail, Result};
use rayon::prelude::*;

use crate::hdr::FusedFrame;
use crate::mock::process::{linear_to_srgb, Plane};
use crate::Source;

/// Half-width of the offset search around a 50 % overlap, as a fraction of the
/// narrower frame: 25 % ..= 75 % of the frame width.
const SEARCH_BAND: f32 = 0.25;
/// Rows sampled by the SAD search; full-resolution rows, evenly spread.
const SAD_ROWS: u32 = 64;
/// Coarse SAD steps across the search band; the winner is refined by one step.
const SAD_STEPS: u32 = 256;
/// Vertical offset search range, as a fraction of the frame height.
const VERTICAL_BAND: f32 = 0.10;
/// Coarse vertical SAD steps across the whole band.
const VERTICAL_STEPS: u32 = 32;
/// Per-channel exposure/WB gain is clamped to this range.
const MAX_GAIN: f32 = 4.0;

/// Stitches `frames` left to right into one `FusedFrame` (linear RGB).
///
/// Frames must share one height; widths may differ. A single frame is returned
/// unchanged.
pub fn stitch(frames: &[&Source]) -> Result<FusedFrame> {
    let Some(first) = frames.first() else {
        bail!("Panorama merge needs at least one frame");
    };
    let (width0, height) = first.size();
    if let Some(&bad) = frames.iter().find(|f| f.size().1 != height) {
        let (w, h) = bad.size();
        bail!("Panorama merge frames must share one height: {height} vs {h} ({w}px wide)");
    }
    if frames.len() == 1 {
        return Ok(FusedFrame { width: width0, height, rgb: first.levels[0].data.clone() });
    }

    let offsets: Vec<(u32, i32)> = frames
        .windows(2)
        .map(|pair| best_offset(&pair[0].levels[0], &pair[1].levels[0]))
        .collect();
    let mut xs = Vec::with_capacity(frames.len());
    let mut ys = Vec::with_capacity(frames.len());
    xs.push(0u32);
    ys.push(0i32);
    for &(dx, dy) in &offsets {
        xs.push(xs.last().unwrap() + dx);
        ys.push(ys.last().unwrap() + dy);
    }
    let canvas_width = xs.last().unwrap() + frames.last().unwrap().size().0;
    let min_y = *ys.iter().min().unwrap();
    let max_y = ys
        .iter()
        .zip(frames)
        .map(|(y, f)| *y as i64 + f.size().1 as i64)
        .max()
        .unwrap();
    let canvas_height = (max_y - min_y as i64) as u32;
    let tops: Vec<usize> = ys.iter().map(|y| (y - min_y) as usize).collect();

    // Chain the per-pair gains so every frame ends up in the first frame's
    // exposure/WB; a pair is measured on raw pixels of the previous frame.
    let mut gains = vec![[1.0f32; 3]];
    for i in 1..frames.len() {
        let local = pair_gain(
            &frames[i - 1].levels[0],
            &frames[i].levels[0],
            offsets[i - 1].0,
            offsets[i - 1].1,
        );
        let prev = gains[i - 1];
        gains.push([prev[0] * local[0], prev[1] * local[1], prev[2] * local[2]]);
    }

    let mut rgb = vec![[0.0f32; 3]; canvas_width as usize * canvas_height as usize];

    rgb.par_chunks_mut(canvas_width as usize).enumerate().for_each(|(y, row)| {
        for (i, frame) in frames.iter().enumerate() {
            let plane = &frame.levels[0];
            let top = tops[i];
            if y < top || y >= top + plane.height as usize {
                continue;
            }
            let x0 = xs[i] as usize;
            // The new frame's first `overlap` columns sit on the composite of
            // the previous frame; ramp the new frame in across them. Where the
            // vertical offset leaves no previous pixel on this row, write
            // directly instead of fading towards empty canvas.
            let overlap = if i == 0 {
                0
            } else {
                (frames[i - 1].size().0 - offsets[i - 1].0).min(plane.width)
            };
            let (prev_top, prev_bottom) = if i == 0 {
                (0, 0)
            } else {
                (tops[i - 1], tops[i - 1] + frames[i - 1].size().1 as usize)
            };
            let gain = gains[i];
            for c in 0..plane.width as usize {
                let raw = plane.data[(y - top) * plane.width as usize + c];
                let pixel = [raw[0] * gain[0], raw[1] * gain[1], raw[2] * gain[2]];
                let dst = x0 + c;
                if c < overlap as usize && y >= prev_top && y < prev_bottom {
                    let alpha = (c as f32 + 0.5) / overlap as f32;
                    let under = row[dst];
                    row[dst] = [
                        under[0] + (pixel[0] - under[0]) * alpha,
                        under[1] + (pixel[1] - under[1]) * alpha,
                        under[2] + (pixel[2] - under[2]) * alpha,
                    ];
                } else {
                    row[dst] = pixel;
                }
            }
        }
    });

    Ok(FusedFrame { width: canvas_width, height: canvas_height, rgb })
}

/// 8-bit sRGB samples for a TIFF, using the engine's own transfer function.
pub fn to_srgb8(frame: &FusedFrame) -> Vec<u8> {
    let mut out = vec![0u8; frame.rgb.len() * 3];
    out.par_chunks_mut(3).zip(frame.rgb.par_iter()).for_each(|(out, pixel)| {
        for k in 0..3 {
            out[k] = (linear_to_srgb(pixel[k].clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
        }
    });
    out
}

/// Offset of `b`'s top-left corner relative to `a`, in pixels.
fn best_offset(a: &Plane, b: &Plane) -> (u32, i32) {
    let min_width = a.width.min(b.width);
    let lo = (min_width as f32 * (0.5 - SEARCH_BAND)) as u32;
    let hi = (min_width as f32 * (0.5 + SEARCH_BAND)) as u32;
    let step = ((hi - lo) / SAD_STEPS).max(1);
    let mut best_x = lo;
    let mut best_cost = f64::INFINITY;
    let mut x = lo;
    while x <= hi {
        let cost = sad(a, b, x, 0);
        if cost < best_cost {
            best_cost = cost;
            best_x = x;
        }
        x += step;
    }
    best_x = refine_x(a, b, best_x, step, lo, hi, 0).0;

    let span = (b.height as f32 * VERTICAL_BAND) as i32;
    let y_step = ((2 * span) as u32 / VERTICAL_STEPS).max(1) as i32;
    let mut best_y = 0;
    best_cost = sad(a, b, best_x, 0);
    for y in (-span..=span).step_by(y_step as usize) {
        let cost = sad(a, b, best_x, y);
        if cost < best_cost {
            best_cost = cost;
            best_y = y;
        }
    }
    best_y = refine_y(a, b, best_x, best_y, y_step, span).0;
    best_x = refine_x(a, b, best_x, step, lo, hi, best_y).0;
    best_y = refine_y(a, b, best_x, best_y, y_step, span).0;
    (best_x, best_y)
}

/// One-pixel refinement of the column offset around `best`.
fn refine_x(a: &Plane, b: &Plane, best: u32, step: u32, lo: u32, hi: u32, sy: i32) -> (u32, f64) {
    let from = best.saturating_sub(step).max(lo);
    let to = (best + step).min(hi);
    let mut winner = (best, f64::INFINITY);
    for x in from..=to {
        let cost = sad(a, b, x, sy);
        if cost < winner.1 {
            winner = (x, cost);
        }
    }
    winner
}

/// One-pixel refinement of the vertical offset around `best`.
fn refine_y(a: &Plane, b: &Plane, sx: u32, best: i32, step: i32, span: i32) -> (i32, f64) {
    let from = (best - step).max(-span);
    let to = (best + step).min(span);
    let mut winner = (best, f64::INFINITY);
    for y in from..=to {
        let cost = sad(a, b, sx, y);
        if cost < winner.1 {
            winner = (y, cost);
        }
    }
    winner
}

/// Per-channel gain that matches `b`'s overlap mean to `a`'s (exposure + WB).
fn pair_gain(a: &Plane, b: &Plane, sx: u32, sy: i32) -> [f32; 3] {
    let overlap = a.width.saturating_sub(sx).min(b.width);
    let first = (-sy).max(0);
    let last = (a.height as i32 - sy).min(b.height as i32);
    if overlap == 0 || last <= first {
        return [1.0; 3];
    }
    let mut mean_a = [0.0f64; 3];
    let mut mean_b = [0.0f64; 3];
    for y in first..last {
        let a_row = (y + sy) as usize * a.width as usize + sx as usize;
        let b_row = y as usize * b.width as usize;
        for c in 0..overlap as usize {
            let pa = a.data[a_row + c];
            let pb = b.data[b_row + c];
            for k in 0..3 {
                mean_a[k] += pa[k] as f64;
                mean_b[k] += pb[k] as f64;
            }
        }
    }
    let n = (last - first) as f64 * overlap as f64;
    let mut gain = [1.0f32; 3];
    for k in 0..3 {
        let mb = mean_b[k] / n;
        if mb > 1e-6 {
            let ratio = (mean_a[k] / n) / mb;
            gain[k] = ratio.clamp(1.0 / MAX_GAIN as f64, MAX_GAIN as f64) as f32;
        }
    }
    gain
}

/// Mean absolute RGB difference over the overlap for one candidate offset.
///
/// Each side's per-channel overlap mean is removed first, so an exposure or WB
/// difference scales the texture instead of dominating the match.
fn sad(a: &Plane, b: &Plane, sx: u32, sy: i32) -> f64 {
    let overlap = a.width.saturating_sub(sx).min(b.width);
    if overlap == 0 {
        return f64::INFINITY;
    }
    let first = (-sy).max(0);
    let last = (a.height as i32 - sy).min(b.height as i32);
    if last <= first {
        return f64::INFINITY;
    }
    let valid = (last - first) as u32;
    let rows = valid.min(SAD_ROWS);
    let row_at = |r: u32| first as usize + (r as u64 * valid as u64 / rows as u64) as usize;
    let mut sum_a = [0.0f64; 3];
    let mut sum_b = [0.0f64; 3];
    for r in 0..rows {
        let by = row_at(r);
        let ay = (by as i32 + sy) as usize;
        let a0 = ay * a.width as usize + sx as usize;
        let b0 = by * b.width as usize;
        for c in 0..overlap as usize {
            let pa = a.data[a0 + c];
            let pb = b.data[b0 + c];
            for k in 0..3 {
                sum_a[k] += pa[k] as f64;
                sum_b[k] += pb[k] as f64;
            }
        }
    }
    let n = rows as f64 * overlap as f64;
    let mean_a = sum_a.map(|v| v / n);
    let mean_b = sum_b.map(|v| v / n);
    let mut sum = 0.0f64;
    for r in 0..rows {
        let by = row_at(r);
        let ay = (by as i32 + sy) as usize;
        let a0 = ay * a.width as usize + sx as usize;
        let b0 = by * b.width as usize;
        for c in 0..overlap as usize {
            let pa = a.data[a0 + c];
            let pb = b.data[b0 + c];
            sum += (((pa[0] as f64 - mean_a[0]) - (pb[0] as f64 - mean_b[0])).abs()
                + ((pa[1] as f64 - mean_a[1]) - (pb[1] as f64 - mean_b[1])).abs()
                + ((pa[2] as f64 - mean_a[2]) - (pb[2] as f64 - mean_b[2])).abs()) as f64;
        }
    }
    sum / (n * 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wider scene split into overlapping crops, so the true shift is known.
    fn scene(width: u32, height: u32) -> Source {
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let t = x as f32 / width as f32;
                // Gentle, unique column profile: the SAD search needs no two
                // columns alike, but the scene itself must not have hard edges.
                let band = (t * 7.0).sin() * 0.03 + (t * 19.0).sin() * 0.015;
                // Stay off black: the sRGB curve amplifies tiny linear steps
                // near zero into visible 8-bit jumps.
                let p = [0.15 + 0.7 * t + band, 0.35 + 0.2 * (y as f32 / height as f32), 0.8 - 0.5 * t];
                let q = p.map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16);
                rgba.extend_from_slice(&[q[0], q[1], q[2], u16::MAX]);
            }
        }
        Source::from_rgba16(width, height, &rgba)
    }

    /// Crop of `scene` starting at `left`.
    fn crop(scene: &Source, left: u32, width: u32) -> Source {
        let (sw, h) = scene.size();
        let mut rgba = Vec::with_capacity((width * h * 4) as usize);
        for y in 0..h {
            for x in left..left + width {
                let px = scene.levels[0].data[(y * sw + x) as usize];
                let q = px.map(|v| (v * 65535.0).round() as u16);
                rgba.extend_from_slice(&[q[0], q[1], q[2], u16::MAX]);
            }
        }
        Source::from_rgba16(width, h, &rgba)
    }

    /// Crop of `scene` starting at `(left, top)`.
    fn crop_xy(scene: &Source, left: u32, top: u32, width: u32, height: u32) -> Source {
        let sw = scene.size().0;
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in top..top + height {
            for x in left..left + width {
                let px = scene.levels[0].data[(y * sw + x) as usize];
                let q = px.map(|v| (v * 65535.0).round() as u16);
                rgba.extend_from_slice(&[q[0], q[1], q[2], u16::MAX]);
            }
        }
        Source::from_rgba16(width, height, &rgba)
    }

    /// Flat, per-pixel-textured scene: equal means everywhere, unique columns.
    fn textured_scene(width: u32, height: u32) -> Source {
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let h = x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663);
                let n = ((h >> 9) & 0xff) as f32 / 255.0 - 0.5;
                let p = [0.5 + 0.12 * n, 0.45 + 0.12 * n, 0.55 - 0.10 * n];
                let q = p.map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16);
                rgba.extend_from_slice(&[q[0], q[1], q[2], u16::MAX]);
            }
        }
        Source::from_rgba16(width, height, &rgba)
    }

    fn srgb8(p: [f32; 3]) -> [u8; 3] {
        p.map(|v| (linear_to_srgb(v.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8)
    }

    #[test]
    fn two_translated_frames_keep_the_overlap_width_and_leave_no_hard_seam() {
        let wide = scene(240, 40);
        let (a, b) = (crop(&wide, 0, 160), crop(&wide, 80, 160));
        let fused = stitch(&[&a, &b]).unwrap();
        assert_eq!(fused.width, 240, "the 80 px overlap was not found");
        assert_eq!(fused.height, 40);

        // The ramp spans the whole overlap, so no neighbouring columns jump.
        let pixels = to_srgb8(&fused);
        let stride = fused.width as usize * 3;
        for y in 0..fused.height as usize {
            for x in 1..fused.width as usize {
                for k in 0..3 {
                    let d = (pixels[y * stride + x * 3 + k] as i16
                        - pixels[y * stride + (x - 1) * 3 + k] as i16)
                        .abs();
                    assert!(d <= 8, "hard seam at ({x},{y}) channel {k}: delta {d}");
                }
            }
        }
    }

    #[test]
    fn a_single_frame_is_returned_unchanged() {
        let mut rgba = Vec::new();
        for y in 0..16u32 {
            for x in 0..32u32 {
                rgba.extend_from_slice(&[(x * 7) as u8, (y * 15) as u8, 128, 255]);
            }
        }
        let src = Source::from_rgba(32, 16, &rgba);
        let fused = stitch(&[&src]).unwrap();
        assert_eq!((fused.width, fused.height), (32, 16));
        let out = to_srgb8(&fused);
        let rgb: Vec<u8> = rgba.chunks_exact(4).flat_map(|p| [p[0], p[1], p[2]]).collect();
        for (a, b) in out.iter().zip(&rgb) {
            assert!(a.abs_diff(*b) <= 1, "pixel changed: {a} vs {b}");
        }
    }

    #[test]
    fn an_80_percent_exposure_frame_is_matched_across_the_seam() {
        let wide = textured_scene(240, 48);
        let a = crop_xy(&wide, 0, 0, 160, 48);
        let mut b = crop_xy(&wide, 80, 0, 160, 48);
        for pixel in &mut b.levels[0].data {
            for v in pixel {
                *v *= 0.8;
            }
        }
        let fused = stitch(&[&a, &b]).unwrap();
        assert_eq!(fused.width, 240, "the 80 px overlap was not found");
        assert_eq!(fused.height, 48);

        // Overlap is canvas x 80..160. Compare the mostly-first and
        // mostly-second halves; without compensation the second is 20 % dark.
        let pixels = to_srgb8(&fused);
        let stride = fused.width as usize * 3;
        let mean = |x0: u32, x1: u32| {
            let mut sum = [0.0f64; 3];
            for y in 0..fused.height as usize {
                for x in x0 as usize..x1 as usize {
                    for k in 0..3 {
                        sum[k] += pixels[y * stride + x * 3 + k] as f64;
                    }
                }
            }
            let n = (x1 - x0) as f64 * fused.height as f64;
            sum.map(|v| v / n)
        };
        let left = mean(81, 91);
        let right = mean(149, 159);
        for k in 0..3 {
            assert!(
                (left[k] - right[k]).abs() < 2.0,
                "seam sides differ on channel {k}: {:.2} vs {:.2}",
                left[k],
                right[k]
            );
        }
    }

    #[test]
    fn a_five_pixel_vertical_offset_is_aligned() {
        let wide = textured_scene(240, 90);
        let a = crop_xy(&wide, 0, 0, 160, 80);
        let b = crop_xy(&wide, 80, 5, 160, 80);
        let fused = stitch(&[&a, &b]).unwrap();
        assert_eq!(fused.width, 240, "the 80 px horizontal overlap was not found");
        assert_eq!(fused.height, 85, "the 5 px vertical offset was not applied");

        // A pixel only the second frame covers must show the scene at its
        // vertically shifted position, not the unshifted one.
        let pixels = to_srgb8(&fused);
        let stride = fused.width as usize * 3;
        for (x, y) in [(200u32, 60u32), (150, 50), (120, 70)] {
            let expected = srgb8(wide.levels[0].data[(y * 240 + x) as usize]);
            let at = y as usize * stride + x as usize * 3;
            for k in 0..3 {
                assert!(
                    pixels[at + k].abs_diff(expected[k]) <= 1,
                    "({x},{y}) channel {k}: {} vs {}",
                    pixels[at + k],
                    expected[k]
                );
            }
        }
    }
}
