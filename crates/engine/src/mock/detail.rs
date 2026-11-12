//! Approximate sharpening, noise reduction, moire, lens correction and film grain.
//! These are display-referred 8-bit sRGB approximations for the preview pipeline, not the production renderer.
//! Known limits: no 16-bit path, no real lens profiles, moire is a fixed 3x3 low-pass, grain is hash noise.

use rayon::prelude::*;

use crate::{FilmGrain, GrainKind, LensCorrection, Moire, NoiseReduction, Sharpening};

/// Applies the detail stages to one sRGB 8-bit RGBA frame, in place.
///
/// `rgba` is row-major RGBA with no padding, `scale` is output pixels per source
/// pixel (`< 1` when the frame was downscaled) and converts source-pixel radii
/// into output-pixel radii. Every stage is skipped when its own parameters are
/// off, so an all-off adjustment set leaves the bytes untouched.
pub(crate) fn post_process(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    adj: &crate::AdjustmentSet,
    scale: f32,
) {
    post_process_with_lens(rgba, width, height, adj, scale, None);
}

/// Same as [`post_process`], with Capture One's measured lens tables when the
/// variant uses a lens profile.
pub(crate) fn post_process_with_lens(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    adj: &crate::AdjustmentSet,
    scale: f32,
    warp: Option<&crate::mock::lens_profile::LensWarp>,
) {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return;
    }
    let Some(expected) = w.checked_mul(h).and_then(|n| n.checked_mul(4)) else {
        return;
    };
    if rgba.len() != expected {
        return;
    }

    let nr = &adj.noise_reduction;
    let lens = &adj.lens;
    let any_active = nr.luminance > 0.0
        || nr.color > 0.0
        || nr.single_pixel > 0.0
        || adj.moire.amount > 0.0
        || adj.sharpening.amount > 0.0
        || lens.distortion != 0.0
        || lens.light_falloff != 0.0
        || lens.chromatic_aberration
        || lens.purple_fringing > 0.0
        || adj.film_grain.impact > 0.0;
    if !any_active {
        return;
    }
    let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };

    stage_luminance_nr(rgba, w, h, nr);
    stage_color_nr(rgba, w, h, nr);
    stage_moire(rgba, w, h, &adj.moire);
    stage_sharpening(rgba, w, h, &adj.sharpening, scale);
    stage_lens(rgba, w, h, lens, warp);
    stage_grain(rgba, w, h, &adj.film_grain);
}

// ───────────────────────────── shared helpers ─────────────────────────────

#[inline]
fn luma8(p: [u8; 3]) -> f32 {
    0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
}

#[inline]
fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

fn luma_plane(buf: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut plane = vec![0f32; w * h];
    plane.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = (y * w + x) * 4;
            row[x] = luma8([buf[i], buf[i + 1], buf[i + 2]]);
        }
    });
    plane
}

/// Rewrites RGB from a new luminance while keeping the original hue ratios.
/// `old_luma` is the plane the caller already computed from the same buffer, so
/// this does not recompute it per pixel.
fn apply_luma(buf: &mut [u8], w: usize, old_luma: &[f32], new_luma: &[f32]) {
    buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = x * 4;
            let p = [row[i], row[i + 1], row[i + 2]];
            let c = old_luma[y * w + x];
            let target = new_luma[y * w + x];
            if (target - c).abs() < 0.05 {
                continue;
            }
            let f = if c > 1.0 { target / c } else { 1.0 };
            for k in 0..3 {
                row[i + k] = (p[k] as f32 * f).clamp(0.0, 255.0).round() as u8;
            }
        }
    });
}

fn box_blur_h(src: &[f32], dst: &mut [f32], w: usize, _h: usize, r: i32) {
    let inv = 1.0 / (2 * r + 1) as f32;
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let base = y * w;
        for x in 0..w {
            let mut sum = 0.0f32;
            for k in -r..=r {
                sum += src[base + clampi(x as i32 + k, 0, w as i32 - 1) as usize];
            }
            row[x] = sum * inv;
        }
    });
}

fn box_blur_v(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: i32) {
    let inv = 1.0 / (2 * r + 1) as f32;
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        row.fill(0.0);
        // Contiguous rows let the compiler vectorize without reordering each sum.
        for k in -r..=r {
            let yy = clampi(y as i32 + k, 0, h as i32 - 1) as usize;
            for (sum, value) in row.iter_mut().zip(&src[yy * w..(yy + 1) * w]) {
                *sum += value;
            }
        }
        for value in row {
            *value *= inv;
        }
    });
}

/// Three box passes approximate a Gaussian of a comparable radius.
fn box_blur3(src: &[f32], w: usize, h: usize, r: i32) -> Vec<f32> {
    let mut a = src.to_vec();
    let mut b = vec![0f32; w * h];
    for _ in 0..3 {
        box_blur_h(&a, &mut b, w, h, r);
        std::mem::swap(&mut a, &mut b);
        box_blur_v(&a, &mut b, w, h, r);
        std::mem::swap(&mut a, &mut b);
    }
    a
}

fn mean3x3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut sum = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let xx = clampi(x as i32 + dx, 0, w as i32 - 1) as usize;
                    let yy = clampi(y as i32 + dy, 0, h as i32 - 1) as usize;
                    sum += src[yy * w + xx];
                }
            }
            row[x] = sum / 9.0;
        }
    });
    out
}

// ───────────────────────────── 1. luminance noise reduction ─────────────────────────────

/// Radius-1 bilateral filter on luminance: neighbours are weighted by how close
/// their luminance is to the centre, so edges survive while speckle is averaged
/// away. `details` narrows the luminance tolerance, which preserves edges more.
/// `single_pixel` additionally replaces pixels that differ from *all* eight
/// neighbours (hot pixels) with the median of those neighbours.
fn stage_luminance_nr(buf: &mut [u8], w: usize, h: usize, adj: &NoiseReduction) {
    let strength = (adj.luminance / 100.0).clamp(0.0, 1.0);
    let single = (adj.single_pixel / 100.0).clamp(0.0, 1.0);
    if strength <= 0.0 && single <= 0.0 {
        return;
    }
    let details = (adj.details / 100.0).clamp(0.0, 1.0);
    let sigma = ((4.0 + 60.0 * strength) / (1.0 + 3.0 * details)).max(0.5);
    let two_sigma2 = 2.0 * sigma * sigma;
    // The range weight only depends on |l - c| (0 ..= 255). A quarter-step
    // table replaces ten million `expf` calls with two loads and a lerp; the
    // interpolation error is far below the 8-bit write-back.
    const LUT_N: usize = 1025;
    let mut range_lut = [0f32; LUT_N];
    for (i, v) in range_lut.iter_mut().enumerate() {
        let d = i as f32 * 0.25;
        *v = (-(d * d) / two_sigma2).exp();
    }
    // Neighbours only read this immutable plane; RGB can be updated in place.
    // One replicated border removes the per-neighbour clamping.
    let pw = w + 2;
    let ph = h + 2;
    let pad = {
        let luma = luma_plane(buf, w, h);
        let mut pad = vec![0f32; pw * ph];
        pad.par_chunks_mut(pw).enumerate().for_each(|(y, row)| {
            let sy = y.saturating_sub(1).min(h - 1);
            row[0] = luma[sy * w];
            row[pw - 1] = luma[sy * w + w - 1];
            row[1..pw - 1].copy_from_slice(&luma[sy * w..sy * w + w]);
        });
        pad
    };
    const OFF: [(i32, i32); 8] = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];
    const SPATIAL: [f32; 8] =
        [1.0 / 3.0, 0.5, 1.0 / 3.0, 0.5, 0.5, 1.0 / 3.0, 0.5, 1.0 / 3.0];
    let row = w * 4;
    buf.par_chunks_mut(row).enumerate().for_each(|(y, out_row)| {
        for x in 0..w {
            let i = x * 4;
            let p = [out_row[i], out_row[i + 1], out_row[i + 2]];
            let c = pad[(y + 1) * pw + x + 1];
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            let mut neighbours = [0f32; 8];
            for (k, (dx, dy)) in OFF.iter().enumerate() {
                let l = pad[(y as i32 + 1 + dy) as usize * pw + (x as i32 + 1 + dx) as usize];
                neighbours[k] = l;
                let ad = (l - c).abs() * 4.0;
                let i0 = (ad as usize).min(LUT_N - 2);
                let t = ad - i0 as f32;
                let range = range_lut[i0] + (range_lut[i0 + 1] - range_lut[i0]) * t;
                let wt = range * SPATIAL[k];
                num += wt * l;
                den += wt;
            }
            let mut target = if den > 1e-6 { num / den } else { c };
            target = c + (target - c) * strength;
            if single > 0.0 {
                let mut isolated = true;
                for v in neighbours {
                    if (v - c).abs() <= 6.0 {
                        isolated = false;
                        break;
                    }
                }
                if isolated {
                    let mut sorted = neighbours;
                    sorted.sort_by(|a, b| a.total_cmp(b));
                    let median = (sorted[3] + sorted[4]) * 0.5;
                    target = target * (1.0 - single) + median * single;
                }
            }
            let new_l = target.clamp(0.0, 255.0);
            if (new_l - c).abs() >= 0.05 {
                let f = if c > 1.0 { new_l / c } else { 1.0 };
                for ch in 0..3 {
                    out_row[i + ch] = (p[ch] as f32 * f).clamp(0.0, 255.0).round() as u8;
                }
            }
        }
    });
}

// ───────────────────────────── 2. colour noise reduction ─────────────────────────────

/// Blurs only the chroma planes of a YCbCr approximation; luminance is copied
/// through untouched, so colour speckle disappears without softening detail.
fn stage_color_nr(buf: &mut [u8], w: usize, h: usize, adj: &NoiseReduction) {
    let strength = (adj.color / 100.0).clamp(0.0, 1.0);
    if strength <= 0.0 {
        return;
    }
    let radius = (1.0 + 2.0 * strength).round() as i32;
    let mut y_plane = vec![0f32; w * h];
    let mut cb = vec![0f32; w * h];
    let mut cr = vec![0f32; w * h];
    y_plane
        .par_chunks_mut(w)
        .zip(cb.par_chunks_mut(w))
        .zip(cr.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, ((yrow, cbrow), crrow))| {
            for x in 0..w {
                let i = (y * w + x) * 4;
                let r = buf[i] as f32;
                let g = buf[i + 1] as f32;
                let b = buf[i + 2] as f32;
                let yy = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                yrow[x] = yy;
                cbrow[x] = (b - yy) / 1.8556;
                crrow[x] = (r - yy) / 1.5748;
            }
        });
    let mut tmp = vec![0f32; w * h];
    box_blur_h(&cb, &mut tmp, w, h, radius);
    box_blur_v(&tmp, &mut cb, w, h, radius);
    box_blur_h(&cr, &mut tmp, w, h, radius);
    box_blur_v(&tmp, &mut cr, w, h, radius);
    buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = x * 4;
            let yy = y_plane[y * w + x];
            let cbl = cb[y * w + x];
            let crl = cr[y * w + x];
            let rr = yy + 1.5748 * crl;
            let bb = yy + 1.8556 * cbl;
            let gg = (yy - 0.2126 * rr - 0.0722 * bb) / 0.7152;
            let rgb = [rr, gg, bb];
            for k in 0..3 {
                row[i + k] = rgb[k].clamp(0.0, 255.0).round() as u8;
            }
        }
    });
}

// ───────────────────────────── 3. moire ─────────────────────────────

/// A 3x3 variable low-pass that only touches high-frequency luminance, so
/// regular patterns (screen or fabric moire) are flattened while flat areas
/// and edges outside the pattern are left alone. `pattern` sets how strong the
/// low-pass is.
fn stage_moire(buf: &mut [u8], w: usize, h: usize, adj: &Moire) {
    let amount = (adj.amount / 100.0).clamp(0.0, 1.0);
    if amount <= 0.0 {
        return;
    }
    let pattern = (adj.pattern / 100.0).clamp(0.0, 1.0);
    let weight = amount * (0.25 + 0.75 * pattern);
    let luma = luma_plane(buf, w, h);
    let low = mean3x3(&luma, w, h);
    let mut out = luma.clone();
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = y * w + x;
            let hf = (luma[i] - low[i]).abs();
            let blend = weight * (hf / 10.0).clamp(0.0, 1.0);
            row[x] = luma[i] + (low[i] - luma[i]) * blend;
        }
    });
    apply_luma(buf, w, &luma, &out);
}

// ───────────────────────────── 4. sharpening ─────────────────────────────

/// Unsharp mask: luminance minus a three-pass box blur (a Gaussian stand-in)
/// scaled by `amount`, with an absolute `threshold` below which nothing is
/// boosted. `halo_suppression` pulls the result back inside the local
/// neighbourhood range, which removes the bright and dark rings a strong
/// unsharp mask leaves around edges.
fn stage_sharpening(buf: &mut [u8], w: usize, h: usize, adj: &Sharpening, scale: f32) {
    let amount = adj.amount / 100.0;
    if amount <= 0.0 {
        return;
    }
    let radius = ((adj.radius.max(0.1)) * scale).round().max(1.0) as i32;
    let threshold = adj.threshold.max(0.0);
    let halo = (adj.halo_suppression / 100.0).clamp(0.0, 1.0);
    let gain = amount * 0.75;

    let luma = luma_plane(buf, w, h);
    let blur = box_blur3(&luma, w, h, radius);
    let mut out = luma.clone();
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = y * w + x;
            let mut detail = luma[i] - blur[i];
            if detail.abs() < threshold {
                detail = 0.0;
            }
            let mut v = luma[i] + detail * gain;
            if halo > 0.0 {
                let mut lo = f32::MAX;
                let mut hi = f32::MIN;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let xx = clampi(x as i32 + dx, 0, w as i32 - 1) as usize;
                        let yy = clampi(y as i32 + dy, 0, h as i32 - 1) as usize;
                        let l = luma[yy * w + xx];
                        lo = lo.min(l);
                        hi = hi.max(l);
                    }
                }
                let clamped = v.clamp(lo, hi);
                v = v * (1.0 - halo) + clamped * halo;
            }
            row[x] = v.clamp(0.0, 255.0);
        }
    });
    apply_luma(buf, w, &luma, &out);
}

// ───────────────────────────── 5. lens correction ─────────────────────────────

fn sample_bilinear(src: &[u8], w: usize, h: usize, x: f32, y: f32) -> [f32; 3] {
    let fx = x - 0.5;
    let fy = y - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = fx - x0;
    let ty = fy - y0;
    let x0 = x0 as i32;
    let y0 = y0 as i32;
    let mut acc = [0f32; 3];
    for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
        for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
            let sx = clampi(x0 + dx, 0, w as i32 - 1) as usize;
            let sy = clampi(y0 + dy, 0, h as i32 - 1) as usize;
            let i = (sy * w + sx) * 4;
            for k in 0..3 {
                acc[k] += src[i + k] as f32 * wx * wy;
            }
        }
    }
    acc
}

/// Radial resampling for lens distortion (positive corrects barrel, negative
/// corrects pincushion), a radial gain for light falloff, per-channel radial
/// offsets for chromatic aberration, and a purple-fringe clean-up pass.
fn stage_lens(
    buf: &mut [u8],
    w: usize,
    h: usize,
    adj: &LensCorrection,
    warp: Option<&crate::mock::lens_profile::LensWarp>,
) {
    let distortion = (adj.distortion / 100.0).clamp(-1.0, 1.0);
    let falloff = (adj.light_falloff / 100.0).clamp(0.0, 1.0);
    let purple = (adj.purple_fringing / 100.0).clamp(0.0, 1.0);
    let ca = adj.chromatic_aberration;
    if distortion == 0.0 && falloff == 0.0 && !ca && purple == 0.0 {
        return;
    }
    if adj.use_profile {
        if let Some(warp) = warp {
            stage_lens_profile(buf, w, h, adj, warp, distortion, falloff, ca);
            if purple > 0.0 {
                stage_purple_fringing(buf, w, h, purple);
            }
            return;
        }
    }
    let src = buf.to_vec();
    let cx = w as f32 * 0.5;
    let cy = h as f32 * 0.5;
    let mut out = vec![0u8; w * h * 4];
    out.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = x * 4;
            let dx = (x as f32 + 0.5 - cx) / cx;
            let dy = (y as f32 + 0.5 - cy) / cy;
            let r2 = dx * dx + dy * dy;
            // Radial remap: sample further out for barrel correction.
            let radial = 1.0 + 0.30 * distortion * r2;
            let sx = cx + dx * cx * radial;
            let sy = cy + dy * cy * radial;
            let outside = sx < 0.0 || sy < 0.0 || sx > w as f32 || sy > h as f32;
            if outside && adj.hide_distorted_areas {
                row[i] = 0;
                row[i + 1] = 0;
                row[i + 2] = 0;
                row[i + 3] = 255;
                continue;
            }
            let mut rgb = sample_bilinear(&src, w, h, sx, sy);
            if ca {
                // Red and blue are sampled at slightly different radii.
                let amount = 0.004 * r2;
                let r = sample_bilinear(
                    &src,
                    w,
                    h,
                    cx + (sx - cx) * (1.0 + amount),
                    cy + (sy - cy) * (1.0 + amount),
                );
                let b = sample_bilinear(
                    &src,
                    w,
                    h,
                    cx + (sx - cx) * (1.0 - amount),
                    cy + (sy - cy) * (1.0 - amount),
                );
                rgb[0] = r[0];
                rgb[2] = b[2];
            }
            if falloff > 0.0 {
                let gain = 1.0 + falloff * 0.35 * r2;
                for k in 0..3 {
                    rgb[k] *= gain;
                }
            }
            for k in 0..3 {
                row[i + k] = rgb[k].clamp(0.0, 255.0).round() as u8;
            }
            row[i + 3] = 255;
        }
    });
    buf.copy_from_slice(&out);
    if purple > 0.0 {
        stage_purple_fringing(buf, w, h, purple);
    }
}

/// Capture One's measured tables. Radius is measured on the render buffer's half
/// diagonal, which equals the table's own reference (the half diagonal of the
/// frame it was measured on) for an uncropped frame.
fn stage_lens_profile(
    buf: &mut [u8],
    w: usize,
    h: usize,
    adj: &LensCorrection,
    warp: &crate::mock::lens_profile::LensWarp,
    distortion: f32,
    falloff: f32,
    ca: bool,
) {
    use crate::mock::lens_profile::LensWarp;

    let src = buf.to_vec();
    let cx = w as f32 * 0.5;
    let cy = h as f32 * 0.5;
    let half_diag = ((w * w + h * h) as f32).sqrt() * 0.5;
    let mut out = vec![0u8; w * h * 4];
    out.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = x * 4;
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let r_px = (dx * dx + dy * dy).sqrt();
            let t = r_px / half_diag;
            // The table gives how far the lens moved a point outward, so the
            // source for an output pixel sits at `t - shift`. The sign follows
            // the measured EF 11-24 barrel table and is unverified against C1.
            let t_src = t - LensWarp::sample(&warp.distortion, t) * distortion;
            let k = if r_px > 1e-4 { t_src / t } else { 1.0 };
            let sx = cx + dx * k;
            let sy = cy + dy * k;
            let outside = sx < 0.0 || sy < 0.0 || sx > w as f32 || sy > h as f32;
            if outside && adj.hide_distorted_areas {
                row[i + 3] = 255;
                continue;
            }
            let mut rgb = sample_bilinear(&src, w, h, sx, sy);
            if ca && r_px > 1e-4 {
                let offset = |shift: &[(f32, f32)]| -> f32 {
                    let f = r_px / half_diag + LensWarp::sample(shift, t);
                    f * half_diag / r_px
                };
                rgb[0] = sample_bilinear(&src, w, h, cx + dx * offset(&warp.red), cy + dy * offset(&warp.red))[0];
                rgb[2] = sample_bilinear(&src, w, h, cx + dx * offset(&warp.blue), cy + dy * offset(&warp.blue))[2];
            }
            if falloff > 0.0 {
                // Table EV is a loss (negative toward the corner); undo it.
                let gain = 2f32.powf(-LensWarp::sample(&warp.falloff, t) * falloff);
                for c in &mut rgb {
                    *c *= gain;
                }
            }
            for k2 in 0..3 {
                row[i + k2] = rgb[k2].clamp(0.0, 255.0).round() as u8;
            }
            row[i + 3] = 255;
        }
    });
    buf.copy_from_slice(&out);
}

/// Desaturates strongly purple pixels that sit on an edge, which is what a
/// lateral chromatic fringe looks like. Neutral or non-purple pixels are kept.
fn stage_purple_fringing(buf: &mut [u8], w: usize, h: usize, purple: f32) {
    let src = buf.to_vec();
    buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = x * 4;
            let j = (y * w + x) * 4;
            let p = [src[j], src[j + 1], src[j + 2]];
            let r = p[0] as f32;
            let g = p[1] as f32;
            let b = p[2] as f32;
            let pn = ((r.min(b) - g) / 255.0).clamp(0.0, 1.0);
            if pn <= 0.0 {
                continue;
            }
            let mut lo = 255.0f32;
            let mut hi = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    if dx == 0 && dy == 0 {
                        continue;
                    }
                    let xx = clampi(x as i32 + dx, 0, w as i32 - 1) as usize;
                    let yy = clampi(y as i32 + dy, 0, h as i32 - 1) as usize;
                    let n = (yy * w + xx) * 4;
                    let l = luma8([src[n], src[n + 1], src[n + 2]]);
                    lo = lo.min(l);
                    hi = hi.max(l);
                }
            }
            let edge = ((hi - lo) / 40.0).clamp(0.0, 1.0);
            let amount = purple * pn * edge;
            if amount <= 0.0 {
                continue;
            }
            let gray = luma8(p);
            for k in 0..3 {
                let v = p[k] as f32 * (1.0 - amount) + gray * amount;
                row[i + k] = v.clamp(0.0, 255.0).round() as u8;
            }
        }
    });
}

// ───────────────────────────── 6. film grain ─────────────────────────────

/// Deterministic hash noise (no RNG state, so previews are reproducible).
/// `granularity` picks the size of the noise cell, `kind` shapes the
/// distribution and how it is weighted.
fn hash_noise(x: u32, y: u32, seed: u32) -> f32 {
    let mut h = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77) ^ seed.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2545_F491);
    h ^= h >> 13;
    (h & 0x00FF_FFFF) as f32 / 0x00FF_FFFF as f32 * 2.0 - 1.0
}

fn grain_value(kind: GrainKind, x: u32, y: u32) -> f32 {
    let h = hash_noise(x, y, 0x9E37);
    match kind {
        // Fine, even noise.
        GrainKind::FineGrain => h,
        // Two octaves: slightly coarser and denser than fine.
        GrainKind::SilverRich => (h * 0.7 + hash_noise(x * 2, y * 2, 0x51D3) * 0.3) * 1.1,
        // Averaged hashes: soft, low-contrast speckle.
        GrainKind::SoftGrain => {
            (h + hash_noise(x + 17, y + 31, 0x1234) + hash_noise(x + 91, y + 7, 0x5678)) / 3.0
        }
        // Cubic shaping: most values stay close to zero, few large ones.
        GrainKind::CubicGrain => h * h * h,
        // Quantised into four levels: tabular, blocky grain.
        GrainKind::TabularGrain => ((h * 2.0).round() / 2.0).clamp(-1.0, 1.0),
        // Harsh: high contrast, peaks survive.
        GrainKind::HarshGrain => h.signum() * h.abs().powf(0.4) * 1.25,
    }
}

fn stage_grain(buf: &mut [u8], w: usize, _h: usize, adj: &FilmGrain) {
    let impact = (adj.impact / 100.0).clamp(0.0, 1.0);
    if impact <= 0.0 {
        return;
    }
    let granularity = (adj.granularity / 100.0).clamp(0.0, 1.0);
    let cell = 1 + (granularity * 3.0).round() as u32;
    let amount = impact * 0.25;
    let kind = adj.kind;
    buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = x * 4;
            let mut n = grain_value(kind, (x as u32) / cell, (y as u32) / cell);
            if kind == GrainKind::SilverRich {
                // Silver-rich grain sits mostly in the midtones.
                let p = [row[i], row[i + 1], row[i + 2]];
                let l = luma8(p) / 255.0;
                n *= 1.0 - (l - 0.5).abs() * 1.2;
            }
            let d = (n * amount * 255.0).round();
            for k in 0..3 {
                row[i + k] = (row[i + k] as f32 + d).clamp(0.0, 255.0).round() as u8;
            }
        }
    });
}
// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Fingerprints from the original stages, including clamped edges and alpha.
    #[test]
    fn detail_stages_match_reference_pixels() {
        let expected = [
            15226162907845603633, 15226162907845603633, 15226162907845603633,
            418812772922979691, 4342420695348529323, 1578698336601148913,
            41615323049186328, 10717340786263556290, 2697018795909425383,
            1126012190753934001, 11750616206334758719, 16575147648956657441,
        ];
        let mut actual = Vec::new();
        for (w, h) in [(1, 1), (1, 17), (19, 1), (37, 23)] {
            let base: Vec<u8> = (0..w * h * 4).map(|i| ((i * 73 + i / 7) % 256) as u8).collect();
            for stage in 0..3 {
                let mut pixels = base.clone();
                match stage {
                    0 => stage_luminance_nr(&mut pixels, w, h, &NoiseReduction {
                        luminance: 73.0, details: 41.0, color: 0.0, single_pixel: 87.0,
                    }),
                    1 => stage_color_nr(&mut pixels, w, h, &NoiseReduction {
                        luminance: 0.0, details: 0.0, color: 100.0, single_pixel: 0.0,
                    }),
                    _ => stage_sharpening(&mut pixels, w, h, &Sharpening {
                        amount: 245.0, radius: 2.8, threshold: 1.3, halo_suppression: 37.0,
                    }, 1.5),
                }
                let hash = pixels.iter().fold(0xcbf29ce484222325u64, |hash, &b| {
                    (hash ^ b as u64).wrapping_mul(0x100000001b3)
                });
                actual.push(hash);
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn vertical_blur_preserves_sum_order_at_edges() {
        for (w, h) in [(1, 1), (1, 17), (19, 1), (37, 23)] {
            let src: Vec<f32> = (0..w * h).map(|i| (i as f32 * 1.37).sin() * 255.0).collect();
            for r in [0, 1, 3, 23] {
                let mut out = vec![0.0; src.len()];
                box_blur_v(&src, &mut out, w, h, r);
                for y in 0..h {
                    for x in 0..w {
                        let mut sum = 0.0f32;
                        for k in -r..=r {
                            let yy = (y as i32 + k).clamp(0, h as i32 - 1) as usize;
                            sum += src[yy * w + x];
                        }
                        assert_eq!(out[y * w + x].to_bits(), (sum * (1.0 / (2 * r + 1) as f32)).to_bits());
                    }
                }
            }
        }
    }

    /// Per-stage cost on a Draft-sized frame. `cargo test -p lumen-engine
    /// bench_detail_stages -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing benchmark, run explicitly with --ignored"]
    fn bench_detail_stages() {
        use std::time::Instant;
        let (w, h) = (1400usize, 933usize);
        let base: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                [(x % 251) as u8, (y % 241) as u8, ((x * 7 + y * 13) % 239) as u8, 255]
            })
            .collect();
        let adj = crate::AdjustmentSet::default();
        let mut moire = adj.moire.clone();
        moire.amount = 60.0;
        moire.pattern = 50.0;
        let mut lens = adj.lens.clone();
        lens.distortion = 60.0;
        lens.light_falloff = 60.0;
        lens.chromatic_aberration = true;
        lens.purple_fringing = 40.0;
        lens.use_profile = false;
        let mut grain = adj.film_grain.clone();
        grain.impact = 60.0;
        let pixels = (w * h) as f64;
        let run = |label: &str, f: &dyn Fn(&mut Vec<u8>)| {
            let mut buf = base.clone();
            let t = Instant::now();
            f(&mut buf);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            println!("{label:24} {ms:8.2} ms   {:6.1} Mpx/s", pixels / ms / 1000.0);
        };
        run("luminance_nr", &|b| stage_luminance_nr(b, w, h, &adj.noise_reduction));
        run("color_nr", &|b| stage_color_nr(b, w, h, &adj.noise_reduction));
        run("moire", &|b| stage_moire(b, w, h, &moire));
        run("sharpening", &|b| stage_sharpening(b, w, h, &adj.sharpening, 1.0));
        run("lens(generic)", &|b| stage_lens(b, w, h, &lens, None));
        run("grain", &|b| stage_grain(b, w, h, &grain));
        run("post_process(all on)", &|b| post_process_with_lens(b, w as u32, h as u32, &adj, 1.0, None));
    }

    /// An adjustment set with every detail stage switched off.
    fn off() -> crate::AdjustmentSet {
        let mut adj = crate::AdjustmentSet::default();
        adj.noise_reduction = NoiseReduction {
            luminance: 0.0,
            details: 0.0,
            color: 0.0,
            single_pixel: 0.0,
        };
        adj.sharpening = Sharpening {
            amount: 0.0,
            radius: 0.8,
            threshold: 1.0,
            halo_suppression: 0.0,
        };
        adj.moire = Moire { amount: 0.0, pattern: 0.0 };
        adj.film_grain = FilmGrain { kind: GrainKind::FineGrain, impact: 0.0, granularity: 50.0 };
        adj.lens = LensCorrection {
            profile: None,
            use_profile: false,
            distortion: 0.0,
            sharpness_falloff: 0.0,
            light_falloff: 0.0,
            chromatic_aberration: false,
            purple_fringing: 0.0,
            hide_distorted_areas: false,
            shift_x: 0.0,
            shift_y: 0.0,
        };
        adj
    }

    fn variance(px: &[u8]) -> f32 {
        let n = px.len() as f32;
        let mean = px.iter().map(|v| *v as f32).sum::<f32>() / n;
        px.iter().map(|v| (*v as f32 - mean).powi(2)).sum::<f32>() / n
    }

    /// The luma-plane reuse in NR/sharpening must not change a single byte:
    /// the stage output equals a per-pixel recompute of the same maths.
    #[test]
    fn luma_plane_reuse_matches_the_naive_recompute() {
        let (w, h) = (37usize, 23usize);
        let buf: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                [((x * 13 + y * 7) % 256) as u8, ((x * 5) % 256) as u8, ((y * 11) % 256) as u8, 255]
            })
            .collect();
        let plane = luma_plane(&buf, w, h);
        for (i, px) in buf.chunks_exact(4).enumerate() {
            assert_eq!(plane[i], luma8([px[0], px[1], px[2]]), "luma plane differs at {i}");
        }
        let target: Vec<f32> = plane.iter().map(|l| (*l * 0.8 + 20.0).clamp(0.0, 255.0)).collect();
        let mut a = buf.clone();
        let mut b = buf.clone();
        apply_luma(&mut a, w, &plane, &target);
        for (y, row) in b.chunks_exact_mut(w * 4).enumerate() {
            for x in 0..w {
                let i = x * 4;
                let p = [row[i], row[i + 1], row[i + 2]];
                let c = luma8(p);
                let new = target[y * w + x];
                if (new - c).abs() < 0.05 {
                    continue;
                }
                let f = if c > 1.0 { new / c } else { 1.0 };
                for k in 0..3 {
                    row[i + k] = (p[k] as f32 * f).clamp(0.0, 255.0).round() as u8;
                }
            }
        }
        assert_eq!(a, b, "apply_luma must match the per-pixel recompute");
    }

    fn luma_variance(buf: &[u8]) -> f32 {
        let l: Vec<u8> = buf.chunks_exact(4).map(|p| luma8([p[0], p[1], p[2]]).round() as u8).collect();
        variance(&l)
    }

    fn checkerboard(size: usize) -> Vec<u8> {
        let mut buf = vec![0u8; size * size * 4];
        for y in 0..size {
            for x in 0..size {
                let v = if (x + y) % 2 == 0 { 230 } else { 26 };
                let i = (y * size + x) * 4;
                buf[i] = v;
                buf[i + 1] = v;
                buf[i + 2] = v;
                buf[i + 3] = 255;
            }
        }
        buf
    }

    fn flat(size: usize, v: u8) -> Vec<u8> {
        let mut buf = vec![v; size * size * 4];
        for i in (0..buf.len()).step_by(4) {
            buf[i + 3] = 255;
        }
        buf
    }

    fn step_edge(size: usize) -> Vec<u8> {
        let mut buf = flat(size, 40);
        for y in 0..size {
            for x in (size / 2)..size {
                let i = (y * size + x) * 4;
                buf[i] = 200;
                buf[i + 1] = 200;
                buf[i + 2] = 200;
            }
        }
        buf
    }

    #[test]
    fn all_off_leaves_bytes_identical() {
        let mut buf = checkerboard(8);
        let before = buf.clone();
        post_process(&mut buf, 8, 8, &off(), 1.0);
        assert_eq!(before, buf);
    }

    #[test]
    fn bad_input_is_safe() {
        let mut buf = vec![0u8; 10];
        post_process(&mut buf, 8, 8, &off(), 1.0);
        post_process(&mut buf, 0, 0, &off(), 1.0);
        let mut one = vec![7u8; 4];
        let mut adj = off();
        adj.sharpening.amount = 100.0;
        post_process(&mut one, 1, 1, &adj, 1.0);
        assert_eq!(one, vec![7u8; 4]);
        let mut empty: Vec<u8> = Vec::new();
        post_process(&mut empty, 4, 4, &adj, 1.0);
    }

    #[test]
    fn luminance_nr_reduces_variance() {
        let mut buf = vec![0u8; 16 * 16 * 4];
        for i in 0..buf.len() / 4 {
            let v = if i % 3 == 0 { 60 } else { 120 };
            buf[i * 4] = v;
            buf[i * 4 + 1] = v;
            buf[i * 4 + 2] = v;
            buf[i * 4 + 3] = 255;
        }
        let before = luma_variance(&buf);
        let mut adj = off();
        adj.noise_reduction.luminance = 80.0;
        adj.noise_reduction.details = 0.0;
        post_process(&mut buf, 16, 16, &adj, 1.0);
        assert!(luma_variance(&buf) < before, "NR should reduce variance");
    }

    #[test]
    fn single_pixel_removes_hot_pixel() {
        let mut buf = flat(16, 100);
        let hot = (8 * 16 + 8) * 4;
        buf[hot] = 250;
        buf[hot + 1] = 250;
        buf[hot + 2] = 250;
        let mut adj = off();
        adj.noise_reduction.single_pixel = 100.0;
        post_process(&mut buf, 16, 16, &adj, 1.0);
        assert!(buf[hot] < 130, "hot pixel should be pulled back: {}", buf[hot]);
    }

    #[test]
    fn color_nr_reduces_chroma_variance_only() {
        let mut buf = vec![0u8; 16 * 16 * 4];
        for i in 0..buf.len() / 4 {
            buf[i * 4] = 128 + ((i * 37) % 21) as u8;
            buf[i * 4 + 1] = 128 + ((i * 53) % 19) as u8;
            buf[i * 4 + 2] = 128 + ((i * 71) % 23) as u8;
            buf[i * 4 + 3] = 255;
        }
        let chroma = |b: &[u8]| -> f32 {
            b.chunks_exact(4)
                .map(|p| (p[0] as f32 - p[1] as f32).abs() + (p[2] as f32 - p[1] as f32).abs())
                .sum::<f32>()
                / (b.len() / 4) as f32
        };
        let chroma_before = chroma(&buf);
        let luma_before = luma_variance(&buf);
        let mut adj = off();
        adj.noise_reduction.color = 100.0;
        post_process(&mut buf, 16, 16, &adj, 1.0);
        assert!(chroma(&buf) < chroma_before, "colour noise should drop");
        assert!(
            (luma_variance(&buf) - luma_before).abs() < 2.0,
            "luminance should be preserved"
        );
    }

    #[test]
    fn moire_flattens_checkerboard() {
        let mut buf = checkerboard(16);
        let before = luma_variance(&buf);
        let mut adj = off();
        adj.moire.amount = 100.0;
        adj.moire.pattern = 100.0;
        post_process(&mut buf, 16, 16, &adj, 1.0);
        assert!(luma_variance(&buf) < before, "moire should flatten the pattern");
    }

    #[test]
    fn sharpening_increases_edge_variance() {
        let mut buf = step_edge(16);
        let before = luma_variance(&buf);
        let mut adj = off();
        adj.sharpening.amount = 200.0;
        adj.sharpening.radius = 1.0;
        adj.sharpening.threshold = 0.0;
        post_process(&mut buf, 16, 16, &adj, 1.0);
        assert!(luma_variance(&buf) > before, "sharpening should raise contrast");
    }

    #[test]
    fn halo_suppression_limits_overshoot() {
        let make = |halo: f32| {
            let mut buf = step_edge(16);
            let mut adj = off();
            adj.sharpening.amount = 200.0;
            adj.sharpening.radius = 1.0;
            adj.sharpening.threshold = 0.0;
            adj.sharpening.halo_suppression = halo;
            post_process(&mut buf, 16, 16, &adj, 1.0);
            buf
        };
        let plain = make(0.0);
        let guarded = make(100.0);
        let dark = 7 * 4;
        let bright = 8 * 4;
        assert!(guarded[dark] >= plain[dark], "dark halo must be pulled back up");
        assert!(guarded[dark] <= 200, "no overshoot past the bright side");
        assert!(guarded[bright] <= plain[bright], "bright halo must be pulled back down");
        assert!(guarded[bright] >= 40, "no undershoot past the dark side");
    }

    #[test]
    fn lens_distortion_moves_pixels_and_can_hide_areas() {
        let base = checkerboard(16);
        let mut buf = base.clone();
        let mut adj = off();
        adj.lens.distortion = 100.0;
        adj.lens.hide_distorted_areas = true;
        post_process(&mut buf, 16, 16, &adj, 1.0);
        assert_ne!(base, buf, "distortion must change the frame");
        assert!(
            buf.chunks_exact(4).any(|p| p[0] == 0 && p[1] == 0 && p[2] == 0),
            "corners should be hidden"
        );
    }

    #[test]
    fn light_falloff_brightens_corners() {
        let mut buf = flat(16, 80);
        let mut adj = off();
        adj.lens.light_falloff = 100.0;
        post_process(&mut buf, 16, 16, &adj, 1.0);
        let corner = 0;
        let centre = (8 * 16 + 8) * 4;
        assert!(buf[corner] > buf[centre], "falloff correction brightens the edge");
    }

    #[test]
    fn grain_is_deterministic_and_adds_variance() {
        let mut a = flat(16, 120);
        let mut b = a.clone();
        let mut adj = off();
        adj.film_grain.impact = 100.0;
        adj.film_grain.granularity = 50.0;
        post_process(&mut a, 16, 16, &adj, 1.0);
        post_process(&mut b, 16, 16, &adj, 1.0);
        assert_eq!(a, b, "grain must be deterministic");
        assert!(luma_variance(&a) > 0.0, "grain must add noise");
    }

    fn warp(distortion: &[(f32, f32)], falloff: &[(f32, f32)]) -> crate::mock::lens_profile::LensWarp {
        crate::mock::lens_profile::LensWarp {
            distortion: distortion.to_vec(),
            falloff: falloff.to_vec(),
            red: Vec::new(),
            blue: Vec::new(),
        }
    }

    /// A white ring at `t` of the half diagonal on a black frame.
    fn ring(size: usize, t: f32) -> Vec<u8> {
        let mut buf = vec![0u8; size * size * 4];
        let c = size as f32 * 0.5;
        let half_diag = ((size * size * 2) as f32).sqrt() * 0.5;
        for y in 0..size {
            for x in 0..size {
                let d = ((x as f32 + 0.5 - c).powi(2) + (y as f32 + 0.5 - c).powi(2)).sqrt();
                if (d / half_diag - t).abs() < 1.5 / half_diag {
                    let i = (y * size + x) * 4;
                    buf[i] = 240;
                    buf[i + 1] = 240;
                    buf[i + 2] = 240;
                }
                buf[(y * size + x) * 4 + 3] = 255;
            }
        }
        buf
    }

    /// Brightest `t` on the horizontal centre line.
    fn brightest_t(buf: &[u8], size: usize) -> f32 {
        let half_diag = ((size * size * 2) as f32).sqrt() * 0.5;
        let y = size / 2;
        let (mut best, mut best_t) = (0u8, 0.0);
        for x in size / 2..size {
            let v = buf[(y * size + x) * 4];
            if v > best {
                best = v;
                best_t = (x as f32 + 0.5 - size as f32 * 0.5) / half_diag;
            }
        }
        best_t
    }

    #[test]
    fn lens_profile_stays_off_until_enabled() {
        let mut a = ring(48, 0.7);
        let mut b = a.clone();
        let mut adj = off();
        adj.lens.distortion = 100.0;
        adj.lens.light_falloff = 100.0;
        post_process_with_lens(&mut a, 48, 48, &adj, 1.0, Some(&warp(&[(0.0, 0.0), (1.0, 0.05)], &[(0.0, 0.0), (1.0, -1.0)])));
        post_process(&mut b, 48, 48, &adj, 1.0);
        assert_eq!(a, b, "the profile must be inert while use_profile is false");
    }

    #[test]
    fn lens_profile_distortion_moves_the_ring_outward() {
        let mut buf = ring(64, 0.5);
        let before = brightest_t(&buf, 64);
        let mut adj = off();
        adj.lens.use_profile = true;
        adj.lens.distortion = 100.0;
        post_process_with_lens(&mut buf, 64, 64, &adj, 1.0, Some(&warp(&[(0.0, 0.0), (1.0, 0.05)], &[])));
        let after = brightest_t(&buf, 64);
        assert!((after - before - 0.05).abs() < 0.01, "ring moved {before} -> {after}, expected +0.05");
    }

    #[test]
    fn lens_profile_normalization_is_sensor_independent() {
        let mut small = ring(32, 0.5);
        let mut large = ring(64, 0.5);
        let w = warp(&[(0.0, 0.0), (1.0, 0.05)], &[]);
        let mut adj = off();
        adj.lens.use_profile = true;
        adj.lens.distortion = 100.0;
        post_process_with_lens(&mut small, 32, 32, &adj, 1.0, Some(&w));
        post_process_with_lens(&mut large, 64, 64, &adj, 1.0, Some(&w));
        let (a, b) = (brightest_t(&small, 32), brightest_t(&large, 64));
        assert!((a - b).abs() < 0.02, "same profile gave {a} and {b} at different sensor sizes");
    }

    #[test]
    fn lens_profile_falloff_applies_two_to_the_ev() {
        let mut buf = flat(16, 80);
        let mut adj = off();
        adj.lens.use_profile = true;
        adj.lens.light_falloff = 100.0;
        let w = warp(&[], &[(0.0, 0.0), (1.0, -1.5)]);
        let half_diag = (16f32 * 16.0 * 2.0).sqrt() * 0.5;
        let gain = |t: f32| 2f32.powf(-crate::mock::lens_profile::LensWarp::sample(&w.falloff, t));
        let corner_t = (7.5f32 * 7.5 * 2.0).sqrt() / half_diag;
        let centre_t = (0.5f32 * 0.5 * 2.0).sqrt() / half_diag;
        let expected = gain(corner_t) / gain(centre_t);
        post_process_with_lens(&mut buf, 16, 16, &adj, 1.0, Some(&w));
        let corner = buf[0] as f32;
        let centre = buf[(8 * 16 + 8) * 4] as f32;
        assert!((corner / centre - expected).abs() < 0.05, "corner/centre was {}, expected {expected}", corner / centre);
    }

    #[test]
    fn lens_profile_clamps_past_the_table_end() {
        let mut a = ring(32, 0.9);
        let mut b = a.clone();
        let mut adj = off();
        adj.lens.use_profile = true;
        adj.lens.distortion = 100.0;
        // Table stops at half the radius; everything beyond uses its last value.
        post_process_with_lens(&mut a, 32, 32, &adj, 1.0, Some(&warp(&[(0.0, 0.0), (0.5, 0.04)], &[])));
        post_process_with_lens(&mut b, 32, 32, &adj, 1.0, Some(&warp(&[(0.0, 0.0), (0.5, 0.04), (1.0, 0.04)], &[])));
        assert_eq!(a, b);
    }
}
