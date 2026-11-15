//! Secondary chromatic-aberration cleanup on sRGB-encoded 8-bit RGBA buffers.
//!
//! Port of Capture One's lens-tool trio: colour defringing on high-contrast
//! edges (magenta and green), purple-fringe suppression, and single-pixel
//! anti-colour-aliasing.  Every correction is chroma-only: luminance is
//! reconstructed exactly, so edge sharpness is untouched.

use rayon::prelude::*;

/// Below this opponent-space chroma magnitude a pixel counts as neutral.
const NEUTRAL: f32 = 0.02;

#[inline]
fn luma(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Rebuild RGB from luminance plus opponent chroma `(R-Y, B-Y)`.
#[inline]
fn with_chroma(y: f32, cr: f32, cb: f32) -> [u8; 3] {
    let r = y + cr;
    let b = y + cb;
    let g = y - (0.2126 * cr + 0.0722 * cb) / 0.7152;
    [
        (r * 255.0).round().clamp(0.0, 255.0) as u8,
        (g * 255.0).round().clamp(0.0, 255.0) as u8,
        (b * 255.0).round().clamp(0.0, 255.0) as u8,
    ]
}

fn dims(rgba: &[u8], w: u32, h: u32) -> Option<(usize, usize)> {
    let (w, h) = (w as usize, h as usize);
    if w == 0 || h == 0 || rgba.len() < w * h * 4 {
        return None;
    }
    Some((w, h))
}

#[inline]
fn lum_at(ymap: &[f32], w: usize, h: usize, x: usize, y: usize) -> f32 {
    let (x, y) = (x.min(w - 1), y.min(h - 1));
    ymap[y * w + x]
}

/// Remove colour fringes on strong edges: magenta/violet and green pixels whose
/// local luma gradient is high are pushed toward neutral, scaled by `amount`
/// (`0.0..=1.0`).  Normal colours and flat areas are left alone.
pub fn defringe(rgba: &mut [u8], w: u32, h: u32, amount: f32) {
    let amount = amount.clamp(0.0, 1.0);
    let Some((w, h)) = dims(rgba, w, h) else {
        return;
    };
    if amount == 0.0 || w < 3 || h < 3 {
        return;
    }

    let mut ymap = vec![0f32; w * h];
    ymap.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let base = y * w * 4;
        for (x, out) in row.iter_mut().enumerate() {
            let p = base + x * 4;
            *out = luma(
                rgba[p] as f32 / 255.0,
                rgba[p + 1] as f32 / 255.0,
                rgba[p + 2] as f32 / 255.0,
            );
        }
    });

    rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let gx = lum_at(&ymap, w, h, x + 1, y) - lum_at(&ymap, w, h, x.saturating_sub(1), y);
            let gy = lum_at(&ymap, w, h, x, y + 1) - lum_at(&ymap, w, h, x, y.saturating_sub(1));
            let edge = (((gx * gx + gy * gy).sqrt() - 0.035) / 0.09).clamp(0.0, 1.0);
            if edge <= 0.0 {
                continue;
            }
            let p = x * 4;
            let r = row[p] as f32 / 255.0;
            let g = row[p + 1] as f32 / 255.0;
            let b = row[p + 2] as f32 / 255.0;
            let yv = luma(r, g, b);
            let (cr, cb) = (r - yv, b - yv);
            // Magenta (+,+) and green (-,-) sit in opposite quadrants of the
            // opponent plane; same-sign chroma is exactly those two hues.
            if cr * cb <= 0.0 {
                continue;
            }
            let sat = cr.abs().max(cb.abs());
            let mask = ((sat - NEUTRAL) / 0.10).clamp(0.0, 1.0);
            let k = amount * edge * mask;
            if k <= 0.0 {
                continue;
            }
            let rgb = with_chroma(yv, cr * (1.0 - k), cb * (1.0 - k));
            row[p..p + 3].copy_from_slice(&rgb);
        }
    });
}

/// Purple-fringe suppression: the magenta half of [`defringe`], with a wider
/// hue gate and a lower edge threshold so faint halos around specular edges go.
pub fn defringe_purple(rgba: &mut [u8], w: u32, h: u32, amount: f32) {
    let amount = amount.clamp(0.0, 1.0);
    let Some((w, h)) = dims(rgba, w, h) else {
        return;
    };
    if amount == 0.0 || w < 3 || h < 3 {
        return;
    }
    let mut ymap = vec![0f32; w * h];
    ymap.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let base = y * w * 4;
        for (x, out) in row.iter_mut().enumerate() {
            let p = base + x * 4;
            *out = luma(
                rgba[p] as f32 / 255.0,
                rgba[p + 1] as f32 / 255.0,
                rgba[p + 2] as f32 / 255.0,
            );
        }
    });
    rgba.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let gx = lum_at(&ymap, w, h, x + 1, y) - lum_at(&ymap, w, h, x.saturating_sub(1), y);
            let gy = lum_at(&ymap, w, h, x, y + 1) - lum_at(&ymap, w, h, x, y.saturating_sub(1));
            let edge = (((gx * gx + gy * gy).sqrt() - 0.02) / 0.08).clamp(0.0, 1.0);
            if edge <= 0.0 {
                continue;
            }
            let p = x * 4;
            let r = row[p] as f32 / 255.0;
            let g = row[p + 1] as f32 / 255.0;
            let b = row[p + 2] as f32 / 255.0;
            let yv = luma(r, g, b);
            let (cr, cb) = (r - yv, b - yv);
            if cr <= 0.0 || cb <= 0.0 {
                continue;
            }
            let sat = cr.min(cb).max(0.0);
            let mask = ((sat - NEUTRAL) / 0.08).clamp(0.0, 1.0);
            let k = amount * edge * mask;
            if k <= 0.0 {
                continue;
            }
            let rgb = with_chroma(yv, cr * (1.0 - k), cb * (1.0 - k));
            row[p..p + 3].copy_from_slice(&rgb);
        }
    });
}

/// Anti colour aliasing: a single pixel whose chroma disagrees with the median
/// of its four neighbours while the luma shows a one-pixel detail is false
/// colour from demosaicing.  Its chroma is blended toward the neighbour median.
pub fn anti_alias(rgba: &mut [u8], w: u32, h: u32, amount: f32) {
    let amount = amount.clamp(0.0, 1.0);
    let Some((w, h)) = dims(rgba, w, h) else {
        return;
    };
    if amount == 0.0 || w < 3 || h < 3 {
        return;
    }

    fn read_row(rgba: &[u8], w: usize, y: usize) -> Vec<(f32, f32, f32)> {
        let base = y * w * 4;
        (0..w)
            .map(|x| {
                let p = base + x * 4;
                let (r, g, b) = (
                    rgba[p] as f32 / 255.0,
                    rgba[p + 1] as f32 / 255.0,
                    rgba[p + 2] as f32 / 255.0,
                );
                let yv = luma(r, g, b);
                (yv, r - yv, b - yv)
            })
            .collect()
    }

    let mut prev = read_row(rgba, w, 0);
    let mut cur = read_row(rgba, w, 1);
    for y in 1..h - 1 {
        let next = read_row(rgba, w, y + 1);
        let base = y * w * 4;
        for x in 1..w - 1 {
            let (yv, cr, cb) = cur[x];
            let n = [prev[x], next[x], cur[x - 1], cur[x + 1]];
            let med_cr = median4(n.map(|p| p.1));
            let med_cb = median4(n.map(|p| p.2));
            let dev = ((cr - med_cr).powi(2) + (cb - med_cb).powi(2)).sqrt();
            if dev <= 0.03 {
                continue;
            }
            let med_y = n.iter().map(|p| p.0).sum::<f32>() * 0.25;
            let detail = (yv - med_y).abs();
            let k = amount
                * ((dev - 0.03) / 0.10).clamp(0.0, 1.0)
                * ((detail - 0.015) / 0.04).clamp(0.0, 1.0);
            if k <= 0.0 {
                continue;
            }
            let p = base + x * 4;
            let rgb = with_chroma(yv, cr + (med_cr - cr) * k, cb + (med_cb - cb) * k);
            rgba[p..p + 3].copy_from_slice(&rgb);
        }
        prev = cur;
        cur = next;
    }
}

/// Median of four values: mean of the two middle ones.
#[inline]
fn median4(mut v: [f32; 4]) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    (v[1] + v[2]) * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;

    fn canvas(w: u32, h: u32, rgb: [u8; 3]) -> Vec<u8> {
        let mut v = vec![0u8; (w * h * 4) as usize];
        for p in v.chunks_exact_mut(4) {
            p[..3].copy_from_slice(&rgb);
            p[3] = 255;
        }
        v
    }

    fn put(v: &mut [u8], w: u32, x: u32, y: u32, rgb: [u8; 3]) {
        let p = ((y * w + x) * 4) as usize;
        v[p..p + 3].copy_from_slice(&rgb);
    }

    /// Black field, white square, one 2 px purple halo outside the white edge.
    fn purple_frame(w: u32, h: u32) -> Vec<u8> {
        let mut v = canvas(w, h, [0, 0, 0]);
        let (a, b) = (w / 4, 3 * w / 4);
        for y in a..b {
            for x in a..b {
                let ring = x < a + 2 || x >= b - 2 || y < a + 2 || y >= b - 2;
                let rgb = if ring { [120, 30, 210] } else { [255, 255, 255] };
                put(&mut v, w, x, y, rgb);
            }
        }
        v
    }

    fn green_frame(w: u32, h: u32) -> Vec<u8> {
        let mut v = canvas(w, h, [0, 0, 0]);
        let (a, b) = (w / 4, 3 * w / 4);
        for y in a..b {
            for x in a..b {
                let ring = x < a + 2 || x >= b - 2 || y < a + 2 || y >= b - 2;
                let rgb = if ring { [20, 200, 40] } else { [255, 255, 255] };
                put(&mut v, w, x, y, rgb);
            }
        }
        v
    }

    fn count(v: &[u8], f: impl Fn([u8; 3]) -> bool) -> usize {
        v.chunks_exact(4)
            .filter(|p| f([p[0], p[1], p[2]]))
            .count()
    }

    fn is_purple(p: [u8; 3]) -> bool {
        p[0] as i32 > p[1] as i32 + 20 && p[2] as i32 > p[1] as i32 + 20
    }

    fn is_green(p: [u8; 3]) -> bool {
        p[1] as i32 > p[0] as i32 + 20 && p[1] as i32 > p[2] as i32 + 20
    }

    /// Sum of squared luma gradients, a proxy for edge energy.
    fn grad_energy(v: &[u8], w: u32, h: u32) -> f64 {
        let (w, h) = (w as usize, h as usize);
        let l = |x: usize, y: usize| {
            let p = (y * w + x) * 4;
            luma(v[p] as f32, v[p + 1] as f32, v[p + 2] as f32)
        };
        let mut sum = 0f64;
        for y in 1..h - 1 {
            for x in 1..w - 1 {
                let gx = (l(x + 1, y) - l(x - 1, y)) as f64;
                let gy = (l(x, y + 1) - l(x, y - 1)) as f64;
                sum += gx * gx + gy * gy;
            }
        }
        sum
    }

    #[test]
    fn identity_amount_zero_is_noop() {
        let mut v = purple_frame(64, 64);
        let orig = v.clone();
        defringe(&mut v, 64, 64, 0.0);
        defringe_purple(&mut v, 64, 64, 0.0);
        anti_alias(&mut v, 64, 64, 0.0);
        assert_eq!(v, orig);
    }

    #[test]
    fn purple_fringe_is_suppressed() {
        let (w, h) = (64u32, 64u32);
        let mut v = purple_frame(w, h);
        let before = count(&v, is_purple);
        defringe(&mut v, w, h, 1.0);
        let after = count(&v, is_purple);
        assert!(before > 100, "fixture has purple pixels: {before}");
        assert!(after * 2 < before, "purple {before} -> {after}");
    }

    #[test]
    fn purple_only_pass_also_works() {
        let (w, h) = (64u32, 64u32);
        let mut v = purple_frame(w, h);
        let before = count(&v, is_purple);
        defringe_purple(&mut v, w, h, 1.0);
        let after = count(&v, is_purple);
        assert!(after * 2 < before, "purple {before} -> {after}");
    }

    #[test]
    fn green_fringe_is_suppressed() {
        let (w, h) = (64u32, 64u32);
        let mut v = green_frame(w, h);
        let before = count(&v, is_green);
        defringe(&mut v, w, h, 1.0);
        let after = count(&v, is_green);
        assert!(before > 100, "fixture has green pixels: {before}");
        assert!(after * 2 < before, "green {before} -> {after}");
    }

    #[test]
    fn normal_colours_are_untouched() {
        let (w, h) = (64u32, 40u32);
        let mut v = canvas(w, h, [0, 0, 0]);
        for y in 0..h {
            for x in 0..w {
                let t = x as f32 / (w - 1) as f32;
                put(
                    &mut v,
                    w,
                    x,
                    y,
                    [
                        (110.0 + 90.0 * t) as u8,
                        (70.0 + 40.0 * t) as u8,
                        (40.0 + 30.0 * t) as u8,
                    ],
                );
            }
        }
        let orig = v.clone();
        defringe(&mut v, w, h, 1.0);
        defringe_purple(&mut v, w, h, 1.0);
        anti_alias(&mut v, w, h, 1.0);
        let max_diff = orig
            .iter()
            .zip(&v)
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        assert!(max_diff <= 1, "max channel change {max_diff}");
    }

    #[test]
    fn edge_sharpness_is_preserved() {
        let (w, h) = (64u32, 64u32);
        let mut v = purple_frame(w, h);
        let before = grad_energy(&v, w, h);
        defringe(&mut v, w, h, 1.0);
        let after = grad_energy(&v, w, h);
        assert!(
            after >= before * 0.9,
            "gradient energy {before:.1} -> {after:.1}"
        );
    }

    #[test]
    fn empty_and_tiny_images_are_safe() {
        let mut empty: Vec<u8> = Vec::new();
        defringe(&mut empty, 0, 0, 1.0);
        defringe_purple(&mut empty, 8, 8, 1.0);
        anti_alias(&mut empty, 8, 8, 1.0);

        let mut one = canvas(1, 1, [200, 10, 200]);
        let orig = one.clone();
        defringe(&mut one, 1, 1, 1.0);
        anti_alias(&mut one, 1, 1, 1.0);
        assert_eq!(one, orig);
    }

    #[test]
    fn anti_alias_kills_single_pixel_false_colour() {
        let (w, h) = (32u32, 32u32);
        let mut v = canvas(w, h, [128, 128, 128]);
        put(&mut v, w, 16, 16, [210, 40, 210]);
        let before = v[((16 * w + 16) * 4) as usize] as i32;
        anti_alias(&mut v, w, h, 1.0);
        let p = ((16 * w + 16) * 4) as usize;
        let after = v[p] as i32;
        assert!(before - after > 40, "false colour {before} -> {after}");
        let neighbour = ((16 * w + 15) * 4) as usize;
        assert_eq!(&v[neighbour..neighbour + 3], &[128, 128, 128]);
    }

    #[test]
    fn report_synthetic_metrics() {
        let (w, h) = (512u32, 512u32);
        let mut v = purple_frame(w, h);
        let ratio = |v: &[u8]| count(v, is_purple) as f64 / (w * h) as f64 * 100.0;
        let before_ratio = ratio(&v);
        let before_grad = grad_energy(&v, w, h);
        let t0 = std::time::Instant::now();
        defringe(&mut v, w, h, 1.0);
        defringe_purple(&mut v, w, h, 1.0);
        let elapsed = t0.elapsed();
        let after_ratio = ratio(&v);
        let after_grad = grad_energy(&v, w, h);
        println!(
            "512x512 purple share {before_ratio:.3}% -> {after_ratio:.3}%, \
             gradient energy {before_grad:.1} -> {after_grad:.1} ({:+.2}%), \
             defringe+purple {elapsed:?}",
            (after_grad / before_grad - 1.0) * 100.0
        );
        let mut dot = canvas(1920, 1080, [120, 120, 120]);
        for i in 0..200 {
            put(&mut dot, 1920, 100 + i * 8, 540, [200, 40, 200]);
        }
        let t0 = std::time::Instant::now();
        anti_alias(&mut dot, 1920, 1080, 1.0);
        println!("1920x1080 anti_alias {:?}", t0.elapsed());
        assert!(after_ratio < before_ratio * 0.5);
    }
}
