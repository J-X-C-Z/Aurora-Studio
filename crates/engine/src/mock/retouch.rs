// 2D approximation of portrait retouching: no face landmarks, no semantic masks.
// Every effect is a pure function of the rendered sRGB RGBA8 frame plus a
// caller-supplied region, so a region only lands where the caller put it.
// Limits: skin/teeth detection is per-pixel colour heuristics, so a region over
// hair or a hand is smoothed like skin and nothing outside the region is touched.

use rayon::prelude::*;

/// Normalised `x, y, w, h` box in frame coordinates plus a soft-edge width,
/// also normalised (fraction of the frame's short side is not assumed).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Region {
    pub rect: [f32; 4],
    pub feather: f32,
}

const SKIN_SIGMA_RANGE: f32 = 20.0;
const BILATERAL_RADIUS: i64 = 2;
const BOX_PASSES: usize = 3;

/// Smoothstep coverage of `rect`: 1 well inside, 0 well outside, a soft ramp of
/// half-width `feather` centred on the border. Non-finite input gives 0.
pub(crate) fn mask_weight(rect: [f32; 4], feather: f32, x: f32, y: f32) -> f32 {
    if !x.is_finite() || !y.is_finite() || !feather.is_finite() {
        return 0.0;
    }
    let [rx, ry, rw, rh] = rect;
    if !rx.is_finite() || !ry.is_finite() || !rw.is_finite() || !rh.is_finite() {
        return 0.0;
    }
    if rw <= 0.0 || rh <= 0.0 {
        return 0.0;
    }
    // Signed distance to the box, positive inside.
    let d = (x - rx).min(rx + rw - x).min(y - ry).min(ry + rh - y);
    let f = feather.abs().max(0.0);
    smoothstep(-f, f, d)
}

/// Edge-preserving skin smoothing: blend the pixel toward a 5x5 bilateral mean
/// by `strength * mask * skin`, keeping `texture` of the original high frequency.
pub(crate) fn skin_smooth(
    rgba: &mut [u8],
    w: u32,
    h: u32,
    region: Region,
    strength: f32,
    texture: f32,
) {
    if !valid(rgba, w, h) || !strength.is_finite() || !texture.is_finite() {
        return;
    }
    let strength = strength.clamp(0.0, 1.0);
    if strength <= 0.0 {
        return;
    }
    let texture = texture.clamp(0.0, 1.0);
    let (wu, hu) = (w as usize, h as usize);
    let src = rgba.to_vec();
    let spatial = bilateral_weights();
    let denom = 2.0 * SKIN_SIGMA_RANGE * SKIN_SIGMA_RANGE;

    rgba.par_chunks_mut(wu * 4).enumerate().for_each(|(y, line)| {
        for x in 0..wu {
            let nx = (x as f32 + 0.5) / w as f32;
            let ny = (y as f32 + 0.5) / h as f32;
            let m = mask_weight(region.rect, region.feather, nx, ny);
            if m <= 0.0 {
                continue;
            }
            let i = (y * wu + x) * 4;
            let c = [src[i] as f32, src[i + 1] as f32, src[i + 2] as f32];
            let k = strength * m * skin_weight(c) * (1.0 - texture);
            if k <= 0.0 {
                continue;
            }
            let mut acc = [0f32; 3];
            let mut wsum = 0f32;
            for &(dx, dy, ws) in &spatial {
                let sx = (x as i64 + dx).clamp(0, wu as i64 - 1) as usize;
                let sy = (y as i64 + dy).clamp(0, hu as i64 - 1) as usize;
                let j = (sy * wu + sx) * 4;
                let n = [src[j] as f32, src[j + 1] as f32, src[j + 2] as f32];
                let d2 = ((n[0] - c[0]).powi(2) + (n[1] - c[1]).powi(2) + (n[2] - c[2]).powi(2)) / 3.0;
                let weight = ws * (-d2 / denom).exp();
                for ch in 0..3 {
                    acc[ch] += n[ch] * weight;
                }
                wsum += weight;
            }
            if wsum <= 0.0 {
                continue;
            }
            for ch in 0..3 {
                let mean = acc[ch] / wsum;
                line[x * 4 + ch] = (c[ch] + (mean - c[ch]) * k).clamp(0.0, 255.0) as u8;
            }
        }
    });
}

/// Brighten a region in HSV: lift V with a soft shoulder so highlights approach
/// but never clip, and scale S by `1 + saturation`.
pub(crate) fn brighten(
    rgba: &mut [u8],
    w: u32,
    h: u32,
    region: Region,
    amount: f32,
    saturation: f32,
) {
    if !valid(rgba, w, h) || !amount.is_finite() || !saturation.is_finite() {
        return;
    }
    let amount = amount.clamp(-1.0, 1.0);
    if amount == 0.0 {
        return;
    }
    let saturation = saturation.clamp(-1.0, 1.0);
    let wu = w as usize;
    rgba.par_chunks_mut(wu * 4).enumerate().for_each(|(y, line)| {
        for x in 0..wu {
            let m = mask_weight(region.rect, region.feather, (x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            if m <= 0.0 {
                continue;
            }
            let p = x * 4;
            let c = [line[p] as f32 / 255.0, line[p + 1] as f32 / 255.0, line[p + 2] as f32 / 255.0];
            let (hue, sat, val) = rgb_to_hsv(c);
            let v = soft_shoulder((val + amount * m).max(0.0));
            let s = (sat * (1.0 + saturation * m)).clamp(0.0, 1.0);
            let out = hsv_to_rgb(hue, s, v);
            for ch in 0..3 {
                line[p + ch] = (out[ch] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    });
}

/// Teeth whitening: pull Lab b* toward neutral and lift L, gated to bright,
/// low-saturation pixels so lips and gums stay put.
pub(crate) fn whiten(rgba: &mut [u8], w: u32, h: u32, region: Region, amount: f32) {
    if !valid(rgba, w, h) || !amount.is_finite() {
        return;
    }
    let amount = amount.clamp(0.0, 1.0);
    if amount <= 0.0 {
        return;
    }
    let wu = w as usize;
    rgba.par_chunks_mut(wu * 4).enumerate().for_each(|(y, line)| {
        for x in 0..wu {
            let m = mask_weight(region.rect, region.feather, (x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            if m <= 0.0 {
                continue;
            }
            let p = x * 4;
            let c = [line[p] as f32 / 255.0, line[p + 1] as f32 / 255.0, line[p + 2] as f32 / 255.0];
            let (_, sat, val) = rgb_to_hsv(c);
            let gate = (1.0 - smoothstep(0.18, 0.32, sat)) * smoothstep(0.45, 0.70, val);
            let k = amount * m * gate;
            if k <= 0.0 {
                continue;
            }
            let (l, a, b) = lab(c);
            let nb = b - (b - 2.0).max(0.0) * 0.85 * k;
            let nl = l + (100.0 - l) * 0.10 * k;
            let out = lab_to_srgb(nl, a, nb);
            for ch in 0..3 {
                line[p + ch] = (out[ch] * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    });
}

/// Linear mix toward `color` (sRGB 0..1) by `opacity * mask`.
pub(crate) fn tint(
    rgba: &mut [u8],
    w: u32,
    h: u32,
    region: Region,
    color: [f32; 3],
    opacity: f32,
) {
    if !valid(rgba, w, h) || !opacity.is_finite() || color.iter().any(|c| !c.is_finite()) {
        return;
    }
    let opacity = opacity.clamp(0.0, 1.0);
    if opacity == 0.0 {
        return;
    }
    let wu = w as usize;
    let want = color.map(|c| c.clamp(0.0, 1.0) * 255.0);
    rgba.par_chunks_mut(wu * 4).enumerate().for_each(|(y, line)| {
        for x in 0..wu {
            let m = mask_weight(region.rect, region.feather, (x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let t = opacity * m;
            if t <= 0.0 {
                continue;
            }
            let p = x * 4;
            for ch in 0..3 {
                let v = line[p + ch] as f32;
                line[p + ch] = (v + (want[ch] - v) * t).clamp(0.0, 255.0) as u8;
            }
        }
    });
}

/// Blur and desaturate everything outside `keep`. `radius` is a fraction of the
/// image's long side; three separable box passes approximate a Gaussian.
pub(crate) fn background_blur(
    rgba: &mut [u8],
    w: u32,
    h: u32,
    keep: Region,
    radius: f32,
    desaturate: f32,
) {
    if !valid(rgba, w, h) || !radius.is_finite() || !desaturate.is_finite() {
        return;
    }
    let desaturate = desaturate.clamp(0.0, 1.0);
    let (wu, hu) = (w as usize, h as usize);
    let r = (radius.max(0.0) * wu.max(hu) as f32).round() as usize;
    let r = r.min(wu.min(hu) / 4);
    if r == 0 {
        return;
    }
    let mut blurred = rgba.to_vec();
    for _ in 0..BOX_PASSES {
        blurred = box_v(&box_h(&blurred, wu, hu, r), wu, hu, r);
    }

    rgba.par_chunks_mut(wu * 4).enumerate().for_each(|(y, line)| {
        for x in 0..wu {
            let m = mask_weight(keep.rect, keep.feather, (x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let t = 1.0 - m;
            if t <= 0.0 {
                continue;
            }
            let p = x * 4;
            let b = [blurred[p] as f32, blurred[p + 1] as f32, blurred[p + 2] as f32];
            let y0 = luma(b);
            for ch in 0..3 {
                let soft = b[ch] + (y0 - b[ch]) * desaturate;
                line[p + ch] = (line[p + ch] as f32 + (soft - line[p + ch] as f32) * t).clamp(0.0, 255.0) as u8;
            }
        }
    });
}

fn valid(rgba: &[u8], w: u32, h: u32) -> bool {
    w > 0 && h > 0 && rgba.len() as u64 == w as u64 * h as u64 * 4
}

fn bilateral_weights() -> Vec<(i64, i64, f32)> {
    let sigma = 1.4f32;
    let mut out = Vec::with_capacity(((BILATERAL_RADIUS * 2 + 1) * (BILATERAL_RADIUS * 2 + 1)) as usize);
    for dy in -BILATERAL_RADIUS..=BILATERAL_RADIUS {
        for dx in -BILATERAL_RADIUS..=BILATERAL_RADIUS {
            let d2 = (dx * dx + dy * dy) as f32;
            out.push((dx, dy, (-d2 / (2.0 * sigma * sigma)).exp()));
        }
    }
    out
}

/// Per-pixel skin likelihood from Cb/Cr windows, hue, saturation and luma.
fn skin_weight(c: [f32; 3]) -> f32 {
    let y = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
    let cb = 128.0 - 0.168_736 * c[0] - 0.331_264 * c[1] + 0.5 * c[2];
    let cr = 128.0 + 0.5 * c[0] - 0.418_688 * c[1] - 0.081_312 * c[2];
    let (hue, sat, _) = rgb_to_hsv([c[0] / 255.0, c[1] / 255.0, c[2] / 255.0]);
    let hue_dist = if hue > 180.0 { 360.0 - hue } else { hue };
    band(cb, 70.0, 84.0, 126.0, 140.0)
        * band(cr, 130.0, 142.0, 172.0, 182.0)
        * (1.0 - smoothstep(45.0, 60.0, hue_dist))
        * smoothstep(0.06, 0.16, sat)
        * smoothstep(18.0, 45.0, y)
}

/// 1 inside `[b, c]`, ramping over `[a, b]` and `[c, d]`.
fn band(x: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    smoothstep(a, b, x) * (1.0 - smoothstep(c, d, x))
}

/// Monotone highlight shoulder: identity below 0.8, asymptotic to 1 above.
fn soft_shoulder(v: f32) -> f32 {
    const KNEE: f32 = 0.8;
    if v <= KNEE {
        v.max(0.0)
    } else {
        KNEE + (1.0 - KNEE) * (1.0 - (-(v - KNEE) / (1.0 - KNEE)).exp())
    }
}

fn rgb_to_hsv(c: [f32; 3]) -> (f32, f32, f32) {
    let max = c[0].max(c[1]).max(c[2]);
    let min = c[0].min(c[1]).min(c[2]);
    let d = max - min;
    let h = if d < 1e-6 {
        0.0
    } else if max == c[0] {
        60.0 * ((c[1] - c[2]) / d).rem_euclid(6.0)
    } else if max == c[1] {
        60.0 * ((c[2] - c[0]) / d + 2.0)
    } else {
        60.0 * ((c[0] - c[1]) / d + 4.0)
    };
    let s = if max < 1e-6 { 0.0 } else { d / max };
    (h, s, max)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let hp = (h.rem_euclid(360.0)) / 60.0;
    let x = 1.0 - (hp.rem_euclid(2.0) - 1.0).abs();
    let (r, g, b) = match hp as i32 {
        0 => (1.0, x, 0.0),
        1 => (x, 1.0, 0.0),
        2 => (0.0, 1.0, x),
        3 => (0.0, x, 1.0),
        4 => (x, 0.0, 1.0),
        _ => (1.0, 0.0, x),
    };
    let m = v * (1.0 - s);
    [r * s * v + m, g * s * v + m, b * s * v + m]
}

fn luma(c: [f32; 3]) -> f32 {
    0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

fn lab_f(t: f32) -> f32 {
    if t > 0.008_856 {
        t.cbrt()
    } else {
        7.787 * t + 16.0 / 116.0
    }
}

fn lab_f_inv(t: f32) -> f32 {
    let t3 = t * t * t;
    if t3 > 0.008_856 {
        t3
    } else {
        (t - 16.0 / 116.0) / 7.787
    }
}

/// sRGB 0..1 to CIE Lab (D65).
fn lab(c: [f32; 3]) -> (f32, f32, f32) {
    let r = srgb_to_linear(c[0]);
    let g = srgb_to_linear(c[1]);
    let b = srgb_to_linear(c[2]);
    let x = 0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b;
    let y = 0.212_672_9 * r + 0.715_152_2 * g + 0.072_175_0 * b;
    let z = 0.019_333_9 * r + 0.119_192_0 * g + 0.950_304_1 * b;
    let fx = lab_f(x / 0.950_47);
    let fy = lab_f(y);
    let fz = lab_f(z / 1.088_83);
    (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
}

fn lab_to_srgb(l: f32, a: f32, b: f32) -> [f32; 3] {
    let fy = (l + 16.0) / 116.0;
    let x = 0.950_47 * lab_f_inv(fy + a / 500.0);
    let y = lab_f_inv(fy);
    let z = 1.088_83 * lab_f_inv(fy - b / 200.0);
    let r = 3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z;
    let g = -0.969_266_0 * x + 1.876_010_8 * y + 0.041_556_0 * z;
    let bl = 0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z;
    [
        linear_to_srgb(r),
        linear_to_srgb(g),
        linear_to_srgb(bl),
    ]
}

/// One horizontal box pass with edge clamping.
fn box_h(src: &[u8], w: usize, _h: usize, r: usize) -> Vec<u8> {
    let stride = w * 4;
    let mut out = vec![0u8; src.len()];
    out.par_chunks_mut(stride).enumerate().for_each(|(y, line)| {
        let row = &src[y * stride..y * stride + stride];
        let mut pre = vec![0f32; (w + 1) * 3];
        for x in 0..w {
            for k in 0..3 {
                pre[(x + 1) * 3 + k] = pre[x * 3 + k] + row[x * 4 + k] as f32;
            }
        }
        for x in 0..w {
            let a = x.saturating_sub(r);
            let b = (x + r).min(w - 1);
            let count = (b - a + 1) as f32;
            for k in 0..3 {
                let sum = pre[(b + 1) * 3 + k] - pre[a * 3 + k];
                line[x * 4 + k] = (sum / count + 0.5).clamp(0.0, 255.0) as u8;
            }
            line[x * 4 + 3] = row[x * 4 + 3];
        }
    });
    out
}

/// One vertical box pass with edge clamping, sliding sums per row band.
fn box_v(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    let stride = w * 4;
    let band = 32;
    let mut out = vec![0u8; src.len()];
    out.par_chunks_mut(stride * band).enumerate().for_each(|(bi, chunk)| {
        let start = bi * band;
        if start >= h {
            return;
        }
        let end = (start + band).min(h);
        let mut acc = vec![0f32; w * 3];
        let mut lo = start.saturating_sub(r);
        let mut hi = (start + r).min(h - 1);
        for y in lo..=hi {
            add_row(src, w, y, &mut acc);
        }
        for y in start..end {
            let count = (hi - lo + 1) as f32;
            let row = (y - start) * stride;
            for x in 0..w {
                for k in 0..3 {
                    chunk[row + x * 4 + k] = (acc[x * 3 + k] / count + 0.5).clamp(0.0, 255.0) as u8;
                }
                chunk[row + x * 4 + 3] = src[y * stride + x * 4 + 3];
            }
            let next_lo = (y + 1).saturating_sub(r);
            let next_hi = (y + 1 + r).min(h - 1);
            while lo < next_lo {
                sub_row(src, w, lo, &mut acc);
                lo += 1;
            }
            while hi < next_hi {
                hi += 1;
                add_row(src, w, hi, &mut acc);
            }
        }
    });
    out
}

fn add_row(src: &[u8], w: usize, y: usize, acc: &mut [f32]) {
    let base = y * w * 4;
    for x in 0..w {
        for k in 0..3 {
            acc[x * 3 + k] += src[base + x * 4 + k] as f32;
        }
    }
}

fn sub_row(src: &[u8], w: usize, y: usize, acc: &mut [f32]) {
    let base = y * w * 4;
    for x in 0..w {
        for k in 0..3 {
            acc[x * 3 + k] -= src[base + x * 4 + k] as f32;
        }
    }
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

    const SKIN: [u8; 3] = [204, 156, 128];

    fn frame(w: u32, h: u32, c: [u8; 3]) -> Vec<u8> {
        let mut buf = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..(w * h) {
            buf.extend_from_slice(&[c[0], c[1], c[2], 255]);
        }
        buf
    }

    fn px(buf: &[u8], w: u32, x: u32, y: u32) -> [u8; 3] {
        let i = ((y * w + x) * 4) as usize;
        [buf[i], buf[i + 1], buf[i + 2]]
    }

    fn region(rect: [f32; 4], feather: f32) -> Region {
        Region { rect, feather }
    }

    /// Variance of luma inside a pixel box.
    fn variance(buf: &[u8], w: u32, x0: u32, y0: u32, x1: u32, y1: u32) -> f32 {
        let mut vals = Vec::new();
        for y in y0..y1 {
            for x in x0..x1 {
                let c = px(buf, w, x, y);
                vals.push(luma([c[0] as f32, c[1] as f32, c[2] as f32]));
            }
        }
        let mean = vals.iter().sum::<f32>() / vals.len() as f32;
        vals.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / vals.len() as f32
    }

    #[test]
    fn mask_weight_ramps_across_the_border() {
        let rect = [0.25, 0.25, 0.5, 0.5];
        assert_eq!(mask_weight(rect, 0.0, 0.5, 0.5), 1.0);
        assert_eq!(mask_weight(rect, 0.0, 0.05, 0.5), 0.0);
        let edge = mask_weight(rect, 0.1, 0.25, 0.5);
        assert!((edge - 0.5).abs() < 0.01, "border weight {edge}");
        assert_eq!(mask_weight(rect, 0.1, f32::NAN, 0.5), 0.0);
    }

    #[test]
    fn zero_amounts_leave_bytes_untouched() {
        let mut buf = frame(16, 16, SKIN);
        let before = buf.clone();
        let r = region([0.0, 0.0, 1.0, 1.0], 0.0);
        let mut skin = buf.clone();
        skin_smooth(&mut skin, 16, 16, r, 0.0, 0.5);
        assert_eq!(skin, before, "skin_smooth with zero strength");
        brighten(&mut buf, 16, 16, r, 0.0, 1.0);
        whiten(&mut buf, 16, 16, r, 0.0);
        tint(&mut buf, 16, 16, r, [1.0, 0.0, 0.0], 0.0);
        let mut blur = vec![80u8; 16 * 16 * 4];
        let before_blur = blur.clone();
        background_blur(&mut buf, 16, 16, r, 0.1, 1.0);
        assert_eq!(buf, before);
        background_blur(&mut blur, 16, 16, r, 0.0, 1.0);
        assert_eq!(blur, before_blur, "background_blur with zero radius");
    }

    #[test]
    fn smooth_lowers_skin_noise_variance() {
        let (w, h) = (48u32, 48u32);
        let mut buf = frame(w, h, SKIN);
        for y in 0..h {
            for x in 0..w {
                let n = if (x + y) % 2 == 0 { 9u8 } else { 0u8 };
                let i = ((y * w + x) * 4) as usize;
                for k in 0..3 {
                    buf[i + k] = buf[i + k].saturating_add(n);
                }
            }
        }
        let before = variance(&buf, w, 4, 4, w - 4, h - 4);
        skin_smooth(&mut buf, w, h, region([0.0, 0.0, 1.0, 1.0], 0.0), 1.0, 0.0);
        let after = variance(&buf, w, 4, 4, w - 4, h - 4);
        assert!(after < before * 0.5, "variance {before} -> {after}");
    }

    #[test]
    fn smooth_keeps_the_hair_edge() {
        let (w, h) = (48u32, 48u32);
        let mut buf = frame(w, h, SKIN);
        for y in 0..h {
            for x in 24..w {
                let i = ((y * w + x) * 4) as usize;
                buf[i..i + 3].copy_from_slice(&[96, 94, 100]);
            }
        }
        let edge_before = px(&buf, w, 20, 20)[0] as i32 - px(&buf, w, 28, 20)[0] as i32;
        skin_smooth(&mut buf, w, h, region([0.0, 0.0, 1.0, 1.0], 0.0), 1.0, 0.0);
        let edge_after = px(&buf, w, 20, 20)[0] as i32 - px(&buf, w, 28, 20)[0] as i32;
        assert!(edge_after > (edge_before as f32 * 0.95) as i32, "edge {edge_before} -> {edge_after}");
    }

    #[test]
    fn brighten_lifts_the_eye_region_only() {
        let (w, h) = (32u32, 32u32);
        let mut buf = frame(w, h, [70, 72, 84]);
        let before_out = px(&buf, w, 2, 2);
        brighten(&mut buf, w, h, region([0.4, 0.4, 0.2, 0.2], 0.0), 0.5, 0.2);
        let inside = px(&buf, w, 16, 16);
        assert!(luma(inside.map(|v| v as f32)) > luma([70.0, 72.0, 84.0]) + 10.0, "inside {inside:?}");
        assert_eq!(px(&buf, w, 2, 2), before_out);
    }

    #[test]
    fn whiten_bleaches_teeth_but_not_lips() {
        let (w, h) = (32u32, 32u32);
        let mut buf = frame(w, h, [238, 232, 200]);
        for y in 16..32 {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                buf[i..i + 3].copy_from_slice(&[168, 92, 96]);
            }
        }
        let lip_before = px(&buf, w, 5, 24);
        whiten(&mut buf, w, h, region([0.0, 0.0, 1.0, 1.0], 0.0), 1.0);
        let teeth = px(&buf, w, 5, 5);
        let yellow_before = 238i32 - 200i32;
        let yellow_after = teeth[0] as i32 - teeth[2] as i32;
        assert!(yellow_after < yellow_before - 8, "yellow {yellow_before} -> {yellow_after}");
        assert_eq!(px(&buf, w, 5, 24), lip_before);
    }

    #[test]
    fn tint_mixes_the_region() {
        let (w, h) = (24u32, 24u32);
        let mut buf = frame(w, h, [120, 120, 120]);
        tint(&mut buf, w, h, region([0.25, 0.25, 0.5, 0.5], 0.0), [1.0, 0.2, 0.2], 1.0);
        let inside = px(&buf, w, 12, 12);
        assert!(inside[0] > 240 && inside[1] < 70, "tinted {inside:?}");
        assert_eq!(px(&buf, w, 1, 1), [120, 120, 120]);
    }

    #[test]
    fn background_blur_keeps_the_keep_region() {
        let (w, h) = (96u32, 96u32);
        let mut buf = frame(w, h, [80, 110, 150]);
        for y in 0..h {
            for x in 0..w {
                let n = if (x * 7 + y * 13) % 5 == 0 { 60u8 } else { 0u8 };
                let i = ((y * w + x) * 4) as usize;
                for k in 0..3 {
                    buf[i + k] = buf[i + k].saturating_add(n);
                }
            }
        }
        let before = buf.clone();
        let keep = region([0.3, 0.3, 0.4, 0.4], 0.02);
        background_blur(&mut buf, w, h, keep, 0.05, 0.5);
        for y in 34..62 {
            for x in 34..62 {
                assert_eq!(px(&buf, w, x, y), px(&before, w, x, y), "pixel {x},{y} moved");
            }
        }
        let out_var = variance(&buf, w, 2, 2, 26, 26);
        let in_var = variance(&before, w, 2, 2, 26, 26);
        assert!(out_var < in_var * 0.25, "outside variance {in_var} -> {out_var}");
    }

    #[test]
    fn bad_input_is_ignored() {
        let mut empty: Vec<u8> = Vec::new();
        let mut short = vec![0u8; 8];
        let r = region([0.0, 0.0, 1.0, 1.0], 0.1);
        skin_smooth(&mut empty, 0, 0, r, 1.0, 0.0);
        skin_smooth(&mut short, 4, 4, r, 1.0, 0.0);
        brighten(&mut short, 4, 4, r, f32::NAN, 0.0);
        whiten(&mut short, 4, 4, r, 1.0);
        tint(&mut short, 4, 4, r, [f32::NAN, 0.0, 0.0], 1.0);
        background_blur(&mut short, 4, 4, r, 0.2, 0.5);
        let mut buf = frame(8, 8, SKIN);
        let before = buf.clone();
        let off = region([5.0, 5.0, 0.2, 0.2], 0.05);
        skin_smooth(&mut buf, 8, 8, off, 1.0, 0.0);
        tint(&mut buf, 8, 8, region([0.2, 0.2, f32::NAN, 0.1], 0.1), [1.0, 0.0, 0.0], 1.0);
        assert_eq!(buf, before);
    }
}
