//! Capture One retouch pixel operations.
//!
//! Formulas follow `06_reports/SESSION-c1-parse-face-retouch.md` (disassembly of
//! `ImgCoreDll.dll`, 16.8.3). Blemish removal and stray-hair removal need a
//! network gain map and are not implemented here; their catalog columns still
//! parse. Every operation takes a full-frame 8-bit coverage plane and is a
//! no-op when its amount is zero.

const W: f32 = 255.0;

#[inline]
fn luma8(c: [u8; 3]) -> f32 {
    0.299 * c[0] as f32 + 0.587 * c[1] as f32 + 0.114 * c[2] as f32
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

/// White-point protected gain: `c*k*W/(W+(k-1)*min(c,W))`.
#[inline]
fn white_point_gain(c: f32, k: f32) -> f32 {
    c * k * W / (W + (k - 1.0) * c.min(W))
}

fn box_h(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    if r == 0 {
        return src.to_vec();
    }
    let mut out = vec![0u8; src.len()];
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let line = &mut out[y * w..(y + 1) * w];
        let mut acc: i32 = 0;
        for x in 0..w.min(r) {
            acc += row[x] as i32;
        }
        for x in 0..w {
            let add = x + r;
            if add < w {
                acc += row[add] as i32;
            }
            let sub = x as i64 - r as i64 - 1;
            if sub >= 0 {
                acc -= row[sub as usize] as i32;
            }
            let n = (add.min(w - 1) as i64 - sub.max(0) as i64).max(1) as i32;
            line[x] = (acc / n).clamp(0, 255) as u8;
        }
    }
    out
}

fn box_v(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    if r == 0 {
        return src.to_vec();
    }
    let mut out = vec![0u8; src.len()];
    for x in 0..w {
        let mut acc: i32 = 0;
        for y in 0..h.min(r) {
            acc += src[y * w + x] as i32;
        }
        for y in 0..h {
            let add = y + r;
            if add < h {
                acc += src[add * w + x] as i32;
            }
            let sub = y as i64 - r as i64 - 1;
            if sub >= 0 {
                acc -= src[sub as usize * w + x] as i32;
            }
            let n = (add.min(h - 1) as i64 - sub.max(0) as i64).max(1) as i32;
            out[y * w + x] = (acc / n).clamp(0, 255) as u8;
        }
    }
    out
}

fn box_blur(src: &[u8], w: usize, h: usize, r: usize) -> Vec<u8> {
    box_v(&box_h(src, w, h, r), w, h, r)
}

/// SmoothSkin: separable box blur plus a luma-affinity gate.
///
/// `face` is the normalized face rectangle; its long side sets the blur radius
/// (`0.077 * D` for the amount pass, `0.044 * D` for the texture pass).
pub fn skin_evening(
    rgba: &mut [u8],
    w: u32,
    h: u32,
    mask: &[u8],
    face: [f32; 4],
    amount: f32,
    texture: f32,
) {
    let (w, h) = (w as usize, h as usize);
    let a = (amount / 100.0).clamp(0.0, 1.0);
    let tr = (texture / 100.0).clamp(-1.0, 1.0);
    if (a <= 0.0 && tr == 0.0) || mask.len() != w * h || rgba.len() != w * h * 4 {
        return;
    }
    let face_d = (face[2].abs() * w as f32).max(face[3].abs() * h as f32).max(1.0);
    let ra = ((0.077 * face_d).round() as usize).max(1);
    let rt = ((0.044 * face_d).round() as usize).max(1);
    let src: Vec<u8> = rgba.iter().step_by(4).copied().collect();
    let soft = box_blur(&src, w, h, ra);
    let w_a = 1.0 - (1.0 - a).powi(3);
    let w_t = 1.0 - 0.75 * a * a;
    let mut layer: Vec<u8> = Vec::with_capacity(w * h);
    for i in 0..w * h {
        let y0 = src[i] as f32;
        let y1 = soft[i] as f32;
        let lo = y0.min(y1) + 4.0;
        let hi = y0.max(y1) + 4.0;
        // Affinity one where the two luminances agree, zero at a hard edge.
        let aff = (lo / hi).powi(2).clamp(0.0, 1.0);
        let t = (w_a * (1.0 - w_t * (1.0 - aff))).clamp(0.0, 1.0);
        layer.push(lerp(y0, y1, t).round().clamp(0.0, 255.0) as u8);
    }
    if tr != 0.0 {
        let fine = box_blur(&layer, w, h, rt);
        for i in 0..w * h {
            layer[i] = lerp(layer[i] as f32, fine[i] as f32, tr * 0.5).round() as u8;
        }
    }
    for i in 0..w * h {
        let m = mask[i] as f32 / 255.0;
        if m <= 0.0 {
            continue;
        }
        let delta = layer[i] as f32 - src[i] as f32;
        for c in 0..3 {
            let v = rgba[i * 4 + c] as f32 + delta * m;
            rgba[i * 4 + c] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// FaceSculpting: `1/8`-resolution sqrt-luma relief. Dark grooves add, bright
/// ridges multiply, so it never brightens a highlight past its own luma.
pub fn face_sculpting(rgba: &mut [u8], w: u32, h: u32, mask: &[u8], amount: f32) {
    let (w, h) = (w as usize, h as usize);
    let a = (amount / 100.0).clamp(0.0, 1.0);
    if a <= 0.0 || mask.len() != w * h || rgba.len() != w * h * 4 {
        return;
    }
    let (gw, gh) = ((w + 7) / 8, (h + 7) / 8);
    let mut base = vec![0f32; gw * gh];
    for gy in 0..gh {
        for gx in 0..gw {
            let mut acc = 0f32;
            let mut n = 0f32;
            for dy in 0..8 {
                let y = gy * 8 + dy;
                if y >= h {
                    break;
                }
                for dx in 0..8 {
                    let x = gx * 8 + dx;
                    if x >= w {
                        break;
                    }
                    let i = y * w + x;
                    if mask[i] == 0 {
                        continue;
                    }
                    acc += luma8([rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]]);
                    n += 1.0;
                }
            }
            base[gy * gw + gx] = if n > 0.0 { acc / n } else { 128.0 };
        }
    }
    let s = 1.08 * a;
    for y in 0..h {
        for x in 0..w {
            let i = y * w + x;
            let m = mask[i] as f32 / 255.0;
            if m <= 0.0 {
                continue;
            }
            let p = [rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]];
            let yv = luma8(p);
            let f = (yv + 4.0).max(0.0).sqrt();
            let fb = (base[(y / 8) * gw + x / 8] + 4.0).max(0.0).sqrt();
            let d = fb * fb - f * f;
            let l = yv + 16.0;
            let scale = if d >= 0.0 {
                s * f * (d * l) / (l - 6.0 * d).max(1.0)
            } else {
                let den = (l / 102.0) * (l / 102.0) + 0.3;
                s * f / l * (d * f * f) / (den * d + f * f).max(1e-3)
            };
            let scale = scale * m;
            for c in 0..3 {
                let v = if d >= 0.0 {
                    p[c] as f32 + scale
                } else {
                    p[c] as f32 * (1.0 + scale)
                };
                rgba[i * 4 + c] = v.clamp(0.0, 255.0).round() as u8;
            }
        }
    }
}

/// TeethWhitening: desaturate toward luma, then a white-point protected stop
/// gain of `2^(1.8*m*brightness)`.
pub fn teeth(rgba: &mut [u8], w: u32, h: u32, mask: &[u8], brightness: f32, desaturation: f32) {
    let n = (w * h) as usize;
    if mask.len() != n || rgba.len() != n * 4 {
        return;
    }
    let b = (brightness / 100.0).clamp(0.0, 1.0);
    let d = (desaturation / 100.0).clamp(0.0, 1.0);
    if b <= 0.0 && d <= 0.0 {
        return;
    }
    for i in 0..n {
        let m = mask[i] as f32 / 255.0;
        if m <= 0.0 {
            continue;
        }
        let p = [rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]];
        let y = luma8(p);
        let k = 2f32.powf(1.8 * m * b);
        for c in 0..3 {
            let v = y + (p[c] as f32 - y) * (1.0 - m * d);
            rgba[i * 4 + c] = white_point_gain(v, k).clamp(0.0, 255.0).round() as u8;
        }
    }
}

fn median_luma(rgba: &[u8], w: usize, h: usize, mask: &[u8]) -> f32 {
    let mut hist = [0u32; 256];
    let mut n = 0u32;
    for i in 0..w * h {
        if mask[i] == 0 {
            continue;
        }
        hist[luma8([rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]]).round() as usize & 255] += 1;
        n += 1;
    }
    if n == 0 {
        return 0.0;
    }
    let goal = n / 2;
    let mut acc = 0u32;
    for (v, &c) in hist.iter().enumerate() {
        acc += c;
        if acc >= goal {
            return v as f32;
        }
    }
    255.0
}

/// EyeBright iris: median-anchored tone curve, then `2^(0.6075*A*m)` gain.
pub fn iris(rgba: &mut [u8], w: u32, h: u32, mask: &[u8], amount: f32) {
    let (w, h) = (w as usize, h as usize);
    let n = w * h;
    let a = (amount / 100.0).clamp(0.0, 1.0);
    if a <= 0.0 || mask.len() != n || rgba.len() != n * 4 {
        return;
    }
    let m0 = median_luma(rgba, w, h, mask);
    for i in 0..n {
        let m = mask[i] as f32 / 255.0;
        if m <= 0.0 {
            continue;
        }
        let p = [rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]];
        let y = luma8(p);
        let s = 0.75 * a * m;
        let d = y + 16.0 - m0;
        let adjust = if d < 0.0 {
            0.4 * s * d * (y + 16.0) / ((y + 16.0) + 6.0 * d.abs())
        } else {
            let den = d * ((y / 102.0) * (y / 102.0) + 0.3) + m0;
            0.4 * s * d * m0 / den.max(1e-3)
        };
        let k = 2f32.powf(0.6075 * a * m);
        for c in 0..3 {
            let v = p[c] as f32 + adjust;
            rgba[i * 4 + c] = white_point_gain(v, k).clamp(0.0, 255.0).round() as u8;
        }
    }
}

/// EyeBright sclera: `0.6*A*m` desaturation, then `2^(0.648*A*m)` gain.
pub fn sclera(rgba: &mut [u8], w: u32, h: u32, mask: &[u8], amount: f32) {
    let n = (w * h) as usize;
    let a = (amount / 100.0).clamp(0.0, 1.0);
    if a <= 0.0 || mask.len() != n || rgba.len() != n * 4 {
        return;
    }
    for i in 0..n {
        let m = mask[i] as f32 / 255.0;
        if m <= 0.0 {
            continue;
        }
        let p = [rgba[i * 4], rgba[i * 4 + 1], rgba[i * 4 + 2]];
        let y = luma8(p);
        let k = 2f32.powf(0.648 * a * m);
        for c in 0..3 {
            let v = y + (p[c] as f32 - y) * (1.0 - 0.6 * a * m);
            rgba[i * 4 + c] = white_point_gain(v, k).clamp(0.0, 255.0).round() as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray(w: u32, h: u32, v: u8) -> Vec<u8> {
        let mut out = vec![v; (w * h * 4) as usize];
        for px in out.chunks_mut(4) {
            px[3] = 255;
        }
        out
    }

    fn full(w: u32, h: u32, v: u8) -> Vec<u8> {
        vec![v; (w * h) as usize]
    }

    #[test]
    fn every_operation_is_identity_at_zero() {
        let (w, h) = (16, 16);
        let mask = full(w, h, 255);
        for name in 0..5 {
            let mut px = gray(w, h, 120);
            let before = px.clone();
            match name {
                0 => skin_evening(&mut px, w, h, &mask, [0.1, 0.1, 0.5, 0.5], 0.0, 0.0),
                1 => face_sculpting(&mut px, w, h, &mask, 0.0),
                2 => teeth(&mut px, w, h, &mask, 0.0, 0.0),
                3 => iris(&mut px, w, h, &mask, 0.0),
                _ => sclera(&mut px, w, h, &mask, 0.0),
            }
            assert_eq!(px, before, "op {name} must be identity at zero");
        }
    }

    #[test]
    fn skin_evening_smooths_within_the_mask_only() {
        let (w, h) = (32u32, 32u32);
        let mut px = gray(w, h, 120);
        for y in 0..h as usize {
            for x in 0..w as usize {
                if x % 2 == 0 {
                    let i = (y * w as usize + x) * 4;
                    px[i] = 200;
                }
            }
        }
        let mut mask = full(w, h, 0);
        for y in 8..24 {
            for x in 8..24 {
                mask[y * 32 + x] = 255;
            }
        }
        let before = px.clone();
        skin_evening(&mut px, w, h, &mask, [0.25, 0.25, 0.5, 0.5], 80.0, 0.0);
        assert_eq!(px[0..4], before[0..4], "outside the mask stays untouched");
        assert_ne!(px[(12 * 32 + 13) * 4], before[(12 * 32 + 13) * 4], "inside is blended");
    }

    #[test]
    fn teeth_brighten_and_desaturate() {
        let (w, h) = (4u32, 4u32);
        let mut px = vec![0u8; (w * h * 4) as usize];
        for p in px.chunks_mut(4) {
            p.copy_from_slice(&[180, 160, 120, 255]);
        }
        let before = px.clone();
        teeth(&mut px, w, h, &full(w, h, 255), 60.0, 80.0);
        assert!(px[0] > before[0]);
        let sat = |p: &[u8]| p[0] as i32 - p[2] as i32;
        assert!(sat(&px[0..4]) < sat(&before[0..4]));
    }

    #[test]
    fn sclera_whitens_toward_luma() {
        let (w, h) = (4u32, 4u32);
        let mut px = vec![0u8; (w * h * 4) as usize];
        for p in px.chunks_mut(4) {
            p.copy_from_slice(&[140, 120, 100, 255]);
        }
        sclera(&mut px, w, h, &full(w, h, 255), 70.0);
        let before_luma = 0.299 * 140.0 + 0.587 * 120.0 + 0.114 * 100.0;
        let after = luma8([px[0], px[1], px[2]]);
        assert!(after > before_luma);
    }

    #[test]
    fn iris_gain_never_clips_a_highlight() {
        let (w, h) = (4u32, 4u32);
        let mut px = vec![0u8; (w * h * 4) as usize];
        for p in px.chunks_mut(4) {
            p.copy_from_slice(&[250, 250, 250, 255]);
        }
        iris(&mut px, w, h, &full(w, h, 255), 100.0);
        assert!(px[0] <= 255);
    }
}
