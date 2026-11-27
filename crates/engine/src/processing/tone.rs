//! Tone operators: local tone mapping (HDR), brightness/contrast, and dehaze.
//!
//! These are pure per-pixel functions that depend only on adjustment value
//! types ([`HighDynamicRange`], [`Exposure`], [`Clarity`], [`Dehaze`]). They
//! were moved out of `mock::process` byte-identically so the tone stage can be
//! shared without pulling in `Source`, `Plan`, caches, or any GPU path. The
//! mock renderer imports them back and re-exports nothing, so its call sites
//! and tests keep the unqualified names.

use crate::{Clarity, ClarityMethod, Dehaze, Exposure, HighDynamicRange};

use super::color::{contrast, luma};

/// Unbounded ramp `z06afa3a107` from `CImgOpLocalToneMappingV8` (`p_041541440.cl`).
#[inline]
fn hdr_ramp(x: f32) -> f32 {
    let u = x * (2.0 * 0.9);
    let v = u.abs() + 1.0;
    let r0 = (u * v) / (v + u * u) * 0.5 + 0.5;
    r0.max(u - r0 + 1.0) * 0.5
}

/// The hard toe `z2714f67010` of `CImgOpLocalToneMappingV8` (`p_041541440.cl`).
#[inline]
fn hdr_toe(x: f32, k: f32) -> f32 {
    let r = hdr_ramp(x) / k;
    let s = r * r;
    let num = (1.0 + r + s) * (r + s * s);
    num / (1.0 + num) * k
}

/// `z386cc3479b` — mirrored toe used by Shadows and Blacks.
#[inline]
fn hdr_inv_toe(x: f32, k: f32) -> f32 {
    hdr_toe(-x, k)
}

const HDR_K_SHADOW: f32 = 2.1;
const HDR_K_HIGHLIGHT: f32 = 3.0;
const HDR_K_BLACK: f32 = 4.2;
const HDR_WHITES_OFFSET: f32 = -10.0 / 9.0;
const HDR_BLACKS_OFFSET: f32 = 5.0 / 3.0;
const HDR_NEG_MULT: f32 = 4.0;
const HDR_MID_GRAY: f32 = 0.18;

/// `CImgOpLocalToneMappingV8` (`p_041541440.cl`, `z3dbd1ff682 == 13`):
/// four-branch log2 EV gain `(g_sh * g_bl) / (g_hi * g_wh)`.
pub(crate) fn apply_hdr(g: [f32; 3], hdr: &HighDynamicRange) -> [f32; 3] {
    if hdr.highlights == 0.0 && hdr.shadows == 0.0 && hdr.whites == 0.0 && hdr.blacks == 0.0 {
        return g;
    }
    let x_ev = (luma(g).max(1e-6) / HDR_MID_GRAY).log2();

    let g_sh = if hdr.shadows != 0.0 {
        let m = (hdr.shadows / 100.0).clamp(-1.0, 1.0);
        let t = hdr_inv_toe(x_ev, HDR_K_SHADOW);
        let ev = if m < 0.0 { -t } else { t };
        2f32.powf(ev) * m.abs() + (1.0 - m.abs())
    } else {
        1.0
    };

    let g_hi = if hdr.highlights != 0.0 {
        let m = (hdr.highlights * 0.6 / 100.0).clamp(-1.0, 1.0);
        let t = hdr_toe(x_ev, HDR_K_HIGHLIGHT);
        let ev = if m < 0.0 { -t } else { t };
        2f32.powf(ev) * m.abs() + (1.0 - m.abs())
    } else {
        1.0
    };

    let g_bl = if hdr.blacks != 0.0 {
        let m = (hdr.blacks / 100.0).clamp(-1.0, 1.0);
        let t1 = hdr_inv_toe(x_ev, HDR_K_BLACK) - hdr_inv_toe(x_ev, HDR_K_SHADOW);
        let x_sh = x_ev + g_sh.log2() + t1;
        let t2 = t1 + hdr_inv_toe(x_sh + HDR_BLACKS_OFFSET, HDR_K_BLACK - HDR_K_SHADOW);
        let ev = if m < 0.0 { -HDR_NEG_MULT * t2 } else { t2 };
        2f32.powf(ev) * m.abs() + (1.0 - m.abs())
    } else {
        1.0
    };

    let g_wh = if hdr.whites != 0.0 {
        let m = (-hdr.whites / 100.0).clamp(-1.0, 1.0);
        let t = hdr_ramp(x_ev - g_hi.log2() + HDR_WHITES_OFFSET);
        let ev = if m < 0.0 { -HDR_NEG_MULT * t } else { t };
        2f32.powf(ev) * m.abs() + (1.0 - m.abs())
    } else {
        1.0
    };

    let gain = (g_sh * g_bl) / (g_hi * g_wh);
    [g[0] * gain, g[1] * gain, g[2] * gain]
}

/// Brightness and contrast, the per-pixel part of the tone stage.
pub(crate) fn tone_user_core(
    mut g: [f32; 3],
    exp: &Exposure,
) -> [f32; 3] {
    if exp.brightness.abs() > 1e-3 {
        let gamma = 2f32.powf(-exp.brightness / 50.0 * 0.6);
        g = [g[0].max(0.0).powf(gamma), g[1].max(0.0).powf(gamma), g[2].max(0.0).powf(gamma)];
    }
    let c = exp.contrast / 50.0 * 0.8;
    g = [contrast(g[0], c), contrast(g[1], c), contrast(g[2], c)];
    g
}

/// The old point-operation clarity. The frame-wide stage is
/// [`super::clarity::apply`]; this stays for layer colour stacks, which have
/// no frame-sized neighbourhood to filter, and for the tone-stage test.
pub(crate) fn tone_user(
    g: [f32; 3],
    exp: &Exposure,
    clarity: &Clarity,
) -> [f32; 3] {
    let mut g = tone_user_core(g, exp);
    let local = clarity.c1_amount() + clarity.c1_structure();
    if local.abs() > 1e-4 {
        let l = luma(g);
        let mid = 4.0 * l * (1.0 - l);
        let k = 1.0 + local * 0.6 * mid;
        g = [0.5 + (g[0] - 0.5) * k, 0.5 + (g[1] - 0.5) * k, 0.5 + (g[2] - 0.5) * k];
        if matches!(clarity.method, ClarityMethod::Punch) {
            let l = luma(g);
            let s = 1.0 + local * 0.25;
            g = [l + (g[0] - l) * s, l + (g[1] - l) * s, l + (g[2] - l) * s];
        }
    }
    g
}

/// Capture One's `soft(x,a)` knee: `max(x,0) + max(a-|x|,0)³/(6a²)`.
fn dehaze_soft(x: f32, a: f32) -> f32 {
    if a <= 0.0 {
        return x.max(0.0);
    }
    let t = (a - x.abs()).max(0.0);
    x.max(0.0) + t * t * t / (6.0 * a * a)
}

#[inline]
fn dehaze_core(omega: f32, a: [f32; 3], ah: [f32; 3], i: [f32; 3]) -> [f32; 3] {
    let eps = 1.1920929e-7f32;
    let mut i1 = i;
    let mut a1 = a;
    let mut recovered = [0.0f32; 3];
    if omega > 0.0 {
        for c in 0..3 {
            i1[c] = dehaze_soft(i1[c] - ah[c], ah[c]);
            a1[c] = dehaze_soft(a1[c] - ah[c], ah[c]);
        }
        let li = luma(i1);
        let la = luma(a1);
        let keep = la / (li + la).max(eps);
        for c in 0..3 {
            recovered[c] = (1.0 - keep) * i1[c];
            i1[c] *= keep;
        }
    }
    let m_raw = (i1[0] / a1[0].max(eps))
        .min(i1[1] / a1[1].max(eps))
        .min(i1[2] / a1[2].max(eps));
    let mut m = m_raw.clamp(0.0, 1.0);
    if omega > 0.0 {
        let mag = (i1[0] * i1[0] + i1[1] * i1[1] + i1[2] * i1[2]).sqrt();
        let dvec = a1.map(|v| m_raw * v);
        let mag_d = ((i1[0] - dvec[0]).powi(2)
            + (i1[1] - dvec[1]).powi(2)
            + (i1[2] - dvec[2]).powi(2))
        .sqrt();
        let ratio = (mag + eps).max(eps);
        let val = (ratio * 3.0 - (mag_d + eps) * 8.0) / ratio;
        let t = 1.0 / (1.0 + ((0.5 - val) * 4.0).exp());
        m *= t;
        for c in 0..3 {
            recovered[c] += (1.0 - t) * i1[c];
            i1[c] *= t;
        }
    }
    let omega_eff = if omega < 0.0 { omega * 2.0 } else { omega };
    let t0 = (1.0 - omega_eff * m).max(0.05);
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        out[c] = (i1[c] - a1[c]) / t0 + a1[c] + recovered[c];
    }
    let max_i = i1[0].max(i1[1]).max(i1[2]);
    if max_i > eps {
        let sat_i = (max_i - i1[0].min(i1[1]).min(i1[2])) / max_i;
        let limit = sat_i * (1.0 + (0.5 + 2.0 * m_raw).powi(2)).sqrt();
        let max_o = out[0].max(out[1]).max(out[2]);
        if max_o > eps {
            let sat_o = (max_o - out[0].min(out[1]).min(out[2])) / max_o;
            if sat_o > limit {
                let k = limit / sat_o;
                let l = luma(out);
                out = [
                    k * out[0] + (1.0 - k) * l,
                    k * out[1] + (1.0 - k) * l,
                    k * out[2] + (1.0 - k) * l,
                ];
            }
        }
    }
    out
}

/// The `CImgOpDehaze2020` kernel (`__kernel void zec60c625ee`, `ImgCoreDll!0x2782f70`).
pub(crate) fn apply_dehaze(g: [f32; 3], d: &Dehaze) -> [f32; 3] {
    let omega = (d.amount / 100.0).clamp(-1.0, 1.0);
    if omega == 0.0 || !omega.is_finite() {
        return g;
    }
    let c = d.shadow_tone.map(|v| v.max(0.0));
    let a = c.map(|v| v * 65535.0);
    if a.iter().all(|v| *v <= f32::EPSILON) {
        return g;
    }
    let level: [(bool, f32, f32); 3] = std::array::from_fn(|ch| {
        let (lo, hi) = (d.level_low[ch], d.level_high[ch]);
        (hi - lo > 1e-3, lo, hi - lo)
    });
    let i: [f32; 3] = std::array::from_fn(|ch| {
        if level[ch].0 {
            (g[ch] - level[ch].1) / level[ch].2 * 65535.0
        } else {
            g[ch] * 65535.0
        }
    });
    let delta = (2.0 * omega).min(1.0);
    let a_min = a[0].min(a[1]).min(a[2]);
    let ah = [a_min * delta; 3];
    let mut out = dehaze_core(omega, a, ah, i);
    if omega > 0.0 {
        let white_ref = [65535.0f32; 3];
        let ref_out = dehaze_core(omega, a, ah, white_ref);
        for ch in 0..3 {
            if ref_out[ch] > 1e-3 {
                out[ch] *= white_ref[ch] / ref_out[ch];
            }
        }
    }
    std::array::from_fn(|ch| {
        if level[ch].0 {
            (level[ch].1 + out[ch] * level[ch].2 / 65535.0).max(0.0)
        } else {
            (out[ch] / 65535.0).max(0.0)
        }
    })
}

/// Capture One `IC_DehazeLevelsCompute` (1%/99% histogram) + `IC_DehazeColorCompute`
/// (Dark Channel Prior min-filter + Top 0.1% min-heap atmospheric light estimate).
/// Returns `(haze_color, level_low, level_high)` normalized to `0..=1`.
pub fn estimate_dehaze_auto(rgb_u16: &[u16], width: usize, height: usize) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let n = width * height;
    if n == 0 || rgb_u16.len() < n * 3 {
        return ([0.8, 0.8, 0.8], [0.0; 3], [1.0; 3]);
    }
    let mut hist = vec![[0u32; 3]; 65536];
    let mut dark = vec![0u16; n];
    let mut y_approx = vec![0u16; n];
    for idx in 0..n {
        let r = rgb_u16[idx * 3];
        let g = rgb_u16[idx * 3 + 1];
        let b = rgb_u16[idx * 3 + 2];
        hist[r as usize][0] += 1;
        hist[g as usize][1] += 1;
        hist[b as usize][2] += 1;
        dark[idx] = r.min(g).min(b);
        y_approx[idx] = (r >> 2).saturating_add(g >> 1).saturating_add(b >> 2);
    }
    let cut = (n / 100) as u32;
    let mut level_low = [0.0f32; 3];
    let mut level_high = [1.0f32; 3];
    for ch in 0..3 {
        let mut acc = 0u32;
        for bin in 0..65536 {
            acc += hist[bin][ch];
            if acc >= cut {
                level_low[ch] = bin as f32 / 65535.0;
                break;
            }
        }
        acc = 0;
        for bin in (0..65536).rev() {
            acc += hist[bin][ch];
            if acc >= cut {
                level_high[ch] = bin as f32 / 65535.0;
                break;
            }
        }
        if level_high[ch] - level_low[ch] < 1e-3 {
            level_low[ch] = 0.0;
            level_high[ch] = 1.0;
        }
    }

    let radius = (width.min(height) / 4).min(16);
    let mut eroded_h = vec![0u16; n];
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius + 1).min(width);
            let mut m = u16::MAX;
            for xx in x0..x1 {
                m = m.min(dark[row + xx]);
            }
            eroded_h[row + x] = m;
        }
    }
    let mut eroded = vec![0u16; n];
    for y in 0..height {
        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius + 1).min(height);
        for x in 0..width {
            let mut m = u16::MAX;
            for yy in y0..y1 {
                m = m.min(eroded_h[yy * width + x]);
            }
            eroded[y * width + x] = m;
        }
    }

    let k_cap = (n / 1000).clamp(100, 1000).min(n);
    // Min-heap of `(dark_val, Reverse(y_val), idx)` so root is the weakest candidate.
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut heap: BinaryHeap<Reverse<(u16, Reverse<u16>, usize)>> = BinaryHeap::with_capacity(k_cap + 1);
    for idx in 0..n {
        let d = eroded[idx];
        let yv = y_approx[idx];
        if d == 0 || yv > 0x4000 {
            continue;
        }
        let cand = Reverse((d, Reverse(yv), idx));
        if heap.len() < k_cap {
            heap.push(cand);
        } else if let Some(top) = heap.peek() {
            if cand.0 > top.0 {
                heap.pop();
                heap.push(cand);
            }
        }
    }
    if heap.is_empty() {
        return ([0.8, 0.8, 0.8], level_low, level_high);
    }
    let mut sum = [0u64; 3];
    let count = heap.len() as f32;
    for Reverse((_, _, idx)) in heap {
        sum[0] += rgb_u16[idx * 3] as u64;
        sum[1] += rgb_u16[idx * 3 + 1] as u64;
        sum[2] += rgb_u16[idx * 3 + 2] as u64;
    }
    let haze = [
        (sum[0] as f32 / (count * 65535.0)).clamp(0.05, 1.0),
        (sum[1] as f32 / (count * 65535.0)).clamp(0.05, 1.0),
        (sum[2] as f32 / (count * 65535.0)).clamp(0.05, 1.0),
    ];
    (haze, level_low, level_high)
}

/// Capture One `CImgOpClippingRecovery` (`p_040758336.cl`, `z03df67c650`):
/// linear-domain per-channel highlight clipping reconstruction + BT.601
/// quadratic desaturation convergence.
#[inline]
pub fn clipping_recovery_pixel(rgb: [f32; 3], wb_gains: [f32; 3], range: f32, strength_v: f32) -> [f32; 3] {
    if strength_v <= 0.0 || range <= 1e-6 {
        return rgb;
    }
    let t_base = range * (-0.05f32).exp2();
    let t = [wb_gains[0] * t_base, wb_gains[1] * t_base, wb_gains[2] * t_base];
    let s3 = 2.0 * (1.931030154f32 * strength_v - 0.3).exp2();
    let span = [
        2.0 * (range - t_base) * wb_gains[0].max(1e-6),
        2.0 * (range - t_base) * wb_gains[1].max(1e-6),
        2.0 * (range - t_base) * wb_gains[2].max(1e-6),
    ];
    let inv_g = [
        1.0 / wb_gains[0].max(1e-6),
        1.0 / wb_gains[1].max(1e-6),
        1.0 / wb_gains[2].max(1e-6),
    ];
    let scale_m = 1.0 / (24.0 * t_base.max(1e-6));

    let [r, g, b] = rgb;
    let mut out = rgb;
    if r > t[0] && r > 1e-6 {
        let a = (((r - t[0]) * (s3 / t[0].max(1e-6))) * (g.min(b) / r)).min(1.0);
        out[0] = (g + b) * 0.5 * a + r * (1.0 - a);
    }
    if g > t[1] && g > 1e-6 {
        let a = (((g - t[1]) * (s3 / t[1].max(1e-6))) * (r.min(b) / g)).min(1.0);
        out[1] = (r + b) * 0.5 * a + g * (1.0 - a);
    }
    if b > t[2] && b > 1e-6 {
        let a = (((b - t[2]) * (s3 / t[2].max(1e-6))) * (g.min(r) / b)).min(1.0);
        out[2] = (g + r) * 0.5 * a + b * (1.0 - a);
    }

    if out[0] > t[0] || out[1] > t[1] || out[2] > t[2] {
        let mut sum = 0.0f32;
        if out[0] > t[0] {
            sum += (out[0] - t[0]) / span[0];
        }
        if out[1] > t[1] {
            sum += (out[1] - t[1]) / span[1];
        }
        if out[2] > t[2] {
            sum += (out[2] - t[2]) / span[2];
        }
        let m = (out[0] * inv_g[0])
            .min(out[1] * inv_g[1])
            .min(out[2] * inv_g[2])
            * scale_m;
        let weight = ((sum * m) * (sum * m)).min(1.0);
        let y601 = 0.299 * out[0] + 0.587 * out[1] + 0.114 * out[2];
        out[0] = out[0] * (1.0 - weight) + y601 * weight;
        out[1] = out[1] * (1.0 - weight) + y601 * weight;
        out[2] = out[2] * (1.0 - weight) + y601 * weight;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clipping_recovery_pulls_single_clipped_channel_toward_neighbors() {
        let out = clipping_recovery_pixel([1.2, 0.7, 0.6], [1.0, 1.0, 1.0], 1.0, 0.8);
        assert!(out[0] < 1.2 && out[0] > 0.65, "{out:?}");
        assert!((out[1] - 0.7).abs() < 1e-5);
    }

    #[test]
    fn estimate_dehaze_auto_extracts_levels_and_haze() {
        let w = 32usize;
        let h = 32usize;
        let mut buf = Vec::with_capacity(w * h * 3);
        for i in 0..(w * h) {
            let v = if i % 10 == 0 { 12000 } else { 28000 };
            buf.extend_from_slice(&[v, v + 500, v + 1000]);
        }
        let (haze, lo, hi) = estimate_dehaze_auto(&buf, w, h);
        assert!(haze[2] >= haze[0] && haze[0] > 0.1, "{haze:?}");
        assert!(lo[0] < hi[0], "{lo:?} vs {hi:?}");
    }
}
