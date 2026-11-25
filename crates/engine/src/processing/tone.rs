//! Tone operators: local tone mapping (HDR), brightness/contrast, and dehaze.
//!
//! These are pure per-pixel functions that depend only on adjustment value
//! types ([`HighDynamicRange`], [`Exposure`], [`Clarity`], [`Dehaze`]). They
//! were moved out of `mock::process` byte-identically so the tone stage can be
//! shared without pulling in `Source`, `Plan`, caches, or any GPU path. The
//! mock renderer imports them back and re-exports nothing, so its call sites
//! and tests keep the unqualified names.

use crate::{Clarity, ClarityMethod, Dehaze, Exposure, HighDynamicRange};

use super::color::{contrast, luma, sstep};

/// The hard toe of `CImgOpLocalToneMappingV8`: kernel `z2714f67010` in the
/// OpenCL blob at ImgCoreDll file offset 41,541,440 (`z3dbd1ff682 == 13`).
/// `k` is the op's curve strength; the kernel applies it to `0.9`.
fn hdr_toe(x: f32, k: f32) -> f32 {
    let u = x * (2.0 * 0.9);
    let v = u.abs() + 1.0;
    let r = (u * v) / (v + u * u) * 0.5 + 0.5;
    let r = r.max(u - r + 1.0) * (1.0 / (2.0 * k));
    let s = r * r;
    let num = (1.0 + r + s) * (r + s * s);
    num / (1.0 + num) * k
}

/// `z386cc3479b` — the mirrored toe the shadow and white branches use.
fn hdr_inv_toe(x: f32, k: f32) -> f32 {
    hdr_toe(-x, k)
}

/// Curve strengths `op+0x8c` / `op+0x90`. The kernel scales, it does not
/// change the shape's sign, so a placeholder of 1 keeps every branch monotone.
const HDR_K_SHADOW: f32 = 1.0;
const HDR_K_HIGHLIGHT: f32 = 1.0;

/// `CImgOpLocalToneMappingV8`: one gain per pixel from a single luma plane,
/// then `out = rgb * gain`, so the panel is channel-neutral.
///
/// Highlight and shadow are kernel-exact. Whites/blacks still use the old
/// zone-window approximation: their branch needs `op+0x94` (white curve) and
/// `op+0xb4` (white offset), which are not recovered yet.
///
/// shortcut: whites/blacks stay approximate until those two fields are read
/// from a C1 export; see `SESSION-c1-parse-hdr-highlights.md`.
pub(crate) fn apply_hdr(mut g: [f32; 3], hdr: &HighDynamicRange) -> [f32; 3] {
    let x = luma(g);
    let mut shift = 0.0;
    if hdr.whites != 0.0 {
        shift += hdr.whites / 100.0 * 0.2 * sstep(0.65, 1.0, x);
    }
    if hdr.blacks != 0.0 {
        shift += hdr.blacks / 100.0 * 0.12 * (1.0 - sstep(0.0, 0.25, x));
    }
    if shift.abs() > 1e-5 {
        let k = (x + shift).max(0.0) / x.max(1e-4);
        g = [g[0] * k, g[1] * k, g[2] * k];
    }
    if hdr.highlights == 0.0 && hdr.shadows == 0.0 {
        return g;
    }
    let x = luma(g);
    let m = (hdr.highlights / 100.0).clamp(-1.0, 1.0);
    let b = if m != 0.0 {
        let t = hdr_toe(x, HDR_K_HIGHLIGHT);
        2f32.powf(if m < 0.0 { -t } else { t }) * m.abs() + (1.0 - m.abs())
    } else {
        1.0
    };
    let m = (hdr.shadows / 100.0).clamp(-1.0, 1.0);
    let a = if m != 0.0 {
        let t = hdr_inv_toe(x, HDR_K_SHADOW);
        2f32.powf(if m < 0.0 { -t } else { t }) * m.abs() + (1.0 - m.abs())
    } else {
        1.0
    };
    let gain = a / b;
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

/// The `CImgOpDehaze2020` kernel (`__kernel void zec60c625ee` in the OpenCL
/// source embedded at `ImgCoreDll!0x277a0d8`). `ZDEHAZECOLOR` is the haze
/// colour `C` (0..1); the host packs `A = 65535 * W * C`.
///
/// The knee radius is `Ah = C * min(1, 2*omega)` — the *colour*, not the
/// 16-bit `A`. With `C` around 0.6 the `soft()` knee is inert for normal
/// pixels; using `A` there shifts every pixel down by the whole atmospheric
/// light and crushes the frame.
///
/// `level_low/high` are the per-channel black/white points (auto levels,
/// 0/1 by default). `omega` is `DehazeAmount * 0.01`.
pub(crate) fn apply_dehaze(g: [f32; 3], d: &Dehaze) -> [f32; 3] {
    let omega = d.amount / 100.0;
    if omega <= 0.0 || !omega.is_finite() {
        return g;
    }
    let c = d.shadow_tone.map(|v| v.max(0.0));
    let a = c.map(|v| v * 65535.0);
    if a.iter().all(|v| *v <= f32::EPSILON) {
        return g;
    }
    // Auto levels normalise the plane before the atmospheric-light estimate.
    let level: [(bool, f32, f32); 3] = std::array::from_fn(|c| {
        let (lo, hi) = (d.level_low[c], d.level_high[c]);
        (hi - lo > 1e-3, lo, hi - lo)
    });
    let i: [f32; 3] = std::array::from_fn(|c| {
        if level[c].0 {
            (g[c] - level[c].1) / level[c].2 * 65535.0
        } else {
            g[c] * 65535.0
        }
    });
    let eps = 1e-6;
    let delta = (2.0 * omega).min(1.0);
    let ah = c.map(|v| v * delta);
    let mut i1 = [0.0f32; 3];
    let mut a1 = [0.0f32; 3];
    for c in 0..3 {
        i1[c] = dehaze_soft(i[c] - ah[c], ah[c]);
        a1[c] = dehaze_soft(a[c] - ah[c], ah[c]);
    }
    let mut recovered = [0.0f32; 3];
    let li = luma(i1);
    let la = luma(a1);
    let keep = la / (li + la).max(eps);
    for c in 0..3 {
        recovered[c] = (1.0 - keep) * i1[c];
        i1[c] *= keep;
    }
    let m_raw = (i1[0] / a1[0].max(eps))
        .min(i1[1] / a1[1].max(eps))
        .min(i1[2] / a1[2].max(eps));
    let m0 = m_raw.clamp(0.0, 1.0);
    let mag = (i1[0] * i1[0] + i1[1] * i1[1] + i1[2] * i1[2]).sqrt();
    let dvec = a1.map(|v| m0 * v);
    let mag_d =
        ((i1[0] - dvec[0]).powi(2) + (i1[1] - dvec[1]).powi(2) + (i1[2] - dvec[2]).powi(2))
            .sqrt();
    let ratio = (mag + eps).max(eps);
    let val = (ratio * 0.375 - (mag_d + eps) * (4.0 / 7.0)) / ratio;
    let t = 1.0 / (1.0 + ((0.5 - val) * 4.0).exp());
    let m = m0 * t;
    for c in 0..3 {
        recovered[c] += (1.0 - t) * i1[c];
        i1[c] *= t;
    }
    let t0 = (1.0 - omega * m).max(0.05);
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        out[c] = (i1[c] - a1[c]) / t0 + a1[c] + recovered[c];
    }
    // Saturation guard: never add more chroma than `sqrt(1+(0.5+2m)²)` allows.
    let max_i = i1[0].max(i1[1]).max(i1[2]);
    if max_i > eps {
        let sat_i = (max_i - i1[0].min(i1[1]).min(i1[2])) / max_i;
        // The kernel anchors the guard on the *unclamped* ratio.
        let limit = sat_i * (1.0 + (0.5 + 2.0 * m_raw).powi(2)).sqrt();
        let max_o = out[0].max(out[1]).max(out[2]);
        if max_o > eps {
            let sat_o = (max_o - out[0].min(out[1]).min(out[2])) / max_o;
            if sat_o > limit {
                let k = limit / sat_o;
                let l = luma(out);
                out = [k * out[0] + (1.0 - k) * l, k * out[1] + (1.0 - k) * l, k * out[2] + (1.0 - k) * l];
            }
        }
    }
    std::array::from_fn(|c| {
        if level[c].0 {
            (level[c].1 + out[c] * level[c].2 / 65535.0).max(0.0)
        } else {
            (out[c] / 65535.0).max(0.0)
        }
    })
}
