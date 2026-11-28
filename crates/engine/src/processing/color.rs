//! Pure color and tone primitives lifted out of the renderer.
//!
//! Transfer curves (`srgb_to_linear` / `linear_to_srgb` and the cached
//! [`GammaLut`]), white balance, luma weights, smooth-step, vignetting falloff,
//! hue/HSV helpers, the curve-LUT lookup, and the pure color-editor coverage and
//! shift helpers (`color_balance_zone_weights`, `color_range_weight`,
//! `band_weight`, `apply_homogeneity`, `color_editor`). They are stateless
//! building blocks that depend only on adjustment value types, never on source
//! pixels, plans, caches, GPU code or portrait producers.
//!
//! This is a mechanical move: the bodies, constants and arithmetic order are
//! unchanged from the renderer that used to define them.

use std::sync::OnceLock;

use crate::adjust::{
    AdjustmentSet, ColorBand, ColorEditor, ColorRange, ColorWheel, Curve, HslShift, VignetteCenter,
    VignetteMethod,
};

pub(crate) fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

pub(crate) fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// Linear → sRGB over `0 ..= 4` linear, so highlight recovery has headroom to pull from.
pub(crate) const GAMMA_LUT_SIZE: usize = 8192;
pub(crate) struct GammaLut(pub(crate) &'static [f32; GAMMA_LUT_SIZE]);

impl GammaLut {
    pub(crate) const SIZE: usize = GAMMA_LUT_SIZE;
    pub(crate) const RANGE: f32 = 4.0;

    pub(crate) fn new() -> Self {
        static TABLE: OnceLock<[f32; GammaLut::SIZE]> = OnceLock::new();
        Self(TABLE.get_or_init(|| {
            std::array::from_fn(|i| {
                linear_to_srgb(i as f32 / (Self::SIZE - 1) as f32 * Self::RANGE)
            })
        }))
    }

    pub(crate) fn get(&self, v: f32) -> f32 {
        let i = (v.max(0.0) / Self::RANGE * (Self::SIZE - 1) as f32) as usize;
        self.0[i.min(Self::SIZE - 1)]
    }
}

/// Catmull-Rom blend of `b` and `c` at `t`, steering with the outer taps.
pub(crate) fn catmull_rom(a: f32, b: f32, c: f32, d: f32, t: f32) -> f32 {
    let (t2, t3) = (t * t, t * t * t);
    0.5 * (2.0 * b + (c - a) * t + (2.0 * a - 5.0 * b + 4.0 * c - d) * t2 + (3.0 * b - 3.0 * c + d - a) * t3)
}

/// Approximate white point of a black body at `kelvin`, normalized to green.
fn kelvin_rgb(kelvin: f32) -> [f32; 3] {
    let t = kelvin.clamp(1000.0, 40000.0) / 100.0;
    let r = if t <= 66.0 { 255.0 } else { 329.698_73 * (t - 60.0).powf(-0.133_204_76) };
    let g = if t <= 66.0 { 99.470_8 * t.ln() - 161.119_57 } else { 288.122_16 * (t - 60.0).powf(-0.075_514_85) };
    let b = if t >= 66.0 {
        255.0
    } else if t <= 19.0 {
        0.0
    } else {
        138.517_73 * (t - 10.0).ln() - 305.044_8
    };
    let (r, g, b) = (r.clamp(1.0, 255.0), g.clamp(1.0, 255.0), b.clamp(1.0, 255.0));
    [srgb_to_linear(r / 255.0) / srgb_to_linear(g / 255.0), 1.0, srgb_to_linear(b / 255.0) / srgb_to_linear(g / 255.0)]
}

pub(crate) fn white_balance_gains(kelvin: f32, tint: f32) -> [f32; 3] {
    let shot = kelvin_rgb(5600.0);
    let set = kelvin_rgb(kelvin);
    let tint_gain = 1.0 - tint / 100.0 * 0.6;
    [shot[0] / set[0], tint_gain, shot[2] / set[2]]
}

/// Pixel gains: exact stored C1 gains when the import had them, otherwise the
/// Kelvin/Tint approximation. K/T is display-only once exact gains exist.
pub(crate) fn render_wb_gains(w: &crate::WhiteBalance) -> [f32; 3] {
    w.gains
        .filter(|g| g.iter().all(|v| v.is_finite() && *v > 0.0))
        .unwrap_or_else(|| white_balance_gains(w.kelvin, w.tint))
}

pub(crate) fn luma(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

/// Rec.601 luma, the weight Capture One's `ZGRADATIONCURVEY` stage uses.
pub(crate) fn luma601(c: [f32; 3]) -> f32 {
    0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
}

pub(crate) fn sstep(a: f32, b: f32, x: f32) -> f32 {
    if b <= a {
        return if x < a { 0.0 } else { 1.0 };
    }
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `CImgOpLensLight`'s manual branch, from the OpenCL kernel
/// `z181b8373ba` (`LensLightRadial.cl` inside `ImgCoreDll.dll`).
///
/// The kernel computes `r = length(scale * (tile + pixel) - centre)`, looks the
/// EV up in a 16-knot spline and returns `exp(-ln2 * ev)`. Measured against
/// Capture One's own renderer, that spline tracks a plain quadratic in the
/// normalised radius to within 0.06 EV of a 2 EV slider:
///
/// ```text
/// elliptic : ev = amount * 2 * ((x-1/2)^2 + (y-1/2)^2)
/// circular : ev = amount * 4 * ((x-1/2)^2*a^2 + (y-1/2)^2) / (a^2 + 1)
/// ```
///
/// with `a` the frame aspect. Both put `amount` EV at the frame corner and
/// nothing at the centre, which is what the slider means. `x` and `y` are
/// normalised to the rendered frame, so the metric follows the crop.
///
/// The sign is Capture One's: `IC_Vignetting.lightAmount` is a *lightening*
/// amount, so +2 EV takes the corner from 6070 to 13107 in that renderer and
/// -2 EV down to 2824.
pub(crate) fn vignette_gain(
    x: f32,
    y: f32,
    method: VignetteMethod,
    amount: f32,
    aspect: f32,
    center: [f32; 2],
) -> f32 {
    if amount == 0.0 || !amount.is_finite() {
        return 1.0;
    }
    let (dx, dy) = (x - center[0], y - center[1]);
    let a2 = if aspect.is_finite() && aspect > 0.0 { aspect * aspect } else { 1.0 };
    let rho2 = match method {
        // Corner-normalised ellipse: the corner sits at 1.0.
        VignetteMethod::EllipticOnCrop => 2.0 * (dx * dx + dy * dy),
        // Circle in pixel space, scaled so the frame corner is again 1.0.
        VignetteMethod::Circular | VignetteMethod::CircularCrop => {
            4.0 * (dx * dx * a2 + dy * dy) / (a2 + 1.0)
        }
    };
    // Measured against the native renderer: the slider is capped at 3 EV in
    // both directions, so a custom centre far from the frame corner cannot
    // push the falloff past that.
    (amount * rho2).clamp(-3.0, 3.0).exp2()
}

/// Resolves `IC_Vignetting.center` into normalised frame coordinates.
///
/// `cam` and `crop` both land on the rendered frame's centre (the sensor centre
/// is the canvas centre until the frame is cropped). `lens` uses
/// `ZLENSOPTICCENTER`, stored in canvas pixels, and falls back to the centre
/// when the lens database had no measurement. `custom` is `IC_Vignetting.x/y`,
/// also canvas pixels. Every result is clamped to the frame, which is what the
/// native renderer does with an out-of-range centre.
pub(crate) fn vignette_center(adj: &AdjustmentSet, size: [f32; 2]) -> [f32; 2] {
    let (w, h) = (size[0].max(1.0), size[1].max(1.0));
    let center = match adj.vignetting.center {
        VignetteCenter::Cam | VignetteCenter::Crop => [0.5, 0.5],
        VignetteCenter::Lens => adj
            .lens
            .optic_center
            .map(|p| [p[0] / w, p[1] / h])
            .unwrap_or([0.5, 0.5]),
        VignetteCenter::Custom => [adj.vignetting.x / w, adj.vignetting.y / h],
    };
    center.map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.5 })
}

pub(crate) fn hue_rgb(hue: f32) -> [f32; 3] {
    let h = (hue.rem_euclid(360.0)) / 60.0;
    let x = 1.0 - (h % 2.0 - 1.0).abs();
    match h as u32 {
        0 => [1.0, x, 0.0],
        1 => [x, 1.0, 0.0],
        2 => [0.0, 1.0, x],
        3 => [0.0, x, 1.0],
        4 => [x, 0.0, 1.0],
        _ => [1.0, 0.0, x],
    }
}

pub(crate) fn rgb_to_hsv(c: [f32; 3]) -> (f32, f32, f32) {
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

pub(crate) fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let base = hue_rgb(h);
    [v * (1.0 - s + s * base[0]), v * (1.0 - s + s * base[1]), v * (1.0 - s + s * base[2])]
}

pub(crate) fn hue_distance(a: f32, b: f32) -> f32 {
    let d = (a - b).rem_euclid(360.0);
    d.min(360.0 - d)
}

/// Offset a color wheel adds, centered on neutral.
pub(crate) fn wheel_offset(w: &ColorWheel, strength: f32) -> [f32; 3] {
    let c = hue_rgb(w.hue);
    let mean = (c[0] + c[1] + c[2]) / 3.0;
    let s = w.saturation * strength;
    let l = w.lightness * 0.25;
    [(c[0] - mean) * s + l, (c[1] - mean) * s + l, (c[2] - mean) * s + l]
}

pub(crate) fn contrast(g: f32, amount: f32) -> f32 {
    if amount.abs() < 1e-4 {
        return g;
    }
    let k = 1.0 + amount;
    if k >= 1.0 {
        // Sigmoid-ish S curve that does not clip hard.
        let x = (g - 0.5) * k;
        0.5 + x / (1.0 + (x.abs() * (k - 1.0) * 0.8))
    } else {
        0.5 + (g - 0.5) * k
    }
}

pub(crate) fn lut_of(c: &Curve) -> Option<[f32; 256]> {
    (!c.is_identity()).then(|| c.lut())
}

pub(crate) fn lut_get(lut: &[f32; 256], v: f32) -> f32 {
    let x = v.clamp(0.0, 1.0) * 255.0;
    let i = (x as usize).min(254);
    let t = x - i as f32;
    lut[i] + (lut[i + 1] - lut[i]) * t
}

pub(crate) fn color_range_weight(r: &ColorRange, c: [f32; 3]) -> f32 {
    let (h, s, _) = rgb_to_hsv(c);
    let half = r.width / 2.0;
    let d = hue_distance(h, r.hue);
    let hue_w = 1.0 - sstep(half, half + r.smoothness.max(0.1), d);
    hue_w * sstep(r.min_saturation * 0.5, r.min_saturation + 0.05, s)
}

/// `wH = L²·protect`, `wM = (4L(1-L))³/2`, `wS = (1-L)²/2` at maximum channel `l`.
pub(crate) fn color_balance_zone_weights(l: f32) -> [f32; 3] {
    // shortcut: the ramp end `.sf` is unresolved; SDR pixels (`l <= 1`) use protect = 1.
    let protect = if l <= 1.0 { 1.0 } else { (2.0 - l).clamp(0.0, 1.0) };
    let bell = 4.0 * l * (1.0 - l);
    [l * l * protect, bell * bell * bell * 0.5, (1.0 - l) * (1.0 - l) * 0.5]
}

pub(crate) fn signed_hue_delta(h: f32, center: f32) -> f32 {
    let d = (h - center).rem_euclid(360.0);
    if d > 180.0 {
        d - 360.0
    } else {
        d
    }
}

/// Coverage of one color-editor band for a pixel at hue `h`, saturation `s`.
/// C1 `HueBandWeight` * `ChromaBandWeight` (no Lightness/Value selection band).
pub(crate) fn band_weight(b: &ColorBand, extra_feather: f32, h: f32, s: f32, _v: f32) -> f32 {
    let d = signed_hue_delta(h, b.center_hue);
    let feather = b.smoothness.max(0.1) + extra_feather;
    let hue_w = if d < b.hue_low {
        sstep(b.hue_low - feather, b.hue_low, d)
    } else if d > b.hue_high {
        1.0 - sstep(b.hue_high, b.hue_high + feather, d)
    } else {
        1.0
    };
    let ds = (s - b.center_saturation) * 100.0;
    let sat_feather = 10.0f32.max(b.smoothness);
    let sat_w = if ds < b.sat_low {
        sstep(b.sat_low - sat_feather, b.sat_low, ds)
    } else if ds > b.sat_high {
        1.0 - sstep(b.sat_high, b.sat_high + sat_feather, ds)
    } else {
        1.0
    };
    (hue_w * sat_w).clamp(0.0, 1.0)
}

/// C1 nonlinear saturation boost `SatBoost(C*, amount)` (`ImgCoreDll!0x132bb70`).
#[inline]
pub(crate) fn c1_sat_boost(c_star: f32, amount: f32) -> f32 {
    let t = 0.01 * amount;
    let c2 = c_star * c_star;
    let c3 = c2 * c_star;
    let p = 0.0092299 * c_star - 3.1379e-5 * c2 + 5.3746e-8 * c3;
    let r = ((-3.8215 * t).exp() - 1.0) / ((-3.8215f32).exp() - 1.0);
    100.0 * (t * (1.0 - p) + p * r)
}

/// C1 asymmetric quadratic damping for skin-tone lightness uniformity (`ImgCoreDll!0x132b295`).
#[inline]
fn skin_luma_uniform_step(l_cur: f32, l_target: f32, uniformity: f32) -> f32 {
    let lt = l_target.clamp(0.001, 99.999);
    let q = if l_cur < lt {
        l_cur / lt
    } else {
        let k = 4.0 * (0.5 * (4.0625f32.sqrt() + 0.25) - 1.0);
        1.0 - k * (l_cur - lt) / (100.0 - lt)
    };
    let factor = (1.0 - 0.25 * q).powi(2);
    let l_uniform = lt + factor * (l_cur - lt);
    (0.01 * uniformity) * (l_uniform - l_cur)
}

pub(crate) fn apply_homogeneity(b: &ColorBand, w: f32, h: &mut f32, s: &mut f32, v: &mut f32) {
    let hue = b.homogeneity[0] / 100.0 * w;
    if hue != 0.0 {
        *h += signed_hue_delta(b.center_hue, *h) * hue;
    }
    let sat = b.homogeneity[1] / 100.0 * w;
    if sat != 0.0 {
        *s += (b.center_saturation - *s) * sat;
    }
    if b.homogeneity[2] != 0.0 {
        let dl = skin_luma_uniform_step(*v * 100.0, b.center_value * 100.0, b.homogeneity[2]) * 0.01 * w;
        *v = (*v + dl).clamp(0.0, 1.0);
    }
}

pub(crate) fn color_editor(g: [f32; 3], ce: &ColorEditor, k: f32) -> [f32; 3] {
    let (h0, s0, v0) = rgb_to_hsv([g[0].max(0.0), g[1].max(0.0), g[2].max(0.0)]);
    if s0 < 0.02 {
        return g;
    }
    let extra = ce.global.smoothness.max(0.0) * 0.5;
    let mut dh = 0.0f32;
    let mut ds = 0.0f32;
    let mut dl = 0.0f32;
    let mut w_sum = 0.0f32;

    // Only terms that move the pixel count towards `w_sum`: an enabled band
    // with a zero shift (the default global group) must not dilute the others.
    if ce.global.enabled && ce.global_shift != HslShift::default() {
        let mut w = band_weight(&ce.global, 0.0, h0, s0, v0);
        if ce.global.inverted {
            w = 1.0 - w;
        }
        w_sum += w;
        dh += ce.global_shift.hue * w;
        ds += ce.global_shift.saturation * w;
        dl += ce.global_shift.lightness * w;
    }

    for r in &ce.skin {
        if let Some((_, band)) = ce.advanced_bands.iter().find(|(id, _)| *id == r.id) {
            if !band.enabled {
                continue;
            }
            if band.homogeneity == [0.0; 3] {
                continue;
            }
            let mut w = band_weight(band, 0.0, h0, s0, v0);
            if band.inverted {
                w = 1.0 - w;
            }
            w_sum += w;
            if band.homogeneity[0] != 0.0 {
                dh += signed_hue_delta(band.center_hue, h0) * (band.homogeneity[0] / 100.0 * w);
            }
            if band.homogeneity[1] != 0.0 {
                ds += (band.center_saturation - s0) * band.homogeneity[1] * w;
            }
            if band.homogeneity[2] != 0.0 {
                dl += skin_luma_uniform_step(v0 * 100.0, band.center_value * 100.0, band.homogeneity[2]) * w;
            }
        } else if let Some(u) = r.uniformity.as_ref().filter(|u| u.hue != 0.0 || u.saturation > 0.0 || u.lightness > 0.0) {
            let w = color_range_weight(r, g);
            w_sum += w;
            dh += signed_hue_delta(r.hue, h0) * (u.hue / 100.0 * w);
            if u.saturation > 0.0 {
                let target_s = (r.min_saturation + 0.35).clamp(0.15, 0.55);
                ds += (target_s - s0) * u.saturation * w;
            }
            if u.lightness > 0.0 {
                dl += skin_luma_uniform_step(v0 * 100.0, 65.0, u.lightness) * w;
            }
        }
    }

    for (i, shift) in ce.basic.iter().enumerate() {
        if *shift == HslShift::default() {
            continue;
        }
        let band = &ce.bands[i];
        if !band.enabled {
            continue;
        }
        let mut w = band_weight(band, extra, h0, s0, v0);
        if band.inverted {
            w = 1.0 - w;
        }
        w_sum += w;
        dh += shift.hue * w;
        ds += shift.saturation * w;
        dl += shift.lightness * w;
    }

    for (r, extra_feather) in ce
        .advanced
        .iter()
        .map(|r| (r, 0.0))
        .chain(ce.skin.iter().map(|r| (r, 0.0)))
        .chain(ce.max_smooth.iter().map(|r| (r, extra)))
        .filter(|(r, _)| r.shift != HslShift::default())
    {
        let w = match ce.advanced_bands.iter().find(|(id, _)| *id == r.id) {
            Some((_, band)) if band.enabled => {
                let w = band_weight(band, extra_feather, h0, s0, v0);
                if band.inverted { 1.0 - w } else { w }
            }
            Some(_) => 0.0,
            None => color_range_weight(r, g),
        };
        w_sum += w;
        dh += r.shift.hue * w;
        ds += r.shift.saturation * w;
        dl += r.shift.lightness * w;
    }

    if w_sum > 1.0 {
        let inv = 1.0 / w_sum;
        dh *= inv;
        ds *= inv;
        dl *= inv;
    }
    let dh = dh.clamp(-180.0, 180.0);
    let ds = ds.clamp(-100.0, 100.0);
    let dl = dl.clamp(-100.0, 100.0);
    let s2 = (s0 * (1.0 + ds / 100.0 * k)).clamp(0.0, 1.0);
    let v2 = (v0 * (1.0 + dl / 100.0 * 0.5 * s0 * k)).max(0.0);
    hsv_to_rgb(h0 + dh * k, s2, v2)
}
