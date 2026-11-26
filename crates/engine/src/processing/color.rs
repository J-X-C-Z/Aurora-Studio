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
    AdjustmentSet, BlackWhite, ColorBalance, ColorBand, ColorEditor, ColorRange, ColorWheel, Curve,
    HslShift, VignetteCenter, VignetteMethod,
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

/// Coverage of one color-editor band for a pixel at hue `h`, saturation `s`,
/// value `v`. `extra_feather` widens every band edge (the global group's
/// smoothness).
pub(crate) fn band_weight(b: &ColorBand, extra_feather: f32, h: f32, s: f32, v: f32) -> f32 {
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
    // Fixed 0.05 V feather: C1 has no value band, so there is no stored unit to borrow.
    let v = v.clamp(0.0, 1.0);
    let value_w = if v < b.value_low {
        sstep(b.value_low - 0.05, b.value_low, v)
    } else if v > b.value_high {
        1.0 - sstep(b.value_high, b.value_high + 0.05, v)
    } else {
        1.0
    };
    (hue_w * sat_w * value_w).clamp(0.0, 1.0)
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
    let light = b.homogeneity[2] / 100.0 * w;
    if light != 0.0 {
        *v += (b.center_value - *v) * light;
    }
}

pub(crate) fn color_editor(g: [f32; 3], ce: &ColorEditor, k: f32) -> [f32; 3] {
    let (h0, s0, v0) = rgb_to_hsv([g[0].max(0.0), g[1].max(0.0), g[2].max(0.0)]);
    if s0 < 0.02 {
        return g;
    }
    let extra = ce.global.smoothness.max(0.0) * 0.5;
    let mut h = h0;
    let mut s = s0;
    let mut v = v0;
    let mut dh = 0.0;
    let mut ds = 0.0;
    let mut dl = 0.0;

    if ce.global.enabled {
        let mut w = band_weight(&ce.global, 0.0, h0, s0, v0);
        if ce.global.inverted {
            w = 1.0 - w;
        }
        dh += ce.global_shift.hue * w;
        ds += ce.global_shift.saturation * w;
        dl += ce.global_shift.lightness * w;
    }

    // Homogeneity is a skin term only. C1's `pie_skin` branch pulls toward the
    // selected Lab chroma/L*; the basic, advanced and max-smooth branches never
    // read their homogeneity fields.
    for r in &ce.skin {
        if let Some((_, band)) = ce.advanced_bands.iter().find(|(id, _)| *id == r.id) {
            if !band.enabled {
                continue;
            }
            let mut w = band_weight(band, 0.0, h0, s0, v0);
            if band.inverted {
                w = 1.0 - w;
            }
            apply_homogeneity(band, w, &mut h, &mut s, &mut v);
        } else if let Some(u) = &r.uniformity {
            let w = color_range_weight(r, g);
            h += signed_hue_delta(r.hue, h) * (u.hue / 100.0 * w);
            if u.saturation > 0.0 {
                let target_s = (r.min_saturation + 0.35).clamp(0.15, 0.55);
                s += (target_s - s) * (u.saturation / 100.0 * w);
            }
            if u.lightness > 0.0 {
                let target_v = 0.65;
                v += (target_v - v) * (u.lightness / 100.0 * w * 0.5);
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
        let mut w = band_weight(band, extra, h, s, v);
        if band.inverted {
            w = 1.0 - w;
        }
        dh += shift.hue * w;
        ds += shift.saturation * w;
        dl += shift.lightness * w;
    }

    // MaxSmooth (`kind=3`) runs C1's basic `pie_smooth` window, hence `extra`.
    for (r, extra_feather) in ce
        .advanced
        .iter()
        .map(|r| (r, 0.0))
        .chain(ce.skin.iter().map(|r| (r, 0.0)))
        .chain(ce.max_smooth.iter().map(|r| (r, extra)))
    {
        let w = match ce.advanced_bands.iter().find(|(id, _)| *id == r.id) {
            Some((_, band)) if band.enabled => {
                let w = band_weight(band, extra_feather, h0, s0, v0);
                if band.inverted { 1.0 - w } else { w }
            }
            Some(_) => 0.0,
            None => color_range_weight(r, g),
        };
        dh += r.shift.hue * w;
        ds += r.shift.saturation * w;
        dl += r.shift.lightness * w;
    }
    let dh = dh.clamp(-180.0, 180.0);
    let ds = ds.clamp(-100.0, 100.0);
    let dl = dl.clamp(-100.0, 100.0);
    let s2 = (s * (1.0 + ds / 100.0 * k)).clamp(0.0, 1.0);
    let v2 = (v * (1.0 + dl / 100.0 * 0.5 * s * k)).max(0.0);
    hsv_to_rgb(h + dh * k, s2, v2)
}

// ── Black & white / colour-balance stages (moved from the renderer) ──

pub(crate) fn color_balance(g: [f32; 3], cb: &ColorBalance, k: f32) -> [f32; 3] {
    let l = luma(g).clamp(0.0, 1.0);
    let ws = (1.0 - l).powi(2);
    let wh = l.powi(2);
    let wm = (1.0 - ws - wh).max(0.0);
    let mut out = g;
    for (wheel, w) in [(&cb.master, 1.0), (&cb.shadow, ws), (&cb.midtone, wm), (&cb.highlight, wh)] {
        if wheel.saturation == 0.0 && wheel.lightness == 0.0 {
            continue;
        }
        let o = wheel_offset(wheel, 0.35);
        for i in 0..3 {
            out[i] += o[i] * w * k;
        }
    }
    out
}

pub(crate) fn black_white(g: [f32; 3], bw: &BlackWhite) -> [f32; 3] {
    // Zero sliders keep the exact display luma and therefore the old buckets.
    let y = if bw.sensitivity.iter().all(|&v| v == 0.0) {
        luma(g)
    } else {
        bw_sector_y(g, &bw.sensitivity)
    }
    .max(0.0);
    if bw.highlight_tone.saturation <= 0.0 && bw.shadow_tone.saturation <= 0.0 {
        return [y, y, y];
    }
    bw_split_tone(y, bw)
}

/// The tonemap kernel's own luma basis, Q10 `0.3328 / 0.6121 / 0.0551`.
const BW_C1_LUMA: [f32; 3] = [0.3328, 0.6121, 0.0551];
const BW_C1_LUMA_Q10: [i32; 3] = [340, 626, 56];

/// `CImgOpBlackWhiteV8_splittone`, host prep at `0x1816B0B40`: the tint is
/// `0.1 * pow(mix(1, hue_ramp, saturation), 2.2)` per channel. Hue is degrees,
/// saturation `0 ..= 100`; channel offsets are `1, 2/3, 1/3` (R, G, B).
fn bw_tint(hue: f32, saturation: f32) -> [f32; 3] {
    let s = (saturation / 100.0).clamp(0.0, 1.0);
    [1.0f32, 2.0 / 3.0, 1.0 / 3.0].map(|off| {
        let x = (hue / 360.0 + off).fract();
        let ramp = ((x * 6.0 - 3.0).abs() - 1.0).clamp(0.0, 1.0);
        0.1 * ((1.0 - s) + s * ramp).powf(2.2)
    })
}

/// The main kernel (`p_041248720.cl`): `dark_hi` is the light tint's luma,
/// `light_lo` the dark tint's max channel; the middle band blends the two
/// linear ramps. The 65536-entry LUT is identity in this build.
pub(crate) fn bw_split_tone(y: f32, bw: &BlackWhite) -> [f32; 3] {
    let light = bw_tint(bw.highlight_tone.hue, bw.highlight_tone.saturation);
    let dark = bw_tint(bw.shadow_tone.hue, bw.shadow_tone.saturation);
    let dark_hi = light[0] * BW_C1_LUMA[0] + light[1] * BW_C1_LUMA[1] + light[2] * BW_C1_LUMA[2];
    let light_lo = dark[0].max(dark[1]).max(dark[2]);
    if light_lo <= dark_hi {
        return [y, y, y];
    }
    let dark_ramp = dark.map(|c| c * y / light_lo);
    let light_ramp = light.map(|c| c * y / (light_lo - dark_hi));
    let out = if y < dark_hi {
        dark_ramp
    } else if y > light_lo {
        light_ramp
    } else {
        let t = (y - dark_hi) / (light_lo - dark_hi);
        [0, 1, 2].map(|i| dark_ramp[i] + (light_ramp[i] - dark_ramp[i]) * t)
    };
    out.map(|v| v.clamp(0.0, 1.0))
}

/// Q10 filter weights from the host builder at `0x1816B1FC0`. Slider order is
/// `[red, yellow, green, cyan, blue, magenta]`; each unit is `0.01` of full
/// scale, i.e. `10.24` Q10 counts. The two tables are complementary at zero:
/// `sum = 1024`, based on the tonemap's own luma basis (not Rec.601).
///
/// The min-side slot is the channel that is *darkest* in that colour, so the
/// secondaries pair with their missing channel: cyan owns min R, magenta min
/// G, yellow min B. Assigning yellow to min G (or magenta to min B) instead
/// makes the matching slider inert on its own colour.
pub(crate) fn bw_weights(sensitivity: &[f32; 6]) -> ([i32; 3], [i32; 3]) {
    const BASE: [f64; 3] = [340.7872, 626.7904, 56.4224];
    let luma = [
        BASE[0] + 10.24 * sensitivity[0] as f64,
        BASE[1] + 10.24 * sensitivity[2] as f64,
        BASE[2] + 10.24 * sensitivity[4] as f64,
    ];
    let comp = [
        (1024.0 - BASE[0]) + 10.24 * sensitivity[3] as f64,
        (1024.0 - BASE[1]) + 10.24 * sensitivity[5] as f64,
        (1024.0 - BASE[2]) + 10.24 * sensitivity[1] as f64,
    ];
    let q = |w: [f64; 3]| w.map(|v| v as i32);
    (q(comp), q(luma))
}

/// `CImgOpBlackWhiteV8_tonemap` (`p_041248720.cl`), 16-bit path: sort RGB, pick
/// the sector by `(argmax, argmin)`, then
/// `t = min + ((mid-min)*wMin + (max-mid)*wMax)/1024` (C truncating division,
/// not a shift). A neutral floor follows when `t*8 < luma`.
pub(crate) fn bw_sector_y(g: [f32; 3], sensitivity: &[f32; 6]) -> f32 {
    let (w_min, w_max) = bw_weights(sensitivity);
    let q = g.map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as i32);
    // The host's argmax favours R unless B >= R, and argmin favours R unless
    // G < R. On a tie the tied channel is the mid one for every pure primary
    // and secondary, so its `max - mid` / `mid - min` term is zero and the
    // choice cannot move the result; keep the order for the other sectors.
    let (i_max, max) = if q[0] > q[2] { (0, q[0]) } else { (2, q[2]) };
    let (i_max, max) = if q[1] > max { (1, q[1]) } else { (i_max, max) };
    let (i_min, min) = if q[1] < q[0] { (1, q[1]) } else { (0, q[0]) };
    let (i_min, min) = if q[2] < min { (2, q[2]) } else { (i_min, min) };
    let mid = q[3 - i_max - i_min];
    let mut t = min + ((mid - min) * w_min[i_min] + (max - mid) * w_max[i_max]) / 1024;
    let y = (q[0] * BW_C1_LUMA_Q10[0] + q[1] * BW_C1_LUMA_Q10[1] + q[2] * BW_C1_LUMA_Q10[2]) / 1024;
    if t * 8 < y {
        t = if t <= 0 {
            0
        } else {
            let yu = y as u32;
            let a = ((yu * yu) >> 5) * 25;
            let b = 1 + (yu.wrapping_mul(288).wrapping_sub((t as u32).wrapping_mul(1024)) >> 5);
            ((a / b) as i32 - (yu / 32) as i32).max(0)
        };
    }
    let shift = bw_shift(&w_min, &w_max);
    (((t >> shift) << shift).clamp(0, 65535) as f32) / 65535.0
}

/// The op pairs a right shift with the splittone stage's left shift; it keeps
/// the Q10 weights inside `0 ..= 1024` when a slider pushes past full scale.
fn bw_shift(w_min: &[i32; 3], w_max: &[i32; 3]) -> u32 {
    let mut m = w_min.iter().chain(w_max).copied().max().unwrap_or(0).max(0);
    let mut shift = 0;
    while m > 1024 {
        m >>= 1;
        shift += 1;
    }
    shift
}
