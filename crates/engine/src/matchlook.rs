//! Match Look: measure a reference image and return the adjustments that move a
//! target image toward its look.
//!
//! Statistics only — no model. Each tool gets one scalar from the two images:
//! gray-world means for white balance, median luma for exposure, the 1 % / 99 %
//! luma points for levels, and mean chroma for saturation. Channel-level color
//! and per-hue shifts are left to the Color Editor, as Capture One's own
//! Match Look model would do better.
//!
//! White balance is expressed as kelvin/tint, so the measured RGB ratio is
//! inverted through the engine's own temperature model
//! ([`crate::mock::process::white_balance_gains`]) instead of a second one.

use anyhow::{bail, Result};
use serde::Serialize;

use crate::{AdjustmentSet, LevelsChannel, WhiteBalanceMode};

/// One tool's movement, for the inspector's "what changed" list.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MatchItem {
    /// [`crate::ParamMeta::key`] the movement applies to, e.g. `Exposure`.
    pub key: &'static str,
    /// The value written into [`AdjustmentSet`], in the tool's own unit.
    pub applied: f32,
    pub unit: &'static str,
    /// `0.0 ..= 1.0`; low when the measurement had too little signal to trust.
    pub confidence: f32,
    /// The same quantity measured on the reference image.
    pub reference: f32,
    /// …and on the target image.
    pub target: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MatchReport {
    /// White balance, exposure, levels, saturation — in the order they render.
    pub items: Vec<MatchItem>,
    /// Perceptual luma distance the four tools could not close, `0.0 ..= 1.0`.
    pub residual: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct MatchLook {
    /// The slice to overlay on the target's existing edits.
    pub adjustments: AdjustmentSet,
    pub report: MatchReport,
}

/// Ranges come from the same catalog the inspector draws its sliders from.
fn range(key: &str) -> (f32, f32) {
    crate::param_meta::CATALOG
        .iter()
        .find(|m| m.key == key)
        .map(|m| (m.min, m.max))
        .unwrap_or((f32::MIN, f32::MAX))
}

fn clamp_to(key: &str, v: f32) -> f32 {
    let (lo, hi) = range(key);
    v.clamp(lo, hi)
}

fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

struct Stats {
    /// Encoded luma percentiles.
    p01: f32,
    p50: f32,
    p99: f32,
    /// Clipping-excluded linear RGB means, for gray-world.
    mean_lin: [f32; 3],
    /// Mean HSV saturation `(max - min) / max` over linear light. Exactly
    /// invariant to an exposure scale, so a darker copy matches as-is.
    chroma: f32,
}

impl Stats {
    fn median_linear(&self) -> f32 {
        srgb_to_linear(self.p50)
    }

    fn luma_linear(&self) -> f32 {
        0.2126 * self.mean_lin[0] + 0.7152 * self.mean_lin[1] + 0.0722 * self.mean_lin[2]
    }
}

fn analyze(pixels: &[u8], width: u32, height: u32) -> Result<Stats> {
    let expected = width as usize * height as usize * 4;
    if width == 0 || height == 0 {
        bail!("match look needs a non-empty image ({width}x{height})");
    }
    if pixels.len() != expected {
        bail!("match look needs {expected} RGBA bytes for {width}x{height}, got {}", pixels.len());
    }
    // ~200k samples is enough for percentiles and keeps large images cheap.
    let step = (((width as f64 * height as f64) / 200_000.0).sqrt().ceil() as usize).max(1);

    let mut hist = [0u32; 256];
    let mut sum = [0f64; 3];
    let mut lit = 0u32;
    let mut chroma_sum = 0f64;
    let mut chroma_n = 0u32;
    let mut all_sum = [0f64; 3];
    let mut seen = 0u32;
    for y in (0..height as usize).step_by(step) {
        for x in (0..width as usize).step_by(step) {
            let i = (y * width as usize + x) * 4;
            let c = [pixels[i] as f32 / 255.0, pixels[i + 1] as f32 / 255.0, pixels[i + 2] as f32 / 255.0];
            let l = 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
            hist[(l * 255.0 + 0.5).clamp(0.0, 255.0) as usize] += 1;
            seen += 1;
            let lin = [srgb_to_linear(c[0]), srgb_to_linear(c[1]), srgb_to_linear(c[2])];
            for k in 0..3 {
                all_sum[k] += lin[k] as f64;
            }
            // Clipped pixels tell nothing about the scene's color cast.
            if (0.02..=0.98).contains(&l) {
                lit += 1;
                let hi = lin[0].max(lin[1]).max(lin[2]);
                if hi > 1e-3 {
                    chroma_sum += ((hi - lin[0].min(lin[1]).min(lin[2])) / hi) as f64;
                    chroma_n += 1;
                }
                for k in 0..3 {
                    sum[k] += lin[k] as f64;
                }
            }
        }
    }
    let (sum, n) = if lit > 0 { (sum, lit) } else { (all_sum, seen) };
    Ok(Stats {
        p01: percentile(&hist, seen, 0.01),
        p50: percentile(&hist, seen, 0.50),
        p99: percentile(&hist, seen, 0.99),
        mean_lin: [
            (sum[0] / n.max(1) as f64) as f32,
            (sum[1] / n.max(1) as f64) as f32,
            (sum[2] / n.max(1) as f64) as f32,
        ],
        chroma: (chroma_sum / chroma_n.max(1) as f64) as f32,
    })
}

fn percentile(hist: &[u32; 256], total: u32, q: f32) -> f32 {
    let target = (q * total as f32) as u32;
    let mut acc = 0;
    for (i, count) in hist.iter().enumerate() {
        acc += count;
        if acc > target {
            return i as f32 / 255.0;
        }
    }
    1.0
}

/// Kelvin whose `r/b` gain matches `ratio`. The gain rises with kelvin, so a
/// binary search over the slider range is exact enough and allocates nothing.
fn kelvin_for_ratio(ratio: f32) -> f32 {
    let (lo, hi) = range("WhiteBalanceTemperature");
    let gain_ratio = |k: f32| {
        let g = crate::mock::process::white_balance_gains(k, 0.0);
        g[0] / g[2]
    };
    if ratio <= gain_ratio(lo) {
        return lo;
    }
    if ratio >= gain_ratio(hi) {
        return hi;
    }
    let (mut lo, mut hi) = (lo, hi);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if gain_ratio(mid) < ratio { lo = mid } else { hi = mid }
    }
    0.5 * (lo + hi)
}

/// Measures both images and returns the adjustments that carry the reference's
/// look onto the target. Empty pixels or a zero dimension is an error; a nearly
/// black or white image yields neutral values with low confidence, never NaN.
pub fn match_look(
    reference: &[u8],
    ref_w: u32,
    ref_h: u32,
    target: &[u8],
    w: u32,
    h: u32,
) -> Result<MatchLook> {
    let r = analyze(reference, ref_w, ref_h)?;
    let t = analyze(target, w, h)?;
    let mut items = Vec::with_capacity(4);

    // ── White balance: gray-world ratios, inverted through the kelvin model ──
    let need = |k: usize| {
        if t.mean_lin[k] > 1e-5 { r.mean_lin[k] / t.mean_lin[k] } else { 1.0 }
    };
    let (nr, ng, nb) = (need(0), need(1), need(2));
    let mut wb_conf = if t.luma_linear() > 0.002 && t.luma_linear() < 0.98 { 0.7 } else { 0.0 };
    if !(nr.is_finite() && ng.is_finite() && nb.is_finite()) || nr <= 0.0 || nb <= 0.0 {
        wb_conf = 0.0;
    }
    let kelvin = if wb_conf > 0.0 { kelvin_for_ratio(nr / nb) } else { 5600.0 };
    // Green relative to the r/b geometric mean; the tint slider maps 1.0 to 0.
    let green_ratio = if wb_conf > 0.0 { ng / (nr * nb).sqrt() } else { 1.0 };
    let tint = clamp_to("WhiteBalanceTint", (1.0 - green_ratio) * 100.0 / 0.6);
    let kelvin = clamp_to("WhiteBalanceTemperature", kelvin);
    items.push(MatchItem {
        key: "WhiteBalanceTemperature",
        applied: kelvin,
        unit: "K",
        confidence: wb_conf,
        reference: r.mean_lin[0] / r.mean_lin[2].max(1e-5),
        target: t.mean_lin[0] / t.mean_lin[2].max(1e-5),
    });
    items.push(MatchItem {
        key: "WhiteBalanceTint",
        applied: tint,
        unit: "",
        confidence: wb_conf,
        reference: 1.0,
        target: ng / (nr * nb).max(1e-5).sqrt(),
    });

    // ── Exposure: median luma ratio in linear light ──
    let (r_med, t_med) = (r.median_linear(), t.median_linear());
    let mut exposure = if t_med > 1e-4 { (r_med / t_med).log2() } else { 0.0 };
    let exp_conf = if t_med > 1e-4 && r_med > 1e-4 { 1.0 } else { 0.0 };
    exposure = clamp_to("Exposure", exposure);
    items.push(MatchItem {
        key: "Exposure",
        applied: exposure,
        unit: "EV",
        confidence: exp_conf,
        reference: r.p50,
        target: t.p50,
    });

    // ── Levels: map post-exposure target endpoints onto the reference's ──
    // Levels render after exposure, so the target's points are moved by the
    // exposure gain first; otherwise the two tools would correct the same shift.
    let gain = 2f32.powf(exposure);
    let mut in_lo = linear_to_srgb(srgb_to_linear(t.p01) * gain).clamp(0.0, 1.0);
    let mut in_hi = linear_to_srgb(srgb_to_linear(t.p99) * gain).clamp(0.0, 1.0);
    let (mut out_lo, mut out_hi) = (r.p01, r.p99);
    let mut levels_conf = 1.0;
    if in_hi - in_lo < 0.02 || out_hi - out_lo < 0.02 {
        // A flat target or reference has no endpoints to map; leave levels alone.
        (in_lo, in_hi, out_lo, out_hi) = (0.0, 1.0, 0.0, 1.0);
        levels_conf = 0.0;
    }
    items.push(MatchItem {
        key: "Levels",
        applied: (out_hi - out_lo) - (in_hi - in_lo),
        unit: "",
        confidence: levels_conf,
        reference: out_hi - out_lo,
        target: in_hi - in_lo,
    });

    // ── Saturation: mean chroma ratio ──
    let mut saturation = if t.chroma > 0.005 { (r.chroma / t.chroma - 1.0) * 100.0 } else { 0.0 };
    let sat_conf = if t.chroma > 0.01 { 1.0 } else { 0.0 };
    saturation = clamp_to("Saturation", saturation);
    items.push(MatchItem {
        key: "Saturation",
        applied: saturation,
        unit: "",
        confidence: sat_conf,
        reference: r.chroma,
        target: t.chroma,
    });

    let mut adjustments = AdjustmentSet::default();
    adjustments.white_balance.mode = WhiteBalanceMode::Custom;
    adjustments.white_balance.kelvin = kelvin;
    adjustments.white_balance.tint = tint;
    adjustments.exposure.exposure = exposure;
    adjustments.exposure.saturation = saturation;
    adjustments.levels.rgb = LevelsChannel {
        input_black: clamp_to("ShadowRGB", in_lo),
        input_white: clamp_to("HighlightRGB", in_hi),
        gamma: 1.0,
        output_black: clamp_to("TargetShadowRGB", out_lo),
        output_white: clamp_to("TargetHighlightRGB", out_hi),
    };

    // Luma the four tools leave between the gray points, after white balance,
    // exposure and saturation have done their part.
    let wb = crate::mock::process::white_balance_gains(kelvin, tint);
    let after = [
        t.mean_lin[0] * wb[0] * gain,
        t.mean_lin[1] * wb[1] * gain,
        t.mean_lin[2] * wb[2] * gain,
    ];
    let after_luma = 0.2126 * after[0] + 0.7152 * after[1] + 0.0722 * after[2];
    let residual = if r.luma_linear() > 1e-4 {
        ((after_luma / r.luma_linear()).log2().abs() / 4.0).clamp(0.0, 1.0)
    } else {
        0.0
    };

    Ok(MatchLook { adjustments, report: MatchReport { items, residual } })
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: u32 = 96;
    const H: u32 = 64;

    /// A colorful gradient: red across, green down, blue the average.
    fn gradient() -> Vec<u8> {
        let mut v = Vec::with_capacity((W * H * 4) as usize);
        for y in 0..H {
            for x in 0..W {
                let fx = x as f32 / (W - 1) as f32;
                let fy = y as f32 / (H - 1) as f32;
                v.push((fx * 255.0) as u8);
                v.push((fy * 255.0) as u8);
                v.push(((fx * 0.6 + fy * 0.4) * 255.0) as u8);
                v.push(255);
            }
        }
        v
    }

    /// Same picture two stops darker, done in linear light like a raw exposure.
    fn two_stops_darker(src: &[u8]) -> Vec<u8> {
        let mut v = Vec::with_capacity(src.len());
        for px in src.chunks(4) {
            for k in 0..3 {
                let lin = srgb_to_linear(px[k] as f32 / 255.0) / 4.0;
                v.push((linear_to_srgb(lin).clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
            }
            v.push(255);
        }
        v
    }

    fn flat(value: u8, w: u32, h: u32) -> Vec<u8> {
        std::iter::repeat([value, value, value, 255]).take((w * h) as usize).flatten().collect()
    }

    fn item<'a>(look: &'a MatchLook, key: &str) -> &'a MatchItem {
        look.report.items.iter().find(|i| i.key == key).expect("reported item")
    }

    fn assert_in_range(look: &MatchLook) {
        let a = &look.adjustments;
        for (v, key) in [
            (a.exposure.exposure, "Exposure"),
            (a.exposure.saturation, "Saturation"),
            (a.white_balance.kelvin, "WhiteBalanceTemperature"),
            (a.white_balance.tint, "WhiteBalanceTint"),
            (a.levels.rgb.input_black, "ShadowRGB"),
            (a.levels.rgb.input_white, "HighlightRGB"),
            (a.levels.rgb.output_black, "TargetShadowRGB"),
            (a.levels.rgb.output_white, "TargetHighlightRGB"),
        ] {
            let (lo, hi) = range(key);
            assert!(v.is_finite(), "{key} is {v}");
            assert!(v >= lo && v <= hi, "{key} {v} outside {lo}..={hi}");
        }
        assert!(a.levels.rgb.input_white > a.levels.rgb.input_black);
        assert!(a.levels.rgb.output_white > a.levels.rgb.output_black);
    }

    #[test]
    fn same_image_is_a_no_op() {
        let img = gradient();
        let look = match_look(&img, W, H, &img, W, H).unwrap();
        assert_in_range(&look);
        assert!(item(&look, "Exposure").applied.abs() < 0.01);
        assert!((item(&look, "WhiteBalanceTemperature").applied - 5600.0).abs() < 10.0);
        assert!(item(&look, "WhiteBalanceTint").applied.abs() < 0.5);
        assert!(item(&look, "Saturation").applied.abs() < 1.0);
        let l = &look.adjustments.levels.rgb;
        assert!((l.input_black - l.output_black).abs() < 0.01);
        assert!((l.input_white - l.output_white).abs() < 0.01);
        assert!(look.report.residual < 0.01);
    }

    #[test]
    fn recovers_a_two_stop_darkened_target() {
        let reference = gradient();
        let target = two_stops_darker(&reference);
        let look = match_look(&reference, W, H, &target, W, H).unwrap();
        assert_in_range(&look);
        let ev = item(&look, "Exposure").applied;
        assert!((ev - 2.0).abs() < 0.15, "exposure {ev}");
        // HSV saturation is exposure-invariant, so darkening must not ask for more.
        let sat = item(&look, "Saturation").applied;
        assert!(sat.abs() < 5.0, "saturation {sat}");
        // Levels must not undo the exposure: the endpoints already land together.
        let l = &look.adjustments.levels.rgb;
        assert!((l.input_black - l.output_black).abs() < 0.05);
        assert!((l.input_white - l.output_white).abs() < 0.05);
    }

    #[test]
    fn flat_images_never_produce_nan() {
        for (r, t) in [(0u8, 0u8), (255, 255), (0, 255), (255, 0), (7, 250)] {
            let reference = flat(r, 16, 16);
            let target = flat(t, 16, 16);
            let look = match_look(&reference, 16, 16, &target, 16, 16).unwrap();
            assert_in_range(&look);
            assert!(look.report.residual.is_finite());
            for i in &look.report.items {
                assert!(i.applied.is_finite() && (0.0..=1.0).contains(&i.confidence), "{}", i.key);
            }
        }
    }

    #[test]
    fn rejects_empty_or_mismatched_buffers() {
        let img = gradient();
        assert!(match_look(&img, 0, H, &img, W, H).is_err());
        assert!(match_look(&img, W, H, &[], W, H).is_err());
        assert!(match_look(&img, W, H, &img, W + 1, H).is_err());
    }

    #[test]
    fn different_sizes_and_alpha_are_fine() {
        let reference = gradient();
        let mut target = gradient();
        for px in target.chunks_mut(4) {
            px[3] = 0;
        }
        let look = match_look(&reference, W, H, &target, W, H).unwrap();
        assert!(look.report.items.len() >= 4);
        assert_in_range(&look);
    }
}
