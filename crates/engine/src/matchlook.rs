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
use serde::{Deserialize, Serialize};

use crate::{AdjustmentSet, Curve, CurvePoint, LevelsChannel, WhiteBalanceMode};

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
    // Work in gain space: C1 stores gains, and K/T is only their display pair.
    let wb_gains = if wb_conf > 0.0 { [nr / ng, 1.0, nb / ng] } else { [1.0, 1.0, 1.0] };
    let (kelvin, tint) = crate::c1_model::gains_to_kelvin_tint(wb_gains);
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
        target: wb_gains[2],
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
    adjustments.white_balance.gains = Some(wb_gains);
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
    let wb = wb_gains;
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

// ───────────────────────── Match Look model head (AI Color Grade) ─────────────────────────
//
// A style's `<AICGR>` stores only the reference thumbnails (base64 multi-frame
// TIFF), the 14 switches and `impact`; the parameters come from four ONNX heads
// under `AIModels/Common/MatchLook`. The layouts below follow the training-label
// order in `P1.C1.Components.PrivilegedMode.MatchLookTrainingProviders` (see
// `out/look-vectors.tsv` in the c1_coloreditor_skin_r271 lane).

/// `MatchLookColor.onnx` output length (`[1, 68]`).
pub const LOOK_COLOR_VECTOR: usize = 68;
/// `MatchLookBw.onnx` output length (`[1, 7]`).
pub const LOOK_BW_VECTOR: usize = 7;

fn rgba_len(width: u32, height: u32) -> Option<usize> {
    (width as usize).checked_mul(height as usize)?.checked_mul(4)
}

/// Runs one Match Look head on an RGBA8 reference and returns its raw output.
/// AICGR stores Adobe RGB 1998 samples, which are passed through without conversion.
///
/// Preprocessing follows the r288 probe: bilinear resize to `size`, planar
/// NCHW, values normalised to 0..1. The model output is the training-label
/// vector, not engine fields.
pub fn run_look_head(
    model: &std::path::Path,
    rgba: &[u8],
    width: u32,
    height: u32,
    size: u32,
) -> Result<Vec<f32>> {
    if width == 0 || height == 0 || size == 0
        || rgba_len(width, height).is_none_or(|bytes| rgba.len() < bytes)
    {
        bail!("Match Look model needs a non-empty RGBA frame");
    }
    let runtime = crate::aimask::Runtime::get()
        .ok_or_else(|| anyhow::anyhow!("onnxruntime is unavailable"))?;
    let session = runtime.open_session(model)?;
    let side = size as usize;
    let mut input = vec![0f32; 3 * side * side];
    for y in 0..side {
        let sy = ((y as f32 + 0.5) * height as f32 / side as f32 - 0.5).max(0.0);
        let y0 = sy.floor() as u32;
        let y1 = (y0 + 1).min(height - 1);
        let fy = sy - y0 as f32;
        for x in 0..side {
            let sx = ((x as f32 + 0.5) * width as f32 / side as f32 - 0.5).max(0.0);
            let x0 = sx.floor() as u32;
            let x1 = (x0 + 1).min(width - 1);
            let fx = sx - x0 as f32;
            let at = |px: u32, py: u32, c: usize| {
                rgba[(py as usize * width as usize + px as usize) * 4 + c] as f32 / 255.0
            };
            let index = y * side + x;
            for channel in 0..3 {
                let top = at(x0, y0, channel) * (1.0 - fx) + at(x1, y0, channel) * fx;
                let bottom = at(x0, y1, channel) * (1.0 - fx) + at(x1, y1, channel) * fx;
                input[channel * side * side + index] = top * (1.0 - fy) + bottom * fy;
            }
        }
    }
    let outputs = runtime.run(&session, &mut input, &[1, 3, size as i64, size as i64])?;
    outputs
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("Match Look model produced no output"))
}

/// The 14 `<AICGR>` switches and `impact`.
///
/// Every switch defaults to true and `impact` to 100; a false switch keeps that
/// tool's predicted parameters out of the result.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LookComponents {
    pub exposure: bool,
    pub whitebalance: bool,
    pub brightness: bool,
    pub contrast: bool,
    pub hdr_highlight_shadow: bool,
    pub hdr_white_black: bool,
    pub target_levels: bool,
    pub color_balance_master: bool,
    pub color_balance_hms: bool,
    pub saturation: bool,
    pub basic_color_edits: bool,
    pub luma_curve: bool,
    pub color_channel_curves: bool,
    pub black_and_white: bool,
    /// 0 ..= 100, how much of the looked-up parameters is applied.
    pub impact: f32,
}

impl Default for LookComponents {
    fn default() -> Self {
        Self {
            exposure: true,
            whitebalance: true,
            brightness: true,
            contrast: true,
            hdr_highlight_shadow: true,
            hdr_white_black: true,
            target_levels: true,
            color_balance_master: true,
            color_balance_hms: true,
            saturation: true,
            basic_color_edits: true,
            luma_curve: true,
            color_channel_curves: true,
            black_and_white: true,
            impact: 100.0,
        }
    }
}

/// One reference frame as C1 feeds the models: 8-bit RGB in Adobe RGB 1998.
#[derive(Clone, Debug, PartialEq)]
pub struct LookThumbnail {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// `<AICGR>`: the reference frames plus the 14 tool switches and `impact`.
#[derive(Clone, Debug, PartialEq)]
pub struct LookReference {
    pub components: LookComponents,
    pub thumbnails: Vec<LookThumbnail>,
}

/// Both heads' raw output.
#[derive(Clone, Debug, PartialEq)]
pub struct LookPrediction {
    pub color: [f32; LOOK_COLOR_VECTOR],
    pub bw: [f32; LOOK_BW_VECTOR],
}

/// The four ONNX heads (`Exp`, `Brightness`, `Color`, `Bw`).
///
/// No runtime is linked into this crate, so [`look_adjustments`] reports
/// [`LookError::NoModel`] until a caller supplies an implementation.
pub trait LookModel {
    fn predict(&self, thumbnails: &[LookThumbnail]) -> Result<LookPrediction, LookError>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LookError {
    /// No ONNX runtime is wired in; the reference cannot be evaluated.
    NoModel,
    /// `<AICGR>` carried no reference frame.
    NoThumbnail,
    /// A thumbnail does not match its own `width × height × 4`.
    BadThumbnail { index: usize, width: u32, height: u32, bytes: usize },
}

impl std::fmt::Display for LookError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoModel => write!(f, "Match Look needs an ONNX runtime for the four heads; none is linked"),
            Self::NoThumbnail => write!(f, "the style's <AICGR> carries no reference thumbnail"),
            Self::BadThumbnail { index, width, height, bytes } => write!(
                f,
                "thumbnail {index} has invalid RGBA dimensions or length ({width}x{height}, {bytes} bytes)"
            ),
        }
    }
}

impl std::error::Error for LookError {}

/// Runs the model on a style's reference frames and returns the adjustments.
pub fn look_adjustments(
    model: Option<&dyn LookModel>,
    thumbnails: &[LookThumbnail],
    components: &LookComponents,
) -> Result<AdjustmentSet, LookError> {
    if thumbnails.is_empty() {
        return Err(LookError::NoThumbnail);
    }
    for (index, thumb) in thumbnails.iter().enumerate() {
        if thumb.width == 0 || thumb.height == 0
            || rgba_len(thumb.width, thumb.height) != Some(thumb.rgba.len())
        {
            return Err(LookError::BadThumbnail {
                index,
                width: thumb.width,
                height: thumb.height,
                bytes: thumb.rgba.len(),
            });
        }
    }
    let model = model.ok_or(LookError::NoModel)?;
    let prediction = model.predict(thumbnails)?;
    let mut out = AdjustmentSet::default();
    apply_look_color(&prediction.color, components, &mut out);
    apply_look_bw(&prediction.bw, components, &mut out);
    Ok(out)
}

fn predict_reference_head<const N: usize>(
    thumbnail: &LookThumbnail,
    name: &str,
    size: u32,
) -> Option<[f32; N]> {
    let path = crate::aimask::model_path(name).ok()?;
    run_look_head(&path, &thumbnail.rgba, thumbnail.width, thumbnail.height, size)
        .map_err(|error| log::warn!("Match Look {name} failed: {error}"))
        .ok()?
        .try_into()
        .ok()
}

fn merge_reference_color_heads(
    mut color: [f32; LOOK_COLOR_VECTOR],
    exposure_wb: [f32; 3],
    brightness_contrast: [f32; 2],
) -> [f32; LOOK_COLOR_VECTOR] {
    // Native extraction consumes Color from index 5 and builds these two tools from separate heads.
    color[..3].copy_from_slice(&exposure_wb);
    color[3..5].copy_from_slice(&brightness_contrast);
    color
}

fn predict_reference_tone(
    thumbnail: &LookThumbnail,
    components: &LookComponents,
) -> Option<([f32; 3], [f32; 2])> {
    let exposure_wb = if components.exposure || components.whitebalance {
        predict_reference_head(thumbnail, "MatchLookExp.onnx", 384)?
    } else {
        [0.5; 3]
    };
    let brightness_contrast = if components.brightness || components.contrast {
        predict_reference_head(thumbnail, "MatchLookBrightness.onnx", 384)?
    } else {
        [0.5; 2]
    };

    Some((exposure_wb, brightness_contrast))
}

pub(crate) fn predict_look_reference(reference: &LookReference) -> Option<LookPrediction> {
    let thumbnail = reference.thumbnails.first()?;
    let components = &reference.components;
    let color = predict_reference_head(thumbnail, "MatchLookColor.onnx", 384)?;
    let bw = predict_reference_head(thumbnail, "MatchLookBw.onnx", 224)?;
    let (exposure_wb, brightness_contrast) = predict_reference_tone(thumbnail, components)?;
    let color = merge_reference_color_heads(color, exposure_wb, brightness_contrast);
    Some(LookPrediction { color, bw })
}

/// Approximates Match Look during style import with its first AICGR reference.
/// Native multi-reference weights and final application depend on the destination image.
pub fn look_from_reference(reference: &LookReference) -> Option<AdjustmentSet> {
    let prediction = predict_look_reference(reference)?;
    let mut out = AdjustmentSet::default();
    apply_look_color(&prediction.color, &reference.components, &mut out);
    apply_look_bw(&prediction.bw, &reference.components, &mut out);
    Some(out)
}

/// Camera-specific conversion used by target-aware Match Look white balance.
pub trait LookWhiteBalanceContext {
    /// Resolve the target's actual gains or camera metadata to Kelvin and tint.
    fn target_temperature(&self, target: &crate::WhiteBalance) -> (f32, f32);
    fn temperature_to_gains(&self, kelvin: f32, tint: f32) -> [f32; 3];
}

/// Inputs for applying the native tone callbacks to an already prepared target.
pub struct LookToneContext<'a> {
    /// Decoded component baselines, in EV, temperature, tint, brightness, contrast order.
    pub subtract_baseline: [f32; 5],
    pub add_baseline: [f32; 5],
    /// Effective weights for exposure, white balance, brightness and contrast.
    /// A missing entry starts at 0.9. Each enabled callback stores 0.7 times its current weight.
    pub weights: [Option<f32>; 4],
    pub auto_white_balance: bool,
    pub white_balance: &'a dyn LookWhiteBalanceContext,
}

fn temperature_interpolate(x: f32, from: &[f32; 6], to: &[f32; 6]) -> f32 {
    let x = x.clamp(from[0], from[5]);
    for i in 0..5 {
        if x <= from[i + 1] {
            return to[i] + (x - from[i]) * ((to[i + 1] - to[i]) / (from[i + 1] - from[i]));
        }
    }
    f32::NAN
}

/// Cosine similarity between Capture One's 1280-value reference and target descriptors.
pub fn look_descriptor_similarity(reference: &[f32; 1280], target: &[f32; 1280]) -> Option<f32> {
    let (mut dot, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
    for (&a, &b) in reference.iter().zip(target) {
        dot += f64::from(a * b);
        aa += f64::from(a * a);
        bb += f64::from(b * b);
    }
    let score = (dot / (aa.sqrt() * bb.sqrt())) as f32;
    score.is_finite().then_some(score)
}

fn look_reference_weights(similarities: &[f32]) -> Option<Vec<f32>> {
    match similarities {
        [] => return None,
        [score] => return score.is_finite().then(|| vec![1.0]),
        _ if !similarities.iter().all(|score| score.is_finite()) => return None,
        _ => {}
    }
    let minimum = similarities.iter().copied().fold(0.0, f32::min);
    let mut weights: Vec<_> = similarities.iter().map(|score| score - minimum).collect();
    let sum: f32 = weights.iter().sum();
    if !sum.is_finite() {
        return None;
    }
    if sum > 0.0 {
        for weight in &mut weights {
            *weight /= sum;
        }
    } else {
        // Native normalization fills 1.0 per reference when the shifted sum is zero.
        weights.fill(1.0);
    }
    Some(weights)
}

fn decode_look_tone(exposure_wb: &[f32], brightness_contrast: &[f32]) -> [f32; 5] {
    [
        exposure_wb[0] * 8.0 - 4.0,
        exposure_wb[1] * 28000.0 - 14000.0,
        exposure_wb[2] * 200.0 - 100.0,
        brightness_contrast[0] * 100.0 - 50.0,
        brightness_contrast[1] * 100.0 - 50.0,
    ]
}

fn merge_reference_tone(
    thumbnails: &[LookThumbnail],
    similarities: &[f32],
    mut predict: impl FnMut(&LookThumbnail) -> Option<([f32; 3], [f32; 2])>,
) -> Option<[f32; 5]> {
    if thumbnails.len() != similarities.len() {
        return None;
    }
    let weights = look_reference_weights(similarities)?;
    let mut merged = [0.0; 5];
    for (thumbnail, weight) in thumbnails.iter().zip(weights) {
        let (exposure_wb, brightness_contrast) = predict(thumbnail)?;
        let decoded = decode_look_tone(&exposure_wb, &brightness_contrast);
        for (slot, value) in merged.iter_mut().zip(decoded) {
            *slot += weight * value;
        }
    }
    Some(merged)
}

/// Applies weighted exposure, white balance, brightness and contrast from every
/// reference frame. Similarities must be supplied in frame order by the caller.
/// Weights and target settings are updated only after all model outputs are available.
pub fn apply_weighted_reference_tone(
    reference: &LookReference,
    similarities: &[f32],
    target: &mut AdjustmentSet,
    context: &mut LookToneContext<'_>,
) -> Option<()> {
    let decoded = merge_reference_tone(&reference.thumbnails, similarities, |thumbnail| {
        predict_reference_tone(thumbnail, &reference.components)
    })?;
    apply_decoded_look_tone(decoded, &reference.components, target, context);
    Some(())
}

const SIMILARITY_SIDE: u32 = 224;

fn similarity_model_path() -> Option<std::path::PathBuf> {
    crate::aimask::model_path("similarity.onnx")
        .or_else(|_| crate::aimask::model_path("Similarity/similarity.onnx"))
        .ok()
}

fn similarity_input(frame: &LookThumbnail) -> Option<Vec<f32>> {
    if frame.width == 0 || frame.height == 0
        || rgba_len(frame.width, frame.height) != Some(frame.rgba.len())
    {
        return None;
    }
    let source = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
        frame.width, frame.height, frame.rgba.as_slice(),
    )?;
    let resized = image::imageops::resize(
        &source, SIMILARITY_SIDE, SIMILARITY_SIDE, image::imageops::FilterType::Triangle,
    );
    // The NHWC model performs its own 1/255 rescaling.
    Some(resized.pixels().flat_map(|pixel| {
        let [red, green, blue, _] = pixel.0;
        [f32::from(red), f32::from(green), f32::from(blue)]
    }).collect())
}

fn score_reference_frames(
    references: &[LookThumbnail],
    target: &LookThumbnail,
    mut describe: impl FnMut(&LookThumbnail) -> Option<[f32; 1280]>,
) -> Option<Vec<f32>> {
    let target_descriptor = describe(target)?;
    references.iter().map(|frame| {
        let descriptor = describe(frame)?;
        look_descriptor_similarity(&descriptor, &target_descriptor)
    }).collect()
}

/// Predicts target similarities from Adobe RGB 1998 RGBA8 previews and applies
/// the combined tone adjustments. Both reference and target previews must
/// already be in Adobe RGB 1998. The single-reference path skips similarity.
pub fn apply_similarity_weighted_reference_tone(
    reference: &LookReference,
    target_preview: &LookThumbnail,
    target: &mut AdjustmentSet,
    context: &mut LookToneContext<'_>,
) -> Option<()> {
    match reference.thumbnails.as_slice() {
        [] => return None,
        [_] => return apply_weighted_reference_tone(reference, &[1.0], target, context),
        _ => {}
    }

    let model = similarity_model_path()?;
    let runtime = crate::aimask::Runtime::get()?;
    let session = runtime.open_session(&model).ok()?;
    let side = i64::from(SIMILARITY_SIDE);
    let scores = score_reference_frames(&reference.thumbnails, target_preview, |frame| {
        let mut input = similarity_input(frame)?;
        let output = runtime.run(&session, &mut input, &[1, side, side, 3]).ok()?;
        output.into_iter().next()?.try_into().ok()
    })?;
    apply_weighted_reference_tone(reference, &scores, target, context)
}

/// Applies the native exposure, brightness, contrast and WB callbacks to the
/// target state at the point where those callbacks run. The caller supplies
/// the decoded component baselines, effective weight state and camera conversion.
pub fn apply_look_tone(
    prediction: &LookPrediction,
    components: &LookComponents,
    target: &mut AdjustmentSet,
    context: &mut LookToneContext<'_>,
) {
    let decoded = decode_look_tone(&prediction.color[..3], &prediction.color[3..5]);
    apply_decoded_look_tone(decoded, components, target, context);
}

fn apply_decoded_look_tone(
    decoded: [f32; 5],
    components: &LookComponents,
    target: &mut AdjustmentSet,
    context: &mut LookToneContext<'_>,
) {
    let impact = components.impact / 100.0;
    let mut values: [f32; 5] = std::array::from_fn(|i| {
        (decoded[i] - context.subtract_baseline[i]) * impact + context.add_baseline[i]
    });
    if context.auto_white_balance {
        values[0] *= 0.3;
        values[1] = 200.0;
        values[2] = 0.2;
    }

    let mut weight = |index: usize| {
        let current = context.weights[index].unwrap_or(0.9);
        context.weights[index] = Some(current * 0.7);
        current
    };

    if components.exposure {
        target.exposure.exposure =
            (target.exposure.exposure + weight(0) * values[0]).clamp(-4.0, 4.0);
    }
    if components.whitebalance {
        const KELVIN: [f32; 6] = [800.0, 1500.0, 7000.0, 8000.0, 12000.0, 14000.0];
        const STEPS: [f32; 6] = [0.0, 28.0, 138.0, 148.0, 168.0, 173.0];
        const SCALE: f64 = 13200.0 / 173.0;
        let s = weight(1);
        let (kelvin, tint) = context.white_balance.target_temperature(&target.white_balance);
        let coordinate = (800.0 + f64::from(temperature_interpolate(kelvin, &KELVIN, &STEPS)) * SCALE) as f32;
        let moved = (f64::from(coordinate) + f64::from(s * values[1])) as f32;
        let steps = (f64::from(moved - 800.0) / SCALE) as f32;
        let kelvin = temperature_interpolate(steps, &STEPS, &KELVIN);
        let tint = (f64::from(tint) + f64::from(s * values[2])).clamp(-100.0, 100.0) as f32;
        target.white_balance.kelvin = kelvin;
        target.white_balance.tint = tint;
        target.white_balance.mode = WhiteBalanceMode::Custom;
        target.white_balance.gains = Some(context.white_balance.temperature_to_gains(kelvin, tint));
    }
    if components.brightness {
        let delta = (weight(2) * 0.5) * values[3];
        target.exposure.brightness = (target.exposure.brightness + delta).clamp(-50.0, 50.0);
    }
    if components.contrast {
        let base = target.exposure.contrast;
        let mut delta = f64::from((weight(3) * values[4]) * 0.5) * 0.75;
        let trial = (f64::from(base) + delta) as f32;
        let headroom = if delta > 0.0 && base >= 0.0 && trial > 15.0 {
            Some((15.0 - base).max(0.0))
        } else if delta < 0.0 && base <= 0.0 && trial < -50.0 {
            Some((-50.0 - base).min(0.0))
        } else {
            None
        };
        if let Some(headroom) = headroom {
            delta = f64::from(headroom + ((delta - f64::from(headroom)) as f32) * 0.2);
        }
        target.exposure.contrast = (f64::from(base) + delta).clamp(-50.0, 50.0) as f32;
    }
}

/// Combined Match Look color tool vector → [`AdjustmentSet`].
///
/// The first five values come from the separate exposure/WB and brightness/contrast heads.
/// The remaining values come from the Color head: HDR,
/// target levels, four color-balance triples, saturation, the eight basic
/// color-editor sectors and the four gradation curves. Values are offsets from
/// neutral; `impact` scales every one of them.
pub fn apply_look_color(v: &[f32; LOOK_COLOR_VECTOR], c: &LookComponents, out: &mut AdjustmentSet) {
    let s = (c.impact / 100.0).clamp(0.0, 1.0);
    let offset = |raw: f32, neutral: f32| (raw - neutral) * s;

    if c.exposure {
        // C1's exposure slider is ±4 EV while this engine's is ±2.5; the look
        // head is trained on the former, so keep its range.
        out.exposure.exposure = (v[0] * 8.0 - 4.0) * s;
    }
    if c.whitebalance {
        // The head predicts deltas; C1 writes gains, so move the current pair
        // and re-derive them through the engine's own temperature model.
        let kelvin = (v[1] * 28000.0 - 14000.0) * s;
        let tint = (v[2] * 200.0 - 100.0) * s;
        if kelvin.abs() > 1e-4 || tint.abs() > 1e-4 {
            out.white_balance.kelvin = (out.white_balance.kelvin + kelvin).clamp(2000.0, 50000.0);
            out.white_balance.tint = (out.white_balance.tint + tint).clamp(-50.0, 50.0);
            out.white_balance.mode = WhiteBalanceMode::Custom;
            out.white_balance.gains = Some(crate::mock::process::white_balance_gains(
                out.white_balance.kelvin,
                out.white_balance.tint,
            ));
        }
    }
    if c.brightness {
        out.exposure.brightness = clamp_to("Brightness", v[3] * 100.0 - 50.0) * s;
    }
    if c.contrast {
        out.exposure.contrast = clamp_to("Contrast", v[4] * 100.0 - 50.0) * s;
    }
    if c.hdr_highlight_shadow {
        out.hdr.highlights = clamp_to("HighlightRecovery", v[5] * 200.0 - 100.0) * s;
        out.hdr.shadows = clamp_to("ShadowRecovery", v[6] * 200.0 - 100.0) * s;
    }
    if c.hdr_white_black {
        out.hdr.whites = clamp_to("WhiteRecovery", v[7] * 200.0 - 100.0) * s;
        out.hdr.blacks = clamp_to("BlackRecovery", v[8] * 200.0 - 100.0) * s;
    }
    if c.target_levels {
        // No typed field yet (registry: Target Levels PARTIAL), so they ride the
        // contract map under the C1 property keys.
        let shadow = clamp_to("TargetShadowRGB", v[9]);
        let highlight = clamp_to("TargetHighlightRGB", v[10]);
        out.extra.insert("TargetShadowRGB".into(), shadow);
        out.extra.insert("TargetHighlightRGB".into(), highlight);
    }
    if c.color_balance_master || c.color_balance_hms {
        // C1 stores the triple as a 0..1 colour whose neutral is 0.5; the
        // renderer's multipliers are `1 ± 2·offset`.
        let gain = |raw: f32| 1.0 + 2.0 * offset(raw, 0.5);
        let mut gains = out.color_balance_gains.unwrap_or_default();
        if c.color_balance_master {
            gains.master = [gain(v[11]), gain(v[12]), gain(v[13])];
        }
        if c.color_balance_hms {
            gains.shadow = [gain(v[14]), gain(v[15]), gain(v[16])];
            gains.midtone = [gain(v[17]), gain(v[18]), gain(v[19])];
            gains.highlight = [gain(v[20]), gain(v[21]), gain(v[22])];
        }
        out.color_balance_gains = (!gains.is_neutral()).then_some(gains);
    }
    if c.saturation {
        out.exposure.saturation = clamp_to("Saturation", v[23] * 200.0 - 100.0) * s;
    }
    if c.basic_color_edits {
        for (sector, shift) in out.color_editor.basic.iter_mut().enumerate() {
            let at = 24 + sector * 3;
            // The sector sliders are UI ranges, not hard limits: C1's own
            // saturation span here is ±100 while this engine's slider is ±80.
            shift.hue = (v[at] * 60.0 - 30.0) * s;
            shift.saturation = (v[at + 1] * 200.0 - 100.0) * s;
            shift.lightness = (v[at + 2] * 10.0 - 5.0) * s;
        }
    }
    if c.luma_curve {
        out.curves.luma = look_curve(&v[48..53], s);
    }
    if c.color_channel_curves {
        out.curves.red = look_curve(&v[53..58], s);
        out.curves.green = look_curve(&v[58..63], s);
        out.curves.blue = look_curve(&v[63..68], s);
    }
}

/// Five control points at `x = 0, ¼, ½, ¾, 1`, moved toward the neutral line by
/// `impact`.
fn look_curve(values: &[f32], impact: f32) -> Curve {
    let points = values
        .iter()
        .enumerate()
        .map(|(i, y)| {
            let x = i as f32 / 4.0;
            CurvePoint::new(x, (x + (y - x) * impact).clamp(0.0, 1.0))
        })
        .collect();
    Curve { points }
}

/// `MatchLookBw` head → [`AdjustmentSet`].
///
/// `[0]` is the enable flag (C1's threshold is `> 0.75`), `[1..6]` are the red,
/// yellow, blue, green, cyan and magenta sensitivities.
pub fn apply_look_bw(v: &[f32; LOOK_BW_VECTOR], c: &LookComponents, out: &mut AdjustmentSet) {
    if !c.black_and_white {
        return;
    }
    let s = (c.impact / 100.0).clamp(0.0, 1.0);
    out.black_white.enabled = v[0] > 0.75;
    let value = |key: &str, i: usize| clamp_to(key, v[i] * 200.0 - 100.0) * s;
    // Head order is R, Y, B, G, C, M; the engine field is R, Y, G, C, B, M.
    out.black_white.sensitivity = [
        value("BwRed", 1),
        value("BwYellow", 2),
        value("BwGreen", 4),
        value("BwCyan", 5),
        value("BwBlue", 3),
        value("BwMagenta", 6),
    ];
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
        // Identical images must produce neutral gains. C1's neutral display
        // pair is 4325 K / -36.5 tint, not 5600 K / 0.
        let wb = &look.adjustments.white_balance;
        let gains = crate::c1_model::kelvin_tint_to_gains(wb.kelvin, wb.tint);
        assert!(
            (gains[0] - 1.0).abs() < 0.05
                && (gains[1] - 1.0).abs() < 0.05
                && (gains[2] - 1.0).abs() < 0.05,
            "{gains:?} at {} K / {} tint",
            wb.kelvin,
            wb.tint
        );
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

    /// A vector where every slot carries a distinct value, so a wrong offset or
    /// a swapped field cannot pass.
    fn distinct() -> [f32; LOOK_COLOR_VECTOR] {
        std::array::from_fn(|i| 0.05 + i as f32 * 0.012)
    }

    #[test]
    fn color_vector_writes_every_field_from_its_own_slot() {
        let v = distinct();
        let mut out = AdjustmentSet::default();
        apply_look_color(&v, &LookComponents::default(), &mut out);

        assert!((out.exposure.exposure - (v[0] * 8.0 - 4.0)).abs() < 1e-5);
        let kelvin = (5500.0 + v[1] * 28000.0 - 14000.0).clamp(2000.0, 50000.0);
        let tint = (v[2] * 200.0 - 100.0).clamp(-50.0, 50.0);
        assert!((out.white_balance.kelvin - kelvin).abs() < 1e-2);
        assert!((out.white_balance.tint - tint).abs() < 1e-5);
        assert!(out.white_balance.gains.is_some(), "a non-zero pair writes gains");
        assert!((out.exposure.brightness - (v[3] * 100.0 - 50.0)).abs() < 1e-5);
        assert!((out.exposure.contrast - (v[4] * 100.0 - 50.0)).abs() < 1e-5);
        assert!((out.hdr.highlights - (v[5] * 200.0 - 100.0)).abs() < 1e-5);
        assert!((out.hdr.shadows - (v[6] * 200.0 - 100.0)).abs() < 1e-5);
        assert!((out.hdr.whites - (v[7] * 200.0 - 100.0)).abs() < 1e-5);
        assert!((out.hdr.blacks - (v[8] * 200.0 - 100.0)).abs() < 1e-5);
        assert!((out.extra["TargetShadowRGB"] - v[9]).abs() < 1e-6);
        assert!((out.extra["TargetHighlightRGB"] - v[10]).abs() < 1e-6);
        let gains = out.color_balance_gains.expect("gains");
        let g = |slot: usize| 1.0 + 2.0 * (v[slot] - 0.5);
        for k in 0..3 {
            assert!((gains.master[k] - g(11 + k)).abs() < 1e-5, "master {k}");
            assert!((gains.shadow[k] - g(14 + k)).abs() < 1e-5, "shadow {k}");
            assert!((gains.midtone[k] - g(17 + k)).abs() < 1e-5, "midtone {k}");
            assert!((gains.highlight[k] - g(20 + k)).abs() < 1e-5, "highlight {k}");
        }
        assert!((out.exposure.saturation - (v[23] * 200.0 - 100.0)).abs() < 1e-5);
        for (sector, shift) in out.color_editor.basic.iter().enumerate() {
            let at = 24 + sector * 3;
            assert!((shift.hue - (v[at] * 60.0 - 30.0)).abs() < 1e-5, "sector {sector} hue");
            assert!((shift.saturation - (v[at + 1] * 200.0 - 100.0)).abs() < 1e-5, "sector {sector} sat");
            assert!((shift.lightness - (v[at + 2] * 10.0 - 5.0)).abs() < 1e-5, "sector {sector} light");
        }
        let channels = [(&out.curves.luma, 48), (&out.curves.red, 53), (&out.curves.green, 58), (&out.curves.blue, 63)];
        for (curve, base) in channels {
            assert_eq!(curve.points.len(), 5);
            for (i, point) in curve.points.iter().enumerate() {
                assert!((point.x - i as f32 / 4.0).abs() < 1e-6);
                assert!((point.y - v[base + i]).abs() < 1e-6, "curve {base} point {i}");
            }
        }
    }

    #[test]
    fn impact_scales_every_offset_and_switches_gate_their_tool() {
        let v = distinct();
        let half = LookComponents { impact: 50.0, ..Default::default() };
        let mut out = AdjustmentSet::default();
        apply_look_color(&v, &half, &mut out);
        assert!((out.exposure.exposure - (v[0] * 8.0 - 4.0) * 0.5).abs() < 1e-5);
        assert!((out.exposure.saturation - (v[23] * 200.0 - 100.0) * 0.5).abs() < 1e-5);
        assert!((out.color_balance_gains.expect("gains").master[0] - (1.0 + 2.0 * (v[11] - 0.5) * 0.5)).abs() < 1e-5);

        let off = LookComponents {
            exposure: false,
            whitebalance: false,
            brightness: false,
            contrast: false,
            hdr_highlight_shadow: false,
            hdr_white_black: false,
            target_levels: false,
            color_balance_master: false,
            color_balance_hms: false,
            saturation: false,
            basic_color_edits: false,
            luma_curve: false,
            color_channel_curves: false,
            black_and_white: false,
            impact: 100.0,
        };
        let mut untouched = AdjustmentSet::default();
        apply_look_color(&v, &off, &mut untouched);
        apply_look_bw(&[1.0; LOOK_BW_VECTOR], &off, &mut untouched);
        assert_eq!(untouched, AdjustmentSet::default());
    }

    #[test]
    fn neutral_color_vector_leaves_the_tools_at_rest() {
        let mut out = AdjustmentSet::default();
        apply_look_color(&[0.5; LOOK_COLOR_VECTOR], &LookComponents::default(), &mut out);
        assert_eq!(out.exposure.exposure, 0.0);
        assert_eq!(out.exposure.brightness, 0.0);
        assert_eq!(out.exposure.contrast, 0.0);
        assert_eq!(out.exposure.saturation, 0.0);
        assert_eq!(out.hdr, crate::HighDynamicRange::default());
        assert_eq!(out.white_balance, crate::WhiteBalance::default());
        assert!(out.color_balance_gains.is_none(), "neutral offsets must not enable gains");
        for shift in &out.color_editor.basic {
            assert_eq!((shift.hue, shift.saturation, shift.lightness), (0.0, 0.0, 0.0));
        }
        // 0.5 everywhere is a flat half-tone curve, not the identity.
        assert!(out.curves.luma.points.iter().all(|p| (p.y - 0.5).abs() < 1e-6));
    }

    #[test]
    fn bw_vector_sets_the_flag_and_all_six_filters() {
        let v = [0.9, 1.0, 0.75, 0.5, 0.25, 0.0, 0.125];
        let mut out = AdjustmentSet::default();
        apply_look_bw(&v, &LookComponents::default(), &mut out);
        assert!(out.black_white.enabled, "0.9 > 0.75 enables B&W");
        let denorm = |x: f32| x * 200.0 - 100.0;
        assert_eq!(
            out.black_white.sensitivity,
            [denorm(v[1]), denorm(v[2]), denorm(v[4]), denorm(v[5]), denorm(v[3]), denorm(v[6])]
        );
        let mut off = out.clone();
        apply_look_bw(&[0.7; LOOK_BW_VECTOR], &LookComponents::default(), &mut off);
        assert!(!off.black_white.enabled, "0.7 is below the threshold");
    }

    struct FixedModel(LookPrediction);

    impl LookModel for FixedModel {
        fn predict(&self, _thumbnails: &[LookThumbnail]) -> Result<LookPrediction, LookError> {
            Ok(self.0.clone())
        }
    }

    #[test]
    fn look_adjustments_needs_a_model_and_a_frame() {
        let components = LookComponents::default();
        assert_eq!(look_adjustments(None, &[], &components), Err(LookError::NoThumbnail));

        let thumb = LookThumbnail { width: 2, height: 2, rgba: vec![0; 16] };
        assert_eq!(look_adjustments(None, &[thumb.clone()], &components), Err(LookError::NoModel));

        let short = LookThumbnail { width: 2, height: 2, rgba: vec![0; 8] };
        assert_eq!(
            look_adjustments(None, &[short], &components),
            Err(LookError::BadThumbnail { index: 0, width: 2, height: 2, bytes: 8 })
        );

        let mut color = [0.5; LOOK_COLOR_VECTOR];
        color[23] = 0.0; // saturation -100
        let model = FixedModel(LookPrediction { color, bw: [1.0; LOOK_BW_VECTOR] });
        let set = look_adjustments(Some(&model), &[thumb], &components).expect("model");
        assert_eq!(set.exposure.saturation, -100.0);
        assert!(set.black_white.enabled);
    }

    #[test]
    fn malformed_reference_is_rejected_before_inference() {
        for (width, height) in [(0, 1), (1, 1), (u32::MAX, u32::MAX)] {
            let thumb = LookThumbnail { width, height, rgba: Vec::new() };
            let error = look_adjustments(None, &[thumb], &LookComponents::default()).unwrap_err();
            assert_eq!(error, LookError::BadThumbnail { index: 0, width, height, bytes: 0 });
            assert!(error.to_string().contains("invalid RGBA"));
            let model = std::path::Path::new("unused.onnx");
            assert!(run_look_head(model, &[], width, height, 384).unwrap_err()
                .to_string().contains("non-empty RGBA"));
        }
    }

    #[test]
    fn exposure_and_brightness_use_separate_head_values() {
        let mut color = [0.5; LOOK_COLOR_VECTOR];
        color[..5].fill(0.0);
        color[23] = 0.75;
        let merged = merge_reference_color_heads(color, [0.75, 0.5, 0.5], [0.75, 0.25]);
        assert_eq!(&merged[..5], &[0.75, 0.5, 0.5, 0.75, 0.25]);
        assert_eq!(&merged[5..], &color[5..]);

        let mut out = AdjustmentSet::default();
        apply_look_color(&merged, &LookComponents::default(), &mut out);
        assert_eq!(out.exposure.exposure, 2.0);
        assert_eq!(out.exposure.brightness, 25.0);
        assert_eq!(out.exposure.contrast, -25.0);
        assert_eq!(out.exposure.saturation, 50.0);
    }

    struct SyntheticCamera;

    impl LookWhiteBalanceContext for SyntheticCamera {
        fn target_temperature(&self, target: &crate::WhiteBalance) -> (f32, f32) {
            let [kelvin, tint, _] = target.gains.expect("synthetic target gains");
            (kelvin, tint)
        }

        fn temperature_to_gains(&self, kelvin: f32, tint: f32) -> [f32; 3] {
            [kelvin, tint, 1.0]
        }
    }

    #[test]
    fn native_tone_combines_target_clamps_and_decays_each_weight() {
        let mut prediction = LookPrediction { color: [0.5; LOOK_COLOR_VECTOR], bw: [0.0; LOOK_BW_VECTOR] };
        prediction.color[..5].copy_from_slice(&[1.0, 1.0, 1.0, 1.0, 0.9]);
        let components = LookComponents { impact: 50.0, ..LookComponents::default() };
        let mut context = LookToneContext {
            subtract_baseline: [0.0; 5], add_baseline: [0.0; 5],
            weights: [None; 4], auto_white_balance: false, white_balance: &SyntheticCamera,
        };
        let mut target = AdjustmentSet::default();
        target.exposure.exposure = 3.0;
        target.exposure.brightness = 45.0;
        target.exposure.contrast = 14.0;
        target.white_balance.gains = Some([7000.0, 90.0, 1.0]);
        apply_look_tone(&prediction, &components, &mut target, &mut context);
        assert_eq!(target.exposure.exposure, 4.0);
        assert_eq!(target.exposure.brightness, 50.0);
        assert!((target.exposure.contrast - 16.15).abs() < 1e-5);
        assert_eq!(target.white_balance.gains, Some([14000.0, 100.0, 1.0]));
        assert_eq!(context.weights, [Some(0.9 * 0.7); 4]);

        target.exposure.exposure = 0.0;
        target.exposure.contrast = -14.0;
        prediction.color[4] = 0.1;
        let components = LookComponents { brightness: false, whitebalance: false, ..components };
        apply_look_tone(&prediction, &components, &mut target, &mut context);
        assert!((target.exposure.exposure - 1.26).abs() < 1e-5);
        assert!((target.exposure.contrast + 18.725).abs() < 1e-5);
        assert_eq!(target.exposure.brightness, 50.0);
        assert_eq!(context.weights[1], Some(0.9 * 0.7));
        assert_eq!(context.weights[2], Some(0.9 * 0.7));

        target.exposure.contrast = -49.0;
        apply_look_tone(&prediction, &components, &mut target, &mut context);
        assert_eq!(target.exposure.contrast, -50.0);
    }

    #[test]
    fn native_tone_uses_component_baselines_and_temperature_space() {
        let mut prediction = LookPrediction { color: [0.5; LOOK_COLOR_VECTOR], bw: [0.0; LOOK_BW_VECTOR] };
        prediction.color[0] = 1.0;
        let mut context = LookToneContext {
            subtract_baseline: [2.0, 0.0, 0.0, 0.0, 0.0],
            add_baseline: [1.0, (13200.0 / 173.0) * 15.0, 5.0, 0.0, 0.0],
            weights: [Some(1.0); 4], auto_white_balance: false, white_balance: &SyntheticCamera,
        };
        let components = LookComponents { impact: 50.0, ..LookComponents::default() };
        let mut target = AdjustmentSet::default();
        target.white_balance.gains = Some([6750.0, 10.0, 1.0]);
        apply_look_tone(&prediction, &components, &mut target, &mut context);
        assert_eq!(target.exposure.exposure, 2.0);
        assert!((target.white_balance.kelvin - 8000.0).abs() < 0.01);
        assert_eq!(target.white_balance.tint, 15.0);

        context.auto_white_balance = true;
        context.weights = [Some(1.0); 4];
        target.exposure.exposure = 0.0;
        apply_look_tone(&prediction, &components, &mut target, &mut context);
        assert!((target.exposure.exposure - 0.6).abs() < 1e-5);
        assert!((target.white_balance.tint - 15.2).abs() < 1e-5);
    }

    #[test]
    fn descriptor_cosine_and_native_similarity_weights() {
        let mut left = [0.0; 1280];
        let mut right = [0.0; 1280];
        left[0] = 2.0;
        right[0] = -3.0;
        assert_eq!(look_descriptor_similarity(&left, &right), Some(-1.0));
        right[0] = 0.0;
        right[1] = 3.0;
        assert_eq!(look_descriptor_similarity(&left, &right), Some(0.0));
        assert_eq!(look_descriptor_similarity(&left, &[0.0; 1280]), None);

        assert_eq!(look_reference_weights(&[0.25, 0.75]), Some(vec![0.25, 0.75]));
        assert_eq!(look_reference_weights(&[-0.5, 0.0, 0.5]), Some(vec![0.0, 1.0 / 3.0, 2.0 / 3.0]));
        assert_eq!(look_reference_weights(&[-0.5, -0.5]), Some(vec![1.0, 1.0]));
        assert_eq!(look_reference_weights(&[0.0]), Some(vec![1.0]));
        assert_eq!(look_reference_weights(&[f32::NAN]), None);
    }

    #[test]
    fn weighted_reference_tone_merges_decoded_slots_before_applying_once() {
        let frames = [
            LookThumbnail { width: 1, height: 1, rgba: vec![0, 0, 0, 255] },
            LookThumbnail { width: 1, height: 1, rgba: vec![255, 0, 0, 255] },
        ];
        let predict = |frame: &LookThumbnail| {
            let value = f32::from(frame.rgba[0]) / 255.0;
            Some(([value, 0.5, 0.5], [value, 0.5]))
        };
        assert_eq!(merge_reference_tone(&frames[..1], &[0.25], predict), Some([-4.0, 0.0, 0.0, -50.0, 0.0]));
        assert_eq!(merge_reference_tone(&frames, &[0.25, 0.75], predict), Some([2.0, 0.0, 0.0, 25.0, 0.0]));
        assert_eq!(merge_reference_tone(&frames, &[0.0, 0.0], |_| Some(([1.0, 0.5, 0.5], [1.0, 0.5]))),
            Some([8.0, 0.0, 0.0, 100.0, 0.0]));
        assert_eq!(merge_reference_tone(&frames, &[0.25], predict), None);

        let mut target = AdjustmentSet::default();
        target.exposure.exposure = 1.0;
        let components = LookComponents { whitebalance: false, impact: 50.0, ..LookComponents::default() };
        let mut context = LookToneContext {
            subtract_baseline: [0.0; 5], add_baseline: [0.0; 5], weights: [None; 4],
            auto_white_balance: false, white_balance: &SyntheticCamera,
        };
        let decoded = merge_reference_tone(&frames, &[0.25, 0.75], predict).unwrap();
        apply_decoded_look_tone(decoded, &components, &mut target, &mut context);
        assert!((target.exposure.exposure - 1.9).abs() < 1e-5);
        assert!((target.exposure.brightness - 5.625).abs() < 1e-5);
        assert_eq!(context.weights[0], Some(0.9 * 0.7));
    }

    #[test]
    fn descriptor_input_resizes_to_nhwc_rgb_with_model_scale() {
        let mut frame = LookThumbnail {
            width: 2, height: 1, rgba: vec![10, 20, 30, 0, 100, 120, 150, 255],
        };
        let input = similarity_input(&frame).unwrap();
        assert_eq!(input.len(), 224 * 224 * 3);
        assert_eq!(&input[..3], &[10.0, 20.0, 30.0]);
        assert_eq!(&input[input.len() - 3..], &[100.0, 120.0, 150.0]);
        frame.rgba.pop();
        assert!(similarity_input(&frame).is_none());
    }

    #[test]
    fn target_descriptor_selects_different_reference_tones() {
        let frames = [
            LookThumbnail { width: 1, height: 1, rgba: vec![255, 0, 0, 255] },
            LookThumbnail { width: 1, height: 1, rgba: vec![0, 0, 255, 255] },
        ];
        let describe = |frame: &LookThumbnail| {
            let mut values = [0.0; 1280];
            values[0] = f32::from(frame.rgba[0]);
            values[1] = f32::from(frame.rgba[2]);
            Some(values)
        };
        let predict = |frame: &LookThumbnail| {
            let value = f32::from(frame.rgba[0]) / 255.0;
            Some(([value, 0.5, 0.5], [0.5, 0.5]))
        };
        let components = LookComponents { whitebalance: false, ..LookComponents::default() };
        for (i, want) in [(0, 3.6), (1, -3.6)] {
            let scores = score_reference_frames(&frames, &frames[i], describe).unwrap();
            assert_eq!(scores[i], 1.0);
            assert_eq!(scores[1 - i], 0.0);
            let decoded = merge_reference_tone(&frames, &scores, predict).unwrap();
            let mut target = AdjustmentSet::default();
            let mut context = LookToneContext {
                subtract_baseline: [0.0; 5], add_baseline: [0.0; 5], weights: [None; 4],
                auto_white_balance: false, white_balance: &SyntheticCamera,
            };
            apply_decoded_look_tone(decoded, &components, &mut target, &mut context);
            assert!((target.exposure.exposure - want).abs() < 1e-5);
        }
    }

    #[test]
    #[ignore = "needs installed Similarity and Match Look Exp models"]
    fn installed_descriptor_model_applies_weighted_tone() {
        let red = LookThumbnail { width: 2, height: 1, rgba: vec![200, 25, 20, 255, 235, 50, 30, 255] };
        let blue = LookThumbnail { width: 2, height: 1, rgba: vec![20, 25, 200, 255, 30, 50, 235, 255] };
        let reference = LookReference {
            thumbnails: vec![red.clone(), blue],
            components: LookComponents {
                whitebalance: false, brightness: false, contrast: false, ..LookComponents::default()
            },
        };
        let mut target = AdjustmentSet::default();
        let mut context = LookToneContext {
            subtract_baseline: [0.0; 5], add_baseline: [0.0; 5], weights: [None; 4],
            auto_white_balance: false, white_balance: &SyntheticCamera,
        };
        apply_similarity_weighted_reference_tone(&reference, &red, &mut target, &mut context)
            .expect("descriptor model and both reference predictions");
        assert!(target.exposure.exposure.is_finite());
        assert_eq!(context.weights, [Some(0.9 * 0.7), None, None, None]);
    }

    #[test]
    fn color_head_runs_on_a_synthetic_frame() {
        let model = std::path::Path::new(
            r"C:\Program Files\Capture One\Capture One\AIModels\Common\MatchLook\MatchLookColor.onnx",
        );
        if !model.is_file() || crate::aimask::Runtime::get().is_none() {
            return;
        }
        let rgba: Vec<u8> = (0..32 * 32)
            .flat_map(|i| [(i % 251) as u8, 128, 200, 255])
            .collect();
        let out = run_look_head(model, &rgba, 32, 32, 384).expect("color head");
        assert_eq!(out.len(), LOOK_COLOR_VECTOR);
        assert!(out.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)), "{out:?}");
    }
}
