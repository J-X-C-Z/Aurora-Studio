//! Skin-adjust family, vendor table rows 39-50.
//!
//! Four recovered pixel laws live here, each an explicit CPU reference port:
//! the wrinkle family (rows 39-42), the neutral-grey SkinAdjustV2 HSL
//! projection (row 46) with its 512 light/dark consumer (row 47), the
//! high-low frequency family (rows 49-50), and the shared mask composer
//! consumers (rows 43-45).
//!
//! Row 48 (`PF90001`) is a contract only. Its model-to-source blend is not
//! statically recovered, so it stays a typed refusal instead of a guess.
//!
//! Masks, planes, model outputs and uniforms are same-job inputs. A missing or
//! misshapen input is a typed refusal; nothing is synthesized here.

use crate::kernels::mask_compose::{apply as compose_apply, MaskComposeMode};
use crate::{Access, EffectMeta, Image, Refusal};

/// Rows whose pixel law is executable here.
pub const META: &[EffectMeta] = &[
    EffectMeta {
        pf: 21025,
        key: "",
        label: "Puppet Wrinkle",
        access: Access::Cpu,
        inputs: &["base", "residual", "mask_rgba", "skin_mask"],
    },
    EffectMeta {
        pf: 21069,
        key: "",
        label: "Cheek Wrinkle",
        access: Access::Cpu,
        inputs: &["base", "residual", "mask_rgba", "skin_mask"],
    },
    EffectMeta {
        pf: 21070,
        key: "",
        label: "Nose Back Wrinkle",
        access: Access::Cpu,
        inputs: &["base", "residual", "mask_rgba", "skin_mask"],
    },
    EffectMeta {
        pf: 21071,
        key: "",
        label: "Perioral Wrinkle",
        access: Access::Cpu,
        inputs: &["base", "residual", "mask_rgba", "skin_mask"],
    },
    EffectMeta {
        pf: 30013,
        key: "neutralGrayFlattening",
        label: "Neutral Grey Flat",
        access: Access::Cpu,
        inputs: &["original", "current", "mask"],
    },
    EffectMeta {
        pf: 30014,
        key: "",
        label: "Neutral Grey 3D",
        access: Access::Cpu,
        inputs: &["source_bgr512", "light", "dark", "alpha", "mask"],
    },
    EffectMeta {
        pf: 30021,
        key: "freqSplit_face",
        label: "HLF Face",
        access: Access::Cpu,
        inputs: &["source", "median", "gauss", "face_mask", "skin_mask"],
    },
    EffectMeta {
        pf: 30022,
        key: "freqSplit_body",
        label: "HLF Body",
        access: Access::Cpu,
        inputs: &["source", "median", "gauss", "face_mask", "skin_mask"],
    },
    EffectMeta {
        pf: 90008,
        key: "Y.removeStretchMark",
        label: "Stretch Mark",
        access: Access::Cpu,
        inputs: &["model_output", "mask"],
    },
    EffectMeta {
        pf: 90089,
        key: "SpotlessBody_Baby",
        label: "Body Blemish Child",
        access: Access::Cpu,
        inputs: &["model_output", "mask"],
    },
    EffectMeta {
        pf: 100008,
        key: "Y.repairBreast",
        label: "Repair Breast",
        access: Access::Cpu,
        inputs: &["model_output", "mask"],
    },
];

/// Rows that are only a recovered contract. `PF90001` gates and validates, but
/// refuses to emit pixels because its model-output blend is not recovered.
pub const REFUSED: &[EffectMeta] = &[EffectMeta {
    pf: 90001,
    key: "textureSmoothSkin",
    label: "Texture Smooth Skin",
    access: Access::Refused,
    inputs: &[
        "ui degree 0..100",
        "per-face engine table (non-empty, finite)",
        "RGBA8 source",
        "ROI-tight u8 mask",
        "roi + affine[6]",
        "SkinDBPipeline service (external merge)",
    ],
}];

const INV255: f32 = 1.0 / 255.0;

fn clamp(value: f32, lo: f32, hi: f32) -> f32 {
    value.max(lo).min(hi)
}

fn mix(from: f32, to: f32, amount: f32) -> f32 {
    from * (1.0 - amount) + to * amount
}

/// `value * 255` truncated toward zero, matching the vendor `cvttss2si` store.
fn trunc_u8(value: f32) -> u8 {
    if value <= 0.0 {
        0
    } else if value >= 1.0 {
        255
    } else {
        (value * 255.0) as u8
    }
}

/// Inclusive-exclusive pixel rectangle inside one frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Roi {
    fn validate(self, width: u32, height: u32) -> Result<(), Refusal> {
        let inside = self.width > 0
            && self.height > 0
            && self.x as u64 + self.width as u64 <= width as u64
            && self.y as u64 + self.height as u64 <= height as u64;
        if inside {
            Ok(())
        } else {
            Err(Refusal::ValueOutOfRange("roi"))
        }
    }
}

fn same_shape(a: &Image, b: &Image, field: &'static str) -> Result<(), Refusal> {
    a.validate()?;
    b.validate()?;
    if a.width == b.width && a.height == b.height {
        Ok(())
    } else {
        Err(Refusal::ShapeMismatch(field))
    }
}

// ---------------------------------------------------------------------------
// Rows 39-42: wrinkle family over the recovered FacialSmoothProcessV2 composite.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WrinkleUniforms {
    pub legal_alpha: f32,
    pub dark_circle_alpha: f32,
    pub diff_factor: f32,
    pub diff_factor2: f32,
    pub color_threshold_factor: f32,
    pub mix_alpha: f32,
    pub diff_min_threshold: f32,
    pub diff_max_threshold: f32,
    pub high_freq_min_threshold: f32,
    pub high_freq_max_threshold: f32,
}

impl WrinkleUniforms {
    fn validate(&self) -> Result<(), Refusal> {
        let finite = [
            self.legal_alpha,
            self.dark_circle_alpha,
            self.diff_factor,
            self.diff_factor2,
            self.color_threshold_factor,
            self.mix_alpha,
            self.diff_min_threshold,
            self.diff_max_threshold,
            self.high_freq_min_threshold,
            self.high_freq_max_threshold,
        ]
        .iter()
        .all(|value| value.is_finite());
        if !finite
            || self.diff_min_threshold > self.diff_max_threshold
            || self.high_freq_min_threshold > self.high_freq_max_threshold
        {
            return Err(Refusal::ValueOutOfRange("uniforms"));
        }
        Ok(())
    }
}

#[cfg(test)]
fn neutral_wrinkle_uniforms() -> WrinkleUniforms {
    WrinkleUniforms {
        legal_alpha: 1.0,
        dark_circle_alpha: 1.0,
        diff_factor: 1.0,
        diff_factor2: 0.0,
        color_threshold_factor: 0.0,
        mix_alpha: 1.0,
        diff_min_threshold: -1.0,
        diff_max_threshold: 1.0,
        high_freq_min_threshold: -1.0,
        high_freq_max_threshold: 1.0,
    }
}

/// `PF21070`/`PF21025` scale UI by 100; `PF21069`/`PF21071` keep the native
/// unscaled getter value.
pub fn wrinkle_degree(pf: u32, ui: f32) -> Result<f32, Refusal> {
    if !matches!(pf, 21025 | 21069 | 21070 | 21071) {
        return Err(Refusal::UnknownEffect(pf));
    }
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    Ok(if matches!(pf, 21025 | 21070) { ui / 100.0 } else { ui })
}

/// `alpha = max(dark_circle * mask.r, legal * mask.b, 0) * skin_mask`, skipping
/// pixels whose alpha is not positive. Alpha is never recomputed from output.
pub fn wrinkle_apply(
    source: &Image,
    base: &Image,
    residual: &Image,
    mask_rgba: &[u8],
    skin_mask: &[u8],
    roi: Roi,
    degree: f32,
    uniforms: WrinkleUniforms,
) -> Result<Image, Refusal> {
    same_shape(source, base, "base")?;
    same_shape(source, residual, "residual")?;
    roi.validate(source.width, source.height)?;
    source.validate()?;
    let pixels = source.pixel_count();
    if mask_rgba.len() != pixels * 4 {
        return Err(Refusal::ShapeMismatch("mask_rgba"));
    }
    if skin_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("skin_mask"));
    }
    if !degree.is_finite() {
        return Err(Refusal::ValueOutOfRange("degree"));
    }
    uniforms.validate()?;

    let mut out = source.clone();
    if degree <= 0.001 {
        return Ok(out);
    }

    for y in roi.y..roi.y + roi.height {
        for x in roi.x..roi.x + roi.width {
            let index = (y * source.width + x) as usize;
            let offset = index * 4;
            let skin = skin_mask[index] as f32 * INV255;
            let alpha = (uniforms.dark_circle_alpha * (mask_rgba[offset] as f32 * INV255))
                .max(uniforms.legal_alpha * (mask_rgba[offset + 2] as f32 * INV255))
                .max(0.0)
                * skin;
            if !(alpha > 0.0) {
                continue;
            }
            for channel in 0..3 {
                let src = source.pixels[offset + channel] as f32 * INV255;
                let base_value = base.pixels[offset + channel] as f32 * INV255;
                let residual_value = residual.pixels[offset + channel] as f32 * INV255;
                let high = src - base_value;
                let diff = clamp(
                    (residual_value - base_value) * uniforms.diff_factor
                        + residual_value * uniforms.diff_factor2,
                    uniforms.diff_min_threshold,
                    uniforms.diff_max_threshold,
                );
                let threshold = mix(
                    residual_value * uniforms.color_threshold_factor,
                    residual_value,
                    uniforms.mix_alpha,
                );
                let upper = threshold.max(src - high);
                let reference = clamp(src + diff - high, 0.0, upper) + high;
                let delta = clamp(
                    mix(src, reference, alpha) - src,
                    uniforms.high_freq_min_threshold,
                    uniforms.high_freq_max_threshold,
                );
                out.pixels[offset + channel] = trunc_u8(src + delta);
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Rows 43-45: the shared 0x5B8CEA0 composer over an explicit model output.
// The arithmetic itself is `kernels::mask_compose`; this only adds the ROI
// writeback that the three rows share.
// ---------------------------------------------------------------------------

/// Replaces ROI pixels of `source` with the shared-stage composition of
/// `model_output`. Outside-ROI pixels and alpha are preserved.
pub fn masked_model_output(
    source: &Image,
    model_output: &Image,
    roi: Roi,
    roi_mask: &[u8],
    alternate: bool,
) -> Result<Image, Refusal> {
    same_shape(source, model_output, "model_output")?;
    roi.validate(source.width, source.height)?;
    if roi_mask.len() != (roi.width * roi.height) as usize {
        return Err(Refusal::ShapeMismatch("roi_mask"));
    }

    // The shared stage takes a full-frame mask; zeros outside the ROI keep the
    // composition an identity there.
    let mut full_mask = vec![0u8; source.pixel_count()];
    for row in 0..roi.height {
        let dst = ((roi.y + row) * source.width + roi.x) as usize;
        let src = (row * roi.width) as usize;
        full_mask[dst..dst + roi.width as usize]
            .copy_from_slice(&roi_mask[src..src + roi.width as usize]);
    }
    let mode = if alternate {
        MaskComposeMode::Alternate
    } else {
        MaskComposeMode::MaskOnly
    };
    let composed = compose_apply(model_output, &full_mask, mode)?;

    let mut out = source.clone();
    let row_bytes = roi.width as usize * 4;
    for row in 0..roi.height {
        let dst = (((roi.y + row) * source.width + roi.x) * 4) as usize;
        out.pixels[dst..dst + row_bytes]
            .copy_from_slice(&composed.pixels[dst..dst + row_bytes]);
    }
    Ok(out)
}

/// `PF90089` gate: UI range, then an explicit native per-face degree, then the
/// strict `degree > 0.001` predicate.
pub fn body_blemish_degree(ui: f32, native_per_face: Option<f32>) -> Result<(f32, bool), Refusal> {
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if ui == 0.0 {
        return Ok((0.0, false));
    }
    let degree = native_per_face.ok_or(Refusal::MissingInput("native_per_face_degree"))?;
    if !degree.is_finite() {
        return Err(Refusal::ValueOutOfRange("native_per_face_degree"));
    }
    Ok((degree, degree > 0.001))
}

/// `PF100008` gate: ROI value resolution, then the strict `> 0.001` render
/// predicate. Non-zero UI without its live inputs is a refusal.
pub fn breast_degree(
    ui: f32,
    roi_value: Option<f32>,
    has_body_skeleton: bool,
    has_roi_points: bool,
) -> Result<(f32, bool), Refusal> {
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if ui == 0.0 {
        return Ok((0.0, false));
    }
    if !has_body_skeleton {
        return Err(Refusal::MissingInput("body_skeleton"));
    }
    if !has_roi_points {
        return Err(Refusal::MissingInput("roi_points"));
    }
    let raw = match roi_value {
        Some(value) if value.is_finite() => value,
        Some(_) => return Err(Refusal::ValueOutOfRange("roi_value")),
        None => return Err(Refusal::MissingInput("roi_value")),
    };
    Ok((clamp(raw / 100.0, 0.0, 1.0), raw > 0.001))
}

/// `PF90008` degree: `strength / 100` with the strict `> 0.001` render gate.
pub fn stretch_mark_degree(strength: f32) -> Result<(f32, bool), Refusal> {
    if !strength.is_finite() || !(0.0..=100.0).contains(&strength) {
        return Err(Refusal::ValueOutOfRange("strength"));
    }
    let degree = strength / 100.0;
    Ok((degree, degree > 0.001))
}

// ---------------------------------------------------------------------------
// Row 46: SkinAdjustV2 SKIN_GREY HSL projection (PF30013 callable branch).
// ---------------------------------------------------------------------------

const HSL_EPSILON: f32 = 0.001;

#[derive(Clone, Copy)]
struct Hsl {
    h: f32,
    s: f32,
    l: f32,
}

fn rgb_to_hsl(r: f32, g: f32, b: f32) -> Hsl {
    let base_max = if g < b { b } else { g };
    let max = r.max(base_max);
    let min = r.min(if g < b { g } else { b });
    let c = max - min;
    let (w, offset) = if r < base_max {
        (r, if g < b { 2.0 / 3.0 } else { -1.0 / 3.0 })
    } else {
        (base_max, if g < b { -1.0 } else { 0.0 })
    };
    let h = ((w - min) / (6.0 * c + HSL_EPSILON) + offset).abs();
    let l = max - c * 0.5;
    let s = c / (1.0 - (l * 2.0 - 1.0).abs() + HSL_EPSILON);
    Hsl { h, s, l }
}

fn hsl_to_rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    let x = h * 6.0;
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let axis = [
        ((x - 3.0).abs() - 1.0).clamp(0.0, 1.0),
        (2.0 - (x - 2.0).abs()).clamp(0.0, 1.0),
        (2.0 - (x - 4.0).abs()).clamp(0.0, 1.0),
    ];
    axis.map(|value| ((value - 0.5) * c + l).clamp(0.0, 1.0))
}

/// `out = current + (HSLToRGB(original.H, original.S, current.L) - current) * alpha`
/// with `alpha = clamp(mask / 255 * alpha_arg, 0, 1)` and a `0.001` skip.
pub fn skin_grey_apply(
    original: &Image,
    current: &Image,
    mask: &[u8],
    roi: Roi,
    alpha_arg: f32,
) -> Result<Image, Refusal> {
    same_shape(original, current, "current")?;
    roi.validate(current.width, current.height)?;
    if mask.len() != current.pixel_count() {
        return Err(Refusal::ShapeMismatch("mask"));
    }
    if !alpha_arg.is_finite() || !(0.0..=1.0).contains(&alpha_arg) {
        return Err(Refusal::ValueOutOfRange("alpha"));
    }

    let mut out = current.clone();
    for y in roi.y..roi.y + roi.height {
        for x in roi.x..roi.x + roi.width {
            let index = (y * current.width + x) as usize;
            let offset = index * 4;
            let alpha = clamp(mask[index] as f32 * INV255 * alpha_arg, 0.0, 1.0);
            if alpha <= 0.001 {
                continue;
            }
            let source = rgb_to_hsl(
                original.pixels[offset] as f32 * INV255,
                original.pixels[offset + 1] as f32 * INV255,
                original.pixels[offset + 2] as f32 * INV255,
            );
            let target = rgb_to_hsl(
                current.pixels[offset] as f32 * INV255,
                current.pixels[offset + 1] as f32 * INV255,
                current.pixels[offset + 2] as f32 * INV255,
            );
            let projected = hsl_to_rgb(source.h, source.s, target.l);
            for channel in 0..3 {
                let now = current.pixels[offset + channel] as f32 * INV255;
                out.pixels[offset + channel] =
                    ((now + (projected[channel] - now) * alpha).clamp(0.0, 1.0) * 255.0)
                        .round() as u8;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Row 47: neutral-grey 512 light/dark consumer (PF30014).
// ---------------------------------------------------------------------------

pub const NEUTRAL_GRAY_3D_SIDE: usize = 512;
pub const NEUTRAL_GRAY_3D_PIXELS: usize = NEUTRAL_GRAY_3D_SIDE * NEUTRAL_GRAY_3D_SIDE;
pub const NEUTRAL_GRAY_3D_BYTES: usize = NEUTRAL_GRAY_3D_PIXELS * 3;
const NEUTRAL_GRAY_3D_GATE: f32 = 0.01;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NeutralGray3dDegrees {
    pub stereoscopic: f32,
    pub facial: f32,
    pub contour: f32,
    pub active: bool,
}

/// `ui/100` for 30014, 30015 and 30016; active needs 30014 above 0.01 and at
/// least one of 30015/30016 above 0.01.
pub fn neutral_gray_3d_degrees(
    ui30014: f32,
    ui30015: f32,
    ui30016: f32,
) -> Result<NeutralGray3dDegrees, Refusal> {
    let degree = |ui: f32| -> Result<f32, Refusal> {
        if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
            return Err(Refusal::ValueOutOfRange("ui"));
        }
        Ok(ui / 100.0)
    };
    let stereoscopic = degree(ui30014)?;
    let facial = degree(ui30015)?;
    let contour = degree(ui30016)?;
    let active = stereoscopic > NEUTRAL_GRAY_3D_GATE
        && (facial > NEUTRAL_GRAY_3D_GATE || contour > NEUTRAL_GRAY_3D_GATE);
    Ok(NeutralGray3dDegrees {
        stereoscopic,
        facial,
        contour,
        active,
    })
}

/// `source + (light - dark) * 127.5 * (alpha * mask)`, round-half-even, clamp.
pub fn neutral_gray_3d_apply(
    source_bgr: &[u8],
    light: &[f32],
    dark: &[f32],
    alpha: &[f32],
    mask: &[f32],
    degrees: NeutralGray3dDegrees,
) -> Result<Vec<u8>, Refusal> {
    if source_bgr.len() != NEUTRAL_GRAY_3D_BYTES {
        return Err(Refusal::ShapeMismatch("source_bgr512"));
    }
    if light.len() != NEUTRAL_GRAY_3D_BYTES || dark.len() != NEUTRAL_GRAY_3D_BYTES {
        return Err(Refusal::ShapeMismatch("light/dark"));
    }
    if alpha.len() != NEUTRAL_GRAY_3D_PIXELS || mask.len() != NEUTRAL_GRAY_3D_PIXELS {
        return Err(Refusal::ShapeMismatch("alpha/mask"));
    }
    let valid_plane = light
        .iter()
        .chain(dark.iter())
        .all(|value| value.is_finite() && *value >= 0.0);
    if !valid_plane {
        return Err(Refusal::ValueOutOfRange("light/dark"));
    }
    let valid_unit = alpha
        .iter()
        .chain(mask.iter())
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value));
    if !valid_unit {
        return Err(Refusal::ValueOutOfRange("alpha/mask"));
    }
    if !degrees.active {
        return Ok(source_bgr.to_vec());
    }

    let mut out = Vec::with_capacity(NEUTRAL_GRAY_3D_BYTES);
    for pixel in 0..NEUTRAL_GRAY_3D_PIXELS {
        let weight = alpha[pixel] * mask[pixel];
        for channel in 0..3 {
            let index = pixel * 3 + channel;
            let encoded =
                source_bgr[index] as f32 + (light[index] - dark[index]) * 127.5 * weight;
            out.push(encoded.round_ties_even().clamp(0.0, 255.0) as u8);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Rows 49-50: high-low frequency retouch.
// ---------------------------------------------------------------------------

const HLF_HIGH_GAIN: f32 = 7.07;
const HLF_MEAN_VAR_CAP: f32 = 1.0;
const HLF_VARIANCE_EPSILON: f32 = 0.1;
const HLF_MIX_FACTOR: f32 = 0.9;
const HLF_PROBE_SCALE: f32 = 4.0;
const HLF_PROBE_BIAS: f32 = 0.2;
const HLF_GAUSS_BIAS: f32 = 0.1;
const HLF_UHF: f32 = 0.01;
const HLF_DIFF_MIN: f32 = -12.0 / 255.0;
const HLF_DIFF_MAX: f32 = 14.0 / 255.0;

fn hlf_pixel_weight(
    source: [f32; 3],
    median: [f32; 3],
    gauss: [f32; 3],
    degree: f32,
) -> f32 {
    let mut sum_variance = 0.0;
    for channel in 0..3 {
        let v = (source[channel] - median[channel]) * HLF_HIGH_GAIN;
        sum_variance += (v * v).min(HLF_MEAN_VAR_CAP);
    }
    let mean_variance = sum_variance / 3.0;
    let weight_factor = mix(
        1.0,
        1.0 - mean_variance / (mean_variance + HLF_VARIANCE_EPSILON),
        HLF_MIX_FACTOR,
    );
    let probe = clamp(
        (source[0].min(gauss[0] - HLF_GAUSS_BIAS) - HLF_PROBE_BIAS) * HLF_PROBE_SCALE,
        0.0,
        1.0,
    );
    weight_factor * probe * degree
}

/// PF30021: the ROI is validated but the vendor fragment covers the whole
/// frame, so every pixel is evaluated. Masks are `skin_mask * face_mask`.
#[allow(clippy::too_many_arguments)]
pub fn hlf_face_apply(
    source: &[f32],
    median: &[f32],
    gauss: &[f32],
    face_mask: &[f32],
    skin_mask: &[f32],
    width: u32,
    height: u32,
    roi: Roi,
    ui: f32,
) -> Result<Vec<f32>, Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ValueOutOfRange("width/height"));
    }
    roi.validate(width, height)?;
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    let pixels = width as usize * height as usize;
    for (plane, field) in [
        (source, "source"),
        (median, "median"),
        (gauss, "gauss"),
    ] {
        if plane.len() != pixels * 3 {
            return Err(Refusal::ShapeMismatch(field));
        }
        if !plane.iter().all(|value| value.is_finite()) {
            return Err(Refusal::ValueOutOfRange(field));
        }
    }
    for (plane, field) in [(face_mask, "face_mask"), (skin_mask, "skin_mask")] {
        if plane.len() != pixels {
            return Err(Refusal::ShapeMismatch(field));
        }
        if !plane.iter().all(|value| value.is_finite()) {
            return Err(Refusal::ValueOutOfRange(field));
        }
    }

    let degree = ui / 100.0;
    let mut out = source.to_vec();
    if degree == 0.0 {
        return Ok(out);
    }

    for pixel in 0..pixels {
        let base = pixel * 3;
        let src = [source[base], source[base + 1], source[base + 2]];
        let med = [median[base], median[base + 1], median[base + 2]];
        let gau = [gauss[base], gauss[base + 1], gauss[base + 2]];
        let weight = hlf_pixel_weight(src, med, gau, degree);
        let coverage = skin_mask[pixel] * face_mask[pixel];
        for channel in 0..3 {
            let high = src[channel] - med[channel];
            let reference = mix(src[channel], gau[channel], weight) + high * HLF_UHF;
            let diff = clamp(reference - src[channel], HLF_DIFF_MIN, HLF_DIFF_MAX);
            out[base + channel] = src[channel] + diff * coverage;
        }
    }
    Ok(out)
}

/// The eight native PF30022 uniforms, in fragment order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyHlfUniforms {
    pub spotless_degree: f32,
    pub retouch_degree: f32,
    pub spotless_acne_threshold: f32,
    pub spotless_high_freq_factor: f32,
    pub spotless_low_freq_alpha: f32,
    pub retouch_high_freq_factor: f32,
    pub diff_min_range: f32,
    pub diff_max_range: f32,
}

impl BodyHlfUniforms {
    fn validate(&self) -> Result<(), Refusal> {
        let values = [
            self.spotless_degree,
            self.retouch_degree,
            self.spotless_acne_threshold,
            self.spotless_high_freq_factor,
            self.spotless_low_freq_alpha,
            self.retouch_high_freq_factor,
            self.diff_min_range,
            self.diff_max_range,
        ];
        if !values.iter().all(|value| value.is_finite()) || self.diff_min_range > self.diff_max_range
        {
            return Err(Refusal::ValueOutOfRange("uniforms"));
        }
        Ok(())
    }
}

/// PF30022: body ROI writeback. Pixels outside the ROI and every alpha byte are
/// preserved. The adjust LUT is a required live input when spotless is active.
pub fn body_hlf_apply(
    source: &Image,
    roi: Roi,
    median_rgb: &[f32],
    gauss_rgb: &[f32],
    face_mask_rgba: &[f32],
    skin_mask: &[f32],
    ui: f32,
    uniforms: BodyHlfUniforms,
    adjust_lut: Option<&dyn Fn(f32) -> [f32; 3]>,
) -> Result<Image, Refusal> {
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    source.validate()?;
    let pixels = source.pixel_count();
    for (plane, field) in [(median_rgb, "median_rgb"), (gauss_rgb, "gauss_rgb")] {
        if plane.len() != pixels * 3 {
            return Err(Refusal::ShapeMismatch(field));
        }
        if !plane.iter().all(|value| value.is_finite()) {
            return Err(Refusal::ValueOutOfRange(field));
        }
    }
    for (plane, field) in [(face_mask_rgba, "face_mask_rgba"), (skin_mask, "skin_mask")] {
        if !plane.iter().all(|value| value.is_finite()) {
            return Err(Refusal::ValueOutOfRange(field));
        }
    }
    if face_mask_rgba.len() != pixels * 4 {
        return Err(Refusal::ShapeMismatch("face_mask_rgba"));
    }
    if skin_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("skin_mask"));
    }
    uniforms.validate()?;

    let degree = ui / 100.0;
    let mut out = source.clone();
    if !(degree > 0.0) {
        return Ok(out);
    }
    roi.validate(source.width, source.height)?;
    if uniforms.spotless_degree > 0.0 && adjust_lut.is_none() {
        return Err(Refusal::MissingInput("adjust_lut"));
    }

    for y in roi.y..roi.y + roi.height {
        for x in roi.x..roi.x + roi.width {
            let index = (y * source.width + x) as usize;
            let offset = index * 4;
            let face = [
                face_mask_rgba[index * 4],
                face_mask_rgba[index * 4 + 1],
                face_mask_rgba[index * 4 + 2],
            ];
            if face[0] <= 0.001 {
                continue;
            }
            let src = [
                source.pixels[offset] as f32 * INV255,
                source.pixels[offset + 1] as f32 * INV255,
                source.pixels[offset + 2] as f32 * INV255,
            ];
            let med = [
                median_rgb[index * 3],
                median_rgb[index * 3 + 1],
                median_rgb[index * 3 + 2],
            ];
            let gau = [
                gauss_rgb[index * 3],
                gauss_rgb[index * 3 + 1],
                gauss_rgb[index * 3 + 2],
            ];
            let high = [src[0] - med[0], src[1] - med[1], src[2] - med[2]];
            let mut result_high = high;
            let mut reference = src;

            if uniforms.spotless_degree > 0.0 {
                let lut = adjust_lut.expect("checked above");
                let probe = [
                    high[0] * 0.5 + 0.5,
                    high[1] * 0.5 + 0.5,
                    high[2] * 0.5 + 0.5,
                ];
                let common = lut(probe[0])[0];
                let channels = lut(common);
                let adjusted = [
                    (channels[0] - 0.5) * 2.0,
                    (channels[1] - 0.5) * 2.0,
                    (channels[2] - 0.5) * 2.0,
                ];
                let magnitude = (adjusted[0] * adjusted[0]
                    + adjusted[1] * adjusted[1]
                    + adjusted[2] * adjusted[2])
                    .sqrt()
                    * 0.577_350_26;
                let darkest = src[0].min(src[1]).min(src[2]);
                if magnitude > uniforms.spotless_acne_threshold || darkest < 0.01 {
                    let spotless = [
                        high[0] * uniforms.spotless_high_freq_factor,
                        high[1] * uniforms.spotless_high_freq_factor,
                        high[2] * uniforms.spotless_high_freq_factor,
                    ];
                    let amount = uniforms.spotless_degree * face[1];
                    result_high = [
                        mix(high[0], spotless[0], amount),
                        mix(high[1], spotless[1], amount),
                        mix(high[2], spotless[2], amount),
                    ];
                    let low_alpha = (uniforms.spotless_low_freq_alpha + 0.2).min(1.0);
                    let weight = low_alpha * uniforms.spotless_degree * face[1];
                    reference = [
                        mix(med[0], gau[0], weight) + result_high[0],
                        mix(med[1], gau[1], weight) + result_high[1],
                        mix(med[2], gau[2], weight) + result_high[2],
                    ];
                }
            }

            if uniforms.retouch_degree > 0.0 {
                let mut variance = [0.0f32; 3];
                for channel in 0..3 {
                    let v = (result_high[channel] * HLF_HIGH_GAIN).min(1.0);
                    variance[channel] = (v * v).min(1.0);
                }
                let probe = clamp(
                    (src[0].min(gau[0] - HLF_GAUSS_BIAS) - HLF_PROBE_BIAS) * HLF_PROBE_SCALE,
                    0.0,
                    1.0,
                );
                let mean_variance = (variance[0] + variance[1] + variance[2]) / 3.0;
                let weight_factor = mix(
                    1.0,
                    1.0 - mean_variance / (mean_variance + HLF_VARIANCE_EPSILON),
                    HLF_MIX_FACTOR,
                );
                let weight = weight_factor * probe * uniforms.retouch_degree;
                for channel in 0..3 {
                    reference[channel] = mix(reference[channel], gau[channel], weight)
                        + result_high[channel] * uniforms.retouch_high_freq_factor;
                }
            }

            let coverage = skin_mask[index] * face[0];
            for channel in 0..3 {
                let diff = clamp(
                    reference[channel] - src[channel],
                    uniforms.diff_min_range,
                    uniforms.diff_max_range,
                );
                let corrected = src[channel] + diff;
                out.pixels[offset + channel] =
                    trunc_u8(src[channel] + (corrected - src[channel]) * coverage);
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Row 48: PF90001 texture smooth. Gate and input contract only.
// ---------------------------------------------------------------------------

pub const TEXTURE_SMOOTH_PF: u32 = 90001;

/// `ui/100`, then a strict `> 0.01` per-face predicate. A non-zero gate with no
/// recovered model-to-source blend is a typed refusal, never an identity claim.
pub fn texture_smooth_apply(
    source: &Image,
    ui: f32,
    per_face_engine: &[f32],
) -> Result<Image, Refusal> {
    source.validate()?;
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if per_face_engine.is_empty() {
        return Err(Refusal::MissingInput("per_face_engine"));
    }
    if !per_face_engine.iter().all(|value| value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("per_face_engine"));
    }
    let engine = ui / 100.0;
    let active = engine > 0.01 && per_face_engine.iter().any(|value| *value > 0.01);
    if !active {
        return Ok(source.clone());
    }
    Err(Refusal::NotImplemented(TEXTURE_SMOOTH_PF))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FULL: Roi = Roi {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
    };

    fn px() -> Image {
        Image::solid(1, 1, [102, 102, 102, 255])
    }

    #[test]
    fn meta_is_sorted_unique_and_executable() {
        for pair in META.windows(2) {
            assert!(pair[0].pf < pair[1].pf, "{} then {}", pair[0].pf, pair[1].pf);
        }
        assert!(META.iter().all(|row| row.access == Access::Cpu));
        assert!(META.iter().all(|row| row.pf != TEXTURE_SMOOTH_PF));
        assert_eq!(REFUSED[0].pf, TEXTURE_SMOOTH_PF);
        assert_eq!(REFUSED[0].access, Access::Refused);
    }

    #[test]
    fn wrinkle_degree_matches_each_recovered_edge() {
        assert_eq!(wrinkle_degree(21070, 50.0), Ok(0.5));
        assert_eq!(wrinkle_degree(21025, 100.0), Ok(1.0));
        assert_eq!(wrinkle_degree(21069, 50.0), Ok(50.0));
        assert_eq!(wrinkle_degree(21071, 50.0), Ok(50.0));
        assert_eq!(
            wrinkle_degree(21070, 101.0),
            Err(Refusal::ValueOutOfRange("ui"))
        );
        assert_eq!(wrinkle_degree(1234, 1.0), Err(Refusal::UnknownEffect(1234)));
    }

    #[test]
    fn wrinkle_below_positive_gate_is_identity() {
        let src = px();
        let dark = Image::solid(1, 1, [0, 0, 0, 255]);
        let out = wrinkle_apply(
            &src,
            &dark,
            &dark,
            &[255; 4],
            &[255],
            FULL,
            0.001,
            neutral_wrinkle_uniforms(),
        )
        .unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn wrinkle_zero_mask_leaves_pixels_alone() {
        let src = px();
        let dark = Image::solid(1, 1, [0, 0, 0, 255]);
        let out = wrinkle_apply(
            &src,
            &dark,
            &dark,
            &[0; 4],
            &[255],
            FULL,
            1.0,
            neutral_wrinkle_uniforms(),
        )
        .unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn wrinkle_composite_moves_alpha_and_truncates() {
        let src = px();
        let dark = Image::solid(1, 1, [0, 0, 0, 255]);
        let residual = Image::solid(1, 1, [51, 51, 51, 0]);
        let out = wrinkle_apply(
            &src,
            &dark,
            &residual,
            &[255; 4],
            &[255],
            FULL,
            1.0,
            neutral_wrinkle_uniforms(),
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0), [153, 153, 153, 255]);
    }

    #[test]
    fn wrinkle_rejects_shapes_and_non_finite_uniforms() {
        let src = px();
        assert_eq!(
            wrinkle_apply(
                &src,
                &src,
                &src,
                &[0; 3],
                &[0],
                FULL,
                1.0,
                neutral_wrinkle_uniforms()
            ),
            Err(Refusal::ShapeMismatch("mask_rgba"))
        );
        let mut uniforms = neutral_wrinkle_uniforms();
        uniforms.mix_alpha = f32::NAN;
        assert_eq!(
            wrinkle_apply(&src, &src, &src, &[0; 4], &[0], FULL, 1.0, uniforms),
            Err(Refusal::ValueOutOfRange("uniforms"))
        );
        assert_eq!(
            wrinkle_apply(
                &src,
                &src,
                &src,
                &[0; 4],
                &[0],
                Roi {
                    x: 1,
                    y: 0,
                    width: 1,
                    height: 1
                },
                1.0,
                neutral_wrinkle_uniforms()
            ),
            Err(Refusal::ValueOutOfRange("roi"))
        );
    }

    #[test]
    fn masked_model_output_reuses_shared_stage_inside_roi_only() {
        let src = Image::solid(2, 1, [10, 20, 30, 77]);
        let model = Image::solid(2, 1, [100, 150, 200, 5]);
        let out = masked_model_output(
            &src,
            &model,
            Roi {
                x: 1,
                y: 0,
                width: 1,
                height: 1,
            },
            &[128],
            false,
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0), [10, 20, 30, 77]);
        assert_eq!(out.pixel(1, 0), [50, 75, 100, 5]);
    }

    #[test]
    fn masked_model_output_rejects_mask_length_and_alternate_needs_mask() {
        let src = Image::solid(1, 1, [1, 2, 3, 4]);
        assert_eq!(
            masked_model_output(&src, &src, FULL, &[], false),
            Err(Refusal::ShapeMismatch("roi_mask"))
        );
    }

    #[test]
    fn row_gates_are_strict_at_their_thresholds() {
        assert_eq!(body_blemish_degree(0.0, None), Ok((0.0, false)));
        assert_eq!(
            body_blemish_degree(20.0, None),
            Err(Refusal::MissingInput("native_per_face_degree"))
        );
        assert_eq!(body_blemish_degree(20.0, Some(0.001)), Ok((0.001, false)));
        assert_eq!(body_blemish_degree(20.0, Some(0.5)), Ok((0.5, true)));

        assert_eq!(
            breast_degree(30.0, Some(50.0), false, true),
            Err(Refusal::MissingInput("body_skeleton"))
        );
        assert_eq!(breast_degree(30.0, Some(50.0), true, true), Ok((0.5, true)));
        assert_eq!(breast_degree(0.0, None, false, false), Ok((0.0, false)));

        assert_eq!(stretch_mark_degree(100.0), Ok((1.0, true)));
        assert_eq!(stretch_mark_degree(0.0), Ok((0.0, false)));
        assert_eq!(
            stretch_mark_degree(-1.0),
            Err(Refusal::ValueOutOfRange("strength"))
        );
    }

    #[test]
    fn skin_grey_skips_low_alpha_and_rejects_bad_alpha() {
        let original = Image::solid(1, 1, [255, 0, 0, 255]);
        let current = Image::solid(1, 1, [128, 128, 128, 255]);
        assert_eq!(
            skin_grey_apply(&original, &current, &[0], FULL, 1.0).unwrap(),
            current
        );
        assert_eq!(
            skin_grey_apply(&original, &current, &[255], FULL, 1.5),
            Err(Refusal::ValueOutOfRange("alpha"))
        );
        assert_eq!(
            skin_grey_apply(&original, &current, &[], FULL, 1.0),
            Err(Refusal::ShapeMismatch("mask"))
        );
    }

    #[test]
    fn skin_grey_keeps_hue_and_saturation_at_current_lightness() {
        let original = Image::solid(1, 1, [255, 0, 0, 255]);
        let current = Image::solid(1, 1, [128, 128, 128, 255]);
        let out = skin_grey_apply(&original, &current, &[255], FULL, 1.0).unwrap();
        let [r, g, b, a] = out.pixel(0, 0);
        assert_eq!(r, 255);
        assert!(g <= 2, "g={g}");
        assert!(b <= 2, "b={b}");
        assert_eq!(a, 255);
    }

    #[test]
    fn neutral_gray_3d_gate_and_light_dark_math() {
        let gate = neutral_gray_3d_degrees(50.0, 0.0, 30.0).unwrap();
        assert!(gate.active);
        assert!(!neutral_gray_3d_degrees(50.0, 0.0, 0.0).unwrap().active);
        assert_eq!(
            neutral_gray_3d_degrees(101.0, 0.0, 50.0),
            Err(Refusal::ValueOutOfRange("ui"))
        );

        let mut source = vec![128u8; NEUTRAL_GRAY_3D_BYTES];
        source[0] = 128;
        let light = vec![0.2f32; NEUTRAL_GRAY_3D_BYTES];
        let dark = vec![0.0f32; NEUTRAL_GRAY_3D_BYTES];
        let alpha = vec![1.0f32; NEUTRAL_GRAY_3D_PIXELS];
        let mask = vec![1.0f32; NEUTRAL_GRAY_3D_PIXELS];
        let out = neutral_gray_3d_apply(&source, &light, &dark, &alpha, &mask, gate).unwrap();
        assert_eq!(out[0], 154);
        assert_eq!(out.len(), NEUTRAL_GRAY_3D_BYTES);

        let inactive = neutral_gray_3d_degrees(0.0, 0.0, 100.0).unwrap();
        assert_eq!(
            neutral_gray_3d_apply(&source, &light, &dark, &alpha, &mask, inactive).unwrap(),
            source
        );
        assert_eq!(
            neutral_gray_3d_apply(&source[1..], &light, &dark, &alpha, &mask, gate),
            Err(Refusal::ShapeMismatch("source_bgr512"))
        );
    }

    #[test]
    fn hlf_face_zero_mask_is_identity_and_weight_matches_law() {
        let source = [0.5f32, 0.5, 0.5];
        let median = [0.4f32, 0.4, 0.4];
        let gauss = [0.6f32, 0.6, 0.6];
        let out = hlf_face_apply(
            &source,
            &median,
            &gauss,
            &[0.0],
            &[1.0],
            1,
            1,
            FULL,
            100.0,
        )
        .unwrap();
        assert_eq!(out, source);

        let out = hlf_face_apply(
            &source,
            &median,
            &gauss,
            &[1.0],
            &[1.0],
            1,
            1,
            FULL,
            100.0,
        )
        .unwrap();
        assert!((out[0] - 0.526_0).abs() < 1e-4, "{}", out[0]);
        assert_eq!(out[0], out[1]);
    }

    #[test]
    fn hlf_face_refuses_bad_ui_and_shapes() {
        let plane = [0.5f32; 3];
        assert_eq!(
            hlf_face_apply(&plane, &plane, &plane, &[1.0], &[1.0], 1, 1, FULL, 101.0),
            Err(Refusal::ValueOutOfRange("ui"))
        );
        assert_eq!(
            hlf_face_apply(&plane[..2], &plane, &plane, &[1.0], &[1.0], 1, 1, FULL, 50.0),
            Err(Refusal::ShapeMismatch("source"))
        );
    }

    #[test]
    fn body_hlf_preserves_alpha_and_outside_roi() {
        let src = Image::solid(2, 1, [128, 128, 128, 42]);
        let uniform = BodyHlfUniforms {
            spotless_degree: 0.0,
            retouch_degree: 1.0,
            spotless_acne_threshold: 0.0,
            spotless_high_freq_factor: 1.0,
            spotless_low_freq_alpha: 0.0,
            retouch_high_freq_factor: 0.0,
            diff_min_range: -0.2,
            diff_max_range: 0.2,
        };
        let median = [0.4f32; 6];
        let gauss = [0.6f32; 6];
        let face = [1.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let skin = [1.0f32, 1.0];
        let out = body_hlf_apply(
            &src,
            Roi {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
            &median,
            &gauss,
            &face,
            &skin,
            100.0,
            uniform,
            None,
        )
        .unwrap();
        let changed = out.pixel(0, 0);
        assert!((133..=135).contains(&changed[0]), "{changed:?}");
        assert_eq!(changed[3], 42);
        assert_eq!(out.pixel(1, 0), [128, 128, 128, 42]);
    }

    #[test]
    fn body_hlf_refuses_spotless_without_lut() {
        let src = Image::solid(1, 1, [128, 128, 128, 255]);
        let uniform = BodyHlfUniforms {
            spotless_degree: 1.0,
            retouch_degree: 0.0,
            spotless_acne_threshold: 0.0,
            spotless_high_freq_factor: 1.0,
            spotless_low_freq_alpha: 0.0,
            retouch_high_freq_factor: 0.0,
            diff_min_range: -0.2,
            diff_max_range: 0.2,
        };
        assert_eq!(
            body_hlf_apply(
                &src,
                FULL,
                &[0.4f32; 3],
                &[0.6f32; 3],
                &[1.0f32, 0.0, 0.0, 1.0],
                &[1.0f32],
                50.0,
                uniform,
                None
            ),
            Err(Refusal::MissingInput("adjust_lut"))
        );
    }

    #[test]
    fn texture_smooth_keeps_typed_refusal() {
        let src = px();
        assert_eq!(texture_smooth_apply(&src, 0.0, &[1.0]).unwrap(), src);
        assert_eq!(
            texture_smooth_apply(&src, 40.0, &[]),
            Err(Refusal::MissingInput("per_face_engine"))
        );
        assert_eq!(
            texture_smooth_apply(&src, 40.0, &[0.5]),
            Err(Refusal::NotImplemented(90001))
        );
        assert_eq!(
            texture_smooth_apply(&src, 40.0, &[0.0]),
            Ok(src.clone())
        );
    }
}
