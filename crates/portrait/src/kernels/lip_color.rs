//! PF107 `LipColor` row decode and `LipColorNormal.frag` pixel law.
//!
//! Evidence
//! --------
//! * Fragment source: `04_reference/legacy-root/Aurora/逆向工程/像素蛋糕/glsl/glsl/
//!   LipColor/LipColorNormal.frag`, fixture copy
//!   `03_algorithms/aurora/work/backend-r27/r6/fixtures/glsl/LipColorNormal.frag`
//!   (SHA256 `b812f8a8fec33af4e4e147b61d905f7ba90a9fc50f6feea17ecad557d9061734`).
//! * Host parser `0x4C16400` config-key → slot table: `backend-r27/r6/REPORT.md`.
//!
//! Scope
//! -----
//! The host's `LipColorMode` / `MultiColorSwitch` → shader-program selection has
//! no static xref and stays `SELECTION_OPEN` (`backend-r27/r6/UNCLOSED.md`,
//! `backend-r31/r6/REPORT.md`). This module implements only the single-colour
//! `LipColorNormal` law. A decoded row is admitted only when its fields are a
//! subset of that shader's uniform set; rows that carry evidence of another
//! family (`MetalColor` / `ShimmerMetalPath`, `MultiColorSwitch`,
//! `TeethWhitenSwitch`) are named refusals. `HighlightParam` is also a named
//! refusal because its host consumer was never located.
//!
//! `u_factorScale`, `u_factorShift`, `u_factorShrink`, `u_factorThreshold`,
//! `u_metallight_alpha` and `u_enable_hsl_adjust` have no recovered config
//! writer. They are explicit caller inputs ([`LipColorUniforms`]) and are never
//! guessed from `MakeupDegreeFactor`, `RadiusScaleFactor` or `HighlightParam`.
//!
//! Stage textures are sampled with an explicit UV convention ([`LipColorUv`]);
//! a missing convention is a named refusal, never an implicit stretch.

use crate::kernels::makeup_material::MaskChannel;
use crate::{Image, Refusal};

pub const LIP_COLOR_PF: u32 = 107;
pub const LIP_COLOR_EFFECT: &str = "LipColor";
pub const LIP_COLOR_SHADER: &str = "LipColorNormal.frag";

/// `LipColorMode` selects a shader family the host resolves at runtime; no
/// static xref maps a mode to `LipColorNormal`. Modes carrying metal, shimmer,
/// multi-colour or teeth-white evidence are refused here.
pub const LIP_COLOR_MODE_OPEN: &str = "PF107_LIP_COLOR_MODE";
/// `HighlightParam { Weight, Threshold }` has no located host consumer.
pub const LIP_COLOR_HIGHLIGHT_OPEN: &str = "PF107_LIP_COLOR_HIGHLIGHT";
/// `TeethWhitenSwitch` selects the `LipColorAndTeehWhite` family.
pub const LIP_COLOR_TEETH_WHITE_OPEN: &str = "PF107_LIP_COLOR_TEETH_WHITE";
/// `Opacity0` has no recovered shader uniform; only 100 is admitted.
pub const LIP_COLOR_OPACITY: &str = "PF107_LIP_COLOR_OPACITY";
/// `AdaptEnv` needs `u_lip_light_degree`, which needs `LightAlpha`.
pub const LIP_COLOR_LIGHT_ALPHA: &str = "PF107_LIP_COLOR_LIGHT_ALPHA";
/// No explicit stage-texture UV convention was supplied.
pub const LIP_COLOR_UV_MISSING: &str = "PF107_LIP_COLOR_UV";
/// The supplied UV convention is non-finite or has a non-positive scale.
pub const LIP_COLOR_UV_INVALID: &str = "PF107_LIP_COLOR_UV_INVALID";

/// `LipColorMode` values observed in the decoded corpus: 0 Matte, 1 Satin,
/// 2 Creamy, 3 Glossy, 4 Shimmer. Mode 6 appears only in a Suit preset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LipColorMode {
    Matte,
    Satin,
    Creamy,
    Glossy,
    Shimmer,
    Other(u32),
}

impl LipColorMode {
    pub fn decode(mode: u32) -> Self {
        match mode {
            0 => Self::Matte,
            1 => Self::Satin,
            2 => Self::Creamy,
            3 => Self::Glossy,
            4 => Self::Shimmer,
            other => Self::Other(other),
        }
    }

    pub fn raw(self) -> u32 {
        match self {
            Self::Matte => 0,
            Self::Satin => 1,
            Self::Creamy => 2,
            Self::Glossy => 3,
            Self::Shimmer => 4,
            Self::Other(value) => value,
        }
    }
}

/// `HighlightParam { Weight, Threshold }` from the vendor preset, carried
/// verbatim. Presence forces [`LIP_COLOR_HIGHLIGHT_OPEN`] at the consumer
/// boundary; the numbers are never reinterpreted as shader constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HighlightParam {
    pub weight: f64,
    pub threshold: f64,
}

/// The decoded `LipColor` effect row. Fields are raw vendor values; percent to
/// unit conversion happens in [`LipColorParams::from_vendor`].
#[derive(Clone, Debug, PartialEq)]
pub struct LipColorRow {
    pub effect_name: &'static str,
    pub config_path: String,
    pub lip_color_mode: u32,
    pub multi_color_switch: bool,
    pub teeth_whiten_switch: bool,
    pub makeup_degree_factor: f64,
    pub light_alpha_percent: Option<f64>,
    pub gloss_alpha_percent: Option<f64>,
    /// `MixColor0` bytes.
    pub mix_color0: [u8; 4],
    pub opacity0_percent: f64,
    pub adapt_env: bool,
    pub remove_highlight: bool,
    pub radius_scale_factor: Option<f64>,
    /// `MetalColor`, present on Shimmer rows.
    pub metal_color: Option<[u8; 4]>,
    pub shimmer_metal_path: Option<String>,
    pub highlight_param: Option<HighlightParam>,
    pub default_degree: f64,
    pub degree_range: [f64; 2],
}

/// `LipColorNormal` uniforms whose config writer was not located. The caller
/// must supply them from same-job evidence (host capture or preset-table dump).
/// `enable_hsl_adjust` is also here: no config key maps to it in the corpus.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LipColorUniforms {
    pub factor_scale: f32,
    pub factor_shift: f32,
    pub factor_shrink: f32,
    pub factor_threshold: f32,
    pub metallight_alpha: f32,
    pub enable_hsl_adjust: bool,
}

impl LipColorUniforms {
    fn validate(self) -> Result<(), Refusal> {
        for value in [
            self.factor_scale,
            self.factor_shift,
            self.factor_shrink,
            self.factor_threshold,
            self.metallight_alpha,
        ] {
            if !value.is_finite() {
                return Err(Refusal::ValueOutOfRange("LipColor uniform"));
            }
        }
        if !(0.0..=1.0).contains(&self.metallight_alpha) {
            return Err(Refusal::ValueOutOfRange("LipColor metallight alpha"));
        }
        Ok(())
    }
}

/// Resolved `LipColorNormal` inputs. `mode` records the preset mode but does not
/// select a program; the caller explicitly chose this law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LipColorParams {
    /// `u_lip_color` = `MixColor0` in `[0, 1]` channels.
    pub lip_color: [f32; 4],
    /// `u_lip_light_degree` = `LightAlpha / 100`.
    pub lip_light_degree: f32,
    pub uniforms: LipColorUniforms,
    pub lipcolor_switch: bool,
    pub adapt_env: bool,
    pub remove_highlight: bool,
    pub mode: LipColorMode,
}

fn percent(value: f64, name: &'static str) -> Result<f32, Refusal> {
    if !value.is_finite() || !(0.0..=100.0).contains(&value) {
        return Err(Refusal::ValueOutOfRange(name));
    }
    Ok((value / 100.0) as f32)
}

impl LipColorParams {
    /// Converts a decoded vendor row plus caller-supplied uniforms into the
    /// shader parameter set.
    ///
    /// `Opacity0` is required to be 100: the host stores it at `+0x2A4` but no
    /// shader uniform reads it, so a non-100 value cannot be silently applied
    /// or silently dropped.
    pub fn from_vendor(
        row: &LipColorRow,
        uniforms: LipColorUniforms,
    ) -> Result<Self, Refusal> {
        if row.effect_name != LIP_COLOR_EFFECT {
            return Err(Refusal::ValueOutOfRange(LIP_COLOR_EFFECT));
        }
        if !row.opacity0_percent.is_finite() || (row.opacity0_percent - 100.0).abs() > 1e-9 {
            return Err(Refusal::MissingInput(LIP_COLOR_OPACITY));
        }
        if row.multi_color_switch {
            return Err(Refusal::MissingInput(LIP_COLOR_MODE_OPEN));
        }
        if row.teeth_whiten_switch {
            return Err(Refusal::MissingInput(LIP_COLOR_TEETH_WHITE_OPEN));
        }
        if row.metal_color.is_some() || row.shimmer_metal_path.is_some() {
            return Err(Refusal::MissingInput(LIP_COLOR_MODE_OPEN));
        }
        if row.highlight_param.is_some() {
            return Err(Refusal::MissingInput(LIP_COLOR_HIGHLIGHT_OPEN));
        }
        match LipColorMode::decode(row.lip_color_mode) {
            LipColorMode::Matte
            | LipColorMode::Satin
            | LipColorMode::Creamy
            | LipColorMode::Glossy => {}
            LipColorMode::Shimmer | LipColorMode::Other(_) => {
                return Err(Refusal::MissingInput(LIP_COLOR_MODE_OPEN));
            }
        }
        if !row.makeup_degree_factor.is_finite() || row.makeup_degree_factor < 0.0 {
            return Err(Refusal::ValueOutOfRange("MakeupDegreeFactor"));
        }
        if let Some(radius) = row.radius_scale_factor {
            if !radius.is_finite() || radius < 0.0 {
                return Err(Refusal::ValueOutOfRange("RadiusScaleFactor"));
            }
        }
        if !row.default_degree.is_finite()
            || row.degree_range.iter().any(|value| !value.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("LipColor degree range"));
        }
        let light_degree = match row.light_alpha_percent {
            Some(value) => percent(value, "LightAlpha")?,
            None if row.adapt_env => return Err(Refusal::MissingInput(LIP_COLOR_LIGHT_ALPHA)),
            None => 0.0,
        };
        if let Some(value) = row.gloss_alpha_percent {
            // `GlossAlpha` has no uniform in `LipColorNormal`; it is only
            // validated here so a malformed value is still named.
            let _ = percent(value, "GlossAlpha")?;
        }
        uniforms.validate()?;
        let params = Self {
            lip_color: row.mix_color0.map(|byte| byte as f32 / 255.0),
            lip_light_degree: light_degree,
            uniforms,
            lipcolor_switch: true,
            adapt_env: row.adapt_env,
            remove_highlight: row.remove_highlight,
            mode: LipColorMode::decode(row.lip_color_mode),
        };
        params.validate()?;
        Ok(params)
    }

    fn validate(self) -> Result<(), Refusal> {
        if self
            .lip_color
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(Refusal::ValueOutOfRange(LIP_COLOR_EFFECT));
        }
        if !self.lip_light_degree.is_finite() || !(0.0..=1.0).contains(&self.lip_light_degree) {
            return Err(Refusal::ValueOutOfRange("LightAlpha"));
        }
        self.uniforms.validate()
    }

    /// Stage textures required by `LipColorNormal.frag`: the combine mask uses
    /// R coverage, G contrast and B gloss; the smooth texture uses G for the
    /// whiten gray and B for the HSL gate. `LipColor` has no `FilterType`.
    pub fn mask_contract(self) -> LipColorMaskContract {
        LipColorMaskContract {
            combine_mask: MaskChannel::Rgb,
            smooth: "smooth texture G/B",
            blend_weight_lut: "blend weight LUT, R row 0 / G,B row 1",
            level_weight_lut: "level weight LUT, R row 0",
            filter_type: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LipColorMaskContract {
    pub combine_mask: MaskChannel,
    pub smooth: &'static str,
    pub blend_weight_lut: &'static str,
    pub level_weight_lut: &'static str,
    /// `LipColor` has no `FilterType`; a caller must never substitute 40/11.
    pub filter_type: Option<u32>,
}

/// `origin + uv * scale` in the target texture's normalised space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UvMapping {
    pub origin: [f32; 2],
    pub scale: [f32; 2],
}

impl UvMapping {
    pub const SAME_AS_SOURCE: Self = Self {
        origin: [0.0, 0.0],
        scale: [1.0, 1.0],
    };

    pub fn crop(origin: [f32; 2], scale: [f32; 2]) -> Self {
        Self { origin, scale }
    }

    fn valid(self) -> bool {
        self.origin
            .iter()
            .chain(self.scale.iter())
            .all(|value| value.is_finite())
            && self.scale.iter().all(|value| *value > 0.0)
    }

    fn map(self, uv: [f32; 2]) -> [f32; 2] {
        [
            self.origin[0] + uv[0] * self.scale[0],
            self.origin[1] + uv[1] * self.scale[1],
        ]
    }
}

/// Explicit UV conventions for the combine mask and smooth textures. `None` is
/// a refusal: the vendor sample UV/origin was not recovered.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LipColorUv {
    pub combine_mask: UvMapping,
    pub smooth: UvMapping,
}

fn checked(image: &Image, name: &'static str) -> Result<(), Refusal> {
    image.validate().map_err(|_| Refusal::ShapeMismatch(name))
}

fn clamp01(value: f32) -> f32 {
    if value.is_nan() {
        0.0
    } else {
        value.clamp(0.0, 1.0)
    }
}

fn sample(image: &Image, uv: [f32; 2]) -> [f32; 4] {
    let u = clamp01(uv[0]);
    let v = clamp01(uv[1]);
    let x = u * image.width as f32 - 0.5;
    let y = v * image.height as f32 - 0.5;
    let x0f = x.floor();
    let y0f = y.floor();
    let tx = x - x0f;
    let ty = y - y0f;
    let x0 = (x0f as i64).clamp(0, image.width as i64 - 1) as usize;
    let x1 = ((x0f as i64) + 1).clamp(0, image.width as i64 - 1) as usize;
    let y0 = (y0f as i64).clamp(0, image.height as i64 - 1) as usize;
    let y1 = ((y0f as i64) + 1).clamp(0, image.height as i64 - 1) as usize;
    let at = |px: usize, py: usize, channel: usize| {
        image.pixels[(py * image.width as usize + px) * 4 + channel] as f32 / 255.0
    };
    let mut out = [0.0f32; 4];
    for (channel, value) in out.iter_mut().enumerate() {
        let top = at(x0, y0, channel) * (1.0 - tx) + at(x1, y0, channel) * tx;
        let bottom = at(x0, y1, channel) * (1.0 - tx) + at(x1, y1, channel) * tx;
        *value = top * (1.0 - ty) + bottom * ty;
    }
    out
}

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}

fn overlay(base: f32, blend: f32) -> f32 {
    if base < 0.5 {
        2.0 * base * blend
    } else {
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
    }
}

const HSL_EPSILON: f32 = 0.001;

/// `RGBtoHCV` + `RGBtoHSL` from `LipColorNormal.frag`, evaluated in the same
/// order as the GLSL.
fn rgb_to_hsl(rgb: [f32; 3]) -> [f32; 3] {
    let p = if rgb[1] < rgb[2] {
        [rgb[2], rgb[1], -1.0, 2.0 / 3.0]
    } else {
        [rgb[1], rgb[2], 0.0, -1.0 / 3.0]
    };
    let q = if rgb[0] < p[0] {
        [p[0], p[1], p[3], rgb[0]]
    } else {
        [rgb[0], p[1], p[2], p[0]]
    };
    let c = q[0] - q[3].min(q[1]);
    let h = ((q[3] - q[1]) / (6.0 * c + HSL_EPSILON) + q[2]).abs();
    let l = q[0] - c * 0.5;
    let s = c / (1.0 - (l * 2.0 - 1.0).abs() + HSL_EPSILON);
    [h, s, l]
}

/// `HSLtoRGB` from `LipColorNormal.frag`.
fn hsl_to_rgb(hsl: [f32; 3]) -> [f32; 3] {
    let x = hsl[0] * 6.0;
    let rgb = [
        (x - 3.0).abs() - 1.0,
        2.0 - (x - 2.0).abs(),
        2.0 - (x - 4.0).abs(),
    ]
    .map(clamp01);
    let c = (1.0 - (2.0 * hsl[2] - 1.0).abs()) * hsl[1];
    rgb.map(|channel| clamp01((channel - 0.5) * c + hsl[2]))
}

/// Executes `LipColorNormal.frag` for one same-job pixel set.
///
/// `blend_weight` and `level_weight` are the gray-indexed lookup textures. The
/// output alpha is copied from the source, exactly as in the shader.
pub fn apply_lip_color(
    source: &Image,
    combine_mask: &Image,
    smooth: &Image,
    blend_weight: &Image,
    level_weight: &Image,
    uv: Option<LipColorUv>,
    params: LipColorParams,
) -> Result<Image, Refusal> {
    checked(source, "source")?;
    checked(combine_mask, "combine_mask")?;
    checked(smooth, "smooth")?;
    checked(blend_weight, "blend_weight")?;
    checked(level_weight, "level_weight")?;
    params.validate()?;
    let uv = uv.ok_or(Refusal::MissingInput(LIP_COLOR_UV_MISSING))?;
    if !uv.combine_mask.valid() || !uv.smooth.valid() {
        return Err(Refusal::ValueOutOfRange(LIP_COLOR_UV_INVALID));
    }
    let mut output = source.clone();
    for y in 0..source.height {
        for x in 0..source.width {
            let index = (y as usize * source.width as usize + x as usize) * 4;
            let pixel_uv = [
                (x as f32 + 0.5) / source.width as f32,
                (y as f32 + 0.5) / source.height as f32,
            ];
            let src = [
                source.pixels[index] as f32 / 255.0,
                source.pixels[index + 1] as f32 / 255.0,
                source.pixels[index + 2] as f32 / 255.0,
            ];
            let gray = 0.299 * src[0] + 0.587 * src[1] + 0.114 * src[2];
            let mask = sample(combine_mask, uv.combine_mask.map(pixel_uv));
            let smooth_sample = sample(smooth, uv.smooth.map(pixel_uv));
            let mut res = src;
            if params.lipcolor_switch && mask[0] > 0.0 {
                let alpha = mask[0] * params.lip_color[3];
                let contrast_mask = mask[1] * params.uniforms.metallight_alpha;
                let gloss = mask[2];

                let whiten_gray = smooth_sample[1];
                let mut contrast = ((whiten_gray * params.uniforms.factor_scale
                    + params.uniforms.factor_shift)
                    * contrast_mask
                    + 0.5 * (1.0 - contrast_mask))
                    .clamp(0.0, 1.0);
                if contrast < 0.5 {
                    contrast = 0.5 - (0.5 - contrast) * params.uniforms.factor_shrink;
                }

                let blend_weight_sample = sample(blend_weight, [gray, 0.0])[0];
                let level_weight_sample = sample(level_weight, [gray, 0.0])[0] * alpha;
                let mix_color = [
                    params.lip_color[0] * src[0],
                    params.lip_color[1] * src[1],
                    params.lip_color[2] * src[2],
                ];
                res = [
                    mix(
                        src[0],
                        mix(mix_color[0], params.lip_color[0], blend_weight_sample),
                        level_weight_sample,
                    ),
                    mix(
                        src[1],
                        mix(mix_color[1], params.lip_color[1], blend_weight_sample),
                        level_weight_sample,
                    ),
                    mix(
                        src[2],
                        mix(mix_color[2], params.lip_color[2], blend_weight_sample),
                        level_weight_sample,
                    ),
                ];
                let diff = (gray - params.uniforms.factor_threshold).max(0.0) * alpha;

                let (blend_weight_g, blend_weight_b) = if params.adapt_env {
                    let mut gray_val = gray;
                    if gray_val < params.lip_light_degree {
                        gray_val += 0.6 * (params.lip_light_degree - gray_val);
                    }
                    let row = sample(blend_weight, [gray_val, 1.0]);
                    (row[1], row[2])
                } else {
                    (1.0, 1.0)
                };

                if params.adapt_env {
                    res = [
                        src[0] * (1.0 - blend_weight_g) + res[0] * blend_weight_g,
                        src[1] * (1.0 - blend_weight_g) + res[1] * blend_weight_g,
                        src[2] * (1.0 - blend_weight_g) + res[2] * blend_weight_g,
                    ];
                    res = res.map(|value| 1.0 - (1.0 - value) * (1.0 - diff * blend_weight_b));
                } else {
                    res = res.map(|value| 1.0 - (1.0 - value) * (1.0 - diff));
                }

                if params.lip_color[3] > 0.001 {
                    let reference = res.map(|value| overlay(value, contrast));
                    if params.adapt_env {
                        let mixed = [
                            mix(res[0], reference[0], params.lip_color[3]),
                            mix(res[1], reference[1], params.lip_color[3]),
                            mix(res[2], reference[2], params.lip_color[3]),
                        ];
                        res = [
                            res[0] * (1.0 - blend_weight_g) + mixed[0] * blend_weight_g,
                            res[1] * (1.0 - blend_weight_g) + mixed[1] * blend_weight_g,
                            res[2] * (1.0 - blend_weight_g) + mixed[2] * blend_weight_g,
                        ];
                    } else {
                        res = [
                            mix(res[0], reference[0], alpha),
                            mix(res[1], reference[1], alpha),
                            mix(res[2], reference[2], alpha),
                        ];
                    }
                }

                if params.adapt_env {
                    res = res.map(|value| {
                        value * (1.0 - gloss * blend_weight_g) + gloss * blend_weight_g
                    });
                } else {
                    res = res.map(|value| value * (1.0 - gloss) + gloss);
                }
            }

            if params.uniforms.enable_hsl_adjust && smooth_sample[2] > 0.0 && params.lip_color[3] > 0.0
            {
                let mut hsl = rgb_to_hsl(res);
                let alpha = mask[0] * smooth_sample[2] * params.lip_color[3];
                hsl[1] = (hsl[1] + alpha * 0.4).clamp(0.0, 1.0);
                hsl[2] = (hsl[2] - alpha * 0.2).clamp(0.0, 1.0);
                res = hsl_to_rgb(hsl);
            }

            if params.remove_highlight && mask[0] > 0.0 {
                let reference = [
                    res[0] * params.lip_color[0],
                    res[1] * params.lip_color[1],
                    res[2] * params.lip_color[2],
                ];
                let amount = mask[0] * params.lip_color[3] * gray;
                res = [
                    mix(res[0], reference[0], amount),
                    mix(res[1], reference[1], amount),
                    mix(res[2], reference[2], amount),
                ];
            }

            for channel in 0..3 {
                output.pixels[index + channel] =
                    (clamp01(res[channel]) * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIFORMS: LipColorUniforms = LipColorUniforms {
        factor_scale: 1.0,
        factor_shift: 0.0,
        factor_shrink: 1.0,
        factor_threshold: 1.0,
        metallight_alpha: 0.0,
        enable_hsl_adjust: false,
    };

    fn row() -> LipColorRow {
        LipColorRow {
            effect_name: LIP_COLOR_EFFECT,
            config_path: "Makeups/Lipstick/MULS00000001/Satin/effect.json".to_owned(),
            lip_color_mode: 1,
            multi_color_switch: false,
            teeth_whiten_switch: false,
            makeup_degree_factor: 1.0,
            light_alpha_percent: None,
            gloss_alpha_percent: None,
            mix_color0: [179, 67, 74, 255],
            opacity0_percent: 100.0,
            adapt_env: false,
            remove_highlight: false,
            radius_scale_factor: Some(1.0),
            metal_color: None,
            shimmer_metal_path: None,
            highlight_param: None,
            default_degree: 50.0,
            degree_range: [0.0, 100.0],
        }
    }

    fn same() -> Option<LipColorUv> {
        Some(LipColorUv {
            combine_mask: UvMapping::SAME_AS_SOURCE,
            smooth: UvMapping::SAME_AS_SOURCE,
        })
    }

    fn assert_close(actual: [u8; 4], expected: [u8; 4], tolerance: i32) {
        for channel in 0..4 {
            let delta = actual[channel] as i32 - expected[channel] as i32;
            assert!(
                delta.abs() <= tolerance,
                "channel {channel}: {actual:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn vendor_fields_map_to_named_uniforms() {
        let params = LipColorParams::from_vendor(&row(), UNIFORMS).unwrap();
        assert_eq!(
            params.lip_color,
            [179.0 / 255.0, 67.0 / 255.0, 74.0 / 255.0, 1.0]
        );
        assert_eq!(params.lip_light_degree, 0.0);
        assert_eq!(params.mode, LipColorMode::Satin);
        assert!(params.mask_contract().filter_type.is_none());
        assert_eq!(params.mask_contract().combine_mask, MaskChannel::Rgb);
    }

    #[test]
    fn light_alpha_becomes_the_adapt_env_degree() {
        let mut matte = row();
        matte.lip_color_mode = 0;
        matte.light_alpha_percent = Some(20.0);
        matte.adapt_env = true;
        assert_eq!(
            LipColorParams::from_vendor(&matte, UNIFORMS)
                .unwrap()
                .lip_light_degree,
            0.2
        );
        let mut missing = row();
        missing.adapt_env = true;
        assert_eq!(
            LipColorParams::from_vendor(&missing, UNIFORMS),
            Err(Refusal::MissingInput(LIP_COLOR_LIGHT_ALPHA))
        );
    }

    #[test]
    fn shader_family_selection_is_a_named_refusal() {
        let mut shimmer = row();
        shimmer.lip_color_mode = 4;
        shimmer.metal_color = Some([247, 223, 231, 255]);
        shimmer.shimmer_metal_path = Some("ShimmerMaterial.jpg".to_owned());
        assert_eq!(
            LipColorParams::from_vendor(&shimmer, UNIFORMS),
            Err(Refusal::MissingInput(LIP_COLOR_MODE_OPEN))
        );
        let mut multi = row();
        multi.multi_color_switch = true;
        assert_eq!(
            LipColorParams::from_vendor(&multi, UNIFORMS),
            Err(Refusal::MissingInput(LIP_COLOR_MODE_OPEN))
        );
        let mut teeth = row();
        teeth.teeth_whiten_switch = true;
        assert_eq!(
            LipColorParams::from_vendor(&teeth, UNIFORMS),
            Err(Refusal::MissingInput(LIP_COLOR_TEETH_WHITE_OPEN))
        );
        let mut unknown = row();
        unknown.lip_color_mode = 6;
        assert_eq!(
            LipColorParams::from_vendor(&unknown, UNIFORMS),
            Err(Refusal::MissingInput(LIP_COLOR_MODE_OPEN))
        );
    }

    #[test]
    fn highlight_and_non_default_opacity_are_named_refusals() {
        let mut highlighted = row();
        highlighted.highlight_param = Some(HighlightParam {
            weight: 0.97,
            threshold: 0.1,
        });
        assert_eq!(
            LipColorParams::from_vendor(&highlighted, UNIFORMS),
            Err(Refusal::MissingInput(LIP_COLOR_HIGHLIGHT_OPEN))
        );
        let mut opacity = row();
        opacity.opacity0_percent = 50.0;
        assert_eq!(
            LipColorParams::from_vendor(&opacity, UNIFORMS),
            Err(Refusal::MissingInput(LIP_COLOR_OPACITY))
        );
    }

    #[test]
    fn non_finite_uniforms_are_refused() {
        let mut uniforms = UNIFORMS;
        uniforms.factor_scale = f32::NAN;
        assert_eq!(
            LipColorParams::from_vendor(&row(), uniforms),
            Err(Refusal::ValueOutOfRange("LipColor uniform"))
        );
        let mut uniforms = UNIFORMS;
        uniforms.metallight_alpha = 2.0;
        assert_eq!(
            LipColorParams::from_vendor(&row(), uniforms),
            Err(Refusal::ValueOutOfRange("LipColor metallight alpha"))
        );
    }

    #[test]
    fn zero_mask_is_identity_and_alpha_is_preserved() {
        let source = Image::solid(2, 2, [10, 20, 30, 77]);
        let mask = Image::solid(2, 2, [0; 4]);
        let lut = Image::solid(1, 1, [255; 4]);
        let params = LipColorParams::from_vendor(&row(), UNIFORMS).unwrap();
        let out = apply_lip_color(&source, &mask, &mask, &lut, &lut, same(), params).unwrap();
        assert_eq!(out, source);
    }

    #[test]
    fn full_level_weight_mixes_the_lip_colour() {
        let source = Image::solid(1, 1, [128, 128, 128, 77]);
        let mask = Image::solid(1, 1, [255, 0, 0, 255]);
        let smooth = Image::solid(1, 1, [0, 128, 0, 255]);
        let blend_weight = Image::solid(1, 1, [0, 0, 0, 255]);
        let level_weight = Image::solid(1, 1, [255, 0, 0, 255]);
        let params = LipColorParams::from_vendor(&row(), UNIFORMS).unwrap();
        let out = apply_lip_color(
            &source,
            &mask,
            &smooth,
            &blend_weight,
            &level_weight,
            same(),
            params,
        )
        .unwrap();
        // mix(src, MixColor0 * src, 1) then a neutral contrast/gloss pass.
        assert_close(out.pixel(0, 0), [90, 34, 37, 77], 1);
    }

    #[test]
    fn remove_highlight_moves_toward_the_multiplication() {
        let source = Image::solid(1, 1, [128, 128, 128, 77]);
        let mask = Image::solid(1, 1, [255, 0, 0, 255]);
        let smooth = Image::solid(1, 1, [0, 128, 0, 255]);
        let lut = Image::solid(1, 1, [0, 0, 0, 255]);
        let mut row = row();
        row.remove_highlight = true;
        let params = LipColorParams::from_vendor(&row, UNIFORMS).unwrap();
        let out = apply_lip_color(&source, &mask, &smooth, &lut, &lut, same(), params).unwrap();
        assert_close(out.pixel(0, 0), [109, 81, 82, 77], 1);
    }

    #[test]
    fn hsl_adjust_changes_the_result_only_when_enabled() {
        let source = Image::solid(1, 1, [128, 64, 64, 200]);
        let mask = Image::solid(1, 1, [255, 0, 0, 255]);
        let smooth = Image::solid(1, 1, [0, 128, 255, 255]);
        let lut = Image::solid(1, 1, [0, 0, 0, 255]);
        let params = LipColorParams::from_vendor(&row(), UNIFORMS).unwrap();
        let off = apply_lip_color(&source, &mask, &smooth, &lut, &lut, same(), params).unwrap();
        assert_close(off.pixel(0, 0), source.pixel(0, 0), 1);

        let mut uniforms = UNIFORMS;
        uniforms.enable_hsl_adjust = true;
        let params = LipColorParams::from_vendor(&row(), uniforms).unwrap();
        let on = apply_lip_color(&source, &mask, &smooth, &lut, &lut, same(), params).unwrap();
        assert_ne!(on.pixel(0, 0), source.pixel(0, 0));
        assert_eq!(on.pixel(0, 0)[3], 200);
    }

    #[test]
    fn missing_or_invalid_uv_is_refused() {
        let image = Image::solid(1, 1, [10, 20, 30, 255]);
        let params = LipColorParams::from_vendor(&row(), UNIFORMS).unwrap();
        assert_eq!(
            apply_lip_color(&image, &image, &image, &image, &image, None, params),
            Err(Refusal::MissingInput(LIP_COLOR_UV_MISSING))
        );
        let bad = Some(LipColorUv {
            combine_mask: UvMapping::crop([0.0, 0.0], [0.0, 1.0]),
            smooth: UvMapping::SAME_AS_SOURCE,
        });
        assert_eq!(
            apply_lip_color(&image, &image, &image, &image, &image, bad, params),
            Err(Refusal::ValueOutOfRange(LIP_COLOR_UV_INVALID))
        );
    }

    #[test]
    fn empty_stage_texture_is_a_shape_refusal() {
        let image = Image::solid(1, 1, [10, 20, 30, 255]);
        let empty = Image {
            width: 0,
            height: 0,
            pixels: Vec::new(),
        };
        let params = LipColorParams::from_vendor(&row(), UNIFORMS).unwrap();
        assert_eq!(
            apply_lip_color(&image, &empty, &image, &image, &image, same(), params),
            Err(Refusal::ShapeMismatch("combine_mask"))
        );
    }
}
