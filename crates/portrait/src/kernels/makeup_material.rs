//! Color makeup material composer, rows 87 / PF106 (BlusherColorMakeup).
//!
//! Recovered consumer: `glsl/OverLay/BlendColorLayerWithMask.frag`.
//! The shader is a single-mask color-layer compositor:
//!   factor = u_mix_color.a * mask_weight
//!   res    = mix(src, Blend(src, blend_color), factor)
//! `u_mix_color.a` is the recovered operand that carries opacity * degree.
//! The blend macro is a compile-time cell input, so it stays explicit here.

use crate::{Access, EffectMeta, Image, Refusal};

/// PF106 and PF116 have concrete single-mask profiles; the blend macro and the
/// material texture stay caller data because the cell preset is chosen at run time.
pub const META: &[EffectMeta] = &[
    EffectMeta {
        pf: 106,
        key: "blusherDegree",
        label: "Blusher color makeup",
        access: Access::Cpu,
        inputs: &[
            "BlusherMaskType=15 owner mask RGBA texture",
            "per-pixel UV",
            "MixColor RGBA",
            "Opacity percent",
            "degree01",
            "BlendMode macro",
        ],
    },
    EffectMeta {
        pf: 116,
        key: "contourDegree",
        label: "Blusher image makeup material",
        access: Access::Cpu,
        inputs: &[
            "material RGBA texture",
            "BlusherMaskType=15 owner mask RGBA texture",
            "per-pixel UV",
            "MixColor RGBA",
            "Opacity percent",
            "degree01",
            "BlendMode macro",
        ],
    },
];

pub const BLUSHER_PF: u32 = 106;
pub const BLUSHER_FILTER_TYPE: u32 = 11;
pub const BLUSHER_MASK_FIELD: &str = "BlusherMaskType";
pub const BLUSHER_MASK_VALUE: u32 = 15;
pub const BLUSHER_SHADER: &str = "glsl/OverLay/BlendColorLayerWithMask.frag";
/// The cell macro is not pinned. The family is pinned; the macro is caller data.
pub const BLUSHER_BLEND_BINDING: &str = "CELL_BLEND_MACRO_OPEN";

pub const BLUSHER_IMAGE_PF: u32 = 116;
pub const BLUSHER_IMAGE_FILTER_TYPE: u32 = 42;
pub const BLUSHER_IMAGE_SHADER: &str = "glsl/OverLay/BlendImageLayerWithMask.frag";
pub const BLUSHER_IMAGE_BLEND_BINDING: &str = "CELL_BLEND_MACRO_OPEN";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskChannel {
    R,
    G,
    B,
    A,
    RG,
    RB,
    RA,
    GB,
    GA,
    BA,
    Rgb,
    Rga,
    Rba,
    Gba,
    Rgba,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    Normal,
    Darken,
    Multiply,
    ColorBurn,
    LinearBurn,
    Lighten,
    Screen,
    ColorDodge,
    LinearDodge,
    Overlay,
    SoftLight,
    HardLight,
    VividLight,
    LinearLight,
    PinLight,
    HardMix,
    Difference,
    Exclusion,
    Subtract,
    Hue,
    Saturation,
    Color,
    Luminosity,
    Average,
    Negation,
    Reflect,
    Glow,
    Phoenix,
}

impl BlendMode {
    /// Maps the shader's compile-time macro name to the exact branch.
    pub fn from_macro(name: &str) -> Result<Self, Refusal> {
        let mode = match name {
            "NORMAL" => Self::Normal,
            "DARKEN" => Self::Darken,
            "MULTIPLY" => Self::Multiply,
            "COLOR_BURN" => Self::ColorBurn,
            "LINEAR_BURN" => Self::LinearBurn,
            "LIGHTEN" => Self::Lighten,
            "SCREEN" => Self::Screen,
            "COLOR_DODGE" => Self::ColorDodge,
            "LINEAR_DODGE" | "ADD" => Self::LinearDodge,
            "OVERLAY" => Self::Overlay,
            "SOFT_LIGHT" => Self::SoftLight,
            "HARD_LIGHT" => Self::HardLight,
            "VIVID_LIGHT" => Self::VividLight,
            "LINEAR_LIGHT" => Self::LinearLight,
            "PIN_LIGHT" => Self::PinLight,
            "HARD_MIX" => Self::HardMix,
            "DIFFERENCE" => Self::Difference,
            "EXCLUSION" => Self::Exclusion,
            "SUBTRACT" => Self::Subtract,
            "HUE" => Self::Hue,
            "SATURATION" => Self::Saturation,
            "COLOR" => Self::Color,
            "LUMINOSITY" => Self::Luminosity,
            "AVERAGE" => Self::Average,
            "NEGATION" => Self::Negation,
            "HARD_REFLECT" => Self::Reflect,
            "HARD_GLOW" => Self::Glow,
            "HARD_PHOENIX" => Self::Phoenix,
            _ => return Err(Refusal::MissingInput("blend_mode_macro")),
        };
        Ok(mode)
    }
}

/// `BlusherMaskType=15` is the only row-87 value already pinned to `MASK_RGBA`.
pub fn blusher_mask_channel(mask_type: u32) -> Result<MaskChannel, Refusal> {
    if mask_type == BLUSHER_MASK_VALUE {
        Ok(MaskChannel::Rgba)
    } else {
        Err(Refusal::MissingInput("blusher_mask_channel"))
    }
}

/// `LipMaskType` table at `0x92CBB18`; `3 -> MASK_R`.
pub fn lipstick_mask_channel(mask_type: u32) -> Result<MaskChannel, Refusal> {
    let index = match mask_type {
        1..=4 => 0,
        5 => 1,
        6 => 2,
        7 => 3,
        8 => 4,
        9 => 5,
        10 => 6,
        11 => 7,
        12 => 8,
        13 => 9,
        14 => 10,
        15 => 11,
        16 => 12,
        17 => 13,
        _ => return Err(Refusal::MissingInput("lipstick_mask_type")),
    };
    Ok([
        MaskChannel::R,
        MaskChannel::G,
        MaskChannel::B,
        MaskChannel::A,
        MaskChannel::RG,
        MaskChannel::RB,
        MaskChannel::RA,
        MaskChannel::GB,
        MaskChannel::GA,
        MaskChannel::BA,
        MaskChannel::Rgb,
        MaskChannel::Rga,
        MaskChannel::Gba,
        MaskChannel::Rgba,
    ][index])
}

/// `FilterType` is the blend macro: 40 Normal, 41 Multiply, 42 SoftLight, ...
/// Jump table at `0x92C21C8`.
pub fn lipstick_blend_mode(filter_type: u32) -> Result<BlendMode, Refusal> {
    Ok(match filter_type {
        10 | 40 => BlendMode::Normal,
        11 | 41 => BlendMode::Multiply,
        12 | 42 => BlendMode::SoftLight,
        13 | 43 => BlendMode::Lighten,
        14 | 44 => BlendMode::Screen,
        15 | 45 => BlendMode::Overlay,
        16 | 46 => BlendMode::ColorBurn,
        17 | 47 => BlendMode::VividLight,
        18 | 48 => BlendMode::Darken,
        19 | 49 => BlendMode::HardLight,
        20 | 50 => BlendMode::ColorDodge,
        21 | 51 => BlendMode::Color,
        22 | 52 => BlendMode::LinearLight,
        23 | 53 => BlendMode::LinearDodge,
        24..=39 => BlendMode::Normal,
        54 => BlendMode::PinLight,
        _ => return Err(Refusal::MissingInput("lipstick_filter_type")),
    })
}

/// `LipStickImageMakeup` host inputs. `u_mix_color.a` is composed once as
/// `degree * opacity * shelter * region`; the shader then multiplies by the
/// material alpha and the selected mask channel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LipStickParams {
    pub mix_color: [u8; 4],
    pub opacity_percent: f32,
    pub degree01: f32,
    pub shelter_weight: f32,
    pub region_factors: f32,
    pub filter_type: u32,
    pub mask_type: u32,
    pub blend_alpha: bool,
}

impl LipStickParams {
    pub fn color_material(self) -> Result<ColorMaterialParams, Refusal> {
        if !self.shelter_weight.is_finite()
            || !(0.0..=1.0).contains(&self.shelter_weight)
            || !self.region_factors.is_finite()
            || !(0.0..=1.0).contains(&self.region_factors)
        {
            return Err(Refusal::ValueOutOfRange("lipstick weights"));
        }
        Ok(ColorMaterialParams {
            mix_color: self.mix_color,
            opacity_percent: self.opacity_percent,
            degree01: self.degree01 * self.shelter_weight * self.region_factors,
            blend_mode: lipstick_blend_mode(self.filter_type)?,
            mask_channel: lipstick_mask_channel(self.mask_type)?,
            blend_alpha: self.blend_alpha,
        })
    }

    /// Single-mask program (`ProgramType=6`): the only mask is the main mask,
    /// whose define is fixed to `MASK_R` by the host (`+0x2B0 = 0`).
    pub fn single_mask_material(self) -> Result<ColorMaterialParams, Refusal> {
        let mut resolved = self.color_material()?;
        resolved.mask_channel = MaskChannel::R;
        Ok(resolved)
    }
}

/// Recovered `ProgramType` selector (`0x61243BC..0x61243F3`):
/// `5` = no main mask, `6` = main only, `7` = main + extra.
pub fn lipstick_program_type(has_main_mask: bool, has_extra_mask: bool) -> u32 {
    if !has_main_mask {
        5
    } else if has_extra_mask {
        7
    } else {
        6
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LipStickMaskProgram {
    NoMask,
    SingleMask,
    DualMask,
}

pub fn lipstick_mask_program(
    has_main_mask: bool,
    has_extra_mask: bool,
) -> LipStickMaskProgram {
    match lipstick_program_type(has_main_mask, has_extra_mask) {
        5 => LipStickMaskProgram::NoMask,
        6 => LipStickMaskProgram::SingleMask,
        _ => LipStickMaskProgram::DualMask,
    }
}

/// Interleaved float32 layout from the embedded vertex shader
/// (`0x96DCAA0`): `t` position, `iq` material UV, and under
/// `ENABLE_MASK_TEXCOORD` `M` main-mask UV and `ap` extra-mask UV.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LipStickVertexLayout {
    pub position: u32,
    pub material_uv: u32,
    pub mask_uv: Option<u32>,
    pub extra_mask_uv: Option<u32>,
    pub stride: u32,
}

pub fn lipstick_vertex_layout(enable_mask_texcoord: bool) -> LipStickVertexLayout {
    if enable_mask_texcoord {
        LipStickVertexLayout {
            position: 0,
            material_uv: 12,
            mask_uv: Some(20),
            extra_mask_uv: Some(28),
            stride: 36,
        }
    } else {
        LipStickVertexLayout {
            position: 0,
            material_uv: 12,
            mask_uv: None,
            extra_mask_uv: None,
            stride: 20,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorMaterialParams {
    pub mix_color: [u8; 4],
    pub opacity_percent: f32,
    pub degree01: f32,
    pub blend_mode: BlendMode,
    pub mask_channel: MaskChannel,
    pub blend_alpha: bool,
}

impl ColorMaterialParams {
    pub fn blusher(
        mix_color: [u8; 4],
        opacity_percent: f32,
        degree01: f32,
        blend_mode: BlendMode,
    ) -> Self {
        Self {
            mix_color,
            opacity_percent,
            degree01,
            blend_mode,
            mask_channel: MaskChannel::Rgba,
            blend_alpha: false,
        }
    }
}

fn checked_pixels(image: &Image, name: &'static str) -> Result<usize, Refusal> {
    if image.width == 0 || image.height == 0 {
        return Err(Refusal::ShapeMismatch(name));
    }
    let pixels = (image.width as usize)
        .checked_mul(image.height as usize)
        .ok_or(Refusal::ShapeMismatch(name))?;
    match pixels.checked_mul(4) {
        Some(expected) if image.pixels.len() == expected => Ok(pixels),
        _ => Err(Refusal::ShapeMismatch(name)),
    }
}

fn clamp01(value: f32) -> f32 {
    if value.is_nan() { 0.0 } else { value.clamp(0.0, 1.0) }
}

fn byte_even(value: f32) -> u8 {
    let v = value.clamp(0.0, 1.0) * 255.0;
    let floor = v.floor();
    let fraction = v - floor;
    let rounded = if fraction > 0.5 {
        floor + 1.0
    } else if fraction < 0.5 {
        floor
    } else if (floor as i32) & 1 == 0 {
        floor
    } else {
        floor + 1.0
    };
    rounded.clamp(0.0, 255.0) as u8
}

fn sample_rgba(image: &Image, u: f32, v: f32) -> [f32; 4] {
    let u = clamp01(u);
    let v = clamp01(v);
    let x = u * image.width as f32 - 0.5;
    let y = v * image.height as f32 - 0.5;
    let x0f = x.floor();
    let y0f = y.floor();
    let tx = x - x0f;
    let ty = y - y0f;
    let x0 = (x0f as i32).clamp(0, image.width as i32 - 1) as u32;
    let x1 = ((x0f as i32) + 1).clamp(0, image.width as i32 - 1) as u32;
    let y0 = (y0f as i32).clamp(0, image.height as i32 - 1) as u32;
    let y1 = ((y0f as i32) + 1).clamp(0, image.height as i32 - 1) as u32;
    let at = |px: u32, py: u32, c: usize| {
        image.pixels[((py * image.width + px) * 4) as usize + c] as f32 / 255.0
    };
    let mut out = [0.0; 4];
    for (channel, value) in out.iter_mut().enumerate() {
        let top = at(x0, y0, channel) * (1.0 - tx) + at(x1, y0, channel) * tx;
        let bottom = at(x0, y1, channel) * (1.0 - tx) + at(x1, y1, channel) * tx;
        *value = top * (1.0 - ty) + bottom * ty;
    }
    out
}

fn mask_weight(mask: [f32; 4], channel: MaskChannel) -> f32 {
    let value = match channel {
        MaskChannel::R => mask[0],
        MaskChannel::G => mask[1],
        MaskChannel::B => mask[2],
        MaskChannel::A => mask[3],
        MaskChannel::RG => mask[0] + mask[1],
        MaskChannel::RB => mask[0] + mask[2],
        MaskChannel::RA => mask[0] + mask[3],
        MaskChannel::GB => mask[1] + mask[2],
        MaskChannel::GA => mask[1] + mask[3],
        MaskChannel::BA => mask[2] + mask[3],
        MaskChannel::Rgb => mask[0] + mask[1] + mask[2],
        MaskChannel::Rga => mask[0] + mask[1] + mask[3],
        MaskChannel::Rba => mask[0] + mask[2] + mask[3],
        MaskChannel::Gba => mask[1] + mask[2] + mask[3],
        MaskChannel::Rgba => mask[0] + mask[1] + mask[2] + mask[3],
    };
    clamp01(value)
}

fn rgb_to_hsl(rgb: [f32; 3]) -> [f32; 3] {
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let p = if g < b {
        [b, g, -1.0, 2.0 / 3.0]
    } else {
        [g, b, 0.0, -1.0 / 3.0]
    };
    let q = if r < p[0] {
        [p[0], p[1], p[3], r]
    } else {
        [r, p[1], p[2], p[0]]
    };
    let c = q[0] - q[3].min(q[1]);
    let h = ((q[3] - q[1]) / (6.0 * c + 1e-7) + q[2]).abs();
    let v = q[0];
    let lum = v - c * 0.5;
    let sat = c / (1.0 - (2.0 * lum - 1.0).abs() + 1e-7);
    [h, sat, lum]
}

fn hue_to_rgb(hue: f32) -> [f32; 3] {
    [
        (6.0 * hue - 3.0).abs() - 1.0,
        2.0 - (6.0 * hue - 2.0).abs(),
        2.0 - (6.0 * hue - 4.0).abs(),
    ]
    .map(|value| value.clamp(0.0, 1.0))
}

fn hsl_to_rgb(hsl: [f32; 3]) -> [f32; 3] {
    let rgb = hue_to_rgb(hsl[0]);
    let c = (1.0 - (2.0 * hsl[2] - 1.0).abs()) * hsl[1];
    [
        (rgb[0] - 0.5) * c + hsl[2],
        (rgb[1] - 0.5) * c + hsl[2],
        (rgb[2] - 0.5) * c + hsl[2],
    ]
}

fn color_dodge(base: f32, blend: f32) -> f32 {
    if blend == 1.0 { blend } else { (base / (1.0 - blend)).min(1.0) }
}

fn color_burn(base: f32, blend: f32) -> f32 {
    if blend == 0.0 { blend } else { (1.0 - (1.0 - base) / blend).max(0.0) }
}

fn soft_light(base: f32, blend: f32) -> f32 {
    if blend < 0.5 {
        2.0 * base * blend + base * base * (1.0 - 2.0 * blend)
    } else {
        base.sqrt() * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend)
    }
}

fn overlay(base: f32, blend: f32) -> f32 {
    if base < 0.5 {
        2.0 * base * blend
    } else {
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
    }
}

fn vivid_light(base: f32, blend: f32) -> f32 {
    if blend < 0.5 {
        color_burn(base, 2.0 * blend)
    } else {
        color_dodge(base, 2.0 * (blend - 0.5))
    }
}

fn linear_light(base: f32, blend: f32) -> f32 {
    if blend < 0.5 {
        (base + 2.0 * blend - 1.0).max(0.0)
    } else {
        (base + 2.0 * (blend - 0.5)).min(1.0)
    }
}

fn pin_light(base: f32, blend: f32) -> f32 {
    if blend < 0.5 { base.min(2.0 * blend) } else { base.max(2.0 * (blend - 0.5)) }
}

fn blend_channel(base: f32, blend: f32, mode: BlendMode) -> f32 {
    match mode {
        BlendMode::Normal => blend,
        BlendMode::Darken => base.min(blend),
        BlendMode::Multiply => base * blend,
        BlendMode::ColorBurn => color_burn(base, blend),
        BlendMode::LinearBurn => (base + blend - 1.0).max(0.0),
        BlendMode::Lighten => base.max(blend),
        BlendMode::Screen => 1.0 - (1.0 - base) * (1.0 - blend),
        BlendMode::ColorDodge => color_dodge(base, blend),
        BlendMode::LinearDodge => (base + blend).min(1.0),
        BlendMode::Overlay => overlay(base, blend),
        BlendMode::SoftLight => soft_light(base, blend),
        BlendMode::HardLight => overlay(blend, base),
        BlendMode::VividLight => vivid_light(base, blend),
        BlendMode::LinearLight => linear_light(base, blend),
        BlendMode::PinLight => pin_light(base, blend),
        BlendMode::HardMix => if vivid_light(base, blend) < 0.5 { 0.0 } else { 1.0 },
        BlendMode::Difference => (base - blend).abs(),
        BlendMode::Exclusion => base + blend - 2.0 * base * blend,
        BlendMode::Subtract => (base + blend - 1.0).max(0.0),
        BlendMode::Average => (base + blend) * 0.5,
        BlendMode::Negation => 1.0 - (1.0 - base - blend).abs(),
        BlendMode::Reflect => if blend == 1.0 { blend } else { (base * base / (1.0 - blend)).min(1.0) },
        BlendMode::Glow => if base == 1.0 { base } else { (blend * blend / (1.0 - base)).min(1.0) },
        BlendMode::Phoenix => base.min(blend) - base.max(blend) + 1.0,
        BlendMode::Hue | BlendMode::Saturation | BlendMode::Color | BlendMode::Luminosity => 0.0,
    }
}

pub(crate) fn blend_rgb(base: [f32; 3], blend: [f32; 3], mode: BlendMode) -> [f32; 3] {
    match mode {
        BlendMode::Hue => {
            let b = rgb_to_hsl(base);
            let s = rgb_to_hsl(blend);
            hsl_to_rgb([s[0], b[1], b[2]])
        }
        BlendMode::Saturation => {
            let b = rgb_to_hsl(base);
            let s = rgb_to_hsl(blend);
            hsl_to_rgb([b[0], s[1], b[2]])
        }
        BlendMode::Color => {
            let b = rgb_to_hsl(base);
            let s = rgb_to_hsl(blend);
            hsl_to_rgb([s[0], s[1], b[2]])
        }
        BlendMode::Luminosity => {
            let b = rgb_to_hsl(base);
            let s = rgb_to_hsl(blend);
            hsl_to_rgb([b[0], b[1], s[2]])
        }
        _ => [
            blend_channel(base[0], blend[0], mode),
            blend_channel(base[1], blend[1], mode),
            blend_channel(base[2], blend[2], mode),
        ],
    }
}

fn store_blend(
    output: &mut [u8], source: [f32; 3], source_alpha: u8,
    reference: [f32; 3], factor: f32, blend_alpha: bool,
) {
    for channel in 0..3 {
        output[channel] = byte_even(if blend_alpha {
            reference[channel]
        } else {
            source[channel] * (1.0 - factor) + reference[channel] * factor
        });
    }
    output[3] = if blend_alpha {
        byte_even(factor * source_alpha as f32 / 255.0)
    } else {
        source_alpha
    };
}

/// Executes the recovered `BlendColorLayerWithMask.frag` pixel law.
///
/// One `uv` serves both textures here because the recovered vertex shader uses
/// independent `a_material_texcoord` / `a_mask_texcoord` attributes. A caller
/// with divergent mask UV cannot express it through this entry point.
pub fn apply_color_layer(
    source: &Image,
    owner_mask: &Image,
    uv: &[[f32; 2]],
    params: ColorMaterialParams,
) -> Result<Image, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    checked_pixels(owner_mask, "owner_mask")?;
    if uv.len() != pixels {
        return Err(Refusal::ShapeMismatch("uv"));
    }
    if !params.opacity_percent.is_finite() || !(0.0..=100.0).contains(&params.opacity_percent) {
        return Err(Refusal::ValueOutOfRange("opacity_percent"));
    }
    if !params.degree01.is_finite() || !(0.0..=1.0).contains(&params.degree01) {
        return Err(Refusal::ValueOutOfRange("degree01"));
    }
    if !uv.iter().all(|p| p[0].is_finite() && p[1].is_finite()) {
        return Err(Refusal::ValueOutOfRange("uv"));
    }
    let blend_color = [
        params.mix_color[0] as f32 / 255.0,
        params.mix_color[1] as f32 / 255.0,
        params.mix_color[2] as f32 / 255.0,
    ];
    let mix_alpha = params.mix_color[3] as f32 / 255.0
        * params.opacity_percent / 100.0
        * params.degree01;
    let mut output = source.clone();
    for pixel in 0..pixels {
        let mask = sample_rgba(owner_mask, uv[pixel][0], uv[pixel][1]);
        let factor = clamp01(mix_alpha * mask_weight(mask, params.mask_channel));
        let base = pixel * 4;
        if factor <= 0.01 {
            continue;
        }
        let src = [
            source.pixels[base] as f32 / 255.0,
            source.pixels[base + 1] as f32 / 255.0,
            source.pixels[base + 2] as f32 / 255.0,
        ];
        let blended = blend_rgb(src, blend_color, params.blend_mode);
        store_blend(&mut output.pixels[base..base + 4], src, source.pixels[base + 3],
            blended, factor, params.blend_alpha);
    }
    Ok(output)
}

/// Executes the recovered `BlendImageLayerWithMask.frag` pixel law.
/// `factor = material.a * u_mix_color.a * mask_weight`.
/// The same one-UV limitation as [`apply_color_layer`] applies.
pub fn apply_image_layer(
    source: &Image,
    material: &Image,
    owner_mask: &Image,
    uv: &[[f32; 2]],
    params: ColorMaterialParams,
) -> Result<Image, Refusal> {
    apply_image_layer_dual(source, material, owner_mask, uv, uv, params)
}

/// `BlendImageLayerWithMask.frag` with the two recovered UV streams:
/// the owner mask uses the geometry UV and the material uses its own
/// `v_bgmasktexcoordOut` stream. `apply_image_layer` is the same-UV form.
pub fn apply_image_layer_dual(
    source: &Image,
    material: &Image,
    owner_mask: &Image,
    mask_uv: &[[f32; 2]],
    material_uv: &[[f32; 2]],
    params: ColorMaterialParams,
) -> Result<Image, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    checked_pixels(material, "material")?;
    checked_pixels(owner_mask, "owner_mask")?;
    if mask_uv.len() != pixels || material_uv.len() != pixels {
        return Err(Refusal::ShapeMismatch("uv"));
    }
    if !params.opacity_percent.is_finite() || !(0.0..=100.0).contains(&params.opacity_percent) {
        return Err(Refusal::ValueOutOfRange("opacity_percent"));
    }
    if !params.degree01.is_finite() || !(0.0..=1.0).contains(&params.degree01) {
        return Err(Refusal::ValueOutOfRange("degree01"));
    }
    if !mask_uv
        .iter()
        .chain(material_uv)
        .all(|p| p[0].is_finite() && p[1].is_finite())
    {
        return Err(Refusal::ValueOutOfRange("uv"));
    }
    let mix_alpha = params.mix_color[3] as f32 / 255.0
        * params.opacity_percent / 100.0
        * params.degree01;
    let mut output = source.clone();
    dual_rows(
        &mut output.pixels, source, material, owner_mask, mask_uv, material_uv,
        params, mix_alpha, 0, source.height, 0,
    );
    Ok(output)
}

/// Pixels from which the banded form beats the serial loop on this machine.
pub const MAKEUP_PARALLEL_MIN_PIXELS: usize = 65_536;

/// Row-band form of [`apply_image_layer_dual`]; bands are disjoint and the
/// output is byte-identical to the serial call.
#[allow(clippy::too_many_arguments)]
pub fn apply_image_layer_dual_parallel(
    source: &Image,
    material: &Image,
    owner_mask: &Image,
    mask_uv: &[[f32; 2]],
    material_uv: &[[f32; 2]],
    params: ColorMaterialParams,
    threads: usize,
) -> Result<Image, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    if threads <= 1 || pixels < MAKEUP_PARALLEL_MIN_PIXELS {
        return apply_image_layer_dual(source, material, owner_mask, mask_uv, material_uv, params);
    }
    checked_pixels(material, "material")?;
    checked_pixels(owner_mask, "owner_mask")?;
    if mask_uv.len() != pixels || material_uv.len() != pixels {
        return Err(Refusal::ShapeMismatch("uv"));
    }
    if !params.opacity_percent.is_finite() || !(0.0..=100.0).contains(&params.opacity_percent) {
        return Err(Refusal::ValueOutOfRange("opacity_percent"));
    }
    if !params.degree01.is_finite() || !(0.0..=1.0).contains(&params.degree01) {
        return Err(Refusal::ValueOutOfRange("degree01"));
    }
    if !mask_uv.iter().chain(material_uv).all(|p| p[0].is_finite() && p[1].is_finite()) {
        return Err(Refusal::ValueOutOfRange("uv"));
    }
    let mut banded = source.clone();
    let mix_alpha = params.mix_color[3] as f32 / 255.0
        * params.opacity_percent / 100.0
        * params.degree01;
    let band_rows = (source.height as usize).div_ceil(threads).max(1);
    let band_bytes = band_rows * source.width as usize * 4;
    let width = source.width;
    std::thread::scope(|scope| {
        for (band, chunk) in banded.pixels.chunks_mut(band_bytes).enumerate() {
            let y0 = (band * band_rows) as u32;
            let rows = (chunk.len() / (width as usize * 4)) as u32;
            scope.spawn(move || {
                dual_rows(
                    chunk, source, material, owner_mask, mask_uv, material_uv,
                    params, mix_alpha, y0, rows, y0,
                );
            });
        }
    });
    Ok(banded)
}

#[allow(clippy::too_many_arguments)]
fn dual_rows(
    out: &mut [u8],
    source: &Image,
    material: &Image,
    owner_mask: &Image,
    mask_uv: &[[f32; 2]],
    material_uv: &[[f32; 2]],
    params: ColorMaterialParams,
    mix_alpha: f32,
    start_y: u32,
    rows: u32,
    band_y0: u32,
) {
    let width = source.width as usize;
    for y in start_y..start_y + rows {
        for x in 0..width {
            let pixel = y as usize * width + x;
            let local = (y - band_y0) as usize * width + x;
            let mask = sample_rgba(owner_mask, mask_uv[pixel][0], mask_uv[pixel][1]);
            let material_sample =
                sample_rgba(material, material_uv[pixel][0], material_uv[pixel][1]);
            let factor = clamp01(
                material_sample[3] * mix_alpha * mask_weight(mask, params.mask_channel),
            );
            if factor <= 0.01 {
                out[local * 4..local * 4 + 4]
                    .copy_from_slice(&source.pixels[pixel * 4..pixel * 4 + 4]);
                continue;
            }
            let src = [
                source.pixels[pixel * 4] as f32 / 255.0,
                source.pixels[pixel * 4 + 1] as f32 / 255.0,
                source.pixels[pixel * 4 + 2] as f32 / 255.0,
            ];
            let blended = blend_rgb(
                src,
                [material_sample[0], material_sample[1], material_sample[2]],
                params.blend_mode,
            );
            store_blend(
                &mut out[local * 4..local * 4 + 4],
                src,
                source.pixels[pixel * 4 + 3],
                blended,
                factor,
                params.blend_alpha,
            );
        }
    }
}

/// PF107 / PF90033 `LipStickImageMakeup` stage. The shader multiplies the
/// material alpha by `u_mix_color.a` and **two** masks: the main mask's R
/// channel plus the extra mask selected by `LipMaskType`. The host composes
/// `degree * opacity * shelter * region` once. Material and mask use their
/// separate UV streams.
#[allow(clippy::too_many_arguments)]
pub fn lipstick_image_layer(
    source: &Image,
    material: &Image,
    mask_main: &Image,
    mask_extra: &Image,
    mask_uv: &[[f32; 2]],
    material_uv: &[[f32; 2]],
    params: LipStickParams,
) -> Result<Image, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    checked_pixels(material, "material")?;
    checked_pixels(mask_main, "mask_main")?;
    checked_pixels(mask_extra, "mask_extra")?;
    if mask_uv.len() != pixels || material_uv.len() != pixels {
        return Err(Refusal::ShapeMismatch("uv"));
    }
    if !mask_uv
        .iter()
        .chain(material_uv)
        .all(|uv| uv[0].is_finite() && uv[1].is_finite())
    {
        return Err(Refusal::ValueOutOfRange("uv"));
    }
    let resolved = params.color_material()?;
    let mix_alpha = resolved.mix_color[3] as f32 / 255.0
        * resolved.opacity_percent / 100.0
        * resolved.degree01;
    let mut output = source.clone();
    for pixel in 0..pixels {
        let material_sample =
            sample_rgba(material, material_uv[pixel][0], material_uv[pixel][1]);
        let main_sample = sample_rgba(mask_main, mask_uv[pixel][0], mask_uv[pixel][1]);
        let extra_sample =
            sample_rgba(mask_extra, mask_uv[pixel][0], mask_uv[pixel][1]);
        let factor = clamp01(
            material_sample[3]
                * mix_alpha
                * mask_weight(main_sample, MaskChannel::R)
                * mask_weight(extra_sample, resolved.mask_channel),
        );
        let base = pixel * 4;
        if factor <= 0.01 {
            continue;
        }
        let src = [
            source.pixels[base] as f32 / 255.0,
            source.pixels[base + 1] as f32 / 255.0,
            source.pixels[base + 2] as f32 / 255.0,
        ];
        let blended = blend_rgb(
            src,
            [material_sample[0], material_sample[1], material_sample[2]],
            resolved.blend_mode,
        );
        store_blend(
            &mut output.pixels[base..base + 4],
            src,
            source.pixels[base + 3],
            blended,
            factor,
            params.blend_alpha,
        );
    }
    Ok(output)
}

/// Single-mask `LipStickImageMakeup` program (`ProgramType=6`). The shader has
/// no `w()`/extra mask; factor is `material.a * u_mix_color.a * MASK_R`.
pub fn lipstick_image_layer_single_mask(
    source: &Image,
    material: &Image,
    mask_main: &Image,
    mask_uv: &[[f32; 2]],
    material_uv: &[[f32; 2]],
    params: LipStickParams,
) -> Result<Image, Refusal> {
    apply_image_layer_dual(
        source,
        material,
        mask_main,
        mask_uv,
        material_uv,
        params.single_mask_material()?,
    )
}

/// PF106 profile: single mask `BlusherMaskType=15` -> `MASK_RGBA`.
pub fn blusher_color_makeup(
    source: &Image,
    owner_mask: &Image,
    uv: &[[f32; 2]],
    mix_color: [u8; 4],
    opacity_percent: f32,
    degree01: f32,
    blend_mode: BlendMode,
) -> Result<Image, Refusal> {
    apply_color_layer(
        source,
        owner_mask,
        uv,
        ColorMaterialParams::blusher(mix_color, opacity_percent, degree01, blend_mode),
    )
}

/// PF116 profile: `BlusherImageMakeup` + `BlusherMaskType=15` -> `MASK_RGBA`.
pub fn blusher_image_makeup(
    source: &Image,
    material: &Image,
    owner_mask: &Image,
    uv: &[[f32; 2]],
    mix_color: [u8; 4],
    opacity_percent: f32,
    degree01: f32,
    blend_mode: BlendMode,
) -> Result<Image, Refusal> {
    apply_image_layer(
        source,
        material,
        owner_mask,
        uv,
        ColorMaterialParams::blusher(mix_color, opacity_percent, degree01, blend_mode),
    )
}

/// Shared parameters for the makeup cells whose class and FilterType are known
/// but whose material/mask come from the same-job resources.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MakeupLayer {
    pub mix_color: [u8; 4],
    pub opacity_percent: f32,
    pub degree01: f32,
    pub filter_type: u32,
    pub mask_channel: MaskChannel,
}

impl MakeupLayer {
    fn params(self) -> Result<ColorMaterialParams, Refusal> {
        Ok(ColorMaterialParams {
            mix_color: self.mix_color,
            opacity_percent: self.opacity_percent,
            degree01: self.degree01,
            blend_mode: lipstick_blend_mode(self.filter_type)?,
            mask_channel: self.mask_channel,
            blend_alpha: false,
        })
    }

    /// Same construction as the CPU entries, for GPU callers.
    pub fn params_of(self) -> Result<ColorMaterialParams, Refusal> {
        self.params()
    }
}

/// `BlendColorLayerWithMask` rows: PF100/117 (mask RB), PF103/30007 (mask RG).
pub fn color_makeup(
    source: &Image,
    owner_mask: &Image,
    uv: &[[f32; 2]],
    layer: MakeupLayer,
) -> Result<Image, Refusal> {
    apply_color_layer(source, owner_mask, uv, layer.params()?)
}

/// `BlendImageLayerWithMask` rows such as PF90032 (mask RGB, FilterType 41/42/44).
pub fn image_makeup(
    source: &Image,
    material: &Image,
    owner_mask: &Image,
    mask_uv: &[[f32; 2]],
    material_uv: &[[f32; 2]],
    layer: MakeupLayer,
) -> Result<Image, Refusal> {
    apply_image_layer_dual(source, material, owner_mask, mask_uv, material_uv, layer.params()?)
}

/// `EyePupilMaskType=0` skips both mask branches at `0x617E31B`, so the factor
/// is `material.a * u_mix_color.a` with no mask term.
pub fn image_makeup_unmasked(
    source: &Image,
    material: &Image,
    material_uv: &[[f32; 2]],
    layer: MakeupLayer,
) -> Result<Image, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    checked_pixels(material, "material")?;
    if material_uv.len() != pixels {
        return Err(Refusal::ShapeMismatch("uv"));
    }
    if !material_uv.iter().all(|p| p[0].is_finite() && p[1].is_finite()) {
        return Err(Refusal::ValueOutOfRange("uv"));
    }
    let params = layer.params()?;
    let mix_alpha = params.mix_color[3] as f32 / 255.0
        * params.opacity_percent / 100.0
        * params.degree01;
    let mut output = source.clone();
    for pixel in 0..pixels {
        let sample = sample_rgba(material, material_uv[pixel][0], material_uv[pixel][1]);
        let factor = clamp01(sample[3] * mix_alpha);
        let base = pixel * 4;
        if factor <= 0.01 {
            continue;
        }
        let src = [
            source.pixels[base] as f32 / 255.0,
            source.pixels[base + 1] as f32 / 255.0,
            source.pixels[base + 2] as f32 / 255.0,
        ];
        let blended = blend_rgb(src, [sample[0], sample[1], sample[2]], params.blend_mode);
        store_blend(
            &mut output.pixels[base..base + 4],
            src,
            source.pixels[base + 3],
            blended,
            factor,
            params.blend_alpha,
        );
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(color: [u8; 4]) -> Image {
        Image::solid(1, 1, color)
    }

    #[test]
    fn blend_alpha_writes_reference_rgb_without_mixing_twice() {
        let source = pixel([100, 150, 200, 100]);
        let mask = pixel([255, 0, 0, 0]);
        let mut params = ColorMaterialParams::blusher(
            [200, 100, 50, 128], 100.0, 1.0, BlendMode::Normal);
        params.blend_alpha = true;
        assert_eq!(apply_color_layer(&source, &mask, &[[0.5, 0.5]], params)
            .unwrap().pixel(0, 0), [200, 100, 50, 50]);

        let material = pixel([20, 40, 60, 128]);
        params.mix_color[3] = 255;
        assert_eq!(apply_image_layer(&source, &material, &mask, &[[0.5, 0.5]], params)
            .unwrap().pixel(0, 0), [20, 40, 60, 50]);
    }

    #[test]
    fn blusher_profile_is_pf106_color_layer() {
        assert_eq!(META.len(), 2);
        assert_eq!(META[0].pf, BLUSHER_PF);
        assert_eq!(BLUSHER_FILTER_TYPE, 11);
        assert_eq!(BLUSHER_MASK_VALUE, 15);
        assert_eq!(BLUSHER_SHADER, "glsl/OverLay/BlendColorLayerWithMask.frag");
        assert_eq!(BLUSHER_BLEND_BINDING, "CELL_BLEND_MACRO_OPEN");
        assert_eq!(META[1].pf, BLUSHER_IMAGE_PF);
        assert_eq!(BLUSHER_IMAGE_FILTER_TYPE, 42);
        assert_eq!(BLUSHER_IMAGE_SHADER, "glsl/OverLay/BlendImageLayerWithMask.frag");
        assert_eq!(blusher_mask_channel(15).unwrap(), MaskChannel::Rgba);
        assert_eq!(
            blusher_mask_channel(14).unwrap_err(),
            Refusal::MissingInput("blusher_mask_channel")
        );
        assert_eq!(BlendMode::from_macro("COLOR").unwrap(), BlendMode::Color);
        assert_eq!(
            BlendMode::from_macro("UNKNOWN_VENDOR_BLEND").unwrap_err(),
            Refusal::MissingInput("blend_mode_macro")
        );
    }

    #[test]
    fn multiply_is_not_source_over() {
        let source = pixel([100, 150, 200, 77]);
        let mask = pixel([255, 255, 255, 255]);
        let out = blusher_color_makeup(
            &source,
            &mask,
            &[[0.5, 0.5]],
            [234, 112, 129, 255],
            100.0,
            1.0,
            BlendMode::Multiply,
        )
        .unwrap();
        let expected = [
            byte_even((100.0 / 255.0) * (234.0 / 255.0)),
            byte_even((150.0 / 255.0) * (112.0 / 255.0)),
            byte_even((200.0 / 255.0) * (129.0 / 255.0)),
            77,
        ];
        assert_eq!(out.pixel(0, 0), expected);
        assert_ne!(out.pixel(0, 0), [234, 112, 129, 77]);
    }

    #[test]
    fn zero_degree_is_identity() {
        let source = pixel([10, 20, 30, 40]);
        let mask = pixel([255, 255, 255, 255]);
        let out = blusher_color_makeup(
            &source, &mask, &[[0.5, 0.5]], [1, 2, 3, 255], 100.0, 0.0,
            BlendMode::Color,
        )
        .unwrap();
        assert_eq!(out, source);
    }

    #[test]
    fn gate_below_point_zero_one_is_identity() {
        let source = pixel([10, 20, 30, 40]);
        let mask = pixel([255, 255, 255, 255]);
        let out = blusher_color_makeup(
            &source, &mask, &[[0.5, 0.5]], [1, 2, 3, 255], 0.5, 1.0,
            BlendMode::Color,
        )
        .unwrap();
        assert_eq!(out, source);
    }

    #[test]
    fn parallel_bands_match_the_serial_layer() {
        let (width, height) = (320u32, 240u32);
        let pixels = (width * height) as usize;
        let source = Image::new(
            width,
            height,
            (0..pixels * 4).map(|i| ((i * 37 + 11) % 256) as u8).collect(),
        )
        .unwrap();
        let material = Image::new(
            160,
            120,
            (0..160 * 120 * 4).map(|i| ((i * 13 + 5) % 256) as u8).collect(),
        )
        .unwrap();
        let mask = Image::new(
            width,
            height,
            (0..pixels * 4).map(|i| ((i * 7 + 3) % 256) as u8).collect(),
        )
        .unwrap();
        let uv: Vec<[f32; 2]> = (0..pixels)
            .map(|i| {
                [
                    (i % width as usize) as f32 / width as f32,
                    (i / width as usize) as f32 / height as f32,
                ]
            })
            .collect();
        let params = ColorMaterialParams {
            mix_color: [200, 120, 60, 200],
            opacity_percent: 80.0,
            degree01: 0.7,
            blend_mode: BlendMode::SoftLight,
            mask_channel: MaskChannel::Rgba,
            blend_alpha: false,
        };
        let serial = apply_image_layer_dual(&source, &material, &mask, &uv, &uv, params).unwrap();
        for threads in [2usize, 5, 8] {
            let parallel = apply_image_layer_dual_parallel(
                &source, &material, &mask, &uv, &uv, params, threads,
            )
            .unwrap();
            assert_eq!(serial, parallel, "threads={threads}");
        }
    }

    #[test]
    fn shape_and_uv_refusal() {
        let source = pixel([10, 20, 30, 40]);
        let mask = pixel([255, 255, 255, 255]);
        assert_eq!(
            apply_color_layer(
                &source, &mask, &[], ColorMaterialParams::blusher([1, 2, 3, 255], 100.0, 1.0, BlendMode::Color)
            )
            .unwrap_err(),
            Refusal::ShapeMismatch("uv")
        );
        let malformed = Image { width: 1, height: 1, pixels: vec![0; 3] };
        assert_eq!(
            apply_image_layer(
                &source, &malformed, &mask, &[[0.5, 0.5]], ColorMaterialParams::blusher([1, 2, 3, 255], 100.0, 1.0, BlendMode::Color)
            )
            .unwrap_err(),
            Refusal::ShapeMismatch("material")
        );
    }

    #[test]
    fn alpha_is_source_unless_blend_alpha() {
        let source = pixel([10, 20, 30, 77]);
        let mask = pixel([255, 255, 255, 255]);
        let keep = apply_color_layer(
            &source, &mask, &[[0.5, 0.5]],
            ColorMaterialParams::blusher([1, 2, 3, 255], 100.0, 1.0, BlendMode::Color),
        )
        .unwrap();
        assert_eq!(keep.pixel(0, 0)[3], 77);
        let mut params = ColorMaterialParams::blusher([1, 2, 3, 255], 100.0, 1.0, BlendMode::Color);
        params.blend_alpha = true;
        let cut = apply_color_layer(&source, &mask, &[[0.5, 0.5]], params).unwrap();
        assert_eq!(cut.pixel(0, 0)[3], 77);
    }

    #[test]
    fn mask_channel_rgba_and_r_differ() {
        let source = pixel([10, 20, 30, 40]);
        let mask = pixel([0, 128, 0, 0]);
        let rgba = apply_color_layer(
            &source, &mask, &[[0.5, 0.5]],
            ColorMaterialParams::blusher([255, 0, 0, 255], 100.0, 1.0, BlendMode::Normal),
        )
        .unwrap();
        let mut params = ColorMaterialParams::blusher([255, 0, 0, 255], 100.0, 1.0, BlendMode::Normal);
        params.mask_channel = MaskChannel::R;
        let r_only = apply_color_layer(&source, &mask, &[[0.5, 0.5]], params).unwrap();
        assert_ne!(rgba, r_only);
        assert_eq!(r_only, source);
    }

    #[test]
    fn image_layer_uses_material_rgb_and_alpha() {
        let source = pixel([100, 150, 200, 77]);
        let material = pixel([200, 100, 50, 128]);
        let mask = pixel([255, 255, 255, 255]);
        let out = blusher_image_makeup(
            &source,
            &material,
            &mask,
            &[[0.5, 0.5]],
            [255, 255, 255, 255],
            100.0,
            1.0,
            BlendMode::Multiply,
        )
        .unwrap();
        let factor = 128.0 / 255.0;
        let expected = [
            byte_even((100.0 / 255.0) * (1.0 - factor) + (100.0 / 255.0) * (200.0 / 255.0) * factor),
            byte_even((150.0 / 255.0) * (1.0 - factor) + (150.0 / 255.0) * (100.0 / 255.0) * factor),
            byte_even((200.0 / 255.0) * (1.0 - factor) + (200.0 / 255.0) * (50.0 / 255.0) * factor),
            77,
        ];
        assert_eq!(out.pixel(0, 0), expected);
    }

    #[test]
    fn full_blend_family_is_finite_and_clamped() {
        let modes = [
            BlendMode::Normal, BlendMode::Darken, BlendMode::Multiply, BlendMode::ColorBurn,
            BlendMode::LinearBurn, BlendMode::Lighten, BlendMode::Screen, BlendMode::ColorDodge,
            BlendMode::LinearDodge, BlendMode::Overlay, BlendMode::SoftLight, BlendMode::HardLight,
            BlendMode::VividLight, BlendMode::LinearLight, BlendMode::PinLight, BlendMode::HardMix,
            BlendMode::Difference, BlendMode::Exclusion, BlendMode::Subtract, BlendMode::Hue,
            BlendMode::Saturation, BlendMode::Color, BlendMode::Luminosity, BlendMode::Average,
            BlendMode::Negation, BlendMode::Reflect, BlendMode::Glow, BlendMode::Phoenix,
        ];
        let source = pixel([37, 141, 222, 255]);
        let mask = pixel([255, 255, 255, 255]);
        for mode in modes {
            let out = blusher_color_makeup(
                &source, &mask, &[[0.5, 0.5]], [234, 112, 129, 255], 100.0, 1.0, mode,
            )
            .unwrap();
            assert_eq!(out.pixel(0, 0)[3], 255);
        }
    }

    #[test]
    fn lipstick_mask_and_filter_tables_are_recovered() {
        assert_eq!(lipstick_mask_channel(3).unwrap(), MaskChannel::R);
        assert_eq!(lipstick_mask_channel(5).unwrap(), MaskChannel::G);
        assert_eq!(lipstick_mask_channel(17).unwrap(), MaskChannel::Rgba);
        assert!(lipstick_mask_channel(0).is_err());
        assert_eq!(lipstick_blend_mode(41).unwrap(), BlendMode::Multiply);
        assert_eq!(lipstick_blend_mode(42).unwrap(), BlendMode::SoftLight);
        assert_eq!(lipstick_blend_mode(40).unwrap(), BlendMode::Normal);
        assert!(lipstick_blend_mode(99).is_err());
    }

    #[test]
    fn lipstick_degree_times_opacity_times_shelter_times_region_once() {
        let source = pixel([100, 150, 200, 255]);
        let material = pixel([128, 64, 32, 255]);
        let mask = pixel([255, 0, 0, 0]);
        let uv = [[0.5, 0.5]];
        let params = LipStickParams {
            mix_color: [0, 0, 0, 255],
            opacity_percent: 100.0,
            degree01: 0.5,
            shelter_weight: 1.0,
            region_factors: 1.0,
            filter_type: 41,
            mask_type: 3,
            blend_alpha: false,
        };
        let direct = apply_image_layer_dual(
            &source,
            &material,
            &mask,
            &uv,
            &uv,
            params.color_material().unwrap(),
        )
        .unwrap();
        let staged =
            lipstick_image_layer(&source, &material, &mask, &mask, &uv, &uv, params).unwrap();
        assert_eq!(direct.pixel(0, 0), staged.pixel(0, 0));
        assert_eq!(staged.pixel(0, 0)[3], 255);

        let green_mask = pixel([0, 255, 0, 0]);
        let r_channel =
            lipstick_image_layer(&source, &material, &mask, &green_mask, &uv, &uv, params)
                .unwrap();
        assert_eq!(r_channel, source);
        let g_params = LipStickParams { mask_type: 5, ..params };
        let g_channel =
            lipstick_image_layer(&source, &material, &mask, &green_mask, &uv, &uv, g_params)
                .unwrap();
        assert_ne!(g_channel, source);
    }

    #[test]
    fn lipstick_material_and_mask_use_separate_uv_streams() {
        let source = pixel([40, 80, 120, 200]);
        let material = Image::new(2, 1, vec![255, 0, 0, 255, 0, 255, 0, 255]).unwrap();
        let mask = pixel([255, 0, 0, 0]);
        let mask_uv = [[0.5, 0.5]];
        let params = LipStickParams {
            mix_color: [0, 0, 0, 255],
            opacity_percent: 100.0,
            degree01: 1.0,
            shelter_weight: 1.0,
            region_factors: 1.0,
            filter_type: 40,
            mask_type: 3,
            blend_alpha: false,
        };
        let left = lipstick_image_layer(
            &source, &material, &mask, &mask, &mask_uv, &[[0.25, 0.5]], params,
        )
        .unwrap();
        let right = lipstick_image_layer(
            &source, &material, &mask, &mask, &mask_uv, &[[0.75, 0.5]], params,
        )
        .unwrap();
        assert_ne!(left.pixel(0, 0), right.pixel(0, 0));
        assert_eq!(left.pixel(0, 0)[3], 200);
        assert_eq!(right.pixel(0, 0)[3], 200);
    }

    #[test]
    fn lipstick_native_goldens_match_both_masks() {
        let source = pixel([100, 150, 200, 255]);
        let material = pixel([80, 120, 160, 255]);
        let main = pixel([255, 0, 0, 255]);
        let extra = pixel([128, 255, 255, 255]);
        let base = LipStickParams {
            mix_color: [255, 255, 255, 255],
            opacity_percent: 100.0,
            degree01: 1.0,
            shelter_weight: 1.0,
            region_factors: 1.0,
            filter_type: 41,
            mask_type: 3,
            blend_alpha: false,
        };
        let uv = [[0.5, 0.5]];
        let multiply =
            lipstick_image_layer(&source, &material, &main, &extra, &uv, &uv, base).unwrap();
        assert_eq!(multiply.pixel(0, 0), [66, 110, 163, 255]);
        let soft = lipstick_image_layer(
            &source,
            &material,
            &main,
            &extra,
            &uv,
            &uv,
            LipStickParams { filter_type: 42, ..base },
        )
        .unwrap();
        assert_eq!(soft.pixel(0, 0), [89, 148, 203, 255]);
        let double = lipstick_image_layer(
            &source,
            &material,
            &pixel([128, 255, 255, 255]),
            &pixel([128, 255, 255, 255]),
            &uv,
            &uv,
            base,
        )
        .unwrap();
        assert_eq!(double.pixel(0, 0), [83, 130, 181, 255]);
        let zero = lipstick_image_layer(
            &source,
            &material,
            &main,
            &pixel([0, 255, 255, 255]),
            &uv,
            &uv,
            base,
        )
        .unwrap();
        assert_eq!(zero, source);
    }

    #[test]
    fn lipstick_program_type_and_vertex_layout_match_r31() {
        assert_eq!(lipstick_program_type(false, false), 5);
        assert_eq!(lipstick_program_type(false, true), 5);
        assert_eq!(lipstick_program_type(true, false), 6);
        assert_eq!(lipstick_program_type(true, true), 7);
        assert_eq!(lipstick_mask_program(true, false), LipStickMaskProgram::SingleMask);
        assert_eq!(lipstick_mask_program(true, true), LipStickMaskProgram::DualMask);
        let off = lipstick_vertex_layout(false);
        assert_eq!((off.position, off.material_uv, off.stride), (0, 12, 20));
        assert_eq!((off.mask_uv, off.extra_mask_uv), (None, None));
        let on = lipstick_vertex_layout(true);
        assert_eq!((on.position, on.material_uv, on.stride), (0, 12, 36));
        assert_eq!((on.mask_uv, on.extra_mask_uv), (Some(20), Some(28)));
    }

    #[test]
    fn lipstick_single_and_dual_mask_gate_is_strict_gt_0_01() {
        let source = pixel([100, 150, 200, 200]);
        let material = pixel([80, 120, 160, 255]);
        let mask = pixel([255, 0, 0, 0]);
        let uv = [[0.5, 0.5]];
        let base = LipStickParams {
            mix_color: [255, 255, 255, 255],
            opacity_percent: 1.0,
            degree01: 1.0,
            shelter_weight: 1.0,
            region_factors: 1.0,
            filter_type: 41,
            mask_type: 3,
            blend_alpha: false,
        };
        // factor == 0.01 -> shader `m > .01` is false -> identity.
        let single = lipstick_image_layer_single_mask(
            &source, &material, &mask, &uv, &uv, base).unwrap();
        let dual = lipstick_image_layer(
            &source, &material, &mask, &mask, &uv, &uv, base).unwrap();
        assert_eq!(single, source);
        assert_eq!(dual, source);

        let above = LipStickParams { opacity_percent: 2.0, ..base };
        let single_above = lipstick_image_layer_single_mask(
            &source, &material, &mask, &uv, &uv, above).unwrap();
        let dual_above =
            lipstick_image_layer(&source, &material, &mask, &mask, &uv, &uv, above).unwrap();
        assert_ne!(single_above, source);
        assert_ne!(dual_above, source);
        assert_eq!(single_above.pixel(0, 0)[3], 200);
        assert_eq!(dual_above.pixel(0, 0)[3], 200);
    }

    #[test]
    fn lipstick_single_mask_is_always_r_but_dual_uses_lip_mask_type() {
        let source = pixel([100, 150, 200, 255]);
        let material = pixel([80, 120, 160, 255]);
        let main = pixel([255, 0, 0, 0]);
        let green = pixel([0, 255, 0, 0]);
        let uv = [[0.5, 0.5]];
        let params = LipStickParams {
            mix_color: [255, 255, 255, 255],
            opacity_percent: 100.0,
            degree01: 1.0,
            shelter_weight: 1.0,
            region_factors: 1.0,
            filter_type: 41,
            mask_type: 5,
            blend_alpha: false,
        };
        // Single-mask host define is fixed MASK_R even when LipMaskType=5.
        let single = lipstick_image_layer_single_mask(
            &source, &material, &main, &uv, &uv, params).unwrap();
        assert_ne!(single, source);
        // Dual-mask extra uses LipMaskType=5 -> G; main R * extra G.
        let dual_zero = lipstick_image_layer(
            &source, &material, &main, &pixel([255, 0, 0, 0]), &uv, &uv, params).unwrap();
        let dual_green = lipstick_image_layer(
            &source, &material, &main, &green, &uv, &uv, params).unwrap();
        assert_eq!(dual_zero, source);
        assert_ne!(dual_green, source);
    }
}
