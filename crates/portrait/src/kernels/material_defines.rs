//! Explicit `BlendImageLayerWithMask.frag` define stages.
//!
//! The frozen `makeup_material` module implements the base shader. This module
//! adds the two defines that change its inputs, with no default-off booleans:
//!
//! ```glsl
//! #ifdef REFINE_MATERIAL_COLOR
//!     if (material_color.a > 0.0) material_color.rgb /= material_color.a;
//! #endif
//! #ifdef FILPMASK
//!     factor = factor * (1.0 - mask_weight);
//! #else
//!     factor = factor * mask_weight;
//! #endif
//! ```

use crate::kernels::makeup_material::{BlendMode, MaskChannel};
use crate::{EffectMeta, Image, Refusal};

/// These are pipeline defines, not new ParamFlag rows.
pub const META: &[EffectMeta] = &[];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceMaskMode {
    /// Shader without `FILPMASK`: `factor *= mask_weight`.
    Multiply,
    /// Shader with `FILPMASK`: `factor *= (1.0 - mask_weight)`.
    Invert,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialColorMode {
    /// Shader without `REFINE_MATERIAL_COLOR`: use RGBA as sampled.
    Raw,
    /// Shader with `REFINE_MATERIAL_COLOR`: `rgb /= a` when `a > 0`.
    Unpremultiply,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DefinedStageParams {
    pub mix_color: [u8; 4],
    pub opacity_percent: f32,
    pub degree01: f32,
    pub blend_mode: BlendMode,
    pub mask_channel: MaskChannel,
    pub blend_alpha: bool,
    pub face_mask: FaceMaskMode,
    pub material_color: MaterialColorMode,
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

/// Exact `GetMaskWeight()` body for `MASK_RGBA`.
/// Shader bytes: `clamp(mask_color.r + mask_color.g + mask_color.b + mask_color.a, 0.0, 1.0)`.
/// This is sum-clamped, not max, union, or dot.
pub fn mask_rgba_sum(mask: [f32; 4]) -> f32 {
    clamp01(mask[0] + mask[1] + mask[2] + mask[3])
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
        MaskChannel::Rgba => mask_rgba_sum(mask),
    };
    clamp01(value)
}

fn rgb_to_hsl(rgb: [f32; 3]) -> [f32; 3] {
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let p = if g < b { [b, g, -1.0, 2.0 / 3.0] } else { [g, b, 0.0, -1.0 / 3.0] };
    let q = if r < p[0] { [p[0], p[1], p[3], r] } else { [r, p[1], p[2], p[0]] };
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
    .map(|v| v.clamp(0.0, 1.0))
}

fn hsl_to_rgb(hsl: [f32; 3]) -> [f32; 3] {
    let rgb = hue_to_rgb(hsl[0]);
    let c = (1.0 - (2.0 * hsl[2] - 1.0).abs()) * hsl[1];
    [(rgb[0] - 0.5) * c + hsl[2], (rgb[1] - 0.5) * c + hsl[2], (rgb[2] - 0.5) * c + hsl[2]]
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
    if base < 0.5 { 2.0 * base * blend } else { 1.0 - 2.0 * (1.0 - base) * (1.0 - blend) }
}

fn vivid_light(base: f32, blend: f32) -> f32 {
    if blend < 0.5 { color_burn(base, 2.0 * blend) } else { color_dodge(base, 2.0 * (blend - 0.5)) }
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
        BlendMode::LinearLight => if blend < 0.5 { (base + 2.0 * blend - 1.0).max(0.0) } else { (base + 2.0 * (blend - 0.5)).min(1.0) },
        BlendMode::PinLight => if blend < 0.5 { base.min(2.0 * blend) } else { base.max(2.0 * (blend - 0.5)) },
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

fn blend_rgb(base: [f32; 3], blend: [f32; 3], mode: BlendMode) -> [f32; 3] {
    match mode {
        BlendMode::Hue => hsl_to_rgb([rgb_to_hsl(blend)[0], rgb_to_hsl(base)[1], rgb_to_hsl(base)[2]]),
        BlendMode::Saturation => hsl_to_rgb([rgb_to_hsl(base)[0], rgb_to_hsl(blend)[1], rgb_to_hsl(base)[2]]),
        BlendMode::Color => hsl_to_rgb([rgb_to_hsl(blend)[0], rgb_to_hsl(blend)[1], rgb_to_hsl(base)[2]]),
        BlendMode::Luminosity => hsl_to_rgb([rgb_to_hsl(base)[0], rgb_to_hsl(base)[1], rgb_to_hsl(blend)[2]]),
        _ => [
            blend_channel(base[0], blend[0], mode),
            blend_channel(base[1], blend[1], mode),
            blend_channel(base[2], blend[2], mode),
        ],
    }
}

/// Full `BlendImageLayerWithMask.frag` with explicit `FILPMASK` and
/// `REFINE_MATERIAL_COLOR` modes.
pub fn apply_image_defined(
    source: &Image,
    material: &Image,
    owner_mask: &Image,
    uv: &[[f32; 2]],
    params: DefinedStageParams,
) -> Result<Image, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    checked_pixels(material, "material")?;
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
    let mix_alpha = params.mix_color[3] as f32 / 255.0
        * params.opacity_percent / 100.0
        * params.degree01;
    let mut output = source.clone();
    for pixel in 0..pixels {
        let mask = sample_rgba(owner_mask, uv[pixel][0], uv[pixel][1]);
        let mut material_color = sample_rgba(material, uv[pixel][0], uv[pixel][1]);
        if params.material_color == MaterialColorMode::Unpremultiply && material_color[3] > 0.0 {
            material_color[0] /= material_color[3];
            material_color[1] /= material_color[3];
            material_color[2] /= material_color[3];
        }
        let weight = mask_weight(mask, params.mask_channel);
        let face = match params.face_mask {
            FaceMaskMode::Multiply => weight,
            FaceMaskMode::Invert => 1.0 - weight,
        };
        let factor = clamp01(material_color[3] * mix_alpha * face);
        let base = pixel * 4;
        if factor <= 0.01 {
            continue;
        }
        let src = [
            source.pixels[base] as f32 / 255.0,
            source.pixels[base + 1] as f32 / 255.0,
            source.pixels[base + 2] as f32 / 255.0,
        ];
        let blended = blend_rgb(src, [material_color[0], material_color[1], material_color[2]], params.blend_mode);
        if params.blend_alpha {
            for channel in 0..3 {
                output.pixels[base + channel] = byte_even(blended[channel]);
            }
            output.pixels[base + 3] = byte_even(factor * source.pixels[base + 3] as f32 / 255.0);
        } else {
            for channel in 0..3 {
                output.pixels[base + channel] =
                    byte_even(src[channel] * (1.0 - factor) + blended[channel] * factor);
            }
            output.pixels[base + 3] = source.pixels[base + 3];
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(c: [u8; 4]) -> Image {
        Image::solid(1, 1, c)
    }

    fn params() -> DefinedStageParams {
        DefinedStageParams {
            mix_color: [255, 255, 255, 255],
            opacity_percent: 100.0,
            degree01: 1.0,
            blend_mode: BlendMode::Multiply,
            mask_channel: MaskChannel::Rgba,
            blend_alpha: false,
            face_mask: FaceMaskMode::Multiply,
            material_color: MaterialColorMode::Raw,
        }
    }

    #[test]
    fn mask_rgba_is_sum_clamped_not_max_union_or_dot() {
        assert_eq!(mask_rgba_sum([0.2, 0.2, 0.2, 0.2]), 0.8);
        assert_eq!(mask_rgba_sum([0.5, 0.5, 0.5, 0.5]), 1.0);
        assert_eq!(mask_rgba_sum([0.1, 0.0, 0.0, 0.0]), 0.1);
        assert_ne!(mask_rgba_sum([0.2, 0.2, 0.2, 0.2]), 0.2);
    }

    #[test]
    fn filpmask_inverts_factor() {
        let source = pixel([100, 150, 200, 77]);
        let material = pixel([200, 100, 50, 255]);
        let mask = pixel([255, 255, 255, 255]);
        let mut p = params();
        p.face_mask = FaceMaskMode::Invert;
        let out = apply_image_defined(&source, &material, &mask, &[[0.5, 0.5]], p).unwrap();
        assert_eq!(out, source);
        let mask_zero = pixel([0, 0, 0, 0]);
        let out2 = apply_image_defined(&source, &material, &mask_zero, &[[0.5, 0.5]], p).unwrap();
        assert_ne!(out2, source);
    }

    #[test]
    fn refine_material_color_unpremultiplies() {
        let source = pixel([100, 100, 100, 255]);
        let material = pixel([128, 128, 128, 128]);
        let mask = pixel([255, 255, 255, 255]);
        let mut p = params();
        p.material_color = MaterialColorMode::Unpremultiply;
        p.blend_mode = BlendMode::Normal;
        let out = apply_image_defined(&source, &material, &mask, &[[0.5, 0.5]], p).unwrap();
        // material rgb 128/255 / (128/255) = 1.0, factor = 128/255.
        let factor = 128.0 / 255.0;
        let expected = byte_even((100.0 / 255.0) * (1.0 - factor) + 1.0 * factor);
        assert_eq!(out.pixel(0, 0)[0], expected);
    }

    #[test]
    fn zero_degree_is_identity() {
        let source = pixel([10, 20, 30, 40]);
        let material = pixel([200, 100, 50, 255]);
        let mask = pixel([255, 255, 255, 255]);
        let mut p = params();
        p.degree01 = 0.0;
        assert_eq!(apply_image_defined(&source, &material, &mask, &[[0.5, 0.5]], p).unwrap(), source);
    }

    #[test]
    fn shape_refusal() {
        let source = pixel([10, 20, 30, 40]);
        let material = pixel([200, 100, 50, 255]);
        let mask = pixel([255, 255, 255, 255]);
        assert_eq!(
            apply_image_defined(&source, &material, &mask, &[], params()).unwrap_err(),
            Refusal::ShapeMismatch("uv")
        );
    }

    #[test]
    fn alpha_preserved() {
        let source = pixel([10, 20, 30, 77]);
        let material = pixel([200, 100, 50, 255]);
        let mask = pixel([255, 255, 255, 255]);
        let out = apply_image_defined(&source, &material, &mask, &[[0.5, 0.5]], params()).unwrap();
        assert_eq!(out.pixel(0, 0)[3], 77);
    }

    #[test]
    fn blend_alpha_writes_reference_rgb_without_mixing_twice() {
        let source = pixel([100, 150, 200, 100]);
        let material = pixel([200, 100, 50, 128]);
        let mask = pixel([255, 255, 255, 255]);
        let mut p = params();
        p.blend_mode = BlendMode::Normal;
        p.blend_alpha = true;
        let out = apply_image_defined(&source, &material, &mask, &[[0.5, 0.5]], p).unwrap();
        assert_eq!(out.pixel(0, 0), [200, 100, 50, 50]);
    }
}
