//! PF115 face sticker, transcribed from `Sticker/StickerWithMask.frag`
//! (`0x966FE41`, 7050 bytes).
//!
//! ```text
//! b = source(ae); t = b; p = GetMaskWeight();  #ifndef FLIPMASK p = 1 - p
//! v = global_scale * p
//! for each element: pos/rot/scale -> material UV -> g = material.a
//!                   t.rgb = mix(t.rgb, Blend(t.rgb, material.rgb), g)
//! out = vec4(mix(b.rgb, t.rgb, v), b.a)
//! ```
//! `MaskType=15` maps to `MASK_RGBA` (`15 -> 14 -> table[14] = 13`).

use crate::kernels::makeup_material::{blend_rgb, lipstick_blend_mode, BlendMode};
use crate::{Access, EffectMeta, Image, Refusal};

pub const META: &[EffectMeta] = &[EffectMeta {
    pf: 115,
    key: "faceDecorationDegree",
    label: "Face decoration (sticker)",
    access: Access::Cpu,
    inputs: &[
        "material RGBA",
        "MASK_RGBA mask",
        "element list (pos/scale/rot)",
        "global scale, fl, fk",
        "FilterType",
        "flip mask flag",
    ],
}];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Element {
    pub pos: [f32; 2],
    pub scale: f32,
    pub rot: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StickerParams {
    pub global_scale: f32,
    /// Horizontal aspect used before the rotation (`fl` in the shader).
    pub fl: f32,
    /// Canvas scale divisor (`fk` in the shader).
    pub fk: f32,
    pub filter_type: u32,
    /// `true` keeps `p`, `false` applies the shader's default `1 - p`.
    pub flip_mask: bool,
    pub blend_alpha: bool,
}

fn mask_weight(pixel: [f32; 4]) -> f32 {
    (pixel[0] + pixel[1] + pixel[2] + pixel[3]).clamp(0.0, 1.0)
}

fn sample(image: &Image, uv: [f32; 2]) -> [f32; 4] {
    let u = uv[0].clamp(0.0, 1.0);
    let v = uv[1].clamp(0.0, 1.0);
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
    let at = |px: usize, py: usize, c: usize| {
        image.pixels[(py * image.width as usize + px) * 4 + c] as f32 / 255.0
    };
    let mut out = [0.0f32; 4];
    for (channel, value) in out.iter_mut().enumerate() {
        let top = at(x0, y0, channel) * (1.0 - tx) + at(x1, y0, channel) * tx;
        let bottom = at(x0, y1, channel) * (1.0 - tx) + at(x1, y1, channel) * tx;
        *value = top * (1.0 - ty) + bottom * ty;
    }
    out
}

pub fn apply(
    source: &Image,
    material: &Image,
    mask: &Image,
    elements: &[Element],
    params: StickerParams,
) -> Result<Image, Refusal> {
    source.validate()?;
    material.validate().map_err(|_| Refusal::ShapeMismatch("material"))?;
    mask.validate().map_err(|_| Refusal::ShapeMismatch("mask"))?;
    if (source.width, source.height) != (mask.width, mask.height) {
        return Err(Refusal::ShapeMismatch("mask"));
    }
    for value in [params.global_scale, params.fl, params.fk] {
        if !value.is_finite() || value <= 0.0 {
            return Err(Refusal::ValueOutOfRange("sticker scale"));
        }
    }
    if elements.iter().any(|element| {
        !element.pos[0].is_finite()
            || !element.pos[1].is_finite()
            || !element.scale.is_finite()
            || element.scale <= 0.0
            || !element.rot.is_finite()
    }) {
        return Err(Refusal::ValueOutOfRange("sticker element"));
    }
    let mode: BlendMode = lipstick_blend_mode(params.filter_type)?;
    let mut output = source.clone();
    for y in 0..source.height {
        for x in 0..source.width {
            let index = (y * source.width + x) as usize;
            let at = index * 4;
            let base = [
                source.pixels[at] as f32 / 255.0,
                source.pixels[at + 1] as f32 / 255.0,
                source.pixels[at + 2] as f32 / 255.0,
            ];
            let mask_sample = sample(mask, [
                (x as f32 + 0.5) / source.width as f32,
                (y as f32 + 0.5) / source.height as f32,
            ]);
            let mut weight = mask_weight(mask_sample);
            if !params.flip_mask {
                weight = 1.0 - weight;
            }
            let v = (params.global_scale * weight).clamp(0.0, 1.0);
            let mut blended = base;
            let ae = [
                (x as f32 + 0.5) / source.width as f32,
                (y as f32 + 0.5) / source.height as f32,
            ];
            for element in elements {
                let dx = ae[0] - element.pos[0];
                let dy = ae[1] - element.pos[1];
                let m = (-element.rot).cos();
                let s = (-element.rot).sin();
                let fx = m * (dx * params.fl) + s * dy;
                let fy = -s * (dx * params.fl) + m * dy;
                let uv = [
                    fx / params.fl / (params.fk * element.scale) + 0.5,
                    fy / (params.fk * element.scale) + 0.5,
                ];
                let material_sample = sample(material, uv);
                let g = material_sample[3].clamp(0.0, 1.0);
                let reference = blend_rgb(
                    blended,
                    [material_sample[0], material_sample[1], material_sample[2]],
                    mode,
                );
                for channel in 0..3 {
                    blended[channel] += (reference[channel] - blended[channel]) * g;
                }
            }
            for channel in 0..3 {
                let value = base[channel] + (blended[channel] - base[channel]) * v;
                output.pixels[at + channel] = (value * 255.0).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ELEMENT: Element = Element { pos: [0.5, 0.5], scale: 1.0, rot: 0.0 };

    fn params() -> StickerParams {
        StickerParams {
            global_scale: 1.0,
            fl: 1.0,
            fk: 1.0,
            filter_type: 10,
            flip_mask: true,
            blend_alpha: false,
        }
    }

    #[test]
    fn zero_mask_is_identity() {
        let source = Image::solid(2, 2, [10, 20, 30, 77]);
        let material = Image::solid(1, 1, [200, 100, 50, 255]);
        let mask = Image::solid(2, 2, [0, 0, 0, 0]);
        assert_eq!(apply(&source, &material, &mask, &[ELEMENT], params()).unwrap(), source);
    }

    #[test]
    fn full_mask_takes_the_material_colour() {
        let source = Image::solid(2, 2, [10, 20, 30, 77]);
        let material = Image::solid(1, 1, [200, 100, 50, 255]);
        let mask = Image::solid(2, 2, [255, 255, 255, 255]);
        let out = apply(&source, &material, &mask, &[ELEMENT], params()).unwrap();
        assert_eq!(out.pixel(0, 0), [200, 100, 50, 77]);
    }

    #[test]
    fn flip_default_inverts_the_mask() {
        let source = Image::solid(1, 1, [10, 20, 30, 77]);
        let material = Image::solid(1, 1, [200, 100, 50, 255]);
        let mask = Image::solid(1, 1, [255, 255, 255, 255]);
        let mut flipped = params();
        flipped.flip_mask = false;
        assert_eq!(apply(&source, &material, &mask, &[ELEMENT], flipped).unwrap(), source);
    }

    #[test]
    fn malformed_element_is_refused() {
        let source = Image::solid(1, 1, [0; 4]);
        let material = Image::solid(1, 1, [0; 4]);
        let mask = Image::solid(1, 1, [0; 4]);
        let bad = Element { pos: [0.5, 0.5], scale: 0.0, rot: 0.0 };
        assert_eq!(
            apply(&source, &material, &mask, &[bad], params()),
            Err(Refusal::ValueOutOfRange("sticker element"))
        );
    }
}
