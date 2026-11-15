//! `glsl/Paint/MaskBlendToEffect.frag`, `BlendMode_Texture_Normal`.
//!
//! Texture roles were recovered from the real PE at `0x5ECF3C0` and
//! `0x5F83950`:
//!
//! | unit | field | shader sampler | role |
//! | --- | --- | --- | --- |
//! | 0 | `+0x18` (`at`) | `s_originalTexture` | original |
//! | 1 | `+0x20` (`aM`) | `s_effectTexture` | effect |
//! | 2 | `+0x28` (`F`) | `s_maskTexture` | mask |
//!
//! `u_blendDegree` is set to `1.0` for PF30009 (`0x5ECF445`), and
//! `u_isReverseAlpha` comes from `[effect+0xD5]` (`0x5ECF459`). The original
//! and mask share the geometry UV; the effect texture has its own UV stream.
//!
//! The shader law is `mix(original.rgb, effect.rgb, mask * degree)` and the
//! output alpha is the original alpha. The target framebuffer blend state
//! (`GL_SRC_ALPHA`/`GL_ONE_MINUS_SRC_ALPHA` at `0x302/0x303`) is a separate
//! host step and is not applied here.

use crate::{Image, Refusal};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalBlend {
    pub degree: f32,
    pub reverse_alpha: bool,
}

impl NormalBlend {
    pub const PF30009: Self = Self {
        degree: 1.0,
        reverse_alpha: false,
    };
}

/// `mix` in encoded RGBA8 space; alpha stays the original alpha.
pub fn pixel(original: [u8; 4], effect: [u8; 4], mask: u8, params: NormalBlend) -> Result<[u8; 4], Refusal> {
    if !params.degree.is_finite() || !(0.0..=1.0).contains(&params.degree) {
        return Err(Refusal::ValueOutOfRange("u_blendDegree"));
    }
    let mask = mask as f32 / 255.0;
    let mask = if params.reverse_alpha { 1.0 - mask } else { mask };
    let weight = mask * params.degree;
    let mut out = [0u8; 4];
    for channel in 0..3 {
        out[channel] = ((original[channel] as f32)
            + (effect[channel] as f32 - original[channel] as f32) * weight)
            .round()
            .clamp(0.0, 255.0) as u8;
    }
    out[3] = original[3];
    Ok(out)
}

/// Full-frame stage. The effect plane may have a different size; `effect_uv`
/// supplies its per-pixel sampling stream. `None` means the same normalised
/// pixel-centre UV as the original.
pub fn apply(
    original: &Image,
    effect: &Image,
    mask: &[u8],
    effect_uv: Option<&[[f32; 2]]>,
    params: NormalBlend,
) -> Result<Image, Refusal> {
    validate(original, effect, mask, effect_uv, params)?;
    let mut out = original.clone();
    rows(
        &mut out.pixels, &original.pixels, effect, mask, effect_uv, params,
        original.width as usize, original.height as usize, 0, original.height as usize,
    )?;
    Ok(out)
}

fn validate(
    original: &Image,
    effect: &Image,
    mask: &[u8],
    effect_uv: Option<&[[f32; 2]]>,
    params: NormalBlend,
) -> Result<(), Refusal> {
    original
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("original"))?;
    effect
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("effect"))?;
    let pixels = original.pixel_count();
    if mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("s_maskTexture"));
    }
    if !params.degree.is_finite() || !(0.0..=1.0).contains(&params.degree) {
        return Err(Refusal::ValueOutOfRange("u_blendDegree"));
    }
    if let Some(uv) = effect_uv {
        if uv.len() != pixels {
            return Err(Refusal::ShapeMismatch("v_bgmasktexcoordOut"));
        }
        if uv.iter().any(|point| !point[0].is_finite() || !point[1].is_finite()) {
            return Err(Refusal::ValueOutOfRange("v_bgmasktexcoordOut"));
        }
    }
    Ok(())
}

/// Pixels from which the banded form beats the serial loop.
pub const TEXTURE_BLEND_PARALLEL_MIN_PIXELS: usize = 65_536;

/// Row-band form of [`apply`]; output is byte-identical to the serial call.
pub fn apply_parallel(
    original: &Image,
    effect: &Image,
    mask: &[u8],
    effect_uv: Option<&[[f32; 2]]>,
    params: NormalBlend,
    threads: usize,
) -> Result<Image, Refusal> {
    let pixels = original.pixel_count();
    if threads <= 1 || pixels < TEXTURE_BLEND_PARALLEL_MIN_PIXELS {
        return apply(original, effect, mask, effect_uv, params);
    }
    validate(original, effect, mask, effect_uv, params)?;
    let width = original.width as usize;
    let height = original.height as usize;
    let band_rows = height.div_ceil(threads).max(1);
    let mut out = original.clone();
    let bytes = width * 4;
    std::thread::scope(|scope| {
        for (band, chunk) in out.pixels.chunks_mut(band_rows * bytes).enumerate() {
            let start_row = band * band_rows;
            let count = chunk.len() / bytes;
            if count == 0 {
                continue;
            }
            let start = start_row * width;
            let source = &original.pixels[start * 4..(start + count * width) * 4];
            let masks = &mask[start..start + count * width];
            let uvs = effect_uv.map(|points| &points[start..start + count * width]);
            scope.spawn(move || {
                rows(
                    chunk, source, effect, masks, uvs, params, width, height, start_row, count,
                )
                .expect("band validated before the scope");
            });
        }
    });
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn rows(
    out: &mut [u8],
    original: &[u8],
    effect: &Image,
    mask: &[u8],
    effect_uv: Option<&[[f32; 2]]>,
    params: NormalBlend,
    width: usize,
    full_height: usize,
    start_row: usize,
    count: usize,
) -> Result<(), Refusal> {
    for y in 0..count {
        for x in 0..width {
            let index = y * width + x;
            let offset = index * 4;
            let uv = effect_uv
                .map(|points| points[index])
                .unwrap_or([
                    (x as f32 + 0.5) / width as f32,
                    ((start_row + y) as f32 + 0.5) / full_height as f32,
                ]);
            let effect_pixel = sample(effect, uv);
            out[offset..offset + 4].copy_from_slice(&pixel(
                original[offset..offset + 4].try_into().unwrap(),
                effect_pixel,
                mask[index],
                params,
            )?);
        }
    }
    Ok(())
}

fn sample(image: &Image, uv: [f32; 2]) -> [u8; 4] {
    let u = uv[0].clamp(0.0, 1.0);
    let v = uv[1].clamp(0.0, 1.0);
    let px = u * image.width as f32 - 0.5;
    let py = v * image.height as f32 - 0.5;
    let x0 = px.floor();
    let y0 = py.floor();
    let tx = px - x0;
    let ty = py - y0;
    let x0 = (x0 as i64).clamp(0, image.width as i64 - 1) as u32;
    let x1 = ((x0 as i64) + 1).clamp(0, image.width as i64 - 1) as u32;
    let y0 = (y0 as i64).clamp(0, image.height as i64 - 1) as u32;
    let y1 = ((y0 as i64) + 1).clamp(0, image.height as i64 - 1) as u32;
    let at = |x: u32, y: u32, channel: usize| {
        image.pixels[((y * image.width + x) * 4) as usize + channel] as f32
    };
    let mut out = [0u8; 4];
    for channel in 0..4 {
        let top = at(x0, y0, channel) + (at(x1, y0, channel) - at(x0, y0, channel)) * tx;
        let bottom = at(x0, y1, channel) + (at(x1, y1, channel) - at(x0, y1, channel)) * tx;
        out[channel] = (top + (bottom - top) * ty).round().clamp(0.0, 255.0) as u8;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(degree: f32, reverse_alpha: bool) -> NormalBlend {
        NormalBlend { degree, reverse_alpha }
    }

    #[test]
    fn shader_vectors_and_reverse_alpha() {
        let original = [10, 20, 30, 200];
        let effect = [110, 220, 130, 0];
        let forward = pixel(original, effect, 128, params(0.5, false)).unwrap();
        assert_eq!(forward, [35, 70, 55, 200]);
        let reverse = pixel(original, effect, 128, params(1.0, true)).unwrap();
        assert_eq!(reverse, [60, 120, 80, 200]);
    }

    #[test]
    fn mask_extremes_are_identity_and_full_effect() {
        let original = [10, 20, 30, 77];
        let effect = [200, 210, 220, 0];
        assert_eq!(pixel(original, effect, 0, params(1.0, false)).unwrap(), original);
        assert_eq!(
            pixel(original, effect, 255, params(1.0, false)).unwrap(),
            [200, 210, 220, 77]
        );
        assert_eq!(pixel(original, effect, 0, params(1.0, true)).unwrap(), [200, 210, 220, 77]);
    }

    #[test]
    fn frame_stage_uses_separate_effect_uv_and_keeps_alpha() {
        let original = Image::new(2, 1, vec![10, 20, 30, 40, 50, 60, 70, 80]).unwrap();
        let effect = Image::new(2, 1, vec![200, 0, 0, 255, 0, 200, 0, 255]).unwrap();
        let mask = [255, 255];
        let same = apply(&original, &effect, &mask, None, params(1.0, false)).unwrap();
        assert_eq!(&same.pixels[0..4], &[200, 0, 0, 40]);
        assert_eq!(&same.pixels[4..8], &[0, 200, 0, 80]);
        let flipped = apply(
            &original,
            &effect,
            &mask,
            Some(&[[0.75, 0.5], [0.25, 0.5]]),
            params(1.0, false),
        )
        .unwrap();
        assert_eq!(&flipped.pixels[0..4], &[0, 200, 0, 40]);
        assert_eq!(&flipped.pixels[4..8], &[200, 0, 0, 80]);
    }

    #[test]
    fn parallel_bands_match_the_serial_frame_stage() {
        let (width, height) = (320u32, 240u32);
        let pixels = (width * height) as usize;
        let original = Image::new(
            width,
            height,
            (0..pixels * 4).map(|i| ((i * 37 + 11) % 256) as u8).collect(),
        )
        .unwrap();
        let effect = Image::new(
            160,
            120,
            (0..160 * 120 * 4).map(|i| ((i * 13 + 5) % 256) as u8).collect(),
        )
        .unwrap();
        let mask: Vec<u8> = (0..pixels).map(|i| ((i * 7 + 3) % 256) as u8).collect();
        let uv: Vec<[f32; 2]> = (0..pixels)
            .map(|i| {
                [
                    (i % width as usize) as f32 / width as f32,
                    (i / width as usize) as f32 / height as f32,
                ]
            })
            .collect();
        for (uvs, params) in [
            (None, NormalBlend::PF30009),
            (Some(uv.as_slice()), NormalBlend::PF30009),
            (Some(uv.as_slice()), NormalBlend { degree: 0.37, reverse_alpha: true }),
        ] {
            let serial = apply(&original, &effect, &mask, uvs, params).unwrap();
            for threads in [2usize, 5, 8] {
                let parallel =
                    apply_parallel(&original, &effect, &mask, uvs, params, threads).unwrap();
                assert_eq!(serial, parallel, "threads={threads}");
            }
        }
    }

    #[test]
    fn refusals_are_typed() {
        let original = Image::solid(1, 1, [1, 2, 3, 4]);
        let effect = Image::solid(1, 1, [5, 6, 7, 8]);
        assert_eq!(
            apply(&original, &effect, &[], None, params(1.0, false)),
            Err(Refusal::ShapeMismatch("s_maskTexture"))
        );
        assert_eq!(
            apply(&original, &effect, &[255], None, params(1.5, false)),
            Err(Refusal::ValueOutOfRange("u_blendDegree"))
        );
        assert_eq!(
            apply(&original, &effect, &[255], Some(&[[f32::NAN, 0.0]]), params(1.0, false)),
            Err(Refusal::ValueOutOfRange("v_bgmasktexcoordOut"))
        );
    }
}
