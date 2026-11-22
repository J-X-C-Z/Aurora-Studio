//! PF407 final colour consumer: type 143 UV-map apply.
//!
//! Recovered fragment (`.rdata 0x96E0D8E`, exactly one
//! `uniform sampler2D f,mo` in the PE):
//!   `vec4 d = texture2D(mo, ae);`
//!   `vec2 b = ae + (mp == 1 ? d.pq : d.st);`
//!   `gl_FragColor = texture2D(f, b)` (or a 9-tap Catmull sample of `f`)
//!
//! `f` is the colour source, `mo` the UV/displacement field, `mp` the channel
//! selector stored at `filter+0x2A8` (`0` = RG, `1` = BA), and `F/O` the
//! source size for `DEF_CATMULL_ROM`.

use crate::{Image, Refusal};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Channel {
    /// `mp == 0`, field `.st` (R/G)
    Rg,
    /// `mp == 1`, field `.pq` (B/A)
    Ba,
}

/// `apply_core(source_colour, uv_field, channel, catmull_size)`.
///
/// `catmull_size = Some((F, O))` selects the `DEF_CATMULL_ROM` branch; `None`
/// is plain `texture2D(f, b)`.
pub fn apply(
    source: &Image,
    field: &Image,
    channel: Channel,
    catmull_size: Option<(f32, f32)>,
) -> Result<Image, Refusal> {
    if source.width == 0
        || source.height == 0
        || source.pixels.len() != source.width as usize * source.height as usize * 4
    {
        return Err(Refusal::ShapeMismatch("source"));
    }
    if field.width != source.width
        || field.height != source.height
        || field.pixels.len() != source.pixels.len()
    {
        return Err(Refusal::ShapeMismatch("uv field"));
    }
    if let Some((width, height)) = catmull_size {
        if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
            return Err(Refusal::ValueOutOfRange("catmull size"));
        }
    }

    let width = source.width as usize;
    let height = source.height as usize;
    let mut output = Vec::with_capacity(source.pixels.len());
    for y in 0..height {
        for x in 0..width {
            let uv = (
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            );
            let d = sample_bilinear(field, uv.0, uv.1);
            let offset = match channel {
                Channel::Rg => (d[0], d[1]),
                Channel::Ba => (d[2], d[3]),
            };
            let target = (uv.0 + offset.0, uv.1 + offset.1);
            let color = match catmull_size {
                Some(size) => sample_catmull_rom(source, target.0, target.1, size),
                None => sample_bilinear(source, target.0, target.1),
            };
            for c in 0..4 {
                output.push(quantize(color[c]));
            }
        }
    }
    Image::new(source.width, source.height, output).ok_or(Refusal::ShapeMismatch("output"))
}

pub(crate) fn sample_bilinear(image: &Image, u: f32, v: f32) -> [f32; 4] {
    let width = image.width as f32;
    let height = image.height as f32;
    let x = (u * width - 0.5).clamp(0.0, width - 1.0);
    let y = (v * height - 0.5).clamp(0.0, height - 1.0);
    let x0 = x.floor();
    let y0 = y.floor();
    let x1 = (x0 + 1.0).min(width - 1.0);
    let y1 = (y0 + 1.0).min(height - 1.0);
    let fx = x - x0;
    let fy = y - y0;
    let mut out = [0.0f32; 4];
    for c in 0..4 {
        let top = lerp(
            texel(image, x0 as usize, y0 as usize, c),
            texel(image, x1 as usize, y0 as usize, c),
            fx,
        );
        let bottom = lerp(
            texel(image, x0 as usize, y1 as usize, c),
            texel(image, x1 as usize, y1 as usize, c),
            fx,
        );
        out[c] = lerp(top, bottom, fy);
    }
    out
}

pub(crate) fn sample_catmull_rom(image: &Image, u: f32, v: f32, size: (f32, f32)) -> [f32; 4] {
    let (tex_w, tex_h) = size;
    let sx = u * tex_w;
    let sy = v * tex_h;
    let px = (sx - 0.5).floor() + 0.5;
    let py = (sy - 0.5).floor() + 0.5;
    let fx = sx - px;
    let fy = sy - py;
    let w = |f: f32| {
        [
            f * (-0.5 + f * (1.0 - 0.5 * f)),
            1.0 + f * f * (-2.5 + 1.5 * f),
            f * (0.5 + f * (2.0 - 1.5 * f)),
            f * f * (-0.5 + 0.5 * f),
        ]
    };
    let wx = w(fx);
    let wy = w(fy);
    let w12x = wx[1] + wx[2];
    let w12y = wy[1] + wy[2];
    let t12x = px + wx[2] / w12x;
    let t12y = py + wy[2] / w12y;
    let taps = [
        (px - 1.0, py - 1.0, wx[0] * wy[0]),
        (t12x, py - 1.0, w12x * wy[0]),
        (px + 2.0, py - 1.0, wx[3] * wy[0]),
        (px - 1.0, t12y, wx[0] * w12y),
        (t12x, t12y, w12x * w12y),
        (px + 2.0, t12y, wx[3] * w12y),
        (px - 1.0, py + 2.0, wx[0] * wy[3]),
        (t12x, py + 2.0, w12x * wy[3]),
        (px + 2.0, py + 2.0, wx[3] * wy[3]),
    ];
    let mut color = [0.0f32; 4];
    for (tx, ty, weight) in taps {
        let tap = sample_bilinear(image, tx / tex_w, ty / tex_h);
        for c in 0..4 {
            color[c] += tap[c] * weight;
        }
    }
    color
}

fn texel(image: &Image, x: usize, y: usize, c: usize) -> f32 {
    image.pixels[(y * image.width as usize + x) * 4 + c] as f32 / 255.0
}

fn quantize(value: f32) -> u8 {
    (value * 255.0).clamp(0.0, 255.0).round_ties_even() as u8
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
#[cfg(test)]
mod tests {
    use super::*;

    /// 4x1 gradient source and a zero/short UV field.
    fn source() -> Image {
        Image::new(4, 1, (0..4 * 4).map(|i| (i * 60) as u8).collect()).unwrap()
    }

    fn field(offset: [f32; 4]) -> Image {
        let mut pixels = Vec::with_capacity(4 * 4);
        for _ in 0..4 {
            for value in offset {
                pixels.push((value * 255.0) as u8);
            }
        }
        Image::new(4, 1, pixels).unwrap()
    }

    #[test]
    fn zero_field_is_identity_and_channels_are_isolated() {
        let src = source();
        assert_eq!(apply(&src, &field([0.0; 4]), Channel::Rg, None).unwrap(), src);
        let rg = apply(&src, &field([0.25, 0.0, 0.0, 0.0]), Channel::Rg, None).unwrap();
        let ba = apply(&src, &field([0.25, 0.0, 0.0, 0.0]), Channel::Ba, None).unwrap();
        assert_ne!(rg, src);
        assert_eq!(ba, src, "BA must ignore the RG offset");
    }

    #[test]
    fn malformed_inputs_and_sizes_are_refused() {
        let src = source();
        let short = Image::new(2, 1, vec![0; 8]).unwrap();
        assert_eq!(apply(&src, &short, Channel::Rg, None), Err(Refusal::ShapeMismatch("uv field")));
        assert_eq!(apply(&src, &field([0.0; 4]), Channel::Rg, Some((0.0, 4.0))),
            Err(Refusal::ValueOutOfRange("catmull size")));
    }

    #[test]
    fn field_tap_is_bilinear_with_clamp_to_edge() {
        // Four texels, red ramp 0/64/128/255; locks the private field sampler
        // to bilinear + clamp-to-edge.
        let field = Image::new(4, 1, vec![
            0, 0, 0, 255,
            64, 0, 0, 255,
            128, 0, 0, 255,
            255, 0, 0, 255,
        ]).unwrap();
        let left = sample_bilinear(&field, 0.125, 0.5);
        let half = sample_bilinear(&field, 0.25, 0.5);
        let right = sample_bilinear(&field, 0.375, 0.5);
        assert!((left[0] - 0.0).abs() < 1e-7);
        assert!((half[0] - 32.0 / 255.0).abs() < 1e-7);
        assert!((right[0] - 64.0 / 255.0).abs() < 1e-7);

        let clamped = sample_bilinear(&field, -10.0, 10.0);
        assert!((clamped[0] - 0.0).abs() < 1e-7);
    }
}
