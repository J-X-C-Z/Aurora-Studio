//! PF90002 face / PF90010 body moisturizing pixel law.
//!
//! Recovered from pixcake.exe SHA256
//! `A84519DA489BC52E4E914AF7880EDE43E3CE58992657723D02197161832464C7`.
//!
//! Both effects share the process shader:
//! `YUV -> curve=beauty(Y) -> Y'=mix(Y, mix(Y, curve, dog), skin*degree*mask?)`
//! `-> inverse YUV`, alpha preserved. PF90002 adds a per-face masked weight
//! pass (`skin map(B,R)`, hardlight from blurred Y); PF90010 uses explicit body
//! skin/dog weight planes and a captured/calibrated degree.
//!
//! Face RVAs: degree shape `0x61FB670`, weight filter `0x61FBA70` (type D3),
//! process filter `0x61FC8C0` (type D4). Body RVAs: gate `0x4DF6CA2/0x4DF6CB2`,
//! degree store `0x4DF6D92`, weight shader `0x96B4985`, process `0x968ECE5`.

use crate::{Access, EffectMeta, Image, Refusal};

pub const FACE_PF: u32 = 90002;
pub const BODY_PF: u32 = 90010;
pub const BODY_RENDER_GATE: f32 = 0.01;

pub struct Lut1D<'a> {
    pub samples: &'a [f32],
}

impl Lut1D<'_> {
    /// GL LINEAR + CLAMP_TO_EDGE: `x = t*N - 0.5`.
    pub fn sample(&self, t: f32) -> Result<f32, Refusal> {
        if self.samples.is_empty() {
            return Err(Refusal::ShapeMismatch("lut1d"));
        }
        if !t.is_finite() {
            return Err(Refusal::ValueOutOfRange("lut1d"));
        }
        if self.samples.len() == 1 {
            return Ok(self.samples[0]);
        }
        if self.samples.iter().any(|v| !v.is_finite()) {
            return Err(Refusal::ValueOutOfRange("lut1d"));
        }
        let x = (t.clamp(0.0, 1.0) * self.samples.len() as f32 - 0.5).max(0.0);
        let floor = x.floor();
        let fraction = x - floor;
        let first = (floor as usize).min(self.samples.len() - 1);
        let second = (first + 1).min(self.samples.len() - 1);
        Ok(mix(self.samples[first], self.samples[second], fraction))
    }
}

pub struct SkinMap<'a> {
    pub width: usize,
    pub height: usize,
    pub samples: &'a [f32],
}

impl SkinMap<'_> {
    /// GL bilinear with `x=blue`, `y=red`.
    pub fn sample(&self, blue: f32, red: f32) -> Result<f32, Refusal> {
        if self.width == 0 || self.height == 0 || self.samples.len() != self.width * self.height {
            return Err(Refusal::ShapeMismatch("skin_map"));
        }
        if self.samples.iter().any(|v| !v.is_finite()) {
            return Err(Refusal::ValueOutOfRange("skin_map"));
        }
        let x = (blue.clamp(0.0, 1.0) * self.width as f32 - 0.5).max(0.0);
        let y = (red.clamp(0.0, 1.0) * self.height as f32 - 0.5).max(0.0);
        let fx = x.floor();
        let fy = y.floor();
        let ax = x - fx;
        let ay = y - fy;
        let get = |x: usize, y: usize| self.samples[y * self.width + x];
        let x0 = (fx as usize).min(self.width - 1);
        let x1 = (x0 + 1).min(self.width - 1);
        let y0 = (fy as usize).min(self.height - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let top = mix(get(x0, y0), get(x1, y0), ax);
        let bottom = mix(get(x0, y1), get(x1, y1), ax);
        Ok(mix(top, bottom, ay))
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
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
    RGB,
    RGA,
    RBA,
    GBA,
    RGBA,
}

pub fn select_mask(mask: [f32; 4], channel: MaskChannel) -> f32 {
    let [r, g, b, a] = mask;
    let sum = |values: &[f32]| values.iter().sum::<f32>().clamp(0.0, 1.0);
    match channel {
        MaskChannel::R => r,
        MaskChannel::G => g,
        MaskChannel::B => b,
        MaskChannel::A => a,
        MaskChannel::RG => sum(&[r, g]),
        MaskChannel::RB => sum(&[r, b]),
        MaskChannel::RA => sum(&[r, a]),
        MaskChannel::GB => sum(&[g, b]),
        MaskChannel::GA => sum(&[g, a]),
        MaskChannel::BA => sum(&[b, a]),
        MaskChannel::RGB => sum(&[r, g, b]),
        MaskChannel::RGA => sum(&[r, g, a]),
        MaskChannel::RBA => sum(&[r, b, a]),
        MaskChannel::GBA => sum(&[g, b, a]),
        MaskChannel::RGBA => sum(&[r, g, b, a]),
    }
}

pub struct FaceParams {
    pub ui: f32,
    pub pivot: f32,
    pub gain: f32,
}

fn mix(from: f32, to: f32, amount: f32) -> f32 {
    from * (1.0 - amount) + to * amount
}

fn forward(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    (
        0.299 * r + 0.587 * g + 0.114 * b,
        -0.169 * r - 0.331 * g + 0.500 * b,
        0.500 * r - 0.419 * g - 0.081 * b,
    )
}

fn inverse(y: f32, u: f32, v: f32) -> [f32; 3] {
    [
        y - 0.00093 * u + 1.401687 * v,
        y - 0.3437 * u - 0.71417 * v,
        y + 1.77216 * u + 0.00099 * v,
    ]
}

fn byte(value: f32) -> u8 {
    (value * 255.0).clamp(0.0, 255.0).round_ties_even() as u8
}

fn pixels_of(image: &Image) -> Result<usize, Refusal> {
    let pixels = (image.width as usize)
        .checked_mul(image.height as usize)
        .ok_or(Refusal::ShapeMismatch("image"))?;
    if image.pixels.len() != pixels * 4 {
        return Err(Refusal::ShapeMismatch("image"));
    }
    Ok(pixels)
}

/// PF90002 face: `degree=(ui/100-pivot)*gain+pivot`, per-face mask + LUT weight
/// pass, then the shared process shader. `blur_y` is the bilateral blur Y plane.
pub fn face_apply(
    source: &Image,
    mask_rgba: &[u8],
    blur_y: &[u8],
    params: &FaceParams,
    channel: MaskChannel,
    skin_map: &SkinMap<'_>,
    hardlight: &Lut1D<'_>,
    beauty: &Lut1D<'_>,
) -> Result<Image, Refusal> {
    let pixels = pixels_of(source)?;
    if mask_rgba.len() != pixels * 4 {
        return Err(Refusal::ShapeMismatch("mask_rgba"));
    }
    if blur_y.len() != pixels {
        return Err(Refusal::ShapeMismatch("blur_y"));
    }
    if !params.ui.is_finite() || !(0.0..=100.0).contains(&params.ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if !params.pivot.is_finite() || !params.gain.is_finite() {
        return Err(Refusal::ValueOutOfRange("pivot/gain"));
    }
    let degree = (params.ui / 100.0 - params.pivot) * params.gain + params.pivot;
    let mut output = source.clone();
    if params.ui <= 0.0 {
        return Ok(output);
    }

    for p in 0..pixels {
        let i = p * 4;
        let r = source.pixels[i] as f32 / 255.0;
        let g = source.pixels[i + 1] as f32 / 255.0;
        let b = source.pixels[i + 2] as f32 / 255.0;
        let blur = blur_y[p] as f32 / 255.0;

        let source_y = 0.299 * r + 0.587 * g + 0.114 * b;
        let skin = skin_map.sample(b, r)?;
        let mut dog = 0.0;
        if skin > 0.0 {
            let q1 = 1.0 - blur;
            let reference = (source_y + 2.0 * q1 - 1.0).clamp(0.0, 1.0);
            let reference = (reference + source_y) * 0.5;
            dog = 1.0 - hardlight.sample(reference)?;
        }

        let mask = [
            mask_rgba[i] as f32 / 255.0,
            mask_rgba[i + 1] as f32 / 255.0,
            mask_rgba[i + 2] as f32 / 255.0,
            mask_rgba[i + 3] as f32 / 255.0,
        ];
        let alpha = skin * degree * select_mask(mask, channel);
        if !(alpha > 0.0) {
            continue;
        }

        let (y, u, v) = forward(r, g, b);
        let curve = beauty.sample(y)?;
        let result_y = mix(y, mix(y, curve, dog), alpha);
        let out = inverse(result_y, u, v);
        output.pixels[i] = byte(out[0]);
        output.pixels[i + 1] = byte(out[1]);
        output.pixels[i + 2] = byte(out[2]);
    }
    Ok(output)
}

/// PF90010 body: explicit skin/dog weight planes and native degree.
pub fn body_apply(
    source: &Image,
    weight_skin: &[f32],
    weight_dog: &[f32],
    beauty: &Lut1D<'_>,
    degree: f32,
    roi: (u32, u32, u32, u32),
) -> Result<Image, Refusal> {
    let pixels = pixels_of(source)?;
    if weight_skin.len() != pixels || weight_dog.len() != pixels {
        return Err(Refusal::ShapeMismatch("weights"));
    }
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("degree"));
    }
    let (x0, y0, w, h) = roi;
    if w == 0 || h == 0 || x0 + w > source.width || y0 + h > source.height {
        return Err(Refusal::ValueOutOfRange("roi"));
    }
    let mut output = source.clone();
    if degree < BODY_RENDER_GATE {
        return Ok(output);
    }
    for y in y0..y0 + h {
        for x in x0..x0 + w {
            let p = (y * source.width + x) as usize;
            let alpha = weight_skin[p] * degree;
            if !(alpha > 0.0) {
                continue;
            }
            let i = p * 4;
            let r = source.pixels[i] as f32 / 255.0;
            let g = source.pixels[i + 1] as f32 / 255.0;
            let b = source.pixels[i + 2] as f32 / 255.0;
            let (yv, u, v) = forward(r, g, b);
            let curve = beauty.sample(yv)?;
            let inner = mix(yv, curve, weight_dog[p]);
            let result = mix(yv, inner, alpha);
            let out = inverse(result, u, v);
            output.pixels[i] = byte(out[0]);
            output.pixels[i + 1] = byte(out[1]);
            output.pixels[i + 2] = byte(out[2]);
        }
    }
    Ok(output)
}
/// Face and body moisture stages. Both take caller-supplied LUTs, blur planes
/// and masks; the body degree must come from a capture (UI/100 is unproven).
pub const META: &[EffectMeta] = &[
    EffectMeta {
        pf: 90002,
        key: "moistureSmoothSkin",
        label: "Moisture smooth (face)",
        access: Access::Cpu,
        inputs: &["source", "mask rgba", "blur Y", "skin map", "pivot/gain", "beauty LUTs"],
    },
    EffectMeta {
        pf: 90010,
        key: "moistureSmoothSkin_body",
        label: "Moisture smooth (body)",
        access: Access::Cpu,
        inputs: &["source", "skin weight", "dog weight", "beauty LUT", "captured degree", "roi"],
    },
];
#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> Image {
        Image::new(2, 1, vec![40, 80, 120, 77, 200, 150, 100, 9]).unwrap()
    }

    #[test]
    fn zero_degree_and_zero_weights_are_identity() {
        let src = source();
        let lut = Lut1D { samples: &[0.0, 0.5, 1.0] };
        let zero = [0.0f32; 2];
        assert_eq!(body_apply(&src, &zero, &zero, &lut, 0.0, (0, 0, 2, 1)).unwrap(), src);
        assert_eq!(body_apply(&src, &zero, &zero, &lut, 1.0, (0, 0, 2, 1)).unwrap(), src);
    }

    #[test]
    fn bad_shapes_and_degrees_are_refused() {
        let src = source();
        let lut = Lut1D { samples: &[0.0, 1.0] };
        let good = [0.5f32; 2];
        assert_eq!(body_apply(&src, &good[..1], &good, &lut, 0.5, (0, 0, 2, 1)),
            Err(Refusal::ShapeMismatch("weights")));
        assert_eq!(body_apply(&src, &good, &good, &lut, 1.5, (0, 0, 2, 1)),
            Err(Refusal::ValueOutOfRange("degree")));
        assert_eq!(body_apply(&src, &good, &good, &lut, 0.5, (1, 0, 2, 1)).is_err(), true);
    }
}
