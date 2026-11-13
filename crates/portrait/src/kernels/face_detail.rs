//! Face/eye detail rows 71-90, recovered from the P23/P24 vendor parse lanes.
//!
//! Every entry point takes its same-job inputs explicitly. Model output, masks,
//! LUT atlases, landmarks and meshes are never synthesized here. A missing or
//! misshapen input is a typed refusal. Geometry that was not recovered (for
//! example the SmileV1/CloseMouthV1 tensors or the PF100005 post-model blend)
//! is refused instead of approximated under a vendor name.

use crate::{Access, EffectMeta, Image, Refusal};

const INV255: f32 = 1.0 / 255.0;

/// Registry rows for this lane. `mod.rs` is owned by the integration lane, so
/// this module is compiled through the candidate harness until it is registered.
pub const META: &[EffectMeta] = &[
    EffectMeta { pf: 8, key: "PF8_UI_KEY_NOT_RECOVERED", label: "Stereo shadow volume", access: Access::Cpu, inputs: &["material RGBA", "face-parsing RGBA mask", "ROI"] },
    EffectMeta { pf: 9, key: "PF9_UI_KEY_NOT_RECOVERED", label: "Stereo highlight volume", access: Access::Cpu, inputs: &["material RGBA", "blusher RGBA mask", "ROI"] },
    EffectMeta { pf: 100, key: "eyebrowDegree_left", label: "Eyebrow color makeup (left)", access: Access::Cpu, inputs: &["EyeBrowMaskType=5 mask (RB)", "UV", "MixColor RGBA", "Opacity percent", "degree01", "FilterType 11"] },
    EffectMeta { pf: 101, key: "silkwormDegree", label: "Silkworm degree", access: Access::Refused, inputs: &["live eye mask/LUT"] },
    EffectMeta { pf: 103, key: "eyelinerDegree_upper", label: "Upper eyeliner color makeup", access: Access::Cpu, inputs: &["EyeMaskType=4 mask (RG)", "UV", "MixColor RGBA", "Opacity percent", "degree01", "FilterType 11"] },
    EffectMeta { pf: 105, key: "eyelashDegree_upper", label: "Upper eyelash color makeup", access: Access::Cpu, inputs: &["EyeLashMaskType=4 L mask as RGBA (GL_LUMINANCE, RG=min(2L,1))", "UV", "MixColor RGBA", "opacity", "degree01", "FilterType 11"] },
    EffectMeta { pf: 106, key: "blusherDegree", label: "Blusher color makeup", access: Access::Cpu, inputs: &["BlusherMaskType=15 owner mask RGBA", "UV", "MixColor RGBA", "Opacity percent", "degree01", "BlendMode macro"] },
    EffectMeta { pf: 107, key: "lipstickDegree", label: "Lipstick image makeup", access: Access::Cpu, inputs: &["material RGBA", "main mask R", "extra mask by LipMaskType", "mask/material UV", "MixColor RGBA", "opacity/degree/shelter/region", "FilterType 40/41/42/54"] },
    EffectMeta { pf: 108, key: "teethWhitening", label: "Teeth whitening", access: Access::Cpu, inputs: &["teeth mask id2.G", "Whiten LUT", "Bright LUT"] },
    EffectMeta { pf: 111, key: "eyePupilDegree", label: "Eye pupil image makeup", access: Access::Cpu, inputs: &["material RGBA", "material UV", "MixColor RGBA", "opacity", "degree01", "FilterType 40 (EyePupilMaskType=0: no mask)"] },
    EffectMeta { pf: 116, key: "contourDegree", label: "Blusher image makeup", access: Access::Cpu, inputs: &["material RGBA", "BlusherMaskType=15 owner mask RGBA", "UV", "MixColor RGBA", "Opacity percent", "degree01", "BlendMode macro"] },
    EffectMeta { pf: 117, key: "eyebrowDegree_right", label: "Eyebrow color makeup (right)", access: Access::Cpu, inputs: &["EyeBrowMaskType=5 mask (RB)", "UV", "MixColor RGBA", "Opacity percent", "degree01", "FilterType 11"] },
    EffectMeta { pf: 118, key: "eyelashDegree_lower", label: "Lower eyelash color makeup", access: Access::Cpu, inputs: &["EyeLashMaskType=4 L mask as RGBA (GL_LUMINANCE, RG=min(2L,1))", "UV", "MixColor RGBA", "opacity", "degree01", "FilterType 11"] },
    EffectMeta { pf: 202, key: "hairLine", label: "Hair line", access: Access::Cpu, inputs: &["194 landmarks", "135 weights", "control slots"] },
    EffectMeta { pf: 232, key: "skullEnhance", label: "Skull top enhance", access: Access::Cpu, inputs: &["landmarks 192..206", "15 factors"] },
    EffectMeta { pf: 21015, key: "skinColorHue", label: "Skin color hue", access: Access::Cpu, inputs: &["skin mask", "TintGreen LUT", "TintMagenta LUT"] },
    EffectMeta { pf: 21105, key: "bodySkinToneUnify", label: "AI body skin unify", access: Access::Cpu, inputs: &["body/face masks", "effect passes"] },
    EffectMeta { pf: 30007, key: "eyelinerDegree_lower", label: "Lower eyeliner color makeup", access: Access::Cpu, inputs: &["EyeMaskType=4 mask (RG)", "UV", "MixColor RGBA", "Opacity percent", "degree01", "FilterType 11"] },
    EffectMeta { pf: 30025, key: "teethBlemishRemoval", label: "Tooth repair", access: Access::Cpu, inputs: &["teeth mask", "repair layer", "ROI"] },
    EffectMeta { pf: 90003, key: "skinColorUniform", label: "Body skin color uniform", access: Access::Cpu, inputs: &["native 0x15F93 value", "body mask", "ROI mask"] },
    EffectMeta { pf: 90021, key: "PF90021_UI_KEY_NOT_RECOVERED", label: "Lip makeup enhance", access: Access::Cpu, inputs: &["mouth mask", "LipsLUT", "LipDodgeLUT"] },
    EffectMeta { pf: 90032, key: "eyeMakeupDegree", label: "Eye image makeup", access: Access::Cpu, inputs: &["material RGBA texture", "EyeMaskType=10 mask (RGB)", "mask/material UV", "MixColor RGBA", "Opacity percent", "degree01", "FilterType 41/42/44"] },
    EffectMeta { pf: 90033, key: "suitMakeupDegree", label: "Suit makeup (ordered passes)", access: Access::Cpu, inputs: &["ordered pass list", "per-pass material RGBA", "per-pass mask + UV", "MixColor/opacity/degree/shelter/region", "FilterType"] },
    EffectMeta { pf: 90087, key: "PF90087_UI_KEY_NOT_RECOVERED", label: "Hair seam fill", access: Access::Cpu, inputs: &["part-line mask", "hair mask"] },
    EffectMeta { pf: 90130, key: "glassesGlareRemoval", label: "Glasses glare removal", access: Access::Cpu, inputs: &["network mask", "effect RGB", "lens mask"] },
    EffectMeta { pf: 100004, key: "redBloodRemoval", label: "Eye red blood removal", access: Access::Cpu, inputs: &["model RGB", "model alpha"] },
    EffectMeta { pf: 100005, key: "eyeWhitePurify", label: "Eye white purify", access: Access::Refused, inputs: &["eye mask", "8-channel model output", "vendor pixel law (open)"] },
];

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

fn check_plane(len: usize, pixels: usize, name: &'static str) -> Result<(), Refusal> {
    if len == pixels { Ok(()) } else { Err(Refusal::ShapeMismatch(name)) }
}

fn clamp01(value: f32) -> f32 {
    if value.is_nan() { 0.0 } else { value.clamp(0.0, 1.0) }
}

fn clamp01_64(value: f64) -> f64 {
    if value.is_nan() { 0.0 } else { value.clamp(0.0, 1.0) }
}

fn byte_trunc(value: f32) -> u8 {
    (value as i32).clamp(0, 255) as u8
}

fn round_ties_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction > 0.5 {
        floor + 1.0
    } else if fraction < 0.5 {
        floor
    } else if (floor as i64) & 1 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

fn byte_even(value: f64) -> u8 {
    round_ties_even(value.clamp(0.0, 1.0) * 255.0).clamp(0.0, 255.0) as u8
}

fn byte_away(value: f32) -> u8 {
    if !value.is_finite() {
        return 0;
    }
    let adjusted = if value >= 0.0 { value + 0.5 } else { value - 0.5 };
    (adjusted as i32).clamp(0, 255) as u8
}

fn sample_bilinear(image: &Image, x: f32, y: f32) -> [f32; 4] {
    let cx = x.clamp(0.0, (image.width - 1) as f32);
    let cy = y.clamp(0.0, (image.height - 1) as f32);
    let x0 = cx.floor() as u32;
    let y0 = cy.floor() as u32;
    let x1 = (x0 + 1).min(image.width - 1);
    let y1 = (y0 + 1).min(image.height - 1);
    let tx = cx - x0 as f32;
    let ty = cy - y0 as f32;
    let at = |px: u32, py: u32, c: usize| image.pixels[((py * image.width + px) * 4) as usize + c] as f32;
    let mut out = [0.0; 4];
    for (channel, value) in out.iter_mut().enumerate() {
        let top = at(x0, y0, channel) * (1.0 - tx) + at(x1, y0, channel) * tx;
        let bottom = at(x0, y1, channel) * (1.0 - tx) + at(x1, y1, channel) * tx;
        *value = top * (1.0 - ty) + bottom * ty;
    }
    out
}

fn sample_bilinear_byte(image: &Image, x: f32, y: f32) -> [u8; 3] {
    let value = sample_bilinear(image, x, y);
    [byte_trunc(value[0]), byte_trunc(value[1]), byte_trunc(value[2])]
}

fn sample_bilinear_byte_even(image: &Image, x: f32, y: f32) -> [u8; 3] {
    let value = sample_bilinear(image, x, y);
    [
        round_ties_even(value[0].clamp(0.0, 255.0) as f64) as u8,
        round_ties_even(value[1].clamp(0.0, 255.0) as f64) as u8,
        round_ties_even(value[2].clamp(0.0, 255.0) as f64) as u8,
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

impl Rect {
    pub fn inside(self, width: u32, height: u32) -> bool {
        self.width > 0
            && self.height > 0
            && self.x >= 0
            && self.y >= 0
            && (self.x as i64 + self.width as i64) <= width as i64
            && (self.y as i64 + self.height as i64) <= height as i64
    }
}

// ---------------------------------------------------------------------------
// Row 71 / PF21105 body skin unify: exact 0x46E4D80 blend.
// ---------------------------------------------------------------------------

pub struct SkinUnifyPass<'a> {
    pub mask: &'a [u8],
    pub effect: &'a Image,
}

#[derive(Debug)]
pub struct SkinUnifyResult {
    pub degree01: f32,
    pub changed: u64,
    pub output: Image,
}

pub fn body_skin_unify(
    source: &Image,
    ui: f32,
    passes: &[SkinUnifyPass<'_>],
) -> Result<SkinUnifyResult, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    if !ui.is_finite() || ui < 0.0 || ui > 100.0 {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    let degree01 = ui / 100.0;
    if degree01 == 0.0 {
        return Ok(SkinUnifyResult { degree01, changed: 0, output: source.clone() });
    }
    if passes.is_empty() {
        return Err(Refusal::MissingInput("post_model_compose_passes"));
    }

    let mut current = source.clone();
    for (index, pass) in passes.iter().enumerate() {
        if pass.mask.len() != pixels {
            return Err(Refusal::ShapeMismatch("pass.mask"));
        }
        let effect_pixels = checked_pixels(pass.effect, "pass.effect")?;
        if effect_pixels != pixels {
            return Err(Refusal::ShapeMismatch("pass.effect"));
        }
        let mut next = current.clone();
        for pixel in 0..pixels {
            let base = pixel * 4;
            let t = clamp01(pass.mask[pixel] as f32 * INV255 * degree01);
            for channel in 0..3 {
                let src = current.pixels[base + channel] as f32;
                let effect = pass.effect.pixels[base + channel] as f32;
                next.pixels[base + channel] = if t <= 0.0 {
                    current.pixels[base + channel]
                } else {
                    byte_trunc(src * (1.0 - t) + effect * t)
                };
            }
            next.pixels[base + 3] = current.pixels[base + 3];
        }
        let _ = index;
        current = next;
    }

    let changed = current
        .pixels
        .chunks_exact(4)
        .zip(source.pixels.chunks_exact(4))
        .filter(|(a, b)| a[0] != b[0] || a[1] != b[1] || a[2] != b[2])
        .count() as u64;
    Ok(SkinUnifyResult { degree01, changed, output: current })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SkinUnifyRoute {
    Degree60,
    Degree58TimesScale,
}

pub fn skin_unify_route_degree(
    route: SkinUnifyRoute,
    degree_at_60: &[f32],
    degree_at_58: &[f32],
    scale_at_2ac: f32,
    face_index: usize,
) -> Result<f32, Refusal> {
    match route {
        SkinUnifyRoute::Degree60 => degree_at_60
            .get(face_index)
            .copied()
            .filter(|value| value.is_finite())
            .ok_or(Refusal::MissingInput("degree_at_0x60")),
        SkinUnifyRoute::Degree58TimesScale => {
            if !scale_at_2ac.is_finite() {
                return Err(Refusal::ValueOutOfRange("scale_at_0x2AC"));
            }
            degree_at_58
                .get(face_index)
                .copied()
                .filter(|value| value.is_finite())
                .map(|value| value * scale_at_2ac)
                .ok_or(Refusal::MissingInput("degree_at_0x58"))
        }
    }
}

// ---------------------------------------------------------------------------
// Row 72 / PF90003 body skin color uniform.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodySkinUniformPrepared {
    pub native: f32,
    pub blend_degree: f32,
    pub roi: Rect,
    pub target: [f32; 3],
    pub samples: u32,
}

pub fn body_skin_uniform(
    source_rgb: &[u8],
    width: u32,
    height: u32,
    body_mask: &[u8],
    roi_mask: &[u8],
    ui: f32,
    face_id: usize,
    native_map: &[f32],
) -> Result<(BodySkinUniformPrepared, Vec<u8>), Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("source_rgb"));
    }
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("source_rgb"))?;
    if source_rgb.len() != pixels * 3 {
        return Err(Refusal::ShapeMismatch("source_rgb"));
    }
    check_plane(body_mask.len(), pixels, "body_mask")?;
    check_plane(roi_mask.len(), pixels, "roi_mask")?;
    if !ui.is_finite() || ui < 0.0 || ui > 100.0 {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    let zero = BodySkinUniformPrepared {
        native: 0.0,
        blend_degree: 0.0,
        roi: Rect { x: 0, y: 0, width: 0, height: 0 },
        target: [0.0; 3],
        samples: 0,
    };
    if ui == 0.0 {
        return Ok((zero, source_rgb.to_vec()));
    }

    let native = native_map
        .get(face_id)
        .copied()
        .filter(|value| value.is_finite())
        .ok_or(Refusal::MissingInput("native_skin_color_uniform"))?;

    let mut min_x = width as i32;
    let mut min_y = height as i32;
    let mut max_x = -1i32;
    let mut max_y = -1i32;
    for y in 0..height as usize {
        for x in 0..width as usize {
            let p = y * width as usize + x;
            if body_mask[p] == 0 || roi_mask[p] == 0 {
                continue;
            }
            min_x = min_x.min(x as i32);
            min_y = min_y.min(y as i32);
            max_x = max_x.max(x as i32);
            max_y = max_y.max(y as i32);
        }
    }
    if max_x < min_x || max_y < min_y {
        return Err(Refusal::MissingInput("body_roi"));
    }
    let roi = Rect { x: min_x, y: min_y, width: max_x - min_x + 1, height: max_y - min_y + 1 };
    if roi.width <= 10 || roi.height <= 10 {
        return Err(Refusal::ValueOutOfRange("body_roi_min_side"));
    }

    let mut sums = [0.0f64; 3];
    let mut samples = 0u32;
    for y in roi.y..roi.y + roi.height {
        for x in roi.x..roi.x + roi.width {
            let p = y as usize * width as usize + x as usize;
            if body_mask[p] < 0xBA || roi_mask[p] == 0 {
                continue;
            }
            let offset = p * 3;
            for channel in 0..3 {
                sums[channel] += source_rgb[offset + channel] as f64;
            }
            samples += 1;
        }
    }
    if samples == 0 {
        return Err(Refusal::MissingInput("body_target_samples"));
    }
    let target = [
        (sums[0] / samples as f64) as f32,
        (sums[1] / samples as f64) as f32,
        (sums[2] / samples as f64) as f32,
    ];
    let blend_degree = native.clamp(0.0, 1.0);
    let prepared = BodySkinUniformPrepared { native, blend_degree, roi, target, samples };

    // shortcut: explicit target transfer, replace when the TradBodySkinTone curve is recovered.
    let mut output = source_rgb.to_vec();
    for y in roi.y..roi.y + roi.height {
        for x in roi.x..roi.x + roi.width {
            let p = y as usize * width as usize + x as usize;
            if body_mask[p] == 0 || roi_mask[p] == 0 {
                continue;
            }
            let alpha = clamp01(body_mask[p] as f32 * INV255 * blend_degree);
            let offset = p * 3;
            for channel in 0..3 {
                let src = source_rgb[offset + channel] as f32;
                output[offset + channel] = byte_trunc(src + (target[channel] - src) * alpha);
            }
        }
    }
    Ok((prepared, output))
}

// ---------------------------------------------------------------------------
// SkinAdjustV2 LUT sampling, shared by rows 73, 74, 79 and 86.
// ---------------------------------------------------------------------------

fn check_lut_rgb(lut: &[u8], name: &'static str) -> Result<(), Refusal> {
    if lut.len() == 512 * 512 * 3 { Ok(()) } else { Err(Refusal::ShapeMismatch(name)) }
}

fn lut_read(lut: &[u8], x: usize, y: usize) -> [f64; 3] {
    let offset = (y * 512 + x) * 3;
    [
        lut[offset] as f64 / 255.0,
        lut[offset + 1] as f64 / 255.0,
        lut[offset + 2] as f64 / 255.0,
    ]
}

fn lut_sample_2d(lut: &[u8], u: f64, v: f64) -> [f64; 3] {
    let x = u * 512.0 - 0.5;
    let y = v * 512.0 - 0.5;
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let fx = x - xi as f64;
    let fy = y - yi as f64;
    let x0 = xi.clamp(0, 511) as usize;
    let x1 = (xi + 1).clamp(0, 511) as usize;
    let y0 = yi.clamp(0, 511) as usize;
    let y1 = (yi + 1).clamp(0, 511) as usize;
    let c00 = lut_read(lut, x0, y0);
    let c10 = lut_read(lut, x1, y0);
    let c01 = lut_read(lut, x0, y1);
    let c11 = lut_read(lut, x1, y1);
    let mut out = [0.0; 3];
    for channel in 0..3 {
        let top = c00[channel] * (1.0 - fx) + c10[channel] * fx;
        let bottom = c01[channel] * (1.0 - fx) + c11[channel] * fx;
        out[channel] = top * (1.0 - fy) + bottom * fy;
    }
    out
}

fn lut3d_sample(lut: &[u8], r: f32, g: f32, b: f32) -> [f64; 3] {
    let r = clamp01_64(r as f64);
    let g = clamp01_64(g as f64);
    let b = clamp01_64(b as f64);
    let f = b * 63.0;
    let fz0 = f.floor();
    let fz1 = f.ceil();
    let t0 = (fz0 / 8.0).floor();
    let s0 = fz0 - t0 * 8.0;
    let t1 = (fz1 / 8.0).floor();
    let s1 = fz1 - t1 * 8.0;
    let u0 = s0 * 0.125 + 0.0009765625 + 0.123046875 * r;
    let v0 = t0 * 0.125 + 0.0009765625 + 0.123046875 * g;
    let u1 = s1 * 0.125 + 0.0009765625 + 0.123046875 * r;
    let v1 = t1 * 0.125 + 0.0009765625 + 0.123046875 * g;
    let p0 = lut_sample_2d(lut, u0, v0);
    let p1 = lut_sample_2d(lut, u1, v1);
    let fraction = f - fz0;
    [
        p0[0] * (1.0 - fraction) + p1[0] * fraction,
        p0[1] * (1.0 - fraction) + p1[1] * fraction,
        p0[2] * (1.0 - fraction) + p1[2] * fraction,
    ]
}

// ---------------------------------------------------------------------------
// Rows 73 and 74: temperature (dA/cold|warm) and tint (dw/green|magenta).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdjustDirection {
    None,
    Cold,
    Warm,
    Green,
    Magenta,
}

fn skin_adjust(
    source: &Image,
    mask: &[u8],
    negative_lut: &[u8],
    positive_lut: &[u8],
    ui: f32,
    negative: AdjustDirection,
    positive: AdjustDirection,
) -> Result<Image, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    check_plane(mask.len(), pixels, "skin_mask")?;
    check_lut_rgb(negative_lut, "negative_lut")?;
    check_lut_rgb(positive_lut, "positive_lut")?;
    if !ui.is_finite() || ui < -100.0 || ui > 100.0 {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    let engine = ((ui as f64).clamp(-100.0, 100.0) + 100.0) / 200.0;
    let signed = 2.0 * engine - 1.0;
    let degree = signed.abs();
    if degree <= 0.001 {
        return Ok(source.clone());
    }
    let _direction = if signed < 0.0 { negative } else { positive };
    let atlas = if signed < 0.0 { negative_lut } else { positive_lut };

    let mut output = source.clone();
    for pixel in 0..pixels {
        let alpha = mask[pixel] as f64 / 255.0 * degree;
        if alpha <= 0.001 {
            continue;
        }
        let offset = pixel * 4;
        let lut = lut3d_sample(
            atlas,
            source.pixels[offset] as f32 * INV255,
            source.pixels[offset + 1] as f32 * INV255,
            source.pixels[offset + 2] as f32 * INV255,
        );
        for channel in 0..3 {
            let src = source.pixels[offset + channel] as f64 / 255.0;
            output.pixels[offset + channel] = byte_even(src * (1.0 - alpha) + lut[channel] * alpha);
        }
        output.pixels[offset + 3] = source.pixels[offset + 3];
    }
    Ok(output)
}

pub fn skin_temperature(
    source: &Image,
    mask: &[u8],
    cold_lut: &[u8],
    warm_lut: &[u8],
    ui: f32,
) -> Result<Image, Refusal> {
    skin_adjust(source, mask, cold_lut, warm_lut, ui, AdjustDirection::Cold, AdjustDirection::Warm)
}

pub fn skin_tint(
    source: &Image,
    mask: &[u8],
    green_lut: &[u8],
    magenta_lut: &[u8],
    ui: f32,
) -> Result<Image, Refusal> {
    skin_adjust(source, mask, green_lut, magenta_lut, ui, AdjustDirection::Green, AdjustDirection::Magenta)
}

// ---------------------------------------------------------------------------
// Row 75 / PF100007 open-mouth smile.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct SmileOpenResult {
    pub active: bool,
    pub degree: f32,
    pub roi: Rect,
    pub touched: u32,
    pub changed: u32,
    pub output: Image,
}

pub fn smile_open(
    source: &Image,
    material: &str,
    ui: f32,
    mouth_landmarks: &[[f32; 2]],
    face: Rect,
    roi_expand: f32,
    lip_mask: &[u8],
    mouth_mask: &[u8],
    model_dx: &[f32],
    model_dy: &[f32],
) -> Result<SmileOpenResult, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    if !ui.is_finite() || ui < 0.0 || ui > 100.0 {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    let degree = ui / 100.0;
    let identity = SmileOpenResult {
        active: false,
        degree,
        roi: Rect { x: 0, y: 0, width: 0, height: 0 },
        touched: 0,
        changed: 0,
        output: source.clone(),
    };
    if material != "Grinning" || degree <= 0.001 {
        return Ok(identity);
    }
    if mouth_landmarks.is_empty() {
        return Err(Refusal::MissingInput("mouth_landmarks"));
    }
    if !roi_expand.is_finite() || roi_expand < 0.0 {
        return Err(Refusal::ValueOutOfRange("roi_expand"));
    }
    check_plane(lip_mask.len(), pixels, "lip_mask")?;
    check_plane(mouth_mask.len(), pixels, "mouth_mask")?;
    check_plane(model_dx.len(), pixels, "model_dx")?;
    check_plane(model_dy.len(), pixels, "model_dy")?;

    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for point in mouth_landmarks {
        if !point[0].is_finite() || !point[1].is_finite() {
            return Err(Refusal::ValueOutOfRange("mouth_landmarks"));
        }
        min_x = min_x.min(point[0]);
        min_y = min_y.min(point[1]);
        max_x = max_x.max(point[0]);
        max_y = max_y.max(point[1]);
    }
    let bbox = Rect {
        x: min_x.floor() as i32,
        y: min_y.floor() as i32,
        width: max_x.ceil() as i32 - min_x.floor() as i32 + 1,
        height: max_y.ceil() as i32 - min_y.floor() as i32 + 1,
    };
    let expanded_w = ((bbox.width as f32 * (1.0 + roi_expand)) as f64).max(1.0);
    let expanded_h = ((bbox.height as f32 * (1.0 + roi_expand)) as f64).max(1.0);
    let expanded_w = round_ties_even(expanded_w) as i32;
    let expanded_h = round_ties_even(expanded_h) as i32;
    let center_x = bbox.x + bbox.width / 2;
    let center_y = bbox.y + bbox.height / 2;
    let expanded = Rect {
        x: center_x - expanded_w / 2,
        y: center_y - expanded_h / 2,
        width: expanded_w,
        height: expanded_h,
    };
    let face_x0 = face.x.max(0);
    let face_y0 = face.y.max(0);
    let face_x1 = (face.x + face.width).min(source.width as i32);
    let face_y1 = (face.y + face.height).min(source.height as i32);
    if face_x1 <= face_x0 || face_y1 <= face_y0 {
        return Ok(SmileOpenResult { roi: expanded, ..identity });
    }
    let x0 = expanded.x.max(face_x0);
    let y0 = expanded.y.max(face_y0);
    let x1 = (expanded.x + expanded.width).min(face_x1);
    let y1 = (expanded.y + expanded.height).min(face_y1);
    if x1 <= x0 || y1 <= y0 {
        return Ok(SmileOpenResult { roi: expanded, ..identity });
    }
    let roi = Rect { x: x0, y: y0, width: x1 - x0, height: y1 - y0 };
    let mut output = source.clone();
    let mut touched = 0u32;
    let mut changed = 0u32;
    for y in roi.y..roi.y + roi.height {
        for x in roi.x..roi.x + roi.width {
            let pixel = y as usize * source.width as usize + x as usize;
            let support = lip_mask[pixel] as f32 * INV255 * (mouth_mask[pixel] as f32 * INV255);
            if support <= 0.001 {
                continue;
            }
            let weight = degree * support;
            if weight <= 0.0 {
                continue;
            }
            touched += 1;
            let sx = x as f32 + weight * model_dx[pixel];
            let sy = y as f32 + weight * model_dy[pixel];
            let sampled = sample_bilinear_byte_even(source, sx, sy);
            let base = pixel * 4;
            for channel in 0..3 {
                if sampled[channel] != source.pixels[base + channel] {
                    changed += 1;
                    break;
                }
            }
            output.pixels[base] = sampled[0];
            output.pixels[base + 1] = sampled[1];
            output.pixels[base + 2] = sampled[2];
        }
    }
    Ok(SmileOpenResult { active: true, degree, roi, touched, changed, output })
}

// ---------------------------------------------------------------------------
// Row 76 / PF21076 closed-mouth landmark alternative.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct ClosedSmileResult {
    pub degree: f32,
    pub roi: Rect,
    pub changed: u32,
    pub alternative: bool,
    pub output: Image,
}

fn point_distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

pub fn smile_closed_landmark_alternative(
    source: &Image,
    landmarks: &[[f32; 2]],
    ui: f32,
) -> Result<ClosedSmileResult, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    if !ui.is_finite() || ui < 0.0 || ui > 100.0 {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if landmarks.len() != 68 {
        return Err(Refusal::MissingInput("landmarks68"));
    }
    for point in landmarks {
        if !point[0].is_finite() || !point[1].is_finite() {
            return Err(Refusal::ValueOutOfRange("landmarks68"));
        }
    }
    let left = landmarks[48];
    let right = landmarks[54];
    let top = landmarks[51];
    let bottom = landmarks[57];
    let mouth_width = point_distance(left, right).max(1.0);
    if mouth_width < 1.0 {
        return Err(Refusal::ValueOutOfRange("mouth_width"));
    }
    let degree = ui / 100.0;
    // shortcut: landmark alternative, replace when CloseMouthV1 tensors are available.
    let mouth_height = (bottom[1] - top[1]).abs().max(1.0);
    let min_x = left[0].min(right[0]) - 0.38 * mouth_width;
    let max_x = left[0].max(right[0]) + 0.38 * mouth_width;
    let min_y = top[1].min(bottom[1]) - 1.25 * mouth_height;
    let max_y = top[1].max(bottom[1]) + 1.25 * mouth_height;
    let x0 = (min_x.floor() as i32).clamp(0, source.width as i32 - 1);
    let y0 = (min_y.floor() as i32).clamp(0, source.height as i32 - 1);
    let x1 = (max_x.ceil() as i32).clamp(0, source.width as i32);
    let y1 = (max_y.ceil() as i32).clamp(0, source.height as i32);
    let roi = Rect { x: x0, y: y0, width: (x1 - x0).max(1), height: (y1 - y0).max(1) };
    let mut output = source.clone();
    if degree <= 0.001 {
        return Ok(ClosedSmileResult { degree, roi, changed: 0, alternative: true, output });
    }

    let mut target = landmarks.to_vec();
    target[48] = [left[0] - 0.16 * mouth_width * degree, left[1] - 0.08 * mouth_width * degree];
    target[54] = [right[0] + 0.16 * mouth_width * degree, right[1] - 0.08 * mouth_width * degree];
    target[51] = [top[0], top[1] - 0.04 * mouth_width * degree];
    target[57] = [bottom[0], bottom[1] + 0.015 * mouth_width * degree];
    let source_controls = [left, top, right, bottom];
    let target_controls = [target[48], target[51], target[54], target[57]];
    let center = [
        (left[0] + right[0] + top[0] + bottom[0]) * 0.25,
        (left[1] + right[1] + top[1] + bottom[1]) * 0.25,
    ];
    let sigma = mouth_width.max(mouth_height) * 0.65;
    let sigma2 = (sigma * sigma).max(1.0e-6);
    let mut changed = 0u32;
    for y in roi.y..roi.y + roi.height {
        for x in roi.x..roi.x + roi.width {
            let px = x as f32;
            let py = y as f32;
            let nx = (px - center[0]) / (0.62 * mouth_width);
            let ny = (py - center[1]) / (1.35 * mouth_height);
            let radius = (nx * nx + ny * ny).sqrt();
            let mask = if radius >= 1.15 {
                0.0
            } else if radius <= 0.80 {
                1.0
            } else {
                clamp01((1.15 - radius) / 0.35)
            };
            if mask <= 0.0 {
                continue;
            }
            let mut total = 0.0f32;
            let mut dx = 0.0f32;
            let mut dy = 0.0f32;
            for control in 0..4 {
                let ddx = px - source_controls[control][0];
                let ddy = py - source_controls[control][1];
                let weight = (-(ddx * ddx + ddy * ddy) / (2.0 * sigma2)).exp() + 1.0e-6;
                total += weight;
                dx += weight * (target_controls[control][0] - source_controls[control][0]);
                dy += weight * (target_controls[control][1] - source_controls[control][1]);
            }
            let sample_x = px - dx / total;
            let sample_y = py - dy / total;
            let sampled = sample_bilinear_byte(source, sample_x, sample_y);
            let alpha = clamp01(mask * degree);
            let pixel = y as usize * source.width as usize + x as usize;
            let base = pixel * 4;
            let mut local_changed = false;
            for channel in 0..3 {
                let value = ((1.0 - alpha) * source.pixels[base + channel] as f32
                    + alpha * sampled[channel] as f32) as i32;
                let value = value.clamp(0, 255) as u8;
                if value != source.pixels[base + channel] {
                    local_changed = true;
                }
                output.pixels[base + channel] = value;
            }
            if local_changed {
                changed += 1;
            }
        }
    }
    let _ = pixels;
    Ok(ClosedSmileResult { degree, roi, changed, alternative: true, output })
}

// ---------------------------------------------------------------------------
// Row 77 / PF8033 smile mouth corner.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine3x4 {
    pub m00: f32,
    pub m01: f32,
    pub m02: f32,
    pub m03: f32,
    pub m10: f32,
    pub m11: f32,
    pub m12: f32,
    pub m13: f32,
}

impl Affine3x4 {
    fn finite(self) -> bool {
        [self.m00, self.m01, self.m02, self.m03, self.m10, self.m11, self.m12, self.m13]
            .iter()
            .all(|value| value.is_finite())
    }

    fn apply(self, point: [f32; 3]) -> [f32; 2] {
        [
            self.m00 * point[0] + self.m01 * point[1] + self.m02 * point[2] + self.m03,
            self.m10 * point[0] + self.m11 * point[1] + self.m12 * point[2] + self.m13,
        ]
    }
}

pub fn mouth_corner_vertex_field(
    ui: f32,
    mean: &[[f32; 3]],
    reshape: &[[f32; 3]],
    projection: Affine3x4,
) -> Result<Vec<[f32; 2]>, Refusal> {
    if !ui.is_finite() || ui < 0.0 || ui > 100.0 {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if mean.len() != 2206 || reshape.len() != 2206 {
        return Err(Refusal::MissingInput("mesh2206"));
    }
    if !projection.finite() {
        return Err(Refusal::ValueOutOfRange("projection"));
    }
    let degree = ui / 100.0;
    let mut out = vec![[0.0f32; 2]; mean.len()];
    for i in 0..mean.len() {
        let m = mean[i];
        let r = reshape[i];
        if !m.iter().chain(r.iter()).all(|value| value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("mesh2206"));
        }
        let morphed = [
            m[0] + degree * (r[0] - m[0]),
            m[1] + degree * (r[1] - m[1]),
            m[2] + degree * (r[2] - m[2]),
        ];
        let a = projection.apply(m);
        let b = projection.apply(morphed);
        out[i] = [b[0] - a[0], b[1] - a[1]];
    }
    Ok(out)
}

pub fn mouth_corner_warp(
    source: &Image,
    dx: &[f32],
    dy: &[f32],
    mask: &[u8],
    ui: f32,
) -> Result<Image, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    check_plane(dx.len(), pixels, "dx")?;
    check_plane(dy.len(), pixels, "dy")?;
    check_plane(mask.len(), pixels, "mask")?;
    if !ui.is_finite() || ui < 0.0 || ui > 100.0 {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if ui == 0.0 {
        return Ok(source.clone());
    }
    let degree = ui / 100.0;
    let mut output = source.clone();
    for y in 0..source.height {
        for x in 0..source.width {
            let pixel = y as usize * source.width as usize + x as usize;
            let weight = mask[pixel] as f32 * INV255 * degree;
            let sx = (x as f32 + weight * dx[pixel]).clamp(0.0, (source.width - 1) as f32);
            let sy = (y as f32 + weight * dy[pixel]).clamp(0.0, (source.height - 1) as f32);
            let sampled = sample_bilinear_byte(source, sx, sy);
            let base = pixel * 4;
            output.pixels[base] = sampled[0];
            output.pixels[base + 1] = sampled[1];
            output.pixels[base + 2] = sampled[2];
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Row 78 / PF30025 tooth repair.
// ---------------------------------------------------------------------------

pub fn tooth_repair(
    source: &Image,
    repair: &Image,
    teeth_mask: &[f32],
    roi: Rect,
    ui: f32,
) -> Result<Image, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    let repair_pixels = checked_pixels(repair, "repair")?;
    if repair_pixels != pixels {
        return Err(Refusal::ShapeMismatch("repair"));
    }
    if !roi.inside(source.width, source.height) {
        return Err(Refusal::ValueOutOfRange("roi"));
    }
    check_plane(teeth_mask.len(), pixels, "teeth_mask")?;
    if !ui.is_finite() || ui < 0.0 || ui > 100.0 {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    for y in roi.y..roi.y + roi.height {
        for x in roi.x..roi.x + roi.width {
            let value = teeth_mask[y as usize * source.width as usize + x as usize];
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(Refusal::ValueOutOfRange("teeth_mask"));
            }
        }
    }
    let degree = ui * 0.01;
    let mut output = source.clone();
    if degree <= 0.001 {
        return Ok(output);
    }
    for y in roi.y..roi.y + roi.height {
        for x in roi.x..roi.x + roi.width {
            let pixel = y as usize * source.width as usize + x as usize;
            let mask = teeth_mask[pixel];
            let weight = mask * degree;
            if weight <= 0.0 {
                continue;
            }
            let base = pixel * 4;
            if weight >= 1.0 {
                output.pixels[base] = repair.pixels[base];
                output.pixels[base + 1] = repair.pixels[base + 1];
                output.pixels[base + 2] = repair.pixels[base + 2];
            } else {
                for channel in 0..3 {
                    let src = source.pixels[base + channel] as f32;
                    let rep = repair.pixels[base + channel] as f32;
                    output.pixels[base + channel] = byte_away(src + (rep - src) * weight);
                }
            }
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Row 79 / PF108-110 teeth whitening.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct TeethWhiteningResult {
    pub output: Image,
    pub combined_mask: Vec<u8>,
    pub dz: f32,
    pub ea: f32,
    pub touched: u32,
}

const TEETH_STORE_THRESHOLDS: [f64; 256] = [
    0.5605493055190891, 0.556803893763572, 0.553058541379869, 0.5493133077397943,
    0.545567836612463, 0.5418230779469013, 0.5380773693323135, 0.5330834276974201,
    0.5293377190828323, 0.5255929604172707, 0.5218472518026829, 0.5181024931371212,
    0.5143567845225334, 0.5106110759079456, 0.5056180842220783, 0.5018733255565166,
    0.5605487674474716, 0.5568040087819099, 0.5530592501163483, 0.5493125915527344,
    0.5455678328871727, 0.5405748412013054, 0.5368281826376915, 0.5330834239721298,
    0.5293386653065681, 0.5255920067429543, 0.5218472480773926, 0.5181024894118309,
    0.514355830848217, 0.5093628391623497, 0.505618080496788, 0.5018733218312263,
    0.5605516135692596, 0.5568049550056458, 0.5530582964420319, 0.549311637878418,
    0.5443186461925507, 0.5405719876289368, 0.5368253290653229, 0.5330862700939178,
    0.529339611530304, 0.5255929529666901, 0.5218462944030762, 0.5168533027172089,
    0.513106644153595, 0.5093637853860855, 0.505620926618576, 0.5018742680549622,
    0.5605506598949432, 0.5568040013313293, 0.551811009645462, 0.5480643510818481,
    0.5443176925182343, 0.5405748337507248, 0.5368281751871109, 0.5330853164196014,
    0.5293386578559875, 0.5255919992923737, 0.5205990076065063, 0.5168523490428925,
    0.5131056904792786, 0.5093666315078735, 0.5056199729442596, 0.5018733143806458,
    0.5605497062206268, 0.5555567145347595, 0.5518100559711456, 0.5480633974075317,
    0.5443167388439178, 0.540570080280304, 0.5368234217166901, 0.533084362745285,
    0.5280913710594177, 0.5243447124958038, 0.5205980539321899, 0.516851395368576,
    0.5131047368049622, 0.5093580782413483, 0.5056190192699432, 0.5018723607063293,
    0.559302419424057, 0.5555557608604431, 0.5518091022968292, 0.5480624437332153,
    0.5443157851696014, 0.5405767261981964, 0.5368300676345825, 0.5318370759487152,
    0.5280904173851013, 0.5243437588214874, 0.5205971002578735, 0.5168504416942596,
    0.5131037831306458, 0.5093647241592407, 0.5056180655956268, 0.5006250739097595,
    0.5593014657497406, 0.5555548071861267, 0.5518081486225128, 0.5480614900588989,
    0.5443224310874939, 0.5393294394016266, 0.5355827808380127, 0.5318361222743988,
    0.5280894637107849, 0.524342805147171, 0.5205961465835571, 0.5168570876121521,
    0.5131104290485382, 0.5081174373626709, 0.504370778799057, 0.5006241202354431,
    0.5593005120754242, 0.5555538535118103, 0.5518147945404053, 0.5480681359767914,
    0.5430675446987152, 0.5393284857273102, 0.5355818271636963, 0.5318351686000824,
    0.5280885100364685, 0.5243418514728546, 0.5206027925014496, 0.5168561339378357,
    0.5118631422519684, 0.5081164836883545, 0.5043698251247406, 0.5630462169647217,
    0.5592919588088989, 0.5555528998374939, 0.5505675077438354, 0.5468132495880127,
    0.5430741906166077, 0.5393199324607849, 0.5355808734893799, 0.5318266153335571,
    0.5280875563621521, 0.5243332982063293, 0.5193479061126709, 0.5156088471412659,
    0.5118545889854431, 0.5081155300140381, 0.5043612718582153, 0.5630452632904053,
    0.5592910051345825, 0.5543056130409241, 0.550566554069519, 0.5468122959136963,
    0.5430732369422913, 0.5393189787864685, 0.5355799198150635, 0.5318256616592407,
    0.5280866026878357, 0.5231012105941772, 0.5193469524383545, 0.5156078934669495,
    0.5118536353111267, 0.5081145763397217, 0.5043603181838989, 0.5617979764938354,
    0.5580589175224304, 0.5543046593666077, 0.5505656003952026, 0.5468113422393799,
    0.5430722832679749, 0.5393180251121521, 0.5355789661407471, 0.5305935740470886,
    0.5268393158912659, 0.5231002569198608, 0.5193459987640381, 0.5156069397926331,
    0.5118526816368103, 0.5081136226654053, 0.5043593645095825, 0.561797022819519,
    0.558057963848114, 0.5543037056922913, 0.5505646467208862, 0.5468103885650635,
    0.5430713295936584, 0.5393170714378357, 0.5343316793441772, 0.5305926203727722,
    0.5268383622169495, 0.5230993032455444, 0.5193450450897217, 0.5156059861183167,
    0.5118517279624939, 0.5068663358688354, 0.5031272768974304, 0.5617960691452026,
    0.5580570101737976, 0.5543027520179749, 0.5505636930465698, 0.5468094348907471,
    0.5418240427970886, 0.5380697846412659, 0.5343307256698608, 0.5305916666984558,
    0.5268374085426331, 0.523098349571228, 0.5193440914154053, 0.5156050324440002,
    0.5106044411659241, 0.506865382194519, 0.503126323223114, 0.5617951154708862,
    0.5580560564994812, 0.5543017983436584, 0.54931640625, 0.5455621480941772,
    0.5418230891227722, 0.5380688309669495, 0.5343297719955444, 0.5305907130241394,
    0.5268364548683167, 0.5230973958969116, 0.5180968046188354, 0.5143577456474304,
    0.5106034874916077, 0.5068644285202026, 0.5031253695487976, 0.5617941617965698,
    0.5580551028251648, 0.5530545115470886, 0.5493154525756836, 0.5455611944198608,
    0.5418221354484558, 0.5380678772926331, 0.534328818321228, 0.530589759349823,
    0.5268355011940002, 0.5218501091003418, 0.518095850944519, 0.514356791973114,
    0.5106025338172913, 0.5068634748458862, 0.5031244158744812, 0.560546875,
    0.556807816028595, 0.5530535578727722, 0.5493144989013672, 0.5455602407455444,
    0.5418211817741394, 0.5380669236183167, 0.5343278646469116, 0.5293424725532532,
    0.5255882143974304, 0.5218491554260254, 0.5180948972702026, 0.5143558382987976,
    0.5106015801429749, 0.5068625211715698, 0.5031234622001648, 255.0,
];

fn store_unorm8(value: f32) -> u8 {
    let v = clamp01(value) as f64 * 255.0;
    let floor = v.floor();
    let index = floor.clamp(0.0, 255.0) as usize;
    let fraction = v - floor;
    let mut result = index as i32;
    if fraction >= TEETH_STORE_THRESHOLDS[index] {
        result += 1;
    }
    result.clamp(0, 255) as u8
}

fn teeth_lut_sample(
    table: &[u8],
    r: f32,
    g: f32,
    b: f32,
    degree: f32,
) -> [u8; 3] {
    let blue = b * 63.0;
    let fb = blue.floor();
    let cb = blue.ceil();
    let q1y = (fb / 8.0).floor();
    let q1x = fb - q1y * 8.0;
    let q2y = (cb / 8.0).floor();
    let q2x = cb - q2y * 8.0;
    let fetch = |quad_x: f32, quad_y: f32| -> [f32; 3] {
        let sx = quad_x * 0.125 + 0.0009765625 + 0.123046875 * r;
        let sy = quad_y * 0.125 + 0.0009765625 + 0.123046875 * g;
        let xs = sx * 512.0 - 0.5;
        let ys = sy * 512.0 - 0.5;
        let xi = xs.floor() as i32;
        let yi = ys.floor() as i32;
        let ax = ((((xs - xi as f32) * 256.0) + 0.5).floor()) / 256.0;
        let ay = ((((ys - yi as f32) * 256.0) + 0.5).floor()) / 256.0;
        let x0 = xi.clamp(0, 511) as usize;
        let x1 = (xi + 1).clamp(0, 511) as usize;
        let y0 = yi.clamp(0, 511) as usize;
        let y1 = (yi + 1).clamp(0, 511) as usize;
        let mut out = [0.0f32; 3];
        for channel in 0..3 {
            let top = table[(y0 * 512 + x0) * 3 + channel] as f32 / 255.0 * (1.0 - ax)
                + table[(y0 * 512 + x1) * 3 + channel] as f32 / 255.0 * ax;
            let bottom = table[(y1 * 512 + x0) * 3 + channel] as f32 / 255.0 * (1.0 - ax)
                + table[(y1 * 512 + x1) * 3 + channel] as f32 / 255.0 * ax;
            out[channel] = top * (1.0 - ay) + bottom * ay;
        }
        out
    };
    let c1 = fetch(q1x, q1y);
    let c2 = fetch(q2x, q2y);
    let fraction = blue - fb;
    let mut out = [0u8; 3];
    for channel in 0..3 {
        let sampled = c1[channel] * (1.0 - fraction) + c2[channel] * fraction;
        let src = [r, g, b][channel];
        out[channel] = store_unorm8(src * (1.0 - degree) + sampled * degree);
    }
    out
}

pub fn teeth_whitening(
    source: &Image,
    mask_g: &[u8],
    whiten_lut: &[u8],
    bright_lut: &[u8],
    common_ui: f32,
    bright_ui: f32,
    deyellow_ui: f32,
) -> Result<TeethWhiteningResult, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    check_plane(mask_g.len(), pixels, "mask_g")?;
    check_lut_rgb(whiten_lut, "whiten_lut")?;
    check_lut_rgb(bright_lut, "bright_lut")?;
    for (name, value) in [
        ("common_ui", common_ui),
        ("bright_ui", bright_ui),
        ("deyellow_ui", deyellow_ui),
    ] {
        if !value.is_finite() || !(0.0..=100.0).contains(&value) {
            return Err(Refusal::ValueOutOfRange(name));
        }
    }
    let common = common_ui / 100.0;
    let bright = bright_ui / 100.0;
    let deyellow = deyellow_ui / 100.0;
    let dz = common * deyellow;
    let ea = common * bright;
    let mut combined = vec![0u8; pixels * 4];
    for i in 0..pixels {
        let mask = clamp01(mask_g[i] as f32 * INV255);
        combined[i * 4] = store_unorm8(mask * dz.clamp(0.0, 1.0));
        combined[i * 4 + 1] = store_unorm8(mask * ea.clamp(0.0, 1.0));
        combined[i * 4 + 2] = 0;
        combined[i * 4 + 3] = 255;
    }
    let mut output = source.clone();
    if !(dz > 0.0 || ea > 0.0) {
        return Ok(TeethWhiteningResult { output, combined_mask: combined, dz, ea, touched: 0 });
    }
    let mut touched = 0u32;
    for i in 0..pixels {
        let whiten_weight = combined[i * 4] as f32 / 255.0;
        let bright_weight = combined[i * 4 + 1] as f32 / 255.0;
        if whiten_weight <= 0.001 && bright_weight <= 0.001 {
            continue;
        }
        let base = i * 4;
        let mut r = output.pixels[base] as f32 / 255.0;
        let mut g = output.pixels[base + 1] as f32 / 255.0;
        let mut b = output.pixels[base + 2] as f32 / 255.0;
        if whiten_weight > 0.001 {
            let sampled = teeth_lut_sample(whiten_lut, r, g, b, whiten_weight);
            r = sampled[0] as f32 / 255.0;
            g = sampled[1] as f32 / 255.0;
            b = sampled[2] as f32 / 255.0;
        }
        if bright_weight > 0.001 {
            let sampled = teeth_lut_sample(bright_lut, r, g, b, bright_weight);
            r = sampled[0] as f32 / 255.0;
            g = sampled[1] as f32 / 255.0;
            b = sampled[2] as f32 / 255.0;
        }
        output.pixels[base] = store_unorm8(r);
        output.pixels[base + 1] = store_unorm8(g);
        output.pixels[base + 2] = store_unorm8(b);
        touched += 1;
    }
    Ok(TeethWhiteningResult { output, combined_mask: combined, dz, ea, touched })
}

// ---------------------------------------------------------------------------
// Row 80 / PF30026 teeth alignment. Geometry is explicit; no model topology is
// invented. The recovered Type1007 grid direction and 0.003921 mask blend are
// used, while the vendor model point-set -> dense field remains open.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToothArch {
    Upper,
    Lower,
}

pub struct ToothGeometry<'a> {
    pub id: i32,
    pub arch: ToothArch,
    pub source: &'a [[f32; 2]],
    pub target: Option<&'a [[f32; 2]]>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TeethAlignmentParameters {
    pub ui: i32,
    pub model_priority: f32,
    pub strength: f32,
}

#[derive(Debug)]
pub struct TeethAlignmentPlan {
    pub parameters: TeethAlignmentParameters,
    pub width: u32,
    pub height: u32,
    pub mask: Vec<u8>,
    pub delta_u: Vec<f32>,
    pub delta_v: Vec<f32>,
    pub layer_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct OrientedShape {
    center: [f32; 2],
    wide: [f32; 2],
    tall: [f32; 2],
    half_wide: f32,
    half_tall: f32,
}

fn sub(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

fn add(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn scale(a: [f32; 2], value: f32) -> [f32; 2] {
    [a[0] * value, a[1] * value]
}

fn dot(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

fn length(a: [f32; 2]) -> f32 {
    dot(a, a).sqrt()
}

fn normalize(a: [f32; 2]) -> [f32; 2] {
    let len = length(a);
    if len <= 0.000001 { [1.0, 0.0] } else { scale(a, 1.0 / len) }
}

fn lerp_point(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

fn fit_shape(points: &[[f32; 2]]) -> Result<OrientedShape, Refusal> {
    if points.len() < 3 {
        return Err(Refusal::MissingInput("tooth_contour3"));
    }
    let mut center = [0.0f32, 0.0];
    for point in points {
        if !point[0].is_finite() || !point[1].is_finite() {
            return Err(Refusal::ValueOutOfRange("tooth_contour"));
        }
        center[0] += point[0];
        center[1] += point[1];
    }
    center[0] /= points.len() as f32;
    center[1] /= points.len() as f32;
    let (mut cxx, mut cxy, mut cyy) = (0.0f32, 0.0f32, 0.0f32);
    for point in points {
        let d = sub(*point, center);
        cxx += d[0] * d[0];
        cxy += d[0] * d[1];
        cyy += d[1] * d[1];
    }
    let angle = 0.5 * (2.0 * cxy).atan2(cxx - cyy);
    let major = [angle.cos(), angle.sin()];
    let mut wide = [-major[1], major[0]];
    if wide[0] < 0.0 || (wide[0].abs() < 0.0001 && wide[1] < 0.0) {
        wide = scale(wide, -1.0);
    }
    let tall = [-wide[1], wide[0]];
    let mut half_wide = 0.0f32;
    let mut half_tall = 0.0f32;
    for point in points {
        let d = sub(*point, center);
        half_wide = half_wide.max(dot(d, wide).abs());
        half_tall = half_tall.max(dot(d, tall).abs());
    }
    Ok(OrientedShape {
        center,
        wide,
        tall,
        half_wide: half_wide.max(0.5),
        half_tall: half_tall.max(0.5),
    })
}

fn solve3(matrix: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    let mut m = matrix;
    for col in 0..3 {
        let mut pivot = col;
        for row in col + 1..3 {
            if m[row][col].abs() > m[pivot][col].abs() {
                pivot = row;
            }
        }
        if m[pivot][col].abs() < 1e-12 {
            return None;
        }
        if pivot != col {
            m.swap(col, pivot);
        }
        let divisor = m[col][col];
        for k in col..4 {
            m[col][k] /= divisor;
        }
        for row in 0..3 {
            if row == col {
                continue;
            }
            let factor = m[row][col];
            for k in col..4 {
                m[row][k] -= factor * m[col][k];
            }
        }
    }
    Some([m[0][3], m[1][3], m[2][3]])
}

fn fit_affine(source: &[[f32; 2]], target: &[[f32; 2]]) -> Result<[f32; 6], Refusal> {
    if source.len() != target.len() || source.len() < 3 {
        return Err(Refusal::ShapeMismatch("affine_points"));
    }
    let (mut sxx, mut sxy, mut sx, mut syy, mut sy, mut n) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    let (mut bx, mut by, mut b1, mut ex, mut ey, mut e1) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for i in 0..source.len() {
        let x = source[i][0] as f64;
        let y = source[i][1] as f64;
        let xr = target[i][0] as f64;
        let yr = target[i][1] as f64;
        sxx += x * x;
        sxy += x * y;
        sx += x;
        syy += y * y;
        sy += y;
        n += 1.0;
        bx += x * xr;
        by += y * xr;
        b1 += xr;
        ex += x * yr;
        ey += y * yr;
        e1 += yr;
    }
    let x = solve3([[sxx, sxy, sx, bx], [sxy, syy, sy, by], [sx, sy, n, b1]])
        .ok_or(Refusal::ValueOutOfRange("affine_degenerate"))?;
    let y = solve3([[sxx, sxy, sx, ex], [sxy, syy, sy, ey], [sx, sy, n, e1]])
        .ok_or(Refusal::ValueOutOfRange("affine_degenerate"))?;
    Ok([x[0] as f32, x[1] as f32, x[2] as f32, y[0] as f32, y[1] as f32, y[2] as f32])
}

fn apply_affine(affine: [f32; 6], point: [f32; 2]) -> [f32; 2] {
    [
        affine[0] * point[0] + affine[1] * point[1] + affine[2],
        affine[3] * point[0] + affine[4] * point[1] + affine[5],
    ]
}

#[derive(Clone, Copy)]
struct ArchCurve {
    a: f32,
    b: f32,
    c: f32,
}

impl ArchCurve {
    fn point_at(self, x: f32) -> [f32; 2] {
        [x, self.a * x * x + self.b * x + self.c]
    }

    fn tangent_at(self, x: f32) -> [f32; 2] {
        [1.0, 2.0 * self.a * x + self.b]
    }

    fn fit(points: &[[f32; 2]]) -> Self {
        if points.len() < 2 {
            return Self { a: 0.0, b: 0.0, c: points.first().map(|p| p[1]).unwrap_or(0.0) };
        }
        if points.len() == 2 {
            let mut dx = points[1][0] - points[0][0];
            if dx.abs() < 0.0001 {
                dx = if points[1][0] < points[0][0] { 0.0001 } else { -0.0001 };
            }
            let slope = (points[1][1] - points[0][1]) / dx;
            return Self { a: 0.0, b: slope, c: points[0][1] - slope * points[0][0] };
        }
        let (mut s0, mut s1, mut s2, mut s3, mut s4) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut ty, mut txy, mut tx2y) = (0.0, 0.0, 0.0);
        for point in points {
            let x = point[0] as f64;
            let y = point[1] as f64;
            let x2 = x * x;
            s0 += 1.0;
            s1 += x;
            s2 += x2;
            s3 += x2 * x;
            s4 += x2 * x2;
            ty += y;
            txy += x * y;
            tx2y += x2 * y;
        }
        match solve3([
            [s4, s3, s2, tx2y],
            [s3, s2, s1, txy],
            [s2, s1, s0, ty],
        ]) {
            Some(v) => Self { a: v[0] as f32, b: v[1] as f32, c: v[2] as f32 },
            None => Self { a: 0.0, b: 0.0, c: (ty / s0) as f32 },
        }
    }
}

fn median(values: &[f32]) -> f32 {
    let mut copy = values.to_vec();
    copy.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let middle = copy.len() / 2;
    if copy.len() & 1 == 1 {
        copy[middle]
    } else {
        (copy[middle - 1] + copy[middle]) * 0.5
    }
}

fn polygon_bounds(polygon: &[[f32; 2]], width: u32, height: u32, pad: i32) -> Rect {
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for point in polygon {
        min_x = min_x.min(point[0]);
        min_y = min_y.min(point[1]);
        max_x = max_x.max(point[0]);
        max_y = max_y.max(point[1]);
    }
    let x = (min_x.floor() as i32 - pad).clamp(0, width as i32);
    let y = (min_y.floor() as i32 - pad).clamp(0, height as i32);
    let right = (max_x.ceil() as i32 + pad).clamp(0, width as i32);
    let bottom = (max_y.ceil() as i32 + pad).clamp(0, height as i32);
    Rect { x, y, width: (right - x).max(0), height: (bottom - y).max(0) }
}

fn point_in_polygon(point: [f32; 2], polygon: &[[f32; 2]]) -> bool {
    let mut inside = false;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let a = polygon[i];
        let b = polygon[j];
        if (a[1] > point[1]) != (b[1] > point[1])
            && point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]
        {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn segment_distance(point: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let ab = sub(b, a);
    let denominator = dot(ab, ab);
    let t = if denominator <= 0.000001 {
        0.0
    } else {
        clamp01(dot(sub(point, a), ab) / denominator)
    };
    length(sub(point, add(a, scale(ab, t))))
}

fn polygon_coverage(point: [f32; 2], polygon: &[[f32; 2]], feather: f32) -> f32 {
    if point_in_polygon(point, polygon) {
        return 1.0;
    }
    if feather <= 0.0 {
        return 0.0;
    }
    let mut distance = f32::MAX;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        distance = distance.min(segment_distance(point, polygon[j], polygon[i]));
        j = i;
    }
    if distance >= feather {
        return 0.0;
    }
    let t = 1.0 - distance / feather;
    t * t * (3.0 - 2.0 * t)
}

fn build_arch_targets(
    shapes: &[(ToothGeometry<'_>, OrientedShape, Option<OrientedShape>)],
    indexes: &mut Vec<usize>,
    targets: &mut [Option<OrientedShape>],
) {
    indexes.sort_by(|a, b| {
        shapes[*a].1.center[0]
            .partial_cmp(&shapes[*b].1.center[0])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut widths = Vec::with_capacity(indexes.len());
    let mut heights = Vec::with_capacity(indexes.len());
    for index in indexes.iter().copied() {
        if let Some(target) = shapes[index].2 {
            targets[index] = Some(target);
            widths.push(target.half_wide);
            heights.push(target.half_tall);
        } else {
            widths.push(shapes[index].1.half_wide);
            heights.push(shapes[index].1.half_tall);
        }
    }
    let median_width = median(&widths);
    let median_height = median(&heights);
    let centers: Vec<[f32; 2]> = indexes.iter().map(|i| shapes[*i].1.center).collect();
    let curve = ArchCurve::fit(&centers);
    for index in indexes.iter().copied() {
        if targets[index].map(|t| t.half_wide > 0.0).unwrap_or(false) {
            continue;
        }
        let center = shapes[index].1.center;
        let arch = curve.point_at(center[0]);
        let mut wide = normalize(curve.tangent_at(center[0]));
        if wide[0] < 0.0 || (wide[0].abs() < 0.0001 && wide[1] < 0.0) {
            wide = scale(wide, -1.0);
        }
        let tall = [-wide[1], wide[0]];
        targets[index] = Some(OrientedShape {
            center: arch,
            wide,
            tall,
            half_wide: median_width,
            half_tall: median_height,
        });
    }
}

fn build_target_contour(
    tooth: &ToothGeometry<'_>,
    source: OrientedShape,
    target: OrientedShape,
    strength: f32,
) -> Result<Vec<[f32; 2]>, Refusal> {
    if let Some(explicit) = tooth.target {
        if explicit.len() == tooth.source.len() {
            let affine = fit_affine(tooth.source, explicit)?;
            return Ok(tooth
                .source
                .iter()
                .map(|point| lerp_point(*point, apply_affine(affine, *point), strength))
                .collect());
        }
    }
    Ok(tooth
        .source
        .iter()
        .map(|point| {
            let nx = dot(sub(*point, source.center), source.wide) / source.half_wide;
            let ny = dot(sub(*point, source.center), source.tall) / source.half_tall;
            add(
                target.center,
                add(scale(target.wide, nx * target.half_wide), scale(target.tall, ny * target.half_tall)),
            )
        })
        .collect())
}

fn build_alignment_layer(
    width: u32,
    height: u32,
    query: [f32; 2],
    source: OrientedShape,
    target: OrientedShape,
    contour: &[[f32; 2]],
    feather: f32,
    mask_sum: &mut [f32],
    delta_u: &mut [f32],
    delta_v: &mut [f32],
    pad: i32,
) -> Result<u32, Refusal> {
    let bounds = polygon_bounds(contour, width, height, pad);
    let mut active = 0u32;
    for ly in 0..bounds.height {
        let y = bounds.y + ly;
        for lx in 0..bounds.width {
            let x = bounds.x + lx;
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let coverage = polygon_coverage(point, contour, feather);
            let m8 = round_ties_even((coverage * 255.0) as f64).clamp(0.0, 255.0) as u8;
            if m8 == 0 {
                continue;
            }
            let nx = dot(sub(point, target.center), target.wide) / target.half_wide;
            let ny = dot(sub(point, target.center), target.tall) / target.half_tall;
            let source_point = add(
                source.center,
                add(scale(source.wide, nx * source.half_wide), scale(source.tall, ny * source.half_tall)),
            );
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            let du = source_point[0] / width as f32 - u;
            let dv = v - source_point[1] / height as f32;
            let weight = m8 as f32 / 255.0;
            let global = y as usize * width as usize + x as usize;
            mask_sum[global] += weight;
            delta_u[global] += du * weight;
            delta_v[global] += dv * weight;
            active += 1;
        }
    }
    let _ = query;
    Ok(active)
}

pub fn teeth_alignment_plan(
    width: u32,
    height: u32,
    ui: i32,
    teeth: &[ToothGeometry<'_>],
    feather: f32,
) -> Result<TeethAlignmentPlan, Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("frame"));
    }
    if !(0..=5).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if !feather.is_finite() || feather < 0.0 {
        return Err(Refusal::ValueOutOfRange("feather"));
    }
    let parameters = TeethAlignmentParameters {
        ui,
        model_priority: ui as f32 * 5.0,
        strength: ui as f32 / 5.0,
    };
    let pixels = width as usize * height as usize;
    if parameters.strength <= 0.0 {
        return Ok(TeethAlignmentPlan {
            parameters,
            width,
            height,
            mask: vec![0; pixels],
            delta_u: vec![0.0; pixels],
            delta_v: vec![0.0; pixels],
            layer_count: 0,
        });
    }
    if teeth.is_empty() {
        return Err(Refusal::MissingInput("tooth_contours"));
    }

    let mut shapes = Vec::with_capacity(teeth.len());
    for tooth in teeth {
        let source = fit_shape(tooth.source)?;
        let target = match tooth.target {
            Some(points) => Some(fit_shape(points)?),
            None => None,
        };
        shapes.push((ToothGeometry { id: tooth.id, arch: tooth.arch, source: tooth.source, target: tooth.target }, source, target));
    }
    let mut targets = vec![None; shapes.len()];
    for arch in [ToothArch::Upper, ToothArch::Lower] {
        let mut indexes: Vec<usize> = shapes
            .iter()
            .enumerate()
            .filter(|(_, shape)| shape.0.arch == arch)
            .map(|(index, _)| index)
            .collect();
        if !indexes.is_empty() {
            build_arch_targets(&shapes, &mut indexes, &mut targets);
        }
    }

    let mut mask_sum = vec![0.0f32; pixels];
    let mut delta_u = vec![0.0f32; pixels];
    let mut delta_v = vec![0.0f32; pixels];
    let mut layer_count = 0;
    let pad = feather.ceil() as i32 + 2;
    for (index, shape) in shapes.iter().enumerate() {
        let target = targets[index].ok_or(Refusal::MissingInput("tooth_target"))?;
        let contour = build_target_contour(&shape.0, shape.1, target, parameters.strength)?;
        build_alignment_layer(
            width,
            height,
            [0.0, 0.0],
            shape.1,
            target,
            &contour,
            feather,
            &mut mask_sum,
            &mut delta_u,
            &mut delta_v,
            pad,
        )?;
        layer_count += 1;
    }

    let mut mask = vec![0u8; pixels];
    for i in 0..pixels {
        if mask_sum[i] <= 0.0 {
            delta_u[i] = 0.0;
            delta_v[i] = 0.0;
            continue;
        }
        mask[i] = round_ties_even((mask_sum[i] * 255.0) as f64).clamp(0.0, 255.0) as u8;
        delta_u[i] /= mask_sum[i];
        delta_v[i] /= mask_sum[i];
    }
    Ok(TeethAlignmentPlan { parameters, width, height, mask, delta_u, delta_v, layer_count })
}

fn sample_plane(plane: &[f32], width: u32, height: u32, u: f32, v: f32) -> f32 {
    let px = u * width as f32 - 0.5;
    let py = v * height as f32 - 0.5;
    let x0f = px.floor();
    let y0f = py.floor();
    let tx = px - x0f;
    let ty = py - y0f;
    let x0 = (x0f as i32).clamp(0, width as i32 - 1) as usize;
    let x1 = ((x0f as i32) + 1).clamp(0, width as i32 - 1) as usize;
    let y0 = (y0f as i32).clamp(0, height as i32 - 1) as usize;
    let y1 = ((y0f as i32) + 1).clamp(0, height as i32 - 1) as usize;
    let w = width as usize;
    let top = plane[y0 * w + x0] * (1.0 - tx) + plane[y0 * w + x1] * tx;
    let bottom = plane[y1 * w + x0] * (1.0 - tx) + plane[y1 * w + x1] * tx;
    top * (1.0 - ty) + bottom * ty
}

fn sample_image_uv(image: &Image, u: f32, v: f32) -> [u8; 3] {
    let px = u * image.width as f32 - 0.5;
    let py = v * image.height as f32 - 0.5;
    let x0f = px.floor();
    let y0f = py.floor();
    let tx = px - x0f;
    let ty = py - y0f;
    let x0 = (x0f as i32).clamp(0, image.width as i32 - 1) as u32;
    let x1 = ((x0f as i32) + 1).clamp(0, image.width as i32 - 1) as u32;
    let y0 = (y0f as i32).clamp(0, image.height as i32 - 1) as u32;
    let y1 = ((y0f as i32) + 1).clamp(0, image.height as i32 - 1) as u32;
    let at = |x: u32, y: u32, c: usize| image.pixels[((y * image.width + x) * 4) as usize + c] as f32;
    let mut out = [0u8; 3];
    for channel in 0..3 {
        let top = at(x0, y0, channel) * (1.0 - tx) + at(x1, y0, channel) * tx;
        let bottom = at(x0, y1, channel) * (1.0 - tx) + at(x1, y1, channel) * tx;
        out[channel] = round_ties_even((top * (1.0 - ty) + bottom * ty) as f64) as u8;
    }
    out
}

pub fn teeth_alignment_warp(source: &Image, plan: &TeethAlignmentPlan) -> Result<Image, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    if source.width != plan.width || source.height != plan.height {
        return Err(Refusal::ShapeMismatch("plan"));
    }
    if plan.delta_u.len() != pixels || plan.delta_v.len() != pixels {
        return Err(Refusal::ShapeMismatch("delta"));
    }
    let mut output = source.clone();
    for y in 0..source.height {
        for x in 0..source.width {
            let u = (x as f32 + 0.5) / source.width as f32;
            let v = (y as f32 + 0.5) / source.height as f32;
            let du = sample_plane(&plan.delta_u, source.width, source.height, u, v);
            let dv = sample_plane(&plan.delta_v, source.width, source.height, u, v);
            let sampled = sample_image_uv(source, u + du, v - dv);
            let base = (y as usize * source.width as usize + x as usize) * 4;
            output.pixels[base] = sampled[0];
            output.pixels[base + 1] = sampled[1];
            output.pixels[base + 2] = sampled[2];
        }
    }
    Ok(output)
}

pub fn teeth_alignment_blend(original: &Image, effect: &Image, mask: &[u8]) -> Result<Image, Refusal> {
    let pixels = checked_pixels(original, "original")?;
    if checked_pixels(effect, "effect")? != pixels {
        return Err(Refusal::ShapeMismatch("effect"));
    }
    check_plane(mask.len(), pixels, "mask")?;
    let mut output = original.clone();
    for pixel in 0..pixels {
        let m8 = mask[pixel];
        let base = pixel * 4;
        if m8 == 255 {
            output.pixels[base] = effect.pixels[base];
            output.pixels[base + 1] = effect.pixels[base + 1];
            output.pixels[base + 2] = effect.pixels[base + 2];
            continue;
        }
        let m = m8 as f32 * 0.003921;
        for channel in 0..3 {
            let value = m * effect.pixels[base + channel] as f32
                + (1.0 - m) * original.pixels[base + channel] as f32;
            output.pixels[base + channel] = byte_trunc(value);
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Row 81 / PF100004 red-vein model composition.
// ---------------------------------------------------------------------------

pub fn red_vein(
    source: &Image,
    effect_rgb: &[u8],
    alpha_plane: &[u8],
    ui: f32,
    strength: f32,
) -> Result<Image, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    if effect_rgb.len() != pixels * 3 {
        return Err(Refusal::ShapeMismatch("effect_rgb"));
    }
    check_plane(alpha_plane.len(), pixels, "alpha_plane")?;
    if !ui.is_finite() || ui < 0.0 || ui > 100.0 {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if !strength.is_finite() {
        return Err(Refusal::ValueOutOfRange("strength"));
    }
    let mut scale = strength.clamp(0.0, 1.0);
    if ui <= 0.0 {
        scale = 0.0;
    }
    let mut output = source.clone();
    for pixel in 0..pixels {
        let alpha = alpha_plane[pixel] as f64 / 255.0 * scale as f64;
        if alpha <= 0.0 {
            continue;
        }
        let base = pixel * 4;
        let effect = pixel * 3;
        for channel in 0..3 {
            let src = source.pixels[base + channel] as f64 / 255.0;
            let eff = effect_rgb[effect + channel] as f64 / 255.0;
            output.pixels[base + channel] = byte_even(src * (1.0 - alpha) + eff * alpha);
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Row 82 / PF100005 eye-white purify: UI/100, 4+4 split, then a typed refusal
// because the vendor post-model blend lives in VMProtect .iVD0 and is open.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeWhiteSplit {
    pub first4: [f32; 4],
    pub second4: [f32; 4],
}

pub fn eye_white_split8(model_output: &[f32]) -> Result<EyeWhiteSplit, Refusal> {
    if model_output.len() != 8 {
        return Err(Refusal::ShapeMismatch("model_output8"));
    }
    if !model_output.iter().all(|value| value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("model_output8"));
    }
    Ok(EyeWhiteSplit {
        first4: [model_output[0], model_output[1], model_output[2], model_output[3]],
        second4: [model_output[4], model_output[5], model_output[6], model_output[7]],
    })
}

pub fn eye_white_purify(
    source: &Image,
    roi: Rect,
    roi_mask: &[u8],
    model_output8: &[f32],
    ui: f32,
) -> Result<Image, Refusal> {
    checked_pixels(source, "source")?;
    if !roi.inside(source.width, source.height) {
        return Err(Refusal::ValueOutOfRange("roi"));
    }
    let roi_pixels = roi.width as usize * roi.height as usize;
    check_plane(roi_mask.len(), roi_pixels, "roi_mask")?;
    eye_white_split8(model_output8)?;
    if !ui.is_finite() || ui < 0.0 || ui > 100.0 {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if ui == 0.0 {
        return Ok(source.clone());
    }
    Err(Refusal::NotImplemented(100005))
}

// ---------------------------------------------------------------------------
// Row 83 / PF90130 glasses glare.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlareMaskEncoding {
    SignedNetwork,
    UnormLens,
}

fn knee(value: f32) -> f32 {
    let value = clamp01(value);
    if value < 0.2 {
        0.0
    } else if value > 0.3 {
        1.0
    } else {
        value
    }
}

fn glare_effect_byte(value: f32) -> u8 {
    round_ties_even(((value as f64 + 1.0).clamp(0.0, 2.0) * 127.5).clamp(0.0, 255.0)) as u8
}

fn glare_mask_byte(value: f32) -> u8 {
    ((clamp01(value) as f64 * 255.0 + 0.5).floor()).clamp(0.0, 255.0) as u8
}

fn glare_blend(mask: u8, effect: u8, source: u8) -> u8 {
    let value = mask as i32 * effect as i32 + (255 - mask as i32) * source as i32 + 1;
    ((value + (value >> 8)) >> 8) as u8
}

pub fn glasses_glare_removal(
    source: &Image,
    effect_rgb: &[f32],
    mask: Option<&[f32]>,
    lens_mask: Option<&[f32]>,
    encoding: GlareMaskEncoding,
    ui_percent: i32,
) -> Result<Image, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    if effect_rgb.len() != pixels * 3 {
        return Err(Refusal::ShapeMismatch("effect_rgb"));
    }
    if mask.is_none() && lens_mask.is_none() {
        return Err(Refusal::MissingInput("network_mask_or_lens_mask"));
    }
    if let Some(values) = mask {
        check_plane(values.len(), pixels, "mask")?;
        if !values.iter().all(|value| value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("mask"));
        }
    }
    if let Some(values) = lens_mask {
        check_plane(values.len(), pixels, "lens_mask")?;
        if !values.iter().all(|value| value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("lens_mask"));
        }
    }
    if !(0..=100).contains(&ui_percent) {
        return Err(Refusal::ValueOutOfRange("ui_percent"));
    }
    let strength = ui_percent as f32 / 100.0;
    let mut output = source.clone();
    for pixel in 0..pixels {
        let unit = match mask {
            None => 1.0,
            Some(values) => match encoding {
                GlareMaskEncoding::SignedNetwork => clamp01((values[pixel] + 1.0) * 0.5),
                GlareMaskEncoding::UnormLens => clamp01(values[pixel]),
            },
        };
        let lens = lens_mask.map(|values| clamp01(values[pixel])).unwrap_or(1.0);
        let mask8 = glare_mask_byte(knee(unit) * lens * strength);
        let base = pixel * 4;
        let effect = pixel * 3;
        for channel in 0..3 {
            let eff = glare_effect_byte(effect_rgb[effect + channel]);
            output.pixels[base + channel] = glare_blend(mask8, eff, source.pixels[base + channel]);
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Rows 84 and 85 / PF9 highlight and PF8 shadow.
// ---------------------------------------------------------------------------

fn soft_light(base: f32, blend: f32) -> f32 {
    if blend < 0.5 {
        2.0 * base * blend + base * base * (1.0 - 2.0 * blend)
    } else {
        base.sqrt() * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend)
    }
}

fn rgba_mask_weight(pixel: [u8; 4]) -> f32 {
    clamp01((pixel[0] as f32 + pixel[1] as f32 + pixel[2] as f32 + pixel[3] as f32) / 255.0)
}

fn sample_rgba_bilinear(image: &Image, u: f32, v: f32) -> [u8; 4] {
    let x = clamp01(u) * image.width as f32 - 0.5;
    let y = clamp01(v) * image.height as f32 - 0.5;
    let x0f = x.floor();
    let y0f = y.floor();
    let tx = x - x0f;
    let ty = y - y0f;
    let x0 = (x0f as i32).clamp(0, image.width as i32 - 1) as u32;
    let x1 = ((x0f as i32) + 1).clamp(0, image.width as i32 - 1) as u32;
    let y0 = (y0f as i32).clamp(0, image.height as i32 - 1) as u32;
    let y1 = ((y0f as i32) + 1).clamp(0, image.height as i32 - 1) as u32;
    let at = |px: u32, py: u32, channel: usize| {
        image.pixels[((py * image.width + px) * 4) as usize + channel] as f32
    };
    let mut out = [0u8; 4];
    for channel in 0..4 {
        let top = at(x0, y0, channel) * (1.0 - tx) + at(x1, y0, channel) * tx;
        let bottom = at(x0, y1, channel) * (1.0 - tx) + at(x1, y1, channel) * tx;
        out[channel] = byte_away(top * (1.0 - ty) + bottom * ty);
    }
    out
}

pub fn stereo_highlight(
    source: &Image,
    material: &Image,
    mask: &Image,
    roi: Rect,
    ui: f64,
    opacity: f32,
    mix_alpha: f32,
) -> Result<Image, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    checked_pixels(material, "material")?;
    checked_pixels(mask, "mask")?;
    if !roi.inside(source.width, source.height) {
        return Err(Refusal::ValueOutOfRange("roi"));
    }
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    for (name, value) in [("opacity", opacity), ("mix_alpha", mix_alpha)] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(Refusal::ValueOutOfRange(name));
        }
    }
    let degree = (ui / 100.0).clamp(0.0, 1.0) as f32;
    let mut output = source.clone();
    for y in roi.y..roi.y + roi.height {
        let v = ((y - roi.y) as f32 + 0.5) / roi.height as f32;
        for x in roi.x..roi.x + roi.width {
            let u = ((x - roi.x) as f32 + 0.5) / roi.width as f32;
            let mat = sample_rgba_bilinear(material, u, v);
            let mask_pixel = sample_rgba_bilinear(mask, u, v);
            let factor = mat[3] as f32 / 255.0
                * degree
                * opacity
                * mix_alpha
                * rgba_mask_weight(mask_pixel);
            let pixel = y as usize * source.width as usize + x as usize;
            let base = pixel * 4;
            if factor <= 0.01 {
                continue;
            }
            for channel in 0..3 {
                let src = source.pixels[base + channel] as f32 / 255.0;
                let blend = mat[channel] as f32 / 255.0;
                let mixed = src * (1.0 - factor) + soft_light(src, blend) * factor;
                output.pixels[base + channel] = byte_away(mixed * 255.0);
            }
        }
    }
    let _ = pixels;
    Ok(output)
}

pub fn stereo_shadow(
    source: &Image,
    material: &Image,
    face_mask: &Image,
    roi: Rect,
    ui: f32,
    mix_color_alpha: f32,
) -> Result<Image, Refusal> {
    checked_pixels(source, "source")?;
    if !roi.inside(source.width, source.height) {
        return Err(Refusal::ValueOutOfRange("roi"));
    }
    if material.width != roi.width as u32
        || material.height != roi.height as u32
        || checked_pixels(material, "material")? != roi.width as usize * roi.height as usize
    {
        return Err(Refusal::ShapeMismatch("material"));
    }
    if face_mask.width != roi.width as u32
        || face_mask.height != roi.height as u32
        || checked_pixels(face_mask, "face_mask")? != roi.width as usize * roi.height as usize
    {
        return Err(Refusal::ShapeMismatch("face_mask"));
    }
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if !mix_color_alpha.is_finite() || !(0.0..=1.0).contains(&mix_color_alpha) {
        return Err(Refusal::ValueOutOfRange("mix_color_alpha"));
    }
    let degree = ui / 100.0;
    let mix_alpha = mix_color_alpha * degree;
    let mut output = source.clone();
    for ly in 0..roi.height as u32 {
        for lx in 0..roi.width as u32 {
            let plane = (ly * roi.width as u32 + lx) as usize;
            let mat_base = plane * 4;
            let mask_weight = clamp01(
                (face_mask.pixels[mat_base] as f32
                    + face_mask.pixels[mat_base + 1] as f32
                    + face_mask.pixels[mat_base + 2] as f32
                    + face_mask.pixels[mat_base + 3] as f32)
                    / 255.0,
            );
            let factor = material.pixels[mat_base + 3] as f32 / 255.0 * mix_alpha * mask_weight;
            if !(factor > 0.01) {
                continue;
            }
            let x = roi.x as u32 + lx;
            let y = roi.y as u32 + ly;
            let base = (y as usize * source.width as usize + x as usize) * 4;
            for channel in 0..3 {
                let src = source.pixels[base + channel] as f32 / 255.0;
                let blend = material.pixels[mat_base + channel] as f32 / 255.0;
                let mixed = src * (1.0 - factor) + soft_light(src, blend) * factor;
                output.pixels[base + channel] = round_ties_even((mixed * 255.0) as f64) as u8;
            }
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Row 86 / PF90021 lip makeup.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LipDirection {
    Neutral,
    Positive,
    Negative,
}

fn positive_lip_effect(input: f32) -> f32 {
    // ParamRange [0,0.5,1] -> PositiveEffectRange [0,0,1].
    if input <= 0.5 {
        0.0
    } else if input >= 1.0 {
        1.0
    } else {
        (input - 0.5) / 0.5
    }
}

fn lip_direction(ui: f32) -> LipDirection {
    let normalized = (ui + 100.0) / 200.0;
    if normalized > 0.5 {
        LipDirection::Positive
    } else if normalized < 0.5 {
        LipDirection::Negative
    } else {
        LipDirection::Neutral
    }
}

fn lip_effect_magnitude(ui: f32) -> f32 {
    let normalized = (ui + 100.0) / 200.0;
    match lip_direction(ui) {
        LipDirection::Neutral => 0.0,
        LipDirection::Positive => positive_lip_effect(normalized),
        LipDirection::Negative => {
            if normalized <= 0.0 {
                1.0
            } else if normalized <= 0.5 {
                1.0 - 2.0 * normalized
            } else {
                0.0
            }
        }
    }
}

fn lip_to_byte(value: f32) -> u8 {
    ((value * 255.0 + 0.5).floor()).clamp(0.0, 255.0) as u8
}

pub fn lip_makeup(
    source: &Image,
    mask: &[f32],
    positive_lut: &[u8],
    negative_lut: &[u8],
    ui: f32,
    opacity_percent: f32,
    face_weight: f32,
    shelter_weight: f32,
) -> Result<Image, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    check_plane(mask.len(), pixels, "mask")?;
    check_lut_rgb(positive_lut, "positive_lut")?;
    check_lut_rgb(negative_lut, "negative_lut")?;
    if !ui.is_finite() || !(-100.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if !opacity_percent.is_finite() || !(0.0..=100.0).contains(&opacity_percent) {
        return Err(Refusal::ValueOutOfRange("opacity_percent"));
    }
    for (name, value) in [("face_weight", face_weight), ("shelter_weight", shelter_weight)] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(Refusal::ValueOutOfRange(name));
        }
    }
    if !mask.iter().all(|value| value.is_finite() && (0.0..=1.0).contains(value)) {
        return Err(Refusal::ValueOutOfRange("mask"));
    }
    let direction = lip_direction(ui);
    let effect = lip_effect_magnitude(ui);
    let u_alpha = effect * opacity_percent / 100.0 * face_weight * shelter_weight;
    let lut = match direction {
        LipDirection::Positive => positive_lut,
        LipDirection::Negative => negative_lut,
        LipDirection::Neutral => return Ok(source.clone()),
    };
    let mut output = source.clone();
    for pixel in 0..pixels {
        let weight = mask[pixel];
        if weight <= 0.0 {
            continue;
        }
        let base = pixel * 4;
        let r = source.pixels[base] as f32 / 255.0;
        let g = source.pixels[base + 1] as f32 / 255.0;
        let b = source.pixels[base + 2] as f32 / 255.0;
        let reference = lut3d_sample(lut, r, g, b);
        let src = [r, g, b];
        for channel in 0..3 {
            let res = (src[channel] + (reference[channel] as f32 - src[channel]) * u_alpha)
                .clamp(0.0, 1.0);
            output.pixels[base + channel] = lip_to_byte(src[channel] + (res - src[channel]) * weight);
        }
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Row 87 / 14-PF makeup degree group.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MakeupDegreeRow {
    pub pf: u32,
    pub key: &'static str,
    pub default_degree: f32,
    pub piecewise: bool,
}

pub const MAKEUP_DEGREE_GROUP: &[MakeupDegreeRow] = &[
    MakeupDegreeRow { pf: 100, key: "eyebrowDegree_left", default_degree: 70.0, piecewise: true },
    MakeupDegreeRow { pf: 101, key: "silkwormDegree", default_degree: 70.0, piecewise: true },
    MakeupDegreeRow { pf: 103, key: "eyelinerDegree_upper", default_degree: 70.0, piecewise: false },
    MakeupDegreeRow { pf: 105, key: "eyelashDegree_upper", default_degree: 70.0, piecewise: false },
    MakeupDegreeRow { pf: 106, key: "blusherDegree", default_degree: 50.0, piecewise: false },
    MakeupDegreeRow { pf: 107, key: "lipstickDegree", default_degree: 50.0, piecewise: false },
    MakeupDegreeRow { pf: 111, key: "eyePupilDegree", default_degree: 50.0, piecewise: false },
    MakeupDegreeRow { pf: 115, key: "faceDecorationDegree", default_degree: 70.0, piecewise: false },
    MakeupDegreeRow { pf: 116, key: "contourDegree", default_degree: 50.0, piecewise: false },
    MakeupDegreeRow { pf: 117, key: "eyebrowDegree_right", default_degree: 70.0, piecewise: true },
    MakeupDegreeRow { pf: 118, key: "eyelashDegree_lower", default_degree: 70.0, piecewise: false },
    MakeupDegreeRow { pf: 30007, key: "eyelinerDegree_lower", default_degree: 70.0, piecewise: false },
    MakeupDegreeRow { pf: 90032, key: "eyeMakeupDegree", default_degree: 70.0, piecewise: false },
    MakeupDegreeRow { pf: 90033, key: "suitMakeupDegree", default_degree: 70.0, piecewise: false },
];

pub fn makeup_degree_row(pf: u32) -> Option<MakeupDegreeRow> {
    MAKEUP_DEGREE_GROUP.iter().copied().find(|row| row.pf == pf)
}

pub fn makeup_degree(pf: u32, ui: f32) -> Result<f32, Refusal> {
    let _row = makeup_degree_row(pf).ok_or(Refusal::MissingInput("makeup_degree_pf"))?;
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    let normalized = ui / 100.0;
    // Both recovered tables ([0,1] -> [0,1] and top-level DegreeRange [0,100])
    // reduce to the same UI/100 scalar.
    Ok(normalized)
}

// ---------------------------------------------------------------------------
// Row 88 / PF90087 hair seam fill.
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct HairSeamResult {
    pub output: Image,
    pub degree: f32,
    pub half_x: f32,
    pub half_y: f32,
    pub probe_limit: u32,
    pub part_line_pixels: u32,
    pub candidate_pixels: u32,
    pub full_weight_pixels: u32,
    pub changed: u32,
    pub alpha_changed: u32,
}

fn sample_mask_at(mask: &[u8], width: usize, height: usize, x: f32, y: f32) -> Option<f32> {
    if !(x >= 0.0) || !(y >= 0.0) || x > (width - 1) as f32 || y > (height - 1) as f32 {
        return None;
    }
    let x0 = x as usize;
    let y0 = y as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let p00 = mask[y0 * width + x0] as f32;
    let p10 = mask[y0 * width + x1] as f32;
    let p01 = mask[y1 * width + x0] as f32;
    let p11 = mask[y1 * width + x1] as f32;
    let top = p00 + (p10 - p00) * tx;
    let bottom = p01 + (p11 - p01) * tx;
    Some(top + (bottom - top) * ty)
}

fn sample_source_at(image: &Image, x: f32, y: f32) -> Option<[f32; 3]> {
    let width = image.width as usize;
    let height = image.height as usize;
    if !(x >= 0.0) || !(y >= 0.0) || x > (width - 1) as f32 || y > (height - 1) as f32 {
        return None;
    }
    let x0 = x as usize;
    let y0 = y as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let at = |px: usize, py: usize, c: usize| image.pixels[(py * width + px) * 4 + c] as f32;
    let mut out = [0.0; 3];
    for channel in 0..3 {
        let top = at(x0, y0, channel) + (at(x1, y0, channel) - at(x0, y0, channel)) * tx;
        let bottom = at(x0, y1, channel) + (at(x1, y1, channel) - at(x0, y1, channel)) * tx;
        out[channel] = top + (bottom - top) * ty;
    }
    Some(out)
}

fn normalize2(x: &mut f32, y: &mut f32) {
    let len = (*x * *x + *y * *y).sqrt();
    if !len.is_finite() || len < 1e-6 {
        *x = 0.0;
        *y = 0.0;
    } else {
        *x /= len;
        *y /= len;
    }
}

fn find_pair(
    source: &Image,
    hair: &[u8],
    x: f32,
    y: f32,
    nx: f32,
    ny: f32,
    radius: u32,
) -> Option<(f32, [f32; 3], [f32; 3])> {
    let width = source.width as usize;
    let height = source.height as usize;
    let mut nx = nx;
    let mut ny = ny;
    normalize2(&mut nx, &mut ny);
    for distance in 1..=radius {
        let d = distance as f32;
        let plus = (x + nx * d, y + ny * d);
        let minus = (x - nx * d, y - ny * d);
        let plus_hair = sample_mask_at(hair, width, height, plus.0, plus.1).unwrap_or(0.0) >= 128.0;
        let minus_hair = sample_mask_at(hair, width, height, minus.0, minus.1).unwrap_or(0.0) >= 128.0;
        if !plus_hair || !minus_hair {
            continue;
        }
        let plus_rgb = match sample_source_at(source, plus.0, plus.1) {
            Some(value) => value,
            None => continue,
        };
        let minus_rgb = match sample_source_at(source, minus.0, minus.1) {
            Some(value) => value,
            None => continue,
        };
        return Some((d, plus_rgb, minus_rgb));
    }
    None
}

fn find_best_angle(
    source: &Image,
    hair: &[u8],
    x: f32,
    y: f32,
    radius: u32,
) -> Option<(f32, [f32; 3], [f32; 3])> {
    let width = source.width as usize;
    let height = source.height as usize;
    let mut best: Option<(f32, [f32; 3], [f32; 3])> = None;
    let mut best_score = f32::INFINITY;
    for direction in 0..16 {
        let angle = direction as f32 * std::f32::consts::PI / 8.0;
        let nx = angle.cos();
        let ny = angle.sin();
        for distance in 1..=radius {
            let d = distance as f32;
            let plus = (x + nx * d, y + ny * d);
            let minus = (x - nx * d, y - ny * d);
            let plus_hair = sample_mask_at(hair, width, height, plus.0, plus.1).unwrap_or(0.0) >= 128.0;
            let minus_hair = sample_mask_at(hair, width, height, minus.0, minus.1).unwrap_or(0.0) >= 128.0;
            if !plus_hair || !minus_hair {
                continue;
            }
            let plus_rgb = match sample_source_at(source, plus.0, plus.1) {
                Some(value) => value,
                None => continue,
            };
            let minus_rgb = match sample_source_at(source, minus.0, minus.1) {
                Some(value) => value,
                None => continue,
            };
            let score = d * 2.0;
            if score < best_score {
                best_score = score;
                best = Some((d, plus_rgb, minus_rgb));
            }
            break;
        }
    }
    best
}

fn estimate_normal(mask: &[u8], width: usize, height: usize, x: usize, y: usize) -> (f32, f32) {
    if x == 0 || x + 1 >= width || y == 0 || y + 1 >= height {
        return (0.0, 0.0);
    }
    let top = (y - 1) * width + x;
    let center = y * width + x;
    let bottom = (y + 1) * width + x;
    let gx = (-(mask[top - 1] as f32) - 2.0 * mask[center - 1] as f32 - mask[bottom - 1] as f32)
        + (mask[top + 1] as f32 + 2.0 * mask[center + 1] as f32 + mask[bottom + 1] as f32);
    let gy = (-(mask[top - 1] as f32) - 2.0 * mask[top] as f32 - mask[top + 1] as f32)
        + (mask[bottom - 1] as f32 + 2.0 * mask[bottom] as f32 + mask[bottom + 1] as f32);
    let mut nx = gx;
    let mut ny = gy;
    normalize2(&mut nx, &mut ny);
    (nx, ny)
}

pub fn hair_seam_fill(
    source: &Image,
    part_line_mask: &[u8],
    hair_mask: &[u8],
    ui: f32,
    normals: Option<(&[f32], &[f32])>,
    max_probe_radius: Option<u32>,
) -> Result<HairSeamResult, Refusal> {
    let pixels = checked_pixels(source, "source")?;
    check_plane(part_line_mask.len(), pixels, "part_line_mask")?;
    check_plane(hair_mask.len(), pixels, "hair_mask")?;
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if let Some((nx, ny)) = normals {
        check_plane(nx.len(), pixels, "normal_x")?;
        check_plane(ny.len(), pixels, "normal_y")?;
        if !nx.iter().chain(ny.iter()).all(|value| value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("normals"));
        }
    }
    let degree = ui / 100.0;
    let mut output = source.clone();
    let half_x = 0.5 * (source.width as f32 - 1.0) * degree;
    let half_y = 0.5 * (source.height as f32 - 1.0) * degree;
    let default_radius = half_x.max(half_y);
    let probe_limit = max_probe_radius
        .filter(|value| *value > 0)
        .unwrap_or_else(|| default_radius.ceil().max(2.0) as u32);
    let mut part_line_pixels = 0u32;
    let mut candidate_pixels = 0u32;
    let mut full_weight_pixels = 0u32;
    let mut changed = 0u32;
    let mut alpha_changed = 0u32;
    if degree == 0.0 {
        return Ok(HairSeamResult {
            output,
            degree,
            half_x,
            half_y,
            probe_limit,
            part_line_pixels,
            candidate_pixels,
            full_weight_pixels,
            changed,
            alpha_changed,
        });
    }
    for y in 0..source.height as usize {
        for x in 0..source.width as usize {
            let pixel = y * source.width as usize + x;
            let mask_value = part_line_mask[pixel];
            if mask_value == 0 {
                continue;
            }
            part_line_pixels += 1;
            let seam = mask_value as f32 / 255.0;
            let weight = clamp01(degree * seam);
            if weight <= 0.0 {
                continue;
            }
            candidate_pixels += 1;
            if weight >= 1.0 {
                full_weight_pixels += 1;
            }
            let (mut nx, mut ny) = match normals {
                Some((normal_x, normal_y)) => (normal_x[pixel], normal_y[pixel]),
                None => estimate_normal(part_line_mask, source.width as usize, source.height as usize, x, y),
            };
            normalize2(&mut nx, &mut ny);
            let pair = if nx != 0.0 || ny != 0.0 {
                find_pair(source, hair_mask, x as f32, y as f32, nx, ny, probe_limit)
            } else {
                None
            };
            let pair = pair.or_else(|| find_best_angle(source, hair_mask, x as f32, y as f32, probe_limit));
            let Some((_, plus, minus)) = pair else {
                continue;
            };
            let base = pixel * 4;
            for channel in 0..3 {
                let repair = plus[channel] * 0.5 + minus[channel] * 0.5;
                let value = source.pixels[base + channel] as f32 * (1.0 - weight) + repair * weight;
                output.pixels[base + channel] = byte_away(value);
            }
            if (0..3).any(|channel| output.pixels[base + channel] != source.pixels[base + channel]) {
                changed += 1;
            }
            if output.pixels[base + 3] != source.pixels[base + 3] {
                alpha_changed += 1;
            }
        }
    }
    Ok(HairSeamResult {
        output,
        degree,
        half_x,
        half_y,
        probe_limit,
        part_line_pixels,
        candidate_pixels,
        full_weight_pixels,
        changed,
        alpha_changed,
    })
}

// ---------------------------------------------------------------------------
// Row 89 / PF232 skull top radial geometry.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkullDegree {
    pub active_part: i32,
    pub signed_degree: f32,
    pub magnitude: f32,
    pub status: &'static str,
}

#[derive(Debug)]
pub struct SkullMesh {
    pub anchor: [f32; 2],
    pub source: Vec<[f32; 2]>,
    pub warped: Vec<[f32; 2]>,
    pub normalized: Vec<[f32; 2]>,
    pub bounds: [f32; 4],
    pub roi_padded: [f32; 4],
    pub signed_degree: f32,
    pub status: &'static str,
}

fn skull_degree(ui: f32) -> SkullDegree {
    let p = (ui + 100.0) / 200.0;
    let signed = ui / 100.0;
    if p > 0.5 {
        SkullDegree {
            active_part: 0,
            signed_degree: signed,
            magnitude: (p - 0.5) * 2.0,
            status: "ACTIVE_PART0_EXPAND",
        }
    } else if p < 0.5 {
        SkullDegree {
            active_part: 1,
            signed_degree: signed,
            magnitude: (0.5 - p) * 2.0,
            status: "ACTIVE_PART1_SHRINK",
        }
    } else {
        SkullDegree { active_part: -1, signed_degree: 0.0, magnitude: 0.0, status: "IDENTITY_CENTER" }
    }
}

fn skull_radial_weight(index: usize) -> f32 {
    let mut t = ((index as i32 - 7).abs() + 7) as f32 / 7.0;
    if t > 0.5 {
        t = (t - 0.5).abs() * 0.5 + 0.5;
    }
    t
}

pub fn skull_top(
    ui: f32,
    landmarks: &[[f32; 2]],
    factors: &[f32],
    width: f32,
    height: f32,
) -> Result<SkullMesh, Refusal> {
    if !ui.is_finite() || !(-100.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if landmarks.len() < 207 || factors.len() < 15 {
        return Err(Refusal::MissingInput("skull_landmarks_or_factors"));
    }
    if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
        return Err(Refusal::ValueOutOfRange("frame"));
    }
    if !landmarks.iter().all(|p| p[0].is_finite() && p[1].is_finite())
        || !factors[..15].iter().all(|value| value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("skull_inputs"));
    }
    let degree = skull_degree(ui);
    let anchor = landmarks[192];
    let mut source = Vec::with_capacity(15);
    let mut warped = Vec::with_capacity(15);
    let mut normalized = Vec::with_capacity(15);
    for i in 0..15 {
        let point = landmarks[192 + i];
        source.push(point);
        let k = degree.signed_degree * skull_radial_weight(i) * factors[i];
        let b = [point[0] + (point[0] - anchor[0]) * k, point[1] + (point[1] - anchor[1]) * k];
        warped.push(b);
        normalized.push([b[0] / width, b[1] / height]);
    }
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for point in &warped {
        min_x = min_x.min(point[0]);
        min_y = min_y.min(point[1]);
        max_x = max_x.max(point[0]);
        max_y = max_y.max(point[1]);
    }
    let bounds = [min_x, min_y, max_x - min_x, max_y - min_y];
    let roi_padded = [bounds[0] - 5.0, bounds[1] - 5.0, bounds[2] + 10.0, bounds[3] + 10.0];
    Ok(SkullMesh {
        anchor,
        source,
        warped,
        normalized,
        bounds,
        roi_padded,
        signed_degree: degree.signed_degree,
        status: degree.status,
    })
}

// ---------------------------------------------------------------------------
// Row 90 / PF202 hair line, Type1007 -> Type1012 136-point / 705-index geometry.
// ---------------------------------------------------------------------------

const TYPE1012_WIDTH: usize = 27;
const TYPE1012_HEIGHT: usize = 5;
const TYPE1012_VERTICES: usize = 136;
const TYPE1012_SLOTS: usize = 27;

pub const TYPE1012_INDICES: [u16; 705] = [
    105, 131, 132, 105, 104, 131, 24, 104, 105, 24, 23, 104, 51, 23, 24, 51,
    50, 23, 78, 50, 51, 78, 77, 50, 135, 77, 78, 135, 76, 77, 135, 75,
    76, 135, 74, 75, 135, 73, 74, 135, 72, 73, 135, 71, 72, 135, 70, 71,
    135, 69, 70, 135, 68, 69, 135, 67, 68, 66, 67, 135, 66, 40, 67, 66,
    39, 40, 65, 39, 66, 65, 64, 39, 64, 65, 135, 63, 64, 135, 64, 63,
    38, 64, 38, 39, 38, 12, 39, 11, 12, 38, 11, 93, 12, 11, 92, 93,
    10, 92, 11, 10, 91, 92, 9, 91, 10, 90, 91, 9, 90, 118, 91, 90,
    117, 118, 89, 117, 90, 116, 117, 89, 80, 52, 53, 80, 51, 52, 80, 79,
    51, 135, 79, 80, 79, 135, 78, 79, 78, 51, 108, 109, 81, 109, 110, 83,
    109, 83, 82, 109, 82, 81, 81, 82, 0, 82, 1, 0, 82, 2, 1, 82,
    83, 2, 83, 84, 2, 83, 111, 84, 110, 111, 83, 111, 112, 84, 84, 112,
    85, 84, 85, 3, 2, 84, 3, 2, 3, 29, 1, 2, 29, 1, 29, 28,
    1, 28, 27, 0, 1, 27, 27, 28, 54, 54, 28, 56, 54, 56, 55, 135,
    54, 55, 56, 30, 57, 29, 30, 56, 29, 3, 30, 3, 4, 30, 3, 85,
    4, 4, 85, 5, 4, 5, 32, 31, 4, 32, 30, 4, 31, 30, 31, 57,
    57, 31, 58, 135, 57, 58, 59, 135, 58, 60, 135, 59, 60, 61, 135, 61,
    60, 33, 34, 61, 33, 34, 62, 61, 35, 62, 34, 35, 63, 62, 63, 35,
    36, 63, 36, 37, 63, 37, 38, 37, 11, 38, 10, 11, 37, 37, 36, 10,
    36, 9, 10, 36, 35, 9, 35, 8, 9, 7, 8, 35, 7, 88, 8, 87,
    88, 7, 87, 114, 88, 113, 114, 87, 86, 113, 87, 112, 113, 86, 85, 112,
    86, 85, 86, 5, 5, 86, 6, 5, 6, 33, 32, 5, 33, 33, 60, 32,
    59, 32, 60, 31, 32, 59, 58, 31, 59, 33, 6, 34, 34, 6, 7, 35,
    34, 7, 6, 87, 7, 86, 87, 6, 114, 115, 88, 88, 115, 89, 88, 89,
    8, 89, 90, 8, 8, 90, 9, 115, 116, 89, 62, 63, 135, 61, 62, 135,
    28, 29, 56, 52, 51, 24, 52, 24, 25, 53, 52, 25, 53, 25, 26, 107,
    133, 134, 107, 106, 133, 26, 106, 107, 26, 25, 106, 25, 105, 106, 25, 24,
    105, 106, 105, 132, 106, 132, 133, 134, 133, 132, 91, 118, 119, 91, 119, 92,
    92, 119, 120, 92, 120, 93, 93, 120, 121, 93, 121, 94, 12, 93, 94, 12,
    94, 13, 39, 12, 13, 39, 13, 40, 40, 13, 41, 68, 40, 41, 67, 40,
    68, 69, 68, 41, 42, 69, 41, 70, 69, 42, 43, 70, 42, 44, 70, 43,
    45, 70, 44, 46, 70, 45, 46, 71, 70, 46, 72, 71, 46, 73, 72, 47,
    73, 46, 48, 73, 47, 74, 73, 48, 75, 74, 48, 49, 75, 48, 76, 75,
    49, 76, 49, 50, 77, 76, 50, 50, 49, 23, 49, 22, 23, 49, 48, 22,
    48, 21, 22, 48, 47, 21, 21, 47, 20, 21, 20, 102, 21, 102, 103, 22,
    21, 103, 22, 103, 104, 23, 22, 104, 104, 103, 131, 103, 130, 131, 103, 102,
    130, 102, 129, 130, 102, 101, 129, 20, 101, 102, 20, 19, 101, 20, 46, 19,
    47, 46, 20, 46, 45, 19, 19, 45, 18, 19, 18, 100, 19, 100, 101, 101,
    100, 128, 101, 128, 129, 100, 127, 128, 100, 99, 127, 18, 99, 100, 18, 17,
    99, 44, 17, 18, 17, 44, 16, 17, 16, 97, 98, 17, 97, 17, 98, 99,
    99, 98, 126, 99, 126, 127, 98, 125, 126, 125, 98, 124, 124, 97, 123, 98,
    97, 124, 97, 96, 123, 97, 16, 96, 16, 15, 96, 16, 43, 15, 44, 43,
    16, 43, 42, 15, 15, 42, 14, 15, 14, 95, 96, 15, 95, 96, 95, 122,
    123, 96, 122, 95, 121, 122, 94, 121, 95, 14, 94, 95, 13, 94, 14, 41,
    13, 14, 42, 41, 14, 45, 44, 18, 65, 66, 135, 135, 56, 55, 135, 57,
    56,
];

const TYPE1012_Y_WEIGHTS: [f32; 5] = [
    0.0,
    0.20000000298023224,
    0.6000000238418579,
    -0.20000004768371582,
    -0.40000003576278687,
];
const TYPE1012_SLOT_ORDER: [usize; 27] = [
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18,
    21, 20, 23, 22, 25, 24, 26,
];
const TYPE1012_BLOCK_OFFSETS: [usize; 14] = [
    0x398, 0x548, 0x538, 0x528, 0x518, 0x508, 0x4F8, 0x4E8, 0x4D8, 0x4C8, 0x4B8,
    0x4A8, 0x498, 0x490,
];

#[derive(Debug)]
pub struct HairLineMesh {
    pub source: Vec<[f32; 2]>,
    pub destination: Vec<[f32; 2]>,
    pub indices: &'static [u16; 705],
    pub maximum_degree: f32,
    pub slot0: f32,
    pub slot1: f32,
}

pub fn hair_line(
    landmark194: &[[f32; 2]],
    weights135: &[f32],
    ui: f32,
    symmetric_second_slot: bool,
    width: u32,
    height: u32,
) -> Result<HairLineMesh, Refusal> {
    if landmark194.len() != 194 {
        return Err(Refusal::MissingInput("landmark194"));
    }
    if weights135.len() != 135 {
        return Err(Refusal::MissingInput("weights135"));
    }
    if !ui.is_finite() || !(-100.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if width == 0 || height == 0 {
        return Err(Refusal::ValueOutOfRange("frame"));
    }
    let mut flat = Vec::with_capacity(388);
    for point in landmark194 {
        if !point[0].is_finite() || !point[1].is_finite() {
            return Err(Refusal::ValueOutOfRange("landmark194"));
        }
        flat.push(point[0]);
        flat.push(point[1]);
    }
    if !weights135.iter().all(|value| value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("weights135"));
    }
    let p = (ui + 100.0) / 200.0;
    let slot0 = 1.0 - 2.0 * p;
    let slot1 = if symmetric_second_slot { slot0 } else { 0.0 };

    let mut slots = vec![0.0f32; TYPE1012_SLOTS * 2];
    let mut output = 0;
    for offset in TYPE1012_BLOCK_OFFSETS {
        let point_count = if offset == 0x490 { 1 } else { 2 };
        for point in 0..point_count {
            let input = offset / 4 + point * 2;
            slots[output] = flat[input];
            slots[output + 1] = flat[input + 1];
            output += 2;
        }
    }
    let anchor_x = flat[0x1B0 / 4];
    let anchor_y = flat[0x1B4 / 4];
    let reference_x = flat[0x4F8 / 4];
    let reference_y = flat[0x4FC / 4];
    let degree0 = slot0 + slot0;
    let degree1 = slot1 + slot1;

    let mut a = vec![0.0f32; TYPE1012_VERTICES * 2];
    for x in 0..TYPE1012_WIDTH {
        let source_slot = TYPE1012_SLOT_ORDER[x];
        let sx = slots[source_slot * 2];
        let sy = slots[source_slot * 2 + 1];
        let dx = anchor_x - sx;
        let dy = anchor_y - sy;
        for y in 0..TYPE1012_HEIGHT {
            let weight = TYPE1012_Y_WEIGHTS[y];
            let at = (y * TYPE1012_WIDTH + x) * 2;
            a[at] = sx + weight * dx * 1.2;
            a[at + 1] = sy + weight * dy * 1.2;
        }
    }
    a[(TYPE1012_VERTICES - 1) * 2] = anchor_x;
    a[(TYPE1012_VERTICES - 1) * 2 + 1] = anchor_y;
    for i in 0..TYPE1012_VERTICES {
        a[i * 2] = a[i * 2].clamp(0.0, width as f32);
        a[i * 2 + 1] = a[i * 2 + 1].clamp(0.0, height as f32);
    }
    let mut b = a.clone();

    let face_x = anchor_x;
    let face_y = anchor_y;
    let min_x = reference_x;
    let min_y = reference_y;
    let x4 = face_y - min_y;
    let x5 = face_x - min_x;
    let norm = (x4 * x4 + x5 * x5).sqrt();
    if !norm.is_finite() || norm < 1e-6 {
        return Err(Refusal::ValueOutOfRange("landmark_anchor_reference"));
    }
    let inv_norm = 1.0 / norm;
    let default_dir_x = x5 * inv_norm;
    let default_dir_y = x4 * inv_norm;
    let stack4 = width as f32;
    let upper_x = stack4 - 20.0;
    let upper_y = -20.0 + height as f32;
    let double_sum = (degree1 + degree0) as f64;
    for i in 0..135 {
        let weight = weights135[i];
        if weight == 0.0 {
            continue;
        }
        let px = b[i * 2];
        let py = b[i * 2 + 1];
        let mut edge_x = if 20.0 >= px || px >= upper_x {
            (stack4 - px).abs().min(px.abs()) * 0.05
        } else {
            1.0
        };
        if py <= 20.0 || py >= upper_y {
            edge_x *= py.abs() * 0.05;
        }
        let (dir_x, dir_y) = if i != TYPE1012_WIDTH / 2 {
            let dx = face_x - px;
            let dy = face_y - py;
            let dir_norm = (dx * dx + dy * dy).sqrt();
            let dir_inv = 1.0 / dir_norm;
            (dx * dir_inv, dy * dir_inv)
        } else {
            (default_dir_x, default_dir_y)
        };
        let dx = face_x - px;
        let dy = face_y - py;
        let distance = (dx * dx + dy * dy).sqrt();
        let mut term = (weight * distance) as f64;
        let row = i % TYPE1012_WIDTH;
        let scalar = if row <= 12 {
            term *= 0.1;
            term *= degree0 as f64;
            term as f32
        } else {
            term *= 0.1;
            if row == 13 {
                term *= double_sum;
                term *= 0.5;
            } else {
                term *= degree1 as f64;
            }
            term as f32
        };
        b[i * 2] = px + edge_x * scalar * dir_x;
        b[i * 2 + 1] = py + edge_x * scalar * dir_y;
    }
    let source = a.chunks_exact(2).map(|pair| [pair[0], pair[1]]).collect();
    let destination = b.chunks_exact(2).map(|pair| [pair[0], pair[1]]).collect();
    Ok(HairLineMesh {
        source,
        destination,
        indices: &TYPE1012_INDICES,
        maximum_degree: degree0.abs().max(degree1.abs()),
        slot0,
        slot1,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image_from<F: Fn(u32, u32) -> [u8; 4]>(width: u32, height: u32, f: F) -> Image {
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&f(x, y));
            }
        }
        Image { width, height, pixels }
    }

    fn constant_lut(rgba: [u8; 3]) -> Vec<u8> {
        rgba.iter().copied().cycle().take(512 * 512 * 3).collect()
    }

    #[test]
    fn meta_is_sorted_unique_and_covers_rows() {
        for pair in META.windows(2) {
            assert!(pair[0].pf < pair[1].pf, "{} then {}", pair[0].pf, pair[1].pf);
        }
        // 8033/21076/100007 live in `smile_warp`; 30026 in `teeth_alignment`.
        for pf in [8, 9, 108, 202, 232, 90087, 100004, 100005, 21105] {
            assert!(META.iter().any(|row| row.pf == pf), "missing pf {pf}");
        }
        assert_eq!(MAKEUP_DEGREE_GROUP.len(), 14);
    }

    #[test]
    fn malformed_public_image_is_refused() {
        let bad = Image { width: 2, height: 2, pixels: vec![0; 15] };
        assert_eq!(
            body_skin_unify(&bad, 0.0, &[]).unwrap_err(),
            Refusal::ShapeMismatch("source")
        );
    }

    #[test]
    fn row71_zero_identity_and_exact_half_blend() {
        let source = Image::solid(1, 1, [100, 150, 200, 77]);
        let effect = Image::solid(1, 1, [200, 250, 50, 0]);
        let zero = body_skin_unify(&source, 0.0, &[]).unwrap();
        assert_eq!(zero.output, source);
        assert_eq!(zero.changed, 0);

        let result = body_skin_unify(
            &source,
            50.0,
            &[SkinUnifyPass { mask: &[128], effect: &effect }],
        )
        .unwrap();
        let t = 128.0f32 * INV255 * 0.5;
        let expected = [
            byte_trunc(100.0 * (1.0 - t) + 200.0 * t),
            byte_trunc(150.0 * (1.0 - t) + 250.0 * t),
            byte_trunc(200.0 * (1.0 - t) + 50.0 * t),
            77,
        ];
        assert_eq!(result.output.pixel(0, 0), expected);
        assert_eq!(result.changed, 1);
    }

    #[test]
    fn row71_requires_explicit_passes() {
        let source = Image::solid(1, 1, [1, 2, 3, 4]);
        assert_eq!(
            body_skin_unify(&source, 50.0, &[]).unwrap_err(),
            Refusal::MissingInput("post_model_compose_passes")
        );
        assert_eq!(
            skin_unify_route_degree(SkinUnifyRoute::Degree60, &[0.25], &[], 1.0, 0).unwrap(),
            0.25
        );
        assert_eq!(
            skin_unify_route_degree(SkinUnifyRoute::Degree58TimesScale, &[], &[0.5], 0.5, 0).unwrap(),
            0.25
        );
    }

    #[test]
    fn row72_identity_transfer_and_roi_gate() {
        let width = 12u32;
        let height = 12u32;
        let source: Vec<u8> = (0..(width * height) as usize)
            .flat_map(|i| {
                let value = ((i * 7) % 200) as u8;
                [value, value / 2, 255 - value]
            })
            .collect();
        let body = vec![255u8; (width * height) as usize];
        let roi = vec![255u8; (width * height) as usize];
        let (prepared, output) =
            body_skin_uniform(&source, width, height, &body, &roi, 100.0, 0, &[0.5]).unwrap();
        assert_eq!(prepared.roi, Rect { x: 0, y: 0, width: 12, height: 12 });
        assert!(prepared.samples > 0);
        assert_ne!(output, source);

        let (_, identity) =
            body_skin_uniform(&source, width, height, &body, &roi, 0.0, 0, &[]).unwrap();
        assert_eq!(identity, source);

        let small_rgb = vec![0u8; 8 * 8 * 3];
        let small = vec![255u8; 8 * 8];
        assert_eq!(
            body_skin_uniform(&small_rgb, 8, 8, &small, &small, 100.0, 0, &[0.5]).unwrap_err(),
            Refusal::ValueOutOfRange("body_roi_min_side")
        );
        assert_eq!(
            body_skin_uniform(&source, width, height, &body, &roi, 10.0, 0, &[]).unwrap_err(),
            Refusal::MissingInput("native_skin_color_uniform")
        );
    }

    #[test]
    fn row73_temperature_identity_and_direction() {
        let source = Image::solid(1, 1, [50, 60, 70, 200]);
        let cold = constant_lut([0, 0, 0]);
        let warm = constant_lut([255, 255, 255]);
        assert_eq!(skin_temperature(&source, &[255], &cold, &warm, 0.0).unwrap(), source);
        let warm_out = skin_temperature(&source, &[255], &cold, &warm, 100.0).unwrap();
        assert_eq!(warm_out.pixel(0, 0), [255, 255, 255, 200]);
        let cold_out = skin_temperature(&source, &[255], &cold, &warm, -100.0).unwrap();
        assert_eq!(cold_out.pixel(0, 0), [0, 0, 0, 200]);
        assert_eq!(
            skin_temperature(&source, &[], &cold, &warm, 50.0).unwrap_err(),
            Refusal::ShapeMismatch("skin_mask")
        );
    }

    #[test]
    fn row74_tint_green_and_magenta() {
        let source = Image::solid(1, 1, [100, 100, 100, 255]);
        let green = constant_lut([0, 255, 0]);
        let magenta = constant_lut([255, 0, 255]);
        assert_eq!(
            skin_tint(&source, &[255], &green, &magenta, -100.0).unwrap().pixel(0, 0),
            [0, 255, 0, 255]
        );
        assert_eq!(
            skin_tint(&source, &[255], &green, &magenta, 100.0).unwrap().pixel(0, 0),
            [255, 0, 255, 255]
        );
    }

    #[test]
    fn row75_grinning_gate_and_warp() {
        let source = image_from(8, 8, |x, y| [(x * 20 + y) as u8, 100, 150, 177]);
        let landmarks = [[2.0, 2.0], [5.0, 2.0], [5.0, 5.0], [2.0, 5.0]];
        let face = Rect { x: 0, y: 0, width: 8, height: 8 };
        let masks = vec![255u8; 64];
        let dx = vec![1.0f32; 64];
        let dy = vec![0.0f32; 64];
        let closed = smile_open(
            &source, "Other", 100.0, &landmarks, face, 0.0, &masks, &masks, &dx, &dy,
        )
        .unwrap();
        assert!(!closed.active);
        assert_eq!(closed.output, source);
        let open = smile_open(
            &source, "Grinning", 100.0, &landmarks, face, 0.0, &masks, &masks, &dx, &dy,
        )
        .unwrap();
        assert!(open.active);
        assert!(open.changed > 0);
        assert_eq!(open.output.pixel(0, 0)[3], 177);
    }

    #[test]
    fn row76_closed_smile_identity_and_landmark_alternative() {
        let source = image_from(16, 16, |x, y| [(x * 10) as u8, (y * 10) as u8, 90, 66]);
        let mut landmarks = [[8.0f32, 8.0]; 68];
        landmarks[48] = [5.0, 8.0];
        landmarks[51] = [8.0, 6.0];
        landmarks[54] = [11.0, 8.0];
        landmarks[57] = [8.0, 10.0];
        let zero = smile_closed_landmark_alternative(&source, &landmarks, 0.0).unwrap();
        assert_eq!(zero.output, source);
        assert!(zero.alternative);
        let active = smile_closed_landmark_alternative(&source, &landmarks, 100.0).unwrap();
        assert!(active.changed > 0);
        assert_eq!(active.output.pixel(0, 0)[3], 66);
        assert_eq!(
            smile_closed_landmark_alternative(&source, &landmarks[..67], 100.0).unwrap_err(),
            Refusal::MissingInput("landmarks68")
        );
    }

    #[test]
    fn row77_vertex_field_and_dense_warp() {
        let mean = vec![[0.0f32, 0.0, 0.0]; 2206];
        let reshape = vec![[1.0f32, 0.0, 0.0]; 2206];
        let projection = Affine3x4 {
            m00: 1.0, m01: 0.0, m02: 0.0, m03: 0.0,
            m10: 0.0, m11: 1.0, m12: 0.0, m13: 0.0,
        };
        let field = mouth_corner_vertex_field(100.0, &mean, &reshape, projection).unwrap();
        assert_eq!(field.len(), 2206);
        assert_eq!(field[0], [1.0, 0.0]);
        let source = image_from(2, 1, |x, _| [(10 + x * 90) as u8, 20, 30, 44]);
        let dx = vec![1.0f32; 2];
        let dy = vec![0.0f32; 2];
        let warped = mouth_corner_warp(&source, &dx, &dy, &[255, 255], 100.0).unwrap();
        assert_eq!(warped.pixel(0, 0), [100, 20, 30, 44]);
        assert_eq!(
            mouth_corner_warp(&source, &[1.0], &[0.0], &[255], 100.0).unwrap_err(),
            Refusal::ShapeMismatch("dx")
        );
        assert_eq!(
            mouth_corner_vertex_field(100.0, &mean[..2205], &reshape, projection).unwrap_err(),
            Refusal::MissingInput("mesh2206")
        );
    }

    #[test]
    fn row78_tooth_repair_full_weight_and_alpha() {
        let source = Image::solid(2, 2, [10, 20, 30, 77]);
        let repair = Image::solid(2, 2, [200, 210, 220, 0]);
        let roi = Rect { x: 0, y: 0, width: 2, height: 2 };
        let out = tooth_repair(&source, &repair, &[1.0; 4], roi, 100.0).unwrap();
        assert_eq!(out.pixel(0, 0), [200, 210, 220, 77]);
        let zero = tooth_repair(&source, &repair, &[1.0; 4], roi, 0.0).unwrap();
        assert_eq!(zero, source);
        assert_eq!(
            tooth_repair(&source, &repair, &[1.1; 4], roi, 100.0).unwrap_err(),
            Refusal::ValueOutOfRange("teeth_mask")
        );
    }

    #[test]
    fn row79_teeth_whitening_degree_coupling_and_lut() {
        let source = Image::solid(1, 1, [40, 50, 60, 90]);
        let white = constant_lut([255, 255, 255]);
        let out = teeth_whitening(&source, &[255], &white, &white, 100.0, 100.0, 100.0).unwrap();
        assert_eq!(out.dz, 1.0);
        assert_eq!(out.ea, 1.0);
        assert_eq!(out.combined_mask, vec![255, 255, 0, 255]);
        assert_eq!(out.output.pixel(0, 0), [255, 255, 255, 90]);
        let zero = teeth_whitening(&source, &[255], &white, &white, 0.0, 70.0, 70.0).unwrap();
        assert_eq!(zero.output, source);
        assert_eq!(zero.touched, 0);
        assert_eq!(
            teeth_whitening(&source, &[], &white, &white, 0.0, 0.0, 0.0).unwrap_err(),
            Refusal::ShapeMismatch("mask_g")
        );
    }

    #[test]
    fn row80_alignment_plan_and_mask_blend() {
        let contour = [[4.0f32, 4.0], [8.0, 4.0], [8.0, 8.0], [4.0, 8.0]];
        let tooth = ToothGeometry { id: 1, arch: ToothArch::Upper, source: &contour, target: None };
        let identity = teeth_alignment_plan(16, 16, 0, &[tooth], 1.5).unwrap();
        assert_eq!(identity.layer_count, 0);
        assert!(identity.mask.iter().all(|value| *value == 0));

        let tooth = ToothGeometry { id: 1, arch: ToothArch::Upper, source: &contour, target: None };
        let plan = teeth_alignment_plan(16, 16, 5, &[tooth], 1.5).unwrap();
        assert_eq!(plan.parameters.model_priority, 25.0);
        assert_eq!(plan.parameters.strength, 1.0);
        assert_eq!(plan.layer_count, 1);
        assert!(plan.mask.iter().any(|value| *value > 0));
        assert!(plan.delta_u.iter().any(|value| value.abs() > 0.0));

        let original = Image::solid(1, 1, [10, 20, 30, 44]);
        let effect = Image::solid(1, 1, [200, 210, 220, 0]);
        let blended = teeth_alignment_blend(&original, &effect, &[255]).unwrap();
        assert_eq!(blended.pixel(0, 0), [200, 210, 220, 44]);
    }

    #[test]
    fn row81_red_vein_zero_and_explicit_model_output() {
        let source = Image::solid(1, 1, [100, 120, 140, 55]);
        let effect = [200u8, 210, 220];
        assert_eq!(red_vein(&source, &effect, &[255], 0.0, 1.0).unwrap(), source);
        assert_eq!(
            red_vein(&source, &effect, &[255], 100.0, 1.0).unwrap().pixel(0, 0),
            [200, 210, 220, 55]
        );
        assert_eq!(
            red_vein(&source, &[0, 1], &[255], 100.0, 1.0).unwrap_err(),
            Refusal::ShapeMismatch("effect_rgb")
        );
    }

    #[test]
    fn row82_eye_white_identity_and_open_pixel_law() {
        let source = Image::solid(2, 2, [100, 110, 120, 130]);
        let roi = Rect { x: 0, y: 0, width: 2, height: 2 };
        assert_eq!(
            eye_white_purify(&source, roi, &[0; 4], &[0.1; 8], 0.0).unwrap(),
            source
        );
        assert_eq!(
            eye_white_purify(&source, roi, &[0; 4], &[0.1; 8], 50.0).unwrap_err(),
            Refusal::NotImplemented(100005)
        );
        assert_eq!(
            eye_white_split8(&[0.1; 7]).unwrap_err(),
            Refusal::ShapeMismatch("model_output8")
        );
        let split = eye_white_split8(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        assert_eq!(split.first4, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(split.second4, [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn row83_glare_knee_blend_and_zero_strength() {
        let source = Image::solid(1, 1, [0, 0, 0, 255]);
        let effect = [0.0f32, 0.0, 0.0];
        let mask = [0.0f32];
        assert_eq!(
            glasses_glare_removal(
                &source, &effect, Some(&mask), None, GlareMaskEncoding::SignedNetwork, 0
            )
            .unwrap(),
            source
        );
        let out = glasses_glare_removal(
            &source, &effect, Some(&mask), None, GlareMaskEncoding::SignedNetwork, 100,
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0), [128, 128, 128, 255]);
        assert_eq!(
            glasses_glare_removal(&source, &effect, None, None, GlareMaskEncoding::UnormLens, 100)
                .unwrap_err(),
            Refusal::MissingInput("network_mask_or_lens_mask")
        );
    }

    #[test]
    fn row84_highlight_threshold_and_soft_light() {
        let source = Image::solid(2, 2, [100, 100, 100, 255]);
        let material = Image::solid(2, 2, [200, 200, 200, 255]);
        let mask = Image::solid(2, 2, [255, 255, 255, 255]);
        let roi = Rect { x: 0, y: 0, width: 2, height: 2 };
        assert_eq!(
            stereo_highlight(&source, &material, &mask, roi, 0.0, 1.0, 1.0).unwrap(),
            source
        );
        let out = stereo_highlight(&source, &material, &mask, roi, 100.0, 1.0, 1.0).unwrap();
        assert_ne!(out.pixel(0, 0)[0], 100);
        assert_eq!(out.pixel(0, 0)[3], 255);
        let outside = Rect { x: 1, y: 1, width: 2, height: 2 };
        assert_eq!(
            stereo_highlight(&source, &material, &mask, outside, 100.0, 1.0, 1.0).unwrap_err(),
            Refusal::ValueOutOfRange("roi")
        );
    }

    #[test]
    fn row85_shadow_degree_and_mask_shape() {
        let source = Image::solid(2, 2, [100, 100, 100, 255]);
        let material = Image::solid(2, 2, [200, 200, 200, 255]);
        let mask = Image::solid(2, 2, [255, 255, 255, 255]);
        let roi = Rect { x: 0, y: 0, width: 2, height: 2 };
        assert_eq!(
            stereo_shadow(&source, &material, &mask, roi, 0.0, 1.0).unwrap(),
            source
        );
        let out = stereo_shadow(&source, &material, &mask, roi, 100.0, 1.0).unwrap();
        assert_ne!(out.pixel(0, 0)[0], 100);
        let wrong = Image::solid(1, 1, [255, 255, 255, 255]);
        assert_eq!(
            stereo_shadow(&source, &wrong, &mask, roi, 100.0, 1.0).unwrap_err(),
            Refusal::ShapeMismatch("material")
        );
    }

    #[test]
    fn row86_lip_neutral_positive_and_negative() {
        let source = Image::solid(1, 1, [50, 60, 70, 88]);
        let white = constant_lut([255, 255, 255]);
        let black = constant_lut([0, 0, 0]);
        assert!((lip_effect_magnitude(20.0) - 0.2).abs() < 1e-6);
        assert!((lip_effect_magnitude(50.0) - 0.5).abs() < 1e-6);
        assert_eq!(
            lip_makeup(&source, &[1.0], &white, &black, 0.0, 80.0, 1.0, 1.0).unwrap(),
            source
        );
        assert_eq!(
            lip_makeup(&source, &[1.0], &white, &black, 100.0, 100.0, 1.0, 1.0)
                .unwrap()
                .pixel(0, 0),
            [255, 255, 255, 88]
        );
        assert_eq!(
            lip_makeup(&source, &[1.0], &white, &black, -100.0, 100.0, 1.0, 1.0)
                .unwrap()
                .pixel(0, 0),
            [0, 0, 0, 88]
        );
        assert_eq!(
            lip_makeup(&source, &[], &white, &black, 0.0, 80.0, 1.0, 1.0).unwrap_err(),
            Refusal::ShapeMismatch("mask")
        );
    }

    #[test]
    fn row87_group_maps_every_pf() {
        for row in MAKEUP_DEGREE_GROUP {
            assert_eq!(makeup_degree(row.pf, 50.0).unwrap(), 0.5, "pf {}", row.pf);
            assert_eq!(makeup_degree(row.pf, 100.0).unwrap(), 1.0, "pf {}", row.pf);
        }
        assert_eq!(makeup_degree(999, 50.0).unwrap_err(), Refusal::MissingInput("makeup_degree_pf"));
        assert_eq!(makeup_degree(106, 101.0).unwrap_err(), Refusal::ValueOutOfRange("ui"));
    }

    #[test]
    fn row88_hair_seam_repair_and_zero_degree() {
        let source = image_from(5, 5, |x, y| [(20 + x * x * 10 + y * 3) as u8, 50, 60, 77]);
        let mut part_line = vec![0u8; 25];
        part_line[2 * 5 + 2] = 255;
        let mut hair = vec![0u8; 25];
        hair[2 * 5] = 255;
        hair[2 * 5 + 4] = 255;
        let zero = hair_seam_fill(&source, &part_line, &hair, 0.0, None, None).unwrap();
        assert_eq!(zero.output, source);
        let out = hair_seam_fill(&source, &part_line, &hair, 100.0, None, Some(4)).unwrap();
        assert_eq!(out.part_line_pixels, 1);
        assert!(out.changed >= 1);
        assert_eq!(out.alpha_changed, 0);
        assert_eq!(out.output.pixel(2, 2)[0], 106);
        assert_eq!(
            hair_seam_fill(&source, &[], &hair, 100.0, None, None).unwrap_err(),
            Refusal::ShapeMismatch("part_line_mask")
        );
    }

    #[test]
    fn row89_skull_center_identity_and_expand() {
        let landmarks: Vec<[f32; 2]> = (0..207)
            .map(|i| [i as f32, 10.0 + (i % 3) as f32 * 2.0])
            .collect();
        let factors = [1.0f32; 15];
        let center = skull_top(0.0, &landmarks, &factors, 400.0, 300.0).unwrap();
        assert_eq!(center.status, "IDENTITY_CENTER");
        assert_eq!(center.source, center.warped);
        let expand = skull_top(100.0, &landmarks, &factors, 400.0, 300.0).unwrap();
        assert_ne!(expand.source[7], expand.warped[7]);
        assert_eq!(expand.roi_padded[2], expand.bounds[2] + 10.0);
        assert_eq!(
            skull_top(100.0, &landmarks[..206], &factors, 400.0, 300.0).unwrap_err(),
            Refusal::MissingInput("skull_landmarks_or_factors")
        );
    }

    #[test]
    fn row90_hair_line_geometry_counts() {
        let landmarks: Vec<[f32; 2]> = (0..194)
            .map(|i| [100.0 + i as f32 * 2.0, 200.0 + (i % 7) as f32 * 3.0])
            .collect();
        let weights = [1.0f32; 135];
        let mesh = hair_line(&landmarks, &weights, 100.0, true, 1000, 1000).unwrap();
        assert_eq!(mesh.source.len(), TYPE1012_VERTICES);
        assert_eq!(mesh.destination.len(), TYPE1012_VERTICES);
        assert_eq!(mesh.indices.len(), 705);
        assert!((mesh.slot0 + 1.0).abs() < 1e-6);
        assert!((mesh.slot1 + 1.0).abs() < 1e-6);
        assert!(mesh.destination.iter().all(|point| point[0].is_finite() && point[1].is_finite()));
        assert_eq!(
            hair_line(&landmarks, &[1.0; 134], 0.0, false, 1000, 1000).unwrap_err(),
            Refusal::MissingInput("weights135")
        );
        assert_eq!(
            hair_line(&landmarks[..193], &weights, 0.0, false, 1000, 1000).unwrap_err(),
            Refusal::MissingInput("landmark194")
        );
    }
}
