//! Hair rows 91-95 and hand/body rows 96-101 pixel laws.
//!
//! Ported from the P25/P26 vendor-parse lanes. Every entry point consumes the
//! same-job model, mask and material surfaces explicitly: nothing is segmented,
//! detected or synthesised here. A missing or malformed input is a typed
//! refusal, never a fallback.
//!
//! Rows 93-95 live behind the vendor same-job dispatch, whose receipt checks
//! stay in the host. These kernels are reference maths only; calling one does
//! not verify a receipt or enable the row in the product.
//!
//! This module is compiled by the lane's standalone rustc harness. It is not
//! registered in `kernels::mod` by this lane.

use crate::{Access, EffectMeta, Image, Refusal};

/// Registry rows, `ParamFlag` order. Rows 91-101 are the hair/hand/body family.
pub const META: &[EffectMeta] = &[
    EffectMeta {
        pf: 405,
        key: "slimBody",
        label: "Body slim",
        access: Access::Cpu,
        inputs: &[
            "PF405_BODY_MASK_R",
            "PF405_SOURCE_UV",
            "PF405_TARGET_UV",
        ],
    },
    EffectMeta {
        pf: 30009,
        key: "hairColorUnify",
        label: "Hair colour unify",
        access: Access::Cpu,
        inputs: &["PF30009_MODEL_EFFECT_RGB", "PF30009_HAIR_MASK"],
    },
    EffectMeta {
        pf: 30030,
        key: "removeBlueVeinsSlider",
        label: "Hand blue veins",
        access: Access::Cpu,
        inputs: &[
            "PF30030_HAND_VEIN_MASK",
            "PF30030_MODEL_CORRECTION_RGB",
        ],
    },
    EffectMeta {
        pf: 30187,
        key: "Vl.handNeutralGray",
        label: "Hand neutral gray",
        access: Access::Cpu,
        inputs: &[
            "PF30187_HAND_MASK",
            "PF30187_MODEL_ALPHA",
            "PF30187_MODEL_NEUTRAL",
        ],
    },
    EffectMeta {
        pf: 30190,
        key: "Vl.handReshape",
        label: "Hand reshape",
        access: Access::Cpu,
        inputs: &[
            "PF30190_SHIFT_X",
            "PF30190_SHIFT_Y",
            "PF30190_HAND_MASK",
            "PF30190_PROTECT_MASK",
        ],
    },
    EffectMeta {
        pf: 44297,
        key: "Gn.removeBlueVeins",
        label: "Foot blue veins",
        access: Access::Cpu,
        inputs: &["PF44297_MODEL_MASK", "PF44297_MODEL_REPAIR"],
    },
    EffectMeta {
        pf: 90041,
        key: "aiBodyReshape",
        label: "AI body reshape",
        access: Access::Cpu,
        inputs: &["PF90041_BODY_MASK", "PF90041_SHIFT_MAP"],
    },
    EffectMeta {
        pf: 90088,
        key: "Ue.removeStrayHair",
        label: "Stray hair removal",
        access: Access::Cpu,
        inputs: &["PF90088_HAIRFLY_V1", "PF90088_RAW_STRAY_HAIR_MASK"],
    },
    EffectMeta {
        pf: 90131,
        key: "removeWhiteHair",
        label: "White hair removal",
        access: Access::Cpu,
        inputs: &[
            "PF90131_RAW_HAIR_MASK",
            "PF90131_REPAIR_IMAGE",
        ],
    },
    EffectMeta {
        pf: 90139,
        key: "hairSmooth",
        label: "Hair smooth",
        access: Access::Cpu,
        inputs: &[
            "PF90139_HAIR_MASK",
            "PF90139_BLEND_SOURCE",
            "PF90139_DETAIL",
        ],
    },
    EffectMeta {
        pf: 90160,
        key: "hairGlossySlider",
        label: "Hair gloss",
        access: Access::Cpu,
        inputs: &["PF90160_MODEL_DETAIL", "PF90160_HAIR_MASK"],
    },
];

fn pixels(image: &Image) -> Result<usize, Refusal> {
    if image.width == 0 || image.height == 0 {
        return Err(Refusal::ShapeMismatch("image"));
    }
    let expected = (image.width as usize)
        .checked_mul(image.height as usize)
        .and_then(|value| value.checked_mul(4))
        .ok_or(Refusal::ShapeMismatch("image"))?;
    if image.pixels.len() != expected {
        return Err(Refusal::ShapeMismatch("image"));
    }
    Ok(expected / 4)
}

fn plane(values: &[u8], count: usize, name: &'static str) -> Result<(), Refusal> {
    if values.len() != count {
        Err(Refusal::ShapeMismatch(name))
    } else {
        Ok(())
    }
}

fn same_shape(a: &Image, b: &Image, name: &'static str) -> Result<usize, Refusal> {
    let count = pixels(a)?;
    if a.width != b.width || a.height != b.height {
        return Err(Refusal::ShapeMismatch(name));
    }
    pixels(b)?;
    Ok(count)
}

fn percent(value: f32, name: &'static str) -> Result<f32, Refusal> {
    if !value.is_finite() || !(0.0..=100.0).contains(&value) {
        Err(Refusal::ValueOutOfRange(name))
    } else {
        Ok(value)
    }
}

fn unit(value: f32, name: &'static str) -> Result<f32, Refusal> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        Err(Refusal::ValueOutOfRange(name))
    } else {
        Ok(value)
    }
}

/// Round half away from zero, then clamp to a byte. Matches the vendor
/// `MathF.Round(value, MidpointRounding.AwayFromZero)` on 0..255 values.
fn to_byte(value: f32) -> u8 {
    value.round().clamp(0.0, 255.0) as u8
}

/// Truncate toward zero, then clamp. Matches the vendor `(int)value` paths.
fn trunc_byte(value: f32) -> u8 {
    (value as i32).clamp(0, 255) as u8
}

fn roi_valid(roi: (u32, u32, u32, u32), width: u32, height: u32) -> bool {
    let (x, y, w, h) = roi;
    w > 0
        && h > 0
        && x.checked_add(w).map_or(false, |right| right <= width)
        && y.checked_add(h).map_or(false, |bottom| bottom <= height)
}

/// Separable box feather with rounded integer means; radius 0 is an exact copy.
fn feather_box(mask: &[u8], width: usize, height: usize, radius: u32) -> Vec<u8> {
    if radius == 0 {
        return mask.to_vec();
    }
    let radius = radius as usize;
    let mut horizontal = vec![0u8; mask.len()];
    for y in 0..height {
        let row = y * width;
        for x in 0..width {
            let start = x.saturating_sub(radius);
            let end = (x + radius).min(width - 1);
            let mut sum = 0u32;
            for sample in start..=end {
                sum += mask[row + sample] as u32;
            }
            let count = (end - start + 1) as u32;
            horizontal[row + x] = ((sum + count / 2) / count) as u8;
        }
    }
    let mut output = vec![0u8; mask.len()];
    for y in 0..height {
        let start = y.saturating_sub(radius);
        let end = (y + radius).min(height - 1);
        let count = (end - start + 1) as u32;
        for x in 0..width {
            let mut sum = 0u32;
            for sample_y in start..=end {
                sum += horizontal[sample_y * width + x] as u32;
            }
            output[y * width + x] = ((sum + count / 2) / count) as u8;
        }
    }
    output
}

/// ---- row 91: PF90088 stray hair -------------------------------------------

/// `Ue.removeStrayHair`. `ui` is 0..100; the native degree/person/background
/// values come from the live per-face maps. Missing area masks mean the
/// PF90113/PF90114 children are at their all-area default, as in the vendor
/// adapter; a missing repair image or raw mask is always a refusal.
pub fn stray_hair(
    source: &Image,
    ui: f32,
    native_degree: f32,
    native_person_area: f32,
    native_background_area: f32,
    repair: Option<&Image>,
    raw_mask: Option<&[u8]>,
    person_mask: Option<&[u8]>,
    background_mask: Option<&[u8]>,
) -> Result<Image, Refusal> {
    let count = pixels(source)?;
    if percent(ui, "PF90088_UI")? == 0.0 {
        return Ok(source.clone());
    }
    let native_degree = unit(native_degree, "PF90088_NATIVE_DEGREE")?;
    let person_area = unit(native_person_area, "PF90088_NATIVE_PERSON_AREA")?;
    let background_area = unit(native_background_area, "PF90088_NATIVE_BACKGROUND_AREA")?;
    if native_degree <= 0.0 {
        return Ok(source.clone());
    }

    let repair = repair.ok_or(Refusal::MissingInput("PF90088_HAIRFLY_V1"))?;
    let raw_mask = raw_mask.ok_or(Refusal::MissingInput("PF90088_RAW_STRAY_HAIR_MASK"))?;
    same_shape(source, repair, "PF90088_HAIRFLY_V1")?;
    plane(raw_mask, count, "PF90088_RAW_STRAY_HAIR_MASK")?;
    if let Some(mask) = person_mask {
        plane(mask, count, "PF90088_PERSON_MASK")?;
    }
    if let Some(mask) = background_mask {
        plane(mask, count, "PF90088_BACKGROUND_MASK")?;
    }
    if raw_mask.iter().all(|&value| value == 0) {
        return Ok(source.clone());
    }

    let mut out = source.clone();
    for index in 0..count {
        let weight = raw_mask[index] as f32 / 255.0;
        if weight <= 0.0 {
            continue;
        }
        let person = person_mask.map_or(1.0, |mask| mask[index] as f32 / 255.0);
        let background = background_mask.map_or(1.0, |mask| mask[index] as f32 / 255.0);
        let region = (person * person_area).max(background * background_area);
        let alpha = (weight * native_degree * region).clamp(0.0, 1.0);
        if alpha <= 0.0 {
            continue;
        }
        let offset = index * 4;
        for channel in 0..3 {
            let src = source.pixels[offset + channel] as f32;
            let dst = repair.pixels[offset + channel] as f32;
            out.pixels[offset + channel] = to_byte(src + (dst - src) * alpha);
        }
        out.pixels[offset + 3] = source.pixels[offset + 3];
    }
    Ok(out)
}

/// ---- row 92: PF90139 hair smooth ------------------------------------------

/// Scalar blend recovered at `0x46E3F40`. Alpha is never written.
///
/// `hair_mask` gates the whole pixel, `mix_mask` defaults to the hair mask,
/// `blend_source` is the smoothed/model result and `detail` is the centred
/// 8-bit detail plane. The vendor Otsu/box fallback is intentionally absent.
pub fn hair_smooth(
    source: &Image,
    ui_strength: f32,
    ui_strand: f32,
    hair_mask: Option<&[u8]>,
    mix_mask: Option<&[u8]>,
    blend_source: Option<&Image>,
    detail: Option<&[u8]>,
) -> Result<Image, Refusal> {
    let count = pixels(source)?;
    let mix_strength = percent(ui_strength, "PF90139_UI")? / 100.0;
    let detail_strength = percent(ui_strand, "PF90140_UI")? / 100.0;
    if mix_strength <= 0.0 {
        return Ok(source.clone());
    }

    let hair_mask = hair_mask.ok_or(Refusal::MissingInput("PF90139_HAIR_MASK"))?;
    let blend_source = blend_source.ok_or(Refusal::MissingInput("PF90139_BLEND_SOURCE"))?;
    let detail = detail.ok_or(Refusal::MissingInput("PF90139_DETAIL"))?;
    plane(hair_mask, count, "PF90139_HAIR_MASK")?;
    same_shape(source, blend_source, "PF90139_BLEND_SOURCE")?;
    plane(detail, count, "PF90139_DETAIL")?;
    let resolved_mix = match mix_mask {
        Some(mask) => {
            plane(mask, count, "PF90139_MIX_MASK")?;
            mask
        }
        None => hair_mask,
    };

    let mut out = source.clone();
    for index in 0..count {
        let gate_byte = hair_mask[index];
        if gate_byte == 0 {
            continue;
        }
        let gate = gate_byte as f32 / 255.0;
        let mask = resolved_mix[index] as f32 / 255.0;
        let mix_weight = mask * mix_strength * gate;
        let detail_add = (detail[index] as f32 - 127.5) * 4.0 * (gate * detail_strength * 0.8);
        let offset = index * 4;
        for channel in 0..3 {
            let base = source.pixels[offset + channel] as f32;
            let mixed = if mix_weight > 0.0 {
                base * (1.0 - mix_weight)
                    + blend_source.pixels[offset + channel] as f32 * mix_weight
            } else {
                base
            };
            out.pixels[offset + channel] = to_byte(mixed + detail_add);
        }
    }
    Ok(out)
}

/// ---- row 93: PF90160 hair gloss -------------------------------------------

/// Additive RGB delta recovered at `0x46E4870`. UI step 1 is exact identity.
pub fn hair_gloss(
    source: &Image,
    ui_step: i32,
    model_detail: &[u8],
    hair_mask: &[u8],
    roi: (u32, u32, u32, u32),
) -> Result<Image, Refusal> {
    let count = pixels(source)?;
    if !(1..=10).contains(&ui_step) {
        return Err(Refusal::ValueOutOfRange("PF90160_UI_STEP"));
    }
    if !roi_valid(roi, source.width, source.height) {
        return Err(Refusal::ShapeMismatch("PF90160_ROI"));
    }
    plane(model_detail, count, "PF90160_MODEL_DETAIL")?;
    plane(hair_mask, count, "PF90160_HAIR_MASK")?;

    let degree = (ui_step - 1) as f32 / 9.0;
    let mut out = source.clone();
    if degree == 0.0 {
        return Ok(out);
    }

    let (roi_x, roi_y, roi_width, roi_height) = roi;
    for y in roi_y..roi_y + roi_height {
        for x in roi_x..roi_x + roi_width {
            let index = (y * source.width + x) as usize;
            let detail = model_detail[index] as f32;
            let mask = hair_mask[index] as f32 / 255.0;
            let delta = ((detail - 127.5) * 2.0) * mask * degree;
            let offset = index * 4;
            for channel in 0..3 {
                out.pixels[offset + channel] =
                    trunc_byte(source.pixels[offset + channel] as f32 + delta);
            }
            out.pixels[offset + 3] = source.pixels[offset + 3];
        }
    }
    Ok(out)
}

/// ---- row 94: PF90131 white hair -------------------------------------------

/// Vendor raw mask direction: `0` is full repair, `255` is identity.
/// Composition order follows `0x697CF90` / `0x697D575`.
pub fn white_hair(
    source: &Image,
    ui: f32,
    raw_mask: &[u8],
    repair: &Image,
) -> Result<Image, Refusal> {
    let count = pixels(source)?;
    let degree = percent(ui, "PF90131_UI")? / 100.0;
    if degree <= 0.001 {
        return Ok(source.clone());
    }
    plane(raw_mask, count, "PF90131_RAW_HAIR_MASK")?;
    same_shape(source, repair, "PF90131_REPAIR_IMAGE")?;

    let mut out = source.clone();
    for index in 0..count {
        let complement = raw_mask[index] as f32 / -255.0 + 1.0;
        let mask_degree = complement * degree;
        let repair_alpha = repair.pixels[index * 4 + 3] as f32 / 255.0;
        let alpha = repair_alpha * mask_degree;
        if alpha == 0.0 {
            continue;
        }
        let offset = index * 4;
        let inverse = 1.0 - alpha;
        for channel in 0..3 {
            let value = repair.pixels[offset + channel] as f32 * alpha
                + source.pixels[offset + channel] as f32 * inverse;
            out.pixels[offset + channel] = trunc_byte(value);
        }
        let alpha_value =
            source.pixels[offset + 3] as f32 + repair.pixels[offset + 3] as f32 * alpha;
        out.pixels[offset + 3] = trunc_byte(alpha_value);
    }
    Ok(out)
}

/// ---- row 95: PF30009 hair colour unify ------------------------------------

/// `engine=(ui+100)/200`, `signed=1-2*engine`, `amount=|signed|`, active above
/// 0.001. `effect_rgb` is packed RGB8, `hair_mask` is one 0..1 float per pixel.
/// The float blend is quantised with round-half-away at the 8-bit boundary.
pub fn hair_color_unify(
    source: &Image,
    ui: f32,
    effect_rgb: &[u8],
    hair_mask: &[f32],
) -> Result<Image, Refusal> {
    let count = pixels(source)?;
    if !ui.is_finite() || !(-100.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("PF30009_UI"));
    }
    if effect_rgb.len() != count * 3 {
        return Err(Refusal::ShapeMismatch("PF30009_MODEL_EFFECT_RGB"));
    }
    if hair_mask.len() != count {
        return Err(Refusal::ShapeMismatch("PF30009_HAIR_MASK"));
    }
    if hair_mask
        .iter()
        .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
    {
        return Err(Refusal::ValueOutOfRange("PF30009_HAIR_MASK"));
    }

    let engine = (ui as f64 + 100.0) / 200.0;
    let signed = 1.0 - 2.0 * engine;
    let amount = signed.abs() as f32;
    let mut out = source.clone();
    if amount <= 0.001 {
        return Ok(out);
    }

    for index in 0..count {
        let q = hair_mask[index] * amount;
        let offset = index * 4;
        for channel in 0..3 {
            let src = source.pixels[offset + channel] as f32;
            let effect = effect_rgb[index * 3 + channel] as f32;
            out.pixels[offset + channel] = to_byte(src + q * (effect - src));
        }
    }
    Ok(out)
}

/// ---- row 96: PF30187 hand neutral gray ------------------------------------

/// Explicit model neutral surface: either the model's neutral BGRA or its
/// normalised BGR residual.
pub enum NeutralTarget<'a> {
    Bgra(&'a Image),
    Diff(&'a [f32]),
}

/// `weight = mask/255 * alpha/255 * degree`; RGB is blended, alpha is copied.
pub fn hand_neutral_gray(
    source: &Image,
    ui: f32,
    hand_mask: &[u8],
    model_alpha: &[u8],
    target: NeutralTarget<'_>,
) -> Result<Image, Refusal> {
    let count = pixels(source)?;
    let degree = percent(ui, "PF30187_UI")? / 100.0;
    if degree <= 0.0 {
        return Ok(source.clone());
    }
    plane(hand_mask, count, "PF30187_HAND_MASK")?;
    plane(model_alpha, count, "PF30187_MODEL_ALPHA")?;

    let neutral: Vec<u8> = match target {
        NeutralTarget::Bgra(image) => {
            same_shape(source, image, "PF30187_MODEL_NEUTRAL")?;
            image.pixels.clone()
        }
        NeutralTarget::Diff(diff) => {
            if diff.len() != count * 3 {
                return Err(Refusal::ShapeMismatch("PF30187_MODEL_DIFF"));
            }
            let mut built = vec![0u8; count * 4];
            for index in 0..count {
                let offset = index * 4;
                for channel in 0..3 {
                    let delta = diff[index * 3 + channel];
                    if !delta.is_finite() {
                        return Err(Refusal::ValueOutOfRange("PF30187_MODEL_DIFF"));
                    }
                    built[offset + channel] =
                        to_byte(source.pixels[offset + channel] as f32 + delta * 255.0);
                }
                built[offset + 3] = source.pixels[offset + 3];
            }
            built
        }
    };

    let mut out = source.clone();
    for index in 0..count {
        let weight =
            hand_mask[index] as f32 / 255.0 * (model_alpha[index] as f32 / 255.0) * degree;
        if weight <= 0.0 {
            continue;
        }
        let offset = index * 4;
        for channel in 0..3 {
            let src = source.pixels[offset + channel] as f32;
            let dst = neutral[offset + channel] as f32;
            out.pixels[offset + channel] = to_byte(src * (1.0 - weight) + dst * weight);
        }
        out.pixels[offset + 3] = source.pixels[offset + 3];
    }
    Ok(out)
}

/// ---- row 97: PF30030 hand blue veins --------------------------------------

fn round_div_255(value: i32, mask: u8) -> i32 {
    let product = value * mask as i32;
    if product >= 0 {
        (product + 127) / 255
    } else {
        -((-product + 127) / 255)
    }
}

/// Correction RGB is 8-bit with neutral 80; the source is BGRA and alpha is
/// untouched. `feather_radius` 0 leaves the mask unchanged.
pub fn hand_blue_vein(
    source: &Image,
    ui: i32,
    vein_mask: &[u8],
    correction_rgb: &[u8],
    feather_radius: u32,
    roi: (u32, u32, u32, u32),
) -> Result<Image, Refusal> {
    let count = pixels(source)?;
    if !(0..=100).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("PF30030_UI"));
    }
    plane(vein_mask, count, "PF30030_HAND_VEIN_MASK")?;
    if correction_rgb.len() != count * 3 {
        return Err(Refusal::ShapeMismatch("PF30030_MODEL_CORRECTION_RGB"));
    }
    if !roi_valid(roi, source.width, source.height) {
        return Err(Refusal::ShapeMismatch("PF30030_ROI"));
    }

    let mut out = source.clone();
    if ui == 0 {
        return Ok(out);
    }
    let feathered = feather_box(vein_mask, source.width as usize, source.height as usize, feather_radius);

    let (roi_x, roi_y, roi_width, roi_height) = roi;
    for y in roi_y..roi_y + roi_height {
        for x in roi_x..roi_x + roi_width {
            let index = (y * source.width + x) as usize;
            let mask = feathered[index];
            if mask == 0 {
                continue;
            }
            let offset = index * 4;
            for channel in 0..3 {
                let scaled = round_div_255(correction_rgb[index * 3 + channel] as i32 - 80, mask);
                out.pixels[offset + channel] =
                    (source.pixels[offset + channel] as i32 + scaled).clamp(0, 255) as u8;
            }
        }
    }
    Ok(out)
}

/// ---- row 98: PF30190 hand reshape -----------------------------------------

pub fn hand_shape_source_scale(width: u32, height: u32) -> Result<f32, Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("image"));
    }
    let max_dimension = width.max(height);
    Ok(if max_dimension < 0x800 {
        1.0
    } else {
        max_dimension as f32 / 2048.0
    })
}

fn sample_bgra_px(image: &Image, px: f32, py: f32) -> [u8; 4] {
    let max_x = (image.width - 1) as f32;
    let max_y = (image.height - 1) as f32;
    let px = px.clamp(0.0, max_x);
    let py = py.clamp(0.0, max_y);
    let x0 = px.floor() as u32;
    let y0 = py.floor() as u32;
    let x1 = (x0 + 1).min(image.width - 1);
    let y1 = (y0 + 1).min(image.height - 1);
    let tx = px - x0 as f32;
    let ty = py - y0 as f32;
    let mut out = [0u8; 4];
    for channel in 0..4usize {
        let top_left = image.pixels[((y0 * image.width + x0) * 4) as usize + channel] as f32;
        let top_right = image.pixels[((y0 * image.width + x1) * 4) as usize + channel] as f32;
        let bottom_left = image.pixels[((y1 * image.width + x0) * 4) as usize + channel] as f32;
        let bottom_right = image.pixels[((y1 * image.width + x1) * 4) as usize + channel] as f32;
        let top = top_left + (top_right - top_left) * tx;
        let bottom = bottom_left + (bottom_right - bottom_left) * tx;
        out[channel] = to_byte(top + (bottom - top) * ty);
    }
    out
}

/// Shift map `pixel + shift * degree * sourceScale`, min-clamped, then a
/// bilinear remap gated by the hand mask. Alpha is taken from the destination
/// pixel, matching the recovered 4-channel remap path.
pub fn hand_shape(
    source: &Image,
    native_parameter: i32,
    shift_x: &[f32],
    shift_y: &[f32],
    hand_mask: &[u8],
    protect_mask: &[u8],
    source_scale: Option<f32>,
) -> Result<Image, Refusal> {
    let count = pixels(source)?;
    if !(0..=100).contains(&native_parameter) {
        return Err(Refusal::ValueOutOfRange("PF30190_NATIVE_PARAMETER"));
    }
    let degree = native_parameter as f32 * 1.2 / 100.0;
    if degree == 0.0 {
        return Ok(source.clone());
    }
    if shift_x.len() != count || shift_y.len() != count {
        return Err(Refusal::ShapeMismatch("PF30190_SHIFT"));
    }
    if shift_x.iter().chain(shift_y).any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("PF30190_SHIFT"));
    }
    plane(hand_mask, count, "PF30190_HAND_MASK")?;
    plane(protect_mask, count, "PF30190_PROTECT_MASK")?;
    let scale = match source_scale {
        Some(value) if value.is_finite() && value > 0.0 => value,
        Some(_) => return Err(Refusal::ValueOutOfRange("PF30190_SOURCE_SCALE")),
        None => hand_shape_source_scale(source.width, source.height)?,
    };

    let mut out = source.clone();
    for index in 0..count {
        if hand_mask[index] == 0 {
            continue;
        }
        let factor = 1.0 - protect_mask[index] as f32 / 255.0;
        let x = (index % source.width as usize) as f32;
        let y = (index / source.width as usize) as f32;
        let map_x = (x + shift_x[index] * factor * degree * scale)
            .min((source.width - 1) as f32)
            .max(0.0);
        let map_y = (y + shift_y[index] * factor * degree * scale)
            .min((source.height - 1) as f32)
            .max(0.0);
        let sampled = sample_bgra_px(source, map_x, map_y);
        let offset = index * 4;
        out.pixels[offset] = sampled[0];
        out.pixels[offset + 1] = sampled[1];
        out.pixels[offset + 2] = sampled[2];
        out.pixels[offset + 3] = source.pixels[offset + 3];
    }
    Ok(out)
}

/// ---- row 99: PF44297 foot blue veins --------------------------------------

/// Chamfer distance transform used by the vendor feather pass.
fn distance_transform(foreground: &[bool], width: usize, height: usize, target: bool) -> Vec<f32> {
    const DIAGONAL: f32 = 1.414_213_6;
    let mut distance = vec![0.0f32; foreground.len()];
    for (index, &is_foreground) in foreground.iter().enumerate() {
        distance[index] = if is_foreground == target {
            0.0
        } else {
            f32::INFINITY
        };
    }

    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let mut value = distance[index];
            if x > 0 {
                value = value.min(distance[index - 1] + 1.0);
            }
            if y > 0 {
                value = value.min(distance[index - width] + 1.0);
                if x > 0 {
                    value = value.min(distance[index - width - 1] + DIAGONAL);
                }
                if x + 1 < width {
                    value = value.min(distance[index - width + 1] + DIAGONAL);
                }
            }
            distance[index] = value;
        }
    }

    for y in (0..height).rev() {
        for x in (0..width).rev() {
            let index = y * width + x;
            let mut value = distance[index];
            if x + 1 < width {
                value = value.min(distance[index + 1] + 1.0);
            }
            if y + 1 < height {
                value = value.min(distance[index + width] + 1.0);
                if x + 1 < width {
                    value = value.min(distance[index + width + 1] + DIAGONAL);
                }
                if x > 0 {
                    value = value.min(distance[index + width - 1] + DIAGONAL);
                }
            }
            distance[index] = value;
        }
    }
    distance
}

fn feather_distance(mask: &[u8], width: usize, height: usize, radius: u32) -> Vec<u8> {
    let foreground: Vec<bool> = mask.iter().map(|&value| value >= 128).collect();
    let to_background = distance_transform(&foreground, width, height, false);
    let to_foreground = distance_transform(&foreground, width, height, true);
    let inverse_span = 1.0 / (2.0 * radius as f32);
    let mut result = vec![0u8; mask.len()];
    for index in 0..mask.len() {
        let signed_distance = if foreground[index] {
            to_background[index] - 0.5
        } else {
            -(to_foreground[index] - 0.5)
        };
        let feathered = (0.5 + signed_distance * inverse_span).clamp(0.0, 1.0);
        let original = mask[index] as f32 / 255.0;
        result[index] = to_byte(original.max(feathered) * 255.0);
    }
    result
}

/// Explicit foot vein mask + model repair, feathered, then RGB-composited at
/// `mask/255 * degree`. Alpha is preserved from the source.
pub fn foot_blue_vein(
    source: &Image,
    ui: f32,
    model_mask: &[u8],
    model_repair: &Image,
    feather_radius: u32,
) -> Result<Image, Refusal> {
    let count = pixels(source)?;
    let degree = percent(ui, "PF44297_UI")? / 100.0;
    if degree <= 0.0 {
        return Ok(source.clone());
    }
    plane(model_mask, count, "PF44297_MODEL_MASK")?;
    same_shape(source, model_repair, "PF44297_MODEL_REPAIR")?;
    if feather_radius < 1 {
        return Err(Refusal::ValueOutOfRange("PF44297_FEATHER_RADIUS"));
    }

    let feathered = feather_distance(
        model_mask,
        source.width as usize,
        source.height as usize,
        feather_radius,
    );
    let mut out = source.clone();
    for index in 0..count {
        let weight = feathered[index] as f32 / 255.0 * degree;
        if weight <= 0.0 {
            continue;
        }
        let offset = index * 4;
        for channel in 0..3 {
            let src = source.pixels[offset + channel] as f32;
            let dst = model_repair.pixels[offset + channel] as f32;
            out.pixels[offset + channel] = to_byte(src * (1.0 - weight) + dst * weight);
        }
        out.pixels[offset + 3] = source.pixels[offset + 3];
    }
    Ok(out)
}

/// ---- row 100: PF90041 AI body reshape -------------------------------------

/// Packed RGBA8 shift texture: `((r*256+g)-32768)/ratio/shiftWidth` for X and
/// `((b*256+a)-32768)/ratio/shiftHeight` for Y.
pub struct PackedShiftMap<'a> {
    pub width: u32,
    pub height: u32,
    pub rgba8: &'a [u8],
    pub ratio: f32,
    pub shift_width: u32,
    pub shift_height: u32,
}

fn sample_plane_uv(plane: &[u8], channel: usize, width: u32, height: u32, u: f32, v: f32) -> f32 {
    let px = u * width as f32 - 0.5;
    let py = v * height as f32 - 0.5;
    let x0 = px.floor() as i32;
    let y0 = py.floor() as i32;
    let tx = px - x0 as f32;
    let ty = py - y0 as f32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let clamp_x0 = x0.clamp(0, width as i32 - 1) as u32;
    let clamp_x1 = x1.clamp(0, width as i32 - 1) as u32;
    let clamp_y0 = y0.clamp(0, height as i32 - 1) as u32;
    let clamp_y1 = y1.clamp(0, height as i32 - 1) as u32;
    let at = |x: u32, y: u32| plane[((y * width + x) * 4 + channel as u32) as usize] as f32;
    let top = at(clamp_x0, clamp_y0) + (at(clamp_x1, clamp_y0) - at(clamp_x0, clamp_y0)) * tx;
    let bottom = at(clamp_x0, clamp_y1) + (at(clamp_x1, clamp_y1) - at(clamp_x0, clamp_y1)) * tx;
    top + (bottom - top) * ty
}

fn sample_bgra_uv(image: &Image, u: f32, v: f32) -> [u8; 4] {
    let px = u * image.width as f32 - 0.5;
    let py = v * image.height as f32 - 0.5;
    let x0 = px.floor() as i32;
    let y0 = py.floor() as i32;
    let tx = px - x0 as f32;
    let ty = py - y0 as f32;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let clamp_x0 = x0.clamp(0, image.width as i32 - 1) as u32;
    let clamp_x1 = x1.clamp(0, image.width as i32 - 1) as u32;
    let clamp_y0 = y0.clamp(0, image.height as i32 - 1) as u32;
    let clamp_y1 = y1.clamp(0, image.height as i32 - 1) as u32;
    let at = |x: u32, y: u32, channel: usize| {
        image.pixels[((y * image.width + x) * 4 + channel as u32) as usize] as f32
    };
    let mut out = [0u8; 4];
    for channel in 0..4 {
        let top = at(clamp_x0, clamp_y0, channel)
            + (at(clamp_x1, clamp_y0, channel) - at(clamp_x0, clamp_y0, channel)) * tx;
        let bottom = at(clamp_x0, clamp_y1, channel)
            + (at(clamp_x1, clamp_y1, channel) - at(clamp_x0, clamp_y1, channel)) * tx;
        out[channel] = to_byte(top + (bottom - top) * ty);
    }
    out
}

/// Shift-map warp plus integer body-mask composite; `mask=255` is the exact
/// shader result. UI 0 is identity.
pub fn body_shift(
    source: &Image,
    ui: i32,
    body_mask: &[u8],
    shift_map: &PackedShiftMap<'_>,
) -> Result<Image, Refusal> {
    let count = pixels(source)?;
    if !(-100..=100).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("PF90041_UI"));
    }
    plane(body_mask, count, "PF90041_BODY_MASK")?;
    if shift_map.width == 0
        || shift_map.height == 0
        || shift_map.shift_width == 0
        || shift_map.shift_height == 0
    {
        return Err(Refusal::ShapeMismatch("PF90041_SHIFT_MAP"));
    }
    if shift_map.rgba8.len() != shift_map.width as usize * shift_map.height as usize * 4 {
        return Err(Refusal::ShapeMismatch("PF90041_SHIFT_MAP"));
    }
    if !shift_map.ratio.is_finite() || shift_map.ratio <= 0.0 {
        return Err(Refusal::ValueOutOfRange("PF90041_SHIFT_RATIO"));
    }
    if ui == 0 {
        return Ok(source.clone());
    }

    let mut out = source.clone();
    for y in 0..source.height {
        for x in 0..source.width {
            let index = (y * source.width + x) as usize;
            let u = (x as f32 + 0.5) / source.width as f32;
            let v = (y as f32 + 0.5) / source.height as f32;
            let r = sample_plane_uv(shift_map.rgba8, 0, shift_map.width, shift_map.height, u, v);
            let g = sample_plane_uv(shift_map.rgba8, 1, shift_map.width, shift_map.height, u, v);
            let b = sample_plane_uv(shift_map.rgba8, 2, shift_map.width, shift_map.height, u, v);
            let a = sample_plane_uv(shift_map.rgba8, 3, shift_map.width, shift_map.height, u, v);
            let shift_x = ((r * 256.0 + g) - 32768.0)
                / shift_map.ratio
                / shift_map.shift_width as f32;
            let shift_y = ((b * 256.0 + a) - 32768.0)
                / shift_map.ratio
                / shift_map.shift_height as f32;

            let warped = sample_bgra_uv(source, u + shift_x, v + shift_y);
            let mask = body_mask[index] as i32;
            let offset = index * 4;
            for channel in 0..3 {
                let effect = warped[channel] as i32;
                let original = source.pixels[offset + channel] as i32;
                out.pixels[offset + channel] =
                    ((effect * mask + original * (255 - mask) + 127) / 255) as u8;
            }
            out.pixels[offset + 3] = source.pixels[offset + 3];
        }
    }
    Ok(out)
}

/// ---- row 101: PF405 body slim ---------------------------------------------

/// PF405 control: `(ui+100)/200`, then the piecewise `[0,0.5,1] -> [-2,0,2]`
/// map (both segments simplify to `4p-2`).
pub fn body_slim_degree(ui_raw: i32) -> Result<f32, Refusal> {
    if !(-100..=100).contains(&ui_raw) {
        return Err(Refusal::ValueOutOfRange("PF405_UI"));
    }
    let normalized = (ui_raw as f64 + 100.0) / 200.0;
    let (from, to, t) = if normalized <= 0.5 {
        (-2.0, 0.0, normalized / 0.5)
    } else {
        (0.0, 2.0, (normalized - 0.5) / 0.5)
    };
    Ok((from + (to - from) * t) as f32)
}

pub struct BodyRefineMesh {
    pub rows: u32,
    pub cols: u32,
    pub vertices: Vec<[f32; 2]>,
    pub indices: Vec<u16>,
}

/// 12x2 / 24-vertex / 102-index BodyRefine mesh from `0x4B7E0F0`. Landmark
/// record points are explicit; nothing is detected here.
pub fn body_refine_mesh(
    width: u32,
    height: u32,
    query: [f32; 2],
    record0: [f32; 2],
    record1: [f32; 2],
    record1b: [f32; 2],
) -> Result<BodyRefineMesh, Refusal> {
    const ROWS: u32 = 12;
    const COLS: u32 = 2;
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("image"));
    }
    if [query, record0, record1, record1b]
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("PF405_LANDMARKS"));
    }

    let center = [
        (record0[0] as f64 + record1[0] as f64) * 0.5,
        (record0[1] as f64 + record1[1] as f64) * 0.5,
    ];
    let origin = [
        (query[0] as f64 + (record1b[0] as f64 + center[0]) * 0.5) * 0.5,
        (query[1] as f64 + (record1b[1] as f64 + center[1]) * 0.5) * 0.5,
    ];
    let dx = center[0] - record1b[0] as f64;
    let dy = center[1] - record1b[1] as f64;
    let radial0 = 0.7 * (dx * dx + dy * dy).sqrt();

    let mut vertices = Vec::with_capacity((ROWS * COLS) as usize);
    for row in 0..ROWS {
        let angle = 2.0 * std::f64::consts::PI * row as f64 / ROWS as f64;
        let sine = angle.sin();
        let cosine = angle.cos();
        for col in 0..COLS {
            let (x, y) = if col == 0 {
                (
                    0.8 * (radial0 * 0.8 * cosine) + origin[0],
                    radial0 * sine + origin[1],
                )
            } else {
                let t = col as f64 / (COLS - 1) as f64;
                let radius = (1.0 - t) * radial0 + t * (5.5 * radial0);
                (
                    0.3 * (radius * 0.8 * cosine) + origin[0],
                    radius * sine + origin[1],
                )
            };
            vertices.push([(x / width as f64) as f32, (y / height as f64) as f32]);
        }
    }

    let mut indices = Vec::with_capacity(102);
    for row in 2..ROWS {
        indices.push(0);
        indices.push(((row - 1) * COLS) as u16);
        indices.push((row * COLS) as u16);
    }
    for column_step in 0..COLS - 1 {
        for row in 0..ROWS {
            let next = row + 1;
            let wrapped = if next == ROWS { 0 } else { next };
            let a = row * COLS + column_step;
            let b = wrapped * COLS + column_step;
            indices.push(a as u16);
            indices.push(b as u16);
            indices.push((a + 1) as u16);
            indices.push(b as u16);
            indices.push((a + 1) as u16);
            indices.push((b + 1) as u16);
        }
    }

    Ok(BodyRefineMesh {
        rows: ROWS,
        cols: COLS,
        vertices,
        indices,
    })
}

/// `BodyRefine.frag` non-UVMAP consumer: `degree = 1 - mask.r`, UV lerp, then a
/// bilinear sample at `u*width`. Alpha is copied from the same source pixel.
pub fn body_slim_warp(
    source: &Image,
    mask_r: &[u8],
    source_uv: (f32, f32),
    target_uv: (f32, f32),
) -> Result<Image, Refusal> {
    let count = pixels(source)?;
    plane(mask_r, count, "PF405_BODY_MASK_R")?;
    let (source_u, source_v) = source_uv;
    let (target_u, target_v) = target_uv;
    if ![source_u, source_v, target_u, target_v]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("PF405_UV"));
    }

    let mut out = source.clone();
    for index in 0..count {
        let mask = mask_r[index] as f32 / 255.0;
        let degree = 1.0 - mask;
        if degree <= 0.01 {
            continue;
        }
        let u = source_u + (target_u - source_u) * degree;
        let v = source_v + (target_v - source_v) * degree;
        let sampled = sample_bgra_px(source, u * source.width as f32, v * source.height as f32);
        let offset = index * 4;
        out.pixels[offset] = sampled[0];
        out.pixels[offset + 1] = sampled[1];
        out.pixels[offset + 2] = sampled[2];
        out.pixels[offset + 3] = source.pixels[offset + 3];
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel(rgba: [u8; 4]) -> Image {
        Image::solid(1, 1, rgba)
    }

    #[test]
    fn meta_is_sorted_by_pf_and_unique() {
        for pair in META.windows(2) {
            assert!(pair[0].pf < pair[1].pf, "{} then {}", pair[0].pf, pair[1].pf);
        }
        for row in META {
            assert_eq!(row.access, Access::Cpu);
        }
    }

    #[test]
    fn packed_image_field_validation() {
        let bad = Image {
            width: 2,
            height: 2,
            pixels: vec![0; 15],
        };
        assert_eq!(
            stray_hair(&bad, 0.0, 1.0, 1.0, 1.0, None, None, None, None),
            Err(Refusal::ShapeMismatch("image"))
        );
    }

    #[test]
    fn row91_full_mask_takes_repair_rgb() {
        let src = pixel([10, 20, 30, 200]);
        let repair = pixel([110, 220, 30, 7]);
        let out = stray_hair(
            &src,
            100.0,
            1.0,
            1.0,
            1.0,
            Some(&repair),
            Some(&[255]),
            None,
            None,
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0), [110, 220, 30, 200]);
    }

    #[test]
    fn row91_partial_alpha_rounds_away() {
        let src = pixel([10, 20, 30, 200]);
        let repair = pixel([110, 220, 30, 7]);
        let out = stray_hair(
            &src,
            100.0,
            1.0,
            1.0,
            1.0,
            Some(&repair),
            Some(&[128]),
            None,
            None,
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0), [60, 120, 30, 200]);
    }

    #[test]
    fn row91_zero_ui_and_zero_degree_are_identity() {
        let src = pixel([10, 20, 30, 200]);
        assert_eq!(
            stray_hair(&src, 0.0, 1.0, 1.0, 1.0, None, None, None, None).unwrap(),
            src
        );
        assert_eq!(
            stray_hair(&src, 50.0, 0.0, 1.0, 1.0, None, None, None, None).unwrap(),
            src
        );
    }

    #[test]
    fn row91_missing_model_inputs_refuse() {
        let src = pixel([10, 20, 30, 200]);
        assert_eq!(
            stray_hair(&src, 50.0, 1.0, 1.0, 1.0, None, Some(&[255]), None, None),
            Err(Refusal::MissingInput("PF90088_HAIRFLY_V1"))
        );
        assert_eq!(
            stray_hair(&src, 50.0, 1.0, 1.0, 1.0, Some(&src), None, None, None),
            Err(Refusal::MissingInput("PF90088_RAW_STRAY_HAIR_MASK"))
        );
    }

    #[test]
    fn row92_zero_strength_is_identity() {
        let src = pixel([10, 20, 30, 200]);
        assert_eq!(
            hair_smooth(&src, 0.0, 100.0, None, None, None, None).unwrap(),
            src
        );
    }

    #[test]
    fn row92_scalar_blend_without_detail() {
        let src = pixel([10, 20, 30, 200]);
        let blend = pixel([200, 200, 200, 0]);
        let out = hair_smooth(
            &src,
            100.0,
            0.0,
            Some(&[255]),
            Some(&[255]),
            Some(&blend),
            Some(&[128]),
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0), [200, 200, 200, 200]);
    }

    #[test]
    fn row92_detail_add_rounds() {
        let src = pixel([10, 20, 30, 200]);
        let blend = pixel([200, 200, 200, 0]);
        let out = hair_smooth(
            &src,
            100.0,
            100.0,
            Some(&[255]),
            Some(&[255]),
            Some(&blend),
            Some(&[128]),
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0), [202, 202, 202, 200]);
    }

    #[test]
    fn row92_gate_zero_and_missing_mask() {
        let src = pixel([10, 20, 30, 200]);
        let blend = pixel([200, 200, 200, 0]);
        assert_eq!(
            hair_smooth(&src, 100.0, 100.0, Some(&[0]), None, Some(&blend), Some(&[200]))
                .unwrap(),
            src
        );
        assert_eq!(
            hair_smooth(&src, 100.0, 0.0, None, None, Some(&blend), Some(&[128])),
            Err(Refusal::MissingInput("PF90139_HAIR_MASK"))
        );
    }

    #[test]
    fn row93_step_one_is_identity() {
        let src = pixel([100, 0, 250, 77]);
        let out = hair_gloss(&src, 1, &[200], &[255], (0, 0, 1, 1)).unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn row93_full_step_adds_detail_delta() {
        let src = pixel([100, 0, 250, 77]);
        let out = hair_gloss(&src, 10, &[200], &[255], (0, 0, 1, 1)).unwrap();
        assert_eq!(out.pixel(0, 0), [245, 145, 255, 77]);
    }

    #[test]
    fn row93_refuses_step_and_roi() {
        let src = pixel([100, 0, 250, 77]);
        assert_eq!(
            hair_gloss(&src, 0, &[200], &[255], (0, 0, 1, 1)),
            Err(Refusal::ValueOutOfRange("PF90160_UI_STEP"))
        );
        assert_eq!(
            hair_gloss(&src, 5, &[200], &[255], (1, 0, 1, 1)),
            Err(Refusal::ShapeMismatch("PF90160_ROI"))
        );
    }

    #[test]
    fn row94_raw_zero_is_full_repair() {
        let src = pixel([10, 20, 30, 200]);
        let repair = pixel([50, 60, 70, 255]);
        let out = white_hair(&src, 100.0, &[0], &repair).unwrap();
        assert_eq!(out.pixel(0, 0), [50, 60, 70, 255]);
    }

    #[test]
    fn row94_raw_255_is_identity() {
        let src = pixel([10, 20, 30, 200]);
        let repair = pixel([50, 60, 70, 255]);
        assert_eq!(white_hair(&src, 100.0, &[255], &repair).unwrap(), src);
    }

    #[test]
    fn row94_zero_ui_is_identity() {
        let src = pixel([10, 20, 30, 200]);
        let repair = pixel([50, 60, 70, 255]);
        assert_eq!(white_hair(&src, 0.0, &[0], &repair).unwrap(), src);
    }

    #[test]
    fn row95_full_centres_on_effect() {
        let src = pixel([10, 20, 30, 77]);
        let out = hair_color_unify(&src, 100.0, &[200, 100, 50], &[1.0]).unwrap();
        assert_eq!(out.pixel(0, 0), [200, 100, 50, 77]);
        let out = hair_color_unify(&src, -100.0, &[200, 100, 50], &[1.0]).unwrap();
        assert_eq!(out.pixel(0, 0), [200, 100, 50, 77]);
    }

    #[test]
    fn row95_centre_and_mask_zero_are_identity() {
        let src = pixel([10, 20, 30, 77]);
        assert_eq!(
            hair_color_unify(&src, 0.0, &[200, 100, 50], &[1.0]).unwrap(),
            src
        );
        assert_eq!(
            hair_color_unify(&src, 100.0, &[200, 100, 50], &[0.0]).unwrap(),
            src
        );
    }

    #[test]
    fn row95_refuses_out_of_range() {
        let src = pixel([10, 20, 30, 77]);
        assert_eq!(
            hair_color_unify(&src, 100.5, &[200, 100, 50], &[1.0]),
            Err(Refusal::ValueOutOfRange("PF30009_UI"))
        );
        assert_eq!(
            hair_color_unify(&src, 100.0, &[200, 100, 50], &[1.5]),
            Err(Refusal::ValueOutOfRange("PF30009_HAIR_MASK"))
        );
    }

    #[test]
    fn row96_full_weight_takes_neutral_target() {
        let src = pixel([10, 20, 30, 77]);
        let neutral = pixel([200, 100, 50, 0]);
        let out = hand_neutral_gray(
            &src,
            100.0,
            &[255],
            &[255],
            NeutralTarget::Bgra(&neutral),
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0), [200, 100, 50, 77]);
    }

    #[test]
    fn row96_partial_weight_blends() {
        let src = pixel([10, 20, 30, 77]);
        let neutral = pixel([200, 100, 50, 0]);
        let out = hand_neutral_gray(
            &src,
            100.0,
            &[128],
            &[255],
            NeutralTarget::Bgra(&neutral),
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0), [105, 60, 40, 77]);
    }

    #[test]
    fn row96_diff_target_is_explicit() {
        let src = pixel([10, 20, 30, 77]);
        let out = hand_neutral_gray(
            &src,
            100.0,
            &[255],
            &[255],
            NeutralTarget::Diff(&[0.5, 0.0, 0.0]),
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0), [138, 20, 30, 77]);
    }

    #[test]
    fn row96_refuses_shape() {
        let src = pixel([10, 20, 30, 77]);
        let neutral = Image::solid(2, 1, [0, 0, 0, 0]);
        assert_eq!(
            hand_neutral_gray(&src, 100.0, &[255], &[255], NeutralTarget::Bgra(&neutral)),
            Err(Refusal::ShapeMismatch("PF30187_MODEL_NEUTRAL"))
        );
    }

    #[test]
    fn row97_neutral_bias_is_subtracted() {
        let src = pixel([10, 20, 30, 77]);
        let out = hand_blue_vein(&src, 100, &[255], &[100, 90, 80], 0, (0, 0, 1, 1)).unwrap();
        assert_eq!(out.pixel(0, 0), [30, 30, 30, 77]);
    }

    #[test]
    fn row97_negative_correction_clamps_and_zero_mask_identity() {
        let src = pixel([10, 20, 30, 77]);
        let out = hand_blue_vein(&src, 100, &[255], &[60, 60, 60], 0, (0, 0, 1, 1)).unwrap();
        assert_eq!(out.pixel(0, 0), [0, 0, 10, 77]);
        assert_eq!(hand_blue_vein(&src, 100, &[0], &[100, 90, 80], 0, (0, 0, 1, 1)).unwrap(), src);
    }

    #[test]
    fn row97_refuses_ui_and_roi() {
        let src = pixel([10, 20, 30, 77]);
        assert_eq!(
            hand_blue_vein(&src, 101, &[255], &[100, 90, 80], 0, (0, 0, 1, 1)),
            Err(Refusal::ValueOutOfRange("PF30030_UI"))
        );
        assert_eq!(
            hand_blue_vein(&src, 100, &[255], &[100, 90, 80], 0, (0, 0, 2, 1)),
            Err(Refusal::ShapeMismatch("PF30030_ROI"))
        );
    }

    #[test]
    fn row98_zero_parameter_is_identity() {
        let src = Image::solid(2, 1, [10, 20, 30, 255]);
        assert_eq!(
            hand_shape(&src, 0, &[1.0, 1.0], &[0.0, 0.0], &[255, 255], &[0, 0], None).unwrap(),
            src
        );
    }

    #[test]
    fn row98_shift_moves_pixel_and_protect_blocks() {
        let src = Image::new(2, 1, vec![0, 0, 0, 255, 100, 100, 100, 255]).unwrap();
        let out = hand_shape(
            &src,
            100,
            &[1.0, 0.0],
            &[0.0, 0.0],
            &[255, 255],
            &[0, 0],
            Some(1.0),
        )
        .unwrap();
        assert_eq!(out.pixel(0, 0), [100, 100, 100, 255]);

        let protected = hand_shape(
            &src,
            100,
            &[1.0, 0.0],
            &[0.0, 0.0],
            &[255, 255],
            &[255, 255],
            Some(1.0),
        )
        .unwrap();
        assert_eq!(protected, src);
    }

    #[test]
    fn row98_source_scale_branch() {
        assert_eq!(hand_shape_source_scale(2047, 100).unwrap(), 1.0);
        assert_eq!(hand_shape_source_scale(4096, 100).unwrap(), 2.0);
    }

    #[test]
    fn row98_refuses_non_finite_shift() {
        let src = pixel([10, 20, 30, 255]);
        assert_eq!(
            hand_shape(&src, 50, &[f32::NAN], &[0.0], &[255], &[0], None),
            Err(Refusal::ValueOutOfRange("PF30190_SHIFT"))
        );
    }

    #[test]
    fn row99_full_mask_takes_repair() {
        let src = pixel([10, 20, 30, 77]);
        let repair = pixel([200, 100, 50, 0]);
        let out = foot_blue_vein(&src, 100.0, &[255], &repair, 1).unwrap();
        assert_eq!(out.pixel(0, 0), [200, 100, 50, 77]);
    }

    #[test]
    fn row99_zero_mask_and_zero_ui_are_identity() {
        let src = pixel([10, 20, 30, 77]);
        let repair = pixel([200, 100, 50, 0]);
        assert_eq!(foot_blue_vein(&src, 100.0, &[0], &repair, 1).unwrap(), src);
        assert_eq!(foot_blue_vein(&src, 0.0, &[255], &repair, 1).unwrap(), src);
    }

    #[test]
    fn row99_refuses_feather_radius() {
        let src = pixel([10, 20, 30, 77]);
        assert_eq!(
            foot_blue_vein(&src, 100.0, &[255], &src, 0),
            Err(Refusal::ValueOutOfRange("PF44297_FEATHER_RADIUS"))
        );
    }

    fn neutral_shift() -> [u8; 4] {
        [128, 0, 128, 0]
    }

    #[test]
    fn row100_zero_shift_and_zero_ui_are_identity() {
        let src = Image::solid(2, 1, [10, 20, 30, 255]);
        let map = PackedShiftMap {
            width: 1,
            height: 1,
            rgba8: &neutral_shift(),
            ratio: 1.0,
            shift_width: 1,
            shift_height: 1,
        };
        assert_eq!(body_shift(&src, 100, &[255, 255], &map).unwrap(), src);
        assert_eq!(body_shift(&src, 0, &[255, 255], &map).unwrap(), src);
    }

    #[test]
    fn row100_half_pixel_shift_samples_neighbour() {
        let src = Image::new(2, 1, vec![10, 20, 30, 255, 200, 210, 220, 255]).unwrap();
        let packed = [128, 1, 128, 0];
        let map = PackedShiftMap {
            width: 1,
            height: 1,
            rgba8: &packed,
            ratio: 1.0,
            shift_width: 2,
            shift_height: 1,
        };
        let out = body_shift(&src, 100, &[255, 255], &map).unwrap();
        assert_eq!(out.pixel(0, 0), [200, 210, 220, 255]);
    }

    #[test]
    fn row100_refuses_ui_and_map_shape() {
        let src = pixel([10, 20, 30, 255]);
        let packed = neutral_shift();
        let map = PackedShiftMap {
            width: 1,
            height: 1,
            rgba8: &packed,
            ratio: 1.0,
            shift_width: 1,
            shift_height: 1,
        };
        assert_eq!(
            body_shift(&src, 101, &[255], &map),
            Err(Refusal::ValueOutOfRange("PF90041_UI"))
        );
        let short = [0u8; 3];
        let bad = PackedShiftMap {
            width: 1,
            height: 1,
            rgba8: &short,
            ratio: 1.0,
            shift_width: 1,
            shift_height: 1,
        };
        assert_eq!(
            body_shift(&src, 100, &[255], &bad),
            Err(Refusal::ShapeMismatch("PF90041_SHIFT_MAP"))
        );
    }

    #[test]
    fn row101_control_piecewise_degrees() {
        assert_eq!(body_slim_degree(-100).unwrap(), -2.0);
        assert_eq!(body_slim_degree(0).unwrap(), 0.0);
        assert_eq!(body_slim_degree(100).unwrap(), 2.0);
        assert_eq!(
            body_slim_degree(101),
            Err(Refusal::ValueOutOfRange("PF405_UI"))
        );
    }

    #[test]
    fn row101_mesh_has_fixed_topology() {
        let mesh = body_refine_mesh(
            100,
            100,
            [50.0, 50.0],
            [40.0, 50.0],
            [60.0, 50.0],
            [50.0, 40.0],
        )
        .unwrap();
        assert_eq!(mesh.rows, 12);
        assert_eq!(mesh.cols, 2);
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.indices.len(), 102);
        assert!((mesh.vertices[0][0] - 0.5448).abs() < 1e-5);
        assert!((mesh.vertices[1][0] - 0.5924).abs() < 1e-5);
    }

    #[test]
    fn row101_warp_full_mask_identity_and_half_warp() {
        let src = Image::new(2, 2, vec![
            0, 0, 0, 255,
            10, 10, 10, 255,
            20, 20, 20, 255,
            30, 30, 30, 255,
        ])
        .unwrap();
        let identity = body_slim_warp(&src, &[255; 4], (0.0, 0.0), (0.5, 0.5)).unwrap();
        assert_eq!(identity, src);

        let warped = body_slim_warp(&src, &[0; 4], (0.0, 0.0), (0.5, 0.5)).unwrap();
        assert_eq!(warped.pixel(0, 0), [30, 30, 30, 255]);
    }
}
