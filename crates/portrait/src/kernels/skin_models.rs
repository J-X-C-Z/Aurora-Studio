// Skin model pixel consumers: PF8068 gloss, PF8069 surface blur, PF8070 clarity.
//
// Sources:
// - `Pf8068Gloss.cs` + `Pf8068Gloss.Tables.cs` (exact three-pass LUT consumer)
// - `TextureDegreeCore.cs` (FaceSurfaceBlur AlgoType=0 negative branch)
// - `Pf8070ClarityCore.cs` (type136 SharpnessWithMask law)
//
// Masks, gloss LUTs, base/blur planes and degrees are caller inputs. The PF8069
// positive FaceSticker branch stays a named refusal.

use crate::{Access, EffectMeta, Image, Refusal};


/// 256-entry LUT lookup used by the gloss pass: `coded/255*256 - 0.5`,
/// edge taps clamped independently.
pub fn gloss_lut_sample(lut: &[u8; 256], coded: u8) -> f32 {
    let position = (coded as f32 / 255.0) * 256.0 - 0.5;
    let lower = position.floor();
    let fraction = position - lower;
    let left = (lower as i32).clamp(0, 255) as usize;
    let right = (lower as i32 + 1).clamp(0, 255) as usize;
    (lut[left] as f32 / 255.0) * (1.0 - fraction) + (lut[right] as f32 / 255.0) * fraction
}

fn check_image(image: &Image, what: &'static str) -> Result<usize, Refusal> {
    image.validate().map_err(|_| Refusal::ShapeMismatch(what))?;
    Ok(image.pixel_count())
}

fn to_byte(value: f32) -> u8 {
    let encoded = value * 255.0;
    if encoded <= 0.0 {
        0
    } else if encoded >= 255.0 {
        255
    } else {
        encoded.round_ties_even() as u8
    }
}

fn quantize(value: f32) -> u8 {
    (value * 255.0).round_ties_even().clamp(0.0, 255.0) as u8
}

fn gloss_pass(
    out: &mut Image,
    mask_r: &[u8],
    extra_mask_r: &[u8],
    lut: &[u8; 256],
    degree: f32,
) {
    if degree == 0.0 {
        return;
    }
    for (index, px) in out.pixels.chunks_exact_mut(4).enumerate() {
        let mask = (mask_r[index] as f32 / 255.0) * (extra_mask_r[index] as f32 / 255.0);
        for channel in 0..3 {
            let source = px[channel] as f32 / 255.0;
            let mapped = gloss_lut_sample(lut, px[channel]);
            let value = (source + (mapped - source) * degree).clamp(0.0, 1.0);
            let result = source + (value - source) * mask;
            px[channel] = to_byte(result);
        }
    }
}

/// PF8068: three passes with per-pass R8 masks and extra masks. Degrees follow
/// `positive=max(2*u-1,0)`, `negative=max(1-2*u,0)`, with `u=native_percent/100`.
/// This is captured config percent (neutral 50), not signed UI (neutral 0).
/// The unresolved UI writer is not replaced with an invented conversion.
/// Compatibility entry: active work refuses missing LUTs; use gloss_apply_with_luts.
pub fn gloss_apply(
    source: &Image,
    masks_r: [&[u8]; 3],
    extra_masks_r: [&[u8]; 3],
    native_percent: f32,
) -> Result<Image, Refusal> {
    gloss_apply_with_luts(source, masks_r, extra_masks_r, native_percent, None)
}

/// Apply caller-owned LUTs in positive1, positive2, negative order.
/// No curves are bundled or generated. Missing LUTs refuse active work;
/// neutral native_percent=50 remains an identity after input validation.
pub fn gloss_apply_with_luts(
    source: &Image,
    masks_r: [&[u8]; 3],
    extra_masks_r: [&[u8]; 3],
    native_percent: f32,
    luts: Option<[&[u8; 256]; 3]>,
) -> Result<Image, Refusal> {
    let pixels = check_image(source, "source")?;
    if !native_percent.is_finite() || !(0.0..=100.0).contains(&native_percent) {
        return Err(Refusal::ValueOutOfRange("native_percent"));
    }
    for mask in masks_r.iter().chain(extra_masks_r.iter()) {
        if mask.len() != pixels {
            return Err(Refusal::ShapeMismatch("mask_r"));
        }
    }

    let normalized = native_percent / 100.0;
    let positive = (2.0 * normalized - 1.0).max(0.0);
    let negative = (1.0 - 2.0 * normalized).max(0.0);

    if positive == 0.0 && negative == 0.0 {
        return Ok(source.clone());
    }
    let luts = luts.ok_or(Refusal::MissingInput("gloss_luts"))?;
    let mut out = source.clone();
    gloss_pass(&mut out, masks_r[0], extra_masks_r[0], luts[0], positive);
    gloss_pass(&mut out, masks_r[1], extra_masks_r[1], luts[1], positive);
    gloss_pass(&mut out, masks_r[2], extra_masks_r[2], luts[2], negative);
    Ok(out)
}

fn remap_piecewise(x: f32, e0: f32, e1: f32, e2: f32) -> f32 {
    if x <= 0.0 {
        e0
    } else if x < 0.5 {
        e0 + (x / 0.5) * (e1 - e0)
    } else if x <= 1.0 {
        e1 + ((x - 0.5) / 0.5) * (e2 - e1)
    } else {
        e2
    }
}

fn sample_bilinear(rgba: &[u8], width: usize, height: usize, u: f32, v: f32) -> [f32; 4] {
    let px = u * width as f32 - 0.5;
    let py = v * height as f32 - 0.5;
    let x0f = px.floor();
    let y0f = py.floor();
    let fx = px - x0f;
    let fy = py - y0f;
    let x0 = (x0f as i32).clamp(0, width as i32 - 1) as usize;
    let x1 = (x0f as i32 + 1).clamp(0, width as i32 - 1) as usize;
    let y0 = (y0f as i32).clamp(0, height as i32 - 1) as usize;
    let y1 = (y0f as i32 + 1).clamp(0, height as i32 - 1) as usize;
    let mut out = [0.0f32; 4];
    for (channel, slot) in out.iter_mut().enumerate() {
        let at = |y: usize, x: usize| {
            rgba[(y * width + x) * 4 + channel] as f32 / 255.0
        };
        let top = at(y0, x0) + (at(y0, x1) - at(y0, x0)) * fx;
        let bottom = at(y1, x0) + (at(y1, x1) - at(y1, x0)) * fx;
        *slot = top + (bottom - top) * fy;
    }
    out
}

fn smooth_step(edge0: f32, edge1: f32, value: f32) -> f32 {
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn surface_pass(
    source: &[u8],
    width: usize,
    height: usize,
    step_u: f32,
    step_v: f32,
    hk: f32,
) -> Vec<u8> {
    let mut result = vec![0u8; width * height * 4];
    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32;
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let center = sample_bilinear(source, width, height, u, v);
            let mut sum = [center[0], center[1], center[2]];
            let mut weight_sum = 1.0f32;
            for distance in 1..=4 {
                for sign in [-1.0f32, 1.0] {
                    let sample = sample_bilinear(
                        source,
                        width,
                        height,
                        u + sign * distance as f32 * step_u,
                        v + sign * distance as f32 * step_v,
                    );
                    let dr = sample[0] - center[0];
                    let dg = sample[1] - center[1];
                    let db = sample[2] - center[2];
                    let weight = 1.0 - smooth_step(0.0, hk, (dr * dr + dg * dg + db * db).sqrt());
                    for channel in 0..3 {
                        sum[channel] += sample[channel] * weight;
                    }
                    weight_sum += weight;
                }
            }
            let offset = (y * width + x) * 4;
            for channel in 0..3 {
                result[offset + channel] = to_byte(sum[channel] / weight_sum);
            }
            result[offset + 3] = to_byte(center[3]);
        }
    }
    result
}

fn gauss_pass(source: &[u8], width: usize, height: usize, step_u: f32, step_v: f32) -> Vec<u8> {
    const WEIGHTS: [f32; 4] = [0.15, 0.12, 0.09, 0.05];
    let mut result = vec![0u8; width * height * 4];
    for y in 0..height {
        let v = (y as f32 + 0.5) / height as f32;
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let center = sample_bilinear(source, width, height, u, v);
            let mut sum = [
                0.18 * center[0],
                0.18 * center[1],
                0.18 * center[2],
                0.18 * center[3],
            ];
            for distance in 1..=4 {
                let weight = WEIGHTS[distance - 1];
                for sign in [-1.0f32, 1.0] {
                    let sample = sample_bilinear(
                        source,
                        width,
                        height,
                        u + sign * distance as f32 * step_u,
                        v + sign * distance as f32 * step_v,
                    );
                    for channel in 0..4 {
                        sum[channel] += sample[channel] * weight;
                    }
                }
            }
            let offset = (y * width + x) * 4;
            for channel in 0..4 {
                result[offset + channel] = to_byte(sum[channel]);
            }
        }
    }
    result
}

/// PF8069 negative branch: type40 surface blur H/V, type33 9-tap gauss H/V,
/// then `layerAlpha/255 * degree * mask/255 * extraMask/255` mix.
/// UI 0 is identity; UI > 0 is the OPEN FaceSticker branch and refuses.
pub fn texture_surface_blur(
    source: &Image,
    mask_r: &[u8],
    extra_mask_r: &[u8],
    ui: f32,
) -> Result<Image, Refusal> {
    let pixels = check_image(source, "source")?;
    if !ui.is_finite() || !(-100.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if mask_r.len() != pixels {
        return Err(Refusal::ShapeMismatch("mask_r"));
    }
    if extra_mask_r.len() != pixels {
        return Err(Refusal::ShapeMismatch("extra_mask_r"));
    }
    if ui > 0.0 {
        return Err(Refusal::NotImplemented(8069));
    }
    if ui == 0.0 {
        return Ok(source.clone());
    }

    let normalized = (ui + 100.0) / 200.0;
    let degree = remap_piecewise(normalized, 0.49, 0.0, 0.0);
    let width = source.width as usize;
    let height = source.height as usize;
    let hk = 10.45 / 255.0;

    let surface_h = surface_pass(&source.pixels, width, height, 4.93 / width as f32, 0.0, hk);
    let surface_v = surface_pass(&surface_h, width, height, 0.0, 4.93 / height as f32, hk);
    let gauss_h = gauss_pass(&surface_v, width, height, 1.45 / width as f32, 0.0);
    let layer = gauss_pass(&gauss_h, width, height, 0.0, 1.45 / height as f32);

    let mut out = source.clone();
    for (index, px) in out.pixels.chunks_exact_mut(4).enumerate() {
        let mix = (layer[index * 4 + 3] as f32 / 255.0)
            * degree
            * (mask_r[index] as f32 / 255.0)
            * (extra_mask_r[index] as f32 / 255.0);
        if !(mix > 0.01) {
            continue;
        }
        for channel in 0..3 {
            let src = px[channel] as f32 / 255.0;
            let dst = layer[index * 4 + channel] as f32 / 255.0;
            px[channel] = to_byte(src + (dst - src) * mix);
        }
        px[3] = source.pixels[index * 4 + 3];
    }
    Ok(out)
}

/// PF8069 with both branches: negative keeps the recovered surface-blur law;
/// positive needs a caller-supplied FilterType45 material layer and derives its
/// degree from the recovered `EnableDegreeRemap` path.
pub fn texture_apply(
    source: &Image,
    material_layer: Option<&Image>,
    mask_r: &[u8],
    extra_mask_r: &[u8],
    ui: f32,
) -> Result<Image, Refusal> {
    if !ui.is_finite() || !(-100.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if ui <= 0.0 {
        return texture_surface_blur(source, mask_r, extra_mask_r, ui);
    }
    let material = material_layer.ok_or(Refusal::MissingInput("material_layer"))?;
    crate::kernels::texture_material::face_sticker_apply(
        source, material, mask_r, extra_mask_r, ui,
    )
}

/// PF8070 type136 clarity. `base`/`blur` are ROI-sized RGBA8 planes; `mask_r`
/// is the same ROI as R8. `threshold` and `factor` come from the effect config
/// (defaults `0.0` and `4.68`).
pub fn clarity_apply(
    source: &Image,
    base: &Image,
    blur: &Image,
    mask_r: &[u8],
    roi: (u32, u32, u32, u32),
    degree: f32,
    threshold: f32,
    factor: f32,
) -> Result<Image, Refusal> {
    check_image(source, "source")?;
    let roi_pixels = check_image(base, "base")?;
    if check_image(blur, "blur")? != roi_pixels || (base.width, base.height) != (blur.width, blur.height) {
        return Err(Refusal::ShapeMismatch("blur"));
    }
    let (roi_x, roi_y, roi_w, roi_h) = roi;
    if (roi_w, roi_h) != (base.width, base.height) {
        return Err(Refusal::ShapeMismatch("roi"));
    }
    if u64::from(roi_x) + u64::from(roi_w) > u64::from(source.width)
        || u64::from(roi_y) + u64::from(roi_h) > u64::from(source.height) {
        return Err(Refusal::ValueOutOfRange("roi"));
    }
    if mask_r.len() != roi_pixels {
        return Err(Refusal::ShapeMismatch("mask_r"));
    }
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("degree"));
    }
    if !threshold.is_finite() || !factor.is_finite() {
        return Err(Refusal::ValueOutOfRange("filter"));
    }
    if degree == 0.0 {
        return Ok(source.clone());
    }

    let mut out = source.clone();
    let width = base.width as usize;
    for y in 0..roi_h as usize {
        for x in 0..roi_w as usize {
            let local = y * width + x;
            let mask = mask_r[local] as f32 / 255.0;
            let amount = degree * mask;
            if !(amount > 0.0) {
                continue;
            }
            let src_index = ((roi_y as usize + y) * source.width as usize + roi_x as usize + x) * 4;
            let base_index = local * 4;
            for channel in 0..3 {
                let src = source.pixels[src_index + channel] as f32 / 255.0;
                let base_value = base.pixels[base_index + channel] as f32 / 255.0;
                let blur_value = blur.pixels[base_index + channel] as f32 / 255.0;
                let diff = base_value - blur_value;
                let negative = (-diff - threshold).max(0.0);
                let positive = (diff - threshold).max(0.0);
                let edge = positive - negative;
                let reference = (src + factor * edge).clamp(0.0, 1.0);
                let result = src + (reference - src) * amount;
                out.pixels[src_index + channel] = quantize(result);
            }
        }
    }
    Ok(out)
}

const GLOSS_INPUTS: &[&str] = &[
    "source", "masks_r[3]", "extra_masks_r[3]", "native_percent", "gloss_luts[3][256] (caller)",
];
const TEXTURE_INPUTS: &[&str] = &[
    "source", "mask_r", "extra_mask_r", "ui", "material_layer (ui>0)",
];
const CLARITY_INPUTS: &[&str] = &[
    "source", "roi_base", "roi_blur", "mask_r", "roi", "degree", "threshold", "factor",
];

pub const META: &[EffectMeta] = &[
    EffectMeta {
        pf: 8068,
        key: "skinGloss",
        label: "皮肤光泽",
        access: Access::Cpu,
        inputs: GLOSS_INPUTS,
    },
    EffectMeta {
        pf: 8069,
        key: "skinTexture",
        label: "Skin texture (negative blur / positive material)",
        access: Access::Cpu,
        inputs: TEXTURE_INPUTS,
    },
    EffectMeta {
        pf: 8070,
        key: "skinClarity",
        label: "清晰度",
        access: Access::Cpu,
        inputs: CLARITY_INPUTS,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic test curves only, not substitutes for product assets.
    const TEST_LUTS: [[u8; 256]; 3] = [[255; 256], [160; 256], [0; 256]];

    #[test]
    fn clarity_rejects_transposed_planes_and_overflowing_roi() {
        let source = Image::solid(2, 2, [100, 90, 80, 77]);
        let base = Image::solid(2, 1, [90, 80, 70, 77]);
        let transposed = Image::solid(1, 2, [80, 70, 60, 77]);
        assert_eq!(clarity_apply(&source, &base, &transposed, &[255; 2],
            (0, 0, 2, 1), 1.0, 0.0, 4.68), Err(Refusal::ShapeMismatch("blur")));
        assert_eq!(clarity_apply(&source, &base, &base, &[255; 2],
            (0, 0, 1, 2), 1.0, 0.0, 4.68), Err(Refusal::ShapeMismatch("roi")));
        assert_eq!(clarity_apply(&source, &base, &base, &[255; 2],
            (u32::MAX, 0, 2, 1), 1.0, 0.0, 4.68), Err(Refusal::ValueOutOfRange("roi")));
    }

    fn pixel(rgba: [u8; 4]) -> Image {
        Image::solid(1, 1, rgba)
    }

    #[test]
    fn meta_only_registers_the_three_pixel_pfs() {
        assert_eq!(META.iter().map(|meta| meta.pf).collect::<Vec<_>>(), [8068, 8069, 8070]);
        assert!(META.iter().all(|meta| meta.access == Access::Cpu));
    }

    #[test]
    fn gloss_lut_is_a_rising_ramp() {
        let mut ramp = [0u8; 256];
        for (index, slot) in ramp.iter_mut().enumerate() {
            *slot = index as u8;
        }
        for coded in [0u8, 1, 64, 128, 200, 255] {
            let sampled = gloss_lut_sample(&ramp, coded);
            assert!((sampled - coded as f32 / 255.0).abs() < 0.01, "coded {coded}");
        }
    }

    #[test]
    fn gloss_positive_and_negative_are_non_identity() {
        let src = pixel([200, 100, 60, 77]);
        let masks = [[255u8; 1]; 3];
        let extras = [[255u8; 1]; 3];
        let positive = gloss_apply_with_luts(&src, [&masks[0], &masks[1], &masks[2]], [&extras[0], &extras[1], &extras[2]], 100.0, Some([&TEST_LUTS[0], &TEST_LUTS[1], &TEST_LUTS[2]])).unwrap();
        let negative = gloss_apply_with_luts(&src, [&masks[0], &masks[1], &masks[2]], [&extras[0], &extras[1], &extras[2]], 0.0, Some([&TEST_LUTS[0], &TEST_LUTS[1], &TEST_LUTS[2]])).unwrap();
        assert_ne!(positive.pixel(0, 0), src.pixel(0, 0));
        assert_ne!(negative.pixel(0, 0), src.pixel(0, 0));
        assert_eq!(positive.pixel(0, 0)[3], 77);
        assert_eq!(negative.pixel(0, 0)[3], 77);
        let expected = to_byte(gloss_lut_sample(&TEST_LUTS[0], 200));
        let second = to_byte(gloss_lut_sample(&TEST_LUTS[1], expected));
        assert_eq!(positive.pixel(0, 0)[0], second);
    }

    #[test]
    fn gloss_midpoint_and_zero_mask_are_identity() {
        let src = pixel([200, 100, 60, 77]);
        let masks = [[255u8; 1]; 3];
        let extras = [[255u8; 1]; 3];
        let mid = gloss_apply_with_luts(&src, [&masks[0], &masks[1], &masks[2]], [&extras[0], &extras[1], &extras[2]], 50.0, Some([&TEST_LUTS[0], &TEST_LUTS[1], &TEST_LUTS[2]])).unwrap();
        assert_eq!(mid, src);
        let zero = [[0u8; 1]; 3];
        let out = gloss_apply_with_luts(&src, [&zero[0], &zero[1], &zero[2]], [&zero[0], &zero[1], &zero[2]], 100.0, Some([&TEST_LUTS[0], &TEST_LUTS[1], &TEST_LUTS[2]])).unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn gloss_refuses_bad_inputs() {
        let source = pixel([200, 100, 60, 77]);
        let masks = [&[255u8][..]; 3];
        for percent in [0.0, 75.0, 100.0] {
            assert_eq!(gloss_apply(&source, masks, masks, percent),
                Err(Refusal::MissingInput("gloss_luts")));
        }
        assert_eq!(gloss_apply(&source, masks, masks, 50.0).unwrap(), source);

        let src = pixel([200, 100, 60, 77]);
        let full = [255u8; 1];
        let empty: [u8; 0] = [];
        assert_eq!(
            gloss_apply(&src, [&full, &full, &full], [&full, &full, &empty], 100.0),
            Err(Refusal::ShapeMismatch("mask_r"))
        );
        assert_eq!(
            gloss_apply(&src, [&full, &full, &full], [&full, &full, &full], f32::NAN),
            Err(Refusal::ValueOutOfRange("native_percent"))
        );
        let bad = Image { width: 1, height: 1, pixels: vec![0; 3] };
        assert_eq!(
            gloss_apply(&bad, [&full, &full, &full], [&full, &full, &full], 100.0),
            Err(Refusal::ShapeMismatch("source"))
        );
    }

    fn texture_tile() -> Image {
        let mut image = Image::solid(3, 3, [40, 40, 40, 200]);
        let center = (4 * 4) as usize;
        image.pixels[center] = 240;
        image.pixels[center + 3] = 99;
        image
    }

    #[test]
    fn texture_negative_branch_blurs_and_preserves_alpha() {
        let src = texture_tile();
        let masks = [255u8; 9];
        let out = texture_surface_blur(&src, &masks, &masks, -100.0).unwrap();
        assert_ne!(out.pixel(1, 1), src.pixel(1, 1));
        assert!(out.pixel(1, 1)[0] < 240);
        assert_eq!(out.pixel(1, 1)[3], src.pixel(1, 1)[3]);
        assert_eq!(out.pixel(0, 0)[3], src.pixel(0, 0)[3]);
    }

    #[test]
    fn texture_positive_branch_needs_material_and_composes_it() {
        let source = pixel([100, 120, 140, 77]);
        let material = pixel([200, 200, 200, 255]);
        let mask = [255u8; 1];
        assert_eq!(texture_apply(&source, None, &mask, &mask, 40.0),
            Err(Refusal::MissingInput("material_layer")));
        let out = texture_apply(&source, Some(&material), &mask, &mask, 100.0).unwrap();
        assert_eq!(out.pixel(0, 0), [200, 200, 200, 77]);
        let zero = texture_apply(&source, Some(&material), &mask, &mask, 0.0).unwrap();
        assert_eq!(zero, source);
        assert_eq!(texture_apply(&source, None, &mask, &mask, 0.0).unwrap(), source);
    }

    #[test]
    fn texture_zero_mask_and_zero_ui_are_identity() {
        let src = texture_tile();
        let masks = [255u8; 9];
        assert_eq!(texture_surface_blur(&src, &masks, &masks, 0.0).unwrap(), src);
        let zero = [0u8; 9];
        assert_eq!(texture_surface_blur(&src, &zero, &masks, -100.0).unwrap(), src);
    }

    #[test]
    fn texture_positive_branch_is_a_named_refusal() {
        let src = texture_tile();
        let masks = [255u8; 9];
        assert_eq!(
            texture_surface_blur(&src, &masks, &masks, 50.0),
            Err(Refusal::NotImplemented(8069))
        );
        assert_eq!(
            texture_surface_blur(&src, &[], &masks, -100.0),
            Err(Refusal::ShapeMismatch("mask_r"))
        );
    }

    fn clarity_inputs() -> (Image, Image, Image) {
        let src = pixel([100, 100, 100, 77]);
        let base = pixel([120, 120, 120, 77]);
        let blur = pixel([80, 80, 80, 77]);
        (src, base, blur)
    }

    #[test]
    fn clarity_sharpens_by_the_recovered_edge_law() {
        let (src, base, blur) = clarity_inputs();
        let out = clarity_apply(&src, &base, &blur, &[255], (0, 0, 1, 1), 1.0, 0.0, 4.68).unwrap();
        assert_eq!(out.pixel(0, 0), [255, 255, 255, 77]);

        let half = clarity_apply(&src, &base, &blur, &[128], (0, 0, 1, 1), 1.0, 0.0, 4.68).unwrap();
        assert!(half.pixel(0, 0)[0] > 100 && half.pixel(0, 0)[0] < 255);
        assert_eq!(half.pixel(0, 0)[3], 77);
    }

    #[test]
    fn clarity_identity_and_refusals() {
        let (src, base, blur) = clarity_inputs();
        assert_eq!(
            clarity_apply(&src, &base, &blur, &[255], (0, 0, 1, 1), 0.0, 0.0, 4.68).unwrap(),
            src
        );
        assert_eq!(
            clarity_apply(&src, &base, &blur, &[0], (0, 0, 1, 1), 1.0, 0.0, 4.68).unwrap(),
            src
        );
        assert_eq!(
            clarity_apply(&src, &base, &blur, &[], (0, 0, 1, 1), 1.0, 0.0, 4.68),
            Err(Refusal::ShapeMismatch("mask_r"))
        );
        assert_eq!(
            clarity_apply(&src, &base, &blur, &[255], (0, 0, 2, 2), 1.0, 0.0, 4.68),
            Err(Refusal::ShapeMismatch("roi"))
        );
        assert_eq!(
            clarity_apply(&src, &base, &blur, &[255], (0, 0, 1, 1), 1.0, f32::NAN, 4.68),
            Err(Refusal::ValueOutOfRange("filter"))
        );
    }
}
