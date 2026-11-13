//! PF90116 eye bag / PF5 dark circle consolidated layer.
//!
//! `skin_color::eye_bag_repair_image` corresponds to **PF90116 / row 29**, not
//! PF5: the three-plane split, INTER_LINEAR resize and repair-image build come
//! from `0x5C1C54F/0x5C1C635/0x145C1AE70` on the eye-bag path. PF5 uses
//! `0x46D5DB0` residual, `0x5BF3B70/0x5BF3E10/0x5BF4AC0` ROIs.
//!
//! PF90116 final RGB composition is recovered at `0x5C1D1BF..0x5C1D3EF`:
//! quantized response times degree, float color target, truncation and clamp.
//! A caller still supplies the aligned model planes; no model is run here.
//! PF5 post-residual add/upsample/clamp remains a candidate; `Vendor` refuses.

#![allow(unused_imports)]

use crate::kernels::skin_color::{
    dark_circle_add_clamp, dark_circle_blend_from_pre_affine, dark_circle_field_from_pre_scale,
    dark_circle_residual, eye_bag_linear_mask_blend,
};
use crate::{Access, EffectMeta, Image, Refusal};

pub use crate::kernels::skin_color::Roi;

pub const META: &[EffectMeta] = &[];

pub const ROW29: EffectMeta = EffectMeta {
    pf: 90116,
    key: "Y.removeEyeBag",
    label: "祛眼袋",
    access: Access::Cpu,
    inputs: &["aligned_color_field_rgb_f32", "aligned_response_mask", "degree"],
};

pub const ROW28: EffectMeta = EffectMeta {
    pf: 5,
    key: "",
    label: "祛黑眼圈",
    access: Access::Refused,
    inputs: &["residual_field", "blend_weight", "degree_placement", "backprojection"],
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlendMode {
    /// Documented consumer candidate, not the vendor pixel law.
    ExplicitLinear,
    /// Recovered PF90116 composition; PF5 still refuses its unresolved final stage.
    Vendor,
}

pub const PF90116_ACTIVE_GATE: f32 = 0.001;
/// `0x8E6CD90` is `1/255`; the vendor multiplies, it does not divide.
const INV_255: f32 = 1.0 / 255.0;

pub use crate::kernels::skin_color::{
    eye_bag_bilinear as bilinear, eye_bag_repair_image as repair_image,
    eye_bag_response_roi as response_roi, EyeBagRoiDecision as RoiDecision,
};

/// `0x5C1C54F` copyTo: response Mat into the chosen ROI working image.
pub fn paste_response(
    response: &[f32],
    response_width: u32,
    response_height: u32,
    parent: &mut [f32],
    parent_width: u32,
    parent_height: u32,
    roi: Roi,
) -> Result<(), Refusal> {
    if roi.is_empty() || roi.x < 0 || roi.y < 0 {
        return Err(Refusal::ValueOutOfRange("roi"));
    }
    let (rw, rh, pw, ph) = (
        response_width as usize,
        response_height as usize,
        parent_width as usize,
        parent_height as usize,
    );
    if response.len() != rw * rh || parent.len() != pw * ph {
        return Err(Refusal::ShapeMismatch("planes"));
    }
    let (x, y, w, h) = (roi.x as usize, roi.y as usize, roi.w as usize, roi.h as usize);
    if x + w > pw || y + h > ph || rw != w || rh != h {
        return Err(Refusal::ShapeMismatch("roi"));
    }
    if response.iter().any(|v| !v.is_finite()) {
        return Err(Refusal::ValueOutOfRange("response"));
    }
    for row in 0..h {
        let src = row * w;
        let dst = (y + row) * pw + x;
        parent[dst..dst + w].copy_from_slice(&response[src..src + w]);
    }
    Ok(())
}

/// Native response selection (`0x5C0A400`): `None` = keep the whole response,
/// `Some(roi)` = crop. Area is an i32 product converted to f32 (never two f32
/// widths); i32 overflow is a typed refusal.
pub fn response_source(roi: Roi, width: u32, height: u32) -> Result<Option<Roi>, Refusal> {
    if width > i32::MAX as u32 || height > i32::MAX as u32 {
        return Err(Refusal::ValueOutOfRange("dimension"));
    }
    let (iw, ih) = (width as i32, height as i32);
    let invalid = roi.x < 0 || roi.y < 0 || roi.w < 1 || roi.h < 1;
    let whole = roi.x == 0 && roi.y == 0 && roi.w == iw && roi.h == ih;
    if invalid || whole {
        return Ok(None);
    }
    let image_area = iw
        .checked_mul(ih)
        .ok_or(Refusal::ValueOutOfRange("response_area"))? as f32;
    let roi_area = roi
        .w
        .checked_mul(roi.h)
        .ok_or(Refusal::ValueOutOfRange("roi_area"))? as f32;
    if image_area * 0.9f32 <= roi_area {
        Ok(None)
    } else {
        Ok(Some(roi))
    }
}

/// Single-plane response preparation: validates the source length before the
/// full/whole/90%-area branch can copy anything, then returns the selected
/// plane and the ROI actually written to `+0x680` (`None` = full image).
pub fn prepare_response(
    response: &[f32],
    width: u32,
    height: u32,
    roi: Roi,
) -> Result<(Vec<f32>, Option<Roi>), Refusal> {
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("response"))?;
    if response.len() != pixels {
        return Err(Refusal::ShapeMismatch("response"));
    }
    if response.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("response"));
    }

    match response_source(roi, width, height)? {
        None => Ok((response.to_vec(), None)),
        Some(crop) => {
            let (x, y, w, h) = (crop.x, crop.y, crop.w, crop.h);
            if x < 0
                || y < 0
                || w < 1
                || h < 1
                || x as usize + w as usize > width as usize
                || y as usize + h as usize > height as usize
            {
                return Err(Refusal::ValueOutOfRange("roi"));
            }
            let mut out = vec![0.0f32; w as usize * h as usize];
            for row in 0..h as usize {
                let src = (y as usize + row) * width as usize + x as usize;
                let dst = row * w as usize;
                out[dst..dst + w as usize].copy_from_slice(&response[src..src + w as usize]);
            }
            Ok((out, Some(crop)))
        }
    }
}

/// `CV_8U` response conversion, including nearest-even rounding and saturation.
pub fn quantize_response(response: f32) -> u8 {
    let rounded = (response * 255.0).round_ties_even();
    if !rounded.is_finite() || !(-2_147_483_648.0..2_147_483_648.0).contains(&rounded) {
        return 0;
    }
    (rounded as i32).clamp(0, 255) as u8
}

pub(crate) fn check_pf90116(
    pixels: usize,
    repair3: &[f32],
    response: &[f32],
    degree: f32,
) -> Result<(), Refusal> {
    if pixels.checked_mul(3) != Some(repair3.len()) || response.len() != pixels {
        return Err(Refusal::ShapeMismatch("pf90116 planes"));
    }
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("degree"));
    }
    if repair3.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("pf90116 color"));
    }
    Ok(())
}

/// PF90116 final stage on an aligned RGB crop. Source floats represent source
/// bytes; repair floats retain the model's sub-byte values until final truncation.
pub fn compose_pf90116(
    original3: &mut [f32],
    repair3: &[f32],
    mask1: &[f32],
    degree: f32,
    mode: BlendMode,
) -> Result<(), Refusal> {
    match mode {
        BlendMode::ExplicitLinear => {
            eye_bag_linear_mask_blend(original3, repair3, mask1, degree)
        }
        BlendMode::Vendor => {
            if original3.len() % 3 != 0 {
                return Err(Refusal::ShapeMismatch("pf90116 planes"));
            }
            check_pf90116(original3.len() / 3, repair3, mask1, degree)?;
            if original3.iter().any(|value| !value.is_finite()) {
                return Err(Refusal::ValueOutOfRange("pf90116 color"));
            }
            // `0x5C0A5B2`: only two sub-gates below 0.001 skip; one degree
            // means the boundary itself stays active.
            if degree < PF90116_ACTIVE_GATE {
                return Ok(());
            }
            for (pixel, &response) in mask1.iter().enumerate() {
                let weight = quantize_response(response) as f32 * INV_255 * degree;
                if weight == 0.0 {
                    continue;
                }
                let one_minus = 1.0 - weight;
                for channel in 0..3 {
                    let index = pixel * 3 + channel;
                    let mixed = repair3[index] * weight + original3[index] * one_minus;
                    // cvttss2si returns INT_MIN for an out-of-range conversion.
                    original3[index] = if !(-2_147_483_648.0..2_147_483_648.0).contains(&mixed) {
                        0.0
                    } else {
                        (mixed as i32).clamp(0, 255) as f32
                    };
                }
            }
            Ok(())
        }
    }
}

/// PF5 native residual (`0x46D5DB0`) plus the candidate add/clamp consumer.
pub fn residual_pf5(
    source_rgb: &[u8],
    width: u32,
    height: u32,
    target3: &[f32],
    blend1: &[f32],
) -> Result<Vec<f32>, Refusal> {
    dark_circle_residual(source_rgb, width, height, target3, blend1)
}

pub fn blend_from_pre_affine(pre: &[f32]) -> Result<Vec<f32>, Refusal> {
    dark_circle_blend_from_pre_affine(pre)
}

pub fn field_from_pre_scale(pre: &[f32]) -> Result<Vec<f32>, Refusal> {
    dark_circle_field_from_pre_scale(pre)
}

/// PF5 final stage. `ExplicitLinear` = source + residual, clamp; `Vendor`
/// refuses until the post-residual composition and degree placement are captured.
pub fn compose_pf5(
    source_rgb: &[u8],
    residual3: &[f32],
    mode: BlendMode,
) -> Result<Vec<u8>, Refusal> {
    match mode {
        BlendMode::ExplicitLinear => dark_circle_add_clamp(source_rgb, residual3),
        // `0x46C6F20` at `0x5C0531D/0x5C053C6`: clamp(src + residual - 80).
        BlendMode::Vendor => pf5_add_bias(source_rgb, residual3),
    }
}

/// PF5 final in-place add with the recovered `0x50` bias, OpenCV-style rounding.
pub fn pf5_add_bias(source_rgb: &[u8], residual3: &[f32]) -> Result<Vec<u8>, Refusal> {
    if residual3.len() != source_rgb.len() {
        return Err(Refusal::ShapeMismatch("residual"));
    }
    if residual3.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("residual"));
    }
    Ok(source_rgb
        .iter()
        .zip(residual3)
        .map(|(source, residual)| {
            let value = *source as f32 + residual - 80.0;
            if value <= 0.0 {
                0
            } else if value >= 255.0 {
                255
            } else {
                value.round_ties_even() as u8
            }
        })
        .collect())
}

/// Recovered PF90116 final crop composition with a float RGB target.
pub fn compose_field_image_pf90116(
    source: &Image,
    repair3: &[f32],
    response: &[f32],
    degree: f32,
) -> Result<Image, Refusal> {
    source.validate()?;
    let mut original3: Vec<f32> = source.pixels.chunks_exact(4)
        .flat_map(|pixel| pixel[..3].iter().map(|&value| value as f32))
        .collect();
    compose_pf90116(&mut original3, repair3, response, degree, BlendMode::Vendor)?;
    let mut output = source.clone();
    for (source, target) in original3.chunks_exact(3).zip(output.pixels.chunks_exact_mut(4)) {
        for channel in 0..3 {
            target[channel] = source[channel] as u8;
        }
    }
    Ok(output)
}

/// Pixels from which the banded eye-bag form beats the serial loop.
pub const EYE_BAG_PARALLEL_MIN_PIXELS: usize = 65_536;

/// Row-band form of [`compose_field_image_pf90116`]; output is byte-identical
/// to the serial call because every band owns whole rows.
pub fn compose_field_image_pf90116_parallel(
    source: &Image,
    repair3: &[f32],
    response: &[f32],
    degree: f32,
    threads: usize,
) -> Result<Image, Refusal> {
    source.validate()?;
    let pixels = source.pixel_count();
    if threads <= 1 || pixels < EYE_BAG_PARALLEL_MIN_PIXELS {
        return compose_field_image_pf90116(source, repair3, response, degree);
    }
    check_pf90116(pixels, repair3, response, degree)?;
    let width = source.width as usize;
    let band_rows = (source.height as usize).div_ceil(threads).max(1);
    let mut output = source.clone();
    let bytes = width * 4;
    std::thread::scope(|scope| {
        for (band, chunk) in output.pixels.chunks_mut(band_rows * bytes).enumerate() {
            let start = band * band_rows * width;
            let rows = chunk.len() / bytes;
            if rows == 0 {
                continue;
            }
            let end = start + rows * width;
            let repair = &repair3[start * 3..end * 3];
            let mask = &response[start..end];
            let source_pixels = &source.pixels[band * band_rows * bytes..][..chunk.len()];
            scope.spawn(move || {
                let mut rgb: Vec<f32> = source_pixels
                    .chunks_exact(4)
                    .flat_map(|pixel| pixel[..3].iter().map(|&value| value as f32))
                    .collect();
                compose_pf90116(&mut rgb, repair, mask, degree, BlendMode::Vendor)
                    .expect("band validated before the scope");
                for (value, target) in rgb.chunks_exact(3).zip(chunk.chunks_exact_mut(4)) {
                    for channel in 0..3 {
                        target[channel] = value[channel] as u8;
                    }
                }
            });
        }
    });
    Ok(output)
}

/// Image-level PF90116 blend: RGB blended, alpha copied from source.
pub fn compose_image_pf90116(
    source: &Image,
    repair: &Image,
    mask1: &[f32],
    degree: f32,
    mode: BlendMode,
) -> Result<Image, Refusal> {
    source.validate()?;
    let pixels = source.pixel_count();
    if source.width != repair.width || source.height != repair.height {
        return Err(Refusal::ShapeMismatch("repair"));
    }
    repair.validate()?;
    let mut original3 = vec![0.0f32; pixels * 3];
    let mut repair3 = vec![0.0f32; pixels * 3];
    for p in 0..pixels {
        for c in 0..3 {
            original3[p * 3 + c] = source.pixels[p * 4 + c] as f32;
            repair3[p * 3 + c] = repair.pixels[p * 4 + c] as f32;
        }
    }
    compose_pf90116(&mut original3, &repair3, mask1, degree, mode)?;
    let mut output = source.clone();
    for p in 0..pixels {
        for c in 0..3 {
            output.pixels[p * 4 + c] = original3[p * 3 + c]
                .clamp(0.0, 255.0)
                .round_ties_even() as u8;
        }
    }
    Ok(output)
}

/// Image-level PF5 candidate: RGB residual add/clamp, alpha copied.
pub fn compose_image_pf5(
    source: &Image,
    residual3: &[f32],
    mode: BlendMode,
) -> Result<Image, Refusal> {
    source.validate()?;
    let pixels = source.pixel_count();
    if residual3.len() != pixels * 3 {
        return Err(Refusal::ShapeMismatch("residual"));
    }
    let mut rgb = vec![0u8; pixels * 3];
    for p in 0..pixels {
        for c in 0..3 {
            rgb[p * 3 + c] = source.pixels[p * 4 + c];
        }
    }
    let blended = compose_pf5(&rgb, residual3, mode)?;
    let mut output = source.clone();
    for p in 0..pixels {
        for c in 0..3 {
            output.pixels[p * 4 + c] = blended[p * 3 + c];
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_bag_vendor_composes_while_dark_circle_stays_open() {
        let source = Image::solid(1, 1, [40, 80, 120, 77]);
        let repair = Image::solid(1, 1, [200, 200, 200, 0]);
        assert_eq!(
            compose_image_pf90116(&source, &repair, &[1.0], 0.5, BlendMode::Vendor),
            Ok(Image::solid(1, 1, [120, 140, 160, 77]))
        );
        assert_eq!(
            compose_image_pf5(&source, &[0.0; 3], BlendMode::Vendor),
            Ok(Image::solid(1, 1, [0, 0, 40, 77]))
        );
        assert_eq!(crate::lookup(5).unwrap().access, Access::Cpu);
        assert_eq!(crate::lookup(90116).unwrap(), ROW29);
    }

    #[test]
    fn vendor_uses_byte_response_and_truncates_float_target() {
        assert_eq!(quantize_response(0.5), 128);
        assert_eq!(quantize_response(1.5 / 255.0), 2);
        assert_eq!(quantize_response(2.5 / 255.0), 2);
        for response in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(quantize_response(response), 0);
        }
        assert_eq!(quantize_response(2.0), 255);
        let source = Image::solid(1, 1, [10, 10, 100, 77]);
        let out = compose_field_image_pf90116(
            &source, &[11.25, 9.5, 0.0], &[0.5], 1.0,
        ).unwrap();
        assert_eq!(out.pixels, [10, 9, 49, 77]);
        let out = compose_field_image_pf90116(
            &source, &[300.0, -10.0, 12.75], &[2.0], 1.0,
        ).unwrap();
        assert_eq!(out.pixels, [255, 0, 12, 77]);
        // At 0.001 the mix weight moves a channel by <1 LSB, so the gate
        // boundary cannot change an 8-bit pixel; only exact 0 is identity here.
        for degree in [0.0, 0.0005, PF90116_ACTIVE_GATE] {
            assert_eq!(
                compose_field_image_pf90116(&source, &[255.0; 3], &[1.0], degree).unwrap(),
                source,
                "degree {degree} stays below one byte of change",
            );
        }
        // q*1/255 differs from q/255 at q=3; the vendor multiplies.
        let three = compose_field_image_pf90116(&source, &[100.0; 3], &[3.0 / 255.0], 1.0)
            .unwrap();
        let weight = 3.0f32 * (1.0f32 / 255.0);
        let expected = (100.0f32 * weight + 10.0 * (1.0 - weight)) as u8;
        assert_eq!(three.pixels[0], expected);
    }

    #[test]
    fn parallel_bands_match_the_serial_eye_bag() {
        let (width, height) = (320u32, 240u32);
        let pixels = (width * height) as usize;
        let source = Image::new(
            width,
            height,
            (0..pixels * 4).map(|i| ((i * 29 + 7) % 256) as u8).collect(),
        )
        .unwrap();
        let repair: Vec<f32> = (0..pixels * 3)
            .map(|i| ((i * 17 + 3) % 311) as f32 * 0.9 - 10.0)
            .collect();
        let response: Vec<f32> = (0..pixels)
            .map(|i| ((i * 13 + 5) % 256) as f32 / 255.0)
            .collect();
        let serial =
            compose_field_image_pf90116(&source, &repair, &response, 0.7).unwrap();
        for threads in [2usize, 5, 8] {
            let parallel = compose_field_image_pf90116_parallel(
                &source, &repair, &response, 0.7, threads,
            )
            .unwrap();
            assert_eq!(serial, parallel, "threads={threads}");
        }
    }

    #[test]
    fn vendor_validates_all_planes_before_mutating_source() {
        let source = [1.0, 2.0, 3.0];
        let mut out = source;
        assert_eq!(compose_pf90116(&mut out, &[], &[1.0], 1.0, BlendMode::Vendor),
            Err(Refusal::ShapeMismatch("pf90116 planes")));
        assert_eq!(out, source);
        assert_eq!(compose_pf90116(&mut out, &[f32::NAN; 3], &[1.0], 1.0, BlendMode::Vendor),
            Err(Refusal::ValueOutOfRange("pf90116 color")));
        assert_eq!(out, source);
    }

    #[test]
    fn explicit_candidates_preserve_source_alpha() {
        let source = Image::solid(1, 1, [40, 80, 120, 77]);
        let repair = Image::solid(1, 1, [200, 200, 200, 0]);
        let eye_bag = compose_image_pf90116(
            &source, &repair, &[1.0], 0.5, BlendMode::ExplicitLinear,
        ).unwrap();
        assert_eq!(eye_bag.pixels, [120, 140, 160, 77]);
        let dark_circle = compose_image_pf5(
            &source, &[-50.0, 0.5, 200.0], BlendMode::ExplicitLinear,
        ).unwrap();
        assert_eq!(dark_circle.pixels, [0, 80, 255, 77]);
    }

    #[test]
    fn response_paste_requires_matching_dimensions_and_preserves_parent_on_error() {
        let roi = Roi { x: 0, y: 0, w: 3, h: 2 };
        let mut parent = vec![9.0; 6];
        assert_eq!(
            paste_response(&[1.0; 6], 2, 3, &mut parent, 3, 2, roi),
            Err(Refusal::ShapeMismatch("roi"))
        );
        assert_eq!(parent, [9.0; 6]);
        paste_response(&[1.0; 6], 3, 2, &mut parent, 3, 2, roi).unwrap();
        assert_eq!(parent, [1.0; 6]);
    }

    #[test]
    fn malformed_images_are_rejected() {
        let source = Image { width: 0, height: 0, pixels: Vec::new() };
        assert_eq!(
            compose_image_pf5(&source, &[], BlendMode::ExplicitLinear),
            Err(Refusal::ShapeMismatch("image"))
        );
        let huge = Image { width: u32::MAX, height: u32::MAX, pixels: Vec::new() };
        assert_eq!(
            compose_image_pf5(&huge, &[], BlendMode::ExplicitLinear),
            Err(Refusal::ShapeMismatch("image"))
        );
    }

    #[test]
    fn response_source_keeps_invalid_whole_and_ninety_percent_full() {
        assert_eq!(
            response_source(Roi { x: 0, y: 0, w: 0, h: 0 }, 100, 100),
            Ok(None)
        );
        assert_eq!(
            response_source(Roi { x: 0, y: 0, w: 100, h: 100 }, 100, 100),
            Ok(None)
        );
        assert_eq!(
            response_source(Roi { x: 0, y: 0, w: 90, h: 100 }, 100, 100),
            Ok(None)
        );
        assert_eq!(
            response_source(Roi { x: 5, y: 5, w: 89, h: 100 }, 100, 100),
            Ok(Some(Roi { x: 5, y: 5, w: 89, h: 100 }))
        );
    }

    #[test]
    fn prepare_response_crops_small_roi_and_rejects_bad_shape() {
        let data: Vec<f32> = (0..9).map(|v| v as f32).collect();
        let (plane, roi) = prepare_response(&data, 3, 3, Roi { x: 1, y: 1, w: 2, h: 2 })
            .unwrap();
        assert_eq!(plane, [4.0, 5.0, 7.0, 8.0]);
        assert_eq!(roi, Some(Roi { x: 1, y: 1, w: 2, h: 2 }));

        assert_eq!(
            prepare_response(&[], 3, 3, Roi { x: 0, y: 0, w: 0, h: 0 }),
            Err(Refusal::ShapeMismatch("response"))
        );
        assert_eq!(
            prepare_response(&data, 3, 3, Roi { x: 2, y: 2, w: 2, h: 2 }),
            Err(Refusal::ValueOutOfRange("roi"))
        );
    }

    #[test]
    fn response_area_is_i32_then_f32_above_2_pow_24() {
        let int_area = 16_777_217i32.checked_mul(5).unwrap() as f32;
        let float_direct = 16_777_217u32 as f32 * 5u32 as f32;
        assert_eq!(int_area, 83_886_088.0);
        assert_eq!(float_direct, 83_886_080.0);

        let roi = Roi { x: 0, y: 0, w: 15_099_495, h: 5 };
        assert_eq!(response_source(roi, 16_777_217, 5), Ok(Some(roi)));
    }

    #[test]
    fn response_area_overflow_is_typed_refusal() {
        assert_eq!(
            response_source(Roi { x: 0, y: 0, w: 10, h: 10 }, 65_536, 65_536),
            Err(Refusal::ValueOutOfRange("response_area"))
        );
        assert_eq!(
            response_source(Roi { x: 0, y: 0, w: 65_536, h: 65_536 }, 100, 100),
            Err(Refusal::ValueOutOfRange("roi_area"))
        );
    }
}
