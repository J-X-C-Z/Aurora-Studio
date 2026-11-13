//! PF21026 stubble removal: the shared repair blend with `w = min(mask_a, mask_b)`.
//!
//! `0x25D3870` sets the AddEx `gamma = 0` and `alpha = strength`; `0x82AB10D`
//! gates on `strength > 0.005`, and `0x82AB128` scales it by `0.75` when the
//! long edge is under 300 px. The UI-to-native strength mapping is not recovered,
//! so the caller passes it explicitly.

use crate::{Access, EffectMeta, Image, Refusal};

pub const META: &[EffectMeta] = &[EffectMeta {
    pf: 21026,
    key: "removeStubble",
    label: "祛胡须",
    access: Access::Cpu,
    inputs: &["mask_a", "mask_b", "repair RGB", "strength"],
}];

pub const STRENGTH_GATE: f32 = 0.005;
pub const SMALL_FRAME_LONG_EDGE: u32 = 300;
pub const SMALL_FRAME_SCALE: f32 = 0.75;

fn quantize(value: f32) -> u8 {
    if !(value > 0.0) {
        0
    } else if value >= 255.0 {
        255
    } else {
        value.round_ties_even() as u8
    }
}

/// `w = min(a, b)`; the vendor keeps the repair colour in a byte plane.
pub fn pixel(base: u8, target: u8, mask_a: u8, mask_b: u8, strength: f32) -> u8 {
    let weight = mask_a.min(mask_b) as f32;
    let mixed = (weight * target as f32 + (255.0 - weight) * base as f32) * (1.0 / 255.0);
    quantize(base as f32 + strength * (mixed - base as f32))
}

/// Gate first, then the vendor's `0.75` scale for frames under 300 px.
pub fn scaled_strength(width: u32, height: u32, strength: f32) -> Result<Option<f32>, Refusal> {
    if !strength.is_finite() || strength < 0.0 {
        return Err(Refusal::ValueOutOfRange("strength"));
    }
    if strength <= STRENGTH_GATE {
        return Ok(None);
    }
    Ok(Some(if width.max(height) < SMALL_FRAME_LONG_EDGE {
        strength * SMALL_FRAME_SCALE
    } else {
        strength
    }))
}

pub fn apply(
    source: &Image,
    repair: &Image,
    mask_a: &[u8],
    mask_b: &[u8],
    strength: f32,
) -> Result<Image, Refusal> {
    validate(source, repair, mask_a, mask_b)?;
    let Some(strength) = scaled_strength(source.width, source.height, strength)? else {
        return Ok(source.clone());
    };
    let mut output = source.clone();
    rows(&mut output.pixels, &source.pixels, &repair.pixels, mask_a, mask_b, strength);
    Ok(output)
}

fn validate(
    source: &Image,
    repair: &Image,
    mask_a: &[u8],
    mask_b: &[u8],
) -> Result<usize, Refusal> {
    source.validate()?;
    repair.validate().map_err(|_| Refusal::ShapeMismatch("repair"))?;
    if (source.width, source.height) != (repair.width, repair.height) {
        return Err(Refusal::ShapeMismatch("repair"));
    }
    let pixels = source.pixel_count();
    if mask_a.len() != pixels || mask_b.len() != pixels {
        return Err(Refusal::ShapeMismatch("mask"));
    }
    Ok(pixels)
}

/// Pixels from which the banded form beats the serial loop.
pub const BEARD_PARALLEL_MIN_PIXELS: usize = 65_536;

/// Row-band form of [`apply`]; bands own whole rows, so the output is
/// byte-identical to the serial call.
pub fn apply_parallel(
    source: &Image,
    repair: &Image,
    mask_a: &[u8],
    mask_b: &[u8],
    strength: f32,
    threads: usize,
) -> Result<Image, Refusal> {
    let pixels = validate(source, repair, mask_a, mask_b)?;
    if threads <= 1 || pixels < BEARD_PARALLEL_MIN_PIXELS {
        return apply(source, repair, mask_a, mask_b, strength);
    }
    let Some(strength) = scaled_strength(source.width, source.height, strength)? else {
        return Ok(source.clone());
    };
    let width = source.width as usize;
    let band_pixels = (source.height as usize).div_ceil(threads).max(1) * width;
    let mut output = source.clone();
    std::thread::scope(|scope| {
        for (band, chunk) in output.pixels.chunks_mut(band_pixels * 4).enumerate() {
            let start = band * band_pixels;
            let count = chunk.len() / 4;
            if count == 0 {
                continue;
            }
            let source_band = &source.pixels[start * 4..(start + count) * 4];
            let repair_band = &repair.pixels[start * 4..(start + count) * 4];
            let mask_a_band = &mask_a[start..start + count];
            let mask_b_band = &mask_b[start..start + count];
            scope.spawn(move || {
                rows(chunk, source_band, repair_band, mask_a_band, mask_b_band, strength);
            });
        }
    });
    Ok(output)
}

fn rows(
    output: &mut [u8],
    source: &[u8],
    repair: &[u8],
    mask_a: &[u8],
    mask_b: &[u8],
    strength: f32,
) {
    for i in 0..mask_a.len() {
        let at = i * 4;
        for channel in 0..3 {
            output[at + channel] = pixel(
                source[at + channel],
                repair[at + channel],
                mask_a[i],
                mask_b[i],
                strength,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_and_mask_extremes_are_identity_or_repair() {
        assert_eq!(pixel(10, 200, 255, 255, 0.0), 10);
        assert_eq!(pixel(10, 200, 0, 255, 1.0), 10);
        assert_eq!(pixel(10, 200, 255, 255, 1.0), 200);
        // w = 128 -> 10 + (200-10)*128/255 = 105.37 -> 105
        assert_eq!(pixel(10, 200, 128, 255, 1.0), 105);
    }

    #[test]
    fn strength_gate_is_strict_and_keeps_the_source() {
        let source = Image::solid(2, 2, [10, 20, 30, 44]);
        let repair = Image::solid(2, 2, [200, 210, 220, 0]);
        let masks = [255u8; 4];
        assert_eq!(
            apply(&source, &repair, &masks, &masks, STRENGTH_GATE).unwrap(),
            source
        );
    }

    #[test]
    fn small_frames_scale_strength_by_three_quarters() {
        let small = Image::solid(200, 100, [0, 0, 0, 255]);
        let big = Image::solid(400, 100, [0, 0, 0, 255]);
        let repair = Image::solid(200, 100, [100, 100, 100, 255]);
        let repair_big = Image::solid(400, 100, [100, 100, 100, 255]);
        let masks_small = [255u8; 200 * 100];
        let masks_big = [255u8; 400 * 100];
        let a = apply(&small, &repair, &masks_small, &masks_small, 1.0).unwrap();
        let b = apply(&big, &repair_big, &masks_big, &masks_big, 1.0).unwrap();
        assert_eq!(a.pixel(0, 0)[0], 75);
        assert_eq!(b.pixel(0, 0)[0], 100);
        assert_eq!(a.pixel(0, 0)[3], 255);
    }

    #[test]
    fn parallel_bands_match_the_serial_beard() {
        let (width, height) = (320u32, 240u32);
        let pixels = (width * height) as usize;
        let source = Image::new(
            width,
            height,
            (0..pixels * 4).map(|i| ((i * 31 + 7) % 256) as u8).collect(),
        )
        .unwrap();
        let repair = Image::new(
            width,
            height,
            (0..pixels * 4).map(|i| ((i * 17 + 3) % 256) as u8).collect(),
        )
        .unwrap();
        let mask_a: Vec<u8> = (0..pixels).map(|i| ((i * 13 + 5) % 256) as u8).collect();
        let mask_b: Vec<u8> = (0..pixels).map(|i| ((i * 29 + 11) % 256) as u8).collect();
        let serial = apply(&source, &repair, &mask_a, &mask_b, 0.8).unwrap();
        for threads in [2usize, 5, 8] {
            let parallel =
                apply_parallel(&source, &repair, &mask_a, &mask_b, 0.8, threads).unwrap();
            assert_eq!(serial, parallel, "threads={threads}");
        }
    }

    #[test]
    fn malformed_inputs_are_refused() {
        let source = Image::solid(2, 2, [0; 4]);
        assert_eq!(
            apply(&source, &source, &[0; 3], &[0; 4], 1.0),
            Err(Refusal::ShapeMismatch("mask"))
        );
        assert_eq!(
            apply(&source, &source, &[0; 4], &[0; 4], f32::NAN),
            Err(Refusal::ValueOutOfRange("strength"))
        );
    }
}
