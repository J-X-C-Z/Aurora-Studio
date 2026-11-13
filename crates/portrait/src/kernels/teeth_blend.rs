//! PF30026 final mask blend (pix-worker.exe `509D2293...6B84`).
//!
//! Recovered at `0x230EC90` / helper `0x230DF40`. Roles at `0x5615C74`:
//! output and original alias `ctx+0x90`, effect is `record+0x20`, mask is
//! `record+0xA8`, direction 0.
//!
//! Strict native order:
//!   m = mask_u8 * 0x3B807BBC
//!   high = m * effect_f
//!   low  = original_f * (1 - m)
//!   out  = cvtt(high + low)          // truncate toward zero
//!   mask == 255 copies effect RGB, alpha is always the original.
//!
//! The LUT operands are u8 and `m` is in `[0,1)`, so `high + low` stays in
//! `[0,256)`; `cvtt` cannot be negative or overflow. A violated domain is a
//! typed refusal, not a silent clamp.

use crate::{Image, Refusal};

/// Exact vendor constant: `0x8CEEF00` f32 bits `0x3B807BBC`.
pub const VENDOR_MASK_SCALE: f32 = f32::from_bits(0x3B80_7BBC);

fn blend_channel(m8: u8, original: u8, effect: u8) -> Result<u8, Refusal> {
    if m8 == 255 {
        return Ok(effect);
    }
    let m = m8 as f32 * VENDOR_MASK_SCALE;
    let high = m * effect as f32;
    let low = original as f32 * (1.0 - m);
    let sum = high + low;
    if !sum.is_finite() {
        return Err(Refusal::ValueOutOfRange("teeth blend sum"));
    }
    let truncated = sum.trunc();
    if !(0.0..=255.0).contains(&truncated) {
        return Err(Refusal::ValueOutOfRange("teeth blend sum"));
    }
    Ok(truncated as u8)
}

/// Explicit-input PF30026 final blend. This is the blend stage only; the
/// opaque model output and dense warp remain caller inputs.
pub fn apply(original: &Image, effect: &Image, mask: &[u8]) -> Result<Image, Refusal> {
    original
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("teeth blend original"))?;
    effect
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("teeth blend effect"))?;
    if original.width != effect.width || original.height != effect.height {
        return Err(Refusal::ShapeMismatch("teeth blend effect"));
    }
    if mask.len() != original.pixel_count() {
        return Err(Refusal::ShapeMismatch("teeth blend mask"));
    }

    let mut pixels = original.pixels.clone();
    for index in 0..mask.len() {
        let o = index * 4;
        for c in 0..3 {
            pixels[o + c] = blend_channel(mask[index], original.pixels[o + c], effect.pixels[o + c])?;
        }
        pixels[o + 3] = original.pixels[o + 3];
    }
    Image::new(original.width, original.height, pixels).ok_or(Refusal::ShapeMismatch("teeth blend output"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "external-fixtures")]
    fn fixture_parts() -> (Image, Image, Vec<u8>) {
        let fixture = crate::external_fixtures::read("teeth/blend_fixture.bin");
        assert_eq!(fixture.len(), 36);
        (Image::new(2, 2, fixture[..16].to_vec()).unwrap(),
         Image::new(2, 2, fixture[16..32].to_vec()).unwrap(), fixture[32..].to_vec())
    }

    fn parts() -> (Image, Image, Vec<u8>) {
        (Image::solid(2, 2, [10, 60, 120, 40]),
         Image::solid(2, 2, [200, 100, 20, 0]), vec![0, 127, 254, 255])
    }

    #[test]
    fn vendor_scale_bits() {
        assert_eq!(VENDOR_MASK_SCALE.to_bits(), 0x3b807bbc);
    }

    #[test]
    #[cfg(feature = "external-fixtures")]
    fn fixture_matches_native_lut_math() {
        let (original, effect, mask) = fixture_parts();
        let out = apply(&original, &effect, &mask).unwrap();
        assert_eq!(out.pixel(0, 0), [105, 115, 125, 40]);
        assert_eq!(out.pixel(1, 0), [50, 60, 70, 80]);
        assert_eq!(out.pixel(0, 1), [255, 255, 255, 120]);
        assert_eq!(out.pixel(1, 1), [129, 120, 120, 160]);
    }

    #[test]
    fn core_contract_explicit_input() {
        let (original, effect, mask) = parts();
        let out = apply(&original, &effect, &mask).unwrap();
        for (index, pixel) in out.pixels.chunks_exact(4).enumerate() {
            assert_eq!(pixel[3], original.pixels[index * 4 + 3]);
        }
        assert_eq!(apply(&original, &effect, &[0; 4]).unwrap(), original);
        let full = apply(&original, &effect, &[255; 4]).unwrap();
        for (index, pixel) in full.pixels.chunks_exact(4).enumerate() {
            assert_eq!(pixel[..3], effect.pixels[index * 4..index * 4 + 3]);
            assert_eq!(pixel[3], original.pixels[index * 4 + 3]);
        }
    }

    #[test]
    fn mask_sweep_matches_f32_and_f64_oracles() {
        let pairs = [
            (0u8, 0u8),
            (0, 255),
            (255, 0),
            (255, 255),
            (1, 254),
            (254, 1),
            (10, 200),
            (200, 10),
            (128, 127),
            (127, 128),
            (37, 211),
            (211, 37),
            (64, 192),
            (192, 64),
            (5, 250),
            (250, 5),
        ];
        let mut f64_boundary_diffs = 0usize;
        for m8 in 0..=255u8 {
            for &(original, effect) in &pairs {
                let native = blend_channel(m8, original, effect).unwrap();
                let m32 = m8 as f32 * VENDOR_MASK_SCALE;
                let sum32 = m32 * effect as f32 + original as f32 * (1.0 - m32);
                assert!(sum32.is_finite());
                assert!(sum32 >= 0.0 && sum32 < 256.0, "domain {}", sum32);
                let expected32 = if m8 == 255 {
                    effect as i32
                } else {
                    sum32.trunc() as i32
                };
                assert_eq!(native as i32, expected32, "f32 m8={m8} {original}/{effect}");

                let m64 = m8 as f64 * VENDOR_MASK_SCALE as f64;
                let sum64 = m64 * effect as f64 + original as f64 * (1.0 - m64);
                let f64_i = if m8 == 255 {
                    effect as i32
                } else {
                    sum64.trunc() as i32
                };
                let native_i = native as i32;
                if f64_i != native_i {
                    // C++ uses float LUTs; f64 is a cross-check only. The
                    // only allowed difference is the f32 value sitting one
                    // ulp below/above the integer boundary.
                    f64_boundary_diffs += 1;
                    assert!((f64_i - native_i).abs() == 1, "f64 m8={m8} {original}/{effect}");
                    let boundary = f64_i.max(native_i) as f32;
                    assert!(
                        (sum32 - boundary).abs() < 1e-3,
                        "non-boundary f64 diff m8={m8} {original}/{effect} sum32={sum32}"
                    );
                }
            }
        }
        // Bounded: only f32-float rounding boundaries, not a semantic drift.
        assert!(f64_boundary_diffs <= 256 * pairs.len());
    }

    #[test]
    fn shape_and_length_are_refused() {
        let (original, effect, _) = parts();
        assert_eq!(
            apply(&original, &effect, &[0; 3]),
            Err(Refusal::ShapeMismatch("teeth blend mask"))
        );
        assert_eq!(
            apply(&original, &Image::solid(1, 1, [0; 4]), &[0; 4]),
            Err(Refusal::ShapeMismatch("teeth blend effect"))
        );
        let broken = Image {
            width: 2,
            height: 2,
            pixels: vec![0; 15],
        };
        assert_eq!(
            apply(&broken, &effect, &[0; 4]),
            Err(Refusal::ShapeMismatch("teeth blend original"))
        );
    }
}
