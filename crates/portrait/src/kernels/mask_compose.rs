//! Shared post-mask composer, recovered at `0x5B8CEA0`.
//!
//! This is a pipeline stage rather than a user-facing effect, so it has no
//! `ParamFlag` row in the catalog.

use crate::{Image, Refusal};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskComposeMode {
    /// `out = trunc(out * (mask / 255))`
    MaskOnly,
    /// `out = trunc(out * (1 - 2 * mask / 255))`
    Alternate,
}

fn factor(mask: u8, mode: MaskComposeMode) -> f32 {
    const INV255: f32 = 1.0 / 255.0;
    let value = mask as f32;
    match mode {
        MaskComposeMode::MaskOnly => value * INV255,
        MaskComposeMode::Alternate => (value + value) * -INV255 + 1.0,
    }
}

/// RGB is scaled, alpha is preserved byte-for-byte, truncation happens before
/// the `[0,255]` clamp. The mask is one byte per RGBA pixel.
pub fn apply(image: &Image, mask: &[u8], mode: MaskComposeMode) -> Result<Image, Refusal> {
    image.validate()?;
    if mask.len() != image.pixel_count() {
        return Err(Refusal::ShapeMismatch("mask"));
    }

    let mut out = image.clone();
    for (index, px) in out.pixels.chunks_exact_mut(4).enumerate() {
        let f = factor(mask[index], mode);
        for channel in 0..3 {
            let scaled = px[channel] as f32 * f;
            px[channel] = (scaled as i32).clamp(0, 255) as u8;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pixel() -> Image {
        Image::solid(1, 1, [100, 150, 200, 77])
    }

    #[test]
    fn mask_only_scales_with_inverse_255_and_keeps_alpha() {
        let out = apply(&pixel(), &[128], MaskComposeMode::MaskOnly).unwrap();
        assert_eq!(out.pixel(0, 0), [50, 75, 100, 77]);
    }

    #[test]
    fn zero_mask_follows_each_branch() {
        let src = pixel();
        assert_eq!(
            apply(&src, &[0], MaskComposeMode::MaskOnly)
                .unwrap()
                .pixel(0, 0),
            [0, 0, 0, 77]
        );
        assert_eq!(apply(&src, &[0], MaskComposeMode::Alternate).unwrap(), src);
    }

    #[test]
    fn full_mask_only_is_identity() {
        let src = pixel();
        assert_eq!(apply(&src, &[255], MaskComposeMode::MaskOnly).unwrap(), src);
    }

    #[test]
    fn alternate_full_mask_clamps_to_zero() {
        let out = apply(&pixel(), &[255], MaskComposeMode::Alternate).unwrap();
        assert_eq!(out.pixel(0, 0), [0, 0, 0, 77]);
    }

    #[test]
    fn wrong_mask_length_is_refused() {
        assert_eq!(
            apply(&pixel(), &[], MaskComposeMode::MaskOnly),
            Err(Refusal::ShapeMismatch("mask"))
        );
    }
}
