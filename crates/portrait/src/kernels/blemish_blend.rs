//! `0x3B12E50` final mask blend used by the blemish family after the shared
//! post-mask composition. Static disassembly only; the net that produces the
//! repaired image and the live mask is still an explicit caller input.

use crate::{Image, Refusal};

pub const BLEND_RVA: u32 = 0x3B12_E50;
pub const BLEND_CALL_SITE_RVA: u32 = 0x3B15_174;

/// `out = trunc(repaired*alpha + base*(1-alpha))`, `alpha = float(mask/255)`,
/// computed in f64 after the float mask conversion. RGB only; alpha follows
/// `base`. This is a mask-weighted lerp, not `source-over`.
pub fn blend(repaired: &Image, base: &Image, mask: &[u8]) -> Result<Image, Refusal> {
    repaired.validate()?;
    base.validate()?;
    if repaired.width != base.width || repaired.height != base.height {
        return Err(Refusal::ShapeMismatch("blend_dimensions"));
    }
    if mask.len() != base.pixel_count() {
        return Err(Refusal::ShapeMismatch("blend_mask"));
    }

    let mut out = base.clone();
    for (index, pixel) in out.pixels.chunks_exact_mut(4).enumerate() {
        let alpha = (mask[index] as f64 / 255.0) as f32;
        let inverse = 1.0 - alpha as f64;
        for channel in 0..3 {
            let source = (repaired.pixels[index * 4 + channel] as f32 * alpha) as f64;
            let destination = base.pixels[index * 4 + channel] as f64 * inverse;
            pixel[channel] = (source + destination).trunc().clamp(0.0, 255.0) as u8;
        }
        pixel[3] = base.pixels[index * 4 + 3];
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(rgba: [u8; 4]) -> Image {
        Image::solid(1, 1, rgba)
    }

    #[test]
    fn mask_endpoints_and_alpha_are_exact() {
        let base = image([100, 150, 200, 77]);
        let repaired = image([9, 9, 9, 255]);
        assert_eq!(blend(&repaired, &base, &[0]).unwrap(), base);
        assert_eq!(blend(&repaired, &base, &[255]).unwrap().pixel(0, 0), [9, 9, 9, 77]);
        assert_eq!(blend(&image([200, 150, 100, 255]), &image([10, 20, 30, 77]), &[128])
            .unwrap().pixel(0, 0), [105, 85, 65, 77]);
    }

    #[test]
    fn malformed_inputs_are_refused() {
        let base = image([1, 2, 3, 4]);
        assert_eq!(blend(&image([0; 4]), &base, &[]), Err(Refusal::ShapeMismatch("blend_mask")));
        assert_eq!(blend(&Image::solid(2, 1, [0; 4]), &base, &[0; 2]),
            Err(Refusal::ShapeMismatch("blend_dimensions")));
    }
}
