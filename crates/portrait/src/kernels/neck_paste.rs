//! PF90015 final paste-back, recovered from `0x7D09690` → `0x366C4B0`.
//!
//! The neck model's `1x3x512x640` colour tail is `127.5*x + 127.5`, i.e. final
//! RGB rather than a delta. The quad that draws it carries no colour/alpha and
//! the path enables no `GL_BLEND`, so the layer is overwritten: the ROI pixels
//! become the repair image. The byte conversion of the tail is upstream and
//! stays the caller's job, so this stage takes the already quantized repair.

use crate::{Access, EffectMeta, Image, Refusal};

pub const META: &[EffectMeta] = &[EffectMeta {
    pf: 90015,
    key: "",
    label: "Neck wrinkles",
    access: Access::Cpu,
    inputs: &["repair RGB", "paste x/y"],
}];

/// Overwrites `source[x.., y..]` with `repair`; source alpha is kept because
/// the paste-back carries no alpha channel of its own.
pub fn paste(source: &Image, repair: &Image, x: u32, y: u32) -> Result<Image, Refusal> {
    source.validate()?;
    repair.validate().map_err(|_| Refusal::ShapeMismatch("repair"))?;
    let (right, bottom) = (x as u64 + repair.width as u64, y as u64 + repair.height as u64);
    if x >= source.width || y >= source.height
        || right > source.width as u64
        || bottom > source.height as u64
    {
        return Err(Refusal::ValueOutOfRange("paste rect"));
    }
    let mut output = source.clone();
    for row in 0..repair.height as usize {
        let from = row * repair.width as usize * 4;
        let to = ((y as usize + row) * source.width as usize + x as usize) * 4;
        let bytes = repair.width as usize * 4;
        output.pixels[to..to + bytes].copy_from_slice(&repair.pixels[from..from + bytes]);
        for column in 0..repair.width as usize {
            let alpha = source.pixels[to + column * 4 + 3];
            output.pixels[to + column * 4 + 3] = alpha;
        }
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paste_overwrites_the_rect_and_keeps_source_alpha() {
        let source = Image::solid(4, 4, [10, 20, 30, 200]);
        let repair = Image::solid(2, 2, [90, 91, 92, 0]);
        let out = paste(&source, &repair, 1, 1).unwrap();
        assert_eq!(out.pixel(1, 1), [90, 91, 92, 200]);
        assert_eq!(out.pixel(2, 2), [90, 91, 92, 200]);
        assert_eq!(out.pixel(0, 0), [10, 20, 30, 200]);
        assert_eq!(out.pixel(3, 3), [10, 20, 30, 200]);
    }

    #[test]
    fn full_frame_repair_replaces_every_pixel() {
        let source = Image::solid(3, 2, [1, 2, 3, 4]);
        let repair = Image::solid(3, 2, [9, 8, 7, 255]);
        let out = paste(&source, &repair, 0, 0).unwrap();
        for x in 0..3 {
            for y in 0..2 {
                assert_eq!(out.pixel(x, y), [9, 8, 7, 4]);
            }
        }
    }

    #[test]
    fn out_of_bounds_and_malformed_inputs_are_refused() {
        let source = Image::solid(4, 4, [0; 4]);
        let repair = Image::solid(2, 2, [1; 4]);
        assert_eq!(
            paste(&source, &repair, 3, 0),
            Err(Refusal::ValueOutOfRange("paste rect"))
        );
        assert_eq!(
            paste(&source, &repair, 0, 3),
            Err(Refusal::ValueOutOfRange("paste rect"))
        );
        let broken = Image { width: 2, height: 2, pixels: vec![0; 3] };
        assert_eq!(
            paste(&source, &broken, 0, 0),
            Err(Refusal::ShapeMismatch("repair"))
        );
    }
}
