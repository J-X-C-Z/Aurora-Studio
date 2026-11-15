//! PF30259 sweat removal: the layer result is the model's in-place image.
//!
//! `0x7D314C6` snapshots `_src-cache.png`, `0x5B78280` runs the model on the
//! same `cv::Mat` pointer, and `0x7D31537` stores `_dst-cache.png`; the helper
//! `0x34FDE30` only names and writes those PNGs, it does not blend. The stage
//! therefore replaces the layer with the post-inference frame. Whether the
//! model output carries its own alpha is not settled, so the source alpha is
//! kept and the caller supplies the already quantized RGB.

use crate::{Access, EffectMeta, Image, Refusal};

pub const META: &[EffectMeta] = &[EffectMeta {
    pf: 30259,
    key: "",
    label: "祛汗珠（脸部）",
    access: Access::Cpu,
    inputs: &["model output RGB (full frame)"],
}];

pub fn apply(source: &Image, model_output: &Image) -> Result<Image, Refusal> {
    source.validate()?;
    model_output
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("model output"))?;
    if (source.width, source.height) != (model_output.width, model_output.height) {
        return Err(Refusal::ShapeMismatch("model output"));
    }
    let mut output = source.clone();
    for pixel in 0..source.pixel_count() {
        let at = pixel * 4;
        output.pixels[at..at + 3].copy_from_slice(&model_output.pixels[at..at + 3]);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_rgb_replaces_the_layer_and_source_alpha_stays() {
        let source = Image::solid(2, 2, [1, 2, 3, 9]);
        let model = Image::solid(2, 2, [90, 91, 92, 255]);
        let out = apply(&source, &model).unwrap();
        assert_eq!(out.pixel(0, 0), [90, 91, 92, 9]);
        assert_eq!(out.pixel(1, 1), [90, 91, 92, 9]);
    }

    #[test]
    fn shape_mismatch_is_refused() {
        let source = Image::solid(2, 2, [0; 4]);
        let model = Image::solid(3, 2, [0; 4]);
        assert_eq!(
            apply(&source, &model),
            Err(Refusal::ShapeMismatch("model output"))
        );
        let broken = Image { width: 2, height: 2, pixels: vec![0; 3] };
        assert_eq!(
            apply(&source, &broken),
            Err(Refusal::ShapeMismatch("model output"))
        );
    }
}
