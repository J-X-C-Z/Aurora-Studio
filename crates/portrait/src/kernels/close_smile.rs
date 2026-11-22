//! PF21076 closed-mouth smile: the CloseMouth layer is a full-frame paste-back.
//!
//! `0x5E2070D` builds the quad through `0x366C4B0 → 0x366C820 → 0x3516350`
//! over the whole `(0, 0, w, h)`, gated by `(gateObj[0x778] & 1) != 0`. No
//! `GL_BLEND` constant appears locally, no mask or feather is read, so the
//! layer becomes the model's RGB when the gate is on. The outer material
//! callback (`[rax+0x10]` at `0x3516458`) could still change blending; this
//! stage therefore keeps the source alpha and is documented as local.

use crate::{Access, EffectMeta, Image, Refusal};

pub const META: &[EffectMeta] = &[EffectMeta {
    pf: 21076,
    key: "smileDegree_closeMouth",
    label: "Closed-mouth smile",
    access: Access::Cpu,
    inputs: &["model output RGB (full frame)", "paste gate flag"],
}];

/// `active` is the `gateObj[0x778] & 1` result the caller captured.
pub fn apply(source: &Image, model_output: &Image, active: bool) -> Result<Image, Refusal> {
    source.validate()?;
    if !active {
        return Ok(source.clone());
    }
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
    fn gate_off_keeps_the_source_and_on_replaces_rgb() {
        let source = Image::solid(2, 1, [1, 2, 3, 7]);
        let model = Image::solid(2, 1, [9, 8, 7, 255]);
        assert_eq!(apply(&source, &model, false).unwrap(), source);
        let on = apply(&source, &model, true).unwrap();
        assert_eq!(on.pixel(0, 0), [9, 8, 7, 7]);
        assert_eq!(on.pixel(1, 0), [9, 8, 7, 7]);
    }

    #[test]
    fn shape_mismatch_is_refused_only_when_active() {
        let source = Image::solid(2, 2, [0; 4]);
        let model = Image::solid(3, 3, [0; 4]);
        assert_eq!(apply(&source, &model, false).unwrap(), source);
        assert_eq!(
            apply(&source, &model, true),
            Err(Refusal::ShapeMismatch("model output"))
        );
    }
}
