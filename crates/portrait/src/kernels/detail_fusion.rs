//! Detail-fusion pixel backend: PF30259 / PF30209 / PF21026.
//!
//! Recovered from the installed PE the reference executable
//! (SHA256 `A84519DA489BC52E4E914AF7880EDE43E3CE58992657723D02197161832464C7`,
//! image base `0x140000000`) and the read-only source lanes
//! `p13-rows26-29-vendor-parse` / `p14-rows30-34-vendor-parse`.
//!
//! The one complete executable branch here is PF21026's min-mask consumer at
//! `0x82aafe0..0x82ab0a7`:
//! `w = min(maskA, maskB)`,
//! `delta[c] = (w*target[c] + (255-w)*base[c]) / 255 - base[c]`.
//! Raw model tensors, their five-model identity, and the ROI-sized planes plus
//! role map are explicit caller inputs; nothing is inferred. The vendor's
//! post-delta transform object
//! (`0x25d8ea0 -> 0x25d3870`, vtable `0xc59c4e8`, applied through
//! `vtable+0x10`) is not recovered and stays open.
//!
//! No registry row is exported in `META`: PF21026 is already present in
//! `skin_color::META`, and registering a second row would break the catalog's
//! unique-PF contract. `ROW34` is the candidate replacement for the controller
//! to merge.

use crate::{Access, EffectMeta, Image, Refusal};

pub const META: &[EffectMeta] = &[];

pub const ROW34: EffectMeta = EffectMeta {
    pf: 21026,
    key: "Y.removeStubble",
    label: "祛胡须",
    access: Access::Cpu,
    inputs: &[
        "five_model_ids",
        "five_output_tensors",
        "explicit_channel_bindings",
        "source_rgb",
        "strength",
    ],
};
// NOT REGISTERED: the catalog keeps PF21026 `Refused` because the post-delta
// add-ex gamma/type and the five-model semantics are still open. `ROW34` is a
// candidate for the controller, not a current availability.

// `0x82ab101..0x82ab14c`: strength gate 0.005, small-dimension scale 0.75.
pub const BEARD_STRENGTH_THRESHOLD: f32 = 0.005;
pub const SMALL_DIMENSION_LIMIT: i32 = 300;
pub const SMALL_DIMENSION_SCALE: f32 = 0.75;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeardModelSpec {
    pub model_id: u32,
    pub slot: &'static str,
    pub rgb_width: usize,
    pub rgb_height: usize,
    pub mask: bool,
}

/// Five models in registry order, from `BeardPf21026Parser`.
pub const BEARD_MODELS: [BeardModelSpec; 5] = [
    BeardModelSpec {
        model_id: 1655581403,
        slot: "+0x58",
        rgb_width: 448,
        rgb_height: 224,
        mask: false,
    },
    BeardModelSpec {
        model_id: 3378116080,
        slot: "+0x38",
        rgb_width: 768,
        rgb_height: 512,
        mask: true,
    },
    BeardModelSpec {
        model_id: 2127140479,
        slot: "+0x48",
        rgb_width: 768,
        rgb_height: 512,
        mask: true,
    },
    BeardModelSpec {
        model_id: 4083080210,
        slot: "+0x40",
        rgb_width: 512,
        rgb_height: 384,
        mask: true,
    },
    BeardModelSpec {
        model_id: 195031232,
        slot: "+0x50",
        rgb_width: 512,
        rgb_height: 384,
        mask: true,
    },
];

/// Five output names, from `BeardPf21026Parser`.
pub const BEARD_OUTPUTS: [&str; 5] = [
    "158607435",
    "2405607714",
    "3434589800",
    "2977284499",
    "2892147817",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelRole {
    RepairR,
    RepairG,
    RepairB,
    MaskA,
    MaskB,
}

pub struct BeardOutput {
    pub image: Image,
    pub delta: Vec<f32>,
    /// Strength handed to the still-open vendor transform (`vtable+0x10`).
    pub transform_strength: f32,
}

fn pixels_of(image: &Image) -> Result<usize, Refusal> {
    image.validate()?;
    Ok(image.pixel_count())
}

fn require_len(name: &'static str, actual: usize, expected: usize) -> Result<(), Refusal> {
    if actual != expected {
        return Err(Refusal::ShapeMismatch(name));
    }
    Ok(())
}

fn require_finite(name: &'static str, values: &[f32]) -> Result<(), Refusal> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange(name));
    }
    Ok(())
}

fn to_byte(value: f32) -> u8 {
    if value <= 0.0 {
        0
    } else if value >= 255.0 {
        255
    } else {
        value.round_ties_even() as u8
    }
}

/// Exact five-model identity check: ids in registry order plus the five
/// recovered output names. Missing or reordered inputs are refused.
pub fn validate_beard_identity(ids: &[u32], outputs: &[&str]) -> Result<(), Refusal> {
    require_len("model_ids", ids.len(), 5)?;
    require_len("output_names", outputs.len(), 5)?;
    for (index, spec) in BEARD_MODELS.iter().enumerate() {
        if ids[index] != spec.model_id {
            return Err(Refusal::ValueOutOfRange("model_ids"));
        }
    }
    for (index, name) in BEARD_OUTPUTS.iter().enumerate() {
        if outputs[index] != *name {
            return Err(Refusal::ValueOutOfRange("output_names"));
        }
    }
    Ok(())
}

pub fn expected_len(spec: &BeardModelSpec) -> usize {
    let planes = 3 + usize::from(spec.mask);
    planes * spec.rgb_width * spec.rgb_height
}

/// Validates the five raw tensors against the recovered native shapes. Any
/// crop/resize to the ROI is caller-owned and not synthesized here.
pub fn validate_beard_tensors(outputs: &[&[f32]]) -> Result<(), Refusal> {
    require_len("outputs", outputs.len(), 5)?;
    for (index, spec) in BEARD_MODELS.iter().enumerate() {
        require_len("model_output", outputs[index].len(), expected_len(spec))?;
        require_finite("model_output", outputs[index])?;
    }
    Ok(())
}

/// Builds the repair RGB image and two masks from explicit ROI-sized planes.
/// Each plane is one byte-value-per-pixel float32 channel at frame size.
pub struct Bound {
    pub repair: Image,
    pub mask_a: Vec<u8>,
    pub mask_b: Vec<u8>,
}

pub fn bind_planes(
    planes: &[(&[f32], ChannelRole)],
    width: u32,
    height: u32,
) -> Result<Bound, Refusal> {
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("frame"))?;
    if width == 0 || height == 0 {
        return Err(Refusal::ValueOutOfRange("frame"));
    }
    let bytes = pixels.checked_mul(4).ok_or(Refusal::ShapeMismatch("frame"))?;
    if planes.len() != 5 {
        return Err(Refusal::ShapeMismatch("bindings"));
    }
    for role in [
        ChannelRole::RepairR,
        ChannelRole::RepairG,
        ChannelRole::RepairB,
        ChannelRole::MaskA,
        ChannelRole::MaskB,
    ] {
        if planes.iter().filter(|(_, bound)| *bound == role).count() != 1 {
            return Err(Refusal::ValueOutOfRange("bindings"));
        }
    }

    for (plane, _) in planes {
        require_len("plane", plane.len(), pixels)?;
        require_finite("plane", plane)?;
    }
    let mut repair = vec![255u8; bytes];
    let mut mask_a = vec![0u8; pixels];
    let mut mask_b = vec![0u8; pixels];
    for (plane, role) in planes {
        require_len("plane", plane.len(), pixels)?;
        require_finite("plane", plane)?;
        match role {
            ChannelRole::RepairR | ChannelRole::RepairG | ChannelRole::RepairB => {
                let channel = match role {
                    ChannelRole::RepairR => 0,
                    ChannelRole::RepairG => 1,
                    _ => 2,
                };
                for p in 0..pixels {
                    repair[p * 4 + channel] = to_byte(plane[p].clamp(0.0, 1.0) * 255.0);
                }
            }
            ChannelRole::MaskA | ChannelRole::MaskB => {
                let target = if *role == ChannelRole::MaskA {
                    &mut mask_a
                } else {
                    &mut mask_b
                };
                for p in 0..pixels {
                    let value = plane[p];
                    if !(0.0..=1.0).contains(&value) {
                        return Err(Refusal::ValueOutOfRange("mask"));
                    }
                    target[p] = to_byte(value * 255.0);
                }
            }
        }
    }

    let repair = Image::new(width, height, repair).ok_or(Refusal::ShapeMismatch("repair"))?;
    Ok(Bound {
        repair,
        mask_a,
        mask_b,
    })
}

/// `0x82aafe0..0x82ab0a7`: per-channel float delta against `base`.
/// `w = min(mask_a, mask_b)`; `delta = (w*target + (255-w)*base)/255 - base`.
pub fn min_mask_delta(
    base: &Image,
    target: &Image,
    mask_a: &[u8],
    mask_b: &[u8],
) -> Result<Vec<f32>, Refusal> {
    let pixels = pixels_of(base)?;
    if base.width != target.width || base.height != target.height {
        return Err(Refusal::ShapeMismatch("target"));
    }
    pixels_of(target)?;
    require_len("mask_a", mask_a.len(), pixels)?;
    require_len("mask_b", mask_b.len(), pixels)?;

    const INV255: f32 = 1.0 / 255.0;
    let mut delta = vec![0.0f32; pixels * 3];
    for p in 0..pixels {
        let weight = mask_a[p].min(mask_b[p]) as f32;
        let inverse = 255.0 - weight;
        for c in 0..3 {
            let base_c = base.pixels[p * 4 + c] as f32;
            let target_c = target.pixels[p * 4 + c] as f32;
            delta[p * 3 + c] = (weight * target_c + inverse * base_c) * INV255 - base_c;
        }
    }
    Ok(delta)
}

/// `base + delta`, clamped to `[0,255]`; alpha is copied from `base`.
/// The vendor applies a still-open transform object to `delta` before its own
/// final conversion, so this is the reference min-mask consumer.
pub fn min_mask_composite(
    base: &Image,
    target: &Image,
    mask_a: &[u8],
    mask_b: &[u8],
) -> Result<Image, Refusal> {
    let delta = min_mask_delta(base, target, mask_a, mask_b)?;
    let mut out = base.clone();
    for (index, value) in delta.iter().enumerate() {
        let p = index / 3;
        let c = index % 3;
        out.pixels[p * 4 + c] = to_byte(base.pixels[p * 4 + c] as f32 + value);
    }
    Ok(out)
}

/// Reference stage: identity -> gate -> explicit binding -> min-mask delta.
/// The returned image visualizes the delta without the unresolved transform;
/// it is not the final vendor image. `strength` is handed to that open stage.
pub fn beard_pixels(
    base: &Image,
    ids: &[u32],
    output_names: &[&str],
    outputs: &[&[f32]],
    planes: &[(&[f32], ChannelRole)],
    strength: f32,
) -> Result<BeardOutput, Refusal> {
    pixels_of(base)?;
    if !strength.is_finite() {
        return Err(Refusal::ValueOutOfRange("strength"));
    }
    validate_beard_identity(ids, output_names)?;
    validate_beard_tensors(outputs)?;

    let width = base.width;
    let height = base.height;
    if !(strength > BEARD_STRENGTH_THRESHOLD) {
        return Ok(BeardOutput {
            image: base.clone(),
            delta: vec![0.0; base.pixel_count() * 3],
            transform_strength: 0.0,
        });
    }

    let bound = bind_planes(planes, width, height)?;
    let delta = min_mask_delta(base, &bound.repair, &bound.mask_a, &bound.mask_b)?;
    let mut image = base.clone();
    for (index, value) in delta.iter().enumerate() {
        let p = index / 3;
        let c = index % 3;
        image.pixels[p * 4 + c] = to_byte(base.pixels[p * 4 + c] as f32 + value);
    }
    let max_dimension = width.max(height) as i32;
    let scale = if max_dimension < SMALL_DIMENSION_LIMIT {
        SMALL_DIMENSION_SCALE
    } else {
        1.0
    };
    Ok(BeardOutput {
        image,
        delta,
        transform_strength: strength * scale,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_mask_delta_and_reference_image_preserve_the_documented_units() {
        let base = Image::solid(1, 1, [10, 20, 30, 77]);
        let target = Image::solid(1, 1, [110, 120, 130, 0]);
        let delta = min_mask_delta(&base, &target, &[255], &[128]).unwrap();
        for value in delta {
            assert!((value - 100.0 * 128.0 / 255.0).abs() < 1e-5);
        }
        assert_eq!(min_mask_composite(&base, &target, &[255], &[128]).unwrap().pixel(0, 0),
            [60, 70, 80, 77]);
        assert_eq!(min_mask_composite(&base, &target, &[0], &[128]).unwrap(), base);
    }

    #[test]
    fn invalid_images_and_unallocatable_bindings_refuse_before_allocation() {
        let bad = Image { width: 0, height: 1, pixels: vec![] };
        assert_eq!(min_mask_delta(&bad, &bad, &[], &[]), Err(Refusal::ShapeMismatch("image")));
        let planes: [(&[f32], ChannelRole); 5] = [
            (&[], ChannelRole::RepairR), (&[], ChannelRole::RepairG),
            (&[], ChannelRole::RepairB), (&[], ChannelRole::MaskA), (&[], ChannelRole::MaskB),
        ];
        assert!(matches!(bind_planes(&planes, u32::MAX, u32::MAX), Err(Refusal::ShapeMismatch("frame"))));
    }
}
