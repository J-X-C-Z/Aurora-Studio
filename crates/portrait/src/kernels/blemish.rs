//! Row 25 PF0 blemish/acne producer (21-slot mask contract).
//!
//! Pinned source artifacts (read-only inputs to this port):
//!   p04b-blemish-pf0-e2e/PATCH-CANDIDATE/Pf0BlemishProducer.cs
//!   p04b-blemish-pf0-e2e/PATCH-CANDIDATE/Pf0SameJobReceiver.cs
//!   p28-priority11-25-face-parse/source/P28FaceMath.cs (P28Row25Pf0)
//!
//! Recovered anchors:
//!   final composition 0x5B8CEA0 (two byte laws, int trunc then clamp 0..255)
//!   pipeline run 0x3B148F0, blend 0x3B12E50 (blend law NOT recovered)
//!   21 slots x 0x88 = 0xB28; slot planes 0..16 = faceParsingV2 planes,
//!   17 = sat8(0,13,14,15,16), 18 = sat8(2,7,11,12),
//!   19 = sat8(4,5), 20 = sat8(9,10).
//!
//! Slots 17-20 are recovered faceParsing descriptor bundles; they are not the
//! composer's C source. C is selected by `0x5B8EF20` through a `0x548`-stride
//! face-part record lookup (`0x2ECE510`, key `+0x28`); its pixel identity
//! remains OPEN (slot18 != C).
//!
//! The two composition laws are the shared `mask_compose` terminal:

//!   product  -> MaskComposeMode::MaskOnly   (dst * source / 255)
//!   additive -> MaskComposeMode::Alternate  (dst * (1 - 2*source/255))
//! This module calls that shared kernel instead of re-implementing the byte law.
//!
//! No model inference, skin detection, repair worker or heuristic is performed.
//! Detector, acne, blemish and repair outputs are explicit same-job inputs; a
//! missing or misshapen one is a typed refusal.
//!
//! Vendor-final status: the blemish TSAI output key `0x88BD92AA` still has no
//! unique pixcook numeric-net binding. That part stays Refused and is NOT
//! counted as complete; only the recovered producer/blend is callable.

use crate::kernels::mask_compose::{self, MaskComposeMode};
use crate::{Access, EffectMeta, Image, Refusal};

pub const SLOT_COUNT: usize = 21;
pub const SLOT_RECORD_STRIDE: usize = 0x88;
pub const SLOT_VECTOR_BYTES: usize = SLOT_COUNT * SLOT_RECORD_STRIDE;
pub const COMPOSITION_RVA: u32 = 0x5B8C_EA0;
pub const PIPELINE_RUN_RVA: u32 = 0x3B1_48F0;
pub const MASK_SCALE: f32 = 255.0;
pub const DETECTOR_NET_ID: u32 = 4_252_205_951;
pub const BLEMISH_OUTPUT_KEY: u32 = 0x88BD_92AA;
pub const DETECTOR_OUTPUT_KEY: u32 = 0xBF01_86FA;

/// Vendor-final model identity is not closed, so it is reported as refused.
pub const VENDOR_FINAL_STATUS: &str =
    "REFUSED: blemish output key 0x88BD92AA has no unique pixcook net binding";

const INPUTS: [&str; 15] = [
    "job_id",
    "source_sha256",
    "face_id",
    "roi",
    "skin_mask",
    "hair_mask",
    "instance_mask",
    "slot_planes_21",
    "same_job_binding_sha256",
    "detector_float32",
    "acne_mask",
    "blemish_model_output",
    "repair_model_output",
    "composition_flag",
    "spotless_degree",
];

/// One registry row. `spotless` is the recovered settings key; the model side
/// is not promoted to a vendor-final claim.
pub const META: &[EffectMeta] = &[EffectMeta {
    pf: 0,
    key: "spotless",
    label: "祛斑祛痘",
    access: Access::Cpu,
    inputs: &INPUTS,
}];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pf0Law {
    AdditiveContrast,
    ProductBlend,
}

impl Pf0Law {
    /// `record +0x3D5` flag: 0 product, 1 additive. Anything else refuses.
    pub fn from_flag(flag: u8) -> Result<Self, Refusal> {
        match flag {
            0 => Ok(Pf0Law::ProductBlend),
            1 => Ok(Pf0Law::AdditiveContrast),
            _ => Err(Refusal::ValueOutOfRange("composition_flag")),
        }
    }

    fn mode(self) -> MaskComposeMode {
        match self {
            Pf0Law::ProductBlend => MaskComposeMode::MaskOnly,
            Pf0Law::AdditiveContrast => MaskComposeMode::Alternate,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Roi {
    fn pixels(self) -> Result<usize, Refusal> {
        (self.width as usize)
            .checked_mul(self.height as usize)
            .filter(|pixels| *pixels > 0)
            .ok_or(Refusal::ShapeMismatch("roi"))
    }

    fn inside(self, width: u32, height: u32) -> bool {
        width > 0
            && height > 0
            && self.width > 0
            && self.height > 0
            && self.x as u64 + self.width as u64 <= width as u64
            && self.y as u64 + self.height as u64 <= height as u64
    }
}

/// 21 one-byte planes of exactly `pixels` bytes each.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlotMasks {
    planes: Vec<Vec<u8>>,
    pixels: usize,
}

impl SlotMasks {
    /// Recovered 17 -> 21 expansion. `planes17` is already quantized to bytes.
    pub fn from_face_parsing_17(planes17: &[u8], pixels: usize) -> Result<Self, Refusal> {
        if pixels == 0 || planes17.len() != 17usize.saturating_mul(pixels) {
            return Err(Refusal::ShapeMismatch("face_parsing_17_planes"));
        }
        let mut planes = Vec::with_capacity(SLOT_COUNT);
        for plane in 0..17 {
            planes.push(planes17[plane * pixels..(plane + 1) * pixels].to_vec());
        }
        for members in [&[0usize, 13, 14, 15, 16][..], &[2, 7, 11, 12][..], &[4, 5][..], &[9, 10][..]] {
            let mut bundle = vec![0u8; pixels];
            for i in 0..pixels {
                let sum: u16 = members.iter().map(|&plane| planes17[plane * pixels + i] as u16).sum();
                bundle[i] = sum.min(255) as u8;
            }
            planes.push(bundle);
        }
        Ok(Self { planes, pixels })
    }

    /// Explicit same-job 21-slot input for callers that already hold all slots.
    pub fn explicit(planes: Vec<Vec<u8>>, pixels: usize) -> Result<Self, Refusal> {
        if pixels == 0 || planes.len() != SLOT_COUNT || planes.iter().any(|p| p.len() != pixels) {
            return Err(Refusal::ShapeMismatch("slot_planes"));
        }
        Ok(Self { planes, pixels })
    }

    pub fn plane(&self, slot: usize) -> Option<&[u8]> {
        self.planes.get(slot).map(Vec::as_slice)
    }

    pub fn pixels(&self) -> usize {
        self.pixels
    }
}

pub struct Frame<'a> {
    pub job_id: &'a str,
    pub source_sha256: &'a str,
    pub face_id: &'a str,
    pub image_width: u32,
    pub image_height: u32,
    pub roi: Roi,
    pub skin_mask: &'a [u8],
    pub hair_mask: &'a [u8],
    pub instance_mask: &'a [u8],
    pub slots: &'a SlotMasks,
    pub binding_sha256: &'a [u8],
    pub detector_output: &'a [u8],
    pub acne_mask: &'a [u8],
    pub blemish_model_output: &'a [u8],
    pub repair_model_output: &'a [u8],
    pub composition_flag: u8,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Output {
    pub law: Pf0Law,
    pub detector: Vec<u8>,
    pub composite: Vec<u8>,
    pub changed: bool,
    pub slot_vector_bytes: usize,
    pub slot_record_stride: usize,
    pub composition_rva: u32,
}

/// Terminal stage: exactly the shared `mask_compose` kernel. RGB is scaled,
/// alpha is preserved byte-for-byte.
pub fn compose(target: &Image, source_mask: &[u8], law: Pf0Law) -> Result<Image, Refusal> {
    mask_compose::apply(target, source_mask, law.mode())
}

/// `float32 LE` detector plane, Tanh range `-1..1`, quantized to `0..255`.
pub fn decode_detector(detector_output: &[u8], pixels: usize) -> Result<Vec<u8>, Refusal> {
    if pixels == 0 || detector_output.len() != pixels.saturating_mul(4) {
        return Err(Refusal::ShapeMismatch("detector_output"));
    }
    let mut plane = vec![0u8; pixels];
    for i in 0..pixels {
        let value = f32::from_le_bytes([
            detector_output[i * 4],
            detector_output[i * 4 + 1],
            detector_output[i * 4 + 2],
            detector_output[i * 4 + 3],
        ]);
        if !value.is_finite() {
            return Err(Refusal::ValueOutOfRange("detector_non_finite"));
        }
        let scaled = ((value.clamp(-1.0, 1.0) + 1.0) * 127.5) as i32;
        plane[i] = scaled.clamp(0, 255) as u8;
    }
    Ok(plane)
}

fn require_text(value: &str, field: &'static str) -> Result<(), Refusal> {
    if value.trim().is_empty() {
        return Err(Refusal::MissingInput(field));
    }
    Ok(())
}

fn require_len(value: &[u8], expected: usize, field: &'static str) -> Result<(), Refusal> {
    if value.len() != expected {
        return Err(Refusal::ShapeMismatch(field));
    }
    Ok(())
}

/// Recovered producer. The detector plane is the composition target; the mask is
/// the acne mask (product) or blemish model output (additive). This matches the
/// R96/R99 receiver, not the older P28 shortcut that reused the acne mask as
/// the target for both laws.
pub fn produce(frame: &Frame<'_>) -> Result<Output, Refusal> {
    require_text(frame.job_id, "job_id")?;
    require_text(frame.source_sha256, "source_sha256")?;
    require_text(frame.face_id, "face_id")?;
    if !frame.roi.inside(frame.image_width, frame.image_height) {
        return Err(Refusal::ShapeMismatch("roi"));
    }
    let pixels = frame.roi.pixels()?;

    if frame.binding_sha256.len() != 32 || frame.binding_sha256.iter().all(|&byte| byte == 0) {
        return Err(Refusal::ShapeMismatch("same_job_binding_sha256"));
    }
    let law = Pf0Law::from_flag(frame.composition_flag)?;

    if frame.slots.pixels() != pixels {
        return Err(Refusal::ShapeMismatch("slot_planes"));
    }
    require_len(frame.skin_mask, pixels, "skin_mask")?;
    require_len(frame.hair_mask, pixels, "hair_mask")?;
    require_len(frame.instance_mask, pixels, "instance_mask")?;
    require_len(frame.acne_mask, pixels, "acne_mask")?;
    require_len(frame.blemish_model_output, pixels, "blemish_model_output")?;
    require_len(frame.repair_model_output, pixels.saturating_mul(3), "repair_model_output")?;

    let detector = decode_detector(frame.detector_output, pixels)?;
    let mask = match law {
        Pf0Law::ProductBlend => frame.acne_mask,
        Pf0Law::AdditiveContrast => frame.blemish_model_output,
    };

    let mut target = Vec::with_capacity(pixels.saturating_mul(4));
    for &value in &detector {
        target.extend_from_slice(&[value, value, value, 255]);
    }
    let target = Image::new(frame.roi.width, frame.roi.height, target)
        .ok_or(Refusal::ShapeMismatch("detector_output"))?;
    let composed = compose(&target, mask, law)?;

    let composite: Vec<u8> = composed.pixels.chunks_exact(4).map(|px| px[0]).collect();
    let changed = composite.iter().any(|&value| value != 0);
    Ok(Output {
        law,
        detector,
        composite,
        changed,
        slot_vector_bytes: SLOT_VECTOR_BYTES,
        slot_record_stride: SLOT_RECORD_STRIDE,
        composition_rva: COMPOSITION_RVA,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planes17() -> Vec<u8> {
        let mut planes = vec![0u8; 17 * 2];
        for (plane, (p0, p1)) in [
            (10, 1),
            (20, 2),
            (30, 3),
            (0, 4),
            (40, 5),
            (50, 6),
            (0, 7),
            (60, 8),
            (0, 9),
            (70, 10),
            (0, 11),
            (80, 12),
            (200, 13),
            (90, 14),
            (100, 15),
            (110, 16),
            (120, 17),
        ]
        .into_iter()
        .enumerate()
        {
            planes[plane * 2] = p0;
            planes[plane * 2 + 1] = p1;
        }
        planes
    }

    fn slots() -> SlotMasks {
        SlotMasks::from_face_parsing_17(&planes17(), 2).unwrap()
    }

    fn frame<'a>(slots: &'a SlotMasks, detector: &'a [u8], acne: &'a [u8],
                 blemish: &'a [u8], repair: &'a [u8], flag: u8) -> Frame<'a> {
        Frame {
            job_id: "job",
            source_sha256: "aa",
            face_id: "face0",
            image_width: 2,
            image_height: 1,
            roi: Roi { x: 0, y: 0, width: 2, height: 1 },
            skin_mask: &[255, 255],
            hair_mask: &[0, 0],
            instance_mask: &[255, 255],
            slots,
            binding_sha256: &[1; 32],
            detector_output: detector,
            acne_mask: acne,
            blemish_model_output: blemish,
            repair_model_output: repair,
            composition_flag: flag,
        }
    }

    #[test]
    fn slot_expansion_uses_saturating_bundles() {
        let slots = slots();
        assert_eq!(slots.pixels(), 2);
        assert_eq!(slots.plane(12).unwrap(), &[200, 13]);
        assert_eq!(slots.plane(17).unwrap(), &[255, 63]);
        assert_eq!(slots.plane(18).unwrap(), &[255, 36]);
        assert_eq!(slots.plane(19).unwrap(), &[90, 11]);
        assert_eq!(slots.plane(20).unwrap(), &[70, 21]);

        let saturated = vec![255u8; 17];
        let slots = SlotMasks::from_face_parsing_17(&saturated, 1).unwrap();
        assert_eq!(slots.plane(17).unwrap(), &[255]);
        assert_eq!(SlotMasks::explicit(vec![vec![0u8; 1]; 21], 1).unwrap().plane(0).unwrap(), &[0]);
    }

    #[test]
    fn product_law_positive_identity_and_alpha() {
        let target = Image::solid(1, 1, [200, 150, 100, 77]);
        let out = compose(&target, &[128], Pf0Law::ProductBlend).unwrap();
        assert_eq!(out.pixel(0, 0), [100, 75, 50, 77]);
        assert_eq!(compose(&target, &[255], Pf0Law::ProductBlend).unwrap(), target);
        assert_eq!(out.pixel(0, 0)[3], 77);
    }

    #[test]
    fn additive_law_positive_identity_and_alpha() {
        let target = Image::solid(1, 1, [200, 150, 100, 77]);
        assert_eq!(compose(&target, &[0], Pf0Law::AdditiveContrast).unwrap(), target);
        let out = compose(&target, &[64], Pf0Law::AdditiveContrast).unwrap();
        assert_eq!(out.pixel(0, 0), [99, 74, 49, 77]);
    }

    #[test]
    fn detector_decode_clamps_and_refuses_non_finite() {
        let mut raw = Vec::new();
        raw.extend_from_slice(&0.5f32.to_le_bytes());
        raw.extend_from_slice(&(-2.0f32).to_le_bytes());
        assert_eq!(decode_detector(&raw, 2).unwrap(), vec![191, 0]);
        let mut bad = Vec::new();
        bad.extend_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(decode_detector(&bad, 1), Err(Refusal::ValueOutOfRange("detector_non_finite")));
    }

    #[test]
    fn produce_product_and_additive() {
        let slots = slots();
        let mut product = Vec::new();
        product.extend_from_slice(&0.5f32.to_le_bytes());
        product.extend_from_slice(&0.0f32.to_le_bytes());
        let acne = [255u8, 0];
        let blemish = [0u8, 64];
        let repair = [0u8; 6];
        let out = produce(&frame(&slots, &product, &acne, &blemish, &repair, 0)).unwrap();
        assert_eq!(out.law, Pf0Law::ProductBlend);
        assert_eq!(out.detector, vec![191, 127]);
        assert_eq!(out.composite, vec![191, 0]);
        assert!(out.changed);
        assert_eq!(out.slot_vector_bytes, 0xB28);
        assert_eq!(out.slot_record_stride, 0x88);
        assert_eq!(out.composition_rva, COMPOSITION_RVA);

        let additive = produce(&frame(&slots, &product, &acne, &blemish, &repair, 1)).unwrap();
        assert_eq!(additive.law, Pf0Law::AdditiveContrast);
        assert_eq!(additive.composite, vec![191, 63]);
    }

    #[test]
    fn shape_and_flag_refusals() {
        let slots = slots();
        let detector = [0u8; 8];
        let ok = [0u8; 2];
        let repair = [0u8; 6];
        let mut f = frame(&slots, &detector, &ok, &ok, &repair, 0);
        f.skin_mask = &[0u8; 1];
        assert_eq!(produce(&f), Err(Refusal::ShapeMismatch("skin_mask")));

        let f = frame(&slots, &detector, &ok, &ok, &repair, 2);
        assert_eq!(produce(&f), Err(Refusal::ValueOutOfRange("composition_flag")));

        let mut f = frame(&slots, &detector, &ok, &ok, &repair, 0);
        f.roi = Roi { x: 1, y: 0, width: 2, height: 1 };
        assert_eq!(produce(&f), Err(Refusal::ShapeMismatch("roi")));

        let mut f = frame(&slots, &detector, &ok, &ok, &repair, 0);
        f.binding_sha256 = &[0u8; 8];
        assert_eq!(produce(&f), Err(Refusal::ShapeMismatch("same_job_binding_sha256")));
    }

    #[test]
    fn explicit_inputs_refuse() {
        assert_eq!(SlotMasks::explicit(vec![vec![0u8; 2]; 20], 2),
            Err(Refusal::ShapeMismatch("slot_planes")));
        assert_eq!(SlotMasks::from_face_parsing_17(&[0u8; 33], 2),
            Err(Refusal::ShapeMismatch("face_parsing_17_planes")));
        let target = Image::solid(1, 1, [10, 20, 30, 40]);
        assert_eq!(compose(&target, &[], Pf0Law::ProductBlend),
            Err(Refusal::ShapeMismatch("mask")));
        assert_eq!(Pf0Law::from_flag(9), Err(Refusal::ValueOutOfRange("composition_flag")));
    }

    #[test]
    fn meta_is_one_callable_pf0_row() {
        assert_eq!(META.len(), 1);
        assert_eq!(META[0].pf, 0);
        assert_eq!(META[0].access, Access::Cpu);
        assert_eq!(META[0].key, "spotless");
        assert!(VENDOR_FINAL_STATUS.starts_with("REFUSED"));
    }
}
