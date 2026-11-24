//! Default same-frame inputs for PF90002 (face moisture) and PF90010 (body
//! moisture), built from the cached `PortraitMasks` people parser.
//!
//! Parser masks are never reused for these rows: `model_inputs.parsing` is a
//! FaceParsing skin crop, and `portrait_stage.rs::non_skin_rows_never_reuse_parser_masks`
//! requires PF90002/PF90010 to refuse without their own same-job inputs.
//!
//! Verified against `crates/portrait/src/kernels/moisture.rs`:
//! - `mask_rgba` is `W*H*4` u8 over the rendered frame; `blur_alpha` is exactly
//!   `W*H` u8. `mask_rgba` is read through `MaskChannel`; `blur_alpha` is the
//!   blurred fourth component (bZ stand-in, producer live-only per R76/s27).
//! - `SkinMap` is a *colour* LUT: `sample(x=source.B, y=source.R)` reads the red
//!   channel of a `skin_map_width x skin_map_height` table. It is not the
//!   FaceParsing grid and the kernel fixes no table size; only
//!   `len == width * height` and finiteness matter. We resample the face plane
//!   into that table as a documented spatial stand-in for the uncaptured bY.
//! - `hardlight` is sampled at `(reference, 0.5)`; `beauty` at `(Y, 0.5)`.
//!   `FaceParams`/`Lut1D`/`SkinMap` have literal constructors only. The neutral
//!   transfer `degree = (ui/100 - pivot) * gain + pivot` is identity with
//!   `pivot = 0.0, gain = 1.0`; the 3-tap `[0, 0.5, 1]` ramp is the neutral
//!   fixture (exact at 0/0.5/1, GL LINEAR between), so no vendor tone curve is
//!   invented.
//! - `body_apply` needs full-frame `weight_skin`/`weight_dog` even for a partial
//!   `roi`, and `degree` in `[0, 1]`. The row gate is `BODY_RENDER_GATE = 0.01`.
//!
//! Mask provenance: `PortraitMasks` comes from `aimask::people_masks` at 512x512
//! and is cached by the mock renderer (`src.portrait_masks()`); `mask_u8` /
//! `mask_f32` resample it to the rendered frame. Missing planes are refused, not
//! zero-filled.

use aurora_portrait::kernels::moisture::MaskChannel;
use aurora_portrait::{Image, Refusal};

use crate::portrait_stage::{
    FaceInputBundle, MaskKind, MoistureBodyInputs, MoistureFaceInputs, PortraitMasks,
    ProducerInputs,
};

/// Resampled face plane used as the neutral bY stand-in (n x n samples).
pub const MOISTURE_SKIN_MAP_SIDE: u32 = 16;
/// Neutral 3-tap ramp for the hardlight/beauty LUT stand-ins (not vendor data).
pub const MOISTURE_LUT: [f32; 3] = [0.0, 0.5, 1.0];
/// Recovered neutral face transfer (`degree == ui / 100`).
pub const MOISTURE_FACE_PIVOT: f32 = 0.0;
pub const MOISTURE_FACE_GAIN: f32 = 1.0;
/// The face plane is mirrored into RGB with alpha forced to 255, so select R.
pub const MOISTURE_FACE_CHANNEL: MaskChannel = MaskChannel::R;
/// PF90010 still gates on its captured control; degree 1.0 is the full value.
pub const MOISTURE_BODY_DEGREE: f32 = 1.0;

#[derive(Clone, Debug, PartialEq)]
pub struct OwnedMoistureFaceInputs {
    pub width: u32,
    pub height: u32,
    pub mask_rgba: Vec<u8>,
    pub blur_alpha: Vec<u8>,
    pub channel: MaskChannel,
    pub skin_map_width: u32,
    pub skin_map_height: u32,
    pub skin_map: Vec<f32>,
    pub hardlight: Vec<f32>,
    pub beauty: Vec<f32>,
    pub pivot: f32,
    pub gain: f32,
}

impl OwnedMoistureFaceInputs {
    pub fn as_borrowed(&self) -> MoistureFaceInputs<'_> {
        MoistureFaceInputs {
            width: self.width,
            height: self.height,
            mask_rgba: &self.mask_rgba,
            blur_alpha: &self.blur_alpha,
            channel: self.channel,
            skin_map_width: self.skin_map_width,
            skin_map_height: self.skin_map_height,
            skin_map: &self.skin_map,
            hardlight: &self.hardlight,
            beauty: &self.beauty,
            pivot: self.pivot,
            gain: self.gain,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OwnedMoistureBodyInputs {
    pub width: u32,
    pub height: u32,
    pub weight_skin: Vec<f32>,
    pub weight_dog: Vec<f32>,
    pub beauty: Vec<f32>,
    pub degree: f32,
    pub roi: (u32, u32, u32, u32),
}

impl OwnedMoistureBodyInputs {
    pub fn as_borrowed(&self) -> MoistureBodyInputs<'_> {
        MoistureBodyInputs {
            width: self.width,
            height: self.height,
            weight_skin: &self.weight_skin,
            weight_dog: &self.weight_dog,
            beauty: &self.beauty,
            degree: self.degree,
            roi: self.roi,
        }
    }
}

impl FaceInputBundle {
    /// Builds and attaches PF90002/PF90010 inputs for this exact frame from the
    /// people-parser masks. Refuses on a ticket/digest mismatch, on empty masks
    /// (`masks.width == 0`), or when the Face/Body plane is absent.
    pub fn attach_moisture_inputs(
        &self,
        source: &Image,
        masks: &PortraitMasks,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        source
            .validate()
            .map_err(|_| Refusal::ShapeMismatch("PF90002 source frame"))?;
        if masks.width == 0 || masks.height == 0 {
            return Err(Refusal::MissingInput("PF90002/PF90010 portrait masks"));
        }
        let width = source.width;
        let height = source.height;
        let pixels = source.pixel_count();

        let face_plane = masks
            .mask_u8(MaskKind::Face, width, height, crate::NormRect::FULL)
            .ok_or(Refusal::MissingInput("PF90002 face mask"))?;
        if face_plane.len() != pixels {
            return Err(Refusal::ShapeMismatch("PF90002 face mask"));
        }
        let mut mask_rgba = Vec::with_capacity(pixels.saturating_mul(4));
        for &value in &face_plane {
            mask_rgba.extend_from_slice(&[value, value, value, 255]);
        }
        let skin_side = MOISTURE_SKIN_MAP_SIDE;
        let skin_map = masks
            .mask_f32(MaskKind::Face, skin_side, skin_side, crate::NormRect::FULL)
            .ok_or(Refusal::MissingInput("PF90002 skin map"))?;
        if skin_map.len() != (skin_side as usize) * (skin_side as usize) {
            return Err(Refusal::ShapeMismatch("PF90002 skin map"));
        }

        let weight_skin = masks
            .mask_f32(MaskKind::Body, width, height, crate::NormRect::FULL)
            .ok_or(Refusal::MissingInput("PF90010 body mask"))?;
        if weight_skin.len() != pixels {
            return Err(Refusal::ShapeMismatch("PF90010 body mask"));
        }

        let face = OwnedMoistureFaceInputs {
            width,
            height,
            mask_rgba,
            // bZ producer is live-only; the resampled face plane is the
            // documented stand-in, not a captured blur plane.
            blur_alpha: face_plane,
            channel: MOISTURE_FACE_CHANNEL,
            skin_map_width: skin_side,
            skin_map_height: skin_side,
            skin_map,
            hardlight: MOISTURE_LUT.to_vec(),
            beauty: MOISTURE_LUT.to_vec(),
            pivot: MOISTURE_FACE_PIVOT,
            gain: MOISTURE_FACE_GAIN,
        };
        // Vendor dog weight is not captured; reuse the skin weight so the row
        // stays a real (mask-gated) pass and the 3-tap neutral LUT is not a
        // fabricated highlight response.
        let body = OwnedMoistureBodyInputs {
            width,
            height,
            weight_dog: weight_skin.clone(),
            weight_skin,
            beauty: MOISTURE_LUT.to_vec(),
            degree: MOISTURE_BODY_DEGREE,
            roi: (0, 0, width, height),
        };
        producer.moisture_face = Some(face);
        producer.moisture_body = Some(body);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const JOB: u64 = 41;

    fn source() -> Image {
        Image::solid(8, 8, [10, 20, 30, 255])
    }

    fn bundle(job: u64, source: &Image) -> FaceInputBundle {
        use crate::portrait_stage::{
            C1FaceDetection, C1FaceMesh, FaceLandmarkProducerInputs, FaceModelIdentity,
            FaceNormalization,
        };
        let normalization = FaceNormalization {
            channel_order: "BGR",
            mean: [127.5; 3],
            scale: [128.0; 3],
        };
        let model = FaceModelIdentity {
            model_id: "test",
            sha256: [0; 32],
        };
        FaceInputBundle {
            job_id: job,
            source_size: [source.width, source.height],
            source_sha256: crate::sha256_32(&source.pixels),
            detection: C1FaceDetection {
                score: 0.9,
                landmarks_source: [[2.0, 4.0]; 5],
                source_to_input: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                input_size: [640, 640],
                normalization,
                provider: "cpu",
                model,
            },
            mesh: C1FaceMesh {
                score: 0.9,
                landmarks_source: vec![[2.0, 2.0, 0.0]],
                source_to_crop: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                crop_side: 4.0,
                input_size: [256, 256],
                normalization,
                provider: "cpu",
                model,
            },
            roi_xywh: [2.0, 2.0, 4.0, 4.0],
            roi_normalized_xywh: [0.25, 0.25, 0.5, 0.5],
            points17: Vec::new(),
            points142: Vec::new(),
            points194: vec![[4.0, 4.0]; 194],
            points414: vec![[4.0, 4.0]; 414],
            landmark_producer: FaceLandmarkProducerInputs {
                resource_id: 1,
                provider: "test".to_string(),
                model_sha256: [0; 32],
                source_sha256: [0; 32],
                source_size: [source.width, source.height],
                points142: Vec::new(),
                crop_bgr: Vec::new(),
                crop_sha256: [0; 32],
                raw52: Vec::new(),
                points52: Vec::new(),
                points194: Vec::new(),
                fit_pre: [[0.0; 3]; 2],
                fit_inverse: [[0.0; 3]; 2],
            },
            model_inputs: None,
        }
    }

    #[test]
    fn parser_masks_fill_the_kernel_plane_contract() {
        let source = source();
        let masks = PortraitMasks::solid(1.0);
        let mut producer = ProducerInputs::default();
        bundle(JOB, &source)
            .attach_moisture_inputs(&source, &masks, &mut producer)
            .unwrap();

        let face = producer.moisture_face.as_ref().unwrap();
        assert_eq!((face.width, face.height), (8, 8));
        assert_eq!(face.mask_rgba.len(), 8 * 8 * 4);
        assert_eq!(face.blur_alpha.len(), 8 * 8);
        assert!(face.mask_rgba.iter().all(|&value| value == 255));
        assert!(face.blur_alpha.iter().all(|&v| v == 255));
        assert_eq!(face.channel, MaskChannel::R);
        assert_eq!(
            (face.skin_map_width, face.skin_map_height),
            (MOISTURE_SKIN_MAP_SIDE, MOISTURE_SKIN_MAP_SIDE)
        );
        assert_eq!(face.skin_map.len(), 256);
        assert!(face.skin_map.iter().all(|v| *v == 1.0));
        assert_eq!(face.hardlight, MOISTURE_LUT.to_vec());
        assert_eq!(face.beauty, MOISTURE_LUT.to_vec());
        assert_eq!((face.pivot, face.gain), (0.0, 1.0));
        assert!(face
            .hardlight
            .iter()
            .chain(&face.beauty)
            .all(|v| v.is_finite()));

        let body = producer.moisture_body.as_ref().unwrap();
        assert_eq!((body.width, body.height), (8, 8));
        assert_eq!(body.weight_skin.len(), 64);
        assert_eq!(body.weight_dog.len(), 64);
        assert!(body.weight_skin.iter().all(|v| *v == 1.0));
        assert_eq!(body.weight_skin, body.weight_dog);
        assert_eq!(body.beauty, MOISTURE_LUT.to_vec());
        assert_eq!(body.degree, 1.0);
        assert_eq!(body.roi, (0, 0, 8, 8));
    }

    #[test]
    fn as_borrowed_keeps_dimension_parity() {
        let source = source();
        let masks = PortraitMasks::solid(0.5);
        let mut producer = ProducerInputs::default();
        bundle(JOB, &source)
            .attach_moisture_inputs(&source, &masks, &mut producer)
            .unwrap();

        let face = producer.moisture_face.as_ref().unwrap();
        let borrowed = face.as_borrowed();
        assert_eq!((borrowed.width, borrowed.height), (face.width, face.height));
        assert_eq!(borrowed.mask_rgba.len(), face.mask_rgba.len());
        assert_eq!(borrowed.blur_alpha.len(), face.blur_alpha.len());
        assert_eq!(borrowed.skin_map.len(), face.skin_map.len());
        assert_eq!(borrowed.skin_map_width, face.skin_map_width);
        assert_eq!(borrowed.skin_map_height, face.skin_map_height);
        assert_eq!(borrowed.hardlight.len(), face.hardlight.len());
        assert_eq!(borrowed.beauty.len(), face.beauty.len());
        assert_eq!(borrowed.channel, face.channel);

        let body = producer.moisture_body.as_ref().unwrap();
        let borrowed = body.as_borrowed();
        assert_eq!((borrowed.width, borrowed.height), (body.width, body.height));
        assert_eq!(borrowed.weight_skin.len(), body.weight_skin.len());
        assert_eq!(borrowed.weight_dog.len(), body.weight_dog.len());
        assert_eq!(borrowed.roi, body.roi);
    }

    #[test]
    fn missing_masks_are_refused_not_zero_filled() {
        let source = source();
        let mut producer = ProducerInputs::default();
        let empty = PortraitMasks::default();
        assert_eq!(
            bundle(JOB, &source).attach_moisture_inputs(&source, &empty, &mut producer),
            Err(Refusal::MissingInput("PF90002/PF90010 portrait masks"))
        );
        assert!(producer.moisture_face.is_none());

        let no_face = PortraitMasks {
            face: None,
            ..PortraitMasks::solid(1.0)
        };
        assert_eq!(
            bundle(JOB, &source).attach_moisture_inputs(&source, &no_face, &mut producer),
            Err(Refusal::MissingInput("PF90002 face mask"))
        );
        assert!(producer.moisture_face.is_none());

        let no_body = PortraitMasks {
            body: None,
            ..PortraitMasks::solid(1.0)
        };
        assert_eq!(
            bundle(JOB, &source).attach_moisture_inputs(&source, &no_body, &mut producer),
            Err(Refusal::MissingInput("PF90010 body mask"))
        );
        assert!(producer.moisture_face.is_none());
    }

    #[test]
    fn ticket_and_digest_mismatches_are_refused() {
        let source = source();
        let masks = PortraitMasks::solid(1.0);
        let mut producer = ProducerInputs::default();
        assert_eq!(
            bundle(0, &source).attach_moisture_inputs(&source, &masks, &mut producer),
            Err(Refusal::ValueOutOfRange("face input job ticket"))
        );
        let other = Image::solid(8, 8, [1, 2, 3, 255]);
        assert!(bundle(JOB, &source)
            .attach_moisture_inputs(&other, &masks, &mut producer)
            .is_err());
    }
}
