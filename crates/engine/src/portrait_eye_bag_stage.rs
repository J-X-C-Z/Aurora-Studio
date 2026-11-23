//! PF90116 same-frame producer contract.
//!
//! This module deliberately contains no model forward. A real forward must
//! issue a `Pf90116FrameReceipt` before any aligned plane reaches the portrait
//! stage. Until that issuer exists, the producer stays fail-closed.

use aurora_portrait::eye_bag_alignment::{
    AlignedEyeBagPlanes, EyeBagVendorGeometryReceipt, VENDOR_AFFINE_FIT_M1_CALL_SITE,
    VENDOR_AFFINE_FIT_M2_CALL_SITE, VENDOR_PF90116_HOST_SHA256,
};
use aurora_portrait::kernels::eye_bag::Roi;
use aurora_portrait::{Image, Refusal};

use super::FaceInputBundle;

pub const PF90116_MODEL_ID: u64 = 814_444_604;
pub const PF90116_INPUT_NAME: &str = "781141184";
pub const PF90116_RESPONSE_NAME: &str = "3204548346";
pub const PF90116_COLOR_NAME: &str = "780929437";
pub const PF90116_MODEL_WIDTH: u32 = 512;
pub const PF90116_MODEL_HEIGHT: u32 = 384;
pub const PF90116_INPUT_SHAPE: [usize; 4] = [1, 3, 384, 512];
pub const PF90116_RESPONSE_SHAPE: [usize; 4] = [1, 1, 384, 512];
pub const PF90116_COLOR_SHAPE: [usize; 4] = [1, 3, 384, 512];
pub const PF90116_ACTIVE_GATE: f32 = 0.001;
pub const PF90116_INPUT_MEAN: [f32; 3] = [127.5; 3];
pub const PF90116_INPUT_SCALE: [f32; 3] = [127.5; 3];
pub const PF90116_MODEL_PAYLOAD_SHA256: [u8; 32] = [
    0x42, 0x86, 0x45, 0xa9, 0x08, 0x3f, 0x07, 0x72, 0x7e, 0x7b, 0xb4, 0xf8, 0xfb, 0x10, 0x68, 0x66,
    0x16, 0x67, 0xe5, 0x05, 0x43, 0x3a, 0xb3, 0x23, 0x5c, 0xcf, 0x65, 0x91, 0x4c, 0x8a, 0x5d, 0x38,
];
pub const PF90116_MODEL_NET_SHA256: [u8; 32] = [
    0x03, 0x67, 0x68, 0xd2, 0xd3, 0xa3, 0x88, 0x38, 0x0d, 0x58, 0xa6, 0x6c, 0x27, 0x48, 0x35, 0xeb,
    0xa4, 0xb4, 0xe4, 0xf2, 0xb0, 0x6f, 0x12, 0xf3, 0x40, 0xdb, 0x40, 0x31, 0xcf, 0x6a, 0x9d, 0xef,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pf90116ColorDomain {
    /// The stage's byte-valued source color domain.
    SourceBytes,
    /// A raw model domain that still needs a proven conversion.
    RawModel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pf90116InputOrder {
    Rgb,
    Bgr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pf90116TensorReceipts {
    pub input: u64,
    pub response: u64,
    pub color: u64,
    /// The aggregate value consumed by `EyeBagVendorGeometryReceipt`.
    pub aggregate: u64,
}

impl Pf90116TensorReceipts {
    fn complete(self) -> bool {
        self.input != 0 && self.response != 0 && self.color != 0 && self.aggregate != 0
    }
}

/// Frame/face facts derived from the same renderer frame as `FaceInputBundle`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pf90116FrameBinding {
    pub job_ticket: u64,
    pub frame_size: [u32; 2],
    pub frame_sha256: [u8; 32],
    pub face_record_id: u64,
    pub landmark_receipt: u64,
    pub degree_bits: u32,
}

impl Pf90116FrameBinding {
    pub fn from_face_bundle(
        job_ticket: u64,
        frame: &Image,
        face: &FaceInputBundle,
        face_record_id: u64,
        landmark_receipt: u64,
        degree: f32,
    ) -> Result<Self, Refusal> {
        if job_ticket == 0 {
            return Err(Refusal::MissingInput("PF90116_SAME_JOB_TICKET"));
        }
        face.validate_ticket(frame, Some(job_ticket))
            .map_err(|_| Refusal::MissingInput("PF90116_FRAME_BINDING"))?;
        if face_record_id == 0 {
            return Err(Refusal::MissingInput("PF90116_GEOMETRY_FACE_RECORD"));
        }
        if landmark_receipt == 0 {
            return Err(Refusal::MissingInput("PF90116_GEOMETRY_LANDMARKS"));
        }
        check_degree(degree)?;
        Ok(Self {
            job_ticket,
            frame_size: [frame.width, frame.height],
            frame_sha256: face.source_sha256,
            face_record_id,
            landmark_receipt,
            degree_bits: degree.to_bits(),
        })
    }
}

/// A receipt that can only be issued by the PF90116 forward owner.
///
/// There is intentionally no public constructor. The skeleton producer cannot
/// issue one, so a caller cannot turn hand-written values into a vendor receipt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pf90116FrameReceipt {
    job_ticket: u64,
    frame_size: [u32; 2],
    frame_sha256: [u8; 32],
    vendor_job_id: u64,
    face_record_id: u64,
    landmark_receipt: u64,
    model_id: u64,
    model_payload_sha256: [u8; 32],
    model_net_sha256: [u8; 32],
    tensors: Pf90116TensorReceipts,
    input_order: Pf90116InputOrder,
    input_mean: [f32; 3],
    input_scale: [f32; 3],
    frame_to_model: [f32; 6],
    model_roi: [i32; 4],
    work_width: u32,
    work_height: u32,
    work_to_destination: [f32; 6],
    destination_roi: [i32; 4],
    color_domain: Pf90116ColorDomain,
    degree_bits: u32,
}

impl Pf90116FrameReceipt {
    pub fn validate(&self, binding: &Pf90116FrameBinding) -> Result<(), Refusal> {
        if self.job_ticket == 0 || self.job_ticket != binding.job_ticket || self.vendor_job_id == 0
        {
            return Err(Refusal::MissingInput("PF90116_GEOMETRY_SAME_JOB"));
        }
        if self.frame_size != binding.frame_size {
            return Err(Refusal::ValueOutOfRange("PF90116_FRAME_SIZE"));
        }
        if self.frame_sha256 != binding.frame_sha256 || self.frame_sha256 == [0; 32] {
            return Err(Refusal::ValueOutOfRange("PF90116_FRAME_DIGEST"));
        }
        if self.face_record_id == 0 || self.face_record_id != binding.face_record_id {
            return Err(Refusal::MissingInput("PF90116_GEOMETRY_FACE_RECORD"));
        }
        if self.landmark_receipt == 0 || self.landmark_receipt != binding.landmark_receipt {
            return Err(Refusal::MissingInput("PF90116_GEOMETRY_LANDMARKS"));
        }
        if self.model_id != PF90116_MODEL_ID {
            return Err(Refusal::ValueOutOfRange("PF90116_MODEL_ID"));
        }
        if self.model_payload_sha256 != PF90116_MODEL_PAYLOAD_SHA256
            || self.model_net_sha256 != PF90116_MODEL_NET_SHA256
        {
            return Err(Refusal::ValueOutOfRange("PF90116_MODEL_IDENTITY"));
        }
        if !self.tensors.complete() {
            return Err(Refusal::MissingInput("PF90116_MODEL_TENSOR_RECEIPT"));
        }
        if self.input_order != Pf90116InputOrder::Rgb
            || self.input_mean != PF90116_INPUT_MEAN
            || self.input_scale != PF90116_INPUT_SCALE
        {
            return Err(Refusal::ValueOutOfRange("PF90116_MODEL_INPUT_CONTRACT"));
        }
        if !valid_affine(self.frame_to_model) || !valid_model_roi(self.model_roi) {
            return Err(Refusal::ValueOutOfRange("PF90116_MODEL_INPUT_GEOMETRY"));
        }
        if self.work_width == 0 || self.work_height == 0 || !valid_affine(self.work_to_destination)
        {
            return Err(Refusal::ValueOutOfRange("PF90116_WARP_AFFINE"));
        }
        if !valid_frame_roi(self.destination_roi, binding.frame_size) {
            return Err(Refusal::ValueOutOfRange("PF90116_DESTINATION_ROI"));
        }
        if self.color_domain != Pf90116ColorDomain::SourceBytes {
            return Err(Refusal::MissingInput("PF90116_COLOR_DOMAIN_RECEIPT"));
        }
        if self.degree_bits != binding.degree_bits {
            return Err(Refusal::ValueOutOfRange("PF90116_DEGREE"));
        }
        check_degree(f32::from_bits(self.degree_bits))
    }

    /// Converts the validated receipt into the existing geometry receipt.
    ///
    /// The caller must first validate against the same frame binding. Keeping
    /// the tensor aggregate in the receipt prevents a geometry-only wrapper
    /// from becoming a claim about the model outputs.
    pub fn geometry_receipt(&self) -> EyeBagVendorGeometryReceipt {
        EyeBagVendorGeometryReceipt {
            job_ticket: self.job_ticket,
            vendor_job_id: self.vendor_job_id,
            face_record_id: self.face_record_id,
            landmark_receipt: self.landmark_receipt,
            model_tensor_receipt: self.tensors.aggregate,
            host_sha256: VENDOR_PF90116_HOST_SHA256,
            m1_call_site: VENDOR_AFFINE_FIT_M1_CALL_SITE,
            m2_call_site: VENDOR_AFFINE_FIT_M2_CALL_SITE,
            work_width: self.work_width,
            work_height: self.work_height,
            work_to_destination: self.work_to_destination,
            destination_roi: Roi {
                x: self.destination_roi[0],
                y: self.destination_roi[1],
                w: self.destination_roi[2],
                h: self.destination_roi[3],
            },
        }
    }
}

/// Contract-only producer. It has no model forward and cannot issue receipts.
pub struct EyeBagFrameProducer {
    _private: (),
}

impl EyeBagFrameProducer {
    pub fn open_from_env() -> Result<Self, Refusal> {
        Err(Refusal::MissingInput("PF90116_MODEL_RUNTIME"))
    }

    pub fn produce(
        &self,
        job_ticket: u64,
        frame: &Image,
        face: &FaceInputBundle,
        degree: f32,
        receipt: &Pf90116FrameReceipt,
    ) -> Result<AlignedEyeBagPlanes, Refusal> {
        let binding = Pf90116FrameBinding::from_face_bundle(
            job_ticket,
            frame,
            face,
            receipt.face_record_id,
            receipt.landmark_receipt,
            degree,
        )?;
        self.produce_validated(&binding, receipt)
    }

    /// The validation boundary used by tests before the missing forward.
    pub fn produce_validated(
        &self,
        binding: &Pf90116FrameBinding,
        receipt: &Pf90116FrameReceipt,
    ) -> Result<AlignedEyeBagPlanes, Refusal> {
        receipt.validate(binding)?;
        Err(Refusal::MissingInput("PF90116_MODEL_RUNTIME"))
    }

    #[cfg(test)]
    pub(crate) fn for_tests() -> Self {
        Self { _private: () }
    }
}

fn check_degree(degree: f32) -> Result<(), Refusal> {
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("PF90116_DEGREE"));
    }
    if degree <= PF90116_ACTIVE_GATE {
        return Err(Refusal::MissingInput("PF90116_ACTIVE_DEGREE"));
    }
    Ok(())
}

fn valid_affine(matrix: [f32; 6]) -> bool {
    if matrix.iter().any(|value| !value.is_finite()) {
        return false;
    }
    let determinant =
        f64::from(matrix[0]) * f64::from(matrix[4]) - f64::from(matrix[1]) * f64::from(matrix[3]);
    determinant.abs() > 1.0e-12
}

fn valid_model_roi(roi: [i32; 4]) -> bool {
    valid_roi(roi, PF90116_MODEL_WIDTH, PF90116_MODEL_HEIGHT)
}

fn valid_frame_roi(roi: [i32; 4], frame_size: [u32; 2]) -> bool {
    valid_roi(roi, frame_size[0], frame_size[1])
}

fn valid_roi(roi: [i32; 4], width: u32, height: u32) -> bool {
    let [x, y, w, h] = roi;
    x >= 0
        && y >= 0
        && w > 0
        && h > 0
        && i64::from(x) + i64::from(w) <= i64::from(width)
        && i64::from(y) + i64::from(h) <= i64::from(height)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding() -> Pf90116FrameBinding {
        Pf90116FrameBinding {
            job_ticket: 7,
            frame_size: [32, 24],
            frame_sha256: [9; 32],
            face_record_id: 0x11,
            landmark_receipt: 0x22,
            degree_bits: 0.8_f32.to_bits(),
        }
    }

    fn receipt(binding: &Pf90116FrameBinding) -> Pf90116FrameReceipt {
        Pf90116FrameReceipt {
            job_ticket: binding.job_ticket,
            frame_size: binding.frame_size,
            frame_sha256: binding.frame_sha256,
            vendor_job_id: 0x33,
            face_record_id: binding.face_record_id,
            landmark_receipt: binding.landmark_receipt,
            model_id: PF90116_MODEL_ID,
            model_payload_sha256: PF90116_MODEL_PAYLOAD_SHA256,
            model_net_sha256: PF90116_MODEL_NET_SHA256,
            tensors: Pf90116TensorReceipts {
                input: 101,
                response: 102,
                color: 103,
                aggregate: 104,
            },
            input_order: Pf90116InputOrder::Rgb,
            input_mean: PF90116_INPUT_MEAN,
            input_scale: PF90116_INPUT_SCALE,
            frame_to_model: [0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
            model_roi: [0, 0, 512, 384],
            work_width: 32,
            work_height: 24,
            work_to_destination: [0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
            destination_roi: [4, 3, 20, 16],
            color_domain: Pf90116ColorDomain::SourceBytes,
            degree_bits: binding.degree_bits,
        }
    }

    #[test]
    fn complete_receipt_validates_and_exposes_geometry() {
        let binding = binding();
        let receipt = receipt(&binding);
        receipt.validate(&binding).expect("complete receipt");
        let geometry = receipt.geometry_receipt();
        assert_eq!(geometry.job_ticket, binding.job_ticket);
        assert_eq!(geometry.face_record_id, binding.face_record_id);
        assert_eq!(geometry.landmark_receipt, binding.landmark_receipt);
        assert_eq!(geometry.model_tensor_receipt, receipt.tensors.aggregate);
        assert_eq!(geometry.destination_roi.w, 20);
    }

    #[test]
    fn receipt_rejects_frame_and_job_mismatch() {
        let binding = binding();
        let wrong_job = Pf90116FrameReceipt {
            job_ticket: 8,
            ..receipt(&binding)
        };
        assert_eq!(
            wrong_job.validate(&binding),
            Err(Refusal::MissingInput("PF90116_GEOMETRY_SAME_JOB"))
        );
        let wrong_frame = Pf90116FrameReceipt {
            frame_sha256: [8; 32],
            ..receipt(&binding)
        };
        assert_eq!(
            wrong_frame.validate(&binding),
            Err(Refusal::ValueOutOfRange("PF90116_FRAME_DIGEST"))
        );
    }

    #[test]
    fn receipt_rejects_missing_face_and_landmark_provenance() {
        let binding = binding();
        let no_face = Pf90116FrameReceipt {
            face_record_id: 0,
            ..receipt(&binding)
        };
        assert_eq!(
            no_face.validate(&binding),
            Err(Refusal::MissingInput("PF90116_GEOMETRY_FACE_RECORD"))
        );
        let no_landmarks = Pf90116FrameReceipt {
            landmark_receipt: 0,
            ..receipt(&binding)
        };
        assert_eq!(
            no_landmarks.validate(&binding),
            Err(Refusal::MissingInput("PF90116_GEOMETRY_LANDMARKS"))
        );
    }

    #[test]
    fn receipt_rejects_model_identity_and_tensor_gaps() {
        let binding = binding();
        let wrong_model = Pf90116FrameReceipt {
            model_id: 4_030_703_027,
            ..receipt(&binding)
        };
        assert_eq!(
            wrong_model.validate(&binding),
            Err(Refusal::ValueOutOfRange("PF90116_MODEL_ID"))
        );
        let missing_tensor = Pf90116FrameReceipt {
            tensors: Pf90116TensorReceipts {
                response: 0,
                ..receipt(&binding).tensors
            },
            ..receipt(&binding)
        };
        assert_eq!(
            missing_tensor.validate(&binding),
            Err(Refusal::MissingInput("PF90116_MODEL_TENSOR_RECEIPT"))
        );
    }

    #[test]
    fn receipt_rejects_unproven_color_and_bad_geometry() {
        let binding = binding();
        let raw_color = Pf90116FrameReceipt {
            color_domain: Pf90116ColorDomain::RawModel,
            ..receipt(&binding)
        };
        assert_eq!(
            raw_color.validate(&binding),
            Err(Refusal::MissingInput("PF90116_COLOR_DOMAIN_RECEIPT"))
        );
        let bad_affine = Pf90116FrameReceipt {
            work_to_destination: [0.0; 6],
            ..receipt(&binding)
        };
        assert_eq!(
            bad_affine.validate(&binding),
            Err(Refusal::ValueOutOfRange("PF90116_WARP_AFFINE"))
        );
        let outside_frame = Pf90116FrameReceipt {
            destination_roi: [30, 20, 4, 4],
            ..receipt(&binding)
        };
        assert_eq!(
            outside_frame.validate(&binding),
            Err(Refusal::ValueOutOfRange("PF90116_DESTINATION_ROI"))
        );
    }

    #[test]
    fn producer_fails_closed_without_forward() {
        let binding = binding();
        let receipt = receipt(&binding);
        assert_eq!(
            EyeBagFrameProducer::for_tests().produce_validated(&binding, &receipt),
            Err(Refusal::MissingInput("PF90116_MODEL_RUNTIME"))
        );
        assert_eq!(
            EyeBagFrameProducer::open_from_env().err(),
            Some(Refusal::MissingInput("PF90116_MODEL_RUNTIME"))
        );
    }
}
