//! PF90116 model-frame alignment for model `814444604`.
//!
//! The producer consumes the model's real input and both model outputs:
//! `781141184` RGB CHW, `3204548346` response HW, and `780929437` RGB CHW.
//! The response grid selects the crop. The supplied runtime affine and
//! destination ROI are used for the frame paste. Geometry is supplied by the
//! caller; this module never substitutes a center crop, identity matrix, or
//! another frame's values.

use crate::Refusal;
use crate::kernels::eye_bag::{self, Roi};

pub const MODEL_ID: u64 = 814_444_604;
pub const MODEL_WIDTH: u32 = 512;
pub const MODEL_HEIGHT: u32 = 384;
pub const MODEL_INPUT_781141184: &str = "781141184";
pub const MODEL_RESPONSE_3204548346: &str = "3204548346";
pub const MODEL_COLOR_780929437: &str = "780929437";
pub const RESPONSE_THRESHOLD: f32 = 0.001;
pub const VENDOR_PF90116_HOST_SHA256: [u8; 32] = [
    0xa8, 0x45, 0x19, 0xda, 0x48, 0x9b, 0xc5, 0x2e, 0x4e, 0x91, 0x4a, 0xf7, 0x88, 0x0e,
    0xde, 0x43, 0xe3, 0xce, 0x58, 0x99, 0x26, 0x57, 0x72, 0x3d, 0x02, 0x19, 0x71, 0x61,
    0x83, 0x24, 0x64, 0xc7,
];
pub const VENDOR_AFFINE_FIT_M1_CALL_SITE: u32 = 0x05c0_abc5;
pub const VENDOR_AFFINE_FIT_M2_CALL_SITE: u32 = 0x05c0_af49;

/// Provenance of the geometry supplied to the producer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EyeBagGeometryClass {
    /// Explicitly supplied candidate geometry. This is not vendor parity.
    Candidate,
    /// Geometry captured from the same vendor render job.
    VendorSameJob,
}

/// Unit domain of the producer's color plane. A vendor/raw model field is not
/// the byte-valued target consumed by the stage without an explicit conversion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EyeBagColorUnits {
    RawModel,
    SourceBytes,
}

/// Capture identity for geometry produced by the vendor's PF90116 affine fitter.
///
/// This receipt is deliberately separate from `geometry_class`: a caller can
/// no longer turn candidate work geometry into `VendorSameJob` by setting an
/// enum. The model fingerprint binds the three exact tensor buffers, while the
/// face/landmark and writer fields bind the affine source to that same job.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyeBagVendorGeometryReceipt {
    pub job_ticket: u64,
    pub vendor_job_id: u64,
    pub face_record_id: u64,
    pub landmark_receipt: u64,
    pub model_tensor_receipt: u64,
    pub host_sha256: [u8; 32],
    pub m1_call_site: u32,
    pub m2_call_site: u32,
    pub work_width: u32,
    pub work_height: u32,
    pub work_to_destination: [f32; 6],
    pub destination_roi: Roi,
}

/// The three real tensors of one model invocation.
#[derive(Clone, Copy, Debug)]
pub struct EyeBagModelTensors<'a> {
    pub model_id: u64,
    /// `781141184`, NCHW `[1,3,384,512]`, CHW in row-major order.
    pub input_781141184_rgb_chw: &'a [f32],
    /// `3204548346`, NCHW `[1,1,384,512]`, HW in row-major order.
    pub response_3204548346_hw: &'a [f32],
    /// `780929437`, NCHW `[1,3,384,512]`, CHW in row-major order.
    pub color_780929437_rgb_chw: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
pub struct EyeBagAlignmentInputs<'a> {
    pub ticket: u64,
    pub frame_width: u32,
    pub frame_height: u32,
    pub tensors: EyeBagModelTensors<'a>,
    /// Units of `780929437` as supplied by the model runtime.
    pub model_color_units: EyeBagColorUnits,
    pub geometry_class: EyeBagGeometryClass,
    /// Required for `VendorSameJob` and forbidden for `Candidate`.
    pub geometry_receipt: Option<EyeBagVendorGeometryReceipt>,
    /// Runtime work Mat dimensions used by the host alignment path.
    pub work_width: Option<u32>,
    pub work_height: Option<u32>,
    /// Exact 2x3 `CV_32F` matrix passed to `cv::warpAffine`; row-major
    /// `[m00,m01,m02,m10,m11,m12]`, destination is ROI-local.
    pub work_to_destination: Option<[f32; 6]>,
    /// Frame destination fields `cell+0xA0..0xAC`.
    pub destination_roi: Option<Roi>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AlignedEyeBagPlanes {
    pub ticket: u64,
    pub frame_width: u32,
    pub frame_height: u32,
    pub model_id: u64,
    /// The two full-frame planes consumed by `portrait_stage` PF90116.
    pub aligned_color_field_rgb_f32: Vec<f32>,
    pub color_units: EyeBagColorUnits,
    pub aligned_response_mask: Vec<f32>,
    /// `781141184` after the same crop/resize/affine step, kept ROI-local to
    /// avoid a third full-frame float plane. It is provenance, not a stage
    /// consumer.
    pub destination_model_input_rgb_f32: Vec<f32>,
    pub source_response_roi: Option<Roi>,
    pub work_roi: Option<Roi>,
    pub destination_roi: Roi,
    pub geometry_class: EyeBagGeometryClass,
    pub geometry_receipt: Option<EyeBagVendorGeometryReceipt>,
}

impl AlignedEyeBagPlanes {
    pub fn validate(&self) -> Result<(), Refusal> {
        if self.ticket == 0 {
            return Err(Refusal::MissingInput("PF90116_SAME_JOB_TICKET"));
        }
        if self.model_id != MODEL_ID {
            return Err(Refusal::ValueOutOfRange("PF90116_MODEL_ID"));
        }
        if self.frame_width == 0 || self.frame_height == 0 {
            return Err(Refusal::ValueOutOfRange("PF90116_FRAME_SIZE"));
        }
        let pixels = checked_rgb_pixels(self.frame_width, self.frame_height, "PF90116_FRAME_SIZE")?;
        if self.aligned_response_mask.len() != pixels
            || self.aligned_color_field_rgb_f32.len() != pixels * 3
        {
            return Err(Refusal::ShapeMismatch("PF90116_ALIGNED_PLANES"));
        }
        if !roi_inside(self.destination_roi, self.frame_width, self.frame_height) {
            return Err(Refusal::ValueOutOfRange("PF90116_DESTINATION_ROI"));
        }
        let destination_pixels = self.destination_roi.w as usize * self.destination_roi.h as usize;
        if self.destination_model_input_rgb_f32.len() != destination_pixels * 3 {
            return Err(Refusal::ShapeMismatch("PF90116_MODEL_INPUT_ALIGNED"));
        }
        if self
            .aligned_color_field_rgb_f32
            .iter()
            .chain(&self.aligned_response_mask)
            .chain(&self.destination_model_input_rgb_f32)
            .any(|value| !value.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("PF90116_ALIGNED_PLANES"));
        }
        match (self.geometry_class, self.geometry_receipt) {
            (EyeBagGeometryClass::Candidate, None) => {}
            (EyeBagGeometryClass::VendorSameJob, Some(receipt)) => {
                if receipt.job_ticket != self.ticket {
                    return Err(Refusal::MissingInput("PF90116_GEOMETRY_SAME_JOB"));
                }
                if receipt.destination_roi != self.destination_roi {
                    return Err(Refusal::ValueOutOfRange("PF90116_GEOMETRY_RECEIPT"));
                }
            }
            (EyeBagGeometryClass::Candidate, Some(_)) => {
                return Err(Refusal::ValueOutOfRange("PF90116_GEOMETRY_RECEIPT"));
            }
            (EyeBagGeometryClass::VendorSameJob, None) => {
                return Err(Refusal::MissingInput("PF90116_GEOMETRY_RECEIPT"));
            }
        }
        Ok(())
    }

    /// True only when the caller explicitly supplied same-job vendor geometry.
    pub fn vendor_geometry_claimed(&self) -> bool {
        self.geometry_class == EyeBagGeometryClass::VendorSameJob
    }
}

impl EyeBagVendorGeometryReceipt {
    pub fn from_capture(
        job_ticket: u64,
        vendor_job_id: u64,
        face_record_id: u64,
        landmark_receipt: u64,
        tensors: &EyeBagModelTensors<'_>,
        work_width: u32,
        work_height: u32,
        work_to_destination: [f32; 6],
        destination_roi: Roi,
    ) -> Self {
        Self {
            job_ticket,
            vendor_job_id,
            face_record_id,
            landmark_receipt,
            model_tensor_receipt: model_tensor_receipt(job_ticket, tensors),
            host_sha256: VENDOR_PF90116_HOST_SHA256,
            m1_call_site: VENDOR_AFFINE_FIT_M1_CALL_SITE,
            m2_call_site: VENDOR_AFFINE_FIT_M2_CALL_SITE,
            work_width,
            work_height,
            work_to_destination,
            destination_roi,
        }
    }

    pub fn validate_for_stage(self, ticket: u64) -> Result<(), Refusal> {
        if self.job_ticket == 0 || self.vendor_job_id == 0 || self.job_ticket != ticket {
            return Err(Refusal::MissingInput("PF90116_GEOMETRY_SAME_JOB"));
        }
        if self.face_record_id == 0 {
            return Err(Refusal::MissingInput("PF90116_GEOMETRY_FACE_RECORD"));
        }
        if self.landmark_receipt == 0 {
            return Err(Refusal::MissingInput("PF90116_GEOMETRY_LANDMARKS"));
        }
        if self.model_tensor_receipt == 0 {
            return Err(Refusal::MissingInput("PF90116_MODEL_TENSOR_RECEIPT"));
        }
        if self.host_sha256 != VENDOR_PF90116_HOST_SHA256
            || self.m1_call_site != VENDOR_AFFINE_FIT_M1_CALL_SITE
            || self.m2_call_site != VENDOR_AFFINE_FIT_M2_CALL_SITE
        {
            return Err(Refusal::MissingInput("PF90116_GEOMETRY_WRITER"));
        }
        if self.work_width == 0
            || self.work_height == 0
            || self.destination_roi.w <= 0
            || self.destination_roi.h <= 0
        {
            return Err(Refusal::ValueOutOfRange("PF90116_GEOMETRY_RECEIPT"));
        }
        invert_affine(self.work_to_destination)
            .map(|_| ())
            .map_err(|_| Refusal::ValueOutOfRange("PF90116_GEOMETRY_RECEIPT"))
    }

    fn refusal(
        self,
        ticket: u64,
        tensors: &EyeBagModelTensors<'_>,
        work_width: u32,
        work_height: u32,
        work_to_destination: [f32; 6],
        destination_roi: Roi,
    ) -> Option<Refusal> {
        if self.job_ticket == 0
            || self.vendor_job_id == 0
            || self.job_ticket != ticket
        {
            return Some(Refusal::MissingInput("PF90116_GEOMETRY_SAME_JOB"));
        }
        if self.face_record_id == 0 {
            return Some(Refusal::MissingInput("PF90116_GEOMETRY_FACE_RECORD"));
        }
        if self.landmark_receipt == 0 {
            return Some(Refusal::MissingInput("PF90116_GEOMETRY_LANDMARKS"));
        }
        if self.model_tensor_receipt == 0
            || self.model_tensor_receipt != model_tensor_receipt(ticket, tensors)
        {
            return Some(Refusal::MissingInput("PF90116_MODEL_TENSOR_RECEIPT"));
        }
        if self.host_sha256 != VENDOR_PF90116_HOST_SHA256
            || self.m1_call_site != VENDOR_AFFINE_FIT_M1_CALL_SITE
            || self.m2_call_site != VENDOR_AFFINE_FIT_M2_CALL_SITE
        {
            return Some(Refusal::MissingInput("PF90116_GEOMETRY_WRITER"));
        }
        if self.work_width != work_width
            || self.work_height != work_height
            || affine_bits(self.work_to_destination) != affine_bits(work_to_destination)
            || self.destination_roi != destination_roi
        {
            return Some(Refusal::ValueOutOfRange("PF90116_GEOMETRY_RECEIPT"));
        }
        None
    }
}

pub fn model_tensor_receipts(
    job_ticket: u64,
    tensors: &EyeBagModelTensors<'_>,
) -> [u64; 4] {
    let plane_receipt = |tag: u64, plane: &[f32]| {
        let mut hash = FNV1A64_OFFSET_BASIS;
        hash = fnv_bytes(hash, b"aurora-pf90116-tensor-v1");
        hash = fnv_u64(hash, job_ticket);
        hash = fnv_u64(hash, tensors.model_id);
        hash = fnv_u64(hash, tag);
        hash = fnv_usize(hash, plane.len());
        for value in plane {
            hash = fnv_u64(hash, value.to_bits() as u64);
        }
        hash
    };
    let input = plane_receipt(1, tensors.input_781141184_rgb_chw);
    let response = plane_receipt(2, tensors.response_3204548346_hw);
    let color = plane_receipt(3, tensors.color_780929437_rgb_chw);
    let mut aggregate = FNV1A64_OFFSET_BASIS;
    aggregate = fnv_bytes(aggregate, b"aurora-pf90116-four-receipt-v1");
    aggregate = fnv_u64(aggregate, job_ticket);
    aggregate = fnv_u64(aggregate, tensors.model_id);
    for receipt in [input, response, color] {
        aggregate = fnv_u64(aggregate, receipt);
    }
    [input, response, color, aggregate]
}

pub fn model_tensor_receipt(job_ticket: u64, tensors: &EyeBagModelTensors<'_>) -> u64 {
    model_tensor_receipts(job_ticket, tensors)[3]
}

fn raw_model_color_to_source_bytes(value: f32) -> f32 {
    (value + 1.0) * 127.5
}

pub fn align_eye_bag(inputs: EyeBagAlignmentInputs<'_>) -> Result<AlignedEyeBagPlanes, Refusal> {
    if inputs.ticket == 0 {
        return Err(Refusal::MissingInput("PF90116_SAME_JOB_TICKET"));
    }
    if inputs.tensors.model_id != MODEL_ID {
        return Err(Refusal::ValueOutOfRange("PF90116_MODEL_ID"));
    }
    if inputs.frame_width == 0 || inputs.frame_height == 0 {
        return Err(Refusal::ValueOutOfRange("PF90116_FRAME_SIZE"));
    }
    let frame_pixels = checked_rgb_pixels(
        inputs.frame_width,
        inputs.frame_height,
        "PF90116_FRAME_SIZE",
    )?;
    let model_pixels = checked_rgb_pixels(MODEL_WIDTH, MODEL_HEIGHT, "PF90116_MODEL_SIZE")?;

    if inputs.tensors.input_781141184_rgb_chw.is_empty() {
        return Err(Refusal::MissingInput("PF90116_MODEL_INPUT_781141184"));
    }
    if inputs.tensors.input_781141184_rgb_chw.len() != model_pixels * 3 {
        return Err(Refusal::ShapeMismatch("PF90116_MODEL_INPUT_781141184"));
    }
    if inputs.tensors.response_3204548346_hw.is_empty() {
        return Err(Refusal::MissingInput("PF90116_MODEL_RESPONSE_3204548346"));
    }
    if inputs.tensors.response_3204548346_hw.len() != model_pixels {
        return Err(Refusal::ShapeMismatch("PF90116_MODEL_RESPONSE_3204548346"));
    }
    if inputs.tensors.color_780929437_rgb_chw.is_empty() {
        return Err(Refusal::MissingInput("PF90116_MODEL_COLOR_780929437"));
    }
    if inputs.tensors.color_780929437_rgb_chw.len() != model_pixels * 3 {
        return Err(Refusal::ShapeMismatch("PF90116_MODEL_COLOR_780929437"));
    }
    if inputs
        .tensors
        .input_781141184_rgb_chw
        .iter()
        .chain(inputs.tensors.response_3204548346_hw)
        .chain(inputs.tensors.color_780929437_rgb_chw)
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("PF90116_MODEL_TENSORS"));
    }

    let work_width = inputs
        .work_width
        .ok_or(Refusal::MissingInput("PF90116_WORK_WIDTH"))?;
    let work_height = inputs
        .work_height
        .ok_or(Refusal::MissingInput("PF90116_WORK_HEIGHT"))?;
    if work_width == 0 || work_height == 0 {
        return Err(Refusal::ValueOutOfRange("PF90116_WORK_SIZE"));
    }
    checked_rgb_pixels(work_width, work_height, "PF90116_WORK_SIZE")?;

    let matrix = inputs
        .work_to_destination
        .ok_or(Refusal::MissingInput("PF90116_WARP_AFFINE"))?;
    let inverse = invert_affine(matrix)?;
    let destination_roi = inputs
        .destination_roi
        .ok_or(Refusal::MissingInput("PF90116_DESTINATION_ROI"))?;
    if !roi_inside(destination_roi, inputs.frame_width, inputs.frame_height) {
        return Err(Refusal::ValueOutOfRange("PF90116_DESTINATION_ROI"));
    }
    let geometry_receipt = match inputs.geometry_class {
        EyeBagGeometryClass::Candidate => {
            if inputs.geometry_receipt.is_some() {
                return Err(Refusal::ValueOutOfRange("PF90116_GEOMETRY_RECEIPT"));
            }
            None
        }
        EyeBagGeometryClass::VendorSameJob => {
            let receipt = inputs
                .geometry_receipt
                .ok_or(Refusal::MissingInput("PF90116_GEOMETRY_RECEIPT"))?;
            if let Some(refusal) = receipt.refusal(
                inputs.ticket,
                &inputs.tensors,
                work_width,
                work_height,
                matrix,
                destination_roi,
            ) {
                return Err(refusal);
            }
            Some(receipt)
        }
    };

    let source_response_roi = response_bounds(inputs.tensors.response_3204548346_hw);
    let work_roi = match source_response_roi {
        Some(raw) => eye_bag::response_source(expand_roi(raw), MODEL_WIDTH, MODEL_HEIGHT)?,
        None => None,
    };
    let crop = work_roi.unwrap_or(Roi {
        x: 0,
        y: 0,
        w: MODEL_WIDTH as i32,
        h: MODEL_HEIGHT as i32,
    });

    let cropped_input = crop_rgb_chw(inputs.tensors.input_781141184_rgb_chw, crop);
    let mut cropped_color = crop_rgb_chw(inputs.tensors.color_780929437_rgb_chw, crop);
    if inputs.model_color_units == EyeBagColorUnits::RawModel {
        for value in &mut cropped_color {
            *value = raw_model_color_to_source_bytes(*value);
        }
    }
    let cropped_response = crop_response(inputs.tensors.response_3204548346_hw, crop);

    let work_input = eye_bag::repair_image(
        &cropped_input,
        crop.w as u32,
        crop.h as u32,
        work_width,
        work_height,
    )?;
    let work_color = eye_bag::repair_image(
        &cropped_color,
        crop.w as u32,
        crop.h as u32,
        work_width,
        work_height,
    )?;
    let work_response = eye_bag::bilinear(
        &cropped_response,
        crop.w as u32,
        crop.h as u32,
        work_width,
        work_height,
    )?;

    let destination_input = warp_plane(
        &work_input,
        work_width,
        work_height,
        3,
        inverse,
        destination_roi.w as u32,
        destination_roi.h as u32,
    );
    let destination_color = warp_plane(
        &work_color,
        work_width,
        work_height,
        3,
        inverse,
        destination_roi.w as u32,
        destination_roi.h as u32,
    );
    let destination_response = warp_plane(
        &work_response,
        work_width,
        work_height,
        1,
        inverse,
        destination_roi.w as u32,
        destination_roi.h as u32,
    );

    let mut aligned_color = vec![0.0; frame_pixels * 3];
    let mut aligned_response = vec![0.0; frame_pixels];
    paste_roi(
        &mut aligned_color,
        inputs.frame_width,
        &destination_color,
        3,
        destination_roi,
    );
    paste_roi(
        &mut aligned_response,
        inputs.frame_width,
        &destination_response,
        1,
        destination_roi,
    );

    let aligned = AlignedEyeBagPlanes {
        ticket: inputs.ticket,
        frame_width: inputs.frame_width,
        frame_height: inputs.frame_height,
        model_id: inputs.tensors.model_id,
        aligned_color_field_rgb_f32: aligned_color,
        color_units: EyeBagColorUnits::SourceBytes,
        aligned_response_mask: aligned_response,
        destination_model_input_rgb_f32: destination_input,
        source_response_roi,
        work_roi,
        destination_roi,
        geometry_class: inputs.geometry_class,
        geometry_receipt,
    };
    aligned.validate()?;
    Ok(aligned)
}

const FNV1A64_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV1A64_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    hash
}

fn fnv_u64(hash: u64, value: u64) -> u64 {
    fnv_bytes(hash, &value.to_le_bytes())
}

fn fnv_usize(hash: u64, value: usize) -> u64 {
    fnv_u64(hash, value as u64)
}

fn affine_bits(matrix: [f32; 6]) -> [u32; 6] {
    matrix.map(f32::to_bits)
}

fn checked_rgb_pixels(width: u32, height: u32, field: &'static str) -> Result<usize, Refusal> {
    (width as usize)
        .checked_mul(height as usize)
        .filter(|pixels| *pixels <= isize::MAX as usize / (3 * std::mem::size_of::<f32>()))
        .ok_or(Refusal::ValueOutOfRange(field))
}

fn invert_affine(matrix: [f32; 6]) -> Result<[f64; 6], Refusal> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("PF90116_WARP_AFFINE"));
    }
    let [a, b, c, d, e, f] = matrix.map(f64::from);
    let determinant = a * e - b * d;
    if determinant.abs() <= 1.0e-12 {
        return Err(Refusal::ValueOutOfRange("PF90116_WARP_AFFINE"));
    }
    Ok([
        e / determinant,
        -b / determinant,
        (b * f - c * e) / determinant,
        -d / determinant,
        a / determinant,
        (c * d - a * f) / determinant,
    ])
}

fn response_bounds(response: &[f32]) -> Option<Roi> {
    let mut min_x = MODEL_WIDTH as i32;
    let mut min_y = MODEL_HEIGHT as i32;
    let mut max_x = -1;
    let mut max_y = -1;
    for y in 0..MODEL_HEIGHT as usize {
        for x in 0..MODEL_WIDTH as usize {
            if response[y * MODEL_WIDTH as usize + x] <= RESPONSE_THRESHOLD {
                continue;
            }
            min_x = min_x.min(x as i32);
            min_y = min_y.min(y as i32);
            max_x = max_x.max(x as i32);
            max_y = max_y.max(y as i32);
        }
    }
    (max_x >= min_x && max_y >= min_y).then_some(Roi {
        x: min_x,
        y: min_y,
        w: max_x - min_x + 1,
        h: max_y - min_y + 1,
    })
}

fn expand_roi(roi: Roi) -> Roi {
    let x = (roi.x - 4).max(0);
    let y = (roi.y - 4).max(0);
    let right = (roi.x + roi.w + 4).min(MODEL_WIDTH as i32);
    let bottom = (roi.y + roi.h + 4).min(MODEL_HEIGHT as i32);
    Roi {
        x,
        y,
        w: right - x,
        h: bottom - y,
    }
}

fn crop_rgb_chw(rgb: &[f32], roi: Roi) -> Vec<f32> {
    let pixels = roi.w as usize * roi.h as usize;
    let plane = MODEL_WIDTH as usize * MODEL_HEIGHT as usize;
    let mut out = vec![0.0; pixels * 3];
    for y in 0..roi.h as usize {
        for x in 0..roi.w as usize {
            let src = (roi.y as usize + y) * MODEL_WIDTH as usize + roi.x as usize + x;
            let dst = y * roi.w as usize + x;
            for channel in 0..3 {
                out[dst * 3 + channel] = rgb[channel * plane + src];
            }
        }
    }
    out
}

fn crop_response(response: &[f32], roi: Roi) -> Vec<f32> {
    let mut out = vec![0.0; roi.w as usize * roi.h as usize];
    for y in 0..roi.h as usize {
        let src = (roi.y as usize + y) * MODEL_WIDTH as usize + roi.x as usize;
        let dst = y * roi.w as usize;
        out[dst..dst + roi.w as usize].copy_from_slice(&response[src..src + roi.w as usize]);
    }
    out
}

fn warp_plane(
    source: &[f32],
    source_width: u32,
    source_height: u32,
    channels: usize,
    inverse: [f64; 6],
    destination_width: u32,
    destination_height: u32,
) -> Vec<f32> {
    let mut out = vec![0.0; destination_width as usize * destination_height as usize * channels];
    for y in 0..destination_height {
        for x in 0..destination_width {
            let dx = f64::from(x);
            let dy = f64::from(y);
            let sx = inverse[0] * dx + inverse[1] * dy + inverse[2];
            let sy = inverse[3] * dx + inverse[4] * dy + inverse[5];
            for channel in 0..channels {
                let value = sample_border_zero(
                    source,
                    source_width as usize,
                    source_height as usize,
                    channels,
                    channel,
                    sx,
                    sy,
                );
                let dst =
                    (y as usize * destination_width as usize + x as usize) * channels + channel;
                out[dst] = value;
            }
        }
    }
    out
}

fn sample_border_zero(
    source: &[f32],
    width: usize,
    height: usize,
    channels: usize,
    channel: usize,
    x: f64,
    y: f64,
) -> f32 {
    // Border zero applies to individual taps, including fractional edges.
    if x <= -1.0 || y <= -1.0 || x >= width as f64 || y >= height as f64 {
        return 0.0;
    }
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let wx = (x - x0 as f64) as f32;
    let wy = (y - y0 as f64) as f32;
    let sample = |px: i64, py: i64| {
        if (0..width as i64).contains(&px) && (0..height as i64).contains(&py) {
            source[(py as usize * width + px as usize) * channels + channel]
        } else {
            0.0
        }
    };
    let top = sample(x0, y0) + (sample(x1, y0) - sample(x0, y0)) * wx;
    let bottom = sample(x0, y1) + (sample(x1, y1) - sample(x0, y1)) * wx;
    top + (bottom - top) * wy
}

fn paste_roi(
    destination: &mut [f32],
    destination_width: u32,
    source: &[f32],
    channels: usize,
    roi: Roi,
) {
    let row_values = roi.w as usize * channels;
    for y in 0..roi.h as usize {
        let src = y * row_values;
        let dst = ((roi.y as usize + y) * destination_width as usize + roi.x as usize) * channels;
        destination[dst..dst + row_values].copy_from_slice(&source[src..src + row_values]);
    }
}

fn roi_inside(roi: Roi, width: u32, height: u32) -> bool {
    roi.x >= 0
        && roi.y >= 0
        && roi.w > 0
        && roi.h > 0
        && roi.x as i64 + roi.w as i64 <= width as i64
        && roi.y as i64 + roi.h as i64 <= height as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_tensors() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let pixels = (MODEL_WIDTH * MODEL_HEIGHT) as usize;
        let input = (0..pixels * 3)
            .map(|index| (index % 97) as f32 / 97.0)
            .collect();
        let color = (0..pixels * 3).map(|index| (index % 251) as f32).collect();
        let mut response = vec![0.0; pixels];
        for y in 160..224 {
            for x in 192..320 {
                response[y * MODEL_WIDTH as usize + x] = 0.5;
            }
        }
        (input, response, color)
    }

    #[test]
    fn geometry_and_model_identity_are_required_instead_of_defaulted() {
        let (input, response, color) = model_tensors();
        let tensors = EyeBagModelTensors {
            model_id: MODEL_ID,
            input_781141184_rgb_chw: &input,
            response_3204548346_hw: &response,
            color_780929437_rgb_chw: &color,
        };
        let base = EyeBagAlignmentInputs {
            ticket: 7,
            frame_width: 32,
            frame_height: 24,
            tensors,
            model_color_units: EyeBagColorUnits::SourceBytes,
            geometry_class: EyeBagGeometryClass::Candidate,
            geometry_receipt: None,
            work_width: None,
            work_height: None,
            work_to_destination: None,
            destination_roi: None,
        };
        assert_eq!(
            align_eye_bag(base),
            Err(Refusal::MissingInput("PF90116_WORK_WIDTH"))
        );
        assert_eq!(
            align_eye_bag(EyeBagAlignmentInputs { ticket: 0, ..base }),
            Err(Refusal::MissingInput("PF90116_SAME_JOB_TICKET"))
        );
        assert_eq!(
            align_eye_bag(EyeBagAlignmentInputs {
                tensors: EyeBagModelTensors {
                    model_id: 4_030_703_027,
                    ..tensors
                },
                ..base
            }),
            Err(Refusal::ValueOutOfRange("PF90116_MODEL_ID"))
        );
    }

    #[test]
    fn all_three_model_tensors_use_the_same_explicit_geometry() {
        let (input, response, color) = model_tensors();
        let aligned = align_eye_bag(EyeBagAlignmentInputs {
            ticket: 11,
            frame_width: 32,
            frame_height: 24,
            tensors: EyeBagModelTensors {
                model_id: MODEL_ID,
                input_781141184_rgb_chw: &input,
                response_3204548346_hw: &response,
                color_780929437_rgb_chw: &color,
            },
            model_color_units: EyeBagColorUnits::RawModel,
            geometry_class: EyeBagGeometryClass::Candidate,
            geometry_receipt: None,
            work_width: Some(32),
            work_height: Some(24),
            work_to_destination: Some([20.0 / 32.0, 0.0, 0.0, 0.0, 16.0 / 24.0, 0.0]),
            destination_roi: Some(Roi {
                x: 4,
                y: 3,
                w: 20,
                h: 16,
            }),
        })
        .expect("explicit geometry");
        assert_eq!(aligned.model_id, MODEL_ID);
        assert_eq!(aligned.color_units, EyeBagColorUnits::SourceBytes);
        assert_eq!(aligned.source_response_roi.unwrap().w, 128);
        assert_eq!(aligned.work_roi.unwrap().w, 136);
        assert_eq!(aligned.destination_model_input_rgb_f32.len(), 20 * 16 * 3);
        assert!(!aligned.vendor_geometry_claimed());
        assert!(
            aligned
                .aligned_response_mask
                .iter()
                .any(|value| *value != 0.0)
        );
    }

    #[test]
    fn vendor_claim_requires_fitted_geometry_receipt() {
        let (input, response, color) = model_tensors();
        let tensors = EyeBagModelTensors {
            model_id: MODEL_ID,
            input_781141184_rgb_chw: &input,
            response_3204548346_hw: &response,
            color_780929437_rgb_chw: &color,
        };
        let inputs = EyeBagAlignmentInputs {
            ticket: 17,
            frame_width: 32,
            frame_height: 24,
            tensors,
            model_color_units: EyeBagColorUnits::SourceBytes,
            geometry_class: EyeBagGeometryClass::VendorSameJob,
            geometry_receipt: None,
            work_width: Some(32),
            work_height: Some(24),
            work_to_destination: Some([0.5, 0.0, 0.0, 0.0, 0.5, 0.0]),
            destination_roi: Some(Roi {
                x: 0,
                y: 0,
                w: 16,
                h: 12,
            }),
        };
        assert_eq!(
            align_eye_bag(inputs),
            Err(Refusal::MissingInput("PF90116_GEOMETRY_RECEIPT"))
        );
    }

    #[test]
    fn vendor_receipt_binds_tensors_landmarks_and_writer_sites() {
        let (input, response, color) = model_tensors();
        let tensors = EyeBagModelTensors {
            model_id: MODEL_ID,
            input_781141184_rgb_chw: &input,
            response_3204548346_hw: &response,
            color_780929437_rgb_chw: &color,
        };
        let matrix = [0.5, 0.0, 0.0, 0.0, 0.5, 0.0];
        let roi = Roi {
            x: 0,
            y: 0,
            w: 16,
            h: 12,
        };
        let receipt = EyeBagVendorGeometryReceipt::from_capture(
            17, 17, 0x1234, 0x5678, &tensors, 32, 24, matrix, roi,
        );
        let base = EyeBagAlignmentInputs {
            ticket: 17,
            frame_width: 32,
            frame_height: 24,
            tensors,
            model_color_units: EyeBagColorUnits::SourceBytes,
            geometry_class: EyeBagGeometryClass::VendorSameJob,
            geometry_receipt: Some(receipt),
            work_width: Some(32),
            work_height: Some(24),
            work_to_destination: Some(matrix),
            destination_roi: Some(roi),
        };
        let aligned = align_eye_bag(base).expect("complete same-job receipt");
        assert!(aligned.vendor_geometry_claimed());
        assert_eq!(aligned.geometry_receipt, Some(receipt));

        assert_eq!(
            align_eye_bag(EyeBagAlignmentInputs {
                geometry_receipt: Some(EyeBagVendorGeometryReceipt {
                    landmark_receipt: 0,
                    ..receipt
                }),
                ..base
            }),
            Err(Refusal::MissingInput("PF90116_GEOMETRY_LANDMARKS"))
        );
        assert_eq!(
            align_eye_bag(EyeBagAlignmentInputs {
                geometry_receipt: Some(EyeBagVendorGeometryReceipt {
                    m2_call_site: 0,
                    ..receipt
                }),
                ..base
            }),
            Err(Refusal::MissingInput("PF90116_GEOMETRY_WRITER"))
        );
    }
}
