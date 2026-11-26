//! PF90116 same-frame producer contract.
//!
//! The producer loads and authenticates the local model, executes the complete
//! preprocessing/ORT/postprocessing chain and issues a same-job receipt only
//! after validating the actual model tensors.

use std::fs;
use std::path::{Path, PathBuf};

use aurora_portrait::eye_bag_alignment::{
    AlignedEyeBagPlanes, EyeBagVendorGeometryReceipt, VENDOR_AFFINE_FIT_M1_CALL_SITE,
    VENDOR_AFFINE_FIT_M2_CALL_SITE, VENDOR_PF90116_HOST_SHA256,
};
use aurora_portrait::eye_bag_alignment::{
    EyeBagAlignmentInputs, EyeBagColorUnits, EyeBagGeometryClass, EyeBagModelTensors, align_eye_bag,
    model_tensor_receipts,
};
use aurora_portrait::kernels::eye_bag::Roi;
use aurora_portrait::model_sha256::sha256;
use aurora_portrait::{Image, Refusal};

use crate::aimask::ort::{Provider, Runtime, Session};

use super::FaceInputBundle;

pub const PF90116_MODEL_ID: u64 = 814_444_604;
pub const PF90116_MODEL_PATH_ENV: &str = "AURORA_PF90116_MODEL";
pub const PF90116_DEFAULT_MODEL_PATH: &str = "models/pf90116/814444604.onnx";
pub const PF90116_GRAPH_INPUT_NAME: &str = "t39";
pub const PF90116_INPUT_NAME: &str = "781141184";
pub const PF90116_RESPONSE_NAME: &str = "3204548346";
pub const PF90116_COLOR_NAME: &str = "780929437";
pub const PF90116_MODEL_WIDTH: u32 = 512;
pub const PF90116_MODEL_HEIGHT: u32 = 384;
pub const PF90116_INPUT_SHAPE: [usize; 4] = [1, 3, 384, 512];
pub const PF90116_RESPONSE_SHAPE: [usize; 4] = [1, 1, 384, 512];
pub const PF90116_COLOR_SHAPE: [usize; 4] = [1, 3, 384, 512];
pub const PF90116_ACTIVE_GATE: f32 = 0.001;
pub const PF90116_INPUT_MEAN: [f32; 3] = [0.0; 3];
pub const PF90116_INPUT_SCALE: [f32; 3] = [255.0; 3];
pub const PF90116_MODEL_PAYLOAD_SHA256: [u8; 32] = [
    0x82, 0xb5, 0xe7, 0x57, 0x56, 0x06, 0x43, 0x0f, 0x89, 0x98, 0x66, 0x60, 0x15, 0x92, 0xc5, 0x0d,
    0xe7, 0xb8, 0x75, 0x63, 0x87, 0xdf, 0x0e, 0x13, 0x65, 0x4c, 0xf5, 0xe1, 0xff, 0xfb, 0x73, 0x12,
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

    fn from_model_tensors(job_ticket: u64, tensors: &EyeBagModelTensors<'_>) -> Result<Self, Refusal> {
        if job_ticket == 0 {
            return Err(Refusal::MissingInput("PF90116_SAME_JOB_TICKET"));
        }
        let [input, response, color, aggregate] = model_tensor_receipts(job_ticket, tensors);
        let receipts = Self {
            input,
            response,
            color,
            aggregate,
        };
        if receipts.complete() {
            Ok(receipts)
        } else {
            Err(Refusal::MissingInput("PF90116_MODEL_TENSOR_RECEIPT"))
        }
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

/// Geometry facts supplied by the same portrait job that owns the frame.
///
/// Tensor receipts are intentionally absent: they can only be issued after
/// preprocessing and inference have produced the actual buffers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pf90116FrameGeometry {
    pub vendor_job_id: u64,
    pub face_record_id: u64,
    pub landmark_receipt: u64,
    pub frame_to_model: [f32; 6],
    pub model_roi: [i32; 4],
    pub work_width: u32,
    pub work_height: u32,
    pub work_to_destination: [f32; 6],
    pub destination_roi: [i32; 4],
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
        if !matches!(
            self.color_domain,
            Pf90116ColorDomain::RawModel | Pf90116ColorDomain::SourceBytes
        ) {
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

    fn from_forward(
        binding: &Pf90116FrameBinding,
        geometry: &Pf90116FrameGeometry,
        tensors: &EyeBagModelTensors<'_>,
    ) -> Result<Self, Refusal> {
        let receipt = Self {
            job_ticket: binding.job_ticket,
            frame_size: binding.frame_size,
            frame_sha256: binding.frame_sha256,
            vendor_job_id: geometry.vendor_job_id,
            face_record_id: geometry.face_record_id,
            landmark_receipt: geometry.landmark_receipt,
            model_id: PF90116_MODEL_ID,
            model_payload_sha256: PF90116_MODEL_PAYLOAD_SHA256,
            model_net_sha256: PF90116_MODEL_NET_SHA256,
            tensors: Pf90116TensorReceipts::from_model_tensors(binding.job_ticket, tensors)?,
            input_order: Pf90116InputOrder::Rgb,
            input_mean: PF90116_INPUT_MEAN,
            input_scale: PF90116_INPUT_SCALE,
            frame_to_model: geometry.frame_to_model,
            model_roi: geometry.model_roi,
            work_width: geometry.work_width,
            work_height: geometry.work_height,
            work_to_destination: geometry.work_to_destination,
            destination_roi: geometry.destination_roi,
            color_domain: Pf90116ColorDomain::RawModel,
            degree_bits: binding.degree_bits,
        };
        receipt.validate(binding)?;
        Ok(receipt)
    }
}

struct LoadedEyeBagModel {
    runtime: &'static Runtime,
    primary: Session,
    cpu: Option<Session>,
}

/// Authenticated local model plus its executable forward boundary.
pub struct EyeBagFrameProducer {
    model_payload_sha256: [u8; 32],
    loaded: Option<LoadedEyeBagModel>,
}

impl EyeBagFrameProducer {
    pub fn open_from_env() -> Result<Self, Refusal> {
        let path = Self::discover_model_path()?;
        Self::open(&path)
    }

    fn discover_model_path() -> Result<PathBuf, Refusal> {
        Self::discover_model_path_from(
            std::env::var_os(PF90116_MODEL_PATH_ENV).map(PathBuf::from),
            Path::new(PF90116_DEFAULT_MODEL_PATH),
        )
    }

    fn discover_model_path_from(
        override_path: Option<PathBuf>,
        canonical_path: &Path,
    ) -> Result<PathBuf, Refusal> {
        match override_path.filter(|path| !path.as_os_str().is_empty()) {
            Some(path) => Ok(path),
            None if canonical_path.is_file() => Ok(canonical_path.to_path_buf()),
            None => Err(Refusal::MissingInput("PF90116_MODEL_PATH")),
        }
    }

    pub fn open(path: &Path) -> Result<Self, Refusal> {
        let payload = fs::read(path).map_err(|_| Refusal::MissingInput("PF90116_MODEL_FILE"))?;
        let model_payload_sha256 = sha256(&payload);
        if model_payload_sha256 != PF90116_MODEL_PAYLOAD_SHA256 {
            return Err(Refusal::ValueOutOfRange("PF90116_MODEL_PAYLOAD_SHA256"));
        }
        let runtime = Runtime::get().ok_or(Refusal::MissingInput("PF90116_MODEL_RUNTIME"))?;
        let primary = runtime
            .open_session(path)
            .map_err(|_| Refusal::MissingInput("PF90116_MODEL_SESSION"))?;
        let cpu = if primary.provider() == Provider::Cpu || runtime.is_native_only() {
            None
        } else {
            Some(
                runtime
                    .open_cpu_session(path)
                    .map_err(|_| Refusal::MissingInput("PF90116_MODEL_CPU_SESSION"))?,
            )
        };
        Ok(Self {
            model_payload_sha256,
            loaded: Some(LoadedEyeBagModel {
                runtime,
                primary,
                cpu,
            }),
        })
    }

    pub fn model_payload_sha256(&self) -> [u8; 32] {
        self.model_payload_sha256
    }

    pub fn produce(
        &self,
        job_ticket: u64,
        frame: &Image,
        face: &FaceInputBundle,
        degree: f32,
        geometry: &Pf90116FrameGeometry,
    ) -> Result<AlignedEyeBagPlanes, Refusal> {
        let binding = Pf90116FrameBinding::from_face_bundle(
            job_ticket,
            frame,
            face,
            geometry.face_record_id,
            geometry.landmark_receipt,
            degree,
        )?;
        self.produce_validated(&binding, frame, geometry)
    }

    /// Executes preprocessing, inference, output validation and F1e alignment.
    pub fn produce_validated(
        &self,
        binding: &Pf90116FrameBinding,
        frame: &Image,
        geometry: &Pf90116FrameGeometry,
    ) -> Result<AlignedEyeBagPlanes, Refusal> {
        let loaded = self
            .loaded
            .as_ref()
            .ok_or(Refusal::MissingInput("PF90116_MODEL_RUNTIME"))?;
        validate_frame_binding(binding, frame)?;
        validate_forward_geometry(binding, geometry)?;
        let mut input = preprocess_frame(frame, geometry)?;
        let (response, color) = loaded.run_with_cpu_fallback(&mut input)?;
        let pixels = (PF90116_MODEL_WIDTH * PF90116_MODEL_HEIGHT) as usize;
        let tensors = EyeBagModelTensors {
            model_id: PF90116_MODEL_ID,
            input_781141184_rgb_chw: &input,
            response_3204548346_hw: &response[..pixels],
            color_780929437_rgb_chw: &color[..pixels * 3],
        };
        let receipt = Pf90116FrameReceipt::from_forward(binding, geometry, &tensors)?;
        self.align_model_outputs(
            binding,
            &receipt,
            &tensors,
            EyeBagColorUnits::RawModel,
        )
    }

    /// Converts authenticated model outputs into the stage's full-frame planes.
    /// Receipt issuance remains owned by the model forward; this method cannot
    /// turn candidate geometry into a same-job receipt.
    pub fn align_model_outputs(
        &self,
        binding: &Pf90116FrameBinding,
        receipt: &Pf90116FrameReceipt,
        tensors: &EyeBagModelTensors<'_>,
        model_color_units: EyeBagColorUnits,
    ) -> Result<AlignedEyeBagPlanes, Refusal> {
        receipt.validate(binding)?;
        if receipt.tensors != Pf90116TensorReceipts::from_model_tensors(binding.job_ticket, tensors)? {
            return Err(Refusal::MissingInput("PF90116_MODEL_TENSOR_RECEIPT"));
        }
        let declared_units = match receipt.color_domain {
            Pf90116ColorDomain::RawModel => EyeBagColorUnits::RawModel,
            Pf90116ColorDomain::SourceBytes => EyeBagColorUnits::SourceBytes,
        };
        if model_color_units != declared_units {
            return Err(Refusal::MissingInput("PF90116_COLOR_DOMAIN_RECEIPT"));
        }
        align_eye_bag(EyeBagAlignmentInputs {
            ticket: binding.job_ticket,
            frame_width: binding.frame_size[0],
            frame_height: binding.frame_size[1],
            tensors: *tensors,
            model_color_units,
            geometry_class: EyeBagGeometryClass::VendorSameJob,
            geometry_receipt: Some(receipt.geometry_receipt()),
            work_width: Some(receipt.work_width),
            work_height: Some(receipt.work_height),
            work_to_destination: Some(receipt.work_to_destination),
            destination_roi: Some(Roi {
                x: receipt.destination_roi[0],
                y: receipt.destination_roi[1],
                w: receipt.destination_roi[2],
                h: receipt.destination_roi[3],
            }),
        })
    }

    #[cfg(test)]
    pub(crate) fn for_tests() -> Self {
        Self {
            model_payload_sha256: PF90116_MODEL_PAYLOAD_SHA256,
            loaded: None,
        }
    }
}

impl LoadedEyeBagModel {
    fn run_with_cpu_fallback(
        &self,
        input: &mut [f32],
    ) -> Result<(Vec<f32>, Vec<f32>), Refusal> {
        let shape: [i64; 4] = PF90116_INPUT_SHAPE.map(|value| value as i64);
        let mut run = |session: &Session| {
            validate_model_contract(session)?;
            let outputs = self
                .runtime
                .run(session, input, &shape)
                .map_err(|_| Refusal::MissingInput("PF90116_MODEL_INFERENCE"))?;
            model_outputs(session, outputs)
        };
        match run(&self.primary) {
            Ok(outputs) => Ok(outputs),
            Err(primary_error) => {
                let Some(cpu) = self.cpu.as_ref() else {
                    return Err(primary_error);
                };
                log::warn!(
                    "PF90116 {} provider failed ({primary_error:?}); retrying on CPU",
                    self.primary.provider().label()
                );
                run(cpu)
            }
        }
    }
}

fn validate_model_contract(session: &Session) -> Result<(), Refusal> {
    let (input_name, input_dims, input_type) = session.input_contract();
    if input_name != PF90116_GRAPH_INPUT_NAME
        || !dims_match(input_dims, &PF90116_INPUT_SHAPE)
        || input_type != 1
    {
        return Err(Refusal::ShapeMismatch("PF90116_MODEL_INPUT_CONTRACT"));
    }
    let outputs = session.output_contract();
    if outputs.len() != 2 {
        return Err(Refusal::ShapeMismatch("PF90116_MODEL_OUTPUTS"));
    }
    let mut response = false;
    let mut color = false;
    for (name, dims, element_type) in outputs {
        if element_type != 1 {
            return Err(Refusal::ShapeMismatch("PF90116_MODEL_OUTPUT_DTYPE"));
        }
        match name {
            PF90116_RESPONSE_NAME
                if !response && dims_match(dims, &PF90116_RESPONSE_SHAPE) =>
            {
                response = true;
            }
            PF90116_COLOR_NAME if !color && dims_match(dims, &PF90116_COLOR_SHAPE) => {
                color = true;
            }
            _ => return Err(Refusal::ShapeMismatch("PF90116_MODEL_OUTPUT_CONTRACT")),
        }
    }
    if !response || !color {
        return Err(Refusal::MissingInput("PF90116_MODEL_OUTPUT_CONTRACT"));
    }
    Ok(())
}

fn validate_frame_binding(binding: &Pf90116FrameBinding, frame: &Image) -> Result<(), Refusal> {
    if binding.frame_size != [frame.width, frame.height] {
        return Err(Refusal::ValueOutOfRange("PF90116_FRAME_SIZE"));
    }
    if binding.frame_sha256 == [0; 32] || binding.frame_sha256 != sha256(&frame.pixels) {
        return Err(Refusal::ValueOutOfRange("PF90116_FRAME_DIGEST"));
    }
    check_degree(f32::from_bits(binding.degree_bits))
}

fn validate_forward_geometry(
    binding: &Pf90116FrameBinding,
    geometry: &Pf90116FrameGeometry,
) -> Result<(), Refusal> {
    if geometry.vendor_job_id == 0 {
        return Err(Refusal::MissingInput("PF90116_GEOMETRY_SAME_JOB"));
    }
    if geometry.face_record_id == 0 || geometry.face_record_id != binding.face_record_id {
        return Err(Refusal::MissingInput("PF90116_GEOMETRY_FACE_RECORD"));
    }
    if geometry.landmark_receipt == 0 || geometry.landmark_receipt != binding.landmark_receipt {
        return Err(Refusal::MissingInput("PF90116_GEOMETRY_LANDMARKS"));
    }
    if !valid_affine(geometry.frame_to_model) || !valid_model_roi(geometry.model_roi) {
        return Err(Refusal::ValueOutOfRange("PF90116_MODEL_INPUT_GEOMETRY"));
    }
    if geometry.work_width == 0
        || geometry.work_height == 0
        || !valid_affine(geometry.work_to_destination)
    {
        return Err(Refusal::ValueOutOfRange("PF90116_WARP_AFFINE"));
    }
    if !valid_frame_roi(geometry.destination_roi, binding.frame_size) {
        return Err(Refusal::ValueOutOfRange("PF90116_DESTINATION_ROI"));
    }
    Ok(())
}

fn preprocess_frame(
    frame: &Image,
    geometry: &Pf90116FrameGeometry,
) -> Result<Vec<f32>, Refusal> {
    let inverse = invert_affine(geometry.frame_to_model)?;
    let [roi_x, roi_y, roi_w, roi_h] = geometry.model_roi;
    let mut input = vec![0.0; PF90116_INPUT_SHAPE.iter().product()];
    let plane = (PF90116_MODEL_WIDTH * PF90116_MODEL_HEIGHT) as usize;
    for y in roi_y..roi_y + roi_h {
        for x in roi_x..roi_x + roi_w {
            let mx = f64::from(x);
            let my = f64::from(y);
            let sx = inverse[0] * mx + inverse[1] * my + inverse[2];
            let sy = inverse[3] * mx + inverse[4] * my + inverse[5];
            for channel in 0..3 {
                let sample = sample_rgba_border_zero(frame, channel, sx, sy);
                input[channel * plane + y as usize * PF90116_MODEL_WIDTH as usize + x as usize] =
                    (sample - PF90116_INPUT_MEAN[channel]) / PF90116_INPUT_SCALE[channel];
            }
        }
    }
    Ok(input)
}

fn sample_rgba_border_zero(frame: &Image, channel: usize, x: f64, y: f64) -> f32 {
    let width = frame.width as usize;
    let height = frame.height as usize;
    if x <= -1.0 || y <= -1.0 || x >= width as f64 || y >= height as f64 {
        return 0.0;
    }
    let x0 = x.floor() as i64;
    let y0 = y.floor() as i64;
    let wx = (x - x0 as f64) as f32;
    let wy = (y - y0 as f64) as f32;
    let sample = |px: i64, py: i64| {
        if (0..width as i64).contains(&px) && (0..height as i64).contains(&py) {
            f32::from(frame.pixels[(py as usize * width + px as usize) * 4 + channel])
        } else {
            0.0
        }
    };
    let top = sample(x0, y0) + (sample(x0 + 1, y0) - sample(x0, y0)) * wx;
    let bottom = sample(x0, y0 + 1) + (sample(x0 + 1, y0 + 1) - sample(x0, y0 + 1)) * wx;
    top + (bottom - top) * wy
}

fn model_outputs(
    session: &Session,
    outputs: Vec<Vec<f32>>,
) -> Result<(Vec<f32>, Vec<f32>), Refusal> {
    let contract = session.output_contract();
    if outputs.len() != 2 || contract.len() != outputs.len() {
        return Err(Refusal::ShapeMismatch("PF90116_MODEL_OUTPUTS"));
    }
    let mut response = None;
    let mut color = None;
    for ((name, dims, element_type), plane) in contract.into_iter().zip(outputs) {
        if element_type != 1 {
            return Err(Refusal::ShapeMismatch("PF90116_MODEL_OUTPUT_DTYPE"));
        }
        match name {
            PF90116_RESPONSE_NAME => {
                if !dims_match(dims, &PF90116_RESPONSE_SHAPE)
                    || plane.len() != PF90116_RESPONSE_SHAPE.iter().product::<usize>()
                    || response.replace(plane).is_some()
                {
                    return Err(Refusal::ShapeMismatch("PF90116_MODEL_RESPONSE_3204548346"));
                }
            }
            PF90116_COLOR_NAME => {
                if !dims_match(dims, &PF90116_COLOR_SHAPE)
                    || plane.len() != PF90116_COLOR_SHAPE.iter().product::<usize>()
                    || color.replace(plane).is_some()
                {
                    return Err(Refusal::ShapeMismatch("PF90116_MODEL_COLOR_780929437"));
                }
            }
            _ => return Err(Refusal::ShapeMismatch("PF90116_MODEL_OUTPUT_NAME")),
        }
    }
    let response = response.ok_or(Refusal::MissingInput("PF90116_MODEL_RESPONSE_3204548346"))?;
    let color = color.ok_or(Refusal::MissingInput("PF90116_MODEL_COLOR_780929437"))?;
    validate_model_output_values(&response, &color)?;
    Ok((response, color))
}

fn validate_model_output_values(response: &[f32], color: &[f32]) -> Result<(), Refusal> {
    if response
        .iter()
        .chain(color)
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("PF90116_MODEL_OUTPUT_VALUES"));
    }
    if response.iter().any(|value| !(0.0..=1.0).contains(value)) {
        return Err(Refusal::ValueOutOfRange("PF90116_MODEL_RESPONSE_RANGE"));
    }
    Ok(())
}

fn dims_match(actual: &[i64], expected: &[usize]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(actual, expected)| *actual == *expected as i64)
}

fn invert_affine(matrix: [f32; 6]) -> Result<[f64; 6], Refusal> {
    if !valid_affine(matrix) {
        return Err(Refusal::ValueOutOfRange("PF90116_MODEL_INPUT_GEOMETRY"));
    }
    let [a, b, c, d, e, f] =
        matrix.map(|value| f64::from(value));
    let determinant = a * e - b * d;
    Ok([
        e / determinant,
        -b / determinant,
        (b * f - c * e) / determinant,
        -d / determinant,
        a / determinant,
        (c * d - a * f) / determinant,
    ])
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
    fn receipt_accepts_raw_model_color_and_rejects_bad_geometry() {
        let binding = binding();
        let raw_color = Pf90116FrameReceipt {
            color_domain: Pf90116ColorDomain::RawModel,
            ..receipt(&binding)
        };
        raw_color
            .validate(&binding)
            .expect("RawModel color is converted by the alignment contract");
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
    fn preprocessing_uses_rgb_bytes_and_same_job_affine() {
        let frame = Image::new(
            2,
            2,
            vec![
                255, 0, 127, 255, 0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255,
            ],
        )
        .expect("frame");
        let geometry = Pf90116FrameGeometry {
            vendor_job_id: 1,
            face_record_id: 2,
            landmark_receipt: 3,
            frame_to_model: [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
            model_roi: [1, 1, 1, 1],
            work_width: 1,
            work_height: 1,
            work_to_destination: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            destination_roi: [0, 0, 1, 1],
        };
        let input = preprocess_frame(&frame, &geometry).expect("preprocess");
        let pixels = (PF90116_MODEL_WIDTH * PF90116_MODEL_HEIGHT) as usize;
        assert_eq!(input[PF90116_MODEL_WIDTH as usize + 1], 1.0);
        assert_eq!(input[pixels + PF90116_MODEL_WIDTH as usize + 1], 0.0);
        assert!(
            (input[pixels * 2 + PF90116_MODEL_WIDTH as usize + 1] - 127.0 / 255.0).abs()
                < f32::EPSILON
        );
        assert_eq!(input[0], 0.0);
    }

    #[test]
    fn model_output_validation_rejects_bad_response_and_non_finite_color() {
        let pixels = (PF90116_MODEL_WIDTH * PF90116_MODEL_HEIGHT) as usize;
        let mut response = vec![0.25; pixels];
        let color = vec![0.0; pixels * 3];
        validate_model_output_values(&response, &color).expect("valid output values");
        response[0] = 1.01;
        assert_eq!(
            validate_model_output_values(&response, &color),
            Err(Refusal::ValueOutOfRange("PF90116_MODEL_RESPONSE_RANGE"))
        );
        response[0] = 0.25;
        let mut color = color;
        color[0] = f32::NAN;
        assert_eq!(
            validate_model_output_values(&response, &color),
            Err(Refusal::ValueOutOfRange("PF90116_MODEL_OUTPUT_VALUES"))
        );
    }

    #[test]
    fn model_discovery_uses_existing_default_and_rejects_missing_artifact() {
        let artifact = std::env::temp_dir().join(format!(
            "aurora-pf90116-model-discovery-{}.onnx",
            std::process::id()
        ));
        fs::write(&artifact, []).expect("create discovery artifact");
        assert_eq!(
            EyeBagFrameProducer::discover_model_path_from(None, &artifact),
            Ok(artifact.clone())
        );
        fs::remove_file(&artifact).expect("remove discovery artifact");
        assert_eq!(
            EyeBagFrameProducer::discover_model_path_from(None, &artifact),
            Err(Refusal::MissingInput("PF90116_MODEL_PATH"))
        );
        let override_path = artifact.with_extension("override.onnx");
        assert_eq!(
            EyeBagFrameProducer::discover_model_path_from(Some(override_path.clone()), &artifact),
            Ok(override_path)
        );
    }
}
