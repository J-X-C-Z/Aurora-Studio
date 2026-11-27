use std::collections::BTreeSet;

use aurora_portrait::kernels::plump_pipeline::{
    ChannelMap, ColorEncoding, Error as PipelineError, FacePlumpFactory, ModelInput,
    OutputSink, PlumpLayer, PlumpProducer, Roi, SessionOutput, SlotValues,
};
use aurora_portrait::model_sha256::sha256;
use aurora_portrait::plump_session::{LoadedSession, SessionBinding};
use aurora_portrait::{Image, Refusal};

use super::PlumpRasterBackend;

pub const PLUMP_PFS: [u32; 8] = [
    30_191, 30_194, 30_197, 30_198, 30_203, 30_204, 30_207, 30_210,
];
pub const SAME_JOB_INPUT_REFUSAL: &str = "PF30191_PF30210_PLUMP_SAME_JOB_INPUTS";
pub const PLUMP_LANDMARK_COUNT: usize = 414;

pub fn supports(pf: u32) -> bool {
    PLUMP_PFS.contains(&pf)
}

pub fn full_face_descriptor_from_ui(value: f32) -> Result<f32, Refusal> {
    if value.is_finite() && value.fract() == 0.0 && (0.0..=4.0).contains(&value) {
        Ok(value)
    } else {
        Err(Refusal::ValueOutOfRange("PF30210 UI"))
    }
}

/// Read a saved full-face level, including normalized values written by older
/// sidecars. A literal 1 is level one, never the normalized maximum. New edits
/// go through the strict integer-only UI contract above.
pub fn full_face_descriptor_from_store(value: f32) -> Result<f32, Refusal> {
    let level = if value > 0.0 && value < 1.0 {
        (value * 4.0).round().clamp(1.0, 4.0)
    } else {
        value
    };
    full_face_descriptor_from_ui(level)
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlumpSameFrameEvidence {
    pub source_sha256: [u8; 32],
    pub source_size: [u32; 2],
    pub landmarks: Vec<[f32; 2]>,
    pub landmarks_sha256: [u8; 32],
    pub roi: Roi,
    pub roi_sha256: [u8; 32],
    pub weights: [f32; 17],
    pub plane4_sha256: Option<[u8; 32]>,
    pub weights_sha256: [u8; 32],
}

impl PlumpSameFrameEvidence {
    pub fn from_same_frame(
        source: &Image,
        landmarks: &[[f32; 2]],
        roi: Roi,
        weights: [f32; 17],
        plane4: Option<&[f32]>,
    ) -> Result<Self, Refusal> {
        if landmarks.len() != PLUMP_LANDMARK_COUNT
            || landmarks.iter().flatten().any(|value| !value.is_finite())
        {
            return Err(Refusal::ShapeMismatch("FacePlump 414 landmarks"));
        }
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("FacePlump 17 weights"));
        }
        let expected = source.width as usize * source.height as usize * 4;
        if source.pixels.len() != expected {
            return Err(Refusal::ShapeMismatch("FacePlump source frame"));
        }
        if roi.width == 0
            || roi.height == 0
            || roi.x > source.width
            || roi.y > source.height
            || roi.width > source.width - roi.x
            || roi.height > source.height - roi.y
        {
            return Err(Refusal::ValueOutOfRange("FacePlump ROI"));
        }
        let landmarks_sha256 = sha256_f32_2(landmarks);
        let roi_sha256 = sha256_roi(roi);
        let plane4_sha256 = plane4.map(sha256_f32);
        if let Some(plane) = plane4 {
            if plane.len() != roi.width as usize * roi.height as usize
                || plane.iter().any(|value| !value.is_finite())
            {
                return Err(Refusal::ShapeMismatch("FacePlump 4th plane"));
            }
        }
        let mut weight_bytes = Vec::with_capacity(17 * 4 + 32);
        for value in weights {
            weight_bytes.extend_from_slice(&value.to_le_bytes());
        }
        if let Some(plane_hash) = plane4_sha256 {
            weight_bytes.extend_from_slice(&plane_hash);
        }
        Ok(Self {
            source_sha256: sha256(&source.pixels),
            source_size: [source.width, source.height],
            landmarks: landmarks.to_vec(),
            landmarks_sha256,
            roi,
            roi_sha256,
            weights,
            plane4_sha256,
            weights_sha256: sha256(&weight_bytes),
        })
    }

    pub fn validate(
        &self,
        source: &Image,
        roi: Roi,
        weights: [f32; 17],
    ) -> Result<(), Refusal> {
        if self.source_size != [source.width, source.height]
            || self.source_sha256 != sha256(&source.pixels)
        {
            return Err(Refusal::ValueOutOfRange("FacePlump same-frame source"));
        }
        if self.landmarks.len() != PLUMP_LANDMARK_COUNT
            || self.landmarks_sha256 != sha256_f32_2(&self.landmarks)
        {
            return Err(Refusal::ValueOutOfRange("FacePlump same-frame landmarks"));
        }
        if self.roi != roi || self.roi_sha256 != sha256_roi(roi) {
            return Err(Refusal::ValueOutOfRange("FacePlump same-frame ROI"));
        }
        if self.weights != weights {
            return Err(Refusal::ValueOutOfRange("FacePlump same-frame weights"));
        }
        let mut weight_bytes = Vec::with_capacity(17 * 4 + 32);
        for value in weights {
            weight_bytes.extend_from_slice(&value.to_le_bytes());
        }
        if let Some(plane_hash) = self.plane4_sha256 {
            weight_bytes.extend_from_slice(&plane_hash);
        }
        if self.weights_sha256 != sha256(&weight_bytes) {
            return Err(Refusal::ValueOutOfRange("FacePlump same-frame weights"));
        }
        Ok(())
    }
}

fn sha256_f32(values: &[f32]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    sha256(&bytes)
}

fn sha256_f32_2(values: &[[f32; 2]]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(values.len() * 8);
    for point in values {
        bytes.extend_from_slice(&point[0].to_le_bytes());
        bytes.extend_from_slice(&point[1].to_le_bytes());
    }
    sha256(&bytes)
}

fn sha256_roi(roi: Roi) -> [u8; 32] {
    sha256(&[
        roi.x.to_le_bytes(),
        roi.y.to_le_bytes(),
        roi.width.to_le_bytes(),
        roi.height.to_le_bytes(),
    ]
    .concat())
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlumpCompositeReceipt {
    pub ticket: u64,
    pub face_id: u32,
    pub requested_pfs: Vec<u32>,
    pub source_sha256: [u8; 32],
    pub backend: PlumpRasterBackend,
    pub model_id: u32,
    pub net_sha256: [u8; 32],
    pub model_input_digest: [u8; 32],
    pub output_tensor: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlumpFinalComposite {
    pub receipt: PlumpCompositeReceipt,
    pub frame: Image,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlumpProducerInputs {
    pub ticket: u64,
    pub face_id: u32,
    pub slots: [f32; 17],
    pub requested_pfs: Vec<u32>,
    pub full_face_descriptor: Option<f32>,
    pub session: SessionBinding,
    pub backend: PlumpRasterBackend,
    pub model_input: Option<ModelInput>,
    pub model_output: Option<SessionOutput>,
    pub channel_map: ChannelMap,
    pub color_encoding: ColorEncoding,
    pub roi: Roi,
    pub final_composite: Option<PlumpFinalComposite>,
    pub same_frame: Option<PlumpSameFrameEvidence>,
}

impl PlumpProducerInputs {
    #[allow(clippy::too_many_arguments)]
    pub fn from_loaded_session(
        session: &LoadedSession,
        ticket: u64,
        face_id: u32,
        slots: [f32; 17],
        requested_pfs: Vec<u32>,
        full_face_descriptor: Option<f32>,
        model_input: Option<ModelInput>,
        model_output: Option<SessionOutput>,
        channel_map: ChannelMap,
        color_encoding: ColorEncoding,
        roi: Roi,
        backend: PlumpRasterBackend,
        final_composite: Option<PlumpFinalComposite>,
        same_frame: Option<PlumpSameFrameEvidence>,
    ) -> Self {
        Self {
            ticket,
            face_id,
            slots,
            requested_pfs,
            full_face_descriptor,
            session: session.binding(),
            backend,
            model_input,
            model_output,
            channel_map,
            color_encoding,
            roi,
            final_composite,
            same_frame,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlumpGroupRun {
    pub requested_pfs: Vec<u32>,
    pub backend: PlumpRasterBackend,
    pub image: Image,
    pub layer: PlumpLayer,
}

pub fn apply(
    current: &Image,
    active_pfs: &BTreeSet<u32>,
    expected_full_face_descriptor: Option<f32>,
    job_id: Option<u64>,
    input: Option<&PlumpProducerInputs>,
) -> Result<Option<PlumpGroupRun>, Refusal> {
    if active_pfs.is_empty() {
        return Ok(None);
    }
    let input = input.ok_or(Refusal::MissingInput(SAME_JOB_INPUT_REFUSAL))?;
    let job_id = job_id.ok_or(Refusal::MissingInput("plump renderer job"))?;
    if input.ticket != job_id {
        return Err(Refusal::ValueOutOfRange("plump same-job ticket"));
    }
    if input.face_id == 0 {
        return Err(Refusal::ValueOutOfRange("plump face id"));
    }
    input
        .same_frame
        .as_ref()
        .ok_or(Refusal::MissingInput("FacePlump same-frame evidence"))?
        .validate(current, input.roi, input.slots)?;

    let requested_pfs = active_pfs.iter().copied().collect::<Vec<_>>();
    if input.requested_pfs != requested_pfs {
        return Err(Refusal::ValueOutOfRange("plump requested PF set"));
    }
    if input.full_face_descriptor != expected_full_face_descriptor {
        return Err(Refusal::ValueOutOfRange(
            "plump PF30210 native descriptor",
        ));
    }

    let values = SlotValues::new(input.slots).map_err(pipeline_refusal)?;
    let request = FacePlumpFactory::build_request(
        input.face_id,
        values,
        &requested_pfs,
        input.full_face_descriptor,
    )
    .map_err(pipeline_refusal)?;
    if !request.is_active() {
        return Err(Refusal::ValueOutOfRange("plump active request"));
    }
    for pf in &requested_pfs {
        let active = if *pf == 30_210 {
            request
                .full_face_plump
                .is_some_and(|descriptor| descriptor != 0.0)
        } else {
            request
                .slot_for_pf(*pf)
                .is_some_and(|value| value != 0.0)
        };
        if !active {
            return Err(Refusal::ValueOutOfRange("plump active PF"));
        }
    }

    let model_input = input
        .model_input
        .as_ref()
        .ok_or(Refusal::MissingInput("plump_model_input"))?;
    let model_output = input
        .model_output
        .as_ref()
        .ok_or(Refusal::MissingInput("plump_model_output"))?;
    let layer = PlumpProducer::produce(
        &input.session,
        model_input,
        model_output,
        input.channel_map,
        input.color_encoding,
        input.roi,
        OutputSink::FacialPlumpingLayer,
    )
    .map_err(pipeline_refusal)?;

    let composite = input
        .final_composite
        .as_ref()
        .ok_or(Refusal::MissingInput("facial_plumping_final_composite"))?;
    validate_composite(
        current,
        input.ticket,
        input.face_id,
        &requested_pfs,
        model_input,
        model_output,
        composite,
        input.backend,
    )?;
    if layer.alpha_binary.bytes.iter().all(|&alpha| alpha == 0)
        && composite.frame.pixels == current.pixels
    {
        return Err(Refusal::MissingInput("FacePlump split-mask coverage"));
    }

    Ok(Some(PlumpGroupRun {
        requested_pfs,
        backend: composite.receipt.backend,
        image: composite.frame.clone(),
        layer,
    }))
}

fn validate_composite(
    current: &Image,
    ticket: u64,
    face_id: u32,
    requested_pfs: &[u32],
    model_input: &ModelInput,
    model_output: &SessionOutput,
    composite: &PlumpFinalComposite,
    backend: PlumpRasterBackend,
) -> Result<(), Refusal> {
    let receipt = &composite.receipt;
    if receipt.ticket != ticket || receipt.face_id != face_id {
        return Err(Refusal::ValueOutOfRange("plump final composite job"));
    }
    if receipt.requested_pfs != requested_pfs {
        return Err(Refusal::ValueOutOfRange("plump final composite PF set"));
    }
    if receipt.source_sha256 != sha256(&current.pixels) {
        return Err(Refusal::ValueOutOfRange(
            "plump final composite same-frame digest",
        ));
    }
    if receipt.backend != backend {
        return Err(Refusal::ValueOutOfRange(
            "plump final composite backend",
        ));
    }
    if receipt.model_id != model_output.stamp.model_id
        || receipt.net_sha256 != model_output.stamp.net_sha256
        || receipt.model_input_digest != model_input.digest
        || receipt.output_tensor != model_output.stamp.output_tensor
    {
        return Err(Refusal::ValueOutOfRange(
            "plump final composite model binding",
        ));
    }
    if composite.frame.width != current.width || composite.frame.height != current.height {
        return Err(Refusal::ShapeMismatch("facial_plumping_final_composite"));
    }
    composite
        .frame
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("facial_plumping_final_composite"))
}

fn pipeline_refusal(error: PipelineError) -> Refusal {
    match error {
        PipelineError::ShapeMismatch(name) => Refusal::ShapeMismatch(name),
        PipelineError::ValueOutOfRange(name) => Refusal::ValueOutOfRange(name),
        PipelineError::MissingInput(name) => Refusal::MissingInput(name),
        PipelineError::SessionMismatch(name) => Refusal::ValueOutOfRange(name),
        PipelineError::UnsupportedPf(pf) => Refusal::NotImplemented(pf),
        PipelineError::NormalCompositorRejected => {
            Refusal::ValueOutOfRange("facial_plumping_normal_compositor")
        }
    }
}
