//! Bridges portrait parameters (`pc:<id>`) to `aurora-portrait` CPU kernels.
//!
//! Only effects whose inputs are available are applied. Effects that need face
//! geometry, per-effect model outputs or materials that are not present are
//! skipped with a debug line instead of being approximated.
//!
//! Mask planes come from the local people parser when its ONNX model is
//! available. No external application install is read by this module.

#[cfg(windows)]
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

use aurora_portrait::cuda::{CudaError, Dispatch};
use aurora_portrait::cuda_pf8050::{self, MESH_POINTS};
use aurora_portrait::cuda_pf90021::{
    Direction as Pf90021Direction, RuntimeScalars as Pf90021RuntimeScalars, VolumeLut64,
};
use aurora_portrait::device::{Backend as DeviceBackend, DeviceRefusal, Selection};
#[cfg(windows)]
use aurora_portrait::cuda_hair_detail::HairDetail;
#[cfg(windows)]
use aurora_portrait::cuda_mesh::MeshRaster;
use aurora_portrait::cuda_hair_detail::{
    GlossProducer, Roi as HairRoi, SmoothParams, SmoothProducer,
};
use aurora_portrait::cuda_makeup::{MakeupRoute, RouteInputs};
use aurora_portrait::cuda_stray_hair::Params as StrayHairParams;
use aurora_portrait::kernels::face_detail::{Rect as ToothRect, ToothRepairProducer};
use aurora_portrait::kernels::{face_geometry, face_mesh, face_width_414, forehead_common};
use aurora_portrait::kernels::makeup_material::{
    MakeupLayer, MaskChannel as MakeupMaskChannel, color_makeup, image_makeup,
    image_makeup_unmasked,
};
use aurora_portrait::kernels::moisture::{FaceParams, Lut1D, MaskChannel, SkinMap};
use aurora_portrait::kernels::skin_adjust::{
    self, AdjustLut, BodyHlfUniforms, Roi, StoreMode, WrinkleUniforms,
};
use aurora_portrait::kernels::smile_warp::{self, Rect};
use aurora_portrait::kernels::suit_makeup::{SuitPass, SuitPassKind};
use aurora_portrait::kernels::teeth_blend::AlignmentFinalBlendOutcome;
use aurora_portrait::kernels::teeth_blend::AlignmentFinalBlendJob;
use aurora_portrait::{Access, Image, Refusal};

use crate::wrinkle_legacy::WrinkleProducerInputs;
use crate::portrait_makeup_bundle::{B3_MAKEUP_ROWS, OwnedMakeupBundle};
use crate::NormRect;

#[path = "portrait_slim_stage.rs"]
mod portrait_slim_stage;
#[path = "portrait_v2_mesh.rs"]
mod portrait_v2_mesh;
pub use portrait_slim_stage::SlimProducerInputs;

#[path = "portrait_landmark_stage.rs"]
mod portrait_landmark_stage;
pub use portrait_landmark_stage::{
    FACE_LANDMARK_52_MODEL_SHA256, FACE_LANDMARK_52_RESOURCE, FaceLandmarkProducerInputs,
};

#[path = "portrait_face_input_stage.rs"]
pub mod portrait_face_input_stage;
pub use portrait_face_input_stage::{
    C1FaceDetection, C1FaceMesh, C1FaceParsing, C1_FACE_DETECTION_MODEL_ID,
    C1_FACE_DETECTION_MODEL_SHA256, C1_FACE_MESH_MODEL_ID, C1_FACE_MESH_MODEL_SHA256,
    C1_FACE_PARSING_MODEL_ID, C1_FACE_PARSING_MODEL_SHA256,
    FACE_LANDMARK_1063025060_MODEL_SHA256, FACE_LANDMARK_323773030_MODEL_SHA256,
    FACE_LANDMARK_CONSTANTS_SHA256, FaceInputBundle, FaceInputModelPaths, FaceInputProducer,
    FaceInputFileStamp, FaceInputResultKey, FaceInputSession, FaceInputSessionKey,
    FaceModelIdentity, FaceNormalization, FaceStageTransform, FACE_STAGE_TRANSFORM_REFUSAL,
};

#[path = "portrait_eye_bag_stage.rs"]
mod portrait_eye_bag_stage;
pub use portrait_eye_bag_stage::{
    EyeBagFrameProducer, Pf90116ColorDomain, Pf90116FrameBinding, Pf90116FrameGeometry,
    Pf90116FrameReceipt,
    Pf90116InputOrder, Pf90116TensorReceipts, PF90116_ACTIVE_GATE, PF90116_COLOR_NAME,
    PF90116_COLOR_SHAPE, PF90116_INPUT_MEAN, PF90116_INPUT_NAME, PF90116_INPUT_SCALE,
    PF90116_INPUT_SHAPE, PF90116_MODEL_HEIGHT, PF90116_MODEL_ID, PF90116_MODEL_NET_SHA256,
    PF90116_MODEL_PAYLOAD_SHA256, PF90116_MODEL_WIDTH, PF90116_RESPONSE_NAME,
    PF90116_RESPONSE_SHAPE,
};

#[path = "portrait_face_stage.rs"]
mod portrait_face_stage;
pub use portrait_face_stage::{FaceGeometry, FaceGeometryDraw, FaceGeometryInputs};
use portrait_face_stage::FaceGeometryRun;

#[path = "portrait_pf8033_cpu_forward.rs"]
mod portrait_pf8033_cpu_forward;
pub use portrait_pf8033_cpu_forward::{
    apply_pf8033_cpu_forward, build_pf8033_cpu_forward_row, load_pf8033_cpu_forward_artifacts,
    pf8033_cpu_forward_cpu_oracle, Pf8033CpuForwardArtifacts, Pf8033CpuForwardReceipt,
    Pf8033CpuForwardRun, Pf8033HostSplitReceiptData, Pf8033ModelReceipt, Pf8033NetworkReceipt,
    Pf8033TensorReceipt, Pf8033WarpReceipt, PF8033_CPU_FORWARD_ARTIFACTS_ENV,
    PF8033_CPU_FORWARD_RECEIPT_ENV, PF8033_DX_FILE, PF8033_DY_FILE, PF8033_FINAL_MODEL_ID,
    PF8033_FINAL_TENSOR_FILE, PF8033_FINAL_TENSOR_ID, PF8033_FRAME_HEIGHT, PF8033_FRAME_WIDTH,
    PF8033_MASK_FILE, PF8033_MODEL_HEIGHT, PF8033_MODEL_IDS, PF8033_MODEL_ROOT_ENV,
    PF8033_MODEL_WIDTH,
};

#[path = "portrait_face_geometry_provider.rs"]
mod portrait_face_geometry_provider;
pub use portrait_face_geometry_provider::{
    build_face_geometry_inputs, landmark_receipt_hashes, mesh_receipt, provider_route,
    FaceGeometryProviderInputs, FaceGeometryProviderMesh, FaceGeometryProviderMeshReceipt,
    FaceGeometryProviderReceipt, FaceGeometryProviderRoute, FaceGeometryProviderRow,
    PROVIDER_ROUTED_PFS,
};

#[path = "portrait_body_stage.rs"]
mod portrait_body_stage;
pub use portrait_body_stage::{
    BodyGeometryInput, BodyGeometryJob, BodyShiftInput, SixPointWarp,
};
#[path = "portrait_body_producer.rs"]
mod portrait_body_producer;
pub use portrait_body_producer::{
    BodyGeometryProducerInputs, BodyLandmarkDomain, BodyLandmarkEffect, BodyLandmarkFrame,
    BodyLandmarkRef,
};

#[path = "portrait_plump_stage.rs"]
mod portrait_plump_stage;
pub use portrait_plump_stage::{
    PLUMP_PFS, PlumpCompositeReceipt, PlumpFinalComposite, PlumpGroupRun, PlumpProducerInputs,
    full_face_descriptor_from_ui, PlumpSameFrameEvidence, PLUMP_LANDMARK_COUNT,
};

#[path = "portrait_plump_onnx.rs"]
mod portrait_plump_onnx;
pub use portrait_plump_onnx::{
    build_plump_producer_inputs, discover_plump_onnx_models, plump_raster_with_cpu_fallback,
    PlumpBuildRequest, PlumpOnnxAsset, PlumpRasterBackend, PlumpRasterOutcome, PlumpRasterReceipt,
};

#[path = "portrait_historical_tone.rs"]
mod portrait_historical_tone;
pub use portrait_historical_tone::{
    HistoricalToneData, HistoricalToneFace, HistoricalToneFixture, HistoricalToneGpu,
    HistoricalToneGpuPlan, HistoricalToneInput, HistoricalToneInputs, HistoricalTonePath,
    HistoricalToneRowOutcome, apply_historical_tone_row, gpu_kernel, gpu_lut_reason,
    plan_historical_tone_backend, GPU_MATERIALIZED_FIXTURE_ENV,
    GPU_PROVEN_HISTORICAL_TONE_PFS,
};

#[path = "portrait_historical_skin.rs"]
mod portrait_historical_skin;
pub use portrait_historical_skin::{
    HistoricalSkinData, HistoricalSkinExternalInputs, HistoricalSkinExternalRow,
    HistoricalSkinGpuPlan, HistoricalSkinImmediateInputs, HistoricalSkinImmediateRow,
    HistoricalSkinInput, HistoricalSkinInputs, HistoricalSkinPass,
    HistoricalSkinPath, HistoricalSkinWrinkle, build_historical_skin_inputs,
    plan_historical_skin_backend, GPU_PROVEN_HISTORICAL_SKIN_PFS,
};

#[path = "portrait_repair_stage.rs"]
mod repair_stage;
pub use repair_stage::{
    RepairModelPlanes, RepairOutput, RepairOutputReceipt, RepairPlaneSha256,
    RepairProducerInputs,
};

#[cfg(test)]
#[path = "portrait_hlf_wrinkle_stage_tests.rs"]
mod hlf_wrinkle_stage_tests;

#[cfg(test)]
#[path = "portrait_hlf_24mp_bench_tests.rs"]
mod hlf_24mp_bench_tests;

#[cfg(test)]
#[path = "portrait_pf8050_stage_tests.rs"]
mod pf8050_stage_tests;

#[cfg(test)]
#[path = "portrait_plump_stage_tests.rs"]
mod plump_stage_tests;

#[cfg(test)]
#[path = "portrait_body_stage_tests.rs"]
mod body_geometry_stage_tests;


#[cfg(test)]
const MASK_EDGE: usize = 512;
const LUT_EDGE: usize = 512;
const LUT_BYTES: usize = LUT_EDGE * LUT_EDGE * 3;

/// Masks sampled once per source image, in normalized image coordinates.
pub struct PortraitMasks {
    pub skin: Option<Vec<f32>>,
    pub hair: Option<Vec<f32>>,
    pub face: Option<Vec<f32>>,
    pub body: Option<Vec<f32>>,
    pub width: usize,
    pub height: usize,
}

impl Default for PortraitMasks {
    fn default() -> Self {
        Self {
            skin: None,
            hair: None,
            face: None,
            body: None,
            width: 0,
            height: 0,
        }
    }
}

impl PortraitMasks {
    pub fn from_rgba(rgba: &[u8], width: u32, height: u32) -> Self {
        let parsed = crate::aimask::people_masks(rgba, width, height).ok();
        let face = parsed.as_ref().map(|p| p.face.clone());
        let body = parsed.as_ref().map(|p| p.body.clone());
        let hair = parsed.as_ref().map(|p| p.hair.clone());

        let mut skin = None;
        if face.is_some() || body.is_some() {
            let mut combined = vec![0.0f32; (width as usize) * (height as usize)];
            if let Some(plane) = &face {
                for (dst, src) in combined.iter_mut().zip(plane) {
                    *dst = dst.max(*src);
                }
            }
            if let Some(plane) = &body {
                for (dst, src) in combined.iter_mut().zip(plane) {
                    *dst = dst.max(*src);
                }
            }
            skin = Some(combined);
        }

        Self {
            skin,
            hair,
            face,
            body,
            width: width as usize,
            height: height as usize,
        }
    }

    fn plane(&self, kind: MaskKind) -> Option<&[f32]> {
        match kind {
            MaskKind::Skin => self.skin.as_deref(),
            MaskKind::Hair => self.hair.as_deref(),
            MaskKind::Face => self.face.as_deref(),
            MaskKind::Body => self.body.as_deref(),
        }
    }

    /// Samples one plane at the rendered frame's resolution and normalized region.
    pub fn mask_u8(
        &self,
        kind: MaskKind,
        out_w: u32,
        out_h: u32,
        region: NormRect,
    ) -> Option<Vec<u8>> {
        self.mask_f32(kind, out_w, out_h, region).map(|out| {
            out.into_iter()
                .map(|value| (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
                .collect()
        })
    }

    /// Samples a mask without quantizing it, so automatic HLF producers keep
    /// the same float mask domain as explicit same-job inputs.
    pub fn mask_f32(
        &self,
        kind: MaskKind,
        out_w: u32,
        out_h: u32,
        region: NormRect,
    ) -> Option<Vec<f32>> {
        let plane = self.plane(kind)?;
        if self.width == 0
            || self.height == 0
            || plane.len() != self.width * self.height
            || out_w == 0
            || out_h == 0
        {
            return None;
        }
        let mut out = Vec::with_capacity((out_w as usize) * (out_h as usize));
        for y in 0..out_h {
            let ny = region.y + (y as f32 + 0.5) / out_h as f32 * region.height;
            for x in 0..out_w {
                let nx = region.x + (x as f32 + 0.5) / out_w as f32 * region.width;
                out.push(sample_plane(plane, self.width, self.height, nx, ny).clamp(0.0, 1.0));
            }
        }
        Some(out)
    }

    /// Build the PF90004 fields that the local people parser can prove.
    /// Vendor parsing/face packings remain absent rather than being guessed.
    pub fn pf90004_parser_inputs(
        &self,
        rgba: &[u8],
        out_w: u32,
        out_h: u32,
        region: NormRect,
    ) -> Result<Pf90004AutoInputs, Refusal> {
        if region != NormRect::FULL {
            return Err(Refusal::ValueOutOfRange("PF90004_ROI"));
        }
        let pixels = usize::try_from(out_w)
            .ok()
            .and_then(|w| usize::try_from(out_h).ok().and_then(|h| w.checked_mul(h)))
            .filter(|&pixels| pixels != 0)
            .ok_or(Refusal::ShapeMismatch("PF90004_FRAME_SIZE"))?;
        if rgba.len() != pixels.saturating_mul(4) {
            return Err(Refusal::ShapeMismatch("PF90004_FRAME_RGBA"));
        }
        let skin_mask = self
            .mask_u8(MaskKind::Skin, out_w, out_h, region)
            .ok_or(Refusal::MissingInput("PF90004_PARSER_SKIN_MASK"))?;
        let value_mean = pf90004_value_mean(rgba, out_w, out_h, &skin_mask)?;
        Ok(Pf90004AutoInputs {
            source: Pf90004AutoSource::PortraitParser,
            skin_mask,
            parsing_rgba: None,
            face_rgba: None,
            value_mean,
            roi: NormRect::FULL,
        })
    }

    #[cfg(test)]
    pub fn solid(value: f32) -> Self {
        let n = MASK_EDGE * MASK_EDGE;
        Self {
            skin: Some(vec![value; n]),
            hair: Some(vec![value; n]),
            face: Some(vec![value; n]),
            body: Some(vec![value; n]),
            width: MASK_EDGE,
            height: MASK_EDGE,
        }
    }
}

/// Provenance is part of the value: parser coverage is not a vendor plane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pf90004AutoSource {
    PortraitParser,
}

/// The subset of PF90004 inputs derivable without inventing vendor packing.
#[derive(Clone, Debug, PartialEq)]
pub struct Pf90004AutoInputs {
    pub source: Pf90004AutoSource,
    pub skin_mask: Vec<u8>,
    pub parsing_rgba: Option<Vec<u8>>,
    pub face_rgba: Option<Vec<u8>>,
    pub value_mean: f32,
    pub roi: NormRect,
}

/// The native statistic at `0x702E3F0`: max-side 320 resize to even
/// dimensions, then the mean red byte over pixels whose resized mask is nonzero.
/// Vendor input surfaces are BGR; Aurora's `Image` is RGBA8, so red is byte 0.
fn pf90004_value_mean(
    rgba: &[u8],
    width: u32,
    height: u32,
    skin_mask: &[u8],
) -> Result<f32, Refusal> {
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("PF90004_FRAME_SIZE"))?;
    if skin_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("PF90004_SKIN_MASK"));
    }
    if rgba.len() != pixels.saturating_mul(4) {
        return Err(Refusal::ShapeMismatch("PF90004_FRAME_RGBA"));
    }

    let max_side = width.max(height) as f64;
    let scale = (320.0_f64 / max_side).min(1.0) as f32;
    let stats_w = (2.0 * (width as f32 * scale * 0.5).round()) as u32;
    let stats_h = (2.0 * (height as f32 * scale * 0.5).round()) as u32;
    if stats_w == 0 || stats_h == 0 {
        return Ok(0.0);
    }

    let sx_step = width as f32 / stats_w as f32;
    let sy_step = height as f32 / stats_h as f32;
    let mut sum = 0.0f32;
    let mut count = 0u32;
    for y in 0..stats_h {
        let sy = (y as f32 + 0.5) * sy_step - 0.5;
        let y0 = sy.max(0.0) as u32;
        let y1 = (y0 + 1).min(height - 1);
        let fy = (sy - y0 as f32).clamp(0.0, 1.0);
        for x in 0..stats_w {
            let sx = (x as f32 + 0.5) * sx_step - 0.5;
            let x0 = sx.max(0.0) as u32;
            let x1 = (x0 + 1).min(width - 1);
            let fx = (sx - x0 as f32).clamp(0.0, 1.0);
            let at = |xx: u32, yy: u32| {
                let i = yy as usize * width as usize + xx as usize;
                (rgba[i * 4] as f32, skin_mask[i] as f32)
            };
            let (r00, m00) = at(x0, y0);
            let (r10, m10) = at(x1, y0);
            let (r01, m01) = at(x0, y1);
            let (r11, m11) = at(x1, y1);
            let red = (r00 * (1.0 - fx) + r10 * fx) * (1.0 - fy)
                + (r01 * (1.0 - fx) + r11 * fx) * fy;
            let mask = (m00 * (1.0 - fx) + m10 * fx) * (1.0 - fy)
                + (m01 * (1.0 - fx) + m11 * fx) * fy;
            if (mask + 0.5).floor() != 0.0 {
                sum += (red + 0.5).floor();
                count += 1;
            }
        }
    }
    Ok(if count == 0 { 0.0 } else { sum / count as f32 })
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskKind {
    Skin,
    Hair,
    Face,
    Body,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortraitBackend {
    Cpu,
    Cuda,
    Skipped,
}

/// Machine-readable reason for a row that produced no pixels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PortraitSkipReason {
    InputRefused(Refusal),
    EdgeGateClosed,
    Cuda(CudaError),
}

#[derive(Clone, Debug, PartialEq)]
pub struct PortraitEffectRun {
    pub pf: u32,
    pub backend: PortraitBackend,
    pub dispatches: Vec<Dispatch>,
    pub fallback_reason: Option<String>,
    pub skipped: bool,
    pub executed: bool,
    pub refusal: Option<Refusal>,
    pub skip_reason: Option<PortraitSkipReason>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct PortraitRunReport {
    pub job_id: Option<u64>,
    pub effects: Vec<PortraitEffectRun>,
}

pub type Pf8050GeometryInputs<'a> = cuda_pf8050::Inputs<'a>;

/// Vendor GL receipts are separate from the six-stage geometry contract.
/// s934 proved PF8033's Vec3/ROI family cannot substitute for PF8050.
#[derive(Clone, Copy, Debug, Default)]
pub struct Pf8050VendorRuntime<'a> {
    pub attach: Option<Pf8050AttachReceipt>,
    pub vec3: Option<&'a [[f32; 3]]>,
    pub roi: Option<NormRect>,
    pub vbo: Option<&'a [u8]>,
    pub fbo: Option<Pf8050FboReceipt>,
    pub readback: Option<&'a [u8]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pf8050AttachReceipt {
    pub job_id: u64,
    pub face_id: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pf8050FboReceipt {
    pub framebuffer_id: u32,
    pub color_attachment: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Pf8050StageInputs<'a> {
    pub geometry: Pf8050GeometryInputs<'a>,
    pub runtime: Pf8050VendorRuntime<'a>,
}

/// Candidate six-stage geometry result. `candidate_only` must remain true
/// until same-job vendor VBO/FBO/readback bytes are captured and consumed.
#[derive(Clone, Debug, PartialEq)]
pub struct PortraitGeometryRun {
    pub backend: PortraitBackend,
    pub selection: Selection,
    pub dispatches: Vec<Dispatch>,
    pub fallback_reason: Option<String>,
    pub refusal: Option<Refusal>,
    pub stages: Vec<[[f32; 2]; MESH_POINTS]>,
    pub candidate_only: bool,
}

/// Asset bytes needed by rows that cannot use the shared parser masks.
#[derive(Clone, Copy, Debug, Default)]
pub struct PortraitAssetInputs<'a> {
    pub teeth_whiten_lut: Option<&'a [u8]>,
    pub teeth_bright_lut: Option<&'a [u8]>,
}

/// Explicit producer output for one renderer job. Missing rows stay absent.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ProducerInputs {
    pub parsing: Option<crate::face_parsing::FaceParsingProducerInputs>,
    pub slim: Option<SlimProducerInputs>,
    pub face_geometry: Option<FaceGeometryInputs>,
    pub body_geometry: Option<BodyGeometryJob>,
    pub repairs: Option<RepairProducerInputs>,
    pub historical_tone: Option<HistoricalToneInputs>,
    pub historical_skin: Option<HistoricalSkinInputs>,
    pub teeth: Option<TeethProducerInputs>,
    pub pf30025: Option<Pf30025ProducerInputs>,
    pub teeth_alignment: Option<TeethAlignmentProducerInputs>,
    pub hair_smooth: Option<HairSmoothProducerInputs>,
    pub hair_gloss: Option<HairGlossProducerInputs>,
    pub eye_bag: Option<aurora_portrait::eye_bag_alignment::AlignedEyeBagPlanes>,
    pub smile: Option<SmileProducerInputs>,
    pub stray_hair: Option<StrayHairProducerInputs>,
    pub pf90021: Option<Pf90021ProducerInputs>,
    pub pf90004: Option<Pf90004ProducerInputs>,
    pub hlf_body_adjust_lut: Option<HlfBodyAdjustLutProducerInputs>,
    pub wrinkle: Option<WrinkleProducerInputs>,
    pub makeup_bundle: Option<OwnedMakeupBundle>,
    pub plump: Option<PlumpProducerInputs>,
    pub pf101: Option<Pf101ProducerInputs>,
    pub pf90130: Option<Pf90130ProducerInputs>,
    pub moisture_face: Option<crate::portrait_moisture_inputs::OwnedMoistureFaceInputs>,
    pub moisture_body: Option<crate::portrait_moisture_inputs::OwnedMoistureBodyInputs>,
}

/// Owned same-job inputs for the recovered PF101 FilterType 41/42/43/44 stage.
/// The mask wrapper can only name face-mask input id 0x15; no eye-white,
/// eye-bag, material-alpha or RGBA-sum mask is accepted as a substitute.
#[derive(Clone, Debug, PartialEq)]
pub struct Pf101ProducerInputs {
    ticket: u64,
    material: Image,
    face_mask: Image,
    mask_uv: Vec<[f32; 2]>,
    material_uv: Vec<[f32; 2]>,
    params: aurora_portrait::kernels::pf101_silkworm::SilkwormParams,
}

impl Pf101ProducerInputs {
    pub fn from_same_job(
        ticket: u64,
        material: Image,
        face_mask: Image,
        mask_uv: Vec<[f32; 2]>,
        material_uv: Vec<[f32; 2]>,
        params: aurora_portrait::kernels::pf101_silkworm::SilkwormParams,
    ) -> Result<Self, Refusal> {
        if ticket == 0 {
            return Err(Refusal::MissingInput("PF101_SAME_JOB_TICKET"));
        }
        material.validate()?;
        face_mask.validate()?;
        if mask_uv.is_empty() || material_uv.is_empty() {
            return Err(Refusal::ShapeMismatch("PF101_UV"));
        }
        if mask_uv
            .iter()
            .chain(&material_uv)
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("PF101_UV"));
        }
        params
            .layer()
            .map_err(|_| Refusal::MissingInput("PF101_FILTERTYPE_SAME_JOB_RECORD"))?;
        Ok(Self {
            ticket,
            material,
            face_mask,
            mask_uv,
            material_uv,
            params,
        })
    }

    pub fn apply(
        &self,
        source: &Image,
        job_id: Option<u64>,
        degree: f32,
    ) -> Result<Image, Refusal> {
        if job_id != Some(self.ticket) {
            return Err(Refusal::ValueOutOfRange("PF101_JOB"));
        }
        let expected = self.params.degree01()?;
        if !degree.is_finite() || degree != expected {
            return Err(Refusal::ValueOutOfRange("PF101_CONTROL"));
        }
        let pixels = source.pixel_count();
        if self.mask_uv.len() != pixels || self.material_uv.len() != pixels {
            return Err(Refusal::ShapeMismatch("PF101_UV"));
        }
        aurora_portrait::kernels::pf101_silkworm::apply_image_layer(
            source,
            &self.material,
            aurora_portrait::kernels::pf101_silkworm::SilkwormMask::face_0x15(&self.face_mask),
            &self.mask_uv,
            &self.material_uv,
            self.params,
        )
    }
}

/// Owned same-job PF90130 producer and aligned geometry. The A/B bridge and
/// tensor identities remain in `Pf90130GlareProducer`; this type only carries
/// the caller's exact geometry into the normal portrait stage.
#[derive(Clone, Debug, PartialEq)]
pub struct Pf90130ProducerInputs {
    producer: Pf90130GlareProducer,
    source_width: u32,
    source_height: u32,
    source_roi: aurora_portrait::kernels::pf90130_chain::PixelRect,
    uv: Vec<[f32; 2]>,
    src_to_dst: [[f32; 3]; 2],
    source_rgb8: Vec<u8>,
    base_rgb8: Vec<u8>,
    add_rgb_f32: Vec<f32>,
    strength: f32,
}

impl Pf90130ProducerInputs {
    pub fn from_same_job(
        producer: Pf90130GlareProducer,
        geometry: Pf90130AlignedGeometry<'_>,
    ) -> Result<Self, Refusal> {
        let roi_end_x = u64::from(geometry.source_roi.x) + u64::from(geometry.source_roi.width);
        let roi_end_y = u64::from(geometry.source_roi.y) + u64::from(geometry.source_roi.height);
        if geometry.source_width == 0
            || geometry.source_height == 0
            || geometry.source_roi.width != PF90130_PLANE_WIDTH
            || geometry.source_roi.height != PF90130_PLANE_HEIGHT
            || roi_end_x > u64::from(geometry.source_width)
            || roi_end_y > u64::from(geometry.source_height)
        {
            return Err(Refusal::ValueOutOfRange("PF90130_SOURCE_ROI"));
        }
        if geometry.uv.len() != PF90130_PIXELS
            || geometry
                .uv
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
            || geometry
                .src_to_dst
                .iter()
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("PF90130_GEOMETRY"));
        }
        if geometry.source_rgb8.len() != PF90130_PIXELS * 3
            || geometry.base_rgb8.len() != PF90130_PIXELS * 3
            || geometry.add_rgb_f32.len() != PF90130_PIXELS * 3
            || geometry
                .add_rgb_f32
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(Refusal::ShapeMismatch("PF90130_GEOMETRY_PLANES"));
        }
        if !geometry.strength.is_finite() || !(0.0..=1.0).contains(&geometry.strength) {
            return Err(Refusal::ValueOutOfRange("PF90130_STRENGTH"));
        }
        Ok(Self {
            producer,
            source_width: geometry.source_width,
            source_height: geometry.source_height,
            source_roi: geometry.source_roi,
            uv: geometry.uv.to_vec(),
            src_to_dst: geometry.src_to_dst,
            source_rgb8: geometry.source_rgb8.to_vec(),
            base_rgb8: geometry.base_rgb8.to_vec(),
            add_rgb_f32: geometry.add_rgb_f32.to_vec(),
            strength: geometry.strength,
        })
    }

    pub fn apply(
        &self,
        source: &Image,
        destination: &Image,
        job_id: Option<u64>,
        degree: f32,
    ) -> Pf90130StageRun {
        if job_id != Some(self.producer.receipt().ticket()) {
            return Pf90130StageRun::refused(Refusal::ValueOutOfRange("PF90130_JOB"));
        }
        if !degree.is_finite() || degree != self.strength {
            return Pf90130StageRun::refused(Refusal::ValueOutOfRange("PF90130_CONTROL"));
        }
        let geometry = Pf90130AlignedGeometry {
            source_width: self.source_width,
            source_height: self.source_height,
            source_roi: self.source_roi,
            uv: &self.uv,
            src_to_dst: self.src_to_dst,
            source_rgb8: &self.source_rgb8,
            base_rgb8: &self.base_rgb8,
            add_rgb_f32: &self.add_rgb_f32,
            strength: self.strength,
        };
        apply_pf90130_producer_stage(
            Some(&self.producer),
            Some(source),
            Some(destination),
            Some(&geometry),
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeethMaskSource {
    FaceParsingV2Id2G,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeethMaskFormat {
    U8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeethLutKind {
    Whiten,
    Bright,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeethLutFormat {
    Rgb8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TeethLutEncoding {
    EncodedSrgb,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TeethLutIdentity {
    pub kind: TeethLutKind,
    pub config_field: &'static str,
    pub source_file: &'static str,
    pub png_bytes: usize,
    pub png_sha256: [u8; 32],
    pub rgb8_bytes: usize,
    pub rgb8_sha256: [u8; 32],
    pub width: u32,
    pub height: u32,
    pub format: TeethLutFormat,
    pub encoding: TeethLutEncoding,
}

pub const PF108_WHITEN_LUT_IDENTITY: TeethLutIdentity = TeethLutIdentity {
    kind: TeethLutKind::Whiten,
    config_field: "WhitenLutPath",
    source_file: "Whiten.png",
    png_bytes: 99_654,
    png_sha256: [
        0x09, 0xd4, 0x2d, 0xf0, 0xc1, 0x93, 0xac, 0xc7, 0x2f, 0x87, 0x00, 0xd6, 0x19, 0x7f,
        0xf5, 0xc2, 0xc2, 0xbd, 0xc3, 0x39, 0x26, 0x78, 0x71, 0x0a, 0xab, 0x0a, 0xec, 0x35,
        0x93, 0x51, 0x7c, 0xc4,
    ],
    rgb8_bytes: 786_432,
    rgb8_sha256: [
        0x1c, 0x78, 0x3a, 0xb2, 0xb3, 0x35, 0xb8, 0xd0, 0x36, 0xc7, 0x7c, 0xe9, 0x93, 0x47,
        0xc1, 0x8b, 0x73, 0xd0, 0xfc, 0x4b, 0x92, 0xed, 0xb3, 0xa3, 0x94, 0x71, 0x21, 0x0c,
        0x33, 0x9b, 0xa2, 0xed,
    ],
    width: 512,
    height: 512,
    format: TeethLutFormat::Rgb8,
    encoding: TeethLutEncoding::EncodedSrgb,
};

pub const PF108_BRIGHT_LUT_IDENTITY: TeethLutIdentity = TeethLutIdentity {
    kind: TeethLutKind::Bright,
    config_field: "BrightLutPath",
    source_file: "Bright.png",
    png_bytes: 196_345,
    png_sha256: [
        0xbb, 0xe8, 0x41, 0xa7, 0x51, 0xdd, 0x7a, 0xa7, 0x8d, 0x0d, 0x13, 0x86, 0xc3, 0xcc,
        0x68, 0x57, 0x22, 0xc6, 0x37, 0x1c, 0x3c, 0x91, 0x30, 0x41, 0xbc, 0x9b, 0xf3, 0x14,
        0x3c, 0x9d, 0x12, 0x6c,
    ],
    rgb8_bytes: 786_432,
    rgb8_sha256: [
        0x11, 0x2f, 0x2d, 0x16, 0xc3, 0x6c, 0xfd, 0xfd, 0xa0, 0x0e, 0x90, 0x2d, 0x74, 0x67,
        0x82, 0x0b, 0x0f, 0xd1, 0x75, 0xb1, 0x4b, 0x10, 0xf5, 0xc8, 0x0e, 0xf6, 0x11, 0xb2,
        0x1f, 0xb8, 0xa6, 0xf7,
    ],
    width: 512,
    height: 512,
    format: TeethLutFormat::Rgb8,
    encoding: TeethLutEncoding::EncodedSrgb,
};

#[derive(Clone, Debug, PartialEq)]
pub struct TeethProducerInputs {
    pub job_id: u64,
    pub mask_width: u32,
    pub mask_height: u32,
    pub mask_source: TeethMaskSource,
    pub mask_format: TeethMaskFormat,
    pub mask_sha256: [u8; 32],
    pub whiten_lut_identity: TeethLutIdentity,
    pub bright_lut_identity: TeethLutIdentity,
    pub mask_g: Vec<u8>,
    pub bright_ui: f32,
    pub deyellow_ui: f32,
    pub whiten_lut: Vec<u8>,
    pub bright_lut: Vec<u8>,
}

/// One renderer job's raw PF100007 outputs plus an optional, explicit host
/// split. The raw three-channel output is an image-like reconstruction field;
/// it is never interpreted as displacement without a [`SmileHostSplitSpec`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SmileProducerInputs {
    pub ticket: u64,
    pub material: Option<String>,
    pub material_uv: Option<Vec<[f32; 2]>>,
    pub material_sha256: Option<[u8; 32]>,
    pub roi: Option<(u32, u32, u32, u32)>,
    pub sigmoid_edge_response_3204548346: Option<SmileSigmoidEdgeResponse>,
    pub rgb_reconstruction_780929437: Option<SmileRgbReconstructionField>,
    pub host_split_spec: Option<SmileHostSplitSpec>,
    pub channel_split: Option<SmileChannelSplit>,
    pub lip_mask: Option<Vec<u8>>,
    pub mouth_mask: Option<Vec<u8>>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SmileSigmoidEdgeResponse {
    pub width: u32,
    pub height: u32,
    pub values: Vec<f32>,
}

impl SmileSigmoidEdgeResponse {
    pub const OUTPUT_ID: u32 = 3_204_548_346;
}

/// Raw `780929437` tensor in `[3, height, width]` channel-major order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SmileRgbReconstructionField {
    pub width: u32,
    pub height: u32,
    pub values: Vec<f32>,
}

impl SmileRgbReconstructionField {
    pub const OUTPUT_ID: u32 = 780_929_437;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmileFieldChannel {
    R = 0,
    G = 1,
    B = 2,
}

impl SmileFieldChannel {
    fn index(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmileAxisSign {
    Positive,
    Negative,
}

impl SmileAxisSign {
    fn multiplier(self) -> f32 {
        match self {
            Self::Positive => 1.0,
            Self::Negative => -1.0,
        }
    }
}

/// One axis of a host-proven split: `pixels = sign * scale * channel + bias`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmileAxisDecode {
    pub channel: SmileFieldChannel,
    pub sign: SmileAxisSign,
    pub scale: f32,
    pub bias: f32,
}

/// Explicit host receipt for decoding the 1024x768 three-channel field.
///
/// This value is intentionally not derived from the raw field. Static
/// evidence has not established that any R/G/B channel is a displacement
/// axis, so the product path must leave this absent and fail closed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmileHostSplitSpec {
    pub job_id: u64,
    pub dx: SmileAxisDecode,
    pub dy: SmileAxisDecode,
    pub model_roi: Rect,
    pub frame_roi: Rect,
}

/// Fully decoded, frame-sized displacement planes. Only this type populates
/// the existing [`SmileInputs::model_dx`] and [`SmileInputs::model_dy`].
#[derive(Clone, Debug, PartialEq)]
pub struct SmileChannelSplit {
    pub spec: SmileHostSplitSpec,
    pub model_dx: Vec<f32>,
    pub model_dy: Vec<f32>,
}

impl SmileSigmoidEdgeResponse {
    fn validate(&self) -> Result<usize, Refusal> {
        let pixels = smile_model_pixels(self.width, self.height, "pf100007 3204548346 shape")?;
        if self.values.len() != pixels {
            return Err(Refusal::ShapeMismatch("pf100007 3204548346 values"));
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("pf100007 3204548346 values"));
        }
        Ok(pixels)
    }
}

impl SmileRgbReconstructionField {
    fn validate(&self) -> Result<usize, Refusal> {
        let pixels = smile_model_pixels(self.width, self.height, "pf100007 780929437 shape")?;
        let expected = pixels
            .checked_mul(3)
            .ok_or(Refusal::ShapeMismatch("pf100007 780929437 values"))?;
        if self.values.len() != expected {
            return Err(Refusal::ShapeMismatch("pf100007 780929437 values"));
        }
        if self.values.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("pf100007 780929437 values"));
        }
        Ok(pixels)
    }
}

impl SmileHostSplitSpec {
    fn validate(self, frame_width: u32, frame_height: u32) -> Result<(), Refusal> {
        if self.job_id == 0 {
            return Err(Refusal::ValueOutOfRange("pf100007 host split ticket"));
        }
        if self.dx.channel == self.dy.channel {
            return Err(Refusal::ValueOutOfRange("pf100007 host split channel"));
        }
        for axis in [self.dx, self.dy] {
            if !axis.scale.is_finite() || axis.scale <= 0.0 || !axis.bias.is_finite() {
                return Err(Refusal::ValueOutOfRange("pf100007 host split axis"));
            }
        }
        if self.model_roi != smile_model_rect() {
            return Err(Refusal::ValueOutOfRange("pf100007 model roi"));
        }
        if self.frame_roi.width == 0
            || self.frame_roi.height == 0
            || self.frame_roi.x > frame_width
            || self.frame_roi.y > frame_height
            || self.frame_roi.width > frame_width - self.frame_roi.x
            || self.frame_roi.height > frame_height - self.frame_roi.y
        {
            return Err(Refusal::ValueOutOfRange("pf100007 frame roi"));
        }
        Ok(())
    }
}

impl SmileProducerInputs {
    /// Decode the raw model field only when a host-proven split spec exists.
    ///
    /// Without that receipt this returns a typed refusal. In particular, raw
    /// R/G/B channels are not promoted to `model_dx`/`model_dy` here.
    pub fn decode_channel_split(
        &self,
        frame_width: u32,
        frame_height: u32,
    ) -> Result<SmileChannelSplit, Refusal> {
        let field = self
            .rgb_reconstruction_780929437
            .as_ref()
            .ok_or(Refusal::MissingInput("pf100007 780929437"))?;
        field.validate()?;
        if let Some(response) = self.sigmoid_edge_response_3204548346.as_ref() {
            response.validate()?;
            if response.width != field.width || response.height != field.height {
                return Err(Refusal::ShapeMismatch("pf100007 response/field grid"));
            }
        }
        let spec = self
            .host_split_spec
            .ok_or(Refusal::MissingInput("pf100007 host split receipt"))?;
        if spec.job_id != self.ticket {
            return Err(Refusal::ValueOutOfRange("pf100007 host split ticket"));
        }
        spec.validate(frame_width, frame_height)?;

        let frame_pixels = (frame_width as usize)
            .checked_mul(frame_height as usize)
            .ok_or(Refusal::ShapeMismatch("pf100007 frame size"))?;
        let mut model_dx = vec![0.0; frame_pixels];
        let mut model_dy = vec![0.0; frame_pixels];
        for y in spec.frame_roi.y..spec.frame_roi.y + spec.frame_roi.height {
            for x in spec.frame_roi.x..spec.frame_roi.x + spec.frame_roi.width {
                let mx = spec.model_roi.x as f32
                    + (x - spec.frame_roi.x) as f32 * spec.model_roi.width as f32
                        / spec.frame_roi.width as f32;
                let my = spec.model_roi.y as f32
                    + (y - spec.frame_roi.y) as f32 * spec.model_roi.height as f32
                        / spec.frame_roi.height as f32;
                let index = y as usize * frame_width as usize + x as usize;
                model_dx[index] = decode_smile_axis(field, spec.dx, mx, my);
                model_dy[index] = decode_smile_axis(field, spec.dy, mx, my);
            }
        }
        Ok(SmileChannelSplit {
            spec,
            model_dx,
            model_dy,
        })
    }
}

fn smile_model_pixels(width: u32, height: u32, name: &'static str) -> Result<usize, Refusal> {
    if width != 1024 || height != 768 {
        return Err(Refusal::ShapeMismatch(name));
    }
    Ok((width as usize) * (height as usize))
}

fn smile_model_rect() -> Rect {
    Rect {
        x: 0,
        y: 0,
        width: 1024,
        height: 768,
    }
}

fn decode_smile_axis(
    field: &SmileRgbReconstructionField,
    axis: SmileAxisDecode,
    x: f32,
    y: f32,
) -> f32 {
    let (r, g, b) = sample_smile_field(field, x, y);
    let value = match axis.channel {
        SmileFieldChannel::R => r,
        SmileFieldChannel::G => g,
        SmileFieldChannel::B => b,
    };
    axis.sign.multiplier() * axis.scale * value + axis.bias
}

fn sample_smile_field(field: &SmileRgbReconstructionField, x: f32, y: f32) -> (f32, f32, f32) {
    let width = field.width as usize;
    let height = field.height as usize;
    let cx = x.clamp(0.0, (width - 1) as f32);
    let cy = y.clamp(0.0, (height - 1) as f32);
    let x0 = cx.floor() as usize;
    let y0 = cy.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = cx - x0 as f32;
    let fy = cy - y0 as f32;
    let plane = width * height;
    let sample = |channel: usize, sx: usize, sy: usize| {
        field.values[channel * plane + sy * width + sx]
    };
    let mut out = [0.0; 3];
    for (channel, value) in out.iter_mut().enumerate() {
        let p00 = sample(channel, x0, y0);
        let p10 = sample(channel, x1, y0);
        let p01 = sample(channel, x0, y1);
        let p11 = sample(channel, x1, y1);
        let top = p00 + (p10 - p00) * fx;
        let bottom = p01 + (p11 - p01) * fx;
        *value = top + (bottom - top) * fy;
    }
    (out[0], out[1], out[2])
}

/// PF90088 same-job native values and planes. Every field stays optional so an
/// incomplete producer row retains the consumer's typed refusal order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StrayHairProducerInputs {
    pub hairfly_v1: Option<Image>,
    pub raw_mask: Option<Vec<u8>>,
    pub person_mask: Option<Vec<u8>>,
    pub background_mask: Option<Vec<u8>>,
    pub native_degree: Option<f32>,
    pub native_person_area: Option<f32>,
    pub native_background_area: Option<f32>,
}

/// PF30025 same-job model outputs and support inputs.
#[derive(Clone, Copy, Debug)]
pub struct Pf30025Inputs<'a> {
    pub receipt: Pf30025SameJobReceipt,
    pub repair_780929437_bgra: &'a [u8],
    pub gate_3204548346: &'a [f32],
    pub teeth_mask: &'a [f32],
    pub roi: ToothRect,
}

pub const PF30025_MODEL_IDS: [u32; 2] = [806_921_480, 1_940_369_920];
pub const PF30025_REPAIR_OUTPUT_780929437: u32 = 780_929_437;
pub const PF30025_GATE_OUTPUT_3204548346: u32 = 3_204_548_346;
pub const PF30025_REPAIR_SLOT_0X228: usize = 0x228;
pub const PF30025_GATE_SLOT_0X280: usize = 0x280;

/// Exact identities copied from one PF30025 readback. The source digest is
/// checked by the renderer receipt before the repair or gate is consumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pf30025SameJobReceipt {
    pub job_id: u64,
    pub model_id: u32,
    pub width: u32,
    pub height: u32,
    pub source_stride: u32,
    pub source_sha256: [u8; 32],
    pub repair_sha256: [u8; 32],
    pub gate_sha256: [u8; 32],
}

/// The already-developed RGBA surface captured by the renderer for this job.
/// It is an input to the producer, not a default or a frame reconstructed from
/// the model output.
#[derive(Clone, Copy, Debug)]
pub struct Pf30025SourceSurface<'a> {
    pub job_id: u64,
    pub width: u32,
    pub height: u32,
    pub row_stride: u32,
    pub rgba: &'a [u8],
}

impl Pf30025SourceSurface<'_> {
    fn validate(&self) -> Result<(usize, [u8; 32]), Refusal> {
        if self.job_id == 0 {
            return Err(Refusal::ValueOutOfRange("PF30025 source job"));
        }
        if self.width == 0 || self.height == 0 {
            return Err(Refusal::ShapeMismatch("PF30025 source surface"));
        }
        let pixels = (self.width as usize)
            .checked_mul(self.height as usize)
            .filter(|&pixels| pixels != 0)
            .ok_or(Refusal::ShapeMismatch("PF30025 source surface"))?;
        let packed_stride = self
            .width
            .checked_mul(4)
            .ok_or(Refusal::ShapeMismatch("PF30025 source surface"))?;
        if self.row_stride != packed_stride
            || self.rgba.len()
                != (self.row_stride as usize)
                    .checked_mul(self.height as usize)
                    .ok_or(Refusal::ShapeMismatch("PF30025 source surface"))?
        {
            return Err(Refusal::ShapeMismatch("PF30025 source surface"));
        }
        Ok((pixels, sha256_32(self.rgba)))
    }
}

/// A same-job PF30025 model result. The worker consumes `780929437` as three
/// B, G, R planes and `3204548346` as one f32 gate per pixel. The source
/// surface and the two model outputs must carry the same job and dimensions.
pub struct Pf30025LiveModelOutputs<'a> {
    pub job_id: u64,
    pub model_id: u32,
    pub source_surface: Pf30025SourceSurface<'a>,
    pub repair_780929437_bgr: [&'a [f32]; 3],
    pub gate_3204548346: &'a [f32],
}

impl Pf30025LiveModelOutputs<'_> {
    pub fn validate(&self) -> Result<(usize, [u8; 32]), Refusal> {
        if self.job_id == 0 {
            return Err(Refusal::ValueOutOfRange("PF30025 model job"));
        }
        if !PF30025_MODEL_IDS.contains(&self.model_id) {
            return Err(Refusal::UnknownEffect(self.model_id));
        }
        if self.source_surface.job_id != self.job_id {
            return Err(Refusal::MissingInput("PF30025_SAME_JOB_RECEIPT"));
        }
        let (pixels, source_sha256) = self.source_surface.validate()?;
        if self
            .repair_780929437_bgr
            .iter()
            .any(|plane| plane.len() != pixels)
            || self.gate_3204548346.len() != pixels
        {
            return Err(Refusal::ShapeMismatch("PF30025 model planes"));
        }
        if self
            .repair_780929437_bgr
            .iter()
            .flat_map(|plane| plane.iter())
            .chain(self.gate_3204548346)
            .any(|value| !value.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("PF30025 model planes"));
        }
        Ok((pixels, source_sha256))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pf30025ProducerInputs {
    pub receipt: Pf30025SameJobReceipt,
    /// Model output `780929437`, per-face slot `+0x228`, packed BGRA8.
    pub repair_780929437_bgra: Vec<u8>,
    /// Model output `3204548346`, per-face slot `+0x280`, one f32 per pixel.
    pub gate_3204548346: Vec<f32>,
    /// Explicit same-job teeth support normalized to `[0,1]`.
    pub teeth_mask: Vec<f32>,
    pub roi: ToothRect,
}

impl Pf30025ProducerInputs {
    pub fn from_live_model(
        output: Pf30025LiveModelOutputs<'_>,
        teeth_mask: Vec<f32>,
        roi: ToothRect,
    ) -> Result<Self, Refusal> {
        let (pixels, source_sha256) = output.validate()?;
        let width = output.source_surface.width;
        let height = output.source_surface.height;
        if teeth_mask.len() != pixels
            || teeth_mask
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err(Refusal::ShapeMismatch("PF30025 teeth mask"));
        }
        if !roi.inside(width, height) {
            return Err(Refusal::ValueOutOfRange("PF30025 roi"));
        }
        let mut repair_780929437_bgra = vec![0u8; pixels * 4];
        for pixel in 0..pixels {
            let offset = pixel * 4;
            for channel in 0..3 {
                repair_780929437_bgra[offset + channel] =
                    pf30025_signed_byte(output.repair_780929437_bgr[channel][pixel]);
            }
            repair_780929437_bgra[offset + 3] = 255;
        }
        let gate_3204548346 = output.gate_3204548346.to_vec();
        Ok(Self {
            receipt: Pf30025SameJobReceipt {
                job_id: output.job_id,
                model_id: output.model_id,
                width,
                height,
                source_stride: output.source_surface.row_stride,
                source_sha256,
                repair_sha256: sha256_32(&repair_780929437_bgra),
                gate_sha256: sha256_32(f32_bytes(&gate_3204548346)),
            },
            repair_780929437_bgra,
            gate_3204548346,
            teeth_mask,
            roi,
        })
    }
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn pf30025_signed_byte(value: f32) -> u8 {
    (value * 127.5 + 127.5).round().clamp(0.0, 255.0) as u8
}

/// PF30026 producer payload for one renderer job. The five payload fields are
/// paired with an explicit gate-role receipt; the model's output order is not
/// itself evidence for gate A/B.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TeethAlignmentProducerInputs {
    pub effect_rgb: Option<Image>,
    pub teeth_mask: Option<Vec<u8>>,
    pub edge_plane_a: Option<Vec<f32>>,
    pub edge_plane_b: Option<Vec<f32>>,
    pub gate_binding: Option<Pf30026GateBinding>,
    pub roi: Option<NormRect>,
}

/// Ordered outputs of `TeethEdgeSegment`.
///
/// The worker constructs output 0 before output 1, both as contiguous
/// `CV_32FC1` planes, with output 1 at `base + rows * cols * 4`. These names
/// describe that construction order only; they do not name anatomical sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pf30026EdgeOutput {
    Output0,
    Output1,
}

/// Same-job receipt that declares which ordered model output fed gate A and
/// which fed gate B.
///
/// The provider selector and gate call join were not captured, so callers must
/// supply this mapping from one renderer job. The stage never infers it from
/// `record+0xA8/+0x130` or from the model output number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pf30026GateBinding {
    pub job_id: u64,
    pub plane_a: Pf30026EdgeOutput,
    pub plane_b: Pf30026EdgeOutput,
}

impl Pf30026GateBinding {
    pub fn from_same_job(
        job_id: u64,
        plane_a: Pf30026EdgeOutput,
        plane_b: Pf30026EdgeOutput,
    ) -> Result<Self, Refusal> {
        let binding = Self {
            job_id,
            plane_a,
            plane_b,
        };
        binding.validate()?;
        Ok(binding)
    }

    fn validate(self) -> Result<(), Refusal> {
        if self.job_id == 0 {
            return Err(Refusal::MissingInput(
                "PF30026 edge gate same-job ticket",
            ));
        }
        if self.plane_a == self.plane_b {
            return Err(Refusal::ValueOutOfRange(
                "PF30026 edge gate binding",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pf90021ProducerInputs {
    pub direction: Pf90021Direction,
    pub positive_lut: VolumeLut64,
    pub negative_lut: VolumeLut64,
    pub mask_r: Vec<u8>,
    pub runtime: Pf90021RuntimeScalars,
    pub same_job: Pf90021SameJobBinding,
    pub direction_receipt: Pf90021DirectionReceipt,
    pub runtime_receipt: Pf90021RuntimeReceipt,
}

/// Opaque identity for the exact pre-effect RGBA8 frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortraitFrameReceipt {
    fingerprint: u64,
    width: u32,
    height: u32,
}

impl PortraitFrameReceipt {
    pub fn from_renderer_frame(
        frame: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Self, Refusal> {
        let expected = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .filter(|&bytes| bytes != 0)
            .ok_or(Refusal::ShapeMismatch("PortraitFrameReceipt dimensions"))?;
        if frame.len() != expected {
            return Err(Refusal::ShapeMismatch("PortraitFrameReceipt frame"));
        }
        let mut hash = fnv1a64(FNV1A64_OFFSET_BASIS, b"aurora-portrait-frame-v1");
        hash = fnv_u64(hash, u64::from(width));
        hash = fnv_u64(hash, u64::from(height));
        hash = fnv1a64(hash, frame);
        Ok(Self {
            fingerprint: hash,
            width,
            height,
        })
    }

    pub fn fingerprint(self) -> u64 {
        self.fingerprint
    }

    pub fn width(self) -> u32 {
        self.width
    }

    pub fn height(self) -> u32 {
        self.height
    }
}

/// Static PF90021 writer candidates all resolve to the FaceParsing channel-0
/// raster path; the map-offset conflict remains a live-receipt gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pf90021MaskSource {
    FaceParsingChannel0Rasterized,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pf90021SameJobBinding {
    pub job_id: u64,
    pub face_id: i32,
    pub ticket: u64,
    pub frame: PortraitFrameReceipt,
    pub mask_texture_id: u64,
    pub mask_channel: u8,
    pub mask_source: Pf90021MaskSource,
    pub roi_bits: [u32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pf90021DirectionReceipt {
    pub effect_range: [f32; 3],
    pub param_range: [f32; 3],
    pub input01: f32,
    pub output01: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pf90021RuntimeReceipt {
    pub degree01: f32,
    pub opacity01: f32,
    pub shelter01: f32,
    pub region01: f32,
}

fn pf90021_direction_from_effect_range(
    effect_range: [f32; 3],
) -> Result<Pf90021Direction, Refusal> {
    match effect_range {
        [0.0, 0.0, 1.0] => Ok(Pf90021Direction::Positive),
        [1.0, 0.0, 0.0] => Ok(Pf90021Direction::Negative),
        _ => Err(Refusal::ValueOutOfRange("PF90021_EFFECT_RANGE")),
    }
}

fn pf90021_degree_remap(direction: Pf90021Direction, input01: f32) -> f32 {
    match direction {
        Pf90021Direction::Neutral => 0.0,
        Pf90021Direction::Positive => {
            if input01 <= 0.5 {
                0.0
            } else if input01 >= 1.0 {
                1.0
            } else {
                (input01 - 0.5) / 0.5
            }
        }
        Pf90021Direction::Negative => {
            if input01 <= 0.0 {
                1.0
            } else if input01 >= 1.0 {
                0.0
            } else if input01 <= 0.5 {
                1.0 - 2.0 * input01
            } else {
                0.0
            }
        }
    }
}

/// The five PF90004 producer outputs plus the exact rendered-frame size they
/// belong to. Planes are RGBA8/R8 coverage values in the consumer's byte domain.
#[derive(Clone, Debug, PartialEq)]
pub struct Pf90004ProducerInputs {
    width: u32,
    height: u32,
    skin_mask: Vec<u8>,
    parsing_rgba: Vec<u8>,
    face_rgba: Vec<u8>,
    value_mean: f32,
    roi: NormRect,
}

impl Pf90004ProducerInputs {
    pub fn from_same_job(
        width: u32,
        height: u32,
        skin_mask: Vec<u8>,
        parsing_rgba: Vec<u8>,
        face_rgba: Vec<u8>,
        value_mean: f32,
        roi: NormRect,
    ) -> Result<Self, Refusal> {
        let bundle = Self {
            width,
            height,
            skin_mask,
            parsing_rgba,
            face_rgba,
            value_mean,
            roi,
        };
        bundle.validate()?;
        Ok(bundle)
    }

    fn validate(&self) -> Result<(), Refusal> {
        let pixels = usize::try_from(self.width)
            .ok()
            .and_then(|width| {
                usize::try_from(self.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .filter(|&pixels| pixels != 0)
            .ok_or(Refusal::ShapeMismatch("PF90004_FRAME_SIZE"))?;
        if self.skin_mask.len() != pixels {
            return Err(Refusal::ShapeMismatch("PF90004_SKIN_MASK"));
        }
        let rgba = pixels
            .checked_mul(4)
            .ok_or(Refusal::ShapeMismatch("PF90004_FRAME_SIZE"))?;
        if self.parsing_rgba.len() != rgba {
            return Err(Refusal::ShapeMismatch("PF90004_PARSING_RGBA"));
        }
        if self.face_rgba.len() != rgba {
            return Err(Refusal::ShapeMismatch("PF90004_FACE_RGBA"));
        }
        if !self.value_mean.is_finite() || !(0.0..=255.0).contains(&self.value_mean) {
            return Err(Refusal::ValueOutOfRange("PF90004_VALUE_MEAN"));
        }
        if self.roi != NormRect::FULL {
            return Err(Refusal::ValueOutOfRange("PF90004_ROI"));
        }
        Ok(())
    }
}

/// Vendor PF30022 L8 curve. The source, median, gauss, face and skin planes
/// remain separate same-job inputs and are never synthesized from this LUT.
#[derive(Clone, Debug, PartialEq)]
pub struct HlfBodyAdjustLutProducerInputs {
    pub width: u32,
    pub height: u32,
    pub luma: Vec<u8>,
}

/// Candidate filter settings for the automatic HLF producer.
///
/// s923 recovered the kernel shapes but not the vendor interval values. These
/// defaults are explicitly candidate values, not a vendor_equivalent claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HlfAutoProducerOptions {
    pub median_interval: u32,
    pub gauss_interval: u32,
}

impl Default for HlfAutoProducerOptions {
    fn default() -> Self {
        Self {
            median_interval: 1,
            gauss_interval: 1,
        }
    }
}

impl HlfAutoProducerOptions {
    fn validate(self) -> Result<Self, Refusal> {
        if self.median_interval == 0 || self.gauss_interval == 0 {
            return Err(Refusal::ValueOutOfRange("HLF auto interval"));
        }
        Ok(self)
    }
}

const HLF_MEDIAN_RADIUS: i32 = 1;
const HLF_GAUSS_SUPPORT: i32 = 4;
const HLF_GAUSS_WEIGHTS: [f32; 9] = [
    0.0162, 0.054, 0.1216, 0.1945, 0.227, 0.1945, 0.1216, 0.054, 0.0162,
];

#[derive(Clone, Debug, PartialEq)]
struct HlfAutoInputs {
    width: u32,
    height: u32,
    source_rgb_full_f32: Vec<f32>,
    median_rgb_full_f32: Vec<f32>,
    gauss_rgb_full_f32: Vec<f32>,
    face_mask_full_f32: Vec<f32>,
    skin_mask_full_f32: Vec<f32>,
    face_mask_rgba_full_f32: Vec<f32>,
    roi: Roi,
    preprocess_dispatches: Vec<Dispatch>,
    preprocess_fallback: Option<String>,
}

impl HlfAutoInputs {
    fn face(&self) -> HlfFaceInputs<'_> {
        HlfFaceInputs {
            width: self.width,
            height: self.height,
            source_rgb_full_f32: &self.source_rgb_full_f32,
            median_rgb_full_f32: &self.median_rgb_full_f32,
            gauss_rgb_full_f32: &self.gauss_rgb_full_f32,
            face_mask_full_f32: &self.face_mask_full_f32,
            skin_mask_full_f32: &self.skin_mask_full_f32,
            roi: self.roi,
        }
    }

    fn body(&self) -> HlfBodyInputs<'_> {
        HlfBodyInputs {
            width: self.width,
            height: self.height,
            median_rgb_full_f32: &self.median_rgb_full_f32,
            gauss_rgb_full_f32: &self.gauss_rgb_full_f32,
            face_mask_rgba_full_f32: &self.face_mask_rgba_full_f32,
            skin_mask_full_f32: &self.skin_mask_full_f32,
            roi: self.roi,
            uniforms: hlf_auto_body_uniforms(),
            adjust_lut: None,
        }
    }
}

fn hlf_auto_body_uniforms() -> BodyHlfUniforms {
    BodyHlfUniforms {
        spotless_degree: 0.0,
        retouch_degree: 1.0,
        spotless_acne_threshold: 0.0,
        spotless_high_freq_factor: 1.0,
        spotless_low_freq_alpha: 0.0,
        retouch_high_freq_factor: 0.0,
        diff_min_range: -0.2,
        diff_max_range: 0.2,
    }
}

fn hlf_auto_roi(region: NormRect, width: u32, height: u32) -> Result<Roi, Refusal> {
    check_norm_rect(region, "HLF auto region")?;
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("HLF auto frame"));
    }
    let x0 = (region.x * width as f32).floor() as u32;
    let y0 = (region.y * height as f32).floor() as u32;
    let x1 = ((region.x + region.width) * width as f32).ceil() as u32;
    let y1 = ((region.y + region.height) * height as f32).ceil() as u32;
    let x0 = x0.min(width - 1);
    let y0 = y0.min(height - 1);
    let x1 = x1.clamp(x0 + 1, width);
    let y1 = y1.clamp(y0 + 1, height);
    Ok(Roi {
        x: x0,
        y: y0,
        width: x1 - x0,
        height: y1 - y0,
    })
}

fn hlf_median_rgb(source: &[f32], width: usize, height: usize, interval: u32) -> Vec<f32> {
    let step = i64::from(interval);
    let mut out = vec![0.0; source.len()];
    for y in 0..height {
        for x in 0..width {
            let dst = (y * width + x) * 3;
            for channel in 0..3 {
                let mut samples = [0.0f32; 9];
                let mut index = 0;
                for dy in -HLF_MEDIAN_RADIUS..=HLF_MEDIAN_RADIUS {
                    let sy = (y as i64 + i64::from(dy) * step)
                        .clamp(0, height as i64 - 1) as usize;
                    for dx in -HLF_MEDIAN_RADIUS..=HLF_MEDIAN_RADIUS {
                        let sx = (x as i64 + i64::from(dx) * step)
                            .clamp(0, width as i64 - 1) as usize;
                        samples[index] = source[(sy * width + sx) * 3 + channel];
                        index += 1;
                    }
                }
                samples.sort_by(f32::total_cmp);
                out[dst + channel] = samples[4];
            }
        }
    }
    out
}

fn hlf_gauss_rgb(source: &[f32], width: usize, height: usize, interval: u32) -> Vec<f32> {
    let step = i64::from(interval);
    let mut horizontal = vec![0.0; source.len()];
    for y in 0..height {
        for x in 0..width {
            let dst = (y * width + x) * 3;
            for channel in 0..3 {
                let mut value = 0.0;
                for dx in -HLF_GAUSS_SUPPORT..=HLF_GAUSS_SUPPORT {
                    let sx = (x as i64 + i64::from(dx) * step)
                        .clamp(0, width as i64 - 1) as usize;
                    let wx = (dx + HLF_GAUSS_SUPPORT) as usize;
                    value += source[(y * width + sx) * 3 + channel] * HLF_GAUSS_WEIGHTS[wx];
                }
                horizontal[dst + channel] = value;
            }
        }
    }
    let mut out = vec![0.0; source.len()];
    for y in 0..height {
        for x in 0..width {
            let dst = (y * width + x) * 3;
            for channel in 0..3 {
                let mut value = 0.0;
                for dy in -HLF_GAUSS_SUPPORT..=HLF_GAUSS_SUPPORT {
                    let sy = (y as i64 + i64::from(dy) * step)
                        .clamp(0, height as i64 - 1) as usize;
                    let wy = (dy + HLF_GAUSS_SUPPORT) as usize;
                    value += horizontal[(sy * width + x) * 3 + channel] * HLF_GAUSS_WEIGHTS[wy];
                }
                out[dst + channel] = value;
            }
        }
    }
    out
}

fn hlf_cpu_preprocess(
    source: &[f32],
    width: u32,
    height: u32,
    options: HlfAutoProducerOptions,
) -> (Vec<f32>, Vec<f32>) {
    (
        hlf_median_rgb(source, width as usize, height as usize, options.median_interval),
        hlf_gauss_rgb(source, width as usize, height as usize, options.gauss_interval),
    )
}

fn hlf_preprocess(
    source: &[f32],
    width: u32,
    height: u32,
    options: HlfAutoProducerOptions,
) -> (Vec<f32>, Vec<f32>, Vec<Dispatch>, Option<String>) {
    #[cfg(windows)]
    {
        if let Some(error) = forced_cuda_failure() {
            let (median, gauss) = hlf_cpu_preprocess(source, width, height, options);
            return (median, gauss, Vec::new(), Some(format!("{error:?}")));
        }
        match aurora_portrait::cuda_hlf::Hlf::open().and_then(|gpu| {
            gpu.preprocess_source_rgb(
                source,
                width,
                height,
                options.median_interval,
                options.gauss_interval,
            )
        }) {
            Ok((median, gauss, dispatches)) => {
                return (median, gauss, dispatches, None);
            }
            Err(error) => {
                let (median, gauss) = hlf_cpu_preprocess(source, width, height, options);
                return (median, gauss, Vec::new(), Some(format!("{error:?}")));
            }
        }
    }
    #[cfg(not(windows))]
    {
        let (median, gauss) = hlf_cpu_preprocess(source, width, height, options);
        (median, gauss, Vec::new(), None)
    }
}

fn produce_hlf_auto(
    current: &Image,
    region: NormRect,
    masks: &PortraitMasks,
    options: HlfAutoProducerOptions,
) -> Result<HlfAutoInputs, Refusal> {
    options.validate()?;
    current.validate()?;
    let pixels = current.pixel_count();
    let face_mask = masks
        .mask_f32(MaskKind::Face, current.width, current.height, region)
        .ok_or(Refusal::MissingInput("HLF auto face mask"))?;
    let skin_mask = masks
        .mask_f32(MaskKind::Skin, current.width, current.height, region)
        .ok_or(Refusal::MissingInput("HLF auto skin mask"))?;
    if face_mask.len() != pixels || skin_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("HLF auto masks"));
    }
    let mut source = Vec::with_capacity(pixels * 3);
    for pixel in current.pixels.chunks_exact(4) {
        source.push(pixel[0] as f32 / 255.0);
        source.push(pixel[1] as f32 / 255.0);
        source.push(pixel[2] as f32 / 255.0);
    }
    let (median, gauss, preprocess_dispatches, preprocess_fallback) =
        hlf_preprocess(&source, current.width, current.height, options);
    let mut face_rgba = Vec::with_capacity(pixels * 4);
    for value in &face_mask {
        face_rgba.extend_from_slice(&[*value, 0.0, 0.0, 1.0]);
    }
    Ok(HlfAutoInputs {
        width: current.width,
        height: current.height,
        source_rgb_full_f32: source,
        median_rgb_full_f32: median,
        gauss_rgb_full_f32: gauss,
        face_mask_full_f32: face_mask,
        skin_mask_full_f32: skin_mask,
        face_mask_rgba_full_f32: face_rgba,
        roi: hlf_auto_roi(region, current.width, current.height)?,
        preprocess_dispatches,
        preprocess_fallback,
    })
}

#[cfg(windows)]
fn try_cuda_hlf_auto(
    current: &Image,
    pf: u32,
    degree: f32,
    region: NormRect,
    masks: &PortraitMasks,
    fallback_reason: &mut Option<String>,
) -> Option<(Image, Vec<Dispatch>)> {
    if let Some(error) = forced_cuda_failure() {
        *fallback_reason = Some(format!("{error:?}"));
        return None;
    }
    if forced_cuda_empty_dispatch() {
        *fallback_reason = Some("test-forced empty CUDA dispatch".to_string());
        return None;
    }
    let options = HlfAutoProducerOptions::default();
    options.validate().ok()?;
    current.validate().ok()?;
    let pixels = current.pixel_count();
    let face_mask =
        masks.mask_f32(MaskKind::Face, current.width, current.height, region)?;
    let skin_mask =
        masks.mask_f32(MaskKind::Skin, current.width, current.height, region)?;
    if face_mask.len() != pixels || skin_mask.len() != pixels {
        return None;
    }
    let mut source_rgb = Vec::with_capacity(pixels * 3);
    for pixel in current.pixels.chunks_exact(4) {
        source_rgb.push(pixel[0] as f32 / 255.0);
        source_rgb.push(pixel[1] as f32 / 255.0);
        source_rgb.push(pixel[2] as f32 / 255.0);
    }
    let roi = hlf_auto_roi(region, current.width, current.height).ok()?;
    let gpu = match aurora_portrait::cuda_hlf::Hlf::open() {
        Ok(gpu) => gpu,
        Err(error) => {
            *fallback_reason = Some(format!("{error:?}"));
            return None;
        }
    };
    let result = match pf {
        30021 => {
            gpu.auto_face_apply(
                &source_rgb,
                &face_mask,
                &skin_mask,
                current.width,
                current.height,
                options.median_interval,
                options.gauss_interval,
                degree * 100.0,
            )
            .and_then(|(rgb, dispatches)| {
                let image = hlf_rgb_f32_to_image(current, &rgb).map_err(CudaError::Refused)?;
                Ok((image, dispatches))
            })
        }
        30022 => {
            let mut face_rgba = Vec::with_capacity(pixels * 4);
            for value in &face_mask {
                face_rgba.extend_from_slice(&[*value, 0.0, 0.0, 1.0]);
            }
            gpu.auto_body_apply(
                current,
                &source_rgb,
                &face_rgba,
                &skin_mask,
                roi,
                options.median_interval,
                options.gauss_interval,
                degree * 100.0,
                hlf_auto_body_uniforms(),
            )
        }
        _ => return None,
    };
    match result {
        Ok((next, dispatches)) if !dispatches.is_empty() => Some((next, dispatches)),
        Ok(_) => {
            *fallback_reason = Some("CUDA returned no dispatch".to_string());
            None
        }
        Err(error) => {
            *fallback_reason = Some(format!("{error:?}"));
            None
        }
    }
}

impl ProducerInputs {
    pub fn write_pf30025(&mut self, bundle: Pf30025ProducerInputs) {
        self.pf30025 = Some(bundle);
    }

    /// Bind the PF30026 model/renderer outputs to this exact job before the
    /// stage can route them to `teeth_alignment`.
    pub fn write_teeth_alignment(
        &mut self,
        bundle: TeethAlignmentProducerInputs,
        receipt: &PortraitJobReceipt,
    ) -> Result<(), Refusal> {
        let gate = bundle
            .gate_binding
            .ok_or(Refusal::MissingInput("PF30026 edge gate binding"))?;
        gate.validate()?;
        if gate.job_id != receipt.job_id() {
            return Err(Refusal::ValueOutOfRange("PF30026 edge gate job"));
        }
        let effect = bundle
            .effect_rgb
            .as_ref()
            .ok_or(Refusal::MissingInput("PF30026 effect RGB"))?;
        if (effect.width, effect.height) != (receipt.width, receipt.height) {
            return Err(Refusal::ShapeMismatch("PF30026 effect RGB"));
        }
        let pixels = (receipt.width as usize)
            .checked_mul(receipt.height as usize)
            .ok_or(Refusal::ShapeMismatch("PF30026 effect RGB"))?;
        let mask = bundle
            .teeth_mask
            .as_ref()
            .ok_or(Refusal::MissingInput("PF30026 teeth mask"))?;
        let edge_a = bundle
            .edge_plane_a
            .as_ref()
            .ok_or(Refusal::MissingInput("PF30026 edge plane A"))?;
        let edge_b = bundle
            .edge_plane_b
            .as_ref()
            .ok_or(Refusal::MissingInput("PF30026 edge plane B"))?;
        if mask.len() != pixels || edge_a.len() != pixels || edge_b.len() != pixels {
            return Err(Refusal::ShapeMismatch("PF30026 model planes"));
        }
        if bundle.roi.is_none() {
            return Err(Refusal::MissingInput("PF30026 ROI"));
        }
        self.teeth_alignment = Some(bundle);
        Ok(())
    }

    /// Copy the PF90139/PF90140 provider view into owned same-job storage.
    pub fn write_hair_smooth(
        &mut self,
        bundle: HairSmoothInputs<'_>,
        receipt: &PortraitJobReceipt,
    ) -> Result<(), Refusal> {
        if bundle.ticket != receipt.job_id() {
            return Err(Refusal::ValueOutOfRange("PF90139_PF90140 job"));
        }
        if (bundle.width, bundle.height) != (receipt.width, receipt.height) {
            return Err(Refusal::ShapeMismatch("PF90139_PF90140 frame"));
        }
        self.hair_smooth = Some(HairSmoothProducerInputs {
            ticket: bundle.ticket,
            width: bundle.width,
            height: bundle.height,
            roi: bundle.roi,
            mix_strength: bundle.mix_strength,
            hair_mask: bundle
                .hair_mask
                .ok_or(Refusal::MissingInput("PF90139_HAIR_MASK"))?
                .to_vec(),
            mix_mask: bundle
                .mix_mask
                .ok_or(Refusal::MissingInput("PF90139_MIX_MASK"))?
                .to_vec(),
            blend_rgb: bundle
                .blend_rgb
                .ok_or(Refusal::MissingInput("PF90139_BLEND_SOURCE"))?
                .to_vec(),
            detail: bundle
                .detail
                .ok_or(Refusal::MissingInput("PF90139_DETAIL"))?
                .to_vec(),
            hair_stride: bundle.hair_stride,
            mix_stride: bundle.mix_stride,
            blend_stride: bundle.blend_stride,
            detail_stride: bundle.detail_stride,
        });
        Ok(())
    }

    /// Copy the PF90160 provider view into owned same-job storage.
    pub fn write_hair_gloss(
        &mut self,
        bundle: HairGlossInputs<'_>,
        receipt: &PortraitJobReceipt,
    ) -> Result<(), Refusal> {
        if bundle.ticket != receipt.job_id() {
            return Err(Refusal::ValueOutOfRange("PF90160 job"));
        }
        if (bundle.width, bundle.height) != (receipt.width, receipt.height) {
            return Err(Refusal::ShapeMismatch("PF90160 frame"));
        }
        self.hair_gloss = Some(HairGlossProducerInputs {
            ticket: bundle.ticket,
            width: bundle.width,
            height: bundle.height,
            roi: bundle.roi,
            dynamic_scale: bundle.dynamic_scale,
            hair_mask: bundle
                .hair_mask
                .ok_or(Refusal::MissingInput("PF90160_HAIR_MASK"))?
                .to_vec(),
            model_detail: bundle
                .model_detail
                .ok_or(Refusal::MissingInput("PF90160_MODEL_DETAIL"))?
                .to_vec(),
            hair_stride: bundle.hair_stride,
            detail_stride: bundle.detail_stride,
        });
        Ok(())
    }

    /// Sole runtime entry for the twelve routed body-geometry rows. The job must
    /// describe this renderer job (`job_id` non-zero, `frame_size` equal to the
    /// frame the receipt was taken from) and every row must carry the payload
    /// kind its kernel dispatches on. Absent payload stays absent: nothing here
    /// fabricates a mask, shift map, UV stream, mesh or control point.
    pub fn write_body_geometry(
        &mut self,
        job: BodyGeometryJob,
        receipt: &PortraitJobReceipt,
    ) -> Result<(), Refusal> {
        if job.job_id == 0 {
            return Err(Refusal::ValueOutOfRange("body geometry job id"));
        }
        if job.job_id != receipt.job_id() {
            return Err(Refusal::ValueOutOfRange("body geometry job id"));
        }
        if job.frame_size != (receipt.width, receipt.height) {
            return Err(Refusal::ShapeMismatch("body geometry frame"));
        }
        for (pf, input) in &job.effects {
            if !portrait_body_stage::supports(*pf) {
                return Err(Refusal::UnknownEffect(*pf));
            }
            if !portrait_body_stage::accepts(*pf, input) {
                return Err(Refusal::ShapeMismatch("body geometry effect kind"));
            }
        }
        self.body_geometry = Some(job);
        Ok(())
    }

    pub fn body_geometry(&self) -> Option<&BodyGeometryJob> {
        self.body_geometry.as_ref()
    }

    /// Sole runtime entry from model planes into the ten repair rows. The planes are
    /// validated and bound to the exact `source` frame before anything is stored, so a
    /// refused call leaves the previous binding untouched. Nothing here synthesizes a
    /// plane: every absent named plane stays a typed refusal.
    pub fn write_repair_planes(
        &mut self,
        ticket: Option<u64>,
        source: &Image,
        planes: RepairModelPlanes,
        receipt: &PortraitJobReceipt,
    ) -> Result<RepairOutputReceipt, Refusal> {
        // A zero or foreign ticket is refused before the planes are read, so a stale
        // producer cannot be masked by an otherwise well-formed payload.
        let ticket = match ticket {
            Some(ticket) if ticket == 0 => return Err(Refusal::MissingInput("repair job ticket")),
            Some(ticket) if ticket == receipt.job_id() => ticket,
            Some(_) => return Err(Refusal::ValueOutOfRange("repair job ticket")),
            None => return Err(Refusal::MissingInput("repair job ticket")),
        };
        if source.width != receipt.width || source.height != receipt.height {
            return Err(Refusal::ShapeMismatch("repair source frame size"));
        }
        // A stale or foreign frame is named as a frame problem before any plane shape is
        // consulted, and before the degree gate could turn it into a silent no-op.
        let frame =
            PortraitFrameReceipt::from_renderer_frame(&source.pixels, source.width, source.height)?;
        if frame.fingerprint() != receipt.frame_receipt().fingerprint() {
            return Err(Refusal::ValueOutOfRange("repair source frame SHA256"));
        }
        let mut inputs = self.repairs.clone().unwrap_or_default();
        let bound = inputs.insert_same_frame(Some(ticket), source, planes)?;
        self.repairs = Some(inputs);
        Ok(bound)
    }

    pub fn repairs(&self) -> Option<&RepairProducerInputs> {
        self.repairs.as_ref()
    }

    pub fn write_parsing(&mut self, bundle: crate::face_parsing::FaceParsingProducerInputs) {
        self.parsing = Some(bundle);
    }

    pub fn write_teeth(&mut self, bundle: TeethProducerInputs) {
        self.teeth = Some(bundle);
    }

    pub fn write_plump(&mut self, bundle: PlumpProducerInputs) {
        self.plump = Some(bundle);
    }

    pub fn write_wrinkle(&mut self, bundle: WrinkleProducerInputs) {
        self.wrinkle = Some(bundle);
    }

    pub fn write_external_historical_skin(
        &mut self,
        inputs: Option<HistoricalSkinExternalInputs>,
    ) -> Result<(), Refusal> {
        self.historical_skin = Some(build_historical_skin_inputs(inputs)?);
        Ok(())
    }

    pub fn write_immediate_historical_skin(
        &mut self,
        frame: &Image,
        parsing: &crate::face_parsing::FaceParsingProducerInputs,
        inputs: portrait_historical_skin::HistoricalSkinImmediateInputs,
    ) -> Result<(), Refusal> {
        self.historical_skin = Some(
            portrait_historical_skin::build_immediate_historical_skin_inputs(
                frame, parsing, inputs,
            )?,
        );
        Ok(())
    }

    pub fn write_historical_tone(
        &mut self,
        payload: crate::portrait_historical_tone_provider::HistoricalToneProviderPayload,
        complete: bool,
    ) -> Result<(), Refusal> {
        self.historical_tone = Some(if complete {
            crate::portrait_historical_tone_provider::build_complete_historical_tone_inputs(payload)?
        } else {
            crate::portrait_historical_tone_provider::build_historical_tone_inputs(payload)?
        });
        Ok(())
    }

    pub fn write_pf90004(&mut self, bundle: Pf90004ProducerInputs) {
        self.pf90004 = Some(bundle);
    }

    pub fn fingerprint(&self) -> u64 {
        let mut hash = fnv1a64(FNV1A64_OFFSET_BASIS, b"aurora-producer-inputs-v3");
        hash = fnv_u64(hash, u64::from(self.parsing.is_some()));
        if let Some(parsing) = self.parsing.as_ref() {
            hash = fnv_u64(hash, parsing.job_id);
            hash = fnv_u64(hash, u64::from(parsing.width));
            hash = fnv_u64(hash, u64::from(parsing.height));
            hash = fnv_u64(hash, u64::from(parsing.channels));
            hash = fnv1a64(hash, &parsing.model_sha256);
            for row in parsing.source_to_crop {
                for value in row {
                    hash = fnv_u64(hash, value.to_bits() as u64);
                }
            }
            hash = fnv_usize(hash, parsing.masks.len());
            hash = fnv1a64(hash, &parsing.masks);
            hash = fnv1a64(hash, parsing.mask_provider.as_bytes());
        }
        if let Some(slim) = self.slim.as_ref() {
            hash = fnv_u64(fnv1a64(hash, b"PF8025/PF8026"), slim.fingerprint());
        }
        if let Some(face) = self.face_geometry.as_ref() {
            hash = fnv_u64(fnv1a64(hash, b"face geometry"), face.fingerprint());
        }
        if let Some(body) = self.body_geometry.as_ref() {
            hash = fnv_u64(fnv1a64(hash, b"body geometry"), body.fingerprint());
        }
        if let Some(repairs) = self.repairs.as_ref() {
            hash = repairs.fingerprint(hash);
        }
        if let Some(tone) = self.historical_tone.as_ref() {
            hash = fnv_u64(
                fnv1a64(hash, b"historical tone"),
                portrait_historical_tone::fingerprint(tone),
            );
        }
        if let Some(skin) = self.historical_skin.as_ref() {
            hash = fnv_u64(
                fnv1a64(hash, b"historical skin"),
                portrait_historical_skin::fingerprint(skin),
            );
        }
        hash = fnv_u64(hash, u64::from(self.teeth.is_some()));
        if let Some(teeth) = self.teeth.as_ref() {
            hash = fnv_u64(hash, teeth.job_id);
            hash = fnv_u64(hash, u64::from(teeth.mask_width));
            hash = fnv_u64(hash, u64::from(teeth.mask_height));
            hash = fnv1a64(hash, &teeth.mask_sha256);
            hash = fnv_usize(hash, teeth.mask_g.len());
            hash = fnv1a64(hash, &teeth.mask_g);
            hash = fnv_u64(hash, teeth.bright_ui.to_bits() as u64);
            hash = fnv_u64(hash, teeth.deyellow_ui.to_bits() as u64);
            hash = fnv_usize(hash, teeth.whiten_lut.len());
            hash = fnv1a64(hash, &teeth.whiten_lut);
            hash = fnv_usize(hash, teeth.bright_lut.len());
            hash = fnv1a64(hash, &teeth.bright_lut);
        }
        hash = match self.smile.as_ref() {
            None => fnv_u64(hash, 0),
            Some(smile) => {
                hash = fnv_u64(hash, 1);
                hash = fnv_u64(hash, smile.ticket);
                hash = fnv_option_bytes(hash, smile.material.as_deref().map(str::as_bytes));
                hash = match smile.roi {
                    None => fnv_u64(hash, 0),
                    Some((x, y, width, height)) => {
                        hash = fnv_u64(hash, 1);
                        for value in [x, y, width, height] {
                            hash = fnv_u64(hash, u64::from(value));
                        }
                        hash
                    }
                };
                hash = match smile.sigmoid_edge_response_3204548346.as_ref() {
                    None => fnv_u64(hash, 0),
                    Some(response) => {
                        hash = fnv_u64(hash, 1);
                        hash = fnv_u64(
                            hash,
                            u64::from(SmileSigmoidEdgeResponse::OUTPUT_ID),
                        );
                        hash = fnv_u64(hash, u64::from(response.width));
                        hash = fnv_u64(hash, u64::from(response.height));
                        fnv_option_f32_plane(hash, Some(response.values.as_slice()))
                    }
                };
                hash = match smile.rgb_reconstruction_780929437.as_ref() {
                    None => fnv_u64(hash, 0),
                    Some(field) => {
                        hash = fnv_u64(hash, 1);
                        hash = fnv_u64(
                            hash,
                            u64::from(SmileRgbReconstructionField::OUTPUT_ID),
                        );
                        hash = fnv_u64(hash, u64::from(field.width));
                        hash = fnv_u64(hash, u64::from(field.height));
                        fnv_option_f32_plane(hash, Some(field.values.as_slice()))
                    }
                };
                hash = match smile.host_split_spec {
                    None => fnv_u64(hash, 0),
                    Some(spec) => {
                        hash = fnv_u64(hash, 1);
                        smile_split_spec_fingerprint(hash, spec)
                    }
                };
                hash = match smile.channel_split.as_ref() {
                    None => fnv_u64(hash, 0),
                    Some(split) => {
                        hash = fnv_u64(hash, 1);
                        hash = smile_split_spec_fingerprint(hash, split.spec);
                        hash = fnv_option_f32_plane(hash, Some(split.model_dx.as_slice()));
                        fnv_option_f32_plane(hash, Some(split.model_dy.as_slice()))
                    }
                };
                hash = fnv_option_bytes(hash, smile.lip_mask.as_deref());
                fnv_option_bytes(hash, smile.mouth_mask.as_deref())
            }
        };
        hash = match self.stray_hair.as_ref() {
            None => fnv_u64(hash, 0),
            Some(stray_hair) => {
                hash = fnv_u64(hash, 1);
                hash = match stray_hair.hairfly_v1.as_ref() {
                    None => fnv_u64(hash, 0),
                    Some(hairfly_v1) => {
                        hash = fnv_u64(hash, 1);
                        hash = fnv_u64(hash, hairfly_v1.width as u64);
                        hash = fnv_u64(hash, hairfly_v1.height as u64);
                        hash = fnv_usize(hash, hairfly_v1.pixels.len());
                        fnv1a64(hash, &hairfly_v1.pixels)
                    }
                };
                hash = fnv_option_bytes(hash, stray_hair.raw_mask.as_deref());
                hash = fnv_option_bytes(hash, stray_hair.person_mask.as_deref());
                hash = fnv_option_bytes(hash, stray_hair.background_mask.as_deref());
                hash = fnv_option_f32(hash, stray_hair.native_degree);
                hash = fnv_option_f32(hash, stray_hair.native_person_area);
                fnv_option_f32(hash, stray_hair.native_background_area)
            }
        };
        hash = match self.wrinkle.as_ref() {
            None => fnv_u64(hash, 0),
            Some(wrinkle) => {
                let mut hash = fnv_u64(hash, 1);
                hash = fnv_u64(hash, wrinkle.ticket);
                hash = fnv_u64(hash, u64::from(wrinkle.width));
                hash = fnv_u64(hash, u64::from(wrinkle.height));
                for plane in [&wrinkle.base_full, &wrinkle.residual_full] {
                    hash = match plane.as_ref() {
                        None => fnv_u64(hash, 0),
                        Some(plane) => {
                            let mut hash = fnv_u64(hash, 1);
                            hash = fnv_u64(hash, u64::from(plane.width));
                            hash = fnv_u64(hash, u64::from(plane.height));
                            hash = fnv_usize(hash, plane.pixels.len());
                            fnv1a64(hash, &plane.pixels)
                        }
                    };
                }
                wrinkle.fnv_fingerprint(hash)
            }
        };
        hash = match self.pf30025.as_ref() {
            None => fnv_u64(hash, 0),
            Some(pf30025) => {
                hash = fnv_u64(hash, 1);
                hash = fnv_u64(hash, pf30025.receipt.job_id);
                hash = fnv_u64(hash, pf30025.receipt.model_id as u64);
                hash = fnv_u64(hash, pf30025.receipt.width as u64);
                hash = fnv_u64(hash, pf30025.receipt.height as u64);
                hash = fnv_u64(hash, pf30025.receipt.source_stride as u64);
                hash = fnv1a64(hash, &pf30025.receipt.source_sha256);
                hash = fnv1a64(hash, &pf30025.receipt.repair_sha256);
                hash = fnv1a64(hash, &pf30025.receipt.gate_sha256);
                hash = fnv_usize(hash, pf30025.repair_780929437_bgra.len());
                hash = fnv1a64(hash, &pf30025.repair_780929437_bgra);
                hash = fnv_usize(hash, pf30025.gate_3204548346.len());
                for value in &pf30025.gate_3204548346 {
                    hash = fnv_u64(hash, value.to_bits() as u64);
                }
                hash = fnv_usize(hash, pf30025.teeth_mask.len());
                for value in &pf30025.teeth_mask {
                    hash = fnv_u64(hash, value.to_bits() as u64);
                }
                for value in [
                    pf30025.roi.x as u32,
                    pf30025.roi.y as u32,
                    pf30025.roi.width as u32,
                    pf30025.roi.height as u32,
                ] {
                    hash = fnv_u64(hash, value as u64);
                }
                hash
            }
        };
        hash = match self.teeth_alignment.as_ref() {
            None => fnv_u64(hash, 0),
            Some(alignment) => {
                hash = fnv_u64(hash, 1);
                hash = match alignment.effect_rgb.as_ref() {
                    None => fnv_u64(hash, 0),
                    Some(effect) => {
                        hash = fnv_u64(hash, 1);
                        hash = fnv_u64(hash, effect.width as u64);
                        hash = fnv_u64(hash, effect.height as u64);
                        hash = fnv_usize(hash, effect.pixels.len());
                        fnv1a64(hash, &effect.pixels)
                    }
                };
                hash = fnv_option_bytes(hash, alignment.teeth_mask.as_deref());
                hash = fnv_option_f32_plane(hash, alignment.edge_plane_a.as_deref());
                hash = fnv_option_f32_plane(hash, alignment.edge_plane_b.as_deref());
                hash = match alignment.gate_binding {
                    None => fnv_u64(hash, 0),
                    Some(binding) => {
                        hash = fnv_u64(hash, 1);
                        hash = fnv_u64(hash, binding.job_id);
                        hash = fnv_u64(
                            hash,
                            match binding.plane_a {
                                Pf30026EdgeOutput::Output0 => 0,
                                Pf30026EdgeOutput::Output1 => 1,
                            },
                        );
                        fnv_u64(
                            hash,
                            match binding.plane_b {
                                Pf30026EdgeOutput::Output0 => 0,
                                Pf30026EdgeOutput::Output1 => 1,
                            },
                        )
                    }
                };
                match alignment.roi {
                    None => fnv_u64(hash, 0),
                    Some(roi) => {
                        hash = fnv_u64(hash, 1);
                        hash = fnv_u64(hash, roi.x.to_bits() as u64);
                        hash = fnv_u64(hash, roi.y.to_bits() as u64);
                        hash = fnv_u64(hash, roi.width.to_bits() as u64);
                        fnv_u64(hash, roi.height.to_bits() as u64)
                    }
                }
            }
        };
        hash = match self.eye_bag.as_ref() {
            None => fnv_u64(hash, 0),
            Some(eye_bag) => {
                hash = fnv_u64(hash, 1);
                hash = fnv_u64(hash, eye_bag.ticket);
                hash = fnv_u64(hash, eye_bag.model_id);
                hash = fnv_u64(hash, u64::from(eye_bag.frame_width));
                hash = fnv_u64(hash, u64::from(eye_bag.frame_height));
                hash = fnv_u64(
                    hash,
                    u64::from(
                        eye_bag.color_units
                            == aurora_portrait::eye_bag_alignment::EyeBagColorUnits::SourceBytes,
                    ),
                );
                hash = fnv_u64(
                    hash,
                    u64::from(
                        eye_bag.geometry_class
                            == aurora_portrait::eye_bag_alignment::EyeBagGeometryClass::VendorSameJob,
                    ),
                );
                hash = match eye_bag.geometry_receipt {
                    None => fnv_u64(hash, 0),
                    Some(receipt) => {
                        hash = fnv_u64(hash, 1);
                        hash = fnv_u64(hash, receipt.job_ticket);
                        hash = fnv_u64(hash, receipt.vendor_job_id);
                        hash = fnv_u64(hash, receipt.face_record_id);
                        hash = fnv_u64(hash, receipt.landmark_receipt);
                        hash = fnv_u64(hash, receipt.model_tensor_receipt);
                        for byte in receipt.host_sha256 {
                            hash = fnv_u64(hash, u64::from(byte));
                        }
                        hash = fnv_u64(hash, u64::from(receipt.m1_call_site));
                        hash = fnv_u64(hash, u64::from(receipt.m2_call_site));
                        hash = fnv_u64(hash, u64::from(receipt.work_width));
                        hash = fnv_u64(hash, u64::from(receipt.work_height));
                        for value in receipt.work_to_destination {
                            hash = fnv_u64(hash, value.to_bits() as u64);
                        }
                        hash = fnv_u64(hash, receipt.destination_roi.x as u64);
                        hash = fnv_u64(hash, receipt.destination_roi.y as u64);
                        hash = fnv_u64(hash, receipt.destination_roi.w as u64);
                        fnv_u64(hash, receipt.destination_roi.h as u64)
                    }
                };
                hash = fnv_usize(hash, eye_bag.aligned_color_field_rgb_f32.len());
                hash = fnv_usize(hash, eye_bag.aligned_response_mask.len());
                hash = fnv_usize(hash, eye_bag.destination_model_input_rgb_f32.len());
                for value in &eye_bag.aligned_color_field_rgb_f32 {
                    hash = fnv_u64(hash, value.to_bits() as u64);
                }
                for value in &eye_bag.aligned_response_mask {
                    hash = fnv_u64(hash, value.to_bits() as u64);
                }
                for value in &eye_bag.destination_model_input_rgb_f32 {
                    hash = fnv_u64(hash, value.to_bits() as u64);
                }
                hash
            }
        };
        hash = fnv_u64(hash, u64::from(self.pf90021.is_some()));
        if let Some(pf90021) = self.pf90021.as_ref() {
            hash = fnv_u64(
                hash,
                match pf90021.direction {
                    Pf90021Direction::Neutral => 0,
                    Pf90021Direction::Positive => 1,
                    Pf90021Direction::Negative => 2,
                },
            );
            for lut in [&pf90021.positive_lut, &pf90021.negative_lut] {
                hash = fnv_usize(hash, lut.nodes().len());
                for node in lut.nodes() {
                    hash = fnv_u64(hash, node.to_bits() as u64);
                }
            }
            hash = fnv_usize(hash, pf90021.mask_r.len());
            hash = fnv1a64(hash, &pf90021.mask_r);
            for value in [
                pf90021.runtime.degree01,
                pf90021.runtime.opacity01,
                pf90021.runtime.shelter,
                pf90021.runtime.region,
            ] {
                hash = fnv_u64(hash, value.to_bits() as u64);
            }
            hash = fnv_u64(hash, pf90021.same_job.job_id);
            hash = fnv_u64(hash, pf90021.same_job.ticket);
            hash = fnv_u64(hash, pf90021.same_job.face_id as u64);
            hash = fnv_u64(hash, pf90021.same_job.frame.fingerprint());
            hash = fnv_u64(hash, pf90021.same_job.mask_texture_id);
            hash = fnv_u64(hash, u64::from(pf90021.same_job.mask_channel));
            hash = fnv_u64(
                hash,
                u64::from(
                    pf90021.same_job.mask_source
                        == Pf90021MaskSource::FaceParsingChannel0Rasterized,
                ),
            );
            for bits in pf90021.same_job.roi_bits {
                hash = fnv_u64(hash, u64::from(bits));
            }
            for value in [
                pf90021.direction_receipt.effect_range[0],
                pf90021.direction_receipt.effect_range[1],
                pf90021.direction_receipt.effect_range[2],
                pf90021.direction_receipt.param_range[0],
                pf90021.direction_receipt.param_range[1],
                pf90021.direction_receipt.param_range[2],
                pf90021.direction_receipt.input01,
                pf90021.direction_receipt.output01,
            ] {
                hash = fnv_u64(hash, value.to_bits() as u64);
            }
            for value in [
                pf90021.runtime_receipt.degree01,
                pf90021.runtime_receipt.opacity01,
                pf90021.runtime_receipt.shelter01,
                pf90021.runtime_receipt.region01,
            ] {
                hash = fnv_u64(hash, value.to_bits() as u64);
            }
        }
        hash = fnv_u64(hash, u64::from(self.pf90004.is_some()));
        if let Some(pf90004) = self.pf90004.as_ref() {
            hash = fnv_u64(hash, u64::from(pf90004.width));
            hash = fnv_u64(hash, u64::from(pf90004.height));
            hash = fnv_usize(hash, pf90004.skin_mask.len());
            hash = fnv1a64(hash, &pf90004.skin_mask);
            hash = fnv_usize(hash, pf90004.parsing_rgba.len());
            hash = fnv1a64(hash, &pf90004.parsing_rgba);
            hash = fnv_usize(hash, pf90004.face_rgba.len());
            hash = fnv1a64(hash, &pf90004.face_rgba);
            hash = fnv_u64(hash, u64::from(pf90004.value_mean.to_bits()));
            for value in [
                pf90004.roi.x,
                pf90004.roi.y,
                pf90004.roi.width,
                pf90004.roi.height,
            ] {
                hash = fnv_u64(hash, u64::from(value.to_bits()));
            }
        }
        hash = fnv_u64(hash, u64::from(self.hlf_body_adjust_lut.is_some()));
        if let Some(lut) = self.hlf_body_adjust_lut.as_ref() {
            hash = fnv_u64(hash, lut.width as u64);
            hash = fnv_u64(hash, lut.height as u64);
            hash = fnv_usize(hash, lut.luma.len());
            hash = fnv1a64(hash, &lut.luma);
        }
        hash = fnv_u64(hash, u64::from(self.makeup_bundle.is_some()));
        if let Some(bundle) = self.makeup_bundle.as_ref() {
            hash = fnv_u64(hash, bundle.fingerprint());
        }
        hash = fnv_u64(hash, u64::from(self.plump.is_some()));
        if let Some(plump) = self.plump.as_ref() {
            hash = fnv_u64(hash, plump.ticket);
            hash = fnv_u64(hash, u64::from(plump.face_id));
            for value in plump.slots {
                hash = fnv_u64(hash, value.to_bits() as u64);
            }
            hash = fnv_usize(hash, plump.requested_pfs.len());
            for pf in &plump.requested_pfs {
                hash = fnv_u64(hash, u64::from(*pf));
            }
            hash = fnv_option_f32(hash, plump.full_face_descriptor);
            hash = fnv_u64(hash, u64::from(plump.session.model_id));
            hash = fnv1a64(hash, &plump.session.net_sha256);
            for value in plump.session.input_shape {
                hash = fnv_u64(hash, u64::from(value));
            }
            hash = fnv_u64(hash, u64::from(plump.session.input_tensor.len() as u32));
            hash = fnv1a64(hash, plump.session.input_tensor.as_bytes());
            hash = fnv_u64(hash, u64::from(plump.session.output_tensor.len() as u32));
            hash = fnv1a64(hash, plump.session.output_tensor.as_bytes());
            hash = match plump.model_input.as_ref() {
                None => fnv_u64(hash, 0),
                Some(input) => {
                    hash = fnv_u64(hash, 1);
                    for value in input.shape {
                        hash = fnv_u64(hash, u64::from(value));
                    }
                    hash = fnv1a64(hash, &input.digest);
                    for value in &input.nchw {
                        hash = fnv_u64(hash, value.to_bits() as u64);
                    }
                    hash
                }
            };
            hash = match plump.model_output.as_ref() {
                None => fnv_u64(hash, 0),
                Some(output) => {
                    hash = fnv_u64(hash, 1);
                    hash = fnv_u64(hash, u64::from(output.stamp.model_id));
                    hash = fnv1a64(hash, &output.stamp.net_sha256);
                    hash = fnv1a64(hash, &output.stamp.input_digest);
                    for value in output.stamp.input_shape {
                        hash = fnv_u64(hash, u64::from(value));
                    }
                    hash = fnv1a64(hash, output.stamp.output_tensor.as_bytes());
                    for value in output.tensor.shape {
                        hash = fnv_u64(hash, u64::from(value));
                    }
                    for value in &output.tensor.data {
                        hash = fnv_u64(hash, value.to_bits() as u64);
                    }
                    hash
                }
            };
            for value in plump.channel_map.color {
                hash = fnv_u64(hash, value as u64);
            }
            hash = fnv_u64(hash, plump.channel_map.split_mask as u64);
            hash = fnv_u64(
                hash,
                u64::from(plump.color_encoding == aurora_portrait::kernels::plump_pipeline::ColorEncoding::Byte),
            );
            for value in [
                plump.roi.x,
                plump.roi.y,
                plump.roi.width,
                plump.roi.height,
            ] {
                hash = fnv_u64(hash, u64::from(value));
            }
            hash = match plump.final_composite.as_ref() {
                None => fnv_u64(hash, 0),
                Some(composite) => {
                    hash = fnv_u64(hash, 1);
                    hash = fnv_u64(hash, composite.receipt.ticket);
                    hash = fnv_u64(hash, u64::from(composite.receipt.face_id));
                    hash = fnv_usize(hash, composite.receipt.requested_pfs.len());
                    for pf in &composite.receipt.requested_pfs {
                        hash = fnv_u64(hash, u64::from(*pf));
                    }
                    hash = fnv1a64(hash, &composite.receipt.source_sha256);
                    hash = fnv_u64(hash, u64::from(composite.receipt.model_id));
                    hash = fnv1a64(hash, &composite.receipt.net_sha256);
                    hash = fnv1a64(hash, &composite.receipt.model_input_digest);
                    hash = fnv1a64(hash, composite.receipt.output_tensor.as_bytes());
                    hash = fnv_u64(hash, u64::from(composite.frame.width));
                    hash = fnv_u64(hash, u64::from(composite.frame.height));
                    hash = fnv_usize(hash, composite.frame.pixels.len());
                    fnv1a64(hash, &composite.frame.pixels)
                }
            };
        }
        hash
    }
}

/// PF108 same-job inputs. `pc:108` supplies the common UI fraction; the other
/// two controls remain explicit fields because no UI-to-native fallback exists.
#[derive(Clone, Copy, Debug)]
pub struct TeethInputs<'a> {
    pub job_id: u64,
    pub mask_width: u32,
    pub mask_height: u32,
    pub mask_source: TeethMaskSource,
    pub mask_format: TeethMaskFormat,
    pub mask_sha256: [u8; 32],
    pub whiten_lut_identity: TeethLutIdentity,
    pub bright_lut_identity: TeethLutIdentity,
    pub mask_g: &'a [u8],
    pub bright_ui: f32,
    pub deyellow_ui: f32,
}

/// PF30026 final-blend inputs from one worker job.
///
/// Every plane is at the rendered frame size. `roi` identifies the normalized
/// source region and must match the stage request.
#[derive(Clone, Copy, Debug, Default)]
pub struct TeethAlignmentInputs<'a> {
    pub effect_rgb: Option<&'a Image>,
    pub teeth_mask: Option<&'a [u8]>,
    pub edge_plane_a: Option<&'a [f32]>,
    pub edge_plane_b: Option<&'a [f32]>,
    pub gate_binding: Option<Pf30026GateBinding>,
    pub roi: Option<NormRect>,
}

/// PF90004 same-job binding. The parser's skin mask is not a substitute for
/// the vendor's skin/parsing/face planes or its valueMean receipt.
#[derive(Clone, Copy, Debug)]
pub struct RadianceInputs<'a> {
    pub frame_width: Option<u32>,
    pub frame_height: Option<u32>,
    pub skin_mask: Option<&'a [u8]>,
    pub parsing_rgba: Option<&'a [u8]>,
    pub face_rgba: Option<&'a [u8]>,
    pub value_mean: Option<f32>,
    pub roi: Option<NormRect>,
}

/// PF21026 uses two independent beard masks and a same-size repair image.
#[derive(Clone, Copy, Debug)]
pub struct BeardInputs<'a> {
    pub mask_a: &'a [u8],
    pub mask_b: &'a [u8],
    pub repair_rgb: &'a Image,
}

/// PF90002 face-moisture inputs. Every plane is for the rendered frame.
#[derive(Clone, Copy)]
pub struct MoistureFaceInputs<'a> {
    pub width: u32,
    pub height: u32,
    pub mask_rgba: &'a [u8],
    pub blur_alpha: &'a [u8],
    pub channel: MaskChannel,
    pub skin_map_width: u32,
    pub skin_map_height: u32,
    pub skin_map: &'a [f32],
    pub hardlight: &'a [f32],
    pub beauty: &'a [f32],
    pub pivot: f32,
    pub gain: f32,
}

/// PF90010 body-moisture inputs. Weight planes remain full-frame for a partial
/// ROI; `degree` is the captured native value rather than `pc:90010`.
#[derive(Clone, Copy, Debug)]
pub struct MoistureBodyInputs<'a> {
    pub width: u32,
    pub height: u32,
    pub weight_skin: &'a [f32],
    pub weight_dog: &'a [f32],
    pub beauty: &'a [f32],
    pub degree: f32,
    pub roi: (u32, u32, u32, u32),
}

/// PF90116 aligned model outputs, never the parser's 512-square grid.
#[derive(Clone, Copy, Debug)]
pub struct EyeBagInputs<'a> {
    pub ticket: u64,
    pub model_id: u64,
    pub color_units: aurora_portrait::eye_bag_alignment::EyeBagColorUnits,
    pub geometry_class: aurora_portrait::eye_bag_alignment::EyeBagGeometryClass,
    pub geometry_receipt:
        Option<aurora_portrait::eye_bag_alignment::EyeBagVendorGeometryReceipt>,
    pub width: u32,
    pub height: u32,
    pub aligned_color_field_rgb_f32: &'a [f32],
    pub aligned_response_mask: &'a [f32],
}

/// PF90139/PF90140 four-plane bundle. Planes are ROI-local with independent strides.
#[derive(Clone, Copy, Debug)]
pub struct HairSmoothInputs<'a> {
    pub ticket: u64,
    pub width: u32,
    pub height: u32,
    pub roi: (u32, u32, u32, u32),
    /// Captured PF90140 descriptor `+0x80`, already normalized to `[0,1]`.
    pub mix_strength: f32,
    pub hair_mask: Option<&'a [u8]>,
    pub mix_mask: Option<&'a [u8]>,
    pub blend_rgb: Option<&'a [u8]>,
    pub detail: Option<&'a [u8]>,
    pub hair_stride: u32,
    pub mix_stride: u32,
    pub blend_stride: u32,
    pub detail_stride: u32,
}

/// PF90160 ROI-local mask/detail bundle and captured descriptor `+0x60` scale.
#[derive(Clone, Copy, Debug)]
pub struct HairGlossInputs<'a> {
    pub ticket: u64,
    pub width: u32,
    pub height: u32,
    pub roi: (u32, u32, u32, u32),
    pub dynamic_scale: f32,
    pub hair_mask: Option<&'a [u8]>,
    pub model_detail: Option<&'a [u8]>,
    pub hair_stride: u32,
    pub detail_stride: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HairSmoothProducerInputs {
    pub ticket: u64,
    pub width: u32,
    pub height: u32,
    pub roi: (u32, u32, u32, u32),
    pub mix_strength: f32,
    pub hair_mask: Vec<u8>,
    pub mix_mask: Vec<u8>,
    pub blend_rgb: Vec<u8>,
    pub detail: Vec<u8>,
    pub hair_stride: u32,
    pub mix_stride: u32,
    pub blend_stride: u32,
    pub detail_stride: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HairGlossProducerInputs {
    pub ticket: u64,
    pub width: u32,
    pub height: u32,
    pub roi: (u32, u32, u32, u32),
    pub dynamic_scale: f32,
    pub hair_mask: Vec<u8>,
    pub model_detail: Vec<u8>,
    pub hair_stride: u32,
    pub detail_stride: u32,
}

/// PF90088's two hard producers and the per-face values consumed by the final
/// composite. `pc:90088` remains the UI gate; native values come from this job.
#[derive(Clone, Copy, Debug, Default)]
pub struct StrayHairInputs<'a> {
    pub hairfly_v1: Option<&'a Image>,
    pub raw_mask: Option<&'a [u8]>,
    pub person_mask: Option<&'a [u8]>,
    pub background_mask: Option<&'a [u8]>,
    pub native_degree: Option<f32>,
    pub native_person_area: Option<f32>,
    pub native_background_area: Option<f32>,
}

/// PF100007 open-mouth model fields and ROI.
#[derive(Clone, Copy, Debug, Default)]
pub struct SmileInputs<'a> {
    pub material: Option<&'a str>,
    pub lip_mask: Option<&'a [u8]>,
    pub mouth_mask: Option<&'a [u8]>,
    pub model_dx: Option<&'a [f32]>,
    pub model_dy: Option<&'a [f32]>,
    pub roi: Option<Rect>,
}

pub enum SuitPassKindInput<'a> {
    Color,
    Image {
        material: Option<&'a Image>,
        material_uv: Option<&'a [[f32; 2]]>,
    },
    ImageUnmasked {
        material: Option<&'a Image>,
        material_uv: Option<&'a [[f32; 2]]>,
    },
    Unavailable(&'static str),
}

/// One PF90033 ordered pass with explicit ownership of every material and UV.
pub struct SuitPassInput<'a> {
    pub kind: SuitPassKindInput<'a>,
    pub mask: Option<&'a Image>,
    pub mask_uv: Option<&'a [[f32; 2]]>,
    pub layer: Option<MakeupLayer>,
    pub shelter: f32,
    pub region: f32,
}

/// Static PF90033 fields carried by the exported suit preset. Runtime pixels,
/// UVs and weights live in [`SuitPassSameJobInputs`].
#[derive(Clone, Copy, Debug)]
pub struct SuitPassPlan<'a> {
    pub order: u32,
    pub effect: &'a str,
    pub degree: Option<f32>,
    pub opacity: Option<f32>,
    pub filter_type: Option<u32>,
    pub material_path: Option<&'a str>,
    pub mask_field: Option<&'a str>,
    pub mask_value: Option<u32>,
    pub shelter_type: Option<i32>,
    pub force_use_shelter: Option<bool>,
    /// Geometry metadata only. This field never becomes the region weight.
    pub region_rect: Option<[i32; 4]>,
    pub mix_color: Option<[u8; 4]>,
}

/// Inputs that must come from the same renderer job as the output frame.
#[derive(Clone, Copy, Debug)]
pub struct SuitPassSameJobInputs<'a> {
    pub mask: Option<&'a Image>,
    pub mask_uv: Option<&'a [[f32; 2]]>,
    pub material: Option<&'a Image>,
    pub material_uv: Option<&'a [[f32; 2]]>,
    pub shelter: Option<f32>,
    pub region: Option<f32>,
}

fn suit_mask_channel(field: &str, value: u32) -> Result<MakeupMaskChannel, Refusal> {
    match (field, value) {
        ("BlusherMaskType", 15) => Ok(MakeupMaskChannel::Rgba),
        ("EyeBrowMaskType", 5) => Ok(MakeupMaskChannel::RB),
        ("EyeMaskType", 10) => Ok(MakeupMaskChannel::Rgb),
        ("EyeMaskType", 4) | ("EyeLashMaskType", 4) => Ok(MakeupMaskChannel::RG),
        ("LipMaskType", 3) => Ok(MakeupMaskChannel::R),
        _ => Err(Refusal::MissingInput("suit mask channel")),
    }
}

fn validate_suit_uv(
    uv: &[[f32; 2]],
    pixel_count: usize,
    name: &'static str,
) -> Result<(), Refusal> {
    if uv.len() != pixel_count {
        return Err(Refusal::ShapeMismatch(name));
    }
    if !uv
        .iter()
        .all(|point| point[0].is_finite() && point[1].is_finite())
    {
        return Err(Refusal::ValueOutOfRange(name));
    }
    Ok(())
}

fn suit_weight(value: Option<f32>, name: &'static str) -> Result<f32, Refusal> {
    let value = required(value, name)?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Refusal::ValueOutOfRange(name));
    }
    Ok(value)
}

/// Converts one static pass plus same-job data into the existing stage input.
/// Missing runtime data is refused by name rather than defaulted.
pub fn produce_suit_pass<'a>(
    pixel_count: usize,
    plan: &SuitPassPlan<'_>,
    inputs: SuitPassSameJobInputs<'a>,
) -> Result<SuitPassInput<'a>, Refusal> {
    let mask = required(inputs.mask, "suit mask")?;
    mask.validate()?;
    let mask_uv = required(inputs.mask_uv, "suit mask uv")?;
    validate_suit_uv(mask_uv, pixel_count, "suit mask uv")?;

    let (mask_field, mask_value) = match (plan.mask_field, plan.mask_value) {
        (Some(field), Some(value)) => (field, value),
        (None, None) => return Err(Refusal::MissingInput("suit mask spec")),
        _ => return Err(Refusal::ShapeMismatch("suit mask spec")),
    };
    let mask_channel = suit_mask_channel(mask_field, mask_value)?;

    let degree = required(plan.degree, "suit degree")?;
    if !degree.is_finite() || !(0.0..=100.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("suit degree"));
    }
    let opacity = required(plan.opacity, "suit opacity")?;
    if !opacity.is_finite() || !(0.0..=100.0).contains(&opacity) {
        return Err(Refusal::ValueOutOfRange("suit opacity"));
    }
    let filter_type = required(plan.filter_type, "suit filter type")?;
    let mix_color = required(plan.mix_color, "suit mix color")?;
    let layer = MakeupLayer {
        mix_color,
        opacity_percent: opacity,
        degree01: degree / 100.0,
        filter_type,
        mask_channel,
    };

    let kind = match plan.effect {
        "BlusherImageMakeup"
        | "NormalEyeBrowImageMakeup"
        | "SymEyeImageMakeup"
        | "SymEyeLashImageMakeup"
        | "LipStickImageMakeup" => {
            required(plan.material_path, "suit material path")?;
            let material = required(inputs.material, "suit material")?;
            material.validate()?;
            let material_uv = required(inputs.material_uv, "suit material uv")?;
            validate_suit_uv(material_uv, pixel_count, "suit material uv")?;
            SuitPassKindInput::Image {
                material: Some(material),
                material_uv: Some(material_uv),
            }
        }
        "LipColor" => SuitPassKindInput::Color,
        _ => SuitPassKindInput::Unavailable("suit pass effect"),
    };

    let skip_shelter_query = plan.force_use_shelter == Some(true)
        || (plan.shelter_type.is_some() && plan.mask_field.is_none());
    let shelter = if skip_shelter_query {
        1.0
    } else {
        suit_weight(inputs.shelter, "suit shelter")?
    };
    let region = suit_weight(inputs.region, "suit region")?;

    Ok(SuitPassInput {
        kind,
        mask: Some(mask),
        mask_uv: Some(mask_uv),
        layer: Some(layer),
        shelter,
        region,
    })
}

/// Fail-fast 12-pass adapter. Plan and same-job arrays are positional and
/// ordered; the returned slice is accepted directly by `PortraitJobInputs`.
pub fn produce_suit_passes<'a>(
    pixel_count: usize,
    plans: Option<&[SuitPassPlan<'_>]>,
    inputs: Option<&'a [SuitPassSameJobInputs<'a>]>,
) -> Result<Vec<SuitPassInput<'a>>, Refusal> {
    let plans = required(plans, "suit passes")?;
    let inputs = required(inputs, "suit job inputs")?;
    if plans.is_empty() {
        return Err(Refusal::MissingInput("suit passes"));
    }
    if plans.len() != inputs.len() {
        return Err(Refusal::ShapeMismatch("suit passes"));
    }
    plans
        .iter()
        .zip(inputs)
        .enumerate()
        .map(|(index, (plan, inputs))| {
            if plan.order != index as u32 + 1 {
                return Err(Refusal::ValueOutOfRange("suit pass order"));
            }
            produce_suit_pass(pixel_count, plan, *inputs)
        })
        .collect()
}

/// One recovered makeup PF. `pc:<pf>` supplies `degree01`; every texture and
/// UV stream stays owned by the same job that produced the frame.
#[derive(Clone, Copy, Debug)]
pub struct MakeupInput<'a> {
    pub pf: u32,
    pub mix_color: [u8; 4],
    pub opacity_percent: f32,
    pub filter_type: u32,
    pub material: Option<&'a Image>,
    pub material_uv: Option<&'a [[f32; 2]]>,
    /// PF111 coverage; `material_uv[p]` is valid only when `writes[p] > 0`.
    pub material_uv_writes: Option<&'a [u16]>,
    pub mask: Option<&'a Image>,
    pub mask_uv: Option<&'a [[f32; 2]]>,
    /// Color-row coverage; `mask_uv[p]` is valid only when `writes[p] > 0`.
    pub mask_uv_writes: Option<&'a [u16]>,
}

/// PF30021 consumes explicit full-frame f32 source/filter/mask planes. The
/// current RGBA8 stage is used only to preserve alpha in the final byte store.
#[derive(Clone, Copy, Debug)]
pub struct HlfFaceInputs<'a> {
    pub width: u32,
    pub height: u32,
    pub source_rgb_full_f32: &'a [f32],
    pub median_rgb_full_f32: &'a [f32],
    pub gauss_rgb_full_f32: &'a [f32],
    pub face_mask_full_f32: &'a [f32],
    pub skin_mask_full_f32: &'a [f32],
    pub roi: Roi,
}

/// PF30022 keeps the current RGBA8 frame as source and requires full-frame
/// f32 median/gauss/mask planes. Only RGB inside `roi` is replaced.
#[derive(Clone, Copy, Debug)]
pub struct HlfBodyInputs<'a> {
    pub width: u32,
    pub height: u32,
    pub median_rgb_full_f32: &'a [f32],
    pub gauss_rgb_full_f32: &'a [f32],
    pub face_mask_rgba_full_f32: &'a [f32],
    pub skin_mask_full_f32: &'a [f32],
    pub roi: Roi,
    pub uniforms: BodyHlfUniforms,
    pub adjust_lut: Option<AdjustLut<'a>>,
}

/// PF21025/21069/21070/21071 share one full-frame wrinkle bundle. `base`,
/// `residual`, masks and ROI all come from the same renderer job.
#[derive(Clone, Copy, Debug)]
pub struct WrinkleInputs<'a> {
    pub base_full: &'a Image,
    pub residual_full: &'a Image,
    pub mask_rgba_full: &'a [u8],
    pub skin_mask_full: &'a [u8],
    pub roi: Roi,
    pub uniforms: WrinkleUniforms,
}

/// PF90021 same-job inputs. The host selects one direction; both LUTs remain
/// explicit so a missing selected asset is not mistaken for neutral.
#[derive(Clone, Copy, Debug)]
pub struct Pf90021Inputs<'a> {
    pub direction: Pf90021Direction,
    pub positive_lut: Option<&'a VolumeLut64>,
    pub negative_lut: Option<&'a VolumeLut64>,
    pub mask_r: Option<&'a [u8]>,
    pub runtime: Option<Pf90021RuntimeScalars>,
    pub same_job: Option<&'a Pf90021SameJobBinding>,
    pub direction_receipt: Option<&'a Pf90021DirectionReceipt>,
    pub runtime_receipt: Option<&'a Pf90021RuntimeReceipt>,
}

const FNV1A64_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV1A64_PRIME: u64 = 0x100000001b3;

fn fnv1a64(mut hash: u64, bytes: &[u8]) -> u64 {
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(FNV1A64_PRIME);
    }
    hash
}

fn fnv_usize(hash: u64, value: usize) -> u64 {
    fnv1a64(hash, &(value as u64).to_le_bytes())
}

fn fnv_u64(hash: u64, value: u64) -> u64 {
    fnv1a64(hash, &value.to_le_bytes())
}

fn smile_split_spec_fingerprint(mut hash: u64, spec: SmileHostSplitSpec) -> u64 {
    hash = fnv_u64(hash, spec.job_id);
    for axis in [spec.dx, spec.dy] {
        hash = fnv_u64(hash, axis.channel.index() as u64);
        hash = fnv_u64(
            hash,
            if matches!(axis.sign, SmileAxisSign::Negative) {
                1
            } else {
                0
            },
        );
        hash = fnv_u64(hash, axis.scale.to_bits() as u64);
        hash = fnv_u64(hash, axis.bias.to_bits() as u64);
    }
    for roi in [spec.model_roi, spec.frame_roi] {
        for value in [roi.x, roi.y, roi.width, roi.height] {
            hash = fnv_u64(hash, u64::from(value));
        }
    }
    hash
}

fn fnv_option_bytes(hash: u64, bytes: Option<&[u8]>) -> u64 {
    match bytes {
        None => fnv_u64(hash, 0),
        Some(bytes) => {
            let hash = fnv_u64(hash, 1);
            fnv1a64(fnv_usize(hash, bytes.len()), bytes)
        }
    }
}

fn fnv_option_f32_plane(hash: u64, values: Option<&[f32]>) -> u64 {
    match values {
        None => fnv_u64(hash, 0),
        Some(values) => {
            let mut hash = fnv_usize(fnv_u64(hash, 1), values.len());
            for value in values {
                hash = fnv_u64(hash, value.to_bits() as u64);
            }
            hash
        }
    }
}

fn fnv_option_f32(hash: u64, value: Option<f32>) -> u64 {
    match value {
        None => fnv_u64(hash, 0),
        Some(value) => fnv_u64(fnv_u64(hash, 1), value.to_bits() as u64),
    }
}

/// Producer-supplied fingerprint for one immutable portrait input bundle.
///
/// The fingerprint binds the pre-effect RGBA8 frame, dimensions, normalized
/// region, the complete parameter table and all four mask planes. A receipt has
/// no public constructor with a caller-chosen id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PortraitJobReceipt {
    fingerprint: u64,
    frame: PortraitFrameReceipt,
    width: u32,
    height: u32,
    frame_len: usize,
    region_bits: [u32; 4],
    mask_width: usize,
    mask_height: usize,
    plane_lengths: [Option<usize>; 4],
}

impl PortraitJobReceipt {
    /// Builds a receipt from the current renderer inputs before portrait effects.
    pub fn from_renderer_receipt(
        frame: &[u8],
        width: u32,
        height: u32,
        region: NormRect,
        params: &BTreeMap<String, f32>,
        masks: &PortraitMasks,
    ) -> Self {
        let frame_receipt = PortraitFrameReceipt::from_renderer_frame(frame, width, height)
            .unwrap_or(PortraitFrameReceipt {
                fingerprint: 0,
                width,
                height,
            });
        let region_bits = [
            region.x.to_bits(),
            region.y.to_bits(),
            region.width.to_bits(),
            region.height.to_bits(),
        ];
        let planes = [&masks.skin, &masks.hair, &masks.face, &masks.body];
        let plane_lengths = [
            masks.skin.as_ref().map(Vec::len),
            masks.hair.as_ref().map(Vec::len),
            masks.face.as_ref().map(Vec::len),
            masks.body.as_ref().map(Vec::len),
        ];

        let mut fingerprint = fnv1a64(FNV1A64_OFFSET_BASIS, b"aurora-portrait-receipt-v1");
        fingerprint = fnv_usize(fingerprint, frame.len());
        fingerprint = fnv1a64(fingerprint, frame);
        fingerprint = fnv_u64(fingerprint, width as u64);
        fingerprint = fnv_u64(fingerprint, height as u64);
        for bits in region_bits {
            fingerprint = fnv_u64(fingerprint, bits as u64);
        }
        fingerprint = fnv_usize(fingerprint, params.len());
        for (key, value) in params {
            fingerprint = fnv_usize(fingerprint, key.len());
            fingerprint = fnv1a64(fingerprint, key.as_bytes());
            fingerprint = fnv_u64(fingerprint, value.to_bits() as u64);
        }
        fingerprint = fnv_usize(fingerprint, masks.width);
        fingerprint = fnv_usize(fingerprint, masks.height);
        for (index, plane) in planes.into_iter().enumerate() {
            fingerprint = fnv_u64(fingerprint, index as u64);
            fingerprint = fnv_u64(fingerprint, plane_lengths[index].unwrap_or(usize::MAX) as u64);
            if let Some(plane) = plane {
                for value in plane {
                    fingerprint = fnv_u64(fingerprint, value.to_bits() as u64);
                }
            }
        }

        Self {
            fingerprint,
            frame: frame_receipt,
            width,
            height,
            frame_len: frame.len(),
            region_bits,
            mask_width: masks.width,
            mask_height: masks.height,
            plane_lengths,
        }
    }

    pub fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    pub fn job_id(&self) -> u64 {
        self.fingerprint
    }

    pub fn frame_receipt(&self) -> PortraitFrameReceipt {
        self.frame
    }

    fn refusal(
        &self,
        frame: &[u8],
        width: u32,
        height: u32,
        region: NormRect,
        params: &BTreeMap<String, f32>,
        masks: &PortraitMasks,
    ) -> Option<Refusal> {
        let expected_frame_len = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4));
        if Some(self.frame_len) != expected_frame_len
            || frame.len() != self.frame_len
            || self.width != width
            || self.height != height
            || self.region_bits
                != [
                    region.x.to_bits(),
                    region.y.to_bits(),
                    region.width.to_bits(),
                    region.height.to_bits(),
                ]
        {
            return Some(Refusal::ShapeMismatch("PortraitJobReceipt frame/region"));
        }

        let expected_plane_len = masks.width.checked_mul(masks.height);
        if self.mask_width != masks.width
            || self.mask_height != masks.height
            || self.plane_lengths
                != [
                    masks.skin.as_ref().map(Vec::len),
                    masks.hair.as_ref().map(Vec::len),
                    masks.face.as_ref().map(Vec::len),
                    masks.body.as_ref().map(Vec::len),
                ]
        {
            return Some(Refusal::ShapeMismatch("PortraitJobReceipt masks"));
        }
        for plane in [&masks.skin, &masks.hair, &masks.face, &masks.body] {
            if plane
                .as_ref()
                .is_some_and(|plane| Some(plane.len()) != expected_plane_len)
            {
                return Some(Refusal::ShapeMismatch("PortraitJobReceipt mask plane"));
            }
        }

        let actual = Self::from_renderer_receipt(frame, width, height, region, params, masks);
        (actual.fingerprint != self.fingerprint)
            .then_some(Refusal::ValueOutOfRange("PortraitJobReceipt digest"))
    }
}

/// Where a job's inputs came from. Only a renderer receipt or test-only
/// renderer job proves that absent fields are absent in the same job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortraitJobInputOrigin {
    MasksOnly,
    RendererJob { job_id: u64 },
}

/// Backend input container for effects whose producer is outside this stage.
pub struct PortraitJobInputs<'a> {
    pub origin: PortraitJobInputOrigin,
    pub receipt: Option<PortraitJobReceipt>,
    pub masks: &'a PortraitMasks,
    pub assets: PortraitAssetInputs<'a>,
    pub parsing: Option<&'a crate::face_parsing::FaceParsingProducerInputs>,
    pub teeth: Option<TeethInputs<'a>>,
    pub pf30025: Option<Pf30025Inputs<'a>>,
    pub teeth_alignment: Option<TeethAlignmentInputs<'a>>,
    pub radiance: Option<RadianceInputs<'a>>,
    pub beard: Option<BeardInputs<'a>>,
    pub moisture_face: Option<MoistureFaceInputs<'a>>,
    pub moisture_body: Option<MoistureBodyInputs<'a>>,
    pub eye_bag: Option<EyeBagInputs<'a>>,
    pub hair_smooth: Option<HairSmoothInputs<'a>>,
    pub hair_gloss: Option<HairGlossInputs<'a>>,
    pub stray_hair: StrayHairInputs<'a>,
    pub smile: SmileInputs<'a>,
    pub suit_passes: Option<&'a [SuitPassInput<'a>]>,
    pub makeup: Option<&'a [MakeupInput<'a>]>,
    pub makeup_bundle: Option<&'a OwnedMakeupBundle>,
    pub hlf_face: Option<HlfFaceInputs<'a>>,
    pub hlf_body: Option<HlfBodyInputs<'a>>,
    pub wrinkle: Option<WrinkleInputs<'a>>,
    pub wrinkle_producer: Option<&'a WrinkleProducerInputs>,
    pub pf90021: Option<Pf90021Inputs<'a>>,
    pub slim: Option<&'a SlimProducerInputs>,
    pub face_geometry: Option<&'a FaceGeometryInputs>,
    pub body_geometry: Option<&'a BodyGeometryJob>,
    pub repairs: Option<&'a RepairProducerInputs>,
    pub historical_tone: Option<&'a HistoricalToneInputs>,
    pub historical_skin: Option<&'a HistoricalSkinInputs>,
    pub plump: Option<&'a PlumpProducerInputs>,
    pub pf101: Option<&'a Pf101ProducerInputs>,
    pub pf90130: Option<&'a Pf90130ProducerInputs>,
}

impl<'a> PortraitJobInputs<'a> {
    pub fn new(masks: &'a PortraitMasks) -> Self {
        Self::masks_only(masks)
    }

    /// Legacy entry: parser masks only. It cannot execute portrait effects.
    pub fn masks_only(masks: &'a PortraitMasks) -> Self {
        Self {
            origin: PortraitJobInputOrigin::MasksOnly,
            receipt: None,
            masks,
            assets: PortraitAssetInputs::default(),
            parsing: None,
            teeth: None,
            pf30025: None,
            teeth_alignment: None,
            radiance: None,
            beard: None,
            moisture_face: None,
            moisture_body: None,
            eye_bag: None,
            hair_smooth: None,
            hair_gloss: None,
            stray_hair: StrayHairInputs::default(),
            smile: SmileInputs::default(),
            suit_passes: None,
            makeup: None,
            makeup_bundle: None,
            hlf_face: None,
            hlf_body: None,
            wrinkle: None,
            wrinkle_producer: None,
            pf90021: None,
            slim: None,
            face_geometry: None,
            body_geometry: None,
            repairs: None,
            historical_tone: None,
            historical_skin: None,
            plump: None,
            pf101: None,
            pf90130: None,
        }
    }

    /// Test-only legacy constructor with a caller-chosen label.
    #[cfg(test)]
    pub fn from_renderer_job(job_id: u64, masks: &'a PortraitMasks) -> Self {
        Self {
            origin: PortraitJobInputOrigin::RendererJob { job_id },
            ..Self::masks_only(masks)
        }
    }

    /// Production entry: the job id is the producer receipt's input fingerprint.
    pub fn from_renderer_receipt(
        receipt: PortraitJobReceipt,
        masks: &'a PortraitMasks,
    ) -> Self {
        Self {
            origin: PortraitJobInputOrigin::RendererJob {
                job_id: receipt.job_id(),
            },
            receipt: Some(receipt),
            ..Self::masks_only(masks)
        }
    }

    /// Test-only entry: keeps the receipt so the digest check still runs, but
    /// labels the job id independently so a stale-producer job id can be tested.
    #[cfg(test)]
    pub fn from_renderer_receipt_with_job_id(
        receipt: PortraitJobReceipt,
        masks: &'a PortraitMasks,
        job_id: u64,
    ) -> Self {
        Self {
            origin: PortraitJobInputOrigin::RendererJob { job_id },
            receipt: Some(receipt),
            ..Self::masks_only(masks)
        }
    }

    pub fn fill_from_producer_inputs(&mut self, producer: &'a ProducerInputs) {
        self.parsing = producer.parsing.as_ref();
        self.slim = producer.slim.as_ref();
        self.face_geometry = producer.face_geometry.as_ref();
        self.body_geometry = producer.body_geometry.as_ref();
        self.repairs = producer.repairs.as_ref();
        self.historical_tone = producer.historical_tone.as_ref();
        self.historical_skin = producer.historical_skin.as_ref();
        self.plump = producer.plump.as_ref();
        if let Some(teeth) = producer.teeth.as_ref() {
            self.teeth = Some(TeethInputs {
                job_id: teeth.job_id,
                mask_width: teeth.mask_width,
                mask_height: teeth.mask_height,
                mask_source: teeth.mask_source,
                mask_format: teeth.mask_format,
                mask_sha256: teeth.mask_sha256,
                whiten_lut_identity: teeth.whiten_lut_identity,
                bright_lut_identity: teeth.bright_lut_identity,
                mask_g: &teeth.mask_g,
                bright_ui: teeth.bright_ui,
                deyellow_ui: teeth.deyellow_ui,
            });
            self.assets.teeth_whiten_lut = Some(&teeth.whiten_lut);
            self.assets.teeth_bright_lut = Some(&teeth.bright_lut);
        }
        if let Some(smile) = producer.smile.as_ref() {
            let split = smile
                .channel_split
                .as_ref()
                .filter(|split| split.spec.job_id == smile.ticket);
            self.smile = SmileInputs {
                material: smile.material.as_deref(),
                lip_mask: smile.lip_mask.as_deref(),
                mouth_mask: smile.mouth_mask.as_deref(),
                model_dx: split.map(|split| split.model_dx.as_slice()),
                model_dy: split.map(|split| split.model_dy.as_slice()),
                roi: smile.roi.map(|(x, y, width, height)| Rect {
                    x,
                    y,
                    width,
                    height,
                }),
            };
        }
        if let Some(stray_hair) = producer.stray_hair.as_ref() {
            self.stray_hair = StrayHairInputs {
                hairfly_v1: stray_hair.hairfly_v1.as_ref(),
                raw_mask: stray_hair.raw_mask.as_deref(),
                person_mask: stray_hair.person_mask.as_deref(),
                background_mask: stray_hair.background_mask.as_deref(),
                native_degree: stray_hair.native_degree,
                native_person_area: stray_hair.native_person_area,
                native_background_area: stray_hair.native_background_area,
            };
        }
        if let Some(pf30025) = producer.pf30025.as_ref() {
            self.pf30025 = Some(Pf30025Inputs {
                receipt: pf30025.receipt,
                repair_780929437_bgra: &pf30025.repair_780929437_bgra,
                gate_3204548346: &pf30025.gate_3204548346,
                teeth_mask: &pf30025.teeth_mask,
                roi: pf30025.roi,
            });
        }
        if let Some(alignment) = producer.teeth_alignment.as_ref() {
            self.teeth_alignment = Some(TeethAlignmentInputs {
                effect_rgb: alignment.effect_rgb.as_ref(),
                teeth_mask: alignment.teeth_mask.as_deref(),
                edge_plane_a: alignment.edge_plane_a.as_deref(),
                edge_plane_b: alignment.edge_plane_b.as_deref(),
                gate_binding: alignment.gate_binding,
                roi: alignment.roi,
            });
        }
        if let Some(hair) = producer.hair_smooth.as_ref() {
            self.hair_smooth = Some(HairSmoothInputs {
                ticket: hair.ticket,
                width: hair.width,
                height: hair.height,
                roi: hair.roi,
                mix_strength: hair.mix_strength,
                hair_mask: Some(&hair.hair_mask),
                mix_mask: Some(&hair.mix_mask),
                blend_rgb: Some(&hair.blend_rgb),
                detail: Some(&hair.detail),
                hair_stride: hair.hair_stride,
                mix_stride: hair.mix_stride,
                blend_stride: hair.blend_stride,
                detail_stride: hair.detail_stride,
            });
        }
        if let Some(hair) = producer.hair_gloss.as_ref() {
            self.hair_gloss = Some(HairGlossInputs {
                ticket: hair.ticket,
                width: hair.width,
                height: hair.height,
                roi: hair.roi,
                dynamic_scale: hair.dynamic_scale,
                hair_mask: Some(&hair.hair_mask),
                model_detail: Some(&hair.model_detail),
                hair_stride: hair.hair_stride,
                detail_stride: hair.detail_stride,
            });
        }
        if let Some(eye_bag) = producer.eye_bag.as_ref() {
            self.eye_bag = Some(EyeBagInputs {
                ticket: eye_bag.ticket,
                model_id: eye_bag.model_id,
                color_units: eye_bag.color_units,
                geometry_class: eye_bag.geometry_class,
                geometry_receipt: eye_bag.geometry_receipt,
                width: eye_bag.frame_width,
                height: eye_bag.frame_height,
                aligned_color_field_rgb_f32: &eye_bag.aligned_color_field_rgb_f32,
                aligned_response_mask: &eye_bag.aligned_response_mask,
            });
        }
        if let Some(pf90021) = producer.pf90021.as_ref() {
            self.pf90021 = Some(Pf90021Inputs {
                direction: pf90021.direction,
                positive_lut: Some(&pf90021.positive_lut),
                negative_lut: Some(&pf90021.negative_lut),
                mask_r: Some(&pf90021.mask_r),
                runtime: Some(pf90021.runtime),
                same_job: Some(&pf90021.same_job),
                direction_receipt: Some(&pf90021.direction_receipt),
                runtime_receipt: Some(&pf90021.runtime_receipt),
            });
        }
        if let Some(pf90004) = producer.pf90004.as_ref() {
            self.radiance = Some(RadianceInputs {
                frame_width: Some(pf90004.width),
                frame_height: Some(pf90004.height),
                skin_mask: Some(&pf90004.skin_mask),
                parsing_rgba: Some(&pf90004.parsing_rgba),
                face_rgba: Some(&pf90004.face_rgba),
                value_mean: Some(pf90004.value_mean),
                roi: Some(pf90004.roi),
            });
        }
        if let (Some(body), Some(lut)) = (
            self.hlf_body.as_mut(),
            producer.hlf_body_adjust_lut.as_ref(),
        ) {
            body.adjust_lut = Some(AdjustLut {
                width: lut.width,
                height: lut.height,
                luma: &lut.luma,
            });
        }
        self.wrinkle_producer = producer.wrinkle.as_ref();
        self.makeup_bundle = producer.makeup_bundle.as_ref();
        self.pf101 = producer.pf101.as_ref();
        self.pf90130 = producer.pf90130.as_ref();
        // Borrow the owned same-frame planes back out of `producer`, which lives
        // for `'a` like `self`. A missing owned bundle stays `None`, so the
        // PF90002/PF90010 rows still refuse with their own MissingInput names.
        self.moisture_face = producer
            .moisture_face
            .as_ref()
            .map(|value| value.as_borrowed());
        self.moisture_body = producer
            .moisture_body
            .as_ref()
            .map(|value| value.as_borrowed());
    }

    fn resolve_wrinkle_inputs(
        &self,
        current: &Image,
    ) -> Result<WrinkleInputs<'a>, Refusal> {
        if let Some(input) = self.wrinkle {
            return Ok(input);
        }
        let producer = self.wrinkle_producer.ok_or(Refusal::MissingInput(
            "PF21025_PF21069_PF21070_PF21071_WRINKLE_SAME_JOB_INPUTS",
        ))?;
        let resolved = producer.resolve(current, self.job_id())?;
        Ok(WrinkleInputs {
            base_full: resolved.base_full,
            residual_full: resolved.residual_full,
            mask_rgba_full: resolved.mask_rgba_full,
            skin_mask_full: resolved.skin_mask_full,
            roi: resolved.roi,
            uniforms: resolved.uniforms,
        })
    }

    pub fn job_id(&self) -> Option<u64> {
        if let Some(receipt) = self.receipt {
            return Some(receipt.job_id());
        }
        match self.origin {
            PortraitJobInputOrigin::MasksOnly => None,
            PortraitJobInputOrigin::RendererJob { job_id } => Some(job_id),
        }
    }
}

fn sample_plane(plane: &[f32], width: usize, height: usize, x: f32, y: f32) -> f32 {
    if width == 0 || height == 0 {
        return 0.0;
    }
    let px = (x.clamp(0.0, 1.0) * width as f32 - 0.5).clamp(0.0, (width as f32 - 1.001).max(0.0));
    let py = (y.clamp(0.0, 1.0) * height as f32 - 0.5).clamp(0.0, (height as f32 - 1.001).max(0.0));
    let x0 = px as usize;
    let y0 = py as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = px - x0 as f32;
    let ty = py - y0 as f32;
    let at = |xx: usize, yy: usize| plane[yy * width + xx];
    let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
    let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
    top + (bottom - top) * ty
}

#[path = "portrait_repair_skin_stage.rs"]
mod repair_skin_stage;
pub use repair_skin_stage::{
    GlareChainFinalWritebackInputs, GlareFinalWritebackInputs, GlareFinalWritebackOutput,
    GlareInputs, GlarePlanarBundle, GlarePlanarInputs, GlarePlanarReceipts, GlareWritebackInputs,
    GlareWritebackOutput, PF90130_A_PLANAR_BYTES, PF90130_B_PLANAR_BYTES,
    PF90130_PIXELS, PF90130_PLANE_HEIGHT, PF90130_PLANE_WIDTH, SkinRepairData, SkinRepairOutput,
    SkinRepairProducerInputs, SpotlessGainmapJob, SpotlessInputs, apply_glare_chain_final_writeback,
    apply_glare_final_writeback, apply_glare_stage_writeback, bridge_glare_planes,
    resolve_spotless_gainmap_amount, run as run_skin_repair, run_spotless_gainmap_16u,
};

#[path = "portrait_pf90130_producer.rs"]
mod pf90130_producer;
pub use pf90130_producer::{
    PF90130_EFFECT_TENSOR_ID, PF90130_INPUT_TENSOR_ID, PF90130_MASK_TENSOR_ID, PF90130_MODEL_ID,
    PF90130_NORMALIZE_SCALE, PF90130_NORMALIZE_SUB, Pf90130AlignedGeometry, Pf90130GlareProducer,
    Pf90130SameJobReceipt,
};

pub(crate) fn sha256_32(bytes: &[u8]) -> [u8; 32] {
    aurora_portrait::model_sha256::sha256(bytes)
}

/// Effects that can run from the frame, a mask and bundled LUT bytes today.
const SUPPORTED_EFFECTS: &[u32] = &[
    101,    // Silkworm material
    108,    // Teeth whitening
    30025,  // Tooth repair
    30026,  // Teeth alignment final blend
    21025,  // Puppet wrinkle
    21069,  // Cheek wrinkle
    21070,  // Nose-back wrinkle
    21071,  // Perioral wrinkle
    8068,   // Skin gloss
    8069,   // Skin texture (negative surface-blur branch)
    8070,   // Skin clarity
    21015,  // Skin color hue
    21026,  // Stubble removal
    90021,  // Lip makeup volume LUT
    30096,  // Skin temperature
    90002,  // Moisture smooth (face)
    90004,  // Skin radiance
    90005,  // Skin whitening
    90007,  // Skin ruddy
    90010,  // Moisture smooth (body)
    90033,  // Suit makeup ordered passes
    90116,  // Eye bags
    90139,  // Hair smooth
    90160,  // Hair gloss
    90088,  // Stray hair removal
    90130,  // Glasses glare removal
    30013,  // Neutral grey flattening
    30021,  // High-low frequency face
    30022,  // High-low frequency body
    100007, // Open-mouth smile
    100,    // Eyebrow color makeup, left
    103,    // Upper eyeliner color makeup
    105,    // Upper eyelash color makeup
    111,    // Eye pupil image makeup, no mask
    117,    // Eyebrow color makeup, right
    118,    // Lower eyelash color makeup
    30007,  // Lower eyeliner color makeup
];

/// Effect ids the Windows bridge can currently run from a frame, a mask and
/// bundled LUT bytes. This is the read-only capability registry; it does not
/// imply automatic landmark or model generation.
pub fn supported_effects() -> &'static [u32] {
    SUPPORTED_EFFECTS
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FaceMeshGroup {
    left: u32,
    right: u32,
}

impl FaceMeshGroup {
    fn contains(self, pf: u32) -> bool {
        pf == self.left || pf == self.right
    }
}

fn face_mesh_group(pf: u32) -> Option<FaceMeshGroup> {
    match pf {
        30092 | 30093 => Some(FaceMeshGroup {
            left: 30092,
            right: 30093,
        }),
        30080 | 30081 => Some(FaceMeshGroup {
            left: 30080,
            right: 30081,
        }),
        30082 | 30083 => Some(FaceMeshGroup {
            left: 30082,
            right: 30083,
        }),
        242 | 243 => Some(FaceMeshGroup {
            left: 242,
            right: 243,
        }),
        8023 | 8024 => Some(FaceMeshGroup {
            left: 8023,
            right: 8024,
        }),
        8027 | 8028 => Some(FaceMeshGroup {
            left: 8027,
            right: 8028,
        }),
        8029 | 8030 => Some(FaceMeshGroup {
            left: 8029,
            right: 8030,
        }),
        _ => None,
    }
}

const FACE_WIDTH_PARTS: &[u8] =
    include_bytes!("../../portrait/assets/portrait/face-width/parts.f32le");
const FOREHEAD_PARTS: &[u8] =
    include_bytes!("../../portrait/assets/portrait/forehead/parts.f32le");
const FACE_RESHAPE_PARTS: &[u8] =
    include_bytes!("../../portrait/assets/portrait/face-reshape/parts.f32le");

struct FaceReshapePartData {
    part: u8,
    parameters: Vec<f32>,
    factors: Vec<f32>,
}

fn builtin_face_reshape_parts() -> Result<Vec<FaceReshapePartData>, Refusal> {
    let bytes = FACE_RESHAPE_PARTS;
    if bytes.len() < 4 {
        return Err(Refusal::ShapeMismatch("face reshape parts header"));
    }
    let count = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut offset = 4usize;
    let mut parts = Vec::with_capacity(count);
    for _ in 0..count {
        if offset + 8 > bytes.len() {
            return Err(Refusal::ShapeMismatch("face reshape part header"));
        }
        let part = u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let len = u32::from_le_bytes(bytes[offset + 4..offset + 8].try_into().unwrap()) as usize;
        offset += 8;
        let need = len.checked_mul(8).ok_or(Refusal::ShapeMismatch("face reshape len"))?;
        if offset + need > bytes.len() || part > 255 {
            return Err(Refusal::ShapeMismatch("face reshape part body"));
        }
        let mut parameters = Vec::with_capacity(len);
        let mut factors = Vec::with_capacity(len);
        for i in 0..len {
            let p = f32::from_le_bytes(bytes[offset + i * 4..offset + (i + 1) * 4].try_into().unwrap());
            let f = f32::from_le_bytes(
                bytes[offset + len * 4 + i * 4..offset + len * 4 + (i + 1) * 4]
                    .try_into()
                    .unwrap(),
            );
            if !p.is_finite() || !f.is_finite() {
                return Err(Refusal::ValueOutOfRange("face reshape part values"));
            }
            parameters.push(p);
            factors.push(f);
        }
        offset += need;
        parts.push(FaceReshapePartData {
            part: part as u8,
            parameters,
            factors,
        });
    }
    Ok(parts)
}

fn radial_mesh_axes(parts: &[&face_geometry::PartGeometry], width: f32, height: f32) -> [Vec<f32>; 2] {
    let axis = |extent: f32| (0..=40).map(|i| i as f32 / 40.0 * extent).collect::<Vec<_>>();
    let [mut xs, mut ys] = [axis(width), axis(height)];
    for part in parts {
        for delta in &part.deltas {
            let radius = (delta.target.x - delta.source.x).hypot(delta.target.y - delta.source.y);
            if radius > 0.0 {
                for fraction in [-1.0, -0.5, 0.0, 0.5, 1.0] {
                    xs.push((delta.source.x + fraction * radius).clamp(0.0, width));
                    ys.push((delta.source.y + fraction * radius).clamp(0.0, height));
                }
            }
        }
    }
    for axis in [&mut xs, &mut ys] {
        axis.sort_unstable_by(f32::total_cmp);
        axis.dedup_by(|left, right| (*left - *right).abs() < 1.0e-4);
    }
    [xs, ys]
}

fn rasterize_radial_warps(source: &Image, parts: &[&face_geometry::PartGeometry]) -> FaceMeshStageResult {
    let w = source.width as f32;
    let h = source.height as f32;
    let [xs, ys] = radial_mesh_axes(parts, w, h);
    let verts = xs.len() * ys.len();
    if verts > usize::from(u16::MAX) + 1 {
        return face_mesh_refused(Refusal::ValueOutOfRange("face reshape radial grid"), None);
    }
    let mut dest = Vec::with_capacity(verts);
    let mut uv = Vec::with_capacity(verts);
    for (gy, &y) in ys.iter().enumerate() {
        for (gx, &x) in xs.iter().enumerate() {
            uv.push([x / w, y / h]);
            if gx == 0 || gx + 1 == xs.len() || gy == 0 || gy + 1 == ys.len() {
                dest.push([x, y]);
            } else {
                let mut pt = face_geometry::Point::new(x, y);
                for warp in parts.iter().filter_map(|part| part.warp.as_ref()) {
                    pt = warp.apply(pt);
                }
                dest.push([pt.x.clamp(0.0, w), pt.y.clamp(0.0, h)]);
            }
        }
    }
    let mut ebo = Vec::with_capacity((xs.len() - 1) * (ys.len() - 1) * 6);
    for gy in 0..ys.len() - 1 {
        for gx in 0..xs.len() - 1 {
            let i00 = (gy * xs.len() + gx) as u16;
            let i10 = i00 + 1;
            let i01 = i00 + xs.len() as u16;
            let i11 = i01 + 1;
            ebo.extend_from_slice(&[i00, i10, i11, i00, i11, i01]);
        }
    }
    let mesh = aurora_portrait::raster::Mesh {
        dest: &dest,
        source_uv: &uv,
        ebo: &ebo,
    };
    let rules = aurora_portrait::raster::Rules {
        winding: aurora_portrait::raster::Winding::PositiveAreaIsFront,
        cull: aurora_portrait::raster::Cull::None,
        edge: aurora_portrait::raster::Edge::Inclusive,
        overlap: aurora_portrait::raster::Overlap::Last,
    };
    let sampler = aurora_portrait::raster::Sampler {
        wrap: aurora_portrait::raster::Wrap::ClampToEdge,
    };
    let edge = aurora_portrait::raster_fragment::Edge::ReshapeNormal;
    let defines = aurora_portrait::raster_fragment::Defines {
        clamp_edge: true,
        uv_map: false,
        uv_map_ba: false,
        catmull_rom: false,
    };
    let clamp = aurora_portrait::raster_fragment::Clamp::disabled();

    #[cfg(windows)]
    {
        let cuda = match forced_cuda_failure() {
            Some(error) => Err(error),
            None => MeshRaster::open().and_then(|gpu| {
                gpu.rasterize_fragment(
                    source,
                    mesh,
                    rules,
                    sampler,
                    edge,
                    defines,
                    clamp,
                    aurora_portrait::raster::Tolerance::RESHAPE,
                )
            }),
        };
        if let Ok(output) = cuda {
            return FaceMeshStageResult {
                image: (output.image != *source).then_some(output.image),
                backend: PortraitBackend::Cuda,
                dispatches: vec![output.select_dispatch, output.emit_dispatch],
                fallback_reason: None,
                refusal: None,
            };
        }
    }

    match aurora_portrait::raster::rasterize_with_fragment(
        source,
        mesh,
        rules,
        sampler,
        edge,
        defines,
        clamp,
        aurora_portrait::raster::Tolerance::RESHAPE,
    ) {
        Ok(image) => FaceMeshStageResult {
            image: (image != *source).then_some(image),
            backend: PortraitBackend::Cpu,
            dispatches: Vec::new(),
            fallback_reason: None,
            refusal: None,
        },
        Err(refusal) => face_mesh_refused(refusal, None),
    }
}

struct ForeheadPartData {
    part: u8,
    weights: [f32; 135],
}

fn builtin_forehead_parts() -> Result<Vec<ForeheadPartData>, Refusal> {
    let mut bytes = FOREHEAD_PARTS;
    let mut parts = Vec::with_capacity(4);
    while !bytes.is_empty() {
        if bytes.len() < 544 {
            return Err(Refusal::ShapeMismatch("forehead parts asset"));
        }
        let read =
            |offset: usize| f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let part = read(0);
        if !part.is_finite() || part.fract() != 0.0 || !(35.0..=38.0).contains(&part) {
            return Err(Refusal::ValueOutOfRange("forehead part id"));
        }
        let mut weights = [0.0; 135];
        for index in 0..135 {
            weights[index] = read(4 + index * 4);
        }
        if weights.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("forehead part values"));
        }
        parts.push(ForeheadPartData {
            part: part as u8,
            weights,
        });
        bytes = &bytes[544..];
    }
    if parts.len() != 4 {
        return Err(Refusal::ShapeMismatch("forehead parts count"));
    }
    Ok(parts)
}

struct FaceWidthPartData {
    part: u8,
    parameters: [f32; 27],
    factors: [f32; 27],
}

fn builtin_face_width_parts() -> Result<Vec<FaceWidthPartData>, Refusal> {
    let mut bytes = FACE_WIDTH_PARTS;
    let mut parts = Vec::with_capacity(4);
    while !bytes.is_empty() {
        if bytes.len() < 220 {
            return Err(Refusal::ShapeMismatch("face width parts asset"));
        }
        let read =
            |offset: usize| f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let part = read(0);
        if !part.is_finite() || part.fract() != 0.0 || !(25.0..=28.0).contains(&part) {
            return Err(Refusal::ValueOutOfRange("face width part id"));
        }
        let mut parameters = [0.0; 27];
        let mut factors = [0.0; 27];
        for index in 0..27 {
            parameters[index] = read(4 + index * 4);
            factors[index] = read(112 + index * 4);
        }
        if parameters
            .iter()
            .chain(&factors)
            .any(|value| !value.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("face width part values"));
        }
        parts.push(FaceWidthPartData {
            part: part as u8,
            parameters,
            factors,
        });
        bytes = &bytes[220..];
    }
    if parts.len() != 4 {
        return Err(Refusal::ShapeMismatch("face width parts count"));
    }
    Ok(parts)
}

#[derive(Clone)]
struct FaceMeshStageResult {
    image: Option<Image>,
    backend: PortraitBackend,
    dispatches: Vec<Dispatch>,
    fallback_reason: Option<String>,
    refusal: Option<Refusal>,
}

fn face_mesh_refused(refusal: Refusal, fallback_reason: Option<String>) -> FaceMeshStageResult {
    FaceMeshStageResult {
        image: None,
        backend: PortraitBackend::Skipped,
        dispatches: Vec::new(),
        fallback_reason,
        refusal: Some(refusal),
    }
}

fn apply_face_mesh_group(
    source: &Image,
    landmark_frame_digest: Option<[u8; 32]>,
    region: NormRect,
    job_id: Option<u64>,
    slim: Option<&SlimProducerInputs>,
    group: FaceMeshGroup,
    controls: [f32; 2],
) -> FaceMeshStageResult {
    if controls
        .iter()
        .any(|value| !value.is_finite() || !(-100.0..=100.0).contains(value))
    {
        return face_mesh_refused(Refusal::ValueOutOfRange("face mesh UI"), None);
    }
    if controls == [0.0; 2] {
        return FaceMeshStageResult {
            image: None,
            backend: PortraitBackend::Skipped,
            dispatches: Vec::new(),
            fallback_reason: None,
            refusal: None,
        };
    }

    let Some(slim) = slim else {
        return face_mesh_refused(Refusal::MissingInput("same-job face mesh"), None);
    };
    if let Err(refusal) = slim.validate_ticket(source, region, job_id) {
        return face_mesh_refused(refusal, None);
    }
    if Some(slim.source_sha256) != landmark_frame_digest {
        return face_mesh_refused(Refusal::ValueOutOfRange("face mesh frame digest"), None);
    }
    let points414 = match face_width_414::produce(&slim.points194, 0.0) {
        Ok(points) => points,
        Err(face_width_414::Error::Points194) => {
            return face_mesh_refused(Refusal::ShapeMismatch("face width 194 landmarks"), None)
        }
        Err(face_width_414::Error::NonFinite) => {
            return face_mesh_refused(Refusal::ValueOutOfRange("face width landmarks"), None)
        }
        Err(face_width_414::Error::UnsupportedScale) => {
            return face_mesh_refused(
                Refusal::ValueOutOfRange("face width producer scale"),
                None,
            )
        }
    };
    if matches!(group.left, 242 | 8023 | 8027 | 8029) {
        let points: Vec<face_geometry::Point> = points414
            .iter()
            .map(|point| face_geometry::Point::new(point[0], point[1]))
            .collect();
        let center = points[55];
        let owned_parts = match builtin_face_reshape_parts() {
            Ok(parts) => parts,
            Err(refusal) => return face_mesh_refused(refusal, None),
        };
        let arrays: Vec<_> = owned_parts
            .iter()
            .map(|part| face_geometry::PartArrays {
                part: part.part,
                parameters: &part.parameters,
                factors: &part.factors,
            })
            .collect();
        let vendor = face_geometry::VendorParts { parts: &arrays };
        let geometry = match group.left {
            242 => face_geometry::sharp_chin_sides(
                controls, &points[..33], center, &vendor, source.width, source.height,
            ),
            8023 => face_geometry::jaw(
                controls[0],
                controls[1],
                &points[..33],
                center,
                &vendor,
                source.width,
                source.height,
            ),
            8027 => face_geometry::cheek_sides(
                &points[..33], center, controls, &vendor, source.width, source.height,
            ),
            8029 => face_geometry::temple(
                &points,
                controls[0],
                controls[1],
                &vendor,
                source.width,
                source.height,
            ),
            _ => unreachable!(),
        };
        let geometry = match geometry {
            Ok(g) => g,
            Err(refusal) => return face_mesh_refused(refusal, None),
        };
        let parts: Vec<_> = geometry
            .parts
            .iter()
            .filter(|part| part.warp.is_some())
            .collect();
        if parts.is_empty() {
            return face_mesh_refused(Refusal::MissingInput("face reshape radial warp"), None);
        }
        return rasterize_radial_warps(source, &parts);
    }

    if matches!(group.left, 30080 | 30082) {
        let fh_points: Vec<forehead_common::Point> = points414
            .iter()
            .map(|point| forehead_common::Point::new(point[0], point[1]))
            .collect();
        let fh_parts = match builtin_forehead_parts() {
            Ok(parts) => parts,
            Err(refusal) => return face_mesh_refused(refusal, None),
        };
        let native = [
            (controls[0] + 100.0) / 200.0,
            (controls[1] + 100.0) / 200.0,
        ];
        let part_ids = if group.left == 30080 {
            &face_mesh::FOREHEAD_WIDTH_PARTS
        } else {
            &face_mesh::FOREHEAD_HEIGHT_PARTS
        };
        let mut current_img = source.clone();
        let mut dispatches = Vec::new();
        let mut used_cuda = false;
        let rules = aurora_portrait::raster::Rules {
            winding: aurora_portrait::raster::Winding::PositiveAreaIsFront,
            cull: aurora_portrait::raster::Cull::None,
            edge: aurora_portrait::raster::Edge::Inclusive,
            overlap: aurora_portrait::raster::Overlap::Last,
        };
        let sampler = aurora_portrait::raster::Sampler {
            wrap: aurora_portrait::raster::Wrap::ClampToEdge,
        };
        let edge = aurora_portrait::raster_fragment::Edge::ReshapeNormal;
        let defines = aurora_portrait::raster_fragment::Defines {
            clamp_edge: true,
            uv_map: false,
            uv_map_ba: false,
            catmull_rom: false,
        };
        let clamp = aurora_portrait::raster_fragment::Clamp::disabled();

        for &part in part_ids {
            let Some(part_data) = fh_parts.iter().find(|item| item.part == part) else {
                return face_mesh_refused(Refusal::MissingInput("forehead part weights"), None);
            };
            let direction = if part % 2 == 1 {
                forehead_common::Direction::Negative
            } else {
                forehead_common::Direction::Positive
            };
            let mesh_out = match forehead_common::build(&forehead_common::Inputs {
                points414: &fh_points,
                weights135: &part_data.weights,
                direction,
                native,
                width: source.width,
                height: source.height,
            }) {
                Ok(mesh) => mesh,
                Err(_) => {
                    return face_mesh_refused(
                        Refusal::ValueOutOfRange("forehead common build"),
                        None,
                    )
                }
            };
            let dest: Vec<[f32; 2]> = mesh_out.b.iter().map(|p| [p.x, p.y]).collect();
            let uv: Vec<[f32; 2]> = mesh_out
                .a
                .iter()
                .map(|p| [p.x / source.width as f32, p.y / source.height as f32])
                .collect();
            let r_mesh = aurora_portrait::raster::Mesh {
                dest: &dest,
                source_uv: &uv,
                ebo: &mesh_out.initial_indices,
            };
            let mut stepped = None;
            #[cfg(windows)]
            {
                let cuda = match forced_cuda_failure() {
                    Some(error) => Err(error),
                    None => MeshRaster::open().and_then(|gpu| {
                        gpu.rasterize_fragment(
                            &current_img,
                            r_mesh,
                            rules,
                            sampler,
                            edge,
                            defines,
                            clamp,
                            aurora_portrait::raster::Tolerance::RESHAPE,
                        )
                    }),
                };
                if let Ok(output) = cuda {
                    current_img = output.image;
                    dispatches.push(output.select_dispatch);
                    dispatches.push(output.emit_dispatch);
                    used_cuda = true;
                    stepped = Some(());
                }
            }
            if stepped.is_none() {
                match aurora_portrait::raster::rasterize_with_fragment(
                    &current_img,
                    r_mesh,
                    rules,
                    sampler,
                    edge,
                    defines,
                    clamp,
                    aurora_portrait::raster::Tolerance::RESHAPE,
                ) {
                    Ok(img) => current_img = img,
                    Err(refusal) => return face_mesh_refused(refusal, None),
                }
            }
        }
        return FaceMeshStageResult {
            image: (current_img != *source).then_some(current_img),
            backend: if used_cuda {
                PortraitBackend::Cuda
            } else {
                PortraitBackend::Cpu
            },
            dispatches,
            fallback_reason: None,
            refusal: None,
        };
    }

    let digest = sha256_32(&source.pixels);
    let mut selected = Vec::new();
    {
        let points: Vec<face_geometry::Point> = points414
            .iter()
            .map(|point| face_geometry::Point::new(point[0], point[1]))
            .collect();
        let owned_parts = match builtin_face_width_parts() {
            Ok(parts) => parts,
            Err(refusal) => return face_mesh_refused(refusal, None),
        };
        let arrays: Vec<_> = owned_parts
            .iter()
            .map(|part| face_geometry::PartArrays {
                part: part.part,
                parameters: &part.parameters,
                factors: &part.factors,
            })
            .collect();
        let vendor = face_geometry::VendorParts { parts: &arrays };
        let left = match face_geometry::face_width(
            controls[0],
            &points,
            &vendor,
            source.width,
            source.height,
        ) {
            Ok(geometry) => geometry,
            Err(refusal) => return face_mesh_refused(refusal, None),
        };
        let right = match face_geometry::face_width(
            controls[1],
            &points,
            &vendor,
            source.width,
            source.height,
        ) {
            Ok(geometry) => geometry,
            Err(refusal) => return face_mesh_refused(refusal, None),
        };

        for &part in &face_mesh::FACE_WIDTH_PARTS {
            let geometry = if matches!(part, 25 | 27) {
                &left
            } else {
                &right
            };
            let Some(part_geometry) = geometry.parts.iter().find(|item| item.part == part) else {
                return face_mesh_refused(Refusal::MissingInput("face mesh draw part"), None);
            };
            let draw = match face_mesh::Draw::from_face_geometry(controls, digest, part_geometry) {
                Ok(draw) => draw,
                Err(refusal) => return face_mesh_refused(refusal, None),
            };
            if !group.contains(draw.pf) {
                return face_mesh_refused(Refusal::ShapeMismatch("face mesh part group"), None);
            }
            selected.push(draw);
        }
    }
    let selected_refs = selected.iter().collect::<Vec<_>>();

    #[cfg(windows)]
    {
        let cuda = match forced_cuda_failure() {
            Some(error) => Err(error),
            None => {
                MeshRaster::open().and_then(|gpu| {
                    face_mesh::rasterize_cuda_with(&gpu, source, &selected_refs)
                })
            }
        };
        return match cuda {
            Ok(output) if !output.dispatches.is_empty() => FaceMeshStageResult {
                image: (output.image != *source).then_some(output.image),
                backend: PortraitBackend::Cuda,
                dispatches: output.dispatches,
                fallback_reason: None,
                refusal: None,
            },
            Ok(_) => {
                let cuda_fallback = "CUDA face mesh returned no dispatch".to_string();
                match face_mesh::rasterize(source, &selected_refs) {
                    Ok(output) => FaceMeshStageResult {
                        image: (output != *source).then_some(output),
                        backend: PortraitBackend::Cpu,
                        dispatches: Vec::new(),
                        fallback_reason: Some(cuda_fallback),
                        refusal: None,
                    },
                    Err(refusal) => face_mesh_refused(
                        refusal.clone(),
                        Some(format!("{cuda_fallback}; CPU {refusal:?}")),
                    ),
                }
            }
            Err(error) => {
                let cuda_fallback = format!("CUDA face mesh fallback: {error:?}");
                match face_mesh::rasterize(source, &selected_refs) {
                    Ok(output) => FaceMeshStageResult {
                        image: (output != *source).then_some(output),
                        backend: PortraitBackend::Cpu,
                        dispatches: Vec::new(),
                        fallback_reason: Some(cuda_fallback),
                        refusal: None,
                    },
                    Err(refusal) => face_mesh_refused(
                        refusal.clone(),
                        Some(format!("{cuda_fallback}; CPU {refusal:?}")),
                    ),
                }
            }
        };
    }
    #[cfg(not(windows))]
    match face_mesh::rasterize(source, &selected_refs) {
        Ok(output) => FaceMeshStageResult {
            image: (output != *source).then_some(output),
            backend: PortraitBackend::Cpu,
            dispatches: Vec::new(),
            fallback_reason: None,
            refusal: None,
        },
        Err(refusal) => face_mesh_refused(refusal, None),
    }
}

fn supports_effect(pf: u32) -> bool {
    SUPPORTED_EFFECTS.contains(&pf)
        || B3_MAKEUP_ROWS.contains(&pf)
        || matches!(pf, 8025 | 8026)
        || portrait_face_stage::pair(pf).is_some()
        || face_mesh_group(pf).is_some()
        || portrait_body_stage::supports(pf)
        || portrait_plump_stage::supports(pf)
        || repair_stage::PFS.contains(&pf)
        || portrait_historical_tone::supports(pf)
        || portrait_historical_skin::PFS.contains(&pf)
}

#[cfg(windows)]
fn try_cuda_body_mesh(
    source: &Image,
    pf: u32,
    job_id: Option<u64>,
    job: Option<&BodyGeometryJob>,
) -> Option<Result<(Image, Vec<aurora_portrait::cuda::Dispatch>), String>> {
    let job = job?;
    if !matches!(pf, 401 | 406 | 410 | 415) {
        return None;
    }
    if job.job_id == 0 || job_id != Some(job.job_id) {
        return Some(Err("CUDA body mesh: body geometry job id".to_string()));
    }
    if job.frame_size != (source.width, source.height)
        || job.source_sha256 != sha256_32(&source.pixels)
    {
        return Some(Err("CUDA body mesh: body geometry frame".to_string()));
    }
    let Some(BodyGeometryInput::Mesh { mesh, passes }) = job.effects.get(&pf) else {
        return Some(Err("CUDA body mesh: body geometry effect".to_string()));
    };
    if passes.is_empty() || passes.iter().any(|pass| pass.defines.uv_map) {
        return Some(Err("CUDA body mesh: color passes".to_string()));
    }
    if let Some(error) = forced_cuda_failure() {
        return Some(Err(format!("CUDA body mesh fallback: {error:?}")));
    }
    let gpu = match aurora_portrait::cuda_mesh::MeshRaster::open() {
        Ok(gpu) => gpu,
        Err(error) => return Some(Err(format!("CUDA body mesh fallback: {error:?}"))),
    };
    let mut current = source.clone();
    let mut dispatches = Vec::new();
    for pass in passes {
        let mut dest = Vec::with_capacity(mesh.destination.len());
        for position in &mesh.destination {
            let m = &pass.matrix;
            let clip_x = m[0] * position.x + m[1] * position.y + m[3];
            let clip_y = m[4] * position.x + m[5] * position.y + m[7];
            let w = m[12] * position.x + m[13] * position.y + m[15];
            if !w.is_finite() || w.abs() < 1e-8 {
                return Some(Err("CUDA body mesh: invalid clip W".to_string()));
            }
            dest.push([
                ((clip_x / w) + 1.0) * 0.5 * source.width as f32,
                ((clip_y / w) + 1.0) * 0.5 * source.height as f32,
            ]);
        }
        let uv = mesh
            .source_uv
            .iter()
            .map(|point| [point.x, point.y])
            .collect::<Vec<_>>();
        let raster_mesh = aurora_portrait::raster::Mesh {
            dest: &dest,
            source_uv: &uv,
            ebo: &mesh.indices,
        };
        let edge = match pass.edge {
            aurora_portrait::kernels::body_geometry::EdgeMode::ReshapeNormal => {
                aurora_portrait::raster_fragment::Edge::ReshapeNormal
            }
            aurora_portrait::kernels::body_geometry::EdgeMode::AdjClampEdge => {
                aurora_portrait::raster_fragment::Edge::AdjClampEdge
            }
        };
        let defines = aurora_portrait::raster_fragment::Defines {
            clamp_edge: pass.defines.clamp_edge,
            uv_map: pass.defines.uv_map,
            uv_map_ba: pass.defines.uv_map_ba,
            catmull_rom: pass.defines.catmull_rom,
        };
        let clamp = aurora_portrait::raster_fragment::Clamp {
            factor: pass.clamp.factor,
            protect_x: pass.clamp.protect_x,
            protect_y: pass.clamp.protect_y,
        };
        let output = match gpu.rasterize_fragment(
            &current,
            raster_mesh,
            aurora_portrait::raster::Rules {
                winding: aurora_portrait::raster::Winding::PositiveAreaIsFront,
                cull: aurora_portrait::raster::Cull::None,
                edge: aurora_portrait::raster::Edge::Inclusive,
                overlap: aurora_portrait::raster::Overlap::Last,
            },
            aurora_portrait::raster::Sampler {
                wrap: aurora_portrait::raster::Wrap::ClampToEdge,
            },
            edge,
            defines,
            clamp,
            aurora_portrait::raster::Tolerance::RESHAPE,
        ) {
            Ok(output) => output,
            Err(error) => return Some(Err(format!("CUDA body mesh fallback: {error:?}"))),
        };
        dispatches.push(output.select_dispatch);
        dispatches.push(output.emit_dispatch);
        current = output.image;
    }
    Some(Ok((current, dispatches)))
}

fn log_report(report: &PortraitRunReport) {
    for effect in &report.effects {
        if let Some(refusal) = &effect.refusal {
            log::debug!("portrait PF{} refused: {refusal:?}", effect.pf);
            continue;
        }
        match effect.backend {
            PortraitBackend::Cuda => log::debug!(
                "portrait PF{} executed on CUDA: {:?}",
                effect.pf,
                effect
                    .dispatches
                    .iter()
                    .map(|dispatch| dispatch.kernel)
                    .collect::<Vec<_>>()
            ),
            PortraitBackend::Cpu => {
                match &effect.fallback_reason {
                    Some(reason) => log::debug!("portrait PF{} CPU fallback: {reason}", effect.pf),
                    None => log::debug!("portrait PF{} executed on CPU", effect.pf),
                }
            }
            PortraitBackend::Skipped => log::debug!(
                "portrait PF{} skipped: {:?} / {:?}",
                effect.pf,
                effect.skip_reason,
                effect.fallback_reason
            ),
        }
    }
}

/// Applies every portrait parameter whose effect has a CPU stage and whose
/// same-job inputs are available. Returns the image with effects applied in
/// ascending ParamFlag order.
pub fn apply_portrait_effects(
    portrait: &mut Vec<u8>,
    width: u32,
    height: u32,
    region: NormRect,
    extra: &BTreeMap<String, f32>,
    masks: &PortraitMasks,
) {
    let report = apply_portrait_effects_with_report(
        portrait, width, height, region, extra, masks,
    );
    log_report(&report);
}

/// Production-style entry point for callers that already own the complete
/// same-job bundle.
pub fn apply_portrait_effects_with_job(
    portrait: &mut Vec<u8>,
    width: u32,
    height: u32,
    region: NormRect,
    extra: &BTreeMap<String, f32>,
    inputs: &PortraitJobInputs<'_>,
) -> PortraitRunReport {
    let report = apply_portrait_effects_with_report_and_inputs(
        portrait, width, height, region, extra, inputs,
    );
    log_report(&report);
    report
}

pub fn apply_portrait_effects_with_report(
    portrait: &mut Vec<u8>,
    width: u32,
    height: u32,
    region: NormRect,
    extra: &BTreeMap<String, f32>,
    masks: &PortraitMasks,
) -> PortraitRunReport {
    apply_portrait_effects_with_report_and_inputs(
        portrait,
        width,
        height,
        region,
        extra,
        &PortraitJobInputs::new(masks),
    )
}

pub fn apply_portrait_effects_with_report_and_inputs(
    portrait: &mut Vec<u8>,
    width: u32,
    height: u32,
    region: NormRect,
    extra: &BTreeMap<String, f32>,
    inputs: &PortraitJobInputs<'_>,
) -> PortraitRunReport {
    apply_portrait_effects_with_report_and_inputs_policy(
        portrait,
        width,
        height,
        region,
        extra,
        inputs,
        cfg!(not(test)),
    )
}

fn apply_portrait_effects_with_report_and_inputs_policy(
    portrait: &mut Vec<u8>,
    width: u32,
    height: u32,
    region: NormRect,
    extra: &BTreeMap<String, f32>,
    inputs: &PortraitJobInputs<'_>,
    require_renderer_receipt: bool,
) -> PortraitRunReport {
    let mut report = PortraitRunReport {
        job_id: inputs.job_id(),
        effects: Vec::new(),
    };
    let missing_renderer_receipt = require_renderer_receipt
        && inputs.receipt.is_none()
        && matches!(
            inputs.origin,
            PortraitJobInputOrigin::RendererJob { .. }
        );
    let receipt_refusal = if missing_renderer_receipt {
        Some(Refusal::MissingInput("renderer receipt"))
    } else {
        inputs.receipt.as_ref().and_then(|receipt| {
            receipt.refusal(portrait, width, height, region, extra, inputs.masks)
        })
    };
    let mut effects: Vec<(u32, f32)> = extra
        .iter()
        .filter_map(|(key, value)| {
            let id: u32 = key.strip_prefix("pc:")?.parse().ok()?;
            Some((id, *value))
        })
        .collect();
    effects.sort_by_key(|&(pf, _)| pf);

    if let Some(refusal) = receipt_refusal {
        for &(pf, _) in &effects {
            let Some(meta) = aurora_portrait::lookup(pf) else {
                continue;
            };
            if meta.access != Access::Cpu || !supports_effect(pf) {
                continue;
            }
            report.effects.push(PortraitEffectRun {
                pf,
                backend: PortraitBackend::Skipped,
                dispatches: Vec::new(),
                fallback_reason: Some(format!("receipt refused: {refusal:?}")),
                skipped: true,
                executed: false,
                refusal: Some(refusal.clone()),
                skip_reason: Some(PortraitSkipReason::InputRefused(refusal.clone())),
            });
        }
        return report;
    }

    let Some(source) = Image::new(width, height, std::mem::take(portrait)) else {
        return report;
    };
    let landmark_frame_digest = (inputs.slim.is_some()
        && effects.iter().any(|(pf, degree)| *degree != 0.0 && face_mesh_group(*pf).is_some()))
        .then(|| sha256_32(&source.pixels));

    let neutral_grey_original = effects
        .iter()
        .any(|&(pf, degree)| pf == 30013 && degree > 0.0 && degree <= 1.0)
        .then(|| source.clone());
    let active_plump = effects
        .iter()
        .filter(|(pf, degree)| portrait_plump_stage::supports(*pf) && *degree != 0.0)
        .map(|(pf, _)| *pf)
        .collect::<BTreeSet<_>>();
    let plump_descriptor = if active_plump.contains(&30_210) {
        extra
            .get("pc:30210")
            .copied()
            .ok_or(Refusal::MissingInput("PF30210 same-job UI"))
            .and_then(full_face_descriptor_from_ui)
            .map(Some)
    } else {
        Ok(None)
    };
    let mut current = source;
    // Keep the main-session optimization: skin effects share one sampled plane.
    // Non-skin rows receive their own explicit planes from `PortraitJobInputs`.
    let skin_mask = inputs.masks.mask_u8(MaskKind::Skin, width, height, region);
    let mut slim_result: Option<Result<bool, Refusal>> = None;
    let mut face_results: BTreeMap<u32, Result<FaceGeometryRun, Refusal>> = BTreeMap::new();
    let mut face_mesh_results: BTreeMap<u32, FaceMeshStageResult> = BTreeMap::new();
    let mut plump_result: Option<Result<PlumpGroupRun, Refusal>> = None;
    let mut effects = effects.into_iter().peekable();
    while let Some((pf, degree)) = effects.next() {
        let Some(meta) = aurora_portrait::lookup(pf) else {
            continue;
        };
        if meta.access != Access::Cpu || !supports_effect(pf) {
            log::debug!(
                "portrait PF{pf} skipped; missing same-job inputs: {:?}",
                meta.inputs
            );
            // A row with a callable producer reports the inputs that producer
            // documents; only a row with no kernel at all stays NotImplemented.
            let refusal = match portrait_face_stage::producer_gap(pf) {
                Some(gap) => Refusal::MissingInput(gap),
                None => Refusal::NotImplemented(pf),
            };
            report.effects.push(PortraitEffectRun {
                pf,
                backend: PortraitBackend::Skipped,
                dispatches: Vec::new(),
                fallback_reason: Some(format!("{refusal:?}")),
                skipped: true,
                executed: false,
                refusal: Some(refusal.clone()),
                skip_reason: Some(PortraitSkipReason::InputRefused(refusal)),
            });
            continue;
        }
        let mut run = PortraitEffectRun {
            pf,
            backend: PortraitBackend::Cpu,
            dispatches: Vec::new(),
            fallback_reason: None,
            skipped: true,
            executed: false,
            refusal: None,
            skip_reason: None,
        };
        if inputs.origin == PortraitJobInputOrigin::MasksOnly {
            let refusal = Refusal::MissingInput("same-job input bundle");
            run.backend = PortraitBackend::Skipped;
            run.fallback_reason =
                Some("skipped: masks-only entry has no same-job producer".to_string());
            run.refusal = Some(refusal.clone());
            run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal));
            report.effects.push(run);
            continue;
        }
        if portrait_plump_stage::supports(pf) {
            if !active_plump.contains(&pf) {
                run.backend = PortraitBackend::Skipped;
                report.effects.push(run);
                continue;
            }
            let result = match plump_result.as_ref() {
                Some(result) => result.clone(),
                None => {
                    let result = match plump_descriptor.as_ref() {
                        Ok(descriptor) => portrait_plump_stage::apply(
                            &current,
                            &active_plump,
                            *descriptor,
                            inputs.job_id(),
                            inputs.plump,
                        ),
                        Err(refusal) => Err(refusal.clone()),
                    }
                    .and_then(|run| {
                        run.ok_or(Refusal::ValueOutOfRange("plump active request"))
                    });
                    if let Ok(group) = result.as_ref() {
                        current = group.image.clone();
                    }
                    plump_result = Some(result.clone());
                    result
                }
            };
            match result {
                Ok(group) => {
                    run.backend = match group.backend {
                        PlumpRasterBackend::Cuda => PortraitBackend::Cuda,
                        PlumpRasterBackend::Cpu => PortraitBackend::Cpu,
                    };
                    run.skipped = false;
                    run.executed = true;
                }
                Err(refusal) => {
                    run.backend = PortraitBackend::Skipped;
                    run.fallback_reason = Some(format!("input refused: {refusal:?}"));
                    run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal.clone()));
                    run.refusal = Some(refusal);
                }
            }
            report.effects.push(run);
            continue;
        }
        if matches!(pf, 8025 | 8026) {
            let result = match slim_result.as_ref() {
                Some(result) => result.clone(),
                None => {
                    let control =
                        |id| extra.get(&format!("pc:{id}")).copied().unwrap_or(0.0);
                    let result = match portrait_slim_stage::apply(
                        &current,
                        region,
                        inputs.job_id(),
                        inputs.slim,
                        [control(8025), control(8026)],
                    ) {
                        Ok(Some(next)) => {
                            current = next;
                            Ok(true)
                        }
                        Ok(None) => Ok(false),
                        Err(refusal) => Err(refusal),
                    };
                    slim_result = Some(result.clone());
                    result
                }
            };
            match result {
                Ok(true) => {
                    run.skipped = false;
                    run.executed = true;
                }
                Ok(false) => run.backend = PortraitBackend::Skipped,
                Err(refusal) => {
                    run.backend = PortraitBackend::Skipped;
                    run.fallback_reason = Some(format!("input refused: {refusal:?}"));
                    run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal.clone()));
                    run.refusal = Some(refusal);
                }
            }
            report.effects.push(run);
            continue;
        }
        if let Some(group) = face_mesh_group(pf) {
            let controls = [
                extra
                    .get(&format!("pc:{}", group.left))
                    .copied()
                    .unwrap_or_default(),
                extra
                    .get(&format!("pc:{}", group.right))
                    .copied()
                    .unwrap_or_default(),
            ];
            let result = face_mesh_results
                .entry(group.left)
                .or_insert_with(|| {
                    let result =
                        apply_face_mesh_group(&current, landmark_frame_digest, region, inputs.job_id(), inputs.slim, group, controls);
                    if let Some(next) = result.image.as_ref() {
                        current = next.clone();
                    }
                    result
                })
                .clone();
            run.backend = result.backend;
            run.dispatches = result.dispatches;
            run.fallback_reason = result.fallback_reason;
            if let Some(refusal) = result.refusal {
                run.backend = PortraitBackend::Skipped;
                run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal.clone()));
                run.refusal = Some(refusal);
            } else if result.image.is_some() && degree != 0.0 {
                run.skipped = false;
                run.executed = true;
            }
            report.effects.push(run);
            continue;
        }
        if let Some((left, right)) = portrait_face_stage::pair(pf) {
            let result = match face_results.get(&left) {
                Some(result) => result.clone(),
                None => {
                    let control =
                        |id| extra.get(&format!("pc:{id}")).copied().unwrap_or(0.0);
                    let controls = [control(left), if left == right { 0.0 } else { control(right) }];
                    let result = match portrait_face_stage::apply(
                        &current,
                        region,
                        inputs.job_id(),
                        inputs.face_geometry,
                        inputs.slim,
                        left,
                        controls,
                    ) {
                        Ok(run) => {
                            if let Some(next) = run.image.as_ref() {
                                current = next.clone();
                            }
                            Ok(run)
                        }
                        Err(refusal) => Err(refusal),
                    };
                    face_results.insert(left, result.clone());
                    result
                }
            };
            match result {
                Ok(face_run) => {
                    run.backend = face_run.backend;
                    run.dispatches = face_run.dispatches;
                    run.fallback_reason = face_run.fallback_reason;
                    if face_run.image.is_some() {
                    run.skipped = false;
                    run.executed = true;
                }
                }
                Err(refusal) => {
                    run.backend = PortraitBackend::Skipped;
                    run.fallback_reason = Some(format!("input refused: {refusal:?}"));
                    run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal.clone()));
                    run.refusal = Some(refusal);
                }
            }
            report.effects.push(run);
            continue;
        }
        if portrait_body_stage::supports(pf) {
            #[cfg(windows)]
            {
                if let Some(result) =
                    try_cuda_body_mesh(&current, pf, inputs.job_id(), inputs.body_geometry)
                {
                    match result {
                        Ok((next, dispatches)) if !dispatches.is_empty() => {
                            current = next;
                            run.backend = PortraitBackend::Cuda;
                            run.dispatches = dispatches;
                            run.skipped = false;
                            run.executed = true;
                            report.effects.push(run);
                            continue;
                        }
                        Ok(_) => {
                            run.fallback_reason =
                                Some("CUDA body mesh returned no dispatch".to_string());
                        }
                        Err(reason) => run.fallback_reason = Some(reason),
                    }
                }
            }
            match portrait_body_stage::apply(
                &current,
                pf,
                degree,
                inputs.job_id(),
                inputs.body_geometry,
            ) {
                Ok(Some(next)) => {
                    current = next;
                    run.backend = PortraitBackend::Cpu;
                    run.skipped = false;
                    run.executed = true;
                }
                Ok(None) => run.backend = PortraitBackend::Skipped,
                Err(refusal) => {
                    run.backend = PortraitBackend::Skipped;
                    run.fallback_reason = Some(format!("input refused: {refusal:?}"));
                    run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal.clone()));
                    run.refusal = Some(refusal);
                }
            }
            report.effects.push(run);
            continue;
        }
        if pf == 101 {
            if degree == 0.0 {
                run.backend = PortraitBackend::Skipped;
            } else {
                match inputs.pf101 {
                    Some(bundle) => match bundle.apply(&current, inputs.job_id(), degree) {
                        Ok(next) => {
                            current = next;
                            run.backend = PortraitBackend::Cpu;
                            run.skipped = false;
                            run.executed = true;
                        }
                        Err(refusal) => {
                            run.backend = PortraitBackend::Skipped;
                            run.fallback_reason = Some(format!("input refused: {refusal:?}"));
                            run.skip_reason =
                                Some(PortraitSkipReason::InputRefused(refusal.clone()));
                            run.refusal = Some(refusal);
                        }
                    },
                    None => {
                        let refusal = Refusal::MissingInput("PF101_TYPED_PRODUCER");
                        run.backend = PortraitBackend::Skipped;
                        run.fallback_reason = Some(format!("input refused: {refusal:?}"));
                        run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal.clone()));
                        run.refusal = Some(refusal);
                    }
                }
            }
            report.effects.push(run);
            continue;
        }
        if pf == 90130 {
            if degree == 0.0 {
                run.backend = PortraitBackend::Skipped;
            } else {
                match inputs.pf90130 {
                    Some(bundle) => {
                        let stage = bundle.apply(&current, &current, inputs.job_id(), degree);
                        run.backend = stage.backend;
                        if let Some(writeback) = stage.writeback {
                            current = writeback.rgba8;
                            run.skipped = false;
                            run.executed = true;
                        } else if stage.refusal.is_none() {
                            run.backend = PortraitBackend::Skipped;
                        }
                        if let Some(refusal) = stage.refusal {
                            run.backend = PortraitBackend::Skipped;
                            run.fallback_reason = Some(format!("input refused: {refusal:?}"));
                            run.skip_reason =
                                Some(PortraitSkipReason::InputRefused(refusal.clone()));
                            run.refusal = Some(refusal);
                        }
                    }
                    None => {
                        let refusal = Refusal::MissingInput("PF90130_TYPED_PRODUCER");
                        run.backend = PortraitBackend::Skipped;
                        run.fallback_reason = Some(format!("input refused: {refusal:?}"));
                        run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal.clone()));
                        run.refusal = Some(refusal);
                    }
                }
            }
            report.effects.push(run);
            continue;
        }
        if repair_stage::PFS.contains(&pf) {
            match repair_stage::apply(&current, pf, degree, inputs.job_id(), inputs.repairs) {
                Ok(Some(next)) => {
                    current = next;
                    run.skipped = false;
                    run.executed = true;
                }
                Ok(None) => run.backend = PortraitBackend::Skipped,
                Err(refusal) => {
                    run.backend = PortraitBackend::Skipped;
                    run.fallback_reason = Some(format!("input refused: {refusal:?}"));
                    run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal.clone()));
                    run.refusal = Some(refusal);
                }
            }
            report.effects.push(run);
            continue;
        }
        if portrait_historical_tone::supports(pf) {
            let gray_3d_controls = [
                extra.get("pc:30015").copied().unwrap_or_default(),
                extra.get("pc:30016").copied().unwrap_or_default(),
            ];
            match portrait_historical_tone::apply(
                &current,
                inputs.job_id(),
                pf,
                degree,
                inputs.historical_tone,
                gray_3d_controls,
            ) {
                Ok(Some(next)) => {
                    current = next;
                    run.skipped = false;
                    run.executed = true;
                }
                Ok(None) => run.backend = PortraitBackend::Skipped,
                Err(refusal) => {
                    run.backend = PortraitBackend::Skipped;
                    run.fallback_reason = Some(format!("input refused: {refusal:?}"));
                    run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal.clone()));
                    run.refusal = Some(refusal);
                }
            }
            report.effects.push(run);
            continue;
        }
        if portrait_historical_skin::PFS.contains(&pf) {
            if let Some(input) = inputs
                .historical_skin
                .and_then(|inputs| inputs.get(&pf))
            {
                let plan =
                    portrait_historical_skin::plan_historical_skin_backend(pf, &input.data);
                if plan.backend == portrait_historical_skin::HistoricalSkinPath::Cpu {
                    run.fallback_reason = plan.fallback_reason;
                }
            }
            match portrait_historical_skin::apply(
                &current,
                inputs.job_id(),
                pf,
                degree,
                inputs.historical_skin,
            ) {
                Ok(Some(next)) => {
                    current = next;
                    run.skipped = false;
                    run.executed = true;
                }
                Ok(None) => run.backend = PortraitBackend::Skipped,
                Err(refusal) => {
                    run.backend = PortraitBackend::Skipped;
                    run.fallback_reason = Some(format!("input refused: {refusal:?}"));
                    run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal.clone()));
                    run.refusal = Some(refusal);
                }
            }
            report.effects.push(run);
            continue;
        }
        if pf == 30013 {
            let result = (|| {
                check_degree01(degree, "PF30013 degree")?;
                let Some(original) = neutral_grey_original.as_ref() else {
                    return Ok(None);
                };
                let mask = required(skin_mask.as_deref(), "PF30013 skin mask")?;
                // The pre-stack reference is aligned only until a portrait warp runs.
                if report.effects.iter().any(|effect| {
                    effect.executed && matches!(effect.pf, 8025 | 8026 | 8050 | 8051)
                }) {
                    return Err(Refusal::MissingInput("PF30013 aligned reference after warp"));
                }
                if original.pixels == current.pixels || !mask.iter().any(|&weight| weight != 0) {
                    return Ok(None);
                }
                let roi = skin_adjust::Roi { x: 0, y: 0, width, height };
                let next = skin_adjust::skin_grey_apply(original, &current, mask, roi, degree)?;
                Ok((next.pixels != current.pixels).then_some(next))
            })();
            match result {
                Ok(Some(next)) => {
                    current = next;
                    run.skipped = false;
                    run.executed = true;
                }
                Ok(None) => run.backend = PortraitBackend::Skipped,
                Err(refusal) => {
                    run.backend = PortraitBackend::Skipped;
                    run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal.clone()));
                    run.refusal = Some(refusal);
                }
            }
            report.effects.push(run);
            continue;
        }
        if pf == 90021 {
            let (next, pf90021_run) = apply_pf90021_stage(
                &current,
                inputs.receipt.as_ref(),
                inputs.pf90021,
            );
            if let Some(next) = next {
                current = next;
            }
            report.effects.push(pf90021_run);
            continue;
        }
        if pf == 30026 {
            let (next, run) = apply_pf30026_stage(&current, region, degree, inputs);
            if let Some(next) = next {
                current = next;
            }
            report.effects.push(run);
            continue;
        }
        if pf == 90033 {
            let (next, suit_run) = apply_suit_effect(&current, inputs.suit_passes);
            if let Some(next) = next {
                current = next;
            }
            report.effects.push(suit_run);
            continue;
        }
        #[cfg(windows)]
        if pf == 90005 {
            if let Some(&(next_pf, next_degree)) = effects.peek() {
                if next_pf == 90007 {
                    if let Some(mask) = skin_mask.as_deref() {
                        if let Some(Ok((next, dispatches))) = try_cuda_skin_adjust_chain(
                            &current,
                            degree,
                            next_degree,
                            mask,
                        ) {
                            if dispatches.len() == 2 {
                                effects.next();
                                current = next;
                                report.effects.push(PortraitEffectRun {
                                    pf,
                                    backend: PortraitBackend::Cuda,
                                    dispatches: vec![dispatches[0].clone()],
                                    fallback_reason: None,
                                    skipped: false,
                                    executed: true,
                                    refusal: None,
                                    skip_reason: None,
                                });
                                report.effects.push(PortraitEffectRun {
                                    pf: next_pf,
                                    backend: PortraitBackend::Cuda,
                                    dispatches: vec![dispatches[1].clone()],
                                    fallback_reason: None,
                                    skipped: false,
                                    executed: true,
                                    refusal: None,
                                    skip_reason: None,
                                });
                                continue;
                            }
                        }
                    }
                }
            }
        }
        let needs_hlf_auto = (pf == 30021 && inputs.hlf_face.is_none())
            || (pf == 30022 && inputs.hlf_body.is_none());
        let mut cuda_hlf_auto_fallback = None;
        #[cfg(windows)]
        if needs_hlf_auto {
            if let Some((next, dispatches)) =
                try_cuda_hlf_auto(
                    &current,
                    pf,
                    degree,
                    region,
                    inputs.masks,
                    &mut cuda_hlf_auto_fallback,
                )
            {
                current = next;
                run.backend = PortraitBackend::Cuda;
                run.dispatches = dispatches;
                run.skipped = false;
                run.executed = true;
                report.effects.push(run);
                continue;
            }
        }
        if let Some(reason) = cuda_hlf_auto_fallback {
            run.fallback_reason = Some(format!("CUDA HLF auto: {reason}"));
        }
        let hlf_auto = if needs_hlf_auto {
            match produce_hlf_auto(
                &current,
                region,
                inputs.masks,
                HlfAutoProducerOptions::default(),
            ) {
                Ok(auto) => Some(auto),
                Err(refusal) => {
                    run.backend = PortraitBackend::Skipped;
                    let reason = format!("input refused: {refusal:?}");
                    run.fallback_reason = Some(match run.fallback_reason.take() {
                        Some(previous) => format!("{previous}; {reason}"),
                        None => reason,
                    });
                    run.refusal = Some(refusal.clone());
                    run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal));
                    report.effects.push(run);
                    continue;
                }
            }
        } else {
            None
        };
        if let Some(auto) = hlf_auto.as_ref() {
            if let Some(reason) = auto.preprocess_fallback.as_ref() {
                let reason = format!("HLF preprocessing CPU fallback: {reason}");
                run.fallback_reason = Some(match run.fallback_reason.take() {
                    Some(previous) => format!("{previous}; {reason}"),
                    None => reason,
                });
            }
        }
        if let Err(refusal) =
            validate_effect_inputs(
                pf,
                &current,
                region,
                extra,
                skin_mask.as_deref(),
                inputs,
                hlf_auto.as_ref(),
            )
        {
            let reason = format!("input refused: {refusal:?}");
            run.backend = PortraitBackend::Skipped;
            run.fallback_reason = Some(match run.fallback_reason.take() {
                Some(previous) => format!("{previous}; {reason}"),
                None => reason,
            });
            run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal.clone()));
            run.refusal = Some(refusal);
            report.effects.push(run);
            continue;
        }
        #[cfg(windows)]
        let cuda_result =
            try_cuda_effect(&current, pf, degree, skin_mask.as_deref()).or_else(|| {
                try_cuda_same_job_effect(&current, pf, degree, inputs, hlf_auto.as_ref())
            });
        #[cfg(windows)]
        if let Some(result) = cuda_result {
            match result {
                Ok((next, dispatches)) if !dispatches.is_empty() => {
                    current = next;
                    run.backend = PortraitBackend::Cuda;
                    if let Some(auto) = hlf_auto.as_ref() {
                        run.dispatches = auto.preprocess_dispatches.clone();
                        run.dispatches.extend(dispatches);
                    } else {
                        run.dispatches = dispatches;
                    }
                    run.skipped = false;
                    run.executed = true;
                    report.effects.push(run);
                    continue;
                }
                Ok(_) => {
                    let reason = "CUDA returned no dispatch; CPU kernel selected for parity";
                    run.fallback_reason = Some(match run.fallback_reason.take() {
                        Some(previous) => format!("{previous}; {reason}"),
                        None => reason.to_string(),
                    });
                }
                Err(err) => {
                    let reason = format!("{err:?}");
                    run.fallback_reason = Some(match run.fallback_reason.take() {
                        Some(previous) => format!("{previous}; CUDA effect: {reason}"),
                        None => reason,
                    });
                }
            }
        }
        let result = apply_cpu_effect(
            &current,
            pf,
            degree,
            extra,
            skin_mask.as_deref(),
            width,
            height,
            inputs,
            hlf_auto.as_ref(),
        );
        match result {
            Ok(next) => {
                current = next;
                run.skipped = false;
                run.executed = true;
            }
            Err(err) => {
                let cpu_reason = format!("{err:?}");
                run.fallback_reason = Some(match run.fallback_reason.take() {
                    Some(gpu_reason) => format!("{gpu_reason}; CPU: {cpu_reason}"),
                    None => format!("CPU: {cpu_reason}"),
                });
                log::debug!("portrait PF{pf} skipped: {err:?}");
                run.refusal = Some(err);
            }
        }
        report.effects.push(run);
    }

    *portrait = current.pixels;
    report
}

#[derive(Clone, Debug, PartialEq)]
pub struct Pf90130StageRun {
    pub backend: PortraitBackend,
    pub refusal: Option<Refusal>,
    pub output: Option<aurora_portrait::kernels::pf90130_chain::Output>,
    pub writeback: Option<GlareFinalWritebackOutput>,
    pub candidate_only: bool,
}

impl Pf90130StageRun {
    fn refused(refusal: Refusal) -> Self {
        Self {
            backend: PortraitBackend::Skipped,
            refusal: Some(refusal),
            output: None,
            writeback: None,
            candidate_only: true,
        }
    }
}

/// Runs the typed PF90130 producer through the wl41 chain and same-frame
/// writeback. No frame or receipt is synthesized when the caller omits it.
pub fn apply_pf90130_producer_stage(
    producer: Option<&Pf90130GlareProducer>,
    source: Option<&Image>,
    destination: Option<&Image>,
    geometry: Option<&Pf90130AlignedGeometry<'_>>,
) -> Pf90130StageRun {
    let Some(producer) = producer else {
        return Pf90130StageRun::refused(Refusal::MissingInput("PF90130_SAME_JOB_RECEIPT"));
    };
    let Some(source) = source else {
        return Pf90130StageRun::refused(Refusal::MissingInput("PF90130_SOURCE_FRAME"));
    };
    let Some(destination) = destination else {
        return Pf90130StageRun::refused(Refusal::MissingInput("PF90130_DESTINATION_FRAME"));
    };
    let Some(geometry) = geometry else {
        return Pf90130StageRun::refused(Refusal::MissingInput("PF90130_SOURCE_GEOMETRY"));
    };
    if let Err(refusal) = producer.bridge() {
        return Pf90130StageRun::refused(refusal);
    }
    let input = producer.chain_input(geometry);
    let prepared = match aurora_portrait::kernels::pf90130_chain::prepare(&input) {
        Ok(prepared) => prepared,
        Err(refusal) => return Pf90130StageRun::refused(refusal),
    };
    let output = aurora_portrait::kernels::pf90130_chain::run_prepared(&prepared);
    let writeback = match apply_glare_chain_final_writeback(
        source,
        destination,
        &input,
        &output,
        &GlareChainFinalWritebackInputs {
            source_sha256: Some(sha256_32(&source.pixels)),
            destination_sha256: Some(sha256_32(&destination.pixels)),
        },
    ) {
        Ok(writeback) => writeback,
        Err(refusal) => return Pf90130StageRun::refused(refusal),
    };
    Pf90130StageRun {
        backend: PortraitBackend::Cpu,
        refusal: None,
        output: Some(output),
        writeback,
        candidate_only: true,
    }
}

/// Runs the PF8050/PF8051 six-stage candidate geometry path.
pub fn apply_pf8050_pf8051_geometry(
    inputs: Pf8050GeometryInputs<'_>,
) -> PortraitGeometryRun {
    let resolved = match cuda_pf8050::resolve(inputs) {
        Ok(resolved) => resolved,
        Err(refusal) => {
            return PortraitGeometryRun {
                backend: PortraitBackend::Skipped,
                selection: Selection::Refused(DeviceRefusal::EffectUnsupported),
                dispatches: Vec::new(),
                fallback_reason: Some(format!("input refused: {refusal:?}")),
                refusal: Some(refusal),
                stages: Vec::new(),
                candidate_only: true,
            };
        }
    };

    #[cfg(windows)]
    let fallback_reason = {
        let cuda_result = match forced_cuda_failure() {
            Some(error) => Err(error),
            None => match cuda_pf8050::Pf8050Stage::open() {
                Ok(gpu) => gpu.apply(resolved),
                Err(error) => Err(error),
            },
        };
        match cuda_result {
            Ok((output, dispatch)) if !dispatch.kernel.is_empty() => {
                return PortraitGeometryRun {
                    backend: PortraitBackend::Cuda,
                    selection: Selection::Selected(DeviceBackend::Cuda),
                    dispatches: vec![dispatch],
                    fallback_reason: None,
                    refusal: None,
                    stages: output.stages,
                    candidate_only: true,
                };
            }
            Ok(_) => Some("CUDA returned no dispatch".to_string()),
            Err(error) => Some(format!("{error:?}")),
        }
    };
    #[cfg(not(windows))]
    let fallback_reason = None;

    let output = cuda_pf8050::apply_cpu(resolved);
    PortraitGeometryRun {
        backend: PortraitBackend::Cpu,
        selection: Selection::Selected(DeviceBackend::Cpu),
        dispatches: Vec::new(),
        fallback_reason,
        refusal: None,
        stages: output.stages,
        candidate_only: true,
    }
}

/// Full product stage entry. Missing vendor receipts are typed refusals; a
/// complete receipt set still reports the open raster consumer, never success.
pub fn apply_pf8050_pf8051_stage(inputs: Pf8050StageInputs<'_>) -> PortraitGeometryRun {
    if let Err(refusal) = require_pf8050_vendor_inputs(inputs.runtime) {
        return PortraitGeometryRun {
            backend: PortraitBackend::Skipped,
            selection: Selection::Refused(DeviceRefusal::EffectUnsupported),
            dispatches: Vec::new(),
            fallback_reason: Some(format!("vendor input refused: {refusal:?}")),
            refusal: Some(refusal),
            stages: Vec::new(),
            candidate_only: false,
        };
    }

    let mut run = apply_pf8050_pf8051_geometry(inputs.geometry);
    run.refusal = Some(Refusal::NotImplemented(8050));
    run.fallback_reason = Some(
        "vendor VBO/FBO/readback consumer is not wired; candidate geometry only".to_string(),
    );
    run
}

fn require_pf8050_vendor_inputs(runtime: Pf8050VendorRuntime<'_>) -> Result<(), Refusal> {
    required(runtime.attach, "PF8050_ATTACH")?;
    required(runtime.vec3, "PF8050_VEC3")?;
    required(runtime.roi, "PF8050_ROI")?;
    required(runtime.vbo, "PF8050_VBO")?;
    required(runtime.fbo, "PF8050_FBO")?;
    required(runtime.readback, "PF8050_READBACK")?;
    Ok(())
}

fn required<T>(value: Option<T>, name: &'static str) -> Result<T, Refusal> {
    value.ok_or(Refusal::MissingInput(name))
}

fn check_plane(len: usize, expected: usize, name: &'static str) -> Result<(), Refusal> {
    if len == expected {
        Ok(())
    } else {
        Err(Refusal::ShapeMismatch(name))
    }
}

fn check_f32_plane(plane: &[f32], expected: usize, name: &'static str) -> Result<(), Refusal> {
    check_plane(plane.len(), expected, name)?;
    if plane.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Refusal::ValueOutOfRange(name))
    }
}

fn check_degree01(value: f32, name: &'static str) -> Result<(), Refusal> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(Refusal::ValueOutOfRange(name))
    }
}

fn check_ui_percent(value: f32, name: &'static str) -> Result<(), Refusal> {
    if value.is_finite() && (0.0..=100.0).contains(&value) {
        Ok(())
    } else {
        Err(Refusal::ValueOutOfRange(name))
    }
}

fn check_roi(width: u32, height: u32, roi: (u32, u32, u32, u32)) -> Result<(), Refusal> {
    let (x, y, w, h) = roi;
    if w == 0
        || h == 0
        || (x as u64 + w as u64) > width as u64
        || (y as u64 + h as u64) > height as u64
    {
        Err(Refusal::ValueOutOfRange("roi"))
    } else {
        Ok(())
    }
}

fn check_skin_roi(width: u32, height: u32, roi: Roi) -> Result<(), Refusal> {
    check_roi(width, height, (roi.x, roi.y, roi.width, roi.height))
}

fn validate_hlf_face(current: &Image, input: HlfFaceInputs<'_>) -> Result<(), Refusal> {
    if (input.width, input.height) != (current.width, current.height) {
        return Err(Refusal::ShapeMismatch("PF30021_HLF frame"));
    }
    check_skin_roi(current.width, current.height, input.roi)?;
    let pixels = current.pixel_count();
    check_f32_plane(
        input.source_rgb_full_f32,
        pixels * 3,
        "PF30021_source_rgb_full_f32",
    )?;
    check_f32_plane(input.median_rgb_full_f32, pixels * 3, "median_rgb_full_f32")?;
    check_f32_plane(input.gauss_rgb_full_f32, pixels * 3, "gauss_rgb_full_f32")?;
    check_f32_plane(input.face_mask_full_f32, pixels, "face_mask_full_f32")?;
    check_f32_plane(input.skin_mask_full_f32, pixels, "skin_mask_full_f32")
}

fn validate_hlf_body(current: &Image, input: HlfBodyInputs<'_>) -> Result<(), Refusal> {
    if (input.width, input.height) != (current.width, current.height) {
        return Err(Refusal::ShapeMismatch("PF30022_HLF frame"));
    }
    check_skin_roi(current.width, current.height, input.roi)?;
    let pixels = current.pixel_count();
    check_f32_plane(input.median_rgb_full_f32, pixels * 3, "median_rgb_full_f32")?;
    check_f32_plane(input.gauss_rgb_full_f32, pixels * 3, "gauss_rgb_full_f32")?;
    check_f32_plane(
        input.face_mask_rgba_full_f32,
        pixels * 4,
        "face_mask_rgba_full_f32",
    )?;
    check_f32_plane(input.skin_mask_full_f32, pixels, "skin_mask_full_f32")?;
    let uniforms = [
        input.uniforms.spotless_degree,
        input.uniforms.retouch_degree,
        input.uniforms.spotless_acne_threshold,
        input.uniforms.spotless_high_freq_factor,
        input.uniforms.spotless_low_freq_alpha,
        input.uniforms.retouch_high_freq_factor,
        input.uniforms.diff_min_range,
        input.uniforms.diff_max_range,
    ];
    if !uniforms.iter().all(|value| value.is_finite())
        || input.uniforms.diff_min_range > input.uniforms.diff_max_range
    {
        return Err(Refusal::ValueOutOfRange("PF30022_HLF uniforms"));
    }
    if let Some(lut) = input.adjust_lut.as_ref() {
        lut.validate()?;
    } else if input.uniforms.spotless_degree > 0.0 {
        return Err(Refusal::MissingInput("adjust_lut"));
    }
    Ok(())
}

fn validate_wrinkle(current: &Image, input: WrinkleInputs<'_>) -> Result<(), Refusal> {
    input.base_full.validate()?;
    input.residual_full.validate()?;
    if (input.base_full.width, input.base_full.height) != (current.width, current.height) {
        return Err(Refusal::ShapeMismatch("wrinkle_base_full"));
    }
    if (input.residual_full.width, input.residual_full.height) != (current.width, current.height) {
        return Err(Refusal::ShapeMismatch("wrinkle_residual_full"));
    }
    let pixels = current.pixel_count();
    check_plane(
        input.mask_rgba_full.len(),
        pixels * 4,
        "wrinkle_mask_rgba_full",
    )?;
    check_plane(input.skin_mask_full.len(), pixels, "wrinkle_skin_mask_full")?;
    check_skin_roi(current.width, current.height, input.roi)?;
    let uniforms = [
        input.uniforms.legal_alpha,
        input.uniforms.dark_circle_alpha,
        input.uniforms.diff_factor,
        input.uniforms.diff_factor2,
        input.uniforms.color_threshold_factor,
        input.uniforms.mix_alpha,
        input.uniforms.diff_min_threshold,
        input.uniforms.diff_max_threshold,
        input.uniforms.high_freq_min_threshold,
        input.uniforms.high_freq_max_threshold,
    ];
    if !uniforms.iter().all(|value| value.is_finite())
        || input.uniforms.diff_min_threshold > input.uniforms.diff_max_threshold
        || input.uniforms.high_freq_min_threshold > input.uniforms.high_freq_max_threshold
    {
        return Err(Refusal::ValueOutOfRange("wrinkle_uniforms"));
    }
    Ok(())
}

fn check_norm_rect(rect: NormRect, name: &'static str) -> Result<(), Refusal> {
    if [
        rect.x,
        rect.y,
        rect.width,
        rect.height,
    ]
    .into_iter()
    .all(f32::is_finite)
        && rect.x >= 0.0
        && rect.y >= 0.0
        && rect.width > 0.0
        && rect.height > 0.0
        && rect.x + rect.width <= 1.0
        && rect.y + rect.height <= 1.0
    {
        Ok(())
    } else {
        Err(Refusal::ValueOutOfRange(name))
    }
}

fn pf90021_skipped(
    reason: PortraitSkipReason,
    detail: String,
    refusal: Option<Refusal>,
) -> (Option<Image>, PortraitEffectRun) {
    (
        None,
        PortraitEffectRun {
            pf: 90021,
            backend: PortraitBackend::Skipped,
            dispatches: Vec::new(),
            fallback_reason: Some(detail),
            skipped: true,
            executed: false,
            refusal,
            skip_reason: Some(reason),
        },
    )
}

fn resolve_pf90021<'a>(
    current: &Image,
    job_receipt: Option<&PortraitJobReceipt>,
    inputs: Option<Pf90021Inputs<'a>>,
) -> Result<
    Option<(
        Pf90021Direction,
        &'a VolumeLut64,
        &'a [u8],
        Pf90021RuntimeScalars,
    )>,
    Refusal,
> {
    let input = required(inputs, "PF90021_SAME_JOB_INPUTS")?;
    if input.direction == Pf90021Direction::Neutral {
        return Ok(None);
    }
    let (lut, missing_lut) = match input.direction {
        Pf90021Direction::Positive => (input.positive_lut, "PF90021_POSITIVE_LUT64"),
        Pf90021Direction::Negative => (input.negative_lut, "PF90021_NEGATIVE_LUT64"),
        Pf90021Direction::Neutral => unreachable!(),
    };
    let lut = required(lut, missing_lut)?;
    let mask_r = required(input.mask_r, "PF90021_MASK_R")?;
    check_plane(mask_r.len(), current.pixel_count(), "PF90021_MASK_R")?;
    let runtime = required(input.runtime, "PF90021_RUNTIME_SCALARS")?;
    runtime.u_alpha()?;
    let same_job = required(input.same_job, "PF90021_JOB_RECEIPT")?;
    if same_job.job_id == 0
        || same_job.ticket == 0
        || same_job.face_id < 0
        || same_job.mask_texture_id == 0
    {
        return Err(Refusal::ValueOutOfRange("PF90021_JOB_RECEIPT"));
    }
    if same_job
        .roi_bits
        .into_iter()
        .map(f32::from_bits)
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("PF90021_ROI_BOUNDS"));
    }
    let job_receipt =
        required(job_receipt, "PF90021_CURRENT_FRAME_RECEIPT")?;
    if same_job.frame != job_receipt.frame_receipt() {
        return Err(Refusal::ValueOutOfRange(
            "PF90021_CURRENT_FRAME_RECEIPT",
        ));
    }
    if same_job.frame.width() != current.width
        || same_job.frame.height() != current.height
    {
        return Err(Refusal::ShapeMismatch(
            "PF90021_CURRENT_FRAME_RECEIPT",
        ));
    }
    if same_job.mask_channel != 0 {
        return Err(Refusal::ValueOutOfRange("PF90021_MASK_CHANNEL"));
    }
    match same_job.mask_source {
        Pf90021MaskSource::FaceParsingChannel0Rasterized => {}
    }
    let direction_receipt =
        required(input.direction_receipt, "PF90021_DIRECTION_RECEIPT")?;
    if !direction_receipt.input01.is_finite()
        || !direction_receipt.output01.is_finite()
    {
        return Err(Refusal::ValueOutOfRange(
            "PF90021_DIRECTION_RECEIPT",
        ));
    }
    let receipt_direction =
        pf90021_direction_from_effect_range(direction_receipt.effect_range)?;
    if receipt_direction != input.direction {
        return Err(Refusal::ValueOutOfRange(
            "PF90021_DIRECTION_RECEIPT",
        ));
    }
    if direction_receipt.param_range != [0.0, 0.5, 1.0] {
        return Err(Refusal::ValueOutOfRange(
            "PF90021_DEGREE_PARAM_RANGE",
        ));
    }
    let expected_degree = pf90021_degree_remap(
        receipt_direction,
        direction_receipt.input01,
    );
    if (direction_receipt.output01 - expected_degree).abs() > 1e-6
        || (runtime.degree01 - direction_receipt.output01).abs() > 1e-6
    {
        return Err(Refusal::ValueOutOfRange(
            "PF90021_DIRECTION_RECEIPT",
        ));
    }
    let runtime_receipt =
        required(input.runtime_receipt, "PF90021_RUNTIME_RECEIPT")?;
    for (actual, receipt) in [
        (runtime.degree01, runtime_receipt.degree01),
        (runtime.opacity01, runtime_receipt.opacity01),
        (runtime.shelter, runtime_receipt.shelter01),
        (runtime.region, runtime_receipt.region01),
    ] {
        if actual.to_bits() != receipt.to_bits() {
            return Err(Refusal::ValueOutOfRange(
                "PF90021_RUNTIME_RECEIPT",
            ));
        }
    }
    Ok(Some((input.direction, lut, mask_r, runtime)))
}

fn apply_pf90021_stage(
    current: &Image,
    job_receipt: Option<&PortraitJobReceipt>,
    inputs: Option<Pf90021Inputs<'_>>,
) -> (Option<Image>, PortraitEffectRun) {
    let resolved = match resolve_pf90021(current, job_receipt, inputs) {
        Ok(Some(resolved)) => resolved,
        Ok(None) => {
            return (
                None,
                PortraitEffectRun {
                    pf: 90021,
                    backend: PortraitBackend::Skipped,
                    dispatches: Vec::new(),
                    fallback_reason: Some("neutral direction; no dispatch".to_string()),
                    skipped: true,
                    executed: false,
                    refusal: None,
                    skip_reason: None,
                },
            );
        }
        Err(error) => {
            return pf90021_skipped(
                PortraitSkipReason::InputRefused(error.clone()),
                format!("input refused: {error:?}"),
                Some(error),
            );
        }
    };
    let (direction, lut, mask_r, runtime) = resolved;

    #[cfg(windows)]
    {
        let result = match forced_cuda_failure() {
            Some(error) => Err(error),
            None => with_pf90021(|gpu| {
                gpu.run(aurora_portrait::cuda_pf90021::Request {
                    source: current,
                    direction,
                    lut: Some(lut),
                    mask_r: Some(mask_r),
                    runtime: Some(runtime),
                })
            }),
        };
        match result {
            Ok(aurora_portrait::cuda_pf90021::Run::Executed { output, dispatch }) => {
                let dispatches = vec![dispatch.lut, dispatch.composite];
                return (
                    Some(output),
                    PortraitEffectRun {
                        pf: 90021,
                        backend: PortraitBackend::Cuda,
                        dispatches,
                        fallback_reason: None,
                        skipped: false,
                        executed: true,
                        refusal: None,
                        skip_reason: None,
                    },
                );
            }
            Ok(aurora_portrait::cuda_pf90021::Run::Bypassed(_)) => {}
            Err(error) => {
                return match aurora_portrait::cuda_pf90021::apply_cpu(
                    current, lut, mask_r, runtime,
                ) {
                    Ok(output) => (
                        Some(output),
                        PortraitEffectRun {
                            pf: 90021,
                            backend: PortraitBackend::Cpu,
                            dispatches: Vec::new(),
                            fallback_reason: Some(format!("{error:?}")),
                            skipped: false,
                            executed: true,
                            refusal: None,
                            skip_reason: None,
                        },
                    ),
                    Err(refusal) => pf90021_skipped(
                        PortraitSkipReason::InputRefused(refusal.clone()),
                        format!("CUDA: {error:?}; CPU: {refusal:?}"),
                        Some(refusal),
                    ),
                };
            }
        }
    }

    match aurora_portrait::cuda_pf90021::apply_cpu(current, lut, mask_r, runtime) {
        Ok(output) => (
            Some(output),
            PortraitEffectRun {
                pf: 90021,
                backend: PortraitBackend::Cpu,
                dispatches: Vec::new(),
                fallback_reason: None,
                skipped: false,
                executed: true,
                refusal: None,
                skip_reason: None,
            },
        ),
        Err(refusal) => pf90021_skipped(
            PortraitSkipReason::InputRefused(refusal.clone()),
            format!("CPU: {refusal:?}"),
            Some(refusal),
        ),
    }
}

#[cfg(windows)]
thread_local! {
    static PF90021: RefCell<Option<aurora_portrait::cuda_pf90021::Pf90021>> =
        const { RefCell::new(None) };
}

#[cfg(windows)]
fn with_pf90021<T>(
    run: impl FnOnce(&aurora_portrait::cuda_pf90021::Pf90021) -> Result<T, CudaError>,
) -> Result<T, CudaError> {
    PF90021.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(aurora_portrait::cuda_pf90021::Pf90021::open()?);
        } else if let Err(error) = slot.as_mut().expect("PF90021 session").resume() {
            *slot = None;
            return Err(error);
        }
        let result = run(slot.as_ref().expect("PF90021 session"));
        if result.is_err() {
            let _ = slot.as_mut().expect("PF90021 session").suspend();
            *slot = None;
            return result;
        }
        match slot.as_mut().expect("PF90021 session").suspend() {
            Ok(()) => result,
            Err(error) => {
                *slot = None;
                if result.is_ok() {
                    Err(error)
                } else {
                    result
                }
            }
        }
    })
}

fn pf30026_skipped(
    reason: PortraitSkipReason,
    detail: String,
    refusal: Option<Refusal>,
) -> PortraitEffectRun {
    PortraitEffectRun {
        pf: 30026,
        backend: PortraitBackend::Skipped,
        dispatches: Vec::new(),
        fallback_reason: Some(detail),
        skipped: true,
        executed: false,
        refusal,
        skip_reason: Some(reason),
    }
}

fn pf30026_cpu_fallback(
    job: AlignmentFinalBlendJob<'_>,
    cuda_reason: String,
) -> (Option<Image>, PortraitEffectRun) {
    match aurora_portrait::kernels::teeth_blend::apply_alignment_final_blend(job) {
        Ok(AlignmentFinalBlendOutcome::Applied(image)) => (
            Some(image),
            PortraitEffectRun {
                pf: 30026,
                backend: PortraitBackend::Cpu,
                dispatches: Vec::new(),
                fallback_reason: Some(cuda_reason),
                skipped: false,
                executed: true,
                refusal: None,
                skip_reason: None,
            },
        ),
        Ok(AlignmentFinalBlendOutcome::Gated) => (
            None,
            pf30026_skipped(
                PortraitSkipReason::EdgeGateClosed,
                format!("{cuda_reason}; CPU edge gate closed"),
                None,
            ),
        ),
        Err(error) => (
            None,
            pf30026_skipped(
                PortraitSkipReason::InputRefused(error.clone()),
                format!("{cuda_reason}; CPU: {error:?}"),
                Some(error),
            ),
        ),
    }
}

fn apply_pf30026_stage<'a>(
    current: &'a Image,
    region: NormRect,
    degree: f32,
    inputs: &'a PortraitJobInputs<'a>,
) -> (Option<Image>, PortraitEffectRun) {
    let refused = |error: Refusal| {
        pf30026_skipped(
            PortraitSkipReason::InputRefused(error.clone()),
            format!("input refused: {error:?}"),
            Some(error),
        )
    };
    if let Err(error) = check_degree01(degree, "PF30026 degree") {
        return (None, refused(error));
    }
    let Some(input) = inputs.teeth_alignment else {
        return (
            None,
            refused(Refusal::MissingInput("PF30026 same-job inputs")),
        );
    };
    let effect_rgb = match required(input.effect_rgb, "PF30026 effect RGB") {
        Ok(value) => value,
        Err(error) => return (None, refused(error)),
    };
    let teeth_mask = match required(input.teeth_mask, "PF30026 teeth mask") {
        Ok(value) => value,
        Err(error) => return (None, refused(error)),
    };
    let gate_binding = match required(input.gate_binding, "PF30026 edge gate binding") {
        Ok(value) => value,
        Err(error) => return (None, refused(error)),
    };
    if let Err(error) = gate_binding.validate() {
        return (None, refused(error));
    }
    if inputs.job_id() != Some(gate_binding.job_id) {
        return (
            None,
            refused(Refusal::MissingInput(
                "PF30026 edge gate same-job ticket",
            )),
        );
    }
    let edge_plane_a = match required(input.edge_plane_a, "PF30026 edge plane A") {
        Ok(value) => value,
        Err(error) => return (None, refused(error)),
    };
    let edge_plane_b = match required(input.edge_plane_b, "PF30026 edge plane B") {
        Ok(value) => value,
        Err(error) => return (None, refused(error)),
    };
    let roi = match required(input.roi, "PF30026 ROI") {
        Ok(value) => value,
        Err(error) => return (None, refused(error)),
    };
    if let Err(error) = check_norm_rect(roi, "PF30026 ROI") {
        return (None, refused(error));
    }
    if roi != region {
        return (None, refused(Refusal::ValueOutOfRange("PF30026 ROI")));
    }
    let job = AlignmentFinalBlendJob {
        original: Some(current),
        effect_rgb: Some(effect_rgb),
        teeth_mask: Some(teeth_mask),
        edge_plane_a: Some(edge_plane_a),
        edge_plane_b: Some(edge_plane_b),
    };
    match aurora_portrait::kernels::teeth_blend::alignment_final_blend_gate(job) {
        Ok(true) => {}
        Ok(false) => {
            return (
                None,
                pf30026_skipped(
                    PortraitSkipReason::EdgeGateClosed,
                    "PF30026 edge gate closed: valid ratio <= 0.14%".to_string(),
                    None,
                ),
            );
        }
        Err(error) => return (None, refused(error)),
    }

    #[cfg(windows)]
    {
        if let Some(error) = forced_cuda_failure() {
            return pf30026_cpu_fallback(job, format!("CUDA unavailable: {error:?}"));
        }
        let gpu = match aurora_portrait::cuda_teeth_blend::TeethBlend::open() {
            Ok(gpu) => gpu,
            Err(error) => return pf30026_cpu_fallback(job, format!("CUDA unavailable: {error:?}")),
        };
        match gpu.apply_alignment_final_blend(job) {
            Ok(aurora_portrait::cuda_teeth_blend::AlignmentFinalBlendGpu::Applied {
                image,
                dispatches,
            }) if !dispatches.is_empty() => (
                Some(image),
                PortraitEffectRun {
                    pf: 30026,
                    backend: PortraitBackend::Cuda,
                    dispatches,
                    fallback_reason: None,
                    skipped: false,
                    executed: true,
                    refusal: None,
                    skip_reason: None,
                },
            ),
            Ok(aurora_portrait::cuda_teeth_blend::AlignmentFinalBlendGpu::Applied { .. }) => {
                pf30026_cpu_fallback(
                    job,
                    "CUDA returned no dispatch; CPU kernel selected for parity".to_string(),
                )
            }
            Ok(aurora_portrait::cuda_teeth_blend::AlignmentFinalBlendGpu::Gated) => (
                None,
                pf30026_skipped(
                    PortraitSkipReason::EdgeGateClosed,
                    "PF30026 edge gate closed: valid ratio <= 0.14%".to_string(),
                    None,
                ),
            ),
            Err(error) => pf30026_cpu_fallback(job, format!("CUDA failed: {error:?}")),
        }
    }

    #[cfg(not(windows))]
    {
        let error = CudaError::Unavailable("PF30026 requires NVIDIA CUDA");
        pf30026_cpu_fallback(job, format!("CUDA unavailable: {error:?}"))
    }
}

fn check_roi_plane(
    bytes: &[u8],
    roi: HairRoi,
    stride: u32,
    bytes_per_pixel: u32,
    name: &'static str,
) -> Result<(), Refusal> {
    let row = roi
        .width
        .checked_mul(bytes_per_pixel)
        .ok_or(Refusal::ShapeMismatch(name))?;
    if stride < row {
        return Err(Refusal::ShapeMismatch(name));
    }
    let required = roi
        .height
        .checked_sub(1)
        .and_then(|last| last.checked_mul(stride))
        .and_then(|head| head.checked_add(row))
        .ok_or(Refusal::ShapeMismatch(name))?;
    if bytes.len() < required as usize {
        return Err(Refusal::ShapeMismatch(name));
    }
    Ok(())
}

fn resolve_hair_smooth<'a>(
    current: &Image,
    detail_strength: f32,
    inputs: &PortraitJobInputs<'a>,
) -> Result<(HairSmoothInputs<'a>, HairRoi), Refusal> {
    check_degree01(detail_strength, "PF90139_DETAIL_STRENGTH")?;
    let hair = required(
        inputs.hair_smooth,
        "PF90139_PF90140_SAME_JOB_PRODUCER",
    )?;
    if hair.ticket == 0 {
        return Err(Refusal::MissingInput(
            "PF90139_PF90140_SAME_JOB_RECEIPT",
        ));
    }
    if (hair.width, hair.height) != (current.width, current.height) {
        return Err(Refusal::ShapeMismatch("PF90139_PF90140 frame"));
    }
    check_degree01(hair.mix_strength, "PF90140_MIX_STRENGTH")?;
    check_roi(current.width, current.height, hair.roi)?;
    let (x, y, width, height) = hair.roi;
    let roi = HairRoi {
        x,
        y,
        width,
        height,
    };
    check_roi_plane(
        required(hair.hair_mask, "PF90139_HAIR_MASK")?,
        roi,
        hair.hair_stride,
        1,
        "PF90139_HAIR_MASK",
    )?;
    check_roi_plane(
        required(hair.mix_mask, "PF90139_MIX_MASK")?,
        roi,
        hair.mix_stride,
        1,
        "PF90139_MIX_MASK",
    )?;
    check_roi_plane(
        required(hair.blend_rgb, "PF90139_BLEND_SOURCE_RGB8")?,
        roi,
        hair.blend_stride,
        3,
        "PF90139_BLEND_SOURCE_RGB8",
    )?;
    check_roi_plane(
        required(hair.detail, "PF90139_DETAIL")?,
        roi,
        hair.detail_stride,
        1,
        "PF90139_DETAIL",
    )?;
    Ok((hair, roi))
}

fn resolve_hair_gloss<'a>(
    current: &Image,
    enabled: f32,
    inputs: &PortraitJobInputs<'a>,
) -> Result<(HairGlossInputs<'a>, HairRoi), Refusal> {
    check_degree01(enabled, "PF90160_ENABLED")?;
    let hair = required(inputs.hair_gloss, "PF90160_SAME_JOB_PRODUCER")?;
    if hair.ticket == 0 {
        return Err(Refusal::MissingInput("PF90160_SAME_JOB_RECEIPT"));
    }
    if (hair.width, hair.height) != (current.width, current.height) {
        return Err(Refusal::ShapeMismatch("PF90160 frame"));
    }
    check_degree01(hair.dynamic_scale, "PF90160_DYNAMIC_SCALE")?;
    check_roi(current.width, current.height, hair.roi)?;
    let (x, y, width, height) = hair.roi;
    let roi = HairRoi {
        x,
        y,
        width,
        height,
    };
    check_roi_plane(
        required(hair.hair_mask, "PF90160_HAIR_MASK")?,
        roi,
        hair.hair_stride,
        1,
        "PF90160_HAIR_MASK",
    )?;
    check_roi_plane(
        required(hair.model_detail, "PF90160_MODEL_DETAIL")?,
        roi,
        hair.detail_stride,
        1,
        "PF90160_MODEL_DETAIL",
    )?;
    Ok((hair, roi))
}

fn find_makeup_input<'a>(
    pf: u32,
    inputs: &PortraitJobInputs<'a>,
) -> Result<MakeupInput<'a>, Refusal> {
    if let Some(input) = inputs
        .makeup
        .and_then(|rows| rows.iter().copied().find(|row| row.pf == pf))
    {
        return Ok(input);
    }
    let bundle = match inputs.makeup_bundle {
        Some(bundle) => bundle,
        None => {
            return Err(match MakeupRoute::resolve(pf) {
                Ok(MakeupRoute::Color { .. }) | Ok(MakeupRoute::ImageMasked { .. }) => {
                    Refusal::MissingInput("makeup mask")
                }
                Ok(MakeupRoute::ImageUnmasked { .. }) => {
                    Refusal::MissingInput("makeup material")
                }
                Err(_) => Refusal::MissingInput("makeup bundle"),
            });
        }
    };
    if let Some(job_id) = inputs.job_id() {
        if bundle.job_id != job_id || bundle.ticket != job_id {
            return Err(Refusal::ValueOutOfRange("makeup bundle job"));
        }
    }
    bundle.makeup_input(pf)
}

/// Contract-test entry for the external makeup adapter. The runtime path keeps calling
/// `find_makeup_input` directly; this only exposes the same lookup so a provider test
/// can prove the bundle it produced is the row the stage actually resolves.
#[cfg(test)]
pub(crate) fn find_makeup_input_for_contract_test<'a>(
    pf: u32,
    inputs: &PortraitJobInputs<'a>,
) -> Result<MakeupInput<'a>, Refusal> {
    find_makeup_input(pf, inputs)
}

fn resolve_makeup_route(pf: u32, input: MakeupInput<'_>) -> Result<MakeupRoute, Refusal> {
    MakeupRoute::resolve_same_job(pf, input.filter_type, input.material.is_some())
}

fn makeup_layer(
    route: MakeupRoute,
    input: MakeupInput<'_>,
    degree: f32,
) -> Result<MakeupLayer, Refusal> {
    if input.filter_type != route.filter_type() {
        return Err(Refusal::ValueOutOfRange("makeup FilterType"));
    }
    check_ui_percent(input.opacity_percent, "opacity_percent")?;
    check_degree01(degree, "degree01")?;
    Ok(MakeupLayer {
        mix_color: input.mix_color,
        opacity_percent: input.opacity_percent,
        degree01: degree,
        filter_type: route.filter_type(),
        mask_channel: route
            .mask_channel()
            .unwrap_or(aurora_portrait::kernels::makeup_material::MaskChannel::A),
    })
}

fn validate_makeup_inputs(
    pf: u32,
    pixels: usize,
    inputs: &PortraitJobInputs<'_>,
) -> Result<(), Refusal> {
    let input = find_makeup_input(pf, inputs)?;
    let route = resolve_makeup_route(pf, input)?;
    match route {
        MakeupRoute::Color { .. } => {
            required(input.mask, "makeup mask")?.validate()?;
            let mask_uv = required(input.mask_uv, "makeup mask UV")?;
            check_plane(mask_uv.len(), pixels, "makeup mask UV")?;
            let writes = required(input.mask_uv_writes, "makeup mask UV")?;
            check_plane(writes.len(), pixels, "makeup mask UV")?;
        }
        MakeupRoute::ImageMasked { .. } => {
            required(input.mask, "makeup mask")?.validate()?;
            let mask_uv = required(input.mask_uv, "makeup mask UV")?;
            check_plane(mask_uv.len(), pixels, "makeup mask UV")?;
            let mask_writes = required(input.mask_uv_writes, "makeup mask UV")?;
            check_plane(mask_writes.len(), pixels, "makeup mask UV")?;
            required(input.material, "makeup material")?.validate()?;
            let material_uv = required(input.material_uv, "makeup material UV")?;
            check_plane(material_uv.len(), pixels, "makeup material UV")?;
            let material_writes =
                required(input.material_uv_writes, "makeup material UV writes")?;
            check_plane(
                material_writes.len(),
                pixels,
                "makeup material UV writes",
            )?;
        }
        MakeupRoute::ImageUnmasked { .. } => {
            required(input.material, "makeup material")?.validate()?;
            let material_uv = required(input.material_uv, "makeup material UV")?;
            check_plane(material_uv.len(), pixels, "makeup material UV")?;
            let writes = required(input.material_uv_writes, "makeup material UV writes")?;
            check_plane(writes.len(), pixels, "makeup material UV writes")?;
            if input.mask.is_some() || input.mask_uv.is_some() {
                return Err(Refusal::ValueOutOfRange("PF111 mask must be absent"));
            }
        }
    }
    Ok(())
}

struct ResolvedRadiance<'a> {
    skin: &'a [u8],
    parsing_rgba: &'a [u8],
    face_rgba: &'a [u8],
    value_mean: f32,
}

fn resolve_radiance_inputs<'a>(
    current: &Image,
    job: Option<RadianceInputs<'a>>,
) -> Result<ResolvedRadiance<'a>, Refusal> {
    let job = required(job, "PF90004_SAME_JOB_INPUTS")?;
    let frame_width = required(job.frame_width, "PF90004_FRAME_WIDTH")?;
    let frame_height = required(job.frame_height, "PF90004_FRAME_HEIGHT")?;
    if frame_width != current.width || frame_height != current.height {
        return Err(Refusal::ShapeMismatch("PF90004_FRAME_SIZE"));
    }
    let skin = required(job.skin_mask, "PF90004_SKIN_MASK")?;
    let parsing_rgba = required(job.parsing_rgba, "PF90004_PARSING_RGBA")?;
    let face_rgba = required(job.face_rgba, "PF90004_FACE_RGBA")?;
    let value_mean = required(job.value_mean, "PF90004_VALUE_MEAN")?;
    let roi = required(job.roi, "PF90004_ROI")?;
    if roi != NormRect::FULL {
        return Err(Refusal::ValueOutOfRange("PF90004_ROI"));
    }
    let pixels = current.pixel_count();
    check_plane(skin.len(), pixels, "skin_mask")?;
    check_plane(parsing_rgba.len(), pixels.saturating_mul(4), "parsing_rgba")?;
    check_plane(face_rgba.len(), pixels.saturating_mul(4), "face_rgba")?;
    if !value_mean.is_finite() || !(0.0..=255.0).contains(&value_mean) {
        return Err(Refusal::ValueOutOfRange("PF90004_VALUE_MEAN"));
    }
    Ok(ResolvedRadiance {
        skin,
        parsing_rgba,
        face_rgba,
        value_mean,
    })
}

fn validate_effect_inputs(
    pf: u32,
    current: &Image,
    region: NormRect,
    extra: &BTreeMap<String, f32>,
    skin_mask: Option<&[u8]>,
    inputs: &PortraitJobInputs<'_>,
    hlf_auto: Option<&HlfAutoInputs>,
) -> Result<(), Refusal> {
    let pixels = current.pixel_count();
    if pf == 90004 {
        check_degree01(
            extra.get("pc:90004").copied().unwrap_or_default(),
            "degree",
        )?;
        if inputs.radiance.is_none() {
            let automatic = inputs.masks.pf90004_parser_inputs(
                &current.pixels,
                current.width,
                current.height,
                region,
            )?;
            if automatic.parsing_rgba.is_none() {
                return Err(Refusal::MissingInput("PF90004_PARSING_RGBA"));
            }
        }
        resolve_radiance_inputs(current, inputs.radiance)?;
        return Ok(());
    }
    if matches!(
        pf,
        8068 | 8069 | 8070 | 21015 | 30096 | 90005 | 90007
    ) {
        required(skin_mask, "skin mask")?;
        match pf {
            21015 if tint_green().is_none() || tint_magenta().is_none() => {
                return Err(Refusal::MissingInput("skin tint LUT"));
            }
            30096 if cold_lut().is_none() || warm_lut().is_none() => {
                return Err(Refusal::MissingInput("skin temperature LUT"));
            }
            90005 if whiten_lut().is_none() => {
                return Err(Refusal::MissingInput("whiten LUT"));
            }
            90007 if ruddy_lut().is_none() => {
                return Err(Refusal::MissingInput("ruddy LUT"));
            }
            _ => {}
        }
        return Ok(());
    }

    match pf {
        pf if B3_MAKEUP_ROWS.contains(&pf) => {
            validate_makeup_inputs(pf, pixels, inputs)?
        }
        108 => {
            check_degree01(extra.get("pc:108").copied().unwrap_or_default(), "degree")?;
            let teeth = required(inputs.teeth, "teeth mask")?;
            if teeth.job_id == 0 {
                return Err(Refusal::ValueOutOfRange("PF108 producer job id"));
            }
            if teeth.mask_source != TeethMaskSource::FaceParsingV2Id2G
                || teeth.mask_format != TeethMaskFormat::U8
            {
                return Err(Refusal::ValueOutOfRange("PF108 mask identity"));
            }
            let expected_mask = (teeth.mask_width as usize)
                .checked_mul(teeth.mask_height as usize)
                .ok_or(Refusal::ShapeMismatch("PF108 mask dimensions"))?;
            if expected_mask != pixels || teeth.mask_g.len() != expected_mask {
                return Err(Refusal::ShapeMismatch("PF108 mask dimensions"));
            }
            if teeth.mask_sha256 != sha256_32(teeth.mask_g) {
                return Err(Refusal::ValueOutOfRange("PF108 mask digest"));
            }
            if teeth.whiten_lut_identity != PF108_WHITEN_LUT_IDENTITY
                || teeth.bright_lut_identity != PF108_BRIGHT_LUT_IDENTITY
            {
                return Err(Refusal::ValueOutOfRange("PF108 LUT identity"));
            }
            check_plane(teeth.mask_g.len(), pixels, "teeth mask")?;
            let whiten = required(inputs.assets.teeth_whiten_lut, "teeth whiten LUT")?;
            let bright = required(inputs.assets.teeth_bright_lut, "teeth bright LUT")?;
            check_plane(whiten.len(), LUT_BYTES, "teeth whiten LUT")?;
            check_plane(bright.len(), LUT_BYTES, "teeth bright LUT")?;
            check_ui_percent(teeth.bright_ui, "teeth bright UI")?;
            check_ui_percent(teeth.deyellow_ui, "teeth deyellow UI")?;
        }
        30025 => {
            check_degree01(
                extra.get("pc:30025").copied().unwrap_or_default(),
                "PF30025 degree",
            )?;
            let pf30025 = required(inputs.pf30025, "pf30025 repair_bgra")?;
            if inputs.job_id() != Some(pf30025.receipt.job_id) {
                return Err(Refusal::MissingInput("PF30025_SAME_JOB_RECEIPT"));
            }
            if pf30025.receipt.width != current.width
                || pf30025.receipt.height != current.height
                || pf30025.receipt.source_stride != current.width.saturating_mul(4)
            {
                return Err(Refusal::ShapeMismatch("PF30025 source surface"));
            }
            if pf30025.receipt.source_sha256 != sha256_32(&current.pixels) {
                return Err(Refusal::MissingInput("PF30025 source surface"));
            }
            if pf30025.repair_780929437_bgra.is_empty() {
                return Err(Refusal::MissingInput("pf30025 repair_bgra"));
            }
            let bytes = pixels
                .checked_mul(4)
                .ok_or(Refusal::ShapeMismatch("pf30025 repair_bgra"))?;
            check_plane(
                pf30025.repair_780929437_bgra.len(),
                bytes,
                "pf30025 repair_bgra",
            )?;
            if !PF30025_MODEL_IDS.contains(&pf30025.receipt.model_id) {
                return Err(Refusal::UnknownEffect(pf30025.receipt.model_id));
            }
            check_f32_plane(
                pf30025.gate_3204548346,
                pixels,
                "PF30025_3204548346",
            )?;
            check_f32_plane(pf30025.teeth_mask, pixels, "pf30025 teeth mask")?;
            if pf30025.receipt.repair_sha256 != sha256_32(pf30025.repair_780929437_bgra)
                || pf30025.receipt.gate_sha256 != sha256_32(f32_bytes(pf30025.gate_3204548346))
            {
                return Err(Refusal::ValueOutOfRange("PF30025 readback receipt"));
            }
            if !pf30025.roi.inside(current.width, current.height) {
                return Err(Refusal::ValueOutOfRange("pf30025 roi"));
            }
        }
        21026 => {
            check_degree01(
                extra.get("pc:21026").copied().unwrap_or_default(),
                "strength",
            )?;
            let beard = required(inputs.beard, "mask_a")?;
            check_plane(beard.mask_a.len(), pixels, "mask_a")?;
            check_plane(beard.mask_b.len(), pixels, "mask_b")?;
            beard.repair_rgb.validate()?;
            if (beard.repair_rgb.width, beard.repair_rgb.height) != (current.width, current.height)
            {
                return Err(Refusal::ShapeMismatch("repair RGB"));
            }
        }
        90002 => {
            check_degree01(extra.get("pc:90002").copied().unwrap_or_default(), "degree")?;
            let face = required(inputs.moisture_face, "moisture face inputs")?;
            if (face.width, face.height) != (current.width, current.height) {
                return Err(Refusal::ShapeMismatch("moisture face frame"));
            }
            check_plane(face.mask_rgba.len(), pixels.saturating_mul(4), "mask_rgba")?;
            check_plane(face.blur_alpha.len(), pixels, "blur_alpha")?;
            check_plane(
                face.skin_map.len(),
                (face.skin_map_width as usize)
                    .checked_mul(face.skin_map_height as usize)
                    .ok_or(Refusal::ShapeMismatch("skin_map"))?,
                "skin_map",
            )?;
            if face.hardlight.is_empty() {
                return Err(Refusal::MissingInput("hardlight LUT"));
            }
            if face.beauty.is_empty() {
                return Err(Refusal::MissingInput("beauty LUT"));
            }
            if !face.pivot.is_finite() || !face.gain.is_finite() {
                return Err(Refusal::ValueOutOfRange("pivot/gain"));
            }
        }
        90010 => {
            check_degree01(extra.get("pc:90010").copied().unwrap_or_default(), "degree")?;
            let body = required(inputs.moisture_body, "moisture body inputs")?;
            if (body.width, body.height) != (current.width, current.height) {
                return Err(Refusal::ShapeMismatch("moisture body frame"));
            }
            check_plane(body.weight_skin.len(), pixels, "weight_skin")?;
            check_plane(body.weight_dog.len(), pixels, "weight_dog")?;
            if body.beauty.is_empty() {
                return Err(Refusal::MissingInput("beauty LUT"));
            }
            check_degree01(body.degree, "captured degree")?;
            check_roi(current.width, current.height, body.roi)?;
        }
        90116 => {
            check_degree01(extra.get("pc:90116").copied().unwrap_or_default(), "degree")?;
            let eye_bag = required(inputs.eye_bag, "aligned_color_field_rgb_f32")?;
            if eye_bag.ticket == 0 || inputs.job_id() != Some(eye_bag.ticket) {
                return Err(Refusal::MissingInput("PF90116_SAME_JOB_TICKET"));
            }
            if eye_bag.model_id != aurora_portrait::eye_bag_alignment::MODEL_ID {
                return Err(Refusal::ValueOutOfRange("PF90116_MODEL_ID"));
            }
            if (eye_bag.width, eye_bag.height) != (current.width, current.height) {
                return Err(Refusal::ShapeMismatch("aligned_color_field_rgb_f32"));
            }
            check_plane(
                eye_bag.aligned_color_field_rgb_f32.len(),
                pixels.saturating_mul(3),
                "aligned_color_field_rgb_f32",
            )?;
            check_plane(
                eye_bag.aligned_response_mask.len(),
                pixels,
                "aligned_response_mask",
            )?;
            if eye_bag.color_units
                != aurora_portrait::eye_bag_alignment::EyeBagColorUnits::SourceBytes
            {
                return Err(Refusal::MissingInput("PF90116_COLOR_TARGET_UNITS"));
            }
            match (
                inputs.receipt.is_some(),
                eye_bag.geometry_class,
                eye_bag.geometry_receipt,
            ) {
                (true, aurora_portrait::eye_bag_alignment::EyeBagGeometryClass::Candidate, _) => {
                    return Err(Refusal::MissingInput("PF90116_GEOMETRY_RECEIPT"));
                }
                (_, aurora_portrait::eye_bag_alignment::EyeBagGeometryClass::Candidate, Some(_)) => {
                    return Err(Refusal::ValueOutOfRange("PF90116_GEOMETRY_RECEIPT"));
                }
                (_, aurora_portrait::eye_bag_alignment::EyeBagGeometryClass::Candidate, None) => {}
                (_, aurora_portrait::eye_bag_alignment::EyeBagGeometryClass::VendorSameJob, None) => {
                    return Err(Refusal::MissingInput("PF90116_GEOMETRY_RECEIPT"));
                }
                (
                    _,
                    aurora_portrait::eye_bag_alignment::EyeBagGeometryClass::VendorSameJob,
                    Some(receipt),
                ) => {
                    receipt.validate_for_stage(eye_bag.ticket)?;
                }
            }
        }
        90139 => {
            resolve_hair_smooth(
                current,
                extra.get("pc:90139").copied().unwrap_or_default(),
                inputs,
            )?;
        }
        90160 => {
            resolve_hair_gloss(
                current,
                extra.get("pc:90160").copied().unwrap_or_default(),
                inputs,
            )?;
        }
        90088 => {
            resolve_stray_hair(
                current,
                extra.get("pc:90088").copied().unwrap_or_default(),
                inputs,
            )?;
        }
        21025 | 21069 | 21070 | 21071 | 90023 | 90024 => {
            check_degree01(
                extra.get(&format!("pc:{pf}")).copied().unwrap_or_default(),
                "wrinkle ui",
            )?;
            let input = inputs.resolve_wrinkle_inputs(current)?;
            validate_wrinkle(current, input)?;
        }
        30021 => {
            check_degree01(
                extra.get("pc:30021").copied().unwrap_or_default(),
                "PF30021 ui",
            )?;
            let input = match inputs.hlf_face {
                Some(input) => input,
                None => hlf_auto
                    .ok_or(Refusal::MissingInput("PF30021_HLF_SAME_JOB_INPUTS"))?
                    .face(),
            };
            validate_hlf_face(current, input)?;
        }
        30022 => {
            check_degree01(
                extra.get("pc:30022").copied().unwrap_or_default(),
                "PF30022 ui",
            )?;
            let input = match inputs.hlf_body {
                Some(input) => input,
                None => hlf_auto
                    .ok_or(Refusal::MissingInput("PF30022_HLF_SAME_JOB_INPUTS"))?
                    .body(),
            };
            validate_hlf_body(current, input)?;
        }
        100007 => {
            check_degree01(
                extra.get("pc:100007").copied().unwrap_or_default(),
                "pf100007 degree",
            )?;
            let material = required(inputs.smile.material, "pf100007 material")?;
            let lip_mask = required(inputs.smile.lip_mask, "pf100007 lip_mask")?;
            let mouth_mask = required(inputs.smile.mouth_mask, "pf100007 mouth_mask")?;
            let model_dx = required(inputs.smile.model_dx, "pf100007 model_dx")?;
            let model_dy = required(inputs.smile.model_dy, "pf100007 model_dy")?;
            let roi = required(inputs.smile.roi, "pf100007 roi")?;
            if material.is_empty() {
                return Err(Refusal::MissingInput("pf100007 material"));
            }
            check_plane(lip_mask.len(), pixels, "pf100007 lip_mask")?;
            check_plane(mouth_mask.len(), pixels, "pf100007 mouth_mask")?;
            check_plane(model_dx.len(), pixels, "pf100007 model_dx")?;
            check_plane(model_dy.len(), pixels, "pf100007 model_dy")?;
            if roi.x > current.width
                || roi.y > current.height
                || roi.width > current.width - roi.x
                || roi.height > current.height - roi.y
            {
                return Err(Refusal::ValueOutOfRange("pf100007 roi"));
            }
            if model_dx
                .iter()
                .chain(model_dy)
                .any(|value| !value.is_finite())
            {
                return Err(Refusal::ValueOutOfRange("pf100007 displacement"));
            }
        }
        _ => {}
    }
    Ok(())
}

fn apply_cpu_effect(
    current: &Image,
    pf: u32,
    degree: f32,
    _extra: &BTreeMap<String, f32>,
    skin_mask: Option<&[u8]>,
    width: u32,
    height: u32,
    inputs: &PortraitJobInputs<'_>,
    hlf_auto: Option<&HlfAutoInputs>,
) -> Result<Image, Refusal> {
    match pf {
        pf if B3_MAKEUP_ROWS.contains(&pf) => {
            apply_makeup_cpu(current, pf, degree, inputs)
        }
        108 => apply_teeth_cpu(current, degree, inputs),
        30025 => apply_pf30025_cpu(current, degree, inputs),
        21025 | 21069 | 21070 | 21071 | 90023 | 90024 => {
            apply_wrinkle_cpu(current, pf, degree, inputs)
        }
        30021 => apply_hlf_face_cpu(current, degree, inputs, hlf_auto),
        30022 => apply_hlf_body_cpu(current, degree, inputs, hlf_auto),
        8068 => apply_skin_gloss(current, degree, skin_mask),
        8069 => apply_skin_texture(current, degree, skin_mask),
        8070 => apply_skin_clarity(current, degree, skin_mask, width, height),
        21015 => apply_skin_tint(current, degree, skin_mask),
        21026 => apply_beard_cpu(current, degree, inputs),
        30096 => apply_skin_temperature(current, degree, skin_mask),
        90002 => apply_moisture_face_cpu(current, degree, inputs),
        90004 => apply_skin_radiance(current, degree, inputs.radiance),
        90005 => apply_skin_whiten(current, degree, skin_mask),
        90007 => apply_skin_ruddy(current, degree, skin_mask),
        90010 => apply_moisture_body_cpu(current, degree, inputs),
        90116 => apply_eye_bag_cpu(current, degree, inputs),
        90139 => apply_hair_smooth_cpu(current, degree, inputs),
        90160 => apply_hair_gloss_cpu(current, degree, inputs),
        90088 => apply_stray_hair_cpu(current, degree, inputs),
        100007 => apply_smile_cpu(current, degree, inputs),
        _ => Ok(current.clone()),
    }
}

fn apply_stray_hair_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let Some(job) = resolve_stray_hair(current, degree, inputs)? else {
        return Ok(current.clone());
    };
    aurora_portrait::kernels::hair_body::stray_hair(
        current,
        job.params.ui,
        job.params.native_degree,
        job.params.native_person_area,
        job.params.native_background_area,
        Some(job.hairfly_v1),
        Some(job.raw_mask),
        job.person_mask,
        job.background_mask,
    )
}

fn apply_hair_smooth_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let (input, roi) = resolve_hair_smooth(current, degree, inputs)?;
    let hair_mask = required(input.hair_mask, "PF90139_HAIR_MASK")?;
    let mix_mask = required(input.mix_mask, "PF90139_MIX_MASK")?;
    let blend = required(input.blend_rgb, "PF90139_BLEND_SOURCE_RGB8")?;
    let detail = required(input.detail, "PF90139_DETAIL")?;
    let mut output = current.clone();
    for y in 0..roi.height as usize {
        for x in 0..roi.width as usize {
            let hair = hair_mask[y * input.hair_stride as usize + x];
            if hair == 0 {
                continue;
            }
            let mix = mix_mask[y * input.mix_stride as usize + x];
            let detail = detail[y * input.detail_stride as usize + x];
            let gate = hair as f32 * 0.003921568859368563;
            let mix_weight = mix as f32 * 0.003921568859368563 * input.mix_strength * gate;
            let detail_add = (detail as f32 - 127.5) * 4.0 * (gate * degree * 0.8);
            let offset =
                ((roi.y as usize + y) * current.width as usize + roi.x as usize + x) * 4;
            let blend_offset = y * input.blend_stride as usize + x * 3;
            for channel in 0..3 {
                let source = current.pixels[offset + channel] as f32;
                let base = if mix_weight > 0.0 {
                    source * (1.0 - mix_weight)
                        + blend[blend_offset + channel] as f32 * mix_weight
                } else {
                    source
                };
                output.pixels[offset + channel] = (base + detail_add)
                    .round_ties_even()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(output)
}

fn apply_hair_gloss_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let (input, roi) = resolve_hair_gloss(current, degree, inputs)?;
    let hair_mask = required(input.hair_mask, "PF90160_HAIR_MASK")?;
    let detail = required(input.model_detail, "PF90160_MODEL_DETAIL")?;
    let mut output = current.clone();
    for y in 0..roi.height as usize {
        for x in 0..roi.width as usize {
            let hair = hair_mask[y * input.hair_stride as usize + x];
            if hair == 0 {
                continue;
            }
            let detail = detail[y * input.detail_stride as usize + x];
            let delta = (detail as f32 - 127.5)
                * 2.0
                * hair as f32
                * 0.003921568859368563
                * input.dynamic_scale;
            let offset =
                ((roi.y as usize + y) * current.width as usize + roi.x as usize + x) * 4;
            for channel in 0..3 {
                output.pixels[offset + channel] =
                    (current.pixels[offset + channel] as f32 + delta).clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(output)
}

fn apply_wrinkle_cpu(
    current: &Image,
    pf: u32,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let input = inputs.resolve_wrinkle_inputs(current)?;
    validate_wrinkle(current, input)?;
    if p15_handles(pf) {
        let resolved = aurora_portrait::cuda_wrinkle::resolve_p15(
            aurora_portrait::cuda_wrinkle::P15Inputs {
                pf,
                source: current,
                base: input.base_full,
                residual: input.residual_full,
                mask_rgba: input.mask_rgba_full,
                skin_mask: input.skin_mask_full,
                roi: input.roi,
                ui: degree * 100.0,
                uniforms: input.uniforms,
            },
        )?;
        return skin_adjust::wrinkle_apply(
            current,
            input.base_full,
            input.residual_full,
            input.mask_rgba_full,
            input.skin_mask_full,
            resolved.roi,
            resolved.degree,
            resolved.uniforms,
        );
    }
    let degree = skin_adjust::wrinkle_degree(pf, degree * 100.0)?;
    skin_adjust::wrinkle_apply(
        current,
        input.base_full,
        input.residual_full,
        input.mask_rgba_full,
        input.skin_mask_full,
        input.roi,
        degree,
        input.uniforms,
    )
}

fn hlf_rgb_f32_to_image(current: &Image, rgb: &[f32]) -> Result<Image, Refusal> {
    check_plane(
        rgb.len(),
        current.pixel_count() * 3,
        "PF30021_HLF output",
    )?;
    let mut pixels = current.pixels.clone();
    for (index, rgb) in rgb.chunks_exact(3).enumerate() {
        let offset = index * 4;
        for channel in 0..3 {
            pixels[offset + channel] = StoreMode::TruncU8.quantize(rgb[channel]);
        }
    }
    Ok(Image {
        width: current.width,
        height: current.height,
        pixels,
    })
}

fn apply_hlf_face_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
    hlf_auto: Option<&HlfAutoInputs>,
) -> Result<Image, Refusal> {
    let input = match inputs.hlf_face {
        Some(input) => input,
        None => hlf_auto
            .ok_or(Refusal::MissingInput("PF30021_HLF_SAME_JOB_INPUTS"))?
            .face(),
    };
    validate_hlf_face(current, input)?;
    let rgb = skin_adjust::hlf_face_apply(
        input.source_rgb_full_f32,
        input.median_rgb_full_f32,
        input.gauss_rgb_full_f32,
        input.face_mask_full_f32,
        input.skin_mask_full_f32,
        current.width,
        current.height,
        input.roi,
        degree * 100.0,
    )?;
    hlf_rgb_f32_to_image(current, &rgb)
}

fn apply_hlf_body_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
    hlf_auto: Option<&HlfAutoInputs>,
) -> Result<Image, Refusal> {
    let input = match inputs.hlf_body {
        Some(input) => input,
        None => hlf_auto
            .ok_or(Refusal::MissingInput("PF30022_HLF_SAME_JOB_INPUTS"))?
            .body(),
    };
    validate_hlf_body(current, input)?;
    skin_adjust::body_hlf_apply(
        current,
        input.roi,
        input.median_rgb_full_f32,
        input.gauss_rgb_full_f32,
        input.face_mask_rgba_full_f32,
        input.skin_mask_full_f32,
        degree * 100.0,
        input.uniforms,
        input.adjust_lut.as_ref(),
    )
}

fn apply_makeup_cpu(
    current: &Image,
    pf: u32,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let input = find_makeup_input(pf, inputs)?;
    let route = resolve_makeup_route(pf, input)?;
    let layer = makeup_layer(route, input, degree)?;
    match route {
        MakeupRoute::Color { .. } => {
            color_makeup(
                current,
                required(input.mask, "makeup mask")?,
                required(input.mask_uv, "makeup mask UV")?,
                layer,
            )
            .and_then(|output| {
                gate_makeup_writes(
                    current,
                    output,
                    required(input.mask_uv_writes, "makeup mask UV")?,
                    "makeup mask UV",
                )
            })
        }
        MakeupRoute::ImageMasked { .. } => {
            image_makeup(
                current,
                required(input.material, "makeup material")?,
                required(input.mask, "makeup mask")?,
                required(input.mask_uv, "makeup mask UV")?,
                required(input.material_uv, "makeup material UV")?,
                layer,
            )
            .and_then(|output| {
                gate_makeup_writes_dual(
                    current,
                    output,
                    required(input.mask_uv_writes, "makeup mask UV")?,
                    required(input.material_uv_writes, "makeup material UV writes")?,
                )
            })
        }
        MakeupRoute::ImageUnmasked { .. } => {
            if input.mask.is_some() || input.mask_uv.is_some() {
                return Err(Refusal::ValueOutOfRange("PF111 mask must be absent"));
            }
            image_makeup_unmasked(
                current,
                required(input.material, "makeup material")?,
                required(input.material_uv, "makeup material UV")?,
                layer,
            )
            .and_then(|output| {
                gate_makeup_writes(
                    current,
                    output,
                    required(input.material_uv_writes, "makeup material UV writes")?,
                    "makeup material UV writes",
                )
            })
        }
    }
}

fn gate_makeup_writes_dual(
    source: &Image,
    mut output: Image,
    mask_writes: &[u16],
    material_writes: &[u16],
) -> Result<Image, Refusal> {
    let pixels = source.pixel_count();
    check_plane(mask_writes.len(), pixels, "makeup mask UV")?;
    check_plane(material_writes.len(), pixels, "makeup material UV writes")?;
    for (pixel, (&mask_writes, &material_writes)) in
        mask_writes.iter().zip(material_writes).enumerate()
    {
        if mask_writes == 0 || material_writes == 0 {
            let base = pixel * 4;
            output.pixels[base..base + 4].copy_from_slice(&source.pixels[base..base + 4]);
        }
    }
    Ok(output)
}

fn gate_makeup_writes(
    source: &Image,
    mut output: Image,
    writes: &[u16],
    name: &'static str,
) -> Result<Image, Refusal> {
    check_plane(writes.len(), source.pixel_count(), name)?;
    for (pixel, writes) in writes.iter().enumerate() {
        if *writes == 0 {
            let base = pixel * 4;
            output.pixels[base..base + 4].copy_from_slice(&source.pixels[base..base + 4]);
        }
    }
    Ok(output)
}

fn apply_teeth_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let teeth = required(inputs.teeth, "teeth mask")?;
    let whiten = required(inputs.assets.teeth_whiten_lut, "teeth whiten LUT")?;
    let bright = required(inputs.assets.teeth_bright_lut, "teeth bright LUT")?;
    Ok(aurora_portrait::kernels::face_detail::teeth_whitening(
        current,
        teeth.mask_g,
        whiten,
        bright,
        degree * 100.0,
        teeth.bright_ui,
        teeth.deyellow_ui,
    )?
    .output)
}

fn apply_pf30025_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let pf30025 = required(inputs.pf30025, "pf30025 repair_bgra")?;
    aurora_portrait::kernels::face_detail::tooth_repair_same_job(
        current,
        Some(ToothRepairProducer {
            repair_780929437_bgra: pf30025.repair_780929437_bgra,
            gate_3204548346: pf30025.gate_3204548346,
            roi: pf30025.roi,
        }),
        degree * 100.0,
    )
}

fn apply_beard_cpu(
    current: &Image,
    strength: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let beard = required(inputs.beard, "mask_a")?;
    aurora_portrait::kernels::beard::apply(
        current,
        beard.repair_rgb,
        beard.mask_a,
        beard.mask_b,
        strength.clamp(0.0, 1.0),
    )
}

fn apply_moisture_face_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let input = required(inputs.moisture_face, "moisture face inputs")?;
    let params = FaceParams {
        ui: degree * 100.0,
        pivot: input.pivot,
        gain: input.gain,
    };
    let skin_map = SkinMap {
        width: input.skin_map_width as usize,
        height: input.skin_map_height as usize,
        samples: input.skin_map,
    };
    aurora_portrait::kernels::moisture::face_apply(
        current,
        input.mask_rgba,
        input.blur_alpha,
        &params,
        input.channel,
        &skin_map,
        &Lut1D {
            samples: input.hardlight,
        },
        &Lut1D {
            samples: input.beauty,
        },
    )
}

fn apply_moisture_body_cpu(
    current: &Image,
    enabled: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let input = required(inputs.moisture_body, "moisture body inputs")?;
    let degree = if enabled > 0.0 { input.degree } else { 0.0 };
    aurora_portrait::kernels::moisture::body_apply(
        current,
        input.weight_skin,
        input.weight_dog,
        &Lut1D {
            samples: input.beauty,
        },
        degree,
        input.roi,
    )
}

fn apply_eye_bag_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let input = required(inputs.eye_bag, "aligned_color_field_rgb_f32")?;
    aurora_portrait::kernels::eye_bag::compose_field_image_pf90116(
        current,
        input.aligned_color_field_rgb_f32,
        input.aligned_response_mask,
        degree,
    )
}

fn apply_smile_cpu(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
) -> Result<Image, Refusal> {
    let material = required(inputs.smile.material, "pf100007 material")?;
    let (active, degree) =
        smile_warp::pf100007_gate(degree * 100.0, Some(material)).map_err(smile_refusal)?;
    if !active {
        return Ok(current.clone());
    }
    let lip_mask = required(inputs.smile.lip_mask, "pf100007 lip_mask")?;
    let mouth_mask = required(inputs.smile.mouth_mask, "pf100007 mouth_mask")?;
    let model_dx = required(inputs.smile.model_dx, "pf100007 model_dx")?;
    let model_dy = required(inputs.smile.model_dy, "pf100007 model_dy")?;
    let roi = required(inputs.smile.roi, "pf100007 roi")?;
    let pixels = smile_warp::pf100007_apply(
        &current.pixels,
        current.width,
        current.height,
        degree,
        lip_mask,
        mouth_mask,
        model_dx,
        model_dy,
        roi,
    )
    .map_err(smile_refusal)?;
    Image::new(current.width, current.height, pixels).ok_or(Refusal::ShapeMismatch("pf100007 bgra"))
}

fn smile_refusal(error: smile_warp::Refusal) -> Refusal {
    match error {
        smile_warp::Refusal::MissingInput(name) => Refusal::MissingInput(name),
        smile_warp::Refusal::ShapeMismatch(name) => Refusal::ShapeMismatch(name),
        smile_warp::Refusal::ValueOutOfRange(name) => Refusal::ValueOutOfRange(name),
        smile_warp::Refusal::VendorConsumerOpen(pf) => Refusal::NotImplemented(pf),
    }
}

fn resolve_suit_passes<'a>(
    passes: Option<&'a [SuitPassInput<'a>]>,
) -> Result<Vec<SuitPass<'a>>, Refusal> {
    let passes = required(passes, "suit passes")?;
    if passes.is_empty() {
        return Err(Refusal::MissingInput("suit passes"));
    }
    let mut resolved = Vec::with_capacity(passes.len());
    for pass in passes {
        let layer = required(pass.layer, "suit layer")?;
        let (kind, unmasked) = match &pass.kind {
            SuitPassKindInput::Color => (SuitPassKind::Color, false),
            SuitPassKindInput::Image {
                material: Some(material),
                material_uv: Some(material_uv),
            } => (
                SuitPassKind::Image {
                    material,
                    material_uv,
                },
                false,
            ),
            SuitPassKindInput::ImageUnmasked {
                material: Some(material),
                material_uv: Some(material_uv),
            } => (
                SuitPassKind::Image {
                    material,
                    material_uv,
                },
                true,
            ),
            SuitPassKindInput::Image { material: None, .. } => {
                return Err(Refusal::MissingInput("suit material"));
            }
            SuitPassKindInput::Image {
                material_uv: None, ..
            } => return Err(Refusal::MissingInput("suit material uv")),
            SuitPassKindInput::ImageUnmasked { material: None, .. } => {
                return Err(Refusal::MissingInput("suit material"));
            }
            SuitPassKindInput::ImageUnmasked {
                material_uv: None, ..
            } => return Err(Refusal::MissingInput("suit material uv")),
            SuitPassKindInput::Unavailable(name) => {
                return Err(Refusal::MissingInput(name));
            }
        };
        let (mask, mask_uv) = if unmasked {
            if pass.mask.is_some() || pass.mask_uv.is_some() {
                return Err(Refusal::ValueOutOfRange(
                    "suit unmasked mask must be absent",
                ));
            }
            static CARRIER: OnceLock<(Image, [[f32; 2]; 1])> = OnceLock::new();
            let carrier =
                CARRIER.get_or_init(|| (Image::solid(1, 1, [255; 4]), [[0.5, 0.5]]));
            (&carrier.0, &carrier.1[..])
        } else {
            (
                required(pass.mask, "suit mask")?,
                required(pass.mask_uv, "suit mask uv")?,
            )
        };
        resolved.push(SuitPass {
            kind,
            mask,
            mask_uv,
            layer,
            shelter: pass.shelter,
            region: pass.region,
        });
    }
    Ok(resolved)
}

fn fold_suit_layer(layer: MakeupLayer, shelter: f32, region: f32) -> Result<MakeupLayer, Refusal> {
    if !shelter.is_finite() || !(0.0..=1.0).contains(&shelter) {
        return Err(Refusal::ValueOutOfRange("suit shelter"));
    }
    if !region.is_finite() || !(0.0..=1.0).contains(&region) {
        return Err(Refusal::ValueOutOfRange("suit region"));
    }
    let degree = layer.degree01 * shelter * region;
    check_degree01(degree, "suit degree")?;
    Ok(MakeupLayer {
        degree01: degree,
        ..layer
    })
}

fn apply_cpu_suit_pass(current: &Image, pass: &SuitPass<'_>) -> Result<Image, Refusal> {
    let layer = fold_suit_layer(pass.layer, pass.shelter, pass.region)?;
    if pass.layer.mask_channel == MakeupMaskChannel::A {
        return match &pass.kind {
            SuitPassKind::Image {
                material,
                material_uv,
            } => image_makeup_unmasked(current, material, material_uv, layer),
            SuitPassKind::Color => Err(Refusal::ValueOutOfRange("suit unmasked color")),
        };
    }
    match &pass.kind {
        SuitPassKind::Color => color_makeup(current, pass.mask, pass.mask_uv, layer),
        SuitPassKind::Image {
            material,
            material_uv,
        } => image_makeup(
            current,
            material,
            pass.mask,
            pass.mask_uv,
            material_uv,
            layer,
        ),
    }
}

#[cfg(windows)]
fn apply_gpu_suit_pass(
    current: &Image,
    pass: &SuitPass<'_>,
    gpu: &aurora_portrait::cuda_makeup::MakeupLayer,
) -> Result<(Image, Vec<Dispatch>), CudaError> {
    let layer =
        fold_suit_layer(pass.layer, pass.shelter, pass.region).map_err(CudaError::Refused)?;
    if pass.layer.mask_channel == MakeupMaskChannel::A {
        return Err(CudaError::Refused(Refusal::MissingInput(
            "CUDA suit unmasked pass",
        )));
    }
    let params = layer.params_of().map_err(CudaError::Refused)?;
    match &pass.kind {
        SuitPassKind::Image {
            material,
            material_uv,
        } => cuda_router::suit_pass(
            current,
            material,
            pass.mask,
            pass.mask_uv,
            material_uv,
            params,
            gpu,
        ),
        SuitPassKind::Color => {
            let [r, g, b, _] = layer.mix_color;
            let material = Image::solid(1, 1, [r, g, b, 255]);
            let flat_uv = vec![[0.5f32, 0.5f32]; current.pixel_count()];
            cuda_router::suit_pass(
                current,
                &material,
                pass.mask,
                pass.mask_uv,
                &flat_uv,
                params,
                gpu,
            )
        }
    }
}

fn apply_suit_effect(
    current: &Image,
    passes: Option<&[SuitPassInput<'_>]>,
) -> (Option<Image>, PortraitEffectRun) {
    let mut run = PortraitEffectRun {
        pf: 90033,
        backend: PortraitBackend::Cpu,
        dispatches: Vec::new(),
        fallback_reason: None,
        skipped: true,
        executed: false,
        refusal: None,
        skip_reason: None,
    };
    let resolved = match resolve_suit_passes(passes) {
        Ok(passes) => passes,
        Err(refusal) => {
            run.backend = PortraitBackend::Skipped;
            run.fallback_reason = Some(format!("input refused: {refusal:?}"));
            run.skip_reason = Some(PortraitSkipReason::InputRefused(refusal.clone()));
            run.refusal = Some(refusal);
            return (None, run);
        }
    };
    let mut next = current.clone();
    let mut reasons = Vec::new();
    let mut all_cuda = true;

    #[cfg(windows)]
    let gpu = match forced_cuda_failure() {
        Some(error) => {
            reasons.push(format!("{error:?}"));
            None
        }
        None => match aurora_portrait::cuda_makeup::MakeupLayer::open() {
            Ok(gpu) => Some(gpu),
            Err(error) => {
                reasons.push(format!("{error:?}"));
                None
            }
        },
    };

    for (index, pass) in resolved.iter().enumerate() {
        #[cfg(windows)]
        if let Some(gpu) = gpu.as_ref() {
            match apply_gpu_suit_pass(&next, pass, gpu) {
                Ok((image, dispatches)) if !dispatches.is_empty() => {
                    next = image;
                    run.dispatches.extend(dispatches);
                    continue;
                }
                Ok(_) => reasons.push(format!("pass {index}: CUDA returned no dispatch")),
                Err(error) => reasons.push(format!("pass {index}: {error:?}")),
            }
        }
        all_cuda = false;
        match apply_cpu_suit_pass(&next, pass) {
            Ok(image) => next = image,
            Err(refusal) => {
                run.refusal = Some(refusal.clone());
                reasons.push(format!("pass {index} CPU: {refusal:?}"));
                run.fallback_reason = (!reasons.is_empty()).then(|| reasons.join("; "));
                return (None, run);
            }
        }
    }
    #[cfg(not(windows))]
    {
        all_cuda = false;
    }

    run.skipped = false;
    run.executed = true;
    run.backend = if all_cuda {
        PortraitBackend::Cuda
    } else {
        PortraitBackend::Cpu
    };
    run.fallback_reason = (!reasons.is_empty()).then(|| reasons.join("; "));
    (Some(next), run)
}

#[cfg(all(windows, test))]
thread_local! {
    static FORCE_CUDA_FAILURE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FORCE_CUDA_EMPTY_DISPATCH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(windows)]
fn forced_cuda_failure() -> Option<CudaError> {
    let render_backend_cpu = std::env::var("AURORA_PORTRAIT_RENDER_BACKEND")
        .is_ok_and(|value| value.trim().eq_ignore_ascii_case("cpu"));
    if render_backend_cpu {
        return Some(CudaError::Unavailable(
            "AURORA_PORTRAIT_RENDER_BACKEND=cpu",
        ));
    }
    if std::env::var_os("AURORA_PORTRAIT_FORCE_CPU").is_some() {
        return Some(CudaError::Unavailable("AURORA_PORTRAIT_FORCE_CPU"));
    }
    #[cfg(test)]
    if FORCE_CUDA_FAILURE.with(std::cell::Cell::get) {
        return Some(CudaError::Unavailable("test-forced CUDA failure"));
    }
    None
}

#[cfg(all(windows, test))]
pub(crate) fn with_forced_cpu<T>(run: impl FnOnce() -> T) -> T {
    struct ResetCudaFailure;

    impl Drop for ResetCudaFailure {
        fn drop(&mut self) {
            FORCE_CUDA_FAILURE.with(|flag| flag.set(false));
        }
    }

    FORCE_CUDA_FAILURE.with(|flag| flag.set(true));
    let _reset = ResetCudaFailure;
    run()
}

#[cfg(windows)]
fn forced_cuda_empty_dispatch() -> bool {
    #[cfg(test)]
    if FORCE_CUDA_EMPTY_DISPATCH.with(std::cell::Cell::get) {
        return true;
    }
    false
}

/// PF90023/PF90024 are the two P15 rows whose mask law is recovered.
///
/// PF21024 is deliberately absent: r76 s63 U3 records that its
/// `mask.r`/`mask.b` contribution was never observed.
fn p15_handles(pf: u32) -> bool {
    matches!(pf, 90023 | 90024)
}

/// The CPU lane always serves the two P15 rows; CUDA is opt-in.
#[cfg(windows)]
fn p15_cuda_enabled() -> bool {
    std::env::var_os("AURORA_PORTRAIT_P15_CUDA").is_some()
}

#[derive(Clone, Copy)]
struct StrayHairJob<'a> {
    params: StrayHairParams,
    hairfly_v1: &'a Image,
    raw_mask: &'a [u8],
    person_mask: Option<&'a [u8]>,
    background_mask: Option<&'a [u8]>,
}

fn resolve_stray_hair<'a>(
    current: &Image,
    degree: f32,
    inputs: &PortraitJobInputs<'a>,
) -> Result<Option<StrayHairJob<'a>>, Refusal> {
    check_degree01(degree, "PF90088_UI")?;
    if degree == 0.0 {
        return Ok(None);
    }
    let source = &inputs.stray_hair;
    let native_degree = required(source.native_degree, "PF90088_NATIVE_DEGREE")?;
    check_degree01(native_degree, "PF90088_NATIVE_DEGREE")?;
    let native_person_area =
        required(source.native_person_area, "PF90088_NATIVE_PERSON_AREA")?;
    check_degree01(native_person_area, "PF90088_NATIVE_PERSON_AREA")?;
    let native_background_area =
        required(source.native_background_area, "PF90088_NATIVE_BACKGROUND_AREA")?;
    check_degree01(
        native_background_area,
        "PF90088_NATIVE_BACKGROUND_AREA",
    )?;
    if native_degree == 0.0 {
        return Ok(None);
    }

    let hairfly_v1 = required(source.hairfly_v1, "PF90088_HAIRFLY_V1")?;
    hairfly_v1
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("PF90088_HAIRFLY_V1"))?;
    if (hairfly_v1.width, hairfly_v1.height) != (current.width, current.height) {
        return Err(Refusal::ShapeMismatch("PF90088_HAIRFLY_V1"));
    }
    let raw_mask = required(source.raw_mask, "PF90088_RAW_STRAY_HAIR_MASK")?;
    check_plane(
        raw_mask.len(),
        current.pixel_count(),
        "PF90088_RAW_STRAY_HAIR_MASK",
    )?;
    if let Some(mask) = source.person_mask {
        check_plane(mask.len(), current.pixel_count(), "PF90088_PERSON_MASK")?;
    }
    if let Some(mask) = source.background_mask {
        check_plane(
            mask.len(),
            current.pixel_count(),
            "PF90088_BACKGROUND_MASK",
        )?;
    }
    if raw_mask.iter().all(|&value| value == 0) {
        return Ok(None);
    }
    Ok(Some(StrayHairJob {
        params: StrayHairParams {
            ui: degree * 100.0,
            native_degree,
            native_person_area,
            native_background_area,
        },
        hairfly_v1,
        raw_mask,
        person_mask: source.person_mask,
        background_mask: source.background_mask,
    }))
}

#[cfg(windows)]
fn try_cuda_same_job_effect(
    current: &Image,
    pf: u32,
    degree: f32,
    inputs: &PortraitJobInputs<'_>,
    hlf_auto: Option<&HlfAutoInputs>,
) -> Option<Result<(Image, Vec<Dispatch>), CudaError>> {
    if let Some(error) = forced_cuda_failure() {
        return Some(Err(error));
    }
    if forced_cuda_empty_dispatch() {
        return Some(Ok((current.clone(), Vec::new())));
    }
    if let Some(result) = cuda_router::same_job(pf, current, degree, inputs) {
        return Some(result);
    }
    match pf {
        90004 => {
            let resolved =
                resolve_radiance_inputs(current, inputs.radiance).expect("validated PF90004 inputs");
            let assets = radiance_assets();
            let fixed = aurora_portrait::kernels::radiance::FixedAssets {
                lut_a: &assets.lut_a,
                lut_b: &assets.lut_b,
                gamma: &assets.gamma,
                matrices: &assets.matrices,
            };
            let gpu = match aurora_portrait::cuda_radiance::Radiance::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(
                gpu.apply(aurora_portrait::cuda_radiance::Request {
                    source: current,
                    skin_r: resolved.skin,
                    parsing_rgba: resolved.parsing_rgba,
                    face_rgba: resolved.face_rgba,
                    ui: (degree * 100.0).clamp(0.0, 100.0),
                    value_mean: resolved.value_mean,
                    assets: &fixed,
                })
                .map(|(image, dispatch)| (image, dispatch.into_iter().collect())),
            )
        }
        pf if p15_handles(pf) && p15_cuda_enabled() => {
            let input = inputs
                .resolve_wrinkle_inputs(current)
                .expect("validated P15 same-job inputs");
            validate_wrinkle(current, input).expect("validated P15 inputs");
            Some(cuda_router::p15(pf, current, degree, input))
        }
        21025 | 21069 | 21070 | 21071 => {
            let input = inputs
                .resolve_wrinkle_inputs(current)
                .expect("validated wrinkle same-job inputs");
            validate_wrinkle(current, input).expect("validated wrinkle inputs");
            let gpu = match aurora_portrait::cuda_wrinkle::WrinkleGpu::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.apply_route(
                pf,
                current,
                input.base_full,
                input.residual_full,
                input.mask_rgba_full,
                input.skin_mask_full,
                input.roi,
                degree * 100.0,
                input.uniforms,
                StoreMode::TruncU8,
            ))
        }
        21024 | 90023 | 90024 => {
            let bound = match portrait_historical_skin::gpu_bound_input(
                current,
                inputs.job_id(),
                pf,
                degree,
                inputs.historical_skin,
            ) {
                Ok(Some(bound)) => bound,
                Ok(None) => return None,
                Err(refusal) => return Some(Err(CudaError::Refused(refusal))),
            };
            let plan =
                portrait_historical_skin::plan_historical_skin_backend(pf, &bound.data);
            if plan.backend != portrait_historical_skin::HistoricalSkinPath::Gpu {
                return None;
            }
            let input = match &bound.data {
                HistoricalSkinData::Glabella(input)
                | HistoricalSkinData::EyeLines(input)
                | HistoricalSkinData::Forehead(input) => input,
                _ => {
                    return Some(Err(CudaError::Refused(Refusal::ValueOutOfRange(
                        "historical skin PF/data",
                    ))));
                }
            };
            let Some(job_id) = inputs.job_id() else {
                return Some(Err(CudaError::Refused(Refusal::MissingInput(
                    "historical skin CUDA job id",
                ))));
            };
            let gpu = match aurora_portrait::cuda_wrinkle::WrinkleGpu::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.apply_historical_same_job(
                job_id,
                pf,
                current,
                &input.base,
                &input.residual,
                &input.mask_rgba,
                &input.skin_mask,
                input.roi,
                degree,
                input.uniforms,
                StoreMode::TruncU8,
            ))
        }
        30021 => {
            let input = match inputs.hlf_face {
                Some(input) => input,
                None => hlf_auto.expect("validated PF30021 HLF inputs").face(),
            };
            validate_hlf_face(current, input).expect("validated PF30021 HLF inputs");
            let gpu = match aurora_portrait::cuda_hlf::Hlf::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(
                gpu.face_apply(
                    input.source_rgb_full_f32,
                    input.median_rgb_full_f32,
                    input.gauss_rgb_full_f32,
                    input.face_mask_full_f32,
                    input.skin_mask_full_f32,
                    current.width,
                    current.height,
                    input.roi,
                    degree * 100.0,
                )
                .and_then(|(rgb, dispatches)| {
                    let image =
                        hlf_rgb_f32_to_image(current, &rgb).map_err(CudaError::Refused)?;
                    Ok((image, dispatches))
                }),
            )
        }
        30022 => {
            let input = match inputs.hlf_body {
                Some(input) => input,
                None => hlf_auto.expect("validated PF30022 HLF inputs").body(),
            };
            validate_hlf_body(current, input).expect("validated PF30022 HLF inputs");
            let gpu = match aurora_portrait::cuda_hlf::Hlf::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.body_apply(
                current,
                input.roi,
                input.median_rgb_full_f32,
                input.gauss_rgb_full_f32,
                input.face_mask_rgba_full_f32,
                input.skin_mask_full_f32,
                degree * 100.0,
                input.uniforms,
                input.adjust_lut.as_ref(),
            ))
        }
        pf if B3_MAKEUP_ROWS.contains(&pf) => {
            let input =
                find_makeup_input(pf, inputs).expect("validated makeup same-job input");
            let route =
                resolve_makeup_route(pf, input).expect("validated makeup same-job route");
            let gpu = match aurora_portrait::cuda_makeup::MakeupLayer::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            let result = gpu.apply_route(
                pf,
                current,
                RouteInputs {
                    mix_color: input.mix_color,
                    opacity_percent: input.opacity_percent,
                    degree01: degree,
                    filter_type: input.filter_type,
                    material: input.material,
                    material_uv: input.material_uv,
                    mask: input.mask,
                    mask_uv: input.mask_uv,
                },
            );
            let (writes, name) = match route {
                MakeupRoute::Color { .. } => (
                    input.mask_uv_writes.expect("validated makeup mask UV"),
                    "makeup mask UV",
                ),
                MakeupRoute::ImageUnmasked { .. } => (
                    input
                        .material_uv_writes
                        .expect("validated PF111 material UV writes"),
                    "makeup material UV writes",
                ),
                MakeupRoute::ImageMasked { .. } => (
                    input.mask_uv_writes.expect("validated makeup mask UV"),
                    "makeup mask UV",
                ),
            };
            Some(result.and_then(|(next, dispatches)| {
                let next = match route {
                    MakeupRoute::ImageMasked { .. } => gate_makeup_writes_dual(
                        current,
                        next,
                        writes,
                        input
                            .material_uv_writes
                            .expect("validated makeup material UV writes"),
                    )
                    .map_err(CudaError::Refused)?,
                    _ => gate_makeup_writes(current, next, writes, name)
                        .map_err(CudaError::Refused)?,
                };
                Ok((next, dispatches))
            }))
        }
        30025 => {
            let input = inputs.pf30025.expect("validated PF30025 repair input");
            let gpu = match aurora_portrait::cuda_pf30025::ToothRepair::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.apply(&aurora_portrait::cuda_pf30025::Job {
                source: current,
                repair_780929437_bgra: input.repair_780929437_bgra,
                gate_3204548346: input.gate_3204548346,
                roi: input.roi,
                ui: degree * 100.0,
            }))
        }
        21026 => {
            let beard = inputs.beard.expect("validated PF21026 beard input");
            let gpu = match aurora_portrait::cuda_beard::Beard::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.apply(
                current,
                beard.repair_rgb,
                beard.mask_a,
                beard.mask_b,
                degree.clamp(0.0, 1.0),
            ))
        }
        90002 => {
            let input = inputs
                .moisture_face
                .expect("validated PF90002 moisture input");
            let params = FaceParams {
                ui: degree * 100.0,
                pivot: input.pivot,
                gain: input.gain,
            };
            let skin_map = SkinMap {
                width: input.skin_map_width as usize,
                height: input.skin_map_height as usize,
                samples: input.skin_map,
            };
            let gpu = match aurora_portrait::cuda_moisture::Moisture::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.face_apply(
                current,
                input.mask_rgba,
                input.blur_alpha,
                &params,
                input.channel,
                &skin_map,
                &Lut1D {
                    samples: input.hardlight,
                },
                &Lut1D {
                    samples: input.beauty,
                },
            ))
        }
        90010 => {
            let input = inputs
                .moisture_body
                .expect("validated PF90010 moisture input");
            let native_degree = if degree > 0.0 { input.degree } else { 0.0 };
            let gpu = match aurora_portrait::cuda_moisture::Moisture::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.body_apply(
                current,
                input.weight_skin,
                input.weight_dog,
                &Lut1D {
                    samples: input.beauty,
                },
                native_degree,
                input.roi,
            ))
        }
        90116 => {
            let input = inputs.eye_bag.expect("validated PF90116 eye-bag input");
            let gpu = match aurora_portrait::cuda_eye_bag::EyeBag::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.compose_field(
                current,
                input.aligned_color_field_rgb_f32,
                input.aligned_response_mask,
                degree,
            ))
        }
        90139 => {
            let (hair, roi) =
                resolve_hair_smooth(current, degree, inputs).expect("validated PF90139 inputs");
            let gpu = match HairDetail::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(
                gpu.smooth(
                    current,
                    roi,
                    SmoothParams {
                        mix_strength: hair.mix_strength,
                        detail_strength: degree,
                    },
                    Some(SmoothProducer {
                        ticket: hair.ticket,
                        hair_mask: hair.hair_mask.expect("validated PF90139 hair mask"),
                        mix_mask: hair.mix_mask.expect("validated PF90139 mix mask"),
                        blend_rgb: hair.blend_rgb.expect("validated PF90139 blend RGB8"),
                        detail: hair.detail.expect("validated PF90139 detail"),
                        hair_stride: hair.hair_stride,
                        mix_stride: hair.mix_stride,
                        blend_stride: hair.blend_stride,
                        detail_stride: hair.detail_stride,
                    }),
                )
                .map(|(image, dispatch)| (image, dispatch.into_iter().collect())),
            )
        }
        90160 => {
            let (hair, roi) =
                resolve_hair_gloss(current, degree, inputs).expect("validated PF90160 inputs");
            let gpu = match HairDetail::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(
                gpu.gloss(
                    current,
                    roi,
                    hair.dynamic_scale,
                    Some(GlossProducer {
                        ticket: hair.ticket,
                        hair_mask: hair.hair_mask.expect("validated PF90160 hair mask"),
                        model_detail: hair.model_detail.expect("validated PF90160 model detail"),
                        hair_stride: hair.hair_stride,
                        detail_stride: hair.detail_stride,
                    }),
                )
                .map(|(image, dispatch)| (image, dispatch.into_iter().collect())),
            )
        }
        90088 => {
            let job = match resolve_stray_hair(current, degree, inputs) {
                Ok(Some(job)) => job,
                Ok(None) => return Some(Ok((current.clone(), Vec::new()))),
                Err(error) => return Some(Err(CudaError::Refused(error))),
            };
            let gpu = match aurora_portrait::cuda_stray_hair::Blend::open() {
                Ok(gpu) => gpu,
                Err(error) => return Some(Err(error)),
            };
            Some(gpu.apply(
                current,
                job.params,
                Some(job.hairfly_v1),
                Some(job.raw_mask),
                job.person_mask,
                job.background_mask,
            ))
        }
        _ => None,
    }
}

/// Product dispatch through `aurora_portrait::effect_router`.
///
/// Each arm forwards the same arguments the previous direct adapter call used,
/// so the CPU and CUDA pixels do not change. `Backend::Cuda` keeps the stage
/// rule that a CUDA result without a dispatch is not evidence: the router turns
/// that case into an error and the stage falls back to the CPU kernel.
#[cfg(windows)]
mod cuda_router {
    use super::*;
    use aurora_portrait::effect_router::{self as router, Backend, CudaEffect, RouteError};

    /// Every PF this stage dispatches through the router.
    #[allow(dead_code)]
    pub(super) const ROUTED_PFS: &[(u32, &str)] = &[
        (108, "teeth_whiten"),
        (8068, "gloss_pass"),
        (21015, "skin_tint_pf21015"),
        (90005, "skin_adjust_lut"),
        (90007, "skin_adjust_lut"),
        (21026, "beard_blend"),
        (90002, "moisture_face"),
        (90010, "moisture_body"),
        (90033, "makeup_layer"),
        (100007, "smile_pf100007"),
        (90116, "eye_bag_compose"),
    ];

    /// Router adapters that are real but must not be bound to these PFs.
    ///
    /// `portrait_map.rs` states the product meaning of each PF. An adapter
    /// whose kernel family does not match that meaning stays unbound instead of
    /// silently changing the effect.
    #[allow(dead_code)]
    pub(super) const BLOCKED_PFS: &[(u32, &str, &str)] = &[
        (
            407,
            "uvmap_type143",
            "PF407 is legLengthen (portrait_map.rs:211) and runs through portrait_body_stage's UV warp; the router adapter is a general UV colour sampler. Missing: a PF407 field_xyzw/channel/catmull producer.",
        ),
        (
            90023,
            "p15_facial_smooth",
            "PF90023 is removeEyeLines; cuda_wrinkle::WrinkleRoute resolves only 21025/21069/21070/21071. Missing: the per-PF ROI, mask channel and uniform set.",
        ),
        (
            90024,
            "p15_facial_smooth",
            "PF90024 is removeForeheadLines; same missing per-PF wrinkle route as 90023.",
        ),
        (
            21024,
            "p15_facial_smooth",
            "PF21024 is removeGlabellaLines; same missing per-PF wrinkle route as 90023.",
        ),
        (
            30009,
            "texture_blend",
            "PF30009 is hairColorUnify (kernels/hair_body.rs:441) and has no CUDA kernel; the router adapter is a different pixel primitive that happens to carry the same PF label.",
        ),
        (
            8033,
            "mesh_attribute",
            "PF8033 is smileMouthCorner; the attribute rasterizer needs dest/attr/faces, while FaceGeometry::Smile carries dx/dy/mask only. Missing: an attribute-pair producer. The smile half (smile_pf8033) is wired in portrait_face_stage.",
        ),
    ];

    /// Reachable adapters that keep their direct call because the router
    /// adapter wraps a different GPU entry point.
    #[allow(dead_code)]
    pub(super) const DIRECT_PFS: &[(u32, &str, &str)] = &[(
        30026,
        "teeth_blend",
        "The stage calls cuda_teeth_blend::apply_alignment_final_blend (Job -> AlignmentFinalBlendGpu); the router adapter wraps TeethBlend::apply(original, effect, mask). Same kernel, different entry.",
    )];

    /// Runs one adapter on the CUDA lane and flattens the router report back
    /// into the stage's `(Image, dispatches)` contract.
    fn run<E: CudaEffect>(
        input: &E::Input<'_>,
    ) -> Result<(E::Output, Vec<Dispatch>), CudaError> {
        match router::route::<E>(input, Backend::Cuda) {
            Ok(routed) => Ok((routed.output, routed.dispatches)),
            Err(RouteError::Cuda(error)) => Err(error),
            Err(RouteError::Cpu(refusal)) => Err(CudaError::Refused(refusal)),
            Err(RouteError::MissingDispatchEvidence(_)) => Err(CudaError::Unavailable(
                "effect_router: CUDA returned no dispatch",
            )),
        }
    }

    /// PF21015/90005/90007 adapter that borrows the thread-local `SkinLut`
    /// session, so routing does not reopen the PTX module on every frame.
    struct CachedSkinLut;

    enum SkinLutJob<'a> {
        Tint {
            source: &'a Image,
            mask_r: &'a [u8],
            tint_green: &'a [u8],
            tint_magenta: &'a [u8],
            ui: f32,
            gpu: &'a aurora_portrait::cuda_lut::SkinLut,
        },
        Adjust {
            source: &'a Image,
            mask_r: &'a [u8],
            lut_rgb512: &'a [u8],
            mix: f32,
            pf: u32,
            gpu: &'a aurora_portrait::cuda_lut::SkinLut,
        },
    }

    impl CudaEffect for CachedSkinLut {
        const NAME: &'static str = "skin_lut_cached";
        const PFS: &'static [u32] = &[21015, 90005, 90007];
        const KERNELS: &'static [&'static str] = &[aurora_portrait::cuda_lut::KERNEL];
        const CPU_ENTRY: &'static str = "kernels::makeup::skin_adjust_lut";

        type Input<'a> = SkinLutJob<'a>;
        type Output = Image;

        fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
            match input {
                SkinLutJob::Tint {
                    source,
                    mask_r,
                    tint_green,
                    tint_magenta,
                    ui,
                    ..
                } => aurora_portrait::kernels::face_detail::skin_tint(
                    source,
                    mask_r,
                    tint_green,
                    tint_magenta,
                    *ui,
                ),
                SkinLutJob::Adjust {
                    source,
                    mask_r,
                    lut_rgb512,
                    mix,
                    ..
                } => aurora_portrait::kernels::makeup::skin_adjust_lut(
                    source,
                    mask_r,
                    lut_rgb512,
                    *mix,
                ),
            }
        }

        fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
            match input {
                SkinLutJob::Tint {
                    source,
                    mask_r,
                    tint_green,
                    tint_magenta,
                    ui,
                    gpu,
                } => gpu
                    .tint(source, mask_r, tint_green, tint_magenta, *ui)
                    .map(|(image, dispatch)| (image, vec![dispatch])),
                SkinLutJob::Adjust {
                    source,
                    mask_r,
                    lut_rgb512,
                    mix,
                    pf,
                    gpu,
                } => gpu
                    .skin_adjust(source, mask_r, lut_rgb512, *mix, *pf)
                    .map(|(image, dispatch)| (image, vec![dispatch])),
            }
        }
    }

    /// PFs that only need a skin mask, previously handled inline by
    /// `try_cuda_effect`. This predicate is the single gate used by the
    /// dispatcher and by its tests.
    pub(super) fn skin_effect_handles(pf: u32) -> bool {
        matches!(pf, 8068 | 21015 | 90005 | 90007)
    }

    pub(super) fn skin_effect(
        pf: u32,
        current: &Image,
        degree: f32,
        mask: &[u8],
    ) -> Option<Result<(Image, Vec<Dispatch>), CudaError>> {
        if !skin_effect_handles(pf) {
            return None;
        }
        match pf {
            8068 => {
                let extra = vec![255u8; mask.len()];
                let input = router::GlossInput {
                    source: current,
                    masks_r: [mask, mask, mask],
                    extra_masks_r: [&extra, &extra, &extra],
                    native_percent: (degree * 100.0).clamp(0.0, 100.0),
                };
                Some(run::<router::Gloss>(&input))
            }
            21015 => {
                let (Some(green), Some(magenta)) = (tint_green(), tint_magenta()) else {
                    return Some(Err(CudaError::Refused(Refusal::MissingInput(
                        "skin tint LUT",
                    ))));
                };
                let ui = ((degree - 0.5) * 200.0).clamp(-100.0, 100.0);
                Some(with_skin_lut(|gpu| {
                    let input = SkinLutJob::Tint {
                        source: current,
                        mask_r: mask,
                        tint_green: green,
                        tint_magenta: magenta,
                        ui,
                        gpu,
                    };
                    run::<CachedSkinLut>(&input)
                }))
            }
            90005 | 90007 => {
                let (lut, missing) = if pf == 90005 {
                    (whiten_lut(), "whiten LUT")
                } else {
                    (ruddy_lut(), "ruddy LUT")
                };
                let Some(lut) = lut else {
                    return Some(Err(CudaError::Refused(Refusal::MissingInput(missing))));
                };
                Some(with_skin_lut(|gpu| {
                    let input = SkinLutJob::Adjust {
                        source: current,
                        mask_r: mask,
                        lut_rgb512: lut,
                        mix: degree.clamp(0.0, 1.0),
                        pf,
                        gpu,
                    };
                    run::<CachedSkinLut>(&input)
                }))
            }
            _ => None,
        }
    }

    /// One PF90033 suit pass, routed while borrowing the session's
    /// `MakeupLayer` so a multi-pass suit still opens the module once.
    struct CachedMakeup;

    struct MakeupJob<'a> {
        source: &'a Image,
        material: &'a Image,
        owner_mask: &'a Image,
        mask_uv: &'a [[f32; 2]],
        material_uv: &'a [[f32; 2]],
        params: aurora_portrait::kernels::makeup_material::ColorMaterialParams,
        gpu: &'a aurora_portrait::cuda_makeup::MakeupLayer,
    }

    impl CudaEffect for CachedMakeup {
        const NAME: &'static str = "makeup_layer_cached";
        const PFS: &'static [u32] = &[90033];
        const KERNELS: &'static [&'static str] = &[aurora_portrait::cuda_makeup::KERNEL];
        const CPU_ENTRY: &'static str = "kernels::makeup_material::apply_image_layer_dual";

        type Input<'a> = MakeupJob<'a>;
        type Output = Image;

        fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
            aurora_portrait::kernels::makeup_material::apply_image_layer_dual(
                input.source,
                input.material,
                input.owner_mask,
                input.mask_uv,
                input.material_uv,
                input.params,
            )
        }

        fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
            input.gpu.apply_image_layer_dual(
                input.source,
                input.material,
                input.owner_mask,
                input.mask_uv,
                input.material_uv,
                input.params,
            )
        }
    }

    /// Runs one suit pass through the router on the CUDA lane.
    pub(super) fn suit_pass(
        current: &Image,
        material: &Image,
        mask: &Image,
        mask_uv: &[[f32; 2]],
        material_uv: &[[f32; 2]],
        params: aurora_portrait::kernels::makeup_material::ColorMaterialParams,
        gpu: &aurora_portrait::cuda_makeup::MakeupLayer,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        let input = MakeupJob {
            source: current,
            material,
            owner_mask: mask,
            mask_uv,
            material_uv,
            params,
            gpu,
        };
        run::<CachedMakeup>(&input)
    }

    /// PFs that read their planes from `PortraitJobInputs`. This predicate is
    /// the single gate used by the dispatcher and by its tests.
    pub(super) fn same_job_handles(pf: u32) -> bool {
        matches!(pf, 108 | 21026 | 90002 | 90010 | 100007 | 90116)
    }

    pub(super) fn same_job(
        pf: u32,
        current: &Image,
        degree: f32,
        inputs: &PortraitJobInputs<'_>,
    ) -> Option<Result<(Image, Vec<Dispatch>), CudaError>> {
        if !same_job_handles(pf) {
            return None;
        }
        match pf {
            108 => {
                let teeth = inputs.teeth.expect("validated PF108 teeth input");
                let whiten = inputs
                    .assets
                    .teeth_whiten_lut
                    .expect("validated PF108 whiten LUT");
                let bright = inputs
                    .assets
                    .teeth_bright_lut
                    .expect("validated PF108 bright LUT");
                let input = router::TeethInput {
                    source: current,
                    mask_g: teeth.mask_g,
                    whiten_lut: whiten,
                    bright_lut: bright,
                    common_ui: degree * 100.0,
                    bright_ui: teeth.bright_ui,
                    deyellow_ui: teeth.deyellow_ui,
                };
                Some(run::<router::TeethWhiten>(&input))
            }
            21026 => {
                let beard = inputs.beard.expect("validated PF21026 beard input");
                let input = router::BeardInput {
                    source: current,
                    repair: beard.repair_rgb,
                    mask_a: beard.mask_a,
                    mask_b: beard.mask_b,
                    strength: degree.clamp(0.0, 1.0),
                };
                Some(run::<router::Beard>(&input))
            }
            90002 => {
                let input = inputs
                    .moisture_face
                    .expect("validated PF90002 moisture input");
                let routed = router::MoistureFaceInput {
                    source: current,
                    mask_rgba: input.mask_rgba,
                    blur_y: input.blur_alpha,
                    ui: degree * 100.0,
                    pivot: input.pivot,
                    gain: input.gain,
                    channel: input.channel,
                    skin_map_width: input.skin_map_width as usize,
                    skin_map_height: input.skin_map_height as usize,
                    skin_map: input.skin_map,
                    hardlight: input.hardlight,
                    beauty: input.beauty,
                };
                Some(run::<router::MoistureFace>(&routed))
            }
            90010 => {
                let input = inputs
                    .moisture_body
                    .expect("validated PF90010 moisture input");
                let native_degree = if degree > 0.0 { input.degree } else { 0.0 };
                let routed = router::MoistureBodyInput {
                    source: current,
                    weight_skin: input.weight_skin,
                    weight_dog: input.weight_dog,
                    beauty: input.beauty,
                    degree: native_degree,
                    roi: input.roi,
                };
                Some(run::<router::MoistureBody>(&routed))
            }
            90116 => {
                let input = inputs.eye_bag.expect("validated PF90116 eye-bag input");
                let routed = router::EyeBagInput {
                    source: current,
                    color3: input.aligned_color_field_rgb_f32,
                    response: input.aligned_response_mask,
                    degree,
                };
                Some(run::<router::EyeBag>(&routed))
            }
            100007 => {
                let material = inputs.smile.material.expect("validated PF100007 material");
                let (active, degree) = match smile_warp::pf100007_gate(
                    degree * 100.0,
                    Some(material),
                ) {
                    Ok(value) => value,
                    Err(error) => {
                        return Some(Err(CudaError::Refused(smile_refusal(error))));
                    }
                };
                if !active {
                    return Some(Ok((current.clone(), Vec::new())));
                }
                let lip_mask = inputs.smile.lip_mask.expect("validated PF100007 lip mask");
                let mouth_mask = inputs
                    .smile
                    .mouth_mask
                    .expect("validated PF100007 mouth mask");
                let model_dx = inputs.smile.model_dx.expect("validated PF100007 dx");
                let model_dy = inputs.smile.model_dy.expect("validated PF100007 dy");
                let roi = inputs.smile.roi.expect("validated PF100007 roi");
                let routed = router::SmileInput {
                    source_bgra: &current.pixels,
                    width: current.width,
                    height: current.height,
                    degree,
                    lip_mask,
                    mouth_mask,
                    model_dx,
                    model_dy,
                    roi,
                };
                Some(
                    run::<router::SmilePf100007>(&routed).and_then(|(pixels, dispatches)| {
                        let image = Image::new(current.width, current.height, pixels).ok_or(
                            CudaError::Refused(Refusal::ShapeMismatch("pf100007 bgra")),
                        )?;
                        Ok((image, dispatches))
                    }),
                )
            }
            _ => None,
        }
    }

    /// PF90023/PF90024 P15 rows: ROI-limited, degree-gated, same composite.
    struct P15Wrinkle;

    struct P15Job<'a> {
        pf: u32,
        source: &'a Image,
        base: &'a Image,
        residual: &'a Image,
        mask_rgba: &'a [u8],
        skin_mask: &'a [u8],
        roi: Roi,
        ui: f32,
        uniforms: WrinkleUniforms,
    }

    fn p15_inputs<'a>(job: &P15Job<'a>) -> aurora_portrait::cuda_wrinkle::P15Inputs<'a> {
        aurora_portrait::cuda_wrinkle::P15Inputs {
            pf: job.pf,
            source: job.source,
            base: job.base,
            residual: job.residual,
            mask_rgba: job.mask_rgba,
            skin_mask: job.skin_mask,
            roi: job.roi,
            ui: job.ui,
            uniforms: job.uniforms,
        }
    }

    impl CudaEffect for P15Wrinkle {
        const NAME: &'static str = "p15_facial_smooth";
        const PFS: &'static [u32] = &[90023, 90024];
        const KERNELS: &'static [&'static str] = &[aurora_portrait::cuda_wrinkle::KERNEL];
        const CPU_ENTRY: &'static str = "kernels::skin_adjust::wrinkle_apply";

        type Input<'a> = P15Job<'a>;
        type Output = Image;

        fn cpu(input: &Self::Input<'_>) -> Result<Image, Refusal> {
            let resolved = aurora_portrait::cuda_wrinkle::resolve_p15(p15_inputs(input))?;
            aurora_portrait::kernels::skin_adjust::wrinkle_apply(
                input.source,
                input.base,
                input.residual,
                input.mask_rgba,
                input.skin_mask,
                resolved.roi,
                resolved.degree,
                resolved.uniforms,
            )
        }

        fn cuda(input: &Self::Input<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
            aurora_portrait::cuda_wrinkle::WrinkleGpu::open()?.apply_route(
                input.pf,
                input.source,
                input.base,
                input.residual,
                input.mask_rgba,
                input.skin_mask,
                input.roi,
                input.ui,
                input.uniforms,
                StoreMode::TruncU8,
            )
        }
    }

    /// Builds one P15 job from the wrinkle producer and routes it on CUDA.
    pub(super) fn p15(
        pf: u32,
        current: &Image,
        degree: f32,
        input: WrinkleInputs<'_>,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        let job = P15Job {
            pf,
            source: current,
            base: input.base_full,
            residual: input.residual_full,
            mask_rgba: input.mask_rgba_full,
            skin_mask: input.skin_mask_full,
            roi: input.roi,
            ui: degree * 100.0,
            uniforms: input.uniforms,
        };
        run::<P15Wrinkle>(&job)
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use aurora_portrait::effect_router::{
            MeshAttributeInput, SmileMaskInput, TextureBlendInput, UvMapInput,
        };
        use aurora_portrait::kernels::texture_mask_blend::NormalBlend;
        use aurora_portrait::kernels::uvmap_color::Channel;

        fn route_cpu<E: CudaEffect>(input: &E::Input<'_>) -> E::Output {
            router::route::<E>(input, Backend::Cpu)
                .expect("router CPU lane")
                .output
        }

        #[test]
        fn routed_pfs_name_real_router_adapters() {
            assert_eq!(ROUTED_PFS.len(), 11);
            for (pf, name) in ROUTED_PFS {
                assert!(router::adapter(name).is_some(), "PF{pf} names {name}");
            }
        }

        #[test]
        fn every_routed_pf_hits_a_dispatch_gate() {
            for (pf, _) in ROUTED_PFS {
                assert!(
                    skin_effect_handles(*pf) || same_job_handles(*pf) || *pf == 90033,
                    "PF{pf} has no dispatch gate"
                );
            }
            assert!(!skin_effect_handles(9999));
            assert!(!same_job_handles(9999));
        }

        #[test]
        fn blocked_pfs_keep_their_adapter_but_are_not_dispatched() {
            for (pf, name, reason) in BLOCKED_PFS {
                assert!(router::adapter(name).is_some(), "{name} is a real adapter");
                assert!(!reason.is_empty(), "PF{pf} needs a stated gap");
                assert!(
                    !ROUTED_PFS.iter().any(|(routed, _)| routed == pf),
                    "PF{pf} must not be routed from this stage"
                );
            }
        }

        #[test]
        fn direct_only_adapter_is_not_claimed_as_routed() {
            for (pf, name, reason) in DIRECT_PFS {
                assert!(router::adapter(name).is_some(), "{name} is a real adapter");
                assert!(!reason.is_empty(), "PF{pf} needs a stated reason");
                assert!(
                    !ROUTED_PFS.iter().any(|(routed, _)| routed == pf),
                    "PF{pf} keeps its direct entry"
                );
            }
        }

        #[test]
        fn texture_blend_degree_is_forwarded() {
            let original = Image::solid(2, 2, [10, 20, 30, 255]);
            let effect = Image::solid(2, 2, [200, 210, 220, 255]);
            let mask = vec![255u8; 4];
            let run = |degree: f32| {
                route_cpu::<router::TextureBlend>(&TextureBlendInput {
                    original: &original,
                    effect: &effect,
                    mask: &mask,
                    effect_uv: None,
                    params: NormalBlend {
                        degree,
                        reverse_alpha: false,
                    },
                })
            };
            assert_eq!(run(0.0), original);
            assert_eq!(run(1.0), effect);
        }

        #[test]
        fn smile_pf8033_degree_and_displacement_are_forwarded() {
            let source = Image::new(
                2,
                1,
                vec![10, 20, 30, 255, 40, 50, 60, 255],
            )
            .expect("2x1 image");
            let dx = [1.0f32, 0.0];
            let dy = [0.0f32, 0.0];
            let mask = [255u8, 255];
            let run = |degree01: f32| {
                route_cpu::<router::SmilePf8033>(&SmileMaskInput {
                    source_bgra: &source.pixels,
                    width: 2,
                    height: 1,
                    dx: &dx,
                    dy: &dy,
                    mask: &mask,
                    degree01,
                })
            };
            assert_eq!(run(0.0), source.pixels);
            assert_ne!(run(1.0), source.pixels);
        }

        #[test]
        fn uvmap_field_is_forwarded() {
            let source =
                Image::new(2, 1, vec![10, 20, 30, 255, 40, 50, 60, 255]).expect("2x1 image");
            let low = [[0.0f32, 0.0, 0.0, 0.0]; 2];
            let high = [[1.0f32, 1.0, 1.0, 1.0]; 2];
            let run = |field: &[[f32; 4]]| {
                route_cpu::<router::UvMap>(&UvMapInput {
                    source: &source,
                    field,
                    channel: Channel::Rg,
                    catmull: None,
                })
            };
            assert_ne!(run(&low), run(&high));
        }

        #[test]
        fn mesh_attribute_rasterizes_and_reports_writes() {
            let dest = [[0.0f32, 0.0], [2.0, 0.0], [0.0, 2.0]];
            let attr = [[1.0f32, 10.0], [3.0, 30.0], [5.0, 50.0]];
            let faces = [0u32, 1, 2];
            let out = route_cpu::<router::MeshAttribute>(&MeshAttributeInput {
                width: 1,
                height: 1,
                dest: &dest,
                attr: &attr,
                faces: &faces,
            });
            assert!(out.writes[0] >= 1, "triangle must cover the pixel");
            assert!(out.values[0][0] > 0.0);
        }

        fn p15_uniforms() -> WrinkleUniforms {
            WrinkleUniforms {
                legal_alpha: 1.0,
                dark_circle_alpha: 1.0,
                diff_factor: 1.0,
                diff_factor2: 0.0,
                color_threshold_factor: 0.0,
                mix_alpha: 1.0,
                diff_min_threshold: -1.0,
                diff_max_threshold: 1.0,
                high_freq_min_threshold: -1.0,
                high_freq_max_threshold: 1.0,
            }
        }

        fn p15_job<'a>(
            pf: u32,
            source: &'a Image,
            base: &'a Image,
            residual: &'a Image,
            mask_rgba: &'a [u8],
            skin_mask: &'a [u8],
            roi: Roi,
            uniforms: WrinkleUniforms,
        ) -> P15Job<'a> {
            P15Job {
                pf,
                source,
                base,
                residual,
                mask_rgba,
                skin_mask,
                roi,
                ui: 50.0,
                uniforms,
            }
        }

        #[test]
        fn p15_handles_only_the_two_recovered_rows() {
            assert!(super::super::p15_handles(90023));
            assert!(super::super::p15_handles(90024));
            assert!(!super::super::p15_handles(21024));
            assert!(router::adapter("p15_facial_smooth").is_some());
        }

        #[test]
        fn p15_cpu_lane_matches_the_wrinkle_kernel() {
            let source = Image::solid(4, 4, [120, 130, 140, 255]);
            let base = Image::solid(4, 4, [100, 110, 120, 255]);
            let residual = Image::solid(4, 4, [60, 70, 80, 255]);
            let mask = vec![200u8; 16 * 4];
            let skin = vec![255u8; 16];
            let roi = Roi {
                x: 1,
                y: 1,
                width: 2,
                height: 2,
            };
            let uniforms = p15_uniforms();
            for pf in [90023u32, 90024] {
                let job = p15_job(
                    pf, &source, &base, &residual, &mask, &skin, roi, uniforms,
                );
                let routed = router::route::<P15Wrinkle>(&job, Backend::Cpu)
                    .expect("P15 CPU lane")
                    .output;
                let resolved =
                    aurora_portrait::cuda_wrinkle::resolve_p15(p15_inputs(&job)).unwrap();
                let direct = aurora_portrait::kernels::skin_adjust::wrinkle_apply(
                    &source,
                    &base,
                    &residual,
                    &mask,
                    &skin,
                    resolved.roi,
                    resolved.degree,
                    resolved.uniforms,
                )
                .unwrap();
                assert_eq!(routed, direct, "PF{pf} CPU bytes");
                assert_eq!(resolved.roi, roi, "PF{pf} ROI");
                assert_eq!(resolved.uniforms, uniforms, "PF{pf} uniforms");
            }
        }

        #[test]
        fn p15_uniform_values_drive_the_cpu_lane() {
            let source = Image::solid(4, 4, [120, 130, 140, 255]);
            let base = Image::solid(4, 4, [100, 110, 120, 255]);
            let residual = Image::solid(4, 4, [60, 70, 80, 255]);
            let mask = vec![200u8; 16 * 4];
            let skin = vec![255u8; 16];
            let roi = Roi {
                x: 0,
                y: 0,
                width: 4,
                height: 4,
            };
            let mut silent = p15_uniforms();
            silent.legal_alpha = 0.0;
            silent.dark_circle_alpha = 0.0;
            let silent = p15_job(90023, &source, &base, &residual, &mask, &skin, roi, silent);
            let silent = router::route::<P15Wrinkle>(&silent, Backend::Cpu)
                .expect("CPU lane")
                .output;
            assert_eq!(silent, source, "zero alpha must be identity");

            let active = p15_job(
                90023,
                &source,
                &base,
                &residual,
                &mask,
                &skin,
                roi,
                p15_uniforms(),
            );
            let active = router::route::<P15Wrinkle>(&active, Backend::Cpu)
                .expect("CPU lane")
                .output;
            assert_ne!(active, source, "non-zero alpha must composite");
        }

        #[test]
        fn p15_glabella_is_refused_by_the_producer() {
            let source = Image::solid(2, 2, [120, 130, 140, 255]);
            let base = Image::solid(2, 2, [100, 110, 120, 255]);
            let residual = Image::solid(2, 2, [60, 70, 80, 255]);
            let mask = vec![200u8; 4 * 4];
            let skin = vec![255u8; 4];
            let error = aurora_portrait::cuda_wrinkle::resolve_p15(
                aurora_portrait::cuda_wrinkle::P15Inputs {
                    pf: 21024,
                    source: &source,
                    base: &base,
                    residual: &residual,
                    mask_rgba: &mask,
                    skin_mask: &skin,
                    roi: Roi {
                        x: 0,
                        y: 0,
                        width: 2,
                        height: 2,
                    },
                    ui: 50.0,
                    uniforms: p15_uniforms(),
                },
            )
            .unwrap_err();
            assert_eq!(
                error,
                Refusal::MissingInput("PF21024 mask.r/mask.b contribution law (r76 s63 U3)")
            );
        }
    }
}

#[cfg(windows)]
fn try_cuda_effect(
    current: &Image,
    pf: u32,
    degree: f32,
    skin_mask: Option<&[u8]>,
) -> Option<Result<(Image, Vec<Dispatch>), CudaError>> {
    if let Some(error) = forced_cuda_failure() {
        return Some(Err(error));
    }
    if !matches!(pf, 8068 | 8069 | 8070 | 21015 | 30096 | 90005 | 90007) {
        return None;
    }
    let mask = match skin_mask {
        Some(mask) => mask,
        None => {
            return Some(Err(CudaError::Refused(
                aurora_portrait::Refusal::MissingInput("skin mask"),
            )));
        }
    };
    match pf {
        8069 => {
            let ui = ((degree - 0.5) * 200.0).clamp(-100.0, 100.0);
            if ui >= 0.0 {
                return None;
            }
            let extra = vec![255u8; mask.len()];
            let gpu = match aurora_portrait::cuda_skin_models::SkinModels::open() {
                Ok(gpu) => gpu,
                Err(err) => return Some(Err(err)),
            };
            Some(gpu.texture_apply(current, mask, &extra, ui))
        }
        8070 => {
            let base = current.clone();
            let blur = blur_rgba(current, 2.0);
            let gpu = match aurora_portrait::cuda_skin_models::SkinModels::open() {
                Ok(gpu) => gpu,
                Err(err) => return Some(Err(err)),
            };
            Some(gpu.clarity_apply(
                current,
                &base,
                &blur,
                mask,
                (0, 0, current.width, current.height),
                degree.clamp(0.0, 1.0),
                0.0,
                4.68,
            ))
        }
        30096 => {
            let (Some(cold), Some(warm)) = (cold_lut(), warm_lut()) else {
                return Some(Err(CudaError::Refused(Refusal::MissingInput(
                    "skin temperature LUT",
                ))));
            };
            let ui = ((degree - 0.5) * 200.0).clamp(-100.0, 100.0);
            Some(with_skin_lut(|gpu| {
                gpu.temperature(current, mask, cold, warm, ui)
                    .map(|(image, dispatch)| (image, vec![dispatch]))
            }))
        }
        8068 | 21015 | 90005 | 90007 => cuda_router::skin_effect(pf, current, degree, mask),
        _ => None,
    }
}

#[cfg(windows)]
fn try_cuda_skin_adjust_chain(
    current: &Image,
    first_degree: f32,
    second_degree: f32,
    mask: &[u8],
) -> Option<Result<(Image, Vec<Dispatch>), CudaError>> {
    if let Some(error) = forced_cuda_failure() {
        return Some(Err(error));
    }
    if forced_cuda_empty_dispatch() {
        return Some(Ok((current.clone(), Vec::new())));
    }
    let (Some(first_lut), Some(second_lut)) = (whiten_lut(), ruddy_lut()) else {
        return None;
    };
    Some(with_skin_lut(|gpu| {
        gpu.skin_adjust_chain(
            current,
            mask,
            first_lut,
            first_degree.clamp(0.0, 1.0),
            second_lut,
            second_degree.clamp(0.0, 1.0),
        )
    }))
}

#[cfg(windows)]
thread_local! {
    static SKIN_LUT: RefCell<Option<aurora_portrait::cuda_lut::SkinLut>> =
        const { RefCell::new(None) };
}

#[cfg(windows)]
fn with_skin_lut<T>(
    run: impl FnOnce(&aurora_portrait::cuda_lut::SkinLut) -> Result<T, CudaError>,
) -> Result<T, CudaError> {
    SKIN_LUT.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(aurora_portrait::cuda_lut::SkinLut::open()?);
        } else if let Err(error) = slot.as_mut().expect("skin LUT session").resume() {
            *slot = None;
            return Err(error);
        }
        let result = run(slot.as_ref().expect("skin LUT session"));
        if result.is_err() {
            let _ = slot.as_mut().expect("skin LUT session").suspend();
            *slot = None;
            return result;
        }
        match slot.as_mut().expect("skin LUT session").suspend() {
            Ok(()) => result,
            Err(error) => {
                *slot = None;
                if result.is_ok() {
                    Err(error)
                } else {
                    result
                }
            }
        }
    })
}

fn decode_lut(png: &[u8], name: &'static str) -> Option<Vec<u8>> {
    let image = image::load_from_memory(png).ok()?.into_rgb8();
    if image.width() as usize != LUT_EDGE || image.height() as usize != LUT_EDGE {
        log::warn!(
            "portrait LUT {name}: expected {LUT_EDGE}x{LUT_EDGE}, got {}x{}",
            image.width(),
            image.height()
        );
        return None;
    }
    let bytes = image.into_raw();
    (bytes.len() == LUT_BYTES).then_some(bytes)
}

fn tint_green() -> Option<&'static [u8]> {
    static LUT: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    LUT.get_or_init(|| {
        decode_lut(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../assets/portrait/skin-adjust/TintGreen.png"
            )),
            "TintGreen.png",
        )
    })
    .as_deref()
}

fn tint_magenta() -> Option<&'static [u8]> {
    static LUT: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    LUT.get_or_init(|| {
        decode_lut(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../assets/portrait/skin-adjust/TintMagenta.png"
            )),
            "TintMagenta.png",
        )
    })
    .as_deref()
}

fn whiten_lut() -> Option<&'static [u8]> {
    static LUT: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    LUT.get_or_init(|| {
        decode_lut(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../assets/portrait/skin-adjust/whiten.png"
            )),
            "whiten.png",
        )
    })
    .as_deref()
}

fn ruddy_lut() -> Option<&'static [u8]> {
    static LUT: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    LUT.get_or_init(|| {
        decode_lut(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../assets/portrait/skin-adjust/ruddy_new.png"
            )),
            "ruddy_new.png",
        )
    })
    .as_deref()
}

fn cold_lut() -> Option<&'static [u8]> {
    static LUT: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    LUT.get_or_init(|| {
        decode_lut(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../assets/portrait/skin-adjust/cold.png"
            )),
            "cold.png",
        )
    })
    .as_deref()
}

fn warm_lut() -> Option<&'static [u8]> {
    static LUT: OnceLock<Option<Vec<u8>>> = OnceLock::new();
    LUT.get_or_init(|| {
        decode_lut(
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../assets/portrait/skin-adjust/warm.png"
            )),
            "warm.png",
        )
    })
    .as_deref()
}

fn apply_skin_tint(
    current: &Image,
    degree: f32,
    mask: Option<&[u8]>,
) -> Result<Image, aurora_portrait::Refusal> {
    let (Some(green), Some(magenta)) = (tint_green(), tint_magenta()) else {
        return Err(aurora_portrait::Refusal::MissingInput("skin tint LUT"));
    };
    let Some(mask) = mask else {
        return Err(aurora_portrait::Refusal::MissingInput("skin mask"));
    };
    let ui = ((degree - 0.5) * 200.0).clamp(-100.0, 100.0);
    let signed = ui.clamp(-100.0, 100.0) / 100.0;
    let lut = if signed <= 0.0 { green } else { magenta };
    aurora_portrait::kernels::makeup::skin_adjust_lut(current, mask, lut, signed.abs())
}

fn apply_skin_whiten(
    current: &Image,
    degree: f32,
    mask: Option<&[u8]>,
) -> Result<Image, aurora_portrait::Refusal> {
    let Some(lut) = whiten_lut() else {
        return Err(aurora_portrait::Refusal::MissingInput("whiten LUT"));
    };
    let Some(mask) = mask else {
        return Err(aurora_portrait::Refusal::MissingInput("skin mask"));
    };
    aurora_portrait::kernels::makeup::skin_adjust_lut(current, mask, lut, degree.clamp(0.0, 1.0))
}

fn apply_skin_temperature(
    current: &Image,
    degree: f32,
    mask: Option<&[u8]>,
) -> Result<Image, aurora_portrait::Refusal> {
    let (Some(cold), Some(warm)) = (cold_lut(), warm_lut()) else {
        return Err(aurora_portrait::Refusal::MissingInput(
            "skin temperature LUT",
        ));
    };
    let Some(mask) = mask else {
        return Err(aurora_portrait::Refusal::MissingInput("skin mask"));
    };
    let ui = ((degree - 0.5) * 200.0).clamp(-100.0, 100.0);
    aurora_portrait::kernels::face_detail::skin_temperature(current, mask, cold, warm, ui)
}

fn apply_skin_ruddy(
    current: &Image,
    degree: f32,
    mask: Option<&[u8]>,
) -> Result<Image, aurora_portrait::Refusal> {
    let Some(lut) = ruddy_lut() else {
        return Err(aurora_portrait::Refusal::MissingInput("ruddy LUT"));
    };
    let Some(mask) = mask else {
        return Err(aurora_portrait::Refusal::MissingInput("skin mask"));
    };
    aurora_portrait::kernels::makeup::skin_adjust_lut(current, mask, lut, degree.clamp(0.0, 1.0))
}

fn apply_skin_gloss(
    current: &Image,
    degree: f32,
    mask: Option<&[u8]>,
) -> Result<Image, aurora_portrait::Refusal> {
    let Some(mask) = mask else {
        return Err(aurora_portrait::Refusal::MissingInput("skin mask"));
    };
    let extra = vec![255u8; mask.len()];
    let native = (degree * 100.0).clamp(0.0, 100.0);
    aurora_portrait::kernels::skin_models::gloss_apply(
        current,
        [&mask, &mask, &mask],
        [&extra, &extra, &extra],
        native,
    )
}

fn apply_skin_texture(
    current: &Image,
    degree: f32,
    mask: Option<&[u8]>,
) -> Result<Image, aurora_portrait::Refusal> {
    let ui = ((degree - 0.5) * 200.0).clamp(-100.0, 100.0);
    if ui >= 0.0 {
        return Ok(current.clone());
    }
    let Some(mask) = mask else {
        return Err(aurora_portrait::Refusal::MissingInput("skin mask"));
    };
    let extra = vec![255u8; mask.len()];
    aurora_portrait::kernels::skin_models::texture_surface_blur(current, &mask, &extra, ui)
}

fn apply_skin_clarity(
    current: &Image,
    degree: f32,
    mask: Option<&[u8]>,
    width: u32,
    height: u32,
) -> Result<Image, aurora_portrait::Refusal> {
    let Some(mask) = mask else {
        return Err(aurora_portrait::Refusal::MissingInput("skin mask"));
    };
    let base = current.clone();
    let blur = blur_rgba(current, 2.0);
    aurora_portrait::kernels::skin_models::clarity_apply(
        current,
        &base,
        &blur,
        &mask,
        (0, 0, width, height),
        degree.clamp(0.0, 1.0),
        0.0,
        4.68,
    )
}

fn blur_rgba(src: &Image, sigma: f32) -> Image {
    let rgba = image::RgbaImage::from_raw(src.width, src.height, src.pixels.clone())
        .expect("image dimensions match its pixel buffer");
    let out = image::imageops::blur(&rgba, sigma);
    Image::new(src.width, src.height, out.into_raw())
        .expect("blur output keeps the source dimensions")
}

struct RadianceAssets {
    lut_a: Vec<u8>,
    lut_b: Vec<u8>,
    gamma: Vec<f32>,
    matrices: Vec<f32>,
}

fn f32_asset(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn radiance_assets() -> &'static RadianceAssets {
    static ASSETS: OnceLock<RadianceAssets> = OnceLock::new();
    ASSETS.get_or_init(|| RadianceAssets {
        lut_a: include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/portrait/radiance/native-lut-a.bin"
        ))
        .to_vec(),
        lut_b: include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/portrait/radiance/native-lut-b.bin"
        ))
        .to_vec(),
        gamma: f32_asset(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/portrait/radiance/native-gamma-rgba32f.bin"
        ))),
        matrices: f32_asset(include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../assets/portrait/radiance/native-color-matrices.bin"
        ))),
    })
}

fn apply_skin_radiance(
    current: &Image,
    degree: f32,
    job: Option<RadianceInputs<'_>>,
) -> Result<Image, aurora_portrait::Refusal> {
    let resolved = resolve_radiance_inputs(current, job)?;
    let planes = aurora_portrait::kernels::radiance::MaskPlanes {
        skin: resolved.skin,
        parsing_rgba: resolved.parsing_rgba,
        face_rgba: resolved.face_rgba,
    };
    let assets = radiance_assets();
    let fixed = aurora_portrait::kernels::radiance::FixedAssets {
        lut_a: &assets.lut_a,
        lut_b: &assets.lut_b,
        gamma: &assets.gamma,
        matrices: &assets.matrices,
    };
    let ui = (degree * 100.0).clamp(0.0, 100.0);
    Ok(aurora_portrait::kernels::radiance::apply(
        current,
        &planes,
        ui,
        resolved.value_mean,
        &fixed,
    )?
    .image)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pf30026_binding(job_id: u64) -> Pf30026GateBinding {
        // Synthetic role receipt for consumer tests. It does not infer vendor
        // A/B from the model output number; the provider selector remains open.
        Pf30026GateBinding {
            job_id,
            plane_a: Pf30026EdgeOutput::Output0,
            plane_b: Pf30026EdgeOutput::Output1,
        }
    }

    fn frame(w: u32, h: u32) -> Vec<u8> {
        let mut rgba = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let v = ((x * 7 + y * 13) % 24) as u8;
                rgba[i] = 190 + v;
                rgba[i + 1] = 150 + v;
                rgba[i + 2] = 110 + v;
                rgba[i + 3] = 255;
            }
        }
        rgba
    }

    fn smile_field(value: f32) -> SmileRgbReconstructionField {
        SmileRgbReconstructionField {
            width: 1024,
            height: 768,
            values: vec![value; 3 * 1024 * 768],
        }
    }

    #[test]
    fn pf100007_raw_rgb_field_never_populates_model_dx() {
        let producer = ProducerInputs {
            smile: Some(SmileProducerInputs {
                ticket: 48,
                material: Some("Grinning".to_string()),
                roi: Some((0, 0, 8, 6)),
                rgb_reconstruction_780929437: Some(smile_field(0.5)),
                lip_mask: Some(vec![255; 48]),
                mouth_mask: Some(vec![255; 48]),
                ..Default::default()
            }),
            ..Default::default()
        };
        let masks = PortraitMasks::solid(1.0);
        let mut inputs = PortraitJobInputs::from_renderer_job(48, &masks);
        inputs.fill_from_producer_inputs(&producer);

        assert!(inputs.smile.model_dx.is_none());
        assert!(inputs.smile.model_dy.is_none());
    }

    #[test]
    fn pf100007_raw_rgb_field_requires_host_split_receipt() {
        let smile = SmileProducerInputs {
            ticket: 48,
            rgb_reconstruction_780929437: Some(smile_field(0.5)),
            ..Default::default()
        };

        assert_eq!(
            smile.decode_channel_split(8, 6),
            Err(Refusal::MissingInput("pf100007 host split receipt"))
        );
    }

    #[test]
    fn pf100007_typed_decoder_uses_declared_axis_semantics() {
        let smile = SmileProducerInputs {
            ticket: 48,
            rgb_reconstruction_780929437: Some(smile_field(0.5)),
            host_split_spec: Some(SmileHostSplitSpec {
                job_id: 48,
                dx: SmileAxisDecode {
                    channel: SmileFieldChannel::R,
                    sign: SmileAxisSign::Positive,
                    scale: 2.0,
                    bias: 0.25,
                },
                dy: SmileAxisDecode {
                    channel: SmileFieldChannel::G,
                    sign: SmileAxisSign::Negative,
                    scale: 1.0,
                    bias: -0.5,
                },
                model_roi: smile_model_rect(),
                frame_roi: Rect {
                    x: 0,
                    y: 0,
                    width: 8,
                    height: 6,
                },
            }),
            ..Default::default()
        };
        let split = smile.decode_channel_split(8, 6).unwrap();

        assert_eq!(split.model_dx, vec![1.25; 48]);
        assert_eq!(split.model_dy, vec![-1.0; 48]);
    }

    fn apply(rgba: &mut Vec<u8>, w: u32, h: u32, extra: &[(&str, f32)]) {
        let map: BTreeMap<String, f32> = extra.iter().map(|(k, v)| (k.to_string(), *v)).collect();
        let masks = PortraitMasks::solid(1.0);
        let inputs = PortraitJobInputs::from_renderer_job(1, &masks);
        let _ = apply_portrait_effects_with_job(rgba, w, h, NormRect::FULL, &map, &inputs);
    }

    #[test]
    fn renderer_receipt_rejects_changed_frame_or_params_before_dispatch() {
        let masks = PortraitMasks::solid(1.0);
        let (width, height) = (8, 8);
        let source = frame(width, height);
        let params = BTreeMap::from([("pc:90005".to_string(), 0.9f32)]);
        let receipt = PortraitJobReceipt::from_renderer_receipt(
            &source,
            width,
            height,
            NormRect::FULL,
            &params,
            &masks,
        );
        let inputs = PortraitJobInputs::from_renderer_receipt(receipt, &masks);

        let mut changed_frame = source.clone();
        changed_frame[0] ^= 1;
        let before = changed_frame.clone();
        let report = apply_portrait_effects_with_job(
            &mut changed_frame,
            width,
            height,
            NormRect::FULL,
            &params,
            &inputs,
        );
        assert_eq!(changed_frame, before);
        assert_eq!(report.job_id, Some(receipt.job_id()));
        assert_eq!(
            report.effects[0].refusal,
            Some(Refusal::ValueOutOfRange("PortraitJobReceipt digest"))
        );
        assert!(report.effects[0].skipped);
        assert!(!report.effects[0].executed);
        assert!(report.effects[0].dispatches.is_empty());

        let mut changed_params = params.clone();
        changed_params.insert("pc:90005".to_string(), 0.8);
        let mut unchanged_frame = source.clone();
        let before = unchanged_frame.clone();
        let report = apply_portrait_effects_with_job(
            &mut unchanged_frame,
            width,
            height,
            NormRect::FULL,
            &changed_params,
            &inputs,
        );
        assert_eq!(unchanged_frame, before);
        assert_eq!(report.job_id, Some(receipt.job_id()));
        assert_eq!(
            report.effects[0].refusal,
            Some(Refusal::ValueOutOfRange("PortraitJobReceipt digest"))
        );
        assert!(report.effects[0].dispatches.is_empty());
    }

    #[test]
    fn renderer_receipt_does_not_promote_masks_only() {
        let masks = PortraitMasks::solid(1.0);
        let mut rgba = frame(8, 8);
        let before = rgba.clone();
        let params = BTreeMap::from([("pc:90005".to_string(), 0.9f32)]);
        let inputs = PortraitJobInputs::masks_only(&masks);
        let report = apply_portrait_effects_with_job(
            &mut rgba,
            8,
            8,
            NormRect::FULL,
            &params,
            &inputs,
        );
        assert_eq!(rgba, before);
        assert_eq!(
            report.effects[0].refusal,
            Some(Refusal::MissingInput("same-job input bundle"))
        );
        assert!(report.effects[0].skipped);
        assert!(report.effects[0].dispatches.is_empty());
    }

    #[test]
    fn renderer_job_without_receipt_is_refused_before_dispatch() {
        let masks = PortraitMasks::solid(1.0);
        let mut rgba = frame(8, 8);
        let before = rgba.clone();
        let params = BTreeMap::from([("pc:90005".to_string(), 0.9f32)]);
        let inputs = PortraitJobInputs::from_renderer_job(9, &masks);
        let report = apply_portrait_effects_with_report_and_inputs_policy(
            &mut rgba,
            8,
            8,
            NormRect::FULL,
            &params,
            &inputs,
            true,
        );
        assert_eq!(rgba, before);
        assert_eq!(report.job_id, Some(9));
        assert_eq!(
            report.effects[0].refusal,
            Some(Refusal::MissingInput("renderer receipt"))
        );
        assert!(report.effects[0].skipped);
        assert!(!report.effects[0].executed);
        assert!(report.effects[0].dispatches.is_empty());
    }

    #[test]
    fn pf90116_three_tensor_alignment_bundle_requires_ticket_and_runs_stage() {
        use aurora_portrait::eye_bag_alignment::{
            align_eye_bag, EyeBagAlignmentInputs, EyeBagColorUnits, EyeBagGeometryClass,
            EyeBagModelTensors, MODEL_HEIGHT, MODEL_ID, MODEL_WIDTH,
        };
        use aurora_portrait::kernels::eye_bag::Roi;

        let job_id = 31;
        let (width, height) = (16u32, 16u32);
        let model_pixels = MODEL_WIDTH as usize * MODEL_HEIGHT as usize;
        let model_input: Vec<f32> = (0..model_pixels * 3)
            .map(|index| (index % 97) as f32 / 97.0)
            .collect();
        let model_color: Vec<f32> = (0..model_pixels * 3)
            .map(|index| ((index * 17 + 23) % 240) as f32)
            .collect();
        let mut model_response = vec![0.0f32; model_pixels];
        for y in 160..224 {
            for x in 192..320 {
                model_response[y * MODEL_WIDTH as usize + x] = 0.5;
            }
        }
        let aligned = align_eye_bag(EyeBagAlignmentInputs {
            ticket: job_id,
            frame_width: width,
            frame_height: height,
            tensors: EyeBagModelTensors {
                model_id: MODEL_ID,
                input_781141184_rgb_chw: &model_input,
                response_3204548346_hw: &model_response,
                color_780929437_rgb_chw: &model_color,
            },
            model_color_units: EyeBagColorUnits::SourceBytes,
            geometry_class: EyeBagGeometryClass::Candidate,
            geometry_receipt: None,
            work_width: Some(width),
            work_height: Some(height),
            // Non-identity geometry prevents an accidental unit-matrix fallback.
            work_to_destination: Some([0.5, 0.1, 0.5, -0.1, 0.6, 1.0]),
            destination_roi: Some(Roi {
                x: 2,
                y: 2,
                w: 10,
                h: 9,
            }),
        })
        .expect("explicit candidate geometry");
        let expected = aurora_portrait::kernels::eye_bag::compose_field_image_pf90116(
            &Image::new(width, height, frame(width, height)).expect("source"),
            &aligned.aligned_color_field_rgb_f32,
            &aligned.aligned_response_mask,
            0.8,
        )
        .expect("CPU reference");

        let producer = ProducerInputs {
            eye_bag: Some(aligned),
            ..ProducerInputs::default()
        };
        let masks = PortraitMasks::solid(1.0);
        let mut inputs = PortraitJobInputs::from_renderer_job(job_id, &masks);
        inputs.fill_from_producer_inputs(&producer);
        #[cfg(windows)]
        {
            let (cpu_actual, cpu_run) = {
                let _force = ForceCudaFailureGuard::new();
                apply_one_with_inputs(90116, width, height, &inputs)
            };
            assert_eq!(cpu_actual, expected.pixels);
            assert_eq!(cpu_run.backend, PortraitBackend::Cpu);
            assert!(cpu_run.dispatches.is_empty());
            eprintln!(
                "PF90116 fallback backend={:?} bytes_equal=true reason=forced-cuda-failure",
                cpu_run.backend
            );
        }
        let (actual, run) = apply_one_with_inputs(90116, width, height, &inputs);
        assert_eq!(actual, expected.pixels);
        assert!(run.executed);
        assert!(run.refusal.is_none(), "{:?}", run.refusal);
        eprintln!(
            "PF90116 backend={:?} kernels={:?} fallback={:?}",
            run.backend,
            run.dispatches
                .iter()
                .map(|dispatch| dispatch.kernel)
                .collect::<Vec<_>>(),
            run.fallback_reason
        );

        let other_masks = PortraitMasks::solid(1.0);
        let mut wrong_ticket = PortraitJobInputs::from_renderer_job(job_id + 1, &other_masks);
        wrong_ticket.fill_from_producer_inputs(&producer);
        let (unchanged, refused) = apply_one_with_inputs(90116, width, height, &wrong_ticket);
        assert_eq!(unchanged, frame(width, height));
        assert_eq!(
            refused.refusal,
            Some(Refusal::MissingInput("PF90116_SAME_JOB_TICKET"))
        );
        assert!(!refused.executed);
        assert!(refused.dispatches.is_empty());
    }

    #[test]
    fn pf90116_production_receipt_rejects_candidate_geometry() {
        let (width, height) = (8u32, 8u32);
        let source = frame(width, height);
        let masks = PortraitMasks::solid(1.0);
        let params = BTreeMap::from([("pc:90116".to_string(), 0.5f32)]);
        let receipt = PortraitJobReceipt::from_renderer_receipt(
            &source,
            width,
            height,
            NormRect::FULL,
            &params,
            &masks,
        );
        let mut inputs = PortraitJobInputs::from_renderer_receipt(receipt, &masks);
        let ticket = inputs.job_id().expect("renderer receipt");
        let color = vec![0.0; 8 * 8 * 3];
        let response = vec![0.0; 8 * 8];
        inputs.eye_bag = Some(EyeBagInputs {
            ticket,
            model_id: aurora_portrait::eye_bag_alignment::MODEL_ID,
            color_units: aurora_portrait::eye_bag_alignment::EyeBagColorUnits::SourceBytes,
            geometry_class: aurora_portrait::eye_bag_alignment::EyeBagGeometryClass::Candidate,
            geometry_receipt: None,
            width,
            height,
            aligned_color_field_rgb_f32: &color,
            aligned_response_mask: &response,
        });
        let current = Image::new(width, height, source).expect("source");
        assert_eq!(
            validate_effect_inputs(
                90116,
                &current,
                NormRect::FULL,
                &params,
                None,
                &inputs,
                None,
            ),
            Err(Refusal::MissingInput("PF90116_GEOMETRY_RECEIPT"))
        );
    }

    #[test]
    fn pf30025_uses_existing_producer_inputs_and_fails_closed() {
        let masks = PortraitMasks::solid(1.0);
        let mut inputs = PortraitJobInputs::from_renderer_job(12, &masks);
        let (before, run) = apply_one_with_inputs(30025, 1, 1, &inputs);
        assert_eq!(
            run.refusal,
            Some(Refusal::MissingInput("pf30025 repair_bgra"))
        );
        assert!(run.skipped);
        assert!(!run.executed);

        let source = frame(1, 1);
        let repair = vec![30, 20, 200, 255];
        let gate = vec![0.5f32];
        let teeth_mask = vec![0.0f32];
        let receipt = Pf30025SameJobReceipt {
            job_id: 12,
            model_id: PF30025_MODEL_IDS[0],
            width: 1,
            height: 1,
            source_stride: 4,
            source_sha256: sha256_32(&source),
            repair_sha256: sha256_32(&repair),
            gate_sha256: sha256_32(f32_bytes(&gate)),
        };
        inputs.pf30025 = Some(Pf30025Inputs {
            receipt,
            repair_780929437_bgra: &repair,
            gate_3204548346: &gate,
            teeth_mask: &teeth_mask,
            roi: ToothRect {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
        });
        let (after, run) = apply_one_with_inputs(30025, 1, 1, &inputs);
        assert_ne!(after, before);
        assert_eq!(run.refusal, None);
        assert!(run.executed);
        assert!(!run.skipped);

        inputs.pf30025 = Some(Pf30025Inputs {
            receipt: Pf30025SameJobReceipt {
                job_id: 13,
                ..receipt
            },
            ..inputs.pf30025.unwrap()
        });
        let (after, run) = apply_one_with_inputs(30025, 1, 1, &inputs);
        assert_eq!(after, before);
        assert_eq!(
            run.refusal,
            Some(Refusal::MissingInput("PF30025_SAME_JOB_RECEIPT"))
        );
        assert!(!run.executed);
    }

    #[cfg(windows)]
    #[test]
    fn pf30025_cuda_matches_cpu_for_bgra_roi_alpha_and_gate() {
        struct Case {
            width: u32,
            height: u32,
            source: Vec<u8>,
            repair_bgra: Vec<u8>,
            teeth_mask: Vec<f32>,
            roi: ToothRect,
            degree01: f32,
            gate: Vec<f32>,
        }

        fn run(case: &Case, force_cpu: bool) -> (Vec<u8>, PortraitEffectRun) {
            let masks = PortraitMasks::solid(1.0);
            let mut inputs = PortraitJobInputs::from_renderer_job(30025, &masks);
            let receipt = Pf30025SameJobReceipt {
                job_id: 30025,
                model_id: PF30025_MODEL_IDS[0],
                width: case.width,
                height: case.height,
                source_stride: case.width * 4,
                source_sha256: sha256_32(&case.source),
                repair_sha256: sha256_32(&case.repair_bgra),
                gate_sha256: sha256_32(f32_bytes(&case.gate)),
            };
            inputs.pf30025 = Some(Pf30025Inputs {
                receipt,
                repair_780929437_bgra: &case.repair_bgra,
                gate_3204548346: &case.gate,
                teeth_mask: &case.teeth_mask,
                roi: case.roi,
            });
            let mut rgba = case.source.clone();
            let _force = force_cpu.then(ForceCudaFailureGuard::new);
            let report = apply_portrait_effects_with_job(
                &mut rgba,
                case.width,
                case.height,
                NormRect::FULL,
                &BTreeMap::from([("pc:30025".to_string(), case.degree01)]),
                &inputs,
            );
            assert_eq!(report.effects.len(), 1);
            (rgba, report.effects.into_iter().next().unwrap())
        }

        let cases = [
            Case {
                width: 1,
                height: 1,
                source: vec![10, 20, 30, 91],
                // R rounds from 10.5 to 11: midpoint away from zero.
                repair_bgra: vec![30, 20, 12, 255],
                teeth_mask: vec![0.25],
                roi: ToothRect {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                },
                degree01: 1.0,
                gate: vec![7.0],
            },
            Case {
                width: 4,
                height: 3,
                source: vec![
                    10, 20, 30, 41, 50, 60, 70, 81, 90, 100, 110, 121, 130, 140, 150, 161,
                    12, 22, 32, 42, 52, 62, 72, 82, 92, 102, 112, 122, 132, 142, 152, 162,
                    14, 24, 34, 44, 54, 64, 74, 84, 94, 104, 114, 124, 134, 144, 154, 164,
                ],
                repair_bgra: vec![
                    200, 180, 160, 0, 210, 190, 170, 1, 220, 200, 180, 2, 230, 210, 190, 3,
                    240, 220, 200, 4, 250, 230, 210, 5, 20, 30, 40, 6, 30, 40, 50, 7,
                    40, 50, 60, 8, 50, 60, 70, 9, 60, 70, 80, 10, 70, 80, 90, 11,
                ],
                teeth_mask: vec![
                    0.0, 0.2, 0.5, 1.0, 0.7, 0.33, 0.25, 0.9, 0.6, 0.15, 0.45, 0.8,
                ],
                roi: ToothRect {
                    x: 1,
                    y: 1,
                    width: 2,
                    height: 1,
                },
                degree01: 0.37,
                gate: vec![0.2, 0.5, 1.0, 0.7, 0.33, 0.25, 0.9, 0.6, 0.15, 0.45, 0.8, 0.1],
            },
            Case {
                width: 3,
                height: 2,
                source: vec![
                    8, 18, 28, 38, 48, 58, 68, 78, 88, 98, 108, 118,
                    13, 23, 33, 43, 53, 63, 73, 83, 93, 103, 113, 123,
                ],
                repair_bgra: vec![
                    180, 160, 140, 255, 190, 170, 150, 254, 200, 180, 160, 253,
                    210, 190, 170, 252, 220, 200, 180, 251, 230, 210, 190, 250,
                ],
                teeth_mask: vec![0.1, 0.4, 0.9, 1.0, 0.55, 0.75],
                roi: ToothRect {
                    x: 0,
                    y: 0,
                    width: 3,
                    height: 2,
                },
                degree01: 1.0,
                gate: vec![1.0, 2.0, 3.0, 0.5, 0.25, 0.75],
            },
        ];

        for case in &cases {
            let (cpu, cpu_run) = run(case, true);
            let (actual, actual_run) = run(case, false);
            let differing_bytes = actual
                .iter()
                .zip(&cpu)
                .filter(|(left, right)| left != right)
                .count();
            assert_eq!(actual, cpu, "PF30025 CPU/CUDA byte parity");
            assert_eq!(differing_bytes, 0);
            assert_eq!(cpu_run.backend, PortraitBackend::Cpu);
            assert!(cpu_run.dispatches.is_empty());
            assert!(actual_run.executed);
            assert!(actual_run.refusal.is_none());
            for pixel in 0..case.width as usize * case.height as usize {
                assert_eq!(
                    actual[pixel * 4 + 3],
                    case.source[pixel * 4 + 3],
                    "PF30025 alpha pixel {pixel}"
                );
                let x = pixel as i32 % case.width as i32;
                let y = pixel as i32 / case.width as i32;
                let inside = x >= case.roi.x
                    && y >= case.roi.y
                    && x < case.roi.x + case.roi.width
                    && y < case.roi.y + case.roi.height;
                if !inside {
                    assert_eq!(
                        &actual[pixel * 4..pixel * 4 + 4],
                        &case.source[pixel * 4..pixel * 4 + 4],
                        "PF30025 ROI pixel {pixel}"
                    );
                }
            }
            if std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_some() {
                assert_eq!(
                    actual_run.backend,
                    PortraitBackend::Cuda,
                    "{:?}",
                    actual_run.fallback_reason
                );
                assert_eq!(actual_run.dispatches.len(), 1);
                assert_eq!(
                    actual_run.dispatches[0].kernel,
                    aurora_portrait::cuda_pf30025::KERNEL
                );
                assert!(!actual_run.dispatches[0].device.is_empty());
                let dispatch = &actual_run.dispatches[0];
                eprintln!(
                    "PF30025 case={}x{} roi={}x{}+{}+{} backend=Cuda kernel={} device={} cc={}.{} driver={} ptx_isa={}.{} differing_bytes={}",
                    case.width,
                    case.height,
                    case.roi.width,
                    case.roi.height,
                    case.roi.x,
                    case.roi.y,
                    dispatch.kernel,
                    dispatch.device,
                    dispatch.compute_cap.0,
                    dispatch.compute_cap.1,
                    dispatch.driver_version,
                    dispatch.ptx_isa.0,
                    dispatch.ptx_isa.1,
                    differing_bytes,
                );
            } else if actual_run.backend == PortraitBackend::Cpu {
                assert!(actual_run.fallback_reason.is_some());
            } else {
                assert_eq!(
                    actual_run.dispatches[0].kernel,
                    aurora_portrait::cuda_pf30025::KERNEL
                );
            }
        }
    }

    fn apply_one_with_inputs(
        pf: u32,
        width: u32,
        height: u32,
        inputs: &PortraitJobInputs<'_>,
    ) -> (Vec<u8>, PortraitEffectRun) {
        let mut rgba = frame(width, height);
        let extra = BTreeMap::from([(format!("pc:{pf}"), 0.8f32)]);
        let report = apply_portrait_effects_with_job(
            &mut rgba,
            width,
            height,
            NormRect::FULL,
            &extra,
            inputs,
        );
        assert_eq!(report.effects.len(), 1, "PF{pf}");
        (rgba, report.effects.into_iter().next().unwrap())
    }

    fn strided_plane(
        values: &[u8],
        width: u32,
        height: u32,
        stride: u32,
        bytes_per_pixel: u32,
    ) -> Vec<u8> {
        let mut plane = vec![0xA5; stride as usize * height as usize];
        let row_bytes = width as usize * bytes_per_pixel as usize;
        for y in 0..height as usize {
            let src = y * row_bytes;
            let dst = y * stride as usize;
            plane[dst..dst + row_bytes].copy_from_slice(&values[src..src + row_bytes]);
        }
        plane
    }

    fn hair_frame(width: u32, height: u32) -> Vec<u8> {
        let count = width as usize * height as usize;
        (0..count)
            .flat_map(|index| {
                [
                    (24 + index * 13 % 180) as u8,
                    (35 + index * 17 % 170) as u8,
                    (46 + index * 19 % 160) as u8,
                    (31 + index * 23 % 200) as u8,
                ]
            })
            .collect()
    }

    fn stray_hair_fixture() -> (Image, Image, Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut source = Vec::with_capacity(5 * 4 * 4);
        let mut repair = Vec::with_capacity(5 * 4 * 4);
        for pixel in 0..20u32 {
            source.extend_from_slice(&[
                (pixel * 13 + 2) as u8,
                (pixel * 11 + 20) as u8,
                (pixel * 7 + 30) as u8,
                (17 + pixel * 9) as u8,
            ]);
            repair.extend_from_slice(&[
                (220 - pixel * 3) as u8,
                (220 - pixel * 5) as u8,
                (130 + pixel * 2) as u8,
                (250 - pixel) as u8,
            ]);
        }
        source[0..4].copy_from_slice(&[2, 20, 30, 17]);
        repair[0..4].copy_from_slice(&[3, 220, 130, 250]);
        (
            Image::new(5, 4, source).unwrap(),
            Image::new(5, 4, repair).unwrap(),
            vec![
                255, 128, 0, 255, 64, 200, 17, 33, 99, 245, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            ],
            vec![
                255, 128, 255, 0, 200, 64, 0, 255, 12, 25, 255, 255, 255, 255, 255, 255, 255,
                255, 255, 255,
            ],
            vec![
                0, 255, 255, 255, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                150, 160,
            ],
        )
    }

    fn stray_hair_stage_inputs<'a>(
        masks: &'a PortraitMasks,
        repair: &'a Image,
        raw: &'a [u8],
        person: &'a [u8],
        background: &'a [u8],
    ) -> PortraitJobInputs<'a> {
        PortraitJobInputs {
            stray_hair: StrayHairInputs {
                hairfly_v1: Some(repair),
                raw_mask: Some(raw),
                person_mask: Some(person),
                background_mask: Some(background),
                native_degree: Some(0.5),
                native_person_area: Some(1.0),
                native_background_area: Some(0.5),
            },
            ..PortraitJobInputs::from_renderer_job(557, masks)
        }
    }

    fn run_stray_hair_stage(
        source: &Image,
        inputs: &PortraitJobInputs<'_>,
        force_cpu: bool,
    ) -> (Vec<u8>, PortraitEffectRun) {
        let mut actual = source.pixels.clone();
        let extra = BTreeMap::from([("pc:90088".to_string(), 0.8f32)]);
        #[cfg(windows)]
        let _force = force_cpu.then(ForceCudaFailureGuard::new);
        #[cfg(not(windows))]
        let _ = force_cpu;
        let report = apply_portrait_effects_with_job(
            &mut actual,
            source.width,
            source.height,
            NormRect::FULL,
            &extra,
            inputs,
        );
        assert_eq!(report.effects.len(), 1);
        (actual, report.effects.into_iter().next().unwrap())
    }

    fn hair_smooth_cpu(
        source: &Image,
        roi: HairRoi,
        detail_strength: f32,
        input: &HairSmoothInputs<'_>,
    ) -> Vec<u8> {
        let mut output = source.pixels.clone();
        for ry in 0..roi.height as usize {
            for rx in 0..roi.width as usize {
                let hair =
                    input.hair_mask.unwrap()[ry * input.hair_stride as usize + rx];
                if hair == 0 {
                    continue;
                }
                let mix = input.mix_mask.unwrap()[ry * input.mix_stride as usize + rx];
                let detail = input.detail.unwrap()[ry * input.detail_stride as usize + rx];
                let gate = hair as f32 * 0.003921568859368563;
                let mix_weight =
                    (mix as f32 * 0.003921568859368563) * input.mix_strength * gate;
                let detail_add =
                    ((detail as f32 - 127.5) * 4.0) * (gate * detail_strength * 0.8);
                let x = roi.x as usize + rx;
                let y = roi.y as usize + ry;
                let offset = (y * source.width as usize + x) * 4;
                let blend_offset =
                    ry * input.blend_stride as usize + rx * 3;
                for channel in 0..3 {
                    let src = source.pixels[offset + channel] as f32;
                    let base = if mix_weight > 0.0 {
                        let blend = input.blend_rgb.unwrap()[blend_offset + channel] as f32;
                        src * (1.0 - mix_weight) + blend * mix_weight
                    } else {
                        src
                    };
                    output[offset + channel] = (base + detail_add)
                        .round_ties_even()
                        .clamp(0.0, 255.0) as u8;
                }
            }
        }
        output
    }

    fn hair_gloss_cpu(
        source: &Image,
        roi: HairRoi,
        input: &HairGlossInputs<'_>,
    ) -> Vec<u8> {
        let mut output = source.pixels.clone();
        for ry in 0..roi.height as usize {
            for rx in 0..roi.width as usize {
                let hair =
                    input.hair_mask.unwrap()[ry * input.hair_stride as usize + rx];
                if hair == 0 {
                    continue;
                }
                let detail =
                    input.model_detail.unwrap()[ry * input.detail_stride as usize + rx];
                let delta =
                    (detail as f32 - 127.5) * 2.0 * hair as f32 * 0.003921568859368563
                        * input.dynamic_scale;
                let x = roi.x as usize + rx;
                let y = roi.y as usize + ry;
                let offset = (y * source.width as usize + x) * 4;
                for channel in 0..3 {
                    output[offset + channel] =
                        (source.pixels[offset + channel] as f32 + delta)
                            .clamp(0.0, 255.0) as u8;
                }
            }
        }
        output
    }

    #[test]
    fn hair_stage_refuses_each_missing_plane_before_launch() {
        let masks = PortraitMasks::solid(1.0);
        let (width, height) = (7u32, 5u32);
        let (roi_x, roi_y, roi_width, roi_height) = (2u32, 1u32, 3u32, 2u32);
        let roi = HairRoi {
            x: roi_x,
            y: roi_y,
            width: roi_width,
            height: roi_height,
        };
        let hair = strided_plane(&[255, 180, 0, 96, 64, 220], roi_width, roi_height, 5, 1);
        let mix = strided_plane(&[180, 240, 64, 32, 128, 255], roi_width, roi_height, 4, 1);
        let blend = strided_plane(
            &[20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 10, 30, 50, 70, 90, 110],
            roi_width,
            roi_height,
            14,
            3,
        );
        let detail =
            strided_plane(&[96, 127, 128, 160, 255, 0], roi_width, roi_height, 6, 1);
        let base = HairSmoothInputs {
            ticket: 17,
            width,
            height,
            roi: (roi_x, roi_y, roi_width, roi_height),
            mix_strength: 0.6,
            hair_mask: Some(&hair),
            mix_mask: Some(&mix),
            blend_rgb: Some(&blend),
            detail: Some(&detail),
            hair_stride: 5,
            mix_stride: 4,
            blend_stride: 14,
            detail_stride: 6,
        };
        let cases = [
            (
                HairSmoothInputs {
                    hair_mask: None,
                    ..base
                },
                "PF90139_HAIR_MASK",
            ),
            (
                HairSmoothInputs {
                    mix_mask: None,
                    ..base
                },
                "PF90139_MIX_MASK",
            ),
            (
                HairSmoothInputs {
                    blend_rgb: None,
                    ..base
                },
                "PF90139_BLEND_SOURCE_RGB8",
            ),
            (
                HairSmoothInputs {
                    detail: None,
                    ..base
                },
                "PF90139_DETAIL",
            ),
        ];
        for (hair_smooth, missing) in cases {
            let mut rgba = frame(width, height);
            let original = rgba.clone();
            let inputs = PortraitJobInputs {
                hair_smooth: Some(hair_smooth),
                ..PortraitJobInputs::from_renderer_job(1, &masks)
            };
            let extra = BTreeMap::from([("pc:90139".to_string(), 0.8)]);
            let report = apply_portrait_effects_with_job(
                &mut rgba,
                width,
                height,
                NormRect::FULL,
                &extra,
                &inputs,
            );
            assert_eq!(rgba, original);
            assert_eq!(
                report.effects[0].refusal,
                Some(Refusal::MissingInput(missing))
            );
            assert!(!report.effects[0].executed);
            assert!(report.effects[0].skipped);
            assert!(report.effects[0].dispatches.is_empty());
        }

        let gloss_hair =
            strided_plane(&[255, 180, 0, 96, 64, 220], roi_width, roi_height, 5, 1);
        let gloss_detail =
            strided_plane(&[128, 127, 200, 64, 255, 0], roi_width, roi_height, 7, 1);
        let gloss = HairGlossInputs {
            ticket: 18,
            width,
            height,
            roi: (roi_x, roi_y, roi_width, roi_height),
            dynamic_scale: 0.5,
            hair_mask: Some(&gloss_hair),
            model_detail: Some(&gloss_detail),
            hair_stride: 5,
            detail_stride: 7,
        };
        for (hair_gloss, missing) in [
            (
                HairGlossInputs {
                    hair_mask: None,
                    ..gloss
                },
                "PF90160_HAIR_MASK",
            ),
            (
                HairGlossInputs {
                    model_detail: None,
                    ..gloss
                },
                "PF90160_MODEL_DETAIL",
            ),
        ] {
            let mut rgba = frame(width, height);
            let original = rgba.clone();
            let inputs = PortraitJobInputs {
                hair_gloss: Some(hair_gloss),
                ..PortraitJobInputs::from_renderer_job(1, &masks)
            };
            let extra = BTreeMap::from([("pc:90160".to_string(), 0.8)]);
            let report = apply_portrait_effects_with_job(
                &mut rgba,
                width,
                height,
                NormRect::FULL,
                &extra,
                &inputs,
            );
            assert_eq!(rgba, original);
            assert_eq!(
                report.effects[0].refusal,
                Some(Refusal::MissingInput(missing))
            );
            assert!(!report.effects[0].executed);
            assert!(report.effects[0].skipped);
            assert!(report.effects[0].dispatches.is_empty());
        }
    }

    #[cfg(windows)]
    #[test]
    fn hair_stage_matches_cpu_and_keeps_roi_and_alpha() {
        let masks = PortraitMasks::solid(1.0);
        let (width, height) = (7u32, 5u32);
        let source = Image::new(width, height, hair_frame(width, height)).unwrap();
        let (roi_x, roi_y, roi_width, roi_height) = (2u32, 1u32, 3u32, 2u32);
        let roi = HairRoi {
            x: roi_x,
            y: roi_y,
            width: roi_width,
            height: roi_height,
        };
        let hair = strided_plane(&[255, 180, 0, 96, 64, 220], roi_width, roi_height, 5, 1);
        let mix = strided_plane(&[180, 240, 64, 32, 128, 255], roi_width, roi_height, 4, 1);
        let blend = strided_plane(
            &[20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 10, 30, 50, 70, 90, 110],
            roi_width,
            roi_height,
            14,
            3,
        );
        let detail =
            strided_plane(&[96, 127, 128, 160, 255, 0], roi_width, roi_height, 6, 1);
        let smooth = HairSmoothInputs {
            ticket: 17,
            width,
            height,
            roi: (roi_x, roi_y, roi_width, roi_height),
            mix_strength: 0.6,
            hair_mask: Some(&hair),
            mix_mask: Some(&mix),
            blend_rgb: Some(&blend),
            detail: Some(&detail),
            hair_stride: 5,
            mix_stride: 4,
            blend_stride: 14,
            detail_stride: 6,
        };
        let expected_smooth = hair_smooth_cpu(&source, roi, 0.8, &smooth);
        let inputs = PortraitJobInputs {
            hair_smooth: Some(smooth),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut actual = source.pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:90139".to_string(), 0.8)]),
            &inputs,
        );
        let run = &report.effects[0];
        if run.backend != PortraitBackend::Cuda {
            assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but PF90139 did not dispatch: {:?}",
                run.fallback_reason
            );
            return;
        }
        assert_eq!(actual, expected_smooth);
        assert_eq!(run.dispatches[0].kernel, "pf90139_hair_smooth_kernel");
        eprintln!(
            "PF90139 stage device={} kernel={}",
            run.dispatches[0].device, run.dispatches[0].kernel
        );

        let gloss_hair =
            strided_plane(&[255, 180, 0, 96, 64, 220], roi_width, roi_height, 5, 1);
        let gloss_detail =
            strided_plane(&[128, 127, 200, 64, 255, 0], roi_width, roi_height, 7, 1);
        let gloss = HairGlossInputs {
            ticket: 18,
            width,
            height,
            roi: (roi_x, roi_y, roi_width, roi_height),
            dynamic_scale: 0.5,
            hair_mask: Some(&gloss_hair),
            model_detail: Some(&gloss_detail),
            hair_stride: 5,
            detail_stride: 7,
        };
        let expected_gloss = hair_gloss_cpu(&source, roi, &gloss);
        let inputs = PortraitJobInputs {
            hair_gloss: Some(gloss),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut actual = source.pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:90160".to_string(), 0.8)]),
            &inputs,
        );
        let run = &report.effects[0];
        assert_eq!(run.backend, PortraitBackend::Cuda, "{:?}", run.fallback_reason);
        assert_eq!(actual, expected_gloss);
        assert_eq!(run.dispatches[0].kernel, "pf90160_hair_gloss_kernel");
        eprintln!(
            "PF90160 stage device={} kernel={}",
            run.dispatches[0].device, run.dispatches[0].kernel
        );

        for y in 0..height {
            for x in 0..width {
                let offset = ((y * width + x) * 4) as usize;
                assert_eq!(actual[offset + 3], source.pixels[offset + 3]);
                if x < roi_x
                    || x >= roi_x + roi_width
                    || y < roi_y
                    || y >= roi_y + roi_height
                {
                    assert_eq!(
                        &actual[offset..offset + 4],
                        &source.pixels[offset..offset + 4]
                    );
                }
            }
        }
    }

    #[cfg(windows)]
    fn run_hair_route_cuda_fault(
        pf: u32,
        empty_dispatch: bool,
    ) -> (Vec<u8>, PortraitEffectRun, Vec<u8>) {
        let masks = PortraitMasks::solid(1.0);
        let (width, height) = (7u32, 5u32);
        let source = Image::new(width, height, hair_frame(width, height)).unwrap();
        let (roi_x, roi_y, roi_width, roi_height) = (2u32, 1u32, 3u32, 2u32);
        let roi = HairRoi {
            x: roi_x,
            y: roi_y,
            width: roi_width,
            height: roi_height,
        };
        let run_case = |inputs: &PortraitJobInputs<'_>, expected: Vec<u8>| {
            let _empty_guard = empty_dispatch.then(ForceCudaEmptyDispatchGuard::new);
            let _failure_guard = (!empty_dispatch).then(ForceCudaFailureGuard::new);
            let mut actual = source.pixels.clone();
            let report = apply_portrait_effects_with_job(
                &mut actual,
                width,
                height,
                NormRect::FULL,
                &BTreeMap::from([(format!("pc:{pf}"), 0.8)]),
                inputs,
            );
            (
                actual,
                report.effects.into_iter().next().unwrap(),
                expected,
            )
        };
        if pf == 90139 {
            let hair =
                strided_plane(&[255, 180, 0, 96, 64, 220], roi_width, roi_height, 5, 1);
            let mix =
                strided_plane(&[180, 240, 64, 32, 128, 255], roi_width, roi_height, 4, 1);
            let blend = strided_plane(
                &[
                    20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 10, 30, 50, 70,
                    90, 110,
                ],
                roi_width,
                roi_height,
                14,
                3,
            );
            let detail =
                strided_plane(&[96, 127, 128, 160, 255, 0], roi_width, roi_height, 6, 1);
            let smooth = HairSmoothInputs {
                ticket: 17,
                width,
                height,
                roi: (roi_x, roi_y, roi_width, roi_height),
                mix_strength: 0.6,
                hair_mask: Some(&hair),
                mix_mask: Some(&mix),
                blend_rgb: Some(&blend),
                detail: Some(&detail),
                hair_stride: 5,
                mix_stride: 4,
                blend_stride: 14,
                detail_stride: 6,
            };
            let expected = hair_smooth_cpu(&source, roi, 0.8, &smooth);
            let inputs = PortraitJobInputs {
                hair_smooth: Some(smooth),
                ..PortraitJobInputs::from_renderer_job(1, &masks)
            };
            run_case(&inputs, expected)
        } else if pf == 90160 {
            let hair =
                strided_plane(&[255, 180, 0, 96, 64, 220], roi_width, roi_height, 5, 1);
            let detail =
                strided_plane(&[128, 127, 200, 64, 255, 0], roi_width, roi_height, 7, 1);
            let gloss = HairGlossInputs {
                ticket: 18,
                width,
                height,
                roi: (roi_x, roi_y, roi_width, roi_height),
                dynamic_scale: 0.5,
                hair_mask: Some(&hair),
                model_detail: Some(&detail),
                hair_stride: 5,
                detail_stride: 7,
            };
            let expected = hair_gloss_cpu(&source, roi, &gloss);
            let inputs = PortraitJobInputs {
                hair_gloss: Some(gloss),
                ..PortraitJobInputs::from_renderer_job(1, &masks)
            };
            run_case(&inputs, expected)
        } else {
            unreachable!("hair CUDA fault test only covers PF90139/PF90160")
        }
    }

    #[cfg(windows)]
    #[test]
    fn n855_pf90139_forced_cuda_failure_falls_back_to_cpu() {
        let (actual, run, expected) = run_hair_route_cuda_fault(90139, false);
        assert_eq!(actual, expected);
        assert_eq!(run.backend, PortraitBackend::Cpu);
        assert!(run.executed);
        assert!(run.dispatches.is_empty());
        assert!(run
            .fallback_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("test-forced CUDA failure")));
    }

    #[cfg(windows)]
    #[test]
    fn n855_pf90160_forced_cuda_failure_falls_back_to_cpu() {
        let (actual, run, expected) = run_hair_route_cuda_fault(90160, false);
        assert_eq!(actual, expected);
        assert_eq!(run.backend, PortraitBackend::Cpu);
        assert!(run.executed);
        assert!(run.dispatches.is_empty());
        assert!(run
            .fallback_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("test-forced CUDA failure")));
    }

    #[cfg(windows)]
    #[test]
    fn n855_empty_cuda_dispatch_falls_back_to_cpu_for_hair_routes() {
        for pf in [90139, 90160] {
            let (actual, run, expected) = run_hair_route_cuda_fault(pf, true);
            assert_eq!(actual, expected, "PF{pf}");
            assert_eq!(run.backend, PortraitBackend::Cpu, "PF{pf}");
            assert!(run.executed, "PF{pf}");
            assert!(run.dispatches.is_empty(), "PF{pf}");
            assert!(
                run.fallback_reason
                    .as_deref()
                    .is_some_and(|reason| reason.contains("CUDA returned no dispatch")),
                "PF{pf}: {:?}",
                run.fallback_reason
            );
        }
    }

    #[test]
    fn pf90088_stage_refuses_missing_same_job_inputs_before_dispatch() {
        let masks = PortraitMasks::solid(1.0);
        let (source, repair, raw, person, background) = stray_hair_fixture();
        let complete =
            stray_hair_stage_inputs(&masks, &repair, &raw, &person, &background);
        let cases = [
            (
                PortraitJobInputs {
                    stray_hair: StrayHairInputs {
                        hairfly_v1: None,
                        ..complete.stray_hair
                    },
                    ..PortraitJobInputs::from_renderer_job(557, &masks)
                },
                "PF90088_HAIRFLY_V1",
            ),
            (
                PortraitJobInputs {
                    stray_hair: StrayHairInputs {
                        raw_mask: None,
                        ..complete.stray_hair
                    },
                    ..PortraitJobInputs::from_renderer_job(557, &masks)
                },
                "PF90088_RAW_STRAY_HAIR_MASK",
            ),
            (
                PortraitJobInputs {
                    stray_hair: StrayHairInputs {
                        native_degree: None,
                        ..complete.stray_hair
                    },
                    ..PortraitJobInputs::from_renderer_job(557, &masks)
                },
                "PF90088_NATIVE_DEGREE",
            ),
        ];
        for (inputs, missing) in cases {
            let mut actual = source.pixels.clone();
            let report = apply_portrait_effects_with_job(
                &mut actual,
                source.width,
                source.height,
                NormRect::FULL,
                &BTreeMap::from([("pc:90088".to_string(), 0.8f32)]),
                &inputs,
            );
            assert_eq!(actual, source.pixels);
            assert_eq!(
                report.effects[0].refusal,
                Some(Refusal::MissingInput(missing))
            );
            assert!(report.effects[0].skipped);
            assert!(!report.effects[0].executed);
            assert!(report.effects[0].dispatches.is_empty());
        }
    }

    #[test]
    fn pf90088_producer_binds_fields_and_keeps_typed_refusal_order() {
        let masks = PortraitMasks::solid(1.0);
        let (source, repair, raw, person, background) = stray_hair_fixture();
        let complete = StrayHairProducerInputs {
            hairfly_v1: Some(repair.clone()),
            raw_mask: Some(raw.clone()),
            person_mask: Some(person.clone()),
            background_mask: Some(background.clone()),
            native_degree: Some(0.5),
            native_person_area: Some(1.0),
            native_background_area: Some(0.5),
        };
        let producer = ProducerInputs {
            stray_hair: Some(complete.clone()),
            ..Default::default()
        };
        let mut inputs = PortraitJobInputs::from_renderer_job(941, &masks);
        inputs.fill_from_producer_inputs(&producer);
        let staged = inputs.stray_hair;
        assert_eq!(staged.hairfly_v1, Some(&repair));
        assert_eq!(staged.raw_mask, Some(raw.as_slice()));
        assert_eq!(staged.person_mask, Some(person.as_slice()));
        assert_eq!(staged.background_mask, Some(background.as_slice()));
        assert_eq!(staged.native_degree, Some(0.5));
        assert_eq!(staged.native_person_area, Some(1.0));
        assert_eq!(staged.native_background_area, Some(0.5));

        let fingerprint = producer.fingerprint();
        assert_ne!(fingerprint, ProducerInputs::default().fingerprint());
        let changed_rows = [
            StrayHairProducerInputs {
                hairfly_v1: None,
                ..complete.clone()
            },
            StrayHairProducerInputs {
                raw_mask: None,
                ..complete.clone()
            },
            StrayHairProducerInputs {
                person_mask: None,
                ..complete.clone()
            },
            StrayHairProducerInputs {
                background_mask: None,
                ..complete.clone()
            },
            StrayHairProducerInputs {
                native_degree: Some(0.25),
                ..complete.clone()
            },
            StrayHairProducerInputs {
                native_degree: Some(0.0),
                ..complete.clone()
            },
            StrayHairProducerInputs {
                native_person_area: Some(0.75),
                ..complete.clone()
            },
            StrayHairProducerInputs {
                native_background_area: Some(0.25),
                ..complete
            },
        ];
        for changed in changed_rows {
            assert_ne!(
                fingerprint,
                ProducerInputs {
                    stray_hair: Some(changed),
                    ..Default::default()
                }
                .fingerprint()
            );
        }

        let cases = [
            (
                StrayHairProducerInputs::default(),
                "PF90088_NATIVE_DEGREE",
            ),
            (
                StrayHairProducerInputs {
                    native_degree: Some(0.5),
                    ..Default::default()
                },
                "PF90088_NATIVE_PERSON_AREA",
            ),
            (
                StrayHairProducerInputs {
                    native_degree: Some(0.5),
                    native_person_area: Some(1.0),
                    ..Default::default()
                },
                "PF90088_NATIVE_BACKGROUND_AREA",
            ),
            (
                StrayHairProducerInputs {
                    native_degree: Some(0.5),
                    native_person_area: Some(1.0),
                    native_background_area: Some(0.5),
                    ..Default::default()
                },
                "PF90088_HAIRFLY_V1",
            ),
            (
                StrayHairProducerInputs {
                    hairfly_v1: Some(repair.clone()),
                    native_degree: Some(0.5),
                    native_person_area: Some(1.0),
                    native_background_area: Some(0.5),
                    ..Default::default()
                },
                "PF90088_RAW_STRAY_HAIR_MASK",
            ),
        ];
        for (candidate, missing) in cases {
            let producer = ProducerInputs {
                stray_hair: Some(candidate),
                ..Default::default()
            };
            let mut inputs = PortraitJobInputs::from_renderer_job(941, &masks);
            inputs.fill_from_producer_inputs(&producer);
            let (actual, run) = run_stray_hair_stage(&source, &inputs, true);
            assert_eq!(actual, source.pixels);
            assert_eq!(run.refusal, Some(Refusal::MissingInput(missing)));
            assert!(run.skipped);
            assert!(!run.executed);
            assert!(run.dispatches.is_empty());
        }
    }

    #[test]
    fn pf90088_stage_cpu_preserves_alpha_and_half_away_rounding() {
        let masks = PortraitMasks::solid(1.0);
        let (source, repair, raw, person, background) = stray_hair_fixture();
        let expected = aurora_portrait::kernels::hair_body::stray_hair(
            &source,
            80.0,
            0.5,
            1.0,
            0.5,
            Some(&repair),
            Some(&raw),
            Some(&person),
            Some(&background),
        )
        .unwrap();
        let inputs =
            stray_hair_stage_inputs(&masks, &repair, &raw, &person, &background);
        let (actual, run) = run_stray_hair_stage(&source, &inputs, true);
        assert_eq!(actual, expected.pixels);
        assert_eq!(&actual[0..4], &[3, 120, 80, 17]);
        for index in 0..source.pixel_count() {
            assert_eq!(actual[index * 4 + 3], source.pixels[index * 4 + 3]);
        }
        assert_eq!(run.backend, PortraitBackend::Cpu);
        assert!(run.executed);
        assert!(run.dispatches.is_empty());
    }

    #[cfg(windows)]
    #[test]
    fn n855_pf90088_empty_cuda_dispatch_falls_back_to_cpu() {
        let masks = PortraitMasks::solid(1.0);
        let (source, repair, raw, person, background) = stray_hair_fixture();
        let expected = aurora_portrait::kernels::hair_body::stray_hair(
            &source,
            80.0,
            0.5,
            1.0,
            0.5,
            Some(&repair),
            Some(&raw),
            Some(&person),
            Some(&background),
        )
        .unwrap();
        let inputs =
            stray_hair_stage_inputs(&masks, &repair, &raw, &person, &background);
        let _empty = ForceCudaEmptyDispatchGuard::new();
        let mut actual = source.pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            source.width,
            source.height,
            NormRect::FULL,
            &BTreeMap::from([("pc:90088".to_string(), 0.8f32)]),
            &inputs,
        );
        let run = &report.effects[0];
        assert_eq!(actual, expected.pixels);
        assert_eq!(run.backend, PortraitBackend::Cpu);
        assert!(run.executed);
        assert!(run.dispatches.is_empty());
        assert!(run
            .fallback_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("CUDA returned no dispatch")));
    }

    #[cfg(windows)]
    #[test]
    fn pf90088_stage_cuda_matches_cpu_byte_for_byte() {
        let masks = PortraitMasks::solid(1.0);
        let (source, repair, raw, person, background) = stray_hair_fixture();
        let expected = aurora_portrait::kernels::hair_body::stray_hair(
            &source,
            80.0,
            0.5,
            1.0,
            0.5,
            Some(&repair),
            Some(&raw),
            Some(&person),
            Some(&background),
        )
        .unwrap();
        let inputs =
            stray_hair_stage_inputs(&masks, &repair, &raw, &person, &background);
        let (actual, run) = run_stray_hair_stage(&source, &inputs, false);
        if run.backend != PortraitBackend::Cuda {
            assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but PF90088 did not dispatch: {:?}",
                run.fallback_reason
            );
            return;
        }
        assert_eq!(actual, expected.pixels);
        assert_eq!(run.dispatches[0].kernel, "pf90088_stray_hair_kernel");
        assert!(!run.dispatches[0].device.is_empty());
        for index in 0..source.pixel_count() {
            assert_eq!(actual[index * 4 + 3], source.pixels[index * 4 + 3]);
        }
        eprintln!(
            "PF90088 stage device={} kernel={} cc={:?} driver={} ptx_isa={:?}",
            run.dispatches[0].device,
            run.dispatches[0].kernel,
            run.dispatches[0].compute_cap,
            run.dispatches[0].driver_version,
            run.dispatches[0].ptx_isa
        );
    }

    #[cfg(windows)]
    pub(super) struct ForceCudaFailureGuard;

    #[cfg(windows)]
    impl ForceCudaFailureGuard {
        pub(super) fn new() -> Self {
            FORCE_CUDA_FAILURE.with(|flag| flag.set(true));
            Self
        }
    }

    #[cfg(windows)]
    impl Drop for ForceCudaFailureGuard {
        fn drop(&mut self) {
            FORCE_CUDA_FAILURE.with(|flag| flag.set(false));
        }
    }

    #[cfg(windows)]
    struct ForceCudaEmptyDispatchGuard;

    #[cfg(windows)]
    impl ForceCudaEmptyDispatchGuard {
        fn new() -> Self {
            FORCE_CUDA_EMPTY_DISPATCH.with(|flag| flag.set(true));
            Self
        }
    }

    #[cfg(windows)]
    impl Drop for ForceCudaEmptyDispatchGuard {
        fn drop(&mut self) {
            FORCE_CUDA_EMPTY_DISPATCH.with(|flag| flag.set(false));
        }
    }

    #[cfg(windows)]
    #[test]
    fn hlf_wrinkle_forced_cpu_paths_match_their_oracles() {
        let width = 7u32;
        let height = 5u32;
        let pixels = width as usize * height as usize;
        let source = Image::new(width, height, frame(width, height)).unwrap();
        let base = Image::new(width, height, source.pixels.iter().rev().copied().collect()).unwrap();
        let residual = Image::solid(width, height, [51, 102, 153, 77]);
        let roi = Roi {
            x: 1,
            y: 1,
            width: 4,
            height: 3,
        };
        let source_rgb: Vec<f32> = (0..pixels * 3)
            .map(|i| (i * 17 % 251) as f32 / 255.0)
            .collect();
        let median_rgb: Vec<f32> = (0..pixels * 3)
            .map(|i| (i * 19 % 241) as f32 / 255.0)
            .collect();
        let gauss_rgb: Vec<f32> = (0..pixels * 3)
            .map(|i| (i * 23 % 239) as f32 / 255.0)
            .collect();
        let face_mask: Vec<f32> = (0..pixels).map(|i| (i % 7) as f32 / 6.0).collect();
        let face_mask_rgba: Vec<f32> = (0..pixels * 4)
            .map(|i| (i % 11) as f32 / 10.0)
            .collect();
        let skin_mask: Vec<f32> = (0..pixels).map(|i| (i % 5) as f32 / 4.0).collect();
        let wrinkle_mask: Vec<u8> = (0..pixels * 4).map(|i| (i * 13 % 256) as u8).collect();
        let wrinkle_skin: Vec<u8> = (0..pixels).map(|i| (i * 29 % 256) as u8).collect();
        let wrinkle_uniforms = WrinkleUniforms {
            legal_alpha: 1.0,
            dark_circle_alpha: 1.0,
            diff_factor: 1.0,
            diff_factor2: 0.0,
            color_threshold_factor: 0.0,
            mix_alpha: 1.0,
            diff_min_threshold: -1.0,
            diff_max_threshold: 1.0,
            high_freq_min_threshold: -1.0,
            high_freq_max_threshold: 1.0,
        };
        let hlf_body_uniforms = BodyHlfUniforms {
            spotless_degree: 0.0,
            retouch_degree: 1.0,
            spotless_acne_threshold: 0.0,
            spotless_high_freq_factor: 1.0,
            spotless_low_freq_alpha: 0.0,
            retouch_high_freq_factor: 0.0,
            diff_min_range: -0.2,
            diff_max_range: 0.2,
        };
        let masks = PortraitMasks::solid(1.0);
        let inputs = PortraitJobInputs {
            hlf_face: Some(HlfFaceInputs {
                width,
                height,
                source_rgb_full_f32: &source_rgb,
                median_rgb_full_f32: &median_rgb,
                gauss_rgb_full_f32: &gauss_rgb,
                face_mask_full_f32: &face_mask,
                skin_mask_full_f32: &skin_mask,
                roi,
            }),
            hlf_body: Some(HlfBodyInputs {
                width,
                height,
                median_rgb_full_f32: &median_rgb,
                gauss_rgb_full_f32: &gauss_rgb,
                face_mask_rgba_full_f32: &face_mask_rgba,
                skin_mask_full_f32: &skin_mask,
                roi,
                uniforms: hlf_body_uniforms,
                adjust_lut: None,
            }),
            wrinkle: Some(WrinkleInputs {
                base_full: &base,
                residual_full: &residual,
                mask_rgba_full: &wrinkle_mask,
                skin_mask_full: &wrinkle_skin,
                roi,
                uniforms: wrinkle_uniforms,
            }),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let _force = ForceCudaFailureGuard::new();
        for pf in [21025, 21069, 21070, 21071, 30021, 30022] {
            let expected = match pf {
                21025 | 21069 | 21070 | 21071 => skin_adjust::wrinkle_apply(
                    &source,
                    &base,
                    &residual,
                    &wrinkle_mask,
                    &wrinkle_skin,
                    roi,
                    skin_adjust::wrinkle_degree(pf, 50.0).unwrap(),
                    wrinkle_uniforms,
                )
                .unwrap(),
                30021 => {
                    let rgb = skin_adjust::hlf_face_apply(
                        &source_rgb,
                        &median_rgb,
                        &gauss_rgb,
                        &face_mask,
                        &skin_mask,
                        width,
                        height,
                        roi,
                        50.0,
                    )
                    .unwrap();
                    hlf_rgb_f32_to_image(&source, &rgb).unwrap()
                }
                30022 => skin_adjust::body_hlf_apply(
                    &source,
                    roi,
                    &median_rgb,
                    &gauss_rgb,
                    &face_mask_rgba,
                    &skin_mask,
                    50.0,
                    hlf_body_uniforms,
                    None,
                )
                .unwrap(),
                _ => unreachable!(),
            };
            let mut actual = source.pixels.clone();
            let extra = BTreeMap::from([(format!("pc:{pf}"), 0.5)]);
            let report = apply_portrait_effects_with_job(
                &mut actual,
                width,
                height,
                NormRect::FULL,
                &extra,
                &inputs,
            );
            assert_eq!(actual, expected.pixels, "PF{pf}");
            assert_eq!(report.effects[0].backend, PortraitBackend::Cpu);
            assert!(report.effects[0].executed);
            assert!(report.effects[0].dispatches.is_empty());
        }
    }

    #[test]
    fn portrait_effects_do_not_crash_on_empty_extra() {
        let mut rgba = frame(16, 16);
        apply(&mut rgba, 16, 16, &[]);
        assert_eq!(rgba.len(), 16 * 16 * 4);
    }

    #[test]
    fn portrait_effects_skip_refused_rows() {
        let mut rgba = frame(16, 16);
        let original = rgba.clone();
        apply(&mut rgba, 16, 16, &[("pc:90001", 50.0)]);
        assert_eq!(rgba, original, "refused rows must not change pixels");
    }

    #[test]
    fn masks_only_entry_is_a_typed_skip_without_dispatch() {
        let masks = PortraitMasks::solid(1.0);
        let extra = BTreeMap::from([
            ("pc:108".to_string(), 0.8f32),
            ("pc:90139".to_string(), 0.8),
            ("pc:90005".to_string(), 0.9),
        ]);
        let mut rgba = frame(16, 16);
        let original = rgba.clone();
        let report = apply_portrait_effects_with_report(
            &mut rgba,
            16,
            16,
            NormRect::FULL,
            &extra,
            &masks,
        );

        assert_eq!(rgba, original, "the legacy entry must not execute effects");
        assert_eq!(report.job_id, None);
        assert_eq!(report.effects.len(), 3);
        for effect in &report.effects {
            assert_eq!(effect.backend, PortraitBackend::Skipped);
            assert!(effect.skipped, "PF{} must be marked skipped", effect.pf);
            assert!(!effect.executed, "PF{} must not execute", effect.pf);
            assert!(
                effect.dispatches.is_empty(),
                "PF{} must not report a fake dispatch",
                effect.pf
            );
            assert_eq!(
                effect.refusal.as_ref(),
                Some(&Refusal::MissingInput("same-job input bundle"))
            );
            assert_eq!(
                effect.skip_reason,
                Some(PortraitSkipReason::InputRefused(Refusal::MissingInput(
                    "same-job input bundle"
                )))
            );
        }
    }

    #[test]
    fn non_skin_rows_never_reuse_parser_masks() {
        let masks = PortraitMasks::solid(1.0);
        let cases = [
            (108u32, "teeth mask"),
            (30026, "PF30026 same-job inputs"),
            (21026, "mask_a"),
            (90021, "PF90021_SAME_JOB_INPUTS"),
            (90002, "moisture face inputs"),
            (90010, "moisture body inputs"),
            (90033, "suit passes"),
            (90116, "aligned_color_field_rgb_f32"),
            (100007, "pf100007 material"),
        ];
        for (pf, missing) in cases {
            let mut rgba = frame(16, 16);
            let original = rgba.clone();
            let extra = BTreeMap::from([(format!("pc:{pf}"), 0.8f32)]);
            let inputs = PortraitJobInputs::from_renderer_job(7, &masks);
            let report =
                apply_portrait_effects_with_job(&mut rgba, 16, 16, NormRect::FULL, &extra, &inputs);
            assert_eq!(rgba, original, "PF{pf} must not zero-fill missing inputs");
            assert_eq!(report.effects.len(), 1);
            let effect = &report.effects[0];
            assert!(effect.skipped, "PF{pf} was skipped");
            assert!(!effect.executed, "PF{pf} was not executed");
            assert!(
                effect.dispatches.is_empty(),
                "PF{pf} must not fake a dispatch"
            );
            assert_eq!(
                effect.refusal.as_ref(),
                Some(&Refusal::MissingInput(missing)),
                "PF{pf} must name its own missing input"
            );
        }
    }

    fn suit_plan(order: u32) -> SuitPassPlan<'static> {
        let (effect, mask_field, mask_value) = match order {
            1..=3 => ("BlusherImageMakeup", "BlusherMaskType", 15),
            4..=5 => ("NormalEyeBrowImageMakeup", "EyeBrowMaskType", 5),
            6..=7 => ("SymEyeImageMakeup", "EyeMaskType", 10),
            8..=9 => ("SymEyeImageMakeup", "EyeMaskType", 4),
            10..=11 => ("SymEyeLashImageMakeup", "EyeLashMaskType", 4),
            12 => ("LipStickImageMakeup", "LipMaskType", 3),
            _ => panic!("suit order must be 1..=12"),
        };
        SuitPassPlan {
            order,
            effect,
            degree: Some(70.0),
            opacity: Some(49.0),
            filter_type: Some(42),
            material_path: Some("Resources/material.png"),
            mask_field: Some(mask_field),
            mask_value: Some(mask_value),
            shelter_type: Some(7),
            force_use_shelter: Some((4..=11).contains(&order)),
            region_rect: Some([95, 40, 901, 1081]),
            mix_color: Some([0, 0, 0, 255]),
        }
    }

    fn suit_stage_pass<'a>(
        mask: Option<&'a Image>,
        mask_uv: Option<&'a [[f32; 2]]>,
        material: Option<&'a Image>,
        material_uv: Option<&'a [[f32; 2]]>,
    ) -> SuitPassInput<'a> {
        SuitPassInput {
            kind: SuitPassKindInput::Image {
                material,
                material_uv,
            },
            mask,
            mask_uv,
            layer: Some(MakeupLayer {
                mix_color: [0, 0, 0, 255],
                opacity_percent: 49.0,
                degree01: 0.7,
                filter_type: 42,
                mask_channel: MakeupMaskChannel::Rgba,
            }),
            shelter: 0.5,
            region: 0.75,
        }
    }

    #[test]
    fn suit_producer_names_every_missing_same_job_input() {
        let plan = suit_plan(1);
        let mask = Image::solid(1, 1, [255, 255, 255, 255]);
        let material = Image::solid(1, 1, [20, 30, 40, 128]);
        let uv = [[0.5, 0.5]];
        let complete = SuitPassSameJobInputs {
            mask: Some(&mask),
            mask_uv: Some(&uv),
            material: Some(&material),
            material_uv: Some(&uv),
            shelter: Some(0.5),
            region: Some(0.75),
        };
        assert!(produce_suit_pass(1, &plan, complete).is_ok());

        let cases = [
            (
                SuitPassSameJobInputs {
                    mask: None,
                    ..complete
                },
                "suit mask",
            ),
            (
                SuitPassSameJobInputs {
                    mask_uv: None,
                    ..complete
                },
                "suit mask uv",
            ),
            (
                SuitPassSameJobInputs {
                    material: None,
                    ..complete
                },
                "suit material",
            ),
            (
                SuitPassSameJobInputs {
                    material_uv: None,
                    ..complete
                },
                "suit material uv",
            ),
            (
                SuitPassSameJobInputs {
                    shelter: None,
                    ..complete
                },
                "suit shelter",
            ),
            (
                SuitPassSameJobInputs {
                    region: None,
                    ..complete
                },
                "suit region",
            ),
        ];
        for (inputs, missing) in cases {
            assert_eq!(
                produce_suit_pass(1, &plan, inputs).err(),
                Some(Refusal::MissingInput(missing))
            );
        }
    }

    #[test]
    fn suit_stage_refuses_missing_existing_inputs() {
        let mask = Image::solid(1, 1, [255, 255, 255, 255]);
        let material = Image::solid(1, 1, [20, 30, 40, 128]);
        let uv = [[0.5, 0.5]];
        let cases = [
            (
                suit_stage_pass(None, Some(&uv), Some(&material), Some(&uv)),
                "suit mask",
            ),
            (
                suit_stage_pass(Some(&mask), None, Some(&material), Some(&uv)),
                "suit mask uv",
            ),
            (
                suit_stage_pass(Some(&mask), Some(&uv), None, Some(&uv)),
                "suit material",
            ),
            (
                suit_stage_pass(Some(&mask), Some(&uv), Some(&material), None),
                "suit material uv",
            ),
        ];
        for (pass, missing) in cases {
            assert_eq!(
                resolve_suit_passes(Some(&[pass])).err(),
                Some(Refusal::MissingInput(missing))
            );
        }
    }

    #[test]
    fn suit_producer_uses_only_pinned_mask_channels() {
        let cases = [
            ("BlusherMaskType", 15, MakeupMaskChannel::Rgba),
            ("EyeBrowMaskType", 5, MakeupMaskChannel::RB),
            ("EyeMaskType", 10, MakeupMaskChannel::Rgb),
            ("EyeMaskType", 4, MakeupMaskChannel::RG),
            ("EyeLashMaskType", 4, MakeupMaskChannel::RG),
            ("LipMaskType", 3, MakeupMaskChannel::R),
        ];
        for (field, value, expected) in cases {
            assert_eq!(suit_mask_channel(field, value), Ok(expected));
        }
        assert_eq!(
            suit_mask_channel("UnknownMaskType", 99),
            Err(Refusal::MissingInput("suit mask channel"))
        );
    }

    #[test]
    fn suit_stage_executes_explicit_unmasked_image_pass() {
        let source = Image::solid(1, 1, [10, 20, 30, 255]);
        let material = Image::solid(1, 1, [100, 150, 200, 128]);
        let uv = [[0.5, 0.5]];
        let layer = MakeupLayer {
            mix_color: [100, 150, 200, 180],
            opacity_percent: 80.0,
            degree01: 0.5,
            filter_type: 41,
            mask_channel: MakeupMaskChannel::A,
        };
        let input = SuitPassInput {
            kind: SuitPassKindInput::ImageUnmasked {
                material: Some(&material),
                material_uv: Some(&uv),
            },
            mask: None,
            mask_uv: None,
            layer: Some(layer),
            shelter: 1.0,
            region: 1.0,
        };
        let binding = [input];
        let resolved = resolve_suit_passes(Some(&binding)).unwrap();
        assert_eq!(resolved.len(), 1);
        let actual = apply_cpu_suit_pass(&source, &resolved[0]).unwrap();
        let expected = image_makeup_unmasked(&source, &material, &uv, layer).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn suit_producer_keeps_skip_shelter_and_region_rect_separate() {
        let mask = Image::solid(1, 1, [255, 255, 255, 255]);
        let material = Image::solid(1, 1, [20, 30, 40, 128]);
        let uv = [[0.5, 0.5]];
        let inputs = SuitPassSameJobInputs {
            mask: Some(&mask),
            mask_uv: Some(&uv),
            material: Some(&material),
            material_uv: Some(&uv),
            shelter: None,
            region: Some(0.75),
        };
        let mut plan = suit_plan(1);
        plan.force_use_shelter = Some(true);
        let before = produce_suit_pass(1, &plan, inputs).unwrap();
        plan.region_rect = Some([0, 0, 0, 0]);
        let after = produce_suit_pass(1, &plan, inputs).unwrap();
        assert_eq!(before.shelter, 1.0);
        assert_eq!(after.shelter, 1.0);
        assert_eq!(before.region, 0.75);
        assert_eq!(after.region, 0.75);
    }

    #[test]
    fn suit_producer_emits_twelve_ordered_existing_inputs() {
        let mask = Image::solid(1, 1, [255, 255, 255, 255]);
        let material = Image::solid(1, 1, [20, 30, 40, 128]);
        let uv = [[0.5, 0.5]];
        let plans: Vec<_> = (1..=12).map(suit_plan).collect();
        let jobs = vec![
            SuitPassSameJobInputs {
                mask: Some(&mask),
                mask_uv: Some(&uv),
                material: Some(&material),
                material_uv: Some(&uv),
                shelter: Some(0.5),
                region: Some(0.75),
            };
            12
        ];
        let produced = produce_suit_passes(1, Some(&plans), Some(&jobs)).unwrap();
        assert_eq!(produced.len(), 12);
        assert!(produced
            .iter()
            .all(|pass| pass.mask.is_some() && pass.mask_uv.is_some()));
        assert!(produced.iter().all(|pass| matches!(
            &pass.kind,
            SuitPassKindInput::Image {
                material: Some(_),
                material_uv: Some(_)
            }
        )));

        let resolved = resolve_suit_passes(Some(&produced)).unwrap();
        assert_eq!(resolved.len(), 12);
        assert_eq!(
            resolved
                .iter()
                .map(|pass| pass.layer.mask_channel)
                .collect::<Vec<_>>(),
            vec![
                MakeupMaskChannel::Rgba,
                MakeupMaskChannel::Rgba,
                MakeupMaskChannel::Rgba,
                MakeupMaskChannel::RB,
                MakeupMaskChannel::RB,
                MakeupMaskChannel::Rgb,
                MakeupMaskChannel::Rgb,
                MakeupMaskChannel::RG,
                MakeupMaskChannel::RG,
                MakeupMaskChannel::RG,
                MakeupMaskChannel::RG,
                MakeupMaskChannel::R,
            ]
        );
        assert!(resolved.iter().all(|pass| pass.shelter == 1.0
            || (pass.shelter == 0.5 && pass.region == 0.75)));
    }

    fn pf90021_volume(offset: f32) -> VolumeLut64 {
        let edge = aurora_portrait::cuda_pf90021::LUT_EDGE;
        let mut nodes = vec![0.0f32; aurora_portrait::cuda_pf90021::LUT_FLOATS];
        let denominator = (3 * (edge - 1)) as f32;
        for z in 0..edge {
            for y in 0..edge {
                for x in 0..edge {
                    let at = ((z * edge + y) * edge + x) * 3;
                    nodes[at] = ((x + y) as f32 / denominator + offset).clamp(0.0, 1.0);
                    nodes[at + 1] =
                        ((y + z) as f32 / denominator + offset).clamp(0.0, 1.0);
                    nodes[at + 2] =
                        ((x + z) as f32 / denominator + offset).clamp(0.0, 1.0);
                }
            }
        }
        VolumeLut64::from_nodes(nodes).unwrap()
    }

    fn pf90021_runtime() -> Pf90021RuntimeScalars {
        Pf90021RuntimeScalars {
            degree01: 0.73,
            opacity01: 0.91,
            shelter: 0.83,
            region: 0.67,
        }
    }

    fn pf90021_runtime_receipt(
        runtime: Pf90021RuntimeScalars,
    ) -> Pf90021RuntimeReceipt {
        Pf90021RuntimeReceipt {
            degree01: runtime.degree01,
            opacity01: runtime.opacity01,
            shelter01: runtime.shelter,
            region01: runtime.region,
        }
    }

    fn pf90021_test_receipts(
        width: u32,
        height: u32,
        masks: &PortraitMasks,
        direction: Pf90021Direction,
        degree_input: f32,
    ) -> (
        PortraitJobReceipt,
        Pf90021SameJobBinding,
        Pf90021DirectionReceipt,
    ) {
        let params = BTreeMap::from([("pc:90021".to_string(), 0.8f32)]);
        let receipt = PortraitJobReceipt::from_renderer_receipt(
            &frame(width, height),
            width,
            height,
            NormRect::FULL,
            &params,
            masks,
        );
        let effect_range = match direction {
            Pf90021Direction::Positive => [0.0, 0.0, 1.0],
            Pf90021Direction::Negative => [1.0, 0.0, 0.0],
            Pf90021Direction::Neutral => unreachable!(),
        };
        (
            receipt,
            Pf90021SameJobBinding {
                job_id: 21,
                face_id: 0x162c,
                ticket: 21,
                frame: receipt.frame_receipt(),
                mask_texture_id: 0x6200,
                mask_channel: 0,
                mask_source: Pf90021MaskSource::FaceParsingChannel0Rasterized,
                roi_bits: [
                    0.0f32.to_bits(),
                    0.0f32.to_bits(),
                    (width as f32).to_bits(),
                    (height as f32).to_bits(),
                ],
            },
            Pf90021DirectionReceipt {
                effect_range,
                param_range: [0.0, 0.5, 1.0],
                input01: degree_input,
                output01: pf90021_degree_remap(direction, degree_input),
            },
        )
    }

    #[test]
    fn pf90021_stage_refuses_missing_assets_and_runtime_before_launch() {
        let masks = PortraitMasks::solid(1.0);
        let positive = pf90021_volume(0.0);
        let negative = pf90021_volume(0.25);
        let mask = vec![128u8; 16 * 16];
        let base = Pf90021Inputs {
            direction: Pf90021Direction::Positive,
            positive_lut: Some(&positive),
            negative_lut: Some(&negative),
            mask_r: Some(&mask),
            runtime: Some(pf90021_runtime()),
            same_job: None,
            direction_receipt: None,
            runtime_receipt: None,
        };
        let cases = [
            (None, "PF90021_SAME_JOB_INPUTS"),
            (
                Some(Pf90021Inputs {
                    positive_lut: None,
                    ..base
                }),
                "PF90021_POSITIVE_LUT64",
            ),
            (
                Some(Pf90021Inputs {
                    direction: Pf90021Direction::Negative,
                    negative_lut: None,
                    ..base
                }),
                "PF90021_NEGATIVE_LUT64",
            ),
            (
                Some(Pf90021Inputs {
                    mask_r: None,
                    ..base
                }),
                "PF90021_MASK_R",
            ),
            (
                Some(Pf90021Inputs {
                    runtime: None,
                    ..base
                }),
                "PF90021_RUNTIME_SCALARS",
            ),
        ];
        for (pf90021, missing) in cases {
            let mut rgba = frame(16, 16);
            let original = rgba.clone();
            let inputs = PortraitJobInputs {
                pf90021,
                ..PortraitJobInputs::from_renderer_job(9, &masks)
            };
            let report = apply_portrait_effects_with_job(
                &mut rgba,
                16,
                16,
                NormRect::FULL,
                &BTreeMap::from([("pc:90021".to_string(), 0.8)]),
                &inputs,
            );
            let run = &report.effects[0];
            assert_eq!(rgba, original, "{missing} must not change pixels");
            assert_eq!(run.backend, PortraitBackend::Skipped, "{missing}");
            assert!(run.skipped, "{missing}");
            assert!(!run.executed, "{missing}");
            assert!(run.dispatches.is_empty(), "{missing}");
            assert_eq!(
                run.refusal,
                Some(Refusal::MissingInput(missing)),
                "{missing}"
            );
            assert_eq!(
                run.skip_reason,
                Some(PortraitSkipReason::InputRefused(Refusal::MissingInput(
                    missing
                ))),
                "{missing}"
            );
        }
    }

    #[test]
    fn pf90021_stage_rejects_foreign_frame_and_direction_receipts() {
        let (width, height) = (16u32, 16u32);
        let masks = PortraitMasks::solid(1.0);
        let params = BTreeMap::from([("pc:90021".to_string(), 0.8f32)]);
        let (_, same_job, direction_receipt) =
            pf90021_test_receipts(
                width,
                height,
                &masks,
                Pf90021Direction::Positive,
                0.865,
            );
        let positive = pf90021_volume(0.0);
        let negative = pf90021_volume(0.25);
        let mask = vec![128u8; width as usize * height as usize];
        let runtime = pf90021_runtime();
        let runtime_receipt = pf90021_runtime_receipt(runtime);

        let mut foreign_frame = frame(width, height);
        foreign_frame[0] ^= 1;
        let foreign_same_job = Pf90021SameJobBinding {
            frame: PortraitFrameReceipt::from_renderer_frame(
                &foreign_frame,
                width,
                height,
            )
            .unwrap(),
            ..same_job
        };
        let current_receipt = PortraitJobReceipt::from_renderer_receipt(
            &frame(width, height),
            width,
            height,
            NormRect::FULL,
            &params,
            &masks,
        );
        let mut inputs =
            PortraitJobInputs::from_renderer_receipt(current_receipt, &masks);
        inputs.pf90021 = Some(Pf90021Inputs {
            direction: Pf90021Direction::Positive,
            positive_lut: Some(&positive),
            negative_lut: Some(&negative),
            mask_r: Some(&mask),
            runtime: Some(runtime),
            same_job: Some(&foreign_same_job),
            direction_receipt: Some(&direction_receipt),
            runtime_receipt: Some(&runtime_receipt),
        });
        let (actual, run) =
            apply_one_with_inputs(90021, width, height, &inputs);
        assert_eq!(actual, frame(width, height));
        assert_eq!(run.backend, PortraitBackend::Skipped);
        assert!(run.dispatches.is_empty());
        assert_eq!(
            run.refusal,
            Some(Refusal::ValueOutOfRange(
                "PF90021_CURRENT_FRAME_RECEIPT"
            ))
        );

        let wrong_direction = Pf90021DirectionReceipt {
            effect_range: [1.0, 0.0, 0.0],
            ..direction_receipt
        };
        let mut inputs =
            PortraitJobInputs::from_renderer_receipt(current_receipt, &masks);
        inputs.pf90021 = Some(Pf90021Inputs {
            direction: Pf90021Direction::Positive,
            positive_lut: Some(&positive),
            negative_lut: Some(&negative),
            mask_r: Some(&mask),
            runtime: Some(runtime),
            same_job: Some(&same_job),
            direction_receipt: Some(&wrong_direction),
            runtime_receipt: Some(&runtime_receipt),
        });
        let (actual, run) =
            apply_one_with_inputs(90021, width, height, &inputs);
        assert_eq!(actual, frame(width, height));
        assert_eq!(run.backend, PortraitBackend::Skipped);
        assert!(run.dispatches.is_empty());
        assert_eq!(
            run.refusal,
            Some(Refusal::ValueOutOfRange(
                "PF90021_DIRECTION_RECEIPT"
            ))
        );
    }

    #[cfg(windows)]
    #[test]
    fn pf90021_same_job_stage_matches_cpu_on_positive_and_negative() {
        let (width, height) = (16u32, 16u32);
        let masks = PortraitMasks::solid(1.0);
        let positive = pf90021_volume(0.0);
        let negative = pf90021_volume(0.25);
        let mask: Vec<u8> = (0..width as usize * height as usize)
            .map(|index| {
                let x = index % width as usize;
                let y = index / width as usize;
                if (4..12).contains(&x) && (4..12).contains(&y) {
                    (index * 37 % 256) as u8
                } else {
                    0
                }
            })
            .collect();
        let runtime = pf90021_runtime();
        let source = Image::new(width, height, frame(width, height)).unwrap();

        for (direction, lut) in [
            (Pf90021Direction::Positive, &positive),
            (Pf90021Direction::Negative, &negative),
        ] {
            let degree_input = match direction {
                Pf90021Direction::Positive => 0.865,
                Pf90021Direction::Negative => 0.135,
                Pf90021Direction::Neutral => unreachable!(),
            };
            let (receipt, same_job, direction_receipt) =
                pf90021_test_receipts(
                    width,
                    height,
                    &masks,
                    direction,
                    degree_input,
                );
            let runtime_receipt = pf90021_runtime_receipt(runtime);
            let expected = aurora_portrait::cuda_pf90021::apply_cpu(
                &source, lut, &mask, runtime,
            )
            .unwrap();
            let mut inputs = PortraitJobInputs::from_renderer_receipt(receipt, &masks);
            inputs.pf90021 = Some(Pf90021Inputs {
                    direction,
                    positive_lut: Some(&positive),
                    negative_lut: Some(&negative),
                    mask_r: Some(&mask),
                    runtime: Some(runtime),
                    same_job: Some(&same_job),
                    direction_receipt: Some(&direction_receipt),
                    runtime_receipt: Some(&runtime_receipt),
                });
            let (actual, run) = apply_one_with_inputs(90021, width, height, &inputs);
            if run.backend != PortraitBackend::Cuda {
                assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "CUDA required but PF90021 did not dispatch: {:?}",
                    run.fallback_reason
                );
                return;
            }
            assert_eq!(actual, expected.pixels, "{direction:?}");
            let differing_bytes = actual
                .iter()
                .zip(&expected.pixels)
                .filter(|(actual, expected)| actual != expected)
                .count();
            let alpha_differing_pixels = actual
                .chunks_exact(4)
                .zip(expected.pixels.chunks_exact(4))
                .filter(|(actual, expected)| actual[3] != expected[3])
                .count();
            let mut changed_outside = 0usize;
            for y in 0..height as usize {
                for x in 0..width as usize {
                    let at = (y * width as usize + x) * 4;
                    assert_eq!(
                        actual[at + 3], source.pixels[at + 3],
                        "alpha changed at {x},{y}"
                    );
                    if !((4..12).contains(&x) && (4..12).contains(&y)) {
                        if actual[at..at + 3] != source.pixels[at..at + 3] {
                            changed_outside += 1;
                        }
                        assert_eq!(
                            &actual[at..at + 3],
                            &source.pixels[at..at + 3],
                            "ROI-exterior changed at {x},{y}"
                        );
                    }
                }
            }
            assert_eq!(differing_bytes, 0, "{direction:?}");
            assert_eq!(alpha_differing_pixels, 0, "{direction:?}");
            assert_eq!(changed_outside, 0, "{direction:?}");
            eprintln!(
                "PF90021_VERIFY direction={direction:?} device={} compared_bytes={} differing_bytes={differing_bytes} alpha_differing_pixels={alpha_differing_pixels} changed_outside_roi={changed_outside}",
                run.dispatches[0].device,
                actual.len(),
            );
            assert!(!run.skipped, "{direction:?}");
            assert!(run.executed, "{direction:?}");
            assert_eq!(
                run.dispatches
                    .iter()
                    .map(|dispatch| dispatch.kernel)
                    .collect::<Vec<_>>(),
                vec![
                    aurora_portrait::cuda_pf90021::LUT_PASS_KERNEL,
                    aurora_portrait::cuda_pf90021::COMPOSITE_KERNEL,
                ],
                "{direction:?}"
            );
            eprintln!(
                "PF90021 {direction:?} device={} kernels={:?}",
                run.dispatches[0].device,
                run.dispatches
                    .iter()
                    .map(|dispatch| dispatch.kernel)
                    .collect::<Vec<_>>()
            );
            for (pixel, (actual_pixel, expected_pixel)) in actual
                .chunks_exact(4)
                .zip(expected.pixels.chunks_exact(4))
                .enumerate()
            {
                assert_eq!(actual_pixel[3], expected_pixel[3], "alpha {pixel}");
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn pf90021_stage_cpu_fallback_matches_reference() {
        let (width, height) = (16u32, 16u32);
        let masks = PortraitMasks::solid(1.0);
        let positive = pf90021_volume(0.0);
        let negative = pf90021_volume(0.25);
        let mask: Vec<u8> = (0..width as usize * height as usize)
            .map(|index| {
                let x = index % width as usize;
                let y = index / width as usize;
                if (4..12).contains(&x) && (4..12).contains(&y) {
                    (index * 37 % 256) as u8
                } else {
                    0
                }
            })
            .collect();
        let runtime = pf90021_runtime();
        let source = Image::new(width, height, frame(width, height)).unwrap();
        let (receipt, same_job, direction_receipt) =
            pf90021_test_receipts(
                width,
                height,
                &masks,
                Pf90021Direction::Negative,
                0.135,
            );
        let runtime_receipt = pf90021_runtime_receipt(runtime);
        let expected = aurora_portrait::cuda_pf90021::apply_cpu(
            &source, &negative, &mask, runtime,
        )
        .unwrap();
        let mut inputs = PortraitJobInputs::from_renderer_receipt(receipt, &masks);
        inputs.pf90021 = Some(Pf90021Inputs {
            direction: Pf90021Direction::Negative,
            positive_lut: Some(&positive),
            negative_lut: Some(&negative),
            mask_r: Some(&mask),
            runtime: Some(runtime),
            same_job: Some(&same_job),
            direction_receipt: Some(&direction_receipt),
            runtime_receipt: Some(&runtime_receipt),
        });

        FORCE_CUDA_FAILURE.with(|flag| flag.set(true));
        let (actual, run) =
            apply_one_with_inputs(90021, width, height, &inputs);
        FORCE_CUDA_FAILURE.with(|flag| flag.set(false));

        assert_eq!(actual, expected.pixels);
        assert_eq!(run.backend, PortraitBackend::Cpu, "{run:?}");
        assert!(run.executed, "{run:?}");
        assert!(!run.skipped, "{run:?}");
        assert!(run.dispatches.is_empty(), "{run:?}");
        assert!(run.fallback_reason.is_some(), "{run:?}");
        for y in 0..height as usize {
            for x in 0..width as usize {
                let at = (y * width as usize + x) * 4;
                assert_eq!(
                    actual[at + 3], source.pixels[at + 3],
                    "alpha changed at {x},{y}"
                );
                if !((4..12).contains(&x) && (4..12).contains(&y)) {
                    assert_eq!(
                        &actual[at..at + 3],
                        &source.pixels[at..at + 3],
                        "ROI-exterior changed at {x},{y}"
                    );
                }
            }
        }
    }

    #[test]
    fn pf30026_same_job_stage_matches_cpu_or_reports_cuda_unavailable() {
        let (width, height) = (9u32, 7u32);
        let pixels = (width * height) as usize;
        let masks = PortraitMasks::solid(1.0);
        let source_pixels = frame(width, height);
        let source = Image::new(width, height, source_pixels.clone()).unwrap();
        let mut effect_pixels = source_pixels.clone();
        for (index, pixel) in effect_pixels.chunks_exact_mut(4).enumerate() {
            pixel[0] = pixel[0].saturating_add((index % 19) as u8);
            pixel[1] = pixel[1].saturating_add(21);
            pixel[2] = pixel[2].saturating_add(17);
            pixel[3] = (index * 13 % 256) as u8;
        }
        let effect = Image::new(width, height, effect_pixels).unwrap();
        let teeth_mask: Vec<u8> = (0..pixels).map(|i| ((i * 29 + 7) % 256) as u8).collect();
        let edge_plane_a = vec![0.5f32; pixels];
        let edge_plane_b = vec![0.5f32; pixels];
        let alignment = TeethAlignmentInputs {
            effect_rgb: Some(&effect),
            teeth_mask: Some(&teeth_mask),
            edge_plane_a: Some(&edge_plane_a),
            edge_plane_b: Some(&edge_plane_b),
            gate_binding: Some(pf30026_binding(1)),
            roi: Some(NormRect::FULL),
        };
        let inputs = PortraitJobInputs {
            teeth_alignment: Some(alignment),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut actual = source_pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:30026".to_string(), 0.8)]),
            &inputs,
        );
        let run = &report.effects[0];
        let job = AlignmentFinalBlendJob {
            original: Some(&source),
            effect_rgb: Some(&effect),
            teeth_mask: Some(&teeth_mask),
            edge_plane_a: Some(&edge_plane_a),
            edge_plane_b: Some(&edge_plane_b),
        };
        let expected =
            aurora_portrait::kernels::teeth_blend::apply_alignment_final_blend(job).unwrap();
        match run.backend {
            PortraitBackend::Cuda => {
                assert!(run.executed);
                assert_eq!(run.dispatches.len(), 1);
                assert_eq!(run.dispatches[0].kernel, aurora_portrait::cuda_teeth_blend::KERNEL);
                let AlignmentFinalBlendOutcome::Applied(expected) = expected else {
                    panic!("open edge gate produced a CPU-gated oracle");
                };
                assert_eq!(actual, expected.pixels, "PF30026 CUDA/CPU byte parity");
                assert!(run.skip_reason.is_none());
                eprintln!(
                    "PF30026 stage device={} kernel={} grid={:?} block={:?} ptx_isa={:?}",
                    run.dispatches[0].device,
                    run.dispatches[0].kernel,
                    run.dispatches[0].grid,
                    run.dispatches[0].block,
                    run.dispatches[0].ptx_isa
                );
            }
            PortraitBackend::Skipped => {
                assert!(!run.executed);
                assert!(run.dispatches.is_empty());
                assert_eq!(actual, source_pixels);
                assert!(matches!(
                    run.skip_reason,
                    Some(PortraitSkipReason::Cuda(CudaError::Unavailable(_)))
                        | Some(PortraitSkipReason::Cuda(CudaError::Driver(_, _)))
                ));
                assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "CUDA required but PF30026 skipped: {:?}",
                    run.skip_reason
                );
            }
            PortraitBackend::Cpu => {
                assert!(run.executed);
                assert!(run.dispatches.is_empty());
                assert!(run.fallback_reason.is_some());
                let AlignmentFinalBlendOutcome::Applied(expected) = expected else {
                    panic!("open edge gate produced a CPU-gated oracle");
                };
                assert_eq!(actual, expected.pixels, "PF30026 CPU fallback parity");
            }
        }
    }

    #[test]
    fn pf30026_closed_gate_is_typed_and_non_destructive() {
        let (width, height) = (5u32, 4u32);
        let pixels = (width * height) as usize;
        let masks = PortraitMasks::solid(1.0);
        let source_pixels = frame(width, height);
        let effect = Image::solid(width, height, [10, 20, 30, 40]);
        let closed = vec![0.1f32; pixels];
        let teeth_mask = vec![128u8; pixels];
        let alignment = TeethAlignmentInputs {
            effect_rgb: Some(&effect),
            teeth_mask: Some(&teeth_mask),
            edge_plane_a: Some(&closed),
            edge_plane_b: Some(&closed),
            gate_binding: Some(pf30026_binding(1)),
            roi: Some(NormRect::FULL),
        };
        let inputs = PortraitJobInputs {
            teeth_alignment: Some(alignment),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut actual = source_pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:30026".to_string(), 0.8)]),
            &inputs,
        );
        let run = &report.effects[0];
        assert_eq!(run.backend, PortraitBackend::Skipped);
        assert!(!run.executed);
        assert!(run.dispatches.is_empty());
        assert_eq!(run.skip_reason, Some(PortraitSkipReason::EdgeGateClosed));
        assert_eq!(actual, source_pixels);
    }

    #[test]
    fn pf30026_missing_inputs_and_roi_mismatch_are_typed() {
        let (width, height) = (4u32, 3u32);
        let pixels = (width * height) as usize;
        let masks = PortraitMasks::solid(1.0);
        let source_pixels = frame(width, height);
        let effect = Image::solid(width, height, [10, 20, 30, 40]);
        let teeth_mask = [128u8; 12];
        let edge = vec![0.5f32; pixels];
        let open = TeethAlignmentInputs {
            effect_rgb: Some(&effect),
            teeth_mask: Some(&teeth_mask),
            edge_plane_a: Some(&edge),
            edge_plane_b: Some(&edge),
            gate_binding: Some(pf30026_binding(1)),
            roi: Some(NormRect::FULL),
        };
        for (alignment, missing) in [
            (
                TeethAlignmentInputs {
                    effect_rgb: None,
                    ..open
                },
                "PF30026 effect RGB",
            ),
            (
                TeethAlignmentInputs {
                    teeth_mask: None,
                    ..open
                },
                "PF30026 teeth mask",
            ),
            (
                TeethAlignmentInputs {
                    gate_binding: None,
                    ..open
                },
                "PF30026 edge gate binding",
            ),
            (
                TeethAlignmentInputs {
                    edge_plane_a: None,
                    ..open
                },
                "PF30026 edge plane A",
            ),
            (
                TeethAlignmentInputs {
                    edge_plane_b: None,
                    ..open
                },
                "PF30026 edge plane B",
            ),
            (
                TeethAlignmentInputs { roi: None, ..open },
                "PF30026 ROI",
            ),
        ] {
            let inputs = PortraitJobInputs {
                teeth_alignment: Some(alignment),
                ..PortraitJobInputs::from_renderer_job(1, &masks)
            };
            let mut actual = source_pixels.clone();
            let report = apply_portrait_effects_with_job(
                &mut actual,
                width,
                height,
                NormRect::FULL,
                &BTreeMap::from([("pc:30026".to_string(), 0.8)]),
                &inputs,
            );
            let run = &report.effects[0];
            assert_eq!(run.backend, PortraitBackend::Skipped);
            assert!(!run.executed);
            assert_eq!(run.refusal, Some(Refusal::MissingInput(missing)));
            assert_eq!(
                run.skip_reason,
                Some(PortraitSkipReason::InputRefused(
                    Refusal::MissingInput(missing)
                ))
            );
            assert_eq!(actual, source_pixels);
        }

        let alignment = TeethAlignmentInputs {
            roi: Some(NormRect {
                x: 0.1,
                y: 0.0,
                width: 0.8,
                height: 1.0,
            }),
            ..open
        };
        let inputs = PortraitJobInputs {
            teeth_alignment: Some(alignment),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut actual = source_pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:30026".to_string(), 0.8)]),
            &inputs,
        );
        assert_eq!(
            report.effects[0].refusal,
            Some(Refusal::ValueOutOfRange("PF30026 ROI"))
        );
        assert_eq!(actual, source_pixels);
    }

    #[test]
    fn pf30026_producer_preserves_fields_and_typed_missing() {
        let (width, height) = (4u32, 3u32);
        let pixels = (width * height) as usize;
        let masks = PortraitMasks::solid(1.0);
        let source_pixels = frame(width, height);
        let effect = Image::solid(width, height, [10, 20, 30, 40]);
        let teeth_mask = vec![128u8; pixels];
        let edge = vec![0.5f32; pixels];
        let alignment = TeethAlignmentProducerInputs {
            effect_rgb: Some(effect.clone()),
            teeth_mask: Some(teeth_mask.clone()),
            edge_plane_a: Some(edge.clone()),
            edge_plane_b: Some(edge.clone()),
            gate_binding: Some(pf30026_binding(1)),
            roi: Some(NormRect::FULL),
        };
        let producer = ProducerInputs {
            teeth_alignment: Some(alignment.clone()),
            ..Default::default()
        };

        let mut inputs = PortraitJobInputs::from_renderer_job(1, &masks);
        inputs.fill_from_producer_inputs(&producer);
        let staged = inputs.teeth_alignment.expect("producer row is present");
        assert_eq!(staged.effect_rgb, Some(&effect));
        assert_eq!(staged.teeth_mask, Some(teeth_mask.as_slice()));
        assert_eq!(staged.edge_plane_a, Some(edge.as_slice()));
        assert_eq!(staged.edge_plane_b, Some(edge.as_slice()));
        assert_eq!(staged.gate_binding, Some(pf30026_binding(1)));
        assert_eq!(staged.roi, Some(NormRect::FULL));

        let expected = aurora_portrait::kernels::teeth_blend::apply_alignment_final_blend(
            AlignmentFinalBlendJob {
                original: Some(&Image::new(width, height, source_pixels.clone()).unwrap()),
                effect_rgb: Some(&effect),
                teeth_mask: Some(&teeth_mask),
                edge_plane_a: Some(&edge),
                edge_plane_b: Some(&edge),
            },
        )
        .unwrap();
        let AlignmentFinalBlendOutcome::Applied(expected) = expected else {
            panic!("open edge gate produced a CPU-gated oracle");
        };
        let mut actual = source_pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:30026".to_string(), 0.8)]),
            &inputs,
        );
        let run = &report.effects[0];
        assert!(run.executed);
        assert!(run.refusal.is_none());
        assert_eq!(actual, expected.pixels);
        for (pixel, source_pixel) in actual
            .chunks_exact(4)
            .zip(source_pixels.chunks_exact(4))
        {
            assert_eq!(pixel[3], source_pixel[3], "PF30026 alpha must stay original");
        }
        eprintln!(
            "PF30026 producer path backend={:?} executed={} dispatches={} alpha=original roi={:?}",
            run.backend,
            run.executed,
            run.dispatches.len(),
            NormRect::FULL
        );

        let mut changed = producer.clone();
        changed.teeth_alignment.as_mut().unwrap().edge_plane_b = Some(vec![0.25f32; pixels]);
        assert_ne!(producer.fingerprint(), changed.fingerprint());

        let cases = [
            (
                TeethAlignmentProducerInputs {
                    effect_rgb: None,
                    ..alignment.clone()
                },
                "PF30026 effect RGB",
            ),
            (
                TeethAlignmentProducerInputs {
                    teeth_mask: None,
                    ..alignment.clone()
                },
                "PF30026 teeth mask",
            ),
            (
                TeethAlignmentProducerInputs {
                    gate_binding: None,
                    ..alignment.clone()
                },
                "PF30026 edge gate binding",
            ),
            (
                TeethAlignmentProducerInputs {
                    edge_plane_a: None,
                    ..alignment.clone()
                },
                "PF30026 edge plane A",
            ),
            (
                TeethAlignmentProducerInputs {
                    edge_plane_b: None,
                    ..alignment.clone()
                },
                "PF30026 edge plane B",
            ),
            (
                TeethAlignmentProducerInputs {
                    roi: None,
                    ..alignment
                },
                "PF30026 ROI",
            ),
        ];
        for (alignment, missing) in cases {
            let producer = ProducerInputs {
                teeth_alignment: Some(alignment),
                ..Default::default()
            };
            let mut inputs = PortraitJobInputs::from_renderer_job(1, &masks);
            inputs.fill_from_producer_inputs(&producer);
            let mut actual = source_pixels.clone();
            let report = apply_portrait_effects_with_job(
                &mut actual,
                width,
                height,
                NormRect::FULL,
                &BTreeMap::from([("pc:30026".to_string(), 0.8)]),
                &inputs,
            );
            assert_eq!(
                report.effects[0].refusal,
                Some(Refusal::MissingInput(missing))
            );
            assert_eq!(actual, source_pixels);
        }
    }

    #[test]
    fn pf30026_edge_gate_binding_rejects_wrong_or_ambiguous_job() {
        assert_eq!(
            Pf30026GateBinding::from_same_job(
                0,
                Pf30026EdgeOutput::Output0,
                Pf30026EdgeOutput::Output1,
            ),
            Err(Refusal::MissingInput(
                "PF30026 edge gate same-job ticket"
            ))
        );
        assert_eq!(
            Pf30026GateBinding::from_same_job(
                7,
                Pf30026EdgeOutput::Output0,
                Pf30026EdgeOutput::Output0,
            ),
            Err(Refusal::ValueOutOfRange(
                "PF30026 edge gate binding"
            ))
        );

        let (width, height) = (4u32, 3u32);
        let masks = PortraitMasks::solid(1.0);
        let source_pixels = frame(width, height);
        let effect = Image::solid(width, height, [10, 20, 30, 40]);
        let edge = vec![0.5f32; 12];
        let alignment = TeethAlignmentInputs {
            effect_rgb: Some(&effect),
            teeth_mask: Some(&[128; 12]),
            edge_plane_a: Some(&edge),
            edge_plane_b: Some(&edge),
            gate_binding: Some(pf30026_binding(8)),
            roi: Some(NormRect::FULL),
        };
        let inputs = PortraitJobInputs {
            teeth_alignment: Some(alignment),
            ..PortraitJobInputs::from_renderer_job(7, &masks)
        };
        let mut actual = source_pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:30026".to_string(), 0.8)]),
            &inputs,
        );
        assert_eq!(actual, source_pixels);
        assert_eq!(
            report.effects[0].refusal,
            Some(Refusal::MissingInput(
                "PF30026 edge gate same-job ticket"
            ))
        );
    }

    #[cfg(windows)]
    #[test]
    fn n855_pf30026_forced_cuda_failure_falls_back_to_cpu() {
        let (width, height) = (4u32, 3u32);
        let masks = PortraitMasks::solid(1.0);
        let source_pixels = frame(width, height);
        let effect = Image::solid(width, height, [10, 20, 30, 40]);
        let edge = [0.5f32; 12];
        let alignment = TeethAlignmentInputs {
            effect_rgb: Some(&effect),
            teeth_mask: Some(&[128; 12]),
            edge_plane_a: Some(&edge),
            edge_plane_b: Some(&edge),
            gate_binding: Some(pf30026_binding(1)),
            roi: Some(NormRect::FULL),
        };
        let inputs = PortraitJobInputs {
            teeth_alignment: Some(alignment),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut actual = source_pixels.clone();
        let _force = ForceCudaFailureGuard::new();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:30026".to_string(), 0.8)]),
            &inputs,
        );
        let run = &report.effects[0];
        let expected = aurora_portrait::kernels::teeth_blend::apply_alignment_final_blend(
            AlignmentFinalBlendJob {
                original: Some(&Image::new(width, height, source_pixels.clone()).unwrap()),
                effect_rgb: Some(&effect),
                teeth_mask: Some(&[128; 12]),
                edge_plane_a: Some(&edge),
                edge_plane_b: Some(&edge),
            },
        )
        .unwrap();
        let AlignmentFinalBlendOutcome::Applied(expected) = expected else {
            panic!("open edge gate produced a CPU-gated oracle");
        };
        assert_eq!(actual, expected.pixels);
        assert_eq!(run.backend, PortraitBackend::Cpu);
        assert!(run.executed);
        assert!(run.dispatches.is_empty());
        eprintln!(
            "PF30026 forced fallback backend={:?} executed={} dispatches={} fallback={:?}",
            run.backend,
            run.executed,
            run.dispatches.len(),
            run.fallback_reason
        );
        assert!(run
            .fallback_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("test-forced CUDA failure")));
        assert!(run.refusal.is_none());
        assert!(run.skip_reason.is_none());
    }

    #[test]
    fn makeup_routes_refuse_missing_same_job_textures() {
        let masks = PortraitMasks::solid(1.0);
        let material = Image::solid(3, 2, [40, 80, 120, 200]);
        let mask = Image::solid(4, 3, [10, 20, 30, 255]);

        for pf in [100u32, 103, 105, 117, 118, 30007] {
            let inputs = PortraitJobInputs::from_renderer_job(1, &masks);
            let (_, effect) = apply_one_with_inputs(pf, 8, 6, &inputs);
            assert_eq!(
                effect.refusal,
                Some(Refusal::MissingInput("makeup mask")),
                "PF{pf}"
            );

            let no_uv = [MakeupInput {
                pf,
                mix_color: [30, 60, 90, 180],
                opacity_percent: 75.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                material_uv_writes: None,
                mask: Some(&mask),
                mask_uv: None,
                mask_uv_writes: None,
            }];
            let inputs = PortraitJobInputs {
                makeup: Some(&no_uv),
                ..PortraitJobInputs::from_renderer_job(1, &masks)
            };
            let (_, effect) = apply_one_with_inputs(pf, 8, 6, &inputs);
            assert_eq!(
                effect.refusal,
                Some(Refusal::MissingInput("makeup mask UV")),
                "PF{pf}"
            );

            let mask_uv = vec![[0.5, 0.5]; 8 * 6];
            let no_writes = [MakeupInput {
                pf,
                mix_color: [30, 60, 90, 180],
                opacity_percent: 75.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                material_uv_writes: None,
                mask: Some(&mask),
                mask_uv: Some(&mask_uv),
                mask_uv_writes: None,
            }];
            let inputs = PortraitJobInputs {
                makeup: Some(&no_writes),
                ..PortraitJobInputs::from_renderer_job(1, &masks)
            };
            let (_, effect) = apply_one_with_inputs(pf, 8, 6, &inputs);
            assert_eq!(
                effect.refusal,
                Some(Refusal::MissingInput("makeup mask UV")),
                "PF{pf}"
            );
        }

        let absent = PortraitJobInputs::from_renderer_job(1, &masks);
        let (_, effect) = apply_one_with_inputs(111, 8, 6, &absent);
        assert_eq!(
            effect.refusal,
            Some(Refusal::MissingInput("makeup material"))
        );

        let no_uv = [MakeupInput {
            pf: 111,
            mix_color: [30, 60, 90, 180],
            opacity_percent: 75.0,
            filter_type: 40,
            material: Some(&material),
            material_uv: None,
            material_uv_writes: None,
            mask: None,
            mask_uv: None,
            mask_uv_writes: None,
        }];
        let inputs = PortraitJobInputs {
            makeup: Some(&no_uv),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let (_, effect) = apply_one_with_inputs(111, 8, 6, &inputs);
        assert_eq!(
            effect.refusal,
            Some(Refusal::MissingInput("makeup material UV"))
        );

        let material_uv = vec![[0.0, 0.0]; 8 * 6];
        let no_writes = [MakeupInput {
            pf: 111,
            mix_color: [30, 60, 90, 180],
            opacity_percent: 75.0,
            filter_type: 40,
            material: Some(&material),
            material_uv: Some(&material_uv),
            material_uv_writes: None,
            mask: None,
            mask_uv: None,
            mask_uv_writes: None,
        }];
        let inputs = PortraitJobInputs {
            makeup: Some(&no_writes),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let (_, effect) = apply_one_with_inputs(111, 8, 6, &inputs);
        assert_eq!(
            effect.refusal,
            Some(Refusal::MissingInput("makeup material UV writes"))
        );
    }

    fn run_makeup_route_case(force_cpu: bool) -> Vec<u8> {
        let (width, height) = (23u32, 17u32);
        let pixels = (width * height) as usize;
        let source = Image::new(width, height, frame(width, height)).unwrap();
        let masks = PortraitMasks::solid(1.0);
        let mask = Image::new(
            8,
            6,
            (0..8 * 6 * 4)
                .map(|index| ((index * 37 + 11) % 256) as u8)
                .collect(),
        )
        .unwrap();
        let material = Image::new(
            9,
            7,
            (0..9 * 7 * 4)
                .map(|index| ((index * 43 + 7) % 256) as u8)
                .collect(),
        )
        .unwrap();
        let mask_uv: Vec<[f32; 2]> = (0..pixels)
            .map(|index| {
                [
                    (index * 17 % 97) as f32 / 96.0,
                    (index * 29 % 89) as f32 / 88.0,
                ]
            })
            .collect();
        let material_uv: Vec<[f32; 2]> = (0..pixels)
            .map(|index| {
                [
                    (index * 13 % 83) as f32 / 82.0,
                    (index * 31 % 79) as f32 / 78.0,
                ]
            })
            .collect();
        let material_uv_writes = vec![1u16; pixels];
        let mask_uv_writes = vec![1u16; pixels];

        let rows = [
            MakeupInput {
                pf: 30007,
                mix_color: [31, 61, 91, 211],
                opacity_percent: 68.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                material_uv_writes: None,
                mask: Some(&mask),
                mask_uv: Some(&mask_uv),
                mask_uv_writes: Some(&mask_uv_writes),
            },
            MakeupInput {
                pf: 111,
                mix_color: [47, 97, 151, 203],
                opacity_percent: 73.0,
                filter_type: 40,
                material: Some(&material),
                material_uv: Some(&material_uv),
                material_uv_writes: Some(&material_uv_writes),
                mask: None,
                mask_uv: None,
                mask_uv_writes: None,
            },
            MakeupInput {
                pf: 100,
                mix_color: [13, 53, 113, 197],
                opacity_percent: 71.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                material_uv_writes: None,
                mask: Some(&mask),
                mask_uv: Some(&mask_uv),
                mask_uv_writes: Some(&mask_uv_writes),
            },
            MakeupInput {
                pf: 118,
                mix_color: [19, 59, 109, 199],
                opacity_percent: 77.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                material_uv_writes: None,
                mask: Some(&mask),
                mask_uv: Some(&mask_uv),
                mask_uv_writes: Some(&mask_uv_writes),
            },
            MakeupInput {
                pf: 103,
                mix_color: [23, 67, 101, 193],
                opacity_percent: 79.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                material_uv_writes: None,
                mask: Some(&mask),
                mask_uv: Some(&mask_uv),
                mask_uv_writes: Some(&mask_uv_writes),
            },
            MakeupInput {
                pf: 117,
                mix_color: [29, 71, 107, 191],
                opacity_percent: 81.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                material_uv_writes: None,
                mask: Some(&mask),
                mask_uv: Some(&mask_uv),
                mask_uv_writes: Some(&mask_uv_writes),
            },
            MakeupInput {
                pf: 105,
                mix_color: [37, 73, 109, 189],
                opacity_percent: 83.0,
                filter_type: 11,
                material: None,
                material_uv: None,
                material_uv_writes: None,
                mask: Some(&mask),
                mask_uv: Some(&mask_uv),
                mask_uv_writes: Some(&mask_uv_writes),
            },
        ];
        let inputs = PortraitJobInputs {
            makeup: Some(&rows),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let degrees = [
            (100u32, 0.61f32),
            (103, 0.62),
            (105, 0.63),
            (111, 0.64),
            (117, 0.65),
            (118, 0.66),
            (30007, 0.67),
        ];
        let extra: BTreeMap<String, f32> = degrees
            .iter()
            .map(|(pf, degree)| (format!("pc:{pf}"), *degree))
            .collect();
        let mut actual = source.pixels.clone();

        #[cfg(windows)]
        let _force = force_cpu.then(ForceCudaFailureGuard::new);

        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &extra,
            &inputs,
        );
        let expected_pfs = [100u32, 103, 105, 111, 117, 118, 30007];
        assert_eq!(
            report
                .effects
                .iter()
                .map(|effect| effect.pf)
                .collect::<Vec<_>>(),
            expected_pfs
        );

        let mut expected = source.clone();
        for (pf, degree) in degrees {
            let route = MakeupRoute::resolve(pf).unwrap();
            let input = rows.iter().copied().find(|row| row.pf == pf).unwrap();
            let layer = makeup_layer(route, input, degree).unwrap();
            expected = match route {
                MakeupRoute::Color { .. } => color_makeup(
                    &expected,
                    input.mask.unwrap(),
                    input.mask_uv.unwrap(),
                    layer,
                )
                .unwrap(),
                MakeupRoute::ImageMasked { .. } => image_makeup(
                    &expected,
                    input.material.unwrap(),
                    input.mask.unwrap(),
                    input.mask_uv.unwrap(),
                    input.material_uv.unwrap(),
                    layer,
                )
                .unwrap(),
                MakeupRoute::ImageUnmasked { .. } => image_makeup_unmasked(
                    &expected,
                    input.material.unwrap(),
                    input.material_uv.unwrap(),
                    layer,
                )
                .unwrap(),
            };
        }
        assert_eq!(actual, expected.pixels, "stage CPU/GPU route parity");

        for effect in &report.effects {
            eprintln!(
                "makeup PF{} backend={:?} kernels={:?} fallback={:?}",
                effect.pf,
                effect.backend,
                effect
                    .dispatches
                    .iter()
                    .map(|dispatch| dispatch.kernel)
                    .collect::<Vec<_>>(),
                effect.fallback_reason
            );
            assert!(effect.executed, "PF{}", effect.pf);
            assert!(effect.refusal.is_none(), "PF{}: {:?}", effect.pf, effect.refusal);
            if force_cpu {
                assert_eq!(effect.backend, PortraitBackend::Cpu, "PF{}", effect.pf);
                assert!(effect.dispatches.is_empty(), "PF{}", effect.pf);
            } else if effect.backend == PortraitBackend::Cpu {
                assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "CUDA required but PF{} fell back: {:?}",
                    effect.pf,
                    effect.fallback_reason
                );
            } else {
                assert!(
                    !effect.dispatches.is_empty(),
                    "PF{} must report its dispatch",
                    effect.pf
                );
            }
        }
        actual
    }

    #[test]
    fn makeup_routes_match_cpu_in_pf_order() {
        run_makeup_route_case(true);
    }

    #[cfg(windows)]
    #[test]
    fn makeup_routes_report_real_backend() {
        run_makeup_route_case(false);
    }

    #[cfg(windows)]
    #[test]
    fn wl51_full_owned_b3_bundle_matches_cpu_cuda_and_fallback() {
        let (width, height) = (32u32, 24u32);
        let source = frame(width, height);
        let masks = PortraitMasks::solid(1.0);
        let producer = ProducerInputs {
            makeup_bundle: Some(
                crate::portrait_makeup_bundle::test_support::complete_owned_bundle_for_job(7),
            ),
            ..ProducerInputs::default()
        };
        let mut inputs = PortraitJobInputs::from_renderer_job(7, &masks);
        inputs.fill_from_producer_inputs(&producer);
        let extra = crate::portrait_makeup_bundle::B3_MAKEUP_ROWS
            .into_iter()
            .map(|pf| (format!("pc:{pf}"), 0.73))
            .collect::<BTreeMap<_, _>>();

        let mut gpu = source.clone();
        let gpu_report = apply_portrait_effects_with_report_and_inputs(
            &mut gpu,
            width,
            height,
            NormRect::FULL,
            &extra,
            &inputs,
        );
        assert_eq!(
            gpu_report
                .effects
                .iter()
                .map(|effect| effect.pf)
                .collect::<Vec<_>>(),
            crate::portrait_makeup_bundle::B3_MAKEUP_ROWS
        );
        for effect in &gpu_report.effects {
            assert!(effect.executed, "PF{} not executed", effect.pf);
            assert_eq!(effect.refusal, None, "PF{} refused", effect.pf);
            assert_eq!(
                effect.backend,
                PortraitBackend::Cuda,
                "PF{} CPU fallback: {:?}",
                effect.pf,
                effect.fallback_reason
            );
            assert!(!effect.dispatches.is_empty(), "PF{} no dispatch", effect.pf);
            let dispatch = &effect.dispatches[0];
            eprintln!(
                "WL51 PF{} backend={:?} kernel={} device={} cc={:?} ptx={:?}",
                effect.pf,
                effect.backend,
                dispatch.kernel,
                dispatch.device,
                dispatch.compute_cap,
                dispatch.ptx_isa
            );
        }

        let _force_cpu = ForceCudaFailureGuard::new();
        let mut cpu = source.clone();
        let cpu_report = apply_portrait_effects_with_report_and_inputs(
            &mut cpu,
            width,
            height,
            NormRect::FULL,
            &extra,
            &inputs,
        );
        for effect in &cpu_report.effects {
            assert!(effect.executed, "PF{} not executed", effect.pf);
            assert_eq!(effect.refusal, None, "PF{} refused", effect.pf);
            assert_eq!(effect.backend, PortraitBackend::Cpu, "PF{}", effect.pf);
            assert!(effect.dispatches.is_empty(), "PF{}", effect.pf);
            eprintln!("WL51 PF{} fallback_backend={:?}", effect.pf, effect.backend);
        }
        assert_eq!(gpu, cpu, "full B3 bundle CPU/CUDA byte parity");
        eprintln!(
            "WL51 CPU_CUDA_BYTES={} DIFFERING_BYTES=0",
            gpu.len()
        );
    }

    struct ColorMaskStageResult {
        before: Vec<u8>,
        after: Vec<u8>,
        writes: Vec<u16>,
        effect: PortraitEffectRun,
    }

    fn run_color_mask_producer_stage(force_cpu: bool) -> ColorMaskStageResult {
        use aurora_portrait::kernels::makeup_mask_uv::{
            Face2dMaskUvInputs, color_mask_profile, produce_face2d_mask_uv,
        };

        let (width, height) = (8u32, 6u32);
        let positions = [[0.0, 0.0], [width as f32, 0.0], [0.0, height as f32]];
        let uv = produce_face2d_mask_uv(Some(Face2dMaskUvInputs {
            ticket: Some(1),
            face_id: Some(0),
            mask_source_points: &positions,
            indices: &[0, 1, 2],
            origin: [0, 0],
            size: [width as i32, height as i32],
            width,
            height,
        }))
        .unwrap();
        let profile = color_mask_profile(100).unwrap();
        let mask = Image::solid(1, 1, [128, 0, 128, 255]);
        let rows = [MakeupInput {
            pf: profile.pf,
            mix_color: profile.mix_color,
            opacity_percent: 100.0,
            filter_type: profile.filter_type,
            material: None,
            material_uv: None,
            material_uv_writes: None,
            mask: Some(&mask),
            mask_uv: Some(uv.uv()),
            mask_uv_writes: Some(uv.writes()),
        }];
        let masks = PortraitMasks::solid(1.0);
        let inputs = PortraitJobInputs {
            makeup: Some(&rows),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut rgba = frame(width, height);
        let before = rgba.clone();
        #[cfg(windows)]
        let _force = force_cpu.then(ForceCudaFailureGuard::new);
        #[cfg(not(windows))]
        let _ = force_cpu;
        let report = apply_portrait_effects_with_job(
            &mut rgba,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:100".to_string(), 0.8f32)]),
            &inputs,
        );
        assert_eq!(report.effects.len(), 1);
        ColorMaskStageResult {
            before,
            after: rgba,
            writes: uv.writes().to_vec(),
            effect: report.effects.into_iter().next().unwrap(),
        }
    }

    fn assert_color_mask_uncovered_pixels_are_not_injected(case: &ColorMaskStageResult) {
        assert!(case.writes.iter().any(|writes| *writes == 0));
        assert!(case.writes.iter().any(|writes| *writes > 0));
        let mut covered_changed = false;
        for (pixel, writes) in case.writes.iter().enumerate() {
            let base = pixel * 4;
            if *writes == 0 {
                assert_eq!(
                    &case.after[base..base + 4],
                    &case.before[base..base + 4],
                    "uncovered pixel {pixel} sampled the [0,0] mask slot"
                );
            } else if case.after[base..base + 4] != case.before[base..base + 4] {
                covered_changed = true;
            }
        }
        assert!(covered_changed, "covered mask pixels must be affected");
    }

    #[test]
    fn color_mask_producer_stage_gate_matches_cpu() {
        let cpu = run_color_mask_producer_stage(true);
        assert_eq!(cpu.effect.backend, PortraitBackend::Cpu);
        assert!(cpu.effect.executed);
        assert!(cpu.effect.dispatches.is_empty());
        assert_color_mask_uncovered_pixels_are_not_injected(&cpu);
    }

    #[cfg(windows)]
    #[test]
    fn color_mask_producer_stage_gate_reports_real_backend() {
        let cpu = run_color_mask_producer_stage(true);
        let actual = run_color_mask_producer_stage(false);
        assert_eq!(actual.after, cpu.after, "color mask producer-stage parity");
        assert_color_mask_uncovered_pixels_are_not_injected(&actual);
        eprintln!(
            "mask PF100 backend={:?} kernels={:?} fallback={:?}",
            actual.effect.backend,
            actual
                .effect
                .dispatches
                .iter()
                .map(|dispatch| dispatch.kernel)
                .collect::<Vec<_>>(),
            actual.effect.fallback_reason
        );
        if actual.effect.backend == PortraitBackend::Cpu {
            assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but PF100 fell back: {:?}",
                actual.effect.fallback_reason
            );
        } else {
            assert!(!actual.effect.dispatches.is_empty());
        }
    }

    struct Pf111StageResult {
        before: Vec<u8>,
        after: Vec<u8>,
        writes: Vec<u16>,
        effect: PortraitEffectRun,
    }

    fn run_pf111_producer_stage(force_cpu: bool) -> Pf111StageResult {
        use aurora_portrait::kernels::pf111_uv::{
            EyePupilPosition49, EyeSide, NormalEyeEbo, NormalEyeUv49, PF111_VERTEX_COUNT,
            Pf111RasterInputs, RasterExtent, rasterize_pixels,
        };

        let (width, height) = (8u32, 8u32);
        let mut positions = vec![[0.0; 2]; PF111_VERTEX_COUNT];
        let mut normal_eye_uv = vec![[0.0; 2]; PF111_VERTEX_COUNT];
        positions[0] = [0.0, 0.0];
        positions[1] = [width as f32, 0.0];
        positions[2] = [0.0, height as f32];
        normal_eye_uv[0] = [0.5, 0.5];
        normal_eye_uv[1] = [0.5, 0.5];
        normal_eye_uv[2] = [0.5, 0.5];
        let ebo = [0u16, 1, 2];
        let uv = rasterize_pixels(Pf111RasterInputs {
            side: EyeSide::Left,
            positions: Some(EyePupilPosition49::new(&positions).unwrap()),
            normal_eye_uv: Some(NormalEyeUv49::new(&normal_eye_uv).unwrap()),
            extent: Some(RasterExtent::new(width, height).unwrap()),
            normal_eye_ebo: Some(NormalEyeEbo::new(&ebo).unwrap()),
        })
        .unwrap();
        let material = Image::solid(1, 1, [0, 0, 255, 255]);
        let rows = [MakeupInput {
            pf: 111,
            mix_color: [255, 255, 255, 255],
            opacity_percent: 100.0,
            filter_type: 40,
            material: Some(&material),
            material_uv: Some(uv.material_uv()),
            material_uv_writes: Some(uv.writes()),
            mask: None,
            mask_uv: None,
            mask_uv_writes: None,
        }];
        let masks = PortraitMasks::solid(1.0);
        let inputs = PortraitJobInputs {
            makeup: Some(&rows),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut rgba = frame(width, height);
        let before = rgba.clone();
        #[cfg(windows)]
        let _force = force_cpu.then(ForceCudaFailureGuard::new);
        #[cfg(not(windows))]
        let _ = force_cpu;
        let report = apply_portrait_effects_with_job(
            &mut rgba,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:111".to_string(), 0.8f32)]),
            &inputs,
        );
        assert_eq!(report.effects.len(), 1);
        Pf111StageResult {
            before,
            after: rgba,
            writes: uv.writes().to_vec(),
            effect: report.effects.into_iter().next().unwrap(),
        }
    }

    fn assert_pf111_uncovered_pixels_are_not_injected(case: &Pf111StageResult) {
        assert!(case.writes.iter().any(|writes| *writes == 0));
        assert!(case.writes.iter().any(|writes| *writes > 0));
        let mut covered_changed = false;
        for (pixel, writes) in case.writes.iter().enumerate() {
            let base = pixel * 4;
            if *writes == 0 {
                assert_eq!(
                    &case.after[base..base + 4],
                    &case.before[base..base + 4],
                    "uncovered pixel {pixel} sampled the [0,0] material slot"
                );
            } else if case.after[base..base + 4] != case.before[base..base + 4] {
                covered_changed = true;
            }
        }
        assert!(covered_changed, "covered PF111 pixels must be affected");
    }

    #[test]
    fn pf111_producer_stage_gate_matches_cpu() {
        let cpu = run_pf111_producer_stage(true);
        assert_eq!(cpu.effect.backend, PortraitBackend::Cpu);
        assert!(cpu.effect.executed);
        assert!(cpu.effect.dispatches.is_empty());
        assert_pf111_uncovered_pixels_are_not_injected(&cpu);
    }

    #[cfg(windows)]
    #[test]
    fn pf111_producer_stage_gate_matches_real_backend() {
        let cpu = run_pf111_producer_stage(true);
        let actual = run_pf111_producer_stage(false);
        assert_eq!(actual.after, cpu.after, "PF111 producer-stage parity");
        assert_pf111_uncovered_pixels_are_not_injected(&actual);
        eprintln!(
            "PF111 producer-stage backend={:?} kernels={:?} fallback={:?}",
            actual.effect.backend,
            actual
                .effect
                .dispatches
                .iter()
                .map(|dispatch| dispatch.kernel)
                .collect::<Vec<_>>(),
            actual.effect.fallback_reason
        );
        if actual.effect.backend == PortraitBackend::Cpu {
            assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but PF111 fell back: {:?}",
                actual.effect.fallback_reason
            );
        } else {
            assert!(!actual.effect.dispatches.is_empty());
        }
    }

    #[test]
    fn pf111_missing_same_job_producer_input_stays_typed() {
        use aurora_portrait::kernels::pf111_uv::{
            EYE_PUPIL_POSITION_MISSING, EyeSide, Pf111RasterInputs, rasterize_pixels,
        };

        assert_eq!(
            rasterize_pixels(Pf111RasterInputs {
                side: EyeSide::Left,
                positions: None,
                normal_eye_uv: None,
                extent: None,
                normal_eye_ebo: None,
            })
            .unwrap_err(),
            Refusal::MissingInput(EYE_PUPIL_POSITION_MISSING)
        );
    }

    #[test]
    fn missing_skin_plane_is_a_typed_refusal() {
        let mut masks = PortraitMasks::solid(1.0);
        masks.skin = None;
        let mut rgba = frame(8, 8);
        let original = rgba.clone();
        let extra = BTreeMap::from([("pc:90005".to_string(), 0.9f32)]);
        let inputs = PortraitJobInputs::from_renderer_job(1, &masks);
        let report =
            apply_portrait_effects_with_job(&mut rgba, 8, 8, NormRect::FULL, &extra, &inputs);
        assert_eq!(rgba, original);
        assert_eq!(
            report.effects[0].refusal.as_ref(),
            Some(&Refusal::MissingInput("skin mask"))
        );
        assert!(report.effects[0].skipped);
        assert!(!report.effects[0].executed);
    }

    fn run_complete_same_job_case(force_cpu: bool) {
        let (width, height) = (17u32, 13u32);
        let pixels = (width * height) as usize;
        let source = Image::new(width, height, frame(width, height)).unwrap();
        let masks = PortraitMasks::solid(1.0);

        let teeth_mask: Vec<u8> = (0..pixels).map(|i| ((i * 29 + 7) % 256) as u8).collect();
        let teeth_lut: Vec<u8> = (0..LUT_BYTES).map(|i| (i % 251) as u8).collect();

        let repair = Image::new(
            width,
            height,
            (0..pixels * 4)
                .map(|i| ((i * 43 + 17) % 256) as u8)
                .collect(),
        )
        .unwrap();
        let beard_a: Vec<u8> = (0..pixels).map(|i| ((i * 19 + 3) % 256) as u8).collect();
        let beard_b: Vec<u8> = (0..pixels).map(|i| ((i * 31 + 11) % 256) as u8).collect();

        let face_mask: Vec<u8> = (0..pixels * 4)
            .map(|i| ((i * 13 + 5) % 256) as u8)
            .collect();
        let blur_y: Vec<u8> = (0..pixels).map(|i| ((i * 7 + 9) % 256) as u8).collect();
        let face_skin_map: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let hardlight = [0.0, 0.25, 0.5, 0.75, 1.0];
        let face_beauty = [0.0, 0.1, 0.35, 0.8, 1.0];

        let body_skin: Vec<f32> = (0..pixels)
            .map(|i| ((i * 17 + 1) % 256) as f32 / 255.0)
            .collect();
        let body_dog: Vec<f32> = (0..pixels)
            .map(|i| ((i * 23 + 7) % 256) as f32 / 255.0)
            .collect();
        let body_beauty = [0.0, 0.2, 0.45, 0.7, 1.0];

        let eye_color: Vec<f32> = (0..pixels * 3)
            .map(|i| ((i * 37 + 19) % 256) as f32)
            .collect();
        let eye_response: Vec<f32> = (0..pixels)
            .map(|i| ((i * 11 + 3) % 256) as f32 / 255.0)
            .collect();

        let lip: Vec<u8> = (0..pixels).map(|i| ((i * 17 + 5) % 256) as u8).collect();
        let mouth: Vec<u8> = (0..pixels).map(|i| ((i * 29 + 7) % 256) as u8).collect();
        let model_dx: Vec<f32> = (0..pixels).map(|i| ((i % 9) as f32 - 4.0) * 0.25).collect();
        let model_dy: Vec<f32> = (0..pixels).map(|i| ((i % 7) as f32 - 3.0) * -0.2).collect();
        let smile_roi = Rect {
            x: 2,
            y: 1,
            width: 12,
            height: 9,
        };

        let suit_mask = Image::solid(width, height, [255, 255, 255, 255]);
        let suit_uv = vec![[0.5f32, 0.5f32]; pixels];
        let suit_layer = MakeupLayer {
            mix_color: [250, 40, 20, 255],
            opacity_percent: 100.0,
            degree01: 0.6,
            filter_type: 10,
            mask_channel: aurora_portrait::kernels::makeup_material::MaskChannel::Rgba,
        };
        let suit_passes = [SuitPassInput {
            kind: SuitPassKindInput::Color,
            mask: Some(&suit_mask),
            mask_uv: Some(&suit_uv),
            layer: Some(suit_layer),
            shelter: 1.0,
            region: 1.0,
        }];

        let inputs = PortraitJobInputs {
            origin: PortraitJobInputOrigin::RendererJob { job_id: 9 },
            receipt: None,
            masks: &masks,
            assets: PortraitAssetInputs {
                teeth_whiten_lut: Some(&teeth_lut),
                teeth_bright_lut: Some(&teeth_lut),
            },
            parsing: None,
            teeth: Some(TeethInputs {
                job_id: 9,
                mask_width: width,
                mask_height: height,
                mask_source: TeethMaskSource::FaceParsingV2Id2G,
                mask_format: TeethMaskFormat::U8,
                mask_sha256: sha256_32(&teeth_mask),
                whiten_lut_identity: PF108_WHITEN_LUT_IDENTITY,
                bright_lut_identity: PF108_BRIGHT_LUT_IDENTITY,
                mask_g: &teeth_mask,
                bright_ui: 60.0,
                deyellow_ui: 70.0,
            }),
            pf30025: None,
            pf101: None,
            pf90130: None,
            teeth_alignment: None,
            radiance: None,
            beard: Some(BeardInputs {
                mask_a: &beard_a,
                mask_b: &beard_b,
                repair_rgb: &repair,
            }),
            moisture_face: Some(MoistureFaceInputs {
                width,
                height,
                mask_rgba: &face_mask,
                blur_alpha: &blur_y,
                channel: aurora_portrait::kernels::moisture::MaskChannel::R,
                skin_map_width: 4,
                skin_map_height: 4,
                skin_map: &face_skin_map,
                hardlight: &hardlight,
                beauty: &face_beauty,
                pivot: 0.5,
                gain: 1.0,
            }),
            moisture_body: Some(MoistureBodyInputs {
                width,
                height,
                weight_skin: &body_skin,
                weight_dog: &body_dog,
                beauty: &body_beauty,
                degree: 0.5,
                roi: (0, 0, width, height),
            }),
            eye_bag: Some(EyeBagInputs {
                ticket: 9,
                model_id: aurora_portrait::eye_bag_alignment::MODEL_ID,
                color_units: aurora_portrait::eye_bag_alignment::EyeBagColorUnits::SourceBytes,
                geometry_class: aurora_portrait::eye_bag_alignment::EyeBagGeometryClass::Candidate,
                geometry_receipt: None,
                width,
                height,
                aligned_color_field_rgb_f32: &eye_color,
                aligned_response_mask: &eye_response,
            }),
            smile: SmileInputs {
                material: Some("Grinning"),
                lip_mask: Some(&lip),
                mouth_mask: Some(&mouth),
                model_dx: Some(&model_dx),
                model_dy: Some(&model_dy),
                roi: Some(smile_roi),
            },
            suit_passes: Some(&suit_passes),
            hair_smooth: None,
            hair_gloss: None,
            stray_hair: StrayHairInputs::default(),
            makeup: None,
            makeup_bundle: None,
            hlf_face: None,
            hlf_body: None,
            wrinkle: None,
            wrinkle_producer: None,
            pf90021: None,
            slim: None,
            face_geometry: None,
            body_geometry: None,
            repairs: None,
            historical_tone: None,
            historical_skin: None,
            plump: None,
        };

        #[cfg(windows)]
        let _force = force_cpu.then(ForceCudaFailureGuard::new);

        let expected = [
            (
                108u32,
                aurora_portrait::kernels::face_detail::teeth_whitening(
                    &source,
                    &teeth_mask,
                    &teeth_lut,
                    &teeth_lut,
                    80.0,
                    60.0,
                    70.0,
                )
                .unwrap()
                .output,
            ),
            (
                21026,
                aurora_portrait::kernels::beard::apply(&source, &repair, &beard_a, &beard_b, 0.8)
                    .unwrap(),
            ),
            (
                90002,
                aurora_portrait::kernels::moisture::face_apply(
                    &source,
                    &face_mask,
                    &blur_y,
                    &FaceParams {
                        ui: 80.0,
                        pivot: 0.5,
                        gain: 1.0,
                    },
                    aurora_portrait::kernels::moisture::MaskChannel::R,
                    &SkinMap {
                        width: 4,
                        height: 4,
                        samples: &face_skin_map,
                    },
                    &Lut1D {
                        samples: &hardlight,
                    },
                    &Lut1D {
                        samples: &face_beauty,
                    },
                )
                .unwrap(),
            ),
            (
                90010,
                aurora_portrait::kernels::moisture::body_apply(
                    &source,
                    &body_skin,
                    &body_dog,
                    &Lut1D {
                        samples: &body_beauty,
                    },
                    0.5,
                    (0, 0, width, height),
                )
                .unwrap(),
            ),
            (
                90116,
                aurora_portrait::kernels::eye_bag::compose_field_image_pf90116(
                    &source,
                    &eye_color,
                    &eye_response,
                    0.8,
                )
                .unwrap(),
            ),
            (
                100007,
                Image::new(
                    width,
                    height,
                    smile_warp::pf100007_apply(
                        &source.pixels,
                        width,
                        height,
                        0.8,
                        &lip,
                        &mouth,
                        &model_dx,
                        &model_dy,
                        smile_roi,
                    )
                    .unwrap(),
                )
                .unwrap(),
            ),
            (
                90033,
                aurora_portrait::kernels::suit_makeup::compose(
                    &source,
                    &[SuitPass {
                        kind: SuitPassKind::Color,
                        mask: &suit_mask,
                        mask_uv: &suit_uv,
                        layer: suit_layer,
                        shelter: 1.0,
                        region: 1.0,
                    }],
                )
                .unwrap(),
            ),
        ];

        for (pf, expected) in expected {
            let (actual, effect) = apply_one_with_inputs(pf, width, height, &inputs);
            assert_eq!(actual, expected.pixels, "PF{pf} CPU fallback pixels");
            assert!(effect.executed, "PF{pf}");
            assert!(effect.refusal.is_none(), "PF{pf}: {:?}", effect.refusal);
            eprintln!(
                "PF{pf} backend={:?} device={:?} kernels={:?} fallback={:?}",
                effect.backend,
                effect
                    .dispatches
                    .first()
                    .map(|dispatch| dispatch.device.as_str()),
                effect
                    .dispatches
                    .iter()
                    .map(|dispatch| dispatch.kernel)
                    .collect::<Vec<_>>(),
                effect.fallback_reason
            );
            if force_cpu {
                assert_eq!(effect.backend, PortraitBackend::Cpu, "PF{pf}");
                assert!(effect.dispatches.is_empty(), "PF{pf}");
                #[cfg(windows)]
                assert!(
                    effect
                        .fallback_reason
                        .as_deref()
                        .is_some_and(|reason| reason.contains("test-forced CUDA failure")),
                    "PF{pf}: {:?}",
                    effect.fallback_reason
                );
            } else {
                match effect.backend {
                    PortraitBackend::Cuda => {
                        assert!(!effect.dispatches.is_empty(), "PF{pf}");
                    }
                    PortraitBackend::Cpu => {
                        assert!(
                            std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                            "CUDA required but PF{pf} fell back: {:?}",
                            effect.fallback_reason
                        );
                        #[cfg(windows)]
                        assert!(
                            effect.fallback_reason.is_some(),
                            "PF{pf} CPU fallback must state why"
                        );
                    }
                    PortraitBackend::Skipped => {
                        panic!("PF{pf} unexpectedly skipped: {:?}", effect.skip_reason)
                    }
                }
            }
        }
    }

    #[test]
    fn complete_same_job_bundle_matches_cpu_oracles() {
        run_complete_same_job_case(true);
    }

    #[cfg(windows)]
    #[test]
    fn complete_same_job_bundle_reports_real_backend() {
        run_complete_same_job_case(false);
    }

    #[test]
    fn skin_lut_effects_change_a_warm_tone() {
        for (pf, degree) in [(21015u32, 0.9f32), (30096, 0.1), (90005, 0.9), (90007, 0.9)] {
            let mut rgba = frame(8, 8);
            let original = rgba.clone();
            apply(&mut rgba, 8, 8, &[(&format!("pc:{pf}"), degree)]);
            assert_ne!(rgba, original, "PF{pf} must change the frame");
        }
    }

    #[test]
    fn skin_texture_and_gloss_change_the_frame() {
        for (pf, degree) in [(8068u32, 0.1f32), (8069, 0.0)] {
            let mut rgba = frame(64, 64);
            let original = rgba.clone();
            apply(&mut rgba, 64, 64, &[(&format!("pc:{pf}"), degree)]);
            assert_ne!(rgba, original, "PF{pf} must change the frame");
        }
    }

    #[test]
    fn skin_texture_neutral_is_identity() {
        let mut rgba = frame(64, 64);
        let original = rgba.clone();
        apply(&mut rgba, 64, 64, &[("pc:8069", 0.5)]);
        assert_eq!(rgba, original);
    }

    #[test]
    fn skin_clarity_runs_with_assets() {
        let mut rgba = frame(8, 8);
        apply(&mut rgba, 8, 8, &[("pc:8070", 0.5)]);
        assert_eq!(rgba.len(), 8 * 8 * 4);
    }

    #[test]
    fn pf90004_requires_an_owned_same_job_bundle() {
        let masks = PortraitMasks::solid(1.0);
        let absent = PortraitJobInputs::from_renderer_job(1, &masks);
        let (_, run) = apply_one_with_inputs(90004, 8, 8, &absent);
        assert_eq!(
            run.refusal,
            Some(Refusal::MissingInput("PF90004_PARSING_RGBA"))
        );

        let skin = vec![255u8; 64];
        let rgba = vec![0u8; 64 * 4];
        let complete = RadianceInputs {
            frame_width: Some(8),
            frame_height: Some(8),
            skin_mask: Some(&skin),
            parsing_rgba: Some(&rgba),
            face_rgba: Some(&rgba),
            value_mean: Some(128.0),
            roi: Some(NormRect::FULL),
        };
        for (radiance, missing) in [
            (
                RadianceInputs {
                    frame_width: None,
                    ..complete
                },
                "PF90004_FRAME_WIDTH",
            ),
            (
                RadianceInputs {
                    frame_height: None,
                    ..complete
                },
                "PF90004_FRAME_HEIGHT",
            ),
            (
                RadianceInputs {
                    skin_mask: None,
                    ..complete
                },
                "PF90004_SKIN_MASK",
            ),
            (
                RadianceInputs {
                    parsing_rgba: None,
                    ..complete
                },
                "PF90004_PARSING_RGBA",
            ),
            (
                RadianceInputs {
                    face_rgba: None,
                    ..complete
                },
                "PF90004_FACE_RGBA",
            ),
            (
                RadianceInputs {
                    value_mean: None,
                    ..complete
                },
                "PF90004_VALUE_MEAN",
            ),
            (
                RadianceInputs { roi: None, ..complete },
                "PF90004_ROI",
            ),
        ] {
            let inputs = PortraitJobInputs {
                radiance: Some(radiance),
                ..PortraitJobInputs::from_renderer_job(1, &masks)
            };
            let (_, run) = apply_one_with_inputs(90004, 8, 8, &inputs);
            assert_eq!(run.refusal, Some(Refusal::MissingInput(missing)));
        }

        let inputs = PortraitJobInputs {
            radiance: Some(RadianceInputs {
                roi: Some(NormRect {
                    x: 0.0,
                    y: 0.0,
                    width: 0.5,
                    height: 1.0,
                }),
                ..complete
            }),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let (_, run) = apply_one_with_inputs(90004, 8, 8, &inputs);
        assert_eq!(
            run.refusal,
            Some(Refusal::ValueOutOfRange("PF90004_ROI"))
        );

        let inputs = PortraitJobInputs {
            radiance: Some(complete),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let (_, run) = apply_one_with_inputs(90004, 8, 8, &inputs);
        assert!(run.refusal.is_none());
        assert!(run.executed);
    }

    #[test]
    fn pf90004_producer_writer_is_atomic_and_frame_exact() {
        let bundle = || {
            Pf90004ProducerInputs::from_same_job(
                8,
                8,
                vec![0u8; 64],
                vec![0u8; 64 * 4],
                vec![0u8; 64 * 4],
                82.42559051513672,
                NormRect::FULL,
            )
            .expect("valid PF90004 bundle")
        };

        for (result, expected) in [
            (
                Pf90004ProducerInputs::from_same_job(
                    0,
                    8,
                    vec![0; 64],
                    vec![0; 64 * 4],
                    vec![0; 64 * 4],
                    128.0,
                    NormRect::FULL,
                ),
                Refusal::ShapeMismatch("PF90004_FRAME_SIZE"),
            ),
            (
                Pf90004ProducerInputs::from_same_job(
                    8,
                    8,
                    vec![0; 63],
                    vec![0; 64 * 4],
                    vec![0; 64 * 4],
                    128.0,
                    NormRect::FULL,
                ),
                Refusal::ShapeMismatch("PF90004_SKIN_MASK"),
            ),
            (
                Pf90004ProducerInputs::from_same_job(
                    8,
                    8,
                    vec![0; 64],
                    vec![0; 64 * 4 - 1],
                    vec![0; 64 * 4],
                    128.0,
                    NormRect::FULL,
                ),
                Refusal::ShapeMismatch("PF90004_PARSING_RGBA"),
            ),
            (
                Pf90004ProducerInputs::from_same_job(
                    8,
                    8,
                    vec![0; 64],
                    vec![0; 64 * 4],
                    vec![0; 64 * 4 - 1],
                    128.0,
                    NormRect::FULL,
                ),
                Refusal::ShapeMismatch("PF90004_FACE_RGBA"),
            ),
            (
                Pf90004ProducerInputs::from_same_job(
                    8,
                    8,
                    vec![0; 64],
                    vec![0; 64 * 4],
                    vec![0; 64 * 4],
                    f32::NAN,
                    NormRect::FULL,
                ),
                Refusal::ValueOutOfRange("PF90004_VALUE_MEAN"),
            ),
            (
                Pf90004ProducerInputs::from_same_job(
                    8,
                    8,
                    vec![0; 64],
                    vec![0; 64 * 4],
                    vec![0; 64 * 4],
                    128.0,
                    NormRect {
                        x: 0.0,
                        y: 0.0,
                        width: 0.5,
                        height: 1.0,
                    },
                ),
                Refusal::ValueOutOfRange("PF90004_ROI"),
            ),
        ] {
            assert_eq!(result.expect_err("invalid bundle"), expected);
        }

        let mut producer = ProducerInputs::default();
        let empty_fingerprint = producer.fingerprint();
        producer.write_pf90004(bundle());
        assert_ne!(producer.fingerprint(), empty_fingerprint);

        let masks = PortraitMasks::solid(1.0);
        let mut inputs = PortraitJobInputs::from_renderer_job(7, &masks);
        inputs.fill_from_producer_inputs(&producer);
        let radiance = inputs.radiance.expect("PF90004 bundle");
        assert_eq!(
            (radiance.frame_width, radiance.frame_height),
            (Some(8), Some(8))
        );
        assert_eq!(radiance.skin_mask.map(<[u8]>::len), Some(64));
        assert_eq!(radiance.parsing_rgba.map(<[u8]>::len), Some(64 * 4));
        assert_eq!(radiance.face_rgba.map(<[u8]>::len), Some(64 * 4));
        assert_eq!(radiance.value_mean, Some(82.42559051513672));
        assert_eq!(radiance.roi, Some(NormRect::FULL));

        let (_, run) = apply_one_with_inputs(90004, 8, 8, &inputs);
        assert!(run.refusal.is_none());
        assert!(run.executed);

        let shape_bundle = Pf90004ProducerInputs::from_same_job(
            4,
            16,
            vec![0; 64],
            vec![0; 64 * 4],
            vec![0; 64 * 4],
            128.0,
            NormRect::FULL,
        )
        .expect("same byte count, different frame shape");
        let mut wrong_shape = ProducerInputs::default();
        wrong_shape.write_pf90004(shape_bundle);
        let mut wrong_inputs = PortraitJobInputs::from_renderer_job(8, &masks);
        wrong_inputs.fill_from_producer_inputs(&wrong_shape);
        let (_, run) = apply_one_with_inputs(90004, 8, 8, &wrong_inputs);
        assert_eq!(
            run.refusal,
            Some(Refusal::ShapeMismatch("PF90004_FRAME_SIZE"))
        );
    }

    #[test]
    fn pf90004_parser_producer_keeps_unproven_planes_absent() {
        let masks = PortraitMasks {
            skin: Some(vec![1.0, 0.0, 0.5, 0.0]),
            hair: None,
            face: Some(vec![1.0, 1.0, 1.0, 1.0]),
            body: Some(vec![0.0; 4]),
            width: 2,
            height: 2,
        };
        let mut rgba = vec![0u8; 16];
        for (pixel, red) in [100u8, 200, 50, 20].into_iter().enumerate() {
            rgba[pixel * 4] = red;
            rgba[pixel * 4 + 3] = 255;
        }
        let auto = masks
            .pf90004_parser_inputs(&rgba, 2, 2, NormRect::FULL)
            .expect("parser fields");
        assert_eq!(auto.source, Pf90004AutoSource::PortraitParser);
        assert_eq!(auto.skin_mask, vec![255, 0, 128, 0]);
        assert_eq!(auto.parsing_rgba, None);
        assert_eq!(auto.face_rgba, None);
        assert_eq!(auto.value_mean, 75.0);
        assert_eq!(auto.roi, NormRect::FULL);
    }

    #[test]
    fn pf90004_value_mean_uses_native_even_statistics_size() {
        let masks = PortraitMasks {
            skin: Some(vec![1.0; 15]),
            hair: None,
            face: None,
            body: None,
            width: 5,
            height: 3,
        };
        let mut rgba = vec![0u8; 15 * 4];
        for pixel in 0..15 {
            rgba[pixel * 4] = 101;
            rgba[pixel * 4 + 1] = 202;
            rgba[pixel * 4 + 2] = 17;
            rgba[pixel * 4 + 3] = 255;
        }
        let auto = masks
            .pf90004_parser_inputs(&rgba, 5, 3, NormRect::FULL)
            .expect("parser fields");
        assert_eq!(auto.value_mean, 101.0);
    }

    #[cfg(windows)]
    #[test]
    fn pf90004_cuda_route_matches_cpu_oracle_and_reports_dispatch() {
        let (width, height) = (35u32, 27u32);
        let pixels = (width * height) as usize;
        let source = Image::new(width, height, frame(width, height)).unwrap();
        let masks = PortraitMasks::solid(1.0);
        let skin: Vec<u8> = (0..pixels).map(|i| (40 + (i * 19) % 216) as u8).collect();
        let parsing: Vec<u8> = (0..pixels * 4)
            .map(|i| ((i * 13 + 7) % 256) as u8)
            .collect();
        let face: Vec<u8> = (0..pixels * 4)
            .map(|i| ((i * 29 + 11) % 256) as u8)
            .collect();
        let radiance = RadianceInputs {
            frame_width: Some(width),
            frame_height: Some(height),
            skin_mask: Some(&skin),
            parsing_rgba: Some(&parsing),
            face_rgba: Some(&face),
            value_mean: Some(143.25),
            roi: Some(NormRect::FULL),
        };
        let inputs = PortraitJobInputs {
            radiance: Some(radiance),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let expected = apply_skin_radiance(&source, 0.47, Some(radiance)).unwrap();
        let mut actual = source.pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:90004".to_string(), 0.47f32)]),
            &inputs,
        );
        assert_eq!(actual, expected.pixels);
        let effect = &report.effects[0];
        assert!(effect.executed);
        assert!(effect.refusal.is_none());
        if effect.backend == PortraitBackend::Cuda {
            assert_eq!(
                effect
                    .dispatches
                    .iter()
                    .map(|dispatch| dispatch.kernel)
                    .collect::<Vec<_>>(),
                vec![aurora_portrait::cuda_radiance::KERNEL]
            );
            let dispatch = &effect.dispatches[0];
            assert_eq!(dispatch.block, (16, 16));
            assert!(!dispatch.device.is_empty());
            eprintln!(
                "PF90004 stage device={} kernel={} cc={:?} driver={} ptx_isa={:?}",
                dispatch.device,
                dispatch.kernel,
                dispatch.compute_cap,
                dispatch.driver_version,
                dispatch.ptx_isa
            );
        } else {
            assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but PF90004 fell back: {:?}",
                effect.fallback_reason
            );
        }
    }

    #[cfg(windows)]
    #[test]
    fn pf90004_forced_cuda_failure_uses_cpu_oracle() {
        let (width, height) = (8u32, 8u32);
        let pixels = (width * height) as usize;
        let source = Image::new(width, height, frame(width, height)).unwrap();
        let masks = PortraitMasks::solid(1.0);
        let skin = vec![255u8; pixels];
        let parsing = vec![0u8; pixels * 4];
        let face = vec![0u8; pixels * 4];
        let radiance = RadianceInputs {
            frame_width: Some(width),
            frame_height: Some(height),
            skin_mask: Some(&skin),
            parsing_rgba: Some(&parsing),
            face_rgba: Some(&face),
            value_mean: Some(128.0),
            roi: Some(NormRect::FULL),
        };
        let inputs = PortraitJobInputs {
            radiance: Some(radiance),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let expected = apply_skin_radiance(&source, 0.5, Some(radiance)).unwrap();
        let mut actual = source.pixels.clone();
        let _force = ForceCudaFailureGuard::new();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &BTreeMap::from([("pc:90004".to_string(), 0.5f32)]),
            &inputs,
        );
        assert_eq!(actual, expected.pixels);
        let effect = &report.effects[0];
        assert_eq!(effect.backend, PortraitBackend::Cpu);
        assert!(effect.dispatches.is_empty());
        assert!(effect.fallback_reason.is_some());
    }

    #[test]
    fn portrait_effects_process_in_pf_order() {
        let mut rgba = frame(16, 16);
        apply(&mut rgba, 16, 16, &[("pc:90007", 0.9), ("pc:21015", 0.9)]);
        assert_eq!(rgba.len(), 16 * 16 * 4);
    }

    #[cfg(windows)]
    #[test]
    fn cuda_ported_stages_match_cpu_pixels_when_available() {
        let source = Image::new(8, 8, frame(8, 8)).expect("frame");
        let masks = PortraitMasks::solid(1.0);
        let skin = masks.mask_u8(MaskKind::Skin, 8, 8, NormRect::FULL);
        for (pf, degree) in [
            (8068u32, 0.1f32),
            (30096, 0.1),
            (30096, 0.9),
            (90005, 0.9),
            (90007, 0.9),
        ] {
            let cpu = match pf {
                8068 => apply_skin_gloss(&source, degree, skin.as_deref()),
                30096 => apply_skin_temperature(&source, degree, skin.as_deref()),
                90005 => apply_skin_whiten(&source, degree, skin.as_deref()),
                90007 => apply_skin_ruddy(&source, degree, skin.as_deref()),
                _ => unreachable!(),
            }
            .expect("cpu");
            match try_cuda_effect(&source, pf, degree, skin.as_deref()) {
                Some(Ok((gpu, dispatches))) => {
                    assert_eq!(gpu.pixels, cpu.pixels, "CUDA/CPU PF{pf} pixels");
                    assert!(!dispatches.is_empty(), "CUDA must report a dispatch");
                    for dispatch in &dispatches {
                        eprintln!(
                            "PF{pf} kernel={} device={} grid={:?} block={:?} ptx_isa={:?}",
                            dispatch.kernel,
                            dispatch.device,
                            dispatch.grid,
                            dispatch.block,
                            dispatch.ptx_isa
                        );
                    }
                }
                Some(Err(err)) => {
                    assert!(
                        std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                        "CUDA required but unavailable: {err:?}"
                    );
                    eprintln!("CUDA unavailable; fallback contract covered elsewhere: {err:?}");
                }
                None => panic!("expected PF{pf} CUDA route"),
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn pf30096_forced_cuda_failure_falls_back_to_cpu() {
        let (width, height) = (8u32, 8u32);
        let source = Image::new(width, height, frame(width, height)).expect("frame");
        let masks = PortraitMasks::solid(1.0);
        let skin = masks.mask_u8(MaskKind::Skin, width, height, NormRect::FULL);
        let expected = apply_skin_temperature(&source, 0.9, skin.as_deref()).expect("cpu");
        let params = BTreeMap::from([("pc:30096".to_string(), 0.9f32)]);
        let inputs = PortraitJobInputs::from_renderer_job(1, &masks);
        let mut actual = source.pixels.clone();
        let _force = ForceCudaFailureGuard::new();
        let report = apply_portrait_effects_with_report_and_inputs(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &params,
            &inputs,
        );
        assert_eq!(actual, expected.pixels);
        let effect = &report.effects[0];
        assert_eq!(effect.backend, PortraitBackend::Cpu);
        assert!(effect.dispatches.is_empty());
        assert!(effect.executed);
        assert!(
            effect
                .fallback_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("test-forced CUDA failure"))
        );
    }

    #[cfg(windows)]
    #[test]
    fn pf30096_report_marks_the_real_backend() {
        let (width, height) = (8u32, 8u32);
        let source = Image::new(width, height, frame(width, height)).expect("frame");
        let masks = PortraitMasks::solid(1.0);
        let skin = masks.mask_u8(MaskKind::Skin, width, height, NormRect::FULL);
        let expected = apply_skin_temperature(&source, 0.9, skin.as_deref()).expect("cpu");
        let params = BTreeMap::from([("pc:30096".to_string(), 0.9f32)]);
        let inputs = PortraitJobInputs::from_renderer_job(1, &masks);
        let mut actual = source.pixels.clone();
        let report = apply_portrait_effects_with_report_and_inputs(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &params,
            &inputs,
        );
        assert_eq!(actual, expected.pixels);
        let effect = &report.effects[0];
        assert!(effect.executed);
        assert!(effect.refusal.is_none());
        match effect.backend {
            PortraitBackend::Cuda => {
                assert_eq!(effect.dispatches.len(), 1);
                assert_eq!(
                    effect.dispatches[0].kernel,
                    aurora_portrait::cuda_lut::TEMPERATURE_KERNEL
                );
            }
            PortraitBackend::Cpu => {
                assert!(std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none());
                assert!(effect.fallback_reason.is_some());
            }
            PortraitBackend::Skipped => panic!("PF30096 unexpectedly skipped"),
        }
    }

    #[cfg(windows)]
    #[test]
    fn cuda_skin_model_routes_match_cpu_when_available() {
        let source = Image::solid(16, 16, [127, 113, 91, 255]);
        let masks = PortraitMasks::solid(1.0);
        let skin = masks.mask_u8(MaskKind::Skin, 16, 16, NormRect::FULL);
        for (pf, degree) in [(8069u32, 0.1f32), (8070, 0.5)] {
            let cpu = match pf {
                8069 => apply_skin_texture(&source, degree, skin.as_deref()),
                8070 => apply_skin_clarity(&source, degree, skin.as_deref(), 16, 16),
                _ => unreachable!(),
            }
            .expect("cpu");
            match try_cuda_effect(&source, pf, degree, skin.as_deref()) {
                Some(Ok((gpu, dispatches))) => {
                    assert_eq!(gpu, cpu);
                    assert!(!dispatches.is_empty());
                }
                Some(Err(err)) => assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "CUDA required but unavailable: {err:?}"
                ),
                None => panic!("expected PF{pf} CUDA route"),
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn cuda_fixed_point_tint_route_matches_cpu_when_available() {
        let source = Image::solid(16, 16, [181, 139, 103, 255]);
        let masks = PortraitMasks::solid(1.0);
        let skin = masks.mask_u8(MaskKind::Skin, 16, 16, NormRect::FULL);
        for degree in [0.1f32, 0.5, 0.9] {
            let cpu = apply_skin_tint(&source, degree, skin.as_deref()).expect("cpu");
            match try_cuda_effect(&source, 21015, degree, skin.as_deref()) {
                Some(Ok((gpu, dispatches))) => {
                    assert_eq!(gpu, cpu);
                    assert_eq!(dispatches.len(), 1);
                }
                Some(Err(err)) => assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "CUDA required but unavailable: {err:?}"
                ),
                None => panic!("expected PF21015 CUDA route"),
            }
        }
    }

    #[test]
    fn run_report_does_not_label_cpu_as_cuda() {
        let mut rgba = frame(8, 8);
        let map = BTreeMap::from([("pc:90005".to_string(), 0.9)]);
        let masks = PortraitMasks::solid(1.0);
        let inputs = PortraitJobInputs::from_renderer_job(42, &masks);
        let report = apply_portrait_effects_with_report_and_inputs(
            &mut rgba,
            8,
            8,
            NormRect::FULL,
            &map,
            &inputs,
        );
        assert_eq!(report.job_id, Some(42));
        assert_eq!(report.effects.len(), 1);
        let effect = &report.effects[0];
        assert_eq!(effect.pf, 90005);
        assert!(!effect.skipped);
        assert!(effect.executed);
        match effect.backend {
            PortraitBackend::Cuda => assert!(!effect.dispatches.is_empty()),
            PortraitBackend::Cpu => assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but report fell back to CPU: {:?}",
                effect.fallback_reason
            ),
            PortraitBackend::Skipped => panic!("PF90005 unexpectedly skipped"),
        }
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "GPU timing; run with --ignored and PORTRAIT_CUDA_REQUIRED=1"]
    fn cuda_skin_lut_timing_p50_p95() {
        use std::time::Instant;

        let (width, height) = (256u32, 256u32);
        let source = Image::new(width, height, frame(width, height)).expect("frame");
        let masks = PortraitMasks::solid(1.0);
        let skin = masks.mask_u8(MaskKind::Skin, width, height, NormRect::FULL);
        let mut cpu_ms = Vec::new();
        let mut gpu_ms = Vec::new();
        for _ in 0..10 {
            let started = Instant::now();
            apply_skin_whiten(&source, 0.9, skin.as_deref()).expect("cpu");
            cpu_ms.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        for _ in 0..10 {
            let started = Instant::now();
            match try_cuda_effect(&source, 90005, 0.9, skin.as_deref()) {
                Some(Ok(_)) => gpu_ms.push(started.elapsed().as_secs_f64() * 1000.0),
                _ => {
                    assert!(
                        std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                        "CUDA timing requested but unavailable"
                    );
                    return;
                }
            }
        }
        cpu_ms.sort_by(|a, b| a.total_cmp(b));
        gpu_ms.sort_by(|a, b| a.total_cmp(b));
        eprintln!(
            "PF90005 256x256 cpu p50={:.3}ms p95={:.3}ms; cuda p50={:.3}ms p95={:.3}ms",
            cpu_ms[5], cpu_ms[9], gpu_ms[5], gpu_ms[9]
        );
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "GPU session timing; run with --ignored and PORTRAIT_CUDA_REQUIRED=1"]
    fn cuda_skin_lut_session_cold_warm_p50_p95() {
        use std::time::Instant;

        let (width, height) = (256u32, 256u32);
        let source = Image::new(width, height, frame(width, height)).expect("frame");
        let masks = PortraitMasks::solid(1.0);
        let skin = masks.mask_u8(MaskKind::Skin, width, height, NormRect::FULL);
        let run = || {
            let started = Instant::now();
            let routed = try_cuda_effect(&source, 90005, 0.9, skin.as_deref());
            (started.elapsed().as_secs_f64() * 1000.0, routed)
        };
        let (cold_ms, cold) = run();
        assert!(matches!(cold, Some(Ok(_))), "cold route must use CUDA");
        for _ in 0..10 {
            assert!(matches!(run().1, Some(Ok(_))), "warmup must stay on CUDA");
        }
        let mut warm_ms = Vec::with_capacity(30);
        for _ in 0..30 {
            let (elapsed, routed) = run();
            assert!(matches!(routed, Some(Ok(_))), "sample must stay on CUDA");
            warm_ms.push(elapsed);
        }
        warm_ms.sort_by(|a, b| a.total_cmp(b));
        eprintln!(
            "PF90005 256x256 session cold={cold_ms:.3}ms warm_p50={:.3}ms warm_p95={:.3}ms",
            warm_ms[14], warm_ms[28]
        );
    }

    #[cfg(windows)]
    fn run_face_width_mesh_stage(force_cuda_failure: bool) -> (Vec<u8>, PortraitRunReport) {
        let (source, mut slim) = portrait_slim_stage::tests::fixture();
        let masks = PortraitMasks::default();
        let extra = BTreeMap::from([
            ("pc:30092".to_string(), 100.0),
            ("pc:30093".to_string(), -100.0),
        ]);
        let receipt = PortraitJobReceipt::from_renderer_receipt(
            &source.pixels,
            source.width,
            source.height,
            NormRect::FULL,
            &extra,
            &masks,
        );
        slim.job_id = receipt.job_id();
        let producer = ProducerInputs {
            slim: Some(slim),
            ..Default::default()
        };
        let mut inputs = PortraitJobInputs::from_renderer_receipt(receipt, &masks);
        inputs.fill_from_producer_inputs(&producer);
        let _force = force_cuda_failure.then(ForceCudaFailureGuard::new);
        let mut frame = source.pixels;
        let report = apply_portrait_effects_with_report_and_inputs_policy(
            &mut frame,
            source.width,
            source.height,
            NormRect::FULL,
            &extra,
            &inputs,
            true,
        );
        (frame, report)
    }

    #[cfg(windows)]
    #[test]
    fn face_width_mesh_auto_records_backend_and_matches_cpu() {
        let (cpu_frame, cpu_report) = run_face_width_mesh_stage(true);
        assert_eq!(
            cpu_report
                .effects
                .iter()
                .map(|run| run.pf)
                .collect::<Vec<_>>(),
            [30092, 30093]
        );
        for run in &cpu_report.effects {
            assert_eq!(run.backend, PortraitBackend::Cpu, "PF{}", run.pf);
            assert!(run.dispatches.is_empty(), "PF{}", run.pf);
            assert!(run
                .fallback_reason
                .as_deref()
                .is_some_and(|reason| reason.contains("test-forced CUDA failure")));
        }

        let (cuda_frame, cuda_report) = run_face_width_mesh_stage(false);
        let all_cuda = cuda_report
            .effects
            .iter()
            .all(|run| run.backend == PortraitBackend::Cuda);
        if !all_cuda {
            assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but face-width fell back: {:?}",
                cuda_report
                    .effects
                    .iter()
                    .map(|run| run.fallback_reason.as_deref())
                    .collect::<Vec<_>>()
            );
            for run in &cuda_report.effects {
                assert_eq!(run.backend, PortraitBackend::Cpu, "PF{}", run.pf);
                assert!(run.dispatches.is_empty(), "PF{}", run.pf);
                assert!(run.executed, "PF{} did not execute", run.pf);
                assert!(run.fallback_reason.is_some(), "PF{}", run.pf);
            }
            assert_eq!(
                cuda_frame, cpu_frame,
                "unavailable-CUDA and forced-CPU face-width bytes differ"
            );
            eprintln!(
                "WL48_STAGE_NO_CUDA_FALLBACK pfs=30092,30093 backend=Cpu dispatches=0 bytes_match=1"
            );
            return;
        }
        assert_eq!(cuda_frame, cpu_frame, "CUDA and CPU face-width bytes differ");
        for run in &cuda_report.effects {
            assert!(run.executed, "PF{} did not execute", run.pf);
            assert_eq!(run.dispatches.len(), 16, "PF{}", run.pf);
            assert!(run.fallback_reason.is_none(), "PF{}", run.pf);
            let device = &run.dispatches[0].device;
            assert!(!device.is_empty());
            assert!(run.dispatches.iter().all(|dispatch| {
                dispatch.device == *device
                    && (dispatch.kernel.contains("mesh_raster_select")
                        || dispatch.kernel.contains("mesh_raster_fragment_emit"))
            }));
            eprintln!(
                "WL48_STAGE_BACKEND pf={} backend=Cuda dispatches=16 device={device}",
                run.pf
            );
        }
    }

    #[test]
    fn owned_moisture_face_borrows_with_identical_geometry() {
        use crate::portrait_moisture_inputs::OwnedMoistureFaceInputs;
        let width = 8u32;
        let height = 6u32;
        let owned = OwnedMoistureFaceInputs {
            width,
            height,
            mask_rgba: vec![0u8; (width * height * 4) as usize],
            blur_alpha: vec![0u8; (width * height) as usize],
            channel: aurora_portrait::kernels::moisture::MaskChannel::R,
            skin_map_width: 2,
            skin_map_height: 2,
            skin_map: vec![0.0f32; 4],
            hardlight: vec![1.0f32; 256],
            beauty: vec![1.0f32; 256],
            pivot: 0.5,
            gain: 1.0,
        };
        let borrowed = owned.as_borrowed();
        assert_eq!((borrowed.width, borrowed.height), (owned.width, owned.height));
        assert_eq!(
            (borrowed.skin_map_width, borrowed.skin_map_height),
            (owned.skin_map_width, owned.skin_map_height)
        );
        assert_eq!(borrowed.mask_rgba.len(), owned.mask_rgba.len());
        assert_eq!(borrowed.blur_alpha.len(), owned.blur_alpha.len());
        assert_eq!(borrowed.skin_map.len(), owned.skin_map.len());
        assert_eq!(borrowed.hardlight.len(), owned.hardlight.len());
        assert_eq!(borrowed.beauty.len(), owned.beauty.len());
        assert_eq!((borrowed.pivot, borrowed.gain), (owned.pivot, owned.gain));
    }
}
