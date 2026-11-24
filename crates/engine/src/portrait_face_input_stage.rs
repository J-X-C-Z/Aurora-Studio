use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::UNIX_EPOCH;

use aurora_portrait::kernels::face_width_414;
use aurora_portrait::{Image, Refusal};
use serde::Deserialize;

use super::{FaceLandmarkProducerInputs, ProducerInputs, SlimProducerInputs};
use crate::aimask::{Provider, Runtime, Session};
use crate::face_parsing::{
    FACE_PARSING_V2_AIMODELS_ENV, FACE_PARSING_V2_MODEL_ENV, FACE_PARSING_V2_MODEL_ROOT_ENV,
    FaceParsingProducer, FaceParsingRoi,
};

pub const C1_FACE_DETECTION_MODEL_ID: &str = "C1-FaceDetection-FP32";
pub const C1_FACE_MESH_MODEL_ID: &str = "C1-FaceMesh";
pub const C1_FACE_PARSING_MODEL_ID: &str = "C1-FaceParsing-FP16";

pub const C1_FACE_DETECTION_MODEL_SHA256: [u8; 32] =
    hex32(b"5c45500f085b636896dbc97e4f7238b2c3b0e4fe2973f7e379742074e57f695d");
pub const C1_FACE_MESH_MODEL_SHA256: [u8; 32] =
    hex32(b"70fe4e14169ca084b03b8103077a4051296e07939a19c1fdfd1f18b3792b4048");
pub const C1_FACE_PARSING_MODEL_SHA256: [u8; 32] =
    hex32(b"fc23d8ed2d1caa2ba6fbf74e93d66f51a6882395c8740dd772a1ea4d4c04c923");
pub const FACE_LANDMARK_1063025060_MODEL_SHA256: [u8; 32] =
    hex32(b"2da68b65b64fac6c3b19a9241d98dc171939e47b59ce9416633b9a6f44d0a360");
pub const FACE_LANDMARK_323773030_MODEL_SHA256: [u8; 32] =
    hex32(b"515763c7538a91a06efdbc0f64d28a16db33080972cf7ee98acccfbc6a45cd58");
pub const FACE_LANDMARK_CONSTANTS_SHA256: [u8; 32] =
    hex32(b"730b25d33e63fb921fc9739c00e21d93335cddd9ecbd8e71e8803d9f5972dbd8");

const DETECTION_SIDE: usize = 640;
const MESH_SIDE: usize = 256;
const COARSE_SIDE: usize = 64;
const DENSE_SIDE: usize = 160;
const PARSING_SIDE: usize = 513;
const MESH_POINTS: usize = 478;
const DETECTION_SCORE_MIN: f32 = 0.5;
const MESH_SCORE_MIN: f32 = 0.5;

const MODEL_INPUT_ERROR: &str = "C1 face model input";
const MODEL_OUTPUT_ERROR: &str = "C1 face model output";
const MODEL_RUNTIME_ERROR: &str = "C1 face model runtime";
const FACE_INPUT_BACKEND_ENV: &str = "AURORA_FACE_INPUT_BACKEND";
const FACE_LANDMARK_ERROR: &str = "C1 face landmarks";
const FACE_MESH_SCORE_ERROR: &str = "C1 face mesh score";
const FACE_TEETH_MAPPING_ERROR: &str = "C1 faceParsing 14-class teeth mapping";
const FACE_PARSING_V2_MODEL_ERROR: &str = "FaceParsingV2 canonical model";
const FACE_PARSING_V2_ROI_ERROR: &str = "FaceParsingV2 ROI";
const FACE_PARSING_V2_PRODUCE_ERROR: &str = "FaceParsingV2 canonical producer";
const FACE_STAGE_TRANSFORM_ERROR: &str = "face input source-to-stage transform";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaceModelIdentity {
    pub model_id: &'static str,
    pub sha256: [u8; 32],
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FaceInputModelPaths {
    pub detection: PathBuf,
    pub mesh: PathBuf,
    pub parsing: PathBuf,
    pub coarse: PathBuf,
    pub dense: PathBuf,
    pub extra52: PathBuf,
    pub landmark_constants: PathBuf,
}

impl FaceInputModelPaths {
    pub fn from_environment(include_parsing: bool) -> Result<Self, Refusal> {
        let c1_root = std::env::var_os("AURORA_C1_DIR")
            .map(PathBuf::from)
            .or_else(|| crate::aimask::c1_roots().into_iter().next())
            .ok_or(Refusal::MissingInput("AURORA_C1_DIR"))?;
        let landmark_root = std::env::var_os("AURORA_FACE_LANDMARK_ROOT")
            .map(PathBuf::from)
            .or_else(default_face_landmark_root)
            .ok_or(Refusal::MissingInput("AURORA_FACE_LANDMARK_ROOT"))?;
        let paths = Self::from_c1_and_landmark_roots(c1_root, landmark_root);
        if include_parsing && !paths.parsing.is_file() {
            return Err(Refusal::MissingInput(FACE_PARSING_V2_MODEL_ENV));
        }
        Ok(paths)
    }

    pub fn from_c1_and_landmark_roots(
        c1_root: impl AsRef<Path>,
        landmark_root: impl AsRef<Path>,
    ) -> Self {
        let c1_root = c1_root.as_ref();
        let landmark_root = landmark_root.as_ref();
        let extra52_local = landmark_root.join("2722162333").join("model.onnx");
        let extra52 = if extra52_local.is_file() {
            extra52_local
        } else {
            landmark_root
                .join("../../../../c1-pixcake/escalations/")
                .join("sol-detector-landmark-producer-v1/2722162333/model.onnx")
        };
        Self {
            detection: c1_root
                .join("AIModels")
                .join("Common")
                .join("FaceDetection")
                .join("faceDetection.onnx"),
            mesh: c1_root
                .join("AIModels")
                .join("Common")
                .join("FaceMesh")
                .join("faceMesh.onnx"),
            parsing: face_parsing_v2_model_path().unwrap_or_default(),
            coarse: landmark_root.join("assets").join("1063025060.onnx"),
            dense: landmark_root.join("assets").join("323773030.onnx"),
            extra52,
            landmark_constants: landmark_root.join("constants.json"),
        }
    }
}

fn default_face_landmark_root() -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            candidates.push(dir.join("models").join("landmark"));
            candidates.push(dir.join("landmark"));
        }
    }
    candidates.push(PathBuf::from(
        r"C:\WorkSpace\Aurora Studio\04_reference\legacy-root\逆向工程\像素蛋糕\analysis\model-recovery-landmark-independent",
    ));
    candidates.into_iter().find(|root| {
        root.join("assets").join("1063025060.onnx").is_file()
            && root.join("assets").join("323773030.onnx").is_file()
            && root.join("constants.json").is_file()
    })
}

fn is_canonical_face_parsing_v2(path: &Path) -> bool {
    let Ok(meta) = fs::metadata(path) else {
        return false;
    };
    if meta.len() != crate::face_parsing::FACE_PARSING_V2_MODEL_BYTES {
        return false;
    }
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    sha256_32(&bytes) == crate::face_parsing::FACE_PARSING_V2_MODEL_SHA256
}

fn face_parsing_v2_model_path() -> Option<PathBuf> {
    std::env::var_os(FACE_PARSING_V2_MODEL_ENV)
        .map(PathBuf::from)
        .map(|path| {
            if path.is_dir() {
                [path.join("2102828093.onnx"), path.join("fbe.onnx")]
                    .into_iter()
                    .find(|c| is_canonical_face_parsing_v2(c))
                    .unwrap_or_else(|| path.join("2102828093.onnx"))
            } else {
                path
            }
        })
        .or_else(|| {
            [FACE_PARSING_V2_MODEL_ROOT_ENV, FACE_PARSING_V2_AIMODELS_ENV]
                .iter()
                .filter_map(std::env::var_os)
                .map(PathBuf::from)
                .flat_map(|root| {
                    [
                        root.join("2102828093.onnx"),
                        root.join("fbe.onnx"),
                        root.join("models").join("2102828093.onnx"),
                        root.join("models").join("fbe.onnx"),
                        root.join("assets").join("models").join("2102828093.onnx"),
                        root.join("assets").join("models").join("fbe.onnx"),
                    ]
                })
                .find(|candidate| is_canonical_face_parsing_v2(candidate))
        })
        .or_else(|| {
            let mut candidates = Vec::new();
            if let Ok(exe) = std::env::current_exe() {
                if let Some(dir) = exe.parent() {
                    candidates.push(dir.join("models").join("2102828093.onnx"));
                    candidates.push(dir.join("models").join("fbe.onnx"));
                    candidates.push(dir.join("2102828093.onnx"));
                    candidates.push(dir.join("fbe.onnx"));
                }
            }
            candidates.push(PathBuf::from(
                r"C:\WorkSpace\Aurora Studio\90_work\build-cache\faceparsing\2102828093.onnx",
            ));
            candidates.push(PathBuf::from(
                r"C:\WorkSpace\Aurora Studio\90_work\build-cache\faceparsing\fbe.onnx",
            ));
            candidates
                .into_iter()
                .find(|candidate| is_canonical_face_parsing_v2(candidate))
        })
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceNormalization {
    pub channel_order: &'static str,
    pub mean: [f32; 3],
    pub scale: [f32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum FaceInputBackend {
    Cpu,
    Cuda,
    DirectMl,
    Auto,
}

impl FaceInputBackend {
    fn from_env() -> Result<Self, Refusal> {
        let raw = std::env::var(FACE_INPUT_BACKEND_ENV).unwrap_or_default();
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "cpu" => Ok(Self::Cpu),
            "cuda" => Ok(Self::Cuda),
            "directml" | "dml" => Ok(Self::DirectMl),
            "auto" => Ok(Self::Auto),
            _ => Err(Refusal::ValueOutOfRange(FACE_INPUT_BACKEND_ENV)),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::DirectMl => "directml",
            Self::Auto => "auto",
        }
    }

    fn open(self, runtime: &Runtime, path: &Path) -> Result<Session, Refusal> {
        let providers = match self {
            Self::Cpu => vec![Provider::Cpu],
            Self::Cuda => vec![Provider::Cuda],
            Self::DirectMl => vec![Provider::DirectMl],
            Self::Auto => vec![Provider::Cuda, Provider::DirectMl, Provider::Cpu],
        };
        runtime.open_session_on(path, &providers).map_err(|error| {
            log::warn!(
                "C1 face model {} on {} refused: {error:#}",
                path.display(),
                self.label()
            );
            Refusal::MissingInput(MODEL_RUNTIME_ERROR)
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct C1FaceDetection {
    pub score: f32,
    pub landmarks_source: [[f32; 2]; 5],
    pub source_to_input: [[f32; 3]; 2],
    pub input_size: [u32; 2],
    pub normalization: FaceNormalization,
    pub provider: &'static str,
    pub model: FaceModelIdentity,
}

#[derive(Clone, Debug, PartialEq)]
pub struct C1FaceMesh {
    pub score: f32,
    pub landmarks_source: Vec<[f32; 3]>,
    pub source_to_crop: [[f32; 3]; 2],
    pub crop_side: f32,
    pub input_size: [u32; 2],
    pub normalization: FaceNormalization,
    pub provider: &'static str,
    pub model: FaceModelIdentity,
}

#[derive(Clone, Debug)]
pub struct C1FaceParsing {
    pub logits: Arc<[f32]>,
    pub class_map: Arc<[u8]>,
    pub source_to_crop: [[f32; 3]; 2],
    pub input_size: [u32; 2],
    pub classes: u32,
    pub normalization: FaceNormalization,
    pub semantic_classes: &'static str,
    pub model: FaceModelIdentity,
}

impl C1FaceParsing {
    pub fn class_logits(&self, class: usize) -> Option<&[f32]> {
        if class >= self.classes as usize {
            return None;
        }
        let plane = PARSING_SIDE * PARSING_SIDE;
        self.logits.get(class * plane..(class + 1) * plane)
    }

    pub fn teeth_mask(&self) -> Result<&[u8], Refusal> {
        Err(Refusal::MissingInput(FACE_TEETH_MAPPING_ERROR))
    }
}

#[derive(Clone, Debug)]
pub struct FaceInputBundle {
    pub job_id: u64,
    pub source_size: [u32; 2],
    pub source_sha256: [u8; 32],
    pub detection: C1FaceDetection,
    pub mesh: C1FaceMesh,
    pub roi_xywh: [f32; 4],
    pub roi_normalized_xywh: [f32; 4],
    pub points17: Vec<[f32; 2]>,
    pub points142: Vec<[f32; 2]>,
    pub points194: Vec<[f32; 2]>,
    pub points414: Vec<[f32; 2]>,
    pub landmark_producer: FaceLandmarkProducerInputs,
    pub model_inputs: Option<ProducerInputs>,
}

impl FaceInputBundle {
    pub fn validate_ticket(&self, source: &Image, job_id: Option<u64>) -> Result<(), Refusal> {
        if self.job_id == 0 || Some(self.job_id) != job_id {
            return Err(Refusal::ValueOutOfRange("face input job ticket"));
        }
        if self.source_size != [source.width, source.height] {
            return Err(Refusal::ShapeMismatch("face input source size"));
        }
        if self.source_sha256 != sha256_32(&source.pixels) {
            return Err(Refusal::ValueOutOfRange("face input source digest"));
        }
        if self.points194.len() != 194
            || self.points414.len() != 414
            || self
                .points194
                .iter()
                .chain(&self.points414)
                .flatten()
                .any(|value| !value.is_finite())
        {
            return Err(Refusal::ShapeMismatch("face input landmarks"));
        }
        if let Some(inputs) = self.model_inputs.as_ref() {
            if let Some(teeth) = inputs.teeth.as_ref() {
                if teeth.mask_width != source.width
                    || teeth.mask_height != source.height
                    || teeth.mask_g.len() != source.pixel_count()
                {
                    return Err(Refusal::ShapeMismatch("PF108 teeth mask"));
                }
            }
        }
        Ok(())
    }

    pub fn normalized_points194(&self) -> Vec<[f32; 2]> {
        normalize_points(
            &self.points194,
            self.source_size[0] as f32,
            self.source_size[1] as f32,
        )
    }

    pub fn normalized_points414(&self) -> Vec<[f32; 2]> {
        normalize_points(
            &self.points414,
            self.source_size[0] as f32,
            self.source_size[1] as f32,
        )
    }

    pub fn slim_inputs(&self, source: &Image) -> Result<SlimProducerInputs, Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        self.landmark_producer.slim_inputs(self.job_id, source)
    }

    pub fn slim_inputs_for_stage(
        &self,
        transform: &FaceStageTransform,
        stage: &Image,
    ) -> Result<SlimProducerInputs, Refusal> {
        if transform.source_size != self.source_size
            || transform.stage_size != [stage.width, stage.height]
        {
            return Err(Refusal::ShapeMismatch(FACE_STAGE_TRANSFORM_ERROR));
        }
        let points194 = self
            .points194
            .iter()
            .map(|point| transform.source_point_to_stage(*point))
            .collect::<Result<Vec<_>, _>>()?;
        SlimProducerInputs::from_landmarks(self.job_id, stage, points194)
    }

    pub fn attach_slim(
        &self,
        source: &Image,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        producer.slim = Some(self.slim_inputs(source)?);
        Ok(())
    }

    /// Binds caller-produced face-geometry draws to this exact face-input job.
    /// This is the attachment point for V-face atlases and other same-frame
    /// producer rows that already carry their captured VBO/EBO/UV bytes.
    pub fn attach_face_geometry(
        &self,
        source: &Image,
        inputs: super::FaceGeometryInputs,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        if inputs.job_id != self.job_id
            || (inputs.width, inputs.height) != (source.width, source.height)
            || inputs.draws.is_empty()
        {
            return Err(Refusal::ShapeMismatch("face geometry same-frame binding"));
        }
        let digest = super::sha256_32(&source.pixels);
        if inputs.draws.iter().any(|draw| draw.source_sha256 != digest) {
            return Err(Refusal::ValueOutOfRange("face geometry frame digest"));
        }
        if let Some(existing) = producer.face_geometry.as_mut() {
            if existing.job_id == inputs.job_id
                && (existing.width, existing.height) == (inputs.width, inputs.height)
            {
                for draw in inputs.draws {
                    if let Some(slot) = existing.draws.iter_mut().find(|d| d.pf == draw.pf) {
                        *slot = draw;
                    } else {
                        existing.draws.push(draw);
                    }
                }
                return Ok(());
            }
        }
        producer.face_geometry = Some(inputs);
        Ok(())
    }

    /// Attaches PF247/PF248 V-face geometry using the captured V-face atlases.
    pub fn attach_vface(
        &self,
        source: &Image,
        controls247_248: [f32; 2],
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        if controls247_248 == [0.0, 0.0] {
            return Ok(());
        }
        let mut rows = Vec::new();
        let mut atlases = Vec::new();
        for row in aurora_portrait::cuda_vface::ROWS {
            let degree = aurora_portrait::cuda_vface::degree_for(
                row,
                controls247_248[0] * 100.0,
                controls247_248[1] * 100.0,
            )?;
            if degree > 0.0 {
                rows.push(row);
                atlases.push(super::portrait_v2_mesh::VFaceAtlas::captured(row));
            }
        }
        let digest = super::sha256_32(&source.pixels);
        let inputs = super::FaceGeometryInputs {
            job_id: self.job_id,
            width: source.width,
            height: source.height,
            draws: vec![super::FaceGeometryDraw {
                pf: 247,
                controls: controls247_248,
                source_sha256: digest,
                geometry: super::FaceGeometry::VFace {
                    rows,
                    encoded_offset: Vec::new(),
                    atlases,
                },
            }],
        };
        self.attach_face_geometry(source, inputs, producer)
    }

    /// Attaches PF202 (`HairLine`) and PF232 (`SkullEnhance`) using the vendor
    /// `FaceSurgery/V2/effect.json` `FacialRefinePart[39]` and `FacialRefinePart[0]`
    /// coefficients.
    pub fn attach_default_face_detail(
        &self,
        source: &Image,
        controls202: [f32; 2],
        controls232: [f32; 2],
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        const PF202_PARAM_LIST5: [f32; 5] = [
            0.139_999_999_9,
            0.130_000_000_1,
            0.100_000_000_1,
            0.056_000_000_2,
            0.0,
        ];
        const PF202_FACTOR_LIST27: [f32; 27] = [
            0.0, 0.0, 0.02, 0.03, 0.06, 0.11, 0.21, 0.27, 0.33, 0.39, 0.45, 0.46, 0.4699, 0.47,
            0.4699, 0.46, 0.45, 0.39, 0.33, 0.27, 0.21, 0.11, 0.06, 0.03, 0.02, 0.0, 0.0,
        ];
        const PF232_FACTORS15: [f32; 15] = [
            0.789_999_96,
            0.799_999_95,
            0.799_999_95,
            0.799_999_95,
            0.769_999_98,
            0.75,
            0.719_999_97,
            0.699_999_99,
            0.719_999_97,
            0.75,
            0.769_999_98,
            0.799_999_95,
            0.799_999_95,
            0.799_999_95,
            0.789_999_96,
        ];
        let mut weights135 = [0.0f32; 135];
        for (r, &ring) in PF202_PARAM_LIST5.iter().enumerate() {
            for (c, &col) in PF202_FACTOR_LIST27.iter().enumerate() {
                weights135[r * 27 + c] = ring * col;
            }
        }
        self.attach_face_detail(
            source,
            &weights135,
            &PF232_FACTORS15,
            controls202,
            controls232,
            true,
            producer,
        )
    }

    /// Discovers local FacePlump ONNX models and builds same-job `PlumpProducerInputs`
    /// for all active `PF30191..=PF30210` controls in `extra`.
    pub fn attach_plump_from_extra(
        &self,
        source: &Image,
        extra: &std::collections::BTreeMap<String, f32>,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        let mut requested_pfs = Vec::new();
        for &pf in &super::PLUMP_PFS {
            if let Some(&val) = extra.get(&format!("pc:{pf}")) {
                if val != 0.0 {
                    requested_pfs.push(pf);
                }
            }
        }
        if requested_pfs.is_empty() {
            return Ok(());
        }
        let assets = super::discover_plump_onnx_models()?;
        let mut slots = [0.0f32; 17];
        for (slot_idx, pf) in [
            (0usize, 30_191u32),
            (3, 30_194),
            (6, 30_197),
            (7, 30_198),
            (12, 30_203),
            (13, 30_204),
            (16, 30_207),
        ] {
            if let Some(&val) = extra.get(&format!("pc:{pf}")) {
                slots[slot_idx] = val;
            }
        }
        for sub_idx in [1usize, 2, 4, 5, 8, 9, 10, 11, 14, 15] {
            slots[sub_idx] = 1.0;
        }
        let full_face_level = if requested_pfs.contains(&30_210) {
            let raw = extra.get("pc:30210").copied().unwrap_or(0.0);
            let normalized = if raw > 0.0 && raw < 1.0 {
                (raw * 4.0).round().clamp(1.0, 4.0)
            } else {
                raw
            };
            let desc = super::full_face_descriptor_from_ui(normalized)?;
            Some(desc as u8)
        } else {
            None
        };
        let x0 = (self.roi_xywh[0].floor().max(0.0) as u32).min(source.width.saturating_sub(1));
        let y0 = (self.roi_xywh[1].floor().max(0.0) as u32).min(source.height.saturating_sub(1));
        let x1 = ((self.roi_xywh[0] + self.roi_xywh[2]).ceil().max(1.0) as u32)
            .clamp(x0 + 1, source.width);
        let y1 = ((self.roi_xywh[1] + self.roi_xywh[3]).ceil().max(1.0) as u32)
            .clamp(y0 + 1, source.height);
        let roi = aurora_portrait::kernels::plump_pipeline::Roi {
            x: x0,
            y: y0,
            width: x1 - x0,
            height: y1 - y0,
        };
        let plump = super::build_plump_producer_inputs(super::PlumpBuildRequest {
            current: source,
            job_id: self.job_id,
            face_id: 1,
            slots,
            requested_pfs,
            full_face_level,
            landmarks: &self.points414,
            roi,
            plane4: None,
            asset: &assets[2],
            channel_map: aurora_portrait::kernels::plump_pipeline::ChannelMap {
                color: [0, 1, 2],
                split_mask: 3,
            },
            color_encoding: aurora_portrait::kernels::plump_pipeline::ColorEncoding::Unit,
        })?;
        producer.write_plump(plump);
        Ok(())
    }

    /// Attaches PF8033 (微笑嘴角) from discovered local CPU-forward artifacts
    /// (`780929437`, `dx`, `dy`, `mask`) when the same-frame receipt matches.
    pub fn attach_pf8033_from_extra(
        &self,
        source: &Image,
        extra: &std::collections::BTreeMap<String, f32>,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        let c8033 = extra.get("pc:8033").copied().unwrap_or(0.0);
        if c8033 == 0.0 {
            return Ok(());
        }
        let receipt_path = std::env::var_os(
            super::portrait_pf8033_cpu_forward::PF8033_CPU_FORWARD_RECEIPT_ENV,
        )
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../assets/portrait/smile/pf8033-cpu-forward.json")
        });
        let receipt_bytes = std::fs::read(&receipt_path)
            .map_err(|_| Refusal::MissingInput("PF8033 CPU forward receipt"))?;
        let receipt = super::portrait_pf8033_cpu_forward::Pf8033CpuForwardReceipt::parse(
            receipt_path,
            &receipt_bytes,
        )?;
        let mut artifacts =
            super::portrait_pf8033_cpu_forward::load_pf8033_cpu_forward_artifacts(&receipt, source)?;
        artifacts.host_split_receipt.job_id = self.job_id;
        let inputs = super::FaceGeometryInputs {
            job_id: self.job_id,
            width: source.width,
            height: source.height,
            draws: vec![super::FaceGeometryDraw {
                pf: 8033,
                controls: [c8033, 0.0],
                source_sha256: sha256_32(&source.pixels),
                geometry: super::FaceGeometry::Smile {
                    dx: artifacts.dx,
                    dy: artifacts.dy,
                    mask: artifacts.mask,
                    host_split_receipt: Some(artifacts.host_split_receipt),
                },
            }],
        };
        self.attach_face_geometry(source, inputs, producer)
    }

    /// Attaches same-frame historical-tone rows (`PF30009` hair color, `PF30014`
    /// 3D neutral gray, `PF100004` eye red-vein removal) when active in `extra`.
    pub fn attach_historical_tone_from_extra(
        &self,
        source: &Image,
        masks: Option<&super::PortraitMasks>,
        extra: &std::collections::BTreeMap<String, f32>,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        use crate::portrait_historical_tone_provider::{
            build_historical_tone_inputs, HistoricalToneEyeRedVeinsPayload,
            HistoricalToneGray3dPayload, HistoricalToneHairColorPayload,
            HistoricalToneProviderPayload, HistoricalToneProviderRow, HistoricalToneRowData,
            HistoricalToneRowJob,
        };

        self.validate_ticket(source, Some(self.job_id))?;
        let control = |pf: u32| extra.get(&format!("pc:{pf}")).copied().unwrap_or(0.0);
        let c30009 = control(30009);
        let c30014 = control(30014);
        let c100004 = control(100004);
        if c30009 == 0.0 && c30014 == 0.0 && c100004 == 0.0 {
            return Ok(());
        }

        let pixels = source.pixel_count();
        let digest = sha256_32(&source.pixels);
        let size = [source.width, source.height];
        let mut rows = Vec::new();

        if c30009 != 0.0 {
            if let Some(hair_mask) = masks.and_then(|m| {
                m.hair
                    .as_ref()
                    .map(|plane| sample_mask_plane(plane, m.width, m.height, source.width, source.height))
            }) {
                let mut sum = [0.0f64; 3];
                let mut wsum = 0.0f64;
                for (i, &w) in hair_mask.iter().enumerate() {
                    if w > 0.05 {
                        let wf = f64::from(w);
                        sum[0] += f64::from(source.pixels[i * 4]) * wf;
                        sum[1] += f64::from(source.pixels[i * 4 + 1]) * wf;
                        sum[2] += f64::from(source.pixels[i * 4 + 2]) * wf;
                        wsum += wf;
                    }
                }
                if wsum > 1.0 {
                    let avg = [
                        (sum[0] / wsum).round().clamp(0.0, 255.0) as u8,
                        (sum[1] / wsum).round().clamp(0.0, 255.0) as u8,
                        (sum[2] / wsum).round().clamp(0.0, 255.0) as u8,
                    ];
                    let mut effect_rgb = Vec::with_capacity(pixels * 3);
                    for px in source.pixels.chunks_exact(4) {
                        effect_rgb.push(((f32::from(px[0]) * 0.65 + f32::from(avg[0]) * 0.35).round()) as u8);
                        effect_rgb.push(((f32::from(px[1]) * 0.65 + f32::from(avg[1]) * 0.35).round()) as u8);
                        effect_rgb.push(((f32::from(px[2]) * 0.65 + f32::from(avg[2]) * 0.35).round()) as u8);
                    }
                    rows.push(HistoricalToneProviderRow {
                        frame: source.clone(),
                        job: HistoricalToneRowJob {
                            source_sha256: digest,
                            size,
                            control: c30009.clamp(0.0, 1.0),
                        },
                        data: HistoricalToneRowData::HairColor(HistoricalToneHairColorPayload {
                            effect_rgb,
                            hair_mask,
                        }),
                    });
                }
            }
        }

        if c30014 != 0.0 {
            if let Some(parsing) = self.model_inputs.as_ref().and_then(|m| m.parsing.as_ref()) {
                let mut p = parsing.clone();
                p.job_id = self.job_id;
                let skin_u8 = crate::face_parsing::project_face_parsing_skin_mask(
                    &p,
                    source.width,
                    source.height,
                )?;
                let mut src_rgb = Vec::with_capacity(pixels * 3);
                for px in source.pixels.chunks_exact(4) {
                    src_rgb.push(px[0] as f32 / 255.0);
                    src_rgb.push(px[1] as f32 / 255.0);
                    src_rgb.push(px[2] as f32 / 255.0);
                }
                let (_median, gauss, _, _) = super::hlf_preprocess(
                    &src_rgb,
                    source.width,
                    source.height,
                    super::HlfAutoProducerOptions::default(),
                );
                let mut light = Vec::with_capacity(pixels);
                let mut dark = Vec::with_capacity(pixels);
                let mut alpha = Vec::with_capacity(pixels);
                let mut mask = Vec::with_capacity(pixels);
                for i in 0..pixels {
                    let l_src = 0.299 * src_rgb[i * 3] + 0.587 * src_rgb[i * 3 + 1] + 0.114 * src_rgb[i * 3 + 2];
                    let l_ref = 0.299 * gauss[i * 3] + 0.587 * gauss[i * 3 + 1] + 0.114 * gauss[i * 3 + 2];
                    let diff = l_ref - l_src;
                    light.push(diff.max(0.0).clamp(0.0, 1.0));
                    dark.push((-diff).max(0.0).clamp(0.0, 1.0));
                    let m = f32::from(skin_u8[i]) / 255.0;
                    alpha.push(m);
                    mask.push(m);
                }
                rows.push(HistoricalToneProviderRow {
                    frame: source.clone(),
                    job: HistoricalToneRowJob {
                        source_sha256: digest,
                        size,
                        control: c30014.clamp(0.0, 1.0),
                    },
                    data: HistoricalToneRowData::Gray3d(HistoricalToneGray3dPayload {
                        light,
                        dark,
                        alpha,
                        mask,
                    }),
                });
            }
        }

        if c100004 != 0.0 {
            if let Some(parsing) = self.model_inputs.as_ref().and_then(|m| m.parsing.as_ref()) {
                let mut p = parsing.clone();
                p.job_id = self.job_id;
                let eye_alpha = crate::face_parsing::project_face_parsing_skin_mask(
                    &p,
                    source.width,
                    source.height,
                )?;
                let mut rgb = Vec::with_capacity(pixels * 3);
                for px in source.pixels.chunks_exact(4) {
                    let luma = (0.299 * f32::from(px[0]) + 0.587 * f32::from(px[1]) + 0.114 * f32::from(px[2]))
                        .clamp(0.0, 255.0);
                    // Suppress excess red relative to local luma for red-vein reduction.
                    rgb.push((f32::from(px[0]) * 0.5 + luma * 0.5).round().clamp(0.0, 255.0) as u8);
                    rgb.push(px[1]);
                    rgb.push(px[2]);
                }
                let mut eye_mask = vec![0u8; pixels];
                // Restrict to the eye bounding boxes from 194 landmarks.
                let mut boxes = Vec::new();
                for (a, b) in [(57usize, 63usize), (77, 83)] {
                    let cx = (self.points194[a][0] + self.points194[b][0]) * 0.5;
                    let cy = (self.points194[a][1] + self.points194[b][1]) * 0.5;
                    let rx = ((self.points194[b][0] - self.points194[a][0]).abs() * 0.65).max(6.0);
                    let ry = (rx * 0.55).max(4.0);
                    boxes.push((cx, cy, rx, ry));
                }
                for y in 0..source.height as usize {
                    for x in 0..source.width as usize {
                        let fx = x as f32;
                        let fy = y as f32;
                        let inside = boxes.iter().any(|&(cx, cy, rx, ry)| {
                            ((fx - cx) / rx).powi(2) + ((fy - cy) / ry).powi(2) <= 1.0
                        });
                        if inside {
                            let idx = y * source.width as usize + x;
                            eye_mask[idx] = eye_alpha[idx].max(160);
                        }
                    }
                }
                rows.push(HistoricalToneProviderRow {
                    frame: source.clone(),
                    job: HistoricalToneRowJob {
                        source_sha256: digest,
                        size,
                        control: c100004.clamp(0.0, 1.0),
                    },
                    data: HistoricalToneRowData::EyeRedVeins(HistoricalToneEyeRedVeinsPayload {
                        effect_rgb: rgb,
                        alpha: eye_mask,
                        strength: c100004.clamp(0.0, 1.0),
                    }),
                });
            }
        }

        if rows.is_empty() {
            return Ok(());
        }
        let gray_3d_controls = [
            control(30015).clamp(-1.0, 1.0),
            control(30016).clamp(-1.0, 1.0),
        ];
        let built = build_historical_tone_inputs(HistoricalToneProviderPayload {
            ticket: self.job_id,
            gray_3d_controls,
            rows,
        })?;
        match producer.historical_tone.as_mut() {
            Some(existing) => {
                for (pf, input) in built {
                    existing.insert(pf, input);
                }
            }
            None => producer.historical_tone = Some(built),
        }
        Ok(())
    }

    /// Attaches same-frame historical-skin rows (`PF21105` body smoothing,
    /// `PF90086` body tone unify) from `PortraitMasks::body` when active.
    pub fn attach_historical_skin_from_extra(
        &self,
        source: &Image,
        masks: Option<&super::PortraitMasks>,
        extra: &std::collections::BTreeMap<String, f32>,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        use super::portrait_historical_skin::{
            HistoricalSkinData, HistoricalSkinImmediateInputs, HistoricalSkinImmediateRow,
            HistoricalSkinPass,
        };

        self.validate_ticket(source, Some(self.job_id))?;
        let control = |pf: u32| extra.get(&format!("pc:{pf}")).copied().unwrap_or(0.0);
        let c21105 = control(21105);
        let c90086 = control(90086);
        if c21105 == 0.0 && c90086 == 0.0 {
            return Ok(());
        }
        let Some(parsing) = self.model_inputs.as_ref().and_then(|m| m.parsing.as_ref()) else {
            return Err(Refusal::MissingInput("FaceParsingV2 parsing planes"));
        };
        let mut p = parsing.clone();
        p.job_id = self.job_id;
        let pixels = source.pixel_count();
        let body_mask_f32 = if let Some(body) = masks.and_then(|m| {
            m.body
                .as_ref()
                .map(|plane| sample_mask_plane(plane, m.width, m.height, source.width, source.height))
        }) {
            body
        } else {
            return Ok(());
        };
        let body_mask_u8: Vec<u8> = body_mask_f32
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();

        let mut rows = Vec::new();
        if c21105 != 0.0 {
            let mut src_rgb = Vec::with_capacity(pixels * 3);
            for px in source.pixels.chunks_exact(4) {
                src_rgb.push(px[0] as f32 / 255.0);
                src_rgb.push(px[1] as f32 / 255.0);
                src_rgb.push(px[2] as f32 / 255.0);
            }
            let (median_rgb, _, _, _) = super::hlf_preprocess(
                &src_rgb,
                source.width,
                source.height,
                super::HlfAutoProducerOptions::default(),
            );
            let mut smooth_rgba = Vec::with_capacity(pixels * 4);
            for i in 0..pixels {
                for c in 0..3 {
                    smooth_rgba.push((median_rgb[i * 3 + c].clamp(0.0, 1.0) * 255.0).round() as u8);
                }
                smooth_rgba.push(source.pixels[i * 4 + 3]);
            }
            let effect = Image::new(source.width, source.height, smooth_rgba)
                .ok_or(Refusal::ShapeMismatch("PF21105 smooth frame"))?;
            rows.push(HistoricalSkinImmediateRow {
                pf: 21105,
                ticket: self.job_id,
                control: c21105.clamp(0.0, 1.0),
                data: HistoricalSkinData::BodySkin(vec![HistoricalSkinPass {
                    mask: body_mask_u8,
                    effect,
                }]),
            });
        }

        if c90086 != 0.0 {
            let m = body_mask_f32;
            let o = vec![0.0f32; pixels];
            rows.push(HistoricalSkinImmediateRow {
                pf: 90086,
                ticket: self.job_id,
                control: c90086.clamp(0.0, 1.0),
                data: HistoricalSkinData::BodyTone {
                    m,
                    o,
                    w_m: c90086.clamp(0.0, 1.0),
                    w_o: 0.0,
                },
            });
        }

        if rows.is_empty() {
            return Ok(());
        }
        producer.write_immediate_historical_skin(
            source,
            &p,
            HistoricalSkinImmediateInputs {
                ticket: self.job_id,
                rows,
            },
        )
    }

    /// Builds PF202/PF232 from this bundle's 194/414 points and explicit
    /// recovered producer coefficients. No weights, factors or EBO are
    /// synthesized when the caller does not provide them.
    pub fn attach_face_detail(
        &self,
        source: &Image,
        weights135: &[f32],
        factors15: &[f32],
        controls202: [f32; 2],
        controls232: [f32; 2],
        symmetric_second_slot: bool,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        let hair = aurora_portrait::kernels::face_detail::hair_line(
            &self.points194,
            weights135,
            controls202[0] * 100.0,
            symmetric_second_slot,
            source.width,
            source.height,
        )?;
        let skull = aurora_portrait::kernels::face_detail::skull_top(
            controls232[0] * 100.0,
            &self.points414,
            factors15,
            source.width as f32,
            source.height as f32,
        )?;
        let digest = super::sha256_32(&source.pixels);
        let rules = aurora_portrait::raster::Rules {
            winding: aurora_portrait::raster::Winding::PositiveAreaIsFront,
            cull: aurora_portrait::raster::Cull::None,
            edge: aurora_portrait::raster::Edge::Inclusive,
            overlap: aurora_portrait::raster::Overlap::Last,
        };
        let inputs = super::FaceGeometryInputs {
            job_id: self.job_id,
            width: source.width,
            height: source.height,
            draws: vec![
                super::FaceGeometryDraw {
                    pf: 202,
                    controls: controls202,
                    source_sha256: digest,
                    geometry: super::FaceGeometry::HairLine {
                        dest: hair.destination,
                        uv: hair
                            .source
                            .iter()
                            .map(|point| {
                                [point[0] / source.width as f32, point[1] / source.height as f32]
                            })
                            .collect(),
                        ebo: hair.indices.to_vec(),
                        rules,
                        edge: aurora_portrait::raster_fragment::Edge::ReshapeNormal,
                        defines: aurora_portrait::raster_fragment::Defines::plain(),
                        clamp: aurora_portrait::raster_fragment::Clamp::disabled(),
                    },
                },
                super::FaceGeometryDraw {
                    pf: 232,
                    controls: controls232,
                    source_sha256: digest,
                    geometry: super::FaceGeometry::Skull {
                        anchor: skull.anchor,
                        source: skull.source,
                        warped: skull.warped,
                    },
                },
            ],
        };
        self.attach_face_geometry(source, inputs, producer)
    }

    pub fn attach_pf108_inputs(
        &self,
        source: &Image,
        bright_ui: f32,
        deyellow_ui: f32,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        if !bright_ui.is_finite()
            || !deyellow_ui.is_finite()
            || !(0.0..=100.0).contains(&bright_ui)
            || !(0.0..=100.0).contains(&deyellow_ui)
        {
            return Err(Refusal::ValueOutOfRange("PF108 UI percent"));
        }
        let inputs = self
            .model_inputs
            .as_ref()
            .ok_or(Refusal::MissingInput(FACE_PARSING_V2_PRODUCE_ERROR))?;
        let mut parsing = inputs
            .parsing
            .clone()
            .ok_or(Refusal::MissingInput("FaceParsingV2 parsing planes"))?;
        let mut teeth = inputs
            .teeth
            .clone()
            .ok_or(Refusal::MissingInput("PF108 teeth mask"))?;
        parsing.job_id = self.job_id;
        teeth.job_id = self.job_id;
        teeth.bright_ui = bright_ui;
        teeth.deyellow_ui = deyellow_ui;
        if producer.parsing.is_none() {
            producer.parsing = Some(parsing);
        }
        if producer.teeth.is_none() {
            producer.teeth = Some(teeth);
        }
        if producer.pf90004.is_none() {
            producer.pf90004 = inputs.pf90004.clone();
        }
        Ok(())
    }

    /// Attaches PF30013 (中性灰磨皮 FlatGray) using the canonical `fbe.onnx`
    /// FaceParsingV2 class-1 skin mask projected to the full source frame.
    pub fn attach_pf30013_inputs(
        &self,
        source: &Image,
        control30013: f32,
        gray_3d_controls: [f32; 2],
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        if control30013 == 0.0 {
            return Ok(());
        }
        let inputs = self
            .model_inputs
            .as_ref()
            .ok_or(Refusal::MissingInput(FACE_PARSING_V2_PRODUCE_ERROR))?;
        let mut parsing = inputs
            .parsing
            .clone()
            .ok_or(Refusal::MissingInput("FaceParsingV2 parsing planes"))?;
        parsing.job_id = self.job_id;
        let face_roi = FaceInputProducer::parsing_roi(&self.detection, self.roi_xywh)?;
        let tone_roi = aurora_portrait::kernels::skin_adjust::Roi {
            x: 0,
            y: 0,
            width: source.width,
            height: source.height,
        };
        let row = crate::portrait_historical_tone_provider::build_pf30013_face_parsing_row(
            self.job_id,
            control30013,
            source.clone(),
            source.clone(),
            face_roi,
            tone_roi,
            &parsing,
        )?;
        let built = crate::portrait_historical_tone_provider::build_historical_tone_inputs(
            crate::portrait_historical_tone_provider::HistoricalToneProviderPayload {
                ticket: self.job_id,
                gray_3d_controls,
                rows: vec![row],
            },
        )?;
        match producer.historical_tone.as_mut() {
            Some(existing) => {
                for (pf, input) in built {
                    existing.insert(pf, input);
                }
            }
            None => producer.historical_tone = Some(built),
        }
        Ok(())
    }

    /// Builds eye, brow, and lower-eyelid mesh draws (`PF8001..=PF8022`, `PF30073`, `PF30079`)
    /// from the 194 face landmarks and attaches them to `producer.face_geometry`.
    pub fn attach_eye_and_brow_geometry(
        &self,
        source: &Image,
        extra: &std::collections::BTreeMap<String, f32>,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        use aurora_portrait::kernels::eye_geometry::{self, Point};
        use aurora_portrait::raster::{Cull, Edge as RasterEdge, Overlap, Rules, Winding};
        use aurora_portrait::raster_fragment::{Clamp, Defines, Edge as FragmentEdge};

        self.validate_ticket(source, Some(self.job_id))?;
        let w = source.width as f32;
        let h = source.height as f32;
        let digest = super::sha256_32(&source.pixels);
        let rules = Rules {
            winding: Winding::PositiveAreaIsFront,
            cull: Cull::None,
            edge: RasterEdge::Inclusive,
            overlap: Overlap::Last,
        };
        let points194: Vec<Point> = self
            .points194
            .iter()
            .map(|p| Point::new(p[0], p[1]))
            .collect();
        let control = |pf: u32| extra.get(&format!("pc:{pf}")).copied().unwrap_or(0.0);
        let mut draws = Vec::new();

        // Helper for two-quad eye effects: 8011/8012 (pupil size), 8013/8014 (eye height), 8019/8020 (eye width)
        let iw = source.width as i32;
        let ih = source.height as i32;
        let unit_scale = Point::new(1.0, 1.0);
        for (left_pf, right_pf, mode) in [
            (8011u32, 8012u32, 0u8),
            (8013, 8014, 1),
            (8019, 8020, 2),
        ] {
            let c_left = control(left_pf);
            let c_right = control(right_pf);
            if c_left == 0.0 && c_right == 0.0 {
                continue;
            }
            let (l_deg, r_deg) = eye_pair_args(mode, c_left, c_right);
            let quads = match mode {
                0 => eye_geometry::row11_pupil_size(
                    &points194,
                    iw,
                    ih,
                    unit_scale,
                    l_deg,
                    r_deg,
                    1.0,
                    1.0,
                )
                .map_err(|_| Refusal::ShapeMismatch("eye pupil size geometry"))?,
                1 => eye_geometry::row12_eye_height(
                    &points194,
                    iw,
                    ih,
                    unit_scale,
                    l_deg,
                    r_deg,
                    1.0,
                )
                .map_err(|_| Refusal::ShapeMismatch("eye height geometry"))?,
                _ => eye_geometry::row13_eye_width(
                    &points194,
                    iw,
                    ih,
                    unit_scale,
                    l_deg,
                    r_deg,
                    eye_geometry::ROW13_FACTOR_LIST4,
                )
                .map_err(|_| Refusal::ShapeMismatch("eye width geometry"))?,
            };
            if quads.quads.is_empty() {
                continue;
            }
            // Deform UV inside the eye quads according to degree so the raster pass warps pixels
            let mut dest = Vec::with_capacity(quads.quads.len() * 4);
            let mut uv = Vec::with_capacity(quads.quads.len() * 4);
            let mut quad_ebo = Vec::with_capacity(quads.quads.len() * 6);
            for (side_idx, quad) in quads.quads.iter().enumerate() {
                let base = (side_idx * 4) as u16;
                quad_ebo.extend_from_slice(&[
                    base,
                    base + 1,
                    base + 2,
                    base + 1,
                    base + 2,
                    base + 3,
                ]);
                let deg01 = eye_pair_deg01(quads.quads.len(), side_idx, c_left, c_right);
                let center_uv = Point::new(
                    (quad.uv[0].x + quad.uv[1].x + quad.uv[2].x + quad.uv[3].x) * 0.25,
                    (quad.uv[0].y + quad.uv[1].y + quad.uv[2].y + quad.uv[3].y) * 0.25,
                );
                for i in 0..4 {
                    dest.push([quad.pixels[i].x, quad.pixels[i].y]);
                    let du = quad.uv[i].x - center_uv.x;
                    let dv = quad.uv[i].y - center_uv.y;
                    let (su, sv) = match mode {
                        0 => {
                            let scale = 1.0 - 0.12 * deg01;
                            (center_uv.x + du * scale, center_uv.y + dv * scale)
                        }
                        1 => {
                            let scale_y = 1.0 - 0.15 * deg01;
                            (quad.uv[i].x, center_uv.y + dv * scale_y)
                        }
                        _ => {
                            let scale_x = 1.0 - 0.15 * deg01;
                            (center_uv.x + du * scale_x, quad.uv[i].y)
                        }
                    };
                    uv.push([su.clamp(0.0, 1.0), sv.clamp(0.0, 1.0)]);
                }
            }
            draws.push(super::FaceGeometryDraw {
                pf: left_pf,
                controls: [c_left, c_right],
                source_sha256: digest,
                geometry: super::FaceGeometry::Eye {
                    dest,
                    uv,
                    ebo: quad_ebo,
                    rules,
                    edge: FragmentEdge::ReshapeNormal,
                    defines: Defines::plain(),
                    clamp: Clamp::disabled(),
                },
            });
        }

        // 120-vertex eye-pair effects: 8015/8016 (eye distance), 8017/8018 (eye angle), 8021/8022 (eye position)
        let mut eye120_ebo = Vec::new();
        for side in 0..2u16 {
            let base = side * 60;
            // Connect upper outer (0..15) to upper inner (15..30), upper inner to lower inner (45..60), lower inner to lower outer (30..45)
            for (r0, r1) in [(0u16, 15u16), (15, 45), (45, 30)] {
                for col in 0..14u16 {
                    let a = base + r0 + col;
                    let b = base + r0 + col + 1;
                    let c = base + r1 + col;
                    let d = base + r1 + col + 1;
                    eye120_ebo.extend_from_slice(&[a, b, c, b, d, c]);
                }
            }
        }
        for (left_pf, right_pf, mode) in [
            (8015u32, 8016u32, 0u8),
            (8017, 8018, 1),
            (8021, 8022, 2),
        ] {
            let c_left = control(left_pf);
            let c_right = control(right_pf);
            if c_left == 0.0 && c_right == 0.0 {
                continue;
            }
            let l_deg = c_left * 100.0;
            let r_deg = c_right * 100.0;
            let (src_pts, dst_pts) = match mode {
                0 => {
                    let out = eye_geometry::row14_eye_distance(
                        &points194,
                        iw,
                        ih,
                        unit_scale,
                        l_deg,
                        r_deg,
                        c_left * 12.0,
                        c_right * 12.0,
                        Some(&eye120_ebo),
                    )
                    .map_err(|_| Refusal::ShapeMismatch("eye distance geometry"))?;
                    (out.source, out.target)
                }
                1 => {
                    // Eye angle tilt around each eye's center using the 120-point eye pair
                    let base_out = eye_geometry::row16_eye_position(
                        &points194,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        Some(&eye120_ebo),
                    )
                    .map_err(|_| Refusal::ShapeMismatch("eye angle geometry"))?;
                    let mut target = base_out.source.clone();
                    for (side_idx, deg01) in [(0usize, c_left), (1, c_right)] {
                        let slice = &mut target[side_idx * 60..(side_idx + 1) * 60];
                        let cx = slice.iter().map(|p| p.x).sum::<f32>() / 60.0;
                        let cy = slice.iter().map(|p| p.y).sum::<f32>() / 60.0;
                        let angle = deg01 * 0.14 * if side_idx == 0 { -1.0 } else { 1.0 };
                        let (s, c) = angle.sin_cos();
                        for p in slice.iter_mut() {
                            let dx = p.x - cx;
                            let dy = p.y - cy;
                            p.x = cx + dx * c - dy * s;
                            p.y = cy + dx * s + dy * c;
                        }
                    }
                    (base_out.source, target)
                }
                _ => {
                    let out = eye_geometry::row16_eye_position(
                        &points194,
                        l_deg,
                        r_deg,
                        c_left * 12.0,
                        c_right * 12.0,
                        Some(&eye120_ebo),
                    )
                    .map_err(|_| Refusal::ShapeMismatch("eye position geometry"))?;
                    (out.source, out.target)
                }
            };
            let dest: Vec<[f32; 2]> = dst_pts.iter().map(|p| [p.x, p.y]).collect();
            let uv: Vec<[f32; 2]> = src_pts
                .iter()
                .map(|p| [(p.x / w).clamp(0.0, 1.0), (p.y / h).clamp(0.0, 1.0)])
                .collect();
            draws.push(super::FaceGeometryDraw {
                pf: left_pf,
                controls: [c_left, c_right],
                source_sha256: digest,
                geometry: super::FaceGeometry::Eye {
                    dest,
                    uv,
                    ebo: eye120_ebo.clone(),
                    rules,
                    edge: FragmentEdge::ReshapeNormal,
                    defines: Defines::plain(),
                    clamp: Clamp::disabled(),
                },
            });
        }

        // Brow effects: 8001/8002 (thickness), 8003/8004 (position), 8005/8006 (distance), 8007/8008 (tilt), 8009/8010 (shape)
        let mut brow_ebo = Vec::new();
        for side in 0..2u16 {
            let base = side * 84;
            for band in 0..2u16 {
                let r0 = base + band * 28;
                let r1 = r0 + 28;
                for col in 0..27u16 {
                    let a = r0 + col;
                    let b = r0 + col + 1;
                    let c = r1 + col;
                    let d = r1 + col + 1;
                    brow_ebo.extend_from_slice(&[a, b, c, b, d, c]);
                }
            }
        }
        // Build the 168-point brow prepared mesh (3 bands of 28 points per brow) and face block from points194
        let left_brow_center = points194[27];
        let right_brow_center = points194[47];
        let brow_span = distance(
            [left_brow_center.x, left_brow_center.y],
            [right_brow_center.x, right_brow_center.y],
        )
        .max(24.0);
        let brow_half_w = brow_span * 0.28;
        let brow_band_h = (brow_span * 0.08).max(4.0);
        let mut prepared168 = Vec::with_capacity(168);
        for (side_idx, center) in [left_brow_center, right_brow_center].iter().enumerate() {
            let dir = if side_idx == 0 { -1.0f32 } else { 1.0f32 };
            for band in 0..3 {
                let y_off = (band as f32 - 1.0) * brow_band_h;
                for col in 0..28 {
                    let t = col as f32 / 27.0 - 0.5;
                    let x = center.x + dir * t * 2.0 * brow_half_w;
                    let arch = (1.0 - (t * 2.0).powi(2)) * brow_band_h * 0.35;
                    let y = center.y + y_off - arch;
                    prepared168.push(Point::new(x, y));
                }
            }
        }
        const FACE_THICKNESS_X: usize = 0x1070 / 4;
        const FACE_THICKNESS_Y: usize = 0x106C / 4;
        const FACE_THICKNESS_A: usize = 0xE20 / 4;
        const FACE_THICKNESS_B: usize = 0xE1C / 4;
        const FACE_THICKNESS_C: usize = 0xAD8 / 4;
        const FACE_THICKNESS_D: usize = 0xAD4 / 4;
        const BROW_AXIS_A: usize = 0x94C / 4;
        const BROW_AXIS_B: usize = 0x99C / 4;
        let mut face_block = vec![0.0f32; BROW_AXIS_B * 2 + 4];
        let brow_dx = right_brow_center.x - left_brow_center.x;
        let brow_dy = right_brow_center.y - left_brow_center.y;
        face_block[FACE_THICKNESS_X] = -brow_dy;
        face_block[FACE_THICKNESS_Y] = brow_dx;
        face_block[FACE_THICKNESS_A] = 0.0;
        face_block[FACE_THICKNESS_B] = 0.0;
        face_block[FACE_THICKNESS_C] = 0.0;
        face_block[FACE_THICKNESS_D] = 0.0;
        face_block[BROW_AXIS_A * 2] = left_brow_center.x;
        face_block[BROW_AXIS_A * 2 + 1] = left_brow_center.y;
        face_block[BROW_AXIS_B * 2] = right_brow_center.x;
        face_block[BROW_AXIS_B * 2 + 1] = right_brow_center.y;

        let uv168: Vec<[f32; 2]> = prepared168
            .iter()
            .map(|p| [(p.x / w).clamp(0.0, 1.0), (p.y / h).clamp(0.0, 1.0)])
            .collect();

        for (left_pf, right_pf, mode) in [
            (8001u32, 8002u32, 0u8),
            (8003, 8004, 1),
            (8005, 8006, 2),
            (8007, 8008, 3),
            (8009, 8010, 4),
        ] {
            let c_left = control(left_pf);
            let c_right = control(right_pf);
            if c_left == 0.0 && c_right == 0.0 {
                continue;
            }
            let l_deg = c_left * 100.0;
            let r_deg = c_right * 100.0;
            let target = match mode {
                0 => eye_geometry::row20_brow_thickness(
                    &prepared168,
                    &prepared168,
                    &face_block,
                    l_deg,
                    r_deg,
                    Some(&brow_ebo),
                )
                .map_err(|_| Refusal::ShapeMismatch("brow thickness geometry"))?
                .target,
                1 => {
                    // Brow vertical position shift on the middle band (28..56 and 112..140)
                    let mut out = prepared168.clone();
                    for (side_idx, deg01) in [(0usize, c_left), (1, c_right)] {
                        let base = side_idx * 84 + 28;
                        let dy = -deg01 * brow_band_h * 0.45;
                        for idx in 0..28 {
                            let weight = (idx as f32 / 27.0 * std::f32::consts::PI).sin();
                            out[base + idx].y += dy * weight;
                        }
                    }
                    out
                }
                2 => eye_geometry::row21_brow_distance(
                    &prepared168,
                    &face_block,
                    l_deg,
                    r_deg,
                    Some(&brow_ebo),
                )
                .map_err(|_| Refusal::ShapeMismatch("brow distance geometry"))?
                .target,
                3 => {
                    // Brow tilt rotation around brow center on the middle band
                    let mut out = prepared168.clone();
                    for (side_idx, (center, deg01)) in
                        [(left_brow_center, c_left), (right_brow_center, c_right)]
                            .into_iter()
                            .enumerate()
                    {
                        let base = side_idx * 84 + 28;
                        let angle = deg01 * 0.14 * if side_idx == 0 { -1.0 } else { 1.0 };
                        let (s, c) = angle.sin_cos();
                        for idx in 0..28 {
                            let weight = (idx as f32 / 27.0 * std::f32::consts::PI).sin();
                            let dx = out[base + idx].x - center.x;
                            let dy = out[base + idx].y - center.y;
                            let rx = center.x + dx * c - dy * s;
                            let ry = center.y + dx * s + dy * c;
                            out[base + idx].x += (rx - out[base + idx].x) * weight;
                            out[base + idx].y += (ry - out[base + idx].y) * weight;
                        }
                    }
                    out
                }
                _ => eye_geometry::row22_brow_shape(
                    &prepared168,
                    &prepared168,
                    &face_block,
                    l_deg,
                    r_deg,
                    Some(&brow_ebo),
                )
                .map_err(|_| Refusal::ShapeMismatch("brow shape geometry"))?
                .target,
            };
            let dest: Vec<[f32; 2]> = target.iter().map(|p| [p.x, p.y]).collect();
            draws.push(super::FaceGeometryDraw {
                pf: left_pf,
                controls: [c_left, c_right],
                source_sha256: digest,
                geometry: super::FaceGeometry::Eye {
                    dest,
                    uv: uv168.clone(),
                    ebo: brow_ebo.clone(),
                    rules,
                    edge: FragmentEdge::ReshapeNormal,
                    defines: Defines::plain(),
                    clamp: Clamp::disabled(),
                },
            });
        }

        // PF30073 / PF30079 (lower eyelid Type1011, 100 prepared points)
        let c30073 = control(30073);
        let c30079 = control(30079);
        if c30073 != 0.0 || c30079 != 0.0 {
            const PF30073_PARAM_LIST: [f32; 6] = [
                0.059_999_998_7,
                0.109_999_999_4,
                0.229_999_989_3,
                0.359_999_984_5,
                0.419_999_986_9,
                0.299_999_982_1,
            ];
            let mut prepared100 = Vec::with_capacity(100);
            for side in 0..2 {
                let eye_center = if side == 0 {
                    midpoint(
                        [points194[57].x, points194[57].y],
                        [points194[63].x, points194[63].y],
                    )
                } else {
                    midpoint(
                        [points194[77].x, points194[77].y],
                        [points194[83].x, points194[83].y],
                    )
                };
                let eye_w = if side == 0 {
                    distance(
                        [points194[57].x, points194[57].y],
                        [points194[63].x, points194[63].y],
                    )
                } else {
                    distance(
                        [points194[77].x, points194[77].y],
                        [points194[83].x, points194[83].y],
                    )
                }
                .max(12.0);
                let eye_h = (eye_w * 0.38).max(5.0);
                // 50 points per side: 0..21 outer lower ring, 21..34 inner lower lid, 34..50 upper lid anchor
                for i in 0..21 {
                    let t = i as f32 / 20.0;
                    let theta = std::f32::consts::PI * (0.1 + 0.8 * t);
                    prepared100.push([
                        eye_center[0] + theta.cos() * eye_w * 0.65,
                        eye_center[1] + theta.sin() * eye_h * 1.15,
                    ]);
                }
                for i in 0..13 {
                    let t = i as f32 / 12.0;
                    let theta = std::f32::consts::PI * (0.15 + 0.7 * t);
                    prepared100.push([
                        eye_center[0] + theta.cos() * eye_w * 0.48,
                        eye_center[1] + theta.sin() * eye_h * 0.65,
                    ]);
                }
                for i in 0..16 {
                    let t = i as f32 / 15.0;
                    let theta = std::f32::consts::PI * (0.1 + 0.8 * t);
                    prepared100.push([
                        eye_center[0] + theta.cos() * eye_w * 0.55,
                        eye_center[1] - theta.sin() * eye_h * 0.75,
                    ]);
                }
            }
            let center_1011 = [points194[42].x, points194[42].y];
            draws.push(super::FaceGeometryDraw {
                pf: 30073,
                controls: [c30073, c30079],
                source_sha256: digest,
                geometry: super::FaceGeometry::Type1011 {
                    prepared: prepared100,
                    param_list: PF30073_PARAM_LIST.to_vec(),
                    center: center_1011,
                },
            });
        }

        // PF44124 / PF48019 (eyebrowHeadLength) and PF49221 / PF41922 (eyebrowTailLength) via Type1018 EBO
        let type1018_ebo = eye_geometry::type1018_ordered_ebo();
        for (left_pf, right_pf, is_head) in [(44124u32, 48019u32, true), (49221, 41922, false)] {
            let c_left = control(left_pf);
            let c_right = control(right_pf);
            if c_left == 0.0 && c_right == 0.0 {
                continue;
            }
            let (left_anchor, right_anchor, dir_l, dir_r) = if is_head {
                (points194[24], points194[44], [1.0f32, 0.0], [-1.0f32, 0.0])
            } else {
                (points194[31], points194[51], [-1.0f32, 0.0], [1.0f32, 0.0])
            };
            let radius = (brow_span * 0.14).max(8.0);
            let mut src_102 = Vec::with_capacity(102);
            let mut dst_102 = Vec::with_capacity(102);
            for _ in 0..6 {
                src_102.push([left_anchor.x, left_anchor.y]);
                dst_102.push([left_anchor.x, left_anchor.y]);
            }
            // Four 24-slot rings split between left (0..12) and right (12..24) brow tips
            for ring in 0..4 {
                let ring_scale = 0.25 * (ring + 1) as f32;
                let weight = 1.0 - ring as f32 * 0.25;
                for slot in 0..24 {
                    let (anchor, dir, deg) = if slot < 12 {
                        (left_anchor, dir_l, c_left)
                    } else {
                        (right_anchor, dir_r, c_right)
                    };
                    let theta = (slot % 12) as f32 / 12.0 * std::f32::consts::TAU;
                    let sx = anchor.x + theta.cos() * radius * ring_scale;
                    let sy = anchor.y + theta.sin() * radius * 0.7 * ring_scale;
                    let shift = deg * radius * 0.35 * weight;
                    src_102.push([(sx / w).clamp(0.0, 1.0), (sy / h).clamp(0.0, 1.0)]);
                    dst_102.push([sx + dir[0] * shift, sy + dir[1] * shift]);
                }
            }
            draws.push(super::FaceGeometryDraw {
                pf: left_pf,
                controls: [c_left, c_right],
                source_sha256: digest,
                geometry: super::FaceGeometry::Eye {
                    dest: dst_102,
                    uv: src_102,
                    ebo: type1018_ebo.clone(),
                    rules,
                    edge: FragmentEdge::ReshapeNormal,
                    defines: Defines::plain(),
                    clamp: Clamp::disabled(),
                },
            });
        }

        if !draws.is_empty() {
            let inputs = super::FaceGeometryInputs {
                job_id: self.job_id,
                width: source.width,
                height: source.height,
                draws,
            };
            self.attach_face_geometry(source, inputs, producer)?;
        }
        Ok(())
    }

    /// Builds same-job wrinkle and historical-skin wrinkle inputs (`PF21024, 21025, 21069, 21070, 21071, 90023, 90024`)
    /// from the source frame's frequency-separated Gaussian/median base and `fbe.onnx` class-1 skin mask.
    pub fn attach_wrinkle_from_extra(
        &self,
        source: &Image,
        extra: &std::collections::BTreeMap<String, f32>,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        use crate::wrinkle_legacy::{
            WrinkleFaceRoiContract, WrinkleLegacySkinMaskProducer, WrinkleLegacySkinMaskTexture,
            WrinkleMaskMiniImage, WrinkleMaskMiniImageProducer, WrinkleProducerInputs,
            WrinkleSameJobReceipt, WRINKLE_MASK_ROI_SIDE,
        };
        use aurora_portrait::kernels::skin_adjust::{Roi, WrinkleUniforms};

        self.validate_ticket(source, Some(self.job_id))?;
        let control = |pf: u32| extra.get(&format!("pc:{pf}")).copied().unwrap_or(0.0);
        let active_routed = [21025u32, 21069, 21070, 21071]
            .into_iter()
            .any(|pf| control(pf) > 0.0);
        let active_hist = [21024u32, 90023, 90024]
            .into_iter()
            .any(|pf| control(pf) > 0.001);
        if !active_routed && !active_hist {
            return Ok(());
        }
        let inputs = self
            .model_inputs
            .as_ref()
            .ok_or(Refusal::MissingInput(FACE_PARSING_V2_PRODUCE_ERROR))?;
        let mut parsing = inputs
            .parsing
            .clone()
            .ok_or(Refusal::MissingInput("FaceParsingV2 parsing planes"))?;
        parsing.job_id = self.job_id;
        let skin_mask = crate::face_parsing::project_face_parsing_skin_mask(
            &parsing,
            source.width,
            source.height,
        )?;

        let pixels = source.pixel_count();
        let mut src_rgb = Vec::with_capacity(pixels * 3);
        for px in source.pixels.chunks_exact(4) {
            src_rgb.push(px[0] as f32 / 255.0);
            src_rgb.push(px[1] as f32 / 255.0);
            src_rgb.push(px[2] as f32 / 255.0);
        }
        let (median_rgb, gauss_rgb, _, _) = super::hlf_preprocess(
            &src_rgb,
            source.width,
            source.height,
            super::HlfAutoProducerOptions::default(),
        );
        let mut base_rgba = Vec::with_capacity(pixels * 4);
        let mut residual_rgba = Vec::with_capacity(pixels * 4);
        for i in 0..pixels {
            for c in 0..3 {
                base_rgba.push((median_rgb[i * 3 + c].clamp(0.0, 1.0) * 255.0).round() as u8);
                residual_rgba.push((gauss_rgb[i * 3 + c].clamp(0.0, 1.0) * 255.0).round() as u8);
            }
            base_rgba.push(source.pixels[i * 4 + 3]);
            residual_rgba.push(source.pixels[i * 4 + 3]);
        }
        let base_img = Image::new(source.width, source.height, base_rgba)
            .ok_or(Refusal::ShapeMismatch("wrinkle base image"))?;
        let residual_img = Image::new(source.width, source.height, residual_rgba)
            .ok_or(Refusal::ShapeMismatch("wrinkle residual image"))?;

        // Compute bounding union of roi_xywh and landmarks_194 + landmarks_56
        let landmarks_56: Vec<[f32; 2]> = self.points194.iter().take(56).copied().collect();
        let mut min_x = self.roi_xywh[0].max(0.0);
        let mut min_y = self.roi_xywh[1].max(0.0);
        let mut max_x = (self.roi_xywh[0] + self.roi_xywh[2]).min(source.width as f32);
        let mut max_y = (self.roi_xywh[1] + self.roi_xywh[3]).min(source.height as f32);
        for pt in self.points194.iter().chain(&landmarks_56) {
            min_x = min_x.min(pt[0]).max(0.0);
            min_y = min_y.min(pt[1]).max(0.0);
            max_x = max_x.max(pt[0]).min(source.width as f32);
            max_y = max_y.max(pt[1]).min(source.height as f32);
        }
        let union_rect = [min_x, min_y, (max_x - min_x).max(2.0), (max_y - min_y).max(2.0)];
        let face_roi = WrinkleFaceRoiContract::from_face_record(
            1,
            true,
            source.width,
            source.height,
            union_rect,
            self.detection.score.clamp(0.0, 1.0),
            self.points194.clone(),
            landmarks_56,
        )?;
        let stage_roi: Roi = face_roi.stage_roi();
        let mut mask_rgba = vec![0u8; pixels * 4];
        for (i, &s) in skin_mask.iter().enumerate() {
            mask_rgba[i * 4] = s;
            mask_rgba[i * 4 + 1] = s;
            mask_rgba[i * 4 + 2] = s;
            mask_rgba[i * 4 + 3] = 255;
        }
        let uniforms = WrinkleUniforms {
            legal_alpha: 0.55,
            dark_circle_alpha: 0.55,
            diff_factor: 0.75,
            diff_factor2: 0.0,
            color_threshold_factor: 0.1,
            mix_alpha: 0.9,
            diff_min_threshold: -0.25,
            diff_max_threshold: 0.25,
            high_freq_min_threshold: -0.25,
            high_freq_max_threshold: 0.25,
        };

        if active_routed && producer.wrinkle.is_none() {
            let legacy_tex =
                WrinkleLegacySkinMaskTexture::from_service_6200(1, 2, 2, vec![255; 16])?;
            let legacy_prod = WrinkleLegacySkinMaskProducer::from_same_job(
                self.job_id,
                source.width,
                source.height,
                legacy_tex,
                skin_mask.clone(),
            )?;
            let mini_img = WrinkleMaskMiniImage::from_capture(
                1,
                vec![255; (WRINKLE_MASK_ROI_SIDE * WRINKLE_MASK_ROI_SIDE * 4) as usize],
            )?;
            let mini_prod = WrinkleMaskMiniImageProducer::from_same_job(
                self.job_id,
                source.width,
                source.height,
                mini_img,
                face_roi,
                mask_rgba.clone(),
            )?;
            let wrinkle_inputs = WrinkleProducerInputs::from_same_job(
                WrinkleSameJobReceipt::from_job_id(self.job_id),
                source.width,
                source.height,
                base_img.clone(),
                residual_img.clone(),
                legacy_prod,
                mini_prod,
                uniforms,
            )?;
            producer.write_wrinkle(wrinkle_inputs);
        }

        if active_hist {
            let wrinkle_payload = super::HistoricalSkinWrinkle {
                base: base_img,
                residual: residual_img,
                mask_rgba,
                skin_mask,
                roi: stage_roi,
                uniforms,
            };
            let mut rows = Vec::new();
            for (pf, variant) in [
                (
                    21024u32,
                    super::HistoricalSkinData::Glabella(wrinkle_payload.clone()),
                ),
                (
                    90023,
                    super::HistoricalSkinData::EyeLines(wrinkle_payload.clone()),
                ),
                (90024, super::HistoricalSkinData::Forehead(wrinkle_payload)),
            ] {
                let c = control(pf);
                if c > 0.001 {
                    rows.push(super::HistoricalSkinImmediateRow {
                        pf,
                        ticket: self.job_id,
                        control: c,
                        data: variant,
                    });
                }
            }
            if !rows.is_empty() {
                producer.write_immediate_historical_skin(
                    source,
                    &parsing,
                    super::HistoricalSkinImmediateInputs {
                        ticket: self.job_id,
                        rows,
                    },
                )?;
            }
        }
        Ok(())
    }
}

/// PF8013/PF8014 (`row12_eye_height`) only emits when both eye quads exist. When one side has no
/// control, feed that side a tiny nonzero degree so the pair-validation path sees two quads; its
/// UV warp stays identity because `eye_pair_deg01` uses the real (zero) control downstream.
const EYE_PAIR_SINGLE_SIDE_EPSILON_DEG: f32 = 1e-4;

/// Args passed to row11/row12/row13 (`c * 100` degrees). For mode 1 only, an inactive side
/// (control exactly 0.0 while the other side is nonzero) is substituted with an epsilon degree.
fn eye_pair_args(mode: u8, c_left: f32, c_right: f32) -> (f32, f32) {
    let l_deg = c_left * 100.0;
    let r_deg = c_right * 100.0;
    if mode == 1 && (c_left == 0.0) != (c_right == 0.0) {
        return if c_left == 0.0 { (EYE_PAIR_SINGLE_SIDE_EPSILON_DEG, r_deg) } else { (l_deg, EYE_PAIR_SINGLE_SIDE_EPSILON_DEG) };
    }
    (l_deg, r_deg)
}

/// Selects the real control for a single emitted quad: with one quad it belongs to whichever side
/// has nonzero control (row11/row13 skip zero-degree sides), regardless of `side_idx`.
fn eye_pair_deg01(quad_count: usize, side_idx: usize, c_left: f32, c_right: f32) -> f32 {
    if quad_count == 1 {
        return if c_left != 0.0 { c_left } else { c_right };
    }
    if side_idx == 0 { c_left } else { c_right }
}

pub const FACE_STAGE_TRANSFORM_REFUSAL: &str = FACE_STAGE_TRANSFORM_ERROR;

pub struct FaceStageTransform {
    source_size: [u32; 2],
    stage_size: [u32; 2],
    region: [f32; 4],
    map: crate::mock::geometry::FrameMap,
}

impl FaceStageTransform {
    pub fn new(
        source_size: [u32; 2],
        stage_size: [u32; 2],
        region: [f32; 4],
        geometry: &crate::Geometry,
    ) -> Result<Self, Refusal> {
        if source_size[0] == 0
            || source_size[1] == 0
            || stage_size[0] == 0
            || stage_size[1] == 0
            || region.iter().any(|value| !value.is_finite())
            || region[2] <= 0.0
            || region[3] <= 0.0
        {
            return Err(Refusal::ValueOutOfRange(FACE_STAGE_TRANSFORM_ERROR));
        }
        Ok(Self {
            source_size,
            stage_size,
            region,
            map: crate::mock::geometry::FrameMap::new(
                (source_size[0], source_size[1]),
                geometry,
            ),
        })
    }

    pub fn source_point_to_stage(&self, point: [f32; 2]) -> Result<[f32; 2], Refusal> {
        if point.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange(FACE_STAGE_TRANSFORM_ERROR));
        }
        let normalized = [
            point[0] / self.source_size[0] as f32,
            point[1] / self.source_size[1] as f32,
        ];
        let frame = self
            .map
            .frame_from_source(normalized)
            .ok_or(Refusal::ValueOutOfRange(FACE_STAGE_TRANSFORM_ERROR))?;
        let stage = [
            (frame[0] - self.region[0]) / self.region[2] * self.stage_size[0] as f32 - 0.5,
            (frame[1] - self.region[1]) / self.region[3] * self.stage_size[1] as f32 - 0.5,
        ];
        if stage.iter().enumerate().any(|(axis, value)| {
            !value.is_finite()
                || *value < -0.5
                || *value > self.stage_size[axis] as f32 + 0.5
        }) {
            return Err(Refusal::ValueOutOfRange(FACE_STAGE_TRANSFORM_ERROR));
        }
        Ok(stage)
    }

    pub fn source_roi_to_stage(&self, roi: [f32; 4]) -> Result<[f32; 4], Refusal> {
        if roi.iter().any(|value| !value.is_finite()) || roi[2] <= 0.0 || roi[3] <= 0.0 {
            return Err(Refusal::ValueOutOfRange(FACE_STAGE_TRANSFORM_ERROR));
        }
        let corners = [
            [roi[0], roi[1]],
            [roi[0] + roi[2], roi[1]],
            [roi[0], roi[1] + roi[3]],
            [roi[0] + roi[2], roi[1] + roi[3]],
        ];
        let mapped = corners
            .iter()
            .map(|point| self.source_point_to_stage(*point))
            .collect::<Result<Vec<_>, _>>()?;
        let mut low = [f32::INFINITY; 2];
        let mut high = [f32::NEG_INFINITY; 2];
        for point in mapped {
            low[0] = low[0].min(point[0]);
            low[1] = low[1].min(point[1]);
            high[0] = high[0].max(point[0]);
            high[1] = high[1].max(point[1]);
        }
        Ok([low[0], low[1], high[0] - low[0], high[1] - low[1]])
    }
}

pub struct FaceInputProducer {
    runtime: &'static Runtime,
    detection: Session,
    mesh: Session,
    coarse: Session,
    dense: Session,
    parsing: Option<FaceParsingProducer>,
    paths: FaceInputModelPaths,
    constants: LandmarkConstants,
}

impl FaceInputProducer {
    pub fn open(paths: FaceInputModelPaths, include_parsing: bool) -> Result<Self, Refusal> {
        Self::open_with_backend(paths, include_parsing, FaceInputBackend::from_env()?)
    }

    fn open_with_backend(
        paths: FaceInputModelPaths,
        include_parsing: bool,
        backend: FaceInputBackend,
    ) -> Result<Self, Refusal> {
        verify_file(&paths.detection, C1_FACE_DETECTION_MODEL_SHA256)?;
        verify_file(&paths.mesh, C1_FACE_MESH_MODEL_SHA256)?;
        verify_file(&paths.coarse, FACE_LANDMARK_1063025060_MODEL_SHA256)?;
        verify_file(&paths.dense, FACE_LANDMARK_323773030_MODEL_SHA256)?;
        verify_file(&paths.extra52, super::FACE_LANDMARK_52_MODEL_SHA256)?;
        verify_file(&paths.landmark_constants, FACE_LANDMARK_CONSTANTS_SHA256)?;
        let constants = LandmarkConstants::load(&paths.landmark_constants)?;
        let runtime = Runtime::get().ok_or(Refusal::MissingInput("ONNX Runtime"))?;
        let detection = backend.open(runtime, &paths.detection)?;
        validate_detection_session(&detection)?;
        let mesh = backend.open(runtime, &paths.mesh)?;
        validate_fixed_session(
            &mesh,
            [1, 3, MESH_SIDE as i64, MESH_SIDE as i64],
            &[
                ("Identity", &[1, 1, 1, 1434]),
                ("Identity_1", &[1, 1, 1, 1]),
                ("Identity_2", &[1, 1]),
            ],
        )?;
        log::info!(
            "C1 face backend {}: detection={} mesh={}",
            backend.label(),
            detection.provider().label(),
            mesh.provider().label()
        );
        let coarse = runtime
            .open_cpu_session(&paths.coarse)
            .map_err(|_| Refusal::MissingInput(MODEL_RUNTIME_ERROR))?;
        validate_fixed_session(
            &coarse,
            [1, 3, COARSE_SIDE as i64, COARSE_SIDE as i64],
            &[("2297057593", &[1, 34, 1, 1])],
        )?;
        let dense = runtime
            .open_cpu_session(&paths.dense)
            .map_err(|_| Refusal::MissingInput(MODEL_RUNTIME_ERROR))?;
        validate_fixed_session(
            &dense,
            [1, 3, DENSE_SIDE as i64, DENSE_SIDE as i64],
            &[
                ("2297057593", &[1, 142, 2]),
                ("2297057596", &[1, 2]),
                ("781171859", &[1, 142, 1, 1]),
            ],
        )?;
        let parsing = if include_parsing {
            Some(FaceParsingProducer::load(&paths.parsing).map_err(|error| {
                log::warn!("FaceParsingV2 canonical model refused: {error:#}");
                Refusal::MissingInput(FACE_PARSING_V2_MODEL_ERROR)
            })?)
        } else {
            None
        };

        Ok(Self {
            runtime,
            detection,
            mesh,
            coarse,
            dense,
            parsing,
            paths,
            constants,
        })
    }

    pub fn produce(&self, job_id: u64, source: &Image) -> Result<FaceInputBundle, Refusal> {
        if job_id == 0 {
            return Err(Refusal::ValueOutOfRange("face input job ticket"));
        }
        source
            .validate()
            .map_err(|_| Refusal::ShapeMismatch("face input source image"))?;
        let detection = self.detect(source)?;
        let (mesh, roi_xywh) = self.mesh_roi(source, &detection)?;
        let points17 = self.points17(source, roi_xywh)?;
        let points142 = self.points142(source, &points17)?;
        let landmark_producer =
            FaceLandmarkProducerInputs::produce(&self.paths.extra52, source, points142.clone())
                .map_err(|_| Refusal::MissingInput("142->52 face landmark model"))?;
        let points194 = landmark_producer.points194.clone();
        let points414 = face_width_414::produce(&points194, 0.0)
            .map_err(|error| match error {
                face_width_414::Error::Points194 => {
                    Refusal::MissingInput("face 194 landmarks for 414 producer")
                }
                face_width_414::Error::NonFinite => Refusal::ValueOutOfRange("face 414 landmarks"),
                face_width_414::Error::UnsupportedScale => {
                    Refusal::ValueOutOfRange("face 414 producer scale")
                }
            })?
            .to_vec();
        let source_width = source.width as f32;
        let source_height = source.height as f32;
        let roi_normalized_xywh = [
            roi_xywh[0] / source_width,
            roi_xywh[1] / source_height,
            roi_xywh[2] / source_width,
            roi_xywh[3] / source_height,
        ];
        let model_inputs = self
            .parsing
            .as_ref()
            .map(|parsing| {
                let roi = Self::parsing_roi(&detection, roi_xywh)?;
                let source_to_crop = roi
                    .source_to_crop()
                    .map_err(|_| Refusal::MissingInput(FACE_PARSING_V2_ROI_ERROR))?;
                parsing
                    .produce(source, job_id, source_to_crop, 0.0, 0.0)
                    .map_err(|error| {
                        log::warn!("FaceParsingV2 canonical producer refused: {error:#}");
                        Refusal::MissingInput(FACE_PARSING_V2_PRODUCE_ERROR)
                    })
            })
            .transpose()?;
        let bundle = FaceInputBundle {
            job_id,
            source_size: [source.width, source.height],
            source_sha256: sha256_32(&source.pixels),
            detection,
            mesh,
            roi_xywh,
            roi_normalized_xywh,
            points17,
            points142,
            points194,
            points414,
            landmark_producer,
            model_inputs,
        };
        bundle.validate_ticket(source, Some(job_id))?;
        Ok(bundle)
    }

    fn detect(&self, source: &Image) -> Result<C1FaceDetection, Refusal> {
        let width = source.width as usize;
        let height = source.height as usize;
        if width == 0 || height == 0 {
            return Err(Refusal::MissingInput(FACE_LANDMARK_ERROR));
        }
        let scale = (DETECTION_SIDE as f32 / width.max(height) as f32).min(1.0);
        let resized_width = ((width as f32 * scale) + 0.5) as usize;
        let resized_height = ((height as f32 * scale) + 0.5) as usize;
        if resized_width == 0 || resized_height == 0 {
            return Err(Refusal::ValueOutOfRange("C1 face detection input size"));
        }
        let left = (DETECTION_SIDE - resized_width) / 2;
        let top = (DETECTION_SIDE - resized_height) / 2;
        let resized = resize_rgb(source, resized_width, resized_height);
        let mut canvas = vec![114u8; DETECTION_SIDE * DETECTION_SIDE * 3];
        for y in 0..resized_height {
            let source_offset = y * resized_width * 3;
            let target_offset = ((top + y) * DETECTION_SIDE + left) * 3;
            canvas[target_offset..target_offset + resized_width * 3]
                .copy_from_slice(&resized[source_offset..source_offset + resized_width * 3]);
        }
        let mut input = vec![0.0f32; 3 * DETECTION_SIDE * DETECTION_SIDE];
        for y in 0..DETECTION_SIDE {
            for x in 0..DETECTION_SIDE {
                let offset = (y * DETECTION_SIDE + x) * 3;
                for channel in 0..3 {
                    input[(2 - channel) * DETECTION_SIDE * DETECTION_SIDE
                        + y * DETECTION_SIDE
                        + x] = (canvas[offset + channel] as f32 - 127.5) / 128.0;
                }
            }
        }
        let outputs = self
            .runtime
            .run(
                &self.detection,
                &mut input,
                &[1, 3, DETECTION_SIDE as i64, DETECTION_SIDE as i64],
            )
            .map_err(|_| Refusal::MissingInput(MODEL_RUNTIME_ERROR))?;
        let anchors = 2
            * ((DETECTION_SIDE / 8).pow(2)
                + (DETECTION_SIDE / 16).pow(2)
                + (DETECTION_SIDE / 32).pow(2));
        if outputs.len() != 3
            || outputs[0].len() != anchors * 4
            || outputs[1].len() != anchors * 2
            || outputs[2].len() != anchors * 10
        {
            return Err(Refusal::ShapeMismatch(MODEL_OUTPUT_ERROR));
        }
        let mut best = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for index in 0..anchors {
            let score = outputs[1][index * 2 + 1];
            if score.is_finite() && score > best_score {
                best = index;
                best_score = score;
            }
        }
        if best_score < DETECTION_SCORE_MIN {
            return Err(Refusal::MissingInput("C1 face detection score"));
        }
        let (stride, grid, cell) = anchor_layout(best);
        let anchor = [
            (cell % grid) as f32 * stride as f32,
            (cell / grid) as f32 * stride as f32,
        ];
        let source_to_input = [[scale, 0.0, left as f32], [0.0, scale, top as f32]];
        let input_to_source = inverse(source_to_input)?;
        let mut landmarks = [[0.0f32; 2]; 5];
        for (index, point) in landmarks.iter_mut().enumerate() {
            let offset = best * 10 + index * 2;
            *point = decode_detection_point(
                [outputs[2][offset], outputs[2][offset + 1]],
                anchor,
                stride as f32,
                input_to_source,
            );
            if point.iter().any(|value| !value.is_finite()) {
                return Err(Refusal::ValueOutOfRange(FACE_LANDMARK_ERROR));
            }
        }
        let eye_width = distance(landmarks[0], landmarks[1]);
        if !eye_width.is_finite() || eye_width < 2.0 {
            return Err(Refusal::MissingInput(FACE_LANDMARK_ERROR));
        }
        Ok(C1FaceDetection {
            score: best_score,
            landmarks_source: landmarks,
            source_to_input,
            input_size: [DETECTION_SIDE as u32, DETECTION_SIDE as u32],
            normalization: FaceNormalization {
                channel_order: "BGR",
                mean: [127.5; 3],
                scale: [128.0; 3],
            },
            provider: self.detection.provider().label(),
            model: FaceModelIdentity {
                model_id: C1_FACE_DETECTION_MODEL_ID,
                sha256: C1_FACE_DETECTION_MODEL_SHA256,
            },
        })
    }

    fn mesh_roi(
        &self,
        source: &Image,
        detection: &C1FaceDetection,
    ) -> Result<(C1FaceMesh, [f32; 4]), Refusal> {
        let eye = midpoint(detection.landmarks_source[0], detection.landmarks_source[1]);
        let mouth = midpoint(detection.landmarks_source[3], detection.landmarks_source[4]);
        let center = [
            eye[0] * 0.35 + mouth[0] * 0.65,
            eye[1] * 0.35 + mouth[1] * 0.65,
        ];
        let eye_distance = distance(detection.landmarks_source[0], detection.landmarks_source[1]);
        let side = (eye_distance * 3.4).max(16.0);
        let origin_x = center[0] - side * 0.5;
        let origin_y = center[1] - side * 0.5;
        let scale = MESH_SIDE as f32 / side;
        let source_to_crop = [
            [scale, 0.0, -origin_x * scale],
            [0.0, scale, -origin_y * scale],
        ];
        let crop = warp_reflect_rgb(source, source_to_crop, MESH_SIDE);
        let mut input = vec![0.0f32; 3 * MESH_SIDE * MESH_SIDE];
        for y in 0..MESH_SIDE {
            for x in 0..MESH_SIDE {
                let offset = (y * MESH_SIDE + x) * 3;
                for channel in 0..3 {
                    input[(2 - channel) * MESH_SIDE * MESH_SIDE + y * MESH_SIDE + x] =
                        crop[offset + channel] as f32 / 127.5 - 1.0;
                }
            }
        }
        let outputs = self
            .runtime
            .run(
                &self.mesh,
                &mut input,
                &[1, 3, MESH_SIDE as i64, MESH_SIDE as i64],
            )
            .map_err(|_| Refusal::MissingInput(MODEL_RUNTIME_ERROR))?;
        if outputs.len() != 3
            || outputs[0].len() != MESH_POINTS * 3
            || outputs[1].len() != 1
            || outputs[2].len() != 1
        {
            return Err(Refusal::ShapeMismatch(MODEL_OUTPUT_ERROR));
        }
        let score = 1.0 - outputs[2][0];
        if !score.is_finite() || score < MESH_SCORE_MIN {
            return Err(Refusal::MissingInput(FACE_MESH_SCORE_ERROR));
        }
        let inverse = inverse(source_to_crop)?;
        let mut landmarks_source = Vec::with_capacity(MESH_POINTS);
        let mut low = [f32::INFINITY; 2];
        let mut high = [f32::NEG_INFINITY; 2];
        for index in 0..MESH_POINTS {
            let crop_x = outputs[0][index * 3];
            let crop_y = outputs[0][index * 3 + 1];
            let depth = outputs[0][index * 3 + 2];
            if !crop_x.is_finite()
                || !crop_y.is_finite()
                || !depth.is_finite()
                || !(-(MESH_SIDE as f32)..=2.0 * MESH_SIDE as f32).contains(&crop_x)
                || !(-(MESH_SIDE as f32)..=2.0 * MESH_SIDE as f32).contains(&crop_y)
            {
                return Err(Refusal::ValueOutOfRange(FACE_LANDMARK_ERROR));
            }
            let source_x = crop_x * inverse[0][0] + crop_y * inverse[0][1] + inverse[0][2];
            let source_y = crop_x * inverse[1][0] + crop_y * inverse[1][1] + inverse[1][2];
            landmarks_source.push([source_x, source_y, depth]);
            low[0] = low[0].min(source_x);
            low[1] = low[1].min(source_y);
            high[0] = high[0].max(source_x);
            high[1] = high[1].max(source_y);
        }
        let extent = [high[0] - low[0], high[1] - low[1]];
        if extent
            .iter()
            .any(|value| !value.is_finite() || *value <= 1.0)
        {
            return Err(Refusal::MissingInput("C1 face ROI"));
        }
        let padded = [extent[0] * 0.18, extent[1] * 0.18];
        let x0 = (low[0] - padded[0]).max(0.0);
        let y0 = (low[1] - padded[1]).max(0.0);
        let x1 = (high[0] + padded[0]).min(source.width as f32);
        let y1 = (high[1] + padded[1]).min(source.height as f32);
        if x1 - x0 < 8.0 || y1 - y0 < 8.0 {
            return Err(Refusal::MissingInput("C1 face ROI"));
        }
        let roi = [x0, y0, x1 - x0, y1 - y0];
        Ok((
            C1FaceMesh {
                score,
                landmarks_source,
                source_to_crop,
                crop_side: side,
                input_size: [MESH_SIDE as u32, MESH_SIDE as u32],
                normalization: FaceNormalization {
                    channel_order: "BGR",
                    mean: [127.5; 3],
                    scale: [127.5; 3],
                },
                provider: self.mesh.provider().label(),
                model: FaceModelIdentity {
                    model_id: C1_FACE_MESH_MODEL_ID,
                    sha256: C1_FACE_MESH_MODEL_SHA256,
                },
            },
            roi,
        ))
    }

    fn points17(&self, source: &Image, roi: [f32; 4]) -> Result<Vec<[f32; 2]>, Refusal> {
        let (crop_matrix, inverse_matrix) = crop17(roi, &self.constants.margin)?;
        let mut input = warp_affine_bgr(source, crop_matrix, COARSE_SIDE)?;
        let outputs = self
            .runtime
            .run(
                &self.coarse,
                &mut input,
                &[1, 3, COARSE_SIDE as i64, COARSE_SIDE as i64],
            )
            .map_err(|_| Refusal::MissingInput(MODEL_RUNTIME_ERROR))?;
        if outputs.len() != 1 || outputs[0].len() != 34 {
            return Err(Refusal::ShapeMismatch(MODEL_OUTPUT_ERROR));
        }
        let mut points = Vec::with_capacity(17);
        for raw in outputs[0].chunks_exact(2) {
            if !raw[0].is_finite()
                || !raw[1].is_finite()
                || !(-0.25..=1.25).contains(&raw[0])
                || !(-0.25..=1.25).contains(&raw[1])
            {
                return Err(Refusal::ValueOutOfRange(FACE_LANDMARK_ERROR));
            }
            points.push(map_point(
                [raw[0] * COARSE_SIDE as f32, raw[1] * COARSE_SIDE as f32],
                inverse_matrix,
            ));
        }
        validate_source_points(&points, source, "C1 face 17 landmarks")?;
        Ok(points)
    }

    fn points142(&self, source: &Image, points17: &[[f32; 2]]) -> Result<Vec<[f32; 2]>, Refusal> {
        let template = self.constants.scaled_template();
        let selected = [0usize, 1, 2, 3, 4, 9, 10, 11, 12];
        let source_points = selected
            .iter()
            .map(|&index| points17[index])
            .collect::<Vec<_>>();
        let target_points = selected
            .iter()
            .map(|&index| template[self.constants.indices17[index] as usize])
            .collect::<Vec<_>>();
        let fit = fit_similarity(&source_points, &target_points)?;
        let crop_matrix = half(fit);
        let mut input = warp_affine_bgr(source, crop_matrix, DENSE_SIDE)?;
        let outputs = self
            .runtime
            .run(
                &self.dense,
                &mut input,
                &[1, 3, DENSE_SIDE as i64, DENSE_SIDE as i64],
            )
            .map_err(|_| Refusal::MissingInput(MODEL_RUNTIME_ERROR))?;
        if outputs.len() != 3
            || outputs[0].len() != 142 * 2
            || outputs[1].len() != 2
            || outputs[2].len() != 142
        {
            return Err(Refusal::ShapeMismatch(MODEL_OUTPUT_ERROR));
        }
        let score = outputs[1][1];
        if !score.is_finite() {
            return Err(Refusal::MissingInput("C1 face 142 landmark score"));
        }
        let inverse_matrix = inverse(fit)?;
        let mut points = Vec::with_capacity(142);
        for raw in outputs[0].chunks_exact(2) {
            if raw.iter().any(|value| !value.is_finite()) {
                return Err(Refusal::ValueOutOfRange(FACE_LANDMARK_ERROR));
            }
            points.push(map_point(
                [raw[0] * DENSE_SIDE as f32, raw[1] * DENSE_SIDE as f32],
                inverse_matrix,
            ));
        }
        validate_source_points(&points, source, "C1 face 142 landmarks")?;
        Ok(points)
    }

    fn parsing_roi(detection: &C1FaceDetection, roi: [f32; 4]) -> Result<FaceParsingRoi, Refusal> {
        if roi.iter().any(|value| !value.is_finite()) || roi[2] <= 0.0 || roi[3] <= 0.0 {
            return Err(Refusal::MissingInput(FACE_PARSING_V2_ROI_ERROR));
        }
        let eye0 = detection.landmarks_source[0];
        let eye1 = detection.landmarks_source[1];
        let dx = eye1[0] - eye0[0];
        let dy = eye1[1] - eye0[1];
        if dx.hypot(dy) < 2.0 {
            return Err(Refusal::MissingInput(FACE_PARSING_V2_ROI_ERROR));
        }
        let mut roll_radians = dy.atan2(dx);
        if roll_radians > std::f32::consts::FRAC_PI_2 {
            roll_radians -= std::f32::consts::PI;
        } else if roll_radians < -std::f32::consts::FRAC_PI_2 {
            roll_radians += std::f32::consts::PI;
        }
        if !roll_radians.is_finite() {
            return Err(Refusal::MissingInput(FACE_PARSING_V2_ROI_ERROR));
        }
        Ok(FaceParsingRoi {
            x: roi[0],
            y: roi[1],
            width: roi[2],
            height: roi[3],
            roll_radians,
        })
    }
}

const MAX_CACHED_FACE_PRODUCERS: usize = 8;
const MAX_CACHED_FACE_RESULTS: usize = 32;
const ROI_CACHE_SCALE: f64 = 1_000_000.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FaceInputFileStamp {
    pub bytes: u64,
    pub modified_nanos: u128,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FaceInputSessionKey {
    pub paths: FaceInputModelPaths,
    pub include_parsing: bool,
    backend: FaceInputBackend,
    pub stamps: Vec<FaceInputFileStamp>,
}

impl FaceInputSessionKey {
    fn from_paths(paths: FaceInputModelPaths, include_parsing: bool) -> Result<Self, Refusal> {
        let backend = FaceInputBackend::from_env()?;
        let paths = FaceInputModelPaths {
            detection: canonical_model_path(&paths.detection)?,
            mesh: canonical_model_path(&paths.mesh)?,
            parsing: if include_parsing {
                canonical_model_path(&paths.parsing)?
            } else {
                paths.parsing
            },
            coarse: canonical_model_path(&paths.coarse)?,
            dense: canonical_model_path(&paths.dense)?,
            extra52: canonical_model_path(&paths.extra52)?,
            landmark_constants: canonical_model_path(&paths.landmark_constants)?,
        };
        let mut stamps = Vec::with_capacity(if include_parsing { 7 } else { 6 });
        for path in [
            &paths.detection,
            &paths.mesh,
            &paths.coarse,
            &paths.dense,
            &paths.extra52,
            &paths.landmark_constants,
        ] {
            stamps.push(file_stamp(path)?);
        }
        if include_parsing {
            stamps.push(file_stamp(&paths.parsing)?);
        }
        Ok(Self {
            paths,
            include_parsing,
            backend,
            stamps,
        })
    }

    pub fn backend_label(&self) -> &'static str {
        self.backend.label()
    }

    pub fn fingerprint(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FaceInputResultKey {
    pub session_fingerprint: u64,
    pub source_sha256: [u8; 32],
    pub source_size: [u32; 2],
    pub roi_quantized: [i64; 4],
}

impl FaceInputResultKey {
    fn from_bundle(
        session: &FaceInputSessionKey,
        bundle: &FaceInputBundle,
    ) -> Result<Self, Refusal> {
        let roi_quantized = quantize_roi(bundle.roi_normalized_xywh)?;
        Ok(Self {
            session_fingerprint: session.fingerprint(),
            source_sha256: bundle.source_sha256,
            source_size: bundle.source_size,
            roi_quantized,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct FaceInputAliasKey {
    session_fingerprint: u64,
    source_sha256: [u8; 32],
    source_size: [u32; 2],
}

#[derive(Default)]
struct FaceInputResultCache {
    results: HashMap<FaceInputResultKey, Arc<FaceInputBundle>>,
    aliases: HashMap<FaceInputAliasKey, FaceInputResultKey>,
}

fn producer_cache() -> &'static Mutex<HashMap<FaceInputSessionKey, Arc<FaceInputProducer>>> {
    static CACHE: OnceLock<Mutex<HashMap<FaceInputSessionKey, Arc<FaceInputProducer>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn result_cache() -> &'static Mutex<FaceInputResultCache> {
    static CACHE: OnceLock<Mutex<FaceInputResultCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(FaceInputResultCache::default()))
}

/// Process-local, thread-safe owner for the five ORT sessions and successful
/// face-input bundles.
pub struct FaceInputSession {
    key: FaceInputSessionKey,
    producer: Arc<FaceInputProducer>,
}

impl FaceInputSession {
    pub fn open_from_env(include_parsing: bool) -> Result<Self, Refusal> {
        Self::open(
            FaceInputModelPaths::from_environment(include_parsing)?,
            include_parsing,
        )
    }

    pub fn open(paths: FaceInputModelPaths, include_parsing: bool) -> Result<Self, Refusal> {
        let key = FaceInputSessionKey::from_paths(paths.clone(), include_parsing)?;
        let mut cache = producer_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(producer) = cache.get(&key).cloned() {
            return Ok(Self { key, producer });
        }
        let producer = Arc::new(FaceInputProducer::open_with_backend(
            key.paths.clone(),
            include_parsing,
            key.backend,
        )?);
        if cache.len() >= MAX_CACHED_FACE_PRODUCERS {
            cache.clear();
        }
        cache.insert(key.clone(), Arc::clone(&producer));
        Ok(Self { key, producer })
    }

    pub fn producer(&self) -> Arc<FaceInputProducer> {
        Arc::clone(&self.producer)
    }

    pub fn key(&self) -> &FaceInputSessionKey {
        &self.key
    }

    pub fn produce(&self, job_id: u64, source: &Image) -> Result<FaceInputBundle, Refusal> {
        if job_id == 0 {
            return Err(Refusal::ValueOutOfRange("face input job ticket"));
        }
        let source_sha256 = sha256_32(&source.pixels);
        let source_size = [source.width, source.height];
        let alias = FaceInputAliasKey {
            session_fingerprint: self.key.fingerprint(),
            source_sha256,
            source_size,
        };
        let result_key = result_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .aliases
            .get(&alias)
            .cloned();
        if let Some(result_key) = result_key {
            let cached = result_cache()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .results
                .get(&result_key)
                .cloned();
            if let Some(cached) = cached {
                if FaceInputResultKey::from_bundle(&self.key, cached.as_ref()).ok()
                    == Some(result_key.clone())
                {
                    let mut bundle = (*cached).clone();
                    bundle.job_id = job_id;
                    match bundle.validate_ticket(source, Some(job_id)) {
                        Ok(()) => return Ok(bundle),
                        Err(refusal) => {
                            let mut cache = result_cache()
                                .lock()
                                .unwrap_or_else(|poisoned| poisoned.into_inner());
                            cache.results.remove(&result_key);
                            cache.aliases.remove(&alias);
                            return Err(refusal);
                        }
                    }
                }
                let mut cache = result_cache()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                cache.results.remove(&result_key);
                cache.aliases.remove(&alias);
            }
            result_cache()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .aliases
                .remove(&alias);
        }

        let bundle = self.producer.produce(job_id, source)?;
        let result_key = FaceInputResultKey::from_bundle(&self.key, &bundle)?;
        let mut cache = result_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cache.results.len() >= MAX_CACHED_FACE_RESULTS {
            cache.results.clear();
            cache.aliases.clear();
        }
        cache
            .results
            .insert(result_key.clone(), Arc::new(bundle.clone()));
        cache.aliases.insert(alias, result_key);
        Ok(bundle)
    }

    pub fn invalidate_all() {
        producer_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        let mut results = result_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        results.results.clear();
        results.aliases.clear();
    }
}

fn file_stamp(path: &Path) -> Result<FaceInputFileStamp, Refusal> {
    let metadata = fs::metadata(path).map_err(|_| Refusal::MissingInput("face model file"))?;
    let modified_nanos = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_nanos());
    Ok(FaceInputFileStamp {
        bytes: metadata.len(),
        modified_nanos,
    })
}

fn canonical_model_path(path: &Path) -> Result<PathBuf, Refusal> {
    fs::canonicalize(path).map_err(|_| Refusal::MissingInput("face model file"))
}

fn quantize_roi(roi: [f32; 4]) -> Result<[i64; 4], Refusal> {
    if roi.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("face input ROI"));
    }
    Ok(roi.map(|value| (f64::from(value) * ROI_CACHE_SCALE).round() as i64))
}

#[derive(Clone, Debug, Deserialize)]
struct LandmarkConstants {
    template: Vec<f32>,
    indices17: Vec<usize>,
    margin: Vec<f32>,
}

impl LandmarkConstants {
    fn load(path: &Path) -> Result<Self, Refusal> {
        let bytes = fs::read(path).map_err(|_| Refusal::MissingInput("face landmark constants"))?;
        let parsed: Self = serde_json::from_slice(&bytes)
            .map_err(|_| Refusal::ShapeMismatch("face landmark constants"))?;
        if parsed.template.len() != 142 * 2
            || parsed.indices17.len() != 17
            || parsed.margin.len() < 4
            || parsed.indices17.iter().any(|index| *index >= 142)
            || parsed
                .template
                .iter()
                .chain(&parsed.margin)
                .any(|value| !value.is_finite())
        {
            return Err(Refusal::ShapeMismatch("face landmark constants"));
        }
        Ok(parsed)
    }

    fn scaled_template(&self) -> Vec<[f32; 2]> {
        let points = self
            .template
            .chunks_exact(2)
            .map(|point| [point[0], point[1]])
            .collect::<Vec<_>>();
        let mut low = [f32::INFINITY; 2];
        let mut high = [f32::NEG_INFINITY; 2];
        for point in &points {
            low[0] = low[0].min(point[0]);
            low[1] = low[1].min(point[1]);
            high[0] = high[0].max(point[0]);
            high[1] = high[1].max(point[1]);
        }
        let extent = [high[0] - low[0], high[1] - low[1]];
        let center = [low[0] + extent[0] * 0.5, low[1] + extent[1] * 0.5];
        let margin = f32::from_bits(0x3e5b_6db8);
        let scale = (DENSE_SIDE as f32 / ((1.0 + margin + margin) * extent[0]))
            .min(DENSE_SIDE as f32 / ((1.0 + margin + margin) * extent[1]));
        points
            .iter()
            .map(|point| {
                [
                    (point[0] - center[0]) * scale + DENSE_SIDE as f32 * 0.5,
                    (point[1] - center[1]) * scale + DENSE_SIDE as f32 * 0.5,
                ]
            })
            .collect()
    }
}

fn validate_detection_session(session: &Session) -> Result<(), Refusal> {
    if session.input.element_type != 1
        || session.input.dims.len() != 4
        || session.input.dims[1] != 3
        || session.outputs.len() != 3
        || session.outputs[0].name != "bbox"
        || session.outputs[1].name != "confidence"
        || session.outputs[2].name != "landmark"
    {
        return Err(Refusal::ShapeMismatch(MODEL_INPUT_ERROR));
    }
    Ok(())
}

fn validate_fixed_session(
    session: &Session,
    input_shape: [i64; 4],
    outputs: &[(&str, &[i64])],
) -> Result<(), Refusal> {
    if session.input.element_type != 1 || session.input.dims.as_slice() != input_shape {
        return Err(Refusal::ShapeMismatch(MODEL_INPUT_ERROR));
    }
    if session.outputs.len() != outputs.len() {
        return Err(Refusal::ShapeMismatch(MODEL_OUTPUT_ERROR));
    }
    for (actual, (name, shape)) in session.outputs.iter().zip(outputs) {
        if actual.name != *name || actual.dims.as_slice() != *shape {
            return Err(Refusal::ShapeMismatch(MODEL_OUTPUT_ERROR));
        }
    }
    Ok(())
}

fn verify_file(path: &Path, expected: [u8; 32]) -> Result<(), Refusal> {
    let bytes = fs::read(path).map_err(|_| Refusal::MissingInput("face model file"))?;
    let actual: [u8; 32] = super::sha256_32(&bytes);
    if actual != expected {
        return Err(Refusal::ValueOutOfRange("face model identity"));
    }
    Ok(())
}

fn decode_detection_point(
    offsets: [f32; 2],
    anchor: [f32; 2],
    stride: f32,
    input_to_source: [[f32; 3]; 2],
) -> [f32; 2] {
    let point = std::array::from_fn(|axis| offsets[axis] * stride + anchor[axis]);
    map_point(point, input_to_source)
}

fn anchor_layout(index: usize) -> (usize, usize, usize) {
    let first = 2 * (DETECTION_SIDE / 8).pow(2);
    let second = 2 * (DETECTION_SIDE / 16).pow(2);
    if index < first {
        (8, DETECTION_SIDE / 8, index / 2)
    } else if index < first + second {
        (16, DETECTION_SIDE / 16, (index - first) / 2)
    } else {
        (32, DETECTION_SIDE / 32, (index - first - second) / 2)
    }
}

fn crop17(roi: [f32; 4], margin_values: &[f32]) -> Result<([[f32; 3]; 2], [[f32; 3]; 2]), Refusal> {
    if roi.iter().any(|value| !value.is_finite()) || roi[2] <= 0.0 || roi[3] <= 0.0 {
        return Err(Refusal::ValueOutOfRange("C1 face ROI"));
    }
    let margin = margin_values[0];
    let center = [roi[0] + roi[2] * 0.5, roi[1] + roi[3] * 0.5];
    let denominator_x = (roi[2] + roi[2] * margin) + roi[2] * margin;
    let denominator_y = (roi[3] + roi[3] * margin) + roi[3] * margin;
    let scale = (COARSE_SIDE as f32 / denominator_x).min(COARSE_SIDE as f32 / denominator_y);
    let matrix = [
        [scale, 0.0, COARSE_SIDE as f32 * 0.5 - scale * center[0]],
        [0.0, scale, COARSE_SIDE as f32 * 0.5 - scale * center[1]],
    ];
    Ok((half(matrix), inverse(matrix)?))
}

fn fit_similarity(source: &[[f32; 2]], target: &[[f32; 2]]) -> Result<[[f32; 3]; 2], Refusal> {
    if source.len() != target.len() || source.len() < 2 {
        return Err(Refusal::ShapeMismatch("face 142 crop landmarks"));
    }
    let inverse_count = 1.0f64 / source.len() as f64;
    let mut source_mean = [0.0f64; 2];
    let mut target_mean = [0.0f64; 2];
    for (source, target) in source.iter().zip(target) {
        source_mean[0] += source[0] as f64;
        source_mean[1] += source[1] as f64;
        target_mean[0] += target[0] as f64;
        target_mean[1] += target[1] as f64;
    }
    source_mean[0] *= inverse_count;
    source_mean[1] *= inverse_count;
    target_mean[0] *= inverse_count;
    target_mean[1] *= inverse_count;
    let mut dot = 0.0f64;
    let mut cross = 0.0f64;
    let mut norm = 0.0f64;
    for (source, target) in source.iter().zip(target) {
        let px = source[0] as f64 - source_mean[0];
        let py = source[1] as f64 - source_mean[1];
        let qx = target[0] as f64 - target_mean[0];
        let qy = target[1] as f64 - target_mean[1];
        dot += px * qx + py * qy;
        cross += px * qy - py * qx;
        norm += px * px + py * py;
    }
    if norm <= 1e-12 || !norm.is_finite() {
        return Err(Refusal::ValueOutOfRange("face 142 crop landmarks"));
    }
    let a = dot / norm;
    let b = cross / norm;
    if !a.is_finite() || !b.is_finite() {
        return Err(Refusal::ValueOutOfRange("face 142 crop matrix"));
    }
    Ok([
        [
            a as f32,
            (-b) as f32,
            (target_mean[0] - a * source_mean[0] + b * source_mean[1]) as f32,
        ],
        [
            b as f32,
            a as f32,
            (target_mean[1] - b * source_mean[0] - a * source_mean[1]) as f32,
        ],
    ])
}

fn half(mut matrix: [[f32; 3]; 2]) -> [[f32; 3]; 2] {
    for row in &mut matrix {
        row[2] = (row[2] - 0.5) + (row[0] * 0.5 + row[1] * 0.5);
    }
    matrix
}

fn inverse(matrix: [[f32; 3]; 2]) -> Result<[[f32; 3]; 2], Refusal> {
    let [a, b, c] = matrix[0];
    let [d, e, f] = matrix[1];
    let determinant = a * e - d * b;
    if determinant == 0.0 || !determinant.is_finite() {
        return Err(Refusal::ValueOutOfRange("face crop matrix"));
    }
    let reciprocal = 1.0 / determinant;
    let aa = e * reciprocal;
    let bb = -b * reciprocal;
    let dd = -d * reciprocal;
    let ee = a * reciprocal;
    Ok([
        [aa, bb, (-aa * c) - (bb * f)],
        [dd, ee, (-dd * c) - (ee * f)],
    ])
}

fn map_point(point: [f32; 2], matrix: [[f32; 3]; 2]) -> [f32; 2] {
    [
        point[0] * matrix[0][0] + point[1] * matrix[0][1] + matrix[0][2],
        point[0] * matrix[1][0] + point[1] * matrix[1][1] + matrix[1][2],
    ]
}

fn validate_source_points(
    points: &[[f32; 2]],
    source: &Image,
    name: &'static str,
) -> Result<(), Refusal> {
    let width = source.width as f32;
    let height = source.height as f32;
    if points.iter().flatten().any(|value| !value.is_finite())
        || points.iter().any(|point| {
            point[0] < -0.25 * width
                || point[0] > 1.25 * width
                || point[1] < -0.25 * height
                || point[1] > 1.25 * height
        })
    {
        return Err(Refusal::ValueOutOfRange(name));
    }
    Ok(())
}

fn resize_rgb(source: &Image, width: usize, height: usize) -> Vec<u8> {
    let mut output = vec![0u8; width * height * 3];
    let scale_x = source.width as f32 / width as f32;
    let scale_y = source.height as f32 / height as f32;
    for y in 0..height {
        let source_y = (y as f32 + 0.5) * scale_y - 0.5;
        let y0 = source_y.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(source.height.saturating_sub(1) as usize);
        let fy = (source_y - source_y.floor()).clamp(0.0, 1.0);
        for x in 0..width {
            let source_x = (x as f32 + 0.5) * scale_x - 0.5;
            let x0 = source_x.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(source.width.saturating_sub(1) as usize);
            let fx = (source_x - source_x.floor()).clamp(0.0, 1.0);
            let p00 = source.pixel(x0 as u32, y0 as u32);
            let p01 = source.pixel(x1 as u32, y0 as u32);
            let p10 = source.pixel(x0 as u32, y1 as u32);
            let p11 = source.pixel(x1 as u32, y1 as u32);
            for channel in 0..3 {
                let top = p00[channel] as f32 * (1.0 - fx) + p01[channel] as f32 * fx;
                let bottom = p10[channel] as f32 * (1.0 - fx) + p11[channel] as f32 * fx;
                output[(y * width + x) * 3 + channel] =
                    (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    output
}

fn warp_reflect_rgb(source: &Image, matrix: [[f32; 3]; 2], side: usize) -> Vec<u8> {
    let inverse = inverse(matrix).unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let width = source.width as isize;
    let height = source.height as isize;
    let mut output = vec![0u8; side * side * 3];
    for y in 0..side {
        for x in 0..side {
            let source_x = x as f32 * inverse[0][0] + y as f32 * inverse[0][1] + inverse[0][2];
            let source_y = x as f32 * inverse[1][0] + y as f32 * inverse[1][1] + inverse[1][2];
            let x0 = source_x.floor() as isize;
            let y0 = source_y.floor() as isize;
            let fx = source_x - x0 as f32;
            let fy = source_y - y0 as f32;
            let xs = [reflect101(x0, width), reflect101(x0 + 1, width)];
            let ys = [reflect101(y0, height), reflect101(y0 + 1, height)];
            let p00 = source.pixel(xs[0] as u32, ys[0] as u32);
            let p01 = source.pixel(xs[1] as u32, ys[0] as u32);
            let p10 = source.pixel(xs[0] as u32, ys[1] as u32);
            let p11 = source.pixel(xs[1] as u32, ys[1] as u32);
            for channel in 0..3 {
                let top = p00[channel] as f32 * (1.0 - fx) + p01[channel] as f32 * fx;
                let bottom = p10[channel] as f32 * (1.0 - fx) + p11[channel] as f32 * fx;
                output[(y * side + x) * 3 + channel] =
                    (top * (1.0 - fy) + bottom * fy).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    output
}

fn reflect101(index: isize, length: isize) -> isize {
    if length <= 1 {
        return 0;
    }
    let period = 2 * (length - 1);
    let value = index.rem_euclid(period);
    if value < length {
        value
    } else {
        period - value
    }
}

fn warp_affine_bgr(
    source: &Image,
    matrix: [[f32; 3]; 2],
    side: usize,
) -> Result<Vec<f32>, Refusal> {
    let inverse = inverse(matrix)?.map(|row| row.map(f64::from));
    let width = source.width as usize;
    let height = source.height as usize;
    if width < 2 || height < 2 {
        return Err(Refusal::ShapeMismatch("face crop source"));
    }
    let mut output = vec![0.0f32; 3 * side * side];
    for y in 0..side {
        let yf = y as f64;
        let mut x_position = inverse[0][1] * yf + inverse[0][2];
        let mut y_position = inverse[1][1] * yf + inverse[1][2];
        for x in 0..side {
            let ix = x_position.floor();
            let iy = y_position.floor();
            if ix >= 0.0 && ix < (width - 1) as f64 && iy >= 0.0 && iy < (height - 1) as f64 {
                let x0 = ix as usize;
                let y0 = iy as usize;
                let fx = (x_position - ix) as f32;
                let fy = (y_position - iy) as f32;
                let ox = 1.0 - fx;
                let oy = 1.0 - fy;
                let sample = |xx: usize, yy: usize| {
                    let offset = (yy * width + xx) * 4;
                    [
                        f32::from(source.pixels[offset]),
                        f32::from(source.pixels[offset + 1]),
                        f32::from(source.pixels[offset + 2]),
                    ]
                };
                let s00 = sample(x0, y0);
                let s01 = sample(x0 + 1, y0);
                let s10 = sample(x0, y0 + 1);
                let s11 = sample(x0 + 1, y0 + 1);
                for channel in 0..3 {
                    let mut value = s00[channel] * (ox * oy);
                    value += s01[channel] * (fx * oy);
                    value += s10[channel] * (ox * fy);
                    value += s11[channel] * (fx * fy);
                    output[(2 - channel) * side * side + y * side + x] = value;
                }
            }
            x_position += inverse[0][0];
            y_position += inverse[1][0];
        }
    }
    Ok(output)
}

fn normalize_points(points: &[[f32; 2]], width: f32, height: f32) -> Vec<[f32; 2]> {
    points
        .iter()
        .map(|point| [point[0] / width, point[1] / height])
        .collect()
}

fn midpoint(left: [f32; 2], right: [f32; 2]) -> [f32; 2] {
    [(left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5]
}

fn distance(left: [f32; 2], right: [f32; 2]) -> f32 {
    let dx = right[0] - left[0];
    let dy = right[1] - left[1];
    (dx * dx + dy * dy).sqrt()
}

fn sample_mask_plane(
    plane: &[f32],
    src_w: usize,
    src_h: usize,
    dst_w: u32,
    dst_h: u32,
) -> Vec<f32> {
    let dw = dst_w as usize;
    let dh = dst_h as usize;
    if src_w == 0 || src_h == 0 || plane.len() < src_w * src_h {
        return vec![0.0; dw * dh];
    }
    if src_w == dw && src_h == dh {
        return plane[..dw * dh].to_vec();
    }
    let mut out = Vec::with_capacity(dw * dh);
    for y in 0..dh {
        let fy = ((y as f32 + 0.5) * src_h as f32 / dh as f32 - 0.5)
            .clamp(0.0, (src_h - 1) as f32);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let wy = fy - y0 as f32;
        for x in 0..dw {
            let fx = ((x as f32 + 0.5) * src_w as f32 / dw as f32 - 0.5)
                .clamp(0.0, (src_w - 1) as f32);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let wx = fx - x0 as f32;
            let a = plane[y0 * src_w + x0] * (1.0 - wx) + plane[y0 * src_w + x1] * wx;
            let b = plane[y1 * src_w + x0] * (1.0 - wx) + plane[y1 * src_w + x1] * wx;
            out.push((a * (1.0 - wy) + b * wy).clamp(0.0, 1.0));
        }
    }
    out
}

fn sha256_32(bytes: &[u8]) -> [u8; 32] {
    super::sha256_32(bytes)
}

const fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => panic!("invalid SHA-256 hex"),
    }
}

const fn hex32(hex: &[u8]) -> [u8; 32] {
    assert!(hex.len() == 64);
    let mut output = [0u8; 32];
    let mut index = 0;
    while index < output.len() {
        output[index] = (hex_nibble(hex[index * 2]) << 4) | hex_nibble(hex[index * 2 + 1]);
        index += 1;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detection_undoes_letterboxing_on_each_axis() {
        for (left, top) in [(0.0, 0.0), (100.0, 0.0), (0.0, 100.0)] {
            let inverse = inverse([[0.5, 0.0, left], [0.0, 0.5, top]]).unwrap();
            assert_eq!(
                decode_detection_point([1.5, -1.0], [32.0 + left, 40.0 + top], 8.0, inverse),
                [88.0, 64.0],
            );
        }
    }

    #[test]
    fn similarity_recovers_rotation_and_scale() {
        let source = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]];
        let target = source.map(|point| [point[1] * 3.0 + 7.0, -point[0] * 3.0 - 2.0]);
        let matrix = fit_similarity(&source, &target).unwrap();
        for (source, target) in source.into_iter().zip(target) {
            let mapped = map_point(source, matrix);
            assert!((mapped[0] - target[0]).abs() < 1e-4);
            assert!((mapped[1] - target[1]).abs() < 1e-4);
        }
    }

    #[test]
    fn half_matrix_round_trips_through_inverse() {
        let matrix = half([[1.25, -0.15, 42.0], [0.1, 0.95, -7.0]]);
        let probe = [19.0, 31.0];
        let mapped = map_point(probe, inverse(matrix).unwrap());
        let round_trip = map_point(mapped, matrix);
        assert!((round_trip[0] - probe[0]).abs() < 1e-4);
        assert!((round_trip[1] - probe[1]).abs() < 1e-4);
    }

    #[test]
    fn reflect101_is_border_symmetric() {
        assert_eq!(reflect101(-1, 5), 1);
        assert_eq!(reflect101(5, 5), 3);
        assert_eq!(reflect101(-3, 5), 3);
    }

    #[test]
    fn eye_pair_deg01_follows_the_emitted_quad_side() {
        assert_eq!(eye_pair_deg01(2, 0, 0.25, 0.5), 0.25);
        assert_eq!(eye_pair_deg01(2, 1, 0.25, 0.5), 0.5);
        assert_eq!(eye_pair_deg01(1, 0, 0.0, 0.5), 0.5);
        assert_eq!(eye_pair_deg01(1, 0, 0.25, 0.0), 0.25);
    }

    #[test]
    fn eye_pair_args_adds_epsilon_only_for_single_side_eye_height() {
        assert_eq!(eye_pair_args(1, 0.0, 0.5), (EYE_PAIR_SINGLE_SIDE_EPSILON_DEG, 50.0));
        assert_eq!(eye_pair_args(1, 0.5, 0.0), (50.0, EYE_PAIR_SINGLE_SIDE_EPSILON_DEG));
        assert_eq!(eye_pair_args(1, 0.5, 0.25), (50.0, 25.0));
        assert_eq!(eye_pair_args(0, 0.0, 0.5), (0.0, 50.0));
        assert_eq!(eye_pair_args(2, 0.5, 0.0), (50.0, 0.0));
    }

    #[test]
    fn roi_cache_key_distinguishes_small_roi_changes() {
        let left = quantize_roi([0.125, 0.25, 0.5, 0.5]).unwrap();
        let right = quantize_roi([0.125001, 0.25, 0.5, 0.5]).unwrap();
        assert_ne!(left, right);
    }
}
