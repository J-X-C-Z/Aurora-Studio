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
const DENSE_SCORE_MIN: f32 = 0.5;

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
        Ok(())
    }
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
        let mut landmarks = [[0.0f32; 2]; 5];
        for (index, point) in landmarks.iter_mut().enumerate() {
            for axis in 0..2 {
                let value = outputs[2][best * 10 + index * 2 + axis];
                if !value.is_finite() {
                    return Err(Refusal::ValueOutOfRange(FACE_LANDMARK_ERROR));
                }
                *point.get_mut(axis).unwrap() =
                    ((value * stride as f32) + anchor[axis] - left as f32) / scale;
            }
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
            source_to_input: [[scale, 0.0, left as f32], [0.0, scale, top as f32]],
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
        if !score.is_finite() || score < DENSE_SCORE_MIN {
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
    fn roi_cache_key_distinguishes_small_roi_changes() {
        let left = quantize_roi([0.125, 0.25, 0.5, 0.5]).unwrap();
        let right = quantize_roi([0.125001, 0.25, 0.5, 0.5]).unwrap();
        assert_ne!(left, right);
    }
}
