use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
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

/// PF-F: bounded same-process bridge between `Source::face_input_image()` and the
/// session cache. `Source`'s own image field is outside that accessor's edit
/// boundary, so the shared `Arc` lives here. Entries are keyed by the immutable
/// decoded plane pointer, its length, and the Source's cached snapshot
/// fingerprint; the inner `Arc` pointer check keeps allocator reuse from
/// matching a stale entry. The table is bounded to two sources.
const MAX_SHARED_FACE_INPUT_IMAGES: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct SharedFaceInputKey {
    plane_ptr: usize,
    plane_len: usize,
    fingerprint: u64,
    /// Dimensions are part of the key: two equal-length planes at one address
    /// can otherwise differ by width/height after allocator reuse.
    size: [u32; 2],
}

struct SharedFaceInputImage {
    image: Arc<Image>,
    sha256: Arc<OnceLock<[u8; 32]>>,
    key: SharedFaceInputKey,
    stamp: u64,
}

type SharedFaceInputTable = Mutex<(HashMap<SharedFaceInputKey, SharedFaceInputImage>, u64)>;

fn shared_face_inputs() -> &'static SharedFaceInputTable {
    static CACHE: OnceLock<SharedFaceInputTable> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new((HashMap::new(), 0)))
}

/// Returns the already-shared image for `key`, if this process still holds it.
pub(crate) fn shared_face_input_image(key: (usize, usize, u64, u32, u32)) -> Option<Arc<Image>> {
    let key = SharedFaceInputKey {
        plane_ptr: key.0,
        plane_len: key.1,
        fingerprint: key.2,
        size: [key.3, key.4],
    };
    let mut guard = shared_face_inputs()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.1 += 1;
    let stamp = guard.1;
    let entry = guard.0.get_mut(&key)?;
    entry.stamp = stamp;
    Some(Arc::clone(&entry.image))
}

/// Returns the RGBA8 pixel capacity currently held in the shared input table
/// for `plane_ptr`/`plane_len` without computing a fingerprint or allocating.
pub(crate) fn shared_face_input_bytes_for_plane(plane_ptr: usize, plane_len: usize) -> usize {
    let guard = shared_face_inputs()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard
        .0
        .iter()
        .find(|(k, _)| k.plane_ptr == plane_ptr && k.plane_len == plane_len)
        .map_or(0, |(_, entry)| entry.image.pixels.capacity())
}

/// Publishes `image` as the shared input for `key` and returns it unchanged.
pub(crate) fn register_face_input_image(
    key: (usize, usize, u64, u32, u32),
    image: Arc<Image>,
) -> Arc<Image> {
    let key = SharedFaceInputKey {
        plane_ptr: key.0,
        plane_len: key.1,
        fingerprint: key.2,
        size: [key.3, key.4],
    };
    let mut guard = shared_face_inputs()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.1 += 1;
    let stamp = guard.1;
    guard.0.insert(
        key,
        SharedFaceInputImage {
            image: Arc::clone(&image),
            sha256: Arc::new(OnceLock::new()),
            key,
            stamp,
        },
    );
    while guard.0.len() > MAX_SHARED_FACE_INPUT_IMAGES {
        let victim = guard
            .0
            .values()
            .min_by_key(|entry| entry.stamp)
            .map(|entry| entry.key);
        match victim {
            Some(victim) => {
                guard.0.remove(&victim);
            }
            None => break,
        }
    }
    image
}

/// Shared SHA-256 for an immutable image produced by this process's Source cache.
/// The `Arc` pointer check makes this sound even if the allocator reuses the
/// plane address for a different image.
fn shared_source_digest(image: &Image) -> Option<Arc<OnceLock<[u8; 32]>>> {
    let guard = shared_face_inputs()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard
        .0
        .values()
        .find(|entry| {
            entry.image.pixels.as_ptr() as usize == image.pixels.as_ptr() as usize
                && entry.image.pixels.len() == image.pixels.len()
                && std::ptr::eq(Arc::as_ptr(&entry.image), image)
        })
        .map(|entry| Arc::clone(&entry.sha256))
}

/// One full-frame SHA-256 shared by every same-frame check inside a render.
///
/// The renderer owns an immutable stage frame while the portrait attachment
/// block runs. Every attachment validates that frame against its bundle and
/// several of them also bind the frame digest into a geometry draw; without a
/// scope each of those calls re-reads the whole RGBA plane. The scope keeps a
/// lazily computed digest for the exact allocation it borrows, so the frame is
/// hashed once and every later check in the same scope reuses that value.
///
/// The borrow (`&'a Image`) is load-bearing: it keeps the plane alive and
/// immutable for as long as an entry exists. A different allocation, a fresh
/// clone, or a frame rendered on another thread never matches and falls back
/// to hashing exactly as before.
pub(crate) struct FrameDigestScope<'a> {
    _source: &'a Image,
    ptr: usize,
    len: usize,
    width: u32,
    height: u32,
}

struct FrameDigestEntry {
    ptr: usize,
    len: usize,
    width: u32,
    height: u32,
    digest: Option<[u8; 32]>,
}

thread_local! {
    static FRAME_DIGEST_SCOPES: RefCell<Vec<FrameDigestEntry>> = RefCell::new(Vec::new());
}

impl<'a> FrameDigestScope<'a> {
    pub(crate) fn new(source: &'a Image) -> Self {
        let entry = FrameDigestEntry {
            ptr: source.pixels.as_ptr() as usize,
            len: source.pixels.len(),
            width: source.width,
            height: source.height,
            digest: None,
        };
        let (ptr, len, width, height) = (entry.ptr, entry.len, entry.width, entry.height);
        FRAME_DIGEST_SCOPES.with(|scopes| scopes.borrow_mut().push(entry));
        Self {
            _source: source,
            ptr,
            len,
            width,
            height,
        }
    }
}

impl Drop for FrameDigestScope<'_> {
    fn drop(&mut self) {
        FRAME_DIGEST_SCOPES.with(|scopes| {
            let mut scopes = scopes.borrow_mut();
            if let Some(index) = scopes.iter().rposition(|entry| {
                entry.ptr == self.ptr
                    && entry.len == self.len
                    && entry.width == self.width
                    && entry.height == self.height
            }) {
                scopes.remove(index);
            }
        });
    }
}

#[cfg(test)]
thread_local! {
    static FRAME_DIGEST_COMPUTES: std::cell::Cell<usize> = std::cell::Cell::new(0);
}

#[cfg(test)]
fn note_frame_digest_compute() {
    FRAME_DIGEST_COMPUTES.with(|count| count.set(count.get() + 1));
}

#[cfg(test)]
fn reset_frame_digest_computes() {
    FRAME_DIGEST_COMPUTES.with(|count| count.set(0));
}

#[cfg(test)]
fn frame_digest_computes() -> usize {
    FRAME_DIGEST_COMPUTES.with(|count| count.get())
}

fn computed_frame_digest(source: &Image) -> [u8; 32] {
    #[cfg(test)]
    note_frame_digest_compute();
    sha256_32(&source.pixels)
}

fn scoped_frame_digest(source: &Image) -> Option<[u8; 32]> {
    FRAME_DIGEST_SCOPES.with(|scopes| {
        let mut scopes = scopes.borrow_mut();
        let entry = scopes.iter_mut().rev().find(|entry| {
            entry.ptr == source.pixels.as_ptr() as usize
                && entry.len == source.pixels.len()
                && entry.width == source.width
                && entry.height == source.height
        })?;
        Some(*entry.digest.get_or_insert_with(|| computed_frame_digest(source)))
    })
}

/// SHA-256 of `source.pixels`, reusing an active [`FrameDigestScope`] or the
/// per-source cache when either already covers this exact allocation.
pub(crate) fn frame_digest(source: &Image) -> [u8; 32] {
    if let Some(digest) = scoped_frame_digest(source) {
        return digest;
    }
    match shared_source_digest(source) {
        Some(digest) => *digest.get_or_init(|| computed_frame_digest(source)),
        None => computed_frame_digest(source),
    }
}

/// The session cache uses exact content SHA-256; shared source images compute
/// it only once per immutable allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FaceInputSourceIdentity {
    Snapshot {
        fingerprint: u64,
        size: [u32; 2],
    },
    ContentSha256 {
        sha256: [u8; 32],
        size: [u32; 2],
    },
}

impl FaceInputSourceIdentity {
    fn size(self) -> [u32; 2] {
        match self {
            Self::Snapshot { size, .. } | Self::ContentSha256 { size, .. } => size,
        }
    }
}

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
        let parsing = if include_parsing {
            face_parsing_v2_model_path().unwrap_or_default()
        } else {
            PathBuf::new()
        };
        let paths = Self::from_roots(c1_root.as_ref(), landmark_root.as_ref(), parsing);
        if include_parsing && !paths.parsing.is_file() {
            return Err(Refusal::MissingInput(FACE_PARSING_V2_MODEL_ENV));
        }
        Ok(paths)
    }

    pub fn from_c1_and_landmark_roots(
        c1_root: impl AsRef<Path>,
        landmark_root: impl AsRef<Path>,
    ) -> Self {
        Self::from_roots(
            c1_root.as_ref(),
            landmark_root.as_ref(),
            face_parsing_v2_model_path().unwrap_or_default(),
        )
    }

    fn from_roots(c1_root: &Path, landmark_root: &Path, parsing: PathBuf) -> Self {
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
            parsing,
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
    candidates.into_iter().find(|root| {
        root.join("assets").join("1063025060.onnx").is_file()
            && root.join("assets").join("323773030.onnx").is_file()
            && root.join("constants.json").is_file()
    })
}

type ParsingModelChecks = Mutex<HashMap<PathBuf, (FaceInputFileStamp, bool)>>;

fn is_canonical_face_parsing_v2(path: &Path) -> bool {
    static CHECKS: OnceLock<ParsingModelChecks> = OnceLock::new();
    matches_parsing_model(
        path,
        crate::face_parsing::FACE_PARSING_V2_MODEL_BYTES,
        crate::face_parsing::FACE_PARSING_V2_MODEL_SHA256,
        CHECKS.get_or_init(|| Mutex::new(HashMap::new())),
    )
}

fn matches_parsing_model(
    path: &Path,
    expected_bytes: u64,
    expected_sha256: [u8; 32],
    checks: &ParsingModelChecks,
) -> bool {
    let Ok(absolute_path) = std::path::absolute(path) else {
        return false;
    };
    let path = absolute_path.as_path();
    let Ok(stamp) = file_stamp(path) else {
        return false;
    };
    if stamp.bytes != expected_bytes {
        return false;
    }
    if let Some((_, verified)) = checks
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(path)
        .filter(|(cached, _)| *cached == stamp)
    {
        return *verified;
    }
    let Ok(bytes) = fs::read(path) else {
        return false;
    };
    let verified = sha256_32(&bytes) == expected_sha256;
    if file_stamp(path) != Ok(stamp) {
        return false;
    }
    if stamp.modified_nanos == 0 {
        return verified;
    }
    let mut checks = checks.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if checks.len() >= 32 {
        checks.clear();
    }
    checks.insert(absolute_path, (stamp, verified));
    verified
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
        let opened = if runtime.is_native_only() {
            runtime.open_session(path)
        } else {
            let providers = match self {
                Self::Cpu => vec![Provider::Cpu],
                Self::Cuda => vec![Provider::Cuda],
                Self::DirectMl => vec![Provider::DirectMl],
                Self::Auto => vec![Provider::Cuda, Provider::DirectMl, Provider::Cpu],
            };
            runtime.open_session_on(path, &providers)
        };
        opened.map_err(|error| {
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
    /// Exact digest of the frame this bundle was produced from, tied to the
    /// allocation that held it. In-process validation reuses it while that
    /// allocation is still the one being checked.
    pub(crate) bound_digest: Option<([u8; 32], usize, usize)>,
}

impl FaceInputBundle {
    fn reticket(&mut self, job_id: u64) {
        self.job_id = job_id;
        if let Some(inputs) = self.model_inputs.as_mut() {
            if let Some(parsing) = inputs.parsing.as_mut() {
                parsing.job_id = job_id;
            }
            if let Some(teeth) = inputs.teeth.as_mut() {
                teeth.job_id = job_id;
            }
        }
    }

    fn digest_for(&self, source: &Image) -> [u8; 32] {
        if let Some((sha256, ptr, len)) = self.bound_digest {
            if ptr == source.pixels.as_ptr() as usize && len == source.pixels.len() {
                return sha256;
            }
        }
        frame_digest(source)
    }

    pub fn validate_ticket(&self, source: &Image, job_id: Option<u64>) -> Result<(), Refusal> {
        let digest = self.digest_for(source);
        self.validate_ticket_with_digest(source, job_id, || digest)
    }

    fn validate_ticket_with_digest(
        &self,
        source: &Image,
        job_id: Option<u64>,
        digest: impl FnOnce() -> [u8; 32],
    ) -> Result<(), Refusal> {
        if self.job_id == 0 || Some(self.job_id) != job_id {
            return Err(Refusal::ValueOutOfRange("face input job ticket"));
        }
        if self.source_size != [source.width, source.height] {
            return Err(Refusal::ShapeMismatch("face input source size"));
        }
        if self.source_sha256 != digest() {
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
        SlimProducerInputs::from_landmarks_with_sha(
            self.job_id,
            source,
            self.landmark_producer.points194.clone(),
            self.digest_for(source),
        )
    }

    pub fn slim_inputs_for_stage(
        &self,
        transform: &FaceStageTransform,
        stage: &Image,
    ) -> Result<SlimProducerInputs, Refusal> {
        let digest = self.digest_for(stage);
        self.slim_inputs_for_stage_with_sha(transform, stage, digest)
    }

    pub fn slim_inputs_for_stage_with_sha(
        &self,
        transform: &FaceStageTransform,
        stage: &Image,
        digest: [u8; 32],
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
        let stage_size = [stage.width, stage.height];
        if let Some(mut slim) = cached_slim(digest, stage_size, &points194) {
            slim.job_id = self.job_id;
            return Ok(slim);
        }
        let slim = SlimProducerInputs::from_landmarks_with_sha(
            self.job_id,
            stage,
            points194.clone(),
            digest,
        )?;
        remember_slim(digest, stage_size, &points194, &slim);
        Ok(slim)
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
        let digest = self.digest_for(source);
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
        let digest = self.digest_for(source);
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
            let desc = super::portrait_plump_stage::full_face_descriptor_from_store(raw)?;
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
            asset: &assets[1],
            channel_map: aurora_portrait::kernels::plump_pipeline::ChannelMap {
                color: [0, 1, 2],
                split_mask: 3,
            },
            color_encoding:
                aurora_portrait::kernels::plump_pipeline::ColorEncoding::VendorResidualSignedUnit,
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
        let receipt_draw = (|| -> Result<super::FaceGeometryDraw, Refusal> {
            let receipt_path = std::env::var_os(
                super::portrait_pf8033_cpu_forward::PF8033_CPU_FORWARD_RECEIPT_ENV,
            )
            .map(std::path::PathBuf::from)
            .or_else(|| aurora_portrait::private_assets::path("portrait/smile/pf8033-cpu-forward.json"))
            .ok_or(Refusal::MissingInput("PF8033 CPU forward receipt"))?;
            let receipt_bytes = std::fs::read(&receipt_path)
                .map_err(|_| Refusal::MissingInput("PF8033 CPU forward receipt"))?;
            let receipt = super::portrait_pf8033_cpu_forward::Pf8033CpuForwardReceipt::parse(
                receipt_path,
                &receipt_bytes,
            )?;
            let mut artifacts =
                super::portrait_pf8033_cpu_forward::load_pf8033_cpu_forward_artifacts(&receipt, source)?;
            artifacts.host_split_receipt.job_id = self.job_id;
            Ok(super::FaceGeometryDraw {
                pf: 8033,
                controls: [c8033, 0.0],
                source_sha256: self.digest_for(source),
                geometry: super::FaceGeometry::Smile {
                    dx: artifacts.dx,
                    dy: artifacts.dy,
                    mask: artifacts.mask,
                    host_split_receipt: Some(artifacts.host_split_receipt),
                },
            })
        })();
        let draw = match receipt_draw {
            Ok(draw) => draw,
            Err(_) => {
                let mut draw = super::portrait_pixcake_smile::build_pf8033_smile_draw(
                    source,
                    &self.points194,
                    c8033,
                )?;
                draw.source_sha256 = self.digest_for(source);
                draw
            }
        };
        let inputs = super::FaceGeometryInputs {
            job_id: self.job_id,
            width: source.width,
            height: source.height,
            draws: vec![draw],
        };
        self.attach_face_geometry(source, inputs, producer)
    }

    /// Attaches same-frame PixCake repair / eye-bag producers (`PF5` dark circle,
    /// `PF90116` eye bag, `PF90080` oil shine, `PF90118` lip line) when active in `extra`.
    pub fn attach_pixcake_repair_and_eyebag_from_extra(
        &self,
        source: &Image,
        masks: Option<&super::PortraitMasks>,
        extra: &std::collections::BTreeMap<String, f32>,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        let active = |pf: u32| {
            extra
                .get(&format!("pc:{pf}"))
                .copied()
                .is_some_and(|v| v.is_finite() && v > 1.0e-6)
        };
        let want_5 = active(5);
        let want_90116 = active(90116) && producer.eye_bag.is_none();
        let want_90080 = active(90080);
        let want_90118 = active(90118);
        if !want_5 && !want_90116 && !want_90080 && !want_90118 {
            return Ok(());
        }

        fn project_plane_u8(
            plane: &[u8],
            crop_w: u32,
            crop_h: u32,
            source_to_crop: [[f32; 3]; 2],
            dst_w: u32,
            dst_h: u32,
        ) -> Vec<u8> {
            let cw = crop_w as usize;
            let ch = crop_h as usize;
            let dw = dst_w as usize;
            let dh = dst_h as usize;
            let mut out = vec![0u8; dw.saturating_mul(dh)];
            if cw == 0 || ch == 0 || plane.len() < cw * ch {
                return out;
            }
            let max_cx = (cw - 1) as f32;
            let max_cy = (ch - 1) as f32;
            for y in 0..dh {
                let sy = y as f32;
                let row_off = y * dw;
                for x in 0..dw {
                    let sx = x as f32;
                    let cx = source_to_crop[0][0] * sx + source_to_crop[0][1] * sy + source_to_crop[0][2];
                    let cy = source_to_crop[1][0] * sx + source_to_crop[1][1] * sy + source_to_crop[1][2];
                    if cx < 0.0 || cy < 0.0 || cx > max_cx || cy > max_cy {
                        continue;
                    }
                    let x0 = cx.floor() as usize;
                    let y0 = cy.floor() as usize;
                    let x1 = (x0 + 1).min(cw - 1);
                    let y1 = (y0 + 1).min(ch - 1);
                    let fx = cx - x0 as f32;
                    let fy = cy - y0 as f32;
                    let v00 = plane[y0 * cw + x0] as f32;
                    let v10 = plane[y0 * cw + x1] as f32;
                    let v01 = plane[y1 * cw + x0] as f32;
                    let v11 = plane[y1 * cw + x1] as f32;
                    let v = (v00 * (1.0 - fx) + v10 * fx) * (1.0 - fy)
                        + (v01 * (1.0 - fx) + v11 * fx) * fy;
                    out[row_off + x] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
            out
        }

        let mut parsing_skin: Option<Vec<u8>> = None;
        let mut parsing_eye: Option<Vec<u8>> = None;
        let mut parsing_upper_lip: Option<Vec<u8>> = None;
        let mut parsing_lower_lip: Option<Vec<u8>> = None;
        if let Some(parsing_ref) = self
            .model_inputs
            .as_ref()
            .and_then(|inputs| inputs.parsing.as_ref())
        {
            let mut parsing = parsing_ref.clone();
            parsing.job_id = self.job_id;
            if let Ok(slots) = parsing.consumer_slots() {
                let proj = |slot_idx: usize| {
                    project_plane_u8(
                        &slots[slot_idx],
                        parsing.width,
                        parsing.height,
                        parsing.source_to_crop,
                        source.width,
                        source.height,
                    )
                };
                if want_5 || want_90116 || want_90080 {
                    parsing_skin = Some(proj(18));
                }
                if want_5 || want_90116 {
                    parsing_eye = Some(proj(19));
                }
                if want_90118 {
                    parsing_upper_lip = Some(proj(9));
                    parsing_lower_lip = Some(proj(10));
                }
            }
        }
        let fallback_skin = masks.and_then(|m| {
            m.mask_u8(
                super::MaskKind::Skin,
                source.width,
                source.height,
                crate::NormRect::FULL,
            )
        });
        let skin_mask_ref = parsing_skin.as_deref().or(fallback_skin.as_deref());

        if want_5 {
            if let Ok(output) = super::portrait_pixcake_eyebag::build_pf5_dark_circle_output(
                source,
                &self.points194,
                skin_mask_ref,
                parsing_eye.as_deref(),
            ) {
                let _ = producer
                    .repairs
                    .get_or_insert_default()
                    .insert(self.job_id, source, output);
            }
        }
        if want_90116 {
            if let Ok(mut planes) = super::portrait_pixcake_eyebag::build_pf90116_eye_bag_planes(
                source,
                &self.points194,
                skin_mask_ref,
                parsing_eye.as_deref(),
            ) {
                planes.ticket = self.job_id;
                producer.eye_bag_landmark = Some(planes);
            }
        }
        if want_90080 {
            if let Some(skin) = skin_mask_ref {
                if let Ok(output) = super::portrait_pixcake_repair::build_pf90080_oil_shine_output(
                    source,
                    skin,
                    &self.points194,
                ) {
                    let _ = producer
                        .repairs
                        .get_or_insert_default()
                        .insert(self.job_id, source, output);
                }
            }
        }
        if want_90118 {
            if let (Some(upper), Some(lower)) =
                (parsing_upper_lip.as_deref(), parsing_lower_lip.as_deref())
            {
                if let Ok(output) = super::portrait_pixcake_repair::build_pf90118_lip_line_output(
                    source,
                    upper,
                    lower,
                    &self.points194,
                ) {
                    let _ = producer
                        .repairs
                        .get_or_insert_default()
                        .insert(self.job_id, source, output);
                }
            }
        }
        Ok(())
    }

    /// Attaches the remaining 19 same-frame PixCake effect producers
    /// (`21026, 90088, 90131, 90130, 30025, 30026, 21076, 100007, 30259, 90008,
    /// 90089, 90034, 100008, 30030, 44297, 30187, 30190, 30059, 90033`) when active.
    pub fn attach_pixcake_remaining_19_from_extra(
        &self,
        source: &Image,
        masks: Option<&super::PortraitMasks>,
        extra: &std::collections::BTreeMap<String, f32>,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        let val = |pf: u32| extra.get(&format!("pc:{pf}")).copied().unwrap_or(0.0);
        let active = |pf: u32| {
            let v = val(pf);
            v.is_finite() && v.abs() > 1.0e-6
        };
        const REMAINING_19: [u32; 19] = [
            21026, 90088, 90131, 90130, 30025, 30026, 21076, 100007, 30259, 90008, 90089, 90034,
            100008, 30030, 44297, 30187, 30190, 30059, 90033,
        ];
        if !REMAINING_19.iter().any(|&pf| active(pf)) {
            return Ok(());
        }

        fn project_plane_u8(
            plane: &[u8],
            crop_w: u32,
            crop_h: u32,
            source_to_crop: [[f32; 3]; 2],
            dst_w: u32,
            dst_h: u32,
        ) -> Vec<u8> {
            let cw = crop_w as usize;
            let ch = crop_h as usize;
            let dw = dst_w as usize;
            let dh = dst_h as usize;
            let mut out = vec![0u8; dw.saturating_mul(dh)];
            if cw == 0 || ch == 0 || plane.len() < cw * ch {
                return out;
            }
            let max_cx = (cw - 1) as f32;
            let max_cy = (ch - 1) as f32;
            for y in 0..dh {
                let sy = y as f32;
                let row_off = y * dw;
                for x in 0..dw {
                    let sx = x as f32;
                    let cx = source_to_crop[0][0] * sx + source_to_crop[0][1] * sy + source_to_crop[0][2];
                    let cy = source_to_crop[1][0] * sx + source_to_crop[1][1] * sy + source_to_crop[1][2];
                    if cx < 0.0 || cy < 0.0 || cx > max_cx || cy > max_cy {
                        continue;
                    }
                    let x0 = cx.floor() as usize;
                    let y0 = cy.floor() as usize;
                    let x1 = (x0 + 1).min(cw - 1);
                    let y1 = (y0 + 1).min(ch - 1);
                    let fx = cx - x0 as f32;
                    let fy = cy - y0 as f32;
                    let v00 = plane[y0 * cw + x0] as f32;
                    let v10 = plane[y0 * cw + x1] as f32;
                    let v01 = plane[y1 * cw + x0] as f32;
                    let v11 = plane[y1 * cw + x1] as f32;
                    let v = (v00 * (1.0 - fx) + v10 * fx) * (1.0 - fy)
                        + (v01 * (1.0 - fx) + v11 * fx) * fy;
                    out[row_off + x] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
            out
        }

        let mut parsing_face: Option<Vec<u8>> = None;
        let mut parsing_skin: Option<Vec<u8>> = None;
        let mut parsing_eye: Option<Vec<u8>> = None;
        let mut parsing_upper_lip: Option<Vec<u8>> = None;
        let mut parsing_lower_lip: Option<Vec<u8>> = None;
        let mut parsing_teeth: Option<Vec<u8>> = None;
        if let Some(parsing_ref) = self
            .model_inputs
            .as_ref()
            .and_then(|inputs| inputs.parsing.as_ref())
        {
            let mut parsing = parsing_ref.clone();
            parsing.job_id = self.job_id;
            if let Ok(slots) = parsing.consumer_slots() {
                let proj = |slot_idx: usize| {
                    project_plane_u8(
                        &slots[slot_idx],
                        parsing.width,
                        parsing.height,
                        parsing.source_to_crop,
                        source.width,
                        source.height,
                    )
                };
                if active(21026) || active(90088) || active(30259) || active(90089) || active(30030)
                    || active(44297) || active(30187) || active(30190) || active(30059) || active(90033)
                {
                    // The skin group already includes the beard and nasal base.
                    let skin = proj(18);
                    parsing_face = Some(skin.clone());
                    parsing_skin = Some(skin);
                }
                if active(90130) {
                    parsing_eye = Some(proj(19));
                }
                if active(100007) {
                    parsing_upper_lip = Some(proj(9));
                    parsing_lower_lip = Some(proj(10));
                }
                if active(30025) || active(30026) {
                    parsing_teeth = Some(proj(20));
                }
            }
        }

        let mask_u8 = |kind: super::MaskKind| {
            masks.and_then(|m| m.mask_u8(kind, source.width, source.height, crate::NormRect::FULL))
        };
        let skin_from_masks = mask_u8(super::MaskKind::Skin);
        let body_from_masks = mask_u8(super::MaskKind::Body);
        let hair_from_masks = mask_u8(super::MaskKind::Hair);
        let face_from_masks = mask_u8(super::MaskKind::Face);
        let skin_ref = parsing_skin
            .as_deref()
            .or(skin_from_masks.as_deref())
            .or(body_from_masks.as_deref());
        let body_skin_ref = body_from_masks
            .as_deref()
            .or(skin_from_masks.as_deref())
            .or(parsing_skin.as_deref());
        let face_ref = parsing_face.as_deref().or(face_from_masks.as_deref());

        // 1. Hair & Glasses Glare: 21026, 90088, 90131, 90130
        if active(21026) {
            if let Ok(beard) = super::portrait_pixcake_hair_glare::build_pf21026_beard_inputs(
                source,
                &self.points194,
                skin_ref,
            ) {
                producer.beard = Some(beard);
            }
        }
        if active(90088) {
            if let Some(hair) = hair_from_masks.as_deref() {
                if let Ok(stray) = super::portrait_pixcake_hair_glare::build_pf90088_stray_hair_inputs(
                    source,
                    hair,
                    skin_ref,
                ) {
                    producer.stray_hair = Some(stray);
                }
            }
        }
        if active(90131) {
            if let Some(hair) = hair_from_masks.as_deref() {
                if let Ok(out) = super::portrait_pixcake_hair_glare::build_pf90131_white_hair_output(
                    source,
                    hair,
                ) {
                    let _ = producer
                        .repairs
                        .get_or_insert_default()
                        .insert(self.job_id, source, out);
                }
            }
        }
        if active(90130) && producer.pf90130.is_none() {
            if let Ok(glare) = super::portrait_pixcake_hair_glare::build_pf90130_glare_row(
                self.job_id,
                source,
                &self.points194,
                parsing_eye.as_deref(),
            ) {
                producer.pf90130_glare = Some(glare);
            }
        }

        // 2. Teeth & Expression: 30025, 30026, 21076, 100007
        let teeth_mask_ref = producer
            .teeth
            .as_ref()
            .filter(|t| t.mask_width == source.width && t.mask_height == source.height)
            .map(|t| t.mask_g.as_slice())
            .or(parsing_teeth.as_deref());
        if active(30025) {
            if let Some(teeth) = teeth_mask_ref {
                if let Ok(inp) = super::portrait_pixcake_teeth_expr::build_pf30025_teeth_repair_inputs(
                    self.job_id,
                    source,
                    &self.points194,
                    teeth,
                ) {
                    producer.pf30025 = Some(inp);
                }
            }
        }
        if active(30026) {
            if let Some(teeth) = teeth_mask_ref {
                if let Ok(inp) = super::portrait_pixcake_teeth_expr::build_pf30026_teeth_alignment_inputs(
                    self.job_id,
                    source,
                    &self.points194,
                    teeth,
                ) {
                    producer.teeth_alignment = Some(inp);
                }
            }
        }
        if active(21076) {
            if let Ok(out) = super::portrait_pixcake_teeth_expr::build_pf21076_closed_mouth_smile_output(
                source,
                &self.points194,
            ) {
                let _ = producer
                    .repairs
                    .get_or_insert_default()
                    .insert(self.job_id, source, out);
            }
        }
        if active(100007) {
            if let (Some(upper), Some(lower)) =
                (parsing_upper_lip.as_deref(), parsing_lower_lip.as_deref())
            {
                if let Ok(smile) = super::portrait_pixcake_teeth_expr::build_pf100007_open_mouth_smile_inputs(
                    self.job_id,
                    source,
                    &self.points194,
                    upper,
                    lower,
                ) {
                    producer.smile = Some(smile);
                }
            }
        }

        // 3. Blemish & Body Repair: 30259, 90008, 90089, 90034, 100008
        if active(30259) {
            if let Some(skin) = skin_ref {
                if let Ok(out) = super::portrait_pixcake_blemish::build_pf30259_sweat_output(
                    source,
                    skin,
                    &self.points194,
                ) {
                    let _ = producer
                        .repairs
                        .get_or_insert_default()
                        .insert(self.job_id, source, out);
                }
            }
        }
        if active(90008) {
            if let Some(body) = body_skin_ref {
                if let Ok(row) = super::portrait_pixcake_blemish::build_pf90008_stretch_mark_row(
                    self.job_id,
                    source,
                    body,
                ) {
                    let ext = super::HistoricalSkinExternalInputs {
                        ticket: self.job_id,
                        rows: vec![row],
                    };
                    if let Ok(built) = ext.build() {
                        match producer.historical_skin.as_mut() {
                            Some(existing) => {
                                for (pf, input) in built {
                                    existing.insert(pf, input);
                                }
                            }
                            None => producer.historical_skin = Some(built),
                        }
                    }
                }
            }
        }
        if active(90089) {
            if let Some(skin) = body_skin_ref {
                if let Ok(out) = super::portrait_pixcake_blemish::build_pf90089_body_blemish_child_output(
                    source,
                    skin,
                    val(90089).clamp(0.0, 1.0),
                ) {
                    let _ = producer
                        .repairs
                        .get_or_insert_default()
                        .insert(self.job_id, source, out);
                }
            }
        }
        if active(90034) {
            if let Ok(out) = super::portrait_pixcake_blemish::build_pf90034_double_chin_output(
                source,
                &self.points194,
            ) {
                let _ = producer
                    .repairs
                    .get_or_insert_default()
                    .insert(self.job_id, source, out);
            }
        }

        // 4. Historical-tone rows: 100008, 30030, 30187, 30190, 30059
        let mut tone_rows = Vec::new();
        if active(100008) {
            if let Some(body) = body_skin_ref {
                if let Ok(row) = super::portrait_pixcake_blemish::build_pf100008_breast_repair_row(
                    self.job_id,
                    source,
                    body,
                    Some(&self.points194),
                ) {
                    tone_rows.push(row);
                }
            }
        }
        if active(30030) {
            if let Some(body) = body_skin_ref {
                if let Ok(row) = super::portrait_pixcake_hands_feet::build_pf30030_hand_veins_row(
                    self.job_id,
                    source,
                    body,
                    face_ref,
                ) {
                    tone_rows.push(row);
                }
            }
        }
        if active(44297) {
            if let Some(body) = body_skin_ref {
                if let Ok(out) = super::portrait_pixcake_hands_feet::build_pf44297_foot_vein_output(
                    source,
                    body,
                    face_ref,
                ) {
                    let _ = producer
                        .repairs
                        .get_or_insert_default()
                        .insert(self.job_id, source, out);
                }
            }
        }
        if active(30187) {
            if let Some(body) = body_skin_ref {
                if let Ok(row) = super::portrait_pixcake_hands_feet::build_pf30187_hand_gray_row(
                    self.job_id,
                    source,
                    body,
                    face_ref,
                ) {
                    tone_rows.push(row);
                }
            }
        }
        if active(30190) {
            if let Some(body) = body_skin_ref {
                if let Ok(row) = super::portrait_pixcake_hands_feet::build_pf30190_hand_shape_row(
                    self.job_id,
                    source,
                    body,
                    face_ref,
                    val(30190).clamp(0.0, 1.0),
                ) {
                    tone_rows.push(row);
                }
            }
        }
        if active(30059) {
            if let Some(skin) = body_skin_ref {
                if let Ok(data) = super::portrait_pixcake_unify_suit::build_pf30059_multi_tone_row(
                    self.job_id,
                    source,
                    skin,
                    face_ref,
                ) {
                    tone_rows.push(
                        crate::portrait_historical_tone_provider::HistoricalToneProviderRow {
                            frame: source.clone(),
                            job: crate::portrait_historical_tone_provider::HistoricalToneRowJob {
                                source_sha256: self.digest_for(source),
                                size: [source.width, source.height],
                                control: val(30059).clamp(-1.0, 1.0),
                            },
                            data,
                        },
                    );
                }
            }
        }
        if !tone_rows.is_empty() {
            if let Ok(built) =
                crate::portrait_historical_tone_provider::build_historical_tone_inputs(
                    crate::portrait_historical_tone_provider::HistoricalToneProviderPayload {
                        ticket: self.job_id,
                        gray_3d_controls: [0.0, 0.0],
                        rows: tone_rows,
                    },
                )
            {
                match producer.historical_tone.as_mut() {
                    Some(existing) => {
                        for (pf, input) in built {
                            existing.insert(pf, input);
                        }
                    }
                    None => producer.historical_tone = Some(built),
                }
            }
        }

        // 5. Suit Makeup: 90033
        if active(90033) {
            if let Ok(passes) =
                super::portrait_pixcake_unify_suit::build_pf90033_owned_suit_passes(
                    source,
                    &self.points194,
                    face_ref,
                    extra,
                )
            {
                producer.owned_suit_passes = Some(passes);
            }
        }

        Ok(())
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
        let digest = self.digest_for(source);
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
            // PF90086 binds M/O as three linear f32 channels per pixel
            // (see kernels::makeup::body_tone_unify). The parsing body mask is
            // scalar-per-pixel; expand it to RGB and keep O zero. This is a
            // derived M/O plane, not a recovered vendor model output.
            let mut m = Vec::with_capacity(pixels * 3);
            for &weight in &body_mask_f32 {
                m.extend_from_slice(&[weight, weight, weight]);
            }
            let o = vec![0.0f32; pixels * 3];
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

    /// Attaches same-frame hair detail producer inputs (`PF90139` hair smooth,
    /// `PF90160` hair gloss) derived from `PortraitMasks::hair` and the current frame.
    pub fn attach_hair_detail_from_extra(
        &self,
        source: &Image,
        masks: Option<&super::PortraitMasks>,
        extra: &std::collections::BTreeMap<String, f32>,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        let c90139 = extra.get("pc:90139").copied().unwrap_or(0.0);
        let c90160 = extra.get("pc:90160").copied().unwrap_or(0.0);
        if !c90139.is_finite() {
            return Err(Refusal::ValueOutOfRange("PF90139_DETAIL_STRENGTH"));
        }
        if !c90160.is_finite() {
            return Err(Refusal::ValueOutOfRange("PF90160_ENABLED"));
        }
        if c90139 <= 0.0 && c90160 <= 0.0 {
            return Ok(());
        }
        let Some(hair_f32) = masks.and_then(|m| {
            m.hair
                .as_ref()
                .map(|plane| sample_mask_plane(plane, m.width, m.height, source.width, source.height))
        }) else {
            return Ok(());
        };

        let w = source.width as usize;
        let h = source.height as usize;
        let mut min_x = w;
        let mut min_y = h;
        let mut max_x = 0usize;
        let mut max_y = 0usize;
        for y in 0..h {
            let row = y * w;
            for x in 0..w {
                let v = (hair_f32[row + x].clamp(0.0, 1.0) * 255.0).round() as u8;
                if v > 0 {
                    min_x = min_x.min(x);
                    min_y = min_y.min(y);
                    max_x = max_x.max(x);
                    max_y = max_y.max(y);
                }
            }
        }
        if min_x > max_x || min_y > max_y {
            return Ok(());
        }
        let roi_x = min_x as u32;
        let roi_y = min_y as u32;
        let roi_w = (max_x - min_x + 1) as u32;
        let roi_h = (max_y - min_y + 1) as u32;
        let rw = roi_w as usize;
        let rh = roi_h as usize;
        let roi_pixels = rw * rh;

        let mut hair_mask_roi = vec![0u8; roi_pixels];
        let mut src_roi_rgb = vec![0u8; roi_pixels * 3];
        for y in 0..rh {
            let sy = min_y + y;
            for x in 0..rw {
                let sx = min_x + x;
                let dst_idx = y * rw + x;
                hair_mask_roi[dst_idx] =
                    (hair_f32[sy * w + sx].clamp(0.0, 1.0) * 255.0).round() as u8;
                let sbase = (sy * w + sx) * 4;
                let dbase = dst_idx * 3;
                src_roi_rgb[dbase] = source.pixels[sbase];
                src_roi_rgb[dbase + 1] = source.pixels[sbase + 1];
                src_roi_rgb[dbase + 2] = source.pixels[sbase + 2];
            }
        }

        // Two-pass 3x3 edge-preserving bilateral smooth within the hair ROI.
        let mut cur_rgb = src_roi_rgb.clone();
        let mut next_rgb = vec![0u8; roi_pixels * 3];
        const INV_TWO_SIGMA_R2: f32 = 1.0 / (2.0 * 28.0 * 28.0);
        for _ in 0..2 {
            for y in 0..rh {
                let y0 = y.saturating_sub(1);
                let y1 = (y + 1).min(rh - 1);
                for x in 0..rw {
                    let idx = y * rw + x;
                    let base = idx * 3;
                    if hair_mask_roi[idx] == 0 {
                        next_rgb[base..base + 3].copy_from_slice(&cur_rgb[base..base + 3]);
                        continue;
                    }
                    let cr = cur_rgb[base] as f32;
                    let cg = cur_rgb[base + 1] as f32;
                    let cb = cur_rgb[base + 2] as f32;
                    let cl = 0.299 * cr + 0.587 * cg + 0.114 * cb;
                    let x0 = x.saturating_sub(1);
                    let x1 = (x + 1).min(rw - 1);
                    let mut sum = [0.0f32; 3];
                    let mut wsum = 0.0f32;
                    for ny in y0..=y1 {
                        let nrow = ny * rw;
                        for nx in x0..=x1 {
                            let nidx = nrow + nx;
                            let mask_w = hair_mask_roi[nidx] as f32 * (1.0 / 255.0);
                            if mask_w <= 0.0 {
                                continue;
                            }
                            let nbase = nidx * 3;
                            let nr = cur_rgb[nbase] as f32;
                            let ng = cur_rgb[nbase + 1] as f32;
                            let nb = cur_rgb[nbase + 2] as f32;
                            let nl = 0.299 * nr + 0.587 * ng + 0.114 * nb;
                            let dl = nl - cl;
                            let spatial = if nx == x && ny == y {
                                4.0
                            } else if nx == x || ny == y {
                                2.0
                            } else {
                                1.0
                            };
                            let weight = spatial * mask_w * (-dl * dl * INV_TWO_SIGMA_R2).exp();
                            sum[0] += nr * weight;
                            sum[1] += ng * weight;
                            sum[2] += nb * weight;
                            wsum += weight;
                        }
                    }
                    if wsum > 1e-6 {
                        next_rgb[base] = (sum[0] / wsum).round().clamp(0.0, 255.0) as u8;
                        next_rgb[base + 1] = (sum[1] / wsum).round().clamp(0.0, 255.0) as u8;
                        next_rgb[base + 2] = (sum[2] / wsum).round().clamp(0.0, 255.0) as u8;
                    } else {
                        next_rgb[base..base + 3].copy_from_slice(&cur_rgb[base..base + 3]);
                    }
                }
            }
            std::mem::swap(&mut cur_rgb, &mut next_rgb);
        }
        let smooth_rgb = cur_rgb;

        if c90139 > 0.0 {
            let mut detail = vec![128u8; roi_pixels];
            for i in 0..roi_pixels {
                if hair_mask_roi[i] == 0 {
                    continue;
                }
                let base = i * 3;
                let l_src = 0.299 * src_roi_rgb[base] as f32
                    + 0.587 * src_roi_rgb[base + 1] as f32
                    + 0.114 * src_roi_rgb[base + 2] as f32;
                let l_sm = 0.299 * smooth_rgb[base] as f32
                    + 0.587 * smooth_rgb[base + 1] as f32
                    + 0.114 * smooth_rgb[base + 2] as f32;
                let hf = (l_src - l_sm) * 0.10;
                detail[i] = (127.5 + hf.clamp(-8.0, 8.0)).round().clamp(0.0, 255.0) as u8;
            }
            let mix_strength = (c90139.clamp(0.0, 1.0) * 0.65).clamp(0.0, 1.0);
            producer.hair_smooth = Some(super::HairSmoothProducerInputs {
                ticket: self.job_id,
                width: source.width,
                height: source.height,
                roi: (roi_x, roi_y, roi_w, roi_h),
                mix_strength,
                hair_mask: hair_mask_roi.clone(),
                mix_mask: hair_mask_roi.clone(),
                blend_rgb: smooth_rgb.clone(),
                detail,
                hair_stride: roi_w,
                mix_stride: roi_w,
                blend_stride: roi_w * 3,
                detail_stride: roi_w,
            });
        }

        if c90160 > 0.0 {
            let mut luma_sum = 0.0f64;
            let mut w_sum = 0.0f64;
            for i in 0..roi_pixels {
                let mw = f64::from(hair_mask_roi[i]);
                if mw > 0.0 {
                    let base = i * 3;
                    let l_sm = 0.299 * f64::from(smooth_rgb[base])
                        + 0.587 * f64::from(smooth_rgb[base + 1])
                        + 0.114 * f64::from(smooth_rgb[base + 2]);
                    luma_sum += l_sm * mw;
                    w_sum += mw;
                }
            }
            let mean_luma = if w_sum > 0.0 {
                (luma_sum / w_sum) as f32
            } else {
                96.0
            };
            let pivot = mean_luma * 0.85;
            let denom = (255.0 - pivot).max(32.0);
            let mut model_detail = vec![128u8; roi_pixels];
            for i in 0..roi_pixels {
                if hair_mask_roi[i] == 0 {
                    continue;
                }
                let base = i * 3;
                let l_sm = 0.299 * smooth_rgb[base] as f32
                    + 0.587 * smooth_rgb[base + 1] as f32
                    + 0.114 * smooth_rgb[base + 2] as f32;
                let rel = ((l_sm - pivot) / denom).clamp(0.0, 1.0);
                let smoothstep = rel * rel * (3.0 - 2.0 * rel);
                let gloss = 0.18 + 0.82 * smoothstep;
                model_detail[i] = (127.5 + gloss * 24.0).round().clamp(128.0, 180.0) as u8;
            }
            producer.hair_gloss = Some(super::HairGlossProducerInputs {
                ticket: self.job_id,
                width: source.width,
                height: source.height,
                roi: (roi_x, roi_y, roi_w, roi_h),
                dynamic_scale: c90160.clamp(0.0, 1.0),
                hair_mask: hair_mask_roi,
                model_detail,
                hair_stride: roi_w,
                detail_stride: roi_w,
            });
        }

        Ok(())
    }

    /// Attaches same-frame `PF90004` (皮肤透亮 / Skin radiance) producer inputs
    /// from `FaceParsingV2` and `PortraitMasks`.
    pub fn attach_pf90004_from_extra(
        &self,
        source: &Image,
        masks: Option<&super::PortraitMasks>,
        extra: &std::collections::BTreeMap<String, f32>,
        producer: &mut ProducerInputs,
    ) -> Result<(), Refusal> {
        self.validate_ticket(source, Some(self.job_id))?;
        let c90004 = extra.get("pc:90004").copied().unwrap_or(0.0);
        if !c90004.is_finite() {
            return Err(Refusal::ValueOutOfRange("PF90004_DEGREE"));
        }
        if c90004 <= 0.0 {
            return Ok(());
        }
        if let Some(existing) = self
            .model_inputs
            .as_ref()
            .and_then(|inputs| inputs.pf90004.as_ref())
        {
            producer.write_pf90004(existing.clone());
            return Ok(());
        }
        let parsing = self
            .model_inputs
            .as_ref()
            .and_then(|inputs| inputs.parsing.as_ref())
            .ok_or(Refusal::MissingInput("FaceParsingV2 parsing planes"))?;
        let mut parsing = parsing.clone();
        parsing.job_id = self.job_id;
        let slots = parsing.consumer_slots()?;
        let det = parsing.source_to_crop[0][0] * parsing.source_to_crop[1][1]
            - parsing.source_to_crop[0][1] * parsing.source_to_crop[1][0];
        if !det.is_finite() || det.abs() <= 1.0e-12 {
            return Err(Refusal::ShapeMismatch("PF90004_PARSING_MATRIX"));
        }

        fn project_u8_plane_bilinear(
            plane: &[u8],
            crop_w: u32,
            crop_h: u32,
            source_to_crop: [[f32; 3]; 2],
            dst_w: u32,
            dst_h: u32,
        ) -> Vec<u8> {
            let cw = crop_w as usize;
            let ch = crop_h as usize;
            let dw = dst_w as usize;
            let dh = dst_h as usize;
            let mut out = vec![0u8; dw.saturating_mul(dh)];
            if cw == 0 || ch == 0 || plane.len() < cw * ch {
                return out;
            }
            let max_cx = (cw - 1) as f32;
            let max_cy = (ch - 1) as f32;
            for y in 0..dh {
                let sy = y as f32;
                let row_off = y * dw;
                for x in 0..dw {
                    let sx = x as f32;
                    let cx = source_to_crop[0][0] * sx + source_to_crop[0][1] * sy + source_to_crop[0][2];
                    let cy = source_to_crop[1][0] * sx + source_to_crop[1][1] * sy + source_to_crop[1][2];
                    if cx < 0.0 || cy < 0.0 || cx > max_cx || cy > max_cy {
                        continue;
                    }
                    let x0 = cx.floor() as usize;
                    let y0 = cy.floor() as usize;
                    let x1 = (x0 + 1).min(cw - 1);
                    let y1 = (y0 + 1).min(ch - 1);
                    let fx = cx - x0 as f32;
                    let fy = cy - y0 as f32;
                    let v00 = plane[y0 * cw + x0] as f32;
                    let v10 = plane[y0 * cw + x1] as f32;
                    let v01 = plane[y1 * cw + x0] as f32;
                    let v11 = plane[y1 * cw + x1] as f32;
                    let v = (v00 * (1.0 - fx) + v10 * fx) * (1.0 - fy)
                        + (v01 * (1.0 - fx) + v11 * fx) * fy;
                    out[row_off + x] = v.round().clamp(0.0, 255.0) as u8;
                }
            }
            out
        }

        fn pf90004_value_mean_rgba(rgba: &[u8], width: u32, height: u32, skin_mask: &[u8]) -> f32 {
            let pixels = (width as usize).saturating_mul(height as usize);
            if pixels == 0 || skin_mask.len() != pixels || rgba.len() < pixels * 4 {
                return 0.0;
            }
            let mut sum = 0.0f64;
            let mut count = 0u64;
            for i in 0..pixels {
                if skin_mask[i] > 0 {
                    sum += f64::from(rgba[i * 4]);
                    count += 1;
                }
            }
            if count == 0 { 0.0 } else { (sum / count as f64) as f32 }
        }

        let project = |plane: &[u8]| {
            project_u8_plane_bilinear(
                plane,
                parsing.width,
                parsing.height,
                parsing.source_to_crop,
                source.width,
                source.height,
            )
        };
        let p0 = project(&slots[17]);
        let p1 = project(&slots[18]);
        let p2 = project(&slots[19]);
        let p3 = project(&slots[20]);
        let face = project(&slots[1]);

        let pixels = source.pixel_count();
        let mut parsing_rgba = Vec::with_capacity(pixels * 4);
        let mut face_rgba = Vec::with_capacity(pixels * 4);
        for i in 0..pixels {
            parsing_rgba.extend_from_slice(&[p0[i], p1[i], p2[i], p3[i]]);
            let fv = face[i];
            face_rgba.extend_from_slice(&[fv, fv, p2[i], fv]);
        }

        let (skin_mask, value_mean) = if let Some(auto) = masks.and_then(|m| {
            m.pf90004_parser_inputs(
                &source.pixels,
                source.width,
                source.height,
                crate::NormRect::FULL,
            )
            .ok()
        }) {
            (auto.skin_mask, auto.value_mean)
        } else {
            let skin = project(&slots[crate::face_parsing::FACE_PARSING_V2_SKIN_CLASS]);
            let skin_mask = if skin.iter().any(|&v| v > 0) {
                skin
            } else {
                face
            };
            let mean = pf90004_value_mean_rgba(
                &source.pixels,
                source.width,
                source.height,
                &skin_mask,
            );
            (skin_mask, mean)
        };

        let bundle = super::Pf90004ProducerInputs::from_same_job(
            source.width,
            source.height,
            skin_mask,
            parsing_rgba,
            face_rgba,
            value_mean,
            crate::NormRect::FULL,
        )?;
        producer.write_pf90004(bundle);
        Ok(())
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
        // Build crown arc [192..207] above forehead/temples for PF232 (points414[192..207] are mouth/chin, not skull crown).
        let mut skull_landmarks = vec![[0.0f32, 0.0f32]; 207];
        skull_landmarks[..194.min(self.points194.len())]
            .copy_from_slice(&self.points194[..194.min(self.points194.len())]);
        let eye_l = self.points194[26];
        let eye_r = self.points194[44];
        let eye_mid = [(eye_l[0] + eye_r[0]) * 0.5, (eye_l[1] + eye_r[1]) * 0.5];
        let chin = self.points194[125];
        let up_dx = eye_mid[0] - chin[0];
        let up_dy = eye_mid[1] - chin[1];
        let face_h = up_dx.hypot(up_dy).max(16.0);
        let (ux, uy) = (up_dx / face_h, up_dy / face_h); // toward forehead/crown
        let (rx, ry) = (eye_r[0] - eye_l[0], eye_r[1] - eye_l[1]);
        let eye_d = rx.hypot(ry).max(12.0);
        let (tx, ty) = (rx / eye_d, ry / eye_d); // left-to-right across face
        let crown_anchor = [eye_mid[0] + ux * face_h * 0.52, eye_mid[1] + uy * face_h * 0.52];
        skull_landmarks[192] = crown_anchor;
        let crown_rx = eye_d * 1.25;
        let crown_ry = face_h * 0.90;
        for k in 0..14usize {
            let theta = std::f32::consts::PI * (0.15 + 0.70 * (k as f32 / 13.0));
            let c = -theta.cos();
            let s = theta.sin();
            skull_landmarks[193 + k] = [
                crown_anchor[0] + tx * (c * crown_rx) + ux * (s * crown_ry),
                crown_anchor[1] + ty * (c * crown_rx) + uy * (s * crown_ry),
            ];
        }
        let skull = aurora_portrait::kernels::face_detail::skull_top(
            controls232[0] * 100.0,
            &skull_landmarks,
            factors15,
            source.width as f32,
            source.height as f32,
        )?;
        let digest = self.digest_for(source);
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
        let teeth = match inputs.teeth.as_ref() {
            Some(teeth) => Some(super::TeethProducerInputs {
                job_id: self.job_id,
                bright_ui,
                deyellow_ui,
                ..teeth.clone()
            }),
            None if bright_ui > 0.0 || deyellow_ui > 0.0 => {
                return Err(Refusal::MissingInput("PF108 teeth mask or LUTs"));
            }
            None => None,
        };
        parsing.job_id = self.job_id;
        if producer.parsing.is_none() {
            producer.parsing = Some(parsing);
        }
        if producer.teeth.is_none() {
            producer.teeth = teeth;
        }
        if producer.pf90004.is_none() {
            producer.pf90004 = inputs.pf90004.clone();
        }
        Ok(())
    }

    /// Attaches PF30013 (中性灰磨皮 FlatGray) using the canonical `fbe.onnx`
    /// FaceParsingV2 facial-skin mask projected to the full source frame.
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
        let any_eye_brow = (8001u32..=8022)
            .chain([30073u32, 30079, 44124, 48019, 41922, 49221])
            .any(|pf| {
                extra
                    .get(&format!("pc:{pf}"))
                    .copied()
                    .unwrap_or(0.0)
                    != 0.0
            });
        if !any_eye_brow {
            return Ok(());
        }
        let w = source.width as f32;
        let h = source.height as f32;
        let digest = self.digest_for(source);
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

        // Helper: build a pinned-boundary 5x5 local orbital grid per eye so interior warps
        // blend smoothly to zero at the outer boundary without hard square edges.
        let build_eye_grid = |eye_ring: &[usize], center_idx: usize, half_w_scale: f32, half_h_scale: f32| -> (Vec<Point>, Vec<f32>, Point, Point, f32, f32) {
            let center = points194[center_idx];
            let p_l = points194[eye_ring[0]];
            let p_r = points194[eye_ring[4]];
            let dx = p_r.x - p_l.x;
            let dy = p_r.y - p_l.y;
            let eye_w = dx.hypot(dy).max(8.0);
            let (mut ux, mut uy) = if eye_w > 1e-3 { (dx / eye_w, dy / eye_w) } else { (1.0, 0.0) };
            if ux < 0.0 {
                ux = -ux;
                uy = -uy;
            }
            let (vx, vy) = (-uy, ux);
            let half_w = (eye_w * half_w_scale).max(8.0);
            let half_h = (eye_w * half_h_scale).max(6.0);
            let mut pts = Vec::with_capacity(25);
            let mut weights = Vec::with_capacity(25);
            for row in 0..5 {
                let ny = row as f32 / 2.0 - 1.0; // -1.0..=1.0
                for col in 0..5 {
                    let nx = col as f32 / 2.0 - 1.0;
                    let r = nx.hypot(ny).min(1.0);
                    let w_falloff = if row == 0 || row == 4 || col == 0 || col == 4 {
                        0.0
                    } else {
                        0.5 * (1.0 + (std::f32::consts::PI * r).cos())
                    };
                    pts.push(Point::new(
                        center.x + nx * half_w * ux + ny * half_h * vx,
                        center.y + nx * half_w * uy + ny * half_h * vy,
                    ));
                    weights.push(w_falloff);
                }
            }
            (pts, weights, Point::new(ux, uy), Point::new(vx, vy), half_w, half_h)
        };
        let mut eye_grid_ebo = Vec::with_capacity(2 * 32 * 3);
        for side in 0..2u16 {
            let base = side * 25;
            for r in 0..4u16 {
                for c in 0..4u16 {
                    let a = base + r * 5 + c;
                    let b = a + 1;
                    let d = a + 5;
                    let e = d + 1;
                    eye_grid_ebo.extend_from_slice(&[a, b, d, b, e, d]);
                }
            }
        }

        // Helper for two-eye grid effects: 8011/8012 (pupil size), 8013/8014 (eye height), 8019/8020 (eye width)
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
            match mode {
                0 => {
                    let _ = eye_geometry::row11_pupil_size(
                        &points194, iw, ih, unit_scale, l_deg, r_deg, 1.0, 1.0,
                    );
                }
                1 => {
                    let _ = eye_geometry::row12_eye_height(
                        &points194, iw, ih, unit_scale, l_deg, r_deg, 1.0,
                    );
                }
                _ => {
                    let _ = eye_geometry::row13_eye_width(
                        &points194, iw, ih, unit_scale, l_deg, r_deg, eye_geometry::ROW13_FACTOR_LIST4,
                    );
                }
            }
            let (l_pts, l_w, l_u, l_v, l_hw, l_hh) = build_eye_grid(&[18, 19, 20, 21, 22, 23, 24, 25], 26, 0.95, 0.72);
            let (r_pts, r_w, r_u, r_v, r_hw, r_hh) = build_eye_grid(&[36, 37, 38, 39, 40, 41, 42, 43], 44, 0.95, 0.72);
            let mut dest = Vec::with_capacity(50);
            let mut uv = Vec::with_capacity(50);
            for (pts, weights, axis_u, axis_v, hw, hh, center, deg01) in [
                (l_pts, l_w, l_u, l_v, l_hw, l_hh, points194[26], c_left),
                (r_pts, r_w, r_u, r_v, r_hw, r_hh, points194[44], c_right),
            ] {
                for (pt, weight) in pts.iter().zip(weights.iter()) {
                    dest.push([pt.x, pt.y]);
                    let rx = pt.x - center.x;
                    let ry = pt.y - center.y;
                    let lu = rx * axis_u.x + ry * axis_u.y;
                    let lv = rx * axis_v.x + ry * axis_v.y;
                    let (nu, nv) = match mode {
                        0 => {
                            let s = (1.0 - 0.34 * deg01 * weight).clamp(0.55, 1.45);
                            (lu * s, lv * s)
                        }
                        1 => {
                            let sy = (1.0 - 0.30 * deg01 * weight).clamp(0.55, 1.45);
                            (lu, lv * sy)
                        }
                        _ => {
                            let sx = (1.0 - 0.28 * deg01 * weight).clamp(0.55, 1.45);
                            (lu * sx, lv)
                        }
                    };
                    let sx = center.x + nu.clamp(-hw, hw) * axis_u.x + nv.clamp(-hh, hh) * axis_v.x;
                    let sy = center.y + nu.clamp(-hw, hw) * axis_u.y + nv.clamp(-hh, hh) * axis_v.y;
                    uv.push([(sx / w).clamp(0.0, 1.0), (sy / h).clamp(0.0, 1.0)]);
                }
            }
            draws.push(super::FaceGeometryDraw {
                pf: left_pf,
                controls: [c_left, c_right],
                source_sha256: digest,
                geometry: super::FaceGeometry::Eye {
                    dest,
                    uv,
                    ebo: eye_grid_ebo.clone(),
                    rules,
                    edge: FragmentEdge::ReshapeNormal,
                    defines: Defines::plain(),
                    clamp: Clamp::disabled(),
                },
            });
        }

        // Smooth orbital grid eye-pair effects: 8015/8016 (eye distance), 8017/8018 (eye angle), 8021/8022 (eye position)
        let ex_dx = points194[44].x - points194[26].x;
        let ex_dy = points194[44].y - points194[26].y;
        let ex_len = ex_dx.hypot(ex_dy).max(12.0);
        let eye_pair_axis = Point::new(ex_dx / ex_len, ex_dy / ex_len);
        let eye_pair_up = Point::new(eye_pair_axis.y, -eye_pair_axis.x);
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
            let (l_pts, l_w, _, _, l_hw, _) = build_eye_grid(&[18, 19, 20, 21, 22, 23, 24, 25], 26, 1.15, 0.85);
            let (r_pts, r_w, _, _, r_hw, _) = build_eye_grid(&[36, 37, 38, 39, 40, 41, 42, 43], 44, 1.15, 0.85);
            let mut dest = Vec::with_capacity(50);
            let mut uv = Vec::with_capacity(50);
            for (side_idx, (pts, weights, center, hw, deg01)) in [
                (l_pts, l_w, points194[26], l_hw, c_left),
                (r_pts, r_w, points194[44], r_hw, c_right),
            ]
            .into_iter()
            .enumerate()
            {
                let angle = deg01 * 0.22 * if side_idx == 0 { -1.0 } else { 1.0 };
                let shift_h = deg01 * hw * 0.26 * if side_idx == 0 { -1.0 } else { 1.0 };
                let shift_v = -deg01 * hw * 0.22;
                for (pt, weight) in pts.iter().zip(weights.iter()) {
                    uv.push([(pt.x / w).clamp(0.0, 1.0), (pt.y / h).clamp(0.0, 1.0)]);
                    let (dx, dy) = match mode {
                        0 => (eye_pair_axis.x * shift_h * weight, eye_pair_axis.y * shift_h * weight),
                        1 => {
                            let (s, c) = (angle * weight).sin_cos();
                            let rx = pt.x - center.x;
                            let ry = pt.y - center.y;
                            ((rx * c - ry * s) - rx, (rx * s + ry * c) - ry)
                        }
                        _ => (eye_pair_up.x * shift_v * weight, eye_pair_up.y * shift_v * weight),
                    };
                    dest.push([pt.x + dx, pt.y + dy]);
                }
            }
            draws.push(super::FaceGeometryDraw {
                pf: left_pf,
                controls: [c_left, c_right],
                source_sha256: digest,
                geometry: super::FaceGeometry::Eye {
                    dest,
                    uv,
                    ebo: eye_grid_ebo.clone(),
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
        // Build the 168-point brow prepared mesh (3 bands of 28 points per brow) from real brow rings `0..=8` and `9..=17`.
        let mean_pts = |indices: std::ops::RangeInclusive<usize>| {
            let mut sx = 0.0f32;
            let mut sy = 0.0f32;
            let count = (indices.end() - indices.start() + 1) as f32;
            for i in indices {
                sx += points194[i].x;
                sy += points194[i].y;
            }
            Point::new(sx / count, sy / count)
        };
        let left_brow_center = mean_pts(0..=8);
        let right_brow_center = mean_pts(9..=17);
        let brow_span = distance(
            [left_brow_center.x, left_brow_center.y],
            [right_brow_center.x, right_brow_center.y],
        )
        .max(24.0);
        let brow_band_h = (brow_span * 0.16).max(8.0);
        let brow_axis = |tail: Point, head: Point, fallback_x: f32| {
            let dx = head.x - tail.x;
            let dy = head.y - tail.y;
            let len = dx.hypot(dy);
            if len > 1e-3 && len.is_finite() {
                let mut tx = dx / len;
                let mut ty = dy / len;
                if tx < 0.0 {
                    tx = -tx;
                    ty = -ty;
                }
                let (mut nx, mut ny) = (ty, -tx); // points upward toward forehead
                if ny > 0.0 {
                    nx = -nx;
                    ny = -ny;
                }
                (Point::new(tx, ty), Point::new(nx, ny), (len * 0.65).max(12.0))
            } else {
                (Point::new(fallback_x.abs(), 0.0), Point::new(0.0, -1.0), (brow_span * 0.30).max(12.0))
            }
        };
        let (l_tx, l_up, l_hw) = brow_axis(points194[0], points194[4], 1.0);
        let (r_tx, r_up, r_hw) = brow_axis(points194[9], points194[13], 1.0);
        let mut prepared168 = Vec::with_capacity(168);
        for (center, axis_t, axis_up, hw) in [
            (left_brow_center, l_tx, l_up, l_hw),
            (right_brow_center, r_tx, r_up, r_hw),
        ] {
            for band in 0..3 {
                // band 0 = top pinned boundary (+up), band 1 = brow center, band 2 = bottom pinned boundary (-up)
                let up_off = (1.0 - band as f32) * brow_band_h;
                for col in 0..28 {
                    let t = col as f32 / 27.0 - 0.5;
                    let s = t * 2.0 * hw;
                    let arch = (1.0 - (t * 2.0).powi(2)) * brow_band_h * 0.22;
                    prepared168.push(Point::new(
                        center.x + axis_t.x * s + axis_up.x * (up_off + arch),
                        center.y + axis_t.y * s + axis_up.y * (up_off + arch),
                    ));
                }
            }
        }
        let uv168: Vec<[f32; 2]> = prepared168
            .iter()
            .map(|p| [(p.x / w).clamp(0.0, 1.0), (p.y / h).clamp(0.0, 1.0)])
            .collect();
        let brow_dx = right_brow_center.x - left_brow_center.x;
        let brow_dy = right_brow_center.y - left_brow_center.y;
        let brow_axis_len = brow_dx.hypot(brow_dy).max(1e-3);
        let pair_tx = Point::new(brow_dx / brow_axis_len, brow_dy / brow_axis_len);

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
            let mut target = prepared168.clone();
            match mode {
                0 => {
                    // Brow thickness: build a 4-band mesh? Within 3-band mesh, shift band 1 slightly or expand inner arch cleanly.
                    // Actually pinch/expand between band 0/2 around band 1 by splitting band 1 vertically using UV warp below;
                    // for vertex target, displace band 1 upward slightly + band 0/2 inward for thinning / outward wait: keep band 0/2 pinned,
                    // deform band 1 is position; for thickness, emit 4-band (224 vert) mesh where bands 1 & 2 separate!
                    let mut thick_ebo = Vec::new();
                    for side in 0..2u16 {
                        let base = side * 112;
                        for band in 0..3u16 {
                            let r0 = base + band * 28;
                            let r1 = r0 + 28;
                            for col in 0..27u16 {
                                let a = r0 + col;
                                let b = r0 + col + 1;
                                let c = r1 + col;
                                let d = r1 + col + 1;
                                thick_ebo.extend_from_slice(&[a, b, c, b, d, c]);
                            }
                        }
                    }
                    let mut src_224 = Vec::with_capacity(224);
                    let mut dst_224 = Vec::with_capacity(224);
                    for (center, axis_t, axis_up, hw, deg01) in [
                        (left_brow_center, l_tx, l_up, l_hw, c_left),
                        (right_brow_center, r_tx, r_up, r_hw, c_right),
                    ] {
                        let bands = [1.0f32, 0.25, -0.25, -1.0];
                        let shifts = [0.0f32, 0.28, -0.28, 0.0];
                        for (up_frac, shift_frac) in bands.into_iter().zip(shifts) {
                            for col in 0..28 {
                                let t = col as f32 / 27.0 - 0.5;
                                let win = (col as f32 / 27.0 * std::f32::consts::PI).sin();
                                let s = t * 2.0 * hw;
                                let arch = (1.0 - (t * 2.0).powi(2)) * brow_band_h * 0.22;
                                let base_pt = Point::new(
                                    center.x + axis_t.x * s + axis_up.x * (up_frac * brow_band_h + arch),
                                    center.y + axis_t.y * s + axis_up.y * (up_frac * brow_band_h + arch),
                                );
                                src_224.push(base_pt);
                                let d = deg01 * brow_band_h * shift_frac * win;
                                dst_224.push(Point::new(base_pt.x + axis_up.x * d, base_pt.y + axis_up.y * d));
                            }
                        }
                    }
                    draws.push(super::FaceGeometryDraw {
                        pf: left_pf,
                        controls: [c_left, c_right],
                        source_sha256: digest,
                        geometry: super::FaceGeometry::Eye {
                            dest: dst_224.iter().map(|p| [p.x, p.y]).collect(),
                            uv: src_224.iter().map(|p| [(p.x / w).clamp(0.0, 1.0), (p.y / h).clamp(0.0, 1.0)]).collect(),
                            ebo: thick_ebo,
                            rules,
                            edge: FragmentEdge::ReshapeNormal,
                            defines: Defines::plain(),
                            clamp: Clamp::disabled(),
                        },
                    });
                    continue;
                }
                1 => {
                    for (side_idx, (axis_up, deg01)) in [(l_up, c_left), (r_up, c_right)].into_iter().enumerate() {
                        let base = side_idx * 84 + 28;
                        let disp = deg01 * brow_band_h * 0.55;
                        for col in 0..28 {
                            let weight = (col as f32 / 27.0 * std::f32::consts::PI).sin();
                            target[base + col].x += axis_up.x * disp * weight;
                            target[base + col].y += axis_up.y * disp * weight;
                        }
                    }
                }
                2 => {
                    // Brow distance: positive moves brows further apart (left -, right + along pair_tx)
                    for (side_idx, (hw, deg01)) in [(l_hw, c_left), (r_hw, c_right)].into_iter().enumerate() {
                        let base = side_idx * 84 + 28;
                        let sign = if side_idx == 0 { -1.0 } else { 1.0 };
                        let disp = sign * deg01 * hw * 0.30;
                        for col in 0..28 {
                            let weight = (col as f32 / 27.0 * std::f32::consts::PI).sin();
                            target[base + col].x += pair_tx.x * disp * weight;
                            target[base + col].y += pair_tx.y * disp * weight;
                        }
                    }
                }
                3 => {
                    for (side_idx, (center, deg01)) in [(left_brow_center, c_left), (right_brow_center, c_right)].into_iter().enumerate() {
                        let base = side_idx * 84 + 28;
                        let angle = deg01 * 0.18 * if side_idx == 0 { -1.0 } else { 1.0 };
                        let (s, c) = angle.sin_cos();
                        for col in 0..28 {
                            let weight = (col as f32 / 27.0 * std::f32::consts::PI).sin();
                            let dx = target[base + col].x - center.x;
                            let dy = target[base + col].y - center.y;
                            let rx = center.x + dx * c - dy * s;
                            let ry = center.y + dx * s + dy * c;
                            target[base + col].x += (rx - target[base + col].x) * weight;
                            target[base + col].y += (ry - target[base + col].y) * weight;
                        }
                    }
                }
                _ => {
                    // Brow shape: single smooth unimodal arch curve along true brow_up normal (no M-dip).
                    for (side_idx, (axis_up, deg01)) in [(l_up, c_left), (r_up, c_right)].into_iter().enumerate() {
                        let base = side_idx * 84 + 28;
                        for col in 0..28 {
                            let u = col as f32 / 27.0;
                            let w_arch = (std::f32::consts::PI * u).sin().max(0.0).powf(1.25);
                            let disp = deg01 * brow_band_h * 0.65 * w_arch;
                            target[base + col].x += axis_up.x * disp;
                            target[base + col].y += axis_up.y * disp;
                        }
                    }
                }
            }
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
                let (c_a, c_b) = if side == 0 { (18usize, 22usize) } else { (40usize, 36usize) };
                let eye_center = midpoint(
                    [points194[c_a].x, points194[c_a].y],
                    [points194[c_b].x, points194[c_b].y],
                );
                let eye_w = distance(
                    [points194[c_a].x, points194[c_a].y],
                    [points194[c_b].x, points194[c_b].y],
                )
                .max(12.0);
                let eye_h = (eye_w * 0.38).max(5.0);
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
            let center_1011 = [
                (points194[57].x / w).clamp(0.0, 1.0),
                (points194[57].y / h).clamp(0.0, 1.0),
            ];
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

        // PF44124 / PF48019 (eyebrowHeadLength) and PF49221 / PF41922 (eyebrowTailLength)
        // via proven Type1018 dual-side 192-vertex mesh + 996-index EBO + SHORT_TABLES / LONG_TABLES.
        let c_head_l = control(44124);
        let c_head_r = control(48019);
        let c_tail_l = control(49221);
        let c_tail_r = control(41922);
        if c_head_l != 0.0 || c_head_r != 0.0 || c_tail_l != 0.0 || c_tail_r != 0.0 {
            use aurora_portrait::kernels::brow_length_exact_vendor::{
                self as brow_vendor, BrowGuide, BrowPart, Point as VendorPoint,
            };

            // Canonical 194-point eyebrow anchors (`0..=8` left brow tail..head, `9..=17` right brow tail..head)
            // and `109..=141` jaw/face contour.
            let build_side_guide = |head: Point, mid: Point, tail: Point, fallback_dir_x: f32| {
                let dx = tail.x - head.x;
                let dy = tail.y - head.y;
                let len = (dx * dx + dy * dy).sqrt();
                if len > 1e-3 && len.is_finite() {
                    let guide = BrowGuide {
                        start: VendorPoint::new(head.x, head.y),
                        center: VendorPoint::new(mid.x, mid.y),
                        end: VendorPoint::new(tail.x, tail.y),
                    };
                    (guide, [dx / len, dy / len], len * 0.5)
                } else {
                    let half = (brow_span * 0.28).max(8.0);
                    let start = VendorPoint::new(mid.x - fallback_dir_x * half, mid.y);
                    let end = VendorPoint::new(mid.x + fallback_dir_x * half, mid.y);
                    let guide = BrowGuide {
                        start,
                        center: VendorPoint::new(mid.x, mid.y),
                        end,
                    };
                    (guide, [fallback_dir_x, 0.0], half)
                }
            };
            let left_mid = Point::new(
                (points194[2].x + points194[6].x) * 0.5,
                (points194[2].y + points194[6].y) * 0.5,
            );
            let right_mid = Point::new(
                (points194[11].x + points194[15].x) * 0.5,
                (points194[11].y + points194[15].y) * 0.5,
            );
            let (guide_left, axis_l, half_l) =
                build_side_guide(points194[4], left_mid, points194[0], -1.0);
            let (guide_right, axis_r, half_r) =
                build_side_guide(points194[13], right_mid, points194[9], 1.0);

            // Build 192-vertex dual-side source mesh: left brow `0..96` (4 rings x 24)
            // and right brow `96..192` (4 rings x 24), where Ring 3 is the outer border
            // anchor ring (zero displacement under `0x707ED50`).
            // INFERRED: ring expansion ratios `[1.0, 1.6, 2.1, 2.5]` follow `0x707F550`
            // outer-ring ratios over a closed 24-point brow contour; exact Catmull-Rom
            // control-point mapping from `points194` to `face_info+0x924` awaits live parity.
            const RING_SCALES: [f32; 4] = [1.0, 1.6, 2.1, 2.5];
            let mut src_192 = Vec::with_capacity(eye_geometry::TYPE1018_VERTEX_COUNT);
            for (guide, axis, half_len) in [
                (guide_left, axis_l, half_l),
                (guide_right, axis_r, half_r),
            ] {
                let mut normal = [-axis[1], axis[0]];
                if normal[1] > 0.0 {
                    normal = [-normal[0], -normal[1]];
                }
                let base_ry = (half_len * 0.28).max(4.0);
                for &ring_scale in &RING_SCALES {
                    let rx = half_len * ring_scale;
                    let ry = base_ry * ring_scale;
                    // Upper arc: indices 0..=12 (head u=-1 -> tail u=+1)
                    for i in 0..=12usize {
                        let u = (i as f32 / 12.0) * 2.0 - 1.0;
                        let arch = (1.0 - u * u).max(0.0).sqrt();
                        let px = guide.center.x + axis[0] * (u * rx) + normal[0] * (arch * ry);
                        let py = guide.center.y + axis[1] * (u * rx) + normal[1] * (arch * ry);
                        src_192.push(Point::new(px, py));
                    }
                    // Lower arc: indices 13..24 (tail -> head, matching `0x707DE40` EBO closure)
                    for k in 0..11usize {
                        let u = 1.0 - ((k + 1) as f32 / 12.0) * 2.0;
                        let arch = (1.0 - u * u).max(0.0).sqrt();
                        let px = guide.center.x + axis[0] * (u * rx) - normal[0] * (arch * ry);
                        let py = guide.center.y + axis[1] * (u * rx) - normal[1] * (arch * ry);
                        src_192.push(Point::new(px, py));
                    }
                }
            }
            debug_assert_eq!(src_192.len(), eye_geometry::TYPE1018_VERTEX_COUNT);

            // Native `0x4C976F0` UV normalization: `src_pts *= (1/W, 1/H)`.
            let uv_192: Vec<[f32; 2]> = src_192.iter().map(|p| [p.x / w, p.y / h]).collect();
            let type1018_ebo = eye_geometry::type1018_dual_ordered_ebo();

            // INFERRED: `points194[0..24]` approximates the face contour for `0x707E820`
            // tail-distance attenuation, and per-side frame visibility (`[face+0x618]` /
            // `[face+0x63c]`) is not yet produced by `points194`, defaulting to `1.0`.
            let contour_slice = &points194[109..=141];
            let tail_atten_l = eye_geometry::brow_tail_contour_atten(
                Point::new(guide_left.end.x, guide_left.end.y),
                half_l * 2.0,
                contour_slice,
            );
            let tail_atten_r = eye_geometry::brow_tail_contour_atten(
                Point::new(guide_right.end.x, guide_right.end.y),
                half_r * 2.0,
                contour_slice,
            );
            let vis_left = 1.0f32;
            let vis_right = 1.0f32;

            let ui_from_signed = |signed: f32| ((signed + 1.0) * 0.5).clamp(0.0, 1.0);

            for (left_pf, right_pf, c_left, c_right, is_head) in [
                (44124u32, 48019u32, c_head_l, c_head_r, true),
                (49221u32, 41922u32, c_tail_l, c_tail_r, false),
            ] {
                if c_left == 0.0 && c_right == 0.0 {
                    continue;
                }
                eye_geometry::stage_ebo_gate(left_pf, &type1018_ebo, src_192.len())
                    .map_err(|_| Refusal::ShapeMismatch("type1018 dual EBO gate"))?;
                let _ = right_pf;

                let ui_l = ui_from_signed(c_left);
                let ui_r = ui_from_signed(c_right);
                let mut dst_192: Vec<[f32; 2]> = src_192.iter().map(|p| [p.x, p.y]).collect();

                for part in [BrowPart::Short, BrowPart::Long] {
                    let degrees = if is_head {
                        let hl = brow_vendor::part_degree(part, 0, ui_l)
                            .map_err(|_| Refusal::ValueOutOfRange("brow head left degree"))?;
                        let hr = brow_vendor::part_degree(part, 1, ui_r)
                            .map_err(|_| Refusal::ValueOutOfRange("brow head right degree"))?;
                        [hl, hr, 0.0, 0.0]
                    } else {
                        let raw_tl = brow_vendor::part_degree(part, 2, ui_l)
                            .map_err(|_| Refusal::ValueOutOfRange("brow tail left degree"))?;
                        let raw_tr = brow_vendor::part_degree(part, 3, ui_r)
                            .map_err(|_| Refusal::ValueOutOfRange("brow tail right degree"))?;
                        let tl = brow_vendor::gate_tail_degree(raw_tl, vis_left) * tail_atten_l;
                        let tr = brow_vendor::gate_tail_degree(raw_tr, vis_right) * tail_atten_r;
                        [0.0, 0.0, tl, tr]
                    };
                    if degrees == [0.0; 4] {
                        continue;
                    }
                    let evaluated = eye_geometry::brow_length_consumer_dual(
                        part,
                        &src_192,
                        guide_left,
                        guide_right,
                        degrees,
                        None,
                    )
                    .map_err(|_| Refusal::ShapeMismatch("type1018 brow length dual consumer"))?;
                    for (dst_pt, delta) in dst_192.iter_mut().zip(evaluated.deltas.iter()) {
                        dst_pt[0] += delta.x;
                        dst_pt[1] += delta.y;
                    }
                }

                draws.push(super::FaceGeometryDraw {
                    pf: left_pf,
                    controls: [c_left, c_right],
                    source_sha256: digest,
                    geometry: super::FaceGeometry::Eye {
                        dest: dst_192,
                        uv: uv_192.clone(),
                        ebo: type1018_ebo.clone(),
                        rules,
                        edge: FragmentEdge::ReshapeNormal,
                        defines: Defines::plain(),
                        clamp: Clamp::disabled(),
                    },
                });
            }
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

    pub(crate) fn face_retouch_mask(
        &self,
        frame: &Image,
    ) -> Result<super::FaceRetouchMask, Refusal> {
        self.validate_ticket(frame, Some(self.job_id))?;
        let parsing = self.model_inputs.as_ref()
            .and_then(|inputs| inputs.parsing.as_ref())
            .filter(|parsing| parsing.job_id == self.job_id)
            .ok_or(Refusal::MissingInput("face retouch parsing"))?;
        let skin = crate::face_parsing::project_face_parsing_skin_mask(
            parsing, frame.width, frame.height,
        )?;
        // The uncut projected landmarks keep the nose zone fixed while panning.
        let bridge = self.points194[54];
        let tip = self.points194[57];
        let center = midpoint(bridge, tip);
        let eye_dist = distance(self.points194[26], self.points194[44]);
        let nose = [
            center[0], center[1], eye_dist * 0.40, distance(bridge, tip) * 0.85,
        ];
        Ok(super::FaceRetouchMask {
            skin,
            face_width: self.roi_xywh[2],
            nose,
        })
    }

    /// Builds same-job wrinkle and historical-skin wrinkle inputs (`PF21024, 21025, 21069, 21070, 21071, 90015, 90023, 90024`)
    /// from the source frame's frequency-separated Gaussian/median base and `fbe.onnx` facial-skin mask.
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
        let mut active_routed = [21025u32, 21069, 21070, 21071, 90015]
            .into_iter()
            .filter(|&pf| control(pf) > 0.0);
        let first_routed = active_routed.next();
        let active_hist = [21024u32, 90023, 90024]
            .into_iter()
            .any(|pf| control(pf) > 0.001);
        if first_routed.is_none() && !active_hist {
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
        let w_u = source.width as usize;
        let _h_u = source.height as usize;

        // Compute bounding union of roi_xywh, landmarks_194, and (when 90015 is active) neck below chin
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
        let c_neck = control(90015);
        if c_neck > 0.0 {
            let eye_dist = distance(self.points194[26], self.points194[44]).max(24.0);
            max_y = (max_y + eye_dist * 1.6).min(source.height as f32);
            min_x = (min_x - eye_dist * 0.3).max(0.0);
            max_x = (max_x + eye_dist * 0.3).min(source.width as f32);
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

        // Two-pass separable box blur inside stage_roi to create a wrinkle-free smooth residual image,
        // paired with `base_img = source.clone()` (so high-freq = 0, wrinkle lines are genuinely smoothed).
        let eye_dist = distance(self.points194[26], self.points194[44]).max(16.0);
        let radius = ((eye_dist * 0.032).round() as usize).clamp(1, 18);
        let rx0 = stage_roi.x as usize;
        let ry0 = stage_roi.y as usize;
        let rx1 = (stage_roi.x + stage_roi.width) as usize;
        let ry1 = (stage_roi.y + stage_roi.height) as usize;
        let mut residual_rgba = source.pixels.clone();
        let mut horiz = source.pixels.clone();
        let blur_pass = |src_buf: &[u8], dst_buf: &mut [u8], horizontal: bool| {
            let win = (radius * 2 + 1) as u32;
            if horizontal {
                for y in ry0..ry1 {
                    let row_off = y * w_u * 4;
                    let mut sum = [0u32; 3];
                    for dx in -(radius as isize)..=(radius as isize) {
                        let sx = (rx0 as isize + dx).clamp(rx0 as isize, (rx1 - 1) as isize) as usize;
                        let p = row_off + sx * 4;
                        for c in 0..3 { sum[c] += src_buf[p + c] as u32; }
                    }
                    for x in rx0..rx1 {
                        let dp = row_off + x * 4;
                        for c in 0..3 { dst_buf[dp + c] = (sum[c] / win) as u8; }
                        let add_x = (x + radius + 1).min(rx1 - 1);
                        let sub_x = x.saturating_sub(radius).max(rx0);
                        let ap = row_off + add_x * 4;
                        let sp = row_off + sub_x * 4;
                        for c in 0..3 { sum[c] = sum[c] + src_buf[ap + c] as u32 - src_buf[sp + c] as u32; }
                    }
                }
            } else {
                for x in rx0..rx1 {
                    let mut sum = [0u32; 3];
                    for dy in -(radius as isize)..=(radius as isize) {
                        let sy = (ry0 as isize + dy).clamp(ry0 as isize, (ry1 - 1) as isize) as usize;
                        let p = (sy * w_u + x) * 4;
                        for c in 0..3 { sum[c] += src_buf[p + c] as u32; }
                    }
                    for y in ry0..ry1 {
                        let dp = (y * w_u + x) * 4;
                        for c in 0..3 { dst_buf[dp + c] = (sum[c] / win) as u8; }
                        let add_y = (y + radius + 1).min(ry1 - 1);
                        let sub_y = y.saturating_sub(radius).max(ry0);
                        let ap = (add_y * w_u + x) * 4;
                        let sp = (sub_y * w_u + x) * 4;
                        for c in 0..3 { sum[c] = sum[c] + src_buf[ap + c] as u32 - src_buf[sp + c] as u32; }
                    }
                }
            }
        };
        for _ in 0..2 {
            blur_pass(&residual_rgba, &mut horiz, true);
            blur_pass(&horiz, &mut residual_rgba, false);
        }
        let base_img = source.clone();
        let residual_img = Image::new(source.width, source.height, residual_rgba)
            .ok_or(Refusal::ShapeMismatch("wrinkle residual image"))?;

        // Helper: soft anatomical ellipse mask for a specific wrinkle region scaled by control weight
        let ellipse_weight = |x: f32, y: f32, cx: f32, cy: f32, rx: f32, ry: f32| -> f32 {
            let d2 = ((x - cx) / rx.max(2.0)).powi(2) + ((y - cy) / ry.max(2.0)).powi(2);
            if d2 >= 1.0 { 0.0 } else { (1.0 - d2).powi(2) }
        };
        let p_le = self.points194[26];
        let p_re = self.points194[44];
        let p_nose_top = self.points194[54];
        let p_nose_tip = self.points194[57];
        let p_chin = self.points194[125];
        let p_ml = self.points194[73];
        let p_mr = self.points194[83];
        let p_glab = midpoint(self.points194[4], self.points194[9]);
        let build_pf_mask = |pf_list: &[u32]| -> Vec<u8> {
            let mut m = vec![0u8; pixels * 4];
            for y in ry0..ry1 {
                let fy = y as f32 + 0.5;
                for x in rx0..rx1 {
                    let idx = y * w_u + x;
                    let sk = skin_mask[idx] as f32 / 255.0;
                    if sk <= 0.01 { continue; }
                    let fx = x as f32 + 0.5;
                    let mut w_max = 0.0f32;
                    for &pf in pf_list {
                        let c = control(pf).clamp(0.0, 1.0);
                        if c <= 0.001 { continue; }
                        let rw = match pf {
                            21024 => ellipse_weight(fx, fy, p_glab[0], p_glab[1], eye_dist * 0.28, eye_dist * 0.25),
                            90024 => ellipse_weight(fx, fy, p_glab[0], p_glab[1] - eye_dist * 0.45, eye_dist * 0.85, eye_dist * 0.42),
                            90023 => {
                                let wl = ellipse_weight(fx, fy, p_le[0], p_le[1] + eye_dist * 0.14, eye_dist * 0.42, eye_dist * 0.26);
                                let wr = ellipse_weight(fx, fy, p_re[0], p_re[1] + eye_dist * 0.14, eye_dist * 0.42, eye_dist * 0.26);
                                wl.max(wr)
                            }
                            21069 => {
                                let wl = ellipse_weight(fx, fy, p_le[0] - eye_dist * 0.10, p_nose_tip[1] + eye_dist * 0.05, eye_dist * 0.42, eye_dist * 0.45);
                                let wr = ellipse_weight(fx, fy, p_re[0] + eye_dist * 0.10, p_nose_tip[1] + eye_dist * 0.05, eye_dist * 0.42, eye_dist * 0.45);
                                wl.max(wr)
                            }
                            21070 => {
                                let mid = midpoint(p_nose_top, p_nose_tip);
                                ellipse_weight(fx, fy, mid[0], mid[1], eye_dist * 0.22, eye_dist * 0.35)
                            }
                            21025 => {
                                let wl = ellipse_weight(fx, fy, p_ml[0] - eye_dist * 0.12, p_ml[1] + eye_dist * 0.22, eye_dist * 0.25, eye_dist * 0.35);
                                let wr = ellipse_weight(fx, fy, p_mr[0] + eye_dist * 0.12, p_mr[1] + eye_dist * 0.22, eye_dist * 0.25, eye_dist * 0.35);
                                wl.max(wr)
                            }
                            21071 => {
                                let mouth_c = midpoint(p_ml, p_mr);
                                ellipse_weight(fx, fy, mouth_c[0], mouth_c[1], eye_dist * 0.55, eye_dist * 0.35)
                            }
                            90015 => ellipse_weight(fx, fy, p_chin[0], p_chin[1] + eye_dist * 0.72, eye_dist * 0.95, eye_dist * 0.85),
                            _ => 0.0,
                        };
                        w_max = w_max.max(rw * c);
                    }
                    let val = (w_max * sk * 255.0).round().clamp(0.0, 255.0) as u8;
                    let off = idx * 4;
                    m[off] = val;
                    m[off + 1] = val;
                    m[off + 2] = val;
                    m[off + 3] = 255;
                }
            }
            m
        };

        let uniforms = WrinkleUniforms {
            legal_alpha: 0.85,
            dark_circle_alpha: 0.85,
            diff_factor: 1.0,
            diff_factor2: 0.0,
            color_threshold_factor: 1.0,
            mix_alpha: 1.0,
            diff_min_threshold: -0.35,
            diff_max_threshold: 0.35,
            high_freq_min_threshold: -0.35,
            high_freq_max_threshold: 0.35,
        };

        if producer.wrinkle.is_none() {
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
                build_pf_mask(first_routed.as_slice()),
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
            let wrinkle_inputs = if let Some(pf) = first_routed {
                // The first mask lives in the legacy slot; only later PFs need extra storage.
                let additional = active_routed
                    .map(|pf| (pf, build_pf_mask(&[pf])))
                    .collect();
                wrinkle_inputs.with_pf_masks(pf, additional)?
            } else {
                wrinkle_inputs
            };
            producer.write_wrinkle(wrinkle_inputs);
        }

        if active_hist {
            let mut rows = Vec::new();
            for pf in [21024u32, 90023, 90024] {
                let c = control(pf);
                if c > 0.001 {
                    let payload = super::HistoricalSkinWrinkle {
                        base: base_img.clone(),
                        residual: residual_img.clone(),
                        mask_rgba: build_pf_mask(&[pf]),
                        skin_mask: skin_mask.clone(),
                        roi: stage_roi,
                        uniforms,
                    };
                    let variant = match pf {
                        21024 => super::HistoricalSkinData::Glabella(payload),
                        90023 => super::HistoricalSkinData::EyeLines(payload),
                        _ => super::HistoricalSkinData::Forehead(payload),
                    };
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
        let runtime = Runtime::get().ok_or(Refusal::MissingInput("ONNX Runtime"))?;
        let (detection, mesh, coarse, dense, extra52_warm, constants, parsing) =
            std::thread::scope(|scope| {
                let detection = scope.spawn(|| -> Result<Session, Refusal> {
                    verify_file(&paths.detection, C1_FACE_DETECTION_MODEL_SHA256)?;
                    let session = backend.open(runtime, &paths.detection)?;
                    validate_detection_session(&session)?;
                    Ok(session)
                });
                let mesh = scope.spawn(|| -> Result<Session, Refusal> {
                    verify_file(&paths.mesh, C1_FACE_MESH_MODEL_SHA256)?;
                    let session = backend.open(runtime, &paths.mesh)?;
                    validate_fixed_session(
                        &session,
                        [1, 3, MESH_SIDE as i64, MESH_SIDE as i64],
                        &[
                            ("Identity", &[1, 1, 1, 1434]),
                            ("Identity_1", &[1, 1, 1, 1]),
                            ("Identity_2", &[1, 1]),
                        ],
                    )?;
                    Ok(session)
                });
                let coarse = scope.spawn(|| -> Result<Session, Refusal> {
                    verify_file(&paths.coarse, FACE_LANDMARK_1063025060_MODEL_SHA256)?;
                    let session = if runtime.is_native_only() {
                        runtime.open_session(&paths.coarse)
                    } else {
                        runtime.open_cpu_session(&paths.coarse)
                    }
                    .map_err(|_| Refusal::MissingInput(MODEL_RUNTIME_ERROR))?;
                    validate_fixed_session(
                        &session,
                        [1, 3, COARSE_SIDE as i64, COARSE_SIDE as i64],
                        &[("2297057593", &[1, 34, 1, 1])],
                    )?;
                    Ok(session)
                });
                let dense = scope.spawn(|| -> Result<Session, Refusal> {
                    verify_file(&paths.dense, FACE_LANDMARK_323773030_MODEL_SHA256)?;
                    let session = if runtime.is_native_only() {
                        runtime.open_session(&paths.dense)
                    } else {
                        runtime.open_cpu_session(&paths.dense)
                    }
                    .map_err(|_| Refusal::MissingInput(MODEL_RUNTIME_ERROR))?;
                    validate_fixed_session(
                        &session,
                        [1, 3, DENSE_SIDE as i64, DENSE_SIDE as i64],
                        &[
                            ("2297057593", &[1, 142, 2]),
                            ("2297057596", &[1, 2]),
                            ("781171859", &[1, 142, 1, 1]),
                        ],
                    )?;
                    Ok(session)
                });
                let extra52_warm = scope.spawn(|| -> Result<(), Refusal> {
                    verify_file(&paths.extra52, super::FACE_LANDMARK_52_MODEL_SHA256)?;
                    let _ = super::portrait_landmark_stage::warm_landmark_session(&paths.extra52);
                    Ok(())
                });
                let constants = scope.spawn(|| -> Result<LandmarkConstants, Refusal> {
                    verify_file(&paths.landmark_constants, FACE_LANDMARK_CONSTANTS_SHA256)?;
                    LandmarkConstants::load(&paths.landmark_constants)
                });
                let parsing = scope.spawn(|| -> Result<Option<FaceParsingProducer>, Refusal> {
                    if include_parsing {
                        Ok(Some(FaceParsingProducer::load(&paths.parsing).map_err(
                            |error| {
                                log::warn!("FaceParsingV2 canonical model refused: {error:#}");
                                Refusal::MissingInput(FACE_PARSING_V2_MODEL_ERROR)
                            },
                        )?))
                    } else {
                        Ok(None)
                    }
                });
                (
                    detection
                        .join()
                        .unwrap_or(Err(Refusal::MissingInput(MODEL_RUNTIME_ERROR))),
                    mesh.join()
                        .unwrap_or(Err(Refusal::MissingInput(MODEL_RUNTIME_ERROR))),
                    coarse
                        .join()
                        .unwrap_or(Err(Refusal::MissingInput(MODEL_RUNTIME_ERROR))),
                    dense
                        .join()
                        .unwrap_or(Err(Refusal::MissingInput(MODEL_RUNTIME_ERROR))),
                    extra52_warm
                        .join()
                        .unwrap_or(Err(Refusal::MissingInput("face model file"))),
                    constants
                        .join()
                        .unwrap_or(Err(Refusal::MissingInput("face landmark constants"))),
                    parsing
                        .join()
                        .unwrap_or(Err(Refusal::MissingInput(FACE_PARSING_V2_MODEL_ERROR))),
                )
            });

        extra52_warm?;
        let constants = constants?;
        let detection = detection?;
        let mesh = mesh?;
        log::info!(
            "C1 face backend {}: detection={} mesh={}",
            backend.label(),
            detection.provider().label(),
            mesh.provider().label()
        );
        let coarse = coarse?;
        let dense = dense?;
        let parsing = parsing?;

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
        self.produce_with_sha(job_id, source, None, true)
    }

    /// Reuse the exact SHA-256 computed for the session's cache identity.
    fn produce_with_sha(
        &self,
        job_id: u64,
        source: &Image,
        source_sha256: Option<[u8; 32]>,
        include_radiance: bool,
    ) -> Result<FaceInputBundle, Refusal> {
        if job_id == 0 {
            return Err(Refusal::ValueOutOfRange("face input job ticket"));
        }
        source
            .validate()
            .map_err(|_| Refusal::ShapeMismatch("face input source image"))?;
        let detection = self.detect(source)?;
        let (mesh, roi_xywh) = match self.mesh_roi(source, &detection) {
            Ok(pair) => pair,
            Err(Refusal::MissingInput(FACE_MESH_SCORE_ERROR)) => {
                self.fallback_mesh_roi_from_detection(source, &detection)?
            }
            Err(err) => return Err(err),
        };
        let (landmark_chain, model_inputs) = if let Some(parsing) = self.parsing.as_ref() {
            std::thread::scope(|scope| {
                let parsing_handle = scope.spawn(|| -> Result<ProducerInputs, Refusal> {
                    let roi = Self::parsing_roi(&detection, roi_xywh)?;
                    let source_to_crop = roi
                        .source_to_crop()
                        .map_err(|_| Refusal::MissingInput(FACE_PARSING_V2_ROI_ERROR))?;
                    let output = if include_radiance {
                        parsing.produce(source, job_id, source_to_crop, 0.0, 0.0)
                    } else {
                        parsing.produce_mask_inputs(source, job_id, source_to_crop, 0.0, 0.0)
                    };
                    output.map_err(|error| {
                            log::warn!("FaceParsingV2 canonical producer refused: {error:#}");
                            Refusal::MissingInput(FACE_PARSING_V2_PRODUCE_ERROR)
                        })
                });
                let landmark_chain = (|| -> Result<_, Refusal> {
                    let points17 = self.points17(source, roi_xywh)?;
                    let points142 = self.points142_with_detection(source, &points17, &detection)?;
                    let landmark_producer = FaceLandmarkProducerInputs::produce_with_source_sha(
                        &self.paths.extra52,
                        source,
                        points142.clone(),
                        source_sha256,
                    )
                    .map_err(|_| Refusal::MissingInput("142->52 face landmark model"))?;
                    let points194 = landmark_producer.points194.clone();
                    let points414 = face_width_414::produce(&points194, 0.0)
                        .map_err(|error| match error {
                            face_width_414::Error::Points194 => {
                                Refusal::MissingInput("face 194 landmarks for 414 producer")
                            }
                            face_width_414::Error::NonFinite => {
                                Refusal::ValueOutOfRange("face 414 landmarks")
                            }
                            face_width_414::Error::UnsupportedScale => {
                                Refusal::ValueOutOfRange("face 414 producer scale")
                            }
                        })?
                        .to_vec();
                    Ok((points17, points142, landmark_producer, points194, points414))
                })();
                let parsing_result = parsing_handle
                    .join()
                    .unwrap_or(Err(Refusal::MissingInput(FACE_PARSING_V2_PRODUCE_ERROR)));
                (landmark_chain, Some(parsing_result))
            })
        } else {
            let points17 = self.points17(source, roi_xywh)?;
            let points142 = self.points142_with_detection(source, &points17, &detection)?;
            let landmark_producer = FaceLandmarkProducerInputs::produce_with_source_sha(
                &self.paths.extra52,
                source,
                points142.clone(),
                source_sha256,
            )
            .map_err(|_| Refusal::MissingInput("142->52 face landmark model"))?;
            let points194 = landmark_producer.points194.clone();
            let points414 = face_width_414::produce(&points194, 0.0)
                .map_err(|error| match error {
                    face_width_414::Error::Points194 => {
                        Refusal::MissingInput("face 194 landmarks for 414 producer")
                    }
                    face_width_414::Error::NonFinite => {
                        Refusal::ValueOutOfRange("face 414 landmarks")
                    }
                    face_width_414::Error::UnsupportedScale => {
                        Refusal::ValueOutOfRange("face 414 producer scale")
                    }
                })?
                .to_vec();
            (
                Ok((points17, points142, landmark_producer, points194, points414)),
                None,
            )
        };
        let (points17, points142, landmark_producer, points194, points414) = landmark_chain?;
        let source_width = source.width as f32;
        let source_height = source.height as f32;
        let roi_normalized_xywh = [
            roi_xywh[0] / source_width,
            roi_xywh[1] / source_height,
            roi_xywh[2] / source_width,
            roi_xywh[3] / source_height,
        ];
        let model_inputs = model_inputs.transpose()?;
        let bundle = FaceInputBundle {
            job_id,
            source_size: [source.width, source.height],
            source_sha256: landmark_producer.source_sha256,
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
            bound_digest: None,
        };
        bundle.validate_ticket_with_digest(source, Some(job_id), || bundle.source_sha256)?;
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
        let mut input = resize_letterbox_bgr_nchw(
            source,
            resized_width,
            resized_height,
            DETECTION_SIDE,
            left,
            top,
        );
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
        let (stride, grid, cell, min_size) = anchor_layout(best);
        let anchor = [
            ((cell % grid) as f32 + 0.5) * stride as f32,
            ((cell / grid) as f32 + 0.5) * stride as f32,
        ];
        let landmark_scale = 0.1 * min_size;
        let source_to_input = [[scale, 0.0, left as f32], [0.0, scale, top as f32]];
        let input_to_source = inverse(source_to_input)?;
        let mut landmarks = [[0.0f32; 2]; 5];
        for (index, point) in landmarks.iter_mut().enumerate() {
            let offset = best * 10 + index * 2;
            *point = decode_detection_point(
                [outputs[2][offset], outputs[2][offset + 1]],
                anchor,
                landmark_scale,
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
        let eye_to_mouth = distance(eye, mouth);
        let side = (eye_distance.max(eye_to_mouth * 1.15) * 3.4).max(16.0);
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
                    input[channel * MESH_SIDE * MESH_SIDE + y * MESH_SIDE + x] =
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
                    channel_order: "RGB",
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

    fn fallback_mesh_roi_from_detection(
        &self,
        source: &Image,
        detection: &C1FaceDetection,
    ) -> Result<(C1FaceMesh, [f32; 4]), Refusal> {
        let eye = midpoint(detection.landmarks_source[0], detection.landmarks_source[1]);
        let mouth = midpoint(detection.landmarks_source[3], detection.landmarks_source[4]);
        let center = [
            (eye[0] + mouth[0]) * 0.5,
            eye[1] * 0.48 + mouth[1] * 0.52,
        ];
        let eye_distance = distance(detection.landmarks_source[0], detection.landmarks_source[1]);
        let eye_to_mouth = distance(eye, mouth);
        let face_w = eye_distance.max(eye_to_mouth * 0.9) * 1.65;
        let face_h = eye_to_mouth.max(eye_distance * 0.9) * 1.95;
        let roi_w = (face_w * 1.36).max(24.0);
        let roi_h = (face_h * 1.36).max(24.0);
        let side = roi_w.max(roi_h);
        let x0 = (center[0] - roi_w * 0.5).max(0.0);
        let y0 = (center[1] - roi_h * 0.52).max(0.0);
        let x1 = (center[0] + roi_w * 0.5).min(source.width as f32);
        let y1 = (center[1] + roi_h * 0.48).min(source.height as f32);
        if x1 - x0 < 8.0 || y1 - y0 < 8.0 {
            return Err(Refusal::MissingInput("C1 face ROI"));
        }
        let scale = MESH_SIDE as f32 / side;
        let source_to_crop = [
            [scale, 0.0, -(center[0] - side * 0.5) * scale],
            [0.0, scale, -(center[1] - side * 0.5) * scale],
        ];
        Ok((
            C1FaceMesh {
                score: detection.score,
                landmarks_source: Vec::new(),
                source_to_crop,
                crop_side: side,
                input_size: [MESH_SIDE as u32, MESH_SIDE as u32],
                normalization: FaceNormalization {
                    channel_order: "RGB",
                    mean: [127.5; 3],
                    scale: [127.5; 3],
                },
                provider: self.mesh.provider().label(),
                model: FaceModelIdentity {
                    model_id: C1_FACE_MESH_MODEL_ID,
                    sha256: C1_FACE_MESH_MODEL_SHA256,
                },
            },
            [x0, y0, x1 - x0, y1 - y0],
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
        self.points142_from_fit(source, fit)
    }

    fn points142_with_detection(
        &self,
        source: &Image,
        points17: &[[f32; 2]],
        detection: &C1FaceDetection,
    ) -> Result<Vec<[f32; 2]>, Refusal> {
        let eye_distance = distance(detection.landmarks_source[0], detection.landmarks_source[1]);
        let p17_le = midpoint(points17[9], points17[10]);
        let p17_re = midpoint(points17[11], points17[12]);
        let max_drift = (eye_distance * 0.25).max(8.0);
        if distance(p17_le, detection.landmarks_source[0]) <= max_drift
            && distance(p17_re, detection.landmarks_source[1]) <= max_drift
        {
            return self.points142(source, points17);
        }
        let template = self.constants.scaled_template();
        let idx = &self.constants.indices17;
        let target_points = [
            midpoint(template[idx[9]], template[idx[10]]),
            midpoint(template[idx[11]], template[idx[12]]),
            template[idx[14]],
            template[idx[15]],
            template[idx[16]],
        ];
        let fit = fit_similarity(&detection.landmarks_source, &target_points)?;
        self.points142_from_fit(source, fit)
    }

    fn points142_from_fit(&self, source: &Image, fit: [[f32; 3]; 2]) -> Result<Vec<[f32; 2]>, Refusal> {
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
/// PF-F: byte budget for cached face-input bundles. A single bundle larger than
/// this is returned to the caller but not cached, so one huge frame cannot
/// evict the whole working set.
const MAX_CACHED_FACE_RESULT_BYTES: usize = 256 * 1024 * 1024;
const ROI_CACHE_SCALE: f64 = 1_000_000.0;
/// PF-F producer identity. The ORT runtime does not expose its version, so the
/// key pins the model SHA-256 set, provider intent, resolved runtime
/// directories, and this session-option revision instead.
const FACE_INPUT_SESSION_OPTIONS_REVISION: &str = "pff-face-input-session-options-v1";

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
    model_fingerprint: u64,
    runtime_fingerprint: u64,
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
            model_fingerprint: model_set_fingerprint(include_parsing),
            runtime_fingerprint: runtime_identity_fingerprint(backend),
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

fn model_set_fingerprint(include_parsing: bool) -> u64 {
    let mut hasher = DefaultHasher::new();
    for sha in [
        C1_FACE_DETECTION_MODEL_SHA256,
        C1_FACE_MESH_MODEL_SHA256,
        FACE_LANDMARK_1063025060_MODEL_SHA256,
        FACE_LANDMARK_323773030_MODEL_SHA256,
        super::FACE_LANDMARK_52_MODEL_SHA256,
        FACE_LANDMARK_CONSTANTS_SHA256,
    ] {
        sha.hash(&mut hasher);
    }
    if include_parsing {
        C1_FACE_PARSING_MODEL_SHA256.hash(&mut hasher);
    }
    hasher.finish()
}

fn runtime_identity_fingerprint(backend: FaceInputBackend) -> u64 {
    let mut hasher = DefaultHasher::new();
    backend.label().hash(&mut hasher);
    FACE_INPUT_SESSION_OPTIONS_REVISION.hash(&mut hasher);
    if let Some(runtime) = Runtime::get() {
        let (library, version, api) = runtime.describe();
        library.to_string_lossy().hash(&mut hasher);
        version.hash(&mut hasher);
        api.hash(&mut hasher);
    }
    for name in [
        "AURORA_ORT_RUNTIME_DIRS",
        "AURORA_ORT_DIR",
        "AURORA_GPU_DLL_DIRS",
        "AURORA_FACE_INPUT_BACKEND",
    ] {
        std::env::var_os(name)
            .map(|value| value.to_string_lossy().into_owned())
            .hash(&mut hasher);
    }
    hasher.finish()
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FaceInputResultKey {
    pub session_fingerprint: u64,
    pub source: FaceInputSourceIdentity,
    pub roi_quantized: [i64; 4],
}

impl FaceInputResultKey {
    fn from_bundle(
        session: &FaceInputSessionKey,
        bundle: &FaceInputBundle,
        source: FaceInputSourceIdentity,
    ) -> Result<Self, Refusal> {
        Ok(Self {
            session_fingerprint: session.fingerprint(),
            source,
            roi_quantized: quantize_roi(bundle.roi_normalized_xywh)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct FaceInputAliasKey {
    session_fingerprint: u64,
    source: FaceInputSourceIdentity,
}

struct FaceInputResultEntry {
    bundle: Arc<FaceInputBundle>,
    bytes: usize,
    last_used: u64,
}

struct FaceInputAliasEntry {
    result: FaceInputResultKey,
    last_used: u64,
}

#[derive(Default)]
struct FaceInputResultCache {
    results: HashMap<FaceInputResultKey, FaceInputResultEntry>,
    aliases: HashMap<FaceInputAliasKey, FaceInputAliasEntry>,
    bytes: usize,
    clock: u64,
}

impl FaceInputResultCache {
    fn lookup(&mut self, alias: &FaceInputAliasKey) -> Option<Arc<FaceInputBundle>> {
        let key = self.aliases.get(alias)?.result.clone();
        self.clock += 1;
        let stamp = self.clock;
        let entry = self.results.get_mut(&key)?;
        entry.last_used = stamp;
        let bundle = Arc::clone(&entry.bundle);
        if let Some(alias_entry) = self.aliases.get_mut(alias) {
            alias_entry.last_used = stamp;
        }
        Some(bundle)
    }

    fn remove_result(&mut self, key: &FaceInputResultKey) {
        if let Some(entry) = self.results.remove(key) {
            self.bytes = self.bytes.saturating_sub(entry.bytes);
        }
        self.aliases.retain(|_, alias| alias.result != *key);
    }

    fn insert(
        &mut self,
        alias: FaceInputAliasKey,
        key: FaceInputResultKey,
        bundle: Arc<FaceInputBundle>,
    ) {
        let bytes = bundle_bytes(&bundle);
        if bytes > MAX_CACHED_FACE_RESULT_BYTES {
            return;
        }
        self.remove_result(&key);
        self.clock += 1;
        let stamp = self.clock;
        self.bytes = self.bytes.saturating_add(bytes);
        self.results.insert(
            key.clone(),
            FaceInputResultEntry {
                bundle,
                bytes,
                last_used: stamp,
            },
        );
        self.aliases.insert(
            alias,
            FaceInputAliasEntry {
                result: key,
                last_used: stamp,
            },
        );
        while self.results.len() > MAX_CACHED_FACE_RESULTS
            || self.bytes > MAX_CACHED_FACE_RESULT_BYTES
        {
            let victim = lru_key(
                self.results
                    .iter()
                    .map(|(key, entry)| (key.clone(), entry.last_used)),
            );
            match victim {
                Some(victim) => self.remove_result(&victim),
                None => break,
            }
        }
    }
}

fn cached_bundle_matches(bundle: &FaceInputBundle, identity: FaceInputSourceIdentity) -> bool {
    let FaceInputSourceIdentity::ContentSha256 { sha256, .. } = identity else {
        return false;
    };
    let size = identity.size();
    let frame_pixels = size[0] as usize * size[1] as usize;
    let matched = bundle.source_size == size
        && bundle.source_sha256 == sha256
        && bundle.points194.len() == 194
        && bundle.points414.len() == 414
        && bundle
            .points194
            .iter()
            .chain(&bundle.points414)
            .flatten()
            .all(|value| value.is_finite())
        && bundle.model_inputs.as_ref().map_or(true, |inputs| {
            inputs.teeth.as_ref().map_or(true, |teeth| {
                teeth.mask_width == size[0]
                    && teeth.mask_height == size[1]
                    && teeth.mask_g.len() == frame_pixels
            })
        });
    matched
}

fn lru_key<K: Clone>(items: impl Iterator<Item = (K, u64)>) -> Option<K> {
    items.min_by_key(|(_, stamp)| *stamp).map(|(key, _)| key)
}

fn bundle_bytes(bundle: &FaceInputBundle) -> usize {
    let mut bytes = std::mem::size_of::<FaceInputBundle>();
    let points = |values: &[[f32; 2]]| std::mem::size_of_val(values);
    bytes += points(&bundle.points17)
        + points(&bundle.points142)
        + points(&bundle.points194)
        + points(&bundle.points414);
    let landmark = &bundle.landmark_producer;
    bytes += points(&landmark.points142)
        + landmark.crop_bgr.len() * std::mem::size_of::<f32>()
        + points(&landmark.raw52)
        + points(&landmark.points52)
        + points(&landmark.points194);
    if let Some(inputs) = bundle.model_inputs.as_ref() {
        if let Some(parsing) = inputs.parsing.as_ref() {
            bytes += parsing.masks.len();
        }
        // A parsing bundle also carries a full-frame teeth mask, two LUTs and
        // the five PF90004 planes. Without these the 256 MiB budget only saw a
        // few MiB and could admit several GB of bundles.
        if let Some(teeth) = inputs.teeth.as_ref() {
            bytes += teeth.mask_g.len() + teeth.whiten_lut.len() + teeth.bright_lut.len();
        }
        if let Some(pf90004) = inputs.pf90004.as_ref() {
            bytes += pf90004.owned_bytes();
        }
    }
    bytes
}

struct FaceInputProducerSlot {
    cell: OnceLock<Result<Arc<FaceInputProducer>, Refusal>>,
    last_used: AtomicU64,
}

impl FaceInputProducerSlot {
    fn new(stamp: u64) -> Self {
        Self {
            cell: OnceLock::new(),
            last_used: AtomicU64::new(stamp),
        }
    }
}

fn next_producer_stamp() -> u64 {
    static CLOCK: AtomicU64 = AtomicU64::new(0);
    CLOCK.fetch_add(1, Ordering::Relaxed) + 1
}

fn producer_cache() -> &'static Mutex<HashMap<FaceInputSessionKey, Arc<FaceInputProducerSlot>>> {
    static CACHE: OnceLock<Mutex<HashMap<FaceInputSessionKey, Arc<FaceInputProducerSlot>>>> =
        OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn result_cache() -> &'static Mutex<FaceInputResultCache> {
    static CACHE: OnceLock<Mutex<FaceInputResultCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(FaceInputResultCache::default()))
}

const MAX_CACHED_FACE_REFUSALS: usize = 8;

struct FaceInputRefusalEntry {
    refusal: Refusal,
    last_used: u64,
}

fn refusal_cache() -> &'static Mutex<HashMap<FaceInputAliasKey, FaceInputRefusalEntry>> {
    static CACHE: OnceLock<Mutex<HashMap<FaceInputAliasKey, FaceInputRefusalEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A per-frame model refusal is deterministic for the same session and frame.
/// Runtime and model-load failures stay retryable.
fn cacheable_refusal(refusal: &Refusal) -> bool {
    match refusal {
        Refusal::ShapeMismatch(_) | Refusal::ValueOutOfRange(_) => true,
        Refusal::MissingInput(reason) => !matches!(
            *reason,
            MODEL_RUNTIME_ERROR | FACE_PARSING_V2_MODEL_ERROR | "ONNX Runtime" | "face model file"
        ),
        Refusal::UnknownEffect(_) | Refusal::NotImplemented(_) => false,
    }
}

fn cached_refusal(alias: &FaceInputAliasKey) -> Option<Refusal> {
    let mut cache = refusal_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let stamp = next_producer_stamp();
    let entry = cache.get_mut(alias)?;
    entry.last_used = stamp;
    Some(entry.refusal.clone())
}

fn remember_refusal(alias: &FaceInputAliasKey, refusal: &Refusal) {
    if !cacheable_refusal(refusal) {
        return;
    }
    let mut cache = refusal_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let stamp = next_producer_stamp();
    cache.insert(
        alias.clone(),
        FaceInputRefusalEntry {
            refusal: refusal.clone(),
            last_used: stamp,
        },
    );
    while cache.len() > MAX_CACHED_FACE_REFUSALS {
        let victim = cache
            .iter()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(key, _)| key.clone());
        match victim {
            Some(victim) => {
                cache.remove(&victim);
            }
            None => break,
        }
    }
}

const MAX_CACHED_SLIMS: usize = 4;

struct SlimCacheEntry {
    digest: [u8; 32],
    stage_size: [u32; 2],
    points194: Vec<[f32; 2]>,
    slim: SlimProducerInputs,
    last_used: u64,
}

fn slim_cache() -> &'static Mutex<Vec<SlimCacheEntry>> {
    static CACHE: OnceLock<Mutex<Vec<SlimCacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn cached_slim(
    digest: [u8; 32],
    stage_size: [u32; 2],
    points194: &[[f32; 2]],
) -> Option<SlimProducerInputs> {
    let mut cache = slim_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let stamp = next_producer_stamp();
    let entry = cache.iter_mut().find(|entry| {
        entry.digest == digest
            && entry.stage_size == stage_size
            && entry.points194 == points194
    })?;
    entry.last_used = stamp;
    Some(entry.slim.clone())
}

fn remember_slim(
    digest: [u8; 32],
    stage_size: [u32; 2],
    points194: &[[f32; 2]],
    slim: &SlimProducerInputs,
) {
    let mut cache = slim_cache()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache.retain(|entry| {
        !(entry.digest == digest
            && entry.stage_size == stage_size
            && entry.points194 == points194)
    });
    while cache.len() >= MAX_CACHED_SLIMS {
        let victim = cache
            .iter()
            .enumerate()
            .min_by_key(|(_, entry)| entry.last_used)
            .map(|(index, _)| index);
        match victim {
            Some(index) => {
                cache.remove(index);
            }
            None => break,
        }
    }
    cache.push(SlimCacheEntry {
        digest,
        stage_size,
        points194: points194.to_vec(),
        slim: slim.clone(),
        last_used: next_producer_stamp(),
    });
}

/// PF-F §11: one in-flight result build per (session, source) alias.
///
/// Two preview threads that miss the result cache at the same time used to run
/// the same detection/mesh/landmark inference twice. The first caller owns the
/// `OnceLock`; the others wait on it without holding either cache mutex. The
/// slot is dropped as soon as the result is in the cache (or the build failed,
/// so a later call can retry).
struct FaceInputResultFlight {
    cell: OnceLock<Result<Arc<FaceInputBundle>, Refusal>>,
}

fn result_flights(
) -> &'static Mutex<HashMap<FaceInputAliasKey, Arc<FaceInputResultFlight>>> {
    static FLIGHTS: OnceLock<Mutex<HashMap<FaceInputAliasKey, Arc<FaceInputResultFlight>>>> =
        OnceLock::new();
    FLIGHTS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn result_flight(alias: &FaceInputAliasKey) -> Arc<FaceInputResultFlight> {
    let mut flights = result_flights()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    Arc::clone(
        flights
            .entry(alias.clone())
            .or_insert_with(|| Arc::new(FaceInputResultFlight { cell: OnceLock::new() })),
    )
}

fn release_result_flight(alias: &FaceInputAliasKey, flight: &Arc<FaceInputResultFlight>) {
    let mut flights = result_flights()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let ours = flights
        .get(alias)
        .is_some_and(|current| Arc::ptr_eq(current, flight));
    if ours {
        flights.remove(alias);
    }
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
        let slot = {
            let mut cache = producer_cache()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let stamp = next_producer_stamp();
            match cache.get_mut(&key) {
                Some(slot) => {
                    slot.last_used.store(stamp, Ordering::Relaxed);
                    Arc::clone(slot)
                }
                None => {
                    let slot = Arc::new(FaceInputProducerSlot::new(stamp));
                    cache.insert(key.clone(), Arc::clone(&slot));
                    while cache.len() > MAX_CACHED_FACE_PRODUCERS {
                        let victim = cache
                            .iter()
                            .min_by_key(|(_, slot)| slot.last_used.load(Ordering::Relaxed))
                            .map(|(key, _)| key.clone());
                        match victim {
                            Some(victim) => {
                                cache.remove(&victim);
                            }
                            None => break,
                        }
                    }
                    slot
                }
            }
        };
        // PF-F P-15: the potentially slow session open runs outside the global
        // cache mutex. `OnceLock::get_or_init` makes same-key callers wait on
        // this slot instead of racing or blocking every other session.
        let created = slot.cell.get_or_init(|| {
            FaceInputProducer::open_with_backend(key.paths.clone(), include_parsing, key.backend)
                .map(Arc::new)
        });
        match created {
            Ok(producer) => Ok(Self {
                key,
                producer: Arc::clone(producer),
            }),
            Err(refusal) => {
                let mut cache = producer_cache()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if let Some(current) = cache.get(&key) {
                    if Arc::ptr_eq(current, &slot) {
                        cache.remove(&key);
                    }
                }
                Err(refusal.clone())
            }
        }
    }

    pub fn producer(&self) -> Arc<FaceInputProducer> {
        Arc::clone(&self.producer)
    }

    pub fn key(&self) -> &FaceInputSessionKey {
        &self.key
    }

    pub fn produce(&self, job_id: u64, source: &Image) -> Result<FaceInputBundle, Refusal> {
        self.produce_for_render(job_id, source, None, true)
    }

    pub(crate) fn produce_for_render(
        &self,
        job_id: u64,
        source: &Image,
        digest: Option<[u8; 32]>,
        include_radiance: bool,
    ) -> Result<FaceInputBundle, Refusal> {
        let digest = digest.unwrap_or_else(|| frame_digest(source));
        let mut bundle = self.produce_with_digest(job_id, source, digest)?;
        if include_radiance
            && let Some(inputs) = bundle.model_inputs.as_mut()
            && let Some(parsing) = inputs.parsing.as_ref()
        {
            inputs.pf90004 = Some(FaceParsingProducer::pf90004_bundle(source, parsing)
                .map_err(|error| {
                    log::warn!("FaceParsingV2 radiance payload refused: {error:#}");
                    Refusal::MissingInput(FACE_PARSING_V2_PRODUCE_ERROR)
                })?);
        }
        Ok(bundle)
    }

    /// Produces a bundle for a caller that already hashed `source`. The digest
    /// becomes the bundle's exact content receipt, so it must cover the whole
    /// frame.
    pub(crate) fn produce_with_digest(
        &self,
        job_id: u64,
        source: &Image,
        digest: [u8; 32],
    ) -> Result<FaceInputBundle, Refusal> {
        if job_id == 0 {
            return Err(Refusal::ValueOutOfRange("face input job ticket"));
        }
        let size = [source.width, source.height];
        let identity = FaceInputSourceIdentity::ContentSha256 { sha256: digest, size };
        let alias = FaceInputAliasKey {
            session_fingerprint: self.key.fingerprint(),
            source: identity,
        };
        if let Some(refusal) = cached_refusal(&alias) {
            return Err(refusal);
        }
        let cached = result_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .lookup(&alias);
        if let Some(cached) = cached {
            if cached_bundle_matches(&cached, identity) {
                let mut bundle = (*cached).clone();
                bundle.reticket(job_id);
                bundle.bound_digest =
                    Some((digest, source.pixels.as_ptr() as usize, source.pixels.len()));
                return Ok(bundle);
            }
            let mut cache = result_cache()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(alias_entry) = cache.aliases.remove(&alias) {
                cache.remove_result(&alias_entry.result);
            }
        }

        // PF-F §11: result single-flight. Concurrent misses on the same alias
        // share one inference; the winner publishes into the result cache and
        // then drops the slot.
        let flight = result_flight(&alias);
        // The `OnceLock` is owned by the flight slot, which is removed from
        // the map only after this call returns. A waiter's closure is never
        // run, so no lock is held while the leader is in `produce_uncached`.
        let outcome = flight
            .cell
            .get_or_init(|| self.produce_uncached(job_id, source, digest, &alias));
        release_result_flight(&alias, &flight);
        match outcome {
            Ok(bundle) => {
                let mut bundle = (**bundle).clone();
                bundle.reticket(job_id);
                bundle.bound_digest =
                    Some((digest, source.pixels.as_ptr() as usize, source.pixels.len()));
                Ok(bundle)
            }
            Err(refusal) => {
                remember_refusal(&alias, refusal);
                Err(refusal.clone())
            }
        }
    }

    /// Slow path shared by the winner of [`Self::produce`]'s single-flight
    /// slot: re-check the result cache (another alias may have completed while
    /// this thread was queued), run the producer, and publish the bundle.
    fn produce_uncached(
        &self,
        job_id: u64,
        source: &Image,
        digest: [u8; 32],
        alias: &FaceInputAliasKey,
    ) -> Result<Arc<FaceInputBundle>, Refusal> {
        let cached = result_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .lookup(alias);
        if let Some(cached) = cached {
            if cached_bundle_matches(&cached, alias.source) {
                return Ok(cached);
            }
            let mut cache = result_cache()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(alias_entry) = cache.aliases.remove(alias) {
                cache.remove_result(&alias_entry.result);
            }
        }
        log::debug!(
            "portrait face uncached source {}x{}, parsing={}",
            source.width, source.height, self.key.include_parsing
        );
        let bundle = self
            .producer
            .produce_with_sha(job_id, source, Some(digest), false)?;
        let result_key = FaceInputResultKey::from_bundle(&self.key, &bundle, alias.source)?;
        let bundle = Arc::new(bundle);
        result_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(alias.clone(), result_key, Arc::clone(&bundle));
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
        results.bytes = 0;
        drop(results);
        result_flights()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        refusal_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        slim_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        verified_file_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        super::portrait_landmark_stage::invalidate_landmark_session_cache();
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
    static CACHE: OnceLock<Mutex<HashMap<PathBuf, PathBuf>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(cached) = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(path)
    {
        return Ok(cached.clone());
    }
    let canonical =
        fs::canonicalize(path).map_err(|_| Refusal::MissingInput("face model file"))?;
    let mut cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if cache.len() >= 32 {
        cache.clear();
    }
    cache.insert(path.to_owned(), canonical.clone());
    Ok(canonical)
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

type VerifiedFileCache = Mutex<HashMap<(PathBuf, [u8; 32]), FaceInputFileStamp>>;

fn verified_file_cache() -> &'static VerifiedFileCache {
    static CACHE: OnceLock<VerifiedFileCache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn verify_file(path: &Path, expected: [u8; 32]) -> Result<(), Refusal> {
    let absolute_path =
        std::path::absolute(path).map_err(|_| Refusal::MissingInput("face model file"))?;
    let stamp = file_stamp(&absolute_path)?;
    let cache_key = (absolute_path.clone(), expected);
    if stamp.modified_nanos != 0 {
        let hit = verified_file_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&cache_key)
            .is_some_and(|cached| *cached == stamp);
        if hit {
            return Ok(());
        }
    }
    let bytes = fs::read(&absolute_path).map_err(|_| Refusal::MissingInput("face model file"))?;
    let actual: [u8; 32] = sha256_32(&bytes);
    if actual != expected {
        return Err(Refusal::ValueOutOfRange("face model identity"));
    }
    if stamp.modified_nanos != 0 && file_stamp(&absolute_path) == Ok(stamp) {
        let mut cache = verified_file_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if cache.len() >= 64 {
            cache.clear();
        }
        cache.insert(cache_key, stamp);
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

fn anchor_layout(index: usize) -> (usize, usize, usize, f32) {
    let first = 2 * (DETECTION_SIDE / 8).pow(2);
    let second = 2 * (DETECTION_SIDE / 16).pow(2);
    if index < first {
        (8, DETECTION_SIDE / 8, index / 2, if index % 2 == 0 { 16.0 } else { 32.0 })
    } else if index < first + second {
        let local = index - first;
        (16, DETECTION_SIDE / 16, local / 2, if local % 2 == 0 { 64.0 } else { 128.0 })
    } else {
        let local = index - first - second;
        (32, DETECTION_SIDE / 32, local / 2, if local % 2 == 0 { 256.0 } else { 512.0 })
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

#[inline]
fn build_x_sample_table(src_width: usize, dst_width: usize) -> Vec<(usize, usize, f32, f32)> {
    let scale_x = src_width as f32 / dst_width as f32;
    let max_x = src_width.saturating_sub(1);
    let mut table = Vec::with_capacity(dst_width);
    for x in 0..dst_width {
        let source_x = (x as f32 + 0.5) * scale_x - 0.5;
        let floor_x = source_x.floor();
        let x0 = floor_x.max(0.0) as usize;
        let x1 = (x0 + 1).min(max_x);
        let fx = (source_x - floor_x).clamp(0.0, 1.0);
        table.push((x0 * 4, x1 * 4, fx, 1.0 - fx));
    }
    table
}

#[cfg_attr(not(test), allow(dead_code))]
fn resize_rgb(source: &Image, width: usize, height: usize) -> Vec<u8> {
    let mut output = vec![0u8; width * height * 3];
    if width == 0 || height == 0 || source.width == 0 || source.height == 0 {
        return output;
    }
    let src_w = source.width as usize;
    let src_h = source.height as usize;
    let row_stride = src_w * 4;
    let max_y = src_h.saturating_sub(1);
    let scale_y = src_h as f32 / height as f32;
    let x_table = build_x_sample_table(src_w, width);
    let pixels = source.pixels.as_slice();
    for y in 0..height {
        let source_y = (y as f32 + 0.5) * scale_y - 0.5;
        let floor_y = source_y.floor();
        let y0 = floor_y.max(0.0) as usize;
        let y1 = (y0 + 1).min(max_y);
        let fy = (source_y - floor_y).clamp(0.0, 1.0);
        let inv_fy = 1.0 - fy;
        let row0 = &pixels[y0 * row_stride..y0 * row_stride + row_stride];
        let row1 = &pixels[y1 * row_stride..y1 * row_stride + row_stride];
        let out_row = &mut output[y * width * 3..(y + 1) * width * 3];
        for (x, &(x0_off, x1_off, fx, inv_fx)) in x_table.iter().enumerate() {
            let dst = x * 3;
            for channel in 0..3 {
                let top = row0[x0_off + channel] as f32 * inv_fx
                    + row0[x1_off + channel] as f32 * fx;
                let bottom = row1[x0_off + channel] as f32 * inv_fx
                    + row1[x1_off + channel] as f32 * fx;
                out_row[dst + channel] =
                    (top * inv_fy + bottom * fy).round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    output
}

/// Fused bilinear resize + 114-pad letterbox + BGR NCHW `(v - 127.5) / 128.0`
/// normalization, bit-identical to `resize_rgb` -> 114 canvas -> BGR NCHW.
fn resize_letterbox_bgr_nchw(
    source: &Image,
    resized_width: usize,
    resized_height: usize,
    side: usize,
    left: usize,
    top: usize,
) -> Vec<f32> {
    let plane = side * side;
    let pad = (114.0f32 - 127.5) / 128.0;
    let mut input = vec![pad; 3 * plane];
    if resized_width == 0 || resized_height == 0 || source.width == 0 || source.height == 0 {
        return input;
    }
    let src_w = source.width as usize;
    let src_h = source.height as usize;
    let row_stride = src_w * 4;
    let max_y = src_h.saturating_sub(1);
    let scale_y = src_h as f32 / resized_height as f32;
    let x_table = build_x_sample_table(src_w, resized_width);
    let pixels = source.pixels.as_slice();
    let (plane_b, rest) = input.split_at_mut(plane);
    let (plane_g, plane_r) = rest.split_at_mut(plane);
    let inv_128 = 1.0f32 / 128.0;

    for y in 0..resized_height {
        let source_y = (y as f32 + 0.5) * scale_y - 0.5;
        let floor_y = source_y.floor();
        let y0 = floor_y.max(0.0) as usize;
        let y1 = (y0 + 1).min(max_y);
        let fy = (source_y - floor_y).clamp(0.0, 1.0);
        let inv_fy = 1.0 - fy;
        let row0 = &pixels[y0 * row_stride..y0 * row_stride + row_stride];
        let row1 = &pixels[y1 * row_stride..y1 * row_stride + row_stride];
        let dst_row_start = (top + y) * side + left;
        let dst_r = &mut plane_r[dst_row_start..dst_row_start + resized_width];
        let dst_g = &mut plane_g[dst_row_start..dst_row_start + resized_width];
        let dst_b = &mut plane_b[dst_row_start..dst_row_start + resized_width];

        for (x, &(x0_off, x1_off, fx, inv_fx)) in x_table.iter().enumerate() {
            let r_top = row0[x0_off] as f32 * inv_fx + row0[x1_off] as f32 * fx;
            let r_bot = row1[x0_off] as f32 * inv_fx + row1[x1_off] as f32 * fx;
            let r_u8 = (r_top * inv_fy + r_bot * fy).round().clamp(0.0, 255.0) as u8;
            dst_r[x] = (r_u8 as f32 - 127.5) * inv_128;

            let g_top = row0[x0_off + 1] as f32 * inv_fx + row0[x1_off + 1] as f32 * fx;
            let g_bot = row1[x0_off + 1] as f32 * inv_fx + row1[x1_off + 1] as f32 * fx;
            let g_u8 = (g_top * inv_fy + g_bot * fy).round().clamp(0.0, 255.0) as u8;
            dst_g[x] = (g_u8 as f32 - 127.5) * inv_128;

            let b_top = row0[x0_off + 2] as f32 * inv_fx + row0[x1_off + 2] as f32 * fx;
            let b_bot = row1[x0_off + 2] as f32 * inv_fx + row1[x1_off + 2] as f32 * fx;
            let b_u8 = (b_top * inv_fy + b_bot * fy).round().clamp(0.0, 255.0) as u8;
            dst_b[x] = (b_u8 as f32 - 127.5) * inv_128;
        }
    }
    input
}

fn warp_reflect_rgb(source: &Image, matrix: [[f32; 3]; 2], side: usize) -> Vec<u8> {
    let inverse = inverse(matrix).unwrap_or([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let width = source.width as isize;
    let height = source.height as isize;
    let row_stride = source.width as usize * 4;
    let pixels = source.pixels.as_slice();
    let mut output = vec![0u8; side * side * 3];
    for y in 0..side {
        let yf = y as f32;
        let base_x = yf * inverse[0][1] + inverse[0][2];
        let base_y = yf * inverse[1][1] + inverse[1][2];
        let out_row = &mut output[y * side * 3..(y + 1) * side * 3];
        for x in 0..side {
            let xf = x as f32;
            let source_x = xf * inverse[0][0] + base_x;
            let source_y = xf * inverse[1][0] + base_y;
            let x0 = source_x.floor() as isize;
            let y0 = source_y.floor() as isize;
            let fx = source_x - x0 as f32;
            let fy = source_y - y0 as f32;
            let inv_fx = 1.0 - fx;
            let inv_fy = 1.0 - fy;
            let rx0 = reflect101(x0, width) as usize * 4;
            let rx1 = reflect101(x0 + 1, width) as usize * 4;
            let ry0 = reflect101(y0, height) as usize * row_stride;
            let ry1 = reflect101(y0 + 1, height) as usize * row_stride;
            let dst = x * 3;
            for channel in 0..3 {
                let top =
                    pixels[ry0 + rx0 + channel] as f32 * inv_fx + pixels[ry0 + rx1 + channel] as f32 * fx;
                let bottom =
                    pixels[ry1 + rx0 + channel] as f32 * inv_fx + pixels[ry1 + rx1 + channel] as f32 * fx;
                out_row[dst + channel] =
                    (top * inv_fy + bottom * fy).round().clamp(0.0, 255.0) as u8;
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
    let max_x = (width - 1) as f64;
    let max_y = (height - 1) as f64;
    let row_stride = width * 4;
    let pixels = source.pixels.as_slice();
    let plane = side * side;
    let mut output = vec![0.0f32; 3 * plane];
    let (plane_b, rest) = output.split_at_mut(plane);
    let (plane_g, plane_r) = rest.split_at_mut(plane);
    let dx_x = inverse[0][0];
    let dx_y = inverse[1][0];
    for y in 0..side {
        let yf = y as f64;
        let mut x_position = inverse[0][1] * yf + inverse[0][2];
        let mut y_position = inverse[1][1] * yf + inverse[1][2];
        let out_row = y * side;
        for x in 0..side {
            let ix = x_position.floor();
            let iy = y_position.floor();
            if ix >= 0.0 && ix < max_x && iy >= 0.0 && iy < max_y {
                let x0 = ix as usize;
                let y0 = iy as usize;
                let fx = (x_position - ix) as f32;
                let fy = (y_position - iy) as f32;
                let ox = 1.0 - fx;
                let oy = 1.0 - fy;
                let w00 = ox * oy;
                let w01 = fx * oy;
                let w10 = ox * fy;
                let w11 = fx * fy;
                let off0 = y0 * row_stride + x0 * 4;
                let off1 = off0 + row_stride;
                let out_idx = out_row + x;
                plane_r[out_idx] = f32::from(pixels[off0]) * w00
                    + f32::from(pixels[off0 + 4]) * w01
                    + f32::from(pixels[off1]) * w10
                    + f32::from(pixels[off1 + 4]) * w11;
                plane_g[out_idx] = f32::from(pixels[off0 + 1]) * w00
                    + f32::from(pixels[off0 + 5]) * w01
                    + f32::from(pixels[off1 + 1]) * w10
                    + f32::from(pixels[off1 + 5]) * w11;
                plane_b[out_idx] = f32::from(pixels[off0 + 2]) * w00
                    + f32::from(pixels[off0 + 6]) * w01
                    + f32::from(pixels[off1 + 2]) * w10
                    + f32::from(pixels[off1 + 6]) * w11;
            }
            x_position += dx_x;
            y_position += dx_y;
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

/// Streaming, zero-copy SHA-256 with runtime x86_64 SHA-NI acceleration.
/// Bit-identical to `aurora_portrait::model_sha256::sha256` without copying
/// the entire 97 MiB RGBA frame into a padded buffer.
pub(crate) fn sha256_32(bytes: &[u8]) -> [u8; 32] {
    const H0: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut state = H0;
    let chunks = bytes.chunks_exact(64);
    let remainder = chunks.remainder();
    let full_bytes = &bytes[..bytes.len() - remainder.len()];

    let mut tail = [0u8; 128];
    tail[..remainder.len()].copy_from_slice(remainder);
    tail[remainder.len()] = 0x80;
    let tail_len = if remainder.len() < 56 { 64 } else { 128 };
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    tail[tail_len - 8..tail_len].copy_from_slice(&bit_len.to_be_bytes());
    let tail_slice = &tail[..tail_len];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::is_x86_feature_detected!("sha")
            && std::is_x86_feature_detected!("sse4.1")
            && std::is_x86_feature_detected!("ssse3")
        {
            // SAFETY: feature detection above guarantees SHA, SSE4.1, and SSSE3.
            unsafe {
                sha256_compress_x86_shani(&mut state, full_bytes);
                sha256_compress_x86_shani(&mut state, tail_slice);
            }
            return finalize_sha256_state(state);
        }
    }

    sha256_compress_scalar(&mut state, full_bytes);
    sha256_compress_scalar(&mut state, tail_slice);
    finalize_sha256_state(state)
}

#[inline]
fn finalize_sha256_state(state: [u32; 8]) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (chunk, word) in out.chunks_exact_mut(4).zip(state) {
        chunk.copy_from_slice(&word.to_be_bytes());
    }
    out
}

const SHA256_K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

fn sha256_compress_scalar(h: &mut [u32; 8], blocks: &[u8]) {
    for block in blocks.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7)
                ^ w[i - 15].rotate_right(18)
                ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17)
                ^ w[i - 2].rotate_right(19)
                ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = *h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(SHA256_K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sha,sse4.1,ssse3")]
unsafe fn sha256_compress_x86_shani(state: &mut [u32; 8], blocks: &[u8]) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    if blocks.is_empty() {
        return;
    }

    let shuf_mask = _mm_set_epi64x(0x0c0d_0e0f_0809_0a0b_u64 as i64, 0x0405_0607_0001_0203_u64 as i64);
    let mut state0 = _mm_set_epi32(
        state[0] as i32,
        state[1] as i32,
        state[4] as i32,
        state[5] as i32,
    );
    let mut state1 = _mm_set_epi32(
        state[2] as i32,
        state[3] as i32,
        state[6] as i32,
        state[7] as i32,
    );

    macro_rules! k_vec {
        ($i:expr) => {
            _mm_set_epi32(
                SHA256_K[$i + 3] as i32,
                SHA256_K[$i + 2] as i32,
                SHA256_K[$i + 1] as i32,
                SHA256_K[$i] as i32,
            )
        };
    }

    macro_rules! rounds4 {
        ($msg:expr, $k_idx:expr) => {{
            let tmp = _mm_add_epi32($msg, k_vec!($k_idx));
            state1 = _mm_sha256rnds2_epu32(state1, state0, tmp);
            let tmp = _mm_shuffle_epi32(tmp, 0x0e);
            state0 = _mm_sha256rnds2_epu32(state0, state1, tmp);
        }};
    }

    macro_rules! schedule {
        ($m0:ident, $m1:ident, $m2:ident, $m3:ident) => {{
            $m0 = _mm_sha256msg1_epu32($m0, $m1);
            let tmp = _mm_alignr_epi8($m3, $m2, 4);
            $m0 = _mm_add_epi32($m0, tmp);
            $m0 = _mm_sha256msg2_epu32($m0, $m3);
        }};
    }

    for block in blocks.chunks_exact(64) {
        let ptr = block.as_ptr();
        let abef_save = state0;
        let cdgh_save = state1;

        let (v0, v1, v2, v3) = unsafe {
            (
                _mm_loadu_si128(ptr.cast::<__m128i>()),
                _mm_loadu_si128(ptr.add(16).cast::<__m128i>()),
                _mm_loadu_si128(ptr.add(32).cast::<__m128i>()),
                _mm_loadu_si128(ptr.add(48).cast::<__m128i>()),
            )
        };

        let mut msg0 = _mm_shuffle_epi8(v0, shuf_mask);
        rounds4!(msg0, 0);

        let mut msg1 = _mm_shuffle_epi8(v1, shuf_mask);
        rounds4!(msg1, 4);

        let mut msg2 = _mm_shuffle_epi8(v2, shuf_mask);
        rounds4!(msg2, 8);

        let mut msg3 = _mm_shuffle_epi8(v3, shuf_mask);
        rounds4!(msg3, 12);

        schedule!(msg0, msg1, msg2, msg3);
        rounds4!(msg0, 16);
        schedule!(msg1, msg2, msg3, msg0);
        rounds4!(msg1, 20);
        schedule!(msg2, msg3, msg0, msg1);
        rounds4!(msg2, 24);
        schedule!(msg3, msg0, msg1, msg2);
        rounds4!(msg3, 28);

        schedule!(msg0, msg1, msg2, msg3);
        rounds4!(msg0, 32);
        schedule!(msg1, msg2, msg3, msg0);
        rounds4!(msg1, 36);
        schedule!(msg2, msg3, msg0, msg1);
        rounds4!(msg2, 40);
        schedule!(msg3, msg0, msg1, msg2);
        rounds4!(msg3, 44);

        schedule!(msg0, msg1, msg2, msg3);
        rounds4!(msg0, 48);
        schedule!(msg1, msg2, msg3, msg0);
        rounds4!(msg1, 52);
        schedule!(msg2, msg3, msg0, msg1);
        rounds4!(msg2, 56);
        schedule!(msg3, msg0, msg1, msg2);
        rounds4!(msg3, 60);

        state0 = _mm_add_epi32(state0, abef_save);
        state1 = _mm_add_epi32(state1, cdgh_save);
    }

    let mut s0 = [0i32; 4];
    let mut s1 = [0i32; 4];
    unsafe {
        _mm_storeu_si128(s0.as_mut_ptr().cast::<__m128i>(), state0);
        _mm_storeu_si128(s1.as_mut_ptr().cast::<__m128i>(), state1);
    }
    state[0] = s0[3] as u32;
    state[1] = s0[2] as u32;
    state[2] = s1[3] as u32;
    state[3] = s1[2] as u32;
    state[4] = s0[1] as u32;
    state[5] = s0[0] as u32;
    state[6] = s1[1] as u32;
    state[7] = s1[0] as u32;
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

    #[test]
    fn fast_sha256_matches_reference_across_block_boundaries() {
        for &len in &[0usize, 1, 3, 55, 56, 57, 63, 64, 65, 119, 120, 127, 128, 129, 4096, 65537] {
            let payload: Vec<u8> = (0..len)
                .map(|i| ((i.wrapping_mul(131).wrapping_add(17)) & 0xff) as u8)
                .collect();
            assert_eq!(
                sha256_32(&payload),
                super::super::sha256_32(&payload),
                "sha256 mismatch at len={len}"
            );
        }
    }

    #[test]
    fn fused_resize_letterbox_bgr_nchw_matches_two_step_reference() {
        let width = 37u32;
        let height = 53u32;
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 7 + y * 3) & 0xff) as u8);
                pixels.push(((x * 5 + y * 11) & 0xff) as u8);
                pixels.push(((x * 13 + y * 2) & 0xff) as u8);
                pixels.push(255);
            }
        }
        let source = Image::new(width, height, pixels).unwrap();
        let side = 32usize;
        let scale = (side as f32 / width.max(height) as f32).min(1.0);
        let rw = ((width as f32 * scale) + 0.5) as usize;
        let rh = ((height as f32 * scale) + 0.5) as usize;
        let left = (side - rw) / 2;
        let top = (side - rh) / 2;

        let resized = resize_rgb(&source, rw, rh);
        let mut canvas = vec![114u8; side * side * 3];
        for y in 0..rh {
            let src_off = y * rw * 3;
            let dst_off = ((top + y) * side + left) * 3;
            canvas[dst_off..dst_off + rw * 3].copy_from_slice(&resized[src_off..src_off + rw * 3]);
        }
        let mut expected = vec![0.0f32; 3 * side * side];
        for y in 0..side {
            for x in 0..side {
                let off = (y * side + x) * 3;
                for c in 0..3 {
                    expected[(2 - c) * side * side + y * side + x] =
                        (canvas[off + c] as f32 - 127.5) / 128.0;
                }
            }
        }

        let actual = resize_letterbox_bgr_nchw(&source, rw, rh, side, left, top);
        assert_eq!(actual, expected);
    }
}

#[cfg(test)]
mod pff_tests {
    use super::*;

    fn clear_shared() {
        let mut guard = shared_face_inputs()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.0.clear();
        guard.1 = 0;
    }

    #[test]
    fn parsing_model_checks_revalidate_replacements_and_preserve_sha() {
        let path = std::env::temp_dir().join(format!(
            "aurora-face-model-{}-{}.onnx",
            std::process::id(),
            std::time::SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        let checks = Mutex::new(HashMap::new());
        let valid = b"known";
        let expected = sha256_32(valid);
        let matches = || matches_parsing_model(&path, valid.len() as u64, expected, &checks);
        let replace = |bytes: &[u8], seconds| {
            use std::io::Write;
            let mut file = fs::File::create(&path).unwrap();
            file.write_all(bytes).unwrap();
            file.set_modified(UNIX_EPOCH + std::time::Duration::from_secs(seconds)).unwrap();
        };
        replace(b"wrong", 1_700_000_000);
        assert!(!matches());
        assert!(!matches());
        replace(valid, 1_700_000_002);
        assert!(matches());
        assert!(matches());
        replace(b"wrong", 1_700_000_004);
        assert!(!matches());
        fs::remove_file(&path).unwrap();
        assert!(!matches());
    }

    #[test]
    fn shared_face_input_reuses_one_arc_and_bounds_entries() {
        clear_shared();
        let first = Arc::new(Image::new(2, 2, vec![7; 16]).expect("image"));
        let registered = register_face_input_image((0x1000, 16, 11, 2, 2), Arc::clone(&first));
        let reused = shared_face_input_image((0x1000, 16, 11, 2, 2)).expect("shared");
        assert!(Arc::ptr_eq(&registered, &reused));
        let digest = shared_source_digest(&reused).expect("registered image");
        assert_eq!(digest.get(), None);
        let expected = sha256_32(&reused.pixels);
        assert_eq!(*digest.get_or_init(|| expected), expected);
        let warm = shared_source_digest(&reused).expect("same registered image");
        assert_eq!(warm.get(), Some(&expected));

        // A different allocation is never trusted by address alone.
        let other = Arc::new(Image::new(2, 2, vec![9; 16]).expect("image"));
        assert!(shared_source_digest(&other).is_none());

        // At most two sources stay resident; the oldest is evicted.
        let second = Arc::new(Image::new(2, 2, vec![8; 16]).expect("image"));
        register_face_input_image((0x2000, 16, 12, 2, 2), Arc::clone(&second));
        let second_digest = shared_source_digest(&second).expect("second image");
        assert_eq!(second_digest.get(), None);
        assert_ne!(*second_digest.get_or_init(|| sha256_32(&second.pixels)), expected);
        let third = Arc::new(Image::new(2, 2, vec![6; 16]).expect("image"));
        register_face_input_image((0x3000, 16, 13, 2, 2), Arc::clone(&third));
        assert!(shared_face_input_image((0x1000, 16, 11, 2, 2)).is_none());
        // Same bytes at the same address with swapped dimensions is a miss.
        assert!(shared_face_input_image((0x2000, 16, 12, 4, 1)).is_none());
        assert!(shared_face_input_image((0x2000, 16, 12, 2, 2)).is_some());
        assert!(shared_face_input_image((0x3000, 16, 13, 2, 2)).is_some());
    }

    #[test]
    fn lru_key_picks_the_oldest_stamp() {
        assert_eq!(
            lru_key(vec![(1u8, 5u64), (2, 3), (3, 9)].into_iter()),
            Some(2)
        );
        assert_eq!(lru_key(std::iter::empty::<(u8, u64)>()), None);
    }

    #[test]
    fn result_flights_share_one_slot_per_alias_and_release_safely() {
        let alias = FaceInputAliasKey {
            session_fingerprint: 1,
            source: FaceInputSourceIdentity::Snapshot {
                fingerprint: 9,
                size: [2, 2],
            },
        };
        let other = FaceInputAliasKey {
            session_fingerprint: 1,
            source: FaceInputSourceIdentity::Snapshot {
                fingerprint: 10,
                size: [2, 2],
            },
        };
        result_flights()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        let first = result_flight(&alias);
        let second = result_flight(&alias);
        assert!(Arc::ptr_eq(&first, &second));
        assert!(!Arc::ptr_eq(&first, &result_flight(&other)));

        release_result_flight(&alias, &first);
        let rebuilt = result_flight(&alias);
        assert!(!Arc::ptr_eq(&first, &rebuilt));
        // A stale waiter must not evict the slot that replaced it.
        release_result_flight(&alias, &first);
        assert!(
            result_flights()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .contains_key(&alias)
        );
        release_result_flight(&alias, &rebuilt);
        result_flights()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    #[test]
    fn content_and_snapshot_identities_are_distinct() {
        let snapshot = FaceInputSourceIdentity::Snapshot {
            fingerprint: 7,
            size: [2, 1],
        };
        assert_eq!(snapshot.size(), [2, 1]);
        let content = FaceInputSourceIdentity::ContentSha256 {
            sha256: [3; 32],
            size: [2, 1],
        };
        assert_eq!(content.size(), [2, 1]);
        assert_ne!(snapshot, content);
    }

    #[test]
    fn frame_refusals_are_cached_but_runtime_failures_retry() {
        refusal_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
        let alias = FaceInputAliasKey {
            session_fingerprint: 3,
            source: FaceInputSourceIdentity::ContentSha256 {
                sha256: [1; 32],
                size: [2, 2],
            },
        };
        let refusal = Refusal::MissingInput("C1 face mesh score");
        assert!(cacheable_refusal(&refusal));
        assert!(!cacheable_refusal(&Refusal::MissingInput(MODEL_RUNTIME_ERROR)));
        remember_refusal(&alias, &refusal);
        assert_eq!(cached_refusal(&alias), Some(refusal));

        let runtime_alias = FaceInputAliasKey {
            session_fingerprint: 3,
            source: FaceInputSourceIdentity::ContentSha256 {
                sha256: [2; 32],
                size: [2, 2],
            },
        };
        remember_refusal(
            &runtime_alias,
            &Refusal::MissingInput(MODEL_RUNTIME_ERROR),
        );
        assert_eq!(cached_refusal(&runtime_alias), None);
        refusal_cache()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clear();
    }

    /// PF-F acceptance microbenchmark: 24MP, ten consecutive requests.
    /// Run with `cargo test --release -p aurora-engine --lib pff_tests
    /// -- --ignored --nocapture --test-threads=1`.
    #[test]
    #[ignore]
    fn face_input_image_ten_calls_are_shared() {
        let (w, h) = (6000u32, 4000u32);
        let rgba = vec![128u8; (w as usize) * (h as usize) * 4];
        let source = crate::mock::process::Source::from_rgba(w, h, &rgba);
        let first_start = std::time::Instant::now();
        let first = source.face_input_image();
        let first_time = first_start.elapsed();
        let hits_start = std::time::Instant::now();
        let mut shared = 0usize;
        for _ in 0..9 {
            let next = source.face_input_image();
            if Arc::ptr_eq(&first, &next) {
                shared += 1;
            }
        }
        eprintln!(
            "PFF after: first={first_time:?} hits={:?} total={:?} shared={}/9 bytes={}",
            hits_start.elapsed(),
            first_time + hits_start.elapsed(),
            shared,
            first.pixels.len()
        );
        assert_eq!(shared, 9);
    }

    /// Real-photo first-produce latency benchmark (4032x6048).
    /// Set `AURORA_C1_DIR`, `AURORA_FACE_LANDMARK_ROOT`, `AURORA_FACE_PARSING_MODEL`,
    /// and optionally `AURORA_BENCH_IMAGE` (PNG/JPG/TIF path), then run with
    /// `--ignored --nocapture --test-threads=1`.
    #[test]
    #[ignore]
    fn bench_first_face_input_produce_latency() {
        let image_path = std::env::var_os("AURORA_BENCH_IMAGE")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(
                    r"C:\WorkSpace\Aurora Studio\90_work\lanes\integration\abi-20261121\A-libraw-_DSC2719.png",
                )
            });
        let img = if image_path.is_file() {
            let decoded = image::open(&image_path).expect("open bench image").to_rgba8();
            let (w, h) = decoded.dimensions();
            Image::new(w, h, decoded.into_raw()).expect("valid bench image")
        } else {
            Image::solid(4032, 6048, [120, 110, 100, 255])
        };

        let t_old_sha = std::time::Instant::now();
        let old_sha = super::super::sha256_32(&img.pixels);
        let old_sha_ms = t_old_sha.elapsed().as_secs_f64() * 1000.0;

        let t_new_sha = std::time::Instant::now();
        let new_sha = sha256_32(&img.pixels);
        let new_sha_ms = t_new_sha.elapsed().as_secs_f64() * 1000.0;
        assert_eq!(old_sha, new_sha);

        let scale = (DETECTION_SIDE as f32 / img.width.max(img.height) as f32).min(1.0);
        let rw = ((img.width as f32 * scale) + 0.5) as usize;
        let rh = ((img.height as f32 * scale) + 0.5) as usize;
        let left = (DETECTION_SIDE - rw) / 2;
        let top = (DETECTION_SIDE - rh) / 2;
        let t_resize = std::time::Instant::now();
        let _input = resize_letterbox_bgr_nchw(&img, rw, rh, DETECTION_SIDE, left, top);
        let resize_ms = t_resize.elapsed().as_secs_f64() * 1000.0;

        FaceInputSession::invalidate_all();
        let t_open = std::time::Instant::now();
        let session = FaceInputSession::open_from_env(true).expect("open FaceInputSession");
        let open_ms = t_open.elapsed().as_secs_f64() * 1000.0;

        let t_first = std::time::Instant::now();
        let bundle = session.produce(1, &img).expect("first produce");
        let first_ms = t_first.elapsed().as_secs_f64() * 1000.0;

        let t_hit = std::time::Instant::now();
        let _bundle2 = session.produce(2, &img).expect("cached produce");
        let hit_ms = t_hit.elapsed().as_secs_f64() * 1000.0;

        eprintln!(
            "BENCH {}x{} ({:.1} MiB): old_sha256={:.1}ms new_sha256={:.1}ms resize_nchw={:.1}ms open_cold={:.1}ms first_produce={:.1}ms cached_produce={:.2}ms det_score={:.4} eye0={:?} eye1={:?} pts194={} parsing={}",
            img.width,
            img.height,
            img.pixels.len() as f64 / (1024.0 * 1024.0),
            old_sha_ms,
            new_sha_ms,
            resize_ms,
            open_ms,
            first_ms,
            hit_ms,
            bundle.detection.score,
            bundle.detection.landmarks_source[0],
            bundle.detection.landmarks_source[1],
            bundle.points194.len(),
            bundle.model_inputs.as_ref().and_then(|m| m.parsing.as_ref()).is_some(),
        );
    }
}


#[cfg(test)]
mod frame_digest_scope_tests {
    use super::*;

    const JOB: u64 = 71;

    fn source(seed: u8) -> Image {
        Image::solid(16, 12, [seed, 20, 30, 255])
    }

    fn bundle(job: u64, source: &Image) -> FaceInputBundle {
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
            source_sha256: sha256_32(&source.pixels),
            detection: C1FaceDetection {
                score: 0.9,
                landmarks_source: [[4.0, 3.0]; 5],
                source_to_input: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                input_size: [640, 640],
                normalization,
                provider: "cpu",
                model,
            },
            mesh: C1FaceMesh {
                score: 0.9,
                landmarks_source: vec![[4.0, 3.0, 0.0]],
                source_to_crop: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                crop_side: 8.0,
                input_size: [256, 256],
                normalization,
                provider: "cpu",
                model,
            },
            roi_xywh: [2.0, 1.5, 12.0, 9.0],
            roi_normalized_xywh: [0.125, 0.125, 0.75, 0.75],
            points17: Vec::new(),
            points142: Vec::new(),
            points194: vec![[8.0, 6.0]; 194],
            points414: vec![[8.0, 6.0]; 414],
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
            bound_digest: None,
        }
    }

    #[test]
    fn one_scope_digest_covers_repeated_frame_checks() {
        let source = source(10);
        let bundle = bundle(JOB, &source);

        reset_frame_digest_computes();
        bundle.validate_ticket(&source, Some(JOB)).unwrap();
        assert_eq!(
            frame_digest_computes(),
            1,
            "outside a scope every check hashes"
        );

        let scope = FrameDigestScope::new(&source);
        for _ in 0..6 {
            bundle.validate_ticket(&source, Some(JOB)).unwrap();
        }
        assert_eq!(
            frame_digest_computes(),
            2,
            "the scope computes the frame digest once and reuses it"
        );

        drop(scope);
        bundle.validate_ticket(&source, Some(JOB)).unwrap();
        assert_eq!(
            frame_digest_computes(),
            3,
            "dropping the scope restores hashing"
        );
    }

    #[test]
    fn scope_never_vouches_for_another_allocation() {
        let first = source(10);
        let same_bytes = first.clone();
        let other = source(99);
        let bundle = bundle(JOB, &first);
        let _scope = FrameDigestScope::new(&first);

        reset_frame_digest_computes();
        // Same bytes in a fresh allocation: still valid, but hashed, not reused.
        bundle.validate_ticket(&same_bytes, Some(JOB)).unwrap();
        assert_eq!(frame_digest_computes(), 1);

        // Different bytes: the scope digest must not mask the mismatch.
        assert_eq!(
            bundle.validate_ticket(&other, Some(JOB)),
            Err(Refusal::ValueOutOfRange("face input source digest"))
        );
        assert_eq!(frame_digest_computes(), 2);
    }

    #[test]
    fn attach_eye_and_brow_emits_192_vertex_996_ebo_with_ring3_anchored() {
        let source = Image::solid(256, 256, [120, 110, 100, 255]);
        let mut b = bundle(JOB, &source);
        // Place synthetic jaw contour (109..=141) far enough from brows and distinct left/right brow landmarks.
        for (idx, p) in (109..=141).enumerate() {
            let theta = idx as f32 / 33.0 * std::f32::consts::TAU;
            b.points194[p] = [128.0 + theta.cos() * 110.0, 136.0 + theta.sin() * 110.0];
        }
        // Left brow (tail=0, mid=2/6, head=4)
        b.points194[0] = [70.0, 96.0];
        b.points194[2] = [90.0, 92.0];
        b.points194[6] = [90.0, 92.0];
        b.points194[4] = [110.0, 96.0];
        // Right brow (tail=9, mid=11/15, head=13)
        b.points194[13] = [146.0, 96.0];
        b.points194[11] = [166.0, 92.0];
        b.points194[15] = [166.0, 92.0];
        b.points194[9] = [186.0, 96.0];

        let mut extra = std::collections::BTreeMap::new();
        // Left head short (-0.6), right head long (+0.8), left tail long (+0.7), right tail short (-0.5)
        extra.insert("pc:44124".to_string(), -0.6);
        extra.insert("pc:48019".to_string(), 0.8);
        extra.insert("pc:49221".to_string(), 0.7);
        extra.insert("pc:41922".to_string(), -0.5);

        let mut producer = ProducerInputs::default();
        b.attach_eye_and_brow_geometry(&source, &extra, &mut producer)
            .expect("attach brow length 192/996");
        let fg = producer
            .face_geometry
            .as_ref()
            .expect("face_geometry attached");
        assert_eq!(fg.draws.len(), 2);

        for (draw, expected_pf, expected_controls) in [
            (&fg.draws[0], 44124u32, [-0.6f32, 0.8f32]),
            (&fg.draws[1], 49221u32, [0.7f32, -0.5f32]),
        ] {
            assert_eq!(draw.pf, expected_pf);
            assert_eq!(draw.controls, expected_controls);
            let super::super::FaceGeometry::Eye { dest, uv, ebo, .. } = &draw.geometry else {
                panic!("expected FaceGeometry::Eye for PF{expected_pf}");
            };
            assert_eq!(dest.len(), 192);
            assert_eq!(uv.len(), 192);
            assert_eq!(ebo.len(), 996);
            assert_eq!(ebo.iter().copied().min(), Some(0));
            assert_eq!(ebo.iter().copied().max(), Some(191));
            assert_eq!(
                aurora_portrait::kernels::eye_geometry::stage_ebo_gate(
                    expected_pf,
                    ebo,
                    dest.len()
                ),
                Ok(332)
            );
            // Ring 3 (left 72..96, right 168..192) must remain anchored at source position (uv * size).
            for slot in (72..96).chain(168..192) {
                let sx = uv[slot][0] * source.width as f32;
                let sy = uv[slot][1] * source.height as f32;
                assert!((dest[slot][0] - sx).abs() < 1e-4, "PF{expected_pf} slot {slot} x drifted");
                assert!((dest[slot][1] - sy).abs() < 1e-4, "PF{expected_pf} slot {slot} y drifted");
            }
            // Active rings must have non-zero displacement.
            let active_moved = (0..72)
                .chain(96..168)
                .any(|slot| {
                    let sx = uv[slot][0] * source.width as f32;
                    let sy = uv[slot][1] * source.height as f32;
                    (dest[slot][0] - sx).abs() > 1e-3 || (dest[slot][1] - sy).abs() > 1e-3
                });
            assert!(active_moved, "PF{expected_pf} active rings did not displace");
        }
    }

    #[test]
    fn eye_and_brow_chained_reshape_warps_textured_face_pixels() {
        let w = 256u32;
        let h = 256u32;
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                // High-contrast textured pattern so any geometric warp shifts pixel values.
                pixels[i] = ((x.wrapping_mul(17) ^ y.wrapping_mul(11)) & 0xff) as u8;
                pixels[i + 1] = ((x.wrapping_mul(7) + y.wrapping_mul(23)) & 0xff) as u8;
                pixels[i + 2] = ((x.wrapping_mul(29) + y.wrapping_mul(5)) & 0xff) as u8;
                pixels[i + 3] = 255;
            }
        }
        let source = Image::new(w, h, pixels).unwrap();
        let stage_sha = super::sha256_32(&source.pixels);
        let mut b = bundle(JOB, &source);
        b.points194.resize(194, [128.0, 128.0]);

        // Realistic 194-point eye/brow/nose layout around center (128, 128)
        // Left brow 0..=8 centered near (96, 88)
        for i in 0..=8usize {
            let t = i as f32 / 8.0;
            b.points194[i] = [110.0 - t * 32.0, 88.0 - (t * std::f32::consts::PI).sin() * 6.0];
        }
        // Right brow 9..=17 centered near (160, 88)
        for i in 0..=8usize {
            let t = i as f32 / 8.0;
            b.points194[9 + i] = [146.0 + t * 32.0, 88.0 - (t * std::f32::consts::PI).sin() * 6.0];
        }
        // Left eye 18..=25 + pupil 26 centered at (96, 108)
        for i in 0..8usize {
            let a = i as f32 / 8.0 * std::f32::consts::TAU;
            b.points194[18 + i] = [96.0 + a.cos() * 14.0, 108.0 + a.sin() * 6.0];
        }
        b.points194[18] = [110.0, 108.0];
        b.points194[22] = [82.0, 108.0];
        b.points194[26] = [96.0, 108.0];
        // Right eye 36..=43 + pupil 44 centered at (160, 108)
        for i in 0..8usize {
            let a = i as f32 / 8.0 * std::f32::consts::TAU;
            b.points194[36 + i] = [160.0 + a.cos() * 14.0, 108.0 + a.sin() * 6.0];
        }
        b.points194[36] = [174.0, 108.0];
        b.points194[40] = [146.0, 108.0];
        b.points194[44] = [160.0, 108.0];
        // Nose tip & lip corners
        b.points194[57] = [128.0, 136.0];
        b.landmark_producer.points194 = b.points194.clone();

        let mut extra = std::collections::BTreeMap::new();
        for (pf, val) in [
            (8001u32, 0.8f32),
            (8002, 0.8),
            (8005, 0.93),
            (8006, 0.93),
            (8009, 1.0),
            (8010, 1.0),
            (8011, 1.0),
            (8012, 1.0),
            (8013, 1.0),
            (8014, 1.0),
            (8015, 1.0),
            (8016, 1.0),
            (30073, 0.7),
        ] {
            extra.insert(format!("pc:{pf}"), val);
        }

        let mut producer = ProducerInputs::default();
        b.attach_slim(&source, &mut producer)
            .expect("attach slim landmarks");
        b.attach_eye_and_brow_geometry(&source, &extra, &mut producer)
            .expect("attach eye/brow geometry");
        let fg = producer.face_geometry.as_ref().expect("face_geometry present");
        let slim = producer.slim.as_ref();

        // Chain every pair sequentially through portrait_face_stage using stage_sha
        // (simulating real chained pipeline execution where current.pixels mutates after step 1).
        let mut current = source.clone();
        for (left, right) in [
            (8001u32, 8002u32),
            (8005, 8006),
            (8009, 8010),
            (8011, 8012),
            (8013, 8014),
            (8015, 8016),
            (30073, 30073),
        ] {
            let controls = [
                extra[&format!("pc:{left}")],
                if left == right { 0.0 } else { extra[&format!("pc:{right}")] },
            ];
            let next = super::super::portrait_face_stage::apply_with_stage_sha(
                &current,
                crate::NormRect::FULL,
                Some(JOB),
                Some(fg),
                slim,
                left,
                controls,
                Some(stage_sha),
            )
            .unwrap_or_else(|e| panic!("PF{left} refused in chain: {e:?}"));
            let next_img = next
                .image
                .unwrap_or_else(|| panic!("PF{left} produced no image"));
            assert_ne!(
                next_img.pixels, current.pixels,
                "PF{left} produced zero pixel change"
            );
            current = next_img;
        }
        // Top-left corner far outside orbital/brow regions stays untouched
        assert_eq!(&current.pixels[..4], &source.pixels[..4]);
    }

    #[test]
    fn wrinkle_pipeline_smooths_all_8_anatomical_zones_on_textured_face() {
        let w = 256u32;
        let h = 256u32;
        // Flat skin color with high-contrast dark crease lines in each anatomical wrinkle region
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                pixels[i] = 200;
                pixels[i + 1] = 160;
                pixels[i + 2] = 135;
                pixels[i + 3] = 255;
            }
        }
        let mut draw_crease = |cx: usize, cy: usize, rx: usize| {
            for x in cx.saturating_sub(rx)..=(cx + rx).min(w as usize - 1) {
                for y in cy.saturating_sub(1)..=(cy + 1).min(h as usize - 1) {
                    let i = (y * w as usize + x) * 4;
                    pixels[i] = 110;
                    pixels[i + 1] = 85;
                    pixels[i + 2] = 70;
                }
            }
        };
        // Zones matching canonical 194 landmarks below:
        draw_crease(128, 86, 6);   // 21024 glabella
        draw_crease(110, 178, 6);  // 21025 marionette left
        draw_crease(88, 136, 8);   // 21069 cheek left
        draw_crease(128, 122, 6);  // 21070 nasalis
        draw_crease(128, 166, 8);  // 21071 perioral
        draw_crease(128, 224, 14); // 90015 neck
        draw_crease(78, 108, 6);   // 90023 periorbital left
        draw_crease(128, 62, 16);  // 90024 forehead

        let source = Image::new(w, h, pixels).unwrap();
        let stage_sha = super::sha256_32(&source.pixels);
        let mut b = bundle(JOB, &source);
        b.roi_xywh = [40.0, 32.0, 176.0, 176.0];
        // Populate anatomical 194 landmarks for brows, eyes, nose, lip, jaw/chin
        for i in 0..=8usize {
            b.points194[i] = [110.0 - i as f32 * 4.0, 86.0];
        }
        for i in 0..=8usize {
            b.points194[9 + i] = [146.0 + i as f32 * 4.0, 86.0];
        }
        for i in 0..8usize {
            let a = i as f32 / 8.0 * std::f32::consts::TAU;
            b.points194[18 + i] = [96.0 + a.cos() * 14.0, 108.0 + a.sin() * 6.0];
            b.points194[36 + i] = [160.0 + a.cos() * 14.0, 108.0 + a.sin() * 6.0];
        }
        b.points194[18] = [110.0, 108.0];
        b.points194[22] = [82.0, 108.0];
        b.points194[26] = [96.0, 108.0]; // left eye anchor
        b.points194[40] = [146.0, 108.0];
        b.points194[36] = [174.0, 108.0];
        b.points194[44] = [160.0, 108.0]; // right eye anchor
        b.points194[54] = [128.0, 110.0]; // nose bridge
        b.points194[57] = [128.0, 134.0]; // nose tip
        b.points194[73] = [116.0, 166.0]; // lip left
        b.points194[83] = [140.0, 166.0]; // lip right
        b.points194[109] = [56.0, 110.0]; // contour left
        b.points194[125] = [128.0, 200.0]; // chin
        b.points194[141] = [200.0, 110.0]; // contour right

        let side = crate::face_parsing::FACE_PARSING_V2_SIDE as usize;
        let plane = side * side;
        let mut parsing_masks =
            vec![0u8; crate::face_parsing::FACE_PARSING_V2_CLASSES * plane];
        let skin_offset = crate::face_parsing::FACE_PARSING_V2_SKIN_CLASS * plane;
        parsing_masks[skin_offset..skin_offset + plane].fill(255);
        let parsing = crate::face_parsing::FaceParsingProducerInputs {
            job_id: JOB,
            width: side as u32,
            height: side as u32,
            channels: crate::face_parsing::FACE_PARSING_V2_CLASSES as u8,
            model_sha256: crate::face_parsing::FACE_PARSING_V2_MODEL_SHA256,
            source_to_crop: crate::face_parsing::FaceParsingRoi {
                x: 0.0,
                y: 0.0,
                width: w as f32,
                height: h as f32,
                roll_radians: 0.0,
            }
            .source_to_crop()
            .unwrap(),
            masks: parsing_masks,
            mask_provider: "cpu",
        };

        let mut extra = std::collections::BTreeMap::new();
        for pf in [21024u32, 21025, 21069, 21070, 21071, 90015, 90023, 90024] {
            extra.insert(format!("pc:{pf}"), 1.0);
        }

        let mut model_inputs = ProducerInputs::default();
        model_inputs.parsing = Some(parsing);
        b.model_inputs = Some(model_inputs);
        let mut producer = ProducerInputs::default();
        b.attach_wrinkle_from_extra(&source, &extra, &mut producer)
            .expect("attach wrinkle inputs");
        let w_inputs = producer.wrinkle.as_ref().expect("wrinkle attached");
        for (pf, x, y, other_x, other_y) in [
            (21025, 110, 178, 128, 122),
            (21069, 88, 136, 128, 166),
            (21070, 128, 122, 128, 166),
            (21071, 128, 166, 128, 122),
            (90015, 128, 224, 128, 122),
        ] {
            let mask = w_inputs
                .resolve_for(&source, Some(JOB), pf)
                .unwrap()
                .mask_rgba_full;
            let at = |x, y| mask[((y * w + x) * 4) as usize];
            assert!(at(x, y) > 0, "PF{pf} missed its own crease");
            assert_eq!(at(other_x, other_y), 0, "PF{pf} leaked to another crease");
        }
        let h_inputs = producer
            .historical_skin
            .as_ref()
            .expect("historical_skin attached");

        let masks = super::super::PortraitMasks::default();
        let mut job_inputs = super::super::PortraitJobInputs::from_renderer_job(JOB, &masks);
        job_inputs.stage_sha256 = Some(stage_sha);
        job_inputs.wrinkle_producer = Some(w_inputs);
        job_inputs.historical_skin = Some(h_inputs);

        // Simulated prior effect modified pixel (0,0) before wrinkles run
        let mut mutated = source.clone();
        mutated.pixels[0] = mutated.pixels[0].saturating_sub(5);

        // Verify all 5 routed wrinkle PFs execute via portrait_stage & brighten dark creases
        for pf in [21025u32, 21069, 21070, 21071, 90015] {
            let out = super::super::apply_cpu_effect(
                &mutated,
                pf,
                1.0,
                &extra,
                None,
                w,
                h,
                crate::NormRect::FULL,
                &job_inputs,
                None,
            )
            .unwrap_or_else(|e| panic!("routed wrinkle PF{pf} refused: {e:?}"));
            assert_ne!(out.pixels, mutated.pixels, "PF{pf} produced zero diff");
        }

        // Verify all 3 historical wrinkle PFs execute even after `mutated` changed frame sha
        for pf in [21024u32, 90023, 90024] {
            let out = super::super::portrait_historical_skin::apply_with_stage_sha(
                &mutated,
                Some(JOB),
                pf,
                1.0,
                Some(h_inputs),
                Some(stage_sha),
            )
            .unwrap_or_else(|e| panic!("historical wrinkle PF{pf} refused: {e:?}"))
            .unwrap_or_else(|| panic!("historical wrinkle PF{pf} returned None"));
            assert_ne!(out.pixels, mutated.pixels, "PF{pf} produced zero diff");
        }
    }
}
