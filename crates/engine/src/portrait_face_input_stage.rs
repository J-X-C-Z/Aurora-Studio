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
        let inputs = super::FaceGeometryInputs {
            job_id: self.job_id,
            width: source.width,
            height: source.height,
            draws: vec![super::FaceGeometryDraw {
                pf: 8033,
                controls: [c8033, 0.0],
                source_sha256: frame_digest(source),
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
        let digest = frame_digest(source);
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
        let digest = frame_digest(source);
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

            // INFERRED: `points194` brow anchor indices (left head/mid/tail = 24/27/31,
            // right head/mid/tail = 44/47/51) and jaw/face contour slice `0..24` approximate
            // PixelCake's internal 293-point `face_info+0x924` and 56-point contour; exact
            // index correspondence still requires live same-photo verification.
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

            let (guide_left, axis_l, half_l) =
                build_side_guide(points194[24], points194[27], points194[31], -1.0);
            let (guide_right, axis_r, half_r) =
                build_side_guide(points194[44], points194[47], points194[51], 1.0);

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
            let contour_slice = &points194[..24];
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

    /// Builds same-job wrinkle and historical-skin wrinkle inputs (`PF21024, 21025, 21069, 21070, 21071, 90023, 90024`)
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
        let coarse = if runtime.is_native_only() {
            runtime.open_session(&paths.coarse)
        } else {
            runtime.open_cpu_session(&paths.coarse)
        }
        .map_err(|_| Refusal::MissingInput(MODEL_RUNTIME_ERROR))?;
        validate_fixed_session(
            &coarse,
            [1, 3, COARSE_SIDE as i64, COARSE_SIDE as i64],
            &[("2297057593", &[1, 34, 1, 1])],
        )?;
        let dense = if runtime.is_native_only() {
            runtime.open_session(&paths.dense)
        } else {
            runtime.open_cpu_session(&paths.dense)
        }
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
        self.produce_with_sha(job_id, source, None)
    }

    /// Reuse the exact SHA-256 computed for the session's cache identity.
    fn produce_with_sha(
        &self,
        job_id: u64,
        source: &Image,
        source_sha256: Option<[u8; 32]>,
    ) -> Result<FaceInputBundle, Refusal> {
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
            source_sha256: source_sha256.unwrap_or_else(|| frame_digest(source)),
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
        self.produce_with_digest(job_id, source, frame_digest(source))
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
        let bundle = self
            .producer
            .produce_with_sha(job_id, source, Some(digest))?;
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
        // Place synthetic contour (0..24) far enough from brows and distinct left/right brow landmarks.
        for i in 0..24 {
            let theta = i as f32 / 24.0 * std::f32::consts::TAU;
            b.points194[i] = [128.0 + theta.cos() * 110.0, 136.0 + theta.sin() * 110.0];
        }
        // Left brow (head=24, mid=27, tail=31)
        b.points194[24] = [110.0, 96.0];
        b.points194[27] = [90.0, 92.0];
        b.points194[31] = [70.0, 96.0];
        // Right brow (head=44, mid=47, tail=51)
        b.points194[44] = [146.0, 96.0];
        b.points194[47] = [166.0, 92.0];
        b.points194[51] = [186.0, 96.0];

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
}
