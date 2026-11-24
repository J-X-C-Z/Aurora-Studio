//! Capture One's own AI segmentation models (`subjectMasking.onnx`,
//! `skinMasking.onnx`), run through ONNX Runtime.
//!
//! The models are the ones a Capture One install ships under
//! `AIModels/Common`: a 320×320 subject masker and a 512×512 six-class people
//! parser. The parser's classes are background, hair, body skin, face skin,
//! clothing and a held object; there is no eye or lip class. Sky has no model
//! at all, so it is a colour heuristic over the subject mask.
//! `AURORA_AIMODELS` points at a folder holding copies of the two files, for
//! machines without Capture One.
//!
//! `AURORA_ORT_LIB` overrides the ONNX Runtime library. Everything degrades to a
//! warning when either is missing; nothing here panics.

pub(crate) mod ort;
pub mod body_pose;

use std::collections::HashMap;
use std::os::raw::c_int;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result, bail};
use rayon::prelude::*;

pub(crate) use ort::{Provider, Runtime, Session};

/// Loaded ONNX Runtime: library path, reported version, API table version.
/// `None` when no usable library was found.
pub fn runtime_info() -> Option<(String, String, u32)> {
    let (path, version, api) = Runtime::get()?.describe();
    Some((path.display().to_string(), version.to_string(), api))
}

/// Snapshot of the provider selection this process actually made. `requested`
/// is the plan; a session's `provider()` is what was really opened.
#[derive(Clone, Debug)]
pub struct ProviderStatus {
    pub exported: Vec<String>,
    pub requested: Vec<String>,
}

pub fn provider_status() -> Option<ProviderStatus> {
    let runtime = Runtime::get()?;
    let plan = runtime.provider_plan();
    Some(ProviderStatus {
        exported: plan.exported.iter().map(|p| p.label().to_string()).collect(),
        requested: plan.labels().into_iter().map(str::to_string).collect(),
    })
}

/// Per-session identity, execution and timing report. All static verification
/// only; GPU node placement inside a session needs an ORT profile on real
/// hardware and is not claimed here.
#[derive(Clone, Debug)]
pub struct SessionStatus {
    pub key: String,
    pub provider: String,
    pub requested: String,
    pub fallback: Option<String>,
    pub input: (String, Vec<i64>, String),
    pub outputs: Vec<(String, Vec<i64>, String, String)>,
    pub cold_ms: f64,
    pub warmup_ms: f64,
    /// Per-run microseconds: input preparation, inference, output conversion.
    pub stages_us: (u64, u64, u64),
    pub measured_runs: u64,
}

fn session_status(session: &Session) -> SessionStatus {
    let (dtype, dims, layout) = session.input_layout();
    let timing = session.timing();
    let stages = timing.stages();
    let (requested, fallback) = session.provider_trace();
    SessionStatus {
        key: session.session_key().short(),
        provider: session.provider().label().to_string(),
        requested,
        fallback,
        input: (dtype, dims.to_vec(), layout.to_string()),
        outputs: session
            .output_layouts()
            .into_iter()
            .map(|(name, dims, dtype, layout)| (name, dims, dtype, layout.to_string()))
            .collect(),
        cold_ms: timing.cold_us() as f64 / 1000.0,
        warmup_ms: timing.warmup_us() as f64 / 1000.0,
        stages_us: stages,
        measured_runs: timing.measured_runs(),
    }
}

/// Status of a loaded model's session, including what provider it opened on
/// and the recorded cold/warm timings. `None` when the model cannot be loaded.
pub fn model_session_status(name: &str) -> Option<SessionStatus> {
    Segmenter::load(name).ok().map(|model| session_status(&model.session))
}

/// Format a status line for probe binaries and selfcheck logs.
pub fn format_session_status(name: &str, status: &SessionStatus) -> String {
    format!(
        "{name}: key={} provider={} requested=[{}] fallback={:?} input={:?} {:?} {} \
         cold={:.3}ms warmup={:.3}ms stages_us={:?} runs={}",
        status.key,
        status.provider,
        status.requested,
        status.fallback,
        status.input.1,
        status.input.0,
        status.input.2,
        status.cold_ms,
        status.warmup_ms,
        status.stages_us,
        status.measured_runs
    )
}

/// The model's output for one image, sampled in normalized image coordinates.
pub struct MaskPlane {
    pub w: usize,
    pub h: usize,
    pub data: Vec<f32>,
}

impl MaskPlane {
    pub fn new(w: usize, h: usize, data: Vec<f32>) -> Self {
        Self { w, h, data }
    }

    /// Bilinear sample at normalized `p`; `0.0` outside the plane.
    pub fn sample(&self, p: [f32; 2]) -> f32 {
        if self.w == 0 || self.h == 0 || !(0.0..=1.0).contains(&p[0]) || !(0.0..=1.0).contains(&p[1]) {
            return 0.0;
        }
        let x = (p[0] * self.w as f32 - 0.5).clamp(0.0, (self.w as f32 - 1.001).max(0.0));
        let y = (p[1] * self.h as f32 - 0.5).clamp(0.0, (self.h as f32 - 1.001).max(0.0));
        let (x0, y0) = (x as usize, y as usize);
        let (x1, y1) = ((x0 + 1).min(self.w - 1), (y0 + 1).min(self.h - 1));
        let (tx, ty) = (x - x0 as f32, y - y0 as f32);
        let at = |xx: usize, yy: usize| self.data[yy * self.w + xx];
        let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * tx;
        let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * tx;
        (top + (bottom - top) * ty).clamp(0.0, 1.0)
    }

    /// Share of the mask above 0.5, for logging and tests.
    pub fn foreground_fraction(&self) -> f32 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().filter(|v| **v > 0.5).count() as f32 / self.data.len() as f32
    }
}

const MAX_CACHED_MASKS: usize = 16;

fn masks() -> &'static Mutex<HashMap<String, Arc<MaskPlane>>> {
    static MASKS: OnceLock<Mutex<HashMap<String, Arc<MaskPlane>>>> = OnceLock::new();
    MASKS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Caches one computed mask under the key `ImageEngine::compute_ai_mask` hands
/// back to the interface.
pub fn store(key: &str, plane: Arc<MaskPlane>) {
    let mut masks = masks().lock().unwrap_or_else(|e| e.into_inner());
    // shortcut: whole-cache eviction, enough for a few hundred megapixels;
    // switch to LRU if sessions ever hold many masked images at once.
    if masks.len() >= MAX_CACHED_MASKS {
        masks.clear();
    }
    masks.insert(key.to_string(), plane);
}

pub fn lookup(key: &str) -> Option<Arc<MaskPlane>> {
    masks().lock().unwrap_or_else(|e| e.into_inner()).get(key).cloned()
}

/// Folders a Capture One install keeps its models and `onnxruntime.dll` in.
pub(crate) fn c1_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(dir) = std::env::var_os("AURORA_C1_DIR") {
        roots.push(PathBuf::from(dir));
    }
    if let Some(program_files) = std::env::var_os("ProgramFiles") {
        roots.push(PathBuf::from(program_files).join("Capture One").join("Capture One"));
    }
    roots
}

fn model_candidates(name: &str) -> Vec<PathBuf> {
    let subfolder = match name {
        "subjectMasking.onnx" => "AIModels/Common/SubjectMasking",
        "skinMasking.onnx" => "AIModels/Common/PeopleMasking",
        "blemishRemoval.onnx" => "AIModels/Common/BlemishRemoval",
        "beautyMarks.onnx" => "AIModels/Common/BeautyMarks",
        "dust.onnx" => "AIModels/Common/DustRemoval",
        "faceDetection.onnx" => "AIModels/Common/FaceDetection",
        "faceMesh.onnx" => "AIModels/Common/FaceMesh",
        "similarity.onnx" => "AIModels/Common/Similarity",
        "eSamEncoderPreUInt8.onnx" => "AIModels/Common/SamEncoder",
        "eSamDecoder.onnx" => "AIModels/Common/SamDecoder",
        "EyeDetector.onnx" | "DenseNet121_FP32.onnx" => "AIModels/Common/ClosedEyesDetection",
        "autoRotation.onnx" => "AIModels/Common/Rotation",
        "MatchLookColor.onnx" | "MatchLookBw.onnx" | "MatchLookBrightness.onnx"
        | "MatchLookExp.onnx" => "AIModels/Common/MatchLook",
        _ => "AIModels/Common",
    };
    let mut paths = Vec::new();
    if let Some(fp16) = fp16_sibling(name).filter(|_| directml_first()) {
        if let Some(dir) = std::env::var_os("AURORA_AIMODELS") {
            let dir = PathBuf::from(dir);
            paths.push(dir.join("DirectML").join(fp16));
            paths.push(dir.join("AIModels").join("DirectML").join(fp16));
        }
        paths.push(PathBuf::from("AIModels/DirectML").join(fp16));
        for root in c1_roots() {
            paths.push(root.join("AIModels").join("DirectML").join(fp16));
        }
    }
    if let Some(dir) = std::env::var_os("AURORA_AIMODELS") {
        let dir = PathBuf::from(dir);
        paths.push(dir.join(name));
        paths.push(dir.join(subfolder).join(name));
        paths.push(dir.join(subfolder.rsplit('/').next().unwrap_or("")).join(name));
    }
    // The subfolder is also a valid relative location, which keeps the search
    // independent of where Capture One is installed (or whether it is at all).
    paths.push(PathBuf::from(subfolder).join(name));
    for root in c1_roots() {
        paths.push(root.join(subfolder).join(name));
        paths.push(root.join(name));
    }
    paths
}

fn directml_first() -> bool {
    Runtime::get().is_some_and(|runtime| {
        runtime.available_providers().first() == Some(&Provider::DirectMl)
    })
}

fn fp16_sibling(name: &str) -> Option<&'static str> {
    // FP16 note (measured 2026-11-23): TRT FP16 keeps mask IoU >= 0.998 on
    // subject/skin, but raw probabilities shift by 6-8%. Use FP16 masks only;
    // never use FP16 probabilities for feathering/alpha.
    match name {
        "subjectMasking.onnx" => Some("subjectMaskingFP16.onnx"),
        "skinMasking.onnx" => Some("skinMaskingFP16.onnx"),
        _ => None,
    }
}

/// Measured cold TensorRT engine build time per model (RTX 4070 Ti, 2026-11-23).
/// Used by the selfcheck to price "TRT first" before it is enabled.
const TRT_BUILD_SECONDS: &[(&str, u64)] = &[
    ("subjectMasking.onnx", 67),
    ("skinMasking.onnx", 54),
    ("blemishRemoval.onnx", 55),
    ("2102828093.onnx", 40),
    ("dust.onnx", 47),
];

pub(crate) fn model_path(name: &str) -> Result<PathBuf> {
    let candidates = model_candidates(name);
    if let Some(path) = candidates.iter().find(|p| p.is_file()) {
        return Ok(path.clone());
    }
    bail!(
        "AI model {name} not found; set AURORA_AIMODELS to a folder with the file (looked in {})",
        candidates.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
    )
}

/// Whether a model file exists anywhere [`model_path`] would look, without
/// opening the ONNX session.
pub fn model_available(name: &str) -> bool {
    model_candidates(name).iter().any(|path| path.is_file())
}

/// Whether prewarm runs one zero inference per model.
fn warmup_enabled() -> bool {
    !std::env::var("AURORA_AI_WARMUP")
        .ok()
        .is_some_and(|value| matches!(value.trim().to_ascii_lowercase().as_str(), "0" | "off" | "false"))
}

/// Builds the reusable ONNX sessions ahead of the first interactive mask.
/// Subject and skin are independent models, so they load in parallel on
/// below-normal-priority threads. A per-model lock prevents duplicate sessions
/// without serializing different models behind one global table lock.
pub fn prewarm_models() -> Result<()> {
    fn prewarm(name: &str) -> Result<()> {
        lower_current_thread_priority();
        if model_available(name) {
            let model = Segmenter::load(name)?;
            // First Run on the prewarm thread: allocator and kernel setup for
            // the EP happens here instead of on the first mask. A failure is
            // logged but does not fail prewarm; the first real mask retries.
            // `AURORA_AI_WARMUP=0` skips it for low-power machines and A/B runs.
            if warmup_enabled() {
                if let Err(error) = model.warmup() {
                    log::warn!("{name} warmup failed: {error:#}");
                }
            }
        }
        Ok(())
    }

    std::thread::scope(|scope| {
        let subject = scope.spawn(|| prewarm("subjectMasking.onnx"));
        let skin = scope.spawn(|| prewarm("skinMasking.onnx"));
        subject
            .join()
            .map_err(|_| anyhow::anyhow!("subject AI prewarm thread panicked"))??;
        skin.join()
            .map_err(|_| anyhow::anyhow!("skin AI prewarm thread panicked"))??;
        Ok(())
    })
}

/// Headless GPU acceptance check: opens subject/skin/blemish on the selected
/// provider and reports warm timings. Triggered by `--gpu-selfcheck` or
/// `AURORA_GPU_SELFCHECK=1`; it never opens a window.
pub fn gpu_selfcheck() -> Result<()> {
    let runtime = Runtime::get().context("ONNX Runtime is not available")?;
    let (library, version, api) = runtime.describe();
    println!("Aurora GPU selfcheck");
    println!("  runtime  : {} v{} api{}", library.display(), version, api);
    let providers = runtime.available_providers();
    println!(
        "  providers: {:?}",
        providers.iter().map(|provider| provider.label()).collect::<Vec<_>>()
    );
    for name in [
        "blemishRemoval.onnx",
        "beautyMarks.onnx",
        "dust.onnx",
        "faceDetection.onnx",
        "faceMesh.onnx",
        "similarity.onnx",
        "eSamEncoderPreUInt8.onnx",
        "eSamDecoder.onnx",
    ] {
        match model_path(name) {
            Ok(path) => println!("  path {name}: {}", path.display()),
            Err(_) => println!("  path {name}: MISSING"),
        }
    }
    let cache_present = ort::trt_cache_present();
    println!("  trt_cache_present: {cache_present}");
    println!("  trt_first_allowed: {}", ort::trt_first_allowed());
    let measured: Vec<_> = TRT_BUILD_SECONDS
        .iter()
        .filter(|(name, _)| model_available(name))
        .collect();
    let estimate: u64 = measured.iter().map(|(_, seconds)| *seconds).sum();
    println!(
        "  trt_first_estimate_s: {estimate} ({} installed models with a measured cold build)",
        measured.len()
    );
    if !cache_present {
        println!(
            "  note: TensorRT stays behind CUDA until AURORA_TRT_CACHE_DIR holds an \
             .engine/.profile or AURORA_ORT_PROVIDERS=tensorrt,... is set"
        );
    }
    for name in ["subjectMasking.onnx", "skinMasking.onnx", "blemishRemoval.onnx"] {
        let path = match model_path(name) {
            Ok(path) => path,
            Err(error) => {
                println!("  {name}: skipped ({error})");
                continue;
            }
        };
        let started = std::time::Instant::now();
        let session = runtime.open_session(&path)?;
        let open_ms = started.elapsed().as_secs_f64() * 1000.0;
        let dims = session.input.dims.clone();
        let count: usize = dims.iter().map(|dim| (*dim).max(1) as usize).product();
        let mut input = vec![0.0f32; count];
        runtime.run(&session, &mut input, &dims)?;
        let mut warm = Vec::new();
        for _ in 0..3 {
            let started = std::time::Instant::now();
            runtime.run(&session, &mut input, &dims)?;
            warm.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        let warm = warm
            .iter()
            .map(|ms| format!("{ms:.2}"))
            .collect::<Vec<_>>()
            .join("/");
        let (requested, fallback) = session.provider_trace();
        let timing = session.timing();
        println!(
            "  {name}: provider={} key={} requested=[{requested}] fallback={fallback:?} \
             open={open_ms:.0}ms cold={:.1}ms warmup={:.1}ms dims={dims:?} warm_ms=[{warm}]",
            session.provider().label(),
            session.session_key().short(),
            timing.cold_us() as f64 / 1000.0,
            timing.warmup_us() as f64 / 1000.0,
        );
    }
    if let Some(status) = model_session_status("subjectMasking.onnx") {
        println!("  status {}", format_session_status("subjectMasking.onnx", &status));
    }
    if let Some(status) = model_session_status("skinMasking.onnx") {
        println!("  status {}", format_session_status("skinMasking.onnx", &status));
    }
    Ok(())
}

#[cfg(windows)]
fn lower_current_thread_priority() {
    use std::ffi::c_void;
    unsafe extern "system" {
        fn GetCurrentThread() -> *mut c_void;
        fn SetThreadPriority(thread: *mut c_void, priority: i32) -> i32;
    }
    const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
    unsafe {
        let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
}

#[cfg(not(windows))]
fn lower_current_thread_priority() {}

/// Segment timings for one mask run. `load_ms` is the lookup/load cost (near
/// zero after prewarm); `cold_ms` is the session creation this session paid,
/// reported again on every run so a cold first mask is attributable.
#[derive(Clone, Debug)]
pub struct MaskTiming {
    pub model: String,
    pub provider: &'static str,
    pub session_key: String,
    pub load_ms: f64,
    pub cold_ms: f64,
    pub preprocess_ms: f64,
    pub infer_ms: f64,
    pub postprocess_ms: f64,
    pub output_dtype: String,
}

fn mask_timings() -> &'static Mutex<HashMap<String, MaskTiming>> {
    static TIMINGS: OnceLock<Mutex<HashMap<String, MaskTiming>>> = OnceLock::new();
    TIMINGS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Last recorded segment timings for a model, for `--gpu-selfcheck` and tests.
pub fn last_mask_timing(model: &str) -> Option<MaskTiming> {
    mask_timings()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(model)
        .cloned()
}

fn record_mask_timing(timing: MaskTiming) {
    log::info!(
        "AI mask {}: provider={} key={} load={:.2}ms cold={:.1}ms preprocess={:.2}ms \
         infer={:.2}ms post={:.2}ms dtype={}",
        timing.model,
        timing.provider,
        timing.session_key,
        timing.load_ms,
        timing.cold_ms,
        timing.preprocess_ms,
        timing.infer_ms,
        timing.postprocess_ms,
        timing.output_dtype
    );
    mask_timings()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .insert(timing.model.clone(), timing);
}

/// A model output may only feed feathering when it is a finite probability.
/// FP16 outputs go through the same gate: half-float rounding is acceptable,
/// NaN/Inf and logits are not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlaneCheck {
    Probability,
    Finite,
}

fn validate_plane(name: &str, plane: &[f32], dtype: c_int, check: PlaneCheck) -> Result<()> {
    let mut non_finite = 0usize;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    for &value in plane {
        if !value.is_finite() {
            non_finite += 1;
            continue;
        }
        min = min.min(value);
        max = max.max(value);
    }
    if non_finite != 0 {
        bail!(
            "{name}: {non_finite} non-finite values in {} output",
            ort::dtype_name(dtype)
        );
    }
    if plane.is_empty() {
        bail!("{name}: model returned an empty output");
    }
    if check == PlaneCheck::Probability && (min < -0.02 || max > 1.02) {
        bail!(
            "{name}: {} output range [{min:.4}, {max:.4}] is not a probability mask; \
             refusing to feather",
            ort::dtype_name(dtype)
        );
    }
    Ok(())
}

/// A loaded model plus what its metadata says about the tensors.
pub(crate) struct Segmenter {
    pub(crate) session: Session,
    pub(crate) path: PathBuf,
    pub(crate) in_w: usize,
    pub(crate) in_h: usize,
    pub(crate) nchw: bool,
    pub(crate) channels: usize,
    pub(crate) out_nchw: bool,
}

type ModelCache = Mutex<HashMap<String, Arc<Segmenter>>>;

fn model_cache() -> &'static ModelCache {
    static MODELS: OnceLock<ModelCache> = OnceLock::new();
    MODELS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn model_lock(name: &str) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();
    let mut locks = LOCKS
        .get_or_init(|| Mutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    locks
        .entry(name.to_owned())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

impl Segmenter {
    fn load(name: &str) -> Result<Arc<Segmenter>> {
        Self::load_with_cpu_fallback(name, false)
    }

    fn load_with_cpu_fallback(name: &str, cpu_only: bool) -> Result<Arc<Segmenter>> {
        let load_lock = model_lock(name);
        let _loading = load_lock.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(model) = model_cache()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(name)
            .filter(|model| !cpu_only || model.session.provider() == Provider::Cpu)
        {
            return Ok(model.clone());
        }
        let full_precision_path = || model_candidates(name)
            .into_iter()
            .find(|candidate| candidate.file_name().is_some_and(|file| file == name) && candidate.is_file());
        let runtime = Runtime::get().context("ONNX Runtime is not available")?;
        let (path, session) = if cpu_only {
            let path = full_precision_path().with_context(|| format!("missing CPU model {name}"))?;
            let session = runtime.open_cpu_session(&path)?;
            (path, session)
        } else {
            let path = model_path(name)?;
            match runtime.open_session(&path) {
                Ok(session) => (path, session),
                Err(error) if path.file_name().is_some_and(|file| fp16_sibling(name).is_some_and(|fp16| file == fp16)) => {
                    let Some(fallback) = full_precision_path() else {
                        return Err(error);
                    };
                    log::warn!("DirectML float16 model failed for {}; retrying {}", path.display(), fallback.display());
                    let session = runtime.open_session(&fallback)?;
                    (fallback, session)
                }
                Err(error) => return Err(error),
            }
        };
        let dims = &session.input.dims;
        if dims.len() != 4
            || (session.input.element_type != ort::DT_FLOAT
                && session.input.element_type != ort::DT_FLOAT16)
        {
            bail!(
                "{name}: expected one float32/float16 NCHW/NHWC input, got {dims:?} dtype {}",
                ort::dtype_name(session.input.element_type)
            );
        }
        // [1,3,H,W] is NCHW, [1,H,W,3] is NHWC.
        let nchw = dims[1] == 3;
        let side = |v: i64, fallback: usize| if v > 0 { v as usize } else { fallback };
        let (in_h, in_w) = if nchw {
            (side(dims[2], 320), side(dims[3], 320))
        } else {
            (side(dims[1], 320), side(dims[2], 320))
        };
        let out_dims = &session.outputs[0].dims;
        let channels = out_dims
            .iter()
            .copied()
            .find(|d| *d > 1 && *d < 64)
            .unwrap_or(1) as usize;
        let out_nchw = channels > 1 && out_dims.get(1) == Some(&(channels as i64));
        let output_dtype = session.outputs[0].element_type;
        if output_dtype != ort::DT_FLOAT && output_dtype != ort::DT_FLOAT16 {
            bail!(
                "{name}: output dtype {} is neither float32 nor float16",
                ort::dtype_name(output_dtype)
            );
        }
        log::info!(
            "AI session {name}: provider={} key={} input {:?} {} {} output {:?} {} {}",
            session.provider().label(),
            session.session_key().short(),
            dims,
            ort::dtype_name(session.input.element_type),
            if nchw { "NCHW" } else { "NHWC" },
            out_dims,
            ort::dtype_name(output_dtype),
            if out_nchw { "NCHW" } else { "NHWC" }
        );
        let model = Arc::new(Segmenter {
            session,
            path,
            in_w,
            in_h,
            nchw,
            channels,
            out_nchw,
        });
        model_cache()
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(name.to_string(), model.clone());
        Ok(model)
    }

    /// Loads the model and runs it on `rgba`, retrying on the CPU session when
    /// the GPU session fails or returns a plane that fails `check`. Returns the
    /// output plus the timing segments; the caller times post-processing.
    fn run_on_image(
        name: &str,
        rgba: &[u8],
        width: usize,
        height: usize,
        check: PlaneCheck,
    ) -> Result<MaskRun> {
        let load_started = std::time::Instant::now();
        let model = Self::load(name)?;
        let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;
        let attempt = |model: &Arc<Segmenter>| -> Result<(Vec<f32>, c_int, f64, f64)> {
            let preprocess_started = std::time::Instant::now();
            let mut input = model.input_tensor(rgba, width, height);
            let preprocess_ms = preprocess_started.elapsed().as_secs_f64() * 1000.0;
            let infer_started = std::time::Instant::now();
            let (output, dtype) = model.run_typed(&mut input)?;
            let infer_ms = infer_started.elapsed().as_secs_f64() * 1000.0;
            validate_plane(name, &output, dtype, check)?;
            Ok((output, dtype, preprocess_ms, infer_ms))
        };
        let (model, output, dtype, preprocess_ms, infer_ms) = match attempt(&model) {
            Ok(result) => (model, result.0, result.1, result.2, result.3),
            Err(error) if model.session.provider() != Provider::Cpu => {
                log::warn!(
                    "{name} failed on {}: {error:#}; retrying on CPU",
                    model.session.provider().label()
                );
                let cpu = Self::load_with_cpu_fallback(name, true)
                    .with_context(|| format!("{name} CPU session after failure: {error:#}"))?;
                let (output, dtype, preprocess_ms, infer_ms) = attempt(&cpu)
                    .with_context(|| format!("{name} CPU retry after failure: {error:#}"))?;
                (cpu, output, dtype, preprocess_ms, infer_ms)
            }
            Err(error) => return Err(error),
        };
        let timing = MaskTiming {
            model: name.to_string(),
            provider: model.session.provider().label(),
            session_key: model.session.session_key().short(),
            load_ms,
            cold_ms: model.session.timing().cold_us() as f64 / 1000.0,
            preprocess_ms,
            infer_ms,
            postprocess_ms: 0.0,
            output_dtype: ort::dtype_name(dtype),
        };
        Ok(MaskRun { model, output, timing })
    }

    fn input_tensor(&self, rgba: &[u8], w: usize, h: usize) -> Vec<f32> {
        let mut tensor = vec![0.0f32; 3 * self.in_w * self.in_h];
        let xs: Vec<_> = (0..self.in_w)
            .map(|x| {
                let sx = (x as f32 + 0.5) * w as f32 / self.in_w as f32 - 0.5;
                let x0 = sx.floor().max(0.0) as usize;
                (x0, (x0 + 1).min(w - 1), (sx - x0 as f32).clamp(0.0, 1.0))
            })
            .collect();
        let ys: Vec<_> = (0..self.in_h)
            .map(|y| {
                let sy = (y as f32 + 0.5) * h as f32 / self.in_h as f32 - 0.5;
                let y0 = sy.floor().max(0.0) as usize;
                (y0, (y0 + 1).min(h - 1), (sy - y0 as f32).clamp(0.0, 1.0))
            })
            .collect();
        let plane = self.in_w * self.in_h;
        if self.nchw {
            tensor.par_chunks_mut(plane).enumerate().for_each(|(c, channel)| {
                for (y, &(y0, y1, fy)) in ys.iter().enumerate() {
                    for (x, &(x0, x1, fx)) in xs.iter().enumerate() {
                        let at = |xx: usize, yy: usize| rgba[(yy * w + xx) * 4 + c] as f32 / 255.0;
                        let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
                        let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
                        channel[y * self.in_w + x] = top * (1.0 - fy) + bottom * fy;
                    }
                }
            });
        } else {
            tensor.par_chunks_mut(self.in_w * 3).enumerate().for_each(|(y, row)| {
                let (y0, y1, fy) = ys[y];
                for (x, &(x0, x1, fx)) in xs.iter().enumerate() {
                    let at = |xx: usize, yy: usize, c: usize| {
                        rgba[(yy * w + xx) * 4 + c] as f32 / 255.0
                    };
                    for c in 0..3 {
                        let top = at(x0, y0, c) * (1.0 - fx) + at(x1, y0, c) * fx;
                        let bottom = at(x0, y1, c) * (1.0 - fx) + at(x1, y1, c) * fx;
                        row[x * 3 + c] = top * (1.0 - fy) + bottom * fy;
                    }
                }
            });
        }
        tensor
    }

    /// Runs the model and returns its first output plus that output's dtype.
    fn run_typed(&self, input: &mut [f32]) -> Result<(Vec<f32>, c_int)> {
        let runtime = Runtime::get().context("ONNX Runtime is not available")?;
        let shape: [i64; 4] = if self.nchw {
            [1, 3, self.in_h as i64, self.in_w as i64]
        } else {
            [1, self.in_h as i64, self.in_w as i64, 3]
        };
        let mut outputs = runtime.run_typed(&self.session, input, &shape)?;
        if outputs.is_empty() {
            bail!("{}: model returned no outputs", self.path.display());
        }
        let plane = outputs.swap_remove(0);
        if !plane.dims.is_empty() && plane.dims.iter().all(|dim| *dim > 0) {
            let expected: i64 = plane.dims.iter().product();
            if expected as usize != plane.values.len() {
                bail!(
                    "{}: output shape {:?} implies {expected} values, got {}",
                    self.path.display(),
                    plane.dims,
                    plane.values.len()
                );
            }
        }
        Ok((plane.values, plane.dtype))
    }

    /// Runs the model and returns its first output at model resolution.
    fn run(&self, input: &mut [f32]) -> Result<Vec<f32>> {
        self.run_typed(input).map(|(values, _)| values)
    }

    /// One inference on a zero tensor at model resolution. Called from the
    /// prewarm thread (below normal priority) so the first interactive mask
    /// does not pay allocator/kernel setup; the session records the warmup time.
    fn warmup(&self) -> Result<f64> {
        let started = std::time::Instant::now();
        self.run(&mut vec![0.0f32; 3 * self.in_w * self.in_h])?;
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        log::info!(
            "AI warmup {}: provider={} {ms:.2} ms",
            self.path
                .file_name()
                .map(|name| name.to_string_lossy().into_owned())
                .unwrap_or_else(|| self.path.display().to_string()),
            self.session.provider().label()
        );
        Ok(ms)
    }
}

/// One inference result plus the segments its caller should extend with the
/// post-processing time it measures itself.
struct MaskRun {
    model: Arc<Segmenter>,
    output: Vec<f32>,
    timing: MaskTiming,
}

impl MaskRun {
    fn finish(mut self, postprocess_ms: f64) -> Arc<Segmenter> {
        self.timing.postprocess_ms = postprocess_ms;
        record_mask_timing(self.timing);
        self.model
    }
}

fn check_rgba(rgba: &[u8], w: usize, h: usize) -> Result<()> {
    if rgba.len() != w.saturating_mul(h).saturating_mul(4) {
        bail!("expected {} RGBA bytes for {w}x{h}, got {}", w * h * 4, rgba.len());
    }
    Ok(())
}

/// Bilinear resample of a single-channel plane.
fn resample(src: &[f32], sw: usize, sh: usize, w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let xs: Vec<_> = (0..w)
        .map(|x| {
            let sx = (x as f32 + 0.5) * sw as f32 / w as f32 - 0.5;
            let x0 = sx.floor().max(0.0) as usize;
            (x0, (x0 + 1).min(sw - 1), (sx - x0 as f32).clamp(0.0, 1.0))
        })
        .collect();
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let sy = (y as f32 + 0.5) * sh as f32 / h as f32 - 0.5;
        let y0 = sy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(sh - 1);
        let fy = (sy - y0 as f32).clamp(0.0, 1.0);
        for (x, &(x0, x1, fx)) in xs.iter().enumerate() {
            let at = |xx: usize, yy: usize| src[yy * sw + xx];
            let top = at(x0, y0) + (at(x1, y0) - at(x0, y0)) * fx;
            let bottom = at(x0, y1) + (at(x1, y1) - at(x0, y1)) * fx;
            row[x] = (top + (bottom - top) * fy).clamp(0.0, 1.0);
        }
    });
    out
}

/// Subject/people/clothing mask: 0..1 per pixel of the input image.
pub fn subject_mask(rgba: &[u8], w: u32, h: u32) -> Result<Vec<f32>> {
    let (w, h) = (w as usize, h as usize);
    check_rgba(rgba, w, h)?;
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    let run = Segmenter::run_on_image("subjectMasking.onnx", rgba, w, h, PlaneCheck::Probability)?;
    let model = run.model.clone();
    let post_started = std::time::Instant::now();
    let expected = model.in_w * model.in_h;
    if run.output.len() < expected {
        bail!(
            "subjectMasking.onnx returned {} values, expected {expected}",
            run.output.len()
        );
    }
    let mask = resample(&run.output[..expected], model.in_w, model.in_h, w, h);
    run.finish(post_started.elapsed().as_secs_f64() * 1000.0);
    Ok(mask)
}

/// The six classes the people parser scores per pixel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PeopleClass {
    Background = 0,
    Hair = 1,
    BodySkin = 2,
    FaceSkin = 3,
    Clothing = 4,
    HeldObject = 5,
}

/// Softmax probability of one parser class, resampled to the input size.
pub fn people_class_mask(rgba: &[u8], w: u32, h: u32, class: PeopleClass) -> Result<Vec<f32>> {
    let (w, h) = (w as usize, h as usize);
    check_rgba(rgba, w, h)?;
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    let run = Segmenter::run_on_image("skinMasking.onnx", rgba, w, h, PlaneCheck::Finite)?;
    let model = run.model.clone();
    let post_started = std::time::Instant::now();
    let pixels = model.in_w * model.in_h;
    let classes = model.channels.max(1);
    if run.output.len() < pixels * classes {
        bail!(
            "skinMasking.onnx returned {} values, expected {}",
            run.output.len(),
            pixels * classes
        );
    }
    let plane = class_plane(&run.output, pixels, classes, model.out_nchw, class as usize);
    let mask = resample(&plane, model.in_w, model.in_h, w, h);
    run.finish(post_started.elapsed().as_secs_f64() * 1000.0);
    Ok(mask)
}

/// Face skin, body skin and hair from one parser inference.
pub struct PeopleMasks {
    pub face: Vec<f32>,
    pub body: Vec<f32>,
    pub hair: Vec<f32>,
}

/// Runs the people parser once and resamples the three portrait classes.
pub fn people_masks(rgba: &[u8], w: u32, h: u32) -> Result<PeopleMasks> {
    let (w, h) = (w as usize, h as usize);
    check_rgba(rgba, w, h)?;
    if w == 0 || h == 0 {
        return Ok(PeopleMasks {
            face: Vec::new(),
            body: Vec::new(),
            hair: Vec::new(),
        });
    }
    let run = Segmenter::run_on_image("skinMasking.onnx", rgba, w, h, PlaneCheck::Finite)?;
    let model = run.model.clone();
    let post_started = std::time::Instant::now();
    let pixels = model.in_w * model.in_h;
    let classes = model.channels.max(1);
    if run.output.len() < pixels * classes {
        bail!(
            "skinMasking.onnx returned {} values, expected {}",
            run.output.len(),
            pixels * classes
        );
    }
    let [face, body, hair] = class_planes(
        &run.output,
        pixels,
        classes,
        model.out_nchw,
        [
            PeopleClass::FaceSkin as usize,
            PeopleClass::BodySkin as usize,
            PeopleClass::Hair as usize,
        ],
    );
    let (face, (body, hair)) = rayon::join(
        || resample(&face, model.in_w, model.in_h, w, h),
        || {
            rayon::join(
                || resample(&body, model.in_w, model.in_h, w, h),
                || resample(&hair, model.in_w, model.in_h, w, h),
            )
        },
    );
    run.finish(post_started.elapsed().as_secs_f64() * 1000.0);
    Ok(PeopleMasks {
        face,
        body,
        hair,
    })
}

/// Per-pixel softmax probability of `class`, in model resolution.
fn class_plane(out: &[f32], pixels: usize, classes: usize, out_nchw: bool, class: usize) -> Vec<f32> {
    let class = class.min(classes.saturating_sub(1));
    (0..pixels)
        .into_par_iter()
        .map(|i| {
            let at = |c: usize| {
                if out_nchw { out[c * pixels + i] } else { out[i * classes + c] }
            };
            let max = (0..classes).fold(f32::MIN, |m, c| m.max(at(c)));
            let sum: f32 = (0..classes).map(|c| (at(c) - max).exp()).sum();
            if sum > 0.0 { (at(class) - max).exp() / sum } else { 0.0 }
        })
        .collect()
}

/// Three softmax probabilities from one logits traversal. `people_masks`
/// previously repeated the max/sum loop once per class; the selected values
/// and operation order here match [`class_plane`] bit for bit.
fn class_planes(
    out: &[f32],
    pixels: usize,
    classes: usize,
    out_nchw: bool,
    requested: [usize; 3],
) -> [Vec<f32>; 3] {
    let selected = requested.map(|class| class.min(classes.saturating_sub(1)));
    let mut planes = std::array::from_fn(|_| vec![0.0f32; pixels]);
    let [first, second, third] = &mut planes;
    first
        .par_iter_mut()
        .zip(second.par_iter_mut())
        .zip(third.par_iter_mut())
        .enumerate()
        .for_each(|(i, ((first, second), third))| {
            let at = |class: usize| {
                if out_nchw {
                    out[class * pixels + i]
                } else {
                    out[i * classes + class]
                }
            };
            let max = (0..classes).fold(f32::MIN, |maximum, class| maximum.max(at(class)));
            let sum: f32 = (0..classes).map(|class| (at(class) - max).exp()).sum();
            let probabilities = selected.map(|class| {
                if sum > 0.0 {
                    (at(class) - max).exp() / sum
                } else {
                    0.0
                }
            });
            *first = probabilities[0];
            *second = probabilities[1];
            *third = probabilities[2];
        });
    planes
}

/// Face skin from the people parser.
pub fn skin_mask(rgba: &[u8], w: u32, h: u32) -> Result<Vec<f32>> {
    people_class_mask(rgba, w, h, PeopleClass::FaceSkin)
}

/// Hair from the people parser. It expects a portrait-scale face; a small face
/// in a wide frame collapses to background, like C1's own parser inputs.
pub fn hair_mask(rgba: &[u8], w: u32, h: u32) -> Result<Vec<f32>> {
    people_class_mask(rgba, w, h, PeopleClass::Hair)
}

/// How much a pixel's colour looks like sky: blue dominance over red and
/// green. Overcast white sky scores 0, which is why this is only a starting
/// mask. `0..1`.
fn sky_prior(r: f32, g: f32, b: f32) -> f32 {
    if b < 0.35 {
        return 0.0; // dark blue shadows are not sky
    }
    let max = r.max(g).max(b);
    let blue = (b - r.max(g)) / max.max(0.05);
    ((blue - 0.04) / 0.12).clamp(0.0, 1.0)
}

/// Sky has no model in `AIModels/Common`. This is a heuristic: the inverse of
/// the subject mask, weighted by a colour prior and faded out towards the
/// bottom of the frame. It will miss a coloured or overcast sky and can keep
/// blue clothing, so it is a starting mask, not a segmentation.
pub fn sky_mask(rgba: &[u8], w: u32, h: u32) -> Result<Vec<f32>> {
    let (w, h) = (w as usize, h as usize);
    check_rgba(rgba, w, h)?;
    if w == 0 || h == 0 {
        return Ok(Vec::new());
    }
    let subject = subject_mask(rgba, w as u32, h as u32)?;
    Ok(sky_mask_from_subject(rgba, w, h, &subject))
}

/// The sky heuristic after the subject model has already run for this frame.
pub(crate) fn sky_mask_from_subject(rgba: &[u8], w: usize, h: usize, subject: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        // Sky is above the horizon: full weight down to 60%, gone by 90%.
        let vertical = ((0.9 - y as f32 / h as f32) / 0.3).clamp(0.0, 1.0);
        for x in 0..w {
            let i = y * w + x;
            let p = &rgba[i * 4..i * 4 + 4];
            let prior = sky_prior(p[0] as f32 / 255.0, p[1] as f32 / 255.0, p[2] as f32 / 255.0);
            row[x] = (1.0 - subject[i]) * prior * vertical;
        }
    });
    out
}

/// One line per tensor, for the probe binary.
pub fn model_summary(name: &str) -> Result<Vec<String>> {
    let model = Segmenter::load(name)?;
    let dims = |d: &[i64]| d.iter().map(i64::to_string).collect::<Vec<_>>().join("x");
    let mut lines = vec![format!(
        "{} input {:?} [{}] {}",
        model.path.display(),
        model.session.input.name,
        dims(&model.session.input.dims),
        ort::dtype_name(model.session.input.element_type)
    )];
    lines.extend(model.session.outputs.iter().map(|o| {
        format!(
            "  output {:?} [{}] {}",
            o.name,
            dims(&o.dims),
            ort::dtype_name(o.element_type)
        )
    }));
    lines.push(format!(
        "  provider={} requested=[{}] fallback={:?} key={}",
        model.session.provider().label(),
        model.session.provider_trace().0,
        model.session.provider_trace().1,
        model.session.session_key().short()
    ));
    Ok(lines)
}

/// The execution provider a model session actually opened on.
pub fn model_provider(name: &str) -> Option<&'static str> {
    Segmenter::load(name)
        .ok()
        .map(|model| model.session.provider().label())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_is_empty() {
        assert!(subject_mask(&[], 0, 0).unwrap().is_empty());
        assert!(skin_mask(&[], 0, 0).unwrap().is_empty());
        assert!(hair_mask(&[], 0, 0).unwrap().is_empty());
        assert!(sky_mask(&[], 0, 0).unwrap().is_empty());
    }

    /// Cold (model load + session creation) and warm inference for both models,
    /// so a provider can be compared on the same machine. Iteration 0 is cold;
    /// the median of the rest is warm.
    #[test]
    #[ignore = "needs Capture One's ONNX models; prints cold and warm inference times"]
    fn bench_real_models_cold_and_warm() {
        let (w, h) = (640u32, 480u32);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let sky = y < h / 3;
                let subject = (x as i32 - w as i32 / 2).abs() < 120 && y > h / 3;
                let px = if sky { [90, 140, 220] } else if subject { [200, 150, 120] } else { [40, 40, 45] };
                rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
        let runs = 6;
        let benches: [(&str, fn(&[u8], u32, u32) -> Result<Vec<f32>>); 2] =
            [("subject", subject_mask), ("skin", skin_mask)];
        for (name, run) in benches {
            let mut times = Vec::with_capacity(runs);
            for _ in 0..runs {
                let start = std::time::Instant::now();
                let mask = run(&rgba, w, h).expect("model");
                times.push(start.elapsed().as_secs_f64() * 1000.0);
                assert_eq!(mask.len(), (w * h) as usize);
            }
            let mut warm = times[1..].to_vec();
            warm.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            let median = warm[warm.len() / 2];
            println!(
                "{name}: cold {:.0} ms, warm median {:.0} ms over {} runs",
                times[0],
                median,
                runs - 1
            );
        }
    }

    #[test]
    fn size_mismatch_errors() {
        assert!(subject_mask(&[0u8; 12], 2, 2).is_err());
        assert!(skin_mask(&[0u8; 16], 300, 300).is_err());
        assert!(hair_mask(&[0u8; 16], 300, 300).is_err());
        assert!(sky_mask(&[0u8; 16], 300, 300).is_err());
    }

    #[test]
    fn class_indices_match_the_people_parser() {
        assert_eq!(PeopleClass::Background as usize, 0);
        assert_eq!(PeopleClass::Hair as usize, 1);
        assert_eq!(PeopleClass::BodySkin as usize, 2);
        assert_eq!(PeopleClass::FaceSkin as usize, 3);
        assert_eq!(PeopleClass::Clothing as usize, 4);
        assert_eq!(PeopleClass::HeldObject as usize, 5);
    }

    #[test]
    fn class_plane_reads_every_channel_at_model_resolution() {
        // Two pixels, six logits each, NHWC: pixel 0 is class 4, pixel 1 class 1.
        let out = [0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0];
        for class in 0..6 {
            assert_eq!(class_plane(&out, 2, 6, false, class).len(), 2);
        }
        assert!(class_plane(&out, 2, 6, false, 4)[0] > 0.99);
        assert!(class_plane(&out, 2, 6, false, 1)[1] > 0.99);
        // NCHW stores the same two pixels channel-major.
        let mut nchw = [0.0f32; 12];
        nchw[4 * 2] = 8.0;
        nchw[1 * 2 + 1] = 8.0;
        assert!(class_plane(&nchw, 2, 6, true, 4)[0] > 0.99);
        assert!(class_plane(&nchw, 2, 6, true, 1)[1] > 0.99);
    }

    #[test]
    fn fused_class_planes_match_the_single_class_path_bit_for_bit() {
        let (pixels, classes) = (7, 6);
        let out: Vec<f32> = (0..pixels * classes)
            .map(|i| ((i * 37 % 101) as f32) / 17.0 - 3.0)
            .collect();
        let requested = [
            PeopleClass::FaceSkin as usize,
            PeopleClass::BodySkin as usize,
            PeopleClass::Hair as usize,
        ];
        let fused = class_planes(&out, pixels, classes, false, requested);
        for (plane, class) in fused.into_iter().zip(requested) {
            assert_eq!(plane, class_plane(&out, pixels, classes, false, class));
        }
    }

    #[test]
    fn sky_prior_prefers_blue_and_rejects_ground() {
        assert!(sky_prior(0.35, 0.55, 0.85) > 0.8);
        assert_eq!(sky_prior(0.35, 0.30, 0.22), 0.0);
        assert_eq!(sky_prior(0.92, 0.93, 0.95), 0.0); // overcast white sky is not detected
        assert_eq!(sky_prior(0.75, 0.52, 0.42), 0.0); // skin
        assert_eq!(sky_prior(0.20, 0.24, 0.29), 0.0); // dark blue shadow
    }

    #[test]
    fn missing_model_is_an_error_not_a_panic() {
        assert!(model_path("not-a-model.onnx").is_err());
        assert!(Segmenter::load("not-a-model.onnx").is_err());
    }

    #[test]
    fn model_locks_serialize_one_model_without_blocking_another() {
        let subject = model_lock("test:subject");
        let subject_again = model_lock("test:subject");
        let skin = model_lock("test:skin");
        assert!(Arc::ptr_eq(&subject, &subject_again));

        let _subject_loading = subject.lock().unwrap();
        assert!(subject_again.try_lock().is_err());
        assert!(skin.try_lock().is_ok());
    }

    #[test]
    fn cache_returns_the_same_plane() {
        let key = "test:cache:subject";
        assert!(lookup(key).is_none());
        let plane = Arc::new(MaskPlane::new(2, 1, vec![0.0, 1.0]));
        store(key, plane.clone());
        let cached = lookup(key).expect("cached");
        assert!(Arc::ptr_eq(&plane, &cached));
        assert_eq!(cached.sample([0.25, 0.5]), 0.0);
        assert!(cached.sample([0.75, 0.5]) > 0.99);
        assert_eq!(cached.foreground_fraction(), 0.5);
    }

    #[test]
    fn windows_install_root_is_searched() {
        if !cfg!(windows) {
            return;
        }
        assert!(
            c1_roots().iter().any(|root| root.ends_with("Capture One/Capture One")),
            "c1_roots must include the Capture One install root: {:?}",
            c1_roots()
        );
    }

    #[test]
    fn rotation_model_is_searched_in_its_subfolder() {
        let candidates = model_candidates("autoRotation.onnx");
        assert!(
            candidates
                .iter()
                .any(|path| path.ends_with("AIModels/Common/Rotation/autoRotation.onnx")),
            "{candidates:?}"
        );
    }

    /// `AURORA_AIMODELS=~/agents/evidence cargo test -p aurora-engine -- --ignored`
    #[test]
    #[ignore = "needs Capture One's ONNX models"]
    fn real_models_produce_a_mask() {
        let (w, h) = (640u32, 480u32);
        let mut rgba = Vec::with_capacity((w * h * 4) as usize);
        for y in 0..h {
            for x in 0..w {
                let sky = y < h / 3;
                let subject = (x as i32 - w as i32 / 2).abs() < 120 && y > h / 3;
                let px = if sky { [90, 140, 220] } else if subject { [200, 150, 120] } else { [40, 40, 45] };
                rgba.extend_from_slice(&[px[0], px[1], px[2], 255]);
            }
        }
        let start = std::time::Instant::now();
        let subject = subject_mask(&rgba, w, h).expect("subject model");
        let subject_ms = start.elapsed().as_secs_f32() * 1000.0;
        let start = std::time::Instant::now();
        let skin = skin_mask(&rgba, w, h).expect("skin model");
        let skin_ms = start.elapsed().as_secs_f32() * 1000.0;
        for (name, mask, ms) in [("subject", subject, subject_ms), ("skin", skin, skin_ms)] {
            let plane = MaskPlane::new(w as usize, h as usize, mask);
            println!(
                "{name}: {}x{}, {:.0} ms, foreground {:.1}%",
                plane.w,
                plane.h,
                ms,
                plane.foreground_fraction() * 100.0
            );
            assert_eq!(plane.data.len(), (w * h) as usize);
            assert!(plane.data.iter().all(|v| (0.0..=1.0).contains(v)));
        }
    }

    fn print_profile_summary(path: &std::path::Path) -> Result<()> {
        let profile: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path)?).context("reading ORT profile")?;
        let events = profile
            .as_array()
            .or_else(|| profile.get("traceEvents").and_then(|v| v.as_array()))
            .context("ORT profile has no traceEvents")?;
        let mut providers = std::collections::BTreeMap::<String, (usize, f64)>::new();
        let mut nodes = std::collections::BTreeMap::<String, (usize, f64)>::new();
        for event in events {
            if event.get("cat").and_then(|v| v.as_str()) != Some("Node") {
                continue;
            }
            let args = event.get("args");
            let provider = args
                .and_then(|v| v.get("provider"))
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let op = args
                .and_then(|v| v.get("op_name"))
                .and_then(|v| v.as_str())
                .or_else(|| event.get("name").and_then(|v| v.as_str()))
                .unwrap_or("unknown");
            let duration_us = event.get("dur").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let total = providers.entry(provider.to_string()).or_default();
            total.0 += 1;
            total.1 += duration_us;
            let total = nodes.entry(format!("{provider}:{op}")).or_default();
            total.0 += 1;
            total.1 += duration_us;
        }
        println!("profile: {}", path.display());
        for (provider, (count, duration_us)) in &providers {
            println!("provider {provider}: nodes={count} total={:.3} ms", duration_us / 1000.0);
        }
        let mut nodes: Vec<_> = nodes.into_iter().collect();
        nodes.sort_by(|a, b| b.1.1.total_cmp(&a.1.1));
        for (node, (count, duration_us)) in nodes.into_iter().take(8) {
            println!("node {node}: runs={count} total={:.3} ms", duration_us / 1000.0);
        }
        Ok(())
    }

    #[test]
    #[ignore = "needs Capture One's ONNX models and AURORA_ORT_PROFILE_DIR"]
    fn real_models_coldstart_and_node_profile() {
        let (w, h) = (640u32, 480u32);
        let rgba: Vec<u8> = (0..w * h).flat_map(|_| [180u8, 160, 140, 255]).collect();
        let runtime_start = std::time::Instant::now();
        Runtime::get().expect("ONNX Runtime");
        println!("runtime load {:.3} ms", runtime_start.elapsed().as_secs_f64() * 1000.0);
        for (name, run) in [
            ("subjectMasking.onnx", subject_mask as fn(&[u8], u32, u32) -> Result<Vec<f32>>),
            ("skinMasking.onnx", skin_mask as fn(&[u8], u32, u32) -> Result<Vec<f32>>),
        ] {
            let session_start = std::time::Instant::now();
            let model = Segmenter::load(name).expect("model session");
            let session_ms = session_start.elapsed().as_secs_f64() * 1000.0;
            let first_start = std::time::Instant::now();
            let first_mask = run(&rgba, w, h).expect("first inference");
            let first_ms = first_start.elapsed().as_secs_f64() * 1000.0;
            let hash = first_mask.iter().fold(0xcbf29ce484222325u64, |hash, value| {
                (hash ^ u64::from(value.to_bits())).wrapping_mul(0x100000001b3)
            });
            let mut warm = Vec::new();
            for _ in 0..4 {
                let start = std::time::Instant::now();
                run(&rgba, w, h).expect("warm inference");
                warm.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            warm.sort_by(f64::total_cmp);
            println!(
                "{name}: provider={} session={session_ms:.3} ms first={first_ms:.3} ms warm_p50={:.3} ms hash={hash:016x}",
                model.session.provider().label(),
                warm[1]
            );
            if let Some(path) = model.session.end_profiling().expect("end ORT profiling") {
                print_profile_summary(&path).expect("summarize ORT profile");
            }
        }
    }

    #[test]
    #[ignore = "needs Capture One's PeopleMasking model; compares fused softmax output and time"]
    fn real_people_masks_fused_softmax_parity_and_bench() {
        let (w, h) = (640u32, 480u32);
        let rgba: Vec<u8> = (0..w * h).flat_map(|_| [180u8, 160, 140, 255]).collect();
        let model = Segmenter::load("skinMasking.onnx").expect("skin model");
        let mut input = model.input_tensor(&rgba, w as usize, h as usize);
        let out = model.run(&mut input).expect("skin inference");
        let pixels = model.in_w * model.in_h;
        let classes = model.channels.max(1);
        let requested = [
            PeopleClass::FaceSkin as usize,
            PeopleClass::BodySkin as usize,
            PeopleClass::Hair as usize,
        ];
        let old = || {
            requested
                .into_iter()
                .map(|class| {
                    let plane = class_plane(&out, pixels, classes, model.out_nchw, class);
                    resample(&plane, model.in_w, model.in_h, w as usize, h as usize)
                })
                .collect::<Vec<_>>()
        };
        let fused = || {
            let [face, body, hair] =
                class_planes(&out, pixels, classes, model.out_nchw, requested);
            let (face, (body, hair)) = rayon::join(
                || resample(&face, model.in_w, model.in_h, w as usize, h as usize),
                || {
                    rayon::join(
                        || resample(&body, model.in_w, model.in_h, w as usize, h as usize),
                        || resample(&hair, model.in_w, model.in_h, w as usize, h as usize),
                    )
                },
            );
            vec![face, body, hair]
        };
        let old_planes = old();
        let fused_planes = fused();
        assert_eq!(old_planes, fused_planes, "fused softmax output changed");

        let mut old_ms = Vec::new();
        let mut fused_ms = Vec::new();
        for _ in 0..10 {
            let start = std::time::Instant::now();
            std::hint::black_box(old());
            old_ms.push(start.elapsed().as_secs_f64() * 1000.0);
            let start = std::time::Instant::now();
            std::hint::black_box(fused());
            fused_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        old_ms.sort_by(f64::total_cmp);
        fused_ms.sort_by(f64::total_cmp);
        println!(
            "people mask extraction: old_p50={:.3} ms fused_p50={:.3} ms speedup={:.2}x",
            old_ms[old_ms.len() / 2],
            fused_ms[fused_ms.len() / 2],
            old_ms[old_ms.len() / 2] / fused_ms[fused_ms.len() / 2].max(f64::EPSILON)
        );
    }

    #[test]
    #[ignore = "needs Capture One's ONNX models"]
    fn real_models_are_ready_after_prewarm() {
        let (w, h) = (640u32, 480u32);
        let rgba: Vec<u8> = (0..w * h).flat_map(|_| [180u8, 160, 140, 255]).collect();
        let start = std::time::Instant::now();
        prewarm_models().expect("prewarm");
        let prewarm_ms = start.elapsed().as_secs_f64() * 1000.0;
        let subject_start = std::time::Instant::now();
        subject_mask(&rgba, w, h).expect("subject after prewarm");
        let subject_ms = subject_start.elapsed().as_secs_f64() * 1000.0;
        let skin_start = std::time::Instant::now();
        skin_mask(&rgba, w, h).expect("skin after prewarm");
        let skin_ms = skin_start.elapsed().as_secs_f64() * 1000.0;
        println!(
            "prewarm={prewarm_ms:.3} ms subject_first={subject_ms:.3} ms skin_first={skin_ms:.3} ms"
        );
    }

    /// The same models through `FileEngine`: compute, cache, and the kinds no
    /// model covers.
    #[test]
    #[ignore = "needs Capture One's ONNX models"]
    fn file_engine_stores_and_reuses_the_mask() {
        use crate::{AiMaskKind, FileEngine, ImageEngine};

        let (w, h) = (320u32, 240u32);
        let rgba: Vec<u8> = (0..w * h).flat_map(|_| [180u8, 160, 140, 255]).collect();
        let dir = std::env::temp_dir().join(format!("portrait-aimask-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        image::RgbaImage::from_raw(w, h, rgba).unwrap().save(dir.join("frame.png")).unwrap();

        let engine = FileEngine::open(&dir).unwrap();
        let image = engine.images()[0].id;
        let key = engine.compute_ai_mask(image, AiMaskKind::Subject).unwrap();
        let plane = lookup(&key).expect("stored under the returned key");
        assert_eq!(plane.w, 1024);
        assert_eq!(engine.compute_ai_mask(image, AiMaskKind::Subject).unwrap(), key);
        for derived in [AiMaskKind::Person, AiMaskKind::Clothing] {
            let derived_key = crate::mask_store::key("frame.png", 1, derived);
            assert!(lookup(&derived_key).is_some(), "{derived:?} was not derived");
        }
        for derived in [AiMaskKind::Background, AiMaskKind::Sky] {
            let derived_key = engine.compute_ai_mask(image, derived).unwrap();
            assert!(lookup(&derived_key).is_some(), "{derived:?} was not stored");
        }
        let hair_key = engine.compute_ai_mask(image, AiMaskKind::Hair).unwrap();
        assert!(lookup(&hair_key).is_some());
        let skin_key = crate::mask_store::key("frame.png", 1, AiMaskKind::Skin);
        assert!(lookup(&skin_key).is_some(), "skin was not derived with hair");
        assert_eq!(engine.compute_ai_mask(image, AiMaskKind::Skin).unwrap(), skin_key);
        let err = engine.compute_ai_mask(image, AiMaskKind::Eyes).unwrap_err();
        assert!(err.to_string().contains("no Eyes class"), "{err}");
        let err = engine.compute_ai_mask(image, AiMaskKind::Lips).unwrap_err();
        assert!(err.to_string().contains("no Lips class"), "{err}");

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Cold-start accounting per model, sequentially: session creation and the
    /// zero-tensor warmup separately. Useful before/after ORT cache changes.
    #[test]
    #[ignore = "needs Capture One's ONNX models; prints load and warmup segments"]
    fn real_models_load_and_warmup_segments() {
        if Runtime::get().is_none() {
            println!("ONNX Runtime unavailable");
            return;
        }
        for name in ["subjectMasking.onnx", "skinMasking.onnx"] {
            if !model_available(name) {
                println!("{name}: not installed");
                continue;
            }
            let started = std::time::Instant::now();
            let model = Segmenter::load(name).expect("model");
            let load_ms = started.elapsed().as_secs_f64() * 1000.0;
            let started = std::time::Instant::now();
            let warm_ms = model.warmup().expect("warmup") ;
            let timing = model.session.timing();
            println!(
                "{name}: provider={} load={load_ms:.1}ms warmup={warm_ms:.1}ms \
                 session_cold={:.1}ms",
                model.session.provider().label(),
                timing.cold_us() as f64 / 1000.0
            );
        }
    }

    #[test]
    fn probability_validation_rejects_non_finite_and_out_of_range_planes() {
        assert!(validate_plane("m", &[0.0, 0.5, 1.0], ort::DT_FLOAT, PlaneCheck::Probability).is_ok());
        assert!(validate_plane("m", &[0.0, 1.0], ort::DT_FLOAT16, PlaneCheck::Probability).is_ok());
        let nan = validate_plane("m", &[0.5, f32::NAN], ort::DT_FLOAT, PlaneCheck::Probability)
            .unwrap_err();
        assert!(nan.to_string().contains("non-finite"), "{nan}");
        let inf = validate_plane("m", &[f32::INFINITY], ort::DT_FLOAT16, PlaneCheck::Probability)
            .unwrap_err();
        assert!(inf.to_string().contains("non-finite"), "{inf}");
        let logits = validate_plane("m", &[-3.0, 4.0], ort::DT_FLOAT16, PlaneCheck::Probability)
            .unwrap_err();
        assert!(logits.to_string().contains("not a probability mask"), "{logits}");
        // Logits are legal for the people parser as long as they are finite.
        assert!(validate_plane("m", &[-30.0, 4.0], ort::DT_FLOAT16, PlaneCheck::Finite).is_ok());
        let empty = validate_plane("m", &[], ort::DT_FLOAT, PlaneCheck::Finite).unwrap_err();
        assert!(empty.to_string().contains("empty"), "{empty}");
    }

    #[test]
    fn fp16_rounded_probabilities_still_pass_the_feather_gate() {
        // A probability rounded to half precision must be accepted; rejecting
        // it would make FP16 providers unusable, accepting NaN/Inf would let
        // garbage into feathering.
        let rounded: Vec<f32> = [0.0f32, 0.001, 0.5, 0.999, 1.0]
            .into_iter()
            .map(|value| ort::half_to_f32(ort::float_to_half(value)))
            .collect();
        assert!(validate_plane("m", &rounded, ort::DT_FLOAT16, PlaneCheck::Probability).is_ok());
    }

    #[test]
    fn mask_timings_are_recorded_per_model() {
        let name = "test-mask-timing";
        let timing = MaskTiming {
            model: name.to_string(),
            provider: "CPU",
            session_key: "0000000000000000".to_string(),
            load_ms: 1.5,
            cold_ms: 2.5,
            preprocess_ms: 3.5,
            infer_ms: 4.5,
            postprocess_ms: 5.5,
            output_dtype: "float32".to_string(),
        };
        record_mask_timing(timing);
        let stored = last_mask_timing(name).expect("timing stored");
        assert_eq!(stored.model, name);
        assert_eq!(stored.output_dtype, "float32");
        assert_eq!(stored.postprocess_ms, 5.5);
        assert!(last_mask_timing("never-recorded").is_none());
    }
}
