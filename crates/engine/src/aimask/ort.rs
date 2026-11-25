//! The small slice of the ONNX Runtime C API this engine needs, loaded at
//! runtime with `dlopen`.
//!
//! `OrtApi` is an ABI-frozen array of function pointers: every field is one
//! pointer, and the order is fixed per API version. The indices below come from
//! the 1.23.2 header (`ORT_API_VERSION 23`, package `libonnxruntime-dev`) and
//! were confirmed by opening both models and reading their tensor metadata.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::io::Read;
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock, atomic::{AtomicBool, Ordering}};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result, anyhow, bail};

use super::lru::{self, ByteLru};

type Status = *mut c_void;
type ApiPtr = *const *const c_void;

const FLOAT: c_int = 1;
/// `ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16`. DirectML's FP16 model siblings use
/// it for both the input and the output tensor.
const FLOAT16: c_int = 10;
const LOG_ERROR: c_int = 3;
const RTLD_NOW: c_int = 2;
const ORT_API_VERSION: u32 = 23;
/// Oldest API table this module is written against; Capture One ships 1.22.
const MIN_ORT_API_VERSION: u32 = 18;
/// Oldest API table this module is written against; Capture One ships 1.22.
const OPT_ENABLE_ALL: c_int = 99;
#[cfg(windows)]
const ORT_RUNTIME_DIRS_ENV: &str = "AURORA_ORT_RUNTIME_DIRS";
/// Project-relative GPU runtime. A bare `onnxruntime.dll` is never attempted:
/// Windows ships an incompatible 1.17 build in System32 and it wins the search.
const PROJECT_ORT_RELATIVE: &str =
    "90_work/gpu-runtime/python/onnxruntime/capi/onnxruntime.dll";
/// Comma-separated provider order, e.g. `tensorrt,cuda,cpu`.
const PROVIDERS_ENV: &str = "AURORA_ORT_PROVIDERS";
/// Extra DLL directories (semicolon-separated) for cuDNN / TensorRT / CUDA.
const DLL_DIRS_ENV: &str = "AURORA_GPU_DLL_DIRS";
#[cfg(windows)]
const LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR: u32 = 0x0000_0100;
#[cfg(windows)]
const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x0000_1000;
#[cfg(windows)]
const CUDA_RUNTIME_DLLS: &[&str] = &[
    "cudart64_13.dll",
    "cublasLt64_13.dll",
    "cublas64_13.dll",
    "cudart64_12.dll",
    "cublasLt64_12.dll",
    "cublas64_12.dll",
    "cufft64_12.dll",
    "cufft64_11.dll",
    "cudnn_engines_runtime_compiled64_9.dll",
    "cudnn_engines_precompiled64_9.dll",
    "cudnn_heuristic64_9.dll",
    "cudnn_ops64_9.dll",
    "cudnn_adv64_9.dll",
    "cudnn_cnn64_9.dll",
    "cudnn_graph64_9.dll",
    "cudnn64_9.dll",
    "cudnn_engines_tensor_ir64_9.dll",
];

// OrtApi field indices (see module comment).
const GET_ERROR_MESSAGE: usize = 2;
const CREATE_ENV: usize = 3;
const CREATE_SESSION: usize = 7;
const RUN: usize = 9;
const CREATE_SESSION_OPTIONS: usize = 10;
const SET_OPTIMIZED_MODEL_FILE_PATH: usize = 11;
const ENABLE_PROFILING: usize = 14;
const SET_SESSION_GRAPH_OPTIMIZATION_LEVEL: usize = 23;
const SET_INTRA_OP_NUM_THREADS: usize = 24;
const SESSION_GET_INPUT_COUNT: usize = 30;
const SESSION_GET_OUTPUT_COUNT: usize = 31;
const SESSION_GET_INPUT_TYPE_INFO: usize = 33;
const SESSION_GET_OUTPUT_TYPE_INFO: usize = 34;
const SESSION_GET_INPUT_NAME: usize = 36;
const SESSION_GET_OUTPUT_NAME: usize = 37;
const CREATE_TENSOR_WITH_DATA: usize = 49;
const GET_TENSOR_MUTABLE_DATA: usize = 51;
const CAST_TYPE_INFO_TO_TENSOR_INFO: usize = 55;
const GET_TENSOR_ELEMENT_TYPE: usize = 60;
const GET_DIMENSIONS_COUNT: usize = 61;
const GET_DIMENSIONS: usize = 62;
const GET_TENSOR_SHAPE_ELEMENT_COUNT: usize = 64;
const GET_TENSOR_TYPE_AND_SHAPE: usize = 65;
const ALLOCATOR_FREE: usize = 76;
const ALLOCATOR_GET_INFO: usize = 77;
const GET_ALLOCATOR_WITH_DEFAULT_OPTIONS: usize = 78;
const SESSION_END_PROFILING: usize = 110;
const RELEASE_ENV: usize = 92;
const RELEASE_STATUS: usize = 93;
const RELEASE_SESSION: usize = 95;
const RELEASE_VALUE: usize = 96;
const RELEASE_TYPE_INFO: usize = 98;
const RELEASE_TENSOR_TYPE_AND_SHAPE: usize = 99;
const RELEASE_SESSION_OPTIONS: usize = 100;

unsafe extern "C" {
    fn dlopen(file: *const c_char, mode: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *const c_char;
}

#[cfg(windows)]
unsafe extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
    fn FreeLibrary(module: *mut c_void) -> i32;
    fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
}
#[repr(C)]
struct OrtApiBase {
    get_api: Option<unsafe extern "C" fn(u32) -> ApiPtr>,
    get_version: Option<unsafe extern "C" fn() -> *const c_char>,
}

/// Borrows one function pointer out of the `OrtApi` array.
unsafe fn function<T: Copy>(api: ApiPtr, index: usize) -> T {
    let slot = unsafe { *api.add(index) };
    debug_assert!(!slot.is_null(), "OrtApi field {index} is null");
    unsafe { std::mem::transmute_copy(&slot) }
}

pub(crate) struct Runtime {
    api: ApiPtr,
    env: *mut c_void,
    allocator: *mut c_void,
    memory_info: *mut c_void,
    handle: *mut c_void,
    library: PathBuf,
    version: String,
    api_version: u32,
    #[cfg(windows)]
    _runtime_modules: Vec<*mut c_void>,
    #[cfg(windows)]
    cuda_runtime_error: Option<String>,
}

/// An execution provider. Only the ones this ONNX Runtime build exports can be
/// requested; `Cpu` is always available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Provider {
    TensorRt,
    Cuda,
    DirectMl,
    /// Apple Core ML execution provider. Only offered when the loaded runtime
    /// exports `OrtSessionOptionsAppendExecutionProvider_CoreML`; the plan
    /// says nothing about which device (ANE/GPU/CPU) a session finally uses.
    CoreMl,
    Cpu,
}

impl Provider {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Provider::TensorRt => "TensorRT",
            Provider::Cuda => "CUDA",
            Provider::DirectMl => "DirectML",
            Provider::CoreMl => "CoreML",
            Provider::Cpu => "CPU",
        }
    }
}

/// What the runtime can export and what the selection logic asked for. This is
/// a *plan*, not evidence of execution: [`Session`] reports the provider a
/// session really opened on, and [`Session::provider_trace`] why the first
/// choices were skipped. GPU nodes can still fall back to CPU inside one
/// session, so "opened on" is not the same as "every node ran there".
#[derive(Clone, Debug)]
pub(crate) struct ProviderPlan {
    pub(crate) exported: Vec<Provider>,
    pub(crate) selected: Vec<Provider>,
    pub(crate) trt_reason: &'static str,
}

impl ProviderPlan {
    pub(crate) fn labels(&self) -> Vec<&'static str> {
        provider_labels(&self.selected)
    }
}

fn provider_symbol(provider: Provider) -> Option<&'static str> {
    match provider {
        Provider::TensorRt => Some("OrtSessionOptionsAppendExecutionProvider_Tensorrt"),
        Provider::Cuda => Some("OrtSessionOptionsAppendExecutionProvider_CUDA"),
        Provider::DirectMl => Some("OrtSessionOptionsAppendExecutionProvider_DML"),
        Provider::CoreMl => Some("OrtSessionOptionsAppendExecutionProvider_CoreML"),
        Provider::Cpu => None,
    }
}

/// Symbol spellings to try when appending a provider. ORT <= 1.22 exports
/// `_TensorRT`, ORT >= 1.23 exports `_Tensorrt`; `GetProcAddress` is case
/// sensitive, so both are probed and the PE export table is scanned
/// case-insensitively on Windows. On macOS `dlsym` is case-sensitive too, so
/// both CoreML spellings are tried (the shipped binary exports `_CoreML`).
fn append_symbol_candidates(provider: Provider) -> &'static [&'static str] {
    match provider {
        Provider::TensorRt => &[
            "OrtSessionOptionsAppendExecutionProvider_Tensorrt",
            "OrtSessionOptionsAppendExecutionProvider_TensorRT",
        ],
        Provider::Cuda => &["OrtSessionOptionsAppendExecutionProvider_CUDA"],
        Provider::DirectMl => &["OrtSessionOptionsAppendExecutionProvider_DML"],
        Provider::CoreMl => &[
            "OrtSessionOptionsAppendExecutionProvider_CoreML",
            "OrtSessionOptionsAppendExecutionProvider_CoreMl",
        ],
        Provider::Cpu => &[],
    }
}

fn export_name_matches(export: &str, wanted: &str) -> bool {
    export.eq_ignore_ascii_case(wanted)
}

/// Default provider order: CUDA first for interactive latency, then DirectML,
/// then CPU. TensorRT is deliberately absent: a cold engine build measured
/// 40-190 s per model (2026-11-23, RTX 4070 Ti), so it is prepended only when
/// a cache is present or the operator asks for it. Kept pure for tests.
fn providers_from(has_cuda: bool, has_direct_ml: bool) -> Vec<Provider> {
    let mut out = Vec::new();
    if has_cuda {
        out.push(Provider::Cuda);
    }
    if has_direct_ml {
        out.push(Provider::DirectMl);
    }
    out.push(Provider::Cpu);
    out
}

/// Default provider order for one OS. macOS has no CUDA/DirectML runtime, so the
/// CPU baseline is the default. Core ML is *not* auto-selected even when the
/// loaded library exports the symbol: the EP may choose FP16 compute, and this
/// tree has no CPU-vs-Core ML parity evidence for continuous mask alpha yet
/// (README: raw FP16 probabilities must not drive feathered masks). It stays
/// reachable through `AURORA_ORT_PROVIDERS=coreml,cpu` or
/// `AURORA_AI_PROVIDER=coreml` once that evidence exists.
/// Every other platform keeps the existing CUDA -> DirectML -> CPU order; a
/// Core ML symbol there is listed as exported but never auto-selected.
/// The `os` argument is explicit so this can be tested on any host.
fn providers_for_os(
    os: &str,
    has_cuda: bool,
    has_direct_ml: bool,
    has_core_ml: bool,
) -> Vec<Provider> {
    if os == "macos" {
        let _ = has_core_ml;
        vec![Provider::Cpu]
    } else {
        providers_from(has_cuda, has_direct_ml)
    }
}

fn with_tensor_rt_first(mut providers: Vec<Provider>) -> Vec<Provider> {
    if !providers.contains(&Provider::TensorRt) {
        providers.insert(0, Provider::TensorRt);
    }
    providers
}

fn provider_from_name(raw: &str) -> Option<Provider> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "tensorrt" | "trt" => Some(Provider::TensorRt),
        "cuda" => Some(Provider::Cuda),
        "dml" | "directml" => Some(Provider::DirectMl),
        "coreml" | "core_ml" => Some(Provider::CoreMl),
        "cpu" => Some(Provider::Cpu),
        _ => None,
    }
}

/// `AURORA_ORT_PROVIDERS=tensorrt,cuda,cpu` wins over `AURORA_AI_PROVIDER`.
/// CPU is always kept as the final fallback; unknown names are logged and
/// skipped rather than silently disabling AI.
fn providers_from_override(raw: &str) -> Vec<Provider> {
    let mut out = Vec::new();
    for part in raw.split(',') {
        if part.trim().is_empty() {
            continue;
        }
        match provider_from_name(part) {
            Some(Provider::Cpu) => {}
            Some(provider) if !out.contains(&provider) => out.push(provider),
            Some(_) => {}
            None => log::warn!("{PROVIDERS_ENV}: ignoring unknown provider {:?}", part.trim()),
        }
    }
    out.push(Provider::Cpu);
    out
}

fn provider_labels(providers: &[Provider]) -> Vec<&'static str> {
    providers.iter().map(|provider| provider.label()).collect()
}

fn log_provider_plan(plan: &ProviderPlan) {
    log::info!(
        "ONNX Runtime providers: exported {:?}, selected {:?}, trt {:?}",
        provider_labels(&plan.exported),
        provider_labels(&plan.selected),
        plan.trt_reason
    );
    for provider in [
        Provider::TensorRt,
        Provider::Cuda,
        Provider::DirectMl,
        Provider::CoreMl,
        Provider::Cpu,
    ] {
        let state = match provider {
            Provider::Cpu => "enabled (fallback)",
            Provider::TensorRt if !plan.exported.contains(&provider) => {
                "not exported by this ONNX Runtime"
            }
            Provider::TensorRt if plan.selected.contains(&Provider::TensorRt) => {
                match plan.trt_reason {
                    "cache" => "enabled first (engine cache present)",
                    _ => "enabled first (explicit request)",
                }
            }
            Provider::TensorRt => {
                "skipped (no engine cache; set AURORA_TRT_CACHE_DIR or \
                 AURORA_ORT_PROVIDERS=tensorrt,...)"
            }
            _ if !plan.exported.contains(&provider) => "not exported by this ONNX Runtime",
            _ if plan.selected.contains(&provider) => "enabled",
            _ => "available but not selected",
        };
        log::info!("provider {}: {}", provider.label(), state);
    }
}

/// Directory the operator pre-populated with TensorRT engine files.
fn trt_cache_dir() -> Option<PathBuf> {
    let raw = std::env::var_os("AURORA_TRT_CACHE_DIR")?;
    if raw.is_empty() {
        return None;
    }
    let dir = PathBuf::from(raw);
    dir.is_dir().then_some(dir)
}

/// ORT's TensorRT EP writes `<prefix>.engine`, `.profile` and `.timing` files.
fn dir_has_engine_cache(dir: &Path, depth: usize) -> bool {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            if depth > 0 && dir_has_engine_cache(&path, depth - 1) {
                return true;
            }
            continue;
        }
        let name = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_ascii_lowercase();
        if name.ends_with(".engine") || name.ends_with(".profile") || name.ends_with(".timing") {
            return true;
        }
    }
    false
}

/// True when `AURORA_TRT_CACHE_DIR` exists, is non-empty and already holds an
/// engine/profile/timing artifact. Cached once: `available_providers` runs per
/// session and should not rescan the disk.
pub(crate) fn trt_cache_present() -> bool {
    static PRESENT: OnceLock<bool> = OnceLock::new();
    *PRESENT.get_or_init(|| trt_cache_dir().is_some_and(|dir| dir_has_engine_cache(&dir, 2)))
}

pub(crate) fn trt_first_allowed() -> bool {
    if forced_provider() == Some(Provider::TensorRt) {
        return true;
    }
    if std::env::var(PROVIDERS_ENV)
        .ok()
        .is_some_and(|raw| raw.split(',').any(|part| provider_from_name(part) == Some(Provider::TensorRt)))
    {
        return true;
    }
    trt_cache_present()
}

/// `AURORA_AI_PROVIDER` pins the provider, for A/B measurement and for machines
/// where the automatic pick misbehaves. Names are case-insensitive; an
/// unrecognised name means automatic.
fn forced_provider() -> Option<Provider> {
    let raw = std::env::var("AURORA_AI_PROVIDER").ok()?;
    provider_from_name(&raw)
}

/// Narrows a probed list to the forced provider. An unavailable forced provider
/// falls back to CPU, never to a different GPU provider.
fn apply_override(probed: Vec<Provider>, forced: Option<Provider>) -> Vec<Provider> {
    let Some(forced) = forced else {
        return probed;
    };
    if !probed.contains(&forced) {
        return vec![Provider::Cpu];
    }
    if forced == Provider::Cpu {
        return vec![Provider::Cpu];
    }
    vec![forced, Provider::Cpu]
}

fn graph_optimization_level() -> c_int {
    match std::env::var("AURORA_ORT_OPT_LEVEL").unwrap_or_default().to_ascii_lowercase().as_str() {
        "disable" | "none" | "0" => 0,
        "basic" | "1" => 1,
        "extended" | "2" => 2,
        _ => OPT_ENABLE_ALL,
    }
}

/// Intra-op threads the session options use. Part of the session identity: a
/// cached optimized graph and a running session are only equivalent when the
/// thread count matches.
fn intra_op_threads() -> c_int {
    std::thread::available_parallelism().map_or(4, |n| n.get()).min(8) as c_int
}

/// TensorRT cache directory as an opaque 64-bit id. The key only needs to tell
/// two directories apart; storing the path would leak a user directory into
/// [`SessionKey`]/logs.
fn trt_dir_identity(raw: Option<&str>) -> String {
    match raw.filter(|dir| !dir.is_empty()) {
        Some(dir) => format!("{:016x}", fnv64(dir.as_bytes())),
        None => "-".to_string(),
    }
}

/// Everything that changes how ORT executes a model, in a stable order. Kept
/// as one string so it can be hashed into [`SessionKey`] and printed in logs.
fn provider_options_fingerprint() -> String {
    let tf32 = std::env::var("NVIDIA_TF32_OVERRIDE").unwrap_or_else(|_| "-".to_string());
    let trt_dir = trt_dir_identity(std::env::var("AURORA_TRT_CACHE_DIR").ok().as_deref());
    format!(
        "opt={};threads={};tf32={tf32};trt_dir={trt_dir};trt_present={}",
        graph_optimization_level(),
        intra_op_threads(),
        trt_cache_present()
    )
}

/// Stable 64-bit FNV-1a. Used for session/cache names so they stay identical
/// across processes and Rust upgrades (unlike `DefaultHasher`, whose output is
/// not a documented artifact).
fn fnv64(bytes: &[u8]) -> u64 {
    fnv64_with(bytes, FNV_OFFSET)
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv64_with(bytes: &[u8], seed: u64) -> u64 {
    let mut hash = seed;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

/// Content fingerprint of a model, plus its length. `crc32` over the whole file
/// (fast) folded with the length, the head and the tail. Models are tens of
/// megabytes, so a full cryptographic digest would cost more cold-start time
/// than it buys; this is an identity check, not a security boundary.
fn model_fingerprint(path: &Path) -> Result<(u64, u64)> {
    type FingerprintCache = Mutex<HashMap<(PathBuf, u64, u128), u64>>;
    static CACHE: OnceLock<FingerprintCache> = OnceLock::new();
    let metadata = std::fs::metadata(path)
        .with_context(|| format!("reading model metadata {}", path.display()))?;
    let len = metadata.len();
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let cache_key = (path.to_path_buf(), len, modified);
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(hash) = cache
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(&cache_key)
    {
        return Ok((*hash, len));
    }

    let mut file = std::fs::File::open(path)
        .with_context(|| format!("opening model {}", path.display()))?;
    let mut crc = crc32fast::Hasher::new();
    let mut head = crc32fast::Hasher::new();
    let mut tail = crc32fast::Hasher::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut offset = 0u64;
    let mut ring: Vec<u8> = Vec::new();
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        let chunk = &buffer[..read];
        crc.update(chunk);
        if offset < 64 << 10 {
            let keep = ((64 << 10) - offset as usize).min(read);
            head.update(&chunk[..keep]);
        }
        // Keep the last 64 KiB seen so far.
        ring.extend_from_slice(chunk);
        if ring.len() > 64 << 10 {
            let drop = ring.len() - (64 << 10);
            ring.drain(..drop);
        }
        offset += read as u64;
    }
    tail.update(&ring);
    let mut hash = FNV_OFFSET;
    for value in [
        u64::from(crc.finalize()),
        len,
        u64::from(head.finalize()),
        u64::from(tail.finalize()),
    ] {
        hash = fnv64_with(&value.to_le_bytes(), hash);
    }
    cache
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .insert(cache_key, hash);
    Ok((hash, len))
}

/// Identity of one reusable ONNX session. A session may be reused only when
/// every component matches: the model content, the ORT build (version, API and
/// the actual dylib/DLL identity), the platform/arch, the model precision, the
/// execution provider configuration and the options that change execution
/// (thread count, graph optimization, TensorRT cache, TF32).
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct SessionKey {
    model_hash: u64,
    model_len: u64,
    ort_version: String,
    ort_api: u32,
    /// e.g. `macos-aarch64`; CPU and Core ML optimized graphs are not
    /// interchangeable across OS/arch, even at the same ORT version.
    platform: String,
    /// Runtime identity as `<file name>:<bytes>`. Deliberately not the absolute
    /// path, so the key/logs never leak a user directory.
    runtime: String,
    /// Model precision implied by its file name (`fp32` / `fp16`). Name-derived,
    /// not a content inspection.
    precision: String,
    /// Provider configuration string, e.g. `CoreML,CPU`. A CPU-only and a
    /// hypothetical Core ML session therefore cannot share an entry.
    providers: String,
    options: String,
}

impl SessionKey {
    /// Same model and runtime, restricted to one provider, for per-EP
    /// optimized-model files and per-EP open attempts.
    pub(crate) fn for_provider(&self, provider: Provider) -> SessionKey {
        let mut key = self.clone();
        key.providers = provider.label().to_string();
        key
    }

    fn canonical(&self) -> String {
        format!(
            "v2|model={:016x}|len={}|ort={}|api={}|platform={}|runtime={}|precision={}|eps={}|opts={}",
            self.model_hash,
            self.model_len,
            self.ort_version,
            self.ort_api,
            self.platform,
            self.runtime,
            self.precision,
            self.providers,
            self.options
        )
    }

    pub(crate) fn digest(&self) -> u64 {
        fnv64(self.canonical().as_bytes())
    }

    /// Identity used by the in-process session registry and its creation locks.
    pub(crate) fn registry_id(&self) -> String {
        self.canonical()
    }

    pub(crate) fn short(&self) -> String {
        format!("{:016x}", self.digest())
    }
}

/// Where ORT writes its optimized graph. `AURORA_ORT_CACHE_DIR` wins; without
/// it the per-user cache is used, because re-running graph optimization on
/// every launch is the largest single part of the 0.8-1.5 s cold start.
fn optimized_cache_dir() -> Option<PathBuf> {
    if std::env::var_os("AURORA_DISABLE_ORT_CACHE").is_some() {
        return None;
    }
    if let Some(dir) = std::env::var_os("AURORA_ORT_CACHE_DIR") {
        return Some(PathBuf::from(dir));
    }
    #[cfg(windows)]
    {
        std::env::var_os("LOCALAPPDATA").map(|dir| {
            PathBuf::from(dir)
                .join("Aurora Studio")
                .join("Cache")
                .join("onnxruntime")
        })
    }
    #[cfg(target_os = "macos")]
    {
        // Apple's per-user cache convention; the path is derived from $HOME, so
        // it never hardcodes a user name.
        std::env::var_os("HOME").map(|dir| {
            PathBuf::from(dir)
                .join("Library")
                .join("Caches")
                .join("Aurora Studio")
                .join("onnxruntime")
        })
    }
    #[cfg(all(not(windows), not(target_os = "macos")))]
    {
        std::env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|dir| PathBuf::from(dir).join(".cache")))
            .map(|dir| dir.join("aurora-studio").join("onnxruntime"))
    }
}

fn optimized_model_cache_path(path: &Path, key: &SessionKey) -> Option<PathBuf> {
    Some(optimized_model_cache_path_in(&optimized_cache_dir()?, path, key))
}

fn optimized_model_cache_path_in(dir: &Path, path: &Path, key: &SessionKey) -> PathBuf {
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    dir.join(format!("{stem}-{}.onnx", key.short()))
}

/// Bytes one session is charged against the registry budget.
///
/// The ONNX Runtime C API slice used here has no call that reports a session's
/// arena size, so this is a documented *lower bound*, not a measurement:
/// `model file bytes + every input/output tensor at its declared dtype`. ORT
/// keeps the weights in memory (a GPU EP usually keeps its own copy as well);
/// the optimized graph, activation arenas and EP scratch buffers are not
/// counted, so the real footprint is larger. The default budget is small for
/// that reason.
const SESSION_MODEL_BYTES_FACTOR: u64 = 1;
/// Idle GPU sessions are expensive. Capture One's models are 30-180 MB, so
/// 512 MiB keeps a working set of a few sessions and evicts the rest.
const SESSION_BUDGET_ENV: &str = "AURORA_ORT_SESSION_BUDGET_MB";
const SESSION_BUDGET_DEFAULT_MIB: u64 = 512;
/// Secondary ceiling, retained from the old map so many tiny sessions cannot
/// grow without bound even under the byte budget.
const SESSION_REGISTRY_MAX_ENTRIES: usize = 64;

fn session_budget_bytes() -> u64 {
    lru::budget_from_env(SESSION_BUDGET_ENV, SESSION_BUDGET_DEFAULT_MIB)
}

fn tensor_bytes(dims: &[i64], element_type: c_int) -> u64 {
    let element_size = match element_type {
        FLOAT => 4,
        FLOAT16 => 2,
        _ => return 0,
    };
    let Some(elements) = dims.iter().try_fold(1u64, |total, dim| {
        u64::try_from(*dim)
            .ok()
            .filter(|dim| *dim > 0)
            .map(|dim| total.saturating_mul(dim))
    }) else {
        // A dynamic dimension cannot be sized from metadata; counting nothing
        // keeps the result a lower bound.
        return 0;
    };
    elements.saturating_mul(element_size)
}

fn session_bytes_estimate(model_len: u64, input: &TensorMeta, outputs: &[TensorMeta]) -> u64 {
    let tensors = std::iter::once(input)
        .chain(outputs)
        .fold(0u64, |total, meta| {
            total.saturating_add(tensor_bytes(&meta.dims, meta.element_type))
        });
    model_len
        .saturating_mul(SESSION_MODEL_BYTES_FACTOR)
        .saturating_add(tensors)
}

impl Session {
    /// What this session keeps resident, as charged to the registry budget.
    fn cached_bytes(&self) -> u64 {
        session_bytes_estimate(self.inner.key.model_len, &self.input, &self.outputs)
    }

    /// `true` when someone outside the registry still holds this session.
    /// Registry entries and callers share one `Arc<SessionInner>`, so a count
    /// of one means the registry is the only owner.
    fn is_shared(&self) -> bool {
        Arc::strong_count(&self.inner) > 1
    }
}

/// Byte-budgeted LRU instead of an unbounded map. A cleared map never released
/// a GPU session either: the finished [`Flight`] keeps a clone of it, so
/// [`drop_finished_flights`] runs before every eviction pass.
type SessionRegistry = Mutex<ByteLru<Session>>;

fn session_registry() -> &'static SessionRegistry {
    static REGISTRY: OnceLock<SessionRegistry> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        Mutex::new(ByteLru::new(
            session_budget_bytes(),
            SESSION_REGISTRY_MAX_ENTRIES,
        ))
    })
}

/// In-flight session creations, keyed by [`SessionKey::registry_id`].
fn session_flights() -> &'static OnceLock<Mutex<HashMap<String, Arc<Flight<Session>>>>> {
    static FLIGHTS: OnceLock<Mutex<HashMap<String, Arc<Flight<Session>>>>> = OnceLock::new();
    &FLIGHTS
}

/// Drops finished single-flight records so their cached session clone stops
/// pinning the ORT session. Waiters keep their own `Arc` and still read the
/// result; a later caller that misses the registry starts a fresh flight, and
/// that flight's work rechecks the registry before creating anything. Only a
/// flight whose `result` is already written is removed, so a creation in
/// progress is never dropped or duplicated.
fn drop_finished_flights() {
    let Some(store) = session_flights().get() else {
        return;
    };
    let mut flights = store.lock().unwrap_or_else(|error| error.into_inner());
    let finished: Vec<String> = flights
        .iter()
        .filter(|(_, flight)| {
            flight
                .result
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_some()
        })
        .map(|(key, _)| key.clone())
        .collect();
    let removed: Vec<Arc<Flight<Session>>> = finished
        .iter()
        .filter_map(|key| flights.remove(key))
        .collect();
    drop(flights);
    // Session clones inside `removed` are released here, outside the lock.
    drop(removed);
}

/// True while a creation for `id` is running. Finished flights are removed by
/// [`drop_finished_flights`], so a cached key is not in flight.
fn session_in_flight(id: &str) -> bool {
    let Some(store) = session_flights().get() else {
        return false;
    };
    store
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .contains_key(id)
}

/// Runs one eviction pass. The caller drops the returned values *after*
/// releasing the registry lock: `SessionInner::drop` calls
/// `OrtReleaseSession`, which must not run under the cache mutex.
fn reclaim_sessions(registry: &mut ByteLru<Session>) -> lru::Eviction<Session> {
    drop_finished_flights();
    registry.evict(|key, session| session.is_shared() || session_in_flight(key))
}

fn log_evictions(eviction: lru::Eviction<Session>) {
    for (_, session) in &eviction.removed {
        log::debug!(
            "ONNX session registry: released {} ({} bytes estimated)",
            session.session_key().short(),
            session.cached_bytes()
        );
    }
    if eviction.pending > 0 {
        log::debug!(
            "ONNX session registry: {} entry(s) over budget still in use; eviction deferred",
            eviction.pending
        );
    }
    // Dropping here releases the ORT sessions; the registry lock is gone.
    drop(eviction);
}

fn registry_lookup(id: &str) -> Option<Session> {
    let (session, eviction) = {
        let mut registry = session_registry()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let session = registry.get(id).cloned();
        let eviction = reclaim_sessions(&mut registry);
        (session, eviction)
    };
    log_evictions(eviction);
    session
}

fn registry_insert(id: String, session: Session) {
    let bytes = session.cached_bytes();
    let (previous, eviction) = {
        let mut registry = session_registry()
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        let previous = registry.insert(id, session, bytes);
        let eviction = reclaim_sessions(&mut registry);
        (previous, eviction)
    };
    // A replaced session is released outside the lock as well.
    drop(previous);
    log_evictions(eviction);
}

/// One in-flight creation. `done` is signalled after `result` is written.
struct Flight<V> {
    result: Mutex<Option<Result<V, String>>>,
    done: std::sync::Condvar,
}

/// Runs `work` once per key. Concurrent callers with the same key wait for the
/// leader and receive a clone of its result. Failures are shared with the
/// waiters but not cached: the next call after a failure starts a new attempt.
/// A panicking leader becomes an error instead of hanging waiters.
fn single_flight<V: Clone>(
    store: &'static OnceLock<Mutex<HashMap<String, Arc<Flight<V>>>>>,
    key: String,
    work: impl FnOnce() -> Result<V>,
) -> Result<V> {
    let flights = store.get_or_init(|| Mutex::new(HashMap::new()));
    // Only one thread may install a flight per key. If a caller arrives after
    // a previous flight was removed, it joins that (already finished) flight
    // instead of starting a second creation for the same generation.
    let (flight, leader) = {
        let mut flights = flights.lock().unwrap_or_else(|error| error.into_inner());
        if let Some(flight) = flights.get(&key) {
            (flight.clone(), false)
        } else {
            let flight = Arc::new(Flight {
                result: Mutex::new(None),
                done: std::sync::Condvar::new(),
            });
            flights.insert(key.clone(), flight.clone());
            (flight, true)
        }
    };
    if leader {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(work))
            .unwrap_or_else(|_| Err(anyhow!("session creation panicked")))
            .map_err(|error| format!("{error:#}"));
        *flight
            .result
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some(outcome);
        flight.done.notify_all();
        // Keep the finished flight under the same lock that installs new
        // flights. Removing it here would let a late caller start a second
        // creation and race the waiters that are still reading this result.
    }
    let mut result = flight
        .result
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    while result.is_none() {
        result = flight
            .done
            .wait(result)
            .unwrap_or_else(|error| error.into_inner());
    }
    match result
        .clone()
        .expect("single_flight result is written before any waiter returns")
    {
        Ok(value) => Ok(value),
        Err(message) => Err(anyhow!("{message}")),
    }
}

/// `os-arch`, e.g. `macos-aarch64`. Part of [`SessionKey`] so optimized graphs
/// never cross a platform boundary.
fn platform_identity() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

/// Runtime identity for [`SessionKey`]: file name plus size, never the
/// absolute path (the key is logged and used as a cache file name).
fn runtime_identity(library: &Path) -> String {
    let name = library
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("onnxruntime");
    match std::fs::metadata(library) {
        Ok(metadata) => format!("{name}:{}", metadata.len()),
        Err(_) => name.to_string(),
    }
}

/// Model precision implied by the file name. This is an identity hint, not a
/// tensor inspection: FP32 and FP16 variants must not share an optimized graph.
fn precision_for_path(path: &Path) -> &'static str {
    let stem = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    if stem.contains("fp16") || stem.contains("float16") {
        "fp16"
    } else {
        "fp32"
    }
}

// ORT objects are internally synchronised; `Run` may be called from any thread.
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl Runtime {
    pub(crate) fn get() -> Option<&'static Runtime> {
        static RUNTIME: OnceLock<Option<Runtime>> = OnceLock::new();
        RUNTIME
            .get_or_init(|| match unsafe { Self::load() } {
                Ok(runtime) => Some(runtime),
                Err(err) => {
                    log::warn!("ONNX Runtime unavailable: {err:#}");
                    None
                }
            })
            .as_ref()
    }

    unsafe fn load() -> Result<Self> {
        let library = resolve_library()?;
        let dll_dirs = configure_gpu_dll_directories(&library);
        if !dll_dirs.is_empty() {
            log::info!("GPU DLL dirs: {dll_dirs:?}");
        }
        let handle = unsafe { load_library(&library)? };
        log::info!("ONNX Runtime from {}", library.display());
        let symbol = CString::new("OrtGetApiBase")?;
        #[cfg(not(windows))]
        let get_base: unsafe extern "C" fn() -> *const OrtApiBase =
            unsafe { std::mem::transmute(dlsym(handle, symbol.as_ptr())) };
        #[cfg(windows)]
        let get_base: unsafe extern "system" fn() -> *const OrtApiBase =
            unsafe { std::mem::transmute(GetProcAddress(handle, symbol.as_ptr() as *const u8)) };
        if get_base as usize == 0 {
            bail!("{} does not export OrtGetApiBase", library.display());
        }
        let base = unsafe { &*get_base() };
        let version = unsafe { CStr::from_ptr((base.get_version.unwrap())()) };
        log::info!("ONNX Runtime version {}", version.to_string_lossy());
        // The table only grows between versions: an older library returns null for
        // a version it does not know, and every entry used here predates all the
        // versions we ask for. Capture One ships 1.22, so asking only for 23 would
        // reject a perfectly usable runtime.
        // Ask for the library's own major first: asking for a newer table makes
        // ORT print "requested API version [23] is not available" to stderr.
        let preferred = version
            .to_string_lossy()
            .split('.')
            .nth(1)
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(ORT_API_VERSION)
            .min(ORT_API_VERSION);
        let mut api = ptr::null();
        let mut used = 0;
        for requested in (MIN_ORT_API_VERSION..=preferred).rev() {
            api = unsafe { (base.get_api.unwrap())(requested) };
            if !api.is_null() {
                used = requested;
                break;
            }
        }
        if api.is_null() {
            bail!("ONNX Runtime is older than API version {MIN_ORT_API_VERSION}");
        }
        log::info!("ONNX Runtime API {used}");
        let create_env: unsafe extern "C" fn(c_int, *const c_char, *mut *mut c_void) -> Status =
            unsafe { function(api, CREATE_ENV) };
        let mut env = ptr::null_mut();
        let logid = CString::new("portrait")?;
        unsafe { check(create_env(LOG_ERROR, logid.as_ptr(), &mut env), api)? };
        let get_allocator: unsafe extern "C" fn(*mut *mut c_void) -> Status =
            unsafe { function(api, GET_ALLOCATOR_WITH_DEFAULT_OPTIONS) };
        let mut allocator = ptr::null_mut();
        unsafe { check(get_allocator(&mut allocator), api)? };
        let allocator_info: unsafe extern "C" fn(*const c_void, *mut *mut c_void) -> Status =
            unsafe { function(api, ALLOCATOR_GET_INFO) };
        let mut memory_info = ptr::null_mut();
        unsafe { check(allocator_info(allocator, &mut memory_info), api)? };
        #[cfg(windows)]
        let (runtime_modules, cuda_runtime_error) = unsafe { preload_cuda_runtime(&library) };
        Ok(Self {
            api,
            env,
            allocator,
            memory_info,
            handle,
            library,
            version: version.to_string_lossy().into_owned(),
            api_version: used,
            #[cfg(windows)]
            _runtime_modules: runtime_modules,
            #[cfg(windows)]
            cuda_runtime_error,
        })
    }

    pub(crate) fn describe(&self) -> (&Path, &str, u32) {
        (&self.library, &self.version, self.api_version)
    }

    /// Identity of a session on `path` with this runtime and provider list.
    /// The list is part of the key because a different order can open a
    /// different provider; [`SessionKey::for_provider`] narrows it to one EP
    /// for per-provider optimized-model files.
    pub(crate) fn session_key(&self, path: &Path, providers: &[Provider]) -> Result<SessionKey> {
        let (model_hash, model_len) = model_fingerprint(path)?;
        Ok(SessionKey {
            model_hash,
            model_len,
            ort_version: self.version.clone(),
            ort_api: self.api_version,
            platform: platform_identity(),
            runtime: runtime_identity(&self.library),
            precision: precision_for_path(path).to_string(),
            providers: provider_labels(providers).join(","),
            options: provider_options_fingerprint(),
        })
    }

    fn symbol(&self, name: &str) -> *mut c_void {
        let Ok(name) = CString::new(name) else {
            return ptr::null_mut();
        };
        #[cfg(not(windows))]
        {
            unsafe { dlsym(self.handle, name.as_ptr()) }
        }
        #[cfg(windows)]
        {
            unsafe { GetProcAddress(self.handle, name.as_ptr() as *const u8) }
        }
    }

    /// Case-insensitive lookup in the loaded module's PE export table.
    #[cfg(windows)]
    unsafe fn symbol_ignore_case(&self, wanted: &str) -> *mut c_void {
        let base = self.handle as *const u8;
        if base.is_null() {
            return ptr::null_mut();
        }
        let e_lfanew = unsafe { *(base.add(0x3C) as *const u32) } as usize;
        let nt = unsafe { base.add(e_lfanew) };
        if unsafe { *(nt as *const u32) } != 0x0000_4550 {
            return ptr::null_mut();
        }
        let optional = unsafe { nt.add(24) };
        let magic = unsafe { *(optional as *const u16) };
        let (export_rva, export_size) = match magic {
            0x20b => (
                unsafe { *(optional.add(112) as *const u32) },
                unsafe { *(optional.add(116) as *const u32) },
            ),
            0x10b => (
                unsafe { *(optional.add(96) as *const u32) },
                unsafe { *(optional.add(100) as *const u32) },
            ),
            _ => return ptr::null_mut(),
        };
        if export_rva == 0 || export_size == 0 {
            return ptr::null_mut();
        }
        let export = unsafe { base.add(export_rva as usize) };
        let name_count = unsafe { *(export.add(24) as *const u32) } as usize;
        let names = unsafe { *(export.add(32) as *const u32) } as usize;
        let ordinals = unsafe { *(export.add(36) as *const u32) } as usize;
        let functions = unsafe { *(export.add(28) as *const u32) } as usize;
        for index in 0..name_count {
            let name_rva = unsafe { *(base.add(names + index * 4) as *const u32) } as usize;
            if name_rva == 0 {
                continue;
            }
            let entry = unsafe { CStr::from_ptr(base.add(name_rva) as *const c_char) };
            let Ok(entry) = entry.to_str() else {
                continue;
            };
            if !export_name_matches(entry, wanted) {
                continue;
            }
            let ordinal = unsafe { *(base.add(ordinals + index * 2) as *const u16) } as usize;
            let function_rva =
                unsafe { *(base.add(functions + ordinal * 4) as *const u32) } as usize;
            if function_rva == 0 {
                return ptr::null_mut();
            }
            return unsafe { base.add(function_rva) } as *mut c_void;
        }
        ptr::null_mut()
    }

    /// First append symbol this library really exports for `provider`.
    fn provider_append_symbol(&self, provider: Provider) -> *mut c_void {
        for candidate in append_symbol_candidates(provider) {
            let pointer = self.symbol(candidate);
            if !pointer.is_null() {
                return pointer;
            }
            #[cfg(windows)]
            {
                let pointer = unsafe { self.symbol_ignore_case(candidate) };
                if !pointer.is_null() {
                    return pointer;
                }
            }
        }
        ptr::null_mut()
    }

    /// Providers this library can really append. The C API exports one
    /// `OrtSessionOptionsAppendExecutionProvider_<name>` symbol per provider it
    /// was built with, so probing the symbols reflects the build, not a wish.
    ///
    /// The plan is computed once per process: it scans the export table, the
    /// TensorRT cache directory and the provider environment variables, and
    /// every session open used to repeat that scan (and its log lines) during
    /// the cold start it was supposed to reduce.
    pub(crate) fn provider_plan(&self) -> &'static ProviderPlan {
        static PLAN: OnceLock<ProviderPlan> = OnceLock::new();
        PLAN.get_or_init(|| {
            let plan = self.compute_provider_plan();
            log_provider_plan(&plan);
            plan
        })
    }

    pub(crate) fn available_providers(&self) -> Vec<Provider> {
        self.provider_plan().selected.clone()
    }

    fn compute_provider_plan(&self) -> ProviderPlan {
        let has_trt = !self.provider_append_symbol(Provider::TensorRt).is_null();
        let has_cuda = !self.provider_append_symbol(Provider::Cuda).is_null();
        let has_dml = !self.provider_append_symbol(Provider::DirectMl).is_null();
        // Core ML is only real when this library exports the append function.
        // The probe, not a platform guess, decides: macOS stays CPU-only when
        // the symbol is absent, and a non-macOS library that exports it is
        // merely listed (never auto-selected).
        let has_core_ml = !self.provider_append_symbol(Provider::CoreMl).is_null();
        if std::env::consts::OS == "macos" && !has_core_ml {
            log::info!(
                "CoreML EP unavailable: this ONNX Runtime does not export \
                 OrtSessionOptionsAppendExecutionProvider_CoreML; using CPU only"
            );
        } else if std::env::consts::OS == "macos" {
            log::info!(
                "CoreML EP exported but not auto-selected: no CPU/Core ML parity \
                 evidence yet; set AURORA_ORT_PROVIDERS=coreml,cpu to opt in"
            );
        }
        let base = providers_for_os(std::env::consts::OS, has_cuda, has_dml, has_core_ml);
        let mut exported = base.clone();
        if has_trt {
            exported.insert(0, Provider::TensorRt);
        }
        // Keep the exported list honest even when the platform default ignores
        // a provider (e.g. an explicit override on a non-macOS host).
        if has_core_ml && !exported.contains(&Provider::CoreMl) {
            let cpu_index = exported.len().saturating_sub(1);
            exported.insert(cpu_index, Provider::CoreMl);
        }

        let override_raw = std::env::var(PROVIDERS_ENV)
            .ok()
            .filter(|value| !value.trim().is_empty());
        let explicit_trt = override_raw.as_deref().is_some_and(|raw| {
            raw.split(',')
                .any(|part| provider_from_name(part) == Some(Provider::TensorRt))
        }) || forced_provider() == Some(Provider::TensorRt);
        let cache_present = trt_cache_present();

        let (chosen, trt_reason) = match override_raw {
            Some(raw) => {
                let requested = providers_from_override(&raw);
                let mut selected = Vec::with_capacity(requested.len());
                for provider in requested {
                    if provider == Provider::Cpu || exported.contains(&provider) {
                        selected.push(provider);
                    } else {
                        log::warn!(
                            "{PROVIDERS_ENV} asked for {}, which this ONNX Runtime does not export",
                            provider.label()
                        );
                    }
                }
                let reason = if selected.contains(&Provider::TensorRt) {
                    "explicit"
                } else {
                    "not-requested"
                };
                (selected, reason)
            }
            None => {
                let mut selected = base.clone();
                let reason = if has_trt && (cache_present || explicit_trt) {
                    selected = with_tensor_rt_first(selected);
                    if cache_present { "cache" } else { "explicit" }
                } else {
                    "no-cache"
                };
                let chosen = apply_override(selected.clone(), forced_provider());
                if let Some(forced) = forced_provider() {
                    if chosen != selected {
                        log::info!("AURORA_AI_PROVIDER pins {}", forced.label());
                    } else if !exported.contains(&forced) {
                        log::warn!(
                            "AURORA_AI_PROVIDER asked for {}, which this ONNX Runtime does not export; \
                             using CPU instead",
                            forced.label()
                        );
                    }
                }
                (chosen, reason)
            }
        };
        ProviderPlan { exported, selected: chosen, trt_reason }
    }


    unsafe fn append_provider(&self, options: *mut c_void, provider: Provider) -> Result<()> {
        if provider == Provider::Cpu {
            return Ok(());
        }
        let pointer = self.provider_append_symbol(provider);
        if pointer.is_null() {
            bail!("{} is not exported by this ONNX Runtime", provider.label());
        }
        let append: unsafe extern "C" fn(*mut c_void, c_int) -> Status =
            unsafe { std::mem::transmute_copy(&pointer) };
        let result = unsafe { check(append(options, 0), self.api) }
            .with_context(|| format!("enabling {}", provider.label()));
        if result.is_ok() {
            match provider {
                Provider::TensorRt => log::info!(
                    "TensorRT EP enabled on device 0 (legacy options: fp16=0, int8=0; \
                     strict FP32 needs NVIDIA_TF32_OVERRIDE=0 before process start)"
                ),
                Provider::Cuda => log::info!(
                    "CUDA EP enabled on device 0 (TF32 follows NVIDIA_TF32_OVERRIDE)"
                ),
                Provider::CoreMl => log::info!(
                    "CoreML EP enabled with default flags; ANE/GPU placement is per node and unverified"
                ),
                _ => {}
            }
        }
        result
    }

    fn provider_blocker(&self, provider: Provider) -> Option<&str> {
        #[cfg(windows)]
        if provider == Provider::Cuda {
            return self.cuda_runtime_error.as_deref();
        }
        let _ = provider;
        None
    }

    /// Opens a session on the first provider that works, falling back to CPU.
    pub(crate) fn open_session(&self, path: &Path) -> Result<Session> {
        self.open_session_on(path, &self.available_providers())
    }

    /// Opens a session using only the requested providers, in order. This is
    /// stricter than `open_session`: callers that require a GPU cannot receive
    /// a CPU session by accident.
    ///
    /// The returned session names the provider it *opened on* and, when an
    /// earlier provider was skipped or failed, why. Callers must not treat the
    /// plan's first entry as the effective provider.
    pub(crate) fn open_session_on(&self, path: &Path, providers: &[Provider]) -> Result<Session> {
        let base_key = self
            .session_key(path, providers)
            .with_context(|| format!("identifying model {}", path.display()))?;
        let id = base_key.registry_id();
        self.session_slot(id, || self.open_session_uncached_on(path, &base_key, providers))
    }

    /// Session creation single-flight. Waiters on the same key block on a
    /// per-key lock and then reuse the leader's session; different models or
    /// providers never queue behind one global lock. The registry holds one
    /// session per key, which also bounds GPU session memory to the number of
    /// distinct (model, runtime, EP, options) tuples actually used.
    fn session_slot(
        &self,
        id: String,
        open: impl FnOnce() -> Result<Session>,
    ) -> Result<Session> {
        if let Some(session) = registry_lookup(&id) {
            return Ok(session);
        }
        single_flight(session_flights(), id.clone(), || {
            if let Some(session) = registry_lookup(&id) {
                return Ok(session);
            }
            let session = open()?;
            registry_insert(id.clone(), session.clone());
            Ok(session)
        })
    }

    fn open_session_uncached_on(
        &self,
        path: &Path,
        base_key: &SessionKey,
        providers: &[Provider],
    ) -> Result<Session> {
        let mut failures: Vec<String> = Vec::new();
        for &provider in providers {
            if provider != Provider::Cpu {
                if self.provider_append_symbol(provider).is_null() {
                    failures.push(format!(
                        "{} is not exported by this ONNX Runtime",
                        provider.label()
                    ));
                    continue;
                }
            }
            if let Some(blocker) = self.provider_blocker(provider) {
                log::warn!(
                    "{} provider skipped for {}: {blocker}",
                    provider.label(),
                    path.display()
                );
                failures.push(format!("{} provider unavailable: {blocker}", provider.label()));
                continue;
            }
            let key = base_key.for_provider(provider);
            let cached = optimized_model_cache_path(path, &key).is_some_and(|path| path.is_file());
            let opened = match self.open_session_with(path, provider, &key, cached) {
                Ok(session) => Ok(session),
                Err(err) if cached => {
                    log::warn!(
                        "{} cached model failed for {}, rebuilding from source: {err:#}",
                        provider.label(),
                        path.display()
                    );
                    self.open_session_with(path, provider, &key, false)
                        .map_err(|rebuild_err| {
                            anyhow!("rebuilding the optimized model cache: {rebuild_err:#}")
                        })
                }
                Err(err) => Err(err),
            };
            match opened {
                Ok(session) => {
                    if !failures.is_empty() {
                        log::warn!(
                            "{} opened on {}, after: {}",
                            path.display(),
                            provider.label(),
                            failures.join("; ")
                        );
                        session.set_fallback(Some(failures.join("; ")));
                    }
                    log::info!(
                        "ONNX Runtime opened {} on {} (key {}); unsupported nodes may still run on CPU",
                        path.display(),
                        provider.label(),
                        key.short()
                    );
                    return Ok(session);
                }
                Err(err) => {
                    let reason = format!("{err:#}");
                    log::warn!(
                        "{} session failed for {}: {reason}",
                        provider.label(),
                        path.display(),
                    );
                    failures.push(format!("{}: {reason}", provider.label()));
                }
            }
        }
        if failures.is_empty() {
            return Err(anyhow!("no requested execution provider available"));
        }
        Err(anyhow!(
            "no requested execution provider could open {} ({})",
            path.display(),
            failures.join("; ")
        ))
    }

    pub(crate) fn open_cpu_session(&self, path: &Path) -> Result<Session> {
        let key = self
            .session_key(path, &[Provider::Cpu])
            .with_context(|| format!("identifying model {}", path.display()))?;
        let id = key.registry_id();
        self.session_slot(id, || {
            self.open_session_with(path, Provider::Cpu, &key, false)
                .with_context(|| format!("opening CPU model {}", path.display()))
        })
    }

    fn open_session_with(
        &self,
        path: &Path,
        provider: Provider,
        key: &SessionKey,
        use_cached_model: bool,
    ) -> Result<Session> {
        let cold_started = std::time::Instant::now();
        let api = self.api;
        let create_options: unsafe extern "C" fn(*mut *mut c_void) -> Status =
            unsafe { function(api, CREATE_SESSION_OPTIONS) };
        let set_threads: unsafe extern "C" fn(*mut c_void, c_int) -> Status =
            unsafe { function(api, SET_INTRA_OP_NUM_THREADS) };
        let set_optimization: unsafe extern "C" fn(*mut c_void, c_int) -> Status =
            unsafe { function(api, SET_SESSION_GRAPH_OPTIMIZATION_LEVEL) };
        let set_optimized_path: unsafe extern "C" fn(*mut c_void, *const c_void) -> Status =
            unsafe { function(api, SET_OPTIMIZED_MODEL_FILE_PATH) };
        let enable_profiling: unsafe extern "C" fn(*mut c_void, *const c_void) -> Status =
            unsafe { function(api, ENABLE_PROFILING) };
        let create_session: unsafe extern "C" fn(
            *const c_void,
            *const c_void,
            *const c_void,
            *mut *mut c_void,
        ) -> Status = unsafe { function(api, CREATE_SESSION) };
        let release_options: unsafe extern "C" fn(*mut c_void) =
            unsafe { function(api, RELEASE_SESSION_OPTIONS) };
        let mut options = ptr::null_mut();
        unsafe { check(create_options(&mut options), api)? };
        let threads = intra_op_threads();
        // The cache already contains ORT's optimized graph; re-running transforms only adds latency.
        let optimization = if use_cached_model { 0 } else { graph_optimization_level() };
        let result = unsafe {
            let result = check(set_threads(options, threads), api).and_then(|()| {
                check(set_optimization(options, optimization), api)
            });
            result
        };
        let cache_path = optimized_model_cache_path(path, key);
        let result = result.and_then(|()| {
            let Some(cache_path) = cache_path.as_ref().filter(|_| !use_cached_model) else {
                return Ok(());
            };
            // The per-user cache is an optimization, never a prerequisite: an
            // unwritable profile must still open the model from source.
            let setup = (|| -> Result<()> {
                if let Some(dir) = cache_path.parent() {
                    std::fs::create_dir_all(dir)?;
                }
                let cache_path = ort_path(cache_path);
                unsafe {
                    check(set_optimized_path(options, cache_path.as_ptr() as *const c_void), api)
                }
                .context("setting the optimized model cache path")
            })();
            if let Err(error) = setup {
                log::warn!(
                    "optimized model cache disabled for {}: {error:#}",
                    cache_path.display()
                );
            }
            Ok(())
        });
        let result = result.and_then(|()| unsafe { self.append_provider(options, provider) });
        let profile_prefix = std::env::var_os("AURORA_ORT_PROFILE_DIR").map(|dir| {
            let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
            PathBuf::from(dir).join(format!("{stem}-{}-{}", provider.label(), std::process::id()))
        });
        let result = result.and_then(|()| {
            let Some(prefix) = &profile_prefix else {
                return Ok(());
            };
            if let Some(dir) = prefix.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let prefix = ort_path(prefix);
            unsafe { check(enable_profiling(options, prefix.as_ptr() as *const c_void), api) }
                .context("enabling ONNX Runtime profiling")
        });
        let mut session = ptr::null_mut();
        let result = result.and_then(|()| {
            // `OrtCreateSession` takes ORTCHAR_T: wchar_t on Windows, char elsewhere.
            let model_path = if use_cached_model {
                cache_path.as_deref().unwrap_or(path)
            } else {
                path
            };
            let path = ort_path(model_path);
            unsafe {
                check(
                    create_session(self.env, path.as_ptr() as *const c_void, options, &mut session),
                    api,
                )
            }
        });
        unsafe { release_options(options) };
        result.with_context(|| {
            if use_cached_model {
                format!("opening cached model for {}", path.display())
            } else {
                format!("opening {}", path.display())
            }
        })?;
        let cold_us = cold_started.elapsed().as_micros() as u64;
        let (input, outputs) = match unsafe { Session::read_metadata(api, session) } {
            Ok(metadata) => metadata,
            Err(error) => {
                let release: unsafe extern "C" fn(*mut c_void) =
                    unsafe { function(api, RELEASE_SESSION) };
                unsafe { release(session) };
                return Err(error)
                    .with_context(|| format!("reading tensor metadata for {}", path.display()));
            }
        };
        Ok(Session {
            inner: Arc::new(SessionInner {
                api,
                ptr: session,
                provider,
                key: key.clone(),
                fallback: Mutex::new(None),
                profile_active: AtomicBool::new(profile_prefix.is_some()),
                timing: Mutex::new(SessionTiming {
                    cold_us,
                    ..SessionTiming::default()
                }),
            }),
            input,
            outputs,
        })
    }

    pub(crate) fn end_profiling(&self, session: &Session) -> Result<Option<PathBuf>> {
        if !session.inner.profile_active.swap(false, Ordering::AcqRel) {
            return Ok(None);
        }
        let end: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut *mut c_char) -> Status =
            unsafe { function(self.api, SESSION_END_PROFILING) };
        let free: unsafe extern "C" fn(*mut c_void, *mut c_void) -> Status =
            unsafe { function(self.api, ALLOCATOR_FREE) };
        let mut raw = ptr::null_mut();
        unsafe { check(end(session.inner.ptr, self.allocator, &mut raw), self.api) }
            .context("ending ONNX Runtime profiling")?;
        if raw.is_null() {
            return Ok(None);
        }
        let path = unsafe { CStr::from_ptr(raw) }.to_string_lossy().into_owned();
        unsafe { check(free(self.allocator, raw.cast()), self.api) }
            .context("freeing the ONNX Runtime profile path")?;
        Ok(Some(PathBuf::from(path)))
    }

    pub(crate) fn run(
        &self,
        session: &Session,
        input: &mut [f32],
        shape: &[i64],
    ) -> Result<Vec<Vec<f32>>> {
        Ok(self
            .run_typed(session, input, shape)?
            .into_iter()
            .map(|plane| plane.values)
            .collect())
    }

    /// Runs the model. `input` is always f32 on the host; a float16 model input
    /// is converted here, so callers do not need a second tensor type. Outputs
    /// are converted back to f32 but keep their dtype for the caller's checks.
    pub(crate) fn run_typed(
        &self,
        session: &Session,
        input: &mut [f32],
        shape: &[i64],
    ) -> Result<Vec<OutputPlane>> {
        let api = self.api;
        let create_tensor: unsafe extern "C" fn(
            *const c_void,
            *mut c_void,
            usize,
            *const i64,
            usize,
            c_int,
            *mut *mut c_void,
        ) -> Status = unsafe { function(api, CREATE_TENSOR_WITH_DATA) };
        let run: unsafe extern "C" fn(
            *mut c_void,
            *const c_void,
            *const *const c_char,
            *const *const c_void,
            usize,
            *const *const c_char,
            usize,
            *mut *mut c_void,
        ) -> Status = unsafe { function(api, RUN) };
        let release_value: unsafe extern "C" fn(*mut c_void) = unsafe { function(api, RELEASE_VALUE) };

        let prepared = std::time::Instant::now();
        let mut half_input: Vec<u16> = Vec::new();
        let (data, byte_len, input_dtype) = match session.input.element_type {
            FLOAT => (
                input.as_mut_ptr().cast::<c_void>(),
                std::mem::size_of_val(input),
                FLOAT,
            ),
            FLOAT16 => {
                half_input.extend(input.iter().map(|value| float_to_half(*value)));
                (
                    half_input.as_mut_ptr().cast::<c_void>(),
                    half_input.len() * std::mem::size_of::<u16>(),
                    FLOAT16,
                )
            }
            other => bail!(
                "model input dtype {} is neither float32 nor float16 (shape {:?})",
                dtype_name(other),
                shape
            ),
        };
        let mut value = ptr::null_mut();
        unsafe {
            check(
                create_tensor(
                    self.memory_info,
                    data,
                    byte_len,
                    shape.as_ptr(),
                    shape.len(),
                    input_dtype,
                    &mut value,
                ),
                api,
            )?;
        }
        let input_name = CString::new(session.input.name.as_str())?;
        let output_names: Vec<CString> = session
            .outputs
            .iter()
            .map(|o| CString::new(o.name.as_str()).map_err(anyhow::Error::from))
            .collect::<Result<_>>()?;
        let input_names = [input_name.as_ptr()];
        let input_values = [value as *const c_void];
        let output_ptrs: Vec<*const c_char> = output_names.iter().map(|n| n.as_ptr()).collect();
        let mut outputs: Vec<*mut c_void> = vec![ptr::null_mut(); output_names.len()];
        let infer_started = std::time::Instant::now();
        let prepared_us = infer_started.saturating_duration_since(prepared).as_micros() as u64;
        let status = unsafe {
            run(
                session.inner.ptr,
                ptr::null(),
                input_names.as_ptr(),
                input_values.as_ptr(),
                1,
                output_ptrs.as_ptr(),
                output_ptrs.len(),
                outputs.as_mut_ptr(),
            )
        };
        let infer_us = infer_started.elapsed().as_micros() as u64;
        let result = unsafe { check(status, api) };
        unsafe { release_value(value) };
        result.context("running the model")?;

        let post_started = std::time::Instant::now();
        let mut planes = Vec::with_capacity(outputs.len());
        for output in outputs {
            let plane = unsafe { read_tensor(api, output) };
            unsafe { release_value(output) };
            planes.push(plane?);
        }
        let post_us = post_started.elapsed().as_micros() as u64;
        session.record_run(prepared_us, infer_us, post_us);
        Ok(planes)
    }
}

/// Path in the encoding `OrtCreateSession` expects: UTF-16 on Windows, bytes elsewhere.
#[cfg(windows)]
fn ort_path(path: &Path) -> Vec<u16> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str().encode_wide().chain(std::iter::once(0)).collect()
}
#[cfg(not(windows))]
fn ort_path(path: &Path) -> Vec<u8> {
    let mut bytes = path.as_os_str().to_string_lossy().as_bytes().to_vec();
    bytes.push(0);
    bytes
}

#[cfg(windows)]
fn runtime_dirs_from(raw: &std::ffi::OsStr) -> Result<Vec<PathBuf>> {
    let dirs: Vec<PathBuf> = std::env::split_paths(raw)
        .filter(|path| !path.as_os_str().is_empty())
        .collect();
    if dirs.is_empty() {
        bail!("{ORT_RUNTIME_DIRS_ENV} contains no directories");
    }
    if let Some(path) = dirs.iter().find(|path| !path.is_absolute()) {
        bail!(
            "{ORT_RUNTIME_DIRS_ENV} entries must be absolute: {}",
            path.display()
        );
    }
    if let Some(path) = dirs.iter().find(|path| !path.is_dir()) {
        bail!(
            "{ORT_RUNTIME_DIRS_ENV} entry is not a directory: {}",
            path.display()
        );
    }
    Ok(dirs)
}

#[cfg(windows)]
unsafe fn preload_cuda_runtime(library: &Path) -> (Vec<*mut c_void>, Option<String>) {
    if forced_provider() == Some(Provider::Cpu) {
        return (Vec::new(), None);
    }
    let dirs = match std::env::var_os(ORT_RUNTIME_DIRS_ENV) {
        Some(raw) => match runtime_dirs_from(&raw) {
            Ok(dirs) => dirs,
            Err(error) => {
                let message = format!("{error:#}");
                log::warn!("CUDA runtime preload disabled: {message}");
                return (Vec::new(), Some(message));
            }
        },
        None => derived_runtime_dirs(library),
    };
    if dirs.is_empty() {
        return (Vec::new(), None);
    }

    let mut modules = Vec::new();
    let mut failures = Vec::new();
    for &name in CUDA_RUNTIME_DLLS {
        let Some(path) = dirs
            .iter()
            .map(|dir| dir.join(name))
            .find(|path| path.is_file())
        else {
            continue;
        };
        let path_wide = ort_path(&path);
        let module = unsafe {
            LoadLibraryExW(
                path_wide.as_ptr(),
                ptr::null_mut(),
                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
            )
        };
        if module.is_null() {
            failures.push(format!("{}: {}", path.display(), std::io::Error::last_os_error()));
        } else {
            log::info!("preloaded CUDA runtime {}", path.display());
            modules.push(module);
        }
    }

    if modules.is_empty() {
        let message = if failures.is_empty() {
            "no CUDA runtime DLLs were found next to the ONNX Runtime".to_string()
        } else {
            format!("CUDA runtime preload failed: {}", failures.join("; "))
        };
        log::warn!("{message}");
        return (Vec::new(), Some(message));
    }
    if failures.is_empty() {
        return (modules, None);
    }
    log::warn!("CUDA runtime preload partially failed: {}", failures.join("; "));
    (modules, None)
}

#[cfg(windows)]
fn derived_runtime_dirs(library: &Path) -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    if let Some(capi) = library.parent() {
        dirs.push(capi.to_path_buf());
        if let Some(python_root) = capi.parent().and_then(Path::parent) {
            dirs.push(python_root.join("nvidia").join("cudnn").join("bin"));
            dirs.push(python_root.join("tensorrt_libs"));
        }
    }
    if let Some(cuda) = std::env::var_os("CUDA_PATH") {
        let cuda = PathBuf::from(cuda);
        dirs.push(cuda.join("bin").join("x64"));
        dirs.push(cuda.join("bin"));
    }
    let mut seen = std::collections::HashSet::new();
    dirs.retain(|dir| dir.is_dir() && seen.insert(dir.clone()));
    dirs
}

/// Library discovery. Windows never attempts a bare `onnxruntime.dll` because
/// `C:\Windows\System32\onnxruntime.dll` (1.17, API 17) would win the search and
/// is too old for this module (`MIN_ORT_API_VERSION` is 18). macOS prefers the
/// app bundle's `Contents/Frameworks` and binary directory, then the official
/// release names; Homebrew and other user install roots are never guessed.
fn library_candidates() -> Vec<PathBuf> {
    library_candidates_from(|key| std::env::var_os(key), &project_roots())
}

/// Official macOS release names for ORT 1.23.2, most specific first. The
/// versioned files are symlinks in the upstream package, but a bundled copy may
/// ship only one of them.
#[cfg(target_os = "macos")]
const MACOS_ORT_LIB_NAMES: [&str; 4] = [
    "libonnxruntime.1.23.2.dylib",
    "libonnxruntime.1.23.dylib",
    "libonnxruntime.1.dylib",
    "libonnxruntime.dylib",
];

/// Adds `libonnxruntime*.dylib` found directly in `dir`, in a deterministic
/// (name-sorted) order. Missing or unreadable directories are not an error:
/// discovery is best-effort and `resolve_library` reports what was tried.
#[cfg(target_os = "macos")]
fn push_macos_dylibs_in(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut found: Vec<PathBuf> = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| {
                    name.starts_with("libonnxruntime") && name.ends_with(".dylib")
                })
        })
        .collect();
    found.sort();
    out.extend(found);
}

fn library_candidates_from(
    get: impl Fn(&str) -> Option<std::ffi::OsString>,
    roots: &[PathBuf],
) -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = get("AURORA_ORT_LIB") {
        let path = PathBuf::from(path);
        // `dlopen` does not search the current directory for a bare name, so a
        // relative override would fail even when `is_file` saw it. macOS keeps
        // the documented absolute-path contract; other platforms are unchanged.
        #[cfg(target_os = "macos")]
        {
            if path.is_absolute() {
                candidates.push(path);
            } else {
                log::warn!(
                    "AURORA_ORT_LIB must be an absolute path on macOS; ignoring {:?}",
                    path
                );
            }
        }
        #[cfg(not(target_os = "macos"))]
        candidates.push(path);
    }
    if cfg!(windows) {
        for root in roots {
            candidates.push(root.join(PROJECT_ORT_RELATIVE));
        }
        if let Some(dir) = get("AURORA_ORT_DIR") {
            let dir = PathBuf::from(dir);
            candidates.push(dir.join("onnxruntime.dll"));
            candidates.push(dir.join("onnxruntime").join("capi").join("onnxruntime.dll"));
            candidates.push(
                dir.join("python")
                    .join("onnxruntime")
                    .join("capi")
                    .join("onnxruntime.dll"),
            );
        }
    } else {
        #[cfg(target_os = "macos")]
        {
            // A bundled app keeps the dylib at `<...>.app/Contents/Frameworks`,
            // one level above the `Contents/MacOS` binary directory. Only these
            // two executable-relative locations are scanned; Homebrew and other
            // per-user install roots are deliberately not guessed.
            if let Ok(exe) = std::env::current_exe() {
                if let Some(bin_dir) = exe.parent() {
                    if let Some(contents) = bin_dir.parent() {
                        push_macos_dylibs_in(&contents.join("Frameworks"), &mut candidates);
                    }
                    push_macos_dylibs_in(bin_dir, &mut candidates);
                }
            }
            candidates.extend(MACOS_ORT_LIB_NAMES.into_iter().map(PathBuf::from));
            // Explicit, operator-supplied directory; never a hardcoded path.
            if let Some(dir) = get("AURORA_ORT_DIR") {
                let dir = PathBuf::from(dir);
                candidates.extend(MACOS_ORT_LIB_NAMES.into_iter().map(|name| dir.join(name)));
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            candidates.extend(
                ["libonnxruntime.so.1.23", "libonnxruntime.so.1", "libonnxruntime.so"]
                    .into_iter()
                    .map(PathBuf::from),
            );
            candidates.push(PathBuf::from(
                "/usr/lib/x86_64-linux-gnu/libonnxruntime.so.1.23",
            ));
        }
    }
    let mut seen = std::collections::HashSet::new();
    candidates.retain(|path| seen.insert(path.clone()));
    candidates
}

/// Platform-neutral "not found" text, listing every candidate that was tried.
/// Kept separate so the message is testable without touching the filesystem.
fn library_not_found_message(candidates: &[PathBuf]) -> String {
    let tried = candidates
        .iter()
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "ONNX Runtime library not found; set AURORA_ORT_LIB to an absolute library path \
         (tried: {tried})"
    )
}

/// Working directory and every executable ancestor, so a checkout works both
/// from a console in the repo root and from `90_work/build-cache/.../release`.
fn project_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut current = exe.parent();
        let mut depth = 0;
        while let Some(dir) = current {
            roots.push(dir.to_path_buf());
            current = dir.parent();
            depth += 1;
            if depth > 8 {
                break;
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    roots.retain(|path| seen.insert(path.clone()));
    roots
}

fn resolve_library() -> Result<PathBuf> {
    let candidates = library_candidates();
    if candidates.is_empty() {
        bail!("no ONNX Runtime candidate; set AURORA_ORT_LIB");
    }
    for path in &candidates {
        if path.is_file() {
            return Ok(path.clone());
        }
    }
    bail!(library_not_found_message(&candidates))
}

#[cfg(windows)]
unsafe fn load_library(path: &Path) -> Result<*mut c_void> {
    const LOAD_WITH_ALTERED_SEARCH_PATH: u32 = 0x8;
    let wide: Vec<u16> = path
        .as_os_str()
        .to_string_lossy()
        .encode_utf16()
        .chain(std::iter::once(0))
        .collect();
    let handle =
        unsafe { LoadLibraryExW(wide.as_ptr(), ptr::null_mut(), LOAD_WITH_ALTERED_SEARCH_PATH) };
    if handle.is_null() {
        bail!(
            "LoadLibraryExW({}) failed: {}",
            path.display(),
            std::io::Error::last_os_error()
        );
    }
    Ok(handle)
}

#[cfg(not(windows))]
unsafe fn load_library(path: &Path) -> Result<*mut c_void> {
    let name = CString::new(path.as_os_str().to_string_lossy().as_bytes())?;
    let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
    if handle.is_null() {
        let reason = unsafe { CStr::from_ptr(dlerror()) }.to_string_lossy().into_owned();
        bail!("dlopen({}) failed: {reason}", path.display());
    }
    Ok(handle)
}

#[cfg(windows)]
fn configure_gpu_dll_directories(library: &Path) -> Vec<PathBuf> {
    use std::os::windows::ffi::OsStrExt;
    unsafe extern "system" {
        fn SetDefaultDllDirectories(directory_flags: u32) -> i32;
        fn AddDllDirectory(new_directory: *const u16) -> *mut c_void;
    }
    const LOAD_LIBRARY_SEARCH_USER_DIRS: u32 = 0x0400;
    const LOAD_LIBRARY_SEARCH_DEFAULT_DIRS: u32 = 0x1000;

    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Some(list) = std::env::var_os(DLL_DIRS_ENV) {
        dirs.extend(std::env::split_paths(&list));
    }
    dirs.extend(derived_runtime_dirs(library));
    let mut seen = std::collections::HashSet::new();
    dirs.retain(|dir| dir.is_dir() && seen.insert(dir.clone()));

    let flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS;
    if unsafe { SetDefaultDllDirectories(flags) } == 0 {
        log::warn!(
            "SetDefaultDllDirectories failed ({}); relying on AddDllDirectory and preloads",
            std::io::Error::last_os_error()
        );
    }
    for dir in &dirs {
        let wide: Vec<u16> = dir
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect();
        if unsafe { AddDllDirectory(wide.as_ptr()) }.is_null() {
            log::debug!("AddDllDirectory({}) failed", dir.display());
        }
    }
    dirs
}

#[cfg(not(windows))]
fn configure_gpu_dll_directories(_library: &Path) -> Vec<PathBuf> {
    Vec::new()
}

unsafe fn check(status: Status, api: ApiPtr) -> Result<()> {
    if status.is_null() {
        return Ok(());
    }
    let message: unsafe extern "C" fn(Status) -> *const c_char =
        unsafe { function(api, GET_ERROR_MESSAGE) };
    let text = unsafe { CStr::from_ptr(message(status)) }.to_string_lossy().into_owned();
    let release: unsafe extern "C" fn(Status) = unsafe { function(api, RELEASE_STATUS) };
    unsafe { release(status) };
    Err(anyhow!("{text}"))
}

/// One model output: f32 values plus the dtype and shape they came from.
pub(crate) struct OutputPlane {
    pub(crate) values: Vec<f32>,
    pub(crate) dtype: c_int,
    pub(crate) dims: Vec<i64>,
}

unsafe fn read_tensor(api: ApiPtr, value: *const c_void) -> Result<OutputPlane> {
    let type_and_shape: unsafe extern "C" fn(*const c_void, *mut *mut c_void) -> Status =
        unsafe { function(api, GET_TENSOR_TYPE_AND_SHAPE) };
    let element_count: unsafe extern "C" fn(*const c_void, *mut usize) -> Status =
        unsafe { function(api, GET_TENSOR_SHAPE_ELEMENT_COUNT) };
    let element_type: unsafe extern "C" fn(*const c_void, *mut c_int) -> Status =
        unsafe { function(api, GET_TENSOR_ELEMENT_TYPE) };
    let data: unsafe extern "C" fn(*const c_void, *mut *mut c_void) -> Status =
        unsafe { function(api, GET_TENSOR_MUTABLE_DATA) };
    let dim_count: unsafe extern "C" fn(*const c_void, *mut usize) -> Status =
        unsafe { function(api, GET_DIMENSIONS_COUNT) };
    let dims_fn: unsafe extern "C" fn(*const c_void, *mut i64, usize) -> Status =
        unsafe { function(api, GET_DIMENSIONS) };
    let release_info: unsafe extern "C" fn(*mut c_void) =
        unsafe { function(api, RELEASE_TENSOR_TYPE_AND_SHAPE) };
    let mut info = ptr::null_mut();
    unsafe { check(type_and_shape(value, &mut info), api)? };
    let mut count = 0usize;
    let mut dtype = -1;
    let mut dim_count_value = 0usize;
    let count_status = unsafe { element_count(info, &mut count) };
    let dtype_status = unsafe { element_type(info, &mut dtype) };
    let dims_status = unsafe { dim_count(info, &mut dim_count_value) };
    let mut dims = vec![0i64; dim_count_value];
    let read_status = unsafe { dims_fn(info, dims.as_mut_ptr(), dims.len()) };
    unsafe { release_info(info) };
    unsafe { check(dtype_status, api)? };
    unsafe { check(count_status, api)? };
    unsafe { check(dims_status, api)? };
    unsafe { check(read_status, api)? };
    if dtype != FLOAT && dtype != FLOAT16 {
        bail!(
            "model output dtype {} is neither float32 nor float16",
            dtype_name(dtype)
        );
    }
    let mut pointer = ptr::null_mut();
    unsafe { check(data(value, &mut pointer), api)? };
    if pointer.is_null() {
        bail!("model output has no data");
    }
    let values = if dtype == FLOAT16 {
        // FP16 logits must never reach feathering/alpha unvalidated: ORT's
        // half-float has the same alignment as u16, and converting it as f32
        // would read past the buffer on odd element counts and produce garbage
        // probabilities. Convert with finite checks right here.
        let raw = unsafe { std::slice::from_raw_parts(pointer as *const u16, count) };
        raw.iter()
            .map(|&bits| {
                let value = half_to_f32(bits);
                if !value.is_finite() {
                    bail!("model FP16 output is not finite");
                }
                Ok(value)
            })
            .collect::<Result<Vec<f32>>>()?
    } else {
        let floats = unsafe { std::slice::from_raw_parts(pointer as *const f32, count) };
        if floats.iter().any(|value| !value.is_finite()) {
            bail!("model float32 output is not finite");
        }
        floats.to_vec()
    };
    Ok(OutputPlane { values, dtype, dims })
}

/// Exact IEEE-754 binary16 → binary32 conversion, without depending on a
/// half-float crate. Subnormals, infinities and NaNs are handled by the bit
/// layout; [`read_tensor`] rejects the last two before any consumer sees them.
/// Element types this module can feed and read. Mirrors the ONNX Runtime enum.
pub(crate) const DT_FLOAT: c_int = FLOAT;
pub(crate) const DT_FLOAT16: c_int = FLOAT16;

pub(crate) fn dtype_name(dtype: c_int) -> String {
    match dtype {
        FLOAT => "float32".to_string(),
        FLOAT16 => "float16".to_string(),
        other => format!("dtype({other})"),
    }
}

fn dtype_label(dtype: c_int) -> String {
    dtype_name(dtype)
}

pub(crate) fn half_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x1F;
    let mantissa = u32::from(bits & 0x03FF) << 13;
    match exponent {
        0 => {
            // Subnormal: (mant/1024) * 2^-14, computed by subtracting 2^-14
            // from the same bit pattern read as a normal f32 with exponent 113.
            let base = f32::from_bits(sign | (113 << 23) | mantissa);
            base - f32::from_bits(sign | (113 << 23))
        }
        31 => f32::from_bits(sign | (255 << 23) | mantissa),
        _ => f32::from_bits(sign | ((u32::from(exponent) + 127 - 15) << 23) | mantissa),
    }
}

/// f32 → IEEE-754 binary16 with round-to-nearest-even. Values beyond half's
/// range become infinities and NaN payloads are kept as quiet NaN.
pub(crate) fn float_to_half(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7F_FFFF;
    if exponent == 0xFF {
        let payload = if mantissa != 0 {
            0x200 | ((mantissa >> 13) as u16 & 0x3FF)
        } else {
            0
        };
        return sign | 0x7C00 | payload;
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 0x1F {
        return sign | 0x7C00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = (14 - half_exponent) as u32;
        let mut half = (mantissa >> shift) as u16;
        let round_bit = 1u32 << (shift - 1);
        if mantissa & round_bit != 0
            && ((mantissa & (round_bit - 1)) != 0 || half & 1 != 0)
        {
            half += 1;
        }
        return sign | half;
    }
    let mut half = (((half_exponent as u32) << 10) as u16) | ((mantissa >> 13) as u16);
    let round = mantissa & 0x1FFF;
    if round > 0x1000 || (round == 0x1000 && half & 1 != 0) {
        half += 1;
    }
    sign | half
}

#[derive(Clone, Debug)]
pub(crate) struct TensorMeta {
    pub(crate) name: String,
    pub(crate) dims: Vec<i64>,
    pub(crate) element_type: c_int,
}

/// Cumulative timings (microseconds) for one session. `cold_us` is the whole
/// `OrtCreateSession` + metadata read; `warmup_us` is the first `Run`, and the
/// inference/post counters include it. `runs` excludes the warmup.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct SessionTiming {
    cold_us: u64,
    warmup_us: u64,
    prepared_us: u64,
    infer_us: u64,
    post_us: u64,
    runs: u64,
}

impl SessionTiming {
    /// Per-inference stage timings, excluding the warmup run.
    pub(crate) fn stages(&self) -> (u64, u64, u64) {
        let runs = self.measured_runs().max(1);
        (
            self.prepared_us / runs,
            self.infer_us / runs,
            self.post_us / runs,
        )
    }

    pub(crate) fn cold_us(&self) -> u64 {
        self.cold_us
    }

    pub(crate) fn warmup_us(&self) -> u64 {
        self.warmup_us
    }

    /// `Run` calls after the first one; the first is the warmup.
    pub(crate) fn measured_runs(&self) -> u64 {
        self.runs.saturating_sub(1)
    }
}

struct SessionInner {
    api: ApiPtr,
    ptr: *mut c_void,
    provider: Provider,
    key: SessionKey,
    /// Human-readable reasons earlier provider choices failed, if any.
    fallback: Mutex<Option<String>>,
    profile_active: AtomicBool,
    timing: Mutex<SessionTiming>,
}

unsafe impl Send for SessionInner {}
unsafe impl Sync for SessionInner {}

impl Drop for SessionInner {
    fn drop(&mut self) {
        if self.ptr.is_null() {
            // A test-only handle without an ORT session behind it.
            return;
        }
        let release: unsafe extern "C" fn(*mut c_void) = unsafe { function(self.api, RELEASE_SESSION) };
        unsafe { release(self.ptr) };
    }
}

/// A reference to one ORT session. Cloning is cheap and results in the same
/// underlying session: session creation is single-flight per [`SessionKey`], so
/// concurrent callers share one open (and its GPU memory) instead of racing.
#[derive(Clone)]
pub(crate) struct Session {
    inner: Arc<SessionInner>,
    pub(crate) input: TensorMeta,
    pub(crate) outputs: Vec<TensorMeta>,
}

impl Session {
    /// The provider the session was opened on. This is the *effective* EP in
    /// the sense that it was actually appended and accepted by ORT; nodes the
    /// provider does not support can still run on CPU inside the same session.
    pub(crate) fn provider(&self) -> Provider {
        self.inner.provider
    }

    /// Provider selection order actually attempted and the reason for a
    /// fallback, if the first choice was skipped or failed.
    pub(crate) fn provider_trace(&self) -> (String, Option<String>) {
        (
            self.inner.key.providers.clone(),
            self.inner
                .fallback
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone(),
        )
    }

    fn set_fallback(&self, reason: Option<String>) {
        *self
            .inner
            .fallback
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = reason;
    }

    pub(crate) fn session_key(&self) -> &SessionKey {
        &self.inner.key
    }

    /// Shape/dtype/layout of the single input, as reported by ORT. The layout
    /// is inferred from the NCHW/NHWC position of the 3-channel dimension.
    pub(crate) fn input_layout(&self) -> (String, &[i64], &'static str) {
        let dims = &self.input.dims;
        let layout = if dims.len() == 4 && dims.get(1) == Some(&3) {
            "NCHW"
        } else if dims.len() == 4 && dims.get(3) == Some(&3) {
            "NHWC"
        } else {
            "unknown"
        };
        (dtype_label(self.input.element_type), dims, layout)
    }

    pub(crate) fn output_layouts(&self) -> Vec<(String, Vec<i64>, String, String)> {
        self.outputs
            .iter()
            .map(|output| {
                let dims = &output.dims;
                let layout = if dims.len() == 4 && dims.get(1).is_some_and(|d| *d > 1 && *d < 64) {
                    "NCHW"
                } else if dims.len() == 4 && dims.get(3).is_some_and(|d| *d > 1 && *d < 64) {
                    "NHWC"
                } else {
                    "unknown"
                };
                (
                    output.name.clone(),
                    dims.clone(),
                    dtype_label(output.element_type),
                    layout.to_string(),
                )
            })
            .collect()
    }

    pub(crate) fn timing(&self) -> SessionTiming {
        *self
            .inner
            .timing
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    fn record_run(&self, prepared_us: u64, infer_us: u64, post_us: u64) {
        let mut timing = self
            .inner
            .timing
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if timing.runs == 0 {
            timing.warmup_us = infer_us + prepared_us + post_us;
            timing.runs = 1;
            return;
        }
        timing.prepared_us += prepared_us;
        timing.infer_us += infer_us;
        timing.post_us += post_us;
        timing.runs += 1;
    }

    pub(crate) fn input_contract(&self) -> (&str, &[i64], c_int) {
        (&self.input.name, &self.input.dims, self.input.element_type)
    }

    pub(crate) fn output_contract(&self) -> Vec<(&str, &[i64], c_int)> {
        self.outputs
            .iter()
            .map(|output| {
                (
                    output.name.as_str(),
                    output.dims.as_slice(),
                    output.element_type,
                )
            })
            .collect()
    }

    pub(crate) fn end_profiling(&self) -> Result<Option<PathBuf>> {
        let runtime = Runtime::get().context("ONNX Runtime is not available")?;
        runtime.end_profiling(self)
    }

    /// Reads the single input and every output tensor's shape and dtype. Kept
    /// separate from construction so a failed metadata read releases the ORT
    /// session exactly once.
    unsafe fn read_metadata(
        api: ApiPtr,
        session: *mut c_void,
    ) -> Result<(TensorMeta, Vec<TensorMeta>)> {
        let input_count: unsafe extern "C" fn(*mut c_void, *mut usize) -> Status =
            unsafe { function(api, SESSION_GET_INPUT_COUNT) };
        let output_count: unsafe extern "C" fn(*mut c_void, *mut usize) -> Status =
            unsafe { function(api, SESSION_GET_OUTPUT_COUNT) };
        let input_name: unsafe extern "C" fn(*mut c_void, usize, *mut c_void, *mut *mut c_char) -> Status =
            unsafe { function(api, SESSION_GET_INPUT_NAME) };
        let output_name: unsafe extern "C" fn(*mut c_void, usize, *mut c_void, *mut *mut c_char) -> Status =
            unsafe { function(api, SESSION_GET_OUTPUT_NAME) };
        let input_type: unsafe extern "C" fn(*mut c_void, usize, *mut *mut c_void) -> Status =
            unsafe { function(api, SESSION_GET_INPUT_TYPE_INFO) };
        let output_type: unsafe extern "C" fn(*mut c_void, usize, *mut *mut c_void) -> Status =
            unsafe { function(api, SESSION_GET_OUTPUT_TYPE_INFO) };
        let cast: unsafe extern "C" fn(*const c_void, *mut *mut c_void) -> Status =
            unsafe { function(api, CAST_TYPE_INFO_TO_TENSOR_INFO) };
        let element_type: unsafe extern "C" fn(*const c_void, *mut c_int) -> Status =
            unsafe { function(api, GET_TENSOR_ELEMENT_TYPE) };
        let dim_count: unsafe extern "C" fn(*const c_void, *mut usize) -> Status =
            unsafe { function(api, GET_DIMENSIONS_COUNT) };
        let dims: unsafe extern "C" fn(*const c_void, *mut i64, usize) -> Status =
            unsafe { function(api, GET_DIMENSIONS) };
        let release_type_info: unsafe extern "C" fn(*mut c_void) = unsafe { function(api, RELEASE_TYPE_INFO) };
        let free: unsafe extern "C" fn(*mut c_void, *mut c_void) = unsafe { function(api, ALLOCATOR_FREE) };
        let get_allocator: unsafe extern "C" fn(*mut *mut c_void) -> Status =
            unsafe { function(api, GET_ALLOCATOR_WITH_DEFAULT_OPTIONS) };

        let mut allocator = ptr::null_mut();
        unsafe { check(get_allocator(&mut allocator), api)? };

        let mut inputs = 0usize;
        let mut outputs = 0usize;
        unsafe { check(input_count(session, &mut inputs), api)? };
        unsafe { check(output_count(session, &mut outputs), api)? };
        if inputs != 1 || outputs == 0 {
            bail!("expected one input and at least one output, got {inputs}/{outputs}");
        }

        let read = |index: usize, is_input: bool| -> Result<TensorMeta> {
            let mut raw_name: *mut c_char = ptr::null_mut();
            let name_status = unsafe {
                if is_input {
                    input_name(session, index, allocator, &mut raw_name)
                } else {
                    output_name(session, index, allocator, &mut raw_name)
                }
            };
            unsafe { check(name_status, api)? };
            let name = unsafe { CStr::from_ptr(raw_name) }.to_string_lossy().into_owned();
            unsafe { free(allocator, raw_name as *mut c_void) };

            let mut type_info = ptr::null_mut();
            let type_status = unsafe {
                if is_input {
                    input_type(session, index, &mut type_info)
                } else {
                    output_type(session, index, &mut type_info)
                }
            };
            unsafe { check(type_status, api)? };
            let mut tensor_info = ptr::null_mut();
            let cast_status = unsafe { cast(type_info, &mut tensor_info) };
            // `tensor_info` borrows from `type_info`, so it has to be read first.
            let meta = (|| -> Result<(c_int, Vec<i64>)> {
                unsafe { check(cast_status, api)? };
                let mut dtype = -1;
                let type_status = unsafe { element_type(tensor_info, &mut dtype) };
                unsafe { check(type_status, api)? };
                let mut count = 0usize;
                let count_status = unsafe { dim_count(tensor_info, &mut count) };
                unsafe { check(count_status, api)? };
                let mut shape = vec![0i64; count];
                let dims_status = unsafe { dims(tensor_info, shape.as_mut_ptr(), count) };
                unsafe { check(dims_status, api)? };
                Ok((dtype, shape))
            })();
            unsafe { release_type_info(type_info) };
            let (element_type, dims) = meta?;
            Ok(TensorMeta { name, dims, element_type })
        };

        let input_meta = read(0, true)?;
        let output_meta = (0..outputs).map(|i| read(i, false)).collect::<Result<Vec<_>>>()?;
        Ok((input_meta, output_meta))
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        let release: unsafe extern "C" fn(*mut c_void) = unsafe { function(self.api, RELEASE_ENV) };
        unsafe { release(self.env) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn runtime_dirs_require_absolute_existing_directories() {
        use std::ffi::OsStr;

        let absolute = std::env::temp_dir();
        assert_eq!(
            runtime_dirs_from(absolute.as_os_str()).unwrap(),
            vec![absolute]
        );
        assert!(runtime_dirs_from(OsStr::new("relative")).is_err());
        assert!(runtime_dirs_from(OsStr::new("")).is_err());
    }

    #[test]
    fn providers_prefer_gpu_and_always_end_with_cpu() {
        assert_eq!(providers_from(false, false), vec![Provider::Cpu]);
        assert_eq!(
            providers_from(true, true),
            vec![Provider::Cuda, Provider::DirectMl, Provider::Cpu]
        );
        assert_eq!(providers_from(true, false), vec![Provider::Cuda, Provider::Cpu]);
        assert_eq!(providers_from(false, true), vec![Provider::DirectMl, Provider::Cpu]);
    }

    #[test]
    fn provider_names_parse_and_round_trip() {
        for (raw, expected) in [
            ("tensorrt", Provider::TensorRt),
            ("trt", Provider::TensorRt),
            ("CUDA", Provider::Cuda),
            ("dml", Provider::DirectMl),
            ("directml", Provider::DirectMl),
            ("coreml", Provider::CoreMl),
            ("core_ml", Provider::CoreMl),
            (" cpu ", Provider::Cpu),
        ] {
            assert_eq!(provider_from_name(raw), Some(expected), "{raw:?}");
        }
        assert_eq!(provider_from_name("core ml"), None);
        assert_eq!(provider_from_name("metal"), None);
        for provider in [
            Provider::TensorRt,
            Provider::Cuda,
            Provider::DirectMl,
            Provider::CoreMl,
            Provider::Cpu,
        ] {
            assert_eq!(
                provider_from_name(provider.label()),
                Some(provider),
                "label {:?} must parse back",
                provider.label()
            );
        }
    }

    #[test]
    fn macos_default_plan_is_cpu_until_coreml_parity_is_measured() {
        assert_eq!(
            providers_for_os("macos", false, false, true),
            vec![Provider::Cpu],
            "CoreML is available but not auto-selected without parity evidence"
        );
        assert_eq!(
            providers_for_os("macos", false, false, false),
            vec![Provider::Cpu],
            "no CoreML symbol means CPU only"
        );
        assert_eq!(
            providers_for_os("macos", true, true, false),
            vec![Provider::Cpu],
            "CUDA/DirectML are not available on macOS"
        );
    }

    #[test]
    fn windows_and_linux_default_plan_ignores_coreml() {
        assert_eq!(
            providers_for_os("windows", true, true, true),
            vec![Provider::Cuda, Provider::DirectMl, Provider::Cpu]
        );
        assert_eq!(
            providers_for_os("windows", false, true, true),
            vec![Provider::DirectMl, Provider::Cpu]
        );
        assert_eq!(providers_for_os("linux", false, false, true), vec![Provider::Cpu]);
    }

    #[test]
    fn overrides_can_request_coreml_and_cpu_is_always_kept() {
        assert_eq!(
            providers_from_override("coreml,cpu"),
            vec![Provider::CoreMl, Provider::Cpu]
        );
        assert_eq!(
            providers_from_override("coreml,coreml"),
            vec![Provider::CoreMl, Provider::Cpu],
            "duplicates are dropped and CPU is appended"
        );
        assert_eq!(providers_from_override("bogus"), vec![Provider::Cpu]);
        // A forced provider the runtime cannot export falls back to CPU, never
        // to a different accelerator.
        assert_eq!(
            apply_override(vec![Provider::Cpu], Some(Provider::CoreMl)),
            vec![Provider::Cpu]
        );
        assert_eq!(
            apply_override(vec![Provider::CoreMl, Provider::Cpu], Some(Provider::CoreMl)),
            vec![Provider::CoreMl, Provider::Cpu]
        );
    }

    #[test]
    fn tensorrt_is_not_in_the_default_order_but_can_be_prepended() {
        let default = providers_from(true, true);
        assert!(!default.contains(&Provider::TensorRt));
        assert_eq!(
            with_tensor_rt_first(default),
            vec![Provider::TensorRt, Provider::Cuda, Provider::DirectMl, Provider::Cpu]
        );
    }

    #[test]
    fn override_pins_one_provider_and_keeps_cpu_as_fallback() {
        let probed = providers_from(false, true); // [DirectMl, Cpu]
        assert_eq!(apply_override(probed.clone(), None), probed);
        assert_eq!(
            apply_override(probed.clone(), Some(Provider::DirectMl)),
            vec![Provider::DirectMl, Provider::Cpu]
        );
        assert_eq!(apply_override(probed.clone(), Some(Provider::Cpu)), vec![Provider::Cpu]);
        // A provider this build cannot export must not select another GPU.
        assert_eq!(
            apply_override(probed.clone(), Some(Provider::Cuda)),
            vec![Provider::Cpu]
        );
    }

    #[test]
    fn provider_export_names_match_the_windows_distributions() {
        assert_eq!(
            provider_symbol(Provider::TensorRt),
            Some("OrtSessionOptionsAppendExecutionProvider_Tensorrt")
        );
        assert_eq!(
            provider_symbol(Provider::Cuda),
            Some("OrtSessionOptionsAppendExecutionProvider_CUDA")
        );
        assert_eq!(
            provider_symbol(Provider::DirectMl),
            Some("OrtSessionOptionsAppendExecutionProvider_DML")
        );
        assert_eq!(provider_symbol(Provider::Cpu), None);
    }

    #[test]
    fn coreml_symbol_candidates_match_the_real_export() {
        assert_eq!(
            provider_symbol(Provider::CoreMl),
            Some("OrtSessionOptionsAppendExecutionProvider_CoreML")
        );
        let candidates = append_symbol_candidates(Provider::CoreMl);
        assert!(
            candidates.contains(&"OrtSessionOptionsAppendExecutionProvider_CoreML"),
            "{candidates:?}"
        );
        // macOS `dlsym` is case-sensitive; the capital-M spelling is a fallback.
        assert!(
            candidates.contains(&"OrtSessionOptionsAppendExecutionProvider_CoreMl"),
            "{candidates:?}"
        );
        assert!(append_symbol_candidates(Provider::Cpu).is_empty());
    }

    #[test]
    fn not_found_message_is_platform_neutral_and_lists_every_candidate() {
        let candidates = vec![
            PathBuf::from("libonnxruntime.1.23.2.dylib"),
            PathBuf::from("/tmp/no-such-dir/libonnxruntime.dylib"),
        ];
        let message = library_not_found_message(&candidates);
        assert!(message.contains("AURORA_ORT_LIB"), "{message}");
        for candidate in &candidates {
            assert!(
                message.contains(&candidate.display().to_string()),
                "missing {:?} in {message}",
                candidate
            );
        }
        assert!(!message.contains(".dll"), "the message must be platform-neutral: {message}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_library_candidates_use_official_names_and_no_homebrew_paths() {
        let names: Vec<String> = library_candidates_from(|_| None, &[])
            .iter()
            .map(|path| path.display().to_string())
            .collect();
        for expected in MACOS_ORT_LIB_NAMES {
            assert!(
                names.iter().any(|name| name == expected),
                "missing {expected:?} in {names:?}"
            );
        }
        assert!(
            !names.iter().any(|name| name.contains("libonnxruntime.23.dylib")),
            "the non-standard short version name must be gone: {names:?}"
        );
        assert!(
            !names.iter().any(|name| name.contains("libonnxruntime.dylib.23")),
            "the non-standard suffix name must be gone: {names:?}"
        );
        assert!(
            !names
                .iter()
                .any(|name| name.to_ascii_lowercase().contains("homebrew")),
            "no Homebrew path may be guessed: {names:?}"
        );
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_library_candidates_try_the_explicit_path_first() {
        let explicit = PathBuf::from("/tmp/aurora-ort/libonnxruntime.dylib");
        let candidates = library_candidates_from(
            |key| (key == "AURORA_ORT_LIB").then(|| explicit.clone().into_os_string()),
            &[],
        );
        assert_eq!(candidates.first(), Some(&explicit));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn macos_ignores_a_relative_aurora_ort_lib() {
        let relative = "vendor/libonnxruntime.dylib";
        let candidates = library_candidates_from(
            |key| (key == "AURORA_ORT_LIB").then(|| std::ffi::OsString::from(relative)),
            &[],
        );
        assert!(
            !candidates.contains(&PathBuf::from(relative)),
            "a relative override must not enter the candidate list: {candidates:?}"
        );
    }

    #[test]
    fn local_runtime_reports_its_providers() {
        let Some(runtime) = Runtime::get() else {
            println!("ONNX Runtime unavailable; no execution provider list to print");
            return;
        };
        let (library, version, api) = runtime.describe();
        let providers = runtime.available_providers();
        let labels: Vec<&str> = providers.iter().map(|p| p.label()).collect();
        println!(
            "ONNX Runtime {} v{} api{} providers: {labels:?}",
            library.display(),
            version,
            api
        );
        assert_eq!(providers.last(), Some(&Provider::Cpu), "CPU is the mandatory fallback");
    }

    fn key_for(model_hash: u64, providers: &str, options: &str) -> SessionKey {
        SessionKey {
            model_hash,
            model_len: 1024,
            ort_version: "1.31.1".to_string(),
            ort_api: 23,
            platform: "windows-x86_64".to_string(),
            runtime: "onnxruntime.dll:1024".to_string(),
            precision: "fp32".to_string(),
            providers: providers.to_string(),
            options: options.to_string(),
        }
    }

    #[test]
    fn session_key_changes_with_every_component() {
        let base = key_for(7, "CUDA,CPU", "opt=99;threads=8");
        assert_eq!(base.digest(), base.clone().digest(), "digest must be stable");
        assert_ne!(base.digest(), key_for(8, "CUDA,CPU", "opt=99;threads=8").digest());
        assert_ne!(base.digest(), key_for(7, "DirectML,CPU", "opt=99;threads=8").digest());
        assert_ne!(base.digest(), key_for(7, "CUDA,CPU", "opt=0;threads=8").digest());
        let mut other_runtime = base.clone();
        other_runtime.ort_version = "1.30.0".to_string();
        assert_ne!(base.digest(), other_runtime.digest());
        let mut other_api = base.clone();
        other_api.ort_api = 22;
        assert_ne!(base.digest(), other_api.digest());
        let mut other_len = base.clone();
        other_len.model_len = 2048;
        assert_ne!(base.digest(), other_len.digest());

        let cpu = base.for_provider(Provider::Cpu);
        assert_eq!(cpu.providers, "CPU");
        assert_ne!(cpu.digest(), base.digest(), "per-EP cache must not collide");
    }

    #[test]
    fn session_key_separates_platform_runtime_precision_and_coreml() {
        let base = key_for(7, "CPU", "opt=99;threads=8");
        let mut mac = base.clone();
        mac.platform = "macos-aarch64".to_string();
        assert_ne!(base.digest(), mac.digest(), "platform/arch must be part of the key");

        let mut other_runtime = base.clone();
        other_runtime.runtime = "libonnxruntime.1.20.1.dylib:54213856".to_string();
        assert_ne!(
            base.digest(),
            other_runtime.digest(),
            "the resolved library identity must be part of the key"
        );

        let mut fp16 = base.clone();
        fp16.precision = "fp16".to_string();
        assert_ne!(base.digest(), fp16.digest(), "precision must be part of the key");

        let coreml = base.for_provider(Provider::CoreMl);
        assert_eq!(coreml.providers, "CoreML");
        assert_ne!(
            coreml.digest(),
            base.digest(),
            "a CPU and a CoreML session must never share a cache entry"
        );
    }

    #[test]
    fn precision_is_derived_from_the_model_file_name() {
        assert_eq!(precision_for_path(Path::new("subjectMaskingFP16.onnx")), "fp16");
        assert_eq!(precision_for_path(Path::new("FaceDetectionFP16-640.onnx")), "fp16");
        assert_eq!(precision_for_path(Path::new("subjectMasking.onnx")), "fp32");
        assert_eq!(precision_for_path(Path::new("skinMasking.onnx")), "fp32");
    }

    #[test]
    fn provider_options_hide_the_tensorrt_cache_path() {
        let id = trt_dir_identity(Some("/Users/alice/private/trt-cache"));
        assert_eq!(id.len(), 16, "{id}");
        assert!(!id.contains("alice") && !id.contains('/'), "{id}");
        assert_eq!(trt_dir_identity(None), "-");
        assert_eq!(trt_dir_identity(Some("")), "-");
        assert_ne!(
            trt_dir_identity(Some("/tmp/cache-a")),
            trt_dir_identity(Some("/tmp/cache-b")),
            "different directories must still produce different keys"
        );
    }

    #[test]
    fn runtime_identity_keeps_only_the_file_name_and_size() {
        let dir = std::env::temp_dir().join(format!("bo8-runtime-id-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("libonnxruntime.1.23.2.dylib");
        std::fs::write(&path, b"not a real dylib").unwrap();
        let identity = runtime_identity(&path);
        assert_eq!(
            identity,
            format!("libonnxruntime.1.23.2.dylib:{}", b"not a real dylib".len())
        );
        assert!(
            !identity.contains(&dir.to_string_lossy().to_string()),
            "the cache key/log must not contain an absolute user path: {identity}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn optimized_cache_path_is_namespaced_by_the_session_key() {
        let dir = Path::new("C:/cache");
        let base = key_for(7, "CUDA", "opt=99");
        let path = optimized_model_cache_path_in(dir, Path::new("subjectMasking.onnx"), &base);
        let other = optimized_model_cache_path_in(
            dir,
            Path::new("subjectMasking.onnx"),
            &key_for(8, "CUDA", "opt=99"),
        );
        assert_eq!(path.parent(), Some(dir));
        assert_ne!(path, other);
        let name = path.file_name().unwrap().to_string_lossy();
        assert!(name.starts_with("subjectMasking-"), "{name}");
        assert!(name.ends_with(".onnx"), "{name}");
    }

    #[test]
    fn model_fingerprint_tracks_content_and_length() {
        let dir = std::env::temp_dir().join(format!("bo8-fingerprint-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.onnx");
        std::fs::write(&path, b"first model body").unwrap();
        let (first, first_len) = model_fingerprint(&path).unwrap();
        let (again, again_len) = model_fingerprint(&path).unwrap();
        assert_eq!((first, first_len), (again, again_len), "fingerprint is cached and stable");
        std::fs::write(&path, b"second model body with a different length").unwrap();
        let (second, second_len) = model_fingerprint(&path).unwrap();
        assert_ne!(first_len, second_len);
        assert_ne!(first, second, "content change must change the fingerprint");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn half_precision_conversion_matches_the_ieee_patterns() {
        assert_eq!(float_to_half(1.0), 0x3C00);
        assert_eq!(float_to_half(-2.0), 0xC000);
        assert_eq!(float_to_half(0.5), 0x3800);
        assert_eq!(float_to_half(0.0), 0x0000);
        assert_eq!(float_to_half(-0.0), 0x8000);
        assert_eq!(float_to_half(65504.0), 0x7BFF, "largest finite half");
        assert_eq!(float_to_half(65_520.0), 0x7C00, "rounds to infinity");
        assert_eq!(float_to_half(f32::INFINITY), 0x7C00);
        assert_eq!(float_to_half(f32::NEG_INFINITY), 0xFC00);
        assert_eq!(float_to_half(2.0f32.powi(-24)), 0x0001, "smallest subnormal");
        assert_eq!(float_to_half(2.0f32.powi(-26)), 0x0000, "underflow to zero");
        assert!(half_to_f32(float_to_half(f32::NAN)).is_nan());

        assert_eq!(half_to_f32(0x3C00), 1.0);
        assert_eq!(half_to_f32(0xC000), -2.0);
        assert_eq!(half_to_f32(0x0001), 2.0f32.powi(-24));
        assert_eq!(half_to_f32(0x7BFF), 65504.0);
        assert_eq!(half_to_f32(0xFC00), f32::NEG_INFINITY);

        // A probability roundtrip through FP16 stays finite and in range.
        for value in [0.0f32, 0.001, 0.25, 0.5, 0.75, 0.999] {
            let roundtrip = half_to_f32(float_to_half(value));
            assert!((roundtrip - value).abs() < 0.001, "{value} -> {roundtrip}");
            assert!((0.0..=1.0).contains(&roundtrip));
        }
    }

    #[test]
    fn dtype_names_use_the_onnx_wire_values() {
        assert_eq!(dtype_name(1), "float32");
        assert_eq!(dtype_name(10), "float16");
        assert_eq!(dtype_name(7), "dtype(7)");
        assert_eq!(DT_FLOAT, 1);
        assert_eq!(DT_FLOAT16, 10);
    }

    #[test]
    fn single_flight_runs_one_leader_and_shares_the_result() {
        static FLIGHTS: OnceLock<Mutex<HashMap<String, Arc<Flight<u32>>>>> = OnceLock::new();
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        CALLS.store(0, Ordering::SeqCst);
        let handles: Vec<_> = (0..8)
            .map(|_| {
                std::thread::spawn(|| {
                    single_flight(&FLIGHTS, "same-key".to_string(), || {
                        CALLS.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        Ok(42u32)
                    })
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap().unwrap(), 42);
        }
        assert_eq!(CALLS.load(Ordering::SeqCst), 1, "waiters must share the leader's work");
    }

    #[test]
    fn single_flight_shares_errors_but_retries_the_next_call() {
        static FLIGHTS: OnceLock<Mutex<HashMap<String, Arc<Flight<u32>>>>> = OnceLock::new();
        static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        CALLS.store(0, Ordering::SeqCst);
        let handles: Vec<_> = (0..4)
            .map(|_| {
                std::thread::spawn(|| {
                    single_flight(&FLIGHTS, "error-key".to_string(), || {
                        CALLS.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(10));
                        Err(anyhow!("open failed"))
                    })
                })
            })
            .collect();
        for handle in handles {
            assert!(handle.join().unwrap().is_err());
        }
        assert_eq!(CALLS.load(Ordering::SeqCst), 1);
        // A new generation (new key) starts a fresh attempt.
        let retry = single_flight(&FLIGHTS, "error-key-retry".to_string(), || {
            CALLS.fetch_add(1, Ordering::SeqCst);
            Ok(7u32)
        });
        assert_eq!(retry.unwrap(), 7);
        assert_eq!(CALLS.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn single_flight_turns_a_panicking_leader_into_an_error() {
        static FLIGHTS: OnceLock<Mutex<HashMap<String, Arc<Flight<u32>>>>> = OnceLock::new();
        let result = single_flight(&FLIGHTS, "panic-key".to_string(), || panic!("boom"));
        // `single_flight` catches the panic itself, so no unwind escapes.
        let error = result.unwrap_err().to_string();
        assert!(error.contains("session creation panicked"), "{error}");
        let next = single_flight(&FLIGHTS, "panic-key-2".to_string(), || Ok(3u32));
        assert_eq!(next.unwrap(), 3, "a panic must not poison later keys");
    }

    #[test]
    fn tensor_bytes_use_declared_dims_and_dtype() {
        let input = TensorMeta {
            name: "x".to_string(),
            dims: vec![1, 3, 320, 320],
            element_type: DT_FLOAT,
        };
        assert_eq!(tensor_bytes(&input.dims, DT_FLOAT), 3 * 320 * 320 * 4);
        assert_eq!(
            tensor_bytes(&[1, 1, 512, 512], DT_FLOAT16),
            512 * 512 * 2,
            "fp16 counts two bytes per element"
        );
        // A dynamic dimension (ONNX symbol) cannot be sized; it counts zero so
        // the estimate stays a lower bound.
        assert_eq!(tensor_bytes(&[1, 3, -1, -1], DT_FLOAT), 0);
        assert_eq!(tensor_bytes(&[1, 3, 320], 7), 0, "unknown dtype");
    }

    #[test]
    fn session_bytes_estimate_is_model_plus_known_tensors() {
        let input = TensorMeta {
            name: "x".to_string(),
            dims: vec![1, 3, 8, 8],
            element_type: DT_FLOAT,
        };
        let outputs = vec![TensorMeta {
            name: "y".to_string(),
            dims: vec![1, 1, 8, 8],
            element_type: DT_FLOAT,
        }];
        let estimate = session_bytes_estimate(10_000, &input, &outputs);
        assert_eq!(estimate, 10_000 + (3 * 64 + 64) * 4);
    }

    #[test]
    fn session_budget_defaults_to_512_mib() {
        // No override in the environment (CI does not set one); documented
        // default. An override cannot be tested here without racing other
        // tests that read the same variable.
        if std::env::var_os(SESSION_BUDGET_ENV).is_none() {
            assert_eq!(session_budget_bytes(), 512 * 1024 * 1024);
        }
    }

    #[test]
    fn registry_evicts_lru_not_everything() {
        // 150-byte budget: two 100-byte entries fit only after one is evicted.
        let mut registry = ByteLru::new(150, 64);
        let _ = registry.insert("a".to_string(), test_session(10), 100);
        let _ = registry.insert("b".to_string(), test_session(20), 100);
        assert!(registry.get("a").is_some(), "a is now the most recent entry");

        // No outside clones; the LRU entry "b" is released, "a" stays.
        let eviction = reclaim_sessions(&mut registry);
        assert_eq!(eviction.removed.len(), 1);
        assert_eq!(eviction.removed[0].0, "b");
        drop(eviction);
        assert!(registry.get("a").is_some(), "the just-used entry survives");
        assert!(registry.get("b").is_none(), "only the LRU entry was released");
        assert_eq!(registry.bytes(), 100);
    }

    /// A `Session` with no ORT handles, for registry accounting tests. The
    /// pointers are never dereferenced: these values are only counted and
    /// dropped, and `SessionInner::drop` is never reached because the test
    /// keeps the last clone alive until the end.
    fn test_session(model_len: u64) -> Session {
        Session {
            inner: Arc::new(SessionInner {
                api: ptr::null_mut(),
                ptr: ptr::null_mut(),
                provider: Provider::Cpu,
                key: key_for(model_len, "CPU", "opt=99"),
                fallback: Mutex::new(None),
                profile_active: AtomicBool::new(false),
                timing: Mutex::new(SessionTiming::default()),
            }),
            input: TensorMeta {
                name: "x".to_string(),
                dims: vec![1, 3, 1, 1],
                element_type: DT_FLOAT,
            },
            outputs: Vec::new(),
        }
    }

    #[test]
    fn registry_defers_eviction_while_a_caller_holds_the_session() {
        let mut registry = ByteLru::new(150, 64);

        let older = test_session(10);
        let _ = registry.insert("a".to_string(), older.clone(), 100);
        let _ = registry.insert("b".to_string(), test_session(20), 100);

        // The caller still holds a clone of "a" (strong_count 2), so the
        // in-use predicate must protect it even though it is the LRU entry.
        let eviction = reclaim_sessions(&mut registry);
        assert!(
            registry.get("a").is_some(),
            "an entry with a live caller clone must not be dropped"
        );
        assert!(registry.get("b").is_none(), "the unused entry is reclaimed");
        assert_eq!(eviction.pending, 0, "no pending marker needed: b was free");
        // A registered key that is currently being created is in flight and
        // must not be evicted either.
        assert!(!session_in_flight("a"), "no flight was ever created for a");
        drop(older);
    }

    #[test]
    #[ignore = "needs Capture One's ONNX models"]
    fn real_model_session_picks_a_provider() {
        let Some(runtime) = Runtime::get() else {
            println!("ONNX Runtime unavailable; cannot open a model");
            return;
        };
        for model in ["skinMasking.onnx", "subjectMasking.onnx"] {
            let Some(path) = crate::aimask::model_candidates(model)
                .into_iter()
                .find(|path| path.is_file())
            else {
                println!("{model} is not installed; nothing to open");
                continue;
            };
            let session = runtime.open_session(&path).unwrap_or_else(|err| panic!("opening {model}: {err:#}"));
            println!("{model} session provider: {}", session.provider().label());
        }
    }
}
