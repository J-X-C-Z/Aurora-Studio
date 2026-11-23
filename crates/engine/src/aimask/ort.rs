//! The small slice of the ONNX Runtime C API this engine needs, loaded at
//! runtime with `dlopen`.
//!
//! `OrtApi` is an ABI-frozen array of function pointers: every field is one
//! pointer, and the order is fixed per API version. The indices below come from
//! the 1.23.2 header (`ORT_API_VERSION 23`, package `libonnxruntime-dev`) and
//! were confirmed by opening both models and reading their tensor metadata.

use std::ffi::{CStr, CString};
use std::hash::{Hash, Hasher};
use std::os::raw::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::ptr;
use std::sync::{OnceLock, atomic::{AtomicBool, Ordering}};
use std::time::UNIX_EPOCH;

use anyhow::{Context, Result, anyhow, bail};

type Status = *mut c_void;
type ApiPtr = *const *const c_void;

const FLOAT: c_int = 1;
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
    Cpu,
}

impl Provider {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Provider::TensorRt => "TensorRT",
            Provider::Cuda => "CUDA",
            Provider::DirectMl => "DirectML",
            Provider::Cpu => "CPU",
        }
    }
}

fn provider_symbol(provider: Provider) -> Option<&'static str> {
    match provider {
        Provider::TensorRt => Some("OrtSessionOptionsAppendExecutionProvider_Tensorrt"),
        Provider::Cuda => Some("OrtSessionOptionsAppendExecutionProvider_CUDA"),
        Provider::DirectMl => Some("OrtSessionOptionsAppendExecutionProvider_DML"),
        Provider::Cpu => None,
    }
}

/// Symbol spellings to try when appending a provider. ORT <= 1.22 exports
/// `_TensorRT`, ORT >= 1.23 exports `_Tensorrt`; `GetProcAddress` is case
/// sensitive, so both are probed and the PE export table is scanned
/// case-insensitively on Windows.
fn append_symbol_candidates(provider: Provider) -> &'static [&'static str] {
    match provider {
        Provider::TensorRt => &[
            "OrtSessionOptionsAppendExecutionProvider_Tensorrt",
            "OrtSessionOptionsAppendExecutionProvider_TensorRT",
        ],
        Provider::Cuda => &["OrtSessionOptionsAppendExecutionProvider_CUDA"],
        Provider::DirectMl => &["OrtSessionOptionsAppendExecutionProvider_DML"],
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

fn optimized_model_cache_path(path: &Path, provider: Provider, optimization: c_int) -> Option<PathBuf> {
    let dir = std::env::var_os("AURORA_ORT_CACHE_DIR")?;
    let metadata = std::fs::metadata(path).ok()?;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    let mut hash = std::collections::hash_map::DefaultHasher::new();
    path.to_string_lossy().hash(&mut hash);
    metadata.len().hash(&mut hash);
    modified.hash(&mut hash);
    provider.label().hash(&mut hash);
    optimization.hash(&mut hash);
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
    Some(PathBuf::from(dir).join(format!("{stem}-{:016x}.onnx", hash.finish())))
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
        #[cfg(target_os = "linux")]
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

    fn symbol(&self, name: &str) -> *mut c_void {
        let Ok(name) = CString::new(name) else {
            return ptr::null_mut();
        };
        #[cfg(target_os = "linux")]
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
    pub(crate) fn available_providers(&self) -> Vec<Provider> {
        let has_trt = !self.provider_append_symbol(Provider::TensorRt).is_null();
        let has_cuda = !self.provider_append_symbol(Provider::Cuda).is_null();
        let has_dml = !self.provider_append_symbol(Provider::DirectMl).is_null();
        let base = providers_from(has_cuda, has_dml);
        let mut exported = base.clone();
        if has_trt {
            exported.insert(0, Provider::TensorRt);
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
        log::info!(
            "ONNX Runtime providers: exported {:?}, selected {:?}",
            provider_labels(&exported),
            provider_labels(&chosen)
        );
        for provider in [
            Provider::TensorRt,
            Provider::Cuda,
            Provider::DirectMl,
            Provider::Cpu,
        ] {
            let state = match provider {
                Provider::Cpu => "enabled (fallback)",
                Provider::TensorRt if !has_trt => "not exported by this ONNX Runtime",
                Provider::TensorRt if chosen.contains(&Provider::TensorRt) => match trt_reason {
                    "cache" => "enabled first (engine cache present)",
                    _ => "enabled first (explicit request)",
                },
                Provider::TensorRt => {
                    "skipped (no engine cache; set AURORA_TRT_CACHE_DIR or \
                     AURORA_ORT_PROVIDERS=tensorrt,...)"
                }
                _ if !exported.contains(&provider) => "not exported by this ONNX Runtime",
                _ if chosen.contains(&provider) => "enabled",
                _ => "available but not selected",
            };
            log::info!("provider {}: {}", provider.label(), state);
        }
        chosen
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
    pub(crate) fn open_session_on(&self, path: &Path, providers: &[Provider]) -> Result<Session> {
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
            let cached = optimized_model_cache_path(path, provider, graph_optimization_level())
                .is_some_and(|path| path.is_file());
            match self.open_session_with(path, provider, cached) {
                Ok(session) => {
                    log::info!(
                        "ONNX Runtime opened {} with requested provider {}; \
                         unsupported nodes may still run on CPU",
                        path.display(),
                        provider.label()
                    );
                    return Ok(session);
                }
                Err(err) => {
                    if cached {
                        log::warn!(
                            "{} cached model failed for {}, rebuilding from source: {err:#}",
                            provider.label(),
                            path.display()
                        );
                        match self.open_session_with(path, provider, false) {
                            Ok(session) => return Ok(session),
                            Err(rebuild_err) => {
                                let reason = format!(
                                    "rebuilding the optimized model cache: {rebuild_err:#}"
                                );
                                log::warn!(
                                    "{} rebuild failed for {}: {reason}",
                                    provider.label(),
                                    path.display()
                                );
                                failures.push(format!("{}: {reason}", provider.label()));
                            }
                        }
                    } else {
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
        self.open_session_with(path, Provider::Cpu, false)
            .with_context(|| format!("opening CPU model {}", path.display()))
    }

    fn open_session_with(&self, path: &Path, provider: Provider, use_cached_model: bool) -> Result<Session> {
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
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(8) as c_int;
        // The cache already contains ORT's optimized graph; re-running transforms only adds latency.
        let optimization = if use_cached_model { 0 } else { graph_optimization_level() };
        let result = unsafe {
            let result = check(set_threads(options, threads), api).and_then(|()| {
                check(set_optimization(options, optimization), api)
            });
            result
        };
        let cache_path = optimized_model_cache_path(path, provider, graph_optimization_level());
        let result = result.and_then(|()| {
            let Some(cache_path) = cache_path.as_ref().filter(|_| !use_cached_model) else {
                return Ok(());
            };
            if let Some(dir) = cache_path.parent() {
                std::fs::create_dir_all(dir)?;
            }
            let cache_path = ort_path(cache_path);
            unsafe { check(set_optimized_path(options, cache_path.as_ptr() as *const c_void), api) }
                .context("setting the optimized model cache path")
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
        let session = Session {
            api,
            ptr: session,
            provider,
            profile_active: AtomicBool::new(profile_prefix.is_some()),
            input: TensorMeta { name: String::new(), dims: Vec::new(), element_type: 0 },
            outputs: Vec::new(),
        };
        Ok(unsafe { session.with_metadata() }?)
    }

    pub(crate) fn end_profiling(&self, session: &Session) -> Result<Option<PathBuf>> {
        if !session.profile_active.swap(false, Ordering::AcqRel) {
            return Ok(None);
        }
        let end: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut *mut c_char) -> Status =
            unsafe { function(self.api, SESSION_END_PROFILING) };
        let free: unsafe extern "C" fn(*mut c_void, *mut c_void) -> Status =
            unsafe { function(self.api, ALLOCATOR_FREE) };
        let mut raw = ptr::null_mut();
        unsafe { check(end(session.ptr, self.allocator, &mut raw), self.api) }
            .context("ending ONNX Runtime profiling")?;
        if raw.is_null() {
            return Ok(None);
        }
        let path = unsafe { CStr::from_ptr(raw) }.to_string_lossy().into_owned();
        unsafe { check(free(self.allocator, raw.cast()), self.api) }
            .context("freeing the ONNX Runtime profile path")?;
        Ok(Some(PathBuf::from(path)))
    }

    pub(crate) fn run(&self, session: &Session, input: &mut [f32], shape: &[i64]) -> Result<Vec<Vec<f32>>> {
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

        let mut value = ptr::null_mut();
        unsafe {
            check(
                create_tensor(
                    self.memory_info,
                    input.as_mut_ptr() as *mut c_void,
                    std::mem::size_of_val(input),
                    shape.as_ptr(),
                    shape.len(),
                    FLOAT,
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
        let status = unsafe {
            run(
                session.ptr,
                ptr::null(),
                input_names.as_ptr(),
                input_values.as_ptr(),
                1,
                output_ptrs.as_ptr(),
                output_ptrs.len(),
                outputs.as_mut_ptr(),
            )
        };
        let result = unsafe { check(status, api) };
        unsafe { release_value(value) };
        result.context("running the model")?;

        let mut planes = Vec::with_capacity(outputs.len());
        for output in outputs {
            let plane = unsafe { read_tensor(api, output) };
            unsafe { release_value(output) };
            planes.push(plane?);
        }
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

/// Every path is explicit. A bare `onnxruntime.dll` is never attempted because
/// `C:\Windows\System32\onnxruntime.dll` (1.17, API 17) would win the search and
/// is too old for this module (`MIN_ORT_API_VERSION` is 18).
fn library_candidates() -> Vec<PathBuf> {
    library_candidates_from(|key| std::env::var_os(key), &project_roots())
}

fn library_candidates_from(
    get: impl Fn(&str) -> Option<std::ffi::OsString>,
    roots: &[PathBuf],
) -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = get("AURORA_ORT_LIB") {
        candidates.push(PathBuf::from(path));
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
        candidates.extend(
            ["libonnxruntime.so.1.23", "libonnxruntime.so.1", "libonnxruntime.so"]
                .into_iter()
                .map(PathBuf::from),
        );
        candidates.push(PathBuf::from(
            "/usr/lib/x86_64-linux-gnu/libonnxruntime.so.1.23",
        ));
    }
    let mut seen = std::collections::HashSet::new();
    candidates.retain(|path| seen.insert(path.clone()));
    candidates
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
    bail!(
        "ONNX Runtime not found; set AURORA_ORT_LIB to an absolute onnxruntime.dll \
         (looked in {})",
        candidates
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
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

#[cfg(target_os = "linux")]
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

unsafe fn read_tensor(api: ApiPtr, value: *const c_void) -> Result<Vec<f32>> {
    let type_and_shape: unsafe extern "C" fn(*const c_void, *mut *mut c_void) -> Status =
        unsafe { function(api, GET_TENSOR_TYPE_AND_SHAPE) };
    let element_count: unsafe extern "C" fn(*const c_void, *mut usize) -> Status =
        unsafe { function(api, GET_TENSOR_SHAPE_ELEMENT_COUNT) };
    let data: unsafe extern "C" fn(*const c_void, *mut *mut c_void) -> Status =
        unsafe { function(api, GET_TENSOR_MUTABLE_DATA) };
    let release_info: unsafe extern "C" fn(*mut c_void) =
        unsafe { function(api, RELEASE_TENSOR_TYPE_AND_SHAPE) };
    let mut info = ptr::null_mut();
    unsafe { check(type_and_shape(value, &mut info), api)? };
    let mut count = 0usize;
    let status = unsafe { element_count(info, &mut count) };
    unsafe { release_info(info) };
    unsafe { check(status, api)? };
    let mut pointer = ptr::null_mut();
    unsafe { check(data(value, &mut pointer), api)? };
    if pointer.is_null() {
        bail!("model output has no data");
    }
    Ok(unsafe { std::slice::from_raw_parts(pointer as *const f32, count) }.to_vec())
}

#[derive(Clone, Debug)]
pub(crate) struct TensorMeta {
    pub(crate) name: String,
    pub(crate) dims: Vec<i64>,
    pub(crate) element_type: c_int,
}

pub(crate) struct Session {
    api: ApiPtr,
    ptr: *mut c_void,
    provider: Provider,
    profile_active: AtomicBool,
    pub(crate) input: TensorMeta,
    pub(crate) outputs: Vec<TensorMeta>,
}

unsafe impl Send for Session {}
unsafe impl Sync for Session {}

impl Session {
    /// The provider the session was opened on. Nodes the provider does not
    /// support can still run on CPU inside the same session.
    pub(crate) fn provider(&self) -> Provider {
        self.provider
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

    unsafe fn with_metadata(self) -> Result<Self> {
        let api = self.api;
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
        unsafe { check(input_count(self.ptr, &mut inputs), api)? };
        unsafe { check(output_count(self.ptr, &mut outputs), api)? };
        if inputs != 1 || outputs == 0 {
            bail!("expected one input and at least one output, got {inputs}/{outputs}");
        }

        let session = self.ptr;
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
        let mut session = self;
        session.input = input_meta;
        session.outputs = output_meta;
        Ok(session)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let release: unsafe extern "C" fn(*mut c_void) = unsafe { function(self.api, RELEASE_SESSION) };
        unsafe { release(self.ptr) };
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
