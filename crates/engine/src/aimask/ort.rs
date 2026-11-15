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

/// GPU providers in preference order, then CPU as the fallback every session
/// ends on. Kept pure so the order is testable without a GPU build.
fn providers_from(has_tensor_rt: bool, has_cuda: bool, has_direct_ml: bool) -> Vec<Provider> {
    let mut out = Vec::new();
    if has_tensor_rt {
        out.push(Provider::TensorRt);
    }
    if has_cuda {
        out.push(Provider::Cuda);
    }
    if has_direct_ml {
        out.push(Provider::DirectMl);
    }
    out.push(Provider::Cpu);
    out
}

/// `AURORA_AI_PROVIDER` pins the provider, for A/B measurement and for machines
/// where the automatic pick misbehaves. Names are case-insensitive; an
/// unrecognised name means automatic.
fn forced_provider() -> Option<Provider> {
    let raw = std::env::var("AURORA_AI_PROVIDER").ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "tensorrt" | "trt" => Some(Provider::TensorRt),
        "cuda" => Some(Provider::Cuda),
        "dml" | "directml" => Some(Provider::DirectMl),
        "cpu" => Some(Provider::Cpu),
        _ => None,
    }
}

/// Narrows a probed list to the forced provider. A provider this library does
/// not export is not a wish that can be granted: the probed order stays.
fn apply_override(probed: Vec<Provider>, forced: Option<Provider>) -> Vec<Provider> {
    let Some(forced) = forced else {
        return probed;
    };
    if !probed.contains(&forced) {
        return probed;
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
        let (library, handle) = unsafe { open_library()? };
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
        Ok(Self {
            api,
            env,
            allocator,
            memory_info,
            handle,
            library,
            version: version.to_string_lossy().into_owned(),
            api_version: used,
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

    /// Providers this library can really append. The C API exports one
    /// `OrtSessionOptionsAppendExecutionProvider_<name>` symbol per provider it
    /// was built with, so probing the symbols reflects the build, not a wish.
    pub(crate) fn available_providers(&self) -> Vec<Provider> {
        let has = |provider: Provider| {
            let name = match provider {
                Provider::TensorRt => "OrtSessionOptionsAppendExecutionProvider_TensorRT",
                Provider::Cuda => "OrtSessionOptionsAppendExecutionProvider_CUDA",
                Provider::DirectMl => "OrtSessionOptionsAppendExecutionProvider_DML",
                Provider::Cpu => return true,
            };
            !self.symbol(name).is_null()
        };
        let probed =
            providers_from(has(Provider::TensorRt), has(Provider::Cuda), has(Provider::DirectMl));
        let forced = forced_provider();
        let chosen = apply_override(probed.clone(), forced);
        if let Some(forced) = forced {
            if chosen != probed {
                log::info!("AURORA_AI_PROVIDER pins {}", forced.label());
            } else {
                log::warn!(
                    "AURORA_AI_PROVIDER asked for {}, which this ONNX Runtime does not export; \
                     keeping the automatic order",
                    forced.label()
                );
            }
        }
        chosen
    }

    unsafe fn append_provider(&self, options: *mut c_void, provider: Provider) -> Result<()> {
        if provider == Provider::Cpu {
            return Ok(());
        }
        let name = match provider {
            Provider::TensorRt => "OrtSessionOptionsAppendExecutionProvider_TensorRT",
            Provider::Cuda => "OrtSessionOptionsAppendExecutionProvider_CUDA",
            Provider::DirectMl => "OrtSessionOptionsAppendExecutionProvider_DML",
            Provider::Cpu => unreachable!(),
        };
        let pointer = self.symbol(name);
        if pointer.is_null() {
            bail!("{} is not exported by this ONNX Runtime", provider.label());
        }
        let append: unsafe extern "C" fn(*mut c_void, c_int) -> Status =
            unsafe { std::mem::transmute_copy(&pointer) };
        unsafe { check(append(options, 0), self.api) }
            .with_context(|| format!("enabling {}", provider.label()))
    }

    /// Opens a session on the first provider that works, falling back to CPU.
    pub(crate) fn open_session(&self, path: &Path) -> Result<Session> {
        let mut last = None;
        for provider in self.available_providers() {
            let cached = optimized_model_cache_path(path, provider, graph_optimization_level())
                .is_some_and(|path| path.is_file());
            match self.open_session_with(path, provider, cached) {
                Ok(session) => {
                    log::info!("ONNX Runtime using {} for {}", provider.label(), path.display());
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
                                last = Some(rebuild_err.context("rebuilding the optimized model cache"));
                            }
                        }
                    } else {
                        last = Some(err);
                    }
                    if provider != Provider::Cpu {
                        log::warn!(
                            "{} session failed for {}, falling back",
                            provider.label(),
                            path.display(),
                        );
                    }
                }
            }
        }
        Err(last.unwrap_or_else(|| anyhow!("no execution provider available")))
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

unsafe fn open_library() -> Result<(PathBuf, *mut c_void)> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(path) = std::env::var_os("AURORA_ORT_LIB") {
        candidates.push(PathBuf::from(path));
    }
    if cfg!(windows) {
        candidates.extend(crate::aimask::c1_roots().into_iter().map(|r| r.join("onnxruntime.dll")));
        candidates.push(PathBuf::from("onnxruntime.dll"));
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
    for path in candidates {
        #[cfg(target_os = "linux")]
        {
            if let Ok(name) = CString::new(path.as_os_str().to_string_lossy().as_bytes()) {
                let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
                if !handle.is_null() {
                    return Ok((path, handle));
                }
            }
        }
        #[cfg(windows)]
        {
            let wide: Vec<u16> = path
                .as_os_str()
                .to_string_lossy()
                .encode_utf16()
                .chain(std::iter::once(0))
                .collect();
            let handle = unsafe { LoadLibraryW(wide.as_ptr()) };
            if !handle.is_null() {
                return Ok((path, handle));
            }
        }
    }
    #[cfg(target_os = "linux")]
    let reason = unsafe { CStr::from_ptr(dlerror()) }.to_string_lossy().into_owned();
    #[cfg(windows)]
    let reason = "LoadLibraryW failed".to_string();
    Err(anyhow!("no ONNX Runtime library found ({reason})"))
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

    #[test]
    fn providers_prefer_gpu_and_always_end_with_cpu() {
        assert_eq!(providers_from(false, false, false), vec![Provider::Cpu]);
        assert_eq!(
            providers_from(true, true, true),
            vec![Provider::TensorRt, Provider::Cuda, Provider::DirectMl, Provider::Cpu]
        );
        assert_eq!(providers_from(false, true, false), vec![Provider::Cuda, Provider::Cpu]);
        assert_eq!(providers_from(false, false, true), vec![Provider::DirectMl, Provider::Cpu]);
    }

    #[test]
    fn override_pins_one_provider_and_keeps_cpu_as_fallback() {
        let probed = providers_from(false, false, true); // [DirectMl, Cpu]
        assert_eq!(apply_override(probed.clone(), None), probed);
        assert_eq!(
            apply_override(probed.clone(), Some(Provider::DirectMl)),
            vec![Provider::DirectMl, Provider::Cpu]
        );
        assert_eq!(apply_override(probed.clone(), Some(Provider::Cpu)), vec![Provider::Cpu]);
        // A provider this build cannot export must not be invented.
        assert_eq!(apply_override(probed.clone(), Some(Provider::Cuda)), probed);
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
