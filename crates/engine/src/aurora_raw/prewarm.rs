//! Startup device warmup for the RAW chain and the native inference engine.
//!
//! The process CUDA domain (`aurora_portrait::cuda`) keeps one module per PTX
//! *text*: the first `Nvidia::open_ptx(image)` for a given image pays the JIT
//! (`cuModuleLoadDataEx`, PTX → SASS for this driver/GPU), every later open of
//! the same text is a cache hit. Until this module existed that first load
//! happened synchronously inside the first RAW decode, so opening a folder and
//! clicking the first RAW paid `~214 ms` before any pixel work started.
//!
//! [`spawn_background`] runs the warmers from a below-normal priority thread
//! while the UI paints its first frame. Nothing here decodes, touches a cache
//! entry or changes a pixel: it only moves an unavoidable JIT off the click
//! path, so a warmup can never alter output.
//!
//! ## Which device is warmed
//!
//! The warmup obeys `AURORA_GPU_BACKEND` (`auto|cuda|metal|dx12|opencl|cpu`),
//! the same contract the native inference sessions use:
//!
//! - `cuda` runs the embedded-PTX warmer and nothing else;
//! - `dx12`, `metal`, `opencl` and `cpu` never reach the CUDA warmer at all, and
//!   the report proves it with the driver's own module-load counter;
//! - `auto` probes the real host ([`crate::aurora_raw::gpu_probe`]) in routing
//!   order — the platform `cfg` never decides that a device exists.
//!
//! Warmers are registered per backend ([`register_device_warmer`]), so the
//! Metal and DX12 lanes add their pipeline/PSO warmup without touching this
//! file. A backend with no registered warmer is reported as such: an absent
//! DX12 PSO cache is never logged as a warm device.
//!
//! `AURORA_RAW_PREWARM=0` (`false`/`off`) skips the whole warmup, probes
//! included. A host without an NVIDIA device keeps working: the decode path's
//! existing CPU/OpenCL fallback is untouched.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use crate::aurora_raw::gpu_probe::{self, BackendKind, BackendRequest, Host};

/// What one warmer did. Never upgraded: `Failed` stays `Failed`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceOutcome {
    /// Modules/programs/pipelines are resident. `loaded` were new in this call,
    /// `cached` were already resident (the driver may have served them from its
    /// on-disk module cache, which the counter cannot distinguish).
    Warmed { loaded: usize, cached: usize },
    /// The warmer ran and the device refused. The decode path keeps its own
    /// CPU/OpenCL fallback.
    Failed(String),
}

impl DeviceOutcome {
    /// Whether anything is now resident because of this warmup (or an earlier
    /// one in this process).
    pub fn is_resident(&self) -> bool {
        match self {
            Self::Warmed { loaded, cached } => loaded + cached > 0,
            Self::Failed(_) => false,
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Warmed { loaded, cached } => {
                format!("{} resident ({loaded} new, {cached} cached)", loaded + cached)
            }
            Self::Failed(reason) => format!("failed: {reason}"),
        }
    }
}

/// One registered warmup body.
pub type WarmFn = Arc<dyn Fn() -> DeviceOutcome + Send + Sync>;

/// A named warmup for one backend.
#[derive(Clone)]
pub struct DeviceWarmer {
    pub backend: BackendKind,
    pub name: &'static str,
    warmer: WarmFn,
}

impl DeviceWarmer {
    fn run(&self) -> DeviceOutcome {
        (self.warmer)()
    }
}

/// One report line per warmer that ran.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevicePrewarm {
    pub backend: BackendKind,
    pub name: &'static str,
    pub outcome: DeviceOutcome,
}

/// The registry every host shares. A warmer is addressed by (backend, name), so
/// registering the same pair twice replaces the body instead of running both.
pub struct WarmerRegistry {
    entries: Mutex<Vec<DeviceWarmer>>,
}

impl WarmerRegistry {
    /// An empty registry, for tests and for a host that starts from scratch.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(Vec::new()),
        }
    }

    /// The shipping registry: the embedded-PTX CUDA warmer.
    pub fn with_builtins() -> Self {
        let registry = Self::new();
        let registered = registry.register(BackendKind::Cuda, "cuda-ptx", Arc::new(warm_cuda_ptx));
        debug_assert!(registered, "the built-in CUDA warmer must register");
        #[cfg(windows)]
        registry.register(BackendKind::Dx12, "dx12-inference-psos", Arc::new(|| {
            match aurora_portrait::native_infer::dx12_session::prewarm_common_pipelines() {
                Ok((loaded,cached)) => DeviceOutcome::Warmed{loaded,cached},
                Err(reason) => DeviceOutcome::Failed(reason),
            }
        }));
        registry
    }

    /// Adds or replaces one warmer. Returns `false` only when the name is empty
    /// (a nameless warmer cannot be identified in the report).
    pub fn register(&self, backend: BackendKind, name: &'static str, warmer: WarmFn) -> bool {
        if name.is_empty() {
            return false;
        }
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(existing) = entries.iter_mut().find(|entry| entry.backend == backend && entry.name == name) {
            existing.warmer = warmer;
        } else {
            entries.push(DeviceWarmer { backend, name, warmer });
        }
        true
    }

    /// The warmers for one backend, in registration order.
    pub fn warmers_for(&self, backend: BackendKind) -> Vec<DeviceWarmer> {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|entry| entry.backend == backend)
            .cloned()
            .collect()
    }

    /// `(backend, name)` of everything registered, for the startup log and the
    /// acceptance tests.
    pub fn registered(&self) -> Vec<(BackendKind, &'static str)> {
        self.entries
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .map(|entry| (entry.backend, entry.name))
            .collect()
    }
}

impl Default for WarmerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

fn global_registry() -> &'static WarmerRegistry {
    static REGISTRY: OnceLock<WarmerRegistry> = OnceLock::new();
    REGISTRY.get_or_init(WarmerRegistry::with_builtins)
}

/// The process registry, so a lane can add its backend warmup at startup.
pub fn registry() -> &'static WarmerRegistry {
    global_registry()
}

/// Registers a warmup for `backend` in the process registry. Returns `false`
/// for an empty name.
///
/// The Metal and DX12 lanes call this with their pipeline/PSO warmers; nothing
/// else has to change, and the `AURORA_GPU_BACKEND` routing already sends a
/// matching request here.
pub fn register_device_warmer(backend: BackendKind, name: &'static str, warmer: WarmFn) -> bool {
    global_registry().register(backend, name, warmer)
}

/// What the warmup did. Reported for the startup log and the acceptance tests.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PrewarmReport {
    /// The `AURORA_GPU_BACKEND` spelling that was used (`auto` when unset).
    pub request: &'static str,
    /// The resolved backend, `None` when the request was invalid or warmup was
    /// switched off (nothing was resolved and nothing was touched).
    pub backend: Option<BackendKind>,
    /// What an `auto` probe answered with, when probes were consulted.
    pub evidence: Option<String>,
    /// Backends an `auto` walk rejected, in order, with their reasons.
    pub rejected: Vec<(BackendKind, String)>,
    /// One line per warmer that ran.
    pub devices: Vec<DevicePrewarm>,
    /// Module loads the CUDA driver recorded during this call. Zero whenever
    /// the resolved backend is not CUDA — that is the proof the switch holds.
    pub cuda_module_loads: usize,
    /// Wall time of this call.
    pub elapsed_ms: u64,
    /// Why nothing ran, when nothing ran. `None` when at least one warmer was
    /// dispatched (which is not the same as "it succeeded").
    pub skipped: Option<String>,
}

impl PrewarmReport {
    /// Whether at least one device is resident now.
    pub fn is_warm(&self) -> bool {
        self.devices.iter().any(|device| device.outcome.is_resident())
    }

    /// The resolved backend, or the request label when resolution did not run.
    pub fn backend_label(&self) -> &'static str {
        self.backend.map_or(self.request, BackendKind::label)
    }

    /// One-line summary for the startup log.
    pub fn summary(&self) -> String {
        if let Some(reason) = &self.skipped {
            return format!(
                "device prewarm skipped ({reason}) [backend={} request={}] in {} ms",
                self.backend_label(),
                self.request,
                self.elapsed_ms
            );
        }
        let devices = self
            .devices
            .iter()
            .map(|device| format!("{}={}", device.name, device.outcome.describe()))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "device prewarm [backend={} request={}]: {devices}; cuda_module_loads={} in {} ms",
            self.backend_label(),
            self.request,
            self.cuda_module_loads,
            self.elapsed_ms
        )
    }
}

/// Returns `false` when `AURORA_RAW_PREWARM` is explicitly disabled
/// (`0`, `false`, or `off`). Unset means enabled.
pub fn prewarm_enabled() -> bool {
    parse_enabled_env(std::env::var("AURORA_RAW_PREWARM").ok().as_deref())
}

fn parse_enabled_env(value: Option<&str>) -> bool {
    match value {
        Some(value) => {
            let trimmed = value.trim();
            !(trimmed == "0"
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off"))
        }
        None => true,
    }
}

/// The PTX images the RAW main path loads, in first-use order, followed by the
/// native inference engine's four modules.
///
/// The Z6III reconstruction comes first because it is the profile the raw
/// first-load baseline was measured on; the generic Bayer / X-Trans demosaic
/// and the linear post stage follow. The inference modules come last: they are
/// only reached by an AI/persona mask, not by opening a RAW, so the RAW chain
/// must never wait behind them. The list is empty on targets where the Windows
/// CUDA modules are not compiled.
pub fn ptx_images() -> Vec<(&'static str, &'static str)> {
    #[cfg(windows)]
    {
        vec![
            ("z6iii_raw", crate::aurora_raw::z6iii::raw_ptx_image()),
            ("raw_demosaic", crate::aurora_raw::demosaic_cuda::demosaic_ptx_image()),
            (
                "raw_linear_post",
                crate::aurora_raw::linear_post_cuda::linear_post_ptx_image(),
            ),
            ("infer_conv_gemm", aurora_portrait::native_infer::ops_conv_gemm::PTX),
            ("infer_elem_norm", aurora_portrait::native_infer::ops_elem_norm::PTX),
            ("infer_spatial_tensor", aurora_portrait::native_infer::ops_spatial_tensor::PTX),
            ("infer_quant_int8", aurora_portrait::native_infer::ops_quant::QUANT_INT8_PTX),
        ]
    }
    #[cfg(not(windows))]
    {
        Vec::new()
    }
}

/// The built-in CUDA warmer: JIT-loads every embedded PTX image once.
#[cfg(windows)]
fn warm_cuda_ptx() -> DeviceOutcome {
    use aurora_portrait::cuda::{driver_metrics, CudaError, Nvidia};

    let mut loaded = 0;
    let mut cached = 0;
    let mut first_error = None;
    for (name, image) in ptx_images() {
        let before = driver_metrics().ptx_module_load_calls;
        match Nvidia::open_ptx(image) {
            Ok(_) => {
                if driver_metrics().ptx_module_load_calls > before {
                    loaded += 1;
                    log::debug!("raw PTX prewarm: {name} new module load observed");
                } else {
                    cached += 1;
                    log::debug!("raw PTX prewarm: {name} already resident");
                }
            }
            Err(error) => {
                log::debug!("raw PTX prewarm: {name} unavailable ({error:?})");
                first_error.get_or_insert(error);
            }
        }
    }
    if loaded + cached == 0 {
        return DeviceOutcome::Failed(match first_error {
            Some(CudaError::Unavailable(what)) => format!("no CUDA device ({what})"),
            Some(other) => format!("CUDA module load failed: {other:?}"),
            None => "empty PTX plan".to_string(),
        });
    }
    DeviceOutcome::Warmed { loaded, cached }
}

/// Everywhere else the Windows CUDA modules are not compiled, so there is
/// nothing to JIT and the decode path keeps its own fallback untouched.
#[cfg(not(windows))]
fn warm_cuda_ptx() -> DeviceOutcome {
    DeviceOutcome::Failed("no embedded CUDA PTX on this target".to_string())
}

/// CUDA module loads the driver has recorded so far. The delta across a warmup
/// is what proves a non-CUDA backend never opened a module.
fn cuda_module_loads() -> usize {
    #[cfg(windows)]
    {
        aurora_portrait::cuda::driver_metrics().ptx_module_load_calls
    }
    #[cfg(not(windows))]
    {
        0
    }
}

/// Runs the warmup for the process's requested backend.
pub fn prewarm_devices() -> PrewarmReport {
    prewarm_with(
        registry(),
        BackendRequest::from_env(),
        Host::current(),
        gpu_probe::probe,
        prewarm_enabled(),
    )
}

/// [`prewarm_devices`] with an injectable registry, request, host and probe.
///
/// This is the whole decision path: an invalid request or a disabled switch
/// returns before anything is probed, an explicit backend is dispatched without
/// a probe, and only `auto` walks the routing table. Warmers are selected by the
/// resolved backend, so a CUDA body is unreachable from any other request.
pub fn prewarm_with(
    registry: &WarmerRegistry,
    request: Result<BackendRequest, String>,
    host: Host,
    probe: impl Fn(BackendKind) -> Result<String, String>,
    enabled: bool,
) -> PrewarmReport {
    let started = Instant::now();
    let mut report = PrewarmReport::default();
    if !enabled {
        report.skipped = Some("AURORA_RAW_PREWARM=0".to_string());
        report.elapsed_ms = started.elapsed().as_millis() as u64;
        return report;
    }
    let request = match request {
        Ok(request) => request,
        Err(error) => {
            report.request = "invalid";
            report.skipped = Some(error);
            report.elapsed_ms = started.elapsed().as_millis() as u64;
            return report;
        }
    };
    report.request = request.label();
    let resolved = match gpu_probe::resolve_with(request, host, probe) {
        Ok(resolved) => resolved,
        Err(error) => {
            report.skipped = Some(error);
            report.elapsed_ms = started.elapsed().as_millis() as u64;
            return report;
        }
    };
    report.backend = Some(resolved.kind);
    report.evidence = resolved.evidence;
    report.rejected = resolved.rejected;

    let warmers = registry.warmers_for(resolved.kind);
    if warmers.is_empty() {
        report.skipped = Some(match resolved.kind {
            BackendKind::Cpu => "cpu backend: no device state to warm".to_string(),
            other => format!(
                "no warmer registered for backend {other}; that device is left cold (nothing was warmed)"
            ),
        });
        report.elapsed_ms = started.elapsed().as_millis() as u64;
        return report;
    }

    let loads_before = cuda_module_loads();
    for warmer in warmers {
        let outcome = warmer.run();
        report.devices.push(DevicePrewarm {
            backend: warmer.backend,
            name: warmer.name,
            outcome,
        });
    }
    report.cuda_module_loads = cuda_module_loads().saturating_sub(loads_before);
    if resolved.kind != BackendKind::Cuda && report.cuda_module_loads > 0 {
        // Structural guarantee broken: the CUDA body is only registered for
        // CUDA. Report it loudly instead of shipping a silent violation.
        log::error!(
            "prewarm for backend {} loaded {} CUDA module(s); the backend switch is broken",
            resolved.kind,
            report.cuda_module_loads
        );
    }
    report.elapsed_ms = started.elapsed().as_millis() as u64;
    report
}

static SPAWNED: AtomicBool = AtomicBool::new(false);
static SPAWN_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// How many times a host asked for the warmup. The process latch means only the
/// first request starts a thread; this counter is how the acceptance test proves
/// the real registration point (`FileEngine::open`) actually asks.
pub fn spawn_requests() -> usize {
    SPAWN_CALLS.load(Ordering::Relaxed)
}

/// Starts [`prewarm_devices`] on a below-normal priority background thread.
///
/// Returns `false` when a warmup already started in this process or the feature
/// is disabled by env. Call it once during application startup, before the
/// first RAW is opened.
pub fn spawn_background() -> bool {
    SPAWN_CALLS.fetch_add(1, Ordering::Relaxed);
    spawn_background_with(prewarm_devices)
}

/// [`spawn_background`] with an injectable body. The acceptance test uses it to
/// observe that the work really runs off the calling thread, without needing a
/// GPU.
pub fn spawn_background_with(body: impl FnOnce() -> PrewarmReport + Send + 'static) -> bool {
    if SPAWNED.swap(true, Ordering::AcqRel) {
        return false;
    }
    if !prewarm_enabled() {
        log::debug!("device prewarm disabled by AURORA_RAW_PREWARM");
        SPAWNED.store(false, Ordering::Release);
        return false;
    }
    let spawned = std::thread::Builder::new()
        .name("aurora-raw-prewarm".into())
        .spawn(move || {
            lower_current_thread_priority();
            let report = body();
            if report.is_warm() {
                log::info!("{}", report.summary());
            } else {
                // Not an error: on a CPU-only host, or before the DX12/Metal
                // lanes register their warmers, "nothing resident" is the truth.
                log::debug!("{}", report.summary());
            }
        });
    match spawned {
        Ok(_) => true,
        Err(error) => {
            log::warn!("device prewarm thread failed to start: {error}");
            SPAWNED.store(false, Ordering::Release);
            false
        }
    }
}

#[cfg(windows)]
pub(crate) fn lower_current_thread_priority() {
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
pub(crate) fn lower_current_thread_priority() {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    fn counting_warmer(counter: &Arc<AtomicUsize>, outcome: DeviceOutcome) -> WarmFn {
        let counter = Arc::clone(counter);
        Arc::new(move || {
            counter.fetch_add(1, Ordering::SeqCst);
            outcome.clone()
        })
    }

    fn never_probed() -> impl Fn(BackendKind) -> Result<String, String> {
        |kind| panic!("probe must not run for {kind}")
    }

    #[test]
    fn env_toggle_parses_disable_values() {
        assert!(parse_enabled_env(None));
        assert!(parse_enabled_env(Some("1")));
        assert!(parse_enabled_env(Some("true")));
        assert!(!parse_enabled_env(Some("0")));
        assert!(!parse_enabled_env(Some(" 0 ")));
        assert!(!parse_enabled_env(Some("FALSE")));
        assert!(!parse_enabled_env(Some("off")));
    }

    /// The plan must hand out the *same* text the decode path loads, otherwise
    /// the warmup JITs a duplicate module and buys nothing. `open_ptx` caches by
    /// text, so equality of the returned `&'static str` content is the contract.
    #[cfg(windows)]
    #[test]
    fn plan_images_are_ptx_and_distinct() {
        let plan = ptx_images();
        assert_eq!(plan.len(), 7, "three RAW and four native inference modules");
        let mut seen = Vec::new();
        for (name, image) in plan {
            assert!(!image.is_empty(), "{name}: empty PTX image");
            assert!(
                image.contains(".version") && image.contains(".target"),
                "{name}: not a PTX text image"
            );
            assert!(!seen.contains(&image), "{name}: duplicate image text");
            seen.push(image);
        }
    }

    /// `AURORA_GPU_BACKEND=cpu` must not reach the CUDA warmer: the switch is
    /// the whole point of asking for the CPU path.
    #[test]
    fn cpu_request_never_runs_the_cuda_warmer() {
        let loads_before = cuda_module_loads();
        let report = prewarm_with(
            &WarmerRegistry::with_builtins(),
            Ok(BackendRequest::Only(BackendKind::Cpu)),
            Host::Windows,
            never_probed(),
            true,
        );
        assert_eq!(report.backend, Some(BackendKind::Cpu));
        assert!(report.devices.is_empty(), "no device warmer may run for cpu: {report:?}");
        assert_eq!(report.cuda_module_loads, 0);
        assert_eq!(cuda_module_loads(), loads_before, "cpu request opened a CUDA module");
        assert_eq!(
            report.skipped.as_deref(),
            Some("cpu backend: no device state to warm")
        );
        assert!(!report.is_warm());
    }

    /// An explicit dx12 request dispatches registered dx12 warmers and nothing
    /// else. This is the extension point the DX12 lane fills in.
    #[test]
    fn explicit_dx12_request_dispatches_only_dx12_warmers() {
        let registry = WarmerRegistry::new();
        let runs = Arc::new(AtomicUsize::new(0));
        assert!(registry.register(
            BackendKind::Dx12,
            "dx12-fake-psos",
            counting_warmer(&runs, DeviceOutcome::Warmed { loaded: 3, cached: 0 }),
        ));
        let loads_before = cuda_module_loads();
        let report = prewarm_with(
            &registry,
            Ok(BackendRequest::Only(BackendKind::Dx12)),
            Host::Windows,
            never_probed(),
            true,
        );
        assert_eq!(report.backend, Some(BackendKind::Dx12));
        assert_eq!(report.devices.len(), 1, "{report:?}");
        assert_eq!(report.devices[0].name, "dx12-fake-psos");
        assert_eq!(report.devices[0].backend, BackendKind::Dx12);
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert_eq!(report.cuda_module_loads, 0);
        assert_eq!(cuda_module_loads(), loads_before, "dx12 request opened a CUDA module");
        assert!(report.is_warm());
        assert!(report.skipped.is_none());
    }

    /// Metal, OpenCL and DX12 have no warmer in this lane yet. That must be
    /// reported as a cold device, never as a warmed one.
    #[test]
    fn a_backend_without_a_warmer_is_reported_cold() {
        for backend in [BackendKind::Metal, BackendKind::Dx12, BackendKind::OpenCl] {
            let report = prewarm_with(
                &WarmerRegistry::new(),
                Ok(BackendRequest::Only(backend)),
                Host::Windows,
                never_probed(),
                true,
            );
            assert_eq!(report.backend, Some(backend));
            assert!(report.devices.is_empty(), "{backend}: {report:?}");
            assert!(!report.is_warm(), "{backend}: a cold device is not warm");
            let skipped = report.skipped.expect("a skipped warmup names its reason");
            assert!(
                skipped.contains("no warmer registered for backend"),
                "{backend}: {skipped}"
            );
            assert_eq!(report.cuda_module_loads, 0, "{backend} touched CUDA");
        }
    }

    /// `auto` must take the probe's answer, not the platform's: a failed CUDA
    /// probe followed by an answering DX12 probe resolves DX12 on Windows.
    #[test]
    fn auto_follows_the_probe_across_a_failed_cuda_probe() {
        let registry = WarmerRegistry::new();
        let runs = Arc::new(AtomicUsize::new(0));
        registry.register(
            BackendKind::Dx12,
            "dx12-fake-psos",
            counting_warmer(&runs, DeviceOutcome::Warmed { loaded: 1, cached: 0 }),
        );
        let report = prewarm_with(
            &registry,
            Ok(BackendRequest::Auto),
            Host::Windows,
            |kind| match kind {
                BackendKind::Cuda => Err("nvcuda.dll: not found".to_string()),
                BackendKind::Dx12 => Ok("2 DXGI adapter(s)".to_string()),
                other => panic!("auto must stop at dx12, asked {other}"),
            },
            true,
        );
        assert_eq!(report.backend, Some(BackendKind::Dx12));
        assert_eq!(report.evidence.as_deref(), Some("2 DXGI adapter(s)"));
        assert_eq!(report.rejected.len(), 1);
        assert_eq!(report.rejected[0].0, BackendKind::Cuda);
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert_eq!(report.cuda_module_loads, 0);
    }

    /// A typo must not become `auto` (which would probe CUDA) and must not warm
    /// anything at all.
    #[test]
    fn an_invalid_request_touches_nothing() {
        let loads_before = cuda_module_loads();
        let report = prewarm_with(
            &WarmerRegistry::with_builtins(),
            BackendRequest::parse("gpu"),
            Host::Windows,
            never_probed(),
            true,
        );
        assert_eq!(report.request, "invalid");
        assert_eq!(report.backend, None);
        assert!(report.devices.is_empty());
        assert!(report.skipped.unwrap().contains("invalid AURORA_GPU_BACKEND"));
        assert_eq!(cuda_module_loads(), loads_before);
    }

    /// `AURORA_RAW_PREWARM=0` returns before the probe: the switch is the last
    /// word, so a broken driver cannot make a disabled warmup slow.
    #[test]
    fn the_switch_returns_before_any_probe() {
        let report = prewarm_with(
            &WarmerRegistry::with_builtins(),
            Ok(BackendRequest::Auto),
            Host::Windows,
            never_probed(),
            false,
        );
        assert_eq!(report.skipped.as_deref(), Some("AURORA_RAW_PREWARM=0"));
        assert_eq!(report.backend, None);
        assert!(report.devices.is_empty());
    }

    /// Re-registering the same (backend, name) replaces the body: two lanes must
    /// not both warm the same device.
    #[test]
    fn re_registering_replaces_instead_of_duplicating() {
        let registry = WarmerRegistry::new();
        let first = Arc::new(AtomicUsize::new(0));
        let second = Arc::new(AtomicUsize::new(0));
        registry.register(
            BackendKind::Dx12,
            "psos",
            counting_warmer(&first, DeviceOutcome::Warmed { loaded: 1, cached: 0 }),
        );
        registry.register(
            BackendKind::Dx12,
            "psos",
            counting_warmer(&second, DeviceOutcome::Warmed { loaded: 2, cached: 0 }),
        );
        assert_eq!(registry.registered().len(), 1);
        assert!(!registry.register(BackendKind::Dx12, "", Arc::new(|| DeviceOutcome::Failed("x".into()))));
        let warmers = registry.warmers_for(BackendKind::Dx12);
        assert_eq!(warmers.len(), 1);
        assert_eq!(warmers[0].run(), DeviceOutcome::Warmed { loaded: 2, cached: 0 });
        assert_eq!(first.load(Ordering::SeqCst), 0);
        assert_eq!(second.load(Ordering::SeqCst), 1);
        assert!(registry.warmers_for(BackendKind::Cuda).is_empty());
    }

    /// Do not advertise an implementation that is not built on this target.
    #[test]
    fn the_shipping_registry_lists_only_implemented_warmers() {
        let registered = WarmerRegistry::with_builtins().registered();
        let mut expected=vec![(BackendKind::Cuda,"cuda-ptx")];
        #[cfg(windows)] expected.push((BackendKind::Dx12,"dx12-inference-psos"));
        assert_eq!(registered,expected);
        assert_eq!(registry().registered(), registered, "the process registry must match");
    }

    /// Second call must not load anything: that is the whole point of the
    /// module cache, and it is what makes the first RAW open free of JIT.
    /// Skipped with a note on hosts without an NVIDIA device (CI).
    #[cfg(windows)]
    #[test]
    fn second_warmup_loads_no_new_modules() {
        let first = prewarm_with(
            &WarmerRegistry::with_builtins(),
            Ok(BackendRequest::Only(BackendKind::Cuda)),
            Host::Windows,
            never_probed(),
            true,
        );
        if !first.is_warm() {
            eprintln!("skipping PTX warmup cache check: {}", first.summary());
            return;
        }
        assert_eq!(first.devices.len(), 1);
        assert_eq!(first.devices[0].name, "cuda-ptx");
        let second = prewarm_with(
            &WarmerRegistry::with_builtins(),
            Ok(BackendRequest::Only(BackendKind::Cuda)),
            Host::Windows,
            never_probed(),
            true,
        );
        assert_eq!(second.cuda_module_loads, 0, "second warmup reloaded: {}", second.summary());
        assert_eq!(
            second.devices[0].outcome,
            DeviceOutcome::Warmed {
                loaded: 0,
                cached: ptx_images().len()
            },
            "{}",
            second.summary()
        );
    }

    #[test]
    fn spawn_runs_body_off_the_calling_thread() {
        // The shared `SPAWNED` latch is process-wide, so this test only asserts
        // the wiring when it wins the race; either way it must not panic.
        let (tx, rx) = std::sync::mpsc::channel();
        let started = spawn_background_with(move || {
            let _ = tx.send(std::thread::current().name().map(str::to_string));
            PrewarmReport {
                request: "auto",
                backend: Some(BackendKind::Cpu),
                skipped: Some("cpu backend: no device state to warm".into()),
                ..PrewarmReport::default()
            }
        });
        if !started {
            eprintln!("prewarm already spawned in this process; body wiring covered elsewhere");
            assert!(!spawn_background_with(|| PrewarmReport::default()), "the latch is single-flight");
            return;
        }
        let name = rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .expect("prewarm thread must run the body");
        assert_eq!(name.as_deref(), Some("aurora-raw-prewarm"));
    }

    #[test]
    fn report_summary_names_the_outcome() {
        let warm = PrewarmReport {
            request: "cuda",
            backend: Some(BackendKind::Cuda),
            devices: vec![DevicePrewarm {
                backend: BackendKind::Cuda,
                name: "cuda-ptx",
                outcome: DeviceOutcome::Warmed { loaded: 2, cached: 1 },
            }],
            cuda_module_loads: 2,
            elapsed_ms: 42,
            ..PrewarmReport::default()
        };
        let text = warm.summary();
        assert!(text.contains("backend=cuda"), "{text}");
        assert!(text.contains("3 resident"), "{text}");
        assert!(text.contains("cuda_module_loads=2"), "{text}");
        assert!(warm.is_warm());

        let skipped = PrewarmReport {
            request: "dx12",
            backend: Some(BackendKind::Dx12),
            skipped: Some("no warmer registered for backend dx12".into()),
            ..PrewarmReport::default()
        };
        assert!(skipped.summary().contains("no warmer registered"));
        assert!(!skipped.is_warm());
        assert_eq!(skipped.backend_label(), "dx12");
    }
}
