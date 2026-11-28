//! Android compute scheduling: Vulkan workgroup/tile selection by real device
//! features, Qualcomm QAIRT/QNN HTP dynamic loading, and ADPF thermal budgets.
//!
//! # Scope and evidence
//!
//! Candidate selection, budget arithmetic and QNN partition verification are
//! pure Rust and run under `cargo test` on the Windows host. The dynamic QNN
//! loader is compiled only for Android; without a device it reports
//! `QnnLoadState::Unavailable` and never claims NPU execution. Nothing in this
//! module performs a photo/parity/performance claim.
//!
//! # Strict independence
//!
//! This file is Android-only. HarmonyOS code lives in
//! [`crate::harmony_ohos`] and must not use any symbol from here.

use std::fmt;

use crate::autotune::{
    CandidateSpec, ExecutionMode, ParamBound, ResourceBudget, VerificationStatus,
};
use crate::device::{Backend, DeviceProbe, Precision};

// ---------------------------------------------------------------------------
// GPU architecture table
// ---------------------------------------------------------------------------

/// Mobile GPU architecture families. The name is a hint for telemetry only;
/// the actual dispatch decision always uses reported Vulkan features.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AndroidGpuArchitecture {
    /// Qualcomm Adreno 6xx/7xx (Vulkan compute, subgroup + FP16 storage).
    Adreno,
    /// ARM Mali-Gxx (Midgard/Bifrost/Valhall), no guaranteed subgroup FP16.
    Mali,
    /// ARM Immortalis-Gxx (Valhall, hardware ray tracing tier).
    Immortalis,
    /// Samsung Xclipse (RDNA-based, wavefront 32/64 semantics).
    Xclipse,
    /// Unknown or caller-declared generic Vulkan device.
    Generic,
}

impl AndroidGpuArchitecture {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Adreno => "adreno",
            Self::Mali => "mali",
            Self::Immortalis => "immortalis",
            Self::Xclipse => "xclipse",
            Self::Generic => "generic",
        }
    }

    /// Best-effort classification from `VkPhysicalDeviceProperties`. A device
    /// name alone never enables a candidate; [`VulkanDeviceFeatures`] gates it.
    pub fn from_device_name(name: &str) -> Self {
        let lower = name.to_ascii_lowercase();
        if lower.contains("adreno") {
            Self::Adreno
        } else if lower.contains("immortalis") {
            Self::Immortalis
        } else if lower.contains("mali") {
            Self::Mali
        } else if lower.contains("xclipse") {
            Self::Xclipse
        } else {
            Self::Generic
        }
    }

    /// Preferred subgroup width hint. Real width comes from the device.
    pub const fn preferred_subgroup_hint(self) -> u32 {
        match self {
            Self::Adreno | Self::Xclipse => 64,
            Self::Mali | Self::Immortalis => 32,
            Self::Generic => 32,
        }
    }
}

impl fmt::Display for AndroidGpuArchitecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Caller-reported `VkPhysicalDevice` facts. Constructed by the Android Vulkan
/// adapter; no inference from OS/SoC name is performed here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanDeviceFeatures {
    pub architecture: AndroidGpuArchitecture,
    pub device_name: String,
    pub driver_version: u32,
    pub api_version: u32,
    pub driver_ready: bool,
    pub kernel_compiled: bool,
    /// `VK_KHR_subgroup` / Vulkan 1.1 core subgroup support.
    pub supports_subgroup: bool,
    /// `subgroup_size` from `vkGetPhysicalDeviceProperties2KHR`.
    pub subgroup_size: u32,
    /// `VK_KHR_16bit_storage` with storage image/buffer FP16 support.
    pub supports_fp16_storage: bool,
    /// `shaderInt16` / `VK_KHR_shader_float16_int8` for arithmetic.
    pub supports_fp16_arithmetic: bool,
    pub max_shared_mem_bytes: u32,
    pub max_workgroup_invocations: u32,
    pub max_image_dimension: u32,
}

impl VulkanDeviceFeatures {
    /// Synthetic feature set for host-side tests. Never use as a device claim.
    pub fn synthetic(architecture: AndroidGpuArchitecture) -> Self {
        Self {
            architecture,
            device_name: format!("synthetic-{architecture}"),
            driver_version: 0,
            api_version: 0,
            driver_ready: true,
            kernel_compiled: true,
            supports_subgroup: true,
            subgroup_size: architecture.preferred_subgroup_hint(),
            supports_fp16_storage: true,
            supports_fp16_arithmetic: true,
            max_shared_mem_bytes: 32 * 1024,
            max_workgroup_invocations: 256,
            max_image_dimension: 8192,
        }
    }

    pub fn to_device_probe(&self) -> DeviceProbe {
        DeviceProbe {
            backend: Backend::Vulkan,
            driver_ready: self.driver_ready,
            kernel_compiled: self.kernel_compiled,
            fp32: true,
            fp16: self.supports_fp16_storage && self.supports_fp16_arithmetic,
            max_width: self.max_image_dimension,
            max_height: self.max_image_dimension,
        }
    }
}

/// Concrete candidate produced by the feature-driven scheduling table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VulkanCandidate {
    pub spec: CandidateSpec,
    /// Chosen `workgroup_x`; a divisor of the subgroup/wavefront hint.
    pub workgroup_x: u32,
    /// Rows per tiled dispatch, derived from available shared memory.
    pub tile_rows: u32,
    /// Fused pixel pipeline precision.
    pub precision: Precision,
    /// Human-readable reason for audit logs.
    pub reason: String,
}

static FUSION_PARAMS: &[ParamBound] = &[
    ParamBound::DiscreteU32 {
        name: "workgroup_x",
        values: &[32, 64, 128, 256],
    },
    ParamBound::U32Range {
        name: "tile_rows",
        min: 4,
        max: 64,
        step: 4,
    },
];

fn candidate_spec(
    id: &'static str,
    precision: Precision,
    max_shared_mem_bytes: u32,
    max_threads_per_group: u32,
) -> CandidateSpec {
    CandidateSpec {
        id,
        algorithm_id: "pixel_stage_fusion",
        backend: Backend::Vulkan,
        kernel_entry: "aurora_fused_rgba8",
        precision,
        params: FUSION_PARAMS,
        max_shared_mem_bytes,
        max_threads_per_group,
        // Feature selection is implemented; device execution is not verified.
        status: VerificationStatus::Implemented,
    }
}

fn choose_workgroup(features: &VulkanDeviceFeatures) -> (u32, &'static str) {
    let hint = if features.supports_subgroup && features.subgroup_size >= 16 {
        features.subgroup_size.clamp(16, 256)
    } else {
        features.architecture.preferred_subgroup_hint().clamp(16, 256)
    };
    let workgroup = if features.max_workgroup_invocations < hint {
        features.max_workgroup_invocations.max(16)
    } else {
        hint
    };
    let workgroup = workgroup.min(256);
    let rounded = match workgroup {
        0..=31 => 32,
        32..=63 => 64,
        64..=127 => 128,
        _ => 256,
    };
    let reason = if features.supports_subgroup {
        "subgroup_size"
    } else {
        "architecture_hint_without_subgroup"
    };
    (rounded, reason)
}

fn choose_tile_rows(features: &VulkanDeviceFeatures) -> (u32, &'static str) {
    // RGBA8 tile: width * rows * 4 bytes in + out. Assume a conservative
    // 64-pixel row at 8 bytes/pixel for the staging estimate.
    let per_row = 64u32 * 8;
    let rows = (features.max_shared_mem_bytes / 2 / per_row).clamp(4, 64);
    let rows = (rows / 4) * 4;
    let reason = if features.max_shared_mem_bytes >= 16 * 1024 {
        "fp16_storage_tile"
    } else {
        "bounded_shared_memory_tile"
    };
    (rows.max(4), reason)
}

/// Select a Vulkan fused-pixel candidate from *reported device features*.
///
/// A name like "Adreno 740" is never sufficient: if `driver_ready`,
/// `kernel_compiled`, subgroup or shared-memory values are absent, the caller
/// receives a CPU-only status and must not dispatch.
pub fn select_vulkan_candidate(features: &VulkanDeviceFeatures) -> Result<VulkanCandidate, &'static str> {
    if !features.driver_ready {
        return Err("vulkan driver not ready");
    }
    if !features.kernel_compiled {
        return Err("fused pixel kernel not compiled");
    }
    if features.max_shared_mem_bytes < 1024 || features.max_workgroup_invocations < 16 {
        return Err("vulkan device shared memory/workgroup limits too small");
    }
    if features.supports_fp16_storage && !features.supports_fp16_arithmetic {
        return Err("FP16 storage without FP16 arithmetic cannot run fused kernels");
    }

    let (workgroup_x, wg_reason) = choose_workgroup(features);
    let (tile_rows, tile_reason) = choose_tile_rows(features);
    let use_fp16 = features.supports_fp16_storage && features.supports_fp16_arithmetic;
    let precision = if use_fp16 {
        Precision::Fp16
    } else {
        Precision::Fp32
    };
    let shared_needed = tile_rows * 64 * 8;
    let id = match (features.architecture, use_fp16) {
        (AndroidGpuArchitecture::Adreno, true) => "adreno-fp16-subgroup-fusion",
        (AndroidGpuArchitecture::Mali, false) => "mali-fp32-tile-fusion",
        (AndroidGpuArchitecture::Mali, true) => "mali-fp16-tile-fusion",
        (AndroidGpuArchitecture::Immortalis, true) => "immortalis-fp16-subgroup-fusion",
        (AndroidGpuArchitecture::Xclipse, true) => "xclipse-fp16-wave64-fusion",
        (AndroidGpuArchitecture::Xclipse, false) => "xclipse-fp32-wave64-fusion",
        (AndroidGpuArchitecture::Adreno, false) => "adreno-fp32-fusion",
        (AndroidGpuArchitecture::Immortalis, false) => "immortalis-fp32-fusion",
        (AndroidGpuArchitecture::Generic, true) => "vulkan-fp16-generic-fusion",
        (AndroidGpuArchitecture::Generic, false) => "vulkan-fp32-generic-fusion",
    };
    let spec = candidate_spec(id, precision, shared_needed, workgroup_x);
    Ok(VulkanCandidate {
        spec,
        workgroup_x,
        tile_rows,
        precision,
        reason: format!(
            "{}:{}:{}:{}x{}",
            features.architecture,
            wg_reason,
            tile_reason,
            workgroup_x,
            tile_rows
        ),
    })
}

/// Reject a candidate that exceeds the ADPF-adjusted mobile resource budget.
pub fn candidate_fits(candidate: &VulkanCandidate, budget: &ResourceBudget) -> bool {
    candidate.spec.fits_budget(budget)
}

// ---------------------------------------------------------------------------
// ADPF thermal / performance hint budget
// ---------------------------------------------------------------------------

/// `AThermalStatus` values from `android/thermal.h` (API 29+).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum AdpfThermalStatus {
    Green = 0,
    Yellow = 1,
    Red = 2,
    Grey = 3,
    Emergency = 4,
    Shutdown = 5,
    Critical = 6,
    Error = 7,
}

impl AdpfThermalStatus {
    pub const fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(Self::Green),
            1 => Some(Self::Yellow),
            2 => Some(Self::Red),
            3 => Some(Self::Grey),
            4 => Some(Self::Emergency),
            5 => Some(Self::Shutdown),
            6 => Some(Self::Critical),
            7 => Some(Self::Error),
            _ => None,
        }
    }
}

/// Which ADPF hint the native bridge should report for the current frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdpfHint {
    None,
    Start,
    Bind,
    Stop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdpfBudget {
    pub thermal_status: AdpfThermalStatus,
    /// Estimated available headroom in 0..100 from `ATrace_getThermalHeadroom`.
    pub thermal_headroom_pct: u32,
    /// Fraction of the mobile budget retained for the fused pipeline.
    pub budget_pct: u32,
    pub max_concurrent_dispatches: u32,
    pub hint: AdpfHint,
}

impl AdpfBudget {
    /// Combine `AThermalStatus` with headroom and an optional `onBattery`
    /// signal. All arithmetic saturates; no device state is inferred here.
    pub fn compute(
        thermal_status: AdpfThermalStatus,
        thermal_headroom: f32,
        low_memory: bool,
    ) -> Self {
        let headroom_pct = if thermal_headroom.is_finite() {
            (thermal_headroom.clamp(0.0, 1.0) * 100.0).round() as u32
        } else {
            0
        };
        let status_pct = match thermal_status {
            AdpfThermalStatus::Green | AdpfThermalStatus::Grey => 100u32,
            AdpfThermalStatus::Yellow => 75,
            AdpfThermalStatus::Red => 45,
            AdpfThermalStatus::Emergency | AdpfThermalStatus::Critical => 15,
            AdpfThermalStatus::Shutdown | AdpfThermalStatus::Error => 0,
        };
        let mut budget_pct = status_pct.min(100 - (100u32.saturating_sub(headroom_pct)) / 2);
        if low_memory {
            budget_pct = budget_pct.min(25);
        }
        let dispatches = if budget_pct >= 75 {
            2
        } else if budget_pct >= 45 {
            1
        } else {
            0
        };
        let hint = if budget_pct == 0 {
            AdpfHint::Stop
        } else if budget_pct < 45 {
            AdpfHint::Bind
        } else if budget_pct >= 75 {
            AdpfHint::Start
        } else {
            AdpfHint::None
        };
        Self {
            thermal_status,
            thermal_headroom_pct: headroom_pct,
            budget_pct,
            max_concurrent_dispatches: dispatches,
            hint,
        }
    }

    /// Apply this ADPF budget to a conservative mobile `ResourceBudget`.
    pub fn apply(&self, base: ResourceBudget, mode: ExecutionMode) -> Result<ResourceBudget, &'static str> {
        if self.max_concurrent_dispatches == 0 {
            if mode == ExecutionMode::Strict {
                return Err("ADPF thermal/memory budget exhausted under Strict mode");
            }
            return Ok(ResourceBudget {
                max_concurrent_dispatches: 1,
                ..base
            });
        }
        let pct = self.budget_pct as u64;
        Ok(ResourceBudget {
            max_vram_bytes: base.max_vram_bytes * pct / 100,
            max_staging_bytes: base.max_staging_bytes * pct / 100,
            max_shared_mem_bytes: base.max_shared_mem_bytes,
            max_threads_per_group: base.max_threads_per_group,
            max_concurrent_dispatches: base
                .max_concurrent_dispatches
                .min(self.max_concurrent_dispatches),
        })
    }
}

// ---------------------------------------------------------------------------
// Qualcomm QAIRT / QNN HTP dynamic loader and cache abstraction
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QnnLibraryId {
    /// `libQnnHtp.so`: Hexagon DSP / HTP partition.
    Htp,
    /// `libQnnCpu.so`: CPU reference backend, never evidence of NPU execution.
    Cpu,
    /// `libQnnSystem.so`: device/partition introspection.
    System,
}

impl QnnLibraryId {
    pub const fn filename(self) -> &'static str {
        match self {
            Self::Htp => "libQnnHtp.so",
            Self::Cpu => "libQnnCpu.so",
            Self::System => "libQnnSystem.so",
        }
    }

    pub const fn is_npu(self) -> bool {
        matches!(self, Self::Htp)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QnnLoadState {
    Unavailable,
    Loaded,
    PartitionRejected,
    ContextCached,
}

/// Calibration state of a QNN binary. A quantized graph without calibration
/// cannot claim a working INT8 HTP partition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QnnCalibration {
    #[default]
    NotRequired,
    Pending,
    Fp16Ready,
    Int8Ready,
}

impl QnnCalibration {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotRequired => "not_required",
            Self::Pending => "pending",
            Self::Fp16Ready => "fp16_ready",
            Self::Int8Ready => "int8_ready",
        }
    }
}

/// Evidence that the graph really executed on a DSP/NPU partition, supplied by
/// the QNN System API (caller-populated on the device).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QnnPartitionEvidence {
    pub backend_name: String,
    pub device_id: u32,
    /// `QNN_DEVICE_TYPE_DSP` equivalent reported by QnnSystem.
    pub is_dsp_device: bool,
    /// Number of graph nodes delegated to HTP.
    pub ops_on_htp: usize,
    /// Number of graph nodes delegated to CPU or other fallback partitions.
    pub ops_on_fallback: usize,
    pub calibration: QnnCalibration,
}

impl QnnPartitionEvidence {
    /// Strict verification: a real NPU partition must be a DSP device, must use
    /// the HTP backend, and must delegate at least one op with zero fallback.
    pub fn verify_strict(&self) -> Result<(), &'static str> {
        if !self.is_dsp_device {
            return Err("QNN backend is not a DSP/NPU device");
        }
        if !self.backend_name.eq_ignore_ascii_case("htp") {
            return Err("QNN backend is not HTP (CPU fallback is not NPU)");
        }
        if self.ops_on_htp == 0 {
            return Err("no graph ops delegated to HTP");
        }
        if self.ops_on_fallback != 0 {
            return Err("graph contains fallback-partition ops");
        }
        match self.calibration {
            QnnCalibration::NotRequired | QnnCalibration::Fp16Ready | QnnCalibration::Int8Ready => Ok(()),
            QnnCalibration::Pending => Err("QNN graph calibration is still pending"),
        }
    }
}

/// Cache key for `QnnContext_createFromBinary`. The caller supplies a model
/// digest; this struct never stores raw model bytes or user data.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct QnnContextCacheKey {
    pub model_sha256: [u8; 32],
    pub binary_size: u64,
    pub precision: Precision,
    pub partition: QnnLibraryId,
    pub sdk_version: u32,
}

impl QnnContextCacheKey {
    pub fn new(
        model_sha256: [u8; 32],
        binary_size: u64,
        precision: Precision,
        sdk_version: u32,
    ) -> Self {
        Self {
            model_sha256,
            binary_size,
            precision,
            partition: QnnLibraryId::Htp,
            sdk_version,
        }
    }

    /// Cache entries for the CPU backend are deliberately impossible: a CPU
    /// context must not be mistaken for NPU evidence.
    pub fn is_npu_key(&self) -> bool {
        self.partition.is_npu()
    }
}

#[derive(Clone, Debug, Default)]
struct QnnCacheEntry {
    /// Opaque handle returned by the device-side QNN loader.
    native_context: usize,
    evidence: QnnPartitionEvidence,
}

#[derive(Debug, Default)]
struct QnnLoaderState {
    htp_loaded: bool,
    cpu_loaded: bool,
    system_loaded: bool,
    cache: std::collections::HashMap<QnnContextCacheKey, QnnCacheEntry>,
}

/// OS-independent facade over the dynamic loader. On non-Android hosts it is
/// always unavailable and the cache is empty, so host tests only exercise
/// bookkeeping and strict verification logic.
#[derive(Debug, Default)]
pub struct QnnDynamicLoader {
    state: QnnLoaderState,
}

impl QnnDynamicLoader {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn state(&self) -> QnnLoadState {
        if !self.state.htp_loaded {
            QnnLoadState::Unavailable
        } else if !self.state.cache.is_empty() {
            QnnLoadState::ContextCached
        } else {
            QnnLoadState::Loaded
        }
    }

    /// Load the QAIRT shared objects with `dlopen(RTLD_NOW)`. Returns `false`
    /// on any host where the libraries are absent; never fabricates an NPU.
    pub fn load(&mut self) -> bool {
        #[cfg(target_os = "android")]
        {
            let htp = crate::android_qnn_vulkan::sys::load_library(QnnLibraryId::Htp.filename());
            let system =
                crate::android_qnn_vulkan::sys::load_library(QnnLibraryId::System.filename());
            self.state.htp_loaded = htp.is_some();
            self.state.system_loaded = system.is_some();
            // CPU backend is loaded only for explicit fallback diagnostics.
            // It is never promoted into an NPU candidate.
            let cpu = crate::android_qnn_vulkan::sys::load_library(QnnLibraryId::Cpu.filename());
            self.state.cpu_loaded = cpu.is_some();
        }
        #[cfg(not(target_os = "android"))]
        {
            self.state.htp_loaded = false;
            self.state.system_loaded = false;
            self.state.cpu_loaded = false;
        }
        self.state.htp_loaded && self.state.system_loaded
    }

    pub fn cache_len(&self) -> usize {
        self.state.cache.len()
    }

    /// Cache lookup for `QnnContext_createFromBinary`. The binary bytes are
    /// never copied into this Rust-side abstract cache; the native loader owns
    /// the actual context.
    pub fn cached_context(&self, key: &QnnContextCacheKey) -> Option<usize> {
        if !key.is_npu_key() || !self.state.htp_loaded {
            return None;
        }
        self.state
            .cache
            .get(key)
            .and_then(|entry| entry.evidence.verify_strict().ok().map(|_| entry.native_context))
    }

    /// Insert a device-created context. Strict mode rejects partitions that
    /// are CPU fallbacks or lack calibration evidence.
    pub fn insert_context(
        &mut self,
        key: QnnContextCacheKey,
        native_context: usize,
        evidence: QnnPartitionEvidence,
        mode: ExecutionMode,
    ) -> Result<(), &'static str> {
        if !key.is_npu_key() {
            return Err("QNN cache key is not an HTP partition");
        }
        if native_context == 0 {
            return Err("QNN context handle is null");
        }
        if mode == ExecutionMode::Strict {
            evidence.verify_strict()?;
        } else if !evidence.is_dsp_device || evidence.ops_on_htp == 0 {
            return Err("QNN device is not an NPU partition");
        }
        self.state
            .cache
            .insert(key, QnnCacheEntry { native_context, evidence });
        Ok(())
    }

    pub fn clear_cache(&mut self) {
        self.state.cache.clear();
    }
}

/// Android-only `dlopen`/`dlsym` shim. Symbols are resolved lazily and kept
/// opaque: the Rust side never calls a QNN function with an unverified handle.
#[cfg(target_os = "android")]
pub mod sys {
    use std::ffi::CString;
    use std::os::raw::{c_char, c_int, c_void};

    const RTLD_NOW: c_int = 2;

    unsafe extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        fn dlclose(handle: *mut c_void) -> c_int;
    }

    /// Opaque library handle wrapper that closes on drop.
    pub struct Library(*mut c_void);

    impl Library {
        pub fn symbol(&self, name: &str) -> Option<*mut c_void> {
            let name = CString::new(name).ok()?;
            let ptr = unsafe { dlsym(self.0, name.as_ptr()) };
            (ptr as *const c_void != std::ptr::null()).then_some(ptr)
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            if !self.0.is_null() {
                unsafe {
                    dlclose(self.0);
                }
            }
        }
    }

    pub fn load_library(name: &str) -> Option<Library> {
        let name = CString::new(name).ok()?;
        let ptr = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
        (!ptr.is_null()).then_some(Library(ptr))
    }

    /// Required QAIRT entry points. Missing any of them disables the NPU
    /// candidate; a CPU-only library is never sufficient.
    pub const REQUIRED_QNN_SYMBOLS: &[&str] = &[
        "QnnBackend_create",
        "QnnBackend_initialize",
        "QnnDevice_create",
        "QnnSystemContext_create",
        "QnnContext_createFromBinary",
        "QnnGraph_execute",
    ];
}

// ---------------------------------------------------------------------------
// Host tests (synthetic feature sets only)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_name_alone_does_not_enable_a_candidate() {
        let mut features = VulkanDeviceFeatures::synthetic(AndroidGpuArchitecture::Adreno);
        features.driver_ready = false;
        assert_eq!(
            select_vulkan_candidate(&features),
            Err("vulkan driver not ready")
        );
        features.driver_ready = true;
        features.kernel_compiled = false;
        assert_eq!(
            select_vulkan_candidate(&features),
            Err("fused pixel kernel not compiled")
        );
        features.kernel_compiled = true;
        features.supports_fp16_storage = true;
        features.supports_fp16_arithmetic = false;
        assert_eq!(
            select_vulkan_candidate(&features),
            Err("FP16 storage without FP16 arithmetic cannot run fused kernels")
        );
    }

    #[test]
    fn architecture_features_drive_workgroup_tile_and_precision() {
        for arch in [
            AndroidGpuArchitecture::Adreno,
            AndroidGpuArchitecture::Mali,
            AndroidGpuArchitecture::Immortalis,
            AndroidGpuArchitecture::Xclipse,
            AndroidGpuArchitecture::Generic,
        ] {
            let features = VulkanDeviceFeatures::synthetic(arch);
            let candidate = select_vulkan_candidate(&features).unwrap();
            assert_eq!(candidate.spec.backend, Backend::Vulkan);
            assert_eq!(candidate.precision, Precision::Fp16);
            assert!(candidate.workgroup_x >= 32 && candidate.workgroup_x <= 256);
            assert!(candidate.tile_rows >= 4 && candidate.tile_rows <= 64);
            assert!(candidate.reason.contains(arch.as_str()));
            assert!(candidate_fits(&candidate, &ResourceBudget::conservative_mobile()));

            // Same name, weaker features: FP32 and a smaller tile.
            let mut weak = features.clone();
            weak.supports_fp16_storage = false;
            weak.supports_subgroup = false;
            weak.subgroup_size = 0;
            weak.max_shared_mem_bytes = 4 * 1024;
            let weak_candidate = select_vulkan_candidate(&weak).unwrap();
            assert_eq!(weak_candidate.precision, Precision::Fp32);
            assert!(weak_candidate.tile_rows <= candidate.tile_rows);
            assert!(weak_candidate.reason.contains("architecture_hint_without_subgroup"));
        }
    }

    #[test]
    fn device_name_classification_is_best_effort_only() {
        assert_eq!(
            AndroidGpuArchitecture::from_device_name("Adreno (TM) 740"),
            AndroidGpuArchitecture::Adreno
        );
        assert_eq!(
            AndroidGpuArchitecture::from_device_name("Mali-G78 MP14"),
            AndroidGpuArchitecture::Mali
        );
        assert_eq!(
            AndroidGpuArchitecture::from_device_name("Immortalis-G715"),
            AndroidGpuArchitecture::Immortalis
        );
        assert_eq!(
            AndroidGpuArchitecture::from_device_name("Samsung Xclipse 940"),
            AndroidGpuArchitecture::Xclipse
        );
        assert_eq!(
            AndroidGpuArchitecture::from_device_name("unknown"),
            AndroidGpuArchitecture::Generic
        );
    }

    #[test]
    fn adpf_budget_is_monotonic_and_restricts_dispatch() {
        let green = AdpfBudget::compute(AdpfThermalStatus::Green, 1.0, false);
        let yellow = AdpfBudget::compute(AdpfThermalStatus::Yellow, 0.5, false);
        let emergency = AdpfBudget::compute(AdpfThermalStatus::Emergency, 0.0, false);
        let shutdown = AdpfBudget::compute(AdpfThermalStatus::Shutdown, 1.0, false);
        assert!(green.budget_pct > yellow.budget_pct);
        assert!(yellow.budget_pct > emergency.budget_pct);
        assert_eq!(shutdown.budget_pct, 0);
        assert_eq!(shutdown.hint, AdpfHint::Stop);
        assert_eq!(shutdown.max_concurrent_dispatches, 0);

        let base = ResourceBudget::conservative_mobile();
        let relaxed = green.apply(base, ExecutionMode::Strict).unwrap();
        assert_eq!(relaxed.max_concurrent_dispatches, 2);
        let reduced = emergency.apply(base, ExecutionMode::AutoFallback).unwrap();
        assert_eq!(reduced.max_concurrent_dispatches, 1);
        assert_eq!(
            emergency.apply(base, ExecutionMode::Strict),
            Err("ADPF thermal/memory budget exhausted under Strict mode")
        );

        let low_mem = AdpfBudget::compute(AdpfThermalStatus::Green, 1.0, true);
        assert!(low_mem.budget_pct <= 25);
        assert_eq!(AdpfThermalStatus::from_i32(99), None);
        assert_eq!(AdpfThermalStatus::from_i32(2), Some(AdpfThermalStatus::Red));
    }

    fn strict_evidence() -> QnnPartitionEvidence {
        QnnPartitionEvidence {
            backend_name: "HTP".to_string(),
            device_id: 3,
            is_dsp_device: true,
            ops_on_htp: 12,
            ops_on_fallback: 0,
            calibration: QnnCalibration::Int8Ready,
        }
    }

    #[test]
    fn qnn_strict_verification_rejects_cpu_fallback() {
        let good = strict_evidence();
        assert_eq!(good.verify_strict(), Ok(()));

        let mut cpu = good.clone();
        cpu.backend_name = "CPU".to_string();
        assert_eq!(
            cpu.verify_strict(),
            Err("QNN backend is not HTP (CPU fallback is not NPU)")
        );

        let mut host = good.clone();
        host.is_dsp_device = false;
        assert_eq!(
            host.verify_strict(),
            Err("QNN backend is not a DSP/NPU device")
        );

        let mut empty = good.clone();
        empty.ops_on_htp = 0;
        assert_eq!(empty.verify_strict(), Err("no graph ops delegated to HTP"));

        let mut mixed = good.clone();
        mixed.ops_on_fallback = 1;
        assert_eq!(
            mixed.verify_strict(),
            Err("graph contains fallback-partition ops")
        );

        let mut pending = good;
        pending.calibration = QnnCalibration::Pending;
        assert_eq!(
            pending.verify_strict(),
            Err("QNN graph calibration is still pending")
        );
    }

    #[test]
    fn qnn_cache_never_promotes_cpu_contexts() {
        let mut loader = QnnDynamicLoader::new();
        // No QAIRT library on the Windows host: loader must remain unavailable.
        assert!(!loader.load());
        assert_eq!(loader.state(), QnnLoadState::Unavailable);
        let key = QnnContextCacheKey::new([7u8; 32], 4096, Precision::Fp16, 1);
        assert!(key.is_npu_key());
        assert_eq!(loader.cached_context(&key), None);

        // Even with a synthetic loaded state, strict verification still checks
        // the partition evidence on insertion.
        loader.state.htp_loaded = true;
        let bad = QnnPartitionEvidence {
            backend_name: "cpu".to_string(),
            ..Default::default()
        };
        assert!(loader.insert_context(key.clone(), 1, bad, ExecutionMode::Strict).is_err());
        assert_eq!(loader.cache_len(), 0);
        loader
            .insert_context(key.clone(), 42, strict_evidence(), ExecutionMode::Strict)
            .unwrap();
        assert_eq!(loader.cached_context(&key), Some(42));
        assert_eq!(loader.state(), QnnLoadState::ContextCached);
        loader.clear_cache();
        assert_eq!(loader.cache_len(), 0);
        assert_eq!(QnnLibraryId::Cpu.filename(), "libQnnCpu.so");
        assert!(!QnnLibraryId::Cpu.is_npu());
    }
}
