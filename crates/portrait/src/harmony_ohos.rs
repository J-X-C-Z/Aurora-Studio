//! HarmonyOS NEXT compute/AI tier planning. Strictly independent of Android.
//!
//! Nothing in this module references Android JNI, `AHardwareBuffer`, QAIRT,
//! ADPF, `libQnn*`, `dlopen`, or any Android NDK header. The OHOS bridge uses
//! the Native SDK C++ Node-API and ArkTS/XComponent; see
//! `native/harmony/aurora_napi.cpp` and `native/harmony/AuroraBridge.ets`.
//!
//! # Scope and evidence
//!
//! Tier ordering, Vulkan feature gating, worker-queue modelling and capability
//! reporting are pure Rust and host-testable. No MindSpore/NNRt/CANN execution
//! is claimed: these tiers are `DESIGNED`/`IMPLEMENTED` planning states until a
//! real Kirin/OHOS device produces evidence.

use std::collections::VecDeque;
use std::fmt;

use crate::autotune::{CandidateSpec, ResourceBudget, VerificationStatus};
use crate::device::{Backend, Precision};

// ---------------------------------------------------------------------------
// OHOS Vulkan probe (no Android extensions)
// ---------------------------------------------------------------------------

/// Caller-reported Vulkan device facts from the OHOS Native SDK. Feature
/// names deliberately use core Vulkan / KHR names only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OhosVulkanProbe {
    pub device_name: String,
    pub api_version: u32,
    pub driver_ready: bool,
    pub kernel_compiled: bool,
    pub supports_fp16_storage: bool,
    pub supports_subgroup: bool,
    pub subgroup_size: u32,
    pub max_shared_mem_bytes: u32,
    pub max_workgroup_invocations: u32,
    pub max_image_dimension: u32,
    pub has_ohos_native_buffer_extension: bool,
}

impl OhosVulkanProbe {
    pub fn synthetic() -> Self {
        Self {
            device_name: "synthetic-ohos-vulkan".to_string(),
            api_version: (1 << 22) | (3 << 12),
            driver_ready: true,
            kernel_compiled: true,
            supports_fp16_storage: true,
            supports_subgroup: true,
            subgroup_size: 32,
            max_shared_mem_bytes: 16 * 1024,
            max_workgroup_invocations: 256,
            max_image_dimension: 8192,
            has_ohos_native_buffer_extension: true,
        }
    }
}

/// Concrete Vulkan tile choice without Android-specific external memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OhosVulkanPlan {
    pub workgroup_x: u32,
    pub tile_rows: u32,
    /// Bounded host staging when no OHOS buffer import extension is present.
    pub use_bounded_staging: bool,
    pub precision: Precision,
    pub reason: String,
}

static OHOS_PARAMS: &[crate::autotune::ParamBound] = &[
    crate::autotune::ParamBound::DiscreteU32 {
        name: "workgroup_x",
        values: &[32, 64, 128, 256],
    },
    crate::autotune::ParamBound::U32Range {
        name: "tile_rows",
        min: 4,
        max: 64,
        step: 4,
    },
];

pub fn ohos_vulkan_candidate(plan: &OhosVulkanPlan) -> CandidateSpec {
    CandidateSpec {
        id: "ohos-vulkan-fused-rgba8",
        algorithm_id: "pixel_stage_fusion",
        backend: Backend::Vulkan,
        kernel_entry: "aurora_fused_rgba8_ohos",
        precision: plan.precision,
        params: OHOS_PARAMS,
        max_shared_mem_bytes: plan.tile_rows * 512,
        max_threads_per_group: plan.workgroup_x,
        status: VerificationStatus::Implemented,
    }
}

/// Plan an OHOS Vulkan dispatch. When the OHOS native buffer extension is
/// missing, the plan switches to bounded staging rather than assuming Android
/// `AHardwareBuffer` semantics.
pub fn plan_ohos_vulkan(probe: &OhosVulkanProbe) -> Result<OhosVulkanPlan, &'static str> {
    if !probe.driver_ready {
        return Err("OHOS Vulkan driver not ready");
    }
    if !probe.kernel_compiled {
        return Err("OHOS fused pixel kernel not compiled");
    }
    if probe.max_shared_mem_bytes < 1024 || probe.max_workgroup_invocations < 16 {
        return Err("OHOS Vulkan limits too small");
    }
    let wg_hint = if probe.supports_subgroup && probe.subgroup_size >= 16 {
        probe.subgroup_size.clamp(16, 256)
    } else {
        32
    };
    let workgroup_x = (probe.max_workgroup_invocations.min(wg_hint)).min(256).max(16);
    let tile_rows = (probe.max_shared_mem_bytes / 2 / 512).clamp(4, 64);
    let precision = if probe.supports_fp16_storage {
        Precision::Fp16
    } else {
        Precision::Fp32
    };
    let bounded = !probe.has_ohos_native_buffer_extension;
    Ok(OhosVulkanPlan {
        workgroup_x,
        tile_rows,
        use_bounded_staging: bounded,
        precision,
        reason: format!(
            "ohos-vulkan:{}:{}:wg{}:tile{}",
            if bounded { "bounded_staging" } else { "native_buffer" },
            if precision == Precision::Fp16 { "fp16" } else { "fp32" },
            workgroup_x,
            tile_rows
        ),
    })
}

// ---------------------------------------------------------------------------
// Three-level OHOS AI tier
// ---------------------------------------------------------------------------

/// HarmonyOS AI tiers, strictly ordered by preference.
///
/// 1. MindSpore Lite Kit (preferred native inference).
/// 2. NNRt (system neural network runtime) — synchronous/blocking and subject
///    to composition concurrency limits, therefore uses a dedicated worker.
/// 3. CANN Kit — only for supported Kirin NPU devices; NOT equivalent to
///    desktop Ascend CANN.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OhosAiTier {
    MindSporeLiteKit,
    NnRt,
    CannKit,
}

impl OhosAiTier {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::MindSporeLiteKit => "mindspore_lite_kit",
            Self::NnRt => "nnrt",
            Self::CannKit => "cann_kit",
        }
    }

    pub const fn priority(self) -> u8 {
        match self {
            Self::MindSporeLiteKit => 1,
            Self::NnRt => 2,
            Self::CannKit => 3,
        }
    }

    /// NNRt must not be invoked on the ArkTS UI thread.
    pub const fn requires_dedicated_worker(self) -> bool {
        matches!(self, Self::NnRt)
    }

    /// CANN Kit is restricted by Kirin NPU support; never assume desktop Ascend
    /// CANN device drivers, toolchains or operator coverage.
    pub const fn requires_kirin_npu(self) -> bool {
        matches!(self, Self::CannKit)
    }
}

impl fmt::Display for OhosAiTier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

pub const OHOS_AI_TIER_ORDER: [OhosAiTier; 3] = [
    OhosAiTier::MindSporeLiteKit,
    OhosAiTier::NnRt,
    OhosAiTier::CannKit,
];

/// Device facts relevant to AI tier selection, supplied by the OHOS host.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OhosAiDeviceInfo {
    pub mindspore_lite_available: bool,
    pub nnrt_available: bool,
    pub cann_kit_available: bool,
    /// `true` only for a Kirin NPU explicitly reported by the device.
    pub kirin_npu_supported: bool,
    /// Kernel/model compiled for the selected tier.
    pub kernel_compiled: bool,
    /// Candidate meets precision and shape requirements.
    pub fp16_ready: bool,
    /// Model binary SHA-256 is present and verified.
    pub model_verified: bool,
}

impl OhosAiDeviceInfo {
    pub fn synthetic() -> Self {
        Self {
            mindspore_lite_available: true,
            nnrt_available: true,
            cann_kit_available: false,
            kirin_npu_supported: false,
            kernel_compiled: true,
            fp16_ready: true,
            model_verified: true,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OhosAiSelection {
    pub tier: OhosAiTier,
    pub reason: String,
    pub status: VerificationStatus,
}

/// Select the highest-priority AI tier that is actually available and verified.
/// CANN is never selected from desktop Ascend assumptions.
pub fn select_ohos_ai_tier(info: &OhosAiDeviceInfo) -> Result<OhosAiSelection, &'static str> {
    if !info.kernel_compiled {
        return Err("OHOS AI kernel not compiled");
    }
    if !info.model_verified {
        return Err("OHOS model binary not verified");
    }
    if !info.fp16_ready {
        return Err("OHOS AI candidate precision unsupported");
    }
    for tier in OHOS_AI_TIER_ORDER {
        let available = match tier {
            OhosAiTier::MindSporeLiteKit => info.mindspore_lite_available,
            OhosAiTier::NnRt => info.nnrt_available,
            OhosAiTier::CannKit => info.cann_kit_available && info.kirin_npu_supported,
        };
        if available {
            return Ok(OhosAiSelection {
                tier,
                reason: match tier {
                    OhosAiTier::MindSporeLiteKit => "MindSpore Lite Kit available".to_string(),
                    OhosAiTier::NnRt => "NNRt available; dedicated worker required".to_string(),
                    OhosAiTier::CannKit => "CANN Kit on supported Kirin NPU".to_string(),
                },
                status: VerificationStatus::Implemented,
            });
        }
    }
    Err("no OHOS AI tier available")
}

// ---------------------------------------------------------------------------
// Node-API async work queue model
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkKind {
    FusedPixels,
    AiInference,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkItem {
    pub id: u64,
    pub kind: WorkKind,
    /// Tier/backend id for telemetry; no native pointer is stored here.
    pub backend: String,
}

#[derive(Debug, Default)]
pub struct OhosWorkQueue {
    pending: VecDeque<WorkItem>,
    running: Vec<WorkItem>,
    next_id: u64,
    completed: u64,
}

impl OhosWorkQueue {
    pub fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            next_id: 1,
            ..Self::default()
        }
    }

    /// Enqueue one `napi_create_async_work` item. UI threads only enqueue;
    /// heavy work never runs on the ArkTS event loop.
    pub fn enqueue(&mut self, kind: WorkKind, backend: &str) -> WorkItem {
        let item = WorkItem {
            id: self.next_id,
            kind,
            backend: backend.to_string(),
        };
        self.next_id = self.next_id.wrapping_add(1);
        self.pending.push_back(item.clone());
        item
    }

    pub fn start_next(&mut self) -> Option<WorkItem> {
        let item = self.pending.pop_front()?;
        self.running.push(item.clone());
        Some(item)
    }

    pub fn finish(&mut self, id: u64) -> bool {
        if let Some(index) = self.running.iter().position(|item| item.id == id) {
            self.running.remove(index);
            self.completed += 1;
            true
        } else {
            false
        }
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    pub fn completed(&self) -> u64 {
        self.completed
    }

    pub fn clear(&mut self) {
        self.pending.clear();
        self.running.clear();
    }
}

/// NNRt blocking isolation: one dedicated worker per in-flight NNRt item,
/// never the ArkTS UI thread. The queue model is testable without OHOS.
pub fn nnrt_worker_required(kind: WorkKind, tier: Option<OhosAiTier>) -> bool {
    kind == WorkKind::AiInference && tier == Some(OhosAiTier::NnRt)
}

/// Apply an OHOS-safe resource budget. OHOS has no ADPF; the host supplies
/// thermal/memory states through its own APIs and this function only clamps.
pub fn ohos_budget(
    base: ResourceBudget,
    thermal_critical: bool,
    memory_critical: bool,
) -> ResourceBudget {
    let vram_pct = if memory_critical { 10u64 } else { 100 };
    let staging_pct = if memory_critical { 5u64 } else { 100 };
    let threads = if thermal_critical { 32u32 } else { 256 };
    ResourceBudget {
        max_vram_bytes: base.max_vram_bytes * vram_pct / 100,
        max_staging_bytes: base.max_staging_bytes * staging_pct / 100,
        max_threads_per_group: base.max_threads_per_group.min(threads),
        max_concurrent_dispatches: if thermal_critical {
            1
        } else {
            base.max_concurrent_dispatches
        },
        ..base
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ohos_vulkan_plan_uses_bounded_staging_without_native_buffer() {
        let mut probe = OhosVulkanProbe::synthetic();
        let native = plan_ohos_vulkan(&probe).unwrap();
        assert!(!native.use_bounded_staging);
        assert_eq!(native.precision, Precision::Fp16);
        let candidate = ohos_vulkan_candidate(&native);
        assert_eq!(candidate.backend, Backend::Vulkan);
        assert_eq!(candidate.kernel_entry, "aurora_fused_rgba8_ohos");

        probe.has_ohos_native_buffer_extension = false;
        let bounded = plan_ohos_vulkan(&probe).unwrap();
        assert!(bounded.use_bounded_staging);
        assert!(bounded.reason.contains("bounded_staging"));

        probe.driver_ready = false;
        assert_eq!(plan_ohos_vulkan(&probe), Err("OHOS Vulkan driver not ready"));
    }

    #[test]
    fn ai_tier_order_prioritizes_mindspore_then_nnrt_then_kirin_cann() {
        let mut info = OhosAiDeviceInfo::synthetic();
        assert_eq!(
            select_ohos_ai_tier(&info).unwrap().tier,
            OhosAiTier::MindSporeLiteKit
        );

        info.mindspore_lite_available = false;
        let nnrt = select_ohos_ai_tier(&info).unwrap();
        assert_eq!(nnrt.tier, OhosAiTier::NnRt);
        assert_eq!(nnrt.status, VerificationStatus::Implemented);
        assert!(OhosAiTier::NnRt.requires_dedicated_worker());

        info.nnrt_available = false;
        info.cann_kit_available = true;
        // CANN must never be selected without a supported Kirin NPU.
        assert_eq!(select_ohos_ai_tier(&info), Err("no OHOS AI tier available"));
        info.kirin_npu_supported = true;
        assert_eq!(
            select_ohos_ai_tier(&info).unwrap().tier,
            OhosAiTier::CannKit
        );
        assert!(OhosAiTier::CannKit.requires_kirin_npu());
        assert_eq!(OhosAiTier::MindSporeLiteKit.priority(), 1);

        info.model_verified = false;
        assert_eq!(select_ohos_ai_tier(&info), Err("OHOS model binary not verified"));
    }

    #[test]
    fn async_queue_keeps_heavy_work_off_ui_thread() {
        let mut queue = OhosWorkQueue::new();
        let pixels = queue.enqueue(WorkKind::FusedPixels, "vulkan");
        let nnrt = queue.enqueue(WorkKind::AiInference, OhosAiTier::NnRt.as_str());
        assert_eq!(queue.pending_len(), 2);
        assert!(nnrt_worker_required(nnrt.kind, Some(OhosAiTier::NnRt)));
        assert!(!nnrt_worker_required(pixels.kind, Some(OhosAiTier::NnRt)));
        assert!(!nnrt_worker_required(
            WorkKind::AiInference,
            Some(OhosAiTier::MindSporeLiteKit)
        ));

        let started = queue.start_next().unwrap();
        assert_eq!(started.id, pixels.id);
        assert_eq!(queue.running_len(), 1);
        assert!(queue.finish(started.id));
        assert!(!queue.finish(started.id));
        assert_eq!(queue.completed(), 1);

        let nnrt_item = queue.start_next().unwrap();
        assert_eq!(nnrt_item.id, nnrt.id);
        assert_eq!(queue.pending_len(), 0);
        queue.clear();
        assert_eq!(queue.running_len(), 0);
    }

    #[test]
    fn ohos_budget_clamps_without_android_hooks() {
        let base = ResourceBudget::conservative_mobile();
        let relaxed = ohos_budget(base, false, false);
        assert_eq!(relaxed.max_vram_bytes, base.max_vram_bytes);
        let critical = ohos_budget(base, true, true);
        assert!(critical.max_vram_bytes < relaxed.max_vram_bytes);
        assert!(critical.max_staging_bytes < relaxed.max_staging_bytes);
        assert_eq!(critical.max_concurrent_dispatches, 1);
        assert_eq!(OhosAiTier::NnRt.as_str(), "nnrt");
        assert_eq!(OHOS_AI_TIER_ORDER.len(), 3);
    }
}
