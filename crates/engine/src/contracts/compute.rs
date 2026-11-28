//! Compute vocabulary: backends, capability, policy, resources and dispatch.
//!
//! A backend is selected by policy, not by chance. [`select_backend`] is the
//! only policy decision in this module: it never silently downgrades a
//! `StrictGpu` request to the CPU reference, and it records why a `PreferGpu`
//! or `Auto` request fell back. A [`DispatchReceipt`] records that a dispatch
//! happened; it is not a claim about pixel quality or performance.

use std::fmt;

use serde::{Deserialize, Serialize};

use super::frame::ContractError;

/// Execution backend family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    /// The CPU reference implementation. Always available.
    Cpu,
    /// NVIDIA CUDA (Windows/Linux NVIDIA GPUs).
    Cuda,
    /// Apple Metal (macOS / iOS / iPadOS Apple Silicon).
    Metal,
    /// Cross-platform Vulkan / WGPU Compute (AMD, Intel, Qualcomm Adreno, HarmonyOS).
    Vulkan,
    /// Windows Direct3D 12 Compute (NVIDIA, AMD, Intel, Windows on ARM).
    Dx12,
    /// AMD HIP / ROCm Compute (RDNA dGPU & APU).
    Hip,
    /// Intel oneAPI SYCL / Level Zero Compute.
    Sycl,
    /// OpenCL, kept for adapters that expose it.
    OpenCl,
    /// Reserved NPU accelerator family (Qualcomm Hexagon, Apple ANE, Intel NPU, HarmonyOS NNRt/CANN).
    Npu,
}

impl BackendKind {
    /// Whether this backend is a GPU compute backend.
    pub const fn is_gpu(self) -> bool {
        matches!(
            self,
            Self::Cuda | Self::Metal | Self::Vulkan | Self::Dx12 | Self::Hip | Self::Sycl | Self::OpenCl
        )
    }

    /// Whether this backend offloads work off the CPU (GPU or NPU).
    pub const fn is_accelerator(self) -> bool {
        !matches!(self, Self::Cpu)
    }

    /// Stable lowercase identifier for diagnostics, logs and distribution reports.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::Vulkan => "vulkan",
            Self::Dx12 => "dx12",
            Self::Hip => "hip",
            Self::Sycl => "sycl",
            Self::OpenCl => "opencl",
            Self::Npu => "npu",
        }
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// How strongly a caller prefers a GPU backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendPolicy {
    /// Prefer an available GPU, but allow the CPU reference and say why.
    Auto,
    /// Use the CPU reference, never a GPU.
    CpuOnly,
    /// Prefer an available GPU, but allow the CPU reference and say why.
    PreferGpu,
    /// Require a GPU; refuse rather than run on the CPU.
    StrictGpu,
}

/// Numeric precision a backend can execute.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// IEEE-754 binary32 throughout.
    F32,
    /// IEEE-754 binary16 storage with the adapter's accumulation policy.
    F16,
    /// Mixed precision: for example F16 storage with F32 accumulation.
    Mixed,
}

/// One backend's reported capability.
///
/// `available` is the adapter's own report, not a probe performed here.
/// `compute_capability` is meaningful for CUDA; other adapters leave it `None`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BackendCapability {
    /// Backend family.
    pub kind: BackendKind,
    /// Whether the adapter can execute on this machine right now.
    pub available: bool,
    /// Adapter-reported device name, if any.
    pub device: Option<String>,
    /// CUDA compute capability `(major, minor)`, if reported.
    pub compute_capability: Option<(i32, i32)>,
    /// Numeric precision the adapter reports.
    pub precision: Precision,
}

/// The outcome of applying a [`BackendPolicy`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct BackendSelection {
    /// Backend that will execute.
    pub backend: BackendKind,
    /// Why the CPU reference was used instead of the preferred GPU.
    pub fallback: Option<FallbackReason>,
}

/// Typed refusal from [`select_backend`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SelectionError {
    /// A `StrictGpu` request found no available GPU backend.
    NoGpuAvailable { policy: BackendPolicy },
}

impl fmt::Display for SelectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoGpuAvailable { policy } => write!(
                f,
                "policy {policy:?} requires a GPU, but no available GPU backend was reported"
            ),
        }
    }
}

impl std::error::Error for SelectionError {}

/// Select a backend under `policy`.
///
/// - `CpuOnly` always returns [`BackendKind::Cpu`] with no fallback. The CPU
///   reference is the definitional implementation and is always considered
///   available; a `Cpu` entry in `capabilities` is informational only.
/// - `StrictGpu` returns the first available GPU capability in slice order, or
///   [`SelectionError::NoGpuAvailable`]. It never returns `Cpu`.
/// - `PreferGpu` and `Auto` return the first available GPU if there is one;
///   otherwise they return `Cpu` with [`FallbackReason::Unsupported`], so the
///   fallback is never silent.
///
/// "First available GPU in slice order" is deliberate: the caller lists its
/// adapters in preference order.
pub fn select_backend(
    policy: BackendPolicy,
    capabilities: &[BackendCapability],
) -> Result<BackendSelection, SelectionError> {
    match policy {
        BackendPolicy::CpuOnly => Ok(BackendSelection {
            backend: BackendKind::Cpu,
            fallback: None,
        }),
        BackendPolicy::StrictGpu => capabilities
            .iter()
            .find(|capability| capability.kind.is_gpu() && capability.available)
            .map(|capability| BackendSelection {
                backend: capability.kind,
                fallback: None,
            })
            .ok_or(SelectionError::NoGpuAvailable { policy }),
        BackendPolicy::PreferGpu | BackendPolicy::Auto => {
            match capabilities
                .iter()
                .find(|capability| capability.kind.is_gpu() && capability.available)
            {
                Some(capability) => Ok(BackendSelection {
                    backend: capability.kind,
                    fallback: None,
                }),
                None => Ok(BackendSelection {
                    backend: BackendKind::Cpu,
                    fallback: Some(FallbackReason::Unsupported),
                }),
            }
        }
    }
}

/// Structured capability diagnostic suitable for platform test builds and UI
/// telemetry so every distribution truthfully reports its active compute
/// backend and any fallback reason.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct BackendDiagnostic {
    /// Policy that was evaluated.
    pub policy: BackendPolicy,
    /// Backend selected to run, or `None` when `StrictGpu` refused.
    pub selected: Option<BackendKind>,
    /// Fallback reason when an accelerated backend could not be used.
    pub fallback: Option<FallbackReason>,
    /// All backends reported as available on this host.
    pub available_backends: Vec<BackendKind>,
    /// All backends explicitly reported as unavailable on this host.
    pub unavailable_backends: Vec<BackendKind>,
    /// Human-readable one-line summary for logs and build verification reports.
    pub summary: String,
}

/// Evaluate `policy` against `capabilities` and emit a structured diagnostic.
pub fn diagnose_backend_selection(
    policy: BackendPolicy,
    capabilities: &[BackendCapability],
) -> BackendDiagnostic {
    let available_backends: Vec<BackendKind> = capabilities
        .iter()
        .filter(|cap| cap.available)
        .map(|cap| cap.kind)
        .collect();
    let unavailable_backends: Vec<BackendKind> = capabilities
        .iter()
        .filter(|cap| !cap.available)
        .map(|cap| cap.kind)
        .collect();

    match select_backend(policy, capabilities) {
        Ok(selection) => {
            let summary = match selection.fallback {
                Some(reason) => format!(
                    "policy={policy:?} selected={} fallback=\"{reason}\"",
                    selection.backend
                ),
                None => format!("policy={policy:?} selected={}", selection.backend),
            };
            BackendDiagnostic {
                policy,
                selected: Some(selection.backend),
                fallback: selection.fallback,
                available_backends,
                unavailable_backends,
                summary,
            }
        }
        Err(err) => BackendDiagnostic {
            policy,
            selected: None,
            fallback: Some(FallbackReason::Unsupported),
            available_backends,
            unavailable_backends,
            summary: format!("policy={policy:?} refused=\"{err}\""),
        },
    }
}

/// A stable identifier for one pipeline operation.
///
/// The identifier is a compile-time string rather than a global counter, so
/// plans and receipts from different processes can be compared directly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize)]
#[serde(transparent)]
pub struct OperationId(&'static str);

impl OperationId {
    /// Construct an operation identifier.
    pub const fn new(id: &'static str) -> Self {
        Self(id)
    }

    /// The identifier as a string slice.
    pub const fn as_str(self) -> &'static str {
        self.0
    }
}

impl fmt::Display for OperationId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.0)
    }
}

/// Host and device memory a plan expects to use, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceRequest {
    /// Pageable host bytes.
    pub cpu_bytes: u64,
    /// Device-local bytes.
    pub device_bytes: u64,
    /// Page-locked host bytes, counted in addition to `cpu_bytes`.
    pub pinned_bytes: u64,
}

impl ResourceRequest {
    /// Total host bytes, pageable plus pinned, with checked addition.
    ///
    /// Returns [`ContractError::Overflow`] instead of wrapping. `device_bytes`
    /// is excluded because it is not host memory.
    pub fn total_host_bytes(&self) -> Result<u64, ContractError> {
        self.cpu_bytes
            .checked_add(self.pinned_bytes)
            .ok_or(ContractError::Overflow {
                what: "total_host_bytes",
            })
    }
}

/// Why a preferred backend was not used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FallbackReason {
    /// The backend is unsupported or unavailable on this machine.
    Unsupported,
    /// The named kernel is missing from the adapter.
    MissingKernel(&'static str),
    /// The named model is missing.
    MissingModel(&'static str),
    /// A required input is missing.
    MissingInput(&'static str),
    /// The backend cannot run at the requested precision.
    PrecisionUnsupported,
    /// The request needs more memory than the budget allows.
    BudgetExceeded {
        /// Bytes required.
        need: u64,
        /// Bytes allowed.
        budget: u64,
    },
    /// The device was lost during execution.
    DeviceLost,
    /// The backend returned an execution error.
    ExecutionError,
}

impl fmt::Display for FallbackReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => f.write_str("no supported backend was available"),
            Self::MissingKernel(kernel) => write!(f, "kernel {kernel} is unavailable"),
            Self::MissingModel(model) => write!(f, "model {model} is missing"),
            Self::MissingInput(input) => write!(f, "input {input} is missing"),
            Self::PrecisionUnsupported => {
                f.write_str("the backend does not support the requested precision")
            }
            Self::BudgetExceeded { need, budget } => write!(
                f,
                "resource budget exceeded: {need} bytes needed, {budget} bytes allowed"
            ),
            Self::DeviceLost => f.write_str("the device was lost during execution"),
            Self::ExecutionError => f.write_str("backend execution failed"),
        }
    }
}

impl std::error::Error for FallbackReason {}

/// A backend-neutral plan: the operations to run and where they will run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ExecutionPlan {
    /// Operations in execution order.
    pub operations: Vec<OperationId>,
    /// Selected backend.
    pub backend: BackendKind,
    /// Why the CPU reference is used instead of the preferred GPU.
    pub fallback: Option<FallbackReason>,
}

/// A backend-neutral record of one dispatch.
///
/// A receipt proves that the adapter issued and completed a dispatch for
/// `operation` on `backend`. It does not prove numerical parity, image quality
/// or performance; those require separate comparison evidence.
///
/// `grid` and `block` mirror the existing GPU launch fields
/// (`(u32, u32)` dimensions) but their exact meaning is backend-defined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct DispatchReceipt {
    /// Operation that was dispatched.
    pub operation: OperationId,
    /// Backend that executed it.
    pub backend: BackendKind,
    /// Kernel entry point, when the backend has named kernels.
    pub kernel: Option<&'static str>,
    /// Launch grid dimensions, when applicable.
    pub grid: Option<(u32, u32)>,
    /// Launch block dimensions, when applicable.
    pub block: Option<(u32, u32)>,
    /// Host-to-device bytes transferred for this dispatch.
    pub host_to_device_bytes: u64,
    /// Device-to-host bytes transferred for this dispatch.
    pub device_to_host_bytes: u64,
    /// Whether the dispatch completed; `false` means it was refused or failed.
    pub completed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn capability(kind: BackendKind, available: bool) -> BackendCapability {
        BackendCapability {
            kind,
            available,
            device: None,
            compute_capability: None,
            precision: Precision::F32,
        }
    }

    #[test]
    fn cpu_only_never_selects_a_gpu() {
        let capabilities = [
            capability(BackendKind::Cuda, true),
            capability(BackendKind::Metal, true),
            capability(BackendKind::Cpu, true),
        ];
        assert_eq!(
            select_backend(BackendPolicy::CpuOnly, &capabilities),
            Ok(BackendSelection {
                backend: BackendKind::Cpu,
                fallback: None,
            })
        );
    }

    #[test]
    fn strict_gpu_refuses_instead_of_returning_cpu() {
        let only_cpu = [capability(BackendKind::Cpu, true)];
        assert_eq!(
            select_backend(BackendPolicy::StrictGpu, &only_cpu),
            Err(SelectionError::NoGpuAvailable {
                policy: BackendPolicy::StrictGpu
            })
        );

        let unavailable_gpu = [
            capability(BackendKind::Cuda, false),
            capability(BackendKind::Cpu, true),
        ];
        assert_eq!(
            select_backend(BackendPolicy::StrictGpu, &unavailable_gpu),
            Err(SelectionError::NoGpuAvailable {
                policy: BackendPolicy::StrictGpu
            })
        );

        let with_gpu = [
            capability(BackendKind::Cuda, true),
            capability(BackendKind::Cpu, true),
        ];
        assert_eq!(
            select_backend(BackendPolicy::StrictGpu, &with_gpu),
            Ok(BackendSelection {
                backend: BackendKind::Cuda,
                fallback: None,
            })
        );
    }

    #[test]
    fn prefer_gpu_falls_back_with_a_reason() {
        let only_cpu = [capability(BackendKind::Cpu, true)];
        let selection =
            select_backend(BackendPolicy::PreferGpu, &only_cpu).expect("CPU fallback is allowed");
        assert_eq!(selection.backend, BackendKind::Cpu);
        assert_eq!(selection.fallback, Some(FallbackReason::Unsupported));

        let unavailable_gpu = [
            capability(BackendKind::Metal, false),
            capability(BackendKind::Cpu, true),
        ];
        let selection = select_backend(BackendPolicy::PreferGpu, &unavailable_gpu)
            .expect("CPU fallback is allowed");
        assert_eq!(selection.backend, BackendKind::Cpu);
        assert!(selection.fallback.is_some());
    }

    #[test]
    fn auto_prefers_an_available_gpu() {
        let capabilities = [
            capability(BackendKind::Metal, true),
            capability(BackendKind::Cuda, true),
            capability(BackendKind::Cpu, true),
        ];
        assert_eq!(
            select_backend(BackendPolicy::Auto, &capabilities),
            Ok(BackendSelection {
                backend: BackendKind::Metal,
                fallback: None,
            })
        );

        let only_cpu = [capability(BackendKind::Cpu, true)];
        let selection =
            select_backend(BackendPolicy::Auto, &only_cpu).expect("CPU fallback is allowed");
        assert_eq!(selection.backend, BackendKind::Cpu);
        assert_eq!(selection.fallback, Some(FallbackReason::Unsupported));
    }

    #[test]
    fn resource_totals_use_checked_addition() {
        let request = ResourceRequest {
            cpu_bytes: 10,
            device_bytes: 20,
            pinned_bytes: 30,
        };
        assert_eq!(request.total_host_bytes(), Ok(40));

        let request = ResourceRequest {
            cpu_bytes: u64::MAX,
            device_bytes: 0,
            pinned_bytes: 1,
        };
        assert_eq!(
            request.total_host_bytes(),
            Err(ContractError::Overflow {
                what: "total_host_bytes"
            })
        );
    }

    #[test]
    fn operation_ids_are_compile_time_and_serialize_transparently() {
        let operation = OperationId::new("detail");
        assert_eq!(operation.as_str(), "detail");
        assert_eq!(operation.to_string(), "detail");
        assert_eq!(
            serde_json::to_string(&operation).expect("serialize op"),
            "\"detail\""
        );
    }

    #[test]
    fn dispatch_receipt_is_adapter_neutral_and_copy() {
        let receipt = DispatchReceipt {
            operation: OperationId::new("detail"),
            backend: BackendKind::Cuda,
            kernel: Some("detail_kernel"),
            grid: Some((8, 4)),
            block: Some((16, 16)),
            host_to_device_bytes: 1024,
            device_to_host_bytes: 512,
            completed: true,
        };
        assert!(receipt.completed);
        let copy = receipt;
        assert_eq!(copy, receipt);
        let json = serde_json::to_string(&receipt).expect("serialize receipt");
        assert!(json.contains("detail_kernel"));
    }

    #[test]
    fn fallback_reasons_display_their_cause() {
        assert_eq!(
            FallbackReason::MissingKernel("nr").to_string(),
            "kernel nr is unavailable"
        );
        assert_eq!(
            FallbackReason::BudgetExceeded {
                need: 64,
                budget: 32
            }
            .to_string(),
            "resource budget exceeded: 64 bytes needed, 32 bytes allowed"
        );
    }

    #[test]
    fn vulkan_and_npu_backend_kinds_and_diagnostics() {
        assert!(BackendKind::Vulkan.is_gpu());
        assert!(BackendKind::Vulkan.is_accelerator());
        assert_eq!(BackendKind::Vulkan.as_str(), "vulkan");

        assert!(!BackendKind::Npu.is_gpu());
        assert!(BackendKind::Npu.is_accelerator());
        assert_eq!(BackendKind::Npu.as_str(), "npu");

        let caps = [
            capability(BackendKind::Vulkan, true),
            capability(BackendKind::Cpu, true),
        ];
        let diag = diagnose_backend_selection(BackendPolicy::Auto, &caps);
        assert_eq!(diag.selected, Some(BackendKind::Vulkan));
        assert_eq!(diag.fallback, None);
        assert!(diag.summary.contains("selected=vulkan"));
    }
}
