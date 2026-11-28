//! Unified compute capability and policy layer (roadmap §8, milestone M1).
//!
//! # What this is
//!
//! The pre-M1 tree has no unified "where does this run?" entry point. It has
//! three unrelated `Backend` enums, six independent stage `enabled()` functions,
//! and more than a hundred `AURORA_*` variables that each call site interprets
//! on its own. [`policy::decide`] is the one decision surface phase 2 will route
//! those call sites through.
//!
//! # What this is not
//!
//! * It does not dispatch anything.
//! * It does not call, wrap, or modify any existing `enabled()`.
//! * It does not probe hardware; [`device::Capabilities`] is supplied by the
//!   caller from probes that already exist.
//! * It does not write the environment. The snapshot is read once and is
//!   immutable for the process lifetime.
//!
//! # Phase 1 status
//!
//! No product call site consults this module yet, so the shipped behaviour is
//! byte-for-byte unchanged. The unit tests decide four modes across the
//! workload table without a GPU or ONNX Runtime.
//!
//! # Phase 2 hand-off
//!
//! `06_reports/ROADMAP-M1-SCHED-20261125.md` lists the exact `file:line`
//! call sites to change. In short: each stage's `enabled()` starts with
//! `policy::decide(&workload, mode)` and returns `plan.backend.is_cpu()` only;
//! call sites that need provenance use `plan.reason`. `PreferGpu` must be the
//! default that reproduces today's "try GPU, fall back to CPU" behaviour, and
//! `Auto` must reproduce the current env defaults exactly.
//!
//! # Example
//!
//! ```
//! use aurora_engine::compute::{policy, ComputeMode, Operation, Workload};
//!
//! // Phase 1 has no installed capability probe, so this resolves to CPU with
//! // an explicit "no device" reason. Never a silent fallback.
//! let plan = policy::decide(&Workload::new(Operation::Develop, 24_000_000), ComputeMode::CpuOnly);
//! assert!(plan.backend.is_cpu());
//! ```

pub mod device;
pub mod mode;
pub mod policy;
pub mod telemetry;

pub use device::{
    Backend, BackendKind, Capabilities, GpuInfo, Precision, Priority, Residency,
};
pub use mode::ComputeMode;
pub use policy::{
    decide, decide_with, install_capabilities, ExecutionPlan, Operation, PolicyInputs, Reason,
    StageGate, Workload,
};
pub use telemetry::{BuildProfile, DecisionCounters, EnvSnapshot, Platform};

/// Takes the one-time read-only environment snapshot and fixes the
/// process-wide policy inputs.
///
/// Call once from the app's startup path, after installing capability probe
/// results with [`install_capabilities`] but before the first preview. Calling
/// it later is harmless: the first call wins, matching the existing
/// `OnceLock`-based switches (`vram_budget`, `crossover_floor`).
pub fn init() {
    let _ = telemetry::snapshot();
    crate::platform_dispatch::install_host_capabilities();
    let _ = PolicyInputs::capture();
}
