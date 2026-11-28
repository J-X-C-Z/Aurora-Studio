//! Shared AutoTune interface, candidate registry types, execution policy, and
//! verification status vocabulary (`DESIGNED` -> `PERF_OK`) for all compute
//! and AI inference backends.
//!
//! This module does **not** implement the AutoTune search engine itself; it
//! exposes bounded parameter spaces, strict vs. fallback dispatch rules,
//! resource budgets, timing receipts, and hardware probe records so the
//! standalone AutoTune subsystem and capability matrix consume a uniform ABI.

use std::fmt;

use crate::device::{Backend, Precision};

/// Verification ladder mandated by the platform adaptation specification.
/// Each level must be backed by real evidence; skipped tests or CPU fallbacks
/// must never be promoted to a higher hardware status.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum VerificationStatus {
    Unsupported,
    Unverified,
    Designed,
    Implemented,
    Compiled,
    SyntheticOk,
    DeviceOk,
    ParityOk,
    PhotoOk,
    ProductionWired,
    PerfOk,
}

impl VerificationStatus {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unsupported => "UNSUPPORTED",
            Self::Unverified => "UNVERIFIED",
            Self::Designed => "DESIGNED",
            Self::Implemented => "IMPLEMENTED",
            Self::Compiled => "COMPILED",
            Self::SyntheticOk => "SYNTHETIC_OK",
            Self::DeviceOk => "DEVICE_OK",
            Self::ParityOk => "PARITY_OK",
            Self::PhotoOk => "PHOTO_OK",
            Self::ProductionWired => "PRODUCTION_WIRED",
            Self::PerfOk => "PERF_OK",
        }
    }
}

impl fmt::Display for VerificationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Strictness policy for candidate dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ExecutionMode {
    /// Fail explicitly if the requested backend/candidate cannot execute.
    Strict,
    /// Allow controlled fallback to another verified backend and record why.
    AutoFallback,
}

/// Bounded tunable parameter exposed by a backend candidate to AutoTune.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParamBound {
    U32Range {
        name: &'static str,
        min: u32,
        max: u32,
        step: u32,
    },
    DiscreteU32 {
        name: &'static str,
        values: &'static [u32],
    },
    BoolFlag {
        name: &'static str,
    },
    EnumChoice {
        name: &'static str,
        choices: &'static [&'static str],
    },
}

impl ParamBound {
    pub const fn name(&self) -> &'static str {
        match self {
            Self::U32Range { name, .. }
            | Self::DiscreteU32 { name, .. }
            | Self::BoolFlag { name }
            | Self::EnumChoice { name, .. } => name,
        }
    }

    /// Validates that an integer parameter falls within declared bounds.
    pub fn accepts_u32(&self, value: u32) -> bool {
        match self {
            Self::U32Range { min, max, step, .. } => {
                value >= *min && value <= *max && (*step == 0 || (value - *min) % *step == 0)
            }
            Self::DiscreteU32 { values, .. } => values.contains(&value),
            Self::BoolFlag { .. } => value <= 1,
            Self::EnumChoice { choices, .. } => (value as usize) < choices.len(),
        }
    }
}

/// Resource limits enforced on candidate execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResourceBudget {
    pub max_vram_bytes: u64,
    pub max_staging_bytes: u64,
    pub max_shared_mem_bytes: u32,
    pub max_threads_per_group: u32,
    pub max_concurrent_dispatches: u32,
}

impl ResourceBudget {
    pub const fn conservative_mobile() -> Self {
        Self {
            max_vram_bytes: 512 * 1024 * 1024,
            max_staging_bytes: 64 * 1024 * 1024,
            max_shared_mem_bytes: 16 * 1024,
            max_threads_per_group: 256,
            max_concurrent_dispatches: 2,
        }
    }

    pub const fn desktop_default() -> Self {
        Self {
            max_vram_bytes: 2048 * 1024 * 1024,
            max_staging_bytes: 256 * 1024 * 1024,
            max_shared_mem_bytes: 48 * 1024,
            max_threads_per_group: 1024,
            max_concurrent_dispatches: 8,
        }
    }
}

/// Static descriptor of one tunable kernel/operator candidate variant.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CandidateSpec {
    pub id: &'static str,
    pub algorithm_id: &'static str,
    pub backend: Backend,
    pub kernel_entry: &'static str,
    pub precision: Precision,
    pub params: &'static [ParamBound],
    pub max_shared_mem_bytes: u32,
    pub max_threads_per_group: u32,
    pub status: VerificationStatus,
}

impl CandidateSpec {
    /// Checks whether this candidate fits inside the given hardware budget.
    pub const fn fits_budget(&self, budget: &ResourceBudget) -> bool {
        self.max_shared_mem_bytes <= budget.max_shared_mem_bytes
            && self.max_threads_per_group <= budget.max_threads_per_group
    }
}

/// Fine-grained execution timing and data movement telemetry for one dispatch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BackendTiming {
    pub queue_submit_ns: u64,
    pub device_exec_ns: u64,
    pub wall_ns: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
}

/// Detailed hardware capability probe exported for capability/hardware matrices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdapterProbeRecord {
    pub backend: Backend,
    pub platform: &'static str,
    pub device_name: String,
    pub driver_version: String,
    pub api_version: String,
    pub driver_ready: bool,
    pub supports_fp32: bool,
    pub supports_fp16: bool,
    pub supports_int8: bool,
    pub supports_subgroup: bool,
    pub max_workgroup_size: [u32; 3],
    pub max_shared_mem_bytes: u32,
    pub vram_bytes: u64,
    pub status: VerificationStatus,
    pub notes: String,
}

/// Percentile summary computed over warm benchmark samples.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TimingSummary {
    pub samples: usize,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

impl TimingSummary {
    pub fn from_nanos(samples_ns: &[u64]) -> Option<Self> {
        if samples_ns.is_empty() {
            return None;
        }
        let mut sorted: Vec<u64> = samples_ns.to_vec();
        sorted.sort_unstable();
        let pct = |p: f64| -> f64 {
            let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
            sorted[idx.min(sorted.len() - 1)] as f64 / 1_000_000.0
        };
        Some(Self {
            samples: sorted.len(),
            p50_ms: pct(0.50),
            p95_ms: pct(0.95),
            p99_ms: pct(0.99),
            min_ms: sorted[0] as f64 / 1_000_000.0,
            max_ms: sorted[sorted.len() - 1] as f64 / 1_000_000.0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_bounds_validate_ranges_and_discrete_sets() {
        let wg = ParamBound::DiscreteU32 {
            name: "workgroup_x",
            values: &[64, 128, 256],
        };
        assert!(wg.accepts_u32(128));
        assert!(!wg.accepts_u32(96));

        let tile = ParamBound::U32Range {
            name: "tile_size",
            min: 64,
            max: 512,
            step: 64,
        };
        assert!(tile.accepts_u32(256));
        assert!(!tile.accepts_u32(100));
        assert!(!tile.accepts_u32(1024));
    }

    #[test]
    fn timing_summary_percentiles_are_ordered() {
        let s = TimingSummary::from_nanos(&[5_000_000, 1_000_000, 3_000_000, 2_000_000, 4_000_000])
            .unwrap();
        assert_eq!(s.samples, 5);
        assert!((s.min_ms - 1.0).abs() < 1e-6);
        assert!((s.p50_ms - 3.0).abs() < 1e-6);
        assert!((s.max_ms - 5.0).abs() < 1e-6);
    }
}
