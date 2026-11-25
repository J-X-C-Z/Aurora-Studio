//! Product-level compute mode.
//!
//! There is no unified mode in the pre-M1 tree: `Auto` only exists inside
//! `aurora_portrait::effect_router`, `CpuOnly` can only be approximated with a
//! pile of environment variables, `PreferGpu` is the undocumented default of
//! every stage, and `StrictGpu` exists only as scattered hard-fail fragments.
//! This type is the single declaration phase 2 will map onto those switches.
//!
//! Parsing is additive: no call site reads [`ComputeMode`] yet, so introducing
//! the type cannot change existing behaviour.

use serde::{Deserialize, Serialize};
use std::fmt;

/// How the product wants CPU and GPU work distributed.
///
/// The four modes are ordered from most restrictive CPU use to most strict GPU
/// use. `CpuOnly` is absolute: no policy decision may return a GPU backend.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeMode {
    /// Let the policy decide per workload, using the stage default and the
    /// existing cost seeds. This is the shipped behaviour today.
    #[default]
    Auto,
    /// Never dispatch compute to a GPU. Every non-CPU candidate becomes `Cpu`
    /// with [`crate::compute::policy::Reason::CpuOnlyMode`].
    CpuOnly,
    /// Use a GPU whenever a capable device exists, even below the measured
    /// crossover; keep the CPU as an explicit fallback.
    PreferGpu,
    /// A GPU is required. CPU-only operations are still allowed (they are not
    /// violations); a GPU-capable operation with no usable device is a hard
    /// failure for the caller to surface.
    StrictGpu,
}

impl ComputeMode {
    /// Every mode, in declaration order, for UI enumeration and tests.
    pub const ALL: [ComputeMode; 4] = [
        ComputeMode::Auto,
        ComputeMode::CpuOnly,
        ComputeMode::PreferGpu,
        ComputeMode::StrictGpu,
    ];

    /// Stable machine name. Used by settings serialization and telemetry.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::CpuOnly => "cpu_only",
            Self::PreferGpu => "prefer_gpu",
            Self::StrictGpu => "strict_gpu",
        }
    }

    /// Human label for a settings dropdown.
    pub fn label(self) -> &'static str {
        match self {
            Self::Auto => "Auto",
            Self::CpuOnly => "CPU only",
            Self::PreferGpu => "Prefer GPU",
            Self::StrictGpu => "GPU only (strict)",
        }
    }

    /// Parses a settings value. Accepts `snake_case`, `kebab-case`, and the
    /// concatenated spelling, case-insensitively. Unknown values are `None`
    /// so a corrupted settings file can fall back to [`ComputeMode::Auto`].
    pub fn parse(raw: &str) -> Option<Self> {
        let normal = raw.trim().to_ascii_lowercase();
        match normal.as_str() {
            "auto" => Some(Self::Auto),
            "cpu_only" | "cpu-only" | "cpuonly" | "cpu" => Some(Self::CpuOnly),
            "prefer_gpu" | "prefer-gpu" | "prefergpu" | "gpu_preferred" => Some(Self::PreferGpu),
            "strict_gpu" | "strict-gpu" | "strictgpu" | "gpu_only" | "gpu-only" => {
                Some(Self::StrictGpu)
            }
            _ => None,
        }
    }

    /// `true` for the two modes that may return a GPU backend.
    pub fn permits_gpu(self) -> bool {
        !matches!(self, Self::CpuOnly)
    }
}

impl fmt::Display for ComputeMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_round_trips_every_mode() {
        for mode in ComputeMode::ALL {
            assert_eq!(ComputeMode::parse(mode.as_str()), Some(mode));
        }
    }

    #[test]
    fn parse_accepts_ui_spellings() {
        assert_eq!(ComputeMode::parse(" CPU-ONLY "), Some(ComputeMode::CpuOnly));
        assert_eq!(ComputeMode::parse("PreferGpu"), Some(ComputeMode::PreferGpu));
        assert_eq!(ComputeMode::parse("gpu-only"), Some(ComputeMode::StrictGpu));
        assert_eq!(ComputeMode::parse("nothing"), None);
    }

    #[test]
    fn only_cpu_only_forbids_gpu() {
        assert!(!ComputeMode::CpuOnly.permits_gpu());
        assert!(ComputeMode::Auto.permits_gpu());
        assert!(ComputeMode::PreferGpu.permits_gpu());
        assert!(ComputeMode::StrictGpu.permits_gpu());
    }
}
