//! Read-only environment snapshot and decision telemetry.
//!
//! Roadmap §8 requires that every existing `AURORA_*` switch be readable from
//! one place, without changing when or how the switches are evaluated. This
//! module captures the process environment **once** on first use and exposes
//! typed accessors over that immutable snapshot. It never calls `set_var` or
//! `remove_var`: a mode change in phase 2 must be delivered through
//! [`crate::compute::policy::PolicyInputs`], not by rewriting the environment.
//!
//! The snapshot stores every `AURORA_*` variable (not only the ones the policy
//! interprets today) so adding a stage later does not require a second read of
//! a process environment that may already have diverged. [`TRACKED_ENV`] names
//! the subset consumed by [`crate::compute::policy`].
//!
//! Flag parsing deliberately mirrors the existing call sites exactly:
//!
//! * `0` / `false` means off, any other value means on
//!   ([`EnvSnapshot::flag_false_off`]) — `develop_gpu`, `pixel_gpu`;
//! * only `0` means off, `false` still means on
//!   ([`EnvSnapshot::flag_zero_off`]) — `nr_gpu`, `clarity_gpu`;
//! * only `1` means on ([`EnvSnapshot::exact_one`]) — `detail_gpu`,
//!   `blemish_gainmap`.
//!
//! No accessor trims its value, matching the product switches.

use std::collections::BTreeMap;
use std::ffi::OsString;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Compute-selection variables interpreted by [`crate::compute::policy`].
///
/// This list is documentation and a future audit seed; the snapshot captures
/// every `AURORA_*` variable regardless.
pub const TRACKED_ENV: &[&str] = &[
    "AURORA_DEVELOP_GPU",
    "AURORA_DEVELOP_MIN_PIXELS",
    "AURORA_NR_GPU",
    "AURORA_CLARITY_GPU",
    "AURORA_DETAIL_GPU",
    "AURORA_PIXEL_GPU",
    "AURORA_PIXEL_VIGNETTE_GPU",
    "AURORA_PIXEL_LCC_GPU",
    "AURORA_PIXEL_GRAIN_GPU",
    "AURORA_PIXEL_LCC_MIN_OUT_RATIO",
    "AURORA_VRAM_BUDGET_MB",
    "AURORA_BLEMISH_GPU",
    "AURORA_BLEMISH_GAINMAP",
    "AURORA_PORTRAIT_RENDER_BACKEND",
    "AURORA_PORTRAIT_FORCE_CPU",
    "AURORA_PORTRAIT_P15_CUDA",
    "AURORA_AI_PROVIDER",
    "AURORA_ORT_PROVIDERS",
    "AURORA_ORT_OPT_LEVEL",
    "AURORA_TRT_CACHE_DIR",
    "AURORA_FACE_INPUT_BACKEND",
    "AURORA_FACE_PARSING_MASK_BACKEND",
    "AURORA_C1_RAW",
    "AURORA_DEMOSAIC",
    "AURORA_GPU_SELFCHECK",
    "AURORA_PERF",
];

/// OS family, captured once. The policy never calls `cfg!` directly so a unit
/// test can exercise the Windows and macOS default tables on any host.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Platform {
    Windows,
    MacOs,
    Linux,
    Other,
}

impl Platform {
    /// The host platform, decided at compile time.
    pub const fn current() -> Self {
        if cfg!(target_os = "windows") {
            Self::Windows
        } else if cfg!(target_os = "macos") {
            Self::MacOs
        } else if cfg!(target_os = "linux") {
            Self::Linux
        } else {
            Self::Other
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Windows => "windows",
            Self::MacOs => "macos",
            Self::Linux => "linux",
            Self::Other => "other",
        }
    }
}

/// Build profile. `nr_gpu::enabled()` currently changes its default under
/// `cfg(test)`; the policy records that difference instead of hiding it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BuildProfile {
    Release,
    Test,
}

impl BuildProfile {
    pub const fn current() -> Self {
        if cfg!(test) {
            Self::Test
        } else {
            Self::Release
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Release => "release",
            Self::Test => "test",
        }
    }
}

/// One immutable copy of the process environment's `AURORA_*` variables.
///
/// Construct with [`EnvSnapshot::capture`] in production or
/// [`EnvSnapshot::from_pairs`] in tests. Cloning is cheap enough for a policy
/// input and keeps `decide_with` free of global state.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EnvSnapshot {
    values: BTreeMap<String, String>,
}

impl EnvSnapshot {
    /// Reads the current process environment once. Call [`snapshot`] instead of
    /// this function when the process-wide singleton is wanted.
    pub fn capture() -> Self {
        Self::capture_from(std::env::vars_os())
    }

    /// Captures the `AURORA_*` subset of an arbitrary environment iterator.
    pub fn capture_from<I>(vars: I) -> Self
    where
        I: IntoIterator<Item = (OsString, OsString)>,
    {
        let mut values = BTreeMap::new();
        for (name, value) in vars {
            let name = name.to_string_lossy();
            if name.starts_with("AURORA_") {
                values.insert(name.into_owned(), value.to_string_lossy().into_owned());
            }
        }
        Self { values }
    }

    /// Test constructor. Empty slices need no special handling because the
    /// element type is part of the signature.
    pub fn from_pairs(pairs: &[(&str, &str)]) -> Self {
        Self {
            values: pairs
                .iter()
                .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
                .collect(),
        }
    }

    /// Raw value, or `None` when the variable is unset.
    pub fn var(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(String::as_str)
    }

    /// `true` when the variable exists, including when its value is empty.
    pub fn is_set(&self, name: &str) -> bool {
        self.values.contains_key(name)
    }

    /// `0` / `false` means off, any other value means on. Unset is `None`.
    pub fn flag_false_off(&self, name: &str) -> Option<bool> {
        self.var(name)
            .map(|value| !(value == "0" || value.eq_ignore_ascii_case("false")))
    }

    /// Only `0` means off; `false` still means on. Unset is `None`.
    pub fn flag_zero_off(&self, name: &str) -> Option<bool> {
        self.var(name).map(|value| value != "0")
    }

    /// Only `1` means on; every other value means off. Unset is `None`.
    pub fn exact_one(&self, name: &str) -> Option<bool> {
        self.var(name).map(|value| value == "1")
    }

    /// Trims and parses an unsigned integer. Unset or malformed values are
    /// `None`, matching the existing `filter_map`-style call sites.
    pub fn parse_u64(&self, name: &str) -> Option<u64> {
        self.var(name)?.trim().parse().ok()
    }

    /// Trims and parses a float. Unset or malformed values are `None`.
    pub fn parse_f64(&self, name: &str) -> Option<f64> {
        self.var(name)?.trim().parse().ok()
    }

    /// Iterates the captured pairs in name order.
    pub fn entries(&self) -> impl Iterator<Item = (&str, &str)> {
        self.values
            .iter()
            .map(|(name, value)| (name.as_str(), value.as_str()))
    }
}

/// Process-wide read-only snapshot. The first call is the only read; every
/// later decision sees the same values even if something else changes the
/// process environment.
pub fn snapshot() -> &'static EnvSnapshot {
    static SNAPSHOT: OnceLock<EnvSnapshot> = OnceLock::new();
    SNAPSHOT.get_or_init(EnvSnapshot::capture)
}

/// Process-wide decision counters.
///
/// The policy layer is not wired into any call site in phase 1, so these stay
/// at zero until phase 2 calls [`record_decision`]. They use relaxed atomics:
/// the counters are diagnostics, not ordering.
#[derive(Debug)]
struct Counters {
    decided: AtomicU64,
    gpu: AtomicU64,
    cpu: AtomicU64,
    strict_violations: AtomicU64,
}

impl Counters {
    const fn new() -> Self {
        Self {
            decided: AtomicU64::new(0),
            gpu: AtomicU64::new(0),
            cpu: AtomicU64::new(0),
            strict_violations: AtomicU64::new(0),
        }
    }
}

static COUNTERS: Counters = Counters::new();

/// A point-in-time copy of the decision counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DecisionCounters {
    pub decided: u64,
    pub gpu: u64,
    pub cpu: u64,
    pub strict_violations: u64,
}

/// Records one executed decision. Phase 2 call sites invoke this next to the
/// actual dispatch; a plan that was computed but never used must not call it.
pub fn record_decision(is_gpu: bool, strict_violation: bool) {
    COUNTERS.decided.fetch_add(1, Ordering::Relaxed);
    if is_gpu {
        COUNTERS.gpu.fetch_add(1, Ordering::Relaxed);
    } else {
        COUNTERS.cpu.fetch_add(1, Ordering::Relaxed);
    }
    if strict_violation {
        COUNTERS.strict_violations.fetch_add(1, Ordering::Relaxed);
    }
}

/// Reads the counters without resetting them.
pub fn counters() -> DecisionCounters {
    DecisionCounters {
        decided: COUNTERS.decided.load(Ordering::Relaxed),
        gpu: COUNTERS.gpu.load(Ordering::Relaxed),
        cpu: COUNTERS.cpu.load(Ordering::Relaxed),
        strict_violations: COUNTERS.strict_violations.load(Ordering::Relaxed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_semantics_match_existing_switches() {
        let env = EnvSnapshot::from_pairs(&[
            ("A", "0"),
            ("B", "false"),
            ("C", "FALSE"),
            ("D", "1"),
            ("E", "2"),
            ("F", "true"),
            ("G", ""),
        ]);

        // 0 / false (case-insensitive) are off; everything else is on.
        assert_eq!(env.flag_false_off("A"), Some(false));
        assert_eq!(env.flag_false_off("B"), Some(false));
        assert_eq!(env.flag_false_off("C"), Some(false));
        assert_eq!(env.flag_false_off("D"), Some(true));
        assert_eq!(env.flag_false_off("E"), Some(true));
        assert_eq!(env.flag_false_off("F"), Some(true));
        assert_eq!(env.flag_false_off("G"), Some(true));
        assert_eq!(env.flag_false_off("Z"), None);

        // Only the literal "0" is off.
        assert_eq!(env.flag_zero_off("A"), Some(false));
        assert_eq!(env.flag_zero_off("B"), Some(true));
        assert_eq!(env.flag_zero_off("D"), Some(true));

        // Only the literal "1" is on.
        assert_eq!(env.exact_one("D"), Some(true));
        assert_eq!(env.exact_one("E"), Some(false));
        assert_eq!(env.exact_one("Z"), None);
    }

    #[test]
    fn capture_filters_to_aurora_prefix() {
        let vars = vec![
            (OsString::from("AURORA_TEST_A"), OsString::from("1")),
            (OsString::from("PATH"), OsString::from("x")),
            (OsString::from("AURORA_TEST_B"), OsString::from("0")),
        ];
        let env = EnvSnapshot::capture_from(vars);
        assert_eq!(env.var("AURORA_TEST_A"), Some("1"));
        assert_eq!(env.var("AURORA_TEST_B"), Some("0"));
        assert_eq!(env.var("PATH"), None);
        assert_eq!(env.entries().count(), 2);
    }

    #[test]
    fn parse_helpers_trim_and_reject_malformed() {
        let env = EnvSnapshot::from_pairs(&[
            ("AURORA_VRAM_BUDGET_MB", " 4096 "),
            ("AURORA_BAD", "many"),
        ]);
        assert_eq!(env.parse_u64("AURORA_VRAM_BUDGET_MB"), Some(4096));
        assert_eq!(env.parse_u64("AURORA_BAD"), None);
        assert_eq!(env.parse_u64("AURORA_MISSING"), None);
    }
}
