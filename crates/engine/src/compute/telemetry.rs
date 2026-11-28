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
/// [`record_decision`] is called next to the real GPU-vs-CPU dispatch points
/// (develop band/resident/plane, NR, detail, clarity). These are the *executed*
/// dispatch outcomes, not the `compute::policy` plan. They use relaxed atomics:
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

/// Records one executed decision. Call sites invoke this next to the actual
/// dispatch; a plan that was computed but never used must not call it.
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

/// Pipeline stage a timing span is charged to.
///
/// Spans are inclusive and may nest (for example `Develop` runs while
/// `Sampling` is also open on the same thread), so the per-stage totals are not
/// additive; compare each against `RenderTotal`. Parallel renders add their
/// wall times together as well.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Stage {
    Decode,
    SourceBuild,
    Sampling,
    Develop,
    Detail,
    Portrait,
    LayersMasks,
    DisplayConversion,
    ExportEncode,
    RenderTotal,
}

impl Stage {
    pub const ALL: [Stage; 10] = [
        Stage::Decode,
        Stage::SourceBuild,
        Stage::Sampling,
        Stage::Develop,
        Stage::Detail,
        Stage::Portrait,
        Stage::LayersMasks,
        Stage::DisplayConversion,
        Stage::ExportEncode,
        Stage::RenderTotal,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Stage::Decode => "decode",
            Stage::SourceBuild => "source_build",
            Stage::Sampling => "sampling",
            Stage::Develop => "develop",
            Stage::Detail => "detail",
            Stage::Portrait => "portrait",
            Stage::LayersMasks => "layers_masks",
            Stage::DisplayConversion => "display_conversion",
            Stage::ExportEncode => "export_encode",
            Stage::RenderTotal => "render_total",
        }
    }
}

struct StageSlot {
    nanos: AtomicU64,
    spans: AtomicU64,
    max_nanos: AtomicU64,
}

const STAGE_COUNT: usize = Stage::ALL.len();

#[allow(clippy::declare_interior_mutable_const)]
const STAGE_INIT: StageSlot = StageSlot {
    nanos: AtomicU64::new(0),
    spans: AtomicU64::new(0),
    max_nanos: AtomicU64::new(0),
};
static STAGES: [StageSlot; STAGE_COUNT] = [STAGE_INIT; STAGE_COUNT];

/// RAII timer: charges the elapsed time to its stage when dropped. One
/// `Instant::now()` on creation and three relaxed atomic ops on drop; no
/// allocation, no locks, no effect on pixels.
#[must_use = "bind the span to a variable; dropping it immediately records ~0"]
pub struct StageSpan {
    stage: Stage,
    start: std::time::Instant,
}

/// Starts timing `stage`; the span ends when the returned guard is dropped.
pub fn span(stage: Stage) -> StageSpan {
    StageSpan { stage, start: std::time::Instant::now() }
}

impl Drop for StageSpan {
    fn drop(&mut self) {
        let nanos = u64::try_from(self.start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        record_stage(self.stage, nanos);
    }
}

/// Charges `nanos` to `stage` directly, for timings measured elsewhere.
pub fn record_stage(stage: Stage, nanos: u64) {
    let slot = &STAGES[stage as usize];
    slot.nanos.fetch_add(nanos, Ordering::Relaxed);
    slot.spans.fetch_add(1, Ordering::Relaxed);
    slot.max_nanos.fetch_max(nanos, Ordering::Relaxed);
}

/// Accumulated time for one stage.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageTotals {
    pub total_nanos: u64,
    pub spans: u64,
    pub max_nanos: u64,
}

impl StageTotals {
    pub fn total_ms(&self) -> f64 {
        self.total_nanos as f64 / 1.0e6
    }

    pub fn mean_ms(&self) -> f64 {
        if self.spans == 0 { 0.0 } else { self.total_ms() / self.spans as f64 }
    }
}

/// Point-in-time copy of every stage's totals, indexed like [`Stage::ALL`].
/// Never resets the counters.
pub fn stage_totals() -> [StageTotals; STAGE_COUNT] {
    let mut out = [StageTotals::default(); STAGE_COUNT];
    for (slot, total) in STAGES.iter().zip(out.iter_mut()) {
        *total = StageTotals {
            total_nanos: slot.nanos.load(Ordering::Relaxed),
            spans: slot.spans.load(Ordering::Relaxed),
            max_nanos: slot.max_nanos.load(Ordering::Relaxed),
        };
    }
    out
}

/// One-line JSON of the decision counters and stage totals, in the shape of the
/// existing `AURORA_PERF {...}` events. The engine prints it when
/// `AURORA_PERF_REPLAY` is set; any diagnostics reader may call it.
pub fn perf_json() -> String {
    use std::fmt::Write;
    let c = counters();
    let mut out = format!(
        "{{\"event\":\"engine_stage_totals\",\"decided\":{},\"gpu\":{},\"cpu\":{},\"strict_violations\":{},\"stages\":{{",
        c.decided, c.gpu, c.cpu, c.strict_violations
    );
    for (i, (stage, t)) in Stage::ALL.iter().zip(stage_totals()).enumerate() {
        if i > 0 {
            out.push(',');
        }
        let _ = write!(
            out,
            "\"{}\":{{\"spans\":{},\"total_ms\":{:.3},\"mean_ms\":{:.3},\"max_ms\":{:.3}}}",
            stage.as_str(),
            t.spans,
            t.total_ms(),
            t.mean_ms(),
            t.max_nanos as f64 / 1.0e6,
        );
    }
    out.push_str("}}");
    out
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
    fn stage_spans_accumulate_and_serialise() {
        // Process-wide counters: assert deltas, not absolutes.
        let before = stage_totals()[Stage::Detail as usize];
        {
            let _span = span(Stage::Detail);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        record_stage(Stage::Detail, 1_000);
        let after = stage_totals()[Stage::Detail as usize];
        assert!(after.spans >= before.spans + 2);
        assert!(after.total_nanos >= before.total_nanos + 2_000_000);
        assert!(after.max_nanos >= 2_000_000);
        let json: serde_json::Value = serde_json::from_str(&perf_json()).expect("valid json");
        assert_eq!(json["event"], "engine_stage_totals");
        for stage in Stage::ALL {
            assert!(json["stages"][stage.as_str()]["spans"].is_u64(), "{}", stage.as_str());
        }
    }

    #[test]
    fn decision_counters_split_gpu_and_cpu() {
        // Process-wide counters other tests may also bump: assert lower bounds.
        let before = counters();
        record_decision(true, false);
        record_decision(false, false);
        record_decision(false, true);
        let after = counters();
        assert!(after.decided >= before.decided + 3);
        assert!(after.gpu > before.gpu);
        assert!(after.cpu >= before.cpu + 2);
        assert!(after.strict_violations > before.strict_violations);
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
