//! §8.2 — pure cost model, EWMA/quantile calibration, hysteresis and failure
//! classification for CPU/GPU placement.
//!
//! This module is a *model*, not a scheduler. It owns no threads, no GPU
//! contexts, no environment variables and no file handles. A future
//! `ComputeScheduler` may consult it at frame boundaries, but nothing here
//! selects, dispatches or cancels work by itself. It deliberately does not
//! import `mock::develop_gpu`: the existing policy table is mirrored below as
//! [`DEVELOP_CROSSOVER_PRIOR`] where it is treated as an *unmeasured prior*,
//! never as ground truth.
//!
//! # Cost structure
//!
//! The model only defines the shape of the two end-to-end latencies:
//!
//! ```text
//! T_cpu = queue + setup + compute + migration
//! T_gpu = queue + setup + H2D + kernel + D2H + sync
//! ```
//!
//! Terms are nanoseconds and additive. `migration` on the CPU side is the
//! caller-side cost of moving a plane back to a CPU-owned buffer (for example
//! after a GPU-produced gain plane); it is *not* the same thing as `D2H`, which
//! is the device-to-host copy inside the GPU path. Keeping them separate makes
//! the "GPU produced, CPU consumed" residency case expressible without inventing
//! a third backend.
//!
//! # Calibration
//!
//! Measurements are tracked per `(backend, residency)` pair so cold and warm
//! runs never contaminate each other. Each track carries an EWMA (mean, sample
//! count, confidence) and a pair of streaming quantiles (median and tail). The
//! estimator is *seeded* by a prior, but confidence only counts real samples,
//! so a prior cannot make a track look warm.
//!
//! # Hysteresis and the no-oscillation argument
//!
//! Let `a = (T_cpu - T_gpu) / T_cpu` be the relative end-to-end advantage of
//! the GPU. With a deadband half-width `h` (`gain_threshold`) the state machine
//! switches CPU→GPU only while `a > +h` and GPU→CPU only while `a < -h`, and
//! only after `confirm_samples = K` consecutive frame-boundary observations of
//! the same candidate, and only after the current state has survived
//! `min_dwell_samples = D` observations.
//!
//! Consequences:
//!
//! * If observation noise keeps `|a| <= h`, neither strict inequality can fire,
//!   so the switch counter is provably zero. This is the case tested by
//!   `hysteresis_never_oscillates_inside_the_deadband`.
//! * After a switch the dwell guard blocks any further switch for `D`
//!   observations, and a switch additionally requires `K` consecutive
//!   candidate confirmations. The confirmations may accrue *during* the dwell
//!   window, so the tightest inter-switch period is `max(K, D)` observations:
//!   `switches(T) <= T / max(K, D)`, a convergent bound rather than a
//!   per-frame flip-flop. This is exercised by
//!   `hysteresis_switch_count_is_bounded_by_switch_period`.
//! * Because the same threshold is used in both directions, the decision has no
//!   asymmetric band that would let a signal at `h + ε` toggle back and forth.
//!
//! # Failure recovery
//!
//! [`GpuFailure`] classifies OOM, device loss and shader/module compilation
//! failure into distinct [`FailureReason`]s and distinct [`RecoveryAction`]s.
//! They must not collapse into one "GPU failed" bucket: OOM retries with a
//! smaller working set, device loss invalidates the context and re-probes,
//! compilation failure disables that stage until the device fingerprint changes.
//!
//! # Device fingerprint
//!
//! [`DeviceFingerprint`] defines the identity a calibration record is keyed by:
//! provider, device index, vendor/device ids, name, driver version, architecture,
//! precision and provider options. [`CalibrationRecord`] defines the persisted
//! JSON envelope. Serialization is provided; **no file IO is implemented here**
//! and none should be added to this module.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Nanoseconds in one millisecond. Latency terms are stored in nanoseconds so
/// sub-millisecond kernels keep integer-free precision.
pub const NANOS_PER_MILLI: f64 = 1.0e6;

/// Confidence above which a track is considered "warm" and may drive the
/// hysteresis state machine instead of the cold-start prior.
///
/// With `confidence = n / (n + 4)` this is exactly four measured samples.
pub const WARM_CONFIDENCE: f64 = 0.5;

/// Smoothing constant used by the confidence formula: `n / (n + K)`.
const CONFIDENCE_HALF_COUNT: f64 = 4.0;

/// Schema version of the persisted calibration envelope.
pub const CALIBRATION_SCHEMA_VERSION: u32 = 1;

// ─────────────────────────────── backend ────────────────────────────────────

/// Which compute endpoint a decision selects for one workload.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cpu,
    Gpu,
}

impl Backend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Gpu => "gpu",
        }
    }
}

/// Cold means the backend was (re)initialised or lost its residency; warm means
/// inputs/modules/context are already resident. They are separate tracks
/// because warm latency can be an order of magnitude below cold latency.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Residency {
    Cold,
    Warm,
}

// ────────────────────────────── cost structure ──────────────────────────────

/// CPU side of the model: `queue + setup + compute + migration`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CpuCostTerms {
    /// Time spent waiting in the caller's queue before the work starts.
    pub queue_ns: f64,
    /// One-time setup amortised over this workload (thread-pool wake, scratch).
    pub setup_ns: f64,
    /// The actual scalar/vector kernel time.
    pub compute_ns: f64,
    /// Caller-side cost of migrating a backend-produced plane into CPU-owned
    /// memory (or waiting for it). Distinct from GPU `d2h_ns`.
    pub migration_ns: f64,
}

impl CpuCostTerms {
    pub const fn new(
        queue_ns: f64,
        setup_ns: f64,
        compute_ns: f64,
        migration_ns: f64,
    ) -> Self {
        Self {
            queue_ns,
            setup_ns,
            compute_ns,
            migration_ns,
        }
    }

    /// `T_cpu = queue + setup + compute + migration`.
    pub fn total_ns(&self) -> f64 {
        self.queue_ns + self.setup_ns + self.compute_ns + self.migration_ns
    }
}

/// GPU side of the model: `queue + setup + H2D + kernel + D2H + sync`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GpuCostTerms {
    pub queue_ns: f64,
    /// Context/module/command-buffer setup amortised over this workload.
    pub setup_ns: f64,
    /// Host-to-device transfer.
    pub h2d_ns: f64,
    /// Device kernel time.
    pub kernel_ns: f64,
    /// Device-to-host transfer.
    pub d2h_ns: f64,
    /// Synchronisation / fence wait / launch completion.
    pub sync_ns: f64,
}

impl GpuCostTerms {
    pub const fn new(
        queue_ns: f64,
        setup_ns: f64,
        h2d_ns: f64,
        kernel_ns: f64,
        d2h_ns: f64,
        sync_ns: f64,
    ) -> Self {
        Self {
            queue_ns,
            setup_ns,
            h2d_ns,
            kernel_ns,
            d2h_ns,
            sync_ns,
        }
    }

    /// `T_gpu = queue + setup + H2D + kernel + D2H + sync`.
    pub fn total_ns(&self) -> f64 {
        self.queue_ns + self.setup_ns + self.h2d_ns + self.kernel_ns + self.d2h_ns + self.sync_ns
    }
}

/// One concrete end-to-end prediction for both backends.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CostModel {
    pub cpu: CpuCostTerms,
    pub gpu: GpuCostTerms,
}

impl CostModel {
    pub const fn new(cpu: CpuCostTerms, gpu: GpuCostTerms) -> Self {
        Self { cpu, gpu }
    }

    pub fn cpu_total_ns(&self) -> f64 {
        self.cpu.total_ns()
    }

    pub fn gpu_total_ns(&self) -> f64 {
        self.gpu.total_ns()
    }

    /// `(T_cpu - T_gpu) / T_cpu`; positive means the GPU is faster.
    /// `None` when the CPU prediction is non-positive or non-finite.
    pub fn gpu_relative_advantage(&self) -> Option<f64> {
        Observation {
            cpu_total_ns: self.cpu.total_ns(),
            gpu_total_ns: self.gpu.total_ns(),
        }
        .gpu_relative_advantage()
    }
}

/// A measured or predicted pair of end-to-end latencies, in nanoseconds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Observation {
    pub cpu_total_ns: f64,
    pub gpu_total_ns: f64,
}

impl Observation {
    /// `(cpu - gpu) / cpu`, positive when the GPU is faster.
    ///
    /// Returns `None` for non-finite or non-positive inputs instead of
    /// producing an infinity that would poison a hysteresis decision.
    pub fn gpu_relative_advantage(&self) -> Option<f64> {
        if !self.cpu_total_ns.is_finite()
            || !self.gpu_total_ns.is_finite()
            || self.cpu_total_ns <= 0.0
            || self.gpu_total_ns < 0.0
        {
            return None;
        }
        Some((self.cpu_total_ns - self.gpu_total_ns) / self.cpu_total_ns)
    }
}

// ───────────────────────────── EWMA and quantiles ───────────────────────────

/// Exponentially weighted moving average with sample count, confidence and an
/// EW variance estimate.
///
/// A prior may seed the mean (see [`Ewma::with_prior`]) but never increments
/// `samples` or `confidence`; a sealed prior therefore cannot masquerade as
/// measured data.
#[derive(Clone, Debug)]
pub struct Ewma {
    alpha: f64,
    mean_ns: Option<f64>,
    variance_ns2: f64,
    samples: u64,
}

impl Ewma {
    /// `alpha` is clamped to `[1e-3, 1.0]`; `1.0` means "last sample wins".
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha: normalize_alpha(alpha),
            mean_ns: None,
            variance_ns2: 0.0,
            samples: 0,
        }
    }

    /// Seed the mean with an unmeasured prior. Confidence stays zero.
    pub fn with_prior(alpha: f64, prior_ns: f64) -> Self {
        let mut ewma = Self::new(alpha);
        if prior_ns.is_finite() && prior_ns >= 0.0 {
            ewma.mean_ns = Some(prior_ns);
        }
        ewma
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn mean_ns(&self) -> Option<f64> {
        self.mean_ns
    }

    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// `n / (n + 4)`; zero before any measured sample.
    pub fn confidence(&self) -> f64 {
        if self.mean_ns.is_none() {
            return 0.0;
        }
        let n = self.samples as f64;
        n / (n + CONFIDENCE_HALF_COUNT)
    }

    /// Relative EW standard deviation (`sqrt(var) / mean`), useful to decide
    /// whether the deadband is wider than the observed noise.
    pub fn relative_noise(&self) -> Option<f64> {
        let mean = self.mean_ns?;
        (mean > 0.0).then(|| self.variance_ns2.max(0.0).sqrt() / mean)
    }

    /// Feed one non-negative, finite latency sample. Invalid samples are ignored
    /// so a failed timing cannot drag the estimate.
    pub fn observe(&mut self, sample_ns: f64) {
        if !sample_ns.is_finite() || sample_ns < 0.0 {
            return;
        }
        match self.mean_ns {
            None => {
                self.mean_ns = Some(sample_ns);
                self.variance_ns2 = 0.0;
            }
            Some(mean) => {
                let delta = sample_ns - mean;
                self.mean_ns = Some(mean + self.alpha * delta);
                self.variance_ns2 =
                    (1.0 - self.alpha) * (self.variance_ns2 + self.alpha * delta * delta);
            }
        }
        self.samples = self.samples.saturating_add(1);
    }
}

fn normalize_alpha(alpha: f64) -> f64 {
    if alpha.is_finite() {
        alpha.clamp(1.0e-3, 1.0)
    } else {
        0.1
    }
}

/// Streaming quantile estimator using the P² algorithm (Jain & Chlamtac, 1985).
///
/// Five markers track one target quantile in constant memory. It is an
/// approximation: the estimate is the middle marker `q[2]`. It never allocates
/// after construction and tolerates sorted, monotonic and noisy input.
#[derive(Clone, Debug)]
pub struct P2Quantile {
    p: f64,
    count: u64,
    initial: [f64; 5],
    q: [f64; 5],
    positions: [f64; 5],
    desired: [f64; 5],
}

impl P2Quantile {
    pub fn new(p: f64) -> Self {
        let p = if p.is_finite() { p.clamp(1.0e-6, 1.0 - 1.0e-6) } else { 0.5 };
        Self {
            p,
            count: 0,
            initial: [0.0; 5],
            q: [0.0; 5],
            positions: [1.0, 2.0, 3.0, 4.0, 5.0],
            desired: [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0],
        }
    }

    pub fn p(&self) -> f64 {
        self.p
    }

    pub fn sample_count(&self) -> u64 {
        self.count
    }

    /// `None` until five samples have established the markers.
    pub fn estimate(&self) -> Option<f64> {
        (self.count >= 5).then_some(self.q[2])
    }

    pub fn observe(&mut self, sample: f64) {
        if !sample.is_finite() {
            return;
        }
        if self.count < 5 {
            self.initial[self.count as usize] = sample;
            self.count += 1;
            if self.count == 5 {
                self.initial.sort_by(|a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                self.q = self.initial;
            }
            return;
        }

        // Locate the cell; the outer markers absorb values outside the band.
        let k = if sample < self.q[0] {
            self.q[0] = sample;
            0
        } else if sample >= self.q[4] {
            self.q[4] = sample;
            3
        } else {
            let mut cell = 0;
            for i in 0..4 {
                if self.q[i] <= sample && sample < self.q[i + 1] {
                    cell = i;
                    break;
                }
            }
            cell
        };

        // All markers to the right of the cell move one rank up.
        for marker in (k + 1)..5 {
            self.positions[marker] += 1.0;
        }
        let increments = [0.0, self.p / 2.0, self.p, (1.0 + self.p) / 2.0, 1.0];
        for marker in 0..5 {
            self.desired[marker] += increments[marker];
        }

        // Adjust the three interior markers when they drift from their targets.
        for j in 1..4 {
            let drift = self.desired[j] - self.positions[j];
            let can_move_up =
                drift >= 1.0 && self.positions[j + 1] - self.positions[j] > 1.0;
            let can_move_down =
                drift <= -1.0 && self.positions[j - 1] - self.positions[j] < -1.0;
            if !(can_move_up || can_move_down) {
                continue;
            }
            let step = if drift > 0.0 { 1.0 } else { -1.0 };
            let q_prev = self.q[j - 1];
            let q_cur = self.q[j];
            let q_next = self.q[j + 1];
            let n_prev = self.positions[j - 1];
            let n_cur = self.positions[j];
            let n_next = self.positions[j + 1];

            let parabolic = q_cur
                + step / (n_next - n_prev)
                    * ((n_cur - n_prev + step) * (q_next - q_cur) / (n_next - n_cur)
                        + (n_next - n_cur - step) * (q_cur - q_prev) / (n_cur - n_prev));

            if parabolic > q_prev && parabolic < q_next {
                self.q[j] = parabolic;
            } else if step > 0.0 {
                self.q[j] = q_cur + (q_next - q_cur) / (n_next - n_cur);
            } else {
                self.q[j] = q_cur - (q_cur - q_prev) / (n_cur - n_prev);
            }
            self.positions[j] += step;
        }

        self.count += 1;
    }
}

/// Median plus upper-tail quantiles over one latency stream.
#[derive(Clone, Debug)]
pub struct QuantileBand {
    median: P2Quantile,
    tail: P2Quantile,
}

impl QuantileBand {
    /// `tail_p` is clamped to `(0, 1)`; 0.95 tracks the p95 tail by default.
    pub fn new(tail_p: f64) -> Self {
        Self {
            median: P2Quantile::new(0.5),
            tail: P2Quantile::new(tail_p),
        }
    }

    pub fn observe(&mut self, sample_ns: f64) {
        self.median.observe(sample_ns);
        self.tail.observe(sample_ns);
    }

    pub fn p50_ns(&self) -> Option<f64> {
        self.median.estimate()
    }

    pub fn tail_ns(&self) -> Option<f64> {
        self.tail.estimate()
    }

    pub fn samples(&self) -> u64 {
        self.median.sample_count()
    }
}

/// One `(backend, residency)` latency track: EWMA for the point estimate plus
/// quantiles for the tail. This is the unit a scheduler keeps per key.
#[derive(Clone, Debug)]
pub struct LatencyTrack {
    ewma: Ewma,
    quantiles: QuantileBand,
}

impl LatencyTrack {
    pub fn new(alpha: f64, tail_p: f64) -> Self {
        Self {
            ewma: Ewma::new(alpha),
            quantiles: QuantileBand::new(tail_p),
        }
    }

    pub fn observe(&mut self, sample_ns: f64) {
        self.ewma.observe(sample_ns);
        self.quantiles.observe(sample_ns);
    }

    pub fn mean_ns(&self) -> Option<f64> {
        self.ewma.mean_ns()
    }

    pub fn samples(&self) -> u64 {
        self.ewma.samples()
    }

    pub fn confidence(&self) -> f64 {
        self.ewma.confidence()
    }

    pub fn p50_ns(&self) -> Option<f64> {
        self.quantiles.p50_ns()
    }

    pub fn tail_ns(&self) -> Option<f64> {
        self.quantiles.tail_ns()
    }

    pub fn relative_noise(&self) -> Option<f64> {
        self.ewma.relative_noise()
    }
}

/// Key for one latency track.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct EstimateKey {
    pub backend: Backend,
    pub residency: Residency,
}

impl EstimateKey {
    pub const fn new(backend: Backend, residency: Residency) -> Self {
        Self { backend, residency }
    }
}

/// Point-in-time view of one track.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Estimate {
    pub key: EstimateKey,
    pub mean_ns: Option<f64>,
    pub p50_ns: Option<f64>,
    pub tail_ns: Option<f64>,
    pub samples: u64,
    pub confidence: f64,
    pub relative_noise: Option<f64>,
}

impl Estimate {
    pub fn is_warm(&self) -> bool {
        self.confidence >= WARM_CONFIDENCE
    }
}

/// EWMA + quantile tracks, separated by backend and residency.
#[derive(Clone, Debug)]
pub struct CostEstimator {
    alpha: f64,
    tail_p: f64,
    tracks: BTreeMap<EstimateKey, LatencyTrack>,
}

impl CostEstimator {
    pub fn new(alpha: f64, tail_p: f64) -> Self {
        Self {
            alpha: normalize_alpha(alpha),
            tail_p,
            tracks: BTreeMap::new(),
        }
    }

    pub fn observe(&mut self, key: EstimateKey, sample_ns: f64) {
        self.tracks
            .entry(key)
            .or_insert_with(|| LatencyTrack::new(self.alpha, self.tail_p))
            .observe(sample_ns);
    }

    pub fn track(&self, key: EstimateKey) -> Option<&LatencyTrack> {
        self.tracks.get(&key)
    }

    pub fn estimate(&self, key: EstimateKey) -> Estimate {
        match self.tracks.get(&key) {
            Some(track) => Estimate {
                key,
                mean_ns: track.mean_ns(),
                p50_ns: track.p50_ns(),
                tail_ns: track.tail_ns(),
                samples: track.samples(),
                confidence: track.confidence(),
                relative_noise: track.relative_noise(),
            },
            None => Estimate {
                key,
                mean_ns: None,
                p50_ns: None,
                tail_ns: None,
                samples: 0,
                confidence: 0.0,
                relative_noise: None,
            },
        }
    }

    /// Both backends for this residency must be warm ([`WARM_CONFIDENCE`]).
    pub fn is_warm(&self, residency: Residency) -> bool {
        self.estimate(EstimateKey::new(Backend::Cpu, residency))
            .is_warm()
            && self
                .estimate(EstimateKey::new(Backend::Gpu, residency))
                .is_warm()
    }

    /// End-to-end comparison for a residency, or `None` until both tracks have
    /// a measured mean. A prior-seeded mean without samples does not qualify.
    pub fn comparison(&self, residency: Residency) -> Option<Observation> {
        let cpu = self
            .track(EstimateKey::new(Backend::Cpu, residency))?
            .mean_ns()?;
        let gpu = self
            .track(EstimateKey::new(Backend::Gpu, residency))?
            .mean_ns()?;
        Some(Observation {
            cpu_total_ns: cpu,
            gpu_total_ns: gpu,
        })
    }
}

// ──────────────────────────── hysteresis state ──────────────────────────────

/// Hysteresis tuning. `gain_threshold` is the roadmap's 15–20 % band; 0.18 is
/// the default midpoint.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HysteresisConfig {
    /// Relative end-to-end gain required to leave the current backend.
    pub gain_threshold: f64,
    /// Consecutive frame-boundary observations of the same candidate required
    /// before switching.
    pub confirm_samples: u32,
    /// Observations the current backend must survive before another switch is
    /// allowed. `0` disables the dwell guard (not recommended in production).
    pub min_dwell_samples: u32,
}

impl Default for HysteresisConfig {
    fn default() -> Self {
        Self {
            gain_threshold: 0.18,
            confirm_samples: 3,
            min_dwell_samples: 5,
        }
    }
}

impl HysteresisConfig {
    /// Clamp nonsensical values instead of panicking in a render loop.
    pub fn normalized(mut self) -> Self {
        if !self.gain_threshold.is_finite() {
            self.gain_threshold = 0.18;
        }
        self.gain_threshold = self.gain_threshold.clamp(0.0, 1.0);
        self.confirm_samples = self.confirm_samples.max(1);
        self
    }

    /// Tightest number of observations between two switches. The candidate
    /// confirmation may accrue during the dwell window, so this is
    /// `max(confirm_samples, min_dwell_samples)` — not their sum. The
    /// oscillation bound is `switches(T) <= T / minimum_switch_period()`.
    pub fn minimum_switch_period(&self) -> u32 {
        self.confirm_samples.max(self.min_dwell_samples)
    }
}

/// Why a decision stayed on its current backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionReason {
    /// The observation had a non-finite or non-positive CPU total.
    NoObservation,
    /// The current backend is already the preferred one, or the advantage is
    /// inside the deadband.
    Stable,
    /// A candidate exists but has not been confirmed for enough frames.
    Confirming,
    /// The candidate is confirmed but the current backend is still in its
    /// minimum dwell window.
    Dwell,
    /// The decision switched this frame.
    Switched,
}

/// One frame-boundary decision. `backend` is the backend to use for the next
/// frame; `switched` tells telemetry whether a transition happened.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Decision {
    pub backend: Backend,
    pub switched: bool,
    pub reason: DecisionReason,
    pub gpu_relative_advantage: Option<f64>,
}

impl Decision {
    fn stay(backend: Backend, reason: DecisionReason, advantage: Option<f64>) -> Self {
        Self {
            backend,
            switched: false,
            reason,
            gpu_relative_advantage: advantage,
        }
    }
}

/// Two-state CPU/GPU selector with a symmetric deadband, candidate
/// confirmation counter and dwell guard. All updates happen at frame
/// boundaries; see the module docs for the no-oscillation argument.
#[derive(Clone, Debug)]
pub struct Hysteresis {
    config: HysteresisConfig,
    current: Backend,
    frames_in_state: u32,
    candidate: Option<Backend>,
    candidate_streak: u32,
    switches: u64,
}

impl Hysteresis {
    pub fn new(config: HysteresisConfig, initial: Backend) -> Self {
        Self {
            config: config.normalized(),
            current: initial,
            frames_in_state: 0,
            candidate: None,
            candidate_streak: 0,
            switches: 0,
        }
    }

    pub fn config(&self) -> HysteresisConfig {
        self.config
    }

    pub fn current(&self) -> Backend {
        self.current
    }

    pub fn switches(&self) -> u64 {
        self.switches
    }

    /// Number of observations the current backend has held since the last
    /// switch (or construction).
    pub fn frames_in_state(&self) -> u32 {
        self.frames_in_state
    }

    /// Feed one frame-boundary observation and receive the next frame's choice.
    pub fn observe(&mut self, observation: Observation) -> Decision {
        self.frames_in_state = self.frames_in_state.saturating_add(1);

        let Some(advantage) = observation.gpu_relative_advantage() else {
            self.reset_candidate();
            return Decision::stay(self.current, DecisionReason::NoObservation, None);
        };

        let threshold = self.config.gain_threshold;
        let raw_candidate = if advantage > threshold {
            Some(Backend::Gpu)
        } else if advantage < -threshold {
            Some(Backend::Cpu)
        } else {
            None
        };

        let candidate = match raw_candidate {
            Some(candidate) if candidate != self.current => candidate,
            _ => {
                self.reset_candidate();
                return Decision::stay(
                    self.current,
                    DecisionReason::Stable,
                    Some(advantage),
                );
            }
        };

        if self.candidate == Some(candidate) {
            self.candidate_streak = self.candidate_streak.saturating_add(1);
        } else {
            self.candidate = Some(candidate);
            self.candidate_streak = 1;
        }

        if self.candidate_streak < self.config.confirm_samples {
            return Decision::stay(
                self.current,
                DecisionReason::Confirming,
                Some(advantage),
            );
        }

        if self.frames_in_state < self.config.min_dwell_samples {
            return Decision::stay(self.current, DecisionReason::Dwell, Some(advantage));
        }

        self.current = candidate;
        self.candidate = None;
        self.candidate_streak = 0;
        self.frames_in_state = 0;
        self.switches = self.switches.saturating_add(1);

        Decision {
            backend: candidate,
            switched: true,
            reason: DecisionReason::Switched,
            gpu_relative_advantage: Some(advantage),
        }
    }

    fn reset_candidate(&mut self) {
        self.candidate = None;
        self.candidate_streak = 0;
    }
}

/// Defers a switch until the next frame boundary so an in-flight frame is never
/// split across backends. `propose` may be called from any event handler; only
/// `take_at_boundary(true)` releases the change.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameBoundaryGate {
    pending: Option<Backend>,
}

impl FrameBoundaryGate {
    pub fn propose(&mut self, decision: Decision) {
        if decision.switched {
            self.pending = Some(decision.backend);
        }
    }

    pub fn pending(&self) -> Option<Backend> {
        self.pending
    }

    /// Returns the new backend exactly once, and only when the caller is at a
    /// frame boundary with no work in flight.
    pub fn take_at_boundary(&mut self, at_boundary: bool) -> Option<Backend> {
        if at_boundary {
            self.pending.take()
        } else {
            None
        }
    }
}

// ─────────────────────────────── cold prior ─────────────────────────────────

/// One row of the existing develop crossover policy table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrossoverRow {
    pub pixels: u64,
    pub gpu_wins: bool,
}

/// Mirror of `CROSSOVER_TABLE` in `mock/develop_gpu.rs` at baseline `8118a4a`.
///
/// That table is a **policy table, not measured develop timings** (the source
/// comment says so). It is reproduced here only as a cold-start *prior*: the
/// scheduler may use it before it has measurements, and must stop consulting it
/// once both tracks are warm. It is not linked to the original constant, so a
/// future measurement can replace it without changing dispatch.
pub const DEVELOP_CROSSOVER_PRIOR: [CrossoverRow; 6] = [
    CrossoverRow { pixels: 65_536, gpu_wins: false }, // 256×256
    CrossoverRow { pixels: 921_600, gpu_wins: false }, // 1280×720
    CrossoverRow { pixels: 2_073_600, gpu_wins: true }, // 1920×1080
    CrossoverRow { pixels: 8_294_400, gpu_wins: true }, // 3840×2160
    CrossoverRow { pixels: 24_385_536, gpu_wins: true }, // 6048×4032 (24.4 MP)
    CrossoverRow { pixels: 44_761_088, gpu_wins: true }, // 8192×5464 (44.8 MP)
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PriorSource {
    /// Copied from a policy table with no real measurement behind it.
    UnmeasuredPolicyTable,
    /// Derived from a measured crossover point.
    MeasuredCalibration,
}

/// Cold-start placement policy.
#[derive(Clone, Debug, PartialEq)]
pub struct PriorModel {
    rows: Vec<CrossoverRow>,
    source: PriorSource,
}

impl PriorModel {
    /// Reproduces the source semantics exactly: the first `gpu_wins` row is the
    /// floor, and every size at or above it is predicted GPU.
    pub fn from_crossover_table(rows: &[CrossoverRow]) -> Self {
        Self {
            rows: rows.to_vec(),
            source: PriorSource::UnmeasuredPolicyTable,
        }
    }

    /// Replace the prior with an actually measured crossover point.
    pub fn from_measured_crossover(pixels: u64) -> Self {
        Self {
            rows: vec![CrossoverRow {
                pixels,
                gpu_wins: true,
            }],
            source: PriorSource::MeasuredCalibration,
        }
    }

    pub fn source(&self) -> PriorSource {
        self.source
    }

    pub fn rows(&self) -> &[CrossoverRow] {
        &self.rows
    }

    /// First `gpu_wins` row; `None` means the prior never predicts GPU.
    pub fn crossover_floor(&self) -> Option<u64> {
        self.rows
            .iter()
            .find(|row| row.gpu_wins)
            .map(|row| row.pixels)
    }

    pub fn predicts_gpu(&self, pixels: u64) -> bool {
        self.crossover_floor()
            .is_some_and(|floor| pixels >= floor)
    }

    /// The only decision available before measurements exist. Once
    /// [`CostEstimator::is_warm`] is true, hysteresis must take over.
    pub fn cold_placement(&self, pixels: u64) -> Backend {
        if self.predicts_gpu(pixels) {
            Backend::Gpu
        } else {
            Backend::Cpu
        }
    }
}

// ────────────────────────── failure classification ──────────────────────────

/// A GPU execution failure, classified at the point it is observed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuFailure {
    /// The allocation did not fit the ledger/device budget.
    OutOfMemory {
        required_bytes: u64,
        budget_bytes: u64,
    },
    /// Driver codes: 700 illegal address, 702 launch failure, 709/710 context
    /// destroyed, 719 unsupported launch, 999 unknown.
    DeviceLost { driver_code: i32 },
    /// PTX/module/kernel compilation or load failed for one stage.
    CompileFailed { stage: &'static str },
}

/// Distinct reason codes. These are intentionally three separate values: a
/// single "GPU failed" reason would lose the recovery decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum FailureReason {
    OutOfMemory,
    DeviceLost,
    CompileFailed,
}

impl FailureReason {
    pub const fn as_str(self) -> &'static str {
        match self {
            FailureReason::OutOfMemory => "out_of_memory",
            FailureReason::DeviceLost => "device_lost",
            FailureReason::CompileFailed => "compile_failed",
        }
    }
}

/// What the scheduler should do after a classified failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum RecoveryAction {
    /// Run this frame on the CPU and shrink the next GPU attempt (tile/band or
    /// lower residency). The device itself is still usable.
    CpuThisFrameThenReduceWorkingSet,
    /// Invalidate the CUDA/Metal context, drop cached modules and re-probe the
    /// device before any further dispatch.
    InvalidateContextAndReprobe,
    /// Stop attempting this stage on this device fingerprint. It is only
    /// reconsidered after the fingerprint changes (driver/module update).
    DisableStageUntilFingerprintChanges,
}

impl GpuFailure {
    pub fn reason(&self) -> FailureReason {
        match self {
            GpuFailure::OutOfMemory { .. } => FailureReason::OutOfMemory,
            GpuFailure::DeviceLost { .. } => FailureReason::DeviceLost,
            GpuFailure::CompileFailed { .. } => FailureReason::CompileFailed,
        }
    }

    pub fn recovery(&self) -> RecoveryAction {
        match self {
            GpuFailure::OutOfMemory { .. } => {
                RecoveryAction::CpuThisFrameThenReduceWorkingSet
            }
            GpuFailure::DeviceLost { .. } => RecoveryAction::InvalidateContextAndReprobe,
            GpuFailure::CompileFailed { .. } => {
                RecoveryAction::DisableStageUntilFingerprintChanges
            }
        }
    }

    /// Whether retrying the identical GPU work without changing anything is
    /// sensible. None of the three failures qualify.
    pub fn retry_same_work_is_safe(&self) -> bool {
        false
    }

    /// OOM and device loss leave the stage usable after recovery; a compile
    /// failure is sticky for the current fingerprint.
    pub fn stage_usable_after_recovery(&self) -> bool {
        !matches!(self, GpuFailure::CompileFailed { .. })
    }
}

// ─────────────────────────── device fingerprint ─────────────────────────────

/// Compute provider identity, including providers the engine does not drive
/// yet (Metal/OpenCL) so calibration records stay forward-compatible.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ComputeProvider {
    Cpu,
    Cuda,
    Metal,
    DirectMl,
    OpenCl,
}

impl ComputeProvider {
    pub const fn as_str(self) -> &'static str {
        match self {
            ComputeProvider::Cpu => "cpu",
            ComputeProvider::Cuda => "cuda",
            ComputeProvider::Metal => "metal",
            ComputeProvider::DirectMl => "directml",
            ComputeProvider::OpenCl => "opencl",
        }
    }
}

/// Numerical precision the calibrated kernels ran at. Calibration results are
/// not transferable between precisions.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    F32,
    F16,
    Bf16,
    Int8,
}

impl Precision {
    pub const fn as_str(self) -> &'static str {
        match self {
            Precision::F32 => "f32",
            Precision::F16 => "f16",
            Precision::Bf16 => "bf16",
            Precision::Int8 => "int8",
        }
    }
}

/// Identity a calibration record is keyed by. Every field is stable across
/// process restarts; volatile counters (load, temperature, free VRAM) are
/// intentionally absent.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceFingerprint {
    pub provider: ComputeProvider,
    pub device_index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub vendor_id: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_id: Option<u32>,
    pub name: String,
    pub driver_version: String,
    /// CUDA compute capability, Metal family, or CPU ISA level.
    pub architecture: String,
    pub precision: Precision,
    /// Provider-specific knobs that change kernel selection, sorted by key.
    #[serde(default)]
    pub provider_options: BTreeMap<String, String>,
}

impl DeviceFingerprint {
    /// Deterministic key: field order is fixed and options are sorted by the
    /// `BTreeMap`, so the same device always yields the same string.
    pub fn stable_key(&self) -> String {
        let mut key = String::new();
        key.push_str(self.provider.as_str());
        key.push('/');
        key.push_str(&self.device_index.to_string());
        key.push('/');
        key.push_str(&option_u32(self.vendor_id));
        key.push('/');
        key.push_str(&option_u32(self.device_id));
        key.push('/');
        key.push_str(&self.name);
        key.push('/');
        key.push_str(&self.driver_version);
        key.push('/');
        key.push_str(&self.architecture);
        key.push('/');
        key.push_str(self.precision.as_str());
        for (name, value) in &self.provider_options {
            key.push('/');
            key.push_str(name);
            key.push('=');
            key.push_str(value);
        }
        key
    }

    /// Suffix for a calibration file. Computing the name is pure; the caller
    /// performs the IO.
    pub fn persistence_key_hex(&self) -> String {
        format!("device-{:016x}", fnv1a_64(self.stable_key().as_bytes()))
    }
}

fn option_u32(value: Option<u32>) -> String {
    match value {
        Some(value) => value.to_string(),
        None => "-".to_string(),
    }
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// JSON envelope for one device's calibration. **Serialization only** — this
/// module never opens, reads or writes a file.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CalibrationRecord {
    pub schema_version: u32,
    pub fingerprint: DeviceFingerprint,
    /// Measured CPU/GPU develop crossover, if calibration produced one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crossover_pixels: Option<u64>,
    /// Total measured samples backing this record.
    pub samples: u64,
    /// Unix seconds at which the record was produced.
    pub updated_unix_s: u64,
    /// Free-form notes; serialized in sorted key order.
    #[serde(default)]
    pub notes: BTreeMap<String, String>,
}

impl CalibrationRecord {
    pub fn new(fingerprint: DeviceFingerprint, updated_unix_s: u64) -> Self {
        Self {
            schema_version: CALIBRATION_SCHEMA_VERSION,
            fingerprint,
            crossover_pixels: None,
            samples: 0,
            updated_unix_s,
            notes: BTreeMap::new(),
        }
    }

    pub fn is_compatible(&self) -> bool {
        self.schema_version == CALIBRATION_SCHEMA_VERSION
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(text: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(text)
    }
}

// ───────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    fn obs(cpu_ns: f64, gpu_ns: f64) -> Observation {
        Observation {
            cpu_total_ns: cpu_ns,
            gpu_total_ns: gpu_ns,
        }
    }

    /// Deterministic LCG so the noise test is reproducible without a crate.
    struct Lcg(u64);

    impl Lcg {
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }

        /// Uniform in `[-amplitude, +amplitude]`.
        fn signed(&mut self, amplitude: f64) -> f64 {
            (self.unit() * 2.0 - 1.0) * amplitude
        }
    }

    #[test]
    fn cpu_and_gpu_totals_follow_the_documented_formula() {
        let cpu = CpuCostTerms::new(10.0, 20.0, 300.0, 5.0);
        let gpu = GpuCostTerms::new(10.0, 30.0, 40.0, 100.0, 50.0, 20.0);
        assert_eq!(cpu.total_ns(), 335.0);
        assert_eq!(gpu.total_ns(), 250.0);
        let model = CostModel::new(cpu, gpu);
        assert_eq!(model.cpu_total_ns(), 335.0);
        assert_eq!(model.gpu_total_ns(), 250.0);
        let advantage = model.gpu_relative_advantage().unwrap();
        assert!((advantage - (335.0 - 250.0) / 335.0).abs() < 1e-12);
    }

    #[test]
    fn cpu_migration_is_not_d2h() {
        // A GPU-produced plane consumed by the CPU carries migration only on the
        // consumer side; the GPU terms still account for their own D2H.
        let cpu = CpuCostTerms::new(0.0, 0.0, 100.0, 900.0);
        let gpu = GpuCostTerms::new(0.0, 0.0, 0.0, 100.0, 0.0, 0.0);
        assert_eq!(cpu.total_ns(), 1000.0);
        assert_eq!(gpu.total_ns(), 100.0);
        assert_ne!(cpu.migration_ns, gpu.d2h_ns);
    }

    #[test]
    fn observation_rejects_invalid_latencies() {
        assert!(obs(0.0, 10.0).gpu_relative_advantage().is_none());
        assert!(obs(-1.0, 10.0).gpu_relative_advantage().is_none());
        assert!(obs(f64::NAN, 10.0).gpu_relative_advantage().is_none());
        assert!(obs(10.0, f64::INFINITY).gpu_relative_advantage().is_none());
        assert!(obs(10.0, -1.0).gpu_relative_advantage().is_none());
        assert_eq!(obs(100.0, 70.0).gpu_relative_advantage().unwrap(), 0.3);
    }

    #[test]
    fn ewma_converges_and_reports_confidence() {
        let mut ewma = Ewma::new(0.2);
        assert_eq!(ewma.mean_ns(), None);
        assert_eq!(ewma.confidence(), 0.0);

        for _ in 0..200 {
            ewma.observe(100.0);
        }
        assert!((ewma.mean_ns().unwrap() - 100.0).abs() < 1e-9);
        assert_eq!(ewma.samples(), 200);
        assert!(ewma.confidence() > 0.95);

        // A later step is tracked without an unbounded jump.
        ewma.observe(200.0);
        assert!(ewma.mean_ns().unwrap() < 130.0);
    }

    #[test]
    fn ewma_prior_is_not_measured_confidence() {
        let seeded = Ewma::with_prior(0.2, 500.0);
        assert_eq!(seeded.mean_ns(), Some(500.0));
        assert_eq!(seeded.samples(), 0);
        assert_eq!(seeded.confidence(), 0.0);

        let mut seeded = seeded;
        for _ in 0..4 {
            seeded.observe(100.0);
        }
        assert_eq!(seeded.samples(), 4);
        assert!(seeded.confidence() >= WARM_CONFIDENCE);
    }

    #[test]
    fn ewma_ignores_invalid_samples() {
        let mut ewma = Ewma::new(0.5);
        ewma.observe(100.0);
        ewma.observe(f64::NAN);
        ewma.observe(-5.0);
        ewma.observe(f64::INFINITY);
        assert_eq!(ewma.samples(), 1);
        assert_eq!(ewma.mean_ns(), Some(100.0));
    }

    #[test]
    fn p2_estimates_median_and_tail() {
        let mut median = P2Quantile::new(0.5);
        let mut tail = P2Quantile::new(0.95);
        for value in 1..=10_000 {
            let sample = value as f64;
            median.observe(sample);
            tail.observe(sample);
        }
        let median = median.estimate().unwrap();
        let tail = tail.estimate().unwrap();
        assert!((median - 5_000.0).abs() < 250.0, "median={median}");
        assert!((tail - 9_500.0).abs() < 500.0, "tail={tail}");
    }

    #[test]
    fn p2_handles_unsorted_input() {
        let mut lcg = Lcg(42);
        let mut p50 = P2Quantile::new(0.5);
        let mut values = Vec::new();
        for _ in 0..5_000 {
            let value = (lcg.unit() * 1_000.0).round();
            values.push(value);
            p50.observe(value);
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let exact = values[values.len() / 2];
        let estimate = p50.estimate().unwrap();
        assert!(
            (estimate - exact).abs() < 60.0,
            "estimate={estimate} exact={exact}"
        );
    }

    #[test]
    fn latency_track_carries_ewma_and_quantiles() {
        let mut track = LatencyTrack::new(0.2, 0.95);
        for value in 1..=1_000 {
            track.observe(value as f64);
        }
        assert_eq!(track.samples(), 1_000);
        assert!(track.mean_ns().unwrap() > 900.0);
        assert!(track.p50_ns().unwrap() > 400.0);
        assert!(track.tail_ns().unwrap() > track.p50_ns().unwrap());
        assert!(track.confidence() > 0.99);
    }

    #[test]
    fn estimator_separates_cold_and_warm_tracks() {
        let mut estimator = CostEstimator::new(0.2, 0.95);
        let cpu_cold = EstimateKey::new(Backend::Cpu, Residency::Cold);
        let cpu_warm = EstimateKey::new(Backend::Cpu, Residency::Warm);
        for _ in 0..10 {
            estimator.observe(cpu_cold, 1_000.0);
        }
        assert_eq!(estimator.estimate(cpu_cold).samples, 10);
        assert_eq!(estimator.estimate(cpu_warm).samples, 0);
        assert!(!estimator.is_warm(Residency::Cold), "gpu track is absent");
        assert!(estimator.comparison(Residency::Cold).is_none());
    }

    #[test]
    fn estimator_comparison_is_warm_after_four_samples_each() {
        let mut estimator = CostEstimator::new(0.2, 0.95);
        let cpu = EstimateKey::new(Backend::Cpu, Residency::Warm);
        let gpu = EstimateKey::new(Backend::Gpu, Residency::Warm);
        for _ in 0..4 {
            estimator.observe(cpu, 1_000.0);
            estimator.observe(gpu, 700.0);
        }
        assert!(estimator.is_warm(Residency::Warm));
        let comparison = estimator.comparison(Residency::Warm).unwrap();
        assert!((comparison.gpu_relative_advantage().unwrap() - 0.3).abs() < 1e-9);
    }

    #[test]
    fn hysteresis_switches_only_after_confirmation() {
        let config = HysteresisConfig {
            gain_threshold: 0.18,
            confirm_samples: 3,
            min_dwell_samples: 2,
        };
        let mut hysteresis = Hysteresis::new(config, Backend::Cpu);
        let steady = obs(1_000.0, 700.0); // +30 % GPU advantage

        assert_eq!(hysteresis.observe(steady).reason, DecisionReason::Confirming);
        assert_eq!(hysteresis.observe(steady).reason, DecisionReason::Confirming);
        let decision = hysteresis.observe(steady);
        assert!(decision.switched);
        assert_eq!(decision.backend, Backend::Gpu);
        assert_eq!(hysteresis.switches(), 1);
    }

    #[test]
    fn hysteresis_never_oscillates_inside_the_deadband() {
        // Adversarial alternating signal just inside ±h, and a random signal
        // inside the same band; both must produce zero switches.
        let config = HysteresisConfig {
            gain_threshold: 0.18,
            confirm_samples: 3,
            min_dwell_samples: 5,
        };

        for initial in [Backend::Cpu, Backend::Gpu] {
            let mut alternating = Hysteresis::new(config, initial);
            for frame in 0..20_000 {
                let advantage = if frame % 2 == 0 { 0.179 } else { -0.179 };
                alternating.observe(obs(1_000.0, 1_000.0 * (1.0 - advantage)));
            }
            assert_eq!(alternating.switches(), 0, "alternating, initial={initial:?}");

            let mut lcg = Lcg(7);
            let mut noisy = Hysteresis::new(config, initial);
            for _ in 0..20_000 {
                let advantage = lcg.signed(0.179);
                noisy.observe(obs(1_000.0, 1_000.0 * (1.0 - advantage)));
            }
            assert_eq!(noisy.switches(), 0, "noisy, initial={initial:?}");
        }

        // Exactly at the threshold is inside the deadband because the
        // comparisons are strict.
        let mut edge = Hysteresis::new(config, Backend::Cpu);
        for _ in 0..100 {
            edge.observe(obs(1_000.0, 820.0)); // +18.0 %
        }
        assert_eq!(edge.switches(), 0);
    }

    #[test]
    fn hysteresis_confirm_counter_resets_on_sign_flip() {
        let config = HysteresisConfig {
            gain_threshold: 0.18,
            confirm_samples: 3,
            min_dwell_samples: 0,
        };
        let mut hysteresis = Hysteresis::new(config, Backend::Cpu);
        for _ in 0..1_000 {
            hysteresis.observe(obs(1_000.0, 700.0));
            hysteresis.observe(obs(1_000.0, 1_300.0));
        }
        assert_eq!(hysteresis.switches(), 0);
        assert_eq!(hysteresis.current(), Backend::Cpu);
    }

    #[test]
    fn hysteresis_respects_minimum_dwell() {
        let config = HysteresisConfig {
            gain_threshold: 0.18,
            confirm_samples: 1,
            min_dwell_samples: 5,
        };
        let mut hysteresis = Hysteresis::new(config, Backend::Cpu);
        let steady = obs(1_000.0, 700.0);
        for frame in 1..5 {
            let decision = hysteresis.observe(steady);
            assert!(!decision.switched);
            assert_eq!(decision.reason, DecisionReason::Dwell, "frame {frame}");
        }
        let decision = hysteresis.observe(steady);
        assert!(decision.switched);
        assert_eq!(hysteresis.switches(), 1);
    }

    #[test]
    fn hysteresis_switch_count_is_bounded_by_switch_period() {
        // Three frames of a strong GPU signal, then three of a strong CPU
        // signal, forever. With K=3, D=5 the state machine really switches
        // (roughly every max(K, D) = 5 frames after the first switch), so this
        // exercises the bound instead of passing vacuously: the switch count
        // must stay under T / max(K, D) + 1.
        let config = HysteresisConfig {
            gain_threshold: 0.18,
            confirm_samples: 3,
            min_dwell_samples: 5,
        };
        let mut hysteresis = Hysteresis::new(config, Backend::Cpu);
        for frame in 0..8_000 {
            let advantage = if frame % 6 < 3 { 0.5 } else { -0.5 };
            hysteresis.observe(obs(1_000.0, 1_000.0 * (1.0 - advantage)));
        }
        assert!(hysteresis.switches() > 100, "test must actually oscillate");
        let bound = 8_000 / u64::from(config.minimum_switch_period()) + 1;
        assert!(
            hysteresis.switches() <= bound,
            "switches={} bound={bound}",
            hysteresis.switches()
        );
    }

    #[test]
    fn hysteresis_switches_back_when_the_other_side_wins() {
        let config = HysteresisConfig {
            gain_threshold: 0.18,
            confirm_samples: 3,
            min_dwell_samples: 2,
        };
        let mut hysteresis = Hysteresis::new(config, Backend::Cpu);
        for _ in 0..3 {
            hysteresis.observe(obs(1_000.0, 700.0));
        }
        assert_eq!(hysteresis.current(), Backend::Gpu);
        for _ in 0..3 {
            hysteresis.observe(obs(1_000.0, 1_300.0));
        }
        assert_eq!(hysteresis.current(), Backend::Cpu);
        assert_eq!(hysteresis.switches(), 2);
    }

    #[test]
    fn hysteresis_ignores_invalid_observations() {
        let mut hysteresis = Hysteresis::new(HysteresisConfig::default(), Backend::Cpu);
        let decision = hysteresis.observe(obs(0.0, 1.0));
        assert_eq!(decision.reason, DecisionReason::NoObservation);
        assert_eq!(hysteresis.switches(), 0);
    }

    #[test]
    fn frame_boundary_gate_defers_until_boundary() {
        let mut gate = FrameBoundaryGate::default();
        let decision = Decision {
            backend: Backend::Gpu,
            switched: true,
            reason: DecisionReason::Switched,
            gpu_relative_advantage: Some(0.3),
        };
        gate.propose(decision);
        assert_eq!(gate.pending(), Some(Backend::Gpu));
        assert_eq!(gate.take_at_boundary(false), None);
        assert_eq!(gate.pending(), Some(Backend::Gpu));
        assert_eq!(gate.take_at_boundary(true), Some(Backend::Gpu));
        assert_eq!(gate.take_at_boundary(true), None);
    }

    #[test]
    fn frame_boundary_gate_ignores_non_switches() {
        let mut gate = FrameBoundaryGate::default();
        gate.propose(Decision {
            backend: Backend::Cpu,
            switched: false,
            reason: DecisionReason::Stable,
            gpu_relative_advantage: Some(0.0),
        });
        assert_eq!(gate.pending(), None);
    }

    #[test]
    fn crossover_prior_matches_the_source_policy_table() {
        let prior = PriorModel::from_crossover_table(&DEVELOP_CROSSOVER_PRIOR);
        assert_eq!(prior.crossover_floor(), Some(2_073_600));
        assert_eq!(prior.cold_placement(65_536), Backend::Cpu);
        assert_eq!(prior.cold_placement(921_600), Backend::Cpu);
        assert_eq!(prior.cold_placement(2_073_599), Backend::Cpu);
        assert_eq!(prior.cold_placement(2_073_600), Backend::Gpu);
        assert_eq!(prior.cold_placement(44_761_088), Backend::Gpu);
        assert_eq!(prior.cold_placement(100_000_000), Backend::Gpu);
        assert_eq!(prior.source(), PriorSource::UnmeasuredPolicyTable);
    }

    #[test]
    fn measured_crossover_replaces_the_unmeasured_prior() {
        let prior = PriorModel::from_measured_crossover(500_000);
        assert_eq!(prior.source(), PriorSource::MeasuredCalibration);
        assert_eq!(prior.cold_placement(499_999), Backend::Cpu);
        assert_eq!(prior.cold_placement(500_000), Backend::Gpu);
    }

    #[test]
    fn hysteresis_can_leave_the_cold_prior_once_measured() {
        // Prior says GPU at 1080p. A measured 30 % CPU advantage must still be
        // able to move the state machine back to CPU: the prior seeds the state,
        // it does not pin it.
        let prior = PriorModel::from_crossover_table(&DEVELOP_CROSSOVER_PRIOR);
        let initial = prior.cold_placement(2_073_600);
        assert_eq!(initial, Backend::Gpu);
        let mut hysteresis = Hysteresis::new(HysteresisConfig::default(), initial);
        for _ in 0..20 {
            hysteresis.observe(obs(1_000.0, 1_300.0));
        }
        assert_eq!(hysteresis.current(), Backend::Cpu);
        assert!(hysteresis.switches() >= 1);
    }

    #[test]
    fn failure_reasons_are_distinct_for_every_variant() {
        let failures = [
            GpuFailure::OutOfMemory {
                required_bytes: 512,
                budget_bytes: 256,
            },
            GpuFailure::DeviceLost { driver_code: 709 },
            GpuFailure::CompileFailed { stage: "develop" },
        ];

        let reasons: BTreeSet<_> = failures.iter().map(GpuFailure::reason).collect();
        assert_eq!(reasons.len(), failures.len());

        // Exhaustive match: adding a variant must update this test.
        for failure in failures {
            match failure {
                GpuFailure::OutOfMemory { .. } => {
                    assert_eq!(failure.reason(), FailureReason::OutOfMemory)
                }
                GpuFailure::DeviceLost { .. } => {
                    assert_eq!(failure.reason(), FailureReason::DeviceLost)
                }
                GpuFailure::CompileFailed { .. } => {
                    assert_eq!(failure.reason(), FailureReason::CompileFailed)
                }
            }
            assert!(!failure.retry_same_work_is_safe());
        }
    }

    #[test]
    fn recovery_actions_differ_by_failure_reason() {
        let oom = GpuFailure::OutOfMemory {
            required_bytes: 1,
            budget_bytes: 0,
        };
        let lost = GpuFailure::DeviceLost { driver_code: 700 };
        let compile = GpuFailure::CompileFailed { stage: "nr" };

        assert_eq!(
            oom.recovery(),
            RecoveryAction::CpuThisFrameThenReduceWorkingSet
        );
        assert_eq!(
            lost.recovery(),
            RecoveryAction::InvalidateContextAndReprobe
        );
        assert_eq!(
            compile.recovery(),
            RecoveryAction::DisableStageUntilFingerprintChanges
        );
        assert_ne!(oom.recovery(), lost.recovery());
        assert_ne!(lost.recovery(), compile.recovery());
        assert_ne!(oom.recovery(), compile.recovery());

        assert!(oom.stage_usable_after_recovery());
        assert!(lost.stage_usable_after_recovery());
        assert!(!compile.stage_usable_after_recovery());
    }

    fn sample_fingerprint() -> DeviceFingerprint {
        let mut options = BTreeMap::new();
        options.insert("opt_level".to_string(), "all".to_string());
        options.insert("cudnn".to_string(), "off".to_string());
        DeviceFingerprint {
            provider: ComputeProvider::Cuda,
            device_index: 0,
            vendor_id: Some(0x10de),
            device_id: Some(0x2684),
            name: "NVIDIA GeForce RTX 4090".to_string(),
            driver_version: "560.94".to_string(),
            architecture: "sm_89".to_string(),
            precision: Precision::F32,
            provider_options: options,
        }
    }

    #[test]
    fn fingerprint_stable_key_is_deterministic_and_sensitive() {
        let fingerprint = sample_fingerprint();
        assert_eq!(fingerprint.stable_key(), fingerprint.stable_key());
        assert_eq!(
            fingerprint.persistence_key_hex(),
            fingerprint.persistence_key_hex()
        );

        let mut different_driver = fingerprint.clone();
        different_driver.driver_version = "561.00".to_string();
        assert_ne!(fingerprint.stable_key(), different_driver.stable_key());
        assert_ne!(
            fingerprint.persistence_key_hex(),
            different_driver.persistence_key_hex()
        );

        let mut different_precision = fingerprint.clone();
        different_precision.precision = Precision::F16;
        assert_ne!(
            fingerprint.stable_key(),
            different_precision.stable_key()
        );

        // BTreeMap ordering makes option insertion order irrelevant.
        let mut reordered = fingerprint.clone();
        let options = std::mem::take(&mut reordered.provider_options);
        reordered.provider_options = options.into_iter().rev().collect();
        assert_eq!(fingerprint.stable_key(), reordered.stable_key());
    }

    #[test]
    fn calibration_record_roundtrips_through_json() {
        let record = CalibrationRecord {
            schema_version: CALIBRATION_SCHEMA_VERSION,
            fingerprint: sample_fingerprint(),
            crossover_pixels: Some(2_500_000),
            samples: 42,
            updated_unix_s: 1_763_000_000,
            notes: BTreeMap::from([("source".to_string(), "unit-test".to_string())]),
        };
        let json = record.to_json().unwrap();
        let decoded = CalibrationRecord::from_json(&json).unwrap();
        assert_eq!(record, decoded);
        assert!(decoded.is_compatible());

        let mut future = record;
        future.schema_version = CALIBRATION_SCHEMA_VERSION + 1;
        assert!(!future.is_compatible());
    }

    #[test]
    fn config_normalizes_bad_values() {
        let config = HysteresisConfig {
            gain_threshold: f64::NAN,
            confirm_samples: 0,
            min_dwell_samples: 0,
        }
        .normalized();
        assert_eq!(config.gain_threshold, 0.18);
        assert_eq!(config.confirm_samples, 1);
        assert_eq!(config.minimum_switch_period(), 1);
    }
}
