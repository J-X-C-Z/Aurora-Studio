//! Unified job lifecycle vocabulary.
//!
//! Preview, decode, model, export and background jobs currently use several
//! independent gates and registries. This module defines one small protocol —
//! scoped job id, state, cancellation token, progress, resource reservation and
//! dispatch evidence — that those gates can adopt without replacing their
//! working per-key single-flight or budget logic.
//!
//! # Status
//!
//! This is vocabulary plus a small in-process registry, **not** a wired
//! scheduler. No existing gate (`file_engine`, `merge_async`, `recipe`,
//! `mock/*`) is replaced in this round; an adapter can map a working gate's
//! per-key state onto [`JobId`]/[`JobState`] later. The registry owns no thread,
//! executor or timer: callers transition and prune it themselves.
//!
//! # Contracts
//!
//! - [`JobId`] is `(scope, serial)`. The serial is drawn from one process-wide
//!   counter, so ids are unique across every [`JobRegistry`] in a process. Within
//!   one scope serials are strictly increasing but not contiguous. Ordering
//!   compares scope first and serial second, so it only says "later within the
//!   same scope" for jobs that share a scope.
//! - [`JobState`] has three terminal states (`Cancelled`, `Failed`,
//!   `Succeeded`). No transition leaves a terminal state; reporting the same
//!   terminal state again is accepted as idempotent.
//! - Progress is clamped to `0.0 ..= 1.0`, and `NaN` becomes `0.0`. Progress is
//!   only meaningful while [`JobState::Running`]; every other state drops it.
//! - Cancellation is **cooperative**. [`CancelToken::cancel`] only sets a flag;
//!   nothing is preempted. A worker may poll [`CancelToken::check`] at safe
//!   boundaries: between pipeline stages, between batches, and before acquiring
//!   a device resource. It must not poll inside a kernel launch or while holding
//!   an exclusive device allocation; it lets the in-flight dispatch finish and
//!   releases reservations before reporting `Cancelled`.
//! - A job's generation is tracked per scope, separately from its serial. A
//!   result carrying an older generation than one already published for the
//!   scope must be discarded; see [`JobRegistry::publish_generation`] and
//!   [`JobRegistry::is_current`].
//!
//! # Resources
//!
//! [`ResourceReservation`] is a description of bytes, not an allocation. The
//! registry stores declarations and, when a job becomes terminal, turns each
//! into a [`ReservationRelease`] marker. Actually granting, accounting and
//! freeing memory stays with the caller's existing budget code.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::contracts::compute::{BackendKind, DispatchReceipt, FallbackReason};

/// The kind of work a job performs.
///
/// The fixed variants are the scopes the UI and the pipeline already reason
/// about. [`JobScope::Other`] lets a new adapter register a static scope name
/// without editing this enum; the name must be `'static` because job ids are
/// cheap `Copy` values, not owned strings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobScope {
    Preview,
    Thumbnail,
    Decode,
    Inference,
    Export,
    Merge,
    Import,
    Capture,
    /// An adapter-defined scope with a stable name.
    Other(&'static str),
}

impl JobScope {
    /// The scope's stable machine-readable name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Preview => "preview",
            Self::Thumbnail => "thumbnail",
            Self::Decode => "decode",
            Self::Inference => "inference",
            Self::Export => "export",
            Self::Merge => "merge",
            Self::Import => "import",
            Self::Capture => "capture",
            Self::Other(name) => name,
        }
    }
}

impl fmt::Display for JobScope {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// A job identity: scope plus a process-wide monotonic serial.
///
/// Construct ids only through [`JobRegistry::submit`] in production code; the
/// public fields exist so tests and evidence tooling can name an id without a
/// registry. Serialization is for evidence and UI snapshots. Deserialization is
/// deliberately not implemented: [`JobScope::Other`] holds a `&'static str`, and
/// reconstructing an arbitrary name would require leaking memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct JobId {
    pub scope: JobScope,
    pub serial: u64,
}

impl JobId {
    pub const fn new(scope: JobScope, serial: u64) -> Self {
        Self { scope, serial }
    }
}

impl fmt::Display for JobId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}:{}", self.scope, self.serial)
    }
}

/// Clamp a progress value. `NaN` is not a valid measurement, so it becomes
/// zero rather than poisoning comparisons downstream.
pub fn clamp_progress(progress: f32) -> f32 {
    if progress.is_nan() {
        0.0
    } else {
        progress.clamp(0.0, 1.0)
    }
}

/// The lifecycle state of one job.
///
/// Terminal states never transition back to a live state. `Running` carries a
/// clamped progress fraction; use [`JobState::running`] or the registry rather
/// than a struct literal when the value comes from a worker.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobState {
    Queued,
    Running { progress: f32 },
    Cancelling,
    Cancelled,
    Failed { error: String },
    Succeeded,
}

impl JobState {
    /// A running state with clamped progress (`NaN` becomes `0.0`).
    pub fn running(progress: f32) -> Self {
        Self::Running { progress: clamp_progress(progress) }
    }

    /// A failed state with a diagnostic message.
    pub fn failed(error: impl Into<String>) -> Self {
        Self::Failed { error: error.into() }
    }

    /// Clamp progress in a state that already exists.
    ///
    /// Every state except `Running` is returned unchanged, which is what
    /// "progress is ignored outside Running" means at the type level.
    pub fn sanitized(self) -> Self {
        match self {
            Self::Running { progress } => Self::running(progress),
            other => other,
        }
    }

    /// `true` for `Cancelled`, `Failed` and `Succeeded`.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Cancelled | Self::Failed { .. } | Self::Succeeded)
    }

    /// `true` for `Queued`, `Running` and `Cancelling`.
    pub fn is_live(&self) -> bool {
        !self.is_terminal()
    }

    /// The clamped progress fraction, or `None` when the state is not running.
    pub fn progress(&self) -> Option<f32> {
        match self {
            Self::Running { progress } => Some(clamp_progress(*progress)),
            _ => None,
        }
    }

    /// Whether the registry may replace `self` with `next`.
    ///
    /// Same-state transitions are idempotent: `Running -> Running` updates
    /// progress, `Cancelling -> Cancelling` is a no-op, and a terminal state may
    /// be re-reported with the exact same terminal value. Any move *out* of a
    /// terminal state, and any skip such as `Queued -> Succeeded`, is rejected.
    pub fn can_transition_to(&self, next: &Self) -> bool {
        if self.is_terminal() {
            return self == next;
        }
        match self {
            Self::Queued => matches!(
                next,
                Self::Running { .. } | Self::Cancelling | Self::Cancelled | Self::Failed { .. }
            ),
            Self::Running { .. } => matches!(
                next,
                Self::Running { .. }
                    | Self::Cancelling
                    | Self::Cancelled
                    | Self::Failed { .. }
                    | Self::Succeeded
            ),
            Self::Cancelling => matches!(
                next,
                Self::Cancelling | Self::Cancelled | Self::Failed { .. } | Self::Succeeded
            ),
            Self::Cancelled | Self::Failed { .. } | Self::Succeeded => false,
        }
    }
}

impl fmt::Display for JobState {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Queued => "queued",
            Self::Running { .. } => "running",
            Self::Cancelling => "cancelling",
            Self::Cancelled => "cancelled",
            Self::Failed { .. } => "failed",
            Self::Succeeded => "succeeded",
        })
    }
}

/// A cooperative cancellation flag shared between the UI and a worker.
///
/// The token is cheap to clone; all clones observe the same atomic flag. It does
/// not stop work by itself: see the module documentation for the safe-boundary
/// polling contract.
#[derive(Clone, Debug, Default)]
pub struct CancelToken {
    cancelled: Arc<AtomicBool>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation. Idempotent; the flag never clears.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Whether cancellation has been requested, from any clone.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// The polling hook for a safe boundary.
    ///
    /// Returns [`JobError::Cancelled`] once cancellation is requested. Callers
    /// should propagate it instead of finishing partial work and reporting it as
    /// success.
    pub fn check(&self) -> Result<(), JobError> {
        if self.is_cancelled() {
            Err(JobError::Cancelled)
        } else {
            Ok(())
        }
    }
}

/// A typed job failure.
///
/// The categories exist so the UI and diagnostics can distinguish "the user's
/// input is wrong" from "the machine ran out of resources" from "the device
/// vanished". None of them is converted to a default value; every constructor
/// keeps the message it was given.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum JobError {
    /// The request itself is invalid (bad range, missing selection, ...).
    UserInput(String),
    /// A stage or format is not supported, with a concrete reason.
    Unsupported { what: &'static str, reason: String },
    /// A named resource pool could not satisfy the request.
    ResourceExhausted { what: &'static str },
    /// Filesystem, permission or decode/encode I/O failed.
    Io(String),
    /// The compute device was lost mid-job; retry is a policy decision.
    DeviceLost,
    /// The algorithm refused or failed to produce a result.
    Algorithm(String),
    /// The cooperative cancellation token was observed at a safe boundary.
    Cancelled,
    /// An invariant was violated; this is a bug, not user input.
    Internal(String),
}

impl JobError {
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }
}

impl fmt::Display for JobError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UserInput(message) => write!(formatter, "invalid user input: {message}"),
            Self::Unsupported { what, reason } => {
                write!(formatter, "unsupported {what}: {reason}")
            }
            Self::ResourceExhausted { what } => write!(formatter, "resource exhausted: {what}"),
            Self::Io(message) => write!(formatter, "I/O error: {message}"),
            Self::DeviceLost => formatter.write_str("compute device lost"),
            Self::Algorithm(message) => write!(formatter, "algorithm error: {message}"),
            Self::Cancelled => formatter.write_str("job cancelled"),
            Self::Internal(message) => write!(formatter, "internal error: {message}"),
        }
    }
}

impl std::error::Error for JobError {}

impl From<std::io::Error> for JobError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

/// Why a state change was refused.
///
/// `PartialEq` only: [`JobState`] carries a float progress fraction and therefore
/// cannot implement `Eq`.
#[derive(Clone, Debug, PartialEq)]
pub enum TransitionError {
    /// The id was never submitted, or was pruned.
    UnknownJob(JobId),
    /// The requested transition violates the lifecycle rules.
    Invalid { id: JobId, from: JobState, to: JobState },
    /// A reservation or similar operation was attempted after a terminal state.
    TerminalJob(JobId),
}

impl fmt::Display for TransitionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownJob(id) => write!(formatter, "unknown job {id}"),
            Self::Invalid { id, from, to } => {
                write!(formatter, "job {id} cannot transition from {from} to {to}")
            }
            Self::TerminalJob(id) => write!(formatter, "job {id} is already terminal"),
        }
    }
}

impl std::error::Error for TransitionError {}

/// A progress report from a worker.
///
/// Build it with [`JobProgress::new`] so out-of-range and `NaN` fractions are
/// clamped. The stage is a static name (for example `"decode"`), not a formatted
/// string, so reports stay allocation-free.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct JobProgress {
    pub fraction: f32,
    pub stage: &'static str,
}

impl JobProgress {
    pub fn new(fraction: f32, stage: &'static str) -> Self {
        Self { fraction: clamp_progress(fraction), stage }
    }

    /// Clamp the fraction in place.
    pub fn sanitized(self) -> Self {
        Self::new(self.fraction, self.stage)
    }
}

/// A declared resource need. This is **not** an allocation and does not reserve
/// anything by itself; the registry only keeps the declaration for accounting
/// and diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ResourceReservation {
    pub cpu_bytes: u64,
    pub device_bytes: u64,
    pub pinned_bytes: u64,
    /// Caller-defined identity used to correlate the release marker.
    pub ticket: u64,
}

impl ResourceReservation {
    pub const fn new(ticket: u64, cpu_bytes: u64, device_bytes: u64, pinned_bytes: u64) -> Self {
        Self { cpu_bytes, device_bytes, pinned_bytes, ticket }
    }

    /// Pageable plus pinned host bytes, saturating instead of wrapping.
    ///
    /// `device_bytes` is excluded because it is not host memory.
    pub fn host_bytes(&self) -> u64 {
        self.cpu_bytes.saturating_add(self.pinned_bytes)
    }

    /// Turn the declaration into a release marker.
    ///
    /// The marker proves that the caller acknowledged the reservation is over.
    /// It does not free memory; the allocator remains the caller's concern.
    pub fn release(self) -> ReservationRelease {
        ReservationRelease { ticket: self.ticket }
    }
}

/// Proof that a [`ResourceReservation`] was released (or dropped at terminal).
///
/// It only carries the ticket. Do not treat it as a memory operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct ReservationRelease {
    pub ticket: u64,
}

/// A backend-neutral record of one dispatch, mirroring the cut-down fields the
/// UI needs from [`DispatchReceipt`].
///
/// `backend` is a static label instead of [`BackendKind`] so evidence can be
/// serialized cheaply without dragging the contracts enum into UI state. Convert
/// a real receipt with [`DispatchEvidence::from_receipt`]; that also reuses
/// [`FallbackReason`] instead of duplicating its variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct DispatchEvidence {
    /// Backend label, e.g. `"cpu"` or `"cuda"`.
    pub backend: &'static str,
    /// Kernel entry point, when the backend names kernels.
    pub kernel: Option<&'static str>,
    /// Launch grid, backend-defined meaning.
    pub grid: Option<(u32, u32)>,
    /// Launch block, backend-defined meaning.
    pub block: Option<(u32, u32)>,
    /// Whether the dispatch completed; `false` means refused or failed.
    pub completed: bool,
    /// Operation identifier, when known.
    pub operation: Option<&'static str>,
    /// Why the preferred backend was not used, if it fell back.
    pub fallback: Option<FallbackReason>,
}

impl DispatchEvidence {
    /// Convert an existing compute receipt into job evidence.
    pub fn from_receipt(receipt: DispatchReceipt) -> Self {
        Self {
            backend: backend_label(receipt.backend),
            kernel: receipt.kernel,
            grid: receipt.grid,
            block: receipt.block,
            completed: receipt.completed,
            operation: Some(receipt.operation.as_str()),
            fallback: None,
        }
    }

    /// Attach the fallback reason a receipt does not carry.
    pub fn with_fallback(mut self, fallback: Option<FallbackReason>) -> Self {
        self.fallback = fallback;
        self
    }
}

impl From<DispatchReceipt> for DispatchEvidence {
    fn from(receipt: DispatchReceipt) -> Self {
        Self::from_receipt(receipt)
    }
}

/// A stable label for a backend family.
pub const fn backend_label(backend: BackendKind) -> &'static str {
    match backend {
        BackendKind::Cpu => "cpu",
        BackendKind::Cuda => "cuda",
        BackendKind::Metal => "metal",
        BackendKind::Vulkan => "vulkan",
        BackendKind::Dx12 => "dx12",
        BackendKind::Hip => "hip",
        BackendKind::Sycl => "sycl",
        BackendKind::OpenCl => "opencl",
        BackendKind::Npu => "npu",
    }
}

/// Everything the registry knows about one job.
#[derive(Clone, Debug)]
pub struct JobRecord {
    id: JobId,
    state: JobState,
    cancel: CancelToken,
    reservations: Vec<ResourceReservation>,
    released: Vec<ReservationRelease>,
    evidence: Vec<DispatchEvidence>,
    generation: Option<u64>,
}

impl JobRecord {
    fn new(id: JobId, cancel: CancelToken) -> Self {
        Self {
            id,
            state: JobState::Queued,
            cancel,
            reservations: Vec::new(),
            released: Vec::new(),
            evidence: Vec::new(),
            generation: None,
        }
    }

    pub fn id(&self) -> JobId {
        self.id
    }

    pub fn state(&self) -> &JobState {
        &self.state
    }

    pub fn cancel_token(&self) -> &CancelToken {
        &self.cancel
    }

    /// Declared reservations still outstanding.
    pub fn reservations(&self) -> &[ResourceReservation] {
        &self.reservations
    }

    /// Markers produced when reservations were released at a terminal state.
    pub fn released(&self) -> &[ReservationRelease] {
        &self.released
    }

    /// Dispatch receipts recorded for this job, in arrival order.
    pub fn evidence(&self) -> &[DispatchEvidence] {
        &self.evidence
    }

    /// The last generation accepted for this job, if any.
    pub fn generation(&self) -> Option<u64> {
        self.generation
    }
}

/// Per-scope counters used by [`RegistrySnapshot`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScopeCounts {
    pub total: usize,
    pub queued: usize,
    pub running: usize,
    pub cancelling: usize,
    pub cancelled: usize,
    pub failed: usize,
    pub succeeded: usize,
}

impl ScopeCounts {
    /// `Queued + Running + Cancelling`.
    pub fn live(&self) -> usize {
        self.queued + self.running + self.cancelling
    }

    /// `Cancelled + Failed + Succeeded`.
    pub fn terminal(&self) -> usize {
        self.cancelled + self.failed + self.succeeded
    }

    fn observe(&mut self, state: &JobState) {
        self.total += 1;
        match state {
            JobState::Queued => self.queued += 1,
            JobState::Running { .. } => self.running += 1,
            JobState::Cancelling => self.cancelling += 1,
            JobState::Cancelled => self.cancelled += 1,
            JobState::Failed { .. } => self.failed += 1,
            JobState::Succeeded => self.succeeded += 1,
        }
    }
}

/// Cheap copy of registry counts for the UI.
///
/// `per_scope` is a `BTreeMap` keyed by [`JobScope`]; it is ordered by scope so
/// snapshots are deterministic. No job payloads or evidence are cloned.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RegistrySnapshot {
    pub total: usize,
    pub queued: usize,
    pub running: usize,
    pub cancelling: usize,
    pub cancelled: usize,
    pub failed: usize,
    pub succeeded: usize,
    pub per_scope: BTreeMap<JobScope, ScopeCounts>,
}

impl RegistrySnapshot {
    /// `Queued + Running + Cancelling` across all scopes.
    pub fn live(&self) -> usize {
        self.queued + self.running + self.cancelling
    }

    /// `Cancelled + Failed + Succeeded` across all scopes.
    pub fn terminal(&self) -> usize {
        self.cancelled + self.failed + self.succeeded
    }

    /// Counters for one scope, or `None` when no job has that scope.
    pub fn scope(&self, scope: JobScope) -> Option<&ScopeCounts> {
        self.per_scope.get(&scope)
    }
}

/// Next process-wide job serial. Starting at 1 keeps zero available for
/// "unset" in evidence tooling.
static NEXT_SERIAL: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Default)]
struct Inner {
    jobs: BTreeMap<JobId, JobRecord>,
    latest_generation: BTreeMap<JobScope, u64>,
}

/// A thread-safe registry of job records.
///
/// The registry is a `Clone` handle around one `parking_lot::Mutex`; cloning it
/// does not clone records, so a UI thread and a worker can share it freely. Every
/// method takes the lock briefly and never calls back into user code while
/// holding it, which keeps snapshot polling and worker transitions from
/// deadlocking.
#[derive(Clone, Debug, Default)]
pub struct JobRegistry {
    inner: Arc<Mutex<Inner>>,
}

impl JobRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new queued job.
    ///
    /// The serial comes from a process-wide atomic, so the returned id is unique
    /// across every registry in the process. Serials within a scope are strictly
    /// increasing but may have gaps.
    pub fn submit(&self, scope: JobScope, cancel: CancelToken) -> JobId {
        let serial = NEXT_SERIAL.fetch_add(1, Ordering::Relaxed);
        let id = JobId::new(scope, serial);
        let mut inner = self.inner.lock();
        inner.jobs.insert(id, JobRecord::new(id, cancel));
        id
    }

    /// Move a job to `new_state`, rejecting unknown ids and illegal edges.
    ///
    /// `Running` progress is clamped before storage. Reaching a terminal state
    /// releases every outstanding reservation, converting each into a
    /// [`ReservationRelease`] marker stored on the record. Re-reporting an
    /// already-terminal state identically is an accepted no-op; any other move
    /// out of it is [`TransitionError::Invalid`].
    pub fn transition(
        &self,
        id: JobId,
        new_state: JobState,
    ) -> Result<(), TransitionError> {
        let new_state = new_state.sanitized();
        let mut inner = self.inner.lock();
        let record = inner
            .jobs
            .get_mut(&id)
            .ok_or(TransitionError::UnknownJob(id))?;
        if !record.state.can_transition_to(&new_state) {
            return Err(TransitionError::Invalid {
                id,
                from: record.state.clone(),
                to: new_state,
            });
        }
        if record.state == new_state {
            return Ok(());
        }
        if new_state.is_terminal() {
            let reservations = std::mem::take(&mut record.reservations);
            record
                .released
                .extend(reservations.into_iter().map(ResourceReservation::release));
        }
        record.state = new_state;
        Ok(())
    }

    /// Request cooperative cancellation of one job.
    ///
    /// Sets the job's [`CancelToken`] and moves a live job to
    /// [`JobState::Cancelling`]. Returns `true` when this call newly requested
    /// cancellation, `false` when the job was already `Cancelling` or already
    /// terminal. Terminal records are never moved and their tokens are not
    /// touched. Unknown ids are an error.
    pub fn cancel(&self, id: JobId) -> Result<bool, TransitionError> {
        let mut inner = self.inner.lock();
        let record = inner
            .jobs
            .get_mut(&id)
            .ok_or(TransitionError::UnknownJob(id))?;
        if record.state.is_terminal() {
            return Ok(false);
        }
        record.cancel.cancel();
        if matches!(record.state, JobState::Cancelling) {
            return Ok(false);
        }
        record.state = JobState::Cancelling;
        Ok(true)
    }

    /// Cancel every live job in `scope`; other scopes and terminal jobs are
    /// untouched. Returns the number of jobs newly moved to `Cancelling`.
    pub fn cancel_scope(&self, scope: JobScope) -> usize {
        let mut inner = self.inner.lock();
        let mut newly_cancelled = 0;
        for record in inner.jobs.values_mut() {
            if record.id.scope != scope || record.state.is_terminal() {
                continue;
            }
            record.cancel.cancel();
            if !matches!(record.state, JobState::Cancelling) {
                record.state = JobState::Cancelling;
                newly_cancelled += 1;
            }
        }
        newly_cancelled
    }

    /// Append one dispatch receipt to a job's evidence.
    ///
    /// Allowed after a terminal state: a dispatch that was already in flight may
    /// report its receipt while the job is being cancelled.
    pub fn record(
        &self,
        id: JobId,
        evidence: DispatchEvidence,
    ) -> Result<(), TransitionError> {
        let mut inner = self.inner.lock();
        let record = inner
            .jobs
            .get_mut(&id)
            .ok_or(TransitionError::UnknownJob(id))?;
        record.evidence.push(evidence);
        Ok(())
    }

    /// Declare a resource reservation for a live job.
    ///
    /// Declarations are additive; a job may reserve several times (for example a
    /// CPU budget and a device budget). Nothing is allocated. Terminal jobs
    /// reject new declarations with [`TransitionError::TerminalJob`].
    pub fn reservation(
        &self,
        id: JobId,
        reservation: ResourceReservation,
    ) -> Result<(), TransitionError> {
        let mut inner = self.inner.lock();
        let record = inner
            .jobs
            .get_mut(&id)
            .ok_or(TransitionError::UnknownJob(id))?;
        if record.state.is_terminal() {
            return Err(TransitionError::TerminalJob(id));
        }
        record.reservations.push(reservation);
        Ok(())
    }

    /// Release all outstanding reservations before the job is terminal.
    ///
    /// Returns the markers; the record keeps them for diagnostics and clears its
    /// live reservation list. Unknown ids are an error.
    pub fn release_reservations(
        &self,
        id: JobId,
    ) -> Result<Vec<ReservationRelease>, TransitionError> {
        let mut inner = self.inner.lock();
        let record = inner
            .jobs
            .get_mut(&id)
            .ok_or(TransitionError::UnknownJob(id))?;
        let reservations = std::mem::take(&mut record.reservations);
        let released: Vec<_> = reservations
            .into_iter()
            .map(ResourceReservation::release)
            .collect();
        record.released.extend(released.iter().copied());
        Ok(released)
    }

    /// Publish a result generation for the job's scope and say whether it wins.
    ///
    /// Generations are tracked per scope, independently of job serials. A
    /// generation strictly newer than the scope's latest is stored and accepted;
    /// an equal generation is accepted without changing the tracker; an older
    /// generation is refused and must not overwrite anything. The accepted value
    /// is also recorded on the job.
    ///
    /// Unknown ids are an error.
    pub fn publish_generation(
        &self,
        id: JobId,
        generation: u64,
    ) -> Result<bool, TransitionError> {
        let mut inner = self.inner.lock();
        if !inner.jobs.contains_key(&id) {
            return Err(TransitionError::UnknownJob(id));
        }
        let latest = inner.latest_generation.get(&id.scope).copied();
        let accepted = latest.is_none_or(|latest| generation >= latest);
        if accepted {
            if latest != Some(generation) {
                inner.latest_generation.insert(id.scope, generation);
            }
            if let Some(record) = inner.jobs.get_mut(&id) {
                record.generation = Some(generation);
            }
        }
        Ok(accepted)
    }

    /// Whether `generation` is not older than the scope's latest published one.
    ///
    /// This is a pure read. Unknown ids, including pruned jobs, return `false`
    /// so a stale worker cannot claim currency for a job the registry no longer
    /// knows. Use [`Self::publish_generation`] to advance the tracker.
    pub fn is_current(&self, id: JobId, generation: u64) -> bool {
        let inner = self.inner.lock();
        if !inner.jobs.contains_key(&id) {
            return false;
        }
        match inner.latest_generation.get(&id.scope) {
            Some(&latest) => generation >= latest,
            None => true,
        }
    }

    /// The latest generation published for a scope, if any.
    pub fn latest_generation(&self, scope: JobScope) -> Option<u64> {
        self.inner.lock().latest_generation.get(&scope).copied()
    }

    /// A cheap counts snapshot for the UI.
    pub fn snapshot(&self) -> RegistrySnapshot {
        let inner = self.inner.lock();
        let mut snapshot = RegistrySnapshot::default();
        for record in inner.jobs.values() {
            snapshot.total += 1;
            let scope = snapshot
                .per_scope
                .entry(record.id.scope)
                .or_insert_with(ScopeCounts::default);
            scope.observe(&record.state);
            match &record.state {
                JobState::Queued => snapshot.queued += 1,
                JobState::Running { .. } => snapshot.running += 1,
                JobState::Cancelling => snapshot.cancelling += 1,
                JobState::Cancelled => snapshot.cancelled += 1,
                JobState::Failed { .. } => snapshot.failed += 1,
                JobState::Succeeded => snapshot.succeeded += 1,
            }
        }
        snapshot
    }

    /// Retain at most `keep` terminal records, dropping the oldest by serial.
    ///
    /// Live jobs are never pruned. The limit is global across scopes: the
    /// `keep` most recently submitted terminal jobs survive. Returns how many
    /// records were removed. Pruned ids become unknown, so late worker results
    /// must be treated as stale.
    pub fn prune_terminal(&self, keep: usize) -> usize {
        let mut inner = self.inner.lock();
        let mut terminal: Vec<(u64, JobId)> = inner
            .jobs
            .values()
            .filter(|record| record.state.is_terminal())
            .map(|record| (record.id.serial, record.id))
            .collect();
        if terminal.len() <= keep {
            return 0;
        }
        terminal.sort_unstable_by_key(|(serial, _)| *serial);
        let remove = terminal.len() - keep;
        for (_, id) in terminal.into_iter().take(remove) {
            inner.jobs.remove(&id);
        }
        remove
    }

    /// Clone one record for inspection or tests.
    pub fn job(&self, id: JobId) -> Option<JobRecord> {
        self.inner.lock().jobs.get(&id).cloned()
    }

    /// Whether the registry currently holds `id`.
    pub fn contains(&self, id: JobId) -> bool {
        self.inner.lock().jobs.contains_key(&id)
    }

    /// Number of retained records, live plus terminal.
    pub fn len(&self) -> usize {
        self.inner.lock().jobs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().jobs.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::compute::OperationId;

    fn state(registry: &JobRegistry, id: JobId) -> JobState {
        registry.job(id).expect("job should exist").state().clone()
    }

    fn evidence() -> DispatchEvidence {
        DispatchEvidence {
            backend: "cpu",
            kernel: None,
            grid: None,
            block: None,
            completed: true,
            operation: Some("test"),
            fallback: None,
        }
    }

    #[test]
    fn valid_transition_sequence() {
        let registry = JobRegistry::new();
        let id = registry.submit(JobScope::Preview, CancelToken::new());
        assert_eq!(state(&registry, id), JobState::Queued);

        registry
            .transition(id, JobState::running(0.25))
            .expect("queued -> running");
        assert_eq!(state(&registry, id), JobState::running(0.25));
        assert_eq!(state(&registry, id).progress(), Some(0.25));

        registry
            .transition(id, JobState::running(0.75))
            .expect("running -> running");
        assert_eq!(state(&registry, id).progress(), Some(0.75));

        registry
            .transition(id, JobState::Succeeded)
            .expect("running -> succeeded");
        assert!(state(&registry, id).is_terminal());
        assert_eq!(state(&registry, id).progress(), None);
        assert_eq!(registry.job(id).expect("record").released(), &[]);
    }

    #[test]
    fn terminal_states_do_not_transition_out() {
        let registry = JobRegistry::new();

        let succeeded = registry.submit(JobScope::Export, CancelToken::new());
        registry
            .transition(succeeded, JobState::running(0.5))
            .expect("running");
        registry
            .transition(succeeded, JobState::Succeeded)
            .expect("succeeded");
        assert!(matches!(
            registry.transition(succeeded, JobState::running(0.9)),
            Err(TransitionError::Invalid { from: JobState::Succeeded, .. })
        ));
        assert!(matches!(
            registry.transition(succeeded, JobState::Cancelling),
            Err(TransitionError::Invalid { .. })
        ));
        // Re-reporting the same terminal state is idempotent, not a reset.
        registry
            .transition(succeeded, JobState::Succeeded)
            .expect("idempotent terminal report");

        let failed = registry.submit(JobScope::Decode, CancelToken::new());
        registry
            .transition(failed, JobState::failed("decode failed"))
            .expect("queued -> failed");
        assert!(matches!(
            registry.transition(failed, JobState::Succeeded),
            Err(TransitionError::Invalid { from: JobState::Failed { .. }, .. })
        ));
        assert!(matches!(
            registry.transition(failed, JobState::running(0.1)),
            Err(TransitionError::Invalid { .. })
        ));

        let cancelled = registry.submit(JobScope::Merge, CancelToken::new());
        registry.cancel(cancelled).expect("cancel");
        registry
            .transition(cancelled, JobState::Cancelled)
            .expect("cancelling -> cancelled");
        assert!(matches!(
            registry.transition(cancelled, JobState::Queued),
            Err(TransitionError::Invalid { from: JobState::Cancelled, .. })
        ));
        assert!(matches!(
            registry.transition(cancelled, JobState::Succeeded),
            Err(TransitionError::Invalid { .. })
        ));
    }

    #[test]
    fn unknown_job_is_rejected_everywhere() {
        let registry = JobRegistry::new();
        let id = JobId::new(JobScope::Other("adapter-scope"), u64::MAX);

        assert_eq!(
            registry.transition(id, JobState::Queued),
            Err(TransitionError::UnknownJob(id))
        );
        assert_eq!(registry.cancel(id), Err(TransitionError::UnknownJob(id)));
        assert_eq!(
            registry.record(id, evidence()),
            Err(TransitionError::UnknownJob(id))
        );
        let reservation = ResourceReservation::new(1, 1, 0, 0);
        assert_eq!(
            registry.reservation(id, reservation),
            Err(TransitionError::UnknownJob(id))
        );
        assert_eq!(
            registry.release_reservations(id),
            Err(TransitionError::UnknownJob(id))
        );
        assert_eq!(
            registry.publish_generation(id, 1),
            Err(TransitionError::UnknownJob(id))
        );
        assert!(!registry.is_current(id, 1));
        assert_eq!(registry.cancel_scope(JobScope::Other("adapter-scope")), 0);
        assert!(!registry.contains(id));
    }

    #[test]
    fn cancel_before_start_during_running_and_after_terminal() {
        let registry = JobRegistry::new();

        let before = registry.submit(JobScope::Decode, CancelToken::new());
        let before_token = registry
            .job(before)
            .expect("record")
            .cancel_token()
            .clone();
        assert_eq!(registry.cancel(before), Ok(true));
        assert_eq!(state(&registry, before), JobState::Cancelling);
        assert!(before_token.is_cancelled());
        assert_eq!(before_token.check(), Err(JobError::Cancelled));
        registry
            .transition(before, JobState::Cancelled)
            .expect("cancelling -> cancelled");

        let during = registry.submit(JobScope::Inference, CancelToken::new());
        let during_token = registry
            .job(during)
            .expect("record")
            .cancel_token()
            .clone();
        registry
            .transition(during, JobState::running(0.4))
            .expect("running");
        assert_eq!(registry.cancel(during), Ok(true));
        assert_eq!(state(&registry, during), JobState::Cancelling);
        assert!(during_token.is_cancelled());
        registry
            .transition(during, JobState::Cancelled)
            .expect("cancelling -> cancelled");

        let after = registry.submit(JobScope::Export, CancelToken::new());
        registry
            .transition(after, JobState::running(1.0))
            .expect("running");
        registry
            .transition(after, JobState::Succeeded)
            .expect("succeeded");
        assert_eq!(registry.cancel(after), Ok(false));
        assert_eq!(state(&registry, after), JobState::Succeeded);
        assert!(matches!(
            registry.transition(after, JobState::Cancelling),
            Err(TransitionError::Invalid { .. })
        ));
    }

    #[test]
    fn scope_wide_cancel_does_not_touch_another_scope() {
        let registry = JobRegistry::new();
        let preview_a = registry.submit(JobScope::Preview, CancelToken::new());
        let preview_b = registry.submit(JobScope::Preview, CancelToken::new());
        let thumbnail = registry.submit(JobScope::Thumbnail, CancelToken::new());
        let terminal = registry.submit(JobScope::Preview, CancelToken::new());

        registry
            .transition(preview_a, JobState::running(0.3))
            .expect("running");
        registry
            .transition(terminal, JobState::failed("gone"))
            .expect("failed");

        assert_eq!(registry.cancel_scope(JobScope::Preview), 2);
        assert_eq!(state(&registry, preview_a), JobState::Cancelling);
        assert_eq!(state(&registry, preview_b), JobState::Cancelling);
        assert_eq!(state(&registry, thumbnail), JobState::Queued);
        assert_eq!(state(&registry, terminal), JobState::failed("gone"));

        let thumbnail_token = registry
            .job(thumbnail)
            .expect("record")
            .cancel_token()
            .clone();
        assert!(!thumbnail_token.is_cancelled());

        // Re-cancelling an already-cancelling scope reports no new work.
        assert_eq!(registry.cancel_scope(JobScope::Preview), 0);
    }

    #[test]
    fn stale_generation_is_refused_and_newer_is_accepted() {
        let registry = JobRegistry::new();
        let id = registry.submit(JobScope::Preview, CancelToken::new());

        assert_eq!(registry.publish_generation(id, 10), Ok(true));
        assert!(registry.is_current(id, 10));
        assert_eq!(registry.latest_generation(JobScope::Preview), Some(10));
        assert_eq!(registry.job(id).expect("record").generation(), Some(10));

        assert_eq!(registry.publish_generation(id, 9), Ok(false));
        assert!(!registry.is_current(id, 9));
        assert_eq!(registry.latest_generation(JobScope::Preview), Some(10));
        assert_eq!(registry.job(id).expect("record").generation(), Some(10));

        assert_eq!(registry.publish_generation(id, 11), Ok(true));
        assert!(registry.is_current(id, 11));
        assert!(!registry.is_current(id, 10));
        assert_eq!(registry.latest_generation(JobScope::Preview), Some(11));

        // Generations are per scope; a different scope starts independent.
        let thumbnail = registry.submit(JobScope::Thumbnail, CancelToken::new());
        assert_eq!(registry.publish_generation(thumbnail, 1), Ok(true));
        assert!(registry.is_current(thumbnail, 1));
        assert_eq!(registry.latest_generation(JobScope::Preview), Some(11));
        assert_eq!(registry.latest_generation(JobScope::Thumbnail), Some(1));

        // A pruned or never-submitted id is never current.
        assert!(!registry.is_current(
            JobId::new(JobScope::Preview, u64::MAX - 1),
            12
        ));
    }

    #[test]
    fn snapshot_counts_and_pruning() {
        let registry = JobRegistry::new();
        let queued = registry.submit(JobScope::Preview, CancelToken::new());
        let running = registry.submit(JobScope::Preview, CancelToken::new());
        let cancelling = registry.submit(JobScope::Decode, CancelToken::new());
        let succeeded = registry.submit(JobScope::Export, CancelToken::new());
        let failed = registry.submit(JobScope::Export, CancelToken::new());

        registry
            .transition(running, JobState::running(0.5))
            .expect("running");
        registry.cancel(cancelling).expect("cancel");
        registry
            .transition(succeeded, JobState::running(1.0))
            .expect("running");
        registry
            .transition(succeeded, JobState::Succeeded)
            .expect("succeeded");
        registry
            .transition(failed, JobState::failed("export failed"))
            .expect("failed");

        let snapshot = registry.snapshot();
        assert_eq!(snapshot.total, 5);
        assert_eq!(snapshot.queued, 1);
        assert_eq!(snapshot.running, 1);
        assert_eq!(snapshot.cancelling, 1);
        assert_eq!(snapshot.cancelled, 0);
        assert_eq!(snapshot.failed, 1);
        assert_eq!(snapshot.succeeded, 1);
        assert_eq!(snapshot.live(), 3);
        assert_eq!(snapshot.terminal(), 2);

        let preview = snapshot.scope(JobScope::Preview).expect("preview counts");
        assert_eq!(preview.total, 2);
        assert_eq!(preview.queued, 1);
        assert_eq!(preview.running, 1);
        assert_eq!(preview.live(), 2);
        let export = snapshot.scope(JobScope::Export).expect("export counts");
        assert_eq!(export.total, 2);
        assert_eq!(export.failed, 1);
        assert_eq!(export.succeeded, 1);
        assert_eq!(export.terminal(), 2);

        // Two terminals; keep the newest by serial and drop the older one.
        assert_eq!(registry.prune_terminal(1), 1);
        assert!(registry.contains(failed));
        assert!(!registry.contains(succeeded));
        assert!(registry.contains(queued));
        assert!(registry.contains(running));
        assert!(registry.contains(cancelling));
        assert_eq!(registry.len(), 4);
        assert_eq!(registry.snapshot().terminal(), 1);

        // Already within the bound, and live jobs are never removed.
        assert_eq!(registry.prune_terminal(1), 0);
        assert_eq!(registry.prune_terminal(0), 1);
        assert_eq!(registry.len(), 3);
        assert!(registry.contains(queued));
        assert!(registry.contains(running));
        assert!(registry.contains(cancelling));
    }

    #[test]
    fn registry_is_usable_from_two_threads() {
        use std::thread;

        let registry = JobRegistry::new();
        let worker_job = registry.submit(JobScope::Preview, CancelToken::new());
        let ui_job = registry.submit(JobScope::Preview, CancelToken::new());

        let worker_registry = registry.clone();
        let worker = thread::spawn(move || {
            worker_registry
                .transition(worker_job, JobState::running(0.5))
                .expect("worker running");
            worker_registry
                .transition(worker_job, JobState::Succeeded)
                .expect("worker succeeded");
            worker_registry.job(worker_job).expect("worker record").id()
        });

        // Poll the snapshot while the worker mutates, then mutate from here too.
        for _ in 0..100 {
            let snapshot = registry.snapshot();
            assert!(snapshot.total >= 2);
        }
        registry
            .transition(ui_job, JobState::running(0.25))
            .expect("ui running");
        assert_eq!(worker.join().expect("worker joins"), worker_job);

        assert_eq!(state(&registry, worker_job), JobState::Succeeded);
        assert_eq!(state(&registry, ui_job), JobState::running(0.25));
        assert_eq!(registry.snapshot().succeeded, 1);
        assert_eq!(registry.snapshot().running, 1);
    }

    #[test]
    fn progress_is_clamped_and_nan_becomes_zero() {
        assert_eq!(JobState::running(-0.5).progress(), Some(0.0));
        assert_eq!(JobState::running(0.0).progress(), Some(0.0));
        assert_eq!(JobState::running(0.5).progress(), Some(0.5));
        assert_eq!(JobState::running(1.5).progress(), Some(1.0));
        assert_eq!(JobState::running(f32::NAN).progress(), Some(0.0));
        assert_eq!(JobState::running(f32::INFINITY).progress(), Some(1.0));
        assert_eq!(JobState::running(f32::NEG_INFINITY).progress(), Some(0.0));

        assert_eq!(JobState::Queued.progress(), None);
        assert_eq!(JobState::Cancelling.progress(), None);
        assert_eq!(JobState::Cancelled.progress(), None);
        assert_eq!(JobState::failed("x").progress(), None);
        assert_eq!(JobState::Succeeded.progress(), None);

        assert_eq!(JobProgress::new(f32::NAN, "decode").fraction, 0.0);
        assert_eq!(JobProgress::new(-3.0, "decode").fraction, 0.0);
        assert_eq!(JobProgress::new(9.0, "decode").fraction, 1.0);
        assert_eq!(
            JobProgress::new(9.0, "decode").sanitized().stage,
            "decode"
        );

        // Even a hand-built NaN state is stored clamped by the registry.
        let registry = JobRegistry::new();
        let id = registry.submit(JobScope::Decode, CancelToken::new());
        registry
            .transition(id, JobState::Running { progress: f32::NAN })
            .expect("running with NaN");
        assert_eq!(state(&registry, id).progress(), Some(0.0));
    }

    #[test]
    fn terminal_transition_releases_reservations() {
        let registry = JobRegistry::new();
        let id = registry.submit(JobScope::Export, CancelToken::new());
        let cpu = ResourceReservation::new(7, 1024, 0, 64);
        let device = ResourceReservation::new(8, 0, 2048, 0);

        registry.reservation(id, cpu).expect("cpu reservation");
        registry.reservation(id, device).expect("device reservation");
        assert_eq!(cpu.host_bytes(), 1088);
        assert_eq!(
            registry.job(id).expect("record").reservations(),
            &[cpu, device]
        );

        registry
            .transition(id, JobState::running(0.1))
            .expect("running");
        registry
            .transition(id, JobState::Succeeded)
            .expect("succeeded");

        let record = registry.job(id).expect("record");
        assert!(record.reservations().is_empty());
        assert_eq!(
            record.released(),
            &[ReservationRelease { ticket: 7 }, ReservationRelease { ticket: 8 }]
        );
        assert!(matches!(
            registry.reservation(id, cpu),
            Err(TransitionError::TerminalJob(terminal)) if terminal == id
        ));
    }

    #[test]
    fn manual_release_returns_markers() {
        let registry = JobRegistry::new();
        let id = registry.submit(JobScope::Merge, CancelToken::new());
        let reservation = ResourceReservation::new(42, 10, 20, 30);
        registry
            .reservation(id, reservation)
            .expect("reservation");

        assert_eq!(
            registry.release_reservations(id),
            Ok(vec![ReservationRelease { ticket: 42 }])
        );
        assert!(registry
            .job(id)
            .expect("record")
            .reservations()
            .is_empty());
        assert_eq!(
            registry.job(id).expect("record").released(),
            &[ReservationRelease { ticket: 42 }]
        );
    }

    #[test]
    fn dispatch_evidence_reuses_receipt_and_fallback() {
        let registry = JobRegistry::new();
        let id = registry.submit(JobScope::Inference, CancelToken::new());
        let receipt = DispatchReceipt {
            operation: OperationId::new("detail.gpu"),
            backend: BackendKind::Cuda,
            kernel: Some("detail_kernel"),
            grid: Some((8, 8)),
            block: Some((16, 16)),
            host_to_device_bytes: 128,
            device_to_host_bytes: 256,
            completed: true,
        };
        let evidence = DispatchEvidence::from_receipt(receipt).with_fallback(Some(
            FallbackReason::BudgetExceeded { need: 4096, budget: 1024 },
        ));

        assert_eq!(evidence.backend, "cuda");
        assert_eq!(evidence.kernel, Some("detail_kernel"));
        assert_eq!(evidence.grid, Some((8, 8)));
        assert_eq!(evidence.block, Some((16, 16)));
        assert_eq!(evidence.operation, Some("detail.gpu"));
        assert_eq!(
            evidence.fallback,
            Some(FallbackReason::BudgetExceeded { need: 4096, budget: 1024 })
        );

        registry
            .record(id, evidence)
            .expect("record dispatch evidence");
        assert_eq!(
            registry.job(id).expect("record").evidence(),
            &[evidence]
        );
        // Evidence remains attachable after a terminal state.
        registry
            .transition(id, JobState::running(1.0))
            .expect("running");
        registry
            .transition(id, JobState::Succeeded)
            .expect("succeeded");
        registry.record(id, evidence).expect("late receipt");
        assert_eq!(registry.job(id).expect("record").evidence().len(), 2);
    }

    #[test]
    fn cancel_token_is_shared_across_clones() {
        let token = CancelToken::new();
        let clone = token.clone();
        assert!(clone.check().is_ok());
        assert!(!clone.is_cancelled());

        token.cancel();
        assert!(clone.is_cancelled());
        assert_eq!(clone.check(), Err(JobError::Cancelled));
        // Cancellation is one-way.
        clone.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn job_ids_are_unique_and_ordered_within_a_scope() {
        let registry = JobRegistry::new();
        let first = registry.submit(JobScope::Preview, CancelToken::new());
        let second = registry.submit(JobScope::Preview, CancelToken::new());
        let other_scope = registry.submit(JobScope::Decode, CancelToken::new());

        assert!(first < second);
        assert!(second.serial > first.serial);
        assert_ne!(first.serial, other_scope.serial);
        assert_eq!(first.to_string(), format!("preview:{}", first.serial));
        assert_eq!(JobScope::Other("custom").as_str(), "custom");
        assert_eq!(JobScope::Other("custom").to_string(), "custom");
    }

    #[test]
    fn job_error_display_and_conversion_keep_messages() {
        assert_eq!(JobError::Cancelled.to_string(), "job cancelled");
        assert!(JobError::Cancelled.is_cancelled());
        assert!(!JobError::DeviceLost.is_cancelled());

        let unsupported = JobError::Unsupported {
            what: "raw codec",
            reason: "the camera body is not supported".into(),
        };
        let text = unsupported.to_string();
        assert!(text.contains("raw codec"), "{text}");
        assert!(text.contains("camera body"), "{text}");

        let io: JobError = std::io::Error::other("disk full").into();
        assert_eq!(io, JobError::Io("disk full".into()));
        assert!(io.to_string().contains("disk full"));

        assert_eq!(
            JobError::ResourceExhausted { what: "vram" }.to_string(),
            "resource exhausted: vram"
        );
        assert_eq!(
            TransitionError::UnknownJob(JobId::new(JobScope::Preview, 1)).to_string(),
            "unknown job preview:1"
        );
    }
}
