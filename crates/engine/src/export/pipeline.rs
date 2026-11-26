//! Bounded three-stage export pipeline with backpressure.
//!
//! The batch entry (`AppState::process` -> `ImageEngine::start_export` per
//! image and recipe) used to run one full `decode -> render -> encode` chain per
//! job thread and let the [`crate::file_engine`] export scheduler decide how many
//! chains may overlap. This module keeps the scheduler as the admission and
//! memory authority and splits the chain itself into three stages:
//!
//! 1. **Source acquire** (`stage1`, runs on the submitting job thread): turns a
//!    [`crate::export::stages::PreparedExport`] into a render-ready source. A
//!    cached / native source passes straight through; the deferred
//!    full-resolution decode is the CPU work of this stage.
//! 2. **Render** (`stage2`, one GPU lane): the real develop pipeline
//!    (`mock::process::render` / `render16`) — the same calls the serial export
//!    made, so pixels cannot diverge.
//! 3. **Encode + publish** (`stage3`, CPU lanes): `recipe::write_export` /
//!    `write_export16` into a staging folder plus the atomic publish under the
//!    engine's publish gate.
//!
//! Stages hand frames over through [`BoundedQueue`]s capped by both item count
//! and bytes. A full queue blocks its producer, which is what bounds peak memory
//! while a slow stage drains; an empty queue always admits, so a frame larger
//! than the whole queue budget still makes progress instead of deadlocking.
//!
//! Deadlock freedom: only lane threads pop from a queue, a lane never waits on a
//! job thread, and every `stage1` producer eventually finishes because
//! `stage2`/`stage3` always drain (publishing is a short critical section under
//! the publish gate). A blocked push is therefore always temporary.
//!
//! [`PipelineMode::Serial`] runs the very same three stage functions inline on
//! the submitting thread. It exists for A/B numbers and for the byte-identity
//! regression: identical bytes follow from identical functions, not from a
//! second implementation.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

use crate::job_lifecycle::CancelToken;

/// Poll interval for every blocking wait; cancellation and shutdown are observed
/// within this bound instead of relying on a wake-up.
const LANE_POLL: Duration = Duration::from_millis(5);

/// Default queue depth per stage. The export scheduler admits a small number of
/// jobs at a time, so a small depth is all the overlap the pipeline can use; a
/// larger depth only delays backpressure.
pub const DEFAULT_QUEUE_ITEMS: usize = 4;

/// Default byte cap per stage. Two 24 MP 16-bit frames plus their encoded
/// payloads fit; a third does not, which keeps a stalled stage from banking a
/// batch worth of frame buffers.
pub const DEFAULT_QUEUE_BYTES: usize = 1024 * 1024 * 1024;

/// Marker error text for a stage that stopped because the job was cancelled.
/// Callers map it with their own cancel token, exactly like the serial worker
/// did; the string only feeds logs and the failure record.
pub const CANCELLED: &str = "cancelled";

/// Which execution shape a run uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineMode {
    /// Stages run one after another on the submitting thread.
    Serial,
    /// Stages run on their lanes, connected by the bounded queues.
    Pipelined,
}

impl PipelineMode {
    fn from_env_value(raw: &str) -> Self {
        match raw.trim().to_ascii_lowercase().as_str() {
            "0" | "serial" | "off" | "inline" => Self::Serial,
            _ => Self::Pipelined,
        }
    }
}

/// Lane and queue limits of a pipeline instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PipelineLimits {
    /// Render lanes. One by default: `mock::process::render` already
    /// parallelises a frame with rayon, and two concurrent GPU develops only
    /// contend for the same device.
    pub render_lanes: usize,
    /// Encode lanes. Encoding is single-threaded per frame, so a second lane
    /// keeps the render lane fed while a large JPEG/PNG is written.
    pub encode_lanes: usize,
    pub queue_items: usize,
    pub queue_bytes: usize,
}

impl Default for PipelineLimits {
    fn default() -> Self {
        Self {
            render_lanes: default_render_lanes(),
            encode_lanes: default_encode_lanes(),
            queue_items: DEFAULT_QUEUE_ITEMS,
            queue_bytes: DEFAULT_QUEUE_BYTES,
        }
    }
}

/// `AURORA_EXPORT_RENDER_LANES` overrides the render lane count. One is the
/// tested default (a single GPU device and a rayon-parallel render); more lanes
/// only help an A/B test on a machine whose develop path leaves cores idle.
fn default_render_lanes() -> usize {
    match std::env::var("AURORA_EXPORT_RENDER_LANES") {
        Ok(raw) => raw.trim().parse::<usize>().map(|n| n.clamp(1, 4)).unwrap_or(1),
        Err(_) => 1,
    }
}

/// `AURORA_EXPORT_ENCODE_LANES` overrides the encode lane count; the default
/// keeps two frames encoding at once.
fn default_encode_lanes() -> usize {
    match std::env::var("AURORA_EXPORT_ENCODE_LANES") {
        Ok(raw) => raw.trim().parse::<usize>().map(|n| n.clamp(1, 8)).unwrap_or(2),
        Err(_) => 2,
    }
}

const MODE_UNSET: usize = usize::MAX;
const MODE_SERIAL: usize = 0;
const MODE_PIPELINED: usize = 1;

/// Process-wide mode override used by the byte-identity regression and by
/// `AURORA_EXPORT_PIPELINE=serial`.
static MODE: AtomicUsize = AtomicUsize::new(MODE_UNSET);
static MODE_LOCK: Mutex<()> = Mutex::new(());

/// The mode export runs with. `AURORA_EXPORT_PIPELINE=0|serial` selects the
/// inline chain; anything else (or an unset variable) keeps the pipeline.
pub fn mode() -> PipelineMode {
    match MODE.load(Ordering::Acquire) {
        MODE_SERIAL => PipelineMode::Serial,
        MODE_PIPELINED => PipelineMode::Pipelined,
        _ => {
            let resolved = std::env::var("AURORA_EXPORT_PIPELINE")
                .map(|raw| PipelineMode::from_env_value(&raw))
                .unwrap_or(PipelineMode::Pipelined);
            let encoded = match resolved {
                PipelineMode::Serial => MODE_SERIAL,
                PipelineMode::Pipelined => MODE_PIPELINED,
            };
            // Another thread may have resolved the same value first.
            let _ = MODE.compare_exchange(MODE_UNSET, encoded, Ordering::AcqRel, Ordering::Acquire);
            resolved
        }
    }
}

/// Pins [`mode`] for the lifetime of the guard and serialises tests that need a
/// specific mode. Holding it across an export keeps other mode-pinning tests
/// out; an unpinned concurrent export still produces correct bytes either way.
pub fn pin_mode(pinned: PipelineMode) -> ModePin {
    let guard = MODE_LOCK.lock();
    let encoded = match pinned {
        PipelineMode::Serial => MODE_SERIAL,
        PipelineMode::Pipelined => MODE_PIPELINED,
    };
    let previous = MODE.swap(encoded, Ordering::AcqRel);
    ModePin { previous, _guard: guard }
}

/// Restores the previous mode when dropped.
pub struct ModePin {
    previous: usize,
    _guard: parking_lot::MutexGuard<'static, ()>,
}

impl Drop for ModePin {
    fn drop(&mut self) {
        MODE.store(self.previous, Ordering::Release);
    }
}

/// Bytes an item occupies while it sits in a queue.
pub trait Queued {
    fn charge_bytes(&self) -> usize;
}

/// Result of a bounded push. A rejected push hands the item back so the caller
/// can complete it (the item owns its completion slot).
pub enum PushOutcome<T> {
    Pushed,
    /// The item's cancel token was set; nothing was queued.
    Cancelled(T),
    /// The queue was closed (pipeline shutdown); nothing was queued.
    Closed(T),
}

struct QueueState<T> {
    items: VecDeque<T>,
    bytes: usize,
    closed: bool,
    peak_bytes: usize,
    blocked_pushes: u64,
    /// Wall-clock the producers spent waiting for room, so a batch report can
    /// tell "the queue bound the submit threads" from "the queue never filled".
    blocked_nanos: u64,
}

/// Bounded, charge-aware, cancel-aware queue between two stages.
///
/// `push` blocks while the queue is full instead of dropping or growing, which
/// is the backpressure that keeps a batch from banking decoded frames. An empty
/// queue always admits: an item bigger than `max_bytes` would otherwise wait for
/// a consumer that can never run.
pub struct BoundedQueue<T> {
    state: Mutex<QueueState<T>>,
    ready: Condvar,
    max_items: usize,
    max_bytes: usize,
}

impl<T: Queued> BoundedQueue<T> {
    pub fn new(max_items: usize, max_bytes: usize) -> Self {
        Self {
            state: Mutex::new(QueueState {
                items: VecDeque::new(),
                bytes: 0,
                closed: false,
                peak_bytes: 0,
                blocked_pushes: 0,
                blocked_nanos: 0,
            }),
            ready: Condvar::new(),
            max_items: max_items.max(1),
            max_bytes: max_bytes.max(1),
        }
    }

    /// Pushes `item`, waiting while the queue is full.
    pub fn push(&self, item: T, cancel: &CancelToken) -> PushOutcome<T> {
        if cancel.is_cancelled() {
            return PushOutcome::Cancelled(item);
        }
        let charge = item.charge_bytes();
        let mut state = self.state.lock();
        let mut counted_block = false;
        let mut waited_since: Option<std::time::Instant> = None;
        loop {
            if state.closed {
                return PushOutcome::Closed(item);
            }
            if cancel.is_cancelled() {
                return PushOutcome::Cancelled(item);
            }
            let fits_items = state.items.len() < self.max_items;
            let fits_bytes = state.bytes.saturating_add(charge) <= self.max_bytes;
            if state.items.is_empty() || (fits_items && fits_bytes) {
                if let Some(started) = waited_since {
                    let waited = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                    state.blocked_nanos = state.blocked_nanos.saturating_add(waited);
                }
                state.bytes = state.bytes.saturating_add(charge);
                state.peak_bytes = state.peak_bytes.max(state.bytes);
                state.items.push_back(item);
                self.ready.notify_all();
                return PushOutcome::Pushed;
            }
            if !counted_block {
                state.blocked_pushes = state.blocked_pushes.saturating_add(1);
                counted_block = true;
                waited_since = Some(std::time::Instant::now());
            }
            self.ready.wait_for(&mut state, LANE_POLL);
        }
    }

    /// Pops the oldest item, waiting while the queue is empty. Returns `None`
    /// once the queue is closed and drained.
    pub fn pop(&self) -> Option<T> {
        let mut state = self.state.lock();
        loop {
            if let Some(item) = state.items.pop_front() {
                state.bytes = state.bytes.saturating_sub(item.charge_bytes());
                self.ready.notify_all();
                return Some(item);
            }
            if state.closed {
                return None;
            }
            self.ready.wait_for(&mut state, LANE_POLL);
        }
    }

    /// Stops accepting pushes and wakes every waiter. Consumers drain what is
    /// already queued before they see `None`.
    pub fn close(&self) {
        self.state.lock().closed = true;
        self.ready.notify_all();
    }

    pub fn len(&self) -> usize {
        self.state.lock().items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bytes(&self) -> usize {
        self.state.lock().bytes
    }

    /// Highest byte charge the queue ever held, for the performance log.
    pub fn peak_bytes(&self) -> usize {
        self.state.lock().peak_bytes
    }

    /// How many pushes had to wait for room; zero means this queue never
    /// applied backpressure.
    pub fn blocked_pushes(&self) -> u64 {
        self.state.lock().blocked_pushes
    }

    /// Total wall-clock producers spent blocked on this queue.
    pub fn blocked_nanos(&self) -> u64 {
        self.state.lock().blocked_nanos
    }
}

/// One item travelling through the pipeline: the frame, its cancel token and
/// the completion the submitting thread waits on.
struct StageItem<T> {
    job: u64,
    value: T,
    cancel: CancelToken,
    done: Arc<Completion>,
}

impl<T: Queued> Queued for StageItem<T> {
    fn charge_bytes(&self) -> usize {
        self.value.charge_bytes()
    }
}

/// Hand-off slot for an item's final result.
struct Completion {
    slot: Mutex<Option<Result<Published, String>>>,
    ready: Condvar,
}

/// The published result of a pipeline run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Published {
    /// Final path, after the atomic rename into the destination.
    pub path: PathBuf,
}

impl Completion {
    fn new() -> Arc<Self> {
        Arc::new(Self { slot: Mutex::new(None), ready: Condvar::new() })
    }

    fn complete(&self, result: Result<Published, String>) {
        *self.slot.lock() = Some(result);
        self.ready.notify_all();
    }

    fn wait(&self) -> Result<Published, String> {
        let mut slot = self.slot.lock();
        loop {
            if let Some(result) = slot.take() {
                return result;
            }
            self.ready.wait_for(&mut slot, LANE_POLL);
        }
    }
}

/// Counters a batch driver or a benchmark can read without touching internals.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineStats {
    pub stage1_items: u64,
    pub stage2_items: u64,
    pub stage3_items: u64,
    pub cancelled_items: u64,
    pub failed_items: u64,
    pub ready_peak_bytes: usize,
    pub rendered_peak_bytes: usize,
    pub ready_blocked_pushes: u64,
    pub rendered_blocked_pushes: u64,
    /// Nanoseconds the submitting threads spent waiting for room in the
    /// ready queue (stage 1 -> stage 2). Zero proves the item/byte cap did not
    /// constrain the submitter on this batch.
    pub ready_blocked_nanos: u64,
    /// Nanoseconds the render lane spent waiting for room in the rendered queue.
    pub rendered_blocked_nanos: u64,
}

type Stage1Fn<P, R> = Arc<dyn Fn(P) -> Result<R, String> + Send + Sync>;
type Stage2Fn<R, F> = Arc<dyn Fn(R, &CancelToken) -> Result<F, String> + Send + Sync>;
type Stage3Fn<F> = Arc<dyn Fn(F, &CancelToken) -> Result<Published, String> + Send + Sync>;

/// A three-stage pipeline over `P -> R -> F -> Published`.
///
/// The stage functions are injected so unit tests can drive the same executor
/// with synthetic frames; the export path installs
/// [`crate::export::stages::acquire_source`],
/// [`crate::export::stages::render_frame`] and
/// [`crate::export::stages::encode_and_publish`].
pub struct Pipeline<P, R, F> {
    limits: PipelineLimits,
    stage1: Stage1Fn<P, R>,
    stage2: Stage2Fn<R, F>,
    stage3: Stage3Fn<F>,
    ready: Arc<BoundedQueue<StageItem<R>>>,
    rendered: Arc<BoundedQueue<StageItem<F>>>,
    live_lanes: Arc<LaneCounter>,
    counters: Arc<Counters>,
}

#[derive(Default)]
struct Counters {
    stage1_items: AtomicU64,
    stage2_items: AtomicU64,
    stage3_items: AtomicU64,
    cancelled_items: AtomicU64,
    failed_items: AtomicU64,
}

/// Tracks render lanes so `shutdown` can close the encode queue only after they
/// stopped.
#[derive(Default)]
struct LaneCounter {
    live: Mutex<usize>,
    done: Condvar,
}

impl LaneCounter {
    fn join(&self) {
        *self.live.lock() += 1;
    }

    fn leave(&self) {
        let mut live = self.live.lock();
        *live = live.saturating_sub(1);
        self.done.notify_all();
    }

    fn wait_for_none(&self) {
        let mut live = self.live.lock();
        while *live > 0 {
            self.done.wait(&mut live);
        }
    }
}

impl<P, R, F> Pipeline<P, R, F>
where
    P: Send + 'static,
    R: Queued + Send + 'static,
    F: Queued + Send + 'static,
{
    /// Builds a pipeline and starts its lanes.
    pub fn start(
        limits: PipelineLimits,
        stage1: Stage1Fn<P, R>,
        stage2: Stage2Fn<R, F>,
        stage3: Stage3Fn<F>,
    ) -> Self {
        let ready = Arc::new(BoundedQueue::new(limits.queue_items, limits.queue_bytes));
        let rendered = Arc::new(BoundedQueue::new(limits.queue_items, limits.queue_bytes));
        let live_lanes = Arc::new(LaneCounter::default());
        let pipeline = Self {
            limits,
            stage1,
            stage2,
            stage3,
            ready,
            rendered,
            live_lanes,
            counters: Arc::new(Counters::default()),
        };
        pipeline.spawn_lanes();
        pipeline
    }

    fn spawn_lanes(&self) {
        for _ in 0..self.limits.render_lanes.max(1) {
            let ready = Arc::clone(&self.ready);
            let rendered = Arc::clone(&self.rendered);
            let live = Arc::clone(&self.live_lanes);
            let counters = Arc::clone(&self.counters);
            let stage2 = Arc::clone(&self.stage2);
            live.join();
            std::thread::Builder::new()
                .name("aurora-export-render".into())
                .spawn(move || {
                    render_lane(ready, rendered, stage2, counters);
                    live.leave();
                })
                .expect("spawn export render lane");
        }
        for _ in 0..self.limits.encode_lanes.max(1) {
            let rendered = Arc::clone(&self.rendered);
            let counters = Arc::clone(&self.counters);
            let stage3 = Arc::clone(&self.stage3);
            std::thread::Builder::new()
                .name("aurora-export-encode".into())
                .spawn(move || encode_lane(rendered, stage3, counters))
                .expect("spawn export encode lane");
        }
    }

    pub fn limits(&self) -> PipelineLimits {
        self.limits
    }

    /// Runs one item end to end with the current [`mode`].
    pub fn run(&self, prepared: P, job: u64, cancel: &CancelToken) -> Result<Published, String> {
        match mode() {
            PipelineMode::Serial => self.run_serial(prepared, cancel),
            PipelineMode::Pipelined => self.run_pipelined(prepared, job, cancel),
        }
    }

    /// Inline chain: the reference shape the pipelined run must match byte for
    /// byte.
    pub fn run_serial(&self, prepared: P, cancel: &CancelToken) -> Result<Published, String> {
        if cancel.is_cancelled() {
            return Err(CANCELLED.to_string());
        }
        let ready = (self.stage1)(prepared)?;
        self.counters.stage1_items.fetch_add(1, Ordering::Relaxed);
        if cancel.is_cancelled() {
            return Err(CANCELLED.to_string());
        }
        let frame = (self.stage2)(ready, cancel)?;
        self.counters.stage2_items.fetch_add(1, Ordering::Relaxed);
        if cancel.is_cancelled() {
            return Err(CANCELLED.to_string());
        }
        let published = (self.stage3)(frame, cancel)?;
        self.counters.stage3_items.fetch_add(1, Ordering::Relaxed);
        Ok(published)
    }

    fn run_pipelined(&self, prepared: P, job: u64, cancel: &CancelToken) -> Result<Published, String> {
        if cancel.is_cancelled() {
            return Err(CANCELLED.to_string());
        }
        let ready = match (self.stage1)(prepared) {
            Ok(ready) => ready,
            Err(err) => {
                self.counters.failed_items.fetch_add(1, Ordering::Relaxed);
                return Err(err);
            }
        };
        self.counters.stage1_items.fetch_add(1, Ordering::Relaxed);
        if cancel.is_cancelled() {
            self.counters.cancelled_items.fetch_add(1, Ordering::Relaxed);
            return Err(CANCELLED.to_string());
        }
        let done = Completion::new();
        let item = StageItem { job, value: ready, cancel: cancel.clone(), done: Arc::clone(&done) };
        match self.ready.push(item, cancel) {
            PushOutcome::Pushed => done.wait(),
            PushOutcome::Cancelled(_) => {
                self.counters.cancelled_items.fetch_add(1, Ordering::Relaxed);
                Err(CANCELLED.to_string())
            }
            PushOutcome::Closed(_) => Err("export pipeline is closed".to_string()),
        }
    }

    /// Stops the lanes: no new items, let the render lanes finish and exit, then
    /// close the encode queue so the encode lanes drain and exit too.
    pub fn shutdown(&self) {
        self.ready.close();
        self.live_lanes.wait_for_none();
        self.rendered.close();
    }

    pub fn stats(&self) -> PipelineStats {
        PipelineStats {
            stage1_items: self.counters.stage1_items.load(Ordering::Relaxed),
            stage2_items: self.counters.stage2_items.load(Ordering::Relaxed),
            stage3_items: self.counters.stage3_items.load(Ordering::Relaxed),
            cancelled_items: self.counters.cancelled_items.load(Ordering::Relaxed),
            failed_items: self.counters.failed_items.load(Ordering::Relaxed),
            ready_peak_bytes: self.ready.peak_bytes(),
            rendered_peak_bytes: self.rendered.peak_bytes(),
            ready_blocked_pushes: self.ready.blocked_pushes(),
            rendered_blocked_pushes: self.rendered.blocked_pushes(),
            ready_blocked_nanos: self.ready.blocked_nanos(),
            rendered_blocked_nanos: self.rendered.blocked_nanos(),
        }
    }

}

fn render_lane<R, F>(
    ready: Arc<BoundedQueue<StageItem<R>>>,
    rendered: Arc<BoundedQueue<StageItem<F>>>,
    stage2: Stage2Fn<R, F>,
    counters: Arc<Counters>,
) where
    R: Queued + Send + 'static,
    F: Queued + Send + 'static,
{
    while let Some(item) = ready.pop() {
        let StageItem { job, value, cancel, done } = item;
        if cancel.is_cancelled() {
            counters.cancelled_items.fetch_add(1, Ordering::Relaxed);
            done.complete(Err(CANCELLED.to_string()));
            continue;
        }
        log::trace!("export job {job} entered the render lane");
        let frame = match stage2(value, &cancel) {
            Ok(frame) => frame,
            Err(err) => {
                counters.failed_items.fetch_add(1, Ordering::Relaxed);
                done.complete(Err(err));
                continue;
            }
        };
        counters.stage2_items.fetch_add(1, Ordering::Relaxed);
        let next = StageItem { job, value: frame, cancel, done };
        let token = next.cancel.clone();
        match rendered.push(next, &token) {
            PushOutcome::Pushed => {}
            PushOutcome::Cancelled(item) => {
                counters.cancelled_items.fetch_add(1, Ordering::Relaxed);
                item.done.complete(Err(CANCELLED.to_string()));
            }
            // The pipeline is shutting down; the item is dropped with its frame
            // and the waiter is released instead of hanging on a closed queue.
            PushOutcome::Closed(item) => {
                counters.cancelled_items.fetch_add(1, Ordering::Relaxed);
                item.done.complete(Err("export pipeline is closed".to_string()));
                break;
            }
        }
    }
}

fn encode_lane<F>(rendered: Arc<BoundedQueue<StageItem<F>>>, stage3: Stage3Fn<F>, counters: Arc<Counters>)
where
    F: Queued + Send + 'static,
{
    while let Some(item) = rendered.pop() {
        let StageItem { value, cancel, done, .. } = item;
        if cancel.is_cancelled() {
            counters.cancelled_items.fetch_add(1, Ordering::Relaxed);
            done.complete(Err(CANCELLED.to_string()));
            continue;
        }
        match stage3(value, &cancel) {
            Ok(published) => {
                counters.stage3_items.fetch_add(1, Ordering::Relaxed);
                done.complete(Ok(published));
            }
            Err(err) => {
                counters.failed_items.fetch_add(1, Ordering::Relaxed);
                done.complete(Err(err));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    impl Queued for Vec<u8> {
        fn charge_bytes(&self) -> usize {
            self.len()
        }
    }

    /// Records the highest concurrency a stage ever saw.
    #[derive(Default)]
    struct Peak {
        current: AtomicUsize,
        peak: AtomicUsize,
    }

    impl Peak {
        fn enter(&self) {
            let now = self.current.fetch_add(1, Ordering::SeqCst) + 1;
            let mut seen = self.peak.load(Ordering::SeqCst);
            while now > seen {
                match self.peak.compare_exchange(seen, now, Ordering::SeqCst, Ordering::SeqCst) {
                    Ok(_) => break,
                    Err(actual) => seen = actual,
                }
            }
        }

        fn leave(&self) {
            self.current.fetch_sub(1, Ordering::SeqCst);
        }

        fn peak(&self) -> usize {
            self.peak.load(Ordering::SeqCst)
        }
    }

    /// `stage2` payload: the frame plus the item id so results stay traceable.
    struct Frame {
        id: u32,
        bytes: Vec<u8>,
    }

    impl Queued for Frame {
        fn charge_bytes(&self) -> usize {
            self.bytes.len()
        }
    }

    fn stage1(raw: u32) -> Result<Vec<u8>, String> {
        Ok(vec![raw as u8; 8])
    }

    fn stage2(frame: Vec<u8>, _cancel: &CancelToken) -> Result<Frame, String> {
        let mut bytes = frame;
        for byte in &mut bytes {
            *byte = byte.wrapping_add(1);
        }
        Ok(Frame { id: bytes[0] as u32, bytes })
    }

    fn stage3(frame: Frame, _cancel: &CancelToken) -> Result<Published, String> {
        let checksum: u64 = frame.bytes.iter().map(|byte| u64::from(*byte)).sum();
        Ok(Published { path: PathBuf::from(format!("{}:{checksum}", frame.id)) })
    }

    fn pipeline(limits: PipelineLimits) -> Arc<Pipeline<u32, Vec<u8>, Frame>> {
        Arc::new(Pipeline::start(limits, Arc::new(stage1), Arc::new(stage2), Arc::new(stage3)))
    }

    fn quiet_limits() -> PipelineLimits {
        PipelineLimits { render_lanes: 1, encode_lanes: 2, queue_items: 4, queue_bytes: 4096 }
    }

    /// Serial and pipelined runs go through the same stage functions, so the
    /// per-item results must be identical, not merely equivalent.
    #[test]
    fn pipelined_and_serial_runs_return_identical_results() {
        let pipeline = pipeline(quiet_limits());
        let serial: Vec<_> = {
            let _pin = pin_mode(PipelineMode::Serial);
            (1..=6u32).map(|id| pipeline.run(id, u64::from(id), &CancelToken::new())).collect()
        };
        let pipelined: Vec<_> = {
            let _pin = pin_mode(PipelineMode::Pipelined);
            (1..=6u32)
                .map(|id| {
                    let pipeline = Arc::clone(&pipeline);
                    std::thread::spawn({
                        let cancel = CancelToken::new();
                        move || pipeline.run(id, u64::from(id), &cancel)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect()
        };
        let mut serial = serial;
        let mut pipelined = pipelined;
        serial.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        pipelined.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        assert_eq!(serial, pipelined);
        assert!(serial.iter().all(|result| result.is_ok()));
        let stats = pipeline.stats();
        assert_eq!(stats.failed_items, 0);
        assert_eq!(stats.cancelled_items, 0);
        pipeline.shutdown();
    }

    /// The cap only throttles a producer that runs ahead of the depth. In a
    /// batch every image owns one submitting job thread, so the real shape is
    /// several concurrent `run` calls: with more of them than queue slots the
    /// push blocks and the wait is counted. A batch whose admission keeps fewer
    /// jobs in flight than the queue depth never sees backpressure — there the
    /// scheduler, not the queue, is what limits the submit threads.
    #[test]
    fn backpressure_counts_only_when_submitters_outnumber_the_depth() {
        let pipeline = Arc::new(Pipeline::start(
            PipelineLimits { render_lanes: 1, encode_lanes: 1, queue_items: 2, queue_bytes: 1 << 20 },
            Arc::new(stage1),
            Arc::new(|frame: Vec<u8>, _cancel: &CancelToken| -> Result<Frame, String> {
                std::thread::sleep(Duration::from_millis(5));
                Ok(Frame { id: frame[0] as u32, bytes: frame })
            }),
            Arc::new(|frame: Frame, _cancel: &CancelToken| -> Result<Published, String> {
                std::thread::sleep(Duration::from_millis(10));
                Ok(Published { path: PathBuf::from(format!("{}", frame.id)) })
            }),
        ));
        let _pin = pin_mode(PipelineMode::Pipelined);
        let handles: Vec<_> = (1..=8u32)
            .map(|id| {
                let pipeline = Arc::clone(&pipeline);
                std::thread::spawn(move || pipeline.run(id, u64::from(id), &CancelToken::new()))
            })
            .collect();
        for handle in handles {
            handle.join().unwrap().unwrap();
        }
        let stats = pipeline.stats();
        assert!(stats.ready_blocked_pushes > 0, "8 submitters through 2 slots must block: {stats:?}");
        assert!(stats.ready_blocked_nanos > 0, "a blocked push must record its wait: {stats:?}");
        assert_eq!(stats.stage3_items, 8);
        pipeline.shutdown();
    }

    /// One GPU lane: two renders never run at the same time, while the encode
    /// lane does overlap it (that cross-stage overlap is the point of the
    /// pipeline, not just concurrency inside one stage).
    #[test]
    fn the_render_lane_is_exclusive_and_the_encode_lane_overlaps_it() {
        let render_peak = Arc::new(Peak::default());
        let render_active = Arc::new(AtomicUsize::new(0));
        let encode_active = Arc::new(AtomicUsize::new(0));
        let overlap = Arc::new(AtomicUsize::new(0));
        let stage2_peak = Arc::clone(&render_peak);
        let stage2_active = Arc::clone(&render_active);
        let stage2_encode_active = Arc::clone(&encode_active);
        let stage2_overlap = Arc::clone(&overlap);
        let stage3_active = Arc::clone(&encode_active);
        let stage3_render_active = Arc::clone(&render_active);
        let stage3_overlap = Arc::clone(&overlap);
        let pipeline = Arc::new(Pipeline::start(
            quiet_limits(),
            Arc::new(|raw: u32| -> Result<Vec<u8>, String> { Ok(vec![raw as u8; 8]) }),
            Arc::new(move |frame: Vec<u8>, _cancel: &CancelToken| -> Result<Frame, String> {
                stage2_peak.enter();
                stage2_active.fetch_add(1, Ordering::SeqCst);
                if stage2_encode_active.load(Ordering::SeqCst) > 0 {
                    stage2_overlap.fetch_add(1, Ordering::SeqCst);
                }
                std::thread::sleep(Duration::from_millis(25));
                stage2_active.fetch_sub(1, Ordering::SeqCst);
                stage2_peak.leave();
                Ok(Frame { id: frame[0] as u32, bytes: frame })
            }),
            Arc::new(move |frame: Frame, _cancel: &CancelToken| -> Result<Published, String> {
                stage3_active.fetch_add(1, Ordering::SeqCst);
                if stage3_render_active.load(Ordering::SeqCst) > 0 {
                    stage3_overlap.fetch_add(1, Ordering::SeqCst);
                }
                std::thread::sleep(Duration::from_millis(25));
                stage3_active.fetch_sub(1, Ordering::SeqCst);
                Ok(Published { path: PathBuf::from(format!("{}", frame.id)) })
            }),
        ));
        let _pin = pin_mode(PipelineMode::Pipelined);
        let handles: Vec<_> = (1..=4u32)
            .map(|id| {
                let pipeline = Arc::clone(&pipeline);
                std::thread::spawn(move || pipeline.run(id, u64::from(id), &CancelToken::new()))
            })
            .collect();
        for handle in handles {
            handle.join().unwrap().unwrap();
        }
        assert_eq!(render_peak.peak(), 1, "the GPU lane must never run two renders at once");
        assert!(
            overlap.load(Ordering::SeqCst) > 0,
            "an encode must overlap a render; without it the pipeline buys nothing"
        );
        pipeline.shutdown();
    }

    /// A failing item reports its own error and the queue keeps draining.
    #[test]
    fn a_failed_item_does_not_block_the_rest() {
        let pipeline = Arc::new(Pipeline::start(
            quiet_limits(),
            Arc::new(stage1),
            Arc::new(|frame: Vec<u8>, _cancel: &CancelToken| -> Result<Frame, String> {
                if frame[0] == 3 {
                    return Err("render exploded".to_string());
                }
                Ok(Frame { id: frame[0] as u32, bytes: frame })
            }),
            Arc::new(stage3),
        ));
        let _pin = pin_mode(PipelineMode::Pipelined);
        let results: Vec<_> = (1..=5u32)
            .map(|id| pipeline.run(id, u64::from(id), &CancelToken::new()))
            .collect();
        assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 4);
        let failures: Vec<_> = results.iter().filter_map(|result| result.as_ref().err()).collect();
        assert_eq!(failures, vec![&"render exploded".to_string()]);
        assert_eq!(pipeline.stats().failed_items, 1);
        pipeline.shutdown();
    }

    /// Backpressure: a full queue blocks its producer until the consumer drains,
    /// and a single oversized item is still admitted into an empty queue so a
    /// frame larger than the whole budget cannot deadlock the batch.
    #[test]
    fn a_full_queue_blocks_pushes_and_an_oversized_item_still_gets_through() {
        let queue: Arc<BoundedQueue<Vec<u8>>> = Arc::new(BoundedQueue::new(2, 16));
        let cancel = CancelToken::new();
        assert!(matches!(queue.push(vec![0u8; 8], &cancel), PushOutcome::Pushed));
        assert!(matches!(queue.push(vec![0u8; 8], &cancel), PushOutcome::Pushed));
        // Third push cannot fit either the item cap or the byte cap.
        let blocked = {
            let queue = Arc::clone(&queue);
            let cancel = cancel.clone();
            std::thread::spawn(move || matches!(queue.push(vec![0u8; 8], &cancel), PushOutcome::Pushed))
        };
        std::thread::sleep(Duration::from_millis(40));
        assert!(!blocked.is_finished(), "a full queue must block the producer");
        assert_eq!(queue.len(), 2);
        assert!(queue.blocked_pushes() >= 1);
        assert_eq!(queue.pop().unwrap().len(), 8);
        assert!(blocked.join().unwrap(), "the blocked push completes once room appears");

        // An empty queue always admits, even when the item exceeds max_bytes.
        let tight: BoundedQueue<Vec<u8>> = BoundedQueue::new(1, 4);
        assert!(matches!(tight.push(vec![0u8; 4096], &cancel), PushOutcome::Pushed));
        assert_eq!(tight.len(), 1);
    }

    /// Cancel wakes a blocked producer instead of leaving it stuck.
    #[test]
    fn cancel_wakes_a_blocked_push() {
        let queue: BoundedQueue<Vec<u8>> = BoundedQueue::new(1, 8);
        let cancel = CancelToken::new();
        assert!(matches!(queue.push(vec![0u8; 8], &cancel), PushOutcome::Pushed));
        let waiter = cancel.clone();
        let handle = std::thread::spawn(move || {
            let outcome = queue.push(vec![1u8; 8], &waiter);
            matches!(outcome, PushOutcome::Cancelled(_) | PushOutcome::Closed(_))
        });
        std::thread::sleep(Duration::from_millis(30));
        cancel.cancel();
        assert!(handle.join().unwrap(), "a cancelled push must return instead of waiting forever");
    }

    /// A cancelled item is dropped by the lanes: it never reaches the encode
    /// lane and reports a cancellation instead of a failure.
    #[test]
    fn a_cancelled_item_never_reaches_the_encode_lane() {
        let encoded = Arc::new(AtomicUsize::new(0));
        let encode_count = Arc::clone(&encoded);
        let pipeline = Arc::new(Pipeline::start(
            quiet_limits(),
            Arc::new(stage1),
            Arc::new(|frame: Vec<u8>, _cancel: &CancelToken| -> Result<Frame, String> {
                std::thread::sleep(Duration::from_millis(5));
                Ok(Frame { id: frame[0] as u32, bytes: frame })
            }),
            Arc::new(move |frame: Frame, _cancel: &CancelToken| -> Result<Published, String> {
                encode_count.fetch_add(1, Ordering::SeqCst);
                Ok(Published { path: PathBuf::from(format!("{}", frame.id)) })
            }),
        ));
        let _pin = pin_mode(PipelineMode::Pipelined);
        let cancel = CancelToken::new();
        cancel.cancel();
        let result = pipeline.run(7, 7, &cancel);
        assert_eq!(result.unwrap_err(), CANCELLED);
        assert_eq!(encoded.load(Ordering::SeqCst), 0, "a cancelled item must not be encoded");
        pipeline.shutdown();
    }
}
