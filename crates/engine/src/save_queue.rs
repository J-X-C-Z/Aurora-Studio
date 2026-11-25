//! Bounded, single-writer persistence queue.
//!
//! The synchronous save path already reports per-document results (see the
//! `save_report` module). This module is the next step: a bounded queue with
//! exactly one writer thread, per-document ordering, coalescing of newer edits,
//! a flush that reports what actually reached the writer, and an explicit
//! "queued" state between "memory accepted" and "written". It intentionally
//! does not use an async runtime and never holds the queue lock across file I/O.
//!
//! # Contract
//!
//! * The caller supplies a [`DocumentRevision`] per path. Revisions must be
//!   strictly monotonic **per path**; the queue does not generate them. A
//!   submission whose revision is less than or equal to the newest revision
//!   already accepted for that path is rejected as [`SubmitOutcome::Superseded`].
//!   This is what guarantees an old edit can never overwrite newer bytes.
//! * Coalescing is per path only. A newer revision for a path replaces the
//!   pending item for the *same* path; items for different paths are never
//!   merged or dropped.
//! * [`SaveQueue::submit`] never blocks for I/O and never waits for the writer.
//!   When `capacity` outstanding items (queued plus the single in-flight item)
//!   are already present it returns [`SubmitOutcome::Full`]. A newer revision
//!   for a path that is already outstanding is still accepted: the older
//!   pending value is replaced, and an already-in-flight write is allowed to
//!   finish before the newer value is written. Rejecting it would let stale
//!   bytes win.
//! * [`SaveQueue::flush`] is a bounded wait, not a barrier command. It waits
//!   until the queue is drained (including the index commit that follows a
//!   batch) or the deadline passes, then reports the newest outcome for every
//!   path with retained outcome detail. Queued or in-flight paths at the
//!   deadline are reported [`PathState::NotWritten`] and stay in the queue.
//!   Calling `flush`
//!   while the writer blocks in `write` returns at the deadline; the queue lock
//!   is not held by the writer during I/O.
//! * [`SaveQueue::close`] and `Drop` stop accepting work and let the writer
//!   drain the items it can within the deadline. Items still queued when the
//!   deadline expires are removed and recorded as [`PathState::NotWritten`] so
//!   they are not dropped silently. A write that is already inside
//!   [`SaveWriter::write`] cannot be cancelled; it may complete after `close`
//!   returns, and its result is recorded on the shared state.
//!
//! **Durability:** this queue orders writes and records failures. It does not
//! call `fsync`, does not make a batch atomic, and makes no durability claim
//! beyond whatever the [`SaveWriter`] implementation provides.
//!
//! # Retention
//!
//! Two pieces of state have different lifetimes:
//!
//! * The **newest revision per path** is retained for the queue's lifetime in a
//!   compact map. It is what rejects an older revision after its detailed
//!   outcome has been pruned, so the "old never overwrites new" guarantee does
//!   not depend on how much detail is kept. It grows by one small entry per
//!   distinct path.
//! * **Outcome detail** (the per-path last result and its error text) is kept
//!   so `flush`/`close` can report what happened. It grows with the number of
//!   distinct paths until the caller bounds it with
//!   [`SaveQueue::prune_saved`], which drops the oldest terminal records while
//!   keeping the newest `keep`. The queue never prunes automatically: a `flush`
//!   must be able to report the batch the caller just submitted.
//!
//! [`SaveQueue::status`] is O(1) and never clones the per-path table; use
//! [`SaveQueue::documents`] (O(n), explicit) or [`SaveQueue::document`]
//! (O(log n)) when the list is actually needed. `flush`/`close` necessarily
//! scan the retained detail because their reports are per path.
//!
//! **Production status:** `FileEngine` is not wired to this queue yet. The
//! existing synchronous save path remains the production behavior; this module
//! is standalone and exercised by its own tests.
//!
//! # Locking
//!
//! One mutex protects the pending deque and the per-path bookkeeping. It is
//! held only for short bookkeeping: accept/replace, pop, record a result, and
//! take a snapshot. [`SaveWriter::write`] and [`SaveWriter::commit_index`] run
//! on the writer thread with **no** queue lock held. The index is committed
//! once per drained batch, after the document writes; an index failure is
//! recorded separately from the per-document results.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};

use crate::document::DocumentRevision;
use crate::AdjustmentSet;

/// Name of the single worker thread started by [`SaveQueue::new`].
pub const SAVE_WRITER_THREAD_NAME: &str = "aurora-save-writer";

/// Default drain deadline used by `Drop`. Explicit [`SaveQueue::close`] callers
/// choose their own budget.
pub const DEFAULT_DROP_DRAIN: Duration = Duration::from_secs(2);

/// The sink that actually persists documents and the catalog index.
///
/// The real implementation will wrap the `Store` (or the equivalent catalog
/// session). Tests use fakes that record calls, fail on demand, or
/// block. Implementations are called from the single writer thread, one at a
/// time; they may block there without holding the queue lock.
pub trait SaveWriter: Send + Sync {
    /// Write one document's adjustments. Called with `rel_path` exactly as it
    /// was submitted; the queue does not normalize or rewrite paths.
    fn write(&self, rel_path: &str, adjustments: &AdjustmentSet) -> anyhow::Result<()>;

    /// Commit the catalog index after a batch of document writes. Called once
    /// per drained batch, and only if that batch processed at least one
    /// document. A failure is reported separately from document failures.
    fn commit_index(&self) -> anyhow::Result<()>;
}

/// Result of one [`SaveQueue::submit`] call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SubmitOutcome {
    /// The document was queued (possibly replacing an older pending revision
    /// for the same path).
    Accepted { revision: DocumentRevision },
    /// A revision at least as new as this one is already accepted for the
    /// path, so this submission was discarded. `newest` is that revision; the
    /// newer edit owns the path.
    Superseded { newest: DocumentRevision },
    /// The bounded queue already holds `capacity` outstanding items and this
    /// is a new path. Retry after a [`SaveQueue::flush`] or after the queue
    /// drains. (A newer revision for an already-outstanding path is accepted
    /// even at capacity.)
    Full { capacity: usize },
    /// The queue has been closed; no further work is accepted.
    Closed,
}

/// The newest known state of one path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PathState {
    /// Accepted and waiting in the bounded deque.
    Queued,
    /// Popped by the writer and currently inside [`SaveWriter::write`].
    InFlight,
    /// The write returned success and (unless a newer revision arrived) it is
    /// the path's newest bytes.
    Saved,
    /// The write returned an error. The writer thread stays alive and continues
    /// with later items.
    Failed { error: String },
    /// An older write lost to a newer revision for the same path. Never the
    /// outcome of the newest revision; exposed for completeness.
    Superseded,
    /// Not confirmed written by the deadline: still queued/in flight at a
    /// flush timeout, or removed by [`SaveQueue::close`].
    NotWritten,
}

/// Newest revision and state of one path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PathStatus {
    /// Path exactly as submitted.
    pub rel_path: String,
    /// Newest revision accepted for this path.
    pub revision: DocumentRevision,
    /// Derived state for that newest revision.
    pub state: PathState,
}

/// Cheap snapshot of the queue, taken under the bookkeeping lock.
#[derive(Clone, Debug)]
pub struct QueueStatus {
    /// Items waiting in the deque (excluding the single in-flight item).
    pub pending: usize,
    /// `0` or `1`: the writer processes one item at a time.
    pub in_flight: usize,
    /// Sum of the JSON payload sizes measured at submit time for queued items.
    pub pending_bytes: usize,
    /// Configured bound on outstanding items (queued plus in-flight).
    pub capacity: usize,
    /// Most recent document or index error seen. Sticky until another error.
    pub last_error: Option<String>,
    /// Result of the most recent completed [`SaveWriter::commit_index`]:
    /// `None` when the last commit succeeded or none has run yet.
    pub index_error: Option<String>,
    /// Whether the writer thread has not yet exited.
    pub writer_alive: bool,
    /// Whether the writer is waiting for work with nothing queued or in flight.
    pub writer_idle: bool,
    /// Whether [`SaveQueue::close`] has been called (or `Drop` is running).
    pub closed: bool,
    /// Number of paths with retained outcome detail. This is not the number of
    /// paths whose newest revision is remembered (that map is retention-only
    /// and not exposed here). Kept so `status()` stays O(1); use
    /// [`SaveQueue::documents`] for the per-path table.
    pub tracked_paths: usize,
}

impl QueueStatus {
    /// True when nothing is queued or in flight and the writer is waiting.
    pub fn is_idle(&self) -> bool {
        self.pending == 0 && self.in_flight == 0 && self.writer_idle
    }
}

/// Per-path results of one [`SaveQueue::flush`] or [`SaveQueue::close`].
///
/// This is a snapshot of the newest state for every path with retained outcome
/// detail, not a delta. It is not a transaction: some documents may be
/// [`PathState::Saved`] while another is [`PathState::Failed`] and the index
/// commit failed.
#[derive(Clone, Debug, Default)]
pub struct FlushReport {
    /// Newest per-path state, sorted by path. At the reporting deadline, paths
    /// that are still queued or in flight are converted to
    /// [`PathState::NotWritten`].
    pub documents: Vec<PathStatus>,
    /// Why the most recent completed index commit failed, if it did. A document
    /// write is never faulted for an index failure.
    pub index_error: Option<String>,
    /// True when the deadline expired before the queue drained (including the
    /// batch's index commit). The report still contains what was confirmed.
    pub timed_out: bool,
    /// Items still queued when the report was taken. In-flight work is counted
    /// by the per-path state, not here.
    pub pending: usize,
}

impl FlushReport {
    /// Number of documents written this snapshot.
    pub fn saved(&self) -> usize {
        self.documents.iter().filter(|d| d.state == PathState::Saved).count()
    }

    /// Number of documents whose write failed.
    pub fn failed(&self) -> usize {
        self.documents.iter().filter(|d| matches!(d.state, PathState::Failed { .. })).count()
    }

    /// Number of documents not confirmed written by the deadline.
    pub fn not_written(&self) -> usize {
        self.documents.iter().filter(|d| d.state == PathState::NotWritten).count()
    }

    /// The newest state for `rel_path`, if the queue has seen it.
    pub fn state_of(&self, rel_path: &str) -> Option<&PathState> {
        self.documents.iter().find(|d| d.rel_path == rel_path).map(|d| &d.state)
    }

    /// True when the queue drained in time, every document is saved (or
    /// superseded), and the index committed successfully.
    pub fn is_complete(&self) -> bool {
        !self.timed_out
            && self.index_error.is_none()
            && self
                .documents
                .iter()
                .all(|d| matches!(d.state, PathState::Saved | PathState::Superseded))
    }
}

/// A bounded, single-writer queue of adjustment documents.
///
/// Construct it with a [`SaveWriter`]; it starts exactly one named worker
/// thread. The queue is `Send + Sync` and its methods take `&self`, so callers
/// may share it (for example behind an `Arc`) between the UI and a flush path.
pub struct SaveQueue {
    inner: Arc<Inner>,
    join: Mutex<Option<JoinHandle<()>>>,
}

impl SaveQueue {
    /// Creates the queue and starts the single writer thread
    /// ([`SAVE_WRITER_THREAD_NAME`]).
    ///
    /// `capacity` bounds admissions: the items waiting in the deque plus the
    /// one being written. It is a soft bound by one item: a newer revision for
    /// a path that is already outstanding is always accepted (the pending value
    /// is replaced, or one pending item is added behind the in-flight write) so
    /// stale bytes can never win. `capacity == 0` means only such a newer
    /// revision can be accepted.
    ///
    /// If the OS refuses to start the thread, the queue is returned already
    /// closed with `last_error` set rather than panicking; subsequent
    /// submissions return [`SubmitOutcome::Closed`].
    pub fn new(writer: Arc<dyn SaveWriter>, capacity: usize) -> Self {
        let inner = Arc::new(Inner {
            writer,
            state: Mutex::new(State::new(capacity)),
            cv: Condvar::new(),
        });
        let worker = Arc::clone(&inner);
        let spawned = std::thread::Builder::new()
            .name(SAVE_WRITER_THREAD_NAME.to_owned())
            .spawn(move || writer_loop(worker));
        let join = match spawned {
            Ok(handle) => Some(handle),
            Err(error) => {
                let mut state = inner.state.lock();
                state.closed = true;
                state.writer_idle = true;
                state.writer_finished = true;
                state.last_error = Some(format!("failed to start {SAVE_WRITER_THREAD_NAME}: {error}"));
                None
            }
        };
        SaveQueue { inner, join: Mutex::new(join) }
    }

    /// Accepts an adjustment document for `rel_path`.
    ///
    /// `revision` is caller-supplied and must strictly increase per path. The
    /// call does only bookkeeping: it never performs I/O and never waits for
    /// the writer, so it is safe on the UI thread.
    pub fn submit(
        &self,
        rel_path: &str,
        adjustments: AdjustmentSet,
        revision: DocumentRevision,
    ) -> SubmitOutcome {
        // Measure the payload once here so `status()` can stay a pure lookup.
        // A serialization failure is not fatal at submit time: the writer will
        // surface it as a per-path failure, and the byte estimate falls back
        // to zero.
        let bytes = serde_json::to_vec(&adjustments).map(|payload| payload.len()).unwrap_or(0);

        let mut state = self.inner.state.lock();
        if state.closed {
            return SubmitOutcome::Closed;
        }

        // Ordering memory is retained for the queue's lifetime, even after the
        // heavier outcome detail for this path is pruned.
        if let Some(newest) = state.newest.get(rel_path).copied() {
            if revision <= newest {
                return SubmitOutcome::Superseded { newest };
            }
        }
        let same_path_outstanding = state
            .records
            .get(rel_path)
            .is_some_and(|record| record.queued.is_some() || record.in_flight.is_some());
        let outstanding = state.pending.len() + usize::from(state.in_flight.is_some());
        if !same_path_outstanding && outstanding >= state.capacity {
            return SubmitOutcome::Full { capacity: state.capacity };
        }

        if same_path_outstanding {
            // A newer revision replaces the pending value for this path. The
            // in-flight item (if any) cannot be cancelled; it will finish and
            // be recorded as superseded, then this value is written.
            let mut replaced_bytes = 0usize;
            state.pending.retain(|item| {
                if item.rel_path == rel_path {
                    replaced_bytes = replaced_bytes.saturating_add(item.bytes);
                    false
                } else {
                    true
                }
            });
            state.pending_bytes = state.pending_bytes.saturating_sub(replaced_bytes);
        }

        state.pending_bytes = state.pending_bytes.saturating_add(bytes);
        state.newest.insert(rel_path.to_owned(), revision);
        let touch = state.next_touch();
        {
            let record = state.records.entry(rel_path.to_owned()).or_insert_with(|| PathRecord {
                newest: revision,
                queued: None,
                in_flight: None,
                last: None,
                last_touch: touch,
            });
            record.newest = revision;
            record.queued = Some(revision);
            record.last_touch = touch;
        }
        state.pending.push_back(QueuedItem {
            rel_path: rel_path.to_owned(),
            adjustments,
            revision,
            bytes,
        });
        state.writer_idle = false;
        self.inner.cv.notify_all();
        SubmitOutcome::Accepted { revision }
    }

    /// Waits for the queue to drain (or `timeout`) and returns a per-path
    /// snapshot.
    ///
    /// "Drained" means: nothing queued, nothing in flight, and the writer has
    /// finished the index commit for the last batch. The wait is bounded by
    /// `timeout`, so calling this from the UI thread while the writer is stuck
    /// in I/O cannot deadlock; it returns with `timed_out == true` and
    /// `NotWritten` for the affected paths. Items are never removed by `flush`.
    pub fn flush(&self, timeout: Duration) -> FlushReport {
        let timed_out = self.inner.wait_for(timeout, |state| {
            state.pending.is_empty() && state.in_flight.is_none() && state.writer_idle
        });
        let state = self.inner.state.lock();
        build_report(&state, timed_out)
    }

    /// Stops accepting work, lets the writer drain for up to `timeout`, and
    /// reports what happened.
    ///
    /// On deadline expiry, items still queued are removed and recorded as
    /// [`PathState::NotWritten`]; a write already inside
    /// [`SaveWriter::write`] cannot be cancelled and may complete later. The
    /// report is a snapshot at the deadline. This does not imply durability.
    ///
    /// If the writer could not be joined by the deadline it is detached; it
    /// holds its own `Arc` to the shared state and exits on its own.
    pub fn close(&self, timeout: Duration) -> FlushReport {
        {
            let mut state = self.inner.state.lock();
            state.closed = true;
            self.inner.cv.notify_all();
        }
        let timed_out = self.inner.wait_for(timeout, |state| state.writer_finished);

        // Whatever is still queued at the deadline will never be written by
        // this queue; remove it and make the loss visible as NotWritten.
        {
            let mut state = self.inner.state.lock();
            if !state.pending.is_empty() {
                let leftover: Vec<QueuedItem> = state.pending.drain(..).collect();
                state.pending_bytes = 0;
                for item in leftover {
                    let touch = state.next_touch();
                    if let Some(record) = state.records.get_mut(&item.rel_path) {
                        record.queued = None;
                        record.last_touch = touch;
                        if record.newest == item.revision {
                            record.last = Some((item.revision, DocOutcome::NotWritten));
                        }
                    }
                }
                self.inner.cv.notify_all();
            }
        }

        let report = {
            let state = self.inner.state.lock();
            build_report(&state, timed_out)
        };

        let finished = self.inner.state.lock().writer_finished;
        let mut join = self.join.lock();
        if finished {
            if let Some(handle) = join.take() {
                // The worker exits without panicking; a panic here would only
                // lose the join, not the persisted work.
                let _ = handle.join();
            }
        } else {
            // Detach. The thread owns an Arc<Inner> and will exit after its
            // current write once it observes `closed` and an empty deque.
            drop(join.take());
        }
        report
    }

    /// Takes a cheap O(1) snapshot under the bookkeeping lock.
    ///
    /// This deliberately does **not** clone the per-path table; use
    /// [`Self::documents`] or [`Self::document`] for that.
    pub fn status(&self) -> QueueStatus {
        let state = self.inner.state.lock();
        QueueStatus {
            pending: state.pending.len(),
            in_flight: usize::from(state.in_flight.is_some()),
            pending_bytes: state.pending_bytes,
            capacity: state.capacity,
            last_error: state.last_error.clone(),
            index_error: state.index_error.clone(),
            writer_alive: !state.writer_finished,
            writer_idle: state.writer_idle,
            closed: state.closed,
            tracked_paths: state.records.len(),
        }
    }

    /// Returns the newest revision and state for every path with retained
    /// outcome detail, sorted by path.
    ///
    /// This is O(n) and allocates; call it only when the table is needed. Paths
    /// whose detail was removed by [`Self::prune_saved`] are absent, but their
    /// newest revision is still remembered for ordering.
    pub fn documents(&self) -> Vec<PathStatus> {
        let state = self.inner.state.lock();
        state.records.iter().map(|(path, record)| path_status(path, record)).collect()
    }

    /// Returns the newest revision and state for one path, if its detail is
    /// still retained. O(log n).
    pub fn document(&self, rel_path: &str) -> Option<PathStatus> {
        let state = self.inner.state.lock();
        state.records.get(rel_path).map(|record| path_status(rel_path, record))
    }

    /// Bounds retained outcome detail to the newest `keep` terminal paths.
    ///
    /// A terminal path has nothing queued or in flight. Records are ordered by
    /// their last touch (submit or completion); the oldest beyond `keep` are
    /// removed. Queued/in-flight records are never removed. The compact
    /// newest-revision map is **not** pruned, so an old revision for a pruned
    /// path is still rejected and a newer revision re-creates its detail.
    ///
    /// `flush`/`close` reports only cover retained detail. Call this after a
    /// flush when the caller has consumed its report, or on a periodic timer;
    /// the queue does not prune automatically because that could discard a
    /// batch result the caller has not read. Returns the number of records
    /// removed.
    pub fn prune_saved(&self, keep: usize) -> usize {
        let mut state = self.inner.state.lock();
        prune_terminal_records(&mut state.records, keep)
    }
}

impl Drop for SaveQueue {
    fn drop(&mut self) {
        let already_closed = self.inner.state.lock().closed;
        let report = self.close(DEFAULT_DROP_DRAIN);
        if already_closed {
            return;
        }
        let not_written = report.not_written();
        if not_written > 0 {
            log::warn!(
                "{not_written} adjustment document(s) were not confirmed written before the \
                 save queue was dropped"
            );
        }
        if let Some(error) = &report.index_error {
            log::warn!("save queue dropped with a catalog index error: {error}");
        }
    }
}

// ───────────────────────────── internal state ─────────────────────────────

struct Inner {
    writer: Arc<dyn SaveWriter>,
    state: Mutex<State>,
    cv: Condvar,
}

impl Inner {
    /// Waits until `done` holds or `timeout` expires. Returns `true` when the
    /// deadline expired first. The queue lock is released while waiting.
    fn wait_for(&self, timeout: Duration, done: impl Fn(&State) -> bool) -> bool {
        let start = Instant::now();
        let mut state = self.state.lock();
        loop {
            if done(&state) {
                return false;
            }
            let elapsed = start.elapsed();
            if elapsed >= timeout {
                return true;
            }
            self.cv.wait_for(&mut state, timeout - elapsed);
        }
    }
}

struct State {
    capacity: usize,
    pending: VecDeque<QueuedItem>,
    pending_bytes: usize,
    /// Newest accepted revision per path. Retained for the queue's lifetime so
    /// pruning outcome detail cannot let an old revision win.
    newest: BTreeMap<String, DocumentRevision>,
    /// Outcome detail, bounded by `prune_saved`.
    records: BTreeMap<String, PathRecord>,
    /// Monotonic touch counter used to choose which terminal records to prune.
    touch_seq: u64,
    /// Path of the item currently inside `SaveWriter::write`, if any.
    in_flight: Option<String>,
    closed: bool,
    writer_idle: bool,
    writer_finished: bool,
    last_error: Option<String>,
    index_error: Option<String>,
}

impl State {
    fn new(capacity: usize) -> Self {
        State {
            capacity,
            pending: VecDeque::new(),
            pending_bytes: 0,
            newest: BTreeMap::new(),
            records: BTreeMap::new(),
            touch_seq: 0,
            in_flight: None,
            closed: false,
            // Before the thread starts, nothing is outstanding, so a flush on a
            // fresh queue returns immediately.
            writer_idle: true,
            writer_finished: false,
            last_error: None,
            index_error: None,
        }
    }

    /// Advances the retention recency counter. Saturates rather than wrapping:
    /// a saturated counter only makes pruning tie-breaking arbitrary, never
    /// affects ordering or correctness.
    fn next_touch(&mut self) -> u64 {
        self.touch_seq = self.touch_seq.saturating_add(1);
        self.touch_seq
    }
}

struct QueuedItem {
    rel_path: String,
    adjustments: AdjustmentSet,
    revision: DocumentRevision,
    /// JSON payload size measured once at submit; used only for reporting.
    bytes: usize,
}

struct PathRecord {
    newest: DocumentRevision,
    queued: Option<DocumentRevision>,
    in_flight: Option<DocumentRevision>,
    /// Most recent completed write for this path, which may be older than
    /// `newest` while a newer revision is still queued.
    last: Option<(DocumentRevision, DocOutcome)>,
    /// Recency for [`prune_terminal_records`]: bumped on accept and on
    /// completion so "newest N paths" means most recently touched.
    last_touch: u64,
}

#[derive(Clone, Debug)]
enum DocOutcome {
    Saved,
    Failed { error: String },
    Superseded,
    NotWritten,
}

impl PathRecord {
    fn state(&self) -> PathState {
        if self.queued == Some(self.newest) {
            return PathState::Queued;
        }
        if self.in_flight == Some(self.newest) {
            return PathState::InFlight;
        }
        match &self.last {
            Some((revision, outcome)) if *revision == self.newest => outcome.state(),
            // Every accepted revision is queued, in flight, or recorded. This
            // fallback keeps an unexpected internal state visible as unwritten
            // rather than claiming success.
            _ => PathState::NotWritten,
        }
    }
}

impl DocOutcome {
    fn state(&self) -> PathState {
        match self {
            DocOutcome::Saved => PathState::Saved,
            DocOutcome::Failed { error } => PathState::Failed { error: error.clone() },
            DocOutcome::Superseded => PathState::Superseded,
            DocOutcome::NotWritten => PathState::NotWritten,
        }
    }
}

fn path_status(path: &str, record: &PathRecord) -> PathStatus {
    PathStatus { rel_path: path.to_owned(), revision: record.newest, state: record.state() }
}

/// Removes terminal outcome records older than the newest `keep`, ordered by
/// `last_touch`. Queued/in-flight records are never removed. Returns the number
/// of records dropped. The newest-revision ordering map is untouched.
fn prune_terminal_records(records: &mut BTreeMap<String, PathRecord>, keep: usize) -> usize {
    let mut terminal: Vec<(u64, String)> = records
        .iter()
        .filter(|(_, record)| record.queued.is_none() && record.in_flight.is_none())
        .map(|(path, record)| (record.last_touch, path.clone()))
        .collect();
    if terminal.len() <= keep {
        return 0;
    }
    // Newest first; ties are resolved by the deterministic BTreeMap iteration
    // order captured above.
    terminal.sort_by(|left, right| right.0.cmp(&left.0));
    let mut removed = 0;
    for (_, path) in terminal.into_iter().skip(keep) {
        if records.remove(&path).is_some() {
            removed += 1;
        }
    }
    removed
}

/// Builds a `FlushReport`. At a reporting deadline, queued/in-flight paths are
/// reported as `NotWritten` because their bytes were not confirmed.
fn build_report(state: &State, timed_out: bool) -> FlushReport {
    let documents = state
        .records
        .iter()
        .map(|(path, record)| {
            let mut status = path_status(path, record);
            if matches!(status.state, PathState::Queued | PathState::InFlight) {
                status.state = PathState::NotWritten;
            }
            status
        })
        .collect();
    FlushReport {
        documents,
        index_error: state.index_error.clone(),
        timed_out,
        pending: state.pending.len(),
    }
}

/// The single writer. Pops one item at a time, writes with no queue lock held,
/// records the result, and commits the index once the batch drains.
fn writer_loop(inner: Arc<Inner>) {
    loop {
        let item = {
            let mut state = inner.state.lock();
            loop {
                if !state.pending.is_empty() {
                    break;
                }
                if state.closed {
                    state.writer_idle = true;
                    state.writer_finished = true;
                    inner.cv.notify_all();
                    return;
                }
                state.writer_idle = true;
                inner.cv.wait(&mut state);
            }

            let item = match state.pending.pop_front() {
                Some(item) => item,
                None => continue,
            };
            state.pending_bytes = state.pending_bytes.saturating_sub(item.bytes);
            state.writer_idle = false;
            state.in_flight = Some(item.rel_path.clone());
            if let Some(record) = state.records.get_mut(&item.rel_path) {
                record.queued = None;
                record.in_flight = Some(item.revision);
            }
            item
        };

        // File I/O with no queue lock held.
        let result = inner.writer.write(&item.rel_path, &item.adjustments);

        let drained = {
            let mut state = inner.state.lock();
            state.in_flight = None;
            let newest = state.records.get(&item.rel_path).map(|record| record.newest);
            let outcome = match result {
                Ok(()) => {
                    if newest.is_some_and(|newest| newest > item.revision) {
                        DocOutcome::Superseded
                    } else {
                        DocOutcome::Saved
                    }
                }
                Err(error) => {
                    let message = format!("{error:#}");
                    state.last_error = Some(format!("{}: {message}", item.rel_path));
                    DocOutcome::Failed { error: message }
                }
            };
            let touch = state.next_touch();
            if let Some(record) = state.records.get_mut(&item.rel_path) {
                record.in_flight = None;
                record.last_touch = touch;
                // Keep the newest completed outcome; FIFO ordering normally
                // makes this item the newest anyway, and this guard makes the
                // invariant explicit.
                let superseded_by_newer_recorded = record
                    .last
                    .as_ref()
                    .is_some_and(|(revision, _)| *revision > item.revision);
                if !superseded_by_newer_recorded {
                    record.last = Some((item.revision, outcome));
                }
            }
            state.pending.is_empty()
        };

        // Commit the index once per drained batch, outside the lock. A new
        // submission may land during the commit; the next batch owns it.
        if drained {
            let index_result = inner.writer.commit_index();
            let mut state = inner.state.lock();
            match index_result {
                Ok(()) => state.index_error = None,
                Err(error) => {
                    let message = format!("{error:#}");
                    state.last_error = Some(format!("catalog index: {message}"));
                    state.index_error = Some(message);
                }
            }
            state.writer_idle = state.pending.is_empty();
            inner.cv.notify_all();
        } else {
            inner.cv.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Condvar as StdCondvar, Mutex as StdMutex};
    use std::thread;
    use std::time::{Duration, Instant};

    fn revision(value: u64) -> DocumentRevision {
        DocumentRevision::new(value)
    }

    /// A synthetic document; `style` is the payload marker the fake records.
    fn doc(marker: &str) -> AdjustmentSet {
        let mut adjustments = AdjustmentSet::default();
        adjustments.style = Some(marker.to_owned());
        adjustments
    }

    fn payload(adjustments: &AdjustmentSet) -> String {
        adjustments.style.clone().unwrap_or_default()
    }

    /// A one-shot gate: the first `write` announces that it entered and then
    /// waits until released. Later writes pass through.
    struct BlockGate {
        entered: StdMutex<bool>,
        entered_cv: StdCondvar,
        released: StdMutex<bool>,
        released_cv: StdCondvar,
    }

    impl BlockGate {
        fn new() -> Self {
            BlockGate {
                entered: StdMutex::new(false),
                entered_cv: StdCondvar::new(),
                released: StdMutex::new(false),
                released_cv: StdCondvar::new(),
            }
        }

        fn block(&self) {
            {
                let mut entered = self.entered.lock().expect("gate entered mutex");
                *entered = true;
                self.entered_cv.notify_all();
            }
            let mut released = self.released.lock().expect("gate released mutex");
            while !*released {
                released = self.released_cv.wait(released).expect("gate released wait");
            }
        }

        fn wait_until_entered(&self, timeout: Duration) -> bool {
            let deadline = Instant::now().checked_add(timeout);
            let mut entered = self.entered.lock().expect("gate entered mutex");
            loop {
                if *entered {
                    return true;
                }
                let remaining = match deadline {
                    Some(deadline) => deadline.saturating_duration_since(Instant::now()),
                    None => Duration::from_secs(60),
                };
                if remaining.is_zero() {
                    return false;
                }
                let (guard, wait) =
                    self.entered_cv.wait_timeout(entered, remaining).expect("gate entered wait");
                entered = guard;
                if wait.timed_out() && !*entered {
                    return false;
                }
            }
        }

        fn release(&self) {
            let mut released = self.released.lock().expect("gate released mutex");
            *released = true;
            self.released_cv.notify_all();
        }
    }

    struct FakeWriter {
        writes: StdMutex<Vec<(String, String)>>,
        fail_paths: Vec<String>,
        index_error: StdMutex<Option<String>>,
        index_calls: AtomicUsize,
        block: Option<Arc<BlockGate>>,
    }

    impl FakeWriter {
        fn base(block: Option<Arc<BlockGate>>) -> Self {
            FakeWriter {
                writes: StdMutex::new(Vec::new()),
                fail_paths: Vec::new(),
                index_error: StdMutex::new(None),
                index_calls: AtomicUsize::new(0),
                block,
            }
        }

        fn new() -> Arc<Self> {
            Arc::new(Self::base(None))
        }

        fn blocking(gate: Arc<BlockGate>) -> Arc<Self> {
            Arc::new(Self::base(Some(gate)))
        }

        fn failing(paths: &[&str]) -> Arc<Self> {
            let mut writer = Self::base(None);
            writer.fail_paths = paths.iter().map(|path| (*path).to_owned()).collect();
            Arc::new(writer)
        }

        fn with_index_error(message: &str) -> Arc<Self> {
            let mut writer = Self::base(None);
            writer.index_error = StdMutex::new(Some(message.to_owned()));
            Arc::new(writer)
        }

        fn writes_for<'a>(&'a self, path: &str) -> Vec<String> {
            self.writes
                .lock()
                .expect("writes mutex")
                .iter()
                .filter(|(written, _)| written == path)
                .map(|(_, marker)| marker.clone())
                .collect()
        }
    }

    impl SaveWriter for FakeWriter {
        fn write(&self, rel_path: &str, adjustments: &AdjustmentSet) -> anyhow::Result<()> {
            if let Some(gate) = &self.block {
                gate.block();
            }
            if self.fail_paths.iter().any(|path| path == rel_path) {
                return Err(anyhow::anyhow!("simulated write failure for {rel_path}"));
            }
            self.writes
                .lock()
                .expect("writes mutex")
                .push((rel_path.to_owned(), payload(adjustments)));
            Ok(())
        }

        fn commit_index(&self) -> anyhow::Result<()> {
            self.index_calls.fetch_add(1, Ordering::SeqCst);
            match self.index_error.lock().expect("index error mutex").clone() {
                Some(message) => Err(anyhow::anyhow!(message)),
                None => Ok(()),
            }
        }
    }

    #[test]
    fn accepts_and_writes_distinct_paths_in_submission_order() {
        let writer = FakeWriter::new();
        let queue = SaveQueue::new(writer.clone(), 8);

        for path in ["a.cr3", "b.cr3", "c.cr3"] {
            assert_eq!(
                queue.submit(path, doc("r1"), revision(1)),
                SubmitOutcome::Accepted { revision: revision(1) }
            );
        }

        let report = queue.flush(Duration::from_secs(5));
        assert!(!report.timed_out, "{report:?}");
        for path in ["a.cr3", "b.cr3", "c.cr3"] {
            assert_eq!(report.state_of(path), Some(&PathState::Saved), "{report:?}");
        }
        assert_eq!(report.saved(), 3);
        assert_eq!(report.not_written(), 0);
        assert!(report.is_complete(), "{report:?}");

        let writes = writer.writes.lock().expect("writes mutex").clone();
        let order: Vec<&str> = writes.iter().map(|(path, _)| path.as_str()).collect();
        assert_eq!(order, ["a.cr3", "b.cr3", "c.cr3"]);
        assert!(writer.index_calls.load(Ordering::SeqCst) >= 1);

        let status = queue.status();
        assert_eq!(status.pending, 0);
        assert_eq!(status.in_flight, 0);
        assert!(status.is_idle(), "{status:?}");
        assert_eq!(status.tracked_paths, 3);
        assert!(queue.documents().iter().all(|document| document.state == PathState::Saved));
    }

    #[test]
    fn prune_saved_bounds_detail_but_keeps_revision_ordering() {
        let writer = FakeWriter::new();
        let queue = SaveQueue::new(writer.clone(), 8);

        for path in ["a.cr3", "b.cr3", "c.cr3"] {
            assert!(matches!(
                queue.submit(path, doc("r1"), revision(1)),
                SubmitOutcome::Accepted { .. }
            ));
        }
        let report = queue.flush(Duration::from_secs(5));
        assert_eq!(queue.documents().len(), 3, "{report:?}");
        assert_eq!(queue.status().tracked_paths, 3);

        // Keep only the most recently touched terminal record.
        assert_eq!(queue.prune_saved(1), 2);
        assert_eq!(queue.documents().len(), 1);
        assert_eq!(queue.status().tracked_paths, 1);
        assert!(queue.document("c.cr3").is_some(), "the newest-touched record survives");

        // Ordering memory survived pruning: a stale revision for a pruned path
        // is still rejected, so it cannot overwrite the newer bytes.
        assert_eq!(
            queue.submit("a.cr3", doc("stale"), revision(0)),
            SubmitOutcome::Superseded { newest: revision(1) }
        );
        // A newer revision is accepted and re-creates the pruned detail.
        assert_eq!(
            queue.submit("a.cr3", doc("r2"), revision(2)),
            SubmitOutcome::Accepted { revision: revision(2) }
        );
        let report = queue.flush(Duration::from_secs(5));
        assert_eq!(report.state_of("a.cr3"), Some(&PathState::Saved), "{report:?}");
        assert_eq!(queue.document("a.cr3").map(|detail| detail.revision), Some(revision(2)));

        // Keeping at least the terminal count removes nothing.
        assert_eq!(queue.prune_saved(usize::MAX), 0);
    }

    #[test]
    fn newer_revision_replaces_pending_older_and_never_writes_it() {
        let gate = Arc::new(BlockGate::new());
        let writer = FakeWriter::blocking(gate.clone());
        let queue = SaveQueue::new(writer.clone(), 8);

        // Occupy the writer so the path below stays pending and is coalesced.
        assert_eq!(
            queue.submit("blocker.cr3", doc("gate"), revision(1)),
            SubmitOutcome::Accepted { revision: revision(1) }
        );
        assert!(gate.wait_until_entered(Duration::from_secs(5)), "writer never started");

        assert_eq!(
            queue.submit("a.cr3", doc("r1"), revision(1)),
            SubmitOutcome::Accepted { revision: revision(1) }
        );
        // A distinct path must survive the coalescing of `a.cr3`.
        assert_eq!(
            queue.submit("b.cr3", doc("b1"), revision(1)),
            SubmitOutcome::Accepted { revision: revision(1) }
        );
        assert_eq!(
            queue.submit("a.cr3", doc("r2"), revision(2)),
            SubmitOutcome::Accepted { revision: revision(2) }
        );
        // A stale revision can no longer win.
        assert_eq!(
            queue.submit("a.cr3", doc("r1-again"), revision(1)),
            SubmitOutcome::Superseded { newest: revision(2) }
        );

        let status = queue.status();
        assert_eq!(status.pending, 2, "a.cr3 must occupy one slot, b.cr3 another: {status:?}");
        assert_eq!(
            queue.document("a.cr3").map(|d| d.state),
            Some(PathState::Queued)
        );

        gate.release();
        let report = queue.flush(Duration::from_secs(5));
        assert!(!report.timed_out, "{report:?}");
        assert_eq!(report.state_of("a.cr3"), Some(&PathState::Saved));
        assert_eq!(report.state_of("b.cr3"), Some(&PathState::Saved));

        let a_writes = writer.writes_for("a.cr3");
        assert_eq!(a_writes, ["r2"], "the replaced r1 must never reach the writer");
        assert_eq!(writer.writes_for("b.cr3"), ["b1"], "coalescing must not drop other paths");
    }

    #[test]
    fn bounded_queue_returns_full_instead_of_blocking() {
        let gate = Arc::new(BlockGate::new());
        let writer = FakeWriter::blocking(gate.clone());
        // Capacity two counts the in-flight item plus one queued item.
        let queue = SaveQueue::new(writer.clone(), 2);

        assert_eq!(
            queue.submit("a.cr3", doc("r1"), revision(1)),
            SubmitOutcome::Accepted { revision: revision(1) }
        );
        assert!(gate.wait_until_entered(Duration::from_secs(5)), "writer never blocked");

        // Fill the bound: one in flight (a) plus one queued (b).
        assert_eq!(
            queue.submit("b.cr3", doc("b1"), revision(1)),
            SubmitOutcome::Accepted { revision: revision(1) }
        );
        let started = Instant::now();
        assert_eq!(
            queue.submit("c.cr3", doc("c1"), revision(1)),
            SubmitOutcome::Full { capacity: 2 }
        );
        assert!(started.elapsed() < Duration::from_secs(1), "submit must not block behind the writer");

        // A newer revision for the path already in flight must still be
        // accepted; rejecting it would leave stale bytes on disk. It may sit
        // one item above `capacity` but must not displace queued b.
        assert_eq!(
            queue.submit("a.cr3", doc("r2"), revision(2)),
            SubmitOutcome::Accepted { revision: revision(2) }
        );
        let status = queue.status();
        assert_eq!(status.pending, 2, "b.cr3 must survive a.cr3's newer revision: {status:?}");
        assert_eq!(queue.document("b.cr3").map(|d| d.state), Some(PathState::Queued));
        assert_eq!(
            queue.document("a.cr3"),
            Some(PathStatus {
                rel_path: "a.cr3".to_owned(),
                revision: revision(2),
                state: PathState::Queued,
            })
        );
        // The soft bound still refuses another new path.
        assert_eq!(
            queue.submit("d.cr3", doc("d1"), revision(1)),
            SubmitOutcome::Full { capacity: 2 }
        );

        gate.release();
        let report = queue.flush(Duration::from_secs(5));
        assert_eq!(report.state_of("a.cr3"), Some(&PathState::Saved), "{report:?}");
        assert_eq!(report.state_of("b.cr3"), Some(&PathState::Saved), "{report:?}");
        assert_eq!(writer.writes_for("a.cr3").last().map(String::as_str), Some("r2"));
        assert_eq!(writer.writes_for("b.cr3"), ["b1"], "the queued other path must still be written");
        assert!(writer.writes_for("c.cr3").is_empty(), "a Full submission must not be written");
        assert!(writer.writes_for("d.cr3").is_empty(), "a Full submission must not be written");
    }

    #[test]
    fn writer_failure_is_reported_per_path_and_writer_survives() {
        let writer = FakeWriter::failing(&["bad.cr3"]);
        let queue = SaveQueue::new(writer.clone(), 8);

        queue.submit("bad.cr3", doc("bad"), revision(1));
        queue.submit("good.cr3", doc("good"), revision(1));

        let report = queue.flush(Duration::from_secs(5));
        assert!(!report.timed_out, "{report:?}");
        match report.state_of("bad.cr3") {
            Some(PathState::Failed { error }) => {
                assert!(error.contains("bad.cr3"), "error should name the path: {error}");
            }
            other => panic!("expected a per-path failure, got {other:?}"),
        }
        assert_eq!(report.state_of("good.cr3"), Some(&PathState::Saved));
        assert_eq!(report.failed(), 1);
        assert!(queue.status().last_error.as_deref().is_some_and(|e| e.contains("bad.cr3")));

        // The writer thread is still alive and takes new work.
        assert_eq!(queue.status().writer_alive, true);
        queue.submit("later.cr3", doc("later"), revision(1));
        let report = queue.flush(Duration::from_secs(5));
        assert_eq!(report.state_of("later.cr3"), Some(&PathState::Saved), "{report:?}");
    }

    #[test]
    fn index_failure_is_reported_separately_and_clears_on_recovery() {
        let writer = FakeWriter::with_index_error("index locked");
        let queue = SaveQueue::new(writer.clone(), 8);

        queue.submit("a.cr3", doc("r1"), revision(1));
        let report = queue.flush(Duration::from_secs(5));
        assert_eq!(report.state_of("a.cr3"), Some(&PathState::Saved), "{report:?}");
        assert_eq!(report.index_error.as_deref(), Some("index locked"));
        assert!(!report.is_complete());

        // A document write is never faulted for the index failure.
        match report.state_of("a.cr3") {
            Some(PathState::Failed { .. }) => panic!("document must stay Saved"),
            _ => {}
        }

        *writer.index_error.lock().expect("index error mutex") = None;
        queue.submit("b.cr3", doc("r2"), revision(1));
        let report = queue.flush(Duration::from_secs(5));
        assert_eq!(report.state_of("b.cr3"), Some(&PathState::Saved), "{report:?}");
        assert_eq!(report.index_error, None);
        assert!(report.is_complete(), "{report:?}");
    }

    #[test]
    fn flush_timeout_reports_not_written_without_dropping_work() {
        let gate = Arc::new(BlockGate::new());
        let writer = FakeWriter::blocking(gate.clone());
        let queue = SaveQueue::new(writer.clone(), 8);

        queue.submit("blocker.cr3", doc("gate"), revision(1));
        assert!(gate.wait_until_entered(Duration::from_secs(5)), "writer never blocked");
        queue.submit("a.cr3", doc("r1"), revision(1));

        let report = queue.flush(Duration::from_millis(50));
        assert!(report.timed_out, "{report:?}");
        assert_eq!(report.state_of("a.cr3"), Some(&PathState::NotWritten));
        assert_eq!(report.pending, 1);

        // The accepted item is still in the queue, not silently lost.
        assert_eq!(queue.status().pending, 1);

        gate.release();
        let report = queue.flush(Duration::from_secs(5));
        assert!(!report.timed_out, "{report:?}");
        assert_eq!(report.state_of("a.cr3"), Some(&PathState::Saved));
        assert_eq!(writer.writes_for("a.cr3"), ["r1"]);
    }

    #[test]
    fn close_drains_what_it_can_and_refuses_new_work() {
        let writer = FakeWriter::new();
        let queue = SaveQueue::new(writer.clone(), 8);

        queue.submit("a.cr3", doc("r1"), revision(1));
        let report = queue.close(Duration::from_secs(5));
        assert!(!report.timed_out, "{report:?}");
        assert_eq!(report.state_of("a.cr3"), Some(&PathState::Saved));
        assert_eq!(writer.writes_for("a.cr3"), ["r1"]);

        assert_eq!(queue.submit("b.cr3", doc("r2"), revision(1)), SubmitOutcome::Closed);
        assert!(queue.status().closed);
        assert!(!queue.status().writer_alive);
    }

    #[test]
    fn close_with_expired_deadline_reports_pending_as_not_written() {
        let gate = Arc::new(BlockGate::new());
        let writer = FakeWriter::blocking(gate.clone());
        let queue = SaveQueue::new(writer.clone(), 8);

        queue.submit("blocker.cr3", doc("gate"), revision(1));
        assert!(gate.wait_until_entered(Duration::from_secs(5)), "writer never blocked");
        queue.submit("a.cr3", doc("r1"), revision(1));

        let report = queue.close(Duration::ZERO);
        assert!(report.timed_out, "{report:?}");
        assert_eq!(report.state_of("a.cr3"), Some(&PathState::NotWritten));
        let status = queue.status();
        assert!(status.closed);
        assert_eq!(status.pending, 0, "abandoned work is removed, not left to leak");

        gate.release();
        // Let the detached worker finish the item that was already in flight.
        let report = queue.flush(Duration::from_secs(5));
        assert_eq!(report.state_of("blocker.cr3"), Some(&PathState::Saved), "{report:?}");
        // `a.cr3` was only queued when the deadline expired; it must not be
        // written by a worker that was told to stop.
        assert!(writer.writes_for("a.cr3").is_empty());
    }

    #[test]
    fn drop_drains_accepted_work() {
        let writer = FakeWriter::new();
        {
            let queue = SaveQueue::new(writer.clone(), 8);
            queue.submit("a.cr3", doc("r1"), revision(1));
        } // Drop runs and gives the worker a bounded chance to drain.

        assert_eq!(writer.writes_for("a.cr3"), ["r1"]);
    }

    #[test]
    fn status_and_flush_stay_responsive_while_writer_blocks() {
        let gate = Arc::new(BlockGate::new());
        let writer = FakeWriter::blocking(gate.clone());
        let queue = Arc::new(SaveQueue::new(writer.clone(), 8));

        queue.submit("a.cr3", doc("r1"), revision(1));
        assert!(gate.wait_until_entered(Duration::from_secs(5)), "writer never blocked");

        // If the queue lock were held across `write`, either call below would
        // never return. The watchdog turns that into a clear failure.
        let probe_queue = Arc::clone(&queue);
        let (sent, received) = mpsc::channel();
        let probe = thread::spawn(move || {
            let status = probe_queue.status();
            let report = probe_queue.flush(Duration::from_millis(50));
            let _ = sent.send((status.in_flight, status.writer_alive, report.timed_out));
        });
        let observed = received.recv_timeout(Duration::from_secs(5));
        // Always release the gate before asserting so a failing test cannot
        // leave the writer blocked.
        gate.release();
        let (in_flight, writer_alive, timed_out) = observed.expect(
            "status()/flush() blocked while the writer was in write(); the queue lock is held \
             across I/O",
        );
        assert_eq!(in_flight, 1);
        assert!(writer_alive);
        assert!(timed_out);

        let report = queue.flush(Duration::from_secs(5));
        assert_eq!(report.state_of("a.cr3"), Some(&PathState::Saved), "{report:?}");
        probe.join().expect("probe thread");
    }
}
