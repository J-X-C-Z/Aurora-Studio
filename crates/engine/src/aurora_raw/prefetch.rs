//! Neighbour prefetch for image switching.
//!
//! Switching to a RAW in fit view pays one full-sensor decode (~0.1–1 s cold,
//! see [`crate::raw_cache`]), which is the dominant part of the "next photo"
//! latency. This module warms the *neighbours* of the current image on one
//! below-normal-priority worker while the user is still looking at the current
//! frame, so stepping through a folder mostly hits the decode cache instead of
//! decoding again.
//!
//! What it is and is not:
//! - It never renders, never changes a pixel and never writes its own cache:
//!   the warmer is [`crate::raw_cache::get_or_decode16`], so a prefetched frame
//!   lands in exactly the entry the interactive path will read (disk key by
//!   content identity + decode route, plus the retained pending frame).
//! - It is one worker thread, so it occupies at most one of
//!   `raw_cache::decode_concurrency()` decode slots (default 2) and the
//!   interactive decode always keeps one. It also reuses the existing
//!   single-flight map, so prefetching the image the user just clicked joins
//!   that decode instead of starting a second one.
//! - Memory is bounded by the existing budgets: `MAX_PENDING_NATIVE_BYTES`
//!   (`AURORA_RAW_PENDING_BYTES`) for retained frames and the disk-cache budget
//!   for stored entries. Nothing new is retained here.
//! - Cancellation is generation-based: a new [`Prefetcher::request`] or
//!   [`Prefetcher::cancel`] bumps the generation, the item in flight is allowed
//!   to finish (it already paid its decode and its result is a valid cache
//!   entry), and every remaining item is dropped. Pending work therefore only
//!   ever holds the newest switch, matching the repo rule that the pending slot
//!   keeps the latest value.
//!
//! `AURORA_RAW_PREFETCH=0` (`false`/`off`) disables it.

use crate::aurora_raw::prewarm::lower_current_thread_priority;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// Offsets warmed around the current image, ahead first: the next photo is the
/// likeliest target when someone steps forward, the previous one covers stepping
/// back, and `+2` covers holding the next key.
pub const DEFAULT_OFFSETS: [i32; 3] = [1, -1, 2];

/// Most neighbours one switch will queue. Bounds how much a single request can
/// keep the worker busy after the user has moved on.
pub const MAX_TARGETS: usize = 4;

/// Decodes one neighbour. Returns whether the frame is now cached.
pub type WarmFn = Arc<dyn Fn(&Path, u32) -> bool + Send + Sync>;

/// Default hold before the worker starts decoding a queued plan.
///
/// A burst of arrow-key steps would otherwise keep a decode in flight the whole
/// time, and the CUDA domain is one mutex: a prefetch decode can sit between the
/// user's click and their preview. Holding the plan for a moment means a burst
/// warms nothing (each new switch replaces the plan) while a deliberate pause
/// still gets its neighbours warm. `AURORA_RAW_PREFETCH_DELAY_MS` overrides it.
pub const DEFAULT_DELAY_MS: u64 = 400;

/// The hold before decoding, clamped to a range that stays a debounce and not a
/// stall.
fn debounce_ms() -> u64 {
    std::env::var("AURORA_RAW_PREFETCH_DELAY_MS")
        .ok()
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_DELAY_MS)
        .min(5_000)
}

/// Returns `false` when `AURORA_RAW_PREFETCH` is explicitly disabled
/// (`0`, `false`, or `off`). Unset means enabled.
pub fn prefetch_enabled() -> bool {
    match std::env::var("AURORA_RAW_PREFETCH") {
        Ok(value) => {
            let trimmed = value.trim();
            !(trimmed == "0"
                || trimmed.eq_ignore_ascii_case("false")
                || trimmed.eq_ignore_ascii_case("off"))
        }
        Err(_) => true,
    }
}

/// The neighbour indices to warm for `current` within `len` images, clamped to
/// the folder, de-duplicated, in priority order.
///
/// Pure arithmetic: the app calls it on the UI thread, so it must not touch the
/// filesystem. An offset that leaves the folder is dropped individually rather
/// than clamped onto a neighbour that is already covered.
pub fn neighbor_indices(current: usize, len: usize, offsets: &[i32]) -> Vec<usize> {
    if len == 0 || current >= len {
        return Vec::new();
    }
    let mut out: Vec<usize> = Vec::with_capacity(offsets.len().min(MAX_TARGETS));
    for &offset in offsets {
        if out.len() >= MAX_TARGETS {
            break;
        }
        let Ok(target) = i64::try_from(current).map(|current| current + i64::from(offset)) else {
            continue;
        };
        if target < 0 || target as usize >= len || target as usize == current {
            continue;
        }
        let target = target as usize;
        if !out.contains(&target) {
            out.push(target);
        }
    }
    out
}

/// Counters for diagnostics and the acceptance tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefetchStats {
    /// Switches that queued a plan.
    pub requested: u64,
    /// Neighbours handed to the warmer.
    pub decoded: u64,
    /// Neighbours skipped because the decode cache already served them.
    pub already_warm: u64,
    /// Neighbours dropped because a newer switch replaced the plan.
    pub cancelled: u64,
    /// Neighbours whose decode failed (the interactive path retries them).
    pub failed: u64,
}

/// The neighbour paths to warm for `current`, ahead-first, already filtered to
/// files that actually have a 16-bit decode (a RAW).
///
/// The RAW filter runs *inside* the offset walk on purpose: filtering afterwards
/// would let a JPEG neighbour consume a slot of [`MAX_TARGETS`] and leave the
/// plan shorter than the folder allows. `raw::is_raw` is an extension check, so
/// this stays off the filesystem on the UI thread.
pub fn neighbor_paths(current: usize, paths: &[PathBuf], offsets: &[i32]) -> Vec<PathBuf> {
    neighbor_paths_by(current, paths, offsets, |p|p.as_path())
}

/// Only the selected adjacent paths are cloned, including in a large sorted catalog.
pub fn neighbor_paths_by<T>(current:usize, paths:&[T], offsets:&[i32], path_of:impl Fn(&T)->&std::path::Path) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::with_capacity(offsets.len().min(MAX_TARGETS));
    for &offset in offsets {
        if out.len() >= MAX_TARGETS {
            break;
        }
        let Ok(target) = i64::try_from(current).map(|current| current + i64::from(offset)) else {
            continue;
        };
        if target < 0 || target as usize >= paths.len() || target as usize == current {
            continue;
        }
        let path=path_of(&paths[target as usize]);
        if !crate::raw::is_raw(path) || out.iter().any(|p|p.as_path()==path) {
            continue;
        }
        out.push(path.to_path_buf());
    }
    out
}

struct Job {
    generation: u64,
    targets: Vec<PathBuf>,
    long_side: u32,
}

struct State {
    generation: u64,
    job: Option<Job>,
    busy: bool,
    stats: PrefetchStats,
    /// Set once the worker thread is running; a failed spawn leaves it `false`
    /// and requests become no-ops instead of queueing forever.
    worker: bool,
}

impl State {
    const fn new() -> Self {
        Self {
            generation: 0,
            job: None,
            busy: false,
            stats: PrefetchStats {
                requested: 0,
                decoded: 0,
                already_warm: 0,
                cancelled: 0,
                failed: 0,
            },
            worker: false,
        }
    }
}

struct Shared {
    warmer: WarmFn,
    state: Mutex<State>,
    ready: Condvar,
    delay: std::time::Duration,
}

/// A prefetch queue with one worker.
///
/// `warmer` is "decode through the shared cache"; the default is
/// [`Self::cache_warmer`]. Tests inject their own so the queue's scheduling,
/// cancellation and counters are covered without a RAW file or a GPU.
pub struct Prefetcher {
    shared: Arc<Shared>,
}

impl Prefetcher {
    /// The production warmer: the interactive 16-bit decode cache.
    pub fn cache_warmer() -> WarmFn {
        Arc::new(|path: &Path, long_side: u32| {
            crate::raw_cache::get_or_decode16(path, long_side).is_some()
        })
    }

    pub fn new(warmer: WarmFn) -> Self {
        Self::with_delay(warmer, 0)
    }

    /// [`Self::new`] plus the hold before a queued plan starts decoding.
    pub fn with_delay(warmer: WarmFn, delay_ms: u64) -> Self {
        Self {
            shared: Arc::new(Shared {
                warmer,
                state: Mutex::new(State::new()),
                ready: Condvar::new(),
                delay: std::time::Duration::from_millis(delay_ms),
            }),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.shared
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Queues `targets` for warming at `long_side`, replacing any plan still
    /// pending. Returns the new generation.
    pub fn request(&self, targets: Vec<PathBuf>, long_side: u32) -> u64 {
        if targets.is_empty() {
            self.cancel();
            return self.generation();
        }
        let targets: Vec<PathBuf> = targets.into_iter().take(MAX_TARGETS).collect();
        let mut state = self.lock();
        state.generation += 1;
        let generation = state.generation;
        // A plan with the same targets that is still *queued* (not yet taken by
        // the worker) is kept, but its token is refreshed: re-queueing it would
        // restart identical work, and leaving the old token would make the
        // worker treat the plan as stale and drop it.
        let identical = state
            .job
            .as_ref()
            .is_some_and(|job| job.long_side == long_side && job.targets == targets);
        if identical {
            if let Some(job) = state.job.as_mut() {
                job.generation = generation;
            }
        } else {
            state.job = Some(Job { generation, targets, long_side });
            state.stats.requested += 1;
        }
        self.ensure_worker(&mut state);
        drop(state);
        self.shared.ready.notify_all();
        generation
    }

    /// Drops the pending plan; the item in flight becomes the last one.
    pub fn cancel(&self) {
        let mut state = self.lock();
        state.generation += 1;
        if let Some(job) = state.job.take() {
            state.stats.cancelled += job.targets.len() as u64;
        }
        drop(state);
        self.shared.ready.notify_all();
    }

    pub fn generation(&self) -> u64 {
        self.lock().generation
    }

    pub fn stats(&self) -> PrefetchStats {
        self.lock().stats
    }

    /// Blocks until the queue is empty and no item is in flight. Test hook: the
    /// app never waits on prefetch.
    pub fn wait_idle(&self) -> bool {
        let mut state = self.lock();
        while state.job.is_some() || state.busy {
            let (next, timeout) = self
                .shared
                .ready
                .wait_timeout(state, std::time::Duration::from_secs(30))
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            state = next;
            if timeout.timed_out() && (state.job.is_some() || state.busy) {
                return false;
            }
        }
        true
    }

    fn ensure_worker(&self, state: &mut State) {
        if state.worker {
            return;
        }
        state.worker = true;
        let shared = Arc::clone(&self.shared);
        let spawned = std::thread::Builder::new()
            .name("aurora-raw-prefetch".into())
            .spawn(move || worker_loop(shared));
        if spawned.is_err() {
            // The caller already holds the state lock; do not re-enter it on spawn failure.
            state.worker = false;
            state.job = None;
            log::debug!("raw prefetch worker failed to start; neighbour warming disabled");
        }
    }
}

fn worker_loop(shared: Arc<Shared>) {
    lower_current_thread_priority();
    loop {
        let job = {
            let mut state = shared
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            while state.job.is_none() {
                state = shared
                    .ready
                    .wait(state)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            let Some(job) = state.job.take() else { continue };
            state.busy = true;
            job
        };
        // Hold the plan for the debounce window. A newer switch notifies early
        // and makes this plan stale, so it is dropped instead of decoded; a
        // deliberate pause runs it. Nothing is held while waiting.
        if !shared.delay.is_zero() {
            let state = shared
                .state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let _ = shared.ready.wait_timeout(state, shared.delay);
        }
        run_job(&shared, &job);
        let mut state = shared
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.busy = false;
        drop(state);
        shared.ready.notify_all();
    }
}

fn run_job(shared: &Shared, job: &Job) {
    for (index, path) in job.targets.iter().enumerate() {
        // Checked per item: a switch during a long decode must not leave the
        // worker chewing through the old plan.
        if is_stale(shared, job.generation) {
            let rest = (job.targets.len() - index) as u64;
            count(shared, |stats| stats.cancelled += rest);
            return;
        }
        if crate::raw_cache::is_source16_warm(path, job.long_side) {
            count(shared, |stats| stats.already_warm += 1);
            continue;
        }
        count(shared, |stats| stats.decoded += 1);
        if !(shared.warmer)(path, job.long_side) {
            count(shared, |stats| stats.failed += 1);
        }
    }
}

fn is_stale(shared: &Shared, generation: u64) -> bool {
    shared
        .state
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .generation
        != generation
}

fn count(shared: &Shared, update: impl FnOnce(&mut PrefetchStats)) {
    update(
        &mut shared
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .stats,
    );
}

fn global() -> &'static Prefetcher {
    static GLOBAL: OnceLock<Prefetcher> = OnceLock::new();
    GLOBAL.get_or_init(|| Prefetcher::with_delay(Prefetcher::cache_warmer(), debounce_ms()))
}

/// Warms the neighbours of `current` in `paths` — the call the app makes after
/// switching images. Does nothing when prefetch is disabled.
///
/// `paths` is the visible order (what the browser shows and what the arrow keys
/// step through), so `current ± 1` is the photo the user is about to see. The
/// long side is [`crate::raw_cache::PREVIEW_LONG_SIDE`], i.e. the entry the next
/// preview will read, not a second copy at another size.
pub fn request_neighbors(current: usize, paths: &[PathBuf]) -> u64 {
    request_neighbors_at(current, paths, crate::raw_cache::PREVIEW_LONG_SIDE)
}

/// [`request_neighbors`] with an explicit long side, for callers that ask for a
/// different cache entry (1:1 zoom prefetch) and for the acceptance harness.
///
/// The folder engine calls this with [`crate::file_engine`]'s source long side,
/// which is the entry the next preview request reads.
pub fn request_neighbors_at(current: usize, paths: &[PathBuf], long_side: u32) -> u64 {
    if !prefetch_enabled() {
        global().cancel();
        return global().generation();
    }
    let targets = neighbor_paths(current, paths, &DEFAULT_OFFSETS);
    global().request(targets, long_side)
}

/// Submit a bounded plan already calculated from the user's visible selection order.
pub fn request_visible_targets(targets:Vec<PathBuf>,long_side:u32) {
    if !prefetch_enabled() { global().cancel();return; }
    global().request(targets.into_iter().filter(|p|crate::raw::is_raw(p)).take(MAX_TARGETS).collect(),long_side);
}

/// Drops any pending neighbour plan (closing the folder, leaving the viewer).
pub fn cancel_pending() {
    global().cancel();
}

/// Current counters of the process-wide prefetcher.
#[allow(dead_code)] // diagnostics and the acceptance harness read it
pub fn stats() -> PrefetchStats {
    global().stats()
}

/// Blocks until the process-wide queue is empty and nothing is in flight.
///
/// The app never waits on prefetch; this exists so an acceptance test can prove
/// what a plan did (and did not) queue before it inspects the counters.
pub fn wait_idle() -> bool {
    global().wait_idle()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn paths(count: usize) -> Vec<PathBuf> {
        (0..count).map(|index| PathBuf::from(format!("P{index}.nef"))).collect()
    }

    #[test]
    fn neighbors_are_ahead_first_forward_and_clamped() {
        assert_eq!(neighbor_indices(5, 10, &DEFAULT_OFFSETS), vec![6, 4, 7]);
        // First image: only forward neighbours, and `-1` drops instead of
        // wrapping onto the last file.
        assert_eq!(neighbor_indices(0, 10, &DEFAULT_OFFSETS), vec![1, 2]);
        // Last image: only backward neighbours.
        assert_eq!(neighbor_indices(9, 10, &DEFAULT_OFFSETS), vec![8]);
        // Single image and empty folder.
        assert!(neighbor_indices(0, 1, &DEFAULT_OFFSETS).is_empty());
        assert!(neighbor_indices(0, 0, &DEFAULT_OFFSETS).is_empty());
        // Current index past the end is a no-op, not a panic.
        assert!(neighbor_indices(11, 10, &DEFAULT_OFFSETS).is_empty());
    }

    #[test]
    fn neighbors_dedupe_and_respect_the_target_cap() {
        // Repeated offsets and an offset that lands on the current image.
        assert_eq!(neighbor_indices(3, 9, &[1, 1, 0, -1]), vec![4, 2]);
        // More candidates than MAX_TARGETS keeps the highest-priority ones.
        let many = [1, -1, 2, -2, 3, -3];
        let out = neighbor_indices(5, 20, &many);
        assert_eq!(out.len(), MAX_TARGETS);
        assert_eq!(out, vec![6, 4, 7, 3]);
        // An offset that would overflow is skipped, not wrapped.
        assert!(neighbor_indices(usize::MAX, usize::MAX, &[1]).is_empty());
    }

    /// Synthetic queue test: no RAW, no GPU. The plan order must be preserved
    /// and every item must reach the warmer exactly once.
    #[test]
    fn queue_runs_the_plan_in_order() {
        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let prefetcher = Prefetcher::new({
            let seen = Arc::clone(&seen);
            Arc::new(move |path: &Path, _long_side: u32| {
                seen.lock().unwrap().push(path.display().to_string());
                true
            })
        });
        let generation = prefetcher.request(paths(3), 1024);
        assert!(generation > 0);
        assert!(prefetcher.wait_idle(), "queue must drain");
        assert_eq!(
            seen.lock().unwrap().clone(),
            vec!["P0.nef", "P1.nef", "P2.nef"],
            "plan order preserved"
        );
        let stats = prefetcher.stats();
        assert_eq!(stats.requested, 1);
        assert_eq!(stats.decoded, 3);
        assert_eq!(stats.failed, 0);
    }

    #[test]
    fn a_new_request_cancels_the_rest_of_the_old_plan() {
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let decoded = Arc::new(AtomicUsize::new(0));
        let prefetcher = Prefetcher::new({
            let gate = Arc::clone(&gate);
            let decoded = Arc::clone(&decoded);
            Arc::new(move |_path: &Path, _long_side: u32| {
                decoded.fetch_add(1, Ordering::SeqCst);
                // Block the first item until the test releases it, so the
                // replacement request lands while the plan is mid-flight.
                let (lock, ready) = &*gate;
                let mut released = lock.lock().unwrap();
                while !*released {
                    released = ready.wait(released).unwrap();
                }
                true
            })
        });
        let first = prefetcher.request(vec![PathBuf::from("A.nef"), PathBuf::from("B.nef")], 512);
        while decoded.load(Ordering::SeqCst) == 0 {
            std::thread::yield_now();
        }
        let second = prefetcher.request(vec![PathBuf::from("C.nef")], 512);
        assert!(second > first);
        {
            let (lock, ready) = &*gate;
            *lock.lock().unwrap() = true;
            ready.notify_all();
        }
        assert!(prefetcher.wait_idle(), "queue must drain after release");
        // One in-flight item of the old plan, plus the new single-item plan.
        assert_eq!(decoded.load(Ordering::SeqCst), 2, "a stale plan stops after one item");
        let stats = prefetcher.stats();
        assert_eq!(stats.requested, 2);
        assert_eq!(stats.cancelled, 1, "{stats:?}");
    }

    #[test]
    fn a_burst_replaces_the_plan_before_any_decode() {
        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let prefetcher = Prefetcher::with_delay(
            {
                let seen = Arc::clone(&seen);
                Arc::new(move |path: &Path, _long_side: u32| {
                    seen.lock().unwrap().push(path.display().to_string());
                    true
                })
            },
            300,
        );
        prefetcher.request(vec![PathBuf::from("A.nef")], 512);
        std::thread::sleep(std::time::Duration::from_millis(25));
        prefetcher.request(vec![PathBuf::from("B.nef")], 512);
        assert!(prefetcher.wait_idle());
        assert_eq!(
            seen.lock().unwrap().clone(),
            vec!["B.nef"],
            "arrow-key bursts must not decode the plan they replaced"
        );
    }

    #[test]
    fn cancel_drops_a_pending_plan_without_decoding() {
        let decoded = Arc::new(AtomicUsize::new(0));
        let prefetcher = Prefetcher::new({
            let decoded = Arc::clone(&decoded);
            Arc::new(move |_path: &Path, _long_side: u32| {
                decoded.fetch_add(1, Ordering::SeqCst);
                true
            })
        });
        let before = prefetcher.generation();
        prefetcher.cancel();
        assert!(prefetcher.generation() > before, "cancel always bumps the generation");
        assert!(prefetcher.wait_idle());
        assert_eq!(decoded.load(Ordering::SeqCst), 0);
        assert_eq!(prefetcher.stats(), PrefetchStats::default());
    }

    #[test]
    fn repeated_and_failing_items_keep_the_counters_honest() {
        let decoded = Arc::new(AtomicUsize::new(0));
        let prefetcher = Prefetcher::new({
            let decoded = Arc::clone(&decoded);
            Arc::new(move |_path: &Path, _long_side: u32| {
                decoded.fetch_add(1, Ordering::SeqCst);
                false
            })
        });
        let plan = vec![PathBuf::from("only.nef")];
        prefetcher.request(plan.clone(), 256);
        prefetcher.request(plan, 256);
        assert!(prefetcher.wait_idle());
        let stats = prefetcher.stats();
        assert!((1..=2).contains(&stats.requested), "{stats:?}");
        assert_eq!(decoded.load(Ordering::SeqCst), stats.decoded as usize, "{stats:?}");
        assert_eq!(stats.failed, stats.decoded, "{stats:?}");
        assert_eq!(stats.cancelled, 0, "{stats:?}");
    }

    #[test]
    fn cache_warmer_is_the_shared_decode_cache() {
        // The production warmer must be the interactive decode path, not a
        // second cache: a synthetic non-RAW file decodes to `None` and is
        // therefore reported as not warm.
        let warmer = Prefetcher::cache_warmer();
        let dir = std::env::temp_dir().join(format!("aurora-prefetch-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("not-a-raw.bin");
        std::fs::write(&file, b"not a raw file").unwrap();
        assert!(!warmer(&file, 1024), "a non-RAW file is never warm");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// The RAW filter must run inside the offset walk: a JPEG neighbour may not
    /// consume a slot of the target cap, and the order must stay ahead-first.
    #[test]
    fn the_raw_filter_runs_inside_the_target_cap() {
        let paths: Vec<PathBuf> = ["A.NEF", "B.JPG", "C.NEF", "D.NEF", "E.JPG"]
            .iter()
            .map(PathBuf::from)
            .collect();
        // current = B.JPG: +1 -> C.NEF, -1 -> A.NEF, +2 -> D.NEF.
        assert_eq!(
            neighbor_paths(1, &paths, &DEFAULT_OFFSETS),
            vec![PathBuf::from("C.NEF"), PathBuf::from("A.NEF"), PathBuf::from("D.NEF")]
        );
        // A folder of JPEGs queues nothing at all: a second cache key would be
        // pure waste, and the interactive path never reads a RAW entry for them.
        let jpegs: Vec<PathBuf> = ["a.jpg", "b.jpg", "c.jpg"].iter().map(PathBuf::from).collect();
        assert!(neighbor_paths(1, &jpegs, &DEFAULT_OFFSETS).is_empty());
        let prefetcher = Prefetcher::new(Arc::new(|_: &Path, _: u32| true));
        prefetcher.request(neighbor_paths(1, &jpegs, &DEFAULT_OFFSETS), 4096);
        assert_eq!(prefetcher.stats(), PrefetchStats::default(), "an empty plan never runs");
    }

    /// The warmer must fill the entry the preview reads. Both sides use
    /// `raw_cache::PREVIEW_LONG_SIDE`, and this pins that contract.
    #[test]
    fn the_default_long_side_is_the_preview_source_size() {
        assert_eq!(
            crate::raw_cache::PREVIEW_LONG_SIDE,
            4096,
            "the preview source long side moved; the prefetch would warm a different cache key"
        );
    }

    /// A prefetch plan must not retain anything of its own: it warms the shared
    /// decode cache, and every retained frame is charged to the cache's own
    /// budget (`MAX_PENDING_NATIVE_BYTES`, and the disk budgets in
    /// `raw_cache::enforce_limits`). Warming four unreadable neighbours leaves
    /// the pending table exactly as it was.
    #[test]
    fn a_plan_retains_no_frames_of_its_own() {
        let dir = std::env::temp_dir().join(format!("aurora-prefetch-mem-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let plan: Vec<PathBuf> = (0..MAX_TARGETS)
            .map(|index| {
                let path = dir.join(format!("stub{index}.nef"));
                std::fs::write(&path, b"not a raw").unwrap();
                path
            })
            .collect();
        let prefetcher = Prefetcher::new(Prefetcher::cache_warmer());
        let before = crate::raw_cache::pending_native_stats();
        prefetcher.request(plan, crate::raw_cache::PREVIEW_LONG_SIDE);
        assert!(prefetcher.wait_idle());
        let after = crate::raw_cache::pending_native_stats();
        assert_eq!(before.entries, after.entries, "prefetch retained a frame");
        assert_eq!(before.bytes, after.bytes, "prefetch retained bytes");
        let stats = prefetcher.stats();
        assert_eq!(stats.decoded, MAX_TARGETS as u64, "{stats:?}");
        assert_eq!(stats.failed, MAX_TARGETS as u64, "unreadable neighbours report failure: {stats:?}");
        let _ = std::fs::remove_dir_all(dir);
    }

    /// `is_source16_warm` must be side-effect free: asking twice must not decode,
    /// create a cache entry or consume the retained frame.
    #[test]
    fn warm_check_does_not_decode_or_create_entries() {
        let dir = std::env::temp_dir().join(format!("aurora-warm-check-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("not-a-raw.bin");
        std::fs::write(&file, b"definitely not a raw").unwrap();
        let before = crate::raw_cache::pending_native_stats();
        assert!(!crate::raw_cache::is_source16_warm(&file, 1024));
        assert!(!crate::raw_cache::is_source16_warm(&file, 0));
        let after = crate::raw_cache::pending_native_stats();
        assert_eq!(before.decodes, after.decodes, "the warm check must not decode");
        assert_eq!(before.entries, after.entries);
        let _ = std::fs::remove_dir_all(dir);
    }
}
