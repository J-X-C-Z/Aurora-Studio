//! Shared CUDA runtime for the mock GPU stages.
//!
//! One place owns what `nr_gpu`, `clarity_gpu`, `detail_gpu` and `pixel_gpu`
//! currently each duplicate: the module handle, the optional non-blocking
//! stream, grow-only device slots, the pinned host pool, kernel dispatch,
//! transfer accounting, failure reasons and device-loss bookkeeping.
//!
//! The driver wrapper in `aurora_portrait::cuda` already keeps one process
//! context and one JIT-loaded module per PTX image; this layer never loads a
//! PTX image twice and reports `ptx_module_load_calls` so that stays provable.
//!
//! Every method returns [`Failure`] instead of a bare `CudaError`; callers
//! return `false`/`None` to their CPU path and record *why* through
//! [`note_fallback`], so a CPU fallback is never counted as a GPU success.

use std::ffi::c_void;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use aurora_portrait::Refusal;
use aurora_portrait::cuda::{CudaError, DeviceBuffer, Nvidia, Stream};

/// Grow-only device slots shared by the four stages. `Frame` holds the pixel
/// buffer, `Scratch` the planes/temporaries, `Aux*` the small LUT/node/stats
/// buffers. A slot only ever grows; a smaller request reuses the allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Slot {
    Frame,
    Scratch,
    Aux0,
    Aux1,
    Aux2,
}

impl Slot {
    const ALL: [Slot; 5] = [Slot::Frame, Slot::Scratch, Slot::Aux0, Slot::Aux1, Slot::Aux2];

    fn index(self) -> usize {
        match self {
            Slot::Frame => 0,
            Slot::Scratch => 1,
            Slot::Aux0 => 2,
            Slot::Aux1 => 3,
            Slot::Aux2 => 4,
        }
    }
}

/// Why a stage could not run on the device. Every variant is reportable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Failure {
    /// The caller switched the stage off (`AURORA_*_GPU=0`, test default).
    Disabled(&'static str),
    /// The frame needs more device memory than the stage keeps resident.
    TooLarge { need: usize, cap: usize },
    /// No driver, no device, or a missing optional entry point.
    Unavailable(&'static str),
    /// A driver call returned a non-zero `CUresult`.
    Driver { call: &'static str, code: i32 },
    /// The wrapper refused the request before touching the driver.
    Refused(&'static str),
    /// A driver error the runtime classifies as a lost/broken device.
    DeviceLost(i32),
    /// The request is structurally invalid for this stage (size, layout).
    Invalid(&'static str),
}

impl Failure {
    /// Short stable tag for counters and log lines.
    pub(crate) fn tag(&self) -> &'static str {
        match self {
            Self::Disabled(_) => "disabled",
            Self::TooLarge { .. } => "too_large",
            Self::Unavailable(_) => "unavailable",
            Self::Driver { .. } => "driver",
            Self::Refused(_) => "refused",
            Self::DeviceLost(_) => "device_lost",
            Self::Invalid(_) => "invalid",
        }
    }
}

impl From<CudaError> for Failure {
    fn from(error: CudaError) -> Self {
        match error {
            CudaError::Unavailable(what) => Self::Unavailable(what),
            CudaError::Driver(call, code) => Self::Driver { call, code },
            CudaError::Refused(refusal) => Self::Refused(refusal_tag(&refusal)),
        }
    }
}

fn refusal_tag(refusal: &Refusal) -> &'static str {
    match refusal {
        Refusal::UnknownEffect(_) => "unknown_effect",
        Refusal::MissingInput(_) => "missing_input",
        Refusal::ShapeMismatch(_) => "shape_mismatch",
        Refusal::ValueOutOfRange(_) => "value_out_of_range",
        Refusal::NotImplemented(_) => "not_implemented",
    }
}

/// Driver codes a lost/reset/illegal-access device returns.
///
/// 700 illegal address, 702 launch failure, 709 context destroyed,
/// 710 context destroyed by reset, 719 unsupported launch, 999 unknown.
pub(crate) fn is_device_lost(code: i32) -> bool {
    matches!(code, 700 | 702 | 709 | 710 | 719 | 999)
}

static LOST_CODE: AtomicI32 = AtomicI32::new(0);
static RESETS: AtomicU64 = AtomicU64::new(0);

/// Non-zero once any stage has seen a device-loss driver code in this process.
pub(crate) fn lost_code() -> Option<i32> {
    let code = LOST_CODE.load(Ordering::Relaxed);
    (code != 0).then_some(code)
}

fn mark_lost(code: i32) {
    if is_device_lost(code) {
        LOST_CODE.store(code, Ordering::Relaxed);
    }
}

/// Per-stage fallback counters, readable from tests and the session report.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct FallbackReport {
    pub disabled: u64,
    pub too_large: u64,
    pub unavailable: u64,
    pub driver: u64,
    pub refused: u64,
    pub device_lost: u64,
    pub invalid: u64,
    /// Last reason per stage tag (`nr`, `clarity`, `detail`, `pixel`).
    pub last: Vec<(&'static str, &'static str)>,
    pub resets: u64,
}

fn fallback_report() -> &'static Mutex<FallbackReport> {
    static REPORT: OnceLock<Mutex<FallbackReport>> = OnceLock::new();
    REPORT.get_or_init(|| Mutex::new(FallbackReport::default()))
}

/// Records one CPU fallback so "GPU ran" and "GPU declined" stay separable.
pub(crate) fn note_fallback(stage: &'static str, failure: &Failure) {
    let mut report = fallback_report().lock().unwrap_or_else(|error| error.into_inner());
    match failure {
        Failure::Disabled(_) => report.disabled += 1,
        Failure::TooLarge { .. } => report.too_large += 1,
        Failure::Unavailable(_) => report.unavailable += 1,
        Failure::Driver { .. } => report.driver += 1,
        Failure::Refused(_) => report.refused += 1,
        Failure::DeviceLost(_) => report.device_lost += 1,
        Failure::Invalid(_) => report.invalid += 1,
    }
    let tag = failure.tag();
    match report.last.iter_mut().find(|(name, _)| *name == stage) {
        Some(entry) => entry.1 = tag,
        None => report.last.push((stage, tag)),
    }
}

/// Snapshot of the process-wide fallback counters plus reset count.
pub(crate) fn fallback_snapshot() -> FallbackReport {
    let mut report = fallback_report().lock().unwrap_or_else(|error| error.into_inner()).clone();
    report.resets = RESETS.load(Ordering::Relaxed);
    report
}

/// Test hook: clears counters and the lost-device flag.
#[cfg(test)]
pub(crate) fn reset_fallback() {
    let mut report = fallback_report().lock().unwrap_or_else(|error| error.into_inner());
    *report = FallbackReport::default();
    LOST_CODE.store(0, Ordering::Relaxed);
}

// ─────────────────────────────── dispatch ─────────────────────────────────

pub(crate) fn ptr<T>(value: &mut T) -> *mut c_void {
    value as *mut T as *mut c_void
}

/// One kernel dispatch record; `count` collapses repeated launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Launch {
    pub kernel: &'static str,
    pub grid: (u32, u32),
    pub block: (u32, u32),
    pub count: u64,
}

/// Device transfer and dispatch accounting for one `Gpu`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Counters {
    pub uploads: u64,
    pub downloads: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub launches: u64,
    pub syncs: u64,
    /// Frames that ran through a device-resident [`FrameHandle`].
    pub fused_frames: u64,
    /// Transfers a resident buffer made unnecessary (grid/plane reuse).
    pub skipped_uploads: u64,
    pub skipped_downloads: u64,
    /// Plane rebuilds skipped because the resident grid was byte-identical.
    pub skipped_plane_builds: u64,
}

struct SendStream(Stream<'static>);

// `Stream` only borrows the driver table, which the boxed module handle keeps
// alive for `Gpu`'s whole lifetime, and `Gpu` is only reached behind a lock.
unsafe impl Send for SendStream {}

// ───────────────────────────── VRAM budget ledger ─────────────────────────

/// Engine-wide device budget when `AURORA_VRAM_BUDGET_MB` is unset.
///
/// Deliberately a constant: probing `driver_memory_info()` here would create the
/// CUDA context on whichever thread first calls the non-allocating admission
/// check (normally the render thread), which the preflight path must not do.
/// A free-VRAM-aware budget is a follow-up once the context lifetime switches
/// to the idle prewarm thread.
pub(crate) const DEFAULT_VRAM_BUDGET_MB: usize = 2560;

/// One process-wide device budget shared by every engine GPU stage.
///
/// Before S7 each stage kept its own `MAX_RETAINED_BYTES` (768 MiB for clarity,
/// detail and pixel, 1536 MiB for NR) and nothing bounded their sum. This
/// ledger is now the authority: every device allocation a stage keeps resident
/// is a [`VramLease`] charged here, so `resident_bytes + need <= budget` is
/// decided once, not per stage. `AURORA_VRAM_BUDGET_MB` overrides the default
/// (positive integer MiB).
pub(crate) fn vram_budget() -> usize {
    static BUDGET: OnceLock<usize> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("AURORA_VRAM_BUDGET_MB")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|mb| *mb > 0)
            .map_or(DEFAULT_VRAM_BUDGET_MB * 1024 * 1024, |mb| {
                mb.saturating_mul(1024 * 1024)
            })
    })
}

/// Snapshot of the unified device ledger, read by tests and `AURORA_PERF`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct VramStats {
    pub budget_bytes: usize,
    pub resident_bytes: usize,
    pub peak_bytes: usize,
    pub eviction_count: u64,
    pub evicted_bytes: u64,
    pub refused_count: u64,
}

impl VramStats {
    /// One `AURORA_PERF` JSON object; field names are stable for replay parsers.
    pub(crate) fn to_perf_json(&self, stage: &'static str) -> String {
        format!(
            "{{\"event\":\"gpu_vram\",\"stage\":\"{stage}\",\"budget_bytes\":{},\"resident_bytes\":{},\
\"peak_bytes\":{},\"eviction_count\":{},\"evicted_bytes\":{},\"refused_count\":{}}}",
            self.budget_bytes,
            self.resident_bytes,
            self.peak_bytes,
            self.eviction_count,
            self.evicted_bytes,
            self.refused_count,
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct LedgerEntry {
    id: u64,
    stage: &'static str,
    bytes: usize,
    evictable: bool,
    last_used: u64,
}

/// Pure bookkeeping for the shared budget. Kept free of globals so the
/// admission/eviction arithmetic is unit-testable without a CUDA device.
#[derive(Debug)]
pub(crate) struct VramLedger {
    budget: usize,
    resident: usize,
    peak: usize,
    evictions: u64,
    evicted_bytes: u64,
    refusals: u64,
    clock: u64,
    entries: Vec<LedgerEntry>,
}

impl VramLedger {
    pub(crate) fn new(budget: usize) -> Self {
        Self {
            budget,
            resident: 0,
            peak: 0,
            evictions: 0,
            evicted_bytes: 0,
            refusals: 0,
            clock: 0,
            entries: Vec::new(),
        }
    }

    /// Sets the bytes one lease holds. `false` means the budget is exhausted,
    /// in which case nothing was recorded.
    pub(crate) fn admit(&mut self, id: u64, stage: &'static str, bytes: usize, evictable: bool) -> bool {
        self.clock += 1;
        let clock = self.clock;
        if let Some(index) = self.entries.iter().position(|entry| entry.id == id) {
            let old = self.entries[index].bytes;
            if bytes > old {
                let delta = bytes - old;
                if self.resident + delta > self.budget {
                    return false;
                }
                self.resident += delta;
                self.peak = self.peak.max(self.resident);
            } else {
                self.resident -= old - bytes;
            }
            let entry = &mut self.entries[index];
            entry.bytes = bytes;
            entry.evictable = evictable;
            entry.last_used = clock;
            return true;
        }
        if self.resident + bytes > self.budget {
            return false;
        }
        self.resident += bytes;
        self.peak = self.peak.max(self.resident);
        self.entries.push(LedgerEntry {
            id,
            stage,
            bytes,
            evictable,
            last_used: clock,
        });
        true
    }

    pub(crate) fn release(&mut self, id: u64) {
        if let Some(index) = self.entries.iter().position(|entry| entry.id == id) {
            let entry = self.entries.remove(index);
            self.resident = self.resident.saturating_sub(entry.bytes);
        }
    }

    pub(crate) fn touch(&mut self, id: u64) {
        self.clock += 1;
        if let Some(entry) = self.entries.iter_mut().find(|entry| entry.id == id) {
            entry.last_used = self.clock;
        }
    }

    /// Re-labels an existing lease without touching `resident`/`peak`. Used by
    /// a cross-stage handoff so the bytes are charged once, to the new owner.
    /// `false` when the id is no longer resident (already released/reclaimed).
    pub(crate) fn reassign(&mut self, id: u64, stage: &'static str) -> bool {
        self.clock += 1;
        let clock = self.clock;
        match self.entries.iter_mut().find(|entry| entry.id == id) {
            Some(entry) => {
                entry.stage = stage;
                entry.last_used = clock;
                true
            }
            None => false,
        }
    }

    pub(crate) fn stage_bytes(&self, stage: &'static str) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.stage == stage)
            .map(|entry| entry.bytes)
            .sum()
    }

    /// Bytes `requester` can actually get back: evictable leases of other
    /// stages that registered a reclaim hook. An `evictable` flag alone is not
    /// enough, because a stage without a hook cannot free its leases, so
    /// crediting them would only turn a later refusal into a silent CPU
    /// fallback.
    fn reclaimable_bytes(&self, requester: &'static str, hooked: &[&'static str]) -> usize {
        self.entries
            .iter()
            .filter(|entry| {
                entry.evictable && entry.stage != requester && hooked.contains(&entry.stage)
            })
            .map(|entry| entry.bytes)
            .sum()
    }

    /// Would a stage that needs `bytes` resident fit, assuming rebuildable
    /// buffers of other stages (only those with a reclaim hook) can be
    /// reclaimed first? Non-allocating.
    pub(crate) fn can_allocate(
        &self,
        stage: &'static str,
        bytes: usize,
        hooked: &[&'static str],
    ) -> bool {
        let own = self.stage_bytes(stage);
        let base = self.resident.saturating_sub(own);
        if base + bytes <= self.budget {
            return true;
        }
        let reclaimable = self.reclaimable_bytes(stage, hooked);
        base.saturating_sub(reclaimable) + bytes <= self.budget
    }

    /// Stages to reclaim, least priority first and oldest use first inside a
    /// priority band. Only stages that hold evictable bytes appear.
    pub(crate) fn victim_order(
        &self,
        requester: &'static str,
        priorities: &[(&'static str, u8)],
    ) -> Vec<&'static str> {
        let mut victims: Vec<(&'static str, u8, u64)> = Vec::new();
        for (stage, priority) in priorities {
            if *stage == requester {
                continue;
            }
            let oldest = self
                .entries
                .iter()
                .filter(|entry| entry.stage == *stage && entry.evictable && entry.bytes > 0)
                .map(|entry| entry.last_used)
                .min();
            if let Some(oldest) = oldest {
                victims.push((stage, *priority, oldest));
            }
        }
        victims.sort_by_key(|(_, priority, oldest)| (*priority, *oldest));
        victims.into_iter().map(|(stage, _, _)| stage).collect()
    }

    pub(crate) fn note_eviction(&mut self, bytes: usize) {
        self.evictions += 1;
        self.evicted_bytes += bytes as u64;
    }

    pub(crate) fn note_refusal(&mut self) {
        self.refusals += 1;
    }

    pub(crate) fn stats(&self) -> VramStats {
        VramStats {
            budget_bytes: self.budget,
            resident_bytes: self.resident,
            peak_bytes: self.peak,
            eviction_count: self.evictions,
            evicted_bytes: self.evicted_bytes,
            refused_count: self.refusals,
        }
    }
}

fn ledger() -> &'static Mutex<VramLedger> {
    static LEDGER: OnceLock<Mutex<VramLedger>> = OnceLock::new();
    LEDGER.get_or_init(|| Mutex::new(VramLedger::new(vram_budget())))
}

fn with_ledger<T>(body: impl FnOnce(&mut VramLedger) -> T) -> T {
    let mut ledger = ledger().lock().unwrap_or_else(|error| error.into_inner());
    body(&mut ledger)
}

/// Reclaim hooks, one per stage that can drop its retained buffers. Priority
/// orders eviction (`develop` is the interactive main path and goes last); the
/// hook must use `try_lock` on its own stage so eviction never blocks.
type ReclaimHook = fn() -> usize;

fn reclaimers() -> &'static Mutex<Vec<(&'static str, u8, ReclaimHook)>> {
    static HOOKS: OnceLock<Mutex<Vec<(&'static str, u8, ReclaimHook)>>> = OnceLock::new();
    HOOKS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Registers (or replaces) one stage's reclaim hook. Called lazily from the
/// stage entry points, so a process that never touches a stage never lists it.
pub(crate) fn vram_register_reclaimer(stage: &'static str, priority: u8, hook: ReclaimHook) {
    let mut hooks = reclaimers().lock().unwrap_or_else(|error| error.into_inner());
    match hooks.iter_mut().find(|(name, _, _)| *name == stage) {
        Some(entry) => {
            entry.1 = priority;
            entry.2 = hook;
        }
        None => hooks.push((stage, priority, hook)),
    }
}

fn reclaimer_snapshot() -> Vec<(&'static str, u8, ReclaimHook)> {
    reclaimers()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone()
}

/// Stages with a registered reclaim hook, in registration order.
fn reclaimer_stages() -> Vec<&'static str> {
    reclaimers()
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .iter()
        .map(|(stage, _, _)| *stage)
        .collect()
}

/// Non-allocating admission check: would `stage` fit `bytes` resident, counting
/// buffers other stages with a reclaim hook could rebuild? Never evicts and
/// never allocates.
pub(crate) fn vram_can_allocate(stage: &'static str, bytes: usize) -> bool {
    let hooked = reclaimer_stages();
    with_ledger(|ledger| ledger.can_allocate(stage, bytes, &hooked))
}

/// Records a lease size, reclaiming other stages' rebuildable buffers when the
/// budget is short. Reclaim hooks run without the ledger lock held, so they may
/// drop their own leases (and re-enter this ledger) safely.
pub(crate) fn vram_set_lease(
    id: u64,
    stage: &'static str,
    bytes: usize,
    evictable: bool,
) -> Result<(), Failure> {
    if with_ledger(|ledger| ledger.admit(id, stage, bytes, evictable)) {
        return Ok(());
    }
    let hooks = reclaimer_snapshot();
    for _ in 0..4 {
        let priorities: Vec<(&'static str, u8)> =
            hooks.iter().map(|(name, priority, _)| (*name, *priority)).collect();
        let victims = with_ledger(|ledger| ledger.victim_order(stage, &priorities));
        if victims.is_empty() {
            break;
        }
        let mut freed = 0usize;
        for victim in victims {
            if let Some((_, _, hook)) = hooks.iter().find(|(name, _, _)| *name == victim) {
                // Runs the stage's own `try_lock` reclaim; never blocks.
                let before = freed;
                freed += hook();
                if freed > before {
                    with_ledger(|ledger| ledger.note_eviction(freed - before));
                }
            }
            if with_ledger(|ledger| ledger.admit(id, stage, bytes, evictable)) {
                return Ok(());
            }
        }
        if freed == 0 {
            break;
        }
    }
    with_ledger(|ledger| ledger.note_refusal());
    Err(Failure::TooLarge {
        need: bytes,
        cap: vram_budget(),
    })
}

pub(crate) fn vram_release(id: u64) {
    with_ledger(|ledger| ledger.release(id));
}

fn vram_touch(id: u64) {
    with_ledger(|ledger| ledger.touch(id));
}

fn vram_reassign(id: u64, stage: &'static str) -> bool {
    with_ledger(|ledger| ledger.reassign(id, stage))
}

/// Real residency of the shared ledger: `resident_bytes`, `peak_bytes`,
/// `eviction_count`, `refused_count` plus the configured budget.
pub(crate) fn vram_stats() -> VramStats {
    with_ledger(|ledger| ledger.stats())
}

/// Emits one `AURORA_PERF` residency line, throttled to at most one per second
/// and only when the replay/profiling switch is on. S1/IN-P can consume the
/// `gpu_vram` event without changing `crates/app/src/perf.rs`.
pub(crate) fn vram_maybe_emit_perf(stage: &'static str) {
    if std::env::var_os("AURORA_PERF_REPLAY").is_none() {
        return;
    }
    static LAST: OnceLock<Mutex<Option<std::time::Instant>>> = OnceLock::new();
    let mut last = LAST
        .get_or_init(|| Mutex::new(None))
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let now = std::time::Instant::now();
    if last.is_some_and(|previous| now.duration_since(previous) < std::time::Duration::from_secs(1)) {
        return;
    }
    *last = Some(now);
    eprintln!("AURORA_PERF {}", vram_stats().to_perf_json(stage));
}

static NEXT_LEASE: AtomicU64 = AtomicU64::new(1);

/// One reservation in the shared ledger. Dropping it gives the bytes back, so
/// the ledger can never drift from what the owners actually keep resident.
pub(crate) struct VramLease {
    id: u64,
    stage: &'static str,
    bytes: usize,
}

impl VramLease {
    pub(crate) fn acquire(
        stage: &'static str,
        bytes: usize,
        evictable: bool,
    ) -> Result<Self, Failure> {
        let id = NEXT_LEASE.fetch_add(1, Ordering::Relaxed);
        vram_set_lease(id, stage, bytes, evictable)?;
        Ok(Self { id, stage, bytes })
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn stage(&self) -> &'static str {
        self.stage
    }

    /// Moves the reservation to another stage's ledger share for a cross-stage
    /// handoff. The id and byte count stay the same, so this neither releases
    /// the bytes (no early return) nor admits them again (no double charge):
    /// exactly one ledger entry of the same size changes owner.
    pub(crate) fn reassign(&mut self, stage: &'static str) -> Result<(), Failure> {
        if self.bytes == 0 || !vram_reassign(self.id, stage) {
            return Err(Failure::Invalid("lease not resident"));
        }
        self.stage = stage;
        Ok(())
    }

    /// Grows (or shrinks) the reservation. A refusal leaves the old size.
    pub(crate) fn resize(&mut self, bytes: usize, evictable: bool) -> Result<(), Failure> {
        if bytes == self.bytes {
            vram_touch(self.id);
            return Ok(());
        }
        vram_set_lease(self.id, self.stage, bytes, evictable)?;
        self.bytes = bytes;
        Ok(())
    }

    /// Gives the bytes back and stops reporting them (used when the owning
    /// device buffer is dropped before the lease).
    pub(crate) fn forget(&mut self) {
        if self.bytes != 0 {
            vram_release(self.id);
            self.bytes = 0;
        }
    }
}

impl Drop for VramLease {
    fn drop(&mut self) {
        self.forget();
    }
}

/// Process-wide, strictly increasing residency epoch. Starts at 1 so `0` never
/// means "current" and a descriptor that skipped allocation is distinguishable.
static NEXT_FRAME_GENERATION: AtomicU64 = AtomicU64::new(1);

fn next_frame_generation() -> u64 {
    NEXT_FRAME_GENERATION.fetch_add(1, Ordering::Relaxed)
}

/// Process context token. The driver wrapper owns exactly one context per
/// process (`cuda.rs` `DOMAIN`), so this is an engine-side identity, not the
/// driver's `CUcontext` pointer (that crate is not owned by this lane).
static NEXT_CONTEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

fn mint_context_token() -> u64 {
    NEXT_CONTEXT_TOKEN.fetch_add(1, Ordering::Relaxed)
}

fn process_context_token() -> u64 {
    static TOKEN: OnceLock<u64> = OnceLock::new();
    *TOKEN.get_or_init(mint_context_token)
}

/// Where a frame's pixels are authoritative right now.
#[allow(dead_code)] // M1 descriptor: consumed by later cross-stage fusion steps
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Residency {
    /// Host memory only; no device allocation is charged for this frame.
    Host,
    /// One device allocation owned by the frame; the lease pays for it.
    #[default]
    Device,
    /// Device allocation mirrored in pinned host memory (staging frames).
    Shared,
}

/// Transfer function / gamut of the stored samples.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum FrameColorSpace {
    /// Scene-linear RGB; the develop tone chain's input.
    #[default]
    LinearRgb,
    /// Display-referred sRGB-encoded RGB.
    Srgb,
    /// Non-color planes (grid, luma, statistics).
    Data,
}

/// Bits per channel. Kept independent of [`FrameFormat`] because a 10-bit
/// source can ride in an f32 frame and an 8-bit display frame is not an f32
/// frame in lower precision.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum FrameBitDepth {
    U8,
    U16,
    #[default]
    F32,
}

/// Pixel region of a frame plus the halo its consumer may read. An all-zero
/// width/height means "the whole frame", so descriptors built before the ROI
/// is known keep working.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FrameRegion {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    /// Extra pixels on every side of the region a kernel reads.
    pub halo: u32,
}

impl FrameRegion {
    pub(crate) fn is_unspecified(&self) -> bool {
        self.width == 0 || self.height == 0
    }

    /// The region actually addressed, halo included: an unspecified ROI becomes
    /// the full `width` x `height` frame.
    pub(crate) fn addressed(&self, width: u32, height: u32) -> Self {
        if self.is_unspecified() {
            Self { x: 0, y: 0, width, height, halo: self.halo }
        } else {
            *self
        }
    }

    /// Whether `self` lies inside a `width` x `height` frame.
    pub(crate) fn fits(&self, width: u32, height: u32) -> bool {
        self.is_unspecified()
            || (self.x.saturating_add(self.width) <= width
                && self.y.saturating_add(self.height) <= height)
    }
}

/// Device/context identity an allocation is stamped with.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FrameIdentity {
    /// The wrapper always opens device ordinal 0 (`cuDeviceGet(&mut device, 0)`).
    pub device_id: u32,
    /// Engine-side process context token (see [`process_context_token`]).
    pub context_id: u64,
}

/// Host-side layout of a device-resident frame. `bytes` is explicit because a
/// plane cache is not `w * h * bpp`.
///
/// Everything after `generation` (inclusive) has a `Default` meaning that
/// preserves the pre-M1 behavior: `FrameHandle::alloc` stamps the generation,
/// device/context identity and `Residency::Device`, so a caller that only fills
/// `width/height/format/bytes` keeps allocating exactly what it did before.
#[allow(dead_code)] // the descriptor is populated ahead of its consumers
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FrameLayout {
    pub width: u32,
    pub height: u32,
    pub format: FrameFormat,
    pub bytes: usize,
    /// Bumped when the owner invalidates the residency (resize, device loss) or
    /// hands the frame to another stage. `0` means "not stamped yet"; every
    /// allocation receives a strictly larger value from a process counter.
    pub generation: u64,
    pub residency: Residency,
    /// Device ordinal and context token; `None` until an allocation stamps them.
    pub device_id: Option<u32>,
    pub context_id: Option<u64>,
    /// Row pitch in bytes; `0` means tightly packed (`width * bytes_per_px`).
    pub stride: usize,
    /// Sub-window this frame covers; unspecified means the whole frame.
    pub region: FrameRegion,
    pub colorspace: FrameColorSpace,
    pub bit_depth: FrameBitDepth,
}

impl Default for FrameLayout {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            format: FrameFormat::F32Rgb12,
            bytes: 0,
            generation: 0,
            residency: Residency::Device,
            device_id: None,
            context_id: None,
            stride: 0,
            region: FrameRegion::default(),
            colorspace: FrameColorSpace::LinearRgb,
            bit_depth: FrameBitDepth::F32,
        }
    }
}

#[allow(dead_code)] // `new` enters production with the first u8/u16 staging frame
impl FrameLayout {
    /// A layout whose derived fields agree with `format`; new callers that use
    /// `U8Rgba8`/`U16Rgba16` should start here instead of listing fields.
    pub(crate) fn new(width: u32, height: u32, format: FrameFormat, bytes: usize) -> Self {
        Self {
            width,
            height,
            format,
            bytes,
            colorspace: format.default_colorspace(),
            bit_depth: format.bit_depth(),
            ..Self::default()
        }
    }

    /// Bytes from one row to the next, or `None` for a composite plane cache.
    pub(crate) fn row_bytes(&self) -> Option<usize> {
        self.format.bytes_per_px().map(|bpp| {
            if self.stride != 0 {
                self.stride
            } else {
                self.width as usize * bpp
            }
        })
    }

    /// Bytes the declared region (halo included) needs, or `None` for a
    /// composite plane cache whose `bytes` is already authoritative.
    pub(crate) fn required_bytes(&self) -> Option<usize> {
        let row = self.row_bytes()?;
        let region = self.region.addressed(self.width, self.height);
        row.checked_mul(region.height as usize + 2 * region.halo as usize)
    }

    /// Stamps what only the allocating stage knows: a fresh generation, the
    /// device/context identity and device residency. Rejects descriptors that
    /// cannot describe one device allocation. Pure arithmetic.
    pub(crate) fn prepare_for_device(
        mut self,
        identity: FrameIdentity,
    ) -> Result<Self, Failure> {
        if self.residency == Residency::Host {
            return Err(Failure::Invalid("host residency has no device frame"));
        }
        if self.bit_depth != self.format.bit_depth() {
            return Err(Failure::Invalid("bit depth does not match format"));
        }
        if !self.region.fits(self.width, self.height) {
            return Err(Failure::Invalid("region exceeds frame"));
        }
        if let (Some(required), Some(row), Some(bpp)) =
            (self.required_bytes(), self.row_bytes(), self.format.bytes_per_px())
        {
            if self.stride != 0 && self.stride < self.width as usize * bpp {
                return Err(Failure::Invalid("stride below one tight row"));
            }
            if self.bytes < required || row == 0 {
                return Err(Failure::Invalid("frame bytes below stride * rows"));
            }
        }
        self.generation = next_frame_generation();
        self.device_id = Some(identity.device_id);
        self.context_id = Some(identity.context_id);
        Ok(self)
    }

    /// Advances `generation` for an owner that invalidated the residency in
    /// place (resize, device loss, content rewrite).
    pub(crate) fn bump_generation(&mut self) -> u64 {
        self.generation = next_frame_generation();
        self.generation
    }
}

#[allow(dead_code)] // u8/u16 variants are the frame-side half of display/export staging
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum FrameFormat {
    /// `Vec<[f32; 3]>`: 12 bytes per pixel, the pre-curve display frame.
    #[default]
    F32Rgb12,
    /// Grid plus the two low-frequency planes built from it.
    F32Planes,
    /// One `u8` per channel, four channels: display/export staging frames.
    U8Rgba8,
    /// One `u16` per channel, four channels: 16-bit export staging frames.
    U16Rgba16,
}

impl FrameFormat {
    /// Bytes one pixel occupies when tightly packed. `None` for the composite
    /// plane cache, whose `FrameLayout::bytes` is not `w * h * bpp`.
    pub(crate) fn bytes_per_px(self) -> Option<usize> {
        match self {
            Self::F32Rgb12 => Some(12),
            Self::F32Planes => None,
            Self::U8Rgba8 => Some(4),
            Self::U16Rgba16 => Some(8),
        }
    }

    pub(crate) fn bit_depth(self) -> FrameBitDepth {
        match self {
            Self::F32Rgb12 | Self::F32Planes => FrameBitDepth::F32,
            Self::U8Rgba8 => FrameBitDepth::U8,
            Self::U16Rgba16 => FrameBitDepth::U16,
        }
    }

    pub(crate) fn default_colorspace(self) -> FrameColorSpace {
        match self {
            Self::F32Rgb12 => FrameColorSpace::LinearRgb,
            Self::F32Planes => FrameColorSpace::Data,
            Self::U8Rgba8 | Self::U16Rgba16 => FrameColorSpace::Srgb,
        }
    }
}

/// A device allocation paired with the ledger lease that pays for it.
///
/// Field order is load-bearing: Rust drops fields in declaration order, so the
/// buffer is freed first and only then does the lease hand its bytes back to
/// the ledger. With the lease first, another thread could pass
/// `can_allocate`/`admit` in the window before `cuMemFree` finished and its
/// `cuMemAlloc` would fail even though the preflight had said yes.
struct LeasedBuffer<B> {
    buffer: B,
    lease: VramLease,
}

/// Device-resident frame shared across the passes of one stage. It is the
/// P-07 step-2 building block: allocation and residency are owned by an object
/// with a layout and a generation instead of an ad-hoc slot offset, so a later
/// cross-stage fusion can hand the same device pointer to the next kernel
/// through [`FrameHandle::hand_off`].
///
/// Ownership bookkeeping of one resident frame, split from the device buffer so
/// the handoff rules are unit-testable with a ledger-only lease (no CUDA).
#[allow(dead_code)] // handoff API lands before its first cross-stage consumer
struct FrameResidency {
    /// Stage the lease currently belongs to (the allocator, or the last stage
    /// the frame was handed to).
    owner: &'static str,
    /// Device ordinal and context token the buffer was allocated under.
    identity: FrameIdentity,
    /// Stream generation the frame was last written on, `None` when the writes
    /// used synchronous copies. A consumer on another stage must sync the
    /// producer before reading; this records what to wait for.
    last_write_sync: Option<u64>,
}

impl FrameResidency {
    fn new(owner: &'static str, identity: FrameIdentity) -> Self {
        Self { owner, identity, last_write_sync: None }
    }

    /// Moves the frame to `next_stage`. The lease keeps its id and byte count,
    /// so the ledger total never moves (no double charge) and no release runs
    /// before the receiving stage owns it (no early return). Only the entry's
    /// stage label changes. The layout generation advances so a stale consumer
    /// cannot mistake the frame for the version it read before the handoff.
    fn hand_off(
        &mut self,
        lease: &mut VramLease,
        layout: &mut FrameLayout,
        next_stage: &'static str,
    ) -> Result<(), Failure> {
        if next_stage == self.owner {
            return Ok(());
        }
        lease.reassign(next_stage)?;
        self.owner = next_stage;
        self.last_write_sync = None;
        layout.bump_generation();
        Ok(())
    }
}

/// Drop order is load-bearing: `buffer` is declared before `lease` so `cuMemFree`
/// runs before the ledger marks the bytes spendable again.
#[allow(dead_code)] // new descriptor/handoff accessors land before their consumers
pub(crate) struct FrameHandle {
    layout: FrameLayout,
    residency: FrameResidency,
    buffer: DeviceBuffer<'static>,
    lease: VramLease,
}

// The buffer borrows the process driver table, kept alive by the owning `Gpu`,
// and every use happens behind the stage's `Mutex<State>`.
unsafe impl Send for FrameHandle {}

#[allow(dead_code)] // new descriptor/handoff accessors land before their consumers
impl FrameHandle {
    pub(crate) fn alloc(gpu: &Gpu, layout: FrameLayout) -> Result<Self, Failure> {
        let identity = FrameIdentity {
            device_id: 0,
            context_id: process_context_token(),
        };
        // Stamps generation/identity and validates the derived fields. The
        // pre-M1 descriptors (plain struct literal, generation: 0) carry no
        // stride/region/bit-depth constraints, so this cannot fail for them.
        let layout = layout.prepare_for_device(identity)?;
        let lease = VramLease::acquire(gpu.stage, layout.bytes.max(1), true)?;
        match gpu.gpu.alloc_buffer(layout.bytes.max(1)) {
            Ok(buffer) => Ok(Self {
                layout,
                residency: FrameResidency::new(gpu.stage, identity),
                buffer: unsafe {
                    std::mem::transmute::<DeviceBuffer<'_>, DeviceBuffer<'static>>(buffer)
                },
                lease,
            }),
            Err(error) => Err(gpu.failed(error)),
        }
    }

    pub(crate) fn layout(&self) -> FrameLayout {
        self.layout
    }

    /// Stage the residency is currently charged to.
    pub(crate) fn owner(&self) -> &'static str {
        self.residency.owner
    }

    /// Device ordinal and context token the buffer was allocated under.
    pub(crate) fn identity(&self) -> FrameIdentity {
        self.residency.identity
    }

    /// Stream generation of the last write, when it went through a stream.
    pub(crate) fn last_write_sync(&self) -> Option<u64> {
        self.residency.last_write_sync
    }

    /// Records that the frame was written through `sync_generation` on the
    /// owning stage's stream. Consumers use it as the handoff dependency.
    pub(crate) fn note_write_sync(&mut self, sync_generation: u64) {
        self.residency.last_write_sync = Some(sync_generation);
        self.layout.bump_generation();
    }

    pub(crate) fn handle(&self) -> u64 {
        self.buffer.handle()
    }

    pub(crate) fn bytes(&self) -> usize {
        self.layout.bytes
    }

    pub(crate) fn lease_bytes(&self) -> usize {
        self.lease.bytes()
    }

    /// Cross-stage handoff: consumes the handle and returns it renamed for
    /// `next_stage`, so the caller physically gives the frame up and the
    /// receiver stores it. The buffer, layout and lease are the same objects:
    /// the lease id and byte count do not change, so the ledger totals and the
    /// peak stay put and the bytes are charged exactly once, to the receiver.
    ///
    /// The device buffer borrows only the process driver table (kept alive by
    /// the `Gpu`s), not the producing stage's `Gpu`, so the frame stays valid
    /// after the producer drops its pool. The receiver must first make the
    /// producer's last write visible (`sync`, or the recorded
    /// [`Self::last_write_sync`]) before launching its own kernels.
    pub(crate) fn hand_off(mut self, next_stage: &'static str) -> Result<Self, Failure> {
        self.residency
            .hand_off(&mut self.lease, &mut self.layout, next_stage)?;
        Ok(self)
    }

    /// Read-only loan for one more stage. The lease stays charged to the
    /// owner, so a borrow can never double-count the bytes.
    pub(crate) fn borrow_to<'a>(&'a self, borrower: &'static str) -> FrameBorrow<'a> {
        FrameBorrow { frame: self, borrower }
    }

    fn buffer(&self) -> &DeviceBuffer<'static> {
        &self.buffer
    }
}

/// A non-owning, read-only view of a resident frame for another stage.
/// Dropping it changes no ledger state: the owner keeps paying for the bytes.
#[allow(dead_code)] // consumed by later cross-stage fusion steps
pub(crate) struct FrameBorrow<'a> {
    frame: &'a FrameHandle,
    borrower: &'static str,
}

#[allow(dead_code)] // consumed by later cross-stage fusion steps
impl FrameBorrow<'_> {
    pub(crate) fn borrower(&self) -> &'static str {
        self.borrower
    }

    pub(crate) fn owner(&self) -> &'static str {
        self.frame.owner()
    }

    pub(crate) fn handle(&self) -> u64 {
        self.frame.handle()
    }

    pub(crate) fn layout(&self) -> FrameLayout {
        self.frame.layout()
    }

    pub(crate) fn bytes(&self) -> usize {
        self.frame.bytes()
    }

    pub(crate) fn lease_bytes(&self) -> usize {
        self.frame.lease_bytes()
    }
}

struct SlotBuffer {
    slot: Slot,
    buffer: DeviceBuffer<'static>,
    bytes: usize,
    lease: VramLease,
}

// `DeviceBuffer` only borrows the driver table; the owner is `Gpu::gpu`.
unsafe impl Send for SlotBuffer {}

/// One stage's device handle: module, stream, grow-only slots, pinned host and
/// counters. Owned by the stage's `Mutex<State>`, exactly like the old pools.
pub(crate) struct Gpu {
    module: &'static str,
    /// Ledger tag; `module` doubles as the stage name.
    stage: &'static str,
    ptx: &'static str,
    gpu: Nvidia,
    stream: Option<SendStream>,
    slots: Vec<SlotBuffer>,
    counters: Counters,
    launches: Vec<Launch>,
    /// Filled from the first real dispatch: the device the kernels ran on.
    device: Option<String>,
    compute_cap: (i32, i32),
}

impl Gpu {
    /// Opens (or joins) the cached module for `ptx`. The driver wrapper's
    /// module cache makes the second and later calls a lookup, not a JIT.
    pub(crate) fn open(module: &'static str, ptx: &'static str) -> Result<Self, Failure> {
        let gpu = Nvidia::open_ptx(ptx).map_err(Failure::from)?;
        log::debug!(
            "gpu_runtime: {module} module open (ptx loads so far {})",
            ptx_module_loads()
        );
        Ok(Self {
            module,
            stage: module,
            ptx,
            gpu,
            stream: None,
            slots: Vec::new(),
            counters: Counters::default(),
            launches: Vec::new(),
            device: None,
            compute_cap: (0, 0),
        })
    }

    /// Opens without a stream; the stage may call [`Self::with_stream`] later.
    pub(crate) fn open_sync(module: &'static str, ptx: &'static str) -> Result<Self, Failure> {
        Self::open(module, ptx)
    }

    pub(crate) fn module(&self) -> &'static str {
        self.module
    }

    /// Creates the non-blocking stream once; `Unavailable` keeps the caller on
    /// the synchronous copy path instead of disabling the stage.
    pub(crate) fn with_stream(mut self) -> Self {
        if self.stream.is_none() {
            self.stream = self
                .gpu
                .create_stream()
                .ok()
                .map(|stream| unsafe { SendStream(std::mem::transmute::<Stream<'_>, Stream<'static>>(stream)) });
        }
        self
    }

    pub(crate) fn stream(&self) -> Option<&Stream<'_>> {
        self.stream.as_ref().map(|stream| &stream.0)
    }

    /// Grow-only slot allocation. Oversized requests are refused so the caller
    /// can allocate per call or fall back, exactly like the old per-pool caps.
    ///
    /// Every retained slot is a lease in the shared VRAM ledger: growth first
    /// admits the delta (which may reclaim other stages' rebuildable buffers),
    /// and a refusal is a `TooLarge` the caller answers by going to the CPU.
    pub(crate) fn reserve(&mut self, slot: Slot, bytes: usize, cap: usize) -> Result<u64, Failure> {
        if bytes > cap {
            return Err(Failure::TooLarge { need: bytes, cap });
        }
        if let Some(index) = self.slots.iter().position(|entry| entry.slot == slot) {
            if bytes <= self.slots[index].bytes {
                vram_touch(self.slots[index].lease.id());
                return Ok(self.slots[index].buffer.handle());
            }
            let old = self.slots[index].bytes;
            self.slots[index].lease.resize(bytes, true)?;
            match self.alloc(bytes) {
                Ok(buffer) => {
                    self.slots[index].buffer = buffer;
                    self.slots[index].bytes = bytes;
                    Ok(self.slots[index].buffer.handle())
                }
                Err(error) => {
                    // Roll the lease back so the ledger never claims bytes that
                    // were not allocated.
                    let _ = self.slots[index].lease.resize(old, true);
                    Err(error)
                }
            }
        } else {
            let lease = VramLease::acquire(self.stage, bytes, true)?;
            match self.alloc(bytes) {
                Ok(buffer) => {
                    let handle = buffer.handle();
                    self.slots.push(SlotBuffer { slot, buffer, bytes, lease });
                    Ok(handle)
                }
                // The lease is dropped on the error path, releasing the bytes.
                Err(error) => Err(error),
            }
        }
    }

    fn alloc(&self, bytes: usize) -> Result<DeviceBuffer<'static>, Failure> {
        match self.gpu.alloc_buffer(bytes.max(1)) {
            Ok(buffer) => Ok(unsafe {
                std::mem::transmute::<DeviceBuffer<'_>, DeviceBuffer<'static>>(buffer)
            }),
            Err(error) => Err(self.failed(error)),
        }
    }

    pub(crate) fn stage(&self) -> &'static str {
        self.stage
    }

    /// Allocates a device-resident frame charged to this stage's ledger share.
    pub(crate) fn frame(&self, layout: FrameLayout) -> Result<FrameHandle, Failure> {
        FrameHandle::alloc(self, layout)
    }

    pub(crate) fn upload_frame(&mut self, frame: &FrameHandle, data: &[u8]) -> Result<(), Failure> {
        self.gpu.upload_buffer(frame.buffer(), data).map_err(|error| self.failed(error))?;
        self.counters.uploads += 1;
        self.counters.h2d_bytes += data.len() as u64;
        Ok(())
    }

    pub(crate) fn download_frame(
        &mut self,
        frame: &FrameHandle,
        data: &mut [u8],
    ) -> Result<(), Failure> {
        self.gpu
            .download_buffer(frame.buffer(), data)
            .map_err(|error| self.failed(error))?;
        self.counters.downloads += 1;
        self.counters.d2h_bytes += data.len() as u64;
        Ok(())
    }

    pub(crate) fn note_fused_frame(&mut self) {
        self.counters.fused_frames += 1;
    }

    pub(crate) fn note_skipped_upload(&mut self) {
        self.counters.skipped_uploads += 1;
    }

    /// Symmetric to [`Self::note_skipped_upload`]: the frame a previous stage
    /// left resident was consumed without a device-to-host copy.
    pub(crate) fn note_skipped_download(&mut self) {
        self.counters.skipped_downloads += 1;
    }

    pub(crate) fn note_skipped_plane_builds(&mut self, count: u64) {
        self.counters.skipped_plane_builds += count;
    }

    /// Handle of an already-reserved slot.
    pub(crate) fn handle(&self, slot: Slot) -> Option<u64> {
        self.slots
            .iter()
            .find(|entry| entry.slot == slot)
            .map(|entry| entry.buffer.handle())
    }

    /// Bytes currently resident in `slot`.
    pub(crate) fn slot_bytes(&self, slot: Slot) -> usize {
        self.slots
            .iter()
            .find(|entry| entry.slot == slot)
            .map_or(0, |entry| entry.bytes)
    }

    /// Total resident bytes across slots (for budget reporting / tests).
    pub(crate) fn resident_bytes(&self) -> usize {
        self.slots.iter().map(|entry| entry.bytes).sum()
    }

    /// One-off device allocation for frames above the retained cap.
    pub(crate) fn transient(&self, bytes: usize) -> Result<DeviceBuffer<'_>, Failure> {
        self.gpu.alloc_buffer(bytes).map_err(Failure::from)
    }

    fn slot_buffer(&self, slot: Slot) -> Result<&DeviceBuffer<'static>, Failure> {
        self.slots
            .iter()
            .find(|entry| entry.slot == slot)
            .map(|entry| &entry.buffer)
            .ok_or(Failure::Invalid("slot not reserved"))
    }

    /// Borrow of a reserved slot, for stage-local uploads and copies.
    pub(crate) fn buffer(&self, slot: Slot) -> Option<&DeviceBuffer<'static>> {
        self.slots
            .iter()
            .find(|entry| entry.slot == slot)
            .map(|entry| &entry.buffer)
    }

    pub(crate) fn upload(&mut self, slot: Slot, data: &[u8]) -> Result<(), Failure> {
        let buffer = self.slot_buffer(slot)?;
        self.gpu.upload_buffer(buffer, data).map_err(|error| self.failed(error))?;
        self.counters.uploads += 1;
        self.counters.h2d_bytes += data.len() as u64;
        Ok(())
    }

    pub(crate) fn download(&mut self, slot: Slot, data: &mut [u8]) -> Result<(), Failure> {
        let buffer = self.slot_buffer(slot)?;
        self.gpu.download_buffer(buffer, data).map_err(|error| self.failed(error))?;
        self.counters.downloads += 1;
        self.counters.d2h_bytes += data.len() as u64;
        Ok(())
    }

    pub(crate) fn upload_to(
        &mut self,
        buffer: &DeviceBuffer<'_>,
        data: &[u8],
    ) -> Result<(), Failure> {
        self.gpu.upload_buffer(buffer, data).map_err(|error| self.failed(error))?;
        self.counters.uploads += 1;
        self.counters.h2d_bytes += data.len() as u64;
        Ok(())
    }

    pub(crate) fn download_from(
        &mut self,
        buffer: &DeviceBuffer<'_>,
        data: &mut [u8],
    ) -> Result<(), Failure> {
        self.gpu.download_buffer(buffer, data).map_err(|error| self.failed(error))?;
        self.counters.downloads += 1;
        self.counters.d2h_bytes += data.len() as u64;
        Ok(())
    }

    /// H2D from pinned (registered) host memory on the shared stream.
    pub(crate) fn upload_async(&mut self, slot: Slot, data: &[u8]) -> Result<(), Failure> {
        let Some(stream) = self.stream.as_ref().map(|stream| &stream.0) else {
            return self.upload(slot, data);
        };
        let buffer = self.slot_buffer(slot)?;
        self.gpu
            .upload_async(buffer, data, stream)
            .map_err(|error| self.failed(error))?;
        self.counters.uploads += 1;
        self.counters.h2d_bytes += data.len() as u64;
        Ok(())
    }

    pub(crate) fn download_async(&mut self, slot: Slot, data: &mut [u8]) -> Result<(), Failure> {
        let Some(stream) = self.stream.as_ref().map(|stream| &stream.0) else {
            return self.download(slot, data);
        };
        let buffer = self.slot_buffer(slot)?;
        self.gpu
            .download_async(buffer, data, stream)
            .map_err(|error| self.failed(error))?;
        self.counters.downloads += 1;
        self.counters.d2h_bytes += data.len() as u64;
        Ok(())
    }

    /// Waits for every queued copy and kernel. No-op without a stream: the
    /// synchronous copies above already drained the context.
    pub(crate) fn sync(&mut self) -> Result<(), Failure> {
        let Some(stream) = self.stream.as_ref().map(|stream| &stream.0) else {
            return Ok(());
        };
        self.gpu.stream_sync(stream).map_err(|error| self.failed(error))?;
        self.counters.syncs += 1;
        Ok(())
    }

    /// Pins a caller-owned slice in place (no staging copy).
    pub(crate) fn register_host<'a>(
        &self,
        data: &'a mut [u8],
    ) -> Result<aurora_portrait::cuda::HostRegistration<'a>, Failure> {
        self.gpu.register_host(data).map_err(Failure::from)
    }

    /// Allocates page-locked host memory; `Unavailable` means the driver has
    /// no `cuMemAllocHost` and the caller keeps the pageable path.
    pub(crate) fn pinned(&self, bytes: usize) -> Result<aurora_portrait::cuda::PinnedBuffer<'_>, Failure> {
        self.gpu.alloc_pinned(bytes).map_err(Failure::from)
    }

    /// Launches on the shared stream when present, else on the default stream.
    pub(crate) fn launch(
        &mut self,
        kernel: &'static str,
        grid: (u32, u32),
        block: (u32, u32),
        args: &mut [*mut c_void],
    ) -> Result<(), Failure> {
        let result = match self.stream.as_ref().map(|stream| &stream.0) {
            Some(stream) => self.gpu.launch_raw_stream(kernel, grid, block, args, stream),
            None => self.gpu.launch_raw(kernel, grid, block, args),
        };
        let dispatch = result.map_err(|error| self.failed(error))?;
        if self.device.is_none() {
            self.device = Some(dispatch.device);
            self.compute_cap = dispatch.compute_cap;
        }
        self.counters.launches += 1;
        match self.launches.iter_mut().find(|entry| entry.kernel == kernel) {
            Some(entry) => entry.count += 1,
            None => self
                .launches
                .push(Launch { kernel, grid, block, count: 1 }),
        }
        Ok(())
    }

    /// Kernels actually dispatched through this stage, in first-use order.
    pub(crate) fn kernels(&self) -> &[Launch] {
        &self.launches
    }

    pub(crate) fn counters(&self) -> Counters {
        self.counters
    }

    /// Device and compute capability reported by the first real dispatch.
    pub(crate) fn device(&self) -> Option<(&str, (i32, i32))> {
        self.device.as_deref().map(|name| (name, self.compute_cap))
    }

    /// Classifies a driver error, marks the process lost when the code says so,
    /// and returns the reportable failure.
    fn failed(&self, error: CudaError) -> Failure {
        let failure = Failure::from(error);
        if let Failure::Driver { code, .. } = failure {
            mark_lost(code);
            if is_device_lost(code) {
                return Failure::DeviceLost(code);
            }
        }
        failure
    }

    /// Drops every device allocation and the stream, then re-opens the module.
    ///
    /// Not called automatically: stage owners decide whether to retry once or
    /// stay on CPU after a lost device, since a destroyed CUDA context cannot
    /// be recovered by reopening just the module handle.
    ///
    /// The wrapper caches one module per PTX image for the process. That cache
    /// is still valid for a recovered driver context in the common TDR case,
    /// and for a destroyed context every later launch fails; the caller retries
    /// once per service call and otherwise stays on CPU, which is why this
    /// returns the reopen result instead of promising recovery.
    pub(crate) fn reopen(&mut self) -> Result<(), Failure> {
        self.slots.clear();
        self.stream = None;
        self.gpu = Nvidia::open_ptx(self.ptx).map_err(Failure::from)?;
        RESETS.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // DeviceBuffer/Stream drops synchronise and free; keep the stage's
        // accounting honest by clearing the list first (children drop after).
        self.slots.clear();
        self.stream = None;
    }
}

/// Allocation the caller keeps in its own `Send` wrapper (tiny LUTs, tap
/// arrays and statistics words that are uploaded or read whole).
pub(crate) fn static_buffer(gpu: &Gpu, bytes: usize) -> Result<DeviceBuffer<'static>, Failure> {
    let buffer = gpu.gpu.alloc_buffer(bytes.max(1)).map_err(Failure::from)?;
    Ok(unsafe { std::mem::transmute::<DeviceBuffer<'_>, DeviceBuffer<'static>>(buffer) })
}

/// Process-wide driver totals: `(h2d_bytes, d2h_bytes, ptx_module_loads)`.
/// Zero on platforms without the CUDA driver module.
pub(crate) fn transfer_totals() -> (u64, u64, u64) {
    #[cfg(windows)]
    {
        let metrics = aurora_portrait::cuda::driver_metrics();
        (
            metrics.host_to_device_bytes as u64,
            metrics.device_to_host_bytes as u64,
            metrics.ptx_module_load_calls as u64,
        )
    }
    #[cfg(not(windows))]
    {
        (0, 0, 0)
    }
}

fn ptx_module_loads() -> u64 {
    transfer_totals().2
}

/// Shared 2D launch geometry helper: `ceil(w/block.x) x ceil(h/block.y)`.
pub(crate) fn grid2(w: usize, h: usize, block: (u32, u32)) -> (u32, u32) {
    (
        ((w as u32) + block.0 - 1) / block.0,
        ((h as u32) + block.1 - 1) / block.1,
    )
}

/// Shared 1D launch geometry helper.
pub(crate) fn grid1(n: usize, block: u32) -> (u32, u32) {
    (((n as u32) + block - 1) / block, 1)
}

// ───────────────────────── portrait runtime bridge (PP-2) ─────────────────

/// Accounting for the portrait side of the same process context.
///
/// `crates/portrait` owns the driver layer this runtime is built on, so the
/// dependency can only point `aurora-engine -> aurora-portrait`. Portrait
/// exposes an observer hook (`cuda::runtime_bridge`); installing it here keeps
/// one report for every kernel the process runs: the four engine stages and
/// every portrait provider, with the same counters and the same fallback
/// reasons.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct PortraitCounters {
    pub module_loads: u64,
    pub launches: u64,
    pub syncs: u64,
    pub uploads: u64,
    pub downloads: u64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    pub failures: u64,
    pub last_failure: Option<(&'static str, &'static str)>,
    pub kernels: Vec<Launch>,
}

fn portrait_report() -> &'static Mutex<PortraitCounters> {
    static REPORT: OnceLock<Mutex<PortraitCounters>> = OnceLock::new();
    REPORT.get_or_init(|| Mutex::new(PortraitCounters::default()))
}

struct PortraitObserver;

impl aurora_portrait::cuda::runtime_bridge::Observer for PortraitObserver {
    fn observe(&self, event: &aurora_portrait::cuda::runtime_bridge::GpuEvent) {
        use aurora_portrait::cuda::runtime_bridge::GpuEvent as E;
        let mut report = portrait_report()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match event {
            E::ModuleLoaded { .. } => report.module_loads += 1,
            E::Launched {
                kernel,
                grid,
                block,
                ..
            } => {
                report.launches += 1;
                match report
                    .kernels
                    .iter_mut()
                    .find(|entry| {
                        entry.kernel == *kernel && entry.grid == *grid && entry.block == *block
                    })
                {
                    Some(entry) => entry.count += 1,
                    None => report.kernels.push(Launch {
                        kernel: *kernel,
                        grid: *grid,
                        block: *block,
                        count: 1,
                    }),
                }
            }
            E::Synchronized => report.syncs += 1,
            E::Uploaded { bytes, .. } => {
                report.uploads += 1;
                report.h2d_bytes += *bytes as u64;
            }
            E::Downloaded { bytes, .. } => {
                report.downloads += 1;
                report.d2h_bytes += *bytes as u64;
            }
            E::Failed { image, call, code } => {
                report.failures += 1;
                report.last_failure = Some((*image, *call));
                note_fallback(
                    "portrait",
                    &Failure::Driver { call, code: *code },
                );
            }
            E::Unavailable { image, what } => {
                report.failures += 1;
                report.last_failure = Some((*image, *what));
                note_fallback("portrait", &Failure::Unavailable(*what));
            }
            E::Refused { image, tag } => {
                report.failures += 1;
                report.last_failure = Some((*image, *tag));
                note_fallback("portrait", &Failure::Refused(*tag));
            }
        }
    }
}

/// Installs the portrait observer once per process. Called from the portrait
/// stage entry so every provider that runs after it is accounted for.
///
/// Lock order: the observer runs while the portrait driver `Domain` is held.
/// It acquires `portrait_report` and, on failure, `fallback_report`. Neither
/// lock may ever be held while calling back into a portrait GPU API.
pub(crate) fn install_portrait_bridge() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = aurora_portrait::cuda::runtime_bridge::install(Arc::new(PortraitObserver));
    });
}

/// Snapshot of the portrait counters; `kernels` lists the real dispatches.
///
/// Read by the merge gate and the report surface; kept callable so a session
/// can prove portrait kernels ran without touching the per-stage APIs.
#[allow(dead_code)]
pub(crate) fn portrait_snapshot() -> PortraitCounters {
    portrait_report()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lost_codes_are_classified() {
        for code in [700, 702, 709, 710, 719, 999] {
            assert!(is_device_lost(code), "{code}");
        }
        for code in [0, 1, 222, 400, 701] {
            assert!(!is_device_lost(code), "{code}");
        }
    }

    #[test]
    fn fallback_report_counts_per_reason() {
        reset_fallback();
        note_fallback("nr", &Failure::Disabled("test"));
        note_fallback("nr", &Failure::TooLarge { need: 10, cap: 1 });
        note_fallback("pixel", &Failure::Driver { call: "cuMemcpyHtoD", code: 1 });
        let report = fallback_snapshot();
        assert_eq!(report.disabled, 1);
        assert_eq!(report.too_large, 1);
        assert_eq!(report.driver, 1);
        assert_eq!(report.last.iter().find(|(name, _)| *name == "nr").unwrap().1, "too_large");
        assert_eq!(report.last.iter().find(|(name, _)| *name == "pixel").unwrap().1, "driver");
        reset_fallback();
    }

    #[test]
    fn grid_helpers_round_up() {
        assert_eq!(grid2(1920, 1080, (16, 16)), (120, 68));
        assert_eq!(grid1(1025, 256), (5, 1));
    }

    // ── shared VRAM ledger ───────────────────────────────────────────────

    #[test]
    fn ledger_tracks_residency_peak_and_refusal() {
        let mut ledger = VramLedger::new(1000);
        assert!(ledger.admit(1, "clarity", 600, true));
        assert!(ledger.admit(2, "detail", 300, true));
        assert_eq!(ledger.stats().resident_bytes, 900);
        assert_eq!(ledger.stats().peak_bytes, 900);
        // A fourth allocation that would cross the budget is refused.
        assert!(!ledger.admit(3, "pixel", 200, true));
        assert_eq!(ledger.stats().resident_bytes, 900);
        // Growing an existing lease by the same delta is refused just as firmly.
        assert!(!ledger.admit(1, "clarity", 900, true));
        // Shrinking and releasing give bytes back and never underflow.
        assert!(ledger.admit(1, "clarity", 100, true));
        assert_eq!(ledger.stats().resident_bytes, 400);
        ledger.release(2);
        ledger.release(999);
        assert_eq!(ledger.stats().resident_bytes, 100);
        ledger.release(1);
        assert_eq!(ledger.stats().resident_bytes, 0);
    }

    #[test]
    fn can_allocate_counts_reclaimable_buffers_only() {
        let mut ledger = VramLedger::new(1000);
        assert!(ledger.admit(1, "clarity", 700, true));
        assert!(ledger.admit(2, "develop", 200, true));
        // 700 resident above develop's own 200 + 900 more does not fit, but
        // clarity's 700 are rebuildable, so it admits.
        let hooked = ["clarity", "develop"];
        assert!(ledger.can_allocate("develop", 900, &hooked));
        // Nothing can make a request larger than the whole budget fit.
        assert!(!ledger.can_allocate("develop", 1001, &hooked));
    }

    #[test]
    fn can_allocate_never_reclaims_non_evictable_buffers() {
        let mut ledger = VramLedger::new(1000);
        assert!(ledger.admit(1, "clarity", 700, true));
        assert!(ledger.admit(2, "develop", 300, false));
        // Only 700 of the 1000 resident bytes can be reclaimed, so 900 more
        // for pixel leaves 200 unreachable.
        let hooked = ["clarity", "develop"];
        assert!(!ledger.can_allocate("pixel", 900, &hooked));
        assert!(ledger.can_allocate("pixel", 700, &hooked));
    }

    #[test]
    fn can_allocate_does_not_credit_stages_without_a_reclaim_hook() {
        let mut ledger = VramLedger::new(1000);
        // `nr` and `pixel` mark their slots evictable but never register a
        // reclaim hook, so nothing can actually free these bytes.
        assert!(ledger.admit(1, "nr", 600, true));
        assert!(ledger.admit(2, "pixel", 300, true));
        let hooked = ["clarity", "detail", "develop"];
        // 100 bytes of free tail fit; 200 need bytes that no hook can free.
        assert!(ledger.can_allocate("clarity", 100, &hooked));
        assert!(!ledger.can_allocate("clarity", 200, &hooked));
        // The same bytes count once their stage has a hook.
        assert!(ledger.can_allocate("clarity", 200, &["clarity", "detail", "develop", "nr"]));
    }

    #[test]
    fn victim_order_is_priority_then_least_recent() {
        let mut ledger = VramLedger::new(4096);
        assert!(ledger.admit(1, "clarity", 100, true));
        assert!(ledger.admit(2, "detail", 100, true));
        assert!(ledger.admit(3, "develop", 100, true));
        // Re-touch clarity so it is the most recent of its priority band.
        ledger.touch(1);
        let order = ledger.victim_order("pixel", &[("develop", 30), ("clarity", 20), ("detail", 10)]);
        assert_eq!(order, vec!["detail", "clarity", "develop"]);
        // The requester never appears in its own victim list.
        let order = ledger.victim_order("clarity", &[("develop", 30), ("clarity", 20), ("detail", 10)]);
        assert_eq!(order, vec!["detail", "develop"]);
    }

    static FAKE_VICTIM: AtomicU64 = AtomicU64::new(0);

    fn fake_reclaim() -> usize {
        let id = FAKE_VICTIM.load(Ordering::Relaxed);
        if id == 0 {
            return 0;
        }
        let bytes = with_ledger(|ledger| {
            ledger
                .entries
                .iter()
                .find(|entry| entry.id == id)
                .map_or(0, |entry| entry.bytes)
        });
        if bytes > 0 {
            vram_release(id);
        }
        bytes
    }

    #[test]
    fn exhausted_budget_reclaims_a_rebuildable_stage() {
        // Accounting only: no device memory is touched by the ledger.
        let budget = vram_budget();
        let victim_id = 9_000_001;
        let victim_bytes = budget / 2;
        vram_set_lease(victim_id, "test-victim", victim_bytes, true).expect("victim admit");
        FAKE_VICTIM.store(victim_id, Ordering::Relaxed);
        vram_register_reclaimer("test-victim", 1, fake_reclaim);

        let before = vram_stats();
        let requester_id = 9_000_002;
        let available = budget.saturating_sub(before.resident_bytes);
        // One byte more than fits even after the victim was admitted.
        let need = available + 1;
        vram_set_lease(requester_id, "test-requester", need, true).expect("reclaim admits");

        let after = vram_stats();
        assert!(after.eviction_count > before.eviction_count, "no eviction recorded");
        assert!(after.evicted_bytes >= victim_bytes as u64);
        assert!(after.resident_bytes <= budget);
        vram_release(requester_id);
        FAKE_VICTIM.store(0, Ordering::Relaxed);
    }

    #[test]
    fn refusal_is_counted_when_nothing_can_be_reclaimed() {
        // Larger than the whole budget: no amount of reclamation can fit it.
        let request = vram_budget().saturating_add(1);
        let id = 9_000_010;
        let before = vram_stats();
        let refused = vram_set_lease(id, "test-refuser", request, true).is_err();
        assert!(refused, "an over-budget request must fail even after reclamation");
        assert!(vram_stats().refused_count > before.refused_count);
        vram_release(id);
    }

    #[test]
    fn lease_drop_returns_the_bytes() {
        let before = vram_stats().resident_bytes;
        {
            let lease = VramLease::acquire("test-lease", 4096, true).expect("admit");
            assert_eq!(lease.bytes(), 4096);
            assert_eq!(vram_stats().resident_bytes, before + 4096);
        }
        assert_eq!(vram_stats().resident_bytes, before);
    }

    #[test]
    fn leased_buffer_frees_the_device_buffer_before_the_lease() {
        // Stands in for `DeviceBuffer`, which needs a CUDA device to build.
        struct Probe {
            lease_id: u64,
            charged_when_dropped: Arc<std::sync::atomic::AtomicBool>,
        }
        impl Drop for Probe {
            fn drop(&mut self) {
                // The ledger must still own the lease while the buffer is being
                // freed; with the lease first these bytes would already be
                // spendable by another stage.
                let charged = with_ledger(|ledger| {
                    ledger
                        .entries
                        .iter()
                        .any(|entry| entry.id == self.lease_id && entry.bytes > 0)
                });
                self.charged_when_dropped.store(charged, Ordering::SeqCst);
            }
        }

        let charged = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let lease = VramLease::acquire("test-drop-order", 4096, true).expect("admit");
        let lease_id = lease.id();
        drop(LeasedBuffer {
            buffer: Probe { lease_id, charged_when_dropped: Arc::clone(&charged) },
            lease,
        });
        assert!(charged.load(Ordering::SeqCst), "lease returned before the buffer was freed");
        assert!(!with_ledger(|ledger| ledger.entries.iter().any(|entry| entry.id == lease_id)));
    }

    #[test]
    fn repeated_image_switching_does_not_leak() {
        let before = vram_stats().resident_bytes;
        let sizes = [64usize, 256, 64, 512, 256];
        for (index, size) in sizes.iter().enumerate() {
            let id = 9_100_000 + index as u64;
            vram_set_lease(id, "test-image", *size * 1024, true).expect("admit");
            vram_release(id);
        }
        assert_eq!(vram_stats().resident_bytes, before, "residency leaked");
    }

    // ── M1: frame residency descriptor and cross-stage handoff ───────────

    #[test]
    fn generation_stamps_are_nonzero_and_strictly_increasing() {
        let first = next_frame_generation();
        let second = next_frame_generation();
        assert!(first >= 1 && second > first, "{first} {second}");
        // A descriptor that was never allocated still has generation 0, so a
        // consumer can tell "not stamped" from a real residency epoch.
        assert_eq!(FrameLayout::default().generation, 0);
        let mut layout = FrameLayout::new(4, 4, FrameFormat::F32Rgb12, 4 * 4 * 12);
        let before = layout.generation;
        let bumped = layout.bump_generation();
        assert!(bumped > before);
        assert_eq!(layout.generation, bumped);
    }

    #[test]
    fn legacy_layout_literal_still_describes_the_same_allocation() {
        // This is the pre-M1 clarity shape: five explicit fields, no ROI, no
        // stride. `prepare_for_device` must accept it and derive the rest.
        let layout = FrameLayout {
            width: 1920,
            height: 1080,
            format: FrameFormat::F32Rgb12,
            bytes: 1920 * 1080 * 12,
            generation: 0,
            ..FrameLayout::default()
        };
        let prepared = layout
            .prepare_for_device(FrameIdentity { device_id: 0, context_id: 7 })
            .expect("legacy layout admits");
        assert_eq!(prepared.bytes, 1920 * 1080 * 12);
        assert!(prepared.generation >= 1, "allocation must stamp a generation");
        assert_eq!(prepared.residency, Residency::Device);
        assert_eq!(prepared.device_id, Some(0));
        assert_eq!(prepared.context_id, Some(7));
        assert_eq!(prepared.row_bytes(), Some(1920 * 12));
        assert_eq!(prepared.region.addressed(1920, 1080).width, 1920);
    }

    #[test]
    fn composite_plane_cache_is_exempt_from_pixel_geometry_checks() {
        // The plane cache's `bytes` is grid + planes, not w * h * bpp, so the
        // stride/rows check must not reject it.
        let layout = FrameLayout {
            width: 97,
            height: 61,
            format: FrameFormat::F32Planes,
            bytes: 64 * 4 + 2 * 32,
            ..FrameLayout::default()
        };
        assert_eq!(layout.format.bytes_per_px(), None);
        assert_eq!(layout.row_bytes(), None);
        assert!(layout
            .prepare_for_device(FrameIdentity { device_id: 0, context_id: 1 })
            .is_ok());
    }

    #[test]
    fn stride_and_roi_halo_are_checked_against_bytes() {
        let frame = FrameLayout::new(100, 50, FrameFormat::U8Rgba8, 100 * 50 * 4);
        assert_eq!(frame.bit_depth, FrameBitDepth::U8);
        assert_eq!(frame.colorspace, FrameColorSpace::Srgb);
        assert_eq!(frame.row_bytes(), Some(400));
        // A padded stride must still fit the declared bytes.
        let mut padded = frame;
        padded.stride = 512;
        padded.bytes = 512 * 50;
        assert!(padded
            .prepare_for_device(FrameIdentity { device_id: 0, context_id: 1 })
            .is_ok());
        // A stride below one tight row is invalid.
        let mut narrow = frame;
        narrow.stride = 399;
        assert!(narrow
            .prepare_for_device(FrameIdentity { device_id: 0, context_id: 1 })
            .is_err());
        // ROI + halo must be covered by `bytes`.
        let mut small = frame;
        small.region = FrameRegion { x: 10, y: 5, width: 20, height: 10, halo: 2 };
        small.bytes = 100 * 14 * 4;
        assert!(small
            .prepare_for_device(FrameIdentity { device_id: 0, context_id: 1 })
            .is_ok());
        small.bytes = 100 * 13 * 4;
        assert!(small
            .prepare_for_device(FrameIdentity { device_id: 0, context_id: 1 })
            .is_err());
        // An ROI outside the frame is rejected.
        let mut outside = frame;
        outside.region = FrameRegion { x: 90, y: 0, width: 20, height: 10, halo: 0 };
        assert!(outside
            .prepare_for_device(FrameIdentity { device_id: 0, context_id: 1 })
            .is_err());
    }

    #[test]
    fn format_variants_report_u8_and_u16_geometry() {
        for (format, bpp, depth, space) in [
            (FrameFormat::U8Rgba8, 4, FrameBitDepth::U8, FrameColorSpace::Srgb),
            (FrameFormat::U16Rgba16, 8, FrameBitDepth::U16, FrameColorSpace::Srgb),
            (FrameFormat::F32Rgb12, 12, FrameBitDepth::F32, FrameColorSpace::LinearRgb),
        ] {
            assert_eq!(format.bytes_per_px(), Some(bpp));
            assert_eq!(format.bit_depth(), depth);
            assert_eq!(format.default_colorspace(), space);
        }
        assert_eq!(FrameFormat::F32Planes.bytes_per_px(), None);
        assert_eq!(FrameFormat::F32Planes.bit_depth(), FrameBitDepth::F32);
        assert_eq!(FrameFormat::F32Planes.default_colorspace(), FrameColorSpace::Data);
    }

    #[test]
    fn handoff_retags_the_lease_without_double_charging_or_releasing() {
        let before = vram_stats();
        let identity = FrameIdentity { device_id: 0, context_id: 3 };
        let mut layout = FrameLayout::new(64, 32, FrameFormat::U8Rgba8, 64 * 32 * 4)
            .prepare_for_device(identity)
            .expect("layout");
        let generation = layout.generation;
        let mut lease = VramLease::acquire("test-handoff-from", layout.bytes, true)
            .expect("admit");
        let id = lease.id();
        let charged = vram_stats();
        assert_eq!(charged.resident_bytes, before.resident_bytes + layout.bytes);

        // A handoff only relabels the ledger entry: same id, same bytes.
        lease.reassign("test-handoff-to").expect("reassign");
        assert_eq!(lease.stage(), "test-handoff-to");
        let after = vram_stats();
        assert_eq!(after.resident_bytes, charged.resident_bytes);
        assert_eq!(after.peak_bytes, charged.peak_bytes);
        assert_eq!(after.eviction_count, charged.eviction_count);
        assert_eq!(lease.id(), id);
        assert_eq!(lease.bytes(), layout.bytes);
        assert_eq!(with_ledger(|ledger| ledger.stage_bytes("test-handoff-from")), 0);
        assert_eq!(with_ledger(|ledger| ledger.stage_bytes("test-handoff-to")), layout.bytes);

        // The frame-side bump is separate from the ledger move.
        assert!(layout.bump_generation() > generation);
        assert_eq!(lease.bytes(), layout.bytes);

        // Normal drop returns the bytes exactly once.
        drop(lease);
        assert_eq!(vram_stats().resident_bytes, before.resident_bytes);
    }

    #[test]
    fn handoff_transfers_owner_and_generation_but_not_bytes() {
        // Exercises the exact `FrameResidency::hand_off` path `FrameHandle`
        // uses, with a real ledger lease but no device buffer.
        let before = vram_stats().resident_bytes;
        let identity = FrameIdentity { device_id: 0, context_id: 11 };
        let mut layout = FrameLayout::new(32, 16, FrameFormat::U16Rgba16, 32 * 16 * 8)
            .prepare_for_device(identity)
            .expect("layout");
        let generation = layout.generation;
        let mut lease = VramLease::acquire("test-owner-a", layout.bytes, true).expect("admit");
        let id = lease.id();
        let mut residency = FrameResidency::new("test-owner-a", identity);
        assert_eq!(residency.owner, "test-owner-a");

        residency
            .hand_off(&mut lease, &mut layout, "test-owner-b")
            .expect("handoff");
        assert_eq!(residency.owner, "test-owner-b");
        assert_eq!(lease.id(), id);
        assert_eq!(lease.bytes(), 32 * 16 * 8);
        assert_eq!(lease.stage(), "test-owner-b");
        assert!(layout.generation > generation);
        assert_eq!(vram_stats().resident_bytes, before + 32 * 16 * 8);
        assert_eq!(with_ledger(|ledger| ledger.stage_bytes("test-owner-a")), 0);
        assert_eq!(
            with_ledger(|ledger| ledger.stage_bytes("test-owner-b")),
            32 * 16 * 8
        );

        // A handoff to the current owner is a no-op, not a second charge.
        let unchanged = layout.generation;
        residency
            .hand_off(&mut lease, &mut layout, "test-owner-b")
            .expect("noop");
        assert_eq!(layout.generation, unchanged);
        assert_eq!(vram_stats().resident_bytes, before + 32 * 16 * 8);

        drop(lease);
        assert_eq!(vram_stats().resident_bytes, before);
    }

    #[test]
    fn handoff_of_a_released_lease_is_refused() {
        let mut lease = VramLease::acquire("test-handoff-gone", 1024, true).expect("admit");
        lease.forget();
        assert!(lease.reassign("test-handoff-next").is_err());
        assert_eq!(with_ledger(|ledger| ledger.stage_bytes("test-handoff-next")), 0);
    }

    #[test]
    fn ledger_reassign_moves_attribution_only() {
        let mut ledger = VramLedger::new(1 << 20);
        assert!(ledger.admit(7001, "from", 4096, true));
        assert!(ledger.admit(7002, "other", 1024, true));
        assert_eq!(ledger.stats().resident_bytes, 5120);
        assert!(ledger.reassign(7001, "to"));
        assert_eq!(ledger.stage_bytes("from"), 0);
        assert_eq!(ledger.stage_bytes("to"), 4096);
        assert_eq!(ledger.stage_bytes("other"), 1024);
        assert_eq!(ledger.stats().resident_bytes, 5120);
        assert_eq!(ledger.stats().peak_bytes, 5120);
        assert!(!ledger.reassign(9999, "to"));
    }

    #[test]
    fn context_token_is_one_per_process() {
        assert!(process_context_token() >= 1);
        assert_eq!(process_context_token(), process_context_token());
        assert!(mint_context_token() > process_context_token());
    }
}
