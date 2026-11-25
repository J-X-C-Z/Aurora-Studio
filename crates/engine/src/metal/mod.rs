//! Internal building blocks for the macOS Metal compute runtime.
//!
//! The public surface lives in [`crate::metal_runtime`]; this module owns the
//! device/queue, MSL library and pipeline caches, the shared-memory buffer
//! pool, the one-command-buffer batch encoder and the committed frame that
//! performs the single CPU readback.
//!
//! Everything here is `cfg(target_os = "macos")` because the parent `mod` in
//! `lib.rs` is gated. Windows keeps using the CUDA stages in `mock/*_gpu.rs`.
//!
//! Error idiom: the shared Windows runtime uses `mock::gpu_runtime::Failure`,
//! but that module is compiled out on macOS (it is Windows-only and imports
//! `aurora_portrait::cuda`). Reaching it would mean importing CUDA types into
//! the macOS build, so this runtime defines its own small [`MetalError`].
//! Unifying the two error enums is a follow-up owned by Mac-A when
//! `gpu_runtime` is split into a backend-neutral ledger plus CUDA backend.

pub(crate) mod device;
pub(crate) mod encode;
pub(crate) mod library;
pub(crate) mod pool;
pub(crate) mod residency;

use std::sync::atomic::{AtomicU64, Ordering};

/// Explicit failure reasons for the Metal runtime.
///
/// Every variant is reportable; nothing silently falls back to a CPU success
/// while claiming the GPU ran.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MetalError {
    /// No device/queue, or an Objective-C factory returned `nil`.
    Unavailable(String),
    /// `newLibraryWithSource:options:error:` rejected the MSL source.
    Compile(String),
    /// A kernel entry is missing or pipeline-state creation failed.
    Pipeline(String),
    /// A committed command buffer finished with `MTLCommandBufferStatusError`.
    CommandBuffer(String),
    /// The request is structurally invalid (shape, binding, readback size).
    Invalid(&'static str),
}

impl std::fmt::Display for MetalError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unavailable(what) => write!(formatter, "Metal unavailable: {what}"),
            Self::Compile(what) => write!(formatter, "Metal compile failed: {what}"),
            Self::Pipeline(what) => write!(formatter, "Metal pipeline failed: {what}"),
            Self::CommandBuffer(what) => write!(formatter, "Metal command buffer failed: {what}"),
            Self::Invalid(what) => write!(formatter, "Metal request invalid: {what}"),
        }
    }
}

impl std::error::Error for MetalError {}

/// Device-reported capabilities, captured once when the process runtime opens.
///
/// `recommended_working_set` is a soft GPU working-set hint, not extra memory
/// on Apple Silicon; it is reported for later admission policy and is **not**
/// a second VRAM ledger.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetalProbe {
    /// `MTLDevice.name`, e.g. `Apple M2`.
    pub name: String,
    /// `MTLDevice.hasUnifiedMemory`.
    pub unified: bool,
    /// `MTLDevice.recommendedMaxWorkingSetSize` in bytes.
    pub recommended_working_set: u64,
    /// `MTLDevice.maxBufferLength` in bytes.
    pub max_buffer_len: u64,
    /// `MTLDevice.registryID`; useful for a future same-device check with GPUI.
    pub registry_id: u64,
}

/// A 2D or 3D dispatch grid in Metal terms.
///
/// `groups` is the number of threadgroups and `threads` is the size of one
/// threadgroup. Build 2D grids from the pipeline's real
/// `threadExecutionWidth` / `maxTotalThreadsPerThreadgroup` via [`Grid::for_2d`];
/// 16x16 is deliberately not assumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Grid {
    pub groups: [u32; 3],
    pub threads: [u32; 3],
}

impl Grid {
    /// Explicit grid; callers that know their kernel's threadgroup policy.
    pub const fn new(groups: [u32; 3], threads: [u32; 3]) -> Self {
        Self { groups, threads }
    }

    /// Tiling derived from a compiled pipeline's reported limits.
    ///
    /// The first dimension is the SIMD width (`threadExecutionWidth`); the
    /// second consumes the remaining `maxTotalThreadsPerThreadgroup` budget.
    /// This produces `threads == [32, 32, 1]` on an M2 rather than a fixed
    /// 16x16, and adapts to other GPUs.
    pub fn for_2d(width: u32, height: u32, limits: library::PipelineLimits) -> Self {
        let max_total = limits.max_total_threads.max(1);
        let simd = limits.thread_execution_width.max(1).min(max_total);
        let rows = (max_total / simd).max(1);
        let groups = [
            width.div_ceil(simd).max(1),
            height.div_ceil(rows).max(1),
            1,
        ];
        Self {
            groups,
            threads: [simd, rows, 1],
        }
    }

    /// Total threads requested per threadgroup.
    pub fn threads_total(&self) -> u64 {
        self.threads[0] as u64 * self.threads[1] as u64 * self.threads[2] as u64
    }
}

/// Process-wide counters for the Metal runtime.
///
/// This is a per-runtime accounting counter, deliberately not a second
/// scheduler and not a second GPU-memory ledger. Buffer bytes are tracked by
/// the pool; command-buffer/encoder/readback counts make the "one batch, one
/// wait" contract assertable from tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MetalCounters {
    /// Command buffers committed (`commit` without waiting).
    pub command_buffers: u64,
    /// Compute encoders created; one per dispatched kernel.
    pub compute_encoders: u64,
    /// Kernel dispatches encoded.
    pub dispatches: u64,
    /// Successful CPU readbacks.
    pub readbacks: u64,
    /// `waitUntilCompleted` calls (must be one per `MetalFrame::read_into`).
    pub waits: u64,
    /// Pipelines compiled (cache misses).
    pub pipelines_compiled: u64,
    /// `MTLBuffer` objects newly allocated.
    pub buffer_allocations: u64,
    /// Allocations satisfied from a pooled buffer.
    pub buffer_reuses: u64,
    /// Bytes currently checked out of the pool.
    pub live_bytes: u64,
    /// Bytes retained in the pool's free lists.
    pub pooled_bytes: u64,
    /// Peak checked-out bytes.
    pub peak_live_bytes: u64,
}

pub(crate) struct RuntimeCounters {
    pub(crate) command_buffers: AtomicU64,
    pub(crate) compute_encoders: AtomicU64,
    pub(crate) dispatches: AtomicU64,
    pub(crate) readbacks: AtomicU64,
    pub(crate) waits: AtomicU64,
    pub(crate) pipelines_compiled: AtomicU64,
}

impl RuntimeCounters {
    pub(crate) fn new() -> Self {
        Self {
            command_buffers: AtomicU64::new(0),
            compute_encoders: AtomicU64::new(0),
            dispatches: AtomicU64::new(0),
            readbacks: AtomicU64::new(0),
            waits: AtomicU64::new(0),
            pipelines_compiled: AtomicU64::new(0),
        }
    }

    pub(crate) fn snapshot(&self, pool: &pool::BufferPool) -> MetalCounters {
        let pool_stats = pool.stats();
        MetalCounters {
            command_buffers: self.command_buffers.load(Ordering::Relaxed),
            compute_encoders: self.compute_encoders.load(Ordering::Relaxed),
            dispatches: self.dispatches.load(Ordering::Relaxed),
            readbacks: self.readbacks.load(Ordering::Relaxed),
            waits: self.waits.load(Ordering::Relaxed),
            pipelines_compiled: self.pipelines_compiled.load(Ordering::Relaxed),
            buffer_allocations: pool_stats.allocations,
            buffer_reuses: pool_stats.reuses,
            live_bytes: pool_stats.live_bytes,
            pooled_bytes: pool_stats.pooled_bytes,
            peak_live_bytes: pool_stats.peak_live_bytes,
        }
    }
}
