//! Public facade for the macOS Metal compute runtime.
//!
//! M2 scope:
//!
//! * one process-wide `MTLDevice` + `MTLCommandQueue`, opened on first use;
//! * MSL compiled at runtime with fast math disabled (precompiled `.metallib`
//!   is a packaging follow-up);
//! * a `BufferPool` of shared buffers for uploads, parameters and readback;
//! * a workspace that encodes several kernels into **one** command buffer and
//!   commits without waiting;
//! * a committed [`MetalFrame`] whose [`MetalFrame::read_into`] performs the
//!   only `waitUntilCompleted` and the only CPU copy.
//!
//! The runtime is not wired into `ImageEngine` or `mock::process` yet; that is
//! Mac-C's next step. It also does not create a scheduler or a second VRAM
//! ledger. The per-runtime counters are for proving the batching contract.
//!
//! Errors use the local [`MetalError`] because `mock::gpu_runtime` (and its
//! `Failure` enum) is Windows-only in the current tree; unifying those is a
//! follow-up owned by Mac-A.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::metal::RuntimeCounters;
use crate::metal::device::MetalDeviceInner;
use crate::metal::encode::{CommandBatch, ResolvedBinding};
use crate::metal::library::{LibraryCache, PipelineCache};
use crate::metal::pool::BufferPool;
use crate::metal::residency::FrameBuffers;

pub use crate::metal::library::PipelineLimits;
pub use crate::metal::pool::MetalBuffer;
pub use crate::metal::residency::MetalFrame;
pub use crate::metal::{Grid, MetalCounters, MetalError, MetalProbe};

/// `(MSL source, kernel entry)` cache key.
///
/// The library cache hashes `source` and keys pipelines by
/// `(source hash, entry)`; fast math is fixed to `false` in the compile
/// options and included in the library key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub source: &'static str,
    pub entry: &'static str,
}

impl PipelineKey {
    pub const fn new(source: &'static str, entry: &'static str) -> Self {
        Self { source, entry }
    }
}

/// Opaque handle for a buffer owned by one [`MetalWorkspace`].
///
/// Bindings use the handle rather than `&MetalBuffer` so a chain can upload
/// several inputs and then dispatch without borrowing the workspace twice.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferId(usize);

impl BufferId {
    pub fn index(self) -> usize {
        self.0
    }
}

/// One kernel binding, in MSL `[[buffer(n)]]` order starting at 0.
pub enum Binding<'a> {
    /// A buffer owned by this workspace (`BufferId` from `upload`/`alloc`).
    Buffer(BufferId),
    /// A small (≤ 4 KiB) parameter struct copied inline by Metal.
    Uniform(&'a [u8]),
}

/// Process-wide Metal runtime.
pub struct MetalRuntime {
    inner: MetalDeviceInner,
    libraries: LibraryCache,
    pipelines: PipelineCache,
    pool: BufferPool,
    counters: RuntimeCounters,
    next_generation: AtomicU64,
}

static SHARED: OnceLock<Result<MetalRuntime, MetalError>> = OnceLock::new();

impl MetalRuntime {
    /// Opens (once) and returns the process runtime.
    ///
    /// The first thread that calls this creates the only device and queue;
    /// later calls return the same runtime. Failure is cached, including the
    /// reason, so repeated probes stay cheap and truthful.
    pub fn shared() -> Result<&'static Self, MetalError> {
        match SHARED.get_or_init(Self::open) {
            Ok(runtime) => Ok(runtime),
            Err(error) => Err(error.clone()),
        }
    }

    fn open() -> Result<Self, MetalError> {
        let inner = MetalDeviceInner::open()?;
        let pool = BufferPool::new(inner.device.clone());
        Ok(Self {
            inner,
            libraries: LibraryCache::new(),
            pipelines: PipelineCache::new(),
            pool,
            counters: RuntimeCounters::new(),
            next_generation: AtomicU64::new(0),
        })
    }

    /// Probes the system device without retaining a second one.
    pub fn probe() -> Option<MetalProbe> {
        Self::shared().ok().map(|runtime| runtime.probe_info().clone())
    }

    /// Device capabilities captured when the runtime opened.
    pub fn probe_info(&self) -> &MetalProbe {
        &self.inner.probe
    }

    /// Starts a frame: one command buffer and a fresh generation.
    pub fn begin(self: &'static Self) -> Result<MetalWorkspace, MetalError> {
        let generation = self.next_generation.fetch_add(1, Ordering::Relaxed) + 1;
        let batch = CommandBatch::new(&self.inner.queue)?;
        Ok(MetalWorkspace {
            runtime: self,
            batch,
            buffers: Vec::new(),
            generation,
        })
    }

    /// Snapshot of the batching/transfer/buffer counters.
    pub fn counters(&self) -> MetalCounters {
        self.counters.snapshot(&self.pool)
    }

    fn pipeline(&self, key: PipelineKey) -> Result<std::sync::Arc<crate::metal::library::PipelineEntry>, MetalError> {
        self.pipelines.get_or_create(
            &self.inner.device,
            &self.libraries,
            key.source,
            key.entry,
            &self.counters,
        )
    }
}

/// A frame under construction: buffers, one command buffer, many dispatches.
pub struct MetalWorkspace {
    runtime: &'static MetalRuntime,
    batch: CommandBatch,
    buffers: Vec<crate::metal::pool::MetalBuffer>,
    generation: u64,
}

impl MetalWorkspace {
    /// Copies a byte slice into a shared buffer and returns its handle.
    pub fn upload(&mut self, bytes: &[u8]) -> Result<BufferId, MetalError> {
        let buffer = self.runtime.pool.upload(bytes)?;
        self.buffers.push(buffer);
        Ok(BufferId(self.buffers.len() - 1))
    }

    /// Allocates a shared output/scratch buffer written by a kernel.
    pub fn alloc(&mut self, bytes: usize) -> Result<BufferId, MetalError> {
        let buffer = self.runtime.pool.alloc(bytes)?;
        self.buffers.push(buffer);
        Ok(BufferId(self.buffers.len() - 1))
    }

    /// Borrows a workspace buffer (for diagnostics; bindings use handles).
    pub fn buffer(&self, id: BufferId) -> Result<&MetalBuffer, MetalError> {
        self.buffers
            .get(id.0)
            .ok_or(MetalError::Invalid("buffer id is out of range"))
    }

    /// Compiles (or fetches) a pipeline and returns its real thread limits.
    pub fn pipeline_limits(&self, key: PipelineKey) -> Result<PipelineLimits, MetalError> {
        self.runtime.pipeline(key).map(|pipeline| pipeline.limits())
    }

    /// Encodes one kernel into its own compute encoder on the shared command
    /// buffer. Never waits.
    pub fn dispatch(
        &self,
        key: PipelineKey,
        bindings: &[Binding<'_>],
        grid: Grid,
    ) -> Result<(), MetalError> {
        let pipeline = self.runtime.pipeline(key)?;
        let resolved = bindings
            .iter()
            .map(|binding| match binding {
                Binding::Buffer(id) => self
                    .buffers
                    .get(id.0)
                    .map(ResolvedBinding::Buffer)
                    .ok_or(MetalError::Invalid("buffer id is out of range")),
                Binding::Uniform(bytes) => Ok(ResolvedBinding::Uniform(bytes)),
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.batch.dispatch(&pipeline, &resolved, grid)?;
        self.runtime
            .counters
            .compute_encoders
            .fetch_add(1, Ordering::Relaxed);
        self.runtime
            .counters
            .dispatches
            .fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Commits the single command buffer without waiting.
    ///
    /// `output` selects the buffer whose bytes `read_into` copies back.
    pub fn commit(self, output: BufferId) -> Result<MetalFrame, MetalError> {
        let Self {
            runtime,
            batch,
            buffers,
            generation,
        } = self;
        let output_len = buffers
            .get(output.0)
            .ok_or(MetalError::Invalid("output buffer id is out of range"))?
            .requested_len();
        let command_buffer = batch.commit()?;
        runtime
            .counters
            .command_buffers
            .fetch_add(1, Ordering::Relaxed);
        Ok(MetalFrame {
            generation,
            command_buffer,
            buffers: std::sync::Mutex::new(Some(FrameBuffers {
                buffers,
                output: output.0,
                output_len,
            })),
            pool: &runtime.pool,
            counters: &runtime.counters,
        })
    }
}
