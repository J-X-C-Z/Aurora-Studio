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
use std::sync::{Mutex, OnceLock};

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
}

struct SendStream(Stream<'static>);

// `Stream` only borrows the driver table, which the boxed module handle keeps
// alive for `Gpu`'s whole lifetime, and `Gpu` is only reached behind a lock.
unsafe impl Send for SendStream {}

struct SlotBuffer {
    slot: Slot,
    buffer: DeviceBuffer<'static>,
    bytes: usize,
}

// `DeviceBuffer` only borrows the driver table; the owner is `Gpu::gpu`.
unsafe impl Send for SlotBuffer {}

/// One stage's device handle: module, stream, grow-only slots, pinned host and
/// counters. Owned by the stage's `Mutex<State>`, exactly like the old pools.
pub(crate) struct Gpu {
    module: &'static str,
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
    pub(crate) fn reserve(&mut self, slot: Slot, bytes: usize, cap: usize) -> Result<u64, Failure> {
        if bytes > cap {
            return Err(Failure::TooLarge { need: bytes, cap });
        }
        if let Some(found) = self.slots.iter_mut().find(|entry| entry.slot == slot) {
            if bytes > found.bytes {
                let buffer = self.gpu.alloc_buffer(bytes.max(1)).map_err(Failure::from)?;
                let buffer = unsafe {
                    std::mem::transmute::<DeviceBuffer<'_>, DeviceBuffer<'static>>(buffer)
                };
                found.buffer = buffer;
                found.bytes = bytes;
            }
            return Ok(found.buffer.handle());
        }
        let buffer = self.gpu.alloc_buffer(bytes.max(1)).map_err(Failure::from)?;
        let buffer =
            unsafe { std::mem::transmute::<DeviceBuffer<'_>, DeviceBuffer<'static>>(buffer) };
        let handle = buffer.handle();
        self.slots.push(SlotBuffer { slot, buffer, bytes });
        Ok(handle)
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
}
