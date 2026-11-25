//! Shared-memory buffer pool with power-of-two size classes.
//!
//! Uploads, kernel parameters (uniforms) and the readback surface all use
//! `MTLResourceStorageModeShared`, so the CPU can see GPU-written bytes after
//! the single wait in [`crate::metal::residency::MetalFrame::read_into`].
//! A private/staging path can be added for large intermediates later; it is
//! not needed by the first two-kernel chain.
//!
//! This pool is a per-runtime allocator, not a second VRAM ledger: it only
//! reports the bytes it actually holds. Admission policy stays with the
//! existing engine ledger (Mac-A owns that split).

use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

use super::MetalError;

/// Smallest size class; keeps tiny masks/parameters from making 1-byte buffers.
const MIN_CLASS_BYTES: usize = 4096;
/// Maximum bytes kept free across all classes. Per-runtime retention policy,
/// not a global budget: excess buffers are simply released to the driver.
const MAX_FREE_BYTES: usize = 256 * 1024 * 1024;
/// Maximum free buffers retained per size class.
const MAX_FREE_PER_CLASS: usize = 8;

fn size_class(requested: usize) -> usize {
    requested.max(MIN_CLASS_BYTES).next_power_of_two()
}

/// An owned `MTLBuffer`. Only the workspace pool hands these out; they are not
/// cloned, so a frame is the unique owner at readback time.
pub struct MetalBuffer {
    buffer: Retained<ProtocolObject<dyn MTLBuffer>>,
    /// Allocated size class in bytes.
    capacity: usize,
    /// Bytes the caller actually asked for.
    requested: usize,
}

// SAFETY: Metal resource objects (`MTLBuffer` in particular) are documented as
// thread-safe handles with no thread affinity; only command encoders and
// command buffers require single-thread use, and those are never stored in
// shared state. This runtime never writes `contents()` from the CPU: uploads
// are copied at creation (`newBufferWithBytes:`) and the only CPU access is
// the immutable readback after `waitUntilCompleted`.
unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

impl MetalBuffer {
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn requested_len(&self) -> usize {
        self.requested
    }

    pub(crate) fn protocol(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.buffer
    }

    /// Base pointer of a shared buffer. Reading is only valid after the command
    /// buffer that writes it has completed.
    pub(crate) fn contents_ptr(&self) -> *const u8 {
        self.buffer.contents().as_ptr() as *const u8
    }
}

#[derive(Default)]
struct PoolInner {
    free: HashMap<usize, Vec<MetalBuffer>>,
    free_bytes: u64,
}

pub(crate) struct PoolStats {
    pub(crate) allocations: u64,
    pub(crate) reuses: u64,
    pub(crate) live_bytes: u64,
    pub(crate) pooled_bytes: u64,
    pub(crate) peak_live_bytes: u64,
}

pub(crate) struct BufferPool {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    inner: Mutex<PoolInner>,
    live_bytes: AtomicU64,
    peak_live_bytes: AtomicU64,
    allocations: AtomicU64,
    reuses: AtomicU64,
}

impl BufferPool {
    pub(crate) fn new(device: Retained<ProtocolObject<dyn MTLDevice>>) -> Self {
        Self {
            device,
            inner: Mutex::new(PoolInner::default()),
            live_bytes: AtomicU64::new(0),
            peak_live_bytes: AtomicU64::new(0),
            allocations: AtomicU64::new(0),
            reuses: AtomicU64::new(0),
        }
    }

    /// Buffers a byte slice into a shared `MTLBuffer` (a copy; unified memory
    /// does not remove this copy).
    pub(crate) fn upload(&self, bytes: &[u8]) -> Result<MetalBuffer, MetalError> {
        if bytes.is_empty() {
            return Err(MetalError::Invalid("cannot upload zero bytes"));
        }
        // A reused class buffer only needs to have enough capacity; the copy
        // overwrites exactly `bytes.len()` bytes.
        let mut buffer = self.alloc_class(size_class(bytes.len()), bytes.len())?;
        let pointer = NonNull::new(buffer.contents_ptr() as *mut c_void)
            .ok_or_else(|| MetalError::Invalid("shared buffer contents is null"))?;
        // SAFETY: `pointer` is the base of a Shared buffer of at least
        // `bytes.len()` bytes and CPU-owned until the next GPU commit.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), pointer.as_ptr() as *mut u8, bytes.len());
        }
        buffer.requested = bytes.len();
        Ok(buffer)
    }

    /// Allocates an uninitialised shared buffer; the kernel writes it.
    pub(crate) fn alloc(&self, requested: usize) -> Result<MetalBuffer, MetalError> {
        if requested == 0 {
            return Err(MetalError::Invalid("cannot allocate zero bytes"));
        }
        self.alloc_class(size_class(requested), requested)
    }

    fn alloc_class(&self, class: usize, requested: usize) -> Result<MetalBuffer, MetalError> {
        if let Some(mut reused) = self.take_free(class) {
            reused.requested = requested;
            self.reuses.fetch_add(1, Ordering::Relaxed);
            let live = self.live_bytes.fetch_add(class as u64, Ordering::Relaxed) + class as u64;
            self.peak_live_bytes.fetch_max(live, Ordering::Relaxed);
            return Ok(reused);
        }

        let buffer = objc2::rc::autoreleasepool(|_| {
            self.device
                .newBufferWithLength_options(class, MTLResourceOptions::StorageModeShared)
        })
        .ok_or_else(|| {
            MetalError::Unavailable(format!("newBufferWithLength({class}) returned nil"))
        })?;

        self.allocations.fetch_add(1, Ordering::Relaxed);
        let live = self.live_bytes.fetch_add(class as u64, Ordering::Relaxed) + class as u64;
        self.peak_live_bytes.fetch_max(live, Ordering::Relaxed);
        Ok(MetalBuffer {
            buffer,
            capacity: class,
            requested,
        })
    }

    fn take_free(&self, class: usize) -> Option<MetalBuffer> {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let buffer = inner.free.get_mut(&class)?.pop()?;
        inner.free_bytes = inner.free_bytes.saturating_sub(buffer.capacity as u64);
        Some(buffer)
    }

    /// Returns a completed buffer to the free list (or releases it when the
    /// retention caps are reached).
    pub(crate) fn recycle(&self, buffer: MetalBuffer) {
        let class = buffer.capacity;
        let mut buffer = Some(buffer);
        {
            let mut inner = self
                .inner
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            let has_room = inner.free_bytes + class as u64 <= MAX_FREE_BYTES as u64
                && inner
                    .free
                    .get(&class)
                    .map_or(true, |list| list.len() < MAX_FREE_PER_CLASS);
            if has_room {
                inner.free_bytes += class as u64;
                let released = buffer.take().expect("buffer is present");
                inner.free.entry(class).or_default().push(released);
            }
        }
        // If the retention caps refused it, release outside the lock; the
        // Retained drop may call into ObjC.
        drop(buffer);
        self.live_bytes
            .fetch_sub(class as u64, Ordering::Relaxed);
    }

    pub(crate) fn stats(&self) -> PoolStats {
        let pooled_bytes = self
            .inner
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .free_bytes;
        PoolStats {
            allocations: self.allocations.load(Ordering::Relaxed),
            reuses: self.reuses.load(Ordering::Relaxed),
            live_bytes: self.live_bytes.load(Ordering::Relaxed),
            pooled_bytes,
            peak_live_bytes: self.peak_live_bytes.load(Ordering::Relaxed),
        }
    }
}
