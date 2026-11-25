//! Committed Metal frame: generation stamp and the one CPU readback.
//!
//! `commit` never waits. `read_into` is the single synchronous point: it waits
//! once, checks the command buffer status, copies the shared output buffer and
//! returns every workspace buffer to the pool. A frame that is dropped without
//! being read is released without waiting; the command buffer retains its
//! resources until the GPU finishes, so this is safe, it just forfeits reuse.

use std::sync::Mutex;
use std::sync::atomic::Ordering;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};

use super::pool::{BufferPool, MetalBuffer};
use super::{MetalError, RuntimeCounters};

pub(crate) struct FrameBuffers {
    pub(crate) buffers: Vec<MetalBuffer>,
    pub(crate) output: usize,
    pub(crate) output_len: usize,
}

/// A committed batch awaiting CPU readback.
pub struct MetalFrame {
    pub(crate) generation: u64,
    pub(crate) command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    pub(crate) buffers: Mutex<Option<FrameBuffers>>,
    pub(crate) pool: &'static BufferPool,
    pub(crate) counters: &'static RuntimeCounters,
}

impl MetalFrame {
    /// Monotonic value assigned when the workspace began.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Bytes the output buffer was requested with.
    pub fn output_len(&self) -> usize {
        self.buffers
            .lock()
            .unwrap_or_else(|poison| poison.into_inner())
            .as_ref()
            .map_or(0, |frame| frame.output_len)
    }

    /// Waits once and copies the shared output buffer into `dst`.
    ///
    /// One-shot: after a successful (or failed execution) read the frame holds
    /// no buffers and a second call returns `Invalid`.
    pub fn read_into(&self, dst: &mut [u8]) -> Result<(), MetalError> {
        let mut guard = self
            .buffers
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let Some(frame) = guard.take() else {
            return Err(MetalError::Invalid("MetalFrame::read_into called twice"));
        };
        if dst.len() < frame.output_len {
            guard.replace(frame);
            return Err(MetalError::Invalid("readback destination is too small"));
        }

        // The only wait in the runtime, inside an autorelease pool because
        // `error()` materialises an NSError.
        self.counters.waits.fetch_add(1, Ordering::Relaxed);
        let outcome = objc2::rc::autoreleasepool(|_| {
            self.command_buffer.waitUntilCompleted();
            let status = self.command_buffer.status();
            if status == MTLCommandBufferStatus::Error {
                Err(self
                    .command_buffer
                    .error()
                    .map_or_else(|| "unknown error".to_string(), |error| error.to_string()))
            } else if status != MTLCommandBufferStatus::Completed {
                // Should be unreachable after waitUntilCompleted.
                Err(format!("unexpected command buffer status {}", status.0))
            } else {
                Ok(())
            }
        });
        if let Err(message) = outcome {
            let FrameBuffers { buffers, .. } = frame;
            for buffer in buffers {
                self.pool.recycle(buffer);
            }
            return Err(MetalError::CommandBuffer(message));
        }

        let FrameBuffers {
            buffers,
            output,
            output_len,
        } = frame;
        let output_buffer = &buffers[output];
        // SAFETY: status is Completed, the buffer is StorageModeShared and the
        // CPU has not written it since the upload, so reading `output_len`
        // bytes is valid.
        unsafe {
            let source = std::slice::from_raw_parts(output_buffer.contents_ptr(), output_len);
            dst[..output_len].copy_from_slice(source);
        }
        self.counters.readbacks.fetch_add(1, Ordering::Relaxed);

        for buffer in buffers {
            self.pool.recycle(buffer);
        }
        Ok(())
    }
}
