//! One command buffer, many compute encoders, no per-kernel wait.
//!
//! [`CommandBatch`] creates exactly one `MTLCommandBuffer`. Each [`dispatch`]
//! opens a fresh compute encoder, binds resources, dispatches and ends the
//! encoder; encoders are ordered by the command buffer itself. `commit` only
//! submits. The single `waitUntilCompleted` lives in
//! `MetalFrame::read_into`.

use std::ffi::c_void;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLSize,
};

use super::Grid;
use super::MetalError;
use super::library::PipelineEntry;
use super::pool::MetalBuffer;

/// Metal guarantees `setBytes:length:atIndex:` copies at most 4 KiB.
const MAX_INLINE_UNIFORM: usize = 4096;

/// A binding resolved against buffers owned by the workspace.
pub(crate) enum ResolvedBinding<'a> {
    Buffer(&'a MetalBuffer),
    /// Small parameter struct copied inline; no buffer lifetime to manage.
    Uniform(&'a [u8]),
}

pub(crate) struct CommandBatch {
    command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
}

impl CommandBatch {
    pub(crate) fn new(
        queue: &ProtocolObject<dyn MTLCommandQueue>,
    ) -> Result<Self, MetalError> {
        let command_buffer = objc2::rc::autoreleasepool(|_| queue.commandBuffer())
            .ok_or_else(|| {
                MetalError::Unavailable("commandQueue.commandBuffer returned nil".to_string())
            })?;
        Ok(Self { command_buffer })
    }

    /// Encodes one kernel into its own compute encoder.
    pub(crate) fn dispatch(
        &self,
        pipeline: &PipelineEntry,
        bindings: &[ResolvedBinding<'_>],
        grid: Grid,
    ) -> Result<(), MetalError> {
        let threads_total = grid.threads_total();
        if threads_total == 0 || threads_total > u64::from(pipeline.max_total_threads) {
            return Err(MetalError::Invalid(
                "threadgroup size exceeds pipeline maxTotalThreadsPerThreadgroup",
            ));
        }
        if grid.threads.iter().any(|value| *value == 0) {
            return Err(MetalError::Invalid("zero-sized threadgroup dimension"));
        }

        objc2::rc::autoreleasepool(|_| {
            let encoder = self
                .command_buffer
                .computeCommandEncoder()
                .ok_or_else(|| {
                    MetalError::Unavailable("computeCommandEncoder returned nil".to_string())
                })?;
            encoder.setComputePipelineState(&pipeline.state);
            for (index, binding) in bindings.iter().enumerate() {
                match binding {
                    ResolvedBinding::Buffer(buffer) => {
                        // SAFETY: `buffer` is a live MTLBuffer owned by the
                        // workspace for the entire batch; index is a literal
                        // binding slot from the MSL signature.
                        unsafe {
                            encoder.setBuffer_offset_atIndex(Some(buffer.protocol()), 0, index);
                        }
                    }
                    ResolvedBinding::Uniform(bytes) => {
                        if bytes.len() > MAX_INLINE_UNIFORM {
                            return Err(MetalError::Invalid(
                                "inline uniform exceeds the 4 KiB setBytes limit",
                            ));
                        }
                        if bytes.is_empty() {
                            continue;
                        }
                        let pointer = NonNull::new(bytes.as_ptr() as *mut c_void)
                            .ok_or_else(|| MetalError::Invalid("uniform pointer is null"))?;
                        // SAFETY: The pointer/length describe a live borrowed
                        // slice; Metal copies the bytes before returning.
                        unsafe {
                            encoder.setBytes_length_atIndex(pointer, bytes.len(), index);
                        }
                    }
                }
            }
            encoder.dispatchThreadgroups_threadsPerThreadgroup(
                MTLSize {
                    width: grid.groups[0] as usize,
                    height: grid.groups[1] as usize,
                    depth: grid.groups[2] as usize,
                },
                MTLSize {
                    width: grid.threads[0] as usize,
                    height: grid.threads[1] as usize,
                    depth: grid.threads[2] as usize,
                },
            );
            encoder.endEncoding();
            Ok(())
        })
    }

    /// Submits without waiting; the frame owns the command buffer afterward.
    pub(crate) fn commit(
        self,
    ) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, MetalError> {
        objc2::rc::autoreleasepool(|_| self.command_buffer.commit());
        Ok(self.command_buffer)
    }
}
