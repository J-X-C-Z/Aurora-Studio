//! One `MTLDevice` and one `MTLCommandQueue` for the whole process.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandQueue, MTLDevice};

use super::{MetalError, MetalProbe};

// `objc2-metal` links the Metal framework but not CoreGraphics;
// `MTLCreateSystemDefaultDevice` is exported by CoreGraphics. Linking an empty
// extern block is the documented way to add the framework (objc2-metal docs).
#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

/// The process-wide device and queue.
///
/// Both `MTLDevice` and `MTLCommandQueue` protocols are declared `Send + Sync`
/// by `objc2-metal`, so this struct is safe to share. Command buffers and
/// compute encoders are *not* kept here: they are created per workspace and
/// stay on the calling thread.
pub(crate) struct MetalDeviceInner {
    pub(crate) device: Retained<ProtocolObject<dyn MTLDevice>>,
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub(crate) probe: MetalProbe,
}

impl MetalDeviceInner {
    pub(crate) fn open() -> Result<Self, MetalError> {
        // The returned device is +1 (Create rule), modelled by `Retained`.
        let device = objc2::rc::autoreleasepool(|_| objc2_metal::MTLCreateSystemDefaultDevice())
            .ok_or_else(|| {
                MetalError::Unavailable("MTLCreateSystemDefaultDevice returned nil".to_string())
            })?;
        let queue = objc2::rc::autoreleasepool(|_| device.newCommandQueue())
            .ok_or_else(|| MetalError::Unavailable("newCommandQueue returned nil".to_string()))?;
        let name = objc2::rc::autoreleasepool(|_| device.name().to_string());
        let probe = MetalProbe {
            name,
            unified: device.hasUnifiedMemory(),
            recommended_working_set: device.recommendedMaxWorkingSetSize(),
            max_buffer_len: device.maxBufferLength() as u64,
            registry_id: device.registryID(),
        };
        Ok(Self {
            device,
            queue,
            probe,
        })
    }
}
