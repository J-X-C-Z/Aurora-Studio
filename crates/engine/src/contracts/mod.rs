//! Public computation contracts shared by every backend and platform adapter.
//!
//! This module is deliberately dependency-free apart from the standard library
//! and `serde`: it describes pixels, coordinates, capability and dispatch, and
//! never pulls in the render pipeline, a device driver or `aurora-portrait`.
//! Frame descriptors and execution plans are the stable vocabulary that the
//! CPU reference, the Windows CUDA adapter and the macOS Metal adapter agree on.
//!
//! Nothing in here allocates device memory or launches a kernel. Implementations
//! live in the compute adapters (`mock::gpu_runtime`, the future Metal adapter);
//! this module only owns the types those adapters exchange.

pub mod compute;
pub mod frame;
pub mod identity;
