//! MSL library and compute-pipeline caches.
//!
//! MSL is compiled at runtime for M2. A precompiled `.metallib` shipped in the
//! bundle is a packaging follow-up; the cache keys already isolate the exact
//! compilation inputs so the switch will not invalidate call sites.

use std::collections::HashMap;
use std::sync::Mutex;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLCompileOptions, MTLComputePipelineState, MTLDevice, MTLLibrary};

use super::MetalError;

/// 64-bit FNV-1a over the MSL bytes. Stable for the process lifetime and
/// independent of `std`'s randomly seeded hasher.
pub(crate) fn source_hash(source: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in source.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Library cache key: MSL source hash plus the compile option that changes
/// floating-point semantics. `fast_math` is always `false` today; keeping it
/// in the key prevents a future fast-math build from reusing a strict library.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct LibraryKey {
    pub(crate) source_hash: u64,
    pub(crate) fast_math: bool,
}

pub(crate) struct LibraryCache {
    entries: Mutex<HashMap<LibraryKey, Retained<ProtocolObject<dyn MTLLibrary>>>>,
}

impl LibraryCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the cached library or compiles `source` with fast math disabled.
    ///
    /// `source` must be `'static` so the key can be derived without copying;
    /// both current kernels come from `include_str!`.
    pub(crate) fn get_or_compile(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        source: &'static str,
    ) -> Result<LibraryKey, MetalError> {
        let key = LibraryKey {
            source_hash: source_hash(source),
            fast_math: false,
        };
        {
            let entries = self
                .entries
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if entries.contains_key(&key) {
                return Ok(key);
            }
        }
        let library = objc2::rc::autoreleasepool(|_| {
            let options = MTLCompileOptions::new();
            // FP32 parity work depends on this: no contract/fast-math rewrite.
            // `setMathMode` supersedes this on macOS 15+, but the deployment
            // target is 12.0 and the old selector is still present at runtime.
            #[allow(deprecated)]
            options.setFastMathEnabled(false);
            let ns_source = NSString::from_str(source);
            device
                .newLibraryWithSource_options_error(&ns_source, Some(&options))
                .map_err(|error| MetalError::Compile(error.to_string()))
        })?;
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        // Another thread may have compiled the same source while we did.
        entries.entry(key).or_insert(library);
        Ok(key)
    }
}

/// One compiled compute pipeline plus the limits used to build its grid.
pub(crate) struct PipelineEntry {
    pub(crate) state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    pub(crate) thread_execution_width: u32,
    pub(crate) max_total_threads: u32,
}

/// Public-by-re-export limits; see [`PipelineEntry`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PipelineLimits {
    pub thread_execution_width: u32,
    pub max_total_threads: u32,
}

impl PipelineEntry {
    pub(crate) fn limits(&self) -> PipelineLimits {
        PipelineLimits {
            thread_execution_width: self.thread_execution_width,
            max_total_threads: self.max_total_threads,
        }
    }
}

type PipelineKey = (u64, &'static str);

pub(crate) struct PipelineCache {
    entries: Mutex<HashMap<PipelineKey, std::sync::Arc<PipelineEntry>>>,
}

impl PipelineCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Returns the cached pipeline, compiling `source` and creating `entry`.
    pub(crate) fn get_or_create(
        &self,
        device: &ProtocolObject<dyn MTLDevice>,
        libraries: &LibraryCache,
        source: &'static str,
        entry: &'static str,
        counters: &super::RuntimeCounters,
    ) -> Result<std::sync::Arc<PipelineEntry>, MetalError> {
        let library_key = libraries.get_or_compile(device, source)?;
        let key = (library_key.source_hash, entry);
        {
            let entries = self
                .entries
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            if let Some(pipeline) = entries.get(&key) {
                return Ok(std::sync::Arc::clone(pipeline));
            }
        }

        let library = {
            let entries = libraries
                .entries
                .lock()
                .unwrap_or_else(|poison| poison.into_inner());
            entries
                .get(&library_key)
                .cloned()
                .ok_or_else(|| MetalError::Pipeline("library disappeared from cache".to_string()))?
        };

        let function = objc2::rc::autoreleasepool(|_| {
            let name = NSString::from_str(entry);
            library.newFunctionWithName(&name).ok_or_else(|| {
                MetalError::Pipeline(format!("kernel entry `{entry}` not found in library"))
            })
        })?;

        let state = objc2::rc::autoreleasepool(|_| {
            device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|error| {
                    MetalError::Pipeline(format!("pipeline `{entry}` failed: {error}"))
                })
        })?;

        let thread_execution_width = state.threadExecutionWidth().min(u32::MAX as usize) as u32;
        let max_total_threads =
            state.maxTotalThreadsPerThreadgroup().min(u32::MAX as usize) as u32;
        let pipeline = std::sync::Arc::new(PipelineEntry {
            state,
            thread_execution_width,
            max_total_threads,
        });
        let mut entries = self
            .entries
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        if let Some(existing) = entries.get(&key) {
            return Ok(std::sync::Arc::clone(existing));
        }
        entries.insert(key, std::sync::Arc::clone(&pipeline));
        counters
            .pipelines_compiled
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(pipeline)
    }
}
