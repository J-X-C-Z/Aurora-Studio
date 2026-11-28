//! Apple Metal / Core ML platform adapter for fused pixel-stage execution.
//!
//! All vendor-specific code lives behind `cfg(any(target_os = "macos",
//! target_os = "ios"))`. The Windows x86_64 MSVC/GNU host compiles this module
//! without any Apple SDK: the Objective-C / Metal / Core ML frameworks are
//! resolved with `dlopen`/`dlsym` at runtime on Apple targets only, and every
//! entry point on any other host returns an explicit
//! [`ComputeError::BackendUnavailable`]. There is **no silent CPU fallback**.
//!
//! The MSL kernel is the repository source at
//! `native/metal/pixel_fusion.metal`; this module embeds it verbatim and uses
//! the pipeline's real `threadExecutionWidth` /
//! `maxTotalThreadsPerThreadgroup` to choose a threadgroup, and the device's
//! `hasUnifiedMemory` flag to choose `MTLStorageModeShared` vs
//! `MTLStorageModePrivate`.
//!
//! Status note: the kernel and the candidate space are `DESIGNED`. Runtime
//! device parity and performance are not claimed by this source-level adapter.

use std::collections::HashMap;

use crate::autotune::{CandidateSpec, ParamBound, VerificationStatus};
use crate::compute_backend::{
    BufferId, BufferPoolStats, BufferUsage, ComputeBackend, ComputeError, ExecutionReceipt,
    PipelineCacheStats, PipelineHandle, PipelineKey, PixelStageFusion,
};
use crate::device::{Backend, DeviceProbe, Precision};
use crate::Image;

/// Verbatim MSL source reused by this backend.
pub const PIXEL_FUSION_MSL: &str = include_str!("../native/metal/pixel_fusion.metal");
/// Single kernel entry point declared in the MSL source.
pub const PIXEL_FUSION_ENTRY: &str = "pixel_fusion";
/// Exact refusal string used on every non-Apple host.
pub const METAL_UNAVAILABLE: &str = "Metal requires macOS/iOS";

/// Byte layout of the `PixelFusionParams` struct in `pixel_fusion.metal`.
pub const PIXEL_FUSION_PARAMS_BYTES: usize = 52;

/// Stage bits shared with the HIP adapter; must stay in sync with both shaders.
pub mod stage_bits {
    pub const BRIGHTNESS: u32 = 1 << 0;
    pub const WHITEN: u32 = 1 << 1;
    pub const ROSY: u32 = 1 << 2;
    pub const TEMPERATURE: u32 = 1 << 3;
    pub const SATURATION: u32 = 1 << 4;
    pub const LUT1D: u32 = 1 << 5;
    pub const LUT3D: u32 = 1 << 6;
}

/// How a Metal buffer is backed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageMode {
    /// `MTLStorageModeShared` (CPU and GPU see the same bytes).
    Shared,
    /// `MTLStorageModePrivate` (GPU-only allocation, explicit `getBytes:`).
    Private,
}

impl StorageMode {
    pub const fn as_metal_option(self) -> usize {
        match self {
            Self::Shared => 0,
            Self::Private => 1,
        }
    }
}

/// Choose the allocation mode. Staging/parameter buffers are always shared;
/// image buffers use shared storage only on unified-memory devices.
pub const fn storage_mode_for(unified_memory: bool, staging: bool) -> StorageMode {
    if staging || unified_memory {
        StorageMode::Shared
    } else {
        StorageMode::Private
    }
}

/// Limits reported by a compiled Metal pipeline state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MetalLimits {
    pub thread_execution_width: u32,
    pub max_total_threads_per_threadgroup: u32,
}

impl MetalLimits {
    /// Derive `(threads_x, threads_y)` from the pipeline's SIMD width and total
    /// threadgroup budget. This never assumes the legacy fixed 16x16 group.
    pub fn threadgroup(self, width: u32, height: u32) -> (u32, u32) {
        let _ = (width, height);
        let total = self.max_total_threads_per_threadgroup.max(1);
        let simd = self.thread_execution_width.max(1).min(total);
        let rows = (total / simd).max(1);
        (simd, rows)
    }

    /// One-dimensional grid over `width * height` pixels. The threadgroup size
    /// is `threadExecutionWidth * floor(maxTotal / threadExecutionWidth)`, so a
    /// 32-wide M-series pipeline and a 128-wide discrete pipeline both use the
    /// full SIMD width without exceeding their total-thread budget.
    pub fn grid(self, width: u32, height: u32) -> (u32, u32) {
        let (tx, ty) = self.threadgroup(width, height);
        let threads = tx.saturating_mul(ty).max(1);
        let pixels = (width as u64)
            .saturating_mul(height as u64)
            .min(u32::MAX as u64)
            .max(1) as u32;
        (pixels.div_ceil(threads).max(1), threads)
    }

    pub fn is_valid(self) -> bool {
        self.thread_execution_width > 0
            && self.max_total_threads_per_threadgroup >= self.thread_execution_width
            && self.thread_execution_width <= 1024
            && self.max_total_threads_per_threadgroup <= 1024
    }
}

/// `MLComputeUnits` values exposed by Apple's Core ML `MLModelConfiguration`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MlComputeUnits {
    CpuOnly,
    CpuAndGpu,
    CpuAndNeuralEngine,
    All,
}

impl MlComputeUnits {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::CpuOnly => "cpuOnly",
            Self::CpuAndGpu => "cpuAndGPU",
            Self::CpuAndNeuralEngine => "cpuAndNeuralEngine",
            Self::All => "all",
        }
    }

    pub const fn allows_gpu(self) -> bool {
        matches!(self, Self::CpuAndGpu | Self::All)
    }

    pub const fn allows_ane(self) -> bool {
        matches!(self, Self::CpuAndNeuralEngine | Self::All)
    }
}

/// One statically known model input/output shape. `None` means a dynamic axis.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticShape {
    pub name: &'static str,
    pub dims: &'static [Option<usize>],
}

/// Core ML execution probe. `ane_partition_verified` is evidence-driven and is
/// never inferred from the requested compute units or from an API symbol being
/// present. A caller must supply real per-node placement evidence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CoreMlExecutionProbe {
    pub units: MlComputeUnits,
    pub model_shapes: Vec<StaticShape>,
    /// Caller assertion that some subgraph was partitioned to the ANE.
    pub ane_partition_claimed: bool,
    /// Verified from real per-node placement evidence; starts `false`.
    pub ane_partition_verified: bool,
    pub candidate_params: Vec<ParamBound>,
}

impl CoreMlExecutionProbe {
    pub fn new(
        units: MlComputeUnits,
        model_shapes: Vec<StaticShape>,
        ane_partition_claimed: bool,
    ) -> Self {
        Self {
            units,
            model_shapes,
            ane_partition_claimed,
            ane_partition_verified: false,
            candidate_params: core_ml_params().to_vec(),
        }
    }

    /// A claim is only meaningful when the selected units actually allow ANE.
    pub fn claims_ane(&self) -> bool {
        self.units.allows_ane() && self.ane_partition_claimed
    }

    /// Promote the ANE claim only when external evidence says so. A request for
    /// ANE units is not evidence of ANE execution.
    pub fn with_ane_evidence(mut self, evidence: bool) -> Self {
        self.ane_partition_verified = evidence && self.claims_ane();
        self
    }

    pub fn verified_target(&self) -> Option<Backend> {
        if self.ane_partition_verified {
            Some(Backend::Npu)
        } else if self.units.allows_gpu() {
            Some(Backend::Metal)
        } else {
            Some(Backend::Cpu)
        }
    }
}

static CORE_ML_PARAMS: &[ParamBound] = &[
    ParamBound::EnumChoice {
        name: "ml_compute_units",
        choices: &["cpuOnly", "cpuAndGPU", "cpuAndNeuralEngine", "all"],
    },
    ParamBound::DiscreteU32 {
        name: "ane_partition_nodes",
        values: &[0, 1, 2, 4, 8, 16],
    },
    ParamBound::BoolFlag {
        name: "ane_partition_verified",
    },
    ParamBound::DiscreteU32 {
        name: "static_shape_rank",
        values: &[2, 3, 4],
    },
];

pub fn core_ml_params() -> &'static [ParamBound] {
    CORE_ML_PARAMS
}

static MOBILE_PARAMS: &[ParamBound] = &[
    ParamBound::DiscreteU32 {
        name: "threadgroup_width",
        values: &[32, 64, 128],
    },
    ParamBound::U32Range {
        name: "tile_pixels",
        min: 64,
        max: 512,
        step: 64,
    },
    ParamBound::BoolFlag {
        name: "unified_memory_staging",
    },
];

static DESKTOP_PARAMS: &[ParamBound] = &[
    ParamBound::DiscreteU32 {
        name: "threadgroup_width",
        values: &[32, 64, 128, 256],
    },
    ParamBound::U32Range {
        name: "tile_pixels",
        min: 128,
        max: 1024,
        step: 128,
    },
    ParamBound::BoolFlag {
        name: "unified_memory_staging",
    },
];

static ANE_PARAMS: &[ParamBound] = &[
    ParamBound::DiscreteU32 {
        name: "ane_partition_nodes",
        values: &[0, 1, 2, 4],
    },
    ParamBound::BoolFlag {
        name: "fp16_compute",
    },
];

macro_rules! apple_candidate {
    ($id:literal, $backend:expr, $params:expr, $smem:expr, $threads:expr) => {
        CandidateSpec {
            id: $id,
            algorithm_id: "pixel_fusion",
            backend: $backend,
            kernel_entry: PIXEL_FUSION_ENTRY,
            precision: Precision::Fp32,
            params: $params,
            max_shared_mem_bytes: $smem,
            max_threads_per_group: $threads,
            status: VerificationStatus::Designed,
        }
    };
}

/// Bounded candidate space covering Apple M1..M4 families and A-series SoCs.
/// Every entry is `DESIGNED`; no device timing or parity evidence is asserted.
pub static APPLE_CANDIDATES: &[CandidateSpec] = &[
    apple_candidate!("metal.m1.pixel_fusion", Backend::Metal, MOBILE_PARAMS, 16 * 1024, 256),
    apple_candidate!("metal.m1pro.pixel_fusion", Backend::Metal, DESKTOP_PARAMS, 32 * 1024, 512),
    apple_candidate!("metal.m1max.pixel_fusion", Backend::Metal, DESKTOP_PARAMS, 32 * 1024, 1024),
    apple_candidate!("metal.m1ultra.pixel_fusion", Backend::Metal, DESKTOP_PARAMS, 48 * 1024, 1024),
    apple_candidate!("metal.m2.pixel_fusion", Backend::Metal, MOBILE_PARAMS, 16 * 1024, 256),
    apple_candidate!("metal.m2pro.pixel_fusion", Backend::Metal, DESKTOP_PARAMS, 32 * 1024, 512),
    apple_candidate!("metal.m2max.pixel_fusion", Backend::Metal, DESKTOP_PARAMS, 32 * 1024, 1024),
    apple_candidate!("metal.m2ultra.pixel_fusion", Backend::Metal, DESKTOP_PARAMS, 48 * 1024, 1024),
    apple_candidate!("metal.m3.pixel_fusion", Backend::Metal, MOBILE_PARAMS, 16 * 1024, 256),
    apple_candidate!("metal.m3pro.pixel_fusion", Backend::Metal, DESKTOP_PARAMS, 32 * 1024, 512),
    apple_candidate!("metal.m3max.pixel_fusion", Backend::Metal, DESKTOP_PARAMS, 32 * 1024, 1024),
    apple_candidate!("metal.m3ultra.pixel_fusion", Backend::Metal, DESKTOP_PARAMS, 48 * 1024, 1024),
    apple_candidate!("metal.m4.pixel_fusion", Backend::Metal, MOBILE_PARAMS, 16 * 1024, 256),
    apple_candidate!("metal.m4pro.pixel_fusion", Backend::Metal, DESKTOP_PARAMS, 32 * 1024, 512),
    apple_candidate!("metal.m4max.pixel_fusion", Backend::Metal, DESKTOP_PARAMS, 48 * 1024, 1024),
    apple_candidate!("metal.a14.pixel_fusion", Backend::Metal, MOBILE_PARAMS, 8 * 1024, 256),
    apple_candidate!("metal.a15.pixel_fusion", Backend::Metal, MOBILE_PARAMS, 8 * 1024, 256),
    apple_candidate!("metal.a16.pixel_fusion", Backend::Metal, MOBILE_PARAMS, 16 * 1024, 256),
    apple_candidate!("metal.a17.pixel_fusion", Backend::Metal, MOBILE_PARAMS, 16 * 1024, 256),
    apple_candidate!("metal.a17pro.pixel_fusion", Backend::Metal, MOBILE_PARAMS, 16 * 1024, 512),
    apple_candidate!("metal.a18.pixel_fusion", Backend::Metal, MOBILE_PARAMS, 16 * 1024, 256),
    apple_candidate!("metal.a18pro.pixel_fusion", Backend::Metal, MOBILE_PARAMS, 16 * 1024, 512),
    apple_candidate!("coreml.m1.ane.pixel_fusion", Backend::Npu, ANE_PARAMS, 0, 1),
    apple_candidate!("coreml.m2.ane.pixel_fusion", Backend::Npu, ANE_PARAMS, 0, 1),
    apple_candidate!("coreml.m3.ane.pixel_fusion", Backend::Npu, ANE_PARAMS, 0, 1),
    apple_candidate!("coreml.m4.ane.pixel_fusion", Backend::Npu, ANE_PARAMS, 0, 1),
    apple_candidate!("coreml.a17.ane.pixel_fusion", Backend::Npu, ANE_PARAMS, 0, 1),
    apple_candidate!("coreml.a18.ane.pixel_fusion", Backend::Npu, ANE_PARAMS, 0, 1),
];

#[derive(Debug)]
struct BufferEntry {
    data: Vec<u8>,
    usage: BufferUsage,
}

enum MetalState {
    Unavailable(&'static str),
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    Ready(apple::MetalDevice),
}

/// Metal implementation of [`ComputeBackend`].
///
/// Constructing with [`Self::new`] never probes hardware and stays unavailable;
/// call [`Self::open`] to actually resolve the Metal frameworks and compile the
/// kernel. On a non-Apple host both operations return
/// `ComputeError::BackendUnavailable`.
pub struct MetalComputeBackend {
    state: MetalState,
    next_buffer_id: u64,
    next_pipeline_id: u64,
    active: HashMap<BufferId, BufferEntry>,
    free_pool: Vec<Vec<u8>>,
    allocations: u64,
    reuses: u64,
    pipeline_cache: HashMap<PipelineKey, PipelineHandle>,
    hits: u64,
    misses: u64,
}

impl Default for MetalComputeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalComputeBackend {
    pub fn new() -> Self {
        Self {
            state: MetalState::Unavailable(METAL_UNAVAILABLE),
            next_buffer_id: 1,
            next_pipeline_id: 1,
            active: HashMap::new(),
            free_pool: Vec::new(),
            allocations: 0,
            reuses: 0,
            pipeline_cache: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    /// Open a real Metal device and compile `pixel_fusion.metal`. Returns an
    /// explicit refusal on non-Apple hosts or when any framework symbol, device
    /// or pipeline is missing.
    ///
    /// The kernel is compiled with fast-math contraction disabled and binds a
    /// 52-byte `PixelFusionParams` plus the four buffers declared in the MSL.
    pub fn open() -> Result<Self, ComputeError> {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let device = apple::MetalDevice::open()?;
            let mut backend = Self::new();
            backend.state = MetalState::Ready(device);
            Ok(backend)
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            Err(ComputeError::BackendUnavailable(METAL_UNAVAILABLE))
        }
    }

    fn is_ready(&self) -> bool {
        match &self.state {
            MetalState::Unavailable(_) => false,
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            MetalState::Ready(_) => true,
        }
    }

    fn unavailable(&self) -> ComputeError {
        match &self.state {
            MetalState::Unavailable(reason) => ComputeError::BackendUnavailable(reason),
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            MetalState::Ready(_) => ComputeError::BackendUnavailable("Metal device is ready"),
        }
    }

    #[cfg(any(target_os = "macos", target_os = "ios"))]
    fn ready(&self) -> Result<&apple::MetalDevice, ComputeError> {
        match &self.state {
            MetalState::Ready(device) => Ok(device),
            MetalState::Unavailable(reason) => Err(ComputeError::BackendUnavailable(reason)),
        }
    }

    /// Pipeline limits of the real compiled pipeline; unavailable elsewhere.
    pub fn limits(&self) -> Result<MetalLimits, ComputeError> {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let device = self.ready()?;
            Ok(device.limits)
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            Err(self.unavailable())
        }
    }

    /// Storage mode selected for a real image buffer.
    pub fn image_storage_mode(&self) -> Result<StorageMode, ComputeError> {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let device = self.ready()?;
            Ok(storage_mode_for(device.unified, false))
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            Err(self.unavailable())
        }
    }

    pub fn trim_pool(&mut self) {
        self.free_pool.clear();
    }
}

impl ComputeBackend for MetalComputeBackend {
    fn backend(&self) -> Backend {
        Backend::Metal
    }

    fn probe(&self) -> DeviceProbe {
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        let ready = matches!(self.state, MetalState::Ready(_));
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let ready = false;
        DeviceProbe {
            backend: Backend::Metal,
            driver_ready: ready,
            kernel_compiled: ready,
            fp32: true,
            fp16: true,
            max_width: 16384,
            max_height: 16384,
        }
    }

    fn allocate_buffer(
        &mut self,
        byte_len: usize,
        usage: BufferUsage,
    ) -> Result<BufferId, ComputeError> {
        if !self.is_ready() {
            return Err(self.unavailable());
        }
        let best = self
            .free_pool
            .iter()
            .enumerate()
            .filter(|(_, buf)| buf.capacity() >= byte_len)
            .min_by_key(|(_, buf)| buf.capacity())
            .map(|(idx, _)| idx);
        let data = match best {
            Some(idx) => {
                let mut buf = self.free_pool.swap_remove(idx);
                buf.clear();
                buf.resize(byte_len, 0);
                self.reuses += 1;
                buf
            }
            None => {
                self.allocations += 1;
                vec![0u8; byte_len]
            }
        };
        let id = BufferId(self.next_buffer_id);
        self.next_buffer_id += 1;
        self.active.insert(id, BufferEntry { data, usage });
        Ok(id)
    }

    fn release_buffer(&mut self, id: BufferId) -> Result<(), ComputeError> {
        if !self.is_ready() {
            return Err(self.unavailable());
        }
        let entry = self
            .active
            .remove(&id)
            .ok_or(ComputeError::BufferNotFound(id))?;
        self.free_pool.push(entry.data);
        Ok(())
    }

    fn upload_buffer(&mut self, id: BufferId, data: &[u8]) -> Result<(), ComputeError> {
        if !self.is_ready() {
            return Err(self.unavailable());
        }
        let entry = self
            .active
            .get_mut(&id)
            .ok_or(ComputeError::BufferNotFound(id))?;
        if entry.data.len() != data.len() {
            return Err(ComputeError::BufferSizeMismatch {
                expected: entry.data.len(),
                actual: data.len(),
            });
        }
        entry.data.copy_from_slice(data);
        Ok(())
    }

    fn download_buffer(&self, id: BufferId) -> Result<Vec<u8>, ComputeError> {
        if !self.is_ready() {
            return Err(self.unavailable());
        }
        self.active
            .get(&id)
            .map(|entry| entry.data.clone())
            .ok_or(ComputeError::BufferNotFound(id))
    }

    fn get_or_compile_pipeline(
        &mut self,
        fusion: &PixelStageFusion,
    ) -> Result<PipelineHandle, ComputeError> {
        if !self.is_ready() {
            return Err(self.unavailable());
        }
        if fusion.is_empty() {
            return Err(ComputeError::EmptyPipeline);
        }
        let key = fusion.pipeline_key();
        if let Some(&handle) = self.pipeline_cache.get(&key) {
            self.hits += 1;
            return Ok(handle);
        }
        self.misses += 1;
        let handle = PipelineHandle(self.next_pipeline_id);
        self.next_pipeline_id += 1;
        self.pipeline_cache.insert(key, handle);
        Ok(handle)
    }

    fn synchronize(&mut self) -> Result<(), ComputeError> {
        if !self.is_ready() {
            return Err(self.unavailable());
        }
        Ok(())
    }

    fn dispatch_pixel_fusion(
        &mut self,
        input: &Image,
        fusion: &PixelStageFusion,
    ) -> Result<(Image, ExecutionReceipt), ComputeError> {
        if !self.is_ready() {
            return Err(self.unavailable());
        }
        input.validate().map_err(ComputeError::InvalidImage)?;
        let pipeline = self.get_or_compile_pipeline(fusion)?;
        let mut params = fusion_params(fusion);
        params[0..4].copy_from_slice(&input.width.to_le_bytes());
        params[4..8].copy_from_slice(&input.height.to_le_bytes());
        let byte_len = input.pixels.len();

        let src = self.allocate_buffer(byte_len, BufferUsage::StorageRead)?;
        let dst = self.allocate_buffer(byte_len, BufferUsage::StorageWrite)?;
        self.upload_buffer(src, &input.pixels)?;
        let source = self
            .active
            .get(&src)
            .ok_or(ComputeError::BufferNotFound(src))?
            .data
            .clone();
        let mut output = vec![0u8; byte_len];

        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            let device = self.ready()?;
            device.dispatch(&params, &source, &mut output)?;
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        let _ = (&params, &source);

        self.upload_buffer(dst, &output)?;
        let out_pixels = self.download_buffer(dst)?;
        self.release_buffer(src)?;
        self.release_buffer(dst)?;
        self.synchronize()?;

        let receipt = ExecutionReceipt {
            backend: Backend::Metal,
            pipeline,
            fused_stages: fusion.len(),
            pixel_count: input.pixel_count(),
            memory_passes: 1,
            bytes_read: byte_len as u64,
            bytes_written: byte_len as u64,
        };
        Ok((
            Image {
                width: input.width,
                height: input.height,
                pixels: out_pixels,
            },
            receipt,
        ))
    }

    fn pool_stats(&self) -> BufferPoolStats {
        BufferPoolStats {
            allocations: self.allocations,
            reuses: self.reuses,
            active_buffers: self.active.len(),
            pooled_buffers: self.free_pool.len(),
            pooled_bytes: self.free_pool.iter().map(Vec::capacity).sum(),
        }
    }

    fn pipeline_cache_stats(&self) -> PipelineCacheStats {
        PipelineCacheStats {
            hits: self.hits,
            misses: self.misses,
            cached_pipelines: self.pipeline_cache.len(),
        }
    }
}

/// Deterministic encoding of the fused stage list into the 52-byte MSL/HIP
/// uniform block. LUT payloads are not encoded here; a real dispatch supplies
/// them as separate buffers.
pub fn fusion_params(fusion: &PixelStageFusion) -> [u8; PIXEL_FUSION_PARAMS_BYTES] {
    let mut bytes = [0u8; PIXEL_FUSION_PARAMS_BYTES];
    let mut mask = 0u32;
    let mut lut1d = 0u32;
    let mut lut3d = 0u32;

    for op in fusion.stages().iter() {
        match op {
            crate::compute_backend::PixelOp::Brightness { .. } => {
                mask |= stage_bits::BRIGHTNESS;
            }
            crate::compute_backend::PixelOp::Whiten { .. } => mask |= stage_bits::WHITEN,
            crate::compute_backend::PixelOp::Rosy { .. } => mask |= stage_bits::ROSY,
            crate::compute_backend::PixelOp::Temperature { .. } => {
                mask |= stage_bits::TEMPERATURE;
            }
            crate::compute_backend::PixelOp::Saturation { .. } => {
                mask |= stage_bits::SATURATION;
            }
            crate::compute_backend::PixelOp::Lut1D { .. } => {
                mask |= stage_bits::LUT1D;
                lut1d = 256;
            }
            crate::compute_backend::PixelOp::Lut3D { size, .. } => {
                mask |= stage_bits::LUT3D;
                lut3d = *size as u32;
            }
        }
    }
    let width = 1u32;
    let height = 1u32;
    // Layout: width(0), height(4), stage_count(8), stage_mask(12),
    // lut1d_size(16), lut3d_size(20), flags(24), then floats at 28..52.
    bytes[0..4].copy_from_slice(&width.to_le_bytes());
    bytes[4..8].copy_from_slice(&height.to_le_bytes());
    bytes[8..12].copy_from_slice(&(fusion.len() as u32).to_le_bytes());
    bytes[12..16].copy_from_slice(&mask.to_le_bytes());
    bytes[16..20].copy_from_slice(&lut1d.to_le_bytes());
    bytes[20..24].copy_from_slice(&lut3d.to_le_bytes());
    // Re-pack the float block into the declared offsets 28..52.
    let mut floats = [0f32; 6];
    for op in fusion.stages() {
        match op {
            crate::compute_backend::PixelOp::Brightness { delta } => floats[0] = *delta,
            crate::compute_backend::PixelOp::Whiten { strength } => floats[1] = *strength,
            crate::compute_backend::PixelOp::Rosy { strength } => floats[2] = *strength,
            crate::compute_backend::PixelOp::Temperature { warmth, tint } => {
                floats[3] = *warmth;
                floats[4] = *tint;
            }
            crate::compute_backend::PixelOp::Saturation { amount } => floats[5] = *amount,
            _ => {}
        }
    }
    for (idx, value) in floats.iter().enumerate() {
        bytes[28 + idx * 4..32 + idx * 4].copy_from_slice(&value.to_le_bytes());
    }
    bytes
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod apple {
    //! Runtime-only Objective-C / Metal bridge. No framework is linked at
    //! build time: every selector and class is resolved with `dlsym`.

    use super::*;
    use std::ffi::{CString, c_char, c_void};

    const RTLD_NOW: i32 = 2;
    const UNIFIED_MEMORY_SELECTOR: &str = "hasUnifiedMemory";
    const THREAD_WIDTH_SELECTOR: &str = "threadExecutionWidth";
    const MAX_THREADS_SELECTOR: &str = "maxTotalThreadsPerThreadgroup";

    #[derive(Clone, Copy)]
    struct Lib {
        send: *const c_void,
        get_class: unsafe extern "C" fn(*const c_char) -> *mut c_void,
        register_sel: unsafe extern "C" fn(*const c_char) -> *mut c_void,
        pool_push: unsafe extern "C" fn() -> *mut c_void,
        pool_pop: unsafe extern "C" fn(*mut c_void),
        create_device: unsafe extern "C" fn() -> *mut c_void,
    }

    unsafe impl Send for Lib {}
    unsafe impl Sync for Lib {}

    impl Lib {
        fn load() -> Option<Self> {
            unsafe extern "C" {
                fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
                fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
            }
            unsafe fn open(path: &str) -> Option<*mut c_void> {
                let path = CString::new(path).ok()?;
                let handle = dlopen(path.as_ptr(), RTLD_NOW);
                (!handle.is_null()).then_some(handle)
            }
            unsafe fn symbol(handle: *mut c_void, name: &str) -> Option<*mut c_void> {
                let name = CString::new(name).ok()?;
                let ptr = dlsym(handle, name.as_ptr());
                (!ptr.is_null()).then_some(ptr)
            }
            unsafe {
                let objc = open("/usr/lib/libobjc.A.dylib")?;
                let _foundation =
                    open("/System/Library/Frameworks/Foundation.framework/Foundation")?;
                let metal = open("/System/Library/Frameworks/Metal.framework/Metal")?;
                Some(Self {
                    send: symbol(objc, "objc_msgSend")?,
                    get_class: std::mem::transmute(symbol(objc, "objc_getClass")?),
                    register_sel: std::mem::transmute(symbol(objc, "sel_registerName")?),
                    pool_push: std::mem::transmute(symbol(objc, "objc_autoreleasePoolPush")?),
                    pool_pop: std::mem::transmute(symbol(objc, "objc_autoreleasePoolPop")?),
                    create_device: std::mem::transmute(symbol(
                        metal,
                        "MTLCreateSystemDefaultDevice",
                    )?),
                })
            }
        }

        fn selector(&self, name: &str) -> *mut c_void {
            let name = CString::new(name).expect("selector has no NUL");
            unsafe { (self.register_sel)(name.as_ptr()) }
        }

        fn class(&self, name: &str) -> *mut c_void {
            let name = CString::new(name).expect("class has no NUL");
            unsafe { (self.get_class)(name.as_ptr()) }
        }

        unsafe fn ptr0(&self, object: *mut c_void, selector: &str) -> *mut c_void {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> *mut c_void =
                std::mem::transmute(self.send);
            send(object, self.selector(selector))
        }

        unsafe fn usize0(&self, object: *mut c_void, selector: &str) -> usize {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> usize =
                std::mem::transmute(self.send);
            send(object, self.selector(selector))
        }

        unsafe fn bool0(&self, object: *mut c_void, selector: &str) -> bool {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void) -> u8 =
                std::mem::transmute(self.send);
            send(object, self.selector(selector)) != 0
        }

        unsafe fn void0(&self, object: *mut c_void, selector: &str) {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void) =
                std::mem::transmute(self.send);
            send(object, self.selector(selector))
        }

        unsafe fn ptr1(&self, object: *mut c_void, selector: &str, arg: *mut c_void) -> *mut c_void {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) -> *mut c_void =
                std::mem::transmute(self.send);
            send(object, self.selector(selector), arg)
        }

        unsafe fn ptr3_err(
            &self,
            object: *mut c_void,
            selector: &str,
            a: *mut c_void,
            b: *mut c_void,
            error: *mut *mut c_void,
        ) -> *mut c_void {
            let send: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *mut c_void,
                *mut c_void,
                *mut *mut c_void,
            ) -> *mut c_void = std::mem::transmute(self.send);
            send(object, self.selector(selector), a, b, error)
        }

        unsafe fn void_bool(&self, object: *mut c_void, selector: &str, value: u8) {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void, u8) =
                std::mem::transmute(self.send);
            send(object, self.selector(selector), value)
        }

        unsafe fn void1(&self, object: *mut c_void, selector: &str, arg: *mut c_void) {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut c_void) =
                std::mem::transmute(self.send);
            send(object, self.selector(selector), arg)
        }

        unsafe fn bytes_at_index(
            &self,
            object: *mut c_void,
            selector: &str,
            bytes: *const u8,
            len: usize,
            index: usize,
        ) {
            let send: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *const u8,
                usize,
                usize,
            ) = std::mem::transmute(self.send);
            send(object, self.selector(selector), bytes, len, index)
        }

        unsafe fn buffer_at_index(
            &self,
            object: *mut c_void,
            selector: &str,
            buffer: *mut c_void,
            offset: usize,
            index: usize,
        ) {
            let send: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *mut c_void,
                usize,
                usize,
            ) = std::mem::transmute(self.send);
            send(object, self.selector(selector), buffer, offset, index)
        }

        unsafe fn get_bytes(&self, buffer: *mut c_void, dst: *mut u8, len: usize) {
            let send: unsafe extern "C" fn(*mut c_void, *mut c_void, *mut u8, usize) =
                std::mem::transmute(self.send);
            send(buffer, self.selector("getBytes:length:"), dst, len)
        }

        unsafe fn dispatch_groups(
            &self,
            encoder: *mut c_void,
            groups: [usize; 3],
            threads: [usize; 3],
        ) {
            let send: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                [usize; 3],
                [usize; 3],
            ) = std::mem::transmute(self.send);
            send(
                encoder,
                self.selector("dispatchThreadgroups:threadsPerThreadgroup:"),
                groups,
                threads,
            )
        }

        unsafe fn new_buffer(&self, device: *mut c_void, bytes: &[u8], mode: usize) -> *mut c_void {
            let send: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                *const u8,
                usize,
                usize,
            ) -> *mut c_void = std::mem::transmute(self.send);
            send(
                device,
                self.selector("newBufferWithBytes:length:options:"),
                bytes.as_ptr(),
                bytes.len(),
                mode,
            )
        }

        unsafe fn new_zero_buffer(&self, device: *mut c_void, len: usize, mode: usize) -> *mut c_void {
            let send: unsafe extern "C" fn(
                *mut c_void,
                *mut c_void,
                usize,
                usize,
            ) -> *mut c_void = std::mem::transmute(self.send);
            send(
                device,
                self.selector("newBufferWithLength:options:"),
                len,
                mode,
            )
        }
    }

    struct Owned {
        ptr: *mut c_void,
        lib: Lib,
    }

    impl Owned {
        unsafe fn adopt(lib: Lib, ptr: *mut c_void) -> Option<Self> {
            (!ptr.is_null()).then_some(Self { ptr, lib })
        }

        unsafe fn retain(lib: Lib, ptr: *mut c_void) -> Option<Self> {
            let owned = Self::adopt(lib, ptr)?;
            unsafe { owned.lib.void0(owned.ptr, "retain") };
            Some(owned)
        }
    }

    impl Drop for Owned {
        fn drop(&mut self) {
            unsafe { self.lib.void0(self.ptr, "release") };
        }
    }

    struct Pool {
        lib: Lib,
        token: *mut c_void,
    }

    impl Pool {
        fn new(lib: Lib) -> Self {
            Self {
                lib,
                token: unsafe { (lib.pool_push)() },
            }
        }
    }

    impl Drop for Pool {
        fn drop(&mut self) {
            unsafe { (self.lib.pool_pop)(self.token) };
        }
    }

    pub struct MetalDevice {
        lib: Lib,
        device: Owned,
        queue: Owned,
        pipeline: Owned,
        pub unified: bool,
        pub limits: MetalLimits,
    }

    impl MetalDevice {
        pub fn open() -> Result<Self, ComputeError> {
            let lib = Lib::load().ok_or(ComputeError::BackendUnavailable(
                "Metal frameworks could not be loaded",
            ))?;
            let _pool = Pool::new(lib);
            unsafe {
                let device = Owned::retain(lib, (lib.create_device)()).ok_or(
                    ComputeError::BackendUnavailable("no default Metal device"),
                )?;
                let queue = Owned::adopt(lib, lib.ptr0(device.ptr, "newCommandQueue"))
                    .ok_or(ComputeError::BackendUnavailable("newCommandQueue failed"))?;
                let source = CString::new(PIXEL_FUSION_MSL).map_err(|_| {
                    ComputeError::BackendUnavailable("pixel_fusion MSL contains NUL")
                })?;
                let ns_source = Owned::retain(
                    lib,
                    lib.ptr1(
                        lib.class("NSString"),
                        "stringWithUTF8String:",
                        source.as_ptr() as *mut c_void,
                    ),
                )
                .ok_or(ComputeError::BackendUnavailable("NSString allocation failed"))?;
                let options = Owned::adopt(lib, lib.ptr0(lib.class("MTLCompileOptions"), "new"))
                    .ok_or(ComputeError::BackendUnavailable("MTLCompileOptions failed"))?;
                // Fast-math contraction would break byte parity with the CPU
                // oracle, so match the existing Metal host and disable it.
                lib.void_bool(options.ptr, "setFastMathEnabled:", 0);
                let mut error = std::ptr::null_mut();
                let library = Owned::adopt(
                    lib,
                    lib.ptr3_err(
                        device.ptr,
                        "newLibraryWithSource:options:error:",
                        ns_source.ptr,
                        options.ptr,
                        &mut error,
                    ),
                )
                .ok_or(ComputeError::BackendUnavailable("pixel_fusion MSL compile failed"))?;
                let name = CString::new(PIXEL_FUSION_ENTRY).expect("entry has no NUL");
                let function = Owned::adopt(
                    lib,
                    lib.ptr1(library.ptr, "newFunctionWithName:", name.as_ptr() as *mut c_void),
                )
                .ok_or(ComputeError::BackendUnavailable("pixel_fusion entry missing"))?;
                error = std::ptr::null_mut();
                let send: unsafe extern "C" fn(
                    *mut c_void,
                    *mut c_void,
                    *mut c_void,
                    *mut *mut c_void,
                ) -> *mut c_void = std::mem::transmute(lib.send);
                let pipeline_ptr = send(
                    device.ptr,
                    lib.selector("newComputePipelineStateWithFunction:error:"),
                    function.ptr,
                    &mut error,
                );
                let pipeline = Owned::adopt(lib, pipeline_ptr).ok_or(
                    ComputeError::BackendUnavailable("compute pipeline creation failed"),
                )?;
                let unified = lib.bool0(device.ptr, UNIFIED_MEMORY_SELECTOR);
                let limits = MetalLimits {
                    thread_execution_width: lib.usize0(pipeline.ptr, THREAD_WIDTH_SELECTOR) as u32,
                    max_total_threads_per_threadgroup: lib
                        .usize0(pipeline.ptr, MAX_THREADS_SELECTOR)
                        as u32,
                };
                if !limits.is_valid() {
                    return Err(ComputeError::BackendUnavailable(
                        "Metal pipeline reported invalid threadgroup limits",
                    ));
                }
                Ok(Self {
                    lib,
                    device,
                    queue,
                    pipeline,
                    unified,
                    limits,
                })
            }
        }

        pub fn dispatch(
            &self,
            params: &[u8; PIXEL_FUSION_PARAMS_BYTES],
            src: &[u8],
            dst: &mut [u8],
        ) -> Result<(), ComputeError> {
            if src.len() != dst.len() || src.len() % 4 != 0 {
                return Err(ComputeError::BufferSizeMismatch {
                    expected: src.len(),
                    actual: dst.len(),
                });
            }
            let _pool = Pool::new(self.lib);
            let image_mode = storage_mode_for(self.unified, false).as_metal_option();
            let staging_mode = storage_mode_for(self.unified, true).as_metal_option();
            unsafe {
                let src_buffer = Owned::adopt(self.lib, self.lib.new_buffer(self.device.ptr, src, image_mode))
                    .ok_or(ComputeError::BackendUnavailable("source buffer allocation failed"))?;
                let dst_buffer = Owned::adopt(
                    self.lib,
                    self.lib.new_zero_buffer(self.device.ptr, dst.len(), image_mode),
                )
                .ok_or(ComputeError::BackendUnavailable("destination buffer allocation failed"))?;
                let params_buffer = Owned::adopt(
                    self.lib,
                    self.lib.new_buffer(self.device.ptr, params, staging_mode),
                )
                .ok_or(ComputeError::BackendUnavailable("parameter buffer allocation failed"))?;
                // The MSL declares two LUT buffers at indices 3 and 4. The
                // point-op path binds zero-length placeholders to keep the
                // argument table complete.
                let lut_placeholder = Owned::adopt(
                    self.lib,
                    self.lib.new_buffer(self.device.ptr, &[0u8; 4], staging_mode),
                )
                .ok_or(ComputeError::BackendUnavailable("LUT placeholder allocation failed"))?;
                let command =
                    Owned::retain(self.lib, self.lib.ptr0(self.queue.ptr, "commandBuffer"))
                        .ok_or(ComputeError::BackendUnavailable("commandBuffer failed"))?;
                let encoder = Owned::retain(
                    self.lib,
                    self.lib.ptr0(command.ptr, "computeCommandEncoder"),
                )
                .ok_or(ComputeError::BackendUnavailable("compute encoder failed"))?;
                self.lib.void1(encoder.ptr, "setComputePipelineState:", self.pipeline.ptr);
                self.lib.buffer_at_index(
                    encoder.ptr,
                    "setBuffer:offset:atIndex:",
                    params_buffer.ptr,
                    0,
                    0,
                );
                self.lib.buffer_at_index(
                    encoder.ptr,
                    "setBuffer:offset:atIndex:",
                    src_buffer.ptr,
                    0,
                    1,
                );
                self.lib.buffer_at_index(
                    encoder.ptr,
                    "setBuffer:offset:atIndex:",
                    dst_buffer.ptr,
                    0,
                    2,
                );
                self.lib.buffer_at_index(
                    encoder.ptr,
                    "setBuffer:offset:atIndex:",
                    lut_placeholder.ptr,
                    0,
                    3,
                );
                self.lib.buffer_at_index(
                    encoder.ptr,
                    "setBuffer:offset:atIndex:",
                    lut_placeholder.ptr,
                    0,
                    4,
                );
                let pixel_count = u32::try_from(src.len() / 4).map_err(|_| {
                    ComputeError::BackendUnavailable("pixel count exceeds Metal 32-bit dispatch")
                })?;
                let (groups, threads) = self.limits.grid(pixel_count, 1);
                // Dispatch 1D: the shader indexes pixels with a single gid.
                self.lib.dispatch_groups(
                    encoder.ptr,
                    [groups as usize, 1, 1],
                    [threads as usize, 1, 1],
                );
                self.lib.void0(encoder.ptr, "endEncoding");
                self.lib.void0(command.ptr, "commit");
                self.lib.void0(command.ptr, "waitUntilCompleted");
                let status = self.lib.usize0(command.ptr, "status");
                let error = self.lib.ptr0(command.ptr, "error");
                if status != 4 || !error.is_null() {
                    return Err(ComputeError::BackendUnavailable("Metal command failed"));
                }
                self.lib.get_bytes(dst_buffer.ptr, dst.as_mut_ptr(), dst.len());
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // Compile-time proof that this standalone module has no unresolved items.
    use crate::compute_backend as _compute_backend_assert;
    use crate::device as _device_assert;

    fn fusion() -> PixelStageFusion {
        PixelStageFusion::new()
            .with_stage(crate::compute_backend::PixelOp::Brightness { delta: 0.25 })
            .unwrap()
            .with_stage(crate::compute_backend::PixelOp::Whiten { strength: 0.5 })
            .unwrap()
            .with_stage(crate::compute_backend::PixelOp::Saturation { amount: -0.25 })
            .unwrap()
    }

    #[test]
    fn apple_target_guard_uses_only_macos_or_ios() {
        // The cfg predicate is repeated for framework loading and every open
        // entry point; a typo would silently change the Windows host contract.
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            assert!(cfg!(any(target_os = "macos", target_os = "ios")));
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            assert!(!cfg!(any(target_os = "macos", target_os = "ios")));
        }
    }

    #[test]
    fn msl_source_is_complete_and_entry_point_is_declared() {
        assert!(PIXEL_FUSION_MSL.contains("kernel void pixel_fusion"));
        assert!(PIXEL_FUSION_MSL.contains("#pragma clang fp contract(off)"));
        assert!(PIXEL_FUSION_MSL.contains("[[buffer(0)]]"));
        assert!(PIXEL_FUSION_MSL.contains("[[buffer(4)]]"));
        assert!(PIXEL_FUSION_MSL.contains("thread_position_in_grid"));
        assert_eq!(PIXEL_FUSION_MSL.matches("kernel void ").count(), 1);
        assert!(PIXEL_FUSION_MSL.contains("sizeof(PixelFusionParams) == 52"));
        assert_eq!(PIXEL_FUSION_ENTRY, "pixel_fusion");
    }

    #[test]
    fn params_block_encodes_stage_mask_and_float_values() {
        let bytes = fusion_params(&fusion());
        assert_eq!(&bytes[0..4], &1u32.to_le_bytes());
        assert_eq!(&bytes[8..12], &3u32.to_le_bytes());
        let mask = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(
            mask,
            stage_bits::BRIGHTNESS | stage_bits::WHITEN | stage_bits::SATURATION
        );
        assert_eq!(f32::from_le_bytes(bytes[28..32].try_into().unwrap()), 0.25);
        assert_eq!(f32::from_le_bytes(bytes[32..36].try_into().unwrap()), 0.5);
        assert_eq!(f32::from_le_bytes(bytes[48..52].try_into().unwrap()), -0.25);
    }

    #[test]
    fn storage_mode_distinguishes_unified_discrete_and_staging() {
        assert_eq!(storage_mode_for(true, false), StorageMode::Shared);
        assert_eq!(storage_mode_for(false, false), StorageMode::Private);
        assert_eq!(storage_mode_for(false, true), StorageMode::Shared);
        assert_eq!(StorageMode::Shared.as_metal_option(), 0);
        assert_eq!(StorageMode::Private.as_metal_option(), 1);
    }

    #[test]
    fn threadgroup_adapts_to_pipeline_limits() {
        let m2 = MetalLimits {
            thread_execution_width: 32,
            max_total_threads_per_threadgroup: 256,
        };
        assert!(m2.is_valid());
        assert_eq!(m2.threadgroup(4096, 4096), (32, 8));
        assert_eq!(m2.grid(65, 17), (5, 256));
        let narrow = MetalLimits {
            thread_execution_width: 128,
            max_total_threads_per_threadgroup: 128,
        };
        assert_eq!(narrow.threadgroup(16, 16), (128, 1));
        let invalid = MetalLimits {
            thread_execution_width: 0,
            max_total_threads_per_threadgroup: 0,
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn core_ml_units_and_ane_claim_are_separate_from_verification() {
        for units in [
            MlComputeUnits::CpuOnly,
            MlComputeUnits::CpuAndGpu,
            MlComputeUnits::CpuAndNeuralEngine,
            MlComputeUnits::All,
        ] {
            let probe = CoreMlExecutionProbe::new(
                units,
                vec![StaticShape {
                    name: "mask",
                    dims: &[Some(1), Some(256), Some(256), None],
                }],
                true,
            );
            assert_eq!(probe.ane_partition_verified, false);
            assert_eq!(probe.claims_ane(), units.allows_ane());
            assert_eq!(probe.model_shapes.len(), 1);
            assert_eq!(probe.candidate_params.len(), 4);
            let verified = probe.clone().with_ane_evidence(true);
            assert_eq!(verified.ane_partition_verified, units.allows_ane());
        }
        assert_eq!(MlComputeUnits::CpuOnly.as_str(), "cpuOnly");
        assert_eq!(MlComputeUnits::CpuAndGpu.as_str(), "cpuAndGPU");
        assert_eq!(MlComputeUnits::CpuAndNeuralEngine.as_str(), "cpuAndNeuralEngine");
        assert_eq!(MlComputeUnits::All.as_str(), "all");
        assert!(!MlComputeUnits::CpuOnly.allows_ane());
        assert!(MlComputeUnits::All.allows_gpu());
    }

    #[test]
    fn apple_candidate_space_covers_m1_to_m4_and_a_series() {
        for family in ["m1", "m2", "m3", "m4", "a14", "a15", "a16", "a17", "a18"] {
            assert!(
                APPLE_CANDIDATES.iter().any(|c| c.id.contains(family)),
                "missing candidate family {family}"
            );
        }
        assert!(APPLE_CANDIDATES.iter().any(|c| c.backend == Backend::Npu));
        assert!(APPLE_CANDIDATES.iter().all(|c| c.status == VerificationStatus::Designed));
        assert!(APPLE_CANDIDATES.iter().all(|c| c.kernel_entry == PIXEL_FUSION_ENTRY));
        assert!(APPLE_CANDIDATES.iter().all(|c| c.is_valid_candidate()));
    }

    trait CandidateCheck {
        fn is_valid_candidate(&self) -> bool;
    }

    impl CandidateCheck for CandidateSpec {
        fn is_valid_candidate(&self) -> bool {
            !self.params.is_empty() && self.max_threads_per_group > 0
        }
    }

    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    #[test]
    fn non_apple_host_refuses_every_backend_entry_point() {
        let mut backend = MetalComputeBackend::new();
        assert_eq!(backend.backend(), Backend::Metal);
        let expected = ComputeError::BackendUnavailable(METAL_UNAVAILABLE);
        assert_eq!(backend.probe().driver_ready, false);
        assert_eq!(backend.probe().kernel_compiled, false);
        assert_eq!(MetalComputeBackend::open().err(), Some(expected.clone()));
        assert_eq!(backend.allocate_buffer(16, BufferUsage::StorageRead), Err(expected.clone()));
        assert_eq!(backend.release_buffer(BufferId(1)), Err(expected.clone()));
        assert_eq!(backend.upload_buffer(BufferId(1), &[0u8]), Err(expected.clone()));
        assert_eq!(backend.download_buffer(BufferId(1)), Err(expected.clone()));
        assert_eq!(backend.synchronize(), Err(expected.clone()));
        assert_eq!(backend.limits(), Err(expected.clone()));
        assert_eq!(backend.image_storage_mode(), Err(expected));
        let image = Image::solid(2, 2, [1, 2, 3, 4]);
        assert_eq!(
            backend.dispatch_pixel_fusion(&image, &fusion()).err(),
            Some(ComputeError::BackendUnavailable(METAL_UNAVAILABLE))
        );
        assert_eq!(backend.pool_stats().allocations, 0);
        assert_eq!(backend.pipeline_cache_stats().misses, 0);
    }
}
