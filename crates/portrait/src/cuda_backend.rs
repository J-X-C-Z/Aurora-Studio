//! CUDA implementation of the unified [`ComputeBackend`] and the single-pass
//! [`PixelStageFusion`] dispatch.
//!
//! The kernel lives in `native/cuda/pixel_fusion.cu` and is compiled to
//! `native/cuda/pixel_fusion.ptx` with `-fmad=false`; the driver, context,
//! JIT module cache and buffer pool are the shared ones in [`crate::cuda`].
//! This module owns only the CUDA-specific side of the platform-neutral
//! contract:
//!
//! * the kernel argument block (`PixelFusionUniforms`) and the LUT blob layout,
//! * the launch configuration (`block_x` / `block_y` / `vector_load`) and its
//!   [`CandidateSpec`] for AutoTune,
//! * a host staging pool plus a pipeline cache with the same bookkeeping as
//!   [`CpuComputeBackend`], so the two backends are directly comparable,
//! * host-precise [`BackendTiming`] and the last real [`Dispatch`] evidence.
//!
//! Honesty rules this module keeps:
//!
//! * `PixelStageFusion::pipeline_key()` hashes stage *kinds*, not parameters,
//!   so the pipeline cache only ever caches a handle identity; the op
//!   descriptor array and LUT blob are rebuilt on every dispatch.
//! * A missing driver, an unbuilt PTX image or a driver error is never
//!   reported as a CUDA success. In [`ExecutionMode::Strict`] the dispatch
//!   fails with [`ComputeError::BackendUnavailable`]; in
//!   [`ExecutionMode::AutoFallback`] it runs the CPU reference and the
//!   [`ExecutionReceipt`] says `Backend::Cpu`.
//! * Device-resident buffers come from the crate-wide named pool in
//!   [`crate::cuda`] (exact-shape reuse, one group per source/LUT/destination
//!   triple); the per-buffer pool this type exposes is the host staging pool,
//!   exactly as `CpuComputeBackend` pools host buffers.

use std::collections::HashMap;
use std::ffi::c_void;
use std::time::Instant;

use crate::autotune::{
    BackendTiming, CandidateSpec, ExecutionMode, ParamBound, VerificationStatus,
};
use crate::compute_backend::{
    BufferId, BufferPoolStats, BufferUsage, ComputeBackend, ComputeError, CpuComputeBackend,
    ExecutionReceipt, PipelineCacheStats, PipelineHandle, PipelineKey, PixelOp, PixelStageFusion,
};
use crate::cuda::{CudaError, Dispatch, Nvidia};
use crate::device::{Backend, DeviceProbe, Precision};
use crate::{Image, Refusal};

/// Fused pixel-stage kernel entry point (`native/cuda/pixel_fusion.cu`).
pub const PIXEL_FUSION_KERNEL: &str = "pixel_fusion_kernel";

/// JIT image for [`PIXEL_FUSION_KERNEL`].
pub const PIXEL_FUSION_PTX: &str = include_str!("../native/cuda/pixel_fusion.ptx");

/// Marker the checked-in placeholder image carries. Never a success path.
pub const PTX_PLACEHOLDER_MARKER: &str = "AURORA-PTX-PLACEHOLDER";

/// Deepest chain the kernel's by-value descriptor array can hold.
pub const MAX_FUSED_STAGES: usize = 16;

/// `PixelFusionOp::kind` values, field-for-field with the `.cu` `#define`s.
pub const OP_BRIGHTNESS: i32 = 0;
pub const OP_WHITEN: i32 = 1;
pub const OP_ROSY: i32 = 2;
pub const OP_TEMPERATURE: i32 = 3;
pub const OP_SATURATION: i32 = 4;
pub const OP_LUT1D: i32 = 5;
pub const OP_LUT3D: i32 = 6;

/// Largest frame the backend advertises. The kernel itself is bounds-checked
/// and has no such limit; this is the probe's declared ceiling.
const MAX_IMAGE_DIMENSION: u32 = 16_384;

/// Name of the shared device-buffer group used by this backend.
const FUSION_POOL: &str = "cuda.pixel_fusion";

/// One encoded [`PixelOp`] inside the kernel's argument block.
///
/// Mirrors `PixelFusionOp` in `native/cuda/pixel_fusion.cu`: 20 bytes, align 4.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FusionOp {
    /// One of [`OP_BRIGHTNESS`] .. [`OP_LUT3D`].
    pub kind: i32,
    /// `Lut1D`: byte offset into the LUT blob. `Lut3D`: float offset.
    pub arg0: i32,
    /// `Lut3D`: cube size. Unused otherwise.
    pub arg1: i32,
    /// First already-clamped parameter.
    pub p0: f32,
    /// Second already-clamped parameter.
    pub p1: f32,
}

/// Kernel argument block for [`PIXEL_FUSION_KERNEL`].
///
/// Mirrors `PixelFusionUniforms`: 336 bytes, align 4.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FusionUniforms {
    pub width: i32,
    pub height: i32,
    pub op_count: i32,
    /// Pixels per thread: 1, 2 or 4.
    pub vec_load: i32,
    pub ops: [FusionOp; MAX_FUSED_STAGES],
}

const _: () = assert!(std::mem::size_of::<FusionOp>() == 20);
const _: () = assert!(std::mem::size_of::<FusionUniforms>() == 336);

/// Tunable launch shape for [`PIXEL_FUSION_KERNEL`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FusionLaunchConfig {
    pub block_x: u32,
    pub block_y: u32,
    pub vector_load: u32,
}

impl Default for FusionLaunchConfig {
    fn default() -> Self {
        Self {
            block_x: 16,
            block_y: 16,
            vector_load: 1,
        }
    }
}

impl FusionLaunchConfig {
    /// Restricts the tunables to the values the kernel's guards are written for.
    pub fn validate(&self) -> Result<(), ComputeError> {
        if !matches!(self.block_x, 8 | 16 | 32) {
            return Err(ComputeError::InvalidParameter(
                "pixel fusion block_x must be 8, 16 or 32",
            ));
        }
        if !matches!(self.block_y, 4 | 8 | 16 | 32) {
            return Err(ComputeError::InvalidParameter(
                "pixel fusion block_y must be 4, 8, 16 or 32",
            ));
        }
        if !matches!(self.vector_load, 1 | 2 | 4) {
            return Err(ComputeError::InvalidParameter(
                "pixel fusion vector_load must be 1, 2 or 4",
            ));
        }
        if self.block_x * self.block_y > 1024 {
            return Err(ComputeError::InvalidParameter(
                "pixel fusion block must not exceed 1024 threads",
            ));
        }
        Ok(())
    }

    /// Launch geometry for one image.
    pub fn grid(&self, width: u32, height: u32) -> (u32, u32) {
        let span_x = self.block_x.saturating_mul(self.vector_load.max(1)).max(1);
        let span_y = self.block_y.max(1);
        (width.div_ceil(span_x).max(1), height.div_ceil(span_y).max(1))
    }

    /// `(block_x, block_y, vector_load)` as AutoTune parameters.
    pub fn parameter_values(&self) -> (u32, u32, u32) {
        (self.block_x, self.block_y, self.vector_load)
    }
}

const BLOCK_X_CHOICES: &[u32] = &[8, 16, 32];
const BLOCK_Y_CHOICES: &[u32] = &[4, 8, 16, 32];
const VECTOR_LOAD_CHOICES: &[u32] = &[1, 2, 4];

const FUSION_PARAMS: &[ParamBound] = &[
    ParamBound::DiscreteU32 {
        name: "block_x",
        values: BLOCK_X_CHOICES,
    },
    ParamBound::DiscreteU32 {
        name: "block_y",
        values: BLOCK_Y_CHOICES,
    },
    ParamBound::DiscreteU32 {
        name: "vector_load",
        values: VECTOR_LOAD_CHOICES,
    },
];

/// AutoTune view of this backend's fused pixel kernel.
///
/// `Implemented` is deliberate: the ladder only reaches `Compiled` once the
/// checked-in PTX is a real `nvcc` output, and `SyntheticOk` / `DeviceOk` once
/// the parity test below actually ran on this host.
pub fn candidate_spec() -> CandidateSpec {
    CandidateSpec {
        id: "cuda.pixel_fusion.fp32",
        algorithm_id: "pixel_stage_fusion",
        backend: Backend::Cuda,
        kernel_entry: PIXEL_FUSION_KERNEL,
        precision: Precision::Fp32,
        params: FUSION_PARAMS,
        max_shared_mem_bytes: 0,
        max_threads_per_group: 1024,
        status: VerificationStatus::Implemented,
    }
}

/// True for the checked-in placeholder, i.e. before `nvcc` produced the module.
pub fn ptx_is_placeholder(ptx: &str) -> bool {
    ptx.contains(PTX_PLACEHOLDER_MARKER)
}

/// Opens the fused-pixel module, refusing a placeholder image.
pub fn open_pixel_fusion() -> Result<Nvidia, CudaError> {
    if ptx_is_placeholder(PIXEL_FUSION_PTX) {
        return Err(CudaError::Unavailable(
            "pixel_fusion.ptx is a placeholder (run nvcc -ptx -fmad=false)",
        ));
    }
    Nvidia::open_ptx(PIXEL_FUSION_PTX)
}

/// Whether a real fused-pixel dispatch can run on this host right now.
pub fn device_available() -> bool {
    open_pixel_fusion().is_ok()
}

/// Encodes a fusion into the kernel's descriptor array plus the LUT blob.
///
/// Parameters are clamped here because `PixelOp::apply_pixel` clamps them
/// internally; the kernel never sees an out-of-range value. LUT tables are
/// appended in stage order and every offset is recorded absolutely, so a
/// `Lut3D` before a `Lut1D` stays 4-byte aligned (768 and `12 * size^3` bytes
/// are both multiples of four).
pub fn encode_stages(fusion: &PixelStageFusion) -> Result<(Vec<FusionOp>, Vec<u8>), ComputeError> {
    let stages = fusion.stages();
    if stages.is_empty() {
        return Err(ComputeError::EmptyPipeline);
    }
    if stages.len() > MAX_FUSED_STAGES {
        return Err(ComputeError::InvalidParameter(
            "pixel stage fusion supports at most 16 stages",
        ));
    }

    let mut ops = Vec::with_capacity(stages.len());
    let mut blob: Vec<u8> = Vec::new();
    for op in stages {
        op.validate()?;
        let encoded = match op {
            PixelOp::Brightness { delta } => FusionOp {
                kind: OP_BRIGHTNESS,
                p0: delta.clamp(-1.0, 1.0),
                ..FusionOp::default()
            },
            PixelOp::Whiten { strength } => FusionOp {
                kind: OP_WHITEN,
                p0: strength.clamp(0.0, 1.0),
                ..FusionOp::default()
            },
            PixelOp::Rosy { strength } => FusionOp {
                kind: OP_ROSY,
                p0: strength.clamp(0.0, 1.0),
                ..FusionOp::default()
            },
            PixelOp::Temperature { warmth, tint } => FusionOp {
                kind: OP_TEMPERATURE,
                p0: warmth.clamp(-1.0, 1.0),
                p1: tint.clamp(-1.0, 1.0),
                ..FusionOp::default()
            },
            PixelOp::Saturation { amount } => FusionOp {
                kind: OP_SATURATION,
                p0: amount.clamp(-1.0, 1.0),
                ..FusionOp::default()
            },
            PixelOp::Lut1D { r, g, b } => {
                let offset = blob.len();
                blob.extend_from_slice(r.as_slice());
                blob.extend_from_slice(g.as_slice());
                blob.extend_from_slice(b.as_slice());
                FusionOp {
                    kind: OP_LUT1D,
                    arg0: i32::try_from(offset).map_err(|_| {
                        ComputeError::InvalidParameter("LUT blob offset exceeds i32")
                    })?,
                    ..FusionOp::default()
                }
            }
            PixelOp::Lut3D { size, table } => {
                let offset = blob.len();
                if offset % 4 != 0 {
                    return Err(ComputeError::InvalidParameter(
                        "3D LUT table is not 4-byte aligned in the blob",
                    ));
                }
                for entry in table {
                    for channel in entry {
                        blob.extend_from_slice(&channel.to_le_bytes());
                    }
                }
                let float_offset = i32::try_from(offset / 4).map_err(|_| {
                    ComputeError::InvalidParameter("3D LUT offset exceeds i32")
                })?;
                FusionOp {
                    kind: OP_LUT3D,
                    arg0: float_offset,
                    arg1: i32::try_from(*size).map_err(|_| {
                        ComputeError::InvalidParameter("3D LUT size exceeds i32")
                    })?,
                    ..FusionOp::default()
                }
            }
        };
        ops.push(encoded);
    }

    // The kernel always receives a readable blob pointer; an unused blob still
    // has to be a legal upload.
    if blob.is_empty() {
        blob.resize(4, 0);
    }
    Ok((ops, blob))
}

/// Builds the kernel argument block for one image.
pub fn fusion_uniforms(
    width: u32,
    height: u32,
    vector_load: u32,
    ops: &[FusionOp],
) -> Result<FusionUniforms, ComputeError> {
    if ops.len() > MAX_FUSED_STAGES {
        return Err(ComputeError::InvalidParameter(
            "pixel stage fusion supports at most 16 stages",
        ));
    }
    let width = i32::try_from(width)
        .map_err(|_| ComputeError::InvalidParameter("image width exceeds i32"))?;
    let height = i32::try_from(height)
        .map_err(|_| ComputeError::InvalidParameter("image height exceeds i32"))?;
    let vec_load = i32::try_from(vector_load)
        .map_err(|_| ComputeError::InvalidParameter("vector_load exceeds i32"))?;
    let mut block = FusionUniforms {
        width,
        height,
        op_count: ops.len() as i32,
        vec_load,
        ops: [FusionOp::default(); MAX_FUSED_STAGES],
    };
    block.ops[..ops.len()].copy_from_slice(ops);
    Ok(block)
}

/// Maps a driver failure onto the backend-neutral error vocabulary.
fn map_cuda_error(error: CudaError) -> ComputeError {
    match error {
        CudaError::Unavailable(what) => ComputeError::BackendUnavailable(what),
        CudaError::Driver(call, _) => ComputeError::BackendUnavailable(call),
        CudaError::Refused(refusal) => ComputeError::InvalidImage(refusal),
    }
}

/// One real fused pass: upload source + LUT, launch, read back.
///
/// The three device buffers are the crate-wide named pool's exact-size group,
/// so a repeated preview of the same image size re-uses the allocation.
fn run_fusion(
    gpu: &Nvidia,
    uniforms: &FusionUniforms,
    lut_blob: &[u8],
    source: &[u8],
    config: FusionLaunchConfig,
) -> Result<(Vec<u8>, Dispatch), CudaError> {
    let bytes = source.len();
    if bytes == 0 || bytes % 4 != 0 {
        return Err(CudaError::Refused(Refusal::ShapeMismatch("fusion source")));
    }
    if lut_blob.is_empty() {
        return Err(CudaError::Refused(Refusal::ShapeMismatch("fusion LUT blob")));
    }
    let sizes = [bytes, lut_blob.len(), bytes];
    let (grid_x, grid_y) = config.grid(uniforms.width.max(0) as u32, uniforms.height.max(0) as u32);
    let block = (config.block_x, config.block_y);

    gpu.with_reusable_buffers(FUSION_POOL, &sizes[..], |buffers| {
        buffers.upload(0, source)?;
        buffers.upload(1, lut_blob)?;

        let mut source_arg = buffers.handle(0);
        let mut lut_arg = buffers.handle(1);
        let mut output_arg = buffers.handle(2);
        let mut block_arg = *uniforms;
        let mut params: [*mut c_void; 4] = [
            &mut source_arg as *mut u64 as *mut c_void,
            &mut lut_arg as *mut u64 as *mut c_void,
            &mut output_arg as *mut u64 as *mut c_void,
            &mut block_arg as *mut FusionUniforms as *mut c_void,
        ];

        let dispatch =
            buffers.launch(PIXEL_FUSION_KERNEL, (grid_x, grid_y), block, &mut params)?;
        let mut output = vec![0u8; bytes];
        buffers.download(2, &mut output)?;
        buffers.mark_consumed();
        Ok((output, dispatch))
    })
}

#[derive(Debug)]
struct StagingEntry {
    data: Vec<u8>,
    usage: BufferUsage,
}

/// CUDA [`ComputeBackend`] for the fused pixel-stage pipeline.
///
/// Construction never touches the driver: the module is opened on the first
/// [`ComputeBackend::dispatch_pixel_fusion`] (or [`Self::probe`]) call, so a
/// build without a GPU still constructs and reports `driver_ready: false`.
pub struct CudaComputeBackend {
    gpu: Option<Nvidia>,
    mode: ExecutionMode,
    config: FusionLaunchConfig,
    next_buffer_id: u64,
    next_pipeline_id: u64,
    staging: HashMap<BufferId, StagingEntry>,
    free_pool: Vec<Vec<u8>>,
    allocations: u64,
    reuses: u64,
    pipelines: HashMap<PipelineKey, PipelineHandle>,
    cache_hits: u64,
    cache_misses: u64,
    last_timing: BackendTiming,
    last_dispatch: Option<Dispatch>,
    cpu_reference: CpuComputeBackend,
}

impl Default for CudaComputeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaComputeBackend {
    /// Strict backend: a missing device is an error, never a silent CPU run.
    pub fn new() -> Self {
        Self::with_config(FusionLaunchConfig::default())
    }

    /// Strict backend with an explicit AutoTune launch shape.
    pub fn with_config(config: FusionLaunchConfig) -> Self {
        Self {
            gpu: None,
            mode: ExecutionMode::Strict,
            config,
            next_buffer_id: 1,
            next_pipeline_id: 1,
            staging: HashMap::new(),
            free_pool: Vec::new(),
            allocations: 0,
            reuses: 0,
            pipelines: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
            last_timing: BackendTiming::default(),
            last_dispatch: None,
            cpu_reference: CpuComputeBackend::new(),
        }
    }

    /// Selects [`ExecutionMode::Strict`] (default) or
    /// [`ExecutionMode::AutoFallback`]. A fallback dispatch runs the CPU
    /// reference and labels its receipt `Backend::Cpu`.
    pub fn with_mode(config: FusionLaunchConfig, mode: ExecutionMode) -> Self {
        let mut backend = Self::with_config(config);
        backend.mode = mode;
        backend
    }

    /// Launch shape currently used for dispatches.
    pub fn config(&self) -> FusionLaunchConfig {
        self.config
    }

    /// Replaces the launch shape (rejected values keep the old one).
    pub fn set_config(&mut self, config: FusionLaunchConfig) -> Result<(), ComputeError> {
        config.validate()?;
        self.config = config;
        Ok(())
    }

    /// Host-side timing of the most recent successful dispatch.
    pub fn last_timing(&self) -> BackendTiming {
        self.last_timing
    }

    /// Evidence of the most recent real launch, if one happened.
    pub fn last_dispatch(&self) -> Option<&Dispatch> {
        self.last_dispatch.as_ref()
    }

    /// Drops every idle host staging buffer.
    pub fn trim_pool(&mut self) {
        self.free_pool.clear();
    }

    /// Opens the module on first use and remembers it.
    fn ensure_device(&mut self) -> Result<(), CudaError> {
        if self.gpu.is_none() {
            self.gpu = Some(open_pixel_fusion()?);
        }
        Ok(())
    }

    /// CPU reference run for [`ExecutionMode::AutoFallback`].
    fn fallback_pixel_fusion(
        &mut self,
        input: &Image,
        fusion: &PixelStageFusion,
        pipeline: PipelineHandle,
        error: CudaError,
    ) -> Result<(Image, ExecutionReceipt), ComputeError> {
        match self.mode {
            ExecutionMode::Strict => Err(map_cuda_error(error)),
            ExecutionMode::AutoFallback => {
                let (image, mut receipt) = self.cpu_reference.dispatch_pixel_fusion(input, fusion)?;
                receipt.pipeline = pipeline;
                receipt.backend = Backend::Cpu;
                Ok((image, receipt))
            }
        }
    }

    /// Real `cuCtxSynchronize` through the shared domain, where the driver
    /// layer exposes it. Every dispatch already ends in a blocking readback,
    /// so on other platforms there is nothing outstanding and `Ok(())` is the
    /// truthful answer.
    #[cfg(windows)]
    fn synchronize_device(&mut self) -> Result<(), ComputeError> {
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.suspend().map_err(map_cuda_error)?;
        }
        Ok(())
    }

    #[cfg(not(windows))]
    fn synchronize_device(&mut self) -> Result<(), ComputeError> {
        Ok(())
    }
}

impl ComputeBackend for CudaComputeBackend {
    fn backend(&self) -> Backend {
        Backend::Cuda
    }

    fn probe(&self) -> DeviceProbe {
        let ready = open_pixel_fusion().is_ok();
        DeviceProbe {
            backend: Backend::Cuda,
            driver_ready: ready,
            kernel_compiled: ready,
            fp32: true,
            fp16: false,
            max_width: MAX_IMAGE_DIMENSION,
            max_height: MAX_IMAGE_DIMENSION,
        }
    }

    fn allocate_buffer(
        &mut self,
        byte_len: usize,
        usage: BufferUsage,
    ) -> Result<BufferId, ComputeError> {
        let best = self
            .free_pool
            .iter()
            .enumerate()
            .filter(|(_, buffer)| buffer.capacity() >= byte_len)
            .min_by_key(|(_, buffer)| buffer.capacity())
            .map(|(index, _)| index);

        let data = match best {
            Some(index) => {
                let mut buffer = self.free_pool.swap_remove(index);
                buffer.clear();
                buffer.resize(byte_len, 0);
                self.reuses += 1;
                buffer
            }
            None => {
                self.allocations += 1;
                vec![0u8; byte_len]
            }
        };

        let id = BufferId(self.next_buffer_id);
        self.next_buffer_id += 1;
        self.staging.insert(id, StagingEntry { data, usage });
        Ok(id)
    }

    fn release_buffer(&mut self, id: BufferId) -> Result<(), ComputeError> {
        let entry = self
            .staging
            .remove(&id)
            .ok_or(ComputeError::BufferNotFound(id))?;
        self.free_pool.push(entry.data);
        Ok(())
    }

    fn upload_buffer(&mut self, id: BufferId, data: &[u8]) -> Result<(), ComputeError> {
        let entry = self
            .staging
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
        let entry = self
            .staging
            .get(&id)
            .ok_or(ComputeError::BufferNotFound(id))?;
        Ok(entry.data.clone())
    }

    fn get_or_compile_pipeline(
        &mut self,
        fusion: &PixelStageFusion,
    ) -> Result<PipelineHandle, ComputeError> {
        if fusion.is_empty() {
            return Err(ComputeError::EmptyPipeline);
        }
        if fusion.len() > MAX_FUSED_STAGES {
            return Err(ComputeError::InvalidParameter(
                "pixel stage fusion supports at most 16 stages",
            ));
        }
        let key = fusion.pipeline_key();
        if let Some(&handle) = self.pipelines.get(&key) {
            self.cache_hits += 1;
            return Ok(handle);
        }
        self.cache_misses += 1;
        let handle = PipelineHandle(self.next_pipeline_id);
        self.next_pipeline_id += 1;
        self.pipelines.insert(key, handle);
        Ok(handle)
    }

    fn synchronize(&mut self) -> Result<(), ComputeError> {
        self.synchronize_device()
    }

    fn dispatch_pixel_fusion(
        &mut self,
        input: &Image,
        fusion: &PixelStageFusion,
    ) -> Result<(Image, ExecutionReceipt), ComputeError> {
        input.validate().map_err(ComputeError::InvalidImage)?;
        let pipeline = self.get_or_compile_pipeline(fusion)?;
        let config = self.config;
        config.validate()?;

        // Rebuilt per dispatch: `pipeline_key` hashes stage kinds, not values.
        let (ops, lut_blob) = encode_stages(fusion)?;
        if let Err(error) = self.ensure_device() {
            return self.fallback_pixel_fusion(input, fusion, pipeline, error);
        }
        let uniforms = fusion_uniforms(input.width, input.height, config.vector_load, &ops)?;

        let started = Instant::now();
        let (pixels, dispatch) = {
            let gpu = match self.gpu.as_ref() {
                Some(gpu) => gpu,
                None => {
                    return Err(ComputeError::BackendUnavailable(
                        "CUDA device handle missing after open",
                    ));
                }
            };
            run_fusion(gpu, &uniforms, &lut_blob, &input.pixels, config).map_err(map_cuda_error)?
        };
        let wall_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;

        let byte_len = input.pixels.len() as u64;
        self.last_timing = BackendTiming {
            queue_submit_ns: 0,
            device_exec_ns: 0,
            wall_ns,
            h2d_bytes: byte_len + lut_blob.len() as u64,
            d2h_bytes: byte_len,
        };
        self.last_dispatch = Some(dispatch);

        let receipt = ExecutionReceipt {
            backend: Backend::Cuda,
            pipeline,
            fused_stages: fusion.len(),
            pixel_count: input.pixel_count(),
            memory_passes: 1,
            bytes_read: byte_len,
            bytes_written: byte_len,
        };
        Ok((
            Image {
                width: input.width,
                height: input.height,
                pixels,
            },
            receipt,
        ))
    }

    fn pool_stats(&self) -> BufferPoolStats {
        BufferPoolStats {
            allocations: self.allocations,
            reuses: self.reuses,
            active_buffers: self.staging.len(),
            pooled_buffers: self.free_pool.len(),
            pooled_bytes: self.free_pool.iter().map(Vec::capacity).sum(),
        }
    }

    fn pipeline_cache_stats(&self) -> PipelineCacheStats {
        PipelineCacheStats {
            hits: self.cache_hits,
            misses: self.cache_misses,
            cached_pipelines: self.pipelines.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_rgba8(width: u32, height: u32) -> Image {
        let boundary = [0u8, 1, 2, 63, 127, 128, 191, 253, 254, 255];
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) as usize;
                let (r, g, b, a) = if index < boundary.len() {
                    let v = boundary[index];
                    (
                        v,
                        255 - v,
                        v.wrapping_mul(37),
                        if index % 2 == 0 { 255 } else { 128 },
                    )
                } else {
                    (
                        ((x * 29 + y * 13) % 256) as u8,
                        ((x * 7 + y * 41 + 85) % 256) as u8,
                        ((x * 53 + y * 19 + 170) % 256) as u8,
                        ((index * 17) % 256) as u8,
                    )
                };
                pixels.extend_from_slice(&[r, g, b, a]);
            }
        }
        Image::new(width, height, pixels).expect("valid synthetic image")
    }

    fn s_curve_lut1d() -> PixelOp {
        let mut r = Box::new([0u8; 256]);
        let mut g = Box::new([0u8; 256]);
        let mut b = Box::new([0u8; 256]);
        for i in 0..256 {
            let x = i as f32 / 255.0;
            let s = x * x * (3.0 - 2.0 * x);
            r[i] = ((s * 1.04).min(1.0) * 255.0).round() as u8;
            g[i] = (s * 255.0).round() as u8;
            b[i] = ((s * 0.96) * 255.0).round() as u8;
        }
        PixelOp::Lut1D { r, g, b }
    }

    fn warm_lut3d(size: usize) -> PixelOp {
        let denom = (size - 1) as f32;
        let mut table = Vec::with_capacity(size * size * size);
        for bi in 0..size {
            for gi in 0..size {
                for ri in 0..size {
                    let r = ri as f32 / denom;
                    let g = gi as f32 / denom;
                    let b = bi as f32 / denom;
                    table.push([
                        (r * 1.05 + 0.02 * (1.0 - r)).clamp(0.0, 1.0),
                        g,
                        (b * 0.92).clamp(0.0, 1.0),
                    ]);
                }
            }
        }
        PixelOp::Lut3D { size, table }
    }

    fn portrait_chain() -> PixelStageFusion {
        PixelStageFusion::new()
            .with_stage(PixelOp::Brightness { delta: 0.08 })
            .unwrap()
            .with_stage(PixelOp::Whiten { strength: 0.65 })
            .unwrap()
            .with_stage(PixelOp::Rosy { strength: 0.45 })
            .unwrap()
            .with_stage(PixelOp::Temperature {
                warmth: 0.25,
                tint: -0.15,
            })
            .unwrap()
            .with_stage(PixelOp::Saturation { amount: 0.30 })
            .unwrap()
            .with_stage(s_curve_lut1d())
            .unwrap()
            .with_stage(warm_lut3d(9))
            .unwrap()
    }

    #[test]
    fn kernel_argument_block_matches_the_cuda_struct() {
        assert_eq!(std::mem::size_of::<FusionOp>(), 20);
        assert_eq!(std::mem::align_of::<FusionOp>(), 4);
        assert_eq!(std::mem::size_of::<FusionUniforms>(), 336);
        assert_eq!(std::mem::align_of::<FusionUniforms>(), 4);

        let (ops, _) = encode_stages(&portrait_chain()).unwrap();
        let uniform = fusion_uniforms(37, 23, 4, &ops).unwrap();
        assert_eq!((uniform.width, uniform.height), (37, 23));
        assert_eq!(uniform.op_count, 7);
        assert_eq!(uniform.vec_load, 4);
        assert_eq!(uniform.ops[0].kind, OP_BRIGHTNESS);
        assert_eq!(uniform.ops[6].kind, OP_LUT3D);
        assert_eq!(uniform.ops[6].arg1, 9);
    }

    #[test]
    fn encode_stages_clamps_parameters_and_rejects_bad_chains() {
        let (ops, blob) = encode_stages(
            &PixelStageFusion::from_stages(vec![
                PixelOp::Brightness { delta: 9.0 },
                PixelOp::Whiten { strength: -3.0 },
                PixelOp::Rosy { strength: 4.0 },
                PixelOp::Temperature {
                    warmth: -7.0,
                    tint: 7.0,
                },
                PixelOp::Saturation { amount: 5.0 },
            ])
            .unwrap(),
        )
        .unwrap();
        assert_eq!(ops[0].p0, 1.0);
        assert_eq!(ops[1].p0, 0.0);
        assert_eq!(ops[2].p0, 1.0);
        assert_eq!((ops[3].p0, ops[3].p1), (-1.0, 1.0));
        assert_eq!(ops[4].p0, 1.0);
        assert_eq!(blob.len(), 4, "an unused blob still uploads four bytes");

        assert_eq!(
            encode_stages(&PixelStageFusion::new()),
            Err(ComputeError::EmptyPipeline)
        );

        let too_deep: Vec<PixelOp> = (0..(MAX_FUSED_STAGES + 1))
            .map(|i| PixelOp::Brightness {
                delta: i as f32 * 0.01,
            })
            .collect();
        assert!(matches!(
            encode_stages(&PixelStageFusion::from_stages(too_deep).unwrap()),
            Err(ComputeError::InvalidParameter(_))
        ));
    }

    #[test]
    fn lut_tables_land_on_four_byte_offsets_in_stage_order() {
        let fusion = PixelStageFusion::from_stages(vec![
            warm_lut3d(3),
            s_curve_lut1d(),
            PixelOp::Brightness { delta: 0.1 },
            warm_lut3d(2),
        ])
        .unwrap();
        let (ops, blob) = encode_stages(&fusion).unwrap();
        assert_eq!(ops[0].kind, OP_LUT3D);
        assert_eq!(ops[0].arg0, 0);
        assert_eq!(ops[1].kind, OP_LUT1D);
        assert_eq!(ops[1].arg0, 3 * 27 * 4);
        assert_eq!(ops[2].kind, OP_BRIGHTNESS);
        assert_eq!(ops[3].kind, OP_LUT3D);
        assert_eq!(ops[3].arg0, 3 * 27 + 768 / 4);
        // 3*27 + 768/4 floats plus 3*8 floats, byte-exact.
        assert_eq!(blob.len(), 3 * 27 * 4 + 768 + 3 * 8 * 4);
        assert_eq!(blob.len() % 4, 0);
    }

    #[test]
    fn launch_config_rejects_unsupported_shapes() {
        assert!(FusionLaunchConfig::default().validate().is_ok());
        assert_eq!(
            FusionLaunchConfig {
                block_x: 12,
                ..FusionLaunchConfig::default()
            }
            .validate(),
            Err(ComputeError::InvalidParameter(
                "pixel fusion block_x must be 8, 16 or 32"
            ))
        );
        assert_eq!(
            FusionLaunchConfig {
                vector_load: 3,
                ..FusionLaunchConfig::default()
            }
            .validate(),
            Err(ComputeError::InvalidParameter(
                "pixel fusion vector_load must be 1, 2 or 4"
            ))
        );

        let config = FusionLaunchConfig {
            block_x: 32,
            block_y: 8,
            vector_load: 4,
        };
        assert!(config.validate().is_ok());
        assert_eq!(config.grid(37, 23), (1, 3));
        assert_eq!(config.grid(129, 1), (2, 1));
        assert_eq!(config.grid(0, 0), (1, 1));
    }

    #[test]
    fn candidate_spec_declares_the_tunable_parameter_space() {
        let spec = candidate_spec();
        assert_eq!(spec.backend, Backend::Cuda);
        assert_eq!(spec.kernel_entry, PIXEL_FUSION_KERNEL);
        assert_eq!(spec.params.len(), 3);
        let block_x = &spec.params[0];
        assert_eq!(block_x.name(), "block_x");
        assert!(block_x.accepts_u32(16));
        assert!(!block_x.accepts_u32(24));
        assert!(spec.fits_budget(&crate::autotune::ResourceBudget::desktop_default()));
    }

    #[test]
    fn staging_pool_reuses_buffers_and_pipelines_cache_by_topology() {
        let mut backend = CudaComputeBackend::new();
        let fusion = portrait_chain();

        let first = backend.get_or_compile_pipeline(&fusion).unwrap();
        let second = backend.get_or_compile_pipeline(&fusion).unwrap();
        assert_eq!(first, second);
        let stats = backend.pipeline_cache_stats();
        assert_eq!((stats.misses, stats.hits, stats.cached_pipelines), (1, 1, 1));

        let id = backend
            .allocate_buffer(64, BufferUsage::StorageWrite)
            .unwrap();
        backend.upload_buffer(id, &[7u8; 64]).unwrap();
        assert_eq!(backend.download_buffer(id).unwrap(), vec![7u8; 64]);
        assert_eq!(
            backend.upload_buffer(id, &[0u8; 63]),
            Err(ComputeError::BufferSizeMismatch {
                expected: 64,
                actual: 63,
            })
        );
        backend.release_buffer(id).unwrap();
        assert_eq!(backend.pool_stats().pooled_buffers, 1);

        let again = backend
            .allocate_buffer(32, BufferUsage::StorageRead)
            .unwrap();
        assert_eq!(backend.pool_stats().reuses, 1);
        assert_eq!(again, BufferId(2));
        backend.trim_pool();
        assert_eq!(backend.pool_stats().pooled_buffers, 0);
    }

    #[test]
    fn strict_mode_never_reports_a_cpu_run_as_cuda() {
        if device_available() {
            return; // the negative case only exists without a device
        }
        let mut backend = CudaComputeBackend::new();
        let image = synthetic_rgba8(9, 7);
        match backend.dispatch_pixel_fusion(&image, &portrait_chain()) {
            Err(ComputeError::BackendUnavailable(_)) => {}
            other => panic!("strict dispatch must refuse without a device, got {other:?}"),
        }
        assert!(backend.last_dispatch().is_none());
        assert!(!backend.probe().is_ready());
    }

    #[test]
    fn auto_fallback_labels_the_receipt_as_cpu() {
        if device_available() {
            return;
        }
        let mut backend =
            CudaComputeBackend::with_mode(FusionLaunchConfig::default(), ExecutionMode::AutoFallback);
        let image = synthetic_rgba8(9, 7);
        let fusion = portrait_chain();
        let (fused, receipt) = backend.dispatch_pixel_fusion(&image, &fusion).unwrap();
        assert_eq!(receipt.backend, Backend::Cpu);
        assert!(backend.last_dispatch().is_none());
        assert_eq!(
            fused.pixels,
            fusion.execute_fused(&image).unwrap().pixels,
            "the fallback must be the verified CPU reference"
        );
    }

    #[test]
    fn gpu_fusion_matches_the_cpu_reference_pixel_for_pixel() {
        if !device_available() {
            return; // documented: no driver, or the PTX image is still a placeholder
        }
        let fusion = portrait_chain();
        let mut backend = CudaComputeBackend::new();
        for (width, height) in [(1, 1), (3, 5), (16, 16), (17, 19), (64, 37)] {
            let image = synthetic_rgba8(width, height);
            let reference = fusion.execute_fused(&image).expect("cpu fused");
            let (fused, receipt) = backend
                .dispatch_pixel_fusion(&image, &fusion)
                .expect("cuda fused dispatch");
            assert_eq!(receipt.backend, Backend::Cuda);
            assert_eq!(receipt.fused_stages, 7);
            assert_eq!(receipt.memory_passes, 1);
            assert_eq!(receipt.pixel_count, (width * height) as usize);
            assert!(backend.last_dispatch().is_some());

            for (index, (got, want)) in fused
                .pixels
                .chunks_exact(4)
                .zip(reference.pixels.chunks_exact(4))
                .enumerate()
            {
                assert_eq!(got[3], want[3], "alpha differs at pixel {index}");
                for channel in 0..3 {
                    let delta = (got[channel] as i32 - want[channel] as i32).abs();
                    assert!(
                        delta <= 1,
                        "channel {channel} at pixel {index} differs by {delta}: {} vs {} for {width}x{height}",
                        got[channel],
                        want[channel]
                    );
                }
            }
        }
    }
}
