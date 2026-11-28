//! Unified cross-platform compute backend abstraction and pixel-level stage fusion.
//!
//! This module defines:
//! - [`ComputeBackend`]: a platform-neutral interface for device capability
//!   probing, reusable buffer pools, compiled pipeline caching, explicit
//!   synchronization, and fused pixel-stage dispatch.
//! - [`CpuComputeBackend`]: the deterministic, byte-exact reference
//!   implementation available on every platform (Windows x86_64/ARM64, macOS,
//!   Linux).
//! - [`PixelStageFusion`]: fuses consecutive pixel-wise point operations
//!   (brightness, whiten, rosy, color temperature/tint, saturation, 1D/3D LUT)
//!   into a single full-image pass to eliminate intermediate VRAM/RAM round
//!   trips while guaranteeing byte-for-byte parity with serial stage execution.

use std::collections::HashMap;
use std::fmt;

use crate::device::{Backend, DeviceProbe};
use crate::{Image, Refusal};

/// Opaque handle to a buffer managed by a [`ComputeBackend`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BufferId(pub u64);

/// Intended memory access pattern for an allocated backend buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BufferUsage {
    StorageRead,
    StorageWrite,
    StorageReadWrite,
    Staging,
}

/// Telemetry counters for a backend's buffer reuse pool.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferPoolStats {
    /// Total number of fresh backing allocations performed.
    pub allocations: u64,
    /// Total number of allocations satisfied by reusing a pooled buffer.
    pub reuses: u64,
    /// Currently checked-out (active) buffers.
    pub active_buffers: usize,
    /// Buffers currently idle in the reuse pool.
    pub pooled_buffers: usize,
    /// Total capacity in bytes retained in the reuse pool.
    pub pooled_bytes: usize,
}

/// Stable cache key identifying a compiled fused pipeline topology.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    /// Human-readable pipeline name (e.g. `"pixel_fusion:brightness>whiten>lut1d"`).
    pub name: String,
    /// Deterministic 64-bit structural signature of the stage sequence.
    pub stage_signature: u64,
}

/// Opaque handle to a cached compute pipeline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PipelineHandle(pub u64);

/// Telemetry counters for a backend's pipeline cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineCacheStats {
    pub hits: u64,
    pub misses: u64,
    pub cached_pipelines: usize,
}

/// Receipt emitted after executing a fused pixel dispatch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionReceipt {
    pub backend: Backend,
    pub pipeline: PipelineHandle,
    pub fused_stages: usize,
    pub pixel_count: usize,
    /// Number of full-image memory passes executed (always `1` for fused execution).
    pub memory_passes: usize,
    pub bytes_read: u64,
    pub bytes_written: u64,
}

/// Typed errors returned by [`ComputeBackend`] operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComputeError {
    InvalidImage(Refusal),
    BufferNotFound(BufferId),
    BufferSizeMismatch { expected: usize, actual: usize },
    EmptyPipeline,
    InvalidParameter(&'static str),
    BackendUnavailable(&'static str),
}

impl fmt::Display for ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidImage(refusal) => write!(f, "invalid image: {refusal:?}"),
            Self::BufferNotFound(id) => write!(f, "buffer {:?} not found", id),
            Self::BufferSizeMismatch { expected, actual } => write!(
                f,
                "buffer size mismatch: expected {expected} bytes, got {actual} bytes"
            ),
            Self::EmptyPipeline => f.write_str("pixel stage fusion has no stages"),
            Self::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            Self::BackendUnavailable(msg) => write!(f, "compute backend unavailable: {msg}"),
        }
    }
}

impl std::error::Error for ComputeError {}

/// One pixel-wise point operation eligible for single-pass [`PixelStageFusion`].
///
/// Every variant transforms an RGBA8 pixel deterministically while preserving
/// the alpha channel (`px[3]`) unchanged.
#[derive(Clone, Debug, PartialEq)]
pub enum PixelOp {
    /// Additive brightness offset in `[-1.0, 1.0]`.
    Brightness { delta: f32 },
    /// Portrait skin whitening / gentle highlight-preserving lift in `[0.0, 1.0]`.
    Whiten { strength: f32 },
    /// Portrait rosy / healthy blush warmth in `[0.0, 1.0]`.
    Rosy { strength: f32 },
    /// Color temperature (`warmth`) and green/magenta `tint`, each in `[-1.0, 1.0]`.
    Temperature { warmth: f32, tint: f32 },
    /// Saturation adjustment in `[-1.0, 1.0]` around BT.709 luminance.
    Saturation { amount: f32 },
    /// Independent per-channel 8-bit 1D LUT.
    Lut1D {
        r: Box<[u8; 256]>,
        g: Box<[u8; 256]>,
        b: Box<[u8; 256]>,
    },
    /// Cubic 3D LUT with trilinear interpolation (`table.len() == size^3`, `size >= 2`).
    /// Entries are stored in `[r_idx + size * (g_idx + size * b_idx)]` order with
    /// normalized RGB values in `[0.0, 1.0]`.
    Lut3D {
        size: usize,
        table: Vec<[f32; 3]>,
    },
}

impl PixelOp {
    /// Short stable tag for pipeline naming and hashing.
    pub fn kind_tag(&self) -> &'static str {
        match self {
            Self::Brightness { .. } => "brightness",
            Self::Whiten { .. } => "whiten",
            Self::Rosy { .. } => "rosy",
            Self::Temperature { .. } => "temperature",
            Self::Saturation { .. } => "saturation",
            Self::Lut1D { .. } => "lut1d",
            Self::Lut3D { .. } => "lut3d",
        }
    }

    /// Validate parameter finiteness and LUT dimensions.
    pub fn validate(&self) -> Result<(), ComputeError> {
        match self {
            Self::Brightness { delta } => {
                if !delta.is_finite() {
                    return Err(ComputeError::InvalidParameter("brightness delta must be finite"));
                }
            }
            Self::Whiten { strength } => {
                if !strength.is_finite() {
                    return Err(ComputeError::InvalidParameter("whiten strength must be finite"));
                }
            }
            Self::Rosy { strength } => {
                if !strength.is_finite() {
                    return Err(ComputeError::InvalidParameter("rosy strength must be finite"));
                }
            }
            Self::Temperature { warmth, tint } => {
                if !warmth.is_finite() || !tint.is_finite() {
                    return Err(ComputeError::InvalidParameter(
                        "temperature warmth/tint must be finite",
                    ));
                }
            }
            Self::Saturation { amount } => {
                if !amount.is_finite() {
                    return Err(ComputeError::InvalidParameter("saturation amount must be finite"));
                }
            }
            Self::Lut1D { .. } => {}
            Self::Lut3D { size, table } => {
                if *size < 2 {
                    return Err(ComputeError::InvalidParameter("3D LUT size must be >= 2"));
                }
                let expected = size
                    .checked_mul(*size)
                    .and_then(|v| v.checked_mul(*size))
                    .ok_or(ComputeError::InvalidParameter("3D LUT size overflow"))?;
                if table.len() != expected {
                    return Err(ComputeError::InvalidParameter(
                        "3D LUT table length must equal size^3",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Apply this single point operation to one RGBA8 pixel.
    ///
    /// Both serial execution and single-pass fused execution invoke this exact
    /// per-pixel kernel so outputs are bit-exact across all inputs.
    #[inline]
    pub fn apply_pixel(&self, px: [u8; 4]) -> [u8; 4] {
        let a = px[3];
        match self {
            Self::Brightness { delta } => {
                let d = delta.clamp(-1.0, 1.0);
                let r = quantize_u8(px[0] as f32 / 255.0 + d);
                let g = quantize_u8(px[1] as f32 / 255.0 + d);
                let b = quantize_u8(px[2] as f32 / 255.0 + d);
                [r, g, b, a]
            }
            Self::Whiten { strength } => {
                let s = strength.clamp(0.0, 1.0);
                if s == 0.0 {
                    return px;
                }
                let lift = |v: u8| -> u8 {
                    let x = v as f32 / 255.0;
                    // Smooth highlight-preserving lift (zero at 0 and 1, peaks in midtones).
                    let y = x + s * 0.35 * (1.0 - x) * x.sqrt();
                    quantize_u8(y)
                };
                [lift(px[0]), lift(px[1]), lift(px[2]), a]
            }
            Self::Rosy { strength } => {
                let s = strength.clamp(0.0, 1.0);
                if s == 0.0 {
                    return px;
                }
                let r0 = px[0] as f32 / 255.0;
                let g0 = px[1] as f32 / 255.0;
                let b0 = px[2] as f32 / 255.0;
                let r = r0 + s * 0.14 * (1.0 - r0);
                let g = g0 - s * 0.03 * g0;
                let b = b0 + s * 0.02 * (1.0 - b0);
                [quantize_u8(r), quantize_u8(g), quantize_u8(b), a]
            }
            Self::Temperature { warmth, tint } => {
                let w = warmth.clamp(-1.0, 1.0);
                let t = tint.clamp(-1.0, 1.0);
                let r = px[0] as f32 / 255.0 + w * 0.15 + t * 0.06;
                let g = px[1] as f32 / 255.0 - t * 0.10;
                let b = px[2] as f32 / 255.0 - w * 0.15 + t * 0.06;
                [quantize_u8(r), quantize_u8(g), quantize_u8(b), a]
            }
            Self::Saturation { amount } => {
                let factor = 1.0 + amount.clamp(-1.0, 1.0);
                let r0 = px[0] as f32 / 255.0;
                let g0 = px[1] as f32 / 255.0;
                let b0 = px[2] as f32 / 255.0;
                let luma = 0.2126 * r0 + 0.7152 * g0 + 0.0722 * b0;
                let r = luma + (r0 - luma) * factor;
                let g = luma + (g0 - luma) * factor;
                let b = luma + (b0 - luma) * factor;
                [quantize_u8(r), quantize_u8(g), quantize_u8(b), a]
            }
            Self::Lut1D { r, g, b } => [r[px[0] as usize], g[px[1] as usize], b[px[2] as usize], a],
            Self::Lut3D { size, table } => {
                let rgb = sample_lut3d(*size, table, px[0], px[1], px[2]);
                [quantize_u8(rgb[0]), quantize_u8(rgb[1]), quantize_u8(rgb[2]), a]
            }
        }
    }
}

#[inline]
fn quantize_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[inline]
fn sample_lut3d(size: usize, table: &[[f32; 3]], r_u8: u8, g_u8: u8, b_u8: u8) -> [f32; 3] {
    let max_idx = (size - 1) as f32;
    let rf = (r_u8 as f32 / 255.0) * max_idx;
    let gf = (g_u8 as f32 / 255.0) * max_idx;
    let bf = (b_u8 as f32 / 255.0) * max_idx;

    let r0 = (rf.floor() as usize).min(size - 1);
    let g0 = (gf.floor() as usize).min(size - 1);
    let b0 = (bf.floor() as usize).min(size - 1);
    let r1 = (r0 + 1).min(size - 1);
    let g1 = (g0 + 1).min(size - 1);
    let b1 = (b0 + 1).min(size - 1);

    let dr = rf - r0 as f32;
    let dg = gf - g0 as f32;
    let db = bf - b0 as f32;

    let at = |ri: usize, gi: usize, bi: usize| -> [f32; 3] {
        table[ri + size * (gi + size * bi)]
    };

    let c000 = at(r0, g0, b0);
    let c100 = at(r1, g0, b0);
    let c010 = at(r0, g1, b0);
    let c110 = at(r1, g1, b0);
    let c001 = at(r0, g0, b1);
    let c101 = at(r1, g0, b1);
    let c011 = at(r0, g1, b1);
    let c111 = at(r1, g1, b1);

    let mut out = [0.0f32; 3];
    for ch in 0..3 {
        let c00 = c000[ch] * (1.0 - dr) + c100[ch] * dr;
        let c10 = c010[ch] * (1.0 - dr) + c110[ch] * dr;
        let c01 = c001[ch] * (1.0 - dr) + c101[ch] * dr;
        let c11 = c011[ch] * (1.0 - dr) + c111[ch] * dr;
        let c0 = c00 * (1.0 - dg) + c10 * dg;
        let c1 = c01 * (1.0 - dg) + c11 * dg;
        out[ch] = c0 * (1.0 - db) + c1 * db;
    }
    out
}

/// Fuses consecutive pixel-wise point operations into a single full-image pass.
///
/// Instead of allocating `N - 1` intermediate full-frame buffers and reading /
/// writing VRAM/RAM `N` times, [`PixelStageFusion`] applies all stages to each
/// pixel in registers before storing the final RGBA8 pixel once.
#[derive(Clone, Debug, Default)]
pub struct PixelStageFusion {
    stages: Vec<PixelOp>,
}

impl PixelStageFusion {
    pub fn new() -> Self {
        Self { stages: Vec::new() }
    }

    pub fn from_stages(stages: Vec<PixelOp>) -> Result<Self, ComputeError> {
        for op in &stages {
            op.validate()?;
        }
        Ok(Self { stages })
    }

    pub fn push(&mut self, op: PixelOp) -> Result<&mut Self, ComputeError> {
        op.validate()?;
        self.stages.push(op);
        Ok(self)
    }

    pub fn with_stage(mut self, op: PixelOp) -> Result<Self, ComputeError> {
        self.push(op)?;
        Ok(self)
    }

    pub fn stages(&self) -> &[PixelOp] {
        &self.stages
    }

    pub fn len(&self) -> usize {
        self.stages.len()
    }

    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Build a deterministic [`PipelineKey`] representing the stage topology
    /// (suitable for caching compiled WGSL/MSL/CUDA kernels per stage chain).
    pub fn pipeline_key(&self) -> PipelineKey {
        const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
        const FNV_PRIME: u64 = 0x0100_0000_01b3;

        let mut hash = FNV_OFFSET;
        let mut name = String::from("pixel_fusion:");
        for (idx, op) in self.stages.iter().enumerate() {
            if idx > 0 {
                name.push('>');
            }
            let tag = op.kind_tag();
            name.push_str(tag);
            for byte in tag.as_bytes() {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(FNV_PRIME);
            }
            hash ^= 0xff;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        PipelineKey {
            name,
            stage_signature: hash,
        }
    }

    /// Execute all stages in a single full-image traversal (1 read + 1 write pass).
    pub fn execute_fused(&self, input: &Image) -> Result<Image, ComputeError> {
        input.validate().map_err(ComputeError::InvalidImage)?;
        if self.stages.is_empty() {
            return Err(ComputeError::EmptyPipeline);
        }
        let mut out = vec![0u8; input.pixels.len()];
        self.execute_fused_into(&input.pixels, &mut out)?;
        Ok(Image {
            width: input.width,
            height: input.height,
            pixels: out,
        })
    }

    /// Single-pass fused execution directly between raw RGBA8 slices.
    pub fn execute_fused_into(&self, src: &[u8], dst: &mut [u8]) -> Result<(), ComputeError> {
        if self.stages.is_empty() {
            return Err(ComputeError::EmptyPipeline);
        }
        if src.len() != dst.len() || src.len() % 4 != 0 {
            return Err(ComputeError::BufferSizeMismatch {
                expected: src.len(),
                actual: dst.len(),
            });
        }
        for (in_px, out_px) in src.chunks_exact(4).zip(dst.chunks_exact_mut(4)) {
            let mut px = [in_px[0], in_px[1], in_px[2], in_px[3]];
            for stage in &self.stages {
                px = stage.apply_pixel(px);
            }
            out_px.copy_from_slice(&px);
        }
        Ok(())
    }

    /// Reference serial execution: runs each stage in a separate full-image pass.
    ///
    /// Provided as the oracle for parity verification tests (`execute_fused`
    /// must match `execute_serial` with 0 byte differences).
    pub fn execute_serial(&self, input: &Image) -> Result<Image, ComputeError> {
        input.validate().map_err(ComputeError::InvalidImage)?;
        if self.stages.is_empty() {
            return Err(ComputeError::EmptyPipeline);
        }
        let mut current = input.clone();
        for stage in &self.stages {
            let mut next = vec![0u8; current.pixels.len()];
            for (in_px, out_px) in current.pixels.chunks_exact(4).zip(next.chunks_exact_mut(4)) {
                let px = stage.apply_pixel([in_px[0], in_px[1], in_px[2], in_px[3]]);
                out_px.copy_from_slice(&px);
            }
            current.pixels = next;
        }
        Ok(current)
    }
}

/// Platform-neutral compute backend interface for CPU, CUDA, Metal, Vulkan/WGPU,
/// and reserved NPU implementations.
pub trait ComputeBackend {
    /// Backend family identity.
    fn backend(&self) -> Backend;

    /// Device capability probe reported by this backend.
    fn probe(&self) -> DeviceProbe;

    /// Allocate (or reuse from pool) a buffer of `byte_len` bytes.
    fn allocate_buffer(
        &mut self,
        byte_len: usize,
        usage: BufferUsage,
    ) -> Result<BufferId, ComputeError>;

    /// Return a buffer to the backend's reuse pool.
    fn release_buffer(&mut self, id: BufferId) -> Result<(), ComputeError>;

    /// Copy host bytes into a managed buffer.
    fn upload_buffer(&mut self, id: BufferId, data: &[u8]) -> Result<(), ComputeError>;

    /// Copy managed buffer bytes back to host memory.
    fn download_buffer(&self, id: BufferId) -> Result<Vec<u8>, ComputeError>;

    /// Compile or fetch a cached pipeline handle for a fused pixel pipeline.
    fn get_or_compile_pipeline(
        &mut self,
        fusion: &PixelStageFusion,
    ) -> Result<PipelineHandle, ComputeError>;

    /// Block until all previously submitted device work completes.
    fn synchronize(&mut self) -> Result<(), ComputeError>;

    /// Dispatch a [`PixelStageFusion`] pipeline over an RGBA8 image in a single pass.
    fn dispatch_pixel_fusion(
        &mut self,
        input: &Image,
        fusion: &PixelStageFusion,
    ) -> Result<(Image, ExecutionReceipt), ComputeError>;

    /// Current buffer pool statistics.
    fn pool_stats(&self) -> BufferPoolStats;

    /// Current pipeline cache statistics.
    fn pipeline_cache_stats(&self) -> PipelineCacheStats;
}

#[derive(Debug)]
struct PooledEntry {
    data: Vec<u8>,
    usage: BufferUsage,
}

/// Deterministic CPU reference implementation of [`ComputeBackend`].
///
/// Serves as the universal byte-exact ground truth across all platforms
/// (Windows x86_64/ARM64, macOS, Linux) and verifies buffer pooling, pipeline
/// caching, and single-pass stage fusion semantics without hardware dependencies.
#[derive(Debug)]
pub struct CpuComputeBackend {
    next_buffer_id: u64,
    next_pipeline_id: u64,
    active_buffers: HashMap<BufferId, PooledEntry>,
    free_pool: Vec<Vec<u8>>,
    allocations: u64,
    reuses: u64,
    pipeline_cache: HashMap<PipelineKey, PipelineHandle>,
    cache_hits: u64,
    cache_misses: u64,
    max_dimension: u32,
}

impl Default for CpuComputeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuComputeBackend {
    pub fn new() -> Self {
        Self {
            next_buffer_id: 1,
            next_pipeline_id: 1,
            active_buffers: HashMap::new(),
            free_pool: Vec::new(),
            allocations: 0,
            reuses: 0,
            pipeline_cache: HashMap::new(),
            cache_hits: 0,
            cache_misses: 0,
            max_dimension: 32768,
        }
    }

    /// Clear all idle buffers retained in the reuse pool.
    pub fn trim_pool(&mut self) {
        self.free_pool.clear();
    }
}

impl ComputeBackend for CpuComputeBackend {
    fn backend(&self) -> Backend {
        Backend::Cpu
    }

    fn probe(&self) -> DeviceProbe {
        DeviceProbe {
            backend: Backend::Cpu,
            driver_ready: true,
            kernel_compiled: true,
            fp32: true,
            fp16: false,
            max_width: self.max_dimension,
            max_height: self.max_dimension,
        }
    }

    fn allocate_buffer(
        &mut self,
        byte_len: usize,
        usage: BufferUsage,
    ) -> Result<BufferId, ComputeError> {
        let best_idx = self
            .free_pool
            .iter()
            .enumerate()
            .filter(|(_, buf)| buf.capacity() >= byte_len)
            .min_by_key(|(_, buf)| buf.capacity())
            .map(|(idx, _)| idx);

        let data = if let Some(idx) = best_idx {
            let mut buf = self.free_pool.swap_remove(idx);
            buf.clear();
            buf.resize(byte_len, 0);
            self.reuses += 1;
            buf
        } else {
            self.allocations += 1;
            vec![0u8; byte_len]
        };

        let id = BufferId(self.next_buffer_id);
        self.next_buffer_id += 1;
        self.active_buffers.insert(id, PooledEntry { data, usage });
        Ok(id)
    }

    fn release_buffer(&mut self, id: BufferId) -> Result<(), ComputeError> {
        let entry = self
            .active_buffers
            .remove(&id)
            .ok_or(ComputeError::BufferNotFound(id))?;
        let _ = entry.usage;
        self.free_pool.push(entry.data);
        Ok(())
    }

    fn upload_buffer(&mut self, id: BufferId, data: &[u8]) -> Result<(), ComputeError> {
        let entry = self
            .active_buffers
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
            .active_buffers
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
        let key = fusion.pipeline_key();
        if let Some(&handle) = self.pipeline_cache.get(&key) {
            self.cache_hits += 1;
            return Ok(handle);
        }
        self.cache_misses += 1;
        let handle = PipelineHandle(self.next_pipeline_id);
        self.next_pipeline_id += 1;
        self.pipeline_cache.insert(key, handle);
        Ok(handle)
    }

    fn synchronize(&mut self) -> Result<(), ComputeError> {
        Ok(())
    }

    fn dispatch_pixel_fusion(
        &mut self,
        input: &Image,
        fusion: &PixelStageFusion,
    ) -> Result<(Image, ExecutionReceipt), ComputeError> {
        input.validate().map_err(ComputeError::InvalidImage)?;
        let pipeline = self.get_or_compile_pipeline(fusion)?;
        let byte_len = input.pixels.len();

        let src_id = self.allocate_buffer(byte_len, BufferUsage::StorageRead)?;
        let dst_id = self.allocate_buffer(byte_len, BufferUsage::StorageWrite)?;
        self.upload_buffer(src_id, &input.pixels)?;

        let run_result = {
            let src_buf = self
                .active_buffers
                .get(&src_id)
                .ok_or(ComputeError::BufferNotFound(src_id))?;
            let mut tmp = vec![0u8; byte_len];
            fusion.execute_fused_into(&src_buf.data, &mut tmp)?;
            tmp
        };
        self.upload_buffer(dst_id, &run_result)?;
        self.synchronize()?;
        let out_pixels = self.download_buffer(dst_id)?;

        self.release_buffer(src_id)?;
        self.release_buffer(dst_id)?;

        let receipt = ExecutionReceipt {
            backend: Backend::Cpu,
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
            active_buffers: self.active_buffers.len(),
            pooled_buffers: self.free_pool.len(),
            pooled_bytes: self.free_pool.iter().map(Vec::capacity).sum(),
        }
    }

    fn pipeline_cache_stats(&self) -> PipelineCacheStats {
        PipelineCacheStats {
            hits: self.cache_hits,
            misses: self.cache_misses,
            cached_pipelines: self.pipeline_cache.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a deterministic synthetic RGBA8 image covering gradients,
    /// boundaries (0, 1, 127, 128, 254, 255), and varied alpha channels.
    fn synthetic_rgba8(width: u32, height: u32) -> Image {
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        let boundary = [0u8, 1, 2, 63, 127, 128, 191, 253, 254, 255];
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) as usize;
                let (r, g, b, a) = if idx < boundary.len() {
                    let v = boundary[idx];
                    (v, 255 - v, v.wrapping_mul(37), if idx % 2 == 0 { 255 } else { 128 })
                } else {
                    let r = ((x * 29 + y * 13) % 256) as u8;
                    let g = ((x * 7 + y * 41 + 85) % 256) as u8;
                    let b = ((x * 53 + y * 19 + 170) % 256) as u8;
                    let a = ((idx * 17) % 256) as u8;
                    (r, g, b, a)
                };
                pixels.extend_from_slice(&[r, g, b, a]);
            }
        }
        Image::new(width, height, pixels).expect("valid synthetic image")
    }

    fn make_s_curve_lut1d() -> PixelOp {
        let mut r = Box::new([0u8; 256]);
        let mut g = Box::new([0u8; 256]);
        let mut b = Box::new([0u8; 256]);
        for i in 0..256 {
            let x = i as f32 / 255.0;
            let s = x * x * (3.0 - 2.0 * x);
            r[i] = quantize_u8((s * 1.04).min(1.0));
            g[i] = quantize_u8(s);
            b[i] = quantize_u8(s * 0.96);
        }
        PixelOp::Lut1D { r, g, b }
    }

    fn make_warm_lut3d(size: usize) -> PixelOp {
        let mut table = Vec::with_capacity(size * size * size);
        let denom = (size - 1) as f32;
        for bi in 0..size {
            for gi in 0..size {
                for ri in 0..size {
                    let r = ri as f32 / denom;
                    let g = gi as f32 / denom;
                    let b = bi as f32 / denom;
                    table.push([
                        (r * 1.05 + 0.02 * (1.0 - r)).clamp(0.0, 1.0),
                        (g * 1.00).clamp(0.0, 1.0),
                        (b * 0.92).clamp(0.0, 1.0),
                    ]);
                }
            }
        }
        PixelOp::Lut3D { size, table }
    }

    fn full_portrait_chain() -> PixelStageFusion {
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
            .with_stage(make_s_curve_lut1d())
            .unwrap()
            .with_stage(make_warm_lut3d(9))
            .unwrap()
    }

    #[test]
    fn fused_matches_serial_byte_exact_across_odd_even_and_boundary_sizes() {
        let fusion = full_portrait_chain();
        let mut backend = CpuComputeBackend::new();

        for (w, h) in [(1, 1), (3, 5), (16, 16), (17, 19), (64, 37)] {
            let input = synthetic_rgba8(w, h);
            let serial = fusion.execute_serial(&input).expect("serial execution");
            let fused_direct = fusion.execute_fused(&input).expect("fused execution");
            let (fused_backend, receipt) = backend
                .dispatch_pixel_fusion(&input, &fusion)
                .expect("backend fused dispatch");

            assert_eq!(
                fused_direct.pixels, serial.pixels,
                "direct fused output must have 0 byte difference vs serial for {w}x{h}"
            );
            assert_eq!(
                fused_backend.pixels, serial.pixels,
                "CpuComputeBackend fused output must have 0 byte difference vs serial for {w}x{h}"
            );
            assert_eq!(receipt.backend, Backend::Cpu);
            assert_eq!(receipt.fused_stages, 7);
            assert_eq!(receipt.memory_passes, 1);
            assert_eq!(receipt.pixel_count, (w * h) as usize);
            assert_eq!(receipt.bytes_read, (w * h * 4) as u64);
            assert_eq!(receipt.bytes_written, (w * h * 4) as u64);

            // Alpha channel must remain strictly unchanged.
            for (src_px, dst_px) in input.pixels.chunks_exact(4).zip(fused_backend.pixels.chunks_exact(4)) {
                assert_eq!(dst_px[3], src_px[3], "alpha must be preserved for {w}x{h}");
            }
        }
    }

    #[test]
    fn fused_matches_serial_byte_exact_at_extreme_parameter_levels() {
        let input = synthetic_rgba8(31, 23);

        let extremes = [
            PixelStageFusion::from_stages(vec![
                PixelOp::Brightness { delta: -1.0 },
                PixelOp::Whiten { strength: 1.0 },
                PixelOp::Rosy { strength: 1.0 },
                PixelOp::Temperature {
                    warmth: -1.0,
                    tint: 1.0,
                },
                PixelOp::Saturation { amount: -1.0 },
            ])
            .unwrap(),
            PixelStageFusion::from_stages(vec![
                PixelOp::Brightness { delta: 1.0 },
                PixelOp::Whiten { strength: 0.0 },
                PixelOp::Rosy { strength: 0.0 },
                PixelOp::Temperature {
                    warmth: 1.0,
                    tint: -1.0,
                },
                PixelOp::Saturation { amount: 1.0 },
                make_s_curve_lut1d(),
                make_warm_lut3d(5),
            ])
            .unwrap(),
            // Out-of-range values must clamp identically in both paths.
            PixelStageFusion::from_stages(vec![
                PixelOp::Brightness { delta: 5.0 },
                PixelOp::Whiten { strength: 3.5 },
                PixelOp::Rosy { strength: -2.0 },
                PixelOp::Temperature {
                    warmth: 9.0,
                    tint: -9.0,
                },
                PixelOp::Saturation { amount: 4.0 },
            ])
            .unwrap(),
        ];

        let mut backend = CpuComputeBackend::new();
        for (case_idx, fusion) in extremes.iter().enumerate() {
            let serial = fusion.execute_serial(&input).expect("serial extreme");
            let (fused, _) = backend
                .dispatch_pixel_fusion(&input, fusion)
                .expect("fused extreme");
            assert_eq!(
                fused.pixels, serial.pixels,
                "extreme parameter case #{case_idx} must have 0 byte difference"
            );
        }
    }

    #[test]
    fn cpu_backend_reuses_pooled_buffers_and_caches_pipelines() {
        let mut backend = CpuComputeBackend::new();
        let fusion = full_portrait_chain();
        let img = synthetic_rgba8(16, 16);

        let (_, receipt1) = backend.dispatch_pixel_fusion(&img, &fusion).unwrap();
        let pool_after_1 = backend.pool_stats();
        let cache_after_1 = backend.pipeline_cache_stats();
        assert_eq!(pool_after_1.allocations, 2);
        assert_eq!(pool_after_1.reuses, 0);
        assert_eq!(pool_after_1.active_buffers, 0);
        assert_eq!(pool_after_1.pooled_buffers, 2);
        assert_eq!(cache_after_1.misses, 1);
        assert_eq!(cache_after_1.hits, 0);

        let (_, receipt2) = backend.dispatch_pixel_fusion(&img, &fusion).unwrap();
        let pool_after_2 = backend.pool_stats();
        let cache_after_2 = backend.pipeline_cache_stats();
        assert_eq!(receipt1.pipeline, receipt2.pipeline);
        assert_eq!(pool_after_2.allocations, 2);
        assert_eq!(pool_after_2.reuses, 2);
        assert_eq!(cache_after_2.hits, 1);
        assert_eq!(cache_after_2.misses, 1);

        backend.trim_pool();
        assert_eq!(backend.pool_stats().pooled_buffers, 0);
    }

    #[test]
    fn backend_rejects_empty_pipeline_and_invalid_parameters() {
        let mut backend = CpuComputeBackend::new();
        let img = synthetic_rgba8(4, 4);
        let empty = PixelStageFusion::new();

        assert_eq!(
            backend.dispatch_pixel_fusion(&img, &empty),
            Err(ComputeError::EmptyPipeline)
        );
        assert_eq!(
            PixelOp::Brightness { delta: f32::NAN }.validate(),
            Err(ComputeError::InvalidParameter(
                "brightness delta must be finite"
            ))
        );
        assert_eq!(
            PixelOp::Lut3D {
                size: 2,
                table: vec![[0.0; 3]; 4],
            }
            .validate(),
            Err(ComputeError::InvalidParameter(
                "3D LUT table length must equal size^3"
            ))
        );
        assert_eq!(backend.probe().backend, Backend::Cpu);
        assert!(backend.probe().is_ready());
    }
}
