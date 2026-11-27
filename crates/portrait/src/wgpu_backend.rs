//! Cross-platform WGPU/Vulkan compute backend for `aurora-portrait`.
//!
//! Uses `wgpu` (targeting Vulkan first, then primary native backends) to compile
//! and execute WGSL compute shaders for RGBA8 portrait pixel/region pipelines:
//! - `skin_lut_fixed`: 512x512 RGB8 3D-atlas LUT + skin mask + mix (`PF90005`, `PF90007`, `PF30096`)
//! - `clarity_apply`: ROI-masked local contrast / high-pass clarity (`PF8070`)
//! - `region_box_smooth`: ROI-masked 2D box smoothing + alpha-preserving blend
//! - `rgba8_stage_fusion`: Two-stage Q8 fixed-point RGBA8 exposure/contrast fusion
//!
//! No CPU fallback is silently substituted inside this module: if no WGPU/Vulkan
//! adapter is available or a parameter/shape check fails, functions return
//! `Err(WgpuError)`.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::{Image, Refusal};

/// Metadata about the active WGPU adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuAdapterInfo {
    pub name: String,
    pub backend: &'static str,
    pub device_type: &'static str,
    pub driver: String,
}

/// Execution receipt returned after a WGPU compute dispatch completes and reads back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WgpuDispatchReceipt {
    pub kernel: &'static str,
    pub adapter: String,
    pub backend: &'static str,
    pub grid: (u32, u32),
    pub threads: (u32, u32),
    pub dispatch_count: u32,
}

/// Errors returned by the WGPU/Vulkan compute backend.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WgpuError {
    Unavailable(String),
    Refused(Refusal),
    DeviceError(String),
}

impl From<Refusal> for WgpuError {
    fn from(value: Refusal) -> Self {
        Self::Refused(value)
    }
}

pub type WgpuResult<T> = Result<T, WgpuError>;

struct WgpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: WgpuAdapterInfo,
    pipelines: Mutex<HashMap<&'static str, CachedPipeline>>,
}

struct CachedPipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

fn backend_label(backend: wgpu::Backend) -> &'static str {
    match backend {
        wgpu::Backend::Vulkan => "vulkan",
        wgpu::Backend::Dx12 => "dx12",
        wgpu::Backend::Metal => "metal",
        wgpu::Backend::Gl => "gl",
        wgpu::Backend::BrowserWebGpu => "webgpu",
        wgpu::Backend::Noop => "noop",
    }
}

fn device_type_label(ty: wgpu::DeviceType) -> &'static str {
    match ty {
        wgpu::DeviceType::DiscreteGpu => "discrete",
        wgpu::DeviceType::IntegratedGpu => "integrated",
        wgpu::DeviceType::VirtualGpu => "virtual",
        wgpu::DeviceType::Cpu => "cpu",
        wgpu::DeviceType::Other => "other",
    }
}

fn init_context(backends: wgpu::Backends, label: &str) -> WgpuResult<WgpuContext> {
    let instance_desc = wgpu::InstanceDescriptor {
        backends,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    };
    let instance = wgpu::Instance::new(instance_desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .map_err(|err| WgpuError::Unavailable(format!("no {label} adapter: {err:?}")))?;

    let raw_info = adapter.get_info();
    if raw_info.backend == wgpu::Backend::Noop || raw_info.device_type == wgpu::DeviceType::Cpu {
        return Err(WgpuError::Unavailable(format!(
            "skipping software/noop adapter '{}'",
            raw_info.name
        )));
    }

    let mut limits = wgpu::Limits::default().using_resolution(adapter.limits());
    limits.max_storage_buffers_per_shader_stage = adapter
        .limits()
        .max_storage_buffers_per_shader_stage
        .min(16)
        .max(8);

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("aurora-portrait-wgpu"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
    }))
    .map_err(|err| WgpuError::Unavailable(format!("{label} request_device failed: {err:?}")))?;

    let info = WgpuAdapterInfo {
        name: raw_info.name,
        backend: backend_label(raw_info.backend),
        device_type: device_type_label(raw_info.device_type),
        driver: raw_info.driver,
    };

    Ok(WgpuContext {
        device,
        queue,
        info,
        pipelines: Mutex::new(HashMap::new()),
    })
}

fn primary_context() -> Result<&'static WgpuContext, &'static WgpuError> {
    static CTX: OnceLock<WgpuResult<WgpuContext>> = OnceLock::new();
    CTX.get_or_init(|| {
        // Prefer Vulkan first for deterministic cross-platform compute; fall back
        // to primary native WGPU backends (e.g. Metal on macOS, DX12 on Windows).
        init_context(wgpu::Backends::VULKAN, "Vulkan")
            .or_else(|_| init_context(wgpu::Backends::PRIMARY, "WGPU"))
    })
    .as_ref()
}

fn vulkan_context() -> Result<&'static WgpuContext, &'static WgpuError> {
    static VK_CTX: OnceLock<WgpuResult<WgpuContext>> = OnceLock::new();
    VK_CTX
        .get_or_init(|| init_context(wgpu::Backends::VULKAN, "Vulkan"))
        .as_ref()
}

/// Returns `true` when a real hardware WGPU compute adapter/device is available.
pub fn available() -> bool {
    primary_context().is_ok()
}

/// Returns `true` when a real hardware Vulkan adapter/device is available via WGPU.
pub fn vulkan_available() -> bool {
    vulkan_context().is_ok()
}

/// Returns metadata for the active primary WGPU adapter, if available.
pub fn adapter_info() -> Option<WgpuAdapterInfo> {
    primary_context().ok().map(|ctx| ctx.info.clone())
}

fn select_context(require_vulkan: bool) -> WgpuResult<&'static WgpuContext> {
    if require_vulkan {
        vulkan_context().map_err(Clone::clone)
    } else {
        primary_context().map_err(Clone::clone)
    }
}

impl WgpuContext {
    fn get_or_create_pipeline(
        &self,
        key: &'static str,
        wgsl: &str,
        entry_point: &str,
        storage_bindings: &[(u32, bool)], // (binding, read_only)
    ) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
        let mut cache = self
            .pipelines
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(cached) = cache.get(key) {
            return (cached.layout.clone(), cached.pipeline.clone());
        }

        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(key),
                source: wgpu::ShaderSource::Wgsl(wgsl.into()),
            });

        let entries: Vec<wgpu::BindGroupLayoutEntry> = storage_bindings
            .iter()
            .map(|&(binding, read_only)| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();

    let layout = self
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(key),
            entries: &entries,
        });

    let pipeline_layout = self
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(key),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(key),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        cache.insert(
            key,
            CachedPipeline {
                layout: layout.clone(),
                pipeline: pipeline.clone(),
            },
        );
        (layout, pipeline)
    }

    fn create_storage_buffer(&self, label: &str, bytes: &[u8], read_only: bool) -> wgpu::Buffer {
        use wgpu::util::DeviceExt;
        let usage = if read_only {
            wgpu::BufferUsages::STORAGE
        } else {
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
        };
        // WGPU requires buffer size >= 4 and multiple of 4 for CopyBufferToBuffer.
        if bytes.len() % 4 == 0 && !bytes.is_empty() {
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: bytes,
                    usage,
                })
        } else {
            let padded_len = bytes.len().max(4).next_multiple_of(4);
            let mut padded = vec![0u8; padded_len];
            padded[..bytes.len()].copy_from_slice(bytes);
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &padded,
                    usage,
                })
        }
    }

    fn create_output_buffer(&self, label: &str, byte_len: usize) -> wgpu::Buffer {
        let size = (byte_len.max(4).next_multiple_of(4)) as u64;
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn dispatch_and_readback(
        &self,
        label: &'static str,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        output_buf: &wgpu::Buffer,
        output_byte_len: usize,
        grid: (u32, u32),
        threads: (u32, u32),
    ) -> WgpuResult<(Vec<u8>, WgpuDispatchReceipt)> {
        let copy_size = (output_byte_len.max(4).next_multiple_of(4)) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("aurora-wgpu-readback"),
            size: copy_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(label),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(label),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(grid.0, grid.1, 1);
        }
        encoder.copy_buffer_to_buffer(output_buf, 0, &staging, 0, copy_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|err| WgpuError::DeviceError(format!("device poll failed: {err:?}")))?;
        rx.recv()
            .map_err(|err| WgpuError::DeviceError(format!("map channel closed: {err:?}")))?
            .map_err(|err| WgpuError::DeviceError(format!("buffer map failed: {err:?}")))?;

        let mapped = slice
            .get_mapped_range()
            .map_err(|err| WgpuError::DeviceError(format!("get_mapped_range failed: {err:?}")))?;
        let out = mapped[..output_byte_len].to_vec();
        drop(mapped);
        staging.unmap();

        Ok((
            out,
            WgpuDispatchReceipt {
                kernel: label,
                adapter: self.info.name.clone(),
                backend: self.info.backend,
                grid,
                threads,
                dispatch_count: 1,
            },
        ))
    }
}

fn u32_words_bytes(words: &[u32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(words.len() * 4);
    for &w in words {
        out.extend_from_slice(&w.to_le_bytes());
    }
    out
}

// ---------------------------------------------------------------------------
// WGSL Shaders
// ---------------------------------------------------------------------------

const SKIN_LUT_FIXED_WGSL: &str = r#"
struct Words {
    w: array<u32>,
};

@group(0) @binding(0) var<storage, read> src: Words;
@group(0) @binding(1) var<storage, read> mask_bytes: Words;
@group(0) @binding(2) var<storage, read> lut_bytes: Words;
@group(0) @binding(3) var<storage, read> prm: Words;
@group(0) @binding(4) var<storage, read_write> dst: Words;

// prm.w[0] = width
// prm.w[1] = height
// prm.w[2] = bitcast<u32>(mix: f32)

fn unpack_byte(buf_word: u32, byte_in_word: u32) -> u32 {
    return (buf_word >> (byte_in_word * 8u)) & 0xffu;
}

fn lut_byte(x: i32, y: i32, channel: u32) -> i32 {
    let byte_idx = (u32(y) * 512u + u32(x)) * 3u + channel;
    let word = lut_bytes.w[byte_idx >> 2u];
    return i32(unpack_byte(word, byte_idx & 3u));
}

fn tap(channel: u32, px: f32, py: f32) -> f32 {
    let u = px * 512.0 - 0.5;
    let v = py * 512.0 - 0.5;
    let i0 = floor(u);
    let j0 = floor(v);
    let kx = i32(floor((u - i0) * 256.0 + 0.5));
    let ky = i32(floor((v - j0) * 256.0 + 0.5));
    let t = (kx * ky + 128) >> 8u;
    let w00 = 256 - kx - ky + t;
    let w10 = kx - t;
    let w01 = ky - t;
    let w11 = t;
    let x0 = clamp(i32(i0), 0, 511);
    let x1 = clamp(i32(i0) + 1, 0, 511);
    let y0 = clamp(i32(j0), 0, 511);
    let y1 = clamp(i32(j0) + 1, 0, 511);
    let q = lut_byte(x0, y0, channel) * w00
        + lut_byte(x1, y0, channel) * w10
        + lut_byte(x0, y1, channel) * w01
        + lut_byte(x1, y1, channel) * w11;
    return f32((q * 257 + 128) >> 8u) / 65535.0;
}

fn sample_lut(r: f32, g: f32, b: f32, channel: u32) -> f32 {
    let blue = b * 63.0;
    let bf = floor(blue);
    let q1y = floor(bf * 0.125);
    let q1x = bf - q1y * 8.0;
    let bc = ceil(blue);
    let q2y = floor(bc * 0.125);
    let q2x = bc - q2y * 8.0;
    let k = 0.0009765625;
    let span = bitcast<f32>(0x3dfc0001u);
    let x1 = span * r + (q1x * 0.125 + k);
    let y1 = span * g + (q1y * 0.125 + k);
    let x2 = q2x * 0.125 + span * r + k;
    let y2 = q2y * 0.125 + span * g + k;
    let low = tap(channel, x1, y1);
    let high = tap(channel, x2, y2);
    return low + (blue - bf) * (high - low);
}

fn quantize_u8(value: f32) -> u32 {
    let value12 = i32(floor(value * 4096.0));
    let q = clamp(value12 * 255 + 2047, 0, 255 * 4096);
    return min(u32(q) >> 12u, 255u);
}

@compute @workgroup_size(16, 16, 1)
fn skin_lut_fixed(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = prm.w[0];
    let h = prm.w[1];
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let idx = gid.y * w + gid.x;
    let p = src.w[idx];
    let mask_u8 = unpack_byte(mask_bytes.w[idx >> 2u], idx & 3u);
    let mix = bitcast<f32>(prm.w[2]);
    let weight = (f32(mask_u8) / 255.0) * mix;
    if (!(weight > 0.001)) {
        dst.w[idx] = p;
        return;
    }

    let r = f32(p & 0xffu) / 255.0;
    let g = f32((p >> 8u) & 0xffu) / 255.0;
    let b = f32((p >> 16u) & 0xffu) / 255.0;
    let a = (p >> 24u) & 0xffu;

    let mr = sample_lut(r, g, b, 0u);
    let mg = sample_lut(r, g, b, 1u);
    let mb = sample_lut(r, g, b, 2u);

    let or_u8 = quantize_u8(r + weight * (mr - r));
    let og_u8 = quantize_u8(g + weight * (mg - g));
    let ob_u8 = quantize_u8(b + weight * (mb - b));

    dst.w[idx] = or_u8 | (og_u8 << 8u) | (ob_u8 << 16u) | (a << 24u);
}
"#;

const CLARITY_WGSL: &str = r#"
struct Words {
    w: array<u32>,
};

@group(0) @binding(0) var<storage, read> src: Words;
@group(0) @binding(1) var<storage, read> base_img: Words;
@group(0) @binding(2) var<storage, read> blur_img: Words;
@group(0) @binding(3) var<storage, read> mask_bytes: Words;
@group(0) @binding(4) var<storage, read> prm: Words;
@group(0) @binding(5) var<storage, read_write> dst: Words;

// prm.w:
//   0: src_w, 1: src_h
//   2: roi_x, 3: roi_y, 4: roi_w, 5: roi_h
//   6: bitcast<u32>(degree)
//   7: bitcast<u32>(threshold)
//   8: bitcast<u32>(factor)

fn quantize_round(value: f32) -> u32 {
    let scaled = clamp(value * 255.0, 0.0, 255.0);
    return min(u32(round(scaled)), 255u);
}

@compute @workgroup_size(16, 16, 1)
fn clarity_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let src_w = prm.w[0];
    let src_h = prm.w[1];
    if (gid.x >= src_w || gid.y >= src_h) {
        return;
    }
    let src_idx = gid.y * src_w + gid.x;
    let p = src.w[src_idx];

    let roi_x = prm.w[2];
    let roi_y = prm.w[3];
    let roi_w = prm.w[4];
    let roi_h = prm.w[5];

    if (gid.x < roi_x || gid.y < roi_y || gid.x >= roi_x + roi_w || gid.y >= roi_y + roi_h) {
        dst.w[src_idx] = p;
        return;
    }

    let lx = gid.x - roi_x;
    let ly = gid.y - roi_y;
    let local = ly * roi_w + lx;
    let mask_u8 = (mask_bytes.w[local >> 2u] >> ((local & 3u) * 8u)) & 0xffu;
    let degree = bitcast<f32>(prm.w[6]);
    let amount = degree * (f32(mask_u8) / 255.0);
    if (!(amount > 0.0)) {
        dst.w[src_idx] = p;
        return;
    }

    let threshold = bitcast<f32>(prm.w[7]);
    let factor = bitcast<f32>(prm.w[8]);
    let bp = base_img.w[local];
    let blp = blur_img.w[local];

    var out_px = p & 0xff000000u;
    for (var c: u32 = 0u; c < 3u; c = c + 1u) {
        let shift = c * 8u;
        let s = f32((p >> shift) & 0xffu) / 255.0;
        let bval = f32((bp >> shift) & 0xffu) / 255.0;
        let lval = f32((blp >> shift) & 0xffu) / 255.0;
        let diff = bval - lval;
        let negative = max(-diff - threshold, 0.0);
        let positive = max(diff - threshold, 0.0);
        let edge = positive - negative;
        let reference = clamp(s + factor * edge, 0.0, 1.0);
        let res = s + (reference - s) * amount;
        out_px = out_px | (quantize_round(res) << shift);
    }
    dst.w[src_idx] = out_px;
}
"#;

const REGION_BOX_SMOOTH_WGSL: &str = r#"
struct Words {
    w: array<u32>,
};

@group(0) @binding(0) var<storage, read> src: Words;
@group(0) @binding(1) var<storage, read> mask_bytes: Words;
@group(0) @binding(2) var<storage, read> prm: Words;
@group(0) @binding(3) var<storage, read_write> dst: Words;

// prm.w:
//   0: width, 1: height
//   2: roi_x, 3: roi_y, 4: roi_w, 5: roi_h
//   6: radius
//   7: bitcast<u32>(degree)

@compute @workgroup_size(16, 16, 1)
fn region_box_smooth(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = prm.w[0];
    let h = prm.w[1];
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let idx = gid.y * w + gid.x;
    let p = src.w[idx];

    let roi_x = prm.w[2];
    let roi_y = prm.w[3];
    let roi_w = prm.w[4];
    let roi_h = prm.w[5];
    if (gid.x < roi_x || gid.y < roi_y || gid.x >= roi_x + roi_w || gid.y >= roi_y + roi_h) {
        dst.w[idx] = p;
        return;
    }

    let mask_u8 = (mask_bytes.w[idx >> 2u] >> ((idx & 3u) * 8u)) & 0xffu;
    let degree = bitcast<f32>(prm.w[7]);
    let alpha = clamp(degree * (f32(mask_u8) / 255.0), 0.0, 1.0);
    if (!(alpha > 0.0)) {
        dst.w[idx] = p;
        return;
    }

    let r = i32(prm.w[6]);
    let cx = i32(gid.x);
    let cy = i32(gid.y);
    let min_x = max(i32(roi_x), cx - r);
    let max_x = min(i32(roi_x + roi_w) - 1, cx + r);
    let min_y = max(i32(roi_y), cy - r);
    let max_y = min(i32(roi_y + roi_h) - 1, cy + r);

    var sum_r: u32 = 0u;
    var sum_g: u32 = 0u;
    var sum_b: u32 = 0u;
    var count: u32 = 0u;

    for (var sy: i32 = min_y; sy <= max_y; sy = sy + 1) {
        let row_base = u32(sy) * w;
        for (var sx: i32 = min_x; sx <= max_x; sx = sx + 1) {
            let sp = src.w[row_base + u32(sx)];
            sum_r = sum_r + (sp & 0xffu);
            sum_g = sum_g + ((sp >> 8u) & 0xffu);
            sum_b = sum_b + ((sp >> 16u) & 0xffu);
            count = count + 1u;
        }
    }

    let inv_count = 1.0 / f32(max(count, 1u));
    let avg_r = f32(sum_r) * inv_count;
    let avg_g = f32(sum_g) * inv_count;
    let avg_b = f32(sum_b) * inv_count;

    let sr = f32(p & 0xffu);
    let sg = f32((p >> 8u) & 0xffu);
    let sb = f32((p >> 16u) & 0xffu);

    let or_u8 = min(u32(round(clamp(sr + (avg_r - sr) * alpha, 0.0, 255.0))), 255u);
    let og_u8 = min(u32(round(clamp(sg + (avg_g - sg) * alpha, 0.0, 255.0))), 255u);
    let ob_u8 = min(u32(round(clamp(sb + (avg_b - sb) * alpha, 0.0, 255.0))), 255u);

    dst.w[idx] = or_u8 | (og_u8 << 8u) | (ob_u8 << 16u) | (p & 0xff000000u);
}
"#;

const RGBA8_STAGE_FUSION_WGSL: &str = r#"
struct Words {
    w: array<u32>,
};

@group(0) @binding(0) var<storage, read> src: Words;
@group(0) @binding(1) var<storage, read> prm: Words;
@group(0) @binding(2) var<storage, read_write> dst: Words;

fn unpack_r(v: u32) -> u32 { return v & 0xffu; }
fn unpack_g(v: u32) -> u32 { return (v >> 8u) & 0xffu; }
fn unpack_b(v: u32) -> u32 { return (v >> 16u) & 0xffu; }
fn unpack_a(v: u32) -> u32 { return (v >> 24u) & 0xffu; }

fn pack_rgba(r: u32, g: u32, b: u32, a: u32) -> u32 {
    return (r & 0xffu) | ((g & 0xffu) << 8u) | ((b & 0xffu) << 16u) | ((a & 0xffu) << 24u);
}

fn stage_a(c: u32, gain: u32, off: u32) -> u32 {
    let t = i32(c) * i32(gain) + 128 + bitcast<i32>(off) * 256;
    return min(u32(max(t, 0)) >> 8u, 255u);
}

fn stage_b(c: u32, contrast: u32) -> u32 {
    let t = (i32(c) - 128) * i32(contrast) + (128 * 256) + 128;
    return min(u32(max(t, 0)) >> 8u, 255u);
}

@compute @workgroup_size(8, 8, 1)
fn op_ab(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = prm.w[0];
    let h = prm.w[1];
    if (gid.x >= w || gid.y >= h) {
        return;
    }
    let i = gid.y * w + gid.x;
    let p = src.w[i];
    if (prm.w[10] == 1u && unpack_a(p) == 0u) {
        dst.w[i] = p;
        return;
    }
    let a_r = stage_a(unpack_r(p), prm.w[2], prm.w[5]);
    let a_g = stage_a(unpack_g(p), prm.w[3], prm.w[6]);
    let a_b = stage_a(unpack_b(p), prm.w[4], prm.w[7]);
    dst.w[i] = pack_rgba(
        stage_b(a_r, prm.w[8]),
        stage_b(a_g, prm.w[8]),
        stage_b(a_b, prm.w[8]),
        unpack_a(p),
    );
}
"#;

// ---------------------------------------------------------------------------
// Public Execution Entry Points
// ---------------------------------------------------------------------------

/// Executes `SkinAdjustV2` 512x512 RGB8 LUT blending (`PF90005`, `PF90007`, `PF30096`)
/// on the WGPU/Vulkan compute device.
pub fn execute_skin_lut_fixed(
    source: &Image,
    mask_r: &[u8],
    lut_rgb512: &[u8],
    mix: f32,
    require_vulkan: bool,
) -> WgpuResult<(Image, WgpuDispatchReceipt)> {
    source.validate()?;
    let pixels = source.pixel_count();
    if mask_r.len() != pixels {
        return Err(WgpuError::Refused(Refusal::ShapeMismatch("mask_r")));
    }
    if lut_rgb512.len() != 512 * 512 * 3 {
        return Err(WgpuError::Refused(Refusal::ShapeMismatch("lut_rgb512")));
    }
    if !mix.is_finite() || !(0.0..=1.0).contains(&mix) {
        return Err(WgpuError::Refused(Refusal::ValueOutOfRange("mix")));
    }

    let ctx = select_context(require_vulkan)?;
    let (layout, pipeline) = ctx.get_or_create_pipeline(
        "skin_lut_fixed",
        SKIN_LUT_FIXED_WGSL,
        "skin_lut_fixed",
        &[
            (0, true),
            (1, true),
            (2, true),
            (3, true),
            (4, false),
        ],
    );

    let prm_bytes = u32_words_bytes(&[source.width, source.height, mix.to_bits()]);
    let src_buf = ctx.create_storage_buffer("skin_lut_src", &source.pixels, true);
    let mask_buf = ctx.create_storage_buffer("skin_lut_mask", mask_r, true);
    let lut_buf = ctx.create_storage_buffer("skin_lut_table", lut_rgb512, true);
    let prm_buf = ctx.create_storage_buffer("skin_lut_prm", &prm_bytes, true);
    let dst_buf = ctx.create_output_buffer("skin_lut_dst", source.pixels.len());

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("skin_lut_fixed"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: mask_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: lut_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: prm_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: dst_buf.as_entire_binding(),
            },
        ],
    });

    let grid = (source.width.div_ceil(16), source.height.div_ceil(16));
    let (pixels_out, receipt) = ctx.dispatch_and_readback(
        "skin_lut_fixed",
        &pipeline,
        &bind_group,
        &dst_buf,
        source.pixels.len(),
        grid,
        (16, 16),
    )?;
    let image = Image::new(source.width, source.height, pixels_out)
        .ok_or(WgpuError::Refused(Refusal::ShapeMismatch("dst")))?;
    Ok((image, receipt))
}

/// Executes `PF8070` ROI-masked local contrast / clarity (`clarity_apply`) on WGPU/Vulkan.
#[allow(clippy::too_many_arguments)]
pub fn execute_clarity(
    source: &Image,
    base: &Image,
    blur: &Image,
    mask_r: &[u8],
    roi: (u32, u32, u32, u32),
    degree: f32,
    threshold: f32,
    factor: f32,
    require_vulkan: bool,
) -> WgpuResult<(Image, WgpuDispatchReceipt)> {
    source.validate()?;
    base.validate()?;
    blur.validate()?;
    let (roi_x, roi_y, roi_w, roi_h) = roi;
    if (roi_w, roi_h) != (base.width, base.height) || (roi_w, roi_h) != (blur.width, blur.height) {
        return Err(WgpuError::Refused(Refusal::ShapeMismatch("roi")));
    }
    if u64::from(roi_x) + u64::from(roi_w) > u64::from(source.width)
        || u64::from(roi_y) + u64::from(roi_h) > u64::from(source.height)
    {
        return Err(WgpuError::Refused(Refusal::ValueOutOfRange("roi")));
    }
    if mask_r.len() != base.pixel_count() {
        return Err(WgpuError::Refused(Refusal::ShapeMismatch("mask_r")));
    }
    if !degree.is_finite()
        || !(0.0..=1.0).contains(&degree)
        || !threshold.is_finite()
        || !factor.is_finite()
    {
        return Err(WgpuError::Refused(Refusal::ValueOutOfRange("clarity")));
    }

    let ctx = select_context(require_vulkan)?;
    let (layout, pipeline) = ctx.get_or_create_pipeline(
        "clarity_apply",
        CLARITY_WGSL,
        "clarity_apply",
        &[
            (0, true),
            (1, true),
            (2, true),
            (3, true),
            (4, true),
            (5, false),
        ],
    );

    let prm_bytes = u32_words_bytes(&[
        source.width,
        source.height,
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        degree.to_bits(),
        threshold.to_bits(),
        factor.to_bits(),
    ]);
    let src_buf = ctx.create_storage_buffer("clarity_src", &source.pixels, true);
    let base_buf = ctx.create_storage_buffer("clarity_base", &base.pixels, true);
    let blur_buf = ctx.create_storage_buffer("clarity_blur", &blur.pixels, true);
    let mask_buf = ctx.create_storage_buffer("clarity_mask", mask_r, true);
    let prm_buf = ctx.create_storage_buffer("clarity_prm", &prm_bytes, true);
    let dst_buf = ctx.create_output_buffer("clarity_dst", source.pixels.len());

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("clarity_apply"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: base_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: blur_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: mask_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: prm_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: dst_buf.as_entire_binding(),
            },
        ],
    });

    let grid = (source.width.div_ceil(16), source.height.div_ceil(16));
    let (pixels_out, receipt) = ctx.dispatch_and_readback(
        "clarity_apply",
        &pipeline,
        &bind_group,
        &dst_buf,
        source.pixels.len(),
        grid,
        (16, 16),
    )?;
    let image = Image::new(source.width, source.height, pixels_out)
        .ok_or(WgpuError::Refused(Refusal::ShapeMismatch("dst")))?;
    Ok((image, receipt))
}

/// CPU reference for ROI-masked 2D box smoothing.
pub fn region_box_smooth_cpu(
    source: &Image,
    mask_r: &[u8],
    roi: (u32, u32, u32, u32),
    radius: u32,
    degree: f32,
) -> WgpuResult<Image> {
    source.validate()?;
    if mask_r.len() != source.pixel_count() {
        return Err(WgpuError::Refused(Refusal::ShapeMismatch("mask_r")));
    }
    let (roi_x, roi_y, roi_w, roi_h) = roi;
    if roi_w == 0
        || roi_h == 0
        || u64::from(roi_x) + u64::from(roi_w) > u64::from(source.width)
        || u64::from(roi_y) + u64::from(roi_h) > u64::from(source.height)
    {
        return Err(WgpuError::Refused(Refusal::ValueOutOfRange("roi")));
    }
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(WgpuError::Refused(Refusal::ValueOutOfRange("degree")));
    }

    let mut out = source.clone();
    let w = source.width as i32;
    let rx0 = roi_x as i32;
    let ry0 = roi_y as i32;
    let rx1 = (roi_x + roi_w) as i32 - 1;
    let ry1 = (roi_y + roi_h) as i32 - 1;
    let r = radius as i32;

    for y in ry0..=ry1 {
        for x in rx0..=rx1 {
            let idx = (y * w + x) as usize;
            let alpha = (degree * (mask_r[idx] as f32 / 255.0)).clamp(0.0, 1.0);
            if !(alpha > 0.0) {
                continue;
            }
            let min_x = rx0.max(x - r);
            let max_x = rx1.min(x + r);
            let min_y = ry0.max(y - r);
            let max_y = ry1.min(y + r);
            let mut sum = [0u32; 3];
            let mut count = 0u32;
            for sy in min_y..=max_y {
                for sx in min_x..=max_x {
                    let sidx = ((sy * w + sx) as usize) * 4;
                    sum[0] += source.pixels[sidx] as u32;
                    sum[1] += source.pixels[sidx + 1] as u32;
                    sum[2] += source.pixels[sidx + 2] as u32;
                    count += 1;
                }
            }
            let inv = 1.0 / (count.max(1) as f32);
            let dst_idx = idx * 4;
            for c in 0..3 {
                let s = source.pixels[dst_idx + c] as f32;
                let avg = sum[c] as f32 * inv;
                let blended = (s + (avg - s) * alpha).clamp(0.0, 255.0).round() as u8;
                out.pixels[dst_idx + c] = blended;
            }
        }
    }
    Ok(out)
}

/// Executes ROI-masked 2D box smoothing on WGPU/Vulkan.
pub fn execute_region_box_smooth(
    source: &Image,
    mask_r: &[u8],
    roi: (u32, u32, u32, u32),
    radius: u32,
    degree: f32,
    require_vulkan: bool,
) -> WgpuResult<(Image, WgpuDispatchReceipt)> {
    source.validate()?;
    if mask_r.len() != source.pixel_count() {
        return Err(WgpuError::Refused(Refusal::ShapeMismatch("mask_r")));
    }
    let (roi_x, roi_y, roi_w, roi_h) = roi;
    if roi_w == 0
        || roi_h == 0
        || u64::from(roi_x) + u64::from(roi_w) > u64::from(source.width)
        || u64::from(roi_y) + u64::from(roi_h) > u64::from(source.height)
    {
        return Err(WgpuError::Refused(Refusal::ValueOutOfRange("roi")));
    }
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(WgpuError::Refused(Refusal::ValueOutOfRange("degree")));
    }

    let ctx = select_context(require_vulkan)?;
    let (layout, pipeline) = ctx.get_or_create_pipeline(
        "region_box_smooth",
        REGION_BOX_SMOOTH_WGSL,
        "region_box_smooth",
        &[(0, true), (1, true), (2, true), (3, false)],
    );

    let prm_bytes = u32_words_bytes(&[
        source.width,
        source.height,
        roi_x,
        roi_y,
        roi_w,
        roi_h,
        radius,
        degree.to_bits(),
    ]);
    let src_buf = ctx.create_storage_buffer("smooth_src", &source.pixels, true);
    let mask_buf = ctx.create_storage_buffer("smooth_mask", mask_r, true);
    let prm_buf = ctx.create_storage_buffer("smooth_prm", &prm_bytes, true);
    let dst_buf = ctx.create_output_buffer("smooth_dst", source.pixels.len());

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("region_box_smooth"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: mask_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: prm_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: dst_buf.as_entire_binding(),
            },
        ],
    });

    let grid = (source.width.div_ceil(16), source.height.div_ceil(16));
    let (pixels_out, receipt) = ctx.dispatch_and_readback(
        "region_box_smooth",
        &pipeline,
        &bind_group,
        &dst_buf,
        source.pixels.len(),
        grid,
        (16, 16),
    )?;
    let image = Image::new(source.width, source.height, pixels_out)
        .ok_or(WgpuError::Refused(Refusal::ShapeMismatch("dst")))?;
    Ok((image, receipt))
}

/// Parameters for two-stage Q8 fixed-point RGBA8 fusion (`stage_a` + `stage_b`).
#[derive(Clone, Copy, Debug)]
pub struct Rgba8FusionParams {
    pub gain: [u32; 3],
    pub off: [i32; 3],
    pub contrast: u32,
    pub strength_percent: u32,
    pub alpha_policy: u32,
}

/// CPU reference for two-stage Q8 fixed-point RGBA8 fusion.
pub fn rgba8_stage_fusion_cpu(source: &Image, params: Rgba8FusionParams) -> WgpuResult<Image> {
    source.validate()?;
    let mut out = source.clone();
    fn stage_a(c: u8, gain: u32, off: i32) -> u8 {
        let t = (c as i32) * (gain as i32) + 128 + off * 256;
        ((t.max(0) as u32) >> 8).min(255) as u8
    }
    fn stage_b(c: u8, contrast: u32) -> u8 {
        let t = (c as i32 - 128) * (contrast as i32) + (128 * 256) + 128;
        ((t.max(0) as u32) >> 8).min(255) as u8
    }
    for px in out.pixels.chunks_exact_mut(4) {
        if params.alpha_policy == 1 && px[3] == 0 {
            continue;
        }
        let ar = stage_a(px[0], params.gain[0], params.off[0]);
        let ag = stage_a(px[1], params.gain[1], params.off[1]);
        let ab = stage_a(px[2], params.gain[2], params.off[2]);
        px[0] = stage_b(ar, params.contrast);
        px[1] = stage_b(ag, params.contrast);
        px[2] = stage_b(ab, params.contrast);
    }
    Ok(out)
}

/// Executes two-stage Q8 fixed-point RGBA8 fusion (`op_ab`) on WGPU/Vulkan.
pub fn execute_rgba8_stage_fusion(
    source: &Image,
    params: Rgba8FusionParams,
    require_vulkan: bool,
) -> WgpuResult<(Image, WgpuDispatchReceipt)> {
    source.validate()?;
    let ctx = select_context(require_vulkan)?;
    let (layout, pipeline) = ctx.get_or_create_pipeline(
        "rgba8_stage_fusion",
        RGBA8_STAGE_FUSION_WGSL,
        "op_ab",
        &[(0, true), (1, true), (2, false)],
    );

    let prm_bytes = u32_words_bytes(&[
        source.width,
        source.height,
        params.gain[0],
        params.gain[1],
        params.gain[2],
        params.off[0] as u32,
        params.off[1] as u32,
        params.off[2] as u32,
        params.contrast,
        params.strength_percent,
        params.alpha_policy,
    ]);
    let src_buf = ctx.create_storage_buffer("fusion_src", &source.pixels, true);
    let prm_buf = ctx.create_storage_buffer("fusion_prm", &prm_bytes, true);
    let dst_buf = ctx.create_output_buffer("fusion_dst", source.pixels.len());

    let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("rgba8_stage_fusion"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: src_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: prm_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: dst_buf.as_entire_binding(),
            },
        ],
    });

    let grid = (source.width.div_ceil(8), source.height.div_ceil(8));
    let (pixels_out, receipt) = ctx.dispatch_and_readback(
        "op_ab",
        &pipeline,
        &bind_group,
        &dst_buf,
        source.pixels.len(),
        grid,
        (8, 8),
    )?;
    let image = Image::new(source.width, source.height, pixels_out)
        .ok_or(WgpuError::Refused(Refusal::ShapeMismatch("dst")))?;
    Ok((image, receipt))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_image(w: u32, h: u32) -> (Image, Vec<u8>) {
        let mut pixels = Vec::with_capacity((w * h * 4) as usize);
        let mut mask = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 7 + y * 13 + 11) & 0xff) as u8;
                let g = (((x * 31) ^ (y * 17) ^ 0x5a) & 0xff) as u8;
                let b = (((x | y) * 5 + 3) & 0xff) as u8;
                let a = match (x + y) % 5 {
                    0 => 0,
                    1 => 128,
                    2 => 1,
                    _ => 255,
                };
                pixels.extend_from_slice(&[r, g, b, a]);
                mask.push((((x * 19 + y * 29) & 0xff) as u8).max(16));
            }
        }
        (Image::new(w, h, pixels).unwrap(), mask)
    }

    fn synthetic_lut512() -> Vec<u8> {
        let mut lut = vec![0u8; 512 * 512 * 3];
        for y in 0..512usize {
            for x in 0..512usize {
                let qx = x / 64;
                let qy = y / 64;
                let r = ((x % 64) * 255 / 63) as u8;
                let g = ((y % 64) * 255 / 63) as u8;
                let b = ((qy * 8 + qx) * 255 / 63) as u8;
                let i = (y * 512 + x) * 3;
                lut[i] = r.saturating_add(12);
                lut[i + 1] = g.saturating_add(6);
                lut[i + 2] = b.saturating_sub(4);
            }
        }
        lut
    }

    fn max_channel_diff(a: &Image, b: &Image) -> u8 {
        assert_eq!(a.width, b.width);
        assert_eq!(a.height, b.height);
        a.pixels
            .iter()
            .zip(b.pixels.iter())
            .map(|(&x, &y)| x.abs_diff(y))
            .max()
            .unwrap_or(0)
    }

    #[test]
    fn probe_reports_real_adapter_without_panicking() {
        let avail = available();
        let vk_avail = vulkan_available();
        let info = adapter_info();
        eprintln!("wgpu available={avail} vulkan_available={vk_avail} info={info:?}");
        if vk_avail {
            assert!(avail);
        }
        assert_eq!(avail, info.is_some());
    }

    #[test]
    fn skin_lut_fixed_matches_cpu_within_1_ulp() {
        if !available() {
            eprintln!("skipping skin_lut_fixed GPU parity test: no WGPU adapter");
            return;
        }
        let lut = synthetic_lut512();
        for &(w, h) in &[(1u32, 1u32), (37, 23), (64, 64)] {
            let (src, mask) = synthetic_image(w, h);
            for &mix in &[0.0f32, 0.35, 1.0] {
                let cpu = crate::kernels::makeup::skin_adjust_lut(&src, &mask, &lut, mix).unwrap();
                let (gpu, receipt) =
                    execute_skin_lut_fixed(&src, &mask, &lut, mix, false).unwrap();
                assert_eq!(receipt.kernel, "skin_lut_fixed");
                assert_eq!(receipt.dispatch_count, 1);
                let diff = max_channel_diff(&cpu, &gpu);
                assert!(
                    diff <= 1,
                    "{w}x{h} mix={mix}: max_diff={diff} exceeds 1 ULP"
                );
                for (c, g) in cpu.pixels.chunks_exact(4).zip(gpu.pixels.chunks_exact(4)) {
                    assert_eq!(c[3], g[3], "alpha must be preserved");
                }
            }
        }
    }

    #[test]
    fn clarity_matches_cpu_within_1_ulp() {
        if !available() {
            eprintln!("skipping clarity GPU parity test: no WGPU adapter");
            return;
        }
        let (src, full_mask) = synthetic_image(37, 23);
        let roi = (5, 3, 19, 13);
        let roi_pixels = (roi.2 * roi.3) as usize;
        let base = Image::new(roi.2, roi.3, src.pixels[..roi_pixels * 4].to_vec()).unwrap();
        let mut blur = base.clone();
        for px in blur.pixels.chunks_exact_mut(4) {
            px[0] = px[0].saturating_sub(8);
            px[1] = px[1].saturating_add(6);
            px[2] = px[2].saturating_sub(4);
        }
        let roi_mask = &full_mask[..roi_pixels];

        let cpu = crate::kernels::skin_models::clarity_apply(
            &src,
            &base,
            &blur,
            roi_mask,
            roi,
            0.75,
            0.02,
            4.68,
        )
        .unwrap();
        let (gpu, receipt) =
            execute_clarity(&src, &base, &blur, roi_mask, roi, 0.75, 0.02, 4.68, false).unwrap();
        assert_eq!(receipt.kernel, "clarity_apply");
        let diff = max_channel_diff(&cpu, &gpu);
        assert!(diff <= 1, "clarity max_diff={diff} exceeds 1 ULP");
    }

    #[test]
    fn region_box_smooth_matches_cpu_within_1_ulp() {
        if !available() {
            eprintln!("skipping region_box_smooth GPU parity test: no WGPU adapter");
            return;
        }
        let (src, mask) = synthetic_image(37, 23);
        let roi = (4, 2, 21, 15);
        let cpu = region_box_smooth_cpu(&src, &mask, roi, 2, 0.8).unwrap();
        let (gpu, receipt) =
            execute_region_box_smooth(&src, &mask, roi, 2, 0.8, false).unwrap();
        assert_eq!(receipt.kernel, "region_box_smooth");
        let diff = max_channel_diff(&cpu, &gpu);
        assert!(diff <= 1, "region_box_smooth max_diff={diff} exceeds 1 ULP");
    }

    #[test]
    fn rgba8_stage_fusion_is_bit_exact_against_cpu() {
        if !available() {
            eprintln!("skipping rgba8_stage_fusion GPU parity test: no WGPU adapter");
            return;
        }
        let params = Rgba8FusionParams {
            gain: [281, 256, 230],
            off: [4, 0, -3],
            contrast: 320,
            strength_percent: 100,
            alpha_policy: 1,
        };
        for &(w, h) in &[(1u32, 1u32), (37, 23), (63, 65)] {
            let (src, _) = synthetic_image(w, h);
            let cpu = rgba8_stage_fusion_cpu(&src, params).unwrap();
            let (gpu, receipt) = execute_rgba8_stage_fusion(&src, params, false).unwrap();
            assert_eq!(receipt.kernel, "op_ab");
            assert_eq!(cpu.pixels, gpu.pixels, "{w}x{h} Q8 fusion must be bit-exact");
        }
    }
}
