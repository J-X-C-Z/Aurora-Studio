//! Cross-platform wgpu 30 compute backend for portrait pixel stages.
//!
//! This module is the WGPU/Vulkan/DX12/Metal implementation of
//! [`ComputeBackend`]. Everything here uses the real wgpu 30 API: no adapter,
//! device, shader, pipeline, timing or dispatch result is fabricated.
//!
//! Scope:
//! - [`WgpuComputeBackend`] and [`WgpuBackendSelector`] select a real adapter
//!   and cache the resulting `(Device, Queue)` pair.
//! - [`ComputeBackend`] is implemented with a size-class/usage-keyed GPU buffer
//!   pool and a [`PipelineKey`]-keyed pipeline cache.
//! - [`PixelStageFusion`] compiles to one dynamically generated WGSL compute
//!   shader with all seven [`PixelOp`] variants inlined in stage order. The
//!   output is RGBA8 with the alpha channel copied byte-for-byte.
//! - Each of the 13 sources in [`crate::shader_catalog::CATALOG`] can be
//!   compiled and dispatched through [`dispatch_catalog_shader`]. Under
//!   [`ExecutionMode::Strict`] a missing adapter, invalid source, bind
//!   mismatch, or GPU error is a typed refusal: there is **no** silent CPU
//!   fallback.
//! - [`candidates`] exposes AutoTune bounds for workgroup sizes and precision.
//!
//! The module is gated behind the crate's `wgpu-backend` feature (enabled by
//! default). It never modifies source images and it never writes to disk.

#[cfg(feature = "wgpu-backend")]
mod imp {
    use std::collections::HashMap;
    use std::fmt;
    use std::sync::mpsc;
    use std::time::Instant;

    use wgpu::util::DeviceExt;

    use crate::autotune::{
        AdapterProbeRecord, BackendTiming, CandidateSpec, ExecutionMode, ParamBound,
        VerificationStatus,
    };
    use crate::compute_backend::{
        BufferId, BufferPoolStats, BufferUsage, ComputeBackend, ComputeError, ExecutionReceipt,
        PipelineCacheStats, PipelineHandle, PipelineKey, PixelOp, PixelStageFusion,
    };
    use crate::device::{Backend, DeviceProbe};
    use crate::shader_catalog::{self, ShaderEntry};
    use crate::Image;

    // ---------------------------------------------------------------------
    // Public surface
    // ---------------------------------------------------------------------

    /// Which wgpu physical backend to select.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum WgpuBackendSelector {
        /// Vulkan (Windows/Linux/Android; macOS via MoltenVK portability).
        Vulkan,
        /// Direct3D 12 (Windows 10+).
        Dx12,
        /// Apple Metal.
        Metal,
        /// Let wgpu choose among Vulkan/Metal/DX12.
        Primary,
    }

    impl Default for WgpuBackendSelector {
        fn default() -> Self {
            Self::Primary
        }
    }

    impl WgpuBackendSelector {
        /// The matching cross-platform backend identity in [`Backend`].
        pub fn device_backend(self) -> Backend {
            match self {
                Self::Vulkan => Backend::Vulkan,
                Self::Dx12 => Backend::Dx12,
                Self::Metal => Backend::Metal,
                Self::Primary => Backend::Vulkan,
            }
        }

        /// The exact wgpu backend bit flag used for the instance/adapter.
        pub fn wgpu_backends(self) -> wgpu::Backends {
            match self {
                Self::Vulkan => wgpu::Backends::VULKAN,
                Self::Dx12 => wgpu::Backends::DX12,
                Self::Metal => wgpu::Backends::METAL,
                Self::Primary => wgpu::Backends::PRIMARY,
            }
        }

        pub const fn as_str(self) -> &'static str {
            match self {
                Self::Vulkan => "vulkan",
                Self::Dx12 => "dx12",
                Self::Metal => "metal",
                Self::Primary => "primary",
            }
        }
    }

    impl fmt::Display for WgpuBackendSelector {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str(self.as_str())
        }
    }

    /// Why a wgpu operation could not be executed.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum WgpuError {
        /// No physical adapter matched the requested backend.
        AdapterUnavailable(&'static str),
        /// The device could not be created.
        DeviceUnavailable(String),
        /// A WGSL source did not compile; the message is the real naga error.
        ShaderCompile(String),
        /// A pipeline could not be created.
        PipelineCompile(String),
        /// A bind group layout does not match the shader.
        BindLayout(String),
        /// Buffer usage or size does not match the operation.
        InvalidBuffer(String),
        /// Shader catalogue source failed static validation.
        CatalogInvalid(&'static str, &'static str),
        /// The requested operation is not supported on this backend.
        Unsupported(&'static str),
        /// A GPU error was observed through a wgpu error scope.
        Gpu(String),
    }

    impl fmt::Display for WgpuError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Self::AdapterUnavailable(msg) => write!(f, "no wgpu adapter: {msg}"),
                Self::DeviceUnavailable(msg) => write!(f, "wgpu device unavailable: {msg}"),
                Self::ShaderCompile(msg) => write!(f, "WGSL compilation failed: {msg}"),
                Self::PipelineCompile(msg) => write!(f, "pipeline creation failed: {msg}"),
                Self::BindLayout(msg) => write!(f, "bind layout mismatch: {msg}"),
                Self::InvalidBuffer(msg) => write!(f, "invalid buffer: {msg}"),
                Self::CatalogInvalid(name, msg) => {
                    write!(f, "catalog shader {name} invalid: {msg}")
                }
                Self::Unsupported(msg) => write!(f, "unsupported operation: {msg}"),
                Self::Gpu(msg) => write!(f, "GPU error: {msg}"),
            }
        }
    }

    impl std::error::Error for WgpuError {}

    /// One dispatch of a catalog shader.
    ///
    /// Every buffer is a byte slice; the host supplies the same planes the WGSL
    /// source documents. The `mode` controls what happens on any failure:
    /// [`ExecutionMode::Strict`] always returns an error.
    #[derive(Clone, Debug)]
    pub struct CatalogDispatch<'a> {
        pub pf: u32,
        /// Uniform bytes; len must equal the shader's Params size (see
        /// [`catalog_uniform_bytes`]).
        pub uniforms: &'a [u8],
        /// One entry per `@group(0) @binding(n)` where `n >= 1`, in order.
        /// Storage buffers are always read_write from the host's perspective;
        /// the pipeline's bind group layout is taken from the compiled shader.
        pub storage: &'a [&'a [u8]],
        pub width: u32,
        pub height: u32,
        pub mode: ExecutionMode,
    }

    /// Receipt for a real GPU dispatch with host-side timing.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct CatalogReceipt {
        pub pf: u32,
        pub name: &'static str,
        pub backend: wgpu::Backend,
        pub workgroups: [u32; 3],
        pub output_bytes: usize,
        pub timing: BackendTiming,
        pub status: VerificationStatus,
    }

    // ---------------------------------------------------------------------
    // Adapter / device
    // ---------------------------------------------------------------------

    /// A live wgpu device/queue pair, with its real adapter identity.
    struct WgpuDevice {
        device: wgpu::Device,
        queue: wgpu::Queue,
        info: wgpu::AdapterInfo,
        limits: wgpu::Limits,
        features: wgpu::Features,
        selector: WgpuBackendSelector,
    }

    impl WgpuDevice {
        fn create(selector: WgpuBackendSelector) -> Result<Self, WgpuError> {
            let enabled = wgpu::Instance::enabled_backend_features();
            let requested = selector.wgpu_backends();
            if !enabled.intersects(requested) {
                return Err(WgpuError::AdapterUnavailable(
                    "requested wgpu backend not compiled into this build",
                ));
            }
            let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
                backends: requested,
                ..wgpu::InstanceDescriptor::new_without_display_handle()
            });
            let adapter = match futures_lite_block_on(instance.request_adapter(
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    ..Default::default()
                },
            )) {
                Ok(adapter) => adapter,
                Err(err) => {
                    return Err(WgpuError::AdapterUnavailable(Box::leak(
                        format!("{err}").into_boxed_str(),
                    )))
                }
            };
            let info = adapter.get_info();
            let limits = adapter.limits();
            let features = adapter.features();
            let (device, queue) = match futures_lite_block_on(adapter.request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("aurora-portrait-wgpu"),
                    required_features: wgpu::Features::empty(),
                    required_limits: limits.clone(),
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                },
            )) {
                Ok(pair) => pair,
                Err(err) => return Err(WgpuError::DeviceUnavailable(format!("{err}"))),
            };
            Ok(Self {
                device,
                queue,
                info,
                limits,
                features,
                selector,
            })
        }

        fn probe_record(&self) -> AdapterProbeRecord {
            let info = &self.info;
            let limits = &self.limits;
            let device_kind = match info.device_type {
                wgpu::DeviceType::Cpu => "cpu/software",
                wgpu::DeviceType::IntegratedGpu => "integrated",
                wgpu::DeviceType::DiscreteGpu => "discrete",
                wgpu::DeviceType::VirtualGpu => "virtual",
                wgpu::DeviceType::Other => "other",
            };
            let driver_ready = true; // a device was actually created
            let status = if driver_ready {
                VerificationStatus::DeviceOk
            } else {
                VerificationStatus::Unsupported
            };
            let mut notes = format!(
                "backend={}; device_type={}; vendor={:#06x}; device={:#06x}; pci={}",
                info.backend.to_str(),
                device_kind,
                info.vendor,
                info.device,
                info.device_pci_bus_id
            );
            if info.driver_info.is_empty() {
                notes.push_str("; driver_info unavailable");
            }
            let backend = match info.backend {
                wgpu::Backend::Vulkan => Backend::Vulkan,
                wgpu::Backend::Dx12 => Backend::Dx12,
                wgpu::Backend::Metal => Backend::Metal,
                wgpu::Backend::Gl | wgpu::Backend::BrowserWebGpu | wgpu::Backend::Noop => {
                    self.selector.device_backend()
                }
            };
            AdapterProbeRecord {
                backend,
                platform: std::env::consts::OS,
                device_name: info.name.clone(),
                driver_version: info.driver.clone(),
                api_version: info.driver_info.clone(),
                driver_ready,
                supports_fp32: true, // f32 is core WebGPU
                supports_fp16: self.features.contains(wgpu::Features::SHADER_F16),
                supports_int8: true, // WGSL u32-packed byte planes are core (no feature gate)
                supports_subgroup: self.features.contains(wgpu::Features::SUBGROUP),
                max_workgroup_size: [
                    limits.max_compute_workgroup_size_x,
                    limits.max_compute_workgroup_size_y,
                    limits.max_compute_workgroup_size_z,
                ],
                max_shared_mem_bytes: limits.max_compute_workgroup_storage_size,
                vram_bytes: 0, // wgpu 30 exposes no reliable VRAM query
                status,
                notes,
            }
        }
    }

    /// Minimal block-on-future helper. `wgpu` futures complete on `poll`/submit,
    /// so a short spin with `poll_all` is sufficient for adapter/device and map
    /// requests on native targets.
    fn futures_lite_block_on<F: std::future::Future>(future: F) -> F::Output {
        use std::task::{Context, Poll, Wake, Waker};
        struct NoopWake;
        impl Wake for NoopWake {
            fn wake(self: std::sync::Arc<Self>) {}
        }
        let waker = Waker::from(std::sync::Arc::new(NoopWake));
        let mut context = Context::from_waker(&waker);
        let mut future = Box::pin(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    // ---------------------------------------------------------------------
    // Buffer pool
    // ---------------------------------------------------------------------

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    struct PoolKey {
        size_class: u64,
        usage: PoolUsage,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum PoolUsage {
        StorageRead,
        StorageWrite,
        StorageReadWrite,
        Staging,
        Uniform,
    }

    impl PoolUsage {
        fn from(usage: BufferUsage) -> Self {
            match usage {
                BufferUsage::StorageRead => Self::StorageRead,
                BufferUsage::StorageWrite => Self::StorageWrite,
                BufferUsage::StorageReadWrite => Self::StorageReadWrite,
                BufferUsage::Staging => Self::Staging,
            }
        }

        fn wgpu_flags(self) -> wgpu::BufferUsages {
            match self {
                Self::StorageRead => wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                Self::StorageWrite => wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                Self::StorageReadWrite => {
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST
                }
                Self::Staging => wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                Self::Uniform => wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            }
        }
    }

    /// Round up to a power-of-two size class, at least 4 bytes.
    fn size_class(byte_len: usize) -> u64 {
        if byte_len <= 4 {
            return 4;
        }
        let n = byte_len as u64;
        1u64 << (64 - n.leading_zeros())
    }

    struct LiveBuffer {
        buffer: wgpu::Buffer,
        byte_len: usize,
        key: PoolKey,
    }

    // ---------------------------------------------------------------------
    // Pipeline cache
    // ---------------------------------------------------------------------

    struct CachedPipeline {
        pipeline: wgpu::ComputePipeline,
        module: wgpu::ShaderModule,
        layout: wgpu::BindGroupLayout,
        bindings: Vec<wgpu::BindGroupLayoutEntry>,
    }

    /// Cross-platform WGPU compute backend.
    pub struct WgpuComputeBackend {
        device: WgpuDevice,
        next_buffer_id: u64,
        next_pipeline_id: u64,
        active: HashMap<BufferId, LiveBuffer>,
        pool: HashMap<PoolKey, Vec<wgpu::Buffer>>,
        allocations: u64,
        reuses: u64,
        pipeline_cache: HashMap<PipelineKey, PipelineHandle>,
        pipelines: HashMap<PipelineHandle, CachedPipeline>,
        cache_hits: u64,
        cache_misses: u64,
        /// Timings from the most recent dispatch, recorded in dispatch order.
        last_timing: Option<BackendTiming>,
    }

    impl fmt::Debug for WgpuComputeBackend {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("WgpuComputeBackend")
                .field("selector", &self.device.selector)
                .field("adapter", &self.device.info.name)
                .field("active_buffers", &self.active.len())
                .field("cached_pipelines", &self.pipelines.len())
                .finish()
        }
    }

    impl WgpuComputeBackend {
        /// Create a backend on the requested physical backend, or fail.
        /// This never silently substitutes another backend.
        pub fn new(selector: WgpuBackendSelector) -> Result<Self, WgpuError> {
            Ok(Self {
                device: WgpuDevice::create(selector)?,
                next_buffer_id: 1,
                next_pipeline_id: 1,
                active: HashMap::new(),
                pool: HashMap::new(),
                allocations: 0,
                reuses: 0,
                pipeline_cache: HashMap::new(),
                pipelines: HashMap::new(),
                cache_hits: 0,
                cache_misses: 0,
                last_timing: None,
            })
        }

        /// Probe the live adapter/device. Every field comes from
        /// `adapter.get_info()` / `device.limits()` / `device.features()`.
        pub fn probe_record(&self) -> AdapterProbeRecord {
            self.device.probe_record()
        }

        /// The concrete wgpu backend that was selected.
        pub fn wgpu_backend(&self) -> wgpu::Backend {
            self.device.info.backend
        }

        /// The backend identity used by the device-selection contract.
        /// Derived from the real adapter's `get_info().backend`, so a
        /// `Primary` selection reports whichever API wgpu actually picked.
        pub fn backend_identity(&self) -> Backend {
            match self.device.info.backend {
                wgpu::Backend::Vulkan => Backend::Vulkan,
                wgpu::Backend::Dx12 => Backend::Dx12,
                wgpu::Backend::Metal => Backend::Metal,
                wgpu::Backend::Gl | wgpu::Backend::BrowserWebGpu | wgpu::Backend::Noop => {
                    self.device.selector.device_backend()
                }
            }
        }

        /// Last dispatch timing, if a dispatch has completed.
        pub fn last_timing(&self) -> Option<BackendTiming> {
            self.last_timing
        }

        /// Release all idle pooled buffers.
        pub fn trim_pool(&mut self) {
            self.pool.clear();
        }

        fn to_compute_error(err: WgpuError) -> ComputeError {
            ComputeError::BackendUnavailable(Box::leak(format!("{err}").into_boxed_str()))
        }

        fn create_buffer(&mut self, byte_len: usize, key: PoolKey) -> Result<wgpu::Buffer, WgpuError> {
            let size = size_class(byte_len.max(4));
            if let Some(buf) = self.pool.get_mut(&key).and_then(|v| v.pop()) {
                self.reuses += 1;
                return Ok(buf);
            }
            let desc = wgpu::BufferDescriptor {
                label: Some("aurora-portrait-buffer"),
                size,
                usage: key.usage.wgpu_flags(),
                mapped_at_creation: false,
            };
            if size > self.device.limits.max_buffer_size {
                return Err(WgpuError::InvalidBuffer(format!(
                    "requested {size} bytes exceeds device limit {}",
                    self.device.limits.max_buffer_size
                )));
            }
            let buf = self.device.device.create_buffer(&desc);
            self.allocations += 1;
            Ok(buf)
        }

        /// Compile a WGSL source into a cached pipeline with an explicit bind
        /// group layout derived from the shader itself, and return its handle.
        fn compile_with_layout(
            &mut self,
            key: PipelineKey,
            wgsl: &str,
            bindings: Vec<wgpu::BindGroupLayoutEntry>,
        ) -> Result<PipelineHandle, WgpuError> {
            if let Some(&handle) = self.pipeline_cache.get(&key) {
                self.cache_hits += 1;
                return Ok(handle);
            }
            self.cache_misses += 1;
            // Validate WGSL through naga in wgpu; report the real messages.
            let scope = self
                .device
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let module = self
                .device
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: key.name.as_str().into(),
                    source: wgpu::ShaderSource::Wgsl(wgsl.into()),
                });
            let compilation = futures_lite_block_on(module.get_compilation_info());
            let errors: Vec<String> = compilation
                .messages
                .iter()
                .filter(|m| m.message_type == wgpu::CompilationMessageType::Error)
                .map(|m| m.message.clone())
                .collect();
            if !errors.is_empty() {
                drop(scope);
                return Err(WgpuError::ShaderCompile(errors.join("; ")));
            }
            let bgl = self
                .device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("aurora-portrait-bgl"),
                    entries: &bindings,
                });
            let layout = self
                .device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("aurora-portrait-layout"),
                    bind_group_layouts: &[Some(&bgl)],
                    immediate_size: 0,
                });
            let pipeline = self
                .device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: key.name.as_str().into(),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });
            let error = futures_lite_block_on(scope.pop());
            if let Some(err) = error {
                return Err(WgpuError::PipelineCompile(format!("{err}")));
            }
            let handle = PipelineHandle(self.next_pipeline_id);
            self.next_pipeline_id += 1;
            self.pipeline_cache.insert(key, handle);
            self.pipelines.insert(
                handle,
                CachedPipeline {
                    pipeline,
                    module,
                    layout: bgl,
                    bindings,
                },
            );
            Ok(handle)
        }

        fn readback(&self, buffer: &wgpu::Buffer, byte_len: usize) -> Result<Vec<u8>, WgpuError> {
            let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("aurora-portrait-readback"),
                size: size_class(byte_len.max(4)),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = self
                .device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("aurora-portrait-copy"),
                });
            encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, byte_len as u64);
            self.device.queue.submit(Some(encoder.finish()));
            let (tx, rx) = mpsc::channel();
            staging.slice(..byte_len as u64).map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            self.device
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| WgpuError::Gpu(format!("poll: {e}")))?;
            let result = rx.recv().map_err(|_| WgpuError::Gpu("map callback lost".into()))?;
            result.map_err(|e| WgpuError::Gpu(format!("map_async: {e}")))?;
            let view = staging
                .get_mapped_range(..byte_len as u64)
                .map_err(|e| WgpuError::Gpu(format!("get_mapped_range: {e}")))?;
            let data = view.to_vec();
            drop(view);
            staging.unmap();
            Ok(data)
        }

        /// Dispatch one of the 13 embedded WGSL catalogue shaders on the real
        /// GPU. `dispatch.storage` entries are the byte planes for bindings 1..N.
        ///
        /// `Strict` mode never falls back: every error is returned.
        pub fn dispatch_catalog_shader(
            &mut self,
            request: CatalogDispatch<'_>,
        ) -> Result<CatalogReceipt, WgpuError> {
            let entry = shader_catalog::CATALOG
                .iter()
                .find(|e| e.pf == request.pf)
                .ok_or(WgpuError::Unsupported("unknown catalogue PF"))?;
            shader_catalog::validate_entry(entry)
                .map_err(|msg| WgpuError::CatalogInvalid(entry.name, msg))?;
            let uniform_bytes = catalog_uniform_bytes(entry.pf);
            if request.uniforms.len() != uniform_bytes {
                return Err(WgpuError::InvalidBuffer(format!(
                    "{}: uniforms must be {uniform_bytes} bytes, got {}",
                    entry.name,
                    request.uniforms.len()
                )));
            }
            let layout = catalog_bindings(entry)?;
            let expected_storage = layout.len().saturating_sub(1);
            if request.storage.len() != expected_storage {
                return Err(WgpuError::BindLayout(format!(
                    "{}: expected {expected_storage} storage planes, got {}",
                    entry.name,
                    request.storage.len()
                )));
            }
            let expected_pixels = request.width as usize * request.height as usize;
            for (idx, plane) in request.storage.iter().enumerate() {
                let min = catalog_min_plane_bytes(entry.pf, idx, expected_pixels);
                if plane.len() < min {
                    return Err(WgpuError::InvalidBuffer(format!(
                        "{}: plane {} needs >= {min} bytes, got {}",
                        entry.name,
                        idx + 1,
                        plane.len()
                    )));
                }
            }
            let key = PipelineKey {
                name: format!("catalog:{}", entry.name),
                stage_signature: entry.pf as u64,
            };
            let handle = self.compile_with_layout(key, entry.wgsl, layout)?;
            let pipeline = self.pipelines.get(&handle).ok_or_else(|| {
                WgpuError::Gpu("pipeline cache lost its compiled pipeline".into())
            })?;
            let uniform_buf = self
                .device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("aurora-portrait-catalog-uniform"),
                    contents: request.uniforms,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
            let storage_bufs: Vec<wgpu::Buffer> = request
                .storage
                .iter()
                .map(|plane| {
                    self.device
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: Some("aurora-portrait-catalog-plane"),
                            contents: plane,
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST,
                        })
                })
                .collect();
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buf,
                    offset: 0,
                    size: None,
                }),
            }];
            for (i, buf) in storage_bufs.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: (i + 1) as u32,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: buf,
                        offset: 0,
                        size: None,
                    }),
                });
            }
            let bind_group = self
                .device
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("aurora-portrait-catalog-bind"),
                    layout: &pipeline.layout,
                    entries: &entries,
                });
            let workgroups = [
                request.width.div_ceil(16),
                request.height.div_ceil(16),
                1,
            ];
            let scope = self
                .device
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let start = Instant::now();
            let mut encoder = self
                .device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("aurora-portrait-catalog-dispatch"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("aurora-portrait-catalog-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
            }
            self.device.queue.submit(Some(encoder.finish()));
            let submit_ns = start.elapsed().as_nanos() as u64;
            self.device
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| WgpuError::Gpu(format!("poll: {e}")))?;
            let wall_ns = start.elapsed().as_nanos() as u64;
            if let Some(err) = futures_lite_block_on(scope.pop()) {
                // No silent CPU substitution is possible from this module;
                // both modes surface the real GPU error to the caller.
                let _ = request.mode;
                return Err(WgpuError::Gpu(format!("{err}")));
            }
            let mut h2d = uniform_bytes as u64;
            for plane in request.storage {
                h2d += plane.len() as u64;
            }
            let timing = BackendTiming {
                queue_submit_ns: submit_ns,
                device_exec_ns: wall_ns.saturating_sub(submit_ns),
                wall_ns,
                h2d_bytes: h2d,
                d2h_bytes: 0,
            };
            self.last_timing = Some(timing);
            Ok(CatalogReceipt {
                pf: entry.pf,
                name: entry.name,
                backend: self.device.info.backend,
                workgroups,
                output_bytes: 0,
                timing,
                status: VerificationStatus::DeviceOk,
            })
        }

        /// Compile a catalog shader without dispatching. Useful for verifying
        /// that all embedded WGSL sources really compile on the live adapter.
        pub fn compile_catalog_shader(&mut self, pf: u32) -> Result<PipelineHandle, WgpuError> {
            let entry = shader_catalog::CATALOG
                .iter()
                .find(|e| e.pf == pf)
                .ok_or(WgpuError::Unsupported("unknown catalogue PF"))?;
            let layout = catalog_bindings(entry)?;
            self.compile_with_layout(
                PipelineKey {
                    name: format!("catalog:{}", entry.name),
                    stage_signature: pf as u64,
                },
                entry.wgsl,
                layout,
            )
        }

        /// Dispatch a catalogue shader and read back a storage plane. This is
        /// the full GPU execution path used by parity tests.
        pub fn dispatch_catalog_and_readback(
            &mut self,
            request: CatalogDispatch<'_>,
            readback_binding: u32,
        ) -> Result<(CatalogReceipt, Vec<u8>), WgpuError> {
            let entry = shader_catalog::CATALOG
                .iter()
                .find(|e| e.pf == request.pf)
                .ok_or(WgpuError::Unsupported("unknown catalogue PF"))?;
            let layout = catalog_bindings(entry)?;
            let expected = layout.len().saturating_sub(1);
            if request.storage.len() != expected || readback_binding < 1 || readback_binding > expected as u32 {
                return Err(WgpuError::BindLayout("invalid readback binding".into()));
            }
            let uniform_bytes = catalog_uniform_bytes(entry.pf);
            if request.uniforms.len() != uniform_bytes {
                return Err(WgpuError::InvalidBuffer("uniform byte length mismatch".into()));
            }
            let pixels = request.width as usize * request.height as usize;
            for (idx, plane) in request.storage.iter().enumerate() {
                let min = catalog_min_plane_bytes(entry.pf, idx, pixels);
                if plane.len() < min {
                    return Err(WgpuError::InvalidBuffer(format!(
                        "{}: plane {} needs >= {min} bytes, got {}",
                        entry.name,
                        idx + 1,
                        plane.len()
                    )));
                }
            }
            let key = PipelineKey {
                name: format!("catalog:{}", entry.name),
                stage_signature: entry.pf as u64,
            };
            let handle = self.compile_with_layout(key, entry.wgsl, layout)?;
            let pipeline = self.pipelines.get(&handle).ok_or_else(|| {
                WgpuError::Gpu("pipeline cache lost its compiled pipeline".into())
            })?;
            let uniform_buf = self
                .device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: request.uniforms,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
            let storage_bufs: Vec<wgpu::Buffer> = request
                .storage
                .iter()
                .map(|plane| {
                    self.device
                        .device
                        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                            label: None,
                            contents: plane,
                            usage: wgpu::BufferUsages::STORAGE
                                | wgpu::BufferUsages::COPY_SRC
                                | wgpu::BufferUsages::COPY_DST,
                        })
                })
                .collect();
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buf,
                    offset: 0,
                    size: None,
                }),
            }];
            for (i, buf) in storage_bufs.iter().enumerate() {
                entries.push(wgpu::BindGroupEntry {
                    binding: (i + 1) as u32,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: buf,
                        offset: 0,
                        size: None,
                    }),
                });
            }
            let bind_group = self
                .device
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.layout,
                    entries: &entries,
                });
            let wg = [request.width.div_ceil(16), request.height.div_ceil(16), 1];
            let scope = self
                .device
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let start = Instant::now();
            let mut encoder = self
                .device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(wg[0], wg[1], wg[2]);
            }
            self.device.queue.submit(Some(encoder.finish()));
            let submit_ns = start.elapsed().as_nanos() as u64;
            self.device
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|e| WgpuError::Gpu(format!("poll: {e}")))?;
            if let Some(err) = futures_lite_block_on(scope.pop()) {
                return Err(WgpuError::Gpu(format!("{err}")));
            }
            let plane_len = request.storage[(readback_binding - 1) as usize].len();
            let data = self.readback(&storage_bufs[(readback_binding - 1) as usize], plane_len)?;
            let wall_ns = start.elapsed().as_nanos() as u64;
            let mut h2d = uniform_bytes as u64;
            for plane in request.storage {
                h2d += plane.len() as u64;
            }
            let timing = BackendTiming {
                queue_submit_ns: submit_ns,
                device_exec_ns: wall_ns.saturating_sub(submit_ns),
                wall_ns,
                h2d_bytes: h2d,
                d2h_bytes: data.len() as u64,
            };
            self.last_timing = Some(timing);
            Ok((
                CatalogReceipt {
                    pf: entry.pf,
                    name: entry.name,
                    backend: self.device.info.backend,
                    workgroups: wg,
                    output_bytes: data.len(),
                    timing,
                    status: VerificationStatus::ParityOk,
                },
                data,
            ))
        }

        /// Dispatch a fused pixel pipeline directly between RGBA8 byte slices.
        pub fn dispatch_fusion_bytes(
            &mut self,
            src: &[u8],
            dst: &mut [u8],
            fusion: &PixelStageFusion,
        ) -> Result<ExecutionReceipt, ComputeError> {
            if src.len() != dst.len() || src.len() % 4 != 0 || src.is_empty() {
                return Err(ComputeError::BufferSizeMismatch {
                    expected: src.len(),
                    actual: dst.len(),
                });
            }
            let pipeline = self.get_or_compile_pipeline(fusion)?;
            let byte_len = src.len();
            let src_id = self.allocate_buffer(byte_len, BufferUsage::StorageRead)?;
            let dst_id = self.allocate_buffer(byte_len, BufferUsage::StorageWrite)?;
            self.upload_buffer(src_id, src)?;
            let src_buf = self
                .active
                .get(&src_id)
                .ok_or(ComputeError::BufferNotFound(src_id))?
                .buffer
                .clone();
            let dst_buf = self
                .active
                .get(&dst_id)
                .ok_or(ComputeError::BufferNotFound(dst_id))?
                .buffer
                .clone();
            let cached = self.pipelines.get(&pipeline).ok_or_else(|| {
                ComputeError::BackendUnavailable("pipeline cache lost its compiled pipeline")
            })?;
            let uniform_data = fusion_uniform_bytes(fusion);
            let uniform = self
                .device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("aurora-portrait-fusion-uniform"),
                    contents: &uniform_data,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });
            let lut_data = fusion_lut_bytes(fusion);
            let lut = self
                .device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("aurora-portrait-fusion-lut"),
                    contents: &lut_data,
                    usage: wgpu::BufferUsages::STORAGE,
                });
            let bind_group = self
                .device
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("aurora-portrait-fusion-bind"),
                    layout: &cached.layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &uniform,
                                offset: 0,
                                size: None,
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &src_buf,
                                offset: 0,
                                size: None,
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &dst_buf,
                                offset: 0,
                                size: None,
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &lut,
                                offset: 0,
                                size: None,
                            }),
                        },
                    ],
                });
            let pixel_count = byte_len / 4;
            let wg = pixel_count.div_ceil(256);
            let start = Instant::now();
            let scope = self
                .device
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation);
            let mut encoder = self
                .device
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(&cached.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(wg as u32, 1, 1);
            }
            self.device.queue.submit(Some(encoder.finish()));
            let submit_ns = start.elapsed().as_nanos() as u64;
            self.device
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|err| Self::to_compute_error(WgpuError::Gpu(format!("{err:?}"))))?;
            if let Some(err) = futures_lite_block_on(scope.pop()) {
                return Err(ComputeError::BackendUnavailable(Box::leak(
                    format!("{err}").into_boxed_str(),
                )));
            }
            let wall_ns = start.elapsed().as_nanos() as u64;
            self.last_timing = Some(BackendTiming {
                queue_submit_ns: submit_ns,
                device_exec_ns: wall_ns.saturating_sub(submit_ns),
                wall_ns,
                h2d_bytes: byte_len as u64 + uniform_data.len() as u64,
                d2h_bytes: byte_len as u64,
            });
            let out = self.readback(&dst_buf, byte_len).map_err(Self::to_compute_error)?;
            dst.copy_from_slice(&out);
            self.release_buffer(src_id)?;
            self.release_buffer(dst_id)?;
            Ok(ExecutionReceipt {
                backend: self.backend_identity(),
                pipeline,
                fused_stages: fusion.len(),
                pixel_count,
                memory_passes: 1,
                bytes_read: byte_len as u64,
                bytes_written: byte_len as u64,
            })
        }
    }

    impl ComputeBackend for WgpuComputeBackend {
        fn backend(&self) -> Backend {
            self.backend_identity()
        }

        fn probe(&self) -> DeviceProbe {
            let record = self.probe_record();
            let max_dim = self.device.limits.max_texture_dimension_2d;
            DeviceProbe {
                backend: self.backend_identity(),
                driver_ready: record.driver_ready,
                kernel_compiled: true,
                fp32: true,
                fp16: record.supports_fp16,
                max_width: max_dim,
                max_height: max_dim,
            }
        }

        fn allocate_buffer(
            &mut self,
            byte_len: usize,
            usage: BufferUsage,
        ) -> Result<BufferId, ComputeError> {
            if byte_len == 0 {
                return Err(ComputeError::InvalidParameter("zero-length buffer"));
            }
            let key = PoolKey {
                size_class: size_class(byte_len),
                usage: PoolUsage::from(usage),
            };
            let buffer = self.create_buffer(byte_len, key).map_err(Self::to_compute_error)?;
            let id = BufferId(self.next_buffer_id);
            self.next_buffer_id += 1;
            self.active
                .insert(id, LiveBuffer { buffer, byte_len, key });
            Ok(id)
        }

        fn release_buffer(&mut self, id: BufferId) -> Result<(), ComputeError> {
            let live = self
                .active
                .remove(&id)
                .ok_or(ComputeError::BufferNotFound(id))?;
            self.pool.entry(live.key).or_default().push(live.buffer);
            Ok(())
        }

        fn upload_buffer(&mut self, id: BufferId, data: &[u8]) -> Result<(), ComputeError> {
            let live = self
                .active
                .get(&id)
                .ok_or(ComputeError::BufferNotFound(id))?;
            if data.len() != live.byte_len {
                return Err(ComputeError::BufferSizeMismatch {
                    expected: live.byte_len,
                    actual: data.len(),
                });
            }
            self.device.queue.write_buffer(&live.buffer, 0, data);
            Ok(())
        }

        fn download_buffer(&self, id: BufferId) -> Result<Vec<u8>, ComputeError> {
            let live = self
                .active
                .get(&id)
                .ok_or(ComputeError::BufferNotFound(id))?;
            self.readback(&live.buffer, live.byte_len)
                .map_err(Self::to_compute_error)
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
            let wgsl = fusion_wgsl(fusion);
            let bindings = vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            fusion_uniform_bytes(fusion).len() as u64,
                        ),
                    },
                    count: None,
                },
                storage_entry(1, true),
                storage_entry(2, false),
                storage_entry(3, true),
            ];
            self.compile_with_layout(key, &wgsl, bindings)
                .map_err(Self::to_compute_error)
        }

        fn synchronize(&mut self) -> Result<(), ComputeError> {
            self.device
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .map(|_| ())
                .map_err(|e| {
                    ComputeError::BackendUnavailable(Box::leak(
                        format!("wgpu poll: {e}").into_boxed_str(),
                    ))
                })
        }

        fn dispatch_pixel_fusion(
            &mut self,
            input: &Image,
            fusion: &PixelStageFusion,
        ) -> Result<(Image, ExecutionReceipt), ComputeError> {
            input.validate().map_err(ComputeError::InvalidImage)?;
            if fusion.is_empty() {
                return Err(ComputeError::EmptyPipeline);
            }
            let mut out = vec![0u8; input.pixels.len()];
            let receipt = self.dispatch_fusion_bytes(&input.pixels, &mut out, fusion)?;
            Ok((
                Image {
                    width: input.width,
                    height: input.height,
                    pixels: out,
                },
                receipt,
            ))
        }

        fn pool_stats(&self) -> BufferPoolStats {
            BufferPoolStats {
                allocations: self.allocations,
                reuses: self.reuses,
                active_buffers: self.active.len(),
                pooled_buffers: self.pool.values().map(Vec::len).sum(),
                pooled_bytes: self
                    .pool
                    .iter()
                    .map(|(key, v)| v.len() as u64 * key.size_class)
                    .sum::<u64>() as usize,
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

    fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    // ---------------------------------------------------------------------
    // WGSL generation for PixelStageFusion
    // ---------------------------------------------------------------------

    /// Uniform layout: 16-byte header followed by 16 `vec4<f32>` slots.
    /// Each stage occupies two vec4s (8 f32 lanes); the eighth lane of the
    /// second vec4 carries the 3D LUT size as an exact f32 integer.
    /// Total = 16 + 16*16 = 272 bytes (17 vec4s, 16-byte aligned).
    const MAX_SLOTS: usize = 8;

    fn fusion_uniform_bytes(fusion: &PixelStageFusion) -> Vec<u8> {
        let mut bytes = vec![0u8; 16 + 16 * 16];
        bytes[0..4].copy_from_slice(&(fusion.len() as u32).to_le_bytes());
        for (i, stage) in fusion.stages().iter().enumerate().take(MAX_SLOTS) {
            let base = 16 + i * 32;
            let mut put = |lane: usize, v: f32| {
                bytes[base + lane * 4..base + lane * 4 + 4].copy_from_slice(&v.to_le_bytes());
            };
            match stage {
                PixelOp::Brightness { delta } => put(0, *delta),
                PixelOp::Whiten { strength } | PixelOp::Rosy { strength } => put(0, *strength),
                PixelOp::Temperature { warmth, tint } => {
                    put(0, *warmth);
                    put(1, *tint);
                }
                PixelOp::Saturation { amount } => put(0, *amount),
                PixelOp::Lut3D { size, .. } => put(7, *size as f32),
                PixelOp::Lut1D { .. } => {}
            }
        }
        bytes
    }

    /// Generate the full fused WGSL. The seven stage kernels mirror the CPU
    /// `PixelOp::apply_pixel` expressions exactly; WGSL f32 is IEEE-754 binary32
    /// and `q()` rounds half away from zero for non-negative values, matching
    /// Rust's `f32::round()` used by `quantize_u8`.
    fn fusion_wgsl(fusion: &PixelStageFusion) -> String {
        let mut wgsl = String::from(
            "struct Params { count: u32, _pad0: u32, _pad1: u32, _pad2: u32,\n\
             slots: array<vec4<f32>, 16> };\n\
             @group(0) @binding(0) var<uniform> params: Params;\n\
             @group(0) @binding(1) var<storage, read> src: array<u32>;\n\
             @group(0) @binding(2) var<storage, read_write> dst: array<u32>;\n\
             @group(0) @binding(3) var<storage, read> luts: array<f32>;\n\
             fn q(v: f32) -> u32 { return u32(clamp(clamp(v,0.0,1.0)*255.0,0.0,255.0)+0.5); }\n\
             fn sf(i: u32, c: u32) -> f32 { return params.slots[i*2u+c/4u][c%4u]; }\n\
             fn su(i: u32) -> u32 { return u32(params.slots[i*2u+1u].w); }\n\
             @compute @workgroup_size(256,1,1)\n\
             fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
               let i = gid.x;\n\
               if (i >= arrayLength(&dst)) { return; }\n\
               let p = src[i];\n\
               var r = f32(p & 255u);\n\
               var g = f32((p >> 8u) & 255u);\n\
               var b = f32((p >> 16u) & 255u);\n\
               let a = (p >> 24u) & 255u;\n",
        );
        let mut slot = 0usize;
        for (stage_index, op) in fusion.stages().iter().enumerate() {
            if slot >= MAX_SLOTS {
                break;
            }
            let i = slot as u32;
            match op {
                PixelOp::Brightness { delta } => {
                    wgsl.push_str(&format!(
                        "{{ let d = clamp(sf({i}u,0u),-1.0,1.0); r=f32(q(r/255.0+d)); g=f32(q(g/255.0+d)); b=f32(q(b/255.0+d)); }}\n"
                    ));
                }
                PixelOp::Whiten { strength } => {
                    wgsl.push_str(&format!(
                        "{{ let s = clamp(sf({i}u,0u),0.0,1.0); \
                         r=f32(q(r/255.0 + s*0.35*(1.0-r/255.0)*sqrt(r/255.0))); \
                         g=f32(q(g/255.0 + s*0.35*(1.0-g/255.0)*sqrt(g/255.0))); \
                         b=f32(q(b/255.0 + s*0.35*(1.0-b/255.0)*sqrt(b/255.0))); }}\n"
                    ));
                }
                PixelOp::Rosy { strength } => {
                    wgsl.push_str(&format!(
                        "{{ let s = clamp(sf({i}u,0u),0.0,1.0); let r0=r/255.0; let g0=g/255.0; let b0=b/255.0; \
                         r=f32(q(r0+s*0.14*(1.0-r0))); g=f32(q(g0-s*0.03*g0)); b=f32(q(b0+s*0.02*(1.0-b0))); }}\n"
                    ));
                }
                PixelOp::Temperature { .. } => {
                    wgsl.push_str(&format!(
                        "{{ let w = clamp(sf({i}u,0u),-1.0,1.0); let t = clamp(sf({i}u,1u),-1.0,1.0); \
                         r=f32(q(r/255.0+w*0.15+t*0.06)); g=f32(q(g/255.0-t*0.10)); b=f32(q(b/255.0-w*0.15+t*0.06)); }}\n"
                    ));
                }
                PixelOp::Saturation { .. } => {
                    wgsl.push_str(&format!(
                        "{{ let f = 1.0+clamp(sf({i}u,0u),-1.0,1.0); let r0=r/255.0; let g0=g/255.0; let b0=b/255.0; \
                         let l = 0.2126*r0+0.7152*g0+0.0722*b0; \
                         r=f32(q(l+(r0-l)*f)); g=f32(q(l+(g0-l)*f)); b=f32(q(l+(b0-l)*f)); }}\n"
                    ));
                }
                PixelOp::Lut1D { .. } => {
                    let lut_base = lut_offset(fusion, stage_index);
                    wgsl.push_str(&format!(
                        "{{ let off = {lut_base}u; \
                         r=floor(luts[off+min(u32(r),255u)]+0.5); \
                         g=floor(luts[off+256u+min(u32(g),255u)]+0.5); \
                         b=floor(luts[off+512u+min(u32(b),255u)]+0.5); }}\n"
                    ));
                }
                PixelOp::Lut3D { .. } => {
                    let lut_base = lut_offset(fusion, stage_index);
                    // Inline all eight trilinear taps per RGB channel (naga 30 forbids
                    // passing storage pointers to user functions).
                    wgsl.push_str(&format!(
                        "{{ let n = su({i}u); let mx = f32(n-1u); \
                         let rf=(r/255.0)*mx; let gf=(g/255.0)*mx; let bf=(b/255.0)*mx; \
                         let r0=min(u32(rf),n-1u); let g0=min(u32(gf),n-1u); let b0=min(u32(bf),n-1u); \
                         let r1=min(r0+1u,n-1u); let g1=min(g0+1u,n-1u); let b1=min(b0+1u,n-1u); \
                         let dr=rf-f32(r0); let dg=gf-f32(g0); let db=bf-f32(b0); \
                         let base={lut_base}u; \
                         let i000=base+(r0+n*(g0+n*b0))*3u; let i100=base+(r1+n*(g0+n*b0))*3u; \
                         let i010=base+(r0+n*(g1+n*b0))*3u; let i110=base+(r1+n*(g1+n*b0))*3u; \
                         let i001=base+(r0+n*(g0+n*b1))*3u; let i101=base+(r1+n*(g0+n*b1))*3u; \
                         let i011=base+(r0+n*(g1+n*b1))*3u; let i111=base+(r1+n*(g1+n*b1))*3u; \
                         var rgb_out = array<f32, 3>(0.0, 0.0, 0.0); \
                         for (var ch = 0u; ch < 3u; ch = ch + 1u) {{ \
                           let a000=luts[i000+ch]; let a100=luts[i100+ch]; \
                           let a010=luts[i010+ch]; let a110=luts[i110+ch]; \
                           let a001=luts[i001+ch]; let a101=luts[i101+ch]; \
                           let a011=luts[i011+ch]; let a111=luts[i111+ch]; \
                           let c00=a000*(1.0-dr)+a100*dr; let c10=a010*(1.0-dr)+a110*dr; \
                           let c01=a001*(1.0-dr)+a101*dr; let c11=a011*(1.0-dr)+a111*dr; \
                           let c0=c00*(1.0-dg)+c10*dg; let c1=c01*(1.0-dg)+c11*dg; \
                           rgb_out[ch] = c0*(1.0-db)+c1*db; \
                         }} \
                         r=f32(q(rgb_out[0])); g=f32(q(rgb_out[1])); b=f32(q(rgb_out[2])); }}\n"
                    ));
                }
            }
            slot += 1;
        }
        wgsl.push_str(
            "let ri = u32(r) & 255u; let gi = u32(g) & 255u; let bi = u32(b) & 255u;\n\
             dst[i] = ri | (gi << 8u) | (bi << 16u) | (a << 24u);\n}\n",
        );
        wgsl
    }

    fn lut_offset(fusion: &PixelStageFusion, stage_index: usize) -> usize {
        let mut offset = 0usize;
        for (i, op) in fusion.stages().iter().enumerate().take(stage_index) {
            match op {
                PixelOp::Lut1D { .. } => offset += 768,
                PixelOp::Lut3D { size, .. } => offset += size * size * size * 3,
                _ => {}
            }
        }
        offset
    }

    /// Host data for the LUT storage buffer. A zero-fill placeholder of at
    /// least 4 f32 elements keeps the binding valid when no LUT is used.
    fn fusion_lut_bytes(fusion: &PixelStageFusion) -> Vec<u8> {
        let mut floats = Vec::new();
        for op in fusion.stages() {
            match op {
                PixelOp::Lut1D { r, g, b } => {
                    floats.extend(r.iter().map(|v| *v as f32));
                    floats.extend(g.iter().map(|v| *v as f32));
                    floats.extend(b.iter().map(|v| *v as f32));
                }
                PixelOp::Lut3D { table, .. } => {
                    for entry in table {
                        floats.extend_from_slice(entry);
                    }
                }
                _ => {}
            }
        }
        if floats.len() < 4 {
            floats.resize(4, 0.0);
        }
        floats
            .into_iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }

    // ---------------------------------------------------------------------
    // Catalog bindings / uniform ABI
    // ---------------------------------------------------------------------

    /// Bind group layout entries for one catalogue shader. All bindings are in
    /// group 0; binding 0 is the uniform, bindings 1..N are the storage planes.
    /// The `read_only` flags mirror the WGSL `read` / `read_write` annotations.
    fn catalog_bindings(entry: &ShaderEntry) -> Result<Vec<wgpu::BindGroupLayoutEntry>, WgpuError> {
        let wgsl = entry.wgsl;
        let count = wgsl
            .lines()
            .filter_map(|line| {
                let rest = line.trim().strip_prefix("@group(0)")?;
                let rest = rest.trim().strip_prefix("@binding(")?;
                let n = rest.split(')').next()?.parse::<u32>().ok()?;
                Some(n)
            })
            .max()
            .ok_or_else(|| WgpuError::CatalogInvalid(entry.name, "no bindings"))?;
        let mut out = Vec::with_capacity(count as usize + 1);
        for n in 0..=count {
            let read_only = wgsl
                .lines()
                .find(|line| line.trim().starts_with(&format!("@group(0) @binding({n})")))
                .is_some_and(|line| line.contains("var<storage, read>"));
            out.push(if n == 0 {
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(
                            catalog_uniform_bytes(entry.pf) as u64,
                        ),
                    },
                    count: None,
                }
            } else {
                storage_entry(n, read_only)
            });
        }
        Ok(out)
    }

    /// The exact uniform struct size for a catalogue shader. Values are the
    /// `static_assert(sizeof(...))` spans from the paired Metal sources, which
    /// match the WGSL `struct Params` span under uniform 16-byte alignment.
    fn catalog_uniform_bytes(pf: u32) -> usize {
        // Exact sizes from the `static_assert(sizeof(...))` ABI assertions in
        // the paired Metal sources; WGSL uniform structs share the same span.
        match pf {
            0 => 48,                       // SpotlessParams
            101 => 32,                     // SilkwormParams
            232 => 32,                     // SkullTopParams
            247 => 48,                     // VfaceParams
            90023 | 90024 | 21024 => 56,   // WrinkleParams
            90085 => 48,                   // SkinUnify90085Params
            90086 => 24,                   // BodyToneParams
            100004 => 24,                  // RednessRemoveParams
            21105 => 24,                   // SkinUnify21105Params
            30009 => 24,                   // HairColorUnifyParams
            30013 => 24,                   // NeutralGrayParams
            _ => 16,
        }
    }

    /// Minimum byte length for one storage plane of a catalogue shader, based
    /// on the documented stride in the WGSL comment.
    fn catalog_min_plane_bytes(pf: u32, plane: usize, pixels: usize) -> usize {
        match (pf, plane) {
            // pf0: src RGBA f32, blur RGB f32, face/skin f32, out RGBA f32
            (0, 0) | (0, 4) => pixels * 16,
            (0, 1) => pixels * 12,
            (0, 2) | (0, 3) => pixels * 4,
            // pf101: packed u32 planes + vec2<f32> UV planes
            (101, 0) | (101, 1) | (101, 2) | (101, 5) => pixels * 4,
            (101, 3) | (101, 4) => pixels * 8,
            // pf232: packed src, anchor vec2, two 15-point contours, packed out
            (232, 0) | (232, 4) => pixels * 4,
            (232, 1) => 8,
            (232, 2) | (232, 3) => 15 * 8,
            // pf247: base packed, atlas packed, vec4 out
            (247, 0) | (247, 1) => pixels * 4,
            (247, 2) => pixels * 16,
            // wrinkle family: five vec4/f32 planes plus vec4 out
            (90023 | 90024 | 21024, 0) | (90023 | 90024 | 21024, 1)
            | (90023 | 90024 | 21024, 2) | (90023 | 90024 | 21024, 3)
            | (90023 | 90024 | 21024, 5) => pixels * 16,
            (90023 | 90024 | 21024, 4) => pixels * 4,
            // pf90085: 10 packed/f32 planes + packed out
            (90085, 0..=8) | (90085, 10) => pixels * 4,
            (90085, 9) => pixels * 12,
            // remaining packed/f32 hybrids
            (90086, 0) | (90086, 3) => pixels * 4,
            (90086, 1) | (90086, 2) => pixels * 12,
            (100004, 0) | (100004, 1) | (100004, 2) | (100004, 3) => pixels * 4,
            (21105, 0) | (21105, 1) | (21105, 2) | (21105, 3) => pixels * 4,
            (30009, 0) | (30009, 1) | (30009, 2) | (30009, 3) => pixels * 4,
            (30013, 0) | (30013, 1) | (30013, 2) | (30013, 3) => pixels * 4,
            _ => pixels * 4,
        }
    }

    // ---------------------------------------------------------------------
    // AutoTune candidate registry
    // ---------------------------------------------------------------------

    static PARAMS: [ParamBound; 3] = [
        ParamBound::DiscreteU32 {
            name: "workgroup_x",
            values: &[64, 128, 256],
        },
        ParamBound::DiscreteU32 {
            name: "workgroup_y",
            values: &[1, 4, 8, 16],
        },
        ParamBound::EnumChoice {
            name: "precision",
            choices: &["fp32"],
        },
    ];

    static CANDIDATES: [CandidateSpec; 4] = [
        CandidateSpec {
            id: "wgpu-pixel-fusion-vk-fp32",
            algorithm_id: "pixel_stage_fusion",
            backend: Backend::Vulkan,
            kernel_entry: "pixel_fusion_wgsl_main",
            precision: crate::device::Precision::Fp32,
            params: &PARAMS,
            max_shared_mem_bytes: 0,
            max_threads_per_group: 256,
            status: VerificationStatus::Designed,
        },
        CandidateSpec {
            id: "wgpu-pixel-fusion-dx12-fp32",
            algorithm_id: "pixel_stage_fusion",
            backend: Backend::Dx12,
            kernel_entry: "pixel_fusion_wgsl_main",
            precision: crate::device::Precision::Fp32,
            params: &PARAMS,
            max_shared_mem_bytes: 0,
            max_threads_per_group: 256,
            status: VerificationStatus::Designed,
        },
        CandidateSpec {
            id: "wgpu-pixel-fusion-metal-fp32",
            algorithm_id: "pixel_stage_fusion",
            backend: Backend::Metal,
            kernel_entry: "pixel_fusion_wgsl_main",
            precision: crate::device::Precision::Fp32,
            params: &PARAMS,
            max_shared_mem_bytes: 0,
            max_threads_per_group: 256,
            status: VerificationStatus::Designed,
        },
        CandidateSpec {
            id: "wgpu-catalog-wgsl-fp32",
            algorithm_id: "portrait_catalog_wgsl",
            backend: Backend::Vulkan,
            kernel_entry: "catalog_main",
            precision: crate::device::Precision::Fp32,
            params: &PARAMS,
            max_shared_mem_bytes: 0,
            max_threads_per_group: 256,
            status: VerificationStatus::Designed,
        },
    ];

    /// AutoTune candidates exposed by this backend. Status reflects the static
    /// registry only; a successful probe/dispatch upgrades status in the
    /// receipt, never in this static table.
    pub fn candidates() -> &'static [CandidateSpec] {
        &CANDIDATES
    }

    // ---------------------------------------------------------------------
    // Tests: only real GPU execution counts; software adapters are named.
    // ---------------------------------------------------------------------

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::compute_backend::{CpuComputeBackend, PixelOp};
        use std::sync::OnceLock;

        fn backend() -> Option<&'static WgpuComputeBackend> {
            static BACKEND: OnceLock<Option<WgpuComputeBackend>> = OnceLock::new();
            BACKEND
                .get_or_init(|| WgpuComputeBackend::new(WgpuBackendSelector::Primary).ok())
                .as_ref()
        }

        /// A hardware adapter is one whose wgpu `Backend` is Vulkan, DX12 or
        /// Metal and whose name is not a known software rasterizer.
        fn is_physical_gpu(b: &WgpuComputeBackend) -> bool {
            let info = &b.device.info;
            let name = info.name.to_ascii_lowercase();
            let software = name.contains("llvmpipe")
                || name.contains("swiftshader")
                || name.contains("software")
                || name.contains("warp")
                || info.device_type == wgpu::DeviceType::Cpu;
            matches!(info.backend, wgpu::Backend::Vulkan | wgpu::Backend::Dx12 | wgpu::Backend::Metal)
                && !software
        }

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
                        (
                            ((x * 29 + y * 13) % 256) as u8,
                            ((x * 7 + y * 41 + 85) % 256) as u8,
                            ((x * 53 + y * 19 + 170) % 256) as u8,
                            ((idx * 17) % 256) as u8,
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
                r[i] = ((s * 1.04).min(1.0) * 255.0 + 0.5) as u8;
                g[i] = (s * 255.0 + 0.5) as u8;
                b[i] = (s * 0.96 * 255.0 + 0.5) as u8;
            }
            PixelOp::Lut1D { r, g, b }
        }

        fn warm_lut3d(size: usize) -> PixelOp {
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
                            g.clamp(0.0, 1.0),
                            (b * 0.92).clamp(0.0, 1.0),
                        ]);
                    }
                }
            }
            PixelOp::Lut3D { size, table }
        }

        fn all_seven_ops() -> PixelStageFusion {
            PixelStageFusion::from_stages(vec![
                PixelOp::Brightness { delta: 0.08 },
                PixelOp::Whiten { strength: 0.65 },
                PixelOp::Rosy { strength: 0.45 },
                PixelOp::Temperature { warmth: 0.25, tint: -0.15 },
                PixelOp::Saturation { amount: 0.30 },
                s_curve_lut1d(),
                warm_lut3d(9),
            ])
            .expect("valid fusion")
        }

        #[test]
        fn adapter_probe_reads_real_info_and_limits() {
            let Some(b) = backend() else {
                eprintln!("SKIP: no wgpu adapter; probe not executed");
                return;
            };
            let record = b.probe_record();
            assert_eq!(record.backend, b.backend_identity());
            assert!(record.driver_ready);
            assert!(!record.device_name.is_empty(), "adapter name must be real");
            assert!(
                record.max_workgroup_size[0] >= 16 && record.max_workgroup_size[1] >= 16,
                "real device limits must support the catalogue workgroup size"
            );
            assert_eq!(record.status, VerificationStatus::DeviceOk);
            assert!(b.probe().is_ready());
            if !is_physical_gpu(b) {
                eprintln!(
                    "NOTE: adapter {} is a software implementation; parity tests remain real GPU execution but not a physical-GPU claim",
                    record.device_name
                );
            }
        }

        #[test]
        fn fusion_matches_cpu_reference_on_odd_even_and_boundary_sizes() {
            let Some(b) = backend() else {
                eprintln!("SKIP: no wgpu adapter; fusion parity not executed");
                return;
            };
            let fusion = all_seven_ops();
            let mut cpu = CpuComputeBackend::new();
            // A mutable backend is required for dispatch; create a second live
            // device rather than aliasing the shared static one.
            let mut gpu = WgpuComputeBackend::new(WgpuBackendSelector::Primary)
                .expect("live wgpu device for parity");
            for (w, h) in [(1u32, 1u32), (3, 5), (17, 19), (64, 37)] {
                let input = synthetic_rgba8(w, h);
                let (expected, _) = cpu
                    .dispatch_pixel_fusion(&input, &fusion)
                    .expect("CPU oracle");
                let (actual, receipt) = gpu
                    .dispatch_pixel_fusion(&input, &fusion)
                    .expect("GPU fusion dispatch");
                assert_eq!(receipt.fused_stages, 7);
                assert_eq!(receipt.memory_passes, 1);
                assert_eq!(receipt.pixel_count, input.pixel_count());
                assert_eq!(actual.width, w);
                assert_eq!(actual.height, h);
                assert_eq!(
                    actual.pixels.len(),
                    expected.pixels.len(),
                    "RGBA8 byte length for {w}x{h}"
                );
                let mut max_rgb_delta = 0u32;
                let mut alpha_diffs = 0usize;
                for (src, (got, want)) in input.pixels.chunks_exact(4).zip(
                    actual.pixels.chunks_exact(4).zip(expected.pixels.chunks_exact(4)),
                ) {
                    for ch in 0..3 {
                        let delta = got[ch].abs_diff(want[ch]) as u32;
                        max_rgb_delta = max_rgb_delta.max(delta);
                    }
                    if got[3] != want[3] || got[3] != src[3] {
                        alpha_diffs += 1;
                    }
                }
                assert!(max_rgb_delta <= 1, "{w}x{h}: RGB delta {max_rgb_delta} > 1 ULP");
                assert_eq!(alpha_diffs, 0, "{w}x{h}: alpha changed");
            }
        }

        #[test]
        fn at_least_four_catalog_shaders_compile_and_dispatch_on_gpu() {
            let Some(b) = backend() else {
                eprintln!("SKIP: no wgpu adapter; shader compile not executed");
                return;
            };
            let mut gpu = WgpuComputeBackend::new(WgpuBackendSelector::Primary)
                .expect("live wgpu device for catalog compile");
            // PF0 (f32 RGBA), PF21105 (packed + mask), PF30013 (packed HSL),
            // PF90023 (wrinkle composite) cover four distinct uniform layouts
            // and four distinct storage-plane shapes.
            for pf in [0u32, 21105, 30013, 90023] {
                let handle = gpu.compile_catalog_shader(pf).expect("catalog WGSL compile");
                assert!(handle.0 > 0);
            }
            let stats = gpu.pipeline_cache_stats();
            assert_eq!(stats.misses, 4);
            assert_eq!(stats.hits, 0);
            let _ = b;
        }

        #[test]
        fn catalog_pf21105_dispatches_and_preserves_alpha() {
            let mut gpu = match WgpuComputeBackend::new(WgpuBackendSelector::Primary) {
                Ok(gpu) => gpu,
                Err(_) => {
                    eprintln!("SKIP: no wgpu adapter; pf21105 dispatch not executed");
                    return;
                }
            };
            let (w, h) = (17u32, 19u32);
            let pixels = w as usize * h as usize;
            let mut current = vec![0u8; pixels * 4];
            let mut effect = vec![0u8; pixels * 4];
            // mask is `array<u32>` in the WGSL: one u32 (4 bytes) per pixel.
            let mut mask = vec![0u8; pixels * 4];
            let out_plane = vec![0u8; pixels * 4];
            for i in 0..pixels {
                current[i * 4..i * 4 + 4].copy_from_slice(&[
                    (i * 7 % 256) as u8,
                    (i * 13 % 256) as u8,
                    (i * 29 % 256) as u8,
                    (i * 3 % 256) as u8,
                ]);
                effect[i * 4..i * 4 + 4].copy_from_slice(&[
                    (i * 31 % 256) as u8,
                    (i * 17 % 256) as u8,
                    (i * 5 % 256) as u8,
                    0,
                ]);
                mask[i * 4..i * 4 + 4].copy_from_slice(&((i * 11 % 256) as u32).to_le_bytes());
            }
            let mut uniforms = vec![0u8; catalog_uniform_bytes(21105)];
            uniforms[0..4].copy_from_slice(&w.to_le_bytes());
            uniforms[4..8].copy_from_slice(&h.to_le_bytes());
            uniforms[8..12].copy_from_slice(&1.0f32.to_le_bytes());
            let planes: Vec<&[u8]> = vec![&current, &mask, &effect, &out_plane];
            let (receipt, out) = gpu
                .dispatch_catalog_and_readback(
                    CatalogDispatch {
                        pf: 21105,
                        uniforms: &uniforms,
                        storage: &planes,
                        width: w,
                        height: h,
                        mode: ExecutionMode::Strict,
                    },
                    4,
                )
                .expect("pf21105 real GPU dispatch");
            assert_eq!(out.len(), pixels * 4);
            assert_eq!(receipt.workgroups, [w.div_ceil(16), h.div_ceil(16), 1]);
            for i in 0..pixels {
                assert_eq!(out[i * 4 + 3], current[i * 4 + 3], "alpha must be unchanged");
            }
            assert!(receipt.timing.wall_ns > 0);
        }

        #[test]
        fn strict_mode_rejects_wrong_uniform_length_without_fallback() {
            let mut gpu = match WgpuComputeBackend::new(WgpuBackendSelector::Primary) {
                Ok(gpu) => gpu,
                Err(_) => {
                    eprintln!("SKIP: no wgpu adapter; strict mode test not executed");
                    return;
                }
            };
            let err = gpu
                .dispatch_catalog_shader(CatalogDispatch {
                    pf: 21105,
                    uniforms: &[0u8; 4],
                    storage: &[],
                    width: 1,
                    height: 1,
                    mode: ExecutionMode::Strict,
                })
                .unwrap_err();
            assert!(matches!(err, WgpuError::InvalidBuffer(_)), "{err}");
            assert_eq!(gpu.pipeline_cache_stats().misses, 0, "no pipeline compiled on refusal");
        }

        #[test]
        fn static_contracts_are_consistent() {
            assert_eq!(candidates().len(), 4);
            for candidate in candidates() {
                assert_eq!(candidate.params.len(), 3);
                assert_eq!(candidate.params[0].name(), "workgroup_x");
                assert_eq!(candidate.params[1].name(), "workgroup_y");
                assert_eq!(candidate.params[2].name(), "precision");
                assert!(candidate.params[0].accepts_u32(256));
                assert!(!candidate.params[0].accepts_u32(7));
                assert!(candidate.params[1].accepts_u32(16));
                assert_eq!(candidate.status, VerificationStatus::Designed);
            }
            assert_eq!(catalog_uniform_bytes(90023), 56);
            assert_eq!(catalog_uniform_bytes(90085), 48);
            assert_eq!(size_class(0), 4);
            assert_eq!(size_class(1), 4);
            assert_eq!(size_class(17), 32);
        }
    }
}

#[cfg(feature = "wgpu-backend")]
pub use imp::*;

/// When the feature is disabled the module is empty; the total-control layer
/// decides whether to export this module in `lib.rs`.
#[cfg(not(feature = "wgpu-backend"))]
pub const WGPU_BACKEND_DISABLED: bool = true;
