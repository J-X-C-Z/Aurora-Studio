//! AMD HIP / ROCm platform adapter for fused pixel-stage execution.
//!
//! Vendor libraries are never linked at build time. On Windows the adapter
//! resolves `amdhip64.dll` with `LoadLibraryA`/`GetProcAddress`; on Linux it
//! resolves `libamdhip64.so` with `dlopen`/`dlsym`. Both require a real driver
//! installation at runtime. When no runtime, device or kernel image exists,
//! every entry point returns an explicit
//! `ComputeError::BackendUnavailable` — no CPU fallback is substituted.
//!
//! The kernel source at `native/hip/pixel_fusion.hip` is embedded verbatim as a
//! source-level candidate. It must be compiled to a HIP module (fatbin/HSACO)
//! by a real ROCm toolchain before the dispatcher can run; this crate does not
//! ship a compiled AMD binary and does not claim parity or performance.

use std::collections::HashMap;
use std::ffi::c_void;

use crate::autotune::{CandidateSpec, ExecutionMode, ParamBound, VerificationStatus};
use crate::compute_backend::{
    BufferId, BufferPoolStats, BufferUsage, ComputeBackend, ComputeError, ExecutionReceipt,
    PipelineCacheStats, PipelineHandle, PipelineKey, PixelStageFusion,
};
use crate::device::{Backend, DeviceProbe, Precision};
use crate::metal_backend::fusion_params;
use crate::Image;

/// Verbatim HIP kernel source reused by this backend.
pub const PIXEL_FUSION_HIP: &str = include_str!("../native/hip/pixel_fusion.hip");
pub const PIXEL_FUSION_ENTRY: &str = "pixel_fusion";
pub const HIP_UNAVAILABLE: &str = "AMD HIP runtime unavailable";

#[cfg(windows)]
const HIP_LIBRARY: &str = "amdhip64.dll";
#[cfg(target_os = "linux")]
const HIP_LIBRARY: &str = "libamdhip64.so";
#[cfg(not(any(windows, target_os = "linux")))]
const HIP_LIBRARY: &str = "amdhip64";

/// One `hipDeviceProp_t` field subset. `integrated != 0` identifies an APU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HipDeviceInfo {
    pub name: String,
    pub integrated: i32,
    pub wavefront_size: u32,
    pub warp_size: u32,
    pub max_threads_per_block: u32,
    pub shared_mem_per_block: u32,
    pub total_global_mem: u64,
    pub major: i32,
    pub minor: i32,
}

impl HipDeviceInfo {
    pub fn is_apu(&self) -> bool {
        self.integrated != 0
    }

    /// ROCm reports wavefront size through `wavefrontSize` on recent runtimes;
    /// older runtimes only expose `warpSize`.
    pub fn wave(&self) -> WaveSize {
        match self.wavefront_size.max(self.warp_size) {
            32 => WaveSize::Wave32,
            64 => WaveSize::Wave64,
            _ => WaveSize::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaveSize {
    Wave32,
    Wave64,
    Unknown,
}

impl WaveSize {
    pub const fn lanes(self) -> u32 {
        match self {
            Self::Wave32 => 32,
            Self::Wave64 => 64,
            Self::Unknown => 0,
        }
    }
}

/// Resource budget applied before any dispatch. An APU shares system memory
/// with the CPU, so staging, tile size and concurrent dispatches are capped to
/// avoid the prefetch stream starving either side of the memory controller.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HipBudget {
    pub max_staging_bytes: u64,
    pub max_shared_mem_bytes: u32,
    pub max_threads_per_group: u32,
    pub max_concurrent_dispatches: u32,
    pub tile_pixels: u32,
}

impl HipBudget {
    pub const fn apu() -> Self {
        Self {
            max_staging_bytes: 64 * 1024 * 1024,
            max_shared_mem_bytes: 16 * 1024,
            max_threads_per_group: 256,
            max_concurrent_dispatches: 1,
            tile_pixels: 128,
        }
    }

    pub const fn discrete() -> Self {
        Self {
            max_staging_bytes: 512 * 1024 * 1024,
            max_shared_mem_bytes: 48 * 1024,
            max_threads_per_group: 1024,
            max_concurrent_dispatches: 4,
            tile_pixels: 512,
        }
    }

    pub fn for_device(info: &HipDeviceInfo) -> Self {
        if info.is_apu() {
            Self::apu()
        } else {
            Self::discrete()
        }
    }

    pub fn accepts(self, bytes: usize, shared: u32, threads: u32) -> bool {
        bytes as u64 <= self.max_staging_bytes
            && shared <= self.max_shared_mem_bytes
            && threads <= self.max_threads_per_group
    }

    /// Strict mode must never promote a candidate that does not fit the
    /// measured device budget.
    pub fn admits(self, candidate: &CandidateSpec, mode: ExecutionMode) -> bool {
        let fits = candidate.max_shared_mem_bytes <= self.max_shared_mem_bytes
            && candidate.max_threads_per_group <= self.max_threads_per_group;
        if mode == ExecutionMode::Strict && !fits {
            return false;
        }
        fits || mode == ExecutionMode::AutoFallback
    }
}

static WAVE32_PARAMS: &[ParamBound] = &[
    ParamBound::DiscreteU32 {
        name: "block_size",
        values: &[32, 64, 128, 256],
    },
    ParamBound::DiscreteU32 {
        name: "wave_size",
        values: &[32],
    },
    ParamBound::U32Range {
        name: "tile_pixels",
        min: 64,
        max: 512,
        step: 64,
    },
];

static WAVE64_PARAMS: &[ParamBound] = &[
    ParamBound::DiscreteU32 {
        name: "block_size",
        values: &[64, 128, 256, 512, 1024],
    },
    ParamBound::DiscreteU32 {
        name: "wave_size",
        values: &[64],
    },
    ParamBound::U32Range {
        name: "tile_pixels",
        min: 128,
        max: 1024,
        step: 128,
    },
];

const fn candidate(id: &'static str, params: &'static [ParamBound], smem: u32, threads: u32) -> CandidateSpec {
    CandidateSpec {
        id,
        algorithm_id: "pixel_fusion",
        backend: Backend::Hip,
        kernel_entry: PIXEL_FUSION_ENTRY,
        precision: Precision::Fp32,
        params,
        max_shared_mem_bytes: smem,
        max_threads_per_group: threads,
        status: VerificationStatus::Designed,
    }
}

/// Wave32/Wave64 parameter space. 32-lane wave candidates include the APU
/// boundary values; 64-lane candidates cover CDNA/RDNA dGPUs.
pub static HIP_CANDIDATES: &[CandidateSpec] = &[
    candidate("hip.wave32.rdna2.apu", WAVE32_PARAMS, 16 * 1024, 256),
    candidate("hip.wave32.rdna3.apu", WAVE32_PARAMS, 16 * 1024, 256),
    candidate("hip.wave32.rdna3.dgpu", WAVE32_PARAMS, 32 * 1024, 512),
    candidate("hip.wave32.rdna4.dgpu", WAVE32_PARAMS, 32 * 1024, 1024),
    candidate("hip.wave64.gcn.dgpu", WAVE64_PARAMS, 48 * 1024, 1024),
    candidate("hip.wave64.cdna.dgpu", WAVE64_PARAMS, 48 * 1024, 1024),
    candidate("hip.wave64.rdna2.dgpu", WAVE64_PARAMS, 32 * 1024, 1024),
];

fn validate_wave_bounds(candidate: &CandidateSpec) -> bool {
    let wave = candidate
        .params
        .iter()
        .find(|bound| bound.name() == "wave_size");
    match wave {
        Some(ParamBound::DiscreteU32 { values, .. }) => {
            values.iter().all(|value| *value == 32 || *value == 64)
                && candidate.max_threads_per_group % values[0].max(1) == 0
        }
        _ => false,
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct HipDeviceProp {
    name: [u8; 256],
    total_global_mem: usize,
    shared_mem_per_block: usize,
    max_threads_per_block: i32,
    warp_size: i32,
    major: i32,
    minor: i32,
    integrated: i32,
    wavefront_size: u32,
}

#[cfg(windows)]
mod loader {
    use super::*;
    use std::ffi::{CStr, c_char, c_int, c_void};

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    #[derive(Clone, Copy)]
    pub struct HipApi {
        pub init: unsafe extern "C" fn(c_int) -> c_int,
        pub get_device_count: unsafe extern "C" fn(*mut c_int) -> c_int,
        pub get_properties: unsafe extern "C" fn(*mut HipDeviceProp, c_int) -> c_int,
        pub malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int,
        pub free: unsafe extern "C" fn(*mut c_void) -> c_int,
        pub memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int,
        pub module_load_data: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> c_int,
        pub module_get_function: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int,
        pub launch: unsafe extern "C" fn(
            *mut c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            usize,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> c_int,
        pub sync: unsafe extern "C" fn() -> c_int,
    }

    pub fn load() -> Result<HipApi, ComputeError> {
        unsafe {
            let handle = LoadLibraryA(b"amdhip64.dll\0".as_ptr() as *const c_char);
            if handle.is_null() {
                return Err(ComputeError::BackendUnavailable(HIP_UNAVAILABLE));
            }
            macro_rules! sym {
                ($name:literal) => {{
                    let address = GetProcAddress(handle, concat!($name, "\0").as_ptr() as *const c_char);
                    if address.is_null() {
                        return Err(ComputeError::BackendUnavailable(concat!("HIP symbol missing: ", $name)));
                    }
                    std::mem::transmute::<*mut c_void, _>(address)
                }};
            }
            Ok(HipApi {
                init: sym!("hipInit"),
                get_device_count: sym!("hipGetDeviceCount"),
                get_properties: sym!("hipGetDeviceProperties"),
                malloc: sym!("hipMalloc"),
                free: sym!("hipFree"),
                memcpy: sym!("hipMemcpy"),
                module_load_data: sym!("hipModuleLoadData"),
                module_get_function: sym!("hipModuleGetFunction"),
                launch: sym!("hipModuleLaunchKernel"),
                sync: sym!("hipDeviceSynchronize"),
            })
        }
    }

    pub fn device_name(prop: &HipDeviceProp) -> String {
        let bytes = prop.name.split(|byte| *byte == 0).next().unwrap_or(&[]);
        unsafe { CStr::from_ptr(bytes.as_ptr() as *const c_char) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(unix)]
mod loader {
    use super::*;
    use std::ffi::{CStr, c_char, c_int, c_void};

    const RTLD_NOW: c_int = 2;

    unsafe extern "C" {
        fn dlopen(path: *const c_char, mode: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    #[derive(Clone, Copy)]
    pub struct HipApi {
        pub init: unsafe extern "C" fn(c_int) -> c_int,
        pub get_device_count: unsafe extern "C" fn(*mut c_int) -> c_int,
        pub get_properties: unsafe extern "C" fn(*mut HipDeviceProp, c_int) -> c_int,
        pub malloc: unsafe extern "C" fn(*mut *mut c_void, usize) -> c_int,
        pub free: unsafe extern "C" fn(*mut c_void) -> c_int,
        pub memcpy: unsafe extern "C" fn(*mut c_void, *const c_void, usize, c_int) -> c_int,
        pub module_load_data: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> c_int,
        pub module_get_function: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> c_int,
        pub launch: unsafe extern "C" fn(
            *mut c_void,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            usize,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> c_int,
        pub sync: unsafe extern "C" fn() -> c_int,
    }

    pub fn load() -> Result<HipApi, ComputeError> {
        unsafe {
            let path = std::ffi::CString::new(HIP_LIBRARY).expect("library name has no NUL");
            let handle = dlopen(path.as_ptr(), RTLD_NOW);
            if handle.is_null() {
                return Err(ComputeError::BackendUnavailable(HIP_UNAVAILABLE));
            }
            macro_rules! sym {
                ($name:literal) => {{
                    let name = std::ffi::CString::new($name).expect("symbol has no NUL");
                    let address = dlsym(handle, name.as_ptr());
                    if address.is_null() {
                        return Err(ComputeError::BackendUnavailable(concat!("HIP symbol missing: ", $name)));
                    }
                    std::mem::transmute::<*mut c_void, _>(address)
                }};
            }
            Ok(HipApi {
                init: sym!("hipInit"),
                get_device_count: sym!("hipGetDeviceCount"),
                get_properties: sym!("hipGetDeviceProperties"),
                malloc: sym!("hipMalloc"),
                free: sym!("hipFree"),
                memcpy: sym!("hipMemcpy"),
                module_load_data: sym!("hipModuleLoadData"),
                module_get_function: sym!("hipModuleGetFunction"),
                launch: sym!("hipModuleLaunchKernel"),
                sync: sym!("hipDeviceSynchronize"),
            })
        }
    }

    pub fn device_name(prop: &HipDeviceProp) -> String {
        let bytes = prop.name.split(|byte| *byte == 0).next().unwrap_or(&[]);
        unsafe { CStr::from_ptr(bytes.as_ptr() as *const c_char) }
            .to_string_lossy()
            .into_owned()
    }
}


enum HipState {
    Unavailable(&'static str),
    Ready {
        api: loader::HipApi,
        info: HipDeviceInfo,
        budget: HipBudget,
        module: *mut c_void,
        function: *mut c_void,
    },
}

#[derive(Debug)]
struct BufferEntry {
    data: Vec<u8>,
    usage: BufferUsage,
}

/// HIP implementation of `ComputeBackend`.
///
/// Construction is inert. [`Self::open`] resolves the vendor library, selects
/// device 0, probes APU/dGPU and attempts to load a compiled HIP module from
/// `hip_module_bytes`. No such binary is bundled: without one the open fails
/// explicitly rather than pretending to run on the GPU.
pub struct HipComputeBackend {
    state: HipState,
    hip_module_bytes: Vec<u8>,
    next_buffer_id: u64,
    next_pipeline_id: u64,
    active: HashMap<BufferId, BufferEntry>,
    free_pool: Vec<Vec<u8>>,
    allocations: u64,
    reuses: u64,
    pipeline_cache: HashMap<PipelineKey, PipelineHandle>,
    hits: u64,
    misses: u64,
    last_mode: ExecutionMode,
}

impl Default for HipComputeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl HipComputeBackend {
    pub fn new() -> Self {
        Self {
            state: HipState::Unavailable(HIP_UNAVAILABLE),
            hip_module_bytes: Vec::new(),
            next_buffer_id: 1,
            next_pipeline_id: 1,
            active: HashMap::new(),
            free_pool: Vec::new(),
            allocations: 0,
            reuses: 0,
            pipeline_cache: HashMap::new(),
            hits: 0,
            misses: 0,
            last_mode: ExecutionMode::Strict,
        }
    }

    /// Install a compiled HIP module image (fatbin/HSACO) from the caller.
    /// Passing the `.hip` source text is refused: `hipModuleLoadData` needs a
    /// real compiled artifact.
    pub fn with_module(mut self, bytes: Vec<u8>) -> Result<Self, ComputeError> {
        if bytes.starts_with(b"//") || bytes.starts_with(b"#include") {
            return Err(ComputeError::BackendUnavailable(
                "HIP module requires compiled fatbin/HSACO bytes, not source",
            ));
        }
        self.hip_module_bytes = bytes;
        Ok(self)
    }

    pub fn with_mode(mut self, mode: ExecutionMode) -> Self {
        self.last_mode = mode;
        self
    }

    /// Open the HIP runtime and load the module. Missing runtime, no device,
    /// or an unloadable kernel are typed refusals.
    pub fn open(mut self) -> Result<Self, ComputeError> {
        self.open_in_place()?;
        Ok(self)
    }

    fn open_in_place(&mut self) -> Result<(), ComputeError> {
        if self.hip_module_bytes.is_empty() {
            return Err(ComputeError::BackendUnavailable(
                "HIP kernel module binary is not supplied",
            ));
        }
        let api = loader::load()?;
        unsafe {
            if (api.init)(0) != 0 {
                return Err(ComputeError::BackendUnavailable("hipInit failed"));
            }
            let mut count = 0i32;
            if (api.get_device_count)(&mut count) != 0 || count <= 0 {
                return Err(ComputeError::BackendUnavailable(
                    "no AMD HIP devices reported",
                ));
            }
            let mut prop: HipDeviceProp = std::mem::zeroed();
            if (api.get_properties)(&mut prop, 0) != 0 {
                return Err(ComputeError::BackendUnavailable(
                    "hipGetDeviceProperties failed",
                ));
            }
            let info = HipDeviceInfo {
                name: loader::device_name(&prop),
                integrated: prop.integrated,
                wavefront_size: prop.wavefront_size,
                warp_size: prop.warp_size as u32,
                max_threads_per_block: prop.max_threads_per_block as u32,
                shared_mem_per_block: prop.shared_mem_per_block as u32,
                total_global_mem: prop.total_global_mem as u64,
                major: prop.major,
                minor: prop.minor,
            };
            let budget = HipBudget::for_device(&info);
            if info.max_threads_per_block < budget.max_threads_per_group {
                return Err(ComputeError::BackendUnavailable(
                    "device thread budget below HIP candidate space",
                ));
            }
            let mut module: *mut c_void = std::ptr::null_mut();
            if (api.module_load_data)(&mut module, self.hip_module_bytes.as_ptr() as *const c_void) != 0
                || module.is_null()
            {
                return Err(ComputeError::BackendUnavailable(
                    "hipModuleLoadData rejected the module image",
                ));
            }
            let name = std::ffi::CString::new(PIXEL_FUSION_ENTRY)
                .expect("kernel entry has no NUL");
            let mut function: *mut c_void = std::ptr::null_mut();
            if (api.module_get_function)(&mut function, module, name.as_ptr()) != 0
                || function.is_null()
            {
                return Err(ComputeError::BackendUnavailable(
                    "pixel_fusion kernel entry missing",
                ));
            }
            self.state = HipState::Ready {
                api,
                info,
                budget,
                module,
                function,
            };
        }
        Ok(())
    }

    fn unavailable(&self) -> ComputeError {
        match &self.state {
            HipState::Unavailable(reason) => ComputeError::BackendUnavailable(reason),
            HipState::Ready { .. } => ComputeError::BackendUnavailable("HIP device is ready"),
        }
    }

    pub fn device_info(&self) -> Result<HipDeviceInfo, ComputeError> {
        match &self.state {
            HipState::Ready { info, .. } => Ok(info.clone()),
            HipState::Unavailable(reason) => Err(ComputeError::BackendUnavailable(reason)),
        }
    }

    pub fn budget(&self) -> Result<HipBudget, ComputeError> {
        match &self.state {
            HipState::Ready { budget, .. } => Ok(*budget),
            HipState::Unavailable(reason) => Err(ComputeError::BackendUnavailable(reason)),
        }
    }

    pub fn trim_pool(&mut self) {
        self.free_pool.clear();
    }
}

impl ComputeBackend for HipComputeBackend {
    fn backend(&self) -> Backend {
        Backend::Hip
    }

    fn probe(&self) -> DeviceProbe {
        let ready = matches!(self.state, HipState::Ready { .. });
        DeviceProbe {
            backend: Backend::Hip,
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
        if !matches!(self.state, HipState::Ready { .. }) {
            return Err(self.unavailable());
        }
        let budget = self.budget()?;
        if !budget.accepts(byte_len, 0, 1) {
            return Err(ComputeError::InvalidParameter("HIP APU staging budget exceeded"));
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
        if !matches!(self.state, HipState::Ready { .. }) {
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
        if !matches!(self.state, HipState::Ready { .. }) {
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
        if !matches!(self.state, HipState::Ready { .. }) {
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
        if !matches!(self.state, HipState::Ready { .. }) {
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
        let budget = self.budget()?;
        if !budget.accepts(0, 0, budget.max_threads_per_group) {
            return Err(ComputeError::BackendUnavailable("HIP budget rejected pipeline"));
        }
        self.misses += 1;
        let handle = PipelineHandle(self.next_pipeline_id);
        self.next_pipeline_id += 1;
        self.pipeline_cache.insert(key, handle);
        Ok(handle)
    }

    fn synchronize(&mut self) -> Result<(), ComputeError> {
        match &self.state {
            HipState::Ready { api, .. } => {
                let code = unsafe { (api.sync)() };
                if code == 0 {
                    Ok(())
                } else {
                    Err(ComputeError::BackendUnavailable("hipDeviceSynchronize failed"))
                }
            }
            HipState::Unavailable(reason) => Err(ComputeError::BackendUnavailable(reason)),
        }
    }

    fn dispatch_pixel_fusion(
        &mut self,
        input: &Image,
        fusion: &PixelStageFusion,
    ) -> Result<(Image, ExecutionReceipt), ComputeError> {
        if !matches!(self.state, HipState::Ready { .. }) {
            return Err(self.unavailable());
        }
        input.validate().map_err(ComputeError::InvalidImage)?;
        let pipeline = self.get_or_compile_pipeline(fusion)?;
        let params = fusion_params(fusion);
        let byte_len = input.pixels.len();
        let budget = self.budget()?;
        if !budget.accepts(byte_len, 0, budget.max_threads_per_group) {
            return Err(ComputeError::InvalidParameter("HIP dispatch exceeds APU budget"));
        }
        let src = self.allocate_buffer(byte_len, BufferUsage::StorageRead)?;
        let dst = self.allocate_buffer(byte_len, BufferUsage::StorageWrite)?;
        self.upload_buffer(src, &input.pixels)?;

        #[cfg(any(windows, target_os = "linux"))]
        {
            use std::ffi::c_void;
            let (api, function) = match &self.state {
                HipState::Ready { api, function, .. } => (*api, *function),
                HipState::Unavailable(_) => return Err(self.unavailable()),
            };
            let source = self
                .active
                .get(&src)
                .ok_or(ComputeError::BufferNotFound(src))?
                .data
                .clone();
            let mut output = vec![0u8; byte_len];
            let mut device_src: *mut c_void = std::ptr::null_mut();
            let mut device_dst: *mut c_void = std::ptr::null_mut();
            let mut device_params: *mut c_void = std::ptr::null_mut();
            unsafe {
                if (api.malloc)(&mut device_src, source.len().max(1)) != 0
                    || (api.malloc)(&mut device_dst, output.len().max(1)) != 0
                    || (api.malloc)(&mut device_params, params.len()) != 0
                {
                    if !device_src.is_null() {
                        let _ = (api.free)(device_src);
                    }
                    if !device_dst.is_null() {
                        let _ = (api.free)(device_dst);
                    }
                    if !device_params.is_null() {
                        let _ = (api.free)(device_params);
                    }
                    return Err(ComputeError::BackendUnavailable("hipMalloc failed"));
                }
                // hipMemcpy kind 1 = HostToDevice, 2 = DeviceToHost.
                let ok = (api.memcpy)(
                    device_src,
                    source.as_ptr() as *const c_void,
                    source.len(),
                    1,
                ) == 0
                    && (api.memcpy)(
                        device_params,
                        params.as_ptr() as *const c_void,
                        params.len(),
                        1,
                    ) == 0;
                if !ok {
                    let _ = (api.free)(device_src);
                    let _ = (api.free)(device_dst);
                    let _ = (api.free)(device_params);
                    return Err(ComputeError::BackendUnavailable("hipMemcpy H2D failed"));
                }
                // HIP passes all five by-value/pointer arguments; the two LUT
                // buffers are nullable because the fused stages here are the
                // point operations, not LUT payloads.
                let mut args: [*mut c_void; 5] = [
                    device_params,
                    device_src,
                    device_dst,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                ];
                let threads = budget.max_threads_per_group.min(256).max(32);
                let blocks = (input.pixel_count() as u32).div_ceil(threads).max(1);
                let status = (api.launch)(
                    function,
                    blocks,
                    1,
                    1,
                    threads,
                    1,
                    1,
                    0,
                    args.as_mut_ptr(),
                    std::ptr::null_mut(),
                );
                let synced = (api.sync)() == 0;
                let read_back = synced
                    && (api.memcpy)(
                        output.as_mut_ptr() as *mut c_void,
                        device_dst,
                        output.len(),
                        2,
                    ) == 0;
                let _ = (api.free)(device_src);
                let _ = (api.free)(device_dst);
                let _ = (api.free)(device_params);
                if status != 0 || !read_back {
                    return Err(ComputeError::BackendUnavailable("HIP dispatch failed"));
                }
            }
            self.upload_buffer(dst, &output)?;
        }

        let out_pixels = self.download_buffer(dst)?;
        self.release_buffer(src)?;
        self.release_buffer(dst)?;
        self.synchronize()?;
        let receipt = ExecutionReceipt {
            backend: Backend::Hip,
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

impl Drop for HipComputeBackend {
    fn drop(&mut self) {
        if let HipState::Ready { api, .. } = &self.state {
            unsafe {
                let _ = (api.sync)();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apu() -> HipDeviceInfo {
        HipDeviceInfo {
            name: "AMD Radeon Graphics".into(),
            integrated: 1,
            wavefront_size: 32,
            warp_size: 32,
            max_threads_per_block: 256,
            shared_mem_per_block: 16 * 1024,
            total_global_mem: 8 * 1024 * 1024 * 1024,
            major: 10,
            minor: 1,
        }
    }

    fn dgpu() -> HipDeviceInfo {
        HipDeviceInfo {
            name: "AMD Radeon RX 7900 XTX".into(),
            integrated: 0,
            wavefront_size: 64,
            warp_size: 64,
            max_threads_per_block: 1024,
            shared_mem_per_block: 64 * 1024,
            total_global_mem: 24 * 1024 * 1024 * 1024,
            major: 11,
            minor: 0,
        }
    }

    #[test]
    fn hip_source_is_complete_and_kernel_signature_is_declared() {
        assert!(PIXEL_FUSION_HIP.contains("extern \"C\" __global__ void pixel_fusion"));
        assert!(PIXEL_FUSION_HIP.contains("#include <hip/hip_runtime.h>"));
        assert!(PIXEL_FUSION_HIP.contains("#pragma clang fp contract(off)"));
        assert!(PIXEL_FUSION_HIP.contains("sizeof(PixelFusionParams) == 52"));
        assert!(PIXEL_FUSION_HIP.contains("__global__"));
        assert_eq!(PIXEL_FUSION_HIP.matches("pixel_fusion(").count(), 1);
    }

    #[test]
    fn hip_library_name_matches_the_host_platform() {
        #[cfg(windows)]
        assert_eq!(HIP_LIBRARY, "amdhip64.dll");
        #[cfg(target_os = "linux")]
        assert_eq!(HIP_LIBRARY, "libamdhip64.so");
        #[cfg(not(any(windows, target_os = "linux")))]
        assert_eq!(HIP_LIBRARY, "amdhip64");
        let _ = HIP_LIBRARY;
    }

    #[test]
    fn hip_driver_symbols_are_resolved_by_name() {
        // The minimal runtime surface required by the dispatcher. No import
        // library is linked: these are passed straight to GetProcAddress/dlsym.
        let symbols = [
            "hipInit",
            "hipGetDeviceCount",
            "hipGetDeviceProperties",
            "hipMalloc",
            "hipFree",
            "hipMemcpy",
            "hipModuleLoadData",
            "hipModuleGetFunction",
            "hipModuleLaunchKernel",
            "hipDeviceSynchronize",
        ];
        assert_eq!(symbols.len(), 10);
        assert!(symbols.iter().all(|name| name.starts_with("hip")));
    }

    #[test]
    fn missing_runtime_is_an_explicit_refusal_not_cpu_fallback() {
        let mut backend = HipComputeBackend::new();
        assert_eq!(backend.backend(), Backend::Hip);
        assert_eq!(backend.probe().driver_ready, false);
        assert_eq!(backend.device_info(), Err(ComputeError::BackendUnavailable(HIP_UNAVAILABLE)));
        assert_eq!(backend.budget(), Err(ComputeError::BackendUnavailable(HIP_UNAVAILABLE)));
        let image = Image::solid(2, 2, [1, 2, 3, 4]);
        assert_eq!(
            backend.dispatch_pixel_fusion(&image, &PixelStageFusion::new()).err(),
            Some(ComputeError::BackendUnavailable(HIP_UNAVAILABLE))
        );
        assert_eq!(backend.allocate_buffer(16, BufferUsage::StorageRead).err(), Some(ComputeError::BackendUnavailable(HIP_UNAVAILABLE)));
        assert_eq!(backend.synchronize(), Err(ComputeError::BackendUnavailable(HIP_UNAVAILABLE)));
    }

    #[test]
    fn strict_mode_rejects_oversized_candidate_and_fits_apu() {
        let apu_budget = HipBudget::for_device(&apu());
        let dgpu_budget = HipBudget::for_device(&dgpu());
        assert!(apu_budget.max_concurrent_dispatches < dgpu_budget.max_concurrent_dispatches);
        assert!(apu_budget.tile_pixels < dgpu_budget.tile_pixels);
        assert!(apu_budget.max_shared_mem_bytes < dgpu_budget.max_shared_mem_bytes);
        let small = HIP_CANDIDATES.iter().find(|c| c.id.contains("apu")).unwrap();
        let large = HIP_CANDIDATES.iter().find(|c| c.id.contains("cdna")).unwrap();
        assert!(apu_budget.admits(small, ExecutionMode::Strict));
        assert!(!apu_budget.admits(large, ExecutionMode::Strict));
        assert!(dgpu_budget.admits(large, ExecutionMode::Strict));
        let backend = HipComputeBackend::new().with_mode(ExecutionMode::Strict);
        assert_eq!(backend.last_mode, ExecutionMode::Strict);
    }

    #[test]
    fn apu_and_dgpu_budgets_are_distinct_and_bounded() {
        assert!(apu().is_apu());
        assert!(!dgpu().is_apu());
        assert_eq!(apu().wave(), WaveSize::Wave32);
        assert_eq!(dgpu().wave(), WaveSize::Wave64);
        assert_eq!(apu().wave().lanes(), 32);
        assert_eq!(dgpu().wave().lanes(), 64);
        assert_eq!(
            HipDeviceInfo { wavefront_size: 0, warp_size: 0, ..apu() }.wave(),
            WaveSize::Unknown
        );
        assert!(HipBudget::apu().accepts(16, 1024, 128));
        assert!(!HipBudget::apu().accepts(65 * 1024 * 1024, 0, 1));
        assert!(!HipBudget::apu().accepts(16, 16 * 1024 + 1, 1));
        assert!(HipBudget::discrete().accepts(16, 16 * 1024 + 1, 512));
    }

    #[test]
    fn wave32_wave64_candidate_bounds_are_valid() {
        for candidate in HIP_CANDIDATES {
            assert!(validate_wave_bounds(candidate), "{}", candidate.id);
            assert_eq!(candidate.status, VerificationStatus::Designed);
            assert!(candidate.max_threads_per_group >= 32);
            assert!(candidate.max_threads_per_group <= 1024);
        }
        let wave32 = HIP_CANDIDATES.iter().find(|c| c.id.contains("wave32")).unwrap();
        let wave64 = HIP_CANDIDATES.iter().find(|c| c.id.contains("wave64")).unwrap();
        assert_ne!(wave32.params, wave64.params);
        assert!(wave32.params.iter().any(|p| p.name() == "wave_size"));
        assert!(wave64.params.iter().any(|p| p.name() == "wave_size"));
    }

    #[test]
    fn source_is_refused_as_a_hip_module_binary() {
        let result = HipComputeBackend::new().with_module(PIXEL_FUSION_HIP.as_bytes().to_vec());
        assert_eq!(
            result.err(),
            Some(ComputeError::BackendUnavailable(
                "HIP module requires compiled fatbin/HSACO bytes, not source"
            ))
        );
        let backend = HipComputeBackend::new().with_module(vec![0u8; 16]).unwrap();
        match backend.open() {
            Ok(_) => panic!("a fake 16-byte module must not load on a real driver"),
            Err(error) => match error {
                ComputeError::BackendUnavailable(_) | ComputeError::InvalidParameter(_) => {}
                other => panic!("unexpected HIP refusal: {other:?}"),
            },
        }
    }
}
