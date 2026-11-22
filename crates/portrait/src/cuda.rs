//! NVIDIA execution path for the P15 `FacialSmoothProcessV2` pixel law.
//!
//! The PTX in `native/cuda/` is the recovered source-order translation compiled
//! with `-fmad=false`. It is JIT-compiled by the installed driver through the
//! CUDA driver API, which is loaded dynamically: this crate links no CUDA
//! import library and gains no dependency.
//!
//! Inputs are the same live-job inputs the CPU lane needs. A missing driver,
//! an unloadable kernel, a short buffer or a non-square ROI is a typed refusal.
//! Producing pixels on the CPU is never reported as GPU success: [`Dispatch`]
//! names the kernel and geometry that actually ran.

use crate::kernels::facial_smooth as cpu;
use crate::{Image, Refusal};

/// Recovered kernel entry points, in job order.
pub const COMPOSITE: &str = "p15_facial_smooth_composite_kernel";
pub const SURFACE: &str = "p15_surface_pass_kernel";
pub const GAUSS: &str = "p15_gauss_pass_kernel";
pub const BOX_BLUR: &str = "p15_box_blur_kernel";

const PTX: &str = include_str!("../native/cuda/p15_facial_smooth.ptx");

/// The vendor gate value the adapter refuses to guess.
pub const GATE: f32 = 0.001;

/// 88-byte kernel argument block, field-for-field with
/// `P15FacialSmoothUniforms` in `native/cuda/p15_facial_smooth_public.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub pf: u32,
    pub pf_value: f32,
    pub gate: f32,
    pub legal_alpha: f32,
    pub dark_circle_alpha: f32,
    pub diff_factor: f32,
    pub diff_factor2: f32,
    pub color_threshold_factor: f32,
    pub mix_alpha: f32,
    pub diff_min_threshold: f32,
    pub diff_max_threshold: f32,
    pub high_frequency_min_threshold: f32,
    pub high_frequency_max_threshold: f32,
    pub surface_radius: f32,
    pub surface_threshold: f32,
    pub gauss_offset: f32,
    pub roi_size: u32,
    pub algo_type: u32,
    pub box_blur_stage: u32,
}

impl Uniforms {
    /// Builds the block for one square ROI job from the CPU lane's uniforms.
    ///
    /// Only `SurfaceBlurBasic` (algo 0) is recovered, so `algo_type` is 0. The
    /// blur-pass fields stay zero until a caller sets them for `SURFACE`,
    /// `GAUSS` or `BOX_BLUR`; the composite kernel does not read them.
    pub fn new(width: u32, height: u32, pf: u32, pf_value: f32, uniforms: &cpu::Uniforms) -> Self {
        Self {
            width,
            height,
            stride: width,
            pf,
            pf_value,
            gate: GATE,
            legal_alpha: uniforms.legal_alpha,
            dark_circle_alpha: uniforms.dark_circle_alpha,
            diff_factor: uniforms.diff_factor,
            diff_factor2: uniforms.diff_factor2,
            color_threshold_factor: uniforms.color_threshold_factor,
            mix_alpha: uniforms.mix_alpha,
            diff_min_threshold: uniforms.diff_min_threshold,
            diff_max_threshold: uniforms.diff_max_threshold,
            high_frequency_min_threshold: uniforms.high_freq_min_threshold,
            high_frequency_max_threshold: uniforms.high_freq_max_threshold,
            surface_radius: 0.0,
            surface_threshold: 0.0,
            gauss_offset: 0.0,
            roi_size: width,
            algo_type: 0,
            box_blur_stage: 0,
        }
    }

    /// Mirrors `p15_validate_launch` minus the texture/GL-state half, which the
    /// caller proves by supplying the buffers.
    pub fn validate(&self) -> Result<(), Refusal> {
        let finite = [
            self.pf_value,
            self.surface_radius,
            self.surface_threshold,
            self.gauss_offset,
            self.legal_alpha,
            self.dark_circle_alpha,
            self.diff_factor,
            self.diff_factor2,
            self.color_threshold_factor,
            self.mix_alpha,
            self.diff_min_threshold,
            self.diff_max_threshold,
            self.high_frequency_min_threshold,
            self.high_frequency_max_threshold,
        ]
        .iter()
        .all(|value| value.is_finite());
        if !finite
            || self.diff_min_threshold > self.diff_max_threshold
            || self.high_frequency_min_threshold > self.high_frequency_max_threshold
        {
            return Err(Refusal::ValueOutOfRange("uniforms"));
        }
        if self.width == 0 || self.height == 0 || self.stride < self.width {
            return Err(Refusal::ShapeMismatch("uniforms"));
        }
        // The kernel guards `x < width` and `y < height`, so the launch geometry
        // needs no square ROI; the vendor 1000x1000 job simply always passes
        // `roi_size == width`. A mismatched ROI side is still refused.
        if self.roi_size != self.width {
            return Err(Refusal::ShapeMismatch("roi_size"));
        }
        if !matches!(
            self.pf,
            90023 | 90024 | 21024 | 21025 | 21069 | 21070 | 21071
        ) {
            return Err(Refusal::UnknownEffect(self.pf));
        }
        if !(self.pf_value > self.gate) || self.gate != GATE {
            return Err(Refusal::ValueOutOfRange("pf_value"));
        }
        if self.algo_type != 0 {
            return Err(Refusal::NotImplemented(self.algo_type));
        }
        Ok(())
    }

    fn buffer_f32(&self) -> usize {
        self.stride as usize * self.height as usize * 4
    }
}

/// Which recovered pass to run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pass {
    Surface,
    Gauss,
    BoxBlur,
}

impl Pass {
    fn kernel(self) -> &'static str {
        match self {
            Pass::Surface => SURFACE,
            Pass::Gauss => GAUSS,
            Pass::BoxBlur => BOX_BLUR,
        }
    }
}

/// Evidence that a real dispatch happened.
#[derive(Clone, Debug, PartialEq)]
pub struct Dispatch {
    pub kernel: &'static str,
    pub grid: (u32, u32),
    pub block: (u32, u32),
    pub ptx_bytes: usize,
    /// PTX ISA level actually handed to the JIT; below the image's own level on
    /// hosts whose driver predates the toolkit that produced it.
    pub ptx_isa: (u32, u32),
    pub device: String,
    pub compute_cap: (i32, i32),
    pub driver_version: i32,
}

/// Process-wide counters for the cached LUT path.
#[cfg(windows)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DriverMetrics {
    pub ptx_module_load_calls: usize,
    pub skin_lut_uploads: usize,
    pub teeth_lut_uploads: usize,
    pub host_to_device_bytes: usize,
    pub device_to_host_bytes: usize,
    pub device_allocations: usize,
    pub context_syncs: usize,
}

/// Device-wide memory reported by the CUDA context at the sampling instant.
#[cfg(windows)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceMemoryInfo {
    pub free_bytes: usize,
    pub total_bytes: usize,
}

/// One launch in a device-resident staging chain.
#[cfg(windows)]
pub(crate) struct ChainStage<'a> {
    pub kernel: &'static str,
    pub grid: (u32, u32),
    pub inputs: &'a [usize],
    pub output: usize,
    pub lut: Option<usize>,
    pub lut_arg: bool,
    pub init_from: Option<usize>,
    pub uniforms: &'a [u8],
}

#[cfg(windows)]
pub fn driver_metrics() -> DriverMetrics {
    driver::metrics()
}

#[cfg(windows)]
pub fn driver_memory_info() -> Result<DeviceMemoryInfo, CudaError> {
    driver::memory_info()
}

#[cfg(all(windows, test))]
pub(crate) fn test_driver_counts() -> [usize; 3] {
    [
        driver::live_contexts(),
        driver::device_allocations(),
        driver::context_syncs(),
    ]
}

/// Why the NVIDIA path produced nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaError {
    /// No usable driver, device or kernel image on this host.
    Unavailable(&'static str),
    /// A driver call returned a non-zero `CUresult`.
    Driver(&'static str, i32),
    /// The same typed refusal the CPU lane returns.
    Refused(Refusal),
}

impl From<Refusal> for CudaError {
    fn from(refusal: Refusal) -> Self {
        CudaError::Refused(refusal)
    }
}

/// PTX ISA level declared by a PTX image.
pub fn ptx_isa(ptx: &str) -> Option<(u32, u32)> {
    let rest = ptx.split(".version ").nth(1)?;
    let token = rest.split_whitespace().next()?;
    let (major, minor) = token.split_once('.')?;
    Some((major.parse().ok()?, minor.parse().ok()?))
}

/// Highest PTX ISA the driver reported by `cuDriverGetVersion` can JIT.
///
/// A driver version is `major * 1000 + minor * 10` and the PTX ISA for it is
/// `(major - 4, minor)`, so CUDA 13.3 maps to ISA 9.3. Anything below CUDA 4
/// yields major 0, which `restamp_ptx` refuses rather than over-downgrading.
pub fn driver_ptx_isa(driver_version: i32) -> (u32, u32) {
    let version = driver_version.max(0) as u32;
    ((version / 1000).saturating_sub(4), (version % 1000) / 10)
}

/// Re-stamps the ISA header down to `target`.
///
/// The driver's JIT refuses a PTX image newer than its own CUDA version, and a
/// toolkit newer than the driver is common on Windows. Every instruction the
/// P15 kernels use predates ISA 7.0, so re-stamping is sound for this image;
/// callers still verify pixels against the CPU lane. Returns `None` when the
/// image is already at or below `target`.
pub fn restamp_ptx(ptx: &str, target: (u32, u32)) -> Option<String> {
    // shortcut: only the bundled P15 image is GPU-checked; rebuild changed PTX with a matching toolkit.
    let current = ptx_isa(ptx)?;
    if current <= target || target.0 < 1 {
        return None;
    }
    let header = format!(".version {}.{}", current.0, current.1);
    let replacement = format!(".version {}.{}", target.0, target.1);
    Some(ptx.replacen(&header, &replacement, 1))
}

#[cfg(windows)]
mod driver {
    use super::{
        ChainStage, CudaError, DeviceMemoryInfo, Dispatch, DriverMetrics, Image, PTX, Pass,
        Refusal, Uniforms, cpu, driver_ptx_isa, ptx_isa, restamp_ptx,
    };
    #[cfg(test)]
    use std::cell::Cell;
    use std::ffi::{CStr, CString, c_char, c_void};
    use std::ptr;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, OnceLock};
    // Test-only process census. The domain is intentionally process-wide.
    #[cfg(test)]
    static LIVE_CONTEXTS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    #[cfg(test)]
    thread_local! {
        static DEVICE_ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
        static CONTEXT_SYNCS: Cell<usize> = const { Cell::new(0) };
    }

    #[cfg(test)]
    fn count_context(delta: isize) {
        use std::sync::atomic::Ordering;
        if delta >= 0 {
            LIVE_CONTEXTS.fetch_add(delta as usize, Ordering::SeqCst);
        } else {
            LIVE_CONTEXTS.fetch_sub((-delta) as usize, Ordering::SeqCst);
        }
    }

    #[cfg(test)]
    pub(crate) fn live_contexts() -> usize {
        LIVE_CONTEXTS.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn count_device_allocation() {
        DEVICE_ALLOCATIONS_TOTAL.fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        DEVICE_ALLOCATIONS.with(|count| count.set(count.get() + 1));
    }

    pub(crate) fn device_allocations() -> usize {
        DEVICE_ALLOCATIONS_TOTAL.load(Ordering::Relaxed)
    }

    fn count_context_sync() {
        CONTEXT_SYNCS_TOTAL.fetch_add(1, Ordering::Relaxed);
        #[cfg(test)]
        CONTEXT_SYNCS.with(|count| count.set(count.get() + 1));
    }

    pub(crate) fn context_syncs() -> usize {
        CONTEXT_SYNCS_TOTAL.load(Ordering::Relaxed)
    }

    type CuResult = i32;
    type CuDevice = i32;
    type CuContext = *mut c_void;
    type CuModule = *mut c_void;
    type CuFunction = *mut c_void;
    type CuPtr = u64;

    const ATTR_CC_MAJOR: i32 = 75;
    const ATTR_CC_MINOR: i32 = 76;
    const BLOCK: (u32, u32) = (16, 16);

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    #[derive(Clone, Copy)]
    struct Api {
        init: unsafe extern "system" fn(u32) -> CuResult,
        device_get: unsafe extern "system" fn(*mut CuDevice, i32) -> CuResult,
        device_name: unsafe extern "system" fn(*mut c_char, i32, CuDevice) -> CuResult,
        device_attribute: unsafe extern "system" fn(*mut i32, i32, CuDevice) -> CuResult,
        driver_version: unsafe extern "system" fn(*mut i32) -> CuResult,
        ctx_create: unsafe extern "system" fn(*mut CuContext, u32, CuDevice) -> CuResult,
        ctx_destroy: unsafe extern "system" fn(CuContext) -> CuResult,
        ctx_set_current: unsafe extern "system" fn(CuContext) -> CuResult,
        ctx_sync: unsafe extern "system" fn() -> CuResult,
        module_load: unsafe extern "system" fn(
            *mut CuModule,
            *const c_void,
            u32,
            *mut i32,
            *mut *mut c_void,
        ) -> CuResult,
        module_unload: unsafe extern "system" fn(CuModule) -> CuResult,
        module_function:
            unsafe extern "system" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
        mem_alloc: unsafe extern "system" fn(*mut CuPtr, usize) -> CuResult,
        mem_free: unsafe extern "system" fn(CuPtr) -> CuResult,
        mem_get_info: unsafe extern "system" fn(*mut usize, *mut usize) -> CuResult,
        copy_in: unsafe extern "system" fn(CuPtr, *const c_void, usize) -> CuResult,
        copy_out: unsafe extern "system" fn(*mut c_void, CuPtr, usize) -> CuResult,
        copy_device: unsafe extern "system" fn(CuPtr, CuPtr, usize) -> CuResult,
        launch: unsafe extern "system" fn(
            CuFunction,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            u32,
            *mut c_void,
            *mut *mut c_void,
            *mut *mut c_void,
        ) -> CuResult,
    }

    fn load() -> Result<Api, CudaError> {
        unsafe {
            let handle = LoadLibraryA(c"nvcuda.dll".as_ptr());
            if handle.is_null() {
                return Err(CudaError::Unavailable("nvcuda.dll"));
            }
            macro_rules! sym {
                ($name:literal) => {{
                    let address =
                        GetProcAddress(handle, concat!($name, "\0").as_ptr() as *const c_char);
                    if address.is_null() {
                        return Err(CudaError::Unavailable($name));
                    }
                    std::mem::transmute::<*mut c_void, _>(address)
                }};
            }
            // 64-bit driver builds export the `_v2` forms; older ones only the plain name.
            macro_rules! sym2 {
                ($v2:literal, $plain:literal) => {{
                    let mut address =
                        GetProcAddress(handle, concat!($v2, "\0").as_ptr() as *const c_char);
                    if address.is_null() {
                        address =
                            GetProcAddress(handle, concat!($plain, "\0").as_ptr() as *const c_char);
                    }
                    if address.is_null() {
                        return Err(CudaError::Unavailable($plain));
                    }
                    std::mem::transmute::<*mut c_void, _>(address)
                }};
            }
            Ok(Api {
                init: sym!("cuInit"),
                device_get: sym!("cuDeviceGet"),
                device_name: sym!("cuDeviceGetName"),
                device_attribute: sym!("cuDeviceGetAttribute"),
                driver_version: sym2!("cuDriverGetVersion", "cuDriverGetVersion"),
                ctx_create: sym2!("cuCtxCreate_v2", "cuCtxCreate"),
                ctx_destroy: sym2!("cuCtxDestroy_v2", "cuCtxDestroy"),
                ctx_set_current: sym!("cuCtxSetCurrent"),
                ctx_sync: sym!("cuCtxSynchronize"),
                module_load: sym!("cuModuleLoadDataEx"),
                module_unload: sym!("cuModuleUnload"),
                module_function: sym!("cuModuleGetFunction"),
                mem_alloc: sym2!("cuMemAlloc_v2", "cuMemAlloc"),
                mem_free: sym2!("cuMemFree_v2", "cuMemFree"),
                mem_get_info: sym2!("cuMemGetInfo_v2", "cuMemGetInfo"),
                copy_in: sym2!("cuMemcpyHtoD_v2", "cuMemcpyHtoD"),
                copy_out: sym2!("cuMemcpyDtoH_v2", "cuMemcpyDtoH"),
                copy_device: sym2!("cuMemcpyDtoD_v2", "cuMemcpyDtoD"),
                launch: sym!("cuLaunchKernel"),
            })
        }
    }

    fn check(code: CuResult, call: &'static str) -> Result<(), CudaError> {
        if code == 0 {
            Ok(())
        } else {
            Err(CudaError::Driver(call, code))
        }
    }

    #[derive(Clone, Copy)]
    struct LoadedModule {
        image: &'static str,
        module: usize,
        ptx_isa: (u32, u32),
        image_bytes: usize,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct BufferShape {
        width: u32,
        height: u32,
        stride: u32,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct StagingShape {
        input_count: usize,
        input_bytes: Vec<usize>,
        output_bytes: usize,
        alias_input0: bool,
    }

    struct CachedStaging {
        shape: StagingShape,
        buffers: Vec<(CuPtr, usize)>,
    }

    struct CachedComposite {
        shape: BufferShape,
        buffers: Vec<(CuPtr, usize)>,
    }

    impl CachedComposite {
        fn matches(&self, shape: BufferShape, sizes: &[usize; 6]) -> bool {
            self.shape == shape
                && self
                    .buffers
                    .iter()
                    .map(|(_, size)| *size)
                    .eq(sizes.iter().copied())
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct LutBufferShape {
        grid: (u32, u32),
        source_bytes: usize,
        mask_bytes: usize,
    }

    struct CachedLutBuffers {
        shape: LutBufferShape,
        buffers: Vec<(CuPtr, usize)>,
    }

    impl CachedLutBuffers {
        fn matches(&self, shape: LutBufferShape) -> bool {
            self.shape == shape
                && self.buffers.iter().map(|(_, size)| *size).eq([
                    shape.source_bytes,
                    shape.source_bytes,
                    shape.mask_bytes,
                ]
                .iter()
                .copied())
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct NamedBufferShape {
        pool: &'static str,
        sizes: Vec<usize>,
    }

    struct CachedNamedBuffers {
        shape: NamedBufferShape,
        buffers: Vec<(CuPtr, usize)>,
    }

    impl CachedNamedBuffers {
        fn matches(&self, shape: &NamedBufferShape) -> bool {
            self.shape == *shape
                && self
                    .buffers
                    .iter()
                    .map(|(_, size)| *size)
                    .eq(shape.sizes.iter().copied())
        }
    }

    /// One process-wide context owns every JIT-compiled PTX module.
    ///
    /// The handle stores no resource and therefore cannot outlive the domain.
    /// Every driver call locks the domain and makes this context current on the
    /// calling thread before touching a module or allocation.
    struct Domain {
        api: Api,
        ctx: usize,
        modules: Vec<LoadedModule>,
        pending: bool,
        composite_buffers: Vec<CachedComposite>,
        lut_buffers: Vec<CachedLutBuffers>,
        generic_staging: Vec<CachedStaging>,
        named_buffers: Vec<CachedNamedBuffers>,
        skin_luts: Vec<CachedLut>,
        teeth_luts: Vec<CachedLut>,
        name: String,
        compute_cap: (i32, i32),
        driver_version: i32,
    }

    static DOMAIN: OnceLock<Result<Mutex<Domain>, CudaError>> = OnceLock::new();
    static PTX_MODULE_LOADS: AtomicUsize = AtomicUsize::new(0);
    static SKIN_LUT_UPLOADS: AtomicUsize = AtomicUsize::new(0);
    static TEETH_LUT_UPLOADS: AtomicUsize = AtomicUsize::new(0);
    static HOST_TO_DEVICE_BYTES: AtomicUsize = AtomicUsize::new(0);
    static DEVICE_TO_HOST_BYTES: AtomicUsize = AtomicUsize::new(0);
    static DEVICE_ALLOCATIONS_TOTAL: AtomicUsize = AtomicUsize::new(0);
    static CONTEXT_SYNCS_TOTAL: AtomicUsize = AtomicUsize::new(0);
    const LUT_CACHE_SLOTS: usize = 2;
    // Two exact shapes cover preview/export interleaving; eviction waits before freeing.
    const COMPOSITE_BUFFER_SLOTS: usize = 2;
    const LUT_BUFFER_SLOTS: usize = 2;
    const TEETH_LUT_CACHE_SLOTS: usize = 2;
    // Two slots cover the common single-pass then four-input blend pair.
    // shortcut: only two exact shapes stay resident, add a byte-budgeted LRU if adapters interleave more.
    const GENERIC_STAGING_SLOTS: usize = 2;
    const NAMED_BUFFER_SLOTS: usize = 8;

    #[derive(Debug)]
    struct CachedLut {
        bytes: Vec<u8>,
        ptr: CuPtr,
    }

    fn domain() -> Result<&'static Mutex<Domain>, CudaError> {
        match DOMAIN.get_or_init(|| Domain::open().map(Mutex::new)) {
            Ok(domain) => Ok(domain),
            Err(error) => Err(error.clone()),
        }
    }

    pub struct Nvidia {
        image: &'static str,
        api: Api,
        name: String,
        compute_cap: (i32, i32),
        driver_version: i32,
        ptx_isa: (u32, u32),
        image_bytes: usize,
    }

    impl Domain {
        fn open() -> Result<Self, CudaError> {
            let api = load()?;
            let mut device: CuDevice = 0;
            let mut ctx: CuContext = ptr::null_mut();
            let mut version = 0;
            unsafe {
                check((api.init)(0), "cuInit")?;
                check((api.driver_version)(&mut version), "cuDriverGetVersion")?;
                check((api.device_get)(&mut device, 0), "cuDeviceGet")?;
                check((api.ctx_create)(&mut ctx, 0, device), "cuCtxCreate")?;
            }
            #[cfg(test)]
            count_context(1);
            let mut domain = Self {
                api,
                ctx: ctx as usize,
                modules: Vec::new(),
                pending: false,
                composite_buffers: Vec::new(),
                lut_buffers: Vec::new(),
                generic_staging: Vec::new(),
                named_buffers: Vec::new(),
                skin_luts: Vec::new(),
                teeth_luts: Vec::new(),
                name: String::new(),
                compute_cap: (0, 0),
                driver_version: version,
            };
            let mut name_buf = [0 as c_char; 256];
            let mut major = 0;
            let mut minor = 0;
            unsafe {
                check(
                    (domain.api.device_name)(name_buf.as_mut_ptr(), name_buf.len() as i32, device),
                    "cuDeviceGetName",
                )?;
                check(
                    (domain.api.device_attribute)(&mut major, ATTR_CC_MAJOR, device),
                    "cuDeviceGetAttribute",
                )?;
                check(
                    (domain.api.device_attribute)(&mut minor, ATTR_CC_MINOR, device),
                    "cuDeviceGetAttribute",
                )?;
            }
            domain.name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
                .to_string_lossy()
                .into_owned();
            domain.compute_cap = (major, minor);
            Ok(domain)
        }

        fn bind(&mut self) -> Result<(), CudaError> {
            unsafe {
                check(
                    (self.api.ctx_set_current)(self.ctx as CuContext),
                    "cuCtxSetCurrent",
                )
            }
        }

        fn load_module(api: &Api, text: &str) -> Result<CuModule, CudaError> {
            let mut image = text.as_bytes().to_vec();
            image.push(0);
            let mut module: CuModule = ptr::null_mut();
            PTX_MODULE_LOADS.fetch_add(1, Ordering::Relaxed);
            unsafe {
                check(
                    (api.module_load)(
                        &mut module,
                        image.as_ptr() as *const c_void,
                        0,
                        ptr::null_mut(),
                        ptr::null_mut(),
                    ),
                    "cuModuleLoadDataEx",
                )?;
            }
            Ok(module)
        }

        fn module(&mut self, image: &'static str) -> Result<LoadedModule, CudaError> {
            // Module cache hits may run on a different host thread, so binding
            // must precede the lookup rather than only the first JIT load.
            self.bind()?;
            if let Some(module) = self.modules.iter().find(|module| module.image == image) {
                return Ok(*module);
            }
            // A toolkit newer than the driver used to make the first load a
            // guaranteed error-222 round trip before the same text was loaded
            // again at a supported ISA. Re-stamping the header up front keeps
            // the cold path to one module load per image.
            let target = driver_ptx_isa(self.driver_version);
            let downgraded = restamp_ptx(image, target);
            let preferred = downgraded.as_deref().unwrap_or(image);
            let module = Self::load_module(&self.api, preferred)?;
            let (isa, loaded) = (
                ptx_isa(preferred).ok_or(CudaError::Unavailable("ptx image"))?,
                preferred.len(),
            );
            let loaded_module = LoadedModule {
                image,
                module: module as usize,
                ptx_isa: isa,
                image_bytes: loaded,
            };
            self.modules.push(loaded_module);
            Ok(loaded_module)
        }

        fn synchronize_pending(&mut self) -> Result<(), CudaError> {
            if self.pending {
                count_context_sync();
                unsafe { check((self.api.ctx_sync)(), "cuCtxSynchronize")? };
                self.pending = false;
            }
            Ok(())
        }

        fn composite_buffers(
            &mut self,
            uniforms: &Uniforms,
            sizes: [usize; 6],
        ) -> Result<[CuPtr; 6], CudaError> {
            let shape = BufferShape {
                width: uniforms.width,
                height: uniforms.height,
                stride: uniforms.stride,
            };
            if let Some(index) = self
                .composite_buffers
                .iter()
                .position(|entry| entry.matches(shape, &sizes))
            {
                let entry = self.composite_buffers.remove(index);
                let ptrs = std::array::from_fn(|slot| entry.buffers[slot].0);
                self.composite_buffers.push(entry);
                return Ok(ptrs);
            }
            if self.composite_buffers.len() == COMPOSITE_BUFFER_SLOTS {
                self.synchronize_pending()?;
                let evicted = self.composite_buffers.remove(0);
                self.free_group(evicted.buffers)?;
            }
            let mut allocations = Buffers {
                api: &self.api,
                ptrs: Vec::new(),
                pending_launch: false,
            };
            for size in sizes {
                allocations.alloc(size)?;
            }
            let buffers: Vec<_> = allocations.ptrs.drain(..).zip(sizes).collect();
            let ptrs = std::array::from_fn(|slot| buffers[slot].0);
            self.composite_buffers
                .push(CachedComposite { shape, buffers });
            Ok(ptrs)
        }

        fn free_group(&mut self, buffers: Vec<(CuPtr, usize)>) -> Result<(), CudaError> {
            let mut first_error = Ok(());
            for (ptr, _) in buffers {
                unsafe {
                    if let Err(error) = check((self.api.mem_free)(ptr), "cuMemFree") {
                        if first_error.is_ok() {
                            first_error = Err(error);
                        }
                    }
                }
            }
            first_error
        }

        fn lut_buffers(
            &mut self,
            grid: (u32, u32),
            source_bytes: usize,
            mask_bytes: usize,
        ) -> Result<[CuPtr; 3], CudaError> {
            let shape = LutBufferShape {
                grid,
                source_bytes,
                mask_bytes,
            };
            if let Some(index) = self
                .lut_buffers
                .iter()
                .position(|entry| entry.matches(shape))
            {
                let entry = self.lut_buffers.remove(index);
                let ptrs = std::array::from_fn(|slot| entry.buffers[slot].0);
                self.lut_buffers.push(entry);
                return Ok(ptrs);
            }
            if self.lut_buffers.len() == LUT_BUFFER_SLOTS {
                self.synchronize_pending()?;
                let evicted = self.lut_buffers.remove(0);
                self.free_group(evicted.buffers)?;
            }
            let sizes = [source_bytes, source_bytes, mask_bytes];
            let mut allocations = Buffers {
                api: &self.api,
                ptrs: Vec::new(),
                pending_launch: false,
            };
            for size in sizes {
                allocations.alloc(size)?;
            }
            let buffers: Vec<_> = allocations.ptrs.drain(..).zip(sizes).collect();
            let ptrs = std::array::from_fn(|slot| buffers[slot].0);
            self.lut_buffers.push(CachedLutBuffers { shape, buffers });
            Ok(ptrs)
        }

        fn generic_staging(&mut self, shape: &StagingShape) -> Result<Vec<CuPtr>, CudaError> {
            if let Some(index) = self
                .generic_staging
                .iter()
                .position(|entry| &entry.shape == shape)
            {
                let entry = self.generic_staging.remove(index);
                let ptrs = entry.buffers.iter().map(|(ptr, _)| *ptr).collect();
                self.generic_staging.push(entry);
                return Ok(ptrs);
            }

            if self.generic_staging.len() == GENERIC_STAGING_SLOTS {
                self.synchronize_pending()?;
                let evicted = self.generic_staging.remove(0);
                for (ptr, _) in evicted.buffers {
                    unsafe {
                        let _ = (self.api.mem_free)(ptr);
                    }
                }
            }
            let mut allocations = Buffers {
                api: &self.api,
                ptrs: Vec::new(),
                pending_launch: false,
            };
            let mut sizes = shape.input_bytes.clone();
            if !shape.alias_input0 {
                sizes.push(shape.output_bytes);
            }
            for size in &sizes {
                allocations.alloc(*size)?;
            }
            let ptrs = allocations.ptrs.iter().copied().collect();
            let buffers = allocations.ptrs.drain(..).zip(sizes).collect();
            self.generic_staging.push(CachedStaging {
                shape: shape.clone(),
                buffers,
            });
            Ok(ptrs)
        }

        fn clear_generic_staging(&mut self) {
            for entry in self.generic_staging.drain(..) {
                for (ptr, _) in entry.buffers {
                    unsafe {
                        let _ = (self.api.mem_free)(ptr);
                    }
                }
            }
        }

        fn named_buffers(&mut self, shape: NamedBufferShape) -> Result<Vec<CuPtr>, CudaError> {
            if let Some(index) = self
                .named_buffers
                .iter()
                .position(|entry| entry.matches(&shape))
            {
                let entry = self.named_buffers.remove(index);
                let ptrs = entry.buffers.iter().map(|(ptr, _)| *ptr).collect();
                self.named_buffers.push(entry);
                return Ok(ptrs);
            }
            if self.named_buffers.len() == NAMED_BUFFER_SLOTS {
                self.synchronize_pending()?;
                let evicted = self.named_buffers.remove(0);
                self.free_group(evicted.buffers)?;
            }
            let mut allocations = Buffers {
                api: &self.api,
                ptrs: Vec::new(),
                pending_launch: false,
            };
            for size in &shape.sizes {
                allocations.alloc((*size).max(1))?;
            }
            let buffers: Vec<(CuPtr, usize)> = allocations
                .ptrs
                .drain(..)
                .zip(shape.sizes.iter().copied())
                .collect();
            let ptrs: Vec<CuPtr> = buffers.iter().map(|(ptr, _)| *ptr).collect();
            self.named_buffers
                .push(CachedNamedBuffers { shape, buffers });
            Ok(ptrs)
        }

        fn cached_skin_lut(&mut self, data: &[u8]) -> Result<CuPtr, CudaError> {
            if let Some(hit) = self.skin_luts.iter().find(|entry| entry.bytes == data) {
                return Ok(hit.ptr);
            }
            if self.skin_luts.len() == LUT_CACHE_SLOTS {
                self.synchronize_pending()?;
                let evicted = self.skin_luts.remove(0);
                unsafe {
                    let _ = (self.api.mem_free)(evicted.ptr);
                }
            }
            let mut ptr: CuPtr = 0;
            unsafe {
                check(
                    (self.api.mem_alloc)(&mut ptr, data.len().max(1)),
                    "cuMemAlloc",
                )?;
            }
            count_device_allocation();
            if let Err(error) = unsafe {
                check(
                    (self.api.copy_in)(ptr, data.as_ptr() as *const c_void, data.len()),
                    "cuMemcpyHtoD",
                )
            } {
                unsafe {
                    let _ = (self.api.mem_free)(ptr);
                }
                return Err(error);
            }
            HOST_TO_DEVICE_BYTES.fetch_add(data.len(), Ordering::Relaxed);
            SKIN_LUT_UPLOADS.fetch_add(1, Ordering::Relaxed);
            self.skin_luts.push(CachedLut {
                bytes: data.to_vec(),
                ptr,
            });
            Ok(ptr)
        }

        fn cached_teeth_lut(&mut self, data: &[u8]) -> Result<CuPtr, CudaError> {
            if let Some(hit) = self.teeth_luts.iter().find(|entry| entry.bytes == data) {
                return Ok(hit.ptr);
            }
            if self.teeth_luts.len() == TEETH_LUT_CACHE_SLOTS {
                let evicted = self.teeth_luts.remove(0);
                unsafe {
                    let _ = (self.api.mem_free)(evicted.ptr);
                }
            }
            let mut ptr: CuPtr = 0;
            unsafe {
                check(
                    (self.api.mem_alloc)(&mut ptr, data.len().max(1)),
                    "cuMemAlloc",
                )?;
                check(
                    (self.api.copy_in)(ptr, data.as_ptr() as *const c_void, data.len()),
                    "cuMemcpyHtoD",
                )?;
            }
            count_device_allocation();
            TEETH_LUT_UPLOADS.fetch_add(1, Ordering::Relaxed);
            self.teeth_luts.push(CachedLut {
                bytes: data.to_vec(),
                ptr,
            });
            Ok(ptr)
        }
    }

    impl Drop for Domain {
        fn drop(&mut self) {
            let _ = self.synchronize_pending();
            for group in std::mem::take(&mut self.composite_buffers) {
                let _ = self.free_group(group.buffers);
            }
            for group in std::mem::take(&mut self.lut_buffers) {
                let _ = self.free_group(group.buffers);
            }
            self.clear_generic_staging();
            for group in std::mem::take(&mut self.named_buffers) {
                let _ = self.free_group(group.buffers);
            }
            for lut in self.skin_luts.drain(..) {
                unsafe {
                    let _ = (self.api.mem_free)(lut.ptr);
                }
            }
            for lut in self.teeth_luts.drain(..) {
                unsafe {
                    let _ = (self.api.mem_free)(lut.ptr);
                }
            }
            unsafe {
                let _ = (self.api.ctx_set_current)(self.ctx as CuContext);
                for module in self.modules.drain(..) {
                    let _ = (self.api.module_unload)(module.module as CuModule);
                }
                if self.ctx != 0 {
                    let _ = (self.api.ctx_destroy)(self.ctx as CuContext);
                }
                #[cfg(test)]
                count_context(-1);
            }
        }
    }

    struct Buffers<'a> {
        api: &'a Api,
        ptrs: Vec<CuPtr>,
        pending_launch: bool,
    }

    impl Buffers<'_> {
        fn alloc(&mut self, bytes: usize) -> Result<CuPtr, CudaError> {
            let mut ptr: CuPtr = 0;
            unsafe { check((self.api.mem_alloc)(&mut ptr, bytes), "cuMemAlloc")? };
            count_device_allocation();
            self.ptrs.push(ptr);
            Ok(ptr)
        }
    }

    impl Drop for Buffers<'_> {
        fn drop(&mut self) {
            if self.pending_launch {
                unsafe {
                    let _ = (self.api.ctx_sync)();
                }
            }
            for ptr in self.ptrs.drain(..) {
                unsafe {
                    let _ = (self.api.mem_free)(ptr);
                }
            }
        }
    }

    /// One owned device allocation, freed on drop. Crate-private so a backend
    /// that needs several buffers alive across two launches can hold them.
    #[allow(dead_code)] // consumed by `cuda_mesh`
    pub struct DeviceBuffer<'a> {
        ptr: CuPtr,
        owner: std::marker::PhantomData<&'a Nvidia>,
    }

    impl DeviceBuffer<'_> {
        #[allow(dead_code)] // consumed by `cuda_mesh`
        pub fn handle(&self) -> u64 {
            self.ptr
        }
    }

    impl Drop for DeviceBuffer<'_> {
        fn drop(&mut self) {
            if let Ok(domain) = domain() {
                let mut domain = domain
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if domain.bind().is_ok() {
                    let _ = domain.synchronize_pending();
                    unsafe {
                        let _ = (domain.api.mem_free)(self.ptr);
                    }
                }
            }
        }
    }

    /// A fixed group of device buffers, allocated once and reused across
    /// launches. Crate-private: a session that repeats the same job shape should
    /// not pay `cuMemAlloc` per launch.
    #[allow(dead_code)] // consumed by the O1h session benchmark
    pub(crate) struct DeviceSet<'a> {
        buffers: Vec<(CuPtr, usize)>,
        owner: std::marker::PhantomData<&'a Nvidia>,
    }

    #[allow(dead_code)] // consumed by the O1h session benchmark
    impl DeviceSet<'_> {
        pub(crate) fn handle(&self, index: usize) -> u64 {
            self.buffers[index].0
        }

        pub(crate) fn len(&self) -> usize {
            self.buffers.len()
        }
    }

    impl Drop for DeviceSet<'_> {
        fn drop(&mut self) {
            if let Ok(domain) = domain() {
                let mut domain = domain
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if domain.bind().is_ok() {
                    let _ = domain.synchronize_pending();
                    for (ptr, _) in self.buffers.drain(..) {
                        unsafe {
                            let _ = (domain.api.mem_free)(ptr);
                        }
                    }
                }
            }
        }
    }

    /// A bounded operation over one exact-shape group from the shared pool.
    /// The domain remains locked until the caller finishes its CPU readback.
    pub(crate) struct BufferSession<'a> {
        gpu: &'a Nvidia,
        domain: &'a mut Domain,
        module: CuModule,
        sizes: Vec<usize>,
        ptrs: Vec<CuPtr>,
    }

    impl BufferSession<'_> {
        pub(crate) fn handle(&self, index: usize) -> u64 {
            self.ptrs[index]
        }

        pub(crate) fn upload(&mut self, index: usize, data: &[u8]) -> Result<(), CudaError> {
            let ptr = self.checked_ptr(index, data.len())?;
            self.gpu.upload(ptr, data)
        }

        pub(crate) fn launch(
            &mut self,
            kernel: &'static str,
            geometry: (u32, u32),
            block: (u32, u32),
            params: &mut [*mut c_void],
        ) -> Result<Dispatch, CudaError> {
            let api = self.domain.api;
            let dispatch =
                self.gpu
                    .launch_on(&api, self.module, kernel, geometry, block, params)?;
            self.domain.pending = true;
            Ok(dispatch)
        }

        pub(crate) fn download(&mut self, index: usize, data: &mut [u8]) -> Result<(), CudaError> {
            let ptr = self.checked_ptr(index, data.len())?;
            self.gpu.download(ptr, data)
        }

        pub(crate) fn mark_consumed(&mut self) {
            self.domain.pending = false;
        }

        fn checked_ptr(&self, index: usize, bytes: usize) -> Result<CuPtr, CudaError> {
            match (self.ptrs.get(index), self.sizes.get(index)) {
                (Some(ptr), Some(size)) if *size == bytes => Ok(*ptr),
                _ => Err(CudaError::Refused(Refusal::ShapeMismatch(
                    "reusable buffer",
                ))),
            }
        }
    }

    fn grid(width: u32, height: u32) -> (u32, u32) {
        (
            width.div_ceil(BLOCK.0).max(1),
            height.div_ceil(BLOCK.1).max(1),
        )
    }

    /// Active pixels of each row, skipping `stride - width` padding columns.
    fn rows<'a>(
        values: &'a [f32],
        uniforms: &Uniforms,
        per_pixel: usize,
    ) -> impl Iterator<Item = &'a [f32]> + 'a {
        let stride = uniforms.stride as usize * per_pixel;
        let active = uniforms.width as usize * per_pixel;
        (0..uniforms.height as usize).map(move |row| &values[row * stride..row * stride + active])
    }

    /// Scalar values must be finite; the kernel writes NaN straight through.
    fn check_pixels(values: &[f32], uniforms: &Uniforms) -> Result<(), Refusal> {
        if rows(values, uniforms, 4).any(|row| row.iter().any(|value| !value.is_finite())) {
            return Err(Refusal::ValueOutOfRange("pixel"));
        }
        Ok(())
    }

    /// Masks are weights: `.r`/`.b` and the skin plane must be finite and in
    /// `[0,1]`, exactly as `composite_pixel` requires.
    fn check_masks(mask: &[f32], skin_mask: &[f32], uniforms: &Uniforms) -> Result<(), Refusal> {
        let weight = |value: f32| value.is_finite() && (0.0..=1.0).contains(&value);
        let mask_bad = rows(mask, uniforms, 4).any(|row| {
            row.chunks_exact(4)
                .any(|pixel| !weight(pixel[0]) || !weight(pixel[2]))
        });
        let skin_bad =
            rows(skin_mask, uniforms, 1).any(|row| row.iter().any(|value| !weight(*value)));
        if mask_bad || skin_bad {
            return Err(Refusal::ValueOutOfRange("mask"));
        }
        Ok(())
    }

    fn rgba_f32(image: &Image) -> Vec<f32> {
        image
            .pixels
            .iter()
            .map(|byte| *byte as f32 / 255.0)
            .collect()
    }

    fn quantize(values: &[f32], source: &Image) -> Image {
        let mut out = source.clone();
        for (index, value) in values.iter().enumerate() {
            out.pixels[index] = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        out
    }

    impl Nvidia {
        /// Returns a handle to the process domain's cached P15 module.
        pub fn open() -> Result<Self, CudaError> {
            Self::open_image(PTX)
        }

        /// Returns a handle after ensuring this PTX image is in the cached domain.
        pub(crate) fn open_image(image: &'static str) -> Result<Self, CudaError> {
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let module = domain.module(image)?;
            Ok(Self {
                image,
                api: domain.api,
                name: domain.name.clone(),
                compute_cap: domain.compute_cap,
                driver_version: domain.driver_version,
                ptx_isa: module.ptx_isa,
                image_bytes: module.image_bytes,
            })
        }

        /// Kept for the LUT session API. The operation lock is already
        /// released between calls, so this only refreshes the current context.
        pub(crate) fn resume(&mut self) -> Result<(), CudaError> {
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            domain.bind()
        }

        /// Kept for the LUT session API. All work is synchronous at this
        /// boundary, and no lock is held after the operation returns.
        pub(crate) fn suspend(&mut self) -> Result<(), CudaError> {
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            domain.bind()?;
            domain.synchronize_pending()
        }

        /// Crate-private: load another recovered image through the same guarded
        /// opener and the same 222-only retry.
        #[allow(dead_code)] // consumed by `cuda_lut`, which lib.rs mounts separately
        pub fn open_ptx(image: &'static str) -> Result<Self, CudaError> {
            Self::open_image(image)
        }

        #[cfg(test)]
        pub(crate) fn clear_composite_buffers(&self) -> Result<(), CudaError> {
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            domain.bind()?;
            domain.synchronize_pending()?;
            for group in std::mem::take(&mut domain.composite_buffers) {
                domain.free_group(group.buffers)?;
            }
            Ok(())
        }

        #[cfg(test)]
        pub(crate) fn clear_generic_staging(&self) -> Result<(), CudaError> {
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            domain.bind()?;
            domain.synchronize_pending()?;
            domain.clear_generic_staging();
            Ok(())
        }

        fn with_domain<T>(
            &self,
            call: impl FnOnce(&mut Domain, CuModule) -> Result<T, CudaError>,
        ) -> Result<T, CudaError> {
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let module = domain.module(self.image)?.module as CuModule;
            call(&mut domain, module)
        }

        /// Crate-private: run one loaded kernel over byte buffers, reusing the
        /// allocate/upload/launch/download path of the P15 stages.
        ///
        /// Parameters are `(inputs..., output, uniforms)`: the output has the
        /// length of `inputs[0]`, and `uniform_bytes` is the kernel's by-value
        /// argument block, whose size the PTX signature fixes.
        #[allow(dead_code)] // consumed by `cuda_lut`, which lib.rs mounts separately
        pub(crate) fn run_u8(
            &self,
            kernel: &'static str,
            grid: (u32, u32),
            inputs: &[&[u8]],
            uniform_bytes: &[u8],
        ) -> Result<(Vec<u8>, Dispatch), CudaError> {
            self.run_u8_with_init(kernel, grid, inputs, uniform_bytes, None)
        }

        /// Runs a chain of kernels over one cached staging group. Only the
        /// final output returns to the host; intermediate slots stay on device.
        pub(crate) fn run_u8_chain(
            &self,
            stages: &[ChainStage<'_>],
            slot_bytes: &[usize],
            uploads: &[(usize, &[u8])],
            output_slot: usize,
        ) -> Result<(Vec<u8>, Vec<Dispatch>), CudaError> {
            if slot_bytes.len() < 2 || output_slot + 1 != slot_bytes.len() {
                return Err(CudaError::Refused(Refusal::ShapeMismatch("chain slots")));
            }
            for &(slot, data) in uploads {
                if slot >= slot_bytes.len() - 1 || data.len() != slot_bytes[slot] {
                    return Err(CudaError::Refused(Refusal::ShapeMismatch("chain upload")));
                }
            }
            for stage in stages {
                if stage.uniforms.is_empty() || stage.uniforms.len() % 4 != 0 {
                    return Err(CudaError::Refused(Refusal::ShapeMismatch("uniforms")));
                }
                if stage.output >= slot_bytes.len()
                    || stage.inputs.iter().any(|&slot| slot >= slot_bytes.len())
                    || stage.lut.is_some_and(|slot| slot >= slot_bytes.len())
                    || stage.init_from.is_some_and(|slot| slot >= slot_bytes.len())
                {
                    return Err(CudaError::Refused(Refusal::ShapeMismatch("chain stage")));
                }
            }
            let output_bytes = slot_bytes[output_slot];
            let shape = StagingShape {
                input_count: slot_bytes.len() - 1,
                input_bytes: slot_bytes[..slot_bytes.len() - 1].to_vec(),
                output_bytes,
                alias_input0: false,
            };

            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let module = domain.module(self.image)?.module as CuModule;
            let api = domain.api;
            let buffers = domain.generic_staging(&shape)?;
            for &(slot, data) in uploads {
                self.upload(buffers[slot], data)?;
            }

            let mut dispatches = Vec::with_capacity(stages.len());
            for stage in stages {
                let mut args: Vec<CuPtr> = stage.inputs.iter().map(|&slot| buffers[slot]).collect();
                let mut params: Vec<*mut c_void> = args
                    .iter_mut()
                    .map(|arg| arg as *mut CuPtr as *mut c_void)
                    .collect();
                if stage.lut_arg {
                    let mut lut = stage.lut.map_or(0, |slot| buffers[slot]);
                    params.push(&mut lut as *mut CuPtr as *mut c_void);
                }
                if let Some(slot) = stage.init_from {
                    self.copy_from_device(
                        buffers[stage.output],
                        buffers[slot],
                        slot_bytes[stage.output],
                    )?;
                }
                let mut output = buffers[stage.output];
                params.push(&mut output as *mut CuPtr as *mut c_void);
                let mut block = stage.uniforms.to_vec();
                params.push(block.as_mut_ptr() as *mut c_void);
                match self.launch_on(&api, module, stage.kernel, stage.grid, BLOCK, &mut params) {
                    Ok(dispatch) => {
                        domain.pending = true;
                        dispatches.push(dispatch);
                    }
                    Err(error) => {
                        let _ = domain.synchronize_pending();
                        return Err(error);
                    }
                }
            }

            let mut output = vec![0u8; output_bytes];
            let downloaded = self.download(buffers[output_slot], &mut output);
            if downloaded.is_ok() {
                domain.pending = false;
            }
            downloaded?;
            Ok((output, dispatches))
        }

        /// Run RGBA8 + mask + one LUT while retaining the LUT on device.
        pub(crate) fn run_u8_cached_lut(
            &self,
            kernel: &'static str,
            grid: (u32, u32),
            source: &[u8],
            mask: &[u8],
            lut: &[u8],
            uniform_bytes: &[u8],
        ) -> Result<(Vec<u8>, Dispatch), CudaError> {
            if source.is_empty() || uniform_bytes.is_empty() || uniform_bytes.len() % 4 != 0 {
                return Err(CudaError::Refused(Refusal::ShapeMismatch("buffers")));
            }
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let module = domain.module(self.image)?.module as CuModule;
            let api = domain.api;
            let [source_ptr, output_ptr, mask_ptr] =
                domain.lut_buffers(grid, source.len(), mask.len())?;
            self.upload(source_ptr, source)?;
            self.upload(mask_ptr, mask)?;
            let lut_ptr = domain.cached_skin_lut(lut)?;

            let mut source_arg = source_ptr;
            let mut mask_arg = mask_ptr;
            let mut lut_arg = lut_ptr;
            let mut output_arg = output_ptr;
            let mut block = uniform_bytes.to_vec();
            let mut params: [*mut c_void; 5] = [
                &mut source_arg as *mut CuPtr as *mut c_void,
                &mut mask_arg as *mut CuPtr as *mut c_void,
                &mut lut_arg as *mut CuPtr as *mut c_void,
                &mut output_arg as *mut CuPtr as *mut c_void,
                block.as_mut_ptr() as *mut c_void,
            ];
            let dispatch = self.launch_on(&api, module, kernel, grid, BLOCK, &mut params)?;
            domain.pending = true;
            let mut output = vec![0u8; source.len()];
            let downloaded = self.download(output_ptr, &mut output);
            if downloaded.is_ok() {
                domain.pending = false;
            }
            downloaded?;
            Ok((output, dispatch))
        }

        /// Run a byte kernel while retaining selected inputs in the process-wide
        /// LUT cache. Non-cached inputs still use the exact-shape staging path.
        pub(crate) fn run_u8_cached_luts(
            &self,
            kernel: &'static str,
            grid: (u32, u32),
            inputs: &[&[u8]],
            cached_inputs: &[usize],
            uniform_bytes: &[u8],
        ) -> Result<(Vec<u8>, Dispatch), CudaError> {
            self.run_u8_mode(
                kernel,
                grid,
                inputs,
                uniform_bytes,
                None,
                false,
                cached_inputs,
            )
        }

        /// Run two LUT stages without returning the intermediate image to the host.
        pub(crate) fn run_u8_cached_lut_chain(
            &self,
            kernel: &'static str,
            grid: (u32, u32),
            source: &[u8],
            mask: &[u8],
            first_lut: &[u8],
            second_lut: &[u8],
            first_uniforms: &[u8],
            second_uniforms: &[u8],
        ) -> Result<(Vec<u8>, Dispatch), CudaError> {
            if source.is_empty()
                || first_uniforms.is_empty()
                || first_uniforms.len() % 4 != 0
                || second_uniforms.is_empty()
                || second_uniforms.len() % 4 != 0
            {
                return Err(CudaError::Refused(Refusal::ShapeMismatch("buffers")));
            }
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let module = domain.module(self.image)?.module as CuModule;
            let api = domain.api;
            // The first kernel writes the scratch image; the second reads it and
            // writes back into the original allocation for the sole download.
            let [image_ptr, scratch_ptr, mask_ptr] =
                domain.lut_buffers(grid, source.len(), mask.len())?;
            self.upload(image_ptr, source)?;
            self.upload(mask_ptr, mask)?;
            let first_lut_ptr = domain.cached_skin_lut(first_lut)?;
            let second_lut_ptr = domain.cached_skin_lut(second_lut)?;

            let mut first_source = image_ptr;
            let mut first_mask = mask_ptr;
            let mut first_lut_arg = first_lut_ptr;
            let mut first_output = scratch_ptr;
            let mut first_block = first_uniforms.to_vec();
            let mut first_params: [*mut c_void; 5] = [
                &mut first_source as *mut CuPtr as *mut c_void,
                &mut first_mask as *mut CuPtr as *mut c_void,
                &mut first_lut_arg as *mut CuPtr as *mut c_void,
                &mut first_output as *mut CuPtr as *mut c_void,
                first_block.as_mut_ptr() as *mut c_void,
            ];
            self.launch_on(&api, module, kernel, grid, BLOCK, &mut first_params)?;

            let mut second_source = scratch_ptr;
            let mut second_mask = mask_ptr;
            let mut second_lut_arg = second_lut_ptr;
            let mut second_output = image_ptr;
            let mut second_block = second_uniforms.to_vec();
            let mut second_params: [*mut c_void; 5] = [
                &mut second_source as *mut CuPtr as *mut c_void,
                &mut second_mask as *mut CuPtr as *mut c_void,
                &mut second_lut_arg as *mut CuPtr as *mut c_void,
                &mut second_output as *mut CuPtr as *mut c_void,
                second_block.as_mut_ptr() as *mut c_void,
            ];
            let dispatch = self.launch_on(&api, module, kernel, grid, BLOCK, &mut second_params)?;

            domain.pending = true;
            let mut output = vec![0u8; source.len()];
            let downloaded = self.download(image_ptr, &mut output);
            if downloaded.is_ok() {
                domain.pending = false;
            }
            downloaded?;
            Ok((output, dispatch))
        }

        /// Reuse `inputs[0]` as the output allocation. The kernel must read the
        /// source pixel before writing that same pixel in every code path.
        pub(crate) fn run_u8_inplace(
            &self,
            kernel: &'static str,
            grid: (u32, u32),
            inputs: &[&[u8]],
            uniform_bytes: &[u8],
        ) -> Result<(Vec<u8>, Dispatch), CudaError> {
            self.run_u8_mode(kernel, grid, inputs, uniform_bytes, None, true, &[])
        }

        /// Same as [`Self::run_u8`], but optionally seeds the output buffer from
        /// one input before launch. ROI kernels that only write inside the ROI
        /// need this so pixels outside the ROI keep the source bytes.
        #[allow(dead_code)]
        pub(crate) fn run_u8_with_init(
            &self,
            kernel: &'static str,
            grid: (u32, u32),
            inputs: &[&[u8]],
            uniform_bytes: &[u8],
            init_output_from: Option<usize>,
        ) -> Result<(Vec<u8>, Dispatch), CudaError> {
            self.run_u8_mode(
                kernel,
                grid,
                inputs,
                uniform_bytes,
                init_output_from,
                false,
                &[],
            )
        }

        /// Run a kernel whose optional LUT argument follows the regular inputs.
        pub(crate) fn run_u8_with_init_optional_lut(
            &self,
            kernel: &'static str,
            grid: (u32, u32),
            inputs: &[&[u8]],
            lut: Option<&[u8]>,
            uniform_bytes: &[u8],
            init_output_from: usize,
        ) -> Result<(Vec<u8>, Dispatch), CudaError> {
            let output_bytes = inputs.first().map_or(0, |first| first.len());
            if output_bytes == 0 || uniform_bytes.is_empty() || uniform_bytes.len() % 4 != 0 {
                return Err(CudaError::Refused(Refusal::ShapeMismatch("buffers")));
            }
            let seed =
                inputs
                    .get(init_output_from)
                    .ok_or(CudaError::Refused(Refusal::ShapeMismatch(
                        "init_output_from",
                    )))?;
            if seed.len() != output_bytes {
                return Err(CudaError::Refused(Refusal::ShapeMismatch(
                    "init_output_from",
                )));
            }
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let module = domain.module(self.image)?.module as CuModule;
            let api = domain.api;
            let mut buffers = Buffers {
                api: &api,
                ptrs: Vec::new(),
                pending_launch: false,
            };
            let mut input_ptrs = Vec::with_capacity(inputs.len());
            for input in inputs {
                let ptr = buffers.alloc(input.len())?;
                self.upload(ptr, input)?;
                input_ptrs.push(ptr);
            }
            let output_ptr = buffers.alloc(output_bytes)?;
            self.copy_from_device(output_ptr, input_ptrs[init_output_from], output_bytes)?;
            let mut lut_ptr = if let Some(bytes) = lut {
                let ptr = buffers.alloc(bytes.len())?;
                self.upload(ptr, bytes)?;
                ptr
            } else {
                0
            };

            let mut params: Vec<*mut c_void> = Vec::with_capacity(inputs.len() + 3);
            for ptr in &mut input_ptrs {
                params.push(ptr as *mut CuPtr as *mut c_void);
            }
            params.push(&mut lut_ptr as *mut CuPtr as *mut c_void);
            let mut output_arg = output_ptr;
            params.push(&mut output_arg as *mut CuPtr as *mut c_void);
            let mut block = uniform_bytes.to_vec();
            params.push(block.as_mut_ptr() as *mut c_void);

            let dispatch = self.launch_on(&api, module, kernel, grid, BLOCK, &mut params)?;
            buffers.pending_launch = true;
            domain.pending = true;
            domain.synchronize_pending()?;
            buffers.pending_launch = false;
            let mut output = vec![0u8; output_bytes];
            self.download(output_ptr, &mut output)?;
            Ok((output, dispatch))
        }

        fn run_u8_mode(
            &self,
            kernel: &'static str,
            grid: (u32, u32),
            inputs: &[&[u8]],
            uniform_bytes: &[u8],
            init_output_from: Option<usize>,
            alias_input0: bool,
            cached_inputs: &[usize],
        ) -> Result<(Vec<u8>, Dispatch), CudaError> {
            let output_bytes = inputs.first().map_or(0, |first| first.len());
            if output_bytes == 0 || uniform_bytes.is_empty() || uniform_bytes.len() % 4 != 0 {
                return Err(CudaError::Refused(Refusal::ShapeMismatch("buffers")));
            }
            if cached_inputs.iter().any(|&index| index >= inputs.len()) {
                return Err(CudaError::Refused(Refusal::ShapeMismatch("cached_inputs")));
            }
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let module = domain.module(self.image)?.module as CuModule;
            let api = domain.api;
            let shape = StagingShape {
                input_count: inputs.len(),
                input_bytes: inputs.iter().map(|input| input.len()).collect(),
                output_bytes,
                alias_input0,
            };
            let buffers = domain.generic_staging(&shape)?;
            let output = if alias_input0 {
                buffers[0]
            } else {
                buffers[shape.input_count]
            };
            for (index, (ptr, input)) in buffers
                .iter()
                .take(shape.input_count)
                .zip(inputs)
                .enumerate()
            {
                if !cached_inputs.contains(&index) {
                    self.upload(*ptr, input)?;
                }
            }
            let mut cached_ptrs = Vec::with_capacity(cached_inputs.len());
            for &index in cached_inputs {
                cached_ptrs.push((index, domain.cached_teeth_lut(inputs[index])?));
            }
            if let Some(index) = init_output_from {
                let seed = inputs
                    .get(index)
                    .ok_or(CudaError::Refused(Refusal::ShapeMismatch(
                        "init_output_from",
                    )))?;
                if seed.len() != output_bytes {
                    return Err(CudaError::Refused(Refusal::ShapeMismatch(
                        "init_output_from",
                    )));
                }
                self.copy_from_device(output, buffers[index], output_bytes)?;
            }

            let count = shape.input_count;
            let mut params: Vec<*mut c_void> = Vec::with_capacity(count + 2);
            let mut args = buffers[..count].to_vec();
            for &(index, ptr) in &cached_ptrs {
                args[index] = ptr;
            }
            for arg in &mut args {
                params.push(arg as *mut CuPtr as *mut c_void);
            }
            let mut output_arg = output;
            params.push(&mut output_arg as *mut CuPtr as *mut c_void);
            let mut block = uniform_bytes.to_vec();
            params.push(block.as_mut_ptr() as *mut c_void);

            let dispatch = self.launch_on(&api, module, kernel, grid, BLOCK, &mut params)?;
            domain.pending = true;
            let mut bytes = vec![0u8; output_bytes];
            let downloaded = self.download(output, &mut bytes);
            if downloaded.is_ok() {
                domain.pending = false;
            }
            downloaded?;
            Ok((bytes, dispatch))
        }

        fn dispatch(&self, kernel: &'static str, grid: (u32, u32), block: (u32, u32)) -> Dispatch {
            Dispatch {
                kernel,
                grid,
                block,
                ptx_bytes: self.image_bytes,
                ptx_isa: self.ptx_isa,
                device: self.name.clone(),
                compute_cap: self.compute_cap,
                driver_version: self.driver_version,
            }
        }

        fn launch_on(
            &self,
            api: &Api,
            module: CuModule,
            kernel: &'static str,
            geometry: (u32, u32),
            block: (u32, u32),
            params: &mut [*mut c_void],
        ) -> Result<Dispatch, CudaError> {
            let symbol = CString::new(kernel).expect("static kernel name");
            let mut function: CuFunction = ptr::null_mut();
            unsafe {
                check(
                    (api.module_function)(&mut function, module, symbol.as_ptr()),
                    "cuModuleGetFunction",
                )?;
                check(
                    (api.launch)(
                        function,
                        geometry.0,
                        geometry.1,
                        1,
                        block.0,
                        block.1,
                        1,
                        0,
                        ptr::null_mut(),
                        params.as_mut_ptr(),
                        ptr::null_mut(),
                    ),
                    "cuLaunchKernel",
                )?;
            }
            Ok(self.dispatch(kernel, geometry, block))
        }

        /// Crate-private: the same launch path with an explicit block shape, for
        /// kernels whose geometry is not the P15 16x16 tiling.
        #[allow(dead_code)] // consumed by `cuda_mesh`
        pub fn launch_raw(
            &self,
            kernel: &'static str,
            geometry: (u32, u32),
            block: (u32, u32),
            params: &mut [*mut c_void],
        ) -> Result<Dispatch, CudaError> {
            self.with_domain(|domain, module| {
                let dispatch =
                    self.launch_on(&domain.api, module, kernel, geometry, block, params)?;
                domain.pending = true;
                Ok(dispatch)
            })
        }

        fn upload(&self, ptr: CuPtr, data: &[u8]) -> Result<(), CudaError> {
            unsafe {
                check(
                    (self.api.copy_in)(ptr, data.as_ptr() as *const c_void, data.len()),
                    "cuMemcpyHtoD",
                )?
            };
            HOST_TO_DEVICE_BYTES.fetch_add(data.len(), Ordering::Relaxed);
            Ok(())
        }

        fn download(&self, ptr: CuPtr, data: &mut [u8]) -> Result<(), CudaError> {
            unsafe {
                check(
                    (self.api.copy_out)(data.as_mut_ptr() as *mut c_void, ptr, data.len()),
                    "cuMemcpyDtoH",
                )
            }?;
            DEVICE_TO_HOST_BYTES.fetch_add(data.len(), Ordering::Relaxed);
            Ok(())
        }

        fn copy_from_device(&self, dst: CuPtr, src: CuPtr, bytes: usize) -> Result<(), CudaError> {
            unsafe { check((self.api.copy_device)(dst, src, bytes), "cuMemcpyDtoD") }
        }

        pub(crate) fn with_reusable_buffers<T>(
            &self,
            pool: &'static str,
            sizes: &[usize],
            call: impl FnOnce(&mut BufferSession<'_>) -> Result<T, CudaError>,
        ) -> Result<T, CudaError> {
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let module = domain.module(self.image)?.module as CuModule;
            let ptrs = domain.named_buffers(NamedBufferShape {
                pool,
                sizes: sizes.to_vec(),
            })?;
            let mut session = BufferSession {
                gpu: self,
                domain: &mut domain,
                module,
                sizes: sizes.to_vec(),
                ptrs,
            };
            match call(&mut session) {
                Ok(result) => Ok(result),
                Err(error) => {
                    if session.domain.pending {
                        let _ = session.domain.synchronize_pending();
                    }
                    Err(error)
                }
            }
        }

        /// Crate-private: allocate a buffer that lives across several launches.
        #[allow(dead_code)] // consumed by `cuda_mesh`
        pub fn alloc_buffer(&self, bytes: usize) -> Result<DeviceBuffer<'_>, CudaError> {
            self.with_domain(|domain, _| {
                let mut ptr: CuPtr = 0;
                unsafe {
                    check((domain.api.mem_alloc)(&mut ptr, bytes), "cuMemAlloc")?;
                }
                count_device_allocation();
                Ok(DeviceBuffer {
                    ptr,
                    owner: std::marker::PhantomData,
                })
            })
        }

        #[allow(dead_code)] // consumed by `cuda_mesh`
        pub fn upload_buffer(
            &self,
            buffer: &DeviceBuffer<'_>,
            data: &[u8],
        ) -> Result<(), CudaError> {
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            domain.bind()?;
            unsafe {
                check(
                    (domain.api.copy_in)(buffer.ptr, data.as_ptr() as *const c_void, data.len()),
                    "cuMemcpyHtoD",
                )
            }
        }

        #[allow(dead_code)] // consumed by `cuda_mesh`
        pub fn download_buffer(
            &self,
            buffer: &DeviceBuffer<'_>,
            data: &mut [u8],
        ) -> Result<(), CudaError> {
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            domain.bind()?;
            let downloaded = unsafe {
                check(
                    (domain.api.copy_out)(data.as_mut_ptr() as *mut c_void, buffer.ptr, data.len()),
                    "cuMemcpyDtoH",
                )
            };
            if downloaded.is_ok() {
                domain.pending = false;
            }
            downloaded
        }

        /// Crate-private: allocate a group of buffers that can be reused for
        /// every launch of the same job shape.
        #[allow(dead_code)] // consumed by the O1h session benchmark
        pub(crate) fn alloc_set(&self, sizes: &[usize]) -> Result<DeviceSet<'_>, CudaError> {
            self.with_domain(|domain, _| {
                let mut buffers = Vec::with_capacity(sizes.len());
                for size in sizes {
                    let mut ptr: CuPtr = 0;
                    unsafe {
                        check(
                            (domain.api.mem_alloc)(&mut ptr, (*size).max(1)),
                            "cuMemAlloc",
                        )?;
                    }
                    count_device_allocation();
                    buffers.push((ptr, *size));
                }
                Ok(DeviceSet {
                    buffers,
                    owner: std::marker::PhantomData,
                })
            })
        }

        #[allow(dead_code)] // consumed by the O1h session benchmark
        pub(crate) fn set_upload(
            &self,
            set: &DeviceSet<'_>,
            index: usize,
            data: &[u8],
        ) -> Result<(), CudaError> {
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            domain.bind()?;
            match set.buffers.get(index) {
                Some((ptr, size)) if data.len() <= *size => unsafe {
                    check(
                        (domain.api.copy_in)(*ptr, data.as_ptr() as *const c_void, data.len()),
                        "cuMemcpyHtoD",
                    )
                },
                _ => Err(CudaError::Refused(Refusal::ShapeMismatch("buffer"))),
            }
        }

        #[allow(dead_code)] // consumed by the O1h session benchmark
        pub(crate) fn set_download(
            &self,
            set: &DeviceSet<'_>,
            index: usize,
            data: &mut [u8],
        ) -> Result<(), CudaError> {
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            domain.bind()?;
            let downloaded = match set.buffers.get(index) {
                Some((ptr, size)) if data.len() <= *size => unsafe {
                    check(
                        (domain.api.copy_out)(data.as_mut_ptr() as *mut c_void, *ptr, data.len()),
                        "cuMemcpyDtoH",
                    )
                },
                _ => Err(CudaError::Refused(Refusal::ShapeMismatch("buffer"))),
            };
            if downloaded.is_ok() {
                domain.pending = false;
            }
            downloaded
        }

        /// Runs the composite kernel on float32 RGBA buffers already in [0,1].
        ///
        /// `mask` is RGBA (`.r` and `.b` are read), `skin_mask` is one float per
        /// pixel, exactly as the recovered contract declares them. This is the
        /// float-in/float-out entry point, so callers can compare against
        /// `composite_pixel` goldens without RGBA8 quantisation.
        pub fn composite_f32(
            &self,
            source: &[f32],
            mask: &[f32],
            skin_mask: &[f32],
            base: &[f32],
            residual: &[f32],
            uniforms: &Uniforms,
        ) -> Result<(Vec<f32>, Dispatch), CudaError> {
            uniforms.validate()?;
            let len = uniforms.buffer_f32();
            for (field, buffer) in [
                ("source", source.len()),
                ("mask_rgba", mask.len()),
                ("base", base.len()),
                ("residual", residual.len()),
            ] {
                if buffer != len {
                    return Err(CudaError::Refused(Refusal::ShapeMismatch(field)));
                }
            }
            if skin_mask.len() != len / 4 {
                return Err(CudaError::Refused(Refusal::ShapeMismatch("skin_mask")));
            }
            for values in [source, base, residual] {
                check_pixels(values, uniforms)?;
            }
            check_masks(mask, skin_mask, uniforms)?;
            let bytes = len * 4;
            let skin_bytes = skin_mask.len() * 4;
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let module = domain.module(self.image)?.module as CuModule;
            let [
                device_source,
                device_mask,
                device_skin,
                device_base,
                device_residual,
                device_output,
            ] = domain
                .composite_buffers(uniforms, [bytes, bytes, skin_bytes, bytes, bytes, bytes])?;
            self.upload(device_source, as_bytes(source))?;
            self.upload(device_mask, as_bytes(mask))?;
            self.upload(device_skin, as_bytes(skin_mask))?;
            self.upload(device_base, as_bytes(base))?;
            self.upload(device_residual, as_bytes(residual))?;

            let mut source_arg = device_source;
            let mut mask_arg = device_mask;
            let mut skin_arg = device_skin;
            let mut base_arg = device_base;
            let mut residual_arg = device_residual;
            let mut output_arg = device_output;
            let mut uniforms_arg = *uniforms;
            let mut params: [*mut c_void; 7] = [
                &mut source_arg as *mut CuPtr as *mut c_void,
                &mut mask_arg as *mut CuPtr as *mut c_void,
                &mut skin_arg as *mut CuPtr as *mut c_void,
                &mut base_arg as *mut CuPtr as *mut c_void,
                &mut residual_arg as *mut CuPtr as *mut c_void,
                &mut output_arg as *mut CuPtr as *mut c_void,
                &mut uniforms_arg as *mut Uniforms as *mut c_void,
            ];
            let dispatch = self.launch_on(
                &domain.api,
                module,
                super::COMPOSITE,
                grid(uniforms.width, uniforms.height),
                BLOCK,
                &mut params,
            )?;
            domain.pending = true;
            let mut output = vec![0f32; len];
            // DtoH copies to pageable host memory wait for prior launches.
            let downloaded = self.download(device_output, as_bytes_mut(&mut output));
            if downloaded.is_ok() {
                domain.pending = false;
            }
            downloaded?;
            Ok((output, dispatch))
        }

        /// Runs one blur pass, returning the pass output and its dispatch.
        pub fn run_pass(
            &self,
            pass: Pass,
            source: &[f32],
            uniforms: &Uniforms,
        ) -> Result<(Vec<f32>, Dispatch), CudaError> {
            uniforms.validate()?;
            let len = uniforms.buffer_f32();
            if source.len() != len {
                return Err(CudaError::Refused(Refusal::ShapeMismatch("source")));
            }
            check_pixels(source, uniforms)?;
            let bytes = len * 4;
            let domain = domain()?;
            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let module = domain.module(self.image)?.module as CuModule;
            let api = domain.api;
            let mut buffers = Buffers {
                api: &api,
                ptrs: Vec::new(),
                pending_launch: false,
            };
            let device_source = buffers.alloc(bytes)?;
            let device_output = buffers.alloc(bytes)?;
            self.upload(device_source, as_bytes(source))?;
            let mut source_arg = device_source;
            let mut output_arg = device_output;
            let mut uniforms_arg = *uniforms;
            let mut params: [*mut c_void; 3] = [
                &mut source_arg as *mut CuPtr as *mut c_void,
                &mut output_arg as *mut CuPtr as *mut c_void,
                &mut uniforms_arg as *mut Uniforms as *mut c_void,
            ];
            let dispatch = self.launch_on(
                &api,
                module,
                pass.kernel(),
                grid(uniforms.width, uniforms.height),
                BLOCK,
                &mut params,
            )?;
            buffers.pending_launch = true;
            domain.pending = true;
            let mut output = vec![0f32; len];
            let downloaded = self.download(device_output, as_bytes_mut(&mut output));
            if downloaded.is_ok() {
                domain.pending = false;
                buffers.pending_launch = false;
            }
            downloaded?;
            Ok((output, dispatch))
        }

        /// Drop-in GPU twin of `kernels::facial_smooth::apply`.
        pub fn facial_smooth(
            &self,
            source: &Image,
            base: &Image,
            residual: &Image,
            mask_r: &[u8],
            mask_b: &[u8],
            skin_mask: &[u8],
            pf: u32,
            pf_value: f32,
            uniforms: &cpu::Uniforms,
        ) -> Result<(Image, Dispatch), CudaError> {
            source.validate()?;
            base.validate()?;
            residual.validate()?;
            if (source.width, source.height) != (base.width, base.height)
                || (source.width, source.height) != (residual.width, residual.height)
            {
                return Err(Refusal::ShapeMismatch("base/residual").into());
            }
            let pixels = source.pixel_count();
            for (field, buffer) in [
                ("mask_r", mask_r.len()),
                ("mask_b", mask_b.len()),
                ("skin_mask", skin_mask.len()),
            ] {
                if buffer != pixels {
                    return Err(Refusal::ShapeMismatch(field).into());
                }
            }
            let uniforms = Uniforms::new(source.width, source.height, pf, pf_value, uniforms);
            let mask: Vec<f32> = (0..pixels)
                .flat_map(|i| [mask_r[i] as f32 / 255.0, 0.0, mask_b[i] as f32 / 255.0, 0.0])
                .collect();
            let skin: Vec<f32> = skin_mask.iter().map(|byte| *byte as f32 / 255.0).collect();
            let (output, dispatch) = self.composite_f32(
                &rgba_f32(source),
                &mask,
                &skin,
                &rgba_f32(base),
                &rgba_f32(residual),
                &uniforms,
            )?;
            Ok((quantize(&output, source), dispatch))
        }
    }

    fn as_bytes(values: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
    }

    fn as_bytes_mut(values: &mut [f32]) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, values.len() * 4) }
    }

    #[cfg(test)]
    mod staging_tests {
        use super::*;

        #[test]
        fn lut_buffer_pool_key_is_exact_grid_and_bytes() {
            let shape = LutBufferShape {
                grid: (375, 250),
                source_bytes: 96_000_000,
                mask_bytes: 24_000_000,
            };
            let entry = CachedLutBuffers {
                shape,
                buffers: vec![(1, 96_000_000), (2, 96_000_000), (3, 24_000_000)],
            };
            assert!(entry.matches(shape));
            assert!(!entry.matches(LutBufferShape {
                grid: (32, 32),
                ..shape
            }));
            assert!(!entry.matches(LutBufferShape {
                source_bytes: 95_999_996,
                ..shape
            }));
            assert!(!entry.matches(LutBufferShape {
                mask_bytes: 23_999_996,
                ..shape
            }));
        }

        #[test]
        fn composite_pool_key_is_exact_shape_and_sizes() {
            let shape = BufferShape {
                width: 64,
                height: 64,
                stride: 64,
            };
            let sizes = [16384, 16384, 4096, 16384, 16384, 16384];
            let entry = CachedComposite {
                shape,
                buffers: sizes
                    .iter()
                    .enumerate()
                    .map(|(index, size)| (index as CuPtr + 1, *size))
                    .collect(),
            };
            assert!(entry.matches(shape, &sizes));
            assert!(!entry.matches(BufferShape { width: 63, ..shape }, &sizes));
            let mut changed = sizes;
            changed[5] -= 4;
            assert!(!entry.matches(shape, &changed));
        }

        #[test]
        fn cache_keys_exact_shape_and_input_count() {
            let domain = match domain() {
                Ok(domain) => domain,
                Err(error) if std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none() => {
                    eprintln!("SKIPPED (no NVIDIA execution path): {error:?}");
                    return;
                }
                Err(error) => panic!("CUDA required but unavailable: {error:?}"),
            };
            {
                let mut domain = domain
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                domain.bind().expect("bind shared context");
                domain
                    .synchronize_pending()
                    .expect("sync before cache clear");
                domain.clear_generic_staging();
            }

            let one = StagingShape {
                input_count: 1,
                input_bytes: vec![3584],
                output_bytes: 3584,
                alias_input0: false,
            };
            let two = StagingShape {
                input_count: 2,
                input_bytes: vec![3584, 1792],
                output_bytes: 3584,
                alias_input0: false,
            };
            let allocations = || device_allocations();
            let cache = |shape: &StagingShape| {
                let mut domain = domain
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                domain.bind().expect("bind shared context");
                domain.generic_staging(shape).expect("staging buffers")
            };

            let before = allocations();
            let first = cache(&one);
            let cold = allocations() - before;
            let before = allocations();
            let repeated = cache(&one);
            let warm = allocations() - before;
            let before = allocations();
            let with_two_inputs = cache(&two);
            let second_shape = allocations() - before;
            let before = allocations();
            let repeated_two = cache(&two);
            let warm_two = allocations() - before;

            assert_eq!(first, repeated);
            assert_eq!(first.len(), 2);
            assert_eq!(cold, 2);
            assert_eq!(warm, 0);
            assert_eq!(with_two_inputs.len(), 3);
            assert_eq!(second_shape, 3);
            assert_eq!(repeated_two, with_two_inputs);
            assert_eq!(warm_two, 0);
            assert_eq!(
                cache(&one),
                first,
                "the second cache slot keeps the first shape"
            );
            eprintln!(
                "one_input cold_alloc={cold} warm_alloc={warm}; two_inputs cold_alloc={second_shape} warm_alloc={warm_two}; two_slots=true"
            );

            let mut domain = domain
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            domain
                .synchronize_pending()
                .expect("sync before cache clear");
            domain.clear_generic_staging();
        }

        #[test]
        fn named_pool_reuses_exact_key_and_shape() {
            let nvidia = match Nvidia::open() {
                Ok(nvidia) => nvidia,
                Err(error) if std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none() => {
                    eprintln!("SKIPPED (no NVIDIA execution path): {error:?}");
                    return;
                }
                Err(error) => panic!("CUDA required but unavailable: {error:?}"),
            };
            let sizes = [4096usize, 8192];
            let before = device_allocations();
            let first = nvidia
                .with_reusable_buffers("test.n35.named", &sizes, |buffers| {
                    Ok(vec![buffers.handle(0), buffers.handle(1)])
                })
                .expect("cold named buffers");
            let cold = device_allocations() - before;
            let before = device_allocations();
            let repeated = nvidia
                .with_reusable_buffers("test.n35.named", &sizes, |buffers| {
                    Ok(vec![buffers.handle(0), buffers.handle(1)])
                })
                .expect("warm named buffers");
            let warm = device_allocations() - before;
            let before = device_allocations();
            let other_key = nvidia
                .with_reusable_buffers("test.n35.named-other", &sizes, |buffers| {
                    Ok(vec![buffers.handle(0), buffers.handle(1)])
                })
                .expect("distinct named buffers");
            let distinct = device_allocations() - before;

            assert_eq!(first, repeated);
            assert_ne!(first, other_key);
            assert_eq!(cold, 2);
            assert_eq!(warm, 0);
            assert_eq!(distinct, 2);
            eprintln!(
                "N35_NAMED_POOL cold={cold} warm={warm} distinct={distinct} buffers={}",
                first.len()
            );
        }
    }

    pub(super) fn metrics() -> DriverMetrics {
        DriverMetrics {
            ptx_module_load_calls: PTX_MODULE_LOADS.load(Ordering::Relaxed),
            skin_lut_uploads: SKIN_LUT_UPLOADS.load(Ordering::Relaxed),
            teeth_lut_uploads: TEETH_LUT_UPLOADS.load(Ordering::Relaxed),
            host_to_device_bytes: HOST_TO_DEVICE_BYTES.load(Ordering::Relaxed),
            device_to_host_bytes: DEVICE_TO_HOST_BYTES.load(Ordering::Relaxed),
            device_allocations: device_allocations(),
            context_syncs: context_syncs(),
        }
    }

    pub(super) fn memory_info() -> Result<DeviceMemoryInfo, CudaError> {
        let domain = domain()?;
        let mut domain = domain
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        domain.bind()?;
        let mut free_bytes = 0usize;
        let mut total_bytes = 0usize;
        unsafe {
            check(
                (domain.api.mem_get_info)(&mut free_bytes, &mut total_bytes),
                "cuMemGetInfo",
            )?;
        }
        Ok(DeviceMemoryInfo {
            free_bytes,
            total_bytes,
        })
    }
}

#[cfg(windows)]
pub use driver::DeviceBuffer;
#[cfg(windows)]
pub use driver::Nvidia;

/// Session buffer group, shared by the LUT stages inside this crate.
#[cfg(windows)]
#[allow(unused_imports)] // used by `cuda_teeth`, which lib.rs mounts separately
pub(crate) use driver::DeviceSet;

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    fn samples() -> cpu::Uniforms {
        cpu::Uniforms {
            legal_alpha: 0.7,
            dark_circle_alpha: 0.9,
            diff_factor: 0.6,
            diff_factor2: 0.25,
            color_threshold_factor: 0.4,
            mix_alpha: 0.35,
            diff_min_threshold: -0.5,
            diff_max_threshold: 0.5,
            high_freq_min_threshold: -0.06,
            high_freq_max_threshold: 0.04,
        }
    }

    fn block(width: u32, height: u32) -> Uniforms {
        let mut uniforms = Uniforms::new(width, height, 90024, 0.5, &samples());
        uniforms.surface_radius = 2.0;
        uniforms.surface_threshold = 30.0;
        uniforms.gauss_offset = 1.5;
        uniforms.box_blur_stage = 1;
        uniforms
    }

    /// Deterministic noise so a failure is reproducible.
    fn noise(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn kernel_argument_block_matches_the_c_header() {
        assert_eq!(size_of::<Uniforms>(), 88);
        assert_eq!(align_of::<Uniforms>(), 4);
        let uniforms = block(8, 8);
        assert_eq!(&uniforms as *const Uniforms as usize % 4, 0);
    }

    #[test]
    fn isa_downgrade_touches_only_the_header() {
        assert_eq!(ptx_isa(PTX), Some((9, 4)));
        let lowered = restamp_ptx(PTX, (9, 3)).expect("9.4 image can be lowered to 9.3");
        assert_eq!(ptx_isa(&lowered), Some((9, 3)));
        assert_eq!(lowered.len(), PTX.len());
        assert_eq!(restamp_ptx(PTX, (9, 4)), None);
        assert_eq!(restamp_ptx(PTX, (10, 0)), None);
    }

    #[test]
    fn driver_version_maps_to_the_highest_jittable_isa() {
        // The shipping host: driver API 13030 must accept ISA 9.3, not 9.4.
        assert_eq!(driver_ptx_isa(13030), (9, 3));
        assert_eq!(driver_ptx_isa(13040), (9, 4));
        assert_eq!(driver_ptx_isa(12020), (8, 2));
        // A driver older than CUDA 4 has no PTX ISA this mapping can name.
        assert_eq!(driver_ptx_isa(3020), (0, 2));
        assert_eq!(restamp_ptx(PTX, driver_ptx_isa(3020)), None);
        // Re-stamping up front is what the loader now does, so the driver that
        // refuses 9.4 lands on 9.3 directly rather than through error 222.
        let target = driver_ptx_isa(13030);
        assert_eq!(
            restamp_ptx(PTX, target).map(|p| ptx_isa(&p)),
            Some(Some(target))
        );
    }

    #[test]
    fn layout_gates_are_refused() {
        let mut uniforms = block(8, 8);
        uniforms.width = 0;
        assert_eq!(uniforms.validate(), Err(Refusal::ShapeMismatch("uniforms")));

        let mut uniforms = block(8, 8);
        uniforms.stride = 4;
        assert_eq!(uniforms.validate(), Err(Refusal::ShapeMismatch("uniforms")));

        let mut uniforms = block(8, 8);
        uniforms.roi_size = 16;
        assert_eq!(uniforms.validate(), Err(Refusal::ShapeMismatch("roi_size")));

        let mut uniforms = block(8, 8);
        uniforms.pf = 90025;
        assert_eq!(uniforms.validate(), Err(Refusal::UnknownEffect(90025)));

        let mut uniforms = block(8, 8);
        uniforms.pf_value = 0.001;
        assert_eq!(
            uniforms.validate(),
            Err(Refusal::ValueOutOfRange("pf_value"))
        );

        let mut uniforms = block(8, 8);
        uniforms.gate = 0.002;
        assert_eq!(
            uniforms.validate(),
            Err(Refusal::ValueOutOfRange("pf_value"))
        );

        let mut uniforms = block(8, 8);
        uniforms.algo_type = 1;
        assert_eq!(uniforms.validate(), Err(Refusal::NotImplemented(1)));

        let mut uniforms = block(8, 8);
        uniforms.mix_alpha = f32::NAN;
        assert_eq!(
            uniforms.validate(),
            Err(Refusal::ValueOutOfRange("uniforms"))
        );

        let mut uniforms = block(8, 8);
        uniforms.diff_min_threshold = 2.0;
        assert_eq!(
            uniforms.validate(),
            Err(Refusal::ValueOutOfRange("uniforms"))
        );
    }

    #[cfg(windows)]
    fn gpu() -> Option<Nvidia> {
        match Nvidia::open() {
            Ok(nvidia) => Some(nvidia),
            Err(error) => {
                if std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_some() {
                    panic!("CUDA required but unavailable: {error:?}");
                }
                eprintln!("SKIPPED (no NVIDIA execution path): {error:?}");
                None
            }
        }
    }

    #[cfg(windows)]
    #[test]
    fn composite_float_output_matches_composite_pixel() {
        let Some(nvidia) = gpu() else { return };
        let (width, height) = (16, 16);
        let pixels = (width * height) as usize;
        let source: Vec<f32> = noise(pixels * 4, 7)
            .iter()
            .map(|b| *b as f32 / 255.0)
            .collect();
        let base: Vec<f32> = noise(pixels * 4, 11)
            .iter()
            .map(|b| *b as f32 / 255.0)
            .collect();
        let residual: Vec<f32> = noise(pixels * 4, 13)
            .iter()
            .map(|b| *b as f32 / 255.0)
            .collect();
        let mask_r = noise(pixels, 17);
        let mask_b = noise(pixels, 19);
        let skin = noise(pixels, 23);
        let mask: Vec<f32> = (0..pixels)
            .flat_map(|i| [mask_r[i] as f32 / 255.0, 0.0, mask_b[i] as f32 / 255.0, 0.0])
            .collect();
        let skin_plane: Vec<f32> = skin.iter().map(|b| *b as f32 / 255.0).collect();
        let uniforms = block(width, height);
        // The single-channel skin plane is one float per pixel; RGBA packing is
        // the bug this entry point refuses rather than silently mis-indexes.
        assert_eq!(
            nvidia
                .composite_f32(
                    &source,
                    &mask,
                    &skin_plane.repeat(4),
                    &base,
                    &residual,
                    &uniforms
                )
                .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("skin_mask")))
        );
        let (gpu, dispatch) = nvidia
            .composite_f32(&source, &mask, &skin_plane, &base, &residual, &uniforms)
            .expect("real GPU dispatch");
        assert_eq!(dispatch.kernel, COMPOSITE);
        assert_eq!(dispatch.ptx_bytes, PTX.len());
        assert!(dispatch.compute_cap.0 >= 8, "PTX needs sm_89 or newer");

        let mut worst = 0f32;
        for pixel in 0..pixels {
            let at = pixel * 4;
            let rgba =
                |values: &[f32]| [values[at], values[at + 1], values[at + 2], values[at + 3]];
            let expected = cpu::composite_pixel(
                rgba(&source),
                rgba(&base),
                rgba(&residual),
                mask_r[pixel] as f32 / 255.0,
                mask_b[pixel] as f32 / 255.0,
                skin[pixel] as f32 / 255.0,
                samples(),
            )
            .unwrap();
            for channel in 0..4 {
                let delta = (expected[channel] - gpu[at + channel]).abs();
                if delta > 1e-5 {
                    panic!(
                        "pixel {pixel} channel {channel}: cpu {:?} gpu {:?} source {:?} base {:?} residual {:?} masks {:?}",
                        expected,
                        &gpu[at..at + 4],
                        rgba(&source),
                        rgba(&base),
                        rgba(&residual),
                        (mask_r[pixel], mask_b[pixel], skin[pixel])
                    );
                }
                worst = worst.max(delta);
            }
        }
        eprintln!(
            "GPU {} cc{}.{} driver {} isa {}.{} kernel {} grid {}x{} block {}x{} worst_float_delta={}",
            dispatch.device,
            dispatch.compute_cap.0,
            dispatch.compute_cap.1,
            dispatch.driver_version,
            dispatch.ptx_isa.0,
            dispatch.ptx_isa.1,
            dispatch.kernel,
            dispatch.grid.0,
            dispatch.grid.1,
            dispatch.block.0,
            dispatch.block.1,
            worst,
        );
    }

    #[cfg(windows)]
    #[test]
    fn rgba8_wrapper_is_byte_identical_to_the_cpu_twin() {
        let Some(nvidia) = gpu() else { return };
        let (width, height) = (64, 64);
        let pixels = (width * height) as usize;
        let source = Image::new(width, height, noise(pixels * 4, 7)).unwrap();
        let base = Image::new(width, height, noise(pixels * 4, 11)).unwrap();
        let residual = Image::new(width, height, noise(pixels * 4, 13)).unwrap();
        let mask_r = noise(pixels, 17);
        let mask_b = noise(pixels, 19);
        let skin = noise(pixels, 23);

        let expected = cpu::apply(
            &source,
            &base,
            &residual,
            &mask_r,
            &mask_b,
            &skin,
            samples(),
        )
        .unwrap();
        let (actual, dispatch) = nvidia
            .facial_smooth(
                &source,
                &base,
                &residual,
                &mask_r,
                &mask_b,
                &skin,
                90024,
                0.5,
                &samples(),
            )
            .expect("real GPU dispatch");

        assert_eq!(dispatch.kernel, COMPOSITE);
        let mut worst = 0i16;
        let mut differing = 0usize;
        for (index, (left, right)) in expected.pixels.iter().zip(&actual.pixels).enumerate() {
            let delta = (*left as i16 - *right as i16).abs();
            differing += usize::from(delta != 0);
            if index % 4 == 3 {
                assert_eq!(delta, 0, "alpha diverged at pixel {}", index / 4);
            }
            worst = worst.max(delta);
        }
        // A one-level tolerance hid nothing here; the twin is exact, so demand it.
        assert_eq!(
            worst,
            0,
            "{differing} of {} bytes differ, max delta {worst}",
            expected.pixels.len()
        );
    }

    #[cfg(windows)]
    fn smooth_case(
        width: u32,
        height: u32,
        constant: bool,
    ) -> (Image, Image, Image, Vec<u8>, Vec<u8>, Vec<u8>) {
        let pixels = (width * height) as usize;
        if constant {
            (
                Image::solid(width, height, [127, 63, 191, 255]),
                Image::solid(width, height, [91, 113, 137, 255]),
                Image::solid(width, height, [19, 29, 43, 255]),
                vec![128; pixels],
                vec![64; pixels],
                vec![200; pixels],
            )
        } else {
            (
                Image::new(width, height, noise(pixels * 4, 7)).unwrap(),
                Image::new(width, height, noise(pixels * 4, 11)).unwrap(),
                Image::new(width, height, noise(pixels * 4, 13)).unwrap(),
                noise(pixels, 17),
                noise(pixels, 19),
                noise(pixels, 23),
            )
        }
    }

    #[cfg(windows)]
    #[test]
    fn reused_shape_buffers_are_byte_identical_for_constant_and_random_fields() {
        let Some(nvidia) = gpu() else { return };
        nvidia
            .clear_composite_buffers()
            .expect("clear composite cache");
        let cases = [
            ("constant_64", 64, 64, true, 6usize),
            ("random_64_same_shape", 64, 64, false, 0usize),
            ("random_37x19_shape_change", 37, 19, false, 6usize),
            ("random_64_return_shape", 64, 64, false, 0usize),
        ];
        for (label, width, height, constant, cold_allocations) in cases {
            let (source, base, residual, mask_r, mask_b, skin) =
                smooth_case(width, height, constant);
            let expected = cpu::apply(
                &source,
                &base,
                &residual,
                &mask_r,
                &mask_b,
                &skin,
                samples(),
            )
            .unwrap();

            let before_syncs = super::driver::context_syncs();
            let before = super::driver::device_allocations();
            let (first, dispatch) = nvidia
                .facial_smooth(
                    &source,
                    &base,
                    &residual,
                    &mask_r,
                    &mask_b,
                    &skin,
                    90024,
                    0.5,
                    &samples(),
                )
                .expect("real GPU dispatch");
            let cold = super::driver::device_allocations() - before;
            let cold_syncs = super::driver::context_syncs() - before_syncs;

            let before_syncs = super::driver::context_syncs();
            let before = super::driver::device_allocations();
            let (second, _) = nvidia
                .facial_smooth(
                    &source,
                    &base,
                    &residual,
                    &mask_r,
                    &mask_b,
                    &skin,
                    90024,
                    0.5,
                    &samples(),
                )
                .expect("warm GPU dispatch");
            let warm = super::driver::device_allocations() - before;
            let warm_syncs = super::driver::context_syncs() - before_syncs;

            assert_eq!(cold, cold_allocations, "{label}: cold allocations");
            assert_eq!(warm, 0, "{label}: warm allocations");
            assert_eq!(
                cold_syncs, 0,
                "{label}: readback owns the cold sync boundary"
            );
            assert_eq!(
                warm_syncs, 0,
                "{label}: readback owns the warm sync boundary"
            );
            assert_eq!(dispatch.kernel, COMPOSITE);
            assert_eq!(first.pixels, expected.pixels, "{label}: CPU byte mismatch");
            assert_eq!(
                second.pixels, expected.pixels,
                "{label}: warm CPU byte mismatch"
            );
            assert_eq!(first.pixels, second.pixels, "{label}: reuse changed output");
            eprintln!(
                "{label} {}x{} cold_alloc={cold} warm_alloc={warm} cold_sync={cold_syncs} warm_sync={warm_syncs} kernel={} device={} bytes={} exact",
                width,
                height,
                dispatch.kernel,
                dispatch.device,
                first.pixels.len()
            );
        }

        let (source, base, residual, mask_r, mask_b, skin) = smooth_case(64, 64, false);
        let expected = cpu::apply(
            &source,
            &base,
            &residual,
            &mask_r,
            &mask_b,
            &skin,
            samples(),
        )
        .unwrap();
        let before_allocations = super::driver::device_allocations();
        let before_syncs = super::driver::context_syncs();
        let mut micros = Vec::with_capacity(20);
        for _ in 0..20 {
            let started = std::time::Instant::now();
            let (actual, _) = nvidia
                .facial_smooth(
                    &source,
                    &base,
                    &residual,
                    &mask_r,
                    &mask_b,
                    &skin,
                    90024,
                    0.5,
                    &samples(),
                )
                .expect("warm timing dispatch");
            micros.push(started.elapsed().as_micros() as u64);
            assert_eq!(actual.pixels, expected.pixels);
        }
        assert_eq!(super::driver::device_allocations(), before_allocations);
        assert_eq!(super::driver::context_syncs(), before_syncs);
        micros.sort_unstable();
        eprintln!(
            "warm_64x64 samples=20 p50_us={} p95_us={} min_us={} max_us={}",
            micros[micros.len() / 2],
            micros[micros.len() * 95 / 100 - 1],
            micros[0],
            micros[micros.len() - 1],
        );
    }

    #[cfg(windows)]
    #[test]
    fn failed_kernel_load_reuses_the_shared_context() {
        {
            let Some(_probe) = gpu() else { return };
        }
        assert_eq!(super::driver::live_contexts(), 1, "one shared context");
        for _ in 0..8 {
            let error = match Nvidia::open_image("not a ptx image") {
                Err(error) => error,
                Ok(_) => panic!("junk PTX must not load"),
            };
            assert!(
                matches!(error, CudaError::Driver("cuModuleLoadDataEx", _)),
                "junk PTX must fail at load, got {error:?}"
            );
            assert_eq!(
                super::driver::live_contexts(),
                1,
                "a failed module load must not rebuild the context"
            );
        }
        let _gpu = Nvidia::open_image(PTX).expect("bundled image still loads");
        assert_eq!(super::driver::live_contexts(), 1);
    }

    #[cfg(windows)]
    #[test]
    fn portrait_stage_handles_share_one_context_without_holding_the_lock() {
        const SECOND_STAGE_PTX: &str = include_str!("../native/cuda/skin_lut.ptx");

        let Some(first) = gpu() else { return };
        let second = Nvidia::open_ptx(SECOND_STAGE_PTX).expect("second stage handle");
        let third = Nvidia::open_ptx(PTX).expect("third stage handle");
        assert_eq!(
            super::driver::live_contexts(),
            1,
            "all handles share one context"
        );

        let uniforms = block(16, 16);
        let source = vec![0.5; uniforms.buffer_f32()];
        first
            .run_pass(Pass::BoxBlur, &source, &uniforms)
            .expect("a retained handle still dispatches");

        drop((first, second, third));
        assert_eq!(
            super::driver::live_contexts(),
            1,
            "the process registry keeps the shared context alive"
        );
    }

    #[cfg(windows)]
    #[test]
    fn non_finite_and_out_of_range_inputs_are_refused() {
        let Some(nvidia) = gpu() else { return };
        let uniforms = block(4, 4);
        let len = uniforms.buffer_f32();
        let pixels = len / 4;
        let plane = vec![0.5f32; len];
        let mask = vec![0.5f32; len];
        let skin = vec![0.5f32; pixels];
        let run = |source: &[f32], mask: &[f32], skin: &[f32]| {
            nvidia
                .composite_f32(source, mask, skin, &plane, &plane, &uniforms)
                .err()
        };
        assert_eq!(
            run(&plane, &mask, &skin),
            None,
            "clean inputs must dispatch"
        );

        for slot in [0usize, len - 1] {
            let mut nan = plane.clone();
            nan[slot] = f32::NAN;
            assert_eq!(
                run(&nan, &mask, &skin),
                Some(CudaError::Refused(Refusal::ValueOutOfRange("pixel")))
            );

            let mut inf = plane.clone();
            inf[slot] = f32::NEG_INFINITY;
            assert_eq!(
                nvidia
                    .composite_f32(&plane, &mask, &skin, &inf, &plane, &uniforms)
                    .err(),
                Some(CudaError::Refused(Refusal::ValueOutOfRange("pixel")))
            );
        }

        for (slot, value) in [(0usize, 1.001f32), (2, -0.001), (0, f32::NAN)] {
            let mut broken = mask.clone();
            broken[slot] = value;
            assert_eq!(
                run(&plane, &broken, &skin),
                Some(CudaError::Refused(Refusal::ValueOutOfRange("mask")))
            );
        }
        // `.g` and `.a` are not weights the recovered law reads, so junk there
        // must not refuse a job the CPU lane would also accept.
        let mut unused = mask.clone();
        unused[1] = f32::NAN;
        unused[3] = 7.5;
        assert_eq!(run(&plane, &unused, &skin), None);
        let mut broken = skin.clone();
        broken[3] = 1.5;
        assert_eq!(
            run(&plane, &mask, &broken),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("mask")))
        );

        let mut broken = vec![0.5f32; len];
        broken[7] = f32::NAN;
        assert_eq!(
            nvidia.run_pass(Pass::BoxBlur, &broken, &uniforms).err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("pixel")))
        );
    }

    #[cfg(windows)]
    #[test]
    fn padded_rows_are_ignored_by_the_input_gates() {
        let Some(nvidia) = gpu() else { return };
        let mut uniforms = block(4, 4);
        uniforms.stride = 6;
        let len = uniforms.buffer_f32();
        let mut source = vec![0.5f32; len];
        let mask = vec![0.5f32; len];
        let skin = vec![0.5f32; len / 4];
        // Garbage in the two padding columns of every row must not be refused.
        for row in 0..uniforms.height as usize {
            for column in uniforms.width as usize..uniforms.stride as usize {
                source[row * uniforms.stride as usize * 4 + column * 4] = f32::NAN;
            }
        }
        let (output, dispatch) = nvidia
            .composite_f32(&source, &mask, &skin, &source, &source, &uniforms)
            .expect("padding is never read by the kernel");
        assert_eq!(output.len(), len);
        assert_eq!(dispatch.grid, (1, 1));
    }

    #[cfg(windows)]
    #[test]
    fn blur_passes_preserve_a_constant_field() {
        let Some(nvidia) = gpu() else { return };
        let uniforms = block(32, 32);
        let pixels = (uniforms.width * uniforms.height) as usize;
        let source: Vec<f32> = (0..pixels)
            .flat_map(|_| [0.25f32, 0.5, 0.75, 1.0])
            .collect();
        for pass in [Pass::Surface, Pass::Gauss, Pass::BoxBlur] {
            let (output, dispatch) = nvidia
                .run_pass(pass, &source, &uniforms)
                .expect("real GPU dispatch");
            assert_eq!(dispatch.kernel, pass.kernel());
            let worst = output
                .iter()
                .zip(&source)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0f32, f32::max);
            assert!(
                worst < 1e-6,
                "{:?} shifted a constant field by {worst}",
                pass
            );
        }
    }
}
