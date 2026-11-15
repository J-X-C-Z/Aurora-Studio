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
        if !matches!(self.pf, 90023 | 90024 | 21024 | 21025 | 21069 | 21070 | 21071) {
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
}

#[cfg(windows)]
pub fn driver_metrics() -> DriverMetrics {
    driver::metrics()
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
        cpu, ptx_isa, restamp_ptx, CudaError, Dispatch, DriverMetrics, Image, Pass, Refusal,
        Uniforms, PTX,
    };
    use std::cell::RefCell;
    #[cfg(test)]
    use std::cell::Cell;
    use std::ffi::{c_char, c_void, CString, CStr};
    use std::ptr;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Mutex, OnceLock};
    // Test-only process census. The domain is intentionally process-wide.
    #[cfg(test)]
    static LIVE_CONTEXTS: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);
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

    #[cfg(test)]
    fn count_device_allocation() {
        DEVICE_ALLOCATIONS.with(|count| count.set(count.get() + 1));
    }

    #[cfg(test)]
    pub(crate) fn device_allocations() -> usize {
        DEVICE_ALLOCATIONS.with(|count| count.get())
    }

    #[cfg(test)]
    fn count_context_sync() {
        CONTEXT_SYNCS.with(|count| count.set(count.get() + 1));
    }

    #[cfg(test)]
    pub(crate) fn context_syncs() -> usize {
        CONTEXT_SYNCS.with(|count| count.get())
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
    /// `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`.
    const UNSUPPORTED_PTX_VERSION: i32 = 222;

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
        module_function: unsafe extern "system" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
        mem_alloc: unsafe extern "system" fn(*mut CuPtr, usize) -> CuResult,
        mem_free: unsafe extern "system" fn(CuPtr) -> CuResult,
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
                    let address = GetProcAddress(handle, concat!($name, "\0").as_ptr() as *const c_char);
                    if address.is_null() {
                        return Err(CudaError::Unavailable($name));
                    }
                    std::mem::transmute::<*mut c_void, _>(address)
                }};
            }
            // 64-bit driver builds export the `_v2` forms; older ones only the plain name.
            macro_rules! sym2 {
                ($v2:literal, $plain:literal) => {{
                    let mut address = GetProcAddress(handle, concat!($v2, "\0").as_ptr() as *const c_char);
                    if address.is_null() {
                        address = GetProcAddress(handle, concat!($plain, "\0").as_ptr() as *const c_char);
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
        composite_buffers: Option<(BufferShape, Vec<(CuPtr, usize)>)>,
        name: String,
        compute_cap: (i32, i32),
        driver_version: i32,
    }

    static DOMAIN: OnceLock<Result<Mutex<Domain>, CudaError>> = OnceLock::new();
    static PTX_MODULE_LOADS: AtomicUsize = AtomicUsize::new(0);
    static SKIN_LUT_UPLOADS: AtomicUsize = AtomicUsize::new(0);
    const LUT_CACHE_SLOTS: usize = 2;

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
        lut_cache: RefCell<Vec<CachedLut>>,
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
                composite_buffers: None,
                name: String::new(),
                compute_cap: (0, 0),
                driver_version: version,
            };
            let mut name_buf = [0 as c_char; 256];
            let mut major = 0;
            let mut minor = 0;
            unsafe {
                check(
                    (domain.api.device_name)(
                        name_buf.as_mut_ptr(),
                        name_buf.len() as i32,
                        device,
                    ),
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
            self.synchronize_pending()?;
            if let Some(module) = self.modules.iter().find(|module| module.image == image) {
                return Ok(*module);
            }
            // Only driver error 222 licenses the header-only compatibility retry.
            let (module, isa, loaded) = match Self::load_module(&self.api, image) {
                Ok(module) => (
                    module,
                    ptx_isa(image).ok_or(CudaError::Unavailable("ptx image"))?,
                    image.len(),
                ),
                Err(CudaError::Driver(_, UNSUPPORTED_PTX_VERSION))
                    if self.driver_version >= 4000 =>
                {
                    let target = (
                        (self.driver_version as u32 / 1000 - 4),
                        (self.driver_version as u32 % 1000) / 10,
                    );
                    let downgraded =
                        restamp_ptx(image, target).ok_or(CudaError::Unavailable("ptx image"))?;
                    let module = Self::load_module(&self.api, &downgraded)?;
                    (
                        module,
                        ptx_isa(&downgraded).unwrap_or(target),
                        downgraded.len(),
                    )
                }
                Err(error) => return Err(error),
            };
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
                #[cfg(test)]
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
            let reusable = self
                .composite_buffers
                .as_ref()
                .is_some_and(|(cached_shape, buffers)| {
                    *cached_shape == shape
                        && buffers
                            .iter()
                            .map(|(_, size)| *size)
                            .eq(sizes.iter().copied())
                });
            if !reusable {
                if let Some((_, old)) = self.composite_buffers.take() {
                    for (ptr, _) in old {
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
                for size in sizes {
                    allocations.alloc(size)?;
                }
                self.composite_buffers = Some((
                    shape,
                    allocations.ptrs.drain(..).zip(sizes).collect(),
                ));
            }
            Ok(std::array::from_fn(|index| {
                self.composite_buffers
                    .as_ref()
                    .expect("composite buffer cache")
                    .1[index]
                    .0
            }))
        }
    }

    impl Drop for Domain {
        fn drop(&mut self) {
            let _ = self.synchronize_pending();
            if let Some((_, buffers)) = self.composite_buffers.take() {
                for (ptr, _) in buffers {
                    unsafe {
                        let _ = (self.api.mem_free)(ptr);
                    }
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
            #[cfg(test)]
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
                let mut domain =
                    domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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
                let mut domain =
                    domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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

    fn grid(width: u32, height: u32) -> (u32, u32) {
        (width.div_ceil(BLOCK.0).max(1), height.div_ceil(BLOCK.1).max(1))
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
        let mask_bad = rows(mask, uniforms, 4)
            .any(|row| row.chunks_exact(4).any(|pixel| !weight(pixel[0]) || !weight(pixel[2])));
        let skin_bad = rows(skin_mask, uniforms, 1).any(|row| row.iter().any(|value| !weight(*value)));
        if mask_bad || skin_bad {
            return Err(Refusal::ValueOutOfRange("mask"));
        }
        Ok(())
    }

    fn rgba_f32(image: &Image) -> Vec<f32> {
        image.pixels.iter().map(|byte| *byte as f32 / 255.0).collect()
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
            let mut domain = domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let module = domain.module(image)?;
            Ok(Self {
                image,
                api: domain.api,
                name: domain.name.clone(),
                compute_cap: domain.compute_cap,
                driver_version: domain.driver_version,
                ptx_isa: module.ptx_isa,
                image_bytes: module.image_bytes,
                lut_cache: RefCell::new(Vec::new()),
            })
        }

        /// Kept for the LUT session API. The operation lock is already
        /// released between calls, so this only refreshes the current context.
        pub(crate) fn resume(&mut self) -> Result<(), CudaError> {
            let domain = domain()?;
            let mut domain = domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            domain.bind()
        }

        /// Kept for the LUT session API. All work is synchronous at this
        /// boundary, and no lock is held after the operation returns.
        pub(crate) fn suspend(&mut self) -> Result<(), CudaError> {
            let domain = domain()?;
            let mut domain = domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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
            let mut domain = domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            domain.bind()?;
            domain.synchronize_pending()?;
            if let Some((_, buffers)) = domain.composite_buffers.take() {
                for (ptr, _) in buffers {
                    unsafe {
                        check((domain.api.mem_free)(ptr), "cuMemFree")?;
                    }
                }
            }
            Ok(())
        }

        fn with_domain<T>(
            &self,
            call: impl FnOnce(&mut Domain, CuModule) -> Result<T, CudaError>,
        ) -> Result<T, CudaError> {
            let domain = domain()?;
            let mut domain = domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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
            if source.is_empty()
                || uniform_bytes.is_empty()
                || uniform_bytes.len() % 4 != 0
            {
                return Err(CudaError::Refused(Refusal::ShapeMismatch("buffers")));
            }
            let domain = domain()?;
            let mut domain = domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let module = domain.module(self.image)?.module as CuModule;
            let api = domain.api;
            let mut buffers = Buffers {
                api: &api,
                ptrs: Vec::new(),
                pending_launch: false,
            };
            let source_ptr = buffers.alloc(source.len())?;
            let mask_ptr = buffers.alloc(mask.len())?;
            let output_ptr = buffers.alloc(source.len())?;
            self.upload(source_ptr, source)?;
            self.upload(mask_ptr, mask)?;
            let lut_ptr = self.cached_lut(&domain, lut)?;

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
            buffers.pending_launch = true;
            domain.pending = true;
            domain.synchronize_pending()?;
            buffers.pending_launch = false;
            let mut output = vec![0u8; source.len()];
            self.download(output_ptr, &mut output)?;
            Ok((output, dispatch))
        }

        fn cached_lut(&self, domain: &Domain, data: &[u8]) -> Result<CuPtr, CudaError> {
            let mut cache = self.lut_cache.borrow_mut();
            if let Some(hit) = cache.iter().find(|entry| entry.bytes == data) {
                return Ok(hit.ptr);
            }
            if cache.len() == LUT_CACHE_SLOTS {
                let evicted = cache.remove(0);
                unsafe {
                    let _ = (domain.api.mem_free)(evicted.ptr);
                }
            }
            let mut ptr: CuPtr = 0;
            unsafe {
                check(
                    (domain.api.mem_alloc)(&mut ptr, data.len().max(1)),
                    "cuMemAlloc",
                )?;
            }
            #[cfg(test)]
            count_device_allocation();
            if let Err(error) = self.upload(ptr, data) {
                unsafe {
                    let _ = (domain.api.mem_free)(ptr);
                }
                return Err(error);
            }
            SKIN_LUT_UPLOADS.fetch_add(1, Ordering::Relaxed);
            cache.push(CachedLut {
                bytes: data.to_vec(),
                ptr,
            });
            Ok(ptr)
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
            self.run_u8_mode(kernel, grid, inputs, uniform_bytes, None, true)
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
            self.run_u8_mode(kernel, grid, inputs, uniform_bytes, init_output_from, false)
        }

        fn run_u8_mode(
            &self,
            kernel: &'static str,
            grid: (u32, u32),
            inputs: &[&[u8]],
            uniform_bytes: &[u8],
            init_output_from: Option<usize>,
            alias_input0: bool,
        ) -> Result<(Vec<u8>, Dispatch), CudaError> {
            let output_bytes = inputs.first().map_or(0, |first| first.len());
            if output_bytes == 0 || uniform_bytes.is_empty() || uniform_bytes.len() % 4 != 0 {
                return Err(CudaError::Refused(Refusal::ShapeMismatch("buffers")));
            }
            let domain = domain()?;
            let mut domain = domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let module = domain.module(self.image)?.module as CuModule;
            let api = domain.api;
            let mut buffers = Buffers {
                api: &api,
                ptrs: Vec::new(),
                pending_launch: false,
            };
            let mut inputs_on_device: Vec<CuPtr> = Vec::with_capacity(inputs.len());
            for input in inputs {
                inputs_on_device.push(buffers.alloc(input.len())?);
            }
            let output = if alias_input0 {
                *inputs_on_device
                    .first()
                    .ok_or(CudaError::Refused(Refusal::ShapeMismatch("buffers")))?
            } else {
                buffers.alloc(output_bytes)?
            };
            for (ptr, input) in inputs_on_device.iter().zip(inputs) {
                self.upload(*ptr, input)?;
            }
            if let Some(index) = init_output_from {
                let seed = inputs
                    .get(index)
                    .ok_or(CudaError::Refused(Refusal::ShapeMismatch("init_output_from")))?;
                if seed.len() != output_bytes {
                    return Err(CudaError::Refused(Refusal::ShapeMismatch("init_output_from")));
                }
                self.copy_from_device(output, inputs_on_device[index], output_bytes)?;
            }

            let count = inputs_on_device.len();
            let mut params: Vec<*mut c_void> = Vec::with_capacity(count + 2);
            for index in 0..count {
                params.push(&mut inputs_on_device[index] as *mut CuPtr as *mut c_void);
            }
            let mut output_arg = output;
            params.push(&mut output_arg as *mut CuPtr as *mut c_void);
            let mut block = uniform_bytes.to_vec();
            params.push(block.as_mut_ptr() as *mut c_void);

            let dispatch = self.launch_on(&api, module, kernel, grid, BLOCK, &mut params)?;
            buffers.pending_launch = true;
            domain.pending = true;
            domain.synchronize_pending()?;
            let mut bytes = vec![0u8; output_bytes];
            self.download(output, &mut bytes)?;
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
                let dispatch = self.launch_on(&domain.api, module, kernel, geometry, block, params)?;
                domain.pending = true;
                Ok(dispatch)
            })
        }

        fn upload(&self, ptr: CuPtr, data: &[u8]) -> Result<(), CudaError> {
            unsafe { check((self.api.copy_in)(ptr, data.as_ptr() as *const c_void, data.len()), "cuMemcpyHtoD") }
        }

        fn download(&self, ptr: CuPtr, data: &mut [u8]) -> Result<(), CudaError> {
            unsafe {
                check((self.api.copy_out)(data.as_mut_ptr() as *mut c_void, ptr, data.len()), "cuMemcpyDtoH")
            }
        }

        fn copy_from_device(&self, dst: CuPtr, src: CuPtr, bytes: usize) -> Result<(), CudaError> {
            unsafe { check((self.api.copy_device)(dst, src, bytes), "cuMemcpyDtoD") }
        }

        /// Crate-private: allocate a buffer that lives across several launches.
        #[allow(dead_code)] // consumed by `cuda_mesh`
        pub fn alloc_buffer(&self, bytes: usize) -> Result<DeviceBuffer<'_>, CudaError> {
            self.with_domain(|domain, _| {
                let mut ptr: CuPtr = 0;
                unsafe {
                    check((domain.api.mem_alloc)(&mut ptr, bytes), "cuMemAlloc")?;
                }
                #[cfg(test)]
                count_device_allocation();
                Ok(DeviceBuffer {
                    ptr,
                    owner: std::marker::PhantomData,
                })
            })
        }

        #[allow(dead_code)] // consumed by `cuda_mesh`
        pub fn upload_buffer(&self, buffer: &DeviceBuffer<'_>, data: &[u8]) -> Result<(), CudaError> {
            let domain = domain()?;
            let mut domain = domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            domain.bind()?;
            domain.synchronize_pending()?;
            unsafe {
                check(
                    (domain.api.copy_in)(
                        buffer.ptr,
                        data.as_ptr() as *const c_void,
                        data.len(),
                    ),
                    "cuMemcpyHtoD",
                )
            }
        }

        #[allow(dead_code)] // consumed by `cuda_mesh`
        pub fn download_buffer(&self, buffer: &DeviceBuffer<'_>, data: &mut [u8]) -> Result<(), CudaError> {
            let domain = domain()?;
            let mut domain = domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            domain.bind()?;
            domain.synchronize_pending()?;
            unsafe {
                check(
                    (domain.api.copy_out)(
                        data.as_mut_ptr() as *mut c_void,
                        buffer.ptr,
                        data.len(),
                    ),
                    "cuMemcpyDtoH",
                )
            }
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
                    #[cfg(test)]
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
        pub(crate) fn set_upload(&self, set: &DeviceSet<'_>, index: usize, data: &[u8]) -> Result<(), CudaError> {
            let domain = domain()?;
            let mut domain = domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            domain.bind()?;
            domain.synchronize_pending()?;
            match set.buffers.get(index) {
                Some((ptr, size)) if data.len() <= *size => unsafe {
                    check(
                        (domain.api.copy_in)(
                            *ptr,
                            data.as_ptr() as *const c_void,
                            data.len(),
                        ),
                        "cuMemcpyHtoD",
                    )
                },
                _ => Err(CudaError::Refused(Refusal::ShapeMismatch("buffer"))),
            }
        }

        #[allow(dead_code)] // consumed by the O1h session benchmark
        pub(crate) fn set_download(&self, set: &DeviceSet<'_>, index: usize, data: &mut [u8]) -> Result<(), CudaError> {
            let domain = domain()?;
            let mut domain = domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            domain.bind()?;
            domain.synchronize_pending()?;
            match set.buffers.get(index) {
                Some((ptr, size)) if data.len() <= *size => unsafe {
                    check(
                        (domain.api.copy_out)(
                            data.as_mut_ptr() as *mut c_void,
                            *ptr,
                            data.len(),
                        ),
                        "cuMemcpyDtoH",
                    )
                },
                _ => Err(CudaError::Refused(Refusal::ShapeMismatch("buffer"))),
            }
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
            let mut domain = domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let module = domain.module(self.image)?.module as CuModule;
            let [
                device_source,
                device_mask,
                device_skin,
                device_base,
                device_residual,
                device_output,
            ] = domain.composite_buffers(
                uniforms,
                [bytes, bytes, skin_bytes, bytes, bytes, bytes],
            )?;
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
            let mut domain = domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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
            domain.synchronize_pending()?;
            let mut output = vec![0f32; len];
            self.download(device_output, as_bytes_mut(&mut output))?;
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
            let skin: Vec<f32> = skin_mask
                .iter()
                .map(|byte| *byte as f32 / 255.0)
                .collect();
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

    impl Drop for Nvidia {
        fn drop(&mut self) {
            let Ok(domain) = domain() else { return };
            let mut domain = domain.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            if domain.bind().is_err() {
                return;
            }
            let _ = domain.synchronize_pending();
            for lut in self.lut_cache.get_mut().drain(..) {
                unsafe {
                    let _ = (domain.api.mem_free)(lut.ptr);
                }
            }
        }
    }

    fn as_bytes(values: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
    }

    fn as_bytes_mut(values: &mut [f32]) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, values.len() * 4) }
    }

    pub(super) fn metrics() -> DriverMetrics {
        DriverMetrics {
            ptx_module_load_calls: PTX_MODULE_LOADS.load(Ordering::Relaxed),
            skin_lut_uploads: SKIN_LUT_UPLOADS.load(Ordering::Relaxed),
        }
    }
}

#[cfg(windows)]
pub use driver::Nvidia;
#[cfg(windows)]
pub use driver::DeviceBuffer;

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
        assert_eq!(uniforms.validate(), Err(Refusal::ValueOutOfRange("pf_value")));

        let mut uniforms = block(8, 8);
        uniforms.gate = 0.002;
        assert_eq!(uniforms.validate(), Err(Refusal::ValueOutOfRange("pf_value")));

        let mut uniforms = block(8, 8);
        uniforms.algo_type = 1;
        assert_eq!(uniforms.validate(), Err(Refusal::NotImplemented(1)));

        let mut uniforms = block(8, 8);
        uniforms.mix_alpha = f32::NAN;
        assert_eq!(uniforms.validate(), Err(Refusal::ValueOutOfRange("uniforms")));

        let mut uniforms = block(8, 8);
        uniforms.diff_min_threshold = 2.0;
        assert_eq!(uniforms.validate(), Err(Refusal::ValueOutOfRange("uniforms")));
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
        let source: Vec<f32> = noise(pixels * 4, 7).iter().map(|b| *b as f32 / 255.0).collect();
        let base: Vec<f32> = noise(pixels * 4, 11).iter().map(|b| *b as f32 / 255.0).collect();
        let residual: Vec<f32> = noise(pixels * 4, 13).iter().map(|b| *b as f32 / 255.0).collect();
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
                .composite_f32(&source, &mask, &skin_plane.repeat(4), &base, &residual, &uniforms)
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
            let rgba = |values: &[f32]| [values[at], values[at + 1], values[at + 2], values[at + 3]];
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

        let expected =
            cpu::apply(&source, &base, &residual, &mask_r, &mask_b, &skin, samples()).unwrap();
        let (actual, dispatch) = nvidia
            .facial_smooth(&source, &base, &residual, &mask_r, &mask_b, &skin, 90024, 0.5, &samples())
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
        assert_eq!(worst, 0, "{differing} of {} bytes differ, max delta {worst}", expected.pixels.len());
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
        nvidia.clear_composite_buffers().expect("clear composite cache");
        let cases = [
            ("constant_64", 64, 64, true, 6usize),
            ("random_64_same_shape", 64, 64, false, 0usize),
            ("random_37x19_shape_change", 37, 19, false, 6usize),
            ("random_64_return_shape", 64, 64, false, 6usize),
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

            assert_eq!(cold, cold_allocations, "{label}: cold allocations");
            assert_eq!(warm, 0, "{label}: warm allocations");
            assert_eq!(dispatch.kernel, COMPOSITE);
            assert_eq!(first.pixels, expected.pixels, "{label}: CPU byte mismatch");
            assert_eq!(second.pixels, expected.pixels, "{label}: warm CPU byte mismatch");
            assert_eq!(first.pixels, second.pixels, "{label}: reuse changed output");
            eprintln!(
                "{label} {}x{} cold_alloc={cold} warm_alloc={warm} bytes={} exact",
                width,
                height,
                first.pixels.len()
            );
        }
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
        assert_eq!(super::driver::live_contexts(), 1, "all handles share one context");

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
            nvidia.composite_f32(source, mask, skin, &plane, &plane, &uniforms).err()
        };
        assert_eq!(run(&plane, &mask, &skin), None, "clean inputs must dispatch");

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
                nvidia.composite_f32(&plane, &mask, &skin, &inf, &plane, &uniforms).err(),
                Some(CudaError::Refused(Refusal::ValueOutOfRange("pixel")))
            );
        }

        for (slot, value) in [(0usize, 1.001f32), (2, -0.001), (0, f32::NAN)] {
            let mut broken = mask.clone();
            broken[slot] = value;
            assert_eq!(run(&plane, &broken, &skin), Some(CudaError::Refused(Refusal::ValueOutOfRange("mask"))));
        }
        // `.g` and `.a` are not weights the recovered law reads, so junk there
        // must not refuse a job the CPU lane would also accept.
        let mut unused = mask.clone();
        unused[1] = f32::NAN;
        unused[3] = 7.5;
        assert_eq!(run(&plane, &unused, &skin), None);
        let mut broken = skin.clone();
        broken[3] = 1.5;
        assert_eq!(run(&plane, &mask, &broken), Some(CudaError::Refused(Refusal::ValueOutOfRange("mask"))));

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
            let (output, dispatch) = nvidia.run_pass(pass, &source, &uniforms).expect("real GPU dispatch");
            assert_eq!(dispatch.kernel, pass.kernel());
            let worst = output
                .iter()
                .zip(&source)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0f32, f32::max);
            assert!(worst < 1e-6, "{:?} shifted a constant field by {worst}", pass);
        }
    }
}
