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
        if self.pf != 90023 && self.pf != 90024 && self.pf != 21024 {
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
        cpu, ptx_isa, restamp_ptx, CudaError, Dispatch, Image, Pass, Refusal, Uniforms, PTX,
    };
    use std::ffi::{c_char, c_void, CString, CStr};
    use std::ptr;
    // Test-only context census, thread-local so parallel tests cannot perturb
    // it. Opens and drops in these tests happen on the same thread.
    #[cfg(test)]
    thread_local! {
        static LIVE_CONTEXTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    #[cfg(test)]
    fn count_context(delta: isize) {
        LIVE_CONTEXTS.with(|live| live.set((live.get() as isize + delta).max(0) as usize));
    }

    #[cfg(test)]
    pub(crate) fn live_contexts() -> usize {
        LIVE_CONTEXTS.with(|live| live.get())
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
    extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    struct Api {
        init: unsafe extern "system" fn(u32) -> CuResult,
        device_get: unsafe extern "system" fn(*mut CuDevice, i32) -> CuResult,
        device_name: unsafe extern "system" fn(*mut c_char, i32, CuDevice) -> CuResult,
        device_attribute: unsafe extern "system" fn(*mut i32, i32, CuDevice) -> CuResult,
        driver_version: unsafe extern "system" fn(*mut i32) -> CuResult,
        ctx_create: unsafe extern "system" fn(*mut CuContext, u32, CuDevice) -> CuResult,
        ctx_destroy: unsafe extern "system" fn(CuContext) -> CuResult,
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

    /// A context, module and device identity. Dropping it releases all three.
    /// One context per process: parallel contexts crashed the driver in tests.
    static GPU_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    pub struct Nvidia {
        /// Held for this context's lifetime; shortcut: one GPU job at a time.
        _job: std::sync::MutexGuard<'static, ()>,
        api: Api,
        ctx: CuContext,
        module: CuModule,
        name: String,
        compute_cap: (i32, i32),
        driver_version: i32,
        ptx_isa: (u32, u32),
        /// Bytes of the image actually loaded, which is the restamped text when
        /// the driver rejected the compiled ISA level.
        image_bytes: usize,
    }

    struct Buffers<'a> {
        api: &'a Api,
        ptrs: Vec<CuPtr>,
    }

    impl Buffers<'_> {
        fn alloc(&mut self, bytes: usize) -> Result<CuPtr, CudaError> {
            let mut ptr: CuPtr = 0;
            unsafe { check((self.api.mem_alloc)(&mut ptr, bytes), "cuMemAlloc")? };
            self.ptrs.push(ptr);
            Ok(ptr)
        }
    }

    impl Drop for Buffers<'_> {
        fn drop(&mut self) {
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
        api: &'a Api,
        ptr: CuPtr,
    }

    impl DeviceBuffer<'_> {
        #[allow(dead_code)] // consumed by `cuda_mesh`
        pub fn handle(&self) -> u64 {
            self.ptr
        }
    }

    impl Drop for DeviceBuffer<'_> {
        fn drop(&mut self) {
            unsafe {
                let _ = (self.api.mem_free)(self.ptr);
            }
        }
    }

    /// A fixed group of device buffers, allocated once and reused across
    /// launches. Crate-private: a session that repeats the same job shape should
    /// not pay `cuMemAlloc` per launch.
    #[allow(dead_code)] // consumed by the O1h session benchmark
    pub(crate) struct DeviceSet<'a> {
        api: &'a Api,
        buffers: Vec<(CuPtr, usize)>,
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
            for (ptr, _) in self.buffers.drain(..) {
                unsafe {
                    let _ = (self.api.mem_free)(ptr);
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
        fn load_module(api: &Api, text: &str) -> Result<CuModule, CudaError> {
            let mut image = text.as_bytes().to_vec();
            image.push(0);
            let mut module: CuModule = ptr::null_mut();
            unsafe {
                check(
                    (api.module_load)(&mut module, image.as_ptr() as *const c_void, 0, ptr::null_mut(), ptr::null_mut()),
                    "cuModuleLoadDataEx",
                )?;
            }
            Ok(module)
        }

        /// Opens device 0 and JIT-loads the recovered PTX image into it.
        pub fn open() -> Result<Self, CudaError> {
            Self::open_image(PTX)
        }

        /// The real opener, taking the image to load. `open` passes the bundled
        /// PTX; tests pass junk to exercise the failure path.
        pub(crate) fn open_image(image: &str) -> Result<Self, CudaError> {
            let job = GPU_LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
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
            // The context exists now, so it is owned before the next fallible
            // call: every `?` below drops this guard and destroys it.
            let mut gpu = Self {
                _job: job,
                api,
                ctx,
                module: ptr::null_mut(),
                name: String::new(),
                compute_cap: (0, 0),
                driver_version: version,
                ptx_isa: (0, 0),
                image_bytes: image.len(),
            };
            // Load the bundled image exactly as compiled. Only the driver saying
            // "unsupported PTX version" licenses a compatibility retry, and the
            // retry is a header rewrite of the same image, never a recompile.
            let (module, isa, loaded) = match Self::load_module(&gpu.api, image) {
                Ok(module) => (module, ptx_isa(image).ok_or(CudaError::Unavailable("ptx image"))?, image.len()),
                Err(CudaError::Driver(_, UNSUPPORTED_PTX_VERSION)) if version >= 4000 => {
                    let target = ((version as u32 / 1000 - 4), (version as u32 % 1000) / 10);
                    let downgraded = restamp_ptx(image, target).ok_or(CudaError::Unavailable("ptx image"))?;
                    let module = Self::load_module(&gpu.api, &downgraded)?;
                    (module, ptx_isa(&downgraded).unwrap_or(target), downgraded.len())
                }
                Err(error) => return Err(error),
            };
            gpu.module = module;
            gpu.ptx_isa = isa;
            gpu.image_bytes = loaded;
            let mut name_buf = [0 as c_char; 256];
            let mut major = 0;
            let mut minor = 0;
            unsafe {
                check((gpu.api.device_name)(name_buf.as_mut_ptr(), name_buf.len() as i32, device), "cuDeviceGetName")?;
                check((gpu.api.device_attribute)(&mut major, ATTR_CC_MAJOR, device), "cuDeviceGetAttribute")?;
                check((gpu.api.device_attribute)(&mut minor, ATTR_CC_MINOR, device), "cuDeviceGetAttribute")?;
            }
            gpu.name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }.to_string_lossy().into_owned();
            gpu.compute_cap = (major, minor);
            Ok(gpu)
        }

        /// Crate-private: load another recovered image through the same guarded
        /// opener and the same 222-only retry.
        #[allow(dead_code)] // consumed by `cuda_lut`, which lib.rs mounts separately
        pub fn open_ptx(image: &str) -> Result<Self, CudaError> {
            Self::open_image(image)
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
            let mut buffers = Buffers { api: &self.api, ptrs: Vec::new() };
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

            let dispatch = self.launch(kernel, grid, &mut params)?;
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

        fn launch(
            &self,
            kernel: &'static str,
            geometry: (u32, u32),
            params: &mut [*mut c_void],
        ) -> Result<Dispatch, CudaError> {
            self.launch_raw(kernel, geometry, BLOCK, params)
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
            let symbol = CString::new(kernel).expect("static kernel name");
            let mut function: CuFunction = ptr::null_mut();
            unsafe {
                check((self.api.module_function)(&mut function, self.module, symbol.as_ptr()), "cuModuleGetFunction")?;
                check(
                    (self.api.launch)(
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
                check((self.api.ctx_sync)(), "cuCtxSynchronize")?;
            }
            Ok(self.dispatch(kernel, geometry, block))
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
            let mut ptr: CuPtr = 0;
            unsafe {
                check((self.api.mem_alloc)(&mut ptr, bytes), "cuMemAlloc")?;
            }
            Ok(DeviceBuffer { api: &self.api, ptr })
        }

        #[allow(dead_code)] // consumed by `cuda_mesh`
        pub fn upload_buffer(&self, buffer: &DeviceBuffer<'_>, data: &[u8]) -> Result<(), CudaError> {
            self.upload(buffer.ptr, data)
        }

        #[allow(dead_code)] // consumed by `cuda_mesh`
        pub fn download_buffer(&self, buffer: &DeviceBuffer<'_>, data: &mut [u8]) -> Result<(), CudaError> {
            self.download(buffer.ptr, data)
        }

        /// Crate-private: allocate a group of buffers that can be reused for
        /// every launch of the same job shape.
        #[allow(dead_code)] // consumed by the O1h session benchmark
        pub(crate) fn alloc_set(&self, sizes: &[usize]) -> Result<DeviceSet<'_>, CudaError> {
            let mut buffers = Vec::with_capacity(sizes.len());
            for size in sizes {
                let mut ptr: CuPtr = 0;
                unsafe {
                    check((self.api.mem_alloc)(&mut ptr, (*size).max(1)), "cuMemAlloc")?;
                }
                buffers.push((ptr, *size));
            }
            Ok(DeviceSet { api: &self.api, buffers })
        }

        #[allow(dead_code)] // consumed by the O1h session benchmark
        pub(crate) fn set_upload(&self, set: &DeviceSet<'_>, index: usize, data: &[u8]) -> Result<(), CudaError> {
            match set.buffers.get(index) {
                Some((ptr, size)) if data.len() <= *size => self.upload(*ptr, data),
                _ => Err(CudaError::Refused(Refusal::ShapeMismatch("buffer"))),
            }
        }

        #[allow(dead_code)] // consumed by the O1h session benchmark
        pub(crate) fn set_download(&self, set: &DeviceSet<'_>, index: usize, data: &mut [u8]) -> Result<(), CudaError> {
            match set.buffers.get(index) {
                Some((ptr, size)) if data.len() <= *size => self.download(*ptr, data),
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
            let mut buffers = Buffers { api: &self.api, ptrs: Vec::new() };
            let device_source = buffers.alloc(bytes)?;
            let device_mask = buffers.alloc(bytes)?;
            let device_skin = buffers.alloc(skin_bytes)?;
            let device_base = buffers.alloc(bytes)?;
            let device_residual = buffers.alloc(bytes)?;
            let device_output = buffers.alloc(bytes)?;
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
            let dispatch = self.launch(super::COMPOSITE, grid(uniforms.width, uniforms.height), &mut params)?;
            let mut output = vec![0f32; len];
            self.download(device_output, as_bytes_mut(&mut output))?;
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
            let mut buffers = Buffers { api: &self.api, ptrs: Vec::new() };
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
            let dispatch = self.launch(pass.kernel(), grid(uniforms.width, uniforms.height), &mut params)?;
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

    fn as_bytes(values: &[f32]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) }
    }

    fn as_bytes_mut(values: &mut [f32]) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, values.len() * 4) }
    }

    impl Drop for Nvidia {
        fn drop(&mut self) {
            unsafe {
                if !self.module.is_null() {
                    let _ = (self.api.module_unload)(self.module);
                }
                if !self.ctx.is_null() {
                    let _ = (self.api.ctx_destroy)(self.ctx);
                }
                #[cfg(test)]
                count_context(-isize::from(!self.ctx.is_null()));
            }
        }
    }
}

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
                if std::env::var_os("PIXCAKE_CUDA_REQUIRED").is_some() {
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
    #[test]
    fn failed_kernel_load_releases_its_context() {
        {
            let Some(_probe) = gpu() else { return };
        }
        assert_eq!(super::driver::live_contexts(), 0, "a dropped Nvidia owns no context");
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
                0,
                "a failed open leaked its context"
            );
        }
        let gpu = Nvidia::open_image(PTX).expect("bundled image still loads");
        assert_eq!(super::driver::live_contexts(), 1);
        drop(gpu);
        assert_eq!(super::driver::live_contexts(), 0);
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
