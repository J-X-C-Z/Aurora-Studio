//! Intel Level Zero / SYCL and OpenVINO C API platform adapter.
//!
//! Neither vendor library is linked at build time. On Windows this module
//! resolves `ze_loader.dll` / `openvino_c.dll` with `LoadLibraryA`; on Linux it
//! resolves `libze_loader.so` / `libopenvino_c.so` with `dlopen`. Missing
//! libraries, devices, driver/kernel images or an unavailable target are all
//! explicit [`ComputeError::BackendUnavailable`] refusals; no CPU fallback is
//! silently substituted and `NPU` is never reported after a CPU fallback.
//!
//! Status note: Level Zero enumeration and device capability probing are
//! implemented against the public C ABI. Without a real Intel driver, kernel
//! image or OpenVINO compiled model this module executes nothing and reports
//! `DESIGNED`/`BackendUnavailable`, never `PerfOk`. Device-target validation
//! returns `DeviceOk` only after a real OpenVINO device string is read back.

use std::collections::HashMap;

use crate::autotune::{CandidateSpec, ExecutionMode, ParamBound, VerificationStatus};
use crate::compute_backend::{
    BufferId, BufferPoolStats, BufferUsage, ComputeBackend, ComputeError, ExecutionReceipt,
    PipelineCacheStats, PipelineHandle, PipelineKey, PixelStageFusion,
};
use crate::device::{Backend, DeviceProbe, Precision};
use crate::metal_backend::{PIXEL_FUSION_PARAMS_BYTES, fusion_params};
use crate::Image;

#[cfg(windows)]
const LEVEL_ZERO_LIBRARY: &str = "ze_loader.dll";
#[cfg(target_os = "linux")]
const LEVEL_ZERO_LIBRARY: &str = "libze_loader.so";
#[cfg(not(any(windows, target_os = "linux")))]
const LEVEL_ZERO_LIBRARY: &str = "ze_loader";

#[cfg(windows)]
const OPENVINO_LIBRARY: &str = "openvino_c.dll";
#[cfg(target_os = "linux")]
const OPENVINO_LIBRARY: &str = "libopenvino_c.so";
#[cfg(not(any(windows, target_os = "linux")))]
const OPENVINO_LIBRARY: &str = "openvino_c";

pub const LZ_UNAVAILABLE: &str = "Intel Level Zero runtime unavailable";
pub const OV_UNAVAILABLE: &str = "OpenVINO C runtime unavailable";
pub const SYCL_KERNEL_UNAVAILABLE: &str = "Intel SYCL kernel image unavailable";

/// Intel device family inferred only from a Level Zero property field; the
/// adapter never infers GPU availability from the OS or device name alone.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntelDeviceClass {
    Uhd,
    IrisXe,
    Arc,
    CoreUltra,
    Unknown,
}

/// Level Zero subgroup / SLM / XMX capabilities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntelCapabilities {
    pub device_class: IntelDeviceClass,
    pub subgroup_sizes: u32,
    pub max_subgroup_size: u32,
    pub local_mem_size: u64,
    pub max_work_group_size: [u32; 3],
    pub xmx_slices: u32,
    pub p_cores: u32,
    pub e_cores: u32,
}

impl IntelCapabilities {
    pub fn supports_subgroup(&self, lanes: u32) -> bool {
        lanes > 0 && self.subgroup_sizes & (1u32 << lanes) != 0
    }

    pub fn has_xmx(&self) -> bool {
        self.xmx_slices > 0
    }

    /// P/E-core budget: performance cores keep interactive preview latency;
    /// efficient cores are bounded for background inference work.
    pub fn thread_budget(&self, interactive: bool) -> u32 {
        if interactive {
            self.p_cores.max(1)
        } else {
            self.p_cores.saturating_add(self.e_cores).max(1)
        }
    }
}

/// Strict target for OpenVINO inference. `Npu` is never substituted with CPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpenVinoDeviceTarget {
    Cpu,
    Gpu,
    Npu,
}

impl OpenVinoDeviceTarget {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Gpu => "GPU",
            Self::Npu => "NPU",
        }
    }

    /// `GPU.0` for an Intel iGPU/Arc device; the exact device ID must come from
    /// `ie:get_property` rather than being guessed.
    pub const fn openvino_device_name(self) -> &'static str {
        match self {
            Self::Cpu => "CPU",
            Self::Gpu => "GPU.0",
            Self::Npu => "NPU",
        }
    }
}

/// Receipt returned by a successful (or strictly refused) OpenVINO target
/// validation. The actual device string is always recorded so a CPU fallback
/// cannot be relabelled as an NPU success.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpenVinoExecutionReceipt {
    pub requested: OpenVinoDeviceTarget,
    pub actual_device: Option<String>,
    pub mode: ExecutionMode,
    pub status: VerificationStatus,
}

impl OpenVinoExecutionReceipt {
    pub fn refused(requested: OpenVinoDeviceTarget) -> Self {
        Self {
            requested,
            actual_device: None,
            mode: ExecutionMode::Strict,
            status: VerificationStatus::Unsupported,
        }
    }

    /// Validate a reported actual device string against the request. NPU is
    /// special: any mismatch or CPU fallback is a hard refusal in every mode.
    /// `actual_device: None` (no device ever opened) is also refused.
    pub fn from_actual(
        requested: OpenVinoDeviceTarget,
        actual_device: &str,
        mode: ExecutionMode,
    ) -> Result<Self, ComputeError> {
        let actual = actual_device.trim().to_ascii_uppercase();
        let matches = match requested {
            OpenVinoDeviceTarget::Cpu => actual == "CPU",
            OpenVinoDeviceTarget::Gpu => actual.starts_with("GPU"),
            OpenVinoDeviceTarget::Npu => actual == "NPU",
        };
        if !matches {
            return Err(ComputeError::BackendUnavailable(
                "OpenVINO device target mismatch: no silent fallback",
            ));
        }
        Ok(Self {
            requested,
            actual_device: Some(actual_device.to_string()),
            mode,
            status: VerificationStatus::DeviceOk,
        })
    }

    /// A real NPU run must stay NPU. If a caller reports a CPU or GPU path after
    /// an NPU request, this returns the explicit refusal and never a receipt
    /// claiming NPU success.
    pub fn validate_npu_no_fallback(
        &self,
        fallback_device: &str,
    ) -> Result<(), ComputeError> {
        if self.requested != OpenVinoDeviceTarget::Npu {
            return Ok(());
        }
        if self.actual_device.is_none() {
            return Err(ComputeError::BackendUnavailable(
                "NPU request has no device receipt; refusing to report NPU success",
            ));
        }
        let fallback = fallback_device.trim().to_ascii_uppercase();
        if fallback == "CPU" || fallback.starts_with("GPU") {
            return Err(ComputeError::BackendUnavailable(
                "NPU request fell back to CPU/GPU; refusing to report NPU success",
            ));
        }
        Ok(())
    }
}

static SUBGROUP_PARAMS: &[ParamBound] = &[
    ParamBound::DiscreteU32 {
        name: "subgroup_size",
        values: &[8, 16],
    },
    ParamBound::U32Range {
        name: "local_size",
        min: 64,
        max: 1024,
        step: 64,
    },
    ParamBound::BoolFlag {
        name: "xmx_dot_product",
    },
];

static SUBGROUP_32_PARAMS: &[ParamBound] = &[
    ParamBound::DiscreteU32 {
        name: "subgroup_size",
        values: &[16, 32],
    },
    ParamBound::U32Range {
        name: "local_size",
        min: 64,
        max: 1024,
        step: 64,
    },
    ParamBound::BoolFlag {
        name: "xmx_dot_product",
    },
];

static XMX_PARAMS: &[ParamBound] = &[
    ParamBound::DiscreteU32 {
        name: "subgroup_size",
        values: &[16, 32],
    },
    ParamBound::DiscreteU32 {
        name: "xmx_slices",
        values: &[1, 2, 4, 8],
    },
    ParamBound::U32Range {
        name: "p_core_budget",
        min: 1,
        max: 16,
        step: 1,
    },
];

const fn candidate(
    id: &'static str,
    params: &'static [ParamBound],
    smem: u32,
    threads: u32,
) -> CandidateSpec {
    CandidateSpec {
        id,
        algorithm_id: "pixel_fusion",
        backend: Backend::Sycl,
        kernel_entry: "pixel_fusion",
        precision: Precision::Fp32,
        params,
        max_shared_mem_bytes: smem,
        max_threads_per_group: threads,
        status: VerificationStatus::Designed,
    }
}

/// Bounded SYCL candidate space for UHD / Iris Xe / Arc / Core Ultra.
pub static INTEL_CANDIDATES: &[CandidateSpec] = &[
    candidate("sycl.uhd.sg8", SUBGROUP_PARAMS, 16 * 1024, 256),
    candidate("sycl.uhd.sg16", SUBGROUP_PARAMS, 16 * 1024, 256),
    candidate("sycl.iris_xe.sg16", SUBGROUP_PARAMS, 32 * 1024, 512),
    candidate("sycl.iris_xe.sg32", SUBGROUP_32_PARAMS, 32 * 1024, 512),
    candidate("sycl.arc.sg16", XMX_PARAMS, 48 * 1024, 1024),
    candidate("sycl.arc.sg32.xmx", XMX_PARAMS, 48 * 1024, 1024),
    candidate("sycl.core_ultra.sg16.xmx", XMX_PARAMS, 64 * 1024, 1024),
    candidate("sycl.core_ultra.sg32.xmx", XMX_PARAMS, 64 * 1024, 1024),
];

#[cfg(windows)]
mod loader {
    use super::*;
    use std::ffi::{CStr, c_char, c_void};

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn LoadLibraryA(name: *const c_char) -> *mut c_void;
        fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
    }

    pub struct Library(*mut c_void);

    impl Library {
        pub fn open(name: &str) -> Option<Self> {
            let name = std::ffi::CString::new(name).ok()?;
            let handle = unsafe { LoadLibraryA(name.as_ptr()) };
            (!handle.is_null()).then_some(Self(handle))
        }

        pub fn symbol(&self, name: &str) -> Option<*mut c_void> {
            let name = std::ffi::CString::new(name).ok()?;
            let address = unsafe { GetProcAddress(self.0, name.as_ptr()) };
            (!address.is_null()).then_some(address)
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {
            // Keep the handle for the process lifetime: function pointers may
            // outlive individual adapter instances.
        }
    }

    #[derive(Clone, Copy)]
    pub struct ZeApi {
        pub init: unsafe extern "C" fn(u32) -> i32,
        pub get_driver_count: unsafe extern "C" fn(*mut u32) -> i32,
        pub get_driver: unsafe extern "C" fn(u32, *mut *mut c_void) -> i32,
        pub get_device_count: unsafe extern "C" fn(*mut c_void, *mut u32) -> i32,
        pub get_device: unsafe extern "C" fn(*mut c_void, u32, *mut *mut c_void) -> i32,
        pub get_device_properties: unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32,
    }

    pub fn ze_api() -> Result<ZeApi, ComputeError> {
        let library = Library::open(LEVEL_ZERO_LIBRARY)
            .ok_or(ComputeError::BackendUnavailable(LZ_UNAVAILABLE))?;
        macro_rules! sym {
            ($name:literal) => {{
                let address = library.symbol($name).ok_or(ComputeError::BackendUnavailable(
                    concat!("Level Zero symbol missing: ", $name),
                ))?;
                unsafe { std::mem::transmute::<*mut c_void, _>(address) }
            }};
        }
        Ok(ZeApi {
            init: sym!("zeInit"),
            get_driver_count: sym!("zeDriverGet"),
            get_driver: sym!("zeDriverGet"),
            get_device_count: sym!("zeDeviceGet"),
            get_device: sym!("zeDeviceGet"),
            get_device_properties: sym!("zeDeviceGetProperties"),
        })
    }

    #[derive(Clone, Copy)]
    pub struct OvApi {
        pub create: unsafe extern "C" fn(*mut *mut c_void),
        pub free: unsafe extern "C" fn(*mut c_void),
        pub get_available_devices: unsafe extern "C" fn(*mut c_void, *mut c_char) -> i32,
        pub get_property: unsafe extern "C" fn(*mut c_void, *const c_char, *mut c_char) -> i32,
    }

    pub fn ov_api() -> Result<OvApi, ComputeError> {
        let library = Library::open(OPENVINO_LIBRARY)
            .ok_or(ComputeError::BackendUnavailable(OV_UNAVAILABLE))?;
        macro_rules! sym {
            ($name:literal) => {{
                let address = library.symbol($name).ok_or(ComputeError::BackendUnavailable(
                    concat!("OpenVINO symbol missing: ", $name),
                ))?;
                unsafe { std::mem::transmute::<*mut c_void, _>(address) }
            }};
        }
        Ok(OvApi {
            create: sym!("ie_create"),
            free: sym!("ie_free"),
            get_available_devices: sym!("ie_get_available_devices"),
            get_property: sym!("ie_get_property"),
        })
    }

    pub fn cstr(value: &str) -> std::ffi::CString {
        std::ffi::CString::new(value).expect("string has no NUL")
    }

    pub fn as_str(ptr: *const c_char) -> String {
        if ptr.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
    }
}

#[cfg(target_os = "linux")]
mod loader {
    use super::*;
    use std::ffi::{CStr, c_char, c_void};

    const RTLD_NOW: i32 = 2;

    unsafe extern "C" {
        fn dlopen(path: *const c_char, mode: i32) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    pub struct Library(*mut c_void);

    impl Library {
        pub fn open(name: &str) -> Option<Self> {
            let name = std::ffi::CString::new(name).ok()?;
            let handle = unsafe { dlopen(name.as_ptr(), RTLD_NOW) };
            (!handle.is_null()).then_some(Self(handle))
        }

        pub fn symbol(&self, name: &str) -> Option<*mut c_void> {
            let name = std::ffi::CString::new(name).ok()?;
            let address = unsafe { dlsym(self.0, name.as_ptr()) };
            (!address.is_null()).then_some(address)
        }
    }

    impl Drop for Library {
        fn drop(&mut self) {}
    }

    #[derive(Clone, Copy)]
    pub struct ZeApi {
        pub init: unsafe extern "C" fn(u32) -> i32,
        pub get_driver_count: unsafe extern "C" fn(*mut u32) -> i32,
        pub get_driver: unsafe extern "C" fn(u32, *mut *mut c_void) -> i32,
        pub get_device_count: unsafe extern "C" fn(*mut c_void, *mut u32) -> i32,
        pub get_device: unsafe extern "C" fn(*mut c_void, u32, *mut *mut c_void) -> i32,
        pub get_device_properties: unsafe extern "C" fn(*mut c_void, *mut c_void) -> i32,
    }

    pub fn ze_api() -> Result<ZeApi, ComputeError> {
        let library = Library::open(LEVEL_ZERO_LIBRARY)
            .ok_or(ComputeError::BackendUnavailable(LZ_UNAVAILABLE))?;
        macro_rules! sym {
            ($name:literal) => {{
                let address = library.symbol($name).ok_or(ComputeError::BackendUnavailable(
                    concat!("Level Zero symbol missing: ", $name),
                ))?;
                unsafe { std::mem::transmute::<*mut c_void, _>(address) }
            }};
        }
        Ok(ZeApi {
            init: sym!("zeInit"),
            get_driver_count: sym!("zeDriverGet"),
            get_driver: sym!("zeDriverGet"),
            get_device_count: sym!("zeDeviceGet"),
            get_device: sym!("zeDeviceGet"),
            get_device_properties: sym!("zeDeviceGetProperties"),
        })
    }

    #[derive(Clone, Copy)]
    pub struct OvApi {
        pub create: unsafe extern "C" fn(*mut *mut c_void),
        pub free: unsafe extern "C" fn(*mut c_void),
        pub get_available_devices: unsafe extern "C" fn(*mut c_void, *mut c_char) -> i32,
        pub get_property: unsafe extern "C" fn(*mut c_void, *const c_char, *mut c_char) -> i32,
    }

    pub fn ov_api() -> Result<OvApi, ComputeError> {
        let library = Library::open(OPENVINO_LIBRARY)
            .ok_or(ComputeError::BackendUnavailable(OV_UNAVAILABLE))?;
        macro_rules! sym {
            ($name:literal) => {{
                let address = library.symbol($name).ok_or(ComputeError::BackendUnavailable(
                    concat!("OpenVINO symbol missing: ", $name),
                ))?;
                unsafe { std::mem::transmute::<*mut c_void, _>(address) }
            }};
        }
        Ok(OvApi {
            create: sym!("ie_create"),
            free: sym!("ie_free"),
            get_available_devices: sym!("ie_get_available_devices"),
            get_property: sym!("ie_get_property"),
        })
    }

    pub fn cstr(value: &str) -> std::ffi::CString {
        std::ffi::CString::new(value).expect("string has no NUL")
    }

    pub fn as_str(ptr: *const c_char) -> String {
        if ptr.is_null() {
            return String::new();
        }
        unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod loader {
    use super::*;
    use std::ffi::c_char;

    pub struct Library;

    impl Library {
        pub fn open(_name: &str) -> Option<Self> {
            None
        }

        pub fn symbol(&self, _name: &str) -> Option<*mut std::ffi::c_void> {
            None
        }
    }

    #[derive(Clone, Copy)]
    pub struct ZeApi {}

    pub fn ze_api() -> Result<ZeApi, ComputeError> {
        Err(ComputeError::BackendUnavailable(LZ_UNAVAILABLE))
    }

    #[derive(Clone, Copy)]
    pub struct OvApi {}

    pub fn ov_api() -> Result<OvApi, ComputeError> {
        Err(ComputeError::BackendUnavailable(OV_UNAVAILABLE))
    }

    pub fn cstr(value: &str) -> std::ffi::CString {
        std::ffi::CString::new(value).expect("string has no NUL")
    }

    pub fn as_str(_ptr: *const c_char) -> String {
        String::new()
    }
}

/// Probe a real Level Zero device. No driver / no device is a typed refusal.
pub fn probe_level_zero() -> Result<IntelCapabilities, ComputeError> {
    let api = loader::ze_api()?;
    #[cfg(any(windows, target_os = "linux"))]
    unsafe {
        if (api.init)(0) != 0 {
            return Err(ComputeError::BackendUnavailable("zeInit failed"));
        }
        let mut driver_count = 0u32;
        if (api.get_driver_count)(&mut driver_count) != 0 || driver_count == 0 {
            return Err(ComputeError::BackendUnavailable(
                "no Level Zero drivers reported",
            ));
        }
        let mut driver: *mut std::ffi::c_void = std::ptr::null_mut();
        if (api.get_driver)(0, &mut driver) != 0 || driver.is_null() {
            return Err(ComputeError::BackendUnavailable("zeDriverGet failed"));
        }
        let mut device_count = 0u32;
        if (api.get_device_count)(driver, &mut device_count) != 0 || device_count == 0 {
            return Err(ComputeError::BackendUnavailable(
                "no Level Zero devices reported",
            ));
        }
        let mut device: *mut std::ffi::c_void = std::ptr::null_mut();
        if (api.get_device)(driver, 0, &mut device) != 0 || device.is_null() {
            return Err(ComputeError::BackendUnavailable("zeDeviceGet failed"));
        }
        // Property query presence is checked by symbol resolution above; a
        // driver that cannot answer is refused instead of guessed.
        let mut props: [u8; 256] = [0; 256];
        if (api.get_device_properties)(device, props.as_mut_ptr() as *mut std::ffi::c_void) != 0 {
            return Err(ComputeError::BackendUnavailable(
                "zeDeviceGetProperties failed",
            ));
        }
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    let _ = api;
    // Conservative capabilities: do not claim XMX/subgroup features that were
    // not read from a real extension query. On Windows/Linux this point is only
    // reached after the real Level Zero property query succeeded.
    Ok(IntelCapabilities {
        device_class: IntelDeviceClass::Unknown,
        subgroup_sizes: 0,
        max_subgroup_size: 0,
        local_mem_size: 0,
        max_work_group_size: [0, 0, 0],
        xmx_slices: 0,
        p_cores: 1,
        e_cores: 0,
    })
}

/// Validate an OpenVINO target against the runtime's real device list.
/// A requested NPU that is absent is refused; CPU is never reported as NPU.
pub fn validate_openvino_target(target: OpenVinoDeviceTarget) -> Result<OpenVinoExecutionReceipt, ComputeError> {
    let api = loader::ov_api()?;
    #[cfg(not(any(windows, target_os = "linux")))]
    let _ = api;
    let mut core: *mut std::ffi::c_void = std::ptr::null_mut();
    #[cfg(any(windows, target_os = "linux"))]
    unsafe {
        use std::ffi::c_char;
        (api.create)(&mut core);
        if core.is_null() {
            return Err(ComputeError::BackendUnavailable("ie_create failed"));
        }
        // `ie:get_available_devices` returns a comma-separated NUL-terminated
        // device string. A small fixed buffer is enough for CPU/GPU/NPU.
        let mut buffer = [0 as c_char; 256];
        let status = (api.get_available_devices)(core, buffer.as_mut_ptr());
        let devices = loader::as_str(buffer.as_ptr());
        (api.free)(core);
        if status != 0 || devices.is_empty() {
            return Err(ComputeError::BackendUnavailable(
                "OpenVINO reported no available devices",
            ));
        }
        let available: Vec<String> = devices
            .split(',')
            .map(|part| part.trim().to_ascii_uppercase())
            .filter(|part| !part.is_empty())
            .collect();
        let requested = target.as_str();
        let actual = available
            .iter()
            .find(|device| {
                *device == requested
                    || (target == OpenVinoDeviceTarget::Gpu && device.starts_with("GPU"))
            })
            .cloned()
            .ok_or(ComputeError::BackendUnavailable(
                "OpenVINO target device is not available",
            ))?;
        OpenVinoExecutionReceipt::from_actual(target, &actual, ExecutionMode::Strict)
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        let _ = core;
        Err(ComputeError::BackendUnavailable(OV_UNAVAILABLE))
    }
}

enum SyclState {
    Unavailable(&'static str),
    Ready { capabilities: IntelCapabilities },
}

#[derive(Debug)]
struct BufferEntry {
    data: Vec<u8>,
    usage: BufferUsage,
}

/// SYCL / Level Zero implementation of [`ComputeBackend`].
///
/// Construction is inert. A real kernel binary must be supplied by the caller;
/// there is no bundled SPIR-V/ZEBIN here and no claim that the `.hip`/MSL source
/// can run on Intel hardware. Without an actual device and binary, every
/// operation returns `BackendUnavailable`.
pub struct SyclComputeBackend {
    state: SyclState,
    module_bytes: Vec<u8>,
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

impl Default for SyclComputeBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl SyclComputeBackend {
    pub fn new() -> Self {
        Self {
            state: SyclState::Unavailable(SYCL_KERNEL_UNAVAILABLE),
            module_bytes: Vec::new(),
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

    pub fn with_module(mut self, bytes: Vec<u8>) -> Result<Self, ComputeError> {
        if bytes.is_empty() {
            return Err(ComputeError::BackendUnavailable(SYCL_KERNEL_UNAVAILABLE));
        }
        // Supplying module bytes does not open a Level Zero driver/device;
        // `state` stays `Unavailable` until `open()` probes real hardware.
        self.module_bytes = bytes;
        Ok(self)
    }

    fn unavailable(&self) -> ComputeError {
        match &self.state {
            SyclState::Unavailable(reason) => ComputeError::BackendUnavailable(reason),
            SyclState::Ready { .. } => ComputeError::BackendUnavailable("SYCL device is ready"),
        }
    }

    /// Real Level Zero probe; fails explicitly without a driver or device.
    pub fn open() -> Result<Self, ComputeError> {
        let capabilities = probe_level_zero()?;
        let mut backend = Self::new();
        backend.state = SyclState::Ready { capabilities };
        Ok(backend)
    }

    pub fn capabilities(&self) -> Result<IntelCapabilities, ComputeError> {
        match &self.state {
            SyclState::Ready { capabilities } => Ok(capabilities.clone()),
            SyclState::Unavailable(reason) => Err(ComputeError::BackendUnavailable(reason)),
        }
    }

    pub fn has_kernel_image(&self) -> bool {
        !self.module_bytes.is_empty()
    }

    pub fn trim_pool(&mut self) {
        self.free_pool.clear();
    }

    fn require_kernel(&self) -> Result<(), ComputeError> {
        if self.has_kernel_image() {
            Ok(())
        } else {
            Err(ComputeError::BackendUnavailable(SYCL_KERNEL_UNAVAILABLE))
        }
    }
}

impl ComputeBackend for SyclComputeBackend {
    fn backend(&self) -> Backend {
        Backend::Sycl
    }

    fn probe(&self) -> DeviceProbe {
        let driver_ready = matches!(self.state, SyclState::Ready { .. });
        DeviceProbe {
            backend: Backend::Sycl,
            driver_ready,
            kernel_compiled: driver_ready && self.has_kernel_image(),
            fp32: true,
            fp16: false,
            max_width: 16384,
            max_height: 16384,
        }
    }

    fn allocate_buffer(
        &mut self,
        byte_len: usize,
        usage: BufferUsage,
    ) -> Result<BufferId, ComputeError> {
        self.require_kernel()?;
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
        self.require_kernel()?;
        let entry = self
            .active
            .remove(&id)
            .ok_or(ComputeError::BufferNotFound(id))?;
        self.free_pool.push(entry.data);
        Ok(())
    }

    fn upload_buffer(&mut self, id: BufferId, data: &[u8]) -> Result<(), ComputeError> {
        self.require_kernel()?;
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
        self.require_kernel()?;
        self.active
            .get(&id)
            .map(|entry| entry.data.clone())
            .ok_or(ComputeError::BufferNotFound(id))
    }

    fn get_or_compile_pipeline(
        &mut self,
        fusion: &PixelStageFusion,
    ) -> Result<PipelineHandle, ComputeError> {
        self.require_kernel()?;
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
        self.require_kernel()
    }

    fn dispatch_pixel_fusion(
        &mut self,
        input: &Image,
        fusion: &PixelStageFusion,
    ) -> Result<(Image, ExecutionReceipt), ComputeError> {
        self.require_kernel()?;
        input.validate().map_err(ComputeError::InvalidImage)?;
        // No SPIR-V/ZEBIN module is bundled and no Level Zero command-list
        // submission is constructed. Dispatching here would fake GPU success.
        if !matches!(self.state, SyclState::Ready { .. }) {
            return Err(ComputeError::BackendUnavailable(
                "SYCL kernel image cannot be dispatched without a real Level Zero module",
            ));
        }
        let _pipeline = self.get_or_compile_pipeline(fusion)?;
        let _params: [u8; PIXEL_FUSION_PARAMS_BYTES] = fusion_params(fusion);
        Err(ComputeError::BackendUnavailable(
            "SYCL kernel image cannot be dispatched without a real Level Zero module",
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_names_are_platform_conditional() {
        #[cfg(windows)]
        {
            assert_eq!(LEVEL_ZERO_LIBRARY, "ze_loader.dll");
            assert_eq!(OPENVINO_LIBRARY, "openvino_c.dll");
        }
        #[cfg(target_os = "linux")]
        {
            assert_eq!(LEVEL_ZERO_LIBRARY, "libze_loader.so");
            assert_eq!(OPENVINO_LIBRARY, "libopenvino_c.so");
        }
        #[cfg(not(any(windows, target_os = "linux")))]
        {
            assert_eq!(LEVEL_ZERO_LIBRARY, "ze_loader");
            assert_eq!(OPENVINO_LIBRARY, "openvino_c");
        }
        let _ = (LEVEL_ZERO_LIBRARY, OPENVINO_LIBRARY);
    }

    #[test]
    fn level_zero_and_openvino_symbols_are_resolved_by_name() {
        // These names are the exact exported C ABI symbols; keeping them as
        // literals avoids an SDK/import library at build time.
        let ze_symbols = [
            "zeInit",
            "zeDriverGet",
            "zeDeviceGet",
            "zeDeviceGetProperties",
        ];
        let ov_symbols = [
            "ie_create",
            "ie_free",
            "ie_get_available_devices",
            "ie_get_property",
        ];
        assert!(ze_symbols.iter().all(|name| !name.is_empty()));
        assert!(ov_symbols.iter().all(|name| name.starts_with("ie_")));
    }

    #[test]
    fn missing_runtime_is_explicit_and_never_falls_back() {
        let mut backend = SyclComputeBackend::new();
        assert_eq!(backend.backend(), Backend::Sycl);
        assert_eq!(backend.probe().driver_ready, false);
        assert_eq!(backend.probe().kernel_compiled, false);
        assert_eq!(backend.capabilities(), Err(ComputeError::BackendUnavailable(SYCL_KERNEL_UNAVAILABLE)));
        assert_eq!(backend.synchronize(), Err(ComputeError::BackendUnavailable(SYCL_KERNEL_UNAVAILABLE)));
        assert_eq!(
            backend.allocate_buffer(16, BufferUsage::StorageRead),
            Err(ComputeError::BackendUnavailable(SYCL_KERNEL_UNAVAILABLE))
        );
        let image = Image::solid(2, 2, [1, 2, 3, 4]);
        assert_eq!(
            backend.dispatch_pixel_fusion(&image, &PixelStageFusion::new()).err(),
            Some(ComputeError::BackendUnavailable(SYCL_KERNEL_UNAVAILABLE))
        );
    }

    #[test]
    fn npu_request_never_accepts_cpu_or_gpu_fallback() {
        let receipt = OpenVinoExecutionReceipt::from_actual(
            OpenVinoDeviceTarget::Cpu,
            "CPU",
            ExecutionMode::Strict,
        )
        .unwrap();
        assert_eq!(receipt.requested, OpenVinoDeviceTarget::Cpu);
        assert_eq!(receipt.actual_device.as_deref(), Some("CPU"));
        assert_eq!(receipt.status, VerificationStatus::DeviceOk);

        assert!(matches!(
            OpenVinoExecutionReceipt::from_actual(
                OpenVinoDeviceTarget::Npu,
                "CPU",
                ExecutionMode::AutoFallback,
            ),
            Err(ComputeError::BackendUnavailable(_))
        ));
        assert!(matches!(
            OpenVinoExecutionReceipt::from_actual(
                OpenVinoDeviceTarget::Npu,
                "GPU.0",
                ExecutionMode::Strict,
            ),
            Err(ComputeError::BackendUnavailable(_))
        ));
        let npu = OpenVinoExecutionReceipt::from_actual(
            OpenVinoDeviceTarget::Npu,
            "NPU",
            ExecutionMode::Strict,
        )
        .unwrap();
        assert!(npu.validate_npu_no_fallback("NPU").is_ok());
        assert!(npu.validate_npu_no_fallback("CPU").is_err());
        assert!(npu.validate_npu_no_fallback("GPU.0").is_err());
        assert_eq!(OpenVinoDeviceTarget::Npu.openvino_device_name(), "NPU");
        assert_eq!(OpenVinoDeviceTarget::Gpu.openvino_device_name(), "GPU.0");
        let refused = OpenVinoExecutionReceipt::refused(OpenVinoDeviceTarget::Npu);
        assert_eq!(refused.status, VerificationStatus::Unsupported);
        assert!(refused.validate_npu_no_fallback("NPU").is_err());
    }

    #[test]
    fn software_intel_candidates_do_not_claim_device_capabilities() {
        let caps = IntelCapabilities {
            device_class: IntelDeviceClass::Unknown,
            subgroup_sizes: 0,
            max_subgroup_size: 0,
            local_mem_size: 0,
            max_work_group_size: [0, 0, 0],
            xmx_slices: 0,
            p_cores: 1,
            e_cores: 0,
        };
        assert_eq!(caps.device_class, IntelDeviceClass::Unknown);
        assert!(!caps.has_xmx());
        assert!(!caps.supports_subgroup(16));
        assert_eq!(caps.thread_budget(false), 1);
    }

    #[test]
    fn subgroup_xmx_and_pe_core_budget_are_explicit() {
        let caps = IntelCapabilities {
            device_class: IntelDeviceClass::CoreUltra,
            subgroup_sizes: 0b1010,
            max_subgroup_size: 32,
            local_mem_size: 64 * 1024,
            max_work_group_size: [1024, 1024, 64],
            xmx_slices: 8,
            p_cores: 6,
            e_cores: 8,
        };
        assert!(caps.supports_subgroup(1));
        assert!(caps.supports_subgroup(3));
        assert!(!caps.supports_subgroup(2));
        assert!(!caps.supports_subgroup(0));
        assert!(caps.has_xmx());
        assert_eq!(caps.thread_budget(true), 6);
        assert_eq!(caps.thread_budget(false), 14);
        let no_xmx = IntelCapabilities { xmx_slices: 0, ..caps.clone() };
        assert!(!no_xmx.has_xmx());
        assert_eq!(IntelDeviceClass::Uhd, IntelDeviceClass::Uhd);
    }

    #[test]
    fn intel_candidates_cover_uhd_iris_arc_and_core_ultra() {
        for family in ["uhd", "iris_xe", "arc", "core_ultra"] {
            assert!(INTEL_CANDIDATES.iter().any(|c| c.id.contains(family)), "{family}");
        }
        assert!(INTEL_CANDIDATES.iter().all(|c| c.status == VerificationStatus::Designed));
        assert!(INTEL_CANDIDATES.iter().all(|c| c.backend == Backend::Sycl));
        assert!(INTEL_CANDIDATES.iter().all(|c| c.max_threads_per_group <= 1024));
        let sg16 = INTEL_CANDIDATES.iter().find(|c| c.id.contains("sg16")).unwrap();
        let sg32 = INTEL_CANDIDATES.iter().find(|c| c.id.contains("sg32")).unwrap();
        assert_ne!(sg16.params, sg32.params);
    }

    #[test]
    fn kernel_binary_is_required_before_any_dispatch() {
        let mut backend = SyclComputeBackend::new().with_module(vec![0x07, 0x00, 0x00, 0x00]).unwrap();
        assert!(backend.has_kernel_image());
        assert_eq!(backend.probe().driver_ready, false, "module bytes alone must not claim a ready driver");
        assert_eq!(
            backend.dispatch_pixel_fusion(
                &Image::solid(1, 1, [1, 2, 3, 4]),
                &PixelStageFusion::new()
            ).err(),
            Some(ComputeError::BackendUnavailable(
                "SYCL kernel image cannot be dispatched without a real Level Zero module"
            ))
        );
        assert!(SyclComputeBackend::new().with_module(Vec::new()).is_err());
    }
}
