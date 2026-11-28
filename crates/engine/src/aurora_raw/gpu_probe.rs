//! Real device availability probes and backend resolution for the RAW warmup.
//!
//! `AURORA_GPU_BACKEND` is the same contract the native inference sessions use
//! (`auto|cuda|metal|dx12|opencl|cpu`, see `native_infer::backend`). The startup
//! warmup has to obey it: warming the wrong device either pays a JIT this
//! process will never use, or opens CUDA on a host the operator explicitly
//! asked to keep on the DX12 / Metal / OpenCL / CPU path.
//!
//! Two rules this module exists to enforce:
//!
//! 1. A `cfg!` only says which probe *can exist on this target*. It never
//!    decides that a device is present — every backend in the route is probed,
//!    and an NVIDIA box that lost its driver resolves exactly like an AMD one.
//! 2. An explicit request is executed, not pre-judged. `AURORA_GPU_BACKEND=cuda`
//!    hands the request to the CUDA warmer even when no device answers, so the
//!    failure is reported by the thing that really tried instead of by a probe
//!    that guessed. `auto` is the only mode that consults the probes first.
//!
//! The probes load nothing that is JIT-compiled: CUDA is judged by the driver's
//! device memory query, OpenCL by `clGetDeviceIDs`, DX12 by the DXGI adapter
//! enumeration. No PTX, MSL or HLSL is compiled here — that is the warmup's
//! own job, and only after the backend is resolved.

pub use aurora_portrait::native_infer::backend::{BackendKind,BackendRequest,Host};

/// The backend the warmup will run, plus how it was decided.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Resolved {
    pub kind: BackendKind,
    /// The evidence behind an `auto` decision (`None` for an explicit request,
    /// which is executed without asking a probe first).
    pub evidence: Option<String>,
    /// Backends the `auto` walk rejected, in order, with their reasons.
    pub rejected: Vec<(BackendKind, String)>,
}

/// Resolves `request` with an injectable probe, so the routing table is testable
/// without a GPU. An explicit request never calls `probe`.
pub fn resolve_with(
    request: BackendRequest,
    host: Host,
    mut probe: impl FnMut(BackendKind) -> Result<String, String>,
) -> Result<Resolved, String> {
    match request {
        BackendRequest::Only(kind) => Ok(Resolved {
            kind,
            evidence: None,
            rejected: Vec::new(),
        }),
        BackendRequest::Auto => {
            let mut rejected = Vec::new();
            for &kind in host.auto_order() {
                match probe(kind) {
                    Ok(evidence) => {
                        return Ok(Resolved {
                            kind,
                            evidence: Some(evidence),
                            rejected,
                        });
                    }
                    Err(reason) => rejected.push((kind, reason)),
                }
            }
            // `probe(Cpu)` answers on every host, so this is a broken probe
            // rather than a real state: report it instead of pretending.
            Err("no backend answered, not even the CPU reference path".to_string())
        }
    }
}

/// [`resolve_with`] over the real probes.
pub fn resolve(request: BackendRequest, host: Host) -> Result<Resolved, String> {
    resolve_with(request, host, probe)
}

/// Probes one backend and returns a one-line evidence string on success.
///
/// `cpu` is always available: it is the reference path the decode chain already
/// falls back to, so it needs no probe and has no device to warm.
pub fn probe(kind: BackendKind) -> Result<String, String> {
    match kind {
        BackendKind::Cpu => Ok("CPU reference path".to_string()),
        BackendKind::Cuda => probe_cuda(),
        BackendKind::Dx12 => probe_dx12(),
        BackendKind::OpenCl => probe_opencl(),
        BackendKind::Metal => probe_metal(),
    }
}

#[cfg(windows)]
fn probe_cuda() -> Result<String, String> {
    // The driver answers with the device's memory at this instant; there is no
    // cheaper call that still proves a *device* (not just a DLL) is present.
    // Nothing is JIT-compiled here, so a probe can never warm a module.
    aurora_portrait::cuda::driver_memory_info()
        .map(|info| {
            format!(
                "CUDA device, {} MiB total",
                info.total_bytes / (1024 * 1024)
            )
        })
        .map_err(|error| format!("CUDA driver: {error:?}"))
}

#[cfg(not(windows))]
fn probe_cuda() -> Result<String, String> {
    Err("CUDA runtime is not compiled into this target".to_string())
}

#[cfg(windows)]
fn probe_metal() -> Result<String, String> {
    Err("Metal is not available on Windows".to_string())
}

#[cfg(target_os = "macos")]
fn probe_metal() -> Result<String, String> {
    aurora_portrait::metal::available().then(|| "Metal default hardware device".to_string())
        .ok_or_else(|| "Metal framework loaded no default device".to_string())
}

#[cfg(all(not(windows), not(target_os = "macos")))]
fn probe_metal() -> Result<String, String> {
    Err("Metal is not available on this target".to_string())
}

#[cfg(windows)]
mod system32 {
    use std::ffi::c_void;

    pub(super) const LOAD_LIBRARY_SEARCH_SYSTEM32: u32 = 0x800;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        pub(super) fn LoadLibraryExW(name: *const u16, file: *mut c_void, flags: u32) -> *mut c_void;
        /// Declared exactly as the other resolvers in this crate declare it, so
        /// the loader's signature cannot drift between modules.
        pub(super) fn GetProcAddress(module: *mut c_void, name: *const u8) -> *mut c_void;
        pub(super) fn FreeLibrary(module: *mut c_void) -> i32;
    }

    /// Loads `name` from System32 only, so a stray copy next to the executable
    /// can never answer for a system component.
    pub(super) fn load(name: &str) -> Option<*mut c_void> {
        let mut wide: Vec<u16> = name.encode_utf16().collect();
        wide.push(0);
        let handle = unsafe {
            LoadLibraryExW(wide.as_ptr(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32)
        };
        (!handle.is_null()).then_some(handle)
    }

    pub(super) fn symbol(module: *mut c_void, name: &str) -> Option<*const c_void> {
        let mut text: Vec<u8> = name.as_bytes().to_vec();
        text.push(0);
        let address = unsafe { GetProcAddress(module, text.as_ptr()) };
        (!address.is_null()).then_some(address as *const c_void)
    }
}

#[cfg(windows)]
fn probe_opencl() -> Result<String, String> {
    use std::ffi::c_void;
    use system32::{FreeLibrary, load, symbol};

    type ClGetPlatformIds = unsafe extern "system" fn(u32, *mut *mut c_void, *mut u32) -> i32;
    type ClGetDeviceIds = unsafe extern "system" fn(*mut c_void, u64, u32, *mut *mut c_void, *mut u32) -> i32;

    const CL_DEVICE_TYPE_ALL: u64 = 0xFFFF_FFFF;

    let library = load("OpenCL.dll").ok_or_else(|| "System32 OpenCL.dll did not load".to_string())?;
    let result = (|| {
        let platforms =
            symbol(library, "clGetPlatformIDs").ok_or_else(|| "OpenCL.dll has no clGetPlatformIDs".to_string())?;
        let devices =
            symbol(library, "clGetDeviceIDs").ok_or_else(|| "OpenCL.dll has no clGetDeviceIDs".to_string())?;
        // SAFETY: both entry points were resolved from the system loader and are
        // called with the exact signatures the ICD loader publishes.
        let platforms: ClGetPlatformIds = unsafe { std::mem::transmute(platforms) };
        let devices: ClGetDeviceIds = unsafe { std::mem::transmute(devices) };
        let mut platform_count = 0u32;
        if unsafe { platforms(0, std::ptr::null_mut(), &mut platform_count) } != 0 || platform_count == 0 {
            return Err("OpenCL: no platform".to_string());
        }
        let mut handles = vec![std::ptr::null_mut(); platform_count as usize];
        if unsafe { platforms(platform_count, handles.as_mut_ptr(), std::ptr::null_mut()) } != 0 {
            return Err("OpenCL: platform list failed".to_string());
        }
        let mut total = 0u32;
        for platform in handles {
            let mut count = 0u32;
            if unsafe { devices(platform, CL_DEVICE_TYPE_ALL, 0, std::ptr::null_mut(), &mut count) } == 0 {
                total += count;
            }
        }
        if total == 0 {
            return Err("OpenCL: no device".to_string());
        }
        Ok(format!("{total} OpenCL device(s) over {platform_count} platform(s)"))
    })();
    unsafe { FreeLibrary(library) };
    result
}

#[cfg(not(windows))]
fn probe_opencl() -> Result<String, String> {
    Err("the system OpenCL loader is a Windows-only path in this build".to_string())
}

#[cfg(windows)]
fn probe_dx12() -> Result<String, String> {
    use aurora_portrait::dx12::{probe_adapter, AdapterRequest};
    let a=probe_adapter(&AdapterRequest::from_env()).map_err(|e|e.to_string())?;
    Ok(format!("DXGI adapter #{} {} ({:04x}, UMA={})", a.index, a.name, a.vendor_id, a.uma))
}

#[cfg(not(windows))]
fn probe_dx12() -> Result<String, String> {
    Err("DX12 is a Windows-only backend".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_probe(fail: &'static [BackendKind]) -> impl FnMut(BackendKind) -> Result<String, String> {
        move |kind| {
            if fail.contains(&kind) {
                Err(format!("{} unavailable in the fake", kind.label()))
            } else {
                Ok(format!("{} answered", kind.label()))
            }
        }
    }

    /// All six env values round-trip, and a typo fails instead of silently
    /// becoming `auto` (which would then probe CUDA on a CPU-only host).
    #[test]
    fn every_env_spelling_round_trips() {
        for spelling in ["auto", "cuda", "metal", "dx12", "opencl", "cpu"] {
            assert_eq!(BackendRequest::parse(spelling).unwrap().label(), spelling);
        }
        assert_eq!(BackendRequest::parse(" DX12 ").unwrap(), BackendRequest::Only(BackendKind::Dx12));
        assert!(BackendRequest::parse("gpu").is_err());
        assert!(BackendRequest::parse("").is_err());
        let message = BackendRequest::parse("gpu").unwrap_err();
        assert!(message.contains("expected auto, cuda, metal, dx12, opencl or cpu"), "{message}");
    }

    /// An explicit request is executed even when the probe would have said no:
    /// the warmer that really tries is the one that reports.
    #[test]
    fn explicit_request_skips_the_probe() {
        let mut probed = Vec::new();
        let resolved = resolve_with(BackendRequest::Only(BackendKind::Cuda), Host::Windows, |kind| {
            probed.push(kind);
            Err("no device".to_string())
        })
        .unwrap();
        assert_eq!(resolved.kind, BackendKind::Cuda);
        assert!(resolved.evidence.is_none());
        assert!(probed.is_empty(), "an explicit request must not consult the routing table");
    }

    /// Windows auto order is CUDA -> DX12 -> OpenCL -> CPU, and it records why
    /// each earlier backend was rejected instead of hiding it.
    #[test]
    fn auto_walks_the_windows_order_and_records_rejections() {
        let resolved =
            resolve_with(BackendRequest::Auto, Host::Windows, fake_probe(&[BackendKind::Cuda, BackendKind::Dx12])).unwrap();
        assert_eq!(resolved.kind, BackendKind::OpenCl);
        assert_eq!(resolved.rejected.len(), 2);
        assert_eq!(resolved.rejected[0].0, BackendKind::Cuda);
        assert_eq!(resolved.rejected[1].0, BackendKind::Dx12);
        assert_eq!(resolved.evidence.as_deref(), Some("opencl answered"));
    }

    /// macOS never probes the Windows driver stack: a macOS host with no Metal
    /// device must settle on CPU without asking for nvcuda/d3d12/OpenCL.
    #[test]
    fn mac_route_never_asks_for_windows_drivers() {
        let mut probed = Vec::new();
        let resolved = resolve_with(BackendRequest::Auto, Host::MacOs, |kind| {
            probed.push(kind);
            if kind == BackendKind::Cpu {
                Ok("cpu answered".to_string())
            } else {
                Err("no Metal device".to_string())
            }
        })
        .unwrap();
        assert_eq!(probed, vec![BackendKind::Metal, BackendKind::Cpu]);
        assert_eq!(resolved.kind, BackendKind::Cpu);
        assert_eq!(resolved.rejected[0].0, BackendKind::Metal);
    }

    /// A backend `cfg`'d out of this build must report that, not claim a device.
    #[test]
    fn uncompiled_backends_report_a_reason_instead_of_evidence() {
        let kind = if cfg!(windows) { BackendKind::Metal } else { BackendKind::Cuda };
        let reason = probe(kind).unwrap_err();
        assert!(!reason.is_empty(), "a missing backend needs a reason");
        assert_eq!(probe(BackendKind::Cpu).unwrap(), "CPU reference path");
    }

    /// On Windows the probe must answer with either evidence or a reason; it
    /// must never panic on a host without the driver, and the DLL searches are
    /// System32-only.
    #[cfg(windows)]
    #[test]
    fn windows_probes_answer_without_touching_the_decode_path() {
        for kind in [BackendKind::Cuda, BackendKind::Dx12, BackendKind::OpenCl] {
            match probe(kind) {
                Ok(evidence) => {
                    assert!(!evidence.is_empty(), "{kind}: empty evidence");
                    eprintln!("{kind} probe: {evidence}");
                }
                Err(reason) => {
                    assert!(!reason.is_empty(), "{kind}: empty reason");
                    eprintln!("{kind} probe rejected this host: {reason}");
                }
            }
        }
    }

    /// The DXGI path is real: this host has adapters, and the enumeration must
    /// count them without leaking or panicking.
    #[cfg(windows)]
    #[test]
    fn dxgi_enumeration_reports_adapters_on_this_host() {
        match probe(BackendKind::Dx12) {
            Ok(evidence) => {
                assert!(evidence.contains("DXGI adapter"), "{evidence}");
                eprintln!("dx12 probe: {evidence}");
            }
            Err(reason) => eprintln!("dx12 probe rejected this host: {reason}"),
        }
    }
}
