//! Bridge from the in-crate GPU runtime vocabulary to the public contracts.
//!
//! The Windows GPU adapters speak `mock::gpu_runtime`'s types: `FrameFormat`,
//! `FrameLayout`, `Launch` (the dispatch record) and `Failure`. The rest of the
//! engine, the job lifecycle and the UI speak `contracts::frame` and
//! `contracts::compute`. This module is the single place that translates
//! between the two, so neither vocabulary has to depend on the other.
//!
//! It never launches a kernel, allocates a buffer or inspects pixels: every
//! function is a pure mapping. That keeps the bridge testable on macOS, where
//! the driver is the `aurora_portrait::cuda` stub, without changing any GPU
//! behaviour.
//!
//! Nothing here is wired into the adapters yet. The runtime still returns its
//! own `Failure`/`Launch` values; a later migration would have the callers
//! convert them at the boundary through these functions.

#![allow(dead_code)] // Used by the bridge tests; adapters will adopt it later.

use aurora_portrait::device::{
    Backend as PortraitBackend, DeviceProbe as PortraitDeviceProbe,
    DeviceRefusal as PortraitDeviceRefusal,
};

use crate::contracts::compute::{
    diagnose_backend_selection, BackendCapability, BackendDiagnostic, BackendKind, BackendPolicy,
    DispatchReceipt, FallbackReason, OperationId, Precision as ContractPrecision, ResourceRequest,
};
use crate::contracts::frame::{
    AlphaMode, ColorSpace, ContractError, CoordinateSpace, FrameDescriptor, PixelFormat,
    PlaneLayout, PlaneSpec, PlanarLayout, ValidRegion,
};
use crate::mock::gpu_runtime::{Failure, FrameFormat, Launch};

/// Operation recorded on a dispatch receipt built from a raw runtime launch.
///
/// `Launch` does not carry a pipeline operation identifier (the stage owner
/// knows it), so the bridge stamps a generic dispatch operation. A caller that
/// knows the real operation overwrites `receipt.operation` afterwards.
pub(crate) const DEFAULT_DISPATCH_OPERATION: OperationId = OperationId::new("gpu.dispatch");

/// Translate a runtime frame format into a checked [`FrameDescriptor`].
///
/// - `F32Rgb12` is the pre-curve display frame: three interleaved `f32`
///   samples per pixel, 12 bytes per pixel, packed (`row_stride = width × 12`),
///   4-byte aligned, linear-light sRGB, in source coordinates, no alpha.
/// - `F32Planes` is the clarity grid plus the two low-frequency planes built
///   from it (`grid + 2 planes`). The runtime lays it out as a *grid* of
///   `gw × gh` samples followed by two planes with their own (downsampled)
///   extents and embedded small block-average cells; `gw`, `gh` and the step
///   are not part of this function's signature. The descriptor therefore
///   models the logical, full-resolution form: three `f32` planes, one sample
///   per pixel, each `width × height`, packed and 4-byte aligned. Mapping the
///   exact resident offsets needs the residency owner's `PlaneKey`/`step`
///   values and is deliberately left as follow-up work; this function never
///   claims byte equality with the live buffer.
///
/// Both paths are validated (dimensions, stride, alignment, plane count,
/// overlap and checked byte length) before the descriptor is returned, so a
/// zero-sized or overflowing request is a typed error rather than a frame that
/// silently computes the wrong size.
pub(crate) fn frame_descriptor(
    format: FrameFormat,
    width: u32,
    height: u32,
) -> Result<FrameDescriptor, ContractError> {
    let (pixel_format, layout, planes) = match format {
        FrameFormat::F32Rgb12 => {
            let row_stride = u64::from(width)
                .checked_mul(12)
                .ok_or(ContractError::Overflow { what: "row_stride" })?;
            (PixelFormat::RgbF32, PlaneLayout::new(0, row_stride, 4), None)
        }
        FrameFormat::F32Planes => {
            let row_stride = u64::from(width)
                .checked_mul(4)
                .ok_or(ContractError::Overflow { what: "row_stride" })?;
            let plane_bytes = row_stride
                .checked_mul(u64::from(height))
                .ok_or(ContractError::Overflow { what: "plane_bytes" })?;
            let mut specs = Vec::with_capacity(3);
            for index in 0..3u64 {
                let offset = plane_bytes
                    .checked_mul(index)
                    .ok_or(ContractError::Overflow { what: "plane_offset" })?;
                specs.push(PlaneSpec::new(offset, row_stride, height));
            }
            (
                PixelFormat::RgbF32,
                // Ignored by the planar byte arithmetic; retained so the field
                // is always present.
                PlaneLayout::new(0, row_stride, 4),
                Some(PlanarLayout::new(specs, 4)),
            )
        }
        FrameFormat::U8Rgba8 => {
            let row_stride = u64::from(width)
                .checked_mul(4)
                .ok_or(ContractError::Overflow { what: "row_stride" })?;
            (PixelFormat::Rgba8, PlaneLayout::new(0, row_stride, 4), None)
        }
        FrameFormat::U16Rgba16 => {
            let row_stride = u64::from(width)
                .checked_mul(8)
                .ok_or(ContractError::Overflow { what: "row_stride" })?;
            (PixelFormat::Rgba16, PlaneLayout::new(0, row_stride, 8), None)
        }
    };

    let descriptor = FrameDescriptor {
        width,
        height,
        format: pixel_format,
        layout,
        planes,
        color_space: ColorSpace::LinearSrgb,
        coordinates: CoordinateSpace::Source,
        alpha_mode: AlphaMode::None,
        valid_region: ValidRegion { x: 0, y: 0, width, height },
    };
    // `byte_len` runs `validate` and forces the checked stride × height
    // arithmetic, so an overflowing frame is rejected here.
    descriptor.byte_len()?;
    Ok(descriptor)
}

/// Map a runtime fallback cause onto the public [`FallbackReason`].
///
/// | `Failure`                 | `FallbackReason`                          |
/// |---------------------------|-------------------------------------------|
/// | `Disabled`                | `Unsupported`                             |
/// | `TooLarge { need, cap }`  | `BudgetExceeded { need, budget: cap }`    |
/// | `Unavailable(what)`       | `Unsupported`                             |
/// | `Driver { .. }`           | `ExecutionError`                          |
/// | `Refused(tag)`            | `MissingInput(tag)`                       |
/// | `DeviceLost(_)`           | `DeviceLost`                              |
/// | `Invalid(_)`              | `Unsupported`                             |
///
/// `Unavailable` is about the environment (no driver, no device or a missing
/// optional entry point), not about one kernel, so it becomes `Unsupported`
/// rather than `MissingKernel`: the named `what` is dropped instead of being
/// misreported as a kernel name. `Invalid` is likewise structurally
/// unsupported and drops its tag. `Refused` tags such as `shape_mismatch` or
/// `not_implemented` are not literally "missing input", but `MissingInput` is
/// the closest contract variant and preserves the tag verbatim.
pub(crate) fn fallback_reason(failure: &Failure) -> FallbackReason {
    match failure {
        Failure::Disabled(_) => FallbackReason::Unsupported,
        Failure::TooLarge { need, cap } => FallbackReason::BudgetExceeded {
            need: *need as u64,
            budget: *cap as u64,
        },
        Failure::Unavailable(_) => FallbackReason::Unsupported,
        Failure::Driver { .. } => FallbackReason::ExecutionError,
        Failure::Refused(tag) => FallbackReason::MissingInput(tag),
        Failure::DeviceLost(_) => FallbackReason::DeviceLost,
        Failure::Invalid(_) => FallbackReason::Unsupported,
    }
}

/// Build a [`DispatchReceipt`] from one runtime launch record.
///
/// Kernel, grid and block are preserved verbatim; the launch happened, so
/// `completed` is `true`. Transfer counters are left at zero because `Launch`
/// does not carry them: the caller adds the bytes it actually moved (the
/// runtime's `Counters::h2d_bytes`/`d2h_bytes` are per stage, not per launch).
///
/// `Launch::count` collapses repeated identical launches into one record and
/// has no `DispatchReceipt` field; emit one receipt per launch *record*, not
/// per collapsed count, or expand to `count` receipts at the call site.
pub(crate) fn dispatch_receipt(launch: &Launch) -> DispatchReceipt {
    dispatch_receipt_for(launch, backend_kind())
}

/// Build a [`DispatchReceipt`] for an explicitly selected [`BackendKind`],
/// allowing CUDA, Metal, Vulkan/WGPU, and CPU dispatches to emit truthful
/// backend attribution across all platforms.
pub(crate) fn dispatch_receipt_for(launch: &Launch, backend: BackendKind) -> DispatchReceipt {
    DispatchReceipt {
        operation: DEFAULT_DISPATCH_OPERATION,
        backend,
        kernel: Some(launch.kernel),
        grid: Some(launch.grid),
        block: Some(launch.block),
        host_to_device_bytes: 0,
        device_to_host_bytes: 0,
        completed: true,
    }
}

/// Translate `aurora_portrait::device::Backend` into the public contract [`BackendKind`].
pub(crate) const fn from_portrait_backend(backend: PortraitBackend) -> BackendKind {
    match backend {
        PortraitBackend::Cpu => BackendKind::Cpu,
        PortraitBackend::Cuda => BackendKind::Cuda,
        PortraitBackend::Metal => BackendKind::Metal,
        PortraitBackend::Vulkan => BackendKind::Vulkan,
        PortraitBackend::Npu => BackendKind::Npu,
    }
}

/// Translate a contract [`BackendKind`] into `aurora_portrait::device::Backend`.
/// Returns `None` for backend families that `aurora-portrait` does not model (e.g. `OpenCl`).
pub(crate) const fn to_portrait_backend(kind: BackendKind) -> Option<PortraitBackend> {
    match kind {
        BackendKind::Cpu => Some(PortraitBackend::Cpu),
        BackendKind::Cuda => Some(PortraitBackend::Cuda),
        BackendKind::Metal => Some(PortraitBackend::Metal),
        BackendKind::Vulkan => Some(PortraitBackend::Vulkan),
        BackendKind::Npu => Some(PortraitBackend::Npu),
        BackendKind::OpenCl => None,
    }
}

/// Map a portrait device selection refusal onto the contract [`FallbackReason`].
pub(crate) fn fallback_from_portrait_refusal(refusal: &PortraitDeviceRefusal) -> FallbackReason {
    match refusal {
        PortraitDeviceRefusal::BackendUnavailable(_) => FallbackReason::Unsupported,
        PortraitDeviceRefusal::KernelMissing => FallbackReason::MissingKernel("portrait_stage"),
        PortraitDeviceRefusal::PrecisionUnsupported(_) => FallbackReason::PrecisionUnsupported,
        PortraitDeviceRefusal::ShapeUnsupported => FallbackReason::MissingInput("shape_unsupported"),
        PortraitDeviceRefusal::SourceUnavailable => FallbackReason::MissingInput("source_pixels"),
        PortraitDeviceRefusal::EffectUnsupported => FallbackReason::Unsupported,
    }
}

/// Convert caller-supplied `aurora_portrait::device::DeviceProbe` slices into
/// contract [`BackendCapability`] entries ordered by cross-platform priority
/// (`Cuda -> Metal -> Vulkan -> Npu -> Cpu`), always appending an available
/// `Cpu` reference capability if none was explicitly provided.
pub(crate) fn capabilities_from_portrait_probes(
    probes: &[PortraitDeviceProbe],
) -> Vec<BackendCapability> {
    let priority = [
        PortraitBackend::Cuda,
        PortraitBackend::Metal,
        PortraitBackend::Vulkan,
        PortraitBackend::Npu,
    ];
    let mut caps = Vec::with_capacity(probes.len() + 1);
    for target in priority {
        for probe in probes.iter().filter(|p| p.backend == target) {
            let precision = match (probe.fp32, probe.fp16) {
                (true, true) => ContractPrecision::Mixed,
                (false, true) => ContractPrecision::F16,
                _ => ContractPrecision::F32,
            };
            caps.push(BackendCapability {
                kind: from_portrait_backend(probe.backend),
                available: probe.is_ready(),
                device: None,
                compute_capability: None,
                precision,
            });
        }
    }
    caps.push(BackendCapability {
        kind: BackendKind::Cpu,
        available: true,
        device: Some("cpu-reference".to_owned()),
        compute_capability: None,
        precision: ContractPrecision::F32,
    });
    caps
}

/// Produce a structured cross-platform [`BackendDiagnostic`] directly from
/// `aurora-portrait` device probes so every platform build truthfully reports
/// its selected backend and any fallback reason.
pub(crate) fn diagnose_portrait_probes(
    policy: BackendPolicy,
    probes: &[PortraitDeviceProbe],
) -> BackendDiagnostic {
    let capabilities = capabilities_from_portrait_probes(probes);
    diagnose_backend_selection(policy, &capabilities)
}

/// The backend family the compiled runtime targets.
///
/// The real driver path (`aurora_portrait::cuda` driving the CUDA runtime) is
/// compiled on Windows only; every other platform builds the stub module and
/// never dispatches to a device. So this is `Cuda` under `cfg(windows)` and
/// `Cpu` elsewhere. It reports what the code can do on this build, not what
/// hardware is installed: a Windows machine with no CUDA device still reports
/// `Cuda`, and an actual fallback is described by [`fallback_reason`].
#[cfg(windows)]
pub(crate) fn backend_kind() -> BackendKind {
    BackendKind::Cuda
}

/// See [`backend_kind`]: non-Windows builds compile the CUDA stub.
#[cfg(not(windows))]
pub(crate) fn backend_kind() -> BackendKind {
    BackendKind::Cpu
}

/// Build a [`ResourceRequest`] for a device-resident frame plus pinned host.
///
/// `frame_bytes` is charged as device-local memory (the runtime's `FrameHandle`
/// lives in VRAM and is leased from the process budget); `pinned_bytes` is the
/// page-locked host staging the caller keeps. No pageable host bytes are
/// implied by this function, so `cpu_bytes` is zero — add a host staging copy
/// at the call site if the pipeline keeps one. Addition stays checked in
/// [`ResourceRequest::total_host_bytes`], not here, because this is only a
/// description.
pub(crate) fn resource_request(frame_bytes: u64, pinned_bytes: u64) -> ResourceRequest {
    ResourceRequest {
        cpu_bytes: 0,
        device_bytes: frame_bytes,
        pinned_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::frame::ContractError::LayoutIsPlanar;

    #[test]
    fn f32_rgb12_maps_to_interleaved_rgb_f32() {
        let descriptor =
            frame_descriptor(FrameFormat::F32Rgb12, 640, 480).expect("valid rgb frame");
        assert!(!descriptor.is_planar());
        assert_eq!(descriptor.format, PixelFormat::RgbF32);
        assert_eq!(descriptor.layout.offset, 0);
        assert_eq!(descriptor.layout.row_stride, 640 * 12);
        assert_eq!(descriptor.layout.alignment, 4);
        assert_eq!(descriptor.color_space, ColorSpace::LinearSrgb);
        assert_eq!(descriptor.coordinates, CoordinateSpace::Source);
        assert_eq!(descriptor.alpha_mode, AlphaMode::None);
        assert_eq!(
            descriptor.valid_region,
            ValidRegion { x: 0, y: 0, width: 640, height: 480 }
        );
        assert_eq!(descriptor.row_bytes(), Ok(640 * 12));
        assert_eq!(
            descriptor.byte_len(),
            Ok(640 * 12 * 480)
        );
    }

    #[test]
    fn f32_planes_maps_to_three_f32_planes() {
        let descriptor =
            frame_descriptor(FrameFormat::F32Planes, 8, 4).expect("valid planar frame");
        assert!(descriptor.is_planar());
        assert_eq!(descriptor.format, PixelFormat::RgbF32);
        let planes = descriptor.planes.as_ref().expect("planar layout");
        assert_eq!(planes.alignment, 4);
        assert_eq!(planes.planes.len(), 3);
        for (index, plane) in planes.planes.iter().enumerate() {
            assert_eq!(plane.row_stride, 8 * 4);
            assert_eq!(plane.height, 4);
            assert_eq!(plane.offset, (index as u64) * 128);
        }
        assert_eq!(planes.byte_len(), Ok(384));
        assert_eq!(descriptor.byte_len(), Ok(384));
        // A planar frame refuses the interleaved-only pitch accessor.
        assert_eq!(
            descriptor.row_bytes(),
            Err(LayoutIsPlanar { what: "row_bytes" })
        );
    }

    #[test]
    fn descriptor_validation_rejects_zero_and_overflow() {
        assert_eq!(
            frame_descriptor(FrameFormat::F32Rgb12, 0, 4),
            Err(ContractError::ZeroDimension { what: "width" })
        );
        assert_eq!(
            frame_descriptor(FrameFormat::F32Rgb12, 4, 0),
            Err(ContractError::ZeroDimension { what: "height" })
        );
        assert_eq!(
            frame_descriptor(FrameFormat::F32Planes, 0, 4),
            Err(ContractError::ZeroDimension { what: "width" })
        );
        // width × height × 12 overflows for the largest dimensions.
        assert!(matches!(
            frame_descriptor(FrameFormat::F32Rgb12, u32::MAX, u32::MAX),
            Err(ContractError::Overflow { .. })
        ));
        assert!(matches!(
            frame_descriptor(FrameFormat::F32Planes, u32::MAX, u32::MAX),
            Err(ContractError::Overflow { .. })
        ));
    }

    #[test]
    fn failures_map_to_every_fallback_reason() {
        assert_eq!(
            fallback_reason(&Failure::Disabled("clarity")),
            FallbackReason::Unsupported
        );
        assert_eq!(
            fallback_reason(&Failure::TooLarge { need: 4096, cap: 2048 }),
            FallbackReason::BudgetExceeded { need: 4096, budget: 2048 }
        );
        assert_eq!(
            fallback_reason(&Failure::Unavailable("nvrtc")),
            FallbackReason::Unsupported
        );
        assert_eq!(
            fallback_reason(&Failure::Driver { call: "cuLaunchKernel", code: 1 }),
            FallbackReason::ExecutionError
        );
        assert_eq!(
            fallback_reason(&Failure::Refused("missing_input")),
            FallbackReason::MissingInput("missing_input")
        );
        assert_eq!(
            fallback_reason(&Failure::DeviceLost(709)),
            FallbackReason::DeviceLost
        );
        assert_eq!(
            fallback_reason(&Failure::Invalid("shape")),
            FallbackReason::Unsupported
        );
    }

    #[test]
    fn dispatch_receipt_preserves_launch_geometry() {
        let launch = Launch {
            kernel: "clarity_planes",
            grid: (16, 8),
            block: (8, 8),
            count: 3,
        };
        let receipt = dispatch_receipt(&launch);
        assert_eq!(receipt.operation, DEFAULT_DISPATCH_OPERATION);
        assert_eq!(receipt.operation.as_str(), "gpu.dispatch");
        assert_eq!(receipt.backend, backend_kind());
        assert_eq!(receipt.kernel, Some("clarity_planes"));
        assert_eq!(receipt.grid, Some((16, 8)));
        assert_eq!(receipt.block, Some((8, 8)));
        assert_eq!(receipt.host_to_device_bytes, 0);
        assert_eq!(receipt.device_to_host_bytes, 0);
        assert!(receipt.completed);
    }

    #[test]
    fn backend_kind_matches_the_compiled_driver() {
        #[cfg(windows)]
        assert_eq!(backend_kind(), BackendKind::Cuda);
        #[cfg(not(windows))]
        assert_eq!(backend_kind(), BackendKind::Cpu);
    }

    #[test]
    fn resource_request_charges_frame_to_device_memory() {
        let request = resource_request(1024, 256);
        assert_eq!(request.device_bytes, 1024);
        assert_eq!(request.pinned_bytes, 256);
        assert_eq!(request.cpu_bytes, 0);
        assert_eq!(request.total_host_bytes(), Ok(256));
    }

    #[test]
    fn fallback_reasons_display_their_cause() {
        let reason = fallback_reason(&Failure::TooLarge { need: 64, cap: 32 });
        assert_eq!(
            reason.to_string(),
            "resource budget exceeded: 64 bytes needed, 32 bytes allowed"
        );
        assert_eq!(
            fallback_reason(&Failure::DeviceLost(709)).to_string(),
            "the device was lost during execution"
        );
    }

    #[test]
    fn portrait_backend_roundtrip_covers_cross_platform_targets() {
        for (portrait, contract) in [
            (PortraitBackend::Cpu, BackendKind::Cpu),
            (PortraitBackend::Cuda, BackendKind::Cuda),
            (PortraitBackend::Metal, BackendKind::Metal),
            (PortraitBackend::Vulkan, BackendKind::Vulkan),
            (PortraitBackend::Npu, BackendKind::Npu),
        ] {
            assert_eq!(from_portrait_backend(portrait), contract);
            assert_eq!(to_portrait_backend(contract), Some(portrait));
        }
        assert_eq!(to_portrait_backend(BackendKind::OpenCl), None);
    }

    #[test]
    fn diagnose_portrait_probes_reports_vulkan_metal_cuda_and_cpu_fallback() {
        let vulkan_ready = PortraitDeviceProbe {
            backend: PortraitBackend::Vulkan,
            driver_ready: true,
            kernel_compiled: true,
            fp32: true,
            fp16: true,
            max_width: 8192,
            max_height: 8192,
        };
        let metal_ready = PortraitDeviceProbe {
            backend: PortraitBackend::Metal,
            driver_ready: true,
            kernel_compiled: true,
            fp32: true,
            fp16: true,
            max_width: 8192,
            max_height: 8192,
        };
        let broken_vulkan = PortraitDeviceProbe {
            driver_ready: false,
            ..vulkan_ready
        };

        // AMD / Intel / Snapdragon Adreno host with Vulkan only.
        let vk_diag = diagnose_portrait_probes(BackendPolicy::Auto, &[vulkan_ready]);
        assert_eq!(vk_diag.selected, Some(BackendKind::Vulkan));
        assert_eq!(vk_diag.fallback, None);
        assert!(vk_diag.summary.contains("selected=vulkan"));

        // Priority ordering: Metal wins over Vulkan when both are ready.
        let metal_diag =
            diagnose_portrait_probes(BackendPolicy::Auto, &[vulkan_ready, metal_ready]);
        assert_eq!(metal_diag.selected, Some(BackendKind::Metal));
        assert_eq!(metal_diag.fallback, None);

        // Broken Vulkan driver degrades cleanly to CPU with explicit fallback reason.
        let fallback_diag = diagnose_portrait_probes(BackendPolicy::PreferGpu, &[broken_vulkan]);
        assert_eq!(fallback_diag.selected, Some(BackendKind::Cpu));
        assert_eq!(fallback_diag.fallback, Some(FallbackReason::Unsupported));
        assert!(fallback_diag.unavailable_backends.contains(&BackendKind::Vulkan));
        assert!(fallback_diag.summary.contains("selected=cpu"));
        assert!(fallback_diag.summary.contains("fallback="));

        // StrictGpu refuses when no GPU is ready.
        let refused_diag = diagnose_portrait_probes(BackendPolicy::StrictGpu, &[broken_vulkan]);
        assert_eq!(refused_diag.selected, None);
        assert_eq!(refused_diag.fallback, Some(FallbackReason::Unsupported));
        assert!(refused_diag.summary.contains("refused="));
    }

    #[test]
    fn dispatch_receipt_for_preserves_explicit_backend() {
        let launch = Launch {
            kernel: "skin_whiten_fused",
            grid: (32, 16),
            block: (16, 16),
            count: 1,
        };
        let vk_receipt = dispatch_receipt_for(&launch, BackendKind::Vulkan);
        assert_eq!(vk_receipt.backend, BackendKind::Vulkan);
        assert_eq!(vk_receipt.kernel, Some("skin_whiten_fused"));
        assert!(vk_receipt.completed);

        assert_eq!(
            fallback_from_portrait_refusal(&PortraitDeviceRefusal::KernelMissing),
            FallbackReason::MissingKernel("portrait_stage")
        );
        assert_eq!(
            fallback_from_portrait_refusal(&PortraitDeviceRefusal::PrecisionUnsupported(
                aurora_portrait::device::Precision::Fp16
            )),
            FallbackReason::PrecisionUnsupported
        );
    }
}
