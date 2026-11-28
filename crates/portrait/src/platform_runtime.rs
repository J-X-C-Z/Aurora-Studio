//! Unified cross-platform runtime probe, AutoTune candidate aggregator, and
//! production `PixelStageFusion` router across CUDA, WGPU (Vulkan/DX12/Metal),
//! Apple Metal, AMD HIP, Intel SYCL/Level Zero, CPU SIMD, and scalar CPU.

use std::time::Instant;

use crate::autotune::{AdapterProbeRecord, BackendTiming, CandidateSpec, ExecutionMode, VerificationStatus};
use crate::compute_backend::{
    ComputeBackend, ComputeError, CpuComputeBackend, ExecutionReceipt, PixelStageFusion,
};
use crate::cuda_backend::{self, CudaComputeBackend, FusionLaunchConfig};
use crate::device::{Backend, DeviceProbe, AUTO_GPU_PRIORITY};
use crate::hip_backend::{HipComputeBackend, HIP_CANDIDATES};
use crate::metal_backend::{MetalComputeBackend, APPLE_CANDIDATES};
use crate::simd_ops::{execute_fused_simd, SimdCapabilities};
use crate::sycl_openvino_backend::{probe_level_zero, INTEL_CANDIDATES};
#[cfg(feature = "wgpu-backend")]
use crate::wgpu_compute::{self as wgpu_backend, WgpuBackendSelector, WgpuComputeBackend};
use crate::Image;

const fn host_platform_label() -> &'static str {
    if cfg!(target_os = "windows") {
        "windows"
    } else if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "ios") {
        "ios"
    } else if cfg!(target_os = "android") {
        "android"
    } else if cfg!(target_os = "linux") {
        "linux"
    } else {
        "unknown"
    }
}

/// Probes all compute backends known to this crate without fabricating any
/// hardware capability or runtime availability.
pub fn probe_all_backends() -> Vec<AdapterProbeRecord> {
    let platform = host_platform_label();
    let mut records = Vec::new();

    // 1. NVIDIA CUDA (PTX + Driver API).
    let cuda_ready = cuda_backend::device_available();
    records.push(AdapterProbeRecord {
        backend: Backend::Cuda,
        platform,
        device_name: if cuda_ready {
            "NVIDIA CUDA Device 0".to_string()
        } else {
            "none".to_string()
        },
        driver_version: if cuda_ready {
            "cuda-driver-api".to_string()
        } else {
            "unavailable".to_string()
        },
        api_version: "PTX".to_string(),
        driver_ready: cuda_ready,
        supports_fp32: true,
        supports_fp16: cuda_ready,
        supports_int8: cuda_ready,
        supports_subgroup: cuda_ready,
        max_workgroup_size: [1024, 1024, 64],
        max_shared_mem_bytes: 48 * 1024,
        vram_bytes: 0,
        status: if cuda_ready {
            VerificationStatus::DeviceOk
        } else {
            VerificationStatus::Unsupported
        },
        notes: if cuda_ready {
            "nvcc-compiled pixel_fusion.ptx & native_infer_ops.ptx ready".to_string()
        } else {
            "CUDA driver or PTX module unavailable on host".to_string()
        },
    });

    // 2. WGPU (Vulkan and DX12 / Metal physical selectors).
    #[cfg(feature = "wgpu-backend")]
    {
        for selector in [
            WgpuBackendSelector::Vulkan,
            WgpuBackendSelector::Dx12,
            WgpuBackendSelector::Metal,
        ] {
            match WgpuComputeBackend::new(selector) {
                Ok(backend) => records.push(backend.probe_record()),
                Err(err) => records.push(AdapterProbeRecord {
                    backend: selector.device_backend(),
                    platform,
                    device_name: "none".to_string(),
                    driver_version: "unavailable".to_string(),
                    api_version: format!("wgpu-{}", selector.as_str()),
                    driver_ready: false,
                    supports_fp32: false,
                    supports_fp16: false,
                    supports_int8: false,
                    supports_subgroup: false,
                    max_workgroup_size: [0, 0, 0],
                    max_shared_mem_bytes: 0,
                    vram_bytes: 0,
                    status: VerificationStatus::Unsupported,
                    notes: format!("{err}"),
                }),
            }
        }
    }

    // 3. Native Apple Metal / CoreML.
    let metal_ok = MetalComputeBackend::open().is_ok();
    records.push(AdapterProbeRecord {
        backend: Backend::Metal,
        platform,
        device_name: if metal_ok {
            "Apple Metal Device".to_string()
        } else {
            "none".to_string()
        },
        driver_version: if metal_ok {
            "Metal.framework".to_string()
        } else {
            "unavailable".to_string()
        },
        api_version: "MSL".to_string(),
        driver_ready: metal_ok,
        supports_fp32: metal_ok,
        supports_fp16: metal_ok,
        supports_int8: false,
        supports_subgroup: metal_ok,
        max_workgroup_size: if metal_ok { [1024, 1024, 64] } else { [0, 0, 0] },
        max_shared_mem_bytes: if metal_ok { 32 * 1024 } else { 0 },
        vram_bytes: 0,
        status: if metal_ok {
            VerificationStatus::DeviceOk
        } else {
            VerificationStatus::Unsupported
        },
        notes: if metal_ok {
            "Metal runtime opened via Objective-C bridge".to_string()
        } else {
            "Requires macOS/iOS Metal.framework".to_string()
        },
    });

    // 4. AMD HIP / ROCm runtime probe.
    let hip_err = HipComputeBackend::new().open().err();
    records.push(AdapterProbeRecord {
        backend: Backend::Hip,
        platform,
        device_name: "none".to_string(),
        driver_version: "unavailable".to_string(),
        api_version: "HIP".to_string(),
        driver_ready: false,
        supports_fp32: false,
        supports_fp16: false,
        supports_int8: false,
        supports_subgroup: false,
        max_workgroup_size: [0, 0, 0],
        max_shared_mem_bytes: 0,
        vram_bytes: 0,
        status: VerificationStatus::Designed,
        notes: hip_err
            .map(|e| format!("{e}"))
            .unwrap_or_else(|| "HIP runtime available without compiled HSACO/fatbin".to_string()),
    });

    // 5. Intel SYCL / Level Zero runtime probe.
    let lz_res = probe_level_zero();
    records.push(AdapterProbeRecord {
        backend: Backend::Sycl,
        platform,
        device_name: match &lz_res {
            Ok(caps) => format!("Intel Level Zero ({:?})", caps.device_class),
            Err(_) => "none".to_string(),
        },
        driver_version: if lz_res.is_ok() {
            "ze_loader".to_string()
        } else {
            "unavailable".to_string()
        },
        api_version: "LevelZero/SYCL".to_string(),
        driver_ready: lz_res.is_ok(),
        supports_fp32: lz_res.is_ok(),
        supports_fp16: false,
        supports_int8: false,
        supports_subgroup: lz_res.as_ref().is_ok_and(|c| c.subgroup_sizes != 0),
        max_workgroup_size: lz_res
            .as_ref()
            .map(|c| c.max_work_group_size)
            .unwrap_or([0, 0, 0]),
        max_shared_mem_bytes: lz_res
            .as_ref()
            .map(|c| c.local_mem_size.min(u32::MAX as u64) as u32)
            .unwrap_or(0),
        vram_bytes: 0,
        status: VerificationStatus::Designed,
        notes: match lz_res {
            Ok(_) => "Level Zero device detected; SPIR-V/ZEBIN kernel image required".to_string(),
            Err(e) => format!("{e}"),
        },
    });

    // 6. CPU (SIMD + scalar reference, always available).
    let simd = SimdCapabilities::detect();
    let simd_summary = format!(
        "sse4.1={} avx2={} fma={} neon={}",
        simd.sse4_1, simd.avx2, simd.fma, simd.neon
    );
    records.push(AdapterProbeRecord {
        backend: Backend::Cpu,
        platform,
        device_name: std::env::consts::ARCH.to_string(),
        driver_version: "native".to_string(),
        api_version: simd_summary.clone(),
        driver_ready: true,
        supports_fp32: true,
        supports_fp16: false,
        supports_int8: false,
        supports_subgroup: false,
        max_workgroup_size: [1, 1, 1],
        max_shared_mem_bytes: 0,
        vram_bytes: 0,
        status: VerificationStatus::ParityOk,
        notes: format!("CPU SIMD capabilities: {simd_summary}"),
    });

    records
}

/// Converts live host probes into `crate::device::DeviceProbe` slices for
/// explicit backend selection (`crate::device::select`).
pub fn device_probes_from_host() -> Vec<DeviceProbe> {
    let mut probes = Vec::new();
    let cuda_ok = cuda_backend::device_available();
    probes.push(DeviceProbe {
        backend: Backend::Cuda,
        driver_ready: cuda_ok,
        kernel_compiled: cuda_ok,
        fp32: true,
        fp16: cuda_ok,
        max_width: 32768,
        max_height: 32768,
    });

    #[cfg(feature = "wgpu-backend")]
    {
        for selector in [WgpuBackendSelector::Vulkan, WgpuBackendSelector::Dx12] {
            let ok = WgpuComputeBackend::new(selector).is_ok();
            probes.push(DeviceProbe {
                backend: selector.device_backend(),
                driver_ready: ok,
                kernel_compiled: ok,
                fp32: ok,
                fp16: false,
                max_width: if ok { 16384 } else { 0 },
                max_height: if ok { 16384 } else { 0 },
            });
        }
    }

    let metal_ok = MetalComputeBackend::open().is_ok();
    probes.push(DeviceProbe {
        backend: Backend::Metal,
        driver_ready: metal_ok,
        kernel_compiled: metal_ok,
        fp32: metal_ok,
        fp16: metal_ok,
        max_width: if metal_ok { 16384 } else { 0 },
        max_height: if metal_ok { 16384 } else { 0 },
    });

    probes
}

/// Aggregates every AutoTune `CandidateSpec` registered across all platform
/// backends in `aurora-portrait`.
pub fn all_autotune_candidates() -> Vec<CandidateSpec> {
    let mut specs = Vec::new();
    specs.push(cuda_backend::candidate_spec());
    #[cfg(feature = "wgpu-backend")]
    specs.extend_from_slice(wgpu_backend::candidates());
    specs.extend_from_slice(SimdCapabilities::candidates());
    specs.extend_from_slice(APPLE_CANDIDATES);
    specs.extend_from_slice(HIP_CANDIDATES);
    specs.extend_from_slice(INTEL_CANDIDATES);

    // Android QNN/Vulkan and HarmonyOS candidates are deferred (not ported).
    specs
}

/// Outcome returned by [`FusionRouter::dispatch_pixel_fusion`].
#[derive(Clone, Debug)]
pub struct RoutedFusionOutcome {
    pub image: Image,
    pub receipt: ExecutionReceipt,
    pub timing: BackendTiming,
    pub fallback_reason: Option<String>,
}

/// Stateful cross-platform router that caches live backend contexts and
/// dispatches [`PixelStageFusion`] under strict or auto-fallback policies.
pub struct FusionRouter {
    cuda: Option<CudaComputeBackend>,
    cuda_probed: bool,
    #[cfg(feature = "wgpu-backend")]
    wgpu_vulkan: Option<WgpuComputeBackend>,
    #[cfg(feature = "wgpu-backend")]
    wgpu_vulkan_probed: bool,
    #[cfg(feature = "wgpu-backend")]
    wgpu_dx12: Option<WgpuComputeBackend>,
    #[cfg(feature = "wgpu-backend")]
    wgpu_dx12_probed: bool,
    cpu: CpuComputeBackend,
}

impl Default for FusionRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionRouter {
    pub fn new() -> Self {
        Self {
            cuda: None,
            cuda_probed: false,
            #[cfg(feature = "wgpu-backend")]
            wgpu_vulkan: None,
            #[cfg(feature = "wgpu-backend")]
            wgpu_vulkan_probed: false,
            #[cfg(feature = "wgpu-backend")]
            wgpu_dx12: None,
            #[cfg(feature = "wgpu-backend")]
            wgpu_dx12_probed: false,
            cpu: CpuComputeBackend::new(),
        }
    }

    fn ensure_cuda(&mut self) -> Option<&mut CudaComputeBackend> {
        if !self.cuda_probed {
            self.cuda_probed = true;
            if cuda_backend::device_available() {
                self.cuda = Some(CudaComputeBackend::with_mode(
                    FusionLaunchConfig::default(),
                    ExecutionMode::Strict,
                ));
            }
        }
        self.cuda.as_mut()
    }

    #[cfg(feature = "wgpu-backend")]
    fn ensure_wgpu_vulkan(&mut self) -> Result<&mut WgpuComputeBackend, ComputeError> {
        if !self.wgpu_vulkan_probed {
            self.wgpu_vulkan_probed = true;
            if let Ok(backend) = WgpuComputeBackend::new(WgpuBackendSelector::Vulkan) {
                self.wgpu_vulkan = Some(backend);
            }
        }
        self.wgpu_vulkan
            .as_mut()
            .ok_or(ComputeError::BackendUnavailable("WGPU Vulkan adapter unavailable"))
    }

    #[cfg(feature = "wgpu-backend")]
    fn ensure_wgpu_dx12(&mut self) -> Result<&mut WgpuComputeBackend, ComputeError> {
        if !self.wgpu_dx12_probed {
            self.wgpu_dx12_probed = true;
            if let Ok(backend) = WgpuComputeBackend::new(WgpuBackendSelector::Dx12) {
                self.wgpu_dx12 = Some(backend);
            }
        }
        self.wgpu_dx12
            .as_mut()
            .ok_or(ComputeError::BackendUnavailable("WGPU DX12 adapter unavailable"))
    }

    fn dispatch_cpu_simd(
        &mut self,
        image: &Image,
        fusion: &PixelStageFusion,
        fallback_reason: Option<String>,
    ) -> Result<RoutedFusionOutcome, ComputeError> {
        image.validate().map_err(ComputeError::InvalidImage)?;
        let started = Instant::now();
        let pipeline = self.cpu.get_or_compile_pipeline(fusion)?;
        let mut pixels = vec![0u8; image.pixels.len()];
        execute_fused_simd(fusion, &image.pixels, &mut pixels)?;
        let wall_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        let out = Image {
            width: image.width,
            height: image.height,
            pixels,
        };
        let bytes = out.pixels.len() as u64;
        Ok(RoutedFusionOutcome {
            image: out,
            receipt: ExecutionReceipt {
                backend: Backend::Cpu,
                pipeline,
                fused_stages: fusion.len(),
                pixel_count: image.pixel_count(),
                memory_passes: 1,
                bytes_read: bytes,
                bytes_written: bytes,
            },
            timing: BackendTiming {
                queue_submit_ns: 0,
                device_exec_ns: wall_ns,
                wall_ns,
                h2d_bytes: 0,
                d2h_bytes: 0,
            },
            fallback_reason,
        })
    }

    fn dispatch_specific(
        &mut self,
        backend: Backend,
        image: &Image,
        fusion: &PixelStageFusion,
    ) -> Result<RoutedFusionOutcome, ComputeError> {
        let started = Instant::now();
        match backend {
            Backend::Cuda => {
                let cuda = self.ensure_cuda().ok_or(ComputeError::BackendUnavailable(
                    "CUDA device or PTX unavailable",
                ))?;
                let (out, receipt) = cuda.dispatch_pixel_fusion(image, fusion)?;
                let wall_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                Ok(RoutedFusionOutcome {
                    image: out,
                    receipt,
                    timing: BackendTiming {
                        wall_ns,
                        h2d_bytes: image.pixels.len() as u64,
                        d2h_bytes: image.pixels.len() as u64,
                        ..BackendTiming::default()
                    },
                    fallback_reason: None,
                })
            }
            #[cfg(feature = "wgpu-backend")]
            Backend::Vulkan => {
                let wgpu = self.ensure_wgpu_vulkan()?;
                let (out, receipt) = wgpu.dispatch_pixel_fusion(image, fusion)?;
                let timing = wgpu.last_timing().unwrap_or(BackendTiming {
                    wall_ns: started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                    ..BackendTiming::default()
                });
                Ok(RoutedFusionOutcome {
                    image: out,
                    receipt,
                    timing,
                    fallback_reason: None,
                })
            }
            #[cfg(feature = "wgpu-backend")]
            Backend::Dx12 => {
                let wgpu = self.ensure_wgpu_dx12()?;
                let (out, receipt) = wgpu.dispatch_pixel_fusion(image, fusion)?;
                let timing = wgpu.last_timing().unwrap_or(BackendTiming {
                    wall_ns: started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                    ..BackendTiming::default()
                });
                Ok(RoutedFusionOutcome {
                    image: out,
                    receipt,
                    timing,
                    fallback_reason: None,
                })
            }
            Backend::Metal => {
                let mut metal = MetalComputeBackend::open()?;
                let (out, receipt) = metal.dispatch_pixel_fusion(image, fusion)?;
                let wall_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                Ok(RoutedFusionOutcome {
                    image: out,
                    receipt,
                    timing: BackendTiming {
                        wall_ns,
                        ..BackendTiming::default()
                    },
                    fallback_reason: None,
                })
            }
            Backend::Hip => {
                let mut hip = HipComputeBackend::new().open()?;
                let (out, receipt) = hip.dispatch_pixel_fusion(image, fusion)?;
                let wall_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                Ok(RoutedFusionOutcome {
                    image: out,
                    receipt,
                    timing: BackendTiming {
                        wall_ns,
                        ..BackendTiming::default()
                    },
                    fallback_reason: None,
                })
            }
            Backend::Sycl => {
                let mut sycl = crate::sycl_openvino_backend::SyclComputeBackend::open()?;
                let (out, receipt) = sycl.dispatch_pixel_fusion(image, fusion)?;
                let wall_ns = started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                Ok(RoutedFusionOutcome {
                    image: out,
                    receipt,
                    timing: BackendTiming {
                        wall_ns,
                        ..BackendTiming::default()
                    },
                    fallback_reason: None,
                })
            }
            Backend::Cpu => self.dispatch_cpu_simd(image, fusion, None),
            #[cfg(not(feature = "wgpu-backend"))]
            Backend::Vulkan | Backend::Dx12 => Err(ComputeError::BackendUnavailable(
                "wgpu-backend feature is disabled",
            )),
            Backend::OpenCl | Backend::Npu => Err(ComputeError::BackendUnavailable(
                "pixel_fusion has no OpenCL/NPU dispatcher",
            )),
        }
    }

    /// Execute `fusion` over `image` under `mode` and optional `preferred`
    /// backend. `ExecutionMode::Strict` never substitutes CPU for an unavailable
    /// accelerator.
    pub fn dispatch_pixel_fusion(
        &mut self,
        image: &Image,
        fusion: &PixelStageFusion,
        mode: ExecutionMode,
        preferred: Option<Backend>,
    ) -> Result<RoutedFusionOutcome, ComputeError> {
        if mode == ExecutionMode::Strict {
            let target = preferred.unwrap_or(AUTO_GPU_PRIORITY[0]);
            return self.dispatch_specific(target, image, fusion);
        }

        let mut order: Vec<Backend> = Vec::with_capacity(6);
        if let Some(p) = preferred {
            order.push(p);
        }
        for b in [Backend::Cuda, Backend::Metal, Backend::Vulkan, Backend::Dx12] {
            if !order.contains(&b) {
                order.push(b);
            }
        }

        let mut reasons = Vec::new();
        for backend in order {
            if backend == Backend::Cpu {
                return self.dispatch_cpu_simd(
                    image,
                    fusion,
                    (!reasons.is_empty()).then(|| reasons.join("; ")),
                );
            }
            match self.dispatch_specific(backend, image, fusion) {
                Ok(mut outcome) => {
                    if !reasons.is_empty() {
                        outcome.fallback_reason = Some(reasons.join("; "));
                    }
                    return Ok(outcome);
                }
                Err(err) => reasons.push(format!("{backend}: {err}")),
            }
        }

        self.dispatch_cpu_simd(
            image,
            fusion,
            (!reasons.is_empty()).then(|| reasons.join("; ")),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute_backend::PixelOp;

    #[test]
    fn host_probe_and_candidates_are_consistent_and_non_empty() {
        let probes = probe_all_backends();
        assert!(probes.iter().any(|p| p.backend == Backend::Cpu && p.driver_ready));
        assert!(probes.iter().any(|p| p.backend == Backend::Cuda));

        let candidates = all_autotune_candidates();
        assert!(candidates.len() >= 18, "got {} candidates", candidates.len());
        assert!(candidates.iter().any(|c| c.backend == Backend::Cuda));
        assert!(candidates.iter().any(|c| c.backend == Backend::Metal));
        assert!(candidates.iter().any(|c| c.backend == Backend::Hip));
        assert!(candidates.iter().any(|c| c.backend == Backend::Sycl));
        assert!(candidates.iter().any(|c| c.backend == Backend::Cpu));
    }

    #[test]
    fn strict_mode_refuses_unavailable_accelerators_without_cpu_fallback() {
        let mut router = FusionRouter::new();
        let img = Image::solid(4, 4, [120, 140, 160, 255]);
        let fusion = PixelStageFusion::new()
            .with_stage(PixelOp::Brightness { delta: 0.1 })
            .unwrap();

        for unavailable in [Backend::Hip, Backend::Sycl, Backend::Npu] {
            let err = router
                .dispatch_pixel_fusion(&img, &fusion, ExecutionMode::Strict, Some(unavailable))
                .unwrap_err();
            assert!(matches!(err, ComputeError::BackendUnavailable(_)), "{unavailable}: {err}");
        }
        #[cfg(not(any(target_os = "macos", target_os = "ios")))]
        {
            let err = router
                .dispatch_pixel_fusion(&img, &fusion, ExecutionMode::Strict, Some(Backend::Metal))
                .unwrap_err();
            assert!(matches!(err, ComputeError::BackendUnavailable(_)));
        }
    }

    #[test]
    fn router_auto_fallback_matches_cpu_oracle() {
        let mut router = FusionRouter::new();
        let mut pixels = Vec::with_capacity(17 * 19 * 4);
        for i in 0..(17 * 19) {
            pixels.push((i * 13 % 256) as u8);
            pixels.push((i * 29 % 256) as u8);
            pixels.push((i * 47 % 256) as u8);
            pixels.push((i * 7 % 256) as u8);
        }
        let img = Image::new(17, 19, pixels).unwrap();
        let fusion = PixelStageFusion::new()
            .with_stage(PixelOp::Brightness { delta: 0.08 })
            .unwrap()
            .with_stage(PixelOp::Whiten { strength: 0.4 })
            .unwrap()
            .with_stage(PixelOp::Saturation { amount: 0.2 })
            .unwrap();
        let expected = fusion.execute_fused(&img).unwrap();
        let outcome = router
            .dispatch_pixel_fusion(&img, &fusion, ExecutionMode::AutoFallback, None)
            .unwrap();
        assert_eq!(outcome.image.pixels.len(), expected.pixels.len());
        for (got, want) in outcome
            .image
            .pixels
            .chunks_exact(4)
            .zip(expected.pixels.chunks_exact(4))
        {
            assert_eq!(got[3], want[3], "alpha must be preserved");
            for ch in 0..3 {
                assert!(got[ch].abs_diff(want[ch]) <= 1, "RGB delta > 1 ULP");
            }
        }
    }
}
