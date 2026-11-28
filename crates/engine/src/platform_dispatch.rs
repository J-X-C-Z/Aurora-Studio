//! Engine-level cross-platform capability probe, policy installation, and
//! unified AutoTune candidate registry.

use aurora_portrait::autotune::{AdapterProbeRecord, CandidateSpec, TimingSummary};
use aurora_portrait::device::Backend as PortraitBackend;
use aurora_portrait::platform_runtime::{all_autotune_candidates, probe_all_backends};

use crate::compute::device::{
    Backend as ComputeDeviceBackend, Capabilities, GpuInfo, Precision as ComputePrecision,
};
use crate::compute::policy::install_capabilities;
use crate::contracts::compute::{
    diagnose_backend_selection, BackendCapability, BackendDiagnostic, BackendKind, BackendPolicy,
    Precision as ContractPrecision,
};
#[cfg(feature = "wgpu-backend")]
use crate::gpu_crossplatform::CrossPlatformGpuEngine;

/// Complete cross-platform probe snapshot combining portrait compute adapters,
/// wgpu cross-platform engine adapters, ONNX Runtime providers, and policy
/// diagnostics.
#[derive(Clone, Debug)]
pub struct PlatformCapabilityReport {
    pub portrait_probes: Vec<AdapterProbeRecord>,
    pub contract_capabilities: Vec<BackendCapability>,
    pub ort_providers: Vec<&'static str>,
    pub auto_diagnostic: BackendDiagnostic,
    pub strict_gpu_diagnostic: BackendDiagnostic,
}

/// Maps the portrait device vocabulary to the contract [`BackendKind`].
///
/// Kept here (not in `compute_bridge`, which is Windows-only) so the probe works
/// on every platform. Must stay in sync with `compute_bridge::from_portrait_backend`.
const fn from_portrait_backend(backend: PortraitBackend) -> BackendKind {
    match backend {
        PortraitBackend::Cpu => BackendKind::Cpu,
        PortraitBackend::Cuda => BackendKind::Cuda,
        PortraitBackend::Metal => BackendKind::Metal,
        PortraitBackend::Vulkan => BackendKind::Vulkan,
        PortraitBackend::Dx12 => BackendKind::Dx12,
        PortraitBackend::Hip => BackendKind::Hip,
        PortraitBackend::Sycl => BackendKind::Sycl,
        PortraitBackend::OpenCl => BackendKind::OpenCl,
        PortraitBackend::Npu => BackendKind::Npu,
    }
}

fn to_compute_backend(backend: PortraitBackend) -> ComputeDeviceBackend {
    match backend {
        PortraitBackend::Cpu => ComputeDeviceBackend::Cpu,
        PortraitBackend::Cuda => ComputeDeviceBackend::Cuda(0),
        PortraitBackend::Metal => ComputeDeviceBackend::Metal(0),
        PortraitBackend::Vulkan => ComputeDeviceBackend::Vulkan(0),
        PortraitBackend::Dx12 => ComputeDeviceBackend::Dx12(0),
        PortraitBackend::Hip => ComputeDeviceBackend::Hip(0),
        PortraitBackend::Sycl => ComputeDeviceBackend::Sycl(0),
        PortraitBackend::OpenCl => ComputeDeviceBackend::OpenCl(0),
        PortraitBackend::Npu => ComputeDeviceBackend::Npu(0),
    }
}

/// Builds a live [`Capabilities`] record from probed host adapters without
/// guessing or fabricating hardware capabilities.
pub fn host_compute_capabilities() -> Capabilities {
    let mut caps = Capabilities::none();
    for probe in probe_all_backends() {
        if probe.backend == PortraitBackend::Cpu {
            continue;
        }
        let dev_backend = to_compute_backend(probe.backend);
        let dev_kind = dev_backend.kind();
        // Only overwrite an existing slot if the new probe is usable or the
        // slot is currently empty.
        let already_usable = caps.gpu(dev_kind).is_some_and(GpuInfo::usable);
        if already_usable && !probe.driver_ready {
            continue;
        }
        let name: &'static str = Box::leak(probe.device_name.into_boxed_str());
        caps = caps.with_gpu(GpuInfo {
            backend: dev_backend,
            name,
            driver_ready: probe.driver_ready,
            kernel_compiled: probe.driver_ready,
            fp32: probe.supports_fp32,
            fp16: probe.supports_fp16,
            vram_bytes: probe.vram_bytes,
        });
    }
    caps
}

/// Probes the host once and installs live capabilities into
/// `crate::compute::policy`.
pub fn install_host_capabilities() {
    install_capabilities(host_compute_capabilities());
}

/// Produces a truthful capability and diagnostic report for the current host.
pub fn probe_host_platform(policy: BackendPolicy) -> PlatformCapabilityReport {
    let portrait_probes = probe_all_backends();
    let mut contract_capabilities: Vec<BackendCapability> = portrait_probes
        .iter()
        .map(|probe| BackendCapability {
            kind: from_portrait_backend(probe.backend),
            available: probe.driver_ready,
            device: (!probe.device_name.is_empty() && probe.device_name != "none")
                .then(|| probe.device_name.clone()),
            compute_capability: None,
            precision: if probe.supports_fp16 && probe.supports_fp32 {
                ContractPrecision::Mixed
            } else if probe.supports_fp16 {
                ContractPrecision::F16
            } else {
                ContractPrecision::F32
            },
        })
        .collect();

    #[cfg(feature = "wgpu-backend")]
    {
        for wgpu_cap in
            CrossPlatformGpuEngine::probe_capabilities(CrossPlatformGpuEngine::NATIVE_BACKENDS)
        {
            if !contract_capabilities
                .iter()
                .any(|c| c.kind == wgpu_cap.kind && c.available)
            {
                contract_capabilities.push(wgpu_cap);
            }
        }
    }

    let ort_providers = crate::aimask::ort::Runtime::get()
        .map(|rt| rt.provider_plan().labels())
        .unwrap_or_default();

    let auto_diagnostic = diagnose_backend_selection(policy, &contract_capabilities);
    let strict_gpu_diagnostic =
        diagnose_backend_selection(BackendPolicy::StrictGpu, &contract_capabilities);

    PlatformCapabilityReport {
        portrait_probes,
        contract_capabilities,
        ort_providers,
        auto_diagnostic,
        strict_gpu_diagnostic,
    }
}

/// Aggregates all AutoTune `CandidateSpec` entries across `aurora-portrait` and
/// `aurora-engine`.
pub fn engine_autotune_candidates() -> Vec<CandidateSpec> {
    let mut specs = all_autotune_candidates();
    #[cfg(feature = "wgpu-backend")]
    if let Ok(engine) =
        CrossPlatformGpuEngine::new(aurora_portrait::autotune::ExecutionMode::Strict)
    {
        specs.extend(engine.candidate_specs());
    }
    specs
}

/// Summarizes warm nanosecond timing samples into P50/P95/P99 milliseconds.
pub fn summarize_timings_ms(samples_ns: &[u64]) -> Option<TimingSummary> {
    TimingSummary::from_nanos(samples_ns)
}

#[allow(dead_code)]
fn _assert_precision_used(_: ComputePrecision) {}
#[allow(dead_code)]
fn _assert_kind_used(_: BackendKind) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_platform_report_lists_truthful_backends_and_candidates() {
        let report = probe_host_platform(BackendPolicy::Auto);
        assert!(
            report
                .contract_capabilities
                .iter()
                .any(|c| c.kind == BackendKind::Cpu && c.available),
            "CPU capability must always be available"
        );
        assert!(!report.auto_diagnostic.summary.is_empty());
        let caps = host_compute_capabilities();
        let _ = caps.any_gpu();
        let candidates = engine_autotune_candidates();
        assert!(candidates.len() >= 18);
    }

    #[test]
    #[ignore = "Release hardware benchmark: run with --release -- --ignored --nocapture"]
    fn bench_crossplatform_1080p_and_4k() {
        use std::collections::HashMap;
        use std::time::Instant;

        use aurora_portrait::autotune::ExecutionMode;
        use aurora_portrait::compute_backend::{
            ComputeBackend, CpuComputeBackend, PixelOp, PixelStageFusion,
        };
        use aurora_portrait::native_infer_cuda::execute_graph_cuda;
        use aurora_portrait::native_infer_graph::{
            execute_graph, Graph, Node, OpKind, TensorBuffer, TensorShape,
        };
        use aurora_portrait::platform_runtime::FusionRouter;
        use aurora_portrait::simd_ops::execute_fused_simd;
        use aurora_portrait::Image;

        use crate::gpu_crossplatform::{
            demosaic_bayer_cpu_ref, detail_histogram_cpu_ref, develop_cpu_ref, CfaPattern,
            DevelopParams,
        };

        let report = probe_host_platform(BackendPolicy::Auto);
        println!("BENCH_HOST_SUMMARY: {}", report.auto_diagnostic.summary);
        for p in &report.portrait_probes {
            println!(
                "BENCH_PROBE backend={} ready={} device={} api={} status={}",
                p.backend, p.driver_ready, p.device_name, p.api_version, p.status
            );
        }

        let fusion = PixelStageFusion::new()
            .with_stage(PixelOp::Brightness { delta: 0.08 })
            .unwrap()
            .with_stage(PixelOp::Whiten { strength: 0.45 })
            .unwrap()
            .with_stage(PixelOp::Rosy { strength: 0.25 })
            .unwrap()
            .with_stage(PixelOp::Temperature {
                warmth: 0.12,
                tint: -0.05,
            })
            .unwrap()
            .with_stage(PixelOp::Saturation { amount: 0.15 })
            .unwrap();

        let mut router = FusionRouter::new();
        let mut cpu = CpuComputeBackend::new();

        for (label, w, h, iters) in [("1080p", 1920u32, 1080u32, 15usize), ("4K", 3840u32, 2160u32, 10usize)] {
            let mut pixels = vec![0u8; (w * h * 4) as usize];
            for (idx, chunk) in pixels.chunks_exact_mut(4).enumerate() {
                chunk[0] = ((idx * 17) % 256) as u8;
                chunk[1] = ((idx * 31) % 256) as u8;
                chunk[2] = ((idx * 43) % 256) as u8;
                chunk[3] = 255;
            }
            let img = Image::new(w, h, pixels).unwrap();
            let mp = (w as f64 * h as f64) / 1_000_000.0;

            // Scalar CPU
            let _ = cpu.dispatch_pixel_fusion(&img, &fusion).unwrap();
            let mut ns = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let _ = cpu.dispatch_pixel_fusion(&img, &fusion).unwrap();
                ns.push(t0.elapsed().as_nanos() as u64);
            }
            let s = TimingSummary::from_nanos(&ns).unwrap();
            println!(
                "BENCH_FUSION size={label} backend=cpu_scalar p50_ms={:.3} p95_ms={:.3} p99_ms={:.3} mp_per_s={:.2}",
                s.p50_ms, s.p95_ms, s.p99_ms, mp * 1000.0 / s.p50_ms
            );

            // SIMD CPU
            let mut dst = vec![0u8; img.pixels.len()];
            execute_fused_simd(&fusion, &img.pixels, &mut dst).unwrap();
            ns.clear();
            for _ in 0..iters {
                let t0 = Instant::now();
                execute_fused_simd(&fusion, &img.pixels, &mut dst).unwrap();
                ns.push(t0.elapsed().as_nanos() as u64);
            }
            let s = TimingSummary::from_nanos(&ns).unwrap();
            println!(
                "BENCH_FUSION size={label} backend=cpu_simd p50_ms={:.3} p95_ms={:.3} p99_ms={:.3} mp_per_s={:.2}",
                s.p50_ms, s.p95_ms, s.p99_ms, mp * 1000.0 / s.p50_ms
            );

            // CUDA, WGPU Vulkan, WGPU DX12
            for b in [
                aurora_portrait::device::Backend::Cuda,
                aurora_portrait::device::Backend::Vulkan,
                aurora_portrait::device::Backend::Dx12,
            ] {
                if router
                    .dispatch_pixel_fusion(&img, &fusion, ExecutionMode::Strict, Some(b))
                    .is_ok()
                {
                    ns.clear();
                    let mut h2d = 0u64;
                    let mut d2h = 0u64;
                    for _ in 0..iters {
                        let t0 = Instant::now();
                        let res = router
                            .dispatch_pixel_fusion(&img, &fusion, ExecutionMode::Strict, Some(b))
                            .unwrap();
                        ns.push(t0.elapsed().as_nanos() as u64);
                        h2d = res.timing.h2d_bytes;
                        d2h = res.timing.d2h_bytes;
                    }
                    let s = TimingSummary::from_nanos(&ns).unwrap();
                    println!(
                        "BENCH_FUSION size={label} backend={b} p50_ms={:.3} p95_ms={:.3} p99_ms={:.3} mp_per_s={:.2} h2d_bytes={h2d} d2h_bytes={d2h}",
                        s.p50_ms, s.p95_ms, s.p99_ms, mp * 1000.0 / s.p50_ms
                    );
                }
            }
        }

        // CrossPlatformGpuEngine: demosaic, develop, detail+histogram at 1080p
        #[cfg(feature = "wgpu-backend")]
        if let Ok(mut gpu) = CrossPlatformGpuEngine::new(ExecutionMode::Strict) {
            let (w, h, iters) = (1920u32, 1080u32, 8usize);
            let bayer: Vec<f32> = (0..(w * h) as usize).map(|i| ((i % 256) as f32) / 255.0).collect();
            let dev_params = DevelopParams {
                wb: [1.1, 1.0, 1.2],
                exposure_ev: 0.25,
                highlights: 0.1,
                shadows: 0.1,
                contrast: 0.1,
                saturation: 0.1,
            };

            let rgba_lin = demosaic_bayer_cpu_ref(&bayer, w as usize, h as usize, CfaPattern::Rggb);
            let mut ns = Vec::with_capacity(iters);
            for _ in 0..iters {
                let t0 = Instant::now();
                let _ = demosaic_bayer_cpu_ref(&bayer, w as usize, h as usize, CfaPattern::Rggb);
                ns.push(t0.elapsed().as_nanos() as u64);
            }
            let s = TimingSummary::from_nanos(&ns).unwrap();
            println!(
                "BENCH_ENGINE stage=demosaic size=1080p backend=cpu_ref p50_ms={:.3} p95_ms={:.3}",
                s.p50_ms, s.p95_ms
            );

            let _ = gpu.demosaic(&bayer, w, h, CfaPattern::Rggb).unwrap();
            ns.clear();
            for _ in 0..iters {
                let t0 = Instant::now();
                let _ = gpu.demosaic(&bayer, w, h, CfaPattern::Rggb).unwrap();
                ns.push(t0.elapsed().as_nanos() as u64);
            }
            let s = TimingSummary::from_nanos(&ns).unwrap();
            println!(
                "BENCH_ENGINE stage=demosaic size=1080p backend=wgpu_vulkan p50_ms={:.3} p95_ms={:.3}",
                s.p50_ms, s.p95_ms
            );

            ns.clear();
            for _ in 0..iters {
                let t0 = Instant::now();
                let _ = develop_cpu_ref(&rgba_lin, &dev_params);
                ns.push(t0.elapsed().as_nanos() as u64);
            }
            let s = TimingSummary::from_nanos(&ns).unwrap();
            println!(
                "BENCH_ENGINE stage=develop size=1080p backend=cpu_ref p50_ms={:.3} p95_ms={:.3}",
                s.p50_ms, s.p95_ms
            );

            let dev_u8 = gpu.develop(&rgba_lin, &dev_params).unwrap();
            ns.clear();
            for _ in 0..iters {
                let t0 = Instant::now();
                let _ = gpu.develop(&rgba_lin, &dev_params).unwrap();
                ns.push(t0.elapsed().as_nanos() as u64);
            }
            let s = TimingSummary::from_nanos(&ns).unwrap();
            println!(
                "BENCH_ENGINE stage=develop size=1080p backend=wgpu_vulkan p50_ms={:.3} p95_ms={:.3}",
                s.p50_ms, s.p95_ms
            );

            ns.clear();
            for _ in 0..iters {
                let t0 = Instant::now();
                let _ = detail_histogram_cpu_ref(&dev_u8, w as usize, h as usize, 0.35);
                ns.push(t0.elapsed().as_nanos() as u64);
            }
            let s = TimingSummary::from_nanos(&ns).unwrap();
            println!(
                "BENCH_ENGINE stage=detail_hist size=1080p backend=cpu_ref p50_ms={:.3} p95_ms={:.3}",
                s.p50_ms, s.p95_ms
            );

            let _ = gpu.detail_histogram(&dev_u8, w, h, 0.35).unwrap();
            ns.clear();
            for _ in 0..iters {
                let t0 = Instant::now();
                let _ = gpu.detail_histogram(&dev_u8, w, h, 0.35).unwrap();
                ns.push(t0.elapsed().as_nanos() as u64);
            }
            let s = TimingSummary::from_nanos(&ns).unwrap();
            println!(
                "BENCH_ENGINE stage=detail_hist size=1080p backend=wgpu_vulkan p50_ms={:.3} p95_ms={:.3}",
                s.p50_ms, s.p95_ms
            );
        }

        // NativeInferGraph Conv2D + SiLU benchmark (1x16x256x256 -> 1x16x256x256)
        let in_shape = TensorShape::new(1, 16, 256, 256);
        let w_shape = TensorShape::new(16, 16, 3, 3);
        let mut g = Graph::new();
        let x_id = g.fresh_id();
        g.declare_runtime_input(x_id);
        let w_id = g.add_constant(
            TensorBuffer::from_f32(w_shape, &vec![0.01f32; w_shape.numel()]).unwrap(),
        );
        let conv_out = g.fresh_id();
        g.add_node(Node::new(
            OpKind::Conv2d {
                stride: (1, 1),
                pad: (1, 1, 1, 1),
                dilation: (1, 1),
                weight: w_id,
                bias: None,
            },
            vec![x_id],
            conv_out,
            "conv",
        ));
        let silu_out = g.fresh_id();
        g.add_node(Node::new(OpKind::SiLU, vec![conv_out], silu_out, "silu"));
        g.set_outputs(vec![silu_out]);
        let mut runtime_inputs = HashMap::new();
        runtime_inputs.insert(
            x_id,
            TensorBuffer::from_f32(in_shape, &vec![0.5f32; in_shape.numel()]).unwrap(),
        );

        let _ = execute_graph(&g, &runtime_inputs).unwrap();
        let mut ns = Vec::with_capacity(5);
        for _ in 0..5 {
            let t0 = Instant::now();
            let _ = execute_graph(&g, &runtime_inputs).unwrap();
            ns.push(t0.elapsed().as_nanos() as u64);
        }
        let s = TimingSummary::from_nanos(&ns).unwrap();
        println!(
            "BENCH_INFER shape=1x16x256x256 backend=cpu_ref p50_ms={:.3} p95_ms={:.3}",
            s.p50_ms, s.p95_ms
        );

        if execute_graph_cuda(&g, &runtime_inputs, ExecutionMode::Strict).is_ok() {
            ns.clear();
            for _ in 0..5 {
                let t0 = Instant::now();
                let _ = execute_graph_cuda(&g, &runtime_inputs, ExecutionMode::Strict).unwrap();
                ns.push(t0.elapsed().as_nanos() as u64);
            }
            let s = TimingSummary::from_nanos(&ns).unwrap();
            println!(
                "BENCH_INFER shape=1x16x256x256 backend=cuda p50_ms={:.3} p95_ms={:.3}",
                s.p50_ms, s.p95_ms
            );
        }
    }
}
