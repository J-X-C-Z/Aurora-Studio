//! Explicit backend selection from caller-supplied device probe results.
//!
//! This module never probes hardware and never infers availability from OS or
//! device names. A GPU is usable only when the caller says so and the selected
//! effect's precision, kernel, and input shape are supported.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
    Metal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Fp32,
    Fp16,
}

pub use crate::Access;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceProbe {
    pub backend: Backend,
    pub driver_ready: bool,
    pub kernel_compiled: bool,
    pub fp32: bool,
    pub fp16: bool,
    pub max_width: u32,
    pub max_height: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DeviceRefusal {
    BackendUnavailable(&'static str),
    KernelMissing,
    PrecisionUnsupported(Precision),
    ShapeUnsupported,
    SourceUnavailable,
    EffectUnsupported,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Selection {
    Selected(Backend),
    Refused(DeviceRefusal),
}

#[derive(Clone)]
pub struct DeviceRequest<'a> {
    pub backend: Option<Backend>,
    pub effect_access: Access,
    pub effect_gpu_ready: bool,
    pub source_ok: bool,
    pub shape: (u32, u32),
    pub precision: Precision,
    pub probes: &'a [DeviceProbe],
}

pub fn select(request: DeviceRequest<'_>) -> Selection {
    let Some(backend) = request.backend else {
        if !request.source_ok || request.shape.0 == 0 || request.shape.1 == 0 {
            return Selection::Refused(refusal(&request, Backend::Cpu));
        }
        for backend in [Backend::Cuda, Backend::Metal] {
            if usable(&request, backend).is_some() {
                return Selection::Selected(backend);
            }
        }
        if usable(&request, Backend::Cpu).is_some() {
            return Selection::Selected(Backend::Cpu);
        }
        // Report the backend that owns the effect: a CPU-accessible effect
        // fails on CPU rules, everything else on the first GPU in the order.
        let home = if request.effect_access == Access::Cpu {
            Backend::Cpu
        } else {
            Backend::Cuda
        };
        return Selection::Refused(refusal(&request, home));
    };

    if !request.source_ok || request.shape.0 == 0 || request.shape.1 == 0 {
        return Selection::Refused(refusal(&request, backend));
    }
    if usable(&request, backend).is_some() {
        return Selection::Selected(backend);
    }
    Selection::Refused(refusal(&request, backend))
}

fn usable(request: &DeviceRequest<'_>, backend: Backend) -> Option<()> {
    if request.effect_access == Access::Refused {
        return None;
    }
    if backend == Backend::Cpu {
        return (request.effect_access == Access::Cpu && request.precision == Precision::Fp32)
            .then_some(());
    }
    if !request.effect_gpu_ready {
        return None;
    }
    let probe = request.probes.iter().find(|probe| probe.backend == backend)?;
    if !probe.driver_ready || !probe.kernel_compiled {
        return None;
    }
    let precision_ok = match request.precision {
        Precision::Fp32 => probe.fp32,
        Precision::Fp16 => probe.fp16,
    };
    if !precision_ok {
        return None;
    }
    (request.shape.0 <= probe.max_width && request.shape.1 <= probe.max_height).then_some(())
}

fn refusal(request: &DeviceRequest<'_>, backend: Backend) -> DeviceRefusal {
    if !request.source_ok {
        return DeviceRefusal::SourceUnavailable;
    }
    if request.shape.0 == 0 || request.shape.1 == 0 {
        return DeviceRefusal::ShapeUnsupported;
    }
    if request.effect_access == Access::Refused {
        return DeviceRefusal::EffectUnsupported;
    }
    if backend == Backend::Cpu {
        return if request.effect_access != Access::Cpu {
            DeviceRefusal::EffectUnsupported
        } else if request.precision != Precision::Fp32 {
            DeviceRefusal::PrecisionUnsupported(request.precision)
        } else {
            DeviceRefusal::EffectUnsupported
        };
    }
    if !request.effect_gpu_ready {
        return DeviceRefusal::EffectUnsupported;
    }
    // `usable` lets an Access::Cpu effect take a GPU stage when the caller says
    // that stage is ready, so the refusal must not reject it before the probe.
    let Some(probe) = request.probes.iter().find(|probe| probe.backend == backend) else {
        return DeviceRefusal::BackendUnavailable(match backend {
            Backend::Cuda => "no caller-probe reports NVIDIA CUDA",
            Backend::Metal => "no caller-probe reports Apple Metal",
            Backend::Cpu => unreachable!(),
        });
    };
    if !probe.driver_ready {
        return DeviceRefusal::BackendUnavailable("driver not ready");
    }
    if !probe.kernel_compiled {
        return DeviceRefusal::KernelMissing;
    }
    match request.precision {
        Precision::Fp32 if !probe.fp32 => DeviceRefusal::PrecisionUnsupported(Precision::Fp32),
        Precision::Fp16 if !probe.fp16 => DeviceRefusal::PrecisionUnsupported(Precision::Fp16),
        _ if request.shape.0 > probe.max_width || request.shape.1 > probe.max_height => DeviceRefusal::ShapeUnsupported,
        _ => DeviceRefusal::EffectUnsupported,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: DeviceProbe = DeviceProbe {
        backend: Backend::Cuda,
        driver_ready: true,
        kernel_compiled: true,
        fp32: true,
        fp16: true,
        max_width: 8192,
        max_height: 8192,
    };

    const METAL: DeviceProbe = DeviceProbe {
        backend: Backend::Metal,
        driver_ready: true,
        kernel_compiled: true,
        fp32: true,
        fp16: true,
        max_width: 4096,
        max_height: 4096,
    };

    fn request<'a>(backend: Option<Backend>, probes: &'a [DeviceProbe]) -> DeviceRequest<'a> {
        DeviceRequest {
            backend,
            effect_access: Access::GpuPlanned,
            effect_gpu_ready: true,
            source_ok: true,
            shape: (4096, 4096),
            precision: Precision::Fp16,
            probes,
        }
    }

    #[test]
    fn real_good_probe_with_ready_selects_cuda_or_metal() {
        assert_eq!(select(request(None, &[GOOD])), Selection::Selected(Backend::Cuda));
        assert_eq!(select(request(None, &[METAL])), Selection::Selected(Backend::Metal));
        assert_eq!(select(request(Some(Backend::Cuda), &[GOOD])), Selection::Selected(Backend::Cuda));
        assert_eq!(select(request(Some(Backend::Metal), &[METAL])), Selection::Selected(Backend::Metal));
    }

    #[test]
    fn gpu_ready_alone_is_not_sufficient() {
        assert!(matches!(select(request(Some(Backend::Cuda), &[])), Selection::Refused(_)));
        let mut bad = GOOD;
        bad.driver_ready = false;
        assert!(matches!(select(request(Some(Backend::Cuda), &[bad])), Selection::Refused(_)));
        bad.driver_ready = true;
        bad.kernel_compiled = false;
        assert!(matches!(select(request(Some(Backend::Cuda), &[bad])), Selection::Refused(_)));
        bad.kernel_compiled = true;
        bad.fp16 = false;
        assert!(matches!(select(request(Some(Backend::Cuda), &[bad])), Selection::Refused(_)));
        let mut refused = request(Some(Backend::Cuda), &[GOOD]);
        refused.effect_access = Access::Refused;
        assert!(matches!(select(refused), Selection::Refused(_)));
    }

    #[test]
    fn ready_false_rejects_even_good_probe() {
        let mut no_ready = request(Some(Backend::Cuda), &[GOOD]);
        no_ready.effect_gpu_ready = false;
        assert!(matches!(select(no_ready), Selection::Refused(_)));
    }

    #[test]
    fn cpu_requires_cpu_access_and_fp32() {
        let mut cpu_request = request(Some(Backend::Cpu), &[]);
        cpu_request.effect_access = Access::Cpu;
        cpu_request.precision = Precision::Fp32;
        assert_eq!(select(cpu_request.clone()), Selection::Selected(Backend::Cpu));
        cpu_request.precision = Precision::Fp16;
        assert!(matches!(select(cpu_request.clone()), Selection::Refused(_)));
    }

    #[test]
    fn auto_rejects_invalid_requests() {
        let mut missing = request(None, &[GOOD]);
        missing.source_ok = false;
        assert!(matches!(select(missing), Selection::Refused(DeviceRefusal::SourceUnavailable)));
        let mut refused = request(None, &[GOOD]);
        refused.effect_access = Access::Refused;
        assert!(matches!(select(refused), Selection::Refused(_)));
        let mut empty = request(None, &[GOOD]);
        empty.shape = (0, 4096);
        assert!(matches!(select(empty), Selection::Refused(DeviceRefusal::ShapeUnsupported)));
    }

    /// A CPU-recovered effect whose GPU stage is ready. `usable` allows this on
    /// a GPU, so every refusal below must name the real blocker, never
    /// `EffectUnsupported`.
    fn cpu_stage<'a>(backend: Backend, probes: &'a [DeviceProbe]) -> DeviceRequest<'a> {
        let mut req = request(Some(backend), probes);
        req.effect_access = Access::Cpu;
        req.precision = Precision::Fp32;
        req
    }

    #[test]
    fn cpu_effect_takes_a_gpu_stage() {
        assert_eq!(
            select(cpu_stage(Backend::Cuda, &[GOOD])),
            Selection::Selected(Backend::Cuda)
        );
        assert_eq!(
            select(cpu_stage(Backend::Metal, &[METAL])),
            Selection::Selected(Backend::Metal)
        );
        // Auto prefers the GPU stage over CPU when both are usable.
        let mut auto = request(None, &[GOOD]);
        auto.effect_access = Access::Cpu;
        auto.precision = Precision::Fp32;
        assert_eq!(
            select(auto),
            Selection::Selected(Backend::Cuda)
        );
    }

    #[test]
    fn cpu_stage_refusal_names_the_missing_probe() {
        assert_eq!(
            select(cpu_stage(Backend::Cuda, &[])),
            Selection::Refused(DeviceRefusal::BackendUnavailable(
                "no caller-probe reports NVIDIA CUDA"
            ))
        );
        assert_eq!(
            select(cpu_stage(Backend::Metal, &[])),
            Selection::Refused(DeviceRefusal::BackendUnavailable(
                "no caller-probe reports Apple Metal"
            ))
        );
    }

    #[test]
    fn cpu_stage_refusal_names_driver_kernel_precision_shape() {
        let mut no_driver = GOOD;
        no_driver.driver_ready = false;
        assert_eq!(
            select(cpu_stage(Backend::Cuda, &[no_driver])),
            Selection::Refused(DeviceRefusal::BackendUnavailable("driver not ready"))
        );

        let mut no_kernel = GOOD;
        no_kernel.kernel_compiled = false;
        assert_eq!(
            select(cpu_stage(Backend::Cuda, &[no_kernel])),
            Selection::Refused(DeviceRefusal::KernelMissing)
        );

        let mut no_fp32 = GOOD;
        no_fp32.fp32 = false;
        assert_eq!(
            select(cpu_stage(Backend::Cuda, &[no_fp32])),
            Selection::Refused(DeviceRefusal::PrecisionUnsupported(Precision::Fp32))
        );

        let mut small = GOOD;
        small.max_width = 1024;
        assert_eq!(
            select(cpu_stage(Backend::Cuda, &[small])),
            Selection::Refused(DeviceRefusal::ShapeUnsupported)
        );
    }

    #[test]
    fn cpu_refusal_names_precision_and_access() {
        let mut fp16 = request(Some(Backend::Cpu), &[]);
        fp16.effect_access = Access::Cpu;
        fp16.precision = Precision::Fp16;
        assert_eq!(
            select(fp16),
            Selection::Refused(DeviceRefusal::PrecisionUnsupported(Precision::Fp16))
        );

        // A GPU-only effect asked for CPU is an effect/backend mismatch.
        let mut gpu_only = request(Some(Backend::Cpu), &[GOOD]);
        gpu_only.precision = Precision::Fp32;
        assert_eq!(
            select(gpu_only),
            Selection::Refused(DeviceRefusal::EffectUnsupported)
        );
    }

    #[test]
    fn auto_reporting_cpu_effect_names_the_cpu_blocker() {
        let mut req = request(None, &[]);
        req.effect_access = Access::Cpu;
        req.effect_gpu_ready = false;
        req.precision = Precision::Fp16;
        assert_eq!(
            select(req),
            Selection::Refused(DeviceRefusal::PrecisionUnsupported(Precision::Fp16))
        );
    }
}
