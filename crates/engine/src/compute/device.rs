//! Backend identity and the capability inputs the policy decides on.
//!
//! The pre-M1 tree has three unrelated `Backend` enums
//! (`aurora_portrait::device::Backend`, `aurora_portrait::effect_router::Backend`,
//! and `aimask::ort::Provider`). They cannot be compared or logged together,
//! and none of them carries a device index. [`Backend`] here is the one shape
//! phase 2 will translate into: CPU, or a GPU kind plus the device id the
//! existing code hard-codes to `0`.
//!
//! This module never probes hardware. [`Capabilities`] is filled by the caller
//! from probes that already exist (`DeviceProbe`, the OpenCL vendor check, the
//! Metal `available()` gate). Phase 1 installs nothing, so the process-wide
//! policy sees `Capabilities::none()` and every GPU candidate is unavailable.

use std::fmt;

/// Where a workload can execute.
///
/// The device index is explicit even though today all three GPU lanes use
/// device 0 — the roadmap's "device auto-probe / device ID" requirement needs a
/// place to put the answer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Backend {
    Cpu,
    Cuda(u32),
    Metal(u32),
    OpenCl(u32),
}

impl Backend {
    /// Backend family, without the device index.
    pub fn kind(self) -> BackendKind {
        match self {
            Self::Cpu => BackendKind::Cpu,
            Self::Cuda(_) => BackendKind::Cuda,
            Self::Metal(_) => BackendKind::Metal,
            Self::OpenCl(_) => BackendKind::OpenCl,
        }
    }

    pub fn is_cpu(self) -> bool {
        matches!(self, Self::Cpu)
    }

    pub fn device_index(self) -> Option<u32> {
        match self {
            Self::Cpu => None,
            Self::Cuda(index) | Self::Metal(index) | Self::OpenCl(index) => Some(index),
        }
    }

    /// Stable label for logs and telemetry, e.g. `cpu`, `cuda:0`.
    pub fn label(self) -> String {
        match self {
            Self::Cpu => "cpu".to_owned(),
            Self::Cuda(index) => format!("cuda:{index}"),
            Self::Metal(index) => format!("metal:{index}"),
            Self::OpenCl(index) => format!("opencl:{index}"),
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.label())
    }
}

/// Backend family used to look up a capability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Cpu,
    Cuda,
    Metal,
    OpenCl,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
            Self::OpenCl => "opencl",
        }
    }
}

/// Arithmetic precision a workload needs. `U8` is the 8-bit staging path and
/// is supported by every usable device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Precision {
    Fp32,
    Fp16,
    U8,
}

impl Precision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::U8 => "u8",
        }
    }
}

/// Where the input already lives. The blemish pageable bridge is the only path
/// with a measured residency penalty today, so this is a policy input rather
/// than a pure diagnostic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Residency {
    /// Host memory reached through a pageable transfer.
    CpuPageable,
    /// Host memory already pinned for the device.
    PinnedHost,
    /// Pixels already resident on the device.
    DeviceResident,
}

impl Residency {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuPageable => "cpu_pageable",
            Self::PinnedHost => "pinned_host",
            Self::DeviceResident => "device_resident",
        }
    }
}

/// Scheduling priority. Phase 1 records it for telemetry and phase 2 will let
/// export use a wider GPU budget than an interactive preview.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Priority {
    Interactive,
    Preview,
    Export,
    Background,
}

impl Priority {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Interactive => "interactive",
            Self::Preview => "preview",
            Self::Export => "export",
            Self::Background => "background",
        }
    }
}

/// One probed GPU. `driver_ready` and `kernel_compiled` mirror
/// `aurora_portrait::device::DeviceProbe`; they are the caller's evidence, not
/// something this module infers from an OS or a device name.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpuInfo {
    pub backend: Backend,
    pub name: &'static str,
    pub driver_ready: bool,
    pub kernel_compiled: bool,
    pub fp32: bool,
    pub fp16: bool,
    /// Device memory in bytes; `0` means "unknown, do not reject on capacity".
    pub vram_bytes: u64,
}

impl GpuInfo {
    /// Both the driver and this stage's kernel must be present. A device that
    /// is merely enumerated does not count.
    pub fn usable(self) -> bool {
        self.driver_ready && self.kernel_compiled && !self.backend.is_cpu()
    }

    pub fn supports(self, precision: Precision) -> bool {
        match precision {
            Precision::Fp32 => self.fp32,
            Precision::Fp16 => self.fp16,
            Precision::U8 => true,
        }
    }
}

/// Probed capability for the three GPU families the engine can target.
///
/// One device per family is enough for phase 1: every current lane uses device
/// 0. Phase 2 may extend this to `Vec<GpuInfo>` without changing
/// [`Backend`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Capabilities {
    pub cuda: Option<GpuInfo>,
    pub metal: Option<GpuInfo>,
    pub opencl: Option<GpuInfo>,
}

impl Capabilities {
    /// No GPU is known to be usable. This is the conservative phase-1 default.
    pub const fn none() -> Self {
        Self {
            cuda: None,
            metal: None,
            opencl: None,
        }
    }

    /// Adds or replaces the device for its backend family. A CPU `GpuInfo` is
    /// ignored.
    pub fn with_gpu(mut self, info: GpuInfo) -> Self {
        match info.backend.kind() {
            BackendKind::Cuda => self.cuda = Some(info),
            BackendKind::Metal => self.metal = Some(info),
            BackendKind::OpenCl => self.opencl = Some(info),
            BackendKind::Cpu => {}
        }
        self
    }

    /// Looks up a family, rejecting a record whose `backend` does not match the
    /// slot it was stored in.
    pub fn gpu(&self, kind: BackendKind) -> Option<GpuInfo> {
        let candidate = match kind {
            BackendKind::Cpu => return None,
            BackendKind::Cuda => self.cuda,
            BackendKind::Metal => self.metal,
            BackendKind::OpenCl => self.opencl,
        };
        candidate.filter(|info| info.backend.kind() == kind)
    }

    pub fn any_gpu(&self) -> bool {
        [BackendKind::Cuda, BackendKind::Metal, BackendKind::OpenCl]
            .into_iter()
            .any(|kind| self.gpu(kind).is_some_and(GpuInfo::usable))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(backend: Backend) -> GpuInfo {
        GpuInfo {
            backend,
            name: "test",
            driver_ready: true,
            kernel_compiled: true,
            fp32: true,
            fp16: true,
            vram_bytes: 0,
        }
    }

    #[test]
    fn backend_labels_and_indices() {
        assert_eq!(Backend::Cpu.label(), "cpu");
        assert_eq!(Backend::Cuda(0).label(), "cuda:0");
        assert_eq!(Backend::Metal(1).label(), "metal:1");
        assert_eq!(Backend::OpenCl(0).label(), "opencl:0");
        assert_eq!(Backend::Cpu.device_index(), None);
        assert_eq!(Backend::Cuda(7).device_index(), Some(7));
        assert_eq!(Backend::OpenCl(2).kind(), BackendKind::OpenCl);
    }

    #[test]
    fn capabilities_only_accept_usable_devices_per_family() {
        let caps = Capabilities::none()
            .with_gpu(probe(Backend::Cuda(0)))
            .with_gpu(probe(Backend::OpenCl(0)))
            .with_gpu(probe(Backend::Cpu));

        assert!(caps.gpu(BackendKind::Cuda).is_some());
        assert!(caps.gpu(BackendKind::OpenCl).is_some());
        assert!(caps.gpu(BackendKind::Metal).is_none());
        assert!(caps.gpu(BackendKind::Cpu).is_none());
        assert!(caps.any_gpu());
    }

    #[test]
    fn unusable_device_is_not_capability() {
        let mut info = probe(Backend::Cuda(0));
        info.driver_ready = false;
        let caps = Capabilities::none().with_gpu(info);
        assert!(!caps.any_gpu());
        assert!(!caps.gpu(BackendKind::Cuda).unwrap().usable());
    }

    #[test]
    fn precision_support_is_per_device() {
        let mut info = probe(Backend::Cuda(0));
        info.fp16 = false;
        assert!(info.supports(Precision::Fp32));
        assert!(!info.supports(Precision::Fp16));
        assert!(info.supports(Precision::U8));
    }
}
