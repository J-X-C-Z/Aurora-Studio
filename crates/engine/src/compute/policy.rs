//! The pure capability/policy layer.
//!
//! [`decide`] answers "where should this workload run, and why?" from two
//! inputs: the one-time environment snapshot (the existing `AURORA_*` switch
//! state) and the caller's capability probe results. It does **not** dispatch
//! anything, does not call any existing `enabled()`, and does not read the
//! environment a second time. Phase 2 wires the existing call sites to
//! [`decide`]; phase 1 only proves the decision table.
//!
//! ## Compatibility order
//!
//! The existing switches keep their meaning. For a GPU-capable operation the
//! order is:
//!
//! 1. `CpuOnly` wins over everything, including `AURORA_*_GPU=1`.
//! 2. A CPU-required operation is always `Cpu`; `StrictGpu` must not report
//!    that as a violation.
//! 3. An explicit `AURORA_*_GPU=0` / `false` (or `AURORA_ORT_PROVIDERS=cpu`,
//!    `AURORA_PORTRAIT_FORCE_CPU`, ...) wins over `PreferGpu` and `StrictGpu`.
//! 4. Otherwise the mode decides: `Auto` follows the shipped default and cost
//!    seeds, `PreferGpu` takes any capable device, `StrictGpu` requires one.
//!
//! ## Reasons are not free-form
//!
//! [`Reason`] is a closed enum so phase 2 can count decisions without parsing
//! strings. [`ExecutionPlan::telemetry_json`] emits the stable one-line record
//! used by the existing `AURORA_PERF` replay format.

use std::fmt;
use std::sync::OnceLock;

use crate::compute::device::{
    Backend, BackendKind, Capabilities, Precision, Priority, Residency,
};
use crate::compute::mode::ComputeMode;
use crate::compute::telemetry::{BuildProfile, EnvSnapshot, Platform};

/// Mirror of `mock::gpu_runtime::DEFAULT_VRAM_BUDGET_MB`. Kept local so this
/// module compiles on every platform and does not depend on the Windows-only
/// mock stages. Phase 2 should source it from the ledger instead.
pub const DEFAULT_VRAM_BUDGET_MB: u64 = 2560;

/// First `true` row of `mock::develop_gpu::CROSSOVER_TABLE` (1920×1080).
pub const DEVELOP_MIN_PIXELS: u64 = 2_073_600;

/// One engine operation that chooses a compute backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Operation {
    Develop,
    NoiseReduction,
    Clarity,
    Detail,
    PixelVignette,
    PixelLcc,
    PixelGrain,
    Blemish,
    Portrait,
    OrtInference,
    FaceInput,
    FaceParsingMask,
    RawOpenCl,
    /// LibRaw/AHD demosaic: CPU algorithm selection, no device path.
    Demosaic,
    /// ICC transform: CPU colour management, no device path.
    IccTransform,
}

const NO_GPU: &[BackendKind] = &[];
const CUDA_ONLY: &[BackendKind] = &[BackendKind::Cuda];
const CUDA_OR_METAL: &[BackendKind] = &[BackendKind::Cuda, BackendKind::Metal];
const OPENCL_ONLY: &[BackendKind] = &[BackendKind::OpenCl];

impl Operation {
    pub const ALL: [Operation; 15] = [
        Operation::Develop,
        Operation::NoiseReduction,
        Operation::Clarity,
        Operation::Detail,
        Operation::PixelVignette,
        Operation::PixelLcc,
        Operation::PixelGrain,
        Operation::Blemish,
        Operation::Portrait,
        Operation::OrtInference,
        Operation::FaceInput,
        Operation::FaceParsingMask,
        Operation::RawOpenCl,
        Operation::Demosaic,
        Operation::IccTransform,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Develop => "develop",
            Self::NoiseReduction => "noise_reduction",
            Self::Clarity => "clarity",
            Self::Detail => "detail",
            Self::PixelVignette => "pixel_vignette",
            Self::PixelLcc => "pixel_lcc",
            Self::PixelGrain => "pixel_grain",
            Self::Blemish => "blemish",
            Self::Portrait => "portrait",
            Self::OrtInference => "ort_inference",
            Self::FaceInput => "face_input",
            Self::FaceParsingMask => "face_parsing_mask",
            Self::RawOpenCl => "raw_opencl",
            Self::Demosaic => "demosaic",
            Self::IccTransform => "icc_transform",
        }
    }

    /// Operations with no GPU implementation. A `StrictGpu` plan for these is
    /// still a valid CPU plan, not a strict violation.
    pub fn is_cpu_required(self) -> bool {
        matches!(self, Self::Demosaic | Self::IccTransform)
    }

    /// Which GPU families this operation can actually use today.
    ///
    /// The list is deliberately narrow: it reflects wired lanes, not a wish.
    /// DirectML, TensorRT and CoreML are execution providers inside ONNX
    /// Runtime, not backends this enum models; they stay on the CPU side of the
    /// decision until phase 2 extends [`Backend`].
    pub fn gpu_kinds(self) -> &'static [BackendKind] {
        match self {
            Self::Demosaic | Self::IccTransform => NO_GPU,
            Self::Portrait => CUDA_OR_METAL,
            Self::RawOpenCl => OPENCL_ONLY,
            Self::Develop
            | Self::NoiseReduction
            | Self::Clarity
            | Self::Detail
            | Self::PixelVignette
            | Self::PixelLcc
            | Self::PixelGrain
            | Self::Blemish
            | Self::OrtInference
            | Self::FaceInput
            | Self::FaceParsingMask => CUDA_ONLY,
        }
    }

    /// What the existing switch means for this operation.
    ///
    /// This is a read of the snapshot, not a policy decision: `Auto` resolves
    /// `DefaultOn` / `DefaultOff`, while `PreferGpu` and `StrictGpu` may
    /// override a default that is merely a historical choice.
    pub fn env_gate(self, env: &EnvSnapshot, platform: Platform, profile: BuildProfile) -> StageGate {
        match self {
            Self::Develop => match env.flag_false_off("AURORA_DEVELOP_GPU") {
                Some(true) => StageGate::ForcedOn,
                Some(false) => StageGate::DisabledByEnv,
                None if platform == Platform::Windows => StageGate::DefaultOn,
                None => StageGate::DefaultOff,
            },
            Self::NoiseReduction => match env.flag_zero_off("AURORA_NR_GPU") {
                Some(true) => StageGate::ForcedOn,
                Some(false) => StageGate::DisabledByEnv,
                // Replicates `!cfg!(test)` in `nr_gpu::enabled`.
                None if profile == BuildProfile::Release => StageGate::DefaultOn,
                None => StageGate::DefaultOff,
            },
            Self::Clarity => match env.flag_zero_off("AURORA_CLARITY_GPU") {
                Some(true) => StageGate::ForcedOn,
                Some(false) => StageGate::DisabledByEnv,
                None => StageGate::DefaultOn,
            },
            Self::Detail => match env.exact_one("AURORA_DETAIL_GPU") {
                Some(true) => StageGate::ForcedOn,
                Some(false) => StageGate::DisabledByEnv,
                None => StageGate::DefaultOff,
            },
            Self::PixelVignette => {
                if pixel_master_disabled(env) {
                    StageGate::DisabledByEnv
                } else {
                    match env.flag_false_off("AURORA_PIXEL_VIGNETTE_GPU") {
                        Some(true) => StageGate::ForcedOn,
                        Some(false) => StageGate::DisabledByEnv,
                        None => StageGate::DefaultOff,
                    }
                }
            }
            Self::PixelLcc => pixel_op_gate(env, "AURORA_PIXEL_LCC_GPU"),
            Self::PixelGrain => pixel_op_gate(env, "AURORA_PIXEL_GRAIN_GPU"),
            Self::Blemish => match env.var("AURORA_BLEMISH_GPU") {
                // Unset is auto, and the measured pageable bridge is slower
                // than the CPU, so `gpu_auto()` stays false.
                None => StageGate::DefaultOff,
                Some("1") => StageGate::ForcedOn,
                Some("0") => StageGate::DisabledByEnv,
                // Any other value still leaves `gpu_forced()` false, so the
                // product auto-skips to the CPU without an explicit opt-out.
                Some(_) => StageGate::DefaultOff,
            },
            Self::Portrait => {
                let render_cpu = env
                    .var("AURORA_PORTRAIT_RENDER_BACKEND")
                    .is_some_and(|value| value.trim().eq_ignore_ascii_case("cpu"));
                if render_cpu || env.is_set("AURORA_PORTRAIT_FORCE_CPU") {
                    StageGate::DisabledByEnv
                } else {
                    StageGate::DefaultOn
                }
            }
            Self::OrtInference => ort_gate(env),
            Self::FaceInput => face_input_gate(env),
            Self::FaceParsingMask => {
                match env
                    .var("AURORA_FACE_PARSING_MASK_BACKEND")
                    .map(|value| value.trim().to_ascii_lowercase())
                {
                    Some(value) if value == "cpu" => StageGate::DisabledByEnv,
                    Some(value) if value == "cuda" => StageGate::ForcedOn,
                    // Unset and unrecognised values both try CUDA today.
                    _ => StageGate::DefaultOn,
                }
            }
            Self::RawOpenCl => {
                if env.var("AURORA_C1_RAW") == Some("0") {
                    StageGate::DisabledByEnv
                } else {
                    StageGate::DefaultOn
                }
            }
            Self::Demosaic | Self::IccTransform => StageGate::DefaultOff,
        }
    }
}

/// What the existing environment says about one operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageGate {
    /// An explicit switch turns the GPU path off. This wins over every mode.
    DisabledByEnv,
    /// An explicit switch forces the GPU path on. It bypasses the cost seeds
    /// but not availability or the VRAM budget.
    ForcedOn,
    /// Unset, and the shipped default is to use the GPU when admitted.
    DefaultOn,
    /// Unset, and the shipped default is the CPU path.
    DefaultOff,
}

fn pixel_master_disabled(env: &EnvSnapshot) -> bool {
    env.flag_false_off("AURORA_PIXEL_GPU") == Some(false)
}

fn pixel_op_gate(env: &EnvSnapshot, name: &str) -> StageGate {
    if pixel_master_disabled(env) {
        return StageGate::DisabledByEnv;
    }
    match env.flag_false_off(name) {
        Some(true) => StageGate::ForcedOn,
        Some(false) => StageGate::DisabledByEnv,
        None => StageGate::DefaultOn,
    }
}

fn ort_gate(env: &EnvSnapshot) -> StageGate {
    if let Some(raw) = env.var("AURORA_ORT_PROVIDERS") {
        for part in raw.split(',') {
            let token = part.trim().to_ascii_lowercase();
            if matches!(
                token.as_str(),
                "tensorrt" | "cuda" | "directml" | "dml" | "coreml"
            ) {
                return StageGate::ForcedOn;
            }
        }
        // An empty list, `cpu`, or only unknown names all end at CPU today.
        return StageGate::DisabledByEnv;
    }
    if let Some(raw) = env.var("AURORA_AI_PROVIDER") {
        return match raw.trim().to_ascii_lowercase().as_str() {
            "cpu" => StageGate::DisabledByEnv,
            "tensorrt" | "cuda" | "directml" | "dml" | "coreml" => StageGate::ForcedOn,
            // An unrecognised name means automatic in `forced_provider`.
            _ => StageGate::DefaultOn,
        };
    }
    StageGate::DefaultOn
}

fn face_input_gate(env: &EnvSnapshot) -> StageGate {
    match env
        .var("AURORA_FACE_INPUT_BACKEND")
        .map(|value| value.trim().to_ascii_lowercase())
    {
        None => StageGate::DefaultOff,
        Some(value) if value.is_empty() || value == "cpu" => StageGate::DefaultOff,
        Some(value) if matches!(value.as_str(), "cuda" | "directml" | "dml" | "auto") => {
            StageGate::ForcedOn
        }
        // The product returns a refusal for an unknown value; the policy is
        // conservative and keeps the CPU lane instead of inventing a GPU.
        Some(_) => StageGate::DisabledByEnv,
    }
}

/// One unit of work offered to the policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Workload {
    pub operation: Operation,
    /// Output pixels in the developed frame.
    pub pixels: u64,
    pub residency: Residency,
    /// Peak host memory the stage will hold, in bytes. Recorded now; phase 2
    /// uses it for the CPU-side budget.
    pub est_ram: u64,
    /// Peak device memory the stage will hold, in bytes. `0` means unknown.
    pub est_vram: u64,
    pub precision: Precision,
    pub priority: Priority,
}

impl Workload {
    /// A preview-priority, pageable, FP32 workload with unknown estimates —
    /// the common case for a call site that has not been taught to estimate.
    pub fn new(operation: Operation, pixels: u64) -> Self {
        Self {
            operation,
            pixels,
            residency: Residency::CpuPageable,
            est_ram: 0,
            est_vram: 0,
            precision: Precision::Fp32,
            priority: Priority::Preview,
        }
    }

    pub fn with_residency(mut self, residency: Residency) -> Self {
        self.residency = residency;
        self
    }

    pub fn with_est_ram(mut self, bytes: u64) -> Self {
        self.est_ram = bytes;
        self
    }

    pub fn with_est_vram(mut self, bytes: u64) -> Self {
        self.est_vram = bytes;
        self
    }

    pub fn with_precision(mut self, precision: Precision) -> Self {
        self.precision = precision;
        self
    }

    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }
}

/// The decision returned to a call site.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExecutionPlan {
    pub backend: Backend,
    /// Why this backend won. Closed enum; see [`Reason::as_str`] for the
    /// stable telemetry spelling.
    pub reason: Reason,
    /// Backend to use if `backend` fails at dispatch. `None` means the caller
    /// has no permission to fall back (strict GPU, CPU-required, CPU-only).
    pub fallback: Option<Backend>,
}

impl ExecutionPlan {
    fn cpu(reason: Reason) -> Self {
        Self {
            backend: Backend::Cpu,
            reason,
            fallback: None,
        }
    }

    fn gpu(backend: Backend, reason: Reason, fallback: Option<Backend>) -> Self {
        Self {
            backend,
            reason,
            fallback,
        }
    }

    pub fn is_gpu(&self) -> bool {
        !self.backend.is_cpu()
    }

    /// `true` when a `StrictGpu` caller must surface an error instead of
    /// quietly rendering on the CPU. CPU-required operations never set this.
    pub fn is_strict_violation(&self) -> bool {
        self.reason.is_strict_violation()
    }

    /// Stable one-line JSON for `AURORA_PERF`-style replay.
    pub fn telemetry_json(&self, workload: &Workload, mode: ComputeMode) -> String {
        let fallback = match self.fallback {
            Some(backend) => format!("\"{}\"", backend.label()),
            None => "null".to_owned(),
        };
        let device = match self.backend.device_index() {
            Some(index) => index.to_string(),
            None => "null".to_owned(),
        };
        format!(
            "{{\"event\":\"compute_decision\",\"mode\":\"{}\",\"operation\":\"{}\",\
\"backend\":\"{}\",\"device\":{},\"reason\":\"{}\",\"fallback\":{},\"pixels\":{},\
\"residency\":\"{}\",\"precision\":\"{}\",\"priority\":\"{}\",\"est_ram\":{},\"est_vram\":{}}}",
            mode.as_str(),
            workload.operation.as_str(),
            self.backend.kind().as_str(),
            device,
            self.reason.as_str(),
            fallback,
            workload.pixels,
            workload.residency.as_str(),
            workload.precision.as_str(),
            workload.priority.as_str(),
            workload.est_ram,
            workload.est_vram,
        )
    }
}

/// Why [`decide_with`] chose its backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reason {
    /// `ComputeMode::CpuOnly` forced the CPU lane.
    CpuOnlyMode,
    /// The operation has no GPU implementation at all.
    CpuRequiredOperation,
    /// An explicit `AURORA_*` switch disabled the GPU path.
    GpuDisabledByEnv,
    /// Unset, and the shipped default is the CPU path.
    GpuNotEnabledByDefault,
    /// No probed device can run this operation at the requested precision or
    /// capacity.
    GpuUnavailable,
    /// `Auto` kept the CPU because the workload is below its crossover seed or
    /// a known residency penalty applies.
    GpuNotBeneficial,
    /// The input is host-pageable and this operation's measured bridge is
    /// slower than the CPU (the blemish gain-map path).
    ResidencyPenalty,
    /// The estimated device working set exceeds `AURORA_VRAM_BUDGET_MB`.
    VramBudgetExceeded,
    /// The mode or the shipped default preferred the GPU.
    GpuPreferred,
    /// An explicit `=1`-style switch forced the GPU.
    GpuForcedByEnv,
    /// `StrictGpu` selected a device.
    GpuRequired,
    /// `StrictGpu` required a device and none was usable. Hard failure.
    StrictGpuUnavailable,
    /// `StrictGpu` required a device and the VRAM budget refused it. Hard
    /// failure.
    StrictGpuBudgetExceeded,
}

impl Reason {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::CpuOnlyMode => "cpu_only_mode",
            Self::CpuRequiredOperation => "cpu_required_operation",
            Self::GpuDisabledByEnv => "gpu_disabled_by_env",
            Self::GpuNotEnabledByDefault => "gpu_not_enabled_by_default",
            Self::GpuUnavailable => "gpu_unavailable",
            Self::GpuNotBeneficial => "gpu_not_beneficial",
            Self::ResidencyPenalty => "residency_penalty",
            Self::VramBudgetExceeded => "vram_budget_exceeded",
            Self::GpuPreferred => "gpu_preferred",
            Self::GpuForcedByEnv => "gpu_forced_by_env",
            Self::GpuRequired => "gpu_required",
            Self::StrictGpuUnavailable => "strict_gpu_unavailable",
            Self::StrictGpuBudgetExceeded => "strict_gpu_budget_exceeded",
        }
    }

    pub fn is_strict_violation(self) -> bool {
        matches!(
            self,
            Self::StrictGpuUnavailable | Self::StrictGpuBudgetExceeded
        )
    }
}

impl fmt::Display for Reason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Everything [`decide_with`] is allowed to look at.
///
/// `capture` takes the process-wide singletons: the environment is read once
/// and the capability set is whatever was installed first. Unit tests construct
/// the struct directly so no test reads the host environment or a GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolicyInputs {
    pub env: EnvSnapshot,
    pub caps: Capabilities,
    pub platform: Platform,
    pub profile: BuildProfile,
}

static INSTALLED_CAPABILITIES: OnceLock<Capabilities> = OnceLock::new();

/// Installs the process-wide capability set once, before the first decision.
/// Returns `false` if a set was already installed. This does not probe
/// anything; phase 2 calls it with the results of the existing probes.
pub fn install_capabilities(caps: Capabilities) -> bool {
    INSTALLED_CAPABILITIES.set(caps).is_ok()
}

fn installed_capabilities() -> Capabilities {
    INSTALLED_CAPABILITIES
        .get()
        .cloned()
        .unwrap_or_else(Capabilities::none)
}

impl PolicyInputs {
    /// Process-wide inputs. The environment snapshot is taken here only on the
    /// first call; later calls reuse it even if the process environment
    /// changes.
    pub fn capture() -> &'static PolicyInputs {
        static GLOBAL: OnceLock<PolicyInputs> = OnceLock::new();
        GLOBAL.get_or_init(|| Self {
            env: EnvSnapshot::capture(),
            caps: installed_capabilities(),
            platform: Platform::current(),
            profile: BuildProfile::current(),
        })
    }

    /// Empty environment, no GPU, release defaults. Tests add caps and env.
    pub fn for_test(platform: Platform) -> Self {
        Self {
            env: EnvSnapshot::default(),
            caps: Capabilities::none(),
            platform,
            profile: BuildProfile::Release,
        }
    }

    pub fn with_env(mut self, env: EnvSnapshot) -> Self {
        self.env = env;
        self
    }

    pub fn with_caps(mut self, caps: Capabilities) -> Self {
        self.caps = caps;
        self
    }

    pub fn with_profile(mut self, profile: BuildProfile) -> Self {
        self.profile = profile;
        self
    }
}

/// Decides one workload using the process-wide snapshot and capabilities.
///
/// This is the function phase 2 call sites will use. It is pure with respect to
/// its inputs: for a fixed [`PolicyInputs`] snapshot the same workload and mode
/// always return the same plan, and it has no side effects.
pub fn decide(workload: &Workload, mode: ComputeMode) -> ExecutionPlan {
    decide_with(workload, mode, PolicyInputs::capture())
}

/// Fully explicit form used by tests and by callers that hold their own
/// capability probe results.
pub fn decide_with(
    workload: &Workload,
    mode: ComputeMode,
    inputs: &PolicyInputs,
) -> ExecutionPlan {
    let operation = workload.operation;

    // 1. CpuOnly is absolute, including over an explicit `=1` force.
    if mode == ComputeMode::CpuOnly {
        return ExecutionPlan::cpu(Reason::CpuOnlyMode);
    }

    // 2. Operations with no device path are CPU work, not a strict failure.
    if operation.is_cpu_required() {
        return ExecutionPlan::cpu(Reason::CpuRequiredOperation);
    }

    // 3. An explicit off switch beats every mode. Phase 2 surfaces this to the
    //    user instead of silently ignoring their A/B setting.
    let gate = operation.env_gate(&inputs.env, inputs.platform, inputs.profile);
    if gate == StageGate::DisabledByEnv {
        return ExecutionPlan::cpu(Reason::GpuDisabledByEnv);
    }

    let candidate = select_backend(workload, &inputs.caps);
    let budget = vram_budget_bytes(&inputs.env);

    match mode {
        ComputeMode::CpuOnly => unreachable!("handled above"),
        ComputeMode::Auto => auto(workload, gate, candidate, budget, &inputs.env),
        ComputeMode::PreferGpu => prefer_gpu(workload, gate, candidate, budget),
        ComputeMode::StrictGpu => strict_gpu(workload, candidate, budget),
    }
}

/// Cost seed: bytes the engine may hold resident across all GPU stages.
pub fn vram_budget_bytes(env: &EnvSnapshot) -> u64 {
    env.parse_u64("AURORA_VRAM_BUDGET_MB")
        .filter(|mb| *mb > 0)
        .unwrap_or(DEFAULT_VRAM_BUDGET_MB)
        .saturating_mul(1024 * 1024)
}

fn fits_budget(workload: &Workload, budget: u64) -> bool {
    // An unknown estimate is not a reason to refuse; the phase-2 ledger is the
    // authority for allocations it can actually account for.
    workload.est_vram == 0 || workload.est_vram <= budget
}

/// First usable family in the operation's own preference order.
fn select_backend(workload: &Workload, caps: &Capabilities) -> Option<Backend> {
    for kind in workload.operation.gpu_kinds() {
        let Some(info) = caps.gpu(*kind) else {
            continue;
        };
        if !info.usable() || !info.supports(workload.precision) {
            continue;
        }
        if info.vram_bytes != 0 && workload.est_vram > info.vram_bytes {
            continue;
        }
        return Some(info.backend);
    }
    None
}

/// Existing cost signals, reduced to what a backend decision needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Benefit {
    Yes,
    BelowCrossover,
    ResidencyPenalty,
}

fn benefit(workload: &Workload, env: &EnvSnapshot) -> Benefit {
    if workload.operation == Operation::Blemish
        && workload.residency != Residency::DeviceResident
    {
        return Benefit::ResidencyPenalty;
    }
    let floor = match workload.operation {
        Operation::Develop => env
            .parse_u64("AURORA_DEVELOP_MIN_PIXELS")
            .filter(|pixels| *pixels > 0)
            .unwrap_or(DEVELOP_MIN_PIXELS),
        _ => 0,
    };
    if workload.pixels < floor {
        Benefit::BelowCrossover
    } else {
        Benefit::Yes
    }
}

fn residency_penalty(workload: &Workload) -> bool {
    workload.operation == Operation::Blemish
        && workload.residency != Residency::DeviceResident
}

fn auto(
    workload: &Workload,
    gate: StageGate,
    candidate: Option<Backend>,
    budget: u64,
    env: &EnvSnapshot,
) -> ExecutionPlan {
    match gate {
        StageGate::DisabledByEnv => ExecutionPlan::cpu(Reason::GpuDisabledByEnv),
        StageGate::DefaultOff => ExecutionPlan::cpu(Reason::GpuNotEnabledByDefault),
        StageGate::ForcedOn => match candidate {
            Some(backend) if fits_budget(workload, budget) => {
                ExecutionPlan::gpu(backend, Reason::GpuForcedByEnv, Some(Backend::Cpu))
            }
            Some(_) => ExecutionPlan::cpu(Reason::VramBudgetExceeded),
            None => ExecutionPlan::cpu(Reason::GpuUnavailable),
        },
        StageGate::DefaultOn => match candidate {
            None => ExecutionPlan::cpu(Reason::GpuUnavailable),
            Some(backend) => match benefit(workload, env) {
                Benefit::BelowCrossover => ExecutionPlan::cpu(Reason::GpuNotBeneficial),
                Benefit::ResidencyPenalty => ExecutionPlan::cpu(Reason::ResidencyPenalty),
                Benefit::Yes if !fits_budget(workload, budget) => {
                    ExecutionPlan::cpu(Reason::VramBudgetExceeded)
                }
                Benefit::Yes => {
                    ExecutionPlan::gpu(backend, Reason::GpuPreferred, Some(Backend::Cpu))
                }
            },
        },
    }
}

fn prefer_gpu(
    workload: &Workload,
    gate: StageGate,
    candidate: Option<Backend>,
    budget: u64,
) -> ExecutionPlan {
    // An explicit `=1` is the highest-priority A/B entry: it also overrides
    // the measured residency penalty (the blemish pageable bridge).
    let forced = gate == StageGate::ForcedOn;
    match candidate {
        None => ExecutionPlan::cpu(Reason::GpuUnavailable),
        Some(_) if !forced && residency_penalty(workload) => {
            ExecutionPlan::cpu(Reason::ResidencyPenalty)
        }
        Some(_) if !fits_budget(workload, budget) => {
            ExecutionPlan::cpu(Reason::VramBudgetExceeded)
        }
        Some(backend) if forced => {
            ExecutionPlan::gpu(backend, Reason::GpuForcedByEnv, Some(Backend::Cpu))
        }
        Some(backend) => ExecutionPlan::gpu(backend, Reason::GpuPreferred, Some(Backend::Cpu)),
    }
}

fn strict_gpu(workload: &Workload, candidate: Option<Backend>, budget: u64) -> ExecutionPlan {
    match candidate {
        None => ExecutionPlan::cpu(Reason::StrictGpuUnavailable),
        Some(_) if !fits_budget(workload, budget) => {
            ExecutionPlan::cpu(Reason::StrictGpuBudgetExceeded)
        }
        // Strict means strict: the residency penalty and the cost seeds do not
        // apply, and there is no CPU fallback in the plan.
        Some(backend) => ExecutionPlan::gpu(backend, Reason::GpuRequired, None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::device::GpuInfo;

    fn cuda(device: u32) -> GpuInfo {
        GpuInfo {
            backend: Backend::Cuda(device),
            name: "test-cuda",
            driver_ready: true,
            kernel_compiled: true,
            fp32: true,
            fp16: true,
            vram_bytes: 16 * 1024 * 1024 * 1024,
        }
    }

    fn opencl(device: u32) -> GpuInfo {
        GpuInfo {
            backend: Backend::OpenCl(device),
            name: "test-opencl",
            driver_ready: true,
            kernel_compiled: true,
            fp32: true,
            fp16: false,
            vram_bytes: 8 * 1024 * 1024 * 1024,
        }
    }

    fn no_caps(platform: Platform) -> PolicyInputs {
        PolicyInputs::for_test(platform)
    }

    fn cuda_caps(platform: Platform, env: &[(&str, &str)]) -> PolicyInputs {
        PolicyInputs::for_test(platform)
            .with_env(EnvSnapshot::from_pairs(env))
            .with_caps(Capabilities::none().with_gpu(cuda(0)))
    }

    fn workload(operation: Operation, pixels: u64) -> Workload {
        Workload::new(operation, pixels)
    }

    #[test]
    fn cpu_only_forces_every_operation_to_cpu() {
        let inputs = cuda_caps(Platform::Windows, &[]);
        for operation in Operation::ALL {
            let plan = decide_with(
                &workload(operation, 24_000_000),
                ComputeMode::CpuOnly,
                &inputs,
            );
            assert_eq!(plan.backend, Backend::Cpu, "{operation:?}");
            assert_eq!(plan.reason, Reason::CpuOnlyMode, "{operation:?}");
            assert_eq!(plan.fallback, None, "{operation:?}");
            assert!(!plan.is_strict_violation(), "{operation:?}");
        }
    }

    #[test]
    fn strict_gpu_requires_a_device_for_gpu_operations() {
        let inputs = cuda_caps(Platform::Windows, &[]);
        let plan = decide_with(
            &workload(Operation::Develop, 24_385_536),
            ComputeMode::StrictGpu,
            &inputs,
        );
        assert_eq!(plan.backend, Backend::Cuda(0));
        assert_eq!(plan.reason, Reason::GpuRequired);
        assert_eq!(plan.fallback, None);
        assert!(!plan.is_strict_violation());
    }

    #[test]
    fn strict_gpu_does_not_misjudge_cpu_required_operations() {
        let inputs = cuda_caps(Platform::Windows, &[]);
        for operation in [Operation::Demosaic, Operation::IccTransform] {
            let plan = decide_with(
                &workload(operation, 24_000_000),
                ComputeMode::StrictGpu,
                &inputs,
            );
            assert_eq!(plan.backend, Backend::Cpu, "{operation:?}");
            assert_eq!(plan.reason, Reason::CpuRequiredOperation, "{operation:?}");
            assert_eq!(plan.fallback, None, "{operation:?}");
            assert!(!plan.is_strict_violation(), "{operation:?}");
        }
    }

    #[test]
    fn strict_gpu_without_a_device_is_a_violation() {
        let inputs = no_caps(Platform::Windows);
        let plan = decide_with(
            &workload(Operation::Develop, 24_385_536),
            ComputeMode::StrictGpu,
            &inputs,
        );
        assert_eq!(plan.backend, Backend::Cpu);
        assert_eq!(plan.reason, Reason::StrictGpuUnavailable);
        assert_eq!(plan.fallback, None);
        assert!(plan.is_strict_violation());
    }

    #[test]
    fn explicit_env_off_wins_over_strict_gpu() {
        let inputs = cuda_caps(Platform::Windows, &[("AURORA_DEVELOP_GPU", "0")]);
        let plan = decide_with(
            &workload(Operation::Develop, 24_385_536),
            ComputeMode::StrictGpu,
            &inputs,
        );
        assert_eq!(plan.backend, Backend::Cpu);
        assert_eq!(plan.reason, Reason::GpuDisabledByEnv);
        assert!(!plan.is_strict_violation());
    }

    #[test]
    fn prefer_gpu_ignores_the_develop_crossover_but_keeps_a_fallback() {
        let inputs = cuda_caps(Platform::Windows, &[]);
        let plan = decide_with(
            &workload(Operation::Develop, 65_536),
            ComputeMode::PreferGpu,
            &inputs,
        );
        assert_eq!(plan.backend, Backend::Cuda(0));
        assert_eq!(plan.reason, Reason::GpuPreferred);
        assert_eq!(plan.fallback, Some(Backend::Cpu));
    }

    #[test]
    fn auto_develop_follows_the_crossover_seed() {
        let inputs = cuda_caps(Platform::Windows, &[]);
        for (pixels, expected) in [
            (921_600u64, Reason::GpuNotBeneficial),
            (2_073_600, Reason::GpuPreferred),
            (24_385_536, Reason::GpuPreferred),
        ] {
            let plan = decide_with(&workload(Operation::Develop, pixels), ComputeMode::Auto, &inputs);
            if expected == Reason::GpuNotBeneficial {
                assert_eq!(plan.backend, Backend::Cpu, "{pixels}");
            } else {
                assert_eq!(plan.backend, Backend::Cuda(0), "{pixels}");
                assert_eq!(plan.fallback, Some(Backend::Cpu), "{pixels}");
            }
            assert_eq!(plan.reason, expected, "{pixels}");
        }
    }

    #[test]
    fn auto_develop_env_floor_overrides_the_seed() {
        let inputs = cuda_caps(
            Platform::Windows,
            &[("AURORA_DEVELOP_MIN_PIXELS", "100")],
        );
        let plan = decide_with(
            &workload(Operation::Develop, 921_600),
            ComputeMode::Auto,
            &inputs,
        );
        assert_eq!(plan.backend, Backend::Cuda(0));
        assert_eq!(plan.reason, Reason::GpuPreferred);
    }

    #[test]
    fn auto_pixel_master_switch_disables_every_pixel_operator() {
        let inputs = cuda_caps(Platform::Windows, &[("AURORA_PIXEL_GPU", "0")]);
        for operation in [Operation::PixelVignette, Operation::PixelLcc, Operation::PixelGrain] {
            let plan = decide_with(&workload(operation, 24_000_000), ComputeMode::Auto, &inputs);
            assert_eq!(plan.backend, Backend::Cpu, "{operation:?}");
            assert_eq!(plan.reason, Reason::GpuDisabledByEnv, "{operation:?}");
        }
    }

    #[test]
    fn auto_vignette_default_is_cpu_and_opt_in_forces_gpu() {
        let default_inputs = cuda_caps(Platform::Windows, &[]);
        let plan = decide_with(
            &workload(Operation::PixelVignette, 24_000_000),
            ComputeMode::Auto,
            &default_inputs,
        );
        assert_eq!(plan.backend, Backend::Cpu);
        assert_eq!(plan.reason, Reason::GpuNotEnabledByDefault);

        let forced_inputs = cuda_caps(
            Platform::Windows,
            &[("AURORA_PIXEL_VIGNETTE_GPU", "1")],
        );
        let plan = decide_with(
            &workload(Operation::PixelVignette, 24_000_000),
            ComputeMode::Auto,
            &forced_inputs,
        );
        assert_eq!(plan.backend, Backend::Cuda(0));
        assert_eq!(plan.reason, Reason::GpuForcedByEnv);
        assert_eq!(plan.fallback, Some(Backend::Cpu));
    }

    #[test]
    fn blemish_pageable_bridge_is_a_residency_penalty_unless_forced() {
        let inputs = cuda_caps(Platform::Windows, &[]);
        let pageable = workload(Operation::Blemish, 24_000_000)
            .with_residency(Residency::CpuPageable);
        let plan = decide_with(&pageable, ComputeMode::PreferGpu, &inputs);
        assert_eq!(plan.backend, Backend::Cpu);
        assert_eq!(plan.reason, Reason::ResidencyPenalty);

        let resident = workload(Operation::Blemish, 24_000_000)
            .with_residency(Residency::DeviceResident);
        let plan = decide_with(&resident, ComputeMode::PreferGpu, &inputs);
        assert_eq!(plan.backend, Backend::Cuda(0));
        assert_eq!(plan.reason, Reason::GpuPreferred);

        let forced = cuda_caps(Platform::Windows, &[("AURORA_BLEMISH_GPU", "1")]);
        let plan = decide_with(&pageable, ComputeMode::Auto, &forced);
        assert_eq!(plan.backend, Backend::Cuda(0));
        assert_eq!(plan.reason, Reason::GpuForcedByEnv);
    }

    #[test]
    fn vram_budget_refuses_an_oversized_plan() {
        let inputs = cuda_caps(Platform::Windows, &[]);
        let big = workload(Operation::Develop, 24_385_536)
            .with_est_vram(4 * 1024 * 1024 * 1024);

        let plan = decide_with(&big, ComputeMode::PreferGpu, &inputs);
        assert_eq!(plan.backend, Backend::Cpu);
        assert_eq!(plan.reason, Reason::VramBudgetExceeded);

        let plan = decide_with(&big, ComputeMode::StrictGpu, &inputs);
        assert_eq!(plan.backend, Backend::Cpu);
        assert_eq!(plan.reason, Reason::StrictGpuBudgetExceeded);
        assert!(plan.is_strict_violation());

        let raised = cuda_caps(
            Platform::Windows,
            &[("AURORA_VRAM_BUDGET_MB", "8192")],
        );
        let plan = decide_with(&big, ComputeMode::PreferGpu, &raised);
        assert_eq!(plan.backend, Backend::Cuda(0));
    }

    #[test]
    fn raw_opencl_uses_the_opencl_device_only() {
        let inputs = PolicyInputs::for_test(Platform::Windows)
            .with_caps(Capabilities::none().with_gpu(opencl(0)));
        let plan = decide_with(
            &workload(Operation::RawOpenCl, 24_000_000),
            ComputeMode::Auto,
            &inputs,
        );
        assert_eq!(plan.backend, Backend::OpenCl(0));
        assert_eq!(plan.reason, Reason::GpuPreferred);

        let c1_off = PolicyInputs::for_test(Platform::Windows)
            .with_env(EnvSnapshot::from_pairs(&[("AURORA_C1_RAW", "0")]))
            .with_caps(Capabilities::none().with_gpu(opencl(0)));
        let plan = decide_with(
            &workload(Operation::RawOpenCl, 24_000_000),
            ComputeMode::Auto,
            &c1_off,
        );
        assert_eq!(plan.backend, Backend::Cpu);
        assert_eq!(plan.reason, Reason::GpuDisabledByEnv);
    }

    #[test]
    fn face_input_is_cpu_by_default_and_opt_in_can_force_a_device() {
        let default_inputs = cuda_caps(Platform::Windows, &[]);
        let plan = decide_with(
            &workload(Operation::FaceInput, 4_000_000),
            ComputeMode::Auto,
            &default_inputs,
        );
        assert_eq!(plan.backend, Backend::Cpu);
        assert_eq!(plan.reason, Reason::GpuNotEnabledByDefault);

        let forced = cuda_caps(
            Platform::Windows,
            &[("AURORA_FACE_INPUT_BACKEND", "cuda")],
        );
        let plan = decide_with(
            &workload(Operation::FaceInput, 4_000_000),
            ComputeMode::Auto,
            &forced,
        );
        assert_eq!(plan.backend, Backend::Cuda(0));
        assert_eq!(plan.reason, Reason::GpuForcedByEnv);
    }

    #[test]
    fn ort_provider_cpu_override_disables_inference_gpu() {
        let inputs = cuda_caps(
            Platform::Windows,
            &[("AURORA_ORT_PROVIDERS", "cpu")],
        );
        let plan = decide_with(
            &workload(Operation::OrtInference, 1024 * 1024),
            ComputeMode::PreferGpu,
            &inputs,
        );
        assert_eq!(plan.backend, Backend::Cpu);
        assert_eq!(plan.reason, Reason::GpuDisabledByEnv);
    }

    #[test]
    fn develop_default_is_platform_dependent() {
        let windows = cuda_caps(Platform::Windows, &[]);
        let plan = decide_with(
            &workload(Operation::Develop, 24_385_536),
            ComputeMode::Auto,
            &windows,
        );
        assert_eq!(plan.backend, Backend::Cuda(0));

        let macos = cuda_caps(Platform::MacOs, &[]);
        let plan = decide_with(
            &workload(Operation::Develop, 24_385_536),
            ComputeMode::Auto,
            &macos,
        );
        assert_eq!(plan.backend, Backend::Cpu);
        assert_eq!(plan.reason, Reason::GpuNotEnabledByDefault);
    }

    #[test]
    fn telemetry_json_is_one_line_and_names_the_decision() {
        let inputs = cuda_caps(Platform::Windows, &[]);
        let plan = decide_with(
            &workload(Operation::Develop, 24_385_536),
            ComputeMode::Auto,
            &inputs,
        );
        let line = plan.telemetry_json(&workload(Operation::Develop, 24_385_536), ComputeMode::Auto);
        assert!(!line.contains('\n'));
        assert!(line.contains("\"event\":\"compute_decision\""));
        assert!(line.contains("\"backend\":\"cuda\""));
        assert!(line.contains("\"device\":0"));
        assert!(line.contains("\"reason\":\"gpu_preferred\""));
        assert!(line.contains("\"fallback\":\"cpu\""));
    }
}
