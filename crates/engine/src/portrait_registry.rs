//! Unified portrait effect descriptor, capability registry and runtime availability.
//!
//! This module implements section 11 of the core architecture refactoring plan:
//! a single reviewable descriptor registry for wired portrait effects that
//! strictly separates three orthogonal capability axes:
//!
//! 1. **Static support** ([`StaticSupport`]): whether CPU/CUDA/Metal kernels and
//!    stage wiring are compiled into this binary.
//! 2. **Verified evidence** ([`EvidenceLevel`] / [`EvidenceProfile`]): what has
//!    actually been proven in `aurora_portrait::capability_registry` (synthetic
//!    tests, GPU runs, same-job inputs, main preview, vendor parity, real photos).
//!    Registration or compilation never implies verification.
//! 3. **Runtime availability** ([`RuntimeAvailability`]): whether the current
//!    machine and current photo job actually have the required backend, models,
//!    assets and per-job inputs. Missing models, assets or inputs are always
//!    reported explicitly and never fabricated as available.

use std::collections::BTreeSet;

use aurora_portrait::capability_registry::{self, CapabilityRow, UNVERIFIED};
use serde::{Deserialize, Serialize};

/// Schema identifier for serialized [`PortraitCapabilitySnapshot`] payloads.
pub const SNAPSHOT_SCHEMA: &str = "aurora/portrait-capability-snapshot/1";

/// Tolerance used when checking whether a stored parameter value is neutral.
pub const NEUTRAL_EPSILON: f32 = 1.0e-6;

/// Logical UI and pipeline category for a portrait effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EffectCategory {
    Skin,
    FaceGeometry,
    FacePlump,
    Expression,
    Teeth,
    Eyes,
    Hair,
    Makeup,
}

impl EffectCategory {
    /// All categories in canonical display order.
    pub const ALL: [Self; 8] = [
        Self::Skin,
        Self::FaceGeometry,
        Self::FacePlump,
        Self::Expression,
        Self::Teeth,
        Self::Eyes,
        Self::Hair,
        Self::Makeup,
    ];

    pub const fn display_name_en(self) -> &'static str {
        match self {
            Self::Skin => "Skin Adjustment",
            Self::FaceGeometry => "Face Reshape",
            Self::FacePlump => "Facial Plumpness",
            Self::Expression => "Expression",
            Self::Teeth => "Teeth",
            Self::Eyes => "Eyes",
            Self::Hair => "Hair",
            Self::Makeup => "Makeup",
        }
    }

    pub const fn display_name_zh(self) -> &'static str {
        match self {
            Self::Skin => "皮肤调整",
            Self::FaceGeometry => "面部精修",
            Self::FacePlump => "面部丰盈",
            Self::Expression => "表情管理",
            Self::Teeth => "牙齿美化",
            Self::Eyes => "眼睛增强",
            Self::Hair => "头发调整",
            Self::Makeup => "妆容调整",
        }
    }
}

/// Stable identifier binding a vendor PF numeric ID to its `AdjustmentSet::extra`
/// storage key (`pc:<pf>`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct EffectId {
    pub pf: u32,
    pub store_key: &'static str,
}

impl EffectId {
    pub const fn new(pf: u32, store_key: &'static str) -> Self {
        Self { pf, store_key }
    }

    /// Parse a numeric PF id from a `pc:<id>` storage key.
    pub fn parse_pf_from_store_key(store_key: &str) -> Option<u32> {
        store_key.strip_prefix("pc:")?.parse::<u32>().ok()
    }
}

/// Storage and UI display unit for an effect parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParamUnit {
    /// Stored as `0.0..=1.0` in `AdjustmentSet::extra`; sliders typically display `0..=100`.
    Normalized01,
    /// Stored directly as `-100.0..=100.0` in `AdjustmentSet::extra`.
    SignedPercent,
    /// Raw dimensionless scalar.
    Scalar,
}

impl ParamUnit {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Normalized01 => "normalized_0_1",
            Self::SignedPercent => "signed_percent",
            Self::Scalar => "scalar",
        }
    }
}

/// Numeric bounds, neutral default and unit for an effect parameter.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamRange {
    pub min: f32,
    pub max: f32,
    pub default: f32,
    pub unit: ParamUnit,
}

impl ParamRange {
    pub const fn normalized_01(default: f32) -> Self {
        Self {
            min: 0.0,
            max: 1.0,
            default,
            unit: ParamUnit::Normalized01,
        }
    }

    pub const fn normalized_range(min: f32, max: f32, default: f32) -> Self {
        Self {
            min,
            max,
            default,
            unit: ParamUnit::Normalized01,
        }
    }

    pub const fn signed_percent(default: f32) -> Self {
        Self {
            min: -100.0,
            max: 100.0,
            default,
            unit: ParamUnit::SignedPercent,
        }
    }

    /// True when `min <= default <= max` and all bounds are finite.
    pub fn is_valid(&self) -> bool {
        self.min.is_finite()
            && self.max.is_finite()
            && self.default.is_finite()
            && self.min <= self.default
            && self.default <= self.max
    }

    /// True when `value` falls within `[min, max]`.
    pub fn contains(&self, value: f32) -> bool {
        value.is_finite() && value >= self.min && value <= self.max
    }

    /// True when `value` is effectively equal to the neutral default.
    pub fn is_neutral(&self, value: f32) -> bool {
        (value - self.default).abs() <= NEUTRAL_EPSILON
    }
}

/// Producer, model, asset or per-job mask dependency required by an effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum InputDependency {
    None,
    Landmarks194,
    FaceParsingV2,
    SkinMask,
    HairMask,
    TeethMask,
    EyeMask,
    MakeupBundle,
    SmileWarpAsset,
}

impl InputDependency {
    /// True if this dependency requires an inference model file on disk / loaded runtime.
    pub const fn is_model_backed(self) -> bool {
        matches!(self, Self::Landmarks194 | Self::FaceParsingV2)
    }

    /// True if this dependency requires external/bundled non-model material or warp assets.
    pub const fn is_asset_backed(self) -> bool {
        matches!(self, Self::MakeupBundle | Self::SmileWarpAsset)
    }

    /// True if this dependency requires per-photo same-job producer output.
    pub const fn is_per_job_input(self) -> bool {
        matches!(
            self,
            Self::Landmarks194
                | Self::FaceParsingV2
                | Self::SkinMask
                | Self::HairMask
                | Self::TeethMask
                | Self::EyeMask
        )
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Landmarks194 => "landmarks_194",
            Self::FaceParsingV2 => "face_parsing_v2",
            Self::SkinMask => "skin_mask",
            Self::HairMask => "hair_mask",
            Self::TeethMask => "teeth_mask",
            Self::EyeMask => "eye_mask",
            Self::MakeupBundle => "makeup_bundle",
            Self::SmileWarpAsset => "smile_warp_asset",
        }
    }
}

/// Axis 1: static compile-time and wiring support inside the engine.
///
/// Static support never implies runtime availability or verified quality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StaticSupport {
    pub cpu_reference: bool,
    pub cuda_kernel: bool,
    pub metal_kernel: bool,
    pub wired_to_stage: bool,
    pub roi_supported: bool,
}

impl StaticSupport {
    pub const fn cpu_and_cuda_wired(roi_supported: bool) -> Self {
        Self {
            cpu_reference: true,
            cuda_kernel: true,
            metal_kernel: false,
            wired_to_stage: true,
            roi_supported,
        }
    }

    pub const fn cpu_wired(roi_supported: bool) -> Self {
        Self {
            cpu_reference: true,
            cuda_kernel: false,
            metal_kernel: false,
            wired_to_stage: true,
            roi_supported,
        }
    }

    pub const fn stored_only() -> Self {
        Self {
            cpu_reference: false,
            cuda_kernel: false,
            metal_kernel: false,
            wired_to_stage: false,
            roi_supported: false,
        }
    }

    pub const fn has_any_compute_backend(&self) -> bool {
        self.cpu_reference || self.cuda_kernel || self.metal_kernel
    }

    pub const fn is_stored_only(&self) -> bool {
        !self.wired_to_stage || !self.has_any_compute_backend()
    }
}

/// Axis 2 (summary): highest verified evidence tier (`Ev0..Ev6`).
///
/// Individual evidence gates remain independent inside [`EvidenceProfile`];
/// this enum provides a compact ordered tier for reporting and filtering.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EvidenceLevel {
    /// No verified evidence recorded (`未验证` or unregistered).
    Ev0Unverified = 0,
    /// Kernel exists in source / compiles statically.
    Ev1KernelExists = 1,
    /// CPU reference implementation verified on synthetic fixtures.
    Ev2CpuReference = 2,
    /// Real GPU kernel execution verified.
    Ev3GpuKernel = 3,
    /// Same-job producer inputs verified end-to-end.
    Ev4SameJobInputs = 4,
    /// Wired and verified in the application main preview.
    Ev5MainPreview = 5,
    /// Vendor parity and/or real-photo acceptance verified.
    Ev6VendorParity = 6,
}

impl EvidenceLevel {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ev0Unverified => "ev0_unverified",
            Self::Ev1KernelExists => "ev1_kernel_exists",
            Self::Ev2CpuReference => "ev2_cpu_reference",
            Self::Ev3GpuKernel => "ev3_gpu_kernel",
            Self::Ev4SameJobInputs => "ev4_same_job_inputs",
            Self::Ev5MainPreview => "ev5_main_preview",
            Self::Ev6VendorParity => "ev6_vendor_parity",
        }
    }
}

/// Axis 2 (detailed): independent seven-gate evidence flags aggregated from
/// `aurora_portrait::capability_registry`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EvidenceProfile {
    pub registered_in_tsv: bool,
    pub kernel_exists: bool,
    pub cpu_reference: bool,
    pub gpu_kernel: bool,
    pub same_job_inputs: bool,
    pub main_preview: bool,
    pub vendor_parity: bool,
    pub real_photo_accept: bool,
}

impl EvidenceProfile {
    pub const fn unverified() -> Self {
        Self {
            registered_in_tsv: false,
            kernel_exists: false,
            cpu_reference: false,
            gpu_kernel: false,
            same_job_inputs: false,
            main_preview: false,
            vendor_parity: false,
            real_photo_accept: false,
        }
    }

    pub fn from_capability_row(row: &CapabilityRow<'_>) -> Self {
        Self {
            registered_in_tsv: true,
            kernel_exists: row.has_evidence(0),
            cpu_reference: row.has_evidence(1),
            gpu_kernel: row.has_evidence(2),
            same_job_inputs: row.has_evidence(3),
            main_preview: row.has_evidence(4),
            vendor_parity: row.has_evidence(5),
            real_photo_accept: row.has_evidence(6),
        }
    }

    /// Highest reached evidence tier across the independent gates.
    pub const fn highest_level(&self) -> EvidenceLevel {
        if self.vendor_parity || self.real_photo_accept {
            EvidenceLevel::Ev6VendorParity
        } else if self.main_preview {
            EvidenceLevel::Ev5MainPreview
        } else if self.same_job_inputs {
            EvidenceLevel::Ev4SameJobInputs
        } else if self.gpu_kernel {
            EvidenceLevel::Ev3GpuKernel
        } else if self.cpu_reference {
            EvidenceLevel::Ev2CpuReference
        } else if self.kernel_exists {
            EvidenceLevel::Ev1KernelExists
        } else {
            EvidenceLevel::Ev0Unverified
        }
    }

    /// Count of independent evidence gates currently satisfied (`0..=7`).
    pub const fn reached_count(&self) -> usize {
        (self.kernel_exists as usize)
            + (self.cpu_reference as usize)
            + (self.gpu_kernel as usize)
            + (self.same_job_inputs as usize)
            + (self.main_preview as usize)
            + (self.vendor_parity as usize)
            + (self.real_photo_accept as usize)
    }
}

/// Axis 3: actual runtime availability for the current host and photo job.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeAvailability {
    /// All required backends, models, assets and per-job inputs are present.
    Available,
    /// One or more required inference models are missing or unloaded.
    MissingModel { required: Vec<InputDependency> },
    /// One or more required material/warp assets are missing.
    MissingAsset { required: Vec<InputDependency> },
    /// One or more required per-photo same-job inputs are missing (e.g. no face/mask detected).
    MissingInput { required: Vec<InputDependency> },
    /// No compatible compute backend is available under the current runtime policy.
    BackendUnavailable { reason: String },
    /// Parameter is only persisted in `AdjustmentSet::extra` and not wired to a live stage.
    StoredOnly,
}

impl RuntimeAvailability {
    pub const fn is_available(&self) -> bool {
        matches!(self, Self::Available)
    }

    pub const fn reason_code(&self) -> &'static str {
        match self {
            Self::Available => "available",
            Self::MissingModel { .. } => "missing_model",
            Self::MissingAsset { .. } => "missing_asset",
            Self::MissingInput { .. } => "missing_input",
            Self::BackendUnavailable { .. } => "backend_unavailable",
            Self::StoredOnly => "stored_only",
        }
    }
}

/// Runtime environment and per-job producer probe context passed to
/// [`EffectDescriptor::evaluate_availability`] and [`capability_snapshot`].
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeProbeContext {
    pub cpu_available: bool,
    pub cuda_available: bool,
    pub metal_available: bool,
    /// When true, CPU fallback is disallowed and a compiled GPU backend must be active.
    pub strict_gpu: bool,
    /// Model-backed dependencies whose model files/sessions are present and ready.
    pub available_models: BTreeSet<InputDependency>,
    /// Asset-backed dependencies whose external/bundled assets are present.
    pub available_assets: BTreeSet<InputDependency>,
    /// Per-photo same-job inputs that were successfully produced for the active job.
    pub available_inputs: BTreeSet<InputDependency>,
}

impl RuntimeProbeContext {
    /// Host with CPU execution enabled, no GPU, and no models/assets/inputs ready.
    pub fn cpu_only_empty() -> Self {
        Self {
            cpu_available: true,
            ..Self::default()
        }
    }

    /// Fully populated context with CPU+CUDA and all known models, assets and inputs ready.
    pub fn fully_ready() -> Self {
        let mut ctx = Self {
            cpu_available: true,
            cuda_available: true,
            metal_available: false,
            strict_gpu: false,
            available_models: BTreeSet::new(),
            available_assets: BTreeSet::new(),
            available_inputs: BTreeSet::new(),
        };
        for dep in [InputDependency::Landmarks194, InputDependency::FaceParsingV2] {
            ctx.available_models.insert(dep);
        }
        for dep in [
            InputDependency::MakeupBundle,
            InputDependency::SmileWarpAsset,
        ] {
            ctx.available_assets.insert(dep);
        }
        for dep in [
            InputDependency::Landmarks194,
            InputDependency::FaceParsingV2,
            InputDependency::SkinMask,
            InputDependency::HairMask,
            InputDependency::TeethMask,
            InputDependency::EyeMask,
        ] {
            ctx.available_inputs.insert(dep);
        }
        ctx
    }
}

/// Static descriptor for one registered portrait effect.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectDescriptor {
    pub id: EffectId,
    pub category: EffectCategory,
    pub section: &'static str,
    pub label_en: &'static str,
    pub label_zh: &'static str,
    pub range: ParamRange,
    pub dependencies: &'static [InputDependency],
    pub static_support: StaticSupport,
}

impl EffectDescriptor {
    /// Look up the matching row in `aurora_portrait::capability_registry` (if any).
    pub fn capability_row(&self) -> Option<CapabilityRow<'static>> {
        capability_registry::for_pf(self.id.pf)
    }

    /// Aggregate independent evidence flags from `aurora_portrait::capability_registry`.
    pub fn evidence_profile(&self) -> EvidenceProfile {
        self.capability_row()
            .map(|row| EvidenceProfile::from_capability_row(&row))
            .unwrap_or_else(EvidenceProfile::unverified)
    }

    /// Highest verified evidence tier (`Ev0..Ev6`) from the capability registry.
    pub fn evidence_level(&self) -> EvidenceLevel {
        self.evidence_profile().highest_level()
    }

    /// Effective static support enriched with backend/ROI metadata from the
    /// `aurora_portrait::capability_registry` row when present.
    pub fn effective_static_support(&self) -> StaticSupport {
        let mut support = self.static_support;
        if let Some(row) = self.capability_row() {
            if is_declared_cell(row.cpu_backend) {
                support.cpu_reference = true;
            }
            if is_declared_cell(row.gpu_backend) && row.gpu_backend.to_ascii_lowercase().contains("cuda") {
                support.cuda_kernel = true;
            }
            if is_declared_cell(row.gpu_backend) && row.gpu_backend.to_ascii_lowercase().contains("metal") {
                support.metal_kernel = true;
            }
            if is_declared_cell(row.roi_support) {
                let lower = row.roi_support.to_ascii_lowercase();
                if lower.contains("yes") || lower.contains("roi") || lower.contains("full") {
                    support.roi_supported = true;
                }
            }
        }
        support
    }

    /// True when `adjustments.extra` contains a non-neutral value for this effect.
    pub fn is_active(&self, adjustments: &crate::AdjustmentSet) -> bool {
        adjustments
            .extra
            .get(self.id.store_key)
            .is_some_and(|&value| !self.range.is_neutral(value))
    }

    /// Evaluate Axis 3 (runtime availability) against `ctx`.
    ///
    /// This deliberately ignores [`EvidenceLevel`] so historical verification can
    /// never fabricate runtime readiness when a model, asset, backend or same-job
    /// input is missing.
    pub fn evaluate_availability(&self, ctx: &RuntimeProbeContext) -> RuntimeAvailability {
        let support = self.effective_static_support();
        if support.is_stored_only() {
            return RuntimeAvailability::StoredOnly;
        }

        let gpu_ready =
            (support.cuda_kernel && ctx.cuda_available) || (support.metal_kernel && ctx.metal_available);
        let cpu_ready = support.cpu_reference && ctx.cpu_available;

        if ctx.strict_gpu {
            if !gpu_ready {
                return RuntimeAvailability::BackendUnavailable {
                    reason: "strict_gpu enabled but no compiled and active GPU backend is available"
                        .to_owned(),
                };
            }
        } else if !gpu_ready && !cpu_ready {
            return RuntimeAvailability::BackendUnavailable {
                reason: "no compatible CPU or GPU backend is available on this host".to_owned(),
            };
        }

        let missing_models: Vec<InputDependency> = self
            .dependencies
            .iter()
            .copied()
            .filter(|dep| dep.is_model_backed() && !ctx.available_models.contains(dep))
            .collect();
        if !missing_models.is_empty() {
            return RuntimeAvailability::MissingModel {
                required: missing_models,
            };
        }

        let missing_assets: Vec<InputDependency> = self
            .dependencies
            .iter()
            .copied()
            .filter(|dep| dep.is_asset_backed() && !ctx.available_assets.contains(dep))
            .collect();
        if !missing_assets.is_empty() {
            return RuntimeAvailability::MissingAsset {
                required: missing_assets,
            };
        }

        let missing_inputs: Vec<InputDependency> = self
            .dependencies
            .iter()
            .copied()
            .filter(|dep| dep.is_per_job_input() && !ctx.available_inputs.contains(dep))
            .collect();
        if !missing_inputs.is_empty() {
            return RuntimeAvailability::MissingInput {
                required: missing_inputs,
            };
        }

        RuntimeAvailability::Available
    }

    /// Build a serializable snapshot entry combining all three capability axes.
    pub fn snapshot_entry(&self, ctx: &RuntimeProbeContext) -> EffectCapabilityEntry {
        let evidence = self.evidence_profile();
        EffectCapabilityEntry {
            pf: self.id.pf,
            store_key: self.id.store_key.to_owned(),
            category: self.category,
            section: self.section.to_owned(),
            label_en: self.label_en.to_owned(),
            label_zh: self.label_zh.to_owned(),
            range: self.range,
            dependencies: self.dependencies.to_vec(),
            static_support: self.effective_static_support(),
            evidence_level: evidence.highest_level(),
            evidence,
            availability: self.evaluate_availability(ctx),
        }
    }
}

fn is_declared_cell(cell: &str) -> bool {
    let trimmed = cell.trim();
    !trimmed.is_empty()
        && trimmed != UNVERIFIED
        && trimmed != "-"
        && !trimmed.eq_ignore_ascii_case("none")
        && !trimmed.eq_ignore_ascii_case("n/a")
}

/// Owned, serializable per-effect capability report inside [`PortraitCapabilitySnapshot`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EffectCapabilityEntry {
    pub pf: u32,
    pub store_key: String,
    pub category: EffectCategory,
    pub section: String,
    pub label_en: String,
    pub label_zh: String,
    pub range: ParamRange,
    pub dependencies: Vec<InputDependency>,
    pub static_support: StaticSupport,
    pub evidence_level: EvidenceLevel,
    pub evidence: EvidenceProfile,
    pub availability: RuntimeAvailability,
}

/// Unified snapshot returned to `ImageEngine` callers and the GPUI frontend.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PortraitCapabilitySnapshot {
    pub schema: String,
    pub registry_schema: String,
    pub total_effects: usize,
    pub available_effects: usize,
    pub effects: Vec<EffectCapabilityEntry>,
}

impl PortraitCapabilitySnapshot {
    pub fn by_store_key(&self, store_key: &str) -> Option<&EffectCapabilityEntry> {
        self.effects.iter().find(|e| e.store_key == store_key)
    }

    pub fn by_pf(&self, pf: u32) -> Option<&EffectCapabilityEntry> {
        self.effects.iter().find(|e| e.pf == pf)
    }

    pub fn by_category(&self, category: EffectCategory) -> Vec<&EffectCapabilityEntry> {
        self.effects
            .iter()
            .filter(|e| e.category == category)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Canonical dependency slices
// ---------------------------------------------------------------------------

const DEPS_SKIN: &[InputDependency] = &[InputDependency::FaceParsingV2, InputDependency::SkinMask];
const DEPS_WRINKLE: &[InputDependency] = &[
    InputDependency::Landmarks194,
    InputDependency::FaceParsingV2,
    InputDependency::SkinMask,
];
const DEPS_LANDMARKS: &[InputDependency] = &[InputDependency::Landmarks194];
const DEPS_EXPRESSION: &[InputDependency] = &[
    InputDependency::Landmarks194,
    InputDependency::SmileWarpAsset,
];
const DEPS_TEETH: &[InputDependency] = &[InputDependency::FaceParsingV2, InputDependency::TeethMask];
const DEPS_EYES: &[InputDependency] = &[
    InputDependency::Landmarks194,
    InputDependency::FaceParsingV2,
    InputDependency::EyeMask,
];
const DEPS_HAIR: &[InputDependency] = &[InputDependency::FaceParsingV2, InputDependency::HairMask];
const DEPS_MAKEUP: &[InputDependency] = &[
    InputDependency::Landmarks194,
    InputDependency::FaceParsingV2,
    InputDependency::MakeupBundle,
];
const DEPS_LIP_ENHANCE: &[InputDependency] = &[
    InputDependency::FaceParsingV2,
    InputDependency::MakeupBundle,
];

macro_rules! skin_desc {
    ($pf:expr, $key:expr, $section:expr, $en:expr, $zh:expr, $default:expr, $deps:expr) => {
        EffectDescriptor {
            id: EffectId::new($pf, $key),
            category: EffectCategory::Skin,
            section: $section,
            label_en: $en,
            label_zh: $zh,
            range: ParamRange::normalized_01($default),
            dependencies: $deps,
            static_support: StaticSupport::cpu_and_cuda_wired(true),
        }
    };
}

macro_rules! face_geom_desc {
    ($pf:expr, $key:expr, $section:expr, $en:expr, $zh:expr) => {
        EffectDescriptor {
            id: EffectId::new($pf, $key),
            category: EffectCategory::FaceGeometry,
            section: $section,
            label_en: $en,
            label_zh: $zh,
            range: ParamRange::signed_percent(0.0),
            dependencies: DEPS_LANDMARKS,
            static_support: StaticSupport::cpu_and_cuda_wired(false),
        }
    };
}

macro_rules! plump_desc {
    ($pf:expr, $key:expr, $en:expr, $zh:expr) => {
        EffectDescriptor {
            id: EffectId::new($pf, $key),
            category: EffectCategory::FacePlump,
            section: "Facial Plumpness",
            label_en: $en,
            label_zh: $zh,
            range: ParamRange::signed_percent(0.0),
            dependencies: DEPS_LANDMARKS,
            static_support: StaticSupport::cpu_and_cuda_wired(false),
        }
    };
}

macro_rules! makeup_desc {
    ($pf:expr, $key:expr, $section:expr, $en:expr, $zh:expr, $default:expr, $deps:expr) => {
        EffectDescriptor {
            id: EffectId::new($pf, $key),
            category: EffectCategory::Makeup,
            section: $section,
            label_en: $en,
            label_zh: $zh,
            range: ParamRange::normalized_01($default),
            dependencies: $deps,
            static_support: StaticSupport::cpu_and_cuda_wired(true),
        }
    };
}

/// Canonical registry of all live-wired portrait effects.
///
/// Aligned with the wired groups in `crates/app/src/tools/portrait.rs`, with
/// duplicate entries deduplicated (notably `pc:8033` is registered once under
/// [`EffectCategory::Expression`]).
pub const WIRED_EFFECT_DESCRIPTORS: &[EffectDescriptor] = &[
    // ── Skin: Skin tone (7) ────────────────────────────────────────────────
    skin_desc!(21015, "pc:21015", "Skin tone", "Skin hue", "肤色色相", 0.5, DEPS_SKIN),
    skin_desc!(30096, "pc:30096", "Skin tone", "Skin temperature", "色调", 0.5, DEPS_SKIN),
    skin_desc!(90005, "pc:90005", "Skin tone", "Whitening", "美白", 0.0, DEPS_SKIN),
    skin_desc!(90007, "pc:90007", "Skin tone", "Rosy", "红润", 0.0, DEPS_SKIN),
    skin_desc!(90004, "pc:90004", "Skin tone", "Skin radiance", "皮肤透亮", 0.0, DEPS_SKIN),
    skin_desc!(21105, "pc:21105", "Skin tone", "AI body skin tone unify", "AI 身体肤色统一", 0.0, DEPS_SKIN),
    skin_desc!(90086, "pc:90086", "Skin tone", "Body skin tone unify", "肤色统一（身体）", 0.0, DEPS_SKIN),
    // ── Skin: Texture (3) ──────────────────────────────────────────────────
    skin_desc!(8068, "pc:8068", "Texture", "Skin gloss", "皮肤光泽", 0.5, DEPS_SKIN),
    EffectDescriptor {
        id: EffectId::new(8069, "pc:8069"),
        category: EffectCategory::Skin,
        section: "Texture",
        label_en: "Skin texture (smoothing)",
        label_zh: "肤质（柔化）",
        // Positive texture enhancement (> 0.5) is not wired yet; 0.5 is neutral.
        range: ParamRange::normalized_range(0.0, 0.5, 0.5),
        dependencies: DEPS_SKIN,
        static_support: StaticSupport::cpu_and_cuda_wired(true),
    },
    skin_desc!(8070, "pc:8070", "Texture", "Skin clarity", "清晰度", 0.0, DEPS_SKIN),
    // ── Skin: Smoothing & Neutral Grey (6) ─────────────────────────────────
    skin_desc!(30021, "pc:30021", "Smoothing & Neutral Grey", "Face smooth (HLF)", "中性灰/高低频磨皮", 0.0, DEPS_SKIN),
    skin_desc!(30022, "pc:30022", "Smoothing & Neutral Grey", "Body smooth (HLF)", "全身磨皮", 0.0, DEPS_SKIN),
    skin_desc!(30013, "pc:30013", "Smoothing & Neutral Grey", "Neutral grey flat", "中性灰平", 0.0, DEPS_SKIN),
    skin_desc!(30014, "pc:30014", "Smoothing & Neutral Grey", "Neutral grey 3D", "中性灰 3D", 0.0, DEPS_SKIN),
    skin_desc!(90002, "pc:90002", "Smoothing & Neutral Grey", "Moisture smooth (face)", "水润磨皮（脸部）", 0.0, DEPS_SKIN),
    skin_desc!(90010, "pc:90010", "Smoothing & Neutral Grey", "Moisture smooth (body)", "水润磨皮（身体）", 0.0, DEPS_SKIN),
    // ── Skin: Wrinkles (7) ─────────────────────────────────────────────────
    skin_desc!(90023, "pc:90023", "Wrinkles", "Eye wrinkles", "祛眼纹", 0.0, DEPS_WRINKLE),
    skin_desc!(90024, "pc:90024", "Wrinkles", "Forehead wrinkles", "祛抬头纹", 0.0, DEPS_WRINKLE),
    skin_desc!(21024, "pc:21024", "Wrinkles", "Glabella wrinkles", "祛眉间纹", 0.0, DEPS_WRINKLE),
    skin_desc!(21025, "pc:21025", "Wrinkles", "Puppet wrinkles", "祛木偶纹", 0.0, DEPS_WRINKLE),
    skin_desc!(21069, "pc:21069", "Wrinkles", "Cheek wrinkles", "祛面颊纹", 0.0, DEPS_WRINKLE),
    skin_desc!(21070, "pc:21070", "Wrinkles", "Nose back wrinkles", "祛鼻背纹", 0.0, DEPS_WRINKLE),
    skin_desc!(21071, "pc:21071", "Wrinkles", "Perioral wrinkles", "祛口周纹", 0.0, DEPS_WRINKLE),
    // ── Expression (1) ─────────────────────────────────────────────────────
    EffectDescriptor {
        id: EffectId::new(8033, "pc:8033"),
        category: EffectCategory::Expression,
        section: "Expression",
        label_en: "Smile mouth corner",
        label_zh: "微笑嘴角",
        range: ParamRange::normalized_01(0.0),
        dependencies: DEPS_EXPRESSION,
        static_support: StaticSupport::cpu_and_cuda_wired(false),
    },
    // ── Makeup (15) ────────────────────────────────────────────────────────
    makeup_desc!(8, "pc:8", "Stereo contour", "Stereo shadow", "立体阴影", 0.0, DEPS_MAKEUP),
    makeup_desc!(9, "pc:9", "Stereo contour", "Stereo highlight", "立体高光", 0.0, DEPS_MAKEUP),
    makeup_desc!(100, "pc:100", "Brows & Eyes", "Eyebrow (L)", "眉妆（左）", 0.0, DEPS_MAKEUP),
    makeup_desc!(117, "pc:117", "Brows & Eyes", "Eyebrow (R)", "眉妆（右）", 0.0, DEPS_MAKEUP),
    makeup_desc!(30007, "pc:30007", "Brows & Eyes", "Eyeshadow", "眼影", 0.0, DEPS_MAKEUP),
    makeup_desc!(103, "pc:103", "Brows & Eyes", "Upper eyeliner", "上眼线", 0.0, DEPS_MAKEUP),
    makeup_desc!(105, "pc:105", "Brows & Eyes", "Upper eyelash", "上睫毛", 0.0, DEPS_MAKEUP),
    makeup_desc!(118, "pc:118", "Brows & Eyes", "Lower eyelash", "下睫毛", 0.0, DEPS_MAKEUP),
    makeup_desc!(111, "pc:111", "Brows & Eyes", "Contact lenses", "美瞳", 0.0, DEPS_MAKEUP),
    makeup_desc!(106, "pc:106", "Blusher & Contour", "Blusher", "腮红", 0.0, DEPS_MAKEUP),
    makeup_desc!(116, "pc:116", "Blusher & Contour", "Contour", "修容", 0.0, DEPS_MAKEUP),
    makeup_desc!(107, "pc:107", "Lips", "Lipstick", "口红", 0.0, DEPS_MAKEUP),
    makeup_desc!(90032, "pc:90032", "Lips", "Lip gloss", "唇彩", 0.0, DEPS_MAKEUP),
    makeup_desc!(90021, "pc:90021", "Lips", "Lip makeup enhance", "唇妆增强", 0.5, DEPS_LIP_ENHANCE),
    makeup_desc!(115, "pc:115", "Sticker", "Face decoration", "面部贴纸", 0.0, DEPS_MAKEUP),
    // ── Eyes (2) ───────────────────────────────────────────────────────────
    EffectDescriptor {
        id: EffectId::new(100004, "pc:100004"),
        category: EffectCategory::Eyes,
        section: "Eyes",
        label_en: "Red blood removal",
        label_zh: "去红血丝",
        range: ParamRange::normalized_01(0.0),
        dependencies: DEPS_EYES,
        static_support: StaticSupport::cpu_and_cuda_wired(true),
    },
    EffectDescriptor {
        id: EffectId::new(101, "pc:101"),
        category: EffectCategory::Eyes,
        section: "Eyes",
        label_en: "Silkworm",
        label_zh: "卧蚕",
        range: ParamRange::normalized_01(0.0),
        dependencies: DEPS_EYES,
        static_support: StaticSupport::cpu_and_cuda_wired(true),
    },
    // ── Hair (1) ───────────────────────────────────────────────────────────
    EffectDescriptor {
        id: EffectId::new(30009, "pc:30009"),
        category: EffectCategory::Hair,
        section: "Hair",
        label_en: "Hair color unify",
        label_zh: "统一发色",
        range: ParamRange::normalized_01(0.0),
        dependencies: DEPS_HAIR,
        static_support: StaticSupport::cpu_and_cuda_wired(true),
    },
    // ── Teeth (3) ──────────────────────────────────────────────────────────
    EffectDescriptor {
        id: EffectId::new(108, "pc:108"),
        category: EffectCategory::Teeth,
        section: "Teeth",
        label_en: "Teeth whitening",
        label_zh: "美牙/牙齿美白",
        range: ParamRange::normalized_01(0.0),
        dependencies: DEPS_TEETH,
        static_support: StaticSupport::cpu_and_cuda_wired(true),
    },
    EffectDescriptor {
        id: EffectId::new(109, "pc:109"),
        category: EffectCategory::Teeth,
        section: "Teeth",
        label_en: "Teeth brightness",
        label_zh: "牙齿亮白",
        range: ParamRange::normalized_01(0.0),
        dependencies: DEPS_TEETH,
        static_support: StaticSupport::cpu_and_cuda_wired(true),
    },
    EffectDescriptor {
        id: EffectId::new(110, "pc:110"),
        category: EffectCategory::Teeth,
        section: "Teeth",
        label_en: "Teeth de-yellow",
        label_zh: "牙齿去黄",
        range: ParamRange::normalized_01(0.0),
        dependencies: DEPS_TEETH,
        static_support: StaticSupport::cpu_and_cuda_wired(true),
    },
    // ── Face Geometry: Left/Right pairs (40) ───────────────────────────────
    face_geom_desc!(30092, "pc:30092", "Face pairs", "Face width (L)", "窄脸（左）"),
    face_geom_desc!(30093, "pc:30093", "Face pairs", "Face width (R)", "窄脸（右）"),
    face_geom_desc!(247, "pc:247", "Face pairs", "V-face (L)", "V 脸（左）"),
    face_geom_desc!(248, "pc:248", "Face pairs", "V-face (R)", "V 脸（右）"),
    face_geom_desc!(242, "pc:242", "Face pairs", "Sharp chin (L)", "尖下巴（左）"),
    face_geom_desc!(243, "pc:243", "Face pairs", "Sharp chin (R)", "尖下巴（右）"),
    face_geom_desc!(8023, "pc:8023", "Face pairs", "Jawline (L)", "下颌角（左）"),
    face_geom_desc!(8024, "pc:8024", "Face pairs", "Jawline (R)", "下颌角（右）"),
    face_geom_desc!(8027, "pc:8027", "Face pairs", "Cheekbone (L)", "颧骨（左）"),
    face_geom_desc!(8028, "pc:8028", "Face pairs", "Cheekbone (R)", "颧骨（右）"),
    face_geom_desc!(8029, "pc:8029", "Face pairs", "Temple (L)", "太阳穴（左）"),
    face_geom_desc!(8030, "pc:8030", "Face pairs", "Temple (R)", "太阳穴（右）"),
    face_geom_desc!(30080, "pc:30080", "Face pairs", "Forehead width (L)", "额头宽窄（左）"),
    face_geom_desc!(30081, "pc:30081", "Face pairs", "Forehead width (R)", "额头宽窄（右）"),
    face_geom_desc!(30082, "pc:30082", "Face pairs", "Forehead height (L)", "额头高低（左）"),
    face_geom_desc!(30083, "pc:30083", "Face pairs", "Forehead height (R)", "额头高低（右）"),
    face_geom_desc!(8019, "pc:8019", "Face pairs", "Eye width (L)", "眼宽（左）"),
    face_geom_desc!(8020, "pc:8020", "Face pairs", "Eye width (R)", "眼宽（右）"),
    face_geom_desc!(8013, "pc:8013", "Face pairs", "Eye height (L)", "眼高（左）"),
    face_geom_desc!(8014, "pc:8014", "Face pairs", "Eye height (R)", "眼高（右）"),
    face_geom_desc!(8015, "pc:8015", "Face pairs", "Eye distance (L)", "眼距（左）"),
    face_geom_desc!(8016, "pc:8016", "Face pairs", "Eye distance (R)", "眼距（右）"),
    face_geom_desc!(8017, "pc:8017", "Face pairs", "Eye tilt (L)", "眼角倾斜（左）"),
    face_geom_desc!(8018, "pc:8018", "Face pairs", "Eye tilt (R)", "眼角倾斜（右）"),
    face_geom_desc!(8011, "pc:8011", "Face pairs", "Pupil size (L)", "眼瞳大小（左）"),
    face_geom_desc!(8012, "pc:8012", "Face pairs", "Pupil size (R)", "眼瞳大小（右）"),
    face_geom_desc!(30073, "pc:30073", "Face pairs", "Lower eyelid (L)", "眼睑下至（左）"),
    face_geom_desc!(30079, "pc:30079", "Face pairs", "Lower eyelid (R)", "眼睑下至（右）"),
    face_geom_desc!(8001, "pc:8001", "Face pairs", "Brow thickness (L)", "眉粗细（左）"),
    face_geom_desc!(8002, "pc:8002", "Face pairs", "Brow thickness (R)", "眉粗细（右）"),
    face_geom_desc!(8005, "pc:8005", "Face pairs", "Brow distance (L)", "眉距（左）"),
    face_geom_desc!(8006, "pc:8006", "Face pairs", "Brow distance (R)", "眉距（右）"),
    face_geom_desc!(8009, "pc:8009", "Face pairs", "Brow shape (L)", "眉形（左）"),
    face_geom_desc!(8010, "pc:8010", "Face pairs", "Brow shape (R)", "眉形（右）"),
    face_geom_desc!(8021, "pc:8021", "Face pairs", "Brow position (L)", "眉位（左）"),
    face_geom_desc!(8022, "pc:8022", "Face pairs", "Brow position (R)", "眉位（右）"),
    face_geom_desc!(44124, "pc:44124", "Face pairs", "Brow head length (L)", "眉头长短（左）"),
    face_geom_desc!(48019, "pc:48019", "Face pairs", "Brow head length (R)", "眉头长短（右）"),
    face_geom_desc!(41922, "pc:41922", "Face pairs", "Brow tail length (L)", "眉尾长短（左）"),
    face_geom_desc!(49221, "pc:49221", "Face pairs", "Brow tail length (R)", "眉尾长短（右）"),
    // ── Face Geometry: Single controls (2; pc:8033 lives in Expression) ────
    face_geom_desc!(202, "pc:202", "Face single", "Hairline", "发际线"),
    face_geom_desc!(232, "pc:232", "Face single", "Skull top enhance", "高颅顶"),
    // ── Face Plump (8) ─────────────────────────────────────────────────────
    plump_desc!(30210, "pc:30210", "Full face plump", "全脸丰盈"),
    plump_desc!(30191, "pc:30191", "Forehead plump", "额头丰盈"),
    plump_desc!(30194, "pc:30194", "Mid-face plump", "面中丰盈"),
    plump_desc!(30197, "pc:30197", "Cheek plump", "面颊丰盈"),
    plump_desc!(30198, "pc:30198", "Nasal base plump", "鼻基底丰盈"),
    plump_desc!(30203, "pc:30203", "Brow arch plump", "眉弓丰盈"),
    plump_desc!(30204, "pc:30204", "Chin plump", "下巴丰盈"),
    plump_desc!(30207, "pc:30207", "Mouth corner lift", "嘴角上扬"),
];

/// Return all registered wired portrait effect descriptors.
pub const fn descriptors() -> &'static [EffectDescriptor] {
    WIRED_EFFECT_DESCRIPTORS
}

/// Look up a registered effect descriptor by its `pc:<id>` storage key.
pub fn descriptor_by_key(store_key: &str) -> Option<&'static EffectDescriptor> {
    WIRED_EFFECT_DESCRIPTORS
        .iter()
        .find(|desc| desc.id.store_key == store_key)
}

/// Look up a registered effect descriptor by its vendor PF number.
pub fn descriptor_by_pf(pf: u32) -> Option<&'static EffectDescriptor> {
    WIRED_EFFECT_DESCRIPTORS
        .iter()
        .find(|desc| desc.id.pf == pf)
}

/// Iterate over all descriptors belonging to `category`.
pub fn descriptors_in_category(
    category: EffectCategory,
) -> impl Iterator<Item = &'static EffectDescriptor> {
    WIRED_EFFECT_DESCRIPTORS
        .iter()
        .filter(move |desc| desc.category == category)
}

/// True if `store_key` is a registered wired portrait effect.
pub fn is_registered_key(store_key: &str) -> bool {
    descriptor_by_key(store_key).is_some()
}

/// Collect all registered effects that have a non-neutral value in `adjustments.extra`.
pub fn active_effects(adjustments: &crate::AdjustmentSet) -> Vec<&'static EffectDescriptor> {
    WIRED_EFFECT_DESCRIPTORS
        .iter()
        .filter(|desc| desc.is_active(adjustments))
        .collect()
}

/// True when any registered portrait effect is non-neutral in `adjustments.extra`.
pub fn has_active_portrait_effect(adjustments: &crate::AdjustmentSet) -> bool {
    WIRED_EFFECT_DESCRIPTORS
        .iter()
        .any(|desc| desc.is_active(adjustments))
}

/// Deduplicated, deterministic list of producer/asset/input dependencies needed
/// for the currently non-neutral portrait effects in `adjustments.extra`.
///
/// Neutral parameters produce an empty list so idle producers are never started.
pub fn required_producers(adjustments: &crate::AdjustmentSet) -> Vec<InputDependency> {
    let mut deps = BTreeSet::new();
    for desc in WIRED_EFFECT_DESCRIPTORS {
        if desc.is_active(adjustments) {
            for &dep in desc.dependencies {
                if dep != InputDependency::None {
                    deps.insert(dep);
                }
            }
        }
    }
    deps.into_iter().collect()
}

/// Build a unified capability snapshot for `ImageEngine` and frontend UI consumption.
pub fn capability_snapshot(ctx: &RuntimeProbeContext) -> PortraitCapabilitySnapshot {
    let effects: Vec<EffectCapabilityEntry> = WIRED_EFFECT_DESCRIPTORS
        .iter()
        .map(|desc| desc.snapshot_entry(ctx))
        .collect();
    let available_effects = effects.iter().filter(|e| e.availability.is_available()).count();
    PortraitCapabilitySnapshot {
        schema: SNAPSHOT_SCHEMA.to_owned(),
        registry_schema: capability_registry::REGISTRY_SCHEMA.to_owned(),
        total_effects: effects.len(),
        available_effects,
        effects,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn all_registered_effects_have_unique_keys_valid_ranges_and_consistent_pf_ids() {
        let mut keys = BTreeSet::new();
        let mut pfs = BTreeSet::new();
        assert_eq!(descriptors().len(), 95);

        for desc in descriptors() {
            assert!(
                keys.insert(desc.id.store_key),
                "duplicate store_key: {}",
                desc.id.store_key
            );
            assert!(pfs.insert(desc.id.pf), "duplicate pf: {}", desc.id.pf);
            assert_eq!(
                EffectId::parse_pf_from_store_key(desc.id.store_key),
                Some(desc.id.pf),
                "store_key {} does not match pf {}",
                desc.id.store_key,
                desc.id.pf
            );
            assert!(
                desc.range.is_valid(),
                "invalid range for {}: {:?}",
                desc.id.store_key,
                desc.range
            );
            assert!(
                desc.range.contains(desc.range.default),
                "default outside [min, max] for {}",
                desc.id.store_key
            );
            assert!(!desc.label_en.is_empty());
            assert!(!desc.label_zh.is_empty());
        }
    }

    #[test]
    fn three_capability_axes_remain_strictly_separated() {
        // PF90005 (Whitening) is statically wired and registered in the TSV,
        // yet on a host without models/inputs it MUST NOT be reported Available.
        let whitening = descriptor_by_pf(90005).expect("PF90005 must be registered");
        let empty_ctx = RuntimeProbeContext::cpu_only_empty();

        assert!(whitening.effective_static_support().wired_to_stage);
        assert!(whitening.effective_static_support().cpu_reference);

        let evidence = whitening.evidence_profile();
        // Vendor parity and real-photo acceptance remain unverified in the embedded TSV.
        assert!(!evidence.vendor_parity);
        assert!(!evidence.real_photo_accept);
        assert!(whitening.evidence_level() < EvidenceLevel::Ev6VendorParity);

        // Runtime availability is independent of static support and historical evidence.
        assert_eq!(
            whitening.evaluate_availability(&empty_ctx),
            RuntimeAvailability::MissingModel {
                required: vec![InputDependency::FaceParsingV2]
            }
        );

        // Even when runtime becomes Available, historical evidence does NOT upgrade to Ev6.
        let full_ctx = RuntimeProbeContext::fully_ready();
        assert_eq!(
            whitening.evaluate_availability(&full_ctx),
            RuntimeAvailability::Available
        );
        assert_eq!(
            whitening.evidence_level(),
            evidence.highest_level()
        );
        assert!(!whitening.evidence_profile().real_photo_accept);
    }

    #[test]
    fn missing_model_asset_input_and_backend_are_explicitly_rejected() {
        let smile = descriptor_by_pf(8033).expect("PF8033 must be registered");
        let teeth = descriptor_by_pf(108).expect("PF108 must be registered");

        // 1. No backend available.
        let no_backend = RuntimeProbeContext::default();
        assert!(matches!(
            smile.evaluate_availability(&no_backend),
            RuntimeAvailability::BackendUnavailable { .. }
        ));

        // 2. Strict GPU requested without CUDA/Metal.
        let strict_no_gpu = RuntimeProbeContext {
            cpu_available: true,
            strict_gpu: true,
            ..RuntimeProbeContext::default()
        };
        assert!(matches!(
            teeth.evaluate_availability(&strict_no_gpu),
            RuntimeAvailability::BackendUnavailable { .. }
        ));

        // 3. CPU present, but Landmarks194 model missing.
        let mut ctx = RuntimeProbeContext::cpu_only_empty();
        assert_eq!(
            smile.evaluate_availability(&ctx),
            RuntimeAvailability::MissingModel {
                required: vec![InputDependency::Landmarks194]
            }
        );

        // 4. Model present, but SmileWarpAsset missing.
        ctx.available_models.insert(InputDependency::Landmarks194);
        assert_eq!(
            smile.evaluate_availability(&ctx),
            RuntimeAvailability::MissingAsset {
                required: vec![InputDependency::SmileWarpAsset]
            }
        );

        // 5. Model + asset present, but per-job Landmarks194 input not yet detected on photo.
        ctx.available_assets.insert(InputDependency::SmileWarpAsset);
        assert_eq!(
            smile.evaluate_availability(&ctx),
            RuntimeAvailability::MissingInput {
                required: vec![InputDependency::Landmarks194]
            }
        );

        // 6. All satisfied -> Available.
        ctx.available_inputs.insert(InputDependency::Landmarks194);
        assert_eq!(
            smile.evaluate_availability(&ctx),
            RuntimeAvailability::Available
        );
    }

    #[test]
    fn stored_only_descriptor_always_reports_stored_only() {
        let dummy = EffectDescriptor {
            id: EffectId::new(999999, "pc:999999"),
            category: EffectCategory::Skin,
            section: "Unwired",
            label_en: "Unwired placeholder",
            label_zh: "未接入占位",
            range: ParamRange::normalized_01(0.0),
            dependencies: &[InputDependency::None],
            static_support: StaticSupport::stored_only(),
        };
        assert_eq!(
            dummy.evaluate_availability(&RuntimeProbeContext::fully_ready()),
            RuntimeAvailability::StoredOnly
        );
    }

    #[test]
    fn required_producers_is_empty_for_default_and_neutral_adjustments() {
        let mut adj = crate::AdjustmentSet::default();
        assert!(!has_active_portrait_effect(&adj));
        assert!(required_producers(&adj).is_empty());

        // Insert neutral values (including non-zero neutral defaults like 0.5).
        adj.extra.insert("pc:21015".into(), 0.5);
        adj.extra.insert("pc:8069".into(), 0.5);
        adj.extra.insert("pc:90021".into(), 0.5);
        adj.extra.insert("pc:90005".into(), 0.0);
        adj.extra.insert("pc:242".into(), 0.0);
        adj.extra.insert("pc:30210".into(), 0.0);
        // Unknown / stored-only keys must not trigger producers either.
        adj.extra.insert("pc:999999".into(), 0.8);

        assert!(!has_active_portrait_effect(&adj));
        assert!(active_effects(&adj).is_empty());
        assert!(required_producers(&adj).is_empty());
    }

    #[test]
    fn required_producers_deduplicates_combined_non_neutral_effects() {
        let mut adj = crate::AdjustmentSet::default();
        // Skin whitening (FaceParsingV2 + SkinMask)
        adj.extra.insert("pc:90005".into(), 0.35);
        // Eye wrinkles (Landmarks194 + FaceParsingV2 + SkinMask)
        adj.extra.insert("pc:90023".into(), 0.4);
        // Teeth whitening (FaceParsingV2 + TeethMask)
        adj.extra.insert("pc:108".into(), 0.5);
        // Smile mouth corner (Landmarks194 + SmileWarpAsset)
        adj.extra.insert("pc:8033".into(), 0.25);
        // Face reshape pair (Landmarks194)
        adj.extra.insert("pc:242".into(), -20.0);

        assert!(has_active_portrait_effect(&adj));
        assert_eq!(active_effects(&adj).len(), 5);

        let producers = required_producers(&adj);
        assert_eq!(
            producers,
            vec![
                InputDependency::Landmarks194,
                InputDependency::FaceParsingV2,
                InputDependency::SkinMask,
                InputDependency::TeethMask,
                InputDependency::SmileWarpAsset,
            ]
        );
    }

    #[test]
    fn capability_snapshot_aggregates_all_categories_and_reflects_probe_context() {
        let partial_ctx = RuntimeProbeContext {
            cpu_available: true,
            available_models: [InputDependency::Landmarks194].into_iter().collect(),
            available_assets: BTreeSet::new(),
            available_inputs: [InputDependency::Landmarks194].into_iter().collect(),
            ..RuntimeProbeContext::default()
        };

        let snapshot = capability_snapshot(&partial_ctx);
        assert_eq!(snapshot.schema, SNAPSHOT_SCHEMA);
        assert_eq!(snapshot.total_effects, descriptors().len());
        // Only Landmarks194-only effects (42 FaceGeometry + 8 FacePlump = 50) should be Available.
        assert_eq!(snapshot.available_effects, 50);

        let face_geom = snapshot.by_category(EffectCategory::FaceGeometry);
        assert_eq!(face_geom.len(), 42);
        assert!(face_geom.iter().all(|e| e.availability.is_available()));

        let plump = snapshot.by_category(EffectCategory::FacePlump);
        assert_eq!(plump.len(), 8);
        assert!(plump.iter().all(|e| e.availability.is_available()));

        // Smile needs SmileWarpAsset which is absent in partial_ctx.
        let smile = snapshot.by_store_key("pc:8033").expect("pc:8033 in snapshot");
        assert_eq!(
            smile.availability,
            RuntimeAvailability::MissingAsset {
                required: vec![InputDependency::SmileWarpAsset]
            }
        );

        // Teeth needs FaceParsingV2 model which is absent in partial_ctx.
        let teeth = snapshot.by_pf(108).expect("PF108 in snapshot");
        assert_eq!(
            teeth.availability,
            RuntimeAvailability::MissingModel {
                required: vec![InputDependency::FaceParsingV2]
            }
        );

        // Snapshot serializes cleanly to JSON for frontend/IPC consumption.
        let json = serde_json::to_string(&snapshot).expect("snapshot JSON");
        assert!(json.contains("\"schema\":\"aurora/portrait-capability-snapshot/1\""));
        assert!(json.contains("\"store_key\":\"pc:90005\""));
    }
}
