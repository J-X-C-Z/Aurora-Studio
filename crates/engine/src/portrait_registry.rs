//! Unified portrait effect descriptor, capability registry and runtime availability.
//!
//! This module implements section 11 of the core architecture refactoring plan:
//! a single reviewable descriptor registry for the portrait effect catalogue,
//! strictly separating three orthogonal capability axes:
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
//!
//! # Coverage
//!
//! [`WIRED_EFFECT_DESCRIPTORS`] carries every effect row of
//! `capability-registry-20261124.tsv` (147 unique primary PFs) plus the two
//! teeth sub-controls `PF109` (brightness) and `PF110` (de-yellow), for 149
//! descriptors in total. `PF109`/`PF110` own no dispatcher route of their own;
//! the engine resolves them into the `PF108` whitening pipeline through
//! `portrait_stage::pf108_family` (a child-only document runs PF108 at full
//! common weight). Being listed here is never a live/`Available` claim: a descriptor is only
//! `Available` when [`StaticSupport::wired_to_stage`] is true *and* every
//! declared model, asset and same-job input is present in
//! [`RuntimeProbeContext`].

use std::collections::BTreeSet;

use aurora_portrait::capability_registry::{self, CapabilityRow, UNVERIFIED};
use serde::{Deserialize, Serialize};

/// Schema identifier for serialized [`PortraitCapabilitySnapshot`] payloads.
pub const SNAPSHOT_SCHEMA: &str = "aurora/portrait-capability-snapshot/2";

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
    /// Body / hands / feet reshaping and body-detail retouch.
    ///
    /// The geometry producers are owned outside this module; the category only
    /// describes where the control belongs.
    Body,
}

impl EffectCategory {
    /// All categories in canonical display order.
    pub const ALL: [Self; 9] = [
        Self::Skin,
        Self::FaceGeometry,
        Self::FacePlump,
        Self::Expression,
        Self::Teeth,
        Self::Eyes,
        Self::Hair,
        Self::Makeup,
        Self::Body,
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
            Self::Body => "Body Reshape",
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
            Self::Body => "全身美型",
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
    /// Normalized storage; the exact signed bounds are carried by [`ParamRange`]
    /// (`0.0..=1.0` for one-way amounts, `-1.0..=1.0` for signed geometry).
    Normalized01,
    /// Stored directly as `-100.0..=100.0` in `AdjustmentSet::extra`.
    SignedPercent,
    /// Raw dimensionless scalar (e.g. the `0..=4` full-face plump level).
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

/// Numeric bounds, neutral default, unit and UI step for an effect parameter.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParamRange {
    pub min: f32,
    pub max: f32,
    pub default: f32,
    pub unit: ParamUnit,
    /// Smallest UI increment. `0.0` means continuous.
    #[serde(default)]
    pub step: f32,
}

impl ParamRange {
    pub const fn normalized_01(default: f32) -> Self {
        Self {
            min: 0.0,
            max: 1.0,
            default,
            unit: ParamUnit::Normalized01,
            step: 0.0,
        }
    }

    pub const fn normalized_range(min: f32, max: f32, default: f32) -> Self {
        Self {
            min,
            max,
            default,
            unit: ParamUnit::Normalized01,
            step: 0.0,
        }
    }

    pub const fn signed_percent(default: f32) -> Self {
        Self {
            min: -100.0,
            max: 100.0,
            default,
            unit: ParamUnit::SignedPercent,
            step: 0.0,
        }
    }

    /// Whole-number scalar control. `PF30210` stores `0..=4` in whole levels;
    /// the app slider rounds and clamps to the same bounds.
    pub const fn scalar_integer(min: f32, max: f32, default: f32) -> Self {
        Self {
            min,
            max,
            default,
            unit: ParamUnit::Scalar,
            step: 1.0,
        }
    }

    /// True when `min <= default <= max` and all bounds are finite.
    pub fn is_valid(&self) -> bool {
        self.min.is_finite()
            && self.max.is_finite()
            && self.default.is_finite()
            && self.step.is_finite()
            && self.step >= 0.0
            && self.min <= self.default
            && self.default <= self.max
    }

    /// True when `value` falls within the declared bounds and step lattice.
    pub fn contains(&self, value: f32) -> bool {
        value.is_finite() && value >= self.min && value <= self.max
            && (self.step == 0.0 || {
                let tick = (value - self.min) / self.step;
                (tick - tick.round()).abs() <= f32::EPSILON
            })
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
    /// Body pose/keypoint inference model and its per-photo keypoint set.
    BodyPose,
    /// Human-parsing (body/skin/clothes/hair) segmentation model and its per-photo mask.
    HumanParsingMask,
    /// Stubble / stray-hair / white-hair segmentation and repair model.
    StubbleStrayHairModel,
    /// Eye-bag / dark-circle residual model.
    EyeBagModel,
    /// Eye-white purification model.
    EyeWhiteModel,
    /// Breast repair model.
    BreastRepairModel,
    /// Teeth repair model.
    TeethRepairModel,
    /// Glasses glare / lens reflection model.
    GlassesGlareModel,
    /// Double-chin shift-field model.
    DoubleChinModel,
    /// Generic skin repair model (veins, stretch marks, sweat, generic texture).
    SkinRepairGenericModel,
    /// Moisture smoothing curve tables (`PF90002`/`PF90010`).
    MoistureCurveAsset,
    /// `FaceStereo` stereo shadow/highlight material set (`PF8`/`PF9`).
    FaceStereoAsset,
    /// `LipsLUT` / `LipDodgeLUT` lip material (`PF90021`).
    LipsLutAsset,
    /// The face-plump model set and its same-job residual planes.
    FacePlumpResidual,
    /// The specific skin tone table(s) required by this effect.
    SkinToneLut,
    /// Face-width, forehead or reshape coefficients consumed by this pair.
    FacePartTable,
}

impl InputDependency {
    /// True if this dependency requires an inference model file on disk / loaded runtime.
    pub const fn is_model_backed(self) -> bool {
        matches!(
            self,
            Self::Landmarks194
                | Self::FaceParsingV2
                | Self::BodyPose
                | Self::HumanParsingMask
                | Self::StubbleStrayHairModel
                | Self::EyeBagModel
                | Self::EyeWhiteModel
                | Self::BreastRepairModel
                | Self::TeethRepairModel
                | Self::GlassesGlareModel
                | Self::DoubleChinModel
                | Self::SkinRepairGenericModel
                | Self::FacePlumpResidual
        )
    }

    /// True if this dependency requires external/bundled non-model material or warp assets.
    pub const fn is_asset_backed(self) -> bool {
        matches!(
            self,
            Self::MakeupBundle
                | Self::SmileWarpAsset
                | Self::MoistureCurveAsset
                | Self::FaceStereoAsset
                | Self::LipsLutAsset
                | Self::SkinToneLut
                | Self::FacePartTable
        )
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
                | Self::BodyPose
                | Self::HumanParsingMask
                | Self::FacePlumpResidual
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
            Self::BodyPose => "body_pose",
            Self::HumanParsingMask => "human_parsing_mask",
            Self::StubbleStrayHairModel => "stubble_stray_hair_model",
            Self::EyeBagModel => "eye_bag_model",
            Self::EyeWhiteModel => "eye_white_model",
            Self::BreastRepairModel => "breast_repair_model",
            Self::TeethRepairModel => "teeth_repair_model",
            Self::GlassesGlareModel => "glasses_glare_model",
            Self::DoubleChinModel => "double_chin_model",
            Self::SkinRepairGenericModel => "skin_repair_generic_model",
            Self::MoistureCurveAsset => "moisture_curve_asset",
            Self::FaceStereoAsset => "face_stereo_asset",
            Self::LipsLutAsset => "lips_lut_asset",
            Self::FacePlumpResidual => "face_plump_residual",
            Self::SkinToneLut => "skin_tone_lut",
            Self::FacePartTable => "face_part_table",
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

    /// CPU reference kernel exists, but no dispatcher route reads the control
    /// yet: the value is stored only and always reports
    /// [`RuntimeAvailability::StoredOnly`].
    pub const fn cpu_reference_only(roi_supported: bool) -> Self {
        Self {
            cpu_reference: true,
            cuda_kernel: false,
            metal_kernel: false,
            wired_to_stage: false,
            roi_supported,
        }
    }

    /// CPU reference and CUDA kernel exist in the crate, but the effect has no
    /// dispatcher route (e.g. `PF8050` geometry, `PF90087` hair seam). Stored
    /// only: a compiled kernel is not a live effect.
    pub const fn unwired_cpu_cuda(roi_supported: bool) -> Self {
        Self {
            cpu_reference: true,
            cuda_kernel: true,
            metal_kernel: false,
            wired_to_stage: false,
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
    /// Cheap discovery succeeded; per-photo mask/material production has not yet been measured.
    PendingSameFrame { required: Vec<InputDependency> },
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
            Self::PendingSameFrame { .. } => "pending_same_frame",
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

    /// Fully populated context: CPU+CUDA and every dependency the registry knows
    /// about is ready. This is a probe fixture, not a claim about any host.
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
        for dep in [
            InputDependency::Landmarks194,
            InputDependency::FaceParsingV2,
            InputDependency::BodyPose,
            InputDependency::HumanParsingMask,
            InputDependency::StubbleStrayHairModel,
            InputDependency::EyeBagModel,
            InputDependency::EyeWhiteModel,
            InputDependency::BreastRepairModel,
            InputDependency::TeethRepairModel,
            InputDependency::GlassesGlareModel,
            InputDependency::DoubleChinModel,
            InputDependency::SkinRepairGenericModel,
            InputDependency::FacePlumpResidual,
        ] {
            ctx.available_models.insert(dep);
        }
        for dep in [
            InputDependency::MakeupBundle,
            InputDependency::SmileWarpAsset,
            InputDependency::MoistureCurveAsset,
            InputDependency::FaceStereoAsset,
            InputDependency::LipsLutAsset,
            InputDependency::SkinToneLut,
            InputDependency::FacePartTable,
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
            InputDependency::BodyPose,
            InputDependency::HumanParsingMask,
            InputDependency::FacePlumpResidual,
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

    /// The implemented routes define support. Historical evidence can name an
    /// isolated kernel or a required ROI; neither makes it a product backend.
    pub fn effective_static_support(&self) -> StaticSupport {
        self.static_support
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
const DEPS_TONE_LUT: &[InputDependency] = &[InputDependency::FaceParsingV2, InputDependency::SkinMask, InputDependency::SkinToneLut];
const DEPS_FACE_PART: &[InputDependency] = &[InputDependency::Landmarks194, InputDependency::FacePartTable];
const DEPS_PLUMP: &[InputDependency] = &[InputDependency::Landmarks194, InputDependency::FacePlumpResidual];
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
const DEPS_TEETH_REPAIR: &[InputDependency] = &[
    InputDependency::FaceParsingV2,
    InputDependency::TeethMask,
    InputDependency::TeethRepairModel,
];
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
    InputDependency::LipsLutAsset,
];
/// Eye image makeup (`PF90032`): the material is composited through
/// `EyeMaskType=10`, not through the lip masks.
const DEPS_EYE_MAKEUP: &[InputDependency] = &[
    InputDependency::FaceParsingV2,
    InputDependency::MakeupBundle,
    InputDependency::EyeMask,
];
const DEPS_STEREO: &[InputDependency] = &[
    InputDependency::FaceParsingV2,
    InputDependency::MakeupBundle,
    InputDependency::FaceStereoAsset,
];
const DEPS_MOISTURE: &[InputDependency] = &[
    InputDependency::FaceParsingV2,
    InputDependency::SkinMask,
    InputDependency::MoistureCurveAsset,
];
const DEPS_BODY: &[InputDependency] = &[
    InputDependency::BodyPose,
    InputDependency::HumanParsingMask,
];
const DEPS_STUBBLE: &[InputDependency] = &[
    InputDependency::FaceParsingV2,
    InputDependency::StubbleStrayHairModel,
];
const DEPS_STUBBLE_HAIR: &[InputDependency] = &[
    InputDependency::FaceParsingV2,
    InputDependency::HairMask,
    InputDependency::StubbleStrayHairModel,
];
const DEPS_EYE_BAG: &[InputDependency] = &[
    InputDependency::FaceParsingV2,
    InputDependency::EyeBagModel,
];
const DEPS_EYE_WHITE: &[InputDependency] = &[
    InputDependency::FaceParsingV2,
    InputDependency::EyeMask,
    InputDependency::EyeWhiteModel,
];
const DEPS_DOUBLE_CHIN: &[InputDependency] = &[
    InputDependency::FaceParsingV2,
    InputDependency::DoubleChinModel,
];
const DEPS_GLARE: &[InputDependency] = &[
    InputDependency::FaceParsingV2,
    InputDependency::GlassesGlareModel,
];
const DEPS_HAND: &[InputDependency] = &[InputDependency::HumanParsingMask];
const DEPS_HAND_VEIN: &[InputDependency] = &[
    InputDependency::HumanParsingMask,
    InputDependency::SkinRepairGenericModel,
];
const DEPS_FOOT_VEIN: &[InputDependency] = &[
    InputDependency::HumanParsingMask,
    InputDependency::SkinRepairGenericModel,
];
const DEPS_STRETCH: &[InputDependency] = &[
    InputDependency::HumanParsingMask,
    InputDependency::SkinRepairGenericModel,
];
const DEPS_SWEAT: &[InputDependency] = &[
    InputDependency::FaceParsingV2,
    InputDependency::SkinRepairGenericModel,
];
const DEPS_BODY_BLEMISH: &[InputDependency] = &[
    InputDependency::FaceParsingV2,
    InputDependency::SkinMask,
    InputDependency::SkinRepairGenericModel,
];
const DEPS_TEXTURE_SMOOTH: &[InputDependency] = &[
    InputDependency::FaceParsingV2,
    InputDependency::SkinRepairGenericModel,
];
const DEPS_LIP_LINE: &[InputDependency] = &[
    InputDependency::FaceParsingV2,
    InputDependency::LipsLutAsset,
];
const DEPS_JAWLINE: &[InputDependency] = &[
    InputDependency::Landmarks194,
    InputDependency::FaceParsingV2,
];
const DEPS_BREAST: &[InputDependency] = &[
    InputDependency::HumanParsingMask,
    InputDependency::BreastRepairModel,
];

macro_rules! desc {
    ($pf:expr, $cat:ident, $section:expr, $en:expr, $zh:expr, $range:expr, $deps:expr, $support:expr) => {
        EffectDescriptor {
            id: EffectId::new($pf, concat!("pc:", stringify!($pf))),
            category: EffectCategory::$cat,
            section: $section,
            label_en: $en,
            label_zh: $zh,
            range: $range,
            dependencies: $deps,
            static_support: $support,
        }
    };
}

/// Canonical registry of every portrait effect in the capability table.
///
/// 147 unique primary PFs from `capability-registry-20261124.tsv` plus the
/// `PF109`/`PF110` teeth sub-controls (149 descriptors). `wired_to_stage` is
/// only set where the engine really routes the PF; a stored-only descriptor
/// always reports [`RuntimeAvailability::StoredOnly`].
pub const WIRED_EFFECT_DESCRIPTORS: &[EffectDescriptor] = &[
    // ── Skin (40) ────────────────────────────────────────────────────
    desc!(0, Skin, "Blemish", "Spotless acne removal", "祛斑祛痘", ParamRange::normalized_01(0.0), DEPS_SKIN, StaticSupport::cpu_wired(true)),
    desc!(8068, Skin, "Texture", "Skin gloss", "皮肤光泽", ParamRange::normalized_01(0.5), DEPS_SKIN, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(8069, Skin, "Texture", "Skin texture (smoothing)", "肤质（柔化）", ParamRange::normalized_range(0.0, 0.5, 0.5), DEPS_SKIN, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(8070, Skin, "Texture", "Skin clarity", "清晰度", ParamRange::normalized_01(0.0), DEPS_SKIN, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(21015, Skin, "Skin tone", "Skin hue", "肤色色相", ParamRange::normalized_01(0.5), DEPS_TONE_LUT, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(21024, Skin, "Wrinkles", "Glabella wrinkles", "祛眉间纹", ParamRange::normalized_01(0.0), DEPS_WRINKLE, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(21025, Skin, "Wrinkles", "Puppet wrinkles", "祛木偶纹", ParamRange::normalized_01(0.0), DEPS_WRINKLE, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(21026, Skin, "Blemish", "Remove stubble", "祛胡须", ParamRange::normalized_01(0.0), DEPS_STUBBLE, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(21069, Skin, "Wrinkles", "Cheek wrinkles", "祛面颊纹", ParamRange::normalized_01(0.0), DEPS_WRINKLE, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(21070, Skin, "Wrinkles", "Nose back wrinkles", "祛鼻背纹", ParamRange::normalized_01(0.0), DEPS_WRINKLE, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(21071, Skin, "Wrinkles", "Perioral wrinkles", "祛口周纹", ParamRange::normalized_01(0.0), DEPS_WRINKLE, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(21104, Skin, "Blemish", "Nose flaws", "祛鼻部瑕疵", ParamRange::normalized_01(0.0), DEPS_SKIN, StaticSupport::cpu_reference_only(false)),
    desc!(21105, Skin, "Skin tone", "AI body skin tone unify", "AI 身体肤色统一", ParamRange::normalized_01(0.0), DEPS_SKIN, StaticSupport::cpu_wired(true)),
    desc!(30013, Skin, "Smoothing & Neutral Grey", "Neutral grey flat", "中性灰平", ParamRange::normalized_01(0.0), DEPS_SKIN, StaticSupport::cpu_wired(true)),
    desc!(30014, Skin, "Smoothing & Neutral Grey", "Neutral grey 3D", "中性灰 3D", ParamRange::normalized_01(0.0), DEPS_SKIN, StaticSupport::cpu_wired(true)),
    desc!(30021, Skin, "Smoothing & Neutral Grey", "Face smooth (HLF)", "中性灰/高低频磨皮", ParamRange::normalized_01(0.0), DEPS_SKIN, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(30022, Skin, "Smoothing & Neutral Grey", "Body smooth (HLF)", "全身磨皮", ParamRange::normalized_01(0.0), DEPS_SKIN, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(30059, Skin, "Skin tone", "Skin tone unify (multi)", "多人肤色统一", ParamRange::normalized_01(0.0), DEPS_SKIN, StaticSupport::cpu_wired(false)),
    desc!(30096, Skin, "Skin tone", "Skin temperature", "色调", ParamRange::normalized_01(0.5), DEPS_TONE_LUT, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(30188, Skin, "Blemish", "Double chin amount", "祛双下巴-程度", ParamRange::normalized_01(0.9), DEPS_DOUBLE_CHIN, StaticSupport::stored_only()),
    desc!(30189, Skin, "Blemish", "Double chin shadow", "祛双下巴-阴影", ParamRange::normalized_01(0.8), DEPS_DOUBLE_CHIN, StaticSupport::stored_only()),
    desc!(30209, Skin, "Blemish", "Jawline enhance", "下颌线增强", ParamRange::normalized_01(0.0), DEPS_JAWLINE, StaticSupport::stored_only()),
    desc!(30259, Skin, "Blemish", "Sweat (face)", "祛汗珠（脸部）", ParamRange::normalized_01(0.0), DEPS_SWEAT, StaticSupport::cpu_wired(false)),
    desc!(90001, Skin, "Texture", "Texture smooth skin", "肤质平滑", ParamRange::normalized_01(0.0), DEPS_TEXTURE_SMOOTH, StaticSupport::stored_only()),
    desc!(90002, Skin, "Smoothing & Neutral Grey", "Moisture smooth (face)", "水润磨皮（脸部）", ParamRange::normalized_01(0.0), DEPS_MOISTURE, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(90003, Skin, "Skin tone", "Body skin tone uniform", "身体肤色均匀", ParamRange::normalized_01(0.0), DEPS_SKIN, StaticSupport::cpu_reference_only(true)),
    desc!(90004, Skin, "Skin tone", "Skin radiance", "皮肤透亮", ParamRange::normalized_01(0.0), DEPS_SKIN, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(90005, Skin, "Skin tone", "Whitening", "美白", ParamRange::normalized_01(0.0), DEPS_TONE_LUT, StaticSupport { metal_kernel: true, ..StaticSupport::cpu_and_cuda_wired(true) }),
    desc!(90007, Skin, "Skin tone", "Rosy", "红润", ParamRange::normalized_01(0.0), DEPS_TONE_LUT, StaticSupport { metal_kernel: true, ..StaticSupport::cpu_and_cuda_wired(true) }),
    desc!(90008, Skin, "Blemish", "Stretch marks", "祛妊娠纹", ParamRange::normalized_01(0.0), DEPS_STRETCH, StaticSupport::cpu_wired(false)),
    desc!(90010, Skin, "Smoothing & Neutral Grey", "Moisture smooth (body)", "水润磨皮（身体）", ParamRange::normalized_01(0.0), DEPS_MOISTURE, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(90015, Skin, "Blemish", "Neck lines", "祛颈纹", ParamRange::normalized_01(0.0), DEPS_WRINKLE, StaticSupport::cpu_wired(false)),
    desc!(90023, Skin, "Wrinkles", "Eye wrinkles", "祛眼纹", ParamRange::normalized_01(0.0), DEPS_WRINKLE, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(90024, Skin, "Wrinkles", "Forehead wrinkles", "祛抬头纹", ParamRange::normalized_01(0.0), DEPS_WRINKLE, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(90034, Skin, "Blemish", "Double chin", "祛双下巴", ParamRange::normalized_01(0.0), DEPS_DOUBLE_CHIN, StaticSupport::cpu_wired(false)),
    desc!(90080, Skin, "Blemish", "Oil shine (face)", "祛油光（脸部）", ParamRange::normalized_01(0.0), DEPS_SKIN, StaticSupport::cpu_wired(false)),
    desc!(90085, Skin, "Skin tone", "Face skin tone unify", "肤色统一（脸部）", ParamRange::normalized_01(0.0), DEPS_SKIN, StaticSupport::cpu_wired(true)),
    desc!(90086, Skin, "Skin tone", "Body skin tone unify", "肤色统一（身体）", ParamRange::normalized_01(0.0), DEPS_SKIN, StaticSupport::cpu_wired(true)),
    desc!(90089, Skin, "Blemish", "Body blemishes (child)", "宝宝去瑕疵", ParamRange::normalized_01(0.0), DEPS_BODY_BLEMISH, StaticSupport::cpu_wired(false)),
    desc!(90118, Skin, "Lips", "Lip line", "唇纹修整", ParamRange::normalized_01(0.0), DEPS_LIP_LINE, StaticSupport::cpu_wired(false)),

    // ── Face geometry (46) ───────────────────────────────────────────
    desc!(202, FaceGeometry, "Face single", "Hairline", "发际线", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(232, FaceGeometry, "Face single", "Skull top enhance", "高颅顶", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(242, FaceGeometry, "Face pairs", "Sharp chin (L)", "尖下巴（左）", ParamRange::signed_percent(0.0), DEPS_FACE_PART, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(243, FaceGeometry, "Face pairs", "Sharp chin (R)", "尖下巴（右）", ParamRange::signed_percent(0.0), DEPS_FACE_PART, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(247, FaceGeometry, "Face pairs", "V-face (L)", "V 脸（左）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(248, FaceGeometry, "Face pairs", "V-face (R)", "V 脸（右）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8001, FaceGeometry, "Face pairs", "Brow thickness (L)", "眉粗细（左）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8002, FaceGeometry, "Face pairs", "Brow thickness (R)", "眉粗细（右）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8005, FaceGeometry, "Face pairs", "Brow distance (L)", "眉距（左）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8006, FaceGeometry, "Face pairs", "Brow distance (R)", "眉距（右）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8009, FaceGeometry, "Face pairs", "Brow shape (L)", "眉形（左）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8010, FaceGeometry, "Face pairs", "Brow shape (R)", "眉形（右）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8011, FaceGeometry, "Face pairs", "Pupil size (L)", "眼瞳大小（左）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8012, FaceGeometry, "Face pairs", "Pupil size (R)", "眼瞳大小（右）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8013, FaceGeometry, "Face pairs", "Eye height (L)", "眼高（左）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8014, FaceGeometry, "Face pairs", "Eye height (R)", "眼高（右）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8015, FaceGeometry, "Face pairs", "Eye distance (L)", "眼距（左）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8016, FaceGeometry, "Face pairs", "Eye distance (R)", "眼距（右）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8017, FaceGeometry, "Face pairs", "Eye tilt (L)", "眼角倾斜（左）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8018, FaceGeometry, "Face pairs", "Eye tilt (R)", "眼角倾斜（右）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8019, FaceGeometry, "Face pairs", "Eye width (L)", "眼宽（左）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8020, FaceGeometry, "Face pairs", "Eye width (R)", "眼宽（右）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8021, FaceGeometry, "Face pairs", "Brow position (L)", "眉位（左）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8022, FaceGeometry, "Face pairs", "Brow position (R)", "眉位（右）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8023, FaceGeometry, "Face pairs", "Jawline (L)", "下颌角（左）", ParamRange::signed_percent(0.0), DEPS_FACE_PART, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8024, FaceGeometry, "Face pairs", "Jawline (R)", "下颌角（右）", ParamRange::signed_percent(0.0), DEPS_FACE_PART, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8025, FaceGeometry, "Face pairs", "Slim face (L)", "瘦脸（左）", ParamRange::normalized_01(0.0), DEPS_LANDMARKS, StaticSupport::cpu_wired(true)),
    desc!(8026, FaceGeometry, "Face pairs", "Slim face (R)", "瘦脸（右）", ParamRange::normalized_01(0.0), DEPS_LANDMARKS, StaticSupport::cpu_wired(true)),
    desc!(8027, FaceGeometry, "Face pairs", "Cheekbone (L)", "颧骨（左）", ParamRange::signed_percent(0.0), DEPS_FACE_PART, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8028, FaceGeometry, "Face pairs", "Cheekbone (R)", "颧骨（右）", ParamRange::signed_percent(0.0), DEPS_FACE_PART, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8029, FaceGeometry, "Face pairs", "Temple (L)", "太阳穴（左）", ParamRange::signed_percent(0.0), DEPS_FACE_PART, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8030, FaceGeometry, "Face pairs", "Temple (R)", "太阳穴（右）", ParamRange::signed_percent(0.0), DEPS_FACE_PART, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(8050, FaceGeometry, "Face pairs", "Eye size (L)", "眼睛大小（左）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::unwired_cpu_cuda(false)),
    desc!(8051, FaceGeometry, "Face pairs", "Eye size (R)", "眼睛大小（右）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::unwired_cpu_cuda(false)),
    desc!(30073, FaceGeometry, "Face pairs", "Lower eyelid (L)", "眼睑下至（左）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(30079, FaceGeometry, "Face pairs", "Lower eyelid (R)", "眼睑下至（右）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(30080, FaceGeometry, "Face pairs", "Forehead width (L)", "额头宽窄（左）", ParamRange::signed_percent(0.0), DEPS_FACE_PART, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(30081, FaceGeometry, "Face pairs", "Forehead width (R)", "额头宽窄（右）", ParamRange::signed_percent(0.0), DEPS_FACE_PART, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(30082, FaceGeometry, "Face pairs", "Forehead height (L)", "额头高低（左）", ParamRange::signed_percent(0.0), DEPS_FACE_PART, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(30083, FaceGeometry, "Face pairs", "Forehead height (R)", "额头高低（右）", ParamRange::signed_percent(0.0), DEPS_FACE_PART, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(30092, FaceGeometry, "Face pairs", "Face width (L)", "窄脸（左）", ParamRange::signed_percent(0.0), DEPS_FACE_PART, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(30093, FaceGeometry, "Face pairs", "Face width (R)", "窄脸（右）", ParamRange::signed_percent(0.0), DEPS_FACE_PART, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(41922, FaceGeometry, "Face pairs", "Brow tail length (R)", "眉尾长短（右）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(44124, FaceGeometry, "Face pairs", "Brow head length (L)", "眉头长短（左）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(48019, FaceGeometry, "Face pairs", "Brow head length (R)", "眉头长短（右）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(49221, FaceGeometry, "Face pairs", "Brow tail length (L)", "眉尾长短（左）", ParamRange::normalized_range(-1.0, 1.0, 0.0), DEPS_LANDMARKS, StaticSupport::cpu_and_cuda_wired(false)),

    // ── Facial plumpness (8) ────────────────────────────────────────
    desc!(30191, FacePlump, "Facial Plumpness", "Forehead plump", "额头丰盈", ParamRange::signed_percent(0.0), DEPS_PLUMP, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(30194, FacePlump, "Facial Plumpness", "Mid-face plump", "面中丰盈", ParamRange::signed_percent(0.0), DEPS_PLUMP, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(30197, FacePlump, "Facial Plumpness", "Cheek plump", "面颊丰盈", ParamRange::signed_percent(0.0), DEPS_PLUMP, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(30198, FacePlump, "Facial Plumpness", "Nasal base plump", "鼻基底丰盈", ParamRange::signed_percent(0.0), DEPS_PLUMP, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(30203, FacePlump, "Facial Plumpness", "Brow arch plump", "眉弓丰盈", ParamRange::signed_percent(0.0), DEPS_PLUMP, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(30204, FacePlump, "Facial Plumpness", "Chin plump", "下巴丰盈", ParamRange::signed_percent(0.0), DEPS_PLUMP, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(30207, FacePlump, "Facial Plumpness", "Mouth corner lift", "嘴角上扬", ParamRange::signed_percent(0.0), DEPS_PLUMP, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(30210, FacePlump, "Facial Plumpness", "Full face plump", "全脸丰盈", ParamRange::scalar_integer(0.0, 4.0, 0.0), DEPS_PLUMP, StaticSupport::cpu_and_cuda_wired(false)),

    // ── Expression (3) ──────────────────────────────────────────────
    desc!(8033, Expression, "Expression", "Smile mouth corner", "微笑嘴角", ParamRange::normalized_01(0.0), DEPS_EXPRESSION, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(21076, Expression, "Expression", "Closed-mouth smile", "微笑（闭嘴）", ParamRange::normalized_01(0.67), DEPS_EXPRESSION, StaticSupport::cpu_wired(false)),
    desc!(100007, Expression, "Expression", "Open-mouth smile", "微笑（张嘴）", ParamRange::normalized_01(0.2), DEPS_EXPRESSION, StaticSupport::cpu_and_cuda_wired(false)),

    // ── Teeth (5) ───────────────────────────────────────────────────
    desc!(108, Teeth, "Teeth", "Teeth whitening", "美牙/牙齿美白", ParamRange::normalized_01(0.0), DEPS_TEETH, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(109, Teeth, "Teeth", "Teeth brightness", "牙齿亮白", ParamRange::normalized_01(0.0), DEPS_TEETH, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(110, Teeth, "Teeth", "Teeth de-yellow", "牙齿去黄", ParamRange::normalized_01(0.0), DEPS_TEETH, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(30025, Teeth, "Teeth", "Tooth repair", "牙齿修复", ParamRange::normalized_01(0.0), DEPS_TEETH_REPAIR, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(30026, Teeth, "Teeth", "Teeth alignment", "牙齿矫正", ParamRange::normalized_01(0.0), DEPS_TEETH, StaticSupport::cpu_and_cuda_wired(false)),

    // ── Eyes (7) ────────────────────────────────────────────────────
    desc!(5, Eyes, "Eyes", "Reduce dark circles", "祛黑眼圈", ParamRange::normalized_01(0.0), DEPS_EYE_BAG, StaticSupport::cpu_wired(false)),
    desc!(90116, Eyes, "Eyes", "Eye bags", "祛眼袋", ParamRange::normalized_01(0.0), DEPS_EYE_BAG, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(90130, Eyes, "Eyes", "Glasses glare", "祛眼镜反光", ParamRange::normalized_01(0.0), DEPS_GLARE, StaticSupport::cpu_wired(false)),
    desc!(101, Eyes, "Eyes", "Silkworm", "卧蚕", ParamRange::normalized_01(0.0), DEPS_EYES, StaticSupport::cpu_wired(true)),
    desc!(90032, Eyes, "Eye makeup", "Eye makeup", "眼妆", ParamRange::normalized_01(0.0), DEPS_EYE_MAKEUP, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(100004, Eyes, "Eyes", "Red blood removal", "去红血丝", ParamRange::normalized_01(0.0), DEPS_EYES, StaticSupport::cpu_wired(true)),
    desc!(100005, Eyes, "Eyes", "Eye white purify", "眼白净化", ParamRange::normalized_01(0.0), DEPS_EYE_WHITE, StaticSupport::cpu_reference_only(false)),

    // ── Hair (6) ────────────────────────────────────────────────────
    desc!(30009, Hair, "Hair", "Hair color unify", "统一发色", ParamRange::normalized_01(0.0), DEPS_HAIR, StaticSupport::cpu_wired(true)),
    desc!(90087, Hair, "Hair", "Hair parting fill", "发缝填充", ParamRange::normalized_01(0.0), DEPS_HAIR, StaticSupport::unwired_cpu_cuda(false)),
    desc!(90088, Hair, "Hair", "Stray hair", "祛碎发", ParamRange::normalized_01(0.0), DEPS_STUBBLE_HAIR, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(90131, Hair, "Hair", "White hair", "祛白发", ParamRange::normalized_01(0.0), DEPS_STUBBLE_HAIR, StaticSupport::cpu_wired(false)),
    desc!(90139, Hair, "Hair", "Hair smooth", "头发柔顺", ParamRange::normalized_01(0.0), DEPS_HAIR, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(90160, Hair, "Hair", "Hair gloss", "头发光泽", ParamRange::normalized_01(0.0), DEPS_HAIR, StaticSupport::cpu_and_cuda_wired(false)),

    // ── Makeup (15) ──────────────────────────────────────────────────
    desc!(8, Makeup, "Stereo contour", "Stereo shadow", "立体阴影", ParamRange::normalized_01(0.0), DEPS_STEREO, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(9, Makeup, "Stereo contour", "Stereo highlight", "立体高光", ParamRange::normalized_01(0.0), DEPS_STEREO, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(100, Makeup, "Brows & Eyes", "Eyebrow (L)", "眉妆（左）", ParamRange::normalized_01(0.0), DEPS_MAKEUP, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(103, Makeup, "Brows & Eyes", "Upper eyeliner", "上眼线", ParamRange::normalized_01(0.0), DEPS_MAKEUP, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(105, Makeup, "Brows & Eyes", "Upper eyelash", "上睫毛", ParamRange::normalized_01(0.0), DEPS_MAKEUP, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(106, Makeup, "Blusher & Contour", "Blusher", "腮红", ParamRange::normalized_01(0.0), DEPS_MAKEUP, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(107, Makeup, "Lips", "Lipstick", "口红", ParamRange::normalized_01(0.0), DEPS_MAKEUP, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(111, Makeup, "Brows & Eyes", "Contact lenses", "美瞳", ParamRange::normalized_01(0.0), DEPS_MAKEUP, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(115, Makeup, "Sticker", "Face decoration", "面部贴纸", ParamRange::normalized_01(0.0), DEPS_MAKEUP, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(116, Makeup, "Blusher & Contour", "Contour", "修容", ParamRange::normalized_01(0.0), DEPS_MAKEUP, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(117, Makeup, "Brows & Eyes", "Eyebrow (R)", "眉妆（右）", ParamRange::normalized_01(0.0), DEPS_MAKEUP, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(118, Makeup, "Brows & Eyes", "Lower eyelash", "下睫毛", ParamRange::normalized_01(0.0), DEPS_MAKEUP, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(30007, Makeup, "Brows & Eyes", "Lower eyeliner", "下眼线", ParamRange::normalized_01(0.0), DEPS_MAKEUP, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(90021, Makeup, "Lips", "Lip makeup enhance", "唇妆增强", ParamRange::normalized_01(0.5), DEPS_LIP_ENHANCE, StaticSupport::cpu_and_cuda_wired(true)),
    desc!(90033, Makeup, "Suit makeup", "Suit makeup look", "主题妆容", ParamRange::normalized_01(0.7), DEPS_MAKEUP, StaticSupport::cpu_and_cuda_wired(false)),

    // ── Body (module B owns the live geometry producers) (19) ────────
    desc!(400, Body, "Body", "Slim arm", "瘦手臂", ParamRange::normalized_01(0.0), DEPS_BODY, StaticSupport::cpu_wired(false)),
    desc!(401, Body, "Body", "Slim waist", "瘦腰", ParamRange::normalized_01(0.0), DEPS_BODY, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(403, Body, "Body", "Breast enhancement", "胸部增强", ParamRange::normalized_01(0.0), DEPS_BODY, StaticSupport::cpu_reference_only(false)),
    desc!(404, Body, "Body", "Lift buttock", "美臀", ParamRange::normalized_01(0.0), DEPS_BODY, StaticSupport::cpu_wired(false)),
    desc!(405, Body, "Body", "Slim body", "瘦全身", ParamRange::normalized_01(0.0), DEPS_BODY, StaticSupport::cpu_wired(false)),
    desc!(406, Body, "Body", "Head size", "小头", ParamRange::normalized_01(0.0), DEPS_BODY, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(407, Body, "Body", "Lengthen legs", "长腿", ParamRange::normalized_01(0.0), DEPS_BODY, StaticSupport::cpu_wired(false)),
    desc!(408, Body, "Body", "Height increase", "增高", ParamRange::normalized_01(0.0), DEPS_BODY, StaticSupport::cpu_wired(false)),
    desc!(410, Body, "Body", "Neck length", "颈长", ParamRange::normalized_01(0.0), DEPS_BODY, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(415, Body, "Body", "Upper body ratio", "上身比例", ParamRange::normalized_01(0.0), DEPS_BODY, StaticSupport::cpu_and_cuda_wired(false)),
    desc!(416, Body, "Body", "Waist lengthen", "拉腰", ParamRange::normalized_01(0.0), DEPS_BODY, StaticSupport::cpu_wired(false)),
    desc!(30017, Body, "Body detail", "Collarbone enhance", "锁骨增强", ParamRange::normalized_01(0.0), DEPS_BODY, StaticSupport::cpu_wired(false)),
    desc!(30030, Body, "Hands", "Hand blue veins", "祛手部青筋", ParamRange::normalized_01(0.0), DEPS_HAND_VEIN, StaticSupport::cpu_wired(false)),
    desc!(30187, Body, "Hands", "Hand dodge & burn", "手部中性灰", ParamRange::normalized_01(0.0), DEPS_HAND, StaticSupport::cpu_wired(false)),
    desc!(30190, Body, "Hands", "Hand reshape", "手部美型", ParamRange::normalized_01(0.0), DEPS_HAND, StaticSupport::cpu_wired(false)),
    desc!(38556, Body, "Body", "Breast lift", "丰胸", ParamRange::normalized_01(0.0), DEPS_BODY, StaticSupport::cpu_wired(false)),
    desc!(44297, Body, "Feet", "Foot blue veins", "祛脚部青筋", ParamRange::normalized_01(0.0), DEPS_FOOT_VEIN, StaticSupport::cpu_wired(false)),
    desc!(90041, Body, "Body", "AI body reshape", "AI 美体", ParamRange::normalized_01(0.5), DEPS_BODY, StaticSupport::cpu_wired(false)),
    desc!(100008, Body, "Body detail", "Breast repair", "胸部修复", ParamRange::normalized_01(0.0), DEPS_BREAST, StaticSupport::cpu_wired(false)),

];

/// Body sub-controls that the app exposes but that own no descriptor row.
///
/// The body geometry producer groups these into the primary `PF417/418`
/// shoulder-width, `PF411/412` square-shoulder, `PF413/414` swan-neck,
/// `PF30074/30075` slim-leg and `PF30094/30095` lengthen-leg panels. They are
/// deliberately *not* registered as effects: the capability table has no row
/// for them and they have no kernel route of their own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BodySubControl {
    pub pf: u32,
    pub group_en: &'static str,
    pub group_zh: &'static str,
    pub label_en: &'static str,
    pub label_zh: &'static str,
}

/// The ten body sub-controls, in panel order.
pub const BODY_SUBCONTROLS: &[BodySubControl] = &[
    BodySubControl {
        pf: 417,
        group_en: "Shoulder width",
        group_zh: "肩宽",
        label_en: "Shoulder width (L)",
        label_zh: "肩宽（左）",
    },
    BodySubControl {
        pf: 418,
        group_en: "Shoulder width",
        group_zh: "肩宽",
        label_en: "Shoulder width (R)",
        label_zh: "肩宽（右）",
    },
    BodySubControl {
        pf: 411,
        group_en: "Square shoulder",
        group_zh: "直角肩",
        label_en: "Square shoulder (L)",
        label_zh: "直角肩（左）",
    },
    BodySubControl {
        pf: 412,
        group_en: "Square shoulder",
        group_zh: "直角肩",
        label_en: "Square shoulder (R)",
        label_zh: "直角肩（右）",
    },
    BodySubControl {
        pf: 413,
        group_en: "Swan neck",
        group_zh: "天鹅颈",
        label_en: "Swan neck (L)",
        label_zh: "天鹅颈（左）",
    },
    BodySubControl {
        pf: 414,
        group_en: "Swan neck",
        group_zh: "天鹅颈",
        label_en: "Swan neck (R)",
        label_zh: "天鹅颈（右）",
    },
    BodySubControl {
        pf: 30074,
        group_en: "Slim leg",
        group_zh: "瘦腿",
        label_en: "Slim thigh",
        label_zh: "瘦大腿",
    },
    BodySubControl {
        pf: 30075,
        group_en: "Slim leg",
        group_zh: "瘦腿",
        label_en: "Slim calf",
        label_zh: "瘦小腿",
    },
    BodySubControl {
        pf: 30094,
        group_en: "Lengthen leg",
        group_zh: "拉长腿",
        label_en: "Lengthen thigh",
        label_zh: "拉长大腿",
    },
    BodySubControl {
        pf: 30095,
        group_en: "Lengthen leg",
        group_zh: "拉长腿",
        label_en: "Lengthen calf",
        label_zh: "拉长小腿",
    },
];

/// Look up the body sub-control metadata for `pf`.
pub fn body_subcontrol(pf: u32) -> Option<&'static BodySubControl> {
    BODY_SUBCONTROLS.iter().find(|control| control.pf == pf)
}

/// Return all registered portrait effect descriptors.
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

/// Cache identity of a portrait *input verdict*, never a render or undo key.
/// Ordinary strength drags reuse their masks and selected assets. Rendering
/// and async edit conflict checks still use the actual unmodified document.
pub fn portrait_input_key(extra: &std::collections::BTreeMap<String, f32>) -> u64 {
    use std::hash::{DefaultHasher, Hash, Hasher};
    let mut hash = DefaultHasher::new();
    // A single attached teeth plane serves the family; only the choice of
    // channels or a malformed family changes whether those inputs are usable.
    let teeth = crate::portrait_stage::pf108_family(extra);
    let channels = teeth.ok().map(|f| (f.is_active(), f.bright01 > 0.0, f.deyellow01 > 0.0));
    channels.hash(&mut hash);
    crate::portrait_look_layer::requested(extra).hash(&mut hash);
    let level = extra.get("pc:30210").copied().unwrap_or(0.0);
    crate::full_face_descriptor_from_store(level).ok().map(|level| level as u8).hash(&mut hash);
    for (key, &value) in extra {
        let Some(control) = key.strip_prefix("pc:") else { continue };
        if let Some(pf) = EffectId::parse_pf_from_store_key(key) {
            if matches!(pf, 108..=110 | 30210) { continue; }
            if let Some(d) = descriptor_by_pf(pf) {
                if d.range.contains(value) { continue; }
            }
        }
        // Explicit style/color/other resource choices and bad scalar inputs
        // must never be served results established for a different request.
        control.hash(&mut hash);
        value.to_bits().hash(&mut hash);
    }
    hash.finish()
}

/// Iterate over all descriptors belonging to `category`.
pub fn descriptors_in_category(
    category: EffectCategory,
) -> impl Iterator<Item = &'static EffectDescriptor> {
    WIRED_EFFECT_DESCRIPTORS
        .iter()
        .filter(move |desc| desc.category == category)
}

/// True if `store_key` is a registered portrait effect.
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

    /// Face geometry controls the app stores as `-100..=100` (the app's
    /// `tools::portrait::face_geometry_binding`); every other face control is
    /// normalized.
    const PERCENT_STORED_FACE_CONTROLS: [u32; 14] = [
        30092, 30093, 242, 243, 8023, 8024, 8027, 8028, 8029, 8030, 30080, 30081, 30082, 30083,
    ];

    /// The thirteen primary body effects (module B owns their geometry producers).
    const BODY_PRIMARY: [u32; 13] = [
        400, 401, 403, 404, 405, 406, 407, 408, 410, 415, 416, 38556, 90041,
    ];

    fn official_effect_pfs() -> BTreeSet<u32> {
        capability_registry::effects()
            .filter_map(|row| row.pf())
            .collect()
    }

    #[test]
    fn registry_covers_the_official_effect_pf_set_plus_the_teeth_sub_controls() {
        let official = official_effect_pfs();
        assert_eq!(official.len(), 147, "capability table effect rows changed");

        let registry: BTreeSet<u32> = descriptors().iter().map(|desc| desc.id.pf).collect();
        let mut expected = official;
        expected.insert(109);
        expected.insert(110);
        assert_eq!(
            registry, expected,
            "registry and official effect PF sets diverged (both directions checked)"
        );
        assert_eq!(descriptors().len(), 149);
        assert_eq!(registry.len(), descriptors().len());
    }

    #[test]
    fn every_descriptor_has_a_unique_key_valid_range_and_matching_store_key() {
        let mut keys = BTreeSet::new();
        let mut pfs = BTreeSet::new();

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
            assert_eq!(
                desc.id.store_key,
                format!("pc:{}", desc.id.pf),
                "store_key must be the canonical pc:<pf> form"
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
            assert!(!desc.dependencies.is_empty());
            assert!(!desc.dependencies.contains(&InputDependency::None));
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
        assert_eq!(whitening.evidence_level(), evidence.highest_level());
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
        adj.extra.insert("pc:30188".into(), 0.9);
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
                InputDependency::SkinToneLut,
                InputDependency::FacePartTable,
            ]
        );
    }

    #[test]
    fn pf109_and_pf110_are_pf108_sub_controls() {
        for pf in [109, 110] {
            let desc = descriptor_by_pf(pf).expect("teeth sub-control descriptor");
            assert_eq!(desc.category, EffectCategory::Teeth);
            assert_eq!(desc.range, ParamRange::normalized_01(0.0));
            assert!(desc.static_support.wired_to_stage);
            assert!(desc.static_support.cpu_reference);
            assert_eq!(
                EffectId::parse_pf_from_store_key(desc.id.store_key),
                Some(pf)
            );
        }
    }

    #[cfg(windows)]
    #[test]
    fn pf109_and_pf110_resolve_into_pf108_through_the_real_family_resolver() {
        use std::collections::BTreeMap;

        let map = |pairs: &[(&str, f32)]| -> BTreeMap<String, f32> {
            pairs
                .iter()
                .map(|(key, value)| ((*key).to_string(), *value))
                .collect()
        };

        // A child-only document must run PF108 at full common weight; the
        // sibling stays neutral instead of inheriting the legacy 70 %.
        let bright_only =
            crate::portrait_stage::pf108_family(&map(&[("pc:109", 0.6)])).expect("PF109 family");
        assert_eq!(bright_only.common01, 1.0);
        assert_eq!(bright_only.bright01, 0.6);
        assert_eq!(bright_only.deyellow01, 0.0);
        assert!(bright_only.is_active());

        let deyellow_only =
            crate::portrait_stage::pf108_family(&map(&[("pc:110", 0.7)])).expect("PF110 family");
        assert_eq!(deyellow_only.common01, 1.0);
        assert_eq!(deyellow_only.bright01, 0.0);
        assert_eq!(deyellow_only.deyellow01, 0.7);

        let parent_only =
            crate::portrait_stage::pf108_family(&map(&[("pc:108", 0.5)])).expect("PF108 family");
        assert_eq!(parent_only.common01, 0.5);

        let neutral =
            crate::portrait_stage::pf108_family(&map(&[("pc:108", 0.0)])).expect("neutral family");
        assert!(!neutral.is_active(), "all-neutral controls must not schedule PF108");

        for pf in [109, 110] {
            let desc = descriptor_by_pf(pf).expect("teeth sub-control descriptor");
            assert!(desc.static_support.wired_to_stage, "PF{pf} runs through PF108");
            assert_eq!(
                desc.evaluate_availability(&RuntimeProbeContext::fully_ready()),
                RuntimeAvailability::Available
            );
        }
    }

    #[test]
    fn pf90032_is_eye_makeup_with_the_eye_mask_dependency() {
        let desc = descriptor_by_pf(90032).expect("PF90032 must be registered");

        // The vendor control is eyeMakeupDegree with EyeMaskType=10, not lip gloss.
        assert_eq!(desc.category, EffectCategory::Eyes);
        assert_eq!(desc.section, "Eye makeup");
        assert_eq!(desc.label_en, "Eye makeup");
        assert_eq!(desc.label_zh, "眼妆");
        assert!(desc.dependencies.contains(&InputDependency::EyeMask));
        assert!(desc.dependencies.contains(&InputDependency::MakeupBundle));

        // It must not be listed twice or under the lipstick section any more.
        assert!(!descriptors_in_category(EffectCategory::Makeup).any(|d| d.id.pf == 90032));
        assert_eq!(
            descriptors()
                .iter()
                .filter(|d| d.id.pf == 90032)
                .count(),
            1
        );
    }

    #[test]
    fn pf30210_uses_the_whole_number_zero_to_four_scalar() {
        let desc = descriptor_by_pf(30210).expect("PF30210 must be registered");

        assert_eq!(desc.category, EffectCategory::FacePlump);
        assert_eq!(desc.range.unit, ParamUnit::Scalar);
        assert_eq!(desc.range.min, 0.0);
        assert_eq!(desc.range.max, 4.0);
        assert_eq!(desc.range.default, 0.0);
        assert_eq!(desc.range.step, 1.0);
        assert!(desc.range.contains(4.0));
        assert!(!desc.range.contains(4.5));
        assert!(!desc.range.contains(1.5));
        assert!(!desc.range.contains(-1.0));

        // The other plump controls stay signed percent.
        for pf in [30191, 30194, 30197, 30198, 30203, 30204, 30207] {
            assert_eq!(
                descriptor_by_pf(pf).unwrap().range.unit,
                ParamUnit::SignedPercent,
                "PF{pf}"
            );
        }
    }

    #[test]
    fn face_geometry_units_match_the_app_storage_binding() {
        let percent_controls: BTreeSet<u32> =
            descriptors_in_category(EffectCategory::FaceGeometry)
                .filter(|desc| desc.range.unit == ParamUnit::SignedPercent)
                .map(|desc| desc.id.pf)
                .collect();
        assert_eq!(
            percent_controls,
            PERCENT_STORED_FACE_CONTROLS.into_iter().collect::<BTreeSet<_>>(),
            "exactly the 14 app face_geometry_binding percent rows store -100..100"
        );

        for desc in descriptors_in_category(EffectCategory::FaceGeometry) {
            if PERCENT_STORED_FACE_CONTROLS.contains(&desc.id.pf) {
                assert_eq!(desc.range.min, -100.0, "PF{}", desc.id.pf);
                assert_eq!(desc.range.max, 100.0, "PF{}", desc.id.pf);
            } else {
                assert_eq!(desc.range.unit, ParamUnit::Normalized01, "PF{}", desc.id.pf);
                // Slim V2's producer explicitly rejects negatives; the remaining
                // normalized face controls are bidirectional.
                let minimum = if matches!(desc.id.pf, 8025 | 8026) { 0.0 } else { -1.0 };
                assert_eq!(desc.range.min, minimum, "PF{}", desc.id.pf);
                assert_eq!(desc.range.max, 1.0, "PF{}", desc.id.pf);
                assert_eq!(desc.range.default, 0.0, "PF{}", desc.id.pf);
            }
        }

        // PF8033 is a one-way amount, registered once under Expression.
        let smile = descriptor_by_pf(8033).expect("PF8033");
        assert_eq!(smile.category, EffectCategory::Expression);
        assert_eq!(smile.range.unit, ParamUnit::Normalized01);
        assert_eq!(smile.range.min, 0.0);
        assert_eq!(smile.range.max, 1.0);
        assert!(!descriptors_in_category(EffectCategory::FaceGeometry).any(|d| d.id.pf == 8033));
    }

    #[test]
    fn body_category_carries_the_primary_effects_and_the_sub_control_mapping() {
        for pf in BODY_PRIMARY {
            assert_eq!(
                descriptor_by_pf(pf).expect("body primary").category,
                EffectCategory::Body,
                "PF{pf}"
            );
        }
        // Body detail consumers live in the same category.
        for pf in [30017, 30030, 30187, 30190, 44297, 100008] {
            assert_eq!(descriptor_by_pf(pf).unwrap().category, EffectCategory::Body, "PF{pf}");
        }

        assert_eq!(BODY_SUBCONTROLS.len(), 10);
        let sub_pfs: BTreeSet<u32> = BODY_SUBCONTROLS.iter().map(|s| s.pf).collect();
        assert_eq!(sub_pfs.len(), 10);
        for control in BODY_SUBCONTROLS {
            assert!(
                descriptor_by_pf(control.pf).is_none(),
                "body sub-control PF{} must not own a descriptor row",
                control.pf
            );
            assert!(!control.label_en.is_empty());
            assert!(!control.label_zh.is_empty());
        }
        assert_eq!(body_subcontrol(411).unwrap().group_zh, "直角肩");
        assert_eq!(body_subcontrol(417).unwrap().group_en, "Shoulder width");
        assert_eq!(body_subcontrol(30095).unwrap().group_en, "Lengthen leg");
        assert!(body_subcontrol(999).is_none());
    }

    #[test]
    fn gpu_flags_stay_within_recorded_evidence() {
        // Explicit negatives: the dispatcher has no CUDA kernel for these.
        for pf in [0, 101, 8025, 8026, 30009, 100004, 90085, 90086, 21105, 400] {
            assert!(
                !descriptor_by_pf(pf).unwrap().effective_static_support().cuda_kernel,
                "PF{pf} must not claim a CUDA kernel"
            );
        }

        // Body: only the four mesh rows go through the CUDA raster; the others
        // are CPU kernels until module B lands the job producer.
        for pf in [400, 404, 405, 407, 408, 416, 38556, 90041, 403] {
            assert!(
                !descriptor_by_pf(pf).unwrap().static_support.cuda_kernel,
                "PF{pf} must not claim a CUDA kernel"
            );
        }
        for pf in [401, 406, 410, 415] {
            assert!(
                descriptor_by_pf(pf).unwrap().static_support.cuda_kernel,
                "PF{pf} mesh raster has a CUDA path"
            );
        }

        // Unwired rows are always stored only, whatever kernel exists.
        for pf in [
            403, 8050, 8051, 90087, 30188, 30189, 30209, 90001, 100005, 21104, 90003,
        ] {
            assert!(
                descriptor_by_pf(pf).unwrap().static_support.is_stored_only(),
                "PF{pf} must stay stored only"
            );
            assert_eq!(
                descriptor_by_pf(pf)
                    .unwrap()
                    .evaluate_availability(&RuntimeProbeContext::fully_ready()),
                RuntimeAvailability::StoredOnly,
                "PF{pf}"
            );
        }

        let unwired = StaticSupport::cpu_reference_only(true);
        assert!(unwired.cpu_reference);
        assert!(!unwired.cuda_kernel);
        assert!(!unwired.wired_to_stage);
        assert!(unwired.is_stored_only());
    }

    #[cfg(windows)]
    #[test]
    fn wired_to_stage_matches_the_real_dispatcher() {
        // `portrait_stage::supports_effect` is the single dispatcher truth:
        // SUPPORTED_EFFECTS, makeup rows, slim, face pairs/mesh, body, plump,
        // repair, historical tone and historical skin. A descriptor may only
        // claim `wired_to_stage` where that route actually reads `pc:<pf>`.
        for desc in descriptors() {
            let expected = match desc.id.pf {
                // Sub-controls run through PF108 (`portrait_stage::pf108_family`),
                // so they own no `pc:<pf>` dispatcher row of their own.
                109 | 110 => crate::portrait_stage::is_bridged_effect(108),
                pf => crate::portrait_stage::is_bridged_effect(pf),
            };
            assert_eq!(
                desc.static_support.wired_to_stage, expected,
                "PF{} wiring claim does not match portrait_stage::supports_effect",
                desc.id.pf
            );
        }
    }

    #[test]
    fn the_new_model_and_asset_dependencies_are_declared_and_probed() {
        const NEW_DEPS: [InputDependency; 13] = [
            InputDependency::BodyPose,
            InputDependency::HumanParsingMask,
            InputDependency::StubbleStrayHairModel,
            InputDependency::EyeBagModel,
            InputDependency::EyeWhiteModel,
            InputDependency::BreastRepairModel,
            InputDependency::TeethRepairModel,
            InputDependency::GlassesGlareModel,
            InputDependency::DoubleChinModel,
            InputDependency::SkinRepairGenericModel,
            InputDependency::MoistureCurveAsset,
            InputDependency::FaceStereoAsset,
            InputDependency::LipsLutAsset,
        ];

        for dep in NEW_DEPS {
            assert!(
                descriptors().iter().any(|desc| desc.dependencies.contains(&dep)),
                "{:?} is declared but never required by a descriptor",
                dep
            );
        }

        let ctx = RuntimeProbeContext::fully_ready();
        for dep in NEW_DEPS {
            let expected_models = dep.is_model_backed();
            let expected_assets = dep.is_asset_backed();
            assert_eq!(
                ctx.available_models.contains(&dep),
                expected_models,
                "{:?} model probe",
                dep
            );
            assert_eq!(
                ctx.available_assets.contains(&dep),
                expected_assets,
                "{:?} asset probe",
                dep
            );
        }

        // A model+mask dependency is both model-backed and a per-job input.
        assert!(InputDependency::BodyPose.is_model_backed());
        assert!(InputDependency::BodyPose.is_per_job_input());
        assert!(InputDependency::HumanParsingMask.is_model_backed());
        assert!(InputDependency::HumanParsingMask.is_per_job_input());
        assert!(InputDependency::SkinRepairGenericModel.is_model_backed());
        assert!(!InputDependency::SkinRepairGenericModel.is_per_job_input());
        assert!(!InputDependency::MoistureCurveAsset.is_model_backed());
        assert!(InputDependency::MoistureCurveAsset.is_asset_backed());

        // Real asset consumers keep their dependency declared.
        assert!(descriptor_by_pf(8).unwrap().dependencies.contains(&InputDependency::FaceStereoAsset));
        assert!(descriptor_by_pf(9).unwrap().dependencies.contains(&InputDependency::FaceStereoAsset));
        assert!(descriptor_by_pf(90002)
            .unwrap()
            .dependencies
            .contains(&InputDependency::MoistureCurveAsset));
        assert!(descriptor_by_pf(90010)
            .unwrap()
            .dependencies
            .contains(&InputDependency::MoistureCurveAsset));
        assert!(descriptor_by_pf(90021).unwrap().dependencies.contains(&InputDependency::LipsLutAsset));
        assert!(descriptor_by_pf(100005)
            .unwrap()
            .dependencies
            .contains(&InputDependency::EyeWhiteModel));
    }

    #[test]
    fn capability_snapshot_reports_all_categories_with_units_and_availability() {
        let partial_ctx = RuntimeProbeContext {
            cpu_available: true,
            available_models: [InputDependency::Landmarks194].into_iter().collect(),
            available_inputs: [InputDependency::Landmarks194].into_iter().collect(),
            ..RuntimeProbeContext::default()
        };

        let snapshot = capability_snapshot(&partial_ctx);
        assert_eq!(snapshot.schema, SNAPSHOT_SCHEMA);
        assert_eq!(snapshot.total_effects, 149);
        assert_eq!(snapshot.effects.len(), descriptors().len());
        assert_eq!(
            snapshot.by_pf(30210).unwrap().range.unit,
            ParamUnit::Scalar
        );
        assert_eq!(snapshot.by_pf(90032).unwrap().category, EffectCategory::Eyes);
        assert_eq!(
            snapshot.by_store_key("pc:109").unwrap().availability,
            RuntimeAvailability::MissingModel {
                required: vec![InputDependency::FaceParsingV2]
            }
        );

        let face_geom = snapshot.by_category(EffectCategory::FaceGeometry);
        assert_eq!(face_geom.len(), 46);
        // Two have no dispatcher route; fourteen additionally need geometry coefficient tables.
        assert_eq!(face_geom.len() - face_geom.iter().filter(|e| e.availability.is_available()).count(), 16);

        let plump = snapshot.by_category(EffectCategory::FacePlump);
        assert_eq!(plump.len(), 8);
        assert!(plump.iter().all(|e| matches!(&e.availability, RuntimeAvailability::MissingModel { required } if required.contains(&InputDependency::FacePlumpResidual))));

        let body = snapshot.by_category(EffectCategory::Body);
        assert_eq!(body.len(), 19);
        assert!(body.iter().all(|e| !e.availability.is_available()));

        // Geometry can use landmarks alone. Plump also needs real residual planes.
        assert_eq!(snapshot.available_effects, 30);

        // Snapshot serializes cleanly to JSON for frontend/IPC consumption.
        let json = serde_json::to_string(&snapshot).expect("snapshot JSON");
        assert!(json.contains("\"schema\":\"aurora/portrait-capability-snapshot/2\""));
        assert!(json.contains("\"store_key\":\"pc:90005\""));
        assert!(json.contains("\"step\":1.0"));
    }
}
