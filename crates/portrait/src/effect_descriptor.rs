//! Effect descriptors: one place that says what an effect needs and what it
//! can actually do.
//!
//! The capability registry records evidence; the effect router records
//! dispatch. A descriptor ties a stable effect id to its vendor PF alias, its
//! parameters (unit, range, default), its input dependencies, its reference
//! implementation and the backend capability it advertises. It is data, not a
//! second dispatcher.
//!
//! # Scope
//!
//! This module is pure data and has no engine dependency: `aurora-portrait`
//! must not depend on `aurora-engine`. It does not execute an effect and it
//! does not replace [`crate::effect_router`] or
//! [`crate::capability_registry`]. It gives those two, and the app-side lists,
//! a common vocabulary they can migrate onto (see ADR-007 and
//! `docs/architecture/A-REGISTRY-AUDIT.md`).
//!
//! # Capability is three independent dimensions
//!
//! A single `bool` cannot describe an effect honestly. These answers differ:
//!
//! * [`StaticState`] — is the adapter compiled into this build?
//! * [`EvidenceState`] — does documented evidence exist (and at which of the
//!   seven registry levels)?
//! * [`RuntimeState`] — can it run *right now* (driver, device, model asset,
//!   LUT, and same-job inputs all present)?
//!
//! Documented evidence is not current runtime success. A kernel can be
//! compiled and evidenced at ev5 and still refuse today because a LUT file or
//! the face mask is missing. [`CapabilityStatus`] keeps the three answers
//! separate; callers must not flatten them back into one flag.

use std::collections::BTreeSet;

// ---------------------------------------------------------------------------
// Identity
// ---------------------------------------------------------------------------

/// Stable, human-readable effect identity. This is the key a UI or preset
/// refers to; it must not change when a vendor PF alias or a parameter range
/// changes. Use a lowercase dotted name, e.g. `"skin.gloss"`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EffectId(pub &'static str);

/// Vendor `ParamFlag` id. A descriptor may carry several aliases when the
/// product registers left/right or preset variants as separate PFs (for
/// example PF242/PF243), but each PF may belong to only one descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VendorPf(pub u32);

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// Unit convention for a [`ParamSpec`].
///
/// Ranges are documented here, once, so a UI can render a slider without
/// re-deriving the convention from the widget code.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamUnit {
    /// Normalized amount in `[0, 1]`. `0` is usually identity, but some skin
    /// controls use `0.5` as neutral; the [`ParamSpec::default`] says which.
    Amount,
    /// Percent in `[0, 100]`.
    Percent,
    /// Distance in frame pixels.
    Pixels,
    /// Angle in degrees.
    Degrees,
    /// Unitless ratio.
    Ratio,
    /// Discrete choice stored as its numeric index.
    Enum,
    /// Boolean stored as `0.0` or `1.0`.
    Bool,
}

impl ParamUnit {
    /// Short stable token for diagnostics and host-side contracts.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Amount => "amount",
            Self::Percent => "percent",
            Self::Pixels => "pixels",
            Self::Degrees => "degrees",
            Self::Ratio => "ratio",
            Self::Enum => "enum",
            Self::Bool => "bool",
        }
    }
}

/// One slider/parameter an effect consumes.
///
/// The numbers come from the parameter metadata that already exists
/// (`docs/frontend-contract/portrait-algorithms.json`, the app's live lists,
/// and the kernel entry-point contracts). Where a number is uncertain the
/// descriptor's [`EffectDescriptor::notes`] says so; do not invent a range to
/// fill this struct.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParamSpec {
    /// Parameter name, e.g. `"degree"`. Not the storage key (`pc:<pf>`); that
    /// mapping belongs to the app/engine boundary.
    pub name: &'static str,
    pub unit: ParamUnit,
    pub min: f32,
    pub max: f32,
    pub default: f32,
    /// Free-form provenance/behaviour note. Keep it factual.
    pub note: &'static str,
}

impl ParamSpec {
    /// The spec is internally consistent: finite bounds, `min <= max`, and a
    /// default inside the range.
    pub fn is_well_formed(&self) -> bool {
        !self.name.is_empty()
            && self.min.is_finite()
            && self.max.is_finite()
            && self.default.is_finite()
            && self.min <= self.max
            && (self.min..=self.max).contains(&self.default)
    }

    /// `value` is finite and inside `[min, max]` on a well-formed spec.
    pub fn is_valid(&self, value: f32) -> bool {
        self.is_well_formed()
            && value.is_finite()
            && (self.min..=self.max).contains(&value)
    }

    /// Clamp `value` into `[min, max]`.
    ///
    /// `NaN` becomes [`Self::default`]. A malformed spec also returns the
    /// default rather than producing a nonsensical clamp. Infinities clamp to
    /// the nearest bound (`f32::clamp` semantics).
    pub fn clamp(&self, value: f32) -> f32 {
        if !self.is_well_formed() || value.is_nan() {
            return self.default;
        }
        value.clamp(self.min, self.max)
    }
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// A same-job input the effect needs. Every variant documents the coordinate
/// space and format in which it arrives, because the wrong space silently
/// produces plausible-but-wrong pixels.
///
/// Nothing here is discovered or inferred by the effect: the caller supplies
/// the same job's producer output. Look-alike inputs from a different job are
/// not substitutes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputDependency {
    /// Landmark points in frame coordinates. The recovered kernels expect a
    /// fixed count per face (for example 194 points) as normalized `[0, 1]`
    /// `(x, y)` pairs; the exact count/shape is named by the consuming effect.
    Landmarks,
    /// Face-parsing mask over the full render frame: row-major `u8`, exactly
    /// `width * height` bytes. Channel/bit usage (for example teeth `id2.G`)
    /// is effect-specific and stated in the effect's note.
    FaceParsingMask,
    /// Body-parsing mask over the full render frame: row-major `u8`, exactly
    /// `width * height` bytes.
    BodyParsingMask,
    /// Caller-supplied precomputed material: a LUT table, gamma table, colour
    /// matrix, or an RGBA material layer in the effect's native packing. It is
    /// an asset, not something derived from the current frame.
    SkinMaterial,
    /// Per-pixel UV map for the warped frame, normalized `[0, 1]`; the map is
    /// frame-sized unless the effect note narrows it to a ROI.
    UvMap,
    /// A named model tensor. The string names the tensor whose exact shape and
    /// dtype are fixed by the producer; do not reshape it to fit.
    Tensor(&'static str),
    /// Integer pixel rect `(x, y, w, h)` of a face crop in frame coordinates.
    FaceRoi,
    /// Integer pixel rect `(x, y, w, h)` of a body crop in frame coordinates.
    BodyRoi,
    /// Triangulated mesh: vertex positions in frame pixels plus triangle
    /// indices, as produced by the same-job face/body fit.
    Mesh,
}

impl InputDependency {
    /// One-line description used by diagnostics; mirrors the variant docs.
    pub fn describe(self) -> &'static str {
        match self {
            Self::Landmarks => "normalized [0,1] landmark points in frame coordinates",
            Self::FaceParsingMask => "row-major u8 face-parsing mask, width*height bytes",
            Self::BodyParsingMask => "row-major u8 body-parsing mask, width*height bytes",
            Self::SkinMaterial => "precomputed LUT/gamma/matrix or RGBA material asset",
            Self::UvMap => "normalized [0,1] per-pixel UV map",
            Self::Tensor(name) => name,
            Self::FaceRoi => "integer (x,y,w,h) face rect in frame pixels",
            Self::BodyRoi => "integer (x,y,w,h) body rect in frame pixels",
            Self::Mesh => "mesh vertices in frame pixels plus triangle indices",
        }
    }
}

// ---------------------------------------------------------------------------
// Backends and the three capability dimensions
// ---------------------------------------------------------------------------

/// Execution lane a descriptor can advertise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BackendKind {
    /// Recovered CPU kernel in `kernels::*`.
    Cpu,
    /// NVIDIA CUDA adapter. The driver body is `#[cfg(windows)]` today; other
    /// hosts expose the same shape but return `Unavailable` (ADR-006).
    Cuda,
    /// Apple Metal adapter. Host-side scaffolding exists; nothing is verified
    /// on a device yet (`contract::METAL` stays `Missing`).
    Metal,
}

impl BackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Metal => "metal",
        }
    }
}

/// Dimension 1 — is the adapter compiled into the current build?
///
/// This is a static property. It says nothing about a driver, device, model or
/// input. `Compiled` does not mean usable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StaticState {
    /// The adapter body is present in this build.
    Compiled,
    /// The adapter is not built for this target, or the descriptor does not
    /// list the backend at all.
    Absent,
}

impl StaticState {
    /// Whether the host adapter body for `backend` is compiled on this target.
    ///
    /// * CPU kernels compile everywhere.
    /// * CUDA adapter bodies are currently `#[cfg(windows)]`; elsewhere only
    ///   the `no_driver` refusal surface exists, so the lane is `Absent`.
    /// * Metal host code compiles on macOS; it is `Absent` elsewhere.
    ///
    /// This is deliberately *not* a runtime probe. See [`RuntimeState`].
    pub fn for_backend(backend: BackendKind) -> Self {
        let compiled = match backend {
            BackendKind::Cpu => true,
            BackendKind::Cuda => cfg!(windows),
            BackendKind::Metal => cfg!(target_os = "macos"),
        };
        if compiled { Self::Compiled } else { Self::Absent }
    }
}

/// The seven evidence levels, matching
/// [`crate::capability_registry::EVIDENCE_LEVELS`] by index and name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum EvidenceLevel {
    KernelExists = 0,
    CpuReference = 1,
    GpuKernel = 2,
    SameJobInputs = 3,
    MainPreview = 4,
    VendorParity = 5,
    RealPhotoAccept = 6,
}

impl EvidenceLevel {
    pub const ALL: [Self; 7] = [
        Self::KernelExists,
        Self::CpuReference,
        Self::GpuKernel,
        Self::SameJobInputs,
        Self::MainPreview,
        Self::VendorParity,
        Self::RealPhotoAccept,
    ];

    /// Same token as the capability registry's `EVIDENCE_LEVELS`.
    pub fn name(self) -> &'static str {
        match self {
            Self::KernelExists => "kernel_exists",
            Self::CpuReference => "cpu_reference",
            Self::GpuKernel => "gpu_kernel",
            Self::SameJobInputs => "same_job_inputs",
            Self::MainPreview => "main_preview",
            Self::VendorParity => "vendor_parity",
            Self::RealPhotoAccept => "real_photo_accept",
        }
    }

    pub fn index(self) -> usize {
        self as usize
    }
}

/// Dimension 2 — does documented evidence exist, and at which level?
///
/// `level` is the *highest documented* level from the capability registry.
/// Documentation is not a live result: a level-4 row can still be unavailable
/// at runtime, and a missing level proves nothing about higher levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvidenceState {
    pub level: Option<EvidenceLevel>,
}

impl EvidenceState {
    /// No evidence is documented (the registry would say `未验证`).
    pub const NONE: Self = Self { level: None };

    /// Evidence is documented at the given level.
    pub const fn documented(level: EvidenceLevel) -> Self {
        Self { level: Some(level) }
    }

    pub fn is_documented(self) -> bool {
        self.level.is_some()
    }

    /// Index into [`crate::capability_registry::EVIDENCE_LEVELS`].
    pub fn registry_index(self) -> Option<usize> {
        self.level.map(EvidenceLevel::index)
    }
}

/// Dimension 3 — can this effect run right now?
///
/// `available == false` should carry a `reason` (missing driver, model, LUT
/// asset, or same-job input). A reason is diagnostic text, not an error code;
/// callers keep their typed refusals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RuntimeState {
    pub available: bool,
    pub reason: Option<&'static str>,
}

impl RuntimeState {
    pub const fn available() -> Self {
        Self { available: true, reason: None }
    }

    pub const fn unavailable(reason: &'static str) -> Self {
        Self { available: false, reason: Some(reason) }
    }
}

/// The three independent answers for one effect on one backend.
///
/// Keeping them as fields (rather than reducing them to `bool`) is the point:
/// "compiled" is not "evidenced" is not "available now". Use the field getters
/// directly; do not add a collapsing `ready()` helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapabilityStatus {
    pub static_state: StaticState,
    pub evidence: EvidenceState,
    pub runtime: RuntimeState,
}

impl CapabilityStatus {
    /// Combine the three dimensions without collapsing them.
    pub fn from(
        static_state: StaticState,
        evidence: EvidenceState,
        runtime: RuntimeState,
    ) -> Self {
        Self { static_state, evidence, runtime }
    }
}

// ---------------------------------------------------------------------------
// Ordering
// ---------------------------------------------------------------------------

/// Hard ordering constraints relative to other effects.
///
/// The engine currently applies `pc:<pf>` entries in ascending PF order
/// (`portrait_stage` sorts the `extra` map by id). `before`/`after` document
/// additional constraints that ascending PF order cannot express; an empty
/// value means the PF order is the only rule.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OrderConstraints {
    pub before: Vec<EffectId>,
    pub after: Vec<EffectId>,
}

impl OrderConstraints {
    pub fn none() -> Self {
        Self::default()
    }
}

// ---------------------------------------------------------------------------
// Descriptor
// ---------------------------------------------------------------------------

/// What one effect is, what it needs, and which lanes exist for it.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectDescriptor {
    pub id: EffectId,
    /// Vendor PF aliases that resolve to this effect.
    pub pfs: Vec<VendorPf>,
    /// Parameters the effect consumes. Empty is allowed when no parameter
    /// metadata is recovered; never guess a range to fill it.
    pub params: Vec<ParamSpec>,
    /// Every same-job input the effect needs.
    pub inputs: Vec<InputDependency>,
    /// The CPU reference entry point that defines correct output.
    pub cpu_reference: &'static str,
    /// Lanes for which an adapter exists in this repository.
    pub backends: Vec<BackendKind>,
    pub order_constraints: OrderConstraints,
    /// Provenance, uncertainties and known gaps. Keep claims factual.
    pub notes: &'static str,
}

impl EffectDescriptor {
    /// `Compiled` only when the descriptor lists the backend *and* the adapter
    /// body is built on this target.
    pub fn static_state(&self, backend: BackendKind) -> StaticState {
        if self.backends.contains(&backend) {
            StaticState::for_backend(backend)
        } else {
            StaticState::Absent
        }
    }

    /// Per-backend capability. Evidence is supplied by the caller (normally
    /// read from [`crate::capability_registry`]); runtime is supplied by the
    /// host that just checked driver/model/asset/input availability.
    pub fn capability(
        &self,
        backend: BackendKind,
        evidence: EvidenceState,
        runtime: RuntimeState,
    ) -> CapabilityStatus {
        CapabilityStatus::from(self.static_state(backend), evidence, runtime)
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Typed registration failure. Duplicate ids or PFs are programming errors,
/// not runtime conditions, so registration returns an error instead of
/// silently keeping the first row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegistryError {
    /// `EffectId::0` was empty.
    EmptyId,
    /// Two descriptors use the same stable id.
    DuplicateId(&'static str),
    /// Two descriptors (or two aliases of one descriptor) use the same PF.
    DuplicatePf(u32),
}

/// Lookup table over descriptors, inserted once at startup.
#[derive(Clone, Debug, Default)]
pub struct EffectRegistry {
    descriptors: Vec<EffectDescriptor>,
}

impl EffectRegistry {
    /// Register a slice of descriptors. The slice is copied; the registry owns
    /// the rows so callers can build them at runtime (`seed_descriptors()`)
    /// without leaking them.
    ///
    /// Rejects empty ids, duplicate ids, and duplicate vendor PFs (including a
    /// PF repeated inside one descriptor). Lookup is linear; the table is
    /// small and this preserves insertion order deterministically.
    pub fn from_descriptors(descriptors: &[EffectDescriptor]) -> Result<Self, RegistryError> {
        let mut ids: BTreeSet<&'static str> = BTreeSet::new();
        let mut pfs: BTreeSet<u32> = BTreeSet::new();
        let mut owned = Vec::with_capacity(descriptors.len());
        for descriptor in descriptors {
            if descriptor.id.0.trim().is_empty() {
                return Err(RegistryError::EmptyId);
            }
            if !ids.insert(descriptor.id.0) {
                return Err(RegistryError::DuplicateId(descriptor.id.0));
            }
            for pf in &descriptor.pfs {
                if !pfs.insert(pf.0) {
                    return Err(RegistryError::DuplicatePf(pf.0));
                }
            }
            owned.push(descriptor.clone());
        }
        Ok(Self { descriptors: owned })
    }

    /// The seeded descriptors, validated. Fails only if the seed table is
    /// edited into an inconsistent state.
    pub fn seeded() -> Result<Self, RegistryError> {
        Self::from_descriptors(&seed_descriptors())
    }

    pub fn get(&self, id: EffectId) -> Option<&EffectDescriptor> {
        self.descriptors.iter().find(|d| d.id == id)
    }

    pub fn by_pf(&self, pf: VendorPf) -> Option<&EffectDescriptor> {
        self.descriptors
            .iter()
            .find(|d| d.pfs.iter().any(|alias| *alias == pf))
    }

    pub fn iter(&self) -> impl Iterator<Item = &EffectDescriptor> {
        self.descriptors.iter()
    }

    pub fn len(&self) -> usize {
        self.descriptors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.descriptors.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Seed data
// ---------------------------------------------------------------------------

/// The six effects seeded in this patch.
///
/// Selection rule: the effect has a real CPU kernel, a row in
/// [`crate::capability_registry`], and a documented default in the existing
/// parameter metadata. The eight rows with ev5 (`main_preview`) evidence are
/// the best-evidenced set; the two omitted here (PF90004 radiance and PF90007
/// rosy) follow the same shape and can be added without changing the API.
///
/// Defaults cross-checked against:
/// * `docs/frontend-contract/portrait-algorithms.json` (`default` per id, where
///   present);
/// * the app's live lists (`crates/app/src/tools/portrait.rs`,
///   `WIRED_SKIN_EFFECTS`) for ids the JSON does not list;
/// * the kernel entry points' validation ranges.
///
/// Each seeded parameter is the *stored* `pc:<pf>` degree the engine consumes
/// (`0.0..=1.0`); the UI displays it as `0..=100`. Effects with a `0.5` neutral
/// use `0.5` as identity.
pub fn seed_descriptors() -> Vec<EffectDescriptor> {
    vec![
        EffectDescriptor {
            id: EffectId("skin.gloss"),
            pfs: vec![VendorPf(8068)],
            params: vec![ParamSpec {
                name: "degree",
                unit: ParamUnit::Amount,
                min: 0.0,
                max: 1.0,
                default: 0.5,
                note: "stored pc:8068; engine maps to native_percent = degree*100 \
                       (0..100, neutral 50). UI shows 0..100.",
            }],
            inputs: vec![InputDependency::FaceParsingMask],
            cpu_reference: "kernels::skin_models::gloss_apply",
            backends: vec![BackendKind::Cpu, BackendKind::Cuda],
            order_constraints: OrderConstraints::none(),
            notes: "Three R8 mask passes plus three extra R8 masks (caller-supplied). \
                    PF8068 has no row in portrait-algorithms.json; default 0.5 comes \
                    from the app live-list neutral and the kernel's native_percent=50 \
                    neutral. The unresolved UI writer is not invented. CUDA adapter: \
                    cuda_gloss / effect_router::Gloss.",
        },
        EffectDescriptor {
            id: EffectId("skin.texture"),
            pfs: vec![VendorPf(8069)],
            params: vec![ParamSpec {
                name: "degree",
                unit: ParamUnit::Amount,
                min: 0.0,
                max: 1.0,
                default: 0.5,
                note: "stored pc:8069; 0.5 neutral. <0.5 selects the surface-blur \
                       branch; >0.5 selects the FilterType45 material branch. UI \
                       shows 0..100.",
            }],
            inputs: vec![
                InputDependency::FaceParsingMask,
                InputDependency::SkinMaterial,
            ],
            cpu_reference: "kernels::skin_models::texture_apply / texture_surface_blur",
            backends: vec![BackendKind::Cpu, BackendKind::Cuda],
            order_constraints: OrderConstraints::none(),
            notes: "Positive branch (>0.5) needs a caller-supplied FilterType45 \
                    material layer and is not wired: apply_skin_texture returns \
                    MissingInput. Only the negative surface-blur branch runs today. \
                    shared_effect_param with FaceSticker/FaceSurfaceBlur per \
                    portrait-algorithms.json. CUDA adapter: cuda_skin_models (windows \
                    body); the registry still records ev3 as 未验证 and gpu_backend as \
                    探针 CPU, so the adapter is compiled but not evidenced.",
        },
        EffectDescriptor {
            id: EffectId("skin.clarity"),
            pfs: vec![VendorPf(8070)],
            params: vec![ParamSpec {
                name: "degree",
                unit: ParamUnit::Amount,
                min: 0.0,
                max: 1.0,
                default: 0.0,
                note: "stored pc:8070; clamped to 0..1 by the engine. Default 0.0 \
                       from the app live list.",
            }],
            inputs: vec![
                InputDependency::FaceParsingMask,
                InputDependency::FaceRoi,
            ],
            cpu_reference: "kernels::skin_models::clarity_apply",
            backends: vec![BackendKind::Cpu, BackendKind::Cuda],
            order_constraints: OrderConstraints::none(),
            notes: "Fixed config constants threshold=0.0 and factor=4.68 (kernel \
                    doc). The product call uses the full frame as the ROI; the kernel \
                    itself accepts an explicit ROI. CUDA: cuda_skin_models (windows \
                    body).",
        },
        EffectDescriptor {
            id: EffectId("skin.color_hue"),
            pfs: vec![VendorPf(21015)],
            params: vec![ParamSpec {
                name: "degree",
                unit: ParamUnit::Amount,
                min: 0.0,
                max: 1.0,
                default: 0.5,
                note: "stored pc:21015; 0.5 neutral. Engine maps to signed UI \
                       ((degree-0.5)*200) and picks the green (<=0) or magenta (>0) \
                       LUT. Default 0.5 from portrait-algorithms.json.",
            }],
            inputs: vec![
                InputDependency::FaceParsingMask,
                InputDependency::SkinMaterial,
            ],
            cpu_reference: "kernels::face_detail::skin_tint",
            backends: vec![BackendKind::Cpu, BackendKind::Cuda],
            order_constraints: OrderConstraints::none(),
            notes: "Green/magenta tint LUTs are external assets resolved at runtime; \
                    the CPU reference refuses with MissingInput when either is absent. \
                    CUDA adapter: cuda_lut::SkinLut / effect_router::SkinTint.",
        },
        EffectDescriptor {
            id: EffectId("skin.temperature"),
            pfs: vec![VendorPf(30096)],
            params: vec![ParamSpec {
                name: "degree",
                unit: ParamUnit::Amount,
                min: 0.0,
                max: 1.0,
                default: 0.5,
                note: "stored pc:30096; 0.5 neutral. Engine maps to signed UI \
                       ((degree-0.5)*200) and picks the cold (<=0) or warm (>0) LUT. \
                       Default 0.5 from portrait-algorithms.json.",
            }],
            inputs: vec![
                InputDependency::FaceParsingMask,
                InputDependency::SkinMaterial,
            ],
            cpu_reference: "kernels::face_detail::skin_temperature",
            backends: vec![BackendKind::Cpu, BackendKind::Cuda],
            order_constraints: OrderConstraints::none(),
            notes: "The capability registry still has key/label as 未验证 for this PF; \
                    the app maps PF30096 -> sediao (evidence.rs catalog_key). Cold/warm \
                    LUTs are external runtime assets. CUDA adapter: cuda_lut; the \
                    registry records ev3/gpu_backend as 未验证, so the adapter is \
                    compiled but not evidenced.",
        },
        EffectDescriptor {
            id: EffectId("skin.whitening"),
            pfs: vec![VendorPf(90005)],
            params: vec![ParamSpec {
                name: "degree",
                unit: ParamUnit::Amount,
                min: 0.0,
                max: 1.0,
                default: 0.0,
                note: "stored pc:90005; clamped to 0..1 and scales the whiten LUT. \
                       Default 0.0 from portrait-algorithms.json.",
            }],
            inputs: vec![
                InputDependency::FaceParsingMask,
                InputDependency::SkinMaterial,
            ],
            cpu_reference: "kernels::makeup::skin_adjust_lut",
            backends: vec![BackendKind::Cpu, BackendKind::Cuda],
            order_constraints: OrderConstraints::none(),
            notes: "Whiten LUT is an external runtime asset; the CPU reference refuses \
                    with MissingInput when absent. CUDA adapter: \
                    effect_router::SkinAdjustLut (also serves PF90007).",
        },
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability_registry;

    fn simple(id: &'static str, pf: u32) -> EffectDescriptor {
        EffectDescriptor {
            id: EffectId(id),
            pfs: vec![VendorPf(pf)],
            params: Vec::new(),
            inputs: Vec::new(),
            cpu_reference: "kernels::test",
            backends: vec![BackendKind::Cpu],
            order_constraints: OrderConstraints::none(),
            notes: "",
        }
    }

    #[test]
    fn duplicate_id_is_rejected() {
        let rows = [simple("skin.gloss", 8068), simple("skin.gloss", 90005)];
        assert_eq!(
            EffectRegistry::from_descriptors(&rows).unwrap_err(),
            RegistryError::DuplicateId("skin.gloss")
        );
    }

    #[test]
    fn duplicate_pf_is_rejected_across_and_within_descriptors() {
        let across = [simple("a", 8068), simple("b", 8068)];
        assert_eq!(
            EffectRegistry::from_descriptors(&across).unwrap_err(),
            RegistryError::DuplicatePf(8068)
        );

        let mut within = simple("a", 8068);
        within.pfs.push(VendorPf(8068));
        assert_eq!(
            EffectRegistry::from_descriptors(&[within]).unwrap_err(),
            RegistryError::DuplicatePf(8068)
        );
    }

    #[test]
    fn empty_id_is_rejected() {
        let mut row = simple("a", 8068);
        row.id = EffectId("   ");
        assert_eq!(
            EffectRegistry::from_descriptors(&[row]).unwrap_err(),
            RegistryError::EmptyId
        );
    }

    #[test]
    fn lookup_by_id_and_by_pf() {
        let registry = EffectRegistry::seeded().expect("seed registry");
        for descriptor in registry.iter() {
            assert_eq!(registry.get(descriptor.id), Some(descriptor));
            for pf in &descriptor.pfs {
                assert_eq!(registry.by_pf(*pf), Some(descriptor));
            }
        }
        assert!(registry.get(EffectId("missing.effect")).is_none());
        assert!(registry.by_pf(VendorPf(999_999)).is_none());
    }

    #[test]
    fn every_seeded_pf_has_a_capability_registry_row() {
        let expected: &[(u32, &str)] = &[
            (8068, "skinGloss"),
            (8069, "skinTexture"),
            (8070, "skinClarity"),
            (21015, "skinColorHue"),
            (30096, "sediao"), // capability registry row is now recovered (2026-11-25)
            (90005, "skinWhitening"),
        ];
        let mut seen = BTreeSet::new();
        for descriptor in seed_descriptors() {
            for pf in &descriptor.pfs {
                assert!(seen.insert(pf.0), "seed repeats PF{}", pf.0);
                let row = capability_registry::for_pf(pf.0)
                    .unwrap_or_else(|| panic!("PF{} missing from capability registry", pf.0));
                assert!(!row.is_test_gap(), "PF{} resolved to a test_gap row", pf.0);
                let expected_key = expected
                    .iter()
                    .find(|(id, _)| *id == pf.0)
                    .map(|(_, key)| *key)
                    .expect("seeded PF not in the expected table");
                assert_eq!(row.key, expected_key, "PF{} key mismatch", pf.0);
            }
        }
        assert_eq!(seen.len(), expected.len());
    }

    #[test]
    fn every_seeded_descriptor_is_internally_consistent() {
        let registry = EffectRegistry::seeded().expect("seed registry");
        assert_eq!(registry.len(), 6);
        for descriptor in registry.iter() {
            assert!(!descriptor.id.0.is_empty());
            assert!(!descriptor.pfs.is_empty(), "{:?} has no PF", descriptor.id);
            assert!(
                !descriptor.cpu_reference.is_empty(),
                "{:?} has no CPU reference",
                descriptor.id
            );
            assert!(
                !descriptor.backends.is_empty(),
                "{:?} has no backend",
                descriptor.id
            );
            assert!(
                descriptor.backends.contains(&BackendKind::Cpu),
                "{:?} must advertise the recovered CPU kernel",
                descriptor.id
            );
            assert!(
                !descriptor.inputs.is_empty(),
                "{:?} needs at least one same-job input",
                descriptor.id
            );
            for param in &descriptor.params {
                assert!(param.is_well_formed(), "bad param: {param:?}");
                assert!(!param.note.is_empty(), "param without note: {param:?}");
            }
            for input in &descriptor.inputs {
                assert!(!input.describe().is_empty());
            }
        }
    }

    #[test]
    fn capability_dimensions_can_disagree() {
        // Compiled CPU kernel, ev5 evidence, but the LUT asset is missing now.
        let now_unavailable = CapabilityStatus::from(
            StaticState::Compiled,
            EvidenceState::documented(EvidenceLevel::MainPreview),
            RuntimeState::unavailable("skin LUT asset not loaded"),
        );
        assert_eq!(now_unavailable.static_state, StaticState::Compiled);
        assert!(now_unavailable.evidence.is_documented());
        assert!(!now_unavailable.runtime.available);
        assert_eq!(
            now_unavailable.runtime.reason,
            Some("skin LUT asset not loaded")
        );

        // Compiled and runnable, but no documented evidence yet.
        let undocumented = CapabilityStatus::from(
            StaticState::Compiled,
            EvidenceState::NONE,
            RuntimeState::available(),
        );
        assert!(undocumented.runtime.available);
        assert!(!undocumented.evidence.is_documented());

        // Documented evidence, but the adapter is absent from this build.
        let not_built = CapabilityStatus::from(
            StaticState::Absent,
            EvidenceState::documented(EvidenceLevel::GpuKernel),
            RuntimeState::unavailable("adapter not built for this target"),
        );
        assert_eq!(not_built.static_state, StaticState::Absent);
        assert_eq!(
            not_built.evidence.registry_index(),
            Some(EvidenceLevel::GpuKernel.index())
        );

        // The three answers are not one bool: different tuples, same length.
        assert_ne!(now_unavailable, undocumented);
        assert_ne!(undocumented, not_built);
    }

    #[test]
    fn descriptor_static_state_respects_the_backend_list() {
        let descriptor = simple("a", 8068);
        assert_eq!(descriptor.static_state(BackendKind::Cpu), StaticState::Compiled);
        assert_eq!(descriptor.static_state(BackendKind::Cuda), StaticState::Absent);
        assert_eq!(descriptor.static_state(BackendKind::Metal), StaticState::Absent);
    }

    #[test]
    fn param_clamping_and_validation() {
        let spec = ParamSpec {
            name: "degree",
            unit: ParamUnit::Amount,
            min: 0.0,
            max: 1.0,
            default: 0.5,
            note: "test",
        };
        assert!(spec.is_well_formed());
        assert_eq!(spec.clamp(-0.25), 0.0);
        assert_eq!(spec.clamp(1.25), 1.0);
        assert_eq!(spec.clamp(0.25), 0.25);
        assert_eq!(spec.clamp(f32::NAN), 0.5);
        assert_eq!(spec.clamp(f32::NEG_INFINITY), 0.0);
        assert_eq!(spec.clamp(f32::INFINITY), 1.0);
        assert!(spec.is_valid(0.0));
        assert!(spec.is_valid(1.0));
        assert!(!spec.is_valid(-0.1));
        assert!(!spec.is_valid(f32::NAN));

        let malformed = ParamSpec { min: 1.0, max: 0.0, ..spec };
        assert!(!malformed.is_well_formed());
        assert!(!malformed.is_valid(0.5));
        assert_eq!(malformed.clamp(0.5), malformed.default);
    }

    #[test]
    fn evidence_level_names_match_the_capability_registry() {
        assert_eq!(EvidenceLevel::ALL.len(), capability_registry::EVIDENCE_LEVELS.len());
        for level in EvidenceLevel::ALL {
            assert_eq!(
                level.name(),
                capability_registry::EVIDENCE_LEVELS[level.index()]
            );
        }
    }

    #[test]
    fn seeded_registry_is_deterministic() {
        let first: Vec<(EffectId, Vec<VendorPf>)> = EffectRegistry::seeded()
            .unwrap()
            .iter()
            .map(|d| (d.id, d.pfs.clone()))
            .collect();
        let second: Vec<(EffectId, Vec<VendorPf>)> = EffectRegistry::seeded()
            .unwrap()
            .iter()
            .map(|d| (d.id, d.pfs.clone()))
            .collect();
        assert_eq!(first, second);
        assert_eq!(first.first().map(|(id, _)| id.0), Some("skin.gloss"));
    }

    #[test]
    fn seeded_capabilities_keep_dimensions_separate() {
        // PF90005 has a real CPU kernel and ev5 evidence in the registry, but the
        // whiten LUT is a runtime asset: the descriptor itself must not claim it
        // is runnable today.
        let registry = EffectRegistry::seeded().unwrap();
        let descriptor = registry.by_pf(VendorPf(90005)).unwrap();
        let row = capability_registry::for_pf(90005).unwrap();
        let level = if row.has_evidence(4) {
            EvidenceLevel::MainPreview
        } else {
            EvidenceLevel::KernelExists
        };
        let status = descriptor.capability(
            BackendKind::Cpu,
            EvidenceState::documented(level),
            RuntimeState::unavailable("whiten LUT asset not loaded"),
        );
        assert_eq!(status.static_state, StaticState::Compiled);
        assert!(status.evidence.is_documented());
        assert!(!status.runtime.available);
    }
}
