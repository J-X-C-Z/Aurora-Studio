//! One structured capability view for the interface.
//!
//! This module is the app-side half of ADR-007's three independent dimensions.
//! It mirrors `aurora_portrait::effect_descriptor` rather than importing it:
//! `portrait-app` depends only on `aurora-engine` (see `crates/app/Cargo.toml`)
//! and this lane does not add a crate dependency.
//!
//! | app type | `effect_descriptor` counterpart | question |
//! |---|---|---|
//! | [`StaticState`] | `StaticState` | is the adapter / control compiled into this build? |
//! | [`EvidenceState`], [`EvidenceLevel`] | `EvidenceState`, `EvidenceLevel` | what does the registry document? |
//! | [`RuntimeState`] | `RuntimeState` | can it run *right now*? |
//!
//! `EvidenceLevel` keeps the descriptor's names, discriminants and seven
//! registry tokens; [`EvidenceLevel::ALL`] is cross-checked against
//! [`crate::evidence::LEVELS`] in the tests. Never collapse the three answers
//! into one `ready()` bool: compiled is not evidenced is not available now.
//!
//! # Sources
//!
//! [`CapabilityView::from_engine`] merges three sources into one deterministic
//! vector:
//!
//! 1. `ImageEngine::capabilities()` and `ImageEngine::portrait_capabilities()`
//!    — current runtime availability, read through the passed engine;
//! 2. the panel's `WIRED_*` rows (passed in — this module never reads the
//!    private lists) — static implementation in this build;
//! 3. the `contract::Status` constants and the embedded registry in
//!    [`crate::evidence`] — documented state.
//!
//! The dimensions are allowed to disagree. A wired effect the engine does not
//! expose is `Compiled` and documented but not runnable; an evidence-only
//! effect is documented while `Absent` from this build. Nothing here changes
//! the UI: this is the interface the `WIRED_*` lists and status constants can
//! migrate onto.
//!
//! # Not a source of truth yet
//!
//! Rows are derived, never edited. The registry TSV stays the evidence source,
//! `WIRED_*` stays the panel-grouping source, and `ImageEngine` stays the
//! runtime source. `id`/labels are owned strings because 38 registry rows have
//! no static key and left/right rows reuse one.

#![allow(dead_code)]

use std::collections::BTreeMap;

use aurora_engine::{EngineCapabilities, ImageEngine, PortraitCapability};

use crate::contract::{self, Status};
use crate::evidence;

// ---------------------------------------------------------------------------
// The three dimensions
// ---------------------------------------------------------------------------

/// Is the adapter / control compiled into this build?
///
/// Mirrors `aurora_portrait::effect_descriptor::StaticState`. `Compiled` does
/// not mean usable; it says nothing about a driver, model or input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum StaticState {
    /// The implementation (or the wired control) is present in this build.
    Compiled,
    /// It is not built for this target, or not wired here.
    #[default]
    Absent,
}

/// The seven evidence levels, matching [`crate::evidence::LEVELS`] by index and
/// name (and `effect_descriptor::EvidenceLevel` by discriminant).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub(crate) enum EvidenceLevel {
    KernelExists = 0,
    CpuReference = 1,
    GpuKernel = 2,
    SameJobInputs = 3,
    MainPreview = 4,
    VendorParity = 5,
    RealPhotoAccept = 6,
}

impl EvidenceLevel {
    pub(crate) const ALL: [Self; 7] = [
        Self::KernelExists,
        Self::CpuReference,
        Self::GpuKernel,
        Self::SameJobInputs,
        Self::MainPreview,
        Self::VendorParity,
        Self::RealPhotoAccept,
    ];

    /// Same token as `evidence::LEVELS` / `effect_descriptor::EvidenceLevel`.
    pub(crate) fn name(self) -> &'static str {
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

    pub(crate) fn index(self) -> usize {
        self as usize
    }
}

/// Documented evidence, at the highest level the registry supports.
///
/// Documentation is not a live result: an ev5 row can still be unavailable at
/// runtime, and a missing level proves nothing about higher levels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct EvidenceState {
    pub level: Option<EvidenceLevel>,
}

impl EvidenceState {
    /// No evidence is documented (the registry says `未验证`).
    pub(crate) const NONE: Self = Self { level: None };

    pub(crate) const fn documented(level: EvidenceLevel) -> Self {
        Self { level: Some(level) }
    }

    pub(crate) fn is_documented(self) -> bool {
        self.level.is_some()
    }

    /// Index into [`crate::evidence::LEVELS`].
    pub(crate) fn registry_index(self) -> Option<usize> {
        self.level.map(EvidenceLevel::index)
    }
}

/// Can the effect run right now?
///
/// `available == false` carries a reason (missing driver, model, LUT asset or
/// same-job input). `unavailable` means "not established", never "verified
/// broken": the app has no runtime probe for most non-portrait features.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RuntimeState {
    pub available: bool,
    pub reason: Option<&'static str>,
}

impl RuntimeState {
    pub(crate) const fn available() -> Self {
        Self { available: true, reason: None }
    }

    pub(crate) const fn unavailable(reason: &'static str) -> Self {
        Self { available: false, reason: Some(reason) }
    }
}

// ---------------------------------------------------------------------------
// Rows
// ---------------------------------------------------------------------------

/// Which source produced the row. Kept explicit so callers can migrate the UI
/// one group at a time without re-deriving it from labels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum CapabilityKind {
    /// A portrait effect keyed by a vendor PF (wired, engine or evidence row).
    #[default]
    Portrait,
    /// One of the hand-written `contract::Status` feature constants.
    Documented,
    /// A boolean from `EngineCapabilities` with no portrait PF.
    Engine,
}

/// One effect or feature with its three answers kept separate.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EffectCapabilityView {
    /// Stable lookup key. Wired rows use their `pc:<pf>` key; unwired portrait
    /// rows use `pf:<pf>`; status constants use `status:<name>`; engine
    /// booleans use `engine:<name>`.
    pub id: String,
    pub label_en: String,
    pub label_zh: String,
    /// Vendor PF for portrait rows; `None` for status/engine features.
    pub pf: Option<u32>,
    pub kind: CapabilityKind,
    pub static_state: StaticState,
    pub evidence: EvidenceState,
    pub runtime: RuntimeState,
    /// The matching `contract::Status` constant, when this row is documented by
    /// one. Portrait effects are documented by the evidence registry instead.
    pub documented_status: Option<Status>,
    /// Engine stage from `PortraitCapability::stage` or the registry TSV.
    pub stage: Option<&'static str>,
    /// `PortraitCapability::engine_bridge`, when the engine reported the row.
    pub engine_bridge: Option<bool>,
    /// `PortraitCapability::automatic`, when the engine reported the row.
    pub automatic: Option<bool>,
}

// ---------------------------------------------------------------------------
// Wired rows
// ---------------------------------------------------------------------------

/// One row of a panel's `WIRED_*` list.
///
/// `portrait-app` owns the lists privately in `tools::portrait`; this module
/// does not read them. Callers pass the rows in, so the lists stay read-only
/// and this round changes no UI behavior.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WiredEffect {
    /// Stable key, normally `pc:<pf>`.
    pub id: &'static str,
    pub label_en: &'static str,
    pub label_zh: &'static str,
    /// Vendor PF parsed from `id` by [`Self::from_key`], or supplied directly.
    pub pf: Option<u32>,
}

impl WiredEffect {
    pub(crate) const fn new(
        id: &'static str,
        label_en: &'static str,
        label_zh: &'static str,
        pf: Option<u32>,
    ) -> Self {
        Self { id, label_en, label_zh, pf }
    }

    /// Build from a `pc:<pf>` (or bare `<pf>`) key.
    pub(crate) fn from_key(id: &'static str, label_en: &'static str, label_zh: &'static str) -> Self {
        let pf = id
            .strip_prefix("pc:")
            .or_else(|| id.strip_prefix("pf:"))
            .and_then(|number| number.parse::<u32>().ok())
            .or_else(|| id.parse::<u32>().ok());
        Self { id, label_en, label_zh, pf }
    }
}

impl From<&(&'static str, &'static str, &'static str)> for WiredEffect {
    fn from(row: &(&'static str, &'static str, &'static str)) -> Self {
        Self::from_key(row.0, row.1, row.2)
    }
}

impl From<&(&'static str, &'static str, &'static str, f32)> for WiredEffect {
    fn from(row: &(&'static str, &'static str, &'static str, f32)) -> Self {
        Self::from_key(row.0, row.1, row.2)
    }
}

// ---------------------------------------------------------------------------
// Documented status constants
// ---------------------------------------------------------------------------

/// `contract::Status` plus the stable id and labels the unified view needs.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DocumentedStatus {
    pub id: &'static str,
    pub label_en: &'static str,
    pub label_zh: &'static str,
    pub status: Status,
}

/// The `contract::Status` constants, mapped to stable ids.
///
/// This table duplicates the constant *names*, not their values: the status
/// text still lives only in `contract.rs`, so the badge wording cannot drift.
pub(crate) const DOCUMENTED_STATUSES: &[DocumentedStatus] = &[
    DocumentedStatus { id: "status:color_editor", label_en: "Color editor", label_zh: "色彩编辑器", status: contract::COLOR_EDITOR },
    DocumentedStatus { id: "status:moire", label_en: "Moire", label_zh: "摩尔纹", status: contract::MOIRE },
    DocumentedStatus { id: "status:lens_correction", label_en: "Lens correction", label_zh: "镜头校正", status: contract::LENS_CORRECTION },
    DocumentedStatus { id: "status:keystone", label_en: "Keystone", label_zh: "梯形校正", status: contract::KEYSTONE },
    DocumentedStatus { id: "status:local_adjustments", label_en: "Local adjustments", label_zh: "局部调整", status: contract::LOCAL_ADJUSTMENTS },
    DocumentedStatus { id: "status:node_graph", label_en: "Node graph", label_zh: "节点图", status: contract::NODE_GRAPH },
    DocumentedStatus { id: "status:spots", label_en: "Spot removal", label_zh: "污点修复", status: contract::SPOTS },
    DocumentedStatus { id: "status:c1_catalog", label_en: "Capture One catalog", label_zh: "Capture One 目录", status: contract::C1_CATALOG },
    DocumentedStatus { id: "status:ai_mask", label_en: "AI masks", label_zh: "AI 蒙版", status: contract::AI_MASK },
    DocumentedStatus { id: "status:tether", label_en: "Tethered capture", label_zh: "联机拍摄", status: contract::TETHER },
    DocumentedStatus { id: "status:display_profile", label_en: "Display profile", label_zh: "显示器配置", status: contract::DISPLAY_PROFILE },
    DocumentedStatus { id: "status:ai_denoise", label_en: "AI denoise", label_zh: "AI 降噪", status: contract::AI_DENOISE },
    DocumentedStatus { id: "status:banding", label_en: "Banding reduction", label_zh: "条纹抑制", status: contract::BANDING },
    DocumentedStatus { id: "status:merge", label_en: "HDR / panorama merge", label_zh: "HDR / 全景合并", status: contract::MERGE },
    DocumentedStatus { id: "status:style_import", label_en: "Style import", label_zh: "风格导入", status: contract::STYLE_IMPORT },
    DocumentedStatus { id: "status:sessions", label_en: "Sessions", label_zh: "Session", status: contract::SESSIONS },
    DocumentedStatus { id: "status:print", label_en: "Print", label_zh: "打印", status: contract::PRINT },
    DocumentedStatus { id: "status:retouch", label_en: "Retouch", label_zh: "修饰", status: contract::RETOUCH },
    DocumentedStatus { id: "status:surface_noise", label_en: "Surface noise", label_zh: "表面噪点", status: contract::SURFACE_NOISE },
    DocumentedStatus { id: "status:soft_proof", label_en: "Soft proof", label_zh: "软打样", status: contract::SOFT_PROOF },
    DocumentedStatus { id: "status:lens_blur", label_en: "Lens blur", label_zh: "镜头模糊", status: contract::LENS_BLUR },
    DocumentedStatus { id: "status:metal", label_en: "Metal GPU", label_zh: "Metal GPU", status: contract::METAL },
];

// ---------------------------------------------------------------------------
// View
// ---------------------------------------------------------------------------

/// A deterministic, id/pf-sorted set of [`EffectCapabilityView`] rows.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct CapabilityView {
    rows: Vec<EffectCapabilityView>,
}

impl CapabilityView {
    /// Merge the engine's runtime view with the panel's wired rows, the
    /// documented status constants and the evidence registry.
    ///
    /// `engine` is passed explicitly; nothing here reads a global engine.
    pub(crate) fn from_engine(engine: &dyn ImageEngine, wired: &[WiredEffect]) -> Self {
        Self::from_sources(engine.capabilities(), engine.portrait_capabilities(), wired)
    }

    /// Pure merge used by [`Self::from_engine`] and by tests that must not
    /// build a GPUI application.
    pub(crate) fn from_sources(
        caps: EngineCapabilities,
        portrait_capabilities: Vec<PortraitCapability>,
        wired: &[WiredEffect],
    ) -> Self {
        let mut portrait: BTreeMap<u32, Row> = BTreeMap::new();
        let mut other: BTreeMap<String, Row> = BTreeMap::new();

        // 1. WIRED_* rows: static implementation in this build. Processed
        //    first so the hand-written labels win over generated ones.
        for effect in wired {
            let mut row = Row {
                label_en: Some(effect.label_en.to_owned()),
                label_zh: Some(effect.label_zh.to_owned()),
                kind: CapabilityKind::Portrait,
                static_state: StaticState::Compiled,
                ..Row::default()
            };
            row.runtime = Some(RuntimeState::unavailable("engine does not expose this effect"));
            match effect.pf {
                Some(pf) => {
                    let entry = portrait.entry(pf).or_default();
                    entry.pf = Some(pf);
                    entry.kind = CapabilityKind::Portrait;
                    entry.static_state = StaticState::Compiled;
                    if entry.id.is_none() {
                        entry.id = Some(effect.id.to_owned());
                    }
                    entry.fill_label(effect.label_en, effect.label_zh);
                    if entry.runtime.is_none() {
                        entry.runtime = row.runtime;
                    }
                }
                None => {
                    let id = effect.id.to_owned();
                    row.id = Some(id.clone());
                    other.insert(id, row);
                }
            }
        }

        // 2. Evidence registry: documented state for every PF it lists, even
        //    when no control and no engine bridge exists.
        for effect in evidence::all() {
            let entry = portrait.entry(effect.pf).or_default();
            entry.pf = Some(effect.pf);
            entry.kind = CapabilityKind::Portrait;
            entry.merge_evidence(highest_level(effect));
            let (label_en, label_zh) = evidence_labels(effect);
            if let Some(label_en) = label_en {
                entry.fill_labels_if_empty(&label_en, label_zh.as_deref().unwrap_or(""));
            }
            if entry.stage.is_none() {
                entry.stage = registry_text(effect.stage);
            }
        }

        // 3. Engine portrait capabilities: current runtime availability, and
        //    the static proof that the adapter is built into this engine.
        for capability in &portrait_capabilities {
            let entry = portrait.entry(capability.pf).or_default();
            entry.pf = Some(capability.pf);
            entry.kind = CapabilityKind::Portrait;
            entry.static_state = StaticState::Compiled;
            entry.runtime = Some(if capability.engine_bridge {
                RuntimeState::available()
            } else {
                RuntimeState::unavailable("engine bridge is disabled")
            });
            entry.stage = Some(capability.stage);
            entry.engine_bridge = Some(capability.engine_bridge);
            entry.automatic = Some(capability.automatic);
            entry.fill_labels_if_empty(&capability.label, "");
        }

        let mut rows: Vec<EffectCapabilityView> = portrait
            .into_iter()
            .map(|(pf, row)| row.into_view(format!("pf:{pf}")))
            .collect();

        // 4. Status constants: documented features with their own dimension.
        for documented in DOCUMENTED_STATUSES {
            other.insert(
                documented.id.to_owned(),
                Row {
                    id: Some(documented.id.to_owned()),
                    label_en: Some(documented.label_en.to_owned()),
                    label_zh: Some(documented.label_zh.to_owned()),
                    kind: CapabilityKind::Documented,
                    static_state: static_from_status(documented.status),
                    evidence: EvidenceState::NONE,
                    runtime: Some(runtime_for_status(documented.id, documented.status, &caps)),
                    documented_status: Some(documented.status),
                    ..Row::default()
                },
            );
        }

        // 5. Engine booleans that have no contract::Status and no portrait PF.
        for row in engine_feature_rows(&caps) {
            if let Some(id) = &row.id {
                other.insert(id.clone(), row);
            }
        }

        rows.extend(other.into_values().map(|row| row.into_view(String::new())));
        rows.sort_by(|left, right| {
            let left_key = (left.pf.unwrap_or(u32::MAX), left.id.as_str());
            let right_key = (right.pf.unwrap_or(u32::MAX), right.id.as_str());
            left_key.cmp(&right_key)
        });

        #[cfg(debug_assertions)]
        {
            let mut ids = std::collections::BTreeSet::new();
            let mut pfs = std::collections::BTreeSet::new();
            for row in &rows {
                debug_assert!(ids.insert(row.id.as_str()), "duplicate capability id {}", row.id);
                if let Some(pf) = row.pf {
                    debug_assert!(pfs.insert(pf), "duplicate capability PF{pf}");
                }
            }
        }

        Self { rows }
    }

    pub(crate) fn rows(&self) -> &[EffectCapabilityView] {
        &self.rows
    }

    pub(crate) fn len(&self) -> usize {
        self.rows.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub(crate) fn get_by_pf(&self, pf: u32) -> Option<&EffectCapabilityView> {
        self.rows.iter().find(|row| row.pf == Some(pf))
    }

    pub(crate) fn get_by_id(&self, id: &str) -> Option<&EffectCapabilityView> {
        self.rows.iter().find(|row| row.id == id)
    }

    pub(crate) fn portrait_rows(&self) -> impl Iterator<Item = &EffectCapabilityView> {
        self.rows.iter().filter(|row| row.kind == CapabilityKind::Portrait)
    }

    pub(crate) fn documented_rows(&self) -> impl Iterator<Item = &EffectCapabilityView> {
        self.rows.iter().filter(|row| row.kind == CapabilityKind::Documented)
    }

    pub(crate) fn engine_rows(&self) -> impl Iterator<Item = &EffectCapabilityView> {
        self.rows.iter().filter(|row| row.kind == CapabilityKind::Engine)
    }
}

/// Build a view from already-collected sources, without an engine or GPUI.
///
/// Same result as [`CapabilityView::from_engine`] for the given inputs; tests
/// use it to keep engine/UI setup out of the capability assertions.
pub(crate) fn capability_snapshot_for_tests(
    caps: EngineCapabilities,
    portrait_capabilities: Vec<PortraitCapability>,
    wired: &[WiredEffect],
) -> CapabilityView {
    CapabilityView::from_sources(caps, portrait_capabilities, wired)
}

// ---------------------------------------------------------------------------
// Merge helpers
// ---------------------------------------------------------------------------

/// One row being merged. Sources fill fields in priority order; the three
/// dimensions are stored separately until [`Row::into_view`].
#[derive(Default)]
struct Row {
    id: Option<String>,
    label_en: Option<String>,
    label_zh: Option<String>,
    pf: Option<u32>,
    kind: CapabilityKind,
    static_state: StaticState,
    evidence: EvidenceState,
    runtime: Option<RuntimeState>,
    documented_status: Option<Status>,
    stage: Option<&'static str>,
    engine_bridge: Option<bool>,
    automatic: Option<bool>,
}

impl Row {
    /// Wired labels are curated, so a wired row always overwrites placeholders.
    fn fill_label(&mut self, label_en: &str, label_zh: &str) {
        if self.label_en.as_deref().map_or(true, str::is_empty) {
            self.label_en = Some(label_en.to_owned());
        }
        if self.label_zh.as_deref().map_or(true, str::is_empty) {
            self.label_zh = Some(label_zh.to_owned());
        }
    }

    fn fill_labels_if_empty(&mut self, label_en: &str, label_zh: &str) {
        if self.label_en.as_deref().map_or(true, str::is_empty) {
            self.label_en = Some(label_en.to_owned());
        }
        if self.label_zh.as_deref().map_or(true, str::is_empty) && !label_zh.is_empty() {
            self.label_zh = Some(label_zh.to_owned());
        }
    }

    fn merge_evidence(&mut self, level: Option<EvidenceLevel>) {
        let Some(level) = level else { return };
        let replace = self
            .evidence
            .level
            .map_or(true, |current| level.index() > current.index());
        if replace {
            self.evidence = EvidenceState::documented(level);
        }
    }

    fn into_view(self, fallback_id: String) -> EffectCapabilityView {
        EffectCapabilityView {
            id: self.id.unwrap_or(fallback_id),
            label_en: self.label_en.unwrap_or_default(),
            label_zh: self.label_zh.unwrap_or_default(),
            pf: self.pf,
            kind: self.kind,
            static_state: self.static_state,
            evidence: self.evidence,
            runtime: self
                .runtime
                .unwrap_or_else(|| RuntimeState::unavailable("engine did not report this effect")),
            documented_status: self.documented_status,
            stage: self.stage,
            engine_bridge: self.engine_bridge,
            automatic: self.automatic,
        }
    }
}

fn highest_level(effect: &evidence::Effect) -> Option<EvidenceLevel> {
    EvidenceLevel::ALL
        .iter()
        .copied()
        .rev()
        .find(|level| effect.has(level.index()))
}

fn registry_text(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    (!trimmed.is_empty() && trimmed != evidence::UNVERIFIED).then_some(trimmed)
}

fn evidence_labels(effect: &evidence::Effect) -> (Option<String>, Option<String>) {
    let key = effect.catalog_key();
    let (label_en, label_zh) = contract::label_for_key(key);
    let label_en = label_en
        .map(str::to_owned)
        .or_else(|| registry_text(key).map(contract::humanized_key))
        .or_else(|| registry_text(effect.label).map(str::to_owned));
    let label_zh = label_zh
        .map(str::to_owned)
        .or_else(|| registry_text(effect.label).map(str::to_owned));
    (label_en, label_zh)
}

fn static_from_status(status: Status) -> StaticState {
    match status {
        Status::Ready | Status::Partial(..) => StaticState::Compiled,
        Status::Pending(..) | Status::Missing(..) => StaticState::Absent,
    }
}

/// Runtime for a documented non-portrait feature.
///
/// Only AI masks and tether have an `EngineCapabilities` probe. Everything
/// else is documented but not probed, so it must report `unavailable` with a
/// reason rather than invent a runtime answer.
fn runtime_for_status(id: &str, status: Status, caps: &EngineCapabilities) -> RuntimeState {
    match id {
        "status:ai_mask" => probed(!caps.ai_mask_kinds.is_empty(), "engine reports no AI mask kinds"),
        "status:tether" => probed(caps.tether, "engine reports no tether backend"),
        _ => match status {
            Status::Pending(..) | Status::Missing(..) => {
                RuntimeState::unavailable("documented as not implemented or not connected")
            }
            Status::Ready | Status::Partial(..) => {
                RuntimeState::unavailable("engine exposes no runtime probe for this feature")
            }
        },
    }
}

fn probed(available: bool, reason: &'static str) -> RuntimeState {
    if available {
        RuntimeState::available()
    } else {
        RuntimeState::unavailable(reason)
    }
}

fn engine_feature_rows(caps: &EngineCapabilities) -> Vec<Row> {
    vec![
        engine_feature("engine:eye_report", "Eye report", "闭眼检测", probed(caps.eye_report, "engine reports no eye report")),
        engine_feature("engine:auto_rotate", "Auto rotate", "自动旋转", probed(caps.auto_rotate, "engine reports no auto rotate")),
        engine_feature("engine:match_look", "Match look", "匹配风格", probed(caps.match_look, "engine reports no match look")),
        engine_feature(
            "engine:blemish_gainmap",
            "Blemish gain-map",
            "去瑕疵增益图",
            probed(caps.blemish_gainmap, "engine reports no blemish gain-map"),
        ),
        engine_feature(
            "engine:blemish_gainmap_model",
            "Blemish gain-map model",
            "去瑕疵增益图模型",
            probed(caps.blemish_gainmap_model, "engine reports the blemish gain-map model missing"),
        ),
        engine_feature(
            "engine:export_formats",
            "Export formats",
            "导出格式",
            probed(!caps.export_formats.is_empty(), "engine reports no export formats"),
        ),
    ]
}

fn engine_feature(id: &'static str, label_en: &'static str, label_zh: &'static str, runtime: RuntimeState) -> Row {
    Row {
        id: Some(id.to_owned()),
        label_en: Some(label_en.to_owned()),
        label_zh: Some(label_zh.to_owned()),
        kind: CapabilityKind::Engine,
        static_state: StaticState::Compiled,
        evidence: EvidenceState::NONE,
        runtime: Some(runtime),
        ..Row::default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::path::PathBuf;

    use aurora_engine::{
        AdjustmentSet, AiMaskKind, AutoResult, AutoTarget, Collection, ExportJob, ImageId, ImageInfo,
        JobStatus, PreviewFrame, PreviewRequest, RgbaImage, StyleInfo,
    };

    use super::*;

    struct FakeEngine {
        caps: EngineCapabilities,
        portrait: Vec<PortraitCapability>,
    }

    impl ImageEngine for FakeEngine {
        fn name(&self) -> String {
            "capability fake".into()
        }

        fn capabilities(&self) -> EngineCapabilities {
            self.caps.clone()
        }

        fn portrait_capabilities(&self) -> Vec<PortraitCapability> {
            self.portrait.clone()
        }

        fn images(&self) -> Vec<ImageInfo> {
            Vec::new()
        }

        fn import(&self, _paths: &[PathBuf]) -> anyhow::Result<Vec<ImageInfo>> {
            Ok(Vec::new())
        }

        fn collections(&self) -> Vec<Collection> {
            Vec::new()
        }

        fn update_info(&self, _info: &ImageInfo) -> anyhow::Result<()> {
            Ok(())
        }

        fn load_adjustments(&self, _image: ImageId) -> AdjustmentSet {
            AdjustmentSet::default()
        }

        fn save_adjustments(&self, _image: ImageId, _adjustments: &AdjustmentSet) -> anyhow::Result<()> {
            Ok(())
        }

        fn new_variant(&self, _image: ImageId, _adjustments: &AdjustmentSet) -> anyhow::Result<ImageInfo> {
            unimplemented!("not needed by the capability view")
        }

        fn thumbnail(&self, _image: ImageId, _adjustments: &AdjustmentSet, _max_size: u32) -> anyhow::Result<RgbaImage> {
            unimplemented!("not needed by the capability view")
        }

        fn render_preview(&self, _request: &PreviewRequest) -> anyhow::Result<PreviewFrame> {
            unimplemented!("not needed by the capability view")
        }

        fn sample_color(&self, _image: ImageId, _adjustments: &AdjustmentSet, _point: [f32; 2]) -> anyhow::Result<[f32; 3]> {
            unimplemented!("not needed by the capability view")
        }

        fn auto_adjust(&self, _image: ImageId, _current: &AdjustmentSet, _target: AutoTarget) -> anyhow::Result<AutoResult> {
            unimplemented!("not needed by the capability view")
        }

        fn pick_white_balance(&self, _image: ImageId, _point: [f32; 2]) -> anyhow::Result<(f32, f32)> {
            unimplemented!("not needed by the capability view")
        }

        fn compute_ai_mask(&self, _image: ImageId, _kind: AiMaskKind) -> anyhow::Result<String> {
            unimplemented!("not needed by the capability view")
        }

        fn icc_profiles(&self) -> Vec<String> {
            Vec::new()
        }

        fn lens_profiles(&self) -> Vec<String> {
            Vec::new()
        }

        fn styles(&self) -> Vec<StyleInfo> {
            Vec::new()
        }

        fn start_export(&self, _job: ExportJob) -> anyhow::Result<()> {
            Ok(())
        }

        fn job_status(&self, _job_id: u64) -> JobStatus {
            JobStatus::Failed { message: "fake".into() }
        }
    }

    fn portrait_capability(pf: u32) -> PortraitCapability {
        PortraitCapability {
            pf,
            key: format!("pf{pf}"),
            label: format!("PF{pf}"),
            stage: "cpu_stage",
            required_inputs: vec!["frame".into(), "mask".into()],
            engine_bridge: true,
            automatic: false,
        }
    }

    const WIRED_SYNTHETIC: &[WiredEffect] = &[
        WiredEffect { id: "pc:8068", label_en: "Skin gloss", label_zh: "皮肤光泽", pf: Some(8068) },
        WiredEffect { id: "pc:90005", label_en: "Whitening", label_zh: "美白", pf: Some(90005) },
        // The registry has no PF109/PF110 row; wired still knows them.
        WiredEffect { id: "pc:109", label_en: "Teeth brightness", label_zh: "牙齿亮白", pf: Some(109) },
        WiredEffect { id: "pc:110", label_en: "Teeth de-yellow", label_zh: "牙齿去黄", pf: Some(110) },
    ];

    #[test]
    fn builder_merges_engine_wired_and_documented_sources() {
        let view = capability_snapshot_for_tests(EngineCapabilities::default(), Vec::new(), &[]);
        assert_eq!(view.portrait_rows().count(), 147, "evidence registry rows");
        assert_eq!(view.documented_rows().count(), DOCUMENTED_STATUSES.len());
        assert_eq!(view.engine_rows().count(), 6);
        assert_eq!(view.len(), 147 + DOCUMENTED_STATUSES.len() + 6);

        let view = capability_snapshot_for_tests(
            EngineCapabilities::default(),
            vec![portrait_capability(8068)],
            WIRED_SYNTHETIC,
        );
        // 147 registry rows plus PF109/PF110, which only the wired list knows.
        assert_eq!(view.portrait_rows().count(), 149);
        assert!(view.get_by_pf(8068).expect("wired PF8068").runtime.available);
        let teeth = view.get_by_pf(109).expect("wired PF109");
        assert_eq!(teeth.static_state, StaticState::Compiled);
        assert_eq!(teeth.evidence.level, None);
        assert!(!teeth.runtime.available, "no engine bridge for PF109");
    }

    #[test]
    fn portrait_rows_are_in_a_deterministic_order() {
        let view = capability_snapshot_for_tests(
            EngineCapabilities::default(),
            vec![portrait_capability(8068)],
            WIRED_SYNTHETIC,
        );
        let keys: Vec<(u32, &str)> = view
            .rows()
            .iter()
            .map(|row| (row.pf.unwrap_or(u32::MAX), row.id.as_str()))
            .collect();
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted, "rows sort by PF, then by id");
        assert_eq!(keys.first(), Some(&(0, "pf:0")));
        assert_eq!(view.get_by_pf(8068).expect("PF8068").id, "pc:8068");
        assert_eq!(view.get_by_pf(8069).expect("PF8069").id, "pf:8069");
    }

    #[test]
    fn dimensions_disagree_for_compiled_evidenced_but_unavailable_effects() {
        let engine = FakeEngine {
            caps: EngineCapabilities::default(),
            portrait: vec![portrait_capability(8068)],
        };
        let view = CapabilityView::from_engine(&engine, WIRED_SYNTHETIC);

        let gloss = view.get_by_pf(8068).expect("PF8068");
        assert_eq!(gloss.static_state, StaticState::Compiled);
        assert!(gloss.evidence.is_documented());
        assert!(gloss.runtime.available);

        // Wired and documented at ev5, but this engine does not expose it:
        // compiled != available now.
        let whitening = view.get_by_pf(90005).expect("PF90005");
        assert_eq!(whitening.static_state, StaticState::Compiled);
        assert_eq!(whitening.evidence.level, Some(EvidenceLevel::MainPreview));
        assert!(!whitening.runtime.available);
        assert_eq!(whitening.runtime.reason, Some("engine does not expose this effect"));

        // Documented at ev5, but neither wired nor engine-exposed.
        let radiance = view.get_by_pf(90004).expect("PF90004");
        assert_eq!(radiance.static_state, StaticState::Absent);
        assert_eq!(radiance.evidence.level, Some(EvidenceLevel::MainPreview));
        assert!(!radiance.runtime.available);
        assert_eq!(radiance.runtime.reason, Some("engine did not report this effect"));
    }

    #[test]
    fn no_duplicate_ids_or_pfs() {
        let engine = FakeEngine {
            caps: EngineCapabilities {
                ai_mask_kinds: vec![AiMaskKind::Subject],
                tether: true,
                ..EngineCapabilities::default()
            },
            portrait: vec![portrait_capability(8068), portrait_capability(90005)],
        };
        let view = CapabilityView::from_engine(&engine, WIRED_SYNTHETIC);

        let ids: BTreeSet<&str> = view.rows().iter().map(|row| row.id.as_str()).collect();
        assert_eq!(ids.len(), view.len(), "ids are unique");
        let pfs: Vec<u32> = view.rows().iter().filter_map(|row| row.pf).collect();
        let unique: BTreeSet<u32> = pfs.iter().copied().collect();
        assert_eq!(unique.len(), pfs.len(), "PFs are unique");
    }

    #[test]
    fn engine_capabilities_drive_runtime_without_claiming_evidence() {
        let engine = FakeEngine {
            caps: EngineCapabilities {
                ai_mask_kinds: vec![AiMaskKind::Subject],
                eye_report: true,
                tether: true,
                ..EngineCapabilities::default()
            },
            portrait: vec![portrait_capability(8068)],
        };
        let view = CapabilityView::from_engine(&engine, &[]);

        assert!(view.get_by_id("status:ai_mask").expect("AI mask row").runtime.available);
        assert!(view.get_by_id("status:tether").expect("tether row").runtime.available);
        let eye = view.get_by_id("engine:eye_report").expect("eye report row");
        assert!(eye.runtime.available);
        assert_eq!(eye.evidence, EvidenceState::NONE, "runtime must not invent evidence");
        assert!(!view.get_by_id("engine:auto_rotate").expect("auto rotate row").runtime.available);
        assert!(!view.get_by_id("engine:export_formats").expect("export row").runtime.available);
    }

    #[test]
    fn status_constants_keep_their_documented_dimension() {
        let view = capability_snapshot_for_tests(EngineCapabilities::default(), Vec::new(), &[]);

        let pending = view.get_by_id("status:ai_denoise").expect("AI denoise row");
        assert_eq!(pending.kind, CapabilityKind::Documented);
        assert_eq!(pending.static_state, StaticState::Absent, "Pending is not compiled");
        assert!(pending.documented_status.is_some());
        assert!(!pending.runtime.available);

        let partial = view.get_by_id("status:spots").expect("spots row");
        assert_eq!(partial.static_state, StaticState::Compiled);
        assert!(partial.runtime.reason.is_some(), "no runtime probe means unverified, not broken");
    }

    #[test]
    fn evidence_levels_match_the_app_registry() {
        assert_eq!(EvidenceLevel::ALL.len(), evidence::LEVELS.len());
        for (index, level) in EvidenceLevel::ALL.iter().enumerate() {
            assert_eq!(level.name(), evidence::LEVELS[index]);
            assert_eq!(level.index(), index);
        }
        assert_eq!(EvidenceLevel::MainPreview.index(), evidence::MAIN_PREVIEW);
        assert_eq!(EvidenceLevel::VendorParity.index(), evidence::VENDOR_PARITY);
        assert_eq!(EvidenceLevel::RealPhotoAccept.index(), evidence::REAL_PHOTO);
    }

    #[test]
    fn wired_tuple_rows_convert_by_key() {
        let rows: Vec<WiredEffect> = [
            ("pc:8068", "Skin gloss", "皮肤光泽", 0.5_f32),
            ("pc:300096", "Missing", "缺失", 0.0_f32),
        ]
        .iter()
        .map(WiredEffect::from)
        .collect();
        assert_eq!(rows[0].pf, Some(8068));
        assert_eq!(rows[0].id, "pc:8068");
        assert_eq!(rows[1].pf, Some(300096));
    }
}
