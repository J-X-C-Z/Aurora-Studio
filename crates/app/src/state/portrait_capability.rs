//! 人像能力快照：按当前图片异步取、按资料版本缓存、只认 `Available`。
//!
//! 模块 A 的三条硬规则：
//!
//! 1. 绘制路径绝不同步调用引擎。`tools::portrait` 只读本模块的缓存；缓存缺失时
//!    状态是 [`CapabilityStatus::Pending`]，界面按「未就绪」呈现，不做任何推断。
//! 2. 只有 [`RuntimeAvailability::Available`] 才是可交互。登记、编译、
//!    `RuntimeProbeContext::fully_ready()`、`has_live_portrait_effect()` 一律不能
//!    当成可用（`has_live_portrait_effect` 仍然只用于「要不要渲染人像阶段」）。
//! 3. 缓存以（图片, 资料版本 + 主机配置指纹）为键。换图、换引擎、能力表版本变化
//!    都让旧结论失效；被后续请求取代的结果直接丢弃，晚到的结果不会覆盖新的。
//!
//! 引擎侧接口是 `ImageEngine::portrait_capability_snapshot(&self, ImageId)`，
//! 返回 `aurora_engine::portrait_registry::PortraitCapabilitySnapshot`。本文件是
//! 应用侧唯一的调用点。

use std::collections::HashMap;
use std::sync::Arc;

use aurora_engine::ImageId;
use aurora_engine::portrait_registry::{
    EffectCapabilityEntry, EffectCategory, InputDependency, ParamUnit, PortraitCapabilitySnapshot,
    RuntimeAvailability, WIRED_EFFECT_DESCRIPTORS, descriptor_by_pf, portrait_input_key,
};
use gpui_kit::{AppContext as _, Context};

use super::AppState;
use crate::contract::{self, Control};
use crate::i18n::{Locale, t};

// ---------------------------------------------------------------------------
// 单位：唯一的百分比 / 归一化判定点
// ---------------------------------------------------------------------------

/// 脸型网格组：`aurora_engine::portrait_stage::face_mesh_group` 的 7 组双控
/// （14 个 PF）在 `AdjustmentSet::extra` 里直接存 -100..=100，其余脸型、眼眉、
/// 发际线、高颅顶、微笑行存归一化值（-1..=1 / 0..=1）。
///
/// 能力描述符已经带了这个区别（`ParamRange::signed_percent` 对 14 个网格组，
/// `normalized_range(-1.0, 1.0, …)` 对其余），所以滑杆单位直接跟着
/// `descriptor.range` 走，本表只在测试里钉住这条不变量。
#[cfg(test)]
const FACE_MESH_PERCENT: [u32; 14] = [
    30092, 30093, 242, 243, 8023, 8024, 8027, 8028, 8029, 8030, 30080, 30081, 30082, 30083,
];

/// 界面值 ↔ `AdjustmentSet::extra` 存储值的换算。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueScale {
    /// 存储是归一化值（-1..=1、0..=1、0..=0.5…），界面显示对应的 -100..=100。
    Percent01,
    /// 界面值就是存储值（-100..=100 百分比、或 0..=4 的整数档位）。
    Identity,
}

/// 一个效果的滑杆范围、默认值与单位，全部以界面值为准。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SliderSpec {
    pub min: f32,
    pub max: f32,
    pub default: f32,
    /// `extra` 里的中性默认值；等于它时写入会删除该键。
    pub stored_default: f32,
    pub scale: ValueScale,
    pub unit: &'static str,
    /// 整数档位（PF30210 的 0..=4）用 1.0，连续滑杆为 `None`。
    pub step: Option<f32>,
    pub decimals: usize,
}

impl SliderSpec {
    pub fn to_ui(&self, stored: f32) -> f32 {
        match self.scale {
            ValueScale::Percent01 => stored * 100.0,
            ValueScale::Identity => stored,
        }
    }

    pub fn to_stored(&self, ui: f32) -> f32 {
        let ui = match self.step {
            Some(step) if step > 0.0 => (ui / step).round() * step,
            _ => ui,
        };
        match self.scale {
            ValueScale::Percent01 => ui / 100.0,
            ValueScale::Identity => ui,
        }
    }
}

/// 一份存储值该怎么显示。
///
/// PF30210（全脸丰盈）是 0..=4 的整数档位，但旧 sidecar 可能存了归一化值：0.9 在
/// 引擎输入与结算里都按第 4 档读，界面也必须显示 4，而不是 `Identity` 直通出来的
/// 0.9。反过来，字面量 1 是第 1 档，不是归一化最大值——这两条判定只有引擎的
/// [`aurora_engine::full_face_descriptor_from_store`] 一处做，应用侧不复制。
///
/// 写入仍然走 [`SliderSpec::to_stored`]（`step = 1.0`），新值一定是整数。
pub fn ui_value(pf: u32, spec: &SliderSpec, stored: f32) -> f32 {
    if pf == 30210 {
        if let Ok(level) = aurora_engine::full_face_descriptor_from_store(stored) {
            return level;
        }
    }
    spec.to_ui(stored)
}

fn percent_spec(min: f32, max: f32, default: f32) -> SliderSpec {
    SliderSpec {
        min,
        max,
        default,
        stored_default: default / 100.0,
        scale: ValueScale::Percent01,
        unit: "%",
        step: None,
        decimals: 0,
    }
}

/// 一个 PF 的滑杆规格。单位、范围与默认值全部来自能力描述符的
/// `ParamRange`，应用侧不再复制任何键表；没有描述符时才退回参数契约。
pub fn slider_spec(pf: u32) -> Option<SliderSpec> {
    if let Some(descriptor) = descriptor_by_pf(pf) {
        let range = descriptor.range;
        return Some(match range.unit {
            // 归一化存储：界面显示 -100..=100（或 0..=100 / 0..=50）。
            ParamUnit::Normalized01 => {
                percent_spec(range.min * 100.0, range.max * 100.0, range.default * 100.0)
            }
            // 百分比直接存储：脸型网格组与那 7 条丰盈行。
            ParamUnit::SignedPercent => SliderSpec {
                min: range.min,
                max: range.max,
                default: range.default,
                stored_default: range.default,
                scale: ValueScale::Identity,
                unit: "%",
                step: None,
                decimals: 0,
            },
            // 整数档位（PF30210 的 0..=4）。
            ParamUnit::Scalar => SliderSpec {
                min: range.min,
                max: range.max,
                default: range.default,
                stored_default: range.default,
                scale: ValueScale::Identity,
                unit: "",
                step: Some(1.0),
                decimals: 0,
            },
        });
    }
    // 没有能力描述符、但有参数契约控件的效果：契约把 0..=1 存成 0..=100 的
    // Amount 控件。
    let param = contract::PORTRAIT
        .iter()
        .find(|param| param.id == i64::from(pf))?;
    match &param.control {
        Control::Amount => Some(percent_spec(0.0, 100.0, param.default * 100.0)),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// 界面分类与分组
// ---------------------------------------------------------------------------

/// 界面分类直接取描述符。`PF90032` 是眼妆（`EyeMaskType=10` 的 RGB 遮罩 +
/// `SymEyeImageMakeup` 素材），描述符把它登记在 `Eyes`，所以它不再出现在唇妆里。
pub fn category_of(pf: u32) -> Option<EffectCategory> {
    descriptor_by_pf(pf).map(|descriptor| descriptor.category)
}

/// 某分类下的全部已登记 PF，顺序与能力描述符表一致。
pub fn pfs_in(category: EffectCategory) -> Vec<u32> {
    WIRED_EFFECT_DESCRIPTORS
        .iter()
        .filter(|descriptor| category_of(descriptor.id.pf) == Some(category))
        .map(|descriptor| descriptor.id.pf)
        .collect()
}

/// 描述符的分节名（`Eye makeup` 只出现在 PF90032 的界面上）。
pub fn section_of(pf: u32) -> Option<&'static str> {
    descriptor_by_pf(pf).map(|descriptor| descriptor.section)
}

// ---------------------------------------------------------------------------
// 可用性状态
// ---------------------------------------------------------------------------

/// 一个 PF 在当前图片上的能力状态。
#[derive(Clone, Copy, Debug)]
pub enum CapabilityStatus<'a> {
    /// 还没有当前图片。
    NoImage,
    /// 快照还没到（保守初态：不可交互）。
    Pending,
    /// 快照到了，但描述符表里没有这个 PF：不推断任何支持。
    Undescribed,
    /// 快照里的登记项；只有 `Available` 可交互。
    Known(&'a EffectCapabilityEntry),
}

impl CapabilityStatus<'_> {
    pub fn is_available(self) -> bool {
        matches!(self, Self::Known(entry) if entry.availability.is_available())
    }

    /// A different texture can repair a missing material or UV overlap. This
    /// only enables a detached candidate probe, never an edit to this document.
    pub fn can_probe_selection(self) -> bool {
        matches!(self, Self::Known(row) if matches!(row.availability,
            RuntimeAvailability::Available | RuntimeAvailability::PendingSameFrame { .. }
            | RuntimeAvailability::MissingAsset { .. } | RuntimeAvailability::MissingInput { .. }))
    }

    /// 一行状态：不可交互时写确切原因，可用时写 `Available`。
    pub fn line(self) -> String {
        match self {
            Self::NoImage => t("No image selected", "尚未选择图片").to_string(),
            Self::Pending => t("Reading portrait capability…", "正在读取人像能力…").to_string(),
            Self::Undescribed => t(
                "No capability descriptor in this build — support not inferred",
                "本版本没有该效果的能力描述符，不推断支持",
            )
            .to_string(),
            Self::Known(entry) => availability_line(&entry.availability),
        }
    }
}

fn dependency_label(dependency: InputDependency) -> String {
    if locale_of() != Locale::ZhCn {
        return dependency.as_str().to_string();
    }
    match dependency {
        InputDependency::None => "无".to_string(),
        InputDependency::Landmarks194 => "194 关键点".to_string(),
        InputDependency::FaceParsingV2 => "人脸解析模型（fbe.onnx）".to_string(),
        InputDependency::SkinMask => "皮肤掩码".to_string(),
        InputDependency::HairMask => "头发掩码".to_string(),
        InputDependency::TeethMask => "牙齿掩码".to_string(),
        InputDependency::EyeMask => "眼部遮罩（EyeMask）".to_string(),
        InputDependency::MakeupBundle => "妆容素材包".to_string(),
        InputDependency::SmileWarpAsset => "微笑形变资产".to_string(),
        InputDependency::BodyPose => "人体姿态关键点模型".to_string(),
        InputDependency::HumanParsingMask => "人体解析模型（body parsing）".to_string(),
        InputDependency::StubbleStrayHairModel => "胡茬 / 碎发 / 白发修复模型".to_string(),
        InputDependency::EyeBagModel => "眼袋 / 黑眼圈残差模型".to_string(),
        InputDependency::EyeWhiteModel => "眼白净化模型".to_string(),
        InputDependency::BreastRepairModel => "胸部修复模型".to_string(),
        InputDependency::TeethRepairModel => "牙齿修复模型".to_string(),
        InputDependency::GlassesGlareModel => "眼镜反光 / 镜片倒影模型".to_string(),
        InputDependency::DoubleChinModel => "双下巴位移场模型".to_string(),
        InputDependency::SkinRepairGenericModel => {
            "通用皮肤修复模型（纹理 / 血管 / 妊娠纹 / 汗）".to_string()
        }
        InputDependency::MoistureCurveAsset => "水润柔化曲线表（PF90002 / PF90010）".to_string(),
        InputDependency::FaceStereoAsset => {
            "FaceStereo 立体阴影 / 高光材质（PF8 / PF9）".to_string()
        }
        InputDependency::FacePlumpResidual => "面部丰盈模型与本图残差".to_string(),
        InputDependency::FacePartTable => "当前脸型所需的网格系数表".to_string(),
        InputDependency::SkinToneLut => "当前肤色效果所需的颜色查找表".to_string(),
        InputDependency::LipsLutAsset => "唇部 LUT 材质（PF90021）".to_string(),
        // 引擎将来新增的依赖先显示 API 名：缺译文是可接受的降级，整块面板编译不过不是。
        #[allow(unreachable_patterns)]
        other => other.as_str().to_string(),
    }
}

fn dependencies_label(dependencies: &[InputDependency]) -> String {
    dependencies
        .iter()
        .map(|dep| dependency_label(*dep))
        .collect::<Vec<_>>()
        .join(" · ")
}

/// 依赖清单的本地化文字：原因行与悬停说明共用同一份措辞。
pub fn inputs_line(dependencies: &[InputDependency]) -> String {
    dependencies_label(dependencies)
}

/// 运行期不可用的确切原因：缺模型 / 缺资产 / 缺同帧输入 / 无后端 / 仅保存。
pub fn availability_line(availability: &RuntimeAvailability) -> String {
    match availability {
        RuntimeAvailability::Available => t("Available", "可用").to_string(),
        RuntimeAvailability::MissingModel { required } => {
            format!(
                "{} {}",
                t("Missing model", "缺模型"),
                dependencies_label(required)
            )
        }
        RuntimeAvailability::MissingAsset { required } => {
            format!(
                "{} {}",
                t("Missing asset", "缺资产"),
                dependencies_label(required)
            )
        }
        RuntimeAvailability::MissingInput { required } => {
            format!(
                "{} {}",
                t("Missing same-frame input", "缺同帧输入"),
                dependencies_label(required)
            )
        }
        // Cheap discovery passed but this photo's masks/materials have not been
        // produced yet. Deliberately not green: it stays disabled and names the
        // action that turns it into a measured answer.
        RuntimeAvailability::PendingSameFrame { required } => {
            let mut line = t(
                "Same-frame inputs not verified for this photo — press refresh",
                "待验证本图输入，点刷新",
            )
            .to_string();
            if !required.is_empty() {
                line.push_str(" · ");
                line.push_str(&dependencies_label(required));
            }
            line
        }
        RuntimeAvailability::BackendUnavailable { reason } => {
            format!(
                "{} · {reason}",
                t("No usable compute backend", "无可用计算后端")
            )
        }
        RuntimeAvailability::StoredOnly => t(
            "Stored only — not wired to a stage",
            "仅保存 · 未接入渲染阶段",
        )
        .to_string(),
    }
}

fn locale_of() -> Locale {
    crate::i18n::locale()
}

// ---------------------------------------------------------------------------
// 缓存
// ---------------------------------------------------------------------------

/// 资料版本（能力表 TSV）与主机配置（引擎名）的指纹。任一变化即整体失效。
fn fingerprint(registry_tsv: &str, engine_name: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let bytes = registry_tsv
        .as_bytes()
        .iter()
        .chain(engine_name.as_bytes())
        .chain(b"\0portrait-capability-v1".iter());
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

struct CachedSnapshot {
    snapshot: Arc<PortraitCapabilitySnapshot>,
    /// 请求时的预览帧代数：更新的帧到来后非 Available 的结论可能已经过时。
    frame: u64,
    /// 请求时的调整指纹（[`extra_key`]）：换了款式/材质，旧结论不再代表这份文档。
    adj_key: u64,
}

impl CachedSnapshot {
    #[cfg(test)]
    fn for_test(snapshot: PortraitCapabilitySnapshot, frame: u64) -> Self {
        Self {
            snapshot: Arc::new(snapshot),
            frame,
            adj_key: 0,
        }
    }
}

struct PendingRequest {
    image: ImageId,
    serial: u64,
    /// 发起请求时的预览帧代数：结果入库后用它判断结论是否已被新帧超越。
    frame: u64,
    /// 发起请求时的调整指纹；结果回来时不一致就丢弃。
    adj_key: u64,
}

/// 按图片缓存的能力快照，以及一个在途请求。
#[derive(Default)]
pub struct PortraitCapabilityCache {
    revision: u64,
    entries: HashMap<ImageId, CachedSnapshot>,
    pending: Option<PendingRequest>,
    serial: u64,
    /// 候选预检的序号：连点两次只让最后一次的结果生效。
    precheck: u64,
}

impl PortraitCapabilityCache {
    pub fn new(registry_tsv: &str, engine_name: &str) -> Self {
        Self {
            revision: fingerprint(registry_tsv, engine_name),
            ..Self::default()
        }
    }

    /// 这张图（这份调整）还没有结论，也没有同一张图的在途请求。
    pub fn needs_request(&self, image: ImageId, adj_key: u64) -> bool {
        if self
            .entries
            .get(&image)
            .is_some_and(|cached| cached.adj_key == adj_key)
        {
            return false;
        }
        !self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.image == image && pending.adj_key == adj_key)
    }

    /// 登记一次请求；返回的序号用来丢弃晚到的旧结果。
    fn begin(&mut self, image: ImageId, frame: u64, adj_key: u64) -> u64 {
        self.serial += 1;
        self.pending = Some(PendingRequest {
            image,
            serial: self.serial,
            frame,
            adj_key,
        });
        self.serial
    }

    /// 接受一份结果。只有最新请求、同一张图、同一份调整的结果会入库；
    /// 返回是否安装成功。
    fn accept(
        &mut self,
        image: ImageId,
        serial: u64,
        adj_key: u64,
        snapshot: PortraitCapabilitySnapshot,
    ) -> bool {
        let frame = match self.pending.as_ref() {
            Some(pending)
                if self.serial == serial
                    && pending.serial == serial
                    && pending.image == image
                    && pending.adj_key == adj_key =>
            {
                pending.frame
            }
            _ => return false,
        };
        self.pending = None;
        self.entries.insert(
            image,
            CachedSnapshot {
                snapshot: Arc::new(snapshot),
                frame,
                adj_key,
            },
        );
        true
    }

    /// 丢掉一张图的结论（刷新按钮、或需要重探时）。
    pub fn invalidate(&mut self, image: ImageId) {
        self.entries.remove(&image);
        if self
            .pending
            .as_ref()
            .is_some_and(|pending| pending.image == image)
        {
            self.pending = None;
            self.serial += 1;
        }
    }

    /// 记录预览帧代数，**不再据此淘汰结论**。
    ///
    /// 以前「非全绿 + 新帧」就丢缓存，结果 149 行里只要有 `StoredOnly` 就永远
    /// 非全绿，每一代都清空、每代都重探。真正该让证据失效的只有：换图（按图
    /// 缓存天然隔离）、资料/引擎身份变化（[`Self::note_revision`]）、显式刷新
    /// （[`Self::invalidate`]），以及输入指纹变化（[`Self::status`] 比较）。
    pub fn note_frame(&mut self, image: ImageId, frame: u64) {
        if let Some(cached) = self.entries.get_mut(&image) {
            cached.frame = cached.frame.max(frame);
        }
    }

    /// 图片（这份调整）的快照；没有新鲜结论时为 `None`（绝不同步去引擎取）。
    pub fn snapshot(
        &self,
        image: Option<ImageId>,
        adj_key: u64,
    ) -> Option<Arc<PortraitCapabilitySnapshot>> {
        image
            .and_then(|image| self.entries.get(&image))
            .filter(|cached| cached.adj_key == adj_key)
            .map(|cached| Arc::clone(&cached.snapshot))
    }

    /// 一个 PF 在当前图片、当前调整上的状态。
    pub fn status(&self, image: Option<ImageId>, adj_key: u64, pf: u32) -> CapabilityStatus<'_> {
        let Some(image) = image else {
            return CapabilityStatus::NoImage;
        };
        let Some(cached) = self.entries.get(&image) else {
            return CapabilityStatus::Pending;
        };
        if cached.adj_key != adj_key {
            // 款式/材质换过：旧结论不描述这份文档，按「还没测」处理，不沿用。
            return CapabilityStatus::Pending;
        }
        match cached.snapshot.by_pf(pf) {
            Some(entry) => CapabilityStatus::Known(entry),
            None => CapabilityStatus::Undescribed,
        }
    }

    /// 当前图片（这份调整）快照里的（可用数, 总数）；还没取到时为 `None`。
    pub fn counts(&self, image: Option<ImageId>, adj_key: u64) -> Option<(usize, usize)> {
        let snapshot = self.snapshot(image, adj_key)?;
        Some((snapshot.available_effects, snapshot.total_effects))
    }

    /// 有没有当前图片（这份调整）的结论（用来区分「未就绪」和「不可用」）。
    pub fn has_snapshot(&self, image: Option<ImageId>, adj_key: u64) -> bool {
        image.is_some_and(|image| {
            self.entries
                .get(&image)
                .is_some_and(|cached| cached.adj_key == adj_key)
        })
    }

    /// 资料版本 / 引擎变化：整个缓存作废（能力表换了，旧结论没有意义）。
    pub fn note_revision(&mut self, registry_tsv: &str, engine_name: &str) {
        let revision = fingerprint(registry_tsv, engine_name);
        if revision != self.revision {
            self.revision = revision;
            self.precheck += 1;
            self.entries.clear();
            self.pending = None;
            self.serial += 1;
        }
    }

    /// 把一次候选预检的快照直接装回缓存。
    ///
    /// 预检已经证明这套候选的输入可得，若不装回去，紧接着的一次自动读取会把它
    /// 当成「未测」并退回 cheap 探针，整套妆容点一次就全灰。
    pub fn install(
        &mut self,
        image: ImageId,
        input_key: u64,
        snapshot: PortraitCapabilitySnapshot,
    ) {
        self.invalidate(image); // an older cheap response must not replace this detail verdict
        self.entries.insert(
            image,
            CachedSnapshot {
                snapshot: Arc::new(snapshot),
                frame: 0,
                adj_key: input_key,
            },
        );
    }

    /// 登记一次候选预检，返回用来判定「是不是最后一次」的序号。
    pub fn next_precheck(&mut self) -> u64 {
        self.precheck += 1;
        self.precheck
    }

    pub fn precheck_is_current(&self, serial: u64) -> bool {
        self.precheck == serial
    }
}

/// 影响可用性的调整指纹：`AdjustmentSet::extra` 里的款式/材质/开关选择。
///
/// 只哈希 `extra`（BTreeMap，顺序稳定），不碰照片、不碰像素；滑杆数值本身
/// 不改变「这一行有没有输入」，但款式与材质会，所以以它为准。
pub fn extra_key(extra: &std::collections::BTreeMap<String, f32>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for (key, value) in extra {
        mix(key.as_bytes());
        mix(&value.to_bits().to_le_bytes());
    }
    hash
}

/// 该 PF 在当前图片、当前调整上是否可交互。未就绪即 false（保守初态）。
pub fn row_available(state: &AppState, pf: u32) -> bool {
    state.capability_status(pf).is_available()
}

/// 这次取快照是自动触发，还是用户明确刷新。
///
/// 自动只跑一次（首次进入面板、换图、换调整），刷新按钮强制重取。两者走同一条
/// 串行化路径，晚到的旧结果一律丢弃；`detail` 由调用方交给引擎，允许刷新走
/// 更贵的同帧实测，而自动首读保持廉价。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CapabilityDetail {
    Auto,
    Explicit,
}

impl CapabilityDetail {
    pub(crate) fn is_explicit(self) -> bool {
        matches!(self, Self::Explicit)
    }
}

// ---------------------------------------------------------------------------
// AppState：异步取快照
// ---------------------------------------------------------------------------

impl AppState {
    /// 影响同帧输入**生产**的指纹：种类/款式/颜色/档位/合法性。引擎与渲染共用
    /// `portrait_input_key`，所以普通强度在合法区间内拖动不改变它——拖滑杆不会
    /// 让已实测的结论失效，也就不会出现「改一次强度就再也拖不动」。
    pub(crate) fn capability_key(&self) -> u64 {
        portrait_input_key(&self.adj().extra)
    }

    /// 「还是不是刚才那份文档」的指纹：点击与候选预检的防过期用，保留完整精度。
    pub(crate) fn document_key(&self) -> u64 {
        extra_key(&self.adj().extra)
    }

    /// 一行效果的当前状态；图片或调整换了就回到「未测」。
    pub(crate) fn capability_status(&self, pf: u32) -> CapabilityStatus<'_> {
        self.portrait_caps
            .status(self.library.current, self.capability_key(), pf)
    }

    /// 当前图片（当前调整）是否还需要一份快照。
    pub(crate) fn portrait_capability_needed(&self) -> bool {
        let key = self.capability_key();
        self.library.current.is_some_and(|image| {
            self.preview_state
                .preview
                .frame
                .as_ref()
                .is_some_and(|f| f.image_id == image)
                && self.portrait_caps.needs_request(image, key)
        })
    }

    /// 在后台取当前图片的能力快照。绘制路径只读缓存，绝不同步等引擎；
    /// 结果回来时才 `notify`，被后续请求取代、换了图或换了调整的结果直接丢弃。
    pub(crate) fn request_portrait_capability(
        &mut self,
        detail: CapabilityDetail,
        cx: &mut Context<Self>,
    ) {
        let Some(image) = self.library.current else {
            return;
        };
        let adj_key = self.capability_key();
        if !self.portrait_caps.needs_request(image, adj_key) {
            return;
        }
        let frame = self.preview_state.preview.generation;
        let serial = self.portrait_caps.begin(image, frame, adj_key);
        let engine = self.engine.clone();
        // 请求带走一份自己的调整快照：用户随后继续拖滑杆不会改变这次请求的含义。
        let adj = self.adj().clone();
        cx.spawn(async move |this, cx| {
            let snapshot = cx
                .background_spawn(async move {
                    // 自动首读与明确刷新走同一个入口，`detail` 只告诉引擎允不允许
                    // 更贵的同帧实测；调整以请求时的快照交出去，不受后续拖动影响。
                    engine.portrait_capability_for(image, &adj, detail.is_explicit())
                })
                .await;
            let _ = this.update(cx, move |this, cx| {
                if this.library.current == Some(image)
                    && this.capability_key() == adj_key
                    && this.portrait_caps.accept(image, serial, adj_key, snapshot)
                {
                    cx.notify();
                }
            });
        })
        .detach();
    }

    /// 刷新按钮：丢掉本图本调整的结论后立即重取（刚装上模型/资产、换了后端时用）。
    pub(crate) fn refresh_portrait_capability(&mut self, cx: &mut Context<Self>) {
        if let Some(image) = self.library.current {
            self.portrait_caps.invalidate(image);
        }
        self.request_portrait_capability(CapabilityDetail::Explicit, cx);
        cx.notify();
    }

    /// 预览新帧装上后调用：只记录代数，不淘汰任何已证结论。
    ///
    /// 同帧输入的产出由输入指纹与显式刷新负责失效；按代清空会让「某行 StoredOnly
    /// 导致非全绿」的图每帧重探，149 行永远换不来稳定结论。
    pub(crate) fn note_portrait_frame(&mut self, image: ImageId, frame: u64) {
        if self
            .portrait_caps
            .entries
            .get(&image)
            .is_some_and(|cached| cached.snapshot.available_effects == 0 && cached.frame != frame)
        {
            self.portrait_caps.invalidate(image);
        }
        self.portrait_caps.note_frame(image, frame);
    }

    /// 资料版本或引擎变了：整份缓存作废（能力表换了，旧结论没有意义）。
    pub(crate) fn note_portrait_revision(&mut self, registry_tsv: &str, engine_name: &str) {
        self.portrait_caps.note_revision(registry_tsv, engine_name);
    }

    /// 一次写很多键的动作（整妆/成套妆容）：先把**已经合成好的候选调整**送到同
    /// 一个 detail 快照做单次异步预检，只有引擎对这份候选仍然给出可用结论时才
    /// 真的写一次 undo。
    ///
    /// 失败不建 undo；回来时如果图片或调整已经变了（用户在拖动滑杆、或者换了
    /// 照片），结论直接丢弃——旧文档的预检结果绝不能覆盖正在编辑的这份。
    pub(crate) fn precheck_then_edit_extra(
        &mut self,
        label: &'static str,
        extra: std::collections::BTreeMap<String, f32>,
        mut required: Vec<u32>,
        cx: &mut Context<Self>,
    ) {
        let Some(image) = self.library.current else {
            return;
        };
        // 防过期用文档指纹（完整精度），装回缓存用输入指纹（只随款式/档位等变）。
        let document = self.document_key();
        required.sort_unstable();
        required.dedup();
        if required.is_empty() {
            return;
        }
        let input = portrait_input_key(&extra);
        let serial = self.portrait_caps.next_precheck();
        let mut candidate = self.adj().clone();
        candidate.extra = extra.clone();
        let engine = self.engine.clone();
        cx.spawn(async move |this, cx| {
            let snapshot = cx
                .background_spawn(
                    async move { engine.portrait_capability_for(image, &candidate, true) },
                )
                .await;
            // 每一行都必须在这份候选上可用；缺任何一行都不写。
            let error = required.iter().find_map(|&pf| match snapshot.by_pf(pf) {
                Some(row) if row.availability.is_available() => None,
                Some(row) => Some(format!("PF{pf}: {}", availability_line(&row.availability))),
                None => Some(format!("PF{pf}: {}", t("Not registered", "未登记"))),
            });
            let _ = this.update(cx, move |this, cx| {
                if !this.portrait_caps.precheck_is_current(serial)
                    || this.library.current != Some(image)
                    || this.document_key() != document
                {
                    return;
                }
                if let Some(reason) = error {
                    this.notify(reason, cx);
                    return;
                }
                // 预检效力留在缓存里：否则紧接着的自动读取会把刚证明的行退回未测，
                // 整套妆容点一次就全灰。
                this.portrait_caps.install(image, input, snapshot);
                this.edit(label, move |a| a.extra = extra, cx);
            });
        })
        .detach();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_engine::portrait_registry::{RuntimeProbeContext, capability_snapshot};
    use std::collections::{BTreeMap, BTreeSet};

    /// 测试用的调整指纹（与缓存里的 0 不同）。
    const KEY: u64 = 0x5eed;

    fn view(image: ImageId) -> HashMap<ImageId, CachedSnapshot> {
        HashMap::from([(
            image,
            CachedSnapshot::for_test(
                capability_snapshot(&RuntimeProbeContext::cpu_only_empty()),
                7,
            ),
        )])
    }

    fn cache_with(image: ImageId) -> PortraitCapabilityCache {
        let mut cache = PortraitCapabilityCache::new("kind\tpf", "engine");
        cache.entries = view(image);
        cache
    }

    /// 保守初态：没有快照时任何 PF 都不可交互。
    #[test]
    fn row_available_is_false_before_the_snapshot_arrives() {
        let cache = PortraitCapabilityCache::new("kind\tpf", "engine");
        assert!(!cache.status(Some(ImageId(1)), 0, 90005).is_available());
        assert!(matches!(
            cache.status(Some(ImageId(1)), 0, 90005),
            CapabilityStatus::Pending
        ));
        assert!(!cache.status(None, 0, 90005).is_available());
        assert!(matches!(
            cache.status(None, 0, 90005),
            CapabilityStatus::NoImage
        ));
        assert_eq!(cache.counts(Some(ImageId(1)), 0), None);
    }

    /// 只有该图新鲜快照里的 Available 才算可用，其它状态一律 false。
    #[test]
    fn only_an_available_entry_for_this_image_is_interactive() {
        // PF242（尖下巴）只依赖 194 关键点：这份探测上下文里模型和同帧输入都有。
        let mut ready = RuntimeProbeContext::cpu_only_empty();
        ready.available_models.insert(InputDependency::Landmarks194);
        ready.available_inputs.insert(InputDependency::Landmarks194);
        ready
            .available_assets
            .insert(InputDependency::FacePartTable);
        let cache = PortraitCapabilityCache {
            revision: 1,
            entries: HashMap::from([(
                ImageId(1),
                CachedSnapshot::for_test(capability_snapshot(&ready), 7),
            )]),
            pending: None,
            serial: 0,
            precheck: 0,
        };
        assert!(cache.status(Some(ImageId(1)), 0, 242).is_available());
        // PF90005 需要人脸解析模型：同一份快照里是 MissingModel。
        assert!(!cache.status(Some(ImageId(1)), 0, 90005).is_available());
        let line = cache.status(Some(ImageId(1)), 0, 90005).line();
        assert!(
            line.contains("Missing model") || line.contains("缺模型"),
            "{line}"
        );
        // 换到另一张图：没有结论，不能借用上一张的可用性。
        assert!(!cache.status(Some(ImageId(2)), 0, 242).is_available());
        assert!(matches!(
            cache.status(Some(ImageId(2)), 0, 242),
            CapabilityStatus::Pending
        ));
    }

    /// 换了款式/材质（调整指纹变了）之后，旧结论不得继续代表这份文档。
    #[test]
    fn a_changed_adjustment_makes_the_verdict_pending_instead_of_reused() {
        let mut ready = RuntimeProbeContext::cpu_only_empty();
        ready.available_models.insert(InputDependency::Landmarks194);
        ready.available_inputs.insert(InputDependency::Landmarks194);
        ready
            .available_assets
            .insert(InputDependency::FacePartTable);
        let mut cache = PortraitCapabilityCache {
            revision: 1,
            entries: HashMap::from([(
                ImageId(1),
                CachedSnapshot::for_test(capability_snapshot(&ready), 7),
            )]),
            pending: None,
            serial: 0,
            precheck: 0,
        };
        assert!(cache.status(Some(ImageId(1)), 0, 242).is_available());
        // 换调整：立刻回到未测，而不是沿用上一份调整的 Available。
        assert!(matches!(
            cache.status(Some(ImageId(1)), KEY, 242),
            CapabilityStatus::Pending
        ));
        assert_eq!(cache.counts(Some(ImageId(1)), KEY), None);
        assert!(
            cache.needs_request(ImageId(1), KEY),
            "a new adjustment needs its own request"
        );
        assert!(
            !cache.needs_request(ImageId(1), 0),
            "the adjustment already cached needs no request"
        );
        // 带新指纹的结果被接受之后才重新有结论。
        let serial = cache.begin(ImageId(1), 7, KEY);
        assert!(cache.accept(ImageId(1), serial, KEY, capability_snapshot(&ready)));
        assert!(cache.status(Some(ImageId(1)), KEY, 242).is_available());
    }

    /// 资料版本/引擎变了：整份缓存作废，旧结论不会跨版本复用。
    #[test]
    fn a_new_registry_revision_clears_the_cache() {
        let mut cache = cache_with(ImageId(1));
        assert!(cache.has_snapshot(Some(ImageId(1)), 0));
        cache.note_revision("kind\tpf", "engine");
        assert!(
            cache.has_snapshot(Some(ImageId(1)), 0),
            "the same revision must keep the cache"
        );
        cache.note_revision("kind\tpf", "another-engine");
        assert!(
            !cache.has_snapshot(Some(ImageId(1)), 0),
            "a different engine or registry must drop every verdict"
        );
    }

    /// 晚到的旧请求结果必须丢弃，不能覆盖新结论。
    #[test]
    fn superseded_results_are_discarded() {
        let mut cache = PortraitCapabilityCache::new("registry", "engine");
        let first = cache.begin(ImageId(1), 1, KEY);
        let second = cache.begin(ImageId(1), 2, KEY);
        assert!(!cache.accept(
            ImageId(1),
            first,
            KEY,
            capability_snapshot(&RuntimeProbeContext::cpu_only_empty())
        ));
        assert!(cache.accept(
            ImageId(1),
            second,
            KEY,
            capability_snapshot(&RuntimeProbeContext::cpu_only_empty())
        ));
        // 同一序号不能被重复安装。
        assert!(!cache.accept(
            ImageId(1),
            second,
            KEY,
            capability_snapshot(&RuntimeProbeContext::cpu_only_empty())
        ));
    }

    /// 结果回来时调整已经换了：丢弃，不能把旧调整的结论写进新调整的槽。
    #[test]
    fn a_result_for_a_superseded_adjustment_is_dropped() {
        let mut cache = PortraitCapabilityCache::new("registry", "engine");
        let stale = cache.begin(ImageId(1), 1, 0);
        assert!(!cache.accept(
            ImageId(1),
            stale,
            KEY,
            capability_snapshot(&RuntimeProbeContext::cpu_only_empty())
        ));
        assert_eq!(cache.counts(Some(ImageId(1)), KEY), None);
    }

    /// 合并场景：连续拖强度保持可交互；换款式要有新证据才写；换图/过期结果不写。
    #[test]
    fn one_scenario_covers_dragging_restyling_and_stale_callbacks() {
        let mut ready = RuntimeProbeContext::cpu_only_empty();
        ready.available_models.insert(InputDependency::Landmarks194);
        ready.available_inputs.insert(InputDependency::Landmarks194);
        ready
            .available_assets
            .insert(InputDependency::FacePartTable);

        let strength = |value: f32| BTreeMap::from([("pc:100".to_owned(), value)]);
        let base = strength(0.2);
        let key = portrait_input_key(&base);
        // 同一套输入：正常拖强度不改变输入指纹，已证结论继续可用。
        assert_eq!(
            portrait_input_key(&strength(0.55)),
            key,
            "dragging an ordinary strength is not a new input set"
        );
        let mut cache = PortraitCapabilityCache {
            revision: 1,
            entries: HashMap::new(),
            pending: None,
            serial: 0,
            precheck: 0,
        };
        cache.install(ImageId(1), key, capability_snapshot(&ready));
        assert!(
            cache.status(Some(ImageId(1)), key, 242).is_available(),
            "the row stays usable while the same inputs are proven"
        );

        // 换款式：输入指纹变化，必须有新证据才能写。
        let restyled =
            BTreeMap::from([("pc:100".to_owned(), 0.2), ("pc:100:style".to_owned(), 2.0)]);
        let restyle_key = portrait_input_key(&restyled);
        assert_ne!(
            restyle_key, key,
            "a different style is a different input set"
        );
        assert!(matches!(
            cache.status(Some(ImageId(1)), restyle_key, 242),
            CapabilityStatus::Pending
        ));
        assert!(cache.needs_request(ImageId(1), restyle_key));

        // 换图：不借用上一张的结论。
        assert!(matches!(
            cache.status(Some(ImageId(2)), restyle_key, 242),
            CapabilityStatus::Pending
        ));

        // 过期的候选结果（别的图）写不进来。
        let stale = cache.begin(ImageId(2), 0, restyle_key);
        assert!(!cache.accept(ImageId(1), stale, restyle_key, capability_snapshot(&ready)));
        assert_eq!(cache.counts(Some(ImageId(1)), restyle_key), None);

        // 预检效力：连点只认最后一次，装回后立刻重新可用（不会点一次就全灰）。
        let first = cache.next_precheck();
        let second = cache.next_precheck();
        assert!(!cache.precheck_is_current(first));
        assert!(cache.precheck_is_current(second));
        cache.install(ImageId(1), restyle_key, capability_snapshot(&ready));
        assert!(
            cache
                .status(Some(ImageId(1)), restyle_key, 242)
                .is_available()
        );
    }

    /// 14 个脸型网格组存百分比，其余脸型/眼眉行存归一化；单位也跟着变。
    #[test]
    fn fourteen_face_mesh_controls_stay_percent_and_the_rest_are_normalized() {
        assert_eq!(FACE_MESH_PERCENT.len(), 14);
        let face = pfs_in(EffectCategory::FaceGeometry);
        let percent: BTreeSet<u32> = face
            .iter()
            .copied()
            .filter(|pf| slider_spec(*pf).is_some_and(|spec| spec.scale == ValueScale::Identity))
            .collect();
        assert_eq!(
            percent,
            FACE_MESH_PERCENT.iter().copied().collect::<BTreeSet<_>>()
        );
        for pf in face.iter().filter(|pf| !FACE_MESH_PERCENT.contains(pf)) {
            let spec = slider_spec(*pf).expect("face nuance spec");
            assert_eq!(
                spec.scale,
                ValueScale::Percent01,
                "PF{pf} must store normalized"
            );
            assert_eq!(spec.unit, "%");
            assert!((spec.to_stored(100.0) - 1.0).abs() < 1e-6);
            assert!((spec.to_stored(-100.0) + 1.0).abs() < 1e-6);
        }
        let mesh = slider_spec(242).expect("PF242");
        assert_eq!((mesh.min, mesh.max), (-100.0, 100.0));
        assert!(
            (mesh.to_stored(-25.0) + 25.0).abs() < 1e-6,
            "face-mesh rows store percent directly"
        );
    }

    /// PF30210 是 0..=4 的离散档位，不是百分比。
    #[test]
    fn pf30210_is_discrete_between_zero_and_four() {
        let spec = slider_spec(30210).expect("PF30210");
        assert_eq!((spec.min, spec.max), (0.0, 4.0));
        assert_eq!(spec.step, Some(1.0));
        assert_eq!(spec.scale, ValueScale::Identity);
        assert_eq!(spec.unit, "");
    }

    /// 旧文档存的归一化 0.9 在引擎里读作第 4 档，界面必须显示 4；
    /// 字面量 1 是第 1 档，不能显示成归一化最大值。
    #[test]
    fn pf30210_display_reads_levels_through_the_engine() {
        let spec = slider_spec(30210).expect("PF30210");
        assert_eq!(ui_value(30210, &spec, 0.9), 4.0, "0.9 is the fourth level");
        assert_eq!(ui_value(30210, &spec, 1.0), 1.0, "a literal 1 is level one");
        assert_eq!(ui_value(30210, &spec, 0.0), 0.0);
        assert_eq!(ui_value(30210, &spec, 3.0), 3.0);
        // 别的行走自己的规格，不被 30210 的档位判定影响：
        // 脸型网格组本来就直接存百分比（`Identity`），0.5 就是 0.5 个单位。
        let mesh = slider_spec(30092).expect("PF30092");
        assert_eq!(
            mesh.scale,
            ValueScale::Identity,
            "face-mesh rows store percent directly"
        );
        assert_eq!(ui_value(30092, &mesh, 0.5), 0.5);
        // 真正的归一化行（Percent01）仍然 ×100：从描述符里现找，不假设某个 PF。
        let normalized = WIRED_EFFECT_DESCRIPTORS
            .iter()
            .map(|descriptor| descriptor.id.pf)
            .filter_map(|pf| slider_spec(pf).map(|spec| (pf, spec)))
            .find(|(_, spec)| spec.scale == ValueScale::Percent01)
            .expect("at least one normalized row");
        assert_eq!(ui_value(normalized.0, &normalized.1, 0.5), 50.0);
        // 写入侧仍然是整数档位。
        assert_eq!(spec.to_stored(2.4), 2.0);
        assert_eq!(spec.to_stored(4.0), 4.0);
    }

    /// PF90032 是眼妆（EyeMaskType=10 的 RGB 遮罩），归入眼睛。
    #[test]
    fn pf90032_is_grouped_under_eyes() {
        assert_eq!(category_of(90032), Some(EffectCategory::Eyes));
        assert_eq!(section_of(90032), Some("Eye makeup"));
        assert!(pfs_in(EffectCategory::Eyes).contains(&90032));
        assert!(!pfs_in(EffectCategory::Makeup).contains(&90032));
    }

    /// 皮肤分节里的两条旧表漏项也要出现（PF0 祛痘、PF90085 脸部肤色统一）。
    #[test]
    fn every_registered_skin_effect_has_a_spec_and_a_unit() {
        let skin = pfs_in(EffectCategory::Skin);
        assert!(skin.len() >= 25, "skin descriptors: {skin:?}");
        assert!(skin.contains(&0) && skin.contains(&90085));
        for pf in skin {
            let spec = slider_spec(pf).unwrap_or_else(|| panic!("PF{pf} has no slider spec"));
            assert!(spec.min < spec.max);
            assert!(
                !spec.unit.is_empty() || spec.step.is_some(),
                "PF{pf} must show a unit or a step"
            );
        }
    }
}
