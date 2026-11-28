//! `body_69slot_bridge.rs` — 标准人体姿态关键点 → `body_exact_vendor` 消费点的显式桥接草案。
//!
//! 纯 `std`、零外部 crate、零 `unsafe`、零 IO。单文件可用：
//!
//! ```text
//! rustc --edition 2021 --test body_69slot_bridge.rs
//! ```
//!
//! ## 这个模块解决什么
//!
//! 美体各 PF 的消费形态彼此不同，且 vendor 骨架语义大面积未回收：
//!
//! ```text
//! COCO-17 / Halpe-26 模型点
//!   └─ ModelLayout::index_table ──▶ Anchor（22 个语义锚点）
//!        ├─ PF404：vendor 69 槽的 8..=13（索引 PROVEN，解剖 NOT_FOUND）
//!        ├─ PF405：记录数组槽 0/21/22 + 查询槽 0（索引 PROVEN，数组归属/解剖 INFERRED）
//!        └─ PF417/418：每侧 3 个骨骼点（索引 PROVEN，关节语义 INFERRED）
//! ```
//!
//! 本模块只做**显式、可测试、可替换**的桥接，不声称任何槽位解剖语义已被反汇编证实。
//!
//! ## 证据口径（与 REPORT.md 一致）
//!
//! - [`Evidence::ProvenIndex`]：反汇编证明“该索引被读取”。**不等于**该索引的解剖含义已证。
//! - [`Evidence::Inferred`]：由上下文/几何推导，未经反汇编证实。所有由姿态点生成的
//!   槽位绑定、PF405 记录点、PF417/418 三点链都属于这一档。
//! - [`Evidence::Derived`]：由已有点做确定性算术（中点、归一化、方向轴），不是原厂观测。
//!
//! ## 硬边界
//!
//! 1. **不提供任何默认假设常量。** [`Pf404HipHypothesis`] 与 [`Pf405HipHypothesis`]
//!    的比例必须由调用方显式给出并通过 `validate`；本模块不内置“看起来合理”的数字。
//! 2. **不合成缺失点。** 锚点缺失/低置信/越界时 [`AnchorSet::require`] 返回具名
//!    [`Refusal::MissingInput`]，绝不零填充、不 clamp。
//! 3. **不把 69 槽表填满。** [`VendorSkeleton`] 是稀疏槽表；只有调用方显式写入的槽才存在。
//!    `PF404` 只读 8..=13，其余槽是否存在与它无关。
//! 4. **69 槽语义仍 `NOT_FOUND`。** 本模块不提供 `oriSkeleton` 的全量槽序；任何
//!    `Anchor → slot` 绑定都必须由调用方携带 [`Evidence`] 自行声明。
//! 5. 这些桥接产物只能用于让接线有类型、可单测；**不能**作为原厂同图对拍或验收依据
//!    （AGENTS.md 第 9、10 条）。

#![forbid(unsafe_code)]

/// 二维 float 向量（与 `body_geometry::Vec2` 同布局，但本文件独立自包含）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    pub fn add(self, other: Vec2) -> Vec2 {
        Vec2::new(self.x + other.x, self.y + other.y)
    }

    pub fn sub(self, other: Vec2) -> Vec2 {
        Vec2::new(self.x - other.x, self.y - other.y)
    }

    pub fn scale(self, factor: f32) -> Vec2 {
        Vec2::new(self.x * factor, self.y * factor)
    }

    pub fn dot(self, other: Vec2) -> f32 {
        self.x * other.x + self.y * other.y
    }

    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }

    /// 单位化。零长度/非有限返回 `None`（调用方转成具名拒绝）。
    pub fn normalize(self) -> Option<Vec2> {
        let length = self.length();
        if !length.is_finite() || length <= MIN_EXTENT {
            return None;
        }
        Some(self.scale(1.0 / length))
    }
}

/// 退化判定的统一阈值（UV 单位）。
const MIN_EXTENT: f32 = 1e-4;

pub fn midpoint(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::new(0.5 * (a.x + b.x), 0.5 * (a.y + b.y))
}

/// 具名拒绝。只使用仓库既有的三种变体，标签保持静态字符串。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    MissingInput(&'static str),
    ValueOutOfRange(&'static str),
    ShapeMismatch(&'static str),
}

/// 槽位绑定的证据等级。见模块文档。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evidence {
    /// 反汇编证明该索引被读取；解剖语义仍可能 `NOT_FOUND`。
    ProvenIndex,
    /// 推导/假设，未经反汇编证实。
    Inferred,
    /// 由已有点做确定性算术得到。
    Derived,
}

impl Evidence {
    pub const fn label(self) -> &'static str {
        match self {
            Evidence::ProvenIndex => "proven-index",
            Evidence::Inferred => "inferred",
            Evidence::Derived => "derived",
        }
    }

    /// 取两者中更弱的一档：`Inferred < Derived < ProvenIndex`。
    /// 只要链路中有一环是假设，整体就不能标成 `ProvenIndex`。
    pub const fn weaker(self, other: Evidence) -> Evidence {
        match (self, other) {
            (Evidence::Inferred, _) | (_, Evidence::Inferred) => Evidence::Inferred,
            (Evidence::Derived, _) | (_, Evidence::Derived) => Evidence::Derived,
            _ => Evidence::ProvenIndex,
        }
    }
}

// ---------------------------------------------------------------------------
// 语义锚点与模型布局
// ---------------------------------------------------------------------------

pub const ANCHOR_COUNT: usize = 22;

/// Aurora 语义锚点（与 `body_pose_adapter::Anchor` 同名同序，A1–A17 + X1–X5）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Anchor {
    Nose,
    LeftEar,
    RightEar,
    HeadTop,
    NeckBase,
    LeftShoulder,
    RightShoulder,
    LeftElbow,
    RightElbow,
    LeftWrist,
    RightWrist,
    LeftHip,
    RightHip,
    MidHip,
    LeftKnee,
    RightKnee,
    LeftAnkle,
    RightAnkle,
    LeftHeel,
    RightHeel,
    LeftToe,
    RightToe,
}

impl Anchor {
    pub const ALL: [Anchor; ANCHOR_COUNT] = [
        Anchor::Nose,
        Anchor::LeftEar,
        Anchor::RightEar,
        Anchor::HeadTop,
        Anchor::NeckBase,
        Anchor::LeftShoulder,
        Anchor::RightShoulder,
        Anchor::LeftElbow,
        Anchor::RightElbow,
        Anchor::LeftWrist,
        Anchor::RightWrist,
        Anchor::LeftHip,
        Anchor::RightHip,
        Anchor::MidHip,
        Anchor::LeftKnee,
        Anchor::RightKnee,
        Anchor::LeftAnkle,
        Anchor::RightAnkle,
        Anchor::LeftHeel,
        Anchor::RightHeel,
        Anchor::LeftToe,
        Anchor::RightToe,
    ];

    pub const fn index(self) -> usize {
        match self {
            Anchor::Nose => 0,
            Anchor::LeftEar => 1,
            Anchor::RightEar => 2,
            Anchor::HeadTop => 3,
            Anchor::NeckBase => 4,
            Anchor::LeftShoulder => 5,
            Anchor::RightShoulder => 6,
            Anchor::LeftElbow => 7,
            Anchor::RightElbow => 8,
            Anchor::LeftWrist => 9,
            Anchor::RightWrist => 10,
            Anchor::LeftHip => 11,
            Anchor::RightHip => 12,
            Anchor::MidHip => 13,
            Anchor::LeftKnee => 14,
            Anchor::RightKnee => 15,
            Anchor::LeftAnkle => 16,
            Anchor::RightAnkle => 17,
            Anchor::LeftHeel => 18,
            Anchor::RightHeel => 19,
            Anchor::LeftToe => 20,
            Anchor::RightToe => 21,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Anchor::Nose => "A13",
            Anchor::LeftEar => "A14",
            Anchor::RightEar => "A15",
            Anchor::HeadTop => "A16",
            Anchor::NeckBase => "A17",
            Anchor::LeftShoulder => "A1",
            Anchor::RightShoulder => "A2",
            Anchor::LeftElbow => "A3",
            Anchor::RightElbow => "A4",
            Anchor::LeftWrist => "A5",
            Anchor::RightWrist => "A6",
            Anchor::LeftHip => "A7",
            Anchor::RightHip => "A8",
            Anchor::MidHip => "X1",
            Anchor::LeftKnee => "A9",
            Anchor::RightKnee => "A10",
            Anchor::LeftAnkle => "A11",
            Anchor::RightAnkle => "A12",
            Anchor::LeftHeel => "X2",
            Anchor::RightHeel => "X3",
            Anchor::LeftToe => "X4",
            Anchor::RightToe => "X5",
        }
    }

    /// 缺失时使用的静态拒绝标签（`&'static str` 不能运行期拼接）。
    pub const fn missing_label(self) -> &'static str {
        match self {
            Anchor::Nose => "body pose anchor A13",
            Anchor::LeftEar => "body pose anchor A14",
            Anchor::RightEar => "body pose anchor A15",
            Anchor::HeadTop => "body pose anchor A16",
            Anchor::NeckBase => "body pose anchor A17",
            Anchor::LeftShoulder => "body pose anchor A1",
            Anchor::RightShoulder => "body pose anchor A2",
            Anchor::LeftElbow => "body pose anchor A3",
            Anchor::RightElbow => "body pose anchor A4",
            Anchor::LeftWrist => "body pose anchor A5",
            Anchor::RightWrist => "body pose anchor A6",
            Anchor::LeftHip => "body pose anchor A7",
            Anchor::RightHip => "body pose anchor A8",
            Anchor::MidHip => "body pose anchor X1",
            Anchor::LeftKnee => "body pose anchor A9",
            Anchor::RightKnee => "body pose anchor A10",
            Anchor::LeftAnkle => "body pose anchor A11",
            Anchor::RightAnkle => "body pose anchor A12",
            Anchor::LeftHeel => "body pose anchor X2",
            Anchor::RightHeel => "body pose anchor X3",
            Anchor::LeftToe => "body pose anchor X4",
            Anchor::RightToe => "body pose anchor X5",
        }
    }
}

/// 支持的模型布局。索引表只列该体系确实存在的点。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelLayout {
    Coco17,
    Halpe26,
}

impl ModelLayout {
    pub const fn name(self) -> &'static str {
        match self {
            ModelLayout::Coco17 => "coco17",
            ModelLayout::Halpe26 => "halpe26",
        }
    }

    pub const fn point_count(self) -> usize {
        match self {
            ModelLayout::Coco17 => 17,
            ModelLayout::Halpe26 => 26,
        }
    }

    pub const fn index_table(self) -> &'static [(usize, Anchor)] {
        match self {
            ModelLayout::Coco17 => COCO17_TABLE,
            ModelLayout::Halpe26 => HALPE26_TABLE,
        }
    }
}

/// COCO-17：0 鼻、3/4 耳、5/6 肩、7/8 肘、9/10 腕、11/12 髋、13/14 膝、15/16 踝。
/// 索引 1/2 是眼睛，**不映射**。
const COCO17_TABLE: &[(usize, Anchor)] = &[
    (0, Anchor::Nose),
    (3, Anchor::LeftEar),
    (4, Anchor::RightEar),
    (5, Anchor::LeftShoulder),
    (6, Anchor::RightShoulder),
    (7, Anchor::LeftElbow),
    (8, Anchor::RightElbow),
    (9, Anchor::LeftWrist),
    (10, Anchor::RightWrist),
    (11, Anchor::LeftHip),
    (12, Anchor::RightHip),
    (13, Anchor::LeftKnee),
    (14, Anchor::RightKnee),
    (15, Anchor::LeftAnkle),
    (16, Anchor::RightAnkle),
];

/// Halpe-26：前 17 与 COCO 同序；17 头顶点、18 颈根、19 中髋、20/21 大趾、24/25 脚跟。
const HALPE26_TABLE: &[(usize, Anchor)] = &[
    (0, Anchor::Nose),
    (3, Anchor::LeftEar),
    (4, Anchor::RightEar),
    (5, Anchor::LeftShoulder),
    (6, Anchor::RightShoulder),
    (7, Anchor::LeftElbow),
    (8, Anchor::RightElbow),
    (9, Anchor::LeftWrist),
    (10, Anchor::RightWrist),
    (11, Anchor::LeftHip),
    (12, Anchor::RightHip),
    (13, Anchor::LeftKnee),
    (14, Anchor::RightKnee),
    (15, Anchor::LeftAnkle),
    (16, Anchor::RightAnkle),
    (17, Anchor::HeadTop),
    (18, Anchor::NeckBase),
    (19, Anchor::MidHip),
    (20, Anchor::LeftToe),
    (21, Anchor::RightToe),
    (24, Anchor::LeftHeel),
    (25, Anchor::RightHeel),
];

/// 单个模型关键点观测。`present` 是显式存在性通道。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObservedKeypoint {
    pub x: f32,
    pub y: f32,
    pub score: f32,
    pub present: bool,
}

impl ObservedKeypoint {
    pub const fn new(x: f32, y: f32, score: f32) -> Self {
        Self {
            x,
            y,
            score,
            present: true,
        }
    }

    /// 存在、有限、`[0,1]` 且在阈值以上。
    pub fn usable(self, min_score: f32) -> bool {
        self.present
            && self.x.is_finite()
            && self.y.is_finite()
            && self.score.is_finite()
            && (0.0..=1.0).contains(&self.x)
            && (0.0..=1.0).contains(&self.y)
            && self.score >= min_score
    }
}

/// 某次推理得到的语义锚点集合。缺失以 `None` 表达，不零填充。
#[derive(Clone, Debug, PartialEq)]
pub struct AnchorSet {
    points: [Option<Vec2>; ANCHOR_COUNT],
}

impl AnchorSet {
    pub fn empty() -> Self {
        Self {
            points: [None; ANCHOR_COUNT],
        }
    }

    pub fn get(&self, anchor: Anchor) -> Option<Vec2> {
        self.points[anchor.index()]
    }

    pub fn present_count(&self) -> usize {
        self.points.iter().filter(|point| point.is_some()).count()
    }

    pub fn require(&self, anchor: Anchor) -> Result<Vec2, Refusal> {
        self.points[anchor.index()].ok_or(Refusal::MissingInput(anchor.missing_label()))
    }
}

/// 模型 K 点 → 语义锚点。fail-closed：不可用的点记为缺失，不报错、不插值。
pub fn adopt(
    layout: ModelLayout,
    model_points: &[ObservedKeypoint],
    min_score: f32,
) -> Result<AnchorSet, Refusal> {
    if !min_score.is_finite() || !(0.0..=1.0).contains(&min_score) {
        return Err(Refusal::ValueOutOfRange("body pose min score"));
    }
    if model_points.len() < layout.point_count() {
        return Err(Refusal::ShapeMismatch("body pose model keypoints"));
    }

    let mut points = [None; ANCHOR_COUNT];
    for &(index, anchor) in layout.index_table() {
        let Some(observed) = model_points.get(index) else {
            continue;
        };
        if observed.usable(min_score) {
            points[anchor.index()] = Some(Vec2::new(observed.x, observed.y));
        }
    }
    Ok(AnchorSet { points })
}

/// 中髋：Halpe-26 原生提供，直接取；其余体系由左右髋中点**派生**（`Derived`）。
pub fn mid_hip(set: &AnchorSet) -> Result<Vec2, Refusal> {
    if let Some(point) = set.get(Anchor::MidHip) {
        return Ok(point);
    }
    let left = set.require(Anchor::LeftHip)?;
    let right = set.require(Anchor::RightHip)?;
    Ok(midpoint(left, right))
}

// ---------------------------------------------------------------------------
// vendor 69 槽稀疏表
// ---------------------------------------------------------------------------

/// vendor `oriSkeleton` 固定点数。
pub const VENDOR_SLOT_COUNT: usize = 69;

/// PF404 几何生产器（`0x3CB0A80`）固定读取的槽位。**索引 PROVEN，解剖 NOT_FOUND。**
pub const PF404_VENDOR_SLOTS: [usize; 6] = [8, 9, 10, 11, 12, 13];

/// PF404 需要的最小骨架长度（最大索引 13 + 1）。
pub const PF404_REQUIRED_LEN: usize = 14;

/// 带证据标签的槽位点。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotPoint {
    pub point: Vec2,
    pub evidence: Evidence,
}

/// 一条显式槽位绑定：该槽来自哪个语义锚点、证据等级如何。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotBinding {
    pub anchor: Anchor,
    pub evidence: Evidence,
}

/// 稀疏 69 槽表。槽位下标即 vendor 槽号；未写入的槽为 `None`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SlotTable {
    slots: [Option<SlotBinding>; VENDOR_SLOT_COUNT],
}

impl SlotTable {
    /// 全部未指派。69 槽语义未回收前的唯一合法初始状态。
    pub const fn unassigned() -> Self {
        Self {
            slots: [None; VENDOR_SLOT_COUNT],
        }
    }

    /// 显式绑定某槽。越界或重复绑定都拒绝。
    pub fn assign(
        &mut self,
        slot: usize,
        anchor: Anchor,
        evidence: Evidence,
    ) -> Result<(), Refusal> {
        if slot >= VENDOR_SLOT_COUNT {
            return Err(Refusal::ValueOutOfRange("vendor slot index"));
        }
        if self.slots[slot].is_some() {
            return Err(Refusal::ValueOutOfRange("duplicate vendor slot"));
        }
        self.slots[slot] = Some(SlotBinding { anchor, evidence });
        Ok(())
    }

    pub fn binding(&self, slot: usize) -> Option<SlotBinding> {
        self.slots.get(slot).copied().flatten()
    }

    pub fn is_complete(&self) -> bool {
        self.slots.iter().all(|slot| slot.is_some())
    }
}

/// 按显式绑定表把锚点展开成稀疏 69 槽。
///
/// 只写入表里声明过的槽；任一被引用锚点缺失 → `MissingInput("body pose anchor …")`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VendorSkeleton {
    slots: [Option<SlotPoint>; VENDOR_SLOT_COUNT],
}

impl VendorSkeleton {
    pub const fn empty() -> Self {
        Self {
            slots: [None; VENDOR_SLOT_COUNT],
        }
    }

    pub fn from_anchors(set: &AnchorSet, table: &SlotTable) -> Result<Self, Refusal> {
        let mut skeleton = Self::empty();
        for slot in 0..VENDOR_SLOT_COUNT {
            let Some(binding) = table.slots[slot] else {
                continue;
            };
            let point = set.require(binding.anchor)?;
            if !point.is_finite() {
                return Err(Refusal::ValueOutOfRange("vendor skeleton point"));
            }
            skeleton.slots[slot] = Some(SlotPoint {
                point,
                evidence: binding.evidence,
            });
        }
        Ok(skeleton)
    }

    /// 显式写入若干槽（用于区域几何合成点）。重复槽/越界/非有限都拒绝。
    pub fn from_points(points: &[(usize, Vec2)], evidence: Evidence) -> Result<Self, Refusal> {
        let mut skeleton = Self::empty();
        for &(slot, point) in points {
            if slot >= VENDOR_SLOT_COUNT {
                return Err(Refusal::ValueOutOfRange("vendor slot index"));
            }
            if skeleton.slots[slot].is_some() {
                return Err(Refusal::ValueOutOfRange("duplicate vendor slot"));
            }
            if !point.is_finite() {
                return Err(Refusal::ValueOutOfRange("vendor skeleton point"));
            }
            skeleton.slots[slot] = Some(SlotPoint { point, evidence });
        }
        Ok(skeleton)
    }

    pub fn slot(&self, slot: usize) -> Option<SlotPoint> {
        self.slots.get(slot).copied().flatten()
    }

    /// 写入 PF404 的 6 个槽（8..=13）。
    pub fn write_pf404(
        &mut self,
        points: [Vec2; 6],
        evidence: Evidence,
    ) -> Result<(), Refusal> {
        if points.iter().any(|point| !point.is_finite()) {
            return Err(Refusal::ValueOutOfRange("PF404 vendor skeleton"));
        }
        for (index, slot) in PF404_VENDOR_SLOTS.iter().enumerate() {
            self.slots[*slot] = Some(SlotPoint {
                point: points[index],
                evidence,
            });
        }
        Ok(())
    }

    /// 按已证索引 `8..=13` 取 PF404 控制点。
    pub fn pf404_control_points(&self) -> Result<Pf404Slots, Refusal> {
        let mut points = [Vec2::new(0.0, 0.0); 6];
        let mut evidence = Evidence::ProvenIndex;
        for (index, slot) in PF404_VENDOR_SLOTS.iter().enumerate() {
            let Some(entry) = self.slots[*slot] else {
                return Err(Refusal::MissingInput("PF404 vendor skeleton"));
            };
            if !entry.point.is_finite() {
                return Err(Refusal::ValueOutOfRange("PF404 vendor skeleton"));
            }
            points[index] = entry.point;
            evidence = evidence.weaker(entry.evidence);
        }
        Ok(Pf404Slots { points, evidence })
    }
}

// ---------------------------------------------------------------------------
// PF404：臀部 6 槽
// ---------------------------------------------------------------------------

/// PF404 的 6 个控制点，顺序严格等于 [`PF404_VENDOR_SLOTS`]。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pf404Slots {
    pub points: [Vec2; 6],
    pub evidence: Evidence,
}

/// PF404 姿态假设的标识，必须随推理收据落盘。
pub const PF404_HYPOTHESIS_ID: &str = "pf404-hip-band-v1-INFERRED";

/// 从左右髋 + 左右膝构造 3×2 骨盆网格所需的比例。
///
/// **这些不是 vendor 常量。** 它们没有默认值，必须由调用方标定后显式传入。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pf404HipHypothesis {
    /// 髋线到臀带中心的垂直距离，按 `|knee_mid - hip_mid|` 的比例。
    pub gluteal_drop: f32,
    /// 臀带单侧上下两点相对中心的高度，按腿长比例。
    pub lobe_half_height: f32,
    /// 外侧点相对髋宽的比例。
    pub lateral_ratio: f32,
}

impl Pf404HipHypothesis {
    pub fn validate(self) -> Result<(), Refusal> {
        let values = [self.gluteal_drop, self.lobe_half_height, self.lateral_ratio];
        if values.iter().any(|value| !value.is_finite() || *value <= 0.0) {
            return Err(Refusal::ValueOutOfRange("PF404 hypothesis"));
        }
        Ok(())
    }
}

/// 由姿态轴建立的骨盆坐标系（私有）。
#[derive(Clone, Copy, Debug)]
struct PoseAxes {
    left_hip: Vec2,
    right_hip: Vec2,
    hip_mid: Vec2,
    /// 髋线 → 膝线方向的单位向量。
    down: Vec2,
    /// 左髋 → 右髋方向的单位向量。
    right: Vec2,
    /// 髋线到膝线的长度（UV）。
    leg: f32,
    /// 髋宽（UV）。
    hip_w: f32,
}

fn pose_axes(set: &AnchorSet) -> Result<PoseAxes, Refusal> {
    let left_hip = set.require(Anchor::LeftHip)?;
    let right_hip = set.require(Anchor::RightHip)?;
    let left_knee = set.require(Anchor::LeftKnee)?;
    let right_knee = set.require(Anchor::RightKnee)?;

    let hip_mid = midpoint(left_hip, right_hip);
    let knee_mid = midpoint(left_knee, right_knee);
    let leg_vec = knee_mid.sub(hip_mid);
    let leg = leg_vec.length();
    if !(leg > MIN_EXTENT) || !leg.is_finite() {
        return Err(Refusal::ValueOutOfRange("body pose leg extent"));
    }
    let down = leg_vec
        .normalize()
        .ok_or(Refusal::MissingInput("body pose down axis"))?;

    let hip_vec = right_hip.sub(left_hip);
    let hip_w = hip_vec.length();
    if !(hip_w > MIN_EXTENT) || !hip_w.is_finite() {
        return Err(Refusal::ValueOutOfRange("body pose hip width"));
    }
    let right = hip_vec
        .normalize()
        .ok_or(Refusal::MissingInput("body pose right axis"))?;

    Ok(PoseAxes {
        left_hip,
        right_hip,
        hip_mid,
        down,
        right,
        leg,
        hip_w,
    })
}

impl Pf404Slots {
    /// 从稀疏 69 槽按已证索引读取。`evidence` 只会等于或弱于槽位自身证据。
    pub fn from_vendor_skeleton(skeleton: &VendorSkeleton) -> Result<Self, Refusal> {
        skeleton.pf404_control_points()
    }

    /// **`INFERRED`**：由左右髋/膝生成 3×2 骨盆网格。
    ///
    /// 槽位假设（没有任何反汇编支持，仅为可替换的桥接）：
    ///
    /// ```text
    /// 9  = 左列上点      10 = 左列下点
    /// 11 = 右列上点      12 = 右列下点
    /// 8  = 左外侧点      13 = 右外侧点
    /// ```
    ///
    /// `M = mid(9,12)`、`C0 = mid(9,10)`、`C1 = mid(11,12)` 与原厂几何的
    /// “对角线 + 两列中点 + 外侧点”结构同构，但这只是假设。
    pub fn from_pose_hypothesis(
        set: &AnchorSet,
        hypothesis: Pf404HipHypothesis,
    ) -> Result<Self, Refusal> {
        hypothesis.validate()?;
        let axes = pose_axes(set)?;

        let left_center = axes
            .left_hip
            .add(axes.down.scale(hypothesis.gluteal_drop * axes.leg));
        let right_center = axes
            .right_hip
            .add(axes.down.scale(hypothesis.gluteal_drop * axes.leg));

        let up = axes.down.scale(-1.0);
        let half = hypothesis.lobe_half_height * axes.leg;
        let lateral = hypothesis.lateral_ratio * axes.hip_w;

        let left_outer = left_center.sub(axes.right.scale(lateral));
        let left_upper = left_center.add(up.scale(half));
        let left_lower = left_center.add(axes.down.scale(half));
        let right_upper = right_center.add(up.scale(half));
        let right_lower = right_center.add(axes.down.scale(half));
        let right_outer = right_center.add(axes.right.scale(lateral));

        let points = [
            left_outer,
            left_upper,
            left_lower,
            right_upper,
            right_lower,
            right_outer,
        ];
        if points.iter().any(|point| !point.is_finite()) {
            return Err(Refusal::ValueOutOfRange("PF404 pose hypothesis"));
        }
        Ok(Self {
            points,
            evidence: Evidence::Inferred,
        })
    }

    /// 写回稀疏 69 槽（8..=13）。证据等级随之写入。
    pub fn write_into(&self, skeleton: &mut VendorSkeleton) -> Result<(), Refusal> {
        skeleton.write_pf404(self.points, self.evidence)
    }
}

// ---------------------------------------------------------------------------
// PF405：4 个记录点
// ---------------------------------------------------------------------------

/// PF405 记录数组中已证的槽位（`0x4B44AC0` 返回值的 0/21/22）。
pub const PF405_RECORD_SLOTS: [usize; 3] = [0, 21, 22];
/// PF405 查询点数组的已证槽位（`0x4B446A0` 的槽 0）。
pub const PF405_QUERY_SLOT: usize = 0;
/// PF405 顶点生成器的已证半径系数 `radial0 = 0.7 * |center - record1b|`。
pub const PF405_RADIAL0_SCALE: f32 = 0.7;

/// PF405 `body_refine_vertices` 的四个入参。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pf405Records {
    pub query: Vec2,
    pub record0: Vec2,
    pub record1: Vec2,
    pub record1b: Vec2,
    pub evidence: Evidence,
}

/// PF405 姿态假设的标识，必须随推理收据落盘。
pub const PF405_HYPOTHESIS_ID: &str = "pf405-hip-ring-v1-INFERRED";

/// 由髋/膝建立 PF405 记录点所需的比例。
///
/// **这些不是 vendor 常量。** 必须由调用方标定后显式传入。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pf405HipHypothesis {
    /// `record1b` 相对髋线中点的向上距离，按腿长比例（必须 > 0）。
    pub record1b_up: f32,
    /// `query` 相对髋线中点的向下偏移，按腿长比例（可为 0）。
    pub query_down: f32,
}

impl Pf405HipHypothesis {
    pub fn validate(self) -> Result<(), Refusal> {
        if !self.record1b_up.is_finite()
            || self.record1b_up <= 0.0
            || !self.query_down.is_finite()
        {
            return Err(Refusal::ValueOutOfRange("PF405 hypothesis"));
        }
        Ok(())
    }
}

impl Pf405Records {
    /// **`INFERRED`**：`record0/record1` 取左右髋，`record1b` 由髋线中点向上、
    /// `query` 由髋线中点向下生成。
    ///
    /// 记录数组槽号 `0/21/22` 与查询槽 `0` 是 `PROVEN` 索引事实；
    /// “槽 0 = 左髋 / 槽 21 = 右髋 / 槽 22 = 上方点”没有反汇编支持。
    pub fn from_pose_hypothesis(
        set: &AnchorSet,
        hypothesis: Pf405HipHypothesis,
    ) -> Result<Self, Refusal> {
        hypothesis.validate()?;
        let axes = pose_axes(set)?;

        let record0 = axes.left_hip;
        let record1 = axes.right_hip;
        let record1b = axes
            .hip_mid
            .add(axes.down.scale(-hypothesis.record1b_up * axes.leg));
        let query = axes
            .hip_mid
            .add(axes.down.scale(hypothesis.query_down * axes.leg));

        let records = Self {
            query,
            record0,
            record1,
            record1b,
            evidence: Evidence::Inferred,
        };
        records.radial0()?;
        Ok(records)
    }

    /// `radial0 = 0.7 * |center - record1b|`，`center = mid(record0, record1)`。
    /// 退化基（中心与 record1b 重合）拒绝，不产生 `NaN`。
    pub fn radial0(&self) -> Result<f32, Refusal> {
        let center = midpoint(self.record0, self.record1);
        let distance = center.sub(self.record1b).length();
        if !distance.is_finite() || !(distance > MIN_EXTENT) {
            return Err(Refusal::MissingInput("PF405 record basis"));
        }
        Ok(PF405_RADIAL0_SCALE * distance)
    }
}

// ---------------------------------------------------------------------------
// PF417 / PF418：肩部端点与三点链
// ---------------------------------------------------------------------------

/// PF417/418 肩部三点链的证据标识。
pub const SHOULDER_HYPOTHESIS_ID: &str = "pf417-418-shoulder-chain-v1-INFERRED";

/// 颈根参考点的来源。`NeckBase` 需要 Halpe-26/OpenPose-25；`ShoulderMidpoint` 由双肩派生。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShoulderReference {
    NeckBase,
    ShoulderMidpoint,
}

impl ShoulderReference {
    pub const fn name(self) -> &'static str {
        match self {
            ShoulderReference::NeckBase => "neck-base",
            ShoulderReference::ShoulderMidpoint => "shoulder-midpoint",
        }
    }
}

/// 一侧肩部的三个骨骼点。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShoulderTriplet {
    pub prev: Vec2,
    pub at: Vec2,
    pub next: Vec2,
}

/// PF417/418 的 A/B 两侧三点链。
///
/// 与 `body_exact_vendor::ShoulderBones` 的字段顺序一致：
/// `[a_prev, a, a_next, b_prev, b, b_next]`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShoulderEndpoints {
    pub left: ShoulderTriplet,
    pub right: ShoulderTriplet,
    /// 颈根或肩线中点。仅用于让 `prev/next` 有明确来源，不进入 vendor 常量。
    pub reference: Vec2,
    pub evidence: Evidence,
}

/// 只取双肩端点（PF417/418 欠定特化的 `p1, p2`）。
pub fn shoulder_endpoints(set: &AnchorSet) -> Result<(Vec2, Vec2), Refusal> {
    Ok((
        set.require(Anchor::LeftShoulder)?,
        set.require(Anchor::RightShoulder)?,
    ))
}

impl ShoulderEndpoints {
    /// **`INFERRED`**：把三关节链绑到 vendor 的 3+3 骨骼点读取。
    ///
    /// 原厂只证明每侧读取 3 个点（A 侧 `+0x18/+0x20/+0x28`，
    /// B 侧 `+0x80/+0x88/+0x90`），没有证明这 6 个点分别是哪些关节。
    /// 这里的 A 侧 `prev` 与 B 侧 `next` 共用同一参考点，是为了匹配二进制
    /// A 侧用 `a - a_prev`、B 侧用 `b - b_next` 的不对称差值窗口。
    pub fn from_pose(
        set: &AnchorSet,
        reference: ShoulderReference,
    ) -> Result<Self, Refusal> {
        let left_shoulder = set.require(Anchor::LeftShoulder)?;
        let right_shoulder = set.require(Anchor::RightShoulder)?;
        let left_elbow = set.require(Anchor::LeftElbow)?;
        let right_elbow = set.require(Anchor::RightElbow)?;

        let reference_point = match reference {
            ShoulderReference::NeckBase => set.require(Anchor::NeckBase)?,
            ShoulderReference::ShoulderMidpoint => midpoint(left_shoulder, right_shoulder),
        };

        let endpoints = Self {
            left: ShoulderTriplet {
                prev: reference_point,
                at: left_shoulder,
                next: left_elbow,
            },
            right: ShoulderTriplet {
                prev: right_elbow,
                at: right_shoulder,
                next: reference_point,
            },
            reference: reference_point,
            evidence: Evidence::Inferred,
        };
        endpoints.line_direction()?;
        Ok(endpoints)
    }

    /// 欠定特化的两个端点 `(p1, p2)`。
    pub fn endpoints(&self) -> (Vec2, Vec2) {
        (self.left.at, self.right.at)
    }

    /// 直接喂给 `body_exact_vendor::ShoulderBones` 的 6 点数组。
    pub fn to_shoulder_bones_array(&self) -> [Vec2; 6] {
        [
            self.left.prev,
            self.left.at,
            self.left.next,
            self.right.prev,
            self.right.at,
            self.right.next,
        ]
    }

    /// 肩线方向（左肩 → 右肩）。退化时拒绝。
    pub fn line_direction(&self) -> Result<Vec2, Refusal> {
        self.right
            .at
            .sub(self.left.at)
            .normalize()
            .ok_or(Refusal::ValueOutOfRange("PF417 shoulder width"))
    }

    /// 肩线法线（顺时针 90°，UV 坐标系下指向下方）。
    /// 原厂 `u_dir` 的语义未证，调用方必须自行选择方向/符号。
    pub fn line_normal(&self) -> Result<Vec2, Refusal> {
        let direction = self.line_direction()?;
        Ok(Vec2::new(-direction.y, direction.x))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5 * (1.0 + a.abs() + b.abs())
    }

    fn vec_close(a: Vec2, b: Vec2) -> bool {
        close(a.x, b.x) && close(a.y, b.y)
    }

    fn coco_points() -> Vec<ObservedKeypoint> {
        let mut points = vec![ObservedKeypoint::new(0.5, 0.5, 0.9); 17];
        points[5] = ObservedKeypoint::new(0.40, 0.30, 0.9); // 左肩
        points[6] = ObservedKeypoint::new(0.60, 0.30, 0.9); // 右肩
        points[7] = ObservedKeypoint::new(0.30, 0.50, 0.9); // 左肘
        points[8] = ObservedKeypoint::new(0.70, 0.50, 0.9); // 右肘
        points[11] = ObservedKeypoint::new(0.40, 0.50, 0.9); // 左髋
        points[12] = ObservedKeypoint::new(0.60, 0.50, 0.9); // 右髋
        points[13] = ObservedKeypoint::new(0.42, 0.80, 0.9); // 左膝
        points[14] = ObservedKeypoint::new(0.58, 0.80, 0.9); // 右膝
        points
    }

    fn coco_set() -> AnchorSet {
        adopt(ModelLayout::Coco17, &coco_points(), 0.3).unwrap()
    }

    fn halpe_points() -> Vec<ObservedKeypoint> {
        let mut points = vec![ObservedKeypoint::new(0.5, 0.5, 0.9); 26];
        points[5] = ObservedKeypoint::new(0.40, 0.30, 0.9); // 左肩
        points[6] = ObservedKeypoint::new(0.60, 0.30, 0.9); // 右肩
        points[7] = ObservedKeypoint::new(0.30, 0.50, 0.9); // 左肘
        points[8] = ObservedKeypoint::new(0.70, 0.50, 0.9); // 右肘
        points[17] = ObservedKeypoint::new(0.50, 0.10, 0.9); // 头顶点
        points[18] = ObservedKeypoint::new(0.50, 0.28, 0.9); // 颈根
        points[19] = ObservedKeypoint::new(0.50, 0.50, 0.9); // 中髋
        points
    }

    // ---- 布局与观测 ----

    #[test]
    fn layout_tables_map_expected_joints() {
        let coco = ModelLayout::Coco17;
        assert_eq!(coco.point_count(), 17);
        assert_eq!(coco.name(), "coco17");
        assert_eq!(
            coco.index_table().iter().find(|(index, _)| *index == 5).map(|(_, anchor)| *anchor),
            Some(Anchor::LeftShoulder)
        );
        assert!(coco.index_table().iter().all(|(index, _)| *index < 17));

        let halpe = ModelLayout::Halpe26;
        assert_eq!(halpe.point_count(), 26);
        assert_eq!(halpe.name(), "halpe26");
        assert_eq!(
            halpe.index_table().iter().find(|(index, _)| *index == 18).map(|(_, anchor)| *anchor),
            Some(Anchor::NeckBase)
        );
        assert!(halpe.index_table().iter().all(|(index, _)| *index < 26));
    }

    #[test]
    fn adopt_maps_coco_and_marks_unavailable_absent() {
        let set = coco_set();
        assert_eq!(set.get(Anchor::LeftShoulder), Some(Vec2::new(0.40, 0.30)));
        assert_eq!(set.get(Anchor::RightKnee), Some(Vec2::new(0.58, 0.80)));
        assert_eq!(set.get(Anchor::HeadTop), None);
        assert_eq!(set.get(Anchor::NeckBase), None);
        assert_eq!(set.get(Anchor::MidHip), None);
        assert_eq!(set.present_count(), 15);

        let empty = AnchorSet::empty();
        assert_eq!(empty.present_count(), 0);
        assert_eq!(empty.get(Anchor::Nose), None);
    }

    #[test]
    fn adopt_is_fail_closed_on_bad_points() {
        let mut points = coco_points();
        points[5] = ObservedKeypoint {
            present: false,
            ..ObservedKeypoint::new(0.4, 0.3, 0.9)
        };
        points[6] = ObservedKeypoint::new(0.6, 0.3, 0.1); // 低分
        points[7] = ObservedKeypoint::new(1.5, 0.5, 0.9); // 越界
        let set = adopt(ModelLayout::Coco17, &points, 0.3).unwrap();
        assert_eq!(set.get(Anchor::LeftShoulder), None);
        assert_eq!(set.get(Anchor::RightShoulder), None);
        assert_eq!(set.get(Anchor::LeftElbow), None);
        assert_eq!(
            set.require(Anchor::LeftShoulder),
            Err(Refusal::MissingInput("body pose anchor A1"))
        );
    }

    #[test]
    fn adopt_rejects_short_input_and_bad_score() {
        let points = coco_points();
        assert_eq!(
            adopt(ModelLayout::Coco17, &points[..16], 0.3),
            Err(Refusal::ShapeMismatch("body pose model keypoints"))
        );
        for bad in [f32::NAN, -0.1, 1.5] {
            assert_eq!(
                adopt(ModelLayout::Coco17, &points, bad),
                Err(Refusal::ValueOutOfRange("body pose min score"))
            );
        }
    }

    #[test]
    fn mid_hip_prefers_native_and_derives_coco() {
        let coco = coco_set();
        assert!(vec_close(
            mid_hip(&coco).unwrap(),
            Vec2::new(0.5, 0.5)
        ));
        let halpe = adopt(ModelLayout::Halpe26, &halpe_points(), 0.3).unwrap();
        assert!(vec_close(
            mid_hip(&halpe).unwrap(),
            Vec2::new(0.5, 0.5)
        ));
    }

    // ---- 69 槽表 ----

    #[test]
    fn constants_match_recovered_indices() {
        assert_eq!(VENDOR_SLOT_COUNT, 69);
        assert_eq!(PF404_VENDOR_SLOTS, [8, 9, 10, 11, 12, 13]);
        assert_eq!(PF404_REQUIRED_LEN, 14);
        assert_eq!(PF405_RECORD_SLOTS, [0, 21, 22]);
        assert_eq!(PF405_QUERY_SLOT, 0);
        assert_eq!(PF405_RADIAL0_SCALE, 0.7);
    }

    #[test]
    fn slot_table_bounds_and_duplicates() {
        let mut table = SlotTable::unassigned();
        assert!(!table.is_complete());
        assert_eq!(table.binding(8), None);
        assert!(table.assign(8, Anchor::LeftHip, Evidence::Inferred).is_ok());
        assert_eq!(
            table.assign(8, Anchor::RightHip, Evidence::Inferred),
            Err(Refusal::ValueOutOfRange("duplicate vendor slot"))
        );
        assert_eq!(
            table.assign(69, Anchor::LeftHip, Evidence::Inferred),
            Err(Refusal::ValueOutOfRange("vendor slot index"))
        );
        let binding = table.binding(8).unwrap();
        assert_eq!(binding.anchor, Anchor::LeftHip);
        assert_eq!(binding.evidence, Evidence::Inferred);
    }

    #[test]
    fn vendor_skeleton_from_anchors_keeps_evidence() {
        let set = coco_set();
        let mut table = SlotTable::unassigned();
        for (offset, slot) in PF404_VENDOR_SLOTS.iter().enumerate() {
            let anchor = if offset % 2 == 0 {
                Anchor::LeftHip
            } else {
                Anchor::RightHip
            };
            table.assign(*slot, anchor, Evidence::ProvenIndex).unwrap();
        }
        let skeleton = VendorSkeleton::from_anchors(&set, &table).unwrap();
        let pf404 = Pf404Slots::from_vendor_skeleton(&skeleton).unwrap();
        assert_eq!(pf404.evidence, Evidence::ProvenIndex);
        assert!(vec_close(pf404.points[0], Vec2::new(0.40, 0.50)));
        assert!(vec_close(pf404.points[1], Vec2::new(0.60, 0.50)));

        // 重建一张带 Inferred 的表，验证 weakest-wins。
        let mut mixed = SlotTable::unassigned();
        for slot in PF404_VENDOR_SLOTS {
            let evidence = if slot == 13 {
                Evidence::Inferred
            } else {
                Evidence::ProvenIndex
            };
            mixed.assign(slot, Anchor::LeftHip, evidence).unwrap();
        }
        let skeleton = VendorSkeleton::from_anchors(&set, &mixed).unwrap();
        assert_eq!(
            skeleton.pf404_control_points().unwrap().evidence,
            Evidence::Inferred
        );
    }

    #[test]
    fn vendor_skeleton_refuses_missing_anchor_and_short_pf404() {
        let set = coco_set();
        let mut table = SlotTable::unassigned();
        table.assign(8, Anchor::HeadTop, Evidence::Inferred).unwrap();
        assert_eq!(
            VendorSkeleton::from_anchors(&set, &table),
            Err(Refusal::MissingInput("body pose anchor A16"))
        );
        assert_eq!(
            VendorSkeleton::empty().pf404_control_points(),
            Err(Refusal::MissingInput("PF404 vendor skeleton"))
        );
    }

    #[test]
    fn vendor_skeleton_from_points_validates() {
        let points = [
            (8usize, Vec2::new(0.3, 0.6)),
            (9, Vec2::new(0.4, 0.55)),
            (10, Vec2::new(0.4, 0.65)),
            (11, Vec2::new(0.6, 0.55)),
            (12, Vec2::new(0.6, 0.65)),
            (13, Vec2::new(0.7, 0.6)),
        ];
        let skeleton = VendorSkeleton::from_points(&points, Evidence::Inferred).unwrap();
        assert_eq!(skeleton.slot(8).unwrap().evidence, Evidence::Inferred);
        assert_eq!(skeleton.pf404_control_points().unwrap().points.len(), 6);
        assert_eq!(
            VendorSkeleton::from_points(&[(8, Vec2::new(0.1, 0.1)), (8, Vec2::new(0.2, 0.2))], Evidence::Inferred),
            Err(Refusal::ValueOutOfRange("duplicate vendor slot"))
        );
        assert_eq!(
            VendorSkeleton::from_points(&[(69, Vec2::new(0.1, 0.1))], Evidence::Inferred),
            Err(Refusal::ValueOutOfRange("vendor slot index"))
        );
        assert_eq!(
            VendorSkeleton::from_points(&[(8, Vec2::new(f32::NAN, 0.1))], Evidence::Inferred),
            Err(Refusal::ValueOutOfRange("vendor skeleton point"))
        );
    }

    // ---- PF404 ----

    #[test]
    fn pf404_pose_hypothesis_is_inferred_and_ordered() {
        let set = coco_set();
        let hypothesis = Pf404HipHypothesis {
            gluteal_drop: 0.5,
            lobe_half_height: 0.25,
            lateral_ratio: 0.5,
        };
        let slots = Pf404Slots::from_pose_hypothesis(&set, hypothesis).unwrap();
        assert_eq!(slots.evidence, Evidence::Inferred);

        let expected = [
            Vec2::new(0.30, 0.65), // 8 左外
            Vec2::new(0.40, 0.575), // 9 左上
            Vec2::new(0.40, 0.725), // 10 左下
            Vec2::new(0.60, 0.575), // 11 右上
            Vec2::new(0.60, 0.725), // 12 右下
            Vec2::new(0.70, 0.65), // 13 右外
        ];
        for (got, want) in slots.points.iter().zip(expected.iter()) {
            assert!(vec_close(*got, *want), "{got:?} vs {want:?}");
        }
        assert!(slots.points[0].x < slots.points[1].x);
        assert!(slots.points[5].x > slots.points[3].x);
        assert_eq!(PF404_HYPOTHESIS_ID, "pf404-hip-band-v1-INFERRED");
    }

    #[test]
    fn pf404_pose_hypothesis_round_trips_through_skeleton() {
        let set = coco_set();
        let slots = Pf404Slots::from_pose_hypothesis(
            &set,
            Pf404HipHypothesis {
                gluteal_drop: 0.5,
                lobe_half_height: 0.25,
                lateral_ratio: 0.5,
            },
        )
        .unwrap();
        let mut skeleton = VendorSkeleton::empty();
        slots.write_into(&mut skeleton).unwrap();
        let read_back = skeleton.pf404_control_points().unwrap();
        assert_eq!(read_back.points, slots.points);
        assert_eq!(read_back.evidence, Evidence::Inferred);
    }

    #[test]
    fn pf404_hypothesis_rejects_bad_parameters_and_pose() {
        let set = coco_set();
        for bad in [
            Pf404HipHypothesis { gluteal_drop: 0.0, lobe_half_height: 0.25, lateral_ratio: 0.5 },
            Pf404HipHypothesis { gluteal_drop: f32::NAN, lobe_half_height: 0.25, lateral_ratio: 0.5 },
            Pf404HipHypothesis { gluteal_drop: 0.5, lobe_half_height: -0.1, lateral_ratio: 0.5 },
        ] {
            assert_eq!(
                Pf404Slots::from_pose_hypothesis(&set, bad),
                Err(Refusal::ValueOutOfRange("PF404 hypothesis"))
            );
        }

        let mut points = coco_points();
        points[13] = points[11]; // 左膝 = 左髋
        points[14] = points[12]; // 右膝 = 右髋
        let degenerate = adopt(ModelLayout::Coco17, &points, 0.3).unwrap();
        assert_eq!(
            Pf404Slots::from_pose_hypothesis(
                &degenerate,
                Pf404HipHypothesis {
                    gluteal_drop: 0.5,
                    lobe_half_height: 0.25,
                    lateral_ratio: 0.5,
                },
            ),
            Err(Refusal::ValueOutOfRange("body pose leg extent"))
        );
    }

    // ---- PF405 ----

    #[test]
    fn pf405_records_from_pose_are_inferred() {
        let set = coco_set();
        let records = Pf405Records::from_pose_hypothesis(
            &set,
            Pf405HipHypothesis {
                record1b_up: 0.5,
                query_down: 0.0,
            },
        )
        .unwrap();
        assert_eq!(records.evidence, Evidence::Inferred);
        assert!(vec_close(records.record0, Vec2::new(0.40, 0.50)));
        assert!(vec_close(records.record1, Vec2::new(0.60, 0.50)));
        assert!(vec_close(records.record1b, Vec2::new(0.50, 0.35)));
        assert!(vec_close(records.query, Vec2::new(0.50, 0.50)));
        assert!(close(records.radial0().unwrap(), 0.7 * 0.15));
        assert_eq!(PF405_HYPOTHESIS_ID, "pf405-hip-ring-v1-INFERRED");
    }

    #[test]
    fn pf405_records_reject_degenerate_inputs() {
        let set = coco_set();
        for bad in [
            Pf405HipHypothesis { record1b_up: 0.0, query_down: 0.0 },
            Pf405HipHypothesis { record1b_up: f32::NAN, query_down: 0.0 },
            Pf405HipHypothesis { record1b_up: 0.5, query_down: f32::NAN },
        ] {
            assert_eq!(
                Pf405Records::from_pose_hypothesis(&set, bad),
                Err(Refusal::ValueOutOfRange("PF405 hypothesis"))
            );
        }

        let mut points = coco_points();
        points[11] = points[12]; // 双髋重合
        let degenerate = adopt(ModelLayout::Coco17, &points, 0.3).unwrap();
        assert_eq!(
            Pf405Records::from_pose_hypothesis(
                &degenerate,
                Pf405HipHypothesis {
                    record1b_up: 0.5,
                    query_down: 0.0,
                },
            ),
            Err(Refusal::ValueOutOfRange("body pose hip width"))
        );
    }

    // ---- PF417 / PF418 ----

    #[test]
    fn shoulder_endpoints_match_coco_and_bones_order() {
        let set = coco_set();
        let endpoints =
            ShoulderEndpoints::from_pose(&set, ShoulderReference::ShoulderMidpoint).unwrap();
        assert_eq!(endpoints.evidence, Evidence::Inferred);
        assert!(vec_close(endpoints.reference, Vec2::new(0.50, 0.30)));
        assert!(vec_close(endpoints.left.prev, Vec2::new(0.50, 0.30)));
        assert!(vec_close(endpoints.left.at, Vec2::new(0.40, 0.30)));
        assert!(vec_close(endpoints.left.next, Vec2::new(0.30, 0.50)));
        assert!(vec_close(endpoints.right.prev, Vec2::new(0.70, 0.50)));
        assert!(vec_close(endpoints.right.at, Vec2::new(0.60, 0.30)));
        assert!(vec_close(endpoints.right.next, Vec2::new(0.50, 0.30)));

        let (p1, p2) = endpoints.endpoints();
        assert!(vec_close(p1, Vec2::new(0.40, 0.30)));
        assert!(vec_close(p2, Vec2::new(0.60, 0.30)));

        let bones = endpoints.to_shoulder_bones_array();
        assert!(vec_close(bones[0], endpoints.left.prev));
        assert!(vec_close(bones[4], endpoints.right.at));

        assert!(vec_close(
            endpoints.line_direction().unwrap(),
            Vec2::new(1.0, 0.0)
        ));
        assert!(vec_close(
            endpoints.line_normal().unwrap(),
            Vec2::new(0.0, 1.0)
        ));
        assert_eq!(SHOULDER_HYPOTHESIS_ID, "pf417-418-shoulder-chain-v1-INFERRED");
        assert_eq!(ShoulderReference::NeckBase.name(), "neck-base");
        assert_eq!(
            ShoulderReference::ShoulderMidpoint.name(),
            "shoulder-midpoint"
        );
    }

    #[test]
    fn shoulder_endpoints_require_reference_and_elbows() {
        let set = coco_set();
        assert_eq!(
            ShoulderEndpoints::from_pose(&set, ShoulderReference::NeckBase),
            Err(Refusal::MissingInput("body pose anchor A17"))
        );

        let mut points = coco_points();
        points[7].present = false;
        let no_elbow = adopt(ModelLayout::Coco17, &points, 0.3).unwrap();
        assert_eq!(
            ShoulderEndpoints::from_pose(&no_elbow, ShoulderReference::ShoulderMidpoint),
            Err(Refusal::MissingInput("body pose anchor A3"))
        );
        // 只要两点端点时，肘缺失不影响。
        let (p1, p2) = shoulder_endpoints(&no_elbow).unwrap();
        assert!(vec_close(p1, Vec2::new(0.40, 0.30)));
        assert!(vec_close(p2, Vec2::new(0.60, 0.30)));

        // Halpe 有颈根。
        let halpe = adopt(ModelLayout::Halpe26, &halpe_points(), 0.3).unwrap();
        assert!(ShoulderEndpoints::from_pose(&halpe, ShoulderReference::NeckBase).is_ok());
    }

    #[test]
    fn shoulder_degenerate_width_is_refused() {
        let mut points = coco_points();
        points[6] = points[5];
        let set = adopt(ModelLayout::Coco17, &points, 0.3).unwrap();
        assert_eq!(
            ShoulderEndpoints::from_pose(&set, ShoulderReference::ShoulderMidpoint),
            Err(Refusal::ValueOutOfRange("PF417 shoulder width"))
        );
    }

    // ---- 证据与锚点 ----

    #[test]
    fn evidence_labels_and_weaker_fold() {
        assert_eq!(Evidence::ProvenIndex.label(), "proven-index");
        assert_eq!(Evidence::Inferred.label(), "inferred");
        assert_eq!(Evidence::Derived.label(), "derived");
        assert_eq!(
            Evidence::ProvenIndex.weaker(Evidence::Inferred),
            Evidence::Inferred
        );
        assert_eq!(
            Evidence::Derived.weaker(Evidence::ProvenIndex),
            Evidence::Derived
        );
        assert_eq!(
            Evidence::ProvenIndex.weaker(Evidence::ProvenIndex),
            Evidence::ProvenIndex
        );
    }

    #[test]
    fn anchor_names_and_indices_round_trip() {
        let mut names = std::collections::HashSet::new();
        for (index, anchor) in Anchor::ALL.iter().enumerate() {
            assert_eq!(anchor.index(), index);
            assert_eq!(Anchor::ALL[anchor.index()], *anchor);
            assert!(names.insert(anchor.name()));
            assert_eq!(
                anchor.missing_label(),
                format!("body pose anchor {}", anchor.name())
            );
        }
        assert_eq!(names.len(), ANCHOR_COUNT);
    }
}
