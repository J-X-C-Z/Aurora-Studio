//! 人体姿态模型输出（K 点）到 Aurora 语义锚点、再到 69 槽 ABI 的适配层。
//!
//! ## 它解决什么问题
//!
//! 现有引擎链路有三层彼此不匹配（`BODY-POSE-ONNX-CONTRACT.md` §3.4）：
//!
//! 1. 开源姿态模型输出 **K 点**（COCO-17 / Halpe-26 / OpenPose-25 / MediaPipe-33 /
//!    WholeBody-133），各体系的索引顺序不同；
//! 2. Aurora 内部只认 **语义锚点**（肩/肘/腕/髋/膝/踝/鼻/耳/头顶点/颈根/中髋/脚跟/脚趾），
//!    见 `BODY-POSE-KEYPOINT-MAP.tsv` 的 `anchor` 列（A1–A17、X1–X5）；
//! 3. 体形着色器 ABI 是 **定长 69 槽**（`body_shader_abi.rs`，`BODY_POSE_KEYPOINT_COUNT = 69`），
//!    且槽位本身**不携带语义**。
//!
//! 本模块只做前两段的显式映射，第三段由一个**调用方显式给出**的 `SlotTable` 完成。
//! 这样做的原因是：vendor `oriSkeleton` 的 69 槽顺序至今未回收（§6.4 / U5），
//! 在语义未定义前，任何"按模型输出顺序直填 69 槽"的做法都是伪造。
//!
//! 三段映射：
//!
//! ```text
//! model index ──(ModelLayout::index_table)──▶ Anchor{A1..A17, X1..X5}
//! Anchor ──(自己维护的 SlotTable，版本化)──▶ slot 0..68
//! ```
//!
//! ## fail-closed 边界
//!
//! - **观测层**：非有限值、`present = false`、`score < min_score`、`x/y ∉ [0,1]`
//!   都记为"该锚点缺失"（`None`），不报错、不插值、不零填充。
//!   这是契约 D4 的"标记缺失"半边；本模块**不做 clamp**，因为 clamp 会合成出
//!   一个模型从未观测到的坐标，违反"禁止合成"。
//! - **输入层**：模型点数少于该体系索引表最大索引 + 1 时返回
//!   `ShapeMismatch("body pose model keypoints")`；`min_score` 非有限或不落在
//!   `[0,1]` 时返回 `ValueOutOfRange("body pose min score")`。
//! - **ABI 层**：`to_69` 要求 69 槽全部已指派语义；只要有任一槽为 `None`，整体返回
//!   `MissingInput("body pose 69 slot semantics")`。任一被引用的锚点缺失时返回
//!   `MissingInput("body pose anchor <name>")`。绝不返回部分填充的 69 槽。
//!
//! ## 与 69 槽 ABI 的关系
//!
//! - `SLOT_COUNT` 必须恒等于 `body_shader_abi::BODY_LANDMARK_COUNT`（69）。
//! - `to_69` 输出 `[Vec2; 69]`，坐标**原样**透传（UV，不 clamp），与
//!   `body_geometry` 的 UV 语义一致；是否越界已由 `adopt` 把关。
//! - `KEYPOINT_SEMANTICS_VERSION` 必须随推理收据落盘；语义表发生破坏性变更时递增。
//!   69 槽未回收前，消费方只允许按 `Anchor` 取点，**禁止按裸索引取点**。
//!
//! 本模块无外部依赖，只使用 `crate::kernels::body_geometry::Vec2` 与 `crate::Refusal`。

use crate::kernels::body_geometry::Vec2;
use crate::Refusal;

/// 语义锚点字典版本。与契约 §6.4 的 `aurora-body-anchor/1` 对应。
///
/// 破坏性变更（锚点改名、改义、删除）必须递增；新增锚点也递增，以便收据可判别。
pub const KEYPOINT_SEMANTICS_VERSION: u32 = 1;

/// 语义锚点总数（A1–A17 + X1–X5）。
pub const ANCHOR_COUNT: usize = 22;

/// Aurora 体形着色器 ABI 的固定槽位数，必须等于 `body_shader_abi::BODY_LANDMARK_COUNT`。
pub const SLOT_COUNT: usize = 69;

/// 默认置信阈值。
///
/// 契约 §13 U7 明确该阈值**尚未标定**；这里只给一个工程默认值，调用方必须显式传入
/// 自己的 `min_score`，以便后续按真实照片回归。
pub const MIN_SCORE_DEFAULT: f32 = 0.3;

/// 语义锚点。
///
/// 枚举顺序即 `Anchor::ALL` 与内部存储顺序；`index()` 返回该顺序下的下标。
/// `name()` 返回与 `BODY-POSE-KEYPOINT-MAP.tsv` 的 `anchor` 列一致的短名
/// （A1–A17、X1–X5），用于拒绝标签和日志。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Anchor {
    /// A13 鼻尖。
    Nose,
    /// A14 左耳。
    LeftEar,
    /// A15 右耳。
    RightEar,
    /// A16 头顶点。仅 Halpe-26 原生提供；其余体系须由鼻 + 双耳推导（本模块不推导）。
    HeadTop,
    /// A17 颈根。仅 Halpe-26 / OpenPose-25 提供（语义陷阱见契约 §6.1）。
    NeckBase,
    /// A1 左肩。
    LeftShoulder,
    /// A2 右肩。
    RightShoulder,
    /// A3 左肘。
    LeftElbow,
    /// A4 右肘。
    RightElbow,
    /// A5 左腕。
    LeftWrist,
    /// A6 右腕。
    RightWrist,
    /// A7 左髋。
    LeftHip,
    /// A8 右髋。
    RightHip,
    /// X1 左右髋中点。仅 Halpe-26 / OpenPose-25 原生提供。
    MidHip,
    /// A9 左膝。
    LeftKnee,
    /// A10 右膝。
    RightKnee,
    /// A11 左踝。
    LeftAnkle,
    /// A12 右踝。
    RightAnkle,
    /// X2 左脚跟。
    LeftHeel,
    /// X3 右脚跟。
    RightHeel,
    /// X4 左脚趾（大趾）；MediaPipe 只有近似脚尖。
    LeftToe,
    /// X5 右脚趾（大趾）；MediaPipe 只有近似脚尖。
    RightToe,
}

impl Anchor {
    /// 全部锚点，顺序与枚举声明一致，也与内部存储数组一致。
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

    /// 在 `Anchor::ALL` / 内部数组中的下标。
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

    /// 与 `BODY-POSE-KEYPOINT-MAP.tsv` 的 `anchor` 列一致的短名。
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

    /// 是否为解剖左侧锚点。模型自身的左右标注必须已按解剖侧归一，本函数只描述语义。
    pub const fn is_left(self) -> bool {
        matches!(
            self,
            Anchor::LeftEar
                | Anchor::LeftShoulder
                | Anchor::LeftElbow
                | Anchor::LeftWrist
                | Anchor::LeftHip
                | Anchor::LeftKnee
                | Anchor::LeftAnkle
                | Anchor::LeftHeel
                | Anchor::LeftToe
        )
    }

    /// 是否为解剖右侧锚点。
    pub const fn is_right(self) -> bool {
        matches!(
            self,
            Anchor::RightEar
                | Anchor::RightShoulder
                | Anchor::RightElbow
                | Anchor::RightWrist
                | Anchor::RightHip
                | Anchor::RightKnee
                | Anchor::RightAnkle
                | Anchor::RightHeel
                | Anchor::RightToe
        )
    }

    /// 该锚点缺失时使用的静态拒绝标签，形如 `"body pose anchor A1"`。
    ///
    /// 之所以做成静态表：`Refusal::MissingInput` 只接受 `&'static str`，
    /// 无法在运行期拼接锚点名。引擎层应复用本函数以保证标签一致。
    pub const fn missing_label(self) -> &'static str {
        MISSING_ANCHOR[self.index()]
    }
}

/// 与 `Anchor` 相同顺序的静态拒绝标签表。测试会校验它和 `Anchor::name()` 不漂移。
const MISSING_ANCHOR: [&str; ANCHOR_COUNT] = [
    "body pose anchor A13", // Nose
    "body pose anchor A14", // LeftEar
    "body pose anchor A15", // RightEar
    "body pose anchor A16", // HeadTop
    "body pose anchor A17", // NeckBase
    "body pose anchor A1",  // LeftShoulder
    "body pose anchor A2",  // RightShoulder
    "body pose anchor A3",  // LeftElbow
    "body pose anchor A4",  // RightElbow
    "body pose anchor A5",  // LeftWrist
    "body pose anchor A6",  // RightWrist
    "body pose anchor A7",  // LeftHip
    "body pose anchor A8",  // RightHip
    "body pose anchor X1",  // MidHip
    "body pose anchor A9",  // LeftKnee
    "body pose anchor A10", // RightKnee
    "body pose anchor A11", // LeftAnkle
    "body pose anchor A12", // RightAnkle
    "body pose anchor X2",  // LeftHeel
    "body pose anchor X3",  // RightHeel
    "body pose anchor X4",  // LeftToe
    "body pose anchor X5",  // RightToe
];

/// 单个模型关键点观测。
///
/// `present` 是显式存在性通道：遮挡、出画、实例归属不确定时置 `false`。
/// `score` 只在该体系确实给出置信度时有意义；无置信度的体系由调用方填 `1.0`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObservedKeypoint {
    pub x: f32,
    pub y: f32,
    pub score: f32,
    pub present: bool,
}

impl ObservedKeypoint {
    /// 构造一个存在且默认可用的观测。
    pub const fn new(x: f32, y: f32, score: f32) -> Self {
        Self {
            x,
            y,
            score,
            present: true,
        }
    }

    /// 判断该观测在给定阈值下是否可以进入语义层。
    ///
    /// 条件与 `adopt` 完全一致：存在、x/y/score 有限、x/y ∈ [0,1]、score ≥ min_score。
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

/// 支持的模型关键点布局。索引表来源：`BODY-POSE-KEYPOINT-MAP.tsv` 与契约 §6.2。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelLayout {
    /// COCO-17。索引 1/2 是左右眼，3/4 是左右耳；本模块不映射眼睛。
    Coco17,
    /// Halpe-26（RTMPose Body26）。前 17 与 COCO 同序，17 头顶点、18 颈根、19 中髋、20/21 大趾、24/25 脚跟。
    Halpe26,
    /// OpenPose BODY-25。1 颈（双肩中点）、8 中髋；19/22 脚尖、21/24 脚跟。
    OpenPose25,
    /// MediaPipe Pose 33。无头顶点、无颈根、无中髋；31/32 是近似脚尖。
    MediaPipe33,
    /// COCO-WholeBody 133 的 body 段（0–22）。本模块只读 body/foot，不读 face/hand。
    WholeBody133Body,
}

impl ModelLayout {
    /// 该布局名称，用于日志与收据。
    pub const fn name(self) -> &'static str {
        match self {
            ModelLayout::Coco17 => "coco17",
            ModelLayout::Halpe26 => "halpe26",
            ModelLayout::OpenPose25 => "openpose25",
            ModelLayout::MediaPipe33 => "mediapipe33",
            ModelLayout::WholeBody133Body => "wholebody133_body",
        }
    }

    /// 模型原生点数（不是索引表长度）。用于测试越界和文档说明。
    ///
    /// 注意 `adopt` 的长度校验按契约使用"索引表最大索引 + 1"；
    /// 对 `WholeBody133Body` 而言 23 个点即可通过校验，因为 body 段只到索引 22，
    /// face/hand 段不参与本适配层。真实模型仍应传满 133 点。
    pub const fn model_point_count(self) -> u32 {
        match self {
            ModelLayout::Coco17 => 17,
            ModelLayout::Halpe26 => 26,
            ModelLayout::OpenPose25 => 25,
            ModelLayout::MediaPipe33 => 33,
            ModelLayout::WholeBody133Body => 133,
        }
    }

    /// 返回 `(model_index, Anchor)` 表；只列该体系里确实存在的点。
    pub const fn index_table(self) -> &'static [(u32, Anchor)] {
        match self {
            ModelLayout::Coco17 => COCO17_TABLE,
            ModelLayout::Halpe26 => HALPE26_TABLE,
            ModelLayout::OpenPose25 => OPENPOSE25_TABLE,
            ModelLayout::MediaPipe33 => MEDIAPIPE33_TABLE,
            ModelLayout::WholeBody133Body => WHOLEBODY133_BODY_TABLE,
        }
    }
}

/// COCO-17：0 鼻、3/4 左右耳、5/6 肩、7/8 肘、9/10 腕、11/12 髋、13/14 膝、15/16 踝。
///
/// **不要**把索引 1/2（左右眼）映射成左右耳。
const COCO17_TABLE: &[(u32, Anchor)] = &[
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

/// Halpe-26：前 17 与 COCO 同序；17 头顶点、18 颈根、19 中髋、20/21 左右大趾、
/// 22/23 左右小趾（不映射）、24/25 左右脚跟。
const HALPE26_TABLE: &[(u32, Anchor)] = &[
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

/// OpenPose BODY-25：0 鼻、1 颈（双肩中点，语义上与 Halpe 颈根不同）、2/5 右/左肩、
/// 3/6 右/左肘、4/7 右/左腕、8 中髋、9/12 右/左髋、10/13 右/左膝、11/14 右/左踝、
/// 17/18 右/左耳、19/22 左/右大趾、21/24 左/右脚跟。无头顶点。
const OPENPOSE25_TABLE: &[(u32, Anchor)] = &[
    (0, Anchor::Nose),
    (1, Anchor::NeckBase),
    (2, Anchor::RightShoulder),
    (3, Anchor::RightElbow),
    (4, Anchor::RightWrist),
    (5, Anchor::LeftShoulder),
    (6, Anchor::LeftElbow),
    (7, Anchor::LeftWrist),
    (8, Anchor::MidHip),
    (9, Anchor::RightHip),
    (10, Anchor::RightKnee),
    (11, Anchor::RightAnkle),
    (12, Anchor::LeftHip),
    (13, Anchor::LeftKnee),
    (14, Anchor::LeftAnkle),
    (17, Anchor::RightEar),
    (18, Anchor::LeftEar),
    (19, Anchor::LeftToe),
    (21, Anchor::LeftHeel),
    (22, Anchor::RightToe),
    (24, Anchor::RightHeel),
];

/// MediaPipe Pose 33：0 鼻、7/8 左右耳、11/12 肩、13/14 肘、15/16 腕、23/24 髋、
/// 25/26 膝、27/28 踝、29/30 脚跟、31/32 脚尖（近似）。无头顶点/颈根/中髋。
const MEDIAPIPE33_TABLE: &[(u32, Anchor)] = &[
    (0, Anchor::Nose),
    (7, Anchor::LeftEar),
    (8, Anchor::RightEar),
    (11, Anchor::LeftShoulder),
    (12, Anchor::RightShoulder),
    (13, Anchor::LeftElbow),
    (14, Anchor::RightElbow),
    (15, Anchor::LeftWrist),
    (16, Anchor::RightWrist),
    (23, Anchor::LeftHip),
    (24, Anchor::RightHip),
    (25, Anchor::LeftKnee),
    (26, Anchor::RightKnee),
    (27, Anchor::LeftAnkle),
    (28, Anchor::RightAnkle),
    (29, Anchor::LeftHeel),
    (30, Anchor::RightHeel),
    (31, Anchor::LeftToe),
    (32, Anchor::RightToe),
];

/// COCO-WholeBody 133 的 body 段：0–16 与 COCO-17 同序；17 左脚大趾、18 左脚小趾、
/// 19 左脚跟、20 右脚大趾、21 右脚小趾、22 右脚跟。
const WHOLEBODY133_BODY_TABLE: &[(u32, Anchor)] = &[
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
    (17, Anchor::LeftToe),
    (19, Anchor::LeftHeel),
    (20, Anchor::RightToe),
    (22, Anchor::RightHeel),
];

/// 某次推理得到的语义锚点集合。
///
/// 字段私有：只能经 `adopt`（或 `empty`）构造，因此集合内每个 `Some` 都已通过
/// 存在性、有限性、范围和置信度检查。缺失的锚点以 `None` 表达，**不使用零填充**。
#[derive(Clone, Debug, PartialEq)]
pub struct AnchorSet {
    points: [Option<ObservedKeypoint>; ANCHOR_COUNT],
}

impl AnchorSet {
    /// 空集合：全部锚点缺失。
    pub fn empty() -> Self {
        Self {
            points: [None; ANCHOR_COUNT],
        }
    }

    /// 取某锚点的观测；缺失时为 `None`。
    pub fn get(&self, anchor: Anchor) -> Option<ObservedKeypoint> {
        self.points[anchor.index()]
    }

    /// 已观测到的锚点数量。
    pub fn present_count(&self) -> usize {
        self.points.iter().filter(|point| point.is_some()).count()
    }

    /// 取某锚点的观测；缺失时返回带锚点名的 `MissingInput`。
    pub fn require(&self, anchor: Anchor) -> Result<ObservedKeypoint, Refusal> {
        self.points[anchor.index()].ok_or(Refusal::MissingInput(anchor.missing_label()))
    }

    /// 按 `Anchor::ALL` 顺序迭代已观测的锚点。
    pub fn iter_present(&self) -> impl Iterator<Item = (Anchor, ObservedKeypoint)> + '_ {
        Anchor::ALL
            .iter()
            .copied()
            .zip(self.points.iter().copied())
            .filter_map(|(anchor, point)| point.map(|point| (anchor, point)))
    }
}

/// 把某模型的 K 点观测映射到语义锚点。fail-closed。
///
/// - 非有限（x/y/score 任一） / `present = false` / `score < min_score` /
///   `x` 或 `y` 不在 `[0,1]` → 该锚点记为 `None`，**不报错**。
/// - `model_points.len()` 小于该布局声明的模型点数（17/26/25/33/133）→
///   `Err(ShapeMismatch("body pose model keypoints"))`。用声明点数而不是
///   "索引表最大索引 + 1"，是为了让 `WholeBody133Body` 必须真的给到 133 点，
///   避免一个 23 点数组被当成全身模型。
/// - `min_score` 非有限或不在 `[0,1]` → `Err(ValueOutOfRange("body pose min score"))`。
///
/// 坐标系约定：输入 `x/y` 是归一化 UV（`[0,1]`），与 `body_geometry` 的 UV 同义；
/// 若模型输出的是像素坐标或 `[0,255]` 网格，调用方必须先按契约 §5 转成 UV。
pub fn adopt(
    layout: ModelLayout,
    model_points: &[ObservedKeypoint],
    min_score: f32,
) -> Result<AnchorSet, Refusal> {
    if !min_score.is_finite() || !(0.0..=1.0).contains(&min_score) {
        return Err(Refusal::ValueOutOfRange("body pose min score"));
    }

    let table = layout.index_table();
    let required = layout.model_point_count() as usize;
    if model_points.len() < required {
        return Err(Refusal::ShapeMismatch("body pose model keypoints"));
    }

    let mut points = [None; ANCHOR_COUNT];
    for &(model_index, anchor) in table {
        // 长度已校验；这里用 get 防御，绝不 panic。
        let Some(observed) = model_points.get(model_index as usize) else {
            continue;
        };
        if observed.usable(min_score) {
            points[anchor.index()] = Some(*observed);
        }
    }
    Ok(AnchorSet { points })
}

/// PF400（`slim_arm`）的六点槽位：源/目标控制点使用同一组锚点。
///
/// 顺序固定为
/// `[LeftShoulder, RightShoulder, LeftElbow, RightElbow, LeftWrist, RightWrist]`。
///
/// PF404（`hip_enhancement`）**不提供**关键点六点绑定，返回 `None`。原因：
/// PF404 的六个控制点是臀部外缘区域点（`B7 left_gluteal` / `B8 right_gluteal`
/// 与 `B6 gluteal_line`），而契约 §6.1 已证明这些是**区域锚点**，不是任何开源
/// 关键点体系中的点（`BODY-POSE-KEYPOINT-MAP.tsv` 中 B6–B8 各体系索引均为 `—`）。
/// 它们必须由"关键点定高度 + 分割定边界"合成，不能从姿态输出直接取。
/// 因此对 PF404 调用本函数得到 `None`，调用方必须显式携带合成后的六点。
pub fn six_point_anchors(pf: u32) -> Option<[Anchor; 6]> {
    match pf {
        400 => Some([
            Anchor::LeftShoulder,
            Anchor::RightShoulder,
            Anchor::LeftElbow,
            Anchor::RightElbow,
            Anchor::LeftWrist,
            Anchor::RightWrist,
        ]),
        // 404 的臀部外缘点不是关键点，见该函数文档。
        _ => None,
    }
}

/// 显式 69 槽语义表。
///
/// 全部 `None`（`UNASSIGNED`）表示"69 槽语义尚未回收"（契约 §6.4 / U5），
/// 此时 `to_69` 必须整体拒绝，而不是按模型输出顺序直填。
/// 该表的维护者是引擎层：只有拿到 vendor `oriSkeleton` 槽序证据后，才允许逐槽填入锚点，
/// 并递增 `KEYPOINT_SEMANTICS_VERSION`。
///
/// ## 本轮（2026-11-25，portrait-parse / pf404-slots）取证范围与结论
///
/// 对 `pix-worker.exe`（SHA-256 `509D2293…96B84`）的静态反汇编得到：
///
/// - `oriSkeleton` 是 `loadExtJsonParam`（RVA `0x1A95020`–`0x1A96E00`）从
///   **外部 extJson** 读取的 69 元素数组，元素只有 `x`/`y` 两个键
///   （`0x1A9685D`/`0x1A96885`），大小断言为 `0x228 == 69*8`
///   （`0x1A96B10`–`0x1A96B26`）。JSON 本身不携带点名或索引语义。
/// - 在 `.rdata`/`.text` 中检索 `left_shoulder`/`right_hip`/`gluteal` 等解剖名，
///   以及任何 69 元素名称表，均为 `NOT_FOUND`；`"x"/"y"` 之外没有点名字符串表。
/// - 唯一回收到的**索引消费**：`EffectButtockRefineCell`（PF404）的几何生产器
///   `0x3CB0A80` 固定读取骨架数组的槽 8..13（详见
///   `crate::kernels::buttock_anchors`）。这不构成解剖语义，且只覆盖 6/69 槽。
///
/// 因此本轮**未**回收任何可写入本表的 `Anchor`。所有槽保持 `None`，
/// `to_69` 继续整体 fail-closed。不要用 COCO-17 / Halpe-26 / WholeBody-133
/// 的常见顺序推测 vendor 槽序。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotTable {
    pub slots: [Option<Anchor>; SLOT_COUNT],
}

impl SlotTable {
    /// 全部未指派的槽表。
    pub const UNASSIGNED: Self = Self {
        slots: [None; SLOT_COUNT],
    };
}

/// 把 `AnchorSet` 展开成 69 槽。
///
/// - 任一槽为 `None` → `Err(MissingInput("body pose 69 slot semantics"))`（优先判定）。
/// - 某槽指向的锚点缺失 → `Err(MissingInput("body pose anchor <name>"))`，`name` 用
///   静态标签（见 `Anchor::missing_label`）。
/// - 成功返回 `[Vec2; 69]`，坐标原样透传（UV，不 clamp）。
pub fn to_69(anchors: &AnchorSet, table: &SlotTable) -> Result<[Vec2; SLOT_COUNT], Refusal> {
    // 语义未回收是比"某个点缺失"更根本的失败，先判，保证拒绝标签稳定。
    if table.slots.iter().any(|slot| slot.is_none()) {
        return Err(Refusal::MissingInput("body pose 69 slot semantics"));
    }

    let mut out = [Vec2::new(0.0, 0.0); SLOT_COUNT];
    for (index, slot) in table.slots.iter().enumerate() {
        let Some(anchor) = *slot else {
            // 已被上面的整体检查覆盖；保留分支是为了不依赖无 panic 的 unreachable。
            return Err(Refusal::MissingInput("body pose 69 slot semantics"));
        };
        let point = anchors.require(anchor)?;
        out[index] = Vec2::new(point.x, point.y);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 为某布局生成下标互不相同的合法观测，坐标严格落在 `[0,1]` 内。
    fn layout_points(layout: ModelLayout) -> Vec<ObservedKeypoint> {
        let count = layout.model_point_count() as usize;
        (0..count)
            .map(|index| {
                let t = (index as f32 + 0.5) / count as f32;
                ObservedKeypoint::new(t, 1.0 - t, 0.9)
            })
            .collect()
    }

    #[test]
    fn index_tables_have_unique_indices_and_anchors_in_range() {
        for layout in [
            ModelLayout::Coco17,
            ModelLayout::Halpe26,
            ModelLayout::OpenPose25,
            ModelLayout::MediaPipe33,
            ModelLayout::WholeBody133Body,
        ] {
            let table = layout.index_table();
            let mut seen_indices = std::collections::HashSet::new();
            let mut seen_anchors = std::collections::HashSet::new();
            for &(model_index, anchor) in table {
                assert!(
                    seen_indices.insert(model_index),
                    "{} 重复 model index {}",
                    layout.name(),
                    model_index
                );
                assert!(
                    seen_anchors.insert(anchor),
                    "{} 重复 anchor {}",
                    layout.name(),
                    anchor.name()
                );
                assert!(
                    model_index < layout.model_point_count(),
                    "{} model index {} 越界（点数 {}）",
                    layout.name(),
                    model_index,
                    layout.model_point_count()
                );
            }
        }
    }

    #[test]
    fn coco17_adopt_maps_joints() {
        let points = layout_points(ModelLayout::Coco17);
        let set = adopt(ModelLayout::Coco17, &points, MIN_SCORE_DEFAULT).unwrap();
        assert_eq!(set.present_count(), 15);

        let expected = [
            (Anchor::LeftShoulder, 5usize),
            (Anchor::RightShoulder, 6),
            (Anchor::LeftElbow, 7),
            (Anchor::RightElbow, 8),
            (Anchor::LeftWrist, 9),
            (Anchor::RightWrist, 10),
            (Anchor::LeftHip, 11),
            (Anchor::RightHip, 12),
            (Anchor::LeftKnee, 13),
            (Anchor::RightKnee, 14),
            (Anchor::LeftAnkle, 15),
            (Anchor::RightAnkle, 16),
            (Anchor::Nose, 0),
            (Anchor::LeftEar, 3),
            (Anchor::RightEar, 4),
        ];
        for (anchor, model_index) in expected {
            assert_eq!(
                set.get(anchor),
                Some(points[model_index]),
                "{} 映射错误",
                anchor.name()
            );
        }
        // 眼睛（1/2）不得被当作耳朵；COCO 也不提供这些锚点。
        assert_eq!(set.get(Anchor::HeadTop), None);
        assert_eq!(set.get(Anchor::NeckBase), None);
        assert_eq!(set.get(Anchor::MidHip), None);
        assert_eq!(set.get(Anchor::LeftHeel), None);
        assert_eq!(set.get(Anchor::RightToe), None);
    }

    #[test]
    fn adopt_marks_unusable_points_absent_without_error() {
        let mut points = layout_points(ModelLayout::Coco17);
        points[5] = ObservedKeypoint {
            score: 0.1,
            ..ObservedKeypoint::new(0.2, 0.2, 0.1)
        }; // LeftShoulder：低置信
        points[6] = ObservedKeypoint {
            present: false,
            ..ObservedKeypoint::new(0.3, 0.3, 0.99)
        }; // RightShoulder：显式缺失
        points[7] = ObservedKeypoint::new(1.5, 0.5, 0.99); // LeftElbow：x 越界
        points[8] = ObservedKeypoint {
            score: f32::NAN,
            ..ObservedKeypoint::new(0.4, 0.4, 0.9)
        }; // RightElbow：score 非有限
        points[9] = ObservedKeypoint::new(0.5, -0.1, 0.9); // LeftWrist：y 越界

        let set = adopt(ModelLayout::Coco17, &points, MIN_SCORE_DEFAULT).unwrap();
        for anchor in [
            Anchor::LeftShoulder,
            Anchor::RightShoulder,
            Anchor::LeftElbow,
            Anchor::RightElbow,
            Anchor::LeftWrist,
        ] {
            assert_eq!(set.get(anchor), None, "{} 应缺失", anchor.name());
        }
        assert_eq!(set.present_count(), 10);
        assert_eq!(set.get(Anchor::RightWrist), Some(points[10]));
    }

    #[test]
    fn adopt_rejects_short_input() {
        let short = layout_points(ModelLayout::Coco17);
        assert_eq!(
            adopt(ModelLayout::Coco17, &short[..16], MIN_SCORE_DEFAULT),
            Err(Refusal::ShapeMismatch("body pose model keypoints"))
        );
        let mediapipe = layout_points(ModelLayout::MediaPipe33);
        assert_eq!(
            adopt(ModelLayout::MediaPipe33, &mediapipe[..32], MIN_SCORE_DEFAULT),
            Err(Refusal::ShapeMismatch("body pose model keypoints"))
        );
    }

    #[test]
    fn adopt_rejects_bad_min_score() {
        let points = layout_points(ModelLayout::Coco17);
        for bad in [f32::NAN, f32::INFINITY, -0.1, 1.5] {
            assert_eq!(
                adopt(ModelLayout::Coco17, &points, bad),
                Err(Refusal::ValueOutOfRange("body pose min score")),
                "min_score {bad} 应被拒绝"
            );
        }
        assert!(adopt(ModelLayout::Coco17, &points, 0.0).is_ok());
        assert!(adopt(ModelLayout::Coco17, &points, 1.0).is_ok());
    }

    #[test]
    fn to_69_unassigned_slots_are_refused() {
        let points = layout_points(ModelLayout::Coco17);
        let set = adopt(ModelLayout::Coco17, &points, MIN_SCORE_DEFAULT).unwrap();
        assert_eq!(
            to_69(&set, &SlotTable::UNASSIGNED),
            Err(Refusal::MissingInput("body pose 69 slot semantics"))
        );
    }

    #[test]
    fn to_69_fills_every_slot() {
        let points = layout_points(ModelLayout::Coco17);
        let set = adopt(ModelLayout::Coco17, &points, MIN_SCORE_DEFAULT).unwrap();
        let table = SlotTable {
            slots: [Some(Anchor::LeftShoulder); SLOT_COUNT],
        };
        let slots = to_69(&set, &table).unwrap();
        let shoulder = points[5];
        assert_eq!(slots[0], Vec2::new(shoulder.x, shoulder.y));
        assert_eq!(slots[68], Vec2::new(shoulder.x, shoulder.y));
    }

    #[test]
    fn to_69_single_unassigned_slot_is_refused() {
        let points = layout_points(ModelLayout::Coco17);
        let set = adopt(ModelLayout::Coco17, &points, MIN_SCORE_DEFAULT).unwrap();
        let mut table = SlotTable {
            slots: [Some(Anchor::LeftShoulder); SLOT_COUNT],
        };
        table.slots[37] = None;
        assert_eq!(
            to_69(&set, &table),
            Err(Refusal::MissingInput("body pose 69 slot semantics"))
        );
    }

    #[test]
    fn to_69_reports_missing_anchor_label() {
        let mut points = layout_points(ModelLayout::Coco17);
        points[5] = ObservedKeypoint::new(f32::NAN, 0.5, 0.9); // LeftShoulder 缺失
        let set = adopt(ModelLayout::Coco17, &points, MIN_SCORE_DEFAULT).unwrap();
        let table = SlotTable {
            slots: [Some(Anchor::LeftShoulder); SLOT_COUNT],
        };
        assert_eq!(
            to_69(&set, &table),
            Err(Refusal::MissingInput("body pose anchor A1"))
        );
        assert_eq!(
            set.require(Anchor::MidHip),
            Err(Refusal::MissingInput("body pose anchor X1"))
        );
    }

    #[test]
    fn six_point_anchors_bind_pf400_only() {
        assert_eq!(
            six_point_anchors(400),
            Some([
                Anchor::LeftShoulder,
                Anchor::RightShoulder,
                Anchor::LeftElbow,
                Anchor::RightElbow,
                Anchor::LeftWrist,
                Anchor::RightWrist,
            ])
        );
        assert_eq!(six_point_anchors(404), None);
        assert_eq!(six_point_anchors(401), None);
        assert_eq!(six_point_anchors(0), None);
    }

    #[test]
    fn anchor_names_are_unique() {
        let mut names = std::collections::HashSet::new();
        for anchor in Anchor::ALL {
            assert!(
                names.insert(anchor.name()),
                "锚点名重复：{}",
                anchor.name()
            );
        }
        assert_eq!(names.len(), ANCHOR_COUNT);
    }

    #[test]
    fn anchor_index_round_trips() {
        for (index, anchor) in Anchor::ALL.iter().enumerate() {
            assert_eq!(anchor.index(), index, "{} 下标错误", anchor.name());
            assert_eq!(Anchor::ALL[anchor.index()], *anchor);
        }
    }

    #[test]
    fn missing_labels_match_anchor_names() {
        for anchor in Anchor::ALL {
            assert_eq!(
                anchor.missing_label(),
                format!("body pose anchor {}", anchor.name()),
                "{} 的拒绝标签漂移",
                anchor.name()
            );
        }
    }

    #[test]
    fn sidedness_is_exclusive_and_complete() {
        let mut left = 0;
        let mut right = 0;
        let mut midline = 0;
        for anchor in Anchor::ALL {
            assert!(
                !(anchor.is_left() && anchor.is_right()),
                "{} 同时被标为左右",
                anchor.name()
            );
            if anchor.is_left() {
                left += 1;
            } else if anchor.is_right() {
                right += 1;
            } else {
                midline += 1;
            }
        }
        assert_eq!(left, 9);
        assert_eq!(right, 9);
        assert_eq!(midline, 4);
        assert!(!Anchor::MidHip.is_left() && !Anchor::MidHip.is_right());
    }

    #[test]
    fn iter_present_skips_absent_in_anchor_order() {
        let mut points = layout_points(ModelLayout::Coco17);
        points[5] = ObservedKeypoint {
            present: false,
            ..ObservedKeypoint::new(0.2, 0.2, 0.9)
        };
        points[16] = ObservedKeypoint {
            present: false,
            ..ObservedKeypoint::new(0.9, 0.9, 0.9)
        };
        let set = adopt(ModelLayout::Coco17, &points, MIN_SCORE_DEFAULT).unwrap();
        let present: Vec<(Anchor, ObservedKeypoint)> = set.iter_present().collect();
        assert_eq!(present.len(), 13);
        assert_eq!(present.len(), set.present_count());
        assert!(present.iter().all(|(_, point)| point.present));
        let mut expected = Anchor::ALL
            .iter()
            .copied()
            .filter(|anchor| set.get(*anchor).is_some());
        for (anchor, _) in &present {
            assert_eq!(Some(*anchor), expected.next());
        }
        assert_eq!(expected.next(), None);
        assert_eq!(set.get(Anchor::LeftShoulder), None);
        assert_eq!(set.get(Anchor::RightAnkle), None);
    }

    #[test]
    fn empty_set_has_no_observations() {
        let set = AnchorSet::empty();
        assert_eq!(set.present_count(), 0);
        assert_eq!(set.iter_present().count(), 0);
        for anchor in Anchor::ALL {
            assert_eq!(set.get(anchor), None);
        }
    }
}
