//! 模块 B 核心：八种 `BodyGeometryInput` 票据的工程几何生成器。
//!
//! 本模块把「PF + 曲线相对中性度数 + 输出画幅 + 语义锚点 + 当前 owner 软分割」
//! 变成 [`BodyGeometryInput`]。它只做**几何生产**：不采样像素、不合成掩码、
//! 不声明任何 OEM 常量。每个数字要么来自已解码的 vendor 控制曲线
//! （[`control_curve`] / [`LEG_FACTORS`]）、要么是带 `[D]` 标记的工程比例，
//! 要么由姿态锚点与分割边缘量出来。
//!
//! ## 输入契约
//!
//! - `degree` 是**曲线相对中性**的值：调用方已经把 `control_curve(pf)` 重映射后的
//!   度数减去该曲线的中性度数（`ControlCurve::default_degree()`）。于是
//!   `degree == 0` 恒等于「滑块在中性位置」：
//!   PF400 ∈ ±0.5、PF404 ∈ 0..0.5、PF405 ∈ ±2、PF407 ∈ 0..0.255、PF408 ∈ 0..1、
//!   PF416 ∈ ±1、PF38556 ∈ 0..1、PF30094 ∈ −1.25..0、PF30095 ∈ −1..0。
//!   PF30094/30095 的相对中性质是负值（vendor 默认度数就是曲线端点），而效果名是
//!   「拉长」，所以这两行取幅度绝对值，方向只由分段决定（见 [`leg_lengthen`]）。
//!   PF90041 没有 vendor 曲线：`degree` 已是**归一化有符号幅度**；若 `|degree| > 1`，
//!   按 −100..100 的 UI 量除以 100 解释（两种约定都能用，见
//!   [`normalized_degree_90041`]）。
//! - `mask` 是当前 owner 的**正向**软分割（0 = 背景，1 = 人体），长度必须等于
//!   `width * height`，像素中心归一化，与 `body_geometry` 的 UV 同域。
//!   内核要的是**保护极性**（1 = 不动）：本模块在
//!   `BodySlim.mask_r`、`Waist.mask_r`、`BreastLift.freeze_mask` 上翻转一次
//!   （复用 [`protect_mask_u8`] / [`protect_plane_f32`]），在
//!   `Shift.mask` 上保持正向（复合式 `effect * mask + original * (255 - mask)`）。
//! - `anchors` 是 `body_pose_adapter` 的语义锚点。缺失锚点直接用
//!   `Anchor::require` 的静态标签拒绝，不插值、不零填充。只有「不影响语义」的
//!   位置才退回掩码量测（如 PF408 的胯线、PF405 的起效线），并逐处标注 `[D]`。
//!
//! ## 每个 PF 的几何来源
//!
//! | PF | 票据 | 几何 |
//! |---|---|---|
//! | 400 | `SixPoint` | 六点 = `six_point_anchors(400)`（肩/肘/腕）；肘、腕沿该行人体质心方向内收 |
//! | 404 | `SixPoint` | 髋线 + 左膝方向的骨盆坐标系；外侧点取**掩码边缘**，其余为髋关节下 0.35 髋宽的 3×2 叶 |
//! | 405 | `BodySlim` | 逐行质心为轴的水平挤压场 + 保护极性掩码（白名单由肩线以下算起） |
//! | 407/30094/30095 | `Leg` | 五列 = (目标 V, 源 V) 分段线性拉长映射，见下节 |
//! | 408 | `Height` | 胯线→人体下缘的竖直拉伸，画面顶/底恒等 |
//! | 416 | `Waist` | 腰带 `WAIST_BAND` 内的竖直拉伸，位移在人体下缘吸收 |
//! | 38556 | `BreastLift` | 胸带中段掩码跨度的 30%/70% 为两叶中心，径向密度场 + 向上刷位 |
//! | 90041 | `Shift` | 与 405 同源的矢状挤压，打包成 16-bit 位移图（`ratio = 16`） |
//!
//! ## PF407/30094/30095 的列语义（generated consumer）
//!
//! `leg_refine_uv::LegColumn` 的字段名是 X 轴 strip 的遗留命名。本模块按
//! **转置轴**语义填列：
//!
//! - `clip_x` 当**目标 Y 的 clip 坐标**（`clip_y = 2 * target_v - 1`）；
//! - `sample_u` 当**源 V**；
//! - 两者同域：`v = 0` 画面顶、`v = 1` 画面底（与 `body_geometry` 的
//!   `dst_v = (y + 0.5) / H` 一致），于是
//!   **恒等 ⇔ `sample_u == (clip_x + 1) / 2`**；
//! - 段内 `source - target < 0`（向上取样）＝内容下移＝拉长；段外恒等。
//!
//! 五列严格递增（`clip_x` 递增、`sample_u` 递增），只映射指定的大腿/小腿段：
//! 段顶以上（含裤腰）恒等，段底以下在画面底边内吸收位移，因此不会去挤腰。
//!
//! ## 中性、边界与不畸变
//!
//! - 中性（`degree == 0`）一律返回**恒等票据**：UV 场 `src == dst`、
//!   六点 `dst == src` 且 `amount == 0`、`Shift` 全 0 位移、`BreastLift.degree01 == 0`。
//! - 场端点恒等（画面顶/底、段顶/段底）＋ 保护极性掩码（含 2% 边框羽化），
//!   所以帧边界与人体边缘不动。
//! - 所有竖直位移都满足 `d(src)/d(dst) > 0`（单调，不自交）：拉伸量被夹在
//!   `0.8 × 分段长度` 之内。
//!
//! ## 证据等级
//!
//! PF404 的六个控制点**不是** vendor 骨架槽 8..13：`body_pose_adapter::SlotTable`
//! 的解剖语义仍是 `NOT_FOUND`，`buttock_anchors` 只证明了索引。这里按
//! `body_69slot_bridge::PF404_HYPOTHESIS_ID`（`pf404-hip-band-v1-INFERRED`）的
//! 3×2 骨盆形状、但用**掩码边缘 + 髋部语义**重新定坐标，证据等级 `INFERRED`。
//! 调用方必须把本函数返回的 `source_points` 原样作为 `BodyGeometryEvidence::landmarks`
//! 传入，否则 `BodyGeometryJob::build` 会以
//! `ValueOutOfRange("body geometry landmark source points")` 拒绝。

use aurora_portrait::kernels::body_geometry::{Brush, PassDefines, Vec2};
use aurora_portrait::kernels::body_geometry_params::{control_curve, LEG_FACTORS};
use aurora_portrait::kernels::body_pose_adapter::{
    six_point_anchors, Anchor, AnchorSet, ObservedKeypoint,
};
use aurora_portrait::kernels::body_region_geometry::{
    protect_mask_u8, protect_plane_f32, TorsoFrame, BREAST_BAND, WAIST_BAND,
};
use aurora_portrait::kernels::leg_refine_uv::LegColumn;
use aurora_portrait::Refusal;

use super::portrait_body_stage::{BodyGeometryInput, BodyShiftInput, SixPointWarp};

/// 退化判定阈值（UV 或像素，按使用处语义）。
const EPS: f32 = 1e-6;

/// 掩码轮廓阈值：>= 0.5 视为该行/该列属于人体轮廓。
const SILHOUETTE: f32 = 0.5;

/// 边框羽化宽度（UV）：位移与保护掩码在画面最外 2% 归零，保证帧边界不被拖走。
const BORDER_MARGIN: f32 = 0.02;

/// 最短可映射段（UV）。视频帧高度 ~1000 px 时约 20 px。
const MIN_SEGMENT: f32 = 0.02;

/// 竖直拉伸后到吸收区的最近距离（UV）。
const MIN_ABSORB: f32 = 0.05;

/// 分段内单个节点的最小间隔（UV）。
const MIN_SPLIT: f32 = 0.005;

// ---- PF400 瘦手臂 [D] ------------------------------------------------------

/// 肘/腕沿解剖内侧的位移，按对应肢体长度（肩→肘、肘→腕）的比例。
const SLIM_ARM_FRACTION: f32 = 0.12;

// ---- PF404 美臀 [D] -------------------------------------------------------

/// 臀部叶中心相对髋关节沿 `down` 的下沉量，按髋宽比例（`pf404-hip-band-v1`）。
const BUTTOCK_GLUTEAL_DROP: f32 = 0.35;
/// 叶上下两点相对叶中心的高度，按髋宽比例。
const BUTTOCK_LOBE_HALF_HEIGHT: f32 = 0.30;
/// 掩码不可用时的外侧点外扩量，按髋宽比例。
const BUTTOCK_LATERAL_FALLBACK: f32 = 0.25;
/// 外侧点满强度外扩量，按髋宽比例。
const BUTTOCK_OUT: f32 = 0.10;
/// 外侧点满强度上提量，按髋宽比例。
const BUTTOCK_LIFT: f32 = 0.06;
/// 叶上点满强度上提量，按髋宽比例。
const BUTTOCK_UPPER: f32 = 0.06;
/// 叶下点满强度下沉量，按髋宽比例。
const BUTTOCK_LOWER: f32 = 0.05;

// ---- PF405 / PF90041 全身挤压 [D] ------------------------------------------

/// 起效线以上的保护带：人体高度比例（无肩锚点时的兜底）。
const HEAD_GUARD: f32 = 0.20;
/// PF405 水平挤压增益：`dx = gain * (u - axis)`。
const BODY_SLIM_GAIN: f32 = 0.10;
/// PF90041 水平挤压增益（与 PF405 同源）。
const BODY_SHIFT_GAIN: f32 = 0.10;
/// PF90041 打包位移图的量化比例：1/16 px 分辨率，量程 ±2048 px。
const SHIFT_RATIO: f32 = 16.0;

// ---- PF407 / PF30094 / PF30095 拉腿 [D] ------------------------------------

/// 段内拉伸比例上限：`stretch <= 0.12 * 段长`。
const LEG_STRETCH: f32 = 0.12;

// ---- PF408 增高 [D] -------------------------------------------------------

/// 胯线→人体下缘的竖直拉伸比例上限。
const HEIGHT_STRETCH: f32 = 0.12;
/// 无髋锚点时把胯线放在人体竖直范围的该比例处。
const MASK_HIP_FRACTION: f32 = 0.55;

// ---- PF416 拉腰 [D] -------------------------------------------------------

/// 腰带段内的竖直拉伸比例上限。
const WAIST_STRETCH: f32 = 0.10;

// ---- PF38556 丰胸 [D] -----------------------------------------------------

/// 两叶半径相对胸带中段掩码半跨度的比例。
const BREAST_RADIUS_RATIO: f32 = 0.55;
/// 满强度刷位位移相对叶片半径的比例。
const BREAST_LIFT_RATIO: f32 = 0.35;
/// 径向密度场的采样数（沿半径）。
const DENSITY_SAMPLES: usize = 32;

/// 生成某行的几何票据。
///
/// 参数语义见模块文档。`width * height` 必须等于 `mask.len()`；`degree` 必须是
/// 有限值；不在本表内的 PF 返回 [`Refusal::UnknownEffect`]（本函数不生成
/// `Mesh` 行 401/406/410/415）。
pub(super) fn generate(
    pf: u32,
    degree: f32,
    width: u32,
    height: u32,
    anchors: &aurora_portrait::kernels::body_pose_adapter::AnchorSet,
    mask: &[f32],
) -> Result<super::portrait_body_stage::BodyGeometryInput, aurora_portrait::Refusal> {
    if !degree.is_finite() {
        return Err(Refusal::ValueOutOfRange("body fields degree"));
    }
    let grid = MaskGrid::new(mask, width, height)?;
    match pf {
        400 => slim_arm(&grid, degree, anchors),
        404 => nice_buttock(&grid, degree, anchors),
        405 => body_slim(&grid, degree, anchors),
        407 | 30094 | 30095 => leg_lengthen(&grid, pf, degree, anchors),
        408 => height_increase(&grid, degree, anchors),
        416 => waist_lengthen(&grid, degree, anchors),
        38556 => breast_lift(&grid, degree, anchors),
        90041 => body_shift(&grid, degree, anchors),
        _ => Err(Refusal::UnknownEffect(pf)),
    }
}

// ---------------------------------------------------------------------------
// 掩码网格
// ---------------------------------------------------------------------------

/// 只读掩码视图：正向 0..1 软分割 + 人体竖直范围。
///
/// 构造时一次性校验长度、有限性、值域与「至少有一个非零像素」，
/// 并记录人体上下缘行号；之后的几何查询都是 O(W) 的行量测。
struct MaskGrid<'a> {
    mask: &'a [f32],
    width: usize,
    height: usize,
    top: usize,
    bottom: usize,
}

impl<'a> MaskGrid<'a> {
    fn new(mask: &'a [f32], width: u32, height: u32) -> Result<Self, Refusal> {
        if width == 0 || height == 0 {
            return Err(Refusal::ShapeMismatch("body fields frame"));
        }
        let (width, height) = (width as usize, height as usize);
        let count = width
            .checked_mul(height)
            .ok_or(Refusal::ShapeMismatch("body fields frame"))?;
        if mask.len() != count {
            return Err(Refusal::ShapeMismatch("body fields mask"));
        }
        let mut top = usize::MAX;
        let mut bottom = 0usize;
        let mut any = false;
        for (index, value) in mask.iter().enumerate() {
            if !value.is_finite() || *value < 0.0 || *value > 1.0 {
                return Err(Refusal::ValueOutOfRange("body fields mask"));
            }
            if *value > 0.0 {
                any = true;
                let row = index / width;
                top = top.min(row);
                bottom = bottom.max(row);
            }
        }
        if !any {
            return Err(Refusal::MissingInput("body fields subject mask"));
        }
        Ok(Self {
            mask,
            width,
            height,
            top,
            bottom,
        })
    }

    fn width(&self) -> usize {
        self.width
    }

    fn height(&self) -> usize {
        self.height
    }

    fn pixel_count(&self) -> usize {
        self.mask.len()
    }

    fn at(&self, x: usize, y: usize) -> f32 {
        self.mask[y * self.width + x]
    }

    /// 像素中心归一化 U。
    fn col_uv(&self, x: usize) -> f32 {
        (x as f32 + 0.5) / self.width as f32
    }

    /// 像素中心归一化 V（0 = 画面顶，1 = 画面底）。
    fn row_uv(&self, y: usize) -> f32 {
        (y as f32 + 0.5) / self.height as f32
    }

    /// UV 的 V 落在哪一行（夹到合法范围）。
    fn row_of(&self, v: f32) -> usize {
        let row = (v * self.height as f32 - 0.5).round();
        if row.is_finite() {
            (row.max(0.0) as usize).min(self.height - 1)
        } else {
            0
        }
    }

    fn subject_top(&self) -> usize {
        self.top.min(self.height - 1)
    }

    fn subject_bottom(&self) -> usize {
        self.bottom.min(self.height - 1)
    }

    /// 该行人体轮廓的 `(左列, 右列)`；该行没有达到轮廓阈值的像素时为 `None`。
    fn row_span(&self, y: usize) -> Option<(usize, usize)> {
        let mut left = None;
        let mut right = None;
        for x in 0..self.width {
            if self.at(x, y) >= SILHOUETTE {
                if left.is_none() {
                    left = Some(x);
                }
                right = Some(x);
            }
        }
        left.zip(right)
    }

    /// 该行掩码加权的质心 U（无像素时为 `None`）。
    fn row_center(&self, y: usize) -> Option<f32> {
        let mut weighted = 0.0f32;
        let mut weight = 0.0f32;
        for x in 0..self.width {
            let value = self.at(x, y);
            weighted += self.col_uv(x) * value;
            weight += value;
        }
        if weight > 0.0 && weighted.is_finite() {
            Some(weighted / weight)
        } else {
            None
        }
    }

    /// 全图掩码加权的质心 U。
    fn global_center(&self) -> f32 {
        let mut weighted = 0.0f64;
        let mut weight = 0.0f64;
        for y in 0..self.height {
            for x in 0..self.width {
                let value = self.at(x, y) as f64;
                weighted += self.col_uv(x) as f64 * value;
                weight += value;
            }
        }
        if weight > 0.0 && weighted.is_finite() {
            (weighted / weight) as f32
        } else {
            0.5
        }
    }
}

// ---------------------------------------------------------------------------
// 小工具
// ---------------------------------------------------------------------------

fn add(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::new(a.x + b.x, a.y + b.y)
}

fn sub(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::new(a.x - b.x, a.y - b.y)
}

fn mul(a: Vec2, k: f32) -> Vec2 {
    Vec2::new(a.x * k, a.y * k)
}

fn length(a: Vec2) -> f32 {
    (a.x * a.x + a.y * a.y).sqrt()
}

fn normalize(a: Vec2) -> Option<Vec2> {
    let len = length(a);
    if len.is_finite() && len > EPS {
        Some(Vec2::new(a.x / len, a.y / len))
    } else {
        None
    }
}

fn uv_point(point: ObservedKeypoint) -> Vec2 {
    Vec2::new(point.x, point.y)
}

/// 两点的**像素距离**（各向同性；UV 的 U/V 步长不同）。
fn pixel_len(grid: &MaskGrid<'_>, a: Vec2, b: Vec2) -> f32 {
    let dx = (b.x - a.x) * grid.width() as f32;
    let dy = (b.y - a.y) * grid.height() as f32;
    (dx * dx + dy * dy).sqrt()
}

/// 沿某单位轴走 1 UV 对应的像素长度，用来把像素尺度换成该轴上的 UV 尺度。
fn axis_pixels(grid: &MaskGrid<'_>, axis: Vec2) -> f32 {
    pixel_len(grid, Vec2::new(0.0, 0.0), axis).max(EPS)
}

/// `smoothstep`：0 以下 0、1 以上 1、中间 `3t² - 2t³`。
fn smoothstep(value: f32) -> f32 {
    let t = value.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// 从 `lo` 到 `hi` 的软起效（`lo >= hi` 时退化为阶跃）。
fn ramp(value: f32, lo: f32, hi: f32) -> f32 {
    if !(hi > lo) {
        return if value >= hi { 1.0 } else { 0.0 };
    }
    smoothstep((value - lo) / (hi - lo))
}

/// 画面边框羽化：最外 `margin` 内线性降到 0，内部为 1。
fn border_fade(u: f32, v: f32, margin: f32) -> f32 {
    if !(margin > 0.0) {
        return 1.0;
    }
    let du = u.min(1.0 - u) / margin;
    let dv = v.min(1.0 - v) / margin;
    du.min(dv).clamp(0.0, 1.0)
}

/// vendor 控制曲线的相对中性质幅度。
///
/// `span = max |degrees[i] - default_degree|`，用来把「曲线相对中性」的 `degree`
/// 归一化到 `[-1, 1]`。曲线不存在（PF90041）时返回
/// `ValueOutOfRange("body control span")`，由各 PF 自行处理。
fn curve_span(pf: u32) -> Result<f32, Refusal> {
    let curve = control_curve(pf).ok_or(Refusal::ValueOutOfRange("body control span"))?;
    let neutral = curve.default_degree()?;
    let mut span = 0.0f32;
    for index in 0..curve.nodes.min(3) {
        span = span.max((curve.degrees[index] - neutral).abs());
    }
    if !span.is_finite() || !(span > 0.0) {
        return Err(Refusal::ValueOutOfRange("body control span"));
    }
    Ok(span)
}

/// 归一化到 `[-1, 1]` 的有符号幅度。`degree == 0` 时严格返回 0。
fn normalized_degree(pf: u32, degree: f32) -> Result<f32, Refusal> {
    if !degree.is_finite() {
        return Err(Refusal::ValueOutOfRange("body fields degree"));
    }
    if degree == 0.0 {
        return Ok(0.0);
    }
    let span = curve_span(pf)?;
    Ok((degree / span).clamp(-1.0, 1.0))
}

/// PF90041 的幅度：没有 vendor 曲线，`degree` 已是归一化有符号幅度；
/// `|degree| > 1` 时按 −100..100 的 UI 量解释。
fn normalized_degree_90041(degree: f32) -> Result<f32, Refusal> {
    if !degree.is_finite() {
        return Err(Refusal::ValueOutOfRange("body fields degree"));
    }
    let value = if degree.abs() > 1.0 {
        degree / 100.0
    } else {
        degree
    };
    Ok(value.clamp(-1.0, 1.0))
}

/// 躯干坐标系：四点锚点（双肩 + 双髋）。
fn torso_frame(anchors: &AnchorSet) -> Result<TorsoFrame, Refusal> {
    let shoulder_left = uv_point(anchors.require(Anchor::LeftShoulder)?);
    let shoulder_right = uv_point(anchors.require(Anchor::RightShoulder)?);
    let hip_left = uv_point(anchors.require(Anchor::LeftHip)?);
    let hip_right = uv_point(anchors.require(Anchor::RightHip)?);
    TorsoFrame::new(shoulder_left, shoulder_right, hip_left, hip_right)
}

/// 左右锚点对的平均 V；只有一个时用那一个，都没有时 `None`。
fn pair_v(anchors: &AnchorSet, left: Anchor, right: Anchor) -> Option<f32> {
    match (anchors.get(left), anchors.get(right)) {
        (Some(a), Some(b)) => Some(0.5 * (a.y + b.y)),
        (Some(one), None) | (None, Some(one)) => Some(one.y),
        (None, None) => None,
    }
}

/// 胯线 V：中髋锚点 → 双髋平均 → 掩码竖直范围的 `MASK_HIP_FRACTION`（`[D]`）。
fn hip_line_v(anchors: &AnchorSet, grid: &MaskGrid<'_>) -> f32 {
    if let Some(mid) = anchors.get(Anchor::MidHip) {
        return mid.y.clamp(0.0, 1.0);
    }
    if let Some(v) = pair_v(anchors, Anchor::LeftHip, Anchor::RightHip) {
        return v.clamp(0.0, 1.0);
    }
    let top = grid.row_uv(grid.subject_top());
    let bottom = grid.row_uv(grid.subject_bottom());
    (top + (bottom - top) * MASK_HIP_FRACTION).clamp(0.0, 1.0)
}

/// 竖直拉伸位移：`start..end` 内从 0 拉到 `amount`，`end..absorb_end` 内回到 0。
///
/// `amount < 0` 表示向上取样（内容下移 = 拉伸）；`amount > 0` 反之。
/// 返回值在 `absorb_end` 之外严格为 0，因此画面/人体下缘保持恒等。
fn stretch_delta(v: f32, start: f32, end: f32, absorb_end: f32, amount: f32) -> f32 {
    if !(absorb_end > end) || !(end > start) || !(v > start) || v >= absorb_end {
        return 0.0;
    }
    if v <= end {
        return -amount * smoothstep((v - start) / (end - start));
    }
    -amount * (1.0 - smoothstep((v - end) / (absorb_end - end)))
}

// ---------------------------------------------------------------------------
// PF400 瘦手臂
// ---------------------------------------------------------------------------

/// PF400：六点 BodyWarp 的瘦手臂袖套。
///
/// `source_points` 严格等于 `six_point_anchors(400)` 的锚点坐标（
/// `[左肩, 右肩, 左肘, 右肘, 左腕, 右腕]`）。`destination_points` 把肘、腕沿
/// **该行人体质心方向**水平内收 `SLIM_ARM_FRACTION × 对应肢体长度`；双肩
/// `dst == src`（位移为 0 ⇒ 余弦阈值半径为 0 ⇒ 不参与），`degrees` 也置 0。
///
/// 内核把 `amount` 平方（`strength * strength`），所以这里传 `sqrt(|amp|)`，
/// 使可见位移对控制量线性。位移符号由 `amp` 决定：正 = 收窄，负 = 外扩。
fn slim_arm(
    grid: &MaskGrid<'_>,
    degree: f32,
    anchors: &AnchorSet,
) -> Result<BodyGeometryInput, Refusal> {
    let amp = normalized_degree(400, degree)?;
    let slots = six_point_anchors(400).ok_or(Refusal::UnknownEffect(400))?;
    let mut source_points = [Vec2::new(0.0, 0.0); 6];
    for (index, anchor) in slots.iter().enumerate() {
        source_points[index] = uv_point(anchors.require(*anchor)?);
    }

    if amp == 0.0 {
        return Ok(BodyGeometryInput::SixPoint(SixPointWarp {
            source_points,
            destination_points: source_points,
            degrees: [1.0; 6],
            amount: 0.0,
            inverse_aspect: grid.width() as f32 / grid.height() as f32,
            valid_point_count: 6,
        }));
    }

    let sign = if amp > 0.0 { 1.0 } else { -1.0 };
    let fallback_axis = 0.5 * (source_points[0].x + source_points[1].x);
    let mut destination_points = source_points;
    let mut degrees = [0.0f32; 6];
    // (关节下标, 肢体起点下标)：肘按肱骨、腕按前臂取尺度。
    for (joint, limb_start) in [(2usize, 0usize), (3, 1), (4, 2), (5, 3)] {
        let anchor_point = source_points[joint];
        let limb_px = pixel_len(grid, source_points[limb_start], anchor_point);
        if !(limb_px > EPS) {
            continue;
        }
        let axis = grid
            .row_center(grid.row_of(anchor_point.y))
            .unwrap_or(fallback_axis);
        let medial = if axis >= anchor_point.x { 1.0 } else { -1.0 };
        let shift_px = sign * medial * SLIM_ARM_FRACTION * limb_px;
        destination_points[joint] = Vec2::new(
            (anchor_point.x + shift_px / grid.width() as f32).clamp(0.0, 1.0),
            anchor_point.y,
        );
        degrees[joint] = 1.0;
    }

    Ok(BodyGeometryInput::SixPoint(SixPointWarp {
        source_points,
        destination_points,
        degrees,
        amount: amp.abs().sqrt(),
        inverse_aspect: grid.width() as f32 / grid.height() as f32,
        valid_point_count: 6,
    }))
}

// ---------------------------------------------------------------------------
// PF404 美臀
// ---------------------------------------------------------------------------

/// PF404：由**髋部语义 + 掩码边缘**合成的六点美臀袖套。
///
/// 骨盆坐标系：`right` = 左髋→右髋，`down` = 髋中点→膝中点（无膝锚点时取
/// 画面向下，标 `[D]`）。叶中心 = 髋关节沿 `down` 下沉
/// `BUTTOCK_GLUTEAL_DROP × 髋宽`，叶上下点再各偏 `BUTTOCK_LOBE_HALF_HEIGHT × 髋宽`；
/// 两个外侧点取该行**掩码轮廓边缘**（拿不到时退回髋关节外
/// `BUTTOCK_LATERAL_FALLBACK × 髋宽`）。
///
/// 满强度位移：外侧点外扩 + 上提、叶上点上提、叶下点下沉，全部按髋宽比例；
/// 方向由 `amp` 的符号决定（正 = 更饱满/更翘）。`amount` 同样预取平方根。
///
/// 证据等级 `INFERRED`：坐标来自掩码与髋部语义，不是 vendor 骨架槽 8..13
/// （见 `buttock_anchors` 的证据边界）。调用方必须把返回的 `source_points`
/// 原样当 `BodyGeometryEvidence::landmarks` 传入。
fn nice_buttock(
    grid: &MaskGrid<'_>,
    degree: f32,
    anchors: &AnchorSet,
) -> Result<BodyGeometryInput, Refusal> {
    let amp = normalized_degree(404, degree)?;
    let hip_left = uv_point(anchors.require(Anchor::LeftHip)?);
    let hip_right = uv_point(anchors.require(Anchor::RightHip)?);
    let hip_width_px = pixel_len(grid, hip_left, hip_right);
    if !hip_width_px.is_finite() || !(hip_width_px > EPS) {
        return Err(Refusal::ValueOutOfRange("body fields hip axis"));
    }
    let right_axis = normalize(sub(hip_right, hip_left))
        .ok_or(Refusal::ValueOutOfRange("body fields hip axis"))?;
    let down_axis = match (
        anchors.get(Anchor::LeftKnee),
        anchors.get(Anchor::RightKnee),
    ) {
        (Some(left), Some(right)) => {
            let knee_mid = mul(add(uv_point(left), uv_point(right)), 0.5);
            let hip_mid = mul(add(hip_left, hip_right), 0.5);
            normalize(sub(knee_mid, hip_mid)).unwrap_or(Vec2::new(0.0, 1.0))
        }
        _ => Vec2::new(0.0, 1.0),
    };
    let up_axis = Vec2::new(-down_axis.x, -down_axis.y);
    let right_px = axis_pixels(grid, right_axis);
    let down_px = axis_pixels(grid, down_axis);

    let gluteal_center = |hip: Vec2| add(hip, mul(down_axis, BUTTOCK_GLUTEAL_DROP * hip_width_px / down_px));
    let left_center = gluteal_center(hip_left);
    let right_center = gluteal_center(hip_right);
    let half = BUTTOCK_LOBE_HALF_HEIGHT * hip_width_px / down_px;

    let edge_point = |center: Vec2, from_right: bool| -> Option<Vec2> {
        let row = grid.row_of(center.y);
        let (left, right) = grid.row_span(row)?;
        let column = if from_right { right } else { left };
        Some(Vec2::new(grid.col_uv(column), grid.row_uv(row)))
    };
    let lateral_fallback = BUTTOCK_LATERAL_FALLBACK * hip_width_px / right_px;
    let left_outer = edge_point(left_center, false)
        .filter(|point| point.x < left_center.x)
        .unwrap_or(add(left_center, mul(right_axis, -lateral_fallback)));
    let right_outer = edge_point(right_center, true)
        .filter(|point| point.x > right_center.x)
        .unwrap_or(add(right_center, mul(right_axis, lateral_fallback)));

    let source_points = [
        left_outer,
        add(left_center, mul(up_axis, half)),
        add(left_center, mul(down_axis, half)),
        add(right_center, mul(up_axis, half)),
        add(right_center, mul(down_axis, half)),
        right_outer,
    ];

    let (sign, amount) = if amp == 0.0 {
        (0.0, 0.0)
    } else {
        (if amp > 0.0 { 1.0 } else { -1.0 }, amp.abs().sqrt())
    };
    let out = BUTTOCK_OUT * hip_width_px * sign / right_px;
    let lift = BUTTOCK_LIFT * hip_width_px * sign / down_px;
    let upper = BUTTOCK_UPPER * hip_width_px * sign / down_px;
    let lower = BUTTOCK_LOWER * hip_width_px * sign / down_px;
    let destination_points = [
        add(
            add(left_outer, mul(right_axis, -out)),
            mul(up_axis, lift),
        ),
        add(source_points[1], mul(up_axis, upper)),
        add(source_points[2], mul(down_axis, lower)),
        add(source_points[3], mul(up_axis, upper)),
        add(source_points[4], mul(down_axis, lower)),
        add(add(right_outer, mul(right_axis, out)), mul(up_axis, lift)),
    ];

    Ok(BodyGeometryInput::SixPoint(SixPointWarp {
        source_points,
        destination_points,
        degrees: [1.0; 6],
        amount,
        inverse_aspect: grid.width() as f32 / grid.height() as f32,
        valid_point_count: 6,
    }))
}

// ---------------------------------------------------------------------------
// PF405 瘦全身
// ---------------------------------------------------------------------------

/// PF405：全身水平挤压 UV 场 + 保护极性掩码。
///
/// `source_uv` 是恒等像素中心，`target_uv` 是满强度取样位置：
/// `u' = u + gain * amp * (u - axis_row) * band(v)`。取样向外 ⇒ 内容向轴收拢 ⇒
/// 人体变窄；`amp < 0` 时反向（显宽）。`axis_row` 是该行掩码加权质心
/// （拿不到时退到全图质心）；`band` 从肩线（缺锚点时人体高度 20%）起效，
/// 避免把头颈一起压窄。
///
/// 保护掩码 = `1 - 掩码 × 边框羽化 × band`，经 [`protect_mask_u8`] 翻成
/// 「255 = 保护」；内核按 `degree = 1 - mask.r` 取样，所以人体内部才动。
/// 本行**只发这一段 BodyRefine 场**，不重复发第二段 warp。
fn body_slim(
    grid: &MaskGrid<'_>,
    degree: f32,
    anchors: &AnchorSet,
) -> Result<BodyGeometryInput, Refusal> {
    let amp = normalized_degree(405, degree)?;
    let (band_start, band_end) = slim_band(grid, anchors);
    let axis_fallback = grid.global_center();
    let count = grid.pixel_count();
    let mut source_uv = Vec::with_capacity(count);
    let mut target_uv = Vec::with_capacity(count);
    let mut mask_r = Vec::with_capacity(count);
    let mut allow_row = vec![0.0f32; grid.width()];

    for y in 0..grid.height() {
        let v = grid.row_uv(y);
        let axis = grid.row_center(y).unwrap_or(axis_fallback);
        let band = ramp(v, band_start, band_end);
        for (x, allow) in allow_row.iter_mut().enumerate() {
            let u = grid.col_uv(x);
            *allow = (grid.at(x, y) * border_fade(u, v, BORDER_MARGIN) * band).clamp(0.0, 1.0);
            let delta = BODY_SLIM_GAIN * amp * (u - axis) * band;
            source_uv.push(Vec2::new(u, v));
            target_uv.push(Vec2::new((u + delta).clamp(0.0, 1.0), v));
        }
        mask_r.extend(protect_mask_u8(&allow_row)?);
    }

    Ok(BodyGeometryInput::BodySlim {
        mask_r,
        source_uv,
        target_uv,
    })
}

/// 全身挤压的竖直起效带：肩线（或人体高度 `HEAD_GUARD`）到人体下缘。
fn slim_band(grid: &MaskGrid<'_>, anchors: &AnchorSet) -> (f32, f32) {
    let top = grid.row_uv(grid.subject_top());
    let bottom = grid.row_uv(grid.subject_bottom());
    let guard = top + (bottom - top) * HEAD_GUARD;
    let shoulder = pair_v(anchors, Anchor::LeftShoulder, Anchor::RightShoulder)
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(guard);
    let start = shoulder.max(guard).clamp(top, (bottom - MIN_SEGMENT).max(top));
    (start, bottom)
}

// ---------------------------------------------------------------------------
// PF407 / PF30094 / PF30095 拉腿
// ---------------------------------------------------------------------------

/// PF407 / PF30094 / PF30095：五列 (目标 V, 源 V) 的分段线性拉长映射。
///
/// 语义按转置轴：`clip_x` = 目标 clip Y，`sample_u` = 源 V，
/// 恒等 ⇔ `sample_u == (clip_x + 1) / 2`（见模块文档）。
///
/// 分段：
/// - `0.0 → 段顶` 恒等（裤腰及以上不动）；
/// - `段顶 → 分割点 → 段底` 内 `src = dst - drop`，`drop` 在分割点取
///   `stretch × w`、在段底取 `stretch`；
/// - `段底 → 1.0` 回到恒等（下缘吸收位移）。
///
/// 分割点与权重：PF407 用膝盖 + vendor [`LEG_FACTORS`] 的大腿/小腿权重；
/// PF30094 用大腿段中点、PF30095 用小腿段中点（各 0.5）。
/// 拉伸量被夹在 `0.8 × 分段长度` 内，保证 `src` 关于 `dst` 单调（不自交）。
/// 段端点夹到掩码的人体竖直范围内，不会映射到 owner 之外。
fn leg_lengthen(
    grid: &MaskGrid<'_>,
    pf: u32,
    degree: f32,
    anchors: &AnchorSet,
) -> Result<BodyGeometryInput, Refusal> {
    let amp = normalized_degree(pf, degree)?.abs();
    let top = grid.row_uv(grid.subject_top());
    let bottom = grid.row_uv(grid.subject_bottom());
    let hip = hip_line_v(anchors, grid);
    let knee = pair_v(anchors, Anchor::LeftKnee, Anchor::RightKnee)
        .map(|v| v.clamp(0.0, 1.0));
    let ankle = pair_v(anchors, Anchor::LeftAnkle, Anchor::RightAnkle)
        .map(|v| v.clamp(0.0, 1.0))
        .unwrap_or(bottom);

    let (raw_start, raw_end, split, weight) = match pf {
        407 => {
            let knee = knee.unwrap_or(0.5 * (hip + ankle));
            (hip, ankle, knee, leg_split_weight())
        }
        30094 => {
            let knee = knee.ok_or(Refusal::MissingInput(Anchor::LeftKnee.missing_label()))?;
            (hip, knee, 0.5 * (hip + knee), 0.5)
        }
        _ => {
            let knee = knee.ok_or(Refusal::MissingInput(Anchor::LeftKnee.missing_label()))?;
            (knee, ankle, 0.5 * (knee + ankle), 0.5)
        }
    };

    let start = raw_start.clamp(top, bottom);
    let end = raw_end.clamp(top, bottom);
    if !(end > start + MIN_SEGMENT) {
        return Err(Refusal::ValueOutOfRange("body fields leg segment"));
    }
    let split = split.clamp(start + MIN_SPLIT, end - MIN_SPLIT);
    let weight = weight.clamp(0.05, 0.95);
    let limit_before = 0.8 * (split - start) / weight;
    let limit_after = 0.8 * (end - split) / (1.0 - weight);
    let stretch = (amp * LEG_STRETCH * (end - start))
        .min(limit_before)
        .min(limit_after)
        .max(0.0);
    let drop = stretch * weight;

    // (目标 V, 源 V)：严格递增，端点为画面顶/底。
    let nodes = [
        (0.0f32, 0.0f32),
        (start, start),
        (split, split - drop),
        (end, end - stretch),
        (1.0, 1.0),
    ];
    let mut columns = [LegColumn {
        sample_u: 0.0,
        clip_x: -1.0,
    }; 5];
    for (index, &(target_v, source_v)) in nodes.iter().enumerate() {
        columns[index] = LegColumn {
            sample_u: source_v.clamp(0.0, 1.0),
            clip_x: (2.0 * target_v - 1.0).clamp(-1.0, 1.0),
        };
    }
    Ok(BodyGeometryInput::Leg {
        columns,
        catmull_size: None,
    })
}

/// PF407 的膝盖分割权重，直接取 vendor `LengthenLegRefine` 的大腿/小腿系数。
fn leg_split_weight() -> f32 {
    let thigh = LEG_FACTORS.thigh.max(0.0);
    let shank = LEG_FACTORS.shank.max(0.0);
    let total = thigh + shank;
    if total > 0.0 && total.is_finite() {
        (thigh / total).clamp(0.1, 0.9)
    } else {
        0.5
    }
}

// ---------------------------------------------------------------------------
// PF408 增高
// ---------------------------------------------------------------------------

/// PF408：胯线到人体下缘的竖直拉伸全帧 UV 场。
///
/// `source_uv` 是取样位置（`height_increase` 的 `hT` 语义，逐像素），
/// `destination_uv` 是恒等像素中心（内核只做长度校验）。
/// 画面顶与胯线恒等，胯线以下向上取样（内容下移 = 变高），
/// 到人体下缘吸满，再在画面底边前回到恒等。`defines` 用 `plain()`：
/// 边界保留靠场的恒等端点，不叠加 `TS_CLAMP_EDGE` 的二次衰减。
fn height_increase(
    grid: &MaskGrid<'_>,
    degree: f32,
    anchors: &AnchorSet,
) -> Result<BodyGeometryInput, Refusal> {
    let amp = normalized_degree(408, degree)?.clamp(0.0, 1.0);
    let count = grid.pixel_count();
    let start = hip_line_v(anchors, grid).clamp(0.0, 1.0);
    let bottom = grid.row_uv(grid.subject_bottom());
    let upper = 1.0 - MIN_ABSORB;
    if amp != 0.0 && start + MIN_SEGMENT > upper {
        return Err(Refusal::ValueOutOfRange("body height leg span"));
    }
    let end = bottom.clamp((start + MIN_SEGMENT).min(upper), upper);
    let stretch = amp * HEIGHT_STRETCH * (end - start).max(0.0);

    let mut source_uv = Vec::with_capacity(count);
    let mut destination_uv = Vec::with_capacity(count);
    for y in 0..grid.height() {
        let v = grid.row_uv(y);
        let delta = stretch_delta(v, start, end, 1.0, stretch);
        for x in 0..grid.width() {
            let u = grid.col_uv(x);
            source_uv.push(Vec2::new(u, (v + delta).clamp(0.0, 1.0)));
            destination_uv.push(Vec2::new(u, v));
        }
    }

    Ok(BodyGeometryInput::Height {
        source_uv,
        destination_uv,
        defines: PassDefines::plain(),
    })
}

// ---------------------------------------------------------------------------
// PF416 拉腰
// ---------------------------------------------------------------------------

/// PF416：腰带段 `WAIST_BAND` 内的竖直拉伸 + 保护极性掩码。
///
/// `warped_uv` 是满强度取样位置：腰带顶恒等，到腰带底向上取样
/// `WAIST_STRETCH × band × amp`（内容下移 = 腰变长），再在人体下缘吸回恒等
/// （吸收区不越出 owner）。`mask_r = 1 - 掩码 × 边框羽化`，内核按
/// `degree = 1 - mask.r` 取样，所以只有人体内部动。
fn waist_lengthen(
    grid: &MaskGrid<'_>,
    degree: f32,
    anchors: &AnchorSet,
) -> Result<BodyGeometryInput, Refusal> {
    let amp = normalized_degree(416, degree)?;
    let frame = torso_frame(anchors)?;
    let first = frame.to_uv(WAIST_BAND.t0, 0.0)?;
    let second = frame.to_uv(WAIST_BAND.t1, 0.0)?;
    let (v0, v1) = if first.y <= second.y {
        (first.y, second.y)
    } else {
        (second.y, first.y)
    };
    let start = v0.clamp(0.0, 1.0);
    let end = v1.clamp(0.0, 1.0);
    if !(end > start + MIN_SEGMENT) {
        return Err(Refusal::ValueOutOfRange("body fields waist band"));
    }
    let absorb = grid
        .row_uv(grid.subject_bottom())
        .max(end + MIN_ABSORB)
        .min(1.0);
    let stretch = amp * WAIST_STRETCH * (end - start);

    let count = grid.pixel_count();
    let mut warped_uv = Vec::with_capacity(count);
    let mut mask_r = Vec::with_capacity(count);
    let mut allow_row = vec![0.0f32; grid.width()];
    for y in 0..grid.height() {
        let v = grid.row_uv(y);
        let delta = stretch_delta(v, start, end, absorb, stretch);
        for (x, allow) in allow_row.iter_mut().enumerate() {
            let u = grid.col_uv(x);
            *allow = grid.at(x, y) * border_fade(u, v, BORDER_MARGIN);
            warped_uv.push(Vec2::new(u, (v + delta).clamp(0.0, 1.0)));
        }
        mask_r.extend(protect_plane_f32(&allow_row)?);
    }

    Ok(BodyGeometryInput::Waist { warped_uv, mask_r })
}

// ---------------------------------------------------------------------------
// PF38556 丰胸
// ---------------------------------------------------------------------------

/// PF38556：胸带内的两叶径向刷位。
///
/// 叶中心取胸带中段行的**掩码跨度** 30% / 70% 处，半径 = 半跨度 ×
/// `BREAST_RADIUS_RATIO`；方向取画面向上（`(0, -1)`，内核 `push_left == false`
/// 时取样向下 ⇒ 内容上提）。`pressure` 是像素位移 = `amp × BREAST_LIFT_RATIO × 半径`，
/// 所以 `degree01` 同时作为门控与幅度。
///
/// `density` 是沿半径的 Hann 剖面（`(N, 1)` 纹理：内核按 `(r²/R², 0.5)` 取样）。
/// `freeze_mask` 用 [`protect_mask_u8`] 直接把**正向**掩码翻成「255 = 冻结」，
/// 内核 `mask = 1 - freeze × warp_direction` 于是正好在人体内为 1。
fn breast_lift(
    grid: &MaskGrid<'_>,
    degree: f32,
    anchors: &AnchorSet,
) -> Result<BodyGeometryInput, Refusal> {
    let amp = normalized_degree(38556, degree)?.abs().clamp(0.0, 1.0);
    let frame = torso_frame(anchors)?;
    let first = frame.to_uv(BREAST_BAND.t0, 0.0)?;
    let second = frame.to_uv(BREAST_BAND.t1, 0.0)?;
    let (v0, v1) = if first.y <= second.y {
        (first.y, second.y)
    } else {
        (second.y, first.y)
    };
    let row_lo = grid.row_of(v0.clamp(0.0, 1.0));
    let row_hi = grid.row_of(v1.clamp(0.0, 1.0));
    let mid_row = (row_lo + row_hi) / 2;
    let (left_px, right_px) = grid
        .row_span(mid_row)
        .ok_or(Refusal::MissingInput("body fields subject mask"))?;
    let span = (right_px - left_px) as f32;
    if !(span > 2.0) {
        return Err(Refusal::ValueOutOfRange("body fields breast band"));
    }
    let row_px = mid_row as f32 + 0.5;
    let radius_px = (BREAST_RADIUS_RATIO * 0.5 * span).max(1.0);
    let lift_px = amp * BREAST_LIFT_RATIO * radius_px;
    let brushes = vec![
        Brush {
            position: Vec2::new(left_px as f32 + 0.30 * span, row_px),
            direction: Vec2::new(0.0, -1.0),
            radius: radius_px,
            pressure: lift_px,
        },
        Brush {
            position: Vec2::new(left_px as f32 + 0.70 * span, row_px),
            direction: Vec2::new(0.0, -1.0),
            radius: radius_px,
            pressure: lift_px,
        },
    ];
    let density = (0..DENSITY_SAMPLES)
        .map(|index| {
            let q = index as f32 / (DENSITY_SAMPLES - 1) as f32;
            0.5 + 0.5 * (std::f32::consts::PI * q).cos()
        })
        .collect::<Vec<f32>>();

    let count = grid.pixel_count();
    let mut freeze_mask = Vec::with_capacity(count);
    let mut row = vec![0.0f32; grid.width()];
    for y in 0..grid.height() {
        for (x, value) in row.iter_mut().enumerate() {
            *value = grid.at(x, y);
        }
        freeze_mask.extend(protect_mask_u8(&row)?);
    }

    Ok(BodyGeometryInput::BreastLift {
        degree01: amp,
        brushes,
        density,
        density_size: (DENSITY_SAMPLES as u32, 1),
        freeze_mask,
        warp_direction: 1.0,
        edge_margin: BORDER_MARGIN,
        push_left: false,
    })
}

// ---------------------------------------------------------------------------
// PF90041 AI 美体
// ---------------------------------------------------------------------------

/// PF90041：与 PF405 同源的水平挤压，打包成 16-bit 位移图。
///
/// `mask` 保持**正向**（0..255）：内核复合式 `effect * mask + original * (1 - mask)`，
/// 所以背景不被拖动。位移图按 /8 降采样（8..192 格），
/// 打包 `packed = 32768 + shift_px * SHIFT_RATIO`，`ratio = SHIFT_RATIO`
/// 对应内核的 `((r * 256 + g) - 32768) / ratio / shiftWidth`。
/// `degree` 归一化为有符号幅度：正 = 收窄，负 = 显宽。
fn body_shift(
    grid: &MaskGrid<'_>,
    degree: f32,
    anchors: &AnchorSet,
) -> Result<BodyGeometryInput, Refusal> {
    let amp = normalized_degree_90041(degree)?;
    let (band_start, band_end) = slim_band(grid, anchors);
    let axis_fallback = grid.global_center();
    let map_w = (grid.width() / 8).clamp(8, 192);
    let map_h = (grid.height() / 8).clamp(8, 192);
    let mut rgba8 = Vec::with_capacity(map_w * map_h * 4);
    for my in 0..map_h {
        let v = (my as f32 + 0.5) / map_h as f32;
        let axis = grid.row_center(grid.row_of(v)).unwrap_or(axis_fallback);
        let band = ramp(v, band_start, band_end);
        for mx in 0..map_w {
            let u = (mx as f32 + 0.5) / map_w as f32;
            let shift_px = BODY_SHIFT_GAIN * amp * (u - axis) * band * grid.width() as f32;
            let (high_x, low_x) = pack_shift(shift_px);
            let (high_y, low_y) = pack_shift(0.0);
            rgba8.extend_from_slice(&[high_x, low_x, high_y, low_y]);
        }
    }

    let count = grid.pixel_count();
    let mut mask = Vec::with_capacity(count);
    let mut inverted = vec![0.0f32; grid.width()];
    for y in 0..grid.height() {
        for (x, value) in inverted.iter_mut().enumerate() {
            *value = 1.0 - grid.at(x, y);
        }
        // protect_mask_u8 输出 255 * (1 - w)；w = 1 - m 时正好是正向掩码 255 * m。
        mask.extend(protect_mask_u8(&inverted)?);
    }

    Ok(BodyGeometryInput::Shift(BodyShiftInput {
        mask,
        map_size: (map_w as u32, map_h as u32),
        rgba8,
        ratio: SHIFT_RATIO,
        shift_size: (grid.width() as u32, grid.height() as u32),
    }))
}

/// 把一个像素位移打包成内核期望的 `(高字节, 低字节)`。
fn pack_shift(shift_px: f32) -> (u8, u8) {
    let value = if shift_px.is_finite() {
        shift_px
    } else {
        0.0
    };
    let packed = (32768.0 + value * SHIFT_RATIO)
        .round_ties_even()
        .clamp(0.0, 65535.0) as u32;
    ((packed >> 8) as u8, (packed & 0xFF) as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_portrait::kernels::body_pose_adapter::{
        adopt, ModelLayout, ObservedKeypoint,
    };

    const SIZE: u32 = 32;

    fn halpe_points() -> Vec<ObservedKeypoint> {
        let mut points = vec![ObservedKeypoint::new(0.5, 0.5, 0.9); 26];
        points[0] = ObservedKeypoint::new(0.50, 0.09, 0.9); // 鼻
        points[3] = ObservedKeypoint::new(0.46, 0.10, 0.9); // 左耳
        points[4] = ObservedKeypoint::new(0.54, 0.10, 0.9); // 右耳
        points[5] = ObservedKeypoint::new(0.40, 0.22, 0.9); // 左肩
        points[6] = ObservedKeypoint::new(0.60, 0.22, 0.9); // 右肩
        points[7] = ObservedKeypoint::new(0.32, 0.38, 0.9); // 左肘
        points[8] = ObservedKeypoint::new(0.68, 0.38, 0.9); // 右肘
        points[9] = ObservedKeypoint::new(0.26, 0.52, 0.9); // 左腕
        points[10] = ObservedKeypoint::new(0.74, 0.52, 0.9); // 右腕
        points[11] = ObservedKeypoint::new(0.42, 0.55, 0.9); // 左髋
        points[12] = ObservedKeypoint::new(0.58, 0.55, 0.9); // 右髋
        points[13] = ObservedKeypoint::new(0.44, 0.75, 0.9); // 左膝
        points[14] = ObservedKeypoint::new(0.56, 0.75, 0.9); // 右膝
        points[15] = ObservedKeypoint::new(0.45, 0.92, 0.9); // 左踝
        points[16] = ObservedKeypoint::new(0.55, 0.92, 0.9); // 右踝
        points[17] = ObservedKeypoint::new(0.50, 0.03, 0.9); // 头顶点
        points[18] = ObservedKeypoint::new(0.50, 0.20, 0.9); // 颈根
        points[19] = ObservedKeypoint::new(0.50, 0.55, 0.9); // 中髋
        points[20] = ObservedKeypoint::new(0.45, 0.97, 0.9); // 左大趾
        points[21] = ObservedKeypoint::new(0.55, 0.97, 0.9); // 右大趾
        points[24] = ObservedKeypoint::new(0.45, 0.95, 0.9); // 左脚跟
        points[25] = ObservedKeypoint::new(0.55, 0.95, 0.9); // 右脚跟
        points
    }

    fn anchors() -> AnchorSet {
        adopt(ModelLayout::Halpe26, &halpe_points(), 0.3).expect("halpe-26 anchors")
    }

    /// 竖直站姿的人体软掩码。
    fn mask() -> Vec<f32> {
        let size = SIZE as usize;
        let mut mask = vec![0.0f32; size * size];
        for y in 0..size {
            for x in 0..size {
                let body = (4..28).contains(&x) && (1..31).contains(&y);
                if body {
                    mask[y * size + x] = 1.0;
                }
            }
        }
        mask
    }

    fn identity_index(pf: u32, degree: f32) -> BodyGeometryInput {
        generate(pf, degree, SIZE, SIZE, &anchors(), &mask()).expect("geometry")
    }

    fn identity_field(input: &BodyGeometryInput) -> bool {
        match input {
            BodyGeometryInput::SixPoint(warp) => {
                warp.destination_points == warp.source_points && warp.amount == 0.0
            }
            BodyGeometryInput::BodySlim {
                source_uv,
                target_uv,
                ..
            } => source_uv == target_uv,
            BodyGeometryInput::Leg { columns, .. } => columns
                .iter()
                .all(|column| (column.sample_u - (column.clip_x + 1.0) * 0.5).abs() <= 1e-6),
            BodyGeometryInput::Height {
                source_uv,
                destination_uv,
                ..
            } => source_uv == destination_uv,
            BodyGeometryInput::Waist { warped_uv, .. } => {
                let size = SIZE as usize;
                warped_uv.iter().enumerate().all(|(index, point)| {
                    let (x, y) = (index % size, index / size);
                    (point.x - (x as f32 + 0.5) / SIZE as f32).abs() <= 1e-6
                        && (point.y - (y as f32 + 0.5) / SIZE as f32).abs() <= 1e-6
                })
            }
            BodyGeometryInput::BreastLift { degree01, .. } => *degree01 == 0.0,
            BodyGeometryInput::Shift(shift) => shift
                .rgba8
                .chunks_exact(4)
                .all(|pixel| pixel[0] == 128 && pixel[1] == 0 && pixel[2] == 128 && pixel[3] == 0),
            BodyGeometryInput::Mesh { .. } => false,
        }
    }

    #[test]
    fn neutral_degree_is_an_identity_ticket() {
        for pf in [400u32, 404, 405, 407, 408, 416, 38556, 90041, 30094, 30095] {
            let ticket = identity_index(pf, 0.0);
            assert!(identity_field(&ticket), "PF{pf} 中性票据必须恒等");
        }
    }

    #[test]
    fn non_neutral_degree_changes_the_geometry() {
        let cases = [
            (400u32, 0.5f32),
            (404, 0.5),
            (405, 2.0),
            (407, 0.255),
            (408, 1.0),
            (416, 1.0),
            (38556, 1.0),
            (90041, 1.0),
            (30094, 1.25),
            (30095, 1.0),
        ];
        for (pf, degree) in cases {
            let ticket = identity_index(pf, degree);
            assert!(!identity_field(&ticket), "PF{pf} 非中性必须改变几何");
        }
    }

    #[test]
    fn leg_columns_are_strictly_increasing_and_keep_the_waist() {
        for pf in [407u32, 30094, 30095] {
            let BodyGeometryInput::Leg { columns, catmull_size } = identity_index(pf, 0.255) else {
                unreachable!()
            };
            assert_eq!(catmull_size, None);
            for pair in columns.windows(2) {
                assert!(pair[1].clip_x > pair[0].clip_x, "PF{pf} clip 必须递增");
                assert!(pair[1].sample_u >= pair[0].sample_u, "PF{pf} 源 V 必须单调");
            }
            // 段顶（含裤腰）落在恒等线上：0 → 0、1 → 1 两端也不例外。
            assert!((columns[0].sample_u - (columns[0].clip_x + 1.0) * 0.5).abs() <= 1e-6);
            assert!((columns[4].sample_u - (columns[4].clip_x + 1.0) * 0.5).abs() <= 1e-6);
        }
    }

    #[test]
    fn mask_contract_is_enforced() {
        let anchors = anchors();
        assert_eq!(
            generate(405, 2.0, SIZE, SIZE, &anchors, &vec![1.0f32; 4]),
            Err(Refusal::ShapeMismatch("body fields mask"))
        );
        let mut bad = mask();
        bad[17] = f32::NAN;
        assert_eq!(
            generate(405, 2.0, SIZE, SIZE, &anchors, &bad),
            Err(Refusal::ValueOutOfRange("body fields mask"))
        );
        assert_eq!(
            generate(408, 1.0, SIZE, SIZE, &anchors, &vec![0.0f32; (SIZE * SIZE) as usize]),
            Err(Refusal::MissingInput("body fields subject mask"))
        );
        assert_eq!(
            generate(400, f32::NAN, SIZE, SIZE, &anchors, &mask()),
            Err(Refusal::ValueOutOfRange("body fields degree"))
        );
        assert_eq!(
            generate(401, 0.5, SIZE, SIZE, &anchors, &mask()),
            Err(Refusal::UnknownEffect(401))
        );
    }

    #[test]
    fn buttock_six_points_stay_inside_the_frame_and_follow_the_mask() {
        let BodyGeometryInput::SixPoint(warp) = identity_index(404, 0.5) else {
            unreachable!()
        };
        for point in warp.source_points {
            assert!((0.0..=1.0).contains(&point.x) && (0.0..=1.0).contains(&point.y));
        }
        assert_eq!(warp.valid_point_count, 6);
        // 外侧点取掩码轮廓（32 宽的人体从第 4 列到第 27 列）。
        assert!(warp.source_points[0].x < 0.42 + 1e-6);
        assert!(warp.source_points[5].x > 0.58 - 1e-6);
    }

    #[test]
    fn body_shift_keeps_the_positive_mask_and_zero_shift_when_neutral() {
        let BodyGeometryInput::Shift(shift) = identity_index(90041, 0.0) else {
            unreachable!()
        };
        assert_eq!(shift.mask.len(), (SIZE * SIZE) as usize);
        assert!(shift.mask.iter().any(|value| *value == 255));
        assert_eq!(shift.shift_size, (SIZE, SIZE));
        assert!(shift.ratio > 0.0);
        let BodyGeometryInput::Shift(moved) = identity_index(90041, 1.0) else {
            unreachable!()
        };
        assert_ne!(moved.rgba8, shift.rgba8);
    }
}
