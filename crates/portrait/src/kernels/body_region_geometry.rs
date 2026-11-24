//! 躯干区域纯几何（UV 空间）。
//!
//! 本模块只做纯 UV 几何：建立躯干坐标系、计算高度带权重、计算管状域权重、
//! 按中线切左右，并把软权重翻成内核要的保护极性。它**不读取 `Image`、
//! 不做形态学、不消费分割类别**，因此可以在没有像素、没有模型权重的环境里
//! 单独测试。
//!
//! # UV 约定
//!
//! 与 [`crate::kernels::body_geometry`] 完全一致：像素 `(x, y)` 的中心是
//! `((x + 0.5) / W, (y + 0.5) / H)`，反向采样为 `x = u * W - 0.5`。
//! 本模块所有 `Vec2` 都是 `[0,1]` 归一化连续坐标（像素中心归一化），
//! **不是像素索引**。关键点适配器必须先把模型坐标修成这一约定再进来
//! （见 `03_algorithms/reports/BODY-POSE-ONNX-CONTRACT.md` §5.3 的半像素修正）。
//!
//! # 阈值来源
//!
//! `WAIST_BAND` / `BREAST_BAND` / `BUTTOCK_BAND` 取自
//! `03_algorithms/reports/BODY-POSE-REGION-MAP.tsv` 的 `height_band_t` 列，
//! `NECK_BAND` 是工程默认值。所有阈值与羽化宽度均标 `[D]`，
//! 来自 `90_work/lanes/portrait-parse/body-onnx-contract/07-seg-class-to-region-mask.md`
//! §3–§4，**需要实拍标定**，不得当作原厂参数。
//!
//! # 边界
//!
//! 这里只输出 `[0,1]` 的软权重（1 = 属于该区域）。真正给 `body_geometry`
//! 内核时要做极性翻转（`1 = 保护`），由 [`protect_mask_u8`] 与
//! [`protect_plane_f32`] 负责，保证与 `body_geometry.rs:540`、
//! `hair_body.rs:1159` 的消费者语义一致。

use crate::kernels::body_geometry::Vec2;
use crate::Refusal;

/// 退化判定阈值：肩宽、躯干高、胯宽都要求大于该值（UV 单位）。
const MIN_EXTENT: f32 = 1e-4;

fn finite(p: Vec2) -> bool {
    p.x.is_finite() && p.y.is_finite()
}

fn sub(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::new(a.x - b.x, a.y - b.y)
}

fn add(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::new(a.x + b.x, a.y + b.y)
}

fn scale(a: Vec2, k: f32) -> Vec2 {
    Vec2::new(a.x * k, a.y * k)
}

fn dot(a: Vec2, b: Vec2) -> f32 {
    a.x * b.x + a.y * b.y
}

fn length(a: Vec2) -> f32 {
    dot(a, a).sqrt()
}

/// 四个点锚点建立的躯干坐标系。
///
/// 坐标全部是 `[0,1]` UV（像素中心归一化）。四个点来自语义关键点
/// `LeftShoulder / RightShoulder / LeftHip / RightHip`；左右以**人物自身**
/// 为准，因此背面/侧身时 `right` 仍指向人物右侧，`u` 的符号不会翻转。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorsoFrame {
    pub shoulder_left: Vec2,
    pub shoulder_right: Vec2,
    pub hip_left: Vec2,
    pub hip_right: Vec2,
}

/// 由 [`TorsoFrame`] 导出的正交基与尺度。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TorsoAxes {
    /// 肩线中点（t=0,u=0）。
    pub origin: Vec2,
    /// 肩线中点到胯线中点的单位向量（t 增大方向，通常约 (0,1)）。
    pub down: Vec2,
    /// 躯干右方向单位向量（u 增大方向）。
    pub right: Vec2,
    /// 肩线中点到胯线中点的长度（UV 单位）。
    pub torso_height: f32,
    /// 肩宽（UV 单位）。
    pub shoulder_width: f32,
}

impl TorsoAxes {
    /// `down · right`。0 表示两轴严格正交；肩线倾斜、胯线与肩线不平行时非 0。
    ///
    /// 契约要求**不做正交化**（真实人体肩线可能倾斜），本函数只是把
    /// 倾斜程度报告给调用方，供其决定是否拒绝或降级。
    pub fn orthogonality(&self) -> f32 {
        dot(self.down, self.right)
    }
}

impl TorsoFrame {
    /// 校验：四点有限；肩宽 > 1e-4；躯干高 > 1e-4；左右肩/胯不能重合。
    ///
    /// 任一条件不满足 → `Err(Refusal::ValueOutOfRange("torso frame"))`。
    pub fn new(
        shoulder_left: Vec2,
        shoulder_right: Vec2,
        hip_left: Vec2,
        hip_right: Vec2,
    ) -> Result<Self, Refusal> {
        if !finite(shoulder_left)
            || !finite(shoulder_right)
            || !finite(hip_left)
            || !finite(hip_right)
        {
            return Err(Refusal::ValueOutOfRange("torso frame"));
        }
        let shoulder_width = length(sub(shoulder_right, shoulder_left));
        let hip_width = length(sub(hip_right, hip_left));
        let shoulder_mid = scale(add(shoulder_left, shoulder_right), 0.5);
        let hip_mid = scale(add(hip_left, hip_right), 0.5);
        let torso_height = length(sub(hip_mid, shoulder_mid));
        if !(shoulder_width > MIN_EXTENT)
            || !(hip_width > MIN_EXTENT)
            || !(torso_height > MIN_EXTENT)
        {
            return Err(Refusal::ValueOutOfRange("torso frame"));
        }
        Ok(Self {
            shoulder_left,
            shoulder_right,
            hip_left,
            hip_right,
        })
    }

    /// 计算正交基。字段是 `pub`，调用方可能绕过 [`TorsoFrame::new`] 直接构造，
    /// 因此这里重新校验一遍。
    pub fn axes(&self) -> Result<TorsoAxes, Refusal> {
        if !finite(self.shoulder_left)
            || !finite(self.shoulder_right)
            || !finite(self.hip_left)
            || !finite(self.hip_right)
        {
            return Err(Refusal::ValueOutOfRange("torso frame"));
        }
        let shoulder_mid = scale(add(self.shoulder_left, self.shoulder_right), 0.5);
        let hip_mid = scale(add(self.hip_left, self.hip_right), 0.5);
        let shoulder_vec = sub(self.shoulder_right, self.shoulder_left);
        let hip_vec = sub(self.hip_right, self.hip_left);
        let axis = sub(hip_mid, shoulder_mid);
        let shoulder_width = length(shoulder_vec);
        let hip_width = length(hip_vec);
        let torso_height = length(axis);
        if !(shoulder_width > MIN_EXTENT)
            || !(hip_width > MIN_EXTENT)
            || !(torso_height > MIN_EXTENT)
        {
            return Err(Refusal::ValueOutOfRange("torso frame"));
        }
        Ok(TorsoAxes {
            origin: shoulder_mid,
            down: scale(axis, 1.0 / torso_height),
            right: scale(shoulder_vec, 1.0 / shoulder_width),
            torso_height,
            shoulder_width,
        })
    }

    /// 返回躯干坐标 `(t, u)`：t=0 肩线、t=1 胯线；
    /// u 以半肩宽归一化（-1 左缘、+1 右缘）。
    ///
    /// `t = dot(p - origin, down) / torso_height`，
    /// `u = dot(p - origin, right) / (shoulder_width * 0.5)`。
    pub fn to_torso(&self, p: Vec2) -> Result<(f32, f32), Refusal> {
        if !finite(p) {
            return Err(Refusal::ValueOutOfRange("torso point"));
        }
        let axes = self.axes()?;
        let rel = sub(p, axes.origin);
        let t = dot(rel, axes.down) / axes.torso_height;
        let u = dot(rel, axes.right) / (axes.shoulder_width * 0.5);
        if !t.is_finite() || !u.is_finite() {
            return Err(Refusal::ValueOutOfRange("torso point"));
        }
        Ok((t, u))
    }

    /// 反向：躯干坐标 → UV，与 [`TorsoFrame::to_torso`] 互为**精确逆**。
    ///
    /// 两轴不做正交化（肩线可能倾斜），因此这里求解 2×2 线性系统
    /// `down·(p-origin) = t·torso_height`、`right·(p-origin) = u·0.5·shoulder_width`，
    /// 而不是直接线性组合。正交时（`down·right == 0`）解退化为
    /// `p = origin + down*(t*torso_height) + right*(u*0.5*shoulder_width)`，
    /// 与契约给出的显式公式完全一致；非正交时显式公式与"互逆"不能同时成立，
    /// 本实现选择互逆，倾斜程度由 [`TorsoAxes::orthogonality`] 报告。
    /// 两轴近似平行（`|down·right| ≈ 1`）时无解 → `ValueOutOfRange("torso frame")`。
    pub fn to_uv(&self, t: f32, u: f32) -> Result<Vec2, Refusal> {
        if !t.is_finite() || !u.is_finite() {
            return Err(Refusal::ValueOutOfRange("torso coordinate"));
        }
        let axes = self.axes()?;
        let a = t * axes.torso_height;
        let b = u * 0.5 * axes.shoulder_width;
        let c = axes.orthogonality();
        let det = 1.0 - c * c;
        if !det.is_finite() || det <= 1e-6 {
            return Err(Refusal::ValueOutOfRange("torso frame"));
        }
        let x = (a - b * c) / det;
        let y = (b - a * c) / det;
        let p = add(axes.origin, add(scale(axes.down, x), scale(axes.right, y)));
        if !finite(p) {
            return Err(Refusal::ValueOutOfRange("torso coordinate"));
        }
        Ok(p)
    }
}

/// 一个矩形高度带 `t0..t1`，并限制 `u` 范围 `u0..u1`。
///
/// "不限" 用足够大的区间表达（例如 `u0 = -1.0, u1 = 1.0` 覆盖整个躯干宽度），
/// 不引入 `Option`，保证结构体 `Copy` 且可作 `const`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionBand {
    pub t0: f32,
    pub t1: f32,
    pub u0: f32,
    pub u1: f32,
}

/// 腰带 `t ∈ 0.58..0.78`（`BODY-POSE-REGION-MAP.tsv`，`[D]`）。
pub const WAIST_BAND: RegionBand = RegionBand {
    t0: 0.58,
    t1: 0.78,
    u0: -1.0,
    u1: 1.0,
};

/// 胸带 `t ∈ 0.06..0.34`（`BODY-POSE-REGION-MAP.tsv`，`[D]`）。
pub const BREAST_BAND: RegionBand = RegionBand {
    t0: 0.06,
    t1: 0.34,
    u0: -1.0,
    u1: 1.0,
};

/// 臀带 `t ∈ 0.95..1.22`。注意 `t1 > 1`：臀带跨过胯线向下延伸。
///
/// 契约式 `t_hi = 1.0 + 0.22 * T_leg / T` 依赖膝线；本常量按
/// `T_leg ≈ T` 的工程默认取值，`[D]`，需实拍标定。
pub const BUTTOCK_BAND: RegionBand = RegionBand {
    t0: 0.95,
    t1: 1.22,
    u0: -1.0,
    u1: 1.0,
};

/// 颈带 `t ∈ -0.35..0.02`、`u ∈ -0.5..0.5`（工程默认，`[D]`）。
///
/// 契约里颈区实际是「脸下沿 → 肩线」的未分类带，由脸掩码底沿决定；
/// 本常量给出的是没有脸掩码时的保守几何默认。
pub const NECK_BAND: RegionBand = RegionBand {
    t0: -0.35,
    t1: 0.02,
    u0: -0.5,
    u1: 0.5,
};

impl RegionBand {
    /// `t1 > t0`、`u1 > u0`、全部有限，否则 `ValueOutOfRange("region band")`。
    pub fn new(t0: f32, t1: f32, u0: f32, u1: f32) -> Result<Self, Refusal> {
        if !t0.is_finite() || !t1.is_finite() || !u0.is_finite() || !u1.is_finite() {
            return Err(Refusal::ValueOutOfRange("region band"));
        }
        if !(t1 > t0) || !(u1 > u0) {
            return Err(Refusal::ValueOutOfRange("region band"));
        }
        Ok(Self { t0, t1, u0, u1 })
    }
}

/// 单轴柔边权重：`value` 在 `[lo+feather, hi-feather]` 内为 1，
/// 在 `lo..lo+feather` 与 `hi-feather..hi` 内线性过渡到 0，其余为 0。
///
/// 两段斜坡各自单调、连续；带边严格为 0。
/// 当 `feather` 大于半带宽时，两条斜坡在中间相交，峰值小于 1。
fn axis_weight(value: f32, lo: f32, hi: f32, feather: f32) -> f32 {
    let rise = ((value - lo) / feather).clamp(0.0, 1.0);
    let fall = ((hi - value) / feather).clamp(0.0, 1.0);
    rise.min(fall)
}

fn band_is_valid(band: &RegionBand) -> bool {
    band.t0.is_finite()
        && band.t1.is_finite()
        && band.u0.is_finite()
        && band.u1.is_finite()
        && band.t1 > band.t0
        && band.u1 > band.u0
}

/// 高度带权重（柔边界），返回 `[0,1]`。
///
/// `feather` 是过渡带宽度（UV），必须 > 0 且有限，否则
/// `ValueOutOfRange("band feather")`。t 方向与 u 方向各自做线性斜坡后相乘，
/// 因此带外与带边都严格为 0。`band` 本身非法（非有限 / 空区间）→
/// `ValueOutOfRange("region band")`，防止绕过 [`RegionBand::new`] 的结构体字面量
/// 把 NaN 带进下游。
pub fn band_weight(
    p: Vec2,
    frame: &TorsoFrame,
    band: RegionBand,
    feather: f32,
) -> Result<f32, Refusal> {
    if !feather.is_finite() || feather <= 0.0 {
        return Err(Refusal::ValueOutOfRange("band feather"));
    }
    if !band_is_valid(&band) {
        return Err(Refusal::ValueOutOfRange("region band"));
    }
    let (t, u) = frame.to_torso(p)?;
    let wt = axis_weight(t, band.t0, band.t1, feather);
    let wu = axis_weight(u, band.u0, band.u1, feather);
    Ok((wt * wu).clamp(0.0, 1.0))
}

/// 管状域权重（标准 capsule）：点到线段 `ab` 的距离 <= `radius` 为 1，
/// `radius..radius+feather` 线性降到 0，更远为 0。
///
/// 端点半径为 0 时退化为线段。`radius` 必须有限且 >= 0，
/// `feather` 必须有限且 > 0，否则 `ValueOutOfRange`。
pub fn tube_weight(p: Vec2, a: Vec2, b: Vec2, radius: f32, feather: f32) -> Result<f32, Refusal> {
    if !finite(p) || !finite(a) || !finite(b) {
        return Err(Refusal::ValueOutOfRange("tube"));
    }
    if !radius.is_finite() || radius < 0.0 {
        return Err(Refusal::ValueOutOfRange("tube radius"));
    }
    if !feather.is_finite() || feather <= 0.0 {
        return Err(Refusal::ValueOutOfRange("tube feather"));
    }
    let ab = sub(b, a);
    let len2 = dot(ab, ab);
    // 退化线段（a == b）：按端点距离计算。
    let closest = if len2 <= 1e-30 {
        a
    } else {
        let t = (dot(sub(p, a), ab) / len2).clamp(0.0, 1.0);
        add(a, scale(ab, t))
    };
    let distance = length(sub(p, closest));
    if !distance.is_finite() {
        return Err(Refusal::ValueOutOfRange("tube"));
    }
    let falloff = ((distance - radius) / feather).clamp(0.0, 1.0);
    Ok((1.0 - falloff).clamp(0.0, 1.0))
}

/// 按躯干中线切左右。返回 `(left_weight, right_weight)`，两者之和为 1。
///
/// `u < 0` 偏左，`u > 0` 偏右。`clamp_ratio ∈ [0, 0.5]` 是中线两侧的线性
/// 过渡半宽（以 u 为单位），0 表示硬切；越界值会被夹到该区间。
/// `u` 非有限时返回 `(0.5, 0.5)`，不 panic。
pub fn split_left_right(u: f32, clamp_ratio: f32) -> (f32, f32) {
    if !u.is_finite() {
        return (0.5, 0.5);
    }
    let ratio = if clamp_ratio.is_finite() {
        clamp_ratio.clamp(0.0, 0.5)
    } else {
        0.0
    };
    if ratio <= 0.0 {
        return if u < 0.0 {
            (1.0, 0.0)
        } else if u > 0.0 {
            (0.0, 1.0)
        } else {
            (0.5, 0.5)
        };
    }
    let left = (0.5 - u / (2.0 * ratio)).clamp(0.0, 1.0);
    (left, 1.0 - left)
}

/// 三角/管状域与高度带的交集权重：先算 band，再乘 tube。
pub fn band_and_tube(
    p: Vec2,
    frame: &TorsoFrame,
    band: RegionBand,
    a: Vec2,
    b: Vec2,
    radius: f32,
    feather: f32,
) -> Result<f32, Refusal> {
    let band_w = band_weight(p, frame, band, feather)?;
    let tube_w = tube_weight(p, a, b, radius, feather)?;
    Ok((band_w * tube_w).clamp(0.0, 1.0))
}

/// 左右镜像测试辅助：把点关于 `x = 0.5` 镜像。
///
/// 只用于测试/调试；签名不返回 `Result`，因此调用方必须自行保证 `p` 有限
/// （非有限值原样传播，不在这里制造新的拒绝路径）。
pub fn mirror_x(p: Vec2) -> Vec2 {
    Vec2::new(1.0 - p.x, p.y)
}

/// 融合权重（1 = 允许形变）→ PF405 的 `mask_r`（1 = 保护），长度 = 像素数。
///
/// 值 = `round_ties_even(255 * (1 - w))`，与 `body_geometry::quantize_even`
/// 的 half-even 量化一致（`body_geometry.rs:876-878`）。
/// `w` 必须有限且 ∈ `[0,1]`，否则 `ValueOutOfRange("region weight")`。
pub fn protect_mask_u8(final_weight: &[f32]) -> Result<Vec<u8>, Refusal> {
    final_weight
        .iter()
        .map(|&w| {
            if !w.is_finite() || !(0.0..=1.0).contains(&w) {
                return Err(Refusal::ValueOutOfRange("region weight"));
            }
            let value = (255.0 * (1.0 - w)).round_ties_even().clamp(0.0, 255.0);
            Ok(value as u8)
        })
        .collect()
}

/// 同上，但输出 PF416 的 `mask_r`：`1 - w`，`f32 ∈ [0,1]`。
///
/// 消费者按 `degree = 1.0 - mask_r` 使用（`body_geometry.rs:540`），
/// 所以 `w = 1`（完全允许形变）对应 `mask_r = 0`。
pub fn protect_plane_f32(final_weight: &[f32]) -> Result<Vec<f32>, Refusal> {
    final_weight
        .iter()
        .map(|&w| {
            if !w.is_finite() || !(0.0..=1.0).contains(&w) {
                return Err(Refusal::ValueOutOfRange("region weight"));
            }
            Ok((1.0 - w).clamp(0.0, 1.0))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5
    }

    /// 标准躯干框：肩 y=0.3、胯 y=0.6、肩宽 0.2，胯宽略窄。
    fn standard_frame() -> TorsoFrame {
        TorsoFrame::new(
            Vec2::new(0.4, 0.3),
            Vec2::new(0.6, 0.3),
            Vec2::new(0.42, 0.6),
            Vec2::new(0.58, 0.6),
        )
        .expect("standard frame")
    }

    #[test]
    fn to_torso_and_to_uv_are_inverse() {
        let frame = standard_frame();
        let points = [
            Vec2::new(0.5, 0.3),
            Vec2::new(0.5, 0.45),
            Vec2::new(0.55, 0.52),
            Vec2::new(0.38, 0.61),
            Vec2::new(0.62, 0.33),
        ];
        for p in points {
            let (t, u) = frame.to_torso(p).expect("to_torso");
            let back = frame.to_uv(t, u).expect("to_uv");
            assert!(close(back.x, p.x), "x {:?} -> {:?}", p, back);
            assert!(close(back.y, p.y), "y {:?} -> {:?}", p, back);
        }

        let axes = frame.axes().expect("axes");
        assert!(close(axes.torso_height, 0.3));
        assert!(close(axes.shoulder_width, 0.2));
        assert!(close(axes.orthogonality(), 0.0));

        // 肩线倾斜时两轴不正交，但 to_uv 仍是 to_torso 的精确逆。
        let skewed = TorsoFrame::new(
            Vec2::new(0.4, 0.3),
            Vec2::new(0.6, 0.34),
            Vec2::new(0.44, 0.6),
            Vec2::new(0.56, 0.62),
        )
        .expect("skewed frame");
        assert!(skewed.axes().expect("skewed axes").orthogonality().abs() > 0.1);
        for p in points {
            let (t, u) = skewed.to_torso(p).expect("skewed to_torso");
            let back = skewed.to_uv(t, u).expect("skewed to_uv");
            assert!(close(back.x, p.x), "skewed x {:?} -> {:?}", p, back);
            assert!(close(back.y, p.y), "skewed y {:?} -> {:?}", p, back);
        }
    }

    #[test]
    fn shoulder_and_hip_midlines_are_axis_anchors() {
        let frame = standard_frame();
        let (t, u) = frame.to_torso(Vec2::new(0.5, 0.3)).expect("shoulder mid");
        assert!(close(t, 0.0), "t={t}");
        assert!(close(u, 0.0), "u={u}");

        let (t, u) = frame.to_torso(Vec2::new(0.5, 0.6)).expect("hip mid");
        assert!(close(t, 1.0), "t={t}");
        assert!(close(u, 0.0), "u={u}");

        // u 以半肩宽归一化：右缘 u=1 落在 shoulder_right。
        let (_, u) = frame.to_torso(Vec2::new(0.6, 0.3)).expect("right shoulder");
        assert!(close(u, 1.0), "u={u}");
        let (_, u) = frame.to_torso(Vec2::new(0.4, 0.3)).expect("left shoulder");
        assert!(close(u, -1.0), "u={u}");
    }

    #[test]
    fn waist_band_is_one_inside_and_zero_outside() {
        let frame = standard_frame();
        let feather = 0.05;
        let inside = band_weight(Vec2::new(0.5, 0.3 + 0.68 * 0.3), &frame, WAIST_BAND, feather)
            .expect("waist inside");
        assert!(close(inside, 1.0), "w={inside}");

        let outside = band_weight(Vec2::new(0.5, 0.3 + 0.9 * 0.3), &frame, WAIST_BAND, feather)
            .expect("waist outside");
        assert!(close(outside, 0.0), "w={outside}");

        // t = 0.60 落在 0.58..0.63 过渡带内，线性权重 = 0.4。
        let ramp = band_weight(Vec2::new(0.5, 0.3 + 0.6 * 0.3), &frame, WAIST_BAND, feather)
            .expect("waist ramp");
        assert!(ramp > 0.0 && ramp < 1.0, "w={ramp}");
        assert!(close(ramp, 0.4), "w={ramp}");
    }

    #[test]
    fn breast_and_buttock_bands_have_correct_edges() {
        let frame = standard_frame();
        let feather = 0.05;
        let at = |t: f32| Vec2::new(0.5, 0.3 + t * 0.3);

        // 胸带 0.06..0.34。
        assert!(close(
            band_weight(at(0.2), &frame, BREAST_BAND, feather).expect("breast inside"),
            1.0
        ));
        assert!(close(
            band_weight(at(0.04), &frame, BREAST_BAND, feather).expect("breast below"),
            0.0
        ));
        assert!(close(
            band_weight(at(0.36), &frame, BREAST_BAND, feather).expect("breast above"),
            0.0
        ));

        // 臀带 0.95..1.22，t=1.1 必须仍然有效（t1 > 1）。
        assert!(close(
            band_weight(at(1.1), &frame, BUTTOCK_BAND, feather).expect("buttock inside"),
            1.0
        ));
        assert!(close(
            band_weight(at(0.9), &frame, BUTTOCK_BAND, feather).expect("buttock above"),
            0.0
        ));
        assert!(close(
            band_weight(at(1.24), &frame, BUTTOCK_BAND, feather).expect("buttock below"),
            0.0
        ));
        // 过渡带 1.17..1.22 内的中点。
        let ramp = band_weight(at(1.195), &frame, BUTTOCK_BAND, feather).expect("buttock ramp");
        assert!(ramp > 0.0 && ramp < 1.0, "w={ramp}");
    }

    #[test]
    fn mirrored_frame_keeps_t_and_flips_u() {
        let frame = TorsoFrame::new(
            Vec2::new(0.38, 0.28),
            Vec2::new(0.63, 0.32),
            Vec2::new(0.44, 0.61),
            Vec2::new(0.57, 0.58),
        )
        .expect("asymmetric frame");
        // 镜像四个锚点并交换左右。
        let mirrored = TorsoFrame::new(
            mirror_x(frame.shoulder_right),
            mirror_x(frame.shoulder_left),
            mirror_x(frame.hip_right),
            mirror_x(frame.hip_left),
        )
        .expect("mirrored frame");

        let points = [
            Vec2::new(0.52, 0.4),
            Vec2::new(0.41, 0.55),
            Vec2::new(0.6, 0.33),
        ];
        for p in points {
            let (t, u) = frame.to_torso(p).expect("original");
            let (mt, mu) = mirrored.to_torso(mirror_x(p)).expect("mirrored");
            assert!(close(t, mt), "t {t} vs {mt}");
            assert!(close(u, -mu), "u {u} vs {mu}");
        }
    }

    #[test]
    fn tube_weight_is_a_capsule() {
        let a = Vec2::new(0.4, 0.4);
        let b = Vec2::new(0.6, 0.4);
        let radius = 0.05;
        let feather = 0.02;

        // 线段中点，半径内。
        let inside = tube_weight(Vec2::new(0.5, 0.4), a, b, radius, feather).expect("inside");
        assert!(close(inside, 1.0));

        // 恰好 radius + feather，边界为 0。
        let edge = tube_weight(Vec2::new(0.5, 0.47), a, b, radius, feather).expect("edge");
        assert!(close(edge, 0.0), "w={edge}");

        // 过渡带中点。
        let ramp = tube_weight(Vec2::new(0.5, 0.46), a, b, radius, feather).expect("ramp");
        assert!(close(ramp, 0.5), "w={ramp}");

        // 端点之外按端点距离：到 b 的距离 0.08 落在过渡带。
        let past_end = tube_weight(Vec2::new(0.68, 0.4), a, b, radius, feather).expect("past end");
        assert!(close(past_end, 0.0), "w={past_end}");
        let near_end = tube_weight(Vec2::new(0.66, 0.4), a, b, radius, feather).expect("near end");
        assert!(near_end > 0.0 && near_end < 1.0, "w={near_end}");

        // 退化线段（a == b）按端点距离计算：0.02 < radius，权重 1。
        let degenerate = tube_weight(Vec2::new(0.42, 0.4), a, a, radius, feather).expect("degenerate");
        assert!(close(degenerate, 1.0));
        // 同一退化点，端点距离 0.1 已在半径 + 羽化之外。
        let far = tube_weight(Vec2::new(0.5, 0.4), a, a, radius, feather).expect("degenerate far");
        assert!(close(far, 0.0), "w={far}");
    }

    #[test]
    fn band_and_tube_multiplies_both_weights() {
        let frame = standard_frame();

        // 胸带内 t=0.2，且落在 y=0.36 的细管上 → 1。
        let tube_inside = (Vec2::new(0.3, 0.36), Vec2::new(0.7, 0.36), 0.01, 0.01);
        let inside = band_and_tube(
            Vec2::new(0.5, 0.36),
            &frame,
            BREAST_BAND,
            tube_inside.0,
            tube_inside.1,
            tube_inside.2,
            tube_inside.3,
        )
        .expect("inside");
        assert!(close(inside, 1.0), "w={inside}");

        // 胸带内 t=0.283（band=1），但到管轴距离 0.025 已超过
        // radius + feather = 0.02 → 0。
        let out_tube = band_and_tube(
            Vec2::new(0.5, 0.385),
            &frame,
            BREAST_BAND,
            tube_inside.0,
            tube_inside.1,
            tube_inside.2,
            tube_inside.3,
        )
        .expect("out tube");
        assert!(close(out_tube, 0.0), "w={out_tube}");

        // t=0.5 在胸带外，即使点在粗管中轴上 → 0。
        let wide_tube = (Vec2::new(0.3, 0.45), Vec2::new(0.7, 0.45), 0.05, 0.05);
        let out_band = band_and_tube(
            Vec2::new(0.5, 0.45),
            &frame,
            BREAST_BAND,
            wide_tube.0,
            wide_tube.1,
            wide_tube.2,
            wide_tube.3,
        )
        .expect("out band");
        assert!(close(out_band, 0.0), "w={out_band}");
        assert!(
            tube_weight(Vec2::new(0.5, 0.45), wide_tube.0, wide_tube.1, wide_tube.2, wide_tube.3)
                .expect("tube alone")
                > 0.9,
            "test premise: the wide tube must actually cover the point"
        );
    }

    #[test]
    fn split_left_right_obeys_midline() {
        let (l, r) = split_left_right(-0.5, 0.5);
        assert!(close(l, 1.0) && close(r, 0.0));
        let (l, r) = split_left_right(0.5, 0.5);
        assert!(close(l, 0.0) && close(r, 1.0));
        let (l, r) = split_left_right(0.0, 0.25);
        assert!(close(l, 0.5) && close(r, 0.5));
        // 硬切。
        let (l, r) = split_left_right(-0.01, 0.0);
        assert!(close(l, 1.0) && close(r, 0.0));
        // 非有限 u 不 panic。
        let (l, r) = split_left_right(f32::NAN, 0.5);
        assert!(close(l, 0.5) && close(r, 0.5));
    }

    #[test]
    fn degenerate_inputs_refuse() {
        let zero_width = TorsoFrame::new(
            Vec2::new(0.5, 0.3),
            Vec2::new(0.5, 0.3),
            Vec2::new(0.42, 0.6),
            Vec2::new(0.58, 0.6),
        );
        assert_eq!(
            zero_width,
            Err(Refusal::ValueOutOfRange("torso frame"))
        );

        let zero_height = TorsoFrame::new(
            Vec2::new(0.4, 0.3),
            Vec2::new(0.6, 0.3),
            Vec2::new(0.4, 0.3),
            Vec2::new(0.6, 0.3),
        );
        assert_eq!(
            zero_height,
            Err(Refusal::ValueOutOfRange("torso frame"))
        );

        let nan = TorsoFrame::new(
            Vec2::new(f32::NAN, 0.3),
            Vec2::new(0.6, 0.3),
            Vec2::new(0.42, 0.6),
            Vec2::new(0.58, 0.6),
        );
        assert_eq!(nan, Err(Refusal::ValueOutOfRange("torso frame")));

        // 绕过构造器直接改字段，axes()/to_torso() 仍须拒绝。
        let mut bypass = standard_frame();
        bypass.shoulder_right = Vec2::new(0.4, 0.3);
        assert!(bypass.axes().is_err());
        assert!(bypass.to_torso(Vec2::new(0.5, 0.5)).is_err());
        assert!(bypass.to_uv(0.5, 0.0).is_err());

        // feather = 0 拒绝。
        let frame = standard_frame();
        assert_eq!(
            band_weight(Vec2::new(0.5, 0.5), &frame, WAIST_BAND, 0.0),
            Err(Refusal::ValueOutOfRange("band feather"))
        );
        assert_eq!(
            tube_weight(
                Vec2::new(0.5, 0.5),
                Vec2::new(0.4, 0.5),
                Vec2::new(0.6, 0.5),
                0.05,
                0.0
            ),
            Err(Refusal::ValueOutOfRange("tube feather"))
        );
        assert_eq!(
            tube_weight(
                Vec2::new(0.5, 0.5),
                Vec2::new(0.4, 0.5),
                Vec2::new(0.6, 0.5),
                -0.1,
                0.02
            ),
            Err(Refusal::ValueOutOfRange("tube radius"))
        );
    }

    #[test]
    fn region_band_new_rejects_empty_ranges() {
        assert!(RegionBand::new(0.5, 0.5, -1.0, 1.0).is_err());
        assert!(RegionBand::new(0.6, 0.5, -1.0, 1.0).is_err());
        assert!(RegionBand::new(0.5, 0.6, 0.0, 0.0).is_err());
        assert!(RegionBand::new(0.5, 0.6, 1.0, -1.0).is_err());
        assert!(RegionBand::new(f32::NAN, 0.6, -1.0, 1.0).is_err());
        assert!(RegionBand::new(0.5, 0.6, -1.0, 1.0).is_ok());
    }

    #[test]
    fn protect_mask_u8_flips_polarity() {
        let weights = [0.0f32, 0.5, 1.0];
        let mask = protect_mask_u8(&weights).expect("mask");
        assert_eq!(mask, vec![255u8, 128, 0]);

        // 全 0 → 全 255；全 1 → 全 0。
        assert_eq!(protect_mask_u8(&[0.0; 4]).expect("all open"), vec![255u8; 4]);
        assert_eq!(protect_mask_u8(&[1.0; 4]).expect("all protect"), vec![0u8; 4]);

        assert_eq!(
            protect_mask_u8(&[1.5]),
            Err(Refusal::ValueOutOfRange("region weight"))
        );
        assert_eq!(
            protect_mask_u8(&[f32::NAN]),
            Err(Refusal::ValueOutOfRange("region weight"))
        );
        assert_eq!(
            protect_mask_u8(&[-0.1]),
            Err(Refusal::ValueOutOfRange("region weight"))
        );
    }

    #[test]
    fn protect_plane_f32_flips_polarity() {
        let weights = [0.0f32, 0.5, 1.0];
        let plane = protect_plane_f32(&weights).expect("plane");
        assert_eq!(plane, vec![1.0f32, 0.5, 0.0]);

        assert_eq!(protect_plane_f32(&[0.0; 4]).expect("all open"), vec![1.0f32; 4]);
        assert_eq!(protect_plane_f32(&[1.0; 4]).expect("all protect"), vec![0.0f32; 4]);

        assert_eq!(
            protect_plane_f32(&[1.5]),
            Err(Refusal::ValueOutOfRange("region weight"))
        );
        assert_eq!(
            protect_plane_f32(&[f32::INFINITY]),
            Err(Refusal::ValueOutOfRange("region weight"))
        );
    }
}
