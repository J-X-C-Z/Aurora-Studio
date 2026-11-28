//! `body_exact_vendor.rs` — 美体精修已证原厂几何与像素公式（SG01–SG10）。
//!
//! 纯 `std`、零外部 crate、零 `unsafe`。单文件交付，可直接：
//!
//! ```text
//! rustc --edition 2021 --test body_exact_vendor.rs
//! ```
//!
//! ## 证据口径
//!
//! 所有常量取自 `pix-worker.exe`（SHA-256
//! `509D22938469906D353D44C355F4AA1E445F81EA56CA288A540AE5553CC96B84`）的静态
//! 反汇编与只读 GLSL。逐条 RVA / 常量池地址见
//! `90_work/lanes/portrait-parse/static-gaps-20261124/task01-body-radial/FINDINGS.md`、
//! `.../task02-body-subcontrols/FINDINGS.md`、`pf404-slots/FINDINGS.md` 与
//! `03_algorithms/reports/PORTRAIT-PARSE-GAPS-20261125.md`。
//!
//! 本文件只做 CPU 数学复现，不接路由、不采样像素、不跑 GPU。`ev6_vendor_parity`
//! 与 `ev7_real_photo_accept` 不因本文件改变（AGENTS.md 第 9、10 条）。
//!
//! ## 分区
//!
//! * SG01–SG03：PF417/418 `ShoulderWidth` 径向控制点 + `BodyLiquid.frag` 余弦衰减。
//! * PF404：`ButtockRefine` 两趟 × 2 点 + 4/2 区域几何。
//! * PF405：`BodyRefine` 12×2 网格顶点与 102 索引。
//! * SG07–SG10：子控 `RenderDegreeRanges` 与 `LengthenLeg` 共享腿阶段插值。
//!
//! ## 明确记录的边界（不得当成已闭合）
//!
//! 1. SG01–SG03 的 `u_src_points/u_dst_points/u_degrees/u_scale` 与宿主 C++ 产物的
//!    绑定在 exe 内没有 uniform 名字符串，属 `INFERRED`；GLSL 逐像素律本身
//!    `PROVEN`。
//! 2. PF417/418 二进制一次读取三个骨骼点（`rax+0x18/0x20/0x28` 与
//!    `+0x80/0x88/0x90`）来生成 A/B 两个控制点三元组。任务书只给两个端点
//!    `p1,p2`；[`shoulder_width_geometry`] 采用把每个三元组折叠到两点上的同式
//!    特化（A 侧锚 `p1`、参考 `p2`；B 侧锚 `p2`、参考 `p1`），不新增任何常量。
//!    完整三点形式由 [`shoulder_control_points`] 提供。
//! 3. 二进制对 A1/A2/B1/B2 的目标点另有逐点开关（`[node+0x10]`、`[node+0x44]`）。
//!    本模块用 [`ShoulderWidthGeometry::applied`] 显式建模，默认全开。
//! 4. PF404 的旋转常量与精确 2×2 矩阵已由 `plump-body-closeout-20261126/REPORT.md`
//!    反汇编 `0x3CB0A80` 证清（`PROVEN`）：常量池使用源码字面量 `0.866f32`
//!    （`0x3f5db22d`），区域 0 对单位向量 `u0 = normalize(P0 - M)` 施加 `+30°`
//!    矩阵 `[0.866, -0.5; 0.5, 0.866]`，区域 1 对 `u1 = normalize(P1 - M)` 施加
//!    `-30°` 矩阵 `[0.866, 0.5; -0.5, 0.866]`。
//! 5. PF404 的 `cell+0x2D0..0x2E0` 两趟缩放已证清（`PROVEN`）：
//!    `cell+0x2D0/0x2D4 = [1.0f, 1.0f]`；`cell+0x2D8/0x2DC`（两趟 `u_scale`）在
//!    `s >= 0` 时为 `[0.7f * s, 0.5f * s]`，`s < 0` 时为 `[0.49f * s, 0.35f * s]`
//!    （因此 `degree = 0` 时两趟 `u_scale` 均为 `0.0`）。
//! 6. PF404 的 69 槽解剖语义仍 `NOT_FOUND`；本模块只使用已证索引 `8..=13`。

#![forbid(unsafe_code)]

use std::f32::consts::PI;

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

    fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    fn add(self, other: Vec2) -> Vec2 {
        Vec2::new(self.x + other.x, self.y + other.y)
    }

    fn sub(self, other: Vec2) -> Vec2 {
        Vec2::new(self.x - other.x, self.y - other.y)
    }

    fn scale(self, factor: f32) -> Vec2 {
        Vec2::new(self.x * factor, self.y * factor)
    }

    fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

fn midpoint(a: Vec2, b: Vec2) -> Vec2 {
    a.add(b).scale(0.5)
}

fn finite_slice(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

/// 具名拒绝：缺失输入 / 数值越界 / 形状不匹配。**不 clamp、不零填充、不合成常量平面。**
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    MissingInput(&'static str),
    ValueOutOfRange(&'static str),
    ShapeMismatch(&'static str),
}

// ---------------------------------------------------------------------------
// SG01–SG03：PF417/418 ShoulderWidth 径向控制点与 BodyLiquid 逐像素公式
// ---------------------------------------------------------------------------

/// `0x3CF2D84` 起的控制点常量池（`0x8CF7FE0` / `0x8BFC200` / `0x8E552D0` /
/// `0x8BF C340` / `0x8E57FA0` / `0x8E55290`）。
pub const SHOULDER_COEFF_0_4: f32 = 0.4;
pub const SHOULDER_COEFF_0_5: f32 = 0.5;
pub const SHOULDER_COEFF_0_6: f32 = 0.6;
pub const SHOULDER_COEFF_0_9: f32 = 0.9;
pub const SHOULDER_COEFF_0_1: f32 = 0.1;
pub const SHOULDER_COEFF_0_8: f32 = 0.8;

/// PF417/418 每次上传的控制点 / 度数个数（A0..A2 + B0..B2）。
pub const SHOULDER_POINT_COUNT: usize = 6;
/// 每个骨骼三元组对应的 `DegreesScale` / `RadiusScale` 元素数。
pub const SHOULDER_SCALE_COUNT: usize = 3;

/// `BodyLiquid.frag` 的 `SmoothCoord` 常量（`2.49 - |4.98*(uv-0.5)|`）。
pub const SMOOTH_COORD_GAIN: f32 = 4.98;
pub const SMOOTH_COORD_BIAS: f32 = 2.49;

/// PF417/418 的三个骨骼点（A 侧）与三个骨骼点（B 侧）。
///
/// 偏移对应 `pix-worker.exe` 0x3CF2DC0..0x3CF2F10：
/// `a_prev=[rax+0x18]`、`a=[rax+0x20]`、`a_next=[rax+0x28]`、
/// `b_prev=[rax+0x80]`、`b=[rax+0x88]`、`b_next=[rax+0x90]`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShoulderBones {
    pub a_prev: Vec2,
    pub a: Vec2,
    pub a_next: Vec2,
    pub b_prev: Vec2,
    pub b: Vec2,
    pub b_next: Vec2,
}

impl ShoulderBones {
    fn finite(self) -> bool {
        [
            self.a_prev,
            self.a,
            self.a_next,
            self.b_prev,
            self.b,
            self.b_next,
        ]
        .iter()
        .all(|point| point.is_finite())
    }
}

/// 按 `0x3CF2DC0..0x3CF2F10` 的固定系数生成 A0/A1/A2/B0/B1/B2（像素坐标，未归一化）。
///
/// ```text
/// A0 = A1s + 0.4*(A1s - A0s)
/// A1 = 0.5*A1s + 0.5*A2s + 0.6*(A1s - A0s)
/// A2 = 0.1*A1s + 0.9*A2s + 0.8*(A1s - A0s)
/// B0 = B1s + 0.4*(B1s - B2s)
/// B1 = 0.5*B0s + 0.5*B1s + 0.6*(B1s - B2s)
/// B2 = 0.9*B0s + 0.1*B1s + 0.8*(B1s - B2s)
/// ```
///
/// 注意 B 侧的差向量是 `B1s - B2s`（`0x3CF2E87`），与 A 侧的 `A1s - A0s`
/// （`0x3CF2DCD`）不是同一个索引窗口；这是原厂字节，不是笔误。
pub fn shoulder_control_points(bones: ShoulderBones) -> [Vec2; SHOULDER_POINT_COUNT] {
    let a_delta = bones.a.sub(bones.a_prev);
    let b_delta = bones.b.sub(bones.b_next);

    let a0 = bones.a.add(a_delta.scale(SHOULDER_COEFF_0_4));
    let a1 = bones
        .a
        .scale(SHOULDER_COEFF_0_5)
        .add(bones.a_next.scale(SHOULDER_COEFF_0_5))
        .add(a_delta.scale(SHOULDER_COEFF_0_6));
    let a2 = bones
        .a
        .scale(SHOULDER_COEFF_0_1)
        .add(bones.a_next.scale(SHOULDER_COEFF_0_9))
        .add(a_delta.scale(SHOULDER_COEFF_0_8));

    let b0 = bones.b.add(b_delta.scale(SHOULDER_COEFF_0_4));
    let b1 = bones
        .b_prev
        .scale(SHOULDER_COEFF_0_5)
        .add(bones.b.scale(SHOULDER_COEFF_0_5))
        .add(b_delta.scale(SHOULDER_COEFF_0_6));
    let b2 = bones
        .b_prev
        .scale(SHOULDER_COEFF_0_9)
        .add(bones.b.scale(SHOULDER_COEFF_0_1))
        .add(b_delta.scale(SHOULDER_COEFF_0_8));

    [a0, a1, a2, b0, b1, b2]
}

/// 与 [`shoulder_control_points`] 相同，再乘 `S = (1/W, 1/H)`（`0x3CF2D84..0x3CF2DBD`）。
pub fn shoulder_control_points_uv(
    bones: ShoulderBones,
    width: u32,
    height: u32,
) -> Result<[Vec2; SHOULDER_POINT_COUNT], Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("image"));
    }
    if !bones.finite() {
        return Err(Refusal::ValueOutOfRange("SG01 landmarks"));
    }
    let inv = Vec2::new(1.0 / width as f32, 1.0 / height as f32);
    let points = shoulder_control_points(bones);
    let mut out = [Vec2::new(0.0, 0.0); SHOULDER_POINT_COUNT];
    for (slot, point) in out.iter_mut().zip(points.iter()) {
        *slot = Vec2::new(point.x * inv.x, point.y * inv.y);
    }
    Ok(out)
}

/// `D_a[k] = DegreesScale[k] * degree`（k = 0..3），`0x3CF2D27..0x3CF2D81`。
pub fn shoulder_degrees(degrees_scale: [f32; SHOULDER_SCALE_COUNT], degree: f32) -> [f32; 3] {
    [
        degrees_scale[0] * degree,
        degrees_scale[1] * degree,
        degrees_scale[2] * degree,
    ]
}

/// 两组度数：`[A 侧, B 侧]`。
///
/// 二进制在 `0x3CF2D0E..0x3CF2D27` 依据 `[r8+0x10] == 1` 交换两个入参，随后把
/// `DegreesScale[k]*xmm1` 写入 `cell+0x340+4k`、`DegreesScale[k]*xmm0` 写入
/// `cell+0x34C+4k`。调用方把已交换后的两个标量传进来即可。
pub fn shoulder_degree_sets(
    degrees_scale: [f32; SHOULDER_SCALE_COUNT],
    degree_a: f32,
    degree_b: f32,
) -> Result<[[f32; 3]; 2], Refusal> {
    if !finite_slice(&degrees_scale) || !degree_a.is_finite() || !degree_b.is_finite() {
        return Err(Refusal::ValueOutOfRange("SG01 degrees"));
    }
    Ok([
        shoulder_degrees(degrees_scale, degree_a),
        shoulder_degrees(degrees_scale, degree_b),
    ])
}

/// `0x3CF2F17..0x3CF2F93` 的半径位移：
/// `offset[k] = RadiusScale[k] * |p1-p2| * normalize(dir) * (1/W, 1/H)`。
///
/// `distance` 必须是 `|p1-p2|`（像素）。方向长度为 0 时拒绝，不合成方向。
pub fn shoulder_radius_offsets(
    radius_scale: [f32; SHOULDER_SCALE_COUNT],
    distance: f32,
    dir: Vec2,
    width: u32,
    height: u32,
) -> Result<[Vec2; 3], Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("image"));
    }
    if !finite_slice(&radius_scale) || !distance.is_finite() || distance < 0.0 {
        return Err(Refusal::ValueOutOfRange("SG01 radius"));
    }
    if !dir.is_finite() {
        return Err(Refusal::ValueOutOfRange("SG01 direction"));
    }
    let length = dir.length();
    if length == 0.0 {
        return Err(Refusal::MissingInput("SG01 direction"));
    }
    let unit = dir.scale(1.0 / length);
    let inv = Vec2::new(1.0 / width as f32, 1.0 / height as f32);
    let base = unit.scale(distance);
    let mut offsets = [Vec2::new(0.0, 0.0); 3];
    for (slot, scale) in offsets.iter_mut().zip(radius_scale.iter()) {
        let offset = base.scale(*scale);
        *slot = Vec2::new(offset.x * inv.x, offset.y * inv.y);
    }
    Ok(offsets)
}

/// PF417/418 一次绘制的全部输入：6 个源点 / 6 个目标点 / 两组度数 / 逐点开关。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShoulderWidthGeometry {
    /// `u_src_points[6]` = `{A0,A1,A2,B0,B1,B2}`（UV）。
    pub source: [Vec2; SHOULDER_POINT_COUNT],
    /// `u_dst_points[6]`；`applied[k] == false` 时等于源点。
    pub destination: [Vec2; SHOULDER_POINT_COUNT],
    /// `u_degrees[6]` 的两组：`[A0..A2, B0..B2]`。
    pub degrees: [[f32; 3]; 2],
    /// 二进制逐点开关（`[node+0x10]` 控 A1/A2，`[node+0x44]` 控 B1/B2）。
    pub applied: [bool; SHOULDER_POINT_COUNT],
}

/// PF417/418 欠定特化入口：只给两个骨骼端点 `p1,p2`、方向、单个 `degree`。
///
/// 折叠映射（不是新常量，而是把三点同式落到两点）：
/// A 侧 `a_prev = a_next = p2`、`a = p1`；
/// B 侧 `b_prev = b_next = p1`、`b = p2`。
/// 于是 A 的差向量为 `p1-p2`，B 的差向量为 `p2-p1`，六个控制点仍严格使用
/// 0.4/0.5/0.6/0.9/0.1/0.8。
#[allow(clippy::too_many_arguments)]
pub fn shoulder_width_geometry(
    p1: Vec2,
    p2: Vec2,
    dir: Vec2,
    degree: f32,
    degrees_scale: [f32; SHOULDER_SCALE_COUNT],
    radius_scale: [f32; SHOULDER_SCALE_COUNT],
    width: u32,
    height: u32,
) -> Result<ShoulderWidthGeometry, Refusal> {
    shoulder_width_geometry_with_degrees(
        p1,
        p2,
        dir,
        degree,
        degree,
        degrees_scale,
        radius_scale,
        width,
        height,
    )
}

/// 同 [`shoulder_width_geometry`]，形参按任务书写成 `[f32; 2]`。
#[allow(clippy::too_many_arguments)]
pub fn shoulder_width_geometry_array(
    p1: [f32; 2],
    p2: [f32; 2],
    dir: [f32; 2],
    degree: f32,
    degrees_scale: [f32; SHOULDER_SCALE_COUNT],
    radius_scale: [f32; SHOULDER_SCALE_COUNT],
    width: u32,
    height: u32,
) -> Result<ShoulderWidthGeometry, Refusal> {
    shoulder_width_geometry(
        Vec2::new(p1[0], p1[1]),
        Vec2::new(p2[0], p2[1]),
        Vec2::new(dir[0], dir[1]),
        degree,
        degrees_scale,
        radius_scale,
        width,
        height,
    )
}

/// 同 [`shoulder_width_geometry`]，但允许 417 / 418 两侧使用不同 degree。
#[allow(clippy::too_many_arguments)]
pub fn shoulder_width_geometry_with_degrees(
    p1: Vec2,
    p2: Vec2,
    dir: Vec2,
    degree_a: f32,
    degree_b: f32,
    degrees_scale: [f32; SHOULDER_SCALE_COUNT],
    radius_scale: [f32; SHOULDER_SCALE_COUNT],
    width: u32,
    height: u32,
) -> Result<ShoulderWidthGeometry, Refusal> {
    let applied = [true; SHOULDER_POINT_COUNT];
    shoulder_width_geometry_gated(
        p1,
        p2,
        dir,
        degree_a,
        degree_b,
        degrees_scale,
        radius_scale,
        width,
        height,
        applied,
    )
}

/// 同 [`shoulder_width_geometry_with_degrees`]，带逐点开关。
#[allow(clippy::too_many_arguments)]
pub fn shoulder_width_geometry_gated(
    p1: Vec2,
    p2: Vec2,
    dir: Vec2,
    degree_a: f32,
    degree_b: f32,
    degrees_scale: [f32; SHOULDER_SCALE_COUNT],
    radius_scale: [f32; SHOULDER_SCALE_COUNT],
    width: u32,
    height: u32,
    applied: [bool; SHOULDER_POINT_COUNT],
) -> Result<ShoulderWidthGeometry, Refusal> {
    let bones = ShoulderBones {
        a_prev: p2,
        a: p1,
        a_next: p2,
        b_prev: p1,
        b: p2,
        b_next: p1,
    };
    let source = shoulder_control_points_uv(bones, width, height)?;
    let distance = p1.sub(p2).length();
    let offsets = shoulder_radius_offsets(radius_scale, distance, dir, width, height)?;
    let degrees = shoulder_degree_sets(degrees_scale, degree_a, degree_b)?;

    let mut destination = source;
    for k in 0..SHOULDER_SCALE_COUNT {
        if applied[k] {
            destination[k] = source[k].add(offsets[k]);
        }
        if applied[k + SHOULDER_SCALE_COUNT] {
            // `0x3CF3019` 的 B 侧方向是 `[ctx+0x30] - [ctx+0x38]`，与 A 侧相反。
            destination[k + SHOULDER_SCALE_COUNT] = source[k + SHOULDER_SCALE_COUNT].sub(offsets[k]);
        }
    }

    Ok(ShoulderWidthGeometry {
        source,
        destination,
        degrees,
        applied,
    })
}

/// `BodyLiquid.frag` 的 `SmoothCoord`：
/// `clamp(min(2.49 - |4.98*(u-0.5)|, 2.49 - |4.98*(v-0.5)|), 0, 1)`。
pub fn smooth_coord(uv: Vec2) -> f32 {
    let x = SMOOTH_COORD_BIAS - (SMOOTH_COORD_GAIN * (uv.x - 0.5)).abs();
    let y = SMOOTH_COORD_BIAS - (SMOOTH_COORD_GAIN * (uv.y - 0.5)).abs();
    x.min(y).clamp(0.0, 1.0)
}

/// `BodyLiquid.frag` 逐像素余弦衰减（归一化 UV 域）。
///
/// ```text
/// for k:
///   P   = dst[k] - src[k]
///   R2  = (P.x * inv_aspect)^2 + P.y^2
///   D2  = ((uv - src[k]).x * inv_aspect)^2 + ((uv - src[k]).y)^2
///   if D2 < R2:
///     w  = (cos(PI * sqrt(D2 / R2)) + 1) / 2
///     total += P * degrees[k] * w * scale^2
/// uv' = uv - total * SmoothCoord(uv)
/// ```
///
/// 严格使用 `D2 < R2`（相等即跳过）。`degrees` 为 0 时恒等。
pub fn body_liquid_uv(
    uv: Vec2,
    source: &[Vec2],
    destination: &[Vec2],
    degrees: &[f32],
    scale: f32,
    inverse_aspect: f32,
) -> Result<Vec2, Refusal> {
    if source.len() != destination.len() || source.len() != degrees.len() {
        return Err(Refusal::ShapeMismatch("BodyLiquid points"));
    }
    if !uv.is_finite() || !scale.is_finite() || !inverse_aspect.is_finite() || inverse_aspect <= 0.0 {
        return Err(Refusal::ValueOutOfRange("BodyLiquid scalars"));
    }
    if !finite_slice(degrees)
        || !source.iter().all(|point| point.is_finite())
        || !destination.iter().all(|point| point.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("BodyLiquid points"));
    }

    let mut total = Vec2::new(0.0, 0.0);
    for k in 0..source.len() {
        let diff = destination[k].sub(source[k]);
        let threshold_x = diff.x * inverse_aspect;
        let threshold = threshold_x * threshold_x + diff.y * diff.y;

        let current = uv.sub(source[k]);
        let current_x = current.x * inverse_aspect;
        let current_dist = current_x * current_x + current.y * current.y;

        if threshold > current_dist {
            let ratio = (current_dist / threshold).sqrt();
            let factor = ((ratio * PI).cos() + 1.0) * 0.5;
            let contribution = diff
                .scale(degrees[k])
                .scale(factor)
                .scale(scale)
                .scale(scale);
            total = total.add(contribution);
        }
    }

    let smoothing = smooth_coord(uv);
    Ok(Vec2::new(
        uv.x - total.x * smoothing,
        uv.y - total.y * smoothing,
    ))
}

/// 把 [`ShoulderWidthGeometry`] 直接交给 [`body_liquid_uv`] 的便捷入口。
pub fn shoulder_body_liquid_uv(
    uv: Vec2,
    geometry: &ShoulderWidthGeometry,
    scale: f32,
    inverse_aspect: f32,
) -> Result<Vec2, Refusal> {
    let mut degrees = [0.0f32; SHOULDER_POINT_COUNT];
    for k in 0..SHOULDER_SCALE_COUNT {
        degrees[k] = geometry.degrees[0][k];
        degrees[k + SHOULDER_SCALE_COUNT] = geometry.degrees[1][k];
    }
    body_liquid_uv(
        uv,
        &geometry.source,
        &geometry.destination,
        &degrees,
        scale,
        inverse_aspect,
    )
}

// ---------------------------------------------------------------------------
// PF404：ButtockRefine 两趟 × 2 点 + 4/2 区域几何
// ---------------------------------------------------------------------------

/// vendor `oriSkeleton` 固定点数（`loadExtJsonParam` 断言 `0x228 == 69*8`）。
pub const VENDOR_SKELETON_POINT_COUNT: usize = 69;
/// PF404 几何生成器（`0x3CB0A80`）读取的固定索引。
pub const PF404_VENDOR_SKELETON_SLOTS: [usize; 6] = [8, 9, 10, 11, 12, 13];

/// `cell+0x2D8`（Pass 0）在 `s >= 0` 时的 `u_scale` 基础系数（`0x08e56780` lane0 = `0x3f333333`）。
pub const PF404_PASS0_SCALE_POSITIVE: f32 = 0.7;
/// `cell+0x2DC`（Pass 1）在 `s >= 0` 时的 `u_scale` 基础系数（`0x08e56780` lane1 = `0x3f000000`）。
pub const PF404_PASS1_SCALE_POSITIVE: f32 = 0.5;
/// `s < 0` 时的额外衰减系数（`0x08ddd8b0` = `[0.7f, 0.7f]`）。
pub const PF404_NEGATIVE_ATTEN: f32 = 0.7;
/// `cell+0x2D8`（Pass 0）在 `s < 0` 时的 `u_scale` 系数（`0.7f * 0.7f = 0.49f`）。
pub const PF404_PASS0_SCALE_NEGATIVE: f32 = 0.49;
/// `cell+0x2DC`（Pass 1）在 `s < 0` 时的 `u_scale` 系数（`0.5f * 0.7f = 0.35f`）。
pub const PF404_PASS1_SCALE_NEGATIVE: f32 = 0.35;
pub const PF404_KA_POSITIVE: f32 = 0.2;
pub const PF404_KA_NEGATIVE: f32 = 0.28;
pub const PF404_KB_POSITIVE: f32 = 0.25;
pub const PF404_KB_NEGATIVE: f32 = 0.1;
pub const PF404_REGION_SCALE: f32 = 5.0;
pub const PF404_EXTRAPOLATE_SCALE: f32 = 7.0;

/// 原厂常量池 `0x08e567a0/b0/d0/e0` 的硬编码字面量 `0.866f32`（`0x3f5db22d`），
/// 并非 `sqrt(3)/2`（`0x3f5db3d7`）。
pub const PF404_ROT_COS_30: f32 = 0.866;
/// 原厂常量池 `0x08e567a0/b0/d0/e0` 的正弦字面量 `0.5f32`（`0x3f000000`）。
pub const PF404_ROT_SIN_30: f32 = 0.5;

/// 区域 0（`0x3cb0cc6..0x3cb0ce7`）精确 2×2 矩阵（`+30°`）：
/// `[[0.866, -0.5], [0.5, 0.866]]`。
fn rotate_region0_pos_30(u: Vec2) -> Vec2 {
    Vec2::new(
        PF404_ROT_COS_30 * u.x - PF404_ROT_SIN_30 * u.y,
        PF404_ROT_SIN_30 * u.x + PF404_ROT_COS_30 * u.y,
    )
}

/// 区域 1（`0x3cb0bea..0x3cb0c05` / `0x3cb0d6f..0x3cb0d8a`）精确 2×2 矩阵（`-30°`）：
/// `[[0.866, 0.5], [-0.5, 0.866]]`。
fn rotate_region1_neg_30(u: Vec2) -> Vec2 {
    Vec2::new(
        PF404_ROT_COS_30 * u.x + PF404_ROT_SIN_30 * u.y,
        -PF404_ROT_SIN_30 * u.x + PF404_ROT_COS_30 * u.y,
    )
}

/// 一个两趟 draw 中的单个点。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WarpPoint {
    pub source: Vec2,
    pub destination: Vec2,
    pub degree: f32,
    /// vendor 在该分支下是否写入该槽（2 区模式只写 1/3）。
    pub active: bool,
}

/// `GLFilterBodyWarp` 的一次调用：2 个点 + 一个 `u_scale`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyWarpPass {
    pub points: [WarpPoint; 2],
    pub scale: f32,
}

/// `ButtockRefine` 的两次 `GLFilterBodyWarp`（`0x3CB07B2` / `0x3CB0912`）。
///
/// 区域顺序固定：pass 0 = `{区域 0, 区域 1}`，pass 1 = `{区域 2, 区域 3}`，
/// 对应 `cell+0x2D0/0x2D4 = [1.0f, 1.0f]` 与 `cell+0x2D8/0x2DC` 两趟 `u_scale`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ButtockWarp {
    pub passes: [BodyWarpPass; 2],
    pub four_regions: bool,
}

/// 按 `0x3CB0A80` 已证算术生成 PF404 的两趟 × 2 点参数（`PROVEN`）。
///
/// - `skeleton`：vendor 骨架点，至少 14 点（读取索引 8..=13）；正常长度 69。
/// - `degree`：`0x5711D20` 重映射后的标量 `s`。
/// - `person_type`：`[node+0x10]`，与 `3` 比较。
/// - `edge_protection`：`[node+0x28]`，真值取反后参与分支。
///
/// 两趟 `u_scale`（`cell+0x2D8/0x2DC`）：
/// - `s >= 0`：`[0.7f * s, 0.5f * s]`
/// - `s < 0`：`[0.49f * s, 0.35f * s]`
///
/// 区域方向退化（`P_i == M`）返回 `MissingInput("PF404 region basis")`，
/// 不产生 `NaN/inf`。
pub fn buttock_region_geometry(
    skeleton: &[Vec2],
    degree: f32,
    person_type: i32,
    edge_protection: bool,
) -> Result<ButtockWarp, Refusal> {
    if skeleton.len() < 14 {
        return Err(Refusal::MissingInput("PF404 vendor skeleton"));
    }
    for index in PF404_VENDOR_SKELETON_SLOTS {
        if !skeleton[index].is_finite() {
            return Err(Refusal::ValueOutOfRange("PF404 vendor skeleton"));
        }
    }
    if !degree.is_finite() {
        return Err(Refusal::ValueOutOfRange("PF404 degree"));
    }

    let pass0_base = PF404_PASS0_SCALE_POSITIVE * degree;
    let pass1_base = PF404_PASS1_SCALE_POSITIVE * degree;
    let positive_branch = pass0_base >= 0.0;

    let (i0, i1, outer_a, outer_b, ka, kb, pass_scales) = if positive_branch {
        (
            9usize,
            12usize,
            8usize,
            13usize,
            PF404_KA_POSITIVE,
            PF404_KB_POSITIVE,
            [pass0_base, pass1_base],
        )
    } else {
        (
            10usize,
            11usize,
            9usize,
            12usize,
            PF404_KA_NEGATIVE,
            PF404_KB_NEGATIVE,
            [
                PF404_PASS0_SCALE_NEGATIVE * degree,
                PF404_PASS1_SCALE_NEGATIVE * degree,
            ],
        )
    };

    let m = midpoint(skeleton[i0], skeleton[i1]);
    let four_regions = !edge_protection || person_type == 3;

    let mut sources = [Vec2::new(0.0, 0.0); 4];
    let mut destinations = [Vec2::new(0.0, 0.0); 4];
    let mut active = [false; 4];

    if four_regions {
        let p0 = midpoint(skeleton[9], skeleton[10]);
        sources[0] = p0;
        let v0 = p0.sub(m);
        let len0 = v0.length();
        if len0 == 0.0 {
            return Err(Refusal::MissingInput("PF404 region basis"));
        }
        let u0 = v0.scale(1.0 / len0);
        destinations[0] = p0.add(rotate_region0_pos_30(u0).scale(PF404_REGION_SCALE * ka * len0));
        active[0] = true;

        let p1 = midpoint(skeleton[11], skeleton[12]);
        sources[1] = p1;
        let v1 = p1.sub(m);
        let len1 = v1.length();
        if len1 == 0.0 {
            return Err(Refusal::MissingInput("PF404 region basis"));
        }
        let u1 = v1.scale(1.0 / len1);
        destinations[1] = p1.add(rotate_region1_neg_30(u1).scale(PF404_REGION_SCALE * ka * len1));
        active[1] = true;
    } else {
        let p1 = midpoint(skeleton[11], skeleton[12]);
        sources[1] = p1;
        let v1 = p1.sub(m);
        let len1 = v1.length();
        if len1 == 0.0 {
            return Err(Refusal::MissingInput("PF404 region basis"));
        }
        let u1 = v1.scale(1.0 / len1);
        destinations[1] = p1.add(rotate_region1_neg_30(u1).scale(PF404_REGION_SCALE * ka * len1));
        active[1] = true;
    }

    let p_outer_a = skeleton[outer_a];
    let p_outer_b = skeleton[outer_b];
    let outer_mid = midpoint(p_outer_a, p_outer_b);

    if four_regions {
        sources[2] = p_outer_a;
        destinations[2] = p_outer_a.add(
            outer_mid
                .sub(p_outer_a)
                .scale(PF404_EXTRAPOLATE_SCALE * kb),
        );
        active[2] = true;
    }

    sources[3] = p_outer_b;
    destinations[3] = p_outer_b.add(
        outer_mid
            .sub(p_outer_b)
            .scale(PF404_EXTRAPOLATE_SCALE * kb),
    );
    active[3] = true;

    let mut passes = [BodyWarpPass {
        points: [WarpPoint {
            source: Vec2::new(0.0, 0.0),
            destination: Vec2::new(0.0, 0.0),
            degree: 0.0,
            active: false,
        }; 2],
        scale: 0.0,
    }; 2];

    for (pass_index, region_base) in [(0usize, 0usize), (1usize, 2usize)] {
        for local in 0..2 {
            let region = region_base + local;
            passes[pass_index].points[local] = WarpPoint {
                source: sources[region],
                destination: destinations[region],
                degree: 1.0,
                active: active[region],
            };
        }
        passes[pass_index].scale = pass_scales[pass_index];
    }

    Ok(ButtockWarp {
        passes,
        four_regions,
    })
}

// ---------------------------------------------------------------------------
// PF405：BodyRefine 12×2 网格顶点与 102 索引
// ---------------------------------------------------------------------------

pub const BODY_REFINE_ROWS: u32 = 12;
pub const BODY_REFINE_COLS: u32 = 2;
pub const BODY_REFINE_VERTEX_COUNT: usize = 24;
pub const BODY_REFINE_INDEX_COUNT: usize = 102;

/// `index_count = (6*cols - 3)*rows - 6`。
pub fn body_refine_index_count(rows: u32, cols: u32) -> Result<usize, Refusal> {
    if rows < 3 || cols < 2 {
        return Err(Refusal::ValueOutOfRange("PF405_MESH_SHAPE"));
    }
    Ok((6 * cols as usize - 3) * rows as usize - 6)
}

/// 恢复的 12×2 网格。
#[derive(Clone, Debug, PartialEq)]
pub struct BodyRefineMesh {
    pub rows: u32,
    pub cols: u32,
    /// `[row * cols + col]`，已除以 `(width, height)`。
    pub vertices: Vec<[f32; 2]>,
    /// 10 个三角扇 + 12 个环绕四边形。
    pub indices: Vec<u16>,
}

/// `0x4B7C2BC..0x4B7C388` 的索引流：先 `(0, (r-1)*cols, r*cols)`（r=2..rows），
/// 再对每个列步 `c` 输出环绕四边形 `(a, b, a+1, b, a+1, b+1)`。
pub fn body_refine_indices(rows: u32, cols: u32) -> Result<Vec<u16>, Refusal> {
    let expected = body_refine_index_count(rows, cols)?;
    let vertex_count = rows as usize * cols as usize;
    if vertex_count > u16::MAX as usize {
        return Err(Refusal::ValueOutOfRange("PF405_MESH_SHAPE"));
    }
    let mut indices = Vec::with_capacity(expected);
    for row in 2..rows {
        indices.push(0);
        indices.push(((row - 1) * cols) as u16);
        indices.push((row * cols) as u16);
    }
    for column_step in 0..cols - 1 {
        for row in 0..rows {
            let next = if row + 1 == rows { 0 } else { row + 1 };
            let a = row * cols + column_step;
            let b = next * cols + column_step;
            indices.push(a as u16);
            indices.push(b as u16);
            indices.push((a + 1) as u16);
            indices.push(b as u16);
            indices.push((a + 1) as u16);
            indices.push((b + 1) as u16);
        }
    }
    Ok(indices)
}

/// `0x4B7E0F0` 的顶点生成（`0x402` 字节）：
///
/// ```text
/// center  = (rec0 + rec1) * 0.5
/// origin  = (query + (rec1b + center) * 0.5) * 0.5
/// radial0 = 0.7 * |center - rec1b|      radial1 = 5.5 * radial0
/// angle   = (row/12) * 2PI
/// col 0:   x = 0.8 * (radial0 * (0.8*cos)), y = radial0 * sin
/// col t>0: radius = (1-t)*radial0 + t*radial1
///          x = 0.3 * (radius * (0.8*cos)), y = radius * sin
/// vertex  = (x, y) / (width, height)
/// ```
///
/// 该函数**没有 degree / mask 输入**；degree 只在下游 `BodyRefine.frag` 逐像素使用。
pub fn body_refine_vertices(
    width: u32,
    height: u32,
    query: [f32; 2],
    rec0: [f32; 2],
    rec1: [f32; 2],
    rec1b: [f32; 2],
) -> Result<BodyRefineMesh, Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("image"));
    }
    if !finite_slice(&[query[0], query[1], rec0[0], rec0[1], rec1[0], rec1[1], rec1b[0], rec1b[1]])
    {
        return Err(Refusal::ValueOutOfRange("PF405_LANDMARKS"));
    }

    let query = Vec2::new(query[0], query[1]);
    let rec0 = Vec2::new(rec0[0], rec0[1]);
    let rec1 = Vec2::new(rec1[0], rec1[1]);
    let rec1b = Vec2::new(rec1b[0], rec1b[1]);

    let center = midpoint(rec0, rec1);
    let origin = midpoint(query, midpoint(rec1b, center));
    let radial0 = 0.7 * center.sub(rec1b).length();
    let radial1 = 5.5 * radial0;

    let inv_width = 1.0 / width as f32;
    let inv_height = 1.0 / height as f32;

    let mut vertices = Vec::with_capacity(BODY_REFINE_VERTEX_COUNT);
    for row in 0..BODY_REFINE_ROWS {
        let angle = (row as f32 / BODY_REFINE_ROWS as f32) * PI * 2.0;
        let sine = angle.sin();
        let cosine = angle.cos();

        let x = 0.8 * (radial0 * (0.8 * cosine)) + origin.x;
        let y = radial0 * sine + origin.y;
        vertices.push([x * inv_width, y * inv_height]);

        for col in 1..BODY_REFINE_COLS {
            let t = col as f32 / (BODY_REFINE_COLS as f32 - 1.0);
            let radius = (1.0 - t) * radial0 + t * radial1;
            let x = 0.3 * (radius * (0.8 * cosine)) + origin.x;
            let y = radius * sine + origin.y;
            vertices.push([x * inv_width, y * inv_height]);
        }
    }

    let indices = body_refine_indices(BODY_REFINE_ROWS, BODY_REFINE_COLS)?;
    Ok(BodyRefineMesh {
        rows: BODY_REFINE_ROWS,
        cols: BODY_REFINE_COLS,
        vertices,
        indices,
    })
}

/// `BodyRefine.frag`：`degree = 1 - mask.r`。
pub fn body_refine_fragment_degree(mask_r: u8) -> f32 {
    1.0 - mask_r as f32 / 255.0
}

/// `BodyRefine.frag` 的非 `UVMAP` 消费端：`degree > 0.01` 时在两组 UV 间 lerp。
pub fn body_refine_fragment_uv(
    source_uv: [f32; 2],
    target_uv: [f32; 2],
    mask_r: u8,
) -> Result<[f32; 2], Refusal> {
    if !finite_slice(&[
        source_uv[0],
        source_uv[1],
        target_uv[0],
        target_uv[1],
    ]) {
        return Err(Refusal::ValueOutOfRange("PF405_UV"));
    }
    let degree = body_refine_fragment_degree(mask_r);
    if degree > 0.01 {
        Ok([
            source_uv[0] + (target_uv[0] - source_uv[0]) * degree,
            source_uv[1] + (target_uv[1] - source_uv[1]) * degree,
        ])
    } else {
        Ok(source_uv)
    }
}

// ---------------------------------------------------------------------------
// SG07–SG10：子控范围与共享腿阶段线性插值
// ---------------------------------------------------------------------------

/// PF411/412 `RightShoulder`（`PoseRefine/effect.json` Effects[3]）。
pub const RIGHT_SHOULDER_RENDER_DEGREE_RANGES: [f32; 2] = [0.0, 0.4];
/// PF413/414 `SwanNeck`（Effects[2]）。
pub const SWAN_NECK_RENDER_DEGREE_RANGES: [f32; 2] = [0.0, 0.55];
/// PF30074/30075 `SlimLegRefine`（Effects[8]，左右腿同范围）。
pub const SLIM_LEG_RENDER_DEGREE_RANGES: [f32; 2] = [-1.0, 1.0];

/// PF30094/30095 共享腿阶段插值（`0x3CCD480` / `0x3CCD4D3` / `0x3CCD540`）：
/// `(f[+0xF0] - f[+0xF4]) * f[+0xE4] + f[+0xF4]`，即 `(hi - lo) * t + lo`。
pub fn lengthen_leg_shared_stage(hi: f32, lo: f32, t: f32) -> f32 {
    (hi - lo) * t + lo
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

    fn skeleton() -> Vec<Vec2> {
        (0..VENDOR_SKELETON_POINT_COUNT)
            .map(|i| Vec2::new(0.1 + 0.01 * i as f32, 0.2 + 0.02 * i as f32))
            .collect()
    }

    // ---- constants ----

    #[test]
    fn proven_constants_are_exact() {
        assert_eq!(SHOULDER_COEFF_0_4, 0.4f32);
        assert_eq!(SHOULDER_COEFF_0_5, 0.5f32);
        assert_eq!(SHOULDER_COEFF_0_6, 0.6f32);
        assert_eq!(SHOULDER_COEFF_0_9, 0.9f32);
        assert_eq!(SHOULDER_COEFF_0_1, 0.1f32);
        assert_eq!(SHOULDER_COEFF_0_8, 0.8f32);
        assert_eq!(SMOOTH_COORD_GAIN, 4.98f32);
        assert_eq!(SMOOTH_COORD_BIAS, 2.49f32);
        assert_eq!(PF404_PASS0_SCALE_POSITIVE, 0.7f32);
        assert_eq!(PF404_PASS1_SCALE_POSITIVE, 0.5f32);
        assert_eq!(PF404_NEGATIVE_ATTEN, 0.7f32);
        assert_eq!(PF404_PASS0_SCALE_NEGATIVE, 0.49f32);
        assert_eq!(PF404_PASS1_SCALE_NEGATIVE, 0.35f32);
        assert_eq!(PF404_KA_POSITIVE, 0.2f32);
        assert_eq!(PF404_KA_NEGATIVE, 0.28f32);
        assert_eq!(PF404_KB_POSITIVE, 0.25f32);
        assert_eq!(PF404_KB_NEGATIVE, 0.1f32);
        assert_eq!(PF404_REGION_SCALE, 5.0f32);
        assert_eq!(PF404_EXTRAPOLATE_SCALE, 7.0f32);
        assert_eq!(PF404_ROT_COS_30, 0.866f32);
        assert_eq!(PF404_ROT_COS_30.to_bits(), 0x3f5d_b22d);
        assert_eq!(PF404_ROT_SIN_30, 0.5f32);
        assert_eq!(PF404_ROT_SIN_30.to_bits(), 0x3f00_0000);
        assert_eq!(PF404_VENDOR_SKELETON_SLOTS, [8, 9, 10, 11, 12, 13]);
        assert_eq!(VENDOR_SKELETON_POINT_COUNT, 69);
        assert_eq!(SHOULDER_POINT_COUNT, 6);
        assert_eq!(SHOULDER_SCALE_COUNT, 3);
    }

    // ---- SG01–SG03 ----

    #[test]
    fn shoulder_degrees_scale_is_multiplicative() {
        let scale = [0.63f32, 0.56, 0.5];
        assert_eq!(shoulder_degrees(scale, 2.0), [1.26, 1.12, 1.0]);
        assert_eq!(shoulder_degrees(scale, 0.0), [0.0, 0.0, 0.0]);
        let sets = shoulder_degree_sets(scale, 0.5, -0.25).unwrap();
        assert_eq!(sets[0], [0.315, 0.28, 0.25]);
        assert_eq!(sets[1], [-0.1575, -0.14, -0.125]);
        assert_eq!(
            shoulder_degree_sets(scale, f32::NAN, 0.0),
            Err(Refusal::ValueOutOfRange("SG01 degrees"))
        );
    }

    #[test]
    fn shoulder_radius_offsets_match_proven_formula() {
        let p1 = Vec2::new(100.0, 100.0);
        let p2 = Vec2::new(60.0, 80.0);
        let distance = p1.sub(p2).length();
        assert!(close(distance, 2000.0f32.sqrt()));
        let dir = Vec2::new(0.0, 5.0);
        let offsets =
            shoulder_radius_offsets([1.0, 0.5, -1.0], distance, dir, 400, 200).unwrap();
        let unit_y = distance * (1.0 / 200.0);
        assert!(vec_close(offsets[0], Vec2::new(0.0, unit_y)));
        assert!(vec_close(offsets[1], Vec2::new(0.0, unit_y * 0.5)));
        assert!(vec_close(offsets[2], Vec2::new(0.0, -unit_y)));
    }

    #[test]
    fn shoulder_radius_offsets_reject_degenerate_inputs() {
        assert_eq!(
            shoulder_radius_offsets([1.0; 3], 1.0, Vec2::new(0.0, 0.0), 10, 10),
            Err(Refusal::MissingInput("SG01 direction"))
        );
        assert_eq!(
            shoulder_radius_offsets([1.0; 3], 1.0, Vec2::new(1.0, 1.0), 0, 10),
            Err(Refusal::ShapeMismatch("image"))
        );
        assert_eq!(
            shoulder_radius_offsets([1.0; 3], f32::INFINITY, Vec2::new(1.0, 1.0), 10, 10),
            Err(Refusal::ValueOutOfRange("SG01 radius"))
        );
        assert_eq!(
            shoulder_radius_offsets([1.0; 3], 1.0, Vec2::new(f32::NAN, 1.0), 10, 10),
            Err(Refusal::ValueOutOfRange("SG01 direction"))
        );
    }

    #[test]
    fn shoulder_control_points_match_hand_computation() {
        let bones = ShoulderBones {
            a_prev: Vec2::new(10.0, 10.0),
            a: Vec2::new(20.0, 30.0),
            a_next: Vec2::new(50.0, 40.0),
            b_prev: Vec2::new(0.0, 0.0),
            b: Vec2::new(5.0, 5.0),
            b_next: Vec2::new(9.0, 1.0),
        };
        let points = shoulder_control_points(bones);
        assert!(vec_close(points[0], Vec2::new(24.0, 38.0)));
        assert!(vec_close(points[1], Vec2::new(41.0, 47.0)));
        assert!(vec_close(points[2], Vec2::new(55.0, 55.0)));
        assert!(vec_close(points[3], Vec2::new(3.4, 6.6)));
        assert!(vec_close(points[4], Vec2::new(0.1, 4.9)));
        assert!(vec_close(points[5], Vec2::new(-2.7, 3.7)));
    }

    #[test]
    fn shoulder_control_points_uv_divides_by_image_size() {
        let bones = ShoulderBones {
            a_prev: Vec2::new(0.0, 0.0),
            a: Vec2::new(10.0, 20.0),
            a_next: Vec2::new(20.0, 30.0),
            b_prev: Vec2::new(5.0, 5.0),
            b: Vec2::new(15.0, 25.0),
            b_next: Vec2::new(25.0, 35.0),
        };
        let pixels = shoulder_control_points(bones);
        let uv = shoulder_control_points_uv(bones, 200, 100).unwrap();
        for k in 0..SHOULDER_POINT_COUNT {
            assert!(vec_close(
                uv[k],
                Vec2::new(pixels[k].x / 200.0, pixels[k].y / 100.0)
            ));
        }
        assert_eq!(
            shoulder_control_points_uv(bones, 0, 100),
            Err(Refusal::ShapeMismatch("image"))
        );
    }

    #[test]
    fn shoulder_width_two_point_collapse_matches_proven_coefficients() {
        let p1 = Vec2::new(20.0, 40.0);
        let p2 = Vec2::new(10.0, 20.0);
        let geometry = shoulder_width_geometry(
            p1,
            p2,
            Vec2::new(0.0, 1.0),
            1.0,
            [1.0; 3],
            [0.0; 3],
            100,
            100,
        )
        .unwrap();

        let expected = [
            Vec2::new(24.0, 48.0),
            Vec2::new(21.0, 42.0),
            Vec2::new(19.0, 38.0),
            Vec2::new(6.0, 12.0),
            Vec2::new(9.0, 18.0),
            Vec2::new(11.0, 22.0),
        ];
        for k in 0..SHOULDER_POINT_COUNT {
            assert!(vec_close(
                geometry.source[k],
                Vec2::new(expected[k].x / 100.0, expected[k].y / 100.0)
            ));
        }
        assert_eq!(geometry.applied, [true; 6]);
        assert_eq!(geometry.destination, geometry.source);
    }

    #[test]
    fn shoulder_width_array_entry_matches_vec2_entry() {
        let array = shoulder_width_geometry_array(
            [20.0, 40.0],
            [10.0, 20.0],
            [3.0, 4.0],
            0.5,
            [0.63, 0.56, 0.5],
            [1.2, 1.2, 1.2],
            200,
            100,
        )
        .unwrap();
        let vec = shoulder_width_geometry(
            Vec2::new(20.0, 40.0),
            Vec2::new(10.0, 20.0),
            Vec2::new(3.0, 4.0),
            0.5,
            [0.63, 0.56, 0.5],
            [1.2, 1.2, 1.2],
            200,
            100,
        )
        .unwrap();
        assert_eq!(array, vec);
    }

    #[test]
    fn shoulder_width_destination_adds_radius_for_a_and_subtracts_for_b() {
        let geometry = shoulder_width_geometry(
            Vec2::new(30.0, 40.0),
            Vec2::new(10.0, 20.0),
            Vec2::new(3.0, 4.0),
            1.0,
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            200,
            100,
        )
        .unwrap();
        let distance = ((20.0f32 * 20.0) + (20.0 * 20.0)).sqrt();
        let offsets = shoulder_radius_offsets(
            [1.0, 1.0, 1.0],
            distance,
            Vec2::new(3.0, 4.0),
            200,
            100,
        )
        .unwrap();
        for k in 0..3 {
            assert!(vec_close(
                geometry.destination[k],
                geometry.source[k].add(offsets[k])
            ));
            assert!(vec_close(
                geometry.destination[k + 3],
                geometry.source[k + 3].sub(offsets[k])
            ));
        }
        assert_eq!(geometry.degrees, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]);
    }

    #[test]
    fn shoulder_width_gated_leaves_point_unchanged() {
        let applied = [true, false, true, true, false, true];
        let geometry = shoulder_width_geometry_gated(
            Vec2::new(30.0, 40.0),
            Vec2::new(10.0, 20.0),
            Vec2::new(1.0, 0.0),
            1.0,
            0.5,
            [1.0; 3],
            [1.0; 3],
            200,
            100,
            applied,
        )
        .unwrap();
        assert_eq!(geometry.applied, applied);
        assert_eq!(geometry.destination[1], geometry.source[1]);
        assert_eq!(geometry.destination[4], geometry.source[4]);
        assert_ne!(geometry.destination[0], geometry.source[0]);
        assert_eq!(geometry.degrees[0], [1.0, 1.0, 1.0]);
        assert_eq!(geometry.degrees[1], [0.5, 0.5, 0.5]);
    }

    #[test]
    fn shoulder_width_swapping_endpoints_swaps_a_and_b_triplets() {
        let p1 = Vec2::new(20.0, 40.0);
        let p2 = Vec2::new(10.0, 20.0);
        let forward = shoulder_width_geometry(
            p1,
            p2,
            Vec2::new(0.0, 1.0),
            0.5,
            [1.0; 3],
            [0.0; 3],
            100,
            100,
        )
        .unwrap();
        let reverse = shoulder_width_geometry(
            p2,
            p1,
            Vec2::new(0.0, 1.0),
            0.5,
            [1.0; 3],
            [0.0; 3],
            100,
            100,
        )
        .unwrap();
        for k in 0..3 {
            assert!(vec_close(forward.source[k], reverse.source[k + 3]));
            assert!(vec_close(forward.source[k + 3], reverse.source[k]));
        }
    }

    #[test]
    fn shoulder_width_rejects_non_finite_landmarks() {
        assert_eq!(
            shoulder_width_geometry(
                Vec2::new(f32::NAN, 0.0),
                Vec2::new(1.0, 2.0),
                Vec2::new(1.0, 0.0),
                1.0,
                [1.0; 3],
                [1.0; 3],
                10,
                10,
            ),
            Err(Refusal::ValueOutOfRange("SG01 landmarks"))
        );
    }

    #[test]
    fn smooth_coord_is_clamped_cosine_window() {
        assert_eq!(smooth_coord(Vec2::new(0.5, 0.5)), 1.0);
        assert_eq!(smooth_coord(Vec2::new(0.0, 0.5)), 0.0);
        assert_eq!(smooth_coord(Vec2::new(1.0, 1.0)), 0.0);
        assert!((smooth_coord(Vec2::new(0.1, 0.5)) - 0.498).abs() < 1e-5);
        assert_eq!(smooth_coord(Vec2::new(0.25, 0.5)), 1.0);
        assert_eq!(smooth_coord(Vec2::new(-2.0, 3.0)), 0.0);
    }

    #[test]
    fn body_liquid_center_weight_is_one() {
        let source = [Vec2::new(0.5, 0.5)];
        let destination = [Vec2::new(0.6, 0.5)];
        let uv = body_liquid_uv(
            Vec2::new(0.5, 0.5),
            &source,
            &destination,
            &[1.0],
            1.0,
            1.0,
        )
        .unwrap();
        assert!(vec_close(uv, Vec2::new(0.4, 0.5)));
    }

    #[test]
    fn body_liquid_scale_is_squared() {
        let source = [Vec2::new(0.5, 0.5)];
        let destination = [Vec2::new(0.55, 0.5)];
        let single = body_liquid_uv(
            Vec2::new(0.5, 0.5),
            &source,
            &destination,
            &[1.0],
            1.0,
            1.0,
        )
        .unwrap();
        let double = body_liquid_uv(
            Vec2::new(0.5, 0.5),
            &source,
            &destination,
            &[1.0],
            2.0,
            1.0,
        )
        .unwrap();
        assert!(close(0.5 - single.x, 0.05));
        assert!(close(0.5 - double.x, 0.2));
        assert_eq!(single.y, 0.5);
        assert_eq!(double.y, 0.5);
    }

    #[test]
    fn body_liquid_uses_strict_radius_boundary() {
        let source = [Vec2::new(0.5, 0.5)];
        let destination = [Vec2::new(0.6, 0.5)];
        let at_boundary = body_liquid_uv(
            Vec2::new(0.6, 0.5),
            &source,
            &destination,
            &[1.0],
            1.0,
            1.0,
        )
        .unwrap();
        assert_eq!(at_boundary, Vec2::new(0.6, 0.5));
        let outside = body_liquid_uv(
            Vec2::new(0.7, 0.5),
            &source,
            &destination,
            &[1.0],
            1.0,
            1.0,
        )
        .unwrap();
        assert_eq!(outside, Vec2::new(0.7, 0.5));
    }

    #[test]
    fn body_liquid_zero_degree_is_identity() {
        let source = [Vec2::new(0.2, 0.3), Vec2::new(0.8, 0.7)];
        let destination = [Vec2::new(0.3, 0.4), Vec2::new(0.9, 0.8)];
        for uv in [Vec2::new(0.2, 0.3), Vec2::new(0.5, 0.5), Vec2::new(0.85, 0.75)] {
            let out = body_liquid_uv(uv, &source, &destination, &[0.0, 0.0], 1.0, 1.0).unwrap();
            assert_eq!(out, uv);
        }
    }

    #[test]
    fn body_liquid_aspect_correction_changes_radius() {
        let source = [Vec2::new(0.5, 0.5)];
        let destination = [Vec2::new(0.6, 0.6)];
        let isotropic = body_liquid_uv(
            Vec2::new(0.5, 0.6),
            &source,
            &destination,
            &[1.0],
            1.0,
            1.0,
        )
        .unwrap();
        let wide = body_liquid_uv(
            Vec2::new(0.5, 0.6),
            &source,
            &destination,
            &[1.0],
            1.0,
            2.0,
        )
        .unwrap();
        assert_ne!(isotropic, wide);
        assert!(wide.x.is_finite() && wide.y.is_finite());
    }

    #[test]
    fn body_liquid_rejects_shape_and_scalar_errors() {
        assert_eq!(
            body_liquid_uv(
                Vec2::new(0.5, 0.5),
                &[Vec2::new(0.0, 0.0)],
                &[Vec2::new(0.0, 0.0), Vec2::new(1.0, 1.0)],
                &[1.0],
                1.0,
                1.0,
            ),
            Err(Refusal::ShapeMismatch("BodyLiquid points"))
        );
        assert_eq!(
            body_liquid_uv(
                Vec2::new(0.5, 0.5),
                &[Vec2::new(0.0, 0.0)],
                &[Vec2::new(1.0, 1.0)],
                &[1.0],
                f32::NAN,
                1.0,
            ),
            Err(Refusal::ValueOutOfRange("BodyLiquid scalars"))
        );
        assert_eq!(
            body_liquid_uv(
                Vec2::new(0.5, 0.5),
                &[Vec2::new(0.0, 0.0)],
                &[Vec2::new(1.0, 1.0)],
                &[1.0],
                1.0,
                0.0,
            ),
            Err(Refusal::ValueOutOfRange("BodyLiquid scalars"))
        );
    }

    #[test]
    fn shoulder_geometry_feeds_body_liquid() {
        let geometry = shoulder_width_geometry(
            Vec2::new(30.0, 40.0),
            Vec2::new(10.0, 20.0),
            Vec2::new(1.0, 0.0),
            0.5,
            [1.0; 3],
            [0.2; 3],
            200,
            100,
        )
        .unwrap();
        let out = shoulder_body_liquid_uv(Vec2::new(0.5, 0.5), &geometry, 0.5, 1.0).unwrap();
        assert!(out.x.is_finite() && out.y.is_finite());
    }

    // ---- PF404 ----

    #[test]
    fn pf404_positive_branch_four_regions_match_hand_computation() {
        let p = skeleton();
        let g = buttock_region_geometry(&p, 0.5, 0, false).unwrap();
        assert!(g.four_regions);
        assert!(close(g.passes[0].scale, 0.7 * 0.5));
        assert!(close(g.passes[1].scale, 0.5 * 0.5));
        assert_eq!(g.passes[0].points[0].degree, 1.0);
        assert_eq!(g.passes[0].points[1].degree, 1.0);
        assert_eq!(g.passes[1].points[0].degree, 1.0);
        assert_eq!(g.passes[1].points[1].degree, 1.0);
        assert!(g.passes.iter().all(|pass| pass.points.iter().all(|point| point.active)));

        let p0 = midpoint(p[9], p[10]);
        let p1 = midpoint(p[11], p[12]);
        let m = midpoint(p[9], p[12]);
        assert!(vec_close(g.passes[0].points[0].source, p0));
        assert!(vec_close(g.passes[0].points[1].source, p1));
        assert!(vec_close(g.passes[1].points[0].source, p[8]));
        assert!(vec_close(g.passes[1].points[1].source, p[13]));

        let v0 = p0.sub(m);
        let len0 = v0.length();
        let expected0 = p0.add(rotate_region0_pos_30(v0.scale(1.0 / len0)).scale(5.0 * 0.2 * len0));
        let v1 = p1.sub(m);
        let len1 = v1.length();
        let expected1 = p1.add(rotate_region1_neg_30(v1.scale(1.0 / len1)).scale(5.0 * 0.2 * len1));
        assert!(vec_close(g.passes[0].points[0].destination, expected0));
        assert!(vec_close(g.passes[0].points[1].destination, expected1));

        let outer_mid = midpoint(p[8], p[13]);
        let expected2 = p[8].add(outer_mid.sub(p[8]).scale(7.0 * 0.25));
        let expected3 = p[13].add(outer_mid.sub(p[13]).scale(7.0 * 0.25));
        assert!(vec_close(g.passes[1].points[0].destination, expected2));
        assert!(vec_close(g.passes[1].points[1].destination, expected3));
    }

    #[test]
    fn pf404_negative_branch_switches_slots_and_scale() {
        let p = skeleton();
        let g = buttock_region_geometry(&p, -0.2, 0, false).unwrap();
        assert!(g.four_regions);
        assert!(close(g.passes[0].scale, 0.49 * -0.2));
        assert!(close(g.passes[1].scale, 0.35 * -0.2));
        assert_eq!(g.passes[0].points[0].degree, 1.0);
        assert_eq!(g.passes[1].points[0].degree, 1.0);
        let m = midpoint(p[10], p[11]);
        let p0 = midpoint(p[9], p[10]);
        let p1 = midpoint(p[11], p[12]);
        assert!(vec_close(g.passes[0].points[0].source, p0));
        assert!(vec_close(g.passes[0].points[1].source, p1));
        assert!(vec_close(g.passes[1].points[0].source, p[9]));
        assert!(vec_close(g.passes[1].points[1].source, p[12]));
        let v0 = p0.sub(m);
        let len0 = v0.length();
        let expected0 =
            p0.add(rotate_region0_pos_30(v0.scale(1.0 / len0)).scale(5.0 * 0.28 * len0));
        assert!(vec_close(g.passes[0].points[0].destination, expected0));
        let v1 = p1.sub(m);
        let len1 = v1.length();
        let expected1 =
            p1.add(rotate_region1_neg_30(v1.scale(1.0 / len1)).scale(5.0 * 0.28 * len1));
        assert!(vec_close(g.passes[0].points[1].destination, expected1));
    }

    #[test]
    fn pf404_two_region_branch_keeps_only_slots_one_and_three() {
        let p = skeleton();
        let g = buttock_region_geometry(&p, 0.5, 1, true).unwrap();
        assert!(!g.four_regions);
        assert!(!g.passes[0].points[0].active);
        assert!(g.passes[0].points[1].active);
        assert!(!g.passes[1].points[0].active);
        assert!(g.passes[1].points[1].active);
        assert_eq!(g.passes[0].points[0].source, Vec2::new(0.0, 0.0));
        assert_eq!(g.passes[1].points[0].source, Vec2::new(0.0, 0.0));
        assert!(vec_close(g.passes[0].points[1].source, midpoint(p[11], p[12])));
        assert!(vec_close(g.passes[1].points[1].source, p[13]));
    }

    #[test]
    fn pf404_person_type_three_forces_four_regions() {
        let p = skeleton();
        let g = buttock_region_geometry(&p, 0.5, 3, true).unwrap();
        assert!(g.four_regions);
        assert!(g.passes[0].points[0].active);
        assert!(g.passes[1].points[0].active);
    }

    #[test]
    fn pf404_rotations_match_exact_0x3cb0a80_matrices() {
        let unit_x = Vec2::new(1.0, 0.0);
        let unit_y = Vec2::new(0.0, 1.0);
        // 区域 0 (+30°): [[0.866, -0.5], [0.5, 0.866]]
        let r0_x = rotate_region0_pos_30(unit_x);
        let r0_y = rotate_region0_pos_30(unit_y);
        assert_eq!(r0_x.x.to_bits(), 0x3f5d_b22d);
        assert_eq!(r0_x.y.to_bits(), 0x3f00_0000);
        assert_eq!(r0_y.x.to_bits(), (-0.5f32).to_bits());
        assert_eq!(r0_y.y.to_bits(), 0x3f5d_b22d);
        // 区域 1 (-30°): [[0.866, 0.5], [-0.5, 0.866]]
        let r1_x = rotate_region1_neg_30(unit_x);
        let r1_y = rotate_region1_neg_30(unit_y);
        assert_eq!(r1_x.x.to_bits(), 0x3f5d_b22d);
        assert_eq!(r1_x.y.to_bits(), (-0.5f32).to_bits());
        assert_eq!(r1_y.x.to_bits(), 0x3f00_0000);
        assert_eq!(r1_y.y.to_bits(), 0x3f5d_b22d);
    }

    #[test]
    fn pf404_rejects_bad_inputs() {
        let p = skeleton();
        assert_eq!(
            buttock_region_geometry(&p[..13], 0.5, 0, false),
            Err(Refusal::MissingInput("PF404 vendor skeleton"))
        );
        assert_eq!(
            buttock_region_geometry(&p, f32::NAN, 0, false),
            Err(Refusal::ValueOutOfRange("PF404 degree"))
        );
        let mut bad = p.clone();
        bad[11] = Vec2::new(f32::NAN, 0.5);
        assert_eq!(
            buttock_region_geometry(&bad, 0.5, 0, false),
            Err(Refusal::ValueOutOfRange("PF404 vendor skeleton"))
        );
        let mut degenerate = p;
        degenerate[10] = degenerate[9];
        degenerate[12] = degenerate[9];
        assert_eq!(
            buttock_region_geometry(&degenerate, 0.5, 0, false),
            Err(Refusal::MissingInput("PF404 region basis"))
        );
    }

    #[test]
    fn pf404_degree_zero_has_zero_scale() {
        let p = skeleton();
        let g = buttock_region_geometry(&p, 0.0, 0, false).unwrap();
        assert_eq!(g.passes[0].scale, 0.0);
        assert_eq!(g.passes[1].scale, 0.0);
        assert_eq!(g.passes[0].points[0].degree, 1.0);
        assert_eq!(g.passes[1].points[0].degree, 1.0);
    }

    // ---- PF405 ----

    #[test]
    fn pf405_shape_constants_and_index_count() {
        assert_eq!(BODY_REFINE_ROWS, 12);
        assert_eq!(BODY_REFINE_COLS, 2);
        assert_eq!(BODY_REFINE_VERTEX_COUNT, 24);
        assert_eq!(BODY_REFINE_INDEX_COUNT, 102);
        assert_eq!(body_refine_index_count(12, 2).unwrap(), 102);
        assert_eq!(
            body_refine_index_count(2, 2),
            Err(Refusal::ValueOutOfRange("PF405_MESH_SHAPE"))
        );
        assert_eq!(
            body_refine_index_count(12, 1),
            Err(Refusal::ValueOutOfRange("PF405_MESH_SHAPE"))
        );
    }

    #[test]
    fn pf405_mesh_has_24_vertices_and_102_indices() {
        let mesh = body_refine_vertices(
            100,
            400,
            [50.0, 200.0],
            [10.0, 20.0],
            [30.0, 40.0],
            [25.0, 35.0],
        )
        .unwrap();
        assert_eq!((mesh.rows, mesh.cols), (12, 2));
        assert_eq!(mesh.vertices.len(), BODY_REFINE_VERTEX_COUNT);
        assert_eq!(mesh.indices.len(), BODY_REFINE_INDEX_COUNT);
    }

    #[test]
    fn pf405_index_topology_is_valid() {
        let indices = body_refine_indices(12, 2).unwrap();
        assert_eq!(indices.len(), 102);
        assert_eq!(indices.len() % 3, 0);
        assert_eq!(indices.len() / 3, 34);
        assert!(indices.iter().all(|index| (*index as usize) < BODY_REFINE_VERTEX_COUNT));
        for triangle in indices.chunks_exact(3) {
            assert_ne!(triangle[0], triangle[1]);
            assert_ne!(triangle[0], triangle[2]);
            assert_ne!(triangle[1], triangle[2]);
        }
        // Fan prefix: rows 2..=11 -> (0, 2, 4), (0, 4, 6), ...
        assert_eq!(&indices[0..6], &[0, 2, 4, 0, 4, 6]);
        // Ribbon start and wrap tail.
        assert_eq!(&indices[30..36], &[0, 2, 1, 2, 1, 3]);
        assert_eq!(&indices[96..102], &[22, 0, 23, 0, 23, 1]);
    }

    #[test]
    fn pf405_row_zero_matches_hand_computed_constants() {
        let mesh = body_refine_vertices(
            100,
            400,
            [50.0, 200.0],
            [10.0, 20.0],
            [30.0, 40.0],
            [25.0, 35.0],
        )
        .unwrap();
        let radial0 = 0.7f32 * 50.0f32.sqrt();
        let radial1 = 5.5f32 * radial0;
        let ox = 36.25f32;
        let oy = 116.25f32;
        let col0_x = 0.8 * (radial0 * 0.8) + ox;
        let col1_x = 0.3 * (radial1 * 0.8) + ox;
        assert!((mesh.vertices[0][0] - col0_x / 100.0).abs() < 1e-6);
        assert!((mesh.vertices[0][1] - oy / 400.0).abs() < 1e-6);
        assert!((mesh.vertices[1][0] - col1_x / 100.0).abs() < 1e-6);
        assert!((mesh.vertices[1][1] - oy / 400.0).abs() < 1e-6);
    }

    #[test]
    fn pf405_coincident_records_collapse_without_nan() {
        let same = [25.0, 35.0];
        let mesh = body_refine_vertices(100, 400, [50.0, 200.0], same, same, same).unwrap();
        assert!(mesh.vertices.iter().flatten().all(|value| value.is_finite()));
        let ox = (50.0 + (same[0] + same[0]) * 0.5) * 0.5;
        let oy = (200.0 + (same[1] + same[1]) * 0.5) * 0.5;
        for vertex in &mesh.vertices {
            assert!((vertex[0] - ox / 100.0).abs() < 1e-6);
            assert!((vertex[1] - oy / 400.0).abs() < 1e-6);
        }
    }

    #[test]
    fn pf405_rejects_bad_sizes_and_landmarks() {
        assert_eq!(
            body_refine_vertices(
                0,
                400,
                [50.0, 200.0],
                [10.0, 20.0],
                [30.0, 40.0],
                [25.0, 35.0]
            ),
            Err(Refusal::ShapeMismatch("image"))
        );
        assert_eq!(
            body_refine_vertices(
                100,
                400,
                [f32::NAN, 200.0],
                [10.0, 20.0],
                [30.0, 40.0],
                [25.0, 35.0]
            ),
            Err(Refusal::ValueOutOfRange("PF405_LANDMARKS"))
        );
    }

    #[test]
    fn pf405_fragment_degree_and_identity() {
        assert_eq!(body_refine_fragment_degree(255), 0.0);
        assert_eq!(body_refine_fragment_degree(0), 1.0);
        assert!((body_refine_fragment_degree(128) - (1.0 - 128.0 / 255.0)).abs() < 1e-7);

        let source = [0.2f32, 0.3];
        let target = [0.6f32, 0.7];
        assert_eq!(body_refine_fragment_uv(source, target, 255).unwrap(), source);
        let full = body_refine_fragment_uv(source, target, 0).unwrap();
        assert!(close(full[0], target[0]) && close(full[1], target[1]));
        let half = body_refine_fragment_uv(source, target, 128).unwrap();
        let degree = body_refine_fragment_degree(128);
        assert!(close(half[0], source[0] + (target[0] - source[0]) * degree));
        assert_eq!(
            body_refine_fragment_uv([f32::NAN, 0.0], target, 0),
            Err(Refusal::ValueOutOfRange("PF405_UV"))
        );
    }

    // ---- SG07–SG10 ----

    #[test]
    fn sub_control_ranges_match_config() {
        assert_eq!(RIGHT_SHOULDER_RENDER_DEGREE_RANGES, [0.0, 0.4]);
        assert_eq!(SWAN_NECK_RENDER_DEGREE_RANGES, [0.0, 0.55]);
        assert_eq!(SLIM_LEG_RENDER_DEGREE_RANGES, [-1.0, 1.0]);
    }

    #[test]
    fn lengthen_leg_shared_stage_is_linear_interpolation() {
        assert_eq!(lengthen_leg_shared_stage(1.25, 0.0, 0.0), 0.0);
        assert_eq!(lengthen_leg_shared_stage(1.25, 0.0, 1.0), 1.25);
        assert_eq!(lengthen_leg_shared_stage(1.25, 0.0, 0.5), 0.625);
        assert_eq!(lengthen_leg_shared_stage(0.0, 1.0, 0.25), 0.75);
    }
}
