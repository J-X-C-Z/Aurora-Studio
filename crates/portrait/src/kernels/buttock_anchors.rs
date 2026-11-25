//! PF404 (`ButtockRefine`) 臀部区域控制点的静态回收（CPU 几何生产器）。
//!
//! 本模块只做**几何生产**：从 vendor 骨架点计算 PF404 的六个控制点及其形变区域。
//! 它不接路由、不采样像素、不写回；缺失/退化输入一律返回具名 `Refusal`。
//!
//! ## 结论摘要（2026-11-25 静态取证）
//!
//! main-tree `body_pose_adapter::six_point_anchors(404)` 的注释称 B6/B7/B8 是
//! “关键点定高度 + 分割定边界”合成的区域锚点。本轮对 `pix-worker.exe`
//! （SHA-256 `509D2293…96B84`，ImageBase `0x140000000`）的反汇编显示：
//!
//! - `EffectButtockRefineCell` 的 vtable 为 `0x8E56800`，`Process` 在
//!   slot 15 = `0x3CB0360`；
//! - 控制点生产器是 `0x3CB0A80`，由 `0x3CB06D0` 调用；
//! - 该函数**只读取骨架点数组**（8 字节 `Vec2`，`[rdx + idx*8]`），
//!   固定使用索引 **8,9,10,11,12,13**，不读取任何分割掩码/轮廓纹理，
//!   也不做像素或掩码采样。
//!
//! 因此 `PROVEN` 的是：PF404 的六个控制点来自 vendor 骨架槽 8..13，经
//! 中点 + 固定旋转/外推常量合成；`INFERRED` 的是这 6 槽属于 vendor
//! 69 点 `oriSkeleton`（见 `body_shader_abi.rs`）；**槽 8..13 的解剖命名
//! 仍为 `NOT_FOUND`**，禁止套用 COCO-17 / Halpe-26 / WholeBody-133 的顺序。
//!
//! ## 消费端（`0x3CB0360` → `GLFilterBodyWarp`）
//!
//! `0x3CB07B2` 与 `0x3CB0912` 两次调用 `[*(cell+0x280)+0x120]` /
//! `[*(cell+0x288)+0x120]`（实现 `0x5713BB0`，字符串
//! `"GLFilterBodyWarp::FilterToFBO"` @ `0x9529E82`），栈参 `2`
//! （`0x3CB07A2`/`0x3CB0902`）。两次 `glUniform2fv` 各上传 **2** 个
//! 源/目标点（`0x5713E14`）。所以 vendor 实际是“两趟各 2 点”，
//! `BodyLiquid.frag` 的 `POINT_CONTAINER_COUNT 6` 只是容量上限，
//! 不是 PF404 真实控制点数。main-tree `six_point_anchors(404)` 的“六点”
//! 前提与二进制不符。
//!
//! ## 已证算术（`0x3CB0A80`）
//!
//! 输入：`p[0..69]`（`Vec2`），`degree`（`0x3CB0360` 由 `0x5711D20` 两点曲线
//! `[0,1] -> [0,0.5]` 重映射），`person_type`（`[node+0x10]`），
//! `edge_protection`（`[node+0x28]` 字节）。`s = 0.7 * degree`。
//!
//! - 方向分支（`ucomiss s, 0` @ `0x3CB0ACD`，`jae` @ `0x3CB0AD0`）：
//!   - `s >= 0`：`(esi,r11,r10,rax) = (9,12,8,13)`，`k_a=0.2`，`k_b=0.25`
//!     （常量 `0x8BFC2A4`, `0x8BA75C8`）；
//!   - `s < 0`：`(10,11,9,12)`，`k_a=0.28`，`k_b=0.1`
//!     （`0x8E56790`, `0x8B99E20`）。
//! - `M = 0.5*(p[esi] + p[r11])`（`0x3CB0B59`）。
//! - 分支（`0x3CB0B84 jne`）：`(!edge_protection || person_type == 3)` → 4 区；
//!   否则 → 2 区（`0x3CB0B8A`）。
//! - 4 区（`0x3CB0C4C` 起）：
//!   - `src[0] = C0 = 0.5*(p9+p10)`，`dst[0] = C0 + 5*k_a*R(-150°)(M-C0)`；
//!   - `src[1] = C1 = 0.5*(p11+p12)`，`dst[1] = C1 + 5*k_a*R(+150°)(M-C1)`；
//!   - `src[2] = p[r10]`，`src[3] = p[rax]`；
//!   - `dst[2] = p[r10] + 3.5*k_b*(p[rax]-p[r10])`，
//!     `dst[3] = p[rax] + 3.5*k_b*(p[r10]-p[rax])`（`0x3CB0E1C`，常量 `7.0`）。
//! - 2 区（`0x3CB0B8A`）：只填 `src[1]/dst[1]` 与 `src[3]/dst[3]`，其余为 `(0,0)`
//!   且 `active = false`。
//! - 尺度（写 `cell+0x2D0`，`0x3CB0AC3`）：`[1.0,1.0,s,s]`；`s<0` 时
//!   `cell+0x2D8..0x2E0 = 0.49*degree`（`0x3CB0AD2`，`0x8DDD8B0`）。
//!   写入值是 `PROVEN`；把 `+0x2D8/+0x2DC` 解释成两趟 GL 的 `u_scale` 是
//!   `INFERRED`（`GLFilterBodyWarp` 的 uniform 名在二进制里是加密串，静态读不出）。
//! - 旋转常量 `0.866 = √3/2`、`0.5` 位于 `0x8E567A0..0x8E567E0`；消费端
//!   `BodyLiquid.frag` 的 `u_scale*u_scale` 是既有已证律，不在本模块重证。
//!
//! ## 未闭合边界
//!
//! - 槽 8..13 的解剖命名：`NOT_FOUND`（见 `body_pose_adapter::SlotTable` 文档）。
//! - 骨架数组与 `oriSkeleton` 同一个对象：`INFERRED`。几何生产器记录类
//!   （`0x1EFB550` 读 `+0x50/+0x58` 向量）与 `loadExtJsonParam` 写出的 0x88
//!   字节骨架记录在布局上不一致，静态无法直接绑定。
//! - 真实照片/GPU 执行：`LIVE_ONLY`，本轮不触碰。
//!
//! ## 拒绝标签
//!
//! - `MissingInput("PF404 vendor skeleton")`：骨架数组不足 14 点（索引越界）。
//! - `MissingInput("PF404 region basis")`：`M` 与区域基点重合，方向不可定义；
//!   vendor 会得到 `inf/NaN`，这里改为拒绝而不是合成。
//! - `ValueOutOfRange("PF404 vendor skeleton")`：任一被引用点非有限。
//! - `ValueOutOfRange("PF404 degree")`：`degree` 非有限。

use crate::kernels::body_geometry::Vec2;
use crate::Refusal;

/// vendor 骨架 `oriSkeleton` 的点数（`loadExtJsonParam` 断言 `0x228 == 69*8`，
/// `pix-worker.exe` RVA `0x1A96B10..0x1A96B26`）。
pub const VENDOR_SKELETON_POINT_COUNT: usize = 69;

/// PF404 控制点直接读取的 vendor 骨架槽。
///
/// **PROVEN**（`0x3CB0A80` 的 `[rdx + idx*8]` 索引）。语义命名 `NOT_FOUND`。
pub const PF404_VENDOR_SKELETON_SLOTS: [usize; 6] = [8, 9, 10, 11, 12, 13];

/// 需要的最小骨架长度（最大索引 13 + 1）。
const REQUIRED_POINTS: usize = 14;

/// 半径/外推常量。全部来自 `0x3CB0A80` 的 RIP 相对常量池。
const SCALE_TO_DEGREE: f32 = 0.7; // 0x8E56780
const KA_POSITIVE: f32 = 0.2; // 0x8BFC2A4
const KA_NEGATIVE: f32 = 0.28; // 0x8E56790
const KB_POSITIVE: f32 = 0.25; // 0x8BA75C8
const KB_NEGATIVE: f32 = 0.1; // 0x8B99E20
const REGION_SCALE: f32 = 5.0; // 0x8E567C0
const EXTRAPOLATE_SCALE: f32 = 7.0; // 0x8E567F0
const ROT_COS150: f32 = 0.866_025_4; // √3/2，0x8E567E0 / 0x8E567B0
const ROT_SIN150: f32 = 0.5; // 0x8E567D0 / 0x8E567A0

/// PF404 的四条区域；`active[i] == false` 表示 vendor 在该分支下不写该槽。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ButtockRegions {
    /// `cell+0x290..0x2B0`（4 个 `Vec2`）。
    pub source: [Vec2; 4],
    /// `cell+0x2B0..0x2D0`（4 个 `Vec2`）。
    pub destination: [Vec2; 4],
    /// vendor 是否写入该区域（2 区分支只写 1/3）。
    pub active: [bool; 4],
    /// `cell+0x2D0..0x2D8`，两趟 GL 的 `u_degrees` 基值。
    pub degrees: [f32; 2],
    /// `cell+0x2D8..0x2E0`，两趟 GL 的 `u_scale`（下游再做平方）。
    pub scale: [f32; 2],
    /// 是否走 4 区分支（`!edge_protection || person_type == 3`）。
    pub four_regions: bool,
}

#[inline]
fn mid(a: Vec2, b: Vec2) -> Vec2 {
    Vec2::new(0.5 * (a.x + b.x), 0.5 * (a.y + b.y))
}

/// 旋转 +150°：`(-c*dx - s*dy, s*dx - c*dy)`（`0x3CB0BF5..0x3CB0C05`）。
#[inline]
fn rotate_pos150(d: Vec2) -> Vec2 {
    Vec2::new(
        -ROT_COS150 * d.x - ROT_SIN150 * d.y,
        ROT_SIN150 * d.x - ROT_COS150 * d.y,
    )
}

/// 旋转 -150°：`(-c*dx + s*dy, -s*dx - c*dy)`（`0x3CB0CD3..0x3CB0CE7`）。
#[inline]
fn rotate_neg150(d: Vec2) -> Vec2 {
    Vec2::new(
        -ROT_COS150 * d.x + ROT_SIN150 * d.y,
        -ROT_SIN150 * d.x - ROT_COS150 * d.y,
    )
}

fn check_skeleton(skeleton: &[Vec2]) -> Result<(), Refusal> {
    if skeleton.len() < REQUIRED_POINTS {
        return Err(Refusal::MissingInput("PF404 vendor skeleton"));
    }
    for &index in &PF404_VENDOR_SKELETON_SLOTS {
        let p = skeleton[index];
        if !p.x.is_finite() || !p.y.is_finite() {
            return Err(Refusal::ValueOutOfRange("PF404 vendor skeleton"));
        }
    }
    Ok(())
}

/// PF404 的六个控制点：vendor 骨架槽 8..13 的原始位置。
///
/// 这是 `0x3CB0A80` 实际读取的六个槽（`[rdx + idx*8]`），不经过任何分割掩码。
/// 返回顺序固定为 [`PF404_VENDOR_SKELETON_SLOTS`]。
///
/// 缺输入（不足 14 点）返回 `MissingInput("PF404 vendor skeleton")`；
/// 非有限坐标返回 `ValueOutOfRange("PF404 vendor skeleton")`。
pub fn buttock_control_points(skeleton: &[Vec2]) -> Result<[Vec2; 6], Refusal> {
    check_skeleton(skeleton)?;
    Ok([
        skeleton[8], skeleton[9], skeleton[10], skeleton[11], skeleton[12], skeleton[13],
    ])
}

/// 按 `0x3CB0A80` 已证算术生成 PF404 的形变区域。
///
/// - `skeleton`：vendor 骨架点（至少 14 点，读取 8..13）。
/// - `degree`：`0x5711D20` 重映射后的度数（`PoseRefine/effect.json` 中 PF404 为
///   `RenderDegreeRanges [0.0, 0.5]`，但二进制也处理负度数，本函数沿用全部已证分支）。
/// - `person_type`：`[node+0x10]`（二进制中与 `3` 比较）。
/// - `edge_protection`：`[node+0x28]` 字节（真值取反后参与分支）。
///
/// 退化输入（区域方向长度为 0）返回 `MissingInput("PF404 region basis")`，
/// 不产生 `NaN/inf`，也不 clamp。
pub fn buttock_region_geometry(
    skeleton: &[Vec2],
    degree: f32,
    person_type: i32,
    edge_protection: bool,
) -> Result<ButtockRegions, Refusal> {
    check_skeleton(skeleton)?;
    if !degree.is_finite() {
        return Err(Refusal::ValueOutOfRange("PF404 degree"));
    }

    let s = SCALE_TO_DEGREE * degree;
    let (esi, r11, r10, rax, ka, kb) = if s >= 0.0 {
        (9usize, 12usize, 8usize, 13usize, KA_POSITIVE, KB_POSITIVE)
    } else {
        (10, 11, 9, 12, KA_NEGATIVE, KB_NEGATIVE)
    };

    let m = mid(skeleton[esi], skeleton[r11]);
    let four_regions = !edge_protection || person_type == 3;

    let mut regions = ButtockRegions {
        source: [Vec2::new(0.0, 0.0); 4],
        destination: [Vec2::new(0.0, 0.0); 4],
        active: [false; 4],
        degrees: [1.0, 1.0],
        scale: if s >= 0.0 { [s, s] } else { [s * SCALE_TO_DEGREE, s * SCALE_TO_DEGREE] },
        four_regions,
    };

    let p0 = skeleton[r10];
    let p3 = skeleton[rax];
    let outer_mid = mid(p0, p3);

    // 区域 3：外推，两支都写。
    regions.source[3] = p3;
    regions.destination[3] = Vec2::new(
        p3.x + EXTRAPOLATE_SCALE * kb * (outer_mid.x - p3.x),
        p3.y + EXTRAPOLATE_SCALE * kb * (outer_mid.y - p3.y),
    );
    regions.active[3] = true;

    if four_regions {
        let c0 = mid(skeleton[9], skeleton[10]);
        let c1 = mid(skeleton[11], skeleton[12]);

        let d0 = Vec2::new(m.x - c0.x, m.y - c0.y);
        if d0.x == 0.0 && d0.y == 0.0 {
            return Err(Refusal::MissingInput("PF404 region basis"));
        }
        let r0 = rotate_neg150(d0);
        regions.source[0] = c0;
        regions.destination[0] = Vec2::new(
            c0.x + REGION_SCALE * ka * r0.x,
            c0.y + REGION_SCALE * ka * r0.y,
        );
        regions.active[0] = true;

        let d1 = Vec2::new(m.x - c1.x, m.y - c1.y);
        if d1.x == 0.0 && d1.y == 0.0 {
            return Err(Refusal::MissingInput("PF404 region basis"));
        }
        let r1 = rotate_pos150(d1);
        regions.source[1] = c1;
        regions.destination[1] = Vec2::new(
            c1.x + REGION_SCALE * ka * r1.x,
            c1.y + REGION_SCALE * ka * r1.y,
        );
        regions.active[1] = true;

        regions.source[2] = p0;
        regions.destination[2] = Vec2::new(
            p0.x + EXTRAPOLATE_SCALE * kb * (outer_mid.x - p0.x),
            p0.y + EXTRAPOLATE_SCALE * kb * (outer_mid.y - p0.y),
        );
        regions.active[2] = true;
    } else {
        let c1 = mid(skeleton[11], skeleton[12]);
        let d1 = Vec2::new(m.x - c1.x, m.y - c1.y);
        if d1.x == 0.0 && d1.y == 0.0 {
            return Err(Refusal::MissingInput("PF404 region basis"));
        }
        let r1 = rotate_pos150(d1);
        regions.source[1] = c1;
        regions.destination[1] = Vec2::new(
            c1.x + REGION_SCALE * ka * r1.x,
            c1.y + REGION_SCALE * ka * r1.y,
        );
        regions.active[1] = true;
    }

    Ok(regions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skeleton() -> Vec<Vec2> {
        (0..VENDOR_SKELETON_POINT_COUNT)
            .map(|i| Vec2::new(0.01 * i as f32, 0.02 * i as f32))
            .collect()
    }

    #[test]
    fn control_points_are_vendor_slots_8_to_13() {
        let p = skeleton();
        let cps = buttock_control_points(&p).unwrap();
        for (k, slot) in PF404_VENDOR_SKELETON_SLOTS.iter().enumerate() {
            assert_eq!(cps[k], p[*slot], "slot {slot}");
        }
    }

    #[test]
    fn control_points_reject_short_skeleton() {
        let p = skeleton();
        assert_eq!(
            buttock_control_points(&p[..13]),
            Err(Refusal::MissingInput("PF404 vendor skeleton"))
        );
    }

    #[test]
    fn control_points_reject_non_finite() {
        let mut p = skeleton();
        p[11] = Vec2::new(f32::NAN, 0.5);
        assert_eq!(
            buttock_control_points(&p),
            Err(Refusal::ValueOutOfRange("PF404 vendor skeleton"))
        );
    }

    #[test]
    fn four_region_geometry_matches_proven_formula() {
        // 构造非退化、各槽互不相同的点。
        let p: Vec<Vec2> = (0..VENDOR_SKELETON_POINT_COUNT)
            .map(|i| Vec2::new(0.10 + 0.01 * i as f32, 0.20 + 0.02 * i as f32))
            .collect();
        let degree = 0.5; // s = 0.35 >= 0 => slots (9,12,8,13), ka=0.2, kb=0.25
        let g = buttock_region_geometry(&p, degree, 0, false).unwrap();

        assert!(g.four_regions);
        assert_eq!(g.active, [true, true, true, true]);
        assert_eq!(g.degrees, [1.0, 1.0]);
        assert_eq!(g.scale, [0.35, 0.35]);

        let c0 = mid(p[9], p[10]);
        let c1 = mid(p[11], p[12]);
        let m = mid(p[9], p[12]);
        assert_eq!(g.source[0], c0);
        assert_eq!(g.source[1], c1);
        assert_eq!(g.source[2], p[8]);
        assert_eq!(g.source[3], p[13]);

        let r0 = rotate_neg150(Vec2::new(m.x - c0.x, m.y - c0.y));
        assert!((g.destination[0].x - (c0.x + 5.0 * 0.2 * r0.x)).abs() < 1e-6);
        assert!((g.destination[0].y - (c0.y + 5.0 * 0.2 * r0.y)).abs() < 1e-6);

        let r1 = rotate_pos150(Vec2::new(m.x - c1.x, m.y - c1.y));
        assert!((g.destination[1].x - (c1.x + 5.0 * 0.2 * r1.x)).abs() < 1e-6);
        assert!((g.destination[1].y - (c1.y + 5.0 * 0.2 * r1.y)).abs() < 1e-6);

        let om = mid(p[8], p[13]);
        assert!((g.destination[2].x - (p[8].x + 7.0 * 0.25 * (om.x - p[8].x))).abs() < 1e-6);
        assert!((g.destination[3].x - (p[13].x + 7.0 * 0.25 * (om.x - p[13].x))).abs() < 1e-6);
    }

    #[test]
    fn two_region_branch_leaves_slots_inactive() {
        let p: Vec<Vec2> = (0..VENDOR_SKELETON_POINT_COUNT)
            .map(|i| Vec2::new(0.10 + 0.01 * i as f32, 0.20 + 0.02 * i as f32))
            .collect();
        // edge_protection = true 且 person_type != 3 => 2 区分支。
        let g = buttock_region_geometry(&p, 0.5, 1, true).unwrap();
        assert!(!g.four_regions);
        assert_eq!(g.active, [false, true, false, true]);
        assert_eq!(g.source[0], Vec2::new(0.0, 0.0));
        assert_eq!(g.source[2], Vec2::new(0.0, 0.0));
    }

    #[test]
    fn negative_degree_switches_slot_windows() {
        let p: Vec<Vec2> = (0..VENDOR_SKELETON_POINT_COUNT)
            .map(|i| Vec2::new(0.10 + 0.01 * i as f32, 0.20 + 0.02 * i as f32))
            .collect();
        let g = buttock_region_geometry(&p, -0.2, 0, false).unwrap();
        // s = -0.14 < 0 => slots (10,11,9,12), ka=0.28, kb=0.1
        assert_eq!(g.source[2], p[9]);
        assert_eq!(g.source[3], p[12]);
        assert!(g.scale[0] < 0.0);
        assert!((g.scale[0] - (-0.14 * 0.7)).abs() < 1e-6);
    }

    #[test]
    fn degenerate_basis_is_refused() {
        let mut p = skeleton();
        // 让 M == C0：p9 == p10，且 p12 == p9（M = mid(p9,p12) = p9 = C0）。
        p[10] = p[9];
        p[12] = p[9];
        assert_eq!(
            buttock_region_geometry(&p, 0.5, 0, false),
            Err(Refusal::MissingInput("PF404 region basis"))
        );
    }

    #[test]
    fn non_finite_degree_is_refused() {
        let p = skeleton();
        assert_eq!(
            buttock_region_geometry(&p, f32::NAN, 0, false),
            Err(Refusal::ValueOutOfRange("PF404 degree"))
        );
    }
}
