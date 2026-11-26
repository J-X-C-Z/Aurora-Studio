//! 原厂「丰盈（FacePlump）」最终像素合成模块（纯 Rust，零外部依赖，零 `unsafe`）。
//!
//! 本模块基于 `C:\Program Files\PixCake\pixcake_2\pixcake.exe`（SHA-256:
//! `A84519DA489BC52E4E914AF7880EDE43E3CE58992657723D02197161832464C7`）只读静态反汇编
//! 结果实现，对应 `face_cache`（vtable `0x09444930`）`slot[1]`（`0x06c1e730`）像素合成链路：
//!
//! 1. **17 槽参数预乘映射**（`EffectFacePlump` `0x05e10c80..0x05e14a70` → `ctx + 0x68..+0x188`）：
//!    将 4 个总控乘子与 13 个滑杆参数映射为 `vec9[0..9]` 与 `vec4[0..4]`。
//! 2. **Mask 羽化**：`StackBlur(radius = 10)`（对应 `0x06c26830` → `0x0260e810` / `0x0260e950`，
//!    窗口 `2r + 1`、除数 `(r + 1)^2`）。
//! 3. **2× 残差叠加**：`0x06c27030`，以 `128.0` 为零点放大 2 倍后截断至整数并按
//!    `alpha = (w / 255) * (s / 255)` 与原图混合，保持 Alpha 通道不变。
//! 4. **双 `3×256` 平面 LUT 分区提亮**：`0x06c29f90`，分别使用 `.rdata` RVA `0x09444110`
//!    （额头强提亮）与 `0x09444410`（面中/鼻基底/下巴/眉区柔提亮），按 `alpha` 混合并保持 Alpha 不变。

#![forbid(unsafe_code)]

const INV_255: f32 = 1.0_f32 / 255.0_f32;

/// 丰盈阶段一/二/三默认 mask 羽化半径（对应 `0x06c26830(..., radius = 10, ...)`）。
pub const PLUMP_STACK_BLUR_RADIUS: usize = 10;

/// 额头强提亮 LUT（提取自 `pixcake.exe` `.rdata` RVA `0x09444110`，长度 `768 = 3 × 256` 字节）。
///
/// SHA-256: `9DE8CE23E1374F466DD9E113729615D0ACA9B24A0124B508E6622247B79892E5`
///
/// ### 平面布局与 RGBA 通道映射
/// 原厂内部图像格式为 BGRA/BGR（`0x401` / `0x301`），LUT 为 `3 × 256` 平面字节表：
/// - `0..256`（`+0x000..+0x0FF`）：**B 通道**查找表（对应 RGBA 像素的 `rgba[idx + 2]`）
/// - `256..512`（`+0x100..+0x1FF`）：**G 通道**查找表（对应 RGBA 像素的 `rgba[idx + 1]`）
/// - `512..768`（`+0x200..+0x2FF`）：**R 通道**查找表（对应 RGBA 像素的 `rgba[idx + 0]`）
///
/// 例如输入灰度 `128` 时，原厂 BGR 查表结果为 `[148, 145, 145]`，对应 RGBA 的 RGB 输出为 `[145, 145, 148]`。
#[rustfmt::skip]
pub const PLUMP_BRIGHTEN_LUT_FOREHEAD: [u8; 768] = [
    // Plane 0: B channel (0..256)
    0, 1, 2, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 18, 19,
    20, 21, 22, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 37, 38,
    39, 40, 41, 42, 44, 45, 47, 48, 49, 51, 52, 53, 54, 55, 56, 58,
    59, 60, 61, 62, 63, 65, 66, 68, 69, 70, 71, 73, 74, 75, 76, 77,
    78, 79, 81, 82, 83, 84, 85, 86, 87, 89, 90, 91, 92, 93, 94, 95,
    96, 98, 99, 100, 101, 102, 103, 104, 105, 107, 108, 109, 110, 111, 112, 113,
    114, 115, 116, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 131,
    132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
    148, 149, 150, 151, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164,
    165, 166, 167, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
    180, 181, 182, 183, 184, 185, 186, 186, 187, 188, 189, 190, 191, 192, 193, 194,
    195, 195, 196, 197, 198, 199, 200, 201, 202, 202, 203, 204, 205, 205, 206, 207,
    207, 208, 209, 210, 211, 212, 212, 213, 214, 215, 216, 216, 217, 218, 219, 220,
    220, 220, 221, 222, 223, 224, 224, 225, 226, 227, 228, 228, 229, 230, 231, 231,
    232, 233, 234, 235, 235, 236, 237, 238, 238, 239, 240, 241, 242, 242, 243, 244,
    245, 245, 246, 247, 248, 248, 249, 249, 250, 250, 251, 252, 253, 253, 254, 255,
    // Plane 1: G channel (256..512)
    0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18,
    19, 20, 21, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 36, 37,
    38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 56,
    57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74,
    75, 76, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92,
    93, 95, 96, 97, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128,
    129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    145, 146, 147, 148, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
    162, 163, 164, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
    177, 178, 179, 180, 181, 182, 183, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 192, 193, 194, 195, 196, 197, 198, 199, 199, 200, 201, 202, 203, 204, 205,
    205, 206, 207, 208, 209, 210, 210, 211, 212, 213, 214, 214, 215, 216, 217, 218,
    219, 219, 220, 221, 222, 223, 223, 224, 225, 226, 227, 227, 228, 229, 230, 230,
    231, 232, 233, 234, 234, 235, 236, 237, 237, 238, 239, 240, 241, 241, 242, 243,
    244, 244, 245, 246, 247, 247, 248, 249, 250, 250, 251, 252, 253, 253, 254, 255,
    // Plane 2: R channel (512..768)
    0, 1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18,
    19, 20, 21, 23, 24, 25, 26, 27, 29, 30, 31, 32, 33, 34, 36, 37,
    38, 39, 40, 41, 43, 44, 45, 46, 47, 49, 50, 51, 52, 53, 54, 56,
    57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74,
    75, 76, 78, 79, 80, 81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92,
    93, 95, 96, 97, 98, 99, 100, 101, 102, 104, 105, 106, 107, 108, 109, 110,
    111, 112, 113, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 128,
    129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144,
    145, 146, 147, 148, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
    162, 163, 164, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176,
    177, 178, 179, 180, 181, 182, 183, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 192, 193, 194, 195, 196, 197, 198, 199, 199, 200, 201, 202, 203, 204, 205,
    205, 206, 207, 208, 209, 210, 210, 211, 212, 213, 214, 214, 215, 216, 217, 218,
    219, 219, 220, 221, 222, 223, 223, 224, 225, 226, 227, 227, 228, 229, 230, 230,
    231, 232, 233, 234, 234, 235, 236, 237, 237, 238, 239, 240, 241, 241, 242, 243,
    244, 244, 245, 246, 247, 247, 248, 249, 250, 250, 251, 252, 253, 253, 254, 255,
];

/// 面中 / 鼻基底 / 下巴 / 眉区柔提亮 LUT（提取自 `pixcake.exe` `.rdata` RVA `0x09444410`，长度 `768 = 3 × 256` 字节）。
///
/// SHA-256: `4C69095648D9A696FE0D7490DB1071CCDDCCCB5FCD3C6838DE398C71E4D289DF`
///
/// ### 平面布局与 RGBA 通道映射
/// 与 [`PLUMP_BRIGHTEN_LUT_FOREHEAD`] 相同，为 `3 × 256` BGR 平面布局：
/// - `0..256`（`+0x000..+0x0FF`）：**B 通道**查找表（对应 RGBA 像素的 `rgba[idx + 2]`）
/// - `256..512`（`+0x100..+0x1FF`）：**G 通道**查找表（对应 RGBA 像素的 `rgba[idx + 1]`）
/// - `512..768`（`+0x200..+0x2FF`）：**R 通道**查找表（对应 RGBA 像素的 `rgba[idx + 0]`）
///
/// 例如输入灰度 `128` 时，原厂 BGR 查表结果为 `[137, 135, 135]`，对应 RGBA 的 RGB 输出为 `[135, 135, 137]`。
#[rustfmt::skip]
pub const PLUMP_BRIGHTEN_LUT_SOFT: [u8; 768] = [
    // Plane 0: B channel (0..256)
    0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 32, 34, 35,
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52,
    53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 67, 68, 69,
    70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85,
    86, 88, 89, 90, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103,
    104, 105, 106, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136,
    137, 138, 139, 140, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
    154, 155, 156, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168,
    169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 183,
    184, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198,
    198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
    214, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 223, 224, 225, 226, 227,
    228, 229, 230, 231, 232, 233, 234, 234, 234, 235, 236, 237, 238, 239, 240, 241,
    242, 242, 243, 244, 245, 246, 247, 248, 249, 249, 250, 251, 252, 253, 254, 255,
    // Plane 1: G channel (256..512)
    0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68,
    69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
    85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
    119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
    135, 136, 137, 138, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
    152, 153, 154, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
    167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
    183, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
    197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
    213, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 222, 223, 224, 225, 226,
    227, 228, 229, 230, 231, 232, 233, 234, 234, 235, 236, 237, 238, 239, 240, 241,
    242, 242, 243, 244, 245, 246, 247, 248, 249, 249, 250, 251, 252, 253, 254, 255,
    // Plane 2: R channel (512..768)
    0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 64, 65, 66, 67, 68,
    69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
    85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101,
    102, 103, 104, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118,
    119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
    135, 136, 137, 138, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151,
    152, 153, 154, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166,
    167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182,
    183, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197,
    197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212,
    213, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 222, 223, 224, 225, 226,
    227, 228, 229, 230, 231, 232, 233, 234, 234, 235, 236, 237, 238, 239, 240, 241,
    242, 242, 243, 244, 245, 246, 247, 248, 249, 249, 250, 251, 252, 253, 254, 255,
];

/// 前端 17 项丰盈参数（4 个区域总控乘子 + 13 个滑杆参数），对应 `EffectFacePlump`（`0x05e10c80..0x05e14a70`）。
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PlumpFrontendParams17 {
    /// 额头区域总控乘子 (`forehead_degree`)
    pub forehead_degree: f32,
    /// 面中区域总控乘子 (`midface_degree`)
    pub midface_degree: f32,
    /// 鼻基底区域总控乘子 (`noseBase_degree`)
    pub nose_base_degree: f32,
    /// 下巴区域总控乘子 (`chin_degree`)
    pub chin_degree: f32,

    /// 额头丰盈平滑 (`foreheadSmooth`)
    pub forehead_smooth: f32,
    /// 左眼窝 (`eyeHoleLeft`)
    pub eye_hole_left: f32,
    /// 右眼窝 (`eyeHoleRight`)
    pub eye_hole_right: f32,
    /// 面中/苹果肌丰盈平滑 (`midfaceSmooth`)
    pub midface_smooth: f32,
    /// 脸颊丰盈 (`cheek`)
    pub cheek: f32,
    /// 鼻基底丰盈平滑 (`noseBaseSmooth`)
    pub nose_base_smooth: f32,
    /// 嘴角丰盈 (`mouthAngulus`)
    pub mouth_angulus: f32,
    /// 下巴丰盈 (`chin`)
    pub chin: f32,
    /// 眉区提亮 (`eyeBrow`)
    pub eye_brow: f32,
    /// 额头提亮 (`foreheadBrighten`)
    pub forehead_brighten: f32,
    /// 面中提亮 (`midfaceBrighten`)
    pub midface_brighten: f32,
    /// 鼻基底提亮 (`noseBaseBrighten`)
    pub nose_base_brighten: f32,
    /// 下巴提亮 (`chinBrighten`)
    pub chin_brighten: f32,
}

/// 映射至 `face_cache::slot[1]`（`0x06c1e730`）的 13 个内部权重槽位（`vec9[0..9]` 与 `vec4[0..4]`）。
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Plump13Weights {
    /// 对应 `ctx + 0x68..+0x128`：
    /// `[0]=额头平滑, [1]=左眼窝, [2]=右眼窝, [3]=面中平滑, [4]=脸颊, [5]=鼻基底平滑, [6]=嘴角, [7]=下巴, [8]=眉区提亮`
    pub vec9: [f32; 9],
    /// 对应 `ctx + 0x140..+0x188`：
    /// `[0]=额头提亮(LUT1), [1]=面中提亮(LUT2), [2]=鼻基底提亮(LUT2), [3]=下巴提亮(LUT2)`
    pub vec4: [f32; 4],
}

impl Plump13Weights {
    /// 对应 `0x06c1f0fa..0x06c1f153` 的全零早退条件：若 13 槽全部 `<= 0.0`，则跳过合成。
    #[inline]
    pub fn is_all_inactive(&self) -> bool {
        self.vec9.iter().all(|&w| w <= 0.0_f32) && self.vec4.iter().all(|&w| w <= 0.0_f32)
    }
}

/// 按 `VTABLE-0x09444930.md` §三.1 将 17 项前端参数预乘映射为 13 槽内部权重（`vec9[0..9]` + `vec4[0..4]`）。
pub fn map_17_slots_to_13_weights(params: &PlumpFrontendParams17) -> Plump13Weights {
    Plump13Weights {
        vec9: [
            params.forehead_smooth * params.forehead_degree,
            params.eye_hole_left,
            params.eye_hole_right,
            params.midface_smooth * params.midface_degree,
            params.cheek,
            params.nose_base_smooth * params.nose_base_degree,
            params.mouth_angulus,
            params.chin * params.chin_degree,
            params.eye_brow,
        ],
        vec4: [
            params.forehead_brighten * params.forehead_degree,
            params.midface_brighten * params.midface_degree,
            params.nose_base_brighten * params.nose_base_degree,
            params.chin_brighten * params.chin_degree,
        ],
    }
}

/// 单通道 `u8` `StackBlur`（对应 `0x0260e810` / `0x0260e950`，窗口 `2r + 1`，除数 `(r + 1)^2`）。
///
/// 采用水平 + 垂直两趟分离卷积（边界像素按边缘钳位复制），每次输出截断除数 `div = (radius + 1)^2`。
/// 当 `width == 0 || height == 0` 返回空向量；当 `radius == 0` 返回原图拷贝。
pub fn stack_blur_u8(src: &[u8], width: usize, height: usize, radius: usize) -> Vec<u8> {
    assert_eq!(
        src.len(),
        width.saturating_mul(height),
        "stack_blur_u8: src.len() ({}) != width * height ({} * {})",
        src.len(),
        width,
        height
    );

    if width == 0 || height == 0 {
        return Vec::new();
    }
    if radius == 0 {
        return src.to_vec();
    }

    let window = 2 * radius + 1;
    let div_sum = (radius + 1) * (radius + 1);
    let mut temp = vec![0_u8; width * height];
    let mut dst = vec![0_u8; width * height];
    let mut stack = vec![0_usize; window];

    // 第一趟：逐行水平 StackBlur
    let max_x = width - 1;
    for y in 0..height {
        let row_offset = y * width;
        let mut sum = 0_usize;
        let mut sum_in = 0_usize;
        let mut sum_out = 0_usize;

        for k in -(radius as isize)..=(radius as isize) {
            let sx = k.clamp(0, max_x as isize) as usize;
            let p = src[row_offset + sx] as usize;
            let stack_idx = (k + radius as isize) as usize;
            stack[stack_idx] = p;
            let weight = radius + 1 - k.unsigned_abs();
            sum += p * weight;
            if k > 0 {
                sum_in += p;
            } else {
                sum_out += p;
            }
        }

        let mut stack_ptr = radius;
        for x in 0..width {
            temp[row_offset + x] = (sum / div_sum) as u8;

            sum -= sum_out;
            let stack_start = (stack_ptr + window - radius) % window;
            sum_out -= stack[stack_start];

            let p_in = src[row_offset + (x + radius + 1).min(max_x)] as usize;
            stack[stack_start] = p_in;
            sum_in += p_in;
            sum += sum_in;

            stack_ptr = (stack_ptr + 1) % window;
            let center = stack[stack_ptr];
            sum_out += center;
            sum_in -= center;
        }
    }

    // 第二趟：逐列垂直 StackBlur
    let max_y = height - 1;
    for x in 0..width {
        let mut sum = 0_usize;
        let mut sum_in = 0_usize;
        let mut sum_out = 0_usize;

        for k in -(radius as isize)..=(radius as isize) {
            let sy = k.clamp(0, max_y as isize) as usize;
            let p = temp[sy * width + x] as usize;
            let stack_idx = (k + radius as isize) as usize;
            stack[stack_idx] = p;
            let weight = radius + 1 - k.unsigned_abs();
            sum += p * weight;
            if k > 0 {
                sum_in += p;
            } else {
                sum_out += p;
            }
        }

        let mut stack_ptr = radius;
        for y in 0..height {
            dst[y * width + x] = (sum / div_sum) as u8;

            sum -= sum_out;
            let stack_start = (stack_ptr + window - radius) % window;
            sum_out -= stack[stack_start];

            let p_in = temp[(y + radius + 1).min(max_y) * width + x] as usize;
            stack[stack_start] = p_in;
            sum_in += p_in;
            sum += sum_in;

            stack_ptr = (stack_ptr + 1) % window;
            let center = stack[stack_ptr];
            sum_out += center;
            sum_in -= center;
        }
    }

    dst
}

#[inline]
fn clamp_trunc_i32_to_u8(v: f32) -> u8 {
    (v.trunc() as i32).clamp(0, 255) as u8
}

/// 丰盈 2× 残差叠加写回（对应 `0x06c27030`）。
///
/// 对每个像素计算门控混合系数 `alpha = (w / 255.0) * (s / 255.0)`（任一 mask 为 `0` 时跳过）；
/// 对 `R/G/B` 三个颜色通道按：
/// - `val = trunc(orig + 2.0 * (pred - 128.0))`
/// - `out = clamp(trunc(val * alpha + orig * (1.0 - alpha)), 0, 255)`
/// 更新 `rgba` 缓冲区，并保持第 4 通道 `A` 不变。
///
/// `pred_rgb` 为 `[R, G, B]` 交错 `f32`（值域 `[0.0, 255.0]`，长度 `width * height * 3`）。
pub fn apply_plump_residual_rgba8(
    rgba: &mut [u8],
    pred_rgb: &[f32],
    weight_mask: &[u8],
    split_mask: &[u8],
    width: usize,
    height: usize,
) {
    let pixel_count = width.saturating_mul(height);
    assert_eq!(rgba.len(), pixel_count * 4, "rgba length mismatch");
    assert_eq!(pred_rgb.len(), pixel_count * 3, "pred_rgb length mismatch");
    assert_eq!(weight_mask.len(), pixel_count, "weight_mask length mismatch");
    assert_eq!(split_mask.len(), pixel_count, "split_mask length mismatch");

    for i in 0..pixel_count {
        let w = weight_mask[i];
        let s = split_mask[i];
        if w == 0 || s == 0 {
            continue;
        }

        let alpha = (w as f32 * INV_255) * (s as f32 * INV_255);
        let one_minus_alpha = 1.0_f32 - alpha;

        let rgba_base = i * 4;
        let pred_base = i * 3;
        for c in 0..3 {
            let orig = rgba[rgba_base + c] as f32;
            let pred = pred_rgb[pred_base + c];
            let val = (orig + 2.0_f32 * (pred - 128.0_f32)).trunc();
            let blended = val * alpha + orig * one_minus_alpha;
            rgba[rgba_base + c] = clamp_trunc_i32_to_u8(blended);
        }
    }
}

/// 丰盈双 `3×256` 平面 LUT 分区提亮写回（对应 `0x06c29f90`）。
///
/// `lut_bgr_3x256` 为原厂 BGR 平面布局（`[0..256]=B`、`[256..512]=G`、`[512..768]=R`）：
/// - `rgba[idx + 0]`（R）查 `lut_bgr_3x256[512 + orig_r]`
/// - `rgba[idx + 1]`（G）查 `lut_bgr_3x256[256 + orig_g]`
/// - `rgba[idx + 2]`（B）查 `lut_bgr_3x256[orig_b]`
/// - `rgba[idx + 3]`（A）保持不变
///
/// 当 `brighten_mask[i] == 0 || split_mask[i] == 0` 时跳过；否则按
/// `alpha = (w / 255.0) * (s / 255.0)` 与 `out = clamp(trunc(lut[orig] * alpha + orig * (1.0 - alpha)), 0, 255)` 混合。
pub fn apply_plump_brighten_lut_rgba8(
    rgba: &mut [u8],
    lut_bgr_3x256: &[u8; 768],
    brighten_mask: &[u8],
    split_mask: &[u8],
    width: usize,
    height: usize,
) {
    let pixel_count = width.saturating_mul(height);
    assert_eq!(rgba.len(), pixel_count * 4, "rgba length mismatch");
    assert_eq!(
        brighten_mask.len(),
        pixel_count,
        "brighten_mask length mismatch"
    );
    assert_eq!(split_mask.len(), pixel_count, "split_mask length mismatch");

    for i in 0..pixel_count {
        let w = brighten_mask[i];
        let s = split_mask[i];
        if w == 0 || s == 0 {
            continue;
        }

        let alpha = (w as f32 * INV_255) * (s as f32 * INV_255);
        let one_minus_alpha = 1.0_f32 - alpha;

        let base = i * 4;
        let orig_r = rgba[base];
        let orig_g = rgba[base + 1];
        let orig_b = rgba[base + 2];

        let lut_r = lut_bgr_3x256[512 + orig_r as usize] as f32;
        let lut_g = lut_bgr_3x256[256 + orig_g as usize] as f32;
        let lut_b = lut_bgr_3x256[orig_b as usize] as f32;

        rgba[base] = clamp_trunc_i32_to_u8(lut_r * alpha + (orig_r as f32) * one_minus_alpha);
        rgba[base + 1] = clamp_trunc_i32_to_u8(lut_g * alpha + (orig_g as f32) * one_minus_alpha);
        rgba[base + 2] = clamp_trunc_i32_to_u8(lut_b * alpha + (orig_b as f32) * one_minus_alpha);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_lengths_and_key_points() {
        assert_eq!(PLUMP_BRIGHTEN_LUT_FOREHEAD.len(), 768);
        assert_eq!(PLUMP_BRIGHTEN_LUT_SOFT.len(), 768);

        // BGR 平面 128 关键点验证：
        // LUT 1 (额头): B[128]=148, G[128]=145, R[128]=145
        assert_eq!(PLUMP_BRIGHTEN_LUT_FOREHEAD[128], 148);
        assert_eq!(PLUMP_BRIGHTEN_LUT_FOREHEAD[256 + 128], 145);
        assert_eq!(PLUMP_BRIGHTEN_LUT_FOREHEAD[512 + 128], 145);

        // LUT 2 (柔提亮): B[128]=137, G[128]=135, R[128]=135
        assert_eq!(PLUMP_BRIGHTEN_LUT_SOFT[128], 137);
        assert_eq!(PLUMP_BRIGHTEN_LUT_SOFT[256 + 128], 135);
        assert_eq!(PLUMP_BRIGHTEN_LUT_SOFT[512 + 128], 135);

        // 端点恒等 [0 -> 0, 255 -> 255]
        for plane in 0..3 {
            let offset = plane * 256;
            assert_eq!(PLUMP_BRIGHTEN_LUT_FOREHEAD[offset], 0);
            assert_eq!(PLUMP_BRIGHTEN_LUT_FOREHEAD[offset + 255], 255);
            assert_eq!(PLUMP_BRIGHTEN_LUT_SOFT[offset], 0);
            assert_eq!(PLUMP_BRIGHTEN_LUT_SOFT[offset + 255], 255);
        }
    }

    #[test]
    fn test_stack_blur_r10_constant_identity_and_edge_smoothing() {
        let w = 48;
        let h = 48;

        // 常量图恒等
        for val in [0_u8, 1, 128, 255] {
            let src = vec![val; w * h];
            let out = stack_blur_u8(&src, w, h, PLUMP_STACK_BLUR_RADIUS);
            assert_eq!(out, src, "constant image {val} must be preserved");
        }

        // 阶跃边界平滑与对称性：左半 0、右半 255
        let mut step = vec![0_u8; w * h];
        for y in 0..h {
            for x in (w / 2)..w {
                step[y * w + x] = 255;
            }
        }
        let blurred = stack_blur_u8(&step, w, h, PLUMP_STACK_BLUR_RADIUS);
        // 远端未受波及区保持 0 和 255
        assert_eq!(blurred[0], 0);
        assert_eq!(blurred[w - 1], 255);
        // 中心过渡带严格单调非减且处于 (0, 255) 之间
        let mid_row = &blurred[(h / 2) * w..(h / 2 + 1) * w];
        assert!(mid_row[w / 2 - 1] > 0 && mid_row[w / 2 - 1] < 128);
        assert!(mid_row[w / 2] >= 128 && mid_row[w / 2] < 255);
        for x in 1..w {
            assert!(mid_row[x] >= mid_row[x - 1]);
        }

        // 空图与 radius=0 边界
        assert!(stack_blur_u8(&[], 0, 0, 10).is_empty());
        let small = vec![10_u8, 20, 30, 40];
        assert_eq!(stack_blur_u8(&small, 2, 2, 0), small);
    }

    #[test]
    fn test_residual_zero_point_and_clamp() {
        // pred=128 时 2*(pred-128)=0，无论 alpha 如何输出与原图恒等，alpha 通道保持不变
        let orig = [100_u8, 150, 200, 77, 10, 245, 0, 255];
        let mut rgba = orig;
        let pred = [128.0_f32, 128.0, 128.0, 128.0, 128.0, 128.0];
        let w = [255_u8, 128];
        let s = [255_u8, 200];
        apply_plump_residual_rgba8(&mut rgba, &pred, &w, &s, 2, 1);
        assert_eq!(rgba, orig);

        // 正残差上溢截断至 255，负残差下溢截断至 0，同时保持 alpha 通道
        let mut clamp_rgba = [200_u8, 50, 100, 123];
        // R: 200 + 2*(200-128) = 344 -> clamp 255
        // G: 50  + 2*(80-128)  = -46 -> clamp 0
        // B: 100 + 2*(138-128) = 120 -> 120
        let clamp_pred = [200.0_f32, 80.0, 138.0];
        apply_plump_residual_rgba8(&mut clamp_rgba, &clamp_pred, &[255], &[255], 1, 1);
        assert_eq!(clamp_rgba, [255, 0, 120, 123]);

        // mask=0 时跳过，原值完全保留
        let mut skip_rgba = [42_u8, 84, 126, 210];
        apply_plump_residual_rgba8(&mut skip_rgba, &[255.0, 0.0, 255.0], &[0], &[255], 1, 1);
        assert_eq!(skip_rgba, [42, 84, 126, 210]);
        apply_plump_residual_rgba8(&mut skip_rgba, &[255.0, 0.0, 255.0], &[255], &[0], 1, 1);
        assert_eq!(skip_rgba, [42, 84, 126, 210]);
    }

    #[test]
    fn test_brighten_lut_exact_values_and_alpha_zero() {
        // orig=128, alpha=1.0 -> LUT 1 输出 RGB [145, 145, 148]，alpha 通道保持不变
        let mut rgba1 = [128_u8, 128, 128, 99];
        apply_plump_brighten_lut_rgba8(
            &mut rgba1,
            &PLUMP_BRIGHTEN_LUT_FOREHEAD,
            &[255],
            &[255],
            1,
            1,
        );
        assert_eq!(rgba1, [145, 145, 148, 99]);

        // orig=128, alpha=1.0 -> LUT 2 输出 RGB [135, 135, 137]，alpha 通道保持不变
        let mut rgba2 = [128_u8, 128, 128, 188];
        apply_plump_brighten_lut_rgba8(
            &mut rgba2,
            &PLUMP_BRIGHTEN_LUT_SOFT,
            &[255],
            &[255],
            1,
            1,
        );
        assert_eq!(rgba2, [135, 135, 137, 188]);

        // alpha=0（w=0 或 s=0）严格恒等
        let mut rgba_w0 = [128_u8, 128, 128, 55];
        apply_plump_brighten_lut_rgba8(
            &mut rgba_w0,
            &PLUMP_BRIGHTEN_LUT_FOREHEAD,
            &[0],
            &[255],
            1,
            1,
        );
        assert_eq!(rgba_w0, [128, 128, 128, 55]);

        let mut rgba_s0 = [128_u8, 128, 128, 66];
        apply_plump_brighten_lut_rgba8(
            &mut rgba_s0,
            &PLUMP_BRIGHTEN_LUT_SOFT,
            &[255],
            &[0],
            1,
            1,
        );
        assert_eq!(rgba_s0, [128, 128, 128, 66]);
    }

    #[test]
    fn test_map_17_slots_to_13_weights() {
        let params = PlumpFrontendParams17 {
            forehead_degree: 0.5,
            midface_degree: 0.8,
            nose_base_degree: 0.25,
            chin_degree: 0.4,
            forehead_smooth: 0.6,
            eye_hole_left: 0.11,
            eye_hole_right: 0.22,
            midface_smooth: 0.75,
            cheek: 0.33,
            nose_base_smooth: 0.8,
            mouth_angulus: 0.44,
            chin: 0.5,
            eye_brow: 0.55,
            forehead_brighten: 0.9,
            midface_brighten: 0.5,
            nose_base_brighten: 0.4,
            chin_brighten: 0.25,
        };

        let weights = map_17_slots_to_13_weights(&params);
        let expected_vec9 = [
            0.6_f32 * 0.5_f32,
            0.11_f32,
            0.22_f32,
            0.75_f32 * 0.8_f32,
            0.33_f32,
            0.8_f32 * 0.25_f32,
            0.44_f32,
            0.5_f32 * 0.4_f32,
            0.55_f32,
        ];
        let expected_vec4 = [
            0.9_f32 * 0.5_f32,
            0.5_f32 * 0.8_f32,
            0.4_f32 * 0.25_f32,
            0.25_f32 * 0.4_f32,
        ];

        for (got, exp) in weights.vec9.iter().zip(expected_vec9.iter()) {
            assert!((got - exp).abs() < 1e-6, "vec9 mismatch: {got} vs {exp}");
        }
        for (got, exp) in weights.vec4.iter().zip(expected_vec4.iter()) {
            assert!((got - exp).abs() < 1e-6, "vec4 mismatch: {got} vs {exp}");
        }
        assert!(!weights.is_all_inactive());

        let zero_weights = map_17_slots_to_13_weights(&PlumpFrontendParams17::default());
        assert!(zero_weights.is_all_inactive());
    }
}
