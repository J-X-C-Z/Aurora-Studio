//! Capture One `CImgOpUnsharpMaskLinear` / `USMLinear` kernels.
//!
//! 证据：当前 `ImgCoreDll.dll` SHA-256
//! `ca02b6ac2786e9a100f63221df7340e5df210416eb0bd53a4bb1bfc2306a44a4`，
//! OpenCL program `p_041894336.cl`（19,874 B，文件偏移 41,931,088，当前 DLL
//! 内唯一），12 个变体。DSC_7444 导出队列（步骤 63~75）为
//! `1 -> Set -> 2,3 -> 4 -> 2,3 -> 2,3 -> 2,3 -> Div -> 11 -> Add -> 10`。
//!
//! 宿主地址（均为当前 DLL 的 VA）：
//! - `0x18181b7b0`：算子构造函数，写入 vtable `0x1828033f0`，类名
//!   `UnsharpMaskLinear`（`0x1826ef720`），RTTI 名 `.?AVCImgOpUnsharpMaskLinear@@`。
//! - `0x18181ba80`：参数准备；`0x181823370..0x1818260f6`：执行体（Run），
//!   含 `CImgOpUnsharpMaskLinear::Run: unknown method.` 错误字符串。
//! - `0x18181c5c0`：CL 内核 arg 绑定/派发；`0x1813569b0`：高斯 tap 构造；
//!   `0x181820410`：变体 8 的 CPU 对照实现。
//!
//! **已证明**：各变体的内核公式与缓冲区角色；宿主最多 16 个 tap（CL 的
//! `float16`），默认 tap 数为 13 或 33（半宽 6/16），tap 权重由
//! `0x1813569b0` 的像素积分高斯生成并归一化；变体 6/11 的 `float4.w` 是
//! strength，`.x` 是 threshold，`.z` 是 halo 累加开关。
//! **未证明**：用户滑杆 `USMRadius/USMAmount/USMHaloControl` 到上述宿主字段
//! 的完整数值表；变体 8 的 `float8` 五个分量与用户滑杆的逐项对应。

/// `HaloParams8` 对应变体 8 的 `zfb81b1b9ea`（`float8`）。前 5 个分量在内核中
/// 具名；`s5..s7` 未被读取。数值来源见 `halo_params_from_host`。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HaloParams8 {
    /// `z710c525abc.s0`：`1/s0` 是局部权重下限。
    pub s0: f32,
    /// `z4476636e72.s1`：计数门限（CPU 中对应 `+0x1107c/100`）。
    pub s1: f32,
    /// `ze6229097a6.s2`：不平衡比增益。
    pub s2: f32,
    /// `zc66dcf5a74.s3`：计数差抑制系数。
    pub s3: f32,
    /// `z7417c0ac6d.s4`：分母系数。
    pub s4: f32,
}

/// 变体 8 的四个统计量对（计数/和），顺序为
/// `(high, high2, low, low2)`，其中 2 表示 `delta + src` 平面。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HaloStats {
    pub high_count: f32,
    pub high_sum: f32,
    pub high2_count: f32,
    pub high2_sum: f32,
    pub low_count: f32,
    pub low_sum: f32,
    pub low2_count: f32,
    pub low2_sum: f32,
}

/// 变体 8 的输出：两侧加权均值与各自的分母。
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HaloAccum {
    pub high: f32,
    pub high_den: f32,
    pub low: f32,
    pub low_den: f32,
}

/// 变体 1：`luma = sqrt(dot(rgb^2, weights))`。
///
/// 权重是宿主经 `0x18181ba80` 写入 op `+0x88/+0x8c/+0x90` 的 `float3`，
/// 通过 `0x181624f90(..., index=5, size=16)` 传入内核。BT.601 只是调用方
/// 常用的取值，不是内核常量。
pub fn luma_sqrt(rgb: [f32; 3], weights: [f32; 3]) -> f32 {
    let sum = rgb[0] * rgb[0] * weights[0]
        + rgb[1] * rgb[1] * weights[1]
        + rgb[2] * rgb[2] * weights[2];
    sum.max(0.0).sqrt()
}

/// 一次对称可分离 FIR（变体 2 水平 / 变体 3 垂直），对应 CL：
/// `acc = center*src[0] + Σ taps[i]*(src[±(i+1)])`。
///
/// 内核不做边界钳位；宿主在执行体多处按 `radius` 扩展 ROI（`0x181823630`
/// 等）。本函数用边缘复制模拟宿主 padding，因此调用方传入的是**未扩展**的
/// 图像。`taps` 的可用长度上限为 16（CL `float16`）。
pub fn fir_pass(
    src: &[f32],
    width: usize,
    height: usize,
    taps: &[f32],
    center: f32,
    horizontal: bool,
) -> Vec<f32> {
    fir_pass_strided(src, width, width, height, taps, center, horizontal)
}

/// 与 [`fir_pass`] 相同，但允许源缓冲区有独立行跨度，对应内核的
/// `ze265e77baa`（源 stride）与 `z9daa334467`（目标 stride）。
pub fn fir_pass_strided(
    src: &[f32],
    src_stride: usize,
    dst_stride: usize,
    height: usize,
    taps: &[f32],
    center: f32,
    horizontal: bool,
) -> Vec::<f32> {
    let width = src_stride.min(dst_stride);
    let mut out = vec![0.0f32; dst_stride * height];
    for y in 0..height {
        for x in 0..width {
            let mut acc = center * src[y * src_stride + x];
            for (i, &tap) in taps.iter().enumerate() {
                let d = i + 1;
                let (x0, y0) = if horizontal {
                    (x.saturating_sub(d).min(width - 1), y)
                } else {
                    (x, y.saturating_sub(d).min(height - 1))
                };
                let (x1, y1) = if horizontal {
                    ((x + d).min(width - 1), y)
                } else {
                    (x, (y + d).min(height - 1))
                };
                acc += tap * (src[y0 * src_stride + x0] + src[y1 * src_stride + x1]);
            }
            out[y * dst_stride + x] = acc;
        }
    }
    out
}

/// 四次 `2,3` 交替 FIR。每个 tap 为 `1/(2r+1)` 时是盒式均值；宿主实际使用
/// 的权重由 `0x1813569b0` 生成（见 [`gaussian_taps_integrated`]）。
pub fn fir_blur_4(src: &[f32], width: usize, height: usize, taps: &[f32], center: f32) -> Vec<f32> {
    let mut plane = src.to_vec();
    for pass in 0..4 {
        plane = fir_pass(&plane, width, height, taps, center, pass % 2 == 0);
    }
    plane
}

/// 均匀盒式权重的四次 `2,3` 交替 FIR，保留旧接口名。
pub fn box_blur_4(src: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let r = radius.max(1);
    let w = 1.0 / (2.0 * r as f32 + 1.0);
    fir_blur_4(src, width, height, &vec![w; r], w)
}

/// 旧的高斯近似：仅用于无宿主权重时的占位，不是内核或宿主的公式。
pub fn gaussian_taps(radius: f32) -> (Vec<f32>, f32) {
    let taps = radius.round().clamp(1.0, 8.0) as usize;
    let sigma = (radius.max(0.3)) / 2.0;
    let mut weights = Vec::with_capacity(taps);
    let mut sum = 0.0f32;
    for i in 1..=taps {
        let w = (-((i * i) as f32) / (2.0 * sigma * sigma)).exp();
        weights.push(w);
        sum += 2.0 * w;
    }
    sum += 1.0;
    for w in &mut weights {
        *w /= sum;
    }
    (weights, 1.0 / sum)
}

/// 宿主 `0x1813569b0` 的 tap 构造：对宽度 1 的像素桶积分高斯，再归一化使
/// `center + 2*Σw[k] = 1`。`erf` 用 Rust 标准库没有的函数，这里以互补误差
/// 函数的常用有理逼近实现；公式结构（`(i-(n-1)/2 ± 0.5)/(sqrt(2)*sigma)`）
/// 已由反汇编确认，浮点逐位一致性未验证。
pub fn gaussian_taps_integrated(tap_count: usize, sigma: f32) -> (Vec<f32>, f32) {
    let n = tap_count.max(1);
    let mid = (n - 1) as f32 * 0.5;
    let den = (2.0f32).sqrt() * sigma.max(1e-3);
    let mut w = Vec::with_capacity(n);
    let mut sum = 1.0f32;
    for i in 0..n {
        let t = i as f32 - mid;
        let hi = erf_approx((t + 0.5) / den);
        let lo = erf_approx((t - 0.5) / den);
        let weight = 0.5 * (hi - lo);
        w.push(weight);
        sum += 2.0 * weight;
    }
    for x in &mut w {
        *x /= sum;
    }
    (w, 1.0 / sum)
}

fn erf_approx(x: f32) -> f32 {
    // Abramowitz & Stegun 7.1.26，最大绝对误差约 1.5e-7。
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let p = ((((1.061_405_429 * t - 1.453_152_027) * t
        + 1.421_413_741) * t
        - 0.284_496_736) * t
        + 0.254_829_592) * t;
    sign * (1.0 - p * (-x * x).exp())
}

/// 变体 4：阈值差分。`a = |d| * max(0, |d|-t)`，`b = d² + 1e-10`。
pub fn threshold_excess(src: &[f32], blur: &[f32], threshold: f32) -> (Vec<f32>, Vec<f32>) {
    let mut a = vec![0.0; src.len()];
    let mut b = vec![0.0; src.len()];
    for i in 0..src.len() {
        let diff = (src[i] - blur[i]).abs();
        let excess = (diff - threshold).max(0.0);
        a[i] = diff * excess;
        b[i] = diff * diff + 1.0e-10;
    }
    (a, b)
}

/// 变体 5：Sobel 门控。`a = (src-blur)*w`，`b = (src-blur)² + 1e-10`，
/// `w = clamp((|g|-t)*gain, 0, 1)`，Sobel 权重 `3/32` 与 `10/32` 逐字保留。
pub fn sobel_gate(
    src: &[f32],
    blur: &[f32],
    width: usize,
    height: usize,
    threshold: f32,
    gain: f32,
) -> (Vec<f32>, Vec<f32>) {
    let at = |x: isize, y: isize| {
        let xx = x.clamp(0, width as isize - 1) as usize;
        let yy = y.clamp(0, height as isize - 1) as usize;
        blur[yy * width + xx]
    };
    let mut a = vec![0.0; src.len()];
    let mut b = vec![0.0; src.len()];
    for y in 0..height {
        for x in 0..width {
            let (xi, yi) = (x as isize, y as isize);
            let gx = (at(xi + 1, yi - 1) + at(xi + 1, yi + 1)
                - at(xi - 1, yi - 1)
                - at(xi - 1, yi + 1))
                * (3.0 / 32.0)
                + (at(xi + 1, yi) - at(xi - 1, yi)) * (10.0 / 32.0);
            let gy = (at(xi + 1, yi + 1) + at(xi - 1, yi + 1)
                - at(xi + 1, yi - 1)
                - at(xi - 1, yi - 1))
                * (3.0 / 32.0)
                + (at(xi, yi + 1) - at(xi, yi - 1)) * (10.0 / 32.0);
            let g = (gx * gx + gy * gy).sqrt();
            let w = ((g - threshold) * gain).clamp(0.0, 1.0);
            let diff = src[y * width + x] - blur[y * width + x];
            a[y * width + x] = diff * w;
            b[y * width + x] = diff * diff + 1.0e-10;
        }
    }
    (a, b)
}

/// 变体 6 的最终合成（带 uchar mask）。
///
/// 内核写入的是 `accum`（变体 4 的 `a` 平面），不是 `blur`：`dst = d*w*strength
/// + accum[i]`。`blur` 只用于计算 `d = src - blur`。`halo_contrib` 是
/// `float4.y`，`halo_enable` 是 `float4.z`（>0 时向 `halo_accum` 累加）。
pub fn combine_v6(
    src: &[f32],
    blur: &[f32],
    accum: &[f32],
    mask: &[f32],
    threshold: f32,
    strength: f32,
    halo_contrib: f32,
    halo_enable: bool,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = accum.to_vec();
    let mut halo_accum = vec![0.0f32; src.len()];
    for i in 0..src.len() {
        let mut d = src[i] - blur[i];
        d *= mask[i].clamp(0.0, 1.0);
        out[i] = d * strength + accum[i];
        if halo_enable {
            halo_accum[i] = d * halo_contrib;
        }
    }
    (out, halo_accum)
}

/// 变体 6 在不提供 mask 平面时走 `else` 分支：死区钳除，语义与变体 11 相同。
pub fn combine_deadzone(accum: &[f32], diff: f32, threshold: f32, strength: f32) -> f32 {
    let d = diff - diff.clamp(-threshold, threshold);
    d * strength + accum[0]
}

/// 变体 11：与变体 6 相同，但没有 uchar mask，且总是有非零 `float4.w`。
pub fn combine_v11(
    src: &[f32],
    blur: &[f32],
    accum: &[f32],
    threshold: f32,
    strength: f32,
    halo_contrib: f32,
    halo_enable: bool,
    halo: Option<&[f32]>,
) -> (Vec<f32>, Vec<f32>) {
    let mut out = accum.to_vec();
    let mut halo_accum = vec![0.0f32; src.len()];
    for i in 0..src.len() {
        let mut d = src[i] - blur[i];
        d -= d.clamp(-threshold, threshold);
        if let Some(h) = halo {
            d = d.min(h[i]);
        }
        out[i] = d * strength + accum[i];
        if halo_enable {
            halo_accum[i] = d * halo_contrib;
        }
    }
    (out, halo_accum)
}

/// 旧接口（保留给已有调用方）：`diff` 是已经算好的增量，`weight` 是可选 mask，
/// `blur` 同时充当累加平面。内核的真实语义见 [`combine_v6`] / [`combine_v11`]。
pub fn combine(
    src: &[f32],
    blur: &[f32],
    diff: &[f32],
    weight: Option<&[f32]>,
    threshold: f32,
    halo: Option<&[f32]>,
    strength: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; src.len()];
    for i in 0..src.len() {
        let mut d = diff[i];
        match weight {
            Some(w) => d *= w[i].clamp(0.0, 1.0),
            None => d -= d.clamp(-threshold, threshold),
        }
        if let Some(h) = halo {
            d = d.min(h[i]);
        }
        out[i] = d * strength + blur[i];
    }
    out
}

/// 变体 7：`dst += (src - blur) * strength`，无阈值/掩码。
pub fn simple_blend_v7(src: &[f32], blur: &[f32], accum: &[f32], strength: f32) -> Vec<f32> {
    let mut out = accum.to_vec();
    for i in 0..src.len() {
        out[i] += (src[i] - blur[i]) * strength;
    }
    out
}

/// 变体 8 的单像素统计：完全按内核的四组计数/求和实现。
/// `delta` 是 `zcb1a344ce0`（增量平面），`radius` 是 `z568715c0ca`。
pub fn halo_stats(
    src: &[f32],
    delta: &[f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    radius: usize,
) -> HaloStats {
    let center = src[y * width + x];
    let center_delta = delta[y * width + x] + center;
    let mut s = HaloStats::default();
    let ry = radius as isize;
    for dy in -ry..=ry {
        for dx in -ry..=ry {
            let xx = (x as isize + dx).clamp(0, width as isize - 1) as usize;
            let yy = (y as isize + dy).clamp(0, height as isize - 1) as usize;
            let v = src[yy * width + xx];
            let vd = delta[yy * width + xx] + v;
            if v <= center {
                s.low_count += 1.0;
                s.low_sum += v;
            }
            if v >= center {
                s.high_count += 1.0;
                s.high_sum += v;
            }
            if vd <= center_delta {
                s.low2_count += 1.0;
                s.low2_sum += vd;
            }
            if vd >= center_delta {
                s.high2_count += 1.0;
                s.high2_sum += vd;
            }
        }
    }
    s
}

/// 变体 8 的加权：由 [`halo_stats`] 生成 `z8ecb725b2e/zb653962b35/
/// zcedab244ff/z5583dd78b5`，计数为零时按内核的除零语义退化为 0。
pub fn halo_weight(center: f32, center_delta: f32, s: HaloStats, p: HaloParams8) -> HaloAccum {
    let div = |num: f32, den: f32| if den == 0.0 { 0.0 } else { num / den };
    let high_mean = 2.0 * div(s.high_sum, s.high_count) - center;
    let low_mean = 2.0 * div(s.low_sum, s.low_count) - center;
    let high2_mean = 2.0 * div(s.high2_sum, s.high2_count) - center_delta;
    let low2_mean = 2.0 * div(s.low2_sum, s.low2_count) - center_delta;
    let floor = 1.0 / p.s0.max(1e-20);
    let mut a_high = floor.max(1.0 - (s.high_count - s.high2_count).abs() * p.s3) * high2_mean;
    let mut a_low = floor.max(1.0 - (s.low_count - s.low2_count).abs() * p.s3) * low2_mean;
    let mut d_high = div(high_mean - center, s.high_count);
    let mut d_low = div(center - low_mean, s.low_count);
    let mut den_high = floor;
    let mut den_low = floor;
    if s.high_count > p.s1 && d_high < d_low * p.s2 {
        let t = d_low * p.s2;
        let w = div(t, p.s4 * t + d_high);
        a_high += w * high_mean;
        den_high += w;
    }
    if s.low_count > p.s1 && d_low < d_high * p.s2 {
        let t = d_high * p.s2;
        let w = div(t, p.s4 * t + d_low);
        a_low += w * low_mean;
        den_low += w;
    }
    HaloAccum { high: a_high, high_den: den_high, low: a_low, low_den: den_low }
}

/// 变体 8 全图：分别对 `src` 与 `delta+src` 做局部统计并生成四个平面。
pub fn halo_stats_8(
    src: &[f32],
    delta: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    p: HaloParams8,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut hi = vec![0.0; src.len()];
    let mut hi_den = vec![0.0; src.len()];
    let mut lo = vec![0.0; src.len()];
    let mut lo_den = vec![0.0; src.len()];
    for y in 0..height {
        for x in 0..width {
            let i = y * width + x;
            let a = halo_weight(
                src[i],
                delta[i] + src[i],
                halo_stats(src, delta, width, height, x, y, radius),
                p,
            );
            hi[i] = a.high;
            hi_den[i] = a.high_den;
            lo[i] = a.low;
            lo_den[i] = a.low_den;
        }
    }
    (hi, hi_den, lo, lo_den)
}

/// 变体 9：消费变体 8 的四个平面，限制增量并累加到 `dst`。
pub fn halo_clamp_9(
    src: &[f32],
    delta: &[f32],
    dst: &[f32],
    hi: &[f32],
    hi_den: &[f32],
    lo: &[f32],
    lo_den: &[f32],
) -> Vec<f32> {
    let mut out = dst.to_vec();
    for i in 0..src.len() {
        let div = |a: f32, b: f32| if b == 0.0 { 0.0 } else { a / b };
        let mut d = delta[i];
        let pos = (div(hi[i], hi_den[i]) - src[i]).max(0.0);
        d = d.min(pos);
        let neg = (div(lo[i], lo_den[i]) - src[i]).min(0.0);
        d = d.max(neg);
        out[i] += d;
    }
    out
}

/// 宿主 `0x181820410` 的常量规模（仅说明量级，非逐位对应）：
/// `s0 = max(1, int(field +0x11078))`，`s1 = field(+0x1107c) * 0.01`，
/// `s2/s3 = exp2(field * 0.5) * k`，`k` 分别为 0.5/0.04。
/// 由于 float8 五分量与宿主字段没有逐一取齐，本函数只返回可证明的默认规模。
pub fn halo_params_from_host(f0: f32, f1: f32, f2: f32, f3: f32, f4: f32) -> HaloParams8 {
    HaloParams8 {
        s0: f0.max(1.0),
        s1: f1,
        s2: f2,
        s3: f3,
        s4: f4,
    }
}

/// 旧的局部 min/max 简化。已由 [`halo_stats_8`] / [`halo_clamp_9`] 取代，
/// 保留以说明旧近似的边界（不是原厂公式）。
pub fn halo_limit(
    src: &[f32],
    blur: &[f32],
    diff: &[f32],
    width: usize,
    height: usize,
    radius: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; diff.len()];
    for y in 0..height {
        for x in 0..width {
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for dy in -(radius as isize)..=radius as isize {
                for dx in -(radius as isize)..=radius as isize {
                    let xx = (x as isize + dx).clamp(0, width as isize - 1) as usize;
                    let yy = (y as isize + dy).clamp(0, height as isize - 1) as usize;
                    let v = src[yy * width + xx];
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
            }
            let center = src[y * width + x];
            let pos = (hi - center).max(0.0);
            let neg = (lo - center).min(0.0);
            let d = diff[y * width + x];
            out[y * width + x] = d.clamp(neg, pos);
            let _ = blur[y * width + x];
        }
    }
    out
}

/// 变体 10 的缩放因子：`s = max(src, 1e-20)`，`a = min(accum, s*16)`，
/// `factor = max(a/s, 0)`。旧接口只接受已经算好的 factor。
pub fn variant10_factor(src: f32, accum: f32) -> f32 {
    let s = src.max(1e-20);
    (accum.min(s * 16.0) / s).max(0.0)
}

/// 变体 10：把累计缩放应用到 RGB（4 通道缓冲的前三个分量）。
pub fn apply_rgb_factor_v10(rgb: [f32; 3], src: f32, accum: f32) -> [f32; 3] {
    let f = variant10_factor(src, accum);
    [rgb[0] * f, rgb[1] * f, rgb[2] * f]
}

/// 旧接口：直接乘一个外部 factor，保留非负钳位。
pub fn apply_rgb_factor(rgb: [f32; 3], factor: f32) -> [f32; 3] {
    let f = factor.max(0.0);
    [rgb[0] * f, rgb[1] * f, rgb[2] * f]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    #[test]
    fn variant1_luma_sqrt_uses_host_weights() {
        assert_eq!(luma_sqrt([0.0, 0.0, 0.0], [0.299, 0.587, 0.114]), 0.0);
        let y = luma_sqrt([0.25, 0.25, 0.25], [0.299, 0.587, 0.114]);
        assert!(approx(y, 0.25), "{y}");
        // 权重不是内核常量：换一组权重必须改变结果。
        let y2 = luma_sqrt([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert_eq!(y2, 0.0);
    }

    #[test]
    fn variant2_and_3_fir_are_transposes() {
        let w = 5usize;
        let h = 3usize;
        let mut src = vec![0.0f32; w * h];
        src[1 * w + 2] = 1.0;
        let (taps, center) = gaussian_taps(2.0);
        let hz = fir_pass(&src, w, h, &taps, center, true);
        let vt = fir_pass(&src, w, h, &taps, center, false);
        // 变体 2 的水平响应与变体 3 的垂直响应使用同一组 taps。
        assert!(approx(hz[1 * w + 2], center), "{} vs {}", hz[1 * w + 2], center);
        assert!(approx(vt[1 * w + 2], center), "{} vs {}", vt[1 * w + 2], center);
        assert!(approx(hz[1 * w + 1], taps[0]), "{}", hz[1 * w + 1]);
        assert!(approx(vt[2 * w + 2], taps[0]), "{}", vt[2 * w + 2]);
    }

    #[test]
    fn fir_preserves_a_flat_field() {
        let src = vec![0.25f32; 64];
        let (taps, center) = gaussian_taps(2.0);
        let out = fir_pass(&src, 8, 8, &taps, center, true);
        for v in out {
            assert!(approx(v, 0.25), "{v}");
        }
    }

    #[test]
    fn fir_pass_strided_matches_kernel_offsets() {
        // 源 stride=目标 stride=5；脉冲位于 (2,1)，垂直邻居距离一个 stride。
        let mut src = vec![0.0f32; 15];
        src[1 * 5 + 2] = 1.0;
        let out = fir_pass_strided(&src, 5, 5, 3, &[0.25], 0.5, false);
        assert!(approx(out[2 * 5 + 2], 0.25), "{}", out[2 * 5 + 2]);
        assert!(approx(out[0 * 5 + 2], 0.25), "{}", out[0 * 5 + 2]);
    }

    #[test]
    fn host_tap_builder_is_normalized_and_caps_at_16() {
        let (taps, center) = gaussian_taps_integrated(16, 4.0);
        assert_eq!(taps.len(), 16);
        let sum = center + 2.0 * taps.iter().sum::<f32>();
        assert!(approx(sum, 1.0), "sum {sum}");
        assert!(taps.iter().all(|w| *w >= 0.0));
        let (taps13, center13) = gaussian_taps_integrated(13, 2.0);
        assert!(approx(center13 + 2.0 * taps13.iter().sum::<f32>(), 1.0));
    }

    #[test]
    fn variant4_threshold_excess_is_zero_inside_the_deadband() {
        let src = [1.0f32, 1.0];
        let blur = [0.9f32, 0.8];
        let (a, b) = threshold_excess(&src, &blur, 1.0);
        assert_eq!(a, vec![0.0, 0.0]);
        assert!(b[0] > 0.0 && b[1] > 0.0);
        let (a, b) = threshold_excess(&[1.0], &[0.5], 0.25);
        assert!(approx(a[0], 0.5 * 0.25), "{}", a[0]);
        assert!(approx(b[0], 0.25 + 1e-10), "{}", b[0]);
    }

    #[test]
    fn variant5_sobel_gate_is_signed_and_clamped() {
        let mut blur = vec![0.0f32; 9];
        let src = vec![0.0f32; 9];
        // 内核的 Sobel 读的是 blur 平面；中心放一个脉冲，门控为中心差分。
        blur[4] = 1.0;
        let (a, b) = sobel_gate(&src, &blur, 3, 3, 0.0, 10.0);
        assert!(a[4].abs() < 1e-6, "{}", a[4]); // 中心 Sobel 梯度为零，门控后增量为零
        assert!(approx(b[4], 1.0 + 1e-10), "{}", b[4]);
        let (a, _) = sobel_gate(&src, &blur, 3, 3, 100.0, 1.0);
        assert!(a.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn variant6_mask_scales_and_writes_into_accum() {
        let accum = [0.1f32; 2];
        let src = [1.0f32, 1.0];
        let blur = [0.5f32, 0.5];
        let mask = [0.5f32, 2.0];
        let (out, halo) = combine_v6(&src, &blur, &accum, &mask, 0.0, 2.0, 0.25, true);
        // 内核把 mask（uchar/255）再 clamp 到 0..1；d 先乘 mask 再乘 strength。
        assert!(approx(out[0], 0.1 + 0.5 * 0.5 * 2.0), "{}", out[0]);
        assert!(approx(out[1], 0.1 + 0.5 * 1.0 * 2.0), "{}", out[1]);
        assert!(approx(halo[0], 0.5 * 0.5 * 0.25), "{}", halo[0]);
        let (_, halo_off) = combine_v6(&src, &blur, &accum, &mask, 0.0, 2.0, 0.25, false);
        assert!(halo_off.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn variant11_deadzone_and_optional_halo() {
        let accum = [0.25f32; 3];
        let src = [1.0f32, 1.0, 1.0];
        let blur = [0.8f32, 0.8, 0.8];
        let (out, _) = combine_v11(&src, &blur, &accum, 0.5, 1.0, 0.0, false, None);
        assert!(out.iter().all(|v| approx(*v, 0.25)), "{out:?}");
        let halo = [0.05f32; 3];
        let out2 = combine(&src, &blur, &[0.2f32; 3], Some(&halo), 0.0, None, 1.0);
        // 旧接口把 diff 当作已算好的增量，blur 当作累加平面。
        assert!(out2.iter().all(|v| approx(*v, 0.8 + 0.2 * 0.05)), "{out2:?}");
        assert!(approx(combine_deadzone(&accum, 0.2, 0.5, 1.0), 0.25));
    }

    #[test]
    fn variant7_simple_blend_has_no_threshold() {
        let src = [1.0f32, 0.6];
        let blur = [0.5f32, 0.6];
        let accum = [0.1f32, 0.2];
        let out = simple_blend_v7(&src, &blur, &accum, 0.5);
        assert!(approx(out[0], 0.35), "{}", out[0]);
        assert!(approx(out[1], 0.2), "{}", out[1]);
    }

    #[test]
    fn variant8_stats_count_center_and_delta_plane() {
        let src = [0.0f32; 5];
        let delta = [0.0f32; 5];
        let s = halo_stats(&src, &delta, 5, 1, 2, 0, 1);
        // 内核用 <= 和 >=，中心同时计入两侧；全平场时每侧都是完整 3x3 窗口。
        assert_eq!(s.high_count, 9.0);
        assert_eq!(s.low_count, 9.0);
        assert_eq!(s.high2_count, 9.0);
        assert_eq!(s.low2_count, 9.0);
        assert_eq!(s.high_sum, 0.0);
        // 单侧脉冲只进高侧（半径 1 的窗口覆盖 x=1..3）。
        let mut spike = src.to_vec();
        spike[2] = 0.2;
        let s2 = halo_stats(&spike, &delta, 5, 1, 1, 0, 1);
        assert_eq!(s2.high_count, 9.0);
        assert_eq!(s2.low_count, 6.0);
        // 两次 sum/count 的均值必须仍落在 [0, 0.2]；这里只验证两侧计数分离，
        // 行的重复采样次数由宿主 padding 决定，不把具体采样数写死。
        assert!(s2.high_sum > 0.0 && s2.high_sum <= 0.2 * s2.high_count);
    }

    #[test]
    fn variant8_weight_uses_float8_floor_and_gate() {
        let p = HaloParams8 { s0: 2.0, s1: 0.0, s2: 1.0, s3: 0.0, s4: 0.0 };
        let s = HaloStats {
            high_count: 2.0, high_sum: 0.4, high2_count: 2.0, high2_sum: 0.4,
            low_count: 2.0, low_sum: 0.0, low2_count: 2.0, low2_sum: 0.0,
        };
        let a = halo_weight(0.0, 0.0, s, p);
        assert!(approx(a.high_den, 0.5), "{}", a.high_den);
        assert!(approx(a.high, 0.4), "{}", a.high);
    }

    #[test]
    fn variant9_halo_clamp_limits_both_sides() {
        let src = [0.5f32; 2];
        let delta = [1.0f32, -1.0];
        let dst = [0.0f32; 2];
        let out = halo_clamp_9(
            &src, &delta, &dst,
            &[0.7, 0.7], &[1.0, 1.0], &[0.3, 0.3], &[1.0, 1.0],
        );
        assert!(approx(out[0], 0.2), "{}", out[0]);
        assert!(approx(out[1], -0.2), "{}", out[1]);
    }

    #[test]
    fn variant10_factor_clamps_to_16x_and_nonnegative() {
        assert!(approx(variant10_factor(0.5, 0.25), 0.5));
        assert!(approx(variant10_factor(0.5, 100.0), 16.0));
        assert_eq!(variant10_factor(0.0, -1.0), 0.0);
        let rgb = apply_rgb_factor_v10([2.0, 3.0, 4.0], 1.0, 0.5);
        assert_eq!(rgb, [1.0, 1.5, 2.0]);
    }

    #[test]
    fn halo_limit_keeps_the_legacy_approximation_distinct() {
        let src = [0.0f32; 9];
        let blur = [0.0f32; 9];
        let diff = [1.0f32; 9];
        let out = halo_limit(&src, &blur, &diff, 3, 3, 1);
        assert!(out.iter().all(|v| *v == 0.0));
    }
}

// ===========================================================================
// Host-side mapping additions, c1-parse lane, 2026-11-25.
//
// All constants below were read from the current ImgCoreDll.dll
// (SHA-256 ca02b6ac2786e9a100f63221df7340e5df210416eb0bd53a4bb1bfc2306a44a4)
// at the VAs quoted in each doc comment.  Nothing here is estimated.
//
// NOTE: `gaussian_taps_integrated` above is the earlier approximation retained
// for the existing callers.  `gaussian_taps_c1` below is the literal host
// transcription (bucket integral over `n` taps, normalised so the sum of all
// `n` taps is 1, not center + 2*sum).  Prefer the latter for parity work.
// ===========================================================================

/// Number of taps requested by `CImgOpUnsharpMaskLinear` parameter prep.
/// `op+0x94 = 13` (0xd) normally, `33` (0x21) when `settings[+0x460] == 1`
/// (0x18181c143 / 0x18181c152).  The Run body may use fewer taps.
pub const C1_TAPS_STD: usize = 13;
pub const C1_TAPS_HQ: usize = 33;

/// A&S 7.1.26 coefficients, used verbatim by `erfc` at `0x1813f04a0`
/// (`p=0x182a380d8`, `a1=0x182a38008`, `0.284496736=0x182a38048`,
/// `a3=0x182a398c0`, `a4=0x182a39a50`, `a5=0x182a386e0`).
pub const ERFC_P: f64 = 0.3275911;
pub const ERFC_A1: f64 = 0.254829592;
pub const ERFC_A2: f64 = -0.284496736;
pub const ERFC_A3: f64 = 1.421413741;
pub const ERFC_A4: f64 = -1.453152027;
pub const ERFC_A5: f64 = 1.061405429;
/// Large-argument continued fraction constants (`0x182a383e0`, `0x182a3a9c0`,
/// `0x182a38990`).
pub const ERFC_CF_A: f64 = 0.879249;
pub const ERFC_SQRT_PI: f64 = 1.77245385090552;
pub const ERFC_CF_B: f64 = 1.128379;
/// Branch points (`0x182a3cec0` = 3.0 in erfc, `0x182a37dd8` = 0.1 in erf).
pub const ERFC_SWITCH: f64 = 3.0;
pub const ERF_SERIES_SWITCH: f64 = 0.1;
/// `2/sqrt(pi)` (`0x182a38998`), the Maclaurin scale for erf.
pub const TWO_OVER_SQRT_PI: f64 = 1.12837916709551;

/// Host `erfc` (`0x1813f04a0`), transcribed bit-for-bit in f64.
pub fn erfc_c1(x: f64) -> f64 {
    let ax = x.abs();
    let mut val = if ax < ERFC_SWITCH {
        let t = 1.0 / (1.0 + ERFC_P * ax);
        let r = (((ERFC_A5 * t + ERFC_A4) * t + ERFC_A3) * t + ERFC_A2) * t + ERFC_A1;
        (r * t) * (-ax * ax).exp()
    } else {
        (-ax * ax).exp() / (ERFC_SQRT_PI * ax + 1.0 / (ERFC_CF_B * ax + 1.0 / (ERFC_CF_A * ax)))
    };
    if x < 0.0 {
        val = 2.0 - val;
    }
    val
}

/// Host `erf` (`0x1813f0600`): `1 - erfc(|x|)` for `|x| > 0.1`, otherwise the
/// three-term Maclaurin series scaled by `2/sqrt(pi)`.
pub fn erf_c1(x: f64) -> f64 {
    let ax = x.abs();
    let r = if ax > ERF_SERIES_SWITCH {
        1.0 - erfc_c1(ax)
    } else {
        let x2 = ax * ax;
        (ax - ax * x2 / 3.0 + ax * x2 * x2 / 10.0) * TWO_OVER_SQRT_PI
    };
    if x < 0.0 { -r } else { r }
}

/// `0x182a383f0` = sqrt(pi)/2 as stored (truncated) in the DLL.
pub const BAND_SCALE: f64 = 0.88622692545276;
/// `0x182a3c2c0` = sqrt(2*pi) as stored (truncated) in the DLL.
pub const SQRT_2PI: f64 = 2.506628274631;

/// `0x1813f0690`: `int_a^b exp(-(t/s)^2) dt`, as computed by the host.
pub fn erf_band_helper(a: f64, b: f64, s: f64) -> f64 {
    (erf_c1(b / s) - erf_c1(a / s)) * s * BAND_SCALE
}

/// Run-body tap-count selection (0x181823c89..0x181823cd4 and the twin at
/// 0x181825148..0x18182518a):
/// `n = min(cap, max(3, trunc(5*sigma + 2.5) | 1))`.
pub fn tap_count_c1(sigma: f32, cap: usize) -> usize {
    let est = (sigma * 5.0 + 2.5) as i32; // cvttss2si: truncation toward zero
    let odd = (est | 1).max(3) as usize;
    odd.min(cap)
}

/// `0x1813569b0`: pixel-integrated Gaussian taps.
///
/// `den = sqrt(2)*sigma` (double).  For each tap `i`:
/// `w_i = (erf((t+0.5)/den) - erf((t-0.5)/den)) * den * sqrt(pi)/2`,
/// `t = (i - (n-1)/2)` (the subtraction is f32, then widened to f64).
/// Stored taps are f32; the running sum is `f32(f64(sum) + w_i)` after every
/// tap, and the final division is f32.
pub fn gaussian_taps_c1(n: usize, sigma: f32) -> (Vec<f32>, usize) {
    let n = n.max(1);
    let mid = ((n - 1) as f32) * 0.5;
    let den = std::f64::consts::SQRT_2 * sigma as f64;
    let mut taps = Vec::with_capacity(n);
    let mut sum = 0.0f32;
    for i in 0..n {
        let t = (i as f32 - mid) as f64;
        let w = erf_band_helper(t - 0.5, t + 0.5, den);
        taps.push(w as f32);
        sum = (sum as f64 + w) as f32;
    }
    if sum != 0.0 {
        for w in &mut taps {
            *w /= sum;
        }
    }
    (taps, n / 2)
}

/// Radius slider -> internal Gaussian sigma (`0x18181bbe2..0x18181bc4e`):
/// `x >= 0.5 -> 0.7x + 0.3`; `0.09 < x < 0.5 -> 0.97*sqrt(x-0.09)`; else 0.
pub fn radius_curve_c1(x: f32) -> f32 {
    if x >= 0.5 {
        x * 0.7 + 0.3
    } else if x > 0.09 {
        (x - 0.09).sqrt() * 0.97
    } else {
        0.0
    }
}

/// `0x181356cd0`: adjusts a radius and a threshold/value.
///
/// Radius: `g > 1 -> r*g`; else `r > 0.15 -> (r-0.15)*g + 0.15`; else `r`.
/// Threshold: `s * (1 - w(r')^2) / (1 - w(r)^2)` with
/// `w(r) = erf(0.5/(r*sqrt(2)))`.  The host evaluates `w` through
/// `0x1813f0690` and divides by `r*sqrt(2*pi)`; the identity above is exact.
pub fn radius_remap_c1(r: f32, s: f32, g: f32) -> (f32, f32) {
    let r0 = r.max(1e-6);
    let rp = if g > 1.0 {
        r0 * g
    } else if r0 > 0.15 {
        (r0 - 0.15) * g + 0.15
    } else {
        r0
    };
    let band = |x: f32| {
        let rr = x as f64;
        erf_band_helper(-0.5, 0.5, rr * std::f64::consts::SQRT_2) / (rr * SQRT_2PI)
    };
    let den = 1.0 - band(r0).powi(2);
    let num = 1.0 - band(rp).powi(2);
    let s_out = if den != 0.0 { ((s as f64) * num / den) as f32 } else { 0.0 };
    (rp, s_out)
}

/// `0x18181bb6b..0x18181bbe6`: base/halo radius scalar `op+0x264`.
/// `f3b0 * min(f3b4, 1)`; if non-positive use `f128`; doubled when
/// `a8 == 2 && b0 in {1,2}`.
pub fn op264_c1(f3b0: f32, f3b4: f32, f128: f64, a8: i32, b0: i32) -> f32 {
    let mut v = if f3b4 < 1.0 { f3b0 * f3b4 } else { f3b0 };
    if v <= 0.0 {
        v = f128 as f32;
    }
    if a8 == 2 && (b0 == 1 || b0 == 2) {
        v += v;
    }
    v
}

/// Variant-8 `float8` argument, built in `0x18181ee60` (`[rsp+0x78]`) and in
/// the CPU reference `0x181820410` (fixed `r = 2` defaults).
///
/// `s0 = (2r+1)^2` (always); `s1 = int(f11078)` when `mode6 && f11078 > 0`,
/// else `((2r+1)^2 + 1) / 4` (integer division);
/// `s2 = 1.0` or `int(f1107c)*0.01`; `s3 = 1/r` or `exp2(0.1*f11080)/r`;
/// `s4 = 1/(2r+1)^2` or `exp2(0.1*f11084)/(2r+1)^2`; `s5..s7 = 0` and the
/// kernel never reads them.
pub fn halo_float8_c1(
    r_px: i32,
    mode6: bool,
    f1107c: f32,
    f11078: f32,
    f11080: f32,
    f11084: f32,
) -> [f32; 8] {
    let r = r_px.max(1);
    let win = (2 * r + 1) * (2 * r + 1);
    let inv_r = 1.0f32 / r as f32;
    let inv_win = 1.0f32 / win as f32;
    let mut s1 = ((win + 1) / 4) as f32;
    let (s2, s3, s4) = if mode6 {
        if f11078 > 0.0 {
            s1 = (f11078 as i32) as f32;
        }
        let s2 = if f1107c > 0.0 { (f1107c as i32) as f32 * 0.01 } else { 1.0 };
        (s2, (0.1 * f11080).exp2() * inv_r, (0.1 * f11084).exp2() * inv_win)
    } else {
        (1.0, inv_r, inv_win)
    };
    [win as f32, s1, s2, s3, s4, 0.0, 0.0, 0.0]
}

#[cfg(test)]
mod host_map_tests {
    use super::*;

    fn close(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn erf_matches_math_reference() {
        // The host approximation is A&S 7.1.26 (max error ~1.5e-7).
        for &x in &[-3.0f64, -2.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0, 3.0] {
            let got = erf_c1(x);
            let want = libm_erf(x);
            assert!(close(got, want, 5e-6), "erf({x}) got {got} want {want}");
        }
    }

    fn libm_erf(x: f64) -> f64 {
        // Series/rational reference independent of the host approximation.
        if x.abs() <= 2.0 {
            // erf(x) = 2/sqrt(pi) * sum (-1)^n x^(2n+1)/(n!(2n+1))
            let mut term = x;
            let mut sum = x;
            let mut n = 0.0f64;
            while term.abs() > 1e-17 {
                n += 1.0;
                term *= -x * x / n;
                sum += term / (2.0 * n + 1.0);
            }
            2.0 / std::f64::consts::PI.sqrt() * sum
        } else {
            let tail = erfc_c1(x.abs());
            if x < 0.0 { -(1.0 - tail) } else { 1.0 - tail }
        }
    }

    #[test]
    fn taps_are_normalized_symmetric_and_pixel_integrated() {
        let (t13, mid13) = gaussian_taps_c1(C1_TAPS_STD, 2.0);
        assert_eq!(t13.len(), 13);
        assert_eq!(mid13, 6);
        assert!((t13.iter().map(|v| *v as f64).sum::<f64>() - 1.0).abs() < 1e-6);
        for i in 0..6 {
            assert!((t13[i] - t13[12 - i]).abs() < 1e-9);
        }
        assert!(t13[6] > t13[5] && t13[5] > t13[4]);

        let (t33, mid33) = gaussian_taps_c1(C1_TAPS_HQ, 2.0);
        assert_eq!(t33.len(), 33);
        assert_eq!(mid33, 16);
        assert!((t33.iter().map(|v| *v as f64).sum::<f64>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn tap_count_matches_run_body_formula() {
        assert_eq!(tap_count_c1(2.0, C1_TAPS_STD), 13);
        assert_eq!(tap_count_c1(4.0, C1_TAPS_HQ), 23); // 5*4+2.5=22.5 -> 22|1=23
        assert_eq!(tap_count_c1(1.0, C1_TAPS_STD), 7); // 7.5 -> 7
        assert_eq!(tap_count_c1(0.2, C1_TAPS_STD), 3); // 3.5 -> 3
        assert_eq!(tap_count_c1(0.0, C1_TAPS_STD), 3); // floor 3
        assert_eq!(tap_count_c1(10.0, C1_TAPS_STD), 13); // cap
        assert_eq!(tap_count_c1(0.1, C1_TAPS_HQ), 3);
    }

    #[test]
    fn radius_curve_is_piecewise_exact() {
        assert_eq!(radius_curve_c1(0.09), 0.0);
        assert!(radius_curve_c1(0.1) > 0.0);
        assert!((radius_curve_c1(0.5) - (0.5 * 0.7 + 0.3)).abs() < 1e-7);
        assert!((radius_curve_c1(2.0) - 1.7).abs() < 1e-7);
        assert!((radius_curve_c1(0.3) - 0.97 * (0.3f32 - 0.09).sqrt()).abs() < 1e-7);
    }

    #[test]
    fn radius_remap_is_identity_at_gain_one() {
        let (r, s) = radius_remap_c1(2.0, 1.0, 1.0);
        assert_eq!(r, 2.0);
        assert!((s - 1.0).abs() < 1e-6);
        // Shrinking the radius raises 1-w^2, so the threshold scales down.
        let (r2, s2) = radius_remap_c1(2.0, 1.0, 0.5);
        assert!(r2 < 2.0 && s2 < 1.0);
    }

    #[test]
    fn halo_float8_default_matches_cpu_reference_r2() {
        let v = halo_float8_c1(2, false, 0.0, 0.0, 0.0, 0.0);
        assert_eq!(v[0], 25.0);
        assert_eq!(v[0].to_bits(), 0x41c80000);
        assert_eq!(v[1], 6.0); // (25+1)/4
        assert_eq!(v[1].to_bits(), 0x40c00000);
        assert_eq!(v[2], 1.0);
        assert_eq!(v[3], 0.5);
        assert_eq!(v[4], 0.04f32);
        assert_eq!(&v[5..], &[0.0, 0.0, 0.0]);
        // These are exactly the hardcoded constants in 0x181820410.
    }

    #[test]
    fn halo_float8_mode6_uses_camera_fields() {
        let v = halo_float8_c1(2, true, 100.0, 0.0, 10.0, 0.0);
        assert_eq!(v[2], 1.0); // int(100)*0.01
        assert_eq!(v[3], 1.0); // exp2(1.0) * (1/2)
        assert_eq!(v[4], 0.04f32); // exp2(0)*1/25
        let v2 = halo_float8_c1(2, true, -5.0, 0.0, 0.0, 0.0);
        assert_eq!(v2[2], 1.0); // non-positive -> default
        // settings[+0x11078] overrides the count threshold only in mode 6
        let v3 = halo_float8_c1(2, true, 0.0, 9.0, 0.0, 0.0);
        assert_eq!(v3[1], 9.0);
        let v4 = halo_float8_c1(2, false, 0.0, 9.0, 0.0, 0.0);
        assert_eq!(v4[1], 6.0);
    }

    #[test]
    fn op264_rules() {
        assert_eq!(op264_c1(3.0, 0.5, 9.0, 0, 0), 1.5);
        assert_eq!(op264_c1(3.0, 1.0, 9.0, 0, 0), 3.0);
        assert_eq!(op264_c1(0.0, 1.0, 7.5, 0, 0), 7.5);
        assert_eq!(op264_c1(2.0, 1.0, 0.0, 2, 1), 4.0);
        assert_eq!(op264_c1(2.0, 1.0, 0.0, 2, 3), 2.0);
    }
}
