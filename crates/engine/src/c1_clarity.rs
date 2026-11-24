//! Capture One Clarity / Structure native formulas.
//!
//! Evidence: current `ImgCoreDll.dll` SHA-256
//! `ca02b6ac2786e9a100f63221df7340e5df210416eb0bd53a4bb1bfc2306a44a4`,
//! OpenCL programs (each occurs exactly once in the DLL):
//! - `p_041294800.cl` (Clarity2012, 18,119 B, file offset 41,331,504),
//!   variants 1..9, SHA-256 `5220221c…bec0a8`
//! - `p_041286768.cl` (ApplyClarity, 3,901 B, file offset 41,326,832)
//! - `p_041282448.cl` (ApplyRGBClarity, 4,299 B, file offset 41,322,512)
//! - `p_041874080.cl` (Structure2012, 3,601 B, file offset 41,910,768),
//!   variants 0/1, SHA-256 `59f92ff3…0cb12`
//!
//! These are two different operators. Clarity is a low-frequency rejection
//! (additive high pass in the legacy kernels, multiplicative scale in the 2012
//! kernels); Structure is a multiplicative local-contrast factor built from
//! two low-frequency buffers. The old engine identity
//! `Structure+100 == Clarity+50` is wrong: native +50/+50 outputs differ on
//! 99.69 % of pixels (mean 858, max 2682 in 16-bit code values,
//! `c1_clarity_native_20261123/out/case_diffs.json`).
//!
//! # Units
//!
//! Two unit conventions appear in the native code and are kept here:
//!
//! - `clarity_ycbcr`, `apply_rgb_clarity`, `block_mean_*`, `sample_grid` are
//!   unit agnostic: `low`/`grid` must be in the same units as the RGB input
//!   (normalised 0..=1 or 16-bit code values). This is the legacy
//!   `CImgOpClarity` path, whose kernels normalise the bilinear grid sum by
//!   `1 / (cell * cell * 4096)` (`ApplyClarity`, cell fixed at 8) or by
//!   `1 / cell^6` (`ApplyRGBClarity`).
//! - `clarity2012_*`, `structure_luma` operate on **16-bit code values**
//!   (0..=65535), because the 2012 kernels add 16 and use `2.5 / 65535`.
//!
//! # Host schedule (current DLL, RTTI/vtable verified)
//!
//! `CImgOpClarity2012` TypeDescriptor RVA `0x2ce3a28`, COL `0x2a6e428`,
//! vtable RVA `0x27706d0`, execution slot 11 = RVA `0x16d14a0` (8,890 B).
//! Its executor runs exactly **three passes**. The pass loop counter is
//! `[rbp-0x10]`, bounded at RVA `0x16d36d8` (`cmp ecx, 3; jge done`), and the
//! selector switch at RVA `0x16d15aa` maps pass 0/1/2 to selectors **3/0/1**
//! (the fourth case `op+0x88` is unreachable with a bound of 3). The object
//! field `op+0x88` (copied from `param+0x1d5c` in the parameter setup at RVA
//! `0x16cb0c0`) is the method; `op+0x8c` (from `param+0x1d54`) is the amount.
//! A pass receives a non-zero amount only when its selector equals `op+0x88`
//! (RVA `0x16d15f2`). Without a mask the amount is mapped by
//! `x*0.012` for `x >= 0` / `x*0.003` for `x < 0`, clamped to `>= -1`, and
//! doubled for selector 3 when negative (RVA `0x16d1e06..0x16d1e52`).
//! A method value of 2 matches no pass, which is why "Classic" is not produced
//! by this executor (see `clarity2012_pass_enabled`).
//!
//! The low-frequency buffers are built at `ceil(width / 8)` per axis: the
//! executor asserts `pyramidBuf0->GetInnerWidth() ==
//! (OperationFullSourceWidth + 7) / 8` (string at RVA `0x18276be90`). Three
//! grid planes are allocated as one `width x 3*height` buffer and exposed as
//! rects at y offsets `0`, `height`, `2*height` (RVA `0x16d1f54..0x16d205a`).
//! The number of *blur levels*, their radii and their weights are **not**
//! statically pinned; see `SUBAGENT-CLARITY.md`.
//!
//! This module is a data/algorithm reference; it does not dispatch GPU work.

/// Clarity method as stored by `ZCLARITYMETHOD` (0 Neutral, 1 Punch,
/// 2 Classic, 3 Natural/Contrast). The legacy `ApplyRGBClarity` kernel only
/// tests for Punch (1); the 2012 executor uses the same value as a pass
/// selector (see module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ClarityMethod {
    Neutral,
    Punch,
    Classic,
    Natural,
}

impl ClarityMethod {
    pub fn from_c1(value: i32) -> Self {
        match value {
            0 => Self::Neutral,
            1 => Self::Punch,
            2 => Self::Classic,
            _ => Self::Natural,
        }
    }

    pub fn to_c1(self) -> i32 {
        match self {
            Self::Neutral => 0,
            Self::Punch => 1,
            Self::Classic => 2,
            Self::Natural => 3,
        }
    }
}

/// `ApplyRGBClarity` amount map: `z3741ed4e99` followed by `zed389ca108`.
///
/// ```text
/// a = base * 0.005
/// if a < 0: a *= 0.5
/// if method == Punch(1): a *= 1.2
/// if cell < 8: a *= 2
/// ```
///
/// `base` is the C1 amount *plus* the mask term; the kernel adds
/// `sum(mask/255 * weight_i)` before this map. Use
/// [`clarity_amount_with_mask`] when a mask is present.
pub fn clarity_amount(method: ClarityMethod, base: f32, downscale: u32) -> f32 {
    clarity_amount_with_mask(method, base, 0.0, downscale)
}

/// Same map as [`clarity_amount`], with the mask term folded in at the
/// position the kernel uses (before the 0.005 scale and the sign test).
pub fn clarity_amount_with_mask(
    method: ClarityMethod,
    base: f32,
    mask_weight: f32,
    downscale: u32,
) -> f32 {
    let mut amount = (base + mask_weight) * 0.005;
    if amount < 0.0 {
        amount *= 0.5;
    }
    if method == ClarityMethod::Punch {
        amount *= 1.2;
    }
    if downscale < 8 {
        amount *= 2.0;
    }
    amount
}

/// Grid normalisation constant used by the kernels.
///
/// `ApplyClarity` uses `1 / (64 * 4096)` with a hard-coded 8-pixel cell and
/// `ApplyRGBClarity` uses `1 / ((cell*cell)^3)` = `1 / cell^6`. For cell = 8
/// both are `1 / 262144`. The grid buffers themselves are built by host code
/// that is not reconstructible from the kernel text alone; `block_mean_*`
/// below returns a plain mean for reference use.
pub fn clarity_grid_norm(cell: u32) -> f32 {
    let c = (cell.max(1) as f32) * (cell.max(1) as f32);
    1.0 / (c * c * c)
}

/// BT.601 split used by `ApplyClarity` / `ApplyRGBClarity`.
pub fn rgb_to_ycbcr(rgb: [f32; 3]) -> (f32, f32, f32) {
    let [r, g, b] = rgb;
    (
        0.299 * r + 0.587 * g + 0.114 * b,
        0.596 * r - 0.275 * g - 0.321 * b,
        0.212 * r - 0.523 * g + 0.311 * b,
    )
}

/// Inverse used by the kernel. These are Capture One's own six coefficients,
/// not the exact mathematical inverse of the forward matrix.
pub fn ycbcr_to_rgb(y: f32, cb: f32, cr: f32) -> [f32; 3] {
    [
        y + 0.956 * cb + 0.621 * cr,
        y - 0.272 * cb - 0.647 * cr,
        y - 1.105 * cb + 1.702 * cr,
    ]
}

/// One low-frequency grid value for a `cell x cell` block, in 0..=1.
/// `ApplyClarity` uses a 12-bit grid (`sum / (cell*4096)`, cell fixed at 8);
/// for `f32` input this is the equivalent block mean.
pub fn block_mean_f32(src: &[f32], width: usize, height: usize, cell: usize) -> Vec<f32> {
    let cell = cell.max(1);
    let gw = (width + cell - 1) / cell;
    let gh = (height + cell - 1) / cell;
    let mut out = vec![0.0; gw * gh];
    for gy in 0..gh {
        for gx in 0..gw {
            let mut sum = 0.0f64;
            let mut count = 0u32;
            for y in gy * cell..((gy + 1) * cell).min(height) {
                for x in gx * cell..((gx + 1) * cell).min(width) {
                    sum += f64::from(src[y * width + x]);
                    count += 1;
                }
            }
            out[gy * gw + gx] = if count == 0 { 0.0 } else { (sum / f64::from(count)) as f32 };
        }
    }
    out
}

/// Convenience 12-bit block mean for the legacy `ApplyClarity` grid: the
/// 16-bit sample is shifted to 12 bits (`>> 4`) and divided by 4096, giving a
/// normalised block mean. This is **not** the Clarity2012 variant 1/2
/// transform (those take `sqrt(v + offset)`; see
/// [`clarity2012_encode_rgb`] and [`clarity2012_encode_luma`]).
pub fn block_mean_u16_12bit(src: &[u16], width: usize, height: usize, cell: usize) -> Vec<f32> {
    let cell = cell.max(1);
    let gw = (width + cell - 1) / cell;
    let gh = (height + cell - 1) / cell;
    let mut out = vec![0.0; gw * gh];
    for gy in 0..gh {
        for gx in 0..gw {
            let mut sum = 0.0f64;
            let mut count = 0u32;
            for y in gy * cell..((gy + 1) * cell).min(height) {
                for x in gx * cell..((gx + 1) * cell).min(width) {
                    sum += f64::from(src[y * width + x] >> 4);
                    count += 1;
                }
            }
            out[gy * gw + gx] =
                if count == 0 { 0.0 } else { (sum / f64::from(count) / 4096.0) as f32 };
        }
    }
    out
}

/// Bilinear grid sample at a full-resolution pixel coordinate. `ApplyClarity`
/// uses `x >> 3` and `x & 7`; `ApplyRGBClarity` uses `x / cell` and
/// `x & (cell - 1)`, so its cell must be a power of two. This helper uses
/// `%`/`/`, which is equivalent for powers of two and is also defined for
/// other cell sizes.
pub fn sample_grid(grid: &[f32], gw: usize, gh: usize, x: usize, y: usize, cell: usize) -> f32 {
    if gw == 0 || gh == 0 {
        return 0.0;
    }
    let cell = cell.max(1);
    let gx = (x / cell).min(gw - 1);
    let gy = (y / cell).min(gh - 1);
    let fx = (x % cell) as f32 / cell as f32;
    let fy = (y % cell) as f32 / cell as f32;
    let gx1 = (gx + 1).min(gw - 1);
    let gy1 = (gy + 1).min(gh - 1);
    let top = grid[gy * gw + gx] * (1.0 - fx) + grid[gy * gw + gx1] * fx;
    let bottom = grid[gy1 * gw + gx] * (1.0 - fx) + grid[gy1 * gw + gx1] * fx;
    top * (1.0 - fy) + bottom * fy
}

/// `ApplyClarity` additive high-pass on Y, chroma unchanged.
///
/// `base_amount` is the kernel's `z77eaff834e` weight (the legacy kernel does
/// **not** apply the 0.005 map), and `mask_weight` is the optional local mask
/// term `sum(mask/255 * weight_i)`. The kernel clamps the result to
/// 0..=65535; callers working in normalised units should clamp to 0..=1.
pub fn clarity_ycbcr(
    rgb: [f32; 3],
    low: f32,
    base_amount: f32,
    mask_weight: f32,
) -> [f32; 3] {
    let (y, cb, cr) = rgb_to_ycbcr(rgb);
    let amount = base_amount + mask_weight;
    let y2 = y + (y - low) * amount;
    ycbcr_to_rgb(y2, cb, cr)
}

/// `ApplyRGBClarity` per-channel additive step.
///
/// ```text
/// delta = RGB - low
/// RGB'  = RGB + delta * w
/// ```
///
/// `delta` is computed from the *input* RGB even when the kernel accumulates
/// into an existing destination (`z5599675f2e == 1`); that accumulation flag
/// is not modelled here. `low` is the bilinear grid sample normalised the same
/// way as the kernel (`1/cell^6`, see [`clarity_grid_norm`]).
pub fn apply_rgb_clarity(
    rgb: [f32; 3],
    low: [f32; 3],
    method: ClarityMethod,
    base_amount: f32,
    mask_weight: f32,
    cell: u32,
) -> [f32; 3] {
    let w = clarity_amount_with_mask(method, base_amount, mask_weight, cell);
    [
        rgb[0] + (rgb[0] - low[0]) * w,
        rgb[1] + (rgb[1] - low[1]) * w,
        rgb[2] + (rgb[2] - low[2]) * w,
    ]
}

/// `Clarity2012` variant 1: `float4 sqrt(RGB + offset)` pyramid encode.
/// Input is 3-channel 16-bit code values; the kernel reads `ushort3` and
/// writes `float4` (alpha 0).
pub fn clarity2012_encode_rgb(rgb: [u16; 3], offset: f32) -> [f32; 3] {
    [
        (f32::from(rgb[0]) + offset).sqrt(),
        (f32::from(rgb[1]) + offset).sqrt(),
        (f32::from(rgb[2]) + offset).sqrt(),
    ]
}

/// `Clarity2012` variant 2: `float sqrt(luma(RGB) + offset)` pyramid encode.
pub fn clarity2012_encode_luma(rgb: [u16; 3], offset: f32) -> f32 {
    let y = 0.299 * f32::from(rgb[0]) + 0.587 * f32::from(rgb[1]) + 0.114 * f32::from(rgb[2]);
    (y + offset).sqrt()
}

/// `Clarity2012` amount map `z3741ed4e99` used by variants 4/6/8:
///
/// ```text
/// w = x * 0.012
/// if w < 0: w = x * 0.003
/// w = max(-1, w)
/// ```
pub fn clarity2012_amount(x: f32) -> f32 {
    let mut w = x * 0.012;
    if w < 0.0 {
        w = x * 0.003;
    }
    w.max(-1.0)
}

/// Pass selectors used by the `CImgOpClarity2012` executor: pass 0/1/2 map to
/// method codes **3/0/1** (RVA `0x16d15aa..0x16d15e0`). A method code of 2
/// matches no pass.
pub fn clarity2012_pass_selectors() -> [i32; 3] {
    [3, 0, 1]
}

/// Whether the executor lets pass `pass` (0..=2) carry the op amount:
/// `selector == op+0x88` (RVA `0x16d15f2`).
pub fn clarity2012_pass_enabled(pass: usize, method_code: i32) -> bool {
    clarity2012_pass_selectors()
        .get(pass)
        .map(|&selector| selector == method_code)
        .unwrap_or(false)
}

/// Host-side amount mapping for a pass (RVA `0x16d1e06..0x16d1e52`):
/// [`clarity2012_amount`] followed by `w *= 2` for selector 3 when `w < 0`,
/// then `max(-1, w)`. With a mask present the executor skips this map and uses
/// the raw amount.
pub fn clarity2012_amount_for_pass(selector: i32, x: f32) -> f32 {
    let mut w = clarity2012_amount(x);
    if selector == 3 && w < 0.0 {
        w *= 2.0;
    }
    w.max(-1.0)
}

fn luma601(rgb: [f32; 3]) -> f32 {
    0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
}

fn max0(rgb: [f32; 3]) -> [f32; 3] {
    [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)]
}

/// `Clarity2012` variant 3 (`p_041294800.cl`): one low buffer.
///
/// ```text
/// Y = luma + 16
/// if amount > 0:  low2 = low*low
///                 d = (Y - low2) * amount / Y
///                 d = d > 0 ? d/(1 + d*compress) : d/(1 - d*compress*16)
///                 s = 1 + d
/// else:           s = (1 + (sqrt(Y) - low) * amount / sqrt(Y))^2
/// RGB *= s, clamp >= 0
/// ```
///
/// `low` is in `sqrt` domain for the negative branch and is squared by the
/// kernel for the positive branch.
pub fn clarity2012_v3(rgb: [f32; 3], low: f32, amount: f32, compress: f32) -> [f32; 3] {
    let y = luma601(rgb) + 16.0;
    let scale = if amount > 0.0 {
        let low2 = low * low;
        let mut d = (y - low2) * amount / y;
        if d > 0.0 {
            d /= 1.0 + d * compress;
        } else {
            d /= 1.0 - d * (compress * 16.0);
        }
        1.0 + d
    } else {
        let sqrt_y = y.sqrt();
        let d = (sqrt_y - low) * amount / sqrt_y;
        let s = 1.0 + d;
        s * s
    };
    max0([rgb[0] * scale, rgb[1] * scale, rgb[2] * scale])
}

/// `Clarity2012` variant 4: variant 3 with a separate low buffer for the
/// negative (sqrt) branch.
pub fn clarity2012_v4(
    rgb: [f32; 3],
    low_pos: f32,
    low_neg: f32,
    amount: f32,
    compress: f32,
) -> [f32; 3] {
    let y = luma601(rgb) + 16.0;
    let scale = if amount > 0.0 {
        let low2 = low_pos * low_pos;
        let mut d = (y - low2) * amount / y;
        if d > 0.0 {
            d /= 1.0 + d * compress;
        } else {
            d /= 1.0 - d * (compress * 16.0);
        }
        1.0 + d
    } else {
        let sqrt_y = y.sqrt();
        let d = (sqrt_y - low_neg) * amount / sqrt_y;
        let s = 1.0 + d;
        s * s
    };
    max0([rgb[0] * scale, rgb[1] * scale, rgb[2] * scale])
}

/// `Clarity2012` variant 5: per-channel additive step in the `RGB + 16`
/// domain, with only negative deltas compressed.
///
/// ```text
/// for each channel C:
///   d = (C + 16 - low_c*low_c) * amount
///   if d < 0: d = d * (C + 16) / ((C + 16) - d * compress)
///   C' = C + d
/// ```
pub fn clarity2012_v5(
    rgb: [f32; 3],
    low_rgb: [f32; 3],
    amount: f32,
    compress: f32,
) -> [f32; 3] {
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let y = rgb[c] + 16.0;
        let low2 = low_rgb[c] * low_rgb[c];
        let mut d = (y - low2) * amount;
        if d < 0.0 {
            d = d * y / (y - d * compress);
        }
        out[c] = rgb[c] + d;
    }
    max0(out)
}

/// `Clarity2012` variant 6: variant 5 with the amount map
/// [`clarity2012_amount`] and a second low buffer selected when `w < 0`.
/// The kernel's output expression is `Y + d - 16` with `Y = RGB` (no `+ 16`
/// here); this is only physically consistent when the input already carries
/// the `+ 16` offset used by variant 5. The literal kernel form is preserved.
pub fn clarity2012_v6(
    rgb: [f32; 3],
    low_pos_rgb: [f32; 3],
    low_neg_rgb: [f32; 3],
    raw_amount: f32,
    compress: f32,
) -> [f32; 3] {
    let w = clarity2012_amount(raw_amount);
    let low = if w < 0.0 { low_neg_rgb } else { low_pos_rgb };
    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let y = rgb[c];
        let low2 = low[c] * low[c];
        let mut d = (y - low2) * w;
        if d < 0.0 {
            d = d * y / (y - d * compress);
        }
        out[c] = y + d - 16.0;
    }
    max0(out)
}

/// `Clarity2012` variant 7: one low buffer, signed-delta branches.
///
/// ```text
/// Y = luma + 16; low2 = low * low; d0 = Y - low2
/// if d0 < 0: d = d0 * Y / (Y + 6|d0|); d *= amount; RGB += d
/// else:      k = Y * (2.5 / 65535)
///            d = d0 * low2 / (low2 + (0.3 + k*k) |d0|); d *= amount
///            RGB *= 1 + d / Y
/// ```
pub fn clarity2012_v7(rgb: [f32; 3], low: f32, amount: f32) -> [f32; 3] {
    let y = luma601(rgb) + 16.0;
    let low2 = low * low;
    let d0 = y - low2;
    if d0 < 0.0 {
        let d = d0 * y / (y + 6.0 * d0.abs()) * amount;
        max0([rgb[0] + d, rgb[1] + d, rgb[2] + d])
    } else {
        let k = y * (2.5 / 65535.0);
        let d = d0 * low2 / (low2 + (0.3 + k * k) * d0.abs()) * amount;
        let s = 1.0 + d / y;
        max0([rgb[0] * s, rgb[1] * s, rgb[2] * s])
    }
}

/// `Clarity2012` variant 8: variant 7 with the amount map and a second low
/// buffer for negative amounts. The raw amount is doubled when negative
/// *before* [`clarity2012_amount`] is applied.
pub fn clarity2012_v8(
    rgb: [f32; 3],
    low_pos: f32,
    low_neg: f32,
    raw_amount: f32,
) -> [f32; 3] {
    let doubled = if raw_amount < 0.0 { raw_amount * 2.0 } else { raw_amount };
    let w = clarity2012_amount(doubled);
    let low = if w < 0.0 { low_neg } else { low_pos };
    let y = luma601(rgb) + 16.0;
    let low2 = low * low;
    let d0 = y - low2;
    if d0 < 0.0 {
        let d = d0 * y / (y + 6.0 * d0.abs()) * w;
        max0([rgb[0] + d, rgb[1] + d, rgb[2] + d])
    } else {
        let k = y * (2.5 / 65535.0);
        let d = d0 * low2 / (low2 + (0.3 + k * k) * d0.abs()) * w;
        let s = 1.0 + d / y;
        max0([rgb[0] * s, rgb[1] * s, rgb[2] * s])
    }
}

/// `Clarity2012` variant 9: `dst = max(dst + (high - src), 0)`. This is a
/// composition/restore pass, not a contrast operator.
pub fn clarity2012_v9(dst: [f32; 3], src: [f32; 3], high: [f32; 3]) -> [f32; 3] {
    max0([
        dst[0] + (high[0] - src[0]),
        dst[1] + (high[1] - src[1]),
        dst[2] + (high[2] - src[2]),
    ])
}

/// Compatibility shim for the old hybrid helper. `use_sqrt_domain == true`
/// selects variant 3; `false` selects variant 7. The old implementation
/// switched on the sign of `amount`, which is not how the native kernels
/// choose a variant; prefer [`clarity2012_v3`] / [`clarity2012_v7`] directly.
pub fn clarity_rgb_step(
    rgb: [f32; 3],
    low: f32,
    amount: f32,
    compress: f32,
    use_sqrt_domain: bool,
) -> [f32; 3] {
    if use_sqrt_domain {
        clarity2012_v3(rgb, low, amount, compress)
    } else {
        clarity2012_v7(rgb, low, amount)
    }
}

/// `Structure2012` local-contrast factor (variant 1).
///
/// `amount` is the **mapped** weight from [`structure_amount`] (the kernel
/// adds the mask term first and then applies the 0.01/0.02 map). `low1` and
/// `low2` are two low-frequency buffers in the same domain (the kernel uses
/// the `sqrt(max(0, luma) + 16)` plane built by variant 0).
pub fn structure_factor(low1: f32, low2: f32, amount: f32, param_y: f32) -> f32 {
    let mut t = (low1 - low2) * amount / low1.max(1e-6);
    if t < 0.0 && amount > 0.0 {
        t /= 1.0 - t * param_y;
    }
    let s = (1.0 + t).max(0.0);
    s * s
}

/// `Structure2012` amount map `zed389ca108`: negative amounts use 0.01,
/// positive use 0.02, clamped to -1. The mask term is added before this map.
pub fn structure_amount(base: f32) -> f32 {
    let a = if base < 0.0 { base * 0.01 } else { base * 0.02 };
    a.max(-1.0)
}

/// Convenience wrapper: map `base_amount + mask_weight`, build the factor from
/// the two low buffers and multiply RGB by it (kernel variant 1).
pub fn apply_structure(
    rgb: [f32; 3],
    low1: f32,
    low2: f32,
    base_amount: f32,
    mask_weight: f32,
    param_y: f32,
) -> [f32; 3] {
    let s = structure_factor(low1, low2, structure_amount(base_amount + mask_weight), param_y);
    [rgb[0] * s, rgb[1] * s, rgb[2] * s]
}

/// `Structure2012` variant 0 luma pre-transfer: `sqrt(max(0, luma) + 16)`.
/// Operates on 16-bit code values, like the kernel.
pub fn structure_luma(rgb: [f32; 3]) -> f32 {
    let y = luma601(rgb).max(0.0);
    (y + 16.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, eps: f32) {
        assert!((a - b).abs() <= eps, "{a} vs {b} (eps {eps})");
    }

    fn close3(a: [f32; 3], b: [f32; 3], eps: f32) {
        for i in 0..3 {
            close(a[i], b[i], eps);
        }
    }

    #[test]
    fn flat_field_is_unchanged_by_clarity_and_structure() {
        let rgb = [0.4, 0.4, 0.4];
        let (y, _, _) = rgb_to_ycbcr(rgb);
        let out = clarity_ycbcr(rgb, y, 0.5, 0.0);
        for c in out {
            assert!((c - 0.4).abs() < 1e-4, "{out:?}");
        }
        let s = structure_factor(0.4, 0.4, 0.5, 0.0);
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn clarity_raises_local_contrast_above_low_pass() {
        let rgb = [0.6, 0.6, 0.6];
        let out = clarity_ycbcr(rgb, 0.4, 0.5, 0.0);
        assert!(out[0] > rgb[0] && out[1] > rgb[1] && out[2] > rgb[2]);
    }

    #[test]
    fn ycbcr_uses_the_six_native_coefficients() {
        // Forward matrix from ApplyClarity / ApplyRGBClarity.
        assert_eq!(rgb_to_ycbcr([1.0, 0.0, 0.0]), (0.299, 0.596, 0.212));
        assert_eq!(rgb_to_ycbcr([0.0, 1.0, 0.0]), (0.587, -0.275, -0.523));
        assert_eq!(rgb_to_ycbcr([0.0, 0.0, 1.0]), (0.114, -0.321, 0.311));
        // Capture One's own inverse, not the true mathematical inverse.
        assert_eq!(ycbcr_to_rgb(1.0, 0.0, 0.0), [1.0, 1.0, 1.0]);
        assert_eq!(ycbcr_to_rgb(0.0, 1.0, 0.0), [0.956, -0.272, -1.105]);
        assert_eq!(ycbcr_to_rgb(0.0, 0.0, 1.0), [0.621, -0.647, 1.702]);
    }

    #[test]
    fn clarity_is_additive_in_y_structure_is_multiplicative() {
        // On a neutral input Cb = Cr = 0, so ApplyClarity's Y-only change
        // gives the same delta on all three channels. (On saturated colours
        // the six inverse coefficients do not round-trip the input, so the
        // per-channel deltas pick up a ~1e-4 residual; that is the kernel's
        // own YCbCr matrix, not a bug in this transcription.)
        let rgb = [0.5, 0.5, 0.5];
        let out = clarity_ycbcr(rgb, 0.4, 0.5, 0.0);
        let d0 = out[0] - rgb[0];
        let d1 = out[1] - rgb[1];
        let d2 = out[2] - rgb[2];
        close(d1, d0, 1e-6);
        close(d2, d0, 1e-6);

        // Structure multiplies all channels by one factor, so the ratios are
        // equal.
        let s = structure_factor(0.45, 0.35, structure_amount(50.0), 0.0);
        let st = apply_structure(rgb, 0.45, 0.35, 50.0, 0.0, 0.0);
        close(st[0] / rgb[0], s, 1e-6);
        close(st[1] / rgb[1], s, 1e-6);
        close(st[2] / rgb[2], s, 1e-6);
        assert!((st[0] - out[0]).abs() > 0.05, "{st:?} vs {out:?}");

        // And the two native kernels are not scalar multiples of each other.
        let scale = out[1] / rgb[1];
        assert!((st[2] - out[2] * scale).abs() > 1e-3);
    }

    #[test]
    fn structure_factor_matches_the_kernel_formula() {
        // t = (0.5 - 0.4) * 0.5 / 0.5 = 0.1; s = 1.1^2
        let s = structure_factor(0.5, 0.4, 0.5, 0.0);
        assert!((s - 1.21).abs() < 1e-5, "{s}");
    }

    #[test]
    fn structure_negative_t_branch_is_compressed() {
        // t = (100 - 110) / 100 = -0.1; t / (1 - t*0.1) = -0.0990099
        let s = structure_factor(100.0, 110.0, structure_amount(50.0), 0.1);
        close(s, 0.811_782_8, 1e-5);

        // Negative weight does not take the rational branch even when t < 0.
        let s_neg = structure_factor(110.0, 100.0, structure_amount(-50.0), 0.1);
        let expected_base: f32 = 1.0 - 0.045_454_5;
        close(s_neg, expected_base.powi(2), 1e-5);
    }

    #[test]
    fn structure_amount_map_and_mask_order() {
        close(structure_amount(50.0), 1.0, 1e-7);
        close(structure_amount(-50.0), -0.5, 1e-7);
        close(structure_amount(-500.0), -1.0, 1e-7);
        close(structure_amount(0.0), 0.0, 1e-7);
        // Mask is added before the 0.02 map.
        close(structure_amount(50.0 + (-25.0)), 0.5, 1e-7);
    }

    #[test]
    fn structure_luma_matches_variant_0() {
        let rgb = [65535.0, 0.0, 0.0];
        let expected = (0.299 * 65535.0 + 16.0f32).sqrt();
        close(structure_luma(rgb), expected, 1e-4);
    }

    #[test]
    fn punch_is_stronger_than_neutral() {
        let a = clarity_amount(ClarityMethod::Neutral, 50.0, 8);
        let b = clarity_amount(ClarityMethod::Punch, 50.0, 8);
        close(b, a * 1.2, 1e-7);
        // Punch's 1.2 multiplier is applied before the downscale doubling.
        let c = clarity_amount(ClarityMethod::Punch, 50.0, 4);
        close(c, a * 1.2 * 2.0, 1e-7);
    }

    #[test]
    fn downscale_below_eight_doubles_amount() {
        let base = clarity_amount(ClarityMethod::Neutral, 50.0, 8);
        close(base, 0.25, 1e-7);
        close(clarity_amount(ClarityMethod::Neutral, 50.0, 7), 0.5, 1e-7);
        close(clarity_amount(ClarityMethod::Neutral, 50.0, 4), 0.5, 1e-7);
    }

    #[test]
    fn negative_amount_is_halved_after_mask_addition() {
        close(clarity_amount(ClarityMethod::Neutral, -50.0, 8), -0.125, 1e-7);
        close(clarity_amount(ClarityMethod::Punch, -50.0, 8), -0.15, 1e-6);
        close(clarity_amount(ClarityMethod::Neutral, -50.0, 4), -0.25, 1e-7);
        // base 10 + mask -210 = -200, then 0.005 -> -1, then halve.
        close(clarity_amount_with_mask(ClarityMethod::Neutral, 10.0, -210.0, 8), -0.5, 1e-7);
        // Mapping the base first and adding the mask later gives a different
        // (wrong) result, so this pins the kernel order.
        close(clarity_amount(ClarityMethod::Neutral, 10.0, 8), 0.05, 1e-7);
    }

    #[test]
    fn clarity2012_amount_map_and_clamp() {
        close(clarity2012_amount(100.0), 1.2, 1e-6);
        close(clarity2012_amount(-100.0), -0.3, 1e-6);
        close(clarity2012_amount(-400.0), -1.0, 1e-7);
        close(clarity2012_amount(0.0), 0.0, 1e-7);
        close(clarity2012_amount(10.0), 0.12, 1e-7);
    }

    #[test]
    fn clarity2012_pass_selector_map_is_3_0_1() {
        assert_eq!(clarity2012_pass_selectors(), [3, 0, 1]);
        assert!(clarity2012_pass_enabled(0, 3));
        assert!(clarity2012_pass_enabled(1, 0));
        assert!(clarity2012_pass_enabled(2, 1));
        assert!(!clarity2012_pass_enabled(0, 0));
        // Method code 2 (Classic) matches no pass in this executor.
        assert!(!clarity2012_pass_enabled(0, 2));
        assert!(!clarity2012_pass_enabled(1, 2));
        assert!(!clarity2012_pass_enabled(2, 2));
        assert!(!clarity2012_pass_enabled(3, 3));
    }

    #[test]
    fn clarity2012_host_amount_doubles_negative_only_for_selector_3() {
        close(clarity2012_amount_for_pass(3, -100.0), -0.6, 1e-6);
        close(clarity2012_amount_for_pass(0, -100.0), -0.3, 1e-6);
        close(clarity2012_amount_for_pass(1, -100.0), -0.3, 1e-6);
        close(clarity2012_amount_for_pass(3, 100.0), 1.2, 1e-6);
        close(clarity2012_amount_for_pass(3, -400.0), -1.0, 1e-7);
    }

    #[test]
    fn clarity2012_v3_positive_and_negative_branches() {
        let rgb = [10000.0, 10000.0, 10000.0];
        let y = 0.299 * 10000.0 + 0.587 * 10000.0 + 0.114 * 10000.0 + 16.0;

        // amount > 0, d > 0: compressed by 1 + d*compress.
        let d = (y - 100.0f32 * 100.0) * 1.0 / y;
        let expected = 1.0 + d / (1.0 + d * 0.5);
        let out = clarity2012_v3(rgb, 100.0, 1.0, 0.5);
        close3(out, [rgb[0] * expected, rgb[1] * expected, rgb[2] * expected], 1e-3);

        // amount > 0, d < 0: compressed by 1 - d*compress*16.
        let d = (y - 12100.0) * 1.0 / y;
        let expected = 1.0 + d / (1.0 - d * 0.5 * 16.0);
        let out = clarity2012_v3(rgb, 110.0, 1.0, 0.5);
        close3(out, [rgb[0] * expected, rgb[1] * expected, rgb[2] * expected], 1e-3);

        // amount <= 0: sqrt branch with an un-squared low.
        let sy = y.sqrt();
        let d = (sy - 90.0) * -0.5 / sy;
        let expected = (1.0 + d) * (1.0 + d);
        let out = clarity2012_v3(rgb, 90.0, -0.5, 0.5);
        close3(out, [rgb[0] * expected, rgb[1] * expected, rgb[2] * expected], 1e-3);
    }

    #[test]
    fn clarity2012_v4_negative_branch_uses_second_low() {
        let rgb = [10000.0, 10000.0, 10000.0];
        let sy = (0.299 * 10000.0 + 0.587 * 10000.0 + 0.114 * 10000.0 + 16.0f32).sqrt();
        let d = (sy - 90.0) * -0.5 / sy;
        let expected = (1.0 + d) * (1.0 + d);
        let out = clarity2012_v4(rgb, 200.0, 90.0, -0.5, 0.5);
        close3(out, [rgb[0] * expected, rgb[1] * expected, rgb[2] * expected], 1e-3);
        // The positive branch uses low_pos, not low_neg.
        let a = clarity2012_v4(rgb, 100.0, 90.0, 1.0, 0.5);
        let b = clarity2012_v4(rgb, 100.0, 300.0, 1.0, 0.5);
        close3(a, b, 1e-3);
    }

    #[test]
    fn clarity2012_v5_is_per_channel_additive() {
        let rgb = [10000.0, 20000.0, 30000.0];
        let low = [100.0, 140.0, 170.0];
        let out = clarity2012_v5(rgb, low, 1.0, 0.5);
        // d_c = (C + 16 - low_c^2); all positive here.
        close3(
            out,
            [
                10000.0 + (10016.0 - 100.0 * 100.0),
                20000.0 + (20016.0 - 140.0 * 140.0),
                30000.0 + (30016.0 - 170.0 * 170.0),
            ],
            1e-3,
        );
        // Negative delta is compressed and the output is clamped at 0.
        let out = clarity2012_v5([10000.0, 10000.0, 10000.0], [200.0, 200.0, 200.0], 1.0, 0.5);
        close3(out, [0.0, 0.0, 0.0], 1e-3);
    }

    #[test]
    fn clarity2012_v6_uses_mapped_amount_and_second_low() {
        let rgb = [10000.0, 10000.0, 10000.0];
        // raw -100 -> w = -0.3, so the negative low buffer (200) is used.
        let out = clarity2012_v6(rgb, [100.0; 3], [200.0; 3], -100.0, 0.5);
        let y = 10000.0f32;
        let d = (y - 40000.0) * -0.3;
        close3(out, [y + d - 16.0; 3], 1e-3);
        // If the positive buffer were used the result would be different.
        let other = clarity2012_v6(rgb, [100.0; 3], [300.0; 3], -100.0, 0.5);
        assert!((out[0] - other[0]).abs() > 1e-3);
    }

    #[test]
    fn clarity2012_v7_signed_delta_branches() {
        let rgb = [10000.0, 10000.0, 10000.0];
        let y = 0.299 * 10000.0 + 0.587 * 10000.0 + 0.114 * 10000.0 + 16.0;

        // d0 > 0: multiplicative, ratio preserved.
        let d0 = y - 100.0f32 * 100.0;
        let k = y * (2.5 / 65535.0);
        let d = d0 * 10000.0 / (10000.0 + (0.3 + k * k) * d0) * 1.0;
        let s = 1.0 + d / y;
        let out = clarity2012_v7(rgb, 100.0, 1.0);
        close3(out, [rgb[0] * s, rgb[1] * s, rgb[2] * s], 1e-3);

        // d0 < 0: additive, the same delta on all channels.
        let d0 = y - 110.0f32 * 110.0;
        let delta = d0 * y / (y + 6.0 * d0.abs()) * 1.0;
        let out = clarity2012_v7(rgb, 110.0, 1.0);
        close3(out, [rgb[0] + delta, rgb[1] + delta, rgb[2] + delta], 1e-3);
    }

    #[test]
    fn clarity2012_v8_doubles_negative_amount_and_uses_second_low() {
        let rgb = [10000.0, 10000.0, 10000.0];
        let y = 0.299 * 10000.0 + 0.587 * 10000.0 + 0.114 * 10000.0 + 16.0;
        // raw -100 -> doubled -200 -> w = -0.6; low_neg = 110 -> d0 < 0.
        let d0 = y - 110.0f32 * 110.0;
        let delta = d0 * y / (y + 6.0 * d0.abs()) * -0.6;
        let out = clarity2012_v8(rgb, 90.0, 110.0, -100.0);
        close3(out, [rgb[0] + delta, rgb[1] + delta, rgb[2] + delta], 1e-3);
        // Using low_pos instead would change the branch/result.
        let other = clarity2012_v8(rgb, 90.0, 300.0, -100.0);
        assert!((out[0] - other[0]).abs() > 1e-3);
    }

    #[test]
    fn clarity2012_v9_combines_residual_and_clamps() {
        let out = clarity2012_v9([10.0, 20.0, 30.0], [5.0; 3], [15.0; 3]);
        close3(out, [20.0, 30.0, 40.0], 1e-6);
        let out = clarity2012_v9([0.0; 3], [5.0; 3], [0.0; 3]);
        close3(out, [0.0; 3], 1e-6);
    }

    #[test]
    fn clarity2012_encode_variants_1_and_2() {
        let rgb = [65535u16, 0, 0];
        let enc = clarity2012_encode_rgb(rgb, 16.0);
        close(enc[0], (65535.0f32 + 16.0).sqrt(), 1e-4);
        close(enc[1], 4.0, 1e-6);
        close(enc[2], 4.0, 1e-6);
        let luma: f32 = 0.299 * 65535.0 + 16.0;
        close(clarity2012_encode_luma(rgb, 16.0), luma.sqrt(), 1e-4);
    }

    #[test]
    fn apply_rgb_clarity_is_additive_with_kernel_amount_map() {
        let rgb = [10000.0, 20000.0, 30000.0];
        let low = [9000.0, 19000.0, 29000.0];
        let out = apply_rgb_clarity(rgb, low, ClarityMethod::Neutral, 50.0, 0.0, 8);
        close3(out, [10250.0, 20250.0, 30250.0], 1e-3);
        let out = apply_rgb_clarity(rgb, low, ClarityMethod::Punch, 50.0, 0.0, 8);
        close3(out, [10300.0, 20300.0, 30300.0], 1e-3);
    }

    #[test]
    fn clarity_grid_norm_is_cell_to_the_sixth() {
        close(clarity_grid_norm(8), 1.0 / 262144.0, 1e-12);
        close(clarity_grid_norm(8), 1.0 / (64.0 * 4096.0), 1e-12);
        close(clarity_grid_norm(4), 1.0 / 4096.0, 1e-12);
    }

    #[test]
    fn grid_sample_hits_block_means_at_block_centres() {
        let src = [
            0.0f32, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0,
            1.0,
        ];
        let grid = block_mean_f32(&src, 4, 4, 2);
        assert_eq!(grid, vec![0.0, 1.0, 0.0, 1.0]);
        let v = sample_grid(&grid, 2, 2, 0, 0, 2);
        assert!((v - 0.0).abs() < 1e-6);
        // Mid-block interpolation reaches the half-way point.
        assert!((sample_grid(&grid, 2, 2, 1, 0, 2) - 0.5).abs() < 1e-6);
    }
}
