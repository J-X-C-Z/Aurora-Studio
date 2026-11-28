//! Capture One's `CImgOpClarity2012` / `CImgOpStructure2012`, reconstructed
//! from `ImgCoreDll.dll` and verified against 16-bit exports.
//!
//! Clarity is a low-frequency subtraction on luma — not a global S-curve:
//!
//! ```text
//! y' = y + w_big * (y - low_big) + w_fine * (y - low_fine)
//! ```
//!
//! `low_big` is a block grid of `0.125 * min(w, h)` cells (C1's
//! `c1_clarity_radius_big`) upsampled bilinearly, `low_fine` uses
//! `100 * w / 11976` (`c1_clarity_radius_small`).  Measured gain at +100 is
//! 1.81 from 3 px to 128 px periods and 1.42 at 512 px.
//!
//! Structure is multiplicative and fixed in pixels:
//! `s = max(0, 1 + (y - low) * w / low)^2`, `rgb' = rgb * s` on a 5x5 box.
//!
//! Both are applied to luma and then carried to R/G/B (clarity adds the same
//! delta to all three, structure multiplies all three by the same gain), which
//! is what the Y/Cb/Cr round trip in the native kernel amounts to.
//!
//! This CPU kernel depends only on [`Clarity`]/[`ClarityMethod`], so it lives
//! in `processing` rather than `mock`; `crate::mock` re-exports it so existing
//! call sites and the GPU module keep compiling unchanged.

use rayon::prelude::*;

use crate::{Clarity, ClarityMethod};

/// Native clarity barely darkens locally-dark detail: the measured
/// native/model ratio is 0.32-0.38 for negative local contrast but 1.0 for
/// positive. 0.6 is the fitted compromise over the chart and a real NEF.
const NEGATIVE_GAIN: f32 = 0.6;

/// Structure box radius in pixels. The native operator's response is
/// independent of the frame size, so this stays absolute.
const STRUCTURE_RADIUS: usize = 2;

/// Clarity weights for one amount. `Classic` and negative amounts only use the
/// fine grid; positive amounts of the other methods add the coarse grid.
#[derive(Clone, Copy)]
struct Kernel {
    k_big: f32,
    k_fine: f32,
    use_big: bool,
    use_fine: bool,
}

fn clarity_kernel(amount: f32, method: ClarityMethod) -> Kernel {
    if amount < 0.0 {
        return Kernel { k_big: 0.0, k_fine: 0.0031 * amount, use_big: false, use_fine: true };
    }
    // Everything that is not one of the three two-scale methods falls back to
    // Classic (fine scale only), including any method added later.
    let classic = Kernel { k_big: 0.0, k_fine: 0.00433 * amount, use_big: false, use_fine: true };
    if !matches!(
        method,
        ClarityMethod::Neutral | ClarityMethod::Punch | ClarityMethod::Natural
    ) {
        return classic;
    }
    // Neutral is the reference weight, Punch ~2% stronger and Contrast
    // (labelled Natural) ~4% weaker: measured on the native exports at +100.
    let scale = match method {
        ClarityMethod::Punch => 1.02,
        ClarityMethod::Natural => 0.96,
        _ => 1.0,
    };
    Kernel {
        k_big: 0.0092 * amount * scale,
        k_fine: 0.00082 * amount * scale,
        use_big: true,
        use_fine: true,
    }
}

/// Coarse cell: C1's radius is a fraction of the short side.
pub(crate) fn coarse_cell(w: usize, h: usize) -> f32 {
    (0.125 * w.min(h) as f32).max(8.0)
}

pub(crate) fn fine_cell(w: usize) -> f32 {
    (100.0 * w as f32 / 11976.0).max(1.0)
}

fn luma_plane(rgba: &[u8], wide: Option<&[u16]>, w: usize, h: usize) -> Vec<f32> {
    let row = w * 4;
    match wide {
        Some(wide) if wide.len() >= w * h * 4 => {
            let mut out = vec![0.0f32; w * h];
            out.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
                let src = &wide[y * row..y * row + row];
                for (x, v) in line.iter_mut().enumerate() {
                    let p = x * 4;
                    let l = 0.299 * src[p] as f32 + 0.587 * src[p + 1] as f32 + 0.114 * src[p + 2] as f32;
                    *v = l * (1.0 / 65535.0);
                }
            });
            out
        }
        _ => {
            let mut out = vec![0.0f32; w * h];
            out.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
                let src = &rgba[y * row..y * row + row];
                for (x, v) in line.iter_mut().enumerate() {
                    let p = x * 4;
                    let l = 0.299 * src[p] as f32 + 0.587 * src[p + 1] as f32 + 0.114 * src[p + 2] as f32;
                    *v = l * (1.0 / 255.0);
                }
            });
            out
        }
    }
}

/// Separable box blur, clamped at the frame edge.
fn box_blur(y: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 {
        return y.to_vec();
    }
    let inv = 1.0 / (2 * r + 1) as f32;
    let mut tmp = vec![0.0f32; w * h];
    tmp.par_chunks_mut(w).enumerate().for_each(|(yy, line)| {
        let row = &y[yy * w..(yy + 1) * w];
        let mut acc = 0.0f32;
        for k in -(r as isize)..=(r as isize) {
            acc += row[(k.max(0) as usize).min(w - 1)];
        }
        for xx in 0..w {
            if xx > 0 {
                acc += row[(xx + r).min(w - 1)];
                acc -= row[(xx as isize - r as isize - 1).max(0) as usize];
            }
            line[xx] = acc * inv;
        }
    });
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(yy, line)| {
        // Rows outside the frame repeat the edge row, matching the horizontal
        // pass. Accumulating whole rows in `k` order keeps, for every `x`, the
        // same additions in the same order as the per-pixel reference while
        // turning the inner loop into a contiguous branch-free f32 add.
        line.fill(0.0);
        for k in -(r as isize)..=(r as isize) {
            let row = (yy as isize + k).clamp(0, h as isize - 1) as usize;
            let src_row = &tmp[row * w..(row + 1) * w];
            for (sum, value) in line.iter_mut().zip(src_row) {
                *sum += *value;
            }
        }
        for value in line.iter_mut() {
            *value *= inv;
        }
    });
    out
}

/// Clarity's low-frequency planes, sampled before the tone curves.
///
/// The native kernel subtracts a low-pass of the raw-stage luma. That low-pass
/// is slow (50-500 px cells), so it does not need the full frame: the caller
/// samples the pre-curve luma on a coarse grid (`step` output pixels per cell)
/// and this struct keeps only the two blurred planes. 1/8 sampling on a
pub(crate) struct Grid {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) gw: usize,
    pub(crate) gh: usize,
    pub(crate) step: usize,
    pub(crate) raw: Vec<f32>,
}

impl Grid {
    pub(crate) fn sample(
        w: usize,
        h: usize,
        step: usize,
        mut luma_at: impl FnMut(usize, usize) -> Option<f32>,
    ) -> Option<Self> {
        if w == 0 || h == 0 {
            return None;
        }
        let step = step.max(1);
        let gw = w.div_ceil(step);
        let gh = h.div_ceil(step);
        let mut raw = vec![0.11f32; gw * gh];
        for gy in 0..gh {
            for gx in 0..gw {
                if let Some(v) = luma_at(gx, gy) {
                    raw[gy * gw + gx] = v;
                }
            }
        }
        Some(Self { w, h, gw, gh, step, raw })
    }

    pub(crate) fn planes(&self, amount: f32, method: ClarityMethod) -> Option<PreCurve> {
        if amount.abs() < 1e-4 {
            return None;
        }
        let (k_big, k_fine, chroma) = PreCurve::params(amount, method);
        let fine = plane_low(&self.raw, self.gw, self.gh, cols_for(self.gw, self.step, fine_cell(self.w)));
        let coarse = plane_low(&self.raw, self.gw, self.gh, cols_for(self.gw, self.step, coarse_cell(self.w, self.h)));
        Some(PreCurve {
            gw: self.gw,
            gh: self.gh,
            step: self.step as f32,
            fine,
            coarse,
            k_big,
            k_fine,
            chroma,
        })
    }
}

/// 6048x4032 frame costs 2 x 756 x 504 x 4 B = 3.0 MB.
pub(crate) struct PreCurve {
    gw: usize,
    gh: usize,
    step: f32,
    fine: Vec<f32>,
    coarse: Vec<f32>,
    k_big: f32,
    k_fine: f32,
    chroma: f32,
}

impl PreCurve {
    pub(crate) fn params(amount: f32, method: ClarityMethod) -> (f32, f32, f32) {
        let kernel = clarity_kernel(amount, method);
        let chroma = if method == ClarityMethod::Punch { 1.0 + 0.0025 * amount } else { 1.0 };
        (kernel.k_big, kernel.k_fine, chroma)
    }

    /// `luma_at(gx, gy)` returns the pre-curve luma for grid cell `(gx, gy)`,
    /// or `None` where the frame has no sample.
    pub(crate) fn build(
        w: usize,
        h: usize,
        step: usize,
        amount: f32,
        method: ClarityMethod,
        luma_at: impl FnMut(usize, usize) -> Option<f32>,
    ) -> Option<Self> {
        if amount.abs() < 1e-4 {
            return None;
        }
        Grid::sample(w, h, step, luma_at)?.planes(amount, method)
    }

    /// Bytes held by the two planes (for the memory assertion).
    pub(crate) fn bytes(&self) -> usize {
        (self.fine.len() + self.coarse.len()) * std::mem::size_of::<f32>()
    }

    /// Bilinear sample of a grid plane at output pixel `(x, y)`.
    fn sample(plane: &[f32], gw: usize, gh: usize, step: f32, x: usize, y: usize) -> f32 {
        let fx = ((x as f32 + 0.5) / step - 0.5).clamp(0.0, gw as f32 - 1.0);
        let fy = ((y as f32 + 0.5) / step - 0.5).clamp(0.0, gh as f32 - 1.0);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(gw - 1);
        let y1 = (y0 + 1).min(gh - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let a = plane[y0 * gw + x0] * (1.0 - tx) + plane[y0 * gw + x1] * tx;
        let b = plane[y1 * gw + x0] * (1.0 - tx) + plane[y1 * gw + x1] * tx;
        a * (1.0 - ty) + b * ty
    }

    /// Applies the clarity delta to a pre-curve pixel.
    pub(crate) fn apply(&self, g: [f32; 3], x: usize, y: usize) -> [f32; 3] {
        let l = 0.299 * g[0] + 0.587 * g[1] + 0.114 * g[2];
        let big = Self::sample(&self.coarse, self.gw, self.gh, self.step, x, y);
        let fine = Self::sample(&self.fine, self.gw, self.gh, self.step, x, y);
        let mut d = self.k_big * (l - big) + self.k_fine * (l - fine);
        if d < 0.0 {
            d *= NEGATIVE_GAIN;
        }
        shade(g, d, 1.0, self.chroma)
    }
}

/// Plane columns that cover one `cell` of the full frame.
pub(crate) fn cols_for(gw: usize, step: usize, cell: f32) -> usize {
    ((gw as f32) * step as f32 / cell.max(1.0)).round().clamp(1.0, gw as f32) as usize
}

/// Block average into `cols` columns, bilinear upsample back to the plane.
/// Mirrors the full-resolution kernel, at plane resolution.
fn plane_low(src: &[f32], gw: usize, gh: usize, cols: usize) -> Vec<f32> {
    let cols = cols.clamp(1, gw);
    let rows = ((cols as f32 * gh as f32 / gw as f32).round() as usize).clamp(1, gh);
    let mut small = vec![0.0f32; rows * cols];
    for j in 0..rows {
        let y0 = j * gh / rows;
        let y1 = ((j + 1) * gh / rows).max(y0 + 1);
        for i in 0..cols {
            let x0 = i * gw / cols;
            let x1 = ((i + 1) * gw / cols).max(x0 + 1);
            let mut sum = 0.0f64;
            let mut n = 0u32;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    sum += src[yy * gw + xx] as f64;
                    n += 1;
                }
            }
            small[j * cols + i] = (sum / n as f64) as f32;
        }
    }
    let mut out = vec![0.0f32; gw * gh];
    for yy in 0..gh {
        let fy = ((yy as f32 + 0.5) * rows as f32 / gh as f32 - 0.5).clamp(0.0, rows as f32 - 1.0);
        let j0 = fy.floor() as usize;
        let j1 = (j0 + 1).min(rows - 1);
        let ty = fy - j0 as f32;
        for xx in 0..gw {
            let fx = ((xx as f32 + 0.5) * cols as f32 / gw as f32 - 0.5).clamp(0.0, cols as f32 - 1.0);
            let i0 = fx.floor() as usize;
            let i1 = (i0 + 1).min(cols - 1);
            let tx = fx - i0 as f32;
            let a = small[j0 * cols + i0] * (1.0 - tx) + small[j0 * cols + i1] * tx;
            let b = small[j1 * cols + i0] * (1.0 - tx) + small[j1 * cols + i1] * tx;
            out[yy * gw + xx] = a * (1.0 - ty) + b * ty;
        }
    }
    out
}

/// Structure: the fixed-radius multiplicative local contrast. It stays on the
/// developed frame — its 5x5 neighbourhood cannot come from the coarse planes.
pub(crate) fn apply_structure(
    rgba: &mut [u8],
    wide: Option<&mut [u16]>,
    w: usize,
    h: usize,
    clarity: &Clarity,
) {
    let amount = clarity.structure;
    if amount.abs() < 1e-4 || w == 0 || h == 0 || rgba.len() < w * h * 4 {
        return;
    }
    let y = luma_plane(rgba, wide.as_deref(), w, h);
    let low = box_blur(&y, w, h, STRUCTURE_RADIUS);
    shade_frame(rgba, wide, w, &y, Some(low.as_slice()), amount, 1.0);
}

/// `(v + delta) * gain` for the structure pass.
#[allow(clippy::too_many_arguments)]
fn shade_frame(
    rgba: &mut [u8],
    wide: Option<&mut [u16]>,
    w: usize,
    y: &[f32],
    low: Option<&[f32]>,
    structure: f32,
    chroma: f32,
) {
    let row = w * 4;
    let k_struct = 0.0028 * structure;
    let gain = |i: usize| -> f32 {
        match low {
            Some(l) => {
                let t = 1.0 + (y[i] - l[i]) * (k_struct / l[i].max(1e-3));
                let t = t.max(0.0);
                t * t
            }
            None => 1.0,
        }
    };

    match wide {
        Some(wide) if wide.len() >= rgba.len() * 2 => {
            let wide_row = w * 4;
            rgba.par_chunks_mut(row)
                .zip(wide.par_chunks_mut(wide_row))
                .enumerate()
                .for_each(|(yy, (line8, line16))| {
                    let base = yy * w;
                    for x in 0..w {
                        let s = gain(base + x);
                        let p = x * 4;
                        let q = shade(
                            [
                                line16[p] as f32 * (1.0 / 65535.0),
                                line16[p + 1] as f32 * (1.0 / 65535.0),
                                line16[p + 2] as f32 * (1.0 / 65535.0),
                            ],
                            0.0,
                            s,
                            chroma,
                        );
                        for k in 0..3 {
                            let v = q[k].clamp(0.0, 1.0);
                            line16[p + k] = (v * 65535.0 + 0.5) as u16;
                            line8[p + k] = (v * 255.0 + 0.5) as u8;
                        }
                    }
                });
        }
        _ => {
            rgba.par_chunks_mut(row).enumerate().for_each(|(yy, line)| {
                let base = yy * w;
                for x in 0..w {
                    let s = gain(base + x);
                    let p = x * 4;
                    let q = shade(
                        [
                            line[p] as f32 * (1.0 / 255.0),
                            line[p + 1] as f32 * (1.0 / 255.0),
                            line[p + 2] as f32 * (1.0 / 255.0),
                        ],
                        0.0,
                        s,
                        chroma,
                    );
                    for k in 0..3 {
                        line[p + k] = (q[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                    }
                }
            });
        }
    }
}

/// `(v + delta) * gain`, then C1's Punch chroma push around the result's luma.
#[inline]
fn shade(p: [f32; 3], d: f32, s: f32, chroma: f32) -> [f32; 3] {
    let mut q = [(p[0] + d) * s, (p[1] + d) * s, (p[2] + d) * s];
    if chroma != 1.0 {
        let l = 0.299 * q[0] + 0.587 * q[1] + 0.114 * q[2];
        q = [
            l + (q[0] - l) * chroma,
            l + (q[1] - l) * chroma,
            l + (q[2] - l) * chroma,
        ];
    }
    q
}



/// Block-average + bilinear low-pass, the full-resolution kernel the
/// previous patch used. Test-only reference for the pre-curve rewrite.
#[allow(dead_code)]
fn grid_low_ref(y: &[f32], w: usize, h: usize, cols: usize) -> Vec<f32> {
    let cols = cols.clamp(1, w);
    let rows = ((cols as f32 * h as f32 / w as f32).round() as usize).clamp(1, h);
    let mut small = vec![0.0f32; rows * cols];
    for j in 0..rows {
        let y0 = j * h / rows;
        let y1 = ((j + 1) * h / rows).max(y0 + 1);
        for i in 0..cols {
            let x0 = i * w / cols;
            let x1 = ((i + 1) * w / cols).max(x0 + 1);
            let mut sum = 0.0f64;
            let mut n = 0u32;
            for yy in y0..y1 {
                for xx in x0..x1 {
                    sum += y[yy * w + xx] as f64;
                    n += 1;
                }
            }
            small[j * cols + i] = (sum / n as f64) as f32;
        }
    }
    let mut out = vec![0.0f32; w * h];
    for yy in 0..h {
        let fy = ((yy as f32 + 0.5) * rows as f32 / h as f32 - 0.5).clamp(0.0, rows as f32 - 1.0);
        let j0 = fy.floor() as usize;
        let j1 = (j0 + 1).min(rows - 1);
        let ty = fy - j0 as f32;
        for xx in 0..w {
            let fx = ((xx as f32 + 0.5) * cols as f32 / w as f32 - 0.5).clamp(0.0, cols as f32 - 1.0);
            let i0 = fx.floor() as usize;
            let i1 = (i0 + 1).min(cols - 1);
            let tx = fx - i0 as f32;
            let a = small[j0 * cols + i0] * (1.0 - tx) + small[j0 * cols + i1] * tx;
            let b = small[j1 * cols + i0] * (1.0 - tx) + small[j1 * cols + i1] * tx;
            out[yy * w + xx] = a * (1.0 - ty) + b * ty;
        }
    }
    out
}

/// The full-resolution clarity pass from `CLARITY-NATIVE.patch`: delta
/// computed on the finished frame.
#[cfg(test)]
pub(crate) fn legacy_apply(rgba: &mut [u8], w: usize, h: usize, clarity: &Clarity) {
    let amount = clarity.clarity;
    if amount.abs() < 1e-4 {
        return;
    }
    let y = luma_plane(rgba, None, w, h);
    let kernel = clarity_kernel(amount, clarity.method);
    let big = grid_low_ref(&y, w, h, (w as f32 / coarse_cell(w, h)).round().max(1.0) as usize);
    let fine = grid_low_ref(&y, w, h, (w as f32 / fine_cell(w)).round().max(1.0) as usize);
    let mut d = vec![0.0f32; w * h];
    for i in 0..w * h {
        let mut v = 0.0;
        if kernel.use_big {
            v += kernel.k_big * (y[i] - big[i]);
        }
        if kernel.use_fine {
            v += kernel.k_fine * (y[i] - fine[i]);
        }
        d[i] = if v < 0.0 { v * NEGATIVE_GAIN } else { v };
    }
    let chroma = if clarity.method == ClarityMethod::Punch { 1.0 + 0.0025 * amount } else { 1.0 };
    for (i, px) in rgba.chunks_exact_mut(4).enumerate() {
        let p = [px[0] as f32 / 255.0, px[1] as f32 / 255.0, px[2] as f32 / 255.0];
        let q = shade(p, d[i], 1.0, chroma);
        for k in 0..3 {
            px[k] = (q[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frozen copy of the pre-vectorisation `box_blur` used as the bit-exact
    /// oracle. Do not "improve" this: it must keep the original per-pixel
    /// accumulation order.
    fn box_blur_scalar_reference(y: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
        if r == 0 {
            return y.to_vec();
        }
        let inv = 1.0 / (2 * r + 1) as f32;
        let mut tmp = vec![0.0f32; w * h];
        for yy in 0..h {
            let row = &y[yy * w..(yy + 1) * w];
            let mut acc = 0.0f32;
            for k in -(r as isize)..=(r as isize) {
                acc += row[(k.max(0) as usize).min(w - 1)];
            }
            for xx in 0..w {
                if xx > 0 {
                    acc += row[(xx + r).min(w - 1)];
                    acc -= row[(xx as isize - r as isize - 1).max(0) as usize];
                }
                tmp[yy * w + xx] = acc * inv;
            }
        }
        let mut out = vec![0.0f32; w * h];
        for yy in 0..h {
            for xx in 0..w {
                let mut acc = 0.0f32;
                for k in -(r as isize)..=(r as isize) {
                    let row = (yy as isize + k).clamp(0, h as isize - 1) as usize;
                    acc += tmp[row * w + xx];
                }
                out[yy * w + xx] = acc * inv;
            }
        }
        out
    }

    /// Deterministic LCG so the cross-check is reproducible without external
    /// fixtures.
    fn lcg_plane(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 40) as f32) / ((1u64 << 24) as f32) * 4.0 - 2.0
            })
            .collect()
    }

    #[test]
    fn box_blur_matches_scalar_reference_bit_exactly() {
        // 1x1, single-row/column edges, non-8-aligned widths and heights.
        for (w, h) in [(1usize, 1usize), (1, 17), (19, 1), (3, 64), (64, 3), (37, 23), (33, 7)] {
            let y = lcg_plane(w * h, 0x9e37_79b9_7f4a_7c15 ^ ((w * 131 + h) as u64));
            for r in [0usize, 1, 2, 5, 23] {
                let expected = box_blur_scalar_reference(&y, w, h, r);
                let actual = box_blur(&y, w, h, r);
                for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert_eq!(a.to_bits(), e.to_bits(), "w={w} h={h} r={r} i={i}");
                }
            }
        }
    }

    // Test-only: `crate::Source` and `crate::render` are the crate-root
    // re-exports of the legacy renderer in `mock`. The production clarity
    // stage above never touches the renderer; only this ignored benchmark
    // builds a `Source` and calls `render`.
    /// Full-frame CPU cost of the two halves of the stage. Run with
    /// `cargo test --release -p aurora-engine --lib --features z6iii-raw-export
    ///  clarity_stage_bench -- --ignored --nocapture`.
    #[test]
    #[ignore = "benchmark: prints timings for a 6048x4032 frame"]
    fn clarity_stage_bench_6048() {
        let (w, h) = (6048usize, 4032usize);
        let mk = |clarity: f32, structure: f32| Clarity {
            method: ClarityMethod::Neutral,
            clarity,
            structure,
        };

        // Old vs new clarity on the same frame, same run, best of 3: this is
        // the ratio the patch is judged on.
        let mut rgba = vec![0u8; w * h * 4];
        for (y, line) in rgba.chunks_mut(w * 4).enumerate() {
            for (x, px) in line.chunks_mut(4).enumerate() {
                let v = ((x * 7 + y * 13) % 251) as u8;
                px[0] = v;
                px[1] = v.wrapping_add(37);
                px[2] = v.wrapping_add(91);
                px[3] = 255;
            }
        }
        let clarity = mk(100.0, 0.0);
        let mut legacy_ms = f64::MAX;
        let mut new_ms = f64::MAX;
        let mut build_ms = f64::MAX;
        let mut bytes = 0usize;
        for _ in 0..3 {
            let mut buf = rgba.clone();
            let t = std::time::Instant::now();
            legacy_apply(&mut buf, w, h, &clarity);
            legacy_ms = legacy_ms.min(t.elapsed().as_secs_f64() * 1000.0);

            let t = std::time::Instant::now();
            let pre = PreCurve::build(w, h, 8, 100.0, ClarityMethod::Neutral, |gx, gy| {
                Some(((gx * 37 + gy * 91) % 251) as f32 / 255.0)
            })
            .expect("planes");
            build_ms = build_ms.min(t.elapsed().as_secs_f64() * 1000.0);
            bytes = pre.bytes();

            let t = std::time::Instant::now();
            buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, line)| {
                for x in 0..w {
                    let g = pre.apply([0.5, 0.45, 0.4], x, y);
                    let p = x * 4;
                    line[p] = (g[0].clamp(0.0, 1.0) * 255.0) as u8;
                    line[p + 1] = (g[1].clamp(0.0, 1.0) * 255.0) as u8;
                    line[p + 2] = (g[2].clamp(0.0, 1.0) * 255.0) as u8;
                    line[p + 3] = 255;
                }
            });
            new_ms = new_ms.min(t.elapsed().as_secs_f64() * 1000.0);
        }
        eprintln!("legacy full-frame clarity {legacy_ms:8.1} ms");
        eprintln!("new: planes {build_ms:8.1} ms + apply {new_ms:8.1} ms = {:.1} ms  ({:.2} MB)",
            build_ms + new_ms, bytes as f64 / 1e6);
        eprintln!("ratio {:.2}x", (build_ms + new_ms) / legacy_ms);
        let mut out = rgba;

        // Structure stays on the developed frame.
        let t = std::time::Instant::now();
        apply_structure(&mut out, None, w, h, &mk(0.0, 100.0));
        eprintln!("structure +100       {:8.1} ms", t.elapsed().as_secs_f64() * 1000.0);
        let t = std::time::Instant::now();
        apply_structure(&mut out, None, w, h, &mk(0.0, 0.0));
        eprintln!("structure 0 (no-op)  {:8.1} ms", t.elapsed().as_secs_f64() * 1000.0);

        // End-to-end: the whole render with and without clarity, so the plane
        // build pays for its real source sampling. 1/4 of the pixels; the
        // stage is linear in them, the fixed costs are not, hence the note.
        let (sw, sh) = (6048u32, 4032u32);
        let mut rgba = vec![0u8; (sw * sh * 4) as usize];
        for (i, px) in rgba.chunks_mut(4).enumerate() {
            let v = ((i * 7) % 251) as u8;
            px[0] = v;
            px[1] = v.wrapping_add(37);
            px[2] = v.wrapping_add(91);
            px[3] = 255;
        }
        let src = crate::Source::from_rgba(sw, sh, &rgba);
        let render_ms = |clarity: f32| -> (f64, Vec<u8>) {
            let adj = crate::AdjustmentSet {
                clarity: mk(clarity, 0.0),
                ..Default::default()
            };
            let t = std::time::Instant::now();
            let out = crate::render(
                &src,
                &adj,
                crate::NormRect::FULL,
                sw,
                false,
                None,
            );
            (t.elapsed().as_secs_f64() * 1000.0, out.rgba)
        };
        render_ms(0.0);
        let (off, base_px) = render_ms(0.0);
        let (on, cl_px) = render_ms(100.0);
        let changed = base_px.iter().zip(&cl_px).filter(|(a, b)| a != b).count();
        eprintln!(
            "end-to-end {sw}x{sh}: clarity 0 {off:8.1} ms, +100 {on:8.1} ms, delta {:8.1} ms, {changed} bytes changed",
            on - off,
        );
    }
}
