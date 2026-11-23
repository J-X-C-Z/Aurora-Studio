//! Approximate sharpening, noise reduction, moire, lens correction and film grain.
//! These are display-referred 8-bit sRGB approximations for the preview pipeline, not the production renderer.
//! Known limits: no 16-bit path, no real lens profiles, moire is a fixed 3x3 low-pass, grain is hash noise.

use rayon::prelude::*;

#[cfg(windows)]
use super::detail_gpu;
#[cfg(not(windows))]
mod detail_gpu {
    use crate::{LensCorrection, Moire, NoiseReduction, Sharpening};

    pub(super) fn enabled() -> bool {
        false
    }

    pub(super) fn post_process(
        _rgba: &mut [u8],
        _w: usize,
        _h: usize,
        _nr: &NoiseReduction,
        _moire: &Moire,
        _sh: &Sharpening,
        _lens: &LensCorrection,
        _scale: f32,
    ) -> bool {
        false
    }
}

use crate::{FilmGrain, GrainKind, LensCorrection, Moire, NoiseReduction, Sharpening};

const PURPLE_BAND_ROWS: usize = 128;
const DIFFRACTION_BAND_ROWS: usize = 128;

/// Full-precision float32 display-sRGB (`0.0..=1.0`) core detail pipeline:
/// runs single-pixel spike removal, center-weighted edge-preserving luminance NR,
/// C1 `ColorNoise2012` linear-domain chroma range filter, and (when `include_sharpen`
/// is true) true separable Gaussian linear-light USM with zero 8-bit quantization
/// round-trips.
pub(crate) fn post_process_core_f32(
    rgb: &mut [[f32; 3]],
    width: u32,
    height: u32,
    adj: &crate::AdjustmentSet,
    scale: f32,
    include_sharpen: bool,
) {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 || rgb.len() != w * h {
        return;
    }
    let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
    let nr = &adj.noise_reduction;
    if nr.single_pixel > 0.0 {
        stage_single_pixel_f32(rgb, w, h, nr);
    }
    if nr.luminance > 0.0 {
        stage_luminance_nr_f32(rgb, w, h, nr);
    }
    if nr.color > 0.0 {
        stage_color_nr_f32(rgb, w, h, nr, scale);
    }
    if include_sharpen && adj.sharpening.amount > 0.0 {
        stage_sharpening_f32(rgb, w, h, &adj.sharpening, scale);
    }
}

/// Support of successive neighborhood stages in output pixels.
pub(crate) fn core_halo(adj: &crate::AdjustmentSet, scale: f32, sharpen: bool) -> usize {
    let nr = &adj.noise_reduction;
    let mut radius = usize::from(nr.single_pixel > 0.0) + usize::from(nr.luminance > 0.0);
    if nr.color > 0.0 {
        radius += color_filter_support(nr.color, scale).0 as usize;
    }
    if sharpen && adj.sharpening.amount > 0.0 {
        let sigma = (adj.sharpening.radius.max(0.1) * scale).clamp(0.35, 12.0);
        radius += gaussian_support(sigma);
    }
    radius
}

fn color_filter_support(amount: f32, scale: f32) -> (isize, isize) {
    let strength = (amount / 100.0).clamp(0.0, 1.0);
    let step = (4.0 * scale).round().clamp(1.0, 8.0) as isize;
    let half_taps = (7.0 * (0.4 + 1.2 * strength)).round().clamp(2.0, 11.0) as isize;
    (half_taps * step, step)
}

fn gaussian_support(sigma: f32) -> usize {
    (sigma * 3.0).ceil().clamp(1.0, 24.0) as usize
}

#[inline]
fn srgb_to_linear_f32(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

fn padded_luma_f32(rgb: &[[f32; 3]], w: usize, h: usize) -> Vec<f32> {
    let pw = w + 2;
    let mut pad = vec![0.0f32; pw * (h + 2)];
    for y in 0..h {
        let src = &rgb[y * w..(y + 1) * w];
        let dst = &mut pad[(y + 1) * pw..(y + 2) * pw];
        for x in 0..w {
            let p = src[x];
            dst[x + 1] = (0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2]) * 255.0;
        }
        dst[0] = dst[1];
        dst[w + 1] = dst[w];
    }
    let (top, rest) = pad.split_at_mut(pw);
    top.copy_from_slice(&rest[..pw]);
    let last_src = h * pw;
    let (head, bottom) = pad.split_at_mut((h + 1) * pw);
    bottom.copy_from_slice(&head[last_src..last_src + pw]);
    pad
}

fn stage_single_pixel_f32(rgb: &mut [[f32; 3]], w: usize, h: usize, adj: &NoiseReduction) {
    let amount = (adj.single_pixel / 100.0).clamp(0.0, 1.0);
    if amount <= 0.0 {
        return;
    }
    let pad = padded_luma_f32(rgb, w, h);
    let threshold = 24.0 - 20.0 * amount;
    let pw = w + 2;
    rgb.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let c = pad[(y + 1) * pw + x + 1];
            let mut nb = [0f32; 8];
            for (k, (dx, dy)) in NEIGHBOURS.iter().enumerate() {
                nb[k] = pad[(y as i32 + 1 + dy) as usize * pw + (x as i32 + 1 + dx) as usize];
            }
            nb.sort_by(|a, b| a.total_cmp(b));
            let median = (nb[3] + nb[4]) * 0.5;
            if c > median + threshold {
                let f = if c > 1.0 { median / c } else { 1.0 };
                let p = &mut row[x];
                p[0] = (p[0] * f).clamp(0.0, 1.0);
                p[1] = (p[1] * f).clamp(0.0, 1.0);
                p[2] = (p[2] * f).clamp(0.0, 1.0);
            }
        }
    });
}

fn stage_luminance_nr_f32(rgb: &mut [[f32; 3]], w: usize, h: usize, adj: &NoiseReduction) {
    let strength = (adj.luminance / 100.0).clamp(0.0, 1.0);
    if strength <= 0.0 {
        return;
    }
    let pad = padded_luma_f32(rgb, w, h);
    let details = (adj.details / 100.0).clamp(0.0, 1.0);
    let sigma = ((2.5 + 26.0 * strength) / (1.0 + 3.5 * details)).max(0.5);
    let two_sigma2 = 2.0 * sigma * sigma;
    const LUT_N: usize = 1025;
    let mut range_lut = [0f32; LUT_N];
    for (i, v) in range_lut.iter_mut().enumerate() {
        let d = i as f32 * 0.25;
        *v = (-(d * d) / two_sigma2).exp();
    }
    let pw = w + 2;
    const SPATIAL: [f32; 8] =
        [1.0 / 3.0, 0.5, 1.0 / 3.0, 0.5, 0.5, 1.0 / 3.0, 0.5, 1.0 / 3.0];
    rgb.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
        for x in 0..w {
            let c = pad[(y + 1) * pw + x + 1];
            let mut num = c;
            let mut den = 1.0f32;
            for (k, (dx, dy)) in NEIGHBOURS.iter().enumerate() {
                let l = pad[(y as i32 + 1 + dy) as usize * pw + (x as i32 + 1 + dx) as usize];
                let ad = (l - c).abs() * 4.0;
                let i0 = (ad as usize).min(LUT_N - 2);
                let t = ad - i0 as f32;
                let range = range_lut[i0] + (range_lut[i0 + 1] - range_lut[i0]) * t;
                let wt = range * SPATIAL[k];
                num += wt * l;
                den += wt;
            }
            let mut target = if den > 1e-6 { num / den } else { c };
            target = c + (target - c) * strength;
            let new_l = target.clamp(0.0, 255.0);
            if (new_l - c).abs() >= 1e-3 {
                let f = if c > 1.0 { new_l / c } else { 1.0 };
                let p = &mut out_row[x];
                p[0] = (p[0] * f).clamp(0.0, 1.0);
                p[1] = (p[1] * f).clamp(0.0, 1.0);
                p[2] = (p[2] * f).clamp(0.0, 1.0);
            }
        }
    });
}

fn stage_color_nr_f32(rgb: &mut [[f32; 3]], w: usize, h: usize, adj: &NoiseReduction, scale: f32) {
    let strength = (adj.color / 100.0).clamp(0.0, 1.0);
    if strength <= 0.0 || w == 0 || h == 0 {
        return;
    }
    const K: [f32; 4] = [0.299, 0.587, 0.114, 35.759_254];
    const PX: f32 = 0.178_877_62;
    const PY: f32 = 0.377_672_3;
    const PZ_BASE: f32 = 0.401_832_58;
    const PW_BASE: f32 = 2.968_700_4;

    let scale_p = 0.35 + 1.3 * strength;
    let pz = PZ_BASE * scale_p;
    let pw = PW_BASE * scale_p;
    let pz_chroma = 2_400.0 + 18_000.0 * strength * strength;
    let pw_chroma = 12.0 + 36.0 * strength;

    let (radius, step) = color_filter_support(adj.color, scale);

    let n = w * h;
    let mut encoded = vec![[0.0f32; 3]; n];
    encoded.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let src = &rgb[y * w..(y + 1) * w];
        for x in 0..w {
            let r = srgb_to_linear_f32(src[x][0]) * 65535.0;
            let g = srgb_to_linear_f32(src[x][1]) * 65535.0;
            let b = srgb_to_linear_f32(src[x][2]) * 65535.0;
            let l = (K[0] * r + K[1] * g + K[2] * b).max(0.0) + K[3];
            let inv_l = 1.0 / l;
            row[x] = [l, (r - g) * inv_l, (b - g) * inv_l];
        }
    });

    let mut horiz = vec![[0.0f32; 3]; n];
    horiz.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
        let src_row = &encoded[y * w..(y + 1) * w];
        let max_x = (w - 1) as isize;
        for x in 0..w {
            let [lc, uc, vc] = src_row[x];
            let t_root = (pz + pw * lc).max(1.0);
            let t_l2 = t_root * t_root;
            let t_c = (pz_chroma + pw_chroma * lc).max(1.0);
            let u_lc = uc * lc;
            let v_lc = vc * lc;
            let mut sum_u = 0.0f32;
            let mut sum_v = 0.0f32;
            let mut sum_w = 0.0f32;
            let mut off = -radius;
            while off <= radius {
                let nx = (x as isize + off).clamp(0, max_x) as usize;
                let [li, ui, vi] = src_row[nx];
                let dl = lc - li;
                let du = u_lc - ui * li;
                let dv = v_lc - vi * li;
                let dl2 = dl * dl;
                let dc = (PX * du * du).max(PY * dv * dv);
                let w_l = t_l2 / (dl2 * dl2 + t_l2);
                let w_c = t_c / (dc + t_c);
                let wt = (w_l * w_c - 0.01).max(0.0);
                sum_u += ui * wt;
                sum_v += vi * wt;
                sum_w += wt;
                off += step;
            }
            if sum_w > 1e-6 {
                let inv_w = 1.0 / sum_w;
                out_row[x] = [lc, sum_u * inv_w, sum_v * inv_w];
            } else {
                out_row[x] = [lc, uc, vc];
            }
        }
    });

    let max_y = (h - 1) as isize;
    rgb.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
        for x in 0..w {
            let [lc, uc, vc] = horiz[y * w + x];
            let t_root = (pz + pw * lc).max(1.0);
            let t_l2 = t_root * t_root;
            let t_c = (pz_chroma + pw_chroma * lc).max(1.0);
            let u_lc = uc * lc;
            let v_lc = vc * lc;
            let mut sum_u = 0.0f32;
            let mut sum_v = 0.0f32;
            let mut sum_w = 0.0f32;
            let mut off = -radius;
            while off <= radius {
                let ny = (y as isize + off).clamp(0, max_y) as usize;
                let [li, ui, vi] = horiz[ny * w + x];
                let dl = lc - li;
                let du = u_lc - ui * li;
                let dv = v_lc - vi * li;
                let dl2 = dl * dl;
                let dc = (PX * du * du).max(PY * dv * dv);
                let w_l = t_l2 / (dl2 * dl2 + t_l2);
                let w_c = t_c / (dc + t_c);
                let wt = (w_l * w_c - 0.01).max(0.0);
                sum_u += ui * wt;
                sum_v += vi * wt;
                sum_w += wt;
                off += step;
            }
            let (u, v) = if sum_w > 1e-6 {
                let inv_w = 1.0 / sum_w;
                (sum_u * inv_w, sum_v * inv_w)
            } else {
                (uc, vc)
            };
            let ul = u * lc;
            let vl = v * lc;
            let g = lc - ul * K[0] - vl * K[2] - K[3];
            let r0 = (ul + g).max(1.0);
            let g0 = g.max(1.0);
            let b0 = (vl + g).max(1.0);
            let sc = ((lc - (K[3] - 1.0)) / ((K[0] * r0 + K[1] * g0 + K[2] * b0).max(0.0) + 1.0))
                .min(1.5);
            let lin = [r0 * sc, g0 * sc, b0 * sc];
            out_row[x] = [
                super::process::linear_to_srgb((lin[0] / 65535.0).clamp(0.0, 1.0)),
                super::process::linear_to_srgb((lin[1] / 65535.0).clamp(0.0, 1.0)),
                super::process::linear_to_srgb((lin[2] / 65535.0).clamp(0.0, 1.0)),
            ];
        }
    });
}

fn stage_sharpening_f32(rgb: &mut [[f32; 3]], w: usize, h: usize, adj: &Sharpening, scale: f32) {
    let amount = adj.amount / 100.0;
    if amount <= 0.0 || w == 0 || h == 0 {
        return;
    }
    let raw_sigma = adj.radius.max(0.1) * scale;
    let sigma = raw_sigma.clamp(0.35, 12.0);
    let scale_atten = (raw_sigma / 0.8).clamp(0.18, 1.0);
    let threshold_lin = (adj.threshold.max(0.0) / 255.0) * 0.5;
    let halo = (adj.halo_suppression / 100.0).clamp(0.0, 1.0);
    let gain = amount * 0.50 * scale_atten;

    let mut luma = vec![0.0f32; w * h];
    luma.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let src = &rgb[y * w..(y + 1) * w];
        for x in 0..w {
            let r = srgb_to_linear_f32(src[x][0]);
            let g = srgb_to_linear_f32(src[x][1]);
            let b = srgb_to_linear_f32(src[x][2]);
            row[x] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        }
    });

    let mut out = gaussian_blur_sep(&luma, w, h, sigma);
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = y * w + x;
            let yc = luma[i];
            let mut detail = yc - row[x];
            if detail.abs() <= threshold_lin {
                detail = 0.0;
            } else {
                detail = detail.signum() * (detail.abs() - threshold_lin);
            }
            let mut v = yc + detail * gain;
            if halo > 0.0 {
                let mut lo = f32::MAX;
                let mut hi = f32::MIN;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let xx = clampi(x as i32 + dx, 0, w as i32 - 1) as usize;
                        let yy = clampi(y as i32 + dy, 0, h as i32 - 1) as usize;
                        let l = luma[yy * w + xx];
                        lo = lo.min(l);
                        hi = hi.max(l);
                    }
                }
                let clamped = v.clamp(lo, hi);
                v = v * (1.0 - halo) + clamped * halo;
            }
            row[x] = v.clamp(0.0, 1.0);
        }
    });

    rgb.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let off = y * w;
        for x in 0..w {
            let yc = luma[off + x];
            let yn = out[off + x];
            if (yn - yc).abs() < 1e-6 {
                continue;
            }
            let r = srgb_to_linear_f32(row[x][0]);
            let g = srgb_to_linear_f32(row[x][1]);
            let b = srgb_to_linear_f32(row[x][2]);
            let scaled = if yc > 1e-4 {
                let f = (yn / yc).clamp(0.2, 4.0);
                [r * f, g * f, b * f]
            } else {
                let d = yn - yc;
                [r + d, g + d, b + d]
            };
            row[x] = [
                super::process::linear_to_srgb(scaled[0].clamp(0.0, 1.0)),
                super::process::linear_to_srgb(scaled[1].clamp(0.0, 1.0)),
                super::process::linear_to_srgb(scaled[2].clamp(0.0, 1.0)),
            ];
        }
    });
}

/// Applies the detail stages to one sRGB 8-bit RGBA frame, in place.
///
/// `rgba` is row-major RGBA with no padding, `scale` is output pixels per source
/// pixel (`< 1` when the frame was downscaled) and converts source-pixel radii
/// into output-pixel radii. Every stage is skipped when its own parameters are
/// off, so an all-off adjustment set leaves the bytes untouched.
pub(crate) fn post_process(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    adj: &crate::AdjustmentSet,
    scale: f32,
) {
    post_process_with_lens(rgba, width, height, adj, scale, None);
}

/// Same as [`post_process`], with Capture One's measured lens tables when the
/// variant uses a lens profile.
pub(crate) fn post_process_with_lens(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    adj: &crate::AdjustmentSet,
    scale: f32,
    warp: Option<&crate::mock::lens_profile::LensWarp>,
) {
    post_process_impl(rgba, width, height, adj, scale, warp, detail_gpu::enabled());
}

fn lens_extras(lens: &LensCorrection) -> bool {
    lens.dust_removal_enabled && !lens.dust_patches.is_empty()
        || lens.uniform_light_enabled && lens.uniform_light > 0.0
        || lens.sharpness_falloff >= 0.5 && lens.radial_blur.valid
        || lens.shift_x != 0.0
        || lens.shift_y != 0.0
        || lens.tilt != 0.0
        || lens.diffraction_enabled
}

/// Which detail stages the frame sends to CUDA and which stay on the CPU.
/// Pure so the routing is testable without a GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DetailStagePlan {
    gpu: bool,
    /// Run the CPU single-pixel stage before the GPU call; the CUDA
    /// `luminance_nr_kernel` law is not byte-identical.
    cpu_single_pixel_first: bool,
    /// Run Capture One's `ColorNoise2012` (`LABRangeFilter2012`) on CPU; the
    /// CUDA PTX still carries the legacy 5x5 box blur approximation.
    cpu_color_nr: bool,
    gpu_sharpen: bool,
    gpu_lens: bool,
    /// Keep the non-parity moire approximation off partially split frames.
    cpu_moire: bool,
    cpu_diffraction: bool,
    cpu_sharpen: bool,
    cpu_lens: bool,
}

fn detail_stage_plan(adj: &crate::AdjustmentSet, gpu_enabled: bool) -> DetailStagePlan {
    let lens = &adj.lens;
    let lens_cpu = lens.use_profile || lens_extras(lens);
    // Diffraction sits between moire and USM. The GPU has no diffraction
    // kernel, so sharpening and lens go back to CPU; moire follows the
    // partial-route rule below.
    let diffraction = lens.diffraction_enabled;
    let single_pixel = adj.noise_reduction.single_pixel > 0.0;
    let color_nr = adj.noise_reduction.color > 0.0;
    let partial = lens_cpu || single_pixel || diffraction || color_nr;
    let cpu_color_nr = gpu_enabled && color_nr;
    let cpu_moire = gpu_enabled && partial && adj.moire.amount > 0.0;
    let defer_after_nr = cpu_color_nr || diffraction || cpu_moire;
    let cpu_lens = gpu_enabled && (lens_cpu || cpu_color_nr);
    DetailStagePlan {
        gpu: gpu_enabled,
        cpu_single_pixel_first: gpu_enabled && single_pixel,
        cpu_color_nr,
        gpu_sharpen: gpu_enabled && adj.sharpening.amount > 0.0 && !defer_after_nr,
        gpu_lens: gpu_enabled && !lens_cpu && !cpu_color_nr,
        cpu_moire,
        cpu_diffraction: gpu_enabled && diffraction,
        cpu_sharpen: gpu_enabled && adj.sharpening.amount > 0.0 && defer_after_nr,
        cpu_lens,
    }
}

fn post_process_impl(
    rgba: &mut [u8],
    width: u32,
    height: u32,
    adj: &crate::AdjustmentSet,
    scale: f32,
    warp: Option<&crate::mock::lens_profile::LensWarp>,
    gpu_enabled: bool,
) {
    let w = width as usize;
    let h = height as usize;
    if w == 0 || h == 0 {
        return;
    }
    let Some(expected) = w.checked_mul(h).and_then(|n| n.checked_mul(4)) else {
        return;
    };
    if rgba.len() != expected {
        return;
    }

    let nr = &adj.noise_reduction;
    let lens = &adj.lens;
    let extras = lens_extras(lens);
    let any_active = nr.luminance > 0.0
        || nr.color > 0.0
        || nr.single_pixel > 0.0
        || adj.moire.amount > 0.0
        || adj.sharpening.amount > 0.0
        || lens.distortion != 0.0
        || lens.light_falloff != 0.0
        || lens.chromatic_aberration
        || lens.purple_fringing != 0.0
        || extras
        || adj.film_grain.impact > 0.0;
    if !any_active {
        return;
    }
    let scale = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };

    let plan = detail_stage_plan(adj, gpu_enabled);
    if plan.cpu_single_pixel_first {
        stage_single_pixel(rgba, w, h, nr);
    }

    let gpu = if plan.gpu {
        // The CUDA single-pixel and legacy 5x5 color NR kernels are older
        // approximations; zero them here when CPU runs the C1 oracle stages.
        let mut gpu_nr = nr.clone();
        gpu_nr.single_pixel = 0.0;
        if plan.cpu_color_nr {
            gpu_nr.color = 0.0;
        }
        let mut gpu_sh = adj.sharpening.clone();
        if !plan.gpu_sharpen {
            gpu_sh.amount = 0.0;
        }
        let mut gpu_lens = lens.clone();
        if !plan.gpu_lens {
            gpu_lens.distortion = 0.0;
            gpu_lens.light_falloff = 0.0;
            gpu_lens.chromatic_aberration = false;
            gpu_lens.purple_fringing = 0.0;
        }
        let gpu_moire = if plan.cpu_moire {
            Moire::default()
        } else {
            adj.moire.clone()
        };
        detail_gpu::post_process(rgba, w, h, &gpu_nr, &gpu_moire, &gpu_sh, &gpu_lens, scale)
    } else {
        false
    };

    if gpu {
        if plan.cpu_color_nr {
            stage_color_nr(rgba, w, h, nr, scale);
        }
        if plan.cpu_moire {
            stage_moire(rgba, w, h, &adj.moire);
        }
        if plan.cpu_diffraction {
            stage_diffraction(rgba, w, h, lens, scale);
        }
        if plan.cpu_sharpen {
            stage_sharpening(rgba, w, h, &adj.sharpening, scale);
        }
        if plan.cpu_lens {
            stage_lens(rgba, w, h, lens, warp);
            stage_dust_removal(rgba, w, h, lens);
            stage_movement(rgba, w, h, lens);
            stage_sharpness_falloff(rgba, w, h, lens);
        }
    } else {
        if !plan.cpu_single_pixel_first {
            if nr.single_pixel > 0.0 && nr.luminance > 0.0 {
                let mut pad = padded_luma(rgba, w, h);
                stage_single_pixel_with_pad(rgba, w, h, nr, &pad);
                refresh_padded_luma(rgba, w, h, &mut pad);
                stage_luminance_nr_with_pad(rgba, w, h, nr, &pad);
            } else {
                stage_single_pixel(rgba, w, h, nr);
                stage_luminance_nr(rgba, w, h, nr);
            }
        } else {
            stage_luminance_nr(rgba, w, h, nr);
        }
        stage_color_nr(rgba, w, h, nr, scale);
        stage_moire(rgba, w, h, &adj.moire);
        // Capture One applies diffraction blur inside CaptureSharpening; this
        // CPU stage goes before the USM pass and has no GPU twin.
        stage_diffraction(rgba, w, h, lens, scale);
        stage_sharpening(rgba, w, h, &adj.sharpening, scale);
        stage_lens(rgba, w, h, lens, warp);
        stage_dust_removal(rgba, w, h, lens);
        stage_movement(rgba, w, h, lens);
        stage_sharpness_falloff(rgba, w, h, lens);
    }
    super::grain::stage(rgba, w, h, &adj.film_grain, scale);
}

// ───────────────────────────── shared helpers ─────────────────────────────

#[inline]
fn luma8(p: [u8; 3]) -> f32 {
    0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
}

#[inline]
fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

fn luma_plane(buf: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut plane = vec![0f32; w * h];
    plane.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = (y * w + x) * 4;
            row[x] = luma8([buf[i], buf[i + 1], buf[i + 2]]);
        }
    });
    plane
}

fn padded_luma(buf: &[u8], w: usize, h: usize) -> Vec<f32> {
    let pw = w + 2;
    let ph = h + 2;
    let mut pad = vec![0f32; pw * ph];
    refresh_padded_luma(buf, w, h, &mut pad);
    pad
}

fn refresh_padded_luma(buf: &[u8], w: usize, h: usize, pad: &mut [f32]) {
    let pw = w + 2;
    pad.par_chunks_mut(pw).enumerate().for_each(|(y, row)| {
        let sy = y.saturating_sub(1).min(h - 1);
        for x in 0..w {
            let i = (sy * w + x) * 4;
            row[x + 1] = luma8([buf[i], buf[i + 1], buf[i + 2]]);
        }
        row[0] = row[1];
        row[pw - 1] = row[pw - 2];
    });
}

/// Rewrites RGB from a new luminance while keeping the original hue ratios.
/// `old_luma` is the plane the caller already computed from the same buffer, so
/// this does not recompute it per pixel.
fn apply_luma(buf: &mut [u8], w: usize, old_luma: &[f32], new_luma: &[f32]) {
    buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = x * 4;
            let p = [row[i], row[i + 1], row[i + 2]];
            let c = old_luma[y * w + x];
            let target = new_luma[y * w + x];
            if (target - c).abs() < 0.05 {
                continue;
            }
            let f = if c > 1.0 { target / c } else { 1.0 };
            for k in 0..3 {
                row[i + k] = (p[k] as f32 * f).clamp(0.0, 255.0).round() as u8;
            }
        }
    });
}

fn box_blur_h(src: &[f32], dst: &mut [f32], w: usize, _h: usize, r: i32) {
    let inv = 1.0 / (2 * r + 1) as f32;
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let base = y * w;
        for x in 0..w {
            let mut sum = 0.0f32;
            for k in -r..=r {
                sum += src[base + clampi(x as i32 + k, 0, w as i32 - 1) as usize];
            }
            row[x] = sum * inv;
        }
    });
}

fn box_blur_v(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: i32) {
    let inv = 1.0 / (2 * r + 1) as f32;
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        row.fill(0.0);
        // Contiguous rows let the compiler vectorize without reordering each sum.
        for k in -r..=r {
            let yy = clampi(y as i32 + k, 0, h as i32 - 1) as usize;
            for (sum, value) in row.iter_mut().zip(&src[yy * w..(yy + 1) * w]) {
                *sum += value;
            }
        }
        for value in row {
            *value *= inv;
        }
    });
}

/// Three box passes approximate a Gaussian of a comparable radius.
fn box_blur3(src: &[f32], w: usize, h: usize, r: i32) -> Vec<f32> {
    let mut a = src.to_vec();
    let mut b = vec![0f32; w * h];
    for _ in 0..3 {
        box_blur_h(&a, &mut b, w, h, r);
        std::mem::swap(&mut a, &mut b);
        box_blur_v(&a, &mut b, w, h, r);
        std::mem::swap(&mut a, &mut b);
    }
    a
}

fn mean3x3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut sum = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let xx = clampi(x as i32 + dx, 0, w as i32 - 1) as usize;
                    let yy = clampi(y as i32 + dy, 0, h as i32 - 1) as usize;
                    sum += src[yy * w + xx];
                }
            }
            row[x] = sum / 9.0;
        }
    });
    out
}

// ───────────────────────────── 1. luminance noise reduction ─────────────────────────────

const NEIGHBOURS: [(i32, i32); 8] =
    [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)];

/// Long-exposure hot-pixel suppression, C1's `ZCLEANLONGEXPOSUREAMOUNT`
/// ("Single Pixel"). A pixel is a spike when its luma is more than
/// `threshold(amount)` above the median of its eight neighbours; it is then
/// replaced with that median. Every other pixel keeps its exact bytes, and the
/// neighbours of a spike are only read, never written.
fn stage_single_pixel(buf: &mut [u8], w: usize, h: usize, adj: &NoiseReduction) {
    if adj.single_pixel <= 0.0 {
        return;
    }
    let pad = padded_luma(buf, w, h);
    stage_single_pixel_with_pad(buf, w, h, adj, &pad);
}

fn stage_single_pixel_with_pad(
    buf: &mut [u8],
    w: usize,
    h: usize,
    adj: &NoiseReduction,
    pad: &[f32],
) {
    let amount = (adj.single_pixel / 100.0).clamp(0.0, 1.0);
    if amount <= 0.0 {
        return;
    }
    // 24 codes at the first notch down to 4 at full strength: more amount
    // accepts smaller spikes, while a real star stays above the noise floor.
    let threshold = 24.0 - 20.0 * amount;
    let pw = w + 2;
    buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let c = pad[(y + 1) * pw + x + 1];
            let mut nb = [0f32; 8];
            for (k, (dx, dy)) in NEIGHBOURS.iter().enumerate() {
                nb[k] = pad[(y as i32 + 1 + dy) as usize * pw + (x as i32 + 1 + dx) as usize];
            }
            nb.sort_by(|a, b| a.total_cmp(b));
            let median = (nb[3] + nb[4]) * 0.5;
            if c > median + threshold {
                let i = x * 4;
                let f = if c > 1.0 { median / c } else { 1.0 };
                for k in 0..3 {
                    row[i + k] = (row[i + k] as f32 * f).clamp(0.0, 255.0).round() as u8;
                }
            }
        }
    });
}

/// Radius-1 bilateral filter on luminance: neighbours are weighted by how close
/// their luminance is to the centre, so edges survive while speckle is averaged
/// away. `details` narrows the luminance tolerance, which preserves edges more.
fn stage_luminance_nr(buf: &mut [u8], w: usize, h: usize, adj: &NoiseReduction) {
    if adj.luminance <= 0.0 {
        return;
    }
    let pad = padded_luma(buf, w, h);
    stage_luminance_nr_with_pad(buf, w, h, adj, &pad);
}

fn stage_luminance_nr_with_pad(
    buf: &mut [u8],
    w: usize,
    h: usize,
    adj: &NoiseReduction,
    pad: &[f32],
) {
    let strength = (adj.luminance / 100.0).clamp(0.0, 1.0);
    if strength <= 0.0 {
        return;
    }
    let details = (adj.details / 100.0).clamp(0.0, 1.0);
    // Preserve fine 1-2px hair/feather micro-texture by keeping sigma tight
    // and including the center pixel's self-weight (1.0) in the bilateral sum.
    let sigma = ((2.5 + 26.0 * strength) / (1.0 + 3.5 * details)).max(0.5);
    let two_sigma2 = 2.0 * sigma * sigma;
    // The range weight only depends on |l - c| (0 ..= 255). A quarter-step
    // table replaces ten million `expf` calls with two loads and a lerp; the
    // interpolation error is far below the 8-bit write-back.
    const LUT_N: usize = 1025;
    let mut range_lut = [0f32; LUT_N];
    for (i, v) in range_lut.iter_mut().enumerate() {
        let d = i as f32 * 0.25;
        *v = (-(d * d) / two_sigma2).exp();
    }
    // Neighbours only read this immutable plane; RGB can be updated in place.
    // One replicated border removes the per-neighbour clamping.
    let pw = w + 2;
    const SPATIAL: [f32; 8] =
        [1.0 / 3.0, 0.5, 1.0 / 3.0, 0.5, 0.5, 1.0 / 3.0, 0.5, 1.0 / 3.0];
    let row = w * 4;
    buf.par_chunks_mut(row).enumerate().for_each(|(y, out_row)| {
        for x in 0..w {
            let i = x * 4;
            let p = [out_row[i], out_row[i + 1], out_row[i + 2]];
            let c = pad[(y + 1) * pw + x + 1];
            let mut num = c;
            let mut den = 1.0f32;
            for (k, (dx, dy)) in NEIGHBOURS.iter().enumerate() {
                let l = pad[(y as i32 + 1 + dy) as usize * pw + (x as i32 + 1 + dx) as usize];
                let ad = (l - c).abs() * 4.0;
                let i0 = (ad as usize).min(LUT_N - 2);
                let t = ad - i0 as f32;
                let range = range_lut[i0] + (range_lut[i0 + 1] - range_lut[i0]) * t;
                let wt = range * SPATIAL[k];
                num += wt * l;
                den += wt;
            }
            let mut target = if den > 1e-6 { num / den } else { c };
            target = c + (target - c) * strength;
            let new_l = target.clamp(0.0, 255.0);
            if (new_l - c).abs() >= 0.05 {
                let f = if c > 1.0 { new_l / c } else { 1.0 };
                for ch in 0..3 {
                    out_row[i + ch] = (p[ch] as f32 * f).clamp(0.0, 255.0).round() as u8;
                }
            }
        }
    });
}

// ───────────────────────────── 2. colour noise reduction ─────────────────────────────

/// Capture One `ColorNoise2012` (`LABRangeFilter2012`) linear-domain chroma
/// range filter: encodes linear 16-bit RGB into `(L, (R-G)/L, (B-G)/L)`, runs
/// separable horizontal and vertical edge-preserving range passes with a
/// quartic falloff `max(0, T/(D^2 + T) - 0.01)` over a wide strided window
/// (r=28, s=4 at default `color = 50`, `scale = 1.0`), then reconstructs RGB
/// with luminance preserved.
fn stage_color_nr(buf: &mut [u8], w: usize, h: usize, adj: &NoiseReduction, scale: f32) {
    let strength = (adj.color / 100.0).clamp(0.0, 1.0);
    if strength <= 0.0 || w == 0 || h == 0 {
        return;
    }

    // Captured C1 ColorNoise2012 constants (ColorNoiseReductionAmount = 50).
    const K: [f32; 4] = [0.299, 0.587, 0.114, 35.759_254];
    const PX: f32 = 0.178_877_62;
    const PY: f32 = 0.377_672_3;
    const PZ_BASE: f32 = 0.401_832_58;
    const PW_BASE: f32 = 2.968_700_4;

    let scale_p = 0.35 + 1.3 * strength;
    let pz = PZ_BASE * scale_p;
    let pw = PW_BASE * scale_p;
    // Shadow-aware chroma variance tolerance so high-ISO shadow chroma blotches
    // are strongly suppressed while luma boundaries stay sharp (no color bleed).
    let pz_chroma = 2_400.0 + 18_000.0 * strength * strength;
    let pw_chroma = 12.0 + 36.0 * strength;

    let (radius, step) = color_filter_support(adj.color, scale);

    let mut srgb_to_lin16 = [0.0f32; 256];
    for (i, slot) in srgb_to_lin16.iter_mut().enumerate() {
        let v = i as f32 / 255.0;
        let lin = if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
        *slot = lin * 65535.0;
    }

    // 1. Encode sRGB -> linear 16-bit -> (L, u, v).
    let n = w * h;
    let mut encoded = vec![[0.0f32; 3]; n];
    encoded.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let base = y * w * 4;
        for x in 0..w {
            let i = base + x * 4;
            let r = srgb_to_lin16[buf[i] as usize];
            let g = srgb_to_lin16[buf[i + 1] as usize];
            let b = srgb_to_lin16[buf[i + 2] as usize];
            let l = (K[0] * r + K[1] * g + K[2] * b).max(0.0) + K[3];
            let inv_l = 1.0 / l;
            row[x] = [l, (r - g) * inv_l, (b - g) * inv_l];
        }
    });

    // 2. Horizontal range filter pass.
    let mut horiz = vec![[0.0f32; 3]; n];
    horiz.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
        let src_row = &encoded[y * w..(y + 1) * w];
        let max_x = (w - 1) as isize;
        for x in 0..w {
            let [lc, uc, vc] = src_row[x];
            let t_root = (pz + pw * lc).max(1.0);
            let t_l2 = t_root * t_root;
            let t_c = (pz_chroma + pw_chroma * lc).max(1.0);
            let u_lc = uc * lc;
            let v_lc = vc * lc;
            let mut sum_u = 0.0f32;
            let mut sum_v = 0.0f32;
            let mut sum_w = 0.0f32;
            let mut off = -radius;
            while off <= radius {
                let nx = (x as isize + off).clamp(0, max_x) as usize;
                let [li, ui, vi] = src_row[nx];
                let dl = lc - li;
                let du = u_lc - ui * li;
                let dv = v_lc - vi * li;
                let dl2 = dl * dl;
                let dc = (PX * du * du).max(PY * dv * dv);
                let w_l = t_l2 / (dl2 * dl2 + t_l2);
                let w_c = t_c / (dc + t_c);
                let wt = (w_l * w_c - 0.01).max(0.0);
                sum_u += ui * wt;
                sum_v += vi * wt;
                sum_w += wt;
                off += step;
            }
            if sum_w > 1e-6 {
                let inv_w = 1.0 / sum_w;
                out_row[x] = [lc, sum_u * inv_w, sum_v * inv_w];
            } else {
                out_row[x] = [lc, uc, vc];
            }
        }
    });

    // 3. Vertical range filter pass + 4. Linear RGB reconstruction -> sRGB u8.
    let max_y = (h - 1) as isize;
    buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, out_row)| {
        for x in 0..w {
            let [lc, uc, vc] = horiz[y * w + x];
            let t_root = (pz + pw * lc).max(1.0);
            let t_l2 = t_root * t_root;
            let t_c = (pz_chroma + pw_chroma * lc).max(1.0);
            let u_lc = uc * lc;
            let v_lc = vc * lc;
            let mut sum_u = 0.0f32;
            let mut sum_v = 0.0f32;
            let mut sum_w = 0.0f32;
            let mut off = -radius;
            while off <= radius {
                let ny = (y as isize + off).clamp(0, max_y) as usize;
                let [li, ui, vi] = horiz[ny * w + x];
                let dl = lc - li;
                let du = u_lc - ui * li;
                let dv = v_lc - vi * li;
                let dl2 = dl * dl;
                let dc = (PX * du * du).max(PY * dv * dv);
                let w_l = t_l2 / (dl2 * dl2 + t_l2);
                let w_c = t_c / (dc + t_c);
                let wt = (w_l * w_c - 0.01).max(0.0);
                sum_u += ui * wt;
                sum_v += vi * wt;
                sum_w += wt;
                off += step;
            }
            let (u, v) = if sum_w > 1e-6 {
                let inv_w = 1.0 / sum_w;
                (sum_u * inv_w, sum_v * inv_w)
            } else {
                (uc, vc)
            };
            let ul = u * lc;
            let vl = v * lc;
            let g = lc - ul * K[0] - vl * K[2] - K[3];
            let r0 = (ul + g).max(1.0);
            let g0 = g.max(1.0);
            let b0 = (vl + g).max(1.0);
            let sc = ((lc - (K[3] - 1.0)) / ((K[0] * r0 + K[1] * g0 + K[2] * b0).max(0.0) + 1.0))
                .min(1.5);
            let lin = [r0 * sc, g0 * sc, b0 * sc];
            let i = x * 4;
            for k in 0..3 {
                let srgb = super::process::linear_to_srgb((lin[k] / 65535.0).clamp(0.0, 1.0));
                out_row[i + k] = (srgb * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    });
}

// ───────────────────────────── 3. moire ─────────────────────────────

/// Chroma-only moire suppression, ported from the `Moire` kernel's v4 branch.
///
/// C1 converts RGB into `(Y, R-G, B-G)`, smooths only the two chroma channels
/// over a `2R+1` window sampled every 2 or 4 pixels, keeps a neighbour only
/// when its luma and luma-scaled chroma are close to the centre, and then
/// blends the result back by `amount`. Luminance is never filtered, so a
/// luminance pattern survives and only colour moire disappears.
///
/// short cut: v4 is a single-axis pass and the host runs it twice; this CPU
/// path does a separable H then V pass with the same guard, B-level parity.
fn stage_moire(buf: &mut [u8], w: usize, h: usize, adj: &Moire) {
    let amount = (adj.amount / 100.0).clamp(0.0, 1.0);
    if amount <= 0.0 || w == 0 || h == 0 || buf.len() < w * h * 4 {
        return;
    }
    // Host: radius = round(2*R*zoom + 1) on a 4-grid, then clamp to 37.
    let radius = adj.pattern.clamp(0.0, 37.0).round() as isize;
    if radius < 1 {
        return;
    }
    let step = if radius < 6 { 2 } else { 4 };
    // short cut: v4's three thresholds live in host fields not fully resolved;
    // these are fixed at roughly 15 and 30 codes of 255.
    const LUMA_TOL: f32 = 15.0 / 255.0;
    const CHROMA_TOL: f32 = 30.0 / 255.0;

    let n = w * h;
    let mut y = vec![0f32; n];
    let mut cr = vec![0f32; n];
    let mut cb = vec![0f32; n];
    for (i, px) in buf.chunks_exact(4).enumerate() {
        let (r, g, b) = (px[0] as f32 / 255.0, px[1] as f32 / 255.0, px[2] as f32 / 255.0);
        y[i] = (r + 2.0 * g + b) * 0.25;
        cr[i] = r - g;
        cb[i] = b - g;
    }
    let avg = |input: &[f32], vertical: bool| -> Vec<f32> {
        let mut out = vec![0f32; n];
        out.par_chunks_mut(w).enumerate().for_each(|(row, dst)| {
            for col in 0..w {
                let (x, yy) = (col as isize, row as isize);
                let c = row * w + col;
                let (yc, crc, cbc) = (y[c], cr[c] * y[c], cb[c] * y[c]);
                let mut sum = 0.0;
                let mut count = 0.0;
                let mut k = -radius;
                while k <= radius {
                    let (sx, sy) = if vertical {
                        (x, (yy + k).clamp(0, h as isize - 1))
                    } else {
                        ((x + k).clamp(0, w as isize - 1), yy)
                    };
                    let j = sy as usize * w + sx as usize;
                    let wgt = if k == 0 {
                        1.0
                    } else {
                        let dy = (y[j] - yc).abs();
                        let dcr = (cr[j] * y[j] - crc).abs();
                        let dcb = (cb[j] * y[j] - cbc).abs();
                        if dy < LUMA_TOL && dcr < CHROMA_TOL && dcb < CHROMA_TOL { 1.0 } else { 0.0 }
                    };
                    sum += input[j] * wgt;
                    count += wgt;
                    k += step;
                }
                dst[col] = if count > 0.0 { sum / count } else { input[c] };
            }
        });
        out
    };
    let cr_f = {
        let cr_h = avg(&cr, false);
        avg(&cr_h, true)
    };
    let cb_f = {
        let cb_h = avg(&cb, false);
        avg(&cb_h, true)
    };

    for (i, px) in buf.chunks_exact_mut(4).enumerate() {
        let (c0, c1) = (cr[i] + (cr_f[i] - cr[i]) * amount, cb[i] + (cb_f[i] - cb[i]) * amount);
        let g = y[i] - (c0 + c1) * 0.25;
        let rgb = [g + c0, g, g + c1];
        for (k, v) in rgb.iter().enumerate() {
            px[k] = (v.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    }
}

// ───────────────────────────── 4. sharpening ─────────────────────────────

fn gaussian_blur_sep(src: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    let half = gaussian_support(sigma);
    let mut kernel = vec![0.0f32; half + 1];
    let inv2s2 = 1.0 / (2.0 * sigma * sigma);
    let mut sum = 1.0f32;
    kernel[0] = 1.0;
    for k in 1..=half {
        let wk = (-((k * k) as f32) * inv2s2).exp();
        kernel[k] = wk;
        sum += 2.0 * wk;
    }
    let inv_sum = 1.0 / sum;
    for v in &mut kernel {
        *v *= inv_sum;
    }

    let max_x = (w - 1) as isize;
    let mut horiz = vec![0.0f32; w * h];
    horiz.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
        let row = &src[y * w..(y + 1) * w];
        for x in 0..w {
            let mut acc = row[x] * kernel[0];
            for k in 1..=half {
                let l = (x as isize - k as isize).clamp(0, max_x) as usize;
                let r = (x as isize + k as isize).clamp(0, max_x) as usize;
                acc += (row[l] + row[r]) * kernel[k];
            }
            out_row[x] = acc;
        }
    });

    let max_y = (h - 1) as isize;
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, out_row)| {
        let c_row = &horiz[y * w..(y + 1) * w];
        for x in 0..w {
            out_row[x] = c_row[x] * kernel[0];
        }
        for k in 1..=half {
            let u = (y as isize - k as isize).clamp(0, max_y) as usize;
            let d = (y as isize + k as isize).clamp(0, max_y) as usize;
            let wk = kernel[k];
            let u_row = &horiz[u * w..(u + 1) * w];
            let d_row = &horiz[d * w..(d + 1) * w];
            for x in 0..w {
                out_row[x] += (u_row[x] + d_row[x]) * wk;
            }
        }
    });
    out
}

/// Capture One `UnsharpMaskLinear` (queue stages 63-75): performs unsharp
/// masking in scene-linear light with a true continuous separable Gaussian
/// kernel so sub-pixel radii (`0.8 px`) resolve fine hair/feather micro-texture
/// without the coarse 1.41px halo of a 3-pass box blur.
fn stage_sharpening(buf: &mut [u8], w: usize, h: usize, adj: &Sharpening, scale: f32) {
    let amount = adj.amount / 100.0;
    if amount <= 0.0 || w == 0 || h == 0 {
        return;
    }
    let raw_sigma = adj.radius.max(0.1) * scale;
    let sigma = raw_sigma.clamp(0.35, 12.0);
    let scale_atten = (raw_sigma / 0.8).clamp(0.18, 1.0);
    let threshold_lin = (adj.threshold.max(0.0) / 255.0) * 0.5;
    let halo = (adj.halo_suppression / 100.0).clamp(0.0, 1.0);
    let gain = amount * 0.50 * scale_atten;

    let mut srgb_to_lin = [0.0f32; 256];
    for (i, slot) in srgb_to_lin.iter_mut().enumerate() {
        let v = i as f32 / 255.0;
        *slot = if v <= 0.04045 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) };
    }

    let mut luma = vec![0.0f32; w * h];
    luma.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let base = y * w * 4;
        for x in 0..w {
            let i = base + x * 4;
            let r = srgb_to_lin[buf[i] as usize];
            let g = srgb_to_lin[buf[i + 1] as usize];
            let b = srgb_to_lin[buf[i + 2] as usize];
            row[x] = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        }
    });

    let mut out = gaussian_blur_sep(&luma, w, h, sigma);
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = y * w + x;
            let yc = luma[i];
            let mut detail = yc - row[x];
            if detail.abs() <= threshold_lin {
                detail = 0.0;
            } else {
                detail = detail.signum() * (detail.abs() - threshold_lin);
            }
            let mut v = yc + detail * gain;
            if halo > 0.0 {
                let mut lo = f32::MAX;
                let mut hi = f32::MIN;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let xx = clampi(x as i32 + dx, 0, w as i32 - 1) as usize;
                        let yy = clampi(y as i32 + dy, 0, h as i32 - 1) as usize;
                        let l = luma[yy * w + xx];
                        lo = lo.min(l);
                        hi = hi.max(l);
                    }
                }
                let clamped = v.clamp(lo, hi);
                v = v * (1.0 - halo) + clamped * halo;
            }
            row[x] = v.clamp(0.0, 1.0);
        }
    });

    buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        let off = y * w;
        for x in 0..w {
            let yc = luma[off + x];
            let yn = out[off + x];
            if (yn - yc).abs() < 1e-6 {
                continue;
            }
            let i = x * 4;
            let r = srgb_to_lin[row[i] as usize];
            let g = srgb_to_lin[row[i + 1] as usize];
            let b = srgb_to_lin[row[i + 2] as usize];
            let scaled = if yc > 1e-4 {
                let f = (yn / yc).clamp(0.2, 4.0);
                [r * f, g * f, b * f]
            } else {
                let d = yn - yc;
                [r + d, g + d, b + d]
            };
            for k in 0..3 {
                let srgb = super::process::linear_to_srgb(scaled[k].clamp(0.0, 1.0));
                row[i + k] = (srgb * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
            }
        }
    });
}

// ───────────────────────────── 5. lens correction ─────────────────────────────

fn sample_bilinear(src: &[u8], w: usize, h: usize, x: f32, y: f32) -> [f32; 3] {
    let fx = x - 0.5;
    let fy = y - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = fx - x0;
    let ty = fy - y0;
    let x0 = x0 as i32;
    let y0 = y0 as i32;
    let mut acc = [0f32; 3];
    for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
        for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
            let sx = clampi(x0 + dx, 0, w as i32 - 1) as usize;
            let sy = clampi(y0 + dy, 0, h as i32 - 1) as usize;
            let i = (sy * w + sx) * 4;
            for k in 0..3 {
                acc[k] += src[i + k] as f32 * wx * wy;
            }
        }
    }
    acc
}

/// Radial resampling for lens distortion (positive corrects barrel, negative
/// corrects pincushion), a radial gain for light falloff, per-channel radial
/// offsets for chromatic aberration, and a purple-fringe clean-up pass.
pub(crate) fn stage_lens(
    buf: &mut [u8],
    w: usize,
    h: usize,
    adj: &LensCorrection,
    warp: Option<&crate::mock::lens_profile::LensWarp>,
) {
    let distortion = (adj.distortion / 100.0).clamp(-1.0, 1.0);
    let falloff = (adj.light_falloff / 100.0).clamp(0.0, 1.0);
    let purple = crate::mock::purple_norm(adj.purple_fringing);
    let ca = adj.chromatic_aberration;
    // The six-curve profile table is applied after the warp, not inside it.
    let ca_in_warp = ca && adj.ca_fringing.is_empty();
    if distortion == 0.0 && falloff == 0.0 && !ca && purple == crate::mock::PURPLE_IDENTITY {
        return;
    }
    if adj.use_profile {
        // A requested profile with no calibration table for this lens leaves
        // the pixels alone, as Capture One does: the slider has no data behind
        // it. Guessing a radial term here bent the whole frame.
        let ca_points = ca
            && (!adj.ca_red_points.is_empty()
                || !adj.ca_blue_points.is_empty()
                || !adj.ca_fringing.is_empty()
                || warp.is_some_and(|w| !w.red.is_empty() || !w.blue.is_empty()));
        let usable = warp.is_some_and(|w| !w.distortion.is_empty() || !w.falloff.is_empty()) || ca_points;
        if usable {
            let empty = crate::mock::lens_profile::LensWarp::default();
            let warp = warp.unwrap_or(&empty);
            stage_lens_profile(buf, w, h, adj, warp, distortion, falloff, ca_in_warp);
            stage_ca_fringes(buf, w, h, adj);
            if purple != crate::mock::PURPLE_IDENTITY {
                stage_purple_fringing(buf, w, h, purple);
            }
        }
        return;
    }
    let src = buf.to_vec();
    let cx = w as f32 * 0.5;
    let cy = h as f32 * 0.5;
    let mut out = vec![0u8; w * h * 4];
    out.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = x * 4;
            let dx = (x as f32 + 0.5 - cx) / cx;
            let dy = (y as f32 + 0.5 - cy) / cy;
            let r2 = dx * dx + dy * dy;
            // Radial remap: sample further out for barrel correction.
            let radial = 1.0 + 0.30 * distortion * r2;
            let sx = cx + dx * cx * radial;
            let sy = cy + dy * cy * radial;
            let outside = sx < 0.0 || sy < 0.0 || sx > w as f32 || sy > h as f32;
            if outside && adj.hide_distorted_areas {
                row[i] = 0;
                row[i + 1] = 0;
                row[i + 2] = 0;
                row[i + 3] = 255;
                continue;
            }
            let mut rgb = sample_bilinear(&src, w, h, sx, sy);
            if ca_in_warp {
                // Red and blue are sampled at slightly different radii.
                let amount = 0.004 * r2;
                let r = sample_bilinear(
                    &src,
                    w,
                    h,
                    cx + (sx - cx) * (1.0 + amount),
                    cy + (sy - cy) * (1.0 + amount),
                );
                let b = sample_bilinear(
                    &src,
                    w,
                    h,
                    cx + (sx - cx) * (1.0 - amount),
                    cy + (sy - cy) * (1.0 - amount),
                );
                rgb[0] = r[0];
                rgb[2] = b[2];
            }
            if falloff > 0.0 {
                let gain = 1.0 + falloff * 0.35 * r2;
                for k in 0..3 {
                    rgb[k] *= gain;
                }
            }
            for k in 0..3 {
                row[i + k] = rgb[k].clamp(0.0, 255.0).round() as u8;
            }
            row[i + 3] = 255;
        }
    });
    buf.copy_from_slice(&out);
    stage_ca_fringes(buf, w, h, adj);
    if purple != crate::mock::PURPLE_IDENTITY {
        stage_purple_fringing(buf, w, h, purple);
    }
}

const DIFFRACTION_LAMBDA_UM: f32 = 0.42;
/// Used when the mount has no sensor entry; the catalog's `ZPIXELWIDTH` is
/// 5.94 µm on a full-frame capture, so 4 µm is a deliberately conservative
/// default that produces a smaller PSF rather than a smeared frame.
const FALLBACK_PIXEL_PITCH_UM: f32 = 4.0;

/// Bessel J1 from Abramowitz & Stegun 9.4.4 / 9.4.6; accurate to ~1e-7,
/// which is far below the 8-bit output of the stage.
fn bessel_j1(x: f32) -> f32 {
    let ax = x.abs();
    if ax < 8.0 {
        let y = x * x;
        let num = x
            * (72_362_614_232.0
                + y * (-7_895_059_235.0
                    + y * (242_396_853.1
                        + y * (-2_972_611.439
                            + y * (15_704.482_60 + y * (-30.160_366_06))))));
        let den = 144_725_228_442.0
            + y * (2_300_535_178.0
                + y * (18_583_304.74 + y * (99_447.433_94 + y * (376.999_139_7 + y))));
        num / den
    } else {
        let z = 8.0 / ax;
        let y = z * z;
        let xx = ax - 2.356_194_491;
        let p1 = 1.0
            + y * (0.183_105e-2
                + y * (-0.351_639_649_6e-4
                    + y * (0.245_752_017_4e-5 + y * (-0.240_337_019e-6))));
        let p2 = 0.046_874_999_95
            + y * (-0.200_269_087_3e-3
                + y * (0.844_919_909_6e-5
                    + y * (-0.882_289_87e-6 + y * 0.105_787_412e-6)));
        let ans = (0.636_619_772 / ax).sqrt() * (xx.cos() * p1 - z * xx.sin() * p2);
        if x < 0.0 {
            -ans
        } else {
            ans
        }
    }
}

/// Photosite pitch from the mount's sensor diagonal and the buffer size.
/// On a preview buffer this already yields the scaled PSF radius.
fn pixel_pitch_um(adj: &LensCorrection, w: usize, h: usize, scale: f32) -> f32 {
    let diagonal = ((w * w + h * h) as f32).sqrt();
    if diagonal <= 0.0 {
        return 0.0;
    }
    match adj
        .mount
        .as_deref()
        .and_then(crate::lens_correct::mount_half_diagonal_um)
    {
        Some(half) => 2.0 * half / diagonal,
        // The fallback is a full-resolution pitch, so a preview buffer needs
        // the matching scale to keep the same angular PSF.
        None => FALLBACK_PIXEL_PITCH_UM / scale.max(1e-6),
    }
}

/// Normalized Airy `(2 J1(v)/v)^2` on a square grid.
fn airy_kernel(radius: i32, v_per_px: f32) -> Vec<f32> {
    let n = (radius * 2 + 1) as usize;
    let mut kernel = vec![0f32; n * n];
    let mut sum = 0f32;
    for ky in 0..n {
        for kx in 0..n {
            let dx = kx as i32 - radius;
            let dy = ky as i32 - radius;
            let r = ((dx * dx + dy * dy) as f32).sqrt();
            let v = v_per_px * r;
            let amplitude = if v < 1e-4 { 1.0 } else { 2.0 * bessel_j1(v) / v };
            let w = (amplitude * amplitude).max(0.0);
            kernel[ky * n + kx] = w;
            sum += w;
        }
    }
    if sum > 0.0 {
        for w in kernel.iter_mut() {
            *w /= sum;
        }
    } else {
        kernel[(radius as usize) * n + radius as usize] = 1.0;
    }
    kernel
}

/// Capture One gates diffraction on f/1..50 and a real photosite pitch; below
/// ~0.35 px the Airy core is a delta on an 8-bit buffer, so leave it alone.
/// shortcut: a limited-radius truncation, not C1's full PSF integration; upgrade
/// when a lens/aperture sweep can calibrate the kernel radius.
fn stage_diffraction(buf: &mut [u8], w: usize, h: usize, adj: &LensCorrection, scale: f32) {
    if !adj.diffraction_enabled {
        return;
    }
    if w == 0 || h == 0 || buf.len() != w * h * 4 {
        return;
    }
    let f_number = adj.aperture;
    if !(1.0..=50.0).contains(&f_number) {
        return;
    }
    let pitch = pixel_pitch_um(adj, w, h, scale);
    if pitch <= 0.0 || !pitch.is_finite() {
        return;
    }
    let v_per_px = std::f32::consts::PI * pitch / (DIFFRACTION_LAMBDA_UM * f_number);
    if v_per_px <= 0.0 || !v_per_px.is_finite() {
        return;
    }
    let first_zero_px = 3.831_705_6 / v_per_px;
    if first_zero_px < 0.35 {
        return;
    }
    let radius = ((first_zero_px * 2.2).ceil() as i32).clamp(1, 6);
    let kernel = airy_kernel(radius, v_per_px);
    let n = (radius * 2 + 1) as usize;
    let radius_px = radius as usize;
    let row_bytes = w * 4;
    let mut source_band = Vec::new();
    let mut previous = vec![0u8; radius_px * row_bytes];
    let mut y0 = 0usize;
    while y0 < h {
        let y1 = (y0 + DIFFRACTION_BAND_ROWS).min(h);
        let source_y0 = y0.saturating_sub(radius_px);
        let source_y1 = (y1 + radius_px).min(h);
        let source_rows = source_y1 - source_y0;
        source_band.resize(source_rows * row_bytes, 0);

        for gy in source_y0..source_y1 {
            let dst = (gy - source_y0) * row_bytes;
            if gy < y0 {
                let src = (gy - source_y0) * row_bytes;
                source_band[dst..dst + row_bytes].copy_from_slice(&previous[src..src + row_bytes]);
            } else {
                let src = gy * row_bytes;
                source_band[dst..dst + row_bytes].copy_from_slice(&buf[src..src + row_bytes]);
            }
        }
        if y1 < h {
            let carry_y0 = y1.saturating_sub(radius_px);
            for gy in carry_y0..y1 {
                let dst = (gy - carry_y0) * row_bytes;
                let src = (gy - source_y0) * row_bytes;
                previous[dst..dst + row_bytes]
                    .copy_from_slice(&source_band[src..src + row_bytes]);
            }
        }

        let dst_start = y0 * row_bytes;
        let dst_end = y1 * row_bytes;
        buf[dst_start..dst_end]
            .par_chunks_mut(row_bytes)
            .enumerate()
            .for_each(|(band_y, row)| {
                let y = y0 + band_y;
                for x in 0..w {
                    let i = x * 4;
                    let mut acc = [0f32; 3];
                    for ky in 0..n {
                        let sy = clampi(y as i32 + ky as i32 - radius, 0, h as i32 - 1) as usize;
                        let source_row = (sy - source_y0) * row_bytes;
                        for kx in 0..n {
                            let sx =
                                clampi(x as i32 + kx as i32 - radius, 0, w as i32 - 1) as usize;
                            let weight = kernel[ky * n + kx];
                            let j = source_row + sx * 4;
                            acc[0] += source_band[j] as f32 * weight;
                            acc[1] += source_band[j + 1] as f32 * weight;
                            acc[2] += source_band[j + 2] as f32 * weight;
                        }
                    }
                    for c in 0..3 {
                        row[i + c] = acc[c].round().clamp(0.0, 255.0) as u8;
                    }
                }
            });
        y0 = y1;
    }
}

/// `ZLENSDUSTREMOVALENABLED` + the LCC's dust rectangles. Mirrors
/// `CImgOpApplyLCC::ApplyLCCDust`: a Q12 multiply, nothing when the table is
/// empty (the common case for a variant without an LCC).
fn stage_dust_removal(buf: &mut [u8], w: usize, h: usize, adj: &LensCorrection) {
    if !adj.dust_removal_enabled || adj.dust_patches.is_empty() || w == 0 || h == 0 {
        return;
    }
    crate::dust_patch::apply(buf, w, h, &adj.dust_patches, &adj.dust_gain);
}

/// `ZLENSSHIFTXY` + `ZLENSTILT` as one 3x3 map, following `CResampler::UpdateGeo`:
/// `tq = atan(m * tan(tilt))`, `k = tan(tq / 2)` shears the perspective rows.
fn stage_movement(buf: &mut [u8], w: usize, h: usize, adj: &LensCorrection) {
    let tilt = adj.tilt.clamp(-45.0, 45.0);
    if w == 0 || h == 0 || (adj.shift_x == 0.0 && adj.shift_y == 0.0 && tilt == 0.0) {
        return;
    }
    let shift_mag = adj.shift_x.hypot(adj.shift_y);
    // Native multiplies the tilt by the shared movement amount; a variant with
    // only a tilt keeps unit scale instead of collapsing to no-op.
    let m = if shift_mag > 0.0 { 0.01 * shift_mag } else { 1.0 };
    let tq = (m * tilt.to_radians().tan()).atan();
    let k = (tq * 0.5).tan();
    let (mut m0, mut m1, mut m2, mut m3, mut m4, mut m5) = (1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0);
    m0 += k * m4;
    m1 += k * m4;
    m2 += k * m4;
    m3 += k * m0;
    m4 += k * m1;
    m5 += k * m2;
    let diag = (((w * w + h * h) as f32).sqrt()).max(1.0);
    m2 /= diag;
    m5 /= diag;
    let tx = adj.shift_x * 0.01 * w as f32 * 0.5;
    let ty = adj.shift_y * 0.01 * h as f32 * 0.5;
    let src = buf.to_vec();
    let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
    buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let xx = x as f32 + 0.5 - cx;
            let yy = y as f32 + 0.5 - cy;
            let a = m0 - m2 * xx;
            let b = m1 - m2 * yy;
            let c = m3 - m5 * xx;
            let d = m4 - m5 * yy;
            let det = d * a - c * b;
            let (u, v) = if det.abs() < 1e-6 {
                (xx, yy)
            } else {
                ((d * xx - c * yy) / det - tx, (a * yy - b * xx) / det - ty)
            };
            let rgb = sample_bilinear(&src, w, h, cx + u, cy + v);
            let i = x * 4;
            for k in 0..3 {
                row[i + k] = rgb[k].clamp(0.0, 255.0).round() as u8;
            }
            row[i + 3] = src[(y * w + x) * 4 + 3];
        }
    });
}

/// One separable 1D Gaussian gather along `d`, weights `exp(-4 ln2 t^2)` and
/// `step = min(0.2, 0.5/|d|)` from `CImgOpFixRadialBlur`'s feather kernel.
fn gauss_gather(src: &[u8], w: usize, h: usize, x: f32, y: f32, d: [f32; 2]) -> [f32; 3] {
    let mag = d[0].hypot(d[1]);
    if mag < 0.1 {
        return sample_bilinear(src, w, h, x, y);
    }
    let step = 0.2f32.min(0.5 / mag);
    let (ux, uy) = (d[0] / mag, d[1] / mag);
    let mut acc = sample_bilinear(src, w, h, x, y);
    let mut sum = 1.0;
    for i in 1..=18 {
        let t = i as f32 * step;
        let weight = (-4.0 * std::f32::consts::LN_2 * t * t).exp();
        if weight < 1e-3 {
            break;
        }
        let plus = sample_bilinear(src, w, h, x + ux * t, y + uy * t);
        let minus = sample_bilinear(src, w, h, x - ux * t, y - uy * t);
        for k in 0..3 {
            acc[k] += (plus[k] + minus[k]) * weight;
        }
        sum += 2.0 * weight;
    }
    [acc[0] / sum, acc[1] / sum, acc[2] / sum]
}

/// `ZLENSRADIALBLUR`: radial then transverse 1D Gaussian gathers. `valid == 0`
/// or `amount < 0.5` is identity, matching `CImgOpFixRadialBlur::IsIdentity`.
fn stage_sharpness_falloff(buf: &mut [u8], w: usize, h: usize, adj: &LensCorrection) {
    let curves = &adj.radial_blur;
    let amount = adj.sharpness_falloff;
    if !curves.valid || amount < 0.5 || w == 0 || h == 0 {
        return;
    }
    if curves.radial.is_empty() && curves.transverse.is_empty() {
        return;
    }
    let s = std::f32::consts::SQRT_2 / (((w * w + h * h) as f32).sqrt()).max(1.0);
    let scale = amount * 0.01 * s;
    let nodes = |points: &[(f32, f32)]| -> Vec<(f32, f32)> {
        points.iter().map(|&(x, y)| (s * x, scale * y)).collect()
    };
    let radial = nodes(&curves.radial);
    let transverse = nodes(&curves.transverse);
    let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
    for (pass, curve) in [&radial, &transverse].into_iter().enumerate() {
        if curve.is_empty() {
            continue;
        }
        let src = buf.to_vec();
        buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cy;
                let r = dx.hypot(dy);
                let disp = crate::mock::lens_profile::LensWarp::sample_curve(curve, r)
                    .clamp(0.0, 11.0)
                    * s;
                let (ux, uy) = if r > 1e-6 { (dx / r, dy / r) } else { (0.0, 0.0) };
                let d = if pass == 0 { [disp * ux, disp * uy] } else { [-disp * uy, disp * ux] };
                let rgb = gauss_gather(&src, w, h, x as f32 + 0.5, y as f32 + 0.5, d);
                let i = x * 4;
                for k in 0..3 {
                    row[i + k] = rgb[k].clamp(0.0, 255.0).round() as u8;
                }
                row[i + 3] = src[(y * w + x) * 4 + 3];
            }
        });
    }
}

/// The lens stages that only exist on the 8-bit preview path.
pub(crate) fn stage_lens_extras(buf: &mut [u8], w: usize, h: usize, adj: &LensCorrection) {
    stage_movement(buf, w, h, adj);
    stage_sharpness_falloff(buf, w, h, adj);
}

/// Capture One's measured tables. Radius is measured on the render buffer's half
/// diagonal, which equals the table's own reference (the half diagonal of the
/// frame it was measured on) for an uncropped frame.
fn stage_lens_profile(
    buf: &mut [u8],
    w: usize,
    h: usize,
    adj: &LensCorrection,
    warp: &crate::mock::lens_profile::LensWarp,
    distortion: f32,
    falloff: f32,
    ca: bool,
) {
    use crate::mock::lens_profile::LensWarp;

    // With the six-curve table present the two-curve splines are not applied.
    let ca = ca && adj.ca_fringing.is_empty();
    let src = buf.to_vec();
    let cx = w as f32 * 0.5;
    let cy = h as f32 * 0.5;
    let half_diag = ((w * w + h * h) as f32).sqrt() * 0.5;
    let mut out = vec![0u8; w * h * 4];
    out.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = x * 4;
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let r_px = (dx * dx + dy * dy).sqrt();
            let t = r_px / half_diag;
            // The table gives how far the lens moved a point outward, so the
            // source for an output pixel sits at `t - shift`. The sign follows
            // the measured EF 11-24 barrel table and is unverified against C1.
            let t_src = t - LensWarp::sample_curve(&warp.distortion, t) * distortion;
            let k = if r_px > 1e-4 { t_src / t } else { 1.0 };
            let sx = cx + dx * k;
            let sy = cy + dy * k;
            let outside = sx < 0.0 || sy < 0.0 || sx > w as f32 || sy > h as f32;
            if outside && adj.hide_distorted_areas {
                row[i + 3] = 255;
                continue;
            }
            let mut rgb = sample_bilinear(&src, w, h, sx, sy);
            if ca && r_px > 1e-4 {
                let red_points = if adj.ca_red_points.is_empty() { &warp.red } else { &adj.ca_red_points };
                let blue_points = if adj.ca_blue_points.is_empty() { &warp.blue } else { &adj.ca_blue_points };
                // The CA table is relative to green, so its shift rides on top
                // of the distortion's own sampling scale `k`. Comparing the
                // ratio against `t` here used to force a resample even with an
                // empty table, misaligning red/blue against green.
                let offset = |shift: &[(f32, f32)]| -> f32 { k + LensWarp::sample(shift, t) / t };
                let red_offset = offset(red_points);
                let blue_offset = offset(blue_points);
                if (red_offset - k).abs() > 1e-5 {
                    rgb[0] =
                        sample_bilinear(&src, w, h, cx + dx * red_offset, cy + dy * red_offset)[0];
                }
                if (blue_offset - k).abs() > 1e-5 {
                    rgb[2] =
                        sample_bilinear(&src, w, h, cx + dx * blue_offset, cy + dy * blue_offset)[2];
                }
            }
            if falloff > 0.0 {
                // Table EV is a loss (negative toward the corner); undo it.
                let gain = 2f32.powf(-LensWarp::sample(&warp.falloff, t) * falloff);
                for c in &mut rgb {
                    *c *= gain;
                }
            }
            for k2 in 0..3 {
                row[i + k2] = rgb[k2].clamp(0.0, 255.0).round() as u8;
            }
            row[i + 3] = 255;
        }
    });
    buf.copy_from_slice(&out);
}

fn sample_bilinear16(src: &[u16], w: usize, h: usize, x: f32, y: f32) -> [f32; 3] {
    let fx = x - 0.5;
    let fy = y - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = fx - x0;
    let ty = fy - y0;
    let x0 = x0 as i32;
    let y0 = y0 as i32;
    let mut acc = [0f32; 3];
    for (dy, wy) in [(0, 1.0 - ty), (1, ty)] {
        for (dx, wx) in [(0, 1.0 - tx), (1, tx)] {
            let sx = clampi(x0 + dx, 0, w as i32 - 1) as usize;
            let sy = clampi(y0 + dy, 0, h as i32 - 1) as usize;
            let i = (sy * w + sx) * 4;
            let weight = wx * wy;
            for k in 0..3 {
                acc[k] += src[i + k] as f32 * weight;
            }
        }
    }
    acc
}

/// 16-bit twin of [`stage_lens_profile`] for the wide export buffer. Applying
/// the warp only to the 8-bit preview and merging its delta would replace
/// 16-bit detail with 8-bit quantisation noise wherever the warp moves pixels.
pub(crate) fn stage_lens_profile16(
    buf: &mut [u16],
    w: usize,
    h: usize,
    adj: &LensCorrection,
    warp: &crate::mock::lens_profile::LensWarp,
    distortion: f32,
    falloff: f32,
    ca: bool,
) {
    use crate::mock::lens_profile::LensWarp;

    let ca = ca && adj.ca_fringing.is_empty();
    let src = buf.to_vec();
    let cx = w as f32 * 0.5;
    let cy = h as f32 * 0.5;
    let half_diag = ((w * w + h * h) as f32).sqrt() * 0.5;
    buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = x * 4;
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let r_px = (dx * dx + dy * dy).sqrt();
            let t = r_px / half_diag;
            let t_src = t - LensWarp::sample_curve(&warp.distortion, t) * distortion;
            let k = if r_px > 1e-4 { t_src / t } else { 1.0 };
            let sx = cx + dx * k;
            let sy = cy + dy * k;
            let outside = sx < 0.0 || sy < 0.0 || sx > w as f32 || sy > h as f32;
            if outside && adj.hide_distorted_areas {
                row[i] = 0;
                row[i + 1] = 0;
                row[i + 2] = 0;
                row[i + 3] = 65535;
                continue;
            }
            let mut rgb = sample_bilinear16(&src, w, h, sx, sy);
            if ca && r_px > 1e-4 {
                let red_points = if adj.ca_red_points.is_empty() { &warp.red } else { &adj.ca_red_points };
                let blue_points = if adj.ca_blue_points.is_empty() { &warp.blue } else { &adj.ca_blue_points };
                let offset = |shift: &[(f32, f32)]| -> f32 { k + LensWarp::sample(shift, t) / t };
                let red_offset = offset(red_points);
                let blue_offset = offset(blue_points);
                if (red_offset - k).abs() > 1e-5 {
                    rgb[0] = sample_bilinear16(
                        &src, w, h, cx + dx * red_offset, cy + dy * red_offset,
                    )[0];
                }
                if (blue_offset - k).abs() > 1e-5 {
                    rgb[2] = sample_bilinear16(
                        &src, w, h, cx + dx * blue_offset, cy + dy * blue_offset,
                    )[2];
                }
            }
            if falloff > 0.0 {
                let gain = 2f32.powf(-LensWarp::sample(&warp.falloff, t) * falloff);
                for c in &mut rgb {
                    *c *= gain;
                }
            }
            for k2 in 0..3 {
                row[i + k2] = rgb[k2].clamp(0.0, 65535.0).round() as u16;
            }
            row[i + 3] = 65535;
        }
    });
}

/// Lateral chromatic aberration from the profile's six curves.
///
/// C1's recovered kernel (`p_041444160.cl`) resamples red and blue along the
/// radial direction from the optic centre and leaves green alone; its two
/// tables are the same shift-vs-radius pair `ZLENSDEFAULTCHROMABR` stores.
/// `ZLENSCHROMAFRINGING` splits that into shift / `ra` / strength per channel.
/// The roles of `ra` (dimensionless radial gain) and `strength` (displacement
/// scale) follow the field names only: no kernel consuming all six curves was
/// recovered, so those two are grade C. `strength == 0` resamples at the same
/// position and leaves the frame byte-identical.
pub(crate) fn stage_ca_fringes(buf: &mut [u8], w: usize, h: usize, adj: &LensCorrection) {
    use crate::mock::lens_profile::LensWarp;

    if !adj.chromatic_aberration || adj.ca_fringing.is_empty() || w == 0 || h == 0 {
        return;
    }
    let half_diag = ((w * w + h * h) as f32).sqrt() * 0.5;
    if half_diag <= 0.0 {
        return;
    }
    // The six tables share one radius axis; normalize x and the shift value by
    // its outer radius, which leaves the shift as a fraction of the radius
    // (the stored pairs are µm against µm). `ra` and `strength` stay raw.
    let mut outer = 0.0f32;
    for curve in [
        &adj.ca_fringing.red_shift,
        &adj.ca_fringing.red_ra,
        &adj.ca_fringing.red_strength,
        &adj.ca_fringing.blue_shift,
        &adj.ca_fringing.blue_ra,
        &adj.ca_fringing.blue_strength,
    ] {
        for &(x, _) in curve {
            if x.is_finite() {
                outer = outer.max(x);
            }
        }
    }
    if outer <= 0.0 {
        return;
    }
    let shift_curve = |curve: &[(f32, f32)]| -> Vec<(f32, f32)> {
        curve
            .iter()
            .filter(|(x, y)| x.is_finite() && y.is_finite())
            .map(|&(x, y)| (x / outer, y / outer))
            .collect()
    };
    let red_shift = shift_curve(&adj.ca_fringing.red_shift);
    let blue_shift = shift_curve(&adj.ca_fringing.blue_shift);
    let sample = |curve: &[(f32, f32)], t: f32| -> f32 {
        if curve.is_empty() {
            0.0
        } else {
            LensWarp::sample_curve(curve, t)
        }
    };
    let weight = |curve: &[(f32, f32)], t: f32| -> f32 {
        if curve.is_empty() {
            1.0
        } else {
            sample(curve, t).clamp(0.0, 1.0)
        }
    };

    let src = buf.to_vec();
    let cx = w as f32 * 0.5;
    let cy = h as f32 * 0.5;
    buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = x * 4;
            let dx = x as f32 + 0.5 - cx;
            let dy = y as f32 + 0.5 - cy;
            let r = (dx * dx + dy * dy).sqrt();
            if r <= 1e-4 {
                continue;
            }
            let t = (r / half_diag).clamp(0.0, 1.0);
            let (ux, uy) = (dx / r, dy / r);
            // shift carries the radial displacement, `ra` the dimensionless
            // radial gain and `strength` scales the move.
            let red = weight(&adj.ca_fringing.red_strength, t)
                * (sample(&red_shift, t) + sample(&adj.ca_fringing.red_ra, t))
                * r;
            let blue = weight(&adj.ca_fringing.blue_strength, t)
                * (sample(&blue_shift, t) + sample(&adj.ca_fringing.blue_ra, t))
                * r;
            if red.abs() <= 1e-6 && blue.abs() <= 1e-6 {
                continue;
            }
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;
            if red.abs() > 1e-6 {
                row[i] = sample_bilinear(&src, w, h, px + ux * red, py + uy * red)[0]
                    .clamp(0.0, 255.0)
                    .round() as u8;
            }
            if blue.abs() > 1e-6 {
                row[i + 2] = sample_bilinear(&src, w, h, px + ux * blue, py + uy * blue)[2]
                    .clamp(0.0, 255.0)
                    .round() as u8;
            }
        }
    });
}

/// Desaturates strongly purple pixels that sit on an edge, which is what a
/// lateral chromatic fringe looks like. Neutral or non-purple pixels are kept.
fn stage_purple_fringing(buf: &mut [u8], w: usize, h: usize, purple: f32) {
    // purple is the 0..1 normalised amount; 0.5 is C1's zero.
    let strength = crate::mock::purple_strength(purple);
    if strength == 0.0 || w == 0 || h == 0 || buf.len() != w * h * 4 {
        return;
    }
    let row_bytes = w * 4;
    let mut source_band = Vec::new();
    let mut previous = vec![0u8; row_bytes];
    let mut y0 = 0usize;
    while y0 < h {
        let y1 = (y0 + PURPLE_BAND_ROWS).min(h);
        let source_y0 = y0.saturating_sub(1);
        let source_y1 = (y1 + 1).min(h);
        let source_rows = source_y1 - source_y0;
        source_band.resize(source_rows * row_bytes, 0);

        for gy in source_y0..source_y1 {
            let dst = (gy - source_y0) * row_bytes;
            if gy < y0 {
                source_band[dst..dst + row_bytes].copy_from_slice(&previous);
            } else {
                let src = gy * row_bytes;
                source_band[dst..dst + row_bytes].copy_from_slice(&buf[src..src + row_bytes]);
            }
        }
        let last = (y1 - 1 - source_y0) * row_bytes;
        previous.copy_from_slice(&source_band[last..last + row_bytes]);

        let dst_start = y0 * row_bytes;
        let dst_end = y1 * row_bytes;
        buf[dst_start..dst_end]
            .par_chunks_mut(row_bytes)
            .enumerate()
            .for_each(|(band_y, row)| {
                let y = y0 + band_y;
                for x in 0..w {
                    let i = x * 4;
                    let j = (y - source_y0) * row_bytes + x * 4;
                    let p = [source_band[j], source_band[j + 1], source_band[j + 2]];
                    let r = p[0] as f32;
                    let g = p[1] as f32;
                    let b = p[2] as f32;
                    let pn = ((r.min(b) - g) / 255.0).clamp(0.0, 1.0);
                    if pn <= 0.0 {
                        continue;
                    }
                    let mut lo = 255.0f32;
                    let mut hi = 0.0f32;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let xx = clampi(x as i32 + dx, 0, w as i32 - 1) as usize;
                            let yy = clampi(y as i32 + dy, 0, h as i32 - 1) as usize;
                            let n = (yy - source_y0) * row_bytes + xx * 4;
                            let l = luma8([
                                source_band[n],
                                source_band[n + 1],
                                source_band[n + 2],
                            ]);
                            lo = lo.min(l);
                            hi = hi.max(l);
                        }
                    }
                    let edge = ((hi - lo) / 40.0).clamp(0.0, 1.0);
                    let amount = strength * pn * edge;
                    if amount == 0.0 {
                        continue;
                    }
                    let gray = luma8(p);
                    for k in 0..3 {
                        let v = p[k] as f32 * (1.0 - amount) + gray * amount;
                        row[i + k] = v.clamp(0.0, 255.0).round() as u8;
                    }
                }
            });
        y0 = y1;
    }
}

// ───────────────────────────── 6. film grain ─────────────────────────────

/// Deterministic hash noise (no RNG state, so previews are reproducible).
/// `granularity` picks the size of the noise cell, `kind` shapes the
/// distribution and how it is weighted.
fn hash_noise(x: u32, y: u32, seed: u32) -> f32 {
    let mut h = x.wrapping_mul(0x9E37_79B1) ^ y.wrapping_mul(0x85EB_CA77) ^ seed.wrapping_mul(0xC2B2_AE3D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2545_F491);
    h ^= h >> 13;
    (h & 0x00FF_FFFF) as f32 / 0x00FF_FFFF as f32 * 2.0 - 1.0
}

// ───────────────────────────── tests ─────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RadialBlur;

    fn detail_plan(adj: &crate::AdjustmentSet, gpu: bool) -> DetailStagePlan {
        detail_stage_plan(adj, gpu)
    }

    #[test]
    fn detail_plan_keeps_supported_stages_on_gpu() {
        let adj = crate::AdjustmentSet {
            noise_reduction: NoiseReduction {
                luminance: 60.0,
                details: 35.0,
                color: 0.0,
                single_pixel: 0.0,
            },
            sharpening: Sharpening {
                amount: 180.0,
                radius: 0.8,
                threshold: 1.0,
                halo_suppression: 20.0,
            },
            moire: Moire { amount: 60.0, pattern: 50.0 },
            lens: LensCorrection { distortion: 20.0, light_falloff: 30.0, ..Default::default() },
            ..Default::default()
        };
        assert_eq!(
            detail_plan(&adj, true),
            DetailStagePlan {
                gpu: true,
                cpu_single_pixel_first: false,
                cpu_color_nr: false,
                gpu_sharpen: true,
                gpu_lens: true,
                cpu_moire: false,
                cpu_diffraction: false,
                cpu_sharpen: false,
                cpu_lens: false,
            }
        );
    }

    #[test]
    fn lens_profile_moves_only_lens_to_cpu() {
        let adj = crate::AdjustmentSet {
            noise_reduction: NoiseReduction { color: 0.0, ..Default::default() },
            lens: LensCorrection { use_profile: true, distortion: 100.0, ..Default::default() },
            ..Default::default()
        };
        let plan = detail_plan(&adj, true);
        assert!(plan.gpu_sharpen);
        assert!(!plan.gpu_lens);
        assert!(plan.cpu_lens);
        assert!(!plan.cpu_sharpen);
        assert!(!plan.cpu_diffraction);
    }

    #[test]
    fn partial_route_keeps_moire_on_cpu() {
        let adj = crate::AdjustmentSet {
            moire: Moire { amount: 60.0, pattern: 50.0 },
            lens: LensCorrection { use_profile: true, distortion: 100.0, ..Default::default() },
            ..Default::default()
        };
        let plan = detail_plan(&adj, true);
        assert!(plan.cpu_moire);
        assert!(!plan.gpu_sharpen);
        assert!(plan.cpu_sharpen);
        assert!(plan.cpu_lens);
    }

    #[test]
    fn lens_extras_move_only_lens_to_cpu() {
        let adj = crate::AdjustmentSet {
            noise_reduction: NoiseReduction { color: 0.0, ..Default::default() },
            sharpening: crate::Sharpening { amount: 50.0, ..Default::default() },
            lens: LensCorrection {
                uniform_light_enabled: true,
                uniform_light: 50.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let plan = detail_plan(&adj, true);
        assert!(plan.gpu_sharpen);
        assert!(!plan.gpu_lens);
        assert!(plan.cpu_lens);
        assert!(!plan.cpu_sharpen);
    }

    #[test]
    fn single_pixel_runs_cpu_first_and_keeps_rest_on_gpu() {
        let adj = crate::AdjustmentSet {
            noise_reduction: NoiseReduction {
                luminance: 60.0,
                details: 35.0,
                color: 0.0,
                single_pixel: 100.0,
            },
            ..Default::default()
        };
        let plan = detail_plan(&adj, true);
        assert!(plan.cpu_single_pixel_first);
        assert!(plan.gpu_sharpen);
        assert!(plan.gpu_lens);
        assert!(!plan.cpu_lens);
    }

    #[test]
    fn color_nr_runs_c1_2012_on_cpu_before_sharpening() {
        let adj = crate::AdjustmentSet {
            noise_reduction: NoiseReduction {
                luminance: 50.0,
                details: 50.0,
                color: 50.0,
                single_pixel: 0.0,
            },
            sharpening: Sharpening { amount: 180.0, ..Default::default() },
            ..Default::default()
        };
        let plan = detail_plan(&adj, true);
        assert!(plan.cpu_color_nr);
        assert!(!plan.gpu_sharpen);
        assert!(plan.cpu_sharpen);
        assert!(plan.cpu_lens);
    }

    #[test]
    fn diffraction_keeps_nr_and_moire_on_gpu_but_sharpens_on_cpu() {
        let adj = crate::AdjustmentSet {
            noise_reduction: NoiseReduction { color: 0.0, ..Default::default() },
            lens: LensCorrection {
                diffraction_enabled: true,
                aperture: 8.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let plan = detail_plan(&adj, true);
        assert!(!plan.gpu_sharpen);
        assert!(!plan.gpu_lens);
        assert!(plan.cpu_diffraction);
        assert!(plan.cpu_sharpen);
        assert!(plan.cpu_lens);
    }

    #[test]
    fn gpu_disabled_keeps_the_full_cpu_route() {
        let adj = crate::AdjustmentSet {
            noise_reduction: NoiseReduction {
                luminance: 60.0,
                details: 35.0,
                color: 40.0,
                single_pixel: 100.0,
            },
            lens: LensCorrection {
                use_profile: true,
                diffraction_enabled: true,
                ..Default::default()
            },
            ..Default::default()
        };
        assert_eq!(
            detail_plan(&adj, false),
            DetailStagePlan {
                gpu: false,
                cpu_single_pixel_first: false,
                cpu_color_nr: false,
                gpu_sharpen: false,
                gpu_lens: false,
                cpu_moire: false,
                cpu_diffraction: false,
                cpu_sharpen: false,
                cpu_lens: false,
            }
        );
    }

    fn cpu_detail_reference(
        rgba: &mut [u8],
        w: usize,
        h: usize,
        adj: &crate::AdjustmentSet,
        scale: f32,
        warp: Option<&crate::mock::lens_profile::LensWarp>,
    ) {
        stage_single_pixel(rgba, w, h, &adj.noise_reduction);
        stage_luminance_nr(rgba, w, h, &adj.noise_reduction);
        stage_color_nr(rgba, w, h, &adj.noise_reduction, scale);
        stage_moire(rgba, w, h, &adj.moire);
        stage_diffraction(rgba, w, h, &adj.lens, scale);
        stage_sharpening(rgba, w, h, &adj.sharpening, scale);
        stage_lens(rgba, w, h, &adj.lens, warp);
        stage_dust_removal(rgba, w, h, &adj.lens);
        stage_movement(rgba, w, h, &adj.lens);
        stage_sharpness_falloff(rgba, w, h, &adj.lens);
        crate::mock::grain::stage(rgba, w, h, &adj.film_grain, scale);
    }

    #[test]
    fn gpu_disabled_matches_the_full_cpu_reference() {
        let (w, h) = (63usize, 47usize);
        let base: Vec<u8> = (0..w * h * 4).map(|i| ((i * 73 + i / 7) % 256) as u8).collect();
        let adj = crate::AdjustmentSet {
            noise_reduction: NoiseReduction {
                luminance: 60.0,
                details: 35.0,
                color: 40.0,
                single_pixel: 100.0,
            },
            sharpening: Sharpening {
                amount: 180.0,
                radius: 0.8,
                threshold: 1.0,
                halo_suppression: 20.0,
            },
            moire: Moire { amount: 60.0, pattern: 50.0 },
            lens: LensCorrection {
                use_profile: true,
                distortion: 100.0,
                uniform_light_enabled: true,
                uniform_light: 50.0,
                diffraction_enabled: true,
                aperture: 8.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut actual = base.clone();
        post_process_impl(&mut actual, w as u32, h as u32, &adj, 1.0, None, false);
        let mut expected = base.clone();
        cpu_detail_reference(&mut expected, w, h, &adj, 1.0, None);
        assert_eq!(actual, expected);
    }

    #[cfg(windows)]
    #[test]
    fn split_gpu_path_matches_cpu_for_profile_single_pixel_and_diffraction() {
        if std::env::var_os("AURORA_DETAIL_GPU").is_none() {
            eprintln!("skipping: set AURORA_DETAIL_GPU=1 to compare CUDA stages");
            return;
        }
        if !detail_gpu::prewarm() {
            eprintln!("skipping: CUDA detail device unavailable");
            return;
        }
        let (w, h) = (63usize, 47usize);
        let base: Vec<u8> = (0..w * h * 4).map(|i| ((i * 73 + i / 7) % 256) as u8).collect();
        let profile = crate::mock::lens_profile::LensWarp {
            distortion: vec![(0.0, 0.0), (1.0, 0.02)],
            ..Default::default()
        };
        let scenarios = [
            (
                "lens profile",
                crate::AdjustmentSet {
                    moire: Moire { amount: 60.0, pattern: 50.0 },
                    lens: LensCorrection {
                        use_profile: true,
                        distortion: 100.0,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                Some(profile.clone()),
            ),
            (
                "single pixel",
                crate::AdjustmentSet {
                    noise_reduction: NoiseReduction {
                        luminance: 60.0,
                        details: 35.0,
                        color: 40.0,
                        single_pixel: 100.0,
                    },
                    sharpening: Sharpening {
                        amount: 180.0,
                        radius: 0.8,
                        threshold: 1.0,
                        halo_suppression: 20.0,
                    },
                    ..Default::default()
                },
                None,
            ),
            (
                "diffraction",
                crate::AdjustmentSet {
                    lens: LensCorrection {
                        diffraction_enabled: true,
                        aperture: 8.0,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            ),
            (
                "lens extras",
                crate::AdjustmentSet {
                    lens: LensCorrection {
                        uniform_light_enabled: true,
                        uniform_light: 50.0,
                        ..Default::default()
                    },
                    ..Default::default()
                },
                None,
            ),
        ];
        for (label, adj, warp) in scenarios {
            let mut cpu = base.clone();
            post_process_impl(&mut cpu, w as u32, h as u32, &adj, 1.0, warp.as_ref(), false);
            let mut gpu = base.clone();
            post_process_impl(&mut gpu, w as u32, h as u32, &adj, 1.0, warp.as_ref(), true);
            assert_eq!(gpu, cpu, "{label} must match the CPU oracle");
        }
    }

    // Fingerprints from the original stages, including clamped edges and alpha.
    #[test]
    fn detail_stages_match_reference_pixels() {
        let expected = [
            15226162907845603633, 15226162907845603633, 15226162907845603633,
            778449501603359228, 14188287539090345076, 7733125734113225351,
            7758119668271440554, 17275283751766918384, 15087915455906674826,
            17729869383540181920, 18074643075832604112, 1221425972588756570,
        ];
        let mut actual = Vec::new();
        for (w, h) in [(1, 1), (1, 17), (19, 1), (37, 23)] {
            let base: Vec<u8> = (0..w * h * 4).map(|i| ((i * 73 + i / 7) % 256) as u8).collect();
            for stage in 0..3 {
                let mut pixels = base.clone();
                match stage {
                    0 => stage_luminance_nr(&mut pixels, w, h, &NoiseReduction {
                        luminance: 73.0, details: 41.0, color: 0.0, single_pixel: 0.0,
                    }),
                    1 => stage_color_nr(&mut pixels, w, h, &NoiseReduction {
                        luminance: 0.0, details: 0.0, color: 100.0, single_pixel: 0.0,
                    }, 1.0),
                    _ => stage_sharpening(&mut pixels, w, h, &Sharpening {
                        amount: 245.0, radius: 2.8, threshold: 1.3, halo_suppression: 37.0,
                    }, 1.5),
                }
                let hash = pixels.iter().fold(0xcbf29ce484222325u64, |hash, &b| {
                    (hash ^ b as u64).wrapping_mul(0x100000001b3)
                });
                actual.push(hash);
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn single_pixel_and_moire_match_reference_pixels() {
        let mut single = Vec::new();
        let mut moire = Vec::new();
        for (w, h) in [(1, 1), (1, 17), (19, 1), (37, 23)] {
            let base: Vec<u8> = (0..w * h * 4).map(|i| ((i * 73 + i / 7) % 256) as u8).collect();
            let mut pixels = base.clone();
            stage_single_pixel(
                &mut pixels,
                w,
                h,
                &NoiseReduction {
                    luminance: 0.0,
                    details: 0.0,
                    color: 0.0,
                    single_pixel: 63.0,
                },
            );
            single.push(pixels.iter().fold(0xcbf29ce484222325u64, |hash, &b| {
                (hash ^ b as u64).wrapping_mul(0x100000001b3)
            }));

            let mut pixels = base;
            stage_moire(&mut pixels, w, h, &Moire { amount: 100.0, pattern: 8.0 });
            moire.push(pixels.iter().fold(0xcbf29ce484222325u64, |hash, &b| {
                (hash ^ b as u64).wrapping_mul(0x100000001b3)
            }));
        }
        assert_eq!(
            single,
            [15226162907845603633, 18259893255946748908, 4115099798523760392, 4669545325169028274]
        );
        assert_eq!(
            moire,
            [15226162907845603633, 14188287539090345076, 17275283751766918384, 18074643075832604112]
        );
    }

    #[test]
    fn fused_single_pixel_and_luminance_match_separate_stages() {
        let (w, h) = (37usize, 23usize);
        let base: Vec<u8> = (0..w * h * 4).map(|i| ((i * 73 + i / 7) % 256) as u8).collect();
        let mut adj = off();
        adj.noise_reduction.single_pixel = 63.0;
        adj.noise_reduction.luminance = 73.0;
        adj.noise_reduction.details = 41.0;

        let mut expected = base.clone();
        stage_single_pixel(&mut expected, w, h, &adj.noise_reduction);
        stage_luminance_nr(&mut expected, w, h, &adj.noise_reduction);

        let mut actual = base;
        post_process(&mut actual, w as u32, h as u32, &adj, 1.0);
        assert_eq!(actual, expected);
    }

    #[test]
    fn vertical_blur_preserves_sum_order_at_edges() {
        for (w, h) in [(1, 1), (1, 17), (19, 1), (37, 23)] {
            let src: Vec<f32> = (0..w * h).map(|i| (i as f32 * 1.37).sin() * 255.0).collect();
            for r in [0, 1, 3, 23] {
                let mut out = vec![0.0; src.len()];
                box_blur_v(&src, &mut out, w, h, r);
                for y in 0..h {
                    for x in 0..w {
                        let mut sum = 0.0f32;
                        for k in -r..=r {
                            let yy = (y as i32 + k).clamp(0, h as i32 - 1) as usize;
                            sum += src[yy * w + x];
                        }
                        assert_eq!(out[y * w + x].to_bits(), (sum * (1.0 / (2 * r + 1) as f32)).to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn padded_luma_matches_the_original_plane_and_border_bits() {
        for (w, h) in [(1, 1), (1, 17), (19, 1), (37, 23)] {
            let buf: Vec<u8> =
                (0..w * h * 4).map(|i| ((i * 73 + i / 7) % 256) as u8).collect();
            let luma = luma_plane(&buf, w, h);
            let pad = padded_luma(&buf, w, h);
            let pw = w + 2;
            for y in 0..h + 2 {
                let sy = y.saturating_sub(1).min(h - 1);
                assert_eq!(pad[y * pw].to_bits(), luma[sy * w].to_bits());
                assert_eq!(
                    pad[y * pw + pw - 1].to_bits(),
                    luma[sy * w + w - 1].to_bits()
                );
                for x in 0..w {
                    assert_eq!(
                        pad[y * pw + x + 1].to_bits(),
                        luma[sy * w + x].to_bits()
                    );
                }
            }
        }
    }

    /// The CUDA detail path must reproduce the CPU stages byte for byte.
    /// Skipped unless `AURORA_DETAIL_GPU=1` requests a hardware comparison.
    #[cfg(windows)]
    #[test]
    fn cuda_detail_matches_the_cpu_stages() {
        if !std::env::var_os("AURORA_DETAIL_GPU").is_some_and(|value| value != "0") {
            eprintln!("skipping: AURORA_DETAIL_GPU is not set");
            return;
        }
        let (w, h) = (63usize, 47usize);
        let base: Vec<u8> = (0..w * h * 4).map(|i| ((i * 73 + i / 7) % 256) as u8).collect();
        let adj = crate::AdjustmentSet {
            noise_reduction: NoiseReduction {
                luminance: 60.0,
                details: 35.0,
                color: 40.0,
                single_pixel: 0.0,
            },
            sharpening: Sharpening {
                amount: 180.0,
                radius: 0.8,
                threshold: 1.0,
                halo_suppression: 20.0,
            },
            // The CUDA moire kernel is a deliberate approximation; the
            // per-stage probe in the split test records that difference.
            moire: Moire::default(),
            ..Default::default()
        };
        let mut cpu = base.clone();
        let saved = std::env::var_os("AURORA_DETAIL_GPU");
        unsafe { std::env::set_var("AURORA_DETAIL_GPU", "0") };
        post_process(&mut cpu, w as u32, h as u32, &adj, 1.0);
        unsafe { std::env::set_var("AURORA_DETAIL_GPU", "1") };
        let mut gpu = base.clone();
        post_process(&mut gpu, w as u32, h as u32, &adj, 1.0);
        assert_eq!(gpu, cpu, "CUDA detail must match the CPU stages");
        assert!(detail_gpu::post_process(
            &mut gpu,
            w,
            h,
            &adj.noise_reduction,
            &adj.moire,
            &adj.sharpening,
            &adj.lens,
            1.0
        ));
        unsafe {
            match saved {
                Some(value) => std::env::set_var("AURORA_DETAIL_GPU", value),
                None => std::env::remove_var("AURORA_DETAIL_GPU"),
            }
        }
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "needs a real NEF and an NVIDIA CUDA device"]
    fn cuda_detail_matches_real_nef_and_benchmarks() {
        use std::path::Path;
        use std::time::Instant;

        let path = std::env::var("AURORA_DETAIL_BENCH_NEF")
            .unwrap_or_else(|_| r"C:\Users\Jxcz2\Desktop\_DSC5673.NEF".to_owned());
        let image = crate::raw::decode(Path::new(&path)).expect("decode NEF");
        assert_eq!(
            crate::raw::from_sensor(Path::new(&path)),
            Some(true),
            "libraw fell back to the embedded preview"
        );
        let (w, h) = image.dimensions();
        let original = image.into_raw();
        let mut adj = crate::AdjustmentSet::default();
        adj.noise_reduction = NoiseReduction {
            luminance: 50.0,
            details: 50.0,
            color: 50.0,
            single_pixel: 0.0,
        };
        adj.moire = Moire {
            amount: 50.0,
            pattern: 50.0,
        };
        adj.lens = LensCorrection {
            distortion: 30.0,
            light_falloff: 30.0,
            chromatic_aberration: true,
            purple_fringing: 40.0,
            hide_distorted_areas: false,
            ..Default::default()
        };
        adj.sharpening = Sharpening {
            amount: 160.0,
            radius: 0.8,
            threshold: 1.0,
            halo_suppression: 0.0,
        };

        let saved = std::env::var_os("AURORA_DETAIL_GPU");
        unsafe { std::env::set_var("AURORA_DETAIL_GPU", "0") };
        let mut cpu = original.clone();
        post_process(&mut cpu, w, h, &adj, 1.0);

        let gpu = detail_gpu::open_for_bench().expect("CUDA context");
        let pixels = w as usize * h as usize;
        let pool = gpu.alloc_buffer(pixels * 4 * 6).expect("device pool");
        let lut = gpu.alloc_buffer(1025 * 4).expect("device lut");
        let run_gpu_adj = |buf: &mut [u8], case: &crate::AdjustmentSet| {
            detail_gpu::post_process_pool(
                &gpu,
                buf,
                w as usize,
                h as usize,
                &case.noise_reduction,
                &case.moire,
                &case.sharpening,
                &case.lens,
                1.0,
                &pool,
                Some(&lut),
            )
            .expect("CUDA detail launch");
        };
        let run_gpu = |buf: &mut [u8]| run_gpu_adj(buf, &adj);
        let mut gpu_frame = original.clone();
        run_gpu(&mut gpu_frame);

        let (mut max_diff, mut over1, mut diff_px) = (0i32, 0usize, 0usize);
        for (a, b) in cpu.chunks_exact(4).zip(gpu_frame.chunks_exact(4)) {
            let mut px_diff = false;
            for k in 0..3 {
                let d = (a[k] as i32 - b[k] as i32).abs();
                max_diff = max_diff.max(d);
                over1 += usize::from(d > 1);
                px_diff |= d > 0;
            }
            diff_px += usize::from(px_diff);
        }
        if std::env::var_os("AURORA_DETAIL_STAGE_PARITY").is_some() {
            let mut stage_base = crate::AdjustmentSet::default();
            stage_base.noise_reduction = NoiseReduction {
                luminance: 0.0,
                details: 0.0,
                color: 0.0,
                single_pixel: 0.0,
            };
            stage_base.moire.amount = 0.0;
            stage_base.sharpening.amount = 0.0;
            stage_base.lens = LensCorrection {
                hide_distorted_areas: false,
                ..Default::default()
            };
            let stages = [
                ("luminance", NoiseReduction { luminance: 50.0, details: 50.0, color: 0.0, single_pixel: 0.0 }, Moire::default(), Sharpening { amount: 0.0, ..Default::default() }, stage_base.lens.clone()),
                ("color", NoiseReduction { luminance: 0.0, details: 0.0, color: 50.0, single_pixel: 0.0 }, Moire::default(), Sharpening { amount: 0.0, ..Default::default() }, stage_base.lens.clone()),
                ("moire", stage_base.noise_reduction.clone(), Moire { amount: 50.0, pattern: 50.0 }, Sharpening { amount: 0.0, ..Default::default() }, stage_base.lens.clone()),
                ("sharpening", stage_base.noise_reduction.clone(), Moire::default(), adj.sharpening.clone(), stage_base.lens.clone()),
                ("lens", stage_base.noise_reduction.clone(), Moire::default(), Sharpening { amount: 0.0, ..Default::default() }, adj.lens.clone()),
                ("lens_distortion", stage_base.noise_reduction.clone(), Moire::default(), Sharpening { amount: 0.0, ..Default::default() }, LensCorrection { distortion: adj.lens.distortion, ..stage_base.lens.clone() }),
                ("lens_falloff", stage_base.noise_reduction.clone(), Moire::default(), Sharpening { amount: 0.0, ..Default::default() }, LensCorrection { light_falloff: adj.lens.light_falloff, ..stage_base.lens.clone() }),
                ("lens_ca", stage_base.noise_reduction.clone(), Moire::default(), Sharpening { amount: 0.0, ..Default::default() }, LensCorrection { chromatic_aberration: true, ..stage_base.lens.clone() }),
                ("lens_purple", stage_base.noise_reduction.clone(), Moire::default(), Sharpening { amount: 0.0, ..Default::default() }, LensCorrection { purple_fringing: adj.lens.purple_fringing, ..stage_base.lens.clone() }),
            ];
            for (label, nr, moire, sharpening, lens) in stages {
                let mut case = stage_base.clone();
                case.noise_reduction = nr;
                case.moire = moire;
                case.sharpening = sharpening;
                case.lens = lens;
                let mut stage_cpu = original.clone();
                unsafe { std::env::set_var("AURORA_DETAIL_GPU", "0") };
                post_process(&mut stage_cpu, w, h, &case, 1.0);
                let mut stage_gpu = original.clone();
                run_gpu_adj(&mut stage_gpu, &case);
                let (mut stage_max, mut stage_over1, mut stage_px) = (0i32, 0usize, 0usize);
                for (a, b) in stage_cpu.chunks_exact(4).zip(stage_gpu.chunks_exact(4)) {
                    let mut different = false;
                    for k in 0..3 {
                        let delta = (a[k] as i32 - b[k] as i32).abs();
                        stage_max = stage_max.max(delta);
                        stage_over1 += usize::from(delta > 1);
                        different |= delta != 0;
                    }
                    stage_px += usize::from(different);
                }
                println!(
                    "DETAIL_STAGE {label} max_diff={stage_max} over1={stage_over1} diff_px={stage_px}"
                );
            }
            unsafe { std::env::set_var("AURORA_DETAIL_GPU", "1") };
        }
        if let Some(dir) = std::env::var_os("AURORA_DETAIL_PARITY_DIR") {
            let dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&dir).expect("create parity directory");
            std::fs::write(dir.join("cpu.rgba"), &cpu).expect("write CPU frame");
            std::fs::write(dir.join("gpu.rgba"), &gpu_frame).expect("write GPU frame");
        }

        let mut cpu_ms = Vec::new();
        let mut gpu_pool_ms = Vec::new();
        let mut gpu_alloc_ms = Vec::new();
        for _ in 0..3 {
            let start = Instant::now();
            let mut frame = original.clone();
            unsafe { std::env::set_var("AURORA_DETAIL_GPU", "0") };
            post_process(&mut frame, w, h, &adj, 1.0);
            cpu_ms.push(start.elapsed().as_secs_f64() * 1000.0);

            let start = Instant::now();
            let mut frame = original.clone();
            run_gpu(&mut frame);
            gpu_pool_ms.push(start.elapsed().as_secs_f64() * 1000.0);

            let start = Instant::now();
            let mut frame = original.clone();
            detail_gpu::post_process_with(
                &gpu,
                &mut frame,
                w as usize,
                h as usize,
                &adj.noise_reduction,
                &adj.moire,
                &adj.sharpening,
                &adj.lens,
                1.0,
            )
            .expect("CUDA detail allocate-and-run");
            gpu_alloc_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        cpu_ms.sort_by(f64::total_cmp);
        gpu_pool_ms.sort_by(f64::total_cmp);
        gpu_alloc_ms.sort_by(f64::total_cmp);
        println!(
            "DETAIL_REAL nef={path} w={w} h={h} cpu_p50_ms={:.1} cpu_p95_ms={:.1} \
gpu_pool_p50_ms={:.1} gpu_pool_p95_ms={:.1} gpu_alloc_p50_ms={:.1} \
gpu_alloc_p95_ms={:.1} speedup_pool_p50={:.2} speedup_alloc_p50={:.2} max_diff={max_diff} \
over1={over1} diff_px={diff_px}",
            cpu_ms[1],
            cpu_ms[2],
            gpu_pool_ms[1],
            gpu_pool_ms[2],
            gpu_alloc_ms[1],
            gpu_alloc_ms[2],
            cpu_ms[1] / gpu_pool_ms[1],
            cpu_ms[1] / gpu_alloc_ms[1],
        );
        assert_eq!((max_diff, over1, diff_px), (0, 0, 0));

        unsafe {
            match saved {
                Some(value) => std::env::set_var("AURORA_DETAIL_GPU", value),
                None => std::env::remove_var("AURORA_DETAIL_GPU"),
            }
        }
    }

    /// Per-stage cost on a Draft-sized frame. `cargo test -p aurora-engine
    /// bench_detail_stages -- --ignored --nocapture`.
    #[test]
    #[ignore = "timing benchmark, run explicitly with --ignored"]
    fn bench_detail_stages() {
        use std::time::Instant;
        let (w, h) = (1400usize, 933usize);
        let base: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                [(x % 251) as u8, (y % 241) as u8, ((x * 7 + y * 13) % 239) as u8, 255]
            })
            .collect();
        let adj = crate::AdjustmentSet::default();
        let mut moire = adj.moire.clone();
        moire.amount = 60.0;
        moire.pattern = 50.0;
        let mut lens = adj.lens.clone();
        lens.distortion = 60.0;
        lens.light_falloff = 60.0;
        lens.chromatic_aberration = true;
        lens.purple_fringing = 40.0;
        lens.use_profile = false;
        let mut grain = adj.film_grain.clone();
        grain.impact = 60.0;
        let pixels = (w * h) as f64;
        let run = |label: &str, f: &dyn Fn(&mut Vec<u8>)| {
            let mut buf = base.clone();
            let t = Instant::now();
            f(&mut buf);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            println!("{label:24} {ms:8.2} ms   {:6.1} Mpx/s", pixels / ms / 1000.0);
        };
        run("luminance_nr", &|b| stage_luminance_nr(b, w, h, &adj.noise_reduction));
        run("color_nr", &|b| stage_color_nr(b, w, h, &adj.noise_reduction, 1.0));
        run("moire", &|b| stage_moire(b, w, h, &moire));
        run("sharpening", &|b| stage_sharpening(b, w, h, &adj.sharpening, 1.0));
        run("lens(generic)", &|b| stage_lens(b, w, h, &lens, None));
        run("grain", &|b| crate::mock::grain::stage(b, w, h, &grain, 1.0));
        run("post_process(all on)", &|b| post_process_with_lens(b, w as u32, h as u32, &adj, 1.0, None));
    }

    /// An adjustment set with every detail stage switched off.
    fn off() -> crate::AdjustmentSet {
        let mut adj = crate::AdjustmentSet::default();
        adj.noise_reduction = NoiseReduction {
            luminance: 0.0,
            details: 0.0,
            color: 0.0,
            single_pixel: 0.0,
        };
        adj.sharpening = Sharpening {
            amount: 0.0,
            radius: 0.8,
            threshold: 1.0,
            halo_suppression: 0.0,
        };
        adj.moire = Moire { amount: 0.0, pattern: 0.0 };
        adj.film_grain =
            FilmGrain { kind: GrainKind::FineGrain, impact: 0.0, granularity: 50.0, density: 0.0 };
        adj.lens = LensCorrection {
            hide_distorted_areas: false,
            ..Default::default()
        };
        adj
    }

    fn variance(px: &[u8]) -> f32 {
        let n = px.len() as f32;
        let mean = px.iter().map(|v| *v as f32).sum::<f32>() / n;
        px.iter().map(|v| (*v as f32 - mean).powi(2)).sum::<f32>() / n
    }

    fn stage_diffraction_full_snapshot_reference(
        buf: &mut [u8],
        w: usize,
        h: usize,
        adj: &LensCorrection,
        scale: f32,
    ) {
        if !adj.diffraction_enabled {
            return;
        }
        let f_number = adj.aperture;
        if !(1.0..=50.0).contains(&f_number) {
            return;
        }
        let pitch = pixel_pitch_um(adj, w, h, scale);
        if pitch <= 0.0 || !pitch.is_finite() {
            return;
        }
        let v_per_px = std::f32::consts::PI * pitch / (DIFFRACTION_LAMBDA_UM * f_number);
        if v_per_px <= 0.0 || !v_per_px.is_finite() {
            return;
        }
        let first_zero_px = 3.831_705_6 / v_per_px;
        if first_zero_px < 0.35 {
            return;
        }
        let radius = ((first_zero_px * 2.2).ceil() as i32).clamp(1, 6);
        let kernel = airy_kernel(radius, v_per_px);
        let n = (radius * 2 + 1) as usize;
        let src = buf.to_vec();
        buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let i = x * 4;
                let mut acc = [0f32; 3];
                for ky in 0..n {
                    let sy = clampi(y as i32 + ky as i32 - radius, 0, h as i32 - 1) as usize;
                    for kx in 0..n {
                        let sx = clampi(x as i32 + kx as i32 - radius, 0, w as i32 - 1) as usize;
                        let weight = kernel[ky * n + kx];
                        let j = (sy * w + sx) * 4;
                        acc[0] += src[j] as f32 * weight;
                        acc[1] += src[j + 1] as f32 * weight;
                        acc[2] += src[j + 2] as f32 * weight;
                    }
                }
                for c in 0..3 {
                    row[i + c] = acc[c].round().clamp(0.0, 255.0) as u8;
                }
            }
        });
    }

    #[test]
    fn diffraction_band_snapshot_matches_full_frame_reference() {
        let (w, h) = (137usize, 259usize);
        let mut base = Vec::with_capacity(w * h * 4);
        for i in 0..w * h {
            base.extend_from_slice(&[
                (i * 73 % 251) as u8,
                (i * 29 + i / 7) as u8,
                (i * 17 + i / 11) as u8,
                255,
            ]);
        }
        let lens = LensCorrection {
            diffraction_enabled: true,
            aperture: 16.0,
            ..Default::default()
        };
        let mut expected = base.clone();
        let mut actual = base.clone();
        stage_diffraction_full_snapshot_reference(&mut expected, w, h, &lens, 1.0);
        stage_diffraction(&mut actual, w, h, &lens, 1.0);
        assert_ne!(actual, base, "the parity fixture must exercise diffraction");
        assert_eq!(actual, expected);
    }

    #[test]
    fn diffraction_is_a_noop_when_disabled() {
        let (w, h) = (32usize, 32usize);
        let buf: Vec<u8> = (0..w * h)
            .flat_map(|i| [(i % 251) as u8, (i % 97) as u8, (i % 53) as u8, 255])
            .collect();
        let mut out = buf.clone();
        stage_diffraction(&mut out, w, h, &LensCorrection::default(), 1.0);
        assert_eq!(out, buf, "disabled diffraction must not touch a byte");
    }

    #[test]
    fn diffraction_spreads_a_point_and_keeps_the_energy() {
        let (w, h) = (48usize, 48usize);
        let mut buf = vec![0u8; w * h * 4];
        for px in buf.chunks_exact_mut(4) {
            px[3] = 255;
        }
        let center = ((h / 2) * w + w / 2) * 4;
        buf[center] = 255;
        buf[center + 1] = 255;
        buf[center + 2] = 255;
        let before: u64 = buf.chunks_exact(4).map(|px| px[0] as u64).sum();

        let lens = LensCorrection { diffraction_enabled: true, aperture: 16.0, ..Default::default() };
        stage_diffraction(&mut buf, w, h, &lens, 1.0);
        assert!(buf[center] < 255, "centre {}", buf[center]);
        assert!(
            buf[center + 4 * 2] > 0 || buf[center - 4 * 2] > 0 || buf[center + 2] > 0,
            "no light reached the neighbours"
        );
        let after: u64 = buf.chunks_exact(4).map(|px| px[0] as u64).sum();
        assert!(
            after as f32 >= before as f32 * 0.93,
            "energy {before} -> {after}"
        );
    }

    #[test]
    fn diffraction_gate_requires_a_working_aperture() {
        let buf = vec![128u8; 16 * 16 * 4];
        for aperture in [0.0, 0.5, 50.5] {
            let lens = LensCorrection { diffraction_enabled: true, aperture, ..Default::default() };
            let mut out = buf.clone();
            stage_diffraction(&mut out, 16, 16, &lens, 1.0);
            assert_eq!(out, buf, "aperture {aperture} must be outside the gate");
        }
    }

    #[test]
    fn pixel_pitch_comes_from_the_mount_and_scales_the_fallback() {
        let mounted = LensCorrection { mount: Some("nikon z".into()), ..Default::default() };
        let pitch = pixel_pitch_um(&mounted, 6048, 4032, 1.0);
        assert!((pitch - 5.946).abs() < 0.01, "pitch {pitch}");
        assert_eq!(pixel_pitch_um(&LensCorrection::default(), 6048, 4032, 1.0), 4.0);
        assert_eq!(pixel_pitch_um(&LensCorrection::default(), 1000, 1000, 0.5), 8.0);
    }

    /// The luma-plane reuse in NR/sharpening must not change a single byte:
    /// the stage output equals a per-pixel recompute of the same maths.
    #[test]
    fn luma_plane_reuse_matches_the_naive_recompute() {
        let (w, h) = (37usize, 23usize);
        let buf: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                [((x * 13 + y * 7) % 256) as u8, ((x * 5) % 256) as u8, ((y * 11) % 256) as u8, 255]
            })
            .collect();
        let plane = luma_plane(&buf, w, h);
        for (i, px) in buf.chunks_exact(4).enumerate() {
            assert_eq!(plane[i], luma8([px[0], px[1], px[2]]), "luma plane differs at {i}");
        }
        let target: Vec<f32> = plane.iter().map(|l| (*l * 0.8 + 20.0).clamp(0.0, 255.0)).collect();
        let mut a = buf.clone();
        let mut b = buf.clone();
        apply_luma(&mut a, w, &plane, &target);
        for (y, row) in b.chunks_exact_mut(w * 4).enumerate() {
            for x in 0..w {
                let i = x * 4;
                let p = [row[i], row[i + 1], row[i + 2]];
                let c = luma8(p);
                let new = target[y * w + x];
                if (new - c).abs() < 0.05 {
                    continue;
                }
                let f = if c > 1.0 { new / c } else { 1.0 };
                for k in 0..3 {
                    row[i + k] = (p[k] as f32 * f).clamp(0.0, 255.0).round() as u8;
                }
            }
        }
        assert_eq!(a, b, "apply_luma must match the per-pixel recompute");
    }

    fn luma_variance(buf: &[u8]) -> f32 {
        let l: Vec<u8> = buf.chunks_exact(4).map(|p| luma8([p[0], p[1], p[2]]).round() as u8).collect();
        variance(&l)
    }

    fn checkerboard(size: usize) -> Vec<u8> {
        let mut buf = vec![0u8; size * size * 4];
        for y in 0..size {
            for x in 0..size {
                let v = if (x + y) % 2 == 0 { 230 } else { 26 };
                let i = (y * size + x) * 4;
                buf[i] = v;
                buf[i + 1] = v;
                buf[i + 2] = v;
                buf[i + 3] = 255;
            }
        }
        buf
    }

    fn flat(size: usize, v: u8) -> Vec<u8> {
        let mut buf = vec![v; size * size * 4];
        for i in (0..buf.len()).step_by(4) {
            buf[i + 3] = 255;
        }
        buf
    }

    fn step_edge(size: usize) -> Vec<u8> {
        let mut buf = flat(size, 40);
        for y in 0..size {
            for x in (size / 2)..size {
                let i = (y * size + x) * 4;
                buf[i] = 200;
                buf[i + 1] = 200;
                buf[i + 2] = 200;
            }
        }
        buf
    }

    #[test]
    fn all_off_leaves_bytes_identical() {
        let mut buf = checkerboard(8);
        let before = buf.clone();
        post_process(&mut buf, 8, 8, &off(), 1.0);
        assert_eq!(before, buf);
    }

    #[test]
    fn bad_input_is_safe() {
        let mut buf = vec![0u8; 10];
        post_process(&mut buf, 8, 8, &off(), 1.0);
        post_process(&mut buf, 0, 0, &off(), 1.0);
        let mut one = vec![7u8; 4];
        let mut adj = off();
        adj.sharpening.amount = 100.0;
        post_process(&mut one, 1, 1, &adj, 1.0);
        assert_eq!(one, vec![7u8; 4]);
        let mut empty: Vec<u8> = Vec::new();
        post_process(&mut empty, 4, 4, &adj, 1.0);
    }

    #[test]
    fn luminance_nr_reduces_variance() {
        let mut buf = vec![0u8; 16 * 16 * 4];
        for i in 0..buf.len() / 4 {
            let v = if i % 3 == 0 { 60 } else { 120 };
            buf[i * 4] = v;
            buf[i * 4 + 1] = v;
            buf[i * 4 + 2] = v;
            buf[i * 4 + 3] = 255;
        }
        let before = luma_variance(&buf);
        let mut adj = off();
        adj.noise_reduction.luminance = 80.0;
        adj.noise_reduction.details = 0.0;
        post_process(&mut buf, 16, 16, &adj, 1.0);
        assert!(luma_variance(&buf) < before, "NR should reduce variance");
    }

    #[test]
    fn single_pixel_removes_hot_pixel() {
        let mut buf = flat(16, 100);
        let hot = (8 * 16 + 8) * 4;
        buf[hot] = 250;
        buf[hot + 1] = 250;
        buf[hot + 2] = 250;
        let mut adj = off();
        adj.noise_reduction.single_pixel = 100.0;
        post_process(&mut buf, 16, 16, &adj, 1.0);
        assert!(buf[hot] < 130, "hot pixel should be pulled back: {}", buf[hot]);
    }

    /// Dark sky, a few one-pixel stars, one hot pixel and a 3x3 bright plateau
    /// (a planet-sized object that must not be taken for a spike).
    fn star_field(size: usize) -> Vec<u8> {
        let mut buf = flat(size, 18);
        let mut put = |x: usize, y: usize, v: u8| {
            let i = (y * size + x) * 4;
            buf[i] = v;
            buf[i + 1] = v;
            buf[i + 2] = v;
        };
        put(3, 4, 210);
        put(11, 2, 180);
        put(6, 12, 160);
        put(size / 2, size / 2, 255);
        for dy in 0..5 {
            for dx in 0..5 {
                put(size / 2 - 8 + dx, size / 2 - 8 + dy, 200);
            }
        }
        buf
    }

    #[test]
    fn single_pixel_zero_is_byte_identical() {
        let (w, h) = (24usize, 24usize);
        let before = star_field(w);
        let mut buf = before.clone();
        let mut adj = off();
        adj.noise_reduction.single_pixel = 0.0;
        post_process(&mut buf, w as u32, h as u32, &adj, 1.0);
        assert_eq!(buf, before, "amount 0 must leave every byte alone");
        stage_single_pixel(&mut buf, w, h, &adj.noise_reduction);
        assert_eq!(buf, before, "the stage itself must be a no-op at 0");
    }

    #[test]
    fn single_pixel_replaces_the_spike_and_leaves_its_neighbours() {
        let (w, h) = (24usize, 24usize);
        let before = star_field(w);
        let mut buf = before.clone();
        let mut adj = off();
        adj.noise_reduction.single_pixel = 100.0;
        post_process(&mut buf, w as u32, h as u32, &adj, 1.0);

        let hot = ((h / 2) * w + w / 2) * 4;
        assert!(buf[hot] < 60, "hot pixel should fall to the sky: {}", buf[hot]);
        for dy in -1i32..=1 {
            for dx in -1i32..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let i = (((h / 2) as i32 + dy) as usize * w
                    + ((w / 2) as i32 + dx) as usize)
                    * 4;
                assert_eq!(&buf[i..i + 3], &before[i..i + 3], "neighbour {dx},{dy} changed");
            }
        }
    }

    #[test]
    fn single_pixel_keeps_a_bright_plateau() {
        let (w, h) = (24usize, 24usize);
        let before = star_field(w);
        let mut buf = before.clone();
        let mut adj = off();
        adj.noise_reduction.single_pixel = 100.0;
        post_process(&mut buf, w as u32, h as u32, &adj, 1.0);
        for dy in 0..3 {
            for dx in 0..3 {
                let i = ((h / 2 - 7 + dy) * w + (w / 2 - 7 + dx)) * 4;
                assert_eq!(&buf[i..i + 3], &before[i..i + 3], "plateau {dx},{dy} changed");
            }
        }
    }

    #[test]
    fn color_nr_reduces_chroma_variance_only() {
        let mut buf = vec![0u8; 16 * 16 * 4];
        for i in 0..buf.len() / 4 {
            buf[i * 4] = 128 + ((i * 37) % 21) as u8;
            buf[i * 4 + 1] = 128 + ((i * 53) % 19) as u8;
            buf[i * 4 + 2] = 128 + ((i * 71) % 23) as u8;
            buf[i * 4 + 3] = 255;
        }
        let chroma = |b: &[u8]| -> f32 {
            b.chunks_exact(4)
                .map(|p| (p[0] as f32 - p[1] as f32).abs() + (p[2] as f32 - p[1] as f32).abs())
                .sum::<f32>()
                / (b.len() / 4) as f32
        };
        let chroma_before = chroma(&buf);
        let luma_before = luma_variance(&buf);
        let mut adj = off();
        adj.noise_reduction.color = 100.0;
        post_process(&mut buf, 16, 16, &adj, 1.0);
        assert!(chroma(&buf) < chroma_before, "colour noise should drop");
        assert!(
            (luma_variance(&buf) - luma_before).abs() < 2.0,
            "luminance should be preserved"
        );
    }

    fn chroma_speckle(size: usize) -> Vec<u8> {
        let mut buf = vec![0u8; size * size * 4];
        for i in 0..size * size {
            let (cb, cr) = match i % 4 {
                0 => (-24i32, 18i32),
                1 => (20, -16),
                2 => (0, 0),
                _ => (-10, 10),
            };
            let y = 118.0f32;
            let rgb = [
                y + 1.402 * cr as f32,
                y - 0.344136 * cb as f32 - 0.714136 * cr as f32,
                y + 1.772 * cb as f32,
            ];
            for (k, v) in rgb.into_iter().enumerate() {
                buf[i * 4 + k] = v.clamp(0.0, 255.0).round() as u8;
            }
            buf[i * 4 + 3] = 255;
        }
        buf
    }

    /// CNR is off at 0, never adds chroma noise as the amount grows, and keeps
    /// the three channels coupled (luma stays put).
    #[test]
    fn cnr_zero_is_identity_and_strength_is_monotonic() {
        let (w, h) = (24usize, 24usize);
        let base = chroma_speckle(w);
        let run = |amount: f32| {
            let mut adj = off();
            adj.noise_reduction.color = amount;
            let mut buf = base.clone();
            stage_color_nr(&mut buf, w, h, &adj.noise_reduction, 1.0);
            buf
        };
        assert_eq!(run(0.0), base, "CNR 0 must be byte-identical");
        let c25 = chroma_variance(&run(25.0));
        let c50 = chroma_variance(&run(50.0));
        let c100 = chroma_variance(&run(100.0));
        assert!(c25 <= chroma_variance(&base));
        assert!(c50 <= c25 + 1e-3 && c100 <= c50 + 1e-3, "CNR must not add chroma noise");
        for amount in [25.0, 50.0, 100.0] {
            let out = run(amount);
            for (i, px) in out.chunks_exact(4).enumerate() {
                let l = luma8([px[0], px[1], px[2]]);
                assert!((l - 118.0).abs() < 6.0, "luma moved at {i}: {l}");
            }
        }
    }

    /// Low-amplitude colour oscillation with a constant luma: the exact shape
    /// colour moire takes, and what the v4 guard is meant to pass.
    fn colour_moire(size: usize) -> Vec<u8> {
        let mut buf = vec![0u8; size * size * 4];
        for y in 0..size {
            for x in 0..size {
                let i = (y * size + x) * 4;
                let (r, b) = if (x / 4 + y / 4) % 2 == 0 { (128, 110) } else { (108, 128) };
                buf[i] = r;
                buf[i + 1] = 118;
                buf[i + 2] = b;
                buf[i + 3] = 255;
            }
        }
        buf
    }

    fn chroma_variance(buf: &[u8]) -> f32 {
        let c: Vec<u8> = buf.chunks_exact(4).map(|p| (p[0] as i16 - p[1] as i16 + 128) as u8).collect();
        variance(&c)
    }

    /// The kernel's own luma: `(R + 2G + B) / 4`, before Rec.709 weighting.
    fn kernel_luma_variance(buf: &[u8]) -> f32 {
        let y: Vec<u8> = buf
            .chunks_exact(4)
            .map(|p| ((p[0] as u32 + 2 * p[1] as u32 + p[2] as u32) / 4) as u8)
            .collect();
        variance(&y)
    }

    #[test]
    fn moire_flattens_colour_pattern_and_keeps_luminance() {
        let mut buf = colour_moire(32);
        let (luma_before, chroma_before) = (kernel_luma_variance(&buf), chroma_variance(&buf));
        let mut adj = off();
        adj.moire.amount = 100.0;
        adj.moire.pattern = 8.0;
        stage_moire(&mut buf, 32, 32, &adj.moire);
        assert!(chroma_variance(&buf) < chroma_before * 0.5, "colour moire should drop");
        assert!((kernel_luma_variance(&buf) - luma_before).abs() < 1e-6, "luminance is not filtered");
    }

    #[test]
    fn moire_leaves_a_luminance_checkerboard_alone() {
        let mut buf = checkerboard(16);
        let before = luma_variance(&buf);
        let mut adj = off();
        adj.moire.amount = 100.0;
        adj.moire.pattern = 8.0;
        stage_moire(&mut buf, 16, 16, &adj.moire);
        assert!((luma_variance(&buf) - before).abs() < 1.0, "moire is chroma-only");
    }

    #[test]
    fn zero_moire_is_identity() {
        let base = colour_moire(16);
        let mut buf = base.clone();
        stage_moire(&mut buf, 16, 16, &off().moire);
        assert_eq!(buf, base);
    }

    #[test]
    fn sharpening_increases_edge_variance() {
        let mut buf = step_edge(16);
        let before = luma_variance(&buf);
        let mut adj = off();
        adj.sharpening.amount = 200.0;
        adj.sharpening.radius = 1.0;
        adj.sharpening.threshold = 0.0;
        post_process(&mut buf, 16, 16, &adj, 1.0);
        assert!(luma_variance(&buf) > before, "sharpening should raise contrast");
    }

    #[test]
    fn halo_suppression_limits_overshoot() {
        let make = |halo: f32| {
            let mut buf = step_edge(16);
            let mut adj = off();
            adj.sharpening.amount = 200.0;
            adj.sharpening.radius = 1.0;
            adj.sharpening.threshold = 0.0;
            adj.sharpening.halo_suppression = halo;
            post_process(&mut buf, 16, 16, &adj, 1.0);
            buf
        };
        let plain = make(0.0);
        let guarded = make(100.0);
        let dark = 7 * 4;
        let bright = 8 * 4;
        assert!(guarded[dark] >= plain[dark], "dark halo must be pulled back up");
        assert!(guarded[dark] <= 200, "no overshoot past the bright side");
        assert!(guarded[bright] <= plain[bright], "bright halo must be pulled back down");
        assert!(guarded[bright] >= 40, "no undershoot past the dark side");
    }

    #[test]
    fn lens_distortion_moves_pixels_and_can_hide_areas() {
        let base = checkerboard(16);
        let mut buf = base.clone();
        let mut adj = off();
        adj.lens.distortion = 100.0;
        adj.lens.hide_distorted_areas = true;
        post_process(&mut buf, 16, 16, &adj, 1.0);
        assert_ne!(base, buf, "distortion must change the frame");
        assert!(
            buf.chunks_exact(4).any(|p| p[0] == 0 && p[1] == 0 && p[2] == 0),
            "corners should be hidden"
        );
    }

    #[test]
    fn light_falloff_brightens_corners() {
        let mut buf = flat(16, 80);
        let mut adj = off();
        adj.lens.light_falloff = 100.0;
        post_process(&mut buf, 16, 16, &adj, 1.0);
        let corner = 0;
        let centre = (8 * 16 + 8) * 4;
        assert!(buf[corner] > buf[centre], "falloff correction brightens the edge");
    }

    #[test]
    fn dust_removal_gate_off_leaves_the_frame_untouched() {
        let base = flat(16, 90);
        let mut buf = base.clone();
        let mut adj = off();
        adj.lens.dust_removal_enabled = false;
        adj.lens.dust_patches = vec![crate::DustPatch {
            x: 0,
            y: 0,
            w: 8,
            h: 8,
            offset: 0,
            bias: 0,
            step: 16,
        }];
        post_process(&mut buf, 16, 16, &adj, 1.0);
        assert_eq!(base, buf, "ZLENSDUSTREMOVALENABLED=0 must keep the patches inert");
    }

    #[test]
    fn dust_removal_applies_patches_when_armed() {
        let mut buf = flat(16, 90);
        let mut adj = off();
        adj.lens.dust_removal_enabled = true;
        adj.lens.dust_patches = vec![crate::DustPatch {
            x: 0,
            y: 0,
            w: 8,
            h: 8,
            offset: 0,
            bias: 0,
            step: 16,
        }];
        post_process(&mut buf, 16, 16, &adj, 1.0);
        assert!(buf[0] > 90, "the patch brightens its rectangle");
        assert_eq!(buf[(12 * 16 + 12) * 4], 90, "outside the patch is untouched");
    }

    fn sharp_curves() -> RadialBlur {
        RadialBlur {
            radial: vec![(0.0, 0.0), (1.0, 200000.0)],
            transverse: vec![(0.0, 0.0), (1.0, 200000.0)],
            valid: true,
        }
    }

    #[test]
    fn sharpness_falloff_below_half_amount_leaves_the_frame_untouched() {
        let base = checkerboard(16);
        let mut buf = base.clone();
        let mut adj = off();
        adj.lens.sharpness_falloff = 0.49;
        adj.lens.radial_blur = sharp_curves();
        stage_sharpness_falloff(&mut buf, 16, 16, &adj.lens);
        assert_eq!(base, buf, "amount < 0.5 is CImgOpFixRadialBlur identity");
    }

    #[test]
    fn sharpness_falloff_needs_a_valid_curve() {
        let base = checkerboard(16);
        let mut buf = base.clone();
        let mut adj = off();
        adj.lens.sharpness_falloff = 100.0;
        adj.lens.radial_blur = RadialBlur { valid: false, ..sharp_curves() };
        stage_sharpness_falloff(&mut buf, 16, 16, &adj.lens);
        assert_eq!(base, buf, "valid = 0 is identity");
    }

    #[test]
    fn sharpness_falloff_with_a_curve_changes_the_frame() {
        let base = checkerboard(32);
        let mut buf = base.clone();
        let mut adj = off();
        adj.lens.sharpness_falloff = 100.0;
        adj.lens.radial_blur = sharp_curves();
        post_process(&mut buf, 32, 32, &adj, 1.0);
        assert_ne!(base, buf, "a valid curve must reach the pixels");
    }

    #[test]
    fn zero_shift_and_tilt_leaves_the_frame_untouched() {
        let base = checkerboard(16);
        let mut buf = base.clone();
        let adj = off();
        stage_movement(&mut buf, 16, 16, &adj.lens);
        assert_eq!(base, buf, "shift_x = shift_y = tilt = 0 is identity");
    }

    fn stage_movement_full_output_reference(
        buf: &mut [u8],
        w: usize,
        h: usize,
        adj: &LensCorrection,
    ) {
        let tilt = adj.tilt.clamp(-45.0, 45.0);
        if w == 0 || h == 0 || (adj.shift_x == 0.0 && adj.shift_y == 0.0 && tilt == 0.0) {
            return;
        }
        let shift_mag = adj.shift_x.hypot(adj.shift_y);
        let m = if shift_mag > 0.0 { 0.01 * shift_mag } else { 1.0 };
        let tq = (m * tilt.to_radians().tan()).atan();
        let k = (tq * 0.5).tan();
        let (mut m0, mut m1, mut m2, mut m3, mut m4, mut m5) =
            (1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0);
        m0 += k * m4;
        m1 += k * m4;
        m2 += k * m4;
        m3 += k * m0;
        m4 += k * m1;
        m5 += k * m2;
        let diag = (((w * w + h * h) as f32).sqrt()).max(1.0);
        m2 /= diag;
        m5 /= diag;
        let tx = adj.shift_x * 0.01 * w as f32 * 0.5;
        let ty = adj.shift_y * 0.01 * h as f32 * 0.5;
        let src = buf.to_vec();
        let (cx, cy) = (w as f32 * 0.5, h as f32 * 0.5);
        let mut out = vec![0u8; w * h * 4];
        out.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let xx = x as f32 + 0.5 - cx;
                let yy = y as f32 + 0.5 - cy;
                let a = m0 - m2 * xx;
                let b = m1 - m2 * yy;
                let c = m3 - m5 * xx;
                let d = m4 - m5 * yy;
                let det = d * a - c * b;
                let (u, v) = if det.abs() < 1e-6 {
                    (xx, yy)
                } else {
                    (
                        (d * xx - c * yy) / det - tx,
                        (a * yy - b * xx) / det - ty,
                    )
                };
                let rgb = sample_bilinear(&src, w, h, cx + u, cy + v);
                let i = x * 4;
                for k in 0..3 {
                    row[i + k] = rgb[k].clamp(0.0, 255.0).round() as u8;
                }
                row[i + 3] = src[(y * w + x) * 4 + 3];
            }
        });
        buf.copy_from_slice(&out);
    }

    #[test]
    fn movement_in_place_matches_full_output_reference() {
        let (w, h) = (97usize, 71usize);
        let mut base = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 230 } else { 26 };
                base.extend_from_slice(&[v, v, v, 255]);
            }
        }
        let mut expected = base.clone();
        let mut actual = base;
        let mut adj = off();
        adj.lens.shift_x = 7.5;
        adj.lens.shift_y = -3.25;
        adj.lens.tilt = 11.0;
        stage_movement_full_output_reference(&mut expected, w, h, &adj.lens);
        stage_movement(&mut actual, w, h, &adj.lens);
        assert_eq!(actual, expected);
    }

    #[test]
    fn shift_moves_the_frame() {
        let base = checkerboard(16);
        let mut buf = base.clone();
        let mut adj = off();
        adj.lens.shift_x = 5.0;
        post_process(&mut buf, 16, 16, &adj, 1.0);
        assert_ne!(base, buf, "a non-zero shift must resample");
    }

    #[test]
    fn grain_is_deterministic_and_adds_variance() {
        let mut a = flat(16, 120);
        let mut b = a.clone();
        let mut adj = off();
        adj.film_grain.impact = 100.0;
        adj.film_grain.granularity = 50.0;
        post_process(&mut a, 16, 16, &adj, 1.0);
        post_process(&mut b, 16, 16, &adj, 1.0);
        assert_eq!(a, b, "grain must be deterministic");
        assert!(luma_variance(&a) > 0.0, "grain must add noise");
    }

    fn warp(distortion: &[(f32, f32)], falloff: &[(f32, f32)]) -> crate::mock::lens_profile::LensWarp {
        crate::mock::lens_profile::LensWarp {
            distortion: distortion.to_vec(),
            falloff: falloff.to_vec(),
            red: Vec::new(),
            blue: Vec::new(),
        }
    }

    /// A white ring at `t` of the half diagonal on a black frame.
    fn ring(size: usize, t: f32) -> Vec<u8> {
        let mut buf = vec![0u8; size * size * 4];
        let c = size as f32 * 0.5;
        let half_diag = ((size * size * 2) as f32).sqrt() * 0.5;
        for y in 0..size {
            for x in 0..size {
                let d = ((x as f32 + 0.5 - c).powi(2) + (y as f32 + 0.5 - c).powi(2)).sqrt();
                if (d / half_diag - t).abs() < 1.5 / half_diag {
                    let i = (y * size + x) * 4;
                    buf[i] = 240;
                    buf[i + 1] = 240;
                    buf[i + 2] = 240;
                }
                buf[(y * size + x) * 4 + 3] = 255;
            }
        }

        buf
    }

    fn axis_gradient(size: usize) -> Vec<u8> {
        let mut buf = vec![0u8; size * size * 4];
        for y in 0..size {
            for x in 0..size {
                let i = (y * size + x) * 4;
                buf[i] = (200.0 - x as f32 * 2.0).clamp(0.0, 255.0).round() as u8;
                buf[i + 1] = 100;
                buf[i + 2] = (x as f32 * 2.0).clamp(0.0, 255.0).round() as u8;
                buf[i + 3] = 255;
            }
        }
        buf
    }

    /// Brightest `t` on the horizontal centre line.
    fn brightest_t(buf: &[u8], size: usize) -> f32 {
        let half_diag = ((size * size * 2) as f32).sqrt() * 0.5;
        let y = size / 2;
        let (mut best, mut best_t) = (0u8, 0.0);
        for x in size / 2..size {
            let v = buf[(y * size + x) * 4];
            if v > best {
                best = v;
                best_t = (x as f32 + 0.5 - size as f32 * 0.5) / half_diag;
            }
        }
        best_t
    }

    #[test]
    fn lens_profile_stays_off_until_enabled() {
        let mut a = ring(48, 0.7);
        let mut b = a.clone();
        let mut adj = off();
        adj.lens.distortion = 100.0;
        adj.lens.light_falloff = 100.0;
        post_process_with_lens(&mut a, 48, 48, &adj, 1.0, Some(&warp(&[(0.0, 0.0), (1.0, 0.05)], &[(0.0, 0.0), (1.0, -1.0)])));
        post_process(&mut b, 48, 48, &adj, 1.0);
        assert_eq!(a, b, "the profile must be inert while use_profile is false");
    }

    #[test]
    fn ca_zero_is_byte_identical() {
        let mut buf = ring(48, 0.7);
        let before = buf.clone();
        let mut adj = off();
        adj.lens.use_profile = true;
        adj.lens.chromatic_aberration = true;
        adj.lens.ca_red_points = vec![(0.0, 0.0), (1.0, 0.0)];
        adj.lens.ca_blue_points = vec![(0.0, 0.0), (1.0, 0.0)];
        post_process_with_lens(&mut buf, 48, 48, &adj, 1.0, Some(&warp(&[], &[])));
        assert_eq!(buf, before);
    }

    #[test]
    fn requested_profile_without_a_table_leaves_pixels_untouched() {
        let mut buf = ring(48, 0.7);
        let before = buf.clone();
        let mut adj = off();
        adj.lens.use_profile = true;
        adj.lens.distortion = 100.0;
        adj.lens.light_falloff = 100.0;
        post_process_with_lens(&mut buf, 48, 48, &adj, 1.0, None);
        assert_eq!(buf, before, "no calibration table must not invent a warp");
    }

    #[test]
    fn ca_moves_red_and_blue_in_opposite_directions() {
        let mut buf = axis_gradient(64);
        let before = buf.clone();
        let mut adj = off();
        adj.lens.use_profile = true;
        adj.lens.chromatic_aberration = true;
        adj.lens.ca_red_points = vec![(0.0, 0.0), (1.0, 0.04)];
        adj.lens.ca_blue_points = vec![(0.0, 0.0), (1.0, -0.04)];
        post_process_with_lens(&mut buf, 64, 64, &adj, 1.0, Some(&warp(&[], &[])));
        let probe = (32 * 64 + 56) * 4;
        assert!(buf[probe] < before[probe], "positive red shift must sample outward");
        assert!(buf[probe + 2] < before[probe + 2], "negative blue shift must sample inward");
        assert!(buf[probe + 1].abs_diff(before[probe + 1]) <= 8, "green must stay close");
    }

    #[test]
    fn lens_profile_distortion_moves_the_ring_outward() {
        let mut buf = ring(64, 0.5);
        let before = brightest_t(&buf, 64);
        let mut adj = off();
        adj.lens.use_profile = true;
        adj.lens.distortion = 100.0;
        post_process_with_lens(&mut buf, 64, 64, &adj, 1.0, Some(&warp(&[(0.0, 0.0), (1.0, 0.05)], &[])));
        let after = brightest_t(&buf, 64);
        assert!((after - before - 0.05).abs() < 0.01, "ring moved {before} -> {after}, expected +0.05");
    }

    #[test]
    fn lens_profile_normalization_is_sensor_independent() {
        let mut small = ring(32, 0.5);
        let mut large = ring(64, 0.5);
        let w = warp(&[(0.0, 0.0), (1.0, 0.05)], &[]);
        let mut adj = off();
        adj.lens.use_profile = true;
        adj.lens.distortion = 100.0;
        post_process_with_lens(&mut small, 32, 32, &adj, 1.0, Some(&w));
        post_process_with_lens(&mut large, 64, 64, &adj, 1.0, Some(&w));
        let (a, b) = (brightest_t(&small, 32), brightest_t(&large, 64));
        assert!((a - b).abs() < 0.02, "same profile gave {a} and {b} at different sensor sizes");
    }

    #[test]
    fn lens_profile_falloff_applies_two_to_the_ev() {
        let mut buf = flat(16, 80);
        let mut adj = off();
        adj.lens.use_profile = true;
        adj.lens.light_falloff = 100.0;
        let w = warp(&[], &[(0.0, 0.0), (1.0, -1.5)]);
        let half_diag = (16f32 * 16.0 * 2.0).sqrt() * 0.5;
        let gain = |t: f32| 2f32.powf(-crate::mock::lens_profile::LensWarp::sample(&w.falloff, t));
        let corner_t = (7.5f32 * 7.5 * 2.0).sqrt() / half_diag;
        let centre_t = (0.5f32 * 0.5 * 2.0).sqrt() / half_diag;
        let expected = gain(corner_t) / gain(centre_t);
        post_process_with_lens(&mut buf, 16, 16, &adj, 1.0, Some(&w));
        let corner = buf[0] as f32;
        let centre = buf[(8 * 16 + 8) * 4] as f32;
        assert!((corner / centre - expected).abs() < 0.05, "corner/centre was {}, expected {expected}", corner / centre);
    }

    #[test]
    fn lens_profile_clamps_past_the_table_end() {
        let mut a = ring(32, 0.9);
        let mut b = a.clone();
        let mut adj = off();
        adj.lens.use_profile = true;
        adj.lens.distortion = 100.0;
        // Table stops at half the radius; everything beyond uses its last value.
        post_process_with_lens(&mut a, 32, 32, &adj, 1.0, Some(&warp(&[(0.0, 0.0), (0.5, 0.04)], &[])));
        post_process_with_lens(&mut b, 32, 32, &adj, 1.0, Some(&warp(&[(0.0, 0.0), (0.5, 0.04), (1.0, 0.04)], &[])));
        assert_eq!(a, b);
    }

    // ── purple fringe: signed -100..100, 0 is the identity ──

    #[test]
    fn purple_norm_maps_the_signed_slider() {
        assert_eq!(crate::mock::purple_norm(-100.0), 0.0);
        assert_eq!(crate::mock::purple_norm(0.0), 0.5);
        assert_eq!(crate::mock::purple_norm(100.0), 1.0);
        assert!((crate::mock::purple_norm(-50.0) - 0.25).abs() < 1e-6);
        assert!((crate::mock::purple_norm(50.0) - 0.75).abs() < 1e-6);
    }

    #[test]
    fn purple_strength_is_symmetric_around_zero() {
        for amount in [5.0f32, 25.0, 50.0, 100.0] {
            let plus = crate::mock::purple_strength(crate::mock::purple_norm(amount));
            let minus = crate::mock::purple_strength(crate::mock::purple_norm(-amount));
            assert!((plus + minus).abs() < 1e-6, "{amount}: {plus} vs {minus}");
            assert!((plus - amount / 100.0).abs() < 1e-6, "{amount}: {plus}");
        }
        assert_eq!(
            crate::mock::purple_strength(crate::mock::purple_norm(0.0)),
            0.0
        );
    }

    /// 3×3 with a purple centre and black/white neighbours, so `pn` and `edge`
    /// both sit at their maximum and the arithmetic is easy to assert.
    fn purple_probe() -> Vec<u8> {
        let mut buf = vec![0u8; 3 * 3 * 4];
        for (i, px) in buf.chunks_mut(4).enumerate() {
            let luma = match i {
                4 => [180, 100, 180],
                _ if i % 2 == 0 => [0, 0, 0],
                _ => [255, 255, 255],
            };
            px[..3].copy_from_slice(&luma);
            px[3] = 255;
        }
        buf
    }

    fn stage_purple_fringing_full_snapshot_reference(
        buf: &mut [u8],
        w: usize,
        h: usize,
        purple: f32,
    ) {
        let strength = crate::mock::purple_strength(purple);
        if strength == 0.0 {
            return;
        }
        let src = buf.to_vec();
        buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let i = x * 4;
                let j = (y * w + x) * 4;
                let p = [src[j], src[j + 1], src[j + 2]];
                let r = p[0] as f32;
                let g = p[1] as f32;
                let b = p[2] as f32;
                let pn = ((r.min(b) - g) / 255.0).clamp(0.0, 1.0);
                if pn <= 0.0 {
                    continue;
                }
                let mut lo = 255.0f32;
                let mut hi = 0.0f32;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let xx = clampi(x as i32 + dx, 0, w as i32 - 1) as usize;
                        let yy = clampi(y as i32 + dy, 0, h as i32 - 1) as usize;
                        let n = (yy * w + xx) * 4;
                        let l = luma8([src[n], src[n + 1], src[n + 2]]);
                        lo = lo.min(l);
                        hi = hi.max(l);
                    }
                }
                let edge = ((hi - lo) / 40.0).clamp(0.0, 1.0);
                let amount = strength * pn * edge;
                if amount == 0.0 {
                    continue;
                }
                let gray = luma8(p);
                for k in 0..3 {
                    let v = p[k] as f32 * (1.0 - amount) + gray * amount;
                    row[i + k] = v.clamp(0.0, 255.0).round() as u8;
                }
            }
        });
    }

    #[test]
    fn purple_band_snapshot_matches_full_frame_reference() {
        let (w, h) = (137usize, 259usize);
        let mut base = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let dark = (x / 3 + y / 5) % 2 == 0;
                let g = if dark { 38 } else { 196 };
                let r = (g + 72 + (x % 31) as u16).min(255);
                let b = (g + 93 + (y % 37) as u16).min(255);
                base.extend_from_slice(&[r as u8, g as u8, b as u8, 255]);
            }
        }

        let mut changed = false;
        for amount in [50.0f32, -50.0] {
            let purple = crate::mock::purple_norm(amount);
            let mut expected = base.clone();
            let mut actual = base.clone();
            stage_purple_fringing_full_snapshot_reference(&mut expected, w, h, purple);
            stage_purple_fringing(&mut actual, w, h, purple);
            changed |= actual != base;
            assert_eq!(actual, expected, "amount {amount}");
        }
        assert!(changed, "the parity fixture must exercise purple pixels");
    }

    #[test]
    fn purple_zero_amount_is_byte_identical() {
        let mut buf = purple_probe();
        let base = buf.clone();
        stage_purple_fringing(&mut buf, 3, 3, crate::mock::purple_norm(0.0));
        assert_eq!(buf, base);

        let mut adj = off();
        adj.lens.purple_fringing = 0.0;
        let mut whole = base.clone();
        stage_lens(&mut whole, 3, 3, &adj.lens, None);
        assert_eq!(whole, base);
    }

    #[test]
    fn purple_positive_and_negative_are_mirror_images() {
        let base = purple_probe();
        let mut up = base.clone();
        let mut down = base.clone();
        stage_purple_fringing(&mut up, 3, 3, crate::mock::purple_norm(50.0));
        stage_purple_fringing(&mut down, 3, 3, crate::mock::purple_norm(-50.0));
        let centre = 4 * 4;
        for k in 0..3 {
            let delta_up = up[centre + k] as i32 - base[centre + k] as i32;
            let delta_down = down[centre + k] as i32 - base[centre + k] as i32;
            assert!(delta_up != 0, "channel {k} did not move");
            assert_eq!(delta_up.signum(), -delta_down.signum(), "channel {k}");
            assert!(
                (delta_up.abs() - delta_down.abs()).abs() <= 1,
                "channel {k}: {delta_up} vs {delta_down}"
            );
        }
        // The neighbours are the edge source and must stay untouched.
        for i in 0..9 {
            if i != 4 {
                assert_eq!(&up[i * 4..i * 4 + 4], &base[i * 4..i * 4 + 4]);
            }
        }
    }

    // ── Lateral chromatic aberration (six-curve profile table) ─────────────

    fn ca_lens(strength: f32, shift: f32, ra: f32) -> LensCorrection {
        let mut lens = LensCorrection::default();
        lens.chromatic_aberration = true;
        lens.ca_fringing.red_shift = vec![(0.0, shift), (1.0, shift)];
        lens.ca_fringing.blue_shift = vec![(0.0, -shift), (1.0, -shift)];
        lens.ca_fringing.red_ra = vec![(0.0, ra), (1.0, ra)];
        lens.ca_fringing.blue_ra = vec![(0.0, -ra), (1.0, -ra)];
        lens.ca_fringing.red_strength = vec![(0.0, strength), (1.0, strength)];
        lens.ca_fringing.blue_strength = vec![(0.0, strength), (1.0, strength)];
        lens
    }

    fn ca_red_at(buf: &[u8], size: usize, x: usize, y: usize) -> i32 {
        buf[(y * size + x) * 4] as i32
    }

    #[test]
    fn ca_without_curves_or_with_zero_strength_is_byte_identical() {
        let base = axis_gradient(48);
        let mut none = base.clone();
        stage_ca_fringes(&mut none, 48, 48, &ca_lens(1.0, 0.0, 0.0));
        assert_eq!(none, base, "empty tables resampled the frame");

        let mut zero = base.clone();
        stage_ca_fringes(&mut zero, 48, 48, &ca_lens(0.0, 0.1, 0.0));
        assert_eq!(zero, base, "strength 0 moved pixels");
    }

    #[test]
    fn ca_shift_moves_the_red_edge_along_the_radius() {
        let base = axis_gradient(48);
        let mut half = base.clone();
        let mut full = base.clone();
        stage_ca_fringes(&mut half, 48, 48, &ca_lens(0.5, 0.1, 0.0));
        stage_ca_fringes(&mut full, 48, 48, &ca_lens(1.0, 0.1, 0.0));
        // R falls with x, so an outward sample lowers R further from the centre.
        let p = (40, 8);
        let (b, h, f) =
            (ca_red_at(&base, 48, p.0, p.1), ca_red_at(&half, 48, p.0, p.1), ca_red_at(&full, 48, p.0, p.1));
        assert!(h < b, "half strength did not move red: {b} -> {h}");
        assert!(f < h, "full strength must move red further: {h} -> {f}");
    }

    #[test]
    fn ca_shift_grows_with_the_radius_and_leaves_the_centre_alone() {
        let base = axis_gradient(48);
        let mut moved = base.clone();
        stage_ca_fringes(&mut moved, 48, 48, &ca_lens(1.0, 0.1, 0.0));
        let centre_delta = (ca_red_at(&base, 48, 24, 24) - ca_red_at(&moved, 48, 24, 24)).abs();
        let corner_delta = (ca_red_at(&base, 48, 44, 44) - ca_red_at(&moved, 48, 44, 44)).abs();
        assert!(centre_delta <= 1, "the optical centre moved: {centre_delta}");
        assert!(corner_delta > centre_delta, "the corner moved less than the centre");
    }

    #[test]
    fn ca_ra_adds_radial_gain_on_top_of_the_shift() {
        let base = axis_gradient(48);
        let mut shifted = base.clone();
        let mut gained = base.clone();
        stage_ca_fringes(&mut shifted, 48, 48, &ca_lens(1.0, 0.05, 0.0));
        stage_ca_fringes(&mut gained, 48, 48, &ca_lens(1.0, 0.05, 0.05));
        let p = (40, 8);
        let (s, g) = (ca_red_at(&shifted, 48, p.0, p.1), ca_red_at(&gained, 48, p.0, p.1));
        assert!(g < s, "ra did not add displacement: {s} -> {g}");
    }

    #[test]
    fn purple_fringing_runs_after_the_ca_resample() {
        let mut base = vec![0u8; 48 * 48 * 4];
        for y in 0..48 {
            for x in 0..48 {
                let i = (y * 48 + x) * 4;
                if x < 24 {
                    base[i] = 220; // purple side: red and blue high, green low
                    base[i + 2] = 220;
                }
                base[i + 1] = 10;
                base[i + 3] = 255;
            }
        }
        let mut ca_only = base.clone();
        stage_ca_fringes(&mut ca_only, 48, 48, &ca_lens(1.0, 0.1, 0.0));
        let mut purple_only = base.clone();
        stage_purple_fringing(&mut purple_only, 48, 48, crate::mock::purple_norm(50.0));
        let mut both = ca_only.clone();
        stage_purple_fringing(&mut both, 48, 48, crate::mock::purple_norm(50.0));
        assert_ne!(both, ca_only, "purple fringing did not run after CA");
        assert_ne!(both, purple_only, "CA did not change what purple saw");
    }
}
