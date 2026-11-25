//! Luma-plane statistics: planes, 3x3 means, box blurs, and the histogram.
//!
//! These are pure pixel/plane helpers over byte buffers; `histogram` returns
//! the value type [`crate::Histogram`]. They were moved out of `mock::process`
//! byte-identically, so they no longer depend on `Source`, `Plan`, caches, or
//! any GPU path. The mock renderer imports them back, keeping its existing
//! call sites and tests on the unqualified names.

use rayon::prelude::*;

use crate::Histogram;

/// Luminance of the frame, 0..255.
pub(crate) fn luma_plane(rgba: &[u8], w: usize, h: usize) -> Vec<f32> {
    let mut plane = vec![0f32; w * h];
    plane.par_iter_mut().enumerate().for_each(|(i, p)| {
        let j = i * 4;
        *p = 0.2126 * rgba[j] as f32 + 0.7152 * rgba[j + 1] as f32 + 0.0722 * rgba[j + 2] as f32;
    });
    plane
}

/// 3x3 mean of a plane.
pub(crate) fn mean3x3(src: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut out = vec![0f32; w * h];
    if w == 0 || h == 0 {
        return out;
    }
    out.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut sum = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let xx = (x as i32 + dx).clamp(0, w as i32 - 1) as usize;
                    let yy = (y as i32 + dy).clamp(0, h as i32 - 1) as usize;
                    sum += src[yy * w + xx];
                }
            }
            row[x] = sum / 9.0;
        }
    });
    out
}

/// Three box passes approximate a Gaussian of a comparable radius.
pub(crate) fn box_blur3(src: &[f32], w: usize, h: usize, r: i32) -> Vec<f32> {
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

pub(crate) fn box_blur_h(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: i32) {
    let inv = 1.0 / (2 * r + 1) as f32;
    if w == 0 || h == 0 {
        return;
    }
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let base = y * w;
        for x in 0..w {
            let mut sum = 0.0f32;
            for k in -r..=r {
                sum += src[base + (x as i32 + k).clamp(0, w as i32 - 1) as usize];
            }
            row[x] = sum * inv;
        }
    });
}

pub(crate) fn box_blur_v(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: i32) {
    let inv = 1.0 / (2 * r + 1) as f32;
    if w == 0 || h == 0 {
        return;
    }
    dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut sum = 0.0f32;
            for k in -r..=r {
                sum += src[(y as i32 + k).clamp(0, h as i32 - 1) as usize * w + x];
            }
            row[x] = sum * inv;
        }
    });
}

pub(crate) fn histogram(rgba: &[u8]) -> Histogram {
    // Fixed arrays keep the per-chunk fold off the heap; a ray working on a
    // 1400 px preview otherwise allocates thousands of small vectors.
    type Bins = ([u32; 256], [u32; 256], [u32; 256], [u32; 256], u64, u64);
    let empty = || -> Bins { ([0; 256], [0; 256], [0; 256], [0; 256], 0, 0) };
    let chunk_bytes = if rgba.is_empty() {
        4
    } else {
        rgba.len()
            .div_ceil(rayon::current_num_threads().max(1) * 4)
            .max(4 * 1024)
            .next_multiple_of(4)
    };
    let (r, g, b, l, lo, hi) = rgba
        .par_chunks(chunk_bytes)
        .fold(
            empty,
            |mut acc, chunk| {
                for p in chunk.chunks_exact(4) {
                    acc.0[p[0] as usize] += 1;
                    acc.1[p[1] as usize] += 1;
                    acc.2[p[2] as usize] += 1;
                    let y = (0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32) as usize;
                    acc.3[y.min(255)] += 1;
                    if p[0] <= 1 && p[1] <= 1 && p[2] <= 1 {
                        acc.4 += 1;
                    }
                    if p[0] >= 254 || p[1] >= 254 || p[2] >= 254 {
                        acc.5 += 1;
                    }
                }
                acc
            },
        )
        .reduce(empty, |mut a, b| {
                for i in 0..256 {
                    a.0[i] += b.0[i];
                    a.1[i] += b.1[i];
                    a.2[i] += b.2[i];
                    a.3[i] += b.3[i];
                }
                a.4 += b.4;
                a.5 += b.5;
                a
            });
    let total = (rgba.len() / 4).max(1) as f32;
    Histogram {
        red: r.to_vec(),
        green: g.to_vec(),
        blue: b.to_vec(),
        luma: l.to_vec(),
        clipped_shadows: lo as f32 / total,
        clipped_highlights: hi as f32 / total,
    }
}
