//! Fujifilm X-Trans 6x6 CFA demosaicing (three-stage structure: directional
//! green interpolation -> green zipper suppression -> R-G / B-G chroma rebuild).

use rayon::prelude::*;

/// Standard Fujifilm X-Trans 6x6 repeating CFA pattern (0=R, 1=G, 2=B).
const XTRANS_STD: [[u8; 6]; 6] = [
    [1, 2, 1, 1, 0, 1],
    [0, 1, 0, 2, 1, 2],
    [1, 2, 1, 1, 0, 1],
    [1, 0, 1, 1, 2, 1],
    [2, 1, 2, 0, 1, 0],
    [1, 0, 1, 1, 2, 1],
];

#[inline]
fn xtrans_color(x: usize, y: usize) -> u8 {
    XTRANS_STD[y % 6][x % 6]
}

/// Demosaics a 14-bit X-Trans active plane into linear 16-bit camera RGB
/// (`0.0..=65535.0`) with black subtraction and white-balance scaling.
pub(crate) fn demosaic_xtrans(
    raw: &[u16],
    raw_width: usize,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
    black: f32,
    wb: [f32; 3],
) -> Option<Vec<[f32; 3]>> {
    if width < 8 || height < 8 || raw.len() < (top + height) * raw_width {
        return None;
    }
    let scale16 = 4.0f32; // 14-bit -> 16-bit
    let mut plane = vec![0.0f32; width * height];
    plane.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let src = (top + y) * raw_width + left;
        for x in 0..width {
            let c = xtrans_color(x, y) as usize;
            let v = (f32::from(raw[src + x]) - black).max(0.0) * scale16 * wb[c];
            row[x] = v.min(65535.0);
        }
    });

    let max_x = (width - 1) as isize;
    let max_y = (height - 1) as isize;
    let at = |buf: &[f32], x: isize, y: isize| -> f32 {
        buf[y.clamp(0, max_y) as usize * width + x.clamp(0, max_x) as usize]
    };

    // Stage 1: directional green interpolation on non-G sites.
    let mut green = vec![0.0f32; width * height];
    green.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let yi = y as isize;
        for x in 0..width {
            if xtrans_color(x, y) == 1 {
                row[x] = plane[y * width + x];
                continue;
            }
            let xi = x as isize;
            let mut sum_h = 0.0f32;
            let mut cnt_h = 0.0f32;
            let mut sum_v = 0.0f32;
            let mut cnt_v = 0.0f32;
            let mut sum_d1 = 0.0f32;
            let mut cnt_d1 = 0.0f32;
            let mut sum_d2 = 0.0f32;
            let mut cnt_d2 = 0.0f32;
            for d in [-1isize, 1] {
                let nx = (xi + d).clamp(0, max_x) as usize;
                let ny = (yi + d).clamp(0, max_y) as usize;
                if xtrans_color(nx, y) == 1 {
                    sum_h += plane[y * width + nx];
                    cnt_h += 1.0;
                }
                if xtrans_color(x, ny) == 1 {
                    sum_v += plane[ny * width + x];
                    cnt_v += 1.0;
                }
                if xtrans_color(nx, ny) == 1 {
                    sum_d1 += plane[ny * width + nx];
                    cnt_d1 += 1.0;
                }
                let ny2 = (yi - d).clamp(0, max_y) as usize;
                if xtrans_color(nx, ny2) == 1 {
                    sum_d2 += plane[ny2 * width + nx];
                    cnt_d2 += 1.0;
                }
            }
            let gh = (at(&plane, xi - 1, yi) - at(&plane, xi + 1, yi)).abs() + 1.0;
            let gv = (at(&plane, xi, yi - 1) - at(&plane, xi, yi + 1)).abs() + 1.0;
            let gd1 = (at(&plane, xi - 1, yi - 1) - at(&plane, xi + 1, yi + 1)).abs() + 1.0;
            let gd2 = (at(&plane, xi - 1, yi + 1) - at(&plane, xi + 1, yi - 1)).abs() + 1.0;

            let mut num = 0.0f32;
            let mut den = 0.0f32;
            if cnt_h > 0.0 {
                let w = 1.0 / gh;
                num += (sum_h / cnt_h) * w;
                den += w;
            }
            if cnt_v > 0.0 {
                let w = 1.0 / gv;
                num += (sum_v / cnt_v) * w;
                den += w;
            }
            if cnt_d1 > 0.0 {
                let w = 0.8 / gd1;
                num += (sum_d1 / cnt_d1) * w;
                den += w;
            }
            if cnt_d2 > 0.0 {
                let w = 0.8 / gd2;
                num += (sum_d2 / cnt_d2) * w;
                den += w;
            }
            row[x] = if den > 0.0 { num / den } else { plane[y * width + x] };
        }
    });

    // Stage 2: fixGreen (mild 3x3 cross median clamp on interpolated G sites).
    let mut green_fixed = green.clone();
    green_fixed.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let yi = y as isize;
        for x in 0..width {
            if xtrans_color(x, y) == 1 {
                continue;
            }
            let xi = x as isize;
            let c = green[y * width + x];
            let n0 = at(&green, xi - 1, yi);
            let n1 = at(&green, xi + 1, yi);
            let n2 = at(&green, xi, yi - 1);
            let n3 = at(&green, xi, yi + 1);
            let lo = n0.min(n1).min(n2).min(n3);
            let hi = n0.max(n1).max(n2).max(n3);
            row[x] = c.clamp(lo, hi);
        }
    });

    // Stage 3: R-G and B-G colour-difference interpolation over 5x5 neighbourhood.
    let mut rgb = vec![[0.0f32; 3]; width * height];
    rgb.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let yi = y as isize;
        for x in 0..width {
            let xi = x as isize;
            let g = green_fixed[y * width + x];
            let mut sum_rg = 0.0f32;
            let mut wt_r = 0.0f32;
            let mut sum_bg = 0.0f32;
            let mut wt_b = 0.0f32;
            for dy in -2isize..=2 {
                let ny = (yi + dy).clamp(0, max_y) as usize;
                for dx in -2isize..=2 {
                    let nx = (xi + dx).clamp(0, max_x) as usize;
                    let c = xtrans_color(nx, ny);
                    if c == 1 {
                        continue;
                    }
                    let dist2 = (dx * dx + dy * dy) as f32;
                    let w = 1.0 / (1.0 + dist2);
                    let diff = plane[ny * width + nx] - green_fixed[ny * width + nx];
                    if c == 0 {
                        sum_rg += diff * w;
                        wt_r += w;
                    } else {
                        sum_bg += diff * w;
                        wt_b += w;
                    }
                }
            }
            let r = if xtrans_color(x, y) == 0 {
                plane[y * width + x]
            } else if wt_r > 0.0 {
                (g + sum_rg / wt_r).clamp(0.0, 65535.0)
            } else {
                g
            };
            let b = if xtrans_color(x, y) == 2 {
                plane[y * width + x]
            } else if wt_b > 0.0 {
                (g + sum_bg / wt_b).clamp(0.0, 65535.0)
            } else {
                g
            };
            row[x] = [r, g.clamp(0.0, 65535.0), b];
        }
    });
    Some(rgb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xtrans_demosaic_recovers_flat_neutral_and_chroma_planes() {
        let (w, h) = (36usize, 36usize);
        let mut raw = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..w {
                raw[y * w + x] = match xtrans_color(x, y) {
                    0 => 3000, // R
                    1 => 4000, // G
                    _ => 2500, // B
                };
            }
        }
        let rgb = demosaic_xtrans(&raw, w, 0, 0, w, h, 1000.0, [1.5, 1.0, 2.0])
            .expect("X-Trans demosaic must succeed");
        let mid = rgb[18 * w + 18];
        // Expected: R=(3000-1000)*4*1.5=12000, G=(4000-1000)*4*1.0=12000, B=(2500-1000)*4*2.0=12000
        for ch in 0..3 {
            assert!((mid[ch] - 12000.0).abs() < 10.0, "ch {ch}: got {}", mid[ch]);
        }
    }
}
