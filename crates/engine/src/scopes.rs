//! Viewer scopes: luminance waveform, RGB parade and vectorscope densities.
//!
//! `Waveform` rows run from level 0 (black, drawn at the bottom) upward, so
//! `at(level, column)` is the density of luma `level` in image column group
//! `column`. `Vectorscope` is indexed `y * size + x`, origin top-left, neutral in
//! the centre, Cb up and Cr right. Densities are 0–255, scaled so the busiest
//! cell of the map is 255.

use rayon::prelude::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Waveform {
    pub columns: usize,
    pub levels: usize,
    pub data: Vec<u8>,
}

impl Waveform {
    pub fn at(&self, level: usize, column: usize) -> u8 {
        self.data[level * self.columns + column]
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vectorscope {
    pub size: usize,
    pub data: Vec<u8>,
}

impl Vectorscope {
    pub fn at(&self, x: usize, y: usize) -> u8 {
        self.data[y * self.size + x]
    }
}

/// Luminance waveform: `columns` groups across, `levels` rows of luma.
pub fn waveform(rgba: &[u8], w: usize, h: usize, columns: usize, levels: usize) -> Waveform {
    if columns == 0 || levels == 0 {
        return Waveform { columns, levels, data: Vec::new() };
    }
    let mut col_of_x = vec![0usize; w];
    for (x, slot) in col_of_x.iter_mut().enumerate() {
        *slot = x * columns / w;
    }
    let counts = accumulate(rgba, w, h, columns * levels, |bins, r, g, b, x| {
        let level = luma(r, g, b) as usize * levels / 256;
        bins[level * columns + col_of_x[x]] += 1;
    });
    Waveform { columns, levels, data: to_u8(&counts) }
}

/// RGB parades: three 256-wide waveforms side by side, R then G then B.
pub fn parades(rgba: &[u8], w: usize, h: usize) -> Waveform {
    const COLUMNS: usize = 256;
    const LEVELS: usize = 256;
    const TOTAL: usize = 3 * COLUMNS;
    let counts = accumulate(rgba, w, h, TOTAL * LEVELS, |bins, r, g, b, x| {
        for (channel, value) in [r, g, b].into_iter().enumerate() {
            let level = value as usize * LEVELS / 256;
            bins[level * TOTAL + channel * COLUMNS + x] += 1;
        }
    });
    Waveform { columns: TOTAL, levels: LEVELS, data: to_u8(&counts) }
}

/// Cb/Cr vectorscope on a `size`×`size` grid.
pub fn vectorscope(rgba: &[u8], w: usize, h: usize, size: usize) -> Vectorscope {
    if size == 0 {
        return Vectorscope { size, data: Vec::new() };
    }
    let half = (size / 2) as i32;
    let edge = size as i32 - 1;
    let counts = accumulate(rgba, w, h, size * size, |bins, r, g, b, _| {
        let r = r as i32;
        let g = g as i32;
        let b = b as i32;
        // BT.601 chroma, ±128 around neutral.
        let cb = (-117 * r - 394 * g + 512 * b) >> 10;
        let cr = (512 * r - 465 * g - 47 * b) >> 10;
        let x = (half + (cr * half).div_euclid(128)).clamp(0, edge);
        let y = (half - (cb * half).div_euclid(128)).clamp(0, edge);
        bins[(y * size as i32 + x) as usize] += 1;
    });
    Vectorscope { size, data: to_u8(&counts) }
}

fn luma(r: u8, g: u8, b: u8) -> u8 {
    ((54 * r as u32 + 183 * g as u32 + 19 * b as u32) >> 8) as u8
}

/// Splits the rows into one band per rayon task, counts into per-band bins and
/// sums them. `add` receives the band's counters, the pixel and its column.
fn accumulate<F>(rgba: &[u8], w: usize, h: usize, bins: usize, add: F) -> Vec<u32>
where
    F: Fn(&mut [u32], u8, u8, u8, usize) + Sync,
{
    let len = w.saturating_mul(h).saturating_mul(4);
    if w == 0 || h == 0 || bins == 0 || rgba.len() < len {
        return vec![0; bins];
    }
    let stride = w * 4;
    let bands = (rayon::current_num_threads() * 4).max(1);
    let rows = (h + bands - 1) / bands;
    rgba[..len]
        .par_chunks(rows * stride)
        .map(|band| {
            let mut local = vec![0u32; bins];
            for row in band.chunks_exact(stride) {
                for (x, px) in row.chunks_exact(4).enumerate() {
                    add(&mut local, px[0], px[1], px[2], x);
                }
            }
            local
        })
        .reduce(
            || vec![0u32; bins],
            |mut a, b| {
                for (slot, value) in a.iter_mut().zip(b) {
                    *slot += value;
                }
                a
            },
        )
}

fn to_u8(counts: &[u32]) -> Vec<u8> {
    let max = counts.iter().copied().max().unwrap_or(0).max(1);
    counts.iter().map(|&c| (c as u64 * 255 / max as u64) as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(w: usize, h: usize, px: [u8; 4]) -> Vec<u8> {
        px.iter().copied().cycle().take(w * h * 4).collect()
    }

    fn lit_rows(wave: &Waveform) -> Vec<usize> {
        (0..wave.levels)
            .filter(|&level| (0..wave.columns).any(|c| wave.at(level, c) > 0))
            .collect()
    }

    #[test]
    fn black_lands_on_bottom_row() {
        let wave = waveform(&image(8, 8, [0, 0, 0, 255]), 8, 8, 8, 256);
        assert_eq!(lit_rows(&wave), vec![0]);
        assert_eq!(wave.at(0, 3), 255);
    }

    #[test]
    fn white_lands_on_top_row() {
        let wave = waveform(&image(8, 8, [255, 255, 255, 255]), 8, 8, 8, 256);
        assert_eq!(lit_rows(&wave), vec![255]);
        assert_eq!(wave.at(255, 0), 255);
    }

    #[test]
    fn mid_gray_sits_at_middle_level() {
        let wave = waveform(&image(4, 4, [128, 128, 128, 255]), 4, 4, 4, 256);
        assert_eq!(lit_rows(&wave), vec![128]);
    }

    #[test]
    fn gray_vectorscope_is_centered() {
        let scope = vectorscope(&image(8, 8, [128, 128, 128, 255]), 8, 8, 64);
        assert_eq!(scope.data.iter().copied().max(), Some(255));
        assert_eq!(scope.at(32, 32), 255);
    }

    #[test]
    fn saturated_colors_point_the_right_way() {
        let peak = |px: [u8; 4]| {
            let scope = vectorscope(&image(8, 8, px), 8, 8, 64);
            let i = scope
                .data
                .iter()
                .enumerate()
                .max_by_key(|entry| *entry.1)
                .map(|entry| entry.0)
                .unwrap();
            ((i % 64) as i32, (i / 64) as i32)
        };
        let (x, _) = peak([255, 0, 0, 255]);
        assert!(x > 32, "red sits right of centre, got x={x}");
        let (x, y) = peak([0, 0, 255, 255]);
        assert!(y < 32, "blue sits above centre, got {x}/{y}");
        let (x, y) = peak([0, 255, 0, 255]);
        assert!(x < 32 && y > 32, "green sits left and below centre, got {x}/{y}");
    }

    #[test]
    fn empty_input_is_safe() {
        assert!(waveform(&[], 0, 0, 16, 16).data.iter().all(|&v| v == 0));
        assert!(waveform(&[], 4, 4, 16, 16).data.iter().all(|&v| v == 0));
        assert!(vectorscope(&[], 4, 4, 16).data.iter().all(|&v| v == 0));
        assert!(parades(&[], 0, 0).data.iter().all(|&v| v == 0));
    }

    #[test]
    fn size_parameters_are_respected() {
        assert!(waveform(&image(4, 4, [1, 2, 3, 0]), 4, 4, 0, 8).data.is_empty());
        assert!(waveform(&image(4, 4, [1, 2, 3, 0]), 4, 4, 8, 0).data.is_empty());
        assert!(vectorscope(&image(4, 4, [1, 2, 3, 0]), 4, 4, 0).data.is_empty());
        let one = waveform(&image(4, 4, [0, 0, 0, 255]), 4, 4, 1, 1);
        assert_eq!((one.columns, one.levels, one.at(0, 0)), (1, 1, 255));
        assert_eq!(vectorscope(&image(4, 4, [255, 0, 0, 255]), 4, 4, 1).at(0, 0), 255);
    }

    #[test]
    fn parades_has_three_sections() {
        let wave = parades(&image(8, 8, [255, 128, 0, 255]), 8, 8);
        assert_eq!((wave.columns, wave.levels), (768, 256));
        assert_eq!(wave.at(255, 0), 255, "red section");
        assert_eq!(wave.at(128, 256), 255, "green section");
        assert_eq!(wave.at(0, 512), 255, "blue section");
    }

    #[test]
    #[ignore]
    fn perf_probe() {
        let (w, h) = (2000, 1333);
        let rgba = image(w, h, [90, 140, 200, 255]);
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let wave = waveform(&rgba, w, h, 512, 256);
            let wave_ms = t.elapsed();
            let t = std::time::Instant::now();
            let scope = vectorscope(&rgba, w, h, 256);
            let vec_ms = t.elapsed();
            println!(
                "waveform {wave_ms:?} ({} B)  vectorscope {vec_ms:?} ({} B)",
                wave.data.len(),
                scope.data.len()
            );
        }
    }
}
