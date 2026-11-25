//! Pure frame quantisers and band helpers lifted out of the renderer.
//!
//! Bilinear preview upsampling, band-region mapping, and the `f32`/IO-frame to
//! `u8`/`u16` quantisers. They are value-in/value-out over plain slices and
//! [`crate::NormRect`]; none of them touch `Source`, `Plan`, the tile cache,
//! the GPU path, portrait producers or I/O.
//!
//! This is a mechanical move: the bodies, signatures, constants and arithmetic
//! order are unchanged from `mock::process`, which imports them back so its
//! existing call sites and tests keep the unqualified names.

use rayon::prelude::*;

use crate::NormRect;

/// Bilinear upsample. `image`'s resize costs more than the half-size render it
/// would be scaling up, so the fast path does it here in parallel.
pub(crate) fn upsample(rgba: &[u8], w: u32, h: u32, tw: u32, th: u32) -> Vec<u8> {
    // The horizontal tap positions are the same for every row.
    let xs: Vec<(u32, u32, f32)> = (0..tw)
        .map(|x| {
            let sx = ((x as f32 + 0.5) * w as f32 / tw as f32 - 0.5).clamp(0.0, w as f32 - 1.0);
            let x0 = sx.floor() as u32;
            (x0, (x0 + 1).min(w - 1), sx - x0 as f32)
        })
        .collect();
    let mut out = vec![0u8; (tw * th * 4) as usize];
    out.par_chunks_mut((tw * 4) as usize).enumerate().for_each(|(y, line)| {
        let sy = ((y as f32 + 0.5) * h as f32 / th as f32 - 0.5).clamp(0.0, h as f32 - 1.0);
        let y0 = sy.floor() as u32;
        let y1 = (y0 + 1).min(h - 1);
        let ty = sy - y0 as f32;
        for (x, &(x0, x1, tx)) in xs.iter().enumerate() {
            let at = |xx: u32, yy: u32| ((yy * w + xx) * 4) as usize;
            let (a, b, c, d) = (at(x0, y0), at(x1, y0), at(x0, y1), at(x1, y1));
            for k in 0..4 {
                let top = rgba[a + k] as f32 + (rgba[b + k] as f32 - rgba[a + k] as f32) * tx;
                let bottom = rgba[c + k] as f32 + (rgba[d + k] as f32 - rgba[c + k] as f32) * tx;
                let v = top + (bottom - top) * ty + 0.5;
                line[x * 4 + k] = v as u8;
            }
        }
    });
    out
}

/// Kernel-space region of output rows `y0 .. y0 + rows`. With a gain plane the
/// kernel never reads the normalised coordinates (only the disabled outer
/// banding does), so a band is bit-identical to the whole-frame call.
pub(crate) fn band_region(region: NormRect, y0: u32, rows: u32, out_h: u32) -> [f32; 4] {
    let top = region.y + y0 as f32 / out_h as f32 * region.height;
    let height = rows as f32 / out_h as f32 * region.height;
    [region.x, top, region.width, height]
}

/// Quantises a developed band (4 floats per pixel) into the 8-bit and optional
/// 16-bit output rows.
pub(crate) fn encode_io_rows(io: &[f32], width: usize, rgba: &mut [u8], rgba16: Option<&mut [u16]>) {
    let row = width * 4;
    match rgba16 {
        Some(wide) => {
            rgba.par_chunks_mut(row)
                .zip(wide.par_chunks_mut(row))
                .zip(io.par_chunks(width * 4))
                .for_each(|((line8, line16), src_row)| {
                    for (x, src) in src_row.chunks_exact(4).enumerate() {
                        let i = x * 4;
                        for k in 0..3 {
                            let v = src[k].clamp(0.0, 1.0);
                            line8[i + k] = (v * 255.0 + 0.5) as u8;
                            line16[i + k] = (v * 65535.0 + 0.5) as u16;
                        }
                        line8[i + 3] = 255;
                        line16[i + 3] = 65535;
                    }
                });
        }
        None => {
            rgba.par_chunks_mut(row)
                .zip(io.par_chunks(width * 4))
                .for_each(|(line8, src_row)| {
                    for (x, src) in src_row.chunks_exact(4).enumerate() {
                        let i = x * 4;
                        for k in 0..3 {
                            line8[i + k] = (src[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                        }
                        line8[i + 3] = 255;
                    }
                });
        }
    }
}

/// Quantises a finished f32 frame into the 8-bit preview and 16-bit export
/// buffers.
pub(crate) fn encode_f32_frame(rgb: &[[f32; 3]], rgba: &mut [u8], rgba16: Option<&mut [u16]>, w: usize) {
    let row = w * 4;
    match rgba16 {
        Some(wide_line) => {
            rgba.par_chunks_mut(row)
                .zip(wide_line.par_chunks_mut(row))
                .zip(rgb.par_chunks(w))
                .for_each(|((line8, line16), src_row)| {
                    for (x, &c) in src_row.iter().enumerate() {
                        let i = x * 4;
                        for k in 0..3 {
                            let v = c[k].clamp(0.0, 1.0);
                            line8[i + k] = (v * 255.0 + 0.5) as u8;
                            line16[i + k] = (v * 65535.0 + 0.5) as u16;
                        }
                        line8[i + 3] = 255;
                        line16[i + 3] = 65535;
                    }
                });
        }
        None => {
            rgba.par_chunks_mut(row)
                .zip(rgb.par_chunks(w))
                .for_each(|(line8, src_row)| {
                    for (x, &c) in src_row.iter().enumerate() {
                        let i = x * 4;
                        for k in 0..3 {
                            line8[i + k] = (c[k].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                        }
                        line8[i + 3] = 255;
                    }
                });
        }
    }
}

pub(crate) fn merge_wide_delta(wide: &mut [u16], before: &[u8], after: &[u8], width: usize) {
    if width == 0 {
        return;
    }
    let row = width * 4;
    wide.par_chunks_mut(row)
        .zip(before.par_chunks(row))
        .zip(after.par_chunks(row))
        .for_each(|((wide_line, base_line), after_line)| {
            for ((px, base), after) in wide_line
                .chunks_exact_mut(4)
                .zip(base_line.chunks_exact(4))
                .zip(after_line.chunks_exact(4))
            {
                for k in 0..3 {
                    px[k] = (px[k] as i32 + (after[k] as i32 - base[k] as i32) * 257)
                        .clamp(0, 65535) as u16;
                }
                px[3] = if after[3] == base[3] { px[3] } else { after[3] as u16 * 257 };
            }
        });
}
