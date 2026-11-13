//! PF407 / row 112 `LengthenLegRefine` UV-map producer.
//!
//! Recovered chain (pixcake.exe `a84519da…`):
//!   `0x4B9C600` EffectLengthenLegRefineCell::Initialize creates two 0x2C8
//!   reshape objects at `cell+0x280/+0x288`, sets second `+0x2A9=1`
//!   (ENABLE_UVMAP), `+0x2AA=[+0x34B]` (UVMAP_BA) and zeroes `+0x2A8`
//!   (TS_CLAMP_EDGE off), then stores 10 float2 vertices at `cell+0x2E0` and
//!   10 float2 UVs at `cell+0x290` (`0x4BA05D0`).
//!
//!   `0x4B9FEF0` (cell vtable slot 15) calls the reshape draw:
//!     rdx = cell+0x290 (hT stream), r8 = cell+0x2E0 (iV stream), r9 = 10
//!   `0x611BB50` (reshape vtable slot 36) uploads both streams into one VBO
//!   (iV at offset 0, hT at offset 80, stride 8), binds program
//!   `[reshape+0x1F8]` via `0x34A0F50`, sets `m/F/O`, and draws with
//!   `0x34A1A60(mode=5 GL_TRIANGLE_STRIP, first=0, count=10)`.
//!
//! The fragment is the type 63 `ReshapeNormal.frag` UVMAP branch, so the draw
//! writes a displacement/UV map, not the final colour image. The downstream
//! colour sampler is still open; this file ports exactly the recovered
//! mesh -> draw -> UV-map producer and takes the five leg columns as explicit
//! model output. It fabricates no landmark, mask or control point.

use crate::{Image, Refusal};

pub const COLUMN_COUNT: usize = 5;
pub const VERTEX_COUNT: usize = COLUMN_COUNT * 2;
pub const DRAW_MODE_TRIANGLE_STRIP: u32 = 5;

/// One leg strip column. `sample_u` is the `hT` (source) U, `clip_x` is the
/// `iV` clip-space X. Top row uses v=1/clip-y=+1, bottom v=0/clip-y=-1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LegColumn {
    pub sample_u: f32,
    pub clip_x: f32,
}

/// Rasterizes the recovered 10-vertex strip through the `ReshapeNormal` UVMAP
/// fragment. `uv_map_ba` is `[cell+0x34B]`; the RG form is the other branch.
pub fn apply(source: &Image, columns: &[LegColumn], uv_map_ba: bool) -> Result<Image, Refusal> {
    source.validate().map_err(|_| Refusal::ShapeMismatch("source"))?;
    if columns.len() != COLUMN_COUNT {
        return Err(Refusal::ShapeMismatch("leg columns"));
    }
    let mut previous = f32::NEG_INFINITY;
    for column in columns {
        if !column.sample_u.is_finite() || !column.clip_x.is_finite() {
            return Err(Refusal::ValueOutOfRange("leg columns"));
        }
        if !(0.0..=1.0).contains(&column.sample_u) {
            return Err(Refusal::ValueOutOfRange("sample_u"));
        }
        if !(-1.0..=1.0).contains(&column.clip_x) {
            return Err(Refusal::ValueOutOfRange("clip_x"));
        }
        if column.clip_x <= previous {
            return Err(Refusal::ValueOutOfRange("clip_x order"));
        }
        previous = column.clip_x;
    }

    let width = source.width as usize;
    let height = source.height as usize;
    let mut screen = [(0.0f32, 0.0f32); VERTEX_COUNT];
    let mut h_t = [(0.0f32, 0.0f32); VERTEX_COUNT];
    for (index, column) in columns.iter().enumerate() {
        let top = index * 2;
        let bottom = top + 1;
        screen[top] = (
            (column.clip_x + 1.0) * 0.5 * width as f32,
            (1.0f32 + 1.0) * 0.5 * height as f32,
        );
        screen[bottom] = (
            (column.clip_x + 1.0) * 0.5 * width as f32,
            (-1.0f32 + 1.0) * 0.5 * height as f32,
        );
        h_t[top] = (column.sample_u, 1.0);
        h_t[bottom] = (column.sample_u, 0.0);
    }

    let mut output = source.pixels.clone();
    for first in 0..VERTEX_COUNT - 2 {
        let i0 = first;
        let i1 = first + 1;
        let i2 = first + 2;
        let (x0, y0) = screen[i0];
        let (x1, y1) = screen[i1];
        let (x2, y2) = screen[i2];
        let area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
        if area.abs() < 1e-8 {
            continue;
        }
        let inv_area = 1.0 / area;
        let min_x = 0.max(x0.min(x1).min(x2).floor() as i64) as usize;
        let max_x = (width as i64 - 1).min(x0.max(x1).max(x2).ceil() as i64).max(0) as usize;
        let min_y = 0.max(y0.min(y1).min(y2).floor() as i64) as usize;
        let max_y = (height as i64 - 1).min(y0.max(y1).max(y2).ceil() as i64).max(0) as usize;
        if min_x > max_x || min_y > max_y {
            continue;
        }

        for py in min_y..=max_y {
            let sample_y = py as f32 + 0.5;
            for px in min_x..=max_x {
                let sample_x = px as f32 + 0.5;
                let mut l0 =
                    ((x1 - sample_x) * (y2 - sample_y) - (y1 - sample_y) * (x2 - sample_x))
                        * inv_area;
                let mut l1 =
                    ((x2 - sample_x) * (y0 - sample_y) - (y2 - sample_y) * (x0 - sample_x))
                        * inv_area;
                let mut l2 = 1.0 - l0 - l1;
                if l0 < -1e-6 || l1 < -1e-6 || l2 < -1e-6 {
                    continue;
                }
                let sum = l0 + l1 + l2;
                if sum <= 0.0 {
                    continue;
                }
                l0 = l0.max(0.0) / sum;
                l1 = l1.max(0.0) / sum;
                l2 = l2.max(0.0) / sum;

                let src_uv = (
                    l0 * h_t[i0].0 + l1 * h_t[i1].0 + l2 * h_t[i2].0,
                    l0 * h_t[i0].1 + l1 * h_t[i1].1 + l2 * h_t[i2].1,
                );
                let dst_uv = (
                    (px as f32 + 0.5) / width as f32,
                    (py as f32 + 0.5) / height as f32,
                );
                let color = uv_map_fragment(source, src_uv, dst_uv, uv_map_ba);
                let offset = (py * width + px) * 4;
                for channel in 0..4 {
                    output[offset + channel] = quantize(color[channel]);
                }
            }
        }
    }

    Image::new(source.width, source.height, output).ok_or(Refusal::ShapeMismatch("output"))
}

fn uv_map_fragment(
    source: &Image,
    src_uv: (f32, f32),
    dst_uv: (f32, f32),
    uv_map_ba: bool,
) -> [f32; 4] {
    let sampled = sample_bilinear(source, src_uv.0, src_uv.1);
    let current = sample_bilinear(source, dst_uv.0, dst_uv.1);
    if uv_map_ba {
        [
            current[0],
            current[1],
            sampled[2] + (src_uv.0 - dst_uv.0),
            sampled[3] + (src_uv.1 - dst_uv.1),
        ]
    } else {
        [
            sampled[0] + (src_uv.0 - dst_uv.0),
            sampled[1] + (src_uv.1 - dst_uv.1),
            current[2],
            current[3],
        ]
    }
}

fn sample_bilinear(source: &Image, u: f32, v: f32) -> [f32; 4] {
    let width = source.width as f32;
    let height = source.height as f32;
    let x = (u * width - 0.5).clamp(0.0, width - 1.0);
    let y = (v * height - 0.5).clamp(0.0, height - 1.0);
    let x0 = x.floor();
    let y0 = y.floor();
    let x1 = (x0 + 1.0).min(width - 1.0);
    let y1 = (y0 + 1.0).min(height - 1.0);
    let fx = x - x0;
    let fy = y - y0;
    let mut out = [0.0f32; 4];
    for channel in 0..4 {
        let top = lerp(
            texel(source, x0 as usize, y0 as usize, channel),
            texel(source, x1 as usize, y0 as usize, channel),
            fx,
        );
        let bottom = lerp(
            texel(source, x0 as usize, y1 as usize, channel),
            texel(source, x1 as usize, y1 as usize, channel),
            fx,
        );
        out[channel] = lerp(top, bottom, fy);
    }
    out
}

fn texel(source: &Image, x: usize, y: usize, channel: usize) -> f32 {
    source.pixels[(y * source.width as usize + x) * 4 + channel] as f32 / 255.0
}

fn quantize(value: f32) -> u8 {
    (value * 255.0).clamp(0.0, 255.0).round_ties_even() as u8
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
