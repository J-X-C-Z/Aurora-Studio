//! PF407 two-stage CPU path: the 10-vertex strip writes a UV offset field, then
//! the type143 colour pass samples the source at `uv + offset`.
//!
//! The offset must stay in float: `src_uv - dst_uv` is signed, and an RGBA8
//! carrier clamps negatives to zero (which silently turns the effect into an
//! identity). Host writeback into the engine's FBO is a separate open contract.

use crate::kernels::leg_refine_uv::LegColumn;
use crate::{EffectMeta, Image, Refusal};

pub const META: &[EffectMeta] = &[EffectMeta {
    pf: 407,
    key: "",
    label: "Lengthen leg refine (two-stage UV warp)",
    access: crate::Access::Cpu,
    inputs: &[
        "colour source (RGBA8)",
        "five leg columns (sample_u 0..1, clip_x -1..1, increasing)",
        "field channel BA / mp=1 when supplied as texture",
        "catmull size (F,O px, optional)",
    ],
}];
const COLUMN_COUNT: usize = 5;
const VERTEX_COUNT: usize = COLUMN_COUNT * 2;

/// Signed UV offset per pixel in `[dx, dy]`; uncovered pixels stay zero.
pub fn offset_field(source_size: (u32, u32), columns: &[LegColumn])
    -> Result<Vec<[f32; 2]>, Refusal>
{
    offset_field_with_threads(source_size, columns, auto_threads(source_size))
}

/// Above this many pixels the strip is worth splitting across rows. Measured on
/// 256^2 / 384^2 / 512^2 / 1024^2 / 2048x1024; see work/backend-r24/o3h/REPORT.md.
/// 256^2 loses (0.37x) and 384^2 breaks even (0.9-1.1x); 512^2 wins (1.4x),
/// so the cut sits at 512^2.
const PARALLEL_MIN_PIXELS: usize = 262_144;

fn auto_threads(source_size: (u32, u32)) -> usize {
    let pixels = source_size.0 as usize * source_size.1 as usize;
    if pixels < PARALLEL_MIN_PIXELS {
        return 1;
    }
    std::thread::available_parallelism()
        .map(|threads| threads.get())
        .unwrap_or(1)
}

/// `threads == 1` is the serial reference. Every row still walks the eight
/// triangles in the original order, so each pixel keeps its last-writer value
/// and the float sequence is unchanged.
pub(crate) fn offset_field_with_threads(
    source_size: (u32, u32),
    columns: &[LegColumn],
    threads: usize,
) -> Result<Vec<[f32; 2]>, Refusal> {
    let (width, height) = (source_size.0 as usize, source_size.1 as usize);
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("source"));
    }
    if columns.len() != COLUMN_COUNT {
        return Err(Refusal::ShapeMismatch("leg columns"));
    }
    let mut previous = f32::NEG_INFINITY;
    for column in columns {
        if !column.sample_u.is_finite() || !column.clip_x.is_finite() {
            return Err(Refusal::ValueOutOfRange("leg columns"));
        }
        if !(0.0..=1.0).contains(&column.sample_u) || !(-1.0..=1.0).contains(&column.clip_x) {
            return Err(Refusal::ValueOutOfRange("leg columns"));
        }
        if column.clip_x <= previous {
            return Err(Refusal::ValueOutOfRange("clip_x order"));
        }
        previous = column.clip_x;
    }

    let mut screen = [(0.0f32, 0.0f32); VERTEX_COUNT];
    let mut h_t = [(0.0f32, 0.0f32); VERTEX_COUNT];
    for (index, column) in columns.iter().enumerate() {
        let top = index * 2;
        let x = (column.clip_x + 1.0) * 0.5 * width as f32;
        screen[top] = (x, height as f32);
        screen[top + 1] = (x, 0.0);
        h_t[top] = (column.sample_u, 1.0);
        h_t[top + 1] = (column.sample_u, 0.0);
    }

    let mut field = vec![[0.0f32; 2]; width * height];
    let threads = threads.max(1).min(height);
    if threads <= 1 {
        rasterize_rows(&mut field, 0, height, width, height, &screen, &h_t);
        return Ok(field);
    }
    let rows_per_thread = height.div_ceil(threads);
    std::thread::scope(|scope| {
        for (index, chunk) in field.chunks_mut(rows_per_thread * width).enumerate() {
            let row0 = index * rows_per_thread;
            let rows = chunk.len() / width;
            let screen = &screen;
            let h_t = &h_t;
            scope.spawn(move || rasterize_rows(chunk, row0, rows, width, height, screen, h_t));
        }
    });
    Ok(field)
}

fn rasterize_rows(
    field: &mut [[f32; 2]],
    row0: usize,
    rows: usize,
    width: usize,
    height: usize,
    screen: &[(f32, f32); VERTEX_COUNT],
    h_t: &[(f32, f32); VERTEX_COUNT],
) {
    if rows == 0 {
        return;
    }
    let last_row = row0 + rows - 1;
    for first in 0..VERTEX_COUNT - 2 {
        let (i0, i1, i2) = (first, first + 1, first + 2);
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
        let min_y = min_y.max(row0);
        let max_y = max_y.min(last_row);
        if min_x > max_x || min_y > max_y {
            continue;
        }
        for py in min_y..=max_y {
            for px in min_x..=max_x {
                let sx = px as f32 + 0.5;
                let sy = py as f32 + 0.5;
                let mut l0 = ((x1 - sx) * (y2 - sy) - (y1 - sy) * (x2 - sx)) * inv_area;
                let mut l1 = ((x2 - sx) * (y0 - sy) - (y2 - sy) * (x0 - sx)) * inv_area;
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
                let src_u = l0 * h_t[i0].0 + l1 * h_t[i1].0 + l2 * h_t[i2].0;
                let src_v = l0 * h_t[i0].1 + l1 * h_t[i1].1 + l2 * h_t[i2].1;
                let dst_u = sx / width as f32;
                let dst_v = sy / height as f32;
                field[(py - row0) * width + px] = [src_u - dst_u, src_v - dst_v];
            }
        }
    }
}

/// Colour pass: sample `source` at `uv + offset`, ties-to-even store.
pub fn apply(
    source: &Image,
    columns: &[LegColumn],
    catmull_size: Option<(f32, f32)>,
) -> Result<Image, Refusal> {
    source.validate()?;
    if let Some((width, height)) = catmull_size {
        if !width.is_finite() || !height.is_finite() || width <= 0.0 || height <= 0.0 {
            return Err(Refusal::ValueOutOfRange("catmull size"));
        }
    }
    let field = offset_field((source.width, source.height), columns)?;
    let width = source.width as usize;
    let height = source.height as usize;
    let mut pixels = Vec::with_capacity(source.pixels.len());
    for y in 0..height {
        for x in 0..width {
            let uv = ((x as f32 + 0.5) / width as f32, (y as f32 + 0.5) / height as f32);
            let delta = field[y * width + x];
            let color = match catmull_size {
                Some(size) => crate::kernels::uvmap_color::sample_catmull_rom(
                    source, uv.0 + delta[0], uv.1 + delta[1], size),
                None => crate::kernels::uvmap_color::sample_bilinear(
                    source, uv.0 + delta[0], uv.1 + delta[1]),
            };
            for channel in color {
                pixels.push((channel * 255.0).clamp(0.0, 255.0).round_ties_even() as u8);
            }
        }
    }
    Image::new(source.width, source.height, pixels).ok_or(Refusal::ShapeMismatch("output"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source() -> Image {
        Image::new(8, 4, (0..8 * 4 * 4).map(|i| (i * 29 % 256) as u8).collect()).unwrap()
    }

    fn columns(clip: [f32; 5]) -> Vec<LegColumn> {
        (0..5).map(|i| LegColumn { sample_u: i as f32 / 4.0, clip_x: clip[i] }).collect()
    }

    #[test]
    fn signed_offsets_survive_and_move_pixels() {
        let src = source();
        let field = offset_field((8, 4), &columns([0.30, 0.35, 0.40, 0.45, 0.50])).unwrap();
        assert!(field.iter().any(|d| d[0] < 0.0), "negative offsets must survive");
        let out = apply(&src, &columns([0.30, 0.35, 0.40, 0.45, 0.50]), None).unwrap();
        let changed = out.pixels.chunks_exact(4).zip(src.pixels.chunks_exact(4))
            .filter(|(a, b)| a != b).count();
        assert!(changed > 0, "offset field must change pixels");
        // The type143 fragment writes the sampled texel including alpha, so this
        // stage may move alpha with the RGB; it does not "preserve" it.
        assert_eq!((out.width, out.height), (src.width, src.height));
    }

    #[test]
    fn malformed_inputs_refuse() {
        let src = source();
        assert_eq!(apply(&src, &columns([0.0; 5])[..4], None),
            Err(Refusal::ShapeMismatch("leg columns")));
        assert_eq!(apply(&src, &columns([0.30, 0.35, 0.40, 0.45, 0.50]), Some((0.0, 4.0))),
            Err(Refusal::ValueOutOfRange("catmull size")));
        assert_eq!(offset_field((8, 4), &columns([0.5, 0.4, 0.3, 0.2, 0.1])),
            Err(Refusal::ValueOutOfRange("clip_x order")));
    }

    fn bits(field: &[[f32; 2]]) -> Vec<[u32; 2]> {
        field
            .iter()
            .map(|d| [d[0].to_bits(), d[1].to_bits()])
            .collect()
    }

    #[test]
    fn row_split_is_bit_identical_to_serial() {
        for clip in [
            [-0.90, -0.45, 0.05, 0.50, 0.90],
            [0.30, 0.35, 0.40, 0.45, 0.50],
            [-0.02, -0.01, 0.0, 0.01, 0.02],
        ] {
            let columns = columns(clip);
            let serial = offset_field_with_threads((64, 96), &columns, 1).unwrap();
            for threads in [2, 3, 7, 16] {
                let split = offset_field_with_threads((64, 96), &columns, threads).unwrap();
                assert_eq!(bits(&serial), bits(&split), "clip {clip:?} threads {threads}");
            }
        }
    }

    #[test]
    fn rejections_do_not_depend_on_thread_count() {
        let bad = columns([0.5, 0.4, 0.3, 0.2, 0.1]);
        for threads in [1, 4] {
            assert_eq!(
                offset_field_with_threads((8, 4), &bad, threads),
                Err(Refusal::ValueOutOfRange("clip_x order"))
            );
            assert_eq!(
                offset_field_with_threads((0, 4), &columns([0.1, 0.2, 0.3, 0.4, 0.5]), threads),
                Err(Refusal::ShapeMismatch("source"))
            );
        }
    }
}
