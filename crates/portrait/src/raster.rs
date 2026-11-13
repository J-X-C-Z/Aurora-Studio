//! Generic headless mesh rasterizer.
//!
//! Geometry contract shared by the NVIDIA `v2warp` paths and the later Metal
//! path:
//! * `dest` is destination geometry in framebuffer pixels, origin top-left,
//!   x right, y down. Pixel `(x, y)` is sampled at its center `(x + 0.5, y + 0.5)`.
//! * `source_uv` is normalized `[0, 1]` source texture coordinates; row 0 of
//!   the source image is `v = 0`.
//! * `ebo` is a `u16` triangle list in original index order; that order is the
//!   overlap resolution order.
//! * All math is `f32`. `area2 = cross(p1 - p0, p2 - p0)`; `|area2| <= 1e-12`
//!   is degenerate and drops the triangle.
//! * Inverse warp: only covered destination pixels are written. Every other
//!   byte is copied from the source, so alpha outside the mesh is preserved.
//! * Alpha is straight (not premultiplied) and is sampled and quantized exactly
//!   like RGB. Nothing is blended.

use crate::{Image, Refusal};

/// Coverage tolerances. The tile contract and the P30 reshape contract differ
/// in both numbers, so callers pass the one their byte evidence names.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tolerance {
    /// Minimum barycentric weight accepted as inside (`>=`).
    pub edge: f32,
    /// `|area2|` at or below this is degenerate.
    pub degenerate_area2: f32,
}

impl Tolerance {
    /// `o2i` tile capture contract: strict edge, `<= 1e-12`.
    pub const TILE: Self = Self { edge: 0.0, degenerate_area2: 1e-12 };
    /// P30 reshape contract: `-1e-6` edge band, `< 1e-8`.
    pub const RESHAPE: Self = Self { edge: -1e-6, degenerate_area2: 1e-8 };
}

/// Which signed area counts as front facing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Winding {
    PositiveAreaIsFront,
    NegativeAreaIsFront,
}

/// Which facing is discarded. `Front` drops front-facing triangles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cull {
    None,
    Front,
    Back,
}

/// How a sample exactly on a triangle edge is treated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    /// Every edge weight must be strictly positive.
    Exclusive,
    /// Zero edge weights count as covered.
    Inclusive,
}

/// How a pixel covered by more than one triangle is resolved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Overlap {
    /// Multi-face pixels stay untouched and keep the source bytes.
    Reject,
    /// Lowest original triangle index wins.
    First,
    /// Highest original triangle index wins.
    Last,
}

/// Raster rules. Every field is explicit; there are no defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rules {
    pub winding: Winding,
    pub cull: Cull,
    pub edge: Edge,
    pub overlap: Overlap,
}

/// Source texture wrap. `ClampToEdge` is the recovered shader default.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wrap {
    ClampToEdge,
    Repeat,
}

/// Explicit sampler: bilinear taps with the given wrap on both axes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sampler {
    pub wrap: Wrap,
}

/// Explicit mesh: destination geometry, source UV and EBO.
#[derive(Clone, Copy, Debug)]
pub struct Mesh<'a> {
    pub dest: &'a [[f32; 2]],
    pub source_uv: &'a [[f32; 2]],
    pub ebo: &'a [u16],
}

/// Rasterizes `mesh` into a copy of `source`.
///
/// Refuses malformed inputs instead of guessing: a wrong `Image` length, a
/// vertex-count mismatch, a non-triangle or out-of-range EBO, or a non-finite
/// vertex.
pub fn rasterize(
    source: &Image,
    mesh: Mesh<'_>,
    rules: Rules,
    sampler: Sampler,
) -> Result<Image, Refusal> {
    validate(source, &mesh)?;
    let (selected, coverage) = select(source, &mesh, rules, Tolerance::TILE);
    let width = source.width as usize;
    let height = source.height as usize;
    let mut pixels = source.pixels.clone();
    for py in 0..height {
        for px in 0..width {
            let p = py * width + px;
            let Some(triangle) = pick(&selected, &coverage, p, rules) else {
                continue;
            };
            let Some((u, v)) = source_uv_at(&mesh, rules, Tolerance::TILE, triangle, px, py) else {
                continue;
            };
            let color = sample_bilinear(source, u, v, sampler.wrap);
            let o = p * 4;
            for c in 0..4 {
                pixels[o + c] = quantize(color[c]);
            }
        }
    }
    Image::new(source.width, source.height, pixels).ok_or(Refusal::ShapeMismatch("output"))
}

/// Same geometry and coverage, but the covered pixel goes through the recovered
/// `ReshapeNormal` fragment (UVMAP / Catmull / edge clamp). `Repeat` sampling is
/// refused here because the fragment taps clamp.
pub fn rasterize_with_fragment(
    source: &Image,
    mesh: Mesh<'_>,
    rules: Rules,
    sampler: Sampler,
    edge: crate::raster_fragment::Edge,
    defines: crate::raster_fragment::Defines,
    clamp: crate::raster_fragment::Clamp,
    tolerance: Tolerance,
) -> Result<Image, Refusal> {
    validate(source, &mesh)?;
    if sampler.wrap == Wrap::Repeat {
        return Err(Refusal::MissingInput("fragment repeat wrap"));
    }
    let (selected, coverage) = select(source, &mesh, rules, tolerance);
    let width = source.width as usize;
    let height = source.height as usize;
    let mut pixels = source.pixels.clone();
    for py in 0..height {
        for px in 0..width {
            let p = py * width + px;
            let Some(triangle) = pick(&selected, &coverage, p, rules) else {
                continue;
            };
            let Some((u, v)) = source_uv_at(&mesh, rules, tolerance, triangle, px, py) else {
                continue;
            };
            let dst = [(px as f32 + 0.5) / width as f32, (py as f32 + 0.5) / height as f32];
            let color = crate::raster_fragment::evaluate(
                source, [u, v], dst, edge, defines, clamp);
            let stored = crate::raster_fragment::to_rgba8(color);
            pixels[p * 4..p * 4 + 4].copy_from_slice(&stored);
        }
    }
    Image::new(source.width, source.height, pixels).ok_or(Refusal::ShapeMismatch("output"))
}

/// Coverage selection shared by both writers.
fn select(source: &Image, mesh: &Mesh<'_>, rules: Rules, tolerance: Tolerance) -> (Vec<u32>, Vec<u8>) {
    let width = source.width as usize;
    let height = source.height as usize;
    let mut selected = vec![u32::MAX; width * height];
    let mut coverage = vec![0u8; width * height];
    for triangle in 0..mesh.ebo.len() / 3 {
        let i = [
            mesh.ebo[triangle * 3] as usize,
            mesh.ebo[triangle * 3 + 1] as usize,
            mesh.ebo[triangle * 3 + 2] as usize,
        ];
        if i[0] == i[1] || i[1] == i[2] || i[0] == i[2] {
            continue;
        }
        let Some(area2) = triangle_area2(mesh.dest, i, rules, tolerance) else {
            continue;
        };
        let (x0, x1, y0, y1) = bbox(mesh.dest, i, width, height);
        for py in y0..=y1 {
            for px in x0..=x1 {
                if weights(mesh.dest, i, area2, px as f32 + 0.5, py as f32 + 0.5, rules.edge, tolerance)
                    .is_none()
                {
                    continue;
                }
                let p = py * width + px;
                coverage[p] = coverage[p].saturating_add(1);
                if selected[p] == u32::MAX || rules.overlap == Overlap::Last {
                    selected[p] = triangle as u32;
                }
            }
        }
    }
    (selected, coverage)
}

fn pick(selected: &[u32], coverage: &[u8], p: usize, rules: Rules) -> Option<usize> {
    if selected[p] == u32::MAX {
        return None;
    }
    if coverage[p] > 1 && rules.overlap == Overlap::Reject {
        return None;
    }
    Some(selected[p] as usize)
}

fn source_uv_at(mesh: &Mesh<'_>, rules: Rules, tolerance: Tolerance, triangle: usize, px: usize, py: usize)
    -> Option<(f32, f32)>
{
    let i = [
        mesh.ebo[triangle * 3] as usize,
        mesh.ebo[triangle * 3 + 1] as usize,
        mesh.ebo[triangle * 3 + 2] as usize,
    ];
    let area2 = triangle_area2(mesh.dest, i, rules, tolerance)?;
    let w = weights(mesh.dest, i, area2, px as f32 + 0.5, py as f32 + 0.5, rules.edge, tolerance)?;
    let u = w[0] * mesh.source_uv[i[0]][0]
        + w[1] * mesh.source_uv[i[1]][0]
        + w[2] * mesh.source_uv[i[2]][0];
    let v = w[0] * mesh.source_uv[i[0]][1]
        + w[1] * mesh.source_uv[i[1]][1]
        + w[2] * mesh.source_uv[i[2]][1];
    Some((u, v))
}
/// Validates the public `Image` shape and every mesh array.
fn validate(source: &Image, mesh: &Mesh<'_>) -> Result<(), Refusal> {
    // `Image` fields are public for FFI callers, so the public shape check runs
    // at entry; non-empty dimensions are part of that contract.
    source.validate()?;
    if mesh.dest.len() != mesh.source_uv.len() {
        return Err(Refusal::ShapeMismatch("source_uv"));
    }
    if mesh.dest.len() < 3 {
        return Err(Refusal::ShapeMismatch("dest"));
    }
    if mesh.ebo.is_empty() {
        return Err(Refusal::MissingInput("ebo"));
    }
    if mesh.ebo.len() % 3 != 0 {
        return Err(Refusal::ShapeMismatch("ebo"));
    }
    if mesh.dest.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Refusal::ValueOutOfRange("dest"));
    }
    if mesh.source_uv.iter().flatten().any(|v| !v.is_finite()) {
        return Err(Refusal::ValueOutOfRange("source_uv"));
    }
    if mesh.ebo.iter().any(|&i| i as usize >= mesh.dest.len()) {
        return Err(Refusal::ValueOutOfRange("ebo index"));
    }
    Ok(())
}

/// Per-triangle facing and area. `None` means degenerate or culled.
fn triangle_area2(dest: &[[f32; 2]], i: [usize; 3], rules: Rules, tolerance: Tolerance) -> Option<f32> {
    let p0 = dest[i[0]];
    let p1 = dest[i[1]];
    let p2 = dest[i[2]];
    let area2 = (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]);
    if area2.abs() <= tolerance.degenerate_area2 {
        return None;
    }
    let front = match rules.winding {
        Winding::PositiveAreaIsFront => area2 > 0.0,
        Winding::NegativeAreaIsFront => area2 < 0.0,
    };
    let discard = match rules.cull {
        Cull::None => false,
        Cull::Front => front,
        Cull::Back => !front,
    };
    if discard {
        return None;
    }
    Some(area2)
}

/// Inclusive pixel range of a triangle, clamped to the image.
fn bbox(
    dest: &[[f32; 2]],
    i: [usize; 3],
    width: usize,
    height: usize,
) -> (usize, usize, usize, usize) {
    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    for &v in &i {
        min[0] = min[0].min(dest[v][0]);
        min[1] = min[1].min(dest[v][1]);
        max[0] = max[0].max(dest[v][0]);
        max[1] = max[1].max(dest[v][1]);
    }
    let clamp = |v: f32, limit: usize| -> usize {
        let v = v.clamp(0.0, (limit - 1) as f32);
        v as usize
    };
    let x0 = clamp((min[0] - 0.5).ceil(), width);
    let x1 = clamp((max[0] - 0.5).floor(), width);
    let y0 = clamp((min[1] - 0.5).ceil(), height);
    let y1 = clamp((max[1] - 0.5).floor(), height);
    (x0, x1, y0, y1)
}

/// Barycentrics of `(sx, sy)`, clamped non-negative and renormalized.
fn weights(
    dest: &[[f32; 2]],
    i: [usize; 3],
    area2: f32,
    sx: f32,
    sy: f32,
    edge: Edge,
    tolerance: Tolerance,
) -> Option<[f32; 3]> {
    let p0 = dest[i[0]];
    let p1 = dest[i[1]];
    let p2 = dest[i[2]];
    let b = [
        ((p1[0] - sx) * (p2[1] - sy) - (p1[1] - sy) * (p2[0] - sx)) / area2,
        ((p2[0] - sx) * (p0[1] - sy) - (p2[1] - sy) * (p0[0] - sx)) / area2,
        ((p0[0] - sx) * (p1[1] - sy) - (p0[1] - sy) * (p1[0] - sx)) / area2,
    ];
    let inside = match edge {
        Edge::Inclusive => b.iter().all(|&w| w >= tolerance.edge),
        Edge::Exclusive => b.iter().all(|&w| w > 0.0),
    };
    if !inside {
        return None;
    }
    let b = [b[0].max(0.0), b[1].max(0.0), b[2].max(0.0)];
    let sum = b[0] + b[1] + b[2];
    if !(sum > 0.0) {
        return None;
    }
    Some([b[0] / sum, b[1] / sum, b[2] / sum])
}

/// Bilinear tap at normalized `(u, v)` with the recovered texel-center rule
/// `p = uv * size - 0.5`. Returns straight-alpha RGBA in `[0, 1]`.
fn sample_bilinear(source: &Image, u: f32, v: f32, wrap: Wrap) -> [f32; 4] {
    let (x0, x1, fx) = axis(u * source.width as f32 - 0.5, source.width, wrap);
    let (y0, y1, fy) = axis(v * source.height as f32 - 0.5, source.height, wrap);
    let mut out = [0.0f32; 4];
    for c in 0..4 {
        let top = lerp(texel(source, x0, y0, c), texel(source, x1, y0, c), fx);
        let bottom = lerp(texel(source, x0, y1, c), texel(source, x1, y1, c), fx);
        out[c] = lerp(top, bottom, fy);
    }
    out
}

fn axis(pos: f32, size: u32, wrap: Wrap) -> (u32, u32, f32) {
    let limit = size as f32 - 1.0;
    let pos = if pos.is_finite() {
        pos
    } else if pos > 0.0 {
        limit
    } else {
        0.0
    };
    match wrap {
        Wrap::ClampToEdge => {
            let p = pos.clamp(0.0, limit);
            let i0 = p.floor();
            let i0 = i0 as u32;
            (i0, (i0 + 1).min(size - 1), p - i0 as f32)
        }
        Wrap::Repeat => {
            let floor = pos.floor();
            let m = size as i64;
            let i0 = (floor as i64).rem_euclid(m) as u32;
            let i1 = (floor as i64 + 1).rem_euclid(m) as u32;
            (i0, i1, pos - floor)
        }
    }
}

fn texel(source: &Image, x: u32, y: u32, channel: usize) -> f32 {
    source.pixels[((y * source.width + x) * 4) as usize + channel] as f32 / 255.0
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// RGBA8 quantization. Ties round to even, matching the recovered CPU oracle
/// and the UNORM conversion used by the GPU backends.
fn quantize(value: f32) -> u8 {
    (value * 255.0).round_ties_even().clamp(0.0, 255.0) as u8
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::raster_fragment::{Clamp, Defines, Edge as FragmentEdge};

    fn frame() -> Image {
        Image::new(4, 2, (0..4 * 2 * 4).map(|i| (i * 37 % 256) as u8).collect()).unwrap()
    }

    /// One quad covering the whole frame, UV identical to destination.
    fn quad() -> (Vec<[f32; 2]>, Vec<[f32; 2]>, Vec<u16>) {
        let dest = vec![[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0]];
        let uv = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        (dest, uv, vec![0, 1, 2, 0, 2, 3])
    }

    const RULES: Rules = Rules {
        winding: Winding::PositiveAreaIsFront,
        cull: Cull::None,
        edge: Edge::Inclusive,
        overlap: Overlap::First,
    };
    const SAMPLER: Sampler = Sampler { wrap: Wrap::ClampToEdge };
    const OFF: Defines = Defines {
        clamp_edge: false,
        uv_map: false,
        uv_map_ba: false,
        catmull_rom: false,
    };

    #[test]
    fn fragment_writer_matches_plain_rasterizer_without_defines() {
        let source = frame();
        let (dest, uv, ebo) = quad();
        let mesh = Mesh { dest: &dest, source_uv: &uv, ebo: &ebo };
        assert_eq!(
            rasterize_with_fragment(&source, mesh, RULES, SAMPLER, FragmentEdge::ReshapeNormal, OFF, Clamp::disabled(), Tolerance::TILE),
            rasterize(&source, mesh, RULES, SAMPLER)
        );
    }

    #[test]
    fn reference_wrap_is_refused_on_the_fragment_path() {
        let source = frame();
        let (dest, uv, ebo) = quad();
        let mesh = Mesh { dest: &dest, source_uv: &uv, ebo: &ebo };
        assert_eq!(
            rasterize_with_fragment(&source, mesh, RULES,
                Sampler { wrap: Wrap::Repeat }, FragmentEdge::ReshapeNormal, OFF,
                Clamp::disabled(), Tolerance::TILE),
            Err(Refusal::MissingInput("fragment repeat wrap"))
        );
    }
}
