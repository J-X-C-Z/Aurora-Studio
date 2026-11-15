//! Pure fragment oracle for the recovered `ReshapeNormal` family.
//!
//! This module owns only what is evaluated *per covered pixel, after* the
//! geometry has produced a source UV and a destination UV. It deliberately
//! contains no mesh, no EBO, no barycentrics and no coverage: that half lives
//! in `raster.rs` and stays frozen.
//!
//! Recovered chain (P30 lane, `ReshapeNormalAdjClampEdge` / `ReshapeNormal`):
//! ```text
//! sample_uv = src_uv
//! if TS_CLAMP_EDGE:  sample_uv = dst_uv + (src_uv - dst_uv) * smoothcoord(src_uv)
//! if ENABLE_UVMAP:   sampled = bilinear(sample_uv); current = bilinear(dst_uv)
//!                    rgba = [sampled.r + dx, sampled.g + dy, current.b, current.a]
//!                    or, with ENABLE_UVMAP_BA,
//!                    rgba = [current.r, current.g, sampled.b + dx, sampled.a + dy]
//! else:              rgba = bilinear(sample_uv) | catmull_rom(sample_uv)
//! ```
//! `dx`/`dy` are `sample_uv - dst_uv`, exactly as the shipped shader's
//! `sample_texcoord - v_dst_texcoord`.
//!
//! Alpha is straight and is sampled and quantized like RGB; nothing blends.

use crate::Image;

/// Which `smoothcoord()` body the pass uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge {
    /// Type 63 `ReshapeNormal.frag`: fixed `2.49 / 4.98`.
    ReshapeNormal,
    /// Type 0x40 `ReshapeNormalAdjClampEdge.frag`: `i_` / `jb` / `ic`.
    AdjClampEdge,
}

/// `i_` / `jb` / `ic`. A negative factor forces `smoothcoord() == 1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clamp {
    pub factor: f32,
    pub protect_x: u8,
    pub protect_y: u8,
}

impl Clamp {
    pub const fn disabled() -> Self {
        Self {
            factor: -1.0,
            protect_x: 0,
            protect_y: 0,
        }
    }
}

/// The four define keys from `0x611B460`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Defines {
    pub clamp_edge: bool,
    pub uv_map: bool,
    pub uv_map_ba: bool,
    pub catmull_rom: bool,
}

impl Defines {
    pub const fn plain() -> Self {
        Self {
            clamp_edge: false,
            uv_map: false,
            uv_map_ba: false,
            catmull_rom: false,
        }
    }
}

/// `smoothcoord()` for type 63.
pub fn smooth_coord_reshape(uv: [f32; 2]) -> f32 {
    let x = 2.49 - (4.98 * (uv[0] - 0.5)).abs();
    let y = 2.49 - (4.98 * (uv[1] - 0.5)).abs();
    x.min(y).clamp(0.0, 1.0)
}

/// `smoothcoord()` for type 0x40.
pub fn smooth_coord_adj(uv: [f32; 2], clamp: Clamp) -> f32 {
    if clamp.factor < 0.0 {
        return 1.0;
    }
    let mut x = 1.0;
    let mut y = 1.0;
    if clamp.protect_x == 1 {
        x = clamp.factor / 2.0 - (clamp.factor * (0.5 - uv[0]).max(0.0)).abs();
    }
    if clamp.protect_x == 2 {
        x = clamp.factor / 2.0 - (clamp.factor * (uv[0] - 0.5).max(0.0)).abs();
    }
    if clamp.protect_y == 1 {
        y = clamp.factor / 2.0 - (clamp.factor * (0.5 - uv[1]).max(0.0)).abs();
    }
    if clamp.protect_y == 2 {
        y = clamp.factor / 2.0 - (clamp.factor * (uv[1] - 0.5).max(0.0)).abs();
    }
    x.min(y).clamp(0.0, 1.0)
}

/// Dispatches to the `smoothcoord()` the pass selected.
pub fn smooth_coord(edge: Edge, uv: [f32; 2], clamp: Clamp) -> f32 {
    match edge {
        Edge::ReshapeNormal => smooth_coord_reshape(uv),
        Edge::AdjClampEdge => smooth_coord_adj(uv, clamp),
    }
}

/// `texture2D` with `GL_LINEAR` and clamp-to-edge, texel-center rule
/// `p = uv * size - 0.5`. Returns straight-alpha RGBA in `[0, 1]`.
pub fn sample_bilinear(source: &Image, u: f32, v: f32) -> [f32; 4] {
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

/// The shipped 9-tap Catmull-Rom. `size` is the shader's `u_Width`/`u_Height`;
/// it may differ from the source image size (the P30 oracle tests that).
pub fn sample_catmull_rom(source: &Image, u: f32, v: f32, size: (f32, f32)) -> [f32; 4] {
    let (width, height) = size;
    let sx = u * width;
    let sy = v * height;
    let px = (sx - 0.5).floor() + 0.5;
    let py = (sy - 0.5).floor() + 0.5;
    let fx = sx - px;
    let fy = sy - py;

    let w0x = weight0(fx);
    let w1x = weight1(fx);
    let w2x = weight2(fx);
    let w3x = weight3(fx);
    let w0y = weight0(fy);
    let w1y = weight1(fy);
    let w2y = weight2(fy);
    let w3y = weight3(fy);
    let w12x = w1x + w2x;
    let w12y = w1y + w2y;
    let t12x = px + w2x / w12x;
    let t12y = py + w2y / w12y;

    let taps = [
        (px - 1.0, py - 1.0, w0x * w0y),
        (t12x, py - 1.0, w12x * w0y),
        (px + 2.0, py - 1.0, w3x * w0y),
        (px - 1.0, t12y, w0x * w12y),
        (t12x, t12y, w12x * w12y),
        (px + 2.0, t12y, w3x * w12y),
        (px - 1.0, py + 2.0, w0x * w3y),
        (t12x, py + 2.0, w12x * w3y),
        (px + 2.0, py + 2.0, w3x * w3y),
    ];

    let mut color = [0.0f32; 4];
    for (tx, ty, weight) in taps {
        let tap = sample_bilinear(source, tx / width, ty / height);
        for channel in 0..4 {
            color[channel] += tap[channel] * weight;
        }
    }
    color
}

/// One fragment evaluation. `src_uv` is the interpolated `hT` (source UV) and
/// `dst_uv` is the covered pixel's `gl_Position.xy * 0.5 + 0.5`.
pub fn evaluate(
    source: &Image,
    src_uv: [f32; 2],
    dst_uv: [f32; 2],
    edge: Edge,
    defines: Defines,
    clamp: Clamp,
) -> [f32; 4] {
    let mut sample_uv = src_uv;
    if defines.clamp_edge {
        let factor = smooth_coord(edge, src_uv, clamp);
        sample_uv = [
            dst_uv[0] + (src_uv[0] - dst_uv[0]) * factor,
            dst_uv[1] + (src_uv[1] - dst_uv[1]) * factor,
        ];
    }

    if defines.uv_map {
        // Both taps stay plain bilinear in the shipped UVMAP branch, even when
        // DEF_CATMULL_ROM is also defined.
        let sampled = sample_bilinear(source, sample_uv[0], sample_uv[1]);
        let current = sample_bilinear(source, dst_uv[0], dst_uv[1]);
        let dx = sample_uv[0] - dst_uv[0];
        let dy = sample_uv[1] - dst_uv[1];
        return if defines.uv_map_ba {
            [current[0], current[1], sampled[2] + dx, sampled[3] + dy]
        } else {
            [sampled[0] + dx, sampled[1] + dy, current[2], current[3]]
        };
    }

    if defines.catmull_rom {
        sample_catmull_rom(
            source,
            sample_uv[0],
            sample_uv[1],
            (source.width as f32, source.height as f32),
        )
    } else {
        sample_bilinear(source, sample_uv[0], sample_uv[1])
    }
}

/// RGBA8 store, ties-to-even, same as the pass writer.
pub fn to_rgba8(color: [f32; 4]) -> [u8; 4] {
    let mut out = [0u8; 4];
    for channel in 0..4 {
        out[channel] = (color[channel] * 255.0).clamp(0.0, 255.0).round_ties_even() as u8;
    }
    out
}

fn weight0(f: f32) -> f32 {
    f * (-0.5 + f * (1.0 - 0.5 * f))
}

fn weight1(f: f32) -> f32 {
    1.0 + f * f * (-2.5 + 1.5 * f)
}

fn weight2(f: f32) -> f32 {
    f * (0.5 + f * (2.0 - 1.5 * f))
}

fn weight3(f: f32) -> f32 {
    f * f * (-0.5 + 0.5 * f)
}

fn texel(source: &Image, x: usize, y: usize, channel: usize) -> f32 {
    source.pixels[(y * source.width as usize + x) * 4 + channel] as f32 / 255.0
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 5x3 fixture from `work/backend-r11/o3c/PARITY-DATA.md`.
    fn p30_tex() -> Image {
        let mut pixels = Vec::with_capacity(5 * 3 * 4);
        for y in 0..3u32 {
            for x in 0..5u32 {
                pixels.extend_from_slice(&[
                    (10 + 30 * x) as u8,
                    (20 + 40 * y) as u8,
                    (11 * x + 7 * y) as u8,
                    (255 - 9 * x - 5 * y) as u8,
                ]);
            }
        }
        Image::new(5, 3, pixels).unwrap()
    }

    const OFF: Defines = Defines { clamp_edge: false, uv_map: false, uv_map_ba: false, catmull_rom: false };
    const CLAMP: Clamp = Clamp::disabled();

    fn rgba(src: [f32; 2], dst: [f32; 2], defines: Defines, edge: Edge, clamp: Clamp) -> [u8; 4] {
        to_rgba8(evaluate(&p30_tex(), src, dst, edge, defines, clamp))
    }

    #[test]
    fn documented_parity_vectors_hold() {
        assert_eq!(rgba([0.30, 0.40], [0.25, 0.35], OFF, Edge::ReshapeNormal, CLAMP),
            [40, 48, 16, 242]);
        assert_eq!(rgba([0.30, 0.40], [0.25, 0.35],
            Defines { catmull_rom: true, ..OFF }, Edge::ReshapeNormal, CLAMP),
            [40, 47, 16, 243]);
        assert_eq!(rgba([0.30, 0.20], [0.20, 0.10],
            Defines { uv_map: true, ..OFF }, Edge::ReshapeNormal, CLAMP), [66, 50, 6, 251]);
        assert_eq!(rgba([0.30, 0.20], [0.20, 0.10],
            Defines { uv_map: true, uv_map_ba: true, ..OFF }, Edge::ReshapeNormal, CLAMP),
            [25, 20, 37, 255]);
        assert_eq!(rgba([0.02, 0.03], [0.30, 0.40],
            Defines { clamp_edge: true, ..OFF }, Edge::ReshapeNormal, CLAMP),
            [36, 44, 14, 244]);
        assert_eq!(rgba([0.02, 0.50], [0.30, 0.40],
            Defines { clamp_edge: true, ..OFF }, Edge::AdjClampEdge,
            Clamp { factor: 0.5, protect_x: 1, protect_y: 0 }), [40, 48, 16, 243]);
    }

    #[test]
    fn uvmap_ignores_catmull_and_is_identity_on_zero_offset() {
        let uv = [0.4, 0.6];
        let plain = evaluate(&p30_tex(), uv, uv, Edge::ReshapeNormal,
            Defines { uv_map: true, ..OFF }, CLAMP);
        let with_catmull = evaluate(&p30_tex(), uv, uv, Edge::ReshapeNormal,
            Defines { uv_map: true, catmull_rom: true, ..OFF }, CLAMP);
        assert_eq!(plain, with_catmull);
        let sampled = sample_bilinear(&p30_tex(), uv[0], uv[1]);
        assert_eq!(plain, sampled);
    }
}
