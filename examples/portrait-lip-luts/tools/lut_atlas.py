"""512x512 PNG atlas <-> 64^3 volume mapping and the two documented samplers.

Standard library only. The geometry is not invented here: it is the layout the
project's own decoders implement.

  * host/CUDA side, `crates/portrait/src/cuda_pf90021.rs`
    `VolumeLut64::from_atlas_rgb8`: "Decodes a tight 512x512 RGB8 atlas using
    px=(z&7)*64+x, py=(z>>3)*64+y, top row first", with
    `crates/portrait/native/cuda/pf90021_volume_lut.cu` `load_node()` indexing
    `((z*64+y)*64+x)*3` and `sample_lut64` taking x from red, y from green and z
    from blue.
  * CPU oracle side, `crates/portrait/src/kernels/face_detail.rs`
    `lut3d_sample` / `lut_sample_2d` / `lut_read`: an RGB8 atlas of exactly
    512*512*3 bytes, tile row `t = floor(z/8)`, tile column `s = z - 8t`,
    `u = s/8 + 0.5/512 + (63/512)*r`, `v = t/8 + 0.5/512 + (63/512)*g`, then a
    bilinear tap at `(512u - 0.5, 512v - 0.5)` with edge clamping.

This module is an independent implementation of that contract, so the shipped
bytes can be checked without a Rust toolchain and without any third-party
package.
"""

from __future__ import annotations

LUT_EDGE = 64
TILE = 64
ATLAS_EDGE = 512
HALF_TEXEL = 0.5 / ATLAS_EDGE
SPAN = (LUT_EDGE - 1) / ATLAS_EDGE
TILE_SPAN = TILE / ATLAS_EDGE


def node_pixel(x: int, y: int, z: int) -> tuple[int, int]:
    """Atlas pixel holding volume node (x = red, y = green, z = blue)."""
    return (z & 7) * TILE + x, (z >> 3) * TILE + y


def volume_index(x: int, y: int, z: int) -> int:
    return ((z * LUT_EDGE + y) * LUT_EDGE + x) * 3


def atlas_to_volume(atlas: bytes, channels: int) -> bytearray:
    """Tight atlas -> 64^3 volume of RGB bytes; alpha is dropped."""
    if len(atlas) != ATLAS_EDGE * ATLAS_EDGE * channels:
        raise ValueError("atlas length does not match a 512x512 atlas")
    volume = bytearray(LUT_EDGE * LUT_EDGE * LUT_EDGE * 3)
    for z in range(LUT_EDGE):
        tile_x, tile_y = (z & 7) * TILE, (z >> 3) * TILE
        for y in range(LUT_EDGE):
            row = (tile_y + y) * ATLAS_EDGE
            for x in range(LUT_EDGE):
                source = (row + tile_x + x) * channels
                target = volume_index(x, y, z)
                volume[target] = atlas[source]
                volume[target + 1] = atlas[source + 1]
                volume[target + 2] = atlas[source + 2]
    return volume


def volume_to_atlas(volume: bytes, channels: int) -> bytes:
    """Volume -> tight atlas; extra channels are filled with 255."""
    if len(volume) != LUT_EDGE * LUT_EDGE * LUT_EDGE * 3:
        raise ValueError("volume must hold 64^3 RGB nodes")
    atlas = bytearray(b"\xff" * (ATLAS_EDGE * ATLAS_EDGE * channels))
    for z in range(LUT_EDGE):
        tile_x, tile_y = (z & 7) * TILE, (z >> 3) * TILE
        for y in range(LUT_EDGE):
            row = (tile_y + y) * ATLAS_EDGE
            for x in range(LUT_EDGE):
                source = volume_index(x, y, z)
                target = (row + tile_x + x) * channels
                atlas[target] = volume[source]
                atlas[target + 1] = volume[source + 1]
                atlas[target + 2] = volume[source + 2]
    return bytes(atlas)


def _texel(atlas: bytes, channels: int, x: int, y: int) -> tuple[float, float, float]:
    offset = (y * ATLAS_EDGE + x) * channels
    return atlas[offset] / 255.0, atlas[offset + 1] / 255.0, atlas[offset + 2] / 255.0


def _clamp_index(value: int) -> int:
    return 0 if value < 0 else (ATLAS_EDGE - 1 if value > ATLAS_EDGE - 1 else value)


def _bilinear(atlas: bytes, channels: int, u: float, v: float) -> tuple[float, float, float]:
    x = u * ATLAS_EDGE - 0.5
    y = v * ATLAS_EDGE - 0.5
    xi, yi = int(x // 1), int(y // 1)  # floor, also for negatives
    fx, fy = x - xi, y - yi
    x0, x1 = _clamp_index(xi), _clamp_index(xi + 1)
    y0, y1 = _clamp_index(yi), _clamp_index(yi + 1)
    c00, c10 = _texel(atlas, channels, x0, y0), _texel(atlas, channels, x1, y0)
    c01, c11 = _texel(atlas, channels, x0, y1), _texel(atlas, channels, x1, y1)
    out = []
    for channel in range(3):
        top = c00[channel] * (1.0 - fx) + c10[channel] * fx
        bottom = c01[channel] * (1.0 - fx) + c11[channel] * fx
        out.append(top * (1.0 - fy) + bottom * fy)
    return out[0], out[1], out[2]


def sample_atlas_f64(atlas: bytes, channels: int, r: float, g: float, b: float):
    """Mirror of `face_detail::lut3d_sample`: RGB8 atlas in, [0,1] out."""
    r = min(max(r, 0.0), 1.0)
    g = min(max(g, 0.0), 1.0)
    b = min(max(b, 0.0), 1.0)
    f = b * (LUT_EDGE - 1)
    z0 = f // 1
    z1 = -(-f // 1)  # ceil
    t0 = (z0 / (ATLAS_EDGE // TILE)) // 1
    s0 = z0 - t0 * (ATLAS_EDGE // TILE)
    t1 = (z1 / (ATLAS_EDGE // TILE)) // 1
    s1 = z1 - t1 * (ATLAS_EDGE // TILE)
    u0 = s0 * TILE_SPAN + HALF_TEXEL + SPAN * r
    v0 = t0 * TILE_SPAN + HALF_TEXEL + SPAN * g
    u1 = s1 * TILE_SPAN + HALF_TEXEL + SPAN * r
    v1 = t1 * TILE_SPAN + HALF_TEXEL + SPAN * g
    p0 = _bilinear(atlas, channels, u0, v0)
    p1 = _bilinear(atlas, channels, u1, v1)
    fraction = f - z0
    return tuple(p0[channel] * (1.0 - fraction) + p1[channel] * fraction
                 for channel in range(3))


def sample_volume_f64(volume: bytes, r: float, g: float, b: float):
    """Mirror of `pf90021_volume_lut.cu::sample_lut64`: trilinear on 64^3."""
    r = min(max(r, 0.0), 1.0)
    g = min(max(g, 0.0), 1.0)
    b = min(max(b, 0.0), 1.0)
    xf, yf, zf = r * 63.0, g * 63.0, b * 63.0
    x0, y0, z0 = int(xf), int(yf), int(zf)
    x1, y1, z1 = min(x0 + 1, 63), min(y0 + 1, 63), min(z0 + 1, 63)
    tx, ty, tz = xf - x0, yf - y0, zf - z0

    def node(x, y, z):
        offset = volume_index(x, y, z)
        return (volume[offset] / 255.0, volume[offset + 1] / 255.0, volume[offset + 2] / 255.0)

    c000, c100 = node(x0, y0, z0), node(x1, y0, z0)
    c010, c110 = node(x0, y1, z0), node(x1, y1, z0)
    c001, c101 = node(x0, y0, z1), node(x1, y0, z1)
    c011, c111 = node(x0, y1, z1), node(x1, y1, z1)
    out = []
    for channel in range(3):
        c00 = c000[channel] + (c100[channel] - c000[channel]) * tx
        c01 = c010[channel] + (c110[channel] - c010[channel]) * tx
        c10 = c001[channel] + (c101[channel] - c001[channel]) * tx
        c11 = c011[channel] + (c111[channel] - c011[channel]) * tx
        c0 = c00 + (c01 - c00) * ty
        c1 = c10 + (c11 - c10) * ty
        out.append(c0 + (c1 - c0) * tz)
    return out[0], out[1], out[2]


def lip_to_byte(value: float) -> int:
    """Mirror of `face_detail::lip_to_byte`: floor(v*255 + 0.5), clamped."""
    scaled = value * 255.0 + 0.5
    scaled = scaled // 1  # floor
    return int(min(max(scaled, 0.0), 255.0))
