//! 16-bit geometry stage for the export chain.
//!
//! The 8-bit preview resamples through `mock::geometry::FrameMap`; the 16-bit
//! export used to stop at the EXIF turn. This stage applies the same inverse
//! map to a linear 16-bit frame: orientation/flip, the radial lens inverse and
//! the crop/keystone chain all come from the caller's map.
//!
//! Resampling is a four-tap bilinear whose weights separate into an x and a y
//! factor (`wx * wy`), with source indices clamped to the frame. The returned
//! `origin` is the source pixel that lands on output `(0, 0)` — the phase the
//! banding stage needs.

/// One 16-bit geometry pass.
///
/// `map` takes a normalized output point and returns the normalized source
/// point (the `FrameMap::map(...).1` half). `out` is the target size.
pub fn apply(
    pixels: &[u16],
    width: u32,
    height: u32,
    channels: usize,
    out: (u32, u32),
    map: impl Fn([f32; 2]) -> [f32; 2],
) -> (Vec<u16>, u32, u32, [f32; 2]) {
    let (w, h) = (width as usize, height as usize);
    let (ow, oh) = (out.0.max(1) as usize, out.1.max(1) as usize);
    let mut buf = vec![0u16; ow * oh * channels];
    if w == 0 || h == 0 || pixels.len() < w * h * channels {
        return (buf, ow as u32, oh as u32, [0.0; 2]);
    }
    let max_x = (w - 1) as f32;
    let max_y = (h - 1) as f32;
    for y in 0..oh {
        let v = (y as f32 + 0.5) / oh as f32;
        for x in 0..ow {
            let u = (x as f32 + 0.5) / ow as f32;
            let s = map([u, v]);
            // Normalized source point to a pixel-centre grid.
            let fx = (s[0] * w as f32 - 0.5).clamp(0.0, max_x);
            let fy = (s[1] * h as f32 - 0.5).clamp(0.0, max_y);
            let x0 = fx.floor() as usize;
            let y0 = fy.floor() as usize;
            let x1 = (x0 + 1).min(w - 1);
            let y1 = (y0 + 1).min(h - 1);
            let wx = fx - x0 as f32;
            let wy = fy - y0 as f32;
            let wx0 = 1.0 - wx;
            let wy0 = 1.0 - wy;
            let dst = (y * ow + x) * channels;
            for c in 0..channels {
                let at = |px: usize, py: usize| pixels[(py * w + px) * channels + c] as f32;
                let top = at(x0, y0) * wx0 + at(x1, y0) * wx;
                let bottom = at(x0, y1) * wx0 + at(x1, y1) * wx;
                let value = top * wy0 + bottom * wy;
                buf[dst + c] = value.round().clamp(0.0, 65535.0) as u16;
            }
        }
    }
    let origin_point = map([0.0, 0.0]);
    let origin = [origin_point[0] * w as f32, origin_point[1] * h as f32];
    (buf, ow as u32, oh as u32, origin)
}

/// Identity check used by the export gate: no resampling when the map is the
/// identity and the size does not change.
pub fn is_identity(out: (u32, u32), source: (u32, u32)) -> bool {
    out == source
}

/// Run the full `FrameMap` chain over a rendered RGB16 frame.
///
/// The exported frame already carries the EXIF turn, so the caller passes the
/// catalog geometry (rotation/flip/crop/keystone plus the radial curve) and the
/// map's source size is the frame the exporter holds.
pub(crate) fn apply_frame(frame: super::Frame, geometry: &crate::Geometry) -> anyhow::Result<super::Frame> {
    Ok(apply_frame_with_origin(frame, geometry)?.0)
}

/// `apply_frame` plus the phase the banding stage needs: the source pixel that
/// lands on output `(0, 0)`, in source pixels.
pub(crate) fn apply_frame_with_origin(
    frame: super::Frame,
    geometry: &crate::Geometry,
) -> anyhow::Result<(super::Frame, [f32; 2])> {
    let map = crate::mock::geometry::FrameMap::new((frame.width, frame.height), geometry);
    if map.is_identity() {
        return Ok((frame, [0.0, 0.0]));
    }
    let (ow, oh) = map.frame_size();
    if ow <= 0.5 || oh <= 0.5 {
        return Ok((frame, [0.0, 0.0]));
    }
    let out = (ow.round() as u32, oh.round() as u32);
    let (pixels, width, height, origin) =
        apply(&frame.pixels, frame.width, frame.height, 3, out, |p| map.map(p).1);
    Ok((super::Frame { width, height, pixels, icc: frame.icc }, origin))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(w: u32, h: u32) -> Vec<u16> {
        (0..(w * h) as usize)
            .flat_map(|i| {
                let v = ((i * 0x1234) & 0xffff) as u16;
                [v, v.wrapping_add(17), v.wrapping_add(33)]
            })
            .collect()
    }

    #[test]
    fn identity_geometry_is_byte_identical() {
        let (w, h) = (37u32, 23u32);
        let src = ramp(w, h);
        let (out, ow, oh, origin) = apply(&src, w, h, 3, (w, h), |p| p);
        assert_eq!((ow, oh), (w, h));
        assert_eq!(out, src);
        assert_eq!(origin, [0.0, 0.0]);
    }

    #[test]
    fn half_pixel_shift_keeps_fourteen_bits() {
        let (w, h) = (64u32, 64u32);
        let mut src = vec![0u16; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let v = (x * 1000) as u16;
                let i = ((y * w + x) * 3) as usize;
                src[i..i + 3].copy_from_slice(&[v, v, v]);
            }
        }
        // Shift by exactly one pixel: every output is a source sample.
        let (out, _, _, _) = apply(&src, w, h, 3, (w, h), |p| [p[0] + 1.0 / w as f32, p[1]]);
        for y in 0..h as usize {
            for x in 0..(w as usize - 1) {
                let got = out[(y * w as usize + x) * 3];
                let want = src[(y * w as usize + x + 1) * 3];
                assert_eq!(got, want, "pixel {x},{y}");
            }
        }
    }

    #[test]
    fn radial_map_moves_the_corner_like_mode6() {
        // A linear 0.95 radial pull: the corner must move by the same ratio.
        let (w, h) = (100u32, 80u32);
        let src = ramp(w, h);
        let centre = [0.5f32, 0.5];
        let map = move |p: [f32; 2]| {
            let d = [p[0] - centre[0], p[1] - centre[1]];
            let r = (d[0] * d[0] + d[1] * d[1]).sqrt();
            let k = if r > 1e-6 { 0.95 } else { 1.0 };
            [centre[0] + d[0] * k, centre[1] + d[1] * k]
        };
        let (_, _, _, origin) = apply(&src, w, h, 3, (w, h), map);
        // Output (0,0) samples 5% closer to the centre.
        assert!((origin[0] - w as f32 * 0.025).abs() < 1e-3, "{origin:?}");
        assert!((origin[1] - h as f32 * 0.025).abs() < 1e-3, "{origin:?}");
    }
}
