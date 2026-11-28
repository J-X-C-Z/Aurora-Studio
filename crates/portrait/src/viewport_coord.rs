//! Zero-dependency, pixel-center-aligned viewport coordinate transforms.
//!
//! This module is the single source of truth for converting between the three
//! coordinate spaces used by the portrait pipeline:
//!
//! * **full-image pixel indices** (`full_x`, `full_y`) — the space that
//!   detections, landmarks, body keypoints and normalized ROIs live in;
//! * **normalized full-image coordinates** (`norm_x`, `norm_y` in `0..=1`);
//! * **destination canvas pixel indices** (`dst_x`, `dst_y`) of a rendered
//!   frame that shows only `region` of the full image.
//!
//! The renderer samples with pixel-center semantics:
//!
//! ```text
//! norm_x = region.x + (dst_x + 0.5) / dst_w * region.w
//! ```
//!
//! These transforms use exactly that identity, and their inverses subtract
//! `0.5` only when converting a pixel *index* to/from its *center*. The
//! historical face/body path instead used
//! `region.x + dst_x / dst_w * region.w - 0.5`, which happens to be exact for
//! `region == FULL` but shifts every landmark by half a destination pixel as
//! soon as the viewport is panned or zoomed (`region != FULL`). That is the
//! half-pixel phase error this module removes; see
//! `non_full_region_has_no_half_pixel_shift` for the regression guard.
//!
//! # Conventions
//!
//! * A pixel index `i` has its center at index `i + 0.5` in the same space.
//!   Full pixel `p` therefore sits at normalized `(p + 0.5) / full_w`.
//! * `region` is the viewport in normalized full-image coordinates: the
//!   `dst_w x dst_h` canvas displays exactly that rectangle.
//! * A 1:1 crop has `region.w == dst_w / full_w` (and likewise for `h`).
//!
//! # Precision
//!
//! All arithmetic is `f64`; the `f32` functions round exactly once, at return.
//! That single rounding is the only reason a result is not exact:
//!
//! * The `*_f64` variants have no such rounding and are strictly inverse:
//!   `dst -> norm -> dst`, `dst -> full -> dst` and `norm -> dst -> norm`
//!   all agree to better than `1e-12` for every region, including odd sizes.
//! * Normalized `f32` results carry at most `6e-8` absolute error (the `f32`
//!   ULP below 1). Round-tripping a destination index through them costs
//!   `dst_w / region.w` times that, e.g. ~`4e-5 px` on a 1365-wide 1:1
//!   viewport. No `f32`-valued normalized coordinate can do better.
//! * Full-pixel `f32` results carry the `f32` ULP of the full-image
//!   coordinate (about `6.1e-5 px` at 1365, `2.4e-4 px` at 4000), which is
//!   why the module exposes `*_f64` for chained landmark math. For `f32`
//!   callers the practical guarantee is: a 1:1 crop with exactly
//!   representable region fields maps `dst_x` to `full_x0 + dst_x` with
//!   **zero** offset (bit-exact), and every other case stays inside one
//!   `f32` ULP of the destination-space coordinate.
//!
//! # Invalid input
//!
//! None of these functions panic. The point transforms return
//! [`INVALID_POINT`] / [`INVALID_POINT_F64`] (a pair of `NaN`s) when a canvas
//! dimension is zero, the region has a non-positive or non-finite extent, or
//! an input coordinate is not finite. [`intersect_roi_with_viewport`] returns
//! `None` in those cases.

/// Normalized rectangle `(x, y, w, h)` in full-image coordinates.
///
/// `w`/`h` are extents, not inclusive end points: the rectangle spans
/// `x ..= x + w` in continuous coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormRect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl NormRect {
    /// The whole image: `(0, 0, 1, 1)`.
    pub const FULL: Self = Self {
        x: 0.0,
        y: 0.0,
        w: 1.0,
        h: 1.0,
    };

    pub const fn new(x: f32, y: f32, w: f32, h: f32) -> Self {
        Self { x, y, w, h }
    }

    /// All four fields are finite (extents may still be non-positive).
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.w.is_finite() && self.h.is_finite()
    }

    /// Finite with strictly positive extents: usable as a viewport or ROI.
    pub fn is_valid(self) -> bool {
        self.is_finite() && self.w > 0.0 && self.h > 0.0
    }

    /// Continuous right edge `x + w`.
    pub fn right(self) -> f32 {
        self.x + self.w
    }

    /// Continuous bottom edge `y + h`.
    pub fn bottom(self) -> f32 {
        self.y + self.h
    }

    /// Continuous normalized intersection with `other`.
    ///
    /// Returns `None` when either rectangle is non-finite or the overlap has
    /// no area.
    pub fn intersection(self, other: Self) -> Option<Self> {
        if !self.is_finite() || !other.is_finite() {
            return None;
        }
        let x0 = self.x.max(other.x);
        let y0 = self.y.max(other.y);
        let x1 = self.right().min(other.right());
        let y1 = self.bottom().min(other.bottom());
        if x1 > x0 && y1 > y0 {
            Some(Self::new(x0, y0, x1 - x0, y1 - y0))
        } else {
            None
        }
    }
}

impl Default for NormRect {
    fn default() -> Self {
        Self::FULL
    }
}

/// Returned by the `f32` point transforms when the input cannot be mapped.
pub const INVALID_POINT: (f32, f32) = (f32::NAN, f32::NAN);

/// Returned by the `f64` point transforms when the input cannot be mapped.
pub const INVALID_POINT_F64: (f64, f64) = (f64::NAN, f64::NAN);

/// Slack, in destination pixels, used when quantizing a floating intersection
/// to integer pixel indices. `f32` normalized viewports carry absolute error
/// up to roughly `full_extent * 2^-23` pixels, so a bound that lands within
/// this distance of an integer is treated as that integer. The slack is far
/// below one pixel and cannot move a boundary onto a different pixel center.
const INDEX_SNAP_EPS: f64 = 1e-3;

#[inline]
fn canvas_ok(w: u32, h: u32) -> bool {
    w != 0 && h != 0
}

#[inline]
fn point_finite_f32(x: f32, y: f32) -> bool {
    x.is_finite() && y.is_finite()
}

#[inline]
fn point_finite_f64(x: f64, y: f64) -> bool {
    x.is_finite() && y.is_finite()
}

/// Center of destination pixel index `dst_index` in normalized coordinates.
#[inline]
fn axis_dst_center_to_norm(dst_index: f64, dst_len: u32, origin: f32, extent: f32) -> f64 {
    origin as f64 + (dst_index + 0.5) / dst_len as f64 * extent as f64
}

/// Normalized coordinate to continuous destination pixel index (`i + 0.5` is
/// the center of pixel `i`).
#[inline]
fn axis_norm_to_dst_index(norm: f64, dst_len: u32, origin: f32, extent: f32) -> f64 {
    (norm - origin as f64) / extent as f64 * dst_len as f64 - 0.5
}

/// Snaps values within [`INDEX_SNAP_EPS`] of an integer onto that integer.
#[inline]
fn snap_index(value: f64) -> f64 {
    let rounded = value.round();
    if (value - rounded).abs() <= INDEX_SNAP_EPS {
        rounded
    } else {
        value
    }
}

/// `f64` core of [`dst_pixel_to_full_norm`]; no result rounding.
pub fn dst_pixel_to_full_norm_f64(
    dx: f64,
    dy: f64,
    dst_w: u32,
    dst_h: u32,
    region: NormRect,
) -> (f64, f64) {
    if !canvas_ok(dst_w, dst_h) || !point_finite_f64(dx, dy) || !region.is_valid() {
        return INVALID_POINT_F64;
    }
    let nx = axis_dst_center_to_norm(dx, dst_w, region.x, region.w);
    let ny = axis_dst_center_to_norm(dy, dst_h, region.y, region.h);
    (nx, ny)
}

/// `f64` core of [`full_norm_to_dst_pixel`]; no result rounding.
pub fn full_norm_to_dst_pixel_f64(
    nx: f64,
    ny: f64,
    dst_w: u32,
    dst_h: u32,
    region: NormRect,
) -> (f64, f64) {
    if !canvas_ok(dst_w, dst_h) || !point_finite_f64(nx, ny) || !region.is_valid() {
        return INVALID_POINT_F64;
    }
    let dx = axis_norm_to_dst_index(nx, dst_w, region.x, region.w);
    let dy = axis_norm_to_dst_index(ny, dst_h, region.y, region.h);
    (dx, dy)
}

/// `f64` core of [`full_pixel_to_dst_pixel`]; no result rounding.
pub fn full_pixel_to_dst_pixel_f64(
    fx: f64,
    fy: f64,
    full_w: u32,
    full_h: u32,
    dst_w: u32,
    dst_h: u32,
    region: NormRect,
) -> (f64, f64) {
    if full_w == 0
        || full_h == 0
        || !canvas_ok(dst_w, dst_h)
        || !point_finite_f64(fx, fy)
        || !region.is_valid()
    {
        return INVALID_POINT_F64;
    }
    let nx = (fx + 0.5) / full_w as f64;
    let ny = (fy + 0.5) / full_h as f64;
    let dx = axis_norm_to_dst_index(nx, dst_w, region.x, region.w);
    let dy = axis_norm_to_dst_index(ny, dst_h, region.y, region.h);
    (dx, dy)
}

/// `f64` core of [`dst_pixel_to_full_pixel`]; no result rounding.
pub fn dst_pixel_to_full_pixel_f64(
    dx: f64,
    dy: f64,
    full_w: u32,
    full_h: u32,
    dst_w: u32,
    dst_h: u32,
    region: NormRect,
) -> (f64, f64) {
    if full_w == 0
        || full_h == 0
        || !canvas_ok(dst_w, dst_h)
        || !point_finite_f64(dx, dy)
        || !region.is_valid()
    {
        return INVALID_POINT_F64;
    }
    let nx = axis_dst_center_to_norm(dx, dst_w, region.x, region.w);
    let ny = axis_dst_center_to_norm(dy, dst_h, region.y, region.h);
    (nx * full_w as f64 - 0.5, ny * full_h as f64 - 0.5)
}

/// Destination canvas pixel index -> normalized full-image coordinate.
///
/// `dst_w`/`dst_h` are the rendered canvas size; `region` is the viewport
/// shown on that canvas. The result is the **center** of destination pixel
/// `(dx, dy)`, expressed in full-image normalized coordinates. This is the
/// renderer's sampling identity.
pub fn dst_pixel_to_full_norm(
    dx: f32,
    dy: f32,
    dst_w: u32,
    dst_h: u32,
    region: NormRect,
) -> (f32, f32) {
    if !point_finite_f32(dx, dy) {
        return INVALID_POINT;
    }
    let (nx, ny) = dst_pixel_to_full_norm_f64(dx as f64, dy as f64, dst_w, dst_h, region);
    (nx as f32, ny as f32)
}

/// Normalized full-image coordinate -> continuous destination canvas index.
///
/// Exact inverse of [`dst_pixel_to_full_norm`]: a pixel *center*
/// `(dx + 0.5) / dst_w` maps back to `dx`, never to `dx +- 0.5`.
pub fn full_norm_to_dst_pixel(
    nx: f32,
    ny: f32,
    dst_w: u32,
    dst_h: u32,
    region: NormRect,
) -> (f32, f32) {
    if !point_finite_f32(nx, ny) {
        return INVALID_POINT;
    }
    let (dx, dy) = full_norm_to_dst_pixel_f64(nx as f64, ny as f64, dst_w, dst_h, region);
    (dx as f32, dy as f32)
}

/// Full-image pixel index -> continuous destination canvas index.
///
/// This is the transform the face/body stages need: landmarks and ROIs are
/// stored in full-image pixel indices, the stage runs on the local canvas.
/// The full pixel *center* is `(full_x + 0.5) / full_w`, so for a 1:1 crop the
/// result is exactly `full_x - full_x0`.
pub fn full_pixel_to_dst_pixel(
    fx: f32,
    fy: f32,
    full_w: u32,
    full_h: u32,
    dst_w: u32,
    dst_h: u32,
    region: NormRect,
) -> (f32, f32) {
    if !point_finite_f32(fx, fy) {
        return INVALID_POINT;
    }
    let (dx, dy) =
        full_pixel_to_dst_pixel_f64(fx as f64, fy as f64, full_w, full_h, dst_w, dst_h, region);
    (dx as f32, dy as f32)
}

/// Destination canvas pixel index -> continuous full-image pixel index.
///
/// Exact inverse of [`full_pixel_to_dst_pixel`]. For `region == FULL` this is
/// the identity `fx == dx`; for a 1:1 crop it is `fx == full_x0 + dx`; for any
/// zoom it is the sub-pixel position the renderer actually sampled.
pub fn dst_pixel_to_full_pixel(
    dx: f32,
    dy: f32,
    full_w: u32,
    full_h: u32,
    dst_w: u32,
    dst_h: u32,
    region: NormRect,
) -> (f32, f32) {
    if !point_finite_f32(dx, dy) {
        return INVALID_POINT;
    }
    let (fx, fy) =
        dst_pixel_to_full_pixel_f64(dx as f64, dy as f64, full_w, full_h, dst_w, dst_h, region);
    (fx as f32, fy as f32)
}

/// Intersects a normalized ROI with the viewport and returns the integer
/// destination pixel bounding box `(x0, y0, w, h)`.
///
/// The box covers exactly the destination pixels **whose centers** lie inside
/// the intersection, using the same pixel-center rule as the renderer. It is
/// clamped to the canvas; `None` means "nothing to process" (empty
/// intersection, invalid rectangle, zero canvas).
///
/// For a 1:1 crop, a single full pixel intersecting the viewport maps to a
/// single destination pixel.
pub fn intersect_roi_with_viewport(
    full_roi: NormRect,
    region: NormRect,
    dst_w: u32,
    dst_h: u32,
) -> Option<(u32, u32, u32, u32)> {
    if !canvas_ok(dst_w, dst_h) || !region.is_valid() || !full_roi.is_valid() {
        return None;
    }
    let inter = full_roi.intersection(region)?;

    let axis_box = |lo_norm: f32, hi_norm: f32, origin: f32, extent: f32, len: u32| {
        let lo = snap_index(axis_norm_to_dst_index(lo_norm as f64, len, origin, extent)).ceil();
        let hi = snap_index(axis_norm_to_dst_index(hi_norm as f64, len, origin, extent)).floor();
        let hi_max = len as f64 - 1.0;
        let lo = lo.max(0.0);
        let hi = hi.min(hi_max);
        if lo > hi {
            None
        } else {
            // Bounded by `len`, so the casts cannot saturate.
            Some((lo as u32, (hi - lo) as u32 + 1))
        }
    };

    let (x0, w) = axis_box(inter.x, inter.right(), region.x, region.w, dst_w)?;
    let (y0, h) = axis_box(inter.y, inter.bottom(), region.y, region.h, dst_h)?;
    Some((x0, y0, w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(region, full_w, full_h, dst_w, dst_h)`.
    const CASES: &[(NormRect, u32, u32, u32, u32)] = &[
        (NormRect::FULL, 1365, 911, 1365, 911),
        (NormRect::new(0.25, 0.33, 0.25, 0.25), 1365, 911, 341, 228),
        (NormRect::new(0.0, 0.0, 0.1, 0.1), 1365, 911, 200, 200),
        (NormRect::new(0.1, 0.2, 0.5, 0.5), 1920, 1080, 960, 540),
        (NormRect::new(0.125, 0.25, 0.25, 0.25), 1360, 912, 340, 228),
        (NormRect::new(0.5, 0.5, 0.25, 0.25), 1365, 911, 683, 456),
        (NormRect::new(0.25, 0.25, 0.5, 0.5), 1365, 911, 1365, 911), // 200% zoom
        (NormRect::new(0.25, 0.33, 0.25, 0.25), 1365, 911, 1365, 911), // 400% zoom
        (NormRect::FULL, 17, 13, 3, 2),
    ];

    fn assert_close(actual: f32, expected: f32, tol: f32, what: &str) {
        assert!(
            (actual - expected).abs() <= tol,
            "{what}: actual {actual:.7} expected {expected:.7} (diff {:.7} > {tol})",
            (actual - expected).abs()
        );
    }

    /// Keeps the exhaustive loops bounded for large canvases.
    fn sample_step(len: u32) -> u32 {
        (len / 89).max(1)
    }

    /// The transforms are exactly inverse when no `f32` rounding is applied:
    /// this is the mathematical guarantee the module exists for.
    #[test]
    fn transforms_are_strictly_inverse_in_f64() {
        for &(region, full_w, full_h, dst_w, dst_h) in CASES {
            let (step_x, step_y) = (sample_step(dst_w), sample_step(dst_h));
            let mut max_norm = 0.0f64;
            let mut max_pixel = 0.0f64;
            for y in (0..dst_h).step_by(step_y as usize) {
                for x in (0..dst_w).step_by(step_x as usize) {
                    let (dx, dy) = (x as f64, y as f64);

                    let (nx, ny) = dst_pixel_to_full_norm_f64(dx, dy, dst_w, dst_h, region);
                    let (bx, by) = full_norm_to_dst_pixel_f64(nx, ny, dst_w, dst_h, region);
                    max_norm = max_norm.max((bx - dx).abs()).max((by - dy).abs());

                    let (fx, fy) =
                        dst_pixel_to_full_pixel_f64(dx, dy, full_w, full_h, dst_w, dst_h, region);
                    let (cx, cy) =
                        full_pixel_to_dst_pixel_f64(fx, fy, full_w, full_h, dst_w, dst_h, region);
                    max_pixel = max_pixel.max((cx - dx).abs()).max((cy - dy).abs());

                    // Both directions must also close in normalized space.
                    let (rx, ry) = dst_pixel_to_full_norm_f64(cx, cy, dst_w, dst_h, region);
                    assert!((rx - nx).abs() < 1e-9 && (ry - ny).abs() < 1e-9);
                }
            }
            assert!(
                max_norm < 1e-9,
                "norm roundtrip {region:?} {dst_w}x{dst_h}: {max_norm:e}"
            );
            assert!(
                max_pixel < 1e-9,
                "pixel roundtrip {region:?} {full_w}x{full_h} -> {dst_w}x{dst_h}: {max_pixel:e}"
            );
        }
    }

    /// The required `f32` API agrees with the `f64` core to within one `f32`
    /// ULP of the normalized coordinate, and normalized round trips stay well
    /// below `1e-5`.
    #[test]
    fn f32_norm_api_is_within_one_ulp_and_roundtrips_below_1e_5() {
        for &(region, _, _, dst_w, dst_h) in CASES {
            let (step_x, step_y) = (sample_step(dst_w), sample_step(dst_h));
            let mut max_norm_err = 0.0f64;
            let mut max_roundtrip = 0.0f64;
            for y in (0..dst_h).step_by(step_y as usize) {
                for x in (0..dst_w).step_by(step_x as usize) {
                    let (nx, ny) = dst_pixel_to_full_norm(x as f32, y as f32, dst_w, dst_h, region);
                    let (rx, ry) =
                        dst_pixel_to_full_norm_f64(x as f64, y as f64, dst_w, dst_h, region);
                    assert!(nx.is_finite() && ny.is_finite());
                    // One f32 ULP below 1.0 is 5.96e-8; allow the next binade.
                    max_norm_err = max_norm_err
                        .max((nx as f64 - rx).abs())
                        .max((ny as f64 - ry).abs());

                    // normalized -> dst -> normalized: the direction in which
                    // the f32 API can honour a < 1e-5 contract.
                    let (dx, dy) = full_norm_to_dst_pixel(nx, ny, dst_w, dst_h, region);
                    let (bx, by) = dst_pixel_to_full_norm(dx, dy, dst_w, dst_h, region);
                    max_roundtrip = max_roundtrip
                        .max((bx as f64 - nx as f64).abs())
                        .max((by as f64 - ny as f64).abs());
                }
            }
            assert!(
                max_norm_err <= 1.2e-7,
                "norm f32 error {region:?}: {max_norm_err:e} exceeds one ULP"
            );
            assert!(
                max_roundtrip < 1e-5,
                "norm->dst->norm {region:?} {dst_w}x{dst_h}: {max_roundtrip:e}"
            );
        }
    }

    /// `dst -> full pixel -> dst` through the `f32` API is bounded by the
    /// `f32` ULP of the full-image coordinate, scaled back into destination
    /// space. `*_f64` has no such bound (tested above).
    #[test]
    fn f32_pixel_roundtrip_stays_within_one_ulp_of_full_space() {
        for &(region, full_w, full_h, dst_w, dst_h) in CASES {
            let (step_x, step_y) = (sample_step(dst_w), sample_step(dst_h));
            let max_full_x = (region.right() as f64 * full_w as f64).max(1.0);
            let max_full_y = (region.bottom() as f64 * full_h as f64).max(1.0);
            let scale_x = dst_w as f64 / (region.w as f64 * full_w as f64);
            let scale_y = dst_h as f64 / (region.h as f64 * full_h as f64);
            let bound_x = max_full_x * f32::EPSILON as f64 * scale_x + 1e-6;
            let bound_y = max_full_y * f32::EPSILON as f64 * scale_y + 1e-6;

            let (mut err_x, mut err_y) = (0.0f64, 0.0f64);
            for y in (0..dst_h).step_by(step_y as usize) {
                for x in (0..dst_w).step_by(step_x as usize) {
                    let (fx, fy) = dst_pixel_to_full_pixel(
                        x as f32, y as f32, full_w, full_h, dst_w, dst_h, region,
                    );
                    let (rx, ry) =
                        full_pixel_to_dst_pixel(fx, fy, full_w, full_h, dst_w, dst_h, region);
                    err_x = err_x.max((rx as f64 - x as f64).abs());
                    err_y = err_y.max((ry as f64 - y as f64).abs());
                }
            }
            assert!(
                err_x <= bound_x && err_y <= bound_y,
                "pixel roundtrip {region:?} {full_w}x{full_h}->{dst_w}x{dst_h}: \
                 ({err_x:e}, {err_y:e}) > ({bound_x:e}, {bound_y:e})"
            );
            // Practical ceiling: even on a 4K-scale coordinate the `f32`
            // round trip stays far below a fifth of a pixel.
            assert!(err_x < 2e-4 && err_y < 2e-4, "({err_x:e}, {err_y:e})");
        }
    }

    /// The exactness requirement: with a `region` that is exactly
    /// representable in `f32` (dyadic fractions), a 1:1 crop maps integer
    /// pixel centers onto `full_x0 + dx` with **zero** error.
    #[test]
    #[allow(clippy::float_cmp)]
    fn one_to_one_crop_maps_pixel_indices_exactly() {
        // 2048x2048 image, 512x512 crop at (512, 1024).
        let region = NormRect::new(0.25, 0.5, 0.25, 0.25);
        for y in 0..512u32 {
            for x in 0..512u32 {
                let (fx, fy) =
                    dst_pixel_to_full_pixel(x as f32, y as f32, 2048, 2048, 512, 512, region);
                assert_eq!(fx, (512 + x) as f32, "x={x} y={y}");
                assert_eq!(fy, (1024 + y) as f32, "x={x} y={y}");
                let (bx, by) = full_pixel_to_dst_pixel(fx, fy, 2048, 2048, 512, 512, region);
                assert_eq!(bx, x as f32);
                assert_eq!(by, y as f32);
            }
        }

        // 1360x912 image, 340x228 crop at (170, 228) — sizes are not powers
        // of two, but every region field stays exact.
        let region = NormRect::new(0.125, 0.25, 0.25, 0.25);
        for y in 0..228u32 {
            for x in 0..340u32 {
                let (fx, fy) =
                    dst_pixel_to_full_pixel(x as f32, y as f32, 1360, 912, 340, 228, region);
                assert_eq!(fx, (170 + x) as f32);
                assert_eq!(fy, (228 + y) as f32);
            }
        }
    }

    /// With genuinely odd dimensions `region.w == dst_w / full_w` cannot be
    /// represented exactly in `f32`, so a residual proportional to the
    /// representation error — not to the transform — remains. It must stay
    /// far below half a pixel.
    #[test]
    fn one_to_one_odd_canvas_residual_is_only_representation() {
        let (full_w, full_h, dst_w, dst_h) = (1365u32, 911u32, 341u32, 228u32);
        let region = NormRect::new(
            0.0,
            0.0,
            dst_w as f32 / full_w as f32,
            dst_h as f32 / full_h as f32,
        );
        let mut max = 0.0f32;
        for y in 0..dst_h {
            for x in 0..dst_w {
                let (fx, fy) = dst_pixel_to_full_pixel(
                    x as f32, y as f32, full_w, full_h, dst_w, dst_h, region,
                );
                max = max.max((fx - x as f32).abs());
                max = max.max((fy - y as f32).abs());
            }
        }
        assert!(
            max < 1e-5,
            "odd-size 1:1 residual {max:.7} px (measured ~1e-8 for this canvas; \
             f32 representation noise must stay below 1e-5)"
        );
    }

    /// Regression for the original bug: for a zoomed viewport the top-left
    /// canvas pixel must sample half a *destination* pixel into the region —
    /// i.e. `region.x * full_w + 0.5 * (region.w * full_w / dst_w)` — not
    /// half a full pixel less.
    #[test]
    fn non_full_region_has_no_half_pixel_shift() {
        let region = NormRect::new(0.25, 0.33, 0.25, 0.25);
        let (full_w, full_h, dst_w, dst_h) = (1365u32, 911u32, 341u32, 228u32);

        let (fx, fy) = dst_pixel_to_full_pixel(0.0, 0.0, full_w, full_h, dst_w, dst_h, region);
        let expected_x =
            (region.x as f64 + 0.5 / dst_w as f64 * region.w as f64) * full_w as f64 - 0.5;
        let expected_y =
            (region.y as f64 + 0.5 / dst_h as f64 * region.h as f64) * full_h as f64 - 0.5;
        assert_close(fx, expected_x as f32, 1e-4, "top-left full x");
        assert_close(fy, expected_y as f32, 1e-4, "top-left full y");

        // The old `region.x + dx / dst_w * region.w - 0.5` form produced
        // 340.75 here; the correct center is above 341.25.
        assert!(fx > 341.25, "half-pixel shift regressed: fx = {fx}");

        // And the renderer identity itself, at the sub-pixel level.
        let (nx, ny) = dst_pixel_to_full_norm(0.0, 0.0, dst_w, dst_h, region);
        let renderer_nx = region.x + 0.5 / dst_w as f32 * region.w;
        let renderer_ny = region.y + 0.5 / dst_h as f32 * region.h;
        assert_close(nx, renderer_nx, 1e-6, "renderer nx");
        assert_close(ny, renderer_ny, 1e-6, "renderer ny");
    }

    #[test]
    fn full_region_is_identity_in_pixel_index_space() {
        for &(full_w, full_h) in &[(1365u32, 911u32), (1, 1), (7, 13)] {
            let (fx, fy) =
                dst_pixel_to_full_pixel(3.0, 5.0, full_w, full_h, full_w, full_h, NormRect::FULL);
            assert_eq!((fx, fy), (3.0, 5.0));
            let (nx, ny) = dst_pixel_to_full_norm(0.0, 0.0, full_w, full_h, NormRect::FULL);
            assert_close(nx, 0.5 / full_w as f32, 1e-7, "FULL first center x");
            assert_close(ny, 0.5 / full_h as f32, 1e-7, "FULL first center y");
            let (bx, by) = full_norm_to_dst_pixel(nx, ny, full_w, full_h, NormRect::FULL);
            assert_close(bx, 0.0, 1e-7, "FULL center back x");
            assert_close(by, 0.0, 1e-7, "FULL center back y");
        }
    }

    #[test]
    fn landmark_on_full_pixel_center_lands_on_its_dst_position() {
        let region = NormRect::new(0.25, 0.33, 0.25, 0.25);
        let (full_w, full_h, dst_w, dst_h) = (1365u32, 911u32, 341u32, 228u32);
        // Center of full pixel (512, 400), inside the viewport.
        let (dx, dy) = full_pixel_to_dst_pixel(512.0, 400.0, full_w, full_h, dst_w, dst_h, region);
        let expected_x =
            (512.5f64 / full_w as f64 - region.x as f64) / region.w as f64 * dst_w as f64 - 0.5;
        let expected_y =
            (400.5f64 / full_h as f64 - region.y as f64) / region.h as f64 * dst_h as f64 - 0.5;
        assert_close(dx, expected_x as f32, 1e-4, "landmark dst x");
        assert_close(dy, expected_y as f32, 1e-4, "landmark dst y");
        assert!(dx >= 0.0 && dx <= (dst_w - 1) as f32, "dx = {dx}");
        assert!(dy >= 0.0 && dy <= (dst_h - 1) as f32, "dy = {dy}");
    }

    #[test]
    fn invalid_inputs_return_sentinel_instead_of_panicking() {
        let nan = f32::NAN;
        let inf = f32::INFINITY;
        let r = NormRect::new(0.25, 0.25, 0.5, 0.5);

        assert!(dst_pixel_to_full_norm(0.0, 0.0, 0, 10, r).0.is_nan());
        assert!(dst_pixel_to_full_norm(0.0, 0.0, 10, 0, r).0.is_nan());
        assert!(full_norm_to_dst_pixel(0.5, 0.5, 0, 10, r).0.is_nan());
        assert!(
            full_pixel_to_dst_pixel(0.0, 0.0, 10, 10, 0, 10, r)
                .0
                .is_nan()
        );
        assert!(
            dst_pixel_to_full_pixel(0.0, 0.0, 0, 10, 10, 10, r)
                .0
                .is_nan()
        );

        for bad in [
            NormRect::new(0.0, 0.0, 0.0, 1.0),
            NormRect::new(0.0, 0.0, -0.25, 1.0),
            NormRect::new(0.0, 0.0, 1.0, 0.0),
            NormRect::new(nan, 0.0, 1.0, 1.0),
            NormRect::new(0.0, inf, 1.0, 1.0),
            NormRect::new(0.0, 0.0, inf, 1.0),
        ] {
            assert!(dst_pixel_to_full_norm(0.0, 0.0, 10, 10, bad).0.is_nan());
            assert!(full_norm_to_dst_pixel(0.5, 0.5, 10, 10, bad).0.is_nan());
            assert!(
                full_pixel_to_dst_pixel(1.0, 1.0, 10, 10, 10, 10, bad)
                    .0
                    .is_nan()
            );
            assert!(
                dst_pixel_to_full_pixel(1.0, 1.0, 10, 10, 10, 10, bad)
                    .0
                    .is_nan()
            );
            assert!(
                dst_pixel_to_full_norm_f64(0.0, 0.0, 10, 10, bad).0.is_nan(),
                "f64 core must reject {bad:?}"
            );
            assert_eq!(
                intersect_roi_with_viewport(NormRect::FULL, bad, 10, 10),
                None
            );
        }

        assert!(dst_pixel_to_full_norm(nan, 0.0, 10, 10, r).0.is_nan());
        assert!(dst_pixel_to_full_norm(0.0, inf, 10, 10, r).0.is_nan());
        assert!(full_norm_to_dst_pixel(nan, 0.5, 10, 10, r).0.is_nan());
        assert!(
            full_pixel_to_dst_pixel(0.0, inf, 10, 10, 10, 10, r)
                .0
                .is_nan()
        );
        assert!(
            dst_pixel_to_full_pixel(inf, 0.0, 10, 10, 10, 10, r)
                .0
                .is_nan()
        );
        assert!(
            dst_pixel_to_full_norm_f64(f64::NAN, 0.0, 10, 10, r)
                .0
                .is_nan()
        );

        assert_eq!(intersect_roi_with_viewport(NormRect::FULL, r, 0, 10), None);
        assert_eq!(intersect_roi_with_viewport(NormRect::FULL, r, 10, 0), None);
        assert_eq!(
            intersect_roi_with_viewport(NormRect::new(nan, 0.0, 0.5, 0.5), r, 10, 10),
            None
        );
        assert_eq!(
            intersect_roi_with_viewport(NormRect::new(0.0, 0.0, 0.0, 0.5), r, 10, 10),
            None
        );
    }

    #[test]
    fn viewport_sized_roi_covers_the_whole_canvas() {
        assert_eq!(
            intersect_roi_with_viewport(NormRect::FULL, NormRect::FULL, 400, 300),
            Some((0, 0, 400, 300))
        );
        let region = NormRect::new(0.25, 0.25, 0.5, 0.5);
        assert_eq!(
            intersect_roi_with_viewport(NormRect::FULL, region, 400, 400),
            Some((0, 0, 400, 400))
        );
        assert_eq!(
            intersect_roi_with_viewport(region, region, 400, 400),
            Some((0, 0, 400, 400))
        );
    }

    #[test]
    fn roi_intersection_uses_pixel_centers() {
        let region = NormRect::new(0.25, 0.25, 0.5, 0.5);
        // Intersection is x/y in 0.25..=0.5, i.e. the first 200 of 400 pixels.
        assert_eq!(
            intersect_roi_with_viewport(NormRect::new(0.0, 0.0, 0.5, 0.5), region, 400, 400),
            Some((0, 0, 200, 200))
        );
        // ROI strictly inside the viewport: 0.3..=0.35 of the full image.
        assert_eq!(
            intersect_roi_with_viewport(NormRect::new(0.3, 0.3, 0.05, 0.05), region, 400, 400),
            Some((40, 40, 40, 40))
        );
        // ROI inside the viewport, 0.375..=0.625.
        assert_eq!(
            intersect_roi_with_viewport(NormRect::new(0.375, 0.375, 0.25, 0.25), region, 400, 400),
            Some((100, 100, 200, 200))
        );
        // Partially outside on the right/bottom: clipped to the canvas.
        assert_eq!(
            intersect_roi_with_viewport(NormRect::new(0.6, 0.6, 0.4, 0.4), region, 400, 400),
            Some((280, 280, 120, 120))
        );
    }

    #[test]
    fn roi_without_overlap_is_none() {
        let region = NormRect::new(0.5, 0.5, 0.25, 0.25);
        assert_eq!(
            intersect_roi_with_viewport(NormRect::new(0.0, 0.0, 0.25, 0.25), region, 400, 400),
            None
        );
        assert_eq!(
            intersect_roi_with_viewport(NormRect::new(0.5, 0.0, 0.25, 0.25), region, 400, 400),
            None
        );
        assert_eq!(
            intersect_roi_with_viewport(NormRect::new(0.75, 0.5, 0.25, 0.25), region, 400, 400),
            None
        );
    }

    #[test]
    fn single_full_pixel_roi_maps_to_one_dst_pixel_at_one_to_one() {
        let (full_w, full_h, dst_w, dst_h) = (1360u32, 912u32, 340u32, 228u32);
        let region = NormRect::new(0.125, 0.25, 0.25, 0.25); // 340x228 at (170, 228)
        let pixel_roi = NormRect::new(
            200.0 / full_w as f32,
            300.0 / full_h as f32,
            1.0 / full_w as f32,
            1.0 / full_h as f32,
        );
        assert_eq!(
            intersect_roi_with_viewport(pixel_roi, region, dst_w, dst_h),
            Some((30, 72, 1, 1))
        );
    }

    #[test]
    fn roi_edges_align_with_corner_transforms() {
        let (dst_w, dst_h) = (341u32, 228u32);
        let region = NormRect::new(0.25, 0.33, 0.25, 0.25);
        let roi = NormRect::new(0.3, 0.36, 0.1, 0.1);
        let (x0, y0, w, h) = intersect_roi_with_viewport(roi, region, dst_w, dst_h).unwrap();

        let (lx, ly) = full_norm_to_dst_pixel(roi.x, roi.y, dst_w, dst_h, region);
        let (rx, ry) = full_norm_to_dst_pixel(roi.right(), roi.bottom(), dst_w, dst_h, region);
        assert!((lx - x0 as f32).abs() < 1.0);
        assert!((ly - y0 as f32).abs() < 1.0);
        assert!((rx - (x0 + w) as f32).abs() < 1.0);
        assert!((ry - (y0 + h) as f32).abs() < 1.0);
        assert_eq!((w, h), (137, 92));
    }

    #[test]
    fn norm_rect_intersection_is_continuous() {
        let a = NormRect::new(0.25, 0.25, 0.5, 0.5);
        let b = NormRect::new(0.0, 0.0, 0.5, 0.5);
        assert_eq!(
            a.intersection(b),
            Some(NormRect::new(0.25, 0.25, 0.25, 0.25))
        );
        assert_eq!(
            NormRect::new(0.0, 0.0, 0.1, 0.1).intersection(NormRect::new(0.5, 0.5, 0.1, 0.1)),
            None
        );
        assert_eq!(a.intersection(NormRect::new(f32::NAN, 0.0, 1.0, 1.0)), None);
        assert_eq!(NormRect::default(), NormRect::FULL);
    }
}
