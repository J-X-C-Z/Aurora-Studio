//! Capture One lateral-CA pixel kernel, transcribed from `p_041444160.cl`
//! (kernel `z88d05a530c`, 4920 bytes).
//!
//! This module implements the **verified** part only:
//! - piecewise cubic Hermite sampling of a `(radius, value, slope)` knot table,
//!   including the kernel's `|x1-x0| < 1e-4` degenerate-segment branch;
//! - the hard +/-3 clamp applied to both colour curves;
//! - radius from an explicit optical centre in **scaled/offset pixel units**;
//! - radial displacement `(K/r)*(dx,dy)*curve`, with the *real-valued* offset
//!   passed to the bilinear sampler (floor base + fractional weights), applied
//!   to R and B only; G is copied unchanged.
//!
//! Host-side fold (2026-11-25): `ImgCoreDll!0x18131c920` builds exactly two
//! tables from a compact struct that holds one count and two <=16-point
//! curves (red at `+0x08`, blue at `+0x88`, count at `+0x04`). Each point is
//! scaled in f32 by `1/reference` per axis (settings vtable `+0x848`; 1.0 when
//! absent) and becomes a `(x, y, slope, 0)` row at parameter-store `+0x1008`
//! (red) / `+0x1220` (blue), bound as kernel arg 9 / arg 11 by
//! `0x18173e2a0`. The builder has no reference to the six
//! `IC_LensChromaFringing` counts/curves at `+0x17f0..+0x1804`; those feed the
//! purple-fringing operator instead. Effective weights for this kernel are
//! therefore shift = 1, ra = 0, strength = 1, and [`fold_table`] implements
//! exactly that. The 2-point slope is the proven secant; n>2 uses the generic
//! spline tangent (central difference, grade C).

/// Hard clamp of the kernel's curve output (`z17a085e5c8 = 3.0f`).
pub const CURVE_CLAMP: f32 = 3.0;
/// Exact-hit tolerance of the kernel's binary search and degenerate segment.
const KNOT_EPS: f32 = 1e-4;
/// Radius floor in the kernel (`max(r, 1e-20f)`).
const R_MIN: f32 = 1e-20;

/// One `float4` knot: x = radius, y = value, z = slope (w unused).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Knot {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Knot {
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// The six curves of `IC_LensChromaFringing` / `<chromaFringe>`.
///
/// Only the names and ordering are confirmed (managed metadata + XML strings).
/// They are **not** folded into the two kernel tables: the table builder
/// (`0x18131c920`) reads only the compact two-curve struct, and the six-curve
/// counts are consumed by the purple-fringing path. This struct remains a
/// naming contract and is deliberately not consumed by [`apply_ca`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CaSixCurves<Table> {
    pub red_shift: Table,
    pub red_ra: Table,
    pub red_strength: Table,
    pub blue_shift: Table,
    pub blue_ra: Table,
    pub blue_strength: Table,
}

/// Kernel parameters, with the same units as the OpenCL arguments.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaParams {
    /// `zfe5849ca60`: pixel offset added before scaling.
    pub offset: (f32, f32),
    /// `zf897261185`: explicit optical centre, in the scaled coordinate space.
    pub center: (f32, f32),
    /// `z6df7ccb0a4`: scale applied to `(x + offset.x + 0.5, y + offset.y + 0.5)`.
    pub scale: f32,
    /// `zcd810bcd22`: linear multiplier of the radial displacement.
    pub k: f32,
}

impl Default for CaParams {
    fn default() -> Self {
        Self { offset: (0.0, 0.0), center: (0.0, 0.0), scale: 1.0, k: 1.0 }
    }
}

/// Kernel binary search `z07f76dfe04`: exact hit within `1e-4`, otherwise the
/// left endpoint of the segment containing `x`.
fn search(x: f32, knots: &[Knot]) -> usize {
    debug_assert!(!knots.is_empty());
    let mut lo = 0usize;
    let mut hi = knots.len() - 1;
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if (x - knots[mid].x).abs() < KNOT_EPS {
            return mid;
        }
        if x < knots[mid].x {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    lo
}

/// Piecewise cubic Hermite sampler `z9369a44638`.
///
/// The last segment reuses its own endpoint as a defensive stand-in for the
/// out-of-range read in the kernel; no C1 runtime table has been observed
/// beyond its last knot, so this fallback is a module-level choice, not a
/// recovered C1 behaviour.
pub fn sample_curve(x: f32, knots: &[Knot]) -> f32 {
    if knots.is_empty() {
        return 0.0;
    }
    if knots.len() == 1 {
        return knots[0].y;
    }
    let i = search(x, knots);
    let j = (i + 1).min(knots.len() - 1);
    let k0 = knots[i];
    let k1 = knots[j];
    let d = k1.x - k0.x;
    if d.abs() < KNOT_EPS {
        return ((k0.y + k1.y) + (x - k0.x) * (k0.z + k1.z)) * 0.5;
    }
    let t = (x - k0.x) / d;
    let tm = t - 1.0;
    let h = k0.y + (k1.y - k0.y) * t * t * (3.0 - 2.0 * t);
    let k = (k0.z * tm + k1.z * t) * t * tm * d;
    h + k
}

/// `zcfffc38948`: Hermite sample hard-clamped to +/-3.
pub fn curve(x: f32, knots: &[Knot]) -> f32 {
    sample_curve(x, knots).clamp(-CURVE_CLAMP, CURVE_CLAMP)
}

/// Bilinear fetch of one channel; the kernel's floor/fraction split applied to
/// a *real-valued* source coordinate (no pre-flooring of the displacement).
/// Pixel access is clamped to the image edge by this module as a defensive
/// measure; the OpenCL kernel itself has no bounds check at this point.
pub fn bilinear_channel(
    pixels: &[f32],
    width: usize,
    height: usize,
    x: f32,
    y: f32,
    channel: usize,
) -> f32 {
    debug_assert!(width > 0 && height > 0 && pixels.len() >= width * height * 4);
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let at = |xi: i32, yi: i32| -> f32 {
        let xx = (xi as isize).clamp(0, width as isize - 1) as usize;
        let yy = (yi as isize).clamp(0, height as isize - 1) as usize;
        pixels[(yy * width + xx) * 4 + channel]
    };
    let xi = x0 as i32;
    let yi = y0 as i32;
    let top = at(xi, yi) + (at(xi + 1, yi) - at(xi, yi)) * fx;
    let bottom = at(xi, yi + 1) + (at(xi + 1, yi + 1) - at(xi, yi + 1)) * fx;
    top + (bottom - top) * fy
}

/// Radial displacement of one pixel, before the clamp is applied to the curve.
///
/// Returns `(dx, dy, radius)` where `dx,dy` are already scaled by `K/r`.
pub fn radial_unit(x: usize, y: usize, p: CaParams) -> (f32, f32, f32) {
    let px = ((x as f32) + p.offset.0 + 0.5) * p.scale;
    let py = ((y as f32) + p.offset.1 + 0.5) * p.scale;
    let dx = px - p.center.0;
    let dy = py - p.center.1;
    let r = dx.hypot(dy).max(R_MIN);
    let factor = p.k / r;
    (dx * factor, dy * factor, r)
}

/// Full kernel step for one pixel: R/B are resampled at the radial offset,
/// G is copied from the source. Output is RGBA with alpha 0, as in the kernel.
///
/// `red_table` and `blue_table` are the host-supplied folded tables produced
/// by [`fold_table`]; the six C1 curve names in [`CaSixCurves`] are *not*
/// consumed here because the native builder does not read them.
pub fn apply_ca(
    src: &[f32],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    p: CaParams,
    red_table: &[Knot],
    blue_table: &[Knot],
) -> [f32; 4] {
    let (ux, uy, r) = radial_unit(x, y, p);
    let dr = curve(r, red_table);
    let db = curve(r, blue_table);
    let green = src[(y * width + x) * 4 + 1];
    [
        bilinear_channel(src, width, height, x as f32 + ux * dr, y as f32 + uy * dr, 0),
        green,
        bilinear_channel(src, width, height, x as f32 + ux * db, y as f32 + uy * db, 2),
        0.0,
    ]
}


/// One point of a C1 CA profile curve: `x` is the radius in the profile unit,
/// `y` the red/blue displacement in the same unit family.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CaPoint {
    pub x: f32,
    pub y: f32,
}

impl CaPoint {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Host-side fold into one kernel table, matching `ImgCoreDll!0x18131c920`.
///
/// The DLL scales x and y in f32 by `1/ref_x` and `1/ref_y` (1.0 when the
/// settings object supplies no reference at vtable `+0x848`), stores the points as
/// doubles and computes the tangent:
/// - `n == 0` -> empty table (the kernel is only bound when `n >= 2`);
/// - `n == 1` -> slope 0;
/// - `n == 2` -> exact secant `(y1-y0)/(x1-x0)` (proven at `0x1814283d1`);
/// - `n >= 3` -> central difference over `[i-1, i+1]`, one-sided at the ends
///   (reconstruction of the generic spline builder, grade C).
///
/// `ra` and `strength` are intentionally absent: the native builder has no
/// input for them. The effective weights are shift = 1, ra = 0, strength = 1.
pub fn fold_table(points: &[CaPoint], ref_x: f32, ref_y: f32) -> Vec<Knot> {
    let inv = |r: f32| if r.is_finite() && r != 0.0 { 1.0f32 / r } else { 1.0 };
    let (inv_x, inv_y) = (inv(ref_x), inv(ref_y));
    let scaled: Vec<(f64, f64)> = points
        .iter()
        .map(|p| ((p.x * inv_x) as f64, (p.y * inv_y) as f64))
        .collect();
    let n = scaled.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let slope = if n < 2 {
            0.0
        } else if n == 2 {
            let dx = scaled[1].0 - scaled[0].0;
            if dx == 0.0 { 0.0 } else { (scaled[1].1 - scaled[0].1) / dx }
        } else {
            let (a, b) = if i == 0 {
                (scaled[0], scaled[1])
            } else if i == n - 1 {
                (scaled[n - 2], scaled[n - 1])
            } else {
                (scaled[i - 1], scaled[i + 1])
            };
            let dx = b.0 - a.0;
            if dx == 0.0 { 0.0 } else { (b.1 - a.1) / dx }
        };
        out.push(Knot::new(scaled[i].0 as f32, scaled[i].1 as f32, slope as f32));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() <= eps
    }

    /// Ramp image: R = x, G = 7 constant, B = y, A = 1. Linear channels make
    /// the bilinear result equal to the sampled coordinate.
    fn ramp(w: usize, h: usize) -> Vec<f32> {
        let mut v = vec![0.0; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                v[i] = x as f32;
                v[i + 1] = 7.0;
                v[i + 2] = y as f32;
                v[i + 3] = 1.0;
            }
        }
        v
    }

    #[test]
    fn hermite_hits_the_known_knots_and_reproduces_a_straight_line() {
        let k = [Knot::new(0.0, 0.0, 0.002), Knot::new(1000.0, 2.0, 0.002)];
        for x in [0.0, 250.0, 500.0, 1000.0] {
            assert!(close(sample_curve(x, &k), 0.002 * x, 2e-5), "x={x}");
        }
    }

    #[test]
    fn hermite_exact_hit_within_one_ten_thousandth() {
        // Binary search treats |X-x_i|<1e-4 as an exact knot hit.
        let k = [
            Knot::new(0.0, 0.0, 0.0),
            Knot::new(10.0, 1.0, 0.0),
            Knot::new(20.0, 4.0, 0.0),
        ];
        assert!(close(sample_curve(10.0 + 5e-5, &k), 1.0, 2e-6));
        assert!(close(sample_curve(10.0 - 5e-5, &k), 1.0, 2e-6));
    }

    #[test]
    fn hermite_degenerate_segment_uses_the_kernel_half_slope_formula() {
        // Two knots 5e-5 apart: the binary search declares an exact hit at
        // knot 1 when X is within 1e-4, then the sampler sees |x1-x0| < 1e-4
        // and uses the kernel's half-sum/half-slope formula. The ordinary
        // Hermite expression would divide by 5e-5 and blow up.
        // With three knots the search's only probe is knot 1 at x=10; a
        // query at 10.0002 is outside its 1e-4 exact-hit window, so search
        // returns index 1 and the sampler sees the 5e-5-wide next segment.
        let k = [
            Knot::new(0.0, 0.0, 0.0),
            Knot::new(10.0, 1.0, 0.0),
            Knot::new(10.00005, 2.0, 3.0),
        ];
        let x = 10.0002;
        let want = ((1.0 + 2.0) + (x - 10.0) * (0.0 + 3.0)) * 0.5;
        assert!(close(sample_curve(x, &k), want, 1e-6));
    }

    #[test]
    fn curve_clamps_to_plus_minus_three() {
        let k = [Knot::new(0.0, 0.0, 0.0), Knot::new(100.0, 10.0, 0.0)];
        assert!(close(curve(100.0, &k), 3.0, 1e-7));
        let n = [Knot::new(0.0, 0.0, 0.0), Knot::new(100.0, -10.0, 0.0)];
        assert!(close(curve(100.0, &n), -3.0, 1e-7));
        // In-range values are untouched by the clamp.
        let s = [Knot::new(0.0, 0.0, 0.002), Knot::new(1000.0, 2.0, 0.002)];
        assert!(close(curve(500.0, &s), 1.0, 2e-5));
    }

    #[test]
    fn radial_displacement_uses_explicit_center_and_scale() {
        let p = CaParams {
            offset: (0.0, 0.0),
            center: (31.5, 31.5),
            scale: 1.0,
            k: 2.0,
        };
        // x=35,y=31 -> px=35.5,py=31.5 => dx=4,dy=0,r=4,K/r=0.5 => ux=2.
        let (ux, uy, r) = radial_unit(35, 31, p);
        assert!(close(r, 4.0, 1e-6));
        assert!(close(ux, 2.0, 1e-6));
        assert!(close(uy, 0.0, 1e-6));
        // scale==2 scales (x+0.5, y+0.5) before subtracting the centre:
        // dx=2*35.5-31.5=39.5, dy=2*31.5-31.5=31.5.
        let p2 = CaParams { scale: 2.0, ..p };
        let (_, _, r2) = radial_unit(35, 31, p2);
        assert!(close(r2, 39.5f32.hypot(31.5), 1e-6));
    }

    #[test]
    fn bilinear_uses_real_valued_displacement_not_floored_offset() {
        let img = ramp(16, 16);
        // A fractional offset must produce a fractional sample: flooring the
        // offset before sampling would make the weight identically zero.
        let v = bilinear_channel(&img, 16, 16, 3.25, 4.75, 0);
        assert!(close(v, 3.25, 1e-6), "R {v}");
        let b = bilinear_channel(&img, 16, 16, 3.25, 4.75, 2);
        assert!(close(b, 4.75, 1e-6), "B {b}");
    }

    #[test]
    fn apply_ca_shifts_red_and_blue_only_with_fractional_weights() {
        let img = ramp(64, 64);
        // Constant +3 red / -3 blue tables. With the centre on the same row
        // and 0.5 px left of the sample, displacement is (+3,0) / (-3,0).
        let red = [Knot::new(0.0, 3.0, 0.0), Knot::new(64.0, 3.0, 0.0)];
        let blue = [Knot::new(0.0, -3.0, 0.0), Knot::new(64.0, -3.0, 0.0)];
        // Centre at the pixel's own sample position makes dx=0.5, dy=0, so
        // the displacement is exactly (+3,0) / (-3,0) along the x axis.
        let p = CaParams { center: (35.0, 31.5), ..CaParams::default() };
        let out = apply_ca(&img, 64, 64, 35, 31, p, &red, &blue);
        assert!(close(out[0], 38.0, 1e-5), "red {}" , out[0]);
        assert!(close(out[1], 7.0, 1e-6), "green must be untouched");
        // The ramp encodes B = y, and the displacement is horizontal, so B
        // stays 31 even though the sample coordinate moved from 35 to 32.
        assert!(close(out[2], 31.0, 1e-5), "blue {}", out[2]);
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn shift_is_clamped_at_three_pixels_even_for_large_table_values() {
        // y=10 at r=100 would ask for 10 px; kernel clamps to 3 px.
        let img = ramp(128, 128);
        let red = [Knot::new(0.0, 0.0, 0.0), Knot::new(100.0, 10.0, 0.0)];
        let blue = [Knot::new(0.0, 0.0, 0.0), Knot::new(100.0, 10.0, 0.0)];
        // radius 0.5 on the +x axis: the Hermite ramp to 10 at r=100 yields
        // 10*t^2*(3-2t) with t=0.005, below the clamp. The hard limit itself
        // is checked in curve_clamps_to_plus_minus_three.
        let p = CaParams { center: (35.0, 31.5), ..CaParams::default() };
        let ramp = ramp(128, 128);
        let out = apply_ca(&ramp, 128, 128, 35, 31, p, &red, &blue);
        let t = 0.5f32 / 100.0;
        let want = 35.0 + 10.0 * t * t * (3.0 - 2.0 * t);
        assert!(close(out[0], want, 2e-5), "red {} want {}", out[0], want);
        assert!(close(curve(100.0, &red), 3.0, 1e-7));
    }

    #[test]
    fn six_curve_names_do_not_claim_a_verified_weight() {
        // The struct exists only as a naming contract. In particular no
        // `strength * (shift + ra) * r` weighting is implemented here.
        let c = CaSixCurves {
            red_shift: [0.0f32],
            red_ra: [0.0f32],
            red_strength: [0.0f32],
            blue_shift: [0.0f32],
            blue_ra: [0.0f32],
            blue_strength: [0.0f32],
        };
        assert_eq!(c.red_strength, c.red_ra);
    }

    #[test]
    fn fold_two_points_use_the_proven_secant() {
        let t = fold_table(&[CaPoint::new(0.0, 0.0), CaPoint::new(1000.0, 2.0)], 1.0, 1.0);
        assert_eq!(t.len(), 2);
        assert!(close(t[0].z, 0.002, 1e-9));
        assert!(close(sample_curve(250.0, &t), 0.5, 1e-6));
    }

    #[test]
    fn fold_scales_both_axes_by_the_reference() {
        // x/2000 and y/4: (1000,2) -> (0.5,0.5), so the slope becomes 1.
        let t = fold_table(&[CaPoint::new(0.0, 0.0), CaPoint::new(1000.0, 2.0)], 2000.0, 4.0);
        assert!(close(t[1].x, 0.5, 1e-7));
        assert!(close(t[1].y, 0.5, 1e-7));
        assert!(close(t[0].z, 1.0, 1e-9));
    }

    #[test]
    fn fold_has_no_ra_or_strength_input() {
        // The function takes exactly two arguments; there is no weight knob.
        // A six-curve struct cannot be passed to it, which is the point: the
        // native builder reads only the compact two-curve struct.
        let _: fn(&[CaPoint], f32, f32) -> Vec<Knot> = fold_table;
    }

    #[test]
    fn folded_table_clamps_and_hits_linear_values() {
        let t = fold_table(
            &[CaPoint::new(0.0, 0.0), CaPoint::new(50.0, 2.0), CaPoint::new(100.0, 4.0)],
            1.0,
            1.0,
        );
        assert!(close(sample_curve(50.0, &t), 2.0, 1e-5));
        assert!(close(sample_curve(25.0, &t), 1.0, 1e-4));
        assert!(close(curve(100.0, &t), 3.0, 1e-7));
    }
}
