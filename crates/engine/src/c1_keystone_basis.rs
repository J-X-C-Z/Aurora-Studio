//! Exact `CResampler::UpdateGeo` keystone basis, transcribed from Capture One's
//! `ImgCoreDll.dll` (0x18141d980 in the 2026-09 build, 0x18141da30 in the
//! 2026-11 build; 2114 bytes, identical instructions).
//!
//! Scope (static reconstruction only): this module computes the six doubles
//! written by `UpdateGeo` and the Cramer inverse at 0x18141eba0. It does **not**
//! reproduce the full `CalculateGeo` (crop/rotation/flip composition), and no
//! C1 UI-on-screen / real-photo comparison has been performed.
//!
//! # Unit conventions at this module's boundary
//!
//! | argument | stored C1 unit | conversion inside `update_geo` |
//! |---|---|---|
//! | `tilt_h`, `tilt_v` | percent (stored `tiltX/tiltY`; C1 parallelize writes `100*cos(A)*tan(B)/c`) | `* double(float(0.01))` |
//! | `f_len` | C1 `c`; <=0 or `IGNORE_FOCAL_LENGTH` means the `c<=0` branch | none (compared to 0) |
//! | `skew` | **degrees** (`ZGEOMETRYKEYSTONESKEW`) | `to_radians() * 0.5`, then `tan` |
//! | `aspect` | **percent** (`ZGEOMETRYKEYSTONEASPECT`) | `/ 100`, +/-1e-4 dead zone |
//! | `roll` | **radians** | none (sin/cos) |
//! | `width`, `height` | crop pixel size (`ctx+0`, `ctx+4`) | `hypot` |
//!
//! The forward map uses **centre-relative** coordinates:
//! `u = (m00*x + m01*y) / (m20*x + m21*y + 1)` and likewise for `v`.
//! The `1e-8` vector computed in the `c<=0` branch at 0x18141dcad is only used
//! to select a helper; 0x18141ddef discards it and keeps the *unnormalised*
//! basis. That behaviour is preserved below and asserted by the tests.

/// Percentage-to-fraction constant at 0x182a37a08: `double(float(0.01))`.
const PCT: f64 = 0.009_999_999_776_482_582;
/// Normalisation threshold at 0x182a37378.
pub const EPS_NORM: f64 = 1e-8;
/// Aspect dead zone, 0x182a37718 / 0x182a3e0e8.
const EPS_ASPECT: f64 = 1e-4;
/// L1 floor of the binary's vector normaliser: `3 * DBL_MIN`.
const L1_MIN: f64 = 6.675_221_575_521_604e-308;
/// C1 keystone flag bit 0: `keystoneFlagIgnoreFocalLength`.
pub const FLAG_IGNORE_FOCAL_LENGTH: u32 = 1;

/// Six doubles from `UpdateGeo` plus the aspect selector and stretch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KeystoneBasis {
    /// First column (x basis): `out+0x00`.
    pub m00: f64,
    pub m10: f64,
    pub m20: f64,
    /// Second column (y basis): `out+0x18`.
    pub m01: f64,
    pub m11: f64,
    pub m21: f64,
    /// `out+0x40/+0x48`: `(1,0)` for positive aspect, `(0,1)` for negative.
    pub aspect_axis: (f64, f64),
    /// `out+0x50`: `1 + |aspect|` in fraction units, 0 in the dead zone.
    pub stretch: f64,
}

/// Reproduce `UpdateGeo(ctx, out, in, arg4, arg5)` for the C1 keystone path.
///
/// `skew` is in **degrees** and `aspect` in **percent** at this boundary; the
/// caller must not pre-convert them. `flags & 1` forces C1's `c` to zero.
pub fn update_geo(
    tilt_h: f64,
    tilt_v: f64,
    f_len: f64,
    skew_deg: f64,
    aspect_pct: f64,
    roll_rad: f64,
    width: f64,
    height: f64,
    flags: u32,
) -> KeystoneBasis {
    let a = tilt_h * PCT;
    let b = tilt_v * PCT;
    let c = if flags & FLAG_IGNORE_FOCAL_LENGTH != 0 { 0.0 } else { f_len };

    // 0x18141da14: only a strictly positive c multiplies the stored tilt.
    let (x, y) = if c > 0.0 { (c * a, c * b) } else { (a, b) };
    let length = x.hypot(y);
    let (s1, s2) = if length > EPS_NORM {
        (x / length, -y / length)
    } else {
        (0.0, 1.0)
    };

    // 0x18141dacd..0x18141db0a: percent -> fraction; +/-1e-4 dead zone.
    let aspect = aspect_pct / 100.0;
    let (stretch, aspect_axis) = if aspect > EPS_ASPECT {
        (1.0 + aspect, (1.0, 0.0))
    } else if aspect < -EPS_ASPECT {
        (1.0 - aspect, (0.0, 1.0))
    } else {
        (0.0, (1.0, 0.0))
    };

    // 0x18141dcad builds a 1e-8-scaled vector only to choose a helper;
    // 0x18141ddef drops it and reloads the raw vector. Keep that distinction.
    let n = if c > 0.0 {
        normalize3([-s1, s2, s1 * x - s2 * y])
    } else {
        [-s1, s2, s1 * x - s2 * y]
    };

    // 0x18141de12..0x18141dee4: rotate by -atan2(s1, s2).
    let theta = -s1.atan2(s2);
    let (sn, cs) = theta.sin_cos();
    let c0 = [n[0] * sn + cs * s2, n[1] * sn + cs * s1, n[2] * sn];
    let c1 = [n[0] * cs - sn * s2, n[1] * cs - sn * s1, n[2] * cs];

    // 0x18141deea..0x18141df59: symmetric shear, tan(skew/2) with skew in radians.
    let t = (skew_deg.to_radians() * 0.5).tan();
    let c0s = [c0[0] + t * c1[0], c0[1] + t * c1[1], c0[2] + t * c1[2]];
    let c1s = [c1[0] + t * c0[0], c1[1] + t * c0[1], c1[2] + t * c0[2]];

    // 0x18141df60..0x18141e005: roll (radians at this boundary).
    let (sr, cr) = roll_rad.sin_cos();
    let mut r0 = [
        c0s[0] * cr - c1s[0] * sr,
        c0s[1] * cr - c1s[1] * sr,
        c0s[2] * cr - c1s[2] * sr,
    ];
    let mut r1 = [
        c1s[0] * cr + c0s[0] * sr,
        c1s[1] * cr + c0s[1] * sr,
        c1s[2] * cr + c0s[2] * sr,
    ];

    // 0x18141e00b: perspective column divided by c*hypot(w,h), or hypot for c<=0.
    let diag = width.hypot(height);
    let d = if c > 0.0 { diag * c } else { diag };
    if d > 0.0 {
        r0[2] /= d;
        r1[2] /= d;
    }

    KeystoneBasis {
        m00: r0[0],
        m10: r0[1],
        m20: r0[2],
        m01: r1[0],
        m11: r1[1],
        m21: r1[2],
        aspect_axis,
        stretch,
    }
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let dot = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let l1 = v[0].abs() + v[1].abs() + v[2].abs();
    let len = if l1 == 0.0 {
        0.0
    } else if dot <= f64::MAX && l1 >= L1_MIN {
        dot.sqrt()
    } else {
        ((v[0] / l1) * (v[0] / l1) + (v[1] / l1) * (v[1] / l1)
            + (v[2] / l1) * (v[2] / l1))
            .sqrt()
            * l1
    };
    if len <= 0.0 {
        [0.0; 3]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

impl KeystoneBasis {
    /// Forward homography on centre-relative pixels.
    pub fn forward(&self, x: f64, y: f64) -> (f64, f64) {
        let w = self.m20 * x + self.m21 * y + 1.0;
        (
            (self.m00 * x + self.m01 * y) / w,
            (self.m10 * x + self.m11 * y) / w,
        )
    }

    /// 0x18141eba0: canvas -> source, Cramer solve, then aspect / flips.
    pub fn inverse(&self, u: f64, v: f64, flip_x: bool, flip_y: bool) -> (f64, f64) {
        let a1 = self.m00 - u * self.m20;
        let b1 = self.m01 - u * self.m21;
        let a2 = self.m10 - v * self.m20;
        let b2 = self.m11 - v * self.m21;
        let det = a1 * b2 - b1 * a2;
        if det == 0.0 || !det.is_finite() {
            return (u, v);
        }
        let mut x = (b2 * u - b1 * v) / det;
        let mut y = (a1 * v - a2 * u) / det;
        if self.stretch > 1.0 {
            let (a0, a1) = self.aspect_axis;
            let p = y * a0 - x * a1;
            let q = y * a1 + x * a0;
            let s = self.stretch;
            x = q * a0 - (p / s) * a1;
            y = (p / s) * a0 + q * a1;
        }
        if flip_x {
            x = -x;
        }
        if flip_y {
            y = -y;
        }
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: f64 = 6048.0;
    const H: f64 = 4032.0;

    fn near(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn zero_tilt_is_the_identity_matrix() {
        let b = update_geo(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, W, H, 0);
        assert!(near(b.m00, 1.0, 1e-12) && near(b.m11, 1.0, 1e-12));
        assert!(near(b.m01, 0.0, 1e-12) && near(b.m10, 0.0, 1e-12));
        assert_eq!(b.m20, 0.0);
        assert_eq!(b.m21, 0.0);
        assert_eq!(b.stretch, 0.0);
        let (u, v) = b.forward(123.0, -45.0);
        assert!(near(u, 123.0, 1e-9) && near(v, -45.0, 1e-9));
    }

    #[test]
    fn section_1_5_anchor_horizontal_tilt_is_left_short_right_tall() {
        let b = update_geo(20.0, 0.0, 1.0, 0.0, 0.0, 0.0, W, H, 0);
        let (_, yl) = b.forward(-W / 2.0, H / 2.0);
        let (_, yr) = b.forward(W / 2.0, H / 2.0);
        assert!(yr > H / 2.0 * 1.05, "right {yr}");
        assert!(yl < H / 2.0 * 0.95, "left {yl}");
        assert!(b.m21.abs() < 1e-18, "pure horizontal tilt must not tilt the y axis");
    }

    #[test]
    fn section_1_5_anchor_vertical_tilt_is_top_narrow_bottom_wide() {
        let b = update_geo(0.0, 20.0, 1.0, 0.0, 0.0, 0.0, W, H, 0);
        let (xt, _) = b.forward(W / 2.0, -H / 2.0);
        let (xb, _) = b.forward(W / 2.0, H / 2.0);
        assert!(xt < W / 2.0 * 0.97, "top {xt}");
        assert!(xb > W / 2.0 * 1.03, "bottom {xb}");
    }

    #[test]
    fn section_1_5_anchor_skew_is_symmetric_tan_half_angle() {
        let b = update_geo(0.0, 0.0, 1.0, 30.0, 0.0, 0.0, W, H, 0);
        let t = (30f64.to_radians() * 0.5).tan();
        assert!(near(b.m01, t, 1e-12), "m01 {} want {}", b.m01, t);
        assert!(near(b.m10, t, 1e-12), "m10 {} want {}", b.m10, t);
        // Degrees are converted at the boundary: tan(15 deg), not tan(15 rad).
        assert!(near(t, 15.0f64.to_radians().tan(), 1e-15));
        assert!(t < 0.3);
    }

    #[test]
    fn section_1_5_anchor_aspect_axis_and_percent_conversion() {
        let p = update_geo(0.0, 0.0, 1.0, 0.0, 15.0, 0.0, W, H, 0);
        assert_eq!(p.aspect_axis, (1.0, 0.0));
        assert!(near(p.stretch, 1.15, 1e-12), "{}", p.stretch);
        let n = update_geo(0.0, 0.0, 1.0, 0.0, -15.0, 0.0, W, H, 0);
        assert_eq!(n.aspect_axis, (0.0, 1.0));
        assert!(near(n.stretch, 1.15, 1e-12), "{}", n.stretch);
        // Inside the +/-1e-4 fraction dead zone: stretch 0, axis (1,0).
        let dead = update_geo(0.0, 0.0, 1.0, 0.0, 0.005, 0.0, W, H, 0);
        assert_eq!(dead.stretch, 0.0);
        assert_eq!(dead.aspect_axis, (1.0, 0.0));
    }

    #[test]
    fn section_1_6_zero_focal_length_keeps_unnormalized_perspective() {
        // 0x18141ddef rebuilds the raw (-s1, s2, s1*x - s2*y) basis: the c<=0
        // row is -a/hypot(w,h), not a/hypot scaled by 1e-8 and not the c>0
        // formula with the 1/sqrt(1+L^2) cosine term.
        let without = update_geo(20.0, 0.0, 0.0, 0.0, 0.0, 0.0, W, H, 0);
        let want = -0.2 / W.hypot(H);
        assert!(near(without.m20, want, 1e-12), "m20 {} want {}", without.m20, want);
        let flagged = update_geo(20.0, 0.0, 3.0, 0.0, 0.0, 0.0, W, H, FLAG_IGNORE_FOCAL_LENGTH);
        assert!(near(flagged.m20, want, 1e-12), "flag must force the c<=0 branch");
        let with = update_geo(20.0, 0.0, 3.0, 0.0, 0.0, 0.0, W, H, 0);
        assert!(with.m20.abs() > 1e-6);
        // f_len > 0 adds the cosine term, so its magnitude is strictly smaller.
        assert!(with.m20.abs() < without.m20.abs());
    }

    #[test]
    fn section_1_7_hundred_tilt_corner_displacements_match_model() {
        // Report 1.7's quoted TL/TR/BR/BL displacements, recomputed with this
        // basis on a 6048x4032 frame (fractions of frame width/height).
        let b = update_geo(100.0, 0.0, 1.0, 0.0, 0.0, 0.0, W, H, 0);
        let expected = [
            ("TL", -W / 2.0, -H / 2.0, 0.2268, 0.1137),
            ("TR", W / 2.0, -H / 2.0, 0.0009, -0.2084),
            ("BR", W / 2.0, H / 2.0, 0.0009, 0.2084),
            ("BL", -W / 2.0, H / 2.0, 0.2268, -0.1137),
        ];
        for (name, x, y, dx, dy) in expected {
            let (u, v) = b.forward(x, y);
            assert!(near((u - x) / W, dx, 5e-5), "{name} dx {}", (u - x) / W);
            assert!(near((v - y) / H, dy, 5e-5), "{name} dy {}", (v - y) / H);
        }
    }

    #[test]
    fn legacy_035_inset_is_not_the_same_shape() {
        let b = update_geo(20.0, 0.0, 1.0, 0.0, 0.0, 0.0, W, H, 0);
        let (_, tl_y) = b.forward(-W / 2.0, -H / 2.0);
        assert!((tl_y + H / 2.0).abs() > 0.01 * H, "legacy has no TL y move");
    }

    #[test]
    fn inverse_undoes_forward_with_roll_and_skew() {
        let b = update_geo(30.0, -20.0, 1.4, 12.0, 0.0, 0.05, W, H, 0);
        for &(x, y) in &[(0.0, 0.0), (1000.0, -800.0), (-2500.0, 1500.0)] {
            let (u, v) = b.forward(x, y);
            let (x2, y2) = b.inverse(u, v, false, false);
            assert!(near(x, x2, 1e-6) && near(y, y2, 1e-6), "{x},{y} -> {x2},{y2}");
        }
    }

    #[test]
    fn unit_scale_independence_of_width_height_ratio_only() {
        // UpdateGeo divides by c*hypot(w,h); doubling both dimensions halves
        // the perspective entries, i.e. only the aspect ratio enters after
        // converting to frame coordinates.
        let a = update_geo(20.0, 15.0, 1.0, 0.0, 0.0, 0.0, W, H, 0);
        let b = update_geo(20.0, 15.0, 1.0, 0.0, 0.0, 0.0, 2.0 * W, 2.0 * H, 0);
        assert!(near(a.m20, 2.0 * b.m20, 1e-15));
        assert!(near(a.m21, 2.0 * b.m21, 1e-15));
    }
}
