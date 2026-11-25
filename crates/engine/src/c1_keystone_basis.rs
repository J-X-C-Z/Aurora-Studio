//! Exact Capture One keystone basis **and** the outer crop / roll / flip
//! composition, transcribed from `ImgCoreDll.dll` (2026-09 build, ImageBase
//! 0x180000000, SHA-256 ca02b6ac...).
//!
//! This is a copy/extension of `crates/engine/src/c1_keystone_basis.rs`.
//! The original `update_geo` transcribes `CResampler::UpdateGeo` 0x18141da30
//! but labels its `skew` argument in degrees and folds `amount` at the call
//! boundary.  The bytes show that the real caller
//! (`CResampler::GetGeoCfg` 0x18141d0c0 -> 0x18141d770 -> EnsureGeo 0x18141e280)
//! passes `tan(skew_deg_to_rad) * amount/100` as the raw `in+0x20` value and
//! that `UpdateGeo` then applies `tan(0.5 * that)`.  The composition module
//! below models exactly that caller, so `update_geo_raw` takes the raw
//! in-struct values (percentages, amount-scaled, skew already `tan(...)`),
//! while the legacy `update_geo` is kept for compatibility with the landed
//! module and its tests.
//!
//! Layout proven from the managed wrapper `P1.C1.ImageCoreWrapper.dll`:
//! `IC_KeystoneRaw { f32 tiltX; f32 tiltY; f32 skew; f32 fLen; f32 aspect;
//! f32 amount; u32 flags; }` embedded at `settings + 0xfa7c`.
//!
//! Composition proven from the p2p table at 0x18142048c and the handlers:
//!   canvas -> raw (dir 2, IC_Canvas2RawMulti 0x181504eb0):
//!       +origin(geo+0xd8,e0) -> flips(geo+0x78,0x79) -> aspect -> H
//!       -> radial -> +crop center(geo+0x38,0x40) -> crop bounds
//!   raw -> canvas (dir 1, IC_Raw2CanvasMulti 0x181504c20):
//!       crop bounds -> -crop center -> radial -> H^-1 -> aspect^-1 -> flips
//!       -> -origin
//!
//! NOT PROVEN: the lens distortion transfer function (`geo+0x80`) is modelled
//! as identity; the exact canvas rectangle produced by the C1 helpers
//! 0x181421620 / 0x181422730 is approximated by the AABB of the four crop
//! corners.  No Capture One run or real-photo comparison was performed.

/// Percentage-to-fraction constant at 0x182a33a24: `float 0.01`.
const F32_PCT: f32 = 0.009_999_999_8;
/// `double(float(0.01))` at 0x182a37a08, used inside `UpdateGeo`.
const D_PCT_UPDATE: f64 = 0.009_999_999_776_482_582;
/// `deg -> rad`, 0x182a37b18 (skew / parallelize).
const DEG2RAD_INC: f64 = 0.017_453_292_519_943_299;
/// `deg -> rad`, 0x182a37b08 (GetGeoCfg roll).
const DEG2RAD_ROLL: f64 = 0.017_453_292_499_999_998;
/// Normalisation threshold, 0x182a37378.
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
    pub m00: f64,
    pub m10: f64,
    pub m20: f64,
    pub m01: f64,
    pub m11: f64,
    pub m21: f64,
    pub aspect_axis: (f64, f64),
    pub stretch: f64,
}

/// Reproduce `UpdateGeo(ctx, out, in, arg4, arg5)` **exactly**, using the raw
/// in-struct values (this is what `EnsureGeo` passes as `in = geo + 0x48`).
///
/// * `tilt_h` / `tilt_v`: `in+0x10` / `in+0x18`, already `tilt_pct * amount/100`.
/// * `skew_in`: `in+0x20`, i.e. `tan(skew_deg * pi/180) * amount/100` from
///   GetGeoCfg 0x18141d770; UpdateGeo itself takes `tan(0.5 * skew_in)`.
/// * `aspect_frac`: `in+0x28`, already `aspect_pct / 100`.
/// * `roll`: `in+0x08`, radians.
pub fn update_geo_raw(
    c: f64,
    roll: f64,
    tilt_h: f64,
    tilt_v: f64,
    skew_in: f64,
    aspect_frac: f64,
    width: f64,
    height: f64,
) -> KeystoneBasis {
    let a = tilt_h * D_PCT_UPDATE;
    let b = tilt_v * D_PCT_UPDATE;
    let (x, y) = if c > 0.0 { (c * a, c * b) } else { (a, b) };
    let length = x.hypot(y);
    let (s1, s2) = if length > EPS_NORM {
        (x / length, -y / length)
    } else {
        (0.0, 1.0)
    };

    let (stretch, aspect_axis) = if aspect_frac > EPS_ASPECT {
        (1.0 + aspect_frac, (1.0, 0.0))
    } else if aspect_frac < -EPS_ASPECT {
        (1.0 - aspect_frac, (0.0, 1.0))
    } else {
        (0.0, (1.0, 0.0))
    };

    let n = if c > 0.0 {
        normalize3([-s1, s2, s1 * x - s2 * y])
    } else {
        [-s1, s2, s1 * x - s2 * y]
    };

    let theta = -s1.atan2(s2);
    let (sn, cs) = theta.sin_cos();
    let c0 = [n[0] * sn + cs * s2, n[1] * sn + cs * s1, n[2] * sn];
    let c1 = [n[0] * cs - sn * s2, n[1] * cs - sn * s1, n[2] * cs];

    let t = (0.5 * skew_in).tan();
    let c0s = [c0[0] + t * c1[0], c0[1] + t * c1[1], c0[2] + t * c1[2]];
    let c1s = [c1[0] + t * c0[0], c1[1] + t * c0[1], c1[2] + t * c0[2]];

    let (sr, cr) = roll.sin_cos();
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

    let d = if c > 0.0 {
        width.hypot(height) * c
    } else {
        width.hypot(height)
    };
    if d != 0.0 {
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

/// Legacy boundary used by the landed module: `skew_deg` in degrees and
/// `aspect_pct` in percent, `flags & 1` forcing `c = 0`.  It matches the
/// landed `c1_keystone_basis.rs`; **do not use it to compose the full C1
/// pipeline** (see `update_geo_raw`).
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
    let c = if flags & FLAG_IGNORE_FOCAL_LENGTH != 0 { 0.0 } else { f_len };
    update_geo_raw(
        c,
        roll_rad,
        tilt_h,
        tilt_v,
        (skew_deg.to_radians() * 0.5).tan(),
        aspect_pct / 100.0,
        width,
        height,
    )
}

fn normalize3(v: [f64; 3]) -> [f64; 3] {
    let dot = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
    let l1 = v[0].abs() + v[1].abs() + v[2].abs();
    let len = if l1 == 0.0 {
        0.0
    } else if dot <= f64::MAX && l1 >= L1_MIN {
        dot.sqrt()
    } else {
        ((v[0] / l1) * (v[0] / l1)
            + (v[1] / l1) * (v[1] / l1)
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

/// The `IC_KeystoneRaw` block plus the resampler's crop/roll/orientation state.
#[derive(Clone, Copy, Debug)]
pub struct C1Settings {
    /// `settings+0xfa58` / `+0xfa5c` (crop pixel size).
    pub crop_w: i32,
    pub crop_h: i32,
    /// `resampler+0x00 / +0x04 / +0x08 / +0x0c` = (x0, x1, y0, y1).
    /// `None` means the sentinel 4-int rect, i.e. full crop.
    pub crop_rect: Option<(i32, i32, i32, i32)>,
    /// `IC_KeystoneRaw`, percentages except flags.
    pub tilt_x: f32,
    pub tilt_y: f32,
    pub skew: f32,
    pub f_len: f32,
    pub aspect: f32,
    pub amount: f32,
    pub flags: u32,
    /// `settings+0xfa68`, degrees.
    pub roll_deg: f32,
    /// `settings+0xfa6c`: 0 none, 1 flip X, 2 flip Y.
    pub orientation: i32,
    /// `settings+0xfa98`, stored in `geo+0x120`.
    pub extra_flags: u32,
    /// `settings+0x10bb8 != 0` and `+0x10bbc/+0x10bc0`.
    pub optic_center: Option<(f32, f32)>,
    /// `settings+0x13980 * settings+0x13984` (float product), `geo+0x28`.
    pub scale: f32,
}

impl Default for C1Settings {
    fn default() -> Self {
        C1Settings {
            crop_w: 6048,
            crop_h: 4032,
            crop_rect: None,
            tilt_x: 0.0,
            tilt_y: 0.0,
            skew: 0.0,
            f_len: 0.0,
            aspect: 0.0,
            amount: 100.0,
            flags: 0,
            roll_deg: 0.0,
            orientation: 0,
            extra_flags: 0,
            optic_center: None,
            scale: 1.0,
        }
    }
}

/// Fully built `geo` (0x1a0 bytes) + the six-double basis.
#[derive(Clone, Copy, Debug)]
pub struct Geo {
    pub c: f64,
    pub amount_frac: f64,
    /// `geo+0x58 = amount/100 * tiltX`.
    pub eff_tilt_x: f64,
    /// `geo+0x60 = amount/100 * tiltY`.
    pub eff_tilt_y: f64,
    /// `geo+0x68 = tan(clamp(skew)) * amount/100`.
    pub skew_in: f64,
    /// `geo+0x70 = aspect/100`.
    pub aspect_frac: f64,
    /// `geo+0x50`, radians.
    pub roll: f64,
    pub crop: (f64, f64, f64, f64),
    /// `geo+0x38 / +0x40`.
    pub center: (f64, f64),
    /// `geo+0x78 / +0x79`.
    pub flip_x: bool,
    pub flip_y: bool,
    /// `geo+0x28`.
    pub scale: f64,
    /// `geo+0x120`.
    pub extra_flags: u32,
    pub basis: KeystoneBasis,
    /// `geo+0xd8 / +0xe0` estimate (canvas origin in centred raw coords).
    pub origin: (f64, f64),
    /// `geo+0x110 / +0x118` estimate (canvas size, as returned by IC_CanvasSize).
    pub canvas_w: f64,
    pub canvas_h: f64,
}

impl Geo {
    /// `GetGeoCfg` 0x18141d0c0 + `0x18141d770` + `EnsureGeo` 0x18141e280.
    pub fn build(s: &C1Settings) -> Geo {
        let c = if s.f_len > 0.0 { s.f_len as f64 } else { 1.0 };
        let c = if s.flags & FLAG_IGNORE_FOCAL_LENGTH != 0 { 0.0 } else { c };

        let amount_frac = (s.amount * F32_PCT) as f64; // float mul @0x18141d7c1
        let eff_tilt_x = (s.tilt_x * amount_frac as f32) as f64; // 0x18141d7dd
        let eff_tilt_y = (s.tilt_y * amount_frac as f32) as f64; // 0x18141d7e9

        let skew_deg = (s.skew as f64).clamp(-45.0, 45.0); // 0x182a3e200/0x182a3d788
        let skew_in = skew_deg.mul_add(DEG2RAD_INC, 0.0).tan() * amount_frac; // 0x18141d81f..846
        let aspect_frac = (s.aspect * F32_PCT) as f64; // 0x18141d84b..85b
        let roll = (s.roll_deg as f64) * DEG2RAD_ROLL; // 0x18141d278

        let flip_x = s.orientation == 1;
        let flip_y = s.orientation == 2;

        let crop = match s.crop_rect {
            Some((x0, y0, x1, y1)) if x1 > x0 && y1 > y0 => {
                (x0 as f64, y0 as f64, x1 as f64, y1 as f64)
            }
            _ => (0.0, 0.0, s.crop_w as f64, s.crop_h as f64),
        };
        let center = match s.optic_center {
            Some((cx, cy)) => (cx as f64, cy as f64),
            None => (
                (s.crop_w as f32 * 0.5) as f64,
                (s.crop_h as f32 * 0.5) as f64,
            ),
        };
        let scale = (s.scale * 1.0) as f64; // already float product at the settings layer

        let basis = update_geo_raw(
            c,
            roll,
            eff_tilt_x,
            eff_tilt_y,
            skew_in,
            aspect_frac,
            s.crop_w as f64,
            s.crop_h as f64,
        );

        let mut g = Geo {
            c,
            amount_frac,
            eff_tilt_x,
            eff_tilt_y,
            skew_in,
            aspect_frac,
            roll,
            crop,
            center,
            flip_x,
            flip_y,
            scale,
            extra_flags: s.extra_flags,
            basis,
            origin: (0.0, 0.0),
            canvas_w: 0.0,
            canvas_h: 0.0,
        };
        let (ox, oy, cw, ch) = canvas_rect_estimate(&g);
        g.origin = (ox, oy);
        g.canvas_w = cw;
        g.canvas_h = ch;
        g
    }

    /// Homography forward (`p2p` dir 9, 0x18141e9e0) on pre-origin coords.
    fn homography(&self, x: f64, y: f64) -> (f64, f64) {
        let m = &self.basis;
        let den = m.m20 * x + m.m21 * y + 1.0;
        (
            (m.m00 * x + m.m01 * y) / den,
            (m.m10 * x + m.m11 * y) / den,
        )
    }

    /// Homography inverse (`0x18141eba0`), Cramer solve.
    fn homography_inv(&self, u: f64, v: f64) -> (f64, f64) {
        let m = &self.basis;
        let a1 = m.m00 - u * m.m20;
        let b1 = m.m01 - u * m.m21;
        let a2 = m.m10 - v * m.m20;
        let b2 = m.m11 - v * m.m21;
        let det = a1 * b2 - b1 * a2;
        if det == 0.0 || !det.is_finite() {
            return (u, v);
        }
        ((b2 * u - b1 * v) / det, (a1 * v - a2 * u) / det)
    }

    /// Aspect forward (dir 9): `X' = A*ax - B*s*ay`, `Y' = B*s*ax + A*ay`.
    fn aspect_fwd(&self, x: f64, y: f64) -> (f64, f64) {
        if self.basis.stretch <= 1.0 {
            return (x, y);
        }
        let (ax, ay) = self.basis.aspect_axis;
        let s = self.basis.stretch;
        let a = ax * x + ay * y;
        let b = ax * y - ay * x;
        (a * ax - (b * s) * ay, (b * s) * ax + a * ay)
    }

    /// Aspect inverse (0x18141eba0).
    fn aspect_inv(&self, x: f64, y: f64) -> (f64, f64) {
        if self.basis.stretch <= 1.0 {
            return (x, y);
        }
        let (ax, ay) = self.basis.aspect_axis;
        let s = self.basis.stretch;
        let p = y * ax - x * ay;
        let q = y * ay + x * ax;
        (q * ax - (p / s) * ay, (p / s) * ax + q * ay)
    }

    /// Canvas -> raw source pixel (p2p dir 2 = 0x18141e9e0 + 0x18141e8a0).
    /// Returns `(raw_x, raw_y, outside_crop)`.
    pub fn canvas_to_raw(&self, cx: f64, cy: f64) -> (f64, f64, bool) {
        let mut x = cx + self.origin.0;
        let mut y = cy + self.origin.1;
        if self.flip_x {
            x = -x;
        }
        if self.flip_y {
            y = -y;
        }
        let (x, y) = self.aspect_fwd(x, y);
        let (u, v) = self.homography(x, y);
        // Radial inverse is identity here (lens distortion NOT PROVEN).
        let raw = (u + self.center.0, v + self.center.1);
        let outside = !(self.crop.0 <= raw.0
            && raw.0 <= self.crop.2
            && self.crop.1 <= raw.1
            && raw.1 <= self.crop.3);
        (raw.0, raw.1, outside)
    }

    /// `p2p` dir 4/5: the input is a canvas coordinate in a preview scaled by
    /// `geo+0x28`; `geo+0x108` (= 1/geo+0x28) is multiplied first (0x18141ece0).
    pub fn canvas_to_raw_scaled(&self, cx: f64, cy: f64) -> (f64, f64, bool) {
        if self.scale == 0.0 {
            return self.canvas_to_raw(cx, cy);
        }
        self.canvas_to_raw(cx / self.scale, cy / self.scale)
    }

    /// Raw source pixel -> canvas (p2p dir 1 = 0x18141e750 + 0x18141eba0).
    /// Returns `(canvas_x, canvas_y, outside_crop)`.
    pub fn raw_to_canvas(&self, rx: f64, ry: f64) -> (f64, f64, bool) {
        let outside = !(self.crop.0 <= rx
            && rx <= self.crop.2
            && self.crop.1 <= ry
            && ry <= self.crop.3);
        let dx = rx - self.center.0;
        let dy = ry - self.center.1;
        // Radial forward is identity here (lens distortion NOT PROVEN).
        let (mut x, mut y) = self.homography_inv(dx, dy);
        let (a, b) = self.aspect_inv(x, y);
        x = a;
        y = b;
        if self.flip_x {
            x = -x;
        }
        if self.flip_y {
            y = -y;
        }
        (x - self.origin.0, y - self.origin.1, outside)
    }
}

/// AABB of the four crop corners in the raw->canvas chain; this mirrors the
/// shape of the C1 helpers (centre + size) but not their exact maths.
fn canvas_rect_estimate(g: &Geo) -> (f64, f64, f64, f64) {
    let mut minx = f64::INFINITY;
    let mut miny = f64::INFINITY;
    let mut maxx = f64::NEG_INFINITY;
    let mut maxy = f64::NEG_INFINITY;
    for (rx, ry) in [
        (g.crop.0, g.crop.1),
        (g.crop.2, g.crop.1),
        (g.crop.2, g.crop.3),
        (g.crop.0, g.crop.3),
    ] {
        let dx = rx - g.center.0;
        let dy = ry - g.center.1;
        let (mut x, mut y) = g.homography_inv(dx, dy);
        let (a, b) = g.aspect_inv(x, y);
        x = a;
        y = b;
        if g.flip_x {
            x = -x;
        }
        if g.flip_y {
            y = -y;
        }
        minx = minx.min(x);
        miny = miny.min(y);
        maxx = maxx.max(x);
        maxy = maxy.max(y);
    }
    (minx, miny, maxx - minx, maxy - miny)
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: f64 = 6048.0;
    const H: f64 = 4032.0;

    fn near(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() <= eps
    }

    fn settings(tilt_x: f32, tilt_y: f32, f_len: f32, amount: f32) -> C1Settings {
        C1Settings {
            tilt_x,
            tilt_y,
            f_len,
            amount,
            ..C1Settings::default()
        }
    }

    #[test]
    fn zero_tilt_is_identity() {
        let g = Geo::build(&C1Settings::default());
        assert!(near(g.basis.m00, 1.0, 1e-12));
        assert!(near(g.basis.m11, 1.0, 1e-12));
        assert_eq!(g.basis.m20, 0.0);
        assert_eq!(g.basis.m21, 0.0);
        assert_eq!(g.canvas_w, W);
        assert_eq!(g.canvas_h, H);
        let (rx, ry, oob) = g.canvas_to_raw(0.0, 0.0);
        assert!(!oob && near(rx, 0.0, 1e-12) && near(ry, 0.0, 1e-12));
        let (rx, ry, oob) = g.canvas_to_raw(W, H);
        assert!(!oob && near(rx, W, 1e-9) && near(ry, H, 1e-9));
    }

    #[test]
    fn amount_multiplies_the_effective_tilt_in_getgeocfg() {
        // Not the stored tilt: geo+0x58/0x60 are amount/100 * tilt.
        let full = Geo::build(&settings(20.0, 0.0, 1.0, 100.0));
        let half = Geo::build(&settings(20.0, 0.0, 1.0, 50.0));
        assert!(near(full.eff_tilt_x, 20.0, 1e-6));
        assert!(near(half.eff_tilt_x, 10.0, 1e-6));
        assert!(near(half.basis.m20, 0.5 * full.basis.m20, 1e-6));
        let zero = Geo::build(&settings(20.0, 0.0, 1.0, 0.0));
        assert_eq!(zero.basis.m20, 0.0);
        assert_eq!(zero.basis.m00, 1.0);
    }

    #[test]
    fn amount_zero_is_identity_only_without_roll_skew_aspect() {
        let g = Geo::build(&settings(20.0, 0.0, 1.0, 0.0));
        assert_eq!(g.basis.m00, 1.0);
        assert_eq!(g.basis.m01, 0.0);
        assert_eq!(g.basis.m10, 0.0);
        assert_eq!(g.basis.m11, 1.0);
        assert_eq!(g.basis.m20, 0.0);
    }

    #[test]
    fn f_len_le_zero_means_c_one_not_keystone_off() {
        let zero = Geo::build(&settings(20.0, 0.0, 0.0, 100.0));
        let one = Geo::build(&settings(20.0, 0.0, 1.0, 100.0));
        assert_eq!(zero.c, 1.0);
        assert_eq!(zero.basis.m20, one.basis.m20);
        assert!(zero.basis.m20.abs() > 1e-9); // perspective is still present
        // Flag bit 0 is the *only* way to reach the c<=0 branch.
        let mut s = settings(20.0, 0.0, 3.0, 100.0);
        s.flags = FLAG_IGNORE_FOCAL_LENGTH;
        let ignored = Geo::build(&s);
        assert_eq!(ignored.c, 0.0);
        assert!(near(ignored.basis.m20, -0.2 / W.hypot(H), 1e-12));
        assert!(ignored.basis.m20.abs() > zero.basis.m20.abs());
    }

    #[test]
    fn roll_maps_canvas_to_source_as_r_minus_roll() {
        let g = Geo::build(&C1Settings {
            roll_deg: 10.0,
            ..C1Settings::default()
        });
        let c10 = 10f64.to_radians();
        // H(canvas) = R(-roll) * canvas (0x18141dff..0x18141e09), so a source
        // feature p lands on canvas R(+roll)*p: +roll rotates the visible image
        // clockwise (canvas y points down).  The UI sign convention itself was
        // not observed in a running Capture One.
        assert!(near(g.basis.m00, c10.cos(), 1e-9));
        assert!(near(g.basis.m10, -c10.sin(), 1e-9));
        assert!(near(g.basis.m01, c10.sin(), 1e-9));
        assert!(near(g.basis.m11, c10.cos(), 1e-9));
        assert!(g.canvas_w > W && g.canvas_h > H);
    }

    #[test]
    fn orientation_sets_flip_bytes_and_round_trips() {
        let mut sx = settings(20.0, 0.0, 1.0, 100.0);
        sx.orientation = 1;
        let fx = Geo::build(&sx);
        assert!(fx.flip_x && !fx.flip_y);
        let mut sy = settings(0.0, 20.0, 1.0, 100.0);
        sy.orientation = 2;
        let fy = Geo::build(&sy);
        assert!(!fy.flip_x && fy.flip_y);
        for g in [&fx, &fy] {
            for (cx, cy) in [(0.0, 0.0), (100.0, 50.0), (4000.0, 2000.0)] {
                let (rx, ry, _) = g.canvas_to_raw(cx, cy);
                let (bx, by, _) = g.raw_to_canvas(rx, ry);
                assert!(near(bx, cx, 1e-9) && near(by, cy, 1e-9));
            }
        }
    }

    #[test]
    fn aspect_positive_stretches_y_and_shrinks_canvas_height() {
        let g = Geo::build(&C1Settings {
            aspect: 15.0,
            ..C1Settings::default()
        });
        assert_eq!(g.basis.aspect_axis, (1.0, 0.0));
        assert!(near(g.basis.stretch, 1.15, 1e-6));
        assert!(near(g.canvas_h, H / 1.15, 1e-3));
        let (rx, ry, _) = g.canvas_to_raw(0.0, 0.0);
        assert!(near(rx, 0.0, 1e-9) && near(ry, 0.0, 1e-9));
        let (rx, ry, _) = g.canvas_to_raw(g.canvas_w, g.canvas_h);
        assert!(near(rx, W, 1e-6) && near(ry, H, 1e-6));
        let n = Geo::build(&C1Settings {
            aspect: -15.0,
            ..C1Settings::default()
        });
        assert_eq!(n.basis.aspect_axis, (0.0, 1.0));
        assert!(near(n.canvas_w, W / 1.15, 1e-3));
    }

    #[test]
    fn skew_is_tan_of_half_the_amount_scaled_tan() {
        // GetGeoCfg: g68 = tan(30 deg) * 1.0 = 0.577350; UpdateGeo: tan(g68/2).
        let g = Geo::build(&C1Settings {
            skew: 30.0,
            ..C1Settings::default()
        });
        assert!(near(g.skew_in, 30f64.to_radians().tan(), 1e-12));
        let want = (0.5 * 30f64.to_radians().tan()).tan();
        assert!(near(g.basis.m01, want, 1e-12), "{} vs {}", g.basis.m01, want);
        assert!(near(g.basis.m10, want, 1e-12));
        assert!((want - 15f64.to_radians().tan()).abs() > 0.02);
        // amount scales the skew input too.
        let mut s = C1Settings {
            skew: 30.0,
            ..C1Settings::default()
        };
        s.amount = 50.0;
        let h = Geo::build(&s);
        assert!(near(h.skew_in, 0.5 * 30f64.to_radians().tan(), 1e-12));
    }

    #[test]
    fn crop_rect_center_and_bounds() {
        let g = Geo::build(&C1Settings {
            crop_w: 1000,
            crop_h: 700,
            crop_rect: Some((100, 50, 1000, 700)),
            tilt_x: 20.0,
            f_len: 1.0,
            ..C1Settings::default()
        });
        assert_eq!(g.center, (500.0, 350.0));
        let (rx, ry, oob) = g.canvas_to_raw(0.0, 0.0);
        assert!(!oob && near(rx, 100.0, 1e-9) && near(ry, 50.0, 1e-9));
        // Off-crop points report outside.
        let (_, _, oob) = g.canvas_to_raw(g.canvas_w, 0.0);
        assert!(oob);
    }

    #[test]
    fn scaled_canvas_premultiplies_by_one_over_geo_0x108() {
        let g = Geo::build(&C1Settings {
            scale: 2.0,
            ..settings(20.0, 0.0, 1.0, 100.0)
        });
        assert!(near(g.scale, 2.0, 1e-6));
        for (cx, cy) in [(200.0, 100.0), (1000.0, 600.0)] {
            let (ax, ay, _) = g.canvas_to_raw_scaled(cx, cy);
            let (bx, by, _) = g.canvas_to_raw(cx / 2.0, cy / 2.0);
            assert!(near(ax, bx, 1e-9) && near(ay, by, 1e-9));
        }
    }

    #[test]
    fn full_round_trip_all_features() {
        let g = Geo::build(&C1Settings {
            crop_w: 5000,
            crop_h: 3333,
            crop_rect: Some((120, 80, 5000, 3333)),
            tilt_x: 25.0,
            tilt_y: -15.0,
            skew: 12.0,
            aspect: 8.0,
            amount: 100.0,
            roll_deg: 7.0,
            f_len: 1.4,
            ..C1Settings::default()
        });
        for (cx, cy) in [
            (0.0, 0.0),
            (100.0, 50.0),
            (g.canvas_w, g.canvas_h),
            (g.canvas_w / 2.0, g.canvas_h / 2.0),
        ] {
            let (rx, ry, _) = g.canvas_to_raw(cx, cy);
            let (bx, by, _) = g.raw_to_canvas(rx, ry);
            assert!(near(bx, cx, 1e-7) && near(by, cy, 1e-7), "{}", format!("{cx},{cy} -> {bx},{by}"));
        }
    }

    #[test]
    fn crosscheck_dump() {
        // Values consumed by ref/crosscheck.py (stdout with --nocapture).
        let cases: [(&str, C1Settings); 6] = [
            ("identity", C1Settings::default()),
            ("tilt20", settings(20.0, 0.0, 1.0, 100.0)),
            ("amount50", settings(20.0, 0.0, 1.0, 50.0)),
            ("c0", C1Settings { flags: FLAG_IGNORE_FOCAL_LENGTH, ..settings(20.0, 0.0, 0.0, 100.0) }),
            ("roll10", C1Settings { roll_deg: 10.0, ..C1Settings::default() }),
            (
                "all",
                C1Settings {
                    crop_w: 5000,
                    crop_h: 3333,
                    crop_rect: Some((120, 80, 5000, 3333)),
                    tilt_x: 25.0,
                    tilt_y: -15.0,
                    skew: 12.0,
                    aspect: 8.0,
                    amount: 100.0,
                    roll_deg: 7.0,
                    f_len: 1.4,
                    ..C1Settings::default()
                },
            ),
        ];
        for (name, s) in cases {
            let g = Geo::build(&s);
            let (rx, ry, _) = g.canvas_to_raw(0.0, 0.0);
            println!(
                "XCHECK,{name},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e},{:.17e}",
                g.basis.m00, g.basis.m01, g.basis.m10, g.basis.m11,
                g.basis.m20, g.basis.m21, g.origin.0, g.origin.1,
                g.canvas_w, g.canvas_h, rx, ry
            );
        }
    }
}
