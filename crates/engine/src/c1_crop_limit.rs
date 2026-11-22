#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Crop {
    pub center: [f64; 2],
    pub size: [f64; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Resize {
    /// `resize=0`: one uniform scale around the anchor.
    Uniform,
    /// `resize=3`: one scale per side, so the rectangle may move.
    KeepArea1,
}

#[derive(Clone, Copy, Debug)]
pub enum Bounds {
    Mode1 { image: [u32; 2] },
    Mode6 { min: [f64; 2], max: [f64; 2] },
    Mode8 { min: [f64; 2], max: [f64; 2] },
    Mode10 { size: [f64; 2] },
}

impl Bounds {
    fn mode(self) -> u8 {
        match self {
            Self::Mode1 { .. } => 2,
            Self::Mode6 { .. } => 7,
            Self::Mode8 { .. } => 9,
            Self::Mode10 { .. } => 10,
        }
    }

    fn edges(self) -> ([f64; 2], [f64; 2]) {
        match self {
            Self::Mode1 { image } => ([0.0; 2], image.map(f64::from)),
            Self::Mode6 { min, max } | Self::Mode8 { min, max } => (min, max),
            Self::Mode10 { size } => ([0.0; 2], size),
        }
    }

    fn distance(self, [x, y]: [f64; 2]) -> f64 {
        let ([left, top], [right, bottom]) = self.edges();
        let dx = (x - left).min(right - x);
        let dy = (y - top).min(bottom - y);
        if dx < 0.0 || dy < 0.0 {
            -dx.min(0.0).hypot(dy.min(0.0))
        } else {
            dx.min(dy)
        }
    }
}

/// `p2p(mode=6)` / `0x141E6A0`: subtract the optical centre, then apply the
/// radial curve. The curve returns the corrected radius and is the identity
/// below radius 1.
pub fn mode6(
    point: [f64; 2],
    optical_center: [f64; 2],
    radial: impl Fn(f64) -> f64,
) -> [f64; 2] {
    let [u, v] = [point[0] - optical_center[0], point[1] - optical_center[1]];
    let r = u.hypot(v);
    if r > 1.0 {
        let scale = radial(r) / r;
        [u * scale, v * scale]
    } else {
        [u, v]
    }
}

/// `p2p(mode=7)` / `0x141E7F0`: the inverse radial curve, then add the optical
/// centre back. `resize=3` trials use this direction against source bounds.
pub fn mode7(
    point: [f64; 2],
    optical_center: [f64; 2],
    radial: impl Fn(f64) -> f64,
) -> [f64; 2] {
    let r2 = point[0] * point[0] + point[1] * point[1];
    let r = r2.sqrt();
    let scale = if r2 > 1.0 { radial(r) / r } else { 1.0 };
    [
        point[0] * scale + optical_center[0],
        point[1] * scale + optical_center[1],
    ]
}

/// `p2p(mode=8)` / `0x141E930`. `matrix` is the 13 doubles at context
/// `+0xa8`: `[a b c d e f tx ty stretch_a stretch_b stretch_scale ..]`.
/// The output is `((a*x + d*y) / d, (b*x + e*y) / d)` with
/// `d = c*x + f*y + 1`. The optional stretch branch is not used by the
/// catalog rows in these tests.
pub fn mode8(point: [f64; 2], matrix: &[f64; 13], flip: [bool; 2]) -> [f64; 2] {
    let [a, b, c, d, e, f, tx, ty, ..] = *matrix;
    let mut x = point[0] + tx;
    let mut y = point[1] + ty;
    if flip[0] {
        x = -x;
    }
    if flip[1] {
        y = -y;
    }
    let divisor = c * x + f * y + 1.0;
    [(a * x + d * y) / divisor, (b * x + e * y) / divisor]
}

/// Limit a canvas-space crop around its existing center. The caller supplies
/// the verified single-point and four-point transforms from the same job.
pub fn limit_center(
    crop: Crop,
    bounds: Bounds,
    resize: Resize,
    map_center: impl FnOnce(u8, [f64; 2]) -> [f64; 2],
    map_trial: impl Fn(u8, [[f64; 2]; 4]) -> [[f64; 2]; 4],
) -> Crop {
    let [cx, cy] = crop.center;
    let [w, h] = crop.size;
    if bounds.distance(map_center(bounds.mode(), crop.center)) <= 0.0 {
        return Crop { size: [0.0; 2], ..crop };
    }

    let left = cx - w / 2.0;
    let right = cx + w / 2.0;
    let top = cy - h / 2.0;
    let bottom = cy + h / 2.0;
    let sides = [
        [[left, top], [right, top]],
        [[left, bottom], [right, bottom]],
        [[left, top], [left, bottom]],
        [[right, top], [right, bottom]],
    ];
    let mut scale = [1.0; 4];
    for (index, [a, b]) in sides.into_iter().enumerate() {
        let points = [crop.center, a, b, crop.center];
        let span = (cx.max(a[0]).max(b[0]) - cx.min(a[0]).min(b[0]))
            .max(cy.max(a[1]).max(b[1]) - cy.min(a[1]).min(b[1]));
        scale[index] = search_side(span, |s| {
            let trial = points.map(|[x, y]| [cx + s * (x - cx), cy + s * (y - cy)]);
            map_trial(bounds.mode(), trial)
                .into_iter()
                .map(|point| bounds.distance(point))
                .fold(f64::INFINITY, f64::min)
        });
    }

    match resize {
        Resize::Uniform => {
            let s = scale.into_iter().fold(1.0, f64::min);
            if s >= 1.0 {
                crop
            } else {
                Crop { size: [w * s, h * s], ..crop }
            }
        }
        Resize::KeepArea1 => {
            let [s_top, s_bottom, s_left, s_right] = scale;
            if scale.iter().all(|s| *s >= 1.0) {
                return crop;
            }
            let left = cx - s_left * w / 2.0;
            let right = cx + s_right * w / 2.0;
            let top = cy - s_top * h / 2.0;
            let bottom = cy + s_bottom * h / 2.0;
            Crop {
                center: [(left + right) / 2.0, (top + bottom) / 2.0],
                size: [right - left, bottom - top],
            }
        }
    }
}

pub fn limit_center_keep_aspect(
    crop: Crop,
    bounds: Bounds,
    map_center: impl FnOnce(u8, [f64; 2]) -> [f64; 2],
    map_trial: impl Fn(u8, [[f64; 2]; 4]) -> [[f64; 2]; 4],
) -> Crop {
    limit_center(crop, bounds, Resize::Uniform, map_center, map_trial)
}

pub fn limit_center_keep_area1(
    crop: Crop,
    bounds: Bounds,
    map_center: impl FnOnce(u8, [f64; 2]) -> [f64; 2],
    map_trial: impl Fn(u8, [[f64; 2]; 4]) -> [[f64; 2]; 4],
) -> Crop {
    limit_center(crop, bounds, Resize::KeepArea1, map_center, map_trial)
}

/// `CalculateGeo` / `0x141E1D0`: compute the mode-8 boundary rectangle from
/// the 5x source rect, the source bounds, the optical centre and the radial
/// curve. The result is the warped min/max box read back by mode 8.
pub fn mode8_boundary(
    crop: Crop,
    source: ([f64; 2], [f64; 2]),
    optical_center: [f64; 2],
    radial: impl Fn(f64) -> f64,
) -> Crop {
    limit_center(
        crop,
        Bounds::Mode6 { min: source.0, max: source.1 },
        Resize::KeepArea1,
        |_, point| mode7(point, optical_center, &radial),
        |_, points| points.map(|point| mode7(point, optical_center, &radial)),
    )
}

/// One C1 lens distortion curve: the `CSpline` the resampler builds at
/// `0x14204E0` from `IC_LensGeometricAberration`.
///
/// `v` is the source radius and `x` the corrected radius, both in raw pixels;
/// `m` is `dx/dv` at that node.  The pairs are the ones the native builder
/// writes, so the curve is the exact radial mapping C1 evaluates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RadialCurve {
    pub v: Vec<f64>,
    pub x: Vec<f64>,
    pub m: Vec<f64>,
}

impl RadialCurve {
    /// `0x14204E0`: build the control points from a `ZLENSDISTORTION` row.
    ///
    /// `points` are the row's `radius, delta` pairs and `amount` its percent.
    /// `pitch` is `IC_SensorGeometry.rawPixelWidth/rawPixelHeight`, in
    /// micrometres per pixel.  The radius normaliser is
    /// `1 / sqrt((pitch.x² + pitch.y²) / 2)`, which turns the profile's
    /// micrometre radii into raw pixels; doing the arithmetic in `f32` in the
    /// native order reproduces the native table bit for bit.
    ///
    /// The native guards are kept: no curve for an empty/one-point table, a
    /// disabled curve, an amount that rounds below one percent, or radii that
    /// are not strictly increasing.
    pub fn from_profile(points: &[(f32, f32)], amount: f32, pitch: [f32; 2]) -> Self {
        let enabled = points.len() >= 2
            && amount * 0.01f32 != 0.0
            && (amount * 0.01f32).abs() >= 0.01
            && points.windows(2).all(|pair| pair[0].0 < pair[1].0);
        if !enabled {
            return Self::default();
        }
        let mut radius = pitch[1] * pitch[1];
        radius += pitch[0] * pitch[0];
        radius *= 0.5f32;
        let k = 1.0f32 / radius.sqrt();
        let amount = amount * 0.01f32;
        let mut curve = Self {
            v: Vec::with_capacity(points.len()),
            x: Vec::with_capacity(points.len()),
            m: Vec::with_capacity(points.len()),
        };
        for &(node, delta) in points {
            curve.v.push(f64::from(node * k));
            curve.x.push(f64::from((amount * delta + node) * k));
        }
        curve.m = curve.derivatives();
        curve
    }

    /// A curve assembled from the native control points and derivative column.
    pub fn from_native_table(v: Vec<f64>, x: Vec<f64>, m: Vec<f64>) -> Self {
        Self { v, x, m }
    }

    /// `0x14278D0`: the interior rows of C1's `CSpline::UpdateDerivatives`,
    /// with the first node clamped to the native start slope of 1 and a natural
    /// end row.  The native end row is not reproduced bit for bit yet, so the
    /// outermost derivative can drift by a couple of percent; callers that need
    /// exactness use [`Self::from_native_table`].
    fn derivatives(&self) -> Vec<f64> {
        let n = self.v.len();
        let mut m = vec![0.0; n];
        m[0] = 1.0;
        if n < 3 {
            return m;
        }
        let a: Vec<f64> = (1..n).map(|i| 1.0 / (self.v[i] - self.v[i - 1])).collect();
        let b: Vec<f64> = (1..n).map(|i| 1.0 / (self.v[i] - self.v[i - 1])).collect();
        // Tridiagonal system over nodes 1 ..= n-1.
        let mut lower = vec![0.0; n];
        let mut diag = vec![0.0; n];
        let mut upper = vec![0.0; n];
        let mut rhs = vec![0.0; n];
        for i in 1..n - 1 {
            lower[i] = a[i - 1];
            upper[i] = b[i];
            diag[i] = 2.0 * (a[i - 1] + b[i]);
            rhs[i] = 3.0
                * ((self.x[i + 1] - self.x[i]) * b[i] * b[i]
                    + (self.x[i] - self.x[i - 1]) * a[i - 1] * a[i - 1]);
        }
        rhs[1] -= a[0] * m[0];
        diag[n - 1] = 2.0 / (self.v[n - 1] - self.v[n - 2]);
        lower[n - 1] = 1.0 / (self.v[n - 1] - self.v[n - 2]);
        rhs[n - 1] = 3.0 * (self.x[n - 1] - self.x[n - 2])
            / ((self.v[n - 1] - self.v[n - 2]) * (self.v[n - 1] - self.v[n - 2]));
        let mut c = vec![0.0; n];
        let mut d = vec![0.0; n];
        c[1] = upper[1] / diag[1];
        d[1] = rhs[1] / diag[1];
        for i in 2..n {
            let den = diag[i] - lower[i] * c[i - 1];
            c[i] = upper[i] / den;
            d[i] = (rhs[i] - lower[i] * d[i - 1]) / den;
        }
        m[n - 1] = d[n - 1];
        for i in (1..n - 1).rev() {
            m[i] = d[i] - c[i] * m[i + 1];
        }
        m
    }

    fn segment(values: &[f64], key: f64) -> usize {
        let mut lo = 0usize;
        let mut hi = values.len() - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if values[mid] <= key {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    /// `0x1426F40`: corrected radius for a source radius, the mode 7 direction.
    pub fn forward(&self, source: f64) -> f64 {
        if self.v.len() < 2 {
            return source;
        }
        let i = Self::segment(&self.v, source);
        let (v0, x0, m0) = (self.v[i], self.x[i], self.m[i]);
        let (v1, x1, m1) = (self.v[i + 1], self.x[i + 1], self.m[i + 1]);
        let h = v1 - v0;
        if h == 0.0 {
            return ((m1 + m0) * source + v1 + x0) * 0.5;
        }
        let t = (source - v0) / h;
        ((t - 1.0) * m0 + m1 * t) * t * (t - 1.0) * h
            + t * t * (x1 - x0) * (3.0 - (t + t))
            + x0
    }

    /// `0x14296C0`: source radius for a corrected radius (mode 6).  The native
    /// routine inverts the same cubic with Cardano's rule; the bisection here
    /// agrees with it to within two `f64` ulp.
    pub fn inverse(&self, corrected: f64) -> f64 {
        if self.v.len() < 2 {
            return corrected;
        }
        let i = Self::segment(&self.x, corrected);
        let (mut lo, mut hi) = (self.v[i], self.v[i + 1]);
        for _ in 0..200 {
            let mid = (lo + hi) * 0.5;
            if self.forward(mid) < corrected {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (lo + hi) * 0.5
    }
}

/// `p2p(mode=6)` with a real lens curve: absolute coordinates to the radial
/// frame, evaluated through [`RadialCurve::inverse`].
pub fn mode6_curve(point: [f64; 2], optical_center: [f64; 2], curve: &RadialCurve) -> [f64; 2] {
    mode6(point, optical_center, |r| curve.inverse(r))
}

/// `p2p(mode=7)` with a real lens curve: the radial frame back to absolute
/// coordinates, evaluated through [`RadialCurve::forward`].
pub fn mode7_curve(point: [f64; 2], optical_center: [f64; 2], curve: &RadialCurve) -> [f64; 2] {
    mode7(point, optical_center, |r| curve.forward(r))
}

fn search_side(span: f64, mut distance: impl FnMut(f64) -> f64) -> f64 {
    if span < 0.01 {
        return 1.0;
    }
    let (mut lo, mut d_lo, mut hi, mut d_hi) = (0.0, -1.0, 1.0, -1.0);
    let mut s = 1.0;
    for _ in 0..51 {
        let d = distance(s);
        if d < 0.0 {
            (hi, d_hi) = (s, d);
        } else {
            (lo, d_lo) = (s, d);
        }
        let (a, b) = (d_hi.abs(), d_lo.abs());
        s = if b <= 5.0 * a && a <= 5.0 * b && d_lo >= 0.0 && d_hi != d_lo {
            (d_hi * lo - hi * d_lo) / (d_hi - d_lo)
        } else {
            (hi + lo) / 2.0
        };
        if (hi - lo) * span < 0.01 {
            break;
        }
    }
    s
}

fn limit(crop: Crop, bounds: Bounds, mode: u8, resize: Resize) -> Crop {
    limit_center(
        crop,
        bounds,
        resize,
        |m, p| {
            assert_eq!(m, mode);
            p
        },
        |m, p| {
            assert_eq!(m, mode);
            p
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(got: [f64; 4], want: [f64; 4]) {
        for (g, w) in got.into_iter().zip(want) {
            assert!((g - w).abs() < 0.02, "{got:?} != {want:?}");
        }
    }

    #[test]
    fn mode6_and_mode7_are_the_native_radial_pair() {
        let origin = [110.5, 73.25];
        let identity = |r: f64| r;
        assert_eq!(mode6([126.5, 61.25], origin, identity), [16.0, -12.0]);
        assert_eq!(mode6(origin, origin, identity), [0.0, 0.0]);
        assert_eq!(mode6([111.5, 73.25], origin, identity), [1.0, 0.0]);
        assert_eq!(mode7([16.0, -12.0], origin, identity), [126.5, 61.25]);
        assert_eq!(mode7([0.0, 0.0], origin, identity), origin);
    }

    #[test]
    fn mode8_matrix_matches_the_native_runtime_rows() {
        // Native 0x141E930 with the identity matrix.
        let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(mode8([123.5, -9.25], &identity, [false; 2]), [123.5, -9.25]);

        // Native 0x141E930 with a=0,b=1,d=-1,e=0: (x,y) -> (-y,x).
        let rotate = [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(mode8([3.5, 7.25], &rotate, [false; 2]), [-7.25, 3.5]);

        // Native 0x141E930 with t=(10,-4) and x flip.
        let translate = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(mode8([3.5, 7.25], &translate, [true, false]), [-13.5, 3.25]);

        // Native projective row: a=2,b=.5,c=.01,d=-1,e=3,f=.02,t=(1,-2).
        let projective =
            [2.0, 0.5, 0.01, -1.0, 3.0, 0.02, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let got = mode8([3.5, 7.25], &projective, [false; 2]);
        assert!((got[0] - 3.260_869_565_217_391_6).abs() < 1e-12);
        assert!((got[1] - 15.652_173_913_043_48).abs() < 1e-12);
    }

    #[test]
    fn resize3_matches_the_native_solver_matrix() {
        let bounds = Bounds::Mode10 { size: [100.0, 100.0] };
        let cases = [
            // requested [cx,cy,w,h] -> native [cx,cy,w,h]
            ([20.0, 35.0, 80.0, 40.0], [29.997_558_593_75, 35.0, 60.004_882_812_5, 20.002_441_406_25]),
            ([40.0, 50.0, 100.0, 40.0], [44.997_558_593_75, 50.0, 90.004_882_812_5, 32.001_953_125]),
            ([50.0, 10.0, 80.0, 40.0], [50.0, 14.999_389_648_437_5, 40.009_765_625, 30.001_220_703_125]),
            ([10.0, 50.0, 80.0, 40.0], [24.997_558_593_75, 50.0, 50.004_882_812_5, 10.002_441_406_25]),
            ([50.0, 10.0, 80.0, 180.0], [50.0, 49.997_558_593_75, 8.891_059_027_777_771, 100.004_882_812_5]),
            ([20.0, 20.0, 60.0, 80.0], [20.0, 23.333_740_234_375, 30.003_662_109_375, 46.672_363_281_25]),
            ([50.0, 50.0, 80.0, 80.0], [50.0, 50.0, 80.0, 80.0]),
        ];
        for (requested, native) in cases {
            let crop = Crop {
                center: [requested[0], requested[1]],
                size: [requested[2], requested[3]],
            };
            let got = limit(crop, bounds, 10, Resize::KeepArea1);
            close([got.center[0], got.center[1], got.size[0], got.size[1]], native);
        }
    }

    #[test]
    fn resize0_is_the_native_uniform_minimum() {
        let bounds = Bounds::Mode10 { size: [100.0, 100.0] };
        for (requested, native) in [
            ([20.0, 35.0, 80.0, 40.0], [20.0, 35.0, 40.004_882_812_5, 20.002_441_406_25]),
            ([50.0, 10.0, 80.0, 180.0], [50.0, 10.0, 8.891_059_027_777_771, 20.004_882_812_5]),
            ([50.0, 50.0, 80.0, 80.0], [50.0, 50.0, 80.0, 80.0]),
        ] {
            let crop = Crop {
                center: [requested[0], requested[1]],
                size: [requested[2], requested[3]],
            };
            let got = limit(crop, bounds, 10, Resize::Uniform);
            close([got.center[0], got.center[1], got.size[0], got.size[1]], native);
        }
    }

    #[test]
    fn mode6_solver_uses_the_native_inverse_direction() {
        let origin = [110.5, 73.25];
        let bounds = Bounds::Mode6 { min: [60.5, 33.25], max: [160.5, 113.25] };
        let crop = Crop { center: [40.0, 0.0], size: [40.0, 30.0] };
        let got = limit_center(
            crop,
            bounds,
            Resize::KeepArea1,
            |mode, p| {
                assert_eq!(mode, 7);
                mode7(p, origin, |r| r)
            },
            |mode, points| {
                assert_eq!(mode, 7);
                points.map(|p| mode7(p, origin, |r| r))
            },
        );
        close(
            [got.center[0], got.center[1], got.size[0], got.size[1]],
            [35.001_220_703_125, 0.0, 30.002_441_406_25, 15.003_662_109_375],
        );
    }

    #[test]
    fn mode8_boundary_matches_the_native_calculate_geo_row() {
        let got = mode8_boundary(
            Crop { center: [40.0, 0.0], size: [40.0, 30.0] },
            ([60.5, 33.25], [160.5, 113.25]),
            [110.5, 73.25],
            |r| r,
        );
        close(
            [got.center[0], got.center[1], got.size[0], got.size[1]],
            [35.001_220_703_125, 0.0, 30.002_441_406_25, 15.003_662_109_375],
        );
    }

    #[test]
    fn outside_and_boundary_centers_return_empty_without_trials() {
        for center in [[-1.0, 50.0], [0.0, 50.0], [100.0, 50.0], [50.0, 101.0]] {
            let crop = limit_center(
                Crop { center, size: [30.0, 20.0] },
                Bounds::Mode10 { size: [100.0, 100.0] },
                Resize::KeepArea1,
                |_, p| p,
                |_, _| panic!("outside center must not enter side search"),
            );
            assert_eq!(crop, Crop { center, size: [0.0; 2] });
        }
    }

    #[test]
    fn termination_uses_span_and_evaluation_cap() {
        assert_eq!(search_side(0.009, |_| panic!("degenerate span")), 1.0);
        let mut count = 0;
        assert_eq!(search_side(1.0, |_| { count += 1; 1.0 }), 1.0);
        assert_eq!(count, 1);
        let mut count = 0;
        search_side(f64::MAX, |s| { count += 1; if s > 0.5 { -1.0 } else { 1.0 } });
        assert_eq!(count, 51);
    }

    /// `ZVARIANTLAYER.ZLENSDISTORTION` for `_DSC5673.NEF`, variant layer 1.
    const DSC5673_LENS: [(f32, f32); 8] = [
        (0.0, 0.0),
        (742.0, -43.3392),
        (1483.9, -86.4269),
        (2225.9, -129.0114),
        (5935.7, -325.4922),
        (14097.2, -526.0558),
        (20032.8, -279.2483),
        (23742.6, 146.6017),
    ];
    /// `IC_SensorGeometry.rawPixelWidth/rawPixelHeight` for the Z6 III frame.
    const DSC5673_PITCH: [f32; 2] = [
        (35.9f64 * 1000.0 / 6048.0) as f32,
        (23.9f64 * 1000.0 / 4032.0) as f32,
    ];
    const DSC5673_OPTIC_CENTER: [f64; 2] = [3067.851_562, 1920.515_625];
    /// Native control points of `0x14204E0`, printed by `native_radial.exe`.
    const DSC5673_NATIVE: [(u64, u64, u64); 10] = [
        (0x0000000000000000, 0x0000000000000000, 0x3ff0000000000000),
        (0x0000000000000000, 0x0000000000000000, 0x3ff0000000000000),
        (0x405f45c7c0000000, 0x405d722c80000000, 0x3feda1aa02b116f7),
        (0x406f453da0000000, 0x406d72fdc0000000, 0x3fee4a9ccec4b81b),
        (0x40777410a0000000, 0x4076181300000000, 0x3fee1e20a5a0c603),
        (0x408f456040000000, 0x408d8e63e0000000, 0x3fee7f8709bbe33b),
        (0x40a2912980000000, 0x40a1dfcac0000000, 0x3ff01aa200c962c6),
        (0x40aa627900000000, 0x40aa045180000000, 0x3ff15322946cbd82),
        (0x40af454ee0000000, 0x40af76bce0000000, 0x3ff2630c73d3d1aa),
        (0x40af454ee0000000, 0x40af76bce0000000, 0x3ff2630c73d3d1aa),
    ];

    fn dsc5673_curve() -> RadialCurve {
        let built = RadialCurve::from_profile(&DSC5673_LENS, 100.0, DSC5673_PITCH);
        // The native table prepends one sentinel node, so catalog point `i`
        // lands on native node `i + 1`.
        assert_eq!(built.v.len() + 1, DSC5673_NATIVE.len() - 1);
        for (i, &(v, x, _)) in DSC5673_NATIVE[1..9].iter().enumerate() {
            assert_eq!(built.v[i].to_bits(), v, "v[{i}]");
            assert_eq!(built.x[i].to_bits(), x, "x[{i}]");
        }
        RadialCurve::from_native_table(
            DSC5673_NATIVE.iter().map(|n| f64::from_bits(n.0)).collect(),
            DSC5673_NATIVE.iter().map(|n| f64::from_bits(n.1)).collect(),
            DSC5673_NATIVE.iter().map(|n| f64::from_bits(n.2)).collect(),
        )
    }

    #[test]
    fn real_dsc5673_lens_table_is_exact() {
        // 0x14204E0 turns the catalog micrometre radii into raw pixels with
        // 1/sqrt((pitch.x² + pitch.y²)/2); every control point must match the
        // native table bit for bit.
        let curve = dsc5673_curve();
        assert_eq!(curve.v[0].to_bits(), 0);
        assert_eq!(curve.x[0].to_bits(), 0);
    }

    #[test]
    fn real_dsc5673_mode6_matches_the_native_curve() {
        let curve = dsc5673_curve();
        // Native p2p(mode=6) output radius, stored as f32.
        for (source, want) in [
            (200.0, 0x435499c7u32),
            (800.0, 0x4453dcdd),
            (1500.0, 0x44c538ae),
            (3000.0, 0x453fc51b),
            (3600.0, 0x45627b44),
        ] {
            let point = [DSC5673_OPTIC_CENTER[0] + source, DSC5673_OPTIC_CENTER[1]];
            let got = mode6_curve(point, DSC5673_OPTIC_CENTER, &curve);
            assert_eq!((got[0] as f32).to_bits(), want, "source {source}");
            assert_eq!(got[1], 0.0);
        }
    }
}
