//! Recovered `194 -> 414` point producer used by PF30092/PF30093 Type1016.
//!
//! Input is the same-frame `face+0x008` `float2[194]` array. Output is the
//! `face+0x924` `float2[414]` view consumed by `face_geometry::face_width`.
//! The implementation follows the fixed 11-stage producer in
//! `backend-r77/s207_pf30092_414_math_audit_finish`.

pub const INPUT_COUNT: usize = 194;
pub const OUTPUT_COUNT: usize = 414;

const DF0_CURVES: [([usize; 3], &[u32], &[u32], usize); 3] = [
    (
        [158, 193, 148],
        &[0x0000_0000, 0x3ea8_f5c3, 0x3f2b_851f],
        &[0x0000_0000, 0x3ea8_f5c3, 0x3f2b_851f, 0x3f80_0000],
        5,
    ),
    (
        [160, 192, 146],
        &[0x0000_0000, 0x3e80_0000, 0x3f00_0000, 0x3f40_0000],
        &[
            0x0000_0000,
            0x3e80_0000,
            0x3f00_0000,
            0x3f40_0000,
            0x3f80_0000,
        ],
        7,
    ),
    (
        [162, 191, 144],
        &[
            0x0000_0000,
            0x3e4c_cccd,
            0x3ecc_cccd,
            0x3f19_999a,
            0x3f4c_cccd,
        ],
        &[
            0x0000_0000,
            0x3e4c_cccd,
            0x3ecc_cccd,
            0x3f19_999a,
            0x3f4c_cccd,
            0x3f80_0000,
        ],
        9,
    ),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Error {
    Points194,
    NonFinite,
    UnsupportedScale,
}

#[inline]
fn fadd(a: f32, b: f32) -> f32 {
    a + b
}

#[inline]
fn fsub(a: f32, b: f32) -> f32 {
    a - b
}

#[inline]
fn fmul(a: f32, b: f32) -> f32 {
    a * b
}

#[inline]
fn fdiv(a: f32, b: f32) -> f32 {
    a / b
}

#[inline]
fn fadd2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [fadd(a[0], b[0]), fadd(a[1], b[1])]
}

#[inline]
fn fsub2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [fsub(a[0], b[0]), fsub(a[1], b[1])]
}

#[inline]
fn fmul2(a: [f32; 2], k: f32) -> [f32; 2] {
    [fmul(a[0], k), fmul(a[1], k)]
}

#[inline]
fn half2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    fmul2(fadd2(a, b), 0.5)
}

#[inline]
fn fdiv2(a: [f32; 2], k: f32) -> [f32; 2] {
    [fdiv(a[0], k), fdiv(a[1], k)]
}

fn cum_arc(points: &[[f32; 2]]) -> Vec<f32> {
    let mut arc = Vec::with_capacity(points.len());
    arc.push(0.0);
    for i in 1..points.len() {
        let dx = fsub(points[i][0], points[i - 1][0]);
        let dy = fsub(points[i][1], points[i - 1][1]);
        let distance = fmul(dx, dx) + fmul(dy, dy);
        arc.push(fadd(arc[i - 1], distance.sqrt()));
    }
    arc
}

#[inline]
fn d64(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        if a == 0.0 || a.is_nan() {
            return f64::NAN;
        }
        return f64::INFINITY.copysign(a * b.signum());
    }
    a / b
}

#[inline]
fn add64(a: f64, b: f64) -> f64 {
    ((a + b) as f32) as f64
}

fn mul64(a: f64, b: f64) -> f64 {
    ((a * b) as f32) as f64
}

fn solve(diag: &[f64], upper: &[f64], lower: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = diag.len();
    let mut d = diag.to_vec();
    let mut u = upper.to_vec();
    let mut l = lower.to_vec();
    let mut inverse = Vec::with_capacity(n);
    for i in 0..n {
        let q = fdiv(1.0, d[i] as f32);
        inverse.push(q as f64);
        d[i] = fmul(d[i] as f32, q) as f64;
        if i > 0 {
            l[i] = fmul(l[i] as f32, q) as f64;
        }
        if i + 1 < n {
            u[i] = fmul(u[i] as f32, q) as f64;
        }
        d[i] = 1.0;
    }
    for i in 0..n.saturating_sub(1) {
        let factor = fdiv(-(l[i + 1] as f32), d[i] as f32);
        l[i + 1] = -factor as f64;
        d[i + 1] = fadd(d[i + 1] as f32, fmul(factor, u[i] as f32)) as f64;
    }
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let s = if i == 0 {
            0.0
        } else {
            fadd(fmul(l[i] as f32, y[i - 1] as f32), 0.0)
        };
        y.push(fsub(fmul(rhs[i] as f32, inverse[i] as f32), s) as f64);
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        let s = if i + 1 < n {
            fadd(0.0, fmul(u[i] as f32, x[i + 1] as f32))
        } else {
            0.0
        };
        x[i] = fdiv(fsub(y[i] as f32, s), d[i] as f32) as f64;
    }
    x
}

struct Spline {
    x: Vec<f32>,
    y: Vec<f32>,
    c: Vec<f32>,
    b: Vec<f64>,
    a: Vec<f32>,
}

fn coefficients(knots: &[f32], values: &[f32]) -> Spline {
    let x: Vec<f32> = knots.to_vec();
    let y: Vec<f32> = values.to_vec();
    let n = x.len();
    let mut d = vec![0.0f64; n];
    let mut u = vec![0.0f64; n];
    let mut l = vec![0.0f64; n];
    let mut rhs = vec![0.0f64; n];
    for i in 1..n.saturating_sub(1) {
        l[i] = (((fsub(x[i], x[i - 1]) as f64) * (1.0 / 3.0)) as f32) as f64;
        d[i] = (((fsub(x[i + 1], x[i - 1]) as f64) * (2.0 / 3.0)) as f32) as f64;
        u[i] = (((fsub(x[i + 1], x[i]) as f64) * (1.0 / 3.0)) as f32) as f64;
        let left = fdiv(fsub(y[i + 1], y[i]), fsub(x[i + 1], x[i]));
        let right = fdiv(fsub(y[i], y[i - 1]), fsub(x[i], x[i - 1]));
        rhs[i] = fsub(left, right) as f64;
    }
    d[0] = 2.0;
    u[0] = 0.0;
    rhs[0] = 0.0;
    d[n - 1] = 2.0;
    l[n - 1] = 0.0;
    rhs[n - 1] = 0.0;
    let b = solve(&d, &u, &l, &rhs);
    let mut a = vec![0.0f32; n];
    let mut c = vec![0.0f32; n];
    for i in 0..n.saturating_sub(1) {
        let h = fsub(x[i + 1], x[i]);
        a[i] = d64(
            ((fsub(b[i + 1] as f32, b[i] as f32)) as f64) * (1.0 / 3.0),
            h as f64,
        ) as f32;
        let left = fdiv(fsub(y[i + 1], y[i]), h) as f64;
        let right = ((2.0 * b[i] + b[i + 1]) * (-1.0 / 3.0)) * h as f64;
        c[i] = (left + right) as f32;
    }
    let h = fsub(x[n - 1], x[n - 2]);
    a[n - 1] = 0.0;
    c[n - 1] = (c[n - 2] as f64
        + ((3.0 * a[n - 2] as f64) * h as f64) * h as f64
        + (2.0 * b[n - 2]) * h as f64) as f32;
    Spline { x, y, c, b, a }
}

fn evaluate(spline: &Spline, t: f32) -> f32 {
    let t = t;
    let mut lo = 0usize;
    let mut hi = spline.x.len();
    while hi > 0 {
        let half = hi / 2;
        let m = lo + half;
        if spline.x[m] <= t {
            lo = m + 1;
            hi -= half + 1;
        } else {
            hi = half;
        }
    }
    let i = lo.saturating_sub(1);
    let h = fsub(t, spline.x[i]);
    if t < spline.x[0] {
        return add64(
            mul64(
                add64(mul64(spline.b[0], h as f64), spline.c[0] as f64),
                h as f64,
            ),
            spline.y[0] as f64,
        ) as f32;
    }
    if t > *spline.x.last().unwrap() {
        return add64(
            mul64(
                add64(
                    mul64(*spline.b.last().unwrap(), h as f64),
                    *spline.c.last().unwrap() as f64,
                ),
                h as f64,
            ),
            *spline.y.last().unwrap() as f64,
        ) as f32;
    }
    add64(
        mul64(
            add64(
                mul64(
                    add64(mul64(spline.a[i] as f64, h as f64), spline.b[i]),
                    h as f64,
                ),
                spline.c[i] as f64,
            ),
            h as f64,
        ),
        spline.y[i] as f64,
    ) as f32
}

fn spline_pair(knots: &[f32], values: &[f32]) -> Spline {
    coefficients(knots, values)
}

fn sample_curve(points: &[[f32; 2]], per_segment: usize) -> Vec<[f32; 2]> {
    let xs: Vec<f32> = points.iter().map(|p| p[0]).collect();
    let ys: Vec<f32> = points.iter().map(|p| p[1]).collect();
    let arc = cum_arc(points);
    let sx = spline_pair(&arc, &xs);
    let sy = spline_pair(&arc, &ys);
    let mut out = Vec::new();
    for i in 1..points.len() {
        let step = fdiv(fsub(arc[i], arc[i - 1]), per_segment as f32);
        for j in 0..per_segment {
            let t = fadd(fmul(j as f32, step), arc[i - 1]);
            out.push([evaluate(&sx, t), evaluate(&sy, t)]);
        }
    }
    out.push(*points.last().unwrap());
    out
}

fn edfd0(points: &[[f32; 2]], count: usize) -> Vec<[f32; 2]> {
    let xs: Vec<f32> = points.iter().map(|p| p[0]).collect();
    let ys: Vec<f32> = points.iter().map(|p| p[1]).collect();
    let arc = cum_arc(points);
    let sx = spline_pair(&arc, &xs);
    let sy = spline_pair(&arc, &ys);
    let step = fdiv(fsub(*arc.last().unwrap(), arc[0]), (count + 1) as f32);
    let mut out = Vec::with_capacity(count);
    for j in 0..count {
        let t = fadd(fmul((j + 1) as f32, step), arc[0]);
        out.push([evaluate(&sx, t), evaluate(&sy, t)]);
    }
    out
}

fn arc_block(out: &mut Vec<[f32; 2]>, src: &[[f32; 2]], points: &[usize], copies: &[usize]) {
    let pts: Vec<[f32; 2]> = points.iter().map(|&i| src[i]).collect();
    let arc = cum_arc(&pts);
    let xs: Vec<f32> = pts.iter().map(|p| p[0]).collect();
    let ys: Vec<f32> = pts.iter().map(|p| p[1]).collect();
    let sx = spline_pair(&arc, &xs);
    let sy = spline_pair(&arc, &ys);
    for k in 0..points.len() - 1 {
        let t = fmul(fadd(arc[k], arc[k + 1]), 0.5);
        out.push([evaluate(&sx, t), evaluate(&sy, t)]);
        out.push(src[copies[k]]);
    }
}

fn sqdist(points: &[[f32; 2]], i: usize, j: usize) -> f32 {
    let dx = fsub(points[i][0], points[j][0]);
    let dy = fsub(points[i][1], points[j][1]);
    fadd(fmul(dx, dx), fmul(dy, dy))
}

fn fdiv32(a: f32, b: f32) -> f32 {
    if b == 0.0 {
        if a == 0.0 || a.is_nan() {
            return f32::NAN;
        }
        return f32::INFINITY.copysign(a * b.signum());
    }
    fdiv(a, b)
}

fn flags(points: &[[f32; 2]]) -> (i32, i32) {
    let sq110 = sqdist(points, 110, 18);
    let sq140 = sqdist(points, 140, 36);
    let d109 = sqdist(points, 109, 18).sqrt();
    let d110 = sq110.sqrt();
    let d141 = sqdist(points, 141, 36).sqrt();
    let d140 = sq140.sqrt();
    let r0 = fdiv32(d141, d140);
    let r1 = fdiv32(d109, d110);
    let b0 = !r0.is_nan() && 1.5 <= r0;
    let b1 = !r1.is_nan() && 1.5 <= r1;
    let obj0 = if (sq110 != 0.0 || sq110.is_nan()) && b1 {
        1
    } else {
        0
    };
    let obj1 = if (sq140 != 0.0 || sq140.is_nan()) && b0 {
        -1
    } else {
        0
    };
    (obj0, obj1)
}

pub fn produce(points194: &[[f32; 2]], scale: f32) -> Result<[[f32; 2]; OUTPUT_COUNT], Error> {
    if points194.len() != INPUT_COUNT {
        return Err(Error::Points194);
    }
    if points194.iter().flatten().any(|value| !value.is_finite()) || !scale.is_finite() {
        return Err(Error::NonFinite);
    }
    if scale.abs() > 20.0 {
        return Err(Error::UnsupportedScale);
    }
    let src: Vec<[f32; 2]> = points194.to_vec();
    let (obj0, obj1) = flags(&src);
    let mut out: Vec<[f32; 2]> = Vec::with_capacity(OUTPUT_COUNT);

    for &value in &src[0..5] {
        out.push(value);
    }
    let a = half2(src[4], src[5]);
    let b = half2(src[6], src[3]);
    let mid = half2(a, b);
    out.push(fadd2(a, fmul2(fsub2(a, mid), 0.5)));
    for &value in &src[5..14] {
        out.push(value);
    }
    let a = half2(src[13], src[14]);
    let b = half2(src[15], src[12]);
    let mid = half2(a, b);
    out.push(fadd2(a, fmul2(fsub2(a, mid), 0.5)));
    for &value in &src[14..18] {
        out.push(value);
    }

    out.push(src[18]);
    arc_block(&mut out, &src, &[18, 19, 20, 21, 22], &[19, 20, 21, 22]);
    arc_block(&mut out, &src, &[22, 23, 24, 25, 18], &[23, 24, 25, 26]);

    out.push(src[36]);
    arc_block(&mut out, &src, &[36, 37, 38, 39, 40], &[37, 38, 39, 40]);
    arc_block(&mut out, &src, &[40, 41, 42, 43, 36], &[41, 42, 43, 44]);

    let stage4 = half2(out[5], out[15]);
    out.push(stage4);
    out.extend_from_slice(&src[54..73]);
    out.extend_from_slice(&src[73..107]);

    let curve = sample_curve(
        &[
            half2(src[73], src[63]),
            half2(src[78], src[58]),
            half2(src[83], src[70]),
        ],
        3,
    );
    out.extend_from_slice(&curve[..7]);
    out.extend_from_slice(&src[109..165]);

    out.push(half2(src[18], src[(obj0 + 109) as usize]));
    let mut fan = [[0.0f32; 2]; 4];
    for k in 0..4 {
        fan[k] = half2(src[60 + k], src[(obj0 + 111 + k as i32) as usize]);
    }
    for (k, count) in [(0usize, 5usize), (1, 4), (2, 3), (3, 3)] {
        let points = [src[(obj0 + 111 + k as i32) as usize], fan[k], src[60 + k]];
        for value in edfd0(&points, count) {
            out.push(value);
        }
    }
    let base = src[(obj0 + 116) as usize];
    let d = fdiv2(fsub2(out[108], base), 3.0);
    out.push(fadd2(base, d));
    out.push(fadd2(base, fadd2(d, d)));
    let base2 = src[(obj0 + 118) as usize];
    let e = fdiv2(fsub2(src[73], base2), 3.0);
    out.push(fadd2(base2, e));
    out.push(fadd2(base2, fadd2(e, e)));

    out.push(half2(src[36], src[(obj1 + 141) as usize]));
    for k in 0..4 {
        fan[k] = half2(src[67 + k], src[(obj1 + 139 - k as i32) as usize]);
    }
    for (k, count) in [(0usize, 5usize), (1, 4), (2, 3), (3, 3)] {
        let points = [src[(obj1 + 139 - k as i32) as usize], fan[k], src[67 + k]];
        for value in edfd0(&points, count) {
            out.push(value);
        }
    }
    let base = src[(obj1 + 134) as usize];
    let d = fdiv2(fsub2(out[114], base), 3.0);
    out.push(fadd2(base, d));
    out.push(fadd2(base, fadd2(d, d)));
    let base2 = src[(obj1 + 132) as usize];
    let e = fdiv2(fsub2(src[83], base2), 3.0);
    out.push(fadd2(base2, e));
    out.push(fadd2(base2, fadd2(e, e)));

    let a = src[(obj0 + 122) as usize];
    let b = src[(obj1 + 128) as usize];
    let third = f32::from_bits(0x3eaa_a64c);
    let d1 = fmul2(fsub2(src[92], a), third);
    let d2 = fmul2(fsub2(src[86], b), third);
    for phase in 0..2 {
        let (p0, pmid, p4) = if phase == 0 {
            (out[189], src[108], out[209])
        } else {
            (out[190], src[107], out[210])
        };
        let e1 = if phase == 0 { d1 } else { fadd2(d1, d1) };
        let e2 = if phase == 0 { d2 } else { fadd2(d2, d2) };
        let curve = sample_curve(&[p0, fadd2(a, e1), pmid, fadd2(b, e2), p4], 2);
        out.extend_from_slice(&curve[1..8]);
    }

    for (indices, first, second, take) in DF0_CURVES {
        let points = [src[indices[0]], src[indices[1]], src[indices[2]]];
        let xs: Vec<f32> = points.iter().map(|p| p[0]).collect();
        let ys: Vec<f32> = points.iter().map(|p| p[1]).collect();
        let arc = cum_arc(&points);
        let sx = spline_pair(&arc, &xs);
        let sy = spline_pair(&arc, &ys);
        let mut sampled = Vec::new();
        for (segment, times) in [(0usize, first), (1usize, second)] {
            let span = fsub(arc[segment + 1], arc[segment]);
            for &bits in times {
                let t = f32::from_bits(bits);
                let at = fadd(fmul(t, span), arc[segment]);
                sampled.push([evaluate(&sx, at), evaluate(&sy, at)]);
            }
        }
        out.extend_from_slice(&sampled[1..=take]);
    }

    let anchor = src[54];
    let tail = [
        f32::from_bits(0x3dcc_cccd),
        f32::from_bits(0x3e61_47ae),
        f32::from_bits(0x3eb3_3333),
    ];
    for factor in tail {
        for j in 0..56 {
            let point = src[109 + j];
            out.push(fadd2(point, fmul2(fsub2(point, anchor), factor)));
        }
    }

    if out.len() != OUTPUT_COUNT {
        return Err(Error::Points194);
    }
    let mut result = [[0.0f32; 2]; OUTPUT_COUNT];
    result.copy_from_slice(&out);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    const INPUT: &[u8] = crate::private_assets::or_empty(crate::private_asset!("fixtures/portrait/pf30092/pf30092.194.f32le"));
    const EXPECTED: &[u8] = crate::private_assets::or_empty(crate::private_asset!("fixtures/portrait/pf30092/pf30092.414.f32le"));

    fn points194() -> Vec<[f32; 2]> {
        INPUT
            .chunks_exact(8)
            .map(|chunk| {
                [
                    f32::from_le_bytes(chunk[..4].try_into().unwrap()),
                    f32::from_le_bytes(chunk[4..].try_into().unwrap()),
                ]
            })
            .collect()
    }

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn fixed_fixture_matches_recovered_414_bytes() {
        let output = produce(&points194(), 0.0).unwrap();
        let raw: Vec<u8> = output
            .iter()
            .flat_map(|point| [point[0].to_le_bytes(), point[1].to_le_bytes()].concat())
            .collect();
        if raw != EXPECTED {
            let index = raw
                .iter()
                .zip(EXPECTED)
                .position(|(left, right)| left != right)
                .unwrap();
            panic!(
                "first byte diff at {} (point {}, component {}): got {} expected {}",
                index,
                index / 8,
                (index % 8) / 4,
                raw[index],
                EXPECTED[index]
            );
        }
    }

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn short_or_non_finite_inputs_refuse() {
        assert_eq!(produce(&points194()[..193], 0.0), Err(Error::Points194));
        let mut points = points194();
        points[0][0] = f32::NAN;
        assert_eq!(produce(&points, 0.0), Err(Error::NonFinite));
    }
}
