//! Type1010 mesh core: `0x6190610` cluster splines and the two passes
//! `0x618FC00` / `0x6190040`.
//!
//! Ported from the bit-exact C# cores in
//! `dispatch-r8-20261106/p08/new-inner-corner` and `p09/new-outer-corner`,
//! which replay the pinned PE (`a84519da…`) fixture `positive_asym` with
//! `rawA/rawB = 4/4` bit equality.
//!
//! Inputs are explicit:
//! - `center` = `face+0x18AC/0x18B0` (0x60C4BA0 success path)
//! - `first`/`second` = the two 5-point clusters
//!   (`+0x120,+0xA0,+0xC0,+0xE0,+0x100` and
//!   `+0x1A8,+0x128,+0x148,+0x168,+0x188`, read order P4,P0,P1,P2,P3)
//! - `params` = the part's 28-float `ParamList`
//! - `degrees` = slots 0..3 (inner L/R, outer L/R)
//!
//! Output is A (prepared) and B (deformed), 120 points each, matching the PE
//! byte layout. Triangulation (`0x60F2B60`) is deliberately not here.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

impl core::ops::Add for Point {
    type Output = Point;
    fn add(self, other: Point) -> Point {
        Point::new(self.x + other.x, self.y + other.y)
    }
}

impl core::ops::Sub for Point {
    type Output = Point;
    fn sub(self, other: Point) -> Point {
        Point::new(self.x - other.x, self.y - other.y)
    }
}

impl core::ops::Mul<f32> for Point {
    type Output = Point;
    fn mul(self, scale: f32) -> Point {
        Point::new(self.x * scale, self.y * scale)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Cluster {
    pub p0: Point,
    pub p1: Point,
    pub p2: Point,
    pub p3: Point,
    pub p4: Point,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    MissingInput(&'static str),
    NonFinite(&'static str),
}

pub const VERTEX_COUNT: usize = 120;
pub const CLUSTER_POINTS: usize = 60;
pub const DEFORM_COUNT: usize = 28;

const CURVE_SCALE: f32 = 0.44999998807907104;
const ENDPOINT_PULL: f32 = 0.800000011920929;
const OUTER_A: f32 = 1.650001049041748;
const OUTER_B: f32 = 1.400001049041748;
const MID_A: f32 = -2.5000009536743164;
const MID_B: f32 = -3.0000009536743164;
const DEGREE_SCALE: f32 = 0.12000010162591934;
const ONE_THIRD: f32 = 0.3330000042915344;
const TWO_THIRDS: f32 = 0.6660000085830688;
const EPSILON: f32 = 0.0001;

fn finite(point: Point) -> bool {
    point.x.is_finite() && point.y.is_finite()
}

/// `0x60EBDF0` three-point type-30 branch, 15 samples.
pub fn sample15(input: [Point; 3]) -> [Point; 15] {
    let p = input;
    let mut t = [0.0f32; 3];
    for i in 1..3 {
        let dx = p[i].x - p[i - 1].x;
        let dy = p[i].y - p[i - 1].y;
        t[i] = t[i - 1] + (dx * dx + dy * dy).sqrt();
    }
    let cx = solve([p[0].x, p[1].x, p[2].x], t);
    let cy = solve([p[0].y, p[1].y, p[2].y], t);
    let mut out = [Point::new(0.0, 0.0); 15];
    let mut write = 0;
    for segment in 0..2 {
        let step = (t[segment + 1] - t[segment]) / 7.0;
        for sample in 0..7 {
            let knot = t[segment] + sample as f32 * step;
            let dx = knot - t[segment];
            out[write] = Point::new(evaluate(&cx, segment, dx), evaluate(&cy, segment, dx));
            write += 1;
        }
    }
    out[14] = p[2];
    out
}

struct Coeff {
    c0: [f32; 3],
    c1: [f32; 3],
    c2: [f32; 3],
    c3: [f32; 3],
}

fn solve(p: [f32; 3], t: [f32; 3]) -> Coeff {
    let h0 = t[1] - t[0];
    let h1 = t[2] - t[1];
    let diagonal = (((t[2] - t[0]) as f64) * (2.0 / 3.0)) as f32;
    let slope0 = (p[1] - p[0]) / h0;
    let slope1 = (p[2] - p[1]) / h1;
    let right = slope1 - slope0;

    let c2_0 = 0.0f32;
    let c2_1 = right * (1.0f32 / diagonal);
    let c2_2 = 0.0f32;

    let c3_0 = ((((c2_1 - c2_0) as f64) * (1.0 / 3.0)) / (h0 as f64)) as f32;
    let c3_1 = ((((c2_2 - c2_1) as f64) * (1.0 / 3.0)) / (h1 as f64)) as f32;

    let c1_0 = ((slope0 as f64)
        - ((2.0 * (c2_0 as f64) + (c2_1 as f64)) * (1.0 / 3.0) * (h0 as f64)))
        as f32;
    let c1_1 = ((slope1 as f64)
        - ((2.0 * (c2_1 as f64) + (c2_2 as f64)) * (1.0 / 3.0) * (h1 as f64)))
        as f32;
    let c1_2 = ((slope1 as f64)
        + (((c2_1 as f64) + 2.0 * (c2_2 as f64)) * (1.0 / 3.0) * (h1 as f64)))
        as f32;

    Coeff {
        c0: p,
        c1: [c1_0, c1_1, c1_2],
        c2: [c2_0, c2_1, c2_2],
        c3: [c3_0, c3_1, 0.0],
    }
}

fn evaluate(c: &Coeff, segment: usize, dx: f32) -> f32 {
    let mut value = c.c3[segment] * dx;
    value += c.c2[segment];
    value *= dx;
    value += c.c1[segment];
    value *= dx;
    value += c.c0[segment];
    value
}

fn blend(a: Point, wa: f32, b: Point, wb: f32) -> Point {
    Point::new(a.x * wa + b.x * wb, a.y * wa + b.y * wb)
}

/// `0x6190610` one 60-point cluster.
pub fn build_cluster(cluster: &Cluster, second_branch: bool) -> [Point; CLUSTER_POINTS] {
    let p0 = cluster.p0;
    let p1 = cluster.p1;
    let p2 = cluster.p2;
    let p3 = cluster.p3;
    let p4 = cluster.p4;

    let p1a = p1 - (p4 - p1) * ENDPOINT_PULL;
    let p3a = p3 - (p4 - p3) * ENDPOINT_PULL;

    let outer_a = p0 + ((p0 - p4) * OUTER_A) * CURVE_SCALE;
    let outer_b = p2 + ((p2 - p4) * OUTER_B) * CURVE_SCALE;
    let (a4, a5) = if second_branch {
        (outer_b, outer_a)
    } else {
        (outer_a, outer_b)
    };

    let mid_a = p1a + ((p4 - p1a) * CURVE_SCALE) * MID_A;
    let mid_b = p3a + ((p4 - p3a) * CURVE_SCALE) * MID_B;

    let s0 = sample15([p0, p1a, p2]);
    let s1 = sample15([a4, mid_a, a5]);
    let s2 = sample15([p2, p3a, p0]);
    let s3 = sample15([a5, mid_b, a4]);

    let guide = [
        blend(s0[4], ONE_THIRD, s2[10], TWO_THIRDS),
        blend(s0[4], TWO_THIRDS, s2[10], ONE_THIRD),
        blend(s0[10], ONE_THIRD, s2[4], TWO_THIRDS),
        blend(s0[10], TWO_THIRDS, s2[4], ONE_THIRD),
    ];

    let mut points = [Point::new(0.0, 0.0); CLUSTER_POINTS];
    let mut write = 0;
    points[write..write + 15].copy_from_slice(&s0);
    write += 15;
    points[write..write + 13].copy_from_slice(&s2[1..14]);
    write += 13;
    points[write..write + 15].copy_from_slice(&s1);
    write += 15;
    points[write..write + 13].copy_from_slice(&s3[1..14]);
    write += 13;
    points[write..write + 4].copy_from_slice(&guide);
    points
}

fn distance(a: Point, b: Point) -> f32 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

fn minimum_difference(tip: Point, points: &[Point; VERTEX_COUNT], a: usize, b: usize) -> f32 {
    let mut minimum = 10000.0f32;
    for i in 0..DEFORM_COUNT {
        let da = distance(tip, points[a + i]);
        let db = distance(tip, points[b + i]);
        minimum = minimum.min(da - db);
    }
    minimum
}

fn apply_pass(
    points: &mut [Point; VERTEX_COUNT],
    center: Point,
    first: &Cluster,
    second: &Cluster,
    weights: &[f32; DEFORM_COUNT],
    degree_x: f32,
    degree_y: f32,
    negate: bool,
) {
    let mut dir1 = (first.p2 - first.p0) * 0.5;
    let mut dir2 = (second.p2 - second.p0) * 0.5;
    if negate {
        dir1 = Point::new(-dir1.x, -dir1.y);
        dir2 = Point::new(-dir2.x, -dir2.y);
    }

    let min1 = minimum_difference(second.p4, points, 88, 60).max(0.0);
    let min2 = minimum_difference(first.p4, points, 28, 0).max(0.0);

    let amp_x = (degree_x * DEGREE_SCALE) * center.x;
    let amp_y = (degree_y * DEGREE_SCALE) * center.y;
    let mut scale1 = 1.0f32;
    let mut scale2 = 1.0f32;
    for i in 0..15 {
        let weight = weights[i];
        let v1 = dir1 * (weight * amp_x);
        let l1 = (v1.x * v1.x + v1.y * v1.y).sqrt();
        if l1 > EPSILON {
            scale1 = scale1.min(min1 / l1);
        }
        let v2 = dir2 * (weight * amp_y);
        let l2 = (v2.x * v2.x + v2.y * v2.y).sqrt();
        if l2 > EPSILON {
            scale2 = scale2.min(min2 / l2);
        }
    }

    let applied_x = amp_x * scale1;
    let applied_y = amp_y * scale2;
    for i in 0..DEFORM_COUNT {
        let weight = weights[i];
        points[i] = points[i] + dir1 * (weight * applied_x);
        points[CLUSTER_POINTS + i] = points[CLUSTER_POINTS + i] + dir2 * (weight * applied_y);
    }
}

#[derive(Debug)]
pub struct Mesh {
    pub prepared_a: [Point; VERTEX_COUNT],
    pub deformed_b: [Point; VERTEX_COUNT],
}

/// Full `0x6190610 -> 0x618FC00 -> 0x6190040` chain.
pub fn compute(
    center: Point,
    first: &Cluster,
    second: &Cluster,
    params: &[f32; 28],
    degrees: [f32; 4],
) -> Result<Mesh, Refusal> {
    if !finite(center) {
        return Err(Refusal::NonFinite("center"));
    }
    for cluster in [first, second] {
        for point in [cluster.p0, cluster.p1, cluster.p2, cluster.p3, cluster.p4] {
            if !finite(point) {
                return Err(Refusal::NonFinite("cluster"));
            }
        }
    }
    for value in params.iter().copied().chain(degrees.iter().copied()) {
        if !value.is_finite() {
            return Err(Refusal::NonFinite("params/degrees"));
        }
    }

    let first_cluster = build_cluster(first, false);
    let second_cluster = build_cluster(second, true);
    let mut prepared_a = [Point::new(0.0, 0.0); VERTEX_COUNT];
    prepared_a[..CLUSTER_POINTS].copy_from_slice(&first_cluster);
    prepared_a[CLUSTER_POINTS..].copy_from_slice(&second_cluster);

    let mut pass1 = [0.0f32; DEFORM_COUNT];
    pass1[10..19].copy_from_slice(&params[10..19]);
    let mut pass2 = [0.0f32; DEFORM_COUNT];
    pass2[..5].copy_from_slice(&params[..5]);
    pass2[24..28].copy_from_slice(&params[24..28]);

    let mut deformed_b = prepared_a;
    apply_pass(
        &mut deformed_b,
        center,
        first,
        second,
        &pass1,
        degrees[0],
        degrees[1],
        false,
    );
    apply_pass(
        &mut deformed_b,
        center,
        first,
        second,
        &pass2,
        degrees[2],
        degrees[3],
        true,
    );

    Ok(Mesh {
        prepared_a,
        deformed_b,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const CENTER: Point = Point::new(0.75, 0.25);
    const FIRST: Cluster = Cluster {
        p0: Point::new(100.0, 100.0),
        p1: Point::new(124.0, 116.0),
        p2: Point::new(78.0, 132.0),
        p3: Point::new(92.0, 84.0),
        p4: Point::new(118.0, 148.0),
    };
    const SECOND: Cluster = Cluster {
        p0: Point::new(200.0, 200.0),
        p1: Point::new(176.0, 186.0),
        p2: Point::new(226.0, 218.0),
        p3: Point::new(212.0, 164.0),
        p4: Point::new(184.0, 242.0),
    };
    const DEGREES: [f32; 4] = [0.8, -0.6, 0.4, -0.2];

    const PARAM8: [f32; 28] = [
        0.3400000035762787, 0.41999998688697815, 0.25, 0.09999999403953552, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.09999999403953552, 0.29999998211860657,
        0.47999998927116394, 0.699999988079071, 0.8999999761581421, 1.0, 0.5,
        0.29999998211860657, 0.07000000029802322, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.10080999881029129, 0.3002699911594391, 0.5, 0.8499999642372131,
    ];
    #[cfg(feature = "external-fixtures")]
    const PARAM9: [f32; 28] = [
        0.3799999952316284, 0.3499999940395355, 0.17999999225139618,
        0.07999999821186066, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.14000000059604645, 0.3400000035762787, 0.5, 0.7400000095367432, 1.0,
        0.5199999809265137, 0.3999999761581421, 0.2800000011920929, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.47999998927116394, 0.7799999713897705, 1.0,
    ];

    #[cfg(feature = "external-fixtures")]

    fn fixture(bytes: &[u8]) -> [Point; VERTEX_COUNT] {
        assert_eq!(bytes.len(), VERTEX_COUNT * 8);
        let mut out = [Point::new(0.0, 0.0); VERTEX_COUNT];
        for (i, slot) in out.iter_mut().enumerate() {
            let x = f32::from_le_bytes([bytes[i * 8], bytes[i * 8 + 1], bytes[i * 8 + 2], bytes[i * 8 + 3]]);
            let y = f32::from_le_bytes([bytes[i * 8 + 4], bytes[i * 8 + 5], bytes[i * 8 + 6], bytes[i * 8 + 7]]);
            *slot = Point::new(x, y);
        }
        out
    }

    fn bits_equal(a: &[Point; VERTEX_COUNT], b: &[Point; VERTEX_COUNT]) -> bool {
        a.iter()
            .zip(b)
            .all(|(x, y)| x.x.to_bits() == y.x.to_bits() && x.y.to_bits() == y.y.to_bits())
    }

    #[test]
    #[cfg(feature = "external-fixtures")]
    fn part8_matches_pe_fixture_bitwise() {
        let mesh = compute(CENTER, &FIRST, &SECOND, &PARAM8, DEGREES).unwrap();
        assert!(bits_equal(
            &mesh.prepared_a,
            &fixture(&crate::external_fixtures::read("eye/positive_asym.part8.rawA.bin"))
        ));
        assert!(bits_equal(
            &mesh.deformed_b,
            &fixture(&crate::external_fixtures::read("eye/positive_asym.part8.rawB.bin"))
        ));
    }

    #[test]
    #[cfg(feature = "external-fixtures")]
    fn part9_matches_pe_fixture_bitwise() {
        let mesh = compute(CENTER, &FIRST, &SECOND, &PARAM9, DEGREES).unwrap();
        assert!(bits_equal(
            &mesh.prepared_a,
            &fixture(&crate::external_fixtures::read("eye/positive_asym.part9.rawA.bin"))
        ));
        assert!(bits_equal(
            &mesh.deformed_b,
            &fixture(&crate::external_fixtures::read("eye/positive_asym.part9.rawB.bin"))
        ));
    }

    #[test]
    fn zero_degrees_are_identity() {
        let mesh = compute(CENTER, &FIRST, &SECOND, &PARAM8, [0.0; 4]).unwrap();
        assert!(bits_equal(&mesh.prepared_a, &mesh.deformed_b));
    }

    #[test]
    fn cluster_has_60_points_and_ends_with_guides() {
        let points = build_cluster(&FIRST, false);
        assert_eq!(points.len(), 60);
        assert_eq!(points[0], FIRST.p0);
    }

    #[test]
    fn sample15_keeps_endpoints_on_a_line() {
        let p = [Point::new(0.0, 0.0), Point::new(1.0, 1.0), Point::new(2.0, 2.0)];
        let samples = sample15(p);
        assert_eq!(samples[0], p[0]);
        assert_eq!(samples[14], p[2]);
        for sample in samples {
            assert!((sample.x - sample.y).abs() < 1e-5);
        }
    }

    #[test]
    fn non_finite_input_is_refused() {
        let mut cluster = FIRST;
        cluster.p0.x = f32::NAN;
        assert_eq!(
            compute(CENTER, &cluster, &SECOND, &PARAM8, DEGREES).unwrap_err(),
            Refusal::NonFinite("cluster")
        );
    }
}
