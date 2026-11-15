//! Type1010 (priority rows 17/18: PF30071/30077 inner, PF30072/30078 outer).
//!
//! Values in this file are tied to the pinned PE
//! `a84519da489bc52e4e914af7880ede43e3ce58992657723d02197161832464c7`.
//!
//! ```text
//! 0x60F03F6  Type1010 factory: alloc 0x98, outer vptr 0x92CA9A0
//! 0x60F0426  inner ctor 0x618F850
//! 0x618F874  handler+0x19 = 0        (EnableDymGrid default)
//! 0x618F878  inner vptr 0x92D2EB0
//! 0x618F902  payload+0x10 = 0x78     -> 120 vertices
//! 0x618F909  alloc 0x3C0             -> 120 * 8 bytes A
//! 0x618F929  alloc 0x3C0             -> 120 * 8 bytes B
//!            (no index allocation in the ctor)
//! 0x92D2EB0  +0x10 = 0x618FA80       -> mesh producer
//! 0x618FB7A  fill A[0..60]  via 0x6190610 (flag 1)
//! 0x618FB97  fill A[60..120] via 0x6190610 (flag 0)
//! 0x618FBAF  B = A (120 * 8 bytes)
//! 0x618FBC4  0x618FC00 pass 1 (degree slots 0/1)
//! 0x618FBD2  0x6190040 pass 2 (degree slots 2/3)
//! 0x618FBDA  call 0x60F2B60          -> EBO is ALWAYS rebuilt
//! ```
//!
//! There is no static Type1010 index table: `0x60F2B60` triangulates the 120
//! A points and writes `3*T` uint16 indices (low words of int32 triples).
//! The cluster/spline preparation (0x6190610) and the two deformation passes
//! (0x618FC00, 0x6190040) are separately recovered in
//! `dispatch-r8-20261106/p08/new-inner-corner` and `p09/new-outer-corner`;
//! this module covers the degree/amplitude math at 0x618FA80 and the EBO
//! contract, and refuses to invent the rest.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// part 8, `RenderDegreeRanges [-1,0,0]`
    Negative = 8,
    /// part 9, `RenderDegreeRanges [0,0,1]`
    Positive = 9,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    BadPart(u8),
    NonFinite(&'static str),
    MissingTriangles,
    TriangleOverflow,
}

/// `0x92D2C20[0]` / `0x92D2DB0[0]`.
pub const SCALE: f32 = 0.12000010162591934;
pub const VERTEX_COUNT: usize = 120;
pub const HALF_COUNT: usize = 60;

/// Per-part config (parts 8/9 share the same four control slots).
pub const CONTROL_SLOTS: [(u32, u8); 4] = [
    (30071, 0),
    (30077, 1),
    (30072, 2),
    (30078, 3),
];

/// `0x4C96360` three-knot float32 interpolation for Type1010.
pub fn map_degree(native: f32, part: Part) -> Result<f32, Refusal> {
    if !native.is_finite() {
        return Err(Refusal::NonFinite("native"));
    }
    let values = match part {
        Part::Negative => [-1.0f32, 0.0, 0.0],
        Part::Positive => [0.0f32, 0.0, 1.0],
    };
    let knots = [0.0f32, 0.5, 1.0];
    for i in 1..3 {
        if native < knots[i - 1] || knots[i] < native {
            continue;
        }
        let span = knots[i] - knots[i - 1];
        return Ok(values[i - 1]
            + (values[i] - values[i - 1]) * ((native - knots[i - 1]) / span));
    }
    Ok(native)
}

/// Degree slot vector written by `0x4C965A9/0x4C965C0`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Degrees {
    pub inner_left: f32,
    pub inner_right: f32,
    pub outer_left: f32,
    pub outer_right: f32,
}

impl Degrees {
    pub fn slots(&self) -> [f32; 4] {
        [
            self.inner_left,
            self.inner_right,
            self.outer_left,
            self.outer_right,
        ]
    }
}

/// `0x618FAF9..0x618FB5B`: center-scaled amplitudes and the shader uniform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Amplitudes {
    /// pass 1 `xmm8` pair (degree slots 0/1)
    pub inner: [f32; 2],
    /// pass 2 `xmm6/xmm7` (degree slots 2/3)
    pub outer: [f32; 2],
    /// `payload+0x24` = max |amplitude| over the four slots
    pub max_abs: f32,
}

pub fn amplitudes(degrees: &Degrees, center: Point) -> Result<Amplitudes, Refusal> {
    let slots = degrees.slots();
    for value in slots.iter().copied().chain([center.x, center.y]) {
        if !value.is_finite() {
            return Err(Refusal::NonFinite("degrees/center"));
        }
    }
    let inner = [
        slots[0] * SCALE * center.x,
        slots[1] * SCALE * center.y,
    ];
    let outer = [
        slots[2] * SCALE * center.x,
        slots[3] * SCALE * center.y,
    ];
    let max_abs = inner
        .iter()
        .chain(outer.iter())
        .fold(0.0f32, |acc, value| acc.max(value.abs()));
    Ok(Amplitudes {
        inner,
        outer,
        max_abs,
    })
}

/// Type1010 has no constructor index buffer; the only EBO source is
/// `0x60F2B60` over the 120 A points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EboSource {
    Dynamic,
}

pub const EBO_SOURCE: EboSource = EboSource::Dynamic;

pub fn index_count(triangle_count: i32) -> Result<usize, Refusal> {
    if triangle_count < 0 {
        return Err(Refusal::TriangleOverflow);
    }
    let count = (triangle_count as usize)
        .checked_mul(3)
        .ok_or(Refusal::TriangleOverflow)?;
    if count > u16::MAX as usize {
        return Err(Refusal::TriangleOverflow);
    }
    Ok(count)
}

/// Low uint16 words of each 12-byte int32 triple (`0x60F2C80`).
pub fn pack_triangles(triples: &[[i32; 3]]) -> Vec<u16> {
    let mut out = Vec::with_capacity(triples.len() * 3);
    for triple in triples {
        for value in triple {
            out.push((*value as u32 & 0xffff) as u16);
        }
    }
    out
}

/// Call-site contract: Type1010 always needs live `0x6124030` triples.
pub fn build_indices(triples: Option<&[[i32; 3]]>) -> Result<Vec<u16>, Refusal> {
    let triples = triples.ok_or(Refusal::MissingTriangles)?;
    Ok(pack_triangles(triples))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn degree_ranges_match_part_8_and_9() {
        assert!((map_degree(0.0, Part::Negative).unwrap() + 1.0).abs() < 1e-6);
        assert_eq!(map_degree(0.5, Part::Negative).unwrap(), 0.0);
        assert_eq!(map_degree(1.0, Part::Negative).unwrap(), 0.0);
        assert_eq!(map_degree(0.0, Part::Positive).unwrap(), 0.0);
        assert_eq!(map_degree(1.0, Part::Positive).unwrap(), 1.0);
        assert_eq!(
            map_degree(f32::NAN, Part::Positive).unwrap_err(),
            Refusal::NonFinite("native")
        );
    }

    #[test]
    fn control_slots_are_fixed() {
        assert_eq!(CONTROL_SLOTS[0], (30071, 0));
        assert_eq!(CONTROL_SLOTS[1], (30077, 1));
        assert_eq!(CONTROL_SLOTS[2], (30072, 2));
        assert_eq!(CONTROL_SLOTS[3], (30078, 3));
    }

    #[test]
    fn amplitudes_follow_center_scaled_012() {
        let degrees = Degrees {
            inner_left: 0.8,
            inner_right: -0.6,
            outer_left: 0.4,
            outer_right: -0.2,
        };
        let center = Point::new(0.75, 0.25);
        let amps = amplitudes(&degrees, center).unwrap();
        assert!((amps.inner[0] - 0.8 * SCALE * 0.75).abs() < 1e-7);
        assert!((amps.inner[1] + 0.6 * SCALE * 0.25).abs() < 1e-7);
        assert!((amps.outer[0] - 0.4 * SCALE * 0.75).abs() < 1e-7);
        assert!((amps.outer[1] + 0.2 * SCALE * 0.25).abs() < 1e-7);
        assert!((amps.max_abs - 0.8 * SCALE * 0.75).abs() < 1e-7);
    }

    #[test]
    fn max_abs_covers_all_four_slots() {
        let degrees = Degrees {
            inner_left: 0.1,
            inner_right: 0.2,
            outer_left: -0.9,
            outer_right: 0.3,
        };
        let amps = amplitudes(&degrees, Point::new(1.0, 1.0)).unwrap();
        assert!((amps.max_abs - 0.9 * SCALE).abs() < 1e-7);
    }

    #[test]
    fn non_finite_degrees_refuse() {
        let degrees = Degrees {
            inner_left: f32::INFINITY,
            ..Default::default()
        };
        assert_eq!(
            amplitudes(&degrees, Point::new(1.0, 1.0)).unwrap_err(),
            Refusal::NonFinite("degrees/center")
        );
    }

    #[test]
    fn ebo_is_dynamic_only() {
        assert_eq!(EBO_SOURCE, EboSource::Dynamic);
        assert_eq!(build_indices(None).unwrap_err(), Refusal::MissingTriangles);
        let triples = [[0i32, 1, 2], [2, 1, 119]];
        assert_eq!(build_indices(Some(&triples)).unwrap(), vec![0, 1, 2, 2, 1, 119]);
    }

    #[test]
    fn index_count_is_three_per_triangle() {
        assert_eq!(index_count(0).unwrap(), 0);
        assert_eq!(index_count(168).unwrap(), 504);
        assert_eq!(index_count(-1).unwrap_err(), Refusal::TriangleOverflow);
    }

    #[test]
    fn packing_keeps_low_words() {
        let packed = pack_triangles(&[[0x0001_0002, 0x0003_0004, -1]]);
        assert_eq!(packed, vec![0x0002, 0x0004, 0xffff]);
    }

    #[test]
    fn mesh_layout_is_two_halves_of_60() {
        assert_eq!(VERTEX_COUNT, 2 * HALF_COUNT);
        assert_eq!(VERTEX_COUNT, 120);
    }
}
