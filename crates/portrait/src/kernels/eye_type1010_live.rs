//! Type1010 live-input contract (parts 8/9, PF30071/30077/30072/30078).
//!
//! Shapes and sources are fixed by the pinned PE (`a84519da…`) and the
//! bit-exact `eye_type1010_core` (rawA/rawB 4/4):
//!
//! ```text
//! 0x618FA80  r9d-style vector check: [r8+8]-[r8] >= 0x10  -> 4 degree slots
//! 0x618FB60  rbx = face + 0x924                            -> 414-point array
//! 0x6190610  cluster reads at 0x924+0x120/0xA0/0xC0/0xE0/0x100
//!                         and  0x924+0x1A8/0x128/0x148/0x168/0x188
//! 0x60C4BA0  center = face+0x18AC/0x18B0
//! ```
//!
//! Read order per cluster is `P4, P0, P1, P2, P3`.

use crate::kernels::eye_type1010_core::{Cluster, Point};

/// Same variant set as `crate::Refusal`; swap for the shared enum on merge.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    MissingInput(&'static str),
    ShapeMismatch(&'static str),
    ValueOutOfRange(&'static str),
}

/// `face+0x924` float2 indices, in read order `P4, P0, P1, P2, P3`.
pub const CLUSTER1_ORDER: [usize; 5] = [0x120 / 8, 0xA0 / 8, 0xC0 / 8, 0xE0 / 8, 0x100 / 8];
pub const CLUSTER2_ORDER: [usize; 5] = [0x1A8 / 8, 0x128 / 8, 0x148 / 8, 0x168 / 8, 0x188 / 8];

/// Highest `face+0x924` index Type1010 reads.
pub const FACE924_MIN_POINTS: usize = 54;

pub const PARAM_COUNT: usize = 28;
pub const DEGREE_COUNT: usize = 4;

/// ParamList ranges consumed by the two passes; the rest must still exist.
pub const PASS1_WEIGHTS: core::ops::Range<usize> = 10..19;
pub const PASS2_WEIGHTS_HEAD: core::ops::Range<usize> = 0..5;
pub const PASS2_WEIGHTS_TAIL: core::ops::Range<usize> = 24..28;

/// PF id per degree slot (`0x4C965A9/0x4C965C0` ControlIndex).
pub const DEGREE_SOURCES: [(u32, usize); 4] =
    [(30071, 0), (30077, 1), (30072, 2), (30078, 3)];

pub struct LiveRaw<'a> {
    /// `face+0x924` float2 view; at least 54 entries.
    pub face924: &'a [[f32; 2]],
    /// `face+0x18AC/0x18B0`.
    pub center: [f32; 2],
    /// The part's 28-float `ParamList`.
    pub params: &'a [f32],
    /// Control vector written by `0x4C96360`, at least 4 floats.
    pub degrees: &'a [f32],
}

#[derive(Debug)]
pub struct LiveInputs {
    pub center: Point,
    pub first: Cluster,
    pub second: Cluster,
    pub params: [f32; PARAM_COUNT],
    pub degrees: [f32; DEGREE_COUNT],
}

fn point(value: [f32; 2], field: &'static str) -> Result<Point, Refusal> {
    if !value[0].is_finite() || !value[1].is_finite() {
        return Err(Refusal::ValueOutOfRange(field));
    }
    Ok(Point {
        x: value[0],
        y: value[1],
    })
}

fn cluster(
    face: &[[f32; 2]],
    order: [usize; 5],
    field: &'static str,
) -> Result<Cluster, Refusal> {
    let mut points = [Point { x: 0.0, y: 0.0 }; 5];
    for (slot, &index) in order.iter().enumerate() {
        points[slot] = point(face[index], field)?;
    }
    Ok(Cluster {
        p4: points[0],
        p0: points[1],
        p1: points[2],
        p2: points[3],
        p3: points[4],
    })
}

pub fn validate(raw: &LiveRaw<'_>) -> Result<LiveInputs, Refusal> {
    if raw.face924.len() < FACE924_MIN_POINTS {
        return Err(Refusal::MissingInput("type1010_face924_54points"));
    }
    if raw.params.len() != PARAM_COUNT {
        return Err(Refusal::ShapeMismatch("type1010_paramlist_28"));
    }
    if raw.degrees.len() < DEGREE_COUNT {
        return Err(Refusal::ShapeMismatch("type1010_degree_slots4"));
    }

    for &value in raw.params {
        if !value.is_finite() {
            return Err(Refusal::ValueOutOfRange("type1010_paramlist"));
        }
    }
    for &value in &raw.degrees[..DEGREE_COUNT] {
        if !value.is_finite() || value.abs() > 1.0 {
            return Err(Refusal::ValueOutOfRange("type1010_degrees"));
        }
    }

    let center = point(raw.center, "type1010_center")?;
    let first = cluster(raw.face924, CLUSTER1_ORDER, "type1010_cluster1")?;
    let second = cluster(raw.face924, CLUSTER2_ORDER, "type1010_cluster2")?;

    let mut params = [0.0f32; PARAM_COUNT];
    params.copy_from_slice(raw.params);
    let mut degrees = [0.0f32; DEGREE_COUNT];
    degrees.copy_from_slice(&raw.degrees[..DEGREE_COUNT]);

    Ok(LiveInputs {
        center,
        first,
        second,
        params,
        degrees,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face924() -> Vec<[f32; 2]> {
        (0..54).map(|i| [i as f32, i as f32 * 2.0]).collect()
    }

    #[test]
    fn offsets_match_the_pe_indices() {
        assert_eq!(CLUSTER1_ORDER, [36, 20, 24, 28, 32]);
        assert_eq!(CLUSTER2_ORDER, [53, 37, 41, 45, 49]);
        assert_eq!(FACE924_MIN_POINTS, 54);
    }

    #[test]
    fn cluster_mapping_is_p4_p0_p1_p2_p3() {
        let face = face924();
        let raw = LiveRaw {
            face924: &face,
            center: [0.75, 0.25],
            params: &[0.0; 28],
            degrees: &[0.0; 4],
        };
        let out = validate(&raw).unwrap();
        assert_eq!((out.first.p4.x, out.first.p4.y), (36.0, 72.0));
        assert_eq!((out.first.p0.x, out.first.p0.y), (20.0, 40.0));
        assert_eq!((out.first.p1.x, out.first.p1.y), (24.0, 48.0));
        assert_eq!((out.first.p2.x, out.first.p2.y), (28.0, 56.0));
        assert_eq!((out.first.p3.x, out.first.p3.y), (32.0, 64.0));
        assert_eq!((out.second.p4.x, out.second.p4.y), (53.0, 106.0));
        assert_eq!((out.second.p0.x, out.second.p0.y), (37.0, 74.0));
        assert_eq!((out.second.p3.x, out.second.p3.y), (49.0, 98.0));
    }

    #[test]
    fn paramlist_shape_is_28() {
        let face = face924();
        let raw = LiveRaw {
            face924: &face,
            center: [0.5, 0.5],
            params: &[0.0; 27],
            degrees: &[0.0; 4],
        };
        assert_eq!(
            validate(&raw).unwrap_err(),
            Refusal::ShapeMismatch("type1010_paramlist_28")
        );
    }

    #[test]
    fn degree_vector_needs_four_slots() {
        let face = face924();
        let raw = LiveRaw {
            face924: &face,
            center: [0.5, 0.5],
            params: &[0.0; 28],
            degrees: &[0.0; 3],
        };
        assert_eq!(
            validate(&raw).unwrap_err(),
            Refusal::ShapeMismatch("type1010_degree_slots4")
        );
    }

    #[test]
    fn short_face_array_is_missing_input() {
        let face: Vec<[f32; 2]> = (0..53).map(|i| [i as f32, 0.0]).collect();
        let raw = LiveRaw {
            face924: &face,
            center: [0.5, 0.5],
            params: &[0.0; 28],
            degrees: &[0.0; 4],
        };
        assert_eq!(
            validate(&raw).unwrap_err(),
            Refusal::MissingInput("type1010_face924_54points")
        );
    }

    #[test]
    fn non_finite_fields_refuse() {
        let face = face924();
        let raw = LiveRaw {
            face924: &face,
            center: [f32::NAN, 0.5],
            params: &[0.0; 28],
            degrees: &[0.0; 4],
        };
        assert_eq!(
            validate(&raw).unwrap_err(),
            Refusal::ValueOutOfRange("type1010_center")
        );
        let mut bad = face.clone();
        bad[20][0] = f32::INFINITY;
        let raw = LiveRaw {
            face924: &bad,
            center: [0.5, 0.5],
            params: &[0.0; 28],
            degrees: &[0.0; 4],
        };
        assert_eq!(
            validate(&raw).unwrap_err(),
            Refusal::ValueOutOfRange("type1010_cluster1")
        );
    }

    #[test]
    fn degree_magnitude_is_bounded() {
        let face = face924();
        let raw = LiveRaw {
            face924: &face,
            center: [0.5, 0.5],
            params: &[0.0; 28],
            degrees: &[0.0, 1.5, 0.0, 0.0],
        };
        assert_eq!(
            validate(&raw).unwrap_err(),
            Refusal::ValueOutOfRange("type1010_degrees")
        );
    }

    #[test]
    fn weight_ranges_are_fixed() {
        assert_eq!(PASS1_WEIGHTS, 10..19);
        assert_eq!(PASS2_WEIGHTS_HEAD, 0..5);
        assert_eq!(PASS2_WEIGHTS_TAIL, 24..28);
        assert_eq!(
            DEGREE_SOURCES,
            [(30071, 0), (30077, 1), (30072, 2), (30078, 3)]
        );
    }
}
