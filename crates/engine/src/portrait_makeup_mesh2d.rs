//! Build Face2D draw bytes and material-UV bytes for the B3 makeup rows from
//! the same-job 194-point landmark array, without an external 3DMM.
//!
//! `points194` is `primary142 ++ secondary52` (R76 s01). The primary-142 roles
//! below are read from the EXE's own static template (RVA `0x91bcd30`,
//! `primary142_template.json`), cross-checked per face in
//! `04_reference/legacy-root/逆向工程/像素蛋糕/analysis/v2-plane-semantics-v1`:
//!
//! ```text
//! 0..=8      left brow ring      9..=17     right brow ring
//! 18..=35    left eye ring       36..=53    right eye ring
//! 54..=71    nose cluster        73..=95    outer lip ring (closed)
//! 96..=106   inner lip ring      109..=141  jaw/chin oval (closed)
//! ```
//!
//! The secondary 52 points have no proven role in any named B3 local region, so
//! they are used only by the whole-face silhouette hull and are never injected
//! into a brow/eye/lip/cheek mapping that is not established.

use std::sync::Arc;

use aurora_portrait::Refusal;

use crate::portrait_makeup_bundle::{
    B3_MAKEUP_ROWS, MakeupRowProfile, OwnedFace2dDrawBytes, OwnedFace2dUvBytes,
};

const POINTS194: usize = 194;
const VERTEX_STRIDE: usize = 8;

pub(crate) const LANDMARKS_INVALID: &str = "makeup mesh2d landmarks";
pub(crate) const FRAME_INVALID: &str = "makeup mesh2d frame size";
pub(crate) const REGION_UNMAPPED: &str = "makeup mesh2d region";

// Ordered closed rings from the static 142-point template. The order is the
// template order: every ring below is already a perimeter walk.
const BROW_LEFT: [usize; 9] = [0, 1, 2, 3, 4, 5, 6, 7, 8];
const BROW_RIGHT: [usize; 9] = [9, 10, 11, 12, 13, 14, 15, 16, 17];
const EYE_LEFT: [usize; 18] = [
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
];
const EYE_RIGHT: [usize; 18] = [
    36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53,
];
const LIP_OUTER: [usize; 23] = [
    73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
];
const LIP_INNER: [usize; 11] = [96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106];

/// Build 2D draw bytes for one makeup row from 194 landmarks in source pixels.
pub(crate) fn build_face2d_draw_bytes(
    points194: &[[f32; 2]],
    frame_size: [u32; 2],
    profile: &MakeupRowProfile,
) -> Result<OwnedFace2dDrawBytes, Refusal> {
    validate_contract(points194, frame_size)?;
    let (vertices, indices) = region_geometry(points194, profile)?;
    if indices.is_empty() {
        return Err(Refusal::ShapeMismatch(REGION_UNMAPPED));
    }

    let mut index_bytes = Vec::with_capacity(indices.len() * 2);
    for index in &indices {
        index_bytes.extend_from_slice(&index.to_le_bytes());
    }

    Ok(OwnedFace2dDrawBytes {
        vertices: encode_positions(&vertices),
        vertex_stride: VERTEX_STRIDE,
        position_offset: 0,
        indices_u16: Arc::from(index_bytes.into_boxed_slice()),
        draw_first_index: 0,
        draw_index_count: indices.len(),
    })
}

/// Build a UV plane covering the row's local region (material sampling coords).
///
/// The plane is the region bounding box normalized to `[0, 1]`; the vertex
/// order is identical to [`build_face2d_draw_bytes`] for the same profile, so
/// the two streams can be uploaded side by side.
pub(crate) fn build_region_uv(
    points194: &[[f32; 2]],
    frame_size: [u32; 2],
    profile: &MakeupRowProfile,
) -> Result<OwnedFace2dUvBytes, Refusal> {
    validate_contract(points194, frame_size)?;
    let (vertices, _) = region_geometry(points194, profile)?;
    if vertices.is_empty() {
        return Err(Refusal::ShapeMismatch(REGION_UNMAPPED));
    }

    let mut minimum = [f32::INFINITY; 2];
    let mut maximum = [f32::NEG_INFINITY; 2];
    for point in &vertices {
        for axis in 0..2 {
            minimum[axis] = minimum[axis].min(point[axis]);
            maximum[axis] = maximum[axis].max(point[axis]);
        }
    }
    let span = [maximum[0] - minimum[0], maximum[1] - minimum[1]];

    let mut bytes = Vec::with_capacity(vertices.len() * VERTEX_STRIDE);
    for point in &vertices {
        let u = normalize_uv(point[0], minimum[0], span[0]);
        let v = normalize_uv(point[1], minimum[1], span[1]);
        bytes.extend_from_slice(&u.to_le_bytes());
        bytes.extend_from_slice(&v.to_le_bytes());
    }

    Ok(OwnedFace2dUvBytes {
        vertices: Arc::from(bytes.into_boxed_slice()),
        vertex_stride: VERTEX_STRIDE,
        offset: 0,
    })
}

fn validate_contract(points194: &[[f32; 2]], frame_size: [u32; 2]) -> Result<(), Refusal> {
    if points194.len() != POINTS194 {
        return Err(Refusal::ShapeMismatch(LANDMARKS_INVALID));
    }
    if frame_size[0] == 0 || frame_size[1] == 0 {
        return Err(Refusal::ShapeMismatch(FRAME_INVALID));
    }
    if points194
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange(LANDMARKS_INVALID));
    }
    Ok(())
}

/// Resolve the row family from the profile's recovered mask semantic and emit
/// its vertex stream plus local triangle indices.
fn region_geometry(
    points194: &[[f32; 2]],
    profile: &MakeupRowProfile,
) -> Result<(Vec<[f32; 2]>, Vec<u16>), Refusal> {
    if !B3_MAKEUP_ROWS.contains(&profile.pf) {
        return Err(Refusal::UnknownEffect(profile.pf));
    }
    let semantic = profile.mask_semantic;
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    if semantic.contains("EyeBrow") {
        push_ring(points194, &BROW_LEFT, &mut vertices, &mut indices)?;
        push_ring(points194, &BROW_RIGHT, &mut vertices, &mut indices)?;
    } else if semantic.contains("EyeLash")
        || semantic.contains("EyeMask")
        || semantic.contains("EyePupil")
    {
        // Eye colour, eyeliner/eyeshadow and eyelash all draw the eye rings.
        push_ring(points194, &EYE_LEFT, &mut vertices, &mut indices)?;
        push_ring(points194, &EYE_RIGHT, &mut vertices, &mut indices)?;
    } else if semantic.contains("Lip") {
        push_ring(points194, &LIP_OUTER, &mut vertices, &mut indices)?;
        push_ring(points194, &LIP_INNER, &mut vertices, &mut indices)?;
    } else if semantic.contains("Blusher") || semantic.contains("blusher") {
        // A cheek patch is supported by the lower eye ring and the upper half
        // of the jaw oval on the same side; take the convex perimeter so the
        // two clusters become one closed ring.
        let mut left = EYE_LEFT.to_vec();
        left.extend(109..=121);
        push_hull(points194, &left, &mut vertices, &mut indices)?;

        let mut right = EYE_RIGHT.to_vec();
        right.extend(129..=141);
        push_hull(points194, &right, &mut vertices, &mut indices)?;
    } else if semantic.contains("face-parsing") || semantic.contains("MASK_RGBA") {
        // Whole-face owner masks: the full 194-point silhouette hull. The
        // secondary 52 points cover the upper face and add the forehead edge
        // that the primary-142 template does not provide.
        let all: Vec<usize> = (0..POINTS194).collect();
        push_hull(points194, &all, &mut vertices, &mut indices)?;
    } else {
        return Err(Refusal::MissingInput(REGION_UNMAPPED));
    }

    if indices.is_empty() {
        return Err(Refusal::ShapeMismatch(REGION_UNMAPPED));
    }
    Ok((vertices, indices))
}

/// Append one closed ring and fan it from the ring's first vertex.
fn push_ring(
    points194: &[[f32; 2]],
    ring: &[usize],
    vertices: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u16>,
) -> Result<(), Refusal> {
    if ring.len() < 3 {
        return Err(Refusal::ShapeMismatch(REGION_UNMAPPED));
    }
    if vertices.len() + ring.len() > u16::MAX as usize {
        return Err(Refusal::ValueOutOfRange(REGION_UNMAPPED));
    }
    let base = vertices.len() as u16;
    for &landmark in ring {
        vertices.push(
            *points194
                .get(landmark)
                .ok_or(Refusal::ValueOutOfRange(LANDMARKS_INVALID))?,
        );
    }
    let count = ring.len() as u16;
    for step in 1..count - 1 {
        indices.extend_from_slice(&[base, base + step, base + step + 1]);
    }
    Ok(())
}

/// Append the convex perimeter of a landmark subset and fan it.
fn push_hull(
    points194: &[[f32; 2]],
    landmarks: &[usize],
    vertices: &mut Vec<[f32; 2]>,
    indices: &mut Vec<u16>,
) -> Result<(), Refusal> {
    if landmarks.len() < 3 {
        return Err(Refusal::ShapeMismatch(REGION_UNMAPPED));
    }
    let selected = landmarks
        .iter()
        .map(|&landmark| {
            points194
                .get(landmark)
                .copied()
                .ok_or(Refusal::ValueOutOfRange(LANDMARKS_INVALID))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let hull = convex_hull(&selected);
    if hull.len() < 3 {
        return Err(Refusal::ShapeMismatch(REGION_UNMAPPED));
    }
    if vertices.len() + hull.len() > u16::MAX as usize {
        return Err(Refusal::ValueOutOfRange(REGION_UNMAPPED));
    }
    let base = vertices.len() as u16;
    for &index in &hull {
        vertices.push(selected[index]);
    }
    let count = hull.len() as u16;
    for step in 1..count - 1 {
        indices.extend_from_slice(&[base, base + step, base + step + 1]);
    }
    Ok(())
}

/// Andrew monotone chain, counter-clockwise, deterministic tie-break by index.
fn convex_hull(points: &[[f32; 2]]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&a, &b| {
        points[a][0]
            .total_cmp(&points[b][0])
            .then_with(|| points[a][1].total_cmp(&points[b][1]))
            .then_with(|| a.cmp(&b))
    });

    let mut lower = Vec::with_capacity(order.len());
    for &index in &order {
        while lower.len() >= 2
            && cross(
                points[lower[lower.len() - 2]],
                points[lower[lower.len() - 1]],
                points[index],
            ) <= 0.0
        {
            lower.pop();
        }
        lower.push(index);
    }

    let mut upper = Vec::with_capacity(order.len());
    for &index in order.iter().rev() {
        while upper.len() >= 2
            && cross(
                points[upper[upper.len() - 2]],
                points[upper[upper.len() - 1]],
                points[index],
            ) <= 0.0
        {
            upper.pop();
        }
        upper.push(index);
    }

    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

fn cross(origin: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f64 {
    ((a[0] - origin[0]) as f64) * ((b[1] - origin[1]) as f64)
        - ((a[1] - origin[1]) as f64) * ((b[0] - origin[0]) as f64)
}

fn normalize_uv(value: f32, minimum: f32, span: f32) -> f32 {
    if span > 0.0 {
        ((value - minimum) / span).clamp(0.0, 1.0)
    } else {
        0.5
    }
}

fn encode_positions(vertices: &[[f32; 2]]) -> Arc<[u8]> {
    let mut bytes = Vec::with_capacity(vertices.len() * VERTEX_STRIDE);
    for point in vertices {
        bytes.extend_from_slice(&point[0].to_le_bytes());
        bytes.extend_from_slice(&point[1].to_le_bytes());
    }
    Arc::from(bytes.into_boxed_slice())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::portrait_makeup_bundle::{row_profile, row_profile_for_effect};

    /// A plausible face-sized ellipse ring set. The structural contract does not
    /// depend on the landmark values, only on their count and finiteness.
    fn synthetic194() -> Vec<[f32; 2]> {
        let tau = std::f32::consts::TAU;
        (0..POINTS194)
            .map(|index| {
                let angle = index as f32 / POINTS194 as f32 * tau;
                let radius = 180.0 + (index % 7) as f32 * 3.0;
                [
                    256.0 + radius * angle.cos(),
                    256.0 + radius * angle.sin(),
                ]
            })
            .collect()
    }

    fn u16s(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect()
    }

    fn assert_stream(points: &[[f32; 2]], profile: &MakeupRowProfile) {
        let draw = build_face2d_draw_bytes(points, [512, 512], profile).unwrap();
        assert_eq!(draw.vertex_stride, 8);
        assert_eq!(draw.position_offset, 0);
        assert_eq!(draw.draw_first_index, 0);
        assert!(!draw.vertices.is_empty());
        assert_eq!(draw.vertices.len() % 8, 0);
        let vertex_count = draw.vertices.len() / 8;
        for point in draw.vertices.chunks_exact(8) {
            let x = f32::from_le_bytes(point[0..4].try_into().unwrap());
            let y = f32::from_le_bytes(point[4..8].try_into().unwrap());
            assert!(x.is_finite() && y.is_finite());
        }

        let indices = u16s(&draw.indices_u16);
        assert!(!indices.is_empty());
        assert_eq!(draw.draw_index_count, indices.len());
        assert_eq!(draw.draw_index_count % 3, 0);
        assert!(indices.iter().all(|&index| (index as usize) < vertex_count));

        let uv = build_region_uv(points, [512, 512], profile).unwrap();
        assert_eq!(uv.vertex_stride, 8);
        assert_eq!(uv.offset, 0);
        assert_eq!(uv.vertices.len(), draw.vertices.len());
        for pair in uv.vertices.chunks_exact(8) {
            let u = f32::from_le_bytes(pair[0..4].try_into().unwrap());
            let v = f32::from_le_bytes(pair[4..8].try_into().unwrap());
            assert!((0.0..=1.0).contains(&u));
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn every_b3_row_builds_a_non_empty_stream() {
        let points = synthetic194();
        for pf in B3_MAKEUP_ROWS {
            assert_stream(&points, &row_profile(pf).unwrap());
        }
        assert_stream(
            &points,
            &row_profile_for_effect(115, "BlusherImageMakeup").unwrap(),
        );
        assert_stream(&points, &row_profile_for_effect(107, "LipColor").unwrap());
        assert_stream(
            &points,
            &row_profile_for_effect(107, "LipStickImageMakeup").unwrap(),
        );
    }

    #[test]
    fn ring_topology_is_stable() {
        let points = synthetic194();

        let brow = build_face2d_draw_bytes(&points, [512, 512], &row_profile(100).unwrap())
            .unwrap();
        assert_eq!(brow.vertices.len() / 8, 18);
        assert_eq!(brow.draw_index_count, 14 * 3);

        let eye = build_face2d_draw_bytes(&points, [512, 512], &row_profile(103).unwrap())
            .unwrap();
        assert_eq!(eye.vertices.len() / 8, 36);
        assert_eq!(eye.draw_index_count, 32 * 3);

        let lips = build_face2d_draw_bytes(&points, [512, 512], &row_profile(107).unwrap())
            .unwrap();
        assert_eq!(lips.vertices.len() / 8, 34);
        assert_eq!(lips.draw_index_count, 30 * 3);
    }

    #[test]
    fn bad_inputs_are_refused_by_name() {
        let points = synthetic194();
        let brow = row_profile(100).unwrap();

        assert_eq!(
            build_face2d_draw_bytes(&points[..193], [512, 512], &brow),
            Err(Refusal::ShapeMismatch(LANDMARKS_INVALID))
        );

        let mut non_finite = points.clone();
        non_finite[7][0] = f32::NAN;
        assert_eq!(
            build_face2d_draw_bytes(&non_finite, [512, 512], &brow),
            Err(Refusal::ValueOutOfRange(LANDMARKS_INVALID))
        );

        assert_eq!(
            build_region_uv(&points, [0, 512], &row_profile(8).unwrap()),
            Err(Refusal::ShapeMismatch(FRAME_INVALID))
        );

        let mut unknown = row_profile(8).unwrap();
        unknown.mask_semantic = "unmapped semantic";
        assert_eq!(
            build_face2d_draw_bytes(&points, [512, 512], &unknown),
            Err(Refusal::MissingInput(REGION_UNMAPPED))
        );
    }
}
