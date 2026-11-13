//! Per-pixel interpolation of a two-component vertex attribute over a mesh.
//!
//! PF8033 needs `dx/dy` at every output pixel while its model only produces
//! one displacement per vertex. This is the attribute rasterizer between the
//! two. Geometry follows the recovered mesh convention used by
//! `leg_refine`/`body_geometry`: pixel centres, signed `cross` area,
//! `|area| < 1e-8` skips, edge tolerance `-1e-6`, and triangle-order
//! last-writer coverage.
//!
//! Coverage has no z-buffer: when two triangles cover one pixel the later
//! triangle wins. The write count exposes those overlaps to the caller.
//! Which screen domain (mean or morphed vertices) and which displacement
//! handedness the vendor uses is not established here; the caller supplies
//! `dest` and `attr` explicitly.

use crate::Refusal;

#[derive(Clone, Debug, PartialEq)]
pub struct DenseAttribute {
    pub values: Vec<[f32; 2]>,
    /// Number of triangles that wrote each pixel; `0` = uncovered.
    pub writes: Vec<u16>,
}

pub fn rasterize(
    width: u32,
    height: u32,
    dest: &[[f32; 2]],
    attr: &[[f32; 2]],
    faces: &[u32],
) -> Result<DenseAttribute, Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("mesh attribute size"));
    }
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("mesh attribute size"))?;
    if dest.is_empty() || dest.len() != attr.len() {
        return Err(Refusal::ShapeMismatch("mesh attribute vertices"));
    }
    if faces.is_empty() || faces.len() % 3 != 0 {
        return Err(Refusal::ShapeMismatch("mesh attribute faces"));
    }
    if dest
        .iter()
        .chain(attr)
        .flat_map(|value| value.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("mesh attribute vertex"));
    }
    if faces.iter().any(|index| *index as usize >= dest.len()) {
        return Err(Refusal::ValueOutOfRange("mesh attribute face index"));
    }

    let width_usize = width as usize;
    let height_usize = height as usize;
    let mut values = vec![[0.0f32; 2]; pixels];
    let mut writes = vec![0u16; pixels];

    for triangle in faces.chunks_exact(3) {
        let index = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        let (p0, p1, p2) = (dest[index[0]], dest[index[1]], dest[index[2]]);
        let area = (p1[0] - p0[0]) * (p2[1] - p0[1])
            - (p1[1] - p0[1]) * (p2[0] - p0[0]);
        if area.abs() < 1e-8 {
            continue;
        }
        let inverse = 1.0 / area;
        let min_x = 0.max(p0[0].min(p1[0]).min(p2[0]).floor() as i64) as usize;
        let max_x = (width_usize as i64 - 1)
            .min(p0[0].max(p1[0]).max(p2[0]).ceil() as i64)
            .max(0) as usize;
        let min_y = 0.max(p0[1].min(p1[1]).min(p2[1]).floor() as i64) as usize;
        let max_y = (height_usize as i64 - 1)
            .min(p0[1].max(p1[1]).max(p2[1]).ceil() as i64)
            .max(0) as usize;
        if min_x > max_x || min_y > max_y {
            continue;
        }

        for py in min_y..=max_y {
            let sy = py as f32 + 0.5;
            for px in min_x..=max_x {
                let sx = px as f32 + 0.5;
                let mut l0 = ((p1[0] - sx) * (p2[1] - sy)
                    - (p1[1] - sy) * (p2[0] - sx))
                    * inverse;
                let mut l1 = ((p2[0] - sx) * (p0[1] - sy)
                    - (p2[1] - sy) * (p0[0] - sx))
                    * inverse;
                let mut l2 = 1.0 - l0 - l1;
                if l0 < -1e-6 || l1 < -1e-6 || l2 < -1e-6 {
                    continue;
                }
                let sum = l0 + l1 + l2;
                if sum <= 0.0 {
                    continue;
                }
                l0 = l0.max(0.0) / sum;
                l1 = l1.max(0.0) / sum;
                l2 = l2.max(0.0) / sum;
                let offset = py * width_usize + px;
                values[offset] = [
                    l0 * attr[index[0]][0]
                        + l1 * attr[index[1]][0]
                        + l2 * attr[index[2]][0],
                    l0 * attr[index[0]][1]
                        + l1 * attr[index[1]][1]
                        + l2 * attr[index[2]][1],
                ];
                writes[offset] = writes[offset].saturating_add(1);
            }
        }
    }

    Ok(DenseAttribute { values, writes })
}

#[cfg(test)]
mod tests {
    use super::*;


    #[cfg(feature = "external-fixtures")]


    fn decode_faces() -> Vec<u32> {
        let bytes = crate::external_fixtures::read("smile/triangles.u32le");
        assert_eq!(bytes.len(), 51228);
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[cfg(feature = "external-fixtures")]

    fn decode_mean() -> Vec<[f32; 2]> {
        let bytes = crate::external_fixtures::read("smile/mean.f32le");
        assert_eq!(bytes.len(), 26472);
        bytes.chunks_exact(12)
            .map(|c| {
                [
                    f32::from_le_bytes([c[0], c[1], c[2], c[3]]),
                    f32::from_le_bytes([c[4], c[5], c[6], c[7]]),
                ]
            })
            .collect()
    }

    #[test]
    fn linear_attribute_matches_the_plane() {
        let dest = [[10.0, 10.0], [30.0, 10.0], [10.0, 20.0]];
        let attr = [[1.0 + 0.25 * 10.0, -2.0 + 0.5 * 10.0],
            [1.0 + 0.25 * 30.0, -2.0 + 0.5 * 10.0],
            [1.0 + 0.25 * 10.0, -2.0 + 0.5 * 20.0]];
        let out = rasterize(40, 24, &dest, &attr, &[0, 1, 2]).unwrap();
        for py in 10..20 {
            for px in 10..30 {
                if out.writes[py * 40 + px] == 0 {
                    continue;
                }
                let value = out.values[py * 40 + px];
                assert!((value[0] - (1.0 + 0.25 * (px as f32 + 0.5))).abs() < 1e-5);
                assert!((value[1] - (-2.0 + 0.5 * (py as f32 + 0.5))).abs() < 1e-5);
            }
        }
        assert!(out.writes.iter().any(|writes| *writes > 0));
    }

    #[test]
    fn later_triangle_wins_and_write_count_reports_overlap() {
        let dest = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]];
        let first = [[1.0, 1.0]; 3];
        let second = [[2.0, 2.0]; 3];
        let one = rasterize(10, 10, &dest, &first, &[0, 1, 2]).unwrap();
        let twice =
            rasterize(10, 10, &dest, &second, &[0, 1, 2, 0, 1, 2]).unwrap();
        assert!(twice
            .writes
            .iter()
            .zip(&one.writes)
            .all(|(twice, once)| *twice == once.saturating_mul(2)));
        assert!(twice
            .values
            .iter()
            .zip(&twice.writes)
            .all(|(value, writes)| {
                *writes == 0
                    || ((value[0] - 2.0).abs() < 1e-6 && (value[1] - 2.0).abs() < 1e-6)
            }));
        assert!(one.writes.iter().all(|writes| *writes <= 1));
    }

    #[test]
    #[cfg(feature = "external-fixtures")]
    fn real_mesh_rasterizes_with_zero_displacement() {
        let mean = decode_mean();
        let faces = decode_faces();
        assert_eq!(mean.len(), 2206);
        assert_eq!(faces.len(), 4269 * 3);
        let zero = vec![[0.0f32; 2]; mean.len()];
        // Simple explicit screen placement; the mesh object space is ±80.
        let dest: Vec<[f32; 2]> = mean
            .iter()
            .map(|point| [(point[0] + 80.0) * 0.5, (80.0 - point[1]) * 0.3])
            .collect();
        let out = rasterize(96, 64, &dest, &zero, &faces).unwrap();
        assert!(out.writes.iter().any(|writes| *writes > 0));
        assert!(out.writes.iter().filter(|writes| **writes > 1).count() > 0);
        assert!(out.values.iter().all(|value| *value == [0.0, 0.0]));
    }

    #[test]
    fn bad_shapes_are_refused() {
        let dest = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]];
        let attr = [[0.0, 0.0]; 3];
        assert_eq!(
            rasterize(0, 4, &dest, &attr, &[0, 1, 2]),
            Err(Refusal::ShapeMismatch("mesh attribute size"))
        );
        assert_eq!(
            rasterize(4, 4, &dest, &attr, &[0, 1]),
            Err(Refusal::ShapeMismatch("mesh attribute faces"))
        );
        assert_eq!(
            rasterize(4, 4, &dest, &attr, &[0, 1, 3]),
            Err(Refusal::ValueOutOfRange("mesh attribute face index"))
        );
    }
}
