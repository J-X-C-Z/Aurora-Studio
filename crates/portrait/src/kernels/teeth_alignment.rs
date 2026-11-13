//! PF30026 teeth alignment geometry, recovered from `pix-worker.exe`.
//!
//! The worker fits a 2x3 affine from four mouth landmarks (`[73,78,83,89]`) to
//! the fixed 1024x768 quad below and gates the frame on the A/B edge planes.
//! Which three of the four points the vendor fits (or whether it least-squares
//! all four) is not proven, so [`Fit`] is an explicit caller choice.

use crate::{Access, EffectMeta, Refusal};

pub const TARGET_QUAD: [[f32; 2]; 4] = [
    [128.0, 352.0],
    [512.0, 224.0],
    [896.0, 352.0],
    [512.0, 480.0],
];

/// Strict `ratio > 0.14` gate; `A+B > 100` after the byte conversion.
pub const EDGE_GATE_PERCENT: f32 = 0.14;
pub const EDGE_SUM_THRESHOLD: u16 = 100;

pub const META: &[EffectMeta] = &[EffectMeta {
    pf: 30026,
    key: "teethAlignment",
    label: "Teeth alignment",
    access: Access::Cpu,
    inputs: &[
        "four mouth landmarks [73,78,83,89]",
        "fit mode (FirstThree | LeastSquares)",
        "edge planes A/B",
        "teeth mask",
        "effect RGB",
    ],
}];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fit {
    /// Exact affine through the first three points; the fourth is unused.
    FirstThree,
    /// Least-squares affine over all four correspondences.
    LeastSquares,
}

fn finite(points: &[[f32; 2]; 4]) -> bool {
    points.iter().flatten().all(|value| value.is_finite())
}

/// `x' = a*u + b*v + tx`, `y' = c*u + d*v + ty`.
pub fn affine_from_points(
    source: &[[f32; 2]; 4],
    target: &[[f32; 2]; 4],
    fit: Fit,
) -> Result<[f32; 6], Refusal> {
    if !finite(source) || !finite(target) {
        return Err(Refusal::ValueOutOfRange("teeth landmarks"));
    }
    match fit {
        Fit::FirstThree => {
            let (u0, v0) = (source[0][0], source[0][1]);
            let du1 = source[1][0] - u0;
            let dv1 = source[1][1] - v0;
            let du2 = source[2][0] - u0;
            let dv2 = source[2][1] - v0;
            let det = du1 * dv2 - du2 * dv1;
            if det.abs() < 1e-6 {
                return Err(Refusal::ValueOutOfRange("teeth landmarks"));
            }
            let dx1 = target[1][0] - target[0][0];
            let dx2 = target[2][0] - target[0][0];
            let dy1 = target[1][1] - target[0][1];
            let dy2 = target[2][1] - target[0][1];
            let a = (dx1 * dv2 - dx2 * dv1) / det;
            let b = (du1 * dx2 - du2 * dx1) / det;
            let c = (dy1 * dv2 - dy2 * dv1) / det;
            let d = (du1 * dy2 - du2 * dy1) / det;
            Ok([a, b, target[0][0] - a * u0 - b * v0,
                c, d, target[0][1] - c * u0 - d * v0])
        }
        Fit::LeastSquares => least_squares(source, target),
    }
}

/// Normal equations for the two independent rows of the affine:
/// `sum [[u,v,1]^T [u,v,1]] w = sum [[u,v,1]]^T x`.
fn least_squares(
    source: &[[f32; 2]; 4],
    target: &[[f32; 2]; 4],
) -> Result<[f32; 6], Refusal> {
    let mut normal = [[0.0f32; 6]; 6];
    let mut rhs = [[0.0f32; 2]; 6];
    for (point, mapped) in source.iter().zip(target) {
        let row = [point[0], point[1], 1.0];
        for i in 0..3 {
            for j in 0..3 {
                normal[i][j] += row[i] * row[j];
                normal[i + 3][j + 3] += row[i] * row[j];
            }
            rhs[i][0] += row[i] * mapped[0];
            rhs[i][1] += row[i] * mapped[1];
            rhs[i + 3][0] += row[i] * mapped[0];
        }
    }
    // Both blocks are identical, so solve once for x and once for y.
    let mut matrix = [[0.0f32; 7]; 3];
    let mut second = [[0.0f32; 7]; 3];
    for i in 0..3 {
        for j in 0..3 {
            matrix[i][j] = normal[i][j];
            second[i][j] = normal[i][j];
        }
        matrix[i][3] = rhs[i][0];
        second[i][3] = rhs[i][1];
    }
    let x = solve3(matrix)?;
    let y = solve3(second)?;
    Ok([x[0], x[1], x[2], y[0], y[1], y[2]])
}

fn solve3(mut m: [[f32; 7]; 3]) -> Result<[f32; 3], Refusal> {
    for column in 0..3 {
        let pivot = (column..3)
            .max_by(|a, b| m[*a][column].abs().total_cmp(&m[*b][column].abs()))
            .unwrap();
        if m[pivot][column].abs() < 1e-9 {
            return Err(Refusal::ValueOutOfRange("teeth landmarks"));
        }
        m.swap(column, pivot);
        for row in 0..3 {
            if row != column {
                let factor = m[row][column] / m[column][column];
                for k in column..4 {
                    m[row][k] -= factor * m[column][k];
                }
            }
        }
    }
    Ok([m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]])
}

pub fn map_point(affine: &[f32; 6], point: [f32; 2]) -> [f32; 2] {
    [
        affine[0] * point[0] + affine[1] * point[1] + affine[2],
        affine[3] * point[0] + affine[4] * point[1] + affine[5],
    ]
}

fn to_byte(value: f32) -> u16 {
    (value * 255.0).round_ties_even().clamp(0.0, 255.0) as u16
}

/// `0.5`-free vendor gate: each plane to a byte, `A+B > 100`, percentage of the
/// fixed ROI. The edge planes are the worker's two `CV_32FC1` masks.
pub fn edge_ratio(a: &[f32], b: &[f32], width: u32, height: u32) -> Result<f32, Refusal> {
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("edge planes"))?;
    if width == 0 || height == 0 || a.len() != pixels || b.len() != pixels {
        return Err(Refusal::ShapeMismatch("edge planes"));
    }
    if a.iter().chain(b).any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("edge planes"));
    }
    let valid = a
        .iter()
        .zip(b)
        .filter(|(left, right)| to_byte(**left) + to_byte(**right) > EDGE_SUM_THRESHOLD)
        .count();
    Ok(valid as f32 / pixels as f32 * 100.0)
}

pub fn edge_gate(a: &[f32], b: &[f32], width: u32, height: u32) -> Result<bool, Refusal> {
    Ok(edge_ratio(a, b, width, height)? > EDGE_GATE_PERCENT)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad() -> [[f32; 2]; 4] {
        TARGET_QUAD
    }

    #[test]
    fn three_points_reproduce_an_exact_affine() {
        let affine = [1.5, -0.25, 7.0, 0.5, 2.0, -3.0];
        let source = [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]];
        let target = source.map(|point| map_point(&affine, point));
        let solved = affine_from_points(&source, &target, Fit::FirstThree).unwrap();
        for point in source {
            let want = map_point(&affine, point);
            let got = map_point(&solved, point);
            assert!((want[0] - got[0]).abs() < 1e-4 && (want[1] - got[1]).abs() < 1e-4);
        }
    }

    #[test]
    fn least_squares_recovers_a_noisy_quad() {
        let affine = [1.0, 0.1, 0.0, 0.1, 1.0, 0.0];
        let source = [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0], [4.0, 3.0]];
        let target = source.map(|point| map_point(&affine, point));
        let solved = affine_from_points(&source, &target, Fit::LeastSquares).unwrap();
        for point in source {
            let want = map_point(&affine, point);
            let got = map_point(&solved, point);
            assert!((want[0] - got[0]).abs() < 1e-3 && (want[1] - got[1]).abs() < 1e-3);
        }
    }

    #[test]
    fn the_target_quad_maps_onto_itself() {
        let solved = affine_from_points(&quad(), &quad(), Fit::LeastSquares).unwrap();
        for point in quad() {
            let got = map_point(&solved, point);
            assert!((point[0] - got[0]).abs() < 1e-3 && (point[1] - got[1]).abs() < 1e-3);
        }
    }

    #[test]
    fn degenerate_and_non_finite_landmarks_are_refused() {
        let flat = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        assert_eq!(
            affine_from_points(&flat, &quad(), Fit::FirstThree),
            Err(Refusal::ValueOutOfRange("teeth landmarks"))
        );
        let mut nan = quad();
        nan[3][1] = f32::NAN;
        assert_eq!(
            affine_from_points(&nan, &quad(), Fit::LeastSquares),
            Err(Refusal::ValueOutOfRange("teeth landmarks"))
        );
    }

    #[test]
    fn edge_gate_uses_bytes_and_a_strict_ratio() {
        let a = [0.5f32, 0.0, 0.0, 0.0];
        let b = [0.5f32, 0.0, 0.0, 0.0];
        assert_eq!(edge_ratio(&a, &b, 2, 2).unwrap(), 25.0);
        assert!(edge_gate(&a, &b, 2, 2).unwrap());
        let quiet = [0.1f32; 4];
        assert_eq!(edge_ratio(&quiet, &quiet, 2, 2).unwrap(), 0.0);
        assert!(!edge_gate(&quiet, &quiet, 2, 2).unwrap());
        assert_eq!(
            edge_ratio(&a, &b, 3, 3),
            Err(Refusal::ShapeMismatch("edge planes"))
        );
    }
}
