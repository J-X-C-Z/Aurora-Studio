use std::path::Path;

use anyhow::{Context, Result, bail};

use super::{SlimProducerInputs, sha256_32};
use crate::aimask::Runtime;
use aurora_portrait::{Image, Refusal};

pub const FACE_LANDMARK_52_RESOURCE: u32 = 2_722_162_333;
pub const FACE_LANDMARK_52_MODEL_SHA256: [u8; 32] = [
    0xff, 0xc0, 0x9d, 0x53, 0xff, 0xc9, 0x07, 0x39, 0x65, 0xcc, 0x3c, 0x50, 0x70, 0xe7, 0x3d, 0x0c,
    0xc7, 0x4a, 0xc0, 0x4c, 0x00, 0x81, 0x3a, 0x1a, 0xf2, 0x53, 0xb9, 0x82, 0x19, 0x8b, 0x97, 0x52,
];

const SIDE: usize = 128;
const POINTS142: usize = 142;
const POINTS52: usize = 52;
const POINTS194: usize = 194;

/// The 33 points the recovered 52-point producer uses to fit its crop.
const TEMPLATE33: [[f32; 2]; 33] = [
    [34.023040771484375, 65.64735412597656],
    [34.20806121826172, 69.64289855957031],
    [34.30506896972656, 73.63529968261719],
    [34.434852600097656, 77.63062286376953],
    [34.80542755126953, 81.63151550292969],
    [35.516387939453125, 85.58230590820312],
    [36.59123992919922, 89.43962097167969],
    [38.08436584472656, 93.14198303222656],
    [39.99932098388672, 96.63400268554688],
    [42.26972198486328, 99.88936614990234],
    [44.82341003417969, 102.92440795898438],
    [47.610008239746094, 105.7530517578125],
    [50.52191925048828, 108.41679382324219],
    [53.497127532958984, 110.94340515136719],
    [56.64358901977539, 113.14938354492188],
    [60.171165466308594, 114.67124938964844],
    [64.00001525878906, 115.19999694824219],
    [67.82886505126953, 114.67137145996094],
    [71.35639953613281, 113.1494140625],
    [74.5029525756836, 110.94336700439453],
    [77.47804260253906, 108.41691589355469],
    [80.3900146484375, 105.75299835205078],
    [83.17655181884766, 102.92437744140625],
    [85.73039245605469, 99.88937377929688],
    [88.00067901611328, 96.63392639160156],
    [89.91583251953125, 93.14212036132812],
    [91.40876770019531, 89.4396743774414],
    [92.48374938964844, 85.58235168457031],
    [93.19445037841797, 81.63154602050781],
    [93.56533813476562, 77.630615234375],
    [93.695068359375, 73.63528442382812],
    [93.7920913696289, 69.64287567138672],
    [93.97689819335938, 65.64733123779297],
];

/// Real inputs and outputs of the recovered 52-point landmark producer.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceLandmarkProducerInputs {
    pub resource_id: u32,
    pub provider: String,
    pub model_sha256: [u8; 32],
    pub source_sha256: [u8; 32],
    pub source_size: [u32; 2],
    pub points142: Vec<[f32; 2]>,
    pub crop_bgr: Vec<f32>,
    pub crop_sha256: [u8; 32],
    pub raw52: Vec<[f32; 2]>,
    pub points52: Vec<[f32; 2]>,
    pub points194: Vec<[f32; 2]>,
    pub fit_pre: [[f32; 3]; 2],
    pub fit_inverse: [[f32; 3]; 2],
}

impl FaceLandmarkProducerInputs {
    /// Runs the authorized 142 -> 52 -> 194 chain from a real source image.
    pub fn produce(model: &Path, source: &Image, points142: Vec<[f32; 2]>) -> Result<Self> {
        source
            .validate()
            .map_err(|error| anyhow::anyhow!("invalid landmark source image: {error:?}"))?;
        if points142.len() != POINTS142
            || points142.iter().flatten().any(|value| !value.is_finite())
        {
            bail!("expected 142 finite source landmarks");
        }

        let model_bytes = std::fs::read(model)
            .with_context(|| format!("reading landmark model {}", model.display()))?;
        let model_sha256: [u8; 32] = sha256_32(&model_bytes);
        if model_sha256 != FACE_LANDMARK_52_MODEL_SHA256 {
            bail!(
                "landmark model identity mismatch: expected {}, got {}",
                hex(&FACE_LANDMARK_52_MODEL_SHA256),
                hex(&model_sha256)
            );
        }

        let runtime = Runtime::get().context("ONNX Runtime is not available")?;
        let session = runtime.open_cpu_session(model)?;
        if session.input.element_type != 1
            || session.input.dims.as_slice() != [1, 3, SIDE as i64, SIDE as i64]
        {
            bail!(
                "landmark model input mismatch: type {} dims {:?}",
                session.input.element_type,
                session.input.dims
            );
        }
        if session.outputs.len() != 1
            || session.outputs[0].dims.as_slice() != [1, POINTS52 as i64, 2]
        {
            bail!(
                "landmark model output mismatch: {:?}",
                session
                    .outputs
                    .iter()
                    .map(|output| &output.dims)
                    .collect::<Vec<_>>()
            );
        }

        let fit_pre = fit_52(&points142)?;
        let crop_matrix = half(fit_pre);
        let mut crop_bgr = warp_bgr(source, crop_matrix)?;
        let crop_sha256 = sha256_f32(&crop_bgr);
        let outputs = runtime.run(&session, &mut crop_bgr, &[1, 3, SIDE as i64, SIDE as i64])?;
        if outputs.len() != 1 || outputs[0].len() != POINTS52 * 2 {
            bail!("landmark model returned an unexpected tensor");
        }

        let raw52 = outputs[0]
            .chunks_exact(2)
            .map(|point| [point[0], point[1]])
            .collect::<Vec<_>>();
        let fit_inverse = inverse(fit_pre)?;
        let points52 = raw52
            .iter()
            .map(|point| {
                [
                    point[0] * fit_inverse[0][0] + point[1] * fit_inverse[0][1] + fit_inverse[0][2],
                    point[0] * fit_inverse[1][0] + point[1] * fit_inverse[1][1] + fit_inverse[1][2],
                ]
            })
            .collect::<Vec<_>>();
        let mut points194 = points142.clone();
        points194.extend_from_slice(&points52);
        if points194.len() != POINTS194
            || points194.iter().flatten().any(|value| !value.is_finite())
        {
            bail!("194-point landmark result is incomplete or non-finite");
        }

        Ok(Self {
            resource_id: FACE_LANDMARK_52_RESOURCE,
            provider: session.provider().label().to_owned(),
            model_sha256,
            source_sha256: sha256_32(&source.pixels),
            source_size: [source.width, source.height],
            points142,
            crop_bgr,
            crop_sha256,
            raw52,
            points52,
            points194,
            fit_pre,
            fit_inverse,
        })
    }

    pub fn slim_inputs(&self, job_id: u64, source: &Image) -> Result<SlimProducerInputs, Refusal> {
        SlimProducerInputs::from_landmarks(job_id, source, self.points194.clone())
    }
}

fn fit_52(points142: &[[f32; 2]]) -> Result<[[f32; 3]; 2]> {
    let source = &points142[109..142];
    let inverse_count = 1.0f32 / source.len() as f32;
    let source_mean = mean_f32(source, inverse_count);
    let target_mean = mean_f32(&TEMPLATE33, inverse_count);
    let source_centered = source
        .iter()
        .map(|point| [point[0] - source_mean[0], point[1] - source_mean[1]])
        .collect::<Vec<_>>();
    let target_centered = TEMPLATE33
        .iter()
        .map(|point| [point[0] - target_mean[0], point[1] - target_mean[1]])
        .collect::<Vec<_>>();

    let mut covariance = [[0.0f32; 2]; 2];
    for row in 0..2 {
        for column in 0..2 {
            let dot = source_centered
                .iter()
                .zip(&target_centered)
                .fold(0.0f32, |sum, (point, target)| {
                    sum + point[row] * target[column]
                });
            covariance[row][column] = inverse_count * dot;
        }
    }
    let (u, sigma, vt) = svd2x2(transpose(covariance))?;
    let determinant_u = u[0][0] * u[1][1] - u[1][0] * u[0][1];
    let determinant_v = vt[0][0] * vt[1][1] - vt[0][1] * vt[1][0];
    let second_sign = if determinant_u * determinant_v < 0.0 {
        -1.0f32
    } else {
        1.0f32
    };
    let rotation = [
        [
            (u[0][0] * 1.0) * vt[0][0] + (u[0][1] * second_sign) * vt[1][0],
            (u[0][0] * 1.0) * vt[0][1] + (u[0][1] * second_sign) * vt[1][1],
        ],
        [
            (u[1][0] * 1.0) * vt[0][0] + (u[1][1] * second_sign) * vt[1][0],
            (u[1][0] * 1.0) * vt[0][1] + (u[1][1] * second_sign) * vt[1][1],
        ],
    ];
    let numerator = sigma[0] * 1.0 + sigma[1] * second_sign;
    let variance = inverse_count * squared_norm(&source_centered);
    if variance == 0.0 || !variance.is_finite() {
        bail!("degenerate 52-point source landmarks");
    }
    let scale = numerator * (1.0f32 / variance);
    let mut result = [[0.0f32; 3]; 2];
    for row in 0..2 {
        result[row][0] = scale * rotation[row][0];
        result[row][1] = scale * rotation[row][1];
        result[row][2] = target_mean[row]
            + (-scale) * (rotation[row][0] * source_mean[0] + rotation[row][1] * source_mean[1]);
    }
    if result.iter().flatten().any(|value| !value.is_finite()) {
        bail!("non-finite 52-point landmark fit");
    }
    Ok(result)
}

fn mean_f32(points: &[[f32; 2]], inverse_count: f32) -> [f32; 2] {
    let mut sum = points[0];
    for point in &points[1..] {
        sum[0] += point[0];
        sum[1] += point[1];
    }
    [sum[0] * inverse_count, sum[1] * inverse_count]
}

fn transpose(matrix: [[f32; 2]; 2]) -> [[f32; 2]; 2] {
    [[matrix[0][0], matrix[1][0]], [matrix[0][1], matrix[1][1]]]
}

fn squared_norm(points: &[[f32; 2]]) -> f32 {
    let x = points
        .iter()
        .map(|point| point[0] * point[0])
        .collect::<Vec<_>>();
    let y = points
        .iter()
        .map(|point| point[1] * point[1])
        .collect::<Vec<_>>();
    reduce_lanes(&x) + reduce_lanes(&y)
}

fn reduce_lanes(products: &[f32]) -> f32 {
    if products.len() < 4 {
        return products[1..]
            .iter()
            .fold(products[0], |sum, value| sum + value);
    }
    let mut lanes = [products[0], products[1], products[2], products[3]];
    let n4 = products.len() & !3;
    if products.len() >= 8 {
        let mut high = [products[4], products[5], products[6], products[7]];
        let n8 = products.len() & !7;
        let mut index = 8;
        while index + 8 <= n8 {
            for lane in 0..4 {
                lanes[lane] += products[index + lane];
                high[lane] += products[index + 4 + lane];
            }
            index += 8;
        }
        for lane in 0..4 {
            lanes[lane] = high[lane] + lanes[lane];
        }
        if n4 > n8 {
            for lane in 0..4 {
                lanes[lane] += products[n8 + lane];
            }
        }
    }
    let mut sum = (lanes[3] + lanes[1]) + (lanes[2] + lanes[0]);
    for value in &products[n4..] {
        sum += value;
    }
    sum
}

fn svd2x2(mut work: [[f32; 2]; 2]) -> Result<([[f32; 2]; 2], [f32; 2], [[f32; 2]; 2])> {
    let precision = 2.0f32 * f32::EPSILON;
    let mut scale = work
        .iter()
        .flatten()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
    if scale == 0.0 {
        scale = 1.0;
    }
    for row in &mut work {
        row[0] /= scale;
        row[1] /= scale;
    }
    let mut u = [[1.0f32, 0.0], [0.0, 1.0]];
    let mut v = [[1.0f32, 0.0], [0.0, 1.0]];
    let mut max_diagonal = work[0][0].abs().max(work[1][1].abs());
    for _ in 0..64 {
        let threshold = f32::MIN_POSITIVE.max(precision * max_diagonal);
        if work[1][0].abs() <= threshold && work[0][1].abs() <= threshold {
            break;
        }
        let (left, right) = real2x2(work);
        work = rotate_left(work, left.0, -left.1);
        u = rotate_right(u, left.0, left.1);
        work = rotate_right(work, right.0, -right.1);
        v = rotate_right(v, right.0, -right.1);
        max_diagonal = max_diagonal.max(work[0][0].abs()).max(work[1][1].abs());
    }
    if work[1][0].abs() > f32::MIN_POSITIVE.max(precision * max_diagonal)
        || work[0][1].abs() > f32::MIN_POSITIVE.max(precision * max_diagonal)
    {
        bail!("52-point landmark SVD did not converge");
    }
    let mut sigma = [0.0f32; 2];
    for index in 0..2 {
        let diagonal = work[index][index];
        sigma[index] = diagonal.abs();
        if diagonal < 0.0 {
            u[0][index] = -u[0][index];
            u[1][index] = -u[1][index];
        }
    }
    sigma[0] *= scale;
    sigma[1] *= scale;
    for index in 0..2 {
        let mut best = index;
        for candidate in index + 1..2 {
            if sigma[candidate] > sigma[best] {
                best = candidate;
            }
        }
        if sigma[best] == 0.0 {
            break;
        }
        if best != index {
            sigma.swap(index, best);
            for row in &mut u {
                row.swap(index, best);
            }
            for row in &mut v {
                row.swap(index, best);
            }
        }
    }
    Ok((u, sigma, transpose(v)))
}

fn make_jacobi(x: f32, y: f32, z: f32) -> (f32, f32) {
    let denominator = 2.0 * y.abs();
    if denominator < f32::MIN_POSITIVE {
        return (1.0, 0.0);
    }
    let tau = (x - z) / denominator;
    let w = (tau * tau + 1.0).sqrt();
    let t = if tau > 0.0 {
        1.0 / (tau + w)
    } else {
        1.0 / (tau - w)
    };
    let sign = if t > 0.0 { 1.0 } else { -1.0 };
    let normal = 1.0 / (t * t + 1.0).sqrt();
    let sine = if y == 0.0 {
        0.0
    } else {
        -sign * (y / y.abs()) * t.abs() * normal
    };
    (normal, sine)
}

fn real2x2(matrix: [[f32; 2]; 2]) -> ((f32, f32), (f32, f32)) {
    let trace = matrix[0][0] + matrix[1][1];
    let skew = matrix[0][1] - matrix[1][0];
    let (rotation_sine, rotation_cosine) = if skew.abs() < f32::MIN_POSITIVE {
        (0.0, 1.0)
    } else {
        let ratio = trace / skew;
        let normal = (1.0 + ratio * ratio).sqrt();
        (1.0 / normal, ratio / normal)
    };
    let local = [
        [
            rotation_cosine * matrix[0][0] - rotation_sine * matrix[1][0],
            rotation_cosine * matrix[0][1] - rotation_sine * matrix[1][1],
        ],
        [
            rotation_sine * matrix[0][0] + rotation_cosine * matrix[1][0],
            rotation_sine * matrix[0][1] + rotation_cosine * matrix[1][1],
        ],
    ];
    let (jacobi_cosine, jacobi_sine) = make_jacobi(local[1][1], local[1][0], local[0][0]);
    (
        (
            rotation_cosine * jacobi_cosine - rotation_sine * (-jacobi_sine),
            rotation_cosine * (-jacobi_sine) + rotation_sine * jacobi_cosine,
        ),
        (jacobi_cosine, jacobi_sine),
    )
}

fn rotate_left(mut matrix: [[f32; 2]; 2], cosine: f32, sine: f32) -> [[f32; 2]; 2] {
    let first = matrix[0];
    let second = matrix[1];
    matrix[0] = [
        cosine * first[0] + sine * second[0],
        cosine * first[1] + sine * second[1],
    ];
    matrix[1] = [
        (-sine) * first[0] + cosine * second[0],
        (-sine) * first[1] + cosine * second[1],
    ];
    matrix
}

fn rotate_right(mut matrix: [[f32; 2]; 2], cosine: f32, sine: f32) -> [[f32; 2]; 2] {
    let first = [matrix[0][0], matrix[1][0]];
    let second = [matrix[0][1], matrix[1][1]];
    for row in 0..2 {
        matrix[row][0] = cosine * first[row] + (-sine) * second[row];
        matrix[row][1] = sine * first[row] + cosine * second[row];
    }
    matrix
}

fn half(mut matrix: [[f32; 3]; 2]) -> [[f32; 3]; 2] {
    for row in &mut matrix {
        row[2] = (row[2] - 0.5) + (row[0] * 0.5 + row[1] * 0.5);
    }
    matrix
}

fn inverse(matrix: [[f32; 3]; 2]) -> Result<[[f32; 3]; 2]> {
    let [a, b, c] = matrix[0];
    let [d, e, f] = matrix[1];
    let determinant = a * e - d * b;
    if determinant == 0.0 || !determinant.is_finite() {
        bail!("non-invertible 52-point crop matrix");
    }
    let reciprocal = 1.0 / determinant;
    let aa = e * reciprocal;
    let bb = -b * reciprocal;
    let dd = -d * reciprocal;
    let ee = a * reciprocal;
    Ok([
        [aa, bb, (-aa * c) - (bb * f)],
        [dd, ee, (-dd * c) - (ee * f)],
    ])
}

fn warp_bgr(source: &Image, matrix: [[f32; 3]; 2]) -> Result<Vec<f32>> {
    let inverse = inverse(matrix)?.map(|row| row.map(f64::from));
    let width = source.width as usize;
    let height = source.height as usize;
    let mut output = vec![0.0f32; 3 * SIDE * SIDE];
    for y in 0..SIDE {
        let yf = y as f64;
        let mut x_position = inverse[0][1] * yf + inverse[0][2];
        let mut y_position = inverse[1][1] * yf + inverse[1][2];
        for x in 0..SIDE {
            let ix = x_position.floor();
            let iy = y_position.floor();
            if ix >= 0.0 && ix < (width - 1) as f64 && iy >= 0.0 && iy < (height - 1) as f64 {
                let x0 = ix as usize;
                let y0 = iy as usize;
                let fx = (x_position - ix) as f32;
                let fy = (y_position - iy) as f32;
                let ox = 1.0 - fx;
                let oy = 1.0 - fy;
                let sample = |xx: usize, yy: usize| {
                    let offset = (yy * width + xx) * 4;
                    [
                        f32::from(source.pixels[offset]),
                        f32::from(source.pixels[offset + 1]),
                        f32::from(source.pixels[offset + 2]),
                    ]
                };
                let s00 = sample(x0, y0);
                let s01 = sample(x0 + 1, y0);
                let s10 = sample(x0, y0 + 1);
                let s11 = sample(x0 + 1, y0 + 1);
                for channel in 0..3 {
                    let mut value = s00[channel] * (ox * oy);
                    value += s01[channel] * (fx * oy);
                    value += s10[channel] * (ox * fy);
                    value += s11[channel] * (fx * fy);
                    output[(2 - channel) * SIDE * SIDE + y * SIDE + x] = value;
                }
            }
            x_position += inverse[0][0];
            y_position += inverse[1][0];
        }
    }
    Ok(output)
}

fn sha256_f32(values: &[f32]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    sha256_32(&bytes)
}

fn hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<Vec<_>>()
        .join("")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_recovers_an_orientation_preserving_similarity() {
        let expected = [[0.8, -0.2, 7.0], [0.2, 0.8, -3.0]];
        let inverse_expected = inverse(expected).unwrap();
        let source = TEMPLATE33
            .iter()
            .map(|point| {
                [
                    point[0] * inverse_expected[0][0]
                        + point[1] * inverse_expected[0][1]
                        + inverse_expected[0][2],
                    point[0] * inverse_expected[1][0]
                        + point[1] * inverse_expected[1][1]
                        + inverse_expected[1][2],
                ]
            })
            .collect::<Vec<_>>();
        let mut points142 = vec![[0.0, 0.0]; 142];
        points142[109..142].copy_from_slice(&source);
        let fitted = fit_52(&points142).unwrap();
        for row in 0..2 {
            for column in 0..3 {
                assert!((fitted[row][column] - expected[row][column]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn half_and_inverse_round_trip_the_crop() {
        let matrix = [[1.25, -0.15, 42.0], [0.1, 0.95, -7.0]];
        let inverse = inverse(half(matrix)).unwrap();
        let check = inverse.map(|row| row.map(f64::from));
        let probe = [19.0f64, 31.0];
        let mapped = [
            probe[0] * check[0][0] + probe[1] * check[0][1] + check[0][2],
            probe[0] * check[1][0] + probe[1] * check[1][1] + check[1][2],
        ];
        let forward = half(matrix).map(|row| row.map(f64::from));
        assert!(
            (mapped[0] * forward[0][0] + mapped[1] * forward[0][1] + forward[0][2] - probe[0])
                .abs()
                < 1e-4
        );
        assert!(
            (mapped[0] * forward[1][0] + mapped[1] * forward[1][1] + forward[1][2] - probe[1])
                .abs()
                < 1e-4
        );
    }
}
