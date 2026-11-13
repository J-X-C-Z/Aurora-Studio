//! Face3d pose/camera composition for the PF8033 smile projection.
//!
//! Recovered from `pixcake.exe` (`a84519da...64c7`) at
//! `0x7119B56..0x7119F29` (pose), `0x7B138E0..0x7B13BC4` (camera) and
//! `0x711A103..0x711AA05` (compose). The native SSE fixture matches within
//! `1e-6 * max(1, |expected|)`.
//!
//! ```text
//! M = inverse(viewport_image) @ embed(A) @ viewport_crop @ aspect_crop
//!     @ projection_crop @ view_crop @ pose(p7)
//! ```
//!
//! `p7 = [w, x, y, z, tx, ty, tz]`; the quaternion is normalised and its
//! rotation transposed, `0.01` scales the 3x3 block and translation stays
//! unscaled. `A` is the caller's crop-to-image 2x3 affine. The returned matrix
//! is clip space without a perspective divide; PF8033 consumes rows 0 and 1
//! as an `Affine3x4`.
//!
//! The caller must supply the same-job `p7` and crop affine. There is no
//! default pose, identity crop or guessed image size.

use crate::Refusal;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mat4 {
    /// Row-major with column-vector convention (`M @ v`).
    pub m: [f32; 16],
}

impl Mat4 {
    pub fn identity() -> Self {
        let mut m = [0.0f32; 16];
        m[0] = 1.0;
        m[5] = 1.0;
        m[10] = 1.0;
        m[15] = 1.0;
        Self { m }
    }

    pub fn mul(&self, rhs: &Mat4) -> Mat4 {
        let mut out = [0.0f32; 16];
        for row in 0..4 {
            for col in 0..4 {
                let mut sum = 0.0f32;
                for k in 0..4 {
                    sum += self.m[row * 4 + k] * rhs.m[k * 4 + col];
                }
                out[row * 4 + col] = sum;
            }
        }
        Mat4 { m: out }
    }

    /// `Affine3x4` layout: `[m00,m01,m02,m03, m10,m11,m12,m13]`.
    pub fn affine3x4(&self) -> [f32; 8] {
        [
            self.m[0], self.m[1], self.m[2], self.m[3],
            self.m[4], self.m[5], self.m[6], self.m[7],
        ]
    }
}

/// `0x7119B56..0x7119F29`.
pub fn pose_matrix(params7: [f32; 7]) -> Result<Mat4, Refusal> {
    if params7.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("face_pose params7"));
    }
    let quat = [params7[1], params7[2], params7[3], params7[0]];
    let norm2 = quat.iter().fold(0.0f32, |acc, value| acc + value * value);
    let q = if norm2 > 0.0 {
        let inv = norm2.sqrt().recip();
        [quat[0] * inv, quat[1] * inv, quat[2] * inv, quat[3] * inv]
    } else {
        quat
    };
    let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
    let xx = x * (x + x);
    let yy = y * (y + y);
    let zz = z * (z + z);
    let xy = x * (y + y);
    let xz = x * (z + z);
    let yz = y * (z + z);
    let wx = w * (x + x);
    let wy = w * (y + y);
    let wz = w * (z + z);
    let rotation = [
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ];
    let mut out = Mat4::identity();
    for row in 0..3 {
        for col in 0..3 {
            out.m[row * 4 + col] = rotation[col][row] * 0.01;
        }
        out.m[row * 4 + 3] = params7[4 + row];
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub view: Mat4,
    pub projection: Mat4,
    pub aspect: Mat4,
    pub viewport: Mat4,
    pub viewport_inverse: Mat4,
}

/// `7B138E0..7B13BC4`, perspective branch.
pub fn camera_matrices(width: u32, height: u32) -> Result<Camera, Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ValueOutOfRange("face_pose image size"));
    }
    let w = width as f32;
    let h = height as f32;
    let near = 1.0f32;
    let far = 100.0f32;
    let half_fov = 39.4f32 * (core::f32::consts::PI / 180.0) * 0.5;
    let f = 1.0 / half_fov.tan();

    let mut view = Mat4::identity();
    view.m[0] = -1.0;
    view.m[10] = -1.0;
    view.m[11] = -6.0;

    let mut projection = Mat4 { m: [0.0; 16] };
    projection.m[0] = f;
    projection.m[5] = f;
    projection.m[10] = -far / (far - near);
    projection.m[11] = -near * far / (far - near);
    projection.m[14] = -1.0;

    let smaller = w.min(h);
    let mut aspect = Mat4::identity();
    aspect.m[0] = smaller / w;
    aspect.m[5] = -smaller / h;

    let mut viewport = Mat4::identity();
    viewport.m[0] = w * 0.5;
    viewport.m[5] = h * 0.5;
    viewport.m[3] = w * 0.5;
    viewport.m[7] = h * 0.5;

    let mut viewport_inverse = Mat4::identity();
    viewport_inverse.m[0] = 2.0 / w;
    viewport_inverse.m[5] = 2.0 / h;
    viewport_inverse.m[3] = -1.0;
    viewport_inverse.m[7] = -1.0;

    Ok(Camera {
        view,
        projection,
        aspect,
        viewport,
        viewport_inverse,
    })
}

/// `711A103..711AA05`; `crop_to_image` is the caller's 2x3 affine.
pub fn compose(
    params7: [f32; 7],
    crop_to_image: [[f32; 3]; 2],
    image_size: (u32, u32),
    crop_size: u32,
) -> Result<Mat4, Refusal> {
    if crop_to_image.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("face_pose crop affine"));
    }
    if crop_size == 0 {
        return Err(Refusal::ValueOutOfRange("face_pose crop size"));
    }
    let image = camera_matrices(image_size.0, image_size.1)?;
    let crop = camera_matrices(crop_size, crop_size)?;
    let mut embed = Mat4::identity();
    embed.m[0] = crop_to_image[0][0];
    embed.m[1] = crop_to_image[0][1];
    embed.m[3] = crop_to_image[0][2];
    embed.m[4] = crop_to_image[1][0];
    embed.m[5] = crop_to_image[1][1];
    embed.m[7] = crop_to_image[1][2];
    let pose = pose_matrix(params7)?;
    Ok(image
        .viewport_inverse
        .mul(&embed)
        .mul(&crop.viewport)
        .mul(&crop.aspect)
        .mul(&crop.projection)
        .mul(&crop.view)
        .mul(&pose))
}

#[cfg(test)]
mod tests {
    use super::*;

    const PARAMS7: [f32; 7] = [
        -0.02479267120361328,
        0.0728079080581665,
        -0.9082702994346619,
        0.2312941551208496,
        1.1501003503799438,
        0.3693595230579376,
        -0.8741798400878906,
    ];
    const AFFINE: [[f32; 3]; 2] = [[1.2, 0.1, 23.0], [-0.05, 0.8, 17.0]];
    /// Original SSE output, SHA256 `D1912274...14542A` (64 bytes).
    const NATIVE: [f32; 16] = [
        0.026857814, 0.0018932893, 0.00043308805, -2.5926967,
        0.0016705572, -0.0202973, 0.012534456, -1.3710551,
        0.00089908426, -0.0047576656, -0.00886491, 4.1674953,
        0.00089009345, -0.004710089, -0.0087762615, 5.12582,
    ];

    #[test]
    fn compose_matches_the_native_sse_fixture() {
        let matrix = compose(PARAMS7, AFFINE, (640, 480), 512).unwrap();
        for index in 0..16 {
            let diff = (matrix.m[index] - NATIVE[index]).abs();
            let scale = NATIVE[index].abs().max(1.0);
            assert!(diff <= 1e-6 * scale, "element {index}");
        }
    }

    #[test]
    fn affine_rows_are_the_projection_input() {
        let matrix = compose(PARAMS7, AFFINE, (640, 480), 512).unwrap();
        assert_eq!(matrix.affine3x4(), [
            matrix.m[0], matrix.m[1], matrix.m[2], matrix.m[3],
            matrix.m[4], matrix.m[5], matrix.m[6], matrix.m[7],
        ]);
    }

    #[test]
    fn identity_quaternion_is_unit_rotation_scaled() {
        let pose = pose_matrix([0.0; 7]).unwrap();
        for index in 0..3 {
            assert!((pose.m[index * 4 + index] - 0.01).abs() < 1e-7);
        }
    }

    #[test]
    fn refusals_are_typed() {
        assert_eq!(
            pose_matrix([f32::NAN, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            Err(Refusal::ValueOutOfRange("face_pose params7"))
        );
        assert_eq!(
            camera_matrices(0, 480),
            Err(Refusal::ValueOutOfRange("face_pose image size"))
        );
        assert_eq!(
            compose(PARAMS7, AFFINE, (640, 480), 0),
            Err(Refusal::ValueOutOfRange("face_pose crop size"))
        );
    }
}
