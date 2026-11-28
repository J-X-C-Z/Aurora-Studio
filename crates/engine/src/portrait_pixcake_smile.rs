//! PF8033 (微笑嘴角) producer from the PixCake `DL3D_Warp` morphable mesh.
//!
//! The private assets under `private-assets/portrait/smile/` are decoded from
//! the original `SmileMouth3D` effect: a 2206-vertex mean shape, three reshape
//! targets (`normal`, `side_right`, `side_left`) and the shared 4269-face
//! `u32le` topology. This module fits the mean mesh to `points194` and
//! rasterizes the per-vertex `(mean -> target)` displacement into a full-frame
//! unit-strength `(dx, dy, mask)` field for [`FaceGeometry::Smile`].
//!
//! The yaw blend and anatomical fit are an inference from the asset geometry;
//! no vendor golden proves metric parity. The field is a geometric producer,
//! not a captured vendor tensor.

use std::path::PathBuf;
use std::sync::OnceLock;

use aurora_portrait::kernels::mesh_attribute;
use aurora_portrait::{Image, Refusal};

use super::portrait_face_stage::{FaceGeometry, FaceGeometryDraw};
use super::sha256_32;

const VERTEX_COUNT: usize = 2206;
const TRIANGLE_COUNT: usize = 4269;
const FLOAT_BYTES: usize = 26472;
const TRIANGLE_BYTES: usize = 51228;
const RELATIVE: &str = "portrait/smile";
const ASSET_FILES: [&str; 5] = [
    "mean.f32le",
    "normal.f32le",
    "side_left.f32le",
    "side_right.f32le",
    "triangles.u32le",
];

/// `mean.f32le`: (0, -9.21, 72.97), the nose bridge / eye-line centre.
const MESH_EYE_LINE: usize = 78;
/// `mean.f32le`: (0, -97.07, 17.50), chin midpoint.
const MESH_CHIN: usize = 74;
/// `mean.f32le`: (±17.20, -11.82, 61.60), eye-ring vertices.
const MESH_EYE_R: usize = 241;
const MESH_EYE_L: usize = 125;
/// `mean.f32le`: (±29.53, -44.30, 47.18), mouth corners.
const MESH_MOUTH_R: usize = 1456;
const MESH_MOUTH_L: usize = 1867;
/// `mean.f32le`: (0, -38.86, 63.14) upper lip, (0, -52.17, 59.73) lower lip.
const MESH_UPPER_LIP: usize = 1143;
const MESH_LOWER_LIP: usize = 1633;
/// Cheek transition vertices at approximately (±54.6, -30.1, 34.6).
const MESH_CHEEK_R: usize = 912;
const MESH_CHEEK_L: usize = 457;

#[derive(Clone, Debug)]
struct SmileAssets {
    mean: Vec<[f32; 3]>,
    normal: Vec<[f32; 3]>,
    side_left: Vec<[f32; 3]>,
    side_right: Vec<[f32; 3]>,
    faces: Vec<u32>,
}

/// Cached vendor asset decode. A failed load is cached as a refusal so a
/// machine without the private assets does not retry disk access per frame.
static ASSETS: OnceLock<Result<SmileAssets, Refusal>> = OnceLock::new();

fn assets() -> Result<&'static SmileAssets, Refusal> {
    ASSETS
        .get_or_init(|| load_assets())
        .as_ref()
        .map_err(Clone::clone)
}

/// Candidates for the private asset root, in precedence order.
fn smile_dirs() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(root) = std::env::var_os("AURORA_PRIVATE_ASSETS") {
        roots.push(PathBuf::from(root).join(RELATIVE));
    }
    if let Some(root) = aurora_portrait::private_assets::path("") {
        roots.push(root.join(RELATIVE));
    }
    if let Some(dir) = std::env::current_dir().ok() {
        roots.push(dir.join("private-assets").join(RELATIVE));
        roots.push(dir.join(RELATIVE));
        roots.push(dir.join("assets").join(RELATIVE));
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.join("private-assets").join(RELATIVE));
            roots.push(dir.join("assets").join(RELATIVE));
            roots.push(dir.join(RELATIVE));
        }
    }
    let mut seen = std::collections::HashSet::new();
    roots.retain(|root| seen.insert(root.clone()));
    roots
}

/// Search from `CARGO_MANIFEST_DIR` upwards as well. Source-tree runs use the
/// repository root's `private-assets/`; installed runs use the exe directory.
fn manifest_dirs() -> Vec<PathBuf> {
    let Ok(manifest) = std::env::var("CARGO_MANIFEST_DIR") else {
        return Vec::new();
    };
    let mut dir = PathBuf::from(manifest);
    let mut out = Vec::new();
    for _ in 0..8 {
        out.push(dir.join("private-assets").join(RELATIVE));
        out.push(dir.join("assets").join(RELATIVE));
        if !dir.pop() {
            break;
        }
    }
    out
}

pub(crate) fn smile_mesh_assets_present() -> bool {
    find_smile_dir().is_some()
}

fn find_smile_dir() -> Option<PathBuf> {
    smile_dirs()
        .into_iter()
        .chain(manifest_dirs())
        .find(|dir| {
            ASSET_FILES
                .iter()
                .all(|name| dir.join(name).is_file())
        })
}

fn decode_f32_vec(bytes: &[u8], name: &'static str) -> Result<Vec<[f32; 3]>, Refusal> {
    if bytes.len() != FLOAT_BYTES {
        return Err(Refusal::ShapeMismatch(name));
    }
    Ok(bytes
        .chunks_exact(12)
        .map(|chunk| {
            [
                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
                f32::from_le_bytes([chunk[8], chunk[9], chunk[10], chunk[11]]),
            ]
        })
        .collect())
}

fn load_assets() -> Result<SmileAssets, Refusal> {
    let dir = find_smile_dir().ok_or(Refusal::MissingInput("PixCake smile assets"))?;
    let mut buffers = Vec::with_capacity(ASSET_FILES.len());
    for name in ASSET_FILES {
        buffers.push(std::fs::read(dir.join(name)).map_err(|_| {
            Refusal::MissingInput("PixCake smile assets")
        })?);
    }
    let faces = if buffers[4].len() == TRIANGLE_BYTES {
        buffers[4]
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>()
    } else {
        return Err(Refusal::ShapeMismatch("PixCake smile triangles"));
    };
    let smile = SmileAssets {
        mean: decode_f32_vec(&buffers[0], "PixCake smile mean")?,
        normal: decode_f32_vec(&buffers[1], "PixCake smile normal")?,
        side_left: decode_f32_vec(&buffers[2], "PixCake smile side_left")?,
        side_right: decode_f32_vec(&buffers[3], "PixCake smile side_right")?,
        faces,
    };
    if smile.mean.len() != VERTEX_COUNT
        || smile.normal.len() != VERTEX_COUNT
        || smile.side_left.len() != VERTEX_COUNT
        || smile.side_right.len() != VERTEX_COUNT
        || smile.faces.len() != TRIANGLE_COUNT * 3
    {
        return Err(Refusal::ShapeMismatch("PixCake smile mesh counts"));
    }
    if smile
        .faces
        .iter()
        .any(|index| *index as usize >= VERTEX_COUNT)
    {
        return Err(Refusal::ValueOutOfRange("PixCake smile face index"));
    }
    if smile
        .mean
        .iter()
        .chain(&smile.normal)
        .chain(&smile.side_left)
        .chain(&smile.side_right)
        .flat_map(|vertex| vertex.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("PixCake smile vertex"));
    }
    Ok(smile)
}

#[derive(Clone, Copy, Debug)]
struct FaceFit {
    center: [f32; 2],
    axis_x: [f32; 2],
    axis_y: [f32; 2],
    scale: f32,
    yaw: f32,
}

fn midpoint(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    (a[0] - b[0]).hypot(a[1] - b[1])
}

/// Estimate the face centre, in-plane roll, metric scale and a coarse yaw.
/// The eye ring is the roll reference; the mouth width, nose-to-mouth height
/// and nose-to-chin height each contribute to the scale estimate.
fn fit_face(points: &[[f32; 2]]) -> Result<FaceFit, Refusal> {
    if points.len() != 194 {
        return Err(Refusal::ShapeMismatch("points194"));
    }
    let required = [4usize, 9, 26, 44, 54, 57, 73, 83, 125];
    if required
        .iter()
        .any(|index| points[*index].iter().any(|value| !value.is_finite()))
    {
        return Err(Refusal::ValueOutOfRange("points194"));
    }
    let eye_left = points[26];
    let eye_right = points[44];
    let eye_mid = midpoint(eye_left, eye_right);
    let mouth_l = points[73];
    let mouth_r = points[83];
    let mouth_mid = midpoint(mouth_l, mouth_r);
    let nose_tip = points[57];
    let nose_top = points[54];
    let chin = points[125];
    let face_center = [
        eye_mid[0] * 0.45 + mouth_mid[0] * 0.30 + chin[0] * 0.10 + nose_tip[0] * 0.15,
        eye_mid[1] * 0.45 + mouth_mid[1] * 0.30 + chin[1] * 0.10 + nose_tip[1] * 0.15,
    ];
    let eye_dist = distance(eye_left, eye_right);
    let mouth_width = distance(mouth_l, mouth_r);
    let nose_mouth = distance(nose_tip, mouth_mid);
    let nose_chin = distance(nose_tip, chin);
    let brow_width = distance(points[4], points[9]);
    if eye_dist < 8.0 || mouth_width < 4.0 || nose_mouth < 5.0 || nose_chin < 8.0 {
        return Err(Refusal::ValueOutOfRange("points194 geometry"));
    }
    let scale = (eye_dist / 34.4
        + mouth_width / 59.1
        + nose_mouth / 21.5
        + nose_chin / 75.0
        + brow_width / 68.0)
        / 5.0;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Refusal::ValueOutOfRange("points194 scale"));
    }
    let eye_dx = eye_right[0] - eye_left[0];
    let eye_dy = eye_right[1] - eye_left[1];
    let eye_len = eye_dx.hypot(eye_dy);
    let axis_x = [eye_dx / eye_len, eye_dy / eye_len];
    let mut axis_y = [-axis_x[1], axis_x[0]];
    let nose_vector = [nose_top[0] - nose_tip[0], nose_top[1] - nose_tip[1]];
    if nose_vector[0] * axis_y[0] + nose_vector[1] * axis_y[1] < 0.0 {
        axis_y = [-axis_y[0], -axis_y[1]];
    }
    // `axis_y` points upward; image y grows downward.
    let yaw = ((mouth_width / eye_dist) - (59.1 / 34.4)).clamp(-1.0, 1.0)
        * (nose_vector[0] * axis_x[0] + nose_vector[1] * axis_x[1])
        / (scale * 30.0);
    Ok(FaceFit {
        center: face_center,
        axis_x,
        axis_y,
        scale,
        yaw: yaw.clamp(-1.0, 1.0),
    })
}

fn project(point: [f32; 3], fit: &FaceFit) -> [f32; 2] {
    let x = point[0] * fit.scale;
    let y = -point[1] * fit.scale;
    [
        fit.center[0] + x * fit.axis_x[0] + y * fit.axis_y[0],
        fit.center[1] + x * fit.axis_x[1] + y * fit.axis_y[1],
    ]
}

fn project_with_z(point: [f32; 3], fit: &FaceFit) -> [f32; 2] {
    let p = project(point, fit);
    let depth = point[2] * fit.scale * 0.18;
    [p[0] + depth * fit.axis_x[0], p[1] + depth * fit.axis_x[1]]
}

/// Blend `normal`, `side_right` and `side_left` towards `mean` by yaw.
/// Side targets carry a stronger unilateral corner lift, so the blend keeps a
/// front-facing component for a frontal face.
fn target_at(assets: &SmileAssets, index: usize, yaw: f32) -> [f32; 3] {
    let mean = assets.mean[index];
    let front = assets.normal[index];
    let right = assets.side_right[index];
    let left = assets.side_left[index];
    let side_weight = yaw.abs().clamp(0.0, 1.0) * 0.75;
    let side = if yaw >= 0.0 { right } else { left };
    let front_weight = 1.0 - side_weight;
    [
        mean[0]
            + front_weight * (front[0] - mean[0])
            + side_weight * (side[0] - mean[0]),
        mean[1]
            + front_weight * (front[1] - mean[1])
            + side_weight * (side[1] - mean[1]),
        mean[2]
            + front_weight * (front[2] - mean[2])
            + side_weight * (side[2] - mean[2]),
    ]
}

fn anchor_pairs(
    assets: &SmileAssets,
    points: &[[f32; 2]],
    fit: &FaceFit,
) -> Vec<([f32; 2], [f32; 2], f32)> {
    let pairs = [
        (MESH_EYE_LINE, midpoint(points[26], points[44]), 1.0),
        (MESH_CHIN, points[125], 0.85),
        (MESH_EYE_R, points[44], 0.9),
        (MESH_EYE_L, points[26], 0.9),
        (MESH_MOUTH_R, points[83], 1.4),
        (MESH_MOUTH_L, points[73], 1.4),
        (MESH_UPPER_LIP, midpoint(points[73], points[83]), 0.9),
        (MESH_LOWER_LIP, midpoint(points[73], points[83]), 0.9),
        (MESH_CHEEK_R, points[83], 0.6),
        (MESH_CHEEK_L, points[73], 0.6),
    ];
    pairs
        .iter()
        .map(|(index, desired, weight)| {
            (
                project_with_z(assets.mean[*index], fit),
                *desired,
                *weight,
            )
        })
        .collect()
}

/// Thin-plate-spline style regularization: move each vertex by a weighted
/// mean of the anchor residuals, so lips, corners and cheeks stay attached
/// without a full 3x3 linear solve per frame.
fn fit_residual(point: [f32; 2], anchors: &[([f32; 2], [f32; 2], f32)]) -> [f32; 2] {
    let mut sum = [0.0f32; 2];
    let mut total = 0.0f32;
    for (source, desired, weight) in anchors {
        let radius = source[0].hypot(source[1]).max(1.0);
        let distance = (point[0] - source[0]).hypot(point[1] - source[1]);
        let influence = weight / (1.0 + (distance / radius).powi(2) * 4.0);
        sum[0] += (desired[0] - source[0]) * influence;
        sum[1] += (desired[1] - source[1]) * influence;
        total += influence;
    }
    if total <= 0.0 {
        return [0.0, 0.0];
    }
    [sum[0] / total, sum[1] / total]
}

/// Soft mouth-centred feather. Vertices far from the lips fade to zero, so
/// the rasterized field has no hard edge at the mesh boundary.
fn region_feather(mean: [f32; 3]) -> f32 {
    let x = mean[0].abs() / 58.0;
    let y = (mean[1] + 45.0) / 50.0;
    let radius2 = x * x + y * y;
    if radius2 >= 1.0 {
        return 0.0;
    }
    let smooth = 1.0 - radius2;
    smooth * smooth
}

/// Build a unit-strength PF8033 displacement field from the vendor mesh.
///
/// The returned geometry intentionally carries `host_split_receipt: None`:
/// this is a geometric producer, not the captured RGB reconstruction field.
/// See the coordinator notes in the module documentation and handoff.
pub(crate) fn build_pf8033_smile_draw(
    frame: &Image,
    face_points194: &[[f32; 2]],
    degree: f32,
) -> Result<FaceGeometryDraw, Refusal> {
    frame.validate()?;
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("PF8033 degree"));
    }
    if degree == 0.0 {
        // A zero-strength draw has no pixels to change and no mesh to load.
        return Ok(FaceGeometryDraw {
            pf: 8033,
            controls: [degree, 0.0],
            source_sha256: sha256_32(&frame.pixels),
            geometry: FaceGeometry::SmileMesh {
                dx: vec![0.0; frame.pixel_count()],
                dy: vec![0.0; frame.pixel_count()],
                mask: vec![0; frame.pixel_count()],
            },
        });
    }
    let assets = assets()?;
    let fit = fit_face(face_points194)?;
    let anchors = anchor_pairs(assets, face_points194, &fit);

    let width = frame.width as usize;
    let height = frame.height as usize;
    let pixels = width * height;
    let mut dest: Vec<[f32; 2]> = Vec::with_capacity(VERTEX_COUNT);
    let mut attr: Vec<[f32; 2]> = Vec::with_capacity(VERTEX_COUNT);
    for index in 0..VERTEX_COUNT {
        let mean = assets.mean[index];
        let source = project_with_z(mean, &fit);
        let residual = fit_residual(source, &anchors);
        let source = [source[0] + residual[0], source[1] + residual[1]];
        let target = project_with_z(target_at(assets, index, fit.yaw), &fit);
        let target = [target[0] + residual[0], target[1] + residual[1]];
        dest.push(source);
        let feather = region_feather(mean);
        attr.push([(target[0] - source[0]) * feather, (target[1] - source[1]) * feather]);
    }

    let raster = mesh_attribute::rasterize(
        frame.width,
        frame.height,
        &dest,
        &attr,
        &assets.faces,
    )?;
    let mut dx = vec![0.0f32; pixels];
    let mut dy = vec![0.0f32; pixels];
    let mut mask = vec![0u8; pixels];
    for (pixel, (value, writes)) in raster.values.iter().zip(&raster.writes).enumerate() {
        if *writes == 0 {
            continue;
        }
        // Last-writer coverage has tiny overlap seams; normalize by the write
        // count so a covered pixel has a continuous displacement field.
        let inv = 1.0 / *writes as f32;
        dx[pixel] = value[0] * inv;
        dy[pixel] = value[1] * inv;
        // The mask is an intensity weight in [0, 255]; the caller applies the
        // remaining `degree` multiplication in `pf8033_dense_warp`.
        let feather = ((value[0] * inv).hypot(value[1] * inv) / fit.scale).clamp(0.0, 1.0);
        mask[pixel] = (feather * 255.0 + 0.5) as u8;
    }
    debug_assert!(dx.iter().chain(&dy).all(|value| value.is_finite()));
    Ok(FaceGeometryDraw {
        pf: 8033,
        controls: [degree, 0.0],
        source_sha256: sha256_32(&frame.pixels),
        geometry: FaceGeometry::SmileMesh { dx, dy, mask },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_face() -> Vec<[f32; 2]> {
        let mut points = vec![[128.0f32, 128.0]; 194];
        points[26] = [96.0, 108.0];
        points[44] = [160.0, 108.0];
        points[4] = [88.0, 96.0];
        points[9] = [168.0, 96.0];
        points[54] = [128.0, 126.0];
        points[57] = [128.0, 136.0];
        points[73] = [113.0, 166.0];
        points[83] = [143.0, 166.0];
        points[125] = [128.0, 210.0];
        points
    }

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn vendor_assets_have_expected_counts() {
        let smile = assets().unwrap();
        assert_eq!(smile.mean.len(), 2206);
        assert_eq!(smile.normal.len(), 2206);
        assert_eq!(smile.side_left.len(), 2206);
        assert_eq!(smile.side_right.len(), 2206);
        assert_eq!(smile.faces.len(), 4269 * 3);
        assert!(smile.faces.iter().all(|index| (*index as usize) < 2206));
    }

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn synthetic_face_produces_a_smooth_smile_field() {
        let frame = Image::solid(256, 256, [0, 0, 0, 255]);
        let draw = build_pf8033_smile_draw(&frame, &synthetic_face(), 1.0).unwrap();
        assert_eq!(draw.pf, 8033);
        let FaceGeometry::SmileMesh { dx, dy, mask } = draw.geometry else {
            panic!("expected smile mesh geometry");
        };
        assert_eq!(dx.len(), 256 * 256);
        assert_eq!(dy.len(), 256 * 256);
        assert_eq!(mask.len(), 256 * 256);
        assert!(dx.iter().chain(&dy).all(|value| value.is_finite()));
        assert!(dx.iter().any(|value| *value != 0.0));
        assert!(dy.iter().any(|value| *value != 0.0));
        assert!(mask.iter().any(|value| *value > 0));
        // Corners are approximately x=113/143, y=166: verify real movement
        // near them and a zero field at the frame boundary.
        let corner = 166 * 256 + 113;
        assert!(dx[corner].abs() > 0.01 || dy[corner].abs() > 0.01);
        for x in 0..256 {
            assert_eq!(dx[x], 0.0);
            assert_eq!(mask[x], 0);
            assert_eq!(dx[255 * 256 + x], 0.0);
        }
        for y in 0..256 {
            assert_eq!(dx[y * 256], 0.0);
            assert_eq!(mask[y * 256], 0);
            assert_eq!(dx[y * 256 + 255], 0.0);
        }
        // A tiny step between neighbouring pixels proves the field is smooth
        // (no single-pixel hard edge).
        let mut max_step = 0.0f32;
        for y in 1..255 {
            for x in 1..255 {
                let i = y * 256 + x;
                max_step = max_step.max((dx[i] - dx[i - 1]).abs());
                max_step = max_step.max((dy[i] - dy[i - 255]).abs());
            }
        }
        assert!(max_step < 5.0, "unexpected discontinuity: {max_step}");
    }

    #[test]
    fn invalid_inputs_are_refused() {
        let frame = Image::solid(64, 64, [0, 0, 0, 255]);
        assert_eq!(
            build_pf8033_smile_draw(&frame, &[], 1.0),
            Err(Refusal::ShapeMismatch("points194"))
        );
        let mut points = synthetic_face();
        points[26] = [f32::NAN, 108.0];
        assert_eq!(
            build_pf8033_smile_draw(&frame, &points, 1.0),
            Err(Refusal::ValueOutOfRange("points194"))
        );
        let mut points = synthetic_face();
        points[44] = points[26];
        assert_eq!(
            build_pf8033_smile_draw(&frame, &points, 1.0),
            Err(Refusal::ValueOutOfRange("points194 geometry"))
        );
        assert_eq!(
            build_pf8033_smile_draw(&frame, &synthetic_face(), f32::NAN),
            Err(Refusal::ValueOutOfRange("PF8033 degree"))
        );
        let bad_frame = Image { width: 4, height: 4, pixels: Vec::new() };
        assert_eq!(
            build_pf8033_smile_draw(&bad_frame, &synthetic_face(), 1.0),
            Err(Refusal::ShapeMismatch("image"))
        );
    }
}
