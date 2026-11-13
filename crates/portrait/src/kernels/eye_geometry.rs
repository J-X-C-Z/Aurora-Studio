//! Eye and brow geometry producers for rows 11-24 (P28 / p06e / p06f port).
//!
//! Pinned source artifacts (read-only inputs to this port):
//!   p28-priority11-25-face-parse/source/P28FaceMath.cs
//!     sha256 d98e28cc920c5fa5c4e20d5b176b06c4a677cc94cd65efbed059e85324bfcb41
//!   p28-priority11-25-face-parse/source/P28Face24Dispatch.cs
//!     sha256 058efaa4456cef6b3346330e00d8d8ca2cf17825a350e9465f950a8540e00d37
//!   p06e-eye-cpu-executors/PATCH-CANDIDATE/FaceEyeCpuExecutors.cs
//!     sha256 ae9fd64916f886d23e714ea058b01c7dc4efa65e8a61d8bd5d526d6209a79b25
//!   p06f-facialrefine-eyebrow/PATCH-CANDIDATE/FaceRefineType1005Cpu.cs
//!     sha256 b943fa9dbf56640baac33e94f206a3e8996c94fefab52ada72717d72224fd46b
//!
//! Recovered RVA landmarks preserved from the sources above:
//!   quad numeric producer 0x4C8D540, eye cell 0x4C8A910, range map 0x4C96360,
//!   spline 0x60EBDF0, deform stage 1 0x618E810, deform stage 2 0x618EB50,
//!   pair2 rotation 0x4C8BE90, pair5 0x4C8B650, FacialRefine tilt gap 0x4C8E9F0,
//!   Type1010 factory 0x60F03F6 / render 0x4C96A60 (mesh producer OPEN),
//!   Type1011 render 0x61913A0 / EBO 0x60F2B60 (EBO not synthesized),
//!   Type1005 handler 0x707A330 / slots 0x707AA00 (2,3) and 0x707B040 (6,7),
//!   brow shape 0x707B910, Type1018 guides 0x707BC70 / 0x707F550, leaf 0x707ED50.
//!
//! Every producer takes explicit same-job inputs. Missing landmarks, meshes,
//! guides, face fields or an index buffer are typed refusals: this module never
//! invents a landmark, a mesh, a triangulation or an index template.
//!
//! Row 12 zero branch follows the real source: native throws
//! `"PF8013/PF8014 requires two eye quads."` instead of emitting a no-op.
//! Rows 11/13 zero are a clean no-emit; row 14/16 use the wrapped
//! `ComputeEyePositionPair5` identity clone, not the raw all-zero mesh defect.

use crate::Access;
use crate::EffectMeta;

/// Local 2-D point in vendor space (pixels or normalized, as supplied).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub fn finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    fn add(self, other: Point) -> Point {
        Point::new(self.x + other.x, self.y + other.y)
    }

    fn sub(self, other: Point) -> Point {
        Point::new(self.x - other.x, self.y - other.y)
    }

    fn scale(self, factor: f32) -> Point {
        Point::new(self.x * factor, self.y * factor)
    }

    fn length(self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub y0: f32,
    pub y1: f32,
    pub x0: f32,
    pub x1: f32,
}

/// Quad emitted by the EyeRefine paths. Pixel positions and UVs are both
/// carried; `QUAD_INDICES` is the recovered 12-byte EBO (`0,1,2,1,2,3`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quad {
    pub pixels: [Point; 4],
    pub uv: [Point; 4],
    pub center: Point,
    pub radius: f32,
    pub bounds: Bounds,
}

pub const QUAD_INDICES: [u16; 6] = [0, 1, 2, 1, 2, 3];

/// Typed refusal. `OpenGeometry` is a recovered degree/control with its mesh
/// producer still closed; `MissingEbo` is a mesh with no captured index buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    NonFinite(&'static str),
    OutOfRange(&'static str),
    MissingLandmarks,
    MissingMesh(&'static str),
    MissingGuide,
    MissingFaceFields,
    MissingParamList,
    MissingFactorList,
    MissingPreparedMesh,
    MissingEbo(&'static str),
    OpenGeometry(&'static str),
    ZeroQuads(&'static str),
    UnsupportedRoute(u32),
}

/// Row output mirroring the recovered producer result shapes.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Output {
    pub degrees: Vec<f32>,
    pub quads: Vec<Quad>,
    pub source: Vec<Point>,
    pub target: Vec<Point>,
    pub mids: Vec<Point>,
    pub scalars: Vec<f32>,
    pub open: Vec<&'static str>,
    pub noop: bool,
}

/// Recovered degrees whose mesh producer is still OPEN. Mesh, EBO and pixels
/// are deliberately absent; `refusal` names the exact gap.
#[derive(Clone, Debug, PartialEq)]
pub struct OpenDegrees {
    pub degrees: Vec<f32>,
    pub refusal: Refusal,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrowPartKind {
    Short,
    Long,
}

/// Type 1018 inputs: 30 guides, 24 param floats, 48 factor floats, 78 targets.
#[derive(Clone, Copy, Debug)]
pub struct BrowLengthInputs<'a> {
    pub a30: &'a [Point],
    pub centerline_p24: &'a [Point],
    pub centerline_f24: &'a [Point],
    pub param_list24: &'a [f32],
    pub factor_list48: &'a [f32],
    pub targets78: &'a [Point],
    pub scale: f32,
}

/// Registry rows for every PF whose math is actually callable here.
///
/// PF8007/8008, PF8003/8004, PF30071/30077 and PF30072/30078 stay out: their
/// FacialRefine/Type1010 mesh producers are OPEN, so no row is claimed.
/// PF8015/8016 has no proven UI key (`UiKeyProven=false`), so its key is empty.
pub const META: &[EffectMeta] = &[
    row(8001, "", "粗细", &ROW20_INPUTS),
    row(8002, "", "粗细", &ROW20_INPUTS),
    row(8005, "", "眉距", &ROW21_INPUTS),
    row(8006, "", "眉距", &ROW21_INPUTS),
    row(8009, "", "形状", &ROW22_INPUTS),
    row(8010, "", "形状", &ROW22_INPUTS),
    row(8011, "", "眼瞳大小", &ROW11_INPUTS),
    row(8012, "", "眼瞳大小", &ROW11_INPUTS),
    row(8013, "", "眼高", &ROW12_INPUTS),
    row(8014, "", "眼高", &ROW12_INPUTS),
    row(8015, "", "眼距", &ROW14_INPUTS),
    row(8016, "", "眼距", &ROW14_INPUTS),
    row(8017, "", "倾斜", &ROW15_INPUTS),
    row(8018, "", "倾斜", &ROW15_INPUTS),
    row(8019, "Be.eyeWidth_left", "眼宽", &ROW13_INPUTS),
    row(8020, "Be.eyeWidth_right", "眼宽", &ROW13_INPUTS),
    row(8021, "", "位置", &ROW16_INPUTS),
    row(8022, "", "位置", &ROW16_INPUTS),
    row(30073, "", "眼睑下至", &ROW19_INPUTS),
    row(30079, "", "眼睑下至", &ROW19_INPUTS),
    row(41922, "", "眉尾长短", &ROW24_INPUTS),
    row(44124, "", "眉头长短", &ROW23_INPUTS),
    row(48019, "", "眉头长短", &ROW23_INPUTS),
    row(49221, "", "眉尾长短", &ROW24_INPUTS),
];

const fn row(
    pf: u32,
    key: &'static str,
    label: &'static str,
    inputs: &'static [&'static str],
) -> EffectMeta {
    EffectMeta { pf, key, label, access: Access::Cpu, inputs }
}

const ROW11_INPUTS: [&str; 7] =
    ["landmarks194", "width", "height", "face_scale", "ui_left", "ui_right", "eye_factor"];
const ROW12_INPUTS: [&str; 7] =
    ["landmarks194", "width", "height", "face_scale", "ui_left", "ui_right", "factor_list1"];
const ROW13_INPUTS: [&str; 7] =
    ["landmarks194", "width", "height", "face_scale", "ui_left", "ui_right", "render_factor"];
const ROW14_INPUTS: [&str; 9] = [
    "landmarks194", "width", "height", "face_scale", "ui_left", "ui_right", "pair5", "indices",
    "same_job_identity",
];
const ROW15_INPUTS: [&str; 10] = [
    "landmarks194", "ui_left", "ui_right", "pair2_source_20", "pair2_indices", "tilt_factor",
    "angle_scale", "geometry_scale", "ring_center", "same_job_identity",
];
const ROW16_INPUTS: [&str; 6] =
    ["landmarks194", "ui_left", "ui_right", "pair5", "indices", "same_job_identity"];
const ROW19_INPUTS: [&str; 7] =
    ["prepared100", "param_list6", "center", "ui_left", "ui_right", "ebo_0x60F2B60", "same_job_identity"];
const ROW20_INPUTS: [&str; 6] =
    ["mesh168", "guide168", "face_fields", "ui_left", "ui_right", "indices"];
const ROW21_INPUTS: [&str; 6] =
    ["mesh168", "raw_face", "ui_left", "ui_right", "indices", "same_job_identity"];
const ROW22_INPUTS: [&str; 6] =
    ["mesh168", "guide168", "face_fields", "ui_left", "ui_right", "indices"];
const ROW23_INPUTS: [&str; 8] = [
    "type1018_inputs", "indices", "ui_left", "ui_right", "head_slots", "part_kind",
    "same_job_identity", "index_builder_0x7080940",
];
const ROW24_INPUTS: [&str; 7] = [
    "type1018_inputs", "indices", "ui_left", "ui_right", "head_slots", "part_kind",
    "same_job_identity",
];

// ---------------------------------------------------------------------------
// Degree maps (0x4C96360 piecewise lerp; UI normalisers).
// ---------------------------------------------------------------------------

const EYE_SIZE_PARAM: [f32; 4] = [0.0, 0.5, 0.501, 1.0];
const EYE_SIZE_DEGREE: [f32; 4] = [-1.0, 0.0, 0.15, 1.3];
const UNIT_PARAM: [f32; 2] = [0.0, 1.0];
const SYM_DEGREE: [f32; 2] = [-1.0, 1.0];
const PAIR_PARAM: [f32; 3] = [0.0, 0.5, 1.0];
const NEG_DEGREE: [f32; 3] = [-1.0, 0.0, 0.0];
const POS_DEGREE: [f32; 3] = [0.0, 0.0, 1.0];
const HEAD_NEG_DEGREE: [f32; 3] = [-1.0, 0.0, 0.0];
const HEAD_POS_DEGREE: [f32; 3] = [0.0, 0.0, 1.0];
const TAIL_NEG_DEGREE: [f32; 3] = [-1.6, 0.0, 0.0];
const TAIL_POS_DEGREE: [f32; 3] = [0.0, 0.0, 1.8];

/// UI percent -> control `0..1` (`(ui + 100) / 200`).
fn control(ui: f32) -> Result<f32, Refusal> {
    if !ui.is_finite() || !(-100.0..=100.0).contains(&ui) {
        return Err(Refusal::OutOfRange("ui_outside_-100_100"));
    }
    Ok((ui + 100.0) / 200.0)
}

/// UI percent -> signed degree `ui / 100` (EyeRefine MapUi).
fn signed(ui: f32) -> Result<f32, Refusal> {
    if !ui.is_finite() || !(-100.0..=100.0).contains(&ui) {
        return Err(Refusal::OutOfRange("ui_outside_-100_100"));
    }
    Ok(ui / 100.0)
}

/// Piecewise lerp. Out-of-table values refuse; they never pass through.
fn map(control: f32, param: &[f32], degree: &[f32]) -> Result<f32, Refusal> {
    if param.len() < 2 || param.len() != degree.len() {
        return Err(Refusal::OutOfRange("degree_table_shape"));
    }
    if !control.is_finite() || control < param[0] || control > param[param.len() - 1] {
        return Err(Refusal::OutOfRange("degree_table_miss"));
    }
    for i in 1..param.len() {
        if control < param[i - 1] || param[i] < control {
            continue;
        }
        let span = param[i] - param[i - 1];
        if span == 0.0 {
            return Err(Refusal::OutOfRange("degree_table_zero_span"));
        }
        return Ok(degree[i - 1]
            + (degree[i] - degree[i - 1]) * ((control - param[i - 1]) / span));
    }
    let last = param.len() - 1;
    Ok(if param[last] == control { degree[last] } else { control })
}

fn map_unit(control: f32) -> Result<f32, Refusal> {
    map(control, &UNIT_PARAM, &SYM_DEGREE)
}

// ---------------------------------------------------------------------------
// Kernels.
// ---------------------------------------------------------------------------

fn distance(a: Point, b: Point) -> f32 {
    a.sub(b).length()
}

fn all_finite(points: &[Point]) -> bool {
    points.iter().all(|p| p.finite())
}

fn all_finite_f32(values: &[f32]) -> bool {
    values.iter().all(|v| v.is_finite())
}

fn clamp(value: f32, low: f32, high: f32) -> f32 {
    value.max(low).min(high)
}

fn trunc(value: f32) -> f32 {
    value.trunc()
}

/// 0x4C8A910 / 0x4C8D540 shared eye frame used by rows 11-14.
struct EyeFrame {
    distance: f32,
    center_left: Point,
    center_right: Point,
}

fn eye_frame(
    landmarks: &[Point],
    width: i32,
    height: i32,
    scale: Point,
) -> Result<EyeFrame, Refusal> {
    if landmarks.len() != 194 {
        return Err(Refusal::MissingLandmarks);
    }
    if width <= 0 || height <= 0 {
        return Err(Refusal::OutOfRange("image_extent_invalid"));
    }
    if !all_finite(landmarks) || !scale.finite() {
        return Err(Refusal::NonFinite("landmark_or_scale_non_finite"));
    }
    let w = width as f32;
    let h = height as f32;
    let inverse_diagonal = 1.0 / (w * w + h * h).sqrt();
    let distance = if scale.x > 0.01 {
        distance(landmarks[18], landmarks[22]) * inverse_diagonal
    } else {
        distance(landmarks[36], landmarks[40]) * inverse_diagonal
    };
    Ok(EyeFrame {
        distance,
        center_left: Point::new(landmarks[26].x / w, landmarks[26].y / h),
        center_right: Point::new(landmarks[44].x / w, landmarks[44].y / h),
    })
}

/// Bbox law from 0x4C8D540: `trunc(x - 0.5) - 2`, clamped to the image.
fn quad_bounds(x0: f32, x1: f32, y0: f32, y1: f32, width: i32, height: i32) -> Bounds {
    let w = width as f32;
    let h = height as f32;
    Bounds {
        y0: clamp(trunc(y0.min(y1).min(h) - 0.5) - 2.0, 0.0, h),
        y1: clamp(trunc(y1.max(y0).max(0.0) + 0.5) + 2.0, 0.0, h),
        x0: clamp(trunc(x0.min(x1).min(w) - 0.5) - 2.0, 0.0, w),
        x1: clamp(trunc(x1.max(x0).max(0.0) + 0.5) + 2.0, 0.0, w),
    }
}

/// `BuildQuad` from the numeric pair producer. `uv_as_pixels` reproduces the
/// row-12 VBO divergence (UV slice written as pixel/size, not centre±radius).
fn build_quad(
    center: Point,
    radius: f32,
    width: i32,
    height: i32,
    aspect: f32,
    uv_as_pixels: bool,
) -> Quad {
    let w = width as f32;
    let h = height as f32;
    let ry = radius / aspect;
    let x0 = (center.x - radius) * w;
    let x1 = (center.x + radius) * w;
    let y0 = (center.y - ry) * h;
    let y1 = (center.y + ry) * h;
    let pixels = [
        Point::new(x0, y1),
        Point::new(x1, y1),
        Point::new(x0, y0),
        Point::new(x1, y0),
    ];
    let uv = if uv_as_pixels {
        [
            Point::new(x0 / w, y1 / h),
            Point::new(x1 / w, y1 / h),
            Point::new(x0 / w, y0 / h),
            Point::new(x1 / w, y0 / h),
        ]
    } else {
        [
            Point::new(center.x - radius, center.y + ry),
            Point::new(center.x + radius, center.y + ry),
            Point::new(center.x - radius, center.y - ry),
            Point::new(center.x + radius, center.y - ry),
        ]
    };
    Quad {
        pixels,
        uv,
        center,
        radius,
        bounds: quad_bounds(x0, x1, y0, y1, width, height),
    }
}

/// Type-30 3-point spline (0x60EBDF0): 7 samples per segment plus endpoint.
fn sample15(p0: Point, p1: Point, p2: Point) -> Option<[Point; 15]> {
    let t1 = distance(p1, p0);
    let t2 = t1 + distance(p2, p1);
    let h0 = t1;
    let h1 = t2 - t1;
    if h0 <= 0.0 || h1 <= 0.0 || t2 <= 0.0 {
        return None;
    }
    let times = [0.0, t1, t2];
    let cx = fit(p0.x, p1.x, p2.x, h0, h1, t2);
    let cy = fit(p0.y, p1.y, p2.y, h0, h1, t2);
    let mut output = [Point::new(0.0, 0.0); 15];
    for segment in 0..2 {
        let step = (times[segment + 1] - times[segment]) / 7.0;
        let c0x = if segment == 0 { p0.x } else { p1.x };
        let c0y = if segment == 0 { p0.y } else { p1.y };
        for s in 0..7 {
            let dx = s as f32 * step;
            output[segment * 7 + s] = Point::new(
                ((cx[segment * 3 + 2] * dx + cx[segment * 3 + 1]) * dx + cx[segment * 3]) * dx
                    + c0x,
                ((cy[segment * 3 + 2] * dx + cy[segment * 3 + 1]) * dx + cy[segment * 3]) * dx
                    + c0y,
            );
        }
    }
    output[14] = p2;
    Some(output)
}

fn fit(a: f32, b: f32, c: f32, h0: f32, h1: f32, t_span: f32) -> [f32; 9] {
    let diagonal = t_span as f64 * (2.0 / 3.0);
    let s0 = (b - a) / h0;
    let s1 = (c - b) / h1;
    let rhs = s1 - s0;
    let c2 = [0.0, (rhs as f64 * (1.0 / diagonal)) as f32, 0.0];
    let c3 = [
        (((c2[1] - c2[0]) as f64 * (1.0 / 3.0)) / h0 as f64) as f32,
        (((c2[2] - c2[1]) as f64 * (1.0 / 3.0)) / h1 as f64) as f32,
        0.0,
    ];
    let c1 = [
        s0 - ((2.0 * c2[0] + c2[1]) * (1.0 / 3.0) * h0),
        s1 - ((2.0 * c2[1] + c2[2]) * (1.0 / 3.0) * h1),
        s1 + ((c2[1] + 2.0 * c2[2]) * (1.0 / 3.0) * h1),
    ];
    [c1[0], c1[1], c1[2], c2[0], c2[1], c2[2], c3[0], c3[1], c3[2]]
}

// ---------------------------------------------------------------------------
// Rows 14/16 shared 120-vertex eye pair mesh (0x618E810 / 0x618EB50).
// ---------------------------------------------------------------------------

const EYE_PAIR_VERTICES: usize = 120;
const CURVE_SCALE: f32 = 0.30000001192092896;
const BEND_SCALE: f32 = 0.8000000119209290;
const CONTROL_SCALE: f32 = 1.5000009536743164;
const AXIS_SCALE: f32 = 0.44999998807907104;
const ADJUST_SCALE: f32 = -3.0000009536743164;
const HEIGHT_SCALE: f32 = 0.40000098943710327;
const PAIR5_SCALE: f32 = 0.44999998807907104;
const MIN_DIFFERENCE_SEED: f32 = 10000.0;
const MIN_DIFFERENCE_SAMPLES: [usize; 7] = [0, 1, 2, 3, 12, 13, 14];

fn build_eye_pair(landmarks: &[Point]) -> Result<Vec<Point>, Refusal> {
    let mut mesh = vec![Point::new(0.0, 0.0); EYE_PAIR_VERTICES];
    append_eye(&mut mesh, landmarks, 36, 20, 24, 28, 32, false, 0)?;
    append_eye(&mut mesh, landmarks, 53, 37, 41, 45, 49, true, 60)?;
    Ok(mesh)
}

fn append_eye(
    mesh: &mut [Point],
    l: &[Point],
    p4: usize,
    q0: usize,
    q1: usize,
    q2: usize,
    q3: usize,
    mirror: bool,
    offset: usize,
) -> Result<(), Refusal> {
    let apex = l[p4];
    let c0 = l[q0];
    let c1 = l[q1];
    let c2 = l[q2];
    let c3 = l[q3];
    let a0 = if mirror {
        c2.sub(apex.sub(c2).scale(CURVE_SCALE))
    } else {
        c0.add(c0.sub(apex).scale(CURVE_SCALE))
    };
    let a2 = if mirror {
        c0.add(c0.sub(apex).scale(CURVE_SCALE))
    } else {
        c2.sub(apex.sub(c2).scale(CURVE_SCALE))
    };
    let a1 = c1.sub(apex.sub(c1).scale(BEND_SCALE));
    let a3 = c3.sub(apex.sub(c3).scale(BEND_SCALE));
    let tail0 = if mirror { c2 } else { c0 };
    let tail2 = if mirror { c0 } else { c2 };
    let a4 = tail0.add(tail0.sub(apex).scale(CONTROL_SCALE).scale(AXIS_SCALE));
    let a5 = tail2.add(tail2.sub(apex).scale(CONTROL_SCALE).scale(AXIS_SCALE));
    let curve1 = sample15(a0, a1, a2).ok_or(Refusal::NonFinite("eye_pair_spline_degenerate"))?;
    let curve3 = sample15(a0, a3, a2).ok_or(Refusal::NonFinite("eye_pair_spline_degenerate"))?;
    let curve2 = sample15(a4, adjust(&curve1, apex), a5)
        .ok_or(Refusal::NonFinite("eye_pair_spline_degenerate"))?;
    let curve4 = sample15(a4, adjust(&curve3, apex), a5)
        .ok_or(Refusal::NonFinite("eye_pair_spline_degenerate"))?;
    mesh[offset..offset + 15].copy_from_slice(&curve2);
    mesh[offset + 15..offset + 30].copy_from_slice(&curve1);
    mesh[offset + 30..offset + 45].copy_from_slice(&curve4);
    mesh[offset + 45..offset + 60].copy_from_slice(&curve3);
    Ok(())
}

fn adjust(curve: &[Point; 15], apex: Point) -> Point {
    let mid = curve[7];
    mid.add(apex.sub(mid).scale(AXIS_SCALE).scale(ADJUST_SCALE))
}

fn min_difference(center: Point, mesh: &[Point], group_a: usize, group_b: usize) -> f32 {
    let mut result = MIN_DIFFERENCE_SEED;
    for &i in &MIN_DIFFERENCE_SAMPLES {
        result = result.min(
            distance(center, mesh[group_a + i]) - distance(center, mesh[group_b + i]),
        );
    }
    result
}

/// 0x618E810 height stage: four 15-point rows shift by ±axis * minDiff * 0.4.
fn apply_height(mesh: &mut [Point], l: &[Point], degree_x: f32, degree_y: f32) -> (Point, Point) {
    let sx = degree_x * HEIGHT_SCALE;
    let sy = degree_y * HEIGHT_SCALE;
    let min_left = min_difference(l[36], mesh, 0, 15);
    let min_right = min_difference(l[53], mesh, 60, 75);
    let d = l[53].sub(l[36]);
    let length = d.length();
    if !(length > 0.0) || !length.is_finite() {
        return (l[36], l[53]);
    }
    let shift_left = Point::new(
        d.x * (-sx) * min_left / length,
        d.y * (-sx) * min_left / length,
    );
    let shift_right = Point::new(d.x * sy * min_right / length, d.y * sy * min_right / length);
    for i in 0..15 {
        mesh[15 + i] = mesh[15 + i].add(shift_left);
        mesh[45 + i] = mesh[45 + i].add(shift_left);
        mesh[75 + i] = mesh[75 + i].add(shift_right);
        mesh[105 + i] = mesh[105 + i].add(shift_right);
    }
    (l[36].add(shift_left), l[53].add(shift_right))
}

/// 0x618EB50 pair5 stage: rows 15/45 follow +x, rows 75/105 follow +y.
fn apply_pair5(
    mesh: &mut [Point],
    l: &[Point],
    apex_left: Point,
    apex_right: Point,
    degree_x: f32,
    degree_y: f32,
) -> bool {
    let axis = l[58].sub(l[54]);
    let length = axis.length();
    if !(length > 0.0) || !length.is_finite() {
        return false;
    }
    let unit = axis.scale(1.0 / length);
    let min_left = min_difference(apex_left, mesh, 0, 15);
    let min_right = min_difference(apex_right, mesh, 60, 75);
    let shift_left = unit.scale(min_left * (degree_x * PAIR5_SCALE));
    let shift_right = unit.scale(min_right * (degree_y * PAIR5_SCALE));
    for i in 0..15 {
        mesh[15 + i] = mesh[15 + i].add(shift_left);
        mesh[45 + i] = mesh[45 + i].add(shift_left);
        mesh[75 + i] = mesh[75 + i].add(shift_right);
        mesh[105 + i] = mesh[105 + i].add(shift_right);
    }
    true
}

// ---------------------------------------------------------------------------
// Index buffers. Only the recovered 6-entry quad EBO is embedded; every mesh
// path takes the same-job index buffer explicitly and refuses when absent.
// ---------------------------------------------------------------------------

const EBO_14: &str = "TOPO_492_INDEX_TEMPLATE_NOT_EMBEDDED";
const EBO_15: &str = "PAIR2_RING_INDEX_BUFFER_NOT_CAPTURED";
const EBO_16: &str = "PAIR5_TOPOLOGY_492_INDEX_TEMPLATE_NOT_EMBEDDED";
const EBO_19: &str = "0x60F2B60";
const EBO_20: &str = "BROW_168_INDEX_BUFFER_NOT_CAPTURED";
const EBO_23: &str = "TYPE1018_INDEX_BUFFER_NOT_CAPTURED_0x7080940";

pub const fn ebo_rva(pf: u32) -> Option<&'static str> {
    match pf {
        8015 | 8016 => Some(EBO_14),
        8017 | 8018 => Some(EBO_15),
        8021 | 8022 => Some(EBO_16),
        30073 | 30079 => Some(EBO_19),
        8001 | 8002 | 8005 | 8006 | 8009 | 8010 => Some(EBO_20),
        44124 | 48019 | 49221 | 41922 => Some(EBO_23),
        _ => None,
    }
}

pub fn accept_indices(
    pf: u32,
    indices: Option<&[u16]>,
    vertices: usize,
) -> Result<usize, Refusal> {
    let Some(indices) = indices else {
        return Err(Refusal::MissingEbo(ebo_rva(pf).unwrap_or("index_buffer_not_captured")));
    };
    if indices.is_empty() || indices.len() % 3 != 0 {
        return Err(Refusal::MissingEbo(ebo_rva(pf).unwrap_or("index_buffer_not_captured")));
    }
    if indices.iter().any(|&index| index as usize >= vertices) {
        return Err(Refusal::OutOfRange("index_out_of_mesh_range"));
    }
    Ok(indices.len() / 3)
}

// ---------------------------------------------------------------------------
// Row 11 - eye size PF8011/PF8012 (pair 0).
// ---------------------------------------------------------------------------

pub fn row11_pupil_size(
    landmarks: &[Point],
    width: i32,
    height: i32,
    scale: Point,
    ui_left: f32,
    ui_right: f32,
    factor_left: f32,
    factor_right: f32,
) -> Result<Output, Refusal> {
    let frame = eye_frame(landmarks, width, height, scale)?;
    if !factor_left.is_finite() || !factor_right.is_finite() {
        return Err(Refusal::NonFinite("eye_factor_non_finite"));
    }
    let degree_left = map(control(ui_left)?, &EYE_SIZE_PARAM, &EYE_SIZE_DEGREE)?;
    let degree_right = map(control(ui_right)?, &EYE_SIZE_PARAM, &EYE_SIZE_DEGREE)?;
    if degree_left == 0.0 && degree_right == 0.0 {
        return Ok(Output {
            degrees: vec![degree_left, degree_right],
            noop: true,
            open: vec!["both_degrees_zero_no_emit"],
            ..Output::default()
        });
    }
    let aspect = height as f32 / width as f32;
    let mut quads = Vec::with_capacity(2);
    for side in 0..2 {
        let degree = if side == 0 { degree_left } else { degree_right };
        let radius_factor = if side == 0 { scale.x } else { scale.y };
        let radius = frame.distance * degree.abs() * 1.5 * radius_factor;
        if !(radius > 0.0) {
            continue;
        }
        let center = if side == 0 { frame.center_left } else { frame.center_right };
        quads.push(build_quad(center, radius, width, height, aspect, false));
    }
    Ok(Output {
        degrees: vec![degree_left, degree_right],
        noop: quads.is_empty(),
        quads,
        scalars: vec![factor_left, factor_right],
        open: vec!["VBO_EBO_GL_CONSUMER_OPEN"],
        ..Output::default()
    })
}

// ---------------------------------------------------------------------------
// Row 12 - eye height PF8013/PF8014 (pair 1, FactorList[1]).
// Real zero branch: the PE continues and builds a zero-radius quad, which
// draws nothing. The C# wrapper's throw is that layer's own validation, not
// the engine's behaviour, so this port returns a clean no-op.
// ---------------------------------------------------------------------------

pub const ROW12_ZERO_QUAD_NATIVE: &str = "PF8013/PF8014 requires two eye quads.";
pub const ROW12_FACTOR_LIST1_OPEN: &str = "FACTORLIST1_VALUE_OPEN";

pub fn row12_eye_height(
    landmarks: &[Point],
    width: i32,
    height: i32,
    scale: Point,
    ui_left: f32,
    ui_right: f32,
    factor: f32,
) -> Result<Output, Refusal> {
    let frame = eye_frame(landmarks, width, height, scale)?;
    if !factor.is_finite() {
        return Err(Refusal::NonFinite("eyerefine_factorlist1_non_finite"));
    }
    let degree_left = map_unit(control(ui_left)?)?;
    let degree_right = map_unit(control(ui_right)?)?;
    let aspect = height as f32 / width as f32;
    let mut quads = Vec::with_capacity(2);
    for side in 0..2 {
        let degree = if side == 0 { degree_left } else { degree_right };
        let radius_factor = if side == 0 { scale.x } else { scale.y };
        let radius = frame.distance * degree.abs() * 1.5 * radius_factor;
        if !(radius > 0.0) {
            continue;
        }
        let center = if side == 0 { frame.center_left } else { frame.center_right };
        quads.push(build_quad(center, radius, width, height, aspect, true));
    }
    if quads.len() != 2 {
        return Ok(Output {
            degrees: vec![degree_left, degree_right],
            quads: Vec::new(),
            scalars: vec![factor],
            noop: true,
            open: vec!["ZERO_RADIUS_QUAD_DRAWS_NOTHING", ROW12_FACTOR_LIST1_OPEN],
            ..Output::default()
        });
    }
    Ok(Output {
        degrees: vec![degree_left, degree_right],
        quads,
        scalars: vec![factor],
        open: vec![ROW12_FACTOR_LIST1_OPEN, "VBO_UV_PIXEL_OVER_SIZE_DIVERGENCE_OPEN"],
        ..Output::default()
    })
}

// ---------------------------------------------------------------------------
// Row 13 - eye width PF8019/PF8020 (pair 4, FactorList[4] = 0.6).
// ---------------------------------------------------------------------------

pub const ROW13_FACTOR_LIST4: f32 = 0.6;
const EDGE_SCALE: f32 = -0.4;
const WIDTH_RADIUS_SCALE: f32 = 1.5;

pub fn row13_eye_width(
    landmarks: &[Point],
    width: i32,
    height: i32,
    scale: Point,
    ui_left: f32,
    ui_right: f32,
    render_factor: f32,
) -> Result<Output, Refusal> {
    if landmarks.len() != 194 {
        return Err(Refusal::MissingLandmarks);
    }
    if !all_finite(landmarks) || !scale.finite() || !render_factor.is_finite() {
        return Err(Refusal::NonFinite("landmark_scale_factor_non_finite"));
    }
    if width <= 0 || height <= 0 {
        return Err(Refusal::OutOfRange("image_extent_invalid"));
    }
    let degree_left = map_unit(control(ui_left)?)?;
    let degree_right = map_unit(control(ui_right)?)?;
    if degree_left == 0.0 && degree_right == 0.0 {
        return Ok(Output {
            degrees: vec![degree_left, degree_right],
            noop: true,
            open: vec!["PF8019_PF8020_BOTH_DEGREES_ZERO_NO_EMIT"],
            ..Output::default()
        });
    }
    let w = width as f32;
    let h = height as f32;
    let inverse_diagonal = 1.0 / (w * w + h * h).sqrt();
    let inverse_min = (1.0 / w).max(1.0 / h);
    let aspect = h / w;
    let eye_distance = if scale.x > 0.01 {
        distance(landmarks[18], landmarks[22]) * inverse_diagonal
    } else {
        distance(landmarks[36], landmarks[40]) * inverse_diagonal
    };
    let center_left = Point::new(landmarks[26].x / w, landmarks[26].y / h);
    let center_right = Point::new(landmarks[44].x / w, landmarks[44].y / h);

    let a0 = distance(landmarks[27], landmarks[35]);
    let a1 = distance(landmarks[49], landmarks[53]);
    let b0 = distance(landmarks[31], landmarks[35]);
    let b1 = distance(landmarks[45], landmarks[53]);
    let shift_left = (inverse_min * a0).min(inverse_min * b0);
    let shift_right = (inverse_min * a1).min(inverse_min * b1);
    let norm_left = (inverse_diagonal * a0).min(inverse_diagonal * b0);
    let norm_right = (inverse_diagonal * a1).min(inverse_diagonal * b1);
    let amount_left =
        scale.x * (degree_left.abs() * (norm_left * EDGE_SCALE + eye_distance) * WIDTH_RADIUS_SCALE);
    let amount_right = scale.y
        * (degree_right.abs() * (norm_right * EDGE_SCALE + eye_distance) * WIDTH_RADIUS_SCALE);

    let dx = landmarks[125].x - landmarks[190].x;
    let dy = landmarks[125].y - landmarks[190].y;
    let length = (dx * dx + dy * dy).sqrt();
    if !(length > 0.0) || !length.is_finite() {
        return Err(Refusal::NonFinite("eye_width_direction_degenerate"));
    }
    let axis_x = clamp(dy / length, -1.0, 1.0);
    let mut angle = axis_x.acos();
    if dx < 0.0 {
        angle = -angle;
    }
    let axis = Point::new(axis_x, angle.sin());

    let mut quads = Vec::with_capacity(2);
    for side in 0..2 {
        let shift = if side == 0 { shift_left } else { shift_right };
        let amount = if side == 0 { amount_left } else { amount_right };
        let center = if side == 0 { center_left } else { center_right };
        if !(amount > 0.0) {
            continue;
        }
        let base_uv = Point::new(center.x - shift * axis.x, center.y + shift * axis.y);
        let amount_y = amount / aspect;
        let uv = [
            Point::new(base_uv.x - amount, base_uv.y + amount_y),
            Point::new(base_uv.x + amount, base_uv.y + amount_y),
            Point::new(base_uv.x - amount, base_uv.y - amount_y),
            Point::new(base_uv.x + amount, base_uv.y - amount_y),
        ];
        let pixels = [
            Point::new(uv[0].x * w, uv[0].y * h),
            Point::new(uv[1].x * w, uv[1].y * h),
            Point::new(uv[2].x * w, uv[2].y * h),
            Point::new(uv[3].x * w, uv[3].y * h),
        ];
        let bounds = quad_bounds(
            pixels[0].x.min(pixels[1].x).min(pixels[2].x).min(pixels[3].x),
            pixels[0].x.max(pixels[1].x).max(pixels[2].x).max(pixels[3].x),
            pixels[0].y.min(pixels[1].y).min(pixels[2].y).min(pixels[3].y),
            pixels[0].y.max(pixels[1].y).max(pixels[2].y).max(pixels[3].y),
            width,
            height,
        );
        quads.push(Quad { pixels, uv, center, radius: amount, bounds });
    }
    Ok(Output {
        degrees: vec![degree_left, degree_right],
        noop: quads.is_empty(),
        quads,
        scalars: vec![amount_left, amount_right, render_factor * 0.5],
        open: vec!["VBO_EBO_GL_CONSUMER_OPEN"],
        ..Output::default()
    })
}

// ---------------------------------------------------------------------------
// Row 14 - eye distance PF8015/PF8016 (pair 3) + pair5 input.
// ---------------------------------------------------------------------------

pub fn row14_eye_distance(
    landmarks: &[Point],
    width: i32,
    height: i32,
    scale: Point,
    ui_left: f32,
    ui_right: f32,
    pair5_left: f32,
    pair5_right: f32,
    indices: Option<&[u16]>,
) -> Result<Output, Refusal> {
    if landmarks.len() != 194 {
        return Err(Refusal::MissingLandmarks);
    }
    if !all_finite(landmarks) || !scale.finite() {
        return Err(Refusal::NonFinite("landmark_or_scale_non_finite"));
    }
    if width <= 0 || height <= 0 {
        return Err(Refusal::OutOfRange("image_extent_invalid"));
    }
    let degree_left = map_unit(control(ui_left)?)?;
    let degree_right = map_unit(control(ui_right)?)?;
    if !pair5_left.is_finite() || !pair5_right.is_finite() {
        return Err(Refusal::NonFinite("pair5_non_finite"));
    }
    let triangles = accept_indices(8015, indices, EYE_PAIR_VERTICES)?;
    let open = vec![
        "A_AS_SOURCE_UV_B_AS_TARGET_UV_OPEN",
        "TOPOLOGY_492_INDEX_TEMPLATE_NOT_EMBEDDED",
        "RAW_COMPUTE_ZERO_MESH_DEFECT_USES_WRAPPED_IDENTITY",
    ];
    let source = build_eye_pair(landmarks)?;
    if degree_left == 0.0 && degree_right == 0.0 && pair5_left == 0.0 && pair5_right == 0.0 {
        return Ok(Output {
            degrees: vec![degree_left, degree_right],
            source: source.clone(),
            target: source,
            scalars: vec![triangles as f32],
            open,
            noop: true,
            ..Output::default()
        });
    }
    let mut target = source.clone();
    let (apex_left, apex_right) = apply_height(&mut target, landmarks, degree_left, degree_right);
    if !apply_pair5(&mut target, landmarks, apex_left, apex_right, pair5_left, pair5_right) {
        return Err(Refusal::NonFinite("eye_axis_degenerate"));
    }
    Ok(Output {
        degrees: vec![degree_left, degree_right],
        source,
        target,
        scalars: vec![pair5_left, pair5_right, triangles as f32],
        open,
        noop: false,
        ..Output::default()
    })
}

// ---------------------------------------------------------------------------
// Row 15 - tilt. EyeRefine pair2 (0x4C8BE90) is closed; FacialRefine PF8007/
// PF8008 (0x4C8E9F0) is OPEN and only resolves degrees.
// ---------------------------------------------------------------------------

pub const ROW15_FACIAL_REFINE_GAP: &str = "0x4C8E9F0";

pub fn row15_eye_tilt(
    landmarks: &[Point],
    ui_left: f32,
    ui_right: f32,
    source_left: &[Point],
    source_right: &[Point],
    factor: f32,
    angle_scale: Point,
    geometry_scale: Point,
    center_left: Point,
    center_right: Point,
    indices: Option<&[u16]>,
) -> Result<Output, Refusal> {
    if landmarks.len() != 194 {
        return Err(Refusal::MissingLandmarks);
    }
    let degree_left = signed(ui_left)?;
    let degree_right = signed(ui_right)?;
    if !factor.is_finite()
        || !angle_scale.finite()
        || !geometry_scale.finite()
        || !center_left.finite()
        || !center_right.finite()
    {
        return Err(Refusal::NonFinite("tilt_live_scalars_non_finite"));
    }
    if source_left.len() != 20 || source_right.len() != 20 {
        return Err(Refusal::MissingMesh("pair2_20_point_sources_required"));
    }
    if !all_finite(source_left) || !all_finite(source_right) {
        return Err(Refusal::NonFinite("pair2_source_non_finite"));
    }
    let triangles = accept_indices(8017, indices, 20)?;
    let open = vec!["FACIALREFINE_TILT_UNIQUE_GAP_CONTROL_TO_RENDER_EDGE rva=0x4C8E9F0"];
    let mut target_left = source_left.to_vec();
    let mut target_right = source_right.to_vec();
    let zero = degree_left == 0.0 && degree_right == 0.0;
    if !zero {
        let angle_left = (-degree_left) * factor * angle_scale.x;
        let angle_right = degree_right * factor * angle_scale.y;
        rotate_ring(&mut target_left, center_left, angle_left, geometry_scale);
        rotate_ring(&mut target_right, center_right, angle_right, geometry_scale);
    }
    let mut source = Vec::with_capacity(40);
    source.extend_from_slice(source_left);
    source.extend_from_slice(source_right);
    let mut target = Vec::with_capacity(40);
    target.extend_from_slice(&target_left);
    target.extend_from_slice(&target_right);
    Ok(Output {
        degrees: vec![degree_left, degree_right],
        source,
        target,
        scalars: vec![
            (-degree_left) * factor * angle_scale.x,
            degree_right * factor * angle_scale.y,
            triangles as f32,
        ],
        open,
        noop: zero,
        ..Output::default()
    })
}

fn rotate_ring(target: &mut [Point], center: Point, angle: f32, geometry_scale: Point) {
    for point in target.iter_mut() {
        *point = Point::new(point.x * geometry_scale.x, point.y * geometry_scale.y);
    }
    let cosine = angle.cos();
    let sine = angle.sin();
    for point in target.iter_mut().take(9).skip(1) {
        let dx = point.x - center.x;
        let dy = point.y - center.y;
        *point = Point::new(
            dx * cosine + dy * sine + center.x,
            dy * cosine - dx * sine + center.y,
        );
    }
}

pub fn row15_facial_refine(ui_left: f32, ui_right: f32) -> Result<OpenDegrees, Refusal> {
    let degree_left = signed(ui_left)?;
    let degree_right = signed(ui_right)?;
    Ok(OpenDegrees {
        degrees: vec![degree_left, degree_right],
        refusal: Refusal::OpenGeometry(
            "PF8007_PF8008_PART14_SLOT4_5_MESH_PRODUCER_OPEN rva=0x4C8E9F0",
        ),
    })
}

// ---------------------------------------------------------------------------
// Row 16 - position. EyeRefine pair5 PF8021/PF8022 (0x4C8B650) is closed;
// FacialRefine PF8003/PF8004 part 14 is OPEN.
// ---------------------------------------------------------------------------

pub fn row16_eye_position(
    landmarks: &[Point],
    ui_left: f32,
    ui_right: f32,
    pair5_left: f32,
    pair5_right: f32,
    indices: Option<&[u16]>,
) -> Result<Output, Refusal> {
    if landmarks.len() != 194 {
        return Err(Refusal::MissingLandmarks);
    }
    let degree_left = signed(ui_left)?;
    let degree_right = signed(ui_right)?;
    if !all_finite(landmarks) || !pair5_left.is_finite() || !pair5_right.is_finite() {
        return Err(Refusal::NonFinite("landmark_or_pair5_non_finite"));
    }
    let triangles = accept_indices(8021, indices, EYE_PAIR_VERTICES)?;
    let open = vec![
        "OPEN: FacialRefine eyebrow-height mesh producer missing",
        "PAIR5_TOPOLOGY_492_INDEX_TEMPLATE_NOT_EMBEDDED",
        "DESCRIPTOR_RANGE_CONFLICT_8021_8022: un-inverted in recovered MapUi",
        "RAW_COMPUTE_ZERO_MESH_DEFECT_USES_WRAPPED_IDENTITY",
    ];
    let source = build_eye_pair(landmarks)?;
    if pair5_left == 0.0 && pair5_right == 0.0 {
        return Ok(Output {
            degrees: vec![degree_left, degree_right],
            source: source.clone(),
            target: source,
            scalars: vec![pair5_left, pair5_right, triangles as f32],
            open,
            noop: true,
            ..Output::default()
        });
    }
    let mut target = source.clone();
    if !apply_pair5(&mut target, landmarks, landmarks[36], landmarks[53], pair5_left, pair5_right) {
        return Err(Refusal::NonFinite("eye_axis_degenerate"));
    }
    Ok(Output {
        degrees: vec![degree_left, degree_right],
        source,
        target,
        scalars: vec![pair5_left, pair5_right, triangles as f32],
        open,
        noop: false,
        ..Output::default()
    })
}

pub fn row16_facial_refine(ui_left: f32, ui_right: f32) -> Result<OpenDegrees, Refusal> {
    let degree_left = signed(ui_left)?;
    let degree_right = signed(ui_right)?;
    Ok(OpenDegrees {
        degrees: vec![degree_left, degree_right],
        refusal: Refusal::OpenGeometry(
            "PF8003_PF8004_PART14_SLOT0_1_MESH_PRODUCER_OPEN",
        ),
    })
}

// ---------------------------------------------------------------------------
// Rows 17/18 - Type1010 corners (0x60F03F6 / render 0x4C96A60). Degrees are
// recovered; the ParamList -> Local_Warp mesh consumer is OPEN.
// ---------------------------------------------------------------------------

pub const TYPE1010_PARAM_LIST_LEN: usize = 28;

pub fn corners_degree_vector(
    priority: u32,
    ui_left: f32,
    ui_right: f32,
    param_list: Option<&[f32]>,
) -> Result<OpenDegrees, Refusal> {
    if priority != 17 && priority != 18 {
        return Err(Refusal::UnsupportedRoute(priority));
    }
    let control_left = control(ui_left)?;
    let control_right = control(ui_right)?;
    if let Some(list) = param_list {
        if list.len() != TYPE1010_PARAM_LIST_LEN {
            return Err(Refusal::OutOfRange("type1010_paramlist_28_required"));
        }
    }
    let (slot_left, slot_right) = if priority == 17 { (0, 1) } else { (2, 3) };
    let mut degrees = vec![0.0f32; 8];
    degrees[slot_left] = map(control_left, &PAIR_PARAM, &NEG_DEGREE)?;
    degrees[slot_right] = map(control_right, &PAIR_PARAM, &NEG_DEGREE)?;
    degrees[4 + slot_left] = map(control_left, &PAIR_PARAM, &POS_DEGREE)?;
    degrees[4 + slot_right] = map(control_right, &PAIR_PARAM, &POS_DEGREE)?;
    Ok(OpenDegrees {
        degrees,
        refusal: Refusal::OpenGeometry(
            "OPEN_TYP1010_PARAMLIST_TO_LOCAL_WARP rva=0x4C96A60 factory=0x60F03F6",
        ),
    })
}

// ---------------------------------------------------------------------------
// Row 19 - lower eyelid PF30073/PF30079 (Type1011 parts 6/7).
// Prepared 100-point payload only; 194 input is never upsampled.
// ---------------------------------------------------------------------------

const EYELID_PREPARED: usize = 100;
const EYELID_PARAM_COUNT: usize = 6;
const EYELID_AMP_SCALE: f32 = 0.6000000238418579;

pub fn row19_lower_eyelid(
    prepared: &[Point],
    param_list: Option<&[f32]>,
    center: Point,
    ui_left: f32,
    ui_right: f32,
    indices: Option<&[u16]>,
) -> Result<Output, Refusal> {
    if prepared.len() != EYELID_PREPARED {
        return Err(Refusal::MissingPreparedMesh);
    }
    if !all_finite(prepared) || !center.finite() {
        return Err(Refusal::NonFinite("prepared_or_center_non_finite"));
    }
    let Some(param_list) = param_list else {
        return Err(Refusal::MissingParamList);
    };
    if param_list.len() != EYELID_PARAM_COUNT {
        return Err(Refusal::MissingParamList);
    }
    if !all_finite_f32(param_list) {
        return Err(Refusal::NonFinite("paramlist_non_finite"));
    }
    let control_left = control(ui_left)?;
    let control_right = control(ui_right)?;
    let triangles = accept_indices(30073, indices, EYELID_PREPARED)?;

    let mut weights = [0.0f32; 14];
    weights[8..14].copy_from_slice(param_list);
    let mut deformed = prepared.to_vec();
    let mut degrees = vec![0.0f32; 4];
    for part in 0..2 {
        let range = if part == 0 { &NEG_DEGREE } else { &POS_DEGREE };
        let degree_left = map(control_left, &PAIR_PARAM, range)?;
        let degree_right = map(control_right, &PAIR_PARAM, range)?;
        degrees[part * 2] = degree_left;
        degrees[part * 2 + 1] = degree_right;
        let amp_x = degree_left * EYELID_AMP_SCALE * center.x;
        let amp_y = degree_right * EYELID_AMP_SCALE * center.y;
        if amp_x == 0.0 && amp_y == 0.0 {
            continue;
        }
        for i in 0..14 {
            let a = deformed[14 + i].add(
                deformed[27 + i].sub(deformed[14 + i]).scale(weights[i] * amp_x),
            );
            deformed[14 + i] = a;
            let b = deformed[63 + i].add(
                deformed[79 + i]
                    .sub(deformed[63 + i])
                    .scale(weights[13 - i] * amp_y),
            );
            deformed[63 + i] = b;
        }
    }
    let noop = deformed == prepared;
    Ok(Output {
        degrees,
        source: prepared.to_vec(),
        target: deformed,
        scalars: vec![EYELID_AMP_SCALE, triangles as f32],
        open: vec![
            "PASS_BITWISE_PREPARED_AND_DEFORMED_SYNTHETIC_FACE; EBO_CONTRACT_ORACLE_ONLY",
            "CENTER_IS_FACE_0x18AC_0x18B0_NOT_A_CENTROID",
        ],
        noop,
        ..Output::default()
    })
}

// ---------------------------------------------------------------------------
// Rows 20/21/22 - brow thickness / distance / shape (Type1005, 168 points).
// Stages accumulate in slot order 6/7, 2/3, 8/9 over the same mesh.
// ---------------------------------------------------------------------------

const BROW_MESH: usize = 168;
const BROW_BAND: usize = 28;
const FACE_THICKNESS_FLOATS: usize = 1053;
const FACE_THICKNESS_X: usize = 0x1070 / 4;
const FACE_THICKNESS_Y: usize = 0x106C / 4;
const FACE_THICKNESS_A: usize = 0xE20 / 4;
const FACE_THICKNESS_B: usize = 0xE1C / 4;
const FACE_THICKNESS_C: usize = 0xAD8 / 4;
const FACE_THICKNESS_D: usize = 0xAD4 / 4;
const BROW_THICKNESS_CONSTANT: f32 = 0.40000098943710327;
const BROW_DISTANCE_CONSTANT: f32 = -0.3500001132488251;
const BROW_AXIS_A: usize = 0x94C / 4;
const BROW_AXIS_B: usize = 0x99C / 4;

fn brow_normal(face: &[f32]) -> Result<Point, Refusal> {
    if face.len() < FACE_THICKNESS_FLOATS {
        return Err(Refusal::MissingFaceFields);
    }
    let x0 = face[FACE_THICKNESS_X] + face[FACE_THICKNESS_A]
        - 2.0 * face[FACE_THICKNESS_C];
    let y0 = face[FACE_THICKNESS_Y] + face[FACE_THICKNESS_B]
        - 2.0 * face[FACE_THICKNESS_D];
    let norm = (x0 * x0 + y0 * y0).sqrt();
    if !(norm > 0.0) || !norm.is_finite() {
        return Err(Refusal::MissingFaceFields);
    }
    Ok(Point::new(y0 / norm, x0 / norm))
}

fn brow_min_edge(guide: &[Point], start: usize) -> f32 {
    let mut value = 100000.0f32;
    for i in start..BROW_BAND {
        value = value.min(distance(guide[112 + i], guide[140 + i]));
        value = value.min(distance(guide[28 + i], guide[56 + i]));
    }
    value
}

fn brow_fallback(guide: &[Point]) -> f32 {
    let mut value = 100000.0f32;
    for k in 5..=9 {
        value = value.min(distance(guide[112 - k], guide[84 + k]) * 0.5);
        value = value.min(distance(guide[28 - k], guide[k]) * 0.5);
    }
    value * 0.5
}

pub fn row20_brow_thickness(
    mesh: &[Point],
    guide: &[Point],
    face: &[f32],
    ui_left: f32,
    ui_right: f32,
    indices: Option<&[u16]>,
) -> Result<Output, Refusal> {
    if mesh.len() != BROW_MESH || guide.len() != BROW_MESH {
        return Err(Refusal::MissingMesh("brow_mesh_and_guide_168_required"));
    }
    if !all_finite(mesh) || !all_finite(guide) {
        return Err(Refusal::NonFinite("mesh_or_guide_non_finite"));
    }
    let normal = brow_normal(face)?;
    let degree_left = map_unit(control(ui_left)?)?;
    let degree_right = map_unit(control(ui_right)?)?;
    let triangles = accept_indices(8001, indices, BROW_MESH)?;

    let radius_left =
        if degree_left > 0.0 { brow_min_edge(guide, 15) } else { brow_fallback(guide) };
    let radius_right =
        if degree_right > 0.0 { brow_min_edge(guide, 15) } else { brow_fallback(guide) };
    let delta_left = normal.scale(degree_left).scale(radius_left * BROW_THICKNESS_CONSTANT);
    let delta_right = normal.scale(degree_right).scale(radius_right * BROW_THICKNESS_CONSTANT);
    let mut target = mesh.to_vec();
    for i in 0..14 {
        target[i] = target[i].add(delta_left);
        target[28 + i] = target[28 + i].add(delta_left);
        target[84 + i] = target[84 + i].add(delta_right);
        target[112 + i] = target[112 + i].add(delta_right);
    }
    for i in 15..28 {
        target[i] = target[i].sub(delta_left);
        target[28 + i] = target[28 + i].sub(delta_left);
        target[84 + i] = target[84 + i].sub(delta_right);
        target[112 + i] = target[112 + i].sub(delta_right);
    }
    Ok(Output {
        degrees: vec![degree_left, degree_right],
        source: mesh.to_vec(),
        target,
        scalars: vec![
            radius_left,
            radius_right,
            delta_left.length(),
            delta_right.length(),
            triangles as f32,
        ],
        open: vec!["BROW_UV_AND_UNIFORMS_OPEN", "GUIDE_194_TO_168_PRODUCER_OPEN"],
        noop: delta_left.length() == 0.0 && delta_right.length() == 0.0,
        ..Output::default()
    })
}

pub fn row21_brow_distance(
    mesh: &[Point],
    raw_face: &[f32],
    ui_left: f32,
    ui_right: f32,
    indices: Option<&[u16]>,
) -> Result<Output, Refusal> {
    if mesh.len() != BROW_MESH {
        return Err(Refusal::MissingMesh("brow_mesh_168_required"));
    }
    if !all_finite(mesh) {
        return Err(Refusal::NonFinite("mesh_non_finite"));
    }
    if raw_face.len() < BROW_AXIS_B * 2 + 2 {
        return Err(Refusal::MissingFaceFields);
    }
    if !all_finite_f32(raw_face) {
        return Err(Refusal::NonFinite("raw_face_non_finite"));
    }
    let degree_left = map_unit(control(ui_left)?)?;
    let degree_right = map_unit(control(ui_right)?)?;
    let triangles = accept_indices(8005, indices, BROW_MESH)?;

    let direction = Point::new(
        raw_face[BROW_AXIS_A * 2] - raw_face[BROW_AXIS_B * 2],
        raw_face[BROW_AXIS_A * 2 + 1] - raw_face[BROW_AXIS_B * 2 + 1],
    );
    let length = direction.length();
    if !(length > 1e-12) || !length.is_finite() {
        return Err(Refusal::NonFinite("eyebrow_distance_axis_degenerate"));
    }
    let axis = direction.scale(1.0 / length);
    let left = axis
        .scale(-BROW_DISTANCE_CONSTANT)
        .scale(degree_left * BROW_DISTANCE_CONSTANT);
    let right = axis
        .scale(BROW_DISTANCE_CONSTANT)
        .scale(degree_right * BROW_DISTANCE_CONSTANT);
    let mut target = mesh.to_vec();
    for i in 0..BROW_BAND {
        target[i] = target[i].add(left);
        target[28 + i] = target[28 + i].add(left);
        target[84 + i] = target[84 + i].add(right);
        target[112 + i] = target[112 + i].add(right);
    }
    let mut mids = vec![Point::new(0.0, 0.0); BROW_BAND * 2];
    for i in 0..BROW_BAND {
        mids[i] = Point::new(
            (target[i].x + target[28 + i].x) * 0.5,
            (target[i].y + target[28 + i].y) * 0.5,
        );
        mids[BROW_BAND + i] = Point::new(
            (target[84 + i].x + target[112 + i].x) * 0.5,
            (target[84 + i].y + target[112 + i].y) * 0.5,
        );
    }
    Ok(Output {
        degrees: vec![degree_left, degree_right],
        source: mesh.to_vec(),
        target,
        mids,
        scalars: vec![left.x, left.y, right.x, right.y, triangles as f32],
        open: vec!["BROW_UV_AND_UNIFORMS_OPEN"],
        noop: left.length() == 0.0 && right.length() == 0.0,
        ..Output::default()
    })
}

const BROW_SHAPE_WEIGHTS: [f32; 15] = [
    -0.20000030,
    -0.20000030,
    -0.20000030,
    0.0,
    0.10000009,
    0.30000013,
    0.5,
    0.69999999,
    0.80000001,
    0.69999999,
    0.5,
    0.30000013,
    0.10000009,
    -0.20000030,
    -0.20000030,
];
const BROW_SHAPE_NEGATIVE: f32 = 0.6500099897384644;
const BROW_SHAPE_POSITIVE: f32 = 1.1000009775161743;

pub fn row22_brow_shape(
    mesh: &[Point],
    guide: &[Point],
    face: &[f32],
    ui_left: f32,
    ui_right: f32,
    indices: Option<&[u16]>,
) -> Result<Output, Refusal> {
    if mesh.len() != BROW_MESH || guide.len() != BROW_MESH {
        return Err(Refusal::MissingGuide);
    }
    if !all_finite(mesh) || !all_finite(guide) {
        return Err(Refusal::NonFinite("mesh_or_guide_non_finite"));
    }
    let normal = brow_normal(face)?;
    let degree_left = map_unit(control(ui_left)?)?;
    let degree_right = map_unit(control(ui_right)?)?;
    let triangles = accept_indices(8009, indices, BROW_MESH)?;

    let mut min_distance = 100000.0f32;
    for i in 0..BROW_BAND {
        min_distance = min_distance.min(distance(guide[56 + i], guide[28 + i]));
        min_distance = min_distance.min(distance(guide[140 + i], guide[112 + i]));
    }
    let mut target = mesh.to_vec();
    for i in 0..BROW_BAND {
        let weight = if i < 14 { BROW_SHAPE_WEIGHTS[i] } else { BROW_SHAPE_WEIGHTS[28 - i] };
        let magnitude = weight * min_distance;
        let scale_left = shape_factor(degree_left) * magnitude * degree_left;
        let scale_right = shape_factor(degree_right) * magnitude * degree_right;
        target[i] = target[i].add(normal.scale(scale_left));
        target[28 + i] = target[28 + i].add(normal.scale(scale_left));
        target[84 + i] = target[84 + i].add(normal.scale(scale_right));
        target[112 + i] = target[112 + i].add(normal.scale(scale_right));
    }
    Ok(Output {
        degrees: vec![degree_left, degree_right],
        source: mesh.to_vec(),
        target,
        scalars: vec![min_distance, triangles as f32],
        open: vec!["BROW_UV_AND_UNIFORMS_OPEN"],
        noop: degree_left == 0.0 && degree_right == 0.0,
        ..Output::default()
    })
}

fn shape_factor(degree: f32) -> f32 {
    if degree < 0.0 { BROW_SHAPE_NEGATIVE } else { BROW_SHAPE_POSITIVE }
}

// ---------------------------------------------------------------------------
// Rows 23/24 - brow head / tail length (Type1018 parts 15/16).
// Producer closed; the 78-target numeric consumer 0x707ED50 is OPEN.
// ---------------------------------------------------------------------------

const BROW_GUIDE_COUNT: usize = 24;
const BROW_TARGETS: usize = 78;
const BROW_LENGTH_AMOUNT: f32 = 0.8000000119209290;
const BROW_HEAD_LONG_FACTOR: f32 = 1.1000009775161743;
pub const BROW_LENGTH_CONSUMER_OPEN: &str =
    "BROW_LENGTH_CONSUMER_0x707ED50_NODE_PLUS_0x50_OPEN";

pub fn brow_length(
    priority: u32,
    kind: BrowPartKind,
    inputs: BrowLengthInputs<'_>,
    ui_left: f32,
    ui_right: f32,
    head_left: f32,
    head_right: f32,
    indices: Option<&[u16]>,
) -> Result<Output, Refusal> {
    if priority != 23 && priority != 24 {
        return Err(Refusal::UnsupportedRoute(priority));
    }
    if inputs.a30.len() < BROW_GUIDE_COUNT
        || inputs.centerline_p24.len() < BROW_GUIDE_COUNT
        || inputs.centerline_f24.len() < BROW_GUIDE_COUNT
    {
        return Err(Refusal::MissingGuide);
    }
    if inputs.param_list24.len() != 24 {
        return Err(Refusal::MissingParamList);
    }
    if inputs.factor_list48.len() != 48 {
        return Err(Refusal::MissingFactorList);
    }
    if inputs.targets78.len() != BROW_TARGETS {
        return Err(Refusal::MissingMesh("type1018_targets78_required"));
    }
    if !all_finite(inputs.a30)
        || !all_finite(inputs.centerline_p24)
        || !all_finite(inputs.centerline_f24)
        || !all_finite(inputs.targets78)
        || !all_finite_f32(inputs.param_list24)
        || !all_finite_f32(inputs.factor_list48)
        || !inputs.scale.is_finite()
        || inputs.scale < 0.0
    {
        return Err(Refusal::NonFinite("type1018_inputs_non_finite_or_negative_scale"));
    }
    let control_left = control(ui_left)?;
    let control_right = control(ui_right)?;
    if !head_left.is_finite() || !head_right.is_finite() {
        return Err(Refusal::NonFinite("head_slots_non_finite"));
    }
    let pf = if priority == 23 { 44124 } else { 49221 };
    let triangles = accept_indices(pf, indices, BROW_TARGETS)?;

    let (negative, positive) = if priority == 23 {
        (&HEAD_NEG_DEGREE, &HEAD_POS_DEGREE)
    } else {
        (&TAIL_NEG_DEGREE, &TAIL_POS_DEGREE)
    };
    let factor = if priority == 23 && kind == BrowPartKind::Long {
        BROW_HEAD_LONG_FACTOR
    } else {
        1.0
    };
    let range = if kind == BrowPartKind::Short { negative } else { positive };
    let degree_left = map(control_left, &PAIR_PARAM, range)? * factor;
    let degree_right = map(control_right, &PAIR_PARAM, range)? * factor;

    let slot_head_left = if priority == 23 { degree_left } else { head_left };
    let slot_head_right = if priority == 23 { degree_right } else { head_right };
    let slot_tail_left = if priority == 24 { degree_left } else { 0.0 };
    let slot_tail_right = if priority == 24 { degree_right } else { 0.0 };
    let axis_p = normalize(inputs.centerline_p24[8].sub(inputs.centerline_p24[0]))
        .ok_or(Refusal::NonFinite("type1018_guide_axis_degenerate"))?;
    let axis_f = normalize(inputs.centerline_f24[8].sub(inputs.centerline_f24[0]))
        .ok_or(Refusal::NonFinite("type1018_guide_axis_degenerate"))?;
    let half_p = inputs.centerline_p24[8].sub(inputs.centerline_p24[0]).length() * 0.5;
    let half_f = inputs.centerline_f24[8].sub(inputs.centerline_f24[0]).length() * 0.5;
    let center_p = inputs.centerline_p24[4];
    let center_f = inputs.centerline_f24[4];
    let mut target = inputs.targets78.to_vec();
    for i in 0..BROW_GUIDE_COUNT {
        let pd = dot(inputs.a30[i].sub(center_p), axis_p);
        let ratio_p = if half_p > 0.001 { pd / half_p } else { 0.0 };
        let selected_p = if pd <= 0.0 { slot_head_left } else { slot_tail_left };
        let amount_p = selected_p * BROW_LENGTH_AMOUNT * ratio_p * inputs.scale;
        target[i] = target[i].add(axis_p.scale(amount_p * inputs.param_list24[i]));
        target[24 + i] = target[24 + i].add(axis_p.scale(amount_p * inputs.factor_list48[i]));
        target[48 + i] =
            target[48 + i].add(axis_p.scale(amount_p * inputs.factor_list48[24 + i]));

        let fd = dot(inputs.a30[6 + i].sub(center_f), axis_f);
        let ratio_f = if half_f > 0.001 { fd / half_f } else { 0.0 };
        let selected_f = if fd <= 0.0 { slot_head_right } else { slot_tail_right };
        let amount_f = selected_f * BROW_LENGTH_AMOUNT * ratio_f * inputs.scale;
        target[6 + i] = target[6 + i].add(axis_f.scale(amount_f * inputs.param_list24[i]));
        target[30 + i] = target[30 + i].add(axis_f.scale(amount_f * inputs.factor_list48[i]));
        target[54 + i] =
            target[54 + i].add(axis_f.scale(amount_f * inputs.factor_list48[24 + i]));
    }
    Ok(Output {
        degrees: vec![degree_left, degree_right],
        source: inputs.targets78.to_vec(),
        target: target.clone(),
        scalars: vec![inputs.scale, half_p, half_f, triangles as f32],
        open: vec![
            BROW_LENGTH_CONSUMER_OPEN,
            "EFFECT_JSON_SHA256_PINNED_FOR_ROW23; ROW24_LOADER_DOES_NOT_COMPARE_PIN",
        ],
        noop: target == inputs.targets78,
        ..Output::default()
    })
}

fn normalize(vector: Point) -> Option<Point> {
    let length = vector.length();
    if !(length > 0.0) || !length.is_finite() {
        return None;
    }
    Some(vector.scale(1.0 / length))
}

fn dot(a: Point, b: Point) -> f32 {
    a.x * b.x + a.y * b.y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn landmarks() -> Vec<Point> {
        (0..194).map(|i| Point::new(100.0 + i as f32 * 3.0, 100.0 + (i % 17) as f32 * 7.0)).collect()
    }

    fn ramp(count: usize, step: f32) -> Vec<Point> {
        (0..count)
            .map(|i| Point::new(100.0 + i as f32 * step, 80.0 + (i % 9) as f32 * step))
            .collect()
    }

    fn face_block() -> Vec<f32> {
        let mut face = vec![1.0f32; 1300];
        face[FACE_THICKNESS_X] = 2.0;
        face[FACE_THICKNESS_A] = 2.0;
        face[FACE_THICKNESS_C] = 0.0;
        face[FACE_THICKNESS_Y] = 2.0;
        face[FACE_THICKNESS_B] = 2.0;
        face[FACE_THICKNESS_D] = 0.0;
        face[BROW_AXIS_A * 2] = 1.0;
        face[BROW_AXIS_A * 2 + 1] = 1.0;
        face[BROW_AXIS_B * 2] = 0.0;
        face[BROW_AXIS_B * 2 + 1] = 0.0;
        face
    }

    fn mesh_indices(vertices: usize) -> Vec<u16> {
        (0..vertices as u16 - 2).flat_map(|i| [i, i + 1, i + 2]).collect()
    }

    #[test]
    fn meta_rows_are_callable_and_unique() {
        assert_eq!(QUAD_INDICES, [0, 1, 2, 1, 2, 3]);
        assert_eq!(META.len(), 24);
        for pair in META.windows(2) {
            assert!(pair[0].pf < pair[1].pf);
        }
        assert!(META.iter().all(|row| row.access == Access::Cpu));
        assert_eq!(META.iter().filter(|row| !row.key.is_empty()).count(), 2);
        assert!(META.iter().all(|row| row.pf != 8007 && row.pf != 8003 && row.pf != 30071));
    }

    #[test]
    fn degrees_cover_piecewise_special_kink() {
        assert!((map(0.5, &EYE_SIZE_PARAM, &EYE_SIZE_DEGREE).unwrap() - 0.0).abs() < 1e-7);
        assert!((map(0.5005, &EYE_SIZE_PARAM, &EYE_SIZE_DEGREE).unwrap() - 0.075).abs() < 1e-4);
        assert_eq!(map(1.5, &EYE_SIZE_PARAM, &EYE_SIZE_DEGREE), Err(Refusal::OutOfRange("degree_table_miss")));
        assert_eq!(control(150.0), Err(Refusal::OutOfRange("ui_outside_-100_100")));
    }

    #[test]
    fn row11_zero_is_no_emit_and_bounds_follow_ui() {
        let zero = row11_pupil_size(&landmarks(), 1000, 1000, Point::new(1.0, 1.0), 0.0, 0.0, 1.0, 1.0).unwrap();
        assert!(zero.noop && zero.quads.is_empty());
        let out = row11_pupil_size(&landmarks(), 1000, 1000, Point::new(1.0, 1.0), 50.0, 50.0, 1.0, 1.0).unwrap();
        assert_eq!(out.quads.len(), 2);
        assert_eq!(row11_pupil_size(&landmarks(), 1000, 1000, Point::new(1.0, 1.0), 101.0, 0.0, 1.0, 1.0),
            Err(Refusal::OutOfRange("ui_outside_-100_100")));
    }

    #[test]
    fn row12_zero_is_a_clean_noop_like_the_engine() {
        let zero = row12_eye_height(&landmarks(), 1000, 1200, Point::new(1.0, 1.0), 0.0, 0.0, 1.0);
        let zero = zero.unwrap();
        assert!(zero.noop && zero.quads.is_empty());
        let half = row12_eye_height(&landmarks(), 1000, 1200, Point::new(1.0, 1.0), 50.0, 0.0, 1.0);
        let half = half.unwrap();
        assert!(half.noop && half.quads.is_empty());
        let both = row12_eye_height(&landmarks(), 1000, 1200, Point::new(1.0, 1.0), 50.0, 50.0, 1.0).unwrap();
        assert_eq!(both.quads.len(), 2);
    }

    #[test]
    fn row13_zero_is_clean_no_emit_and_factor_is_halved() {
        let zero = row13_eye_width(&landmarks(), 1000, 1000, Point::new(1.0, 1.0), 0.0, 0.0, ROW13_FACTOR_LIST4).unwrap();
        assert!(zero.noop && zero.quads.is_empty());
        let out = row13_eye_width(&landmarks(), 1000, 1000, Point::new(1.0, 1.0), 50.0, 50.0, ROW13_FACTOR_LIST4).unwrap();
        assert_eq!(out.quads.len(), 2);
        assert!((out.scalars[2] - 0.3).abs() < 1e-7);
    }

    #[test]
    fn mesh_rows_refuse_without_index_buffer() {
        assert_eq!(
            row14_eye_distance(&landmarks(), 1000, 1000, Point::new(1.0, 1.0), 50.0, 50.0, 0.0, 0.0, None),
            Err(Refusal::MissingEbo(EBO_14))
        );
        let ring = ramp(20, 2.0);
        assert_eq!(
            row15_eye_tilt(&landmarks(), 50.0, 50.0, &ring, &ring, 1.0, Point::new(1.0, 1.0),
                Point::new(1.0, 1.0), Point::new(100.0, 100.0), Point::new(100.0, 100.0), None),
            Err(Refusal::MissingEbo(EBO_15))
        );
        assert_eq!(
            row16_eye_position(&landmarks(), 50.0, 50.0, 0.1, 0.1, None),
            Err(Refusal::MissingEbo(EBO_16))
        );
        assert_eq!(
            row19_lower_eyelid(&ramp(100, 2.0), Some(&[0.1; 6]), Point::new(0.5, 0.5), 50.0, 50.0, None),
            Err(Refusal::MissingEbo(EBO_19))
        );
        assert_eq!(
            row20_brow_thickness(&ramp(168, 2.0), &ramp(168, 2.0), &face_block(), 50.0, 50.0, None),
            Err(Refusal::MissingEbo(EBO_20))
        );
    }

    #[test]
    fn row14_pairs_with_pair5_and_identity_clone() {
        let indices = mesh_indices(EYE_PAIR_VERTICES);
        let zero = row14_eye_distance(&landmarks(), 1000, 1000, Point::new(1.0, 1.0), 0.0, 0.0, 0.0, 0.0, Some(&indices)).unwrap();
        assert!(zero.noop);
        assert_eq!(zero.source, zero.target);
        assert_eq!(zero.source.len(), EYE_PAIR_VERTICES);
        assert!(zero.source.iter().any(|point| *point != Point::new(0.0, 0.0)));
        let out = row14_eye_distance(&landmarks(), 1000, 1000, Point::new(1.0, 1.0), 50.0, 50.0, 0.1, 0.1, Some(&indices)).unwrap();
        assert!(!out.noop);
        assert_ne!(out.source, out.target);
        assert_eq!(out.target.len(), EYE_PAIR_VERTICES);
    }

    #[test]
    fn row15_zero_skips_geometry_scale_and_rotates_only_1_to_8() {
        let indices = mesh_indices(20);
        let ring: Vec<Point> = (0..20).map(|i| Point::new(i as f32, (i * i) as f32)).collect();
        let zero = row15_eye_tilt(&landmarks(), 0.0, 0.0, &ring, &ring, 1.0,
            Point::new(1.0, 1.0), Point::new(4.0, 4.0),
            Point::new(1000.0, 1000.0), Point::new(1000.0, 1000.0), Some(&indices)).unwrap();
        assert!(zero.noop);
        assert_eq!(zero.target, zero.source);
        let out = row15_eye_tilt(&landmarks(), 50.0, -50.0, &ring, &ring, 1.0,
            Point::new(1.0, 1.0), Point::new(1.0, 1.0),
            Point::new(0.0, 0.0), Point::new(0.0, 0.0), Some(&indices)).unwrap();
        assert_eq!(out.target[0], out.source[0]);
        assert_eq!(out.target[9], out.source[9]);
        assert_ne!(out.target[4], out.source[4]);
    }

    #[test]
    fn row16_zero_uses_wrapped_identity() {
        let indices = mesh_indices(EYE_PAIR_VERTICES);
        let zero = row16_eye_position(&landmarks(), 0.0, 0.0, 0.0, 0.0, Some(&indices)).unwrap();
        assert!(zero.noop);
        assert_eq!(zero.source, zero.target);
        assert!(zero.open.iter().any(|item| item.contains("RAW_COMPUTE_ZERO_MESH_DEFECT")));
        let out = row16_eye_position(&landmarks(), 0.0, 0.0, 0.1, 0.0, Some(&indices)).unwrap();
        assert_ne!(out.source, out.target);
    }

    #[test]
    fn rows_15_16_facial_refine_and_corners_stay_open() {
        let tilt = row15_facial_refine(50.0, 50.0).unwrap();
        assert_eq!(tilt.degrees, vec![0.5, 0.5]);
        assert_eq!(tilt.refusal,
            Refusal::OpenGeometry("PF8007_PF8008_PART14_SLOT4_5_MESH_PRODUCER_OPEN rva=0x4C8E9F0"));
        let position = row16_facial_refine(50.0, 50.0).unwrap();
        assert_eq!(position.degrees, vec![0.5, 0.5]);
        assert_eq!(position.refusal,
            Refusal::OpenGeometry("PF8003_PF8004_PART14_SLOT0_1_MESH_PRODUCER_OPEN"));
        let inner = corners_degree_vector(17, 50.0, 50.0, None).unwrap();
        assert_eq!(inner.degrees, vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0]);
        assert_eq!(inner.refusal,
            Refusal::OpenGeometry("OPEN_TYP1010_PARAMLIST_TO_LOCAL_WARP rva=0x4C96A60 factory=0x60F03F6"));
        let outer = corners_degree_vector(18, -50.0, -50.0, None).unwrap();
        assert_eq!(outer.degrees, vec![0.0, 0.0, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(corners_degree_vector(18, 50.0, 50.0, Some(&[0.0; 27])),
            Err(Refusal::OutOfRange("type1010_paramlist_28_required")));
    }

    #[test]
    fn row19_zero_and_param_reversal() {
        let prepared = ramp(100, 2.0);
        let indices = mesh_indices(100);
        let zero = row19_lower_eyelid(&prepared, Some(&[0.5; 6]), Point::new(0.5, 0.5), 0.0, 0.0, Some(&indices)).unwrap();
        assert!(zero.noop);
        assert_eq!(zero.source, zero.target);
        let out = row19_lower_eyelid(&prepared, Some(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            Point::new(0.5, 0.5), 50.0, 50.0, Some(&indices)).unwrap();
        assert!(!out.noop);
        assert_ne!(out.target[14].y, out.target[63].y);
    }

    #[test]
    fn rows20_22_apply_and_zero_identity() {
        let mesh = ramp(168, 2.0);
        let guide = ramp(168, 2.0);
        let face = face_block();
        let indices = mesh_indices(168);
        let zero = row20_brow_thickness(&mesh, &guide, &face, 0.0, 0.0, Some(&indices)).unwrap();
        assert!(zero.noop);
        assert_eq!(zero.target, zero.source);
        let out = row20_brow_thickness(&mesh, &guide, &face, 50.0, 50.0, Some(&indices)).unwrap();
        assert_ne!(out.target, out.source);
        let distance = row21_brow_distance(&mesh, &face, 50.0, 50.0, Some(&indices)).unwrap();
        assert_eq!(distance.mids.len(), 56);
        let shape = row22_brow_shape(&mesh, &guide, &face, -50.0, 50.0, Some(&indices)).unwrap();
        assert!(!shape.noop);
        assert_eq!(shape.target.len(), 168);
    }

    #[test]
    fn rows23_24_head_tail_slots_differ() {
        let a30 = ramp(30, 2.0);
        let p24 = ramp(30, 2.0);
        let f24 = ramp(30, 2.0);
        let params = vec![0.5f32; 24];
        let factors = vec![1.0f32; 48];
        let targets = ramp(78, 2.0);
        let indices = mesh_indices(78);
        let inputs = BrowLengthInputs {
            a30: &a30, centerline_p24: &p24, centerline_f24: &f24,
            param_list24: &params, factor_list48: &factors, targets78: &targets, scale: 1.0,
        };
        let head = brow_length(23, BrowPartKind::Long, inputs, 50.0, 50.0, 0.5, 0.5, Some(&indices)).unwrap();
        let tail = brow_length(24, BrowPartKind::Long, inputs, 50.0, 50.0, 0.5, 0.5, Some(&indices)).unwrap();
        assert!(tail.degrees[0] > head.degrees[0]);
        assert_ne!(head.target, tail.target);
        let zero_inputs = BrowLengthInputs { ..inputs };
        let zero = brow_length(23, BrowPartKind::Short, zero_inputs, 0.0, 0.0, 0.0, 0.0, Some(&indices)).unwrap();
        assert!(zero.noop);
    }

    #[test]
    fn explicit_inputs_refuse() {
        assert_eq!(ROW15_FACIAL_REFINE_GAP, "0x4C8E9F0");
        assert_eq!(row11_pupil_size(&landmarks()[..193], 1000, 1000, Point::new(1.0, 1.0), 0.0, 0.0, 1.0, 1.0),
            Err(Refusal::MissingLandmarks));
        assert_eq!(row20_brow_thickness(&ramp(167, 2.0), &ramp(168, 2.0), &face_block(), 50.0, 50.0, Some(&mesh_indices(168))),
            Err(Refusal::MissingMesh("brow_mesh_and_guide_168_required")));
        assert_eq!(row19_lower_eyelid(&ramp(100, 2.0), Some(&[0.0; 5]), Point::new(0.5, 0.5), 50.0, 50.0, Some(&mesh_indices(100))),
            Err(Refusal::MissingParamList));
        assert_eq!(accept_indices(8001, Some(&[0, 1, 400]), 168),
            Err(Refusal::OutOfRange("index_out_of_mesh_range")));
    }
}
