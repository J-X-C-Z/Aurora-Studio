//! PF8050/PF8051 overall-size six-stage candidate geometry consumer on NVIDIA.
//!
//! The caller supplies the same-job producer receipt, the 144-point mesh,
//! 684-entry index buffer, falloff weights and derived object fields. This
//! module never builds a mesh, index buffer or face from a mask fallback.
//! A successful run proves the six-stage candidate geometry consumer against
//! the s928 fixture; it does not prove same-job inputs, vendor VBO/FBO binding
//! or final rasterized pixels.

use crate::cuda::{CudaError, Dispatch};
#[cfg(windows)]
use crate::cuda::Nvidia;
use crate::Refusal;

pub const KERNEL: &str = "pf8050_six_stage_kernel";
pub const MESH_POINTS: usize = 144;
pub const INDEX_COUNT: usize = 684;
pub const STAGE_COUNT: usize = 6;
const PROJECT_PTX: &str = include_str!("../native/cuda/pf8050_stage.ptx");

const PROFILE_15: [f32; 15] = [
    1.0004600286483765,
    0.9006100296974182,
    0.7004799842834473,
    0.5005099773406982,
    0.40018001198768616,
    0.300819993019104,
    0.20074999332427979,
    0.10027000308036804,
    0.20054000616073608,
    0.3002699911594391,
    0.40081000328063965,
    0.5002700090408325,
    0.7004899978637695,
    0.9007200002670288,
    1.0004600286483765,
];

const IND14: [usize; 14] = [0, 1, 2, 3, 11, 12, 13, 14, 27, 26, 25, 17, 16, 15];

#[derive(Clone, Copy, Debug)]
pub struct Producer<'a> {
    pub job_id: u64,
    pub face_id: u32,
    pub points_194: &'a [[f32; 2]],
    pub points_526: &'a [[f32; 2]],
}

#[derive(Clone, Copy, Debug)]
pub struct Weights {
    pub stage0_raw: [f32; 28],
    pub stage2: [f32; 28],
}

/// s928 compact candidate ABI. Direct, gate and profile centers share the
/// `center_*` slots; the candidate fixture satisfies that equality, while live
/// producers must preflight the full 12-pair object fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectFields {
    pub center_left: [f32; 2],
    pub center_right: [f32; 2],
    pub mid_left: [f32; 2],
    pub mid_right: [f32; 2],
    pub axis1_left: [f32; 2],
    pub axis1_right: [f32; 2],
    pub axis2_left: [f32; 2],
    pub axis2_right: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceFields {
    pub f0: [f32; 2],
    pub f1: [f32; 2],
    pub direction_point: [f32; 2],
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Inputs<'a> {
    pub producer: Option<Producer<'a>>,
    pub mesh_a: Option<&'a [[f32; 2]]>,
    pub indices: Option<&'a [u16]>,
    pub weights: Option<&'a Weights>,
    pub object: Option<&'a ObjectFields>,
    pub face: Option<&'a FaceFields>,
    pub degrees: Option<&'a [f32; 12]>,
}

#[derive(Clone, Copy, Debug)]
pub struct Resolved<'a> {
    pub producer: Producer<'a>,
    pub mesh_a: &'a [[f32; 2]],
    pub indices: &'a [u16],
    pub weights: &'a Weights,
    pub object: &'a ObjectFields,
    pub face: &'a FaceFields,
    pub degrees: &'a [f32; 12],
}

#[derive(Clone, Debug, PartialEq)]
pub struct Output {
    pub stages: Vec<[[f32; 2]; MESH_POINTS]>,
}

impl Output {
    pub fn final_mesh(&self) -> &[[f32; 2]; MESH_POINTS] {
        &self.stages[STAGE_COUNT - 1]
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct Uniforms {
    degrees: [f32; 12],
    center_left: [f32; 2],
    center_right: [f32; 2],
    mid_left: [f32; 2],
    mid_right: [f32; 2],
    axis1_left: [f32; 2],
    axis1_right: [f32; 2],
    axis2_left: [f32; 2],
    axis2_right: [f32; 2],
    face_center: [f32; 2],
    face_direction_point: [f32; 2],
    cos_left: f32,
    sin_left: f32,
    cos_right: f32,
    sin_right: f32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 144);

pub fn resolve(inputs: Inputs<'_>) -> Result<Resolved<'_>, Refusal> {
    let producer = inputs
        .producer
        .ok_or(Refusal::MissingInput("PF8050_SAME_JOB_PRODUCER"))?;
    if producer.points_194.len() != 194 {
        return Err(Refusal::ShapeMismatch("PF8050_POINTS194"));
    }
    if producer.points_526.len() != 526 {
        return Err(Refusal::ShapeMismatch("PF8050_POINTS526"));
    }
    if producer
        .points_194
        .iter()
        .chain(producer.points_526)
        .flat_map(|point| point.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("PF8050_PRODUCER_NONFINITE"));
    }

    let mesh_a = inputs
        .mesh_a
        .ok_or(Refusal::MissingInput("PF8050_MESH_A"))?;
    if mesh_a.len() != MESH_POINTS {
        return Err(Refusal::ShapeMismatch("PF8050_MESH_A"));
    }
    if mesh_a
        .iter()
        .flat_map(|point| point.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("PF8050_MESH_A"));
    }

    let indices = inputs
        .indices
        .ok_or(Refusal::MissingInput("PF8050_INDICES"))?;
    if indices.len() != INDEX_COUNT {
        return Err(Refusal::ShapeMismatch("PF8050_INDICES"));
    }
    if indices.iter().any(|index| *index as usize >= MESH_POINTS) {
        return Err(Refusal::ValueOutOfRange("PF8050_INDICES"));
    }

    let weights = inputs
        .weights
        .ok_or(Refusal::MissingInput("PF8050_WEIGHTS"))?;
    if weights
        .stage0_raw
        .iter()
        .chain(weights.stage2.iter())
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("PF8050_WEIGHTS"));
    }

    let object = inputs
        .object
        .ok_or(Refusal::MissingInput("PF8050_OBJECT_FIELDS"))?;
    if !object_is_finite(object) {
        return Err(Refusal::ValueOutOfRange("PF8050_OBJECT_FIELDS"));
    }
    let face = inputs
        .face
        .ok_or(Refusal::MissingInput("PF8050_FACE_FIELDS"))?;
    if !point_is_finite(face.f0)
        || !point_is_finite(face.f1)
        || !point_is_finite(face.direction_point)
    {
        return Err(Refusal::ValueOutOfRange("PF8050_FACE_FIELDS"));
    }
    let degrees = inputs
        .degrees
        .ok_or(Refusal::MissingInput("PF8050_DEGREES12"))?;
    if degrees.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("PF8050_DEGREES12"));
    }

    Ok(Resolved {
        producer,
        mesh_a,
        indices,
        weights,
        object,
        face,
        degrees,
    })
}

fn point_is_finite(point: [f32; 2]) -> bool {
    point[0].is_finite() && point[1].is_finite()
}

fn object_is_finite(object: &ObjectFields) -> bool {
    [
        object.center_left,
        object.center_right,
        object.mid_left,
        object.mid_right,
        object.axis1_left,
        object.axis1_right,
        object.axis2_left,
        object.axis2_right,
    ]
    .into_iter()
    .all(point_is_finite)
}

#[inline]
fn fadd(left: f32, right: f32) -> f32 {
    left + right
}

#[inline]
fn fsub(left: f32, right: f32) -> f32 {
    left - right
}

#[inline]
fn fmul(left: f32, right: f32) -> f32 {
    left * right
}

#[inline]
fn fdiv(left: f32, right: f32) -> f32 {
    (left as f64 / right as f64) as f32
}

#[inline]
fn fsqrt(value: f32) -> f32 {
    (value as f64).sqrt() as f32
}

#[inline]
fn add2(left: [f32; 2], right: [f32; 2]) -> [f32; 2] {
    [fadd(left[0], right[0]), fadd(left[1], right[1])]
}

#[inline]
fn sub2(left: [f32; 2], right: [f32; 2]) -> [f32; 2] {
    [fsub(left[0], right[0]), fsub(left[1], right[1])]
}

#[inline]
fn scale2(point: [f32; 2], scale: f32) -> [f32; 2] {
    [fmul(point[0], scale), fmul(point[1], scale)]
}

#[inline]
fn length2(vector: [f32; 2]) -> f32 {
    fsqrt(fadd(fmul(vector[0], vector[0]), fmul(vector[1], vector[1])))
}

#[inline]
fn minps(left: f32, right: f32) -> f32 {
    if left.is_nan() || right.is_nan() {
        right
    } else if right < left {
        right
    } else {
        left
    }
}

fn stage0_weights(raw: &[f32; 28]) -> [f32; 28] {
    let mut weights = raw.map(|value| fmul(value, 1.5));
    let step = fdiv(1.0, 7.0);
    let a = fmul(fsub(1.0, 0.25), step);
    let b = fmul(step, fsub(1.0, a));
    weights[0] = fmul(weights[0], b);
    weights[14] = fmul(weights[14], b);
    let mut rem = fsub(fsub(1.0, a), a);
    let mut t = fadd(step, step);

    for left in [1usize, 3, 5] {
        let right = 28 - left;
        weights[left] = fmul(weights[left], t);
        let sum = fadd(step, t);
        weights[right] = fmul(weights[right], fmul(t, rem));
        rem = fsub(rem, a);
        weights[left + 1] = fmul(weights[left + 1], sum);
        t = fadd(step, sum);
        weights[right - 1] = fmul(weights[right - 1], fmul(sum, rem));
        rem = fsub(rem, a);
    }

    t = fsub(t, step);
    rem = fadd(rem, a);
    weights[7] = fmul(weights[7], t);
    weights[21] = fmul(weights[21], fmul(rem, t));
    for index in 8..14 {
        weights[index] = fmul(weights[index], t);
        let right = 28 - index;
        weights[right] = fmul(weights[right], fmul(rem, t));
        t = fsub(t, step);
        rem = fadd(rem, a);
    }
    weights
}

fn point_polygon_test(polygon: &[[f32; 2]], point: [f32; 2]) -> f32 {
    if polygon.is_empty() {
        return -1.0;
    }
    let px = point[0];
    let py = point[1];
    let mut inside = false;
    let mut previous = polygon[polygon.len() - 1];
    for current in polygon {
        let x1 = current[0];
        let y1 = current[1];
        if py == y1 && px == x1 {
            return 0.0;
        }
        if py == previous[1]
            && py == y1
            && previous[0].min(x1) <= px
            && px <= previous[0].max(x1)
        {
            return 0.0;
        }
        if (y1 > py) != (previous[1] > py) {
            let cross = (f64::from(x1) - f64::from(previous[0]))
                * (f64::from(py) - f64::from(previous[1]))
                - (f64::from(y1) - f64::from(previous[1]))
                    * (f64::from(px) - f64::from(previous[0]));
            if cross == 0.0 {
                return 0.0;
            }
            if (previous[1] < y1 && cross < 0.0)
                || (previous[1] > y1 && cross > 0.0)
            {
                inside = !inside;
            }
        }
        previous = *current;
    }
    if inside { 1.0 } else { -1.0 }
}

fn stage0(
    a: &[[f32; 2]; MESH_POINTS],
    b: &mut [[f32; 2]; MESH_POINTS],
    p_left: f32,
    p_right: f32,
    object: &ObjectFields,
    raw_weights: &[f32; 28],
) {
    for index in 0..15 {
        let left = a[1 + index];
        let right = a[73 + index];
        let left_delta = scale2(sub2(left, object.center_left), p_left);
        let right_delta = scale2(sub2(right, object.center_right), p_right);
        b[1 + index] = add2(left, scale2(left_delta, 0.2));
        b[73 + index] = add2(right, scale2(right_delta, 0.2));
    }

    let weights = stage0_weights(raw_weights);
    for (index, weight) in weights.into_iter().enumerate() {
        let left = a[16 + index];
        let right = a[88 + index];
        let left_delta = scale2(sub2(object.mid_left, left), weight);
        let right_delta = scale2(sub2(object.mid_right, right), weight);
        b[16 + index] = sub2(left, scale2(left_delta, p_left));
        b[88 + index] = sub2(right, scale2(right_delta, p_right));
    }
}

fn distance_gate_min4(
    b: &[[f32; 2]; MESH_POINTS],
    center_left: [f32; 2],
    center_right: [f32; 2],
) -> [f32; 4] {
    let mut lanes = [10000.0f32, 10000.0, 0.0, 0.0];
    for index in IND14 {
        let diff = [
            fsub(
                length2(sub2(center_left, b[44 + index])),
                length2(sub2(center_left, b[16 + index])),
            ),
            fsub(
                length2(sub2(center_right, b[116 + index])),
                length2(sub2(center_right, b[88 + index])),
            ),
            0.0,
            0.0,
        ];
        lanes = [
            minps(diff[0], lanes[0]),
            minps(diff[1], lanes[1]),
            minps(diff[2], lanes[2]),
            minps(diff[3], lanes[3]),
        ];
    }
    lanes
}

fn stage1(
    b: &mut [[f32; 2]; MESH_POINTS],
    p_left: f32,
    p_right: f32,
    object: &ObjectFields,
) -> [f32; 2] {
    let mut min_diff = [10000.0f32, 10000.0, 0.0, 0.0];
    for index in 0..28 {
        let diff = [
            fsub(
                length2(sub2(object.center_right, b[116 + index])),
                length2(sub2(object.center_right, b[88 + index])),
            ),
            fsub(
                length2(sub2(object.center_left, b[44 + index])),
                length2(sub2(object.center_left, b[16 + index])),
            ),
            0.0,
            0.0,
        ];
        min_diff[0] = minps(diff[0], min_diff[0]);
        min_diff[1] = minps(diff[1], min_diff[1]);
        min_diff[2] = minps(diff[2], min_diff[2]);
        min_diff[3] = minps(diff[3], min_diff[3]);
    }

    let axis_left = scale2(object.axis1_left, 0.5);
    let axis_right = scale2(object.axis1_right, 0.5);
    let mut min_ratio = [1.0f32, 1.0, 0.0, 0.0];
    for profile in PROFILE_15 {
        let weighted_right = fmul(profile, p_right);
        let weighted_left = fmul(profile, p_left);
        let horizontal_right = fmul(axis_right[1], weighted_right);
        let vertical_right = fmul(weighted_right, axis_right[0]);
        let horizontal_left = fmul(axis_left[1], weighted_left);
        let vertical_left = fmul(weighted_left, axis_left[0]);
        let norm_right = fsqrt(fadd(
            fmul(horizontal_right, horizontal_right),
            fmul(vertical_right, vertical_right),
        ));
        let norm_left = fsqrt(fadd(
            fmul(horizontal_left, horizontal_left),
            fmul(vertical_left, vertical_left),
        ));
        min_ratio[0] = minps(fdiv(min_diff[0], norm_right), min_ratio[0]);
        min_ratio[1] = minps(fdiv(min_diff[1], norm_left), min_ratio[1]);
    }

    let m_right = min_ratio[0];
    let m_left = min_ratio[1];
    let coeff_left = fmul(p_left, m_left);
    let coeff_right = fmul(p_right, m_right);

    b[16] = add2(
        b[16],
        scale2(
            axis_left,
            fmul(fmul(PROFILE_15[0], -1.0), coeff_left),
        ),
    );
    b[88] = add2(
        b[88],
        scale2(
            axis_right,
            fmul(fmul(PROFILE_15[0], -1.0), coeff_right),
        ),
    );
    b[30] = add2(
        b[30],
        scale2(
            axis_left,
            fmul(fmul(PROFILE_15[14], 1.0), coeff_left),
        ),
    );
    b[102] = add2(
        b[102],
        scale2(
            axis_right,
            fmul(fmul(PROFILE_15[14], 1.0), coeff_right),
        ),
    );

    for index in 1..14 {
        let sign = if index < 7 { -1.0 } else { 1.0 };
        let left = scale2(
            scale2(scale2(axis_left, sign), PROFILE_15[index]),
            coeff_left,
        );
        let right = scale2(
            scale2(scale2(axis_right, sign), PROFILE_15[index]),
            coeff_right,
        );
        b[16 + index] = add2(b[16 + index], left);
        b[44 - index] = add2(b[44 - index], left);
        b[88 + index] = add2(b[88 + index], right);
        b[116 - index] = add2(b[116 - index], right);
    }


    commit_contour_candidates(b, p_left, p_right, object);
    [m_left, m_right]
}

fn commit_contour_candidates(
    b: &mut [[f32; 2]; MESH_POINTS],
    p_left: f32,
    p_right: f32,
    object: &ObjectFields,
) {
    for index in 0..15 {
        let left_delta = scale2(sub2(b[1 + index], object.center_left), 0.1599999964237213);
        let right_delta = scale2(sub2(b[73 + index], object.center_right), 0.1599999964237213);
        let candidate_left = add2(b[1 + index], scale2(left_delta, p_left));
        let candidate_right = add2(b[73 + index], scale2(right_delta, p_right));
        let left_gate = point_polygon_test(&b[16..44], candidate_left);
        let right_gate = point_polygon_test(&b[88..116], candidate_right);
        if left_gate == 1.0 {
            b[1 + index] = candidate_left;
        }
        if right_gate == 1.0 {
            b[73 + index] = candidate_right;
        }
    }
}

fn stage2(
    b: &mut [[f32; 2]; MESH_POINTS],
    p_left: f32,
    p_right: f32,
    object: &ObjectFields,
    weights: &[f32; 28],
) {
    let left_step = scale2(object.axis2_left, p_left);
    let right_step = scale2(object.axis2_right, p_right);
    for (index, weight) in weights.iter().copied().enumerate() {
        let left = scale2(left_step, weight);
        let right = scale2(right_step, weight);
        if index < 15 {
            b[16 + index] = add2(b[16 + index], left);
            b[88 + index] = add2(b[88 + index], right);
        } else {
            b[16 + index] = sub2(b[16 + index], left);
            b[88 + index] = sub2(b[88 + index], right);
        }
    }
    commit_contour_candidates(b, p_left, p_right, object);
}

fn rotations(degrees: &[f32; 12]) -> (f32, f32, f32, f32) {
    let p3_left = fmul(degrees[6], 0.4500001072883606);
    let p3_right = fmul(degrees[7], 0.4500001072883606);
    let cos_left = f64::from(p3_left).cos() as f32;
    let sin_left = f64::from(p3_left).sin() as f32;
    let cos_right = f64::from(-p3_right).cos() as f32;
    let sin_right = f64::from(-p3_right).sin() as f32;
    (cos_left, sin_left, cos_right, sin_right)
}

fn stage3(
    b: &mut [[f32; 2]; MESH_POINTS],
    object: &ObjectFields,
    cos_left: f32,
    sin_left: f32,
    cos_right: f32,
    sin_right: f32,
) {
    for (side, center, cosine, sine) in [
        (0usize, object.center_left, cos_left, sin_left),
        (1usize, object.center_right, cos_right, sin_right),
    ] {
        let base = side * 72;
        for index in 1..44 {
            let point = b[base + index];
            let dx = fsub(point[0], center[0]);
            let dy = fsub(point[1], center[1]);
            b[base + index] = [
                fadd(center[0], fadd(fmul(dx, cosine), fmul(dy, sine))),
                fadd(center[1], fadd(fmul(fmul(-1.0, dx), sine), fmul(dy, cosine))),
            ];
        }
    }
}

fn stage4(
    b: &mut [[f32; 2]; MESH_POINTS],
    p_left: f32,
    p_right: f32,
    center_left: &mut [f32; 2],
    center_right: &mut [f32; 2],
) {
    let axis = sub2(*center_right, *center_left);
    let axis_len = length2(axis);
    let lanes = distance_gate_min4(b, *center_left, *center_right);
    let m_left = lanes[0];
    let m_right = lanes[1];
    let v_left = [
        fdiv(fmul(fmul(-p_left, axis[0]), m_left), axis_len),
        fdiv(fmul(fmul(-p_left, axis[1]), m_left), axis_len),
    ];
    let v_right = [
        fdiv(fmul(fmul(p_right, axis[0]), m_right), axis_len),
        fdiv(fmul(fmul(p_right, axis[1]), m_right), axis_len),
    ];
    for point in &mut b[0..44] {
        *point = add2(*point, v_left);
    }
    for point in &mut b[72..116] {
        *point = add2(*point, v_right);
    }
    *center_left = add2(*center_left, v_left);
    *center_right = add2(*center_right, v_right);
}

fn stage5(
    b: &mut [[f32; 2]; MESH_POINTS],
    p_left: f32,
    p_right: f32,
    center_left: &mut [f32; 2],
    center_right: &mut [f32; 2],
    face: &FaceFields,
) {
    let face_center = [
        fmul(fadd(face.f0[0], face.f1[0]), 0.5),
        fmul(fadd(face.f0[1], face.f1[1]), 0.5),
    ];
    let direction = sub2(face.direction_point, face_center);
    let direction_len = length2(direction);
    let face_direction = [
        fdiv(direction[0], direction_len),
        fdiv(direction[1], direction_len),
    ];
    let lanes = distance_gate_min4(b, *center_left, *center_right);
    let m_left = lanes[0];
    let m_right = lanes[1];
    let amp_left = [
        fmul(fmul(p_left, face_direction[0]), m_left),
        fmul(fmul(p_left, face_direction[1]), m_left),
    ];
    let amp_right = [
        fmul(fmul(p_right, face_direction[0]), m_right),
        fmul(fmul(p_right, face_direction[1]), m_right),
    ];
    for point in &mut b[0..44] {
        *point = add2(*point, amp_left);
    }
    for point in &mut b[72..116] {
        *point = add2(*point, amp_right);
    }
    *center_left = add2(*center_left, amp_left);
    *center_right = add2(*center_right, amp_right);
}

pub fn apply_cpu(resolved: Resolved<'_>) -> Output {
    let a: [[f32; 2]; MESH_POINTS] = std::array::from_fn(|index| resolved.mesh_a[index]);
    let mut b = a;
    let mut stages = Vec::with_capacity(STAGE_COUNT);
    let mut center_left = resolved.object.center_left;
    let mut center_right = resolved.object.center_right;
    let degrees = resolved.degrees;

    let p0 = [
        fmul(fmul(degrees[0], 0.25), 0.2),
        fmul(fmul(degrees[1], 0.25), 0.2),
    ];
    let p1 = [
        fmul(degrees[2], 0.12000010162591934),
        fmul(degrees[3], 0.8000010251998901),
    ];
    let p2 = [
        fmul(degrees[4], 0.06000009924173355),
        fmul(degrees[5], 0.4500001072883606),
    ];
    let p4 = [
        fmul(degrees[8], 0.4500001072883606),
        fmul(degrees[9], 0.4500001072883606),
    ];
    let p5 = [
        fmul(degrees[10], 0.4500001072883606),
        fmul(degrees[11], 0.4500001072883606),
    ];

    stage0(
        &a,
        &mut b,
        p0[0],
        p0[1],
        resolved.object,
        &resolved.weights.stage0_raw,
    );
    stages.push(b);

    let _ = stage1(&mut b, p1[0], p1[1], resolved.object);
    stages.push(b);

    stage2(
        &mut b,
        p2[0],
        p2[1],
        resolved.object,
        &resolved.weights.stage2,
    );
    stages.push(b);

    let (cos_left, sin_left, cos_right, sin_right) = rotations(degrees);
    stage3(&mut b, resolved.object, cos_left, sin_left, cos_right, sin_right);
    stages.push(b);

    stage4(
        &mut b,
        p4[0],
        p4[1],
        &mut center_left,
        &mut center_right,
    );
    stages.push(b);

    stage5(
        &mut b,
        p5[0],
        p5[1],
        &mut center_left,
        &mut center_right,
        resolved.face,
    );
    stages.push(b);

    Output { stages }
}

#[cfg(windows)]
pub struct Pf8050Stage {
    gpu: Nvidia,
}

#[cfg(windows)]
impl Pf8050Stage {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self {
            gpu: Nvidia::open_ptx(PROJECT_PTX)?,
        })
    }

    pub fn apply(&self, resolved: Resolved<'_>) -> Result<(Output, Dispatch), CudaError> {
        let (cos_left, sin_left, cos_right, sin_right) = rotations(resolved.degrees);
        let uniforms = Uniforms {
            degrees: *resolved.degrees,
            center_left: resolved.object.center_left,
            center_right: resolved.object.center_right,
            mid_left: resolved.object.mid_left,
            mid_right: resolved.object.mid_right,
            axis1_left: resolved.object.axis1_left,
            axis1_right: resolved.object.axis1_right,
            axis2_left: resolved.object.axis2_left,
            axis2_right: resolved.object.axis2_right,
            face_center: [
                fmul(fadd(resolved.face.f0[0], resolved.face.f1[0]), 0.5),
                fmul(fadd(resolved.face.f0[1], resolved.face.f1[1]), 0.5),
            ],
            face_direction_point: resolved.face.direction_point,
            cos_left,
            sin_left,
            cos_right,
            sin_right,
        };

        let mesh_bytes = std::mem::size_of_val(resolved.mesh_a);
        let w0_bytes = std::mem::size_of_val(&resolved.weights.stage0_raw);
        let w2_bytes = std::mem::size_of_val(&resolved.weights.stage2);
        let output_bytes = STAGE_COUNT * MESH_POINTS * std::mem::size_of::<[f32; 2]>();
        self.gpu.with_reusable_buffers(
            "cuda_pf8050.apply",
            &[mesh_bytes, w0_bytes, w2_bytes, output_bytes],
            |buffers| {
                buffers.upload(0, f32_pair_bytes(resolved.mesh_a))?;
                buffers.upload(1, f32_bytes(&resolved.weights.stage0_raw))?;
                buffers.upload(2, f32_bytes(&resolved.weights.stage2))?;

                let mut mesh_arg = buffers.handle(0);
                let mut w0_arg = buffers.handle(1);
                let mut w2_arg = buffers.handle(2);
                let mut uniform_arg = uniforms;
                let mut output_arg = buffers.handle(3);
                let mut params: [*mut std::ffi::c_void; 5] = [
                    &mut mesh_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut w0_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut w2_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut uniform_arg as *mut Uniforms as *mut std::ffi::c_void,
                    &mut output_arg as *mut u64 as *mut std::ffi::c_void,
                ];
                let dispatch = buffers.launch(KERNEL, (1, 1), (1, 1), &mut params)?;
                let mut raw = vec![0u8; output_bytes];
                buffers.download(3, &mut raw)?;
                buffers.mark_consumed();

                let mut stages = Vec::with_capacity(STAGE_COUNT);
                for stage in 0..STAGE_COUNT {
                    let offset = stage * MESH_POINTS * 8;
                    let stage_bytes = &raw[offset..offset + MESH_POINTS * 8];
                    stages.push(std::array::from_fn(|index| {
                        let point = index * 8;
                        [
                            f32::from_le_bytes(stage_bytes[point..point + 4].try_into().unwrap()),
                            f32::from_le_bytes(
                                stage_bytes[point + 4..point + 8].try_into().unwrap(),
                            ),
                        ]
                    }));
                }
                Ok((Output { stages }, dispatch))
            },
        )
    }
}

#[cfg(windows)]
fn f32_pair_bytes(values: &[[f32; 2]]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr() as *const u8,
            std::mem::size_of_val(values),
        )
    }
}

#[cfg(windows)]
fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr() as *const u8,
            std::mem::size_of_val(values),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEGREES: [f32; 12] = [
        0.42, -0.31, 0.19, -0.17, 0.23, -0.13, 0.11, -0.09, 0.15, -0.07, 0.21,
        -0.12,
    ];
    const OBJECT: ObjectFields = ObjectFields {
        center_left: [55.0, 55.0],
        center_right: [125.0, 55.0],
        mid_left: [52.0, 55.0],
        mid_right: [122.0, 55.0],
        axis1_left: [3.0, 0.0],
        axis1_right: [3.0, 0.0],
        axis2_left: [-2.0, 2.0],
        axis2_right: [-2.0, 2.0],
    };
    const FACE: FaceFields = FaceFields {
        f0: [70.0, 50.0],
        f1: [90.0, 51.0],
        direction_point: [86.0, 34.0],
    };
    const EXPECTED: [&str; 6] = [
        "9C08508E0D3C4E0E96E2447920FB4E5658BFBCAC1738140E6399242FBF390C5A",
        "ED84FE772FC45DA72B758A46DD3C7219FBE7F9DEA0D4DB7D35B1EAF3711BDD70",
        "E53CA2F88AD12DF79F1F4EF6298969F35D34C54993320AEB88FD6C0D12316135",
        "A6EEC32BC6E35EF00429A356D96ACD7931F3E8274F5883F0295BC9A5197E5B6D",
        "CB1386DCE7300FA2CE7C678CA2507F9C63E78EB0D8E1318885415CE998C9D78B",
        "BD4BBC20FFD5F743ED0CDD1B197789D2F483798F2203D01915BE33D9B9894751",
    ];
    const MESH_A: &[u8] =
        include_bytes!("../tests/fixtures/pf8050_candidate_v1/mesh_a.f32le");
    const POINTS194: &[u8] =
        include_bytes!("../tests/fixtures/pf8050_candidate_v1/points194.f32le");
    const POINTS526: &[u8] =
        include_bytes!("../tests/fixtures/pf8050_candidate_v1/points526.f32le");
    const INDICES: &[u8] =
        include_bytes!("../tests/fixtures/pf8050_candidate_v1/ebo.u16le");
    const W0_RAW_BYTES: &[u8] =
        include_bytes!("../tests/fixtures/pf8050_candidate_v1/weights_w0_raw.f32le");
    const W0_SHAPED_BYTES: &[u8] =
        include_bytes!("../tests/fixtures/pf8050_candidate_v1/weights_w0_shaped.f32le");
    const W2_BYTES: &[u8] =
        include_bytes!("../tests/fixtures/pf8050_candidate_v1/weights_w2.f32le");
    const GOLDEN_STAGES: [&[u8]; 6] = [
        include_bytes!("../tests/fixtures/pf8050_candidate_v1/stage0.f32le"),
        include_bytes!("../tests/fixtures/pf8050_candidate_v1/stage1.f32le"),
        include_bytes!("../tests/fixtures/pf8050_candidate_v1/stage2.f32le"),
        include_bytes!("../tests/fixtures/pf8050_candidate_v1/stage3.f32le"),
        include_bytes!("../tests/fixtures/pf8050_candidate_v1/stage4.f32le"),
        include_bytes!("../tests/fixtures/pf8050_candidate_v1/stage5.f32le"),
    ];

    fn pairs(bytes: &[u8]) -> Vec<[f32; 2]> {
        bytes
            .chunks_exact(8)
            .map(|point| {
                [
                    f32::from_le_bytes(point[0..4].try_into().unwrap()),
                    f32::from_le_bytes(point[4..8].try_into().unwrap()),
                ]
            })
            .collect()
    }

    fn indices(bytes: &[u8]) -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|index| u16::from_le_bytes([index[0], index[1]]))
            .collect()
    }

    fn floats28(bytes: &[u8]) -> [f32; 28] {
        std::array::from_fn(|index| {
            f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap())
        })
    }

    fn fixture() -> (
        Vec<[f32; 2]>,
        Vec<[f32; 2]>,
        Vec<[f32; 2]>,
        Vec<u16>,
        Weights,
    ) {
        (
            pairs(POINTS194),
            pairs(POINTS526),
            pairs(MESH_A),
            indices(INDICES),
            Weights {
                stage0_raw: floats28(W0_RAW_BYTES),
                stage2: floats28(W2_BYTES),
            },
        )
    }

    fn resolved<'a>(
        producer: &'a Producer<'a>,
        mesh: &'a [[f32; 2]],
        indices: &'a [u16],
        weights: &'a Weights,
        degrees: &'a [f32; 12],
    ) -> Result<Resolved<'a>, Refusal> {
        resolve(Inputs {
            producer: Some(*producer),
            mesh_a: Some(mesh),
            indices: Some(indices),
            weights: Some(weights),
            object: Some(&OBJECT),
            face: Some(&FACE),
            degrees: Some(degrees),
        })
    }

    #[test]
    fn cpu_matches_the_pinned_s928_candidate_stage_hashes() {
        let (points_194, points_526, mesh, indices, weights) = fixture();
        let producer = Producer {
            job_id: 580,
            face_id: 0,
            points_194: &points_194,
            points_526: &points_526,
        };
        let resolved = resolved(&producer, &mesh, &indices, &weights, &DEGREES).unwrap();
        let output = apply_cpu(resolved);
        for (index, (stage, expected)) in output.stages.iter().zip(EXPECTED).enumerate() {
            let actual_bytes = stage_bytes(stage);
            if actual_bytes != GOLDEN_STAGES[index] {
                let byte = actual_bytes
                    .iter()
                    .zip(GOLDEN_STAGES[index])
                    .position(|(left, right)| left != right)
                    .unwrap();
                panic!(
                    "stage {index} first differs at byte {byte}, point {}: cpu {:?} golden {:?}",
                    byte / 8,
                    stage[byte / 8],
                    pairs(GOLDEN_STAGES[index])[byte / 8]
                );
            }
            assert_eq!(stage_sha(stage), expected);
        }
    }

    #[test]
    fn cpu_matches_the_pinned_s928_shaped_w0_bytes() {
        let (_, _, _, _, weights) = fixture();
        assert_eq!(stage0_weights(&weights.stage0_raw), floats28(W0_SHAPED_BYTES));
    }

    #[test]
    fn debug_stage0_weights() {
        let (_, _, _, _, weights) = fixture();
        eprintln!(
            "{:?}",
            stage0_weights(&weights.stage0_raw)
                .iter()
                .map(|value| format!("{value:?}"))
                .collect::<Vec<_>>()
        );
        let stage0 = pairs(GOLDEN_STAGES[0]);
        let mut mesh: [[f32; 2]; MESH_POINTS] =
            std::array::from_fn(|index| stage0[index]);
        let p1 = [
            fmul(DEGREES[2], 0.12000010162591934),
            fmul(DEGREES[3], 0.8000010251998901),
        ];
        eprintln!("stage1 m={:?}", stage1(&mut mesh, p1[0], p1[1], &OBJECT));
    }

    #[test]
    fn missing_producer_mesh_indices_or_weights_is_typed() {
        let (points_194, points_526, mesh, indices, weights) = fixture();
        let producer = Producer {
            job_id: 580,
            face_id: 0,
            points_194: &points_194,
            points_526: &points_526,
        };
        let base = Inputs {
            producer: Some(producer),
            mesh_a: Some(&mesh),
            indices: Some(&indices),
            weights: Some(&weights),
            object: Some(&OBJECT),
            face: Some(&FACE),
            degrees: Some(&DEGREES),
        };
        assert_eq!(
            resolve(Inputs {
                producer: None,
                ..base
            })
            .unwrap_err(),
            Refusal::MissingInput("PF8050_SAME_JOB_PRODUCER")
        );
        assert_eq!(
            resolve(Inputs {
                mesh_a: None,
                ..base
            })
            .unwrap_err(),
            Refusal::MissingInput("PF8050_MESH_A")
        );
        assert_eq!(
            resolve(Inputs {
                indices: None,
                ..base
            })
            .unwrap_err(),
            Refusal::MissingInput("PF8050_INDICES")
        );
        assert_eq!(
            resolve(Inputs {
                weights: None,
                ..base
            })
            .unwrap_err(),
            Refusal::MissingInput("PF8050_WEIGHTS")
        );
    }

    fn stage_sha(stage: &[[f32; 2]; MESH_POINTS]) -> String {
        sha256_hex(&stage_bytes(stage))
    }

    fn stage_bytes(stage: &[[f32; 2]; MESH_POINTS]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(MESH_POINTS * 8);
        for point in stage {
            bytes.extend_from_slice(&point[0].to_le_bytes());
            bytes.extend_from_slice(&point[1].to_le_bytes());
        }
        bytes
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut state: [u32; 8] = [
            0x6a09e667,
            0xbb67ae85,
            0x3c6ef372,
            0xa54ff53a,
            0x510e527f,
            0x9b05688c,
            0x1f83d9ab,
            0x5be0cd19,
        ];
        let constants: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
            0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
            0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
            0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
        ];
        let bit_len = (bytes.len() as u64) * 8;
        let mut padded = bytes.to_vec();
        padded.push(0x80);
        while padded.len() % 64 != 56 {
            padded.push(0);
        }
        padded.extend_from_slice(&bit_len.to_be_bytes());
        for chunk in padded.chunks_exact(64) {
            let mut words = [0u32; 64];
            for (index, word) in words.iter_mut().take(16).enumerate() {
                *word = u32::from_be_bytes(
                    chunk[index * 4..index * 4 + 4].try_into().unwrap(),
                );
            }
            for index in 16..64 {
                let s0 = words[index - 15].rotate_right(7)
                    ^ words[index - 15].rotate_right(18)
                    ^ (words[index - 15] >> 3);
                let s1 = words[index - 2].rotate_right(17)
                    ^ words[index - 2].rotate_right(19)
                    ^ (words[index - 2] >> 10);
                words[index] = words[index - 16]
                    .wrapping_add(s0)
                    .wrapping_add(words[index - 7])
                    .wrapping_add(s1);
            }
            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
            for index in 0..64 {
                let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let choice = (e & f) ^ (!e & g);
                let temp1 = h
                    .wrapping_add(s1)
                    .wrapping_add(choice)
                    .wrapping_add(constants[index])
                    .wrapping_add(words[index]);
                let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let majority = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = s0.wrapping_add(majority);
                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }
            state[0] = state[0].wrapping_add(a);
            state[1] = state[1].wrapping_add(b);
            state[2] = state[2].wrapping_add(c);
            state[3] = state[3].wrapping_add(d);
            state[4] = state[4].wrapping_add(e);
            state[5] = state[5].wrapping_add(f);
            state[6] = state[6].wrapping_add(g);
            state[7] = state[7].wrapping_add(h);
        }
        state
            .iter()
            .map(|word| format!("{word:08X}"))
            .collect::<String>()
    }

    #[cfg(windows)]
    #[test]
    fn real_gpu_matches_cpu_stage_bytes_on_rtx() {
        let (points_194, points_526, mesh, indices, weights) = fixture();
        let producer = Producer {
            job_id: 580,
            face_id: 0,
            points_194: &points_194,
            points_526: &points_526,
        };
        let resolved = resolved(&producer, &mesh, &indices, &weights, &DEGREES).unwrap();
        let expected = apply_cpu(resolved);
        let gpu = match Pf8050Stage::open() {
            Ok(gpu) => gpu,
            Err(CudaError::Unavailable(reason)) => {
                assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "CUDA required but unavailable: {reason}"
                );
                eprintln!("SKIP_NO_NVIDIA_DRIVER: {reason}");
                return;
            }
            Err(error) => panic!("NVIDIA path present but broken: {error:?}"),
        };
        let (actual, dispatch) = gpu.apply(resolved).expect("real PF8050 GPU dispatch");
        assert_eq!(dispatch.kernel, KERNEL);
        assert_eq!(dispatch.grid, (1, 1));
        assert_eq!(dispatch.block, (1, 1));
        assert!(dispatch.device.contains("RTX 4070 Ti"), "{}", dispatch.device);
        assert_eq!(dispatch.compute_cap, (8, 9));
        eprintln!(
            "GPU {} cc={:?} driver={} ptx={:?} bytes={} grid={:?} block={:?}",
            dispatch.device,
            dispatch.compute_cap,
            dispatch.driver_version,
            dispatch.ptx_isa,
            dispatch.ptx_bytes,
            dispatch.grid,
            dispatch.block
        );

        let mut compared = 0usize;
        for (stage_index, (cpu_stage, gpu_stage)) in
            expected.stages.iter().zip(&actual.stages).enumerate()
        {
            let cpu_bytes = stage_bytes(cpu_stage);
            let gpu_bytes = stage_bytes(gpu_stage);
            assert_eq!(
                cpu_bytes, gpu_bytes,
                "stage {stage_index} differs: CPU {} GPU {}",
                sha256_hex(&cpu_bytes),
                sha256_hex(&gpu_bytes)
            );
            compared += cpu_bytes.len();
        }
        assert_eq!(compared, 6_912);
    }
}
