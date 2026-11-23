//! Recovered PF8025/PF8026 194 -> 526 geometry and captured V2 topology.

use aurora_portrait::cuda_vface::Row;
use aurora_portrait::Refusal;

pub const VERTEX_COUNT: usize = 526;
pub const INDEX_COUNT: usize = 2982;
pub const TOPOLOGY_SHA256: &str =
    "707746096CCCCAF8DF2234DAF6605CB6E335BF56E656051F7078E64BC4EB9FB8";
pub const FACE_COEFF_SHA256: &str =
    "5F102C6057311F67648174AA60F67971637E17A811C5D697BB74EC12EC72F30B";
pub const TRIANGLE_WEIGHTS_SHA256: &str =
    "0F3EB71525714617C16D2D855D77FADC89D9290A9DA12C63A97089E1866CC422";
pub const OFFSET_UV_SHA256: &str =
    "24734F9CA9ABA22C86E571E085DC46122200F0D55E8951EADA118D7A2F1C613D";

const TOPOLOGY: &[u8] = include_bytes!("../assets/portrait/vface/topology2982.u16le");
const FACE_COEFF: &[u8] = include_bytes!("../assets/portrait/vface/face_coeff526.f32le");
const TRIANGLE_WEIGHTS: &[u8] =
    include_bytes!("../assets/portrait/vface/triangle_weights994.f32le");
const OFFSET_UV: &[u8] = include_bytes!("../assets/portrait/vface/uv526.f32le");
const GRID_2010_POSITIVE: &[u8] =
    include_bytes!("../assets/portrait/vface/atlas/Grid_2010_Positive.rgba8");
const GRID_2011_POSITIVE: &[u8] =
    include_bytes!("../assets/portrait/vface/atlas/Grid_2011_Positive.rgba8");
const GRID_2010_NEGATIVE: &[u8] =
    include_bytes!("../assets/portrait/vface/atlas/Grid_2010_Negative.rgba8");
const GRID_2011_NEGATIVE: &[u8] =
    include_bytes!("../assets/portrait/vface/atlas/Grid_2011_Negative.rgba8");

#[derive(Clone, Debug, PartialEq)]
pub struct V2MeshGeometry {
    pub framebuffer_positions_w1: Vec<[f32; 2]>,
    pub texcoord: Vec<[f32; 2]>,
    pub offset_texcoord: Vec<[f32; 2]>,
    pub face_scales: Vec<[f32; 2]>,
    pub ebo: Vec<u16>,
    pub mat2: [f32; 4],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VFaceAtlas {
    pub row: Row,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl VFaceAtlas {
    pub fn captured(row: Row) -> Self {
        let [_, _, width, height] = row.spec().rectangle;
        let pixels = match row {
            Row::LeftPositive => GRID_2010_POSITIVE,
            Row::RightPositive => GRID_2011_POSITIVE,
            Row::LeftNegative => GRID_2010_NEGATIVE,
            Row::RightNegative => GRID_2011_NEGATIVE,
        };
        Self {
            row,
            width,
            height,
            pixels: pixels.to_vec(),
        }
    }

    pub fn validate(&self) -> Result<(), Refusal> {
        let [_, _, width, height] = self.row.spec().rectangle;
        if (self.width, self.height) != (width, height)
            || self.pixels.len() != (width * height * 4) as usize
        {
            return Err(Refusal::ShapeMismatch("PF247/248 captured atlas"));
        }
        Ok(())
    }
}

pub fn from_landmarks(
    points194: &[[f32; 2]],
    width: u32,
    height: u32,
) -> Result<V2MeshGeometry, Refusal> {
    if points194.len() != 194
        || width == 0
        || height == 0
        || points194.iter().flatten().any(|value| !value.is_finite())
    {
        return Err(Refusal::ShapeMismatch("PF8025/PF8026 points194/frame"));
    }
    if hex(&super::sha256_32(TOPOLOGY)) != TOPOLOGY_SHA256
        || hex(&super::sha256_32(FACE_COEFF)) != FACE_COEFF_SHA256
        || hex(&super::sha256_32(TRIANGLE_WEIGHTS)) != TRIANGLE_WEIGHTS_SHA256
        || hex(&super::sha256_32(OFFSET_UV)) != OFFSET_UV_SHA256
    {
        return Err(Refusal::ValueOutOfRange("PF8025/PF8026 captured mesh digest"));
    }

    let positions = stage1(points194, StageMode::Position);
    let scale_positions = stage1(points194, StageMode::FaceScale);
    let texcoord = positions
        .iter()
        .map(|point| [point[0] / width as f32, point[1] / height as f32])
        .collect();
    let face_scales = build_face_scales(&scale_positions)?;
    let ebo = TOPOLOGY
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .collect::<Vec<_>>();
    if ebo.len() != INDEX_COUNT
        || positions.len() != VERTEX_COUNT
        || face_scales.len() != VERTEX_COUNT
        || ebo.iter().any(|&index| index as usize >= VERTEX_COUNT)
    {
        return Err(Refusal::ShapeMismatch("PF8025/PF8026 V2 mesh"));
    }

    Ok(V2MeshGeometry {
        framebuffer_positions_w1: positions,
        texcoord,
        offset_texcoord: f32_asset(OFFSET_UV),
        face_scales,
        ebo,
        mat2: mat2(points194, width, height),
    })
}

#[derive(Clone, Copy)]
enum StageMode {
    Position,
    FaceScale,
}

fn stage1(points: &[[f32; 2]], mode: StageMode) -> Vec<[f32; 2]> {
    let mut out = vec![[0.0; 2]; VERTEX_COUNT];
    out[0..5].copy_from_slice(&points[0..5]);
    out[5] = cubic_edge(points[3], points[4], points[5], points[6]);
    out[6] = points[5];
    out[7] = points[6];
    out[8..15].copy_from_slice(&points[7..14]);
    out[15] = cubic_edge(points[12], points[13], points[14], points[15]);
    out[16..20].copy_from_slice(&points[14..18]);

    let fan_a = NaturalSpline::new(&points[18..23]);
    for (index, point) in midpoints_5(&fan_a).iter().enumerate() {
        out[20 + index] = *point;
    }
    let fan_b = NaturalSpline::new(&points[22..26]);
    for (index, point) in midpoints_4(&fan_b)
        .iter()
        .chain(std::iter::once(&points[26]))
        .enumerate()
    {
        out[29 + index] = *point;
    }
    let brow_a = NaturalSpline::new(&points[36..41]);
    for (index, point) in midpoints_5(&brow_a).iter().enumerate() {
        out[37 + index] = *point;
    }
    let brow_b = NaturalSpline::new(&points[40..44]);
    for (index, point) in midpoints_4(&brow_b)
        .iter()
        .chain(std::iter::once(&points[44]))
        .enumerate()
    {
        out[46 + index] = *point;
    }

    out[54] = average(out[5], out[15]);
    out[55..74].copy_from_slice(&points[54..73]);
    out[74..108].copy_from_slice(&points[73..107]);
    let middle = NaturalSpline::new(&[
        average(points[63], points[73]),
        average(points[58], points[78]),
        average(points[70], points[83]),
    ]);
    let mut row = 108;
    for segment in 0..2 {
        for fraction in [1.0 / 3.0, 2.0 / 3.0, 1.0] {
            out[row] = middle.at(segment, fraction);
            row += 1;
        }
    }
    out[114] = average(points[70], points[83]);
    out[115..171].copy_from_slice(&points[109..165]);

    out[171] = average(points[18], points[109]);
    let left_a = NaturalSpline::new(&[
        points[111],
        average(points[60], points[111]),
        points[60],
    ]);
    let left_b = NaturalSpline::new(&[
        points[112],
        average(points[61], points[112]),
        points[61],
    ]);
    let left_c = NaturalSpline::new(&[
        points[113],
        average(points[62], points[113]),
        points[62],
    ]);
    let left_d = NaturalSpline::new(&[
        points[114],
        average(points[63], points[114]),
        points[63],
    ]);
    for (index, fraction) in [1.0 / 3.0, 2.0 / 3.0, 1.0].into_iter().enumerate() {
        out[172 + index] = left_a.at(0, fraction);
    }
    for (index, fraction) in [1.0 / 3.0, 2.0 / 3.0].into_iter().enumerate() {
        out[175 + index] = left_a.at(1, fraction);
    }
    for (index, fraction) in [0.4, 0.8].into_iter().enumerate() {
        out[177 + index] = left_b.at(0, fraction);
    }
    for (index, fraction) in [0.2, 0.6].into_iter().enumerate() {
        out[179 + index] = left_b.at(1, fraction);
    }
    for (index, fraction) in [0.25, 0.5, 0.75].into_iter().enumerate() {
        out[181 + index] = left_c.at(0, fraction);
        out[184 + index] = left_d.at(0, fraction);
    }
    out[187] = lerp(points[116], out[108], 1.0 / 3.0);
    out[188] = lerp(points[116], out[108], 2.0 / 3.0);
    out[189] = lerp(points[118], points[73], 1.0 / 3.0);
    out[190] = lerp(points[118], points[73], 2.0 / 3.0);

    out[191] = average(points[36], points[141]);
    let right_a = NaturalSpline::new(&[
        points[139],
        average(points[67], points[139]),
        points[67],
    ]);
    let right_b = NaturalSpline::new(&[
        points[138],
        average(points[68], points[138]),
        points[68],
    ]);
    let right_c = NaturalSpline::new(&[
        points[137],
        average(points[69], points[137]),
        points[69],
    ]);
    let right_d = NaturalSpline::new(&[
        points[136],
        average(points[70], points[136]),
        points[70],
    ]);
    for (index, fraction) in [1.0 / 6.0, 1.0 / 3.0, 0.5, 2.0 / 3.0, 5.0 / 6.0]
        .into_iter()
        .enumerate()
    {
        out[192 + index] = right_a.at(0, fraction);
    }
    for (index, fraction) in [0.2, 0.4, 0.6, 0.8].into_iter().enumerate() {
        out[197 + index] = right_b.at(0, fraction);
    }
    for (index, fraction) in [0.25, 0.5, 0.75].into_iter().enumerate() {
        out[201 + index] = right_c.at(0, fraction);
        out[204 + index] = right_d.at(0, fraction);
    }
    out[207] = lerp(points[134], out[114], 1.0 / 3.0);
    out[208] = lerp(points[134], out[114], 2.0 / 3.0);
    out[209] = lerp(points[132], points[83], 1.0 / 3.0);
    out[210] = lerp(points[132], points[83], 2.0 / 3.0);

    let one_third = 0.333_299_994_468_688_96f32;
    let cheek_a = NaturalSpline::new(&[
        out[189],
        lerp(points[122], points[92], one_third),
        points[108],
        lerp(points[128], points[86], one_third),
        out[209],
    ]);
    let cheek_b = NaturalSpline::new(&[
        out[190],
        lerp(points[122], points[92], 2.0 * one_third),
        points[107],
        lerp(points[128], points[86], 2.0 * one_third),
        out[210],
    ]);
    for (index, point) in midpoints_5(&cheek_a).iter().take(7).enumerate() {
        out[211 + index] = *point;
    }
    for (index, point) in midpoints_5(&cheek_b).iter().take(7).enumerate() {
        out[218 + index] = *point;
    }

    let chin_a = NaturalSpline::new(&[points[158], points[193], points[148]]);
    let chin_b = NaturalSpline::new(&[points[160], points[192], points[146]]);
    let chin_c = NaturalSpline::new(&[points[162], points[191], points[144]]);
    for (index, fraction) in [0.33, 0.67].into_iter().enumerate() {
        out[225 + index] = chin_a.at(0, fraction);
    }
    out[227] = chin_a.at(1, 0.0);
    for (index, fraction) in [0.33, 0.67].into_iter().enumerate() {
        out[228 + index] = chin_a.at(1, fraction);
    }
    for (index, fraction) in [0.25, 0.5, 0.75, 1.0].into_iter().enumerate() {
        out[230 + index] = chin_b.at(0, fraction);
    }
    for (index, fraction) in [0.25, 0.5, 0.75].into_iter().enumerate() {
        out[234 + index] = chin_b.at(1, fraction);
    }
    for (index, fraction) in [0.2, 0.4, 0.6, 0.8, 1.0].into_iter().enumerate() {
        out[237 + index] = chin_c.at(0, fraction);
    }
    for (index, fraction) in [0.2, 0.4, 0.6, 0.8].into_iter().enumerate() {
        out[242 + index] = chin_c.at(1, fraction);
    }

    let axis = Axis::new(points);
    for band in 1..=5 {
        let strength = band as f32 * 0.1;
        for index in 0..56 {
            out[246 + (band - 1) * 56 + index] =
                axis.transform(out[115 + index], points, mode, strength);
        }
    }
    out
}

fn build_face_scales(positions: &[[f32; 2]]) -> Result<Vec<[f32; 2]>, Refusal> {
    let coefficients = f32_scalar_asset(FACE_COEFF);
    let weights = f32_scalar_asset(TRIANGLE_WEIGHTS);
    let indices = TOPOLOGY
        .chunks_exact(2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]) as usize)
        .collect::<Vec<_>>();
    let mut accumulated = vec![0.0f32; VERTEX_COUNT];
    let mut counts = vec![0u32; VERTEX_COUNT];
    for triangle in 0..994 {
        let [ia, ib, ic] = [
            indices[triangle * 3],
            indices[triangle * 3 + 1],
            indices[triangle * 3 + 2],
        ];
        let [a, b, c] = [positions[ia], positions[ib], positions[ic]];
        let [ab, bc, ca] = [distance(a, b), distance(b, c), distance(c, a)];
        let semiperimeter = ((bc + ab) + ca) * 0.5;
        let mut heron = (semiperimeter - ab) * semiperimeter;
        heron *= semiperimeter - bc;
        heron *= semiperimeter - ca;
        if heron < 0.0 || heron.is_nan() {
            return Err(Refusal::ValueOutOfRange("PF8025 face-scale Heron"));
        }
        let contribution = if heron == 0.0 || weights[triangle] == 0.0 {
            0.0
        } else {
            heron.sqrt().sqrt() * weights[triangle]
        };
        for index in [ia, ib, ic] {
            accumulated[index] += contribution;
            counts[index] += 1;
        }
    }
    let mut out = vec![[1.0; 2]; VERTEX_COUNT];
    for index in 0..VERTEX_COUNT {
        if counts[index] != 0 {
            out[index] = [
                accumulated[index] / counts[index] as f32,
                distance(positions[55], positions[index]) * coefficients[index],
            ];
        }
    }
    out[55] = out[54];
    for (dst, a, b) in [
        (74, 75, 94), (96, 75, 94), (97, 93, 77), (107, 93, 77),
        (98, 91, 78), (106, 91, 78), (99, 90, 79), (105, 90, 79),
        (100, 89, 80), (104, 89, 80), (101, 87, 82), (103, 87, 82),
        (84, 85, 83), (102, 85, 83),
    ] {
        out[dst] = average(out[a], out[b]);
    }
    for band in 1..=5 {
        let factor = (1.0 + band as f32 * -0.800_001_025_199_890_1).max(0.0);
        for index in 246 + (band - 1) * 56..246 + band * 56 {
            out[index][0] *= factor;
        }
    }
    Ok(out)
}

struct Axis {
    sin: f32,
    cos: f32,
    center: [f32; 2],
}

impl Axis {
    fn new(points: &[[f32; 2]]) -> Self {
        let (vx, vy) = (
            points[141][0] - points[109][0],
            points[141][1] - points[109][1],
        );
        let length = (vx * vx + vy * vy).sqrt();
        let cosine = ((306.0 * vx - 4.6 * vy) / (length * 306.034_58)).clamp(-1.0, 1.0);
        let unsigned = cosine.acos();
        let phi = if 306.0 * vy + 4.6 * vx < 0.0 {
            unsigned
        } else {
            -unsigned
        };
        Self {
            sin: phi.sin(),
            cos: phi.cos(),
            center: [
                0.5 * (points[111][0] + points[139][0]),
                0.5 * (points[153][1] + points[125][1]),
            ],
        }
    }

    fn transform(
        &self,
        point: [f32; 2],
        points: &[[f32; 2]],
        mode: StageMode,
        strength: f32,
    ) -> [f32; 2] {
        match mode {
            StageMode::FaceScale => {
                let center = points[57];
                [
                    center[0] + (1.0 + strength) * (point[0] - center[0]),
                    center[1] + (1.0 + strength) * (point[1] - center[1]),
                ]
            }
            StageMode::Position => {
                let m00 = 1.5 * self.sin * self.sin + 2.75 * self.cos * self.cos;
                let m01 = -1.25 * self.sin * self.cos;
                let m11 = 1.5 * self.cos * self.cos + 2.75 * self.sin * self.sin;
                let dx = strength * (point[0] - self.center[0]);
                let dy = strength * (point[1] - self.center[1]);
                [
                    self.center[0] + dx * m00 + dy * m01,
                    self.center[1] + dx * m01 + dy * m11,
                ]
            }
        }
    }
}

struct NaturalSpline {
    points: Vec<[f32; 2]>,
    knots: Vec<f64>,
    second_x: Vec<f64>,
    second_y: Vec<f64>,
}

impl NaturalSpline {
    fn new(points: &[[f32; 2]]) -> Self {
        let count = points.len();
        let mut knots = vec![0.0; count];
        for index in 1..count {
            knots[index] = knots[index - 1] + distance(points[index - 1], points[index]) as f64;
            if knots[index] <= knots[index - 1] {
                knots[index] = knots[index - 1] + f64::EPSILON;
            }
        }
        let second_x = natural_second(&knots, points.iter().map(|p| p[0] as f64).collect());
        let second_y = natural_second(&knots, points.iter().map(|p| p[1] as f64).collect());
        Self {
            points: points.to_vec(),
            knots,
            second_x,
            second_y,
        }
    }

    fn at(&self, segment: usize, fraction: f32) -> [f32; 2] {
        let x = self.knots[segment]
            + (self.knots[segment + 1] - self.knots[segment])
                * fraction.clamp(0.0, 1.0) as f64;
        [
            spline_eval(&self.knots, &self.second_x, &self.points, segment, x, 0),
            spline_eval(&self.knots, &self.second_y, &self.points, segment, x, 1),
        ]
    }
}

fn natural_second(knots: &[f64], values: Vec<f64>) -> Vec<f64> {
    let count = values.len();
    let mut second = vec![0.0; count];
    if count <= 2 {
        return second;
    }
    let mut work = vec![0.0; count];
    for index in 1..count - 1 {
        let sig =
            (knots[index] - knots[index - 1]) / (knots[index + 1] - knots[index - 1]);
        let p = sig * second[index - 1] + 2.0;
        second[index] = (sig - 1.0) / p;
        work[index] = (values[index + 1] - values[index]) / (knots[index + 1] - knots[index])
            - (values[index] - values[index - 1]) / (knots[index] - knots[index - 1]);
        work[index] = (6.0 * work[index] / (knots[index + 1] - knots[index - 1])
            - sig * work[index - 1])
            / p;
    }
    for index in (0..count - 1).rev() {
        second[index] = second[index] * second[index + 1] + work[index];
    }
    second
}

fn spline_eval(
    knots: &[f64],
    second: &[f64],
    points: &[[f32; 2]],
    segment: usize,
    x: f64,
    lane: usize,
) -> f32 {
    let h = knots[segment + 1] - knots[segment];
    let a = (knots[segment + 1] - x) / h;
    let b = (x - knots[segment]) / h;
    (a * points[segment][lane] as f64
        + b * points[segment + 1][lane] as f64
        + ((a * a * a - a) * second[segment] + (b * b * b - b) * second[segment + 1])
            * h
            * h
            / 6.0) as f32
}

fn midpoints_5(spline: &NaturalSpline) -> Vec<[f32; 2]> {
    let mut out = vec![spline.points[0]];
    for segment in 0..4 {
        out.push(spline.at(segment, 0.5));
        out.push(spline.points[segment + 1]);
    }
    out
}

fn midpoints_4(spline: &NaturalSpline) -> Vec<[f32; 2]> {
    let mut out = Vec::new();
    for segment in 0..3 {
        out.push(spline.at(segment, 0.5));
        out.push(spline.points[segment + 1]);
    }
    out
}

fn cubic_edge(a: [f32; 2], b: [f32; 2], c: [f32; 2], d: [f32; 2]) -> [f32; 2] {
    let p = average(b, c);
    let q = average(a, d);
    let middle = average(p, q);
    [p[0] + 0.5 * (p[0] - middle[0]), p[1] + 0.5 * (p[1] - middle[1])]
}

fn mat2(points: &[[f32; 2]], width: u32, height: u32) -> [f32; 4] {
    let (vx, vy) = (
        points[141][0] - points[109][0],
        points[141][1] - points[109][1],
    );
    let length = (vx * vx + vy * vy).sqrt();
    let mut theta = (vx / length).clamp(-1.0, 1.0).acos();
    if vy <= 0.0 {
        theta = -theta;
    }
    let (c, s) = (theta.cos(), theta.sin());
    [
        1000.0 * c / width as f32,
        -1500.0 * s / width as f32,
        1000.0 * s / height as f32,
        1500.0 * c / height as f32,
    ]
}

fn f32_asset(bytes: &[u8]) -> Vec<[f32; 2]> {
    bytes
        .chunks_exact(8)
        .map(|bytes| {
            [
                f32::from_le_bytes(bytes[0..4].try_into().unwrap()),
                f32::from_le_bytes(bytes[4..8].try_into().unwrap()),
            ]
        })
        .collect()
}

fn f32_scalar_asset(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn average(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])]
}

fn lerp(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

fn distance(a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    (dx * dx + dy * dy).sqrt()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02X}")).collect()
}
