//! Type1012 forehead PF30080..PF30083, `EnableCommonMesh=true` branch.
//!
//! This candidate targets the *current* branch, not the alternate FBO path.
//! The branch is the same geometry consumer:
//!
//! ```text
//! 0x4C96A60  cmp [cell+0x2c5], 1
//! 0x4C96AB2  jmp 0x4C96AD0                 ; common mesh (current)
//! 0x4C96ACB  jmp 0x4C972C0                 ; alternate per-part FBO
//! 0x4C96AD0  call [cell vtable+0x68] -> 0x4C96810
//! 0x4C96871  -> 0x4C96080
//! 0x4C962BF  call 0x4C96360               ; degree map + part builder
//! 0x4C96703  call [part+0x50] vptr+0x10
//!            Type1012 vptr 0x946E870, +0x10 = 0x7078480
//! ```
//!
//! `0x4C972C0` reaches the same `0x4C96360 -> vtable+0x10 = 0x7078480`; it
//! additionally normalizes A by `1/(W,H)` and draws into a 280x280 FBO.
//! Common mesh instead hands the payload to `0x4C8E9F0`. The 136-point A/B,
//! 135 weights and 705-index buffer are therefore identical in both branches.
//!
//! Producer constants (fixed PE, verified by the R1b disassembly scripts):
//!
//! ```text
//! 0x7078098  payload+0x10 = 0x88      -> 136 vertices
//! 0x707809F  alloc 0x440              -> 136 * 8 bytes A
//! 0x70780BF  alloc 0x440              -> 136 * 8 bytes B
//! 0x70780E0  payload+0x20 = 0x2C1     -> 705 indices
//! 0x70780E7  alloc 0x582              -> 705 * 2 bytes
//! 0x70780F5  lea rdx, 0x946E290       -> initial index table
//! 0x707808A  width=27, height=5       -> 135 ParamList entries
//! 0x4C9647B  control stride 0x68
//! 0x4C965C0  degree -> output[ControlIndex]
//! ```
//!
//! This module is standalone: it has no crate or GPU dependency and refuses
//! any missing/misshapen input instead of synthesizing landmarks.

/// One 2D point in native pixel space (`face+0x924` float2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

impl core::ops::Add for Point {
    type Output = Point;
    fn add(self, other: Point) -> Point {
        Point::new(self.x + other.x, self.y + other.y)
    }
}

impl core::ops::Sub for Point {
    type Output = Point;
    fn sub(self, other: Point) -> Point {
        Point::new(self.x - other.x, self.y - other.y)
    }
}

impl core::ops::Mul<f32> for Point {
    type Output = Point;
    fn mul(self, scale: f32) -> Point {
        Point::new(self.x * scale, self.y * scale)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// part 35 / 37, `RenderDegreeRanges [0,0,-1]`
    Negative,
    /// part 36 / 38, `RenderDegreeRanges [1,0,0]`
    Positive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    MissingPoints,
    MissingWeights,
    MissingTopology,
    NonFinite(&'static str),
    BadDimension,
    BadIndexTable,
}

pub const VERTEX_COUNT: usize = 136;
pub const INDEX_COUNT: usize = 705;
pub const WEIGHT_COUNT: usize = 135;


pub struct Inputs<'a> {
    /// Caller-owned topology: exactly INDEX_COUNT indices below VERTEX_COUNT.
    /// An empty slice is refused; no bundled or synthesized topology is used.
    pub topology: &'a [u16],
    /// `face+0x924`, 414 float2 in native working coordinates.
    pub points414: &'a [Point],
    /// `ParamList` for the chosen part, exactly 135 floats.
    pub weights135: &'a [f32],
    /// part 35/36 (width) or 37/38 (height) direction row.
    pub direction: Direction,
    /// Native values in `RenderParamRanges` units: `[PF30080, PF30081]`
    /// or `[PF30082, PF30083]`.
    pub native: [f32; 2],
    pub width: u32,
    pub height: u32,
}

#[derive(Debug)]
pub struct Mesh {
    /// 136 undeformed grid points, already clamped to the image rect.
    pub a: [Point; VERTEX_COUNT],
    /// 136 deformed points; `0x7078840` writes this copy only.
    pub b: [Point; VERTEX_COUNT],
    /// Doubled degree slots, as consumed by `0x7078480` (`addps xmm6,xmm6`).
    pub degree: [f32; 2],
    /// `0x946E290` initial table. `0x60C0A60` may replace it via live
    /// `0x60F2B60`; that replacement is not synthesized here.
    pub initial_indices: [u16; INDEX_COUNT],
}

/// `0x7078620` block offsets / 8: 13 swapped float2 blocks plus one point.
const SLOT_BLOCKS: [usize; 14] = [
    115, 169, 167, 165, 163, 161, 159, 157, 155, 153, 151, 149, 147, 146,
];

fn y_weights() -> [f32; 5] {
    [
        0.0,
        f32::from_bits(0x3E4C_CCCD),
        f32::from_bits(0x3F19_9999),
        f32::from_bits(0xBE4C_CCE1),
        f32::from_bits(0xBECC_CCC9),
    ]
}

/// `0x4C96360` three-knot float32 interpolation; outside stays unchanged.
fn map_degree(value: f32, knots: &[f32; 3], values: &[f32; 3]) -> f32 {
    for i in 1..3 {
        if value < knots[i - 1] || knots[i] < value {
            continue;
        }
        let span = knots[i] - knots[i - 1];
        return values[i - 1] + (values[i] - values[i - 1]) * ((value - knots[i - 1]) / span);
    }
    value
}

/// `0x7078620`: 13 `pshufd 0x4E` blocks, five rows per column, anchor last.
fn build_grid(points414: &[Point], anchor: Point, width: f32, height: f32) -> [Point; VERTEX_COUNT] {
    let mut slots = [Point::new(0.0, 0.0); 27];
    let mut at = 0;
    for (block, &base) in SLOT_BLOCKS.iter().enumerate() {
        if block == 13 {
            slots[at] = points414[base];
            at += 1;
            continue;
        }
        // pshufd 0x4E swaps the two float2 inside each 16-byte block.
        slots[at] = points414[base + 1];
        slots[at + 1] = points414[base];
        at += 2;
    }

    let mut grid = [Point::new(0.0, 0.0); VERTEX_COUNT];
    for (column, slot) in slots.iter().enumerate() {
        for row in 0..5 {
            let weight = y_weights()[row] * 1.2;
            let p = *slot + (anchor - *slot) * weight;
            grid[row * 27 + column] =
                Point::new(p.x.clamp(0.0, width), p.y.clamp(0.0, height));
        }
    }
    grid[135] = anchor;
    grid
}

/// `0x7078840`: `term = weight * distance * 0.1`, cosine-free radial push
/// along `normalize(anchor - point)`, edge clamp `0.05`.
fn deform(
    grid: &[Point; VERTEX_COUNT],
    weights: &[f32],
    degree: [f32; 2],
    anchor: Point,
    reference: Point,
    width: f32,
    height: f32,
) -> [Point; VERTEX_COUNT] {
    let mut out = *grid;
    let ax = anchor.x - reference.x;
    let ay = anchor.y - reference.y;
    let mut norm = (ax * ax + ay * ay).sqrt();
    if norm == 0.0 {
        norm = 1.0;
    }
    let default_dir = (ax / norm, ay / norm);

    for i in 0..135 {
        let weight = weights[i];
        if weight == 0.0 {
            continue;
        }
        let px = grid[i].x;
        let py = grid[i].y;

        let mut edge = if px <= 20.0 || px >= width - 20.0 {
            (width - px).abs().min(px.abs()) * 0.05
        } else {
            1.0
        };
        if py <= 20.0 || py >= height - 20.0 {
            edge *= py.abs() * 0.05;
        }

        let (dir_x, dir_y) = if i % 27 != 13 {
            let dx = anchor.x - px;
            let dy = anchor.y - py;
            let length = (dx * dx + dy * dy).sqrt();
            (dx / length, dy / length)
        } else {
            default_dir
        };

        let distance =
            ((anchor.x - px) * (anchor.x - px) + (anchor.y - py) * (anchor.y - py)).sqrt();
        let row = i % 27;
        let factor = if row <= 12 {
            f64::from(degree[0])
        } else if row == 13 {
            f64::from(degree[0] + degree[1]) * 0.5
        } else {
            f64::from(degree[1])
        };
        let scalar = (f64::from(weight * distance) * 0.1 * factor) as f32;
        out[i] = Point::new(px + edge * scalar * dir_x, py + edge * scalar * dir_y);
    }
    out
}

/// Callable current-branch producer for one part (35..38).
pub fn build(input: &Inputs<'_>) -> Result<Mesh, Refusal> {
    if input.points414.len() < 242 {
        return Err(Refusal::MissingPoints);
    }
    if input.weights135.len() != WEIGHT_COUNT {
        return Err(Refusal::MissingWeights);
    }
    if input.width == 0 || input.height == 0 {
        return Err(Refusal::BadDimension);
    }
    for point in &input.points414[..242] {
        if !point.x.is_finite() || !point.y.is_finite() {
            return Err(Refusal::NonFinite("points414"));
        }
    }
    for &weight in input.weights135 {
        if !weight.is_finite() {
            return Err(Refusal::NonFinite("weights135"));
        }
    }
    if !input.native[0].is_finite() || !input.native[1].is_finite() {
        return Err(Refusal::NonFinite("native"));
    }

    if input.topology.is_empty() {
        return Err(Refusal::MissingTopology);
    }
    if input.topology.len() != INDEX_COUNT
        || input.topology.iter().any(|&index| index as usize >= VERTEX_COUNT)
    {
        return Err(Refusal::BadIndexTable);
    }

    let ranges = match input.direction {
        Direction::Negative => [0.0, 0.0, -1.0],
        Direction::Positive => [1.0, 0.0, 0.0],
    };
    // 0x70784B9 doubles both slots before 0x7078840.
    let degree = [
        map_degree(input.native[0], &[0.0, 0.5, 1.0], &ranges) * 2.0,
        map_degree(input.native[1], &[0.0, 0.5, 1.0], &ranges) * 2.0,
    ];

    let anchor = input.points414[54];
    let reference = input.points414[159];
    let grid = build_grid(input.points414, anchor, input.width as f32, input.height as f32);
    let b = deform(
        &grid,
        input.weights135,
        degree,
        anchor,
        reference,
        input.width as f32,
        input.height as f32,
    );

    let mut indices = [0u16; INDEX_COUNT];
    indices.copy_from_slice(input.topology);

    Ok(Mesh {
        a: grid,
        b,
        degree,
        initial_indices: indices,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic test topology only; never used by the runtime producer.
    const TOPOLOGY: [u16; INDEX_COUNT] = [0; INDEX_COUNT];

    fn points() -> Vec<Point> {
        (0..414)
            .map(|i| Point::new((i % 40) as f32 * 50.0, (i / 40) as f32 * 40.0))
            .collect()
    }

    fn weights() -> Vec<f32> {
        let mut w = vec![0.0f32; 135];
        w[0] = 1.0;
        w[26] = 1.0;
        w[13] = 1.0;
        w
    }

    fn input<'a>(
        points414: &'a [Point],
        weights135: &'a [f32],
        native: [f32; 2],
        direction: Direction,
    ) -> Inputs<'a> {
        Inputs {
            topology: &TOPOLOGY,
            points414,
            weights135,
            direction,
            native,
            width: 100_000,
            height: 80_000,
        }
    }

    #[test]
    fn caller_topology_is_preserved() {
        let mesh = build(&input(&points(), &weights(), [0.5, 0.5], Direction::Positive)).unwrap();
        assert_eq!(mesh.initial_indices, TOPOLOGY);
        assert_eq!(mesh.initial_indices.len(), 705);
        assert!(mesh.initial_indices.iter().all(|&i| (i as usize) < 136));
    }

    #[test]
    fn pshufd_swaps_each_float2_block() {
        let mut p = points();
        for (i, point) in p.iter_mut().enumerate() {
            *point = Point::new(i as f32, 0.0);
        }
        let mesh = build(&input(&p, &weights(), [0.5, 0.5], Direction::Positive)).unwrap();
        assert_eq!(mesh.a[0], Point::new(116.0, 0.0));
        assert_eq!(mesh.a[1], Point::new(115.0, 0.0));
        assert_eq!(mesh.a[2], Point::new(170.0, 0.0));
        assert_eq!(mesh.a[26], Point::new(146.0, 0.0));
        assert_eq!(mesh.a[135], p[54]);
    }

    #[test]
    fn mid_native_is_identity() {
        let mesh = build(&input(&points(), &weights(), [0.5, 0.5], Direction::Negative)).unwrap();
        assert_eq!(mesh.degree, [0.0, 0.0]);
        assert_eq!(mesh.a, mesh.b);
    }

    #[test]
    fn direction_controls_degree_sign() {
        let negative =
            build(&input(&points(), &weights(), [1.0, 0.0], Direction::Negative)).unwrap();
        assert_eq!(negative.degree, [-2.0, 0.0]);
        let positive =
            build(&input(&points(), &weights(), [1.0, 0.0], Direction::Positive)).unwrap();
        assert_eq!(positive.degree, [0.0, 2.0]);
        let positive_half =
            build(&input(&points(), &weights(), [0.0, 0.0], Direction::Positive)).unwrap();
        assert_eq!(positive_half.degree, [2.0, 2.0]);
    }

    #[test]
    fn deformation_moves_only_active_rows_and_keeps_pixel_space() {
        let mesh = build(&input(&points(), &weights(), [1.0, 1.0], Direction::Negative)).unwrap();
        assert_ne!(mesh.a[0], mesh.b[0]);
        assert_ne!(mesh.a[26], mesh.b[26]);
        assert_eq!(mesh.a[1], mesh.b[1]);
        // A stays in native pixel coordinates; no 1/(W,H) normalization.
        assert!(mesh.a.iter().any(|p| p.x > 1.0 || p.y > 1.0));
    }

    #[test]
    fn grid_is_clamped_to_the_image_rect() {
        let mut p = points();
        p[54] = Point::new(5000.0, -5000.0);
        let mesh = build(&Inputs {
            topology: &TOPOLOGY,
            points414: &p,
            weights135: &weights(),
            direction: Direction::Positive,
            native: [0.5, 0.5],
            width: 1000,
            height: 800,
        })
        .unwrap();
        assert!(mesh.a[..135].iter().all(|q| (0.0..=1000.0).contains(&q.x)));
        assert!(mesh.a[..135].iter().all(|q| (0.0..=800.0).contains(&q.y)));
        assert_eq!(mesh.a[135], Point::new(5000.0, -5000.0));
    }

    #[test]
    fn refusals_are_typed() {
        let short_points = &points()[..100];
        assert_eq!(
            build(&input(short_points, &weights(), [0.5, 0.5], Direction::Positive)).unwrap_err(),
            Refusal::MissingPoints
        );
        let short_weights = &weights()[..134];
        assert_eq!(
            build(&input(&points(), short_weights, [0.5, 0.5], Direction::Positive)).unwrap_err(),
            Refusal::MissingWeights
        );
        let mut p = points();
        p[10].x = f32::NAN;
        assert_eq!(
            build(&input(&p, &weights(), [0.5, 0.5], Direction::Positive)).unwrap_err(),
            Refusal::NonFinite("points414")
        );
        let mut bad = vec![0.0f32; 135];
        bad[3] = f32::INFINITY;
        assert_eq!(
            build(&input(&points(), &bad, [0.5, 0.5], Direction::Positive)).unwrap_err(),
            Refusal::NonFinite("weights135")
        );
    }

    #[test]
    fn row_13_uses_averaged_degrees() {
        let mut w = vec![0.0f32; 135];
        w[13] = 1.0;
        let mesh = build(&input(&points(), &w, [1.0, 0.0], Direction::Negative)).unwrap();
        let row13 = mesh.a[13];
        let expected_dir = {
            let anchor = points()[54];
            let reference = points()[159];
            let dx = anchor.x - reference.x;
            let dy = anchor.y - reference.y;
            let len = (dx * dx + dy * dy).sqrt();
            (dx / len, dy / len)
        };
        let distance = {
            let anchor = points()[54];
            ((anchor.x - row13.x).powi(2) + (anchor.y - row13.y).powi(2)).sqrt()
        };
        // row 13 factor = (degree0 + degree1) * 0.5 = (-2 + 0) * 0.5
        let scalar = (f64::from(distance) * 0.1 * f64::from(-1.0)) as f32;
        let expected = Point::new(
            row13.x + scalar * expected_dir.0,
            row13.y + scalar * expected_dir.1,
        );
        assert!((mesh.b[13].x - expected.x).abs() < 1e-3);
        assert!((mesh.b[13].y - expected.y).abs() < 1e-3);
    }
    #[test]
    fn missing_malformed_and_out_of_range_topology_are_refused() {
        let p = points();
        let w = weights();
        let mut args = input(&p, &w, [0.5, 0.5], Direction::Positive);
        args.topology = &[];
        assert_eq!(build(&args).unwrap_err(), Refusal::MissingTopology);
        args.topology = &[0; INDEX_COUNT - 1];
        assert_eq!(build(&args).unwrap_err(), Refusal::BadIndexTable);
        let mut indices = [0; INDEX_COUNT];
        indices[INDEX_COUNT - 1] = VERTEX_COUNT as u16;
        args.topology = &indices;
        assert_eq!(build(&args).unwrap_err(), Refusal::BadIndexTable);
        let mut valid_indices = indices;
        valid_indices[INDEX_COUNT - 1] = (VERTEX_COUNT - 1) as u16;
        args.topology = &valid_indices;
        assert_eq!(build(&args).unwrap().initial_indices, valid_indices);
    }

}
