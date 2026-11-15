//! `FaceReshape24` geometry rows 1..10 (PF247..PF30093).
//!
//! Ported from the P27 final path
//! (`p27-slim-face-final-path/source/P27FinalPath.cs`,
//! `P27CorrectedRows.cs`, `RecoveredVendorMath.cs`). Every producer is pure
//! CPU math over caller-supplied same-job arrays: no synthesized landmarks, no
//! ellipse fallback, no unknown EBO.
//!
//! `Refusal` has no `NonFinite` variant, so a non-finite input or a value
//! outside its recovered domain both refuse as `ValueOutOfRange(field)`.
//! Missing vendor arrays refuse as `MissingInput`, wrong array lengths as
//! `ShapeMismatch`.
//!
//! These rows output geometry (vertex deltas / texture offsets), not pixels,
//! so they do not read `Image`. The one exception is the slim ROI texel, which
//! callers decode with [`Texel::from_image`].

use crate::{Access, EffectMeta, Image, Refusal};

/// One 2D point in pixel space.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    fn finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
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

impl core::ops::Mul<Point> for f32 {
    type Output = Point;
    fn mul(self, point: Point) -> Point {
        point * self
    }
}

/// One RGBA atlas texel used by the V2 texture-offset rows (PF8025/8026,
/// PF247/248). Values are 0..1, exactly like the vendor float4 upload.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Texel {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Texel {
    /// Decodes one pixel of an RGBA8 atlas so callers can hand the slim/V-face
    /// rows the same float4 the vendor shader receives.
    pub fn from_image(image: &Image, x: u32, y: u32) -> Self {
        let [r, g, b, a] = image.pixel(x, y);
        Self {
            r: r as f32 / 255.0,
            g: g as f32 / 255.0,
            b: b as f32 / 255.0,
            a: a as f32 / 255.0,
        }
    }
}

/// One vertex move: `target - source` is the delta the warp applies.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VertexDelta {
    pub index: usize,
    pub source: Point,
    pub target: Point,
}

/// Texture-space offset added to the sampled ROI coordinate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TextureOffset {
    pub overlay_index: u8,
    pub pf: u32,
    pub u: f32,
    pub v: f32,
}

/// Per-part warp shared by every point-mapping row.
#[derive(Clone, Debug)]
pub enum Warp {
    Radial(RadialMesh),
    ScaleAbout { center: Point, scale: f32 },
}

impl Warp {
    pub fn apply(&self, point: Point) -> Point {
        match self {
            Warp::Radial(mesh) => mesh.warp(point),
            Warp::ScaleAbout { center, scale } => {
                point + (point - *center) * *scale
            }
        }
    }
}

/// One produced part. `part` and `pf` are the vendor part id and ParamFlag.
#[derive(Clone, Debug)]
pub struct PartGeometry {
    pub part: u8,
    pub pf: u32,
    pub degree: f32,
    pub second_degree: f32,
    pub deltas: Vec<VertexDelta>,
    pub warp: Option<Warp>,
}

/// Result of one geometry row.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub priority: u8,
    pub parts: Vec<PartGeometry>,
    pub offsets: Vec<TextureOffset>,
}

/// Per-part arrays from the pinned `effect.json` (`P27VendorSource`).
#[derive(Clone, Copy, Debug)]
pub struct PartArrays<'a> {
    pub part: u8,
    pub parameters: &'a [f32],
    pub factors: &'a [f32],
}

/// Looks up a part by vendor part id, never by list position.
pub struct VendorParts<'a> {
    pub parts: &'a [PartArrays<'a>],
}

impl<'a> VendorParts<'a> {
    fn get(&self, part: u8) -> Option<&'a PartArrays<'a>> {
        self.parts.iter().find(|entry| entry.part == part)
    }
}

// --------------------------------------------------------------- registry

const TEMPLE_INPUTS: &[&str] = &[
    "face+0x924 prepared 237+ points",
    "ui -100..100 (native = ui/100)",
    "effect.json parts 23/24 parameter+factor",
    "image width/height",
];
const CHEEK_INPUTS: &[&str] = &[
    "33 face+0x368 points",
    "face+0x1B0 center",
    "ui -100..100",
    "effect.json parts 33/34 parameter+factor",
    "image width/height",
];
const JAW_INPUTS: &[&str] = &[
    "33 face+0x368 points",
    "face+0x1B0 center",
    "left/right ui -100..100",
    "effect.json parts 29..32 parameter+factor",
    "image width/height",
];
const CHIN_INPUTS: &[&str] = &[
    "33 face+0x368 points",
    "face+0x1B0 center",
    "ui -100..100",
    "effect.json parts 17..20 parameter+factor",
    "image width/height",
];
const WIDTH_INPUTS: &[&str] = &[
    "414 face+0x924 points",
    "ui -100..100",
    "effect.json parts 25..28 parameter+factor",
    "image width/height",
];
const FOREHEAD_INPUTS: &[&str] = &[
    "27 source slots",
    "135 negative + 135 positive weights",
    "anchor + reference points",
    "left/right ui -100..100",
    "image width/height",
];
const OVERALL_INPUTS: &[&str] = &[
    "6 eye landmarks",
    "72 + 72 neutral patch",
    "left/right ui -100..100",
];

/// Only ParamFlags whose producer below is callable today.
pub const META: &[EffectMeta] = &[
    EffectMeta {
        pf: 247,
        key: "face_v",
        label: "V face",
        access: Access::Cpu,
        inputs: &["ui -100..100", "roi atlas texel (RGBA 0..1)"],
    },
    EffectMeta {
        pf: 248,
        key: "face_v",
        label: "V face",
        access: Access::Cpu,
        inputs: &["ui -100..100", "roi atlas texel (RGBA 0..1)"],
    },
    EffectMeta {
        pf: 242,
        key: "face_chin",
        label: "Sharp chin",
        access: Access::Cpu,
        inputs: CHIN_INPUTS,
    },
    EffectMeta {
        pf: 243,
        key: "face_chin",
        label: "Sharp chin",
        access: Access::Cpu,
        inputs: CHIN_INPUTS,
    },
    EffectMeta {
        pf: 8023,
        key: "face_jaw",
        label: "Jaw",
        access: Access::Cpu,
        inputs: JAW_INPUTS,
    },
    EffectMeta {
        pf: 8024,
        key: "face_jaw",
        label: "Jaw",
        access: Access::Cpu,
        inputs: JAW_INPUTS,
    },
    EffectMeta {
        pf: 8025,
        key: "slim",
        label: "Slim face",
        access: Access::Cpu,
        inputs: &["ui 0..100", "roi atlas texel (RGBA 0..1)"],
    },
    EffectMeta {
        pf: 8026,
        key: "slim",
        label: "Slim face",
        access: Access::Cpu,
        inputs: &["ui 0..100", "roi atlas texel (RGBA 0..1)"],
    },
    EffectMeta {
        pf: 8027,
        key: "face_cheek",
        label: "Cheek",
        access: Access::Cpu,
        inputs: CHEEK_INPUTS,
    },
    EffectMeta {
        pf: 8028,
        key: "face_cheek",
        label: "Cheek",
        access: Access::Cpu,
        inputs: CHEEK_INPUTS,
    },
    EffectMeta {
        pf: 8029,
        key: "face_temple",
        label: "Temple",
        access: Access::Cpu,
        inputs: TEMPLE_INPUTS,
    },
    EffectMeta {
        pf: 8030,
        key: "face_temple",
        label: "Temple",
        access: Access::Cpu,
        inputs: TEMPLE_INPUTS,
    },
    EffectMeta {
        pf: 8050,
        key: "face_overall_size",
        label: "Overall size",
        access: Access::Cpu,
        inputs: OVERALL_INPUTS,
    },
    EffectMeta {
        pf: 8051,
        key: "face_overall_size",
        label: "Overall size",
        access: Access::Cpu,
        inputs: OVERALL_INPUTS,
    },
    EffectMeta {
        pf: 30080,
        key: "face_forehead_width",
        label: "Forehead width",
        access: Access::Cpu,
        inputs: FOREHEAD_INPUTS,
    },
    EffectMeta {
        pf: 30081,
        key: "face_forehead_width",
        label: "Forehead width",
        access: Access::Cpu,
        inputs: FOREHEAD_INPUTS,
    },
    EffectMeta {
        pf: 30082,
        key: "face_forehead_height",
        label: "Forehead height",
        access: Access::Cpu,
        inputs: FOREHEAD_INPUTS,
    },
    EffectMeta {
        pf: 30083,
        key: "face_forehead_height",
        label: "Forehead height",
        access: Access::Cpu,
        inputs: FOREHEAD_INPUTS,
    },
    EffectMeta {
        pf: 30092,
        key: "face_width",
        label: "Face width",
        access: Access::Cpu,
        inputs: WIDTH_INPUTS,
    },
    EffectMeta {
        pf: 30093,
        key: "face_width",
        label: "Face width",
        access: Access::Cpu,
        inputs: WIDTH_INPUTS,
    },
];

// ------------------------------------------------------------ degree helpers

/// `0x4C96360`: piecewise linear over `knots`; values outside stay unchanged.
fn map(value: f32, knots: &[f32], values: &[f32]) -> f32 {
    for i in 1..knots.len() {
        if value < knots[i - 1] || knots[i] < value {
            continue;
        }
        let span = knots[i] - knots[i - 1];
        return values[i - 1] + (values[i] - values[i - 1]) * ((value - knots[i - 1]) / span);
    }
    value
}

fn check_degree(degree: f32, field: &'static str) -> Result<f32, Refusal> {
    if !degree.is_finite() || degree.abs() > 1.5 {
        return Err(Refusal::ValueOutOfRange(field));
    }
    Ok(degree)
}

fn signed_ui(ui: f32, field: &'static str) -> Result<f32, Refusal> {
    if !ui.is_finite() || !(-100.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange(field));
    }
    Ok((ui + 100.0) / 200.0)
}

fn unsigned_ui(ui: f32, field: &'static str) -> Result<f32, Refusal> {
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange(field));
    }
    Ok(ui / 100.0)
}

const KNOTS: [f32; 3] = [0.0, 0.5, 1.0];
const NEGATIVE: [f32; 3] = [1.0, 0.0, 0.0];
const POSITIVE: [f32; 3] = [0.0, 0.0, 1.0];

// --------------------------------------------------------------- radial mesh

/// Shared radial warp of Type1014/1015/1016: `sign(degree) * degree^2 *
/// falloff * factor` along `target - source`, with cosine falloff in
/// normalized image space.
#[derive(Clone, Debug)]
pub struct RadialMesh {
    src_n: Vec<Point>,
    dst_n: Vec<Point>,
    limit: Vec<f32>,
    factors: Vec<f32>,
    active: Vec<usize>,
    sign: f32,
    degree_squared: f32,
    split_degree: f32,
    split_at: usize,
    ratio: f32,
    width: f32,
    height: f32,
    outside: bool,
    fixed_source: bool,
}

pub struct WheelMeshArgs<'a> {
    pub source: &'a [Point],
    pub target: &'a [Point],
    pub factors: &'a [f32],
    pub active: &'a [usize],
    pub degree: f32,
    pub width: u32,
    pub height: u32,
    pub outside: bool,
    pub split_at: usize,
    pub split_degree: f32,
    pub fixed_source: bool,
}

fn radial_mesh(args: WheelMeshArgs<'_>) -> RadialMesh {
    let width = args.width as f32;
    let height = args.height as f32;
    let count = args.source.len();
    let mut src_n = Vec::with_capacity(count);
    let mut dst_n = Vec::with_capacity(count);
    let mut limit = Vec::with_capacity(count);
    for i in 0..count {
        let s = Point::new(args.source[i].x / width, args.source[i].y / height);
        let d = Point::new(args.target[i].x / width, args.target[i].y / height);
        let ratio = width / height;
        let dx = ratio * (d.x - s.x);
        let dy = d.y - s.y;
        limit.push(dx * dx + dy * dy);
        src_n.push(s);
        dst_n.push(d);
    }
    RadialMesh {
        src_n,
        dst_n,
        limit,
        factors: args.factors.to_vec(),
        active: args.active.to_vec(),
        sign: if args.degree > 0.0 { 1.0 } else { -1.0 },
        degree_squared: args.degree * args.degree,
        split_degree: args.split_degree,
        split_at: args.split_at,
        ratio: width / height,
        width,
        height,
        outside: args.outside,
        fixed_source: args.fixed_source,
    }
}

impl RadialMesh {
    pub fn warp(&self, point: Point) -> Point {
        let mut x = point.x / self.width;
        let mut y = point.y / self.height;
        let (original_x, original_y) = (x, y);
        for &i in &self.active {
            let limit = self.limit[i];
            if limit <= 0.0 {
                continue;
            }
            let (bx, by) = if self.fixed_source {
                (original_x, original_y)
            } else {
                (x, y)
            };
            let dx = self.ratio * (bx - self.src_n[i].x);
            let dy = by - self.src_n[i].y;
            let distance_squared = dx * dx + dy * dy;
            let phase = if self.outside {
                if distance_squared <= limit {
                    continue;
                }
                (limit / distance_squared).sqrt()
            } else {
                if distance_squared >= limit {
                    continue;
                }
                (distance_squared / limit).sqrt()
            };
            let falloff = 0.5 * (1.0 + (core::f32::consts::PI * phase).cos());
            let (sign, squared) = if i >= self.split_at {
                (
                    if self.split_degree > 0.0 { 1.0 } else { -1.0 },
                    self.split_degree * self.split_degree,
                )
            } else {
                (self.sign, self.degree_squared)
            };
            let scale = sign * squared * falloff * self.factors[i];
            x += scale * (self.dst_n[i].x - self.src_n[i].x);
            y += scale * (self.dst_n[i].y - self.src_n[i].y);
        }
        Point::new(x * self.width, y * self.height)
    }

    fn active_deltas(&self, source: &[Point], target: &[Point]) -> Vec<VertexDelta> {
        self.active
            .iter()
            .map(|&i| VertexDelta {
                index: i,
                source: source[i],
                target: target[i],
            })
            .collect()
    }
}

// ------------------------------------------------- row 1: slim PF8025/PF8026

/// FacialRefineV2 part 14/15: `ControlRange [0,1] -> ValueRange [0,1.5]`,
/// offset `((r-g)*0.25, (b-a)*0.25) * degree`, no vertex delta.
pub fn slim(ui: f32, roi: Texel) -> Result<Geometry, Refusal> {
    let control = unsigned_ui(ui, "slim_ui")?;
    let degree = check_degree(map(control, &[0.0, 1.0], &[0.0, 1.5]), "slim_degree")?;
    let u = (roi.r - roi.g) * 0.25 * degree;
    let v = (roi.b - roi.a) * 0.25 * degree;
    Ok(Geometry {
        priority: 1,
        parts: vec![
            PartGeometry {
                part: 14,
                pf: 8025,
                degree,
                second_degree: 0.0,
                deltas: Vec::new(),
                warp: None,
            },
            PartGeometry {
                part: 15,
                pf: 8026,
                degree,
                second_degree: 0.0,
                deltas: Vec::new(),
                warp: None,
            },
        ],
        offsets: vec![TextureOffset {
            overlay_index: 0,
            pf: 8025,
            u,
            v,
        }],
    })
}

// ----------------------------------------------- row 2: temple PF8029/PF8030

const CURVE_A: [usize; 4] = [230, 168, 116, 117];
const CURVE_B: [usize; 4] = [236, 150, 146, 145];
const TEMPLE_ANCHOR: usize = 55;

/// Type1014 part 23/24: 26 Catmull samples, `target = source + 2*param*
/// (anchor - source)`, fixed-origin cosine falloff, samples 0..12 left and
/// 13..25 right.
pub fn temple(
    points: &[Point],
    ui_left: f32,
    ui_right: f32,
    vendor: &VendorParts<'_>,
    width: u32,
    height: u32,
) -> Result<Geometry, Refusal> {
    if points.len() <= 236 {
        return Err(Refusal::MissingInput("temple prepared 237+ points"));
    }
    if width == 0 || height == 0 {
        return Err(Refusal::ValueOutOfRange("image_dimensions"));
    }
    for point in points {
        if !point.finite() {
            return Err(Refusal::ValueOutOfRange("temple_points"));
        }
    }
    let left_degree = check_degree(ui_left / 100.0, "temple_ui_left")?;
    let right_degree = check_degree(ui_right / 100.0, "temple_ui_right")?;
    if !(-100.0..=100.0).contains(&ui_left) || !(-100.0..=100.0).contains(&ui_right) {
        return Err(Refusal::ValueOutOfRange("temple_ui"));
    }

    let mut source = [Point::new(0.0, 0.0); 26];
    sample_curve(points, &CURVE_A, &mut source[..13]);
    sample_curve(points, &CURVE_B, &mut source[13..]);
    let anchor = points[TEMPLE_ANCHOR];

    let mut parts = Vec::with_capacity(2);
    for (i, degree) in [left_degree, right_degree].into_iter().enumerate() {
        let part = 23 + i as u8;
        let arrays = vendor
            .get(part)
            .ok_or(Refusal::MissingInput("effect.json temple part 23/24"))?;
        if arrays.parameters.len() != 26 || arrays.factors.len() != 26 {
            return Err(Refusal::ShapeMismatch("temple 26 params/factors"));
        }
        let mut target = source;
        let mut active = Vec::with_capacity(26);
        for j in 0..26 {
            target[j] = source[j] + (2.0 * arrays.parameters[j]) * (anchor - source[j]);
            if arrays.parameters[j] != 0.0 && arrays.factors[j] != 0.0 {
                active.push(j);
            }
        }
        let mesh = radial_mesh(WheelMeshArgs {
            source: &source,
            target: &target,
            factors: arrays.factors,
            active: &active,
            degree,
            width,
            height,
            outside: false,
            split_at: 13,
            split_degree: right_degree,
            fixed_source: true,
        });
        parts.push(PartGeometry {
            part,
            pf: 8029 + i as u32,
            degree,
            second_degree: 0.0,
            deltas: mesh.active_deltas(&source, &target),
            warp: Some(Warp::Radial(mesh)),
        });
    }

    Ok(Geometry {
        priority: 2,
        parts,
        offsets: Vec::new(),
    })
}

fn sample_curve(points: &[Point], indices: &[usize; 4], output: &mut [Point]) {
    let c = [
        points[indices[0]],
        points[indices[1]],
        points[indices[2]],
        points[indices[3]],
    ];
    output[0] = c[0];
    let mut at = 1;
    for segment in 0usize..3 {
        let p0 = c[segment.saturating_sub(1)];
        let p1 = c[segment];
        let p2 = c[segment + 1];
        let p3 = c[(segment + 2).min(3)];
        let m1 = (p2 - p0) * 0.5;
        let m2 = (p3 - p1) * 0.5;
        for step in 1..=4 {
            let t = step as f32 / 4.0;
            let t2 = t * t;
            let t3 = t2 * t;
            output[at] = p1 * (2.0 * t3 - 3.0 * t2 + 1.0)
                + m1 * (t3 - 2.0 * t2 + t)
                + p2 * (-2.0 * t3 + 3.0 * t2)
                + m2 * (t3 - t2);
            at += 1;
        }
    }
}

// ------------------------------------------------ row 3: cheek PF8027/PF8028

/// Type1015 part 33/34, 33 prepared points around `center`,
/// `degree = 1 - 2p`, `p = (ui + 100) / 200`.
pub fn cheek(
    points: &[Point],
    center: Point,
    ui: f32,
    vendor: &VendorParts<'_>,
    width: u32,
    height: u32,
) -> Result<Geometry, Refusal> {
    if points.len() < 33 {
        return Err(Refusal::MissingInput("cheek 33 points"));
    }
    if !center.finite() {
        return Err(Refusal::ValueOutOfRange("cheek_center"));
    }
    if width == 0 || height == 0 {
        return Err(Refusal::ValueOutOfRange("image_dimensions"));
    }
    let parameter = signed_ui(ui, "cheek_ui")?;
    let arrays = vendor
        .get(33)
        .ok_or(Refusal::MissingInput("effect.json cheek part 33"))?;
    vendor
        .get(34)
        .ok_or(Refusal::MissingInput("effect.json cheek part 34"))?;
    let degree = check_degree(map(parameter, &KNOTS, &[1.0, 0.0, -1.0]), "cheek_degree")?;
    build_type1015_pair(
        points, center, arrays, 33, 34, 8027, 8028, degree, degree, width, height,
    )
}

// ------------------------------------------------- row 4: jaw PF8023/PF8024

/// Type1015 parts 29..32: `ui <= 0` selects 31/32 (`[1,0,0]`),
/// `ui > 0` selects 29/30 (`[0,0,1]`).
pub fn jaw(
    ui_left: f32,
    ui_right: f32,
    points: &[Point],
    center: Point,
    vendor: &VendorParts<'_>,
    width: u32,
    height: u32,
) -> Result<Geometry, Refusal> {
    if points.len() < 33 {
        return Err(Refusal::MissingInput("jaw 33 points"));
    }
    if !center.finite() {
        return Err(Refusal::ValueOutOfRange("jaw_center"));
    }
    if width == 0 || height == 0 {
        return Err(Refusal::ValueOutOfRange("image_dimensions"));
    }
    let left = signed_ui(ui_left, "jaw_ui_left")?;
    let right = signed_ui(ui_right, "jaw_ui_right")?;
    let left_index = if ui_left <= 0.0 { 2usize } else { 0 };
    let right_index = if ui_right <= 0.0 { 3usize } else { 1 };
    let left_degree = check_degree(
        map(left, &KNOTS, if left_index == 2 { &NEGATIVE } else { &POSITIVE }),
        "jaw_degree_left",
    )?;
    let right_degree = check_degree(
        map(right, &KNOTS, if right_index == 3 { &NEGATIVE } else { &POSITIVE }),
        "jaw_degree_right",
    )?;
    let left_arrays = vendor.get(29 + left_index as u8).ok_or(Refusal::MissingInput(
        "effect.json jaw parts 29..32",
    ))?;
    let right_arrays = vendor.get(29 + right_index as u8).ok_or(Refusal::MissingInput(
        "effect.json jaw parts 29..32",
    ))?;
    let left_part = build_type1015(
        points,
        center,
        left_arrays,
        29 + left_index as u8,
        8023,
        left_degree,
        width,
        height,
    )?;
    let right_part = build_type1015(
        points,
        center,
        right_arrays,
        29 + right_index as u8,
        8024,
        right_degree,
        width,
        height,
    )?;
    Ok(Geometry {
        priority: 4,
        parts: vec![left_part, right_part],
        offsets: Vec::new(),
    })
}

// ------------------------------------------------- row 5: V face PF247/PF248

/// FacialRefineV2 rows 0..3: positive `[0,0,1]`, negative `[1,0,0]`,
/// split `[degree, degree]`, offset `decode(roi) * degree`.
pub fn v_face(ui_left: f32, ui_right: f32, roi: Texel) -> Result<Geometry, Refusal> {
    let left = signed_ui(ui_left, "vface_ui_left")?;
    let right = signed_ui(ui_right, "vface_ui_right")?;
    let mut parts = Vec::with_capacity(4);
    let mut offsets = Vec::with_capacity(4);
    for i in 0..4usize {
        let is_left = i == 0 || i == 2;
        let negative = i >= 2;
        let pf = if is_left { 247 } else { 248 };
        let p = if is_left { left } else { right };
        let degree = check_degree(
            map(p, &KNOTS, if negative { &NEGATIVE } else { &POSITIVE }),
            "vface_degree",
        )?;
        parts.push(PartGeometry {
            part: i as u8,
            pf,
            degree,
            second_degree: degree,
            deltas: Vec::new(),
            warp: None,
        });
        if degree != 0.0 {
            offsets.push(TextureOffset {
                overlay_index: i as u8,
                pf,
                u: (roi.r - roi.g) * 0.25 * degree,
                v: (roi.b - roi.a) * 0.25 * degree,
            });
        }
    }
    Ok(Geometry {
        priority: 5,
        parts,
        offsets,
    })
}

// ---------------------------------------------- row 6: face width PF30092/93

/// Type1016 parts 25..28: `A27` from the 414-point array, mirrored `Q`,
/// `dest = A + 2*param*(Q - A)`, warps points outside the radius.
pub fn face_width(
    ui: f32,
    points414: &[Point],
    vendor: &VendorParts<'_>,
    width: u32,
    height: u32,
) -> Result<Geometry, Refusal> {
    if points414.len() < 242 {
        return Err(Refusal::MissingInput("face width 414 points"));
    }
    if width == 0 || height == 0 {
        return Err(Refusal::ValueOutOfRange("image_dimensions"));
    }
    let p = signed_ui(ui, "face_width_ui")?;
    let prepared = type1016_prepared_a(points414);
    let axis = type1016_axis(points414);
    let mirrored = type1016_mirrored(&prepared, &axis);

    let mut parts = Vec::with_capacity(4);
    for i in 0..4usize {
        let arrays = vendor
            .get(25 + i as u8)
            .ok_or(Refusal::MissingInput("effect.json face width parts 25..28"))?;
        if arrays.parameters.len() != 27 || arrays.factors.len() != 27 {
            return Err(Refusal::ShapeMismatch("face width 27 params/factors"));
        }
        let degree = check_degree(
            map(p, &KNOTS, if i < 2 { &NEGATIVE } else { &POSITIVE }),
            "face_width_degree",
        )?;
        let mut target = [Point::new(0.0, 0.0); 27];
        let mut active = Vec::with_capacity(27);
        for j in 0..27 {
            let parameter = arrays.parameters[j];
            let factor = arrays.factors[j];
            target[j] = prepared[j];
            if parameter == 0.0 || factor == 0.0 || parameter.abs() <= 0.01 {
                continue;
            }
            active.push(j);
            target[j] = prepared[j] + (2.0 * parameter) * (mirrored[j] - prepared[j]);
        }
        let mesh = radial_mesh(WheelMeshArgs {
            source: &prepared,
            target: &target,
            factors: arrays.factors,
            active: &active,
            degree,
            width,
            height,
            outside: true,
            split_at: usize::MAX,
            split_degree: 0.0,
            fixed_source: false,
        });
        parts.push(PartGeometry {
            part: 25 + i as u8,
            pf: if i % 2 == 0 { 30092 } else { 30093 },
            degree,
            second_degree: 0.0,
            deltas: mesh.active_deltas(&prepared, &target),
            warp: Some(Warp::Radial(mesh)),
        });
    }
    Ok(Geometry {
        priority: 6,
        parts,
        offsets: Vec::new(),
    })
}

// ------------------------------------------------ row 7: sharp chin PF242/243

/// Type1015 parts 17..20: `i < 2` uses `[0,0,0.85]`, else `[1,0,0]`.
pub fn sharp_chin(
    ui: f32,
    points: &[Point],
    center: Point,
    vendor: &VendorParts<'_>,
    width: u32,
    height: u32,
) -> Result<Geometry, Refusal> {
    if points.len() < 33 {
        return Err(Refusal::MissingInput("chin 33 points"));
    }
    if !center.finite() {
        return Err(Refusal::ValueOutOfRange("chin_center"));
    }
    if width == 0 || height == 0 {
        return Err(Refusal::ValueOutOfRange("image_dimensions"));
    }
    let p = signed_ui(ui, "chin_ui")?;
    let mut parts = Vec::with_capacity(4);
    for i in 0..4usize {
        let arrays = vendor
            .get(17 + i as u8)
            .ok_or(Refusal::MissingInput("effect.json chin parts 17..20"))?;
        let degree = check_degree(
            map(p, &KNOTS, if i < 2 { &[0.0, 0.0, 0.85] } else { &NEGATIVE }),
            "chin_degree",
        )?;
        parts.push(build_type1015(
            points,
            center,
            arrays,
            17 + i as u8,
            if i % 2 == 0 { 242 } else { 243 },
            degree,
            width,
            height,
        )?);
    }
    Ok(Geometry {
        priority: 7,
        parts,
        offsets: Vec::new(),
    })
}

// -------------------------------------- rows 8/9: forehead PF30080..PF30083

const TYPE1012_CONSUMED: [usize; 27] = [
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25,
    24, 26,
];

/// `0x7078620` `pshufd 0x4E`: swap the two float2 points in each 16-byte
/// block, slot 26 stays last.
pub fn reorder_type1012_slots(slots: &[Point; 27]) -> [Point; 27] {
    let mut swapped = [Point::new(0.0, 0.0); 27];
    for i in 0..27 {
        swapped[i] = slots[TYPE1012_CONSUMED[i]];
    }
    swapped
}

/// Type1012 alternate consumer, parts 35/36 (width) or 37/38 (height):
/// 27x5 grid, 135 weights, doubled degree slots, `term = weight * distance *
/// 0.1`, edge clamp 0.05.
pub fn forehead(
    priority: u8,
    ui_left: f32,
    ui_right: f32,
    slots27: &[Point; 27],
    anchor: Point,
    reference: Point,
    weights_negative: &[f32],
    weights_positive: &[f32],
    width: u32,
    height: u32,
) -> Result<Geometry, Refusal> {
    let (negative_part, positive_part) = match priority {
        8 => (35u8, 36u8),
        9 => (37u8, 38u8),
        other => return Err(Refusal::NotImplemented(u32::from(other))),
    };
    if !anchor.finite() || !reference.finite() {
        return Err(Refusal::ValueOutOfRange("type1012_anchor"));
    }
    if width == 0 || height == 0 {
        return Err(Refusal::ValueOutOfRange("image_dimensions"));
    }
    if weights_negative.len() != 135 || weights_positive.len() != 135 {
        return Err(Refusal::ShapeMismatch("type1012 135 weights"));
    }
    let p_left = signed_ui(ui_left, "type1012_ui_left")?;
    let p_right = signed_ui(ui_right, "type1012_ui_right")?;
    let negative = build_type1012(
        negative_part,
        if priority == 8 { 30080 } else { 30082 },
        weights_negative,
        &reorder_type1012_slots(slots27),
        anchor,
        reference,
        map(p_left, &KNOTS, &[0.0, 0.0, -1.0]),
        map(p_right, &KNOTS, &[0.0, 0.0, -1.0]),
        width,
        height,
    )?;
    let positive = build_type1012(
        positive_part,
        if priority == 8 { 30081 } else { 30083 },
        weights_positive,
        &reorder_type1012_slots(slots27),
        anchor,
        reference,
        map(p_left, &KNOTS, &[1.0, 0.0, 0.0]),
        map(p_right, &KNOTS, &[1.0, 0.0, 0.0]),
        width,
        height,
    )?;
    Ok(Geometry {
        priority,
        parts: vec![negative, positive],
        offsets: Vec::new(),
    })
}

fn y_weights() -> [f32; 5] {
    [
        0.0,
        f32::from_bits(0x3E4C_CCCD),
        f32::from_bits(0x3F19_9999),
        f32::from_bits(0xBE4C_CCE1),
        f32::from_bits(0xBECC_CCC9),
    ]
}

#[allow(clippy::too_many_arguments)]
fn build_type1012(
    part: u8,
    pf: u32,
    weights: &[f32],
    slots: &[Point; 27],
    anchor: Point,
    reference: Point,
    degree0: f32,
    degree1: f32,
    width: u32,
    height: u32,
) -> Result<PartGeometry, Refusal> {
    let width_f = width as f32;
    let height_f = height as f32;
    let mut a = [Point::new(0.0, 0.0); 136];
    for x in 0..27 {
        let slot = slots[x];
        for y in 0..5 {
            let weight = y_weights()[y] * 1.2;
            let p = slot + (anchor - slot) * weight;
            a[y * 27 + x] = Point::new(p.x.clamp(0.0, width_f), p.y.clamp(0.0, height_f));
        }
    }
    a[135] = anchor;

    let d0 = degree0 + degree0;
    let d1 = degree1 + degree1;
    let mut b = a;

    let ax = anchor.x - reference.x;
    let ay = anchor.y - reference.y;
    let mut norm = (ax * ax + ay * ay).sqrt();
    if norm == 0.0 {
        norm = 1.0;
    }
    let (default_x, default_y) = (ax / norm, ay / norm);

    for i in 0..135 {
        let weight = weights[i];
        if weight == 0.0 {
            continue;
        }
        let px = b[i].x;
        let py = b[i].y;

        let mut edge = if px <= 20.0 || px >= width_f - 20.0 {
            (width_f - px).abs().min(px.abs()) * 0.05
        } else {
            1.0
        };
        if py <= 20.0 || py >= height_f - 20.0 {
            edge *= py.abs() * 0.05;
        }

        let (dir_x, dir_y) = if i != 13 {
            let dx = anchor.x - px;
            let dy = anchor.y - py;
            let length = (dx * dx + dy * dy).sqrt();
            (dx / length, dy / length)
        } else {
            (default_x, default_y)
        };

        let distance = ((anchor.x - px) * (anchor.x - px) + (anchor.y - py) * (anchor.y - py))
            .sqrt();
        let mut term = f64::from(weight * distance) * 0.1;
        let row = i % 27;
        term *= if row <= 12 {
            f64::from(d0)
        } else if row == 13 {
            f64::from(d0 + d1) * 0.5
        } else {
            f64::from(d1)
        };
        let scalar = term as f32;
        b[i] = Point::new(px + edge * scalar * dir_x, py + edge * scalar * dir_y);
    }

    Ok(PartGeometry {
        part,
        pf,
        degree: degree0,
        second_degree: degree1,
        deltas: (0..136)
            .map(|i| VertexDelta {
                index: i,
                source: a[i],
                target: b[i],
            })
            .collect(),
        warp: None,
    })
}

// ----------------------------------------- row 10: overall size PF8050/PF8051

/// Type1002 part 10: two 72-point patches, `scale = 0.125 * degree`, only
/// slots 1..15 move.
pub fn overall_size(
    ui_left: f32,
    ui_right: f32,
    eye_landmarks6: &[Point],
    left_patch72: &[Point],
    right_patch72: &[Point],
) -> Result<Geometry, Refusal> {
    if eye_landmarks6.len() < 6
        || left_patch72.len() < 72
        || right_patch72.len() < 72
    {
        return Err(Refusal::MissingInput(
            "overall size 6 landmarks + two 72 patches",
        ));
    }
    for point in eye_landmarks6 {
        if !point.finite() {
            return Err(Refusal::ValueOutOfRange("eye_landmarks"));
        }
    }
    let left = signed_ui(ui_left, "overall_ui_left")?;
    let right = signed_ui(ui_right, "overall_ui_right")?;
    let degree_left = check_degree(
        map(left, &KNOTS, &[-1.0, 0.0, 1.3]),
        "overall_degree_left",
    )?;
    let degree_right = check_degree(
        map(right, &KNOTS, &[-1.0, 0.0, 1.3]),
        "overall_degree_right",
    )?;
    let left_center = (eye_landmarks6[0] + eye_landmarks6[1] + eye_landmarks6[2]) * (1.0 / 3.0);
    let right_center =
        (eye_landmarks6[3] + eye_landmarks6[4] + eye_landmarks6[5]) * (1.0 / 3.0);

    let left_scale = 0.125 * degree_left;
    let right_scale = 0.125 * degree_right;
    Ok(Geometry {
        priority: 10,
        parts: vec![
            PartGeometry {
                part: 10,
                pf: 8050,
                degree: degree_left,
                second_degree: 0.0,
                deltas: patch_deltas(left_patch72, left_center, left_scale),
                warp: Some(Warp::ScaleAbout {
                    center: left_center,
                    scale: left_scale,
                }),
            },
            PartGeometry {
                part: 10,
                pf: 8051,
                degree: degree_right,
                second_degree: 0.0,
                deltas: patch_deltas(right_patch72, right_center, right_scale),
                warp: Some(Warp::ScaleAbout {
                    center: right_center,
                    scale: right_scale,
                }),
            },
        ],
        offsets: Vec::new(),
    })
}

fn patch_deltas(patch: &[Point], center: Point, scale: f32) -> Vec<VertexDelta> {
    (1..=15)
        .map(|i| VertexDelta {
            index: i,
            source: patch[i],
            target: patch[i] + (patch[i] - center) * scale,
        })
        .collect()
}

// ------------------------------------------------- shared Type1015 builder

fn build_type1015_pair(
    points: &[Point],
    center: Point,
    arrays: &PartArrays<'_>,
    first_part: u8,
    second_part: u8,
    first_pf: u32,
    second_pf: u32,
    first_degree: f32,
    second_degree: f32,
    width: u32,
    height: u32,
) -> Result<Geometry, Refusal> {
    Ok(Geometry {
        priority: 3,
        parts: vec![
            build_type1015(
                points, center, arrays, first_part, first_pf, first_degree, width, height,
            )?,
            build_type1015(
                points, center, arrays, second_part, second_pf, second_degree, width, height,
            )?,
        ],
        offsets: Vec::new(),
    })
}

#[allow(clippy::too_many_arguments)]
fn build_type1015(
    points: &[Point],
    center: Point,
    arrays: &PartArrays<'_>,
    part: u8,
    pf: u32,
    degree: f32,
    width: u32,
    height: u32,
) -> Result<PartGeometry, Refusal> {
    if arrays.parameters.len() != 33 || arrays.factors.len() != 33 {
        return Err(Refusal::ShapeMismatch("type1015 33 params/factors"));
    }
    let mut source = [Point::new(0.0, 0.0); 33];
    let mut target = [Point::new(0.0, 0.0); 33];
    let mut active = Vec::with_capacity(33);
    for i in 0..33 {
        let p = points[i];
        source[i] = p;
        target[i] = p;
        let parameter = arrays.parameters[i];
        let factor = arrays.factors[i];
        if parameter == 0.0 || factor == 0.0 || parameter.abs() <= 0.01 {
            continue;
        }
        active.push(i);
        target[i] = p + (2.0 * parameter) * (p - center);
    }
    let mesh = radial_mesh(WheelMeshArgs {
        source: &source,
        target: &target,
        factors: arrays.factors,
        active: &active,
        degree,
        width,
        height,
        outside: false,
        split_at: usize::MAX,
        split_degree: 0.0,
        fixed_source: false,
    });
    Ok(PartGeometry {
        part,
        pf,
        degree,
        second_degree: 0.0,
        deltas: mesh.active_deltas(&source, &target),
        warp: Some(Warp::Radial(mesh)),
    })
}

// ------------------------------------------------- shared Type1016 builder

struct AxisContext {
    axis_start: Point,
    axis_end: Point,
    x748: f32,
    x788: f32,
    x1b0: f32,
    x1b8: f32,
    x1d0: f32,
    center_1d8: Point,
    x418: f32,
}

fn type1016_prepared_a(points: &[Point]) -> [Point; 27] {
    let mut prepared = [Point::new(0.0, 0.0); 27];
    prepared[0] = points[165];
    prepared[1] = points[168];
    prepared[2] = points[170];

    let mut controls = [points[115], points[120], points[125]];
    let mut sampled = sample_curve_8(&controls);
    for i in 1..=5 {
        prepared[2 + i] = sampled[i];
    }
    controls = [points[120], points[125], points[131]];
    sampled = sample_curve_8(&controls);
    for i in 2..=7 {
        prepared[6 + i] = sampled[i];
    }
    controls = [points[131], points[137], points[142]];
    sampled = sample_curve_8(&controls);
    for i in 1..=6 {
        prepared[13 + i] = sampled[i];
    }
    controls = [points[142], points[147], points[148]];
    sampled = sample_curve_8(&controls);
    for i in 3..=6 {
        prepared[17 + i] = sampled[i];
    }

    prepared[24] = points[148];
    prepared[25] = points[150];
    prepared[26] = points[153];
    prepared
}

fn type1016_axis(points: &[Point]) -> AxisContext {
    AxisContext {
        axis_start: points[28],
        axis_end: points[45],
        x748: points[233].x,
        x788: points[241].x,
        x1b0: points[54].x,
        x1b8: points[55].x,
        x1d0: points[58].x,
        center_1d8: points[59],
        x418: points[131].x,
    }
}

fn type1016_mirrored(prepared: &[Point; 27], context: &AxisContext) -> [Point; 27] {
    let mut template = [Point::new(0.0, 0.0); 14];
    template[0] = Point::new(context.x748, (prepared[0].y + prepared[26].y) * 0.5);
    template[1] = Point::new(context.x788, (prepared[1].y + prepared[25].y) * 0.5);
    template[2] = Point::new(context.x1b0, (prepared[2].y + prepared[24].y) * 0.5);
    let middle_x = (context.x1b8 + context.x1d0) * 0.5;
    for i in 3..=7 {
        template[i] = Point::new(middle_x, (prepared[i].y + prepared[26 - i].y) * 0.5);
    }
    let lower_x = (context.center_1d8.x + context.x418) * 0.5;
    for i in 8..=12 {
        template[i] = Point::new(lower_x, (prepared[i].y + prepared[26 - i].y) * 0.5);
    }
    template[13] = context.center_1d8;

    let ax = context.axis_end.x - context.axis_start.x;
    let ay = context.axis_end.y - context.axis_start.y;
    let length = (ax * ax + ay * ay).sqrt();
    let reciprocal = if length == 0.0 { 1.0 } else { 1.0 / length };

    let mut targets = [Point::new(0.0, 0.0); 27];
    for i in 0..13 {
        targets[i] = offset_point(prepared[i], template[i], ax, ay, reciprocal, 1.0);
    }
    targets[13] = context.center_1d8;
    for i in 14..27 {
        targets[i] = offset_point(
            prepared[i],
            template[26 - i],
            ax,
            ay,
            reciprocal,
            -1.0,
        );
    }
    targets
}

fn offset_point(
    point: Point,
    template: Point,
    axis_x: f32,
    axis_y: f32,
    reciprocal: f32,
    direction: f32,
) -> Point {
    let dx = point.x - template.x;
    let dy = point.y - template.y;
    let scale = (dx * dx + dy * dy).sqrt() * reciprocal * direction;
    Point::new(point.x + axis_x * scale, point.y + axis_y * scale)
}

fn sample_curve_8(controls: &[Point; 3]) -> [Point; 8] {
    let dx = controls[2].x - controls[0].x;
    let dy = controls[2].y - controls[0].y;
    let mut length = (dx * dx + dy * dy).sqrt();
    if length < 0.001 {
        length = 0.001;
    }
    let cosine = dx / length;
    let sine = -dy / length;

    let mut local_x = [0.0f32; 3];
    let mut local_y = [0.0f32; 3];
    for i in 0..3 {
        let rx = controls[i].x - controls[0].x;
        let ry = controls[i].y - controls[0].y;
        local_x[i] = cosine * rx - sine * ry;
        local_y[i] = sine * rx + cosine * ry;
    }

    let step = (local_x[2] - local_x[0]) / 7.0;
    let mut output = [Point::new(0.0, 0.0); 8];
    for (sample, slot) in output.iter_mut().enumerate() {
        let x = local_x[0] + sample as f32 * step;
        let mut y = 0.0;
        for i in 0..3 {
            let mut weight = 1.0;
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let mut denominator = local_x[i] - local_x[j];
                if denominator.abs() < 0.0001 {
                    denominator = 0.0001;
                }
                weight *= (x - local_x[j]) / denominator;
            }
            y += weight * local_y[i];
        }
        *slot = Point::new(
            controls[0].x + cosine * x + sine * y,
            controls[0].y - sine * x + cosine * y,
        );
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f32, y: f32) -> Point {
        Point::new(x, y)
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-5 * (1.0 + b.abs())
    }

    fn uniform_points(count: usize) -> Vec<Point> {
        (0..count).map(|i| p(i as f32, i as f32 * 2.0)).collect()
    }

    #[test]
    fn meta_registers_only_callable_cpu_pf() {
        assert_eq!(META.len(), 20);
        let mut pfs: Vec<u32> = META.iter().map(|row| row.pf).collect();
        pfs.sort_unstable();
        pfs.dedup();
        assert_eq!(pfs.len(), 20);
        assert!(META.iter().all(|row| row.access == Access::Cpu));
        assert!(META.iter().all(|row| !row.inputs.is_empty()));
    }

    #[test]
    fn slim_maps_ui_to_value_range() {
        let roi = Texel {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        };
        assert_eq!(slim(0.0, roi).unwrap().parts[0].degree, 0.0);
        assert!(close(slim(50.0, roi).unwrap().parts[0].degree, 0.75));
        assert!(close(slim(100.0, roi).unwrap().parts[0].degree, 1.5));
        assert_eq!(
            slim(-1.0, roi).unwrap_err(),
            Refusal::ValueOutOfRange("slim_ui")
        );
    }

    #[test]
    fn slim_offset_is_decode_times_degree() {
        let roi = Texel {
            r: 1.0,
            g: 0.0,
            b: 0.5,
            a: 0.0,
        };
        let out = slim(100.0, roi).unwrap();
        assert!(close(out.offsets[0].u, 0.375));
        assert!(close(out.offsets[0].v, 0.1875));
        assert!(out.parts.iter().all(|part| part.deltas.is_empty()));
    }

    #[test]
    fn texel_decodes_from_image_pixel() {
        let atlas = Image::new(1, 1, vec![255, 0, 128, 0]).unwrap();
        let roi = Texel::from_image(&atlas, 0, 0);
        assert_eq!(roi.r, 1.0);
        assert!(close(roi.b, 128.0 / 255.0));
    }

    #[test]
    fn temple_needs_237_points() {
        let points = uniform_points(10);
        let vendor = VendorParts { parts: &[] };
        assert_eq!(
            temple(&points, 0.0, 0.0, &vendor, 100, 100).unwrap_err(),
            Refusal::MissingInput("temple prepared 237+ points")
        );
    }

    #[test]
    fn temple_uses_curves_anchor_and_per_part_degrees() {
        let points = uniform_points(300);
        let mut params23 = [0.0f32; 26];
        params23[0] = 1.0;
        let factors = [1.0f32; 26];
        let params24 = [0.0f32; 26];
        let parts = [
            PartArrays {
                part: 23,
                parameters: &params23,
                factors: &factors,
            },
            PartArrays {
                part: 24,
                parameters: &params24,
                factors: &factors,
            },
        ];
        let out = temple(&points, 100.0, -100.0, &VendorParts { parts: &parts }, 100, 100)
            .unwrap();
        assert!(close(out.parts[0].degree, 1.0));
        assert!(close(out.parts[1].degree, -1.0));
        let delta = out.parts[0].deltas[0];
        assert_eq!(delta.index, 0);
        assert_eq!(delta.source, points[230]);
        assert_eq!(delta.target, p(-120.0, -240.0));
        assert!(out.parts[1].deltas.is_empty());
    }

    #[test]
    fn temple_refuses_non_finite_points() {
        let mut points = uniform_points(300);
        points[0].x = f32::NAN;
        let parts = [];
        assert_eq!(
            temple(&points, 0.0, 0.0, &VendorParts { parts: &parts }, 100, 100).unwrap_err(),
            Refusal::ValueOutOfRange("temple_points")
        );
    }

    fn cheek_vendor<'a>(
        params: &'a [f32; 33],
        factors: &'a [f32; 33],
        other: &'a [f32; 33],
    ) -> [PartArrays<'a>; 2] {
        [
            PartArrays {
                part: 33,
                parameters: params,
                factors,
            },
            PartArrays {
                part: 34,
                parameters: other,
                factors,
            },
        ]
    }

    #[test]
    fn cheek_degree_is_one_minus_two_p() {
        let points = uniform_points(33);
        let params = [0.0f32; 33];
        let factors = [1.0f32; 33];
        let other = [0.0f32; 33];
        let parts = cheek_vendor(&params, &factors, &other);
        let vendor = VendorParts { parts: &parts };
        let at = |ui: f32| cheek(&points, p(50.0, 50.0), ui, &vendor, 100, 100).unwrap();
        assert!(close(at(-100.0).parts[0].degree, 1.0));
        assert!(close(at(0.0).parts[0].degree, 0.0));
        assert!(close(at(100.0).parts[0].degree, -1.0));
        assert_eq!(at(0.0).parts[0].part, 33);
        assert_eq!(at(0.0).parts[1].part, 34);
    }

    #[test]
    fn cheek_refuses_out_of_range_and_bad_shape() {
        let points = uniform_points(33);
        let params = [0.0f32; 33];
        let factors = [1.0f32; 33];
        let other = [0.0f32; 33];
        let parts = cheek_vendor(&params, &factors, &other);
        let vendor = VendorParts { parts: &parts };
        assert_eq!(
            cheek(&points, p(50.0, 50.0), 101.0, &vendor, 100, 100).unwrap_err(),
            Refusal::ValueOutOfRange("cheek_ui")
        );
        let short = [0.0f32; 32];
        let bad = [
            PartArrays {
                part: 33,
                parameters: &short,
                factors: &factors,
            },
            PartArrays {
                part: 34,
                parameters: &other,
                factors: &factors,
            },
        ];
        assert_eq!(
            cheek(&points, p(50.0, 50.0), 0.0, &VendorParts { parts: &bad }, 100, 100)
                .unwrap_err(),
            Refusal::ShapeMismatch("type1015 33 params/factors")
        );
    }

    #[test]
    fn jaw_selects_part_by_ui_sign() {
        let points = uniform_points(33);
        let factors = [1.0f32; 33];
        let a29 = [0.0f32; 33];
        let a30 = [0.0f32; 33];
        let a31 = [0.0f32; 33];
        let a32 = [0.0f32; 33];
        let parts = [
            PartArrays {
                part: 29,
                parameters: &a29,
                factors: &factors,
            },
            PartArrays {
                part: 30,
                parameters: &a30,
                factors: &factors,
            },
            PartArrays {
                part: 31,
                parameters: &a31,
                factors: &factors,
            },
            PartArrays {
                part: 32,
                parameters: &a32,
                factors: &factors,
            },
        ];
        let vendor = VendorParts { parts: &parts };
        let positive = jaw(50.0, 50.0, &points, p(0.0, 0.0), &vendor, 100, 100).unwrap();
        assert_eq!(positive.parts[0].part, 29);
        assert_eq!(positive.parts[1].part, 30);
        assert!(positive.parts[0].degree > 0.0);
        let negative = jaw(-50.0, 0.0, &points, p(0.0, 0.0), &vendor, 100, 100).unwrap();
        assert_eq!(negative.parts[0].part, 31);
        assert_eq!(negative.parts[1].part, 32);
        assert!(negative.parts[0].degree > 0.0);
    }

    #[test]
    fn jaw_requires_four_vendor_parts() {
        let points = uniform_points(33);
        let factors = [1.0f32; 33];
        let zeros = [0.0f32; 33];
        let parts = [PartArrays {
            part: 29,
            parameters: &zeros,
            factors: &factors,
        }];
        assert_eq!(
            jaw(10.0, 10.0, &points, p(0.0, 0.0), &VendorParts { parts: &parts }, 100, 100)
                .unwrap_err(),
            Refusal::MissingInput("effect.json jaw parts 29..32")
        );
    }

    #[test]
    fn v_face_selects_positive_and_negative_rows() {
        let roi = Texel {
            r: 1.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        };
        let left = v_face(-100.0, 100.0, roi).unwrap();
        assert!(close(left.parts[0].degree, 0.0));
        assert!(close(left.parts[1].degree, 1.0));
        assert!(close(left.parts[2].degree, 1.0));
        assert!(close(left.parts[3].degree, 0.0));
        assert_eq!(left.offsets.len(), 2);
        assert_eq!(left.offsets[0].overlay_index, 1);
        assert_eq!(left.offsets[1].overlay_index, 2);
        let right = v_face(100.0, 100.0, roi).unwrap();
        assert!(close(right.parts[0].degree, 1.0));
        assert!(close(right.parts[2].degree, 0.0));
    }

    #[test]
    fn v_face_refuses_non_finite_ui() {
        let roi = Texel {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        };
        assert_eq!(
            v_face(f32::INFINITY, 0.0, roi).unwrap_err(),
            Refusal::ValueOutOfRange("vface_ui_left")
        );
    }

    fn width_vendor<'a>(
        params: &'a [[f32; 27]; 4],
        factors: &'a [f32; 27],
    ) -> [PartArrays<'a>; 4] {
        [
            PartArrays {
                part: 25,
                parameters: &params[0],
                factors,
            },
            PartArrays {
                part: 26,
                parameters: &params[1],
                factors,
            },
            PartArrays {
                part: 27,
                parameters: &params[2],
                factors,
            },
            PartArrays {
                part: 28,
                parameters: &params[3],
                factors,
            },
        ]
    }

    #[test]
    fn face_width_uses_negative_then_positive_degrees() {
        let points = uniform_points(242);
        let params = [[0.0f32; 27]; 4];
        let factors = [1.0f32; 27];
        let parts = width_vendor(&params, &factors);
        let out = face_width(100.0, &points, &VendorParts { parts: &parts }, 100, 100).unwrap();
        assert_eq!(out.parts.len(), 4);
        assert_eq!(out.parts[0].pf, 30092);
        assert_eq!(out.parts[1].pf, 30093);
        assert!(close(out.parts[0].degree, 0.0));
        assert!(close(out.parts[2].degree, 1.0));
        assert!(out.parts.iter().all(|part| part.deltas.is_empty()));
    }

    #[test]
    fn face_width_needs_414_source() {
        let points = uniform_points(100);
        let params = [[0.0f32; 27]; 4];
        let factors = [1.0f32; 27];
        let parts = width_vendor(&params, &factors);
        assert_eq!(
            face_width(0.0, &points, &VendorParts { parts: &parts }, 100, 100).unwrap_err(),
            Refusal::MissingInput("face width 414 points")
        );
    }

    #[test]
    fn chin_negative_band_is_085_and_positive_is_one() {
        let points = uniform_points(33);
        let factors = [1.0f32; 33];
        let rows = [
            [0.0f32; 33],
            [0.0f32; 33],
            [0.0f32; 33],
            [0.0f32; 33],
        ];
        let parts = [
            PartArrays {
                part: 17,
                parameters: &rows[0],
                factors: &factors,
            },
            PartArrays {
                part: 18,
                parameters: &rows[1],
                factors: &factors,
            },
            PartArrays {
                part: 19,
                parameters: &rows[2],
                factors: &factors,
            },
            PartArrays {
                part: 20,
                parameters: &rows[3],
                factors: &factors,
            },
        ];
        let out = sharp_chin(100.0, &points, p(0.0, 0.0), &VendorParts { parts: &parts }, 100, 100)
            .unwrap();
        assert!(close(out.parts[0].degree, 0.85));
        assert!(close(out.parts[2].degree, 0.0));
        let neg = sharp_chin(-100.0, &points, p(0.0, 0.0), &VendorParts { parts: &parts }, 100, 100)
            .unwrap();
        assert!(close(neg.parts[0].degree, 0.0));
        assert!(close(neg.parts[2].degree, 1.0));
        assert_eq!(out.parts[0].pf, 242);
        assert_eq!(out.parts[1].pf, 243);
    }

    #[test]
    fn type1012_slot_order_matches_pshufd_0x4e() {
        let mut slots = [Point::new(0.0, 0.0); 27];
        for (i, slot) in slots.iter_mut().enumerate() {
            *slot = p(i as f32, 0.0);
        }
        let swapped = reorder_type1012_slots(&slots);
        assert_eq!(swapped[0].x, 1.0);
        assert_eq!(swapped[1].x, 0.0);
        assert_eq!(swapped[26].x, 26.0);
    }

    fn forehead_slots() -> [Point; 27] {
        let mut slots = [Point::new(0.0, 0.0); 27];
        for (i, slot) in slots.iter_mut().enumerate() {
            *slot = p(50.0 + i as f32, 50.0);
        }
        slots
    }

    fn forehead_weights(index: usize) -> [f32; 135] {
        let mut weights = [0.0f32; 135];
        weights[index] = 1.0;
        weights
    }

    #[test]
    fn forehead_builds_136_deltas_with_doubled_degrees() {
        let slots = forehead_slots();
        let negative = forehead_weights(0);
        let positive = [0.0f32; 135];
        let out = forehead(
            8,
            100.0,
            100.0,
            &slots,
            p(10.0, 10.0),
            p(0.0, 0.0),
            &negative,
            &positive,
            100,
            100,
        )
        .unwrap();
        assert_eq!(out.parts.len(), 2);
        assert_eq!(out.parts[0].pf, 30080);
        assert_eq!(out.parts[1].pf, 30081);
        assert_eq!(out.parts[0].deltas.len(), 136);
        assert!(close(out.parts[0].degree, -1.0));
        assert!(close(out.parts[0].second_degree, -1.0));
        let moved = out.parts[0].deltas[0];
        assert!(moved.target != moved.source);
        assert!(out.parts[1].deltas.iter().all(|d| d.target == d.source));
    }

    #[test]
    fn forehead_priority_and_shape_are_checked() {
        let slots = forehead_slots();
        let weights = [0.0f32; 135];
        assert_eq!(
            forehead(
                7, 0.0, 0.0, &slots, p(10.0, 10.0), p(0.0, 0.0), &weights, &weights, 100, 100
            )
            .unwrap_err(),
            Refusal::NotImplemented(7)
        );
        let short = [0.0f32; 134];
        assert_eq!(
            forehead(
                9, 0.0, 0.0, &slots, p(10.0, 10.0), p(0.0, 0.0), &short, &weights, 100, 100
            )
            .unwrap_err(),
            Refusal::ShapeMismatch("type1012 135 weights")
        );
    }

    #[test]
    fn overall_size_scales_slots_1_to_15() {
        let eyes: Vec<Point> = (0..6).map(|i| p(i as f32, 0.0)).collect();
        let patch: Vec<Point> = (0..72).map(|i| p(i as f32, 10.0)).collect();
        let out = overall_size(100.0, 0.0, &eyes, &patch, &patch).unwrap();
        assert!(close(out.parts[0].degree, 1.3));
        assert!(close(out.parts[1].degree, 0.0));
        assert_eq!(out.parts[0].deltas.len(), 15);
        assert_eq!(out.parts[0].deltas[0].index, 1);
        assert_eq!(out.parts[0].deltas[14].index, 15);
        assert!(out.parts[1].deltas.iter().all(|d| d.target == d.source));
        let center = p(1.0, 0.0);
        let warped = out.parts[0].warp.as_ref().unwrap().apply(p(11.0, 0.0));
        assert!(close(warped.x, 11.0 + (11.0 - center.x) * 0.1625));
    }

    #[test]
    fn overall_size_needs_six_landmarks_and_both_patches() {
        let eyes: Vec<Point> = (0..5).map(|i| p(i as f32, 0.0)).collect();
        let patch: Vec<Point> = (0..72).map(|i| p(i as f32, 10.0)).collect();
        assert_eq!(
            overall_size(0.0, 0.0, &eyes, &patch, &patch).unwrap_err(),
            Refusal::MissingInput("overall size 6 landmarks + two 72 patches")
        );
    }
}
