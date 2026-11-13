//! Body rows 102-112 final geometry/pixel path.
//!
//! Ten rows share recovered vendor math. 103/104/108/110 all execute the
//! `GLReshapeNormalFilter::DrawMeshTriGrid` chain (0x61216E0 / 0x6120010 /
//! 0x6120630) with the type 63 / type 0x40 fragment; 102 uses the same
//! `ReshapeNormal` fragment over explicit per-pixel UVs. 105/111 are the
//! six-point BodyWarp/BodyLiquid cosine-falloff laws, 106 is the PF403
//! Dilate -> Gaussian -> MeanBlur source-mask pipeline, 107 is
//! LiquifyRunWarp, 109 is the BodyRefine per-pixel sampler.
//!
//! 112 (PF407) stays a typed refusal: `0x904CCF0` is `.rdata`
//! `"LengthenLegRefine"`, not code, and no leg mesh/EBO/matrix/consumer RVA
//! exists. Nothing here fabricates a mesh, mask, density texture or default
//! control point. Missing or malformed input is a typed refusal.

use crate::{Access, EffectMeta, Image, Refusal};

/// `ShaderPi` from the recovered 6-point shaders.
const SHADER_PI: f32 = 3.141_592_653_59;

/// Only rows that actually produce pixels are registered. 407 is absent
/// because it has no recovered pixel consumer.
pub const META: &[EffectMeta] = &[
    EffectMeta {
        pf: 400,
        key: "slimArm",
        label: "ArmRefine (row 105)",
        access: Access::Cpu,
        inputs: &[
            "source",
            "source_points",
            "destination_points",
            "degrees",
            "strength",
            "inverse_aspect",
            "valid_point_count",
        ],
    },
    EffectMeta {
        pf: 401,
        key: "slimWaist",
        label: "SlimWaist (row 108)",
        access: Access::Cpu,
        inputs: &["source", "mesh", "passes"],
    },
    EffectMeta {
        pf: 403,
        key: "enhanceBreastMask",
        label: "EnhanceBreast source mask (row 106)",
        access: Access::Cpu,
        inputs: &["source_mask", "radius01"],
    },
    EffectMeta {
        pf: 404,
        key: "niceButtock",
        label: "ButtockRefine (row 111)",
        access: Access::Cpu,
        inputs: &[
            "source",
            "source_points",
            "destination_points",
            "degrees",
            "scale",
            "inverse_aspect",
            "valid_point_count",
        ],
    },
    EffectMeta {
        pf: 406,
        key: "headZoom",
        label: "HeadZoom (row 103)",
        access: Access::Cpu,
        inputs: &["source", "mesh", "passes"],
    },
    EffectMeta {
        pf: 408,
        key: "heightIncrease",
        label: "HeightIncrease (row 102)",
        access: Access::Cpu,
        inputs: &["source", "source_uv", "destination_uv", "defines"],
    },
    EffectMeta {
        pf: 410,
        key: "neckLength",
        label: "NeckLength (row 104)",
        access: Access::Cpu,
        inputs: &["source", "mesh", "passes"],
    },
    EffectMeta {
        pf: 415,
        key: "upBodyRatio",
        label: "UpBodyRatio (row 110)",
        access: Access::Cpu,
        inputs: &["source", "mesh", "passes"],
    },
    EffectMeta {
        pf: 416,
        key: "waistLengthen",
        label: "WaistLengthen (row 109)",
        access: Access::Cpu,
        inputs: &["source", "warped_uv", "mask_r"],
    },
    EffectMeta {
        pf: 38556,
        key: "breastLift",
        label: "BreastLift (row 107)",
        access: Access::Cpu,
        inputs: &[
            "source",
            "degree01",
            "brushes",
            "density",
            "freeze_mask",
            "warp_direction",
            "edge_margin",
        ],
    },
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec2 {
    pub x: f32,
    pub y: f32,
}

impl Vec2 {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    fn finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }
}

/// Type 63 (`ReshapeNormal`) versus type 0x40 (`ReshapeNormalAdjClampEdge`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeMode {
    ReshapeNormal,
    AdjClampEdge,
}

/// The four define keys from `0x611B460`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PassDefines {
    pub clamp_edge: bool,
    pub uv_map: bool,
    pub uv_map_ba: bool,
    pub catmull_rom: bool,
}

impl PassDefines {
    pub const fn plain() -> Self {
        Self {
            clamp_edge: false,
            uv_map: false,
            uv_map_ba: false,
            catmull_rom: false,
        }
    }
}

/// `i_` / `jb` / `ic`. A negative clamp factor forces `smoothcoord() == 1`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Clamp {
    pub factor: f32,
    pub protect_x: u8,
    pub protect_y: u8,
}

impl Clamp {
    pub const fn disabled() -> Self {
        Self {
            factor: -1.0,
            protect_x: 0,
            protect_y: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pass {
    pub edge: EdgeMode,
    pub defines: PassDefines,
    pub clamp: Clamp,
    /// Row-major 4x4, as `m` in the shader.
    pub matrix: [f32; 16],
}

impl Pass {
    /// `P30Matrices.NormalizedToClipNoFlip`: normalized [0,1] to clip.
    pub const fn normalized_to_clip() -> [f32; 16] {
        [
            2.0, 0.0, 0.0, -1.0, 0.0, 2.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
            1.0,
        ]
    }
}

/// `hT` (source UV) and `iV` (destination position) streams plus triangles.
#[derive(Clone, Debug, PartialEq)]
pub struct Mesh {
    pub source_uv: Vec<Vec2>,
    pub destination: Vec<Vec2>,
    pub indices: Vec<u16>,
}

impl Mesh {
    pub fn new(
        source_uv: Vec<Vec2>,
        destination: Vec<Vec2>,
        indices: Vec<u16>,
    ) -> Result<Self, Refusal> {
        if source_uv.len() != destination.len() {
            return Err(Refusal::ShapeMismatch("mesh vertices"));
        }
        if source_uv.len() < 3 {
            return Err(Refusal::ShapeMismatch("mesh vertices"));
        }
        if indices.is_empty() || indices.len() % 3 != 0 {
            return Err(Refusal::ShapeMismatch("mesh indices"));
        }
        if !source_uv.iter().all(|p| p.finite()) || !destination.iter().all(|p| p.finite()) {
            return Err(Refusal::ValueOutOfRange("mesh vertices"));
        }
        let count = source_uv.len();
        if indices.iter().any(|&i| i as usize >= count) {
            return Err(Refusal::ValueOutOfRange("mesh indices"));
        }
        Ok(Self {
            source_uv,
            destination,
            indices,
        })
    }
}

/// Exact `0x4BA3680` two-triangles-per-cell order (PF401's 11x11 grid).
pub fn grid_indices(rows: usize, columns: usize) -> Result<Vec<u16>, Refusal> {
    if rows < 2 || columns < 2 {
        return Err(Refusal::ShapeMismatch("grid"));
    }
    let cells = (rows - 1) * (columns - 1) * 6;
    if cells > u16::MAX as usize {
        return Err(Refusal::ValueOutOfRange("grid"));
    }
    let mut indices = Vec::with_capacity(cells);
    for row in 0..rows - 1 {
        let lower = row * columns;
        let upper = (row + 1) * columns;
        for col in 0..columns - 1 {
            indices.push((upper + col) as u16);
            indices.push((lower + col + 1) as u16);
            indices.push((lower + col) as u16);
            indices.push((upper + col) as u16);
            indices.push((upper + col + 1) as u16);
            indices.push((lower + col + 1) as u16);
        }
    }
    Ok(indices)
}

fn check_image(image: &Image, what: &'static str) -> Result<(), Refusal> {
    image.validate().map_err(|_| Refusal::ShapeMismatch(what))
}

fn out_image(source: &Image, pixels: Vec<u8>) -> Result<Image, Refusal> {
    Image::new(source.width, source.height, pixels).ok_or(Refusal::ShapeMismatch("output"))
}

/// Rows 103/104/108/110 (and the 102 grid path): rasterize `hT`/`iV` through
/// one or more recovered passes, each pass reading the previous output.
pub fn reshape_mesh(source: &Image, mesh: &Mesh, passes: &[Pass]) -> Result<Image, Refusal> {
    check_image(source, "source")?;
    if passes.is_empty() {
        return Err(Refusal::MissingInput("passes"));
    }
    for pass in passes {
        if !pass.matrix.iter().all(|v| v.is_finite()) {
            return Err(Refusal::ValueOutOfRange("matrix"));
        }
        if pass.edge == EdgeMode::AdjClampEdge
            && (pass.clamp.protect_x > 2 || pass.clamp.protect_y > 2)
        {
            return Err(Refusal::ValueOutOfRange("clamp protect"));
        }
        if !pass.clamp.factor.is_finite() {
            return Err(Refusal::ValueOutOfRange("clamp factor"));
        }
    }

    let mut current = source.clone();
    for pass in passes {
        current = rasterize(&current, mesh, pass);
    }
    Ok(current)
}

/// Row 102 (PF408): `ReshapeNormal.frag` over explicit per-pixel source and
/// destination UVs. `TS_CLAMP_EDGE` uses the type-63 `smoothcoord`.
pub fn height_increase(
    source: &Image,
    source_uv: &[Vec2],
    destination_uv: &[Vec2],
    defines: PassDefines,
) -> Result<Image, Refusal> {
    check_image(source, "source")?;
    let pixels = source.pixel_count();
    if source_uv.len() != pixels || destination_uv.len() != pixels {
        return Err(Refusal::ShapeMismatch("uv"));
    }
    if !source_uv.iter().all(|p| p.finite()) || !destination_uv.iter().all(|p| p.finite()) {
        return Err(Refusal::ValueOutOfRange("uv"));
    }

    let pass = Pass {
        edge: EdgeMode::ReshapeNormal,
        defines,
        clamp: Clamp::disabled(),
        matrix: Pass::normalized_to_clip(),
    };
    let width = source.width as usize;
    let height = source.height as usize;
    let mut pixels_out = source.pixels.clone();
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let dst = Vec2::new(
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            );
            let color = fragment(source, source_uv[index], dst, &pass);
            write_pixel(&mut pixels_out, index, color);
        }
    }
    out_image(source, pixels_out)
}

/// Row 105 (PF400 / `0x9678359`): six-point BodyWarp cosine falloff.
#[allow(clippy::too_many_arguments)]
pub fn slim_arm(
    source: &Image,
    source_points: &[Vec2],
    destination_points: &[Vec2],
    degrees: &[f32],
    strength: f32,
    inverse_aspect: f32,
    valid_point_count: usize,
) -> Result<Image, Refusal> {
    check_image(source, "source")?;
    if source_points.len() != 6
        || destination_points.len() != 6
        || degrees.len() != 6
        || valid_point_count > 6
    {
        return Err(Refusal::ShapeMismatch("six point warp"));
    }
    if !source_points.iter().all(|p| p.finite())
        || !destination_points.iter().all(|p| p.finite())
        || !degrees.iter().all(|v| v.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("six point warp"));
    }
    if !strength.is_finite() || !inverse_aspect.is_finite() || inverse_aspect <= 0.0 {
        return Err(Refusal::ValueOutOfRange("strength"));
    }

    let width = source.width as usize;
    let height = source.height as usize;
    let mut pixels = Vec::with_capacity(source.pixels.len());
    for y in 0..height {
        for x in 0..width {
            let uv = Vec2::new(
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            );
            let sample = body_warp_uv(
                uv,
                source_points,
                destination_points,
                degrees,
                strength,
                inverse_aspect,
                valid_point_count,
            );
            pixels.extend_from_slice(&sample_rgba8_even(source, sample.x, sample.y));
        }
    }
    out_image(source, pixels)
}

/// Row 106 (PF403): 3x3 Dilate -> two 9-tap X/Y Gaussian pairs -> 15x15 mean.
pub fn enhance_breast_mask(source: &Image, radius01: f32) -> Result<Image, Refusal> {
    check_image(source, "source_mask")?;
    if !radius01.is_finite() || radius01 < 0.0 {
        return Err(Refusal::ValueOutOfRange("radius01"));
    }
    let width = source.width as usize;
    let height = source.height as usize;
    let params = MaskParams {
        width,
        height,
        dilation_u: radius01 * 130.0,
        dilation_v: radius01 * 130.0,
        gaussian_u: radius01 * 15.0 / width as f32,
        gaussian_v: radius01 * 15.0 / height as f32,
    };

    let mut current: Vec<[u8; 4]> = source
        .pixels
        .chunks_exact(4)
        .map(|p| [p[0], p[1], p[2], p[3]])
        .collect();
    current = dilate(&current, &params);
    for _ in 0..2 {
        current = gaussian(&current, &params, 0.0, params.gaussian_v);
        current = gaussian(&current, &params, params.gaussian_u, 0.0);
    }
    current = mean_blur(&current, &params);

    let mut pixels = Vec::with_capacity(source.pixels.len());
    for p in current {
        pixels.extend_from_slice(&p);
    }
    out_image(source, pixels)
}

#[derive(Clone, Copy, Debug)]
pub struct Brush {
    pub position: Vec2,
    pub direction: Vec2,
    pub radius: f32,
    pub pressure: f32,
}

/// Row 107 (PF38556 / LiquifyRunWarp). `degree01 == 0` is identity.
#[allow(clippy::too_many_arguments)]
pub fn breast_lift(
    source: &Image,
    degree01: f32,
    brushes: &[Brush],
    density: &[f32],
    density_size: (u32, u32),
    freeze_mask: &[u8],
    warp_direction: f32,
    edge_margin: f32,
    push_left: bool,
) -> Result<Image, Refusal> {
    check_image(source, "source")?;
    if !degree01.is_finite() || !(0.0..=1.0).contains(&degree01) {
        return Err(Refusal::ValueOutOfRange("degree01"));
    }
    if degree01 == 0.0 {
        return Ok(source.clone());
    }
    if brushes.is_empty() {
        return Err(Refusal::MissingInput("brushes"));
    }
    let (density_width, density_height) = (density_size.0 as usize, density_size.1 as usize);
    if density_width == 0
        || density_height == 0
        || density.len() != density_width * density_height
        || !density.iter().all(|v| v.is_finite())
    {
        return Err(Refusal::ShapeMismatch("density"));
    }
    if freeze_mask.len() != source.pixel_count() {
        return Err(Refusal::ShapeMismatch("freeze_mask"));
    }
    if !warp_direction.is_finite() || !edge_margin.is_finite() || edge_margin <= 0.0 {
        return Err(Refusal::ValueOutOfRange("edge_margin"));
    }
    for brush in brushes {
        if !brush.position.finite()
            || !brush.direction.finite()
            || !brush.radius.is_finite()
            || brush.radius <= 0.0
            || !brush.pressure.is_finite()
        {
            return Err(Refusal::ValueOutOfRange("brush"));
        }
    }

    let mut current = source.clone();
    for brush in brushes {
        current = warp_brush(
            &current,
            brush,
            warp_direction,
            density,
            density_width,
            density_height,
            freeze_mask,
            edge_margin,
            push_left,
        );
    }
    Ok(current)
}

/// Row 109 (PF416 / BodyRefine.frag): `sample = uv + (warped - uv) * (1 - mask.r)`.
pub fn waist_length(
    source: &Image,
    warped_uv: &[Vec2],
    mask_r: &[f32],
) -> Result<Image, Refusal> {
    check_image(source, "source")?;
    let pixels = source.pixel_count();
    if warped_uv.len() != pixels || mask_r.len() != pixels {
        return Err(Refusal::ShapeMismatch("body refine"));
    }
    if !warped_uv.iter().all(|p| p.finite())
        || !mask_r.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v))
    {
        return Err(Refusal::ValueOutOfRange("body refine"));
    }

    let width = source.width as usize;
    let height = source.height as usize;
    let mut pixels_out = Vec::with_capacity(source.pixels.len());
    for y in 0..height {
        for x in 0..width {
            let index = y * width + x;
            let uv = Vec2::new(
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            );
            let degree = 1.0 - mask_r[index];
            let sample = if degree > 0.01 {
                Vec2::new(
                    uv.x + (warped_uv[index].x - uv.x) * degree,
                    uv.y + (warped_uv[index].y - uv.y) * degree,
                )
            } else {
                uv
            };
            pixels_out.extend_from_slice(&sample_rgba8_half_up(source, sample.x, sample.y));
        }
    }
    out_image(source, pixels_out)
}

/// Row 111 (PF404): BodyLiquid6. `scale` is squared in the shader order.
#[allow(clippy::too_many_arguments)]
pub fn hip_enhancement(
    source: &Image,
    source_points: &[Vec2],
    destination_points: &[Vec2],
    degrees: &[f32],
    scale: f32,
    inverse_aspect: f32,
    valid_point_count: usize,
) -> Result<Image, Refusal> {
    check_image(source, "source")?;
    if source_points.len() != 6
        || destination_points.len() != 6
        || degrees.len() != 6
        || valid_point_count > 6
    {
        return Err(Refusal::ShapeMismatch("six point warp"));
    }
    if !source_points.iter().all(|p| p.finite())
        || !destination_points.iter().all(|p| p.finite())
        || !degrees.iter().all(|v| v.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("six point warp"));
    }
    if !scale.is_finite() || !inverse_aspect.is_finite() || inverse_aspect <= 0.0 {
        return Err(Refusal::ValueOutOfRange("scale"));
    }

    let width = source.width as usize;
    let height = source.height as usize;
    let mut pixels = Vec::with_capacity(source.pixels.len());
    for y in 0..height {
        for x in 0..width {
            let uv = Vec2::new(
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            );
            let sample =
                body_liquid_uv(uv, source_points, destination_points, degrees, scale, inverse_aspect, valid_point_count);
            pixels.extend_from_slice(&sample_rgba8_half_up(source, sample.x, sample.y));
        }
    }
    out_image(source, pixels)
}

/// Row 112 (PF407): no recovered pixel consumer.
pub fn leg_length(_source: &Image) -> Result<Image, Refusal> {
    Err(Refusal::NotImplemented(407))
}

fn rasterize(source: &Image, mesh: &Mesh, pass: &Pass) -> Image {
    let width = source.width as usize;
    let height = source.height as usize;
    let mut output = source.pixels.clone();
    let mut screen = vec![(f32::NAN, f32::NAN, 0.0f32, 0.0f32); mesh.source_uv.len()];

    for (index, position) in mesh.destination.iter().enumerate() {
        let m = &pass.matrix;
        let clip_x = m[0] * position.x + m[1] * position.y + m[3];
        let clip_y = m[4] * position.x + m[5] * position.y + m[7];
        let w = m[12] * position.x + m[13] * position.y + m[15];
        if !w.is_finite() || w.abs() < 1e-8 {
            continue;
        }
        screen[index] = (
            ((clip_x / w) + 1.0) * 0.5 * width as f32,
            ((clip_y / w) + 1.0) * 0.5 * height as f32,
            w,
            1.0 / w,
        );
    }

    for triangle in mesh.indices.chunks_exact(3) {
        let (i0, i1, i2) = (
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        );
        if i0 == i1 || i1 == i2 || i0 == i2 {
            continue;
        }
        let (x0, y0, _, inv_w0) = screen[i0];
        let (x1, y1, _, inv_w1) = screen[i1];
        let (x2, y2, _, inv_w2) = screen[i2];
        if x0.is_nan() || x1.is_nan() || x2.is_nan() {
            continue;
        }
        let area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
        if area.abs() < 1e-8 {
            continue;
        }
        let inv_area = 1.0 / area;
        let min_x = 0.max(x0.min(x1).min(x2).floor() as i64) as usize;
        let max_x = (width as i64 - 1).min(x0.max(x1).max(x2).ceil() as i64).max(0) as usize;
        let min_y = 0.max(y0.min(y1).min(y2).floor() as i64) as usize;
        let max_y = (height as i64 - 1).min(y0.max(y1).max(y2).ceil() as i64).max(0) as usize;
        if min_x > max_x || min_y > max_y {
            continue;
        }

        for py in min_y..=max_y {
            let sample_y = py as f32 + 0.5;
            for px in min_x..=max_x {
                let sample_x = px as f32 + 0.5;
                let mut l0 =
                    ((x1 - sample_x) * (y2 - sample_y) - (y1 - sample_y) * (x2 - sample_x))
                        * inv_area;
                let mut l1 =
                    ((x2 - sample_x) * (y0 - sample_y) - (y2 - sample_y) * (x0 - sample_x))
                        * inv_area;
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

                let denom = l0 * inv_w0 + l1 * inv_w1 + l2 * inv_w2;
                if denom <= 0.0 || !denom.is_finite() {
                    continue;
                }
                let a0 = l0 * inv_w0 / denom;
                let a1 = l1 * inv_w1 / denom;
                let a2 = l2 * inv_w2 / denom;
                let src_uv = Vec2::new(
                    a0 * mesh.source_uv[i0].x
                        + a1 * mesh.source_uv[i1].x
                        + a2 * mesh.source_uv[i2].x,
                    a0 * mesh.source_uv[i0].y
                        + a1 * mesh.source_uv[i1].y
                        + a2 * mesh.source_uv[i2].y,
                );
                let dst_uv = Vec2::new(
                    (px as f32 + 0.5) / width as f32,
                    (py as f32 + 0.5) / height as f32,
                );
                let color = fragment(source, src_uv, dst_uv, pass);
                write_pixel(&mut output, py * width + px, color);
            }
        }
    }
    Image {
        width: source.width,
        height: source.height,
        pixels: output,
    }
}

fn fragment(source: &Image, src_uv: Vec2, dst_uv: Vec2, pass: &Pass) -> [f32; 4] {
    let mut sample_uv = src_uv;
    if pass.defines.clamp_edge {
        let factor = match pass.edge {
            EdgeMode::ReshapeNormal => smooth_coord_reshape(src_uv),
            EdgeMode::AdjClampEdge => smooth_coord_adj(src_uv, pass.clamp),
        };
        sample_uv = Vec2::new(
            dst_uv.x + (src_uv.x - dst_uv.x) * factor,
            dst_uv.y + (src_uv.y - dst_uv.y) * factor,
        );
    }

    if pass.defines.uv_map {
        let sampled = sample_bilinear_float(source, sample_uv.x, sample_uv.y);
        let current = sample_bilinear_float(source, dst_uv.x, dst_uv.y);
        return if pass.defines.uv_map_ba {
            [
                current[0],
                current[1],
                sampled[2] + (sample_uv.x - dst_uv.x),
                sampled[3] + (sample_uv.y - dst_uv.y),
            ]
        } else {
            [
                sampled[0] + (sample_uv.x - dst_uv.x),
                sampled[1] + (sample_uv.y - dst_uv.y),
                current[2],
                current[3],
            ]
        };
    }

    if pass.defines.catmull_rom {
        sample_catmull_rom(source, sample_uv.x, sample_uv.y)
    } else {
        sample_bilinear_float(source, sample_uv.x, sample_uv.y)
    }
}

pub fn smooth_coord_reshape(uv: Vec2) -> f32 {
    let x = 2.49 - (4.98 * (uv.x - 0.5)).abs();
    let y = 2.49 - (4.98 * (uv.y - 0.5)).abs();
    x.min(y).clamp(0.0, 1.0)
}

pub fn smooth_coord_adj(uv: Vec2, clamp: Clamp) -> f32 {
    if clamp.factor < 0.0 {
        return 1.0;
    }
    let mut x = 1.0;
    let mut y = 1.0;
    if clamp.protect_x == 1 {
        x = clamp.factor / 2.0 - (clamp.factor * (0.5 - uv.x).max(0.0)).abs();
    }
    if clamp.protect_x == 2 {
        x = clamp.factor / 2.0 - (clamp.factor * (uv.x - 0.5).max(0.0)).abs();
    }
    if clamp.protect_y == 1 {
        y = clamp.factor / 2.0 - (clamp.factor * (0.5 - uv.y).max(0.0)).abs();
    }
    if clamp.protect_y == 2 {
        y = clamp.factor / 2.0 - (clamp.factor * (uv.y - 0.5).max(0.0)).abs();
    }
    x.min(y).clamp(0.0, 1.0)
}

fn sample_bilinear_float(source: &Image, u: f32, v: f32) -> [f32; 4] {
    let width = source.width as f32;
    let height = source.height as f32;
    let x = (u * width - 0.5).clamp(0.0, width - 1.0);
    let y = (v * height - 0.5).clamp(0.0, height - 1.0);
    let x0 = x.floor();
    let y0 = y.floor();
    let x1 = (x0 + 1.0).min(width - 1.0);
    let y1 = (y0 + 1.0).min(height - 1.0);
    let fx = x - x0;
    let fy = y - y0;

    let mut out = [0.0f32; 4];
    for channel in 0..4 {
        let top = lerp(
            texel(source, x0 as usize, y0 as usize, channel),
            texel(source, x1 as usize, y0 as usize, channel),
            fx,
        );
        let bottom = lerp(
            texel(source, x0 as usize, y1 as usize, channel),
            texel(source, x1 as usize, y1 as usize, channel),
            fx,
        );
        out[channel] = lerp(top, bottom, fy);
    }
    out
}

fn sample_catmull_rom(source: &Image, u: f32, v: f32) -> [f32; 4] {
    let width = source.width as f32;
    let height = source.height as f32;
    let sx = u * width;
    let sy = v * height;
    let px = (sx - 0.5).floor() + 0.5;
    let py = (sy - 0.5).floor() + 0.5;
    let fx = sx - px;
    let fy = sy - py;

    let x0 = weight0(fx);
    let x1 = weight1(fx);
    let x2 = weight2(fx);
    let x3 = weight3(fx);
    let y0 = weight0(fy);
    let y1 = weight1(fy);
    let y2 = weight2(fy);
    let y3 = weight3(fy);
    let w12x = x1 + x2;
    let w12y = y1 + y2;
    let t12x = px + x2 / w12x;
    let t12y = py + y2 / w12y;

    let taps = [
        (px - 1.0, py - 1.0, x0 * y0),
        (t12x, py - 1.0, w12x * y0),
        (px + 2.0, py - 1.0, x3 * y0),
        (px - 1.0, t12y, x0 * w12y),
        (t12x, t12y, w12x * w12y),
        (px + 2.0, t12y, x3 * w12y),
        (px - 1.0, py + 2.0, x0 * y3),
        (t12x, py + 2.0, w12x * y3),
        (px + 2.0, py + 2.0, x3 * y3),
    ];

    let mut color = [0.0f32; 4];
    for (tx, ty, weight) in taps {
        let tap = sample_bilinear_float(source, tx / width, ty / height);
        for channel in 0..4 {
            color[channel] += tap[channel] * weight;
        }
    }
    color
}

fn weight0(f: f32) -> f32 {
    f * (-0.5 + f * (1.0 - 0.5 * f))
}

fn weight1(f: f32) -> f32 {
    1.0 + f * f * (-2.5 + 1.5 * f)
}

fn weight2(f: f32) -> f32 {
    f * (0.5 + f * (2.0 - 1.5 * f))
}

fn weight3(f: f32) -> f32 {
    f * f * (-0.5 + 0.5 * f)
}

fn texel(source: &Image, x: usize, y: usize, channel: usize) -> f32 {
    source.pixels[(y * source.width as usize + x) * 4 + channel] as f32 / 255.0
}

fn write_pixel(pixels: &mut [u8], index: usize, color: [f32; 4]) {
    for channel in 0..4 {
        pixels[index * 4 + channel] = quantize_even(color[channel]);
    }
}

fn quantize_even(value: f32) -> u8 {
    (value * 255.0).clamp(0.0, 255.0).round_ties_even() as u8
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn smooth_coord(uv: Vec2) -> f32 {
    smooth_coord_reshape(uv)
}

#[allow(clippy::too_many_arguments)]
fn body_warp_uv(
    uv: Vec2,
    source_points: &[Vec2],
    destination_points: &[Vec2],
    degrees: &[f32],
    strength: f32,
    inverse_aspect: f32,
    valid_point_count: usize,
) -> Vec2 {
    let mut total_x = 0.0f32;
    let mut total_y = 0.0f32;
    for i in 0..valid_point_count {
        let src = source_points[i];
        let dst = destination_points[i];
        let bx = dst.x - src.x;
        let by = dst.y - src.y;
        let threshold_x = bx * inverse_aspect;
        let threshold = threshold_x * threshold_x + by * by;
        let current_x = (uv.x - src.x) * inverse_aspect;
        let current_y = uv.y - src.y;
        let current = current_x * current_x + current_y * current_y;
        if threshold > current {
            let ratio = (current / threshold).sqrt();
            let factor = ((ratio * SHADER_PI).cos() + 1.0) * 0.5;
            total_x += bx * degrees[i] * factor * strength * strength;
            total_y += by * degrees[i] * factor * strength * strength;
        }
    }
    let smooth = smooth_coord(uv);
    Vec2::new(uv.x - total_x * smooth, uv.y - total_y * smooth)
}

#[allow(clippy::too_many_arguments)]
fn body_liquid_uv(
    uv: Vec2,
    source_points: &[Vec2],
    destination_points: &[Vec2],
    degrees: &[f32],
    scale: f32,
    inverse_aspect: f32,
    valid_point_count: usize,
) -> Vec2 {
    let mut total_x = 0.0f32;
    let mut total_y = 0.0f32;
    for i in 0..valid_point_count {
        let src = source_points[i];
        let dst = destination_points[i];
        let bx = dst.x - src.x;
        let by = dst.y - src.y;
        let threshold_x = bx * inverse_aspect;
        let threshold = threshold_x * threshold_x + by * by;
        let current_x = (uv.x - src.x) * inverse_aspect;
        let current_y = uv.y - src.y;
        let current = current_x * current_x + current_y * current_y;
        if threshold > current {
            let ratio = (current / threshold).sqrt();
            let factor = ((ratio * SHADER_PI).cos() + 1.0) * 0.5;
            total_x += bx * degrees[i] * factor * scale * scale;
            total_y += by * degrees[i] * factor * scale * scale;
        }
    }
    let smooth = smooth_coord(uv);
    Vec2::new(uv.x - total_x * smooth, uv.y - total_y * smooth)
}

fn sample_rgba8_half_up(source: &Image, u: f32, v: f32) -> [u8; 4] {
    sample_rgba8(source, u, v, false)
}

fn sample_rgba8_even(source: &Image, u: f32, v: f32) -> [u8; 4] {
    sample_rgba8(source, u, v, true)
}

fn sample_rgba8(source: &Image, u: f32, v: f32, ties_even: bool) -> [u8; 4] {
    let width = source.width as f32;
    let height = source.height as f32;
    let x = u * width - 0.5;
    let y = v * height - 0.5;
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let x0i = (x0 as i64).clamp(0, source.width as i64 - 1) as usize;
    let x1i = (x0 as i64 + 1).clamp(0, source.width as i64 - 1) as usize;
    let y0i = (y0 as i64).clamp(0, source.height as i64 - 1) as usize;
    let y1i = (y0 as i64 + 1).clamp(0, source.height as i64 - 1) as usize;

    let mut out = [0u8; 4];
    for channel in 0..4 {
        let v00 = source.pixels[(y0i * source.width as usize + x0i) * 4 + channel] as f32;
        let v10 = source.pixels[(y0i * source.width as usize + x1i) * 4 + channel] as f32;
        let v01 = source.pixels[(y1i * source.width as usize + x0i) * 4 + channel] as f32;
        let v11 = source.pixels[(y1i * source.width as usize + x1i) * 4 + channel] as f32;
        let top = v00 + (v10 - v00) * fx;
        let bottom = v01 + (v11 - v01) * fx;
        out[channel] = if ties_even {
            lerp(top, bottom, fy).clamp(0.0, 255.0).round_ties_even() as u8
        } else {
            (lerp(top, bottom, fy) + 0.5).clamp(0.0, 255.0) as u8
        };
    }
    out
}

#[derive(Clone, Copy, Debug)]
struct MaskParams {
    width: usize,
    height: usize,
    dilation_u: f32,
    dilation_v: f32,
    gaussian_u: f32,
    gaussian_v: f32,
}

const GAUSSIAN_WEIGHTS: [f32; 9] = [
    0.05, 0.09, 0.12, 0.15, 0.18, 0.15, 0.12, 0.09, 0.05,
];

fn dilate(source: &[[u8; 4]], p: &MaskParams) -> Vec<[u8; 4]> {
    let mut target = vec![[0u8; 4]; source.len()];
    for y in 0..p.height {
        for x in 0..p.width {
            let u = (x as f32 + 0.5) / p.width as f32;
            let v = (y as f32 + 0.5) / p.height as f32;
            let center = sample_mask(source, p, u, v);
            let mut best = [0u8, 0, 0, 0];
            for oy in -1i32..=1 {
                for ox in -1i32..=1 {
                    let q = sample_mask(
                        source,
                        p,
                        u + ox as f32 * p.dilation_u,
                        v + oy as f32 * p.dilation_v,
                    );
                    for c in 0..3 {
                        best[c] = best[c].max(q[c]);
                    }
                }
            }
            best[3] = center[3];
            target[y * p.width + x] = best;
        }
    }
    target
}

fn gaussian(source: &[[u8; 4]], p: &MaskParams, step_u: f32, step_v: f32) -> Vec<[u8; 4]> {
    let mut target = vec![[0u8; 4]; source.len()];
    for y in 0..p.height {
        for x in 0..p.width {
            let u = (x as f32 + 0.5) / p.width as f32;
            let v = (y as f32 + 0.5) / p.height as f32;
            let mut sum = [0.0f32; 4];
            for tap in -4i32..=4 {
                let weight = GAUSSIAN_WEIGHTS[(tap + 4) as usize];
                let q = sample_mask(
                    source,
                    p,
                    u + tap as f32 * step_u,
                    v + tap as f32 * step_v,
                );
                for c in 0..4 {
                    sum[c] += q[c] as f32 / 255.0 * weight;
                }
            }
            target[y * p.width + x] = sum.map(quantize_even);
        }
    }
    target
}

fn mean_blur(source: &[[u8; 4]], p: &MaskParams) -> Vec<[u8; 4]> {
    const KERNEL: i32 = 15;
    let mut target = vec![[0u8; 4]; source.len()];
    let step_u = 1.0 / p.width as f32;
    let step_v = 1.0 / p.height as f32;
    let start_u = -(KERNEL - 1) as f32 * 0.5 * step_u;
    let start_v = -(KERNEL - 1) as f32 * 0.5 * step_v;
    let normalizer = 1.0 / (KERNEL * KERNEL) as f32;
    for y in 0..p.height {
        for x in 0..p.width {
            let u = (x as f32 + 0.5) / p.width as f32;
            let v = (y as f32 + 0.5) / p.height as f32;
            let mut sum = [0.0f32; 4];
            for ky in 0..KERNEL {
                for kx in 0..KERNEL {
                    let q = sample_mask(
                        source,
                        p,
                        u + start_u + kx as f32 * step_u,
                        v + start_v + ky as f32 * step_v,
                    );
                    for c in 0..4 {
                        sum[c] += q[c] as f32 / 255.0;
                    }
                }
            }
            target[y * p.width + x] = sum.map(|value| quantize_even(value * normalizer));
        }
    }
    target
}

fn sample_mask(source: &[[u8; 4]], p: &MaskParams, u: f32, v: f32) -> [u8; 4] {
    let x = u * p.width as f32 - 0.5;
    let y = v * p.height as f32 - 0.5;
    let x0 = x.floor();
    let y0 = y.floor();
    let tx = x - x0;
    let ty = y - y0;
    let x0i = (x0 as i64).clamp(0, p.width as i64 - 1) as usize;
    let x1i = (x0 as i64 + 1).clamp(0, p.width as i64 - 1) as usize;
    let y0i = (y0 as i64).clamp(0, p.height as i64 - 1) as usize;
    let y1i = (y0 as i64 + 1).clamp(0, p.height as i64 - 1) as usize;
    let a = source[y0i * p.width + x0i];
    let b = source[y0i * p.width + x1i];
    let c = source[y1i * p.width + x0i];
    let d = source[y1i * p.width + x1i];
    let mut out = [0u8; 4];
    for channel in 0..4 {
        let top = lerp_quantized(a[channel], b[channel], tx);
        let bottom = lerp_quantized(c[channel], d[channel], tx);
        out[channel] = lerp_quantized(top, bottom, ty);
    }
    out
}

fn lerp_quantized(a: u8, b: u8, t: f32) -> u8 {
    let value = a as f32 / 255.0 + (b as f32 - a as f32) / 255.0 * t;
    quantize_even(value)
}

#[allow(clippy::too_many_arguments)]
fn warp_brush(
    source: &Image,
    brush: &Brush,
    warp_direction: f32,
    density: &[f32],
    density_width: usize,
    density_height: usize,
    freeze_mask: &[u8],
    edge_margin: f32,
    push_left: bool,
) -> Image {
    let width = source.width as usize;
    let height = source.height as usize;
    let mut output = vec![0u8; source.pixels.len()];
    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            let (sample_u, sample_v) = brush_sample_uv(
                u,
                v,
                width as f32,
                height as f32,
                brush,
                warp_direction,
                density,
                density_width,
                density_height,
                freeze_mask,
                edge_margin,
                push_left,
            );
            let rgba = sample_rgba8_half_up(source, sample_u, sample_v);
            let offset = (y * width + x) * 4;
            output[offset..offset + 4].copy_from_slice(&rgba);
        }
    }
    Image {
        width: source.width,
        height: source.height,
        pixels: output,
    }
}

#[allow(clippy::too_many_arguments)]
fn brush_sample_uv(
    u: f32,
    v: f32,
    mesh_width: f32,
    mesh_height: f32,
    brush: &Brush,
    warp_direction: f32,
    density: &[f32],
    density_width: usize,
    density_height: usize,
    freeze_mask: &[u8],
    edge_margin: f32,
    push_left: bool,
) -> (f32, f32) {
    let src_x = u * mesh_width;
    let src_y = v * mesh_height;
    let dx = src_x - brush.position.x;
    let dy = src_y - brush.position.y;
    if dx * dx + dy * dy > brush.radius * brush.radius {
        return (u, v);
    }

    let radius_squared = brush.radius * brush.radius;
    let density_value = sample_density(
        density,
        density_width,
        density_height,
        (dx * dx + dy * dy) / radius_squared,
        0.5,
    );
    let freeze = sample_freeze(freeze_mask, mesh_width as usize, mesh_height as usize, u, v);
    let mask = 1.0 - freeze * warp_direction;
    let edge = edge_factor(u, v, edge_margin);
    let scale = density_value * brush.pressure * edge * mask;

    let (mut dst_u, mut dst_v) = if push_left {
        (
            u - brush.direction.y * scale / mesh_height,
            v + brush.direction.x * scale / mesh_width,
        )
    } else {
        (
            u - brush.direction.x * scale / mesh_width,
            v - brush.direction.y * scale / mesh_height,
        )
    };
    let mut sample_u = dst_u.clamp(0.0, 1.0);
    let mut sample_v = dst_v.clamp(0.0, 1.0);
    if dst_u >= 1.0 || dst_u <= 0.0 {
        sample_u = u - (u - dst_u);
    }
    if dst_v >= 1.0 || dst_v <= 0.0 {
        sample_v = v - (v - dst_v);
    }
    dst_u = sample_u;
    dst_v = sample_v;
    (dst_u, dst_v)
}

fn edge_factor(u: f32, v: f32, edge_margin: f32) -> f32 {
    let dist_x = u.min(1.0 - u);
    let dist_y = v.min(1.0 - v);
    (dist_x.min(dist_y) / edge_margin).clamp(0.0, 1.0)
}

fn sample_density(
    texture: &[f32],
    width: usize,
    height: usize,
    u: f32,
    v: f32,
) -> f32 {
    let x = u.clamp(0.0, 1.0) * width as f32 - 0.5;
    let y = v.clamp(0.0, 1.0) * height as f32 - 0.5;
    let x0 = x.floor();
    let y0 = y.floor();
    let x0i = (x0 as i64).clamp(0, width as i64 - 1) as usize;
    let y0i = (y0 as i64).clamp(0, height as i64 - 1) as usize;
    let x1i = ((x0 as i64 + 1).clamp(0, width as i64 - 1)) as usize;
    let y1i = ((y0 as i64 + 1).clamp(0, height as i64 - 1)) as usize;
    let fx = x - x0;
    let fy = y - y0;
    let top = lerp(texture[y0i * width + x0i], texture[y0i * width + x1i], fx);
    let bottom = lerp(texture[y1i * width + x0i], texture[y1i * width + x1i], fx);
    lerp(top, bottom, fy)
}

fn sample_freeze(mask: &[u8], width: usize, height: usize, u: f32, v: f32) -> f32 {
    let x = u.clamp(0.0, 1.0) * width as f32 - 0.5;
    let y = v.clamp(0.0, 1.0) * height as f32 - 0.5;
    let x0 = x.floor();
    let y0 = y.floor();
    let x0i = (x0 as i64).clamp(0, width as i64 - 1) as usize;
    let y0i = (y0 as i64).clamp(0, height as i64 - 1) as usize;
    let x1i = ((x0 as i64 + 1).clamp(0, width as i64 - 1)) as usize;
    let y1i = ((y0 as i64 + 1).clamp(0, height as i64 - 1)) as usize;
    let fx = x - x0;
    let fy = y - y0;
    let top = lerp(
        mask[y0i * width + x0i] as f32,
        mask[y0i * width + x1i] as f32,
        fx,
    );
    let bottom = lerp(
        mask[y1i * width + x0i] as f32,
        mask[y1i * width + x1i] as f32,
        fx,
    );
    lerp(top, bottom, fy) / 255.0
}
