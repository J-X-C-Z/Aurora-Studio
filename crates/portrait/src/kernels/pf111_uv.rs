//! PF111's typed 49-vertex material-UV producer.
//!
//! The producer accepts the same-job eight-point EyePupil model package, the
//! packed result of its two 25-point contours, the selected material extent,
//! and the Normal Eye mesh `+0x10` index package. It owns only the recovered
//! `uv = point / (width, height)` transform.

use crate::Refusal;
use crate::kernels::mesh_attribute::{self, DenseAttribute};

pub const EYE_PUPIL_POINT_COUNT: usize = 8;
pub const CONTOUR_POINT_COUNT: usize = 25;
pub const PF111_VERTEX_COUNT: usize = 49;
pub const NORMAL_EYE_EBO_OFFSET: usize = 0x10;

pub const MODEL_MISSING: &str = "PF111 eye pupil model 8";
pub const CONTOURS_MISSING: &str = "PF111 eye pupil contour 49";
pub const MATERIAL_EXTENT_MISSING: &str = "PF111 material extent";
pub const NORMAL_EYE_EBO_MISSING: &str = "PF111 Normal Eye EBO";
pub const EYE_PUPIL_POSITION_MISSING: &str = "PF111 EyePupil position 49";
pub const NORMAL_EYE_UV_MISSING: &str = "PF111 Normal Eye UV 49";
pub const RASTER_EXTENT_MISSING: &str = "PF111 raster extent";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EyeSide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyePupilModel8 {
    points: [[f32; 2]; EYE_PUPIL_POINT_COUNT],
}

impl EyePupilModel8 {
    pub fn new(points: [[f32; 2]; EYE_PUPIL_POINT_COUNT]) -> Result<Self, Refusal> {
        if !points
            .iter()
            .all(|point| point[0].is_finite() && point[1].is_finite())
        {
            return Err(Refusal::ValueOutOfRange(MODEL_MISSING));
        }
        Ok(Self { points })
    }

    pub fn points(self) -> [[f32; 2]; EYE_PUPIL_POINT_COUNT] {
        self.points
    }
}

/// The packed output of the eight-point -> two 25-point contour stage.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PackedContour49<'a> {
    points: &'a [[f32; 2]],
}

impl<'a> PackedContour49<'a> {
    pub fn new(points: &'a [[f32; 2]]) -> Result<Self, Refusal> {
        if points.len() != PF111_VERTEX_COUNT {
            return Err(Refusal::ShapeMismatch(CONTOURS_MISSING));
        }
        if !points
            .iter()
            .all(|point| point[0].is_finite() && point[1].is_finite())
        {
            return Err(Refusal::ValueOutOfRange(CONTOURS_MISSING));
        }
        Ok(Self { points })
    }

    pub fn points(self) -> &'a [[f32; 2]] {
        self.points
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaterialExtent {
    width: u32,
    height: u32,
}

impl MaterialExtent {
    pub fn new(width: u32, height: u32) -> Result<Self, Refusal> {
        if width == 0 || height == 0 {
            return Err(Refusal::ValueOutOfRange(MATERIAL_EXTENT_MISSING));
        }
        Ok(Self { width, height })
    }

    pub fn width(self) -> u32 {
        self.width
    }

    pub fn height(self) -> u32 {
        self.height
    }
}

/// Normal Eye mesh `+0x10`, bound as `GL_ELEMENT_ARRAY_BUFFER` for PF111 draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormalEyeEbo<'a> {
    indices: &'a [u16],
}

impl<'a> NormalEyeEbo<'a> {
    pub fn new(indices: &'a [u16]) -> Result<Self, Refusal> {
        if indices.is_empty() {
            return Err(Refusal::ShapeMismatch(NORMAL_EYE_EBO_MISSING));
        }
        if indices
            .iter()
            .any(|index| *index as usize >= PF111_VERTEX_COUNT)
        {
            return Err(Refusal::ValueOutOfRange(NORMAL_EYE_EBO_MISSING));
        }
        Ok(Self { indices })
    }

    pub fn indices(self) -> &'a [u16] {
        self.indices
    }
}

fn dense_pair_stream<'a>(
    values: &'a [[f32; 2]],
    missing: &'static str,
) -> Result<&'a [[f32; 2]], Refusal> {
    if values.len() != PF111_VERTEX_COUNT {
        return Err(Refusal::ShapeMismatch(missing));
    }
    if values.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange(missing));
    }
    Ok(values)
}

/// EyePupil mesh `+0x08`: destination geometry in image pixels, top-left origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EyePupilPosition49<'a> {
    positions: &'a [[f32; 2]],
}

impl<'a> EyePupilPosition49<'a> {
    pub fn new(positions: &'a [[f32; 2]]) -> Result<Self, Refusal> {
        Ok(Self {
            positions: dense_pair_stream(positions, EYE_PUPIL_POSITION_MISSING)?,
        })
    }

    pub fn positions(self) -> &'a [[f32; 2]] {
        self.positions
    }
}

/// Normal Eye mesh `+0x08`: one `float2` UV per PF111 vertex.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NormalEyeUv49<'a> {
    uv: &'a [[f32; 2]],
}

impl<'a> NormalEyeUv49<'a> {
    pub fn new(uv: &'a [[f32; 2]]) -> Result<Self, Refusal> {
        Ok(Self {
            uv: dense_pair_stream(uv, NORMAL_EYE_UV_MISSING)?,
        })
    }

    pub fn uv(self) -> &'a [[f32; 2]] {
        self.uv
    }
}

/// Output frame extent. Pixel centres are `(x + 0.5, y + 0.5)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RasterExtent {
    width: u32,
    height: u32,
}

impl RasterExtent {
    pub fn new(width: u32, height: u32) -> Result<Self, Refusal> {
        if width == 0 || height == 0 {
            return Err(Refusal::ValueOutOfRange(RASTER_EXTENT_MISSING));
        }
        Ok(Self { width, height })
    }

    pub fn width(self) -> u32 {
        self.width
    }

    pub fn height(self) -> u32 {
        self.height
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Pf111UvInputs<'a> {
    pub side: EyeSide,
    pub model: Option<EyePupilModel8>,
    pub contours: Option<PackedContour49<'a>>,
    pub material_extent: Option<MaterialExtent>,
    pub normal_eye_ebo: Option<NormalEyeEbo<'a>>,
}

#[derive(Clone, Copy, Debug)]
pub struct Pf111UvMesh<'a> {
    side: EyeSide,
    model: EyePupilModel8,
    material_uv: [[f32; 2]; PF111_VERTEX_COUNT],
    normal_eye_ebo: NormalEyeEbo<'a>,
}

impl<'a> Pf111UvMesh<'a> {
    pub fn side(self) -> EyeSide {
        self.side
    }

    pub fn model(self) -> EyePupilModel8 {
        self.model
    }

    pub fn material_uv(&self) -> &[[f32; 2]; PF111_VERTEX_COUNT] {
        &self.material_uv
    }

    pub fn normal_eye_ebo(self) -> NormalEyeEbo<'a> {
        self.normal_eye_ebo
    }
}

pub fn produce(inputs: Pf111UvInputs<'_>) -> Result<Pf111UvMesh<'_>, Refusal> {
    let model = inputs.model.ok_or(Refusal::MissingInput(MODEL_MISSING))?;
    let contours = inputs
        .contours
        .ok_or(Refusal::MissingInput(CONTOURS_MISSING))?;
    let extent = inputs
        .material_extent
        .ok_or(Refusal::MissingInput(MATERIAL_EXTENT_MISSING))?;
    let normal_eye_ebo = inputs
        .normal_eye_ebo
        .ok_or(Refusal::MissingInput(NORMAL_EYE_EBO_MISSING))?;

    let width = extent.width() as f32;
    let height = extent.height() as f32;
    let material_uv = std::array::from_fn(|index| {
        let point = contours.points()[index];
        [point[0] / width, point[1] / height]
    });

    Ok(Pf111UvMesh {
        side: inputs.side,
        model,
        material_uv,
        normal_eye_ebo,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct Pf111RasterInputs<'a> {
    pub side: EyeSide,
    pub positions: Option<EyePupilPosition49<'a>>,
    pub normal_eye_uv: Option<NormalEyeUv49<'a>>,
    pub extent: Option<RasterExtent>,
    pub normal_eye_ebo: Option<NormalEyeEbo<'a>>,
}

/// PF111's per-pixel Makeup inputs. The effect has no mask stream.
///
/// `material_uv` is dense, but only pixels with `writes[p] > 0` were covered by
/// the EBO. Uncovered slots are `[0, 0]` and must not be sampled as product UV.
#[derive(Clone, Debug, PartialEq)]
pub struct Pf111PixelUv {
    side: EyeSide,
    mask_uv: Option<Vec<[f32; 2]>>,
    material_uv: Vec<[f32; 2]>,
    writes: Vec<u16>,
}

impl Pf111PixelUv {
    pub fn side(&self) -> EyeSide {
        self.side
    }

    pub fn mask_uv(&self) -> Option<&[[f32; 2]]> {
        self.mask_uv.as_deref()
    }

    pub fn material_uv(&self) -> &[[f32; 2]] {
        &self.material_uv
    }

    pub fn writes(&self) -> &[u16] {
        &self.writes
    }

    pub fn covered(&self, pixel: usize) -> bool {
        self.writes.get(pixel).is_some_and(|writes| *writes > 0)
    }
}

pub(crate) struct RasterArgs<'a> {
    pub side: EyeSide,
    pub width: u32,
    pub height: u32,
    pub positions: &'a [[f32; 2]],
    pub normal_eye_uv: &'a [[f32; 2]],
    pub faces: Vec<u32>,
}

pub(crate) fn raster_args(inputs: Pf111RasterInputs<'_>) -> Result<RasterArgs<'_>, Refusal> {
    let positions = inputs
        .positions
        .ok_or(Refusal::MissingInput(EYE_PUPIL_POSITION_MISSING))?;
    let normal_eye_uv = inputs
        .normal_eye_uv
        .ok_or(Refusal::MissingInput(NORMAL_EYE_UV_MISSING))?;
    let extent = inputs
        .extent
        .ok_or(Refusal::MissingInput(RASTER_EXTENT_MISSING))?;
    let normal_eye_ebo = inputs
        .normal_eye_ebo
        .ok_or(Refusal::MissingInput(NORMAL_EYE_EBO_MISSING))?;
    if normal_eye_ebo.indices().len() % 3 != 0 {
        return Err(Refusal::ShapeMismatch(NORMAL_EYE_EBO_MISSING));
    }

    Ok(RasterArgs {
        side: inputs.side,
        width: extent.width(),
        height: extent.height(),
        positions: positions.positions(),
        normal_eye_uv: normal_eye_uv.uv(),
        faces: normal_eye_ebo
            .indices()
            .iter()
            .map(|index| u32::from(*index))
            .collect(),
    })
}

pub(crate) fn from_dense(side: EyeSide, dense: DenseAttribute) -> Pf111PixelUv {
    Pf111PixelUv {
        side,
        mask_uv: None,
        material_uv: dense.values,
        writes: dense.writes,
    }
}

/// CPU reference for PF111's 49-vertex Normal Eye stream.
///
/// Geometry uses image pixels with a top-left origin and Y increasing down.
/// Triangle coverage follows `mesh_attribute::rasterize`; no vendor-specific
/// raster sample rule is claimed.
pub fn rasterize_pixels(inputs: Pf111RasterInputs<'_>) -> Result<Pf111PixelUv, Refusal> {
    let args = raster_args(inputs)?;
    let dense = mesh_attribute::rasterize(
        args.width,
        args.height,
        args.positions,
        args.normal_eye_uv,
        &args.faces,
    )?;
    Ok(from_dense(args.side, dense))
}
