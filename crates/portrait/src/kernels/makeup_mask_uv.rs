//! Typed row profiles and a generic mask-UV mesh rasterizer.
//!
//! The six color rows share a shader contract but not a mesh stream. Callers
//! must provide the same-job positions, mask UVs and EBO for the row's draw.

use crate::Refusal;
use crate::kernels::makeup_material::MaskChannel;
use crate::kernels::mesh_attribute;

pub const COLOR_MASK_PFS: [u32; 6] = [100, 103, 105, 117, 118, 30007];
pub const MASK_UV_MISSING: &str = "makeup mask UV";
pub const MASK_UV_TICKET_MISSING: &str = "makeup mask UV ticket";
pub const MASK_UV_FACE_MISSING: &str = "makeup mask UV face";
pub const MASK_UV_ORIGIN_MISSING: &str = "makeup mask UV origin";
pub const MASK_UV_SIZE_MISSING: &str = "makeup mask UV size";
pub const FACE2D_MASK_ATTRIBUTE_STRIDE: u32 = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MakeupRoiRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorMaskProfile {
    pub pf: u32,
    pub mask_channel: MaskChannel,
    pub mix_color: [u8; 4],
    pub filter_type: u32,
    /// Placement metadata only; never a substitute for per-pixel mask UV.
    pub roi: MakeupRoiRect,
}

pub fn color_mask_profile(pf: u32) -> Result<ColorMaskProfile, Refusal> {
    let (mask_channel, mix_color, x, y, width, height) = match pf {
        100 => (MaskChannel::RB, [100, 82, 70, 255], 210, 410, 260, 110),
        117 => (MaskChannel::RB, [100, 82, 70, 255], 540, 410, 260, 110),
        103 => (MaskChannel::RG, [39, 15, 6, 255], 200, 400, 300, 300),
        30007 => (MaskChannel::RG, [39, 15, 6, 255], 500, 400, 300, 300),
        105 => (MaskChannel::RG, [100, 45, 43, 255], 200, 400, 300, 300),
        118 => (MaskChannel::RG, [142, 71, 68, 255], 200, 400, 300, 300),
        _ => return Err(Refusal::UnknownEffect(pf)),
    };
    Ok(ColorMaskProfile {
        pf,
        mask_channel,
        mix_color,
        filter_type: 11,
        roi: MakeupRoiRect {
            x,
            y,
            width,
            height,
        },
    })
}

/// One row's explicit same-job triangle stream in output-image pixels.
#[derive(Clone, Copy, Debug)]
pub struct MeshMaskUvInputs<'a> {
    /// Same-job identity. Zero is never accepted as a missing-value sentinel.
    pub ticket: Option<u64>,
    /// Face identity from the mesh wrapper; zero is valid per-face data.
    pub face_id: Option<u32>,
    pub positions: &'a [[f32; 2]],
    pub mask_uv: &'a [[f32; 2]],
    pub indices: &'a [u16],
    pub width: u32,
    pub height: u32,
}

/// A raw Face2D source point stream already identified by the caller as the
/// mask-texcoord stream. The transform is the recovered producer operation:
/// `uv = (point - origin) / size`, without clamp or Y flip.
#[derive(Clone, Copy, Debug)]
pub struct Face2dMaskUvInputs<'a> {
    pub ticket: Option<u64>,
    pub face_id: Option<u32>,
    pub mask_source_points: &'a [[f32; 2]],
    pub indices: &'a [u16],
    pub origin: [i32; 2],
    pub size: [i32; 2],
    pub width: u32,
    pub height: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MeshMaskPixelUv {
    ticket: Option<u64>,
    face_id: Option<u32>,
    uv: Vec<[f32; 2]>,
    writes: Vec<u16>,
}

impl MeshMaskPixelUv {
    pub fn ticket(&self) -> Option<u64> {
        self.ticket
    }

    pub fn face_id(&self) -> Option<u32> {
        self.face_id
    }

    pub fn uv(&self) -> &[[f32; 2]] {
        &self.uv
    }

    pub fn writes(&self) -> &[u16] {
        &self.writes
    }

    pub fn covered(&self, pixel: usize) -> bool {
        self.writes.get(pixel).is_some_and(|writes| *writes > 0)
    }
}

/// Rasterizes an explicit mesh. A missing stream is the only case that becomes
/// `MissingInput("makeup mask UV")`; malformed supplied data keeps its shape or
/// range refusal.
pub fn produce_mesh_mask_uv(
    inputs: Option<MeshMaskUvInputs<'_>>,
) -> Result<MeshMaskPixelUv, Refusal> {
    let inputs = inputs.ok_or(Refusal::MissingInput(MASK_UV_MISSING))?;
    let ticket = inputs
        .ticket
        .ok_or(Refusal::MissingInput(MASK_UV_TICKET_MISSING))?;
    if ticket == 0 {
        return Err(Refusal::MissingInput(MASK_UV_TICKET_MISSING));
    }
    let face_id = inputs
        .face_id
        .ok_or(Refusal::MissingInput(MASK_UV_FACE_MISSING))?;
    if inputs.width == 0 || inputs.height == 0 {
        return Err(Refusal::ShapeMismatch(MASK_UV_MISSING));
    }
    if inputs.positions.len() != inputs.mask_uv.len() {
        return Err(Refusal::ShapeMismatch(MASK_UV_MISSING));
    }
    let faces: Vec<u32> = inputs
        .indices
        .iter()
        .map(|index| u32::from(*index))
        .collect();
    let dense = mesh_attribute::rasterize(
        inputs.width,
        inputs.height,
        inputs.positions,
        inputs.mask_uv,
        &faces,
    )
    .map_err(|refusal| match refusal {
        Refusal::ShapeMismatch("mesh attribute vertices") => {
            Refusal::ShapeMismatch(MASK_UV_MISSING)
        }
        Refusal::ShapeMismatch("mesh attribute faces") => Refusal::ShapeMismatch(MASK_UV_MISSING),
        Refusal::ValueOutOfRange("mesh attribute vertex") => {
            Refusal::ValueOutOfRange(MASK_UV_MISSING)
        }
        Refusal::ValueOutOfRange("mesh attribute face index") => {
            Refusal::ValueOutOfRange(MASK_UV_MISSING)
        }
        other => other,
    })?;
    Ok(MeshMaskPixelUv {
        ticket: Some(ticket),
        face_id: Some(face_id),
        uv: dense.values,
        writes: dense.writes,
    })
}

pub fn produce_face2d_mask_uv(
    inputs: Option<Face2dMaskUvInputs<'_>>,
) -> Result<MeshMaskPixelUv, Refusal> {
    let inputs = inputs.ok_or(Refusal::MissingInput(MASK_UV_MISSING))?;
    let ticket = inputs
        .ticket
        .ok_or(Refusal::MissingInput(MASK_UV_TICKET_MISSING))?;
    if ticket == 0 {
        return Err(Refusal::MissingInput(MASK_UV_TICKET_MISSING));
    }
    let face_id = inputs
        .face_id
        .ok_or(Refusal::MissingInput(MASK_UV_FACE_MISSING))?;
    if inputs.size[0] == 0 || inputs.size[1] == 0 {
        return Err(Refusal::ValueOutOfRange(MASK_UV_SIZE_MISSING));
    }
    if inputs.mask_source_points.is_empty() {
        return Err(Refusal::ShapeMismatch(MASK_UV_MISSING));
    }
    if inputs
        .mask_source_points
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange(MASK_UV_ORIGIN_MISSING));
    }
    let origin = [inputs.origin[0] as f32, inputs.origin[1] as f32];
    let size = [inputs.size[0] as f32, inputs.size[1] as f32];
    let mask_uv: Vec<[f32; 2]> = inputs
        .mask_source_points
        .iter()
        .map(|point| {
            [
                (point[0] - origin[0]) / size[0],
                (point[1] - origin[1]) / size[1],
            ]
        })
        .collect();
    produce_mesh_mask_uv(Some(MeshMaskUvInputs {
        ticket: Some(ticket),
        face_id: Some(face_id),
        positions: inputs.mask_source_points,
        mask_uv: &mask_uv,
        indices: inputs.indices,
        width: inputs.width,
        height: inputs.height,
    }))
}
