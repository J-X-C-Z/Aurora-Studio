//! Same-job makeup mask/UV bundle for the fourteen B3 rows.
//!
//! Static presets name materials and mask semantics. They do not contain owner
//! mask pixels, Face2D VBO/EBO bytes, or draw-time UV. This module keeps those
//! as separate inputs and refuses by name when any of them is absent.

use aurora_portrait::kernels::makeup_mask_uv::{
    Face2dMaskUvInputs, MeshMaskPixelUv, MeshMaskUvInputs, produce_face2d_mask_uv,
    produce_mesh_mask_uv,
};
use aurora_portrait::kernels::makeup_material::MaskChannel;
use aurora_portrait::{Image, Refusal};
use std::sync::Arc;

use crate::portrait_stage::{MakeupInput, SuitPassInput, SuitPassKindInput};

pub const B3_MAKEUP_ROWS: [u32; 14] = [
    8, 9, 100, 103, 105, 106, 107, 111, 115, 116, 117, 118, 30007, 90032,
];
pub const SUIT_MATERIAL_MASK_COUNT: usize = 12;

pub const BUNDLE_MISSING: &str = "makeup bundle";
pub const VBO_BYTES_MISSING: &str = "makeup VBO bytes";
pub const EBO_BYTES_MISSING: &str = "makeup EBO bytes";
pub const MATERIAL_VBO_BYTES_MISSING: &str = "makeup material VBO bytes";
pub const OWNER_MASK_MISSING: &str = "makeup owner mask";
pub const MATERIAL_MISSING: &str = "makeup material";
pub const MATERIAL_UV_MISSING: &str = "makeup material UV";
pub const SOURCE_SHA_MISSING: &str = "makeup source SHA";
pub const SOURCE_EFFECT_MISSING: &str = "makeup source effect";
pub const SOURCE_VARIANT_UNSUPPORTED: &str = "makeup source variant";
pub const SOURCE_3DMM_MISSING: &str = "makeup 3d face source";
pub const SUIT_BUNDLE_MISSING: &str = "suit material bundle";
pub const SUIT_MASK_MISSING: &str = "suit mask";
pub const SUIT_MATERIAL_MISSING: &str = "suit material";
pub const SUIT_MATERIAL_UV_MISSING: &str = "suit material uv";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MakeupRowKind {
    ColorMask,
    MaterialOnly,
    HistoricalColor,
    HistoricalImage,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MakeupRowProfile {
    pub pf: u32,
    pub kind: MakeupRowKind,
    pub mask_semantic: &'static str,
    /// Static family default. Draw-time defines remain separate evidence.
    pub declared_mask_channel: Option<MaskChannel>,
    pub requires_material: bool,
}

impl MakeupRowProfile {
    pub fn requires_owner_mask(self) -> bool {
        self.kind != MakeupRowKind::MaterialOnly
    }

    pub fn is_historical(self) -> bool {
        matches!(
            self.kind,
            MakeupRowKind::HistoricalColor | MakeupRowKind::HistoricalImage
        )
    }
}

pub fn row_profile(pf: u32) -> Result<MakeupRowProfile, Refusal> {
    let (kind, mask_semantic, declared_mask_channel, requires_material) = match pf {
        100 => (
            MakeupRowKind::ColorMask,
            "EyeBrowMaskType=5",
            Some(MaskChannel::RB),
            false,
        ),
        117 => (
            MakeupRowKind::ColorMask,
            "EyeBrowMaskType=5",
            Some(MaskChannel::RB),
            false,
        ),
        103 | 30007 => (
            MakeupRowKind::ColorMask,
            "EyeMaskType=4",
            Some(MaskChannel::RG),
            false,
        ),
        105 | 118 => (
            MakeupRowKind::ColorMask,
            "EyeLashMaskType=4; L8 uploaded as RG=min(2L,1)",
            Some(MaskChannel::RG),
            false,
        ),
        111 => (
            MakeupRowKind::MaterialOnly,
            "EyePupilMaskType=0",
            None,
            true,
        ),
        106 => (
            MakeupRowKind::HistoricalColor,
            "BlusherMaskType=15",
            Some(MaskChannel::Rgba),
            false,
        ),
        116 => (
            MakeupRowKind::HistoricalImage,
            "BlusherMaskType=15",
            Some(MaskChannel::Rgba),
            true,
        ),
        8 => (
            MakeupRowKind::HistoricalImage,
            "face-parsing RGBA",
            Some(MaskChannel::Rgba),
            true,
        ),
        9 => (
            MakeupRowKind::HistoricalImage,
            "blusher RGBA",
            Some(MaskChannel::Rgba),
            true,
        ),
        107 => (
            MakeupRowKind::HistoricalImage,
            "LipMaskType main mask; extra mask optional",
            Some(MaskChannel::R),
            true,
        ),
        115 => (
            MakeupRowKind::HistoricalImage,
            "MASK_RGBA",
            Some(MaskChannel::Rgba),
            true,
        ),
        90032 => (
            MakeupRowKind::HistoricalImage,
            "EyeMaskType=10",
            Some(MaskChannel::Rgb),
            true,
        ),
        _ => return Err(Refusal::UnknownEffect(pf)),
    };
    Ok(MakeupRowProfile {
        pf,
        kind,
        mask_semantic,
        declared_mask_channel,
        requires_material,
    })
}

/// Static effect families observed for each B3 row. A row with more than one
/// family needs the selected `effect.json` variant before its inputs are known.
pub fn allowed_source_effects(pf: u32) -> Result<&'static [&'static str], Refusal> {
    Ok(match pf {
        8 | 9 | 116 => &["BlusherImageMakeup"],
        100 | 117 => &["NormalEyeBrowColorMakeup"],
        103 | 30007 => &["SymEyeColorMakeup"],
        105 | 118 => &["SymEyeLashColorMakeup"],
        106 => &["BlusherColorMakeup"],
        107 => &["LipColor", "LipStickImageMakeup"],
        111 => &["EyePupilMakeup"],
        115 => &["3DMMFaceSticker", "BlusherImageMakeup"],
        90032 => &["SymEyeImageMakeup"],
        _ => return Err(Refusal::UnknownEffect(pf)),
    })
}

/// Resolves the row contract only after the selected effect family is known.
/// `3DMMFaceSticker` is not accepted as a 2D mask/UV row.
pub fn row_profile_for_effect(pf: u32, effect: &str) -> Result<MakeupRowProfile, Refusal> {
    if !allowed_source_effects(pf)?.contains(&effect) {
        return Err(Refusal::ValueOutOfRange(SOURCE_VARIANT_UNSUPPORTED));
    }

    let mut profile = row_profile(pf)?;
    match (pf, effect) {
        (107, "LipColor") => {
            profile.kind = MakeupRowKind::ColorMask;
            profile.mask_semantic = "LipColor mask";
            profile.declared_mask_channel = Some(MaskChannel::R);
            profile.requires_material = false;
        }
        (107, "LipStickImageMakeup") => {
            profile.kind = MakeupRowKind::HistoricalImage;
            profile.mask_semantic = "LipMaskType main mask; extra mask optional";
            profile.declared_mask_channel = Some(MaskChannel::R);
            profile.requires_material = true;
        }
        (115, "BlusherImageMakeup") => {
            profile.kind = MakeupRowKind::HistoricalImage;
            profile.mask_semantic = "BlusherMaskType=15";
            profile.declared_mask_channel = Some(MaskChannel::Rgba);
            profile.requires_material = true;
        }
        (115, "3DMMFaceSticker") => {
            return Err(Refusal::MissingInput(SOURCE_3DMM_MISSING));
        }
        _ => {}
    }
    Ok(profile)
}

/// One of the twelve `MU00000001` material/owner-mask pairings. The material
/// pixels are static assets; the owner-mask pixels and UVs are runtime inputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuitMaterialMaskSlot {
    pub order: u8,
    pub group: &'static str,
    pub effect: &'static str,
    pub material: &'static str,
    pub mask_field: &'static str,
    pub mask_value: u32,
    pub mask_channel: MaskChannel,
    pub force_use_shelter: bool,
}

pub const SUIT_MATERIAL_MASK_SLOTS: [SuitMaterialMaskSlot; SUIT_MATERIAL_MASK_COUNT] = [
    SuitMaterialMaskSlot {
        order: 1,
        group: "Bronzer",
        effect: "BlusherImageMakeup",
        material: "Resources/effect/config/Makeups/Suit/MU00000001/Bronzer/xr.png",
        mask_field: "BlusherMaskType",
        mask_value: 15,
        mask_channel: MaskChannel::Rgba,
        force_use_shelter: false,
    },
    SuitMaterialMaskSlot {
        order: 2,
        group: "Bronzer",
        effect: "BlusherImageMakeup",
        material: "Resources/effect/config/Makeups/Suit/MU00000001/Bronzer/xr_SideFace.png",
        mask_field: "BlusherMaskType",
        mask_value: 15,
        mask_channel: MaskChannel::Rgba,
        force_use_shelter: false,
    },
    SuitMaterialMaskSlot {
        order: 3,
        group: "Blusher",
        effect: "BlusherImageMakeup",
        material: "Resources/effect/config/Makeups/Suit/MU00000001/Blusher/sh.png",
        mask_field: "BlusherMaskType",
        mask_value: 15,
        mask_channel: MaskChannel::Rgba,
        force_use_shelter: false,
    },
    SuitMaterialMaskSlot {
        order: 4,
        group: "Eyebrow",
        effect: "NormalEyeBrowImageMakeup",
        material: "Resources/effect/config/Makeups/Suit/MU00000001/Eyebrow/mmz.png",
        mask_field: "EyeBrowMaskType",
        mask_value: 5,
        mask_channel: MaskChannel::RB,
        force_use_shelter: true,
    },
    SuitMaterialMaskSlot {
        order: 5,
        group: "Eyebrow",
        effect: "NormalEyeBrowImageMakeup",
        material: "Resources/effect/config/Makeups/Suit/MU00000001/Eyebrow/mmy.png",
        mask_field: "EyeBrowMaskType",
        mask_value: 5,
        mask_channel: MaskChannel::RB,
        force_use_shelter: true,
    },
    SuitMaterialMaskSlot {
        order: 6,
        group: "Eyemakeup",
        effect: "SymEyeImageMakeup",
        material: "Resources/effect/config/Makeups/Suit/MU00000001/Eyemakeup/yyz.png",
        mask_field: "EyeMaskType",
        mask_value: 10,
        mask_channel: MaskChannel::Rgb,
        force_use_shelter: true,
    },
    SuitMaterialMaskSlot {
        order: 7,
        group: "Eyemakeup",
        effect: "SymEyeImageMakeup",
        material: "Resources/effect/config/Makeups/Suit/MU00000001/Eyemakeup/yyy.png",
        mask_field: "EyeMaskType",
        mask_value: 10,
        mask_channel: MaskChannel::Rgb,
        force_use_shelter: true,
    },
    SuitMaterialMaskSlot {
        order: 8,
        group: "Eyeliner",
        effect: "SymEyeImageMakeup",
        material: "Resources/effect/config/Makeups/Suit/MU00000001/Eyeliner/yxy.png",
        mask_field: "EyeMaskType",
        mask_value: 4,
        mask_channel: MaskChannel::RG,
        force_use_shelter: true,
    },
    SuitMaterialMaskSlot {
        order: 9,
        group: "Eyeliner",
        effect: "SymEyeImageMakeup",
        material: "Resources/effect/config/Makeups/Suit/MU00000001/Eyeliner/yxz.png",
        mask_field: "EyeMaskType",
        mask_value: 4,
        mask_channel: MaskChannel::RG,
        force_use_shelter: true,
    },
    SuitMaterialMaskSlot {
        order: 10,
        group: "Eyelash",
        effect: "SymEyeLashImageMakeup",
        material: "Resources/effect/config/Makeups/Suit/MU00000001/Eyelash/jmz.png",
        mask_field: "EyeLashMaskType",
        mask_value: 4,
        mask_channel: MaskChannel::RG,
        force_use_shelter: true,
    },
    SuitMaterialMaskSlot {
        order: 11,
        group: "Eyelash",
        effect: "SymEyeLashImageMakeup",
        material: "Resources/effect/config/Makeups/Suit/MU00000001/Eyelash/jmy.png",
        mask_field: "EyeLashMaskType",
        mask_value: 4,
        mask_channel: MaskChannel::RG,
        force_use_shelter: true,
    },
    SuitMaterialMaskSlot {
        order: 12,
        group: "Lipstick",
        effect: "LipStickImageMakeup",
        material: "Resources/effect/config/Makeups/Suit/MU00000001/Lipstick/zc.png",
        mask_field: "LipMaskType",
        mask_value: 3,
        mask_channel: MaskChannel::R,
        force_use_shelter: false,
    },
];

pub fn suit_material_mask_slot(order: u8) -> Result<SuitMaterialMaskSlot, Refusal> {
    SUIT_MATERIAL_MASK_SLOTS
        .get(order.saturating_sub(1) as usize)
        .copied()
        .filter(|slot| slot.order == order)
        .ok_or(Refusal::MissingInput("suit material slot"))
}

/// Raw draw bytes. Empty VBO/EBO is a missing input, never a zero geometry.
#[derive(Clone, Copy, Debug)]
pub struct Face2dDrawBytes<'a> {
    pub vertices: &'a [u8],
    pub vertex_stride: usize,
    pub position_offset: usize,
    pub indices_u16: &'a [u8],
    pub draw_first_index: usize,
    pub draw_index_count: usize,
}

impl<'a> Face2dDrawBytes<'a> {
    pub fn decode_positions(&self) -> Result<Vec<[f32; 2]>, Refusal> {
        self.decode_attribute(self.position_offset, VBO_BYTES_MISSING)
    }

    pub fn decode_attribute(
        &self,
        offset: usize,
        name: &'static str,
    ) -> Result<Vec<[f32; 2]>, Refusal> {
        decode_xy_attribute(
            self.vertices,
            self.vertex_stride,
            offset,
            VBO_BYTES_MISSING,
            name,
        )
    }

    pub fn decode_indices(&self) -> Result<Vec<u16>, Refusal> {
        if self.indices_u16.is_empty() {
            return Err(Refusal::MissingInput(EBO_BYTES_MISSING));
        }
        if self.indices_u16.len() % 2 != 0 || self.draw_index_count == 0 {
            return Err(Refusal::ShapeMismatch(EBO_BYTES_MISSING));
        }
        if self.draw_index_count % 3 != 0 {
            return Err(Refusal::ShapeMismatch(EBO_BYTES_MISSING));
        }
        let start = self
            .draw_first_index
            .checked_mul(2)
            .ok_or(Refusal::ShapeMismatch(EBO_BYTES_MISSING))?;
        let length = self
            .draw_index_count
            .checked_mul(2)
            .ok_or(Refusal::ShapeMismatch(EBO_BYTES_MISSING))?;
        let end = start
            .checked_add(length)
            .ok_or(Refusal::ShapeMismatch(EBO_BYTES_MISSING))?;
        let bytes = self
            .indices_u16
            .get(start..end)
            .ok_or(Refusal::ShapeMismatch(EBO_BYTES_MISSING))?;
        Ok(bytes
            .chunks_exact(2)
            .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
            .collect())
    }
}

fn decode_xy_attribute(
    vertices: &[u8],
    vertex_stride: usize,
    offset: usize,
    missing: &'static str,
    shape: &'static str,
) -> Result<Vec<[f32; 2]>, Refusal> {
    if vertices.is_empty() {
        return Err(Refusal::MissingInput(missing));
    }
    if vertex_stride < 8
        || offset > vertex_stride.saturating_sub(8)
        || vertices.len() % vertex_stride != 0
    {
        return Err(Refusal::ShapeMismatch(shape));
    }
    let mut values = Vec::with_capacity(vertices.len() / vertex_stride);
    for vertex in vertices.chunks_exact(vertex_stride) {
        let x = f32::from_le_bytes(vertex[offset..offset + 4].try_into().expect("four-byte x"));
        let y = f32::from_le_bytes(
            vertex[offset + 4..offset + 8]
                .try_into()
                .expect("four-byte y"),
        );
        values.push([x, y]);
    }
    Ok(values)
}

/// A distinct `float2` VBO for the material attribute. It is intentionally not
/// an offset into the position VBO: the vendor EyePupil draw can take position
/// and `iq` from different Face2D mesh objects.
#[derive(Clone, Copy, Debug)]
pub struct Face2dUvBytes<'a> {
    pub vertices: &'a [u8],
    pub vertex_stride: usize,
    pub offset: usize,
}

impl Face2dUvBytes<'_> {
    pub fn decode(&self) -> Result<Vec<[f32; 2]>, Refusal> {
        decode_xy_attribute(
            self.vertices,
            self.vertex_stride,
            self.offset,
            MATERIAL_VBO_BYTES_MISSING,
            MATERIAL_UV_MISSING,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct MakeupBundleInputRow<'a> {
    pub pf: u32,
    pub source_effect: Option<&'a str>,
    pub face_id: Option<u32>,
    pub draw: Face2dDrawBytes<'a>,
    pub origin: [i32; 2],
    pub size: [i32; 2],
    pub owner_mask: Option<&'a Image>,
    pub extra_owner_mask: Option<&'a Image>,
    pub material: Option<&'a Image>,
    pub material_uv: Option<Face2dUvBytes<'a>>,
}

pub struct MakeupBundleInputs<'a> {
    pub job_id: u64,
    pub ticket: Option<u64>,
    pub frame_size: [u32; 2],
    pub source_sha256: [u8; 32],
    pub rows: &'a [MakeupBundleInputRow<'a>],
}

pub struct MakeupBundleRow<'a> {
    pub profile: MakeupRowProfile,
    pub face_id: u32,
    pub owner_mask: Option<&'a Image>,
    pub extra_owner_mask: Option<&'a Image>,
    pub material: Option<&'a Image>,
    pub mask_uv: Option<MeshMaskPixelUv>,
    pub material_uv: Option<MeshMaskPixelUv>,
}

pub struct MakeupBundle<'a> {
    pub job_id: u64,
    pub ticket: u64,
    pub frame_size: [u32; 2],
    pub source_sha256: [u8; 32],
    pub rows: Vec<MakeupBundleRow<'a>>,
}

impl<'a> MakeupBundleRow<'a> {
    /// Adapts one complete row to the existing `MakeupInput` carrier. Historical
    /// rows still need their historical stage, but the mask/material evidence is
    /// no longer discarded.
    pub fn to_makeup_input(
        &self,
        mix_color: [u8; 4],
        opacity_percent: f32,
        filter_type: u32,
    ) -> Result<MakeupInput<'_>, Refusal> {
        if !opacity_percent.is_finite() || !(0.0..=100.0).contains(&opacity_percent) {
            return Err(Refusal::ValueOutOfRange("makeup opacity"));
        }
        match self.profile.kind {
            MakeupRowKind::MaterialOnly => {
                if self.owner_mask.is_some() || self.mask_uv.is_some() {
                    return Err(Refusal::ValueOutOfRange("PF111 mask must be absent"));
                }
                let material = self
                    .material
                    .ok_or(Refusal::MissingInput(MATERIAL_MISSING))?;
                let material_uv = self
                    .material_uv
                    .as_ref()
                    .ok_or(Refusal::MissingInput(MATERIAL_UV_MISSING))?;
                Ok(MakeupInput {
                    pf: self.profile.pf,
                    mix_color,
                    opacity_percent,
                    filter_type,
                    material: Some(material),
                    material_uv: Some(material_uv.uv()),
                    material_uv_writes: Some(material_uv.writes()),
                    mask: None,
                    mask_uv: None,
                    mask_uv_writes: None,
                })
            }
            MakeupRowKind::ColorMask
            | MakeupRowKind::HistoricalColor
            | MakeupRowKind::HistoricalImage => {
                let mask = self
                    .owner_mask
                    .ok_or(Refusal::MissingInput(OWNER_MASK_MISSING))?;
                let mask_uv = self
                    .mask_uv
                    .as_ref()
                    .ok_or(Refusal::MissingInput("makeup mask UV"))?;
                let (material, material_uv, material_uv_writes) = if self.profile.requires_material
                {
                    let material = self
                        .material
                        .ok_or(Refusal::MissingInput(MATERIAL_MISSING))?;
                    let material_uv = self
                        .material_uv
                        .as_ref()
                        .ok_or(Refusal::MissingInput(MATERIAL_UV_MISSING))?;
                    (
                        Some(material),
                        Some(material_uv.uv()),
                        Some(material_uv.writes()),
                    )
                } else {
                    (None, None, None)
                };
                Ok(MakeupInput {
                    pf: self.profile.pf,
                    mix_color,
                    opacity_percent,
                    filter_type,
                    material,
                    material_uv,
                    material_uv_writes,
                    mask: Some(mask),
                    mask_uv: Some(mask_uv.uv()),
                    mask_uv_writes: Some(mask_uv.writes()),
                })
            }
        }
    }
}

fn validate_image(image: &Image, name: &'static str) -> Result<(), Refusal> {
    image.validate().map_err(|_| Refusal::ShapeMismatch(name))
}

fn map_material_uv_refusal(error: Refusal) -> Refusal {
    match error {
        Refusal::ShapeMismatch(_) => Refusal::ShapeMismatch(MATERIAL_UV_MISSING),
        Refusal::ValueOutOfRange(_) => Refusal::ValueOutOfRange(MATERIAL_UV_MISSING),
        other => other,
    }
}

pub fn produce_makeup_bundle<'a>(
    inputs: Option<MakeupBundleInputs<'a>>,
) -> Result<MakeupBundle<'a>, Refusal> {
    let inputs = inputs.ok_or(Refusal::MissingInput(BUNDLE_MISSING))?;
    if inputs.job_id == 0 || inputs.ticket != Some(inputs.job_id) {
        return Err(Refusal::ValueOutOfRange("makeup bundle job"));
    }
    if inputs.frame_size[0] == 0 || inputs.frame_size[1] == 0 {
        return Err(Refusal::ShapeMismatch("makeup frame size"));
    }
    if inputs.source_sha256.iter().all(|byte| *byte == 0) {
        return Err(Refusal::MissingInput(SOURCE_SHA_MISSING));
    }
    if inputs.rows.is_empty() {
        return Err(Refusal::MissingInput("makeup bundle rows"));
    }

    let mut source_rows = inputs.rows.to_vec();
    source_rows.sort_by_key(|row| row.pf);
    let mut rows = Vec::with_capacity(source_rows.len());
    let mut previous_pf = None;
    for source in source_rows {
        if !B3_MAKEUP_ROWS.contains(&source.pf) {
            return Err(Refusal::UnknownEffect(source.pf));
        }
        if previous_pf == Some(source.pf) {
            return Err(Refusal::ValueOutOfRange("makeup bundle PF"));
        }
        previous_pf = Some(source.pf);

        let effect = source
            .source_effect
            .ok_or(Refusal::MissingInput(SOURCE_EFFECT_MISSING))?;
        let profile = row_profile_for_effect(source.pf, effect)?;
        let face_id = source
            .face_id
            .ok_or(Refusal::MissingInput("makeup mask UV face"))?;

        let owner_mask = if profile.requires_owner_mask() {
            let mask = source
                .owner_mask
                .ok_or(Refusal::MissingInput(OWNER_MASK_MISSING))?;
            validate_image(mask, OWNER_MASK_MISSING)?;
            Some(mask)
        } else {
            if source.owner_mask.is_some() {
                return Err(Refusal::ValueOutOfRange("PF111 mask must be absent"));
            }
            None
        };

        if source.pf != 107 && source.extra_owner_mask.is_some() {
            return Err(Refusal::ValueOutOfRange("makeup extra mask row"));
        }
        if let Some(extra_mask) = source.extra_owner_mask {
            validate_image(extra_mask, OWNER_MASK_MISSING)?;
        }

        let positions = source.draw.decode_positions()?;
        let indices = source.draw.decode_indices()?;
        let mask_uv = if profile.requires_owner_mask() {
            Some(produce_face2d_mask_uv(Some(Face2dMaskUvInputs {
                ticket: Some(inputs.job_id),
                face_id: Some(face_id),
                mask_source_points: &positions,
                indices: &indices,
                origin: source.origin,
                size: source.size,
                width: inputs.frame_size[0],
                height: inputs.frame_size[1],
            }))?)
        } else {
            None
        };

        let (material, material_uv) = if profile.requires_material {
            let material = source
                .material
                .ok_or(Refusal::MissingInput(MATERIAL_MISSING))?;
            validate_image(material, MATERIAL_MISSING)?;
            let material_uv_bytes = source
                .material_uv
                .ok_or(Refusal::MissingInput(MATERIAL_UV_MISSING))?;
            let values = material_uv_bytes.decode()?;
            let uv = produce_mesh_mask_uv(Some(MeshMaskUvInputs {
                ticket: Some(inputs.job_id),
                face_id: Some(face_id),
                positions: &positions,
                mask_uv: &values,
                indices: &indices,
                width: inputs.frame_size[0],
                height: inputs.frame_size[1],
            }))
            .map_err(map_material_uv_refusal)?;
            (Some(material), Some(uv))
        } else {
            if source.material.is_some() || source.material_uv.is_some() {
                return Err(Refusal::ValueOutOfRange("makeup material row"));
            }
            (None, None)
        };

        rows.push(MakeupBundleRow {
            profile,
            face_id,
            owner_mask,
            extra_owner_mask: source.extra_owner_mask,
            material,
            mask_uv,
            material_uv,
        });
    }

    Ok(MakeupBundle {
        job_id: inputs.job_id,
        ticket: inputs.job_id,
        frame_size: inputs.frame_size,
        source_sha256: inputs.source_sha256,
        rows,
    })
}

/// Owned draw bytes accepted from an external renderer producer. The values are
/// copied out of the producer's transient command buffer exactly once.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedFace2dDrawBytes {
    pub vertices: Arc<[u8]>,
    pub vertex_stride: usize,
    pub position_offset: usize,
    pub indices_u16: Arc<[u8]>,
    pub draw_first_index: usize,
    pub draw_index_count: usize,
}

impl OwnedFace2dDrawBytes {
    fn borrowed(&self) -> Face2dDrawBytes<'_> {
        Face2dDrawBytes {
            vertices: &self.vertices,
            vertex_stride: self.vertex_stride,
            position_offset: self.position_offset,
            indices_u16: &self.indices_u16,
            draw_first_index: self.draw_first_index,
            draw_index_count: self.draw_index_count,
        }
    }
}

/// Owned material-attribute VBO. It stays distinct from the position VBO.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnedFace2dUvBytes {
    pub vertices: Arc<[u8]>,
    pub vertex_stride: usize,
    pub offset: usize,
}

impl OwnedFace2dUvBytes {
    fn borrowed(&self) -> Face2dUvBytes<'_> {
        Face2dUvBytes {
            vertices: &self.vertices,
            vertex_stride: self.vertex_stride,
            offset: self.offset,
        }
    }
}

/// One complete B3 row supplied by the external producer. Pixels are shared by
/// `Arc`; draw bytes and UV vertices are immutable owned buffers.
#[derive(Clone, Debug)]
pub struct OwnedMakeupBundleInputRow {
    pub pf: u32,
    pub source_effect: Option<String>,
    pub face_id: Option<u32>,
    pub draw: OwnedFace2dDrawBytes,
    pub origin: [i32; 2],
    pub size: [i32; 2],
    pub owner_mask: Option<Arc<Image>>,
    pub extra_owner_mask: Option<Arc<Image>>,
    pub material: Option<Arc<Image>>,
    pub material_uv: Option<OwnedFace2dUvBytes>,
    pub mix_color: [u8; 4],
    pub opacity_percent: f32,
    pub filter_type: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct OwnedMakeupBundleInputs<'a> {
    pub job_id: u64,
    pub ticket: Option<u64>,
    pub frame_size: [u32; 2],
    pub source_sha256: [u8; 32],
    pub rows: &'a [OwnedMakeupBundleInputRow],
}

/// Fully owned producer payload. No field points into the renderer command
/// buffer, so `ProducerInputs` can retain it for the whole portrait job.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedMakeupBundleRow {
    pub profile: MakeupRowProfile,
    pub source_effect: String,
    pub face_id: u32,
    pub owner_mask: Option<Arc<Image>>,
    pub extra_owner_mask: Option<Arc<Image>>,
    pub material: Option<Arc<Image>>,
    pub mask_uv: Option<MeshMaskPixelUv>,
    pub material_uv: Option<MeshMaskPixelUv>,
    pub mix_color: [u8; 4],
    pub opacity_percent: f32,
    pub filter_type: u32,
}

impl OwnedMakeupBundleRow {
    pub fn to_makeup_input(&self) -> Result<MakeupInput<'_>, Refusal> {
        if !self.opacity_percent.is_finite() || !(0.0..=100.0).contains(&self.opacity_percent) {
            return Err(Refusal::ValueOutOfRange("makeup opacity"));
        }
        match self.profile.kind {
            MakeupRowKind::MaterialOnly => {
                if self.owner_mask.is_some() || self.mask_uv.is_some() {
                    return Err(Refusal::ValueOutOfRange("PF111 mask must be absent"));
                }
                let material = self
                    .material
                    .as_deref()
                    .ok_or(Refusal::MissingInput(MATERIAL_MISSING))?;
                let material_uv = self
                    .material_uv
                    .as_ref()
                    .ok_or(Refusal::MissingInput(MATERIAL_UV_MISSING))?;
                Ok(MakeupInput {
                    pf: self.profile.pf,
                    mix_color: self.mix_color,
                    opacity_percent: self.opacity_percent,
                    filter_type: self.filter_type,
                    material: Some(material),
                    material_uv: Some(material_uv.uv()),
                    material_uv_writes: Some(material_uv.writes()),
                    mask: None,
                    mask_uv: None,
                    mask_uv_writes: None,
                })
            }
            MakeupRowKind::ColorMask
            | MakeupRowKind::HistoricalColor
            | MakeupRowKind::HistoricalImage => {
                let mask = self
                    .owner_mask
                    .as_deref()
                    .ok_or(Refusal::MissingInput(OWNER_MASK_MISSING))?;
                let mask_uv = self
                    .mask_uv
                    .as_ref()
                    .ok_or(Refusal::MissingInput("makeup mask UV"))?;
                let (material, material_uv, material_uv_writes) = if self.profile.requires_material
                {
                    let material = self
                        .material
                        .as_deref()
                        .ok_or(Refusal::MissingInput(MATERIAL_MISSING))?;
                    let material_uv = self
                        .material_uv
                        .as_ref()
                        .ok_or(Refusal::MissingInput(MATERIAL_UV_MISSING))?;
                    (
                        Some(material),
                        Some(material_uv.uv()),
                        Some(material_uv.writes()),
                    )
                } else {
                    (None, None, None)
                };
                Ok(MakeupInput {
                    pf: self.profile.pf,
                    mix_color: self.mix_color,
                    opacity_percent: self.opacity_percent,
                    filter_type: self.filter_type,
                    material,
                    material_uv,
                    material_uv_writes,
                    mask: Some(mask),
                    mask_uv: Some(mask_uv.uv()),
                    mask_uv_writes: Some(mask_uv.writes()),
                })
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OwnedMakeupBundle {
    pub job_id: u64,
    pub ticket: u64,
    pub frame_size: [u32; 2],
    pub source_sha256: [u8; 32],
    pub rows: Vec<OwnedMakeupBundleRow>,
}

impl OwnedMakeupBundle {
    pub fn makeup_input(&self, pf: u32) -> Result<MakeupInput<'_>, Refusal> {
        self.rows
            .iter()
            .find(|row| row.profile.pf == pf)
            .ok_or(Refusal::MissingInput("makeup bundle row"))?
            .to_makeup_input()
    }

    pub fn fingerprint(&self) -> u64 {
        fn mix(mut hash: u64, bytes: &[u8]) -> u64 {
            for byte in bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
            hash
        }
        let mut hash = 0xcbf2_9ce4_8422_2325;
        hash = mix(hash, &self.job_id.to_le_bytes());
        hash = mix(hash, &self.ticket.to_le_bytes());
        hash = mix(hash, &self.frame_size[0].to_le_bytes());
        hash = mix(hash, &self.frame_size[1].to_le_bytes());
        hash = mix(hash, &self.source_sha256);
        for row in &self.rows {
            hash = mix(hash, &row.profile.pf.to_le_bytes());
            hash = mix(hash, row.source_effect.as_bytes());
            hash = mix(hash, &row.face_id.to_le_bytes());
            hash = mix(hash, &row.mix_color);
            hash = mix(hash, &row.opacity_percent.to_bits().to_le_bytes());
            hash = mix(hash, &row.filter_type.to_le_bytes());
            for image in [&row.owner_mask, &row.extra_owner_mask, &row.material]
                .into_iter()
                .flatten()
            {
                hash = mix(hash, &image.width.to_le_bytes());
                hash = mix(hash, &image.height.to_le_bytes());
            }
            for uv in [&row.mask_uv, &row.material_uv].into_iter().flatten() {
                for point in uv.uv() {
                    hash = mix(hash, &point[0].to_bits().to_le_bytes());
                    hash = mix(hash, &point[1].to_bits().to_le_bytes());
                }
                for writes in uv.writes() {
                    hash = mix(hash, &writes.to_le_bytes());
                }
            }
        }
        hash
    }
}

/// Validates a producer-owned B3 payload and retains only owned data. The
/// borrowed adapter is transient inside this call.
pub fn produce_owned_makeup_bundle(
    inputs: Option<OwnedMakeupBundleInputs<'_>>,
) -> Result<OwnedMakeupBundle, Refusal> {
    let inputs = inputs.ok_or(Refusal::MissingInput(BUNDLE_MISSING))?;
    let borrowed_rows = inputs
        .rows
        .iter()
        .map(|row| MakeupBundleInputRow {
            pf: row.pf,
            source_effect: row.source_effect.as_deref(),
            face_id: row.face_id,
            draw: row.draw.borrowed(),
            origin: row.origin,
            size: row.size,
            owner_mask: row.owner_mask.as_deref(),
            extra_owner_mask: row.extra_owner_mask.as_deref(),
            material: row.material.as_deref(),
            material_uv: row.material_uv.as_ref().map(OwnedFace2dUvBytes::borrowed),
        })
        .collect::<Vec<_>>();
    let produced = produce_makeup_bundle(Some(MakeupBundleInputs {
        job_id: inputs.job_id,
        ticket: inputs.ticket,
        frame_size: inputs.frame_size,
        source_sha256: inputs.source_sha256,
        rows: &borrowed_rows,
    }))?;

    let mut rows = Vec::with_capacity(produced.rows.len());
    for row in produced.rows {
        let source = inputs
            .rows
            .iter()
            .find(|source| source.pf == row.profile.pf)
            .ok_or(Refusal::MissingInput("makeup bundle row"))?;
        rows.push(OwnedMakeupBundleRow {
            profile: row.profile,
            source_effect: source
                .source_effect
                .clone()
                .ok_or(Refusal::MissingInput(SOURCE_EFFECT_MISSING))?,
            face_id: row.face_id,
            owner_mask: source.owner_mask.as_ref().map(Arc::clone),
            extra_owner_mask: source.extra_owner_mask.as_ref().map(Arc::clone),
            material: source.material.as_ref().map(Arc::clone),
            mask_uv: row.mask_uv,
            material_uv: row.material_uv,
            mix_color: source.mix_color,
            opacity_percent: source.opacity_percent,
            filter_type: source.filter_type,
        });
    }

    Ok(OwnedMakeupBundle {
        job_id: produced.job_id,
        ticket: produced.ticket,
        frame_size: produced.frame_size,
        source_sha256: produced.source_sha256,
        rows,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct SuitMaterialBundleInput<'a> {
    pub order: u8,
    pub face_id: Option<u32>,
    pub draw: Face2dDrawBytes<'a>,
    pub origin: [i32; 2],
    pub size: [i32; 2],
    pub owner_mask: &'a Image,
    pub material: &'a Image,
    pub material_uv: Face2dUvBytes<'a>,
    pub degree: f32,
    pub opacity: f32,
    pub filter_type: u32,
    pub mix_color: [u8; 4],
    pub shelter: Option<f32>,
    pub region: f32,
}

pub struct SuitMaterialBundleInputs<'a> {
    pub job_id: u64,
    pub ticket: Option<u64>,
    pub frame_size: [u32; 2],
    pub source_sha256: [u8; 32],
    pub rows: &'a [SuitMaterialBundleInput<'a>],
}

pub struct SuitMaterialPass<'a> {
    pub slot: SuitMaterialMaskSlot,
    pub owner_mask: &'a Image,
    pub material: &'a Image,
    pub mask_uv: MeshMaskPixelUv,
    pub material_uv: MeshMaskPixelUv,
    pub layer: aurora_portrait::kernels::makeup_material::MakeupLayer,
    pub shelter: f32,
    pub region: f32,
}

impl<'a> SuitMaterialPass<'a> {
    pub fn as_suit_pass(&self) -> SuitPassInput<'_> {
        SuitPassInput {
            kind: SuitPassKindInput::Image {
                material: Some(self.material),
                material_uv: Some(self.material_uv.uv()),
            },
            mask: Some(self.owner_mask),
            mask_uv: Some(self.mask_uv.uv()),
            layer: Some(self.layer),
            shelter: self.shelter,
            region: self.region,
        }
    }
}

pub struct SuitMaterialBundle<'a> {
    pub job_id: u64,
    pub ticket: u64,
    pub frame_size: [u32; 2],
    pub source_sha256: [u8; 32],
    pub passes: Vec<SuitMaterialPass<'a>>,
}

fn checked_weight(value: Option<f32>, name: &'static str) -> Result<f32, Refusal> {
    let value = value.ok_or(Refusal::MissingInput(name))?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Refusal::ValueOutOfRange(name));
    }
    Ok(value)
}

pub fn produce_suit_material_bundle<'a>(
    inputs: Option<SuitMaterialBundleInputs<'a>>,
) -> Result<SuitMaterialBundle<'a>, Refusal> {
    let inputs = inputs.ok_or(Refusal::MissingInput(SUIT_BUNDLE_MISSING))?;
    if inputs.job_id == 0 || inputs.ticket != Some(inputs.job_id) {
        return Err(Refusal::ValueOutOfRange("suit material job"));
    }
    if inputs.frame_size[0] == 0 || inputs.frame_size[1] == 0 {
        return Err(Refusal::ShapeMismatch("suit material frame"));
    }
    if inputs.source_sha256.iter().all(|byte| *byte == 0) {
        return Err(Refusal::MissingInput(SOURCE_SHA_MISSING));
    }
    if inputs.rows.len() != SUIT_MATERIAL_MASK_COUNT {
        return Err(Refusal::ShapeMismatch("suit material slots"));
    }

    let mut source_rows = inputs.rows.to_vec();
    source_rows.sort_by_key(|row| row.order);
    let mut passes = Vec::with_capacity(SUIT_MATERIAL_MASK_COUNT);
    for (index, source) in source_rows.into_iter().enumerate() {
        let slot = suit_material_mask_slot(source.order)?;
        if slot.order as usize != index + 1 {
            return Err(Refusal::ValueOutOfRange("suit material order"));
        }
        let face_id = source
            .face_id
            .ok_or(Refusal::MissingInput("suit mask uv face"))?;
        validate_image(source.owner_mask, SUIT_MASK_MISSING)?;
        validate_image(source.material, SUIT_MATERIAL_MISSING)?;

        let positions = source.draw.decode_positions()?;
        let indices = source.draw.decode_indices()?;
        let mask_uv = produce_face2d_mask_uv(Some(Face2dMaskUvInputs {
            ticket: Some(inputs.job_id),
            face_id: Some(face_id),
            mask_source_points: &positions,
            indices: &indices,
            origin: source.origin,
            size: source.size,
            width: inputs.frame_size[0],
            height: inputs.frame_size[1],
        }))?;
        let values = source.material_uv.decode().map_err(|error| match error {
            Refusal::MissingInput(MATERIAL_UV_MISSING) => {
                Refusal::MissingInput(SUIT_MATERIAL_UV_MISSING)
            }
            Refusal::MissingInput(MATERIAL_VBO_BYTES_MISSING) => {
                Refusal::MissingInput(SUIT_MATERIAL_UV_MISSING)
            }
            other => other,
        })?;
        let material_uv = produce_mesh_mask_uv(Some(MeshMaskUvInputs {
            ticket: Some(inputs.job_id),
            face_id: Some(face_id),
            positions: &positions,
            mask_uv: &values,
            indices: &indices,
            width: inputs.frame_size[0],
            height: inputs.frame_size[1],
        }))
        .map_err(|error| match error {
            Refusal::ShapeMismatch(_) => Refusal::ShapeMismatch(SUIT_MATERIAL_UV_MISSING),
            Refusal::ValueOutOfRange(_) => Refusal::ValueOutOfRange(SUIT_MATERIAL_UV_MISSING),
            other => other,
        })?;

        if !source.degree.is_finite() || !(0.0..=100.0).contains(&source.degree) {
            return Err(Refusal::ValueOutOfRange("suit degree"));
        }
        if !source.opacity.is_finite() || !(0.0..=100.0).contains(&source.opacity) {
            return Err(Refusal::ValueOutOfRange("suit opacity"));
        }
        if source.filter_type == 0 {
            return Err(Refusal::MissingInput("suit filter type"));
        }
        let shelter = if slot.force_use_shelter {
            1.0
        } else {
            checked_weight(source.shelter, "suit shelter")?
        };
        let region = checked_weight(Some(source.region), "suit region")?;
        passes.push(SuitMaterialPass {
            slot,
            owner_mask: source.owner_mask,
            material: source.material,
            mask_uv,
            material_uv,
            layer: aurora_portrait::kernels::makeup_material::MakeupLayer {
                mix_color: source.mix_color,
                opacity_percent: source.opacity,
                degree01: source.degree / 100.0,
                filter_type: source.filter_type,
                mask_channel: slot.mask_channel,
            },
            shelter,
            region,
        });
    }

    Ok(SuitMaterialBundle {
        job_id: inputs.job_id,
        ticket: inputs.job_id,
        frame_size: inputs.frame_size,
        source_sha256: inputs.source_sha256,
        passes,
    })
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    fn xy_bytes(points: &[[f32; 2]]) -> Arc<[u8]> {
        points
            .iter()
            .flat_map(|point| {
                point[0]
                    .to_le_bytes()
                    .into_iter()
                    .chain(point[1].to_le_bytes())
            })
            .collect::<Vec<_>>()
            .into()
    }

    fn draw(frame_size: [u32; 2]) -> OwnedFace2dDrawBytes {
        OwnedFace2dDrawBytes {
            vertices: xy_bytes(&[
                [0.0, 0.0],
                [frame_size[0] as f32, 0.0],
                [0.0, frame_size[1] as f32],
            ]),
            vertex_stride: 8,
            position_offset: 0,
            indices_u16: [0u16, 1, 2]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>()
                .into(),
            draw_first_index: 0,
            draw_index_count: 3,
        }
    }

    fn material_uv() -> OwnedFace2dUvBytes {
        OwnedFace2dUvBytes {
            vertices: xy_bytes(&[[0.05, 0.1], [0.95, 0.2], [0.15, 0.9]]),
            vertex_stride: 8,
            offset: 0,
        }
    }

    fn owner_mask(pf: u32) -> Arc<Image> {
        Arc::new(Image::new(
            4,
            3,
            (0..4 * 3 * 4)
                .map(|index| ((index * 37 + pf as usize * 11) % 256) as u8)
                .collect(),
        )
        .unwrap())
    }

    fn material(pf: u32) -> Arc<Image> {
        Arc::new(Image::new(
            5,
            4,
            (0..5 * 4 * 4)
                .map(|index| ((index * 43 + pf as usize * 7) % 256) as u8)
                .collect(),
        )
        .unwrap())
    }

    pub(crate) fn complete_owned_bundle_for_job(job_id: u64) -> OwnedMakeupBundle {
        let frame_size = [32, 24];
        let rows = B3_MAKEUP_ROWS
            .into_iter()
            .map(|pf| {
                let (source_effect, filter_type, has_material) = match pf {
                    8 | 9 | 115 | 116 => ("BlusherImageMakeup", if pf == 115 { 41 } else { 42 }, true),
                    100 | 117 => ("NormalEyeBrowColorMakeup", 11, false),
                    103 | 30007 => ("SymEyeColorMakeup", 11, false),
                    105 | 118 => ("SymEyeLashColorMakeup", 11, false),
                    106 => ("BlusherColorMakeup", 11, false),
                    107 => ("LipStickImageMakeup", 42, true),
                    111 => ("EyePupilMakeup", 40, true),
                    90032 => ("SymEyeImageMakeup", 41, true),
                    _ => unreachable!(),
                };
                OwnedMakeupBundleInputRow {
                    pf,
                    source_effect: Some(source_effect.to_string()),
                    face_id: Some(pf % 3),
                    draw: draw(frame_size),
                    origin: [0, 0],
                    size: [frame_size[0] as i32, frame_size[1] as i32],
                    owner_mask: (pf != 111).then(|| owner_mask(pf)),
                    extra_owner_mask: None,
                    material: has_material.then(|| material(pf)),
                    material_uv: has_material.then(material_uv),
                    mix_color: [37, 151, 223, 219],
                    opacity_percent: 73.0,
                    filter_type,
                }
            })
            .collect::<Vec<_>>();
        produce_owned_makeup_bundle(Some(OwnedMakeupBundleInputs {
            job_id,
            ticket: Some(job_id),
            frame_size,
            source_sha256: [0x51; 32],
            rows: &rows,
        }))
        .unwrap()
    }

    fn single_row(pf: u32) -> OwnedMakeupBundleInputRow {
        let (source_effect, filter_type, has_material) = match pf {
            100 => ("NormalEyeBrowColorMakeup", 11, false),
            106 => ("BlusherColorMakeup", 11, false),
            116 => ("BlusherImageMakeup", 42, true),
            _ => unreachable!(),
        };
        OwnedMakeupBundleInputRow {
            pf,
            source_effect: Some(source_effect.to_string()),
            face_id: Some(0),
            draw: draw([32, 24]),
            origin: [0, 0],
            size: [32, 24],
            owner_mask: (pf != 111).then(|| owner_mask(pf)),
            extra_owner_mask: None,
            material: has_material.then(|| material(pf)),
            material_uv: has_material.then(material_uv),
            mix_color: [37, 151, 223, 219],
            opacity_percent: 73.0,
            filter_type,
        }
    }

    fn produce_row(row: OwnedMakeupBundleInputRow) -> Result<OwnedMakeupBundle, Refusal> {
        produce_owned_makeup_bundle(Some(OwnedMakeupBundleInputs {
            job_id: 7,
            ticket: Some(7),
            frame_size: [32, 24],
            source_sha256: [0x51; 32],
            rows: &[row],
        }))
    }

    #[test]
    fn owned_missing_fields_are_named_refusals() {
        let mut missing_mask = single_row(106);
        missing_mask.owner_mask = None;
        assert_eq!(
            produce_row(missing_mask).err(),
            Some(Refusal::MissingInput(OWNER_MASK_MISSING))
        );

        let mut missing_vbo = single_row(100);
        missing_vbo.draw.vertices = Arc::from([]);
        assert_eq!(
            produce_row(missing_vbo).err(),
            Some(Refusal::MissingInput(VBO_BYTES_MISSING))
        );

        let mut missing_ebo = single_row(100);
        missing_ebo.draw.indices_u16 = Arc::from([]);
        assert_eq!(
            produce_row(missing_ebo).err(),
            Some(Refusal::MissingInput(EBO_BYTES_MISSING))
        );

        let mut missing_material = single_row(116);
        missing_material.material = None;
        assert_eq!(
            produce_row(missing_material).err(),
            Some(Refusal::MissingInput(MATERIAL_MISSING))
        );

        let mut missing_material_uv = single_row(116);
        missing_material_uv.material_uv = None;
        assert_eq!(
            produce_row(missing_material_uv).err(),
            Some(Refusal::MissingInput(MATERIAL_UV_MISSING))
        );
    }

    #[test]
    fn complete_owned_bundle_outlives_its_transient_inputs() {
        let bundle = {
            let rows = vec![single_row(100), single_row(106), single_row(116)];
            let frame_size = [32, 24];
            produce_owned_makeup_bundle(Some(OwnedMakeupBundleInputs {
                job_id: 7,
                ticket: Some(7),
                frame_size,
                source_sha256: [0x51; 32],
                rows: &rows,
            }))
            .unwrap()
        };
        let input = bundle.makeup_input(116).unwrap();
        assert!(input.mask.is_some());
        assert!(input.material.is_some());
        assert!(input.mask_uv_writes.unwrap().iter().any(|writes| *writes > 0));
        assert!(
            input
                .material_uv_writes
                .unwrap()
                .iter()
                .any(|writes| *writes > 0)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xy_bytes(points: &[[f32; 2]]) -> Vec<u8> {
        let mut vertices = Vec::new();
        for point in points {
            vertices.extend_from_slice(&point[0].to_le_bytes());
            vertices.extend_from_slice(&point[1].to_le_bytes());
        }
        vertices
    }

    fn triangle_draw() -> (Vec<u8>, Vec<u8>) {
        let vertices = xy_bytes(&[[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]]);
        let indices = [0u16, 1, 2]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        (vertices, indices)
    }

    #[test]
    fn b3_contract_and_suit_slots_are_complete() {
        assert_eq!(B3_MAKEUP_ROWS.len(), 14);
        assert_eq!(SUIT_MATERIAL_MASK_SLOTS.len(), 12);
        for (index, slot) in SUIT_MATERIAL_MASK_SLOTS.iter().enumerate() {
            assert_eq!(slot.order as usize, index + 1);
        }
        assert_eq!(
            aurora_portrait::kernels::makeup_mask_uv::color_mask_profile(100)
                .unwrap()
                .mask_channel,
            MaskChannel::RB
        );
    }

    #[test]
    fn absent_vbo_bytes_is_a_typed_refusal() {
        let mask = Image::solid(2, 2, [255; 4]);
        let row = MakeupBundleInputRow {
            pf: 100,
            source_effect: Some("NormalEyeBrowColorMakeup"),
            face_id: Some(3),
            draw: Face2dDrawBytes {
                vertices: &[],
                vertex_stride: 8,
                position_offset: 0,
                indices_u16: &[],
                draw_first_index: 0,
                draw_index_count: 0,
            },
            origin: [0, 0],
            size: [4, 4],
            owner_mask: Some(&mask),
            extra_owner_mask: None,
            material: None,
            material_uv: None,
        };
        let inputs = MakeupBundleInputs {
            job_id: 7,
            ticket: Some(7),
            frame_size: [4, 4],
            source_sha256: [1; 32],
            rows: &[row],
        };
        assert_eq!(
            produce_makeup_bundle(Some(inputs)).err(),
            Some(Refusal::MissingInput(VBO_BYTES_MISSING))
        );
    }

    #[test]
    fn color_row_emits_mask_uv_and_coverage() {
        let (vertices, indices) = triangle_draw();
        let mask = Image::solid(2, 2, [255; 4]);
        let row = MakeupBundleInputRow {
            pf: 100,
            source_effect: Some("NormalEyeBrowColorMakeup"),
            face_id: Some(3),
            draw: Face2dDrawBytes {
                vertices: &vertices,
                vertex_stride: 8,
                position_offset: 0,
                indices_u16: &indices,
                draw_first_index: 0,
                draw_index_count: 3,
            },
            origin: [0, 0],
            size: [4, 4],
            owner_mask: Some(&mask),
            extra_owner_mask: None,
            material: None,
            material_uv: None,
        };
        let inputs = MakeupBundleInputs {
            job_id: 7,
            ticket: Some(7),
            frame_size: [4, 4],
            source_sha256: [1; 32],
            rows: &[row],
        };
        let bundle = produce_makeup_bundle(Some(inputs)).unwrap();
        let produced = &bundle.rows[0];
        assert!(
            produced
                .mask_uv
                .as_ref()
                .unwrap()
                .writes()
                .iter()
                .any(|writes| *writes > 0)
        );
        let consumer = produced
            .to_makeup_input([100, 82, 70, 255], 75.0, 11)
            .unwrap();
        assert!(consumer.mask_uv_writes.is_some());
    }

    #[test]
    fn selected_effect_variant_is_required() {
        let (vertices, indices) = triangle_draw();
        let mask = Image::solid(2, 2, [255; 4]);
        let row = MakeupBundleInputRow {
            pf: 100,
            source_effect: None,
            face_id: Some(3),
            draw: Face2dDrawBytes {
                vertices: &vertices,
                vertex_stride: 8,
                position_offset: 0,
                indices_u16: &indices,
                draw_first_index: 0,
                draw_index_count: 3,
            },
            origin: [0, 0],
            size: [4, 4],
            owner_mask: Some(&mask),
            extra_owner_mask: None,
            material: None,
            material_uv: None,
        };
        let inputs = MakeupBundleInputs {
            job_id: 7,
            ticket: Some(7),
            frame_size: [4, 4],
            source_sha256: [1; 32],
            rows: &[row],
        };
        assert_eq!(
            produce_makeup_bundle(Some(inputs)).err(),
            Some(Refusal::MissingInput(SOURCE_EFFECT_MISSING))
        );
    }

    #[test]
    fn source_variants_change_contract_and_3dmm_is_refused() {
        assert!(
            !row_profile_for_effect(107, "LipColor")
                .unwrap()
                .requires_material
        );
        assert!(
            row_profile_for_effect(107, "LipStickImageMakeup")
                .unwrap()
                .requires_material
        );
        assert_eq!(
            row_profile_for_effect(115, "3DMMFaceSticker"),
            Err(Refusal::MissingInput(SOURCE_3DMM_MISSING))
        );
        assert!(
            row_profile_for_effect(115, "BlusherImageMakeup")
                .unwrap()
                .requires_material
        );
        assert_eq!(
            row_profile_for_effect(8, "LipColor"),
            Err(Refusal::ValueOutOfRange(SOURCE_VARIANT_UNSUPPORTED))
        );
    }

    #[test]
    fn material_uv_uses_its_own_nonconstant_vbo() {
        let positions = xy_bytes(&[[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]]);
        let material_uv = xy_bytes(&[[0.1, 0.2], [0.8, 0.3], [0.2, 0.9]]);
        let indices = [0u16, 1, 2]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let material = Image::solid(2, 2, [20, 30, 40, 200]);
        let row = MakeupBundleInputRow {
            pf: 111,
            source_effect: Some("EyePupilMakeup"),
            face_id: Some(3),
            draw: Face2dDrawBytes {
                vertices: &positions,
                vertex_stride: 8,
                position_offset: 0,
                indices_u16: &indices,
                draw_first_index: 0,
                draw_index_count: 3,
            },
            origin: [0, 0],
            size: [4, 4],
            owner_mask: None,
            extra_owner_mask: None,
            material: Some(&material),
            material_uv: Some(Face2dUvBytes {
                vertices: &material_uv,
                vertex_stride: 8,
                offset: 0,
            }),
        };
        let inputs = MakeupBundleInputs {
            job_id: 7,
            ticket: Some(7),
            frame_size: [4, 4],
            source_sha256: [1; 32],
            rows: &[row],
        };
        let bundle = produce_makeup_bundle(Some(inputs)).unwrap();
        let produced = bundle.rows[0].material_uv.as_ref().unwrap();
        let mut covered_values = produced
            .uv()
            .iter()
            .zip(produced.writes())
            .filter_map(|(uv, writes)| (*writes > 0).then_some(*uv))
            .collect::<Vec<_>>();
        covered_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        covered_values.dedup();
        assert!(covered_values.len() > 1);
        assert_ne!(positions, material_uv);
    }

    #[test]
    fn empty_material_uv_vbo_is_a_typed_refusal() {
        let (positions, indices) = triangle_draw();
        let material = Image::solid(2, 2, [20, 30, 40, 200]);
        let row = MakeupBundleInputRow {
            pf: 111,
            source_effect: Some("EyePupilMakeup"),
            face_id: Some(3),
            draw: Face2dDrawBytes {
                vertices: &positions,
                vertex_stride: 8,
                position_offset: 0,
                indices_u16: &indices,
                draw_first_index: 0,
                draw_index_count: 3,
            },
            origin: [0, 0],
            size: [4, 4],
            owner_mask: None,
            extra_owner_mask: None,
            material: Some(&material),
            material_uv: Some(Face2dUvBytes {
                vertices: &[],
                vertex_stride: 8,
                offset: 0,
            }),
        };
        let inputs = MakeupBundleInputs {
            job_id: 7,
            ticket: Some(7),
            frame_size: [4, 4],
            source_sha256: [1; 32],
            rows: &[row],
        };
        assert_eq!(
            produce_makeup_bundle(Some(inputs)).err(),
            Some(Refusal::MissingInput(MATERIAL_VBO_BYTES_MISSING))
        );
    }

    #[test]
    fn static_mask_asset_cannot_replace_owner_mask_bytes() {
        let (vertices, indices) = triangle_draw();
        let row = MakeupBundleInputRow {
            pf: 106,
            source_effect: Some("BlusherColorMakeup"),
            face_id: Some(3),
            draw: Face2dDrawBytes {
                vertices: &vertices,
                vertex_stride: 8,
                position_offset: 0,
                indices_u16: &indices,
                draw_first_index: 0,
                draw_index_count: 3,
            },
            origin: [0, 0],
            size: [4, 4],
            owner_mask: None,
            extra_owner_mask: None,
            material: None,
            material_uv: None,
        };
        let inputs = MakeupBundleInputs {
            job_id: 7,
            ticket: Some(7),
            frame_size: [4, 4],
            source_sha256: [1; 32],
            rows: &[row],
        };
        assert_eq!(
            produce_makeup_bundle(Some(inputs)).err(),
            Some(Refusal::MissingInput(OWNER_MASK_MISSING))
        );
    }

    #[test]
    fn face_sticker_3dmm_variant_is_not_accepted_as_a_2d_mask_row() {
        let (vertices, indices) = triangle_draw();
        let mask = Image::solid(2, 2, [255; 4]);
        let row = MakeupBundleInputRow {
            pf: 115,
            source_effect: Some("3DMMFaceSticker"),
            face_id: Some(3),
            draw: Face2dDrawBytes {
                vertices: &vertices,
                vertex_stride: 8,
                position_offset: 0,
                indices_u16: &indices,
                draw_first_index: 0,
                draw_index_count: 3,
            },
            origin: [0, 0],
            size: [4, 4],
            owner_mask: Some(&mask),
            extra_owner_mask: None,
            material: None,
            material_uv: None,
        };
        let inputs = MakeupBundleInputs {
            job_id: 7,
            ticket: Some(7),
            frame_size: [4, 4],
            source_sha256: [1; 32],
            rows: &[row],
        };
        assert_eq!(
            produce_makeup_bundle(Some(inputs)).err(),
            Some(Refusal::MissingInput(SOURCE_3DMM_MISSING))
        );
    }

    #[test]
    fn lip_color_variant_does_not_require_material() {
        let (vertices, indices) = triangle_draw();
        let mask = Image::solid(2, 2, [255; 4]);
        let row = MakeupBundleInputRow {
            pf: 107,
            source_effect: Some("LipColor"),
            face_id: Some(3),
            draw: Face2dDrawBytes {
                vertices: &vertices,
                vertex_stride: 8,
                position_offset: 0,
                indices_u16: &indices,
                draw_first_index: 0,
                draw_index_count: 3,
            },
            origin: [0, 0],
            size: [4, 4],
            owner_mask: Some(&mask),
            extra_owner_mask: None,
            material: None,
            material_uv: None,
        };
        let inputs = MakeupBundleInputs {
            job_id: 7,
            ticket: Some(7),
            frame_size: [4, 4],
            source_sha256: [1; 32],
            rows: &[row],
        };
        let bundle = produce_makeup_bundle(Some(inputs)).unwrap();
        assert!(!bundle.rows[0].profile.requires_material);
        assert!(bundle.rows[0].material.is_none());
        assert!(bundle.rows[0].material_uv.is_none());
    }
}
