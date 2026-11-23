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
use serde::Deserialize;
use std::sync::{Arc, OnceLock};

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
pub const SUIT_PRESETS_INVALID: &str = "suit presets";
pub const SUIT_PASS_TOTAL: usize = 355;
pub const SUIT_PASS_PF90033: usize = 353;
pub const SUIT_PASS_SEPARATE: usize = 2;
pub const SUIT_LIVE_JOB_INPUTS_MISSING: &str = "suit live job inputs";
pub const SUIT_LIVE_JOB_MISMATCH: &str = "suit live job";
pub const SUIT_LIVE_BINDING_MISMATCH: &str = "suit live pass binding";

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
    pub suit: OwnedSuitBundle,
    pub suit_live: Option<OwnedSuitLiveBundle>,
}

impl OwnedMakeupBundle {
    pub fn makeup_input(&self, pf: u32) -> Result<MakeupInput<'_>, Refusal> {
        self.rows
            .iter()
            .find(|row| row.profile.pf == pf)
            .ok_or(Refusal::MissingInput("makeup bundle row"))?
            .to_makeup_input()
    }

    pub fn attach_owned_suit_live(&mut self, live: OwnedSuitLiveBundle) -> Result<(), Refusal> {
        if live.job_id != self.job_id
            || live.ticket != self.ticket
            || live.frame_size != self.frame_size
            || live.source_sha256 != self.source_sha256
        {
            return Err(Refusal::ValueOutOfRange(SUIT_LIVE_JOB_MISMATCH));
        }
        self.suit_live = Some(live);
        Ok(())
    }

    pub fn suit_pass_inputs(&self) -> Result<Vec<SuitPassInput<'_>>, Refusal> {
        self.suit_live
            .as_ref()
            .ok_or(Refusal::MissingInput(SUIT_LIVE_JOB_INPUTS_MISSING))?
            .pass_inputs()
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
        hash = mix(hash, &self.suit.total_passes.to_le_bytes());
        hash = mix(hash, &self.suit.executable_passes.to_le_bytes());
        for preset in &self.suit.presets {
            hash = mix(hash, preset.id.as_bytes());
            for pass in &preset.passes {
                hash = mix(hash, &pass.order.to_le_bytes());
                hash = mix(hash, &pass.pf.to_le_bytes());
                hash = mix(hash, pass.effect.as_bytes());
                for material in &pass.material_paths {
                    hash = mix(hash, material.as_bytes());
                }
            }
        }
        for pass in &self.suit.separate_passes {
            hash = mix(hash, &pass.order.to_le_bytes());
            hash = mix(hash, &pass.pf.to_le_bytes());
        }
        if let Some(live) = &self.suit_live {
            hash = mix(hash, &live.live_executable_passes.to_le_bytes());
            hash = mix(hash, &live.refusals.len().to_le_bytes());
            for pass in &live.passes {
                hash = mix(hash, pass.preset_id.as_bytes());
                hash = mix(hash, &pass.order.to_le_bytes());
                hash = mix(hash, &pass.shelter.to_bits().to_le_bytes());
                hash = mix(hash, &pass.region.to_bits().to_le_bytes());
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
    let suit = load_owned_suit_bundle()?;

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
        suit,
        suit_live: None,
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

#[derive(Clone, Debug, PartialEq)]
pub struct SuitPassBindingContract {
    pub face2d_vbo: &'static str,
    pub face2d_ebo: &'static str,
    pub owner_mask: &'static str,
    pub mask_uv: &'static str,
    pub material_paths: Vec<String>,
    pub material_uv: &'static str,
    pub shelter: &'static str,
    pub region: &'static str,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OwnedSuitPass {
    pub preset_id: String,
    pub order: u32,
    pub pf: u32,
    pub group: String,
    pub effect: String,
    pub degree: f32,
    pub opacity: Option<f32>,
    pub filter_type: Option<u32>,
    pub mask_field: Option<String>,
    pub mask_value: Option<u32>,
    pub mix_color: [u8; 4],
    pub shelter_type: Option<i32>,
    pub force_use_shelter: bool,
    pub region_rect: Option<[i32; 4]>,
    pub material_paths: Vec<String>,
}

impl OwnedSuitPass {
    pub fn binding_contract(&self) -> SuitPassBindingContract {
        SuitPassBindingContract {
            face2d_vbo: "same-job Face2D VBO",
            face2d_ebo: "same-job Face2D EBO",
            owner_mask: "same-job owner-mask RGBA",
            mask_uv: "same-job mask UV",
            material_paths: self.material_paths.clone(),
            material_uv: "same-job material UV",
            shelter: "live shelter weight",
            region: "live region weight",
        }
    }

    /// Static preset fields gate execution order and asset identity. The
    /// `shelter_type` and `region_rect` fields are selectors, never weights.
    pub fn require_executable(&self) -> Result<(), Refusal> {
        if self.pf != 90033 {
            return Err(Refusal::ValueOutOfRange("suit pass pf"));
        }
        if self.region_rect.is_none() {
            return Err(Refusal::MissingInput("suit region"));
        }
        if self.shelter_type.is_none() {
            return Err(Refusal::MissingInput("suit shelter"));
        }
        if self.opacity.is_none() {
            return Err(Refusal::MissingInput("suit opacity"));
        }
        if self.filter_type.is_none_or(|filter| filter == 0) {
            return Err(Refusal::MissingInput("suit filter type"));
        }
        if self.mask_field.is_none() || self.mask_value.is_none() {
            return Err(Refusal::MissingInput("suit mask"));
        }
        if self.material_paths.is_empty() {
            return Err(Refusal::MissingInput("suit material"));
        }
        if !matches!(
            self.effect.as_str(),
            "BlusherImageMakeup"
                | "EyePupilMakeup"
                | "LipColor"
                | "LipStickImageMakeup"
                | "NormalEyeBrowImageMakeup"
                | "NormalEyeImageMakeup"
                | "NormalEyeLashImageMakeup"
                | "SilkWorm"
                | "SymEyeBrowImageMakeup"
                | "SymEyeImageMakeup"
                | "SymEyeLashColorMakeup"
                | "SymEyeLashImageMakeup"
        ) {
            return Err(Refusal::NotImplemented("suit pass effect"));
        }
        Ok(())
    }

    pub fn is_executable(&self) -> bool {
        self.require_executable().is_ok()
    }

    pub fn require_live_weights(
        &self,
        weights: Option<SuitLiveWeights>,
    ) -> Result<SuitLiveWeights, Refusal> {
        self.require_executable()?;
        let weights = weights.ok_or(Refusal::MissingInput("suit live weights"))?;
        if !weights.shelter.is_finite() || !(0.0..=1.0).contains(&weights.shelter) {
            return Err(Refusal::ValueOutOfRange("suit shelter"));
        }
        if !weights.region.is_finite() || !(0.0..=1.0).contains(&weights.region) {
            return Err(Refusal::ValueOutOfRange("suit region"));
        }
        Ok(weights)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SuitLiveWeights {
    pub shelter: f32,
    pub region: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OwnedSuitPreset {
    pub id: String,
    pub passes: Vec<OwnedSuitPass>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OwnedSuitBundle {
    pub presets: Vec<OwnedSuitPreset>,
    pub separate_passes: Vec<OwnedSuitPass>,
    pub total_passes: usize,
    pub executable_passes: usize,
}

#[derive(Deserialize)]
struct SuitPresetDocument {
    presets: Vec<SuitPresetSource>,
}

#[derive(Deserialize)]
struct SuitPresetSource {
    id: String,
    passes: Vec<SuitPassSource>,
}

#[derive(Deserialize)]
struct SuitPassSource {
    order: u32,
    pf: u32,
    group: String,
    effect: String,
    degree: f32,
    #[serde(default)]
    opacity: Option<f32>,
    #[serde(default)]
    filter_type: Option<u32>,
    #[serde(default)]
    mask: Option<SuitMaskSource>,
    #[serde(default)]
    mix_color: Option<[u8; 4]>,
    #[serde(default)]
    shelter_type: Option<i32>,
    #[serde(default)]
    force_use_shelter: Option<bool>,
    #[serde(default)]
    region_rect: Option<[i32; 4]>,
    #[serde(default)]
    material_paths: Vec<String>,
}

#[derive(Deserialize)]
struct SuitMaskSource {
    field: String,
    value: u32,
}

const SUIT_PRESETS_JSON: &str =
    include_str!("../../../assets/portrait/suit/presets.json");
static SUIT_BUNDLE: OnceLock<Result<OwnedSuitBundle, Refusal>> = OnceLock::new();

fn parse_owned_suit_bundle() -> Result<OwnedSuitBundle, Refusal> {
    let document: SuitPresetDocument = serde_json::from_str(SUIT_PRESETS_JSON)
        .map_err(|_| Refusal::ShapeMismatch(SUIT_PRESETS_INVALID))?;
    let mut presets = Vec::with_capacity(document.presets.len());
    let mut separate_passes = Vec::new();
    let mut total_passes = 0usize;
    let mut pf90033_passes = 0usize;
    let mut executable_passes = 0usize;

    for preset in document.presets {
        let mut passes = Vec::with_capacity(preset.passes.len());
        for source in preset.passes {
            let pass = OwnedSuitPass {
                preset_id: preset.id.clone(),
                order: source.order,
                pf: source.pf,
                group: source.group,
                effect: source.effect,
                degree: source.degree,
                opacity: source.opacity,
                filter_type: source.filter_type,
                mask_field: source.mask.as_ref().map(|mask| mask.field.clone()),
                mask_value: source.mask.as_ref().map(|mask| mask.value),
                mix_color: source.mix_color.unwrap_or([0, 0, 0, 255]),
                shelter_type: source.shelter_type,
                force_use_shelter: source.force_use_shelter.unwrap_or(false),
                region_rect: source.region_rect,
                material_paths: source.material_paths,
            };
            if pass.pf == 90033 {
                pf90033_passes += 1;
                if pass.is_executable() {
                    executable_passes += 1;
                }
            } else {
                separate_passes.push(pass.clone());
            }
            passes.push(pass);
        }
        passes.sort_by_key(|pass| pass.order);
        if passes
            .iter()
            .enumerate()
            .any(|(index, pass)| pass.order != index as u32 + 1)
        {
            return Err(Refusal::ShapeMismatch(SUIT_PRESETS_INVALID));
        }
        total_passes += passes.len();
        presets.push(OwnedSuitPreset {
            id: preset.id,
            passes,
        });
    }

    if presets.len() != 24
        || total_passes != SUIT_PASS_TOTAL
        || pf90033_passes != SUIT_PASS_PF90033
        || separate_passes.len() != SUIT_PASS_SEPARATE
    {
        return Err(Refusal::ShapeMismatch(SUIT_PRESETS_INVALID));
    }

    Ok(OwnedSuitBundle {
        presets,
        separate_passes,
        total_passes,
        executable_passes,
    })
}

pub fn load_owned_suit_bundle() -> Result<OwnedSuitBundle, Refusal> {
    SUIT_BUNDLE
        .get_or_init(parse_owned_suit_bundle)
        .clone()
}

/// One PF90033 pass's live payload. The VBO, EBO, mask, material and two UV
/// streams are copied from the same renderer job as the output frame.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedSuitPassJobInput {
    pub preset_id: String,
    pub order: u32,
    pub pf: u32,
    pub job_id: u64,
    pub ticket: Option<u64>,
    pub face_id: Option<u32>,
    pub draw: OwnedFace2dDrawBytes,
    pub origin: [i32; 2],
    pub size: [i32; 2],
    pub owner_mask: Option<Arc<Image>>,
    pub material: Option<Arc<Image>>,
    pub material_uv: Option<OwnedFace2dUvBytes>,
    pub shelter: Option<f32>,
    pub region: Option<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OwnedSuitPassRefusal {
    pub preset_id: String,
    pub order: u32,
    pub effect: String,
    pub cause: Refusal,
}

impl OwnedSuitPassRefusal {
    pub fn label(&self) -> String {
        format!("{}#{} {}: {:?}", self.preset_id, self.order, self.effect, self.cause)
    }
}

/// Fully owned live PF90033 pass. `draw` retains the exact same-job VBO/EBO,
/// while the two rasterized UV streams are ready for CPU or CUDA consumption.
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedSuitLivePass {
    pub preset_id: String,
    pub order: u32,
    pub pf: u32,
    pub effect: String,
    pub face_id: u32,
    pub draw: OwnedFace2dDrawBytes,
    pub owner_mask: Arc<Image>,
    pub mask_uv: MeshMaskPixelUv,
    pub material: Arc<Image>,
    pub material_uv: MeshMaskPixelUv,
    pub layer: aurora_portrait::kernels::makeup_material::MakeupLayer,
    pub shelter: f32,
    pub region: f32,
}

impl OwnedSuitLivePass {
    pub fn as_suit_pass_input(&self) -> SuitPassInput<'_> {
        SuitPassInput {
            kind: SuitPassKindInput::Image {
                material: Some(self.material.as_ref()),
                material_uv: Some(self.material_uv.uv()),
            },
            mask: Some(self.owner_mask.as_ref()),
            mask_uv: Some(self.mask_uv.uv()),
            layer: Some(self.layer),
            shelter: self.shelter,
            region: self.region,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OwnedSuitLiveBundle {
    pub job_id: u64,
    pub ticket: u64,
    pub frame_size: [u32; 2],
    pub source_sha256: [u8; 32],
    pub static_executable_passes: usize,
    pub live_executable_passes: usize,
    pub passes: Vec<OwnedSuitLivePass>,
    pub refusals: Vec<OwnedSuitPassRefusal>,
}

impl OwnedSuitLiveBundle {
    /// All-or-nothing adapter for the existing stage/CUDA path. Per-pass
    /// refusals remain inspectable on the live bundle instead of being skipped.
    pub fn pass_inputs(&self) -> Result<Vec<SuitPassInput<'_>>, Refusal> {
        if !self.refusals.is_empty()
            || self.live_executable_passes != self.static_executable_passes
        {
            return Err(Refusal::MissingInput(SUIT_LIVE_JOB_INPUTS_MISSING));
        }
        Ok(self
            .passes
            .iter()
            .map(OwnedSuitLivePass::as_suit_pass_input)
            .collect())
    }
}

fn suit_live_weight(value: Option<f32>, name: &'static str) -> Result<f32, Refusal> {
    let value = value.ok_or(Refusal::MissingInput(name))?;
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(Refusal::ValueOutOfRange(name));
    }
    Ok(value)
}

fn suit_live_mask_channel(field: &str, value: u32) -> Result<MaskChannel, Refusal> {
    match (field, value) {
        ("BlusherMaskType", 15) => Ok(MaskChannel::Rgba),
        ("EyeBrowMaskType", 5) => Ok(MaskChannel::RB),
        ("EyeMaskType", 10) => Ok(MaskChannel::Rgb),
        ("EyeMaskType", 4) | ("EyeLashMaskType", 4) => Ok(MaskChannel::RG),
        ("LipMaskType", 3) => Ok(MaskChannel::R),
        _ => Err(Refusal::MissingInput("suit mask channel")),
    }
}

fn produce_live_suit_pass(
    job_id: u64,
    frame_size: [u32; 2],
    pass: &OwnedSuitPass,
    input: &OwnedSuitPassJobInput,
) -> Result<OwnedSuitLivePass, Refusal> {
    if input.job_id != job_id || input.ticket != Some(job_id) {
        return Err(Refusal::ValueOutOfRange(SUIT_LIVE_JOB_MISMATCH));
    }
    if input.preset_id != pass.preset_id || input.order != pass.order || input.pf != pass.pf {
        return Err(Refusal::ValueOutOfRange(SUIT_LIVE_BINDING_MISMATCH));
    }
    let face_id = input
        .face_id
        .ok_or(Refusal::MissingInput("suit mask uv face"))?;
    if input.draw.vertices.is_empty() {
        return Err(Refusal::MissingInput("suit face2d vbo"));
    }
    if input.draw.indices_u16.is_empty() {
        return Err(Refusal::MissingInput("suit face2d ebo"));
    }
    let positions = input.draw.borrowed().decode_positions()?;
    let indices = input.draw.borrowed().decode_indices()?;
    let owner_mask = input
        .owner_mask
        .as_deref()
        .ok_or(Refusal::MissingInput(SUIT_MASK_MISSING))?;
    validate_image(owner_mask, SUIT_MASK_MISSING)?;
    let material = input
        .material
        .as_deref()
        .ok_or(Refusal::MissingInput(SUIT_MATERIAL_MISSING))?;
    validate_image(material, SUIT_MATERIAL_MISSING)?;
    if pass.material_paths.is_empty() {
        return Err(Refusal::MissingInput(SUIT_MATERIAL_MISSING));
    }
    let material_values = input
        .material_uv
        .as_ref()
        .ok_or(Refusal::MissingInput(SUIT_MATERIAL_UV_MISSING))?
        .decode()
        .map_err(|error| match error {
            Refusal::MissingInput(MATERIAL_VBO_BYTES_MISSING)
            | Refusal::MissingInput(MATERIAL_UV_MISSING) => {
                Refusal::MissingInput(SUIT_MATERIAL_UV_MISSING)
            }
            other => other,
        })?;

    let mask_uv = produce_face2d_mask_uv(Some(Face2dMaskUvInputs {
        ticket: Some(job_id),
        face_id: Some(face_id),
        mask_source_points: &positions,
        indices: &indices,
        origin: input.origin,
        size: input.size,
        width: frame_size[0],
        height: frame_size[1],
    }))?;
    let material_uv = produce_mesh_mask_uv(Some(MeshMaskUvInputs {
        ticket: Some(job_id),
        face_id: Some(face_id),
        positions: &positions,
        mask_uv: &material_values,
        indices: &indices,
        width: frame_size[0],
        height: frame_size[1],
    }))
    .map_err(|error| match error {
        Refusal::ShapeMismatch(_) | Refusal::ValueOutOfRange(_) => {
            Refusal::ShapeMismatch(SUIT_MATERIAL_UV_MISSING)
        }
        other => other,
    })?;

    let (mask_field, mask_value) = match (pass.mask_field.as_deref(), pass.mask_value) {
        (Some(field), Some(value)) => (field, value),
        _ => return Err(Refusal::MissingInput("suit mask")),
    };
    let degree = pass.degree;
    if !degree.is_finite() || !(0.0..=100.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("suit degree"));
    }
    let opacity = pass
        .opacity
        .ok_or(Refusal::MissingInput("suit opacity"))?;
    if !opacity.is_finite() || !(0.0..=100.0).contains(&opacity) {
        return Err(Refusal::ValueOutOfRange("suit opacity"));
    }
    let filter_type = pass
        .filter_type
        .filter(|value| *value != 0)
        .ok_or(Refusal::MissingInput("suit filter type"))?;
    let shelter = if pass.force_use_shelter {
        1.0
    } else {
        suit_live_weight(input.shelter, "suit shelter")?
    };
    let region = suit_live_weight(input.region, "suit region")?;

    Ok(OwnedSuitLivePass {
        preset_id: pass.preset_id.clone(),
        order: pass.order,
        pf: pass.pf,
        effect: pass.effect.clone(),
        face_id,
        draw: input.draw.clone(),
        owner_mask: Arc::clone(input.owner_mask.as_ref().expect("validated owner mask")),
        mask_uv,
        material: Arc::clone(input.material.as_ref().expect("validated material")),
        material_uv,
        layer: aurora_portrait::kernels::makeup_material::MakeupLayer {
            mix_color: pass.mix_color,
            opacity_percent: opacity,
            degree01: degree / 100.0,
            filter_type,
            mask_channel: suit_live_mask_channel(mask_field, mask_value)?,
        },
        shelter,
        region,
    })
}

/// Builds every statically executable PF90033 pass or records its exact
/// refusal. The two separate PF107/PF112 rows are intentionally not relabelled
/// as PF90033. Missing live bytes are never replaced with synthetic defaults.
pub fn produce_owned_suit_live_bundle(
    job_id: u64,
    ticket: Option<u64>,
    frame_size: [u32; 2],
    source_sha256: [u8; 32],
    inputs: Option<&[OwnedSuitPassJobInput]>,
) -> Result<OwnedSuitLiveBundle, Refusal> {
    if job_id == 0 || ticket != Some(job_id) {
        return Err(Refusal::ValueOutOfRange(SUIT_LIVE_JOB_MISMATCH));
    }
    if frame_size[0] == 0 || frame_size[1] == 0 {
        return Err(Refusal::ShapeMismatch("suit live frame"));
    }
    if source_sha256.iter().all(|byte| *byte == 0) {
        return Err(Refusal::MissingInput(SOURCE_SHA_MISSING));
    }
    let inputs = inputs.unwrap_or(&[]);
    let static_bundle = load_owned_suit_bundle()?;
    for input in inputs {
        let known = static_bundle.presets.iter().any(|preset| {
            preset.id == input.preset_id
                && preset
                    .passes
                    .iter()
                    .any(|pass| pass.order == input.order && pass.pf == input.pf)
        });
        if !known {
            return Err(Refusal::ValueOutOfRange(SUIT_LIVE_BINDING_MISMATCH));
        }
    }

    let mut passes = Vec::new();
    let mut refusals = Vec::new();
    let mut static_executable_passes = 0usize;
    for preset in &static_bundle.presets {
        for pass in preset.passes.iter().filter(|pass| pass.pf == 90033) {
            if let Err(cause) = pass.require_executable() {
                refusals.push(OwnedSuitPassRefusal {
                    preset_id: pass.preset_id.clone(),
                    order: pass.order,
                    effect: pass.effect.clone(),
                    cause,
                });
                continue;
            }
            static_executable_passes += 1;
            let input = inputs
                .iter()
                .find(|input| {
                    input.preset_id == pass.preset_id
                        && input.order == pass.order
                        && input.pf == pass.pf
                })
                .ok_or(Refusal::MissingInput(SUIT_LIVE_JOB_INPUTS_MISSING));
            match input.and_then(|input| produce_live_suit_pass(job_id, frame_size, pass, input)) {
                Ok(live) => passes.push(live),
                Err(cause) => refusals.push(OwnedSuitPassRefusal {
                    preset_id: pass.preset_id.clone(),
                    order: pass.order,
                    effect: pass.effect.clone(),
                    cause,
                }),
            }
        }
    }

    Ok(OwnedSuitLiveBundle {
        job_id,
        ticket: job_id,
        frame_size,
        source_sha256,
        static_executable_passes,
        live_executable_passes: passes.len(),
        passes,
        refusals,
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
    fn suit_presets_keep_order_and_name_every_static_gap() {
        let suit = load_owned_suit_bundle().unwrap();
        assert_eq!(suit.presets.len(), 24);
        assert_eq!(suit.total_passes, 355);
        // D2 claimed 338, but the checked-in presets contain sixteen
        // region/shelter gaps among 353 PF90033 passes.
        assert_eq!(suit.executable_passes, 337);
        assert_eq!(suit.separate_passes.len(), 2);
        let mut separate_pf: Vec<_> = suit
            .separate_passes
            .iter()
            .map(|pass| pass.pf)
            .collect();
        separate_pf.sort();
        assert_eq!(separate_pf, [107, 112]);
        assert_eq!(
            suit.presets
                .iter()
                .map(|preset| preset.passes.len())
                .sum::<usize>(),
            355
        );
        for preset in &suit.presets {
            for (index, pass) in preset.passes.iter().enumerate() {
                assert_eq!(pass.order, index as u32 + 1);
                assert_eq!(pass.binding_contract().face2d_vbo, "same-job Face2D VBO");
                assert_eq!(pass.binding_contract().face2d_ebo, "same-job Face2D EBO");
            }
        }

        let shelter_gap = suit
            .presets
            .iter()
            .flat_map(|preset| &preset.passes)
            .find(|pass| pass.pf == 90033 && pass.effect == "SymEyeBrowImageMakeup")
            .unwrap();
        assert_eq!(
            shelter_gap.require_executable(),
            Err(Refusal::MissingInput("suit shelter"))
        );
        let region_gap = suit
            .presets
            .iter()
            .flat_map(|preset| &preset.passes)
            .find(|pass| pass.pf == 90033 && pass.effect == "LipColor")
            .unwrap();
        assert_eq!(
            region_gap.require_executable(),
            Err(Refusal::MissingInput("suit region"))
        );

        let executable = suit
            .presets
            .iter()
            .flat_map(|preset| &preset.passes)
            .find(|pass| pass.is_executable())
            .unwrap();
        assert_eq!(
            executable.require_live_weights(None),
            Err(Refusal::MissingInput("suit live weights"))
        );
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
