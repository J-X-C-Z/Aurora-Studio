//! Same-job capture and BFM mapping gate for the PF90033 suit rows.
//!
//! Static suit presets and materials are useful provenance, but they do not
//! contain a Face2D draw, owner-mask pixels, draw-time UVs, or live
//! shelter/region weights. This module binds captured values to the D3 bundle
//! and reports the first exact missing field for every statically executable
//! pass. It never substitutes a template or synthetic topology for a capture.

use std::fmt;

use aurora_portrait::kernels::makeup_material::MaskChannel;
use aurora_portrait::model_sha256::sha256 as sha256_32;
use aurora_portrait::Refusal;

use crate::face_parsing::FACE_PARSING_V2_MODEL_SHA256;
use crate::portrait_makeup_bundle::{
    load_owned_suit_bundle, produce_owned_suit_live_bundle, OwnedSuitLiveBundle,
    OwnedSuitPassJobInput, SOURCE_SHA_MISSING, SUIT_LIVE_BINDING_MISMATCH,
    SUIT_LIVE_JOB_INPUTS_MISSING, SUIT_LIVE_JOB_MISMATCH, SUIT_MASK_MISSING, SUIT_MATERIAL_MISSING,
    SUIT_MATERIAL_UV_MISSING,
};

pub const D4_CAPTURE_OWNER_MASK: &str = "suit same-job owner-mask capture";
pub const D4_CAPTURE_MASK_UV: &str = "suit same-job mask UV capture";
pub const D4_CAPTURE_MATERIAL_UV: &str = "suit same-job material UV capture";
pub const D4_CAPTURE_FACE2D_VBO: &str = "suit same-job Face2D VBO capture";
pub const D4_CAPTURE_FACE2D_EBO: &str = "suit same-job Face2D EBO capture";
pub const SUIT_BFM_OBJ_SHA256: [u8; 32] = [
    0x49, 0xbc, 0xd4, 0x14, 0x8f, 0xcb, 0x38, 0x36, 0x1d, 0xcf, 0xca, 0x1d, 0xfb, 0x91, 0x4a, 0xc7,
    0x71, 0xa4, 0xae, 0x37, 0x52, 0x39, 0xb8, 0x24, 0xf9, 0xa9, 0xee, 0xef, 0x32, 0x5b, 0xbf, 0x3a,
];
pub const SUIT_BFM_VERTEX_COUNT: usize = 9_794;
pub const SUIT_BFM_TEXCOORD_COUNT: usize = 9_794;
pub const SUIT_BFM_FACE_COUNT: usize = 19_512;
pub const SUIT_BFM_LANDMARK_COUNT: usize = 414;
pub const SUIT_BFM_LOCAL_BINDING_MISSING: &str = "suit BFM same-job suit-local UV/weights";
pub const SUIT_BFM_OBJ_INVALID: &str = "suit BFM OBJ";
pub const SUIT_BFM_OBJ_SHA_MISMATCH: &str = "suit BFM OBJ SHA-256";
pub const SUIT_BFM_PARSING_PIXELS: &str = "suit BFM FaceParsingV2 pixels";
pub const SUIT_BFM_LANDMARKS_MISSING: &str = "suit BFM face landmarks";
pub const SUIT_BFM_LANDMARK_VERTEX_MAP_MISSING: &str = "suit BFM landmark-to-vertex map";
pub const SUIT_BFM_LANDMARK_VERTEX_MAP_INVALID: &str = "suit BFM landmark-to-vertex candidate map";
pub const SUIT_BFM_LANDMARK_UV_INVALID: &str = "suit BFM landmark vertex UV";
pub const SUIT_BFM_LANDMARK_REGION_MISMATCH: &str = "suit BFM landmark FaceParsingV2 region";
pub const SUIT_BFM_SUIT_LOCAL_UV_MISSING: &str = "suit BFM suit-local material UV";
pub const SUIT_BFM_OWNER_MASK_CLASS_MAP_MISSING: &str = "suit owner-mask 17-class channel map";
pub const SUIT_FACEPARSING_R_CLASSES: &[u8] = &[0, 13, 14, 15, 16];
pub const SUIT_FACEPARSING_G_CLASSES: &[u8] = &[2, 7, 11, 12];
pub const SUIT_FACEPARSING_B_CLASSES: &[u8] = &[4, 5];
pub const SUIT_FACEPARSING_A_CLASSES: &[u8] = &[9, 10];
pub const SUIT_FACEPARSING_CLASS_1_MISSING: &str =
    "suit owner-mask FaceParsingV2 class 1 channel";
pub const SUIT_FACEPARSING_CLASS_3_MISSING: &str =
    "suit owner-mask FaceParsingV2 class 3 channel";
pub const SUIT_FACEPARSING_CLASS_6_MISSING: &str =
    "suit owner-mask FaceParsingV2 class 6 channel";
pub const SUIT_FACEPARSING_CLASS_8_MISSING: &str =
    "suit owner-mask FaceParsingV2 class 8 channel";
pub const SUIT_BFM_SHELTER_WEIGHT_MISSING: &str = "suit BFM shelter weight";
pub const SUIT_BFM_REGION_WEIGHT_MISSING: &str = "suit BFM region weight";

/// The recovered consumer RGBA is a saturating bundle, not a dense class
/// remap. Classes 1, 3, 6 and 8 have no evidenced owner-mask channel.
pub fn suit_owner_mask_class_channel(class: u8) -> Result<MaskChannel, &'static str> {
    match class {
        0 | 13..=16 => Ok(MaskChannel::R),
        2 | 7 | 11 | 12 => Ok(MaskChannel::G),
        4 | 5 => Ok(MaskChannel::B),
        9 | 10 => Ok(MaskChannel::A),
        1 => Err(SUIT_FACEPARSING_CLASS_1_MISSING),
        3 => Err(SUIT_FACEPARSING_CLASS_3_MISSING),
        6 => Err(SUIT_FACEPARSING_CLASS_6_MISSING),
        8 => Err(SUIT_FACEPARSING_CLASS_8_MISSING),
        _ => Err(SUIT_BFM_OWNER_MASK_CLASS_MAP_MISSING),
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SuitBfmTopology {
    pub positions: Vec<[f32; 3]>,
    pub texcoords: Vec<[f32; 2]>,
    pub indices_u16: Vec<u16>,
    pub texcoord_indices_u16: Vec<u16>,
    pub vertex_uv_identity: bool,
    pub all_vertices_referenced: bool,
    pub all_texcoords_referenced: bool,
    pub has_material_directive: bool,
    pub has_group_directive: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SuitBfmSameJobReport {
    pub bfm: SuitBfmTopology,
    pub parsing_pixels: usize,
    pub parsing_classes: usize,
    pub parsing_class_ids_valid: bool,
    pub landmarks: usize,
    pub blocking_inputs: Vec<&'static str>,
    pub same_job: SuitSameJobReport,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuitBfmLandmarkCandidate {
    pub landmark: u16,
    pub vertex: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SuitBfmLandmarkMapReport {
    pub supplied_landmarks: usize,
    pub finite_landmarks: usize,
    pub nonfinite_landmarks: Vec<usize>,
    pub supplied_candidates: usize,
    pub accepted_candidates: usize,
    pub uv_valid: bool,
    pub regions_valid: bool,
    pub blockers: Vec<&'static str>,
    pub mapping_proven: bool,
}

fn parse_obj_index(value: &str) -> Result<u16, Refusal> {
    let value = value
        .parse::<usize>()
        .map_err(|_| Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID))?;
    u16::try_from(
        value
            .checked_sub(1)
            .ok_or(Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID))?,
    )
    .map_err(|_| Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID))
}

fn parse_obj_vertex_uv(token: &str) -> Result<(u16, u16), Refusal> {
    let mut fields = token.split('/');
    let vertex = fields
        .next()
        .filter(|field| !field.is_empty())
        .ok_or(Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID))?;
    let texcoord = fields
        .next()
        .filter(|field| !field.is_empty())
        .ok_or(Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID))?;
    Ok((parse_obj_index(vertex)?, parse_obj_index(texcoord)?))
}

fn parse_bfm_obj(
    bytes: &[u8],
    expected_vertices: usize,
    expected_texcoords: usize,
    expected_faces: usize,
) -> Result<SuitBfmTopology, Refusal> {
    let text =
        std::str::from_utf8(bytes).map_err(|_| Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID))?;
    let mut positions = Vec::new();
    let mut texcoords = Vec::new();
    let mut indices_u16 = Vec::new();
    let mut texcoord_indices_u16 = Vec::new();
    let mut has_material_directive = false;
    let mut has_group_directive = false;

    for line in text.lines() {
        let mut fields = line.split_whitespace();
        match fields.next() {
            Some("v") => {
                let values = [fields.next(), fields.next(), fields.next()];
                let [Some(x), Some(y), Some(z)] = values else {
                    return Err(Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID));
                };
                let point = [
                    x.parse::<f32>()
                        .map_err(|_| Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID))?,
                    y.parse::<f32>()
                        .map_err(|_| Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID))?,
                    z.parse::<f32>()
                        .map_err(|_| Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID))?,
                ];
                if point.iter().any(|value| !value.is_finite()) {
                    return Err(Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID));
                }
                positions.push(point);
            }
            Some("vt") => {
                let values = [fields.next(), fields.next()];
                let [Some(u), Some(v)] = values else {
                    return Err(Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID));
                };
                let uv = [
                    u.parse::<f32>()
                        .map_err(|_| Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID))?,
                    v.parse::<f32>()
                        .map_err(|_| Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID))?,
                ];
                if uv.iter().any(|value| !value.is_finite()) {
                    return Err(Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID));
                }
                texcoords.push(uv);
            }
            Some("f") => {
                let face: Vec<&str> = fields.collect();
                if face.len() != 3 {
                    return Err(Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID));
                }
                for corner in face {
                    let (vertex, texcoord) = parse_obj_vertex_uv(corner)?;
                    indices_u16.push(vertex);
                    texcoord_indices_u16.push(texcoord);
                }
            }
            Some("mtllib") | Some("usemtl") => has_material_directive = true,
            Some("g") | Some("o") => has_group_directive = true,
            _ => {}
        }
    }

    if positions.len() != expected_vertices
        || texcoords.len() != expected_texcoords
        || indices_u16.len() != expected_faces * 3
        || texcoords.len() != positions.len()
        || indices_u16
            .iter()
            .any(|index| usize::from(*index) >= positions.len())
        || texcoord_indices_u16
            .iter()
            .any(|index| usize::from(*index) >= texcoords.len())
    {
        return Err(Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID));
    }

    let mut position_referenced = vec![false; positions.len()];
    let mut texcoord_referenced = vec![false; texcoords.len()];
    let mut vertex_uv_identity = true;
    for (vertex, texcoord) in indices_u16.iter().zip(&texcoord_indices_u16) {
        position_referenced[usize::from(*vertex)] = true;
        texcoord_referenced[usize::from(*texcoord)] = true;
        vertex_uv_identity &= vertex == texcoord;
    }

    Ok(SuitBfmTopology {
        positions,
        texcoords,
        indices_u16,
        texcoord_indices_u16,
        vertex_uv_identity,
        all_vertices_referenced: position_referenced.into_iter().all(|referenced| referenced),
        all_texcoords_referenced: texcoord_referenced.into_iter().all(|referenced| referenced),
        has_material_directive,
        has_group_directive,
    })
}

pub fn parse_suit_bfm_obj(bytes: &[u8]) -> Result<SuitBfmTopology, Refusal> {
    parse_bfm_obj(
        bytes,
        SUIT_BFM_VERTEX_COUNT,
        SUIT_BFM_TEXCOORD_COUNT,
        SUIT_BFM_FACE_COUNT,
    )
}

fn bfm_blocked_report() -> Result<SuitSameJobReport, Refusal> {
    let bundle = load_owned_suit_bundle()?;
    let mut refusals = Vec::with_capacity(bundle.executable_passes);
    for pass in bundle
        .presets
        .iter()
        .flat_map(|preset| &preset.passes)
        .filter(|pass| pass.pf == 90033 && pass.is_executable())
    {
        refusals.push(SuitSameJobRefusal {
            preset_id: pass.preset_id.clone(),
            order: pass.order,
            effect: pass.effect.clone(),
            field: SUIT_BFM_LOCAL_BINDING_MISSING,
            cause: Refusal::MissingInput(SUIT_BFM_LOCAL_BINDING_MISSING),
        });
    }
    Ok(SuitSameJobReport {
        static_executable_passes: bundle.executable_passes,
        supplied_inputs: 0,
        live: None,
        refusals,
    })
}

fn bfm_mapping_blockers(_bfm: &SuitBfmTopology) -> Vec<&'static str> {
    vec![
        SUIT_BFM_LANDMARK_VERTEX_MAP_MISSING,
        SUIT_BFM_SUIT_LOCAL_UV_MISSING,
        SUIT_BFM_OWNER_MASK_CLASS_MAP_MISSING,
        SUIT_BFM_SHELTER_WEIGHT_MISSING,
        SUIT_BFM_REGION_WEIGHT_MISSING,
    ]
}

/// Audits a proposed 414 -> BFM correspondence without inventing one.
///
/// Every accepted edge needs an in-range unique landmark/vertex pair, finite
/// normalized UV, and the same canonical FaceParsingV2 class on both sides.
/// `landmark_classes` must come from the same-job 384x384 parsing field after
/// the source-to-crop transform; `vertex_classes` must describe the pinned BFM.
pub fn audit_suit_bfm_landmark_map(
    job_id: u64,
    ticket: Option<u64>,
    source_sha256: [u8; 32],
    parsing_model_sha256: [u8; 32],
    parsing_pixels: &[u8],
    landmarks: &[[f32; 2]],
    bfm: &SuitBfmTopology,
    candidates: Option<&[SuitBfmLandmarkCandidate]>,
    landmark_classes: Option<&[u8]>,
    vertex_classes: Option<&[u8]>,
) -> SuitBfmLandmarkMapReport {
    let nonfinite_landmarks = landmarks
        .iter()
        .enumerate()
        .filter_map(|(index, point)| {
            point
                .iter()
                .any(|coordinate| !coordinate.is_finite())
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let finite_landmarks = landmarks.len() - nonfinite_landmarks.len();
    let candidates = candidates.unwrap_or(&[]);
    let mut blockers = Vec::new();

    if job_id == 0 || ticket != Some(job_id) {
        blockers.push(SUIT_LIVE_JOB_MISMATCH);
    }
    if source_sha256.iter().all(|byte| *byte == 0) {
        blockers.push(SOURCE_SHA_MISSING);
    }
    if parsing_model_sha256 != FACE_PARSING_V2_MODEL_SHA256 {
        blockers.push("suit FaceParsingV2 model SHA-256");
    }
    let parsing_shape_valid = parsing_pixels.len()
        == crate::face_parsing::FACE_PARSING_V2_SIDE as usize
            * crate::face_parsing::FACE_PARSING_V2_SIDE as usize;
    let parsing_class_ids_valid = parsing_shape_valid
        && parsing_pixels
            .iter()
            .all(|class| usize::from(*class) < crate::face_parsing::FACE_PARSING_V2_CLASSES);
    if !parsing_shape_valid {
        blockers.push(SUIT_BFM_PARSING_PIXELS);
    } else if !parsing_class_ids_valid {
        blockers.push("suit FaceParsingV2 17-class labels");
    }
    if landmarks.len() != SUIT_BFM_LANDMARK_COUNT || !nonfinite_landmarks.is_empty() {
        blockers.push(SUIT_BFM_LANDMARKS_MISSING);
    }
    if candidates.is_empty() {
        blockers.push(SUIT_BFM_LANDMARK_VERTEX_MAP_MISSING);
    }

    let mut landmark_seen = vec![false; landmarks.len()];
    let mut vertex_seen = vec![false; bfm.positions.len()];
    let mut map_shape_valid = candidates.len() == landmarks.len();
    let mut uv_valid = map_shape_valid;
    for candidate in candidates {
        let landmark_index = usize::from(candidate.landmark);
        let vertex_index = usize::from(candidate.vertex);
        if landmark_index >= landmarks.len()
            || vertex_index >= bfm.positions.len()
            || landmark_seen[landmark_index]
            || vertex_seen[vertex_index]
        {
            map_shape_valid = false;
            uv_valid = false;
            continue;
        }
        landmark_seen[landmark_index] = true;
        vertex_seen[vertex_index] = true;
        let Some(uv) = bfm.texcoords.get(vertex_index) else {
            map_shape_valid = false;
            uv_valid = false;
            continue;
        };
        if uv
            .iter()
            .any(|coordinate| !coordinate.is_finite() || !(0.0..=1.0).contains(coordinate))
        {
            uv_valid = false;
        }
    }
    if !candidates.is_empty() && !map_shape_valid {
        blockers.push(SUIT_BFM_LANDMARK_VERTEX_MAP_INVALID);
    }
    if !candidates.is_empty() && !uv_valid {
        blockers.push(SUIT_BFM_LANDMARK_UV_INVALID);
    }

    let classes_shape_valid = landmark_classes.is_some_and(|classes| {
        classes.len() == landmarks.len()
            && classes
                .iter()
                .all(|class| usize::from(*class) < crate::face_parsing::FACE_PARSING_V2_CLASSES)
    }) && vertex_classes.is_some_and(|classes| {
        classes.len() == bfm.positions.len()
            && classes
                .iter()
                .all(|class| usize::from(*class) < crate::face_parsing::FACE_PARSING_V2_CLASSES)
    });
    let mut regions_valid = map_shape_valid && uv_valid && classes_shape_valid;
    if regions_valid {
        let landmark_classes = landmark_classes.unwrap_or(&[]);
        let vertex_classes = vertex_classes.unwrap_or(&[]);
        regions_valid = candidates.iter().all(|candidate| {
            landmark_classes[usize::from(candidate.landmark)]
                == vertex_classes[usize::from(candidate.vertex)]
        });
    }
    if !candidates.is_empty() && !regions_valid {
        blockers.push(SUIT_BFM_LANDMARK_REGION_MISMATCH);
    }

    let accepted_candidates = if map_shape_valid && uv_valid && regions_valid {
        candidates.len()
    } else {
        0
    };
    let mapping_proven = blockers.is_empty()
        && accepted_candidates == SUIT_BFM_LANDMARK_COUNT
        && finite_landmarks == SUIT_BFM_LANDMARK_COUNT;
    SuitBfmLandmarkMapReport {
        supplied_landmarks: landmarks.len(),
        finite_landmarks,
        nonfinite_landmarks,
        supplied_candidates: candidates.len(),
        accepted_candidates,
        uv_valid,
        regions_valid,
        blockers,
        mapping_proven,
    }
}

/// Audits the recovered BFM asset and the same-job FaceParsingV2/landmark
/// producers before D4 admission. The pinned BFM carries a generic face UV
/// atlas, no landmark-to-vertex map, no suit-local UV, and no live
/// shelter/region weights, so it cannot become a suit VBO/EBO or owner mask.
pub fn produce_suit_bfm_same_job_bundle(
    job_id: u64,
    ticket: Option<u64>,
    frame_size: [u32; 2],
    source_sha256: [u8; 32],
    bfm_obj: &[u8],
    bfm_obj_sha256: [u8; 32],
    parsing_model_sha256: [u8; 32],
    parsing_pixels: &[u8],
    landmarks: &[[f32; 2]],
) -> Result<SuitBfmSameJobReport, Refusal> {
    if job_id == 0 || ticket != Some(job_id) {
        return Err(Refusal::ValueOutOfRange(SUIT_LIVE_JOB_MISMATCH));
    }
    if frame_size[0] == 0 || frame_size[1] == 0 {
        return Err(Refusal::ShapeMismatch("suit live frame"));
    }
    if source_sha256.iter().all(|byte| *byte == 0) {
        return Err(Refusal::MissingInput(SOURCE_SHA_MISSING));
    }
    if parsing_model_sha256 != FACE_PARSING_V2_MODEL_SHA256 {
        return Err(Refusal::ValueOutOfRange("suit FaceParsingV2 model SHA-256"));
    }
    if parsing_pixels.len()
        != crate::face_parsing::FACE_PARSING_V2_SIDE as usize
            * crate::face_parsing::FACE_PARSING_V2_SIDE as usize
    {
        return Err(Refusal::ShapeMismatch(SUIT_BFM_PARSING_PIXELS));
    }
    let parsing_class_ids_valid = parsing_pixels
        .iter()
        .all(|class| usize::from(*class) < crate::face_parsing::FACE_PARSING_V2_CLASSES);
    if !parsing_class_ids_valid {
        return Err(Refusal::ValueOutOfRange(
            "suit FaceParsingV2 17-class labels",
        ));
    }
    if landmarks.len() != SUIT_BFM_LANDMARK_COUNT
        || landmarks
            .iter()
            .flatten()
            .any(|coordinate| !coordinate.is_finite())
    {
        return Err(Refusal::MissingInput(SUIT_BFM_LANDMARKS_MISSING));
    }
    if bfm_obj_sha256 != SUIT_BFM_OBJ_SHA256 {
        return Err(Refusal::ValueOutOfRange(SUIT_BFM_OBJ_SHA_MISMATCH));
    }
    if capture_sha256(bfm_obj) != SUIT_BFM_OBJ_SHA256 {
        return Err(Refusal::ValueOutOfRange(SUIT_BFM_OBJ_SHA_MISMATCH));
    }

    let bfm = parse_suit_bfm_obj(bfm_obj)?;
    let blocking_inputs = bfm_mapping_blockers(&bfm);
    Ok(SuitBfmSameJobReport {
        bfm,
        parsing_pixels: parsing_pixels.len(),
        parsing_classes: crate::face_parsing::FACE_PARSING_V2_CLASSES,
        parsing_class_ids_valid,
        landmarks: landmarks.len(),
        blocking_inputs,
        same_job: bfm_blocked_report()?,
    })
}

/// Provenance hashes that must describe the bytes admitted to the D4 gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SuitSameJobSources {
    pub parsing_model_sha256: [u8; 32],
    pub face2d_vbo_sha256: [u8; 32],
    pub face2d_ebo_sha256: [u8; 32],
    pub owner_mask_sha256: [u8; 32],
    pub mask_uv_sha256: [u8; 32],
    pub material_uv_sha256: [u8; 32],
}

impl SuitSameJobSources {
    pub fn missing_field(self) -> Option<&'static str> {
        if self.parsing_model_sha256 == [0; 32] {
            return Some("FaceParsingV2 model SHA-256");
        }
        if self.face2d_vbo_sha256 == [0; 32] {
            return Some(D4_CAPTURE_FACE2D_VBO);
        }
        if self.face2d_ebo_sha256 == [0; 32] {
            return Some(D4_CAPTURE_FACE2D_EBO);
        }
        if self.owner_mask_sha256 == [0; 32] {
            return Some(D4_CAPTURE_OWNER_MASK);
        }
        if self.mask_uv_sha256 == [0; 32] {
            return Some(D4_CAPTURE_MASK_UV);
        }
        if self.material_uv_sha256 == [0; 32] {
            return Some(D4_CAPTURE_MATERIAL_UV);
        }
        None
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SuitSameJobRefusal {
    pub preset_id: String,
    pub order: u32,
    pub effect: String,
    pub field: &'static str,
    pub cause: Refusal,
}

impl fmt::Display for SuitSameJobRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}#{} {} {}: {:?}",
            self.preset_id, self.order, self.effect, self.field, self.cause
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SuitSameJobReport {
    pub static_executable_passes: usize,
    pub supplied_inputs: usize,
    pub live: Option<OwnedSuitLiveBundle>,
    pub refusals: Vec<SuitSameJobRefusal>,
}

impl SuitSameJobReport {
    pub fn live_executable_passes(&self) -> usize {
        self.live
            .as_ref()
            .map_or(0, |live| live.live_executable_passes)
    }
}

fn first_missing_input_field(
    pass: &crate::portrait_makeup_bundle::OwnedSuitPass,
    input: &OwnedSuitPassJobInput,
) -> Option<(&'static str, Refusal)> {
    if input.job_id != input.ticket.unwrap_or(0) {
        return Some((
            SUIT_LIVE_JOB_MISMATCH,
            Refusal::ValueOutOfRange(SUIT_LIVE_JOB_MISMATCH),
        ));
    }
    if input.draw.vertices.is_empty() {
        return Some((
            D4_CAPTURE_FACE2D_VBO,
            Refusal::MissingInput(D4_CAPTURE_FACE2D_VBO),
        ));
    }
    if input.draw.indices_u16.is_empty() {
        return Some((
            D4_CAPTURE_FACE2D_EBO,
            Refusal::MissingInput(D4_CAPTURE_FACE2D_EBO),
        ));
    }
    if input.owner_mask.is_none() {
        return Some((SUIT_MASK_MISSING, Refusal::MissingInput(SUIT_MASK_MISSING)));
    }
    if input.material.is_none() {
        return Some((
            SUIT_MATERIAL_MISSING,
            Refusal::MissingInput(SUIT_MATERIAL_MISSING),
        ));
    }
    if input.material_uv.is_none() {
        return Some((
            D4_CAPTURE_MATERIAL_UV,
            Refusal::MissingInput(SUIT_MATERIAL_UV_MISSING),
        ));
    }
    if input.mask_uv.is_none() {
        return Some((
            D4_CAPTURE_MASK_UV,
            Refusal::MissingInput(D4_CAPTURE_MASK_UV),
        ));
    }
    if !pass.force_use_shelter && input.shelter.is_none() {
        return Some(("suit shelter", Refusal::MissingInput("suit shelter")));
    }
    if input.region.is_none() {
        return Some(("suit region", Refusal::MissingInput("suit region")));
    }
    None
}

/// Admits only one complete same-job capture. Every static PF90033 pass gets
/// the exact absent field named; a partial capture is never attached.
pub fn produce_suit_same_job_bundle(
    job_id: u64,
    ticket: Option<u64>,
    frame_size: [u32; 2],
    source_sha256: [u8; 32],
    sources: SuitSameJobSources,
    inputs: Option<&[OwnedSuitPassJobInput]>,
) -> Result<SuitSameJobReport, Refusal> {
    if job_id == 0 || ticket != Some(job_id) {
        return Err(Refusal::ValueOutOfRange(SUIT_LIVE_JOB_MISMATCH));
    }
    if frame_size[0] == 0 || frame_size[1] == 0 {
        return Err(Refusal::ShapeMismatch("suit live frame"));
    }
    if source_sha256.iter().all(|byte| *byte == 0) {
        return Err(Refusal::MissingInput(SOURCE_SHA_MISSING));
    }
    if sources.parsing_model_sha256 != FACE_PARSING_V2_MODEL_SHA256 {
        return Err(Refusal::ValueOutOfRange("suit FaceParsingV2 model SHA-256"));
    }

    let static_bundle = load_owned_suit_bundle()?;
    let inputs = inputs.unwrap_or(&[]);
    let mut refusals = Vec::new();
    for preset in &static_bundle.presets {
        for pass in preset.passes.iter().filter(|pass| pass.pf == 90033) {
            if pass.require_executable().is_err() {
                continue;
            }
            let Some(input) = inputs.iter().find(|input| {
                input.preset_id == pass.preset_id
                    && input.order == pass.order
                    && input.pf == pass.pf
            }) else {
                refusals.push(SuitSameJobRefusal {
                    preset_id: pass.preset_id.clone(),
                    order: pass.order,
                    effect: pass.effect.clone(),
                    field: SUIT_LIVE_JOB_INPUTS_MISSING,
                    cause: Refusal::MissingInput(SUIT_LIVE_JOB_INPUTS_MISSING),
                });
                continue;
            };
            if input.job_id != job_id || input.ticket != Some(job_id) {
                refusals.push(SuitSameJobRefusal {
                    preset_id: pass.preset_id.clone(),
                    order: pass.order,
                    effect: pass.effect.clone(),
                    field: SUIT_LIVE_JOB_MISMATCH,
                    cause: Refusal::ValueOutOfRange(SUIT_LIVE_JOB_MISMATCH),
                });
                continue;
            }
            if input.preset_id != pass.preset_id || input.order != pass.order || input.pf != pass.pf
            {
                refusals.push(SuitSameJobRefusal {
                    preset_id: pass.preset_id.clone(),
                    order: pass.order,
                    effect: pass.effect.clone(),
                    field: SUIT_LIVE_BINDING_MISMATCH,
                    cause: Refusal::ValueOutOfRange(SUIT_LIVE_BINDING_MISMATCH),
                });
                continue;
            }
            if let Some((field, cause)) = first_missing_input_field(pass, input) {
                refusals.push(SuitSameJobRefusal {
                    preset_id: pass.preset_id.clone(),
                    order: pass.order,
                    effect: pass.effect.clone(),
                    field,
                    cause,
                });
            }
        }
    }

    if !refusals.is_empty() {
        return Ok(SuitSameJobReport {
            static_executable_passes: static_bundle.executable_passes,
            supplied_inputs: inputs.len(),
            live: None,
            refusals,
        });
    }

    if let Some(field) = sources.missing_field() {
        return Err(Refusal::MissingInput(field));
    }
    let live =
        produce_owned_suit_live_bundle(job_id, ticket, frame_size, source_sha256, Some(inputs))?;
    if live.live_executable_passes != live.static_executable_passes || !live.refusals.is_empty() {
        return Err(Refusal::MissingInput(SUIT_LIVE_JOB_INPUTS_MISSING));
    }
    Ok(SuitSameJobReport {
        static_executable_passes: live.static_executable_passes,
        supplied_inputs: inputs.len(),
        live: Some(live),
        refusals: Vec::new(),
    })
}

/// Hashes a captured file without interpreting its pixels or topology.
pub fn capture_sha256(bytes: &[u8]) -> [u8; 32] {
    sha256_32(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn absent_capture_names_all_337_static_passes() {
        let report = produce_suit_same_job_bundle(
            41,
            Some(41),
            [1024, 768],
            [7; 32],
            SuitSameJobSources {
                parsing_model_sha256: FACE_PARSING_V2_MODEL_SHA256,
                face2d_vbo_sha256: [1; 32],
                face2d_ebo_sha256: [2; 32],
                owner_mask_sha256: [3; 32],
                mask_uv_sha256: [4; 32],
                material_uv_sha256: [5; 32],
            },
            None,
        )
        .expect("missing capture is a report, not an Err");

        assert_eq!(report.static_executable_passes, 337);
        assert_eq!(report.supplied_inputs, 0);
        assert_eq!(report.live_executable_passes(), 0);
        assert_eq!(report.refusals.len(), 337);
        assert!(report
            .refusals
            .iter()
            .all(|refusal| refusal.field == SUIT_LIVE_JOB_INPUTS_MISSING));
    }

    #[test]
    fn bfm_parser_validates_geometry_uv_indices_and_references() {
        let valid = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n";
        let bfm = parse_bfm_obj(valid, 3, 3, 1).unwrap();
        assert_eq!(bfm.positions.len(), 3);
        assert_eq!(bfm.texcoords.len(), 3);
        assert_eq!(bfm.indices_u16, [0, 1, 2]);
        assert_eq!(bfm.texcoord_indices_u16, [0, 1, 2]);
        assert!(bfm.vertex_uv_identity);
        assert!(bfm.all_vertices_referenced);
        assert!(bfm.all_texcoords_referenced);
        assert!(!bfm.has_material_directive);
        assert!(!bfm.has_group_directive);

        let invalid_vertex = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 4/3\n";
        assert_eq!(
            parse_bfm_obj(invalid_vertex, 3, 3, 1),
            Err(Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID))
        );

        let invalid_uv = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/4\n";
        assert_eq!(
            parse_bfm_obj(invalid_uv, 3, 3, 1),
            Err(Refusal::ShapeMismatch(SUIT_BFM_OBJ_INVALID))
        );
    }

    #[test]
    fn generic_bfm_lists_each_unrecovered_mapping() {
        let bfm = parse_bfm_obj(
            b"v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nf 1/1 2/2 3/3\n",
            3,
            3,
            1,
        )
        .unwrap();
        assert_eq!(
            bfm_mapping_blockers(&bfm),
            [
                SUIT_BFM_LANDMARK_VERTEX_MAP_MISSING,
                SUIT_BFM_SUIT_LOCAL_UV_MISSING,
                SUIT_BFM_OWNER_MASK_CLASS_MAP_MISSING,
                SUIT_BFM_SHELTER_WEIGHT_MISSING,
                SUIT_BFM_REGION_WEIGHT_MISSING,
            ]
        );
    }

    #[test]
    fn owner_mask_class_map_is_exact_and_names_the_four_unmapped_classes() {
        for class in SUIT_FACEPARSING_R_CLASSES {
            assert_eq!(suit_owner_mask_class_channel(*class), Ok(MaskChannel::R));
        }
        for class in SUIT_FACEPARSING_G_CLASSES {
            assert_eq!(suit_owner_mask_class_channel(*class), Ok(MaskChannel::G));
        }
        for class in SUIT_FACEPARSING_B_CLASSES {
            assert_eq!(suit_owner_mask_class_channel(*class), Ok(MaskChannel::B));
        }
        for class in SUIT_FACEPARSING_A_CLASSES {
            assert_eq!(suit_owner_mask_class_channel(*class), Ok(MaskChannel::A));
        }
        assert_eq!(
            suit_owner_mask_class_channel(1),
            Err(SUIT_FACEPARSING_CLASS_1_MISSING)
        );
        assert_eq!(
            suit_owner_mask_class_channel(3),
            Err(SUIT_FACEPARSING_CLASS_3_MISSING)
        );
        assert_eq!(
            suit_owner_mask_class_channel(6),
            Err(SUIT_FACEPARSING_CLASS_6_MISSING)
        );
        assert_eq!(
            suit_owner_mask_class_channel(8),
            Err(SUIT_FACEPARSING_CLASS_8_MISSING)
        );
        assert_eq!(
            suit_owner_mask_class_channel(17),
            Err(SUIT_BFM_OWNER_MASK_CLASS_MAP_MISSING)
        );
    }

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn generic_bfm_blocks_every_executable_pass() {
        let report = bfm_blocked_report().unwrap();
        assert_eq!(report.static_executable_passes, 337);
        assert_eq!(report.live_executable_passes(), 0);
        assert_eq!(report.refusals.len(), 337);
        assert!(report
            .refusals
            .iter()
            .all(|refusal| refusal.field == SUIT_BFM_LOCAL_BINDING_MISSING));
    }

    fn landmark_fixture(bytes: &[u8]) -> Vec<[f32; 2]> {
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

    fn toy_bfm(vertices: usize) -> SuitBfmTopology {
        let mut obj = String::new();
        for vertex in 0..vertices {
            obj.push_str(&format!("v {vertex} 0 0\n"));
        }
        for _ in 0..vertices {
            obj.push_str("vt 0.5 0.5\n");
        }
        for face in 0..vertices / 3 {
            let a = face * 3 + 1;
            let b = a + 1;
            let c = a + 2;
            obj.push_str(&format!("f {a}/{a} {b}/{b} {c}/{c}\n"));
        }
        parse_bfm_obj(obj.as_bytes(), vertices, vertices, vertices / 3).unwrap()
    }

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn real_414_probe_fixture_has_eight_nonfinite_points_and_no_map() {
        let landmarks =
            landmark_fixture(aurora_portrait::private_asset!("fixtures/portrait/pf30092/pf30092.414.f32le").unwrap());
        assert_eq!(landmarks.len(), SUIT_BFM_LANDMARK_COUNT);
        let report = audit_suit_bfm_landmark_map(
            1,
            Some(1),
            [7; 32],
            FACE_PARSING_V2_MODEL_SHA256,
            &vec![
                1;
                crate::face_parsing::FACE_PARSING_V2_SIDE as usize
                    * crate::face_parsing::FACE_PARSING_V2_SIDE as usize
            ],
            &landmarks,
            &toy_bfm(3),
            None,
            None,
            None,
        );
        assert_eq!(
            report.nonfinite_landmarks,
            [108, 109, 110, 111, 112, 113, 187, 188]
        );
        assert_eq!(report.finite_landmarks, 406);
        assert_eq!(report.accepted_candidates, 0);
        assert!(!report.mapping_proven);
        assert!(report.blockers.contains(&SUIT_BFM_LANDMARKS_MISSING));
        assert!(report
            .blockers
            .contains(&SUIT_BFM_LANDMARK_VERTEX_MAP_MISSING));
    }

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn finite_414_points_without_correspondence_are_not_a_map() {
        let landmarks = landmark_fixture(aurora_portrait::private_asset!("fixtures/portrait/pf30092/pf30092.dense414.f32le").unwrap());
        let report = audit_suit_bfm_landmark_map(
            2,
            Some(2),
            [8; 32],
            FACE_PARSING_V2_MODEL_SHA256,
            &vec![
                1;
                crate::face_parsing::FACE_PARSING_V2_SIDE as usize
                    * crate::face_parsing::FACE_PARSING_V2_SIDE as usize
            ],
            &landmarks,
            &toy_bfm(3),
            None,
            Some(&vec![1; SUIT_BFM_LANDMARK_COUNT]),
            Some(&[1; 3]),
        );
        assert_eq!(report.finite_landmarks, SUIT_BFM_LANDMARK_COUNT);
        assert_eq!(report.supplied_candidates, 0);
        assert_eq!(report.accepted_candidates, 0);
        assert!(!report.mapping_proven);
        assert!(report
            .blockers
            .contains(&SUIT_BFM_LANDMARK_VERTEX_MAP_MISSING));
    }

    #[test]
    fn complete_candidate_map_requires_uv_and_matching_parsing_regions() {
        let vertices = SUIT_BFM_LANDMARK_COUNT;
        let bfm = toy_bfm(vertices);
        let landmarks = vec![[1.0, 1.0]; SUIT_BFM_LANDMARK_COUNT];
        let candidates = (0..SUIT_BFM_LANDMARK_COUNT)
            .map(|index| SuitBfmLandmarkCandidate {
                landmark: index as u16,
                vertex: index as u16,
            })
            .collect::<Vec<_>>();
        let report = audit_suit_bfm_landmark_map(
            3,
            Some(3),
            [9; 32],
            FACE_PARSING_V2_MODEL_SHA256,
            &vec![
                1;
                crate::face_parsing::FACE_PARSING_V2_SIDE as usize
                    * crate::face_parsing::FACE_PARSING_V2_SIDE as usize
            ],
            &landmarks,
            &bfm,
            Some(&candidates),
            Some(&vec![1; SUIT_BFM_LANDMARK_COUNT]),
            Some(&vec![1; vertices]),
        );
        assert!(report.uv_valid);
        assert!(report.regions_valid);
        assert_eq!(report.accepted_candidates, SUIT_BFM_LANDMARK_COUNT);
        assert!(report.mapping_proven);

        let mut vertex_classes = vec![1; vertices];
        vertex_classes[173] = 2;
        let report = audit_suit_bfm_landmark_map(
            3,
            Some(3),
            [9; 32],
            FACE_PARSING_V2_MODEL_SHA256,
            &vec![
                1;
                crate::face_parsing::FACE_PARSING_V2_SIDE as usize
                    * crate::face_parsing::FACE_PARSING_V2_SIDE as usize
            ],
            &landmarks,
            &bfm,
            Some(&candidates),
            Some(&vec![1; SUIT_BFM_LANDMARK_COUNT]),
            Some(&vertex_classes),
        );
        assert!(!report.regions_valid);
        assert_eq!(report.accepted_candidates, 0);
        assert!(!report.mapping_proven);
        assert!(report.blockers.contains(&SUIT_BFM_LANDMARK_REGION_MISMATCH));
    }
}
