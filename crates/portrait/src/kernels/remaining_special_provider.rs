//! Typed provider boundary for the remaining special rows.
//!
//! This module does not add vendor pixel laws. It exposes the same-job inputs
//! that are already understood, delegates to existing recovered kernels where
//! one exists, and refuses before image mutation where the static PE has no
//! final pixel consumer.

use crate::{Image, Refusal};

use super::{body_geometry, double_chin, face_detail, makeup_material, skin_adjust};

pub const PF90001_PIXEL_LAW_OPEN: &str =
    "PF90001 requires the SkinDB service response and final model-to-source blend";
pub const PF90003_PIXEL_LAW_OPEN: &str =
    "PF90003 TradBodySkinTone curve, LUT, shader and writeback are END_STATIC";
pub const PF101_LIVE_ONLY: &str =
    "PF101 FilterType 41/42/43/44 material branches and common draw are statically callable; the xmm1 factor reaches vtable+0xF8 after 0x4AD6C50, but the dynamic vtable target and same-job compiled defines/GL+FBO/readback remain unbound";
pub const PF30209_END_STATIC: &str =
    "PF30209 stage +0x3A0/+0x3B8 is not forwarded by the fixed PE";
/// PF100005 has no local model/tensor/handler receipt. The 8-plane layout,
/// `cv::max(maskA,maskB)` and `boundingRect` are static facts; the final
/// group blend and UI-to-flag binding are LIVE_ONLY.
pub const PF100005_LIVE_ONLY: &str =
    "PF100005 8 CV_32FC1 planes and mask max/boundingRect are static; the final RGB/source blend, UI-to-group binding and handler-to-layer receipt are LIVE_ONLY";
/// PF90087's final CPU/CUDA blend and native per-part degree map are already
/// callable. The only recoverable geometry is PF202's hairline template; it
/// produces a part-line band, not the HairSeamV0 `hairMask`, whose channel,
/// size, binding and repair model are absent from the fixed PE.
pub const PF90087_HAIRLINE_GEOMETRY_OPEN: &str =
    "PF90087 HairSeamV0 part-line band is constructible from PF202's recovered 194-point hairline template, but the native hairMask channel/size/binding, the HairSeam repair model and the part-to-band assignment are LIVE_ONLY/NOT-FOUND; no pixel result is inferred";
/// PF8/PF9 know the preset pixel law, BlusherMaskType=15 family, ROIRect and
/// MeshType 2/4, but the FaceStereo owner-mask producer, the 3D-to-UV
/// projection/UV set and the side-face mask ownership are not in the fixed PE.
pub const FACE_STEREO_HOST_MASK_OPEN: &str =
    "PF8/PF9 FaceStereo owner mask (BlusherMaskType=15 RGBA) has no 3D projection/UV producer from FaceMesh landmarks; MeshType 2/4, ROIRect and FilterType=42 are static facts only";
/// PF90034's per-face weights are known; the dense shift field is a
/// DoubleChinV2Result model output (`AIDataService +0x1C8`), not a landmark
/// displacement. PF30209's jaw window is a separate boolean activation path.
pub const PF90034_SHADOW_FIELD_OPEN: &str =
    "PF90034 geometry/shadow weights f32(PF30188*PF90034)/f32(PF90034*PF30189) are typed producers, but the DoubleChinV2Result dense 2-channel shift field, FaceParsing A/B/C/D grouping, ROI writeback and shadow composite are LIVE_ONLY; the 194-point jaw window is PF30209, not a shift-field substitute";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProviderStatus {
    TypedProducer,
    EndStatic,
    LiveOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RemainingSpecialContract {
    pub pf: u32,
    pub status: ProviderStatus,
    pub required_same_job_inputs: &'static [&'static str],
    pub reason: &'static str,
}

pub const CONTRACTS: [RemainingSpecialContract; 12] = [
    RemainingSpecialContract {
        pf: 90001,
        status: ProviderStatus::LiveOnly,
        required_same_job_inputs: &["source", "ui", "per_face_engine", "SkinDB response"],
        reason: PF90001_PIXEL_LAW_OPEN,
    },
    RemainingSpecialContract {
        pf: 90003,
        status: ProviderStatus::EndStatic,
        required_same_job_inputs: &[
            "source_rgb",
            "body_mask",
            "roi_mask",
            "native 0x15F93",
            "TradBodySkinTone curve/LUT",
        ],
        reason: PF90003_PIXEL_LAW_OPEN,
    },
    RemainingSpecialContract {
        pf: 403,
        status: ProviderStatus::TypedProducer,
        required_same_job_inputs: &["source_mask", "radius01"],
        reason: body_geometry::PF403_FINAL_CONSUMER_OPEN,
    },
    RemainingSpecialContract {
        pf: 101,
        status: ProviderStatus::LiveOnly,
        required_same_job_inputs: &[
            "face mask input id 0x15",
            "wcz/wcy material RGBA",
            "material/mask UV",
            "W(maskIndex=0)",
            "W(maskIndex=1)",
            "AdjustFactor gate",
            "FilterType 41/42/43/44 effect record",
            "compiled defines",
            "same-job GL program/FBO/readback",
        ],
        reason: PF101_LIVE_ONLY,
    },
    RemainingSpecialContract {
        pf: 30188,
        status: ProviderStatus::TypedProducer,
        required_same_job_inputs: &["native PF90034", "native PF30188"],
        reason: "recovered producer is PF90034 * PF30188; final model output is open",
    },
    RemainingSpecialContract {
        pf: 30189,
        status: ProviderStatus::TypedProducer,
        required_same_job_inputs: &["native PF90034", "native PF30189"],
        reason: "recovered producer is PF90034 * PF30189; final shadow composite is open",
    },
    RemainingSpecialContract {
        pf: 30209,
        status: ProviderStatus::EndStatic,
        required_same_job_inputs: &[
            "face_194_packet",
            "jawline_indices",
            "mask",
            "model_output",
        ],
        reason: PF30209_END_STATIC,
    },
    RemainingSpecialContract {
        pf: 100005,
        status: ProviderStatus::LiveOnly,
        required_same_job_inputs: &[
            "source RGBA",
            "8 CV_32FC1 model planes",
            "group A/B enable flags",
            "ui",
            "model SHA-256",
            "frame SHA-256",
            "handler-to-layer receipt",
        ],
        reason: PF100005_LIVE_ONLY,
    },
    RemainingSpecialContract {
        pf: 90087,
        status: ProviderStatus::TypedProducer,
        required_same_job_inputs: &[
            "194-point landmark packet",
            "135 PF202 hairline weights",
            "per-part native degree map",
            "PF90087 hairMask plane",
            "PF90087 repair RGB + coverage",
            "same-job ROI",
        ],
        reason: PF90087_HAIRLINE_GEOMETRY_OPEN,
    },
    RemainingSpecialContract {
        pf: 8,
        status: ProviderStatus::LiveOnly,
        required_same_job_inputs: &[
            "FaceStereo yy material for MeshType 2/4",
            "BlusherMaskType=15 owner mask RGBA",
            "3D-to-mask UV or captured mask plane",
            "degree01",
            "same-job ROI",
        ],
        reason: FACE_STEREO_HOST_MASK_OPEN,
    },
    RemainingSpecialContract {
        pf: 9,
        status: ProviderStatus::LiveOnly,
        required_same_job_inputs: &[
            "FaceStereo gg material for MeshType 2/4",
            "BlusherMaskType=15 owner mask RGBA",
            "3D-to-mask UV or captured mask plane",
            "degree01",
            "same-job ROI",
        ],
        reason: FACE_STEREO_HOST_MASK_OPEN,
    },
    RemainingSpecialContract {
        pf: 90034,
        status: ProviderStatus::TypedProducer,
        required_same_job_inputs: &[
            "native per-face PF90034/PF30188/PF30189",
            "DoubleChinV2Result dense 2-channel shift field",
            "FaceParsing A/B/C/D planes",
            "same-job ROI writeback",
            "shadow composite",
        ],
        reason: PF90034_SHADOW_FIELD_OPEN,
    },
];

pub fn contract(pf: u32) -> Option<RemainingSpecialContract> {
    CONTRACTS.iter().copied().find(|row| row.pf == pf)
}

/// Canonical per-face weights live in the double-chin kernel; this alias keeps
/// the provider's public name stable.
pub use super::double_chin::Weights as DoubleChinWeights;

/// Static PF30188/30189 producer. These are already-native values; no UI scale
/// or clamp is inferred here. The caller still owns the model and ROI stages.
pub fn double_chin_weights(
    native_pf90034: f32,
    native_pf30188: f32,
    native_pf30189: f32,
) -> Result<DoubleChinWeights, Refusal> {
    if !native_pf90034.is_finite()
        || !native_pf30188.is_finite()
        || !native_pf30189.is_finite()
    {
        return Err(Refusal::ValueOutOfRange("double chin native values"));
    }
    Ok(DoubleChinWeights::from_degrees(
        native_pf90034,
        native_pf30188,
        native_pf30189,
    ))
}

/// PF403 delegates to the recovered source-mask kernel.
pub fn enhance_breast_mask(source_mask: &Image, radius01: f32) -> Result<Image, Refusal> {
    body_geometry::enhance_breast_mask(source_mask, radius01)
}

/// PF90001 delegates only to the existing gate. An active non-zero request
/// remains refused until the SkinDB response has a recovered blend law.
pub fn texture_smooth(
    source: &Image,
    ui: f32,
    per_face_engine: &[f32],
) -> Result<Image, Refusal> {
    skin_adjust::texture_smooth_apply(source, ui, per_face_engine)
}

/// PF90003 validates the same-job shape and permits only the zero-control
/// identity. It deliberately does not call the old explicit-target shortcut.
pub fn body_skin_uniform_contract(
    source_rgb: &[u8],
    width: u32,
    height: u32,
    body_mask: &[u8],
    roi_mask: &[u8],
    ui: f32,
    native_0x15f93: f32,
) -> Result<Vec<u8>, Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch("source_rgb"));
    }
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("source_rgb"))?;
    if source_rgb.len() != pixels * 3 {
        return Err(Refusal::ShapeMismatch("source_rgb"));
    }
    if body_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("body_mask"));
    }
    if roi_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("roi_mask"));
    }
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if ui == 0.0 {
        return Ok(source_rgb.to_vec());
    }
    if !native_0x15f93.is_finite() {
        return Err(Refusal::ValueOutOfRange("native 0x15F93"));
    }
    Err(Refusal::MissingInput(PF90003_PIXEL_LAW_OPEN))
}

#[derive(Clone, Copy, Debug)]
pub struct Pf101SameJobInputs<'a> {
    pub eye_mask: &'a [u8],
    pub eye_mask_type: u32,
    pub face_blur_factor: f32,
    pub lut: &'a [u8],
}

/// PF101 remains live-only: even structurally complete inputs cannot select a
/// final program/pass sequence from the fixed PE.
pub fn silkworm(inputs: Pf101SameJobInputs<'_>) -> Result<Image, Refusal> {
    if inputs.eye_mask.is_empty() {
        return Err(Refusal::MissingInput("PF101 eye_mask"));
    }
    if inputs.lut.is_empty() {
        return Err(Refusal::MissingInput("PF101 lut"));
    }
    if !inputs.face_blur_factor.is_finite() {
        return Err(Refusal::ValueOutOfRange("PF101 face_blur_factor"));
    }
    let _eye_mask_type = inputs.eye_mask_type;
    Err(Refusal::MissingInput(PF101_LIVE_ONLY))
}

/// PF30209 operates on the vendor 194-point face packet.
pub const JAWLINE_POINTS: usize = 194;
/// `center = L[54]` (`0x80A0A2C`/`0x80A0A44`).
pub const JAWLINE_CENTER: usize = 54;
/// Fixed window `L[109..=141]` (`0x80A0A1C`/`0x80A0A24`).
pub const JAWLINE_WINDOW_START: usize = 109;
pub const JAWLINE_WINDOW_END: usize = 141;
/// `L[109..=141]` is 33 points; the vendor reads point 141 inclusively.
pub const JAWLINE_WINDOW_LEN: usize = JAWLINE_WINDOW_END - JAWLINE_WINDOW_START + 1;

/// The recovered static PF30209 consumer only tests `any(value > 0)`
/// (`0x7D150B8` `movss`/`ucomiss`/`seta`). The magnitude is never used as a
/// weight in the fixed PE. An empty map has no active node.
pub fn jawline_active(per_face_pf30209: &[f32]) -> bool {
    per_face_pf30209.iter().any(|value| *value > 0.0)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct JawlineWindow {
    pub center: [f32; 2],
    pub contour: [[f32; 2]; JAWLINE_WINDOW_LEN],
}

/// Extract `center = L[54]` and the fixed 33-point window `L[109..=141]`.
///
/// This is the raw landmark range, not the vendor "ordered" contour: the
/// ordered index table is a same-job input the fixed PE does not contain.
pub fn jawline_window(points: &[[f32; 2]]) -> Result<JawlineWindow, Refusal> {
    if points.is_empty() {
        return Err(Refusal::MissingInput("PF30209 face_194_packet"));
    }
    if points.len() != JAWLINE_POINTS {
        return Err(Refusal::ShapeMismatch("PF30209 face_194_packet"));
    }
    if points.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("PF30209 face_194_packet"));
    }
    let mut contour = [[0.0f32; 2]; JAWLINE_WINDOW_LEN];
    contour.copy_from_slice(&points[JAWLINE_WINDOW_START..=JAWLINE_WINDOW_END]);
    Ok(JawlineWindow {
        center: points[JAWLINE_CENTER],
        contour,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct Pf30209SameJobInputs<'a> {
    pub face_194_packet: &'a [[f32; 2]],
    pub jawline_indices: &'a [u16],
    pub mask: &'a [u8],
    pub model_output: &'a [f32],
    pub per_face_pf30209: &'a [f32],
}

/// PF30209 is END_STATIC at the fixed PE boundary. Inputs are checked only so
/// callers cannot mistake an empty placeholder for a live-only residual; the
/// boolean gate and 33-point window are still evaluated and validated.
pub fn jawline(inputs: Pf30209SameJobInputs<'_>) -> Result<Image, Refusal> {
    if inputs.face_194_packet.is_empty() {
        return Err(Refusal::MissingInput("PF30209 face_194_packet"));
    }
    if inputs.jawline_indices.is_empty() {
        return Err(Refusal::MissingInput("PF30209 jawline_indices"));
    }
    if inputs.mask.is_empty() {
        return Err(Refusal::MissingInput("PF30209 mask"));
    }
    if inputs.model_output.is_empty() {
        return Err(Refusal::MissingInput("PF30209 model_output"));
    }
    if inputs.per_face_pf30209.is_empty() {
        return Err(Refusal::MissingInput("PF30209 per_face_pf30209"));
    }
    let _window = jawline_window(inputs.face_194_packet)?;
    let _active = jawline_active(inputs.per_face_pf30209);
    if inputs.per_face_pf30209.iter().any(|value| !value.is_finite())
        || inputs.model_output.iter().any(|value| !value.is_finite())
    {
        return Err(Refusal::ValueOutOfRange("PF30209 finite input"));
    }
    Err(Refusal::MissingInput(PF30209_END_STATIC))
}

/// PF100005 boundary: the producer planes and per-group enable flags are
/// checked, but no local model blob or handler receipt exists, so the full
/// effect fails closed. `face_detail::eye_white_active_mask` and
/// `eye_white_bounding_rect` cover the parts that are static.
pub fn eye_white_purify_provider(
    inputs: super::face_detail::EyeWhiteSameJobInputs<'_>,
) -> Result<Image, Refusal> {
    super::face_detail::eye_white_purify_same_job(inputs)
}

/// PF202's hairline mesh has a recovered 27-source-slot x 5-row lattice
/// (`TYPE1012_WIDTH=27`, `TYPE1012_HEIGHT=5`) built only from 194-point
/// landmark consumers. It is the only locally recovered hairline segment
/// family; the PF90087 part-to-segment assignment is still LIVE_ONLY.
pub const HAIRLINE_SOURCE_SLOTS: usize = 27;
pub const HAIRLINE_LATTICE_ROWS: usize = 5;
/// Recovered byte offsets (`TYPE1012_BLOCK_OFFSETS`) into the landmark194
/// packet. Each entry is a byte offset of one x f32; its y f32 is +4 bytes.
pub const HAIRLINE_LANDMARK_OFFSETS: [usize; 14] = [
    0x398, 0x548, 0x538, 0x528, 0x518, 0x508, 0x4F8, 0x4E8, 0x4D8, 0x4C8, 0x4B8, 0x4A8,
    0x498, 0x490,
];
/// Recovered `TYPE1012_Y_WEIGHTS`; the last two rows extend past the source
/// slot in the opposite direction and are retained verbatim.
pub const HAIRLINE_ROW_WEIGHTS: [f32; HAIRLINE_LATTICE_ROWS] = [
    0.0,
    0.20000000298023224,
    0.6000000238418579,
    -0.20000004768371582,
    -0.40000003576278687,
];
/// Recovered `TYPE1012_SLOT_ORDER`: the lattice consumes the 27 raw slots in
/// this pairwise-swapped order, not in block order.
pub const HAIRLINE_SLOT_ORDER: [usize; HAIRLINE_SOURCE_SLOTS] = [
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22,
    25, 24, 26,
];
/// Recovered x-byte offsets for the anchor (`0x1B0`) and reference (`0x4F8`)
/// landmark points; y is +4 bytes.
pub const HAIRLINE_ANCHOR_OFFSET: usize = 0x1B0;
pub const HAIRLINE_REFERENCE_OFFSET: usize = 0x4F8;
/// PF202 type1012 mesh has 136 source vertices: the 27x5 lattice plus the
/// anchor as the final vertex.
pub const HAIRLINE_LATTICE_VERTICES: usize = HAIRLINE_SOURCE_SLOTS * HAIRLINE_LATTICE_ROWS + 1;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairlinePartLine {
    pub source_slots: [[f32; 2]; HAIRLINE_SOURCE_SLOTS],
    pub lattice: [[f32; 2]; HAIRLINE_LATTICE_VERTICES],
}

/// Extract the recovered PF202 hairline source slots and span the same row
/// lattice the type1012 kernel uses, from a valid 194-point packet.
pub fn hairline_part_line(points194: &[[f32; 2]]) -> Result<HairlinePartLine, Refusal> {
    if points194.len() != 194 {
        return Err(Refusal::ShapeMismatch("PF90087_part_line_194_landmarks"));
    }
    if points194.iter().flatten().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("PF90087_part_line_194_landmarks"));
    }
    let flat: Vec<f32> = points194.iter().flat_map(|p| [p[0], p[1]]).collect();
    let mut source_slots = [[0.0f32; 2]; HAIRLINE_SOURCE_SLOTS];
    let mut slot = 0usize;
    // The recovered kernel addresses the flattened 194-point packet in byte
    // offsets: x at `offset`, y at `offset + 4`, next point at `offset + 8`.
    let read = |byte_offset: usize| -> Option<[f32; 2]> {
        let index = byte_offset.checked_div(4)?;
        let y = index.checked_add(1)?;
        if y >= flat.len() {
            return None;
        }
        Some([flat[index], flat[y]])
    };
    for &offset in &HAIRLINE_LANDMARK_OFFSETS {
        source_slots[slot] = read(offset).ok_or(Refusal::ShapeMismatch("PF90087_part_line_offset"))?;
        slot += 1;
        // 0x490 is the single final point; all earlier blocks have two.
        if offset != 0x490 {
            source_slots[slot] = read(offset + 8)
                .ok_or(Refusal::ShapeMismatch("PF90087_part_line_offset"))?;
            slot += 1;
        }
    }
    if slot != HAIRLINE_SOURCE_SLOTS {
        return Err(Refusal::ShapeMismatch("PF90087_part_line_slots"));
    }
    let anchor = read(HAIRLINE_ANCHOR_OFFSET).ok_or(Refusal::ShapeMismatch("PF90087_anchor_reference"))?;
    let reference = read(HAIRLINE_REFERENCE_OFFSET)
        .ok_or(Refusal::ShapeMismatch("PF90087_anchor_reference"))?;
    let delta = [anchor[0] - reference[0], anchor[1] - reference[1]];
    if delta[0].hypot(delta[1]) < 1e-6 {
        return Err(Refusal::ValueOutOfRange("PF90087_anchor_reference"));
    }
    let mut lattice = [[0.0f32; 2]; HAIRLINE_LATTICE_VERTICES];
    for (x, &slot) in HAIRLINE_SLOT_ORDER.iter().enumerate() {
        let source = source_slots[slot];
        let dx = anchor[0] - source[0];
        let dy = anchor[1] - source[1];
        for row in 0..HAIRLINE_LATTICE_ROWS {
            let weight = HAIRLINE_ROW_WEIGHTS[row];
            lattice[row * HAIRLINE_SOURCE_SLOTS + x] =
                [source[0] + weight * dx * 1.2, source[1] + weight * dy * 1.2];
        }
    }
    // The recovered kernel overwrites the last lattice vertex (row 4, col 26)
    // with the anchor, leaving 136 vertices total.
    lattice[HAIRLINE_LATTICE_VERTICES - 1] = anchor;
    Ok(HairlinePartLine {
        source_slots,
        lattice,
    })
}

fn distance_to_segment(point: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let length2 = dx * dx + dy * dy;
    if length2 <= 0.0 {
        return ((point[0] - a[0]).powi(2) + (point[1] - a[1]).powi(2)).sqrt();
    }
    let t = (((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2).clamp(0.0, 1.0);
    ((point[0] - a[0] - t * dx).powi(2) + (point[1] - a[1] - t * dy).powi(2)).sqrt()
}

fn strand_distance(point: [f32; 2], strand: &[[f32; 2]]) -> f32 {
    strand
        .windows(2)
        .map(|pair| distance_to_segment(point, pair[0], pair[1]))
        .fold(f32::INFINITY, f32::min)
}

/// Rasterize a finite-width band around the recovered hairline strands. This
/// is a geometry producer only: the returned plane feeds
/// `face_detail::hair_seam_repair_degree` / `hair_seam_blend_degree` and is
/// named `PF90087_PART_LINE_MASK_CANDIDATE`, never the native HairSeam mask.
pub fn hairline_part_line_mask(
    points194: &[[f32; 2]],
    width: u32,
    height: u32,
    band_px: f32,
) -> Result<Vec<u8>, Refusal> {
    let line = hairline_part_line(points194)?;
    if width == 0 || height == 0 || !band_px.is_finite() || band_px <= 0.0 {
        return Err(Refusal::ValueOutOfRange("PF90087_PART_LINE_BAND"));
    }
    let pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch("PF90087_PART_LINE_FRAME"))?;
    let mut mask = vec![0u8; pixels];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let point = [x as f32 + 0.5, y as f32 + 0.5];
            let distance = strand_distance(point, &line.lattice);
            mask[y * width as usize + x] = if distance <= band_px { 255 } else { 0 };
        }
    }
    Ok(mask)
}

/// Typed PF90087 same-job packet. It checks the per-part native degree map and
/// the candidate part-line geometry, then requires a distinct same-job
/// `hairMask`; neither a landmark hull nor this band can substitute for it.
#[derive(Clone, Copy, Debug)]
pub struct HairSeamSameJobInputs<'a> {
    pub points194: &'a [[f32; 2]],
    pub weights135: &'a [f32],
    pub parts: &'a [face_detail::HairSeamPartDegree],
    pub hair_mask: Option<&'a [u8]>,
}

pub fn hair_seam_provider(inputs: HairSeamSameJobInputs<'_>) -> Result<Image, Refusal> {
    let active = face_detail::hair_seam_active_parts(inputs.parts)?;
    let _degree = active.first().ok_or(Refusal::MissingInput("PF90087_PART_DEGREE_MAP"))?;
    let _line = hairline_part_line(inputs.points194)?;
    if inputs.weights135.len() != 135 {
        return Err(Refusal::ShapeMismatch("PF135_HAIRLINE_WEIGHTS"));
    }
    if inputs.hair_mask.is_none() {
        return Err(Refusal::MissingInput("PF90087_HAIR_MASK"));
    }
    // The native mask binding and model are not recovered; producing repair
    // RGB from a candidate band would invent the missing input.
    Err(Refusal::MissingInput(PF90087_HAIRLINE_GEOMETRY_OPEN))
}

/// PF8/PF9 evaluation boundary: the pixel law and static preset can run only
/// when a same-job BlusherMaskType=15 owner mask and material are supplied.
/// No 194/414 landmark or FaceMesh 3D projection in this tree produces either.
pub fn face_stereo_host_mask_boundary(pf: u32) -> Result<(), Refusal> {
    if pf != makeup_material::FACE_STEREO_SHADOW_PF
        && pf != makeup_material::FACE_STEREO_HIGHLIGHT_PF
    {
        return Err(Refusal::UnknownEffect(pf));
    }
    // The static preset family check never manufactures the owner mask.
    makeup_material::face_stereo_mask_channel()?;
    Err(Refusal::MissingInput(FACE_STEREO_HOST_MASK_OPEN))
}

/// PF90034 evaluation boundary: after validating the recovered per-face
/// weights, it still refuses because the DoubleChinV2Result shift field and
/// shadow composite are not produced by any local stage.
pub fn double_chin_shadow_boundary(
    weights: DoubleChinWeights,
) -> Result<(), Refusal> {
    let _active = weights.is_active();
    Err(Refusal::MissingInput(PF90034_SHADOW_FIELD_OPEN))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contracts_cover_each_row_once() {
        let mut pfs: Vec<u32> = CONTRACTS.iter().map(|row| row.pf).collect();
        pfs.sort_unstable();
        assert_eq!(pfs, [8, 9, 101, 403, 30188, 30189, 30209, 90001, 90003, 90034, 90087, 100005]);
        for row in CONTRACTS {
            assert_eq!(contract(row.pf), Some(row));
        }
    }

    fn synthetic_points194() -> Vec<[f32; 2]> {
        (0..194)
            .map(|i| [100.0 + i as f32 * 2.0, 200.0 + (i % 7) as f32 * 3.0])
            .collect()
    }

    #[test]
    fn hairline_row_weights_and_offsets_match_recovered_table() {
        assert_eq!(
            HAIRLINE_ROW_WEIGHTS,
            [
                0.0,
                0.20000000298023224,
                0.6000000238418579,
                -0.20000004768371582,
                -0.40000003576278687,
            ]
        );
        assert_eq!(
            HAIRLINE_LANDMARK_OFFSETS,
            [
                0x398, 0x548, 0x538, 0x528, 0x518, 0x508, 0x4F8, 0x4E8, 0x4D8, 0x4C8, 0x4B8,
                0x4A8, 0x498, 0x490,
            ]
        );
        assert_eq!(
            HAIRLINE_SLOT_ORDER,
            [
                1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21,
                20, 23, 22, 25, 24, 26,
            ]
        );
        assert_eq!(HAIRLINE_LATTICE_VERTICES, 136);
    }

    #[test]
    fn hairline_part_line_uses_slot_order_anchor_and_136_vertices() {
        let points = synthetic_points194();
        let flat: Vec<f32> = points.iter().flat_map(|p| [p[0], p[1]]).collect();
        let line = hairline_part_line(&points).unwrap();
        assert_eq!(line.source_slots.len(), 27);
        assert_eq!(line.lattice.len(), 136);
        assert!(line.source_slots.iter().flatten().all(|v| v.is_finite()));
        assert!(line.lattice.iter().flatten().all(|v| v.is_finite()));
        let at = |byte_offset: usize| -> [f32; 2] {
            [flat[byte_offset / 4], flat[byte_offset / 4 + 1]]
        };
        // Source slots are stored in raw block order.
        assert_eq!(line.source_slots[0], at(0x398));
        assert_eq!(line.source_slots[1], at(0x398 + 8));
        assert_eq!(line.source_slots[2], at(0x548));
        assert_eq!(line.source_slots[26], at(0x490));
        // The lattice consumes them via TYPE1012_SLOT_ORDER, and row 0 is the
        // source slot itself. Slot order [1,0,...] means lattice[0] takes
        // raw slot 1 (0x398+8) and lattice[1] takes raw slot 0 (0x398).
        assert_eq!(HAIRLINE_SLOT_ORDER[0], 1);
        assert_eq!(HAIRLINE_SLOT_ORDER[1], 0);
        assert_eq!(line.lattice[0], at(0x398 + 8));
        assert_eq!(line.lattice[1], at(0x398));
        // Final vertex is overwritten with the recovered anchor landmark.
        assert_eq!(line.lattice[135], at(0x1B0));
        // Row 1 retains the recovered 0.2 weight verbatim.
        let source = at(0x398 + 8);
        let anchor = at(0x1B0);
        let expected = [
            source[0] + 0.20000000298023224 * (anchor[0] - source[0]) * 1.2,
            source[1] + 0.20000000298023224 * (anchor[1] - source[1]) * 1.2,
        ];
        assert_eq!(line.lattice[27], expected);
    }

    #[test]
    fn hairline_part_line_rejects_bad_packets() {
        assert_eq!(
            hairline_part_line(&[[0.0f32; 2]; 193]),
            Err(Refusal::ShapeMismatch("PF90087_part_line_194_landmarks"))
        );
        let mut points = synthetic_points194();
        points[10][0] = f32::NAN;
        assert_eq!(
            hairline_part_line(&points),
            Err(Refusal::ValueOutOfRange("PF90087_part_line_194_landmarks"))
        );
        // Anchor and reference must be distinct (the recovered kernel checks
        // the direction norm before consuming the 135 weights).
        let mut points = synthetic_points194();
        let anchor_point = HAIRLINE_ANCHOR_OFFSET / 8;
        let reference_point = HAIRLINE_REFERENCE_OFFSET / 8;
        points[anchor_point] = points[reference_point];
        assert_eq!(
            hairline_part_line(&points),
            Err(Refusal::ValueOutOfRange("PF90087_anchor_reference"))
        );
    }

    #[test]
    fn hairline_part_line_mask_is_finite_nonzero_and_validates_frame() {
        let points = synthetic_points194();
        let mask = hairline_part_line_mask(&points, 512, 512, 4.0).unwrap();
        assert_eq!(mask.len(), 512 * 512);
        assert!(mask.iter().any(|&v| v != 0));
        assert!(mask.iter().all(|&v| v == 0 || v == 255));
        assert_eq!(
            hairline_part_line_mask(&points, 0, 16, 4.0),
            Err(Refusal::ValueOutOfRange("PF90087_PART_LINE_BAND"))
        );
        assert_eq!(
            hairline_part_line_mask(&points, 16, 16, 0.0),
            Err(Refusal::ValueOutOfRange("PF90087_PART_LINE_BAND"))
        );
        assert_eq!(
            hairline_part_line_mask(&points, 16, 16, f32::NAN),
            Err(Refusal::ValueOutOfRange("PF90087_PART_LINE_BAND"))
        );
    }

    #[test]
    fn hair_seam_provider_fails_closed_without_native_hair_mask() {
        let points = synthetic_points194();
        let weights = [1.0f32; 135];
        let parts = [face_detail::HairSeamPartDegree { part: 1, degree: 0.5 }];
        assert_eq!(
            hair_seam_provider(HairSeamSameJobInputs {
                points194: &points,
                weights135: &weights,
                parts: &parts,
                hair_mask: None,
            }),
            Err(Refusal::MissingInput("PF90087_HAIR_MASK"))
        );
        // Even with a native mask plane, the repair model / binding is absent.
        let mask = [255u8; 16];
        assert_eq!(
            hair_seam_provider(HairSeamSameJobInputs {
                points194: &points,
                weights135: &weights,
                parts: &parts,
                hair_mask: Some(&mask),
            }),
            Err(Refusal::MissingInput(PF90087_HAIRLINE_GEOMETRY_OPEN))
        );
        // Malformed geometry never reaches the missing-mask check.
        assert_eq!(
            hair_seam_provider(HairSeamSameJobInputs {
                points194: &points[..193],
                weights135: &weights,
                parts: &parts,
                hair_mask: None,
            }),
            Err(Refusal::ShapeMismatch("PF90087_part_line_194_landmarks"))
        );
        assert_eq!(
            hair_seam_provider(HairSeamSameJobInputs {
                points194: &points,
                weights135: &weights[..134],
                parts: &parts,
                hair_mask: None,
            }),
            Err(Refusal::ShapeMismatch("PF135_HAIRLINE_WEIGHTS"))
        );
    }

    #[test]
    fn pf8_pf9_and_pf90034_are_evaluation_boundaries_only() {
        for pf in [8, 9] {
            assert_eq!(
                face_stereo_host_mask_boundary(pf),
                Err(Refusal::MissingInput(FACE_STEREO_HOST_MASK_OPEN))
            );
        }
        assert_eq!(
            face_stereo_host_mask_boundary(90034),
            Err(Refusal::UnknownEffect(90034))
        );
        for (geometry, shadow) in [(0.5f32, 0.4f32), (0.0, 0.0)] {
            assert_eq!(
                double_chin_shadow_boundary(DoubleChinWeights { geometry, shadow }),
                Err(Refusal::MissingInput(PF90034_SHADOW_FIELD_OPEN))
            );
        }
        assert_eq!(contract(8).unwrap().status, ProviderStatus::LiveOnly);
        assert_eq!(contract(9).unwrap().status, ProviderStatus::LiveOnly);
        assert_eq!(contract(90034).unwrap().status, ProviderStatus::TypedProducer);
        assert_eq!(contract(90087).unwrap().status, ProviderStatus::TypedProducer);
    }

    #[test]
    fn double_chin_weights_preserve_recovered_product() {
        assert_eq!(
            double_chin_weights(0.5, 0.4, 0.2),
            Ok(DoubleChinWeights {
                geometry: 0.2,
                shadow: 0.1,
            })
        );
        assert_eq!(
            double_chin_weights(f32::NAN, 0.4, 0.2),
            Err(Refusal::ValueOutOfRange("double chin native values"))
        );
    }

    #[test]
    fn pf403_uses_the_recovered_mask_kernel() {
        let mask = Image::solid(4, 4, [32, 64, 128, 255]);
        let out = enhance_breast_mask(&mask, 0.0).unwrap();
        assert_eq!((out.width, out.height), (4, 4));
        assert_eq!(out.pixels.len(), mask.pixels.len());
    }

    #[test]
    fn pf90003_does_not_use_the_explicit_target_shortcut() {
        let source = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120];
        let mask = vec![255; 4];
        assert_eq!(
            body_skin_uniform_contract(&source, 2, 2, &mask, &mask, 50.0, 0.5),
            Err(Refusal::MissingInput(PF90003_PIXEL_LAW_OPEN))
        );
        assert_eq!(
            body_skin_uniform_contract(&source, 2, 2, &mask, &mask, 0.0, f32::NAN),
            Ok(source)
        );
    }

    #[test]
    fn live_and_end_static_rows_refuse_after_shape_checks() {
        assert!(matches!(
            silkworm(Pf101SameJobInputs {
                eye_mask: &[1],
                eye_mask_type: 0,
                face_blur_factor: 1.0,
                lut: &[1],
            }),
            Err(Refusal::MissingInput(PF101_LIVE_ONLY))
        ));
        let packet = [[0.0f32; 2]; JAWLINE_POINTS];
        assert!(matches!(
            jawline(Pf30209SameJobInputs {
                face_194_packet: &packet,
                jawline_indices: &[0],
                mask: &[1],
                model_output: &[0.0],
                per_face_pf30209: &[1.0],
            }),
            Err(Refusal::MissingInput(PF30209_END_STATIC))
        ));
        assert_eq!(
            jawline(Pf30209SameJobInputs {
                face_194_packet: &packet,
                jawline_indices: &[0],
                mask: &[1],
                model_output: &[0.0],
                per_face_pf30209: &[],
            }),
            Err(Refusal::MissingInput("PF30209 per_face_pf30209"))
        );
    }

    #[test]
    fn pf30209_activation_is_a_boolean_any_greater_than_zero() {
        assert!(!jawline_active(&[]));
        assert!(!jawline_active(&[0.0, -1.0, -0.0]));
        assert!(jawline_active(&[0.0, f32::MIN_POSITIVE]));
        assert!(jawline_active(&[1e-9, 0.0]));
        // Magnitude is not a multiplier: 1e-9 activates exactly like 100.0.
        assert_eq!(jawline_active(&[1e-9]), jawline_active(&[100.0]));
    }

    #[test]
    fn pf30209_window_is_center_54_plus_109_through_141() {
        assert_eq!(JAWLINE_WINDOW_LEN, 33);
        let mut points = [[0.0f32; 2]; JAWLINE_POINTS];
        for (index, point) in points.iter_mut().enumerate() {
            *point = [index as f32, -(index as f32)];
        }
        let window = jawline_window(&points).unwrap();
        assert_eq!(window.center, [54.0, -54.0]);
        assert_eq!(window.contour.len(), 33);
        assert_eq!(window.contour[0], [109.0, -109.0]);
        assert_eq!(window.contour[32], [141.0, -141.0]);
        // Neighbours outside the fixed window are not included.
        assert!(!window.contour.contains(&points[108]));
        assert!(!window.contour.contains(&points[142]));
    }

    #[test]
    fn pf30209_window_refuses_missing_or_malformed_packets() {
        assert_eq!(
            jawline_window(&[]),
            Err(Refusal::MissingInput("PF30209 face_194_packet"))
        );
        assert_eq!(
            jawline_window(&[[0.0; 2]; JAWLINE_POINTS - 1]),
            Err(Refusal::ShapeMismatch("PF30209 face_194_packet"))
        );
        let mut points = [[0.0f32; 2]; JAWLINE_POINTS];
        points[JAWLINE_WINDOW_END][1] = f32::NAN;
        assert_eq!(
            jawline_window(&points),
            Err(Refusal::ValueOutOfRange("PF30209 face_194_packet"))
        );
    }

    #[test]
    fn pf30209_stays_outside_the_double_chin_producer() {
        // stage+0x3A0/+0x3B8 are not forwarded by 0x7AF2FD0, so 30209 is
        // END_STATIC while the 30188/30189 producers remain typed producers.
        assert_eq!(contract(30209).unwrap().status, ProviderStatus::EndStatic);
        assert_eq!(contract(30188).unwrap().status, ProviderStatus::TypedProducer);
        assert_eq!(contract(30189).unwrap().status, ProviderStatus::TypedProducer);
    }

    #[test]
    fn provider_weights_are_the_kernel_products_and_share_the_gate() {
        let weights = double_chin_weights(0.5, 0.4, 0.2).unwrap();
        assert_eq!(weights, double_chin::Weights { geometry: 0.2, shadow: 0.1 });
        assert!(weights.is_active());
        assert!(!double_chin_weights(0.0, 1.0, 1.0).unwrap().is_active());
    }

    #[test]
    fn pf100005_provider_names_the_missing_final_input() {
        let source = crate::Image::solid(1, 1, [100, 110, 120, 130]);
        let err = eye_white_purify_provider(crate::kernels::face_detail::EyeWhiteSameJobInputs {
            source: &source,
            model_planes: None,
            group_a: None,
            group_b: None,
            ui: Some(50.0),
            model_sha256: None,
            frame_sha256: None,
            receipt: None,
        })
        .unwrap_err();
        assert_eq!(err, Refusal::MissingInput("PF100005_MODEL_SHA256"));
    }
}
