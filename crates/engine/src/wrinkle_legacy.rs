use aurora_portrait::kernels::skin_adjust::{Roi, WrinkleUniforms};
use aurora_portrait::{Image, Refusal};

use crate::portrait_stage::PortraitJobReceipt;

pub const WRINKLE_LEGACY_SKIN_MASK_SERVICE_OFFSET: u32 = 0x6200;
pub const WRINKLE_LEGACY_SKIN_MASK_TEXTURE_INDEX: u32 = 0;
pub const WRINKLE_LEGACY_SKIN_MASK_FILTER_SLOT: u32 = 1;
pub const WRINKLE_FACE_ROI_WRITER_RVA: u32 = 0x60bfd80;
pub const WRINKLE_MASK_ROI_SIDE: u32 = 1000;
pub const WRINKLE_LANDMARK_194_COUNT: usize = 194;
pub const WRINKLE_LANDMARK_56_COUNT: usize = 56;
pub const WRINKLE_SHADER_UPLOAD_REFUSAL: &str = "wrinkle shader uniform upload";
pub const WRINKLE_FACEPARSING_MASK_MAPPING_REFUSAL: &str = "FaceParsingV2 to wrinkle mask mapping";

/// The renderer receipt that makes all wrinkle planes, ROI and uniforms belong
/// to one immutable portrait job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrinkleSameJobReceipt {
    job_id: u64,
}

impl WrinkleSameJobReceipt {
    pub fn from_renderer_receipt(receipt: &PortraitJobReceipt) -> Self {
        Self {
            job_id: receipt.job_id(),
        }
    }

    pub fn from_job_id(job_id: u64) -> Self {
        Self { job_id }
    }

    pub fn job_id(self) -> u64 {
        self.job_id
    }

    #[cfg(test)]
    pub(crate) fn from_test_job_id(job_id: u64) -> Self {
        Self { job_id }
    }
}

/// Ten native `WrinklesV1Params` result values retained with their job id.
///
/// The result-to-shader-name join and upload point are not established, so
/// this receipt intentionally has no conversion into [`WrinkleUniforms`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WrinkleV1ParamsResultReceipt {
    job_id: u64,
    results: [f32; 10],
}

impl WrinkleV1ParamsResultReceipt {
    pub fn from_same_job(
        receipt: WrinkleSameJobReceipt,
        results: [f32; 10],
    ) -> Result<Self, Refusal> {
        if receipt.job_id() == 0 {
            return Err(Refusal::MissingInput("WrinklesV1Params receipt"));
        }
        if results.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("WrinklesV1Params result"));
        }
        Ok(Self {
            job_id: receipt.job_id(),
            results,
        })
    }

    pub fn job_id(self) -> u64 {
        self.job_id
    }

    pub fn results(self) -> [f32; 10] {
        self.results
    }

    pub fn result_offsets(self) -> [u32; 10] {
        WRINKLE_V1_PARAM_SLOTS.map(|slot| slot.result_offset)
    }

    pub fn to_shader_uniforms(self) -> Result<WrinkleUniforms, Refusal> {
        Err(Refusal::MissingInput(WRINKLE_SHADER_UPLOAD_REFUSAL))
    }
}

pub fn faceparsing_v2_wrinkle_masks_refusal() -> Refusal {
    Refusal::MissingInput(WRINKLE_FACEPARSING_MASK_MAPPING_REFUSAL)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WrinkleV1ParamSlot {
    pub param_offset: u32,
    pub setter_rva: u32,
    pub default: f32,
    pub result_offset: u32,
}

/// The ten `WrinklesV1Params` vectors in native parameter order.
///
/// No row is joined to a shader name or upload point here.
pub const WRINKLE_V1_PARAM_SLOTS: [WrinkleV1ParamSlot; 10] = [
    WrinkleV1ParamSlot {
        param_offset: 0x50,
        setter_rva: 0x5bb05b0,
        default: 0.55,
        result_offset: 0x200,
    },
    WrinkleV1ParamSlot {
        param_offset: 0x90,
        setter_rva: 0x5bb06e0,
        default: 0.55,
        result_offset: 0x204,
    },
    WrinkleV1ParamSlot {
        param_offset: 0xd0,
        setter_rva: 0x5bb09a0,
        default: 0.75,
        result_offset: 0x1fc,
    },
    WrinkleV1ParamSlot {
        param_offset: 0x110,
        setter_rva: 0x5bb0c60,
        default: 0.0,
        result_offset: 0x208,
    },
    WrinkleV1ParamSlot {
        param_offset: 0x150,
        setter_rva: 0x5bb0f20,
        default: 0.0,
        result_offset: 0x20c,
    },
    WrinkleV1ParamSlot {
        param_offset: 0x190,
        setter_rva: 0x5bb1ce0,
        default: 0.0,
        result_offset: 0x220,
    },
    WrinkleV1ParamSlot {
        param_offset: 0x1d0,
        setter_rva: 0x5bb11e0,
        default: 0.1,
        result_offset: 0x210,
    },
    WrinkleV1ParamSlot {
        param_offset: 0x210,
        setter_rva: 0x5bb14a0,
        default: 0.9,
        result_offset: 0x214,
    },
    WrinkleV1ParamSlot {
        param_offset: 0x250,
        setter_rva: 0x5bb1760,
        default: 0.0,
        result_offset: 0x218,
    },
    WrinkleV1ParamSlot {
        param_offset: 0x290,
        setter_rva: 0x5bb1a20,
        default: 0.0,
        result_offset: 0x21c,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrinkleShaderSlot {
    pub uniform: &'static str,
    pub lb_f_index: u8,
}

/// Static shader/C# contract only. The native result-to-name join is unproven.
pub const WRINKLE_V1_SHADER_SLOTS: [WrinkleShaderSlot; 10] = [
    WrinkleShaderSlot {
        uniform: "u_legal_alpha",
        lb_f_index: 0,
    },
    WrinkleShaderSlot {
        uniform: "u_dark_circle_alpha",
        lb_f_index: 1,
    },
    WrinkleShaderSlot {
        uniform: "u_diff_minthreshold",
        lb_f_index: 2,
    },
    WrinkleShaderSlot {
        uniform: "u_diff_maxthreshold",
        lb_f_index: 3,
    },
    WrinkleShaderSlot {
        uniform: "u_diff_factor",
        lb_f_index: 4,
    },
    WrinkleShaderSlot {
        uniform: "u_mix_alpha",
        lb_f_index: 5,
    },
    WrinkleShaderSlot {
        uniform: "u_high_freq_minthreshold",
        lb_f_index: 6,
    },
    WrinkleShaderSlot {
        uniform: "u_high_freq_maxthreshold",
        lb_f_index: 7,
    },
    WrinkleShaderSlot {
        uniform: "u_diff_factor2",
        lb_f_index: 9,
    },
    WrinkleShaderSlot {
        uniform: "u_color_thresh_factor",
        lb_f_index: 10,
    },
];

/// One captured legacy `service+0x6200` texture for one face.
///
/// Fields stay private so this type cannot be constructed as FaceParsingV2
/// `service+0x6230` data or with a different texture index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrinkleLegacySkinMaskTexture {
    face_id: u32,
    width: u32,
    height: u32,
    rgba8: Vec<u8>,
}

impl WrinkleLegacySkinMaskTexture {
    pub fn from_service_6200(
        face_id: u32,
        width: u32,
        height: u32,
        rgba8: Vec<u8>,
    ) -> Result<Self, Refusal> {
        if face_id == 0 {
            return Err(Refusal::MissingInput("wrinkle legacy skinmask face"));
        }
        if width == 0 || height == 0 {
            return Err(Refusal::ShapeMismatch("wrinkle legacy skinmask texture"));
        }
        let pixels = checked_pixels(width, height, "wrinkle legacy skinmask texture")?;
        if rgba8.len() != pixels * 4 {
            return Err(Refusal::ShapeMismatch("wrinkle legacy skinmask texture"));
        }
        Ok(Self {
            face_id,
            width,
            height,
            rgba8,
        })
    }

    pub fn face_id(&self) -> u32 {
        self.face_id
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn rgba8(&self) -> &[u8] {
        &self.rgba8
    }

    pub fn service_offset(&self) -> u32 {
        WRINKLE_LEGACY_SKIN_MASK_SERVICE_OFFSET
    }

    pub fn texture_index(&self) -> u32 {
        WRINKLE_LEGACY_SKIN_MASK_TEXTURE_INDEX
    }

    pub fn filter_slot(&self) -> u32 {
        WRINKLE_LEGACY_SKIN_MASK_FILTER_SLOT
    }
}

/// Explicit full-frame `.r` projection for the legacy `+0x6200` capture.
///
/// The host owns the texture-to-frame projection. This producer never derives
/// the full-frame bytes from a FaceParsing plane or a UI skin mask.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrinkleLegacySkinMaskProducer {
    ticket: u64,
    frame_width: u32,
    frame_height: u32,
    source: WrinkleLegacySkinMaskTexture,
    skin_mask_r_full: Vec<u8>,
}

impl WrinkleLegacySkinMaskProducer {
    pub fn from_same_job(
        ticket: u64,
        frame_width: u32,
        frame_height: u32,
        source: WrinkleLegacySkinMaskTexture,
        skin_mask_r_full: Vec<u8>,
    ) -> Result<Self, Refusal> {
        let producer = Self {
            ticket,
            frame_width,
            frame_height,
            source,
            skin_mask_r_full,
        };
        producer.validate()?;
        Ok(producer)
    }

    fn validate(&self) -> Result<(), Refusal> {
        if self.ticket == 0 {
            return Err(Refusal::MissingInput("wrinkle legacy skinmask ticket"));
        }
        let pixels = checked_pixels(
            self.frame_width,
            self.frame_height,
            "wrinkle legacy skinmask frame",
        )?;
        if self.skin_mask_r_full.len() != pixels {
            return Err(Refusal::ShapeMismatch(
                "wrinkle legacy skinmask frame plane",
            ));
        }
        Ok(())
    }

    pub fn ticket(&self) -> u64 {
        self.ticket
    }

    pub fn frame_width(&self) -> u32 {
        self.frame_width
    }

    pub fn frame_height(&self) -> u32 {
        self.frame_height
    }

    pub fn source(&self) -> &WrinkleLegacySkinMaskTexture {
        &self.source
    }

    pub fn skin_mask_r_full(&self) -> &[u8] {
        &self.skin_mask_r_full
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WrinkleMaskMiniImage {
    face_id: u32,
    rgba8: Vec<u8>,
}

impl WrinkleMaskMiniImage {
    pub fn from_capture(face_id: u32, rgba8: Vec<u8>) -> Result<Self, Refusal> {
        if face_id == 0 {
            return Err(Refusal::MissingInput("wrinkle mask mini-image face"));
        }
        let pixels = checked_pixels(
            WRINKLE_MASK_ROI_SIDE,
            WRINKLE_MASK_ROI_SIDE,
            "wrinkle mask mini-image",
        )?;
        if rgba8.len() != pixels * 4 {
            return Err(Refusal::ShapeMismatch("wrinkle mask mini-image"));
        }
        Ok(Self { face_id, rgba8 })
    }

    pub fn face_id(&self) -> u32 {
        self.face_id
    }

    pub fn width(&self) -> u32 {
        WRINKLE_MASK_ROI_SIDE
    }

    pub fn height(&self) -> u32 {
        WRINKLE_MASK_ROI_SIDE
    }

    pub fn rgba8(&self) -> &[u8] {
        &self.rgba8
    }
}

/// The `face+0x1638` output and both landmark arrays consumed by
/// `0x60bfd80`. `writer_rect` is the rounded, clamped union of the source rect
/// and all 250 landmarks.
#[derive(Clone, Debug, PartialEq)]
pub struct WrinkleFaceRoiContract {
    face_id: u32,
    valid: bool,
    source_width: u32,
    source_height: u32,
    rect_xywh: [f32; 4],
    quality: f32,
    landmarks_194: Vec<[f32; 2]>,
    landmarks_56: Vec<[f32; 2]>,
    writer_rect: [u32; 4],
}

impl WrinkleFaceRoiContract {
    #[allow(clippy::too_many_arguments)]
    pub fn from_face_record(
        face_id: u32,
        valid: bool,
        source_width: u32,
        source_height: u32,
        rect_xywh: [f32; 4],
        quality: f32,
        landmarks_194: Vec<[f32; 2]>,
        landmarks_56: Vec<[f32; 2]>,
    ) -> Result<Self, Refusal> {
        if face_id == 0 {
            return Err(Refusal::MissingInput("wrinkle face roi face"));
        }
        if !valid {
            return Err(Refusal::MissingInput("wrinkle face roi valid"));
        }
        if source_width == 0 || source_height == 0 {
            return Err(Refusal::ShapeMismatch("wrinkle face roi source"));
        }
        if landmarks_194.len() != WRINKLE_LANDMARK_194_COUNT
            || landmarks_56.len() != WRINKLE_LANDMARK_56_COUNT
        {
            return Err(Refusal::ShapeMismatch("wrinkle face roi landmarks"));
        }
        if !rect_xywh.iter().all(|value| value.is_finite())
            || rect_xywh[2] <= 0.0
            || rect_xywh[3] <= 0.0
        {
            return Err(Refusal::ValueOutOfRange("wrinkle face roi rect"));
        }
        if !quality.is_finite() || !(0.0..=1.0).contains(&quality) {
            return Err(Refusal::ValueOutOfRange("wrinkle face roi quality"));
        }
        if landmarks_194
            .iter()
            .chain(&landmarks_56)
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("wrinkle face roi landmarks"));
        }
        let writer_rect = rounded_writer_rect(
            source_width,
            source_height,
            rect_xywh,
            landmarks_194.iter().chain(&landmarks_56),
        )?;
        let declared = rounded_rect(source_width, source_height, rect_xywh)?;
        if writer_rect != declared {
            return Err(Refusal::ValueOutOfRange("wrinkle face roi writer union"));
        }
        Ok(Self {
            face_id,
            valid,
            source_width,
            source_height,
            rect_xywh,
            quality,
            landmarks_194,
            landmarks_56,
            writer_rect,
        })
    }

    pub fn face_id(&self) -> u32 {
        self.face_id
    }

    pub fn valid(&self) -> bool {
        self.valid
    }

    pub fn source_width(&self) -> u32 {
        self.source_width
    }

    pub fn source_height(&self) -> u32 {
        self.source_height
    }

    pub fn rect_xywh(&self) -> [f32; 4] {
        self.rect_xywh
    }

    pub fn quality(&self) -> f32 {
        self.quality
    }

    pub fn landmarks_194(&self) -> &[[f32; 2]] {
        &self.landmarks_194
    }

    pub fn landmarks_56(&self) -> &[[f32; 2]] {
        &self.landmarks_56
    }

    pub fn writer_rva(&self) -> u32 {
        WRINKLE_FACE_ROI_WRITER_RVA
    }

    pub fn stage_roi(&self) -> Roi {
        Roi {
            x: self.writer_rect[0],
            y: self.writer_rect[1],
            width: self.writer_rect[2],
            height: self.writer_rect[3],
        }
    }
}

/// One face's `h` mini-image, ROI contract and the explicit full-frame RGBA
/// output of the 1000-square resample/contour pass.
#[derive(Clone, Debug, PartialEq)]
pub struct WrinkleMaskMiniImageProducer {
    ticket: u64,
    frame_width: u32,
    frame_height: u32,
    mini_image: WrinkleMaskMiniImage,
    face_roi: WrinkleFaceRoiContract,
    mask_rgba_full: Vec<u8>,
}

impl WrinkleMaskMiniImageProducer {
    pub fn from_same_job(
        ticket: u64,
        frame_width: u32,
        frame_height: u32,
        mini_image: WrinkleMaskMiniImage,
        face_roi: WrinkleFaceRoiContract,
        mask_rgba_full: Vec<u8>,
    ) -> Result<Self, Refusal> {
        let producer = Self {
            ticket,
            frame_width,
            frame_height,
            mini_image,
            face_roi,
            mask_rgba_full,
        };
        producer.validate()?;
        Ok(producer)
    }

    fn validate(&self) -> Result<(), Refusal> {
        if self.ticket == 0 {
            return Err(Refusal::MissingInput("wrinkle mask mini-image ticket"));
        }
        let pixels = checked_pixels(
            self.frame_width,
            self.frame_height,
            "wrinkle mask mini-image frame",
        )?;
        if self.mask_rgba_full.len() != pixels * 4 {
            return Err(Refusal::ShapeMismatch(
                "wrinkle mask mini-image frame plane",
            ));
        }
        if self.mini_image.face_id() != self.face_roi.face_id() {
            return Err(Refusal::ValueOutOfRange("wrinkle mask mini-image face"));
        }
        Ok(())
    }

    pub fn ticket(&self) -> u64 {
        self.ticket
    }

    pub fn frame_width(&self) -> u32 {
        self.frame_width
    }

    pub fn frame_height(&self) -> u32 {
        self.frame_height
    }

    pub fn mini_image(&self) -> &WrinkleMaskMiniImage {
        &self.mini_image
    }

    pub fn face_roi(&self) -> &WrinkleFaceRoiContract {
        &self.face_roi
    }

    pub fn mask_rgba_full(&self) -> &[u8] {
        &self.mask_rgba_full
    }
}

/// Typed same-job producer for PF21025/21069/21070/21071.
///
/// Masks and ROI come only from the captured legacy `+0x6200` texture and the
/// per-face 1000-square mask path. Uniform values remain an explicit caller
/// input; there is no `WrinklesV1Params` default expansion or shader upload.
#[derive(Clone, Debug, PartialEq)]
pub struct WrinkleProducerInputs {
    pub(crate) ticket: u64,
    pub(crate) width: u32,
    pub(crate) height: u32,
    legacy_skin_mask: Option<WrinkleLegacySkinMaskProducer>,
    mask_mini_image: Option<WrinkleMaskMiniImageProducer>,
    pub(crate) base_full: Option<Image>,
    pub(crate) residual_full: Option<Image>,
    uniforms: Option<WrinkleUniforms>,
}

#[derive(Debug)]
pub(crate) struct ResolvedWrinkleInputs<'a> {
    pub base_full: &'a Image,
    pub residual_full: &'a Image,
    pub mask_rgba_full: &'a [u8],
    pub skin_mask_full: &'a [u8],
    pub roi: Roi,
    pub uniforms: WrinkleUniforms,
}

impl WrinkleProducerInputs {
    pub fn ticket(&self) -> u64 {
        self.ticket
    }

    pub fn frame_size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn from_same_job(
        receipt: WrinkleSameJobReceipt,
        width: u32,
        height: u32,
        base_full: Image,
        residual_full: Image,
        legacy_skin_mask: WrinkleLegacySkinMaskProducer,
        mask_mini_image: WrinkleMaskMiniImageProducer,
        uniforms: WrinkleUniforms,
    ) -> Result<Self, Refusal> {
        let ticket = receipt.job_id();
        let producer = Self {
            ticket,
            width,
            height,
            legacy_skin_mask: Some(legacy_skin_mask),
            mask_mini_image: Some(mask_mini_image),
            base_full: Some(base_full),
            residual_full: Some(residual_full),
            uniforms: Some(uniforms),
        };
        producer.validate_shape()?;
        Ok(producer)
    }

    fn validate_shape(&self) -> Result<(), Refusal> {
        if self.ticket == 0 {
            return Err(Refusal::MissingInput("wrinkle producer ticket"));
        }
        let pixels = checked_pixels(self.width, self.height, "wrinkle producer frame")?;
        let base = self
            .base_full
            .as_ref()
            .ok_or(Refusal::MissingInput("wrinkle base_full"))?;
        let residual = self
            .residual_full
            .as_ref()
            .ok_or(Refusal::MissingInput("wrinkle residual_full"))?;
        if (base.width, base.height) != (self.width, self.height) {
            return Err(Refusal::ShapeMismatch("wrinkle base_full"));
        }
        if (residual.width, residual.height) != (self.width, self.height) {
            return Err(Refusal::ShapeMismatch("wrinkle residual_full"));
        }
        if base.pixels.len() != pixels * 4 || residual.pixels.len() != pixels * 4 {
            return Err(Refusal::ShapeMismatch("wrinkle producer frame"));
        }

        let legacy = self
            .legacy_skin_mask
            .as_ref()
            .ok_or(Refusal::MissingInput("wrinkle legacy skinmask"))?;
        legacy.validate()?;
        if legacy.ticket() != self.ticket {
            return Err(Refusal::ValueOutOfRange("wrinkle legacy skinmask ticket"));
        }
        if (legacy.frame_width(), legacy.frame_height()) != (self.width, self.height) {
            return Err(Refusal::ShapeMismatch("wrinkle legacy skinmask frame"));
        }

        let mask = self
            .mask_mini_image
            .as_ref()
            .ok_or(Refusal::MissingInput("wrinkle mask mini-image"))?;
        mask.validate()?;
        if mask.ticket() != self.ticket {
            return Err(Refusal::ValueOutOfRange("wrinkle mask mini-image ticket"));
        }
        if (mask.frame_width(), mask.frame_height()) != (self.width, self.height) {
            return Err(Refusal::ShapeMismatch("wrinkle mask mini-image frame"));
        }
        if legacy.source().face_id() != mask.face_roi().face_id() {
            return Err(Refusal::ValueOutOfRange("wrinkle producer face"));
        }
        if (
            mask.face_roi().source_width(),
            mask.face_roi().source_height(),
        ) != (self.width, self.height)
        {
            return Err(Refusal::ShapeMismatch("wrinkle face roi source"));
        }
        let uniforms = self
            .uniforms
            .ok_or(Refusal::MissingInput("wrinkle uniforms"))?;
        validate_wrinkle_uniforms(uniforms)?;
        Ok(())
    }

    pub(crate) fn resolve<'a>(
        &'a self,
        current: &Image,
        job_id: Option<u64>,
    ) -> Result<ResolvedWrinkleInputs<'a>, Refusal> {
        self.validate_shape()?;
        let job_id = job_id.ok_or(Refusal::MissingInput("renderer receipt"))?;
        if self.ticket != job_id {
            return Err(Refusal::ValueOutOfRange("wrinkle producer ticket"));
        }
        if (self.width, self.height) != (current.width, current.height) {
            return Err(Refusal::ShapeMismatch("wrinkle producer frame"));
        }
        let legacy = self
            .legacy_skin_mask
            .as_ref()
            .ok_or(Refusal::MissingInput("wrinkle legacy skinmask"))?;
        let mask = self
            .mask_mini_image
            .as_ref()
            .ok_or(Refusal::MissingInput("wrinkle mask mini-image"))?;
        Ok(ResolvedWrinkleInputs {
            base_full: self
                .base_full
                .as_ref()
                .ok_or(Refusal::MissingInput("wrinkle base_full"))?,
            residual_full: self
                .residual_full
                .as_ref()
                .ok_or(Refusal::MissingInput("wrinkle residual_full"))?,
            mask_rgba_full: mask.mask_rgba_full(),
            skin_mask_full: legacy.skin_mask_r_full(),
            roi: mask.face_roi().stage_roi(),
            uniforms: self
                .uniforms
                .ok_or(Refusal::MissingInput("wrinkle uniforms"))?,
        })
    }

    pub fn fnv_fingerprint(&self, seed: u64) -> u64 {
        let mut hash = seed;
        hash = hash_u64(hash, u64::from(self.legacy_skin_mask.is_some()));
        if let Some(legacy) = self.legacy_skin_mask.as_ref() {
            hash = hash_u64(hash, legacy.ticket());
            hash = hash_u64(hash, u64::from(legacy.frame_width()));
            hash = hash_u64(hash, u64::from(legacy.frame_height()));
            hash = hash_u64(hash, u64::from(legacy.source().face_id()));
            hash = hash_u64(hash, u64::from(legacy.source().width()));
            hash = hash_u64(hash, u64::from(legacy.source().height()));
            hash = hash_u64(hash, u64::from(WRINKLE_LEGACY_SKIN_MASK_SERVICE_OFFSET));
            hash = hash_bytes(hash, legacy.source().rgba8());
            hash = hash_bytes(hash, legacy.skin_mask_r_full());
        }
        hash = hash_u64(hash, u64::from(self.mask_mini_image.is_some()));
        if let Some(mask) = self.mask_mini_image.as_ref() {
            hash = hash_u64(hash, mask.ticket());
            hash = hash_u64(hash, u64::from(mask.frame_width()));
            hash = hash_u64(hash, u64::from(mask.frame_height()));
            let roi = mask.face_roi();
            hash = hash_u64(hash, u64::from(roi.face_id()));
            hash = hash_u64(hash, u64::from(roi.source_width()));
            hash = hash_u64(hash, u64::from(roi.source_height()));
            for value in roi.rect_xywh() {
                hash = hash_u64(hash, value.to_bits() as u64);
            }
            hash = hash_u64(hash, roi.quality().to_bits() as u64);
            for point in roi.landmarks_194().iter().chain(roi.landmarks_56()) {
                for value in point {
                    hash = hash_u64(hash, value.to_bits() as u64);
                }
            }
            hash = hash_bytes(hash, mask.mini_image().rgba8());
            hash = hash_bytes(hash, mask.mask_rgba_full());
        }
        hash = match self.uniforms {
            None => hash_u64(hash, 0),
            Some(uniforms) => {
                let mut hash = hash_u64(hash, 1);
                for value in wrinkle_uniform_values(uniforms) {
                    hash = hash_u64(hash, value.to_bits() as u64);
                }
                hash
            }
        };
        hash
    }
}

pub fn validate_wrinkle_uniforms(uniforms: WrinkleUniforms) -> Result<(), Refusal> {
    let values = wrinkle_uniform_values(uniforms);
    if !values.iter().all(|value| value.is_finite())
        || uniforms.diff_min_threshold > uniforms.diff_max_threshold
        || uniforms.high_freq_min_threshold > uniforms.high_freq_max_threshold
    {
        return Err(Refusal::ValueOutOfRange("wrinkle_uniforms"));
    }
    Ok(())
}

fn wrinkle_uniform_values(uniforms: WrinkleUniforms) -> [f32; 10] {
    [
        uniforms.legal_alpha,
        uniforms.dark_circle_alpha,
        uniforms.diff_factor,
        uniforms.diff_factor2,
        uniforms.color_threshold_factor,
        uniforms.mix_alpha,
        uniforms.diff_min_threshold,
        uniforms.diff_max_threshold,
        uniforms.high_freq_min_threshold,
        uniforms.high_freq_max_threshold,
    ]
}

fn rounded_writer_rect<'a>(
    source_width: u32,
    source_height: u32,
    rect: [f32; 4],
    landmarks: impl Iterator<Item = &'a [f32; 2]>,
) -> Result<[u32; 4], Refusal> {
    let mut min_x = rect[0];
    let mut min_y = rect[1];
    let mut max_x = rect[0] + rect[2];
    let mut max_y = rect[1] + rect[3];
    for point in landmarks {
        min_x = min_x.min(point[0]);
        min_y = min_y.min(point[1]);
        max_x = max_x.max(point[0]);
        max_y = max_y.max(point[1]);
    }
    let x0 = round_and_clamp(min_x, source_width);
    let y0 = round_and_clamp(min_y, source_height);
    let x1 = round_and_clamp(max_x, source_width);
    let y1 = round_and_clamp(max_y, source_height);
    if x1 <= x0 || y1 <= y0 {
        return Err(Refusal::ShapeMismatch("wrinkle face roi writer union"));
    }
    Ok([x0, y0, x1 - x0, y1 - y0])
}

fn rounded_rect(
    source_width: u32,
    source_height: u32,
    rect: [f32; 4],
) -> Result<[u32; 4], Refusal> {
    let x0 = round_and_clamp(rect[0], source_width);
    let y0 = round_and_clamp(rect[1], source_height);
    let x1 = round_and_clamp(rect[0] + rect[2], source_width);
    let y1 = round_and_clamp(rect[1] + rect[3], source_height);
    if x1 <= x0 || y1 <= y0 {
        return Err(Refusal::ShapeMismatch("wrinkle face roi rect"));
    }
    Ok([x0, y0, x1 - x0, y1 - y0])
}

fn round_and_clamp(value: f32, limit: u32) -> u32 {
    (value.max(0.0) + 0.5).floor().min(limit as f32) as u32
}

fn checked_pixels(width: u32, height: u32, name: &'static str) -> Result<usize, Refusal> {
    if width == 0 || height == 0 {
        return Err(Refusal::ShapeMismatch(name));
    }
    (width as usize)
        .checked_mul(height as usize)
        .ok_or(Refusal::ShapeMismatch(name))
}

fn hash_u64(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    hash = hash_u64(hash, bytes.len() as u64);
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniforms() -> WrinkleUniforms {
        WrinkleUniforms {
            legal_alpha: 0.55,
            dark_circle_alpha: 0.55,
            diff_factor: 0.75,
            diff_factor2: 0.0,
            color_threshold_factor: 0.0,
            mix_alpha: 0.1,
            diff_min_threshold: -1.0,
            diff_max_threshold: 1.0,
            high_freq_min_threshold: -1.0,
            high_freq_max_threshold: 1.0,
        }
    }

    fn legacy(face_id: u32, ticket: u64, width: u32, height: u32) -> WrinkleLegacySkinMaskProducer {
        WrinkleLegacySkinMaskProducer::from_same_job(
            ticket,
            width,
            height,
            WrinkleLegacySkinMaskTexture::from_service_6200(face_id, 2, 2, vec![17; 2 * 2 * 4])
                .unwrap(),
            vec![91; width as usize * height as usize],
        )
        .unwrap()
    }

    fn roi(face_id: u32, source_width: u32, source_height: u32) -> WrinkleFaceRoiContract {
        WrinkleFaceRoiContract::from_face_record(
            face_id,
            true,
            source_width,
            source_height,
            [1.0, 1.0, 3.0, 2.0],
            0.25,
            vec![[2.0, 2.0]; WRINKLE_LANDMARK_194_COUNT],
            vec![[2.0, 2.0]; WRINKLE_LANDMARK_56_COUNT],
        )
        .unwrap()
    }

    fn mask(face_id: u32, ticket: u64, width: u32, height: u32) -> WrinkleMaskMiniImageProducer {
        WrinkleMaskMiniImageProducer::from_same_job(
            ticket,
            width,
            height,
            WrinkleMaskMiniImage::from_capture(
                face_id,
                vec![37; WRINKLE_MASK_ROI_SIDE as usize * WRINKLE_MASK_ROI_SIDE as usize * 4],
            )
            .unwrap(),
            roi(face_id, width, height),
            vec![73; width as usize * height as usize * 4],
        )
        .unwrap()
    }

    fn image(width: u32, height: u32, seed: u8) -> Image {
        Image::new(
            width,
            height,
            vec![seed; width as usize * height as usize * 4],
        )
        .unwrap()
    }

    #[test]
    fn source_constants_are_legacy_6200_id0_not_v2_6230() {
        assert_eq!(WRINKLE_LEGACY_SKIN_MASK_SERVICE_OFFSET, 0x6200);
        assert_eq!(WRINKLE_LEGACY_SKIN_MASK_TEXTURE_INDEX, 0);
        assert_eq!(WRINKLE_LEGACY_SKIN_MASK_FILTER_SLOT, 1);
        assert_ne!(WRINKLE_LEGACY_SKIN_MASK_SERVICE_OFFSET, 0x6230);
    }

    #[test]
    fn face_roi_requires_both_landmark_sets_and_writer_union() {
        let missing = WrinkleFaceRoiContract::from_face_record(
            1,
            true,
            32,
            24,
            [1.0, 1.0, 3.0, 2.0],
            0.25,
            vec![[2.0, 2.0]; 193],
            vec![[2.0, 2.0]; 56],
        );
        assert_eq!(
            missing,
            Err(Refusal::ShapeMismatch("wrinkle face roi landmarks"))
        );

        let outside = WrinkleFaceRoiContract::from_face_record(
            1,
            true,
            32,
            24,
            [1.0, 1.0, 3.0, 2.0],
            0.25,
            vec![[8.0, 2.0]; 194],
            vec![[2.0, 2.0]; 56],
        );
        assert_eq!(
            outside,
            Err(Refusal::ValueOutOfRange("wrinkle face roi writer union"))
        );
        assert_eq!(roi(1, 32, 24).writer_rva(), WRINKLE_FACE_ROI_WRITER_RVA);
        assert_eq!(
            roi(1, 32, 24).stage_roi(),
            Roi {
                x: 1,
                y: 1,
                width: 3,
                height: 2,
            }
        );
    }

    #[test]
    fn mini_image_is_fixed_1000_square() {
        assert_eq!(
            WrinkleMaskMiniImage::from_capture(1, vec![0; 4]),
            Err(Refusal::ShapeMismatch("wrinkle mask mini-image"))
        );
    }

    #[test]
    fn typed_bundle_uses_explicit_full_frame_outputs() {
        let width = 5;
        let height = 4;
        let ticket = 48;
        let producer = WrinkleProducerInputs::from_same_job(
            WrinkleSameJobReceipt::from_test_job_id(ticket),
            width,
            height,
            image(width, height, 11),
            image(width, height, 12),
            legacy(7, ticket, width, height),
            mask(7, ticket, width, height),
            uniforms(),
        )
        .unwrap();
        let current = image(width, height, 13);
        let resolved = producer.resolve(&current, Some(ticket)).unwrap();
        assert_eq!(resolved.mask_rgba_full, vec![73; 5 * 4 * 4]);
        assert_eq!(resolved.skin_mask_full, vec![91; 5 * 4]);
        assert_eq!(
            resolved.roi,
            Roi {
                x: 1,
                y: 1,
                width: 3,
                height: 2,
            }
        );
    }

    #[test]
    fn typed_bundle_rejects_cross_job_face_and_missing_uniforms() {
        let width = 5;
        let height = 4;
        assert_eq!(
            WrinkleProducerInputs::from_same_job(
                WrinkleSameJobReceipt::from_test_job_id(48),
                width,
                height,
                image(width, height, 11),
                image(width, height, 12),
                legacy(7, 49, width, height),
                mask(7, 48, width, height),
                uniforms(),
            ),
            Err(Refusal::ValueOutOfRange("wrinkle legacy skinmask ticket"))
        );
        assert_eq!(
            WrinkleProducerInputs::from_same_job(
                WrinkleSameJobReceipt::from_test_job_id(48),
                width,
                height,
                image(width, height, 11),
                image(width, height, 12),
                legacy(7, 48, width, height),
                mask(8, 48, width, height),
                uniforms(),
            ),
            Err(Refusal::ValueOutOfRange("wrinkle producer face"))
        );

        let mut producer = WrinkleProducerInputs::from_same_job(
            WrinkleSameJobReceipt::from_test_job_id(48),
            width,
            height,
            image(width, height, 11),
            image(width, height, 12),
            legacy(7, 48, width, height),
            mask(7, 48, width, height),
            uniforms(),
        )
        .unwrap();
        producer.uniforms = None;
        assert_eq!(
            producer
                .resolve(&image(width, height, 13), Some(48))
                .unwrap_err(),
            Refusal::MissingInput("wrinkle uniforms")
        );
        assert_eq!(
            validate_wrinkle_uniforms(WrinkleUniforms {
                legal_alpha: f32::NAN,
                ..uniforms()
            }),
            Err(Refusal::ValueOutOfRange("wrinkle_uniforms"))
        );
        assert_eq!(
            WRINKLE_SHADER_UPLOAD_REFUSAL,
            "wrinkle shader uniform upload"
        );
        let receipt = WrinkleV1ParamsResultReceipt::from_same_job(
            WrinkleSameJobReceipt::from_test_job_id(48),
            [0.0; 10],
        )
        .unwrap();
        assert_eq!(
            receipt.result_offsets(),
            WRINKLE_V1_PARAM_SLOTS.map(|slot| slot.result_offset)
        );
        assert_eq!(
            receipt.to_shader_uniforms(),
            Err(Refusal::MissingInput(WRINKLE_SHADER_UPLOAD_REFUSAL))
        );
        assert_eq!(
            faceparsing_v2_wrinkle_masks_refusal(),
            Refusal::MissingInput(WRINKLE_FACEPARSING_MASK_MAPPING_REFUSAL)
        );
    }
}
