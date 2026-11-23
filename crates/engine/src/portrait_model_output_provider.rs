//! Same-job model-output provider for portrait producers.
//!
//! The provider is intentionally a binding layer, not another inference
//! implementation. It accepts one immutable set of session tensors and converts
//! them into the existing typed producer inputs only after every required
//! tensor has the same job, source, model/tensor identity, shape and digest.

use std::collections::BTreeMap;

use aurora_portrait::eye_bag_alignment::EyeBagModelTensors;
use aurora_portrait::kernels::face_detail::Rect as ToothRect;
use aurora_portrait::kernels::moisture::MaskChannel;
use aurora_portrait::kernels::plump_pipeline::SessionOutput;
use aurora_portrait::model_sha256::sha256;
use aurora_portrait::{Image, Refusal};

use crate::hairfly_selector::{
    model_for_selector, resolve_hairfly_producer, selector_argmax, HairFlyNativeFaceReceipt,
    HairFlyProducerRefusal, HairFlyRegisteredModel, HairFlySelectorReceipt, SameJobFace,
    SELECTOR_MODEL_HASH_ID, SELECTOR_OUTPUT_NAME,
};
use crate::portrait_stage::{
    BeardInputs, HairGlossInputs, HairSmoothInputs, MoistureBodyInputs, MoistureFaceInputs,
    Pf30025LiveModelOutputs, Pf30025ProducerInputs, Pf30025SourceSurface, Pf30026EdgeOutput,
    Pf30026GateBinding, Pf90130GlareProducer, SmileProducerInputs, SmileRgbReconstructionField,
    SmileSigmoidEdgeResponse, StrayHairProducerInputs, TeethAlignmentProducerInputs,
    TeethMaskFormat, TeethMaskSource, TeethProducerInputs, PF108_BRIGHT_LUT_IDENTITY,
    PF108_WHITEN_LUT_IDENTITY, PF90130_EFFECT_TENSOR_ID, PF90130_INPUT_TENSOR_ID,
    PF90130_MASK_TENSOR_ID,
};
use crate::NormRect;

pub const PF108_WHITEN: &str = "PF108_WHITEN_LUT";
pub const PF108_BRIGHT: &str = "PF108_BRIGHT_LUT";

pub const PF21026_MASK_158607435: &str = "158607435";
pub const PF21026_MASK_2405607714: &str = "2405607714";
pub const PF21026_MASK_3434589800: &str = "3434589800";
pub const PF21026_MASK_2977284499: &str = "2977284499";
pub const PF21026_MASK_2892147817: &str = "2892147817";
pub const PF21026_REPAIR_WRITER_RVA: u32 = 0x823A3B0;
pub const PF21026_REPAIR_OUTPUT3_LOOKUP_SLOT: u16 = 0x350;
pub const PF21026_REPAIR_OUTPUT4_LOOKUP_SLOT: u16 = 0x4B0;
pub const PF21026_REPAIR_CONSUMER_RVA: u32 = 0x82AA2E0;
pub const PF21026_REPAIR_NATIVE_MAT_OFFSET: u16 = 0x4A8;

static HAIRFLY_LARGE_MODEL_IDS: [u64; 4] =
    [3_378_116_080, 2_127_140_479, 4_083_080_210, 195_031_232];
static SMILE_STAGE_MODEL_IDS: [u64; 2] = [806_921_480, 1_940_369_920];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VendorModel {
    pub id: u64,
    pub family: &'static str,
    pub source_file: &'static str,
    pub source_bytes: u64,
    pub source_sha256: &'static str,
    pub private_fb_sha256: Option<&'static str>,
    pub derived_mnn_sha256: Option<&'static str>,
    pub width: u16,
    pub height: u16,
}

/// The eight payloads landed by G3. These hashes identify the encrypted source
/// files; they are not substitutes for a render-job output receipt.
pub const VENDOR_MODELS: [VendorModel; 8] = [
    VendorModel {
        id: 1_655_581_403,
        family: "HairFlyV1 selector",
        source_file: "extract/pixcook/1655581403",
        source_bytes: 2_410_861,
        source_sha256: "6a025f05c26aca02def0d81e2c443d0e3dc01e61f2af00a525aca6d4ca60d14d",
        private_fb_sha256: Some(
            "4e34a40ccdd0223160e937876e1c7cd0a57911a024141f3304aa4a169afcf96e",
        ),
        derived_mnn_sha256: Some(
            "d0e9fc5bac55e4c14c40faa8684138a3e7df3eb24c89bf73aaccdfef6ad3d730",
        ),
        width: 448,
        height: 224,
    },
    VendorModel {
        id: 3_378_116_080,
        family: "HairFlyV1/PF21026 five-head",
        source_file: "extract/pixcook/3378116080",
        source_bytes: 7_318_717,
        source_sha256: "7d3f2aec4c4f815d758892f7c5a8ad66fff336c5fe9272a05613e0e5aa52a68b",
        private_fb_sha256: Some(
            "9bb50549c57e0c71068e6a036bc8091e7025455d624cb8a069190f8f47d45dd7",
        ),
        derived_mnn_sha256: Some(
            "8fb49c3f87c45724451b358aa5f601fad612165ae635fcb8313b5517df8eb373",
        ),
        width: 768,
        height: 512,
    },
    VendorModel {
        id: 2_127_140_479,
        family: "HairFlyV1/PF21026 five-head",
        source_file: "extract/pixcook/2127140479",
        source_bytes: 7_318_717,
        source_sha256: "0e027141a594e4a20e01b3fc61cd02c9b1e57c3160488b294a9746dc659c12c6",
        private_fb_sha256: Some(
            "484f501893cbd015f8eb4179264d41982304925df9930b2d647f747cf7f2ee57",
        ),
        derived_mnn_sha256: Some(
            "d5e2e84fea1afa14f2623a99768679098389591ca03bd2a59917157324dc2c70",
        ),
        width: 768,
        height: 512,
    },
    VendorModel {
        id: 4_083_080_210,
        family: "HairFlyV1/PF21026 five-head",
        source_file: "extract/pixcook/4083080210",
        source_bytes: 7_208_025,
        source_sha256: "8140003eed6e5606742be72b64ef04ad240a6252fc1ecef71bc556ff6ae668fd",
        private_fb_sha256: Some(
            "d12a7482e25f265890353949bc8ae71ab0b42d635a8896d42d9bc3185519895b",
        ),
        derived_mnn_sha256: Some(
            "b9c47bdbfefb09bfe3e695b4391a955af5b9af92b25c54da7a79ca05aee9b36b",
        ),
        width: 512,
        height: 384,
    },
    VendorModel {
        id: 195_031_232,
        family: "HairFlyV1/PF21026 five-head deep",
        source_file: "extract/pixcook/195031232",
        source_bytes: 7_224_497,
        source_sha256: "054ad6613315b13688eeb7b159d8c86a0e54969cf3a9c8d15fd350396367f998",
        private_fb_sha256: Some(
            "bfc7533a2b9cb837578a31a42a1e9ee3282166fd37ea27ac9d515b0dc540c29f",
        ),
        derived_mnn_sha256: Some(
            "175763190cc93fa83ece0da80d84dc53338de986a1325270765d486969b76296",
        ),
        width: 512,
        height: 384,
    },
    VendorModel {
        id: 806_921_480,
        family: "SmileV1 768x1024 stage 1/2",
        source_file: "extract/pixcook/806921480",
        source_bytes: 2_662_989,
        source_sha256: "87342bf07c1b32bb872277c2d048e3ced15f4416f800f4c1ee2b3e9cb54e7689",
        private_fb_sha256: None,
        derived_mnn_sha256: Some(
            "0b474879c04dddcd4d26177f09580ff1ad5345f8b7a64cc36ee1524c35d47b45",
        ),
        width: 1024,
        height: 768,
    },
    VendorModel {
        id: 1_940_369_920,
        family: "SmileV1 768x1024 stage 1/2",
        source_file: "extract/pixcook/1940369920",
        source_bytes: 2_662_989,
        source_sha256: "ed8b7587d71f4c18d37a509829d20b4e9140cb77b4618ee8eb9ffad5fc0c0faf",
        private_fb_sha256: None,
        derived_mnn_sha256: Some(
            "811afd84bc4266821848a65ce3940e533f29d666b0a56f8d2ea261c202bf06b2",
        ),
        width: 1024,
        height: 768,
    },
    VendorModel {
        id: 3_157_078_880,
        family: "SmileV1 512x512 residual/refine",
        source_file: "extract/pixcook/3157078880",
        source_bytes: 5_024_629,
        source_sha256: "eac03538db8e2c93e5db9b33157583064943a99a4afacdf393114d48f10c395a",
        private_fb_sha256: None,
        derived_mnn_sha256: Some(
            "d5ea7f78d631f6fa2c7949dcf60a895970e58fe349c595c84913335a578ec52c",
        ),
        width: 512,
        height: 512,
    },
];

pub fn vendor_model(id: u64) -> Option<&'static VendorModel> {
    VENDOR_MODELS.iter().find(|model| model.id == id)
}

pub const PF30026_EFFECT_RGB: &str = "PF30026_EFFECT_RGB";
pub const PF30026_TEETH_MASK: &str = "PF30026_TEETH_MASK";
pub const PF30026_EDGE_0: &str = "PF30026_EDGE_0";
pub const PF30026_EDGE_1: &str = "PF30026_EDGE_1";

pub const PF90088_SELECTOR: &str = "1644271730";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HairFlyMaskChannel {
    Ch0,
    Ch1,
    Ch2,
}

impl HairFlyMaskChannel {
    fn index(self) -> usize {
        match self {
            Self::Ch0 => 0,
            Self::Ch1 => 1,
            Self::Ch2 => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HairFlyMaskPostprocess {
    NativeBorderFeatherMinRoiWarpU8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairFlyRoiWarpDrawBuffer {
    pub model_roi: [u32; 4],
    pub frame_roi: [u32; 4],
    pub warp: [[f32; 3]; 2],
    pub vertex_count: u32,
    pub vertex_buffer_sha256: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HairFlyMaskDecodeReceipt {
    pub same_job: SameJobFace,
    pub model_hash_id: u32,
    pub channel: HairFlyMaskChannel,
    pub postprocess: HairFlyMaskPostprocess,
    pub mask_b_plane_sha256: [u8; 32],
    pub raw_mask_sha256: [u8; 32],
    pub draw_buffer: HairFlyRoiWarpDrawBuffer,
}

pub const PF90139_HAIR_MASK: &str = "PF90139_HAIR_MASK";
pub const PF90139_MIX_MASK: &str = "PF90139_MIX_MASK";
pub const PF90139_BLEND_RGB: &str = "PF90139_BLEND_SOURCE";
pub const PF90139_DETAIL: &str = "PF90139_DETAIL";

pub const PF90160_HAIR_MASK: &str = "PF90160_HAIR_MASK";
pub const PF90160_DETAIL: &str = "PF90160_MODEL_DETAIL";

pub const PF90002_MASK_RGBA: &str = "PF90002_MASK_RGBA";
pub const PF90002_BLUR_ALPHA: &str = "PF90002_BLUR_ALPHA";
pub const PF90002_SKIN_MAP: &str = "PF90002_SKIN_MAP";
pub const PF90002_HARDLIGHT: &str = "PF90002_HARDLIGHT";
pub const PF90002_BEAUTY: &str = "PF90002_BEAUTY";

pub const PF90010_WEIGHT_SKIN: &str = "PF90010_WEIGHT_SKIN";
pub const PF90010_WEIGHT_DOG: &str = "PF90010_WEIGHT_DOG";
pub const PF90010_BEAUTY: &str = "PF90010_BEAUTY";

static PF30025_MODEL_IDS: [u64; 2] = [806_921_480, 1_940_369_920];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PortraitModelTarget {
    Pf108,
    Pf21026,
    Pf30025,
    Pf30026,
    Pf90116,
    Pf90130,
    Pf90088,
    Pf90139Pf90160,
    Pf90002Pf90010,
    Pf100007,
}

impl PortraitModelTarget {
    pub const ALL: [Self; 10] = [
        Self::Pf108,
        Self::Pf21026,
        Self::Pf30025,
        Self::Pf30026,
        Self::Pf90116,
        Self::Pf90130,
        Self::Pf90088,
        Self::Pf90139Pf90160,
        Self::Pf90002Pf90010,
        Self::Pf100007,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Pf108 => "PF108",
            Self::Pf21026 => "PF21026",
            Self::Pf30025 => "PF30025",
            Self::Pf30026 => "PF30026",
            Self::Pf90116 => "PF90116",
            Self::Pf90130 => "PF90130",
            Self::Pf90088 => "PF90088",
            Self::Pf90139Pf90160 => "PF90139/PF90160",
            Self::Pf90002Pf90010 => "PF90002/PF90010",
            Self::Pf100007 => "PF100007",
        }
    }

    pub fn requirements(self) -> Vec<TensorRequirement> {
        use ModelRequirement::{AnyOf, Exact, NonZero};
        use ShapeRequirement::{ExactShape, NonEmpty};

        match self {
            Self::Pf108 => vec![
                TensorRequirement::new(PF108_WHITEN, "WhitenLutPath", NonEmpty, NonZero, 0),
                TensorRequirement::new(PF108_BRIGHT, "BrightLutPath", NonEmpty, NonZero, 0),
            ],
            Self::Pf21026 => vec![
                TensorRequirement::new(
                    PF90088_SELECTOR,
                    "1644271730",
                    ExactShape([1, 1, 1, 2]),
                    Exact(SELECTOR_MODEL_HASH_ID as u64),
                    1,
                ),
                TensorRequirement::new(
                    PF21026_MASK_158607435,
                    PF21026_MASK_158607435,
                    NonEmpty,
                    AnyOf(&HAIRFLY_LARGE_MODEL_IDS),
                    2,
                ),
                TensorRequirement::new(
                    PF21026_MASK_2405607714,
                    PF21026_MASK_2405607714,
                    NonEmpty,
                    AnyOf(&HAIRFLY_LARGE_MODEL_IDS),
                    2,
                ),
                TensorRequirement::new(
                    PF21026_MASK_3434589800,
                    PF21026_MASK_3434589800,
                    NonEmpty,
                    AnyOf(&HAIRFLY_LARGE_MODEL_IDS),
                    2,
                ),
                TensorRequirement::new(
                    PF21026_MASK_2977284499,
                    PF21026_MASK_2977284499,
                    NonEmpty,
                    AnyOf(&HAIRFLY_LARGE_MODEL_IDS),
                    2,
                ),
                TensorRequirement::new(
                    PF21026_MASK_2892147817,
                    PF21026_MASK_2892147817,
                    NonEmpty,
                    AnyOf(&HAIRFLY_LARGE_MODEL_IDS),
                    2,
                ),
            ],
            Self::Pf30025 => vec![
                TensorRequirement::new(
                    "780929437",
                    "780929437",
                    NonEmpty,
                    AnyOf(&PF30025_MODEL_IDS),
                    0,
                ),
                TensorRequirement::new(
                    "3204548346",
                    "3204548346",
                    NonEmpty,
                    AnyOf(&PF30025_MODEL_IDS),
                    0,
                ),
            ],
            Self::Pf30026 => vec![
                TensorRequirement::new(
                    PF30026_EFFECT_RGB,
                    PF30026_EFFECT_RGB,
                    NonEmpty,
                    NonZero,
                    0,
                ),
                TensorRequirement::new(
                    PF30026_TEETH_MASK,
                    PF30026_TEETH_MASK,
                    NonEmpty,
                    NonZero,
                    0,
                ),
                TensorRequirement::new(PF30026_EDGE_0, PF30026_EDGE_0, NonEmpty, NonZero, 0),
                TensorRequirement::new(PF30026_EDGE_1, PF30026_EDGE_1, NonEmpty, NonZero, 0),
            ],
            Self::Pf90116 => vec![
                TensorRequirement::new(
                    "781141184",
                    "781141184",
                    ExactShape([1, 3, 384, 512]),
                    Exact(814_444_604),
                    0,
                ),
                TensorRequirement::new(
                    "3204548346",
                    "3204548346",
                    ExactShape([1, 1, 384, 512]),
                    Exact(814_444_604),
                    0,
                ),
                TensorRequirement::new(
                    "780929437",
                    "780929437",
                    ExactShape([1, 3, 384, 512]),
                    Exact(814_444_604),
                    0,
                ),
            ],
            Self::Pf90130 => vec![
                TensorRequirement::new(
                    "PF90130_INPUT",
                    "PF90130_INPUT",
                    ExactShape([1, 384, 768, 4]),
                    Exact(1_463_593_431),
                    0,
                ),
                TensorRequirement::new(
                    "PF90130_EFFECT",
                    "PF90130_EFFECT",
                    ExactShape([1, 384, 768, 4]),
                    Exact(1_463_593_431),
                    0,
                ),
                TensorRequirement::new(
                    "PF90130_MASK",
                    "PF90130_MASK",
                    ExactShape([1, 384, 768, 4]),
                    Exact(1_463_593_431),
                    0,
                ),
            ],
            Self::Pf90088 => vec![
                TensorRequirement::new(
                    PF90088_SELECTOR,
                    "1644271730",
                    ExactShape([1, 1, 1, 2]),
                    Exact(SELECTOR_MODEL_HASH_ID as u64),
                    3,
                ),
                TensorRequirement::new(
                    PF21026_MASK_158607435,
                    PF21026_MASK_158607435,
                    NonEmpty,
                    AnyOf(&HAIRFLY_LARGE_MODEL_IDS),
                    4,
                ),
                TensorRequirement::new(
                    PF21026_MASK_2405607714,
                    PF21026_MASK_2405607714,
                    NonEmpty,
                    AnyOf(&HAIRFLY_LARGE_MODEL_IDS),
                    4,
                ),
                TensorRequirement::new(
                    PF21026_MASK_3434589800,
                    PF21026_MASK_3434589800,
                    NonEmpty,
                    AnyOf(&HAIRFLY_LARGE_MODEL_IDS),
                    4,
                ),
                TensorRequirement::new(
                    PF21026_MASK_2977284499,
                    PF21026_MASK_2977284499,
                    NonEmpty,
                    AnyOf(&HAIRFLY_LARGE_MODEL_IDS),
                    4,
                ),
                TensorRequirement::new(
                    PF21026_MASK_2892147817,
                    PF21026_MASK_2892147817,
                    NonEmpty,
                    AnyOf(&HAIRFLY_LARGE_MODEL_IDS),
                    4,
                ),
            ],
            Self::Pf90139Pf90160 => vec![
                TensorRequirement::new(
                    PF90139_HAIR_MASK,
                    PF90139_HAIR_MASK,
                    NonEmpty,
                    Exact(20_005_000),
                    1,
                ),
                TensorRequirement::new(
                    PF90139_MIX_MASK,
                    PF90139_MIX_MASK,
                    NonEmpty,
                    Exact(20_005_000),
                    1,
                ),
                TensorRequirement::new(
                    PF90139_BLEND_RGB,
                    PF90139_BLEND_RGB,
                    NonEmpty,
                    Exact(20_005_000),
                    1,
                ),
                TensorRequirement::new(
                    PF90139_DETAIL,
                    PF90139_DETAIL,
                    NonEmpty,
                    Exact(20_005_000),
                    1,
                ),
                TensorRequirement::new(
                    PF90160_HAIR_MASK,
                    PF90160_HAIR_MASK,
                    NonEmpty,
                    Exact(20_005_100),
                    2,
                ),
                TensorRequirement::new(
                    PF90160_DETAIL,
                    PF90160_DETAIL,
                    NonEmpty,
                    Exact(20_005_100),
                    2,
                ),
            ],
            Self::Pf90002Pf90010 => vec![
                TensorRequirement::new(PF90002_MASK_RGBA, PF90002_MASK_RGBA, NonEmpty, NonZero, 1),
                TensorRequirement::new(PF90002_BLUR_ALPHA, PF90002_BLUR_ALPHA, NonEmpty, NonZero, 1),
                TensorRequirement::new(PF90002_SKIN_MAP, PF90002_SKIN_MAP, NonEmpty, NonZero, 1),
                TensorRequirement::new(PF90002_HARDLIGHT, PF90002_HARDLIGHT, NonEmpty, NonZero, 1),
                TensorRequirement::new(PF90002_BEAUTY, PF90002_BEAUTY, NonEmpty, NonZero, 1),
                TensorRequirement::new(
                    PF90010_WEIGHT_SKIN,
                    PF90010_WEIGHT_SKIN,
                    NonEmpty,
                    NonZero,
                    2,
                ),
                TensorRequirement::new(
                    PF90010_WEIGHT_DOG,
                    PF90010_WEIGHT_DOG,
                    NonEmpty,
                    NonZero,
                    2,
                ),
                TensorRequirement::new(PF90010_BEAUTY, PF90010_BEAUTY, NonEmpty, NonZero, 2),
            ],
            Self::Pf100007 => vec![
                TensorRequirement::new(
                    "780929437",
                    "780929437",
                    ExactShape([1, 3, 768, 1024]),
                    AnyOf(&SMILE_STAGE_MODEL_IDS),
                    0,
                ),
                TensorRequirement::new(
                    "3204548346",
                    "3204548346",
                    ExactShape([1, 1, 768, 1024]),
                    AnyOf(&SMILE_STAGE_MODEL_IDS),
                    0,
                ),
            ],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShapeRequirement {
    ExactShape([u32; 4]),
    NonEmpty,
}

impl ShapeRequirement {
    fn accepts(self, shape: [u32; 4]) -> bool {
        match self {
            Self::ExactShape(expected) => shape == expected,
            Self::NonEmpty => checked_len(shape).is_some(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelRequirement {
    Exact(u64),
    AnyOf(&'static [u64]),
    NonZero,
}

impl ModelRequirement {
    fn accepts(self, model_id: u64) -> bool {
        match self {
            Self::Exact(expected) => model_id == expected,
            Self::AnyOf(ids) => ids.contains(&model_id),
            Self::NonZero => model_id != 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorRequirement {
    pub role: &'static str,
    pub tensor_name: &'static str,
    pub shape: ShapeRequirement,
    pub model: ModelRequirement,
    pub model_group: u8,
}

impl TensorRequirement {
    pub const fn new(
        role: &'static str,
        tensor_name: &'static str,
        shape: ShapeRequirement,
        model: ModelRequirement,
        model_group: u8,
    ) -> Self {
        Self {
            role,
            tensor_name,
            shape,
            model,
            model_group,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SessionTensor {
    pub job_id: u64,
    pub source_digest: [u8; 32],
    pub model_id: u64,
    pub tensor_name: &'static str,
    pub shape: [u32; 4],
    pub digest: [u8; 32],
    bytes: Vec<u8>,
    values: Option<Vec<f32>>,
}

impl SessionTensor {
    pub fn from_f32(
        job_id: u64,
        source_digest: [u8; 32],
        model_id: u64,
        tensor_name: &'static str,
        shape: [u32; 4],
        values: &[f32],
    ) -> Result<Self, Refusal> {
        let expected = checked_len(shape).ok_or(Refusal::ShapeMismatch("model output shape"))?;
        if values.len() != expected {
            return Err(Refusal::ShapeMismatch("model output values"));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(Refusal::ValueOutOfRange("model output values"));
        }
        let mut bytes = Vec::with_capacity(expected * 4);
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Self::from_bytes_with_values(
            job_id,
            source_digest,
            model_id,
            tensor_name,
            shape,
            bytes,
            Some(values.to_vec()),
        )
    }

    pub fn from_bytes(
        job_id: u64,
        source_digest: [u8; 32],
        model_id: u64,
        tensor_name: &'static str,
        shape: [u32; 4],
        bytes: &[u8],
    ) -> Result<Self, Refusal> {
        Self::from_bytes_with_values(
            job_id,
            source_digest,
            model_id,
            tensor_name,
            shape,
            bytes.to_vec(),
            None,
        )
    }

    pub fn from_session_output(job_id: u64, output: &SessionOutput) -> Result<Self, Refusal> {
        let shape = output.tensor.shape;
        let expect = checked_len(shape).ok_or(Refusal::ShapeMismatch("model output shape"))?;
        if output.tensor.data.len() != expect {
            return Err(Refusal::ShapeMismatch("model output values"));
        }
        let mut bytes = Vec::with_capacity(expect * 4);
        for value in &output.tensor.data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Self::from_bytes_with_values(
            job_id,
            output.stamp.input_digest,
            u64::from(output.stamp.model_id),
            output.stamp.output_tensor,
            shape,
            bytes,
            Some(output.tensor.data.clone()),
        )
    }

    fn from_bytes_with_values(
        job_id: u64,
        source_digest: [u8; 32],
        model_id: u64,
        tensor_name: &'static str,
        shape: [u32; 4],
        bytes: Vec<u8>,
        values: Option<Vec<f32>>,
    ) -> Result<Self, Refusal> {
        if job_id == 0 {
            return Err(Refusal::MissingInput("model output job"));
        }
        if source_digest == [0; 32] {
            return Err(Refusal::MissingInput("model output source"));
        }
        if model_id == 0 {
            return Err(Refusal::MissingInput("model output model id"));
        }
        if tensor_name.is_empty() {
            return Err(Refusal::MissingInput("model output tensor name"));
        }
        let expected = checked_len(shape).ok_or(Refusal::ShapeMismatch("model output shape"))?;
        if bytes.is_empty() {
            return Err(Refusal::ShapeMismatch("model output bytes"));
        }
        if let Some(values) = values.as_ref() {
            if bytes.len() != expected * 4
                || values.len() != expected
                || values.iter().any(|value| !value.is_finite())
            {
                return Err(Refusal::ValueOutOfRange("model output values"));
            }
        }
        Ok(Self {
            job_id,
            source_digest,
            model_id,
            tensor_name,
            shape,
            digest: sha256(&bytes),
            bytes,
            values,
        })
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn values(&self) -> Result<&[f32], Refusal> {
        self.values
            .as_deref()
            .ok_or(Refusal::MissingInput("model output f32 values"))
    }

    pub fn pixels(&self) -> Result<usize, Refusal> {
        let [_, channels, height, width] = self.shape;
        (channels as usize)
            .checked_mul(height as usize)
            .and_then(|value| value.checked_mul(width as usize))
            .ok_or(Refusal::ShapeMismatch("model output pixels"))
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SessionModelOutputs {
    outputs: Vec<SessionTensor>,
}

impl SessionModelOutputs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, output: SessionTensor) -> Result<(), Refusal> {
        if self
            .outputs
            .iter()
            .any(|existing| existing.tensor_name == output.tensor_name)
        {
            return Err(Refusal::ValueOutOfRange("duplicate model output tensor"));
        }
        self.outputs.push(output);
        Ok(())
    }

    pub fn get(&self, tensor_name: &str) -> Option<&SessionTensor> {
        self.outputs
            .iter()
            .find(|output| output.tensor_name == tensor_name)
    }

    pub fn bind(
        &self,
        target: PortraitModelTarget,
        job_id: u64,
        source_digest: [u8; 32],
    ) -> Result<BoundModelOutputs<'_>, Refusal> {
        let mut tensors = BTreeMap::new();
        let mut group_models = BTreeMap::new();
        for requirement in target.requirements() {
            let tensor = self
                .get(requirement.tensor_name)
                .ok_or(Refusal::MissingInput(requirement.role))?;
            if tensor.job_id != job_id {
                return Err(Refusal::ValueOutOfRange(match target {
                    PortraitModelTarget::Pf108 => "PF108_MODEL_JOB",
                    PortraitModelTarget::Pf21026 => "PF21026_MODEL_JOB",
                    PortraitModelTarget::Pf30025 => "PF30025_MODEL_JOB",
                    PortraitModelTarget::Pf30026 => "PF30026_MODEL_JOB",
                    PortraitModelTarget::Pf90116 => "PF90116_MODEL_JOB",
                    PortraitModelTarget::Pf90130 => "PF90130_MODEL_JOB",
                    PortraitModelTarget::Pf90088 => "PF90088_MODEL_JOB",
                    PortraitModelTarget::Pf90139Pf90160 => "PF90139_PF90160_MODEL_JOB",
                    PortraitModelTarget::Pf90002Pf90010 => "PF90002_PF90010_MODEL_JOB",
                    PortraitModelTarget::Pf100007 => "PF100007_MODEL_JOB",
                }));
            }
            if tensor.source_digest != source_digest {
                return Err(Refusal::ValueOutOfRange(match target {
                    PortraitModelTarget::Pf108 => "PF108_MODEL_SOURCE",
                    PortraitModelTarget::Pf21026 => "PF21026_MODEL_SOURCE",
                    PortraitModelTarget::Pf30025 => "PF30025_MODEL_SOURCE",
                    PortraitModelTarget::Pf30026 => "PF30026_MODEL_SOURCE",
                    PortraitModelTarget::Pf90116 => "PF90116_MODEL_SOURCE",
                    PortraitModelTarget::Pf90130 => "PF90130_MODEL_SOURCE",
                    PortraitModelTarget::Pf90088 => "PF90088_MODEL_SOURCE",
                    PortraitModelTarget::Pf90139Pf90160 => "PF90139_PF90160_MODEL_SOURCE",
                    PortraitModelTarget::Pf90002Pf90010 => "PF90002_PF90010_MODEL_SOURCE",
                    PortraitModelTarget::Pf100007 => "PF100007_MODEL_SOURCE",
                }));
            }
            if !requirement.model.accepts(tensor.model_id) {
                return Err(Refusal::ValueOutOfRange(requirement.role));
            }
            if !requirement.shape.accepts(tensor.shape) {
                return Err(Refusal::ShapeMismatch(requirement.role));
            }
            if tensor.digest != sha256(&tensor.bytes) {
                return Err(Refusal::ValueOutOfRange(requirement.role));
            }
            if let Some(previous) = group_models.insert(requirement.model_group, tensor.model_id) {
                if previous != tensor.model_id {
                    return Err(Refusal::ValueOutOfRange("model id group"));
                }
            }
            tensors.insert(requirement.tensor_name, tensor);
        }
        Ok(BoundModelOutputs {
            target,
            job_id,
            source_digest,
            tensors,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoundModelOutputs<'a> {
    pub target: PortraitModelTarget,
    pub job_id: u64,
    pub source_digest: [u8; 32],
    tensors: BTreeMap<&'static str, &'a SessionTensor>,
}

impl<'a> BoundModelOutputs<'a> {
    pub fn tensor(&self, name: &str) -> Option<&'a SessionTensor> {
        self.tensors.get(name).copied()
    }

    fn required(
        &self,
        name: &'static str,
        missing: &'static str,
    ) -> Result<&'a SessionTensor, Refusal> {
        self.tensor(name).ok_or(Refusal::MissingInput(missing))
    }

    fn values(&self, name: &'static str, missing: &'static str) -> Result<&'a [f32], Refusal> {
        self.required(name, missing)?.values()
    }

    pub fn teeth_inputs<'b>(
        &'b self,
        mask: TeethMaskBinding<'b>,
    ) -> Result<TeethProducerInputs, Refusal> {
        if self.target != PortraitModelTarget::Pf108 {
            return Err(Refusal::ValueOutOfRange("PF108 target"));
        }
        if mask.job_id != self.job_id {
            return Err(Refusal::ValueOutOfRange("PF108 mask job"));
        }
        if sha256(mask.mask_g) != mask.mask_sha256 {
            return Err(Refusal::ValueOutOfRange("PF108 mask digest"));
        }
        let whiten = self.required(PF108_WHITEN, "PF108_WHITEN_LUT")?;
        let bright = self.required(PF108_BRIGHT, "PF108_BRIGHT_LUT")?;
        if whiten.bytes().len() != PF108_WHITEN_LUT_IDENTITY.rgb8_bytes
            || bright.bytes().len() != PF108_BRIGHT_LUT_IDENTITY.rgb8_bytes
        {
            return Err(Refusal::ShapeMismatch("PF108 LUT"));
        }
        Ok(TeethProducerInputs {
            job_id: self.job_id,
            mask_width: mask.width,
            mask_height: mask.height,
            mask_source: mask.source,
            mask_format: mask.format,
            mask_sha256: mask.mask_sha256,
            whiten_lut_identity: PF108_WHITEN_LUT_IDENTITY,
            bright_lut_identity: PF108_BRIGHT_LUT_IDENTITY,
            mask_g: mask.mask_g.to_vec(),
            bright_ui: mask.bright_ui,
            deyellow_ui: mask.deyellow_ui,
            whiten_lut: whiten.bytes().to_vec(),
            bright_lut: bright.bytes().to_vec(),
        })
    }

    fn hairfly_model(&self, aux_gate_high: bool) -> Result<HairFlyRegisteredModel, Refusal> {
        let selector = self.required(PF90088_SELECTOR, "PF90088_SELECTOR")?;
        let logits = selector.values()?;
        if logits.len() != 2 {
            return Err(Refusal::ShapeMismatch("PF90088 selector logits"));
        }
        let argmax = selector_argmax([logits[0], logits[1]]).map_err(hairfly_refusal)?;
        model_for_selector(argmax, aux_gate_high)
            .ok_or(Refusal::ValueOutOfRange("PF90088_SELECTOR_ROUTE"))
    }

    fn hairfly_tensor(
        &self,
        name: &'static str,
        route: HairFlyRegisteredModel,
        route_error: &'static str,
    ) -> Result<&'a SessionTensor, Refusal> {
        let tensor = self.required(name, name)?;
        if tensor.model_id != u64::from(route.hash_id) {
            return Err(Refusal::ValueOutOfRange(route_error));
        }
        if tensor.shape[0] != 1
            || tensor.shape[2] != u32::from(route.height)
            || tensor.shape[3] != u32::from(route.width)
        {
            return Err(Refusal::ShapeMismatch(name));
        }
        Ok(tensor)
    }

    /// Bind PF21026 only after the host has decoded the raw tensors. `mask_b`
    /// is channel-major `[3, H, W]`; its consumer plane must be named
    /// explicitly. The repair port is explicit because output 3 is the graph
    /// repair while only output 4 has a located native u8 conversion.
    pub fn beard<'b>(&'b self, repair: BeardRepairBinding<'b>) -> Result<BeardInputs<'b>, Refusal> {
        if self.target != PortraitModelTarget::Pf21026 {
            return Err(Refusal::ValueOutOfRange("PF21026 target"));
        }
        repair
            .frame
            .validate_against(self.job_id, self.source_digest)?;
        repair.repair_rgb.validate()?;
        if (repair.frame.width, repair.frame.height)
            != (repair.repair_rgb.width, repair.repair_rgb.height)
        {
            return Err(Refusal::ShapeMismatch("PF21026 repair frame"));
        }
        let route = self.hairfly_model(repair.aux_gate_high)?;
        let mask_a_tensor = self.hairfly_tensor(
            PF21026_MASK_158607435,
            route,
            "PF21026_MODEL_ROUTE",
        )?;
        let mask_b_tensor = self.hairfly_tensor(
            PF21026_MASK_2405607714,
            route,
            "PF21026_MODEL_ROUTE",
        )?;
        let repair_tensor =
            self.hairfly_tensor(repair.repair_output.tensor_name(), route, "PF21026_MODEL_ROUTE")?;
        let consumer = repair
            .consumer
            .ok_or(Refusal::MissingInput("PF21026_REPAIR_NATIVE_MAT_CONSUMER"))?;
        if consumer.job_id != self.job_id {
            return Err(Refusal::ValueOutOfRange("PF21026_REPAIR_CONSUMER_JOB"));
        }
        if consumer.output != repair.repair_output {
            return Err(Refusal::ValueOutOfRange("PF21026_REPAIR_CONSUMER_OUTPUT"));
        }
        if consumer.producer_rva != PF21026_REPAIR_WRITER_RVA {
            return Err(Refusal::ValueOutOfRange("PF21026_REPAIR_WRITER_RECEIPT"));
        }
        if consumer.consumer_rva != PF21026_REPAIR_CONSUMER_RVA
            || consumer.native_mat_offset != PF21026_REPAIR_NATIVE_MAT_OFFSET
        {
            return Err(Refusal::MissingInput("PF21026_REPAIR_NATIVE_MAT_EDGE"));
        }
        let lookup_slot = match consumer.output {
            BeardRepairOutput::Output3 => PF21026_REPAIR_OUTPUT3_LOOKUP_SLOT,
            BeardRepairOutput::Output4 => PF21026_REPAIR_OUTPUT4_LOOKUP_SLOT,
        };
        if consumer.producer_lookup_slot != lookup_slot {
            return Err(Refusal::ValueOutOfRange(
                "PF21026_REPAIR_LOOKUP_SLOT_RECEIPT",
            ));
        }
        if consumer.output == BeardRepairOutput::Output3 {
            return Err(Refusal::MissingInput(
                "PF21026_REPAIR_OUTPUT3_NATIVE_U8_CONSUMER",
            ));
        }
        if mask_a_tensor.shape[1] != 1 {
            return Err(Refusal::MissingInput("PF21026_MASK_A_CHANNEL_DECODE"));
        }
        if mask_b_tensor.shape[1] != 3 {
            return Err(Refusal::MissingInput("PF21026_MASK_B_CHANNEL_DECODE"));
        }
        if repair_tensor.shape[1] != 3 {
            return Err(Refusal::ShapeMismatch("PF21026 repair RGB"));
        }
        if mask_a_tensor.digest != repair.mask_a_source_sha256
            || mask_b_tensor.digest != repair.mask_b_source_sha256
            || repair_tensor.digest != repair.repair_source_sha256
        {
            return Err(Refusal::ValueOutOfRange("PF21026_DECODE_RECEIPT"));
        }
        let pixels = (repair.repair_rgb.width as usize)
            .checked_mul(repair.repair_rgb.height as usize)
            .ok_or(Refusal::ShapeMismatch("PF21026 repair RGB"))?;
        let mask_b_bytes = pixels
            .checked_mul(3)
            .ok_or(Refusal::ShapeMismatch("PF21026 mask_b planes"))?;
        if repair.mask_a.len() != pixels || repair.mask_b.len() != mask_b_bytes {
            return Err(Refusal::ShapeMismatch("PF21026 masks/repair frame"));
        }
        if sha256(repair.mask_b) != repair.mask_b_decoded_sha256 {
            return Err(Refusal::ValueOutOfRange("PF21026_MASK_B_DECODE_RECEIPT"));
        }
        if sha256(&repair.repair_rgb.pixels) != consumer.native_mat_sha256 {
            return Err(Refusal::ValueOutOfRange(
                "PF21026_REPAIR_NATIVE_MAT_RECEIPT",
            ));
        }
        let mask_b_start = repair.mask_b_channel.index() * pixels;
        Ok(BeardInputs {
            mask_a: repair.mask_a,
            mask_b: &repair.mask_b[mask_b_start..mask_b_start + pixels],
            repair_rgb: repair.repair_rgb,
        })
    }

    pub fn pf30025_inputs(
        &self,
        source_surface: Pf30025SourceSurface<'_>,
        teeth_mask: Vec<f32>,
        roi: ToothRect,
    ) -> Result<Pf30025ProducerInputs, Refusal> {
        if self.target != PortraitModelTarget::Pf30025 {
            return Err(Refusal::ValueOutOfRange("PF30025 target"));
        }
        let repair = self.values("780929437", "PF30025_REPAIR_780929437")?;
        let gate = self.values("3204548346", "PF30025_GATE_3204548346")?;
        let pixels = source_surface
            .width
            .checked_mul(source_surface.height)
            .map(|value| value as usize)
            .ok_or(Refusal::ShapeMismatch("PF30025 source surface"))?;
        if repair.len() != pixels * 3 {
            return Err(Refusal::ShapeMismatch("PF30025 repair planes"));
        }
        let repair_plane_0 = &repair[0..pixels];
        let repair_plane_1 = &repair[pixels..pixels * 2];
        let repair_plane_2 = &repair[pixels * 2..pixels * 3];
        Pf30025ProducerInputs::from_live_model(
            Pf30025LiveModelOutputs {
                job_id: self.job_id,
                model_id: self
                    .required("780929437", "PF30025_REPAIR_780929437")?
                    .model_id as u32,
                source_surface,
                repair_780929437_bgr: [repair_plane_0, repair_plane_1, repair_plane_2],
                gate_3204548346: gate,
            },
            teeth_mask,
            roi,
        )
    }

    pub fn teeth_alignment_inputs(
        &self,
        gate_binding: Pf30026GateBinding,
        roi: NormRect,
    ) -> Result<TeethAlignmentProducerInputs, Refusal> {
        if self.target != PortraitModelTarget::Pf30026 {
            return Err(Refusal::ValueOutOfRange("PF30026 target"));
        }
        if gate_binding.job_id != self.job_id {
            return Err(Refusal::ValueOutOfRange("PF30026 gate job"));
        }
        let effect = self.required(PF30026_EFFECT_RGB, "PF30026_EFFECT_RGB")?;
        let effect_pixels = (effect.shape[2] as usize)
            .checked_mul(effect.shape[3] as usize)
            .ok_or(Refusal::ShapeMismatch("PF30026 effect RGB"))?;
        if effect.bytes().len() != effect_pixels * 4 {
            return Err(Refusal::ShapeMismatch("PF30026 effect RGB"));
        }
        let effect_rgb = Image::new(effect.shape[3], effect.shape[2], effect.bytes().to_vec())
            .ok_or(Refusal::ShapeMismatch("PF30026 effect RGB"))?;
        let teeth_mask = self
            .required(PF30026_TEETH_MASK, "PF30026_TEETH_MASK")?
            .bytes()
            .to_vec();
        if teeth_mask.len() != effect_pixels {
            return Err(Refusal::ShapeMismatch("PF30026 teeth mask"));
        }
        let edge_0 = self.values(PF30026_EDGE_0, "PF30026_EDGE_0")?.to_vec();
        let edge_1 = self.values(PF30026_EDGE_1, "PF30026_EDGE_1")?.to_vec();
        if edge_0.len() != effect_pixels || edge_1.len() != effect_pixels {
            return Err(Refusal::ShapeMismatch("PF30026 edge planes"));
        }
        Ok(TeethAlignmentProducerInputs {
            effect_rgb: Some(effect_rgb),
            teeth_mask: Some(teeth_mask),
            edge_plane_a: Some(match gate_binding.plane_a {
                Pf30026EdgeOutput::Output0 => edge_0.clone(),
                Pf30026EdgeOutput::Output1 => edge_1.clone(),
            }),
            edge_plane_b: Some(match gate_binding.plane_b {
                Pf30026EdgeOutput::Output0 => edge_0,
                Pf30026EdgeOutput::Output1 => edge_1,
            }),
            gate_binding: Some(gate_binding),
            roi: Some(roi),
        })
    }

    pub fn eye_bag_tensors(&self) -> Result<EyeBagModelTensors<'_>, Refusal> {
        if self.target != PortraitModelTarget::Pf90116 {
            return Err(Refusal::ValueOutOfRange("PF90116 target"));
        }
        let input = self.required("781141184", "PF90116_MODEL_INPUT_781141184")?;
        let response = self.required("3204548346", "PF90116_MODEL_RESPONSE_3204548346")?;
        let color = self.required("780929437", "PF90116_MODEL_COLOR_780929437")?;
        Ok(EyeBagModelTensors {
            model_id: input.model_id,
            input_781141184_rgb_chw: input.values()?,
            response_3204548346_hw: response.values()?,
            color_780929437_rgb_chw: color.values()?,
        })
    }

    pub fn pf90130_producer(&self) -> Result<Pf90130GlareProducer, Refusal> {
        if self.target != PortraitModelTarget::Pf90130 {
            return Err(Refusal::ValueOutOfRange("PF90130 target"));
        }
        let input = self.required("PF90130_INPUT", "PF90130_INPUT")?;
        let effect = self.required("PF90130_EFFECT", "PF90130_EFFECT")?;
        let mask = self.required("PF90130_MASK", "PF90130_MASK")?;
        Pf90130GlareProducer::from_same_job(
            self.job_id,
            input.model_id as u32,
            PF90130_INPUT_TENSOR_ID,
            input.bytes(),
            PF90130_EFFECT_TENSOR_ID,
            effect.bytes(),
            PF90130_MASK_TENSOR_ID,
            mask.bytes(),
        )
    }

    pub fn stray_hair_inputs(
        &self,
        binding: StrayHairModelBinding<'_>,
    ) -> Result<StrayHairProducerInputs, Refusal> {
        if self.target != PortraitModelTarget::Pf90088 {
            return Err(Refusal::ValueOutOfRange("PF90088 target"));
        }
        binding
            .frame
            .validate_against(self.job_id, self.source_digest)?;
        binding
            .hairfly_v1
            .validate()
            .map_err(|_| Refusal::ShapeMismatch("PF90088 hairfly"))?;
        if (binding.frame.width, binding.frame.height)
            != (binding.hairfly_v1.width, binding.hairfly_v1.height)
        {
            return Err(Refusal::ShapeMismatch("PF90088 hairfly frame"));
        }
        let selector = self.required(PF90088_SELECTOR, "PF90088_SELECTOR")?;
        let logits = selector.values()?;
        if logits.len() != 2 {
            return Err(Refusal::ShapeMismatch("PF90088 selector logits"));
        }
        let selector_receipt = HairFlySelectorReceipt {
            same_job: SameJobFace {
                job_id: self.job_id,
                face_index: binding.face_index,
            },
            model_hash_id: selector.model_id as u32,
            output_name: SELECTOR_OUTPUT_NAME,
            logits: [logits[0], logits[1]],
            aux_gate_high: binding.aux_gate_high,
        };
        let native_receipt = HairFlyNativeFaceReceipt {
            same_job: selector_receipt.same_job,
            pf90088: binding.pf90088,
            pf90113: binding.pf90113,
            pf90114: binding.pf90114,
        };
        let route = resolve_hairfly_producer(&selector_receipt, &native_receipt)
            .map_err(hairfly_refusal)?;
        let port =
            self.hairfly_tensor(binding.port_output.tensor_name(), route.model, "PF90088_MODEL_ROUTE")?;
        if port.shape[1] != 3 || port.digest != binding.port_source_sha256 {
            return Err(Refusal::ValueOutOfRange("PF90088_PORT_DECODE_RECEIPT"));
        }
        let mask_a =
            self.hairfly_tensor(PF21026_MASK_158607435, route.model, "PF90088_MODEL_ROUTE")?;
        let mask_b =
            self.hairfly_tensor(PF21026_MASK_2405607714, route.model, "PF90088_MODEL_ROUTE")?;
        let mask_decode = binding
            .mask_decode
            .ok_or(Refusal::MissingInput("PF90088_RAW_MASK_CHANNEL_RECEIPT"))?;
        mask_decode.validate(
            self.job_id,
            binding.face_index,
            route.model,
            binding.frame.width,
            binding.frame.height,
        )?;
        if mask_a.shape[1] != 1 || mask_b.shape[1] != 3 {
            return Err(Refusal::MissingInput("PF90088_RAW_MASK_CHANNEL_DECODE"));
        }
        if hairfly_mask_plane_sha256(mask_b, mask_decode.channel)?
            != mask_decode.mask_b_plane_sha256
        {
            return Err(Refusal::ValueOutOfRange(
                "PF90088_RAW_MASK_CHANNEL_RECEIPT",
            ));
        }
        if sha256(binding.raw_mask) != mask_decode.raw_mask_sha256 {
            return Err(Refusal::ValueOutOfRange(
                "PF90088_RAW_MASK_POSTPROCESS_RECEIPT",
            ));
        }
        let mut mask_pair = Vec::with_capacity(64);
        mask_pair.extend_from_slice(&mask_a.digest);
        mask_pair.extend_from_slice(&mask_b.digest);
        if sha256(&mask_pair) != binding.raw_mask_source_sha256 {
            return Err(Refusal::ValueOutOfRange("PF90088_RAW_MASK_DECODE_RECEIPT"));
        }
        let pixels = (binding.frame.width as usize)
            .checked_mul(binding.frame.height as usize)
            .ok_or(Refusal::ShapeMismatch("PF90088 raw mask"))?;
        if binding.raw_mask.len() != pixels {
            return Err(Refusal::ShapeMismatch("PF90088 raw mask"));
        }
        for (mask, role) in [
            (binding.person_mask, "PF90088_PERSON_MASK"),
            (binding.background_mask, "PF90088_BACKGROUND_MASK"),
        ] {
            if matches!(mask, Some(mask) if mask.len() != pixels) {
                return Err(Refusal::ShapeMismatch(role));
            }
        }
        Ok(StrayHairProducerInputs {
            hairfly_v1: Some(binding.hairfly_v1.clone()),
            raw_mask: Some(binding.raw_mask.to_vec()),
            person_mask: binding.person_mask.map(ToOwned::to_owned),
            background_mask: binding.background_mask.map(ToOwned::to_owned),
            native_degree: Some(route.native_index3.degree),
            native_person_area: Some(binding.pf90113),
            native_background_area: Some(binding.pf90114),
        })
    }

    pub fn hair_smooth_inputs(
        &self,
        binding: HairSmoothBinding,
    ) -> Result<HairSmoothInputs<'_>, Refusal> {
        if self.target != PortraitModelTarget::Pf90139Pf90160 {
            return Err(Refusal::ValueOutOfRange("PF90139 target"));
        }
        Ok(HairSmoothInputs {
            ticket: self.job_id,
            width: binding.width,
            height: binding.height,
            roi: binding.roi,
            mix_strength: binding.mix_strength,
            hair_mask: Some(
                self.required(PF90139_HAIR_MASK, "PF90139_HAIR_MASK")?
                    .bytes(),
            ),
            mix_mask: Some(self.required(PF90139_MIX_MASK, "PF90139_MIX_MASK")?.bytes()),
            blend_rgb: Some(
                self.required(PF90139_BLEND_RGB, "PF90139_BLEND_SOURCE")?
                    .bytes(),
            ),
            detail: Some(self.required(PF90139_DETAIL, "PF90139_DETAIL")?.bytes()),
            hair_stride: binding.hair_stride,
            mix_stride: binding.mix_stride,
            blend_stride: binding.blend_stride,
            detail_stride: binding.detail_stride,
        })
    }

    pub fn hair_gloss_inputs(
        &self,
        binding: HairGlossBinding,
    ) -> Result<HairGlossInputs<'_>, Refusal> {
        if self.target != PortraitModelTarget::Pf90139Pf90160 {
            return Err(Refusal::ValueOutOfRange("PF90160 target"));
        }
        Ok(HairGlossInputs {
            ticket: self.job_id,
            width: binding.width,
            height: binding.height,
            roi: binding.roi,
            dynamic_scale: binding.dynamic_scale,
            hair_mask: Some(
                self.required(PF90160_HAIR_MASK, "PF90160_HAIR_MASK")?
                    .bytes(),
            ),
            model_detail: Some(
                self.required(PF90160_DETAIL, "PF90160_MODEL_DETAIL")?
                    .bytes(),
            ),
            hair_stride: binding.hair_stride,
            detail_stride: binding.detail_stride,
        })
    }

    pub fn moisture_face_inputs(
        &self,
        binding: MoistureFaceBinding,
    ) -> Result<MoistureFaceInputs<'_>, Refusal> {
        if self.target != PortraitModelTarget::Pf90002Pf90010 {
            return Err(Refusal::ValueOutOfRange("PF90002 target"));
        }
        binding
            .frame
            .validate_against(self.job_id, self.source_digest)?;
        let pixels = binding
            .width
            .checked_mul(binding.height)
            .map(|value| value as usize)
            .ok_or(Refusal::ShapeMismatch("PF90002 frame"))?;
        let mask_rgba = self
            .required(PF90002_MASK_RGBA, "PF90002_MASK_RGBA")?
            .bytes();
        let blur_alpha = self
            .required(PF90002_BLUR_ALPHA, "PF90002_BLUR_ALPHA")?
            .bytes();
        if mask_rgba.len() != pixels.saturating_mul(4) || blur_alpha.len() != pixels {
            return Err(Refusal::ShapeMismatch("PF90002 frame planes"));
        }
        let skin_pixels = (binding.skin_map_width as usize)
            .checked_mul(binding.skin_map_height as usize)
            .ok_or(Refusal::ShapeMismatch("PF90002 skin map"))?;
        let skin_map = self.values(PF90002_SKIN_MAP, "PF90002_SKIN_MAP")?;
        let hardlight = self.values(PF90002_HARDLIGHT, "PF90002_HARDLIGHT")?;
        let beauty = self.values(PF90002_BEAUTY, "PF90002_BEAUTY")?;
        if skin_map.len() != skin_pixels || hardlight.is_empty() || beauty.is_empty() {
            return Err(Refusal::ShapeMismatch("PF90002 model planes"));
        }
        Ok(MoistureFaceInputs {
            width: binding.width,
            height: binding.height,
            mask_rgba,
            blur_alpha,
            channel: binding.channel,
            skin_map_width: binding.skin_map_width,
            skin_map_height: binding.skin_map_height,
            skin_map,
            hardlight,
            beauty,
            pivot: binding.pivot,
            gain: binding.gain,
        })
    }

    pub fn moisture_body_inputs(
        &self,
        binding: MoistureBodyBinding,
    ) -> Result<MoistureBodyInputs<'_>, Refusal> {
        if self.target != PortraitModelTarget::Pf90002Pf90010 {
            return Err(Refusal::ValueOutOfRange("PF90010 target"));
        }
        binding
            .frame
            .validate_against(self.job_id, self.source_digest)?;
        let pixels = binding
            .width
            .checked_mul(binding.height)
            .map(|value| value as usize)
            .ok_or(Refusal::ShapeMismatch("PF90010 frame"))?;
        let weight_skin = self.values(PF90010_WEIGHT_SKIN, "PF90010_WEIGHT_SKIN")?;
        let weight_dog = self.values(PF90010_WEIGHT_DOG, "PF90010_WEIGHT_DOG")?;
        let beauty = self.values(PF90010_BEAUTY, "PF90010_BEAUTY")?;
        if weight_skin.len() != pixels || weight_dog.len() != pixels || beauty.is_empty() {
            return Err(Refusal::ShapeMismatch("PF90010 model planes"));
        }
        Ok(MoistureBodyInputs {
            width: binding.width,
            height: binding.height,
            weight_skin,
            weight_dog,
            beauty,
            degree: binding.degree,
            roi: binding.roi,
        })
    }

    pub fn smile_inputs(&self) -> Result<SmileProducerInputs, Refusal> {
        if self.target != PortraitModelTarget::Pf100007 {
            return Err(Refusal::ValueOutOfRange("PF100007 target"));
        }
        let color = self.required("780929437", "PF100007_MODEL_FIELD_780929437")?;
        let response = self.required("3204548346", "PF100007_MODEL_RESPONSE_3204548346")?;
        let [_, color_channels, color_height, color_width] = color.shape;
        let [_, response_channels, response_height, response_width] = response.shape;
        if color_channels != 3
            || response_channels != 1
            || color_height != response_height
            || color_width != response_width
        {
            return Err(Refusal::ShapeMismatch("PF100007 model grid"));
        }
        Ok(SmileProducerInputs {
            ticket: self.job_id,
            sigmoid_edge_response_3204548346: Some(SmileSigmoidEdgeResponse {
                width: color_width,
                height: color_height,
                values: response.values()?.to_vec(),
            }),
            rgb_reconstruction_780929437: Some(SmileRgbReconstructionField {
                width: color_width,
                height: color_height,
                values: color.values()?.to_vec(),
            }),
            ..SmileProducerInputs::default()
        })
    }

    pub fn smile_inputs_with_assets(
        &self,
        assets: SmileAssetBinding<'_>,
    ) -> Result<SmileProducerInputs, Refusal> {
        if self.target != PortraitModelTarget::Pf100007 {
            return Err(Refusal::ValueOutOfRange("PF100007 target"));
        }
        assets
            .frame
            .validate_against(self.job_id, self.source_digest)?;
        if assets.material.is_empty() {
            return Err(Refusal::MissingInput("PF100007_MATERIAL"));
        }
        if assets.material_sha256 == [0; 32] {
            return Err(Refusal::MissingInput("PF100007_MATERIAL_SHA256"));
        }
        let material_uv = assets
            .material_uv
            .ok_or(Refusal::MissingInput("PF100007_MATERIAL_UV_SOURCE"))?;
        let pixels = (assets.frame.width as usize)
            .checked_mul(assets.frame.height as usize)
            .ok_or(Refusal::ShapeMismatch("PF100007 asset frame"))?;
        if assets.lip_mask.is_empty() || assets.mouth_mask.is_empty() {
            return Err(Refusal::MissingInput(if assets.lip_mask.is_empty() {
                "PF100007_LIP_MASK_SOURCE"
            } else {
                "PF100007_MOUTH_MASK_SOURCE"
            }));
        }
        if assets.lip_mask.len() != pixels || assets.mouth_mask.len() != pixels {
            return Err(Refusal::ShapeMismatch("PF100007 masks/assets"));
        }
        if material_uv.len() != pixels
            || material_uv
                .iter()
                .flat_map(|point| point.iter())
                .any(|value| !value.is_finite())
        {
            return Err(Refusal::ShapeMismatch("PF100007 material_uv"));
        }
        let (roi_x, roi_y, roi_width, roi_height) = assets.roi;
        if assets.roi.0 > assets.frame.width
            || roi_y > assets.frame.height
            || roi_width > assets.frame.width - roi_x
            || roi_height > assets.frame.height - roi_y
        {
            return Err(Refusal::ValueOutOfRange("PF100007 roi"));
        }
        let host_split = assets
            .host_split_spec
            .ok_or(Refusal::MissingInput("PF100007_DX_DY_HOST_SPLIT"))?;
        if (
            host_split.frame_roi.x,
            host_split.frame_roi.y,
            host_split.frame_roi.width,
            host_split.frame_roi.height,
        ) != assets.roi
        {
            return Err(Refusal::ValueOutOfRange("PF100007_ROI_SPLIT_MISMATCH"));
        }
        let mut smile = self.smile_inputs()?;
        smile.material = Some(assets.material.to_owned());
        smile.material_uv = Some(material_uv.to_owned());
        smile.material_sha256 = Some(assets.material_sha256);
        smile.lip_mask = Some(assets.lip_mask.to_vec());
        smile.mouth_mask = Some(assets.mouth_mask.to_vec());
        smile.roi = Some(assets.roi);
        smile.host_split_spec = Some(host_split);
        let channel_split = smile.decode_channel_split(assets.frame.width, assets.frame.height)?;
        smile.channel_split = Some(channel_split);
        Ok(smile)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TeethMaskBinding<'a> {
    pub job_id: u64,
    pub width: u32,
    pub height: u32,
    pub source: TeethMaskSource,
    pub format: TeethMaskFormat,
    pub mask_g: &'a [u8],
    pub mask_sha256: [u8; 32],
    pub bright_ui: f32,
    pub deyellow_ui: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrayHairPortOutput {
    MaskedDelta,
    Repair,
    CoarseRepair,
}

impl StrayHairPortOutput {
    fn tensor_name(self) -> &'static str {
        match self {
            Self::MaskedDelta => PF21026_MASK_3434589800,
            Self::Repair => PF21026_MASK_2977284499,
            Self::CoarseRepair => PF21026_MASK_2892147817,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct StrayHairModelBinding<'a> {
    pub frame: SameFrame,
    pub face_index: u32,
    pub aux_gate_high: bool,
    pub pf90088: f32,
    pub pf90113: f32,
    pub pf90114: f32,
    pub port_output: StrayHairPortOutput,
    pub hairfly_v1: &'a Image,
    pub raw_mask: &'a [u8],
    pub port_source_sha256: [u8; 32],
    pub raw_mask_source_sha256: [u8; 32],
    pub mask_decode: Option<HairFlyMaskDecodeReceipt>,
    pub person_mask: Option<&'a [u8]>,
    pub background_mask: Option<&'a [u8]>,
}

#[derive(Clone, Copy, Debug)]
pub struct HairSmoothBinding {
    pub width: u32,
    pub height: u32,
    pub roi: (u32, u32, u32, u32),
    pub mix_strength: f32,
    pub hair_stride: u32,
    pub mix_stride: u32,
    pub blend_stride: u32,
    pub detail_stride: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct HairGlossBinding {
    pub width: u32,
    pub height: u32,
    pub roi: (u32, u32, u32, u32),
    pub dynamic_scale: f32,
    pub hair_stride: u32,
    pub detail_stride: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct MoistureFaceBinding {
    pub frame: SameFrame,
    pub width: u32,
    pub height: u32,
    pub channel: MaskChannel,
    pub skin_map_width: u32,
    pub skin_map_height: u32,
    pub pivot: f32,
    pub gain: f32,
}

#[derive(Clone, Copy, Debug)]
pub struct MoistureBodyBinding {
    pub frame: SameFrame,
    pub width: u32,
    pub height: u32,
    pub degree: f32,
    pub roi: (u32, u32, u32, u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SameFrame {
    pub job_id: u64,
    pub source_digest: [u8; 32],
    pub width: u32,
    pub height: u32,
}

impl SameFrame {
    fn validate_against(self, job_id: u64, source_digest: [u8; 32]) -> Result<(), Refusal> {
        if self.job_id != job_id {
            return Err(Refusal::ValueOutOfRange("same-frame job"));
        }
        if self.source_digest != source_digest {
            return Err(Refusal::ValueOutOfRange("same-frame source"));
        }
        if self.width == 0 || self.height == 0 {
            return Err(Refusal::ValueOutOfRange("same-frame dimensions"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BeardRepairBinding<'a> {
    pub frame: SameFrame,
    pub aux_gate_high: bool,
    pub repair_output: BeardRepairOutput,
    pub mask_a: &'a [u8],
    pub mask_b: &'a [u8],
    pub mask_b_channel: HairFlyMaskChannel,
    pub repair_rgb: &'a Image,
    pub mask_a_source_sha256: [u8; 32],
    pub mask_b_source_sha256: [u8; 32],
    pub mask_b_decoded_sha256: [u8; 32],
    pub repair_source_sha256: [u8; 32],
    pub consumer: Option<Pf21026RepairConsumerReceipt>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BeardRepairOutput {
    Output3,
    Output4,
}

impl BeardRepairOutput {
    fn tensor_name(self) -> &'static str {
        match self {
            Self::Output3 => PF21026_MASK_2977284499,
            Self::Output4 => PF21026_MASK_2892147817,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pf21026RepairConsumerReceipt {
    pub job_id: u64,
    pub output: BeardRepairOutput,
    pub producer_rva: u32,
    pub producer_lookup_slot: u16,
    pub consumer_rva: u32,
    pub native_mat_offset: u16,
    pub native_mat_sha256: [u8; 32],
}

fn hairfly_mask_plane_sha256(
    tensor: &SessionTensor,
    channel: HairFlyMaskChannel,
) -> Result<[u8; 32], Refusal> {
    if tensor.shape[1] != 3 {
        return Err(Refusal::ShapeMismatch("PF90088 mask_b channels"));
    }
    let pixels = (tensor.shape[2] as usize)
        .checked_mul(tensor.shape[3] as usize)
        .ok_or(Refusal::ShapeMismatch("PF90088 mask_b pixels"))?;
    let values = tensor.values()?;
    let start = channel.index() * pixels;
    let plane = values
        .get(start..start + pixels)
        .ok_or(Refusal::ShapeMismatch("PF90088 mask_b channel"))?;
    if plane.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("PF90088 mask_b values"));
    }
    let mut bytes = Vec::with_capacity(pixels * 4);
    for value in plane {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(sha256(&bytes))
}

impl HairFlyMaskDecodeReceipt {
    fn validate(
        self,
        job_id: u64,
        face_index: u32,
        route: HairFlyRegisteredModel,
        frame_width: u32,
        frame_height: u32,
    ) -> Result<(), Refusal> {
        if self.same_job.job_id != job_id || self.same_job.face_index != face_index {
            return Err(Refusal::ValueOutOfRange("PF90088_ROI_WARP_SAME_JOB"));
        }
        if self.model_hash_id != route.hash_id {
            return Err(Refusal::ValueOutOfRange("PF90088_ROI_WARP_MODEL"));
        }
        check_receipt_roi(
            self.draw_buffer.model_roi,
            u32::from(route.width),
            u32::from(route.height),
            "PF90088_ROI_WARP_MODEL_ROI",
        )?;
        check_receipt_roi(
            self.draw_buffer.frame_roi,
            frame_width,
            frame_height,
            "PF90088_ROI_WARP_FRAME_ROI",
        )?;
        if self
            .draw_buffer
            .warp
            .iter()
            .flatten()
            .any(|value| !value.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("PF90088_ROI_WARP_MATRIX"));
        }
        if self.draw_buffer.vertex_count == 0
            || self.draw_buffer.vertex_buffer_sha256 == [0; 32]
        {
            return Err(Refusal::MissingInput("PF90088_ROI_WARP_DRAW_BUFFER"));
        }
        Ok(())
    }
}

fn check_receipt_roi(
    roi: [u32; 4],
    frame_width: u32,
    frame_height: u32,
    name: &'static str,
) -> Result<(), Refusal> {
    let [x, y, width, height] = roi;
    if width == 0
        || height == 0
        || !x.checked_add(width).is_some_and(|right| right <= frame_width)
        || !y
            .checked_add(height)
            .is_some_and(|bottom| bottom <= frame_height)
    {
        return Err(Refusal::ValueOutOfRange(name));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub struct SmileAssetBinding<'a> {
    pub frame: SameFrame,
    pub material: &'a str,
    pub material_sha256: [u8; 32],
    pub material_uv: Option<&'a [[f32; 2]]>,
    pub lip_mask: &'a [u8],
    pub mouth_mask: &'a [u8],
    pub roi: (u32, u32, u32, u32),
    pub host_split_spec: Option<crate::portrait_stage::SmileHostSplitSpec>,
}

fn hairfly_refusal(error: HairFlyProducerRefusal) -> Refusal {
    match error {
        HairFlyProducerRefusal::JobMismatch => Refusal::ValueOutOfRange("PF90088_NATIVE_JOB"),
        HairFlyProducerRefusal::FaceMismatch => Refusal::ValueOutOfRange("PF90088_NATIVE_FACE"),
        HairFlyProducerRefusal::SelectorModelMismatch => {
            Refusal::ValueOutOfRange("PF90088_SELECTOR_MODEL")
        }
        HairFlyProducerRefusal::SelectorOutputMismatch => {
            Refusal::ValueOutOfRange("PF90088_SELECTOR_OUTPUT")
        }
        HairFlyProducerRefusal::NativeNonFinite(field) => Refusal::ValueOutOfRange(field),
        HairFlyProducerRefusal::NativeOutOfRange(field) => Refusal::ValueOutOfRange(field),
        HairFlyProducerRefusal::SelectorNonFiniteLogit
        | HairFlyProducerRefusal::SelectorNotSameJob
        | HairFlyProducerRefusal::SelectorArgmaxOutOfRange => {
            Refusal::ValueOutOfRange("PF90088_SELECTOR_RECEIPT")
        }
    }
}

fn checked_len(shape: [u32; 4]) -> Option<usize> {
    let [n, c, h, w] = shape;
    (n == 1 && c > 0 && h > 0 && w > 0)
        .then(|| {
            (n as usize)
                .checked_mul(c as usize)
                .and_then(|value| value.checked_mul(h as usize))
                .and_then(|value| value.checked_mul(w as usize))
        })
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    fn output(
        job: u64,
        source: [u8; 32],
        model: u64,
        name: &'static str,
        shape: [u32; 4],
        values: &[f32],
    ) -> SessionTensor {
        SessionTensor::from_f32(job, source, model, name, shape, values).unwrap()
    }

    fn pf100007_outputs(job: u64, source: [u8; 32]) -> SessionModelOutputs {
        let mut outputs = SessionModelOutputs::new();
        outputs
            .insert(output(
                job,
                source,
                806_921_480,
                "780929437",
                [1, 3, 768, 1024],
                &vec![0.25; 3 * 768 * 1024],
            ))
            .unwrap();
        outputs
            .insert(output(
                job,
                source,
                806_921_480,
                "3204548346",
                [1, 1, 768, 1024],
                &vec![0.5; 768 * 1024],
            ))
            .unwrap();
        outputs
    }

    #[test]
    fn vendor_catalog_contains_the_eight_decoded_payloads() {
        assert_eq!(VENDOR_MODELS.len(), 8);
        assert_eq!(
            vendor_model(1_655_581_403).unwrap().source_sha256,
            "6a025f05c26aca02def0d81e2c443d0e3dc01e61f2af00a525aca6d4ca60d14d"
        );
        assert_eq!(
            vendor_model(3_157_078_880).unwrap().derived_mnn_sha256,
            Some("d5ea7f78d631f6fa2c7949dcf60a895970e58fe349c595c84913335a578ec52c")
        );
    }

    #[test]
    fn complete_pf100007_binds_without_rewriting_model_data() {
        let job = 41;
        let source = digest(7);
        let outputs = pf100007_outputs(job, source);
        let bound = outputs
            .bind(PortraitModelTarget::Pf100007, job, source)
            .unwrap();
        let smile = bound.smile_inputs().unwrap();
        assert_eq!(smile.ticket, job);
        assert_eq!(
            smile.rgb_reconstruction_780929437.as_ref().unwrap().values[0],
            0.25
        );
    }

    #[test]
    fn pf100007_assets_require_uv_masks_and_host_split() {
        let job = 43;
        let source = digest(13);
        let outputs = pf100007_outputs(job, source);
        let bound = outputs
            .bind(PortraitModelTarget::Pf100007, job, source)
            .unwrap();
        let frame = SameFrame {
            job_id: job,
            source_digest: source,
            width: 1,
            height: 1,
        };
        let missing_uv = SmileAssetBinding {
            frame,
            material: "material",
            material_sha256: [1; 32],
            material_uv: None,
            lip_mask: &[1],
            mouth_mask: &[1],
            roi: (0, 0, 1, 1),
            host_split_spec: None,
        };
        assert_eq!(
            bound.smile_inputs_with_assets(missing_uv),
            Err(Refusal::MissingInput(
                "PF100007_MATERIAL_UV_LIP_MOUTH_MASK"
            ))
        );

        let uv = [[0.0, 0.0]];
        let missing_split = SmileAssetBinding {
            material_uv: Some(&uv),
            ..missing_uv
        };
        assert_eq!(
            bound.smile_inputs_with_assets(missing_split),
            Err(Refusal::MissingInput("PF100007_DX_DY_HOST_SPLIT"))
        );
    }

    #[test]
    fn every_missing_target_tensor_is_named() {
        let job = 42;
        let source = digest(8);
        let outputs = SessionModelOutputs::new();
        assert_eq!(
            outputs.bind(PortraitModelTarget::Pf90116, job, source),
            Err(Refusal::MissingInput("781141184"))
        );
    }

    #[test]
    fn wrong_job_and_source_are_rejected() {
        let source = digest(9);
        let mut outputs = SessionModelOutputs::new();
        outputs
            .insert(output(
                1,
                source,
                9,
                "780929437",
                [1, 3, 768, 1024],
                &vec![0.0; 3 * 768 * 1024],
            ))
            .unwrap();
        assert_eq!(
            outputs.bind(PortraitModelTarget::Pf100007, 2, source),
            Err(Refusal::ValueOutOfRange("PF100007_MODEL_JOB"))
        );
        assert_eq!(
            outputs.bind(PortraitModelTarget::Pf100007, 1, digest(10)),
            Err(Refusal::ValueOutOfRange("PF100007_MODEL_SOURCE"))
        );
    }

    #[test]
    fn duplicate_tensor_names_are_rejected() {
        let mut outputs = SessionModelOutputs::new();
        let source = digest(11);
        let tensor = output(1, source, 9, "780929437", [1, 1, 1, 1], &[0.0]);
        outputs.insert(tensor.clone()).unwrap();
        assert_eq!(
            outputs.insert(tensor),
            Err(Refusal::ValueOutOfRange("duplicate model output tensor"))
        );
    }

    #[test]
    fn digest_is_recomputed_and_shape_is_checked() {
        let source = digest(12);
        let tensor = output(1, source, 9, "t", [1, 1, 1, 2], &[0.0, 1.0]);
        assert_eq!(tensor.digest, sha256(tensor.bytes()));
        assert_eq!(
            SessionTensor::from_f32(1, source, 9, "t", [1, 1, 1, 2], &[0.0]),
            Err(Refusal::ShapeMismatch("model output values"))
        );
    }

    #[test]
    fn source_is_required() {
        assert_eq!(
            SessionTensor::from_f32(1, [0; 32], 9, "t", [1, 1, 1, 1], &[0.0]),
            Err(Refusal::MissingInput("model output source"))
        );
    }

    #[test]
    fn target_matrix_covers_all_requested_rows() {
        assert_eq!(PortraitModelTarget::ALL.len(), 10);
        for target in PortraitModelTarget::ALL {
            assert!(!target.requirements().is_empty(), "{}", target.label());
        }
    }
}
