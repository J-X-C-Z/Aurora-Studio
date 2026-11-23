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

pub const PF30026_EFFECT_RGB: &str = "PF30026_EFFECT_RGB";
pub const PF30026_TEETH_MASK: &str = "PF30026_TEETH_MASK";
pub const PF30026_EDGE_0: &str = "PF30026_EDGE_0";
pub const PF30026_EDGE_1: &str = "PF30026_EDGE_1";

pub const PF90088_HAIRFLY_PORT: &str = "PF90088_HAIRFLY_PORT";
pub const PF90088_RAW_MASK: &str = "PF90088_RAW_STRAY_HAIR_MASK";

pub const PF90139_HAIR_MASK: &str = "PF90139_HAIR_MASK";
pub const PF90139_MIX_MASK: &str = "PF90139_MIX_MASK";
pub const PF90139_BLEND_RGB: &str = "PF90139_BLEND_SOURCE";
pub const PF90139_DETAIL: &str = "PF90139_DETAIL";

pub const PF90160_HAIR_MASK: &str = "PF90160_HAIR_MASK";
pub const PF90160_DETAIL: &str = "PF90160_MODEL_DETAIL";

pub const PF90002_MASK_RGBA: &str = "PF90002_MASK_RGBA";
pub const PF90002_BLUR_Y: &str = "PF90002_BLUR_Y";
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
                    PF21026_MASK_158607435,
                    PF21026_MASK_158607435,
                    NonEmpty,
                    Exact(3_378_116_080),
                    0,
                ),
                TensorRequirement::new(
                    PF21026_MASK_2405607714,
                    PF21026_MASK_2405607714,
                    NonEmpty,
                    Exact(3_378_116_080),
                    0,
                ),
                TensorRequirement::new(
                    PF21026_MASK_3434589800,
                    PF21026_MASK_3434589800,
                    NonEmpty,
                    Exact(3_378_116_080),
                    0,
                ),
                TensorRequirement::new(
                    PF21026_MASK_2977284499,
                    PF21026_MASK_2977284499,
                    NonEmpty,
                    Exact(3_378_116_080),
                    0,
                ),
                TensorRequirement::new(
                    PF21026_MASK_2892147817,
                    PF21026_MASK_2892147817,
                    NonEmpty,
                    Exact(3_378_116_080),
                    0,
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
                    PF90088_HAIRFLY_PORT,
                    "PF90088_HAIRFLY_PORT",
                    NonEmpty,
                    NonZero,
                    0,
                ),
                TensorRequirement::new(
                    PF90088_RAW_MASK,
                    "PF90088_RAW_STRAY_HAIR_MASK",
                    NonEmpty,
                    NonZero,
                    0,
                ),
                TensorRequirement::new("158607435", "158607435", NonEmpty, NonZero, 0),
                TensorRequirement::new("2405607714", "2405607714", NonEmpty, NonZero, 0),
                TensorRequirement::new("3434589800", "3434589800", NonEmpty, NonZero, 0),
                TensorRequirement::new("2977284499", "2977284499", NonEmpty, NonZero, 0),
                TensorRequirement::new("2892147817", "2892147817", NonEmpty, NonZero, 0),
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
                TensorRequirement::new(PF90002_BLUR_Y, PF90002_BLUR_Y, NonEmpty, NonZero, 1),
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
                    NonZero,
                    0,
                ),
                TensorRequirement::new(
                    "3204548346",
                    "3204548346",
                    ExactShape([1, 1, 768, 1024]),
                    NonZero,
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

    pub fn beard<'b>(
        &'b self,
        mask_a: &'static str,
        mask_b: &'static str,
        repair_rgb: &'b Image,
    ) -> Result<BeardInputs<'b>, Refusal> {
        if self.target != PortraitModelTarget::Pf21026 {
            return Err(Refusal::ValueOutOfRange("PF21026 target"));
        }
        Ok(BeardInputs {
            mask_a: self.required(mask_a, "PF21026_MASK_A")?.bytes(),
            mask_b: self.required(mask_b, "PF21026_MASK_B")?.bytes(),
            repair_rgb,
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
        width: u32,
        height: u32,
        hairfly_port: &'static str,
        companion_masks: StrayHairCompanions<'_>,
        native: StrayHairNative,
    ) -> Result<StrayHairProducerInputs, Refusal> {
        if self.target != PortraitModelTarget::Pf90088 {
            return Err(Refusal::ValueOutOfRange("PF90088 target"));
        }
        let hairfly = self.required(hairfly_port, "PF90088_HAIRFLY_PORT")?;
        let raw = self.required(PF90088_RAW_MASK, "PF90088_RAW_STRAY_HAIR_MASK")?;
        if hairfly.bytes().len() != (width as usize) * (height as usize) * 4 {
            return Err(Refusal::ShapeMismatch("PF90088 hairfly"));
        }
        let hairfly_v1 = Image::new(width, height, hairfly.bytes().to_vec())
            .ok_or(Refusal::ShapeMismatch("PF90088 hairfly"))?;
        if raw.bytes().len() != (width as usize) * (height as usize) {
            return Err(Refusal::ShapeMismatch("PF90088 raw mask"));
        }
        Ok(StrayHairProducerInputs {
            hairfly_v1: Some(hairfly_v1),
            raw_mask: Some(raw.bytes().to_vec()),
            person_mask: companion_masks.person_mask.map(ToOwned::to_owned),
            background_mask: companion_masks.background_mask.map(ToOwned::to_owned),
            native_degree: Some(native.degree),
            native_person_area: Some(native.person_area),
            native_background_area: Some(native.background_area),
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
        Ok(MoistureFaceInputs {
            width: binding.width,
            height: binding.height,
            mask_rgba: self
                .required(PF90002_MASK_RGBA, "PF90002_MASK_RGBA")?
                .bytes(),
            blur_y: self.required(PF90002_BLUR_Y, "PF90002_BLUR_Y")?.bytes(),
            channel: binding.channel,
            skin_map_width: binding.skin_map_width,
            skin_map_height: binding.skin_map_height,
            skin_map: self.values(PF90002_SKIN_MAP, "PF90002_SKIN_MAP")?,
            hardlight: self.values(PF90002_HARDLIGHT, "PF90002_HARDLIGHT")?,
            beauty: self.values(PF90002_BEAUTY, "PF90002_BEAUTY")?,
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
        Ok(MoistureBodyInputs {
            width: binding.width,
            height: binding.height,
            weight_skin: self.values(PF90010_WEIGHT_SKIN, "PF90010_WEIGHT_SKIN")?,
            weight_dog: self.values(PF90010_WEIGHT_DOG, "PF90010_WEIGHT_DOG")?,
            beauty: self.values(PF90010_BEAUTY, "PF90010_BEAUTY")?,
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

#[derive(Clone, Copy, Debug, Default)]
pub struct StrayHairCompanions<'a> {
    pub person_mask: Option<&'a [u8]>,
    pub background_mask: Option<&'a [u8]>,
}

#[derive(Clone, Copy, Debug)]
pub struct StrayHairNative {
    pub degree: f32,
    pub person_area: f32,
    pub background_area: f32,
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
    pub width: u32,
    pub height: u32,
    pub degree: f32,
    pub roi: (u32, u32, u32, u32),
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

    #[test]
    fn complete_pf100007_binds_without_rewriting_model_data() {
        let job = 41;
        let source = digest(7);
        let mut outputs = SessionModelOutputs::new();
        outputs
            .insert(output(
                job,
                source,
                9,
                "780929437",
                [1, 3, 768, 1024],
                &vec![0.25; 3 * 768 * 1024],
            ))
            .unwrap();
        outputs
            .insert(output(
                job,
                source,
                9,
                "3204548346",
                [1, 1, 768, 1024],
                &vec![0.5; 768 * 1024],
            ))
            .unwrap();
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
