//! Native FaceParsingV2 producer.
//!
//! The recovered `2102828093.onnx` graph takes one RGB8 crop normalized to
//! `[1,3,384,384]` and returns probabilities `[1,17,384,384]`. The consumer
//! path quantizes each class, appends the four recovered class groups, then
//! uses `id2.G = LUT[slot20]` as the PF108 teeth support plane.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, bail, ensure};
#[cfg(windows)]
use aurora_portrait::cuda_model_preprocess::ModelPreprocess;
use aurora_portrait::cuda_model_preprocess::{QuantUniforms, f32_to_u8_cpu};
use aurora_portrait::model_sha256::{hex, sha256 as sha256_32};
use aurora_portrait::{Image, Refusal};

use crate::NormRect;
use crate::aimask::{Runtime, Session};
use crate::portrait_stage::{
    PF108_BRIGHT_LUT_IDENTITY, PF108_WHITEN_LUT_IDENTITY, Pf90004ProducerInputs,
    ProducerInputs, TeethMaskFormat, TeethMaskSource, TeethProducerInputs,
};

pub const FACE_PARSING_V2_MODEL_ID: &str = "2102828093";
pub const FACE_PARSING_V2_MODEL_FILE: &str = "fbe.onnx";
pub const FACE_PARSING_V2_SIDE: u32 = 384;
pub const FACE_PARSING_V2_CLASSES: usize = 17;
pub const FACE_PARSING_V2_SKIN_CLASS: usize = 1;
pub const FACE_PARSING_V2_MODEL_BYTES: u64 = 23_059_132;
pub const FACE_PARSING_V2_MODEL_ENV: &str = "AURORA_FACE_PARSING_MODEL";
pub const FACE_PARSING_V2_MODEL_ROOT_ENV: &str = "AURORA_FACE_PARSING_MODEL_ROOT";
pub const FACE_PARSING_V2_AIMODELS_ENV: &str = "AURORA_AIMODELS";
pub const FACE_PARSING_V2_MASK_BACKEND_ENV: &str = "AURORA_FACE_PARSING_MASK_BACKEND";
pub const FACE_PARSING_V2_EXPECT_PROVIDER_ENV: &str = "AURORA_FACE_PARSING_EXPECT_PROVIDER";
pub const FACE_PARSING_V2_MODEL_SHA256: [u8; 32] =
    hex32(b"fbe3fe1be3c0c287593b825075a9820315017e3cbebf266f519a6fc27be885b3");
pub const FACE_PARSING_V2_STALE_MODEL_SHA256: [u8; 32] =
    hex32(b"bedf8a42381cc1b447587645204ab77f6cd8c1cf8ecc0b321e44b2078ec9345f");
pub const FACE_PARSING_V2_TEETH_LUT_SHA256: [u8; 32] =
    hex32(b"6fcfab2f56596d37014b86a2a33e06de1119f9aa6d3b70b6648a96d4e47e5113");
pub const FACE_PARSING_V2_PREPARED_TENSOR_SHA256: [u8; 32] =
    hex32(b"7a31747aacd185c9c4d5c59fc2108e4f767d1d6c387676ba06b525d8e497bb7b");
pub const FACE_PARSING_V2_CACHED_ORT_MASKS_SHA256: [u8; 32] =
    hex32(b"71967bb3ef6199f1fa3f59c667b8b07938c631eae928def4b5ff6334bc42837e");
pub const FACE_PARSING_V2_FIXTURE_TEETH_CROP_SHA256: [u8; 32] =
    hex32(b"4d94406153b57ce6bbdce0cc534e5703550a80e64c8cac46e3c6ec9c626a7644");
pub const FACE_PARSING_V2_FIXTURE_TEETH_FULL_SHA256: [u8; 32] =
    hex32(b"790548d33979781b48a97267fcd746213d5ebd5a3fa773a323db1005e3f16209");

const TEETH_LUT_HEX: &[u8] = b"000101030405060708090a0b0d0e0e0f1012121315161718191b1c1d1e1f21212324252627292a2c2c2d2f31323334353738393b3c3d3e40414344464648494b4c4d4f505253555658595b5b5d5e606263656668696a6c6d6f707273757677797b7d7e7f8182848588898a8d8e9091949697999b9d9ea1a2a4a5a7a9abacafb0b2b4b5b8babbbcbec0c2c3c5c7c9cacbcdcfd0d1d2d4d6d7d8dadbdcdddee0e0e2e3e4e5e6e7e8e9eaeaecededeeeeeff0f1f1f2f2f3f3f4f5f6f6f6f6f8f7f8f8f8f9f9fafafafafbfbfcfbfbfcfcfdfdfdfdfdfdfdfdfdfefdfefdfefefefefefffefefffefefefffefefffefffefffffeffffffffffffffffffffffffffff";
const MODEL_MEAN: [f32; 3] = [123.675_003, 116.279_999, 103.529_999];
const MODEL_SCALE: [f32; 3] = [0.017_124_753, 0.017_507_004, 0.017_429_194];
// The class planes are contiguous, so one 1-channel launch preserves NCHW order.
// Subtracting 0.5 before midpoint-away rounding is exactly trunc(probability * 255).
const PROBABILITY_QUANT: QuantUniforms = QuantUniforms {
    width: FACE_PARSING_V2_SIDE,
    height: FACE_PARSING_V2_SIDE * FACE_PARSING_V2_CLASSES as u32,
    stride: FACE_PARSING_V2_SIDE,
    channels: 1,
    scale: 255.0,
    offset: -0.5,
    pad0: 0,
    pad1: 0,
};

/// PF90004 consumes four recovered class-group planes as RGBA. The first 17
/// slots remain the model's native classes; these groups are slots 17..=20.
pub const PF90004_PARSING_RGBA_CLASSES: [&[usize]; 4] = [
    &[0, 13, 14, 15, 16],
    &[2, 7, 11, 12],
    &[4, 5],
    &[9, 10],
];
const PF90004_FACE_CLASSES: &[usize] = &[1];
const WHITEN_PNG: &[u8] = include_bytes!("../../portrait/assets/portrait/teeth/Whiten.png");
const BRIGHT_PNG: &[u8] = include_bytes!("../../portrait/assets/portrait/teeth/Bright.png");

struct LutIdentity {
    source_file: &'static str,
    png_bytes: usize,
    png_sha256: [u8; 32],
    rgb8_bytes: usize,
    rgb8_sha256: [u8; 32],
    width: u32,
    height: u32,
}

const WHITEN_LUT_IDENTITY: LutIdentity = LutIdentity {
    source_file: "Whiten.png",
    png_bytes: 99_654,
    png_sha256: [
        0x09, 0xd4, 0x2d, 0xf0, 0xc1, 0x93, 0xac, 0xc7, 0x2f, 0x87, 0x00, 0xd6, 0x19, 0x7f,
        0xf5, 0xc2, 0xc2, 0xbd, 0xc3, 0x39, 0x26, 0x78, 0x71, 0x0a, 0xab, 0x0a, 0xec, 0x35,
        0x93, 0x51, 0x7c, 0xc4,
    ],
    rgb8_bytes: 786_432,
    rgb8_sha256: [
        0x1c, 0x78, 0x3a, 0xb2, 0xb3, 0x35, 0xb8, 0xd0, 0x36, 0xc7, 0x7c, 0xe9, 0x93, 0x47,
        0xc1, 0x8b, 0x73, 0xd0, 0xfc, 0x4b, 0x92, 0xed, 0xb3, 0xa3, 0x94, 0x71, 0x21, 0x0c,
        0x33, 0x9b, 0xa2, 0xed,
    ],
    width: 512,
    height: 512,
};

const BRIGHT_LUT_IDENTITY: LutIdentity = LutIdentity {
    source_file: "Bright.png",
    png_bytes: 196_345,
    png_sha256: [
        0xbb, 0xe8, 0x41, 0xa7, 0x51, 0xdd, 0x7a, 0xa7, 0x8d, 0x0d, 0x13, 0x86, 0xc3, 0xcc,
        0x68, 0x57, 0x22, 0xc6, 0x37, 0x1c, 0x3c, 0x91, 0x30, 0x41, 0xbc, 0x9b, 0xf3, 0x14,
        0x3c, 0x9d, 0x12, 0x6c,
    ],
    rgb8_bytes: 786_432,
    rgb8_sha256: [
        0x11, 0x2f, 0x2d, 0x16, 0xc3, 0x6c, 0xfd, 0xfd, 0xa0, 0x0e, 0x90, 0x2d, 0x74, 0x67,
        0x82, 0x0b, 0x0f, 0xd1, 0x75, 0xb1, 0x4b, 0x10, 0xf5, 0xc8, 0x0e, 0xf6, 0x11, 0xb2,
        0x1f, 0xb8, 0xa6, 0xf7,
    ],
    width: 512,
    height: 512,
};

const fn hex_nibble(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        b'A'..=b'F' => byte - b'A' + 10,
        _ => panic!("invalid SHA-256 hex"),
    }
}

const fn hex32(hex: &[u8]) -> [u8; 32] {
    assert!(hex.len() == 64);
    let mut out = [0u8; 32];
    let mut index = 0;
    while index < out.len() {
        out[index] = (hex_nibble(hex[index * 2]) << 4) | hex_nibble(hex[index * 2 + 1]);
        index += 1;
    }
    out
}

fn sha256_file(path: &Path) -> Result<[u8; 32]> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read FaceParsingV2 model {}", path.display()))?;
    Ok(sha256_32(&bytes))
}

fn verify_model_identity(bytes: u64, sha256: [u8; 32]) -> Result<()> {
    ensure!(
        bytes == FACE_PARSING_V2_MODEL_BYTES,
        "FaceParsingV2 model size mismatch: expected {}, got {bytes}",
        FACE_PARSING_V2_MODEL_BYTES
    );
    if sha256 == FACE_PARSING_V2_STALE_MODEL_SHA256 {
        bail!(
            "FaceParsingV2 rejected stale bedf8a42... model: reversed Scale data and AveragePool ceil_mode are not canonical"
        );
    }
    ensure!(
        sha256 == FACE_PARSING_V2_MODEL_SHA256,
        "FaceParsingV2 requires canonical fbe3fe1b..., got {}",
        hex(&sha256)
    );
    Ok(())
}

/// Explicit source-to-crop framing. `bbox_xywh` is in source pixel coordinates;
/// `roll_radians` is clockwise-positive in the recovered parsing contract.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceParsingRoi {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub roll_radians: f32,
}

impl FaceParsingRoi {
    /// Recovered FaceParsingV2 framing: margin 0.4, long side to 384, no flip.
    pub fn source_to_crop(self) -> Result<[[f32; 3]; 2], Refusal> {
        if [self.x, self.y, self.width, self.height, self.roll_radians]
            .iter()
            .any(|value| !value.is_finite())
            || self.width <= 0.0
            || self.height <= 0.0
        {
            return Err(Refusal::ValueOutOfRange("FaceParsingV2 ROI"));
        }
        let size = FACE_PARSING_V2_SIDE as f32;
        let scale = (1.0 - 0.4) * size / self.width.max(self.height);
        let angle = -self.roll_radians;
        let cosine = angle.cos();
        let sine = angle.sin();
        let mut matrix = [
            [scale * cosine, -scale * sine, 0.0],
            [scale * sine, scale * cosine, 0.0],
        ];
        let center = [self.x + self.width * 0.5, self.y + self.height * 0.5];
        matrix[0][2] = size * 0.5 - (matrix[0][0] * center[0] + matrix[0][1] * center[1]);
        matrix[1][2] = size * 0.5 - (matrix[1][0] * center[0] + matrix[1][1] * center[1]);
        Ok(matrix)
    }
}

/// Quantized model-frame parsing plus the exact crop transform that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceParsingProducerInputs {
    pub job_id: u64,
    pub width: u32,
    pub height: u32,
    pub channels: u8,
    pub model_sha256: [u8; 32],
    pub source_to_crop: [[f32; 3]; 2],
    /// Class-major `17 x 384 x 384`, quantized with `trunc(probability * 255)`.
    pub masks: Vec<u8>,
    /// Backend that actually quantized `masks`: `cuda` or `cpu`.
    pub mask_provider: &'static str,
}

impl FaceParsingProducerInputs {
    pub fn class(&self, class: usize) -> Option<&[u8]> {
        if class >= usize::from(self.channels)
            || self.masks.len()
                != usize::from(self.channels)
                    .saturating_mul(self.width as usize)
                    .saturating_mul(self.height as usize)
        {
            return None;
        }
        let plane = self.width as usize * self.height as usize;
        self.masks.get(class * plane..(class + 1) * plane)
    }

    /// Expands the canonical 17 FaceParsingV2 planes into the recovered
    /// 21-slot consumer layout. Slots 17..=20 are saturating bundles of
    /// `(0,13,14,15,16)`, `(2,7,11,12)`, `(4,5)`, and `(9,10)`.
    pub fn consumer_slots(&self) -> Result<Vec<Vec<u8>>, Refusal> {
        if self.job_id == 0
            || self.width != FACE_PARSING_V2_SIDE
            || self.height != FACE_PARSING_V2_SIDE
            || self.channels as usize != FACE_PARSING_V2_CLASSES
            || self.model_sha256 != FACE_PARSING_V2_MODEL_SHA256
        {
            return Err(Refusal::ValueOutOfRange(
                "FaceParsingV2 canonical 17-class input",
            ));
        }
        let plane = self.width as usize * self.height as usize;
        if self.masks.len() != FACE_PARSING_V2_CLASSES * plane {
            return Err(Refusal::ShapeMismatch("FaceParsingV2 17-class masks"));
        }

        let mut slots = Vec::with_capacity(21);
        for class in 0..FACE_PARSING_V2_CLASSES {
            slots.push(self.masks[class * plane..(class + 1) * plane].to_vec());
        }
        for members in [
            &[0usize, 13, 14, 15, 16][..],
            &[2, 7, 11, 12][..],
            &[4, 5][..],
            &[9, 10][..],
        ] {
            let mut bundle = vec![0u8; plane];
            for pixel in 0..plane {
                let sum = members
                    .iter()
                    .map(|class| u16::from(self.masks[class * plane + pixel]))
                    .sum::<u16>();
                bundle[pixel] = sum.min(255) as u8;
            }
            slots.push(bundle);
        }
        Ok(slots)
    }

    /// Packs recovered consumer slots 17..=20 into RGBA in R,G,B,A order.
    /// This only transforms canonical fbe output; it does not infer a
    /// row-specific semantic that was not recovered.
    pub fn consumer_rgba(&self) -> Result<Image, Refusal> {
        let slots = self.consumer_slots()?;
        let pixels = self.width as usize * self.height as usize;
        let mut rgba = Vec::with_capacity(pixels.saturating_mul(4));
        for pixel in 0..pixels {
            for slot in &slots[17..21] {
                rgba.push(slot[pixel]);
            }
        }
        Image::new(self.width, self.height, rgba)
            .ok_or(Refusal::ShapeMismatch("FaceParsingV2 consumer RGBA"))
    }
}

/// Projects canonical fbe class 1 (skin) from the 384px model crop back to the
/// exact source frame. The canonical model digest and all 17 planes are required.
pub fn project_face_parsing_skin_mask(
    parsing: &FaceParsingProducerInputs,
    source_width: u32,
    source_height: u32,
) -> Result<Vec<u8>, Refusal> {
    if source_width == 0 || source_height == 0 {
        return Err(Refusal::MissingInput("FaceParsingV2 skin source frame"));
    }
    if parsing.channels as usize != FACE_PARSING_V2_CLASSES
        || parsing.model_sha256 != FACE_PARSING_V2_MODEL_SHA256
    {
        return Err(Refusal::ValueOutOfRange(
            "FaceParsingV2 canonical 17-class fbe model",
        ));
    }
    let plane = (parsing.width as usize)
        .checked_mul(parsing.height as usize)
        .ok_or(Refusal::ShapeMismatch("FaceParsingV2 skin plane"))?;
    if parsing.width == 0
        || parsing.height == 0
        || parsing.masks.len() != FACE_PARSING_V2_CLASSES.saturating_mul(plane)
    {
        return Err(Refusal::ShapeMismatch("FaceParsingV2 17-class planes"));
    }
    let skin = parsing
        .class(FACE_PARSING_V2_SKIN_CLASS)
        .ok_or(Refusal::MissingInput("FaceParsingV2 class1 skin mask"))?;
    let crop_to_source = inverse_native(parsing.source_to_crop)
        .map_err(|_| Refusal::ShapeMismatch("FaceParsingV2 skin crop matrix"))?;
    project_teeth_mask(
        skin,
        parsing.width,
        parsing.height,
        crop_to_source,
        source_width,
        source_height,
    )
    .map_err(|_| Refusal::ShapeMismatch("FaceParsingV2 skin mask projection"))
}

pub struct FaceParsingInference {
    pub parsing: FaceParsingProducerInputs,
    /// FaceParsingV2 `id2.G` at the 384x384 model crop resolution.
    pub teeth_crop: Vec<u8>,
    /// Exact NCHW float32 tensor passed to ONNX Runtime.
    pub input: Vec<f32>,
    pub provider: &'static str,
    /// Backend that quantized the model probabilities into `parsing.masks`.
    pub mask_provider: &'static str,
}

/// Loaded ONNX session plus the two TeethWhitenV3 LUTs and the 256-byte id2 lookup.
pub struct FaceParsingProducer {
    model_path: PathBuf,
    session: Session,
    model_sha256: [u8; 32],
    teeth_lut: [u8; 256],
    whiten_lut: Vec<u8>,
    bright_lut: Vec<u8>,
    #[cfg(windows)]
    mask_gpu: Option<ModelPreprocess>,
}

impl FaceParsingProducer {
    pub fn load(model_path: impl AsRef<Path>) -> Result<Self> {
        let model_path = model_path.as_ref().to_path_buf();
        ensure!(
            model_path.is_file(),
            "FaceParsingV2 model is missing: {}",
            model_path.display()
        );
        let bytes = std::fs::metadata(&model_path)
            .with_context(|| format!("metadata {}", model_path.display()))?
            .len();
        let model_sha256 = sha256_file(&model_path)?;
        verify_model_identity(bytes, model_sha256)?;
        let teeth_lut = decode_hex_256(TEETH_LUT_HEX)?;
        ensure!(
            sha256_32(&teeth_lut) == FACE_PARSING_V2_TEETH_LUT_SHA256,
            "FaceParsingV2 teeth lookup identity mismatch"
        );
        let whiten_lut = decode_lut(WHITEN_PNG, &WHITEN_LUT_IDENTITY)?;
        let bright_lut = decode_lut(BRIGHT_PNG, &BRIGHT_LUT_IDENTITY)?;

        let runtime = Runtime::get().context("ONNX Runtime is not available")?;
        let session = runtime
            .open_session(&model_path)
            .with_context(|| format!("open FaceParsingV2 {}", model_path.display()))?;
        ensure!(
            session.input.dims == [1, 3, 384, 384] && session.input.element_type == 1,
            "FaceParsingV2 expects float32 [1,3,384,384], got {:?} type {}",
            session.input.dims,
            session.input.element_type
        );
        ensure!(
            session.outputs.len() == 1
                && session.outputs[0].dims == [1, 17, 384, 384]
                && session.outputs[0].element_type == 1,
            "FaceParsingV2 expects one float32 [1,17,384,384] output"
        );
        #[cfg(windows)]
        let mask_gpu = match std::env::var(FACE_PARSING_V2_MASK_BACKEND_ENV)
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "cpu" => None,
            "cuda" => Some(
                ModelPreprocess::open()
                    .map_err(|error| {
                        anyhow::anyhow!(
                            "FaceParsingV2 CUDA mask quantization was required: {error:?}"
                        )
                    })?,
            ),
            _ => match ModelPreprocess::open() {
                Ok(gpu) => Some(gpu),
                Err(error) => {
                    log::warn!("FaceParsingV2 CUDA mask quantization unavailable: {error:?}");
                    None
                }
            },
        };
        Ok(Self {
            model_path,
            session,
            model_sha256,
            teeth_lut,
            whiten_lut,
            bright_lut,
            #[cfg(windows)]
            mask_gpu,
        })
    }

    pub fn from_env() -> Result<Self> {
        let direct = std::env::var_os(FACE_PARSING_V2_MODEL_ENV).map(PathBuf::from);
        let path = direct
            .or_else(|| {
                [FACE_PARSING_V2_MODEL_ROOT_ENV, FACE_PARSING_V2_AIMODELS_ENV]
                    .iter()
                    .filter_map(std::env::var_os)
                    .map(PathBuf::from)
                    .flat_map(|root| {
                        [
                            root.join(FACE_PARSING_V2_MODEL_FILE),
                            root.join("models").join(FACE_PARSING_V2_MODEL_FILE),
                            root.join("assets")
                                .join("models")
                                .join(FACE_PARSING_V2_MODEL_FILE),
                            root.join("2102828093.onnx"),
                            root.join("models").join("2102828093.onnx"),
                            root.join("assets")
                                .join("models")
                                .join("2102828093.onnx"),
                        ]
                    })
                    .find(|candidate| candidate.is_file())
            })
            .with_context(|| {
                format!(
                    "FaceParsingV2 model is not configured; set {FACE_PARSING_V2_MODEL_ENV} or {FACE_PARSING_V2_MODEL_ROOT_ENV}"
                )
            })?;
        Self::load(path)
    }

    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    pub fn model_sha256(&self) -> [u8; 32] {
        self.model_sha256
    }

    pub fn infer(
        &self,
        frame: &Image,
        job_id: u64,
        source_to_crop: [[f32; 3]; 2],
    ) -> Result<FaceParsingInference> {
        frame
            .validate()
            .map_err(|refusal| anyhow::anyhow!("FaceParsingV2 frame: {refusal:?}"))?;
        ensure!(job_id != 0, "FaceParsingV2 job ticket must be nonzero");
        validate_matrix(source_to_crop)?;
        let input = prepare_input(frame, source_to_crop)?;
        let runtime = Runtime::get().context("ONNX Runtime is not available")?;
        let mut input = input;
        let mut outputs = runtime.run(
            &self.session,
            &mut input,
            &[
                1,
                3,
                FACE_PARSING_V2_SIDE as i64,
                FACE_PARSING_V2_SIDE as i64,
            ],
        )?;
        ensure!(
            outputs.len() == 1,
            "FaceParsingV2 returned {} outputs",
            outputs.len()
        );
        let probability = outputs.swap_remove(0);
        let plane = FACE_PARSING_V2_SIDE as usize * FACE_PARSING_V2_SIDE as usize;
        ensure!(
            probability.len() == FACE_PARSING_V2_CLASSES * plane,
            "FaceParsingV2 returned {} values, expected {}",
            probability.len(),
            FACE_PARSING_V2_CLASSES * plane
        );
        if let Some(index) = probability
            .iter()
            .position(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            bail!(
                "FaceParsingV2 output is not probability data at index {index}: {}",
                probability[index]
            );
        }
        let (masks, mask_provider) = self.quantize_masks(&probability)?;

        // Recovered consumer order: 17 raw slots followed by class groups
        // (0,13,14,15,16), (2,7,11,12), (4,5), (9,10). id2.G is slot 20.
        let groups: [(usize, &[usize]); 4] = [
            (0, &[13, 14, 15, 16][..]),
            (2, &[7, 11, 12][..]),
            (4, &[5][..]),
            (9, &[10][..]),
        ];
        let mut group_20 = vec![0u8; plane];
        let group = groups[3];
        for pixel in 0..plane {
            let mut sum = u16::from(masks[group.0 * plane + pixel]);
            for class in group.1 {
                sum = sum.saturating_add(u16::from(masks[class * plane + pixel]));
            }
            group_20[pixel] = sum.min(255) as u8;
        }
        let teeth_crop = group_20
            .into_iter()
            .map(|value| self.teeth_lut[value as usize])
            .collect();

        Ok(FaceParsingInference {
            parsing: FaceParsingProducerInputs {
                job_id,
                width: FACE_PARSING_V2_SIDE,
                height: FACE_PARSING_V2_SIDE,
                channels: FACE_PARSING_V2_CLASSES as u8,
                model_sha256: self.model_sha256,
                source_to_crop,
                masks,
                mask_provider,
            },
            teeth_crop,
            input,
            provider: self.session.provider().label(),
            mask_provider,
        })
    }

    fn quantize_masks(&self, probability: &[f32]) -> Result<(Vec<u8>, &'static str)> {
        #[cfg(windows)]
        if let Some(gpu) = self.mask_gpu.as_ref() {
            match gpu.f32_to_u8(probability, &PROBABILITY_QUANT) {
                Ok((masks, _)) => return Ok((masks, "cuda")),
                Err(error) => {
                    log::warn!("FaceParsingV2 CUDA mask quantization failed; using CPU: {error:?}");
                }
            }
        }
        Ok((quantize_mask_cpu(probability)?, "cpu"))
    }

    fn pf90004_bundle(
        &self,
        frame: &Image,
        parsing: &FaceParsingProducerInputs,
        crop_to_source: [[f32; 3]; 2],
    ) -> Result<Pf90004ProducerInputs> {
        ensure!(
            parsing.model_sha256 == FACE_PARSING_V2_MODEL_SHA256,
            "PF90004 requires the canonical FaceParsingV2 model"
        );
        ensure!(
            parsing.width == FACE_PARSING_V2_SIDE
                && parsing.height == FACE_PARSING_V2_SIDE
                && usize::from(parsing.channels) == FACE_PARSING_V2_CLASSES,
            "PF90004 parser grid/classes do not match FaceParsingV2"
        );
        let class_planes = pack_class_groups(
            &parsing.masks,
            FACE_PARSING_V2_SIDE as usize,
            FACE_PARSING_V2_SIDE as usize,
        )?;
        let projected: Vec<Vec<u8>> = class_planes
            .iter()
            .map(|plane| {
                project_teeth_mask(
                    plane,
                    FACE_PARSING_V2_SIDE,
                    FACE_PARSING_V2_SIDE,
                    crop_to_source,
                    frame.width,
                    frame.height,
                )
            })
            .collect::<Result<_>>()?;
        let face = project_class_plane(
            &parsing.masks,
            PF90004_FACE_CLASSES,
            FACE_PARSING_V2_SIDE as usize,
            FACE_PARSING_V2_SIDE as usize,
            crop_to_source,
            frame.width,
            frame.height,
        )?;
        let pixels = usize::try_from(frame.width)
            .ok()
            .and_then(|width| {
                usize::try_from(frame.height)
                    .ok()
                    .and_then(|height| width.checked_mul(height))
            })
            .ok_or_else(|| anyhow::anyhow!("PF90004 frame size overflow"))?;
        let mut parsing_rgba = Vec::with_capacity(pixels.saturating_mul(4));
        let mut face_rgba = Vec::with_capacity(pixels.saturating_mul(4));
        for pixel in 0..pixels {
            parsing_rgba.extend_from_slice(&[
                projected[0][pixel],
                projected[1][pixel],
                projected[2][pixel],
                projected[3][pixel],
            ]);
            let value = face[pixel];
            face_rgba.extend_from_slice(&[value, value, projected[2][pixel], value]);
        }
        let value_mean = pf90004_value_mean(frame, &face)?;
        Pf90004ProducerInputs::from_same_job(
            frame.width,
            frame.height,
            face,
            parsing_rgba,
            face_rgba,
            value_mean,
            NormRect::FULL,
        )
        .map_err(|refusal| anyhow::anyhow!("PF90004 producer refused: {refusal:?}"))
    }

    /// Produces `parsing`, `teeth`, and the same-frame PF90004 bundle.
    /// A zero-support teeth plane stays explicit; the caller must not treat it
    /// as a positive PF108 selection.
    pub fn produce(
        &self,
        frame: &Image,
        job_id: u64,
        source_to_crop: [[f32; 3]; 2],
        bright_ui: f32,
        deyellow_ui: f32,
    ) -> Result<ProducerInputs> {
        ensure!(
            bright_ui.is_finite()
                && deyellow_ui.is_finite()
                && (0.0..=100.0).contains(&bright_ui)
                && (0.0..=100.0).contains(&deyellow_ui),
            "PF108 UI values must be finite and in 0..=100"
        );
        let inference = self.infer(frame, job_id, source_to_crop)?;
        let crop_to_source = inverse_native(source_to_crop)?;
        let mask_g = project_teeth_mask(
            &inference.teeth_crop,
            FACE_PARSING_V2_SIDE,
            FACE_PARSING_V2_SIDE,
            crop_to_source,
            frame.width,
            frame.height,
        )?;
        let mask_sha256 = sha256_32(&mask_g);
        let pf90004 = self.pf90004_bundle(frame, &inference.parsing, crop_to_source)?;
        Ok(ProducerInputs {
            parsing: Some(inference.parsing),
            teeth: Some(TeethProducerInputs {
                job_id,
                mask_width: frame.width,
                mask_height: frame.height,
                mask_source: TeethMaskSource::FaceParsingV2Id2G,
                mask_format: TeethMaskFormat::U8,
                mask_sha256,
                whiten_lut_identity: PF108_WHITEN_LUT_IDENTITY,
                bright_lut_identity: PF108_BRIGHT_LUT_IDENTITY,
                mask_g,
                bright_ui,
                deyellow_ui,
                whiten_lut: self.whiten_lut.clone(),
                bright_lut: self.bright_lut.clone(),
            }),
            pf90004: Some(pf90004),
            ..ProducerInputs::default()
        })
    }

    /// Candidate application entry. The automatic native chain is
    /// `1882168271 -> 1063025060 -> 323773030 -> source-frame ROI -> this fbe
    /// model`; the caller supplies the same-job ROI and no detected face is a
    /// typed refusal, never a full-frame fallback.
    pub fn attach_to_adjustment(
        &self,
        adjustments: &mut crate::AdjustmentSet,
        frame: &Image,
        job_id: u64,
        detected_face_roi: Option<FaceParsingRoi>,
        bright_ui: f32,
        deyellow_ui: f32,
    ) -> Result<Arc<ProducerInputs>> {
        let roi = required_face_roi(detected_face_roi)?;
        let source_to_crop = roi
            .source_to_crop()
            .map_err(|refusal| anyhow::anyhow!("FaceParsingV2 ROI refused: {refusal:?}"))?;
        let bundle = Arc::new(self.produce(
            frame,
            job_id,
            source_to_crop,
            bright_ui,
            deyellow_ui,
        )?);
        adjustments.portrait_producer_inputs = Some(Arc::clone(&bundle));
        Ok(bundle)
    }
}

fn quantize_mask_cpu(probability: &[f32]) -> Result<Vec<u8>> {
    f32_to_u8_cpu(probability, &PROBABILITY_QUANT)
        .map_err(|refusal| anyhow::anyhow!("FaceParsingV2 mask quantization: {refusal:?}"))
}

fn pack_class_groups(
    masks: &[u8],
    width: usize,
    height: usize,
) -> Result<[Vec<u8>; 4]> {
    let plane = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("FaceParsingV2 plane overflow"))?;
    ensure!(
        masks.len()
            == plane
                .checked_mul(FACE_PARSING_V2_CLASSES)
                .ok_or_else(|| anyhow::anyhow!("FaceParsingV2 mask overflow"))?,
        "FaceParsingV2 class-major plane shape mismatch"
    );
    let mut groups = [
        vec![0u8; plane],
        vec![0u8; plane],
        vec![0u8; plane],
        vec![0u8; plane],
    ];
    for (output, classes) in groups.iter_mut().zip(PF90004_PARSING_RGBA_CLASSES) {
        for (pixel, value) in output.iter_mut().enumerate() {
            let mut sum = 0u16;
            for &class in classes {
                sum = sum.saturating_add(u16::from(masks[class * plane + pixel]));
            }
            *value = sum.min(255) as u8;
        }
    }
    Ok(groups)
}

fn project_class_plane(
    masks: &[u8],
    classes: &[usize],
    width: usize,
    height: usize,
    crop_to_source: [[f32; 3]; 2],
    frame_width: u32,
    frame_height: u32,
) -> Result<Vec<u8>> {
    let plane = width
        .checked_mul(height)
        .ok_or_else(|| anyhow::anyhow!("FaceParsingV2 plane overflow"))?;
    ensure!(
        masks.len() >= plane * FACE_PARSING_V2_CLASSES,
        "FaceParsingV2 class-major plane shape mismatch"
    );
    let mut packed = vec![0u8; plane];
    for (pixel, value) in packed.iter_mut().enumerate() {
        let mut sum = 0u16;
        for &class in classes {
            sum = sum.saturating_add(u16::from(masks[class * plane + pixel]));
        }
        *value = sum.min(255) as u8;
    }
    project_teeth_mask(
        &packed,
        width as u32,
        height as u32,
        crop_to_source,
        frame_width,
        frame_height,
    )
}

fn pf90004_value_mean(frame: &Image, skin_mask: &[u8]) -> Result<f32> {
    ensure!(
        skin_mask.len() == frame.pixel_count(),
        "PF90004 skin plane shape mismatch"
    );
    let mut sum = 0u64;
    let mut count = 0u64;
    for (pixel, &skin) in skin_mask.iter().enumerate() {
        if skin == 0 {
            continue;
        }
        let offset = pixel * 4;
        let value = frame.pixels[offset]
            .max(frame.pixels[offset + 1])
            .max(frame.pixels[offset + 2]);
        sum += u64::from(value);
        count += 1;
    }
    ensure!(count != 0, "PF90004_PARSER_SKIN_EMPTY");
    Ok(sum as f32 / count as f32)
}

fn required_face_roi(roi: Option<FaceParsingRoi>) -> Result<FaceParsingRoi> {
    roi.ok_or_else(|| anyhow::anyhow!("FaceParsingV2_NO_FACE_ROI"))
}

fn decode_lut(bytes: &[u8], identity: &LutIdentity) -> Result<Vec<u8>> {
    ensure!(
        bytes.len() == identity.png_bytes && sha256_32(bytes) == identity.png_sha256,
        "PF108 {} PNG identity mismatch",
        identity.source_file
    );
    let image = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)
        .context("decode PF108 LUT PNG")?
        .to_rgb8();
    ensure!(
        image.dimensions() == (identity.width, identity.height),
        "PF108 {} dimensions mismatch",
        identity.source_file
    );
    let rgb = image.into_raw();
    ensure!(
        rgb.len() == identity.rgb8_bytes && sha256_32(&rgb) == identity.rgb8_sha256,
        "PF108 {} RGB8 identity mismatch",
        identity.source_file
    );
    Ok(rgb)
}

fn decode_hex_256(hex: &[u8]) -> Result<[u8; 256]> {
    ensure!(hex.len() == 512, "FaceParsingV2 teeth lookup hex length");
    let mut out = [0u8; 256];
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = (hex_nibble(hex[index * 2]) << 4) | hex_nibble(hex[index * 2 + 1]);
    }
    Ok(out)
}

fn validate_matrix(matrix: [[f32; 3]; 2]) -> Result<()> {
    ensure!(
        matrix.iter().flatten().all(|value| value.is_finite()),
        "FaceParsingV2 source-to-crop matrix is not finite"
    );
    let determinant = matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1];
    ensure!(
        determinant.is_finite() && determinant.abs() > 1e-12,
        "FaceParsingV2 source-to-crop matrix is singular"
    );
    Ok(())
}

fn prepare_input(frame: &Image, source_to_crop: [[f32; 3]; 2]) -> Result<Vec<f32>> {
    let inverse = inverse_native(source_to_crop)?;
    let out_side = FACE_PARSING_V2_SIDE as usize;
    let plane = out_side * out_side;
    let mut tensor = vec![0.0f32; 3 * plane];
    let width = frame.width as usize;
    let height = frame.height as usize;
    ensure!(
        width >= 2 && height >= 2,
        "FaceParsingV2 source is too small"
    );
    for y in 0..out_side {
        let mut x_coord = f64::from(inverse[0][1]) * y as f64 + f64::from(inverse[0][2]);
        let mut y_coord = f64::from(inverse[1][1]) * y as f64 + f64::from(inverse[1][2]);
        for x in 0..out_side {
            let ix = x_coord.floor() as i64;
            let iy = y_coord.floor() as i64;
            if ix >= 0 && iy >= 0 && (ix as usize) < width - 1 && (iy as usize) < height - 1 {
                let fx = (x_coord - ix as f64) as f32;
                let fy = (y_coord - iy as f64) as f32;
                let ox = 1.0f32 - fx;
                let oy = 1.0f32 - fy;
                let w00 = ox * oy;
                let w10 = fx * oy;
                let w01 = ox * fy;
                let w11 = fx * fy;
                let base = iy as usize * width + ix as usize;
                for channel in 0..3 {
                    let at = |pixel: usize| frame.pixels[pixel * 4 + channel] as f32;
                    let mut value = at(base) * w00;
                    value = value + at(base + 1) * w10;
                    value = value + at(base + width) * w01;
                    value = value + at(base + width + 1) * w11;
                    tensor[channel * plane + y * out_side + x] =
                        (value - MODEL_MEAN[channel]) * MODEL_SCALE[channel];
                }
            }
            x_coord += f64::from(inverse[0][0]);
            y_coord += f64::from(inverse[1][0]);
        }
    }
    Ok(tensor)
}

fn inverse_native(matrix: [[f32; 3]; 2]) -> Result<[[f32; 3]; 2]> {
    validate_matrix(matrix)?;
    let [a, b, c] = matrix[0];
    let [d, e, f] = matrix[1];
    let determinant = a * e - d * b;
    let reciprocal = 1.0f32 / determinant;
    let aa = e * reciprocal;
    let bb = -b * reciprocal;
    let dd = -d * reciprocal;
    let ee = a * reciprocal;
    let cc = -aa * c - bb * f;
    let ff = -dd * c - ee * f;
    Ok([[aa, bb, cc], [dd, ee, ff]])
}

fn project_teeth_mask(
    crop: &[u8],
    crop_width: u32,
    crop_height: u32,
    crop_to_source: [[f32; 3]; 2],
    source_width: u32,
    source_height: u32,
) -> Result<Vec<u8>> {
    ensure!(
        crop.len() == crop_width as usize * crop_height as usize,
        "FaceParsingV2 teeth crop shape mismatch"
    );
    let (matrix, scissor) = normalized_projection_and_scissor(
        crop_to_source,
        source_width,
        source_height,
        crop_width,
        crop_height,
    )?;
    let mut output = vec![0u8; source_width as usize * source_height as usize];
    let x0 = scissor[0].max(0);
    let y0 = scissor[1].max(0);
    let x1 = (scissor[0] + scissor[2]).min(source_width as i32);
    let y1 = (scissor[1] + scissor[3]).min(source_height as i32);
    if x1 <= x0 || y1 <= y0 {
        return Ok(output);
    }
    for y in y0..y1 {
        let gy = (y as f32 + 0.5) / source_height as f32;
        for x in x0..x1 {
            let gx = (x as f32 + 0.5) / source_width as f32;
            let d0 = matrix[0][0] * gx + matrix[0][1] * gy + matrix[0][2];
            let d1 = matrix[1][0] * gx + matrix[1][1] * gy + matrix[1][2];
            let d2 = matrix[2][0] * gx + matrix[2][1] * gy + matrix[2][2];
            let sample = if d2 == 0.0 {
                0.0
            } else {
                sample_gl_linear(
                    crop,
                    crop_width as usize,
                    crop_height as usize,
                    d0 / d2,
                    d1 / d2,
                )
            };
            output[y as usize * source_width as usize + x as usize] =
                (sample + 0.5).floor().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(output)
}

fn normalized_projection_and_scissor(
    crop_to_source: [[f32; 3]; 2],
    source_width: u32,
    source_height: u32,
    crop_width: u32,
    crop_height: u32,
) -> Result<([[f32; 3]; 3], [i32; 4])> {
    ensure!(
        source_width > 0 && source_height > 0 && crop_width > 0 && crop_height > 0,
        "FaceParsingV2 projection dimensions must be positive"
    );
    let [a, b, c] = crop_to_source[0];
    let [d, e, f] = crop_to_source[1];
    let normalized = [
        [
            f64::from(a) / f64::from(source_width),
            f64::from(b) / f64::from(source_width),
            f64::from(c) / f64::from(source_width),
        ],
        [
            f64::from(d) / f64::from(source_height),
            f64::from(e) / f64::from(source_height),
            f64::from(f) / f64::from(source_height),
        ],
        [0.0, 0.0, 1.0],
    ];
    let determinant = normalized[0][0] * normalized[1][1] - normalized[0][1] * normalized[1][0];
    ensure!(
        determinant.is_finite() && determinant.abs() > 1e-20,
        "FaceParsingV2 normalized projection is singular"
    );
    let inverse = [
        [
            (normalized[1][1] / determinant) as f32 / crop_width as f32,
            (-normalized[0][1] / determinant) as f32 / crop_width as f32,
            ((normalized[0][1] * normalized[1][2] - normalized[0][2] * normalized[1][1])
                / determinant) as f32
                / crop_width as f32,
        ],
        [
            (-normalized[1][0] / determinant) as f32 / crop_height as f32,
            (normalized[0][0] / determinant) as f32 / crop_height as f32,
            ((normalized[0][2] * normalized[1][0] - normalized[0][0] * normalized[1][2])
                / determinant) as f32
                / crop_height as f32,
        ],
        [0.0, 0.0, 1.0],
    ];

    let points = [
        [0.0f32, 0.0f32],
        [1.0, crop_height as f32],
        [crop_width as f32, 1.0],
        [crop_width as f32, crop_height as f32],
    ];
    let mut min = [i32::MAX, i32::MAX];
    let mut max = [i32::MIN, i32::MIN];
    for [x, y] in points {
        let px =
            ((normalized[0][0] as f32 * x + normalized[0][1] as f32 * y + normalized[0][2] as f32)
                * source_width as f32)
                .trunc() as i32;
        let py =
            ((normalized[1][0] as f32 * x + normalized[1][1] as f32 * y + normalized[1][2] as f32)
                * source_height as f32)
                .trunc() as i32;
        min[0] = min[0].min(px);
        min[1] = min[1].min(py);
        max[0] = max[0].max(px);
        max[1] = max[1].max(py);
    }
    Ok((inverse, [min[0], min[1], max[0] - min[0], max[1] - min[1]]))
}

fn sample_gl_linear(texture: &[u8], width: usize, height: usize, u: f32, v: f32) -> f32 {
    if !u.is_finite() || !v.is_finite() || !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
        return 0.0;
    }
    let x = (u * width as f32 - 0.5).clamp(0.0, width.saturating_sub(1) as f32);
    let y = (v * height as f32 - 0.5).clamp(0.0, height.saturating_sub(1) as f32);
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let top = texture[y0 * width + x0] as f32 * (1.0 - fx) + texture[y0 * width + x1] as f32 * fx;
    let bottom =
        texture[y1 * width + x0] as f32 * (1.0 - fx) + texture[y1 * width + x1] as f32 * fx;
    top * (1.0 - fy) + bottom * fy
}

fn sha256_f32le(values: &[f32]) -> [u8; 32] {
    let mut bytes = Vec::with_capacity(values.len() * std::mem::size_of::<f32>());
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    sha256_32(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovered_fixture_matrix_is_reproduced() {
        let matrix = FaceParsingRoi {
            x: 210.0,
            y: 200.0,
            width: 150.0,
            height: 195.0,
            roll_radians: 0.0,
        }
        .source_to_crop()
        .unwrap();
        let expected = [
            [1.181_538_5, 0.0, -144.738_46],
            [-0.0, 1.181_538_5, -159.507_69],
        ];
        for row in 0..2 {
            for column in 0..3 {
                assert!((matrix[row][column] - expected[row][column]).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn identity_projection_preserves_the_mask() {
        let mut crop = vec![0u8; 8 * 8];
        crop[3 * 8 + 4] = 173;
        let projected =
            project_teeth_mask(&crop, 8, 8, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 8, 8).unwrap();
        assert_eq!(projected, crop);
    }

    #[test]
    fn teeth_group_uses_recovered_slot_20() {
        let plane = 384 * 384;
        let mut probability = vec![0.0f32; 17 * plane];
        probability[9 * plane] = 1.0;
        let masks = quantize_mask_cpu(&probability).unwrap();
        let mut sum = u16::from(masks[9 * plane]);
        sum = sum.saturating_add(u16::from(masks[10 * plane]));
        let lut = decode_hex_256(TEETH_LUT_HEX).unwrap();
        assert_eq!(lut[sum.min(255) as usize], lut[255]);
        probability[9 * plane] = 0.0;
        probability[0 * plane] = 1.0;
        let masks = quantize_mask_cpu(&probability).unwrap();
        assert_eq!(masks[9 * plane], 0);
        assert_eq!(masks[10 * plane], 0);
    }

    #[test]
    fn probability_quantization_preserves_pf108_trunc_bytes() {
        let expected_len =
            FACE_PARSING_V2_CLASSES * FACE_PARSING_V2_SIDE as usize * FACE_PARSING_V2_SIDE as usize;
        let mut probability = Vec::with_capacity(expected_len);
        for value in 0..=255u32 {
            let base = value as f32 / 255.0;
            probability.push(base);
            if value < 255 {
                probability.push(f32::from_bits(base.to_bits() + 1));
            }
            if base.to_bits() > 0 {
                probability.push(f32::from_bits(base.to_bits() - 1));
            }
        }
        let mut state = 0x9E37_79B9u32;
        while probability.len() < expected_len {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            probability.push((state >> 8) as f32 / 16_777_215.0);
        }
        probability.truncate(expected_len);
        let expected: Vec<u8> = probability
            .iter()
            .map(|value| (value * 255.0).trunc() as u8)
            .collect();
        assert_eq!(quantize_mask_cpu(&probability).unwrap(), expected);
    }

    #[test]
    fn canonical_fbe_class1_projects_as_pf30013_skin_mask() {
        let side = FACE_PARSING_V2_SIDE as usize;
        let plane = side * side;
        let mut masks = vec![9u8; FACE_PARSING_V2_CLASSES * plane];
        masks[FACE_PARSING_V2_SKIN_CLASS * plane..(FACE_PARSING_V2_SKIN_CLASS + 1) * plane]
            .fill(200);
        let parsing = FaceParsingProducerInputs {
            job_id: 17,
            width: FACE_PARSING_V2_SIDE,
            height: FACE_PARSING_V2_SIDE,
            channels: FACE_PARSING_V2_CLASSES as u8,
            model_sha256: FACE_PARSING_V2_MODEL_SHA256,
            source_to_crop: FaceParsingRoi {
                x: 0.0,
                y: 0.0,
                width: 2.0,
                height: 1.0,
                roll_radians: 0.0,
            }
            .source_to_crop()
            .unwrap(),
            masks,
            mask_provider: "cpu",
        };
        let skin = project_face_parsing_skin_mask(&parsing, 2, 1).unwrap();
        assert_eq!(skin, [200, 200]);

        let mut stale = parsing;
        stale.model_sha256 = FACE_PARSING_V2_STALE_MODEL_SHA256;
        assert_eq!(
            project_face_parsing_skin_mask(&stale, 2, 1),
            Err(Refusal::ValueOutOfRange(
                "FaceParsingV2 canonical 17-class fbe model"
            ))
        );
    }

    #[test]
    fn no_face_roi_is_rejected_before_inference() {
        let error = required_face_roi(None).unwrap_err().to_string();
        assert_eq!(error, "FaceParsingV2_NO_FACE_ROI");
        assert_ne!(
            FACE_PARSING_V2_MODEL_SHA256,
            FACE_PARSING_V2_STALE_MODEL_SHA256
        );
    }

    #[test]
    fn stale_model_identity_is_rejected_at_canonical_size() {
        verify_model_identity(
            FACE_PARSING_V2_MODEL_BYTES,
            FACE_PARSING_V2_MODEL_SHA256,
        )
        .expect("canonical model identity");
        let error = verify_model_identity(
            FACE_PARSING_V2_MODEL_BYTES,
            FACE_PARSING_V2_STALE_MODEL_SHA256,
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("stale bedf8a42..."), "{error}");
    }

    #[test]
    #[ignore = "needs AURORA_AI_PROVIDER=cpu, AURORA_FACE_PARSING_MASK_BACKEND=cpu, AURORA_FACE_PARSING_MODEL, AURORA_FACE_PARSING_IMAGE and AURORA_ORT_LIB"]
    fn real_face_parsing_bundle_executes_pf108_on_cpu() {
        run_real_pf108_backend(true, "CPU");
    }

    #[test]
    #[ignore = "needs AURORA_AI_PROVIDER=cuda, AURORA_FACE_PARSING_MASK_BACKEND=cuda, CUDA, AURORA_FACE_PARSING_MODEL, AURORA_FACE_PARSING_IMAGE and AURORA_ORT_LIB"]
    fn real_face_parsing_bundle_executes_pf108_on_cuda() {
        run_real_pf108_backend(false, "Cuda");
    }

    fn run_real_pf108_backend(force_cpu: bool, expected_backend: &str) {
        use std::collections::BTreeMap;

        use crate::portrait_stage::{
            PortraitBackend, PortraitJobInputs, PortraitMasks, apply_portrait_effects_with_job,
        };

        let model = std::env::var_os(FACE_PARSING_V2_MODEL_ENV)
            .map(PathBuf::from)
            .or_else(|| std::env::var_os(FACE_PARSING_V2_MODEL_ROOT_ENV).map(PathBuf::from))
            .expect("AURORA_FACE_PARSING_MODEL or AURORA_FACE_PARSING_MODEL_ROOT");
        let model = if model.is_dir() {
            model.join("2102828093.onnx")
        } else {
            model
        };
        let photo = std::env::var_os("AURORA_FACE_PARSING_IMAGE")
            .map(PathBuf::from)
            .expect("AURORA_FACE_PARSING_IMAGE");
        let decoded = image::open(photo).unwrap().to_rgba8();
        let frame = Image::new(decoded.width(), decoded.height(), decoded.into_raw()).unwrap();
        let roi = FaceParsingRoi {
            x: 210.0,
            y: 200.0,
            width: 150.0,
            height: 195.0,
            roll_radians: 0.0,
        };
        let matrix = roi.source_to_crop().unwrap();
        let producer = FaceParsingProducer::load(model).unwrap();
        assert_eq!(
            producer.model_sha256(),
            FACE_PARSING_V2_MODEL_SHA256,
            "model_sha256={}",
            hex(&producer.model_sha256())
        );
        let inference = producer.infer(&frame, 4502, matrix).unwrap();
        let expected_provider = if force_cpu { "cpu" } else { "cuda" };
        assert_eq!(
            inference.provider.to_ascii_lowercase(),
            expected_provider,
            "provider={} mask_provider={}",
            inference.provider,
            inference.mask_provider
        );
        assert_eq!(
            inference.mask_provider,
            expected_provider,
            "provider={} mask_provider={}",
            inference.provider,
            inference.mask_provider
        );
        if let Ok(expected) = std::env::var(FACE_PARSING_V2_EXPECT_PROVIDER_ENV) {
            assert_eq!(
                inference.provider.to_ascii_lowercase(),
                expected.trim().to_ascii_lowercase(),
                "provider={} mask_provider={}",
                inference.provider,
                inference.mask_provider
            );
        }
        assert_eq!(
            sha256_f32le(&inference.input),
            FACE_PARSING_V2_PREPARED_TENSOR_SHA256
        );
        assert_eq!(
            sha256_32(&inference.teeth_crop),
            FACE_PARSING_V2_FIXTURE_TEETH_CROP_SHA256
        );
        eprintln!(
            "PF108_MASK_EVIDENCE model_sha256={} provider={} mask_provider={} prepared_tensor_sha256={} live_masks_sha256={} cached_ort_masks_sha256={}",
            hex(&producer.model_sha256()),
            inference.provider,
            inference.mask_provider,
            hex(&sha256_f32le(&inference.input)),
            hex(&sha256_32(&inference.parsing.masks)),
            hex(&FACE_PARSING_V2_CACHED_ORT_MASKS_SHA256),
        );
        let producer_inputs = producer.produce(&frame, 4502, matrix, 60.0, 70.0).unwrap();
        let teeth = producer_inputs.teeth.as_ref().unwrap();
        assert_eq!(teeth.mask_g.len(), frame.pixel_count());
        assert!(teeth.mask_g.iter().any(|value| *value != 0));
        assert_eq!(
            sha256_32(&teeth.mask_g),
            FACE_PARSING_V2_FIXTURE_TEETH_FULL_SHA256
        );
        assert_eq!(
            teeth.mask_g.iter().filter(|value| **value != 0).count(),
            30
        );
        assert_eq!(teeth.job_id, 4502);
        assert_eq!((teeth.mask_width, teeth.mask_height), (512, 512));
        assert_eq!(teeth.mask_source, TeethMaskSource::FaceParsingV2Id2G);
        assert_eq!(teeth.mask_format, TeethMaskFormat::U8);
        assert_eq!(teeth.mask_sha256, sha256_32(&teeth.mask_g));
        assert_eq!(teeth.whiten_lut_identity, PF108_WHITEN_LUT_IDENTITY);
        assert_eq!(teeth.bright_lut_identity, PF108_BRIGHT_LUT_IDENTITY);
        assert_eq!(
            producer_inputs.parsing.as_ref().unwrap().masks.len(),
            17 * 384 * 384
        );

        let masks = PortraitMasks::default();
        let mut job = PortraitJobInputs::from_renderer_job(4502, &masks);
        job.fill_from_producer_inputs(&producer_inputs);
        let extra = BTreeMap::from([
            ("pc:108".to_string(), 0.8),
            ("pc:109".to_string(), 0.6),
            ("pc:110".to_string(), 0.7),
        ]);
        let mut rendered = frame.pixels.clone();
        let mut run = || {
            apply_portrait_effects_with_job(
                &mut rendered,
                frame.width,
                frame.height,
                crate::NormRect::FULL,
                &extra,
                &job,
            )
        };
        #[cfg(windows)]
        let report = if force_cpu {
            crate::portrait_stage::with_forced_cpu(run)
        } else {
            run()
        };
        #[cfg(not(windows))]
        let report = run();
        let row = report
            .effects
            .iter()
            .find(|row| row.pf == 108)
            .expect("PF108 report");
        assert!(row.executed, "{row:?}");
        assert_eq!(
            row.backend,
            if expected_backend == "CPU" {
                PortraitBackend::Cpu
            } else {
                PortraitBackend::Cuda
            },
            "{row:?}"
        );
        assert!(row.refusal.is_none());
        let mut changed_inside = 0usize;
        let mut changed_outside = 0usize;
        let mut changed_alpha = 0usize;
        for (index, (before, after)) in frame
            .pixels
            .chunks_exact(4)
            .zip(rendered.chunks_exact(4))
            .enumerate()
        {
            if before[3] != after[3] {
                changed_alpha += 1;
            }
            if before[..3] != after[..3] {
                if teeth.mask_g[index] != 0 {
                    changed_inside += 1;
                } else {
                    changed_outside += 1;
                }
            }
        }
        assert!(changed_inside > 0);
        assert_eq!(changed_outside, 0);
        assert_eq!(changed_alpha, 0);
        if let Some(path) = std::env::var_os("AURORA_FACE_PARSING_CAPTURE") {
            std::fs::write(PathBuf::from(path), &rendered).unwrap();
        }
        eprintln!(
            "PF108_{expected_backend}_PASS model_sha256={} provider={} mask_provider={} render_backend={:?} support={} changed_inside={changed_inside} changed_outside={changed_outside} changed_alpha={changed_alpha} output_sha256={}",
            hex(&producer.model_sha256()),
            inference.provider,
            inference.mask_provider,
            row.backend,
            teeth.mask_g.iter().filter(|value| **value != 0).count(),
            hex(&sha256_32(&rendered)),
        );
    }
}
