//! PF90130 model-output producer.
//!
//! The two graph outputs are accepted only by their proven tensor IDs. The
//! API has no output-slot parameter: job offsets `+0x1d8` / `+0x230` are not
//! accepted as a substitute for tensor identity, and no channel permutation is
//! synthesized when the bytes do not match the proven NHWC4 layout.

use super::{
    repair_skin_stage::{
        GlareChainFinalWritebackInputs, GlareFinalWritebackOutput, GlarePlanarBundle,
        GlarePlanarInputs, GlarePlanarReceipts, GlareWritebackInputs, GlareWritebackOutput,
        PF90130_A_PLANAR_BYTES, PF90130_B_PLANAR_BYTES, PF90130_PIXELS, PF90130_PLANE_HEIGHT,
        PF90130_PLANE_WIDTH, apply_glare_chain_final_writeback, apply_glare_stage_writeback,
        bridge_glare_planes,
    },
    sha256_32,
};
use aurora_portrait::{
    Image, Refusal,
    kernels::pf90130_chain::{self, PixelRect},
};

pub const PF90130_MODEL_ID: u32 = 1_463_593_431;
pub const PF90130_INPUT_TENSOR_ID: u32 = 0xB70F_CA85;
pub const PF90130_EFFECT_TENSOR_ID: u32 = 0xB175_C225;
pub const PF90130_MASK_TENSOR_ID: u32 = 0x6051_C50E;
/// The proven preprocess destination and input tensor are both 768x384.
pub const PF90130_NORMALIZE_SUB: f32 = 127.5;
pub const PF90130_NORMALIZE_SCALE: f32 = 0.007843_137_718_737_125;

const PF90130_NHWC4_BYTES: usize = PF90130_PIXELS * 16;

/// One immutable model/output receipt for the 768x384 aligned PF90130 frame.
///
/// The fields are private so a receipt cannot name an arbitrary slot or lane
/// order after construction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pf90130SameJobReceipt {
    ticket: u64,
    model_id: u32,
    input_tensor_id: u32,
    effect_tensor_id: u32,
    mask_tensor_id: u32,
    input_nhwc4_sha256: [u8; 32],
    effect_nhwc4_sha256: [u8; 32],
    mask_nhwc4_sha256: [u8; 32],
    material_planar_sha256: [u8; 32],
    mask_planar_sha256: [u8; 32],
}

impl Pf90130SameJobReceipt {
    pub fn ticket(&self) -> u64 {
        self.ticket
    }

    pub fn model_id(&self) -> u32 {
        self.model_id
    }

    pub fn input_tensor_id(&self) -> u32 {
        self.input_tensor_id
    }

    pub fn effect_tensor_id(&self) -> u32 {
        self.effect_tensor_id
    }

    pub fn mask_tensor_id(&self) -> u32 {
        self.mask_tensor_id
    }

    pub fn input_nhwc4_sha256(&self) -> [u8; 32] {
        self.input_nhwc4_sha256
    }

    pub fn effect_nhwc4_sha256(&self) -> [u8; 32] {
        self.effect_nhwc4_sha256
    }

    pub fn mask_nhwc4_sha256(&self) -> [u8; 32] {
        self.mask_nhwc4_sha256
    }

    pub fn material_planar_sha256(&self) -> [u8; 32] {
        self.material_planar_sha256
    }

    pub fn mask_planar_sha256(&self) -> [u8; 32] {
        self.mask_planar_sha256
    }
}

/// Owned PF90130 model outputs after the same-job contract has been checked.
#[derive(Clone, Debug, PartialEq)]
pub struct Pf90130GlareProducer {
    receipt: Pf90130SameJobReceipt,
    effect_nhwc4: Vec<u8>,
    mask_nhwc4: Vec<u8>,
}

impl Pf90130GlareProducer {
    /// Binds the exact model, input node and two output nodes to one job.
    ///
    /// The raw payloads are the physical `(h,w,4)` float32 tensors:
    /// effect lanes 0..2 -> A0/A1/A2, mask lane 0 -> B0. Non-zero padding
    /// bytes are refused before any bridge or postprocess runs.
    #[allow(clippy::too_many_arguments)]
    pub fn from_same_job(
        ticket: u64,
        model_id: u32,
        input_tensor_id: u32,
        input_nhwc4: &[u8],
        effect_tensor_id: u32,
        effect_nhwc4: &[u8],
        mask_tensor_id: u32,
        mask_nhwc4: &[u8],
    ) -> Result<Self, Refusal> {
        if ticket == 0 {
            return Err(Refusal::MissingInput("PF90130_SAME_JOB_RECEIPT"));
        }
        if model_id != PF90130_MODEL_ID {
            return Err(Refusal::ValueOutOfRange("PF90130_MODEL_ID"));
        }
        if input_tensor_id != PF90130_INPUT_TENSOR_ID {
            return Err(Refusal::ValueOutOfRange("PF90130_INPUT_TENSOR"));
        }
        if effect_tensor_id != PF90130_EFFECT_TENSOR_ID {
            return Err(Refusal::ValueOutOfRange("PF90130_EFFECT_TENSOR"));
        }
        if mask_tensor_id != PF90130_MASK_TENSOR_ID {
            return Err(Refusal::ValueOutOfRange("PF90130_MASK_TENSOR"));
        }

        validate_input_nhwc4(input_nhwc4)?;
        validate_effect_nhwc4(effect_nhwc4)?;
        validate_mask_nhwc4(mask_nhwc4)?;
        let (material_planar_sha256, mask_planar_sha256) = planar_sha256s(effect_nhwc4, mask_nhwc4);
        let receipt = Pf90130SameJobReceipt {
            ticket,
            model_id,
            input_tensor_id,
            effect_tensor_id,
            mask_tensor_id,
            input_nhwc4_sha256: sha256_32(input_nhwc4),
            effect_nhwc4_sha256: sha256_32(effect_nhwc4),
            mask_nhwc4_sha256: sha256_32(mask_nhwc4),
            material_planar_sha256,
            mask_planar_sha256,
        };
        Ok(Self {
            receipt,
            effect_nhwc4: effect_nhwc4.to_vec(),
            mask_nhwc4: mask_nhwc4.to_vec(),
        })
    }

    pub fn receipt(&self) -> &Pf90130SameJobReceipt {
        &self.receipt
    }

    /// wl35 bridge entry: only the explicit A0/A1/A2 and B0 mapping is used.
    pub fn bridge(&self) -> Result<GlarePlanarBundle, Refusal> {
        bridge_glare_planes(&self.planar_inputs())
    }

    /// wl35 stage entry after the exact planar bridge.
    pub fn run_stage_writeback(
        &self,
        source: &Image,
        destination: &Image,
        control: f32,
        lens_mask: Option<&[f32]>,
    ) -> Result<Option<GlareWritebackOutput>, Refusal> {
        require_aligned_frame(source, destination)?;
        apply_glare_stage_writeback(
            source,
            destination,
            &GlareWritebackInputs {
                control: Some(control),
                source_sha256: Some(sha256_32(&source.pixels)),
                destination_sha256: Some(sha256_32(&destination.pixels)),
                planes: self.planar_inputs(),
                lens_mask,
            },
        )
    }

    /// Builds the wl41 chain input with the producer ticket and raw A/B bytes.
    pub fn chain_input<'a>(
        &'a self,
        geometry: &'a Pf90130AlignedGeometry<'_>,
    ) -> pf90130_chain::Inputs<'a> {
        pf90130_chain::Inputs {
            ticket: Some(self.receipt.ticket),
            width: PF90130_PLANE_WIDTH,
            height: PF90130_PLANE_HEIGHT,
            source_width: Some(geometry.source_width),
            source_height: Some(geometry.source_height),
            source_roi: Some(geometry.source_roi),
            roi: Some(PixelRect {
                x: 0,
                y: 0,
                width: PF90130_PLANE_WIDTH,
                height: PF90130_PLANE_HEIGHT,
            }),
            uv: Some(geometry.uv),
            src_to_dst: Some(geometry.src_to_dst),
            source_rgb8: Some(geometry.source_rgb8),
            base_rgb8: Some(geometry.base_rgb8),
            add_rgb_f32: Some(geometry.add_rgb_f32),
            material_nhwc4: Some(&self.effect_nhwc4),
            mask_nhwc4: Some(&self.mask_nhwc4),
            strength: Some(geometry.strength),
        }
    }

    /// Runs the wl41 six-kernel chain and its same-frame residual writeback.
    pub fn run_residual_writeback(
        &self,
        source: &Image,
        destination: &Image,
        geometry: &Pf90130AlignedGeometry<'_>,
    ) -> Result<Option<GlareFinalWritebackOutput>, Refusal> {
        require_aligned_frame(source, destination)?;
        let _planes = self.bridge()?;
        let input = self.chain_input(geometry);
        let output = pf90130_chain::run(&input)?;
        apply_glare_chain_final_writeback(
            source,
            destination,
            &input,
            &output,
            &GlareChainFinalWritebackInputs {
                source_sha256: Some(sha256_32(&source.pixels)),
                destination_sha256: Some(sha256_32(&destination.pixels)),
            },
        )
    }

    fn planar_inputs(&self) -> GlarePlanarInputs<'_> {
        GlarePlanarInputs {
            ticket: Some(self.receipt.ticket),
            material_nhwc4: Some(&self.effect_nhwc4),
            mask_nhwc4: Some(&self.mask_nhwc4),
            receipts: Some(GlarePlanarReceipts {
                material_raw_sha256: self.receipt.effect_nhwc4_sha256,
                mask_raw_sha256: self.receipt.mask_nhwc4_sha256,
                material_planar_sha256: self.receipt.material_planar_sha256,
                mask_planar_sha256: self.receipt.mask_planar_sha256,
            }),
        }
    }
}

/// Same-job geometry consumed by the wl41 chain. The aligned source and
/// destination images are fixed at 768x384 by `run_residual_writeback`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pf90130AlignedGeometry<'a> {
    pub source_width: u32,
    pub source_height: u32,
    pub source_roi: PixelRect,
    pub uv: &'a [[f32; 2]],
    pub src_to_dst: [[f32; 3]; 2],
    pub source_rgb8: &'a [u8],
    pub base_rgb8: &'a [u8],
    pub add_rgb_f32: &'a [f32],
    pub strength: f32,
}

fn require_aligned_frame(source: &Image, destination: &Image) -> Result<(), Refusal> {
    source.validate()?;
    destination.validate()?;
    if (source.width, source.height) != (PF90130_PLANE_WIDTH, PF90130_PLANE_HEIGHT)
        || (destination.width, destination.height) != (PF90130_PLANE_WIDTH, PF90130_PLANE_HEIGHT)
    {
        return Err(Refusal::ShapeMismatch("PF90130_ALIGNED_FRAME"));
    }
    Ok(())
}

fn lane_bits(pixel: &[u8], lane: usize) -> u32 {
    u32::from_le_bytes(
        pixel[lane * 4..lane * 4 + 4]
            .try_into()
            .expect("four-byte lane"),
    )
}

fn validate_input_nhwc4(input: &[u8]) -> Result<(), Refusal> {
    if input.len() != PF90130_NHWC4_BYTES {
        return Err(Refusal::ShapeMismatch("PF90130_INPUT_NHWC4"));
    }
    for pixel in input.chunks_exact(16) {
        if pixel[12..16].iter().any(|byte| *byte != 0) {
            return Err(Refusal::ValueOutOfRange("PF90130_INPUT_PADDING"));
        }
        for lane in 0..3 {
            if !f32::from_bits(lane_bits(pixel, lane)).is_finite() {
                return Err(Refusal::ValueOutOfRange("PF90130_INPUT_NHWC4"));
            }
        }
    }
    Ok(())
}

fn validate_effect_nhwc4(effect: &[u8]) -> Result<(), Refusal> {
    if effect.len() != PF90130_NHWC4_BYTES {
        return Err(Refusal::ShapeMismatch("PF90130_EFFECT_NHWC4"));
    }
    for pixel in effect.chunks_exact(16) {
        if pixel[12..16].iter().any(|byte| *byte != 0) {
            return Err(Refusal::ValueOutOfRange("PF90130_EFFECT_PADDING"));
        }
        for lane in 0..3 {
            if !f32::from_bits(lane_bits(pixel, lane)).is_finite() {
                return Err(Refusal::ValueOutOfRange("PF90130_EFFECT_NHWC4"));
            }
        }
    }
    Ok(())
}

fn validate_mask_nhwc4(mask: &[u8]) -> Result<(), Refusal> {
    if mask.len() != PF90130_NHWC4_BYTES {
        return Err(Refusal::ShapeMismatch("PF90130_MASK_NHWC4"));
    }
    for pixel in mask.chunks_exact(16) {
        if pixel[4..16].iter().any(|byte| *byte != 0) {
            return Err(Refusal::ValueOutOfRange("PF90130_MASK_PADDING"));
        }
        if !f32::from_bits(lane_bits(pixel, 0)).is_finite() {
            return Err(Refusal::ValueOutOfRange("PF90130_MASK_NHWC4"));
        }
    }
    Ok(())
}

fn planar_sha256s(effect: &[u8], mask: &[u8]) -> ([u8; 32], [u8; 32]) {
    let mut material_planar = Vec::with_capacity(PF90130_A_PLANAR_BYTES);
    for plane in 0..3 {
        for pixel in effect.chunks_exact(16) {
            material_planar.extend_from_slice(&pixel[plane * 4..plane * 4 + 4]);
        }
    }
    let mut mask_planar = Vec::with_capacity(PF90130_B_PLANAR_BYTES);
    for pixel in mask.chunks_exact(16) {
        mask_planar.extend_from_slice(&pixel[..4]);
    }
    (sha256_32(&material_planar), sha256_32(&mask_planar))
}
