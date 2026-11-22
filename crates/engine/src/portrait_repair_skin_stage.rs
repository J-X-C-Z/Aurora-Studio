use super::sha256_32;
use aurora_portrait::kernels::{blemish, face_detail, pf90130_chain, skin_color};
use aurora_portrait::{Image, Refusal};
use std::collections::BTreeMap;

/// Same-job inputs for the three skin-repair rows left open after wl32.
#[derive(Clone, Debug, PartialEq)]
pub enum SkinRepairData {
    Spotless(SpotlessInputs),
    Nostril(NostrilInputs),
    Glare(GlareInputs),
}

impl SkinRepairData {
    fn pf(&self) -> u32 {
        match self {
            Self::Spotless(_) => 0,
            Self::Nostril(_) => 21104,
            Self::Glare(_) => 90130,
        }
    }
}

/// PF0 producer inputs. The final C Mat and vendor writeback are intentionally
/// outside this type until the local evidence proves which plane is selected.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpotlessInputs {
    pub face_id: String,
    pub roi: blemish::Roi,
    pub skin_mask: Vec<u8>,
    pub hair_mask: Vec<u8>,
    pub instance_mask: Vec<u8>,
    pub slots: blemish::SlotMasks,
    pub binding_sha256: [u8; 32],
    pub detector_output: Vec<u8>,
    pub acne_mask: Vec<u8>,
    pub blemish_model_output: Vec<u8>,
    pub repair_model_output: Vec<u8>,
    pub composition_flag: u8,
}

/// PF21104 consumer planes. They are full-frame after the same-face producer.
#[derive(Clone, Debug, PartialEq)]
pub struct NostrilInputs {
    pub m: Vec<f32>,
    pub a: Vec<f32>,
    pub v: Vec<f32>,
    pub t: Vec<f32>,
}

/// PF90130 consumer planes, not raw NHWC4. The raw-to-planar bridge remains a
/// separate runtime producer and is deliberately not guessed here.
#[derive(Clone, Debug, PartialEq)]
pub struct GlareInputs {
    pub effect_rgb: Vec<f32>,
    pub mask: Option<Vec<f32>>,
    pub lens_mask: Option<Vec<f32>>,
    pub encoding: face_detail::GlareMaskEncoding,
}

/// Typed row output. Spotless stops before the unresolved final C Mat; Nostril
/// returns RGB and its A-derived mask; Glare returns the exact RGB8 stage.
#[derive(Clone, Debug, PartialEq)]
pub enum SkinRepairOutput {
    Spotless {
        law: blemish::Pf0Law,
        detector8: Vec<u8>,
        mask8: Vec<u8>,
        repair_rgb8: Vec<u8>,
        changed: bool,
    },
    Nostril {
        rgb8: Vec<u8>,
        mask8: Vec<u8>,
        active: bool,
    },
    Glare {
        rgba8: Image,
    },
}

#[derive(Clone, Debug, PartialEq)]
struct BoundSkinRepair {
    ticket: u64,
    size: [u32; 2],
    source_sha256: [u8; 32],
    control: f32,
    data: SkinRepairData,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SkinRepairProducerInputs(BTreeMap<u32, BoundSkinRepair>);

impl SkinRepairProducerInputs {
    pub fn insert(
        &mut self,
        ticket: u64,
        source: &Image,
        control: f32,
        data: SkinRepairData,
    ) -> Result<(), Refusal> {
        source.validate()?;
        if ticket == 0 || !(0.0..=1.0).contains(&control) {
            return Err(Refusal::ValueOutOfRange("skin repair binding"));
        }
        self.0.insert(
            data.pf(),
            BoundSkinRepair {
                ticket,
                size: [source.width, source.height],
                source_sha256: sha256_32(&source.pixels),
                control,
                data,
            },
        );
        Ok(())
    }
}

pub fn run(
    source: &Image,
    ticket: Option<u64>,
    pf: u32,
    control: f32,
    inputs: Option<&SkinRepairProducerInputs>,
) -> Result<Option<SkinRepairOutput>, Refusal> {
    source.validate()?;
    if !control.is_finite() || !(0.0..=1.0).contains(&control) {
        return Err(Refusal::ValueOutOfRange("skin repair control"));
    }
    if control == 0.0 {
        return Ok(None);
    }
    let bound = inputs
        .and_then(|inputs| inputs.0.get(&pf))
        .ok_or(Refusal::MissingInput("skin repair same-job input"))?;
    if bound.data.pf() != pf {
        return Err(Refusal::ValueOutOfRange("skin repair payload PF"));
    }
    if bound.ticket == 0 || Some(bound.ticket) != ticket {
        return Err(Refusal::ValueOutOfRange("skin repair ticket"));
    }
    if bound.size != [source.width, source.height] {
        return Err(Refusal::ShapeMismatch("skin repair frame size"));
    }
    if bound.control != control {
        return Err(Refusal::ValueOutOfRange("skin repair payload control"));
    }
    if bound.source_sha256 != sha256_32(&source.pixels) {
        return Err(Refusal::ValueOutOfRange("skin repair frame digest"));
    }

    let output = match &bound.data {
        SkinRepairData::Spotless(inputs) => {
            let job = bound.ticket.to_string();
            let frame = blemish::Frame {
                job_id: &job,
                source_sha256: "validated-source-sha256",
                face_id: &inputs.face_id,
                image_width: source.width,
                image_height: source.height,
                roi: inputs.roi,
                skin_mask: &inputs.skin_mask,
                hair_mask: &inputs.hair_mask,
                instance_mask: &inputs.instance_mask,
                slots: &inputs.slots,
                binding_sha256: &inputs.binding_sha256,
                detector_output: &inputs.detector_output,
                acne_mask: &inputs.acne_mask,
                blemish_model_output: &inputs.blemish_model_output,
                repair_model_output: &inputs.repair_model_output,
                composition_flag: inputs.composition_flag,
            };
            let result = blemish::produce(&frame)?;
            SkinRepairOutput::Spotless {
                law: result.law,
                detector8: result.detector,
                mask8: result.composite,
                repair_rgb8: inputs.repair_model_output.clone(),
                changed: result.changed,
            }
        }
        SkinRepairData::Nostril(inputs) => {
            let rgb = rgba_rgb(&source.pixels);
            let result = skin_color::nostril_apply(
                source.width,
                source.height,
                &rgb,
                &inputs.m,
                &inputs.a,
                &inputs.v,
                &inputs.t,
                control,
            )?;
            SkinRepairOutput::Nostril {
                rgb8: result.rgb,
                mask8: result.mask,
                active: result.active,
            }
        }
        SkinRepairData::Glare(inputs) => {
            let ui_percent = (control * 100.0).round() as i32;
            let output = face_detail::glasses_glare_removal(
                source,
                &inputs.effect_rgb,
                inputs.mask.as_deref(),
                inputs.lens_mask.as_deref(),
                inputs.encoding,
                ui_percent,
            )?;
            if output == *source {
                return Ok(None);
            }
            SkinRepairOutput::Glare { rgba8: output }
        }
    };
    Ok(Some(output))
}

pub const PF90130_PLANE_WIDTH: u32 = 768;
pub const PF90130_PLANE_HEIGHT: u32 = 384;
pub const PF90130_PIXELS: usize = PF90130_PLANE_WIDTH as usize * PF90130_PLANE_HEIGHT as usize;
pub const PF90130_A_PLANAR_BYTES: usize = PF90130_PIXELS * 3 * 4;
pub const PF90130_B_PLANAR_BYTES: usize = PF90130_PIXELS * 4;
const PF90130_RAW_BYTES: usize = PF90130_PIXELS * 16;

/// Receipts required before the raw NHWC4 payload is allowed to cross into
/// the vendor consumer plane layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlarePlanarReceipts {
    pub material_raw_sha256: [u8; 32],
    pub mask_raw_sha256: [u8; 32],
    pub material_planar_sha256: [u8; 32],
    pub mask_planar_sha256: [u8; 32],
}

/// PF90130 A/B raw payloads for the 768x384 aligned consumer. The caller must
/// provide both raw and planar digests; the bridge never treats identity lane
/// order as a default.
#[derive(Clone, Debug, PartialEq)]
pub struct GlarePlanarInputs<'a> {
    pub ticket: Option<u64>,
    pub material_nhwc4: Option<&'a [u8]>,
    pub mask_nhwc4: Option<&'a [u8]>,
    pub receipts: Option<GlarePlanarReceipts>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlarePlanarBundle {
    pub ticket: u64,
    pub effect_rgb: Vec<f32>,
    pub mask: Vec<f32>,
    pub material_planar: Vec<u8>,
    pub mask_planar: Vec<u8>,
    pub material_planar_sha256: [u8; 32],
    pub mask_planar_sha256: [u8; 32],
}

/// Performs the exact formal-artifact bridge:
/// `NHWC4 lane 0..2 -> A0/A1/A2` and `NHWC4 lane 0 -> B0`.
pub fn bridge_glare_planes(input: &GlarePlanarInputs<'_>) -> Result<GlarePlanarBundle, Refusal> {
    let ticket = input
        .ticket
        .filter(|ticket| *ticket != 0)
        .ok_or(Refusal::MissingInput("PF90130_PLANAR_TICKET"))?;
    let material = input
        .material_nhwc4
        .ok_or(Refusal::MissingInput("PF90130_MATERIAL_NHWC4"))?;
    let mask = input
        .mask_nhwc4
        .ok_or(Refusal::MissingInput("PF90130_MASK_NHWC4"))?;
    if material.len() != PF90130_RAW_BYTES || mask.len() != PF90130_RAW_BYTES {
        return Err(Refusal::ShapeMismatch("PF90130_NHWC4"));
    }
    let receipts = input
        .receipts
        .ok_or(Refusal::MissingInput("PF90130_PLANAR_RECEIPTS"))?;
    if sha256_32(material) != receipts.material_raw_sha256 {
        return Err(Refusal::ValueOutOfRange("PF90130_MATERIAL_RAW_SHA256"));
    }
    if sha256_32(mask) != receipts.mask_raw_sha256 {
        return Err(Refusal::ValueOutOfRange("PF90130_MASK_RAW_SHA256"));
    }

    let mut material_values = Vec::with_capacity(PF90130_PIXELS);
    let mut mask_values = Vec::with_capacity(PF90130_PIXELS);
    for (material_pixel, mask_pixel) in material.chunks_exact(16).zip(mask.chunks_exact(16)) {
        let mut material_bits = [0u32; 4];
        let mut mask_bits = [0u32; 4];
        for lane in 0..4 {
            material_bits[lane] = u32::from_le_bytes(
                material_pixel[lane * 4..lane * 4 + 4]
                    .try_into()
                    .expect("four-byte material lane"),
            );
            mask_bits[lane] = u32::from_le_bytes(
                mask_pixel[lane * 4..lane * 4 + 4]
                    .try_into()
                    .expect("four-byte mask lane"),
            );
        }
        if material_bits[3] != 0 {
            return Err(Refusal::ValueOutOfRange("PF90130_MATERIAL_PADDING"));
        }
        if mask_bits[1..].iter().any(|bits| *bits != 0) {
            return Err(Refusal::ValueOutOfRange("PF90130_MASK_PADDING"));
        }
        let mut material_pixel = [0.0f32; 4];
        let mut mask_pixel = [0.0f32; 4];
        for lane in 0..4 {
            material_pixel[lane] = f32::from_bits(material_bits[lane]);
            mask_pixel[lane] = f32::from_bits(mask_bits[lane]);
        }
        if material_pixel[..3].iter().any(|value| !value.is_finite()) || !mask_pixel[0].is_finite()
        {
            return Err(Refusal::ValueOutOfRange("PF90130_NHWC4"));
        }
        material_values.push(material_pixel);
        mask_values.push(mask_pixel[0]);
    }

    let mut material_planar = Vec::with_capacity(PF90130_A_PLANAR_BYTES);
    let mut effect_rgb = Vec::with_capacity(PF90130_PIXELS * 3);
    for plane in 0..3 {
        for pixel in &material_values {
            material_planar.extend_from_slice(&pixel[plane].to_le_bytes());
            effect_rgb.push(pixel[plane]);
        }
    }
    let mut mask_planar = Vec::with_capacity(PF90130_B_PLANAR_BYTES);
    for value in &mask_values {
        mask_planar.extend_from_slice(&value.to_le_bytes());
    }
    let material_planar_sha256 = sha256_32(&material_planar);
    let mask_planar_sha256 = sha256_32(&mask_planar);
    if material_planar_sha256 != receipts.material_planar_sha256 {
        return Err(Refusal::ValueOutOfRange("PF90130_MATERIAL_PLANAR_SHA256"));
    }
    if mask_planar_sha256 != receipts.mask_planar_sha256 {
        return Err(Refusal::ValueOutOfRange("PF90130_MASK_PLANAR_SHA256"));
    }

    Ok(GlarePlanarBundle {
        ticket,
        effect_rgb,
        mask: mask_values,
        material_planar,
        mask_planar,
        material_planar_sha256,
        mask_planar_sha256,
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlareWritebackInputs<'a> {
    pub control: Option<f32>,
    pub source_sha256: Option<[u8; 32]>,
    pub destination_sha256: Option<[u8; 32]>,
    pub planes: GlarePlanarInputs<'a>,
    pub lens_mask: Option<&'a [f32]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlareWritebackOutput {
    pub ticket: u64,
    pub rgba8: Image,
    pub planes: GlarePlanarBundle,
}

/// Runs the existing typed PF90130 stage after the exact planar bridge, then
/// writes only RGB into the destination frame. Destination alpha is preserved.
pub fn apply_glare_stage_writeback(
    source: &Image,
    destination: &Image,
    inputs: &GlareWritebackInputs<'_>,
) -> Result<Option<GlareWritebackOutput>, Refusal> {
    source.validate()?;
    destination.validate()?;
    if source.width != destination.width || source.height != destination.height {
        return Err(Refusal::ShapeMismatch("PF90130_WRITEBACK_FRAME"));
    }
    let control = inputs
        .control
        .filter(|value| value.is_finite() && (0.0..=1.0).contains(value))
        .ok_or(Refusal::MissingInput("PF90130_CONTROL"))?;
    if control == 0.0 {
        return Ok(None);
    }
    let source_sha256 = inputs
        .source_sha256
        .ok_or(Refusal::MissingInput("PF90130_SOURCE_FRAME_SHA256"))?;
    if source_sha256 != sha256_32(&source.pixels) {
        return Err(Refusal::ValueOutOfRange("PF90130_SOURCE_FRAME_SHA256"));
    }
    let destination_sha256 = inputs
        .destination_sha256
        .ok_or(Refusal::MissingInput("PF90130_DESTINATION_FRAME_SHA256"))?;
    if destination_sha256 != sha256_32(&destination.pixels) {
        return Err(Refusal::ValueOutOfRange("PF90130_DESTINATION_FRAME_SHA256"));
    }
    let planes = bridge_glare_planes(&inputs.planes)?;
    let mut typed = SkinRepairProducerInputs::default();
    typed.insert(
        planes.ticket,
        source,
        control,
        SkinRepairData::Glare(GlareInputs {
            effect_rgb: planes.effect_rgb.clone(),
            mask: Some(planes.mask.clone()),
            lens_mask: inputs.lens_mask.map(ToOwned::to_owned),
            encoding: face_detail::GlareMaskEncoding::SignedNetwork,
        }),
    )?;
    let Some(SkinRepairOutput::Glare { rgba8 }) =
        run(source, Some(planes.ticket), 90130, control, Some(&typed))?
    else {
        return Ok(None);
    };
    let mut output = destination.clone();
    for (destination, source) in output
        .pixels
        .chunks_exact_mut(4)
        .zip(rgba8.pixels.chunks_exact(4))
    {
        destination[..3].copy_from_slice(&source[..3]);
    }
    Ok(Some(GlareWritebackOutput {
        ticket: planes.ticket,
        rgba8: output,
        planes,
    }))
}

fn round_ties_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction > 0.5 {
        floor + 1.0
    } else if fraction < 0.5 {
        floor
    } else if (floor as i64) & 1 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlareFinalWritebackInputs<'a> {
    pub ticket: Option<u64>,
    pub residual_ticket: Option<u64>,
    pub source_sha256: Option<[u8; 32]>,
    pub destination_sha256: Option<[u8; 32]>,
    pub residual_rgb_f32: Option<&'a [f32]>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GlareFinalWritebackOutput {
    pub ticket: u64,
    pub rgba8: Image,
}

/// Implements the final vendor worker at `0x46F0E80`: add the post-second
/// residual to source RGB with nearest-even conversion, clamp, and preserve
/// destination alpha. Zero residuals leave the destination pixel unchanged.
pub fn apply_glare_final_writeback(
    source: &Image,
    destination: &Image,
    inputs: &GlareFinalWritebackInputs<'_>,
) -> Result<GlareFinalWritebackOutput, Refusal> {
    source.validate()?;
    destination.validate()?;
    if source.width != destination.width || source.height != destination.height {
        return Err(Refusal::ShapeMismatch("PF90130_FINAL_FRAME"));
    }
    let ticket = inputs
        .ticket
        .filter(|ticket| *ticket != 0)
        .ok_or(Refusal::MissingInput("PF90130_FINAL_TICKET"))?;
    if inputs.residual_ticket != Some(ticket) {
        return Err(Refusal::ValueOutOfRange("PF90130_FINAL_RECEIPT"));
    }
    let source_sha256 = inputs
        .source_sha256
        .ok_or(Refusal::MissingInput("PF90130_FINAL_SOURCE_SHA256"))?;
    if source_sha256 != sha256_32(&source.pixels) {
        return Err(Refusal::ValueOutOfRange("PF90130_FINAL_SOURCE_SHA256"));
    }
    let destination_sha256 = inputs
        .destination_sha256
        .ok_or(Refusal::MissingInput("PF90130_FINAL_DESTINATION_SHA256"))?;
    if destination_sha256 != sha256_32(&destination.pixels) {
        return Err(Refusal::ValueOutOfRange("PF90130_FINAL_DESTINATION_SHA256"));
    }
    let residual = inputs
        .residual_rgb_f32
        .ok_or(Refusal::MissingInput("PF90130_FINAL_RESIDUAL"))?;
    let expected = source
        .pixel_count()
        .checked_mul(3)
        .ok_or(Refusal::ShapeMismatch("PF90130_FINAL_RESIDUAL"))?;
    if residual.len() != expected {
        return Err(Refusal::ShapeMismatch("PF90130_FINAL_RESIDUAL"));
    }
    if residual.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("PF90130_FINAL_RESIDUAL"));
    }
    let mut output = destination.clone();
    for pixel in 0..source.pixel_count() {
        let values = &residual[pixel * 3..pixel * 3 + 3];
        if values.iter().all(|value| *value == 0.0) {
            continue;
        }
        let offset = pixel * 4;
        for channel in 0..3 {
            let value =
                round_ties_even((source.pixels[offset + channel] as f32 + values[channel]) as f64);
            output.pixels[offset + channel] = value.clamp(0.0, 255.0) as u8;
        }
    }
    Ok(GlareFinalWritebackOutput {
        ticket,
        rgba8: output,
    })
}

/// Receipts for binding `pf90130_chain::Output.post_second_f32` to the final
/// worker. The chain input/output must name the same full frame as `source`.
#[derive(Clone, Debug, PartialEq)]
pub struct GlareChainFinalWritebackInputs {
    pub source_sha256: Option<[u8; 32]>,
    pub destination_sha256: Option<[u8; 32]>,
}

/// Binds the six-kernel `post_second_f32` plane to `0x46F0E80`.
///
/// The chain emits one f32 RGB triple per destination pixel in row-major
/// order. The final worker consumes that plane at the same pixel index, so
/// width, height, ROI, ticket and source RGB all have to match the frame that
/// the residual is written into.
pub fn apply_glare_chain_final_writeback(
    source: &Image,
    destination: &Image,
    chain_input: &pf90130_chain::Inputs<'_>,
    chain_output: &pf90130_chain::Output,
    inputs: &GlareChainFinalWritebackInputs,
) -> Result<Option<GlareFinalWritebackOutput>, Refusal> {
    source.validate()?;
    destination.validate()?;
    if source.width != destination.width || source.height != destination.height {
        return Err(Refusal::ShapeMismatch("PF90130_FINAL_FRAME"));
    }
    let ticket = chain_input
        .ticket
        .filter(|ticket| *ticket != 0)
        .ok_or(Refusal::MissingInput("PF90130_FINAL_TICKET"))?;
    if chain_output.ticket != ticket {
        return Err(Refusal::ValueOutOfRange("PF90130_FINAL_RECEIPT"));
    }
    if chain_input.width != source.width || chain_input.height != source.height {
        return Err(Refusal::ShapeMismatch("PF90130_FINAL_FRAME"));
    }
    let full = pf90130_chain::PixelRect {
        x: 0,
        y: 0,
        width: source.width,
        height: source.height,
    };
    if chain_input.roi != Some(full) {
        return Err(Refusal::ValueOutOfRange("PF90130_ROI"));
    }
    let rgb_len = source
        .pixel_count()
        .checked_mul(3)
        .ok_or(Refusal::ShapeMismatch("PF90130_FINAL_FRAME"))?;
    if chain_output.blend_rgb8.len() != rgb_len {
        return Err(Refusal::ShapeMismatch("PF90130_FINAL_FRAME"));
    }
    let chain_source = chain_input
        .source_rgb8
        .ok_or(Refusal::MissingInput("PF90130_SOURCE_RGB8"))?;
    if chain_source.len() != rgb_len
        || chain_source
            .chunks_exact(3)
            .zip(source.pixels.chunks_exact(4))
            .any(|(rgb, rgba)| rgb != &rgba[..3])
    {
        return Err(Refusal::ValueOutOfRange("PF90130_CHAIN_SOURCE_FRAME"));
    }
    if !chain_output.post_gate_open {
        if !chain_output.post_second_f32.is_empty() {
            return Err(Refusal::ShapeMismatch("PF90130_FINAL_GATE"));
        }
        return Ok(None);
    }
    apply_glare_final_writeback(
        source,
        destination,
        &GlareFinalWritebackInputs {
            ticket: Some(ticket),
            residual_ticket: Some(chain_output.ticket),
            source_sha256: inputs.source_sha256,
            destination_sha256: inputs.destination_sha256,
            residual_rgb_f32: Some(&chain_output.post_second_f32),
        },
    )
    .map(Some)
}

fn rgba_rgb(rgba: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(rgba.len() / 4 * 3);
    for pixel in rgba.chunks_exact(4) {
        rgb.extend_from_slice(&pixel[..3]);
    }
    rgb
}
