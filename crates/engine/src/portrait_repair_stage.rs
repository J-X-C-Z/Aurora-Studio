use super::{check_degree01, fnv1a64, fnv_u64, sha256_32};
use aurora_portrait::kernels::{
    close_smile, double_chin, eye_bag, hair_body, neck_paste, skin_adjust, skin_color, sweat,
};
use aurora_portrait::{Image, Refusal};
use std::collections::BTreeMap;

/// Outputs at the recovered CPU pixel boundary. Coordinates are renderer pixels.
#[derive(Clone, Debug, PartialEq)]
pub enum RepairOutput {
    /// PF5. Full-frame RGB residual after warp and masking, in the native +80 encoding.
    DarkCircle {
        residual_rgb: Vec<f32>,
    },
    ClosedMouthSmile {
        model: Image,
        paste_enabled: bool,
    },
    Sweat {
        model: Image,
    },
    NeckLine {
        repair: Image,
        origin: [u32; 2],
    },
    DoubleChin {
        field: Vec<[f32; 2]>,
        mode: double_chin::Mode,
        scale: f32,
    },
    OilShine {
        effect_rgb: Vec<u8>,
        mask: Vec<u8>,
    },
    BodyBlemishChild {
        model: Image,
        roi: skin_adjust::Roi,
        roi_mask: Vec<u8>,
        native_per_face_degree: f32,
        alternate: bool,
    },
    FootBlueVein {
        repair: Image,
        model_mask: Vec<u8>,
        feather_radius: u32,
    },
    /// PF90118. ROI-local repair and two resolved FaceParsing masks.
    LipLine {
        repair: Image,
        origin: [u32; 2],
        mask_a: Vec<u8>,
        mask_b: Vec<u8>,
    },
    WhiteHair {
        repair: Image,
        raw_mask: Vec<u8>,
    },
}

impl RepairOutput {
    pub fn pf(&self) -> u32 {
        match self {
            Self::DarkCircle { .. } => 5,
            Self::ClosedMouthSmile { .. } => 21076,
            Self::Sweat { .. } => 30259,
            Self::NeckLine { .. } => 90015,
            Self::DoubleChin { .. } => 90034,
            Self::OilShine { .. } => 90080,
            Self::BodyBlemishChild { .. } => 90089,
            Self::FootBlueVein { .. } => 44297,
            Self::LipLine { .. } => 90118,
            Self::WhiteHair { .. } => 90131,
        }
    }
}

/// One raw producer plane and the digest recorded when it entered the
/// same-frame bundle. The digest covers only the named plane, never a
/// synthesized final frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepairPlaneSha256 {
    pub name: &'static str,
    pub sha256: [u8; 32],
}

/// Public proof of the exact job/frame/plane bindings stored in
/// `RepairProducerInputs`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RepairOutputReceipt {
    pub ticket: u64,
    pub pf: u32,
    pub source_sha256: [u8; 32],
    pub plane_sha256: Vec<RepairPlaneSha256>,
}

/// Raw model-side inputs for the ten rows whose final seam is already in
/// [`RepairOutput`]. Every payload is optional so a missing named plane reaches
/// the builder as a typed refusal instead of being synthesized.
#[derive(Clone, Debug, PartialEq)]
pub enum RepairModelPlanes {
    DarkCircle {
        residual_rgb: Option<Vec<f32>>,
    },
    ClosedMouthSmile {
        model: Option<Image>,
        paste_enabled: Option<bool>,
    },
    Sweat {
        model: Option<Image>,
    },
    NeckLine {
        repair: Option<Image>,
        roi: Option<skin_adjust::Roi>,
    },
    DoubleChin {
        field: Option<Vec<[f32; 2]>>,
        mode: Option<double_chin::Mode>,
        scale: Option<f32>,
    },
    OilShine {
        effect_rgb: Option<Vec<u8>>,
        mask: Option<Vec<u8>>,
    },
    BodyBlemishChild {
        model: Option<Image>,
        roi: Option<skin_adjust::Roi>,
        roi_mask: Option<Vec<u8>>,
        native_per_face_degree: Option<f32>,
        alternate: Option<bool>,
    },
    FootBlueVein {
        repair: Option<Image>,
        model_mask: Option<Vec<u8>>,
        feather_radius: Option<u32>,
    },
    LipLine {
        repair: Option<Image>,
        origin: Option<[u32; 2]>,
        mask_a: Option<Vec<u8>>,
        mask_b: Option<Vec<u8>>,
    },
    WhiteHair {
        repair: Option<Image>,
        raw_mask: Option<Vec<u8>>,
    },
}

impl RepairModelPlanes {
    pub fn pf(&self) -> u32 {
        match self {
            Self::DarkCircle { .. } => 5,
            Self::ClosedMouthSmile { .. } => 21076,
            Self::Sweat { .. } => 30259,
            Self::NeckLine { .. } => 90015,
            Self::DoubleChin { .. } => 90034,
            Self::OilShine { .. } => 90080,
            Self::BodyBlemishChild { .. } => 90089,
            Self::FootBlueVein { .. } => 44297,
            Self::LipLine { .. } => 90118,
            Self::WhiteHair { .. } => 90131,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct BoundOutput {
    ticket: u64,
    source_sha256: [u8; 32],
    output: RepairOutput,
    plane_sha256: Vec<RepairPlaneSha256>,
}

impl BoundOutput {
    fn receipt(&self, pf: u32) -> RepairOutputReceipt {
        RepairOutputReceipt {
            ticket: self.ticket,
            pf,
            source_sha256: self.source_sha256,
            plane_sha256: self.plane_sha256.clone(),
        }
    }
}

/// Model output bindings carried in the renderer's producer bundle.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RepairProducerInputs(BTreeMap<u32, BoundOutput>);

impl RepairProducerInputs {
    /// Bind a result to the exact input pixels consumed by its producer.
    /// If an earlier effect changes the frame, bind to that effect's output.
    pub fn insert(
        &mut self,
        ticket: u64,
        source: &Image,
        output: RepairOutput,
    ) -> Result<(), Refusal> {
        source.validate()?;
        if ticket == 0 {
            return Err(Refusal::ValueOutOfRange("repair job ticket"));
        }
        let pf = output.pf();
        let plane_sha256 = repair_output_plane_sha256(&output);
        self.0
            .insert(pf, bind(ticket, source, output, plane_sha256));
        Ok(())
    }

    /// Build a typed `RepairOutput` from same-job model planes and bind it to
    /// the exact source frame. This performs validation only; the existing
    /// renderer consumer still owns the final composition/seam.
    pub fn insert_same_frame(
        &mut self,
        ticket: Option<u64>,
        source: &Image,
        planes: RepairModelPlanes,
    ) -> Result<RepairOutputReceipt, Refusal> {
        source.validate()?;
        let pf = planes.pf();
        let ticket = ticket
            .filter(|ticket| *ticket != 0)
            .ok_or(Refusal::MissingInput(match pf {
                5 => "PF5_JOB_TICKET",
                21076 => "PF21076_JOB_TICKET",
                30259 => "PF30259_JOB_TICKET",
                90015 => "PF90015_JOB_TICKET",
                90034 => "PF90034_JOB_TICKET",
                90080 => "PF90080_JOB_TICKET",
                90089 => "PF90089_JOB_TICKET",
                44297 => "PF44297_JOB_TICKET",
                90118 => "PF90118_JOB_TICKET",
                90131 => "PF90131_JOB_TICKET",
                _ => "repair job ticket",
            }))?;
        let (output, plane_sha256) = build_repair_output(source, planes)?;
        if output.pf() != pf {
            return Err(Refusal::ValueOutOfRange("repair producer PF"));
        }
        let bound = bind(ticket, source, output, plane_sha256);
        let receipt = bound.receipt(pf);
        self.0.insert(pf, bound);
        Ok(receipt)
    }

    pub fn receipt(&self, pf: u32) -> Option<RepairOutputReceipt> {
        self.0.get(&pf).map(|bound| bound.receipt(pf))
    }

    pub(super) fn fingerprint(&self, mut hash: u64) -> u64 {
        if self.0.is_empty() {
            return hash;
        }
        hash = fnv_u64(fnv1a64(hash, b"repair outputs"), self.0.len() as u64);
        for (&pf, bound) in &self.0 {
            hash = ints(hash, [u64::from(pf), bound.ticket]);
            hash = bytes(hash, &bound.source_sha256);
            for plane in &bound.plane_sha256 {
                hash = bytes(fnv1a64(hash, plane.name.as_bytes()), &plane.sha256);
            }
            hash = match &bound.output {
                RepairOutput::DarkCircle { residual_rgb } => floats(hash, residual_rgb.iter()),
                RepairOutput::ClosedMouthSmile {
                    model,
                    paste_enabled,
                } => fnv_u64(image_hash(hash, model), u64::from(*paste_enabled)),
                RepairOutput::Sweat { model } => image_hash(hash, model),
                RepairOutput::NeckLine { repair, origin } => {
                    ints(image_hash(hash, repair), origin.map(u64::from))
                }
                RepairOutput::DoubleChin { field, mode, scale } => floats(
                    ints(
                        hash,
                        [
                            u64::from(*mode == double_chin::Mode::Absolute),
                            u64::from(scale.to_bits()),
                        ],
                    ),
                    field.iter().flatten(),
                ),
                RepairOutput::OilShine { effect_rgb, mask } => bytes(bytes(hash, effect_rgb), mask),
                RepairOutput::BodyBlemishChild {
                    model,
                    roi,
                    roi_mask,
                    native_per_face_degree,
                    alternate,
                } => ints(
                    bytes(image_hash(hash, model), roi_mask),
                    [
                        u64::from(roi.x),
                        u64::from(roi.y),
                        u64::from(roi.width),
                        u64::from(roi.height),
                        u64::from(native_per_face_degree.to_bits()),
                        u64::from(*alternate),
                    ],
                ),
                RepairOutput::FootBlueVein {
                    repair,
                    model_mask,
                    feather_radius,
                } => fnv_u64(
                    bytes(image_hash(hash, repair), model_mask),
                    u64::from(*feather_radius),
                ),
                RepairOutput::LipLine {
                    repair,
                    origin,
                    mask_a,
                    mask_b,
                } => ints(
                    bytes(bytes(image_hash(hash, repair), mask_a), mask_b),
                    origin.map(u64::from),
                ),
                RepairOutput::WhiteHair { repair, raw_mask } => {
                    bytes(image_hash(hash, repair), raw_mask)
                }
            };
        }
        hash
    }
}

fn bind(
    ticket: u64,
    source: &Image,
    output: RepairOutput,
    plane_sha256: Vec<RepairPlaneSha256>,
) -> BoundOutput {
    BoundOutput {
        ticket,
        source_sha256: sha256_32(&source.pixels),
        output,
        plane_sha256,
    }
}

fn build_repair_output(
    source: &Image,
    planes: RepairModelPlanes,
) -> Result<(RepairOutput, Vec<RepairPlaneSha256>), Refusal> {
    let pixel_count = source.pixel_count();
    match planes {
        RepairModelPlanes::DarkCircle { residual_rgb } => {
            let residual_rgb = residual_rgb.ok_or(Refusal::MissingInput("PF5_RESIDUAL_RGB"))?;
            require_len("PF5_RESIDUAL_RGB", residual_rgb.len(), pixel_count * 3)?;
            require_finite("PF5_RESIDUAL_RGB", residual_rgb.iter())?;
            let plane_sha256 = vec![f32_plane_sha256("PF5_RESIDUAL_RGB", &residual_rgb)];
            Ok((RepairOutput::DarkCircle { residual_rgb }, plane_sha256))
        }
        RepairModelPlanes::ClosedMouthSmile {
            model,
            paste_enabled,
        } => {
            let model = model.ok_or(Refusal::MissingInput("PF21076_MODEL_RGB"))?;
            let paste_enabled =
                paste_enabled.ok_or(Refusal::MissingInput("PF21076_PASTE_ENABLED"))?;
            require_image_shape(source, &model, "PF21076_MODEL_RGB")?;
            let plane_sha256 = vec![image_plane_sha256("PF21076_MODEL_RGB", &model)];
            Ok((
                RepairOutput::ClosedMouthSmile {
                    model,
                    paste_enabled,
                },
                plane_sha256,
            ))
        }
        RepairModelPlanes::Sweat { model } => {
            let model = model.ok_or(Refusal::MissingInput("PF30259_MODEL_RGB"))?;
            require_image_shape(source, &model, "PF30259_MODEL_RGB")?;
            let plane_sha256 = vec![image_plane_sha256("PF30259_MODEL_RGB", &model)];
            Ok((RepairOutput::Sweat { model }, plane_sha256))
        }
        RepairModelPlanes::NeckLine { repair, roi } => {
            let repair = repair.ok_or(Refusal::MissingInput("PF90015_REPAIR_RGB"))?;
            let roi = roi.ok_or(Refusal::MissingInput("PF90015_ROI"))?;
            require_roi(source, roi, "PF90015_ROI")?;
            require_plane_matches_roi(&repair, roi, "PF90015_ROI")?;
            let plane_sha256 = vec![image_plane_sha256("PF90015_REPAIR_RGB", &repair)];
            Ok((
                RepairOutput::NeckLine {
                    repair,
                    origin: [roi.x, roi.y],
                },
                plane_sha256,
            ))
        }
        RepairModelPlanes::DoubleChin { field, mode, scale } => {
            let field = field.ok_or(Refusal::MissingInput("PF90034_SHIFT_FIELD"))?;
            let mode = mode.ok_or(Refusal::MissingInput("PF90034_SHIFT_MODE"))?;
            let scale = scale.ok_or(Refusal::MissingInput("PF90034_SHIFT_SCALE"))?;
            require_len("PF90034_SHIFT_FIELD", field.len(), pixel_count)?;
            require_finite("PF90034_SHIFT_FIELD", field.iter().flatten())?;
            if !scale.is_finite() {
                return Err(Refusal::ValueOutOfRange("PF90034_SHIFT_SCALE"));
            }
            let plane_sha256 = vec![vec2_plane_sha256("PF90034_SHIFT_FIELD", &field)];
            Ok((
                RepairOutput::DoubleChin { field, mode, scale },
                plane_sha256,
            ))
        }
        RepairModelPlanes::OilShine { effect_rgb, mask } => {
            let effect_rgb = effect_rgb.ok_or(Refusal::MissingInput("PF90080_EFFECT_RGB"))?;
            let mask = mask.ok_or(Refusal::MissingInput("PF90080_MASK"))?;
            require_len("PF90080_EFFECT_RGB", effect_rgb.len(), pixel_count * 3)?;
            require_len("PF90080_MASK", mask.len(), pixel_count)?;
            let plane_sha256 = vec![
                byte_plane_sha256("PF90080_EFFECT_RGB", &effect_rgb),
                byte_plane_sha256("PF90080_MASK", &mask),
            ];
            Ok((RepairOutput::OilShine { effect_rgb, mask }, plane_sha256))
        }
        RepairModelPlanes::BodyBlemishChild {
            model,
            roi,
            roi_mask,
            native_per_face_degree,
            alternate,
        } => {
            let model = model.ok_or(Refusal::MissingInput("PF90089_MODEL_RGB"))?;
            let roi = roi.ok_or(Refusal::MissingInput("PF90089_ROI"))?;
            let roi_mask = roi_mask.ok_or(Refusal::MissingInput("PF90089_ROI_MASK"))?;
            let native_per_face_degree = native_per_face_degree
                .ok_or(Refusal::MissingInput("PF90089_NATIVE_PER_FACE_DEGREE"))?;
            let alternate = alternate.ok_or(Refusal::MissingInput("PF90089_ALTERNATE"))?;
            require_image_shape(source, &model, "PF90089_MODEL_RGB")?;
            require_roi(source, roi, "PF90089_ROI")?;
            require_len(
                "PF90089_ROI_MASK",
                roi_mask.len(),
                roi.width as usize * roi.height as usize,
            )?;
            if !native_per_face_degree.is_finite() {
                return Err(Refusal::ValueOutOfRange("PF90089_NATIVE_PER_FACE_DEGREE"));
            }
            let plane_sha256 = vec![
                image_plane_sha256("PF90089_MODEL_RGB", &model),
                byte_plane_sha256("PF90089_ROI_MASK", &roi_mask),
            ];
            Ok((
                RepairOutput::BodyBlemishChild {
                    model,
                    roi,
                    roi_mask,
                    native_per_face_degree,
                    alternate,
                },
                plane_sha256,
            ))
        }
        RepairModelPlanes::FootBlueVein {
            repair,
            model_mask,
            feather_radius,
        } => {
            let repair = repair.ok_or(Refusal::MissingInput("PF44297_MODEL_REPAIR"))?;
            let model_mask = model_mask.ok_or(Refusal::MissingInput("PF44297_MODEL_MASK"))?;
            let feather_radius =
                feather_radius.ok_or(Refusal::MissingInput("PF44297_FEATHER_RADIUS"))?;
            require_image_shape(source, &repair, "PF44297_MODEL_REPAIR")?;
            require_len("PF44297_MODEL_MASK", model_mask.len(), pixel_count)?;
            if feather_radius < 1 {
                return Err(Refusal::ValueOutOfRange("PF44297_FEATHER_RADIUS"));
            }
            let plane_sha256 = vec![
                image_plane_sha256("PF44297_MODEL_REPAIR", &repair),
                byte_plane_sha256("PF44297_MODEL_MASK", &model_mask),
            ];
            Ok((
                RepairOutput::FootBlueVein {
                    repair,
                    model_mask,
                    feather_radius,
                },
                plane_sha256,
            ))
        }
        RepairModelPlanes::LipLine {
            repair,
            origin,
            mask_a,
            mask_b,
        } => {
            let repair = repair.ok_or(Refusal::MissingInput("PF90118_REPAIR_RGB"))?;
            let origin = origin.ok_or(Refusal::MissingInput("PF90118_ORIGIN"))?;
            let mask_a = mask_a.ok_or(Refusal::MissingInput("PF90118_MASK_A"))?;
            let mask_b = mask_b.ok_or(Refusal::MissingInput("PF90118_MASK_B"))?;
            require_origin_plane(source, &repair, origin, "PF90118_ORIGIN")?;
            let mask_len = repair.width as usize * repair.height as usize;
            require_len("PF90118_MASK_A", mask_a.len(), mask_len)?;
            require_len("PF90118_MASK_B", mask_b.len(), mask_len)?;
            let plane_sha256 = vec![
                image_plane_sha256("PF90118_REPAIR_RGB", &repair),
                byte_plane_sha256("PF90118_MASK_A", &mask_a),
                byte_plane_sha256("PF90118_MASK_B", &mask_b),
            ];
            Ok((
                RepairOutput::LipLine {
                    repair,
                    origin,
                    mask_a,
                    mask_b,
                },
                plane_sha256,
            ))
        }
        RepairModelPlanes::WhiteHair { repair, raw_mask } => {
            let repair = repair.ok_or(Refusal::MissingInput("PF90131_REPAIR_RGB"))?;
            let raw_mask = raw_mask.ok_or(Refusal::MissingInput("PF90131_RAW_HAIR_MASK"))?;
            require_image_shape(source, &repair, "PF90131_REPAIR_RGB")?;
            require_len("PF90131_RAW_HAIR_MASK", raw_mask.len(), pixel_count)?;
            let plane_sha256 = vec![
                image_plane_sha256("PF90131_REPAIR_RGB", &repair),
                byte_plane_sha256("PF90131_RAW_HAIR_MASK", &raw_mask),
            ];
            Ok((RepairOutput::WhiteHair { repair, raw_mask }, plane_sha256))
        }
    }
}

fn require_len(name: &'static str, actual: usize, expected: usize) -> Result<(), Refusal> {
    if actual == expected {
        Ok(())
    } else {
        Err(Refusal::ShapeMismatch(name))
    }
}

fn require_finite<'a>(
    name: &'static str,
    values: impl Iterator<Item = &'a f32>,
) -> Result<(), Refusal> {
    if values.into_iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(Refusal::ValueOutOfRange(name))
    }
}

fn require_image_shape(source: &Image, image: &Image, name: &'static str) -> Result<(), Refusal> {
    image.validate().map_err(|_| Refusal::ShapeMismatch(name))?;
    if source.width == image.width && source.height == image.height {
        Ok(())
    } else {
        Err(Refusal::ShapeMismatch(name))
    }
}

fn require_roi(source: &Image, roi: skin_adjust::Roi, name: &'static str) -> Result<(), Refusal> {
    let inside = roi.width > 0
        && roi.height > 0
        && u64::from(roi.x) + u64::from(roi.width) <= u64::from(source.width)
        && u64::from(roi.y) + u64::from(roi.height) <= u64::from(source.height);
    if inside {
        Ok(())
    } else {
        Err(Refusal::ValueOutOfRange(name))
    }
}

fn require_plane_matches_roi(
    plane: &Image,
    roi: skin_adjust::Roi,
    name: &'static str,
) -> Result<(), Refusal> {
    plane.validate().map_err(|_| Refusal::ShapeMismatch(name))?;
    if plane.width == roi.width && plane.height == roi.height {
        Ok(())
    } else {
        Err(Refusal::ShapeMismatch(name))
    }
}

fn require_origin_plane(
    source: &Image,
    plane: &Image,
    [x, y]: [u32; 2],
    name: &'static str,
) -> Result<(), Refusal> {
    plane.validate().map_err(|_| Refusal::ShapeMismatch(name))?;
    let right = u64::from(x) + u64::from(plane.width);
    let bottom = u64::from(y) + u64::from(plane.height);
    if x < source.width
        && y < source.height
        && right <= u64::from(source.width)
        && bottom <= u64::from(source.height)
    {
        Ok(())
    } else {
        Err(Refusal::ValueOutOfRange(name))
    }
}

fn repair_output_plane_sha256(output: &RepairOutput) -> Vec<RepairPlaneSha256> {
    match output {
        RepairOutput::DarkCircle { residual_rgb } => {
            vec![f32_plane_sha256("PF5_RESIDUAL_RGB", residual_rgb)]
        }
        RepairOutput::ClosedMouthSmile { model, .. } => {
            vec![image_plane_sha256("PF21076_MODEL_RGB", model)]
        }
        RepairOutput::Sweat { model } => vec![image_plane_sha256("PF30259_MODEL_RGB", model)],
        RepairOutput::NeckLine { repair, .. } => {
            vec![image_plane_sha256("PF90015_REPAIR_RGB", repair)]
        }
        RepairOutput::DoubleChin { field, .. } => {
            vec![vec2_plane_sha256("PF90034_SHIFT_FIELD", field)]
        }
        RepairOutput::OilShine { effect_rgb, mask } => vec![
            byte_plane_sha256("PF90080_EFFECT_RGB", effect_rgb),
            byte_plane_sha256("PF90080_MASK", mask),
        ],
        RepairOutput::BodyBlemishChild {
            model, roi_mask, ..
        } => vec![
            image_plane_sha256("PF90089_MODEL_RGB", model),
            byte_plane_sha256("PF90089_ROI_MASK", roi_mask),
        ],
        RepairOutput::FootBlueVein {
            repair, model_mask, ..
        } => vec![
            image_plane_sha256("PF44297_MODEL_REPAIR", repair),
            byte_plane_sha256("PF44297_MODEL_MASK", model_mask),
        ],
        RepairOutput::LipLine {
            repair,
            mask_a,
            mask_b,
            ..
        } => vec![
            image_plane_sha256("PF90118_REPAIR_RGB", repair),
            byte_plane_sha256("PF90118_MASK_A", mask_a),
            byte_plane_sha256("PF90118_MASK_B", mask_b),
        ],
        RepairOutput::WhiteHair { repair, raw_mask } => vec![
            image_plane_sha256("PF90131_REPAIR_RGB", repair),
            byte_plane_sha256("PF90131_RAW_HAIR_MASK", raw_mask),
        ],
    }
}

fn byte_plane_sha256(name: &'static str, data: &[u8]) -> RepairPlaneSha256 {
    RepairPlaneSha256 {
        name,
        sha256: sha256_32(data),
    }
}

fn f32_plane_sha256(name: &'static str, data: &[f32]) -> RepairPlaneSha256 {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for value in data {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    byte_plane_sha256(name, &bytes)
}

fn vec2_plane_sha256(name: &'static str, data: &[[f32; 2]]) -> RepairPlaneSha256 {
    let mut bytes = Vec::with_capacity(data.len() * 8);
    for point in data {
        bytes.extend_from_slice(&point[0].to_le_bytes());
        bytes.extend_from_slice(&point[1].to_le_bytes());
    }
    byte_plane_sha256(name, &bytes)
}

fn image_plane_sha256(name: &'static str, image: &Image) -> RepairPlaneSha256 {
    let mut bytes = Vec::with_capacity(8 + image.pixels.len());
    bytes.extend_from_slice(&image.width.to_le_bytes());
    bytes.extend_from_slice(&image.height.to_le_bytes());
    bytes.extend_from_slice(&image.pixels);
    byte_plane_sha256(name, &bytes)
}

fn ints(hash: u64, values: impl IntoIterator<Item = u64>) -> u64 {
    values.into_iter().fold(hash, fnv_u64)
}

fn floats<'a>(hash: u64, values: impl Iterator<Item = &'a f32>) -> u64 {
    ints(hash, values.map(|value| u64::from(value.to_bits())))
}

fn bytes(hash: u64, data: &[u8]) -> u64 {
    fnv1a64(fnv_u64(hash, data.len() as u64), data)
}

fn image_hash(hash: u64, image: &Image) -> u64 {
    bytes(
        ints(hash, [u64::from(image.width), u64::from(image.height)]),
        &image.pixels,
    )
}

pub(super) const PFS: &[u32] = &[
    5, 21076, 30259, 90015, 90034, 90080, 90089, 44297, 90118, 90131,
];

pub(super) fn apply(
    source: &Image,
    pf: u32,
    degree: f32,
    job_id: Option<u64>,
    inputs: Option<&RepairProducerInputs>,
) -> Result<Option<Image>, Refusal> {
    source.validate()?;
    check_degree01(degree, "repair UI")?;
    check_repair_source_binding(source, pf, job_id, inputs)?;
    if degree == 0.0 || (pf == 90131 && degree <= 0.001) {
        return Ok(None);
    }
    let bound = inputs
        .and_then(|inputs| inputs.0.get(&pf))
        .ok_or(Refusal::MissingInput(match pf {
            5 => "PF5_SAME_FRAME_OUTPUT",
            21076 => "PF21076_SAME_FRAME_OUTPUT",
            30259 => "PF30259_SAME_FRAME_OUTPUT",
            90015 => "PF90015_SAME_FRAME_OUTPUT",
            90034 => "PF90034_SAME_FRAME_OUTPUT",
            90080 => "PF90080_SAME_FRAME_OUTPUT",
            90089 => "PF90089_SAME_FRAME_OUTPUT",
            44297 => "PF44297_SAME_FRAME_OUTPUT",
            90118 => "PF90118_SAME_FRAME_OUTPUT",
            90131 => "PF90131_SAME_FRAME_OUTPUT",
            _ => "repair same-frame output",
        }))?;

    // Model-side degrees are already reflected in the prepared outputs.
    match &bound.output {
        RepairOutput::DarkCircle { residual_rgb } => {
            eye_bag::compose_image_pf5(source, residual_rgb, eye_bag::BlendMode::Vendor)
        }
        RepairOutput::ClosedMouthSmile {
            model,
            paste_enabled,
        } => {
            if !paste_enabled {
                return Ok(None);
            }
            close_smile::apply(source, model, *paste_enabled)
        }
        RepairOutput::Sweat { model } => sweat::apply(source, model),
        RepairOutput::NeckLine {
            repair,
            origin: [x, y],
        } => neck_paste::paste(source, repair, *x, *y),
        RepairOutput::DoubleChin { field, mode, scale } => {
            double_chin::warp(source, field, *mode, *scale)
        }
        RepairOutput::OilShine { effect_rgb, mask } => {
            skin_color::oil_shine_mask_blend(source, effect_rgb, mask, degree)
        }
        RepairOutput::BodyBlemishChild {
            model,
            roi,
            roi_mask,
            native_per_face_degree,
            alternate,
        } => {
            let (_, active) =
                skin_adjust::body_blemish_degree(degree * 100.0, Some(*native_per_face_degree))?;
            if !active {
                return Ok(None);
            }
            skin_adjust::masked_model_output(source, model, *roi, roi_mask, *alternate)
        }
        RepairOutput::FootBlueVein {
            repair,
            model_mask,
            feather_radius,
        } => hair_body::foot_blue_vein(source, degree * 100.0, model_mask, repair, *feather_radius),
        RepairOutput::LipLine {
            repair,
            origin,
            mask_a,
            mask_b,
        } => lip_line(source, repair, *origin, mask_a, mask_b),
        RepairOutput::WhiteHair { repair, raw_mask } => {
            hair_body::white_hair(source, degree * 100.0, raw_mask, repair)
        }
    }
    .map(Some)
}

fn check_repair_source_binding(
    source: &Image,
    pf: u32,
    job_id: Option<u64>,
    inputs: Option<&RepairProducerInputs>,
) -> Result<(), Refusal> {
    let Some(bound) = inputs.and_then(|inputs| inputs.0.get(&pf)) else {
        return Ok(());
    };
    if bound.ticket == 0 || job_id != Some(bound.ticket) {
        return Err(Refusal::ValueOutOfRange("repair job ticket"));
    }
    if bound.source_sha256 != sha256_32(&source.pixels) {
        return Err(Refusal::ValueOutOfRange("repair source frame SHA256"));
    }
    Ok(())
}

fn lip_line(
    source: &Image,
    repair: &Image,
    [x, y]: [u32; 2],
    mask_a: &[u8],
    mask_b: &[u8],
) -> Result<Image, Refusal> {
    repair.validate()?;
    let right = x
        .checked_add(repair.width)
        .filter(|&end| end <= source.width)
        .ok_or(Refusal::ValueOutOfRange("lip ROI"))?;
    let bottom = y
        .checked_add(repair.height)
        .filter(|&end| end <= source.height)
        .ok_or(Refusal::ValueOutOfRange("lip ROI"))?;
    let mut pixels = Vec::with_capacity(repair.pixels.len());
    for row in y..bottom {
        let offset = row as usize * source.width as usize;
        pixels.extend_from_slice(
            &source.pixels[(offset + x as usize) * 4..(offset + right as usize) * 4],
        );
    }
    let tile = Image {
        width: repair.width,
        height: repair.height,
        pixels,
    };
    let composed = skin_color::shared_min_mask_composite(&tile, repair, mask_a, mask_b)?;
    neck_paste::paste(source, &composed, x, y)
}
