use super::{FNV1A64_OFFSET_BASIS, check_degree01, fnv_u64, fnv1a64};
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

#[derive(Clone, Debug, PartialEq)]
struct BoundOutput {
    job_id: u64,
    frame_digest: u64,
    output: RepairOutput,
}

/// Model output bindings carried in the renderer's producer bundle.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RepairProducerInputs(BTreeMap<u32, BoundOutput>);

impl RepairProducerInputs {
    /// Bind a result to the exact input pixels consumed by its producer.
    /// If an earlier effect changes the frame, bind to that effect's output.
    pub fn insert(
        &mut self,
        job_id: u64,
        source: &Image,
        output: RepairOutput,
    ) -> Result<(), Refusal> {
        source.validate()?;
        self.0.insert(
            output.pf(),
            BoundOutput {
                job_id,
                frame_digest: image_hash(FNV1A64_OFFSET_BASIS, source),
                output,
            },
        );
        Ok(())
    }

    pub(super) fn fingerprint(&self, mut hash: u64) -> u64 {
        if self.0.is_empty() {
            return hash;
        }
        hash = fnv_u64(fnv1a64(hash, b"repair outputs"), self.0.len() as u64);
        for (&pf, bound) in &self.0 {
            hash = ints(hash, [u64::from(pf), bound.job_id, bound.frame_digest]);
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
    check_degree01(degree, "repair UI")?;
    if degree == 0.0 || (pf == 90131 && degree <= 0.001) {
        return Ok(None);
    }
    let bound = inputs
        .and_then(|inputs| inputs.0.get(&pf))
        .ok_or(Refusal::MissingInput("same-frame repair output"))?;
    if job_id != Some(bound.job_id) {
        return Err(Refusal::ValueOutOfRange("repair job ticket"));
    }
    if bound.frame_digest != image_hash(FNV1A64_OFFSET_BASIS, source) {
        return Err(Refusal::ValueOutOfRange("repair source frame"));
    }

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
