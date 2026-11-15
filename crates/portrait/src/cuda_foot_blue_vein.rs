//! PF44297 foot blue-vein final composite on NVIDIA.
//!
//! This is the recovered local final stage, not the vendor `FootVeinsPipeline`
//! producer. A job must name the same-job model producer and supply its tight
//! `PF44297_MODEL_MASK` and `PF44297_MODEL_REPAIR` planes explicitly. The
//! adapter feathers the mask with the CPU oracle's exact chamfer pass, then
//! launches one CUDA blend over the full frame.
//!
//! ROI, degree and rounding stay explicit:
//! - only `(0, 0, width, height)` is accepted; the vendor foot ROI/affine is
//!   still unresolved and is never guessed from a hand or body mask;
//! - `degree = ui / 100`, and a zero degree returns the source without dispatch;
//! - `w = feathered/255 * degree`; RGB uses midpoint-away-from-zero rounding;
//! - alpha is copied from the source.

use crate::cuda::{CudaError, Dispatch};
use crate::kernels::hair_body;
use crate::{Image, Refusal};

pub const PF: u32 = 44297;
pub const KERNEL: &str = "foot_blue_vein_blend_kernel";
const BLOCK: u32 = 16;
const PTX: &str = include_str!("../native/cuda/foot_blue_vein.ptx");

/// Capability token for one exact PF44297 same-job model result.
///
/// The fields are private so a caller cannot fabricate a second producer shape.
/// Passing this token is an assertion by the host that the mask and repair came
/// from the same `FootVeinsPipeline` request; it is not producer discovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProducerReceipt(());

pub const PF44297_PRODUCER: ProducerReceipt = ProducerReceipt(());

/// Source-pixel ROI. The current recovered consumer is full-frame only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Roi {
    pub const fn full(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }
}

/// All live-job inputs for the PF44297 local final stage.
pub struct Job<'a> {
    pub source: &'a Image,
    pub producer: Option<ProducerReceipt>,
    pub model_mask: Option<&'a [u8]>,
    pub model_repair: Option<&'a Image>,
    pub ui: f32,
    pub feather_radius: u32,
    pub roi: Roi,
}

/// 32-byte argument block, matching `native/cuda/foot_blue_vein.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub degree: f32,
    pub roi_x: u32,
    pub roi_y: u32,
    pub roi_w: u32,
    pub roi_h: u32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 32);

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

/// Validates before any device upload. Returns the normalized degree.
pub fn validate_job(job: &Job<'_>) -> Result<f32, Refusal> {
    job.source
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("PF44297_SOURCE"))?;
    let full = Roi::full(job.source.width, job.source.height);
    if job.roi != full {
        return Err(Refusal::ShapeMismatch("PF44297_ROI"));
    }
    if job.producer.is_none() {
        return Err(Refusal::MissingInput("PF44297_MODEL_PRODUCER"));
    }
    let Some(mask) = job.model_mask else {
        return Err(Refusal::MissingInput("PF44297_MODEL_MASK"));
    };
    if mask.len() != job.source.pixel_count() {
        return Err(Refusal::ShapeMismatch("PF44297_MODEL_MASK"));
    }
    let Some(repair) = job.model_repair else {
        return Err(Refusal::MissingInput("PF44297_MODEL_REPAIR"));
    };
    repair
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("PF44297_MODEL_REPAIR"))?;
    if (repair.width, repair.height) != (job.source.width, job.source.height) {
        return Err(Refusal::ShapeMismatch("PF44297_MODEL_REPAIR"));
    }
    if !job.ui.is_finite() || !(0.0..=100.0).contains(&job.ui) {
        return Err(Refusal::ValueOutOfRange("PF44297_UI"));
    }
    if job.feather_radius < 1 {
        return Err(Refusal::ValueOutOfRange("PF44297_FEATHER_RADIUS"));
    }
    job.source
        .width
        .checked_mul(4)
        .ok_or(Refusal::ShapeMismatch("PF44297_SOURCE"))?;
    Ok(job.ui / 100.0)
}

#[cfg(windows)]
pub struct FootBlueVein {
    gpu: crate::cuda::Nvidia,
}

#[cfg(windows)]
impl FootBlueVein {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self {
            gpu: crate::cuda::Nvidia::open_ptx(PTX)?,
        })
    }

    pub fn apply(&self, job: &Job<'_>) -> Result<(Image, Vec<Dispatch>), CudaError> {
        let degree = validate_job(job).map_err(CudaError::Refused)?;
        if degree <= 0.0 {
            return Ok((job.source.clone(), Vec::new()));
        }

        let mask = job.model_mask.expect("validated model mask");
        let repair = job.model_repair.expect("validated model repair");
        let feathered = hair_body::feather_distance(
            mask,
            job.source.width as usize,
            job.source.height as usize,
            job.feather_radius,
        );
        let uniforms = Uniforms {
            width: job.source.width,
            height: job.source.height,
            stride: job.source.width * 4,
            degree,
            roi_x: job.roi.x,
            roi_y: job.roi.y,
            roi_w: job.roi.width,
            roi_h: job.roi.height,
        };
        let grid = (
            job.source.width.div_ceil(BLOCK).max(1),
            job.source.height.div_ceil(BLOCK).max(1),
        );
        let (pixels, dispatch) = self.gpu.run_u8_with_init(
            KERNEL,
            grid,
            &[&job.source.pixels, &repair.pixels, &feathered],
            as_bytes(&uniforms),
            Some(0),
        )?;
        let image = Image::new(job.source.width, job.source.height, pixels)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, vec![dispatch]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(pixel: [u8; 4]) -> Image {
        Image::new(1, 1, pixel.to_vec()).unwrap()
    }

    fn job<'a>(source: &'a Image, mask: &'a [u8], repair: &'a Image) -> Job<'a> {
        Job {
            source,
            producer: Some(PF44297_PRODUCER),
            model_mask: Some(mask),
            model_repair: Some(repair),
            ui: 50.0,
            feather_radius: 1,
            roi: Roi::full(1, 1),
        }
    }

    #[test]
    fn missing_producer_mask_and_repair_are_typed_refusals() {
        let source = image([10, 20, 30, 40]);
        let repair = image([100, 110, 120, 0]);
        let mask = [255];

        let mut missing = job(&source, &mask, &repair);
        missing.producer = None;
        assert_eq!(
            validate_job(&missing),
            Err(Refusal::MissingInput("PF44297_MODEL_PRODUCER"))
        );

        let mut missing = job(&source, &mask, &repair);
        missing.model_mask = None;
        assert_eq!(
            validate_job(&missing),
            Err(Refusal::MissingInput("PF44297_MODEL_MASK"))
        );

        let mut missing = job(&source, &mask, &repair);
        missing.model_repair = None;
        assert_eq!(
            validate_job(&missing),
            Err(Refusal::MissingInput("PF44297_MODEL_REPAIR"))
        );
    }

    #[test]
    fn roi_mask_repair_degree_and_radius_are_explicit() {
        let source = Image::solid(3, 3, [10, 20, 30, 40]);
        let repair = Image::solid(3, 3, [100, 110, 120, 0]);
        let mask = vec![255; 9];
        let mut input = Job {
            source: &source,
            producer: Some(PF44297_PRODUCER),
            model_mask: Some(&mask),
            model_repair: Some(&repair),
            ui: 100.0,
            feather_radius: 1,
            roi: Roi {
                x: 1,
                y: 1,
                width: 2,
                height: 2,
            },
        };
        assert_eq!(
            validate_job(&input),
            Err(Refusal::ShapeMismatch("PF44297_ROI"))
        );

        input.roi = Roi::full(3, 3);
        input.model_mask = Some(&mask[..8]);
        assert_eq!(
            validate_job(&input),
            Err(Refusal::ShapeMismatch("PF44297_MODEL_MASK"))
        );

        input.model_mask = Some(&mask);
        let wrong_repair = Image::solid(2, 3, [0, 0, 0, 0]);
        input.model_repair = Some(&wrong_repair);
        assert_eq!(
            validate_job(&input),
            Err(Refusal::ShapeMismatch("PF44297_MODEL_REPAIR"))
        );

        input.model_repair = Some(&repair);
        input.ui = 100.5;
        assert_eq!(
            validate_job(&input),
            Err(Refusal::ValueOutOfRange("PF44297_UI"))
        );

        input.ui = 100.0;
        input.feather_radius = 0;
        assert_eq!(
            validate_job(&input),
            Err(Refusal::ValueOutOfRange("PF44297_FEATHER_RADIUS"))
        );
    }
}
