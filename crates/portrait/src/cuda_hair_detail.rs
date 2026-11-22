//! PF90139/PF90140 hair smooth and PF90160 hair gloss on NVIDIA.
//!
//! The vendor path is CPU, not a shader. This slice moves only the recovered
//! consumer law to CUDA. The same-job producer is not implemented here: when a
//! non-identity strength needs live planes and the producer is absent, the
//! adapter returns [`Refusal::MissingInput`] before any launch.
//!
//! Latest slot evidence: PF90140 maps to HairSmooth descriptor `+0x80`
//! (`mix_strength`), PF90139 maps to `+0x88` (`detail_strength`).

#![cfg_attr(not(windows), allow(dead_code))]

use crate::{Image, Refusal};

#[cfg(windows)]
use crate::cuda::{CudaError, Dispatch, Nvidia};

pub const SMOOTH_KERNEL: &str = "pf90139_hair_smooth_kernel";
pub const GLOSS_KERNEL: &str = "pf90160_hair_gloss_kernel";
const BLOCK: u32 = 16;
const PTX: &str = include_str!("../native/cuda/hair_detail.ptx");

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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothParams {
    /// PF90140 `hairSmooth_strand`, normalized before descriptor `+0x80`.
    pub mix_strength: f32,
    /// PF90139 `hairSmooth`, normalized before descriptor `+0x88`.
    pub detail_strength: f32,
}

#[derive(Clone, Copy)]
pub struct SmoothProducer<'a> {
    /// Nonzero identity shared by every live job input.
    pub ticket: u64,
    pub hair_mask: &'a [u8],
    pub mix_mask: &'a [u8],
    /// Vendor producer output is packed RGB8, never RGBA8.
    pub blend_rgb: &'a [u8],
    pub detail: &'a [u8],
    pub hair_stride: u32,
    pub mix_stride: u32,
    pub blend_stride: u32,
    pub detail_stride: u32,
}

#[derive(Clone, Copy)]
pub struct GlossProducer<'a> {
    pub ticket: u64,
    pub hair_mask: &'a [u8],
    pub model_detail: &'a [u8],
    pub hair_stride: u32,
    pub detail_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SmoothUniforms {
    pub width: u32,
    pub height: u32,
    pub roi_x: u32,
    pub roi_y: u32,
    pub roi_width: u32,
    pub roi_height: u32,
    pub source_stride: u32,
    pub hair_stride: u32,
    pub mix_stride: u32,
    pub blend_stride: u32,
    pub detail_stride: u32,
    pub mix_strength: f32,
    pub detail_strength: f32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlossUniforms {
    pub width: u32,
    pub height: u32,
    pub roi_x: u32,
    pub roi_y: u32,
    pub roi_width: u32,
    pub roi_height: u32,
    pub source_stride: u32,
    pub hair_stride: u32,
    pub detail_stride: u32,
    pub scale: f32,
    pub reserved0: u32,
    pub reserved1: u32,
}

const _: () = assert!(std::mem::size_of::<SmoothUniforms>() == 64);
const _: () = assert!(std::mem::size_of::<GlossUniforms>() == 48);

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn source_stride(source: &Image) -> Result<u32, Refusal> {
    source
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("source"))?;
    source
        .width
        .checked_mul(4)
        .ok_or(Refusal::ShapeMismatch("source"))
}

fn check_roi(source: &Image, roi: Roi) -> Result<(), Refusal> {
    if roi.width == 0 || roi.height == 0 {
        return Err(Refusal::ShapeMismatch("roi"));
    }
    if u64::from(roi.x) + u64::from(roi.width) > u64::from(source.width)
        || u64::from(roi.y) + u64::from(roi.height) > u64::from(source.height)
    {
        return Err(Refusal::ValueOutOfRange("roi"));
    }
    Ok(())
}

fn check_plane(
    bytes: &[u8],
    width: u32,
    height: u32,
    stride: u32,
    bytes_per_pixel: u32,
    name: &'static str,
) -> Result<(), Refusal> {
    let row = width
        .checked_mul(bytes_per_pixel)
        .ok_or(Refusal::ShapeMismatch(name))?;
    if stride < row {
        return Err(Refusal::ShapeMismatch(name));
    }
    let required = height
        .checked_sub(1)
        .and_then(|last| last.checked_mul(stride))
        .and_then(|head| head.checked_add(row))
        .ok_or(Refusal::ShapeMismatch(name))?;
    if bytes.len() < required as usize {
        return Err(Refusal::ShapeMismatch(name));
    }
    Ok(())
}

fn normalized(value: f32, name: &'static str) -> Result<f32, Refusal> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(Refusal::ValueOutOfRange(name))
    }
}

fn check_smooth<'a>(
    source: &Image,
    roi: Roi,
    params: SmoothParams,
    producer: Option<SmoothProducer<'a>>,
) -> Result<(SmoothProducer<'a>, SmoothUniforms), Refusal> {
    let stride = source_stride(source)?;
    check_roi(source, roi)?;
    let mix_strength = normalized(params.mix_strength, "PF90140_MIX_STRENGTH")?;
    let detail_strength = normalized(params.detail_strength, "PF90139_DETAIL_STRENGTH")?;
    let producer = producer.ok_or(Refusal::MissingInput("PF90139_PF90140_SAME_JOB_PRODUCER"))?;
    if producer.ticket == 0 {
        return Err(Refusal::MissingInput("PF90139_PF90140_SAME_JOB_RECEIPT"));
    }
    check_plane(
        producer.hair_mask,
        roi.width,
        roi.height,
        producer.hair_stride,
        1,
        "PF90139_HAIR_MASK",
    )?;
    check_plane(
        producer.mix_mask,
        roi.width,
        roi.height,
        producer.mix_stride,
        1,
        "PF90139_MIX_MASK",
    )?;
    check_plane(
        producer.blend_rgb,
        roi.width,
        roi.height,
        producer.blend_stride,
        3,
        "PF90139_BLEND_SOURCE_RGB8",
    )?;
    check_plane(
        producer.detail,
        roi.width,
        roi.height,
        producer.detail_stride,
        1,
        "PF90139_DETAIL",
    )?;
    let uniforms = SmoothUniforms {
        width: source.width,
        height: source.height,
        roi_x: roi.x,
        roi_y: roi.y,
        roi_width: roi.width,
        roi_height: roi.height,
        source_stride: stride,
        hair_stride: producer.hair_stride,
        mix_stride: producer.mix_stride,
        blend_stride: producer.blend_stride,
        detail_stride: producer.detail_stride,
        mix_strength,
        detail_strength,
        reserved0: 0,
        reserved1: 0,
        reserved2: 0,
    };
    Ok((producer, uniforms))
}

fn check_gloss<'a>(
    source: &Image,
    roi: Roi,
    scale: f32,
    producer: Option<GlossProducer<'a>>,
) -> Result<(GlossProducer<'a>, GlossUniforms), Refusal> {
    let stride = source_stride(source)?;
    check_roi(source, roi)?;
    let scale = normalized(scale, "PF90160_DYNAMIC_SCALE")?;
    let producer = producer.ok_or(Refusal::MissingInput("PF90160_SAME_JOB_PRODUCER"))?;
    if producer.ticket == 0 {
        return Err(Refusal::MissingInput("PF90160_SAME_JOB_RECEIPT"));
    }
    check_plane(
        producer.hair_mask,
        roi.width,
        roi.height,
        producer.hair_stride,
        1,
        "PF90160_HAIR_MASK",
    )?;
    check_plane(
        producer.model_detail,
        roi.width,
        roi.height,
        producer.detail_stride,
        1,
        "PF90160_MODEL_DETAIL",
    )?;
    let uniforms = GlossUniforms {
        width: source.width,
        height: source.height,
        roi_x: roi.x,
        roi_y: roi.y,
        roi_width: roi.width,
        roi_height: roi.height,
        source_stride: stride,
        hair_stride: producer.hair_stride,
        detail_stride: producer.detail_stride,
        scale,
        reserved0: 0,
        reserved1: 0,
    };
    Ok((producer, uniforms))
}

#[cfg(windows)]
pub struct HairDetail {
    gpu: Nvidia,
}

#[cfg(windows)]
impl HairDetail {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self {
            gpu: Nvidia::open_ptx(PTX)?,
        })
    }

    /// PF90139 effect composite. Both normalized strengths being zero is the
    /// only identity path; a zero mix strength with nonzero detail still runs.
    pub fn smooth(
        &self,
        source: &Image,
        roi: Roi,
        params: SmoothParams,
        producer: Option<SmoothProducer<'_>>,
    ) -> Result<(Image, Option<Dispatch>), CudaError> {
        source_stride(source)?;
        check_roi(source, roi)?;
        let mix_strength = normalized(params.mix_strength, "PF90140_MIX_STRENGTH")?;
        let detail_strength = normalized(params.detail_strength, "PF90139_DETAIL_STRENGTH")?;
        if mix_strength == 0.0 && detail_strength == 0.0 {
            return Ok((source.clone(), None));
        }
        let (producer, uniforms) = check_smooth(source, roi, params, producer)?;
        let grid = (
            roi.width.div_ceil(BLOCK).max(1),
            roi.height.div_ceil(BLOCK).max(1),
        );
        let (pixels, dispatch) = self.gpu.run_u8_with_init(
            SMOOTH_KERNEL,
            grid,
            &[
                &source.pixels,
                producer.hair_mask,
                producer.mix_mask,
                producer.blend_rgb,
                producer.detail,
            ],
            as_bytes(&uniforms),
            Some(0),
        )?;
        let image = Image::new(source.width, source.height, pixels)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, Some(dispatch)))
    }

    /// PF90160 additive detail. `scale` is the descriptor `+0x60` dynamic f32,
    /// already normalized by the producer. UI 0..100 to this value remains open.
    pub fn gloss(
        &self,
        source: &Image,
        roi: Roi,
        scale: f32,
        producer: Option<GlossProducer<'_>>,
    ) -> Result<(Image, Option<Dispatch>), CudaError> {
        source_stride(source)?;
        check_roi(source, roi)?;
        if normalized(scale, "PF90160_DYNAMIC_SCALE")? == 0.0 {
            return Ok((source.clone(), None));
        }
        let (producer, uniforms) = check_gloss(source, roi, scale, producer)?;
        let grid = (
            roi.width.div_ceil(BLOCK).max(1),
            roi.height.div_ceil(BLOCK).max(1),
        );
        let (pixels, dispatch) = self.gpu.run_u8_with_init(
            GLOSS_KERNEL,
            grid,
            &[&source.pixels, producer.hair_mask, producer.model_detail],
            as_bytes(&uniforms),
            Some(0),
        )?;
        let image = Image::new(source.width, source.height, pixels)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, Some(dispatch)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_matches_the_c_header() {
        assert_eq!(std::mem::size_of::<SmoothUniforms>(), 64);
        assert_eq!(std::mem::size_of::<GlossUniforms>(), 48);
        assert_eq!(std::mem::align_of::<SmoothUniforms>(), 4);
        assert_eq!(std::mem::align_of::<GlossUniforms>(), 4);
    }

    #[test]
    fn ptx_contains_both_entries_and_by_value_uniforms() {
        assert!(PTX.contains(&format!(".entry {SMOOTH_KERNEL}(")));
        assert!(PTX.contains(&format!(".entry {GLOSS_KERNEL}(")));
        assert!(PTX.contains("pf90139_hair_smooth_kernel_param_6[64]"));
        assert!(PTX.contains("pf90160_hair_gloss_kernel_param_4[48]"));
    }

    #[test]
    fn missing_producer_and_receipt_are_typed_refusals() {
        let source = Image::solid(2, 2, [10, 20, 30, 40]);
        let roi = Roi::full(2, 2);
        let params = SmoothParams {
            mix_strength: 0.5,
            detail_strength: 0.0,
        };
        assert_eq!(
            check_smooth(&source, roi, params, None).err(),
            Some(Refusal::MissingInput("PF90139_PF90140_SAME_JOB_PRODUCER"))
        );

        let planes = [255u8; 4];
        let blend = [1u8; 12];
        let producer = SmoothProducer {
            ticket: 0,
            hair_mask: &planes,
            mix_mask: &planes,
            blend_rgb: &blend,
            detail: &planes,
            hair_stride: 2,
            mix_stride: 2,
            blend_stride: 6,
            detail_stride: 2,
        };
        assert_eq!(
            check_smooth(&source, roi, params, Some(producer)).err(),
            Some(Refusal::MissingInput("PF90139_PF90140_SAME_JOB_RECEIPT"))
        );
    }

    #[test]
    fn zero_mix_is_not_identity_when_detail_is_active() {
        let source = Image::solid(2, 2, [10, 20, 30, 40]);
        let roi = Roi::full(2, 2);
        let mask = [255u8; 4];
        let blend = [10u8; 12];
        let detail = [255u8; 4];
        let producer = SmoothProducer {
            ticket: 1,
            hair_mask: &mask,
            mix_mask: &mask,
            blend_rgb: &blend,
            detail: &detail,
            hair_stride: 2,
            mix_stride: 2,
            blend_stride: 6,
            detail_stride: 2,
        };
        assert!(
            check_smooth(
                &source,
                roi,
                SmoothParams {
                    mix_strength: 0.0,
                    detail_strength: 0.5
                },
                Some(producer)
            )
            .is_ok()
        );
    }

    #[test]
    fn plane_stride_and_blend_rgb_are_checked() {
        let source = Image::solid(2, 2, [10, 20, 30, 40]);
        let roi = Roi::full(2, 2);
        let mask = [255u8; 4];
        let blend = [1u8; 12];
        let producer = SmoothProducer {
            ticket: 1,
            hair_mask: &mask,
            mix_mask: &mask,
            blend_rgb: &blend[..11],
            detail: &mask,
            hair_stride: 2,
            mix_stride: 2,
            blend_stride: 6,
            detail_stride: 2,
        };
        assert_eq!(
            check_smooth(
                &source,
                roi,
                SmoothParams {
                    mix_strength: 0.5,
                    detail_strength: 0.0
                },
                Some(producer)
            )
            .err(),
            Some(Refusal::ShapeMismatch("PF90139_BLEND_SOURCE_RGB8"))
        );
    }
}
