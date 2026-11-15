//! PF90002 face / PF90010 body moisturizing on NVIDIA.
//!
//! Same law as `kernels::moisture`: YUV forward, `beauty` curve on Y, the
//! face-only `dog` term from the blurred Y and the hardlight LUT, then the two
//! mixes and the inverse YUV. Alpha is copied. Inputs are explicit buffers:
//! source, mask, weights, skin map and LUTs.
//!
//! Driver and the 222-only PTX retry live in [`crate::cuda`]; both stages go
//! through the shared `run_u8` path.

use crate::cuda::{CudaError, Dispatch};
use crate::kernels::moisture::{FaceParams, Lut1D, MaskChannel, SkinMap, BODY_RENDER_GATE};
use crate::{Image, Refusal};

/// Recovered kernel entry points.
pub const FACE_KERNEL: &str = "moisture_face_kernel";
pub const BODY_KERNEL: &str = "moisture_body_kernel";
const BLOCK: u32 = 16;

const PTX: &str = include_str!("../native/cuda/moisture.ptx");

/// 48-byte face block, matching `native/cuda/moisture.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FaceUniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub channel: u32,
    pub degree: f32,
    pub skin_width: u32,
    pub skin_height: u32,
    pub hardlight_len: u32,
    pub beauty_len: u32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u32,
}

/// 48-byte body block, matching `native/cuda/moisture.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BodyUniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub beauty_len: u32,
    pub degree: f32,
    pub roi_x: u32,
    pub roi_y: u32,
    pub roi_w: u32,
    pub roi_h: u32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u32,
}

const _: () = assert!(std::mem::size_of::<FaceUniforms>() == 48);
const _: () = assert!(std::mem::size_of::<BodyUniforms>() == 48);

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>())
    }
}

fn f32_bytes(values: &[f32]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values)) }
}

fn channel_index(channel: MaskChannel) -> u32 {
    match channel {
        MaskChannel::R => 0,
        MaskChannel::G => 1,
        MaskChannel::B => 2,
        MaskChannel::A => 3,
        MaskChannel::RG => 4,
        MaskChannel::RB => 5,
        MaskChannel::RA => 6,
        MaskChannel::GB => 7,
        MaskChannel::GA => 8,
        MaskChannel::BA => 9,
        MaskChannel::RGB => 10,
        MaskChannel::RGA => 11,
        MaskChannel::RBA => 12,
        MaskChannel::GBA => 13,
        MaskChannel::RGBA => 14,
    }
}

fn check_lut(samples: &[f32]) -> Result<(), Refusal> {
    if samples.is_empty() || samples.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ShapeMismatch("lut1d"));
    }
    Ok(())
}

fn check_skin_map(skin_map: &SkinMap<'_>) -> Result<(), Refusal> {
    if skin_map.width == 0
        || skin_map.height == 0
        || skin_map.samples.len() != skin_map.width * skin_map.height
    {
        return Err(Refusal::ShapeMismatch("skin_map"));
    }
    if skin_map.samples.iter().any(|value| !value.is_finite()) {
        return Err(Refusal::ValueOutOfRange("skin_map"));
    }
    Ok(())
}

fn check_roi(width: u32, height: u32, roi: (u32, u32, u32, u32)) -> Result<(), Refusal> {
    let (x, y, w, h) = roi;
    if w == 0 || h == 0 || x > width || y > height || w > width - x || h > height - y {
        return Err(Refusal::ValueOutOfRange("roi"));
    }
    Ok(())
}

fn grid(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(BLOCK).max(1), height.div_ceil(BLOCK).max(1))
}

#[cfg(windows)]
pub struct Moisture {
    gpu: crate::cuda::Nvidia,
}

#[cfg(windows)]
impl Moisture {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self { gpu: crate::cuda::Nvidia::open_ptx(PTX)? })
    }

    /// GPU twin of `moisture::face_apply`.
    pub fn face_apply(
        &self,
        source: &Image,
        mask_rgba: &[u8],
        blur_y: &[u8],
        params: &FaceParams,
        channel: MaskChannel,
        skin_map: &SkinMap<'_>,
        hardlight: &Lut1D<'_>,
        beauty: &Lut1D<'_>,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        source.validate().map_err(|_| CudaError::Refused(Refusal::ShapeMismatch("image")))?;
        let pixels = source.pixel_count();
        if mask_rgba.len() != pixels * 4 {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("mask_rgba")));
        }
        if blur_y.len() != pixels {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("blur_y")));
        }
        if !params.ui.is_finite() || !(0.0..=100.0).contains(&params.ui) {
            return Err(CudaError::Refused(Refusal::ValueOutOfRange("ui")));
        }
        if !params.pivot.is_finite() || !params.gain.is_finite() {
            return Err(CudaError::Refused(Refusal::ValueOutOfRange("pivot/gain")));
        }
        check_skin_map(skin_map)?;
        check_lut(hardlight.samples)?;
        check_lut(beauty.samples)?;
        if params.ui <= 0.0 {
            return Ok((source.clone(), Vec::new()));
        }
        let degree = (params.ui / 100.0 - params.pivot) * params.gain + params.pivot;
        let uniforms = FaceUniforms {
            width: source.width,
            height: source.height,
            stride: source.width * 4,
            channel: channel_index(channel),
            degree,
            skin_width: skin_map.width as u32,
            skin_height: skin_map.height as u32,
            hardlight_len: hardlight.samples.len() as u32,
            beauty_len: beauty.samples.len() as u32,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
        };
        let (pixels_out, dispatch) = self.gpu.run_u8_inplace(
            FACE_KERNEL,
            grid(source.width, source.height),
            &[
                &source.pixels,
                mask_rgba,
                blur_y,
                f32_bytes(skin_map.samples),
                f32_bytes(hardlight.samples),
                f32_bytes(beauty.samples),
            ],
            as_bytes(&uniforms),
        )?;
        let image = Image::new(source.width, source.height, pixels_out)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, vec![dispatch]))
    }

    /// GPU twin of `moisture::body_apply`.
    pub fn body_apply(
        &self,
        source: &Image,
        weight_skin: &[f32],
        weight_dog: &[f32],
        beauty: &Lut1D<'_>,
        degree: f32,
        roi: (u32, u32, u32, u32),
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        source.validate().map_err(|_| CudaError::Refused(Refusal::ShapeMismatch("image")))?;
        let pixels = source.pixel_count();
        if weight_skin.len() != pixels || weight_dog.len() != pixels {
            return Err(CudaError::Refused(Refusal::ShapeMismatch("weights")));
        }
        if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
            return Err(CudaError::Refused(Refusal::ValueOutOfRange("degree")));
        }
        let (x0, y0, w, h) = roi;
        check_roi(source.width, source.height, roi)?;
        check_lut(beauty.samples)?;
        if degree < BODY_RENDER_GATE {
            return Ok((source.clone(), Vec::new()));
        }
        let uniforms = BodyUniforms {
            width: source.width,
            height: source.height,
            stride: source.width * 4,
            beauty_len: beauty.samples.len() as u32,
            degree,
            roi_x: x0,
            roi_y: y0,
            roi_w: w,
            roi_h: h,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
        };
        let (pixels_out, dispatch) = self.gpu.run_u8_inplace(
            BODY_KERNEL,
            grid(w, h),
            &[
                &source.pixels,
                f32_bytes(weight_skin),
                f32_bytes(weight_dog),
                f32_bytes(beauty.samples),
            ],
            as_bytes(&uniforms),
        )?;
        let image = Image::new(source.width, source.height, pixels_out)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, vec![dispatch]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_blocks_match_the_c_header() {
        assert_eq!(std::mem::size_of::<FaceUniforms>(), 48);
        assert_eq!(std::mem::size_of::<BodyUniforms>(), 48);
        assert_eq!(std::mem::align_of::<FaceUniforms>(), 4);
        assert_eq!(channel_index(MaskChannel::R), 0);
        assert_eq!(channel_index(MaskChannel::BA), 9);
        assert_eq!(channel_index(MaskChannel::RGBA), 14);
    }

    #[test]
    fn lut_and_skin_map_refusals_match_the_cpu_lane() {
        assert_eq!(check_lut(&[]), Err(Refusal::ShapeMismatch("lut1d")));
        assert_eq!(check_lut(&[0.0, f32::NAN]), Err(Refusal::ShapeMismatch("lut1d")));
        let map = SkinMap { width: 2, height: 2, samples: &[0.0; 3] };
        assert_eq!(check_skin_map(&map), Err(Refusal::ShapeMismatch("skin_map")));
    }

    #[test]
    fn roi_bounds_are_checked_without_overflow() {
        assert_eq!(check_roi(37, 19, (5, 3, 11, 7)), Ok(()));
        assert_eq!(check_roi(37, 19, (27, 0, 11, 1)), Err(Refusal::ValueOutOfRange("roi")));
        assert_eq!(check_roi(37, 19, (u32::MAX, 0, 2, 1)), Err(Refusal::ValueOutOfRange("roi")));
        assert_eq!(check_roi(37, 19, (0, 0, 0, 1)), Err(Refusal::ValueOutOfRange("roi")));
    }
}
