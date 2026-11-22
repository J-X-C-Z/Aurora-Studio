//! PF8069 negative FaceSurfaceBlur / PF8070 type136 clarity on NVIDIA.
//!
//! The CPU oracle remains `kernels::skin_models`. PF8069 preserves all four
//! RGBA8 intermediate quantisation points and the final type10 mix; PF8070
//! writes only the supplied ROI and keeps every other source byte.

#[cfg(windows)]
use crate::cuda::CudaError;
#[cfg(windows)]
use crate::cuda::Dispatch;
use crate::{Image, Refusal};

pub const SURFACE_KERNEL: &str = "pf8069_surface_kernel";
pub const GAUSS_KERNEL: &str = "pf8069_gauss_kernel";
pub const BLEND_KERNEL: &str = "pf8069_blend_kernel";
pub const CLARITY_KERNEL: &str = "pf8070_clarity_kernel";
pub const MISS_GATE: f32 = 0.01;
pub const SURFACE_SCALE: f32 = 4.93;
pub const GAUSS_SCALE: f32 = 1.45;
pub const SURFACE_THRESHOLD: f32 = 10.45;

const BLOCK: u32 = 16;
const PTX: &str = include_str!("../native/cuda/skin_models_8069_8070.ptx");

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FilterUniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub vertical: u32,
    pub step_u: f32,
    pub step_v: f32,
    pub hk: f32,
    pub reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlendUniforms {
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub mask_row_bytes: u32,
    pub degree: f32,
    pub reserved0: u32,
    pub reserved1: u32,
    pub reserved2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClarityUniforms {
    pub source_width: u32,
    pub stride: u32,
    pub roi_x: u32,
    pub roi_y: u32,
    pub roi_w: u32,
    pub roi_h: u32,
    pub base_width: u32,
    pub base_height: u32,
    pub degree: f32,
    pub threshold: f32,
    pub factor: f32,
    pub reserved: u32,
}

const _: () = assert!(std::mem::size_of::<FilterUniforms>() == 32);
const _: () = assert!(std::mem::size_of::<BlendUniforms>() == 32);
const _: () = assert!(std::mem::size_of::<ClarityUniforms>() == 48);

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn grid(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(BLOCK).max(1), height.div_ceil(BLOCK).max(1))
}

fn remap_piecewise(x: f32, e0: f32, e1: f32, e2: f32) -> f32 {
    if x <= 0.0 {
        e0
    } else if x < 0.5 {
        e0 + (x / 0.5) * (e1 - e0)
    } else if x <= 1.0 {
        e1 + ((x - 0.5) / 0.5) * (e2 - e1)
    } else {
        e2
    }
}

fn check_texture(
    source: &Image,
    mask_r: &[u8],
    extra_mask_r: &[u8],
    ui: f32,
) -> Result<(), Refusal> {
    source
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("source"))?;
    if !ui.is_finite() || !(-100.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("ui"));
    }
    if mask_r.len() != source.pixel_count() {
        return Err(Refusal::ShapeMismatch("mask_r"));
    }
    if extra_mask_r.len() != source.pixel_count() {
        return Err(Refusal::ShapeMismatch("extra_mask_r"));
    }
    if ui > 0.0 {
        return Err(Refusal::NotImplemented(8069));
    }
    Ok(())
}

fn check_clarity(
    source: &Image,
    base: &Image,
    blur: &Image,
    mask_r: &[u8],
    roi: (u32, u32, u32, u32),
    degree: f32,
    threshold: f32,
    factor: f32,
) -> Result<(), Refusal> {
    source
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("source"))?;
    let roi_pixels = base
        .validate()
        .map(|_| base.pixel_count())
        .map_err(|_| Refusal::ShapeMismatch("base"))?;
    blur.validate()
        .map_err(|_| Refusal::ShapeMismatch("blur"))?;
    if blur.pixel_count() != roi_pixels || (base.width, base.height) != (blur.width, blur.height) {
        return Err(Refusal::ShapeMismatch("blur"));
    }
    let (roi_x, roi_y, roi_w, roi_h) = roi;
    if (roi_w, roi_h) != (base.width, base.height) {
        return Err(Refusal::ShapeMismatch("roi"));
    }
    if u64::from(roi_x) + u64::from(roi_w) > u64::from(source.width)
        || u64::from(roi_y) + u64::from(roi_h) > u64::from(source.height)
    {
        return Err(Refusal::ValueOutOfRange("roi"));
    }
    if mask_r.len() != roi_pixels {
        return Err(Refusal::ShapeMismatch("mask_r"));
    }
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("degree"));
    }
    if !threshold.is_finite() || !factor.is_finite() {
        return Err(Refusal::ValueOutOfRange("filter"));
    }
    Ok(())
}

#[cfg(windows)]
pub struct SkinModels {
    gpu: crate::cuda::Nvidia,
}

#[cfg(windows)]
impl SkinModels {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self {
            gpu: crate::cuda::Nvidia::open_ptx(PTX)?,
        })
    }

    pub fn texture_apply(
        &self,
        source: &Image,
        mask_r: &[u8],
        extra_mask_r: &[u8],
        ui: f32,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        check_texture(source, mask_r, extra_mask_r, ui)?;
        if ui == 0.0 {
            return Ok((source.clone(), Vec::new()));
        }

        let degree = remap_piecewise((ui + 100.0) / 200.0, 0.49, 0.0, 0.0);
        let surface = [
            FilterUniforms {
                width: source.width,
                height: source.height,
                stride: source.width * 4,
                vertical: 0,
                step_u: SURFACE_SCALE / source.width as f32,
                step_v: 0.0,
                hk: SURFACE_THRESHOLD / 255.0,
                reserved: 0,
            },
            FilterUniforms {
                width: source.width,
                height: source.height,
                stride: source.width * 4,
                vertical: 1,
                step_u: 0.0,
                step_v: SURFACE_SCALE / source.height as f32,
                hk: SURFACE_THRESHOLD / 255.0,
                reserved: 0,
            },
        ];
        let gauss = [
            FilterUniforms {
                width: source.width,
                height: source.height,
                stride: source.width * 4,
                vertical: 0,
                step_u: GAUSS_SCALE / source.width as f32,
                step_v: 0.0,
                hk: 0.0,
                reserved: 0,
            },
            FilterUniforms {
                width: source.width,
                height: source.height,
                stride: source.width * 4,
                vertical: 1,
                step_u: 0.0,
                step_v: GAUSS_SCALE / source.height as f32,
                hk: 0.0,
                reserved: 0,
            },
        ];

        let mut current = source.pixels.clone();
        let mut dispatches = Vec::with_capacity(5);
        for uniforms in surface {
            let (next, dispatch) = self.gpu.run_u8(
                SURFACE_KERNEL,
                grid(source.width, source.height),
                &[&current],
                as_bytes(&uniforms),
            )?;
            current = next;
            dispatches.push(dispatch);
        }
        for uniforms in gauss {
            let (next, dispatch) = self.gpu.run_u8(
                GAUSS_KERNEL,
                grid(source.width, source.height),
                &[&current],
                as_bytes(&uniforms),
            )?;
            current = next;
            dispatches.push(dispatch);
        }

        let blend = BlendUniforms {
            width: source.width,
            height: source.height,
            stride: source.width * 4,
            mask_row_bytes: source.width,
            degree,
            reserved0: 0,
            reserved1: 0,
            reserved2: 0,
        };
        let (pixels, dispatch) = self.gpu.run_u8(
            BLEND_KERNEL,
            grid(source.width, source.height),
            &[&source.pixels, &current, mask_r, extra_mask_r],
            as_bytes(&blend),
        )?;
        dispatches.push(dispatch);
        let image = Image::new(source.width, source.height, pixels)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, dispatches))
    }

    pub fn clarity_apply(
        &self,
        source: &Image,
        base: &Image,
        blur: &Image,
        mask_r: &[u8],
        roi: (u32, u32, u32, u32),
        degree: f32,
        threshold: f32,
        factor: f32,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        check_clarity(source, base, blur, mask_r, roi, degree, threshold, factor)?;
        if degree == 0.0 {
            return Ok((source.clone(), Vec::new()));
        }
        let (roi_x, roi_y, roi_w, roi_h) = roi;
        let uniforms = ClarityUniforms {
            source_width: source.width,
            stride: source.width * 4,
            roi_x,
            roi_y,
            roi_w,
            roi_h,
            base_width: base.width,
            base_height: base.height,
            degree,
            threshold,
            factor,
            reserved: 0,
        };
        let (pixels, dispatch) = self.gpu.run_u8_with_init(
            CLARITY_KERNEL,
            grid(roi_w, roi_h),
            &[&source.pixels, &base.pixels, &blur.pixels, mask_r],
            as_bytes(&uniforms),
            Some(0),
        )?;
        let image = Image::new(source.width, source.height, pixels)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, vec![dispatch]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_blocks_match_the_c_header() {
        assert_eq!(std::mem::size_of::<FilterUniforms>(), 32);
        assert_eq!(std::mem::size_of::<BlendUniforms>(), 32);
        assert_eq!(std::mem::size_of::<ClarityUniforms>(), 48);
        assert_eq!(std::mem::align_of::<FilterUniforms>(), 4);
        assert_eq!(std::mem::align_of::<BlendUniforms>(), 4);
        assert_eq!(std::mem::align_of::<ClarityUniforms>(), 4);
    }

    #[test]
    fn ptx_contains_every_declared_entry() {
        for kernel in [SURFACE_KERNEL, GAUSS_KERNEL, BLEND_KERNEL, CLARITY_KERNEL] {
            assert!(PTX.contains(&format!(".entry {kernel}(")), "{kernel}");
        }
    }

    #[test]
    fn texture_checks_match_the_oracle() {
        let image = Image::solid(2, 1, [10, 20, 30, 40]);
        let mask = [255u8; 2];
        assert_eq!(
            check_texture(&image, &mask, &mask, 1.0),
            Err(Refusal::NotImplemented(8069))
        );
        assert_eq!(
            check_texture(&image, &mask[..1], &mask, -1.0),
            Err(Refusal::ShapeMismatch("mask_r"))
        );
        assert_eq!(
            check_texture(&image, &mask, &mask, f32::NAN),
            Err(Refusal::ValueOutOfRange("ui"))
        );
    }

    #[test]
    fn clarity_checks_match_the_oracle() {
        let source = Image::solid(2, 2, [100, 90, 80, 77]);
        let base = Image::solid(2, 1, [90, 80, 70, 77]);
        let transposed = Image::solid(1, 2, [80, 70, 60, 77]);
        assert_eq!(
            check_clarity(
                &source,
                &base,
                &transposed,
                &[255; 2],
                (0, 0, 2, 1),
                1.0,
                0.0,
                4.68
            ),
            Err(Refusal::ShapeMismatch("blur"))
        );
        assert_eq!(
            check_clarity(
                &source,
                &base,
                &base,
                &[255; 2],
                (0, 0, 1, 2),
                1.0,
                0.0,
                4.68
            ),
            Err(Refusal::ShapeMismatch("roi"))
        );
    }
}
