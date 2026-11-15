//! PF30013 neutral-gray flattening and PF30014 neutral-gray 3D on NVIDIA.
//!
//! The entries mirror `kernels::skin_adjust`; they do not discover a face mask,
//! 3D asset or ROI. PF30013 crops its explicit ROI before dispatch and writes
//! only RGB back into the current image, preserving alpha and pixels outside
//! the ROI. PF30014 requires an explicit 512x512 3D plane set and ROI binding;
//! either missing input is a typed refusal, never a synthesized frame.

use crate::cuda::{CudaError, Dispatch};
use crate::kernels::skin_adjust::{
    NeutralGray3dDegrees, Roi, NEUTRAL_GRAY_3D_BYTES, NEUTRAL_GRAY_3D_PIXELS,
    NEUTRAL_GRAY_3D_SIDE,
};
use crate::{Image, Refusal};

pub const SKIN_GREY_KERNEL: &str = "skin_grey_kernel";
pub const NEUTRAL_GRAY_3D_KERNEL: &str = "neutral_gray_3d_kernel";
pub const NEUTRAL_GRAY_3D_ROI: Roi = Roi {
    x: 0,
    y: 0,
    width: NEUTRAL_GRAY_3D_SIDE as u32,
    height: NEUTRAL_GRAY_3D_SIDE as u32,
};

const BLOCK: (u32, u32) = (16, 16);
const PTX: &str = include_str!("../native/cuda/neutral_gray.ptx");

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct SkinGreyUniforms {
    width: i32,
    height: i32,
    alpha: f32,
}

const _: () = assert!(std::mem::size_of::<SkinGreyUniforms>() == 12);

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct NeutralGray3dUniforms {
    active: i32,
}

const _: () = assert!(std::mem::size_of::<NeutralGray3dUniforms>() == 4);

/// The same-job inputs for PF30014. Packing is BGR8, matching the CPU lane.
#[derive(Clone, Copy)]
pub struct NeutralGray3dPlanes<'a> {
    pub source_bgr: &'a [u8],
    pub light: &'a [f32],
    pub dark: &'a [f32],
    pub alpha: &'a [f32],
    pub mask: &'a [f32],
}

/// Resolves both external bindings the recovered 3D consumer needs. A missing
/// asset or ROI is refused before any driver call.
pub fn require_neutral_gray_3d<'a>(
    asset: Option<NeutralGray3dPlanes<'a>>,
    roi: Option<Roi>,
) -> Result<NeutralGray3dPlanes<'a>, Refusal> {
    let asset = asset.ok_or(Refusal::MissingInput("neutral_gray_3d_asset"))?;
    let roi = roi.ok_or(Refusal::MissingInput("neutral_gray_3d_roi"))?;
    if roi != NEUTRAL_GRAY_3D_ROI {
        return Err(Refusal::ValueOutOfRange("neutral_gray_3d_roi"));
    }
    Ok(asset)
}

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

fn slice_as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            values.as_ptr() as *const u8,
            std::mem::size_of_val(values),
        )
    }
}

fn roi_valid(roi: Roi, width: u32, height: u32) -> bool {
    roi.width > 0
        && roi.height > 0
        && roi.x as u64 + roi.width as u64 <= width as u64
        && roi.y as u64 + roi.height as u64 <= height as u64
}

fn check_skin_grey(
    original: &Image,
    current: &Image,
    mask: &[u8],
    roi: Roi,
    alpha: f32,
) -> Result<(), Refusal> {
    original
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("original"))?;
    current
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("current"))?;
    if original.width != current.width || original.height != current.height {
        return Err(Refusal::ShapeMismatch("current"));
    }
    if !roi_valid(roi, current.width, current.height) {
        return Err(Refusal::ValueOutOfRange("roi"));
    }
    if mask.len() != current.pixel_count() {
        return Err(Refusal::ShapeMismatch("mask"));
    }
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(Refusal::ValueOutOfRange("alpha"));
    }
    Ok(())
}

fn check_neutral_gray_3d(planes: NeutralGray3dPlanes<'_>) -> Result<(), Refusal> {
    if planes.source_bgr.len() != NEUTRAL_GRAY_3D_BYTES {
        return Err(Refusal::ShapeMismatch("source_bgr512"));
    }
    if planes.light.len() != NEUTRAL_GRAY_3D_BYTES
        || planes.dark.len() != NEUTRAL_GRAY_3D_BYTES
    {
        return Err(Refusal::ShapeMismatch("light/dark"));
    }
    if planes.alpha.len() != NEUTRAL_GRAY_3D_PIXELS
        || planes.mask.len() != NEUTRAL_GRAY_3D_PIXELS
    {
        return Err(Refusal::ShapeMismatch("alpha/mask"));
    }
    if !planes
        .light
        .iter()
        .chain(planes.dark.iter())
        .all(|value| value.is_finite() && *value >= 0.0)
    {
        return Err(Refusal::ValueOutOfRange("light/dark"));
    }
    if !planes
        .alpha
        .iter()
        .chain(planes.mask.iter())
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    {
        return Err(Refusal::ValueOutOfRange("alpha/mask"));
    }
    Ok(())
}

fn crop_rgba(image: &Image, roi: Roi) -> Vec<u8> {
    let row_bytes = roi.width as usize * 4;
    let mut out = Vec::with_capacity(row_bytes * roi.height as usize);
    for y in roi.y..roi.y + roi.height {
        let start = (y as usize * image.width as usize + roi.x as usize) * 4;
        out.extend_from_slice(&image.pixels[start..start + row_bytes]);
    }
    out
}

fn crop_mask(mask: &[u8], image: &Image, roi: Roi) -> Vec<u8> {
    let mut out = Vec::with_capacity((roi.width * roi.height) as usize);
    for y in roi.y..roi.y + roi.height {
        let start = y as usize * image.width as usize + roi.x as usize;
        out.extend_from_slice(&mask[start..start + roi.width as usize]);
    }
    out
}

fn paste_rgb(current: &mut Image, roi: Roi, pixels: &[u8]) {
    let row_bytes = roi.width as usize * 4;
    for row in 0..roi.height as usize {
        let dst = ((roi.y as usize + row) * current.width as usize + roi.x as usize) * 4;
        let src = row * row_bytes;
        for column in 0..roi.width as usize {
            let at = column * 4;
            current.pixels[dst + at..dst + at + 3]
                .copy_from_slice(&pixels[src + at..src + at + 3]);
        }
    }
}

/// One open CUDA image containing both recovered entries.
#[cfg(windows)]
pub struct NeutralGrayGpu(crate::cuda::Nvidia);

#[cfg(windows)]
impl NeutralGrayGpu {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self(crate::cuda::Nvidia::open_ptx(PTX)?))
    }

    /// PF30013: `out = current + (HSL(original.H/S, current.L) - current) *
    /// clamp(mask/255 * alpha)`, RGB only inside the explicit ROI.
    pub fn skin_grey(
        &self,
        original: &Image,
        current: &Image,
        mask: &[u8],
        roi: Roi,
        alpha: f32,
    ) -> Result<(Image, Dispatch), CudaError> {
        check_skin_grey(original, current, mask, roi, alpha)?;
        let original_roi = crop_rgba(original, roi);
        let current_roi = crop_rgba(current, roi);
        let mask_roi = crop_mask(mask, current, roi);
        let uniforms = SkinGreyUniforms {
            width: roi.width as i32,
            height: roi.height as i32,
            alpha,
        };
        let grid = (
            roi.width.div_ceil(BLOCK.0).max(1),
            roi.height.div_ceil(BLOCK.1).max(1),
        );
        let (pixels, dispatch) = self.0.run_u8(
            SKIN_GREY_KERNEL,
            grid,
            &[&original_roi, &current_roi, &mask_roi],
            as_bytes(&uniforms),
        )?;
        let mut out = current.clone();
        paste_rgb(&mut out, roi, &pixels);
        Ok((out, dispatch))
    }

    /// PF30014. The inactive gate is the CPU identity path and therefore has
    /// no dispatch; it is never reported as GPU execution.
    pub fn neutral_gray_3d(
        &self,
        asset: Option<NeutralGray3dPlanes<'_>>,
        roi: Option<Roi>,
        degrees: NeutralGray3dDegrees,
    ) -> Result<(Vec<u8>, Option<Dispatch>), CudaError> {
        let planes = require_neutral_gray_3d(asset, roi)?;
        check_neutral_gray_3d(planes)?;
        if !degrees.active {
            return Ok((planes.source_bgr.to_vec(), None));
        }
        let uniforms = NeutralGray3dUniforms { active: 1 };
        let side = NEUTRAL_GRAY_3D_SIDE as u32;
        let grid = (side.div_ceil(BLOCK.0), side.div_ceil(BLOCK.1));
        let (pixels, dispatch) = self.0.run_u8(
            NEUTRAL_GRAY_3D_KERNEL,
            grid,
            &[
                planes.source_bgr,
                slice_as_bytes(planes.light),
                slice_as_bytes(planes.dark),
                slice_as_bytes(planes.alpha),
                slice_as_bytes(planes.mask),
            ],
            as_bytes(&uniforms),
        )?;
        Ok((pixels, Some(dispatch)))
    }
}
