//! PF90087 final hair-seam blend on NVIDIA.
//!
//! The normal/Sobel and bilateral / 16-direction search stays in the CPU
//! producer. This consumer takes its tight RGB8 repair plane and compacted
//! coverage mask, then applies the final source-order mix through `run_u8`.

use crate::cuda::{CudaError, Dispatch};
use crate::{Image, Refusal};

/// Kernel entry point.
pub const KERNEL: &str = "pf90087_hair_seam_blend_kernel";
const BLOCK: u32 = 16;

const PTX: &str = include_str!("../native/cuda/hair_seam_blend.ptx");

/// 32-byte kernel argument block, matching `native/cuda/hair_seam_blend.h`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Uniforms {
    pub width: u32,
    pub height: u32,
    pub degree: f32,
    pub roi_x: u32,
    pub roi_y: u32,
    pub roi_w: u32,
    pub roi_h: u32,
    pub reserved: u32,
}

const _: () = assert!(std::mem::size_of::<Uniforms>() == 32);

fn as_bytes<T>(value: &T) -> &[u8] {
    unsafe { std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>()) }
}

/// Checks shared with `kernels::face_detail::hair_seam_blend`.
fn check(
    source: &Image,
    part_line_mask: Option<&[u8]>,
    repair_rgb: Option<&[u8]>,
    repair_mask: Option<&[u8]>,
    ui: f32,
    roi: (u32, u32, u32, u32),
) -> Result<f32, Refusal> {
    source
        .validate()
        .map_err(|_| Refusal::ShapeMismatch("source"))?;
    let pixels = source.pixel_count();
    if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
        return Err(Refusal::ValueOutOfRange("PF90087_UI"));
    }
    let (x, y, width, height) = roi;
    if x > source.width
        || y > source.height
        || width > source.width - x
        || height > source.height - y
    {
        return Err(Refusal::ValueOutOfRange("PF90087_ROI"));
    }
    let degree = ui / 100.0;
    if degree == 0.0 || width == 0 || height == 0 {
        return Ok(degree);
    }
    let part_line_mask = part_line_mask.ok_or(Refusal::MissingInput("PF90087_PART_LINE_MASK"))?;
    let repair_rgb = repair_rgb.ok_or(Refusal::MissingInput("PF90087_REPAIR_RGB"))?;
    let repair_mask = repair_mask.ok_or(Refusal::MissingInput("PF90087_REPAIR_MASK"))?;
    if part_line_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("PF90087_PART_LINE_MASK"));
    }
    if repair_rgb.len() != pixels * 3 {
        return Err(Refusal::ShapeMismatch("PF90087_REPAIR_RGB"));
    }
    if repair_mask.len() != pixels {
        return Err(Refusal::ShapeMismatch("PF90087_REPAIR_MASK"));
    }
    Ok(degree)
}

#[cfg(windows)]
pub struct Blend {
    gpu: crate::cuda::Nvidia,
}

#[cfg(windows)]
impl Blend {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self {
            gpu: crate::cuda::Nvidia::open_ptx(PTX)?,
        })
    }

    /// Full geometry is supplied by the caller; only `roi` is dispatched.
    /// Every pixel outside the ROI keeps its source bytes, including alpha.
    pub fn apply(
        &self,
        source: &Image,
        part_line_mask: Option<&[u8]>,
        repair_rgb: Option<&[u8]>,
        repair_mask: Option<&[u8]>,
        ui: f32,
        roi: (u32, u32, u32, u32),
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        let degree = check(source, part_line_mask, repair_rgb, repair_mask, ui, roi)?;
        if degree == 0.0 || roi.2 == 0 || roi.3 == 0 {
            return Ok((source.clone(), Vec::new()));
        }
        let uniforms = Uniforms {
            width: source.width,
            height: source.height,
            degree,
            roi_x: roi.0,
            roi_y: roi.1,
            roi_w: roi.2,
            roi_h: roi.3,
            reserved: 0,
        };
        let grid = (roi.2.div_ceil(BLOCK).max(1), roi.3.div_ceil(BLOCK).max(1));
        let (pixels, dispatch) = self.gpu.run_u8_inplace(
            KERNEL,
            grid,
            &[
                &source.pixels,
                part_line_mask.expect("validated active part-line mask"),
                repair_rgb.expect("validated repair RGB"),
                repair_mask.expect("validated repair coverage"),
            ],
            as_bytes(&uniforms),
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
    fn uniform_block_and_ptx_signature_match_run_u8() {
        assert_eq!(std::mem::size_of::<Uniforms>(), 32);
        assert_eq!(std::mem::align_of::<Uniforms>(), 4);
        let signature = PTX
            .split(&format!(".entry {KERNEL}("))
            .nth(1)
            .expect("PTX kernel entry")
            .split(')')
            .next()
            .unwrap();
        let params: Vec<_> = signature
            .lines()
            .filter(|line| line.contains(".param"))
            .collect();
        assert_eq!(params.len(), 6);
        assert!(params[5].contains(".b8") && params[5].contains("[32]"));
    }

    #[test]
    fn typed_checks_match_the_cpu_consumer() {
        let source = Image::solid(2, 2, [1, 2, 3, 4]);
        let masks = [255u8; 4];
        let repair = [9u8; 12];
        assert_eq!(
            check(
                &source,
                Some(&masks),
                Some(&repair),
                Some(&masks),
                100.0,
                (0, 0, 2, 2)
            ),
            Ok(1.0)
        );
        assert_eq!(
            check(
                &source,
                None,
                Some(&repair),
                Some(&masks),
                100.0,
                (0, 0, 2, 2)
            ),
            Err(Refusal::MissingInput("PF90087_PART_LINE_MASK"))
        );
        assert_eq!(
            check(
                &source,
                Some(&masks),
                Some(&repair[..11]),
                Some(&masks),
                100.0,
                (0, 0, 2, 2)
            ),
            Err(Refusal::ShapeMismatch("PF90087_REPAIR_RGB"))
        );
        assert_eq!(check(&source, None, None, None, 0.0, (0, 0, 2, 2)), Ok(0.0));
        assert_eq!(
            check(
                &source,
                Some(&masks),
                Some(&repair),
                Some(&masks),
                100.0,
                (2, 0, 1, 1)
            ),
            Err(Refusal::ValueOutOfRange("PF90087_ROI"))
        );
    }
}
