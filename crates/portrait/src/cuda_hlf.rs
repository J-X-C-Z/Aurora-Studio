//! PF30021 / PF30022 frequency-separation consumer slices.
//!
//! Explicit median/gauss planes remain same-job caller inputs. The auto
//! producer may use the candidate GPU preprocessing below; its interval and
//! border behavior are not vendor_equivalent.

#[cfg(windows)]
use crate::cuda::{ChainStage, CudaError, Dispatch, Nvidia};
use crate::kernels::skin_adjust::{AdjustLut, BodyHlfUniforms, Roi};
use crate::{Image, Refusal};

pub const FACE_KERNEL: &str = "pf30021_hlf_face_kernel";
pub const BODY_KERNEL: &str = "pf30022_hlf_body_kernel";
pub const MEDIAN_KERNEL: &str = "hlf_median_rgb_kernel";
pub const GAUSS_KERNEL: &str = "hlf_gauss_rgb_kernel";

const BLOCK: u32 = 16;
const PTX: &str = include_str!("../native/cuda/hlf.ptx");

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct FaceUniforms {
    width: u32,
    height: u32,
    degree: f32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
struct HlfBodyUniforms {
    width: u32,
    height: u32,
    roi_x: u32,
    roi_y: u32,
    roi_w: u32,
    roi_h: u32,
    degree: f32,
    retouch_degree: f32,
    retouch_high_freq_factor: f32,
    diff_min_range: f32,
    diff_max_range: f32,
    lut_width: u32,
    lut_height: u32,
    spotless_degree: f32,
    spotless_acne_threshold: f32,
    spotless_high_freq_factor: f32,
    spotless_low_freq_alpha: f32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PreprocessUniforms {
    width: u32,
    height: u32,
    interval: u32,
    reserved: u32,
}

const _: () = assert!(std::mem::size_of::<FaceUniforms>() == 16);
const _: () = assert!(std::mem::size_of::<HlfBodyUniforms>() == 72);
const _: () = assert!(std::mem::size_of::<PreprocessUniforms>() == 16);

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

fn grid(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(BLOCK).max(1), height.div_ceil(BLOCK).max(1))
}

fn check_plane3(plane: &[f32], pixels: usize, field: &'static str) -> Result<(), Refusal> {
    if plane.len() != pixels * 3 {
        return Err(Refusal::ShapeMismatch(field));
    }
    if !plane.iter().all(|value| value.is_finite()) {
        return Err(Refusal::ValueOutOfRange(field));
    }
    Ok(())
}

fn check_plane1(plane: &[f32], pixels: usize, field: &'static str) -> Result<(), Refusal> {
    if plane.len() != pixels {
        return Err(Refusal::ShapeMismatch(field));
    }
    if !plane.iter().all(|value| value.is_finite()) {
        return Err(Refusal::ValueOutOfRange(field));
    }
    Ok(())
}

fn body_uniforms_valid(uniforms: BodyHlfUniforms) -> bool {
    [
        uniforms.spotless_degree,
        uniforms.retouch_degree,
        uniforms.spotless_acne_threshold,
        uniforms.spotless_high_freq_factor,
        uniforms.spotless_low_freq_alpha,
        uniforms.retouch_high_freq_factor,
        uniforms.diff_min_range,
        uniforms.diff_max_range,
    ]
    .iter()
    .all(|value| value.is_finite())
        && uniforms.diff_min_range <= uniforms.diff_max_range
}

#[cfg(windows)]
pub struct Hlf {
    gpu: Nvidia,
}

#[cfg(windows)]
impl Hlf {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self {
            gpu: Nvidia::open_ptx(PTX)?,
        })
    }

    /// Candidate PF30021/PF30022 preprocessing. The CPU reference uses the
    /// same 3x3 median and fixed 9-tap separable Gaussian with clamp-to-edge.
    pub fn preprocess_source_rgb(
        &self,
        source: &[f32],
        width: u32,
        height: u32,
        median_interval: u32,
        gauss_interval: u32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<Dispatch>), CudaError> {
        if width == 0 || height == 0 {
            return Err(Refusal::ValueOutOfRange("width/height").into());
        }
        if median_interval == 0 || gauss_interval == 0 {
            return Err(Refusal::ValueOutOfRange("HLF auto interval").into());
        }
        let pixels = width as usize * height as usize;
        check_plane3(source, pixels, "source")?;
        let uniforms = |interval| PreprocessUniforms {
            width,
            height,
            interval,
            reserved: 0,
        };
        let (median_bytes, median_dispatch) = self.gpu.run_u8(
            MEDIAN_KERNEL,
            grid(width, height),
            &[as_bytes(source)],
            as_bytes(&[uniforms(median_interval)]),
        )?;
        let (gauss_bytes, gauss_dispatch) = self.gpu.run_u8(
            GAUSS_KERNEL,
            grid(width, height),
            &[as_bytes(source)],
            as_bytes(&[uniforms(gauss_interval)]),
        )?;
        fn decode(bytes: Vec<u8>) -> Vec<f32> {
            bytes
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().expect("four-byte word")))
                .collect()
        }
        Ok((
            decode(median_bytes),
            decode(gauss_bytes),
            vec![median_dispatch, gauss_dispatch],
        ))
    }

    /// Fused automatic PF30021 path: the median and Gaussian planes stay on
    /// device and are consumed by the face kernel before the sole download.
    pub fn auto_face_apply(
        &self,
        source: &[f32],
        face_mask: &[f32],
        skin_mask: &[f32],
        width: u32,
        height: u32,
        median_interval: u32,
        gauss_interval: u32,
        ui: f32,
    ) -> Result<(Vec<f32>, Vec<Dispatch>), CudaError> {
        if width == 0 || height == 0 {
            return Err(Refusal::ValueOutOfRange("width/height").into());
        }
        if median_interval == 0 || gauss_interval == 0 {
            return Err(Refusal::ValueOutOfRange("HLF auto interval").into());
        }
        if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
            return Err(Refusal::ValueOutOfRange("ui").into());
        }
        let pixels = width as usize * height as usize;
        check_plane3(source, pixels, "source")?;
        check_plane1(face_mask, pixels, "face_mask")?;
        check_plane1(skin_mask, pixels, "skin_mask")?;
        if ui == 0.0 {
            return Ok((source.to_vec(), Vec::new()));
        }

        let source_bytes = as_bytes(source);
        let face_bytes = as_bytes(face_mask);
        let skin_bytes = as_bytes(skin_mask);
        let slots = vec![
            source_bytes.len(),
            source_bytes.len(),
            source_bytes.len(),
            face_bytes.len(),
            skin_bytes.len(),
            source_bytes.len(),
        ];
        let median_inputs = [0];
        let gauss_inputs = [0];
        let face_inputs = [0, 1, 2, 3, 4];
        let median_uniform = PreprocessUniforms {
            width,
            height,
            interval: median_interval,
            reserved: 0,
        };
        let gauss_uniform = PreprocessUniforms {
            width,
            height,
            interval: gauss_interval,
            reserved: 0,
        };
        let face_uniform = FaceUniforms {
            width,
            height,
            degree: ui / 100.0,
            reserved: 0,
        };
        let stages = [
            ChainStage {
                kernel: MEDIAN_KERNEL,
                grid: grid(width, height),
                inputs: &median_inputs,
                output: 1,
                lut: None,
                lut_arg: false,
                init_from: None,
                uniforms: as_bytes(std::slice::from_ref(&median_uniform)),
            },
            ChainStage {
                kernel: GAUSS_KERNEL,
                grid: grid(width, height),
                inputs: &gauss_inputs,
                output: 2,
                lut: None,
                lut_arg: false,
                init_from: None,
                uniforms: as_bytes(std::slice::from_ref(&gauss_uniform)),
            },
            ChainStage {
                kernel: FACE_KERNEL,
                grid: grid(width, height),
                inputs: &face_inputs,
                output: slots.len() - 1,
                lut: None,
                lut_arg: false,
                init_from: None,
                uniforms: as_bytes(std::slice::from_ref(&face_uniform)),
            },
        ];
        let uploads = [
            (0, source_bytes),
            (3, face_bytes),
            (4, skin_bytes),
        ];
        let (bytes, dispatches) =
            self.gpu
                .run_u8_chain(&stages, &slots, &uploads, slots.len() - 1)?;
        Ok((
            bytes
                .chunks_exact(4)
                .map(|word| f32::from_le_bytes(word.try_into().expect("four-byte word")))
                .collect(),
            dispatches,
        ))
    }

    /// Fused automatic PF30022 path. `source_rgb` feeds preprocessing while
    /// the original RGBA8 frame feeds the body kernel; only its result leaves
    /// the device.
    #[allow(clippy::too_many_arguments)]
    pub fn auto_body_apply(
        &self,
        source: &Image,
        source_rgb: &[f32],
        face_mask_rgba: &[f32],
        skin_mask: &[f32],
        roi: Roi,
        median_interval: u32,
        gauss_interval: u32,
        ui: f32,
        uniforms: BodyHlfUniforms,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        source.validate()?;
        let pixels = source.pixel_count();
        check_plane3(source_rgb, pixels, "source_rgb")?;
        if face_mask_rgba.len() != pixels * 4 {
            return Err(Refusal::ShapeMismatch("face_mask_rgba").into());
        }
        if skin_mask.len() != pixels {
            return Err(Refusal::ShapeMismatch("skin_mask").into());
        }
        if !face_mask_rgba
            .iter()
            .chain(skin_mask)
            .all(|value| value.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("masks").into());
        }
        if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
            return Err(Refusal::ValueOutOfRange("ui").into());
        }
        if median_interval == 0 || gauss_interval == 0 {
            return Err(Refusal::ValueOutOfRange("HLF auto interval").into());
        }
        if !body_uniforms_valid(uniforms) {
            return Err(Refusal::ValueOutOfRange("uniforms").into());
        }
        if uniforms.spotless_degree > 0.0 {
            return Err(Refusal::MissingInput("adjust_lut").into());
        }
        if ui == 0.0 || (!(uniforms.retouch_degree > 0.0)) {
            return Ok((source.clone(), Vec::new()));
        }
        roi.validate(source.width, source.height)?;

        let source_bytes = as_bytes(source_rgb);
        let face_bytes = as_bytes(face_mask_rgba);
        let skin_bytes = as_bytes(skin_mask);
        let frame_bytes = &source.pixels;
        let slots = vec![
            source_bytes.len(),
            source_bytes.len(),
            source_bytes.len(),
            face_bytes.len(),
            skin_bytes.len(),
            frame_bytes.len(),
            frame_bytes.len(),
        ];
        let median_inputs = [0];
        let gauss_inputs = [0];
        let body_inputs = [5, 1, 2, 3, 4];
        let median_uniform = PreprocessUniforms {
            width: source.width,
            height: source.height,
            interval: median_interval,
            reserved: 0,
        };
        let gauss_uniform = PreprocessUniforms {
            width: source.width,
            height: source.height,
            interval: gauss_interval,
            reserved: 0,
        };
        let body_uniform = HlfBodyUniforms {
            width: source.width,
            height: source.height,
            roi_x: roi.x,
            roi_y: roi.y,
            roi_w: roi.width,
            roi_h: roi.height,
            degree: ui / 100.0,
            retouch_degree: uniforms.retouch_degree,
            retouch_high_freq_factor: uniforms.retouch_high_freq_factor,
            diff_min_range: uniforms.diff_min_range,
            diff_max_range: uniforms.diff_max_range,
            lut_width: 0,
            lut_height: 0,
            spotless_degree: 0.0,
            spotless_acne_threshold: 0.0,
            spotless_high_freq_factor: 1.0,
            spotless_low_freq_alpha: 0.0,
            reserved: 0,
        };
        let stages = [
            ChainStage {
                kernel: MEDIAN_KERNEL,
                grid: grid(source.width, source.height),
                inputs: &median_inputs,
                output: 1,
                lut: None,
                lut_arg: false,
                init_from: None,
                uniforms: as_bytes(std::slice::from_ref(&median_uniform)),
            },
            ChainStage {
                kernel: GAUSS_KERNEL,
                grid: grid(source.width, source.height),
                inputs: &gauss_inputs,
                output: 2,
                lut: None,
                lut_arg: false,
                init_from: None,
                uniforms: as_bytes(std::slice::from_ref(&gauss_uniform)),
            },
            ChainStage {
                kernel: BODY_KERNEL,
                grid: grid(roi.width, roi.height),
                inputs: &body_inputs,
                output: slots.len() - 1,
                lut: None,
                lut_arg: true,
                init_from: Some(5),
                uniforms: as_bytes(std::slice::from_ref(&body_uniform)),
            },
        ];
        let uploads = [
            (0, source_bytes),
            (3, face_bytes),
            (4, skin_bytes),
            (5, frame_bytes),
        ];
        let (pixels_out, dispatches) =
            self.gpu
                .run_u8_chain(&stages, &slots, &uploads, slots.len() - 1)?;
        let image = Image::new(source.width, source.height, pixels_out)
            .ok_or(CudaError::Refused(Refusal::ShapeMismatch("output")))?;
        Ok((image, dispatches))
    }

    /// PF30021 evaluates the full frame. `roi` is validated for parity with the
    /// vendor wrapper but deliberately does not crop the dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn face_apply(
        &self,
        source: &[f32],
        median: &[f32],
        gauss: &[f32],
        face_mask: &[f32],
        skin_mask: &[f32],
        width: u32,
        height: u32,
        roi: Roi,
        ui: f32,
    ) -> Result<(Vec<f32>, Vec<Dispatch>), CudaError> {
        if width == 0 || height == 0 {
            return Err(Refusal::ValueOutOfRange("width/height").into());
        }
        roi.validate(width, height)?;
        if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
            return Err(Refusal::ValueOutOfRange("ui").into());
        }
        let pixels = width as usize * height as usize;
        check_plane3(source, pixels, "source")?;
        check_plane3(median, pixels, "median")?;
        check_plane3(gauss, pixels, "gauss")?;
        check_plane1(face_mask, pixels, "face_mask")?;
        check_plane1(skin_mask, pixels, "skin_mask")?;
        if ui == 0.0 {
            return Ok((source.to_vec(), Vec::new()));
        }

        let uniforms = FaceUniforms {
            width,
            height,
            degree: ui / 100.0,
            reserved: 0,
        };
        let (bytes, dispatch) = self.gpu.run_u8(
            FACE_KERNEL,
            grid(width, height),
            &[
                as_bytes(source),
                as_bytes(median),
                as_bytes(gauss),
                as_bytes(face_mask),
                as_bytes(skin_mask),
            ],
            as_bytes(&[uniforms]),
        )?;
        let output = bytes
            .chunks_exact(4)
            .map(|word| f32::from_le_bytes(word.try_into().expect("four-byte word")))
            .collect();
        Ok((output, vec![dispatch]))
    }

    /// PF30022 WithRoi consumer. The spotless branch uploads the shared
    /// single-row [`AdjustLut`] as an L8 device buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn body_apply(
        &self,
        source: &Image,
        roi: Roi,
        median: &[f32],
        gauss: &[f32],
        face_mask: &[f32],
        skin_mask: &[f32],
        ui: f32,
        uniforms: BodyHlfUniforms,
        adjust_lut: Option<&AdjustLut<'_>>,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        if !ui.is_finite() || !(0.0..=100.0).contains(&ui) {
            return Err(Refusal::ValueOutOfRange("ui").into());
        }
        source.validate()?;
        let pixels = source.pixel_count();
        check_plane3(median, pixels, "median_rgb")?;
        check_plane3(gauss, pixels, "gauss_rgb")?;
        if face_mask.len() != pixels * 4 {
            return Err(Refusal::ShapeMismatch("face_mask_rgba").into());
        }
        if skin_mask.len() != pixels {
            return Err(Refusal::ShapeMismatch("skin_mask").into());
        }
        if !face_mask
            .iter()
            .chain(skin_mask)
            .all(|value| value.is_finite())
        {
            return Err(Refusal::ValueOutOfRange("masks").into());
        }
        if !body_uniforms_valid(uniforms) {
            return Err(Refusal::ValueOutOfRange("uniforms").into());
        }
        let degree = ui / 100.0;
        if !(degree > 0.0) {
            return Ok((source.clone(), Vec::new()));
        }
        roi.validate(source.width, source.height)?;
        if let Some(lut) = adjust_lut {
            lut.validate()?;
        } else if uniforms.spotless_degree > 0.0 {
            return Err(Refusal::MissingInput("adjust_lut").into());
        }
        if !(uniforms.spotless_degree > 0.0) && !(uniforms.retouch_degree > 0.0) {
            return Ok((source.clone(), Vec::new()));
        }

        let lut = adjust_lut.filter(|_| uniforms.spotless_degree > 0.0);
        let uniforms = HlfBodyUniforms {
            width: source.width,
            height: source.height,
            roi_x: roi.x,
            roi_y: roi.y,
            roi_w: roi.width,
            roi_h: roi.height,
            degree,
            retouch_degree: uniforms.retouch_degree,
            retouch_high_freq_factor: uniforms.retouch_high_freq_factor,
            diff_min_range: uniforms.diff_min_range,
            diff_max_range: uniforms.diff_max_range,
            lut_width: lut.map_or(0, |lut| lut.width),
            lut_height: lut.map_or(0, |lut| lut.height),
            spotless_degree: uniforms.spotless_degree,
            spotless_acne_threshold: uniforms.spotless_acne_threshold,
            spotless_high_freq_factor: uniforms.spotless_high_freq_factor,
            spotless_low_freq_alpha: uniforms.spotless_low_freq_alpha,
            reserved: 0,
        };
        let (pixels_out, dispatch) = self.gpu.run_u8_with_init_optional_lut(
            BODY_KERNEL,
            grid(roi.width, roi.height),
            &[
                &source.pixels,
                as_bytes(median),
                as_bytes(gauss),
                as_bytes(face_mask),
                as_bytes(skin_mask),
            ],
            lut.map(|lut| lut.luma),
            as_bytes(&[uniforms]),
            0,
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
    fn abi_matches_the_c_header() {
        assert_eq!(std::mem::size_of::<FaceUniforms>(), 16);
        assert_eq!(std::mem::size_of::<HlfBodyUniforms>(), 72);
        assert_eq!(std::mem::size_of::<PreprocessUniforms>(), 16);
    }

    #[test]
    fn ptx_contains_both_entries() {
        for kernel in [FACE_KERNEL, BODY_KERNEL, MEDIAN_KERNEL, GAUSS_KERNEL] {
            assert!(PTX.contains(&format!(".entry {kernel}(")), "{kernel}");
        }
    }
}
