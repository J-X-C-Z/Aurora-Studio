//! PF21025/21069/21070/21071 wrinkle family on the P15 CUDA composite.
//!
//! The per-channel law is `kernels::skin_adjust::wrinkle_apply`, which is the
//! same expression as `kernels::facial_smooth::composite_pixel`. This adapter
//! therefore crops to the requested ROI, reuses the recovered P15 kernel, and
//! makes the byte store explicit: CPU truncation and GPU rounding are different
//! modes, never an assumed equivalence.

use crate::cuda::{self, CudaError, Dispatch};
use crate::cuda_job_upload::{
    JobBufferDesc, JobBufferPlan, JobBufferPurpose, JobBufferRole,
};
use crate::kernels::facial_smooth::Uniforms as P15CpuUniforms;
use crate::kernels::skin_adjust::{self, Roi, StoreMode, WrinkleUniforms};
use crate::{Image, Refusal};

pub const KERNEL: &str = cuda::COMPOSITE;
pub const U8_KERNEL: &str = cuda::COMPOSITE_U8;
pub const ROUTED_PFS: [u32; 4] = [21025, 21069, 21070, 21071];
pub const HISTORICAL_SKIN_PFS: [u32; 3] = [21024, 90023, 90024];
const INV255: f32 = 1.0 / 255.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WrinkleRoute {
    Puppet,
    Cheek,
    NoseBack,
    Perioral,
}

impl WrinkleRoute {
    pub fn resolve(pf: u32) -> Result<Self, Refusal> {
        match pf {
            21025 => Ok(Self::Puppet),
            21069 => Ok(Self::Cheek),
            21070 => Ok(Self::NoseBack),
            21071 => Ok(Self::Perioral),
            _ => Err(Refusal::UnknownEffect(pf)),
        }
    }

    pub const fn pf(self) -> u32 {
        match self {
            Self::Puppet => 21025,
            Self::Cheek => 21069,
            Self::NoseBack => 21070,
            Self::Perioral => 21071,
        }
    }
}

fn p15_uniforms(uniforms: WrinkleUniforms) -> P15CpuUniforms {
    P15CpuUniforms {
        legal_alpha: uniforms.legal_alpha,
        dark_circle_alpha: uniforms.dark_circle_alpha,
        diff_factor: uniforms.diff_factor,
        diff_factor2: uniforms.diff_factor2,
        color_threshold_factor: uniforms.color_threshold_factor,
        mix_alpha: uniforms.mix_alpha,
        diff_min_threshold: uniforms.diff_min_threshold,
        diff_max_threshold: uniforms.diff_max_threshold,
        high_freq_min_threshold: uniforms.high_freq_min_threshold,
        high_freq_max_threshold: uniforms.high_freq_max_threshold,
    }
}

fn crop_image(image: &Image, roi: Roi) -> Image {
    Image::new(roi.width, roi.height, crop_rgba(&image.pixels, image.width, roi))
        .expect("validated ROI has non-zero dimensions")
}

fn crop_rgba(values: &[u8], width: u32, roi: Roi) -> Vec<u8> {
    let row_bytes = roi.width as usize * 4;
    let mut out = Vec::with_capacity(roi.height as usize * row_bytes);
    for y in roi.y..roi.y + roi.height {
        let start = ((y * width + roi.x) as usize) * 4;
        out.extend_from_slice(&values[start..start + row_bytes]);
    }
    out
}

fn crop_plane(values: &[u8], width: u32, roi: Roi) -> Vec<u8> {
    let mut out = Vec::with_capacity(roi.width as usize * roi.height as usize);
    for y in roi.y..roi.y + roi.height {
        let start = (y * width + roi.x) as usize;
        out.extend_from_slice(&values[start..start + roi.width as usize]);
    }
    out
}

fn rgba_f32(values: &[u8]) -> Vec<f32> {
    values.iter().map(|byte| *byte as f32 * INV255).collect()
}

fn check(
    source: &Image,
    base: &Image,
    residual: &Image,
    mask_rgba: &[u8],
    skin_mask: &[u8],
    roi: Roi,
    uniforms: WrinkleUniforms,
) -> Result<(), Refusal> {
    source.validate()?;
    base.validate()?;
    residual.validate()?;
    if (source.width, source.height) != (base.width, base.height)
        || (source.width, source.height) != (residual.width, residual.height)
    {
        return Err(Refusal::ShapeMismatch("base/residual"));
    }
    roi.validate(source.width, source.height)?;
    if mask_rgba.len() != source.pixel_count() * 4 {
        return Err(Refusal::ShapeMismatch("mask_rgba"));
    }
    if skin_mask.len() != source.pixel_count() {
        return Err(Refusal::ShapeMismatch("skin_mask"));
    }
    uniforms.validate()
}

/// CPU reference for the same route and explicit degree scaling.
#[allow(clippy::too_many_arguments)]
pub fn reference(
    pf: u32,
    source: &Image,
    base: &Image,
    residual: &Image,
    mask_rgba: &[u8],
    skin_mask: &[u8],
    roi: Roi,
    ui: f32,
    uniforms: WrinkleUniforms,
) -> Result<Image, Refusal> {
    let route = WrinkleRoute::resolve(pf)?;
    let degree = skin_adjust::wrinkle_degree(route.pf(), ui)?;
    skin_adjust::wrinkle_apply(
        source, base, residual, mask_rgba, skin_mask, roi, degree, uniforms,
    )
}

#[cfg(windows)]
pub struct WrinkleGpu {
    gpu: cuda::Nvidia,
}

#[cfg(windows)]
impl WrinkleGpu {
    pub fn open() -> Result<Self, CudaError> {
        Ok(Self {
            gpu: cuda::Nvidia::open()?,
        })
    }

    /// Routes one of the four PFs through the P15 composite. Only RGB inside
    /// `roi` is written; alpha and every outside pixel are copied from source.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_route(
        &self,
        pf: u32,
        source: &Image,
        base: &Image,
        residual: &Image,
        mask_rgba: &[u8],
        skin_mask: &[u8],
        roi: Roi,
        ui: f32,
        uniforms: WrinkleUniforms,
        store: StoreMode,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        let route = WrinkleRoute::resolve(pf).map_err(CudaError::Refused)?;
        let degree = skin_adjust::wrinkle_degree(route.pf(), ui).map_err(CudaError::Refused)?;
        check(
            source, base, residual, mask_rgba, skin_mask, roi, uniforms,
        )
        .map_err(CudaError::Refused)?;
        if degree <= cuda::GATE {
            return Ok((source.clone(), Vec::new()));
        }

        let source_roi = crop_image(source, roi);
        let base_roi = crop_image(base, roi);
        let residual_roi = crop_image(residual, roi);
        let mask_roi = crop_rgba(mask_rgba, source.width, roi);
        let skin_roi = crop_plane(skin_mask, source.width, roi);
        let mask_f32: Vec<f32> = (0..skin_roi.len())
            .flat_map(|index| {
                [
                    mask_roi[index * 4] as f32 * INV255,
                    0.0,
                    mask_roi[index * 4 + 2] as f32 * INV255,
                    0.0,
                ]
            })
            .collect();
        let skin_f32: Vec<f32> = skin_roi
            .iter()
            .map(|byte| *byte as f32 * INV255)
            .collect();
        let p15 = cuda::Uniforms::new(
            roi.width,
            roi.height,
            route.pf(),
            degree,
            &p15_uniforms(uniforms),
        );
        let (float, dispatch) = self.gpu.composite_f32(
            &rgba_f32(&source_roi.pixels),
            &mask_f32,
            &skin_f32,
            &rgba_f32(&base_roi.pixels),
            &rgba_f32(&residual_roi.pixels),
            &p15,
        )?;

        let mut out = source.clone();
        for y in 0..roi.height {
            for x in 0..roi.width {
                let local = ((y * roi.width + x) as usize) * 4;
                let destination = (((roi.y + y) * source.width + roi.x + x) as usize) * 4;
                for channel in 0..3 {
                    out.pixels[destination + channel] = store.quantize(float[local + channel]);
                }
            }
        }
        Ok((out, vec![dispatch]))
    }

    /// Runs one historical wrinkle row from the caller's shared job arena.
    /// P15 consumes the fbe class-1 skin plane, RGBA mask, base/residual and
    /// control; this ABI has no LUT channel.
    #[allow(clippy::too_many_arguments)]
    pub fn apply_historical_same_job(
        &self,
        job_id: u64,
        pf: u32,
        source: &Image,
        base: &Image,
        residual: &Image,
        mask_rgba: &[u8],
        skin_mask: &[u8],
        roi: Roi,
        control: f32,
        uniforms: WrinkleUniforms,
        store: StoreMode,
    ) -> Result<(Image, Vec<Dispatch>), CudaError> {
        if !HISTORICAL_SKIN_PFS.contains(&pf) {
            return Err(CudaError::Refused(Refusal::UnknownEffect(pf)));
        }
        if !control.is_finite() || !(0.0..=1.0).contains(&control) {
            return Err(CudaError::Refused(Refusal::ValueOutOfRange(
                "historical skin control",
            )));
        }
        check(
            source, base, residual, mask_rgba, skin_mask, roi, uniforms,
        )
        .map_err(CudaError::Refused)?;
        if control <= cuda::GATE {
            return Ok((source.clone(), Vec::new()));
        }

        let source_roi = crop_image(source, roi);
        let base_roi = crop_image(base, roi);
        let residual_roi = crop_image(residual, roi);
        let mask_roi = crop_rgba(mask_rgba, source.width, roi);
        let skin_roi = crop_plane(skin_mask, source.width, roi);
        let output_seed = vec![0f32; skin_roi.len() * 4];
        let control_payload = [control];
        let plan = JobBufferPlan::new(
            job_id,
            (roi.width, roi.height),
            vec![
                JobBufferDesc {
                    role: JobBufferRole::HistoricalSkin,
                    ordinal: 0,
                    purpose: JobBufferPurpose::ReadOnlyInput,
                    payload: &source_roi.pixels,
                },
                JobBufferDesc {
                    role: JobBufferRole::HistoricalSkin,
                    ordinal: 1,
                    purpose: JobBufferPurpose::ReadOnlyInput,
                    payload: &mask_roi,
                },
                JobBufferDesc {
                    role: JobBufferRole::HistoricalSkin,
                    ordinal: 2,
                    purpose: JobBufferPurpose::ReadOnlyInput,
                    payload: &skin_roi,
                },
                JobBufferDesc {
                    role: JobBufferRole::HistoricalSkin,
                    ordinal: 3,
                    purpose: JobBufferPurpose::ReadOnlyInput,
                    payload: &base_roi.pixels,
                },
                JobBufferDesc {
                    role: JobBufferRole::HistoricalSkin,
                    ordinal: 4,
                    purpose: JobBufferPurpose::ReadOnlyInput,
                    payload: &residual_roi.pixels,
                },
                JobBufferDesc {
                    role: JobBufferRole::HistoricalSkin,
                    ordinal: 5,
                    purpose: JobBufferPurpose::Seed,
                    payload: &output_seed,
                },
                JobBufferDesc {
                    role: JobBufferRole::HistoricalSkin,
                    ordinal: 6,
                    purpose: JobBufferPurpose::ReadOnlyInput,
                    payload: &control_payload,
                },
            ],
        )
        .map_err(CudaError::Refused)?;
        let p15 = cuda::Uniforms::new(
            roi.width,
            roi.height,
            pf,
            control,
            &p15_uniforms(uniforms),
        );
        let ((float, dispatch), _) = self
            .gpu
            .with_job_buffers(&plan, |session| {
                let mut source_arg = session.handle(JobBufferRole::HistoricalSkin, 0)?;
                let mut mask_arg = session.handle(JobBufferRole::HistoricalSkin, 1)?;
                let mut skin_arg = session.handle(JobBufferRole::HistoricalSkin, 2)?;
                let mut base_arg = session.handle(JobBufferRole::HistoricalSkin, 3)?;
                let mut residual_arg = session.handle(JobBufferRole::HistoricalSkin, 4)?;
                let mut output_arg = session.handle(JobBufferRole::HistoricalSkin, 5)?;
                let mut uniforms_arg = p15;
                let mut params: [*mut std::ffi::c_void; 7] = [
                    &mut source_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut mask_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut skin_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut base_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut residual_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut output_arg as *mut u64 as *mut std::ffi::c_void,
                    &mut uniforms_arg as *mut cuda::Uniforms as *mut std::ffi::c_void,
                ];
                let dispatch = session.launch(
                    U8_KERNEL,
                    (
                        roi.width.div_ceil(16).max(1),
                        roi.height.div_ceil(16).max(1),
                    ),
                    (16, 16),
                    &mut params,
                )?;
                let mut float = vec![0f32; skin_roi.len() * 4];
                session.download_f32(JobBufferRole::HistoricalSkin, 5, &mut float)?;
                session.mark_consumed();
                Ok((float, dispatch))
            })?;

        let mut out = source.clone();
        for y in 0..roi.height {
            for x in 0..roi.width {
                let local = ((y * roi.width + x) as usize) * 4;
                let destination =
                    (((roi.y + y) * source.width + roi.x + x) as usize) * 4;
                for channel in 0..3 {
                    out.pixels[destination + channel] =
                        store.quantize(float[local + channel]);
                }
            }
        }
        Ok((out, vec![dispatch]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_family_is_exact() {
        assert_eq!(
            ROUTED_PFS.map(|pf| WrinkleRoute::resolve(pf).unwrap().pf()),
            ROUTED_PFS
        );
        assert_eq!(
            WrinkleRoute::resolve(21026),
            Err(Refusal::UnknownEffect(21026))
        );
    }

    #[test]
    fn trunc_and_round_are_not_aliases() {
        assert_eq!(StoreMode::TruncU8.quantize(0.5), 127);
        assert_eq!(StoreMode::Round.quantize(0.5), 128);
        assert_ne!(
            StoreMode::TruncU8.quantize(0.5),
            StoreMode::Round.quantize(0.5)
        );
    }

    #[test]
    fn byte_inputs_follow_the_cpu_inv255_lane() {
        let actual = rgba_f32(&[3, 6, 7, 255]);
        let expected: Vec<f32> = [3.0, 6.0, 7.0, 255.0]
            .into_iter()
            .map(|byte| byte * INV255)
            .collect();
        assert_eq!(actual, expected);
        assert_ne!(actual[0], 3.0f32 / 255.0);
    }

    #[cfg(windows)]
    #[test]
    fn family_reuses_composite_buffers_and_shared_context() {
        let gpu = match WrinkleGpu::open() {
            Ok(gpu) => gpu,
            Err(error) => {
                assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "CUDA required but unavailable: {error:?}"
                );
                return;
            }
        };
        gpu.gpu.clear_composite_buffers().unwrap();
        let (width, height) = (37, 29);
        let pixels = (width * height) as usize;
        let bytes = |seed: usize| {
            (0..pixels * 4)
                .map(|index| ((index * 37 + seed) % 256) as u8)
                .collect::<Vec<_>>()
        };
        let source = Image::new(width, height, bytes(7)).unwrap();
        let base = Image::new(width, height, bytes(11)).unwrap();
        let residual = Image::new(width, height, bytes(13)).unwrap();
        let mask = bytes(17);
        let skin = (0..pixels)
            .map(|index| ((index * 29 + 5) % 256) as u8)
            .collect::<Vec<_>>();
        let roi = Roi {
            x: 5,
            y: 4,
            width: 23,
            height: 17,
        };
        let uniforms = WrinkleUniforms {
            legal_alpha: 1.0,
            dark_circle_alpha: 1.0,
            diff_factor: 1.0,
            diff_factor2: 0.0,
            color_threshold_factor: 0.0,
            mix_alpha: 1.0,
            diff_min_threshold: -1.0,
            diff_max_threshold: 1.0,
            high_freq_min_threshold: -1.0,
            high_freq_max_threshold: 1.0,
        };

        let mut previous = cuda::test_driver_counts();
        for (index, pf) in ROUTED_PFS.into_iter().enumerate() {
            let expected = reference(
                pf, &source, &base, &residual, &mask, &skin, roi, 50.0, uniforms,
            )
            .unwrap();
            let (actual, _) = gpu
                .apply_route(
                    pf,
                    &source,
                    &base,
                    &residual,
                    &mask,
                    &skin,
                    roi,
                    50.0,
                    uniforms,
                    StoreMode::TruncU8,
                )
                .unwrap();
            assert_eq!(actual, expected, "PF{pf} byte mismatch");

            let current = cuda::test_driver_counts();
            assert_eq!(current[0], 1, "PF{pf} must share one context");
            assert_eq!(
                current[1] - previous[1],
                if index == 0 { 6 } else { 0 },
                "PF{pf} composite allocation count"
            );
            assert_eq!(
                current[2], previous[2],
                "PF{pf} must not issue an explicit context synchronize"
            );
            eprintln!(
                "PF{pf} contexts={} alloc_delta={} explicit_ctx_sync_delta={}",
                current[0],
                current[1] - previous[1],
                current[2] - previous[2]
            );
            previous = current;
        }
    }
}
