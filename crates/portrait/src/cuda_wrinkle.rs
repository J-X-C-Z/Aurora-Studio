//! PF21025/21069/21070/21071 wrinkle family on the P15 CUDA composite.
//!
//! The per-channel law is `kernels::skin_adjust::wrinkle_apply`, which is the
//! same expression as `kernels::facial_smooth::composite_pixel`. This adapter
//! therefore crops to the requested ROI, reuses the recovered P15 kernel, and
//! makes the byte store explicit: CPU truncation and GPU rounding are different
//! modes, never an assumed equivalence.

use crate::cuda::{self, CudaError, Dispatch};
use crate::kernels::facial_smooth::Uniforms as P15CpuUniforms;
use crate::kernels::skin_adjust::{self, Roi, StoreMode, WrinkleUniforms};
use crate::{Image, Refusal};

pub const KERNEL: &str = cuda::COMPOSITE;
pub const ROUTED_PFS: [u32; 4] = [21025, 21069, 21070, 21071];

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
    values.iter().map(|byte| *byte as f32 / 255.0).collect()
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
                    mask_roi[index * 4] as f32 / 255.0,
                    0.0,
                    mask_roi[index * 4 + 2] as f32 / 255.0,
                    0.0,
                ]
            })
            .collect();
        let skin_f32: Vec<f32> = skin_roi
            .iter()
            .map(|byte| *byte as f32 / 255.0)
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
