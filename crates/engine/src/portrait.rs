//! Portrait stages with explicit inputs from the same image and processing job.
//! The registry describes stages; it does not imply automatic landmark, mask,
//! material or model generation. Existing portrait sliders remain pending.

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail, ensure};
use aurora_pixcake::{self as backend, Image, kernels};
use serde::Serialize;

use crate::{ImageInfo, ProcessRecipe, RgbaImage};

#[derive(Clone, Debug, Serialize)]
pub struct Capability {
    pub pf: u32,
    pub key: String,
    pub label: String,
    pub stage: &'static str,
    pub required_inputs: Vec<String>,
    pub engine_bridge: bool,
    pub automatic: bool,
}

pub fn capabilities() -> Vec<Capability> {
    backend::catalog().into_iter().map(|row| Capability {
        pf: row.pf,
        key: row.key.into(),
        label: row.label.into(),
        stage: match row.access {
            backend::Access::Cpu => "cpu_stage",
            backend::Access::GpuPlanned => "parameters_only",
            backend::Access::Refused => "unavailable",
        },
        required_inputs: row.inputs.iter().map(|input| (*input).into()).collect(),
        engine_bridge: matches!(row.pf, 8069 | 90023 | 90024 | 21024 | 90015),
        automatic: false,
    }).collect()
}

/// The caller can reuse its CUDA session across requests. An explicit GPU
/// request never silently falls back to CPU.
pub enum Execution<'a> {
    Cpu,
    Cuda(&'a backend::cuda::Nvidia),
}

pub enum Operation<'a> {
    MaskCompose { mask: &'a [u8], mode: kernels::mask_compose::MaskComposeMode },
    TextureSmooth { mask_r: &'a [u8], extra_mask_r: &'a [u8], ui: f32 },
    FacialSmooth {
        pf: u32,
        native_value: f32,
        base: &'a Image,
        residual: &'a Image,
        mask_r: &'a [u8],
        mask_b: &'a [u8],
        skin_mask: &'a [u8],
        uniforms: kernels::facial_smooth::Uniforms,
    },
    NeckPaste { repair: &'a Image, x: u32, y: u32 },
}

pub struct Request<'a> {
    pub source: &'a Image,
    pub operation: Operation<'a>,
    pub execution: Execution<'a>,
}

#[derive(Debug)]
pub struct Frame {
    pub image: RgbaImage,
    /// Present only after an actual CUDA launch, including device details.
    pub cuda_dispatch: Option<backend::cuda::Dispatch>,
}

fn input_error(error: backend::Refusal) -> anyhow::Error {
    anyhow!("portrait input: {error:?}")
}

pub fn render(request: &Request<'_>) -> Result<Frame> {
    request.source.validate().map_err(input_error)?;
    let mut cuda_dispatch = None;
    let image = match &request.operation {
        Operation::TextureSmooth { mask_r, extra_mask_r, ui } => {
            ensure!(matches!(request.execution, Execution::Cpu), "texture smoothing requires CPU execution");
            kernels::skin_models::texture_surface_blur(request.source, mask_r, extra_mask_r, *ui).map_err(input_error)?
        }
        Operation::MaskCompose { mask, mode } => {
            ensure!(matches!(request.execution, Execution::Cpu), "mask composition requires CPU execution");
            kernels::mask_compose::apply(request.source, mask, *mode).map_err(input_error)?
        }
        Operation::NeckPaste { repair, x, y } => {
            ensure!(matches!(request.execution, Execution::Cpu), "neck paste requires CPU execution");
            kernels::neck_paste::paste(request.source, repair, *x, *y).map_err(input_error)?
        }
        Operation::FacialSmooth { pf, native_value, base, residual, mask_r, mask_b, skin_mask, uniforms } => {
            ensure!(matches!(*pf, 90023 | 90024 | 21024), "unsupported facial smooth parameter {pf}");
            ensure!(native_value.is_finite() && *native_value >= 0.0, "invalid native portrait value");
            // The native gate is independent of the ten caller-supplied uniforms.
            if *native_value <= 0.01 {
                request.source.clone()
            } else {
                match &request.execution {
                    Execution::Cpu => kernels::facial_smooth::apply(
                        request.source, base, residual, mask_r, mask_b, skin_mask, *uniforms,
                    ).map_err(input_error)?,
                    Execution::Cuda(device) => {
                        let (image, dispatch) = device.facial_smooth(
                            request.source, base, residual, mask_r, mask_b, skin_mask,
                            *pf, *native_value, uniforms,
                        ).map_err(|error| anyhow!("portrait CUDA: {error:?}"))?;
                        cuda_dispatch = Some(dispatch);
                        image
                    }
                }
            }
        }
    };
    Ok(Frame {
        image: RgbaImage::new(image.width, image.height, image.pixels),
        cuda_dispatch,
    })
}

/// Portrait stages are RGBA8. Refuse a preview-sized or 16-bit export instead
/// of silently enlarging or reducing the original's precision.
pub fn export(request: &Request<'_>, info: &ImageInfo, recipe: &ProcessRecipe, directory: &Path) -> Result<PathBuf> {
    ensure!(recipe.bit_depth == 8, "portrait stages currently support 8-bit export only");
    ensure!((request.source.width, request.source.height) == (info.width, info.height),
        "portrait export requires full-resolution source and inputs");
    if !recipe.format.is_supported() { bail!("portrait export format is unavailable"); }
    let frame = render(request)?;
    let RgbaImage { width, height, pixels } = frame.image;
    crate::recipe::write_export(directory, info, recipe, info.variant as usize,
        std::sync::Arc::unwrap_or_clone(pixels), width, height, |_| {})
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ImageEngine, MockEngine};

    fn uniforms() -> kernels::facial_smooth::Uniforms {
        kernels::facial_smooth::Uniforms {
            legal_alpha: 1.0, dark_circle_alpha: 1.0, diff_factor: 1.0,
            diff_factor2: 0.0, color_threshold_factor: 0.0, mix_alpha: 1.0,
            diff_min_threshold: -1.0, diff_max_threshold: 1.0,
            high_freq_min_threshold: -1.0, high_freq_max_threshold: 1.0,
        }
    }

    #[test]
    fn facial_stage_changes_pixels_and_keeps_the_native_gate() {
        let source = Image::solid(4, 3, [100, 110, 120, 77]);
        let base = Image::solid(4, 3, [50, 60, 70, 255]);
        let residual = Image::solid(4, 3, [180, 190, 200, 255]);
        let mask = [255; 12];
        let request = Request { source: &source, execution: Execution::Cpu,
            operation: Operation::FacialSmooth { pf: 90024, native_value: 0.5,
                base: &base, residual: &residual, mask_r: &mask, mask_b: &mask,
                skin_mask: &mask, uniforms: uniforms() } };
        let frame = render(&request).unwrap();
        assert_ne!(frame.image.pixels.as_ref(), &source.pixels);
        assert!(frame.image.pixels.chunks_exact(4).all(|pixel| pixel[3] == 77));
        let inactive = Request { source: &source, execution: Execution::Cpu,
            operation: Operation::FacialSmooth { pf: 90024, native_value: 0.0,
                base: &base, residual: &residual, mask_r: &[], mask_b: &[],
                skin_mask: &[], uniforms: uniforms() } };
        assert_eq!(render(&inactive).unwrap().image.pixels.as_ref(), &source.pixels);
    }

    #[cfg(windows)]
    #[test]
    #[ignore = "requires an actual NVIDIA CUDA device"]
    fn real_cuda_portrait_bridge_matches_cpu_and_preserves_alpha() {
        let device = backend::cuda::Nvidia::open().expect("real CUDA execution required");
        let source = Image::solid(4, 3, [100, 110, 120, 77]);
        let base = Image::solid(4, 3, [50, 60, 70, 255]);
        let residual = Image::solid(4, 3, [180, 190, 200, 255]);
        let mask = [255; 12];
        let mut request = Request { source: &source, execution: Execution::Cpu,
            operation: Operation::FacialSmooth { pf: 90024, native_value: 0.5,
                base: &base, residual: &residual, mask_r: &mask, mask_b: &mask,
                skin_mask: &mask, uniforms: uniforms() } };
        let cpu = render(&request).unwrap();
        request.execution = Execution::Cuda(&device);
        let gpu = render(&request).unwrap();
        assert_eq!(gpu.image.pixels, cpu.image.pixels);
        let dispatch = gpu.cuda_dispatch.expect("an actual kernel must have launched");
        assert!(!dispatch.device.is_empty());
        println!("portrait CUDA: {} driver={} cc={:?}", dispatch.device, dispatch.driver_version, dispatch.compute_cap);
    }

    #[test]
    fn engine_bridge_preserves_channels_alpha_and_rejects_missing_mask() {
        let engine = MockEngine::new();
        let source = Image::solid(2, 1, [100, 150, 200, 77]);
        let mut request = Request { source: &source, execution: Execution::Cpu,
            operation: Operation::MaskCompose { mask: &[128, 255], mode: kernels::mask_compose::MaskComposeMode::MaskOnly } };
        let frame = engine.render_portrait_stage(&request).unwrap();
        assert_eq!(frame.image.pixels.as_slice(), &[50, 75, 100, 77, 100, 150, 200, 77]);
        assert!(frame.cuda_dispatch.is_none());
        request.operation = Operation::MaskCompose { mask: &[], mode: kernels::mask_compose::MaskComposeMode::MaskOnly };
        assert!(engine.render_portrait_stage(&request).unwrap_err().to_string().contains("mask"));
        let catalog = engine.portrait_capabilities();
        assert!(catalog.iter().all(|row| !row.automatic));
        assert!(catalog.iter().any(|row| row.pf == 90015 && row.engine_bridge));
        assert!(catalog.iter().any(|row| row.pf == 90001 && row.stage == "unavailable"));
        println!("PORTRAIT_CAPABILITIES_JSON={}", serde_json::to_string(&catalog).unwrap());
    }

    #[test]
    fn texture_bridge_preserves_unmasked_pixels_and_rejects_unavailable_positive_branch() {
        let mut source = Image::solid(5, 3, [100, 110, 120, 77]);
        source.pixels[4 * 7] = 110;
        let mask = [255; 15];
        let mut request = Request { source: &source, execution: Execution::Cpu,
            operation: Operation::TextureSmooth { mask_r: &[0; 15], extra_mask_r: &mask, ui: -100.0 } };
        assert_eq!(render(&request).unwrap().image.pixels.as_ref(), &source.pixels);
        request.operation = Operation::TextureSmooth { mask_r: &mask, extra_mask_r: &mask, ui: -100.0 };
        let smoothed = render(&request).unwrap();
        assert_ne!(smoothed.image.pixels.as_ref(), &source.pixels);
        assert!(smoothed.image.pixels.chunks_exact(4).all(|pixel| pixel[3] == 77));
        request.operation = Operation::TextureSmooth { mask_r: &mask, extra_mask_r: &mask, ui: 1.0 };
        assert!(render(&request).unwrap_err().to_string().contains("NotImplemented"));
    }

    #[test]
    fn neck_preview_and_export_use_the_same_pixels_and_reject_precision_loss() {
        let engine = MockEngine::new();
        let source = Image::solid(2, 2, [10, 20, 30, 77]);
        let repair = Image::solid(1, 1, [90, 91, 92, 0]);
        let request = Request { source: &source, execution: Execution::Cpu,
            operation: Operation::NeckPaste { repair: &repair, x: 1, y: 0 } };
        let preview = engine.render_portrait_stage(&request).unwrap();
        let mut info = engine.images()[0].clone();
        info.width = 2; info.height = 2;
        let mut recipe = ProcessRecipe::default();
        recipe.format = crate::ExportFormat::Png;
        recipe.output_sharpening = crate::OutputSharpening::Disabled;
        recipe.icc_profile = "unavailable-synthetic-profile".into();
        let dir = std::env::temp_dir().join(format!("aurora-portrait-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = engine.export_portrait_stage(&request, &info, &recipe, &dir).unwrap();
        assert_eq!(image::open(path).unwrap().to_rgba8().as_raw(), preview.image.pixels.as_ref());
        recipe.bit_depth = 16;
        assert!(engine.export_portrait_stage(&request, &info, &recipe, &dir).unwrap_err().to_string().contains("8-bit"));
        recipe.bit_depth = 8; info.width = 4;
        assert!(engine.export_portrait_stage(&request, &info, &recipe, &dir).unwrap_err().to_string().contains("full-resolution"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
