//! PF90033 suit makeup: an ordered list of makeup passes over one source.
//!
//! `config.json.Effects[]` then each `effect.json.Effects[]` fix the order. The
//! per-pass factor is applied exactly once:
//! `degree * opacity/100 * shelter * region * material.a * mask`.
//! `LipColor` is its own cell and is refused here rather than blended as Normal.

use crate::kernels::makeup_material::{
    color_makeup, image_makeup, MakeupLayer,
};
use crate::{Image, Refusal};

pub enum SuitPassKind<'a> {
    /// `BlendColorLayerWithMask`: the blend colour is `layer.mix_color`.
    Color,
    /// `BlendImageLayerWithMask`: the blend colour is sampled from a material.
    Image {
        material: &'a Image,
        material_uv: &'a [[f32; 2]],
    },
}

pub struct SuitPass<'a> {
    pub kind: SuitPassKind<'a>,
    pub mask: &'a Image,
    pub mask_uv: &'a [[f32; 2]],
    pub layer: MakeupLayer,
    pub shelter: f32,
    pub region: f32,
}

fn folded(layer: MakeupLayer, shelter: f32, region: f32) -> Result<MakeupLayer, Refusal> {
    if !shelter.is_finite() || !(0.0..=1.0).contains(&shelter) {
        return Err(Refusal::ValueOutOfRange("suit shelter"));
    }
    if !region.is_finite() || !(0.0..=1.0).contains(&region) {
        return Err(Refusal::ValueOutOfRange("suit region"));
    }
    let degree = layer.degree01 * shelter * region;
    if !degree.is_finite() || !(0.0..=1.0).contains(&degree) {
        return Err(Refusal::ValueOutOfRange("suit degree"));
    }
    Ok(MakeupLayer { degree01: degree, ..layer })
}

/// Applies every pass in order. Each pass sees the previous pass's output.
pub fn compose(source: &Image, passes: &[SuitPass<'_>]) -> Result<Image, Refusal> {
    source.validate()?;
    let mut current = source.clone();
    for pass in passes {
        let layer = folded(pass.layer, pass.shelter, pass.region)?;
        current = match &pass.kind {
            SuitPassKind::Color => {
                color_makeup(&current, pass.mask, pass.mask_uv, layer)?
            }
            SuitPassKind::Image { material, material_uv } => image_makeup(
                &current,
                material,
                pass.mask,
                pass.mask_uv,
                material_uv,
                layer,
            )?,
        };
    }
    Ok(current)
}

/// A `LipColor` pass has no `FilterType` and its own cell; callers must run it
/// separately instead of folding it into this list.
pub fn lip_color_is_open() -> Refusal {
    Refusal::MissingInput("suit_lipcolor_cell")
}

/// NVIDIA twin of [`compose`]. A colour pass becomes a 1×1 material so both
/// kinds share the recovered image-layer kernel; there is no CPU fallback.
#[cfg(windows)]
pub fn compose_gpu(
    source: &Image,
    passes: &[SuitPass<'_>],
    gpu: &crate::cuda_makeup::MakeupLayer,
) -> Result<Image, crate::cuda::CudaError> {
    use crate::cuda::CudaError;
    source.validate().map_err(CudaError::Refused)?;
    let mut current = source.clone();
    for pass in passes {
        let layer = folded(pass.layer, pass.shelter, pass.region)
            .map_err(CudaError::Refused)?;
        let params = layer.params_of().map_err(CudaError::Refused)?;
        let pixels = current.pixel_count();
        let flat_uv = vec![[0.5f32, 0.5f32]; pixels];
        current = match &pass.kind {
            SuitPassKind::Image { material, material_uv } => gpu
                .apply_image_layer_dual(
                    &current, material, pass.mask, pass.mask_uv, material_uv, params,
                )?
                .0,
            SuitPassKind::Color => {
                // The colour layer folds `MixColor.a` into mix_alpha, so the
                // synthetic material must be fully opaque to avoid counting it twice.
                let [r, g, b, _] = layer.mix_color;
                let material = Image::solid(1, 1, [r, g, b, 255]);
                gpu.apply_image_layer_dual(
                    &current, &material, pass.mask, pass.mask_uv, &flat_uv, params,
                )?
                .0
            }
        };
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::makeup_material::MaskChannel;

    fn solid(rgba: [u8; 4]) -> Image {
        Image::solid(1, 1, rgba)
    }

    fn layer(mix_color: [u8; 4], degree: f32) -> MakeupLayer {
        MakeupLayer {
            mix_color,
            opacity_percent: 100.0,
            degree01: degree,
            filter_type: 10,
            mask_channel: MaskChannel::Rgba,
        }
    }

    fn uv() -> [[f32; 2]; 1] {
        [[0.5, 0.5]]
    }

    #[test]
    fn pass_order_changes_the_result() {
        let source = solid([10, 20, 30, 255]);
        let mask = solid([255, 255, 255, 255]);
        let uvs = uv();
        let red = layer([255, 0, 0, 255], 1.0);
        let blue = layer([0, 0, 255, 255], 1.0);
        let passes = [
            SuitPass {
                kind: SuitPassKind::Color,
                mask: &mask,
                mask_uv: &uvs,
                layer: red,
                shelter: 1.0,
                region: 1.0,
            },
            SuitPass {
                kind: SuitPassKind::Color,
                mask: &mask,
                mask_uv: &uvs,
                layer: blue,
                shelter: 1.0,
                region: 1.0,
            },
        ];
        assert_eq!(compose(&source, &passes).unwrap().pixels, [0, 0, 255, 255]);
        let reversed = [
            SuitPass { kind: SuitPassKind::Color, mask: &mask, mask_uv: &uvs,
                layer: blue, shelter: 1.0, region: 1.0 },
            SuitPass { kind: SuitPassKind::Color, mask: &mask, mask_uv: &uvs,
                layer: red, shelter: 1.0, region: 1.0 },
        ];
        assert_eq!(compose(&source, &reversed).unwrap().pixels, [255, 0, 0, 255]);
    }

    #[test]
    fn factor_is_folded_once_and_gated() {
        let source = solid([0, 0, 0, 255]);
        let mask = solid([255, 255, 255, 255]);
        let uvs = uv();
        let pass = SuitPass {
            kind: SuitPassKind::Color,
            mask: &mask,
            mask_uv: &uvs,
            layer: layer([255, 255, 255, 255], 0.5),
            shelter: 0.5,
            region: 0.5,
        };
        // 0.5 * 0.5 * 0.5 = 0.125 -> 31.875 -> 32 after nearest-even.
        assert_eq!(compose(&source, &[pass]).unwrap().pixels, [32, 32, 32, 255]);
        let quiet = SuitPass {
            kind: SuitPassKind::Color,
            mask: &mask,
            mask_uv: &uvs,
            layer: layer([255, 255, 255, 255], 0.005),
            shelter: 1.0,
            region: 1.0,
        };
        assert_eq!(compose(&source, &[quiet]).unwrap().pixels, source.pixels);
    }

    #[test]
    fn image_pass_samples_its_own_material_uv() {
        let source = solid([10, 20, 30, 77]);
        let material = solid([255, 0, 0, 255]);
        let mask = solid([255, 255, 255, 255]);
        let uvs = uv();
        let pass = SuitPass {
            kind: SuitPassKind::Image { material: &material, material_uv: &uvs },
            mask: &mask,
            mask_uv: &uvs,
            layer: layer([255, 255, 255, 255], 1.0),
            shelter: 1.0,
            region: 1.0,
        };
        assert_eq!(compose(&source, &[pass]).unwrap().pixels, [255, 0, 0, 77]);
    }

    #[test]
    fn lip_color_is_not_silently_composed() {
        assert_eq!(lip_color_is_open(), Refusal::MissingInput("suit_lipcolor_cell"));
    }

    #[cfg(windows)]
    #[test]
    fn gpu_compose_matches_the_cpu_order() {
        let Ok(gpu) = crate::cuda_makeup::MakeupLayer::open() else {
            assert!(
                std::env::var_os("PIXCAKE_CUDA_REQUIRED").is_none(),
                "CUDA required but unavailable"
            );
            return;
        };
        let (width, height) = (8u32, 5u32);
        let pixels = (width * height) as usize;
        let source = Image::new(
            width,
            height,
            (0..pixels * 4).map(|i| ((i * 37 + 11) % 256) as u8).collect(),
        )
        .unwrap();
        let mask = Image::new(
            width,
            height,
            (0..pixels * 4).map(|i| ((i * 13 + 5) % 256) as u8).collect(),
        )
        .unwrap();
        let material = Image::solid(2, 2, [200, 40, 90, 220]);
        let mask_uv: Vec<[f32; 2]> = (0..pixels)
            .map(|i| [(i % width as usize) as f32 / width as f32, (i / width as usize) as f32 / height as f32])
            .collect();
        let material_uv = mask_uv.clone();
        let passes = [
            SuitPass {
                kind: SuitPassKind::Color,
                mask: &mask,
                mask_uv: &mask_uv,
                layer: layer([250, 120, 30, 200], 0.8),
                shelter: 0.9,
                region: 0.7,
            },
            SuitPass {
                kind: SuitPassKind::Image { material: &material, material_uv: &material_uv },
                mask: &mask,
                mask_uv: &mask_uv,
                layer: layer([255, 255, 255, 255], 1.0),
                shelter: 1.0,
                region: 1.0,
            },
        ];
        let cpu = compose(&source, &passes).unwrap();
        let gpu_out = compose_gpu(&source, &passes, &gpu).unwrap();
        let differing = cpu.pixels.iter().zip(&gpu_out.pixels).filter(|(a, b)| a != b).count();
        assert_eq!(differing, 0);
    }
}
