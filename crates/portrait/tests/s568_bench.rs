//! S568 generic staging-cache benchmark.
//!
//! Run with `PORTRAIT_CUDA_REQUIRED=1 S568_OUT=<path> cargo test --release
//! -p aurora-portrait --test s568_bench -- --ignored --nocapture`.

#![cfg(windows)]

use aurora_portrait::cuda::Dispatch;
use aurora_portrait::cuda_makeup::MakeupLayer;
use aurora_portrait::cuda_skin_models::SkinModels;
use aurora_portrait::kernels::{makeup_material, skin_models};
use aurora_portrait::Image;
use std::hint::black_box;
use std::time::{Duration, Instant};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }

    fn byte(&mut self) -> u8 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u8
    }

    fn image(&mut self, width: u32, height: u32) -> Image {
        Image::new(
            width,
            height,
            (0..width * height * 4).map(|_| self.byte()).collect(),
        )
        .unwrap()
    }
}

fn percentile(samples: &[Duration], percent: usize) -> f64 {
    let index = ((samples.len() - 1) * percent).div_ceil(100);
    samples[index].as_secs_f64() * 1e3
}

fn differing_bytes(expected: &Image, actual: &Image) -> usize {
    expected
        .pixels
        .iter()
        .zip(&actual.pixels)
        .filter(|(left, right)| left != right)
        .count()
}

fn stats(samples: &mut [Duration]) -> (f64, f64, f64, f64) {
    samples.sort();
    (
        percentile(samples, 50),
        percentile(samples, 95),
        samples[0].as_secs_f64() * 1e3,
        samples[samples.len() - 1].as_secs_f64() * 1e3,
    )
}

fn skin_run(
    gpu: &SkinModels,
    source: &Image,
    mask: &[u8],
    extra: &[u8],
) -> (Image, Dispatch) {
    let (output, dispatches) = gpu
        .texture_apply(source, mask, extra, -25.0)
        .expect("real CUDA PF8069 dispatch");
    (output, dispatches[0].clone())
}

fn makeup_run(
    gpu: &MakeupLayer,
    source: &Image,
    material: &Image,
    mask: &Image,
    mask_uv: &[[f32; 2]],
    material_uv: &[[f32; 2]],
) -> (Image, Dispatch) {
    let params = makeup_material::ColorMaterialParams {
        mix_color: [255, 255, 255, 220],
        opacity_percent: 100.0,
        degree01: 1.0,
        blend_mode: makeup_material::BlendMode::SoftLight,
        mask_channel: makeup_material::MaskChannel::Rgba,
        blend_alpha: true,
    };
    let (output, dispatches) = gpu
        .apply_image_layer_dual(source, material, mask, mask_uv, material_uv, params)
        .expect("real CUDA makeup dispatch");
    (output, dispatches[0].clone())
}

#[test]
#[ignore = "manual real-GPU P50/P95 profile"]
fn s568_bench() {
    let side = 512u32;
    let pixels = (side * side) as usize;
    let mut rng = Rng::new(568);
    let source = rng.image(side, side);
    let mask = (0..pixels).map(|_| rng.byte()).collect::<Vec<_>>();
    let extra = (0..pixels).map(|_| rng.byte()).collect::<Vec<_>>();
    let material = rng.image(64, 64);
    let owner_mask = rng.image(64, 64);
    let mask_uv = (0..pixels)
        .map(|_| [rng.byte() as f32 / 255.0, rng.byte() as f32 / 255.0])
        .collect::<Vec<_>>();
    let material_uv = (0..pixels)
        .map(|_| [rng.byte() as f32 / 255.0, rng.byte() as f32 / 255.0])
        .collect::<Vec<_>>();

    let skin_expected = skin_models::texture_surface_blur(&source, &mask, &extra, -25.0)
        .expect("CPU PF8069 oracle");
    let makeup_params = makeup_material::ColorMaterialParams {
        mix_color: [255, 255, 255, 220],
        opacity_percent: 100.0,
        degree01: 1.0,
        blend_mode: makeup_material::BlendMode::SoftLight,
        mask_channel: makeup_material::MaskChannel::Rgba,
        blend_alpha: true,
    };
    let makeup_expected = makeup_material::apply_image_layer_dual(
        &source,
        &material,
        &owner_mask,
        &mask_uv,
        &material_uv,
        makeup_params,
    )
    .expect("CPU makeup oracle");

    let skin_gpu = SkinModels::open().expect("PORTRAIT_CUDA_REQUIRED=1");
    let skin_started = Instant::now();
    let (skin_cold_output, skin_dispatch) = skin_run(&skin_gpu, &source, &mask, &extra);
    let skin_cold_ms = skin_started.elapsed().as_secs_f64() * 1e3;
    assert_eq!(differing_bytes(&skin_expected, &skin_cold_output), 0);
    for _ in 0..5 {
        black_box(skin_run(&skin_gpu, &source, &mask, &extra));
    }
    let mut skin_samples = Vec::with_capacity(100);
    for _ in 0..100 {
        let started = Instant::now();
        let (output, _) = skin_run(&skin_gpu, &source, &mask, &extra);
        skin_samples.push(started.elapsed());
        assert_eq!(differing_bytes(&skin_expected, &output), 0);
    }

    let makeup_gpu = MakeupLayer::open().expect("PORTRAIT_CUDA_REQUIRED=1");
    let makeup_started = Instant::now();
    let (makeup_cold_output, makeup_dispatch) = makeup_run(
        &makeup_gpu,
        &source,
        &material,
        &owner_mask,
        &mask_uv,
        &material_uv,
    );
    let makeup_cold_ms = makeup_started.elapsed().as_secs_f64() * 1e3;
    assert_eq!(differing_bytes(&makeup_expected, &makeup_cold_output), 0);
    for _ in 0..5 {
        black_box(makeup_run(
            &makeup_gpu,
            &source,
            &material,
            &owner_mask,
            &mask_uv,
            &material_uv,
        ));
    }
    let mut makeup_samples = Vec::with_capacity(100);
    for _ in 0..100 {
        let started = Instant::now();
        let (output, _) = makeup_run(
            &makeup_gpu,
            &source,
            &material,
            &owner_mask,
            &mask_uv,
            &material_uv,
        );
        makeup_samples.push(started.elapsed());
        assert_eq!(differing_bytes(&makeup_expected, &output), 0);
    }

    let (skin_p50, skin_p95, skin_min, skin_max) = stats(&mut skin_samples);
    let (makeup_p50, makeup_p95, makeup_min, makeup_max) = stats(&mut makeup_samples);
    let json = format!(
        concat!(
            "{{\"schema\":\"aurora/s568-generic-buffer-cache/1\",",
            "\"device\":\"{}\",\"compute_cap\":\"{}.{}\",\"driver_version\":{},",
            "\"ptx_isa\":\"{}.{}\",\"width\":{},\"height\":{},\"samples\":100,",
            "\"skin_models\":{{\"p50_ms\":{:.6},\"p95_ms\":{:.6},",
            "\"min_ms\":{:.6},\"max_ms\":{:.6},\"cold_ms\":{:.6},",
            "\"differing_bytes\":0,\"kernel\":\"{}\"}},",
            "\"makeup\":{{\"p50_ms\":{:.6},\"p95_ms\":{:.6},",
            "\"min_ms\":{:.6},\"max_ms\":{:.6},\"cold_ms\":{:.6},",
            "\"differing_bytes\":0,\"kernel\":\"{}\"}}}}"
        ),
        skin_dispatch.device,
        skin_dispatch.compute_cap.0,
        skin_dispatch.compute_cap.1,
        skin_dispatch.driver_version,
        skin_dispatch.ptx_isa.0,
        skin_dispatch.ptx_isa.1,
        side,
        side,
        skin_p50,
        skin_p95,
        skin_min,
        skin_max,
        skin_cold_ms,
        skin_dispatch.kernel,
        makeup_p50,
        makeup_p95,
        makeup_min,
        makeup_max,
        makeup_cold_ms,
        makeup_dispatch.kernel,
    );
    println!("{json}");
    if let Ok(path) = std::env::var("S568_OUT") {
        std::fs::write(path, format!("{json}\n")).expect("write S568_OUT");
    }
}
