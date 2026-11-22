//! Real NVIDIA coverage for the PF21025/21069/21070/21071 wrinkle family.

#![cfg(windows)]

use std::time::{Duration, Instant};

use aurora_portrait::cuda::CudaError;
use aurora_portrait::cuda_wrinkle::{reference, WrinkleGpu, ROUTED_PFS};
use aurora_portrait::kernels::skin_adjust::{
    Roi, StoreMode, WrinkleUniforms,
};
use aurora_portrait::Image;

#[derive(Clone)]
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
    }

    fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 32) as u32
    }

    fn byte(&mut self) -> u8 {
        (self.next_u32() >> 24) as u8
    }
}

fn gpu() -> Option<WrinkleGpu> {
    match WrinkleGpu::open() {
        Ok(gpu) => Some(gpu),
        Err(CudaError::Unavailable(reason)) => {
            eprintln!("SKIP_NO_NVIDIA_DRIVER: {reason}");
            assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but unavailable: {reason}"
            );
            None
        }
        Err(broken) => panic!("NVIDIA path present but broken: {broken:?}"),
    }
}

fn active_uniforms() -> WrinkleUniforms {
    WrinkleUniforms {
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
    }
}

fn percentile(samples: &mut [Duration], percent: usize) -> f64 {
    samples.sort_unstable();
    let index = ((samples.len() - 1) * percent + 99) / 100;
    samples[index].as_secs_f64() * 1_000.0
}

#[test]
fn explicit_store_modes_diverge_on_a_real_dispatch() {
    let Some(gpu) = gpu() else { return };
    let source = Image::solid(1, 1, [0, 0, 0, 77]);
    let base = Image::solid(1, 1, [0, 0, 0, 0]);
    let residual = Image::solid(1, 1, [255, 255, 255, 0]);
    let mask = [0, 0, 255, 255];
    let skin = [255];
    let roi = Roi {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
    };
    let uniforms = WrinkleUniforms {
        legal_alpha: 0.5,
        dark_circle_alpha: 0.0,
        ..active_uniforms()
    };

    let (truncated, trunc_dispatch) = gpu
        .apply_route(
            21025,
            &source,
            &base,
            &residual,
            &mask,
            &skin,
            roi,
            1.0,
            uniforms,
            StoreMode::TruncU8,
        )
        .expect("truncating GPU dispatch");
    let (rounded, round_dispatch) = gpu
        .apply_route(
            21025,
            &source,
            &base,
            &residual,
            &mask,
            &skin,
            roi,
            1.0,
            uniforms,
            StoreMode::Round,
        )
        .expect("rounding GPU dispatch");

    assert_eq!(truncated.pixel(0, 0), [127, 127, 127, 77]);
    assert_eq!(rounded.pixel(0, 0), [128, 128, 128, 77]);
    assert_eq!(trunc_dispatch[0].kernel, aurora_portrait::cuda_wrinkle::KERNEL);
    assert_eq!(round_dispatch[0].kernel, aurora_portrait::cuda_wrinkle::KERNEL);
    eprintln!(
        "store_mode_discriminator: trunc={:?} round={:?} device={} cc={:?} driver={} ptx_isa={:?} grid={:?}",
        truncated.pixel(0, 0),
        rounded.pixel(0, 0),
        round_dispatch[0].device,
        round_dispatch[0].compute_cap,
        round_dispatch[0].driver_version,
        round_dispatch[0].ptx_isa,
        round_dispatch[0].grid
    );
}

#[test]
fn four_pf_routes_match_cpu_inside_roi() {
    let Some(gpu) = gpu() else { return };
    let (width, height) = (37, 29);
    let pixels = width * height;
    let mut rng = Rng::new(0x514);
    let source = Image::new(
        width,
        height,
        (0..pixels * 4).map(|_| rng.byte()).collect(),
    )
    .unwrap();
    let base = Image::new(
        width,
        height,
        (0..pixels * 4).map(|_| rng.byte()).collect(),
    )
    .unwrap();
    let residual = Image::new(
        width,
        height,
        (0..pixels * 4).map(|_| rng.byte()).collect(),
    )
    .unwrap();
    let mask: Vec<u8> = (0..pixels * 4).map(|_| rng.byte()).collect();
    let skin: Vec<u8> = (0..pixels).map(|_| rng.byte()).collect();
    let roi = Roi {
        x: 5,
        y: 4,
        width: 23,
        height: 17,
    };
    let uniforms = active_uniforms();

    for pf in ROUTED_PFS {
        let expected = reference(
            pf, &source, &base, &residual, &mask, &skin, roi, 50.0, uniforms,
        )
        .unwrap();
        let (actual, dispatch) = gpu
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
        assert_eq!(actual, expected, "PF{pf} truncating GPU/CPU mismatch");

        let (rounded, _) = gpu
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
                StoreMode::Round,
            )
            .unwrap();
        let mut differing = 0usize;
        let mut over_one = 0usize;
        for y in roi.y..roi.y + roi.height {
            for x in roi.x..roi.x + roi.width {
                let left = expected.pixel(x, y);
                let right = rounded.pixel(x, y);
                for channel in 0..3 {
                    let delta = left[channel].abs_diff(right[channel]);
                    differing += usize::from(delta != 0);
                    over_one += usize::from(delta > 1);
                }
                assert_eq!(left[3], right[3], "PF{pf} alpha diverged");
            }
        }
        assert_eq!(over_one, 0, "PF{pf} exceeded the one-level store delta");
        assert!(differing > 0, "PF{pf} silently collapsed round into trunc");
        eprintln!(
            "PF{pf} trunc_vs_round_differing_rgb={differing} over_one_lsb={over_one} kernel={} device={} ptx_isa={:?} roi={}x{}",
            dispatch[0].kernel,
            dispatch[0].device,
            dispatch[0].ptx_isa,
            roi.width,
            roi.height
        );
    }
}

#[test]
fn route_timing_is_reported_from_real_dispatch() {
    let Some(gpu) = gpu() else { return };
    let (width, height) = (256, 256);
    let pixels = width * height;
    let mut rng = Rng::new(0x515);
    let source = Image::new(
        width,
        height,
        (0..pixels * 4).map(|_| rng.byte()).collect(),
    )
    .unwrap();
    let base = Image::new(
        width,
        height,
        (0..pixels * 4).map(|_| rng.byte()).collect(),
    )
    .unwrap();
    let residual = Image::new(
        width,
        height,
        (0..pixels * 4).map(|_| rng.byte()).collect(),
    )
    .unwrap();
    let mask: Vec<u8> = (0..pixels * 4).map(|_| rng.byte()).collect();
    let skin: Vec<u8> = (0..pixels).map(|_| rng.byte()).collect();
    let roi = Roi {
        x: 32,
        y: 32,
        width: 160,
        height: 160,
    };
    let mut samples = Vec::with_capacity(30);
    for _ in 0..30 {
        let start = Instant::now();
        let (output, dispatch) = gpu
            .apply_route(
                21025,
                &source,
                &base,
                &residual,
                &mask,
                &skin,
                roi,
                50.0,
                active_uniforms(),
                StoreMode::Round,
            )
            .unwrap();
        std::hint::black_box(output);
        assert_eq!(
            dispatch[0].kernel,
            aurora_portrait::cuda_wrinkle::KERNEL
        );
        samples.push(start.elapsed());
    }
    let p50 = percentile(&mut samples.clone(), 50);
    let p95 = percentile(&mut samples, 95);
    eprintln!(
        "route_adapter_end_to_end_roi=160x160 samples=30 p50_ms={p50:.6} p95_ms={p95:.6}"
    );
}
