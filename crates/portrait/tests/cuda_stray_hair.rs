// Real NVIDIA coverage for the PF90088 final stray-hair composite.

#![cfg(windows)]

use aurora_portrait::cuda::CudaError;
use aurora_portrait::cuda_stray_hair::{Blend, Params, KERNEL};
use aurora_portrait::kernels::hair_body;
use aurora_portrait::{Image, Refusal};
use std::time::Instant;

fn gpu() -> Option<Blend> {
    match Blend::open() {
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

fn params() -> Params {
    Params {
        ui: 100.0,
        native_degree: 0.5,
        native_person_area: 1.0,
        native_background_area: 0.5,
    }
}

fn fixture() -> (Image, Image, Vec<u8>, Vec<u8>, Vec<u8>) {
    let mut source = Vec::with_capacity(5 * 4 * 4);
    let mut repair = Vec::with_capacity(5 * 4 * 4);
    for pixel in 0..20u32 {
        source.extend_from_slice(&[
            (pixel * 13 + 2) as u8,
            (pixel * 11 + 20) as u8,
            (pixel * 7 + 30) as u8,
            (17 + pixel * 9) as u8,
        ]);
        repair.extend_from_slice(&[
            (220 - pixel * 3) as u8,
            (220 - pixel * 5) as u8,
            (130 + pixel * 2) as u8,
            (250 - pixel) as u8,
        ]);
    }
    source[0..4].copy_from_slice(&[2, 20, 30, 17]);
    repair[0..4].copy_from_slice(&[3, 220, 130, 250]);

    let raw = vec![
        255, 128, 0, 255, 64, 200, 17, 33, 99, 245, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        10,
    ];
    let person = vec![
        255, 128, 255, 0, 200, 64, 0, 255, 12, 25, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255,
    ];
    let background =
        vec![0, 255, 255, 255, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
    (
        Image::new(5, 4, source).unwrap(),
        Image::new(5, 4, repair).unwrap(),
        raw,
        person,
        background,
    )
}

#[test]
fn fixture_matches_the_cpu_lane_and_preserves_alpha() {
    let Some(gpu) = gpu() else { return };
    let (source, repair, raw, person, background) = fixture();
    let expected = hair_body::stray_hair(
        &source,
        100.0,
        0.5,
        1.0,
        0.5,
        Some(&repair),
        Some(&raw),
        Some(&person),
        Some(&background),
    )
    .unwrap();
    let (actual, dispatches) = gpu
        .apply(
            &source,
            params(),
            Some(&repair),
            Some(&raw),
            Some(&person),
            Some(&background),
        )
        .expect("real GPU dispatch");

    assert_eq!(actual.pixels, expected.pixels);
    assert_eq!(actual.pixel(0, 0), [3, 120, 80, 17]);
    assert_eq!(dispatches.len(), 1);
    assert_eq!(dispatches[0].kernel, KERNEL);
    for index in 0..source.pixel_count() {
        assert_eq!(actual.pixels[index * 4 + 3], source.pixels[index * 4 + 3]);
    }
    eprintln!(
        "PF90088 kernel={} device={} cc={:?} driver={} ptx_isa={:?}",
        dispatches[0].kernel,
        dispatches[0].device,
        dispatches[0].compute_cap,
        dispatches[0].driver_version,
        dispatches[0].ptx_isa
    );
}

#[test]
fn absent_area_masks_default_to_all_area() {
    let Some(gpu) = gpu() else { return };
    let (source, repair, raw, _, _) = fixture();
    let expected = hair_body::stray_hair(
        &source,
        100.0,
        0.5,
        1.0,
        0.5,
        Some(&repair),
        Some(&raw),
        None,
        None,
    )
    .unwrap();
    let (actual, _) = gpu
        .apply(
            &source,
            params(),
            Some(&repair),
            Some(&raw),
            None,
            None,
        )
        .expect("real GPU dispatch");
    assert_eq!(actual.pixels, expected.pixels);
}

#[test]
fn missing_same_job_producers_are_typed_before_launch() {
    let Some(gpu) = gpu() else { return };
    let (source, repair, raw, _, _) = fixture();
    assert_eq!(
        gpu.apply(&source, params(), None, Some(&raw), None, None)
            .err(),
        Some(CudaError::Refused(Refusal::MissingInput(
            "PF90088_HAIRFLY_V1"
        )))
    );
    assert_eq!(
        gpu.apply(&source, params(), Some(&repair), None, None, None)
            .err(),
        Some(CudaError::Refused(Refusal::MissingInput(
            "PF90088_RAW_STRAY_HAIR_MASK"
        )))
    );
}

#[test]
fn large_frame_matches_and_reports_per_call_latency() {
    let Some(gpu) = gpu() else { return };
    let (width, height) = (1280u32, 720u32);
    let pixels = (width * height) as usize;
    let source = Image::new(
        width,
        height,
        (0..pixels * 4)
            .map(|index| ((index * 37 + index / 17 + 11) % 256) as u8)
            .collect(),
    )
    .unwrap();
    let repair = Image::new(
        width,
        height,
        (0..pixels * 4)
            .map(|index| ((index * 13 + index / 29 + 5) % 256) as u8)
            .collect(),
    )
    .unwrap();
    let raw: Vec<u8> = (0..pixels)
        .map(|index| ((index * 29 + index / 7 + 3) % 256) as u8)
        .collect();
    let person: Vec<u8> = (0..pixels)
        .map(|index| ((index * 11 + 19) % 256) as u8)
        .collect();
    let background: Vec<u8> = (0..pixels)
        .map(|index| ((index * 5 + 211) % 256) as u8)
        .collect();
    let expected = hair_body::stray_hair(
        &source,
        100.0,
        0.5,
        1.0,
        0.5,
        Some(&repair),
        Some(&raw),
        Some(&person),
        Some(&background),
    )
    .unwrap();

    let mut samples = Vec::new();
    for _ in 0..12 {
        let started = Instant::now();
        let (actual, _) = gpu
            .apply(
                &source,
                params(),
                Some(&repair),
                Some(&raw),
                Some(&person),
                Some(&background),
            )
            .expect("large fixture GPU dispatch");
        samples.push(started.elapsed());
        assert_eq!(actual.pixels, expected.pixels);
    }
    samples.sort_unstable();
    let p50 = samples[samples.len() / 2];
    let p95 = samples[(samples.len() * 95).div_ceil(100) - 1];
    eprintln!(
        "PF90088 perf 1280x720 samples=12 allocation_upload_launch_readback=true p50_us={} p95_us={}",
        p50.as_micros(),
        p95.as_micros()
    );
}
