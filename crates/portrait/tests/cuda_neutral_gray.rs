//! Real NVIDIA coverage for PF30013/PF30014 neutral-gray stages.
//!
//! PF30013 uses a checked-in RGBA8 fixture. PF30014 uses deterministic 512x512
//! same-job planes because the original 3D asset is absent; that proves the
//! pixel consumer, not production asset discovery.

#![cfg(windows)]

use aurora_portrait::cuda::CudaError;
use aurora_portrait::cuda_neutral_gray::{
    require_neutral_gray_3d, NeutralGray3dPlanes, NeutralGrayGpu, NEUTRAL_GRAY_3D_KERNEL,
    NEUTRAL_GRAY_3D_ROI, SKIN_GREY_KERNEL,
};
use aurora_portrait::kernels::skin_adjust::{
    neutral_gray_3d_apply, neutral_gray_3d_degrees, skin_grey_apply, Roi,
    NEUTRAL_GRAY_3D_BYTES, NEUTRAL_GRAY_3D_PIXELS, NEUTRAL_GRAY_3D_SIDE,
};
use aurora_portrait::{Image, Refusal};

const PF30013_FIXTURE: &[u8] =
    aurora_portrait::private_assets::or_empty(aurora_portrait::private_asset!("fixtures/portrait/neutral_gray/pf30013.bin"));

fn gpu() -> Option<NeutralGrayGpu> {
    match NeutralGrayGpu::open() {
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

fn u16_at(bytes: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(bytes[at..at + 2].try_into().unwrap())
}

fn f32_at(bytes: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
}

struct Fixture {
    original: Image,
    current: Image,
    mask: Vec<u8>,
    expected: Vec<u8>,
    roi: Roi,
    alpha: f32,
}

fn fixture() -> Fixture {
    let bytes = PF30013_FIXTURE;
    assert_eq!(&bytes[..4], b"NG13");
    let width = u16_at(bytes, 4) as u32;
    let height = u16_at(bytes, 6) as u32;
    let roi = Roi {
        x: u16_at(bytes, 8) as u32,
        y: u16_at(bytes, 10) as u32,
        width: u16_at(bytes, 12) as u32,
        height: u16_at(bytes, 14) as u32,
    };
    let alpha = f32_at(bytes, 16);
    let rgba = (width * height * 4) as usize;
    let mask_bytes = (width * height) as usize;
    let mut at = 24;
    let original = Image::new(width, height, bytes[at..at + rgba].to_vec()).unwrap();
    at += rgba;
    let current = Image::new(width, height, bytes[at..at + rgba].to_vec()).unwrap();
    at += rgba;
    let mask = bytes[at..at + mask_bytes].to_vec();
    at += mask_bytes;
    let expected = bytes[at..at + rgba].to_vec();
    at += rgba;
    assert_eq!(at, bytes.len(), "fixture has trailing bytes");
    Fixture {
        original,
        current,
        mask,
        expected,
        roi,
        alpha,
    }
}

fn rng(seed: &mut u32) -> u8 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 17;
    *seed ^= *seed << 5;
    *seed as u8
}

fn planes() -> (Vec<u8>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut seed = 0x3001_4000;
    let source = (0..NEUTRAL_GRAY_3D_BYTES)
        .map(|_| rng(&mut seed))
        .collect();
    let mut plane = |scale: f32| {
        (0..NEUTRAL_GRAY_3D_BYTES)
            .map(|_| rng(&mut seed) as f32 / 255.0 * scale)
            .collect::<Vec<_>>()
    };
    let light = plane(1.0);
    let dark = plane(0.8);
    let mut unit = || {
        (0..NEUTRAL_GRAY_3D_PIXELS)
            .map(|_| rng(&mut seed) as f32 / 255.0)
            .collect::<Vec<_>>()
    };
    let alpha = unit();
    let mask = unit();
    (source, light, dark, alpha, mask)
}

#[test]
#[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
fn pf30013_fixture_matches_cpu_and_gpu() {
    let case = fixture();
    let expected = skin_grey_apply(
        &case.original,
        &case.current,
        &case.mask,
        case.roi,
        case.alpha,
    )
    .unwrap();
    assert_eq!(expected.pixels, case.expected, "checked-in CPU fixture drifted");

    let Some(gpu) = gpu() else { return };
    let (actual, dispatch) = gpu
        .skin_grey(
            &case.original,
            &case.current,
            &case.mask,
            case.roi,
            case.alpha,
        )
        .expect("real PF30013 GPU dispatch");
    assert_eq!(dispatch.kernel, SKIN_GREY_KERNEL);
    assert_eq!(actual.pixels, expected.pixels);
    eprintln!(
        "PF30013 {} cc{}.{} driver {} isa {}.{} grid {}x{} block {}x{}",
        dispatch.device,
        dispatch.compute_cap.0,
        dispatch.compute_cap.1,
        dispatch.driver_version,
        dispatch.ptx_isa.0,
        dispatch.ptx_isa.1,
        dispatch.grid.0,
        dispatch.grid.1,
        dispatch.block.0,
        dispatch.block.1,
    );
}

#[test]
fn pf30013_refuses_the_same_entry_contract_as_cpu() {
    let Some(gpu) = gpu() else { return };
    let case = fixture();
    assert_eq!(
        gpu.skin_grey(
            &case.original,
            &case.current,
            &case.mask[..case.mask.len() - 1],
            case.roi,
            case.alpha,
        )
        .err(),
        Some(CudaError::Refused(Refusal::ShapeMismatch("mask")))
    );
    assert_eq!(
        gpu.skin_grey(
            &case.original,
            &case.current,
            &case.mask,
            Roi {
                x: case.current.width,
                y: 0,
                width: 1,
                height: 1,
            },
            case.alpha,
        )
        .err(),
        Some(CudaError::Refused(Refusal::ValueOutOfRange("roi")))
    );
    assert_eq!(
        gpu.skin_grey(
            &case.original,
            &case.current,
            &case.mask,
            case.roi,
            f32::NAN,
        )
        .err(),
        Some(CudaError::Refused(Refusal::ValueOutOfRange("alpha")))
    );
}

#[test]
fn pf30014_full_planes_match_cpu_byte_for_byte() {
    let Some(gpu) = gpu() else { return };
    let (source, light, dark, alpha, mask) = planes();
    let degrees = neutral_gray_3d_degrees(50.0, 0.0, 30.0).unwrap();
    let expected =
        neutral_gray_3d_apply(&source, &light, &dark, &alpha, &mask, degrees).unwrap();
    let (actual, dispatch) = gpu
        .neutral_gray_3d(
            Some(NeutralGray3dPlanes {
                source_bgr: &source,
                light: &light,
                dark: &dark,
                alpha: &alpha,
                mask: &mask,
            }),
            Some(NEUTRAL_GRAY_3D_ROI),
            degrees,
        )
        .expect("real PF30014 GPU dispatch");
    assert_eq!(dispatch.as_ref().unwrap().kernel, NEUTRAL_GRAY_3D_KERNEL);
    assert_eq!(actual, expected);
    let dispatch = dispatch.unwrap();
    eprintln!(
        "PF30014 {} cc{}.{} driver {} isa {}.{} grid {}x{} block {}x{}",
        dispatch.device,
        dispatch.compute_cap.0,
        dispatch.compute_cap.1,
        dispatch.driver_version,
        dispatch.ptx_isa.0,
        dispatch.ptx_isa.1,
        dispatch.grid.0,
        dispatch.grid.1,
        dispatch.block.0,
        dispatch.block.1,
    );
}

#[test]
fn pf30014_inactive_gate_is_cpu_identity_without_dispatch() {
    let Some(gpu) = gpu() else { return };
    let (source, light, dark, alpha, mask) = planes();
    let degrees = neutral_gray_3d_degrees(0.0, 0.0, 100.0).unwrap();
    let (actual, dispatch) = gpu
        .neutral_gray_3d(
            Some(NeutralGray3dPlanes {
                source_bgr: &source,
                light: &light,
                dark: &dark,
                alpha: &alpha,
                mask: &mask,
            }),
            Some(NEUTRAL_GRAY_3D_ROI),
            degrees,
        )
        .unwrap();
    assert_eq!(actual, source);
    assert!(dispatch.is_none());
}

#[test]
fn pf30014_store_is_round_to_nearest_even() {
    let Some(gpu) = gpu() else { return };
    let mut source = vec![0u8; NEUTRAL_GRAY_3D_BYTES];
    source[3] = 1;
    let light = vec![1.0f32 / 255.0; NEUTRAL_GRAY_3D_BYTES];
    let dark = vec![0.0f32; NEUTRAL_GRAY_3D_BYTES];
    let alpha = vec![1.0f32; NEUTRAL_GRAY_3D_PIXELS];
    let mask = vec![1.0f32; NEUTRAL_GRAY_3D_PIXELS];
    let degrees = neutral_gray_3d_degrees(50.0, 50.0, 0.0).unwrap();
    let expected =
        neutral_gray_3d_apply(&source, &light, &dark, &alpha, &mask, degrees).unwrap();
    assert_eq!(&expected[..6], &[0, 0, 0, 2, 0, 0]);
    let (actual, _) = gpu
        .neutral_gray_3d(
            Some(NeutralGray3dPlanes {
                source_bgr: &source,
                light: &light,
                dark: &dark,
                alpha: &alpha,
                mask: &mask,
            }),
            Some(NEUTRAL_GRAY_3D_ROI),
            degrees,
        )
        .unwrap();
    assert_eq!(&actual[..6], &expected[..6]);
}

#[test]
fn pf30014_missing_asset_or_roi_is_a_typed_refusal() {
    let Some(gpu) = gpu() else { return };
    let degrees = neutral_gray_3d_degrees(50.0, 50.0, 0.0).unwrap();
    assert_eq!(
        gpu.neutral_gray_3d(None, Some(NEUTRAL_GRAY_3D_ROI), degrees)
            .err(),
        Some(CudaError::Refused(Refusal::MissingInput(
            "neutral_gray_3d_asset"
        )))
    );

    let (source, light, dark, alpha, mask) = planes();
    let planes = NeutralGray3dPlanes {
        source_bgr: &source,
        light: &light,
        dark: &dark,
        alpha: &alpha,
        mask: &mask,
    };
    assert_eq!(
        gpu.neutral_gray_3d(Some(planes), None, degrees).err(),
        Some(CudaError::Refused(Refusal::MissingInput(
            "neutral_gray_3d_roi"
        )))
    );
    assert_eq!(
        require_neutral_gray_3d(
            Some(planes),
            Some(Roi {
                x: 0,
                y: 0,
                width: (NEUTRAL_GRAY_3D_SIDE - 1) as u32,
                height: NEUTRAL_GRAY_3D_SIDE as u32,
            }),
        )
        .err(),
        Some(Refusal::ValueOutOfRange("neutral_gray_3d_roi"))
    );
}

#[test]
fn adapter_end_to_end_timing() {
    let Some(gpu) = gpu() else { return };
    let side = NEUTRAL_GRAY_3D_SIDE as u32;
    let pixels = (side * side) as usize;
    let mut seed = 0x5120_u32;
    let original = Image::new(
        side,
        side,
        (0..pixels * 4).map(|_| rng(&mut seed)).collect(),
    )
    .unwrap();
    let current = Image::new(
        side,
        side,
        (0..pixels * 4).map(|_| rng(&mut seed)).collect(),
    )
    .unwrap();
    let mask = (0..pixels).map(|_| rng(&mut seed)).collect::<Vec<_>>();
    let full = Roi {
        x: 0,
        y: 0,
        width: side,
        height: side,
    };
    let mut pf30013 = Vec::with_capacity(30);
    for _ in 0..30 {
        let start = std::time::Instant::now();
        gpu.skin_grey(&original, &current, &mask, full, 0.5)
            .unwrap();
        pf30013.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let (source, light, dark, alpha, mask) = planes();
    let degrees = neutral_gray_3d_degrees(50.0, 30.0, 20.0).unwrap();
    let mut pf30014 = Vec::with_capacity(30);
    for _ in 0..30 {
        let start = std::time::Instant::now();
        gpu.neutral_gray_3d(
            Some(NeutralGray3dPlanes {
                source_bgr: &source,
                light: &light,
                dark: &dark,
                alpha: &alpha,
                mask: &mask,
            }),
            Some(NEUTRAL_GRAY_3D_ROI),
            degrees,
        )
        .unwrap();
        pf30014.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    for (name, mut samples) in [("PF30013", pf30013), ("PF30014", pf30014)] {
        samples.sort_by(f64::total_cmp);
        let p50 = samples[samples.len() / 2];
        let p95 = samples[(samples.len() * 95).div_ceil(100) - 1];
        eprintln!("{name} 512x512 adapter P50={p50:.4} ms P95={p95:.4} ms");
    }
}
