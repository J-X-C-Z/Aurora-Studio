#![cfg(windows)]

use aurora_portrait::cuda::CudaError;
use aurora_portrait::cuda_lip_line::LipLineGpu;
use aurora_portrait::kernels::skin_color::{lip_line_roi, shared_min_mask_composite, Roi};
use aurora_portrait::Image;

fn gpu() -> Option<LipLineGpu> {
    match LipLineGpu::open() {
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

#[test]
fn exact_fixture_roi_and_alpha() {
    let points = [
        (10.0, 10.0),
        (20.0, 10.0),
        (20.0, 20.0),
        (10.0, 20.0),
        (15.0, 15.0),
        (12.0, 14.0),
        (18.0, 14.0),
        (12.0, 18.0),
        (18.0, 18.0),
        (15.0, 10.0),
        (15.0, 20.0),
    ];
    assert_eq!(
        lip_line_roi(&points, 100, 100).unwrap(),
        Roi {
            x: 9,
            y: 9,
            w: 12,
            h: 12,
        }
    );

    let Some(gpu) = gpu() else { return };
    let source = Image::solid(1, 1, [10, 20, 30, 77]);
    let repair = Image::solid(1, 1, [110, 120, 130, 9]);
    let expected = shared_min_mask_composite(&source, &repair, &[255], &[128]).unwrap();
    let (actual, dispatch) = gpu.apply(&source, &repair, &[255], &[128]).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.pixel(0, 0), [60, 70, 80, 77]);
    assert_eq!(dispatch[0].kernel, "beard_blend_kernel");
    eprintln!(
        "PF90118 kernel={} device={} cc={:?} driver={} ptx_isa={:?}",
        dispatch[0].kernel,
        dispatch[0].device,
        dispatch[0].compute_cap,
        dispatch[0].driver_version,
        dispatch[0].ptx_isa
    );
}

#[test]
fn cpu_gpu_full_tile_byte_parity() {
    let (width, height) = (37u32, 29u32);
    let pixels = (width * height) as usize;
    let source = Image::new(
        width,
        height,
        (0..pixels * 4)
            .map(|i| ((i * 31 + 7) % 256) as u8)
            .collect(),
    )
    .unwrap();
    let repair = Image::new(
        width,
        height,
        (0..pixels * 4)
            .map(|i| ((i * 17 + 3) % 256) as u8)
            .collect(),
    )
    .unwrap();
    let mask_a: Vec<u8> = (0..pixels).map(|i| ((i * 13 + 5) % 256) as u8).collect();
    let mask_b: Vec<u8> = (0..pixels).map(|i| ((i * 29 + 11) % 256) as u8).collect();
    let expected = shared_min_mask_composite(&source, &repair, &mask_a, &mask_b).unwrap();

    let Some(gpu) = gpu() else { return };
    let (actual, dispatch) = gpu.apply(&source, &repair, &mask_a, &mask_b).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(dispatch[0].kernel, "beard_blend_kernel");
}
