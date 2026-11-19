#![cfg(windows)]

use aurora_portrait::cuda_pf90021::{
    apply_cpu, Direction, Pf90021, Request, Run, RuntimeScalars, VolumeLut64, COMPOSITE_KERNEL,
    LUT_FLOATS, LUT_PASS_KERNEL,
};
use aurora_portrait::Image;

fn runtime() -> RuntimeScalars {
    RuntimeScalars {
        degree01: 0.8,
        opacity01: 0.85,
        shelter: 1.0,
        region: 1.0,
    }
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[test]
fn one_pixel_div255_boundary_is_byte_exact_on_real_cuda() {
    let mapped = f32::from_bits(0x3e7a_faf8);
    let volume = VolumeLut64::from_nodes(vec![mapped; LUT_FLOATS]).unwrap();
    let source = Image::solid(1, 1, [0, 0, 0, 255]);
    let mask = [3u8];
    let cpu = apply_cpu(&source, &volume, &mask, runtime()).unwrap();
    assert_eq!(cpu.pixels, [0, 0, 0, 255]);

    let result = Pf90021::open()
        .expect("real CUDA required for PF90021 parity fixture")
        .run(Request {
            source: &source,
            direction: Direction::Positive,
            lut: Some(&volume),
            mask_r: Some(&mask),
            runtime: Some(runtime()),
        })
        .expect("real two-pass PF90021 dispatch");
    let Run::Executed { output, dispatch } = result else {
        panic!("non-neutral fixture bypassed");
    };
    assert_eq!(output.pixels, cpu.pixels, "one-pixel CPU/CUDA mismatch");
    assert_eq!(dispatch.lut.kernel, LUT_PASS_KERNEL);
    assert_eq!(dispatch.composite.kernel, COMPOSITE_KERNEL);
    println!(
        "PF90021_MIN device={} driver={} ptx={}.{} source=[0,0,0,255] mask=3 lut_bits=3E7AFAF8 output={:?}",
        dispatch.lut.device,
        dispatch.lut.driver_version,
        dispatch.lut.ptx_isa.0,
        dispatch.lut.ptx_isa.1,
        output.pixels,
    );
}

#[test]
fn s566_pf90021_fixture_is_byte_exact_on_real_cuda() {
    let (width, height) = (512u32, 512u32);
    let pixels = width as usize * height as usize;
    let mut source = vec![0u8; pixels * 4];
    for y in 0..height {
        for x in 0..width {
            let at = (y as usize * width as usize + x as usize) * 4;
            let noise = ((x * 17 + y * 31 + x * y) % 37) as u8;
            source[at] = 178u8.saturating_add(noise);
            source[at + 1] = 132u8.saturating_add(noise / 2);
            source[at + 2] = 91u8.saturating_add(noise / 3);
            source[at + 3] = 255;
        }
    }
    let mask: Vec<u8> = (0..pixels)
        .map(|index| ((index * 29 + 61) % 256) as u8)
        .collect();
    let volume = VolumeLut64::from_nodes(
        (0..LUT_FLOATS)
            .map(|index| ((index * 37) % 1024) as f32 / 1023.0)
            .collect(),
    )
    .unwrap();
    let image = Image::new(width, height, source).unwrap();
    let cpu = apply_cpu(&image, &volume, &mask, runtime()).unwrap();
    let result = Pf90021::open()
        .expect("real CUDA required for PF90021 parity fixture")
        .run(Request {
            source: &image,
            direction: Direction::Positive,
            lut: Some(&volume),
            mask_r: Some(&mask),
            runtime: Some(runtime()),
        })
        .unwrap();
    let Run::Executed { output, dispatch } = result else {
        panic!("non-neutral fixture bypassed");
    };
    assert_eq!(dispatch.lut.kernel, LUT_PASS_KERNEL);
    assert_eq!(dispatch.composite.kernel, COMPOSITE_KERNEL);

    let different = cpu
        .pixels
        .iter()
        .zip(&output.pixels)
        .filter(|(expected, actual)| expected != actual)
        .count();
    let first: Vec<String> = cpu
        .pixels
        .iter()
        .zip(&output.pixels)
        .enumerate()
        .filter(|(_, (expected, actual))| expected != actual)
        .take(12)
        .map(|(index, (expected, actual))| format!("{index}:{expected}->{actual}"))
        .collect();
    assert_eq!(
        different, 0,
        "CPU/CUDA bytes must match exactly; first={}",
        first.join(",")
    );
    println!(
        "PF90021_S566 device={} driver={} ptx={}.{} cpu_fnv64={:016X} gpu_fnv64={:016X} differing_bytes={different}",
        dispatch.lut.device,
        dispatch.lut.driver_version,
        dispatch.lut.ptx_isa.0,
        dispatch.lut.ptx_isa.1,
        fnv64(&cpu.pixels),
        fnv64(&output.pixels),
    );
}
