//! Acceptance bench for the CUDA blemish consumer on the 6048x4032 DSC_9868
//! frame.  Run with:
//!
//! ```text
//! cargo run --release -p aurora-engine --example blemish_gpu_bench
//! ```
//!
//! `AURORA_BLEMISH_FIXTURE` overrides the fixture directory.

#[cfg(not(windows))]
fn main() {
    eprintln!("blemish_gpu_bench requires Windows CUDA; skipped on this platform");
}

#[cfg(windows)]
mod cuda_bench {
use aurora_engine::blemish_gainmap::{
    apply_gainmap_16u, apply_gainmap_16u_cpu, apply_gainmap_16u_scalar, apply_gainmap_8u,
    apply_gainmap_8u_cpu, apply_gainmap_8u_scalar, GainMap, SpotlessGainmapInputs,
    SpotlessGainmapInputs8,
};
use aurora_engine::blemish_gainmap_cuda as cuda;
use std::path::PathBuf;
use std::time::Instant;

const W: usize = 6048;
const H: usize = 4032;

fn read_u16(path: &PathBuf) -> Vec<u16> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(2)
        .map(|p| u16::from_le_bytes([p[0], p[1]]))
        .collect()
}

fn bytes_of<T>(values: &[T]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    }
}

fn bytes_of_mut<T>(values: &mut [T]) -> &mut [u8] {
    unsafe {
        std::slice::from_raw_parts_mut(values.as_mut_ptr() as *mut u8, std::mem::size_of_val(values))
    }
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

pub(crate) fn main() {
    let dir = std::env::args_os()
        .nth(1)
        .or_else(|| std::env::var_os("AURORA_BLEMISH_FIXTURE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            eprintln!("usage: blemish_gpu_bench <fixture_dir> (or set AURORA_BLEMISH_FIXTURE)");
            std::process::exit(2);
        });
    println!("fixture: {}", dir.display());

    let src = read_u16(&dir.join("s1_linear_9868.u16"));
    let golden = read_u16(&dir.join("engine_9868_blemish.u16"));
    assert_eq!(src.len(), W * H * 3, "source shape");
    assert_eq!(golden.len(), W * H * 3, "golden shape");
    let rgba: Vec<f32> = std::fs::read(dir.join("gainmap_9868_450x300.f32"))
        .unwrap()
        .chunks_exact(4)
        .map(|p| f32::from_le_bytes([p[0], p[1], p[2], p[3]]))
        .collect();
    let map = GainMap {
        width: 450,
        height: 300,
        rgba,
    };
    let input = SpotlessGainmapInputs {
        rgb16: &src,
        width: W as u32,
        height: H as u32,
        gain_map: &map,
        amount: 0.5,
        protection_mask: None,
    };

    let t = Instant::now();
    let scalar = apply_gainmap_16u_scalar(&input).unwrap();
    let scalar_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(scalar, golden, "CPU scalar != golden");

    let mut cpu_samples = Vec::new();
    let mut cpu = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        cpu = apply_gainmap_16u_cpu(&input).unwrap();
        cpu_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    assert_eq!(cpu, golden, "CPU fast != golden");
    let cpu_ms = median(cpu_samples);

    let mut gpu_samples = Vec::new();
    let mut dispatch = None;
    for _ in 0..5 {
        let t = Instant::now();
        let (out, record) = cuda::apply_16u(&input).expect("CUDA 16u");
        gpu_samples.push(t.elapsed().as_secs_f64() * 1e3);
        assert_eq!(out, golden, "GPU 16u != golden");
        dispatch = Some(record);
    }
    let gpu_host_ms = median(gpu_samples);
    let dispatch = dispatch.unwrap();

    let gpu = cuda::open().expect("CUDA module");
    let mut pin_src = gpu.alloc_pinned(W * H * 3 * 2).unwrap();
    let mut pin_out = gpu.alloc_pinned(W * H * 3 * 2).unwrap();
    pin_src.copy_from(bytes_of(&src));
    let mut pinned_samples = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        cuda::apply_16u_pinned(&input, &pin_src, &mut pin_out).expect("CUDA 16u pinned");
        pinned_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let pinned_ms = median(pinned_samples);
    let mut pinned_out = vec![0u16; W * H * 3];
    pin_out.copy_to(bytes_of_mut(&mut pinned_out));
    assert_eq!(pinned_out, golden, "GPU 16u pinned != golden");

    let mut staged_samples = Vec::new();
    let mut staged = vec![0u16; W * H * 3];
    for _ in 0..5 {
        let t = Instant::now();
        pin_src.copy_from(bytes_of(&src));
        cuda::apply_16u_pinned(&input, &pin_src, &mut pin_out).expect("CUDA 16u staged");
        pin_out.copy_to(bytes_of_mut(&mut staged));
        staged_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let staged_ms = median(staged_samples);
    assert_eq!(staged, golden, "GPU 16u staged != golden");

    let dev_src = cuda::alloc_and_upload(&gpu, bytes_of(&src)).unwrap();
    let dev_map = cuda::alloc_and_upload(&gpu, bytes_of(&map.rgba)).unwrap();
    let dev_out = gpu.alloc_buffer(W * H * 3 * 2).unwrap();
    let mut out = vec![0u16; W * H * 3];
    let mut run_device = |times: usize| -> f64 {
        let t = Instant::now();
        for _ in 0..times {
            cuda::apply_16u_device(
                dev_src.handle(),
                dev_map.handle(),
                0,
                dev_out.handle(),
                W as u32,
                H as u32,
                &map,
                0.5,
                false,
            )
            .unwrap();
        }
        gpu.download_buffer(&dev_out, bytes_of_mut(&mut out)).unwrap();
        t.elapsed().as_secs_f64() * 1e3
    };
    run_device(3);
    let one = median((0..5).map(|_| run_device(1)).collect());
    let many = median((0..5).map(|_| run_device(21)).collect());
    let kernel_ms = (many - one) / 20.0;
    assert_eq!(out, golden, "GPU device-resident != golden");

    let src8: Vec<u8> = src.iter().map(|v| (v / 257) as u8).collect();
    let input8 = SpotlessGainmapInputs8 {
        rgb8: &src8,
        width: W as u32,
        height: H as u32,
        gain_map: &map,
        amount: 0.5,
        protection_mask: None,
    };
    let t = Instant::now();
    let cpu8 = apply_gainmap_8u_scalar(&input8).unwrap();
    let scalar8_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let fast8 = apply_gainmap_8u_cpu(&input8).unwrap();
    let cpu8_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(cpu8, fast8, "8u CPU fast != scalar");

    let mut gpu8_samples = Vec::new();
    let mut gpu8 = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        gpu8 = cuda::apply_8u(&src8, W as u32, H as u32, &map, 0.5, None)
            .expect("CUDA 8u")
            .0;
        gpu8_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    assert_eq!(gpu8, cpu8, "GPU 8u != CPU");
    let gpu8_host_ms = median(gpu8_samples);

    let mut pin_src8 = gpu.alloc_pinned(W * H * 3).unwrap();
    let mut pin_out8 = gpu.alloc_pinned(W * H * 3).unwrap();
    pin_src8.copy_from(&src8);
    let mut pinned8_samples = Vec::new();
    for _ in 0..5 {
        let t = Instant::now();
        cuda::apply_8u_pinned(
            &src8,
            W as u32,
            H as u32,
            &map,
            0.5,
            &pin_src8,
            &mut pin_out8,
        )
        .expect("CUDA 8u pinned");
        pinned8_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let pinned8_ms = median(pinned8_samples);
    let mut pinned_out8 = vec![0u8; W * H * 3];
    pin_out8.copy_to(&mut pinned_out8);
    assert_eq!(pinned_out8, cpu8, "GPU 8u pinned != CPU");

    let mut staged8_samples = Vec::new();
    let mut staged8 = vec![0u8; W * H * 3];
    for _ in 0..5 {
        let t = Instant::now();
        pin_src8.copy_from(&src8);
        cuda::apply_8u_pinned(
            &src8,
            W as u32,
            H as u32,
            &map,
            0.5,
            &pin_src8,
            &mut pin_out8,
        )
        .expect("CUDA 8u staged");
        pin_out8.copy_to(&mut staged8);
        staged8_samples.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let staged8_ms = median(staged8_samples);
    assert_eq!(staged8, cpu8, "GPU 8u staged != CPU");

    let dev_src8 = cuda::alloc_and_upload(&gpu, &src8).unwrap();
    let dev_out8 = gpu.alloc_buffer(W * H * 3).unwrap();
    let mut out8 = vec![0u8; W * H * 3];
    let mut run_device8 = |times: usize| -> f64 {
        let t = Instant::now();
        for _ in 0..times {
            cuda::apply_8u_device(
                dev_src8.handle(),
                dev_map.handle(),
                0,
                dev_out8.handle(),
                W as u32,
                H as u32,
                &map,
                0.5,
                false,
            )
            .unwrap();
        }
        gpu.download_buffer(&dev_out8, &mut out8).unwrap();
        t.elapsed().as_secs_f64() * 1e3
    };
    run_device8(3);
    let one8 = median((0..5).map(|_| run_device8(1)).collect());
    let many8 = median((0..5).map(|_| run_device8(21)).collect());
    let kernel8_ms = (many8 - one8) / 20.0;
    assert_eq!(out8, cpu8, "GPU 8u device-resident != CPU");

    println!(
        "{} grid={:?} block={:?} ptx={} B",
        dispatch.kernel, dispatch.grid, dispatch.block, dispatch.ptx_bytes
    );
    println!("device: {}", dispatch.device);
    println!();
    println!("24.4 MP (6048x4032), amount=0.5, 16-bit RGB");
    println!("| path | ms |");
    println!("|---|---:|");
    println!("| CPU scalar oracle | {scalar_ms:.1} |");
    println!("| CPU fast (rayon) | {cpu_ms:.1} |");
    println!("| CUDA host frame (H2D+kernel+D2H) | {gpu_host_ms:.2} |");
    println!("| CUDA pinned host frame | {pinned_ms:.2} |");
    println!("| CUDA pinned + pageable copy in/out | {staged_ms:.2} |");
    println!("| CUDA kernel only (device-resident) | {kernel_ms:.2} |");
    println!();
    println!("same frame as 8-bit RGB");
    println!("| path | ms |");
    println!("|---|---:|");
    println!("| CPU scalar oracle | {scalar8_ms:.1} |");
    println!("| CPU fast (rayon) | {cpu8_ms:.1} |");
    println!("| CUDA host frame (H2D+kernel+D2H) | {gpu8_host_ms:.2} |");
    println!("| CUDA pinned host frame | {pinned8_ms:.2} |");
    println!("| CUDA pinned + pageable copy in/out | {staged8_ms:.2} |");
    println!("| CUDA kernel only (device-resident) | {kernel8_ms:.2} |");
    println!();
    println!("transfer size: {} MiB per direction (u16 RGB)", W * H * 3 * 2 / (1024 * 1024));
}
}

#[cfg(windows)]
fn main() {
    cuda_bench::main();
}
