//! Timed PF108-110 run on a 512x512 frame through one reused session.
//!
//!   rustc --edition 2021 -O work\backend-r11\o1i\bench.rs -o work\backend-r11\o1i\bench.exe

extern crate self as aurora_portrait;

#[path = "../../../crates/portrait/src/lib.rs"]
mod pc;

pub use pc::*;

#[path = "../../../crates/portrait/src/cuda_teeth.rs"]
mod cuda_teeth;

use cuda_teeth::Teeth;
use pc::kernels::face_detail;
use pc::Image;
use std::time::Instant;

const WHITEN: &[u8] = include_bytes!("whiten.rgb");
const BRIGHT: &[u8] = include_bytes!("bright.rgb");

fn noise(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect()
}

fn main() {
    let side = 512u32;
    let pixels = (side * side) as usize;
    let source = Image::new(side, side, noise(pixels * 4, 7)).unwrap();
    let mask = noise(pixels, 11);
    let (common, bright, deyellow) = (100.0f32, 100.0f32, 100.0f32);

    let gpu = match Teeth::open() {
        Ok(gpu) => gpu,
        Err(error) => {
            println!("{{\"status\":\"NO_GPU_PATH\",\"error\":\"{error:?}\"}}");
            std::process::exit(2);
        }
    };
    let session = gpu.session(side, side, WHITEN, BRIGHT).expect("session");
    let started = Instant::now();
    let (first, dispatches) = session
        .apply(&source, &mask, common, bright, deyellow)
        .expect("real GPU dispatch");
    let cold = started.elapsed().as_secs_f64();

    let mut samples = Vec::new();
    for _ in 0..20 {
        let started = Instant::now();
        session.apply(&source, &mask, common, bright, deyellow).expect("real GPU dispatch");
        samples.push(started.elapsed().as_secs_f64() * 1e3);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_ms = samples[samples.len() / 2];

    let started = Instant::now();
    let expected =
        face_detail::teeth_whitening(&source, &mask, WHITEN, BRIGHT, common, bright, deyellow).unwrap();
    let cpu_ms = started.elapsed().as_secs_f64() * 1e3;

    let differing = expected.output.pixels.iter().zip(&first.pixels).filter(|(a, b)| a != b).count();
    let alpha_bad = (0..pixels).filter(|i| first.pixels[i * 4 + 3] != source.pixels[i * 4 + 3]).count();
    assert_eq!(differing, 0, "{differing} bytes differ from the CPU oracle");
    assert_eq!(alpha_bad, 0, "{alpha_bad} pixels changed alpha");
    assert_eq!(dispatches.len(), 1);
    println!(
        "{{\"status\":\"PASS\",\"device\":\"{}\",\"compute_cap\":\"{}.{}\",\"driver\":{},\"ptx_isa\":\"{}.{}\",\"kernel\":\"{}\",\"grid\":\"{}x{}\",\"block\":\"{}x{}\",\"canvas\":\"{side}x{side}\",\"pixels\":{pixels},\"gpu_seconds_cold\":{cold:.4},\"gpu_ms_median\":{median_ms:.3},\"cpu_ms\":{cpu_ms:.3},\"differing_bytes\":{differing},\"alpha_mismatch_pixels\":{alpha_bad},\"allocations\":5}}",
        dispatches[0].device,
        dispatches[0].compute_cap.0,
        dispatches[0].compute_cap.1,
        dispatches[0].driver_version,
        dispatches[0].ptx_isa.0,
        dispatches[0].ptx_isa.1,
        dispatches[0].kernel,
        dispatches[0].grid.0,
        dispatches[0].grid.1,
        dispatches[0].block.0,
        dispatches[0].block.1,
    );
}
