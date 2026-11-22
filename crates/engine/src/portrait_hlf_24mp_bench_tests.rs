#![cfg(windows)]

use super::*;
use aurora_portrait::cuda::{
    driver_memory_info, driver_metrics, CudaError, DeviceMemoryInfo, Dispatch, DriverMetrics,
};
use aurora_portrait::cuda_hlf::{
    Hlf, BODY_KERNEL, FACE_KERNEL, GAUSS_KERNEL, MEDIAN_KERNEL,
};
use aurora_portrait::kernels::skin_adjust::{
    body_hlf_apply, hlf_face_apply, Roi, StoreMode,
};
use std::collections::BTreeMap;
use std::time::Instant;

const WIDTH: u32 = 6000;
const HEIGHT: u32 = 4000;
const PIXELS: usize = WIDTH as usize * HEIGHT as usize;
const WARM_RUNS: usize = 20;
const CPU_WARM_RUNS: usize = 7;
const STAGE_WARM_RUNS: usize = 5;
const DEGREE: f32 = 0.73;

#[derive(Clone, Copy, Debug, Default)]
struct PlaneDiff {
    differing_words: usize,
    differing_bytes: usize,
    max_abs: f32,
}

#[derive(Clone, Copy, Debug, Default)]
struct ByteDiff {
    differing_bytes: usize,
    max_abs: u8,
}

enum CpuEffect {
    Face(Vec<f32>),
    Body(Image),
}

enum CudaEffect {
    Face(Vec<f32>, Vec<Dispatch>),
    Body(Image, Vec<Dispatch>),
}

fn timed<T>(call: impl FnOnce() -> T) -> (T, f64) {
    let started = Instant::now();
    let value = call();
    (value, started.elapsed().as_secs_f64() * 1000.0)
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    let rank = (fraction * sorted.len() as f64).ceil() as usize;
    sorted[rank.saturating_sub(1).min(sorted.len() - 1)]
}

fn delta(after: DriverMetrics, before: DriverMetrics) -> DriverMetrics {
    DriverMetrics {
        ptx_module_load_calls: after.ptx_module_load_calls - before.ptx_module_load_calls,
        skin_lut_uploads: after.skin_lut_uploads - before.skin_lut_uploads,
        teeth_lut_uploads: after.teeth_lut_uploads - before.teeth_lut_uploads,
        host_to_device_bytes: after.host_to_device_bytes - before.host_to_device_bytes,
        device_to_host_bytes: after.device_to_host_bytes - before.device_to_host_bytes,
        device_allocations: after.device_allocations - before.device_allocations,
        context_syncs: after.context_syncs - before.context_syncs,
    }
}

fn print_metrics(label: &str, ms: f64, metrics: DriverMetrics) {
    eprintln!(
        "HLF24_BENCH metrics label={label} ms={ms:.6} module_loads={} allocs={} syncs={} h2d={} d2h={}",
        metrics.ptx_module_load_calls,
        metrics.device_allocations,
        metrics.context_syncs,
        metrics.host_to_device_bytes,
        metrics.device_to_host_bytes
    );
}

fn print_memory(label: &str, memory: DeviceMemoryInfo) {
    eprintln!(
        "HLF24_BENCH memory label={label} free={} used={} total={}",
        memory.free_bytes,
        memory.total_bytes - memory.free_bytes,
        memory.total_bytes
    );
}

fn noise_byte(x: u32, y: u32, channel: u32) -> u8 {
    let mut value = x
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add(y.wrapping_mul(0x85EB_CA77))
        .wrapping_add(channel.wrapping_mul(0xC2B2_AE3D));
    value ^= value >> 16;
    value = value.wrapping_mul(0x7FEB_352D);
    value ^= value >> 15;
    (value >> 24) as u8
}

fn pattern_image() -> Image {
    let mut pixels = vec![0u8; PIXELS * 4];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let offset = ((y as usize * WIDTH as usize) + x as usize) * 4;
            pixels[offset] = noise_byte(x, y, 0);
            pixels[offset + 1] = noise_byte(x, y, 1);
            pixels[offset + 2] = noise_byte(x, y, 2);
            pixels[offset + 3] = 255;
        }
    }
    Image::new(WIDTH, HEIGHT, pixels).expect("24MP image")
}

fn pattern_masks() -> PortraitMasks {
    let mut face = vec![0.0f32; PIXELS];
    let mut skin = vec![0.0f32; PIXELS];
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let index = y as usize * WIDTH as usize + x as usize;
            face[index] = noise_byte(x / 8, y / 8, 7) as f32 / 255.0;
            skin[index] = noise_byte(x / 4, y / 4, 11) as f32 / 255.0;
        }
    }
    PortraitMasks {
        skin: Some(skin),
        hair: None,
        face: Some(face),
        body: None,
        width: WIDTH as usize,
        height: HEIGHT as usize,
    }
}

fn source_rgb(image: &Image) -> Vec<f32> {
    let mut source = Vec::with_capacity(PIXELS * 3);
    for pixel in image.pixels.chunks_exact(4) {
        source.push(pixel[0] as f32 / 255.0);
        source.push(pixel[1] as f32 / 255.0);
        source.push(pixel[2] as f32 / 255.0);
    }
    source
}

fn face_rgba(face: &[f32]) -> Vec<f32> {
    let mut rgba = Vec::with_capacity(PIXELS * 4);
    for value in face {
        rgba.extend_from_slice(&[*value, 0.0, 0.0, 1.0]);
    }
    rgba
}

fn full_roi() -> Roi {
    Roi {
        x: 0,
        y: 0,
        width: WIDTH,
        height: HEIGHT,
    }
}

fn cpu_effect(
    pf: u32,
    image: &Image,
    source: &[f32],
    median: &[f32],
    gauss: &[f32],
    face: &[f32],
    skin: &[f32],
    body_face: &[f32],
) -> CpuEffect {
    match pf {
        30021 => CpuEffect::Face(
            hlf_face_apply(
                source,
                median,
                gauss,
                face,
                skin,
                WIDTH,
                HEIGHT,
                full_roi(),
                DEGREE * 100.0,
            )
            .expect("PF30021 CPU effect"),
        ),
        30022 => CpuEffect::Body(
            body_hlf_apply(
                image,
                full_roi(),
                median,
                gauss,
                body_face,
                skin,
                DEGREE * 100.0,
                hlf_auto_body_uniforms(),
                None,
            )
            .expect("PF30022 CPU effect"),
        ),
        _ => unreachable!(),
    }
}

fn cuda_effect(
    gpu: &Hlf,
    pf: u32,
    image: &Image,
    source: &[f32],
    median: &[f32],
    gauss: &[f32],
    face: &[f32],
    skin: &[f32],
    body_face: &[f32],
) -> Result<CudaEffect, CudaError> {
    match pf {
        30021 => gpu
            .face_apply(
                source,
                median,
                gauss,
                face,
                skin,
                WIDTH,
                HEIGHT,
                full_roi(),
                DEGREE * 100.0,
            )
            .map(|(rgb, dispatches)| CudaEffect::Face(rgb, dispatches)),
        30022 => gpu
            .body_apply(
                image,
                full_roi(),
                median,
                gauss,
                body_face,
                skin,
                DEGREE * 100.0,
                hlf_auto_body_uniforms(),
                None,
            )
            .map(|(out, dispatches)| CudaEffect::Body(out, dispatches)),
        _ => unreachable!(),
    }
}

fn plane_diff(cpu: &[f32], cuda: &[f32]) -> PlaneDiff {
    assert_eq!(cpu.len(), cuda.len());
    let mut result = PlaneDiff::default();
    for (&expected, &actual) in cpu.iter().zip(cuda) {
        let expected_bits = expected.to_bits();
        let actual_bits = actual.to_bits();
        if expected_bits == actual_bits {
            continue;
        }
        result.differing_words += 1;
        result.differing_bytes += (expected_bits ^ actual_bits)
            .to_le_bytes()
            .iter()
            .filter(|byte| **byte != 0)
            .count();
        result.max_abs = result.max_abs.max((expected - actual).abs());
    }
    result
}

fn image_diff(cpu: &Image, cuda: &Image) -> ByteDiff {
    assert_eq!((cpu.width, cpu.height), (cuda.width, cuda.height));
    let mut result = ByteDiff::default();
    for (&expected, &actual) in cpu.pixels.iter().zip(&cuda.pixels) {
        if expected == actual {
            continue;
        }
        result.differing_bytes += 1;
        result.max_abs = result.max_abs.max(expected.abs_diff(actual));
    }
    result
}

fn quantized_face_diff(cpu: &[f32], cuda: &[f32]) -> ByteDiff {
    assert_eq!(cpu.len(), cuda.len());
    let mut result = ByteDiff::default();
    for (&expected, &actual) in cpu.iter().zip(cuda) {
        let expected = StoreMode::TruncU8.quantize(expected);
        let actual = StoreMode::TruncU8.quantize(actual);
        if expected == actual {
            continue;
        }
        result.differing_bytes += 1;
        result.max_abs = result.max_abs.max(expected.abs_diff(actual));
    }
    result
}

fn capture_stage(
    pf: u32,
    image: &Image,
    masks: &PortraitMasks,
) -> (f64, DriverMetrics, Vec<Dispatch>, PortraitEffectRun) {
    let inputs = PortraitJobInputs::from_renderer_job(1, masks);
    let extra = BTreeMap::from([(format!("pc:{pf}"), DEGREE)]);
    let mut pixels = image.pixels.clone();
    let before = driver_metrics();
    let started = Instant::now();
    let report = apply_portrait_effects_with_job(
        &mut pixels,
        WIDTH,
        HEIGHT,
        NormRect::FULL,
        &extra,
        &inputs,
    );
    let elapsed = started.elapsed().as_secs_f64() * 1000.0;
    let after = driver_metrics();
    assert_eq!(report.effects.len(), 1);
    let run = report.effects.into_iter().next().unwrap();
    assert_eq!(run.backend, PortraitBackend::Cuda);
    assert!(run.executed);
    assert_eq!(run.refusal, None);
    let dispatches = run.dispatches.clone();
    (elapsed, delta(after, before), dispatches, run)
}

fn run_bench(pf: u32) {
    assert_eq!(
        std::env::var("PORTRAIT_CUDA_REQUIRED").as_deref(),
        Ok("1"),
        "set PORTRAIT_CUDA_REQUIRED=1"
    );
    eprintln!(
        "HLF24_BENCH start pf={pf} width={WIDTH} height={HEIGHT} pixels={PIXELS} warm_runs={WARM_RUNS}"
    );

    let image = pattern_image();
    let masks = pattern_masks();
    let face = masks.face.as_ref().unwrap();
    let skin = masks.skin.as_ref().unwrap();
    let body_face = face_rgba(face);
    let source = source_rgb(&image);

    let ((cpu_median, cpu_gauss), cpu_preprocess_cold_ms) = timed(|| {
        hlf_cpu_preprocess(
            &source,
            WIDTH,
            HEIGHT,
            HlfAutoProducerOptions::default(),
        )
    });
    let mut cpu_preprocess_warm = Vec::with_capacity(CPU_WARM_RUNS);
    for _ in 0..CPU_WARM_RUNS {
        let ((median, gauss), ms) = timed(|| {
            hlf_cpu_preprocess(
                &source,
                WIDTH,
                HEIGHT,
                HlfAutoProducerOptions::default(),
            )
        });
        drop((median, gauss));
        cpu_preprocess_warm.push(ms);
    }
    cpu_preprocess_warm.sort_by(f64::total_cmp);
    eprintln!(
        "HLF24_BENCH cpu_preprocess cold_ms={cpu_preprocess_cold_ms:.6} p50_ms={:.6} p95_ms={:.6} n={CPU_WARM_RUNS}",
        percentile(&cpu_preprocess_warm, 0.50),
        percentile(&cpu_preprocess_warm, 0.95)
    );

    let before_open = driver_metrics();
    let (gpu, open_ms) = timed(|| Hlf::open().expect("HLF CUDA open"));
    let open_metrics = delta(driver_metrics(), before_open);
    print_metrics("cuda_open", open_ms, open_metrics);
    assert_eq!(open_metrics.ptx_module_load_calls, 1);
    let memory_after_open = driver_memory_info().expect("CUDA memory after open");
    print_memory("after_open", memory_after_open);

    let before_preprocess = driver_metrics();
    let (preprocess_result, preprocess_cold_ms) = timed(|| {
        gpu.preprocess_source_rgb(&source, WIDTH, HEIGHT, 1, 1)
            .expect("HLF preprocessing")
    });
    let (cuda_median, cuda_gauss, preprocess_dispatches) = preprocess_result;
    let after_preprocess = driver_metrics();
    let preprocess_cold_metrics = delta(after_preprocess, before_preprocess);
    print_metrics(
        "cuda_preprocess_cold",
        preprocess_cold_ms,
        preprocess_cold_metrics,
    );
    assert_eq!(preprocess_dispatches.len(), 2);
    assert_eq!(preprocess_dispatches[0].kernel, MEDIAN_KERNEL);
    assert_eq!(preprocess_dispatches[1].kernel, GAUSS_KERNEL);
    for dispatch in &preprocess_dispatches {
        eprintln!(
            "HLF24_BENCH dispatch stage=preprocess kernel={} grid={}x{} block={}x{} device={} ptx_isa={:?}",
            dispatch.kernel,
            dispatch.grid.0,
            dispatch.grid.1,
            dispatch.block.0,
            dispatch.block.1,
            dispatch.device,
            dispatch.ptx_isa
        );
    }

    let median_diff = plane_diff(&cpu_median, &cuda_median);
    let gauss_diff = plane_diff(&cpu_gauss, &cuda_gauss);
    eprintln!(
        "HLF24_BENCH preprocess_diff median_words={} median_bytes={} median_max_abs={:.9} gauss_words={} gauss_bytes={} gauss_max_abs={:.9}",
        median_diff.differing_words,
        median_diff.differing_bytes,
        median_diff.max_abs,
        gauss_diff.differing_words,
        gauss_diff.differing_bytes,
        gauss_diff.max_abs
    );
    assert_eq!(median_diff.differing_words, 0);
    assert_eq!(gauss_diff.differing_words, 0);
    drop((cuda_median, cuda_gauss));
    print_memory(
        "after_preprocess_cold",
        driver_memory_info().expect("CUDA memory after preprocessing"),
    );

    for _ in 0..3 {
        let (median, gauss, _) = gpu
            .preprocess_source_rgb(&source, WIDTH, HEIGHT, 1, 1)
            .expect("preprocess warmup");
        drop((median, gauss));
    }
    let mut preprocess_warm = Vec::with_capacity(WARM_RUNS);
    let mut preprocess_warm_metrics = DriverMetrics::default();
    for index in 0..WARM_RUNS {
        let before = driver_metrics();
        let ((median, gauss, dispatches), ms) = timed(|| {
            gpu.preprocess_source_rgb(&source, WIDTH, HEIGHT, 1, 1)
                .expect("preprocess sample")
        });
        let after = driver_metrics();
        assert_eq!(dispatches.len(), 2);
        if index == 0 {
            preprocess_warm_metrics = delta(after, before);
        }
        preprocess_warm.push(ms);
        drop((median, gauss));
    }
    preprocess_warm.sort_by(f64::total_cmp);
    print_metrics(
        "cuda_preprocess_warm_representative",
        percentile(&preprocess_warm, 0.50),
        preprocess_warm_metrics,
    );
    eprintln!(
        "HLF24_BENCH cuda_preprocess warm_p50_ms={:.6} warm_p95_ms={:.6} n={WARM_RUNS}",
        percentile(&preprocess_warm, 0.50),
        percentile(&preprocess_warm, 0.95)
    );

    let (cpu_effect_value, cpu_effect_cold_ms) = timed(|| {
        cpu_effect(
            pf,
            &image,
            &source,
            &cpu_median,
            &cpu_gauss,
            face,
            skin,
            &body_face,
        )
    });
    let mut cpu_effect_warm = Vec::with_capacity(CPU_WARM_RUNS);
    for _ in 0..CPU_WARM_RUNS {
        let (value, ms) = timed(|| {
            cpu_effect(
                pf,
                &image,
                &source,
                &cpu_median,
                &cpu_gauss,
                face,
                skin,
                &body_face,
            )
        });
        drop(value);
        cpu_effect_warm.push(ms);
    }
    cpu_effect_warm.sort_by(f64::total_cmp);
    eprintln!(
        "HLF24_BENCH cpu_effect cold_ms={cpu_effect_cold_ms:.6} p50_ms={:.6} p95_ms={:.6} n={CPU_WARM_RUNS}",
        percentile(&cpu_effect_warm, 0.50),
        percentile(&cpu_effect_warm, 0.95)
    );

    let before_effect = driver_metrics();
    let (cuda_effect_value, cuda_effect_cold_ms) = timed(|| {
        cuda_effect(
            &gpu,
            pf,
            &image,
            &source,
            &cpu_median,
            &cpu_gauss,
            face,
            skin,
            &body_face,
        )
        .expect("PF CUDA effect")
    });
    let after_effect = driver_metrics();
    print_metrics(
        "cuda_effect_cold",
        cuda_effect_cold_ms,
        delta(after_effect, before_effect),
    );

    match (&cpu_effect_value, &cuda_effect_value) {
        (CpuEffect::Face(cpu), CudaEffect::Face(cuda, dispatches)) => {
            assert_eq!(dispatches.len(), 1);
            assert_eq!(dispatches[0].kernel, FACE_KERNEL);
            let diff = plane_diff(cpu, cuda);
            let quantized = quantized_face_diff(cpu, cuda);
            eprintln!(
                "HLF24_BENCH effect_diff f32_words={} f32_bytes={} f32_max_abs={:.9} u8_bytes={} u8_max_abs={}",
                diff.differing_words,
                diff.differing_bytes,
                diff.max_abs,
                quantized.differing_bytes,
                quantized.max_abs
            );
            assert_eq!(diff.differing_words, 0);
            assert_eq!(quantized.differing_bytes, 0);
            print_dispatch("effect", &dispatches[0]);
        }
        (CpuEffect::Body(cpu), CudaEffect::Body(cuda, dispatches)) => {
            assert_eq!(dispatches.len(), 1);
            assert_eq!(dispatches[0].kernel, BODY_KERNEL);
            let diff = image_diff(cpu, cuda);
            eprintln!(
                "HLF24_BENCH effect_diff u8_bytes={} u8_max_abs={}",
                diff.differing_bytes, diff.max_abs
            );
            assert_eq!(diff.differing_bytes, 0);
            print_dispatch("effect", &dispatches[0]);
        }
        _ => panic!("CPU/CUDA effect kind mismatch"),
    }

    for _ in 0..2 {
        let value = cuda_effect(
            &gpu,
            pf,
            &image,
            &source,
            &cpu_median,
            &cpu_gauss,
            face,
            skin,
            &body_face,
        )
        .expect("effect warmup");
        drop(value);
    }
    let mut cuda_effect_warm = Vec::with_capacity(WARM_RUNS);
    let mut cuda_effect_warm_metrics = DriverMetrics::default();
    for index in 0..WARM_RUNS {
        let before = driver_metrics();
        let (value, ms) = timed(|| {
            cuda_effect(
                &gpu,
                pf,
                &image,
                &source,
                &cpu_median,
                &cpu_gauss,
                face,
                skin,
                &body_face,
            )
            .expect("effect sample")
        });
        let after = driver_metrics();
        if index == 0 {
            cuda_effect_warm_metrics = delta(after, before);
        }
        cuda_effect_warm.push(ms);
        drop(value);
    }
    cuda_effect_warm.sort_by(f64::total_cmp);
    print_metrics(
        "cuda_effect_warm_representative",
        percentile(&cuda_effect_warm, 0.50),
        cuda_effect_warm_metrics,
    );
    eprintln!(
        "HLF24_BENCH cuda_effect warm_p50_ms={:.6} warm_p95_ms={:.6} n={WARM_RUNS}",
        percentile(&cuda_effect_warm, 0.50),
        percentile(&cuda_effect_warm, 0.95)
    );
    let memory_after_effect = driver_memory_info().expect("CUDA memory after effect");
    print_memory("after_effect_warm", memory_after_effect);
    eprintln!(
        "HLF24_BENCH memory free_drop_from_open={}",
        memory_after_open.free_bytes.saturating_sub(memory_after_effect.free_bytes)
    );

    let stage_expected = if pf == 30021 { FACE_KERNEL } else { BODY_KERNEL };
    let mut stage_warm = Vec::with_capacity(STAGE_WARM_RUNS);
    let mut stage_warm_metrics = DriverMetrics::default();
    for index in 0..STAGE_WARM_RUNS {
        let (ms, metrics, dispatches, _run) = capture_stage(pf, &image, &masks);
        let kernels: Vec<_> = dispatches.iter().map(|dispatch| dispatch.kernel).collect();
        assert_eq!(kernels, [MEDIAN_KERNEL, GAUSS_KERNEL, stage_expected]);
        if index == 0 {
            print_metrics("stage_first_after_components", ms, metrics);
            for dispatch in &dispatches {
                print_dispatch("stage", dispatch);
            }
        } else if index == 1 {
            stage_warm_metrics = metrics;
        }
        stage_warm.push(ms);
    }
    stage_warm.sort_by(f64::total_cmp);
    print_metrics(
        "stage_warm_representative",
        percentile(&stage_warm, 0.50),
        stage_warm_metrics,
    );
    eprintln!(
        "HLF24_BENCH stage warm_p50_ms={:.6} warm_p95_ms={:.6} n={STAGE_WARM_RUNS}",
        percentile(&stage_warm, 0.50),
        percentile(&stage_warm, 0.95)
    );
    print_memory(
        "after_stage",
        driver_memory_info().expect("CUDA memory after stage"),
    );

    let retained = match pf {
        30021 => 1_344_000_000usize + 576_000_000usize,
        30022 => 576_000_000usize,
        _ => unreachable!(),
    };
    let transient = if pf == 30022 { 1_248_000_000usize } else { 0 };
    eprintln!(
        "HLF24_BENCH estimated_device_bytes retained={retained} transient_peak={transient} peak={}",
        retained + transient
    );
    eprintln!("HLF24_BENCH done pf={pf}");
}

fn print_dispatch(stage: &str, dispatch: &Dispatch) {
    eprintln!(
        "HLF24_BENCH dispatch stage={stage} kernel={} grid={}x{} block={}x{} device={} ptx_isa={:?} driver={} cc={}.{}",
        dispatch.kernel,
        dispatch.grid.0,
        dispatch.grid.1,
        dispatch.block.0,
        dispatch.block.1,
        dispatch.device,
        dispatch.ptx_isa,
        dispatch.driver_version,
        dispatch.compute_cap.0,
        dispatch.compute_cap.1
    );
}

#[test]
#[ignore = "24MP real-device benchmark; run explicitly with PORTRAIT_CUDA_REQUIRED=1"]
fn hlf_24mp_pf30021_bench() {
    run_bench(30021);
}

#[test]
#[ignore = "24MP real-device benchmark; run explicitly with PORTRAIT_CUDA_REQUIRED=1"]
fn hlf_24mp_pf30022_bench() {
    run_bench(30022);
}
