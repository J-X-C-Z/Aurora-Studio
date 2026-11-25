//! Windows CUDA acceptance benchmark for the N35 reuse path.
//!
//! The kernels it drives (`facial_smooth`, eye bag, PF30025, PF90021, mesh
//! raster/attribute) only exist in the Windows driver build. On other platforms
//! the crate keeps this example as an explicit no-op instead of failing
//! `cargo check --all-targets` with unresolved CUDA items.

#[cfg(not(windows))]
fn main() {
    eprintln!(
        "n35_reuse_acceptance is a Windows CUDA example; nothing to run on this platform."
    );
}

#[cfg(windows)]
mod win_example {
use aurora_portrait::cuda::{DriverMetrics, driver_metrics};
use aurora_portrait::cuda_eye_bag::{self, EyeBag};
use aurora_portrait::cuda_mesh_attribute::MeshAttribute;
use aurora_portrait::cuda_pf30025::{self, ToothRepair};
use aurora_portrait::cuda_pf90021::{
    self, Direction, Pf90021, Request, Run, RuntimeScalars, VolumeLut64,
};
use aurora_portrait::kernels::mesh_attribute;
use aurora_portrait::kernels::{face_detail, facial_smooth};
use aurora_portrait::raster::{self, Cull, Edge, Mesh, Overlap, Rules, Sampler, Winding, Wrap};
use aurora_portrait::{Image, raster_fragment};
use std::time::{Duration, Instant};

const WARM_512: usize = 100;
const WARM_24MP: usize = 30;

#[derive(Clone, Debug)]
struct Stats {
    p50_ms: f64,
    p95_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

#[derive(Clone, Debug)]
struct Case {
    name: &'static str,
    width: u32,
    height: u32,
    bytes: usize,
    first_alloc: usize,
    warm_alloc: usize,
    first_sync: usize,
    warm_sync: usize,
    first_hash: u64,
    parity: bool,
    max_delta: u8,
    selected_nonzero: u64,
    coverage_nonzero: u64,
    selected_hash: u64,
    coverage_hash: u64,
    kernels: String,
    device: String,
    stats: Stats,
}

fn pattern(width: u32, height: u32, seed: u32) -> Image {
    let len = width as usize * height as usize * 4;
    let mut state = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    let pixels = (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 24) as u8
        })
        .collect();
    Image::new(width, height, pixels).unwrap()
}

fn plane(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed.wrapping_mul(2_246_822_519).wrapping_add(3_266_489_917);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state & 0x00ff_ffff) as f32 / 16_777_215.0
        })
        .collect()
}

fn mask_bytes(len: usize, seed: u32) -> Vec<u8> {
    let mut state = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            (state >> 16) as u8
        })
        .collect()
}

fn fnv64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

fn fnv64_u32(values: &[u32]) -> u64 {
    values.iter().fold(0xcbf29ce484222325u64, |hash, value| {
        value.to_le_bytes().iter().fold(hash, |inner, byte| {
            (inner ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
        })
    })
}

fn fnv64_pairs(values: &[[f32; 2]]) -> u64 {
    let bytes = unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, std::mem::size_of_val(values))
    };
    fnv64(bytes)
}

fn compare(expected: &[u8], actual: &[u8]) -> (bool, u8) {
    let max_delta = expected
        .iter()
        .zip(actual)
        .map(|(left, right)| left.abs_diff(*right))
        .max()
        .unwrap_or(0);
    (expected == actual, max_delta)
}

fn nearest_rank(sorted: &[Duration], numerator: usize, denominator: usize) -> f64 {
    let index = ((sorted.len() * numerator).div_ceil(denominator)).saturating_sub(1);
    sorted[index].as_secs_f64() * 1_000.0
}

fn measure(mut call: impl FnMut(), samples: usize) -> Stats {
    for _ in 0..10 {
        call();
    }
    let mut times = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        call();
        times.push(start.elapsed());
    }
    times.sort_unstable();
    Stats {
        p50_ms: nearest_rank(&times, 50, 100),
        p95_ms: nearest_rank(&times, 95, 100),
        min_ms: times[0].as_secs_f64() * 1_000.0,
        max_ms: times[times.len() - 1].as_secs_f64() * 1_000.0,
    }
}

fn allocations(before: DriverMetrics, after: DriverMetrics) -> (usize, usize) {
    (
        after.device_allocations - before.device_allocations,
        after.context_syncs - before.context_syncs,
    )
}

fn run_facial_smooth(width: u32, height: u32) -> Case {
    let source = pattern(width, height, 11);
    let base = pattern(width, height, 13);
    let residual = pattern(width, height, 17);
    let len = source.pixel_count();
    let mask_r = mask_bytes(len, 19);
    let mask_b = mask_bytes(len, 23);
    let skin = mask_bytes(len, 29);
    let uniforms = facial_smooth::Uniforms {
        legal_alpha: 0.7,
        dark_circle_alpha: 0.9,
        diff_factor: 0.6,
        diff_factor2: 0.25,
        color_threshold_factor: 0.4,
        mix_alpha: 0.35,
        diff_min_threshold: -0.5,
        diff_max_threshold: 0.5,
        high_freq_min_threshold: -0.06,
        high_freq_max_threshold: 0.04,
    };
    let expected =
        facial_smooth::apply(&source, &base, &residual, &mask_r, &mask_b, &skin, uniforms).unwrap();
    let gpu = aurora_portrait::cuda::Nvidia::open().expect("required CUDA device");

    let before = driver_metrics();
    let (first, dispatch) = gpu
        .facial_smooth(
            &source, &base, &residual, &mask_r, &mask_b, &skin, 90024, 0.5, &uniforms,
        )
        .expect("P15 dispatch");
    let (first_alloc, first_sync) = allocations(before, driver_metrics());

    let before = driver_metrics();
    let (warm, warm_dispatch) = gpu
        .facial_smooth(
            &source, &base, &residual, &mask_r, &mask_b, &skin, 90024, 0.5, &uniforms,
        )
        .expect("warm P15 dispatch");
    let (warm_alloc, warm_sync) = allocations(before, driver_metrics());
    let (_, max_delta) = compare(&expected.pixels, &first.pixels);
    assert_eq!(first, expected);
    assert_eq!(warm, expected);
    assert_eq!(warm_alloc, 0);
    assert_eq!(warm_sync, 0);
    assert_eq!(dispatch.kernel, warm_dispatch.kernel);

    let stats = measure(
        || {
            let (_, _) = gpu
                .facial_smooth(
                    &source, &base, &residual, &mask_r, &mask_b, &skin, 90024, 0.5, &uniforms,
                )
                .unwrap();
        },
        if width == 512 { WARM_512 } else { WARM_24MP },
    );
    Case {
        name: if width == 512 {
            "facial_smooth_512"
        } else {
            "facial_smooth_24mp"
        },
        width,
        height,
        bytes: first.pixels.len(),
        first_alloc,
        warm_alloc,
        first_sync,
        warm_sync,
        first_hash: fnv64(&first.pixels),
        parity: true,
        max_delta,
        selected_nonzero: 0,
        coverage_nonzero: 0,
        selected_hash: 0,
        coverage_hash: 0,
        kernels: dispatch.kernel.to_string(),
        device: dispatch.device,
        stats,
    }
}

fn run_eye_bag(width: u32, height: u32) -> Case {
    let source = pattern(width, height, 31);
    let color = pattern(width, height, 37);
    let mask = plane(source.pixel_count(), 41);
    let expected = cuda_eye_bag::reference(&source, &color, &mask, 0.73).unwrap();
    let gpu = EyeBag::open().expect("required CUDA device");

    let before = driver_metrics();
    let (first, dispatch) = gpu
        .compose(&source, &color, &mask, 0.73)
        .expect("PF90116 dispatch");
    let (first_alloc, first_sync) = allocations(before, driver_metrics());

    let before = driver_metrics();
    let (warm, warm_dispatch) = gpu
        .compose(&source, &color, &mask, 0.73)
        .expect("warm PF90116 dispatch");
    let (warm_alloc, warm_sync) = allocations(before, driver_metrics());
    let (_, max_delta) = compare(&expected.pixels, &first.pixels);
    assert_eq!(first, expected);
    assert_eq!(warm, expected);
    assert_eq!(warm_alloc, 0);
    assert_eq!(warm_sync, 0);
    assert_eq!(dispatch, warm_dispatch);

    let stats = measure(
        || {
            let (_, _) = gpu.compose(&source, &color, &mask, 0.73).unwrap();
        },
        if width == 512 { WARM_512 } else { WARM_24MP },
    );
    Case {
        name: if width == 512 {
            "pf90116_512"
        } else {
            "pf90116_24mp"
        },
        width,
        height,
        bytes: first.pixels.len(),
        first_alloc,
        warm_alloc,
        first_sync,
        warm_sync,
        first_hash: fnv64(&first.pixels),
        parity: true,
        max_delta,
        selected_nonzero: 0,
        coverage_nonzero: 0,
        selected_hash: 0,
        coverage_hash: 0,
        kernels: dispatch
            .first()
            .map_or(String::new(), |item| item.kernel.to_string()),
        device: dispatch
            .first()
            .map_or(String::new(), |item| item.device.clone()),
        stats,
    }
}

fn run_pf30025(width: u32, height: u32) -> Case {
    let source = pattern(width, height, 43);
    let pixels = source.pixel_count();
    let repair = mask_bytes(pixels * 4, 47);
    let gate = plane(pixels, 53);
    let roi = face_detail::Rect {
        x: (width / 20) as i32,
        y: (height / 20) as i32,
        width: (width * 9 / 10) as i32,
        height: (height * 9 / 10) as i32,
    };
    let producer = face_detail::ToothRepairProducer {
        repair_780929437_bgra: &repair,
        gate_3204548346: &gate,
        roi,
    };
    let expected = face_detail::tooth_repair_same_job(&source, Some(producer), 83.0).unwrap();
    let gpu = ToothRepair::open().expect("required CUDA device");
    let job = cuda_pf30025::Job {
        source: &source,
        repair_780929437_bgra: &repair,
        gate_3204548346: &gate,
        roi,
        ui: 83.0,
    };

    let before = driver_metrics();
    let (first, dispatch) = gpu.apply(&job).expect("PF30025 dispatch");
    let (first_alloc, first_sync) = allocations(before, driver_metrics());

    let before = driver_metrics();
    let (warm, warm_dispatch) = gpu.apply(&job).expect("warm PF30025 dispatch");
    let (warm_alloc, warm_sync) = allocations(before, driver_metrics());
    let (_, max_delta) = compare(&expected.pixels, &first.pixels);
    assert_eq!(first, expected);
    assert_eq!(warm, expected);
    assert_eq!(warm_alloc, 0);
    assert_eq!(warm_sync, 0);
    assert_eq!(dispatch, warm_dispatch);

    let stats = measure(
        || {
            let (_, _) = gpu.apply(&job).unwrap();
        },
        if width == 512 { WARM_512 } else { WARM_24MP },
    );
    Case {
        name: if width == 512 {
            "pf30025_512"
        } else {
            "pf30025_24mp"
        },
        width,
        height,
        bytes: first.pixels.len(),
        first_alloc,
        warm_alloc,
        first_sync,
        warm_sync,
        first_hash: fnv64(&first.pixels),
        parity: true,
        max_delta,
        selected_nonzero: 0,
        coverage_nonzero: 0,
        selected_hash: 0,
        coverage_hash: 0,
        kernels: dispatch[0].kernel.to_string(),
        device: dispatch[0].device.clone(),
        stats,
    }
}

fn pf90021_volume() -> VolumeLut64 {
    let mut nodes = vec![0.0f32; cuda_pf90021::LUT_FLOATS];
    for z in 0..cuda_pf90021::LUT_EDGE {
        for y in 0..cuda_pf90021::LUT_EDGE {
            for x in 0..cuda_pf90021::LUT_EDGE {
                let at = ((z * cuda_pf90021::LUT_EDGE + y) * cuda_pf90021::LUT_EDGE + x) * 3;
                nodes[at] = (x + y) as f32 / 126.0;
                nodes[at + 1] = (y + z) as f32 / 126.0;
                nodes[at + 2] = (x + z) as f32 / 126.0;
            }
        }
    }
    VolumeLut64::from_nodes(nodes).unwrap()
}

fn run_pf90021(width: u32, height: u32) -> Case {
    let source = pattern(width, height, 59);
    let mask = mask_bytes(source.pixel_count(), 61);
    let volume = pf90021_volume();
    let runtime = RuntimeScalars {
        degree01: 0.73,
        opacity01: 0.91,
        shelter: 0.83,
        region: 0.67,
    };
    let expected = cuda_pf90021::apply_cpu(&source, &volume, &mask, runtime).unwrap();
    let gpu = Pf90021::open().expect("required CUDA device");

    let before = driver_metrics();
    let first = gpu
        .run(Request {
            source: &source,
            direction: Direction::Positive,
            lut: Some(&volume),
            mask_r: Some(&mask),
            runtime: Some(runtime),
        })
        .expect("PF90021 dispatch");
    let (first_alloc, first_sync) = allocations(before, driver_metrics());
    let Run::Executed {
        output: first,
        dispatch,
    } = first
    else {
        panic!("PF90021 bypassed");
    };

    let before = driver_metrics();
    let warm = gpu
        .run(Request {
            source: &source,
            direction: Direction::Positive,
            lut: Some(&volume),
            mask_r: Some(&mask),
            runtime: Some(runtime),
        })
        .expect("warm PF90021 dispatch");
    let (warm_alloc, warm_sync) = allocations(before, driver_metrics());
    let Run::Executed {
        output: warm,
        dispatch: warm_dispatch,
    } = warm
    else {
        panic!("warm PF90021 bypassed");
    };
    let (_, max_delta) = compare(&expected.pixels, &first.pixels);
    assert_eq!(first, expected);
    assert_eq!(warm, expected);
    assert_eq!(warm_alloc, 0);
    assert_eq!(warm_sync, 0);
    assert_eq!(dispatch, warm_dispatch);

    let stats = measure(
        || {
            let result = gpu
                .run(Request {
                    source: &source,
                    direction: Direction::Positive,
                    lut: Some(&volume),
                    mask_r: Some(&mask),
                    runtime: Some(runtime),
                })
                .unwrap();
            assert!(matches!(result, Run::Executed { .. }));
        },
        if width == 512 { WARM_512 } else { WARM_24MP },
    );
    Case {
        name: if width == 512 {
            "pf90021_512"
        } else {
            "pf90021_24mp"
        },
        width,
        height,
        bytes: first.pixels.len(),
        first_alloc,
        warm_alloc,
        first_sync,
        warm_sync,
        first_hash: fnv64(&first.pixels),
        parity: true,
        max_delta,
        selected_nonzero: 0,
        coverage_nonzero: 0,
        selected_hash: 0,
        coverage_hash: 0,
        kernels: format!("{},{}", dispatch.lut.kernel, dispatch.composite.kernel),
        device: dispatch.composite.device,
        stats,
    }
}

fn mesh_case(width: f32, height: f32) -> ([[f32; 2]; 4], [[f32; 2]; 4], [u16; 6]) {
    (
        [
            [2.0, 2.0],
            [width - 2.0, 2.0],
            [width - 2.0, height - 2.0],
            [2.0, height - 2.0],
        ],
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        [0, 1, 2, 0, 2, 3],
    )
}

fn run_mesh(vec: bool) -> Case {
    let (width, height) = if vec {
        (6000u32, 4000u32)
    } else {
        (512u32, 512u32)
    };
    let source = pattern(width, height, 67);
    let (dest, uv, ebo) = mesh_case(width as f32, height as f32);
    let mesh = Mesh {
        dest: &dest,
        source_uv: &uv,
        ebo: &ebo,
    };
    let rules = Rules {
        winding: Winding::PositiveAreaIsFront,
        cull: Cull::None,
        edge: Edge::Inclusive,
        overlap: Overlap::First,
    };
    let sampler = Sampler {
        wrap: Wrap::ClampToEdge,
    };
    let expected = raster::rasterize(&source, mesh, rules, sampler).unwrap();
    let gpu = aurora_portrait::cuda_mesh::MeshRaster::open().expect("required CUDA device");

    let before = driver_metrics();
    let first = gpu
        .rasterize(&source, mesh, rules, sampler)
        .expect("face mesh dispatch");
    let (first_alloc, first_sync) = allocations(before, driver_metrics());

    let before = driver_metrics();
    let warm = gpu
        .rasterize(&source, mesh, rules, sampler)
        .expect("warm face mesh dispatch");
    let (warm_alloc, warm_sync) = allocations(before, driver_metrics());
    let (_, max_delta) = compare(&expected.pixels, &first.image.pixels);
    assert_eq!(first.image, expected);
    assert_eq!(warm.image, expected);
    assert_eq!(warm.selected, first.selected);
    assert_eq!(warm.coverage, first.coverage);
    assert_eq!(warm_alloc, 0);
    assert_eq!(warm_sync, 0);
    assert_eq!(first.select_dispatch, warm.select_dispatch);
    assert_eq!(first.emit_dispatch, warm.emit_dispatch);
    let selected_nonzero = first.selected.iter().filter(|value| **value != 0).count();
    let coverage_nonzero = first.coverage.iter().filter(|value| **value != 0).count();
    assert!(selected_nonzero > 0, "selection plane stayed empty");
    assert!(coverage_nonzero > 0, "coverage plane stayed empty");

    let stats = measure(
        || {
            let output = gpu.rasterize(&source, mesh, rules, sampler).unwrap();
            assert_eq!(output.image.width, width);
        },
        if vec { WARM_24MP } else { WARM_512 },
    );
    Case {
        name: if vec {
            "face_mesh_raster_24mp"
        } else {
            "face_mesh_raster_512"
        },
        width,
        height,
        bytes: first.image.pixels.len(),
        first_alloc,
        warm_alloc,
        first_sync,
        warm_sync,
        first_hash: fnv64(&first.image.pixels),
        parity: true,
        max_delta,
        selected_nonzero: selected_nonzero as u64,
        coverage_nonzero: coverage_nonzero as u64,
        selected_hash: fnv64_u32(&first.selected),
        coverage_hash: fnv64_u32(&first.coverage),
        kernels: format!(
            "{},{}",
            first.select_dispatch.kernel, first.emit_dispatch.kernel
        ),
        device: first.emit_dispatch.device.clone(),
        stats,
    }
}

fn run_mesh_fragment_512() -> Case {
    let (width, height) = (512u32, 512u32);
    let source = pattern(width, height, 71);
    let target = pattern(width, height, 73);
    let (dest, uv, ebo) = mesh_case(width as f32, height as f32);
    let mesh = Mesh {
        dest: &dest,
        source_uv: &uv,
        ebo: &ebo,
    };
    let rules = Rules {
        winding: Winding::PositiveAreaIsFront,
        cull: Cull::None,
        edge: Edge::Inclusive,
        overlap: Overlap::Last,
    };
    let sampler = Sampler {
        wrap: Wrap::ClampToEdge,
    };
    let defines = raster_fragment::Defines {
        clamp_edge: true,
        uv_map: false,
        uv_map_ba: false,
        catmull_rom: true,
    };
    let clamp = raster_fragment::Clamp::disabled();
    let tolerance = raster::Tolerance::RESHAPE;
    let expected = raster::rasterize_with_fragment(
        &source,
        mesh,
        rules,
        sampler,
        raster_fragment::Edge::ReshapeNormal,
        defines,
        clamp,
        tolerance,
    )
    .unwrap();
    let gpu = aurora_portrait::cuda_mesh::MeshRaster::open().expect("required CUDA device");

    let before = driver_metrics();
    let first = gpu
        .rasterize_fragment_into(
            &source,
            &target,
            mesh,
            rules,
            sampler,
            raster_fragment::Edge::ReshapeNormal,
            defines,
            clamp,
            tolerance,
        )
        .expect("face mesh fragment dispatch");
    let (first_alloc, first_sync) = allocations(before, driver_metrics());

    let before = driver_metrics();
    let warm = gpu
        .rasterize_fragment_into(
            &source,
            &target,
            mesh,
            rules,
            sampler,
            raster_fragment::Edge::ReshapeNormal,
            defines,
            clamp,
            tolerance,
        )
        .expect("warm face mesh fragment dispatch");
    let (warm_alloc, warm_sync) = allocations(before, driver_metrics());
    let (_, max_delta) = compare(&expected.pixels, &first.image.pixels);
    assert_eq!(first.image, expected);
    assert_eq!(warm.image, expected);
    assert_eq!(first.selected, warm.selected);
    assert_eq!(first.coverage, warm.coverage);
    assert_eq!(warm_alloc, 0);
    assert_eq!(warm_sync, 0);

    let stats = measure(
        || {
            let output = gpu
                .rasterize_fragment_into(
                    &source,
                    &target,
                    mesh,
                    rules,
                    sampler,
                    raster_fragment::Edge::ReshapeNormal,
                    defines,
                    clamp,
                    tolerance,
                )
                .unwrap();
            assert_eq!(output.image.width, width);
        },
        WARM_512,
    );
    Case {
        name: "face_mesh_fragment_512",
        width,
        height,
        bytes: first.image.pixels.len(),
        first_alloc,
        warm_alloc,
        first_sync,
        warm_sync,
        first_hash: fnv64(&first.image.pixels),
        parity: true,
        max_delta,
        selected_nonzero: first.selected.iter().filter(|value| **value != 0).count() as u64,
        coverage_nonzero: first.coverage.iter().filter(|value| **value != 0).count() as u64,
        selected_hash: fnv64_u32(&first.selected),
        coverage_hash: fnv64_u32(&first.coverage),
        kernels: format!(
            "{},{}",
            first.select_dispatch.kernel, first.emit_dispatch.kernel
        ),
        device: first.emit_dispatch.device.clone(),
        stats,
    }
}

fn decode_u32_fixture(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn decode_pair_fixture(bytes: &[u8]) -> Vec<[f32; 2]> {
    bytes
        .chunks_exact(12)
        .map(|chunk| {
            [
                f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                f32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
            ]
        })
        .collect()
}

fn run_mesh_attribute_512() -> Case {
    const FACES: &[u8] = include_bytes!("../../../assets/portrait/smile/triangles.u32le");
    const MEAN: &[u8] = include_bytes!("../../../assets/portrait/smile/mean.f32le");
    let (width, height) = (512u32, 512u32);
    let mean = decode_pair_fixture(MEAN);
    let faces = decode_u32_fixture(FACES);
    let dest: Vec<[f32; 2]> = mean
        .iter()
        .map(|point| [(point[0] + 80.0) * 2.6, (80.0 - point[1]) * 1.6])
        .collect();
    let attr: Vec<[f32; 2]> = mean
        .iter()
        .map(|point| [point[0] * 0.01, point[1] * -0.02])
        .collect();
    let expected = mesh_attribute::rasterize(width, height, &dest, &attr, &faces).unwrap();
    let gpu = MeshAttribute::open().expect("required CUDA device");

    let before = driver_metrics();
    let (first, dispatches) = gpu
        .rasterize(width, height, &dest, &attr, &faces)
        .expect("face attribute dispatch");
    let (first_alloc, first_sync) = allocations(before, driver_metrics());

    let before = driver_metrics();
    let (warm, warm_dispatches) = gpu
        .rasterize(width, height, &dest, &attr, &faces)
        .expect("warm face attribute dispatch");
    let (warm_alloc, warm_sync) = allocations(before, driver_metrics());
    assert_eq!(first.writes, expected.writes);
    assert_eq!(warm.writes, expected.writes);
    for (left, right) in expected.values.iter().zip(&first.values) {
        assert_eq!(left[0].to_bits(), right[0].to_bits());
        assert_eq!(left[1].to_bits(), right[1].to_bits());
    }
    assert_eq!(warm.values, first.values);
    assert_eq!(warm_dispatches, dispatches);
    assert_eq!(warm_alloc, 0);
    assert_eq!(warm_sync, 0);
    assert_eq!(dispatches.len(), 2);

    let stats = measure(
        || {
            gpu.rasterize(width, height, &dest, &attr, &faces).unwrap();
        },
        WARM_512,
    );
    Case {
        name: "face_attribute_512",
        width,
        height,
        bytes: std::mem::size_of_val(first.values.as_slice()),
        first_alloc,
        warm_alloc,
        first_sync,
        warm_sync,
        first_hash: fnv64_pairs(&first.values),
        parity: first.values == expected.values,
        max_delta: 0,
        selected_nonzero: first.writes.iter().filter(|value| **value > 0).count() as u64,
        coverage_nonzero: first.writes.iter().filter(|value| **value > 1).count() as u64,
        selected_hash: fnv64_u32(
            &first
                .writes
                .iter()
                .map(|value| u32::from(*value))
                .collect::<Vec<_>>(),
        ),
        coverage_hash: fnv64_pairs(&first.values),
        kernels: format!("{},{}", dispatches[0].kernel, dispatches[1].kernel),
        device: dispatches[1].device.clone(),
        stats,
    }
}

fn push_case(cases: &mut Vec<Case>, case: Case) {
    eprintln!(
        "N35_CASE {} {}x{} bytes={} first_alloc={} warm_alloc={} first_sync={} warm_sync={} parity={} max_delta={} selected_nonzero={} coverage_nonzero={} selected_hash={:016X} coverage_hash={:016X} p50_ms={:.4} p95_ms={:.4} kernels={} device={}",
        case.name,
        case.width,
        case.height,
        case.bytes,
        case.first_alloc,
        case.warm_alloc,
        case.first_sync,
        case.warm_sync,
        case.parity,
        case.max_delta,
        case.selected_nonzero,
        case.coverage_nonzero,
        case.selected_hash,
        case.coverage_hash,
        case.stats.p50_ms,
        case.stats.p95_ms,
        case.kernels,
        case.device
    );
    cases.push(case);
}

    pub(crate) fn main() {
    assert!(std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_some());
    let mut cases = Vec::new();
    push_case(&mut cases, run_facial_smooth(6000, 4000));
    push_case(&mut cases, run_eye_bag(6000, 4000));
    push_case(&mut cases, run_pf30025(6000, 4000));
    push_case(&mut cases, run_pf90021(6000, 4000));
    push_case(&mut cases, run_mesh_attribute_512());
    push_case(&mut cases, run_mesh(true));

    println!("{{");
    println!("  \"status\": \"PASS\",");
    println!("  \"device_required\": \"NVIDIA GeForce RTX 4070 Ti\",");
    println!("  \"samples_24mp\": {WARM_24MP},");
    println!("  \"samples_512\": {WARM_512},");
    println!("  \"cases\": [");
    for (index, case) in cases.iter().enumerate() {
        println!(
            "    {{\"name\":\"{}\",\"width\":{},\"height\":{},\"bytes\":{},\"first_alloc\":{},\"warm_alloc\":{},\"first_sync\":{},\"warm_sync\":{},\"hash64\":\"{:016X}\",\"parity\":{},\"max_delta\":{},\"selected_nonzero\":{},\"coverage_nonzero\":{},\"selected_hash\":\"{:016X}\",\"coverage_hash\":\"{:016X}\",\"p50_ms\":{:.6},\"p95_ms\":{:.6},\"min_ms\":{:.6},\"max_ms\":{:.6},\"kernels\":\"{}\",\"device\":\"{}\"}}{}",
            case.name,
            case.width,
            case.height,
            case.bytes,
            case.first_alloc,
            case.warm_alloc,
            case.first_sync,
            case.warm_sync,
            case.first_hash,
            case.parity,
            case.max_delta,
            case.selected_nonzero,
            case.coverage_nonzero,
            case.selected_hash,
            case.coverage_hash,
            case.stats.p50_ms,
            case.stats.p95_ms,
            case.stats.min_ms,
            case.stats.max_ms,
            case.kernels,
            case.device,
            if index + 1 == cases.len() { "" } else { "," }
        );
    }
    println!("  ]");
    println!("}}");
}
}

#[cfg(windows)]
fn main() {
    win_example::main()
}
