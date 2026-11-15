
use aurora_portrait::kernels::{
    blemish_blend,
    beard_transform,
    body_geometry::{self, Clamp, EdgeMode, Mesh, Pass, PassDefines, Vec2},
    face_detail::teeth_whitening,
    facial_smooth::{self, Uniforms},
    mask_compose::{self, MaskComposeMode},
    skin_models,
};
use aurora_portrait::raster::{
    self, Cull, Edge as RasterEdge, Mesh as RasterMesh, Overlap, Rules, Sampler, Winding, Wrap,
};
use aurora_portrait::raster_fragment::{
    Clamp as FragmentClamp, Defines as FragmentDefines, Edge as FragmentEdge,
};
use aurora_portrait::Image;
use std::hint::black_box;
use std::time::Instant;

fn lcg(seed: &mut u64) -> u8 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 33) as u8
}

fn checksum(image: &Image) -> u64 {
    image.pixels.iter().fold(0u64, |sum, byte| sum.wrapping_mul(31).wrapping_add(*byte as u64))
}

fn assert_alpha_preserved(source: &Image, output: &Image) {
    assert_eq!(source.pixels.len(), output.pixels.len());
    for (source_px, output_px) in source.pixels.chunks_exact(4).zip(output.pixels.chunks_exact(4)) {
        assert_eq!(source_px[3], output_px[3], "alpha changed");
    }
}

fn measure<F>(mut run: F, source: &Image) -> (Vec<u128>, u64, usize)
where
    F: FnMut() -> Image,
{
    let mut times = Vec::new();
    let mut checksum_value = 0;
    let mut output_bytes = 0;
    for round in 0..=5 {
        let start = Instant::now();
        let out = run();
        let elapsed = start.elapsed().as_micros();
        if round == 0 {
            checksum_value = checksum(&out);
        } else {
            assert_eq!(checksum(&out), checksum_value, "nondeterministic output");
            times.push(elapsed);
        }
        output_bytes = out.pixels.len();
    }
    (times, checksum_value, output_bytes)
}

fn median(values: &mut [u128]) -> u128 {
    values.sort_unstable();
    values[values.len() / 2]
}

fn report(name: &str, side: u32, times: &mut [u128], checksum_value: u64, output_bytes: usize, input_bytes: usize) {
    println!("{name} {side}x{side}: median={}us input_bytes={input_bytes} output_bytes={output_bytes} checksum={checksum_value}", median(times));
}

fn build_mesh(side: u32) -> (Mesh, RasterMesh<'static>) {
    let columns = 17usize;
    let rows = 17usize;
    let mut source_uv = Vec::with_capacity(columns * rows);
    let mut destination = Vec::with_capacity(columns * rows);
    for y in 0..rows {
        for x in 0..columns {
            let u = x as f32 / (columns - 1) as f32;
            let v = y as f32 / (rows - 1) as f32;
            source_uv.push(Vec2::new(u, v));
            destination.push(Vec2::new(u * side as f32, v * side as f32));
        }
    }
    let indices = body_geometry::grid_indices(rows, columns).unwrap();
    let dest_f32: Vec<[f32; 2]> = destination.iter().map(|p| [p.x, p.y]).collect();
    let uv_f32: Vec<[f32; 2]> = source_uv.iter().map(|p| [p.x, p.y]).collect();
    let body_mesh = Mesh::new(source_uv, destination, indices.clone()).unwrap();
    let raster_mesh = RasterMesh {
        dest: Box::leak(dest_f32.into_boxed_slice()),
        source_uv: Box::leak(uv_f32.into_boxed_slice()),
        ebo: Box::leak(indices.into_boxed_slice()),
    };
    (body_mesh, raster_mesh)
}

fn raster_bench(side: u32) {
    let pixels = (side * side) as usize;
    let mut seed = 0xfeed_c0de_beef_face;
    let source = Image::new(side, side, (0..pixels * 4).map(|_| lcg(&mut seed)).collect()).unwrap();
    let (body_mesh, raster_mesh) = build_mesh(side);
    let pass = Pass {
        edge: EdgeMode::ReshapeNormal,
        defines: PassDefines::plain(),
        clamp: Clamp::disabled(),
        matrix: Pass::normalized_to_clip(),
    };
    let rules = Rules {
        winding: Winding::PositiveAreaIsFront,
        cull: Cull::None,
        edge: RasterEdge::Inclusive,
        overlap: Overlap::First,
    };
    let sampler = Sampler { wrap: Wrap::ClampToEdge };
    let fragment_edge = FragmentEdge::ReshapeNormal;
    let fragment_defines = FragmentDefines::plain();
    let fragment_clamp = FragmentClamp::disabled();

    let (mut times, sum, output_bytes) = measure(|| {
        raster::rasterize(black_box(&source), raster_mesh, rules, sampler).unwrap()
    }, &source);
    report("raster generic", side, &mut times, sum, output_bytes, source.pixels.len());

    let (mut times, sum, output_bytes) = measure(|| {
        raster::rasterize_with_fragment(black_box(&source), raster_mesh, rules, sampler, fragment_edge, fragment_defines, fragment_clamp, aurora_portrait::raster::Tolerance::TILE).unwrap()
    }, &source);
    report("raster fragment", side, &mut times, sum, output_bytes, source.pixels.len());

    let (mut times, sum, output_bytes) = measure(|| {
        body_geometry::reshape_mesh(black_box(&source), &body_mesh, &[pass]).unwrap()
    }, &source);
    report("body_geometry", side, &mut times, sum, output_bytes, source.pixels.len());
}

fn main() {
    for side in [256u32, 512u32] {
        raster_bench(side);
    }

    for side in [256u32, 1024u32] { teeth_bench(side); }
    for side in [256u32, 1024u32] {
        let pixels = (side * side) as usize;
        let mut seed = 0x1234_5678_9abc_def0;
        let mut random_rgba = || {
            let bytes = (0..4).map(|_| lcg(&mut seed)).collect::<Vec<_>>();
            [bytes[0], bytes[1], bytes[2], bytes[3]]
        };
        let source = Image::new(side, side, (0..pixels).flat_map(|_| random_rgba()).collect()).unwrap();
        let base = Image::new(side, side, (0..pixels).flat_map(|_| random_rgba()).collect()).unwrap();
        let repaired = Image::new(side, side, (0..pixels).flat_map(|_| random_rgba()).collect()).unwrap();
        let mask = (0..pixels).map(|_| lcg(&mut seed)).collect::<Vec<_>>();
        let delta = (0..pixels * 3).map(|_| lcg(&mut seed) as f32 / 255.0 - 0.5).collect::<Vec<_>>();

        let (mut times, sum, output_bytes) = measure(|| {
            mask_compose::apply(black_box(&source), black_box(&mask), black_box(MaskComposeMode::MaskOnly)).unwrap()
        }, &source);
        report("mask_compose", side, &mut times, sum, output_bytes, source.pixels.len() + mask.len());

        let (mut times, sum, output_bytes) = measure(|| {
            facial_smooth::apply(black_box(&source), black_box(&base), black_box(&repaired), black_box(&mask), black_box(&mask), black_box(&mask), black_box(uniforms())).unwrap()
        }, &source);
        report("facial_smooth serial", side, &mut times, sum, output_bytes, source.pixels.len() * 3 + mask.len() * 3);

        let (mut times, sum, output_bytes) = measure(|| {
            facial_smooth::apply_parallel(black_box(&source), black_box(&base), black_box(&repaired), black_box(&mask), black_box(&mask), black_box(&mask), black_box(uniforms())).unwrap()
        }, &source);
        report("facial_smooth parallel", side, &mut times, sum, output_bytes, source.pixels.len() * 3 + mask.len() * 3);

        let (mut times, sum, output_bytes) = measure(|| {
            blemish_blend::blend(black_box(&repaired), black_box(&base), black_box(&mask)).unwrap()
        }, &base);
        report("blemish_blend", side, &mut times, sum, output_bytes, repaired.pixels.len() + base.pixels.len() + mask.len());

        let (mut times, sum, output_bytes) = measure(|| {
            beard_transform::store_to_base(black_box(&base), black_box(&delta), black_box(1.0), black_box(0.0)).unwrap()
        }, &base);
        report("beard_transform", side, &mut times, sum, output_bytes, base.pixels.len() + delta.len() * 4);

        let (mut times, sum, output_bytes) = measure(|| {
            skin_models::gloss_apply(black_box(&source), [&mask; 3], [&mask; 3], black_box(75.0)).unwrap()
        }, &source);
        report("skin_models gloss", side, &mut times, sum, output_bytes, source.pixels.len() + mask.len() * 6);
    }
}

fn uniforms() -> Uniforms {
    Uniforms {
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

fn teeth_bench(side: u32) {
    let pixels = (side * side) as usize;
    let mut seed = 0x1234_5678_9abc_def0;
    let source = Image::new(side, side, (0..pixels * 4).map(|_| lcg(&mut seed)).collect()).unwrap();
    let mask = (0..pixels).map(|_| lcg(&mut seed)).collect::<Vec<_>>();
    let lut = (0..512 * 512 * 3).map(|_| lcg(&mut seed)).collect::<Vec<_>>();

    let reference = teeth_whitening(black_box(&source), black_box(&mask), black_box(&lut), black_box(&lut), black_box(50.0), black_box(50.0), black_box(50.0)).unwrap();
    let mut times = Vec::new();
    let mut checksum_value = 0;
    let mut output_bytes = 0;
    for round in 0..=5 {
        let start = Instant::now();
        let out = teeth_whitening(black_box(&source), black_box(&mask), black_box(&lut), black_box(&lut), black_box(50.0), black_box(50.0), black_box(50.0)).unwrap();
        let elapsed = start.elapsed().as_micros();
        assert_eq!(checksum(&out.output), checksum(&reference.output), "teeth nondeterministic");
        assert_eq!(out.output.pixels, reference.output.pixels, "teeth output differs");
        if round == 0 {
            checksum_value = checksum(&out.output);
        } else {
            times.push(elapsed);
        }
        output_bytes = out.output.pixels.len();
    }
    report("teeth_whitening", side, &mut times, checksum_value, output_bytes, source.pixels.len() + mask.len() + lut.len() * 2);
}

