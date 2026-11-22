use super::*;
use aurora_portrait::raster::{Cull, Edge, Overlap, Winding};

pub(crate) fn fixture() -> (Image, SlimProducerInputs) {
    let mut source = Image::solid(8, 8, [0, 0, 0, 255]);
    for (i, rgba) in source.pixels.chunks_exact_mut(4).enumerate() {
        rgba[0] = (i % 8 * 32) as u8;
    }
    let mut positions = vec![[0.0, 0.0]; 526];
    positions[..4].copy_from_slice(&[[0.0, 0.0], [8.0, 0.0], [0.0, 8.0], [8.0, 8.0]]);
    let mut texcoord = vec![[0.0, 0.0]; 526];
    texcoord[..4].copy_from_slice(&[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]);
    let mut offset_texcoord = vec![[0.0, 0.0]; 526];
    offset_texcoord[..4].copy_from_slice(&[[0.10, 0.5], [0.90, 0.5], [0.10, 0.5], [0.90, 0.5]]);
    let mut ebo = vec![0; 2982];
    ebo[..6].copy_from_slice(&[0, 1, 2, 1, 3, 2]);
    let inputs = SlimProducerInputs {
        job_id: 28,
        width: source.width,
        height: source.height,
        source_sha256: sha256_32(&source.pixels),
        grid: [255, 0, 0, 0].repeat(255 * 171),
        base: vec![[0.0; 4]; 280 * 280],
        points194: (0..194)
            .map(|i| [(i % 8) as f32, (i / 25) as f32])
            .collect(),
        framebuffer_positions_w1: positions,
        texcoord,
        offset_texcoord,
        face_scales: vec![[1.0, 1.0]; 526],
        ebo,
        rules: Rules {
            winding: Winding::PositiveAreaIsFront,
            cull: Cull::None,
            edge: Edge::Inclusive,
            overlap: Overlap::First,
        },
        mat2: [1.0, 0.0, 0.0, 1.0],
        f_o: [8.0, 8.0],
    };
    (source, inputs)
}

#[test]
fn slim_passes_paired_degrees_to_existing_v2_and_produces_pixels() {
    let (source, input) = fixture();
    let run = |ui| {
        apply(&source, NormRect::FULL, Some(28), Some(&input), ui)
            .unwrap()
            .unwrap()
    };
    let paired = run([1.0, 0.5]);
    let expected = cuda_v2fuseoffset::apply_cpu(&PipelineInput {
        ticket: 28,
        source: &source,
        grid: &input.grid,
        base: &input.base,
        left_degree: 1.5,
        right_degree: 0.75,
        points194: &input.points194,
        mesh: input.mesh(),
        rules: input.rules,
        mat2: input.mat2,
        f_o: input.f_o,
    })
    .unwrap();
    assert_eq!(paired, expected.image);
    assert_ne!(paired, source);
    assert_ne!(paired, run([1.0, 0.0]));
    assert_ne!(paired, run([0.0, 0.5]));
}

#[test]
fn slim_rejects_missing_stale_and_malformed_inputs() {
    let (source, mut input) = fixture();
    let run = |image: &Image, job, bundle: Option<&SlimProducerInputs>| {
        apply(image, NormRect::FULL, job, bundle, [0.8, 0.3])
    };
    assert!(matches!(
        run(&source, Some(28), None),
        Err(Refusal::MissingInput(_))
    ));
    assert!(run(&source, Some(29), Some(&input)).is_err());
    assert!(run(&source, None, Some(&input)).is_err());
    let mut stale = source.clone();
    stale.pixels[0] ^= 1;
    assert!(run(&stale, Some(28), Some(&input)).is_err());
    let mut crop = NormRect::FULL;
    crop.width = 0.5;
    assert!(apply(&source, crop, Some(28), Some(&input), [0.8, 0.3]).is_err());
    assert!(
        apply(
            &source,
            NormRect::FULL,
            Some(28),
            Some(&input),
            [0.8, f32::NAN]
        )
        .is_err()
    );
    input.width -= 1;
    assert!(run(&source, Some(28), Some(&input)).is_err());
    input.width += 1;
    input.f_o[0] -= 1.0;
    assert!(run(&source, Some(28), Some(&input)).is_err());
    input.f_o[0] += 1.0;
    input.points194.pop();
    assert!(run(&source, Some(28), Some(&input)).is_err());
    input.points194.push([f32::NAN, 0.0]);
    assert!(run(&source, Some(28), Some(&input)).is_err());
    input.points194[193] = [0.0, 0.0];
    input.ebo[0] = 526;
    assert!(run(&source, Some(28), Some(&input)).is_err());
    input.ebo[0] = 0;
    input.grid.pop();
    assert!(run(&source, Some(28), Some(&input)).is_err());
    input.grid.push(0);
    input.base[0][0] = f32::INFINITY;
    assert!(run(&source, Some(28), Some(&input)).is_err());
    input.base[0][0] = 0.0;
    input.framebuffer_positions_w1.fill([0.0, 0.0]);
    assert!(run(&source, Some(28), Some(&input)).is_err());
    assert!(
        apply(&source, NormRect::FULL, None, None, [0.0, 0.0])
            .unwrap()
            .is_none()
    );
}

#[test]
fn slim_fingerprint_tracks_geometry_and_resources() {
    let (_, mut input) = fixture();
    let original = input.fingerprint();
    input.texcoord[0][0] = 0.02;
    let modified = input.fingerprint();
    assert_ne!(original, modified);
    input.base[0][0] = 0.5;
    assert_ne!(modified, input.fingerprint());
}
