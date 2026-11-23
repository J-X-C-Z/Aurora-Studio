use super::*;

fn type1011_input(
    source: &Image,
    slim: &SlimProducerInputs,
    pf: u32,
) -> FaceGeometryInputs {
    FaceGeometryInputs {
        job_id: slim.job_id,
        width: source.width,
        height: source.height,
        draws: vec![FaceGeometryDraw {
            pf,
            controls: [0.5, 0.5],
            source_sha256: sha256_32(&source.pixels),
            geometry: FaceGeometry::Type1011 {
                prepared: (0..100)
                    .map(|index| {
                        [
                            0.4 + (index % 10) as f32 * 0.75,
                            0.4 + (index / 10) as f32 * 0.75,
                        ]
                    })
                    .collect(),
                param_list: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                center: [0.5, 0.5],
            },
        }],
    }
}

#[test]
fn routes_cover_available_face_and_eye_draws() {
    let requested = (8001..=8033)
        .filter(|pf| !matches!(pf, 8025 | 8026))
        .chain([
            242, 243, 247, 248, 30073, 30079, 30080, 30081, 30082, 30083, 44124, 48019,
            49221, 41922,
        ]);
    let mut missing = Vec::new();
    let mut supported = 0;
    for pf in requested {
        if let Some((left, right)) = pair(pf) {
            assert!(pf == left || pf == right);
            assert_eq!(pair(right), Some((left, right)));
            assert!(aurora_portrait::lookup(pf).is_some(), "PF{pf}");
            supported += 1;
        } else {
            missing.push(pf);
        }
    }
    assert_eq!(supported, 39);
    assert_eq!(missing, [8003, 8004, 8007, 8008, 8031, 8032]);
}

#[test]
fn prepared_draws_change_pixels_and_refuse_stale_bindings() {
    let (source, mut slim) = super::super::portrait_slim_stage::tests::fixture();
    let mut input = FaceGeometryInputs {
        job_id: slim.job_id,
        width: source.width,
        height: source.height,
        draws: vec![FaceGeometryDraw {
            pf: 242,
            controls: [0.0, 0.5],
            source_sha256: sha256_32(&source.pixels),
            geometry: FaceGeometry::V2Positions(vec![(0, [2.0, 0.0])]),
        }],
    };
    let controls = [0.0, 0.5];
    let mut offset = slim.base.clone();
    offset.fill([0.5, 0.0, 0.0, 0.0]);
    let size = (source.width * source.height) as usize;
    // The shared geometry remains valid across draws with their own digest.
    slim.source_sha256[0] ^= 1;
    for (pf, geometry) in [
        (242, input.draws[0].geometry.clone()),
        (
            247,
            FaceGeometry::VFace {
                rows: vec![Row::RightPositive],
                encoded_offset: offset,
                atlases: Vec::new(),
            },
        ),
        (
            8019,
            FaceGeometry::Eye {
                dest: vec![[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]],
                uv: vec![[0.25, 0.0], [1.0, 0.0], [0.25, 1.0]],
                ebo: vec![0, 1, 2],
                rules: slim.rules,
                edge: raster_fragment::Edge::ReshapeNormal,
                defines: raster_fragment::Defines::plain(),
                clamp: raster_fragment::Clamp::disabled(),
            },
        ),
        (
            8033,
            FaceGeometry::Smile {
                dx: vec![2.0; size],
                dy: vec![0.0; size],
                mask: vec![255; size],
                host_split_receipt: Some(super::SmileHostSplitReceipt {
                    job_id: slim.job_id,
                    model_output_id: super::super::SmileRgbReconstructionField::OUTPUT_ID,
                    model_output_sha256: super::sha256_32(b"780929437 fixture"),
                    dx_sha256: super::hash_f32(&vec![2.0; size]),
                    dy_sha256: super::hash_f32(&vec![0.0; size]),
                    mask_sha256: super::sha256_32(&vec![255; size]),
                }),
            },
        ),
    ] {
        let controls = if pf == 8033 { [0.5, 0.0] } else { controls };
        input.draws[0].pf = pf;
        input.draws[0].controls = controls;
        input.draws[0].geometry = geometry;
        let run = |job, bundle: &FaceGeometryInputs, v2| {
            apply(
                &source,
                NormRect::FULL,
                Some(job),
                Some(bundle),
                v2,
                pf,
                controls,
            )
        };
        let output = run(slim.job_id, &input, Some(&slim))
            .unwrap()
            .image
            .unwrap();
        let changed_pixels = output
            .pixels
            .chunks_exact(4)
            .zip(source.pixels.chunks_exact(4))
            .filter(|(after, before)| after != before)
            .count();
        assert!(changed_pixels > 0, "PF{pf}");
        println!("PF{pf}: {changed_pixels} changed pixels");
        assert_eq!(
            run(slim.job_id + 1, &input, Some(&slim)),
            Err(Refusal::ValueOutOfRange("face geometry job ticket")),
        );
        if pf == 242 || pf == 247 {
            assert!(matches!(
                run(slim.job_id, &input, None),
                Err(Refusal::MissingInput(_))
            ));
        }
        let hash = input.fingerprint();
        input.draws[0].source_sha256[0] ^= 1;
        assert_ne!(hash, input.fingerprint());
        assert_eq!(
            run(slim.job_id, &input, Some(&slim)),
            Err(Refusal::ValueOutOfRange("face geometry frame digest")),
        );
        input.draws[0].source_sha256[0] ^= 1;
        input.draws[0].controls[0] = 0.25;
        assert!(run(slim.job_id, &input, Some(&slim)).is_err());
    }
}

#[test]
fn remaining_geometry_rows_return_named_same_job_input_refusals() {
    let (source, slim) = super::super::portrait_slim_stage::tests::fixture();
    let input = FaceGeometryInputs {
        job_id: slim.job_id,
        width: source.width,
        height: source.height,
        draws: Vec::new(),
    };
    let cases = [
        (
            30073,
            "PF30073/PF30079 Type1011 EBO 0x60F2B60 and frame draw",
        ),
        (
            30079,
            "PF30073/PF30079 Type1011 EBO 0x60F2B60 and frame draw",
        ),
        (
            30080,
            "PF30080..PF30083 Type1012 texture/V2 handoff",
        ),
        (
            30081,
            "PF30080..PF30083 Type1012 texture/V2 handoff",
        ),
        (
            30082,
            "PF30080..PF30083 Type1012 texture/V2 handoff",
        ),
        (
            30083,
            "PF30080..PF30083 Type1012 texture/V2 handoff",
        ),
        (
            44124,
            "PF44124/PF48019/PF49221/PF41922 Type1018 EBO 0x7080940 and frame draw",
        ),
        (
            48019,
            "PF44124/PF48019/PF49221/PF41922 Type1018 EBO 0x7080940 and frame draw",
        ),
        (
            49221,
            "PF44124/PF48019/PF49221/PF41922 Type1018 EBO 0x7080940 and frame draw",
        ),
        (
            41922,
            "PF44124/PF48019/PF49221/PF41922 Type1018 EBO 0x7080940 and frame draw",
        ),
    ];
    for (pf, missing) in cases {
        assert_eq!(
            apply(
                &source,
                NormRect::FULL,
                Some(slim.job_id),
                Some(&input),
                Some(&slim),
                pf,
                [0.5, 0.5],
            ),
            Err(Refusal::MissingInput(missing)),
            "PF{pf}"
        );
    }
}

#[test]
fn normal_stage_routes_remaining_geometry_to_typed_refusals() {
    use std::collections::BTreeMap;

    let (source, slim) = super::super::portrait_slim_stage::tests::fixture();
    let masks = super::super::PortraitMasks::solid(1.0);
    let input = FaceGeometryInputs {
        job_id: slim.job_id,
        width: source.width,
        height: source.height,
        draws: Vec::new(),
    };
    let mut job = super::super::PortraitJobInputs::from_renderer_job(slim.job_id, &masks);
    job.slim = Some(&slim);
    job.face_geometry = Some(&input);
    let pfs = [
        30073, 30079, 30080, 30081, 30082, 30083, 44124, 48019, 49221, 41922,
    ];
    let controls = pfs
        .iter()
        .map(|pf| (format!("pc:{pf}"), 0.5f32))
        .collect::<BTreeMap<_, _>>();
    let mut pixels = source.pixels.clone();
    let report = super::super::apply_portrait_effects_with_job(
        &mut pixels,
        source.width,
        source.height,
        NormRect::FULL,
        &controls,
        &job,
    );
    assert_eq!(report.effects.len(), pfs.len());
    for effect in report.effects {
        assert_eq!(
            effect.refusal,
            Some(Refusal::MissingInput(missing_draw(effect.pf))),
            "PF{}",
            effect.pf
        );
        assert!(effect.skipped, "PF{}", effect.pf);
        assert!(!effect.executed, "PF{}", effect.pf);
    }
    assert_eq!(pixels, source.pixels);
}

#[cfg(windows)]
#[test]
fn remaining_eye_routes_use_cuda_when_available_and_fallback_matches_bytes() {
    let (source, slim) = super::super::portrait_slim_stage::tests::fixture();
    let geometry = FaceGeometry::Eye {
        dest: vec![[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]],
        uv: vec![[0.25, 0.0], [1.0, 0.0], [0.25, 1.0]],
        ebo: vec![0, 1, 2],
        rules: slim.rules,
        edge: raster_fragment::Edge::ReshapeNormal,
        defines: raster_fragment::Defines::plain(),
        clamp: raster_fragment::Clamp::disabled(),
    };
    for pf in [44124, 48019, 49221, 41922] {
        let input = FaceGeometryInputs {
            job_id: slim.job_id,
            width: source.width,
            height: source.height,
            draws: vec![FaceGeometryDraw {
                pf,
                controls: [0.5, 0.5],
                source_sha256: sha256_32(&source.pixels),
                geometry: geometry.clone(),
            }],
        };
        let cpu = {
            super::super::FORCE_CUDA_FAILURE.with(|flag| flag.set(true));
            let result = apply(
                &source,
                NormRect::FULL,
                Some(slim.job_id),
                Some(&input),
                Some(&slim),
                pf,
                [0.5, 0.5],
            );
            super::super::FORCE_CUDA_FAILURE.with(|flag| flag.set(false));
            result.unwrap()
        };
        assert_eq!(cpu.backend, PortraitBackend::Cpu, "PF{pf}");
        assert!(cpu.fallback_reason.is_some(), "PF{pf}");

        let cuda = apply(
            &source,
            NormRect::FULL,
            Some(slim.job_id),
            Some(&input),
            Some(&slim),
            pf,
            [0.5, 0.5],
        )
        .unwrap();
        if cuda.backend == PortraitBackend::Cuda {
            assert_eq!(cuda.image, cpu.image, "PF{pf}");
            assert_eq!(cuda.dispatches.len(), 2, "PF{pf}");
            eprintln!(
                "PF{pf}: CUDA {} on {}",
                cuda.dispatches[0].kernel, cuda.dispatches[0].device
            );
        } else {
            assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but PF{pf} fell back: {:?}",
                cuda.fallback_reason
            );
        }
    }
}

#[test]
fn type1011_prepared100_routes_to_face_mesh_without_changing_landmark_domains() {
    let (source, slim) = super::super::portrait_slim_stage::tests::fixture();
    for pf in [30073, 30079] {
        let input = type1011_input(&source, &slim, pf);
        let run = apply(
            &source,
            NormRect::FULL,
            Some(slim.job_id),
            Some(&input),
            Some(&slim),
            pf,
            [0.5, 0.5],
        )
        .unwrap();
        assert!(run.image.is_some(), "PF{pf}");
        if run.backend == PortraitBackend::Cuda {
            assert!(!run.dispatches.is_empty(), "PF{pf}");
        } else {
            assert_eq!(run.backend, PortraitBackend::Cpu, "PF{pf}");
        }

        let mut wrong_194 = slim.clone();
        wrong_194.points194.pop();
        assert_eq!(
            apply(
                &source,
                NormRect::FULL,
                Some(slim.job_id),
                Some(&input),
                Some(&wrong_194),
                pf,
                [0.5, 0.5],
            ),
            Err(Refusal::ShapeMismatch(
                "Type1011 same-job 194 landmarks"
            ))
        );
    }
}

/// The recovered eye and brow rows ride the same CUDA fragment contract as the
/// WL rows once the caller supplies an index buffer, so they must agree with
/// the CPU raster byte for byte rather than stay pinned to CPU.
#[cfg(windows)]
#[test]
fn eye_and_brow_family_routes_use_cuda_when_available_and_fallback_matches_bytes() {
    let (source, slim) = super::super::portrait_slim_stage::tests::fixture();
    let geometry = FaceGeometry::Eye {
        dest: vec![[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]],
        uv: vec![[0.25, 0.0], [1.0, 0.0], [0.25, 1.0]],
        ebo: aurora_portrait::kernels::eye_geometry::QUAD_INDICES[..3].to_vec(),
        rules: slim.rules,
        edge: raster_fragment::Edge::ReshapeNormal,
        defines: raster_fragment::Defines::plain(),
        clamp: raster_fragment::Clamp::disabled(),
    };
    // Only the left row of each pair is dispatched; the right control rides as
    // `controls[1]`, matching the stage's pair contract.
    for pf in [8001u32, 8005, 8009, 8011, 8013, 8015, 8017, 8019, 8021] {
        let input = FaceGeometryInputs {
            job_id: slim.job_id,
            width: source.width,
            height: source.height,
            draws: vec![FaceGeometryDraw {
                pf,
                controls: [0.5, 0.0],
                source_sha256: sha256_32(&source.pixels),
                geometry: geometry.clone(),
            }],
        };
        let run = |force_cuda_failure: bool| {
            if force_cuda_failure {
                super::super::FORCE_CUDA_FAILURE.with(|flag| flag.set(true));
            }
            let result = apply(
                &source,
                NormRect::FULL,
                Some(slim.job_id),
                Some(&input),
                Some(&slim),
                pf,
                [0.5, 0.0],
            );
            if force_cuda_failure {
                super::super::FORCE_CUDA_FAILURE.with(|flag| flag.set(false));
            }
            result.unwrap()
        };
        let cpu = run(true);
        assert_eq!(cpu.backend, PortraitBackend::Cpu, "PF{pf}");
        assert!(cpu.fallback_reason.is_some(), "PF{pf}");

        let cuda = run(false);
        if cuda.backend == PortraitBackend::Cuda {
            assert_eq!(cuda.image, cpu.image, "PF{pf}");
            assert_eq!(cuda.dispatches.len(), 2, "PF{pf}");
            eprintln!(
                "PF{pf}: CUDA {} on {}",
                cuda.dispatches[0].kernel, cuda.dispatches[0].device
            );
        } else {
            assert_eq!(cuda.image, cpu.image, "PF{pf} fallback bytes differ");
            assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but PF{pf} fell back: {:?}",
                cuda.fallback_reason
            );
        }
    }
}

/// A registry row whose producer is callable but unrouted must say which
/// inputs are missing; only a row with no kernel stays `NotImplemented`.
#[test]
fn producer_only_rows_report_named_inputs_and_unrouted_rows_stay_unimplemented() {
    for (pf, inputs) in [
        (202u32, "194 landmarks + 135 weights + control slots"),
        (232, "landmarks 192..206 + 15 factors"),
    ] {
        let meta = aurora_portrait::lookup(pf).expect("registry row");
        let gap = producer_gap(pf).expect("named producer gap");
        for input in inputs.split(" + ") {
            assert!(
                meta.inputs.iter().any(|known| *known == input),
                "PF{pf} {input}"
            );
            assert!(gap.contains(input), "PF{pf} gap omits {input}");
        }
        assert!(pair(pf).is_none(), "PF{pf} must stay unrouted");
    }
    // 8003/8004/8031/8032 have no registry row at all, and the eye/brow family
    // keeps its kernel-owned EBO names instead of a generic label.
    for pf in [8003u32, 8004, 8007, 8008, 8031, 8032] {
        assert!(
            aurora_portrait::lookup(pf).is_none(),
            "PF{pf} has no registry row"
        );
        assert_eq!(producer_gap(pf), None, "PF{pf}");
    }
    for pf in [8001u32, 8005, 8009] {
        assert_eq!(
            missing_draw(pf),
            aurora_portrait::kernels::eye_geometry::ebo_rva(pf).unwrap(),
            "PF{pf}"
        );
    }
    for pf in [8011u32, 8013, 8019] {
        assert_eq!(missing_draw(pf), "same-job face geometry", "PF{pf}");
    }
}
