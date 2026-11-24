use super::*;
use super::portrait_body_stage::BodyVendorParams;
use aurora_portrait::kernels::body_geometry::{Brush, Clamp, EdgeMode, Pass, PassDefines, Vec2};
use aurora_portrait::kernels::leg_refine_uv::LegColumn;

const SIZE: (u32, u32) = (8, 4);

fn frame() -> Image {
    Image::new(
        SIZE.0,
        SIZE.1,
        (0..128).map(|i| (i * 29 % 256) as u8).collect(),
    )
    .unwrap()
}

fn fixture(pf: u32) -> BodyGeometryInput {
    use BodyGeometryInput::*;
    let count = (SIZE.0 * SIZE.1) as usize;
    let shifted = vec![Vec2::new(0.7, 0.5); count];
    match pf {
        400 | 404 => SixPoint(SixPointWarp {
            source_points: [Vec2::new(0.5, 0.5); 6],
            destination_points: [Vec2::new(0.8, 0.5); 6],
            degrees: [1.0; 6],
            amount: 1.0,
            inverse_aspect: 1.0,
            valid_point_count: 1,
        }),
        401 | 406 | 410 | 415 => {
            let destination = vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(1.0, 0.0),
                Vec2::new(0.0, 1.0),
                Vec2::new(1.0, 1.0),
            ];
            Mesh {
                mesh: aurora_portrait::kernels::body_geometry::Mesh::new(
                    destination
                        .iter()
                        .map(|p| Vec2::new(p.x * 0.75, p.y))
                        .collect(),
                    destination,
                    vec![0, 1, 2, 1, 3, 2],
                )
                .unwrap(),
                passes: vec![Pass {
                    edge: EdgeMode::ReshapeNormal,
                    defines: PassDefines::plain(),
                    clamp: Clamp::disabled(),
                    matrix: Pass::normalized_to_clip(),
                }],
            }
        }
        405 => BodySlim {
            mask_r: vec![0; count],
            source_uv: shifted.clone(),
            target_uv: shifted,
        },
        407 => Leg {
            columns: std::array::from_fn(|i| LegColumn {
                sample_u: i as f32 / 4.0,
                clip_x: 0.30 + i as f32 * 0.05,
            }),
            catmull_size: None,
        },
        408 => Height {
            source_uv: shifted,
            destination_uv: (0..count)
                .map(|i| Vec2::new(((i % 8) as f32 + 0.5) / 8.0, ((i / 8) as f32 + 0.5) / 4.0))
                .collect(),
            defines: PassDefines::plain(),
        },
        416 => Waist {
            warped_uv: shifted,
            mask_r: vec![0.0; count],
        },
        38556 => BreastLift {
            degree01: 0.5,
            brushes: vec![Brush {
                position: Vec2::new(4.0, 2.0),
                direction: Vec2::new(1.0, 0.0),
                radius: 3.0,
                pressure: 1.0,
            }],
            density: vec![1.0],
            density_size: (1, 1),
            freeze_mask: vec![0; count],
            warp_direction: 1.0,
            edge_margin: 0.1,
            push_left: false,
        },
        90041 => Shift(BodyShiftInput {
            mask: vec![255; count],
            map_size: (1, 1),
            rgba8: vec![128, 1, 128, 0],
            ratio: 1.0,
            shift_size: SIZE,
        }),
        _ => panic!("missing fixture for PF{pf}"),
    }
}

fn producer(pf: u32, job_id: u64) -> BodyGeometryJob {
    BodyGeometryJob {
        job_id,
        frame_size: SIZE,
        source_sha256: sha256_32(&frame().pixels),
        effects: BTreeMap::from([(pf, fixture(pf))]),
        body_params: BodyVendorParams::for_effects(&BTreeMap::from([(pf, fixture(pf))])),
    }
}

const ROUTED: [u32; 12] = [
    400, 401, 404, 405, 406, 407, 408, 410, 415, 416, 38556, 90041,
];

#[test]
fn body_adapter_writes_pixels_for_each_supported_pf() {
    let source = frame();
    for pf in ROUTED {
        let body = producer(pf, 7);
        assert!(portrait_body_stage::supports(pf));
        let output = portrait_body_stage::apply(&source, pf, 50.0, Some(7), Some(&body))
            .unwrap()
            .unwrap();
        assert_eq!((output.width, output.height), SIZE, "PF{pf}");
        assert_ne!(output.pixels, source.pixels, "PF{pf}");
        if matches!(pf, 405 | 90041) {
            assert!(
                output
                    .pixels
                    .chunks_exact(4)
                    .zip(source.pixels.chunks_exact(4))
                    .all(|(a, b)| a[3] == b[3]),
                "PF{pf}"
            );
        }
    }
}

#[test]
fn body_adapter_checks_job_payload_and_color_passes() {
    let source = frame();
    let mut body = producer(401, 7);
    let run = |pf, ui, id, body: Option<&BodyGeometryJob>| {
        portrait_body_stage::apply(&source, pf, ui, Some(id), body)
    };
    assert_eq!(run(401, 0.0, 7, None), Ok(None));
    assert_eq!(
        run(401, f32::NAN, 7, Some(&body)),
        Err(Refusal::ValueOutOfRange("body geometry control"))
    );
    assert_eq!(
        run(401, 50.0, 7, None),
        Err(Refusal::MissingInput("body geometry job"))
    );
    assert_eq!(
        run(401, 50.0, 8, Some(&body)),
        Err(Refusal::ValueOutOfRange("body geometry job id"))
    );
    assert_eq!(
        run(405, 50.0, 7, Some(&body)),
        Err(Refusal::MissingInput("body geometry effect"))
    );
    let original = body.fingerprint();
    let BodyGeometryInput::Mesh { mesh, .. } = body.effects.get_mut(&401).unwrap() else {
        unreachable!();
    };
    mesh.indices[0] = u16::MAX;
    assert_eq!(
        run(401, 50.0, 7, Some(&body)),
        Err(Refusal::ValueOutOfRange("mesh indices"))
    );
    assert_ne!(body.fingerprint(), original);
    let BodyGeometryInput::Mesh { passes, .. } = body.effects.get_mut(&401).unwrap() else {
        unreachable!();
    };
    passes[0].defines.uv_map = true;
    assert_eq!(
        run(401, 50.0, 7, Some(&body)),
        Err(Refusal::ValueOutOfRange(
            "body geometry requires color passes"
        ))
    );
}

#[test]
fn body_vendor_params_are_typed_and_not_invented() {
    use aurora_portrait::kernels::body_geometry_params::{control_curve, HEAD_ZOOM, LEG_FACTORS};

    // Every routed PF except PF90041 has a decoded vendor control curve in the
    // two PoseRefine configs. PF90041 has no ParamFlag in the decoded tree.
    for pf in ROUTED {
        if pf != 90041 {
            assert!(control_curve(pf).is_some(), "PF{pf}");
        }
    }
    assert!(control_curve(90041).is_none());
    assert_eq!(control_curve(406).unwrap().default_degree(), Ok(0.0));
    assert_eq!(control_curve(407).unwrap().default_degree(), Ok(1.0));
    assert_eq!(control_curve(400).unwrap().default_degree(), Ok(0.0));
    assert_eq!(control_curve(405).unwrap().default_degree(), Ok(0.0));

    let body = producer(406, 7);
    assert_eq!(body.body_params.head_zoom, Some(HEAD_ZOOM));
    assert_eq!(body.head_zoom_degree(), Some(0.5));
    let body = producer(407, 7);
    assert_eq!(body.body_params.leg_factors, Some(LEG_FACTORS));

    let source = frame();
    let mut missing = producer(406, 7);
    missing.body_params.head_zoom = None;
    assert_eq!(
        portrait_body_stage::apply(&source, 406, 50.0, Some(7), Some(&missing)),
        Err(Refusal::MissingInput("PF406 vendor HeadZoom factors"))
    );
    let mut missing = producer(407, 7);
    missing.body_params.leg_factors = None;
    assert_eq!(
        portrait_body_stage::apply(&source, 407, 50.0, Some(7), Some(&missing)),
        Err(Refusal::MissingInput("PF407 vendor leg factors"))
    );
}

// Engine-only tests exercise the normal receipt / producer / report contract.
#[test]
fn body_normal_stage_receipt_and_writeback() {
    let source = frame();
    let masks = PortraitMasks::default();
    let region = NormRect::FULL;
    for pf in ROUTED.into_iter().chain([403]) {
        let extra = BTreeMap::from([(format!("pc:{pf}"), 50.0)]);
        let receipt = PortraitJobReceipt::from_renderer_receipt(
            &source.pixels,
            SIZE.0,
            SIZE.1,
            region,
            &extra,
            &masks,
        );
        let mut owner = ProducerInputs::default();
        if pf != 403 {
            owner
                .write_body_geometry(producer(pf, receipt.job_id()), &receipt)
                .unwrap();
        }
        let mut job = PortraitJobInputs::from_renderer_receipt(receipt, &masks);
        job.fill_from_producer_inputs(&owner);
        let mut pixels = source.pixels.clone();
        let report =
            apply_portrait_effects_with_job(&mut pixels, SIZE.0, SIZE.1, region, &extra, &job);
        assert_eq!(report.effects.len(), 1, "PF{pf}");
        let run = &report.effects[0];
        if pf == 403 {
            assert_eq!(run.refusal, Some(Refusal::NotImplemented(403)));
            assert_eq!(pixels, source.pixels);
        } else {
            assert!(
                run.executed && !run.skipped && run.refusal.is_none(),
                "PF{pf}"
            );
            if matches!(pf, 401 | 406 | 410 | 415) {
                assert!(matches!(
                    run.backend,
                    PortraitBackend::Cuda | PortraitBackend::Cpu
                ));
            } else {
                assert_eq!(run.backend, PortraitBackend::Cpu);
            }
            assert_ne!(pixels, source.pixels, "PF{pf}");
        }
    }
}

#[test]
fn body_normal_stage_rejects_stale_receipt_and_stale_job() {
    let source = frame();
    let masks = PortraitMasks::default();
    let region = NormRect::FULL;
    let extra = BTreeMap::from([("pc:407".into(), 50.0)]);
    let receipt = PortraitJobReceipt::from_renderer_receipt(
        &source.pixels,
        SIZE.0,
        SIZE.1,
        region,
        &extra,
        &masks,
    );
    let owner = ProducerInputs {
        body_geometry: Some(producer(407, 7)),
        ..Default::default()
    };
    let mut job = PortraitJobInputs::from_renderer_receipt_with_job_id(receipt, &masks, 8);
    job.fill_from_producer_inputs(&owner);
    let mut pixels = source.pixels.clone();
    let report = apply_portrait_effects_with_job(&mut pixels, SIZE.0, SIZE.1, region, &extra, &job);
    assert_eq!(
        report.effects[0].refusal,
        Some(Refusal::ValueOutOfRange("body geometry job id"))
    );
    assert_eq!(pixels, source.pixels);
    pixels[0] ^= 1;
    let changed = pixels.clone();
    let report = apply_portrait_effects_with_job(&mut pixels, SIZE.0, SIZE.1, region, &extra, &job);
    assert_eq!(
        report.effects[0].refusal,
        Some(Refusal::ValueOutOfRange("PortraitJobReceipt digest"))
    );
    assert_eq!(pixels, changed);
}

#[test]
fn body_producer_entry_accepts_synthetic_jobs_for_each_routed_pf() {
    let source = frame();
    let masks = PortraitMasks::default();
    let region = NormRect::FULL;
    let mut all: BTreeMap<u32, BodyGeometryInput> = BTreeMap::new();
    for pf in ROUTED {
        let extra = BTreeMap::from([(format!("pc:{pf}"), 50.0)]);
        let receipt = PortraitJobReceipt::from_renderer_receipt(
            &source.pixels,
            SIZE.0,
            SIZE.1,
            region,
            &extra,
            &masks,
        );
        let job_id = receipt.job_id();
        let mut owner = ProducerInputs::default();
        assert!(owner.body_geometry().is_none(), "PF{pf} unwritten at rest");
        owner
            .write_body_geometry(producer(pf, job_id), &receipt)
            .unwrap();
        let written = owner.body_geometry().expect("PF{pf} written");
        assert_eq!((written.job_id, written.frame_size), (job_id, SIZE), "PF{pf}");
        assert!(written.effects.contains_key(&pf), "PF{pf}");
        all.insert(pf, fixture(pf));

        let mut job = PortraitJobInputs::from_renderer_receipt(receipt, &masks);
        job.fill_from_producer_inputs(&owner);
        let entered = job.body_geometry.expect("PF{pf} reaches PortraitJobInputs");
        assert_eq!(entered.job_id, job_id, "PF{pf}");

        let mut pixels = source.pixels.clone();
        let report =
            apply_portrait_effects_with_job(&mut pixels, SIZE.0, SIZE.1, region, &extra, &job);
        let run = &report.effects[0];
        assert!(run.executed && !run.skipped && run.refusal.is_none(), "PF{pf}");
        if matches!(pf, 401 | 406 | 410 | 415) {
            assert!(
                matches!(run.backend, PortraitBackend::Cuda | PortraitBackend::Cpu),
                "PF{pf}"
            );
        } else {
            assert_eq!(run.backend, PortraitBackend::Cpu, "PF{pf}");
        }
        assert_ne!(pixels, source.pixels, "PF{pf}");
    }

    // One job may carry every routed row at once for the same renderer job.
    let extra: BTreeMap<String, f32> = ROUTED
        .iter()
        .map(|pf| (format!("pc:{pf}"), 50.0))
        .collect();
    let receipt = PortraitJobReceipt::from_renderer_receipt(
        &source.pixels,
        SIZE.0,
        SIZE.1,
        region,
        &extra,
        &masks,
    );
    let mut owner = ProducerInputs::default();
    owner
        .write_body_geometry(
            BodyGeometryJob {
                job_id: receipt.job_id(),
                frame_size: SIZE,
                source_sha256: sha256_32(&source.pixels),
                body_params: BodyVendorParams::for_effects(&all),
                effects: all,
            },
            &receipt,
        )
        .unwrap();
    assert_eq!(
        owner.body_geometry().map(|job| job.effects.len()),
        Some(ROUTED.len())
    );
}

#[test]
fn body_producer_entry_refuses_payloads_it_cannot_honor() {
    let source = frame();
    let masks = PortraitMasks::default();
    let region = NormRect::FULL;
    let extra = BTreeMap::from([("pc:401".to_string(), 50.0)]);
    let receipt = PortraitJobReceipt::from_renderer_receipt(
        &source.pixels,
        SIZE.0,
        SIZE.1,
        region,
        &extra,
        &masks,
    );
    // A refused write must leave the sole entry point unwritten.
    let write = |job: BodyGeometryJob| {
        let mut owner = ProducerInputs::default();
        let result = owner.write_body_geometry(job, &receipt);
        assert_eq!(owner.body_geometry().is_none(), result.is_err());
        result
    };

    assert_eq!(
        write(producer(401, 0)),
        Err(Refusal::ValueOutOfRange("body geometry job id"))
    );
    assert_eq!(
        write(producer(401, receipt.job_id() ^ 1)),
        Err(Refusal::ValueOutOfRange("body geometry job id"))
    );
    assert_eq!(
        write(BodyGeometryJob {
            job_id: receipt.job_id(),
            frame_size: (SIZE.0 + 1, SIZE.1),
            source_sha256: sha256_32(&source.pixels),
            body_params: BodyVendorParams::default(),
            effects: BTreeMap::from([(401, fixture(401))]),
        }),
        Err(Refusal::ShapeMismatch("body geometry frame"))
    );
    // Right job, wrong payload kind for the row.
    assert_eq!(
        write(BodyGeometryJob {
            job_id: receipt.job_id(),
            frame_size: SIZE,
            source_sha256: sha256_32(&source.pixels),
            body_params: BodyVendorParams::default(),
            effects: BTreeMap::from([(401, fixture(400))]),
        }),
        Err(Refusal::ShapeMismatch("body geometry effect kind"))
    );
    // PF403 stays unrouted: refuse rather than invent a mask contract.
    assert_eq!(
        write(BodyGeometryJob {
            job_id: receipt.job_id(),
            frame_size: SIZE,
            source_sha256: sha256_32(&source.pixels),
            body_params: BodyVendorParams::default(),
            effects: BTreeMap::from([(403, fixture(400))]),
        }),
        Err(Refusal::UnknownEffect(403))
    );
}
