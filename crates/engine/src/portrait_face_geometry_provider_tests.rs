use aurora_portrait::{raster, raster_fragment, Refusal};

use super::*;

fn points(count: usize, seed: f32) -> Vec<[f32; 2]> {
    (0..count)
        .map(|index| [seed + index as f32 * 0.25, seed + (index % 17) as f32 * 0.5])
        .collect()
}

fn mesh() -> FaceGeometryProviderMesh {
    let positions = vec![[1.0, 1.0], [7.0, 1.0], [1.0, 7.0], [7.0, 7.0]];
    let uv = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let ebo = vec![0, 1, 2, 1, 2, 3];
    FaceGeometryProviderMesh {
        mesh_id: 7,
        job_id: 42,
        vbo: pack_mesh_vbo(&positions, &uv),
        positions,
        ebo,
        uv,
    }
}

fn base_input() -> FaceGeometryProviderInputs {
    let points142 = points(142, 10.0);
    let points194 = points(194, 20.0);
    let meshes = vec![mesh()];
    let mesh_receipts = meshes.iter().map(mesh_receipt).collect();
    let source_sha256 = sha256_32(&[1, 2, 3, 4]);
    let (points142_sha256, points194_sha256) = landmark_receipt_hashes(&points142, &points194);
    FaceGeometryProviderInputs {
        job_id: 42,
        width: 4,
        height: 4,
        source_sha256,
        points142,
        points194,
        meshes,
        receipt: FaceGeometryProviderReceipt {
            job_id: 42,
            width: 4,
            height: 4,
            source_sha256,
            points142_sha256,
            points194_sha256,
            meshes: mesh_receipts,
        },
        rows: Vec::new(),
    }
}

fn eye() -> FaceGeometry {
    FaceGeometry::Eye {
        dest: mesh().positions,
        uv: mesh().uv,
        ebo: mesh().ebo,
        rules: raster::Rules {
            winding: raster::Winding::PositiveAreaIsFront,
            cull: raster::Cull::None,
            edge: raster::Edge::Inclusive,
            overlap: raster::Overlap::First,
        },
        edge: raster_fragment::Edge::ReshapeNormal,
        defines: raster_fragment::Defines::plain(),
        clamp: raster_fragment::Clamp::disabled(),
    }
}

fn ready_row(pf: u32) -> FaceGeometryProviderRow {
    let geometry = match provider_route(pf).unwrap() {
        FaceGeometryProviderRoute::V2Positions => FaceGeometry::V2Positions(vec![(0, [2.0, 1.0])]),
        FaceGeometryProviderRoute::VFace => FaceGeometry::VFace {
            rows: vec![aurora_portrait::cuda_vface::Row::RightPositive],
            encoded_offset: Vec::new(),
            atlases: vec![super::super::portrait_v2_mesh::VFaceAtlas::captured(
                aurora_portrait::cuda_vface::Row::RightPositive,
            )],
        },
        FaceGeometryProviderRoute::Type1011 => FaceGeometry::Type1011 {
            prepared: (0..100)
                .map(|index| {
                    [
                        0.25 + (index % 10) as f32 * 0.35,
                        0.25 + (index / 10) as f32 * 0.35,
                    ]
                })
                .collect(),
            param_list: vec![0.1; 6],
            center: [2.0, 2.0],
        },
        FaceGeometryProviderRoute::Eye => eye(),
        FaceGeometryProviderRoute::Smile => FaceGeometry::Smile {
            dx: vec![1.0; 16],
            dy: vec![0.0; 16],
            mask: vec![255; 16],
            host_split_receipt: Some(SmileHostSplitReceipt {
                job_id: 42,
                model_output_id: super::super::SmileRgbReconstructionField::OUTPUT_ID,
                model_output_sha256: super::sha256_32(b"780929437 fixture"),
                dx_sha256: hash_f32(&[1.0; 16]),
                dy_sha256: hash_f32(&[0.0; 16]),
                mask_sha256: sha256_32(&[255; 16]),
            }),
        },
        route => panic!("not a ready route: {route:?}"),
    };
    FaceGeometryProviderRow {
        pf,
        controls: [0.5, 0.5],
        mesh_id: Some(7),
        geometry,
    }
}

#[test]
fn route_table_covers_requested_rows_with_named_refusals() {
    for &pf in PROVIDER_ROUTED_PFS {
        assert!(provider_route(pf).is_some(), "PF{pf}");
    }
    assert!(matches!(
        provider_route(8003),
        Some(FaceGeometryProviderRoute::Refused(_))
    ));
    assert!(matches!(
        provider_route(8031),
        Some(FaceGeometryProviderRoute::Refused(_))
    ));
    assert!(matches!(
        provider_route(30084),
        Some(FaceGeometryProviderRoute::Refused(_))
    ));
    // Type1012 forehead rows keep their own handoff semantics, not an eye draw.
    assert!(matches!(
        provider_route(30080),
        Some(FaceGeometryProviderRoute::Refused(_))
    ));
    assert!(matches!(
        provider_route(30083),
        Some(FaceGeometryProviderRoute::Refused(_))
    ));
    assert_eq!(
        provider_route(8025),
        Some(FaceGeometryProviderRoute::SlimContract)
    );
    assert_eq!(
        provider_route(30073),
        Some(FaceGeometryProviderRoute::Type1011)
    );
    assert_eq!(
        provider_route(30092),
        Some(FaceGeometryProviderRoute::FaceMeshContract)
    );
}

#[test]
fn builder_binds_ready_rows_to_the_source_digest() {
    for pf in [
        242, 243, 247, 248, 8001, 8022, 30073, 30079, 44124, 48019, 49221, 41922, 8033,
    ] {
        let mut input = base_input();
        input.rows = vec![ready_row(pf)];
        let output = build_face_geometry_inputs(&input).unwrap();
        assert_eq!(output.job_id, 42);
        assert_eq!((output.width, output.height), (4, 4));
        assert_eq!(output.draws.len(), 1);
        assert_eq!(output.draws[0].pf, pf);
        assert_eq!(output.draws[0].controls, [0.5, 0.5]);
        assert_eq!(output.draws[0].source_sha256, input.source_sha256);
    }
}

#[test]
fn stale_receipts_and_cross_job_meshes_refuse_by_name() {
    let mut input = base_input();
    input.rows = vec![ready_row(242)];

    input.receipt.source_sha256[0] ^= 1;
    assert_eq!(
        build_face_geometry_inputs(&input),
        Err(Refusal::ValueOutOfRange(
            "face geometry provider receipt source digest"
        ))
    );
    input.receipt.source_sha256[0] ^= 1;

    input.receipt.points194_sha256[0] ^= 1;
    assert_eq!(
        build_face_geometry_inputs(&input),
        Err(Refusal::ValueOutOfRange(
            "face geometry provider receipt 194 points"
        ))
    );
    input.receipt.points194_sha256[0] ^= 1;

    input.meshes[0].job_id = 41;
    assert_eq!(
        build_face_geometry_inputs(&input),
        Err(Refusal::ValueOutOfRange("face geometry provider mesh job"))
    );
    input.meshes[0].job_id = 42;

    input.rows[0].mesh_id = Some(99);
    assert_eq!(
        build_face_geometry_inputs(&input),
        Err(Refusal::MissingInput("face geometry provider V2 mesh"))
    );
}

#[test]
fn missing_points_or_vbo_bytes_are_named_refusals() {
    let mut input = base_input();
    input.rows = vec![ready_row(8001)];
    input.points142.pop();
    assert_eq!(
        build_face_geometry_inputs(&input),
        Err(Refusal::MissingInput(
            "face geometry provider same-job 142 points"
        ))
    );

    let mut input = base_input();
    input.rows = vec![ready_row(8001)];
    input.meshes[0].vbo[0] ^= 1;
    assert_eq!(
        build_face_geometry_inputs(&input),
        Err(Refusal::ShapeMismatch(
            "face geometry provider mesh VBO bytes"
        ))
    );
}

#[test]
fn type1011_rows_require_exactly_prepared100() {
    let mut input = base_input();
    input.rows = vec![ready_row(30073)];
    if let FaceGeometry::Type1011 { prepared, .. } = &mut input.rows[0].geometry {
        prepared.pop();
    }
    assert_eq!(
        build_face_geometry_inputs(&input),
        Err(Refusal::ShapeMismatch(
            "face geometry provider Type1011 prepared100"
        ))
    );
}

#[test]
fn open_and_contract_split_rows_do_not_emit_face_geometry() {
    for (pf, expected) in [
        (
            8003,
            Refusal::MissingInput("PF8003/PF8004/PF8007/PF8008 FacialRefine mesh producer is open"),
        ),
        (
            30092,
            Refusal::MissingInput("PF30092/PF30093 use the face-mesh group contract"),
        ),
        (
            8025,
            Refusal::MissingInput("PF8025/PF8026 use SlimProducerInputs"),
        ),
        (
            30080,
            Refusal::MissingInput(
                "PF30080..PF30083 Type1012 forehead texture/V2 handoff is not an eye-quad draw",
            ),
        ),
    ] {
        let mut input = base_input();
        input.rows = vec![FaceGeometryProviderRow {
            pf,
            controls: [0.5, 0.5],
            mesh_id: Some(7),
            geometry: eye(),
        }];
        assert_eq!(build_face_geometry_inputs(&input), Err(expected));
    }
}

#[test]
fn pf8033_effect_contract_pins_warp_type_flag_and_ranges() {
    let effect: serde_json::Value =
        serde_json::from_str(include_str!("../../../assets/portrait/smile/effect.json")).unwrap();
    let effects = effect["Effects"].as_array().unwrap();
    assert_eq!(effects.len(), 1);
    let effect = &effects[0];
    assert_eq!(effect["EffectName"], "DL3D_Warp");
    assert_eq!(effect["WarpType"], 1);
    assert_eq!(effect["ParamFlag"], 8033);
    assert_eq!(effect["DegreeRange"], serde_json::json!([0, 100]));
    assert_eq!(
        effect["AngleRange"],
        serde_json::json!([28.420000076293945, -28.420000076293945])
    );
    assert_eq!(
        effect["DistRatioRange"],
        serde_json::json!([5.050000190734863, 0.0009999939939007163])
    );
}
