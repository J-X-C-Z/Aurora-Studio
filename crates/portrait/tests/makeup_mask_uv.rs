use aurora_portrait::Refusal;
use aurora_portrait::kernels::makeup_mask_uv::{
    COLOR_MASK_PFS, FACE2D_MASK_ATTRIBUTE_STRIDE, Face2dMaskUvInputs, MASK_UV_FACE_MISSING,
    MASK_UV_MISSING, MASK_UV_SIZE_MISSING, MASK_UV_TICKET_MISSING, MakeupRoiRect, MeshMaskUvInputs,
    color_mask_profile, produce_face2d_mask_uv, produce_mesh_mask_uv,
};
use aurora_portrait::kernels::makeup_material::MaskChannel;

#[test]
fn six_row_profiles_are_exact_and_typed() {
    assert_eq!(COLOR_MASK_PFS, [100, 103, 105, 117, 118, 30007]);
    assert_eq!(FACE2D_MASK_ATTRIBUTE_STRIDE, 8);
    let expected = [
        (
            100,
            MaskChannel::RB,
            [100, 82, 70, 255],
            MakeupRoiRect {
                x: 210,
                y: 410,
                width: 260,
                height: 110,
            },
        ),
        (
            103,
            MaskChannel::RG,
            [39, 15, 6, 255],
            MakeupRoiRect {
                x: 200,
                y: 400,
                width: 300,
                height: 300,
            },
        ),
        (
            105,
            MaskChannel::RG,
            [100, 45, 43, 255],
            MakeupRoiRect {
                x: 200,
                y: 400,
                width: 300,
                height: 300,
            },
        ),
        (
            117,
            MaskChannel::RB,
            [100, 82, 70, 255],
            MakeupRoiRect {
                x: 540,
                y: 410,
                width: 260,
                height: 110,
            },
        ),
        (
            118,
            MaskChannel::RG,
            [142, 71, 68, 255],
            MakeupRoiRect {
                x: 200,
                y: 400,
                width: 300,
                height: 300,
            },
        ),
        (
            30007,
            MaskChannel::RG,
            [39, 15, 6, 255],
            MakeupRoiRect {
                x: 500,
                y: 400,
                width: 300,
                height: 300,
            },
        ),
    ];
    for (pf, channel, mix_color, roi) in expected {
        let profile = color_mask_profile(pf).unwrap();
        let route = aurora_portrait::cuda_makeup::MakeupRoute::resolve(pf).unwrap();
        assert_eq!(route.filter_type(), profile.filter_type, "PF{pf}");
        assert_eq!(route.mask_channel(), Some(profile.mask_channel), "PF{pf}");
        assert_eq!(profile.mask_channel, channel, "PF{pf}");
        assert_eq!(profile.mix_color, mix_color, "PF{pf}");
        assert_eq!(profile.filter_type, 11, "PF{pf}");
        assert_eq!(profile.roi, roi, "PF{pf}");
    }
    assert_eq!(
        color_mask_profile(111).unwrap_err(),
        Refusal::UnknownEffect(111)
    );
}

#[test]
fn arbitrary_same_job_mesh_rasterizes_to_uv_and_writes() {
    // SYNTHETIC_CONTRACT: three vertices prove this producer is not tied to
    // PF111's 49-vertex Normal Eye mesh.
    let positions = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]];
    let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let out = produce_mesh_mask_uv(Some(MeshMaskUvInputs {
        ticket: Some(7),
        face_id: Some(0),
        positions: &positions,
        mask_uv: &uv,
        indices: &[0, 1, 2],
        width: 4,
        height: 4,
    }))
    .unwrap();

    assert_eq!(out.ticket(), Some(7));
    assert_eq!(out.face_id(), Some(0));
    assert!(out.covered(0));
    assert_eq!(out.uv()[0], [0.125, 0.125]);
    assert!(!out.covered(15));
    assert_eq!(out.uv()[15], [0.0, 0.0]);
}

#[test]
fn absent_mesh_stream_is_the_same_job_mask_uv_refusal() {
    assert_eq!(
        produce_mesh_mask_uv(None).unwrap_err(),
        Refusal::MissingInput(MASK_UV_MISSING)
    );
}

#[test]
fn supplied_bad_mesh_keeps_shape_and_range_refusals() {
    let positions = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]];
    let uv = [[0.0, 0.0]; 2];
    assert_eq!(
        produce_mesh_mask_uv(Some(MeshMaskUvInputs {
            ticket: Some(1),
            face_id: Some(1),
            positions: &positions,
            mask_uv: &uv,
            indices: &[0, 1, 2],
            width: 4,
            height: 4,
        }))
        .unwrap_err(),
        Refusal::ShapeMismatch(MASK_UV_MISSING)
    );

    let uv = [[0.0, 0.0]; 3];
    assert_eq!(
        produce_mesh_mask_uv(Some(MeshMaskUvInputs {
            ticket: Some(1),
            face_id: Some(1),
            positions: &positions,
            mask_uv: &uv,
            indices: &[0, 1],
            width: 4,
            height: 4,
        }))
        .unwrap_err(),
        Refusal::ShapeMismatch(MASK_UV_MISSING)
    );
}

#[test]
fn face2d_source_uses_recovered_origin_size_and_identity() {
    let source = [[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]];
    let out = produce_face2d_mask_uv(Some(Face2dMaskUvInputs {
        ticket: Some(42),
        face_id: Some(3),
        mask_source_points: &source,
        indices: &[0, 1, 2],
        origin: [0, 0],
        size: [8, 8],
        width: 8,
        height: 8,
    }))
    .unwrap();

    assert_eq!(out.ticket(), Some(42));
    assert_eq!(out.face_id(), Some(3));
    assert!(out.covered(0));
    assert_eq!(out.uv()[0], [0.0625, 0.0625]);
    assert!(out.uv().iter().all(|uv| uv[0] >= 0.0 && uv[1] >= 0.0));
}

#[test]
fn face2d_source_requires_ticket_face_and_nonzero_extent() {
    let source = [[0.0, 0.0], [8.0, 0.0], [0.0, 8.0]];
    let base = Face2dMaskUvInputs {
        ticket: Some(42),
        face_id: Some(3),
        mask_source_points: &source,
        indices: &[0, 1, 2],
        origin: [0, 0],
        size: [8, 8],
        width: 8,
        height: 8,
    };

    let mut missing_ticket = base;
    missing_ticket.ticket = None;
    assert_eq!(
        produce_face2d_mask_uv(Some(missing_ticket)).unwrap_err(),
        Refusal::MissingInput(MASK_UV_TICKET_MISSING)
    );

    let mut zero_ticket = base;
    zero_ticket.ticket = Some(0);
    assert_eq!(
        produce_face2d_mask_uv(Some(zero_ticket)).unwrap_err(),
        Refusal::MissingInput(MASK_UV_TICKET_MISSING)
    );

    let mut missing_face = base;
    missing_face.face_id = None;
    assert_eq!(
        produce_face2d_mask_uv(Some(missing_face)).unwrap_err(),
        Refusal::MissingInput(MASK_UV_FACE_MISSING)
    );

    let mut zero_size = base;
    zero_size.size[0] = 0;
    assert_eq!(
        produce_face2d_mask_uv(Some(zero_size)).unwrap_err(),
        Refusal::ValueOutOfRange(MASK_UV_SIZE_MISSING)
    );
}
