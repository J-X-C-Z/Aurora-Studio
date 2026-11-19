use aurora_portrait::Refusal;
use aurora_portrait::kernels::pf111_uv::{
    CONTOURS_MISSING, EYE_PUPIL_POSITION_MISSING, EyePupilModel8, EyePupilPosition49, EyeSide,
    MATERIAL_EXTENT_MISSING, MODEL_MISSING, MaterialExtent, NORMAL_EYE_EBO_MISSING,
    NORMAL_EYE_UV_MISSING, NormalEyeEbo, NormalEyeUv49, PF111_VERTEX_COUNT, PackedContour49,
    Pf111RasterInputs, Pf111UvInputs, RASTER_EXTENT_MISSING, RasterExtent, produce,
    rasterize_pixels,
};

fn model() -> EyePupilModel8 {
    EyePupilModel8::new(std::array::from_fn(|index| {
        [index as f32, (index * index) as f32]
    }))
    .unwrap()
}

fn contours(vertices: &[[f32; 2]]) -> PackedContour49<'_> {
    PackedContour49::new(vertices).unwrap()
}

#[test]
fn missing_same_job_inputs_stay_missing() {
    let absent = Pf111UvInputs {
        side: EyeSide::Left,
        model: None,
        contours: None,
        material_extent: None,
        normal_eye_ebo: None,
    };
    assert_eq!(
        produce(absent).unwrap_err(),
        Refusal::MissingInput(MODEL_MISSING)
    );

    let vertices = [[0.0, 0.0]; PF111_VERTEX_COUNT];
    let no_contours = Pf111UvInputs {
        model: Some(model()),
        ..absent
    };
    assert_eq!(
        produce(no_contours).unwrap_err(),
        Refusal::MissingInput(CONTOURS_MISSING)
    );

    let no_extent = Pf111UvInputs {
        contours: Some(contours(&vertices)),
        ..no_contours
    };
    assert_eq!(
        produce(no_extent).unwrap_err(),
        Refusal::MissingInput(MATERIAL_EXTENT_MISSING)
    );

    let no_ebo = Pf111UvInputs {
        material_extent: Some(MaterialExtent::new(150, 150).unwrap()),
        ..no_extent
    };
    assert_eq!(
        produce(no_ebo).unwrap_err(),
        Refusal::MissingInput(NORMAL_EYE_EBO_MISSING)
    );
}

#[test]
fn material_uv_is_point_over_material_extent_without_flip_or_clamp() {
    let vertices: Vec<[f32; 2]> = (0..PF111_VERTEX_COUNT)
        .map(|index| [index as f32 * 2.0, index as f32 * 3.0])
        .collect();
    let indices = [0, 1, 2, 47, 48, 0];
    let mesh = produce(Pf111UvInputs {
        side: EyeSide::Right,
        model: Some(model()),
        contours: Some(contours(&vertices)),
        material_extent: Some(MaterialExtent::new(100, 150).unwrap()),
        normal_eye_ebo: Some(NormalEyeEbo::new(&indices).unwrap()),
    })
    .unwrap();

    assert_eq!(mesh.side(), EyeSide::Right);
    assert_eq!(mesh.material_uv()[0], [0.0, 0.0]);
    assert_eq!(mesh.material_uv()[48], [0.96, 0.96]);
    assert_eq!(mesh.normal_eye_ebo().indices(), &indices);
}

#[test]
fn typed_inputs_reject_wrong_shapes_and_non_finite_points() {
    let short = [[0.0, 0.0]; PF111_VERTEX_COUNT - 1];
    assert_eq!(
        PackedContour49::new(&short).unwrap_err(),
        Refusal::ShapeMismatch(CONTOURS_MISSING)
    );

    let nan = [[f32::NAN, 0.0]; 8];
    assert_eq!(
        EyePupilModel8::new(nan).unwrap_err(),
        Refusal::ValueOutOfRange(MODEL_MISSING)
    );

    assert_eq!(
        MaterialExtent::new(0, 150).unwrap_err(),
        Refusal::ValueOutOfRange(MATERIAL_EXTENT_MISSING)
    );
    assert_eq!(
        NormalEyeEbo::new(&[0, 1, PF111_VERTEX_COUNT as u16]).unwrap_err(),
        Refusal::ValueOutOfRange(NORMAL_EYE_EBO_MISSING)
    );

    let short_positions = [[0.0, 0.0]; PF111_VERTEX_COUNT - 1];
    assert_eq!(
        EyePupilPosition49::new(&short_positions).unwrap_err(),
        Refusal::ShapeMismatch(EYE_PUPIL_POSITION_MISSING)
    );
    let non_finite_uv = [[f32::INFINITY, 0.0]; PF111_VERTEX_COUNT];
    assert_eq!(
        NormalEyeUv49::new(&non_finite_uv).unwrap_err(),
        Refusal::ValueOutOfRange(NORMAL_EYE_UV_MISSING)
    );
}

fn raster_input<'a>(
    positions: &'a [[f32; 2]],
    uv: &'a [[f32; 2]],
    ebo: &'a [u16],
) -> Pf111RasterInputs<'a> {
    Pf111RasterInputs {
        side: EyeSide::Left,
        positions: Some(EyePupilPosition49::new(positions).unwrap()),
        normal_eye_uv: Some(NormalEyeUv49::new(uv).unwrap()),
        extent: Some(RasterExtent::new(4, 4).unwrap()),
        normal_eye_ebo: Some(NormalEyeEbo::new(ebo).unwrap()),
    }
}

#[test]
fn raster_refuses_missing_same_job_streams() {
    let positions = [[0.0; 2]; PF111_VERTEX_COUNT];
    let uv = [[0.0; 2]; PF111_VERTEX_COUNT];
    let ebo = [0, 1, 2];
    let absent = Pf111RasterInputs {
        side: EyeSide::Left,
        positions: None,
        normal_eye_uv: None,
        extent: None,
        normal_eye_ebo: None,
    };
    assert_eq!(
        rasterize_pixels(absent).unwrap_err(),
        Refusal::MissingInput(EYE_PUPIL_POSITION_MISSING)
    );

    let base = raster_input(&positions, &uv, &ebo);
    assert_eq!(
        rasterize_pixels(Pf111RasterInputs {
            normal_eye_uv: None,
            ..base
        })
        .unwrap_err(),
        Refusal::MissingInput(NORMAL_EYE_UV_MISSING)
    );
    assert_eq!(
        rasterize_pixels(Pf111RasterInputs {
            extent: None,
            ..base
        })
        .unwrap_err(),
        Refusal::MissingInput(RASTER_EXTENT_MISSING)
    );
    assert_eq!(
        rasterize_pixels(Pf111RasterInputs {
            normal_eye_ebo: None,
            ..base
        })
        .unwrap_err(),
        Refusal::MissingInput(NORMAL_EYE_EBO_MISSING)
    );
}

#[test]
fn synthetic_triangle_checks_boundary_interpolation_and_top_left_y() {
    // SYNTHETIC_CONTRACT: this is not a vendor mesh or readback.
    let mut positions = vec![[0.0, 0.0]; PF111_VERTEX_COUNT];
    let mut uv = vec![[0.0, 0.0]; PF111_VERTEX_COUNT];
    positions[0] = [0.0, 0.0];
    positions[1] = [4.0, 0.0];
    positions[2] = [0.0, 4.0];
    uv[0] = [0.0, 0.0];
    uv[1] = [1.0, 0.0];
    uv[2] = [0.0, 1.0];
    let out = rasterize_pixels(raster_input(&positions, &uv, &[0, 1, 2])).unwrap();

    let close = |actual: [f32; 2], expected: [f32; 2]| {
        (actual[0] - expected[0]).abs() < 1e-6 && (actual[1] - expected[1]).abs() < 1e-6
    };
    assert!(out.covered(0));
    assert!(close(out.material_uv()[0], [0.125, 0.125]));

    // x + y == 4 is the diagonal boundary; the CPU contract includes it.
    assert!(out.covered(3));
    assert!(close(out.material_uv()[3], [0.875, 0.125]));
    assert!(out.covered(12));
    assert!(close(out.material_uv()[12], [0.125, 0.875]));

    // Outside the triangle stays unwritten and cannot be treated as a UV.
    assert!(!out.covered(15));
    assert_eq!(out.material_uv()[15], [0.0, 0.0]);
    assert!(out.mask_uv().is_none(), "PF111 has no mask UV stream");
}

#[test]
fn synthetic_non_square_interpolation_keeps_v_increasing_downward() {
    // SYNTHETIC_CONTRACT: UV values are chosen only to expose the Y contract.
    let mut positions = vec![[0.0, 0.0]; PF111_VERTEX_COUNT];
    let mut uv = vec![[0.0, 0.0]; PF111_VERTEX_COUNT];
    positions[0] = [0.0, 0.0];
    positions[1] = [5.0, 0.0];
    positions[2] = [0.0, 3.0];
    uv[0] = [0.1, 0.2];
    uv[1] = [0.9, 0.2];
    uv[2] = [0.1, 0.8];
    let inputs = Pf111RasterInputs {
        extent: Some(RasterExtent::new(5, 3).unwrap()),
        ..raster_input(&positions, &uv, &[0, 1, 2])
    };
    let out = rasterize_pixels(inputs).unwrap();

    let top = out.material_uv()[0];
    let bottom = out.material_uv()[2 * 5];
    assert!((top[0] - 0.18).abs() < 1e-6);
    assert!((top[1] - 0.3).abs() < 1e-6);
    assert!((bottom[0] - 0.18).abs() < 1e-6);
    assert!((bottom[1] - 0.7).abs() < 1e-6);
    assert!(top[1] < bottom[1], "top-left image Y must not be flipped");
}

#[test]
fn ebo_must_remain_a_u16_triangle_list() {
    let positions = [[0.0; 2]; PF111_VERTEX_COUNT];
    let uv = [[0.0; 2]; PF111_VERTEX_COUNT];
    let inputs = raster_input(&positions, &uv, &[0, 1]);
    assert_eq!(
        rasterize_pixels(inputs).unwrap_err(),
        Refusal::ShapeMismatch(NORMAL_EYE_EBO_MISSING)
    );
}
