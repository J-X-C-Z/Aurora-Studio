use super::*;

static VEC3: [[f32; 3]; 1] = [[0.0, 0.0, 0.0]];
static VBO: [u8; 2304] = [0; 2304];
static READBACK: [u8; 4] = [0; 4];

struct Fixture {
    points194: Vec<[f32; 2]>,
    points526: Vec<[f32; 2]>,
    mesh: Vec<[f32; 2]>,
    indices: Vec<u16>,
    weights: cuda_pf8050::Weights,
    object: cuda_pf8050::ObjectFields,
    face: cuda_pf8050::FaceFields,
    degrees: [f32; 12],
}

impl Fixture {
    fn new() -> Self {
        Self {
            points194: (0..194)
                .map(|index| [index as f32, (index % 17) as f32])
                .collect(),
            points526: (0..526)
                .map(|index| [index as f32 * 0.5, (index % 23) as f32])
                .collect(),
            mesh: (0..MESH_POINTS)
                .map(|index| [index as f32 * 0.5 + 1.0, (index % 31) as f32 * 0.25])
                .collect(),
            indices: (0..cuda_pf8050::INDEX_COUNT)
                .map(|index| (index % MESH_POINTS) as u16)
                .collect(),
            weights: cuda_pf8050::Weights {
                stage0_raw: std::array::from_fn(|index| 0.25 + index as f32 * 0.01),
                stage2: std::array::from_fn(|index| 0.5 - index as f32 * 0.01),
            },
            object: cuda_pf8050::ObjectFields {
                center_left: [55.0, 55.0],
                center_right: [125.0, 55.0],
                mid_left: [52.0, 55.0],
                mid_right: [122.0, 55.0],
                axis1_left: [3.0, 0.0],
                axis1_right: [3.0, 0.0],
                axis2_left: [-2.0, 2.0],
                axis2_right: [-2.0, 2.0],
            },
            face: cuda_pf8050::FaceFields {
                f0: [70.0, 50.0],
                f1: [90.0, 51.0],
                direction_point: [86.0, 34.0],
            },
            degrees: [
                0.42, -0.31, 0.19, -0.17, 0.23, -0.13, 0.11, -0.09, 0.15, -0.07,
                0.21, -0.12,
            ],
        }
    }

    fn geometry(&self) -> Pf8050GeometryInputs<'_> {
        cuda_pf8050::Inputs {
            producer: Some(cuda_pf8050::Producer {
                job_id: 940,
                face_id: 0,
                points_194: &self.points194,
                points_526: &self.points526,
            }),
            mesh_a: Some(&self.mesh),
            indices: Some(&self.indices),
            weights: Some(&self.weights),
            object: Some(&self.object),
            face: Some(&self.face),
            degrees: Some(&self.degrees),
        }
    }
}

fn full_runtime() -> Pf8050VendorRuntime<'static> {
    Pf8050VendorRuntime {
        attach: Some(Pf8050AttachReceipt {
            job_id: 940,
            face_id: 0,
        }),
        vec3: Some(&VEC3),
        roi: Some(NormRect::FULL),
        vbo: Some(&VBO),
        fbo: Some(Pf8050FboReceipt {
            framebuffer_id: 1,
            color_attachment: 0,
            width: 1,
            height: 1,
        }),
        readback: Some(&READBACK),
    }
}

#[test]
fn geometry_entry_returns_six_candidate_stages() {
    let fixture = Fixture::new();
    let inputs = fixture.geometry();
    let expected = cuda_pf8050::apply_cpu(cuda_pf8050::resolve(inputs).expect("valid geometry"));
    let run = apply_pf8050_pf8051_geometry(inputs);

    assert!(run.candidate_only);
    assert_eq!(run.refusal, None);
    assert_eq!(run.stages, expected.stages);
    assert_eq!(run.stages.len(), cuda_pf8050::STAGE_COUNT);
}

#[test]
fn vendor_stage_keeps_each_missing_binding_typed() {
    let fixture = Fixture::new();
    let attach = full_runtime().attach;
    let fbo = full_runtime().fbo;
    let cases = [
        (Pf8050VendorRuntime::default(), "PF8050_ATTACH"),
        (
            Pf8050VendorRuntime {
                attach,
                ..Default::default()
            },
            "PF8050_VEC3",
        ),
        (
            Pf8050VendorRuntime {
                attach,
                vec3: Some(&VEC3),
                ..Default::default()
            },
            "PF8050_ROI",
        ),
        (
            Pf8050VendorRuntime {
                attach,
                vec3: Some(&VEC3),
                roi: Some(NormRect::FULL),
                ..Default::default()
            },
            "PF8050_VBO",
        ),
        (
            Pf8050VendorRuntime {
                attach,
                vec3: Some(&VEC3),
                roi: Some(NormRect::FULL),
                vbo: Some(&VBO),
                ..Default::default()
            },
            "PF8050_FBO",
        ),
        (
            Pf8050VendorRuntime {
                attach,
                vec3: Some(&VEC3),
                roi: Some(NormRect::FULL),
                vbo: Some(&VBO),
                fbo,
                ..Default::default()
            },
            "PF8050_READBACK",
        ),
    ];

    for (runtime, missing) in cases {
        let run = apply_pf8050_pf8051_stage(Pf8050StageInputs {
            geometry: fixture.geometry(),
            runtime,
        });
        assert_eq!(run.refusal, Some(Refusal::MissingInput(missing)));
        assert!(!run.candidate_only);
        assert!(run.stages.is_empty());
    }
}

#[test]
fn complete_vendor_receipts_do_not_promote_candidate_geometry() {
    let fixture = Fixture::new();
    let run = apply_pf8050_pf8051_stage(Pf8050StageInputs {
        geometry: fixture.geometry(),
        runtime: full_runtime(),
    });

    assert!(run.candidate_only);
    assert_eq!(run.refusal, Some(Refusal::NotImplemented(8050)));
    assert_eq!(run.stages.len(), cuda_pf8050::STAGE_COUNT);
}
