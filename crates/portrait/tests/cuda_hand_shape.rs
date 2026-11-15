// Real NVIDIA coverage for mode 1 of `uvmap_color_kernel`.

#![cfg(windows)]

use aurora_portrait::cuda::CudaError;
use aurora_portrait::cuda_uvmap::{DirectPolicy, Uvmap, KERNEL};
use aurora_portrait::kernels::hair_body;
use aurora_portrait::{Image, Refusal};
use serde_json::Value;

const PF30190_FIXTURE: &str = include_str!("fixtures/pf30190_uvmap.json");
const DIRECT_FIXTURE: &str = include_str!("fixtures/direct_policies.json");

fn gpu() -> Option<Uvmap> {
    match Uvmap::open() {
        Ok(gpu) => Some(gpu),
        Err(CudaError::Unavailable(reason)) => {
            eprintln!("SKIP_NO_NVIDIA_DRIVER: {reason}");
            assert!(
                std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                "CUDA required but unavailable: {reason}"
            );
            None
        }
        Err(broken) => panic!("NVIDIA path present but broken: {broken:?}"),
    }
}

fn bytes(value: &Value) -> Vec<u8> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_u64().unwrap() as u8)
        .collect()
}

fn floats(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap() as f32)
        .collect()
}

fn field(value: &Value) -> Vec<[f32; 4]> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|pixel| {
            let pixel = pixel.as_array().unwrap();
            [
                pixel[0].as_f64().unwrap() as f32,
                pixel[1].as_f64().unwrap() as f32,
                pixel[2].as_f64().unwrap() as f32,
                pixel[3].as_f64().unwrap() as f32,
            ]
        })
        .collect()
}

fn image(value: &Value) -> Image {
    Image::new(
        value["width"].as_u64().unwrap() as u32,
        value["height"].as_u64().unwrap() as u32,
        bytes(&value["source_rgba8"]),
    )
    .unwrap()
}

fn assert_dispatch(dispatches: &[aurora_portrait::cuda::Dispatch]) {
    assert_eq!(dispatches.len(), 1);
    assert_eq!(dispatches[0].kernel, KERNEL);
    assert!(dispatches[0].ptx_bytes > 0);
    eprintln!(
        "GPU {} cc={}.{} driver={} ptx={:?} grid={:?}",
        dispatches[0].device,
        dispatches[0].compute_cap.0,
        dispatches[0].compute_cap.1,
        dispatches[0].driver_version,
        dispatches[0].ptx_isa,
        dispatches[0].grid
    );
}

#[test]
fn pf30190_fixture_matches_cpu_and_real_gpu() {
    let Some(gpu) = gpu() else { return };
    let fixture: Value = serde_json::from_str(PF30190_FIXTURE).unwrap();
    let source = image(&fixture["image"]);
    let shift_x = floats(&fixture["job"]["shift_x_f32"]);
    let shift_y = floats(&fixture["job"]["shift_y_f32"]);
    let hand_mask = bytes(&fixture["job"]["hand_mask"]);
    let protect_mask = bytes(&fixture["job"]["protect_mask"]);
    let expected = bytes(&fixture["expected"]["cpu_hand_shape_rgba8"]);

    let cpu = hair_body::hand_shape(
        &source,
        fixture["job"]["native_parameter"].as_i64().unwrap() as i32,
        &shift_x,
        &shift_y,
        &hand_mask,
        &protect_mask,
        Some(fixture["job"]["source_scale"].as_f64().unwrap() as f32),
    )
    .unwrap();
    assert_eq!(cpu.pixels, expected);

    let (actual, dispatches) = gpu
        .apply_hand_shape(
            &source,
            fixture["job"]["native_parameter"].as_i64().unwrap() as i32,
            Some(&shift_x),
            Some(&shift_y),
            Some(&hand_mask),
            Some(&protect_mask),
            Some(fixture["job"]["source_scale"].as_f64().unwrap() as f32),
        )
        .unwrap();
    assert_eq!(actual.pixels, expected, "PF30190 GPU fixture parity");
    assert_dispatch(&dispatches);
}

#[test]
fn direct_policies_match_quantizer_and_alpha_fixture() {
    let Some(gpu) = gpu() else { return };
    let fixture: Value = serde_json::from_str(DIRECT_FIXTURE).unwrap();
    let source = image(&fixture["image"]);
    let body_field = field(&fixture["field_body_xyzw"]);
    let pf30190_field = field(&fixture["field_pf30190_xyzw"]);
    let cases = [
        (
            DirectPolicy::Pf30190,
            pf30190_field.as_slice(),
            "pf30190_rgba8",
        ),
        (DirectPolicy::Pf400, body_field.as_slice(), "pf400_rgba8"),
        (
            DirectPolicy::Pf404Or416,
            body_field.as_slice(),
            "pf404_416_rgba8",
        ),
    ];
    for (policy, direct_field, name) in cases {
        let expected = bytes(&fixture["expected"][name]);
        let (actual, dispatches) = gpu.apply_direct(&source, direct_field, policy).unwrap();
        assert_eq!(actual.pixels, expected, "{name}");
        assert_dispatch(&dispatches);
    }

    let tie = &fixture["tie_case"];
    let tie_source = image(tie);
    let tie_field = field(&tie["field_xyzw"]);
    for (policy, name) in [
        (DirectPolicy::Pf30190, "pf30190_rgba8"),
        (DirectPolicy::Pf400, "pf400_rgba8"),
        (DirectPolicy::Pf404Or416, "pf404_416_rgba8"),
    ] {
        let expected = bytes(&tie["expected"][name]);
        let (actual, _) = gpu.apply_direct(&tie_source, &tie_field, policy).unwrap();
        assert_eq!(actual.pixels, expected, "tie {name}");
    }

    for case in fixture["size_matrix"].as_array().unwrap() {
        let case_source = image(case);
        let case_field = field(&case["field_xyzw"]);
        for (policy, name) in [
            (DirectPolicy::Pf30190, "pf30190_rgba8"),
            (DirectPolicy::Pf400, "pf400_rgba8"),
            (DirectPolicy::Pf404Or416, "pf404_416_rgba8"),
        ] {
            let expected = bytes(&case["expected"][name]);
            let (actual, _) = gpu
                .apply_direct(&case_source, &case_field, policy)
                .unwrap();
            assert_eq!(
                actual.pixels, expected,
                "{}x{} {name}",
                case_source.width, case_source.height
            );
        }
    }
}

#[test]
fn missing_same_job_inputs_and_bad_fields_are_typed_refusals() {
    let Some(gpu) = gpu() else { return };
    let source = Image::solid(2, 1, [10, 20, 30, 40]);
    assert_eq!(
        gpu.apply_hand_shape(&source, 50, None, None, None, None, None)
            .unwrap_err(),
        CudaError::Refused(Refusal::MissingInput("PF30190_SHIFT"))
    );
    assert_eq!(
        gpu.apply_hand_shape(
            &source,
            50,
            Some(&[1.0, 0.0]),
            Some(&[0.0, 0.0]),
            None,
            None,
            Some(1.0),
        )
        .unwrap_err(),
        CudaError::Refused(Refusal::MissingInput("PF30190_HAND_MASK"))
    );
    assert_eq!(
        gpu.apply_hand_shape(
            &source,
            50,
            Some(&[1.0, 0.0]),
            Some(&[0.0, 0.0]),
            Some(&[255, 255]),
            None,
            Some(1.0),
        )
        .unwrap_err(),
        CudaError::Refused(Refusal::MissingInput("PF30190_PROTECT_MASK"))
    );

    let (identity, dispatches) = gpu
        .apply_hand_shape(&source, 0, None, None, None, None, None)
        .unwrap();
    assert_eq!(identity, source);
    assert!(dispatches.is_empty());

    let mut bad = vec![[0.0f32, 0.0, 0.0, 1.0]; 2];
    bad[0][0] = f32::NAN;
    assert_eq!(
        gpu.apply_direct(&source, &bad, DirectPolicy::Pf400)
            .unwrap_err(),
        CudaError::Refused(Refusal::ValueOutOfRange("direct uv field"))
    );
    bad[0][0] = 0.0;
    bad[0][3] = 0.5;
    assert_eq!(
        gpu.apply_direct(&source, &bad, DirectPolicy::Pf400)
            .unwrap_err(),
        CudaError::Refused(Refusal::ValueOutOfRange("direct uv field"))
    );
}
