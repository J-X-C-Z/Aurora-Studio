#[cfg(feature = "external-fixtures")]
#[path = "support/mod.rs"]
mod external_fixtures;

// PF100007 Metal contract and golden-pack tests.
//
// The four fixture cases live in `tests/fixtures/metal_smile`. The CPU test
// recomputes every expected byte with `pf100007_apply`; the macOS-only GPU
// test loads the same files through the same loader and compares GPU bytes.

#[cfg(feature = "external-fixtures")]
use std::fs;
#[cfg(feature = "external-fixtures")]
use std::path::Path;

use aurora_pixcake::kernels::smile_warp::Rect;
#[cfg(feature = "external-fixtures")]
use aurora_pixcake::kernels::smile_warp;
#[cfg(feature = "external-fixtures")]
use aurora_pixcake::metal::MetalContext;
use aurora_pixcake::metal_smile::{SmileInput, SmileParams, SMILE_MSL, SMILE_PARAMS_BYTES};
use aurora_pixcake::Refusal;

#[cfg(feature = "external-fixtures")]

fn require_macos() {
    if !cfg!(target_os = "macos") {
        panic!("requires macOS and a Metal device; non-macOS is not a pass");
    }
}

#[cfg(feature = "external-fixtures")]

struct GoldenCase {
    name: String,
    width: u32,
    height: u32,
    degree: f32,
    roi: Rect,
    source: Vec<u8>,
    lip: Vec<u8>,
    mouth: Vec<u8>,
    dx: Vec<f32>,
    dy: Vec<f32>,
    expected: Vec<u8>,
}

#[cfg(feature = "external-fixtures")]

fn read_f32(path: &Path) -> Vec<f32> {
    fs::read(path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Single loader shared by the CPU and GPU tests.
#[cfg(feature = "external-fixtures")]
fn load_cases() -> Vec<GoldenCase> {
    let root = external_fixtures::root().join("metal_smile");
    let index = fs::read_to_string(root.join("index.tsv")).expect("index.tsv");
    let mut cases = Vec::new();
    for line in index.lines().skip(1).filter(|line| !line.trim().is_empty()) {
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 8, "bad index row: {line}");
        let name = fields[0];
        let dir = root.join("cases").join(name);
        cases.push(GoldenCase {
            name: name.to_string(),
            width: fields[1].parse().unwrap(),
            height: fields[2].parse().unwrap(),
            degree: fields[3].parse().unwrap(),
            roi: Rect {
                x: fields[4].parse().unwrap(),
                y: fields[5].parse().unwrap(),
                width: fields[6].parse().unwrap(),
                height: fields[7].parse().unwrap(),
            },
            source: fs::read(dir.join("source.bgra")).unwrap(),
            lip: fs::read(dir.join("lip.u8")).unwrap(),
            mouth: fs::read(dir.join("mouth.u8")).unwrap(),
            dx: read_f32(&dir.join("dx.f32le")),
            dy: read_f32(&dir.join("dy.f32le")),
            expected: fs::read(dir.join("expected.bgra")).unwrap(),
        });
    }
    cases
}

#[cfg(feature = "external-fixtures")]

fn input<'a>(case: &'a GoldenCase) -> SmileInput<'a> {
    SmileInput {
        source_bgra: &case.source,
        width: case.width,
        height: case.height,
        degree: case.degree,
        lip_mask: &case.lip,
        mouth_mask: &case.mouth,
        model_dx: &case.dx,
        model_dy: &case.dy,
        roi: case.roi,
    }
}

#[test]
fn smile_params_are_32_bytes_and_msl_is_one_kernel() {
    assert_eq!(SMILE_PARAMS_BYTES, 32);
    assert_eq!(std::mem::size_of::<SmileParams>(), 32);
    assert_eq!(SMILE_MSL.matches("kernel void ").count(), 1);
    assert!(SMILE_MSL.contains("kernel void pf100007_smile"));
    assert!(SMILE_MSL.contains("static_assert(sizeof(SmileParams) == 32"));
    for index in 0..7 {
        assert!(SMILE_MSL.contains(&format!("[[buffer({index})]]")));
    }
    assert!(SMILE_MSL.contains("uchar(sm_clamp(value, 0.0f, 255.0f))"));
    assert!(SMILE_MSL.contains("out[offset + 3u] = source[offset + 3u]"));
}

#[test]
fn validation_matches_cpu_entry_checks() {
    let source = vec![0u8; 4 * 3 * 4];
    let lip = vec![255u8; 12];
    let mouth = vec![255u8; 12];
    let dx = vec![1.0f32; 12];
    let dy = vec![0.5f32; 12];
    let valid = SmileInput {
        source_bgra: &source,
        width: 4,
        height: 3,
        degree: 0.8,
        lip_mask: &lip,
        mouth_mask: &mouth,
        model_dx: &dx,
        model_dy: &dy,
        roi: Rect {
            x: 0,
            y: 0,
            width: 4,
            height: 3,
        },
    };
    assert_eq!(valid.validate(), Ok(12));

    let zero = SmileInput {
        width: 0,
        roi: Rect {
            x: 0,
            y: 0,
            width: 0,
            height: 3,
        },
        ..valid
    };
    assert_eq!(
        zero.validate(),
        Err(Refusal::ShapeMismatch("pf100007 bgra"))
    );

    let overflow = SmileInput {
        source_bgra: &[],
        width: u32::MAX,
        height: u32::MAX,
        roi: Rect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        },
        ..valid
    };
    assert_eq!(
        overflow.validate(),
        Err(Refusal::ShapeMismatch("pf100007 bgra"))
    );

    for roi in [
        Rect {
            x: 5,
            y: 0,
            width: 1,
            height: 1,
        },
        Rect {
            x: 1,
            y: 0,
            width: u32::MAX,
            height: 1,
        },
    ] {
        assert_eq!(
            SmileInput { roi, ..valid }.validate(),
            Err(Refusal::ValueOutOfRange("pf100007 roi"))
        );
    }

    let mut bad_dx = dx.clone();
    bad_dx[1] = f32::NAN;
    let bad = SmileInput {
        model_dx: &bad_dx,
        ..valid
    };
    assert_eq!(
        bad.validate(),
        Err(Refusal::ValueOutOfRange("pf100007 displacement"))
    );
}

#[test]
#[cfg(feature = "external-fixtures")]
fn golden_pack_cpu_recomputes_all_four_cases() {
    let cases = load_cases();
    assert_eq!(cases.len(), 4);
    for case in &cases {
        let pixels = (case.width * case.height) as usize;
        assert_eq!(case.source.len(), pixels * 4, "{} source", case.name);
        assert_eq!(case.lip.len(), pixels, "{} lip", case.name);
        assert_eq!(case.mouth.len(), pixels, "{} mouth", case.name);
        assert_eq!(case.dx.len(), pixels, "{} dx", case.name);
        assert_eq!(case.dy.len(), pixels, "{} dy", case.name);
        assert_eq!(case.expected.len(), pixels * 4, "{} expected", case.name);

        let params = input(case).params().expect("params");
        let bytes = unsafe {
            std::slice::from_raw_parts(
                &params as *const SmileParams as *const u8,
                std::mem::size_of::<SmileParams>(),
            )
        };
        assert_eq!(bytes.len(), 32);
        assert_eq!(
            u32::from_le_bytes(bytes[0..4].try_into().unwrap()),
            case.width
        );

        let cpu = smile_warp::pf100007_apply(
            &case.source,
            case.width,
            case.height,
            case.degree,
            &case.lip,
            &case.mouth,
            &case.dx,
            &case.dy,
            case.roi,
        )
        .unwrap();
        assert_eq!(cpu, case.expected, "{} CPU oracle drift", case.name);
        for i in 0..pixels {
            assert_eq!(
                cpu[i * 4 + 3],
                case.source[i * 4 + 3],
                "{} alpha",
                case.name
            );
        }
    }
}

#[test]
#[ignore = "requires macOS Metal device"]
#[cfg(feature = "external-fixtures")]
fn golden_pack_gpu_matches_expected_per_case() {
    require_macos();
    let context = MetalContext::new().expect("MetalContext");
    for case in load_cases() {
        let gpu = context.smile(&input(&case)).expect("smile dispatch");
        assert_eq!(gpu, case.expected, "{} GPU mismatch", case.name);
        for i in 0..(case.width * case.height) as usize {
            assert_eq!(
                gpu[i * 4 + 3],
                case.source[i * 4 + 3],
                "{} alpha",
                case.name
            );
        }
    }
}
