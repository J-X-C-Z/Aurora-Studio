//! Type63 Catmull/UVMAP mesh+fragment gate.
//!
//! The CPU half pins the recovered two-pass contract. The Windows half compares
//! each pass and the chained result byte-for-byte against the CUDA fragment
//! emit kernel.

use aurora_portrait::{Image, raster, raster_fragment};
use aurora_portrait::kernels::face_mesh;

const WIDTH: u32 = 1000;
const HEIGHT: u32 = 800;

const RULES: raster::Rules = raster::Rules {
    winding: raster::Winding::PositiveAreaIsFront,
    cull: raster::Cull::None,
    edge: raster::Edge::Inclusive,
    overlap: raster::Overlap::Last,
};
const SAMPLER: raster::Sampler = raster::Sampler {
    wrap: raster::Wrap::ClampToEdge,
};
const PASS1: raster_fragment::Defines = raster_fragment::Defines {
    clamp_edge: true,
    uv_map: false,
    uv_map_ba: false,
    catmull_rom: true,
};
const PASS2: raster_fragment::Defines = raster_fragment::Defines {
    clamp_edge: true,
    uv_map: true,
    uv_map_ba: false,
    catmull_rom: false,
};
const PASSES: [raster_fragment::Defines; 2] = [PASS1, PASS2];

fn points(bytes: &[u8]) -> Vec<[f32; 2]> {
    bytes
        .chunks_exact(8)
        .map(|chunk| {
            [
                f32::from_le_bytes(chunk[..4].try_into().unwrap()),
                f32::from_le_bytes(chunk[4..].try_into().unwrap()),
            ]
        })
        .collect()
}

fn normalized(values: &[[f32; 2]]) -> Vec<[f32; 2]> {
    values
        .iter()
        .map(|point| [point[0] / WIDTH as f32, point[1] / HEIGHT as f32])
        .collect()
}

fn source() -> Image {
    Image::new(
        WIDTH,
        HEIGHT,
        (0..WIDTH * HEIGHT * 4)
            .map(|index| ((index * 37 + 11) % 256) as u8)
            .collect(),
    )
    .unwrap()
}

fn authority() -> Vec<(Vec<[f32; 2]>, Vec<[f32; 2]>, Vec<u16>)> {
    let ebo = include_bytes!("../testdata/pf30092.ebo.u16le")
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    [
        include_bytes!("../testdata/pf30092.a27.f32le").as_slice(),
        include_bytes!("../testdata/pf30092.q27.f32le").as_slice(),
    ]
    .into_iter()
    .map(|raw| {
        let positions = points(raw);
        // The current main tree has the A27/Q27/EBO authority but no wl44 UV
        // producer; offset the normalized positions so both fragment branches
        // see a non-identity source UV.
        let source_uv = normalized(&positions)
            .into_iter()
            .map(|point| [point[0] + 0.13, point[1] + 0.09])
            .collect();
        (positions, source_uv, ebo.clone())
    })
    .collect()
}

fn cpu_apply(
    source: &Image,
    draw: &(Vec<[f32; 2]>, Vec<[f32; 2]>, Vec<u16>),
    defines: raster_fragment::Defines,
) -> Image {
    let (positions, source_uv, ebo) = draw;
    raster::rasterize_with_fragment(
        source,
        raster::Mesh {
            dest: positions,
            source_uv,
            ebo,
        },
        RULES,
        SAMPLER,
        raster_fragment::Edge::ReshapeNormal,
        defines,
        raster_fragment::Clamp::disabled(),
        raster::Tolerance::RESHAPE,
    )
    .unwrap()
}

fn cpu_chain(source: &Image, draws: &[(Vec<[f32; 2]>, Vec<[f32; 2]>, Vec<u16>)]) -> Image {
    let mut current = source.clone();
    for draw in draws {
        for defines in PASSES {
            current = cpu_apply(&current, draw, defines);
        }
    }
    current
}

fn hash64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

#[test]
fn type63_two_pass_cpu_contract_is_stable() {
    assert_eq!(
        PASS1,
        raster_fragment::Defines {
            clamp_edge: true,
            uv_map: false,
            uv_map_ba: false,
            catmull_rom: true,
        }
    );
    assert_eq!(
        PASS2,
        raster_fragment::Defines {
            clamp_edge: true,
            uv_map: true,
            uv_map_ba: false,
            catmull_rom: false,
        }
    );

    let source = source();
    let draws = authority();
    let pass1 = cpu_apply(&source, &draws[0], PASS1);
    let pass2 = cpu_apply(&pass1, &draws[0], PASS2);
    assert_ne!(pass1, source, "Catmull pass must change the covered pixels");
    assert_ne!(pass2, pass1, "UVMAP pass must change the Catmull result");
    let output = cpu_chain(&source, &draws);
    let covered = output
        .pixels
        .chunks_exact(4)
        .zip(source.pixels.chunks_exact(4))
        .filter(|(left, right)| left != right)
        .count();
    assert!(
        covered > 0,
        "the recovered passes must write covered pixels"
    );
    assert_ne!(output, source);
    assert_eq!(hash64(&output.pixels), 0x963A08DFD4724F51);
    eprintln!(
        "WL47_CPU_TWO_PASS_OK sha256_like={:016X} covered={covered}",
        hash64(&output.pixels)
    );
}

#[cfg(windows)]
mod gpu {
    use super::*;
    use aurora_portrait::cuda::CudaError;
    use aurora_portrait::cuda_mesh::{MeshRaster, select_grid};

    fn gpu() -> Option<MeshRaster> {
        match MeshRaster::open() {
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

    #[test]
    fn cuda_fragment_matches_cpu_pass_by_pass_and_chained() {
        let Some(gpu) = gpu() else { return };
        let source = source();
        let draws = authority();
        let mut cpu = source.clone();
        let mut cuda = source.clone();

        for (draw_index, (positions, source_uv, ebo)) in draws.iter().enumerate() {
            let mesh = raster::Mesh {
                dest: positions,
                source_uv,
                ebo,
            };
            for (pass_index, defines) in PASSES.into_iter().enumerate() {
                let expected = raster::rasterize_with_fragment(
                    &cpu,
                    mesh,
                    RULES,
                    SAMPLER,
                    raster_fragment::Edge::ReshapeNormal,
                    defines,
                    raster_fragment::Clamp::disabled(),
                    raster::Tolerance::RESHAPE,
                )
                .unwrap();
                let actual = gpu
                    .rasterize_fragment(
                        &cuda,
                        mesh,
                        RULES,
                        SAMPLER,
                        raster_fragment::Edge::ReshapeNormal,
                        defines,
                        raster_fragment::Clamp::disabled(),
                        raster::Tolerance::RESHAPE,
                    )
                    .expect("real GPU fragment dispatch");
                let differing = expected
                    .pixels
                    .iter()
                    .zip(&actual.image.pixels)
                    .filter(|(left, right)| left != right)
                    .count();
                assert_eq!(
                    differing,
                    0,
                    "draw {draw_index} pass {pass_index}: {differing} of {} bytes differ",
                    expected.pixels.len()
                );
                assert_eq!(
                    actual.select_dispatch.grid.0,
                    select_grid(ebo.len() / 3, HEIGHT as usize).unwrap()
                );
                assert_eq!(actual.emit_dispatch.grid.0, (WIDTH * HEIGHT).div_ceil(128));
                cpu = expected;
                cuda = actual.image;
            }
        }

        assert_eq!(cpu, cuda);
        assert_ne!(cuda, source);
        eprintln!(
            "WL47_FACE_MESH_PARITY_OK sha256_like={:016X}",
            hash64(&cuda.pixels)
        );
    }

    #[test]
    fn face_mesh_api_matches_cpu_and_returns_real_dispatch_receipts() {
        let Some(gpu) = gpu() else { return };
        let source = source();
        let authority = authority();
        let draws = [25, 26, 27, 28]
            .into_iter()
            .enumerate()
            .map(|(index, part)| {
                let (positions, source_uv, ebo) = authority[index / 2].clone();
                face_mesh::Draw::new(
                    part,
                    if index % 2 == 0 { 30092 } else { 30093 },
                    [50.0, 50.0],
                    [0; 32],
                    positions,
                    source_uv,
                    ebo,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let refs = draws.iter().collect::<Vec<_>>();
        let expected = face_mesh::rasterize(&source, &refs).unwrap();
        let actual = face_mesh::rasterize_cuda_with(&gpu, &source, &refs).unwrap();
        assert_eq!(actual.image, expected);
        assert_eq!(actual.dispatches.len(), 16);
        let device = &actual.dispatches[0].device;
        assert!(!device.is_empty());
        assert!(actual.dispatches.iter().all(|dispatch| {
            dispatch.device == *device
                && (dispatch.kernel.contains("mesh_raster_select")
                    || dispatch.kernel.contains("mesh_raster_fragment_emit"))
        }));
        eprintln!(
            "WL47_FACE_MESH_API_PARITY_OK dispatches={} device={device} hash64={:016X}",
            actual.dispatches.len(),
            hash64(&actual.image.pixels)
        );
    }
}
