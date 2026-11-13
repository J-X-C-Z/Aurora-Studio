// Real NVIDIA execution of the 13 O3d parity vectors.
//
// Source of truth for the vectors: work/backend-r11/o3d/parity-vectors.json
// (mirrored here so the test needs no JSON parser). Each vector is evaluated by
// `mesh_fragment_evaluate` on the GPU with one thread per point and an 88-byte
// by-value ABI block, then compared byte-for-byte with `raster_fragment`.
//
// `tap` vectors are the same eval with an identity UV pair: clamp_edge off, so
// sample_uv == src_uv == dst_uv, and the tap family is selected by defines.

#[cfg(windows)]
mod fragment_gpu {
    use aurora_pixcake::cuda::{CudaError, Dispatch, Nvidia};
    use aurora_pixcake::raster_fragment::{self, Clamp, Defines, Edge};
    use aurora_pixcake::Image;
    use std::ffi::c_void;

    const PTX: &str = include_str!("../native/cuda/mesh_fragment.ptx");
    const BLOCK: (u32, u32) = (128, 1);

    /// 88-byte kernel block, matching `MeshFragmentInput` in mesh_fragment.h.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Abi {
        source: u64,
        out: u64,
        src_uv: u64,
        dst_uv: u64,
        width: i32,
        height: i32,
        count: i32,
        edge: i32,
        clamp_factor: f32,
        protect_x: i32,
        protect_y: i32,
        clamp_edge: i32,
        uv_map: i32,
        uv_map_ba: i32,
        catmull_rom: i32,
        catmull_width: f32,
        catmull_height: f32,
    }

    const _: () = assert!(std::mem::size_of::<Abi>() == 88);

    fn entry(fragment: &str) -> Option<&'static str> {
        PTX.lines()
            .filter_map(|line| {
                let trimmed = line.trim_start();
                trimmed
                    .strip_prefix(".visible .entry ")
                    .or_else(|| trimmed.strip_prefix(".entry "))
            })
            .filter_map(|rest| rest.split('(').next())
            .find(|symbol| symbol.contains(fragment))
    }

    fn gpu() -> Option<Nvidia> {
        match Nvidia::open_ptx(PTX) {
            Ok(gpu) => Some(gpu),
            Err(CudaError::Unavailable(reason)) => {
                eprintln!("SKIP_NO_NVIDIA_DRIVER: {reason}");
                assert!(
                    std::env::var_os("PIXCAKE_CUDA_REQUIRED").is_none(),
                    "CUDA required but unavailable: {reason}"
                );
                None
            }
            Err(broken) => panic!("NVIDIA path present but broken: {broken:?}"),
        }
    }

    /// 5x3 `p30_tex` fixture from work/backend-r11/o3c/PARITY-DATA.md.
    fn p30_tex() -> Image {
        let mut pixels = Vec::with_capacity(5 * 3 * 4);
        for y in 0..3u32 {
            for x in 0..5u32 {
                pixels.extend_from_slice(&[
                    (10 + 30 * x) as u8,
                    (20 + 40 * y) as u8,
                    (11 * x + 7 * y) as u8,
                    (255 - 9 * x - 5 * y) as u8,
                ]);
            }
        }
        Image::new(5, 3, pixels).unwrap()
    }

    struct Vector {
        id: &'static str,
        src_uv: [f32; 2],
        dst_uv: [f32; 2],
        edge: Edge,
        defines: Defines,
        clamp: Clamp,
        catmull_size: (f32, f32),
        rgba8: [u8; 4],
    }

    const PLAIN: Defines = Defines { clamp_edge: false, uv_map: false, uv_map_ba: false, catmull_rom: false };
    const OFF: Clamp = Clamp::disabled();

    fn vectors() -> Vec<Vector> {
        let clip = Clamp { factor: 0.5, protect_x: 1, protect_y: 0 };
        vec![
            Vector {
                id: "bilinear_offcenter",
                src_uv: [0.30, 0.40],
                dst_uv: [0.25, 0.35],
                edge: Edge::ReshapeNormal,
                defines: PLAIN,
                clamp: OFF,
                catmull_size: (0.0, 0.0),
                rgba8: [40, 48, 16, 242],
            },
            Vector {
                id: "catmull_offcenter",
                src_uv: [0.30, 0.40],
                dst_uv: [0.25, 0.35],
                edge: Edge::ReshapeNormal,
                defines: Defines { catmull_rom: true, ..PLAIN },
                clamp: OFF,
                catmull_size: (0.0, 0.0),
                rgba8: [40, 47, 16, 243],
            },
            Vector {
                id: "uvmap_rg",
                src_uv: [0.30, 0.20],
                dst_uv: [0.20, 0.10],
                edge: Edge::ReshapeNormal,
                defines: Defines { uv_map: true, ..PLAIN },
                clamp: OFF,
                catmull_size: (0.0, 0.0),
                rgba8: [66, 50, 6, 251],
            },
            Vector {
                id: "uvmap_ba",
                src_uv: [0.30, 0.20],
                dst_uv: [0.20, 0.10],
                edge: Edge::ReshapeNormal,
                defines: Defines { uv_map: true, uv_map_ba: true, ..PLAIN },
                clamp: OFF,
                catmull_size: (0.0, 0.0),
                rgba8: [25, 20, 37, 255],
            },
            Vector {
                id: "clamp_edge_type63_corner",
                src_uv: [0.02, 0.03],
                dst_uv: [0.30, 0.40],
                edge: Edge::ReshapeNormal,
                defines: Defines { clamp_edge: true, ..PLAIN },
                clamp: OFF,
                catmull_size: (0.0, 0.0),
                rgba8: [36, 44, 14, 244],
            },
            Vector {
                id: "clamp_edge_type63_center",
                src_uv: [0.52, 0.48],
                dst_uv: [0.30, 0.40],
                edge: Edge::ReshapeNormal,
                defines: Defines { clamp_edge: true, ..PLAIN },
                clamp: OFF,
                catmull_size: (0.0, 0.0),
                rgba8: [73, 58, 30, 231],
            },
            Vector {
                id: "adjclamp_protect_x1",
                src_uv: [0.02, 0.50],
                dst_uv: [0.30, 0.40],
                edge: Edge::AdjClampEdge,
                defines: Defines { clamp_edge: true, ..PLAIN },
                clamp: clip,
                catmull_size: (0.0, 0.0),
                rgba8: [40, 48, 16, 243],
            },
            Vector {
                id: "adjclamp_disabled",
                src_uv: [0.02, 0.03],
                dst_uv: [0.30, 0.40],
                edge: Edge::AdjClampEdge,
                defines: Defines { clamp_edge: true, ..PLAIN },
                clamp: OFF,
                catmull_size: (0.0, 0.0),
                rgba8: [10, 20, 0, 255],
            },
            Vector {
                id: "boundary_uv_zero",
                src_uv: [0.0, 0.0],
                dst_uv: [0.0, 0.0],
                edge: Edge::ReshapeNormal,
                defines: PLAIN,
                clamp: OFF,
                catmull_size: (0.0, 0.0),
                rgba8: [10, 20, 0, 255],
            },
            Vector {
                id: "catmull_fo_size",
                src_uv: [0.30, 0.40],
                dst_uv: [0.30, 0.40],
                edge: Edge::ReshapeNormal,
                defines: Defines { catmull_rom: true, ..PLAIN },
                clamp: OFF,
                catmull_size: (10.0, 6.0),
                rgba8: [40, 48, 16, 243],
            },
            Vector {
                id: "catmull_native_size",
                src_uv: [0.30, 0.40],
                dst_uv: [0.30, 0.40],
                edge: Edge::ReshapeNormal,
                defines: Defines { catmull_rom: true, ..PLAIN },
                clamp: OFF,
                catmull_size: (0.0, 0.0),
                rgba8: [40, 47, 16, 243],
            },
            Vector {
                id: "texel_left",
                src_uv: [0.10, 0.50],
                dst_uv: [0.10, 0.50],
                edge: Edge::ReshapeNormal,
                defines: PLAIN,
                clamp: OFF,
                catmull_size: (0.0, 0.0),
                rgba8: [10, 60, 7, 250],
            },
            Vector {
                id: "texel_right",
                src_uv: [0.30, 0.50],
                dst_uv: [0.30, 0.50],
                edge: Edge::ReshapeNormal,
                defines: PLAIN,
                clamp: OFF,
                catmull_size: (0.0, 0.0),
                rgba8: [40, 60, 18, 241],
            },
        ]
    }

    /// Rust oracle for one vector, including the `tap` family.
    fn oracle(source: &Image, vector: &Vector) -> [u8; 4] {
        let tap = vector.src_uv == vector.dst_uv
            && !vector.defines.clamp_edge
            && vector.catmull_size != (0.0, 0.0);
        if tap {
            // Size override only exists for the standalone Catmull tap.
            let color = raster_fragment::sample_catmull_rom(
                source,
                vector.src_uv[0],
                vector.src_uv[1],
                vector.catmull_size,
            );
            return raster_fragment::to_rgba8(color);
        }
        let color = raster_fragment::evaluate(
            source,
            vector.src_uv,
            vector.dst_uv,
            vector.edge,
            vector.defines,
            vector.clamp,
        );
        raster_fragment::to_rgba8(color)
    }

    fn evaluate_gpu(gpu: &Nvidia, kernel: &'static str, source: &Image, vector: &Vector) -> ([u8; 4], Dispatch) {
        let source_buf = gpu.alloc_buffer(source.pixels.len()).expect("alloc source");
        let src_buf = gpu.alloc_buffer(8).expect("alloc src_uv");
        let dst_buf = gpu.alloc_buffer(8).expect("alloc dst_uv");
        let out_buf = gpu.alloc_buffer(4).expect("alloc out");
        gpu.upload_buffer(&source_buf, &source.pixels).expect("upload source");
        // 8 bytes per UV pair, written as one buffer each.
        let mut src_bytes = Vec::with_capacity(8);
        src_bytes.extend_from_slice(&vector.src_uv[0].to_le_bytes());
        src_bytes.extend_from_slice(&vector.src_uv[1].to_le_bytes());
        let mut dst_bytes = Vec::with_capacity(8);
        dst_bytes.extend_from_slice(&vector.dst_uv[0].to_le_bytes());
        dst_bytes.extend_from_slice(&vector.dst_uv[1].to_le_bytes());
        gpu.upload_buffer(&src_buf, &src_bytes).expect("upload src_uv");
        gpu.upload_buffer(&dst_buf, &dst_bytes).expect("upload dst_uv");

        let mut abi = Abi {
            source: source_buf.handle(),
            out: out_buf.handle(),
            src_uv: src_buf.handle(),
            dst_uv: dst_buf.handle(),
            width: source.width as i32,
            height: source.height as i32,
            count: 1,
            edge: match vector.edge {
                Edge::ReshapeNormal => 0,
                Edge::AdjClampEdge => 1,
            },
            clamp_factor: vector.clamp.factor,
            protect_x: vector.clamp.protect_x as i32,
            protect_y: vector.clamp.protect_y as i32,
            clamp_edge: i32::from(vector.defines.clamp_edge),
            uv_map: i32::from(vector.defines.uv_map),
            uv_map_ba: i32::from(vector.defines.uv_map_ba),
            catmull_rom: i32::from(vector.defines.catmull_rom),
            catmull_width: vector.catmull_size.0,
            catmull_height: vector.catmull_size.1,
        };
        let mut params: [*mut c_void; 1] = [&mut abi as *mut Abi as *mut c_void];
        let dispatch = gpu
            .launch_raw(kernel, (1, 1), BLOCK, &mut params)
            .expect("real GPU dispatch");
        let mut rgba = [0u8; 4];
        gpu.download_buffer(&out_buf, &mut rgba).expect("download fragment");
        (rgba, dispatch)
    }

    #[test]
    fn all_thirteen_vectors_run_on_the_gpu_and_match_the_oracle() {
        let Some(gpu) = gpu() else { return };
        let kernel = entry("mesh_fragment_evaluate").expect("evaluate entry");
        let source = p30_tex();
        for vector in vectors() {
            let (actual, dispatch) = evaluate_gpu(&gpu, kernel, &source, &vector);
            assert_eq!(dispatch.kernel, kernel);
            assert_eq!(dispatch.block, BLOCK);
            assert!(dispatch.ptx_bytes > 0);
            assert_eq!(
                actual, vector.rgba8,
                "{}: documented rgba8 must reproduce on the GPU",
                vector.id
            );
            assert_eq!(
                actual,
                oracle(&source, &vector),
                "{}: GPU must equal raster_fragment",
                vector.id
            );
        }
    }

    #[test]
    fn non_square_source_and_secondary_channel_offsets_are_covered() {
        let Some(gpu) = gpu() else { return };
        let kernel = entry("mesh_fragment_evaluate").expect("evaluate entry");
        // 5x3 is already non-square; this pins the UVMAP secondary offsets that
        // write into .r/.g (dx, dy) and into .b/.a (dx, dy).
        let source = p30_tex();
        assert_ne!(source.width, source.height);
        let rg = vectors().into_iter().find(|v| v.id == "uvmap_rg").unwrap();
        let ba = vectors().into_iter().find(|v| v.id == "uvmap_ba").unwrap();
        let (rg_out, _) = evaluate_gpu(&gpu, kernel, &source, &rg);
        let (ba_out, _) = evaluate_gpu(&gpu, kernel, &source, &ba);
        assert_eq!(rg_out, rg.rgba8);
        assert_eq!(ba_out, ba.rgba8);
        assert_ne!(rg_out, ba_out);
    }

    // Timing lives behind `--ignored` so the normal suite stays fast.
    //   check.exe fragment_gpu::thirteen_vector_timing --ignored --nocapture
    #[test]
    #[ignore]
    fn thirteen_vector_timing() {
        let Some(gpu) = gpu() else { return };
        let kernel = entry("mesh_fragment_evaluate").expect("evaluate entry");
        let source = p30_tex();
        let vectors = vectors();
        let iterations = 200;
        let started = std::time::Instant::now();
        for _ in 0..iterations {
            for vector in &vectors {
                let (actual, _) = evaluate_gpu(&gpu, kernel, &source, vector);
                assert_eq!(actual, vector.rgba8, "{} drifted during timing", vector.id);
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        let per_vector = elapsed * 1e6 / (iterations * vectors.len()) as f64;
        eprintln!(
            "GPU fragments: {} launches in {:.3} s, {:.1} us per vector (alloc+upload+launch+readback)",
            iterations * vectors.len(),
            elapsed,
            per_vector
        );
    }
}
