// Real NVIDIA coverage for the O3b mesh rasterizer.
//
// Every case is compared byte-for-byte against `raster::rasterize`. Meshes are
// synthetic fixtures: no 526 mesh asset and no Capture One contact.

#[cfg(windows)]
mod mesh_gpu {
    use aurora_pixcake::cuda::CudaError;
    use aurora_pixcake::cuda_mesh::{MeshRaster, Output};
    use aurora_pixcake::raster::{self, Cull, Edge, Mesh, Overlap, Rules, Sampler, Winding, Wrap};
    use aurora_pixcake::Image;

    #[derive(Clone)]
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
        }

        fn byte(&mut self) -> u8 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u8
        }
    }

    fn gpu() -> Option<MeshRaster> {
        match MeshRaster::open() {
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

    fn canvas(width: u32, height: u32, seed: u64) -> Image {
        let mut rng = Rng::new(seed);
        Image::new(width, height, (0..width * height * 4).map(|_| rng.byte()).collect()).unwrap()
    }

    fn rules(winding: Winding, cull: Cull, edge: Edge, overlap: Overlap) -> Rules {
        Rules { winding, cull, edge, overlap }
    }

    fn sampler(wrap: Wrap) -> Sampler {
        Sampler { wrap }
    }

    fn drawn(out: &Output, source: &Image) -> bool {
        out.image.pixels != source.pixels
    }

    fn covered(out: &Output) -> usize {
        out.coverage.iter().filter(|&&count| count > 0).count()
    }

    // Byte-for-byte parity, plus the dispatch geometry actually used.
    fn parity(
        gpu: &MeshRaster,
        source: &Image,
        mesh: Mesh<'_>,
        rule: Rules,
        sample: Sampler,
        label: &str,
    ) -> Output {
        let expected = raster::rasterize(source, mesh, rule, sample).unwrap();
        let actual = gpu.rasterize(source, mesh, rule, sample).expect("real GPU dispatch");
        let differing = expected.pixels.iter().zip(&actual.image.pixels).filter(|(a, b)| a != b).count();
        assert_eq!(differing, 0, "{label}: {differing} of {} bytes differ", expected.pixels.len());
        assert_eq!(actual.image.width, source.width);
        assert_eq!(actual.image.height, source.height);
        for dispatch in [&actual.select_dispatch, &actual.emit_dispatch] {
            assert_eq!(dispatch.block, (128, 1), "{label}: 1D 128-thread block");
            assert_eq!(dispatch.grid.1, 1);
            assert!(dispatch.grid.0 >= 1);
            assert!(dispatch.ptx_bytes > 0);
        }
        let triangles = mesh.ebo.len() / 3;
        let pixels = (source.width * source.height) as usize;
        assert_eq!(actual.select_dispatch.grid.0, triangles.div_ceil(128) as u32, "{label} select grid");
        assert_eq!(actual.emit_dispatch.grid.0, pixels.div_ceil(128) as u32, "{label} emit grid");
        actual
    }

    fn quad(width: f32, height: f32) -> ([[f32; 2]; 4], [[f32; 2]; 4], [u16; 6]) {
        let dest = [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]];
        let uv = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        (dest, uv, [0, 1, 2, 0, 2, 3])
    }

    #[test]
    fn cull_and_winding_cover_every_combination() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(9, 7, 3);
        let (dest, uv, ebo) = quad(9.0, 7.0);
        let mesh = Mesh { dest: &dest, source_uv: &uv, ebo: &ebo };
        for winding in [Winding::PositiveAreaIsFront, Winding::NegativeAreaIsFront] {
            for cull in [Cull::None, Cull::Front, Cull::Back] {
                let rule = rules(winding, cull, Edge::Exclusive, Overlap::First);
                parity(&gpu, &source, mesh, rule, sampler(Wrap::ClampToEdge), "cull matrix");
            }
        }
        let triangle_dest = [[1.0, 1.0], [8.0, 1.0], [1.0, 6.0]];
        let triangle_uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let triangle = Mesh { dest: &triangle_dest, source_uv: &triangle_uv, ebo: &[0, 1, 2] };
        let keep_positive = gpu
            .rasterize(
                &source,
                triangle,
                rules(Winding::PositiveAreaIsFront, Cull::Back, Edge::Exclusive, Overlap::First),
                sampler(Wrap::ClampToEdge),
            )
            .unwrap();
        let keep_negative = gpu
            .rasterize(
                &source,
                triangle,
                rules(Winding::NegativeAreaIsFront, Cull::Back, Edge::Exclusive, Overlap::First),
                sampler(Wrap::ClampToEdge),
            )
            .unwrap();
        assert!(
            drawn(&keep_positive, &source) ^ drawn(&keep_negative, &source),
            "exactly one facing must survive CULL_BACK"
        );
    }

    #[test]
    fn overlap_reject_first_last_agree_with_the_oracle() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(11, 9, 5);
        let dest = [
            [1.0, 1.0],
            [10.0, 2.0],
            [2.0, 8.0],
            [2.0, 2.0],
            [10.0, 7.0],
            [1.0, 8.0],
        ];
        let uv = [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.5],
            [0.5, 0.0],
        ];
        let mesh = Mesh { dest: &dest, source_uv: &uv, ebo: &[0, 1, 2, 3, 4, 5] };
        let outputs: Vec<Output> = [Overlap::Reject, Overlap::First, Overlap::Last]
            .into_iter()
            .map(|overlap| {
                let rule = rules(Winding::PositiveAreaIsFront, Cull::None, Edge::Exclusive, overlap);
                parity(&gpu, &source, mesh, rule, sampler(Wrap::ClampToEdge), "overlap")
            })
            .collect();
        assert!(outputs[0].coverage.iter().any(|&count| count > 1), "fixture must overlap");
        assert!(drawn(&outputs[1], &source), "first must draw");
        assert!(drawn(&outputs[2], &source), "last must draw");
        assert_ne!(outputs[1].selected, outputs[2].selected, "first and last must differ");
    }

    // o3b/CULL-ADJUDICATION: CULL_FRONT drops front-facing triangles.
    // 12 cells: area sign x winding x cull, keep = the triangle covers anything.
    #[test]
    fn cull_truth_table_matches_the_adjudication() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(8, 8, 17);
        let dest = [[1.0, 1.0], [6.0, 1.0], [1.0, 5.0]];
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        for (label, ebo) in [("area2>0", [0u16, 1, 2]), ("area2<0", [0, 2, 1])] {
            let mesh = Mesh { dest: &dest, source_uv: &uv, ebo: &ebo };
            for winding in [Winding::PositiveAreaIsFront, Winding::NegativeAreaIsFront] {
                let front = matches!(
                    (winding, label),
                    (Winding::PositiveAreaIsFront, "area2>0") | (Winding::NegativeAreaIsFront, "area2<0")
                );
                for cull in [Cull::None, Cull::Front, Cull::Back] {
                    let expected_keep = match cull {
                        Cull::None => true,
                        Cull::Front => !front,
                        Cull::Back => front,
                    };
                    let rule = rules(winding, cull, Edge::Exclusive, Overlap::First);
                    let out = parity(&gpu, &source, mesh, rule, sampler(Wrap::ClampToEdge), "truth table");
                    assert_eq!(
                        covered(&out) > 0,
                        expected_keep,
                        "{label} winding {winding:?} cull {cull:?} must {}",
                        if expected_keep { "keep" } else { "drop" }
                    );
                }
            }
        }
    }

    #[test]
    fn edge_inclusive_and_exclusive_differ_on_the_boundary() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(6, 6, 7);
        let dest = [[1.5, 1.5], [4.5, 1.5], [1.5, 4.5]];
        let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mesh = Mesh { dest: &dest, source_uv: &uv, ebo: &[0, 1, 2] };
        let mut outputs = Vec::new();
        for edge in [Edge::Exclusive, Edge::Inclusive] {
            let rule = rules(Winding::PositiveAreaIsFront, Cull::None, edge, Overlap::First);
            outputs.push(parity(&gpu, &source, mesh, rule, sampler(Wrap::ClampToEdge), "edge"));
        }
        assert!(
            covered(&outputs[1]) > covered(&outputs[0]),
            "inclusive must cover the boundary sample"
        );
    }

    #[test]
    fn wrap_modes_agree_with_the_oracle() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(8, 6, 11);
        let dest = [[0.0, 0.0], [8.0, 0.0], [0.0, 6.0]];
        let uv = [[-0.4, -0.3], [1.7, 0.2], [0.3, 1.6]];
        let mesh = Mesh { dest: &dest, source_uv: &uv, ebo: &[0, 1, 2] };
        let mut images = Vec::new();
        for wrap in [Wrap::ClampToEdge, Wrap::Repeat] {
            let rule = rules(Winding::PositiveAreaIsFront, Cull::None, Edge::Exclusive, Overlap::First);
            images.push(parity(&gpu, &source, mesh, rule, sampler(wrap), "wrap").image);
        }
        assert_ne!(images[0].pixels, images[1].pixels, "clamp and repeat must differ here");
    }

    #[test]
    fn non_square_canvases_and_uncovered_alpha_are_preserved() {
        let Some(gpu) = gpu() else { return };
        for (width, height) in [(1, 1), (7, 5), (13, 9), (17, 3)] {
            let source = canvas(width, height, width as u64 * 31 + height as u64);
            let dest = [[0.0, 0.0], [4.0, 0.0], [0.0, 3.0]];
            let uv = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
            let mesh = Mesh { dest: &dest, source_uv: &uv, ebo: &[0, 1, 2] };
            let rule = rules(Winding::PositiveAreaIsFront, Cull::None, Edge::Exclusive, Overlap::Reject);
            let out = parity(&gpu, &source, mesh, rule, sampler(Wrap::ClampToEdge), "non-square");
            let mut uncovered = 0;
            for pixel in 0..(width * height) as usize {
                if out.coverage[pixel] == 0 {
                    for channel in 0..4 {
                        assert_eq!(
                            out.image.pixels[pixel * 4 + channel],
                            source.pixels[pixel * 4 + channel],
                            "{width}x{height} uncovered pixel {pixel} must keep the source"
                        );
                    }
                    uncovered += 1;
                }
            }
            if width * height > 1 {
                assert!(uncovered > 0, "{width}x{height} fixture must leave pixels uncovered");
            }
        }
    }

    #[test]
    fn degenerate_triangles_are_dropped_without_error() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(5, 5, 13);
        let dest = [[0.0, 0.0], [4.0, 4.0], [2.0, 2.0]];
        let uv = [[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]];
        let mesh = Mesh { dest: &dest, source_uv: &uv, ebo: &[0, 1, 2] };
        let rule = rules(Winding::PositiveAreaIsFront, Cull::None, Edge::Exclusive, Overlap::Reject);
        let out = parity(&gpu, &source, mesh, rule, sampler(Wrap::ClampToEdge), "degenerate");
        assert_eq!(out.image, source, "a degenerate triangle draws nothing");
        assert!(out.coverage.iter().all(|&count| count == 0));
    }
}
