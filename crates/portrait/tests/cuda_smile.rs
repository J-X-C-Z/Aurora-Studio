// PF100007 smile warp on the GPU, byte-for-byte against `pf100007_apply`.
//
// The dedicated kernel keeps the CPU's pixel-space math: byte-space bilinear,
// truncating store, copied alpha, and full-image dispatch so anything outside
// the ROI is the source byte.

#[cfg(windows)]
mod smile_gpu {
    use aurora_pixcake::cuda::CudaError;
    use aurora_pixcake::cuda_smile::{Smile, KERNEL, MASK_KERNEL};
    use aurora_pixcake::kernels::smile_warp::{self, Rect};
    use aurora_pixcake::Refusal;

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

        fn coord(&mut self, span: f32) -> f32 {
            (self.byte() as f32 / 255.0) * 2.0 * span - span
        }
    }

    fn gpu() -> Option<Smile> {
        match Smile::open() {
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

    struct Job {
        width: u32,
        height: u32,
        source: Vec<u8>,
        lip: Vec<u8>,
        mouth: Vec<u8>,
        dx: Vec<f32>,
        dy: Vec<f32>,
        roi: Rect,
        degree: f32,
    }

    fn job(width: u32, height: u32, seed: u64, roi: Rect, degree: f32, spread: f32) -> Job {
        let pixels = (width * height) as usize;
        let mut rng = Rng::new(seed);
        let source: Vec<u8> = (0..pixels * 4).map(|_| rng.byte()).collect();
        let lip: Vec<u8> = (0..pixels).map(|_| rng.byte()).collect();
        let mouth: Vec<u8> = (0..pixels).map(|_| rng.byte()).collect();
        let dx: Vec<f32> = (0..pixels).map(|_| rng.coord(spread)).collect();
        let dy: Vec<f32> = (0..pixels).map(|_| rng.coord(spread)).collect();
        Job { width, height, source, lip, mouth, dx, dy, roi, degree }
    }

    impl Job {
        fn cpu(&self) -> Vec<u8> {
            smile_warp::pf100007_apply(
                &self.source,
                self.width,
                self.height,
                self.degree,
                &self.lip,
                &self.mouth,
                &self.dx,
                &self.dy,
                self.roi,
            )
            .unwrap()
        }

        fn gpu(&self, gpu: &Smile) -> Vec<u8> {
            let (out, dispatches) = gpu
                .apply(
                    &self.source,
                    self.width,
                    self.height,
                    self.degree,
                    &self.lip,
                    &self.mouth,
                    &self.dx,
                    &self.dy,
                    self.roi,
                )
                .expect("real GPU dispatch");
            assert!(dispatches.is_empty() || dispatches[0].kernel == KERNEL);
            out
        }
    }

    fn assert_identical(job: &Job, gpu: &Smile, label: &str) {
        let expected = job.cpu();
        let actual = job.gpu(gpu);
        assert_eq!(expected.len(), actual.len(), "{label}: length");
        for (index, (left, right)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(
                left, right,
                "{label}: byte {index} (pixel {}, channel {}) cpu {left} gpu {right}",
                index / 4,
                index % 4
            );
        }
    }

    #[test]
    fn pf8033_dense_warp_matches_its_cpu_consumer() {
        let Some(gpu) = gpu() else { return };
        for (width, height) in [(1u32, 1u32), (17, 13), (16, 12)] {
            let roi = Rect { x: 0, y: 0, width, height };
            let mut input = job(width, height, 321, roi, 1.0, 0.0);
            for mask in [0u8, 1, 255] {
                input.lip.fill(mask);
                for degree in [0.0f32, 0.001, 0.5, 1.0] {
                    for (dx, dy) in [(3.0, 0.0), (-3.0, 0.0), (0.0, -2.0), (0.37, -0.21)] {
                        input.dx.fill(dx);
                        input.dy.fill(dy);
                        let expected = smile_warp::pf8033_dense_warp(
                            &input.source, width, height, &input.dx, &input.dy, &input.lip, degree,
                        ).unwrap();
                        let (actual, _) = gpu.apply_pf8033(
                            &input.source, width, height, &input.dx, &input.dy, &input.lip, degree,
                        ).expect("PF8033 CUDA stage");
                        assert_eq!(actual, expected, "{width}x{height}, mask={mask}, degree={degree}, shift=({dx},{dy})");
                        for (actual, source) in actual.chunks_exact(4).zip(input.source.chunks_exact(4)) {
                            assert_eq!(actual[3], source[3]);
                        }
                        if width > 1 && mask == 255 && degree == 1.0 && dx == 3.0 {
                            assert_ne!(actual, input.source, "the displacement must move source pixels");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pf8033_reports_its_own_input_refusals() {
        let Some(gpu) = gpu() else { return };
        let source = [10, 20, 30, 77];
        let apply = |width, height, dx: &[f32], dy: &[f32], mask: &[u8], degree| {
            gpu.apply_pf8033(&source, width, height, dx, dy, mask, degree)
        };
        assert!(matches!(
            apply(0, 1, &[0.0], &[0.0], &[255], 1.0),
            Err(CudaError::Refused(Refusal::ShapeMismatch("pf8033 source_bgra")))
        ));
        assert!(matches!(
            apply(1, 1, &[], &[0.0], &[255], 1.0),
            Err(CudaError::Refused(Refusal::MissingInput("pf8033 dx/dy/mask")))
        ));
        assert!(matches!(
            apply(1, 1, &[0.0], &[0.0], &[255], f32::NAN),
            Err(CudaError::Refused(Refusal::ValueOutOfRange("pf8033 degree01")))
        ));
        assert!(matches!(
            apply(1, 1, &[f32::INFINITY], &[0.0], &[255], 1.0),
            Err(CudaError::Refused(Refusal::ValueOutOfRange("pf8033 displacement")))
        ));
    }

    #[test]
    fn nonzero_integer_displacements_move_pixels_in_each_direction() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (17u32, 13u32);
        let roi = Rect { x: 0, y: 0, width, height };
        for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1), (-2, 1)] {
            let mut job = job(width, height, 123, roi, 1.0, 0.0);
            job.lip.fill(255);
            job.mouth.fill(255);
            job.dx.fill(dx as f32);
            job.dy.fill(dy as f32);
            assert_identical(&job, &gpu, "integer displacement");
            let actual = job.gpu(&gpu);
            let output_index = (3 * width as usize + 3) * 4;
            let input_index = ((3 + dy) as usize * width as usize + (3 + dx) as usize) * 4;
            assert_eq!(
                &actual[output_index..output_index + 3],
                &job.source[input_index..input_index + 3]
            );
            assert_eq!(actual[output_index + 3], job.source[output_index + 3]);
            assert_ne!(actual, job.source);
        }
    }

    #[test]
    fn integer_and_fractional_displacements_are_byte_identical() {
        let Some(gpu) = gpu() else { return };
        for (width, height) in [(1u32, 1u32), (17, 13), (33, 20)] {
            let roi = Rect { x: 0, y: 0, width, height };
            for spread in [0.0f32, 2.0, 6.0] {
                let job = job(width, height, width as u64 * 31 + height as u64, roi, 0.85, spread);
                assert_identical(&job, &gpu, "full frame");
            }
        }
    }

    #[test]
    fn non_full_roi_leaves_untouched_region_and_alpha_alone() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (29u32, 21u32);
        let roi = Rect { x: 3, y: 4, width: 17, height: 9 };
        let job = job(width, height, 77, roi, 1.0, 3.0);
        // Non-constant alpha: force a distinct alpha per pixel.
        let mut job = job;
        for pixel in 0..(width * height) as usize {
            job.source[pixel * 4 + 3] = (pixel * 7 % 251) as u8;
        }
        assert_identical(&job, &gpu, "roi");
        let actual = job.gpu(&gpu);
        for pixel in 0..(width * height) as usize {
            let x = (pixel as u32) % width;
            let y = (pixel as u32) / width;
            let inside = x >= roi.x
                && y >= roi.y
                && x < roi.x + roi.width
                && y < roi.y + roi.height;
            if !inside {
                assert_eq!(
                    &actual[pixel * 4..pixel * 4 + 4],
                    &job.source[pixel * 4..pixel * 4 + 4],
                    "pixel ({x},{y}) outside the ROI must be the source"
                );
            }
            assert_eq!(actual[pixel * 4 + 3], job.source[pixel * 4 + 3], "alpha is copied");
        }
    }

    #[test]
    fn zero_support_and_roi_edges_are_identity() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (17u32, 13u32);
        let pixels = (width * height) as usize;
        // support <= 0.001 everywhere: lip 0, and lip 1 with mouth 1.
        let zero = job(width, height, 91, Rect { x: 0, y: 0, width, height }, 1.0, 4.0);
        let mut zero_lip = zero;
        zero_lip.lip = vec![0u8; pixels];
        assert_identical(&zero_lip, &gpu, "zero lip");
        assert_eq!(zero_lip.gpu(&gpu), zero_lip.source);

        let mut tiny = job(width, height, 93, Rect { x: 0, y: 0, width, height }, 1.0, 4.0);
        tiny.lip = vec![1u8; pixels];
        tiny.mouth = vec![1u8; pixels];
        assert_identical(&tiny, &gpu, "gate");
        assert_eq!(tiny.gpu(&gpu), tiny.source, "support 1/255^2 is below the gate");

        // ROI at the extreme corner, one pixel.
        let corner = job(width, height, 97, Rect { x: width - 1, y: height - 1, width: 1, height: 1 }, 1.0, 5.0);
        assert_identical(&corner, &gpu, "corner roi");
        let large = job(width, height, 99, Rect { x: 0, y: 0, width: 1, height: 1 }, 1.0, 5.0);
        assert_identical(&large, &gpu, "one pixel roi");
    }

    #[test]
    fn entry_checks_match_the_cpu_lane() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (9u32, 7u32);
        let pixels = (width * height) as usize;
        let job = job(width, height, 101, Rect { x: 0, y: 0, width, height }, 1.0, 2.0);

        assert_eq!(
            gpu.apply(&job.source, 0, height, 1.0, &job.lip, &job.mouth, &job.dx, &job.dy, job.roi)
                .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("pf100007 bgra")))
        );
        assert_eq!(
            gpu.apply(&job.source, width, height, 1.0, &job.lip, &job.mouth, &job.dx, &job.dy, Rect {
                x: 5,
                y: 0,
                width: 5,
                height: 1,
            })
            .err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("pf100007 roi")))
        );
        assert_eq!(
            gpu.apply(&job.source, width, height, 1.0, &job.lip, &job.mouth, &job.dx, &job.dy, Rect {
                x: u32::MAX,
                y: 0,
                width: 2,
                height: 1,
            })
            .err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("pf100007 roi")))
        );
        let mut bad = job.dx.clone();
        bad[3] = f32::NAN;
        assert_eq!(
            gpu.apply(&job.source, width, height, 1.0, &job.lip, &job.mouth, &bad, &job.dy, job.roi).err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("pf100007 displacement")))
        );
        assert_eq!(
            gpu.apply(&job.source, width, height, 1.0, &job.lip[..pixels - 1], &job.mouth, &job.dx, &job.dy, job.roi)
                .err(),
            Some(CudaError::Refused(Refusal::MissingInput("pf100007 lip_mask")))
        );
        assert_eq!(
            gpu.apply(&job.source, width, height, 1.5, &job.lip, &job.mouth, &job.dx, &job.dy, job.roi).err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("pf100007 degree")))
        );
    }
    #[test]
    fn pf8033_path_uses_the_mask_only_entry() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (17u32, 13u32);
        let pixels = (width * height) as usize;
        let mut rng = Rng::new(5);
        let source: Vec<u8> = (0..pixels * 4).map(|_| rng.byte()).collect();
        let mask = vec![255u8; pixels];
        let dx: Vec<f32> = (0..pixels).map(|_| rng.coord(2.0)).collect();
        let dy: Vec<f32> = (0..pixels).map(|_| rng.coord(2.0)).collect();
        let (out, dispatches) = gpu
            .apply_pf8033(&source, width, height, &dx, &dy, &mask, 1.0)
            .unwrap();
        assert_eq!(dispatches.len(), 1);
        assert_eq!(dispatches[0].kernel, MASK_KERNEL, "PF8033 must not carry a mouth plane");
        let expected = smile_warp::pf8033_dense_warp(&source, width, height, &dx, &dy, &mask, 1.0).unwrap();
        assert_eq!(out, expected);
        assert_ne!(out, source, "the warp must move pixels");
    }

    #[test]
    fn short_source_is_refused_before_dispatch() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (9u32, 7u32);
        let job = job(
            width,
            height,
            111,
            Rect { x: 0, y: 0, width, height },
            1.0,
            2.0,
        );
        let short = &job.source[..job.source.len() - 1];
        assert_eq!(
            gpu.apply(
                short,
                width,
                height,
                1.0,
                &job.lip,
                &job.mouth,
                &job.dx,
                &job.dy,
                job.roi,
            )
            .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("pf100007 bgra")))
        );
        assert_eq!(
            gpu.apply_pf8033(short, width, height, &job.dx, &job.dy, &job.lip, 1.0)
                .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("pf8033 source_bgra")))
        );
    }

    #[test]
    fn scaled_signed_displacements_are_not_abs() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (5u32, 1u32);
        let source: Vec<u8> = (0..5u32)
            .flat_map(|x| [10 + 20 * x as u8, 20 + 20 * x as u8, 30 + 20 * x as u8, 200 + x as u8])
            .collect();
        let mask = vec![255u8; 5];
        let dy = vec![0.0f32; 5];
        let negative = vec![-2.0f32; 5];
        let positive = vec![2.0f32; 5];

        let cpu_neg =
            smile_warp::pf8033_dense_warp(&source, width, height, &negative, &dy, &mask, 0.5)
                .unwrap();
        let (gpu_neg, _) = gpu
            .apply_pf8033(&source, width, height, &negative, &dy, &mask, 0.5)
            .unwrap();
        assert_eq!(gpu_neg, cpu_neg);
        assert_eq!(&gpu_neg[4..7], &source[0..3], "negative dx must move left");
        assert_eq!(gpu_neg[7], source[7], "alpha must be copied");

        let cpu_pos =
            smile_warp::pf8033_dense_warp(&source, width, height, &positive, &dy, &mask, 0.5)
                .unwrap();
        let (gpu_pos, _) = gpu
            .apply_pf8033(&source, width, height, &positive, &dy, &mask, 0.5)
            .unwrap();
        assert_eq!(gpu_pos, cpu_pos);
        assert_eq!(&gpu_pos[4..7], &source[8..11], "positive dx must move right");

        let roi = Rect { x: 0, y: 0, width, height };
        let lip = vec![255u8; 5];
        let mouth = vec![255u8; 5];
        let cpu_open = smile_warp::pf100007_apply(
            &source, width, height, 0.5, &lip, &mouth, &negative, &dy, roi,
        )
        .unwrap();
        let (gpu_open, _) = gpu
            .apply(
                &source, width, height, 0.5, &lip, &mouth, &negative, &dy, roi,
            )
            .unwrap();
        assert_eq!(gpu_open, cpu_open);
        assert_eq!(&gpu_open[4..7], &source[0..3], "open-mouth dx must stay negative");
        assert_eq!(gpu_open[7], source[7], "open-mouth alpha must be copied");
    }
}
