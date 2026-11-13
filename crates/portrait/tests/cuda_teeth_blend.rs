// PF30026 teeth blend on the GPU, byte-for-byte against the CPU stage.

#[cfg(windows)]
mod teeth_blend_gpu {
    use aurora_pixcake::cuda::CudaError;
    use aurora_pixcake::cuda_teeth_blend::{TeethBlend, KERNEL};
    use aurora_pixcake::kernels::teeth_blend;
    use aurora_pixcake::{Image, Refusal};

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

    fn gpu() -> Option<TeethBlend> {
        match TeethBlend::open() {
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

    fn parity(gpu: &TeethBlend, original: &Image, effect: &Image, mask: &[u8], label: &str) {
        let expected = teeth_blend::apply(original, effect, mask).unwrap();
        let (actual, dispatches) = gpu.apply(original, effect, mask).expect("real GPU dispatch");
        let differing = expected.pixels.iter().zip(&actual.pixels).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing, 0,
            "{label}: {differing} of {} bytes differ",
            expected.pixels.len()
        );
        assert_eq!(dispatches.len(), 1);
        assert_eq!(dispatches[0].kernel, KERNEL);
    }

    #[test]
    fn matches_the_cpu_stage_on_random_and_mask_endpoints() {
        let Some(gpu) = gpu() else { return };
        for (width, height) in [(1u32, 1u32), (17, 13), (32, 24)] {
            let pixels = (width * height) as usize;
            let original = canvas(width, height, width as u64 * 31 + height as u64);
            let effect = canvas(width, height, width as u64 * 37 + height as u64);
            let mut rng = Rng::new(width as u64 * 41 + height as u64);
            let random: Vec<u8> = (0..pixels).map(|_| rng.byte()).collect();
            parity(&gpu, &original, &effect, &random, "random");
            for value in [0u8, 1, 128, 254, 255] {
                let mask = vec![value; pixels];
                parity(&gpu, &original, &effect, &mask, "uniform mask");
            }
        }
    }

    #[test]
    fn alpha_always_follows_the_original() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (17u32, 13u32);
        let pixels = (width * height) as usize;
        let mut original = canvas(width, height, 43);
        let effect = canvas(width, height, 47);
        for pixel in 0..pixels {
            original.pixels[pixel * 4 + 3] = (pixel * 7 % 251) as u8;
        }
        let mask: Vec<u8> = (0..pixels).map(|i| (i * 13 % 256) as u8).collect();
        let (actual, _) = gpu.apply(&original, &effect, &mask).unwrap();
        for pixel in 0..pixels {
            assert_eq!(actual.pixels[pixel * 4 + 3], original.pixels[pixel * 4 + 3]);
        }
        parity(&gpu, &original, &effect, &mask, "alpha");
    }

    #[test]
    fn refusals_match_the_cpu_lane() {
        let Some(gpu) = gpu() else { return };
        let original = canvas(4, 4, 53);
        let effect = canvas(4, 4, 59);
        assert_eq!(
            gpu.apply(&original, &effect, &[0; 15]).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("teeth blend mask")))
        );
        let small = canvas(2, 2, 61);
        assert_eq!(
            gpu.apply(&original, &small, &[0; 16]).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("teeth blend effect")))
        );
    }

    #[test]
    fn malformed_original_is_refused() {
        let Some(gpu) = gpu() else { return };
        let effect = canvas(4, 4, 61);
        let short = Image {
            width: 4,
            height: 4,
            pixels: vec![0; 15],
        };
        assert_eq!(
            gpu.apply(&short, &effect, &[0; 16]).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch(
                "teeth blend original"
            )))
        );
    }

    #[test]
    fn full_mask_sweep_is_byte_exact() {
        let Some(gpu) = gpu() else { return };
        let original = canvas(16, 16, 67);
        let effect = canvas(16, 16, 71);
        let mask: Vec<u8> = (0..256).map(|value| value as u8).collect();
        let expected = teeth_blend::apply(&original, &effect, &mask).unwrap();
        let (actual, _) = gpu.apply(&original, &effect, &mask).unwrap();
        let differing = expected
            .pixels
            .iter()
            .zip(&actual.pixels)
            .filter(|(left, right)| left != right)
            .count();
        assert_eq!(differing, 0, "full mask sweep drifted");
    }
}
