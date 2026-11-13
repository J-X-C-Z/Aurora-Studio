// PF90116 eye-bag composite on the GPU, byte-for-byte against the R3 law.

#[cfg(windows)]
mod eye_bag_gpu {
    use aurora_pixcake::cuda::CudaError;
    use aurora_pixcake::cuda_eye_bag::{reference, EyeBag, KERNEL};
    use aurora_pixcake::kernels::eye_bag::{self, BlendMode};
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

    fn gpu() -> Option<EyeBag> {
        match EyeBag::open() {
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

    fn response(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..len).map(|_| rng.byte() as f32 / 255.0).collect()
    }

    fn parity(gpu: &EyeBag, source: &Image, color: &Image, mask: &[f32], degree: f32, label: &str) {
        let expected = reference(source, color, mask, degree).unwrap();
        let (actual, dispatches) = gpu.compose(source, color, mask, degree).expect("real GPU dispatch");
        let differing = expected.pixels.iter().zip(&actual.pixels).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing, 0,
            "{label} degree {degree}: {differing} of {} bytes differ",
            expected.pixels.len()
        );
        for dispatch in &dispatches {
            assert_eq!(dispatch.kernel, KERNEL);
        }
        for pixel in 0..source.pixel_count() {
            assert_eq!(actual.pixels[pixel * 4 + 3], source.pixels[pixel * 4 + 3], "{label}: alpha copied");
        }
    }

    #[test]
    fn matches_the_r3_law_across_degrees_and_endpoints() {
        let Some(gpu) = gpu() else { return };
        for (width, height) in [(1u32, 1u32), (17, 13), (64, 48)] {
            let pixels = (width * height) as usize;
            let source = canvas(width, height, width as u64 * 31 + height as u64);
            let color = canvas(width, height, width as u64 * 37 + height as u64);
            let mask = response(pixels, width as u64 * 41 + height as u64);
            for degree in [0.0f32, 0.001, 0.5, 1.0] {
                parity(&gpu, &source, &color, &mask, degree, "random");
            }
            // Endpoints: a pure white repair over a black source and vice versa.
            let black = Image::solid(width, height, [0, 0, 0, 128]);
            let white = Image::solid(width, height, [255, 255, 255, 9]);
            let full = vec![1.0f32; pixels];
            parity(&gpu, &black, &white, &full, 1.0, "black->white");
            parity(&gpu, &white, &black, &full, 1.0, "white->black");
            parity(&gpu, &black, &white, &full, 0.5, "black->white half");
        }
    }

    #[test]
    fn zero_mask_nan_mask_and_degree_gate_are_identity() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (17u32, 13u32);
        let pixels = (width * height) as usize;
        let source = canvas(width, height, 51);
        let color = canvas(width, height, 53);

        // degree 0: no launch at all.
        let mask = response(pixels, 59);
        let (actual, dispatches) = gpu.compose(&source, &color, &mask, 0.0).unwrap();
        assert!(dispatches.is_empty());
        assert_eq!(actual, source);

        // response 0 -> q 0 -> m 0: launched, but every pixel keeps the source.
        let zero = vec![0.0f32; pixels];
        let (actual, dispatches) = gpu.compose(&source, &color, &zero, 1.0).unwrap();
        assert_eq!(dispatches.len(), 1);
        assert_eq!(actual, source);

        // NaN response -> NaN m -> skip.
        let nan = vec![f32::NAN; pixels];
        let (actual, _) = gpu.compose(&source, &color, &nan, 1.0).unwrap();
        assert_eq!(actual, source);
        assert_eq!(reference(&source, &color, &nan, 1.0).unwrap(), source);
    }

    #[test]
    fn refusals_match_the_production_cpu_entry() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(4, 4, 61);
        let color = canvas(4, 4, 63);
        assert_eq!(
            gpu.compose(&source, &color, &vec![0.0f32; 15][..], 1.0).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("response")))
        );
        assert_eq!(
            gpu.compose(&source, &color, &vec![0.0f32; 16][..], 1.5).err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("degree")))
        );
        assert_eq!(
            gpu.compose(&source, &color, &vec![0.0f32; 16][..], f32::NAN).err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("degree")))
        );
        assert_eq!(
            eye_bag::compose_pf90116(&mut [0.0f32; 3], &[0.0f32; 3], &[1.0f32], 1.0, BlendMode::Vendor),
            Ok(())
        );
    }

    #[test]
    fn float_color_and_response_quantization_match_the_production_cpu() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(17, 13, 71);
        let color: Vec<f32> = (0..source.pixel_count() * 3)
            .map(|i| (i % 311) as f32 * 1.125 - 20.75)
            .collect();
        let values = [
            -1.0, 0.0, 0.5, 1.0, 2.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY,
            0.5 / 255.0, 1.5 / 255.0, 2.5 / 255.0,
        ];
        let response: Vec<f32> = (0..source.pixel_count()).map(|i| values[i % values.len()]).collect();
        for degree in [0.0, 0.001, 0.00101, 0.37, 1.0] {
            let expected = eye_bag::compose_field_image_pf90116(
                &source, &color, &response, degree,
            ).unwrap();
            let (actual, _) = gpu.compose_field(&source, &color, &response, degree).unwrap();
            assert_eq!(actual, expected, "degree={degree}");
        }
        assert_eq!(
            gpu.compose_field(&source, &color[..3], &response, 1.0).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("pf90116 planes"))),
        );
    }
}
