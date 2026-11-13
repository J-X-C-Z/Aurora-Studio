// Real NVIDIA coverage for PF8068 SKIN_GLOSS.
//
// Synthetic caller-owned curves, byte-for-byte against `skin_models::gloss_apply_with_luts`, with the three-pass gate
// made visible through the returned dispatch list.

#[cfg(windows)]
mod gloss_gpu {
    use aurora_pixcake::cuda::CudaError;
    use aurora_pixcake::cuda_gloss::{Gloss, KERNEL};
    use aurora_pixcake::kernels::skin_models;
    use aurora_pixcake::{Image, Refusal};

    // Test curves only; no product or vendor LUT data.
    const TEST_LUTS: [[u8; 256]; 3] = [[255; 256], [160; 256], [0; 256]];
    fn luts() -> Option<[&'static [u8; 256]; 3]> {
        Some([&TEST_LUTS[0], &TEST_LUTS[1], &TEST_LUTS[2]])
    }

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

    fn gpu() -> Option<Gloss> {
        match Gloss::open() {
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

    fn masks(pixels: usize, seed: u64) -> ([Vec<u8>; 3], [Vec<u8>; 3]) {
        let mut rng = Rng::new(seed);
        let make = |rng: &mut Rng| (0..pixels).map(|_| rng.byte()).collect::<Vec<u8>>();
        (
            [make(&mut rng), make(&mut rng), make(&mut rng)],
            [make(&mut rng), make(&mut rng), make(&mut rng)],
        )
    }

    fn parity(
        gpu: &Gloss,
        source: &Image,
        mask: &[Vec<u8>; 3],
        extra: &[Vec<u8>; 3],
        percent: f32,
        label: &str,
    ) {
        let expected = skin_models::gloss_apply_with_luts(
            source,
            [&mask[0], &mask[1], &mask[2]],
            [&extra[0], &extra[1], &extra[2]],
            percent, luts())
        .unwrap();
        let (actual, dispatches) = gpu
            .apply_with_luts(source, [&mask[0], &mask[1], &mask[2]], [&extra[0], &extra[1], &extra[2]], percent, luts())
            .expect("real GPU dispatch");
        let differing = expected.pixels.iter().zip(&actual.pixels).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing,
            0,
            "{label} {percent}%: {differing} of {} bytes differ",
            expected.pixels.len()
        );
        assert_eq!(actual.width, source.width);
        assert_eq!(actual.height, source.height);
        for dispatch in &dispatches {
            assert_eq!(dispatch.kernel, KERNEL);
            assert_eq!(dispatch.block, (16, 16));
            assert!(dispatch.ptx_bytes > 0);
        }
        // The gate must be visible: one launch per non-zero degree.
        let expected_passes = [0.0f32, 50.0, 75.0, 100.0]
            .iter()
            .position(|&value| value == percent)
            .map(|index| [1usize, 0, 2, 2][index]);
        if let Some(expected_passes) = expected_passes {
            assert_eq!(dispatches.len(), expected_passes, "{label} {percent}% launch count");
        }
    }

    #[test]
    fn gloss_matches_the_cpu_oracle_across_percentages() {
        let Some(gpu) = gpu() else { return };
        for (width, height) in [(1, 1), (17, 13), (32, 32)] {
            let source = canvas(width, height, width as u64 * 17 + height as u64);
            let (mask, extra) = masks((width * height) as usize, width as u64 * 7 + height as u64);
            for percent in [0.0f32, 50.0, 75.0, 100.0] {
                parity(&gpu, &source, &mask, &extra, percent, "grid");
            }
        }
    }

    #[test]
    fn zero_mask_leaves_every_byte_alone() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(17, 13, 41);
        let pixels = (17 * 13) as usize;
        let zero = vec![0u8; pixels];
        let masks = [zero.clone(), zero.clone(), zero];
        let extra = [vec![255u8; pixels], vec![255u8; pixels], vec![255u8; pixels]];
        // 100% runs two passes, so a skipped blend would still be visible.
        let (actual, dispatches) = gpu
            .apply_with_luts(&source, [&masks[0], &masks[1], &masks[2]], [&extra[0], &extra[1], &extra[2]], 100.0, luts())
            .unwrap();
        assert_eq!(dispatches.len(), 2);
        assert_eq!(actual, source, "mask 0 must be the identity");
    }

    #[test]
    fn degree_gate_skips_passes_exactly_like_the_cpu() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(8, 8, 43);
        let (mask, extra) = masks(64, 47);
        let run = |percent: f32| {
            gpu.apply_with_luts(&source, [&mask[0], &mask[1], &mask[2]], [&extra[0], &extra[1], &extra[2]], percent, luts())
                .unwrap()
        };
        assert_eq!(run(0.0).1.len(), 1, "0% runs only the negative pass");
        assert_eq!(run(50.0).1.len(), 0, "50% is the neutral gate");
        assert_eq!(run(50.0).0, source, "neutral gate is the identity");
        assert_eq!(run(75.0).1.len(), 2);
        assert_eq!(run(100.0).1.len(), 2);
        assert!(run(0.0).0.pixels != source.pixels, "0% must darken via the negative table");
    }

    #[test]
    fn alpha_is_copied_and_saturated_pixels_stay_in_range() {
        let Some(gpu) = gpu() else { return };
        let mut pixels = Vec::with_capacity(9 * 4);
        for pixel in 0..9u8 {
            pixels.extend_from_slice(&[pixel.wrapping_mul(28), 255 - pixel * 3, pixel * 17, 200 + pixel]);
        }
        let source = Image::new(3, 3, pixels).unwrap();
        let all = vec![255u8; 9];
        let (actual, _) = gpu.apply_with_luts(&source, [&all, &all, &all], [&all, &all, &all], 100.0, luts()).unwrap();
        for pixel in 0..9usize {
            assert_eq!(actual.pixels[pixel * 4 + 3], source.pixels[pixel * 4 + 3], "alpha copied");
        }
        assert_eq!(
            actual,
            skin_models::gloss_apply_with_luts(&source, [&all, &all, &all], [&all, &all, &all], 100.0, luts()).unwrap()
        );
    }

    #[test]
    fn missing_tables_refuse_without_dispatch_and_neutral_needs_none() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(4, 4, 17);
        let masks = [&[255; 16][..]; 3];
        assert_eq!(gpu.apply(&source, masks, masks, 100.0).err(),
            Some(CudaError::Refused(Refusal::MissingInput("gloss_luts"))));
        let (neutral, dispatches) = gpu.apply(&source, masks, masks, 50.0).unwrap();
        assert_eq!(neutral, source);
        assert!(dispatches.is_empty());
    }

    #[test]
    fn bad_material_is_refused_before_launch() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(4, 4, 53);
        let (mask, extra) = masks(16, 59);
        let short = [&mask[0][..15], &mask[1], &mask[2]];
        assert_eq!(
            gpu.apply_with_luts(&source, short, [&extra[0], &extra[1], &extra[2]], 100.0, luts()).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("mask_r")))
        );
        assert_eq!(
            gpu.apply_with_luts(
                &source,
                [&mask[0], &mask[1], &mask[2]],
                [&extra[0], &extra[1], &extra[2]],
                f32::NAN, luts())
            .err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("native_percent")))
        );
        assert_eq!(
            gpu.apply_with_luts(
                &source,
                [&mask[0], &mask[1], &mask[2]],
                [&extra[0], &extra[1], &extra[2]],
                101.0, luts())
            .err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("native_percent")))
        );
    }
}
