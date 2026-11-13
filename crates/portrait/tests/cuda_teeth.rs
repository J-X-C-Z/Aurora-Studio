#[cfg(feature = "external-fixtures")]
#[path = "support/mod.rs"]
mod external_fixtures;

// Real NVIDIA coverage for PF108-PF110 teeth whitening.
//
// The atlases are the shipped assets decoded exactly like
// `processing::pixcake_assets::load_lut_png` (PngDecoder -> into_rgb8), dumped
// to work/backend-r11/o1i/{whiten,bright}.rgb by o1i/decode.

#[cfg(all(windows, feature = "external-fixtures"))]
mod teeth_gpu {
    use aurora_pixcake::cuda::CudaError;
    use aurora_pixcake::cuda_teeth::{Teeth, KERNEL, LUT_BYTES};
    use aurora_pixcake::kernels::face_detail;
    use aurora_pixcake::{Image, Refusal};

    fn whiten_lut() -> &'static [u8] {
        static BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
        BYTES.get_or_init(|| super::external_fixtures::read("teeth/whiten.rgb"))
    }
    fn bright_lut() -> &'static [u8] {
        static BYTES: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
        BYTES.get_or_init(|| super::external_fixtures::read("teeth/bright.rgb"))
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

    fn gpu() -> Option<Teeth> {
        match Teeth::open() {
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

    fn masks(pixels: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng::new(seed);
        (0..pixels).map(|_| rng.byte()).collect()
    }

    fn parity(
        gpu: &Teeth,
        source: &Image,
        mask: &[u8],
        common: f32,
        bright: f32,
        deyellow: f32,
        label: &str,
    ) {
        assert_eq!(whiten_lut().len(), LUT_BYTES);
        assert_eq!(bright_lut().len(), LUT_BYTES);
        let expected = face_detail::teeth_whitening(
            source, mask, whiten_lut(), bright_lut(), common, bright, deyellow,
        )
        .unwrap();
        let (actual, dispatches) = gpu
            .apply(source, mask, whiten_lut(), bright_lut(), common, bright, deyellow)
            .expect("real GPU dispatch");
        let differing =
            expected.output.pixels.iter().zip(&actual.pixels).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing,
            0,
            "{label} ({common},{bright},{deyellow}): {differing} of {} bytes differ",
            expected.output.pixels.len()
        );
        assert_eq!(dispatches.len(), usize::from(expected.touched > 0));
        for dispatch in &dispatches {
            assert_eq!(dispatch.kernel, KERNEL);
            assert!(dispatch.ptx_bytes > 0);
        }
        for pixel in 0..source.pixel_count() {
            assert_eq!(
                actual.pixels[pixel * 4 + 3],
                source.pixels[pixel * 4 + 3],
                "{label}: alpha must be copied"
            );
        }
    }

    #[test]
    fn teeth_stage_matches_the_cpu_oracle_across_ui_combinations() {
        let Some(gpu) = gpu() else { return };
        for (width, height) in [(1u32, 1u32), (17, 13), (32, 24)] {
            let pixels = (width * height) as usize;
            let source = canvas(width, height, width as u64 * 31 + height as u64);
            let mask = masks(pixels, width as u64 * 7 + height as u64);
            for (common, bright, deyellow) in [
                (100.0f32, 100.0f32, 100.0f32),
                (50.0, 70.0, 60.0),
                (100.0, 0.0, 100.0),
                (100.0, 100.0, 0.0),
                (30.0, 30.0, 30.0),
            ] {
                parity(&gpu, &source, &mask, common, bright, deyellow, "ui");
            }
        }
    }

    #[test]
    fn zero_degree_and_mask_gate_are_honoured() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(17, 13, 41);
        let pixels = (17 * 13) as usize;
        let mask = masks(pixels, 43);

        // common = 0 -> both weights 0 -> no launch, identity.
        let (actual, dispatches) = gpu.apply(&source, &mask, whiten_lut(), bright_lut(), 0.0, 70.0, 70.0).unwrap();
        assert!(dispatches.is_empty());
        assert_eq!(actual, source);

        // mask 0 -> gate 0.001 on both weights -> identity even at full degree.
        let zero_mask = vec![0u8; pixels];
        let (actual, dispatches) =
            gpu.apply(&source, &zero_mask, whiten_lut(), bright_lut(), 100.0, 100.0, 100.0).unwrap();
        assert_eq!(dispatches.len(), 1);
        assert_eq!(actual, source, "mask.g == 0 must copy every byte");

        // mask 1 quantizes to byte 1, i.e. weight 1/255 = 0.0039 > 0.001, so the
        // stage does run; it must still equal the CPU lane exactly.
        let tiny_mask = vec![1u8; pixels];
        let (actual, _) = gpu.apply(&source, &tiny_mask, whiten_lut(), bright_lut(), 100.0, 100.0, 100.0).unwrap();
        let expected =
            face_detail::teeth_whitening(&source, &tiny_mask, whiten_lut(), bright_lut(), 100.0, 100.0, 100.0)
                .unwrap()
                .output;
        assert_eq!(actual, expected);
    }

    #[test]
    fn bright_only_and_whiten_only_paths_both_match() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(9, 9, 47);
        let mask = masks(81, 53);
        // deyellow 0 -> dz 0, bright 100 -> ea 1: whiten pass skipped per pixel.
        parity(&gpu, &source, &mask, 100.0, 100.0, 0.0, "bright only");
        // bright 0 -> ea 0, deyellow 100 -> dz 1: bright pass skipped per pixel.
        parity(&gpu, &source, &mask, 100.0, 0.0, 100.0, "whiten only");
    }

    #[test]
    fn bad_material_is_refused_before_launch() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(4, 4, 59);
        let mask = masks(16, 61);
        assert_eq!(
            gpu.apply(&source, &mask[..15], whiten_lut(), bright_lut(), 100.0, 100.0, 100.0).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("mask_g")))
        );
        assert_eq!(
            gpu.apply(&source, &mask, &whiten_lut()[..LUT_BYTES - 3], bright_lut(), 100.0, 100.0, 100.0).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("whiten_lut")))
        );
        assert_eq!(
            gpu.apply(&source, &mask, whiten_lut(), bright_lut(), f32::NAN, 100.0, 100.0).err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("common_ui")))
        );
        assert_eq!(
            gpu.apply(&source, &mask, whiten_lut(), bright_lut(), 100.0, 101.0, 100.0).err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("bright_ui")))
        );
    }

    #[test]
    fn session_revalidates_the_source_before_upload() {
        let Some(gpu) = gpu() else { return };
        let session = gpu.session(4, 4, whiten_lut(), bright_lut()).unwrap();
        let short = Image {
            width: 4,
            height: 4,
            pixels: vec![0; 15],
        };
        assert_eq!(
            session.apply(&short, &[0; 16], 100.0, 100.0, 100.0).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("source")))
        );
    }

    #[test]
    fn one_shot_rejects_a_short_source_image() {
        let Some(gpu) = gpu() else { return };
        let short = Image {
            width: 4,
            height: 4,
            pixels: vec![0; 15],
        };
        assert_eq!(
            gpu.apply(&short, &[0; 16], whiten_lut(), bright_lut(), 100.0, 100.0, 100.0)
                .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("source")))
        );
    }

    #[test]
    fn ui_weight_rounding_has_no_1_byte_drift_on_full_mask_sweep() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(16, 16, 71);
        let mask: Vec<u8> = (0..256).map(|value| value as u8).collect();
        // These common/ui pairs expose the 1-ulp f32 difference between
        // `common * (ui / 100)` and `(common * ui) / 100`.
        for (common, bright, deyellow) in [
            (1.0f32, 10.0f32, 10.0f32),
            (1.0, 20.0, 20.0),
            (2.0, 10.0, 10.0),
            (3.0, 10.0, 10.0),
        ] {
            let expected =
                face_detail::teeth_whitening(
                    &source, &mask, whiten_lut(), bright_lut(), common, bright, deyellow,
                )
                .unwrap()
                .output;
            let (actual, _) = gpu
                .apply(&source, &mask, whiten_lut(), bright_lut(), common, bright, deyellow)
                .unwrap();
            let differing = expected
                .pixels
                .iter()
                .zip(&actual.pixels)
                .filter(|(left, right)| left != right)
                .count();
            assert_eq!(
                differing, 0,
                "ui ({common},{bright},{deyellow}) has one-byte drift"
            );
        }
    }
}
