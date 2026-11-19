// Real NVIDIA coverage for the PF90087 final seam blend. The search is
// producer-side; this test supplies an explicit repair plane and coverage mask.

#[cfg(windows)]
mod hair_seam_gpu {
    use aurora_portrait::cuda::CudaError;
    use aurora_portrait::cuda_hair_seam::{Blend, KERNEL};
    use aurora_portrait::kernels::face_detail::hair_seam_blend;
    use aurora_portrait::{Image, Refusal};

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

    fn gpu() -> Option<Blend> {
        match Blend::open() {
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

    fn canvas(width: u32, height: u32, seed: u64) -> Image {
        let mut rng = Rng::new(seed);
        Image::new(
            width,
            height,
            (0..width * height * 4).map(|_| rng.byte()).collect(),
        )
        .unwrap()
    }

    fn parity(
        gpu: &Blend,
        source: &Image,
        part_line: &[u8],
        repair: &[u8],
        repair_mask: &[u8],
        ui: f32,
        roi: (u32, u32, u32, u32),
        label: &str,
    ) {
        let expected = hair_seam_blend(
            source,
            Some(part_line),
            Some(repair),
            Some(repair_mask),
            ui,
            roi,
        )
        .unwrap();
        let (actual, dispatches) = gpu
            .apply(
                source,
                Some(part_line),
                Some(repair),
                Some(repair_mask),
                ui,
                roi,
            )
            .expect("real GPU dispatch");
        assert_eq!(actual, expected.output, "{label}");
        if ui == 0.0 || roi.2 == 0 || roi.3 == 0 {
            assert!(dispatches.is_empty(), "{label}");
        } else {
            assert_eq!(dispatches.len(), 1, "{label}");
            assert_eq!(dispatches[0].kernel, KERNEL, "{label}");
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                let dispatch = &dispatches[0];
                eprintln!(
                    "PF90087 kernel={} device={} cc={:?} driver={} ptx_isa={:?} grid={:?}",
                    dispatch.kernel,
                    dispatch.device,
                    dispatch.compute_cap,
                    dispatch.driver_version,
                    dispatch.ptx_isa,
                    dispatch.grid,
                );
            });
        }
        for pixel in 0..source.pixel_count() {
            assert_eq!(
                actual.pixels[pixel * 4 + 3],
                source.pixels[pixel * 4 + 3],
                "{label}: alpha changed"
            );
        }
    }

    #[test]
    fn random_roi_and_mask_shapes_match_cpu_byte_for_byte() {
        let Some(gpu) = gpu() else { return };
        for (width, height) in [(1u32, 1u32), (17, 13), (64, 48)] {
            let pixels = (width * height) as usize;
            let source = canvas(width, height, width as u64 * 31 + height as u64);
            let mut rng = Rng::new(width as u64 * 37 + height as u64);
            let part_line: Vec<u8> = (0..pixels).map(|_| rng.byte()).collect();
            let repair: Vec<u8> = (0..pixels * 3).map(|_| rng.byte()).collect();
            let repair_mask: Vec<u8> = (0..pixels)
                .map(|_| if rng.byte() & 1 == 0 { 0 } else { 255 })
                .collect();
            for ui in [0.0, 0.37, 50.0, 100.0] {
                parity(
                    &gpu,
                    &source,
                    &part_line,
                    &repair,
                    &repair_mask,
                    ui,
                    (0, 0, width, height),
                    "full-frame",
                );
            }
            if width > 2 && height > 2 {
                parity(
                    &gpu,
                    &source,
                    &part_line,
                    &repair,
                    &repair_mask,
                    62.0,
                    (1, 1, width - 2, height - 2),
                    "cropped ROI",
                );
            }
        }
    }

    #[test]
    fn mask_endpoints_and_rounding_are_not_ties_even() {
        let Some(gpu) = gpu() else { return };
        let source = Image::solid(1, 1, [0, 1, 2, 77]);
        let part_line = [255u8];
        let repair = [1u8, 0, 1];
        let repair_mask = [255u8];
        let expected = hair_seam_blend(
            &source,
            Some(&part_line),
            Some(&repair),
            Some(&repair_mask),
            50.0,
            (0, 0, 1, 1),
        )
        .unwrap();
        assert_eq!(expected.output, Image::solid(1, 1, [1, 1, 2, 77]));
        let (actual, _) = gpu
            .apply(
                &source,
                Some(&part_line),
                Some(&repair),
                Some(&repair_mask),
                50.0,
                (0, 0, 1, 1),
            )
            .unwrap();
        assert_eq!(actual, expected.output);
    }

    #[test]
    fn all_mask_bytes_and_alpha_bytes_are_byte_exact() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (16u32, 16u32);
        let pixels = (width * height) as usize;
        let source = Image::new(
            width,
            height,
            (0..pixels)
                .flat_map(|pixel| {
                    [
                        (pixel * 37) as u8,
                        (pixel * 53 + 1) as u8,
                        (pixel * 71 + 2) as u8,
                        pixel as u8,
                    ]
                })
                .collect(),
        )
        .unwrap();
        let part_line: Vec<u8> = (0..pixels).map(|value| value as u8).collect();
        let repair: Vec<u8> = (0..pixels * 3)
            .map(|index| (index * 19 + 7) as u8)
            .collect();
        let repair_mask = vec![255u8; pixels];
        parity(
            &gpu,
            &source,
            &part_line,
            &repair,
            &repair_mask,
            73.0,
            (0, 0, width, height),
            "mask and alpha sweep",
        );
    }

    #[test]
    fn zero_degree_and_empty_roi_do_not_dispatch() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(4, 3, 41);
        let (zero, dispatches) = gpu
            .apply(&source, None, None, None, 0.0, (0, 0, 4, 3))
            .unwrap();
        assert_eq!(zero, source);
        assert!(dispatches.is_empty());
        let (empty, dispatches) = gpu
            .apply(&source, None, None, None, 100.0, (4, 3, 0, 0))
            .unwrap();
        assert_eq!(empty, source);
        assert!(dispatches.is_empty());
    }

    #[test]
    fn typed_refusals_happen_before_launch() {
        let Some(gpu) = gpu() else { return };
        let source = Image::solid(2, 2, [1, 2, 3, 4]);
        assert_eq!(
            gpu.apply(&source, None, None, None, 100.0, (0, 0, 2, 2))
                .err(),
            Some(CudaError::Refused(Refusal::MissingInput(
                "PF90087_PART_LINE_MASK"
            )))
        );
        assert_eq!(
            gpu.apply(
                &source,
                Some(&[255; 4]),
                Some(&[0; 12]),
                Some(&[255; 4]),
                100.0,
                (3, 0, 1, 1),
            )
            .err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("PF90087_ROI")))
        );
    }
}
