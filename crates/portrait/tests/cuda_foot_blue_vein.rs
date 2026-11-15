// PF44297 foot blue-vein final composite, byte-for-byte against the CPU oracle.

#[cfg(windows)]
mod foot_blue_vein_gpu {
    use aurora_portrait::cuda::CudaError;
    use aurora_portrait::cuda_foot_blue_vein::{FootBlueVein, Job, KERNEL, PF44297_PRODUCER, Roi};
    use aurora_portrait::kernels::hair_body::foot_blue_vein;
    use aurora_portrait::{Image, Refusal};

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

    fn gpu() -> Option<FootBlueVein> {
        match FootBlueVein::open() {
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

    fn image(width: u32, height: u32, seed: u64) -> Image {
        let mut rng = Rng::new(seed);
        Image::new(
            width,
            height,
            (0..width * height * 4).map(|_| rng.byte()).collect(),
        )
        .unwrap()
    }

    fn mask(width: u32, height: u32, seed: u64) -> Vec<u8> {
        let mut rng = Rng::new(seed);
        (0..width * height)
            .map(|index| {
                let x = index % width;
                let y = index / width;
                let blob = x.abs_diff(width / 3) + y.abs_diff(height / 2) < 3;
                if blob || rng.byte() > 224 { 255 } else { 0 }
            })
            .collect()
    }

    fn assert_byte_exact(expected: &Image, actual: &Image, label: &str) {
        let mut differing = 0usize;
        let mut max_delta = 0u8;
        let mut alpha_mismatches = 0usize;
        for (index, (left, right)) in expected.pixels.iter().zip(&actual.pixels).enumerate() {
            let delta = left.abs_diff(*right);
            differing += usize::from(delta != 0);
            max_delta = max_delta.max(delta);
            if index % 4 == 3 && delta != 0 {
                alpha_mismatches += 1;
            }
        }
        assert_eq!(
            differing, 0,
            "{label}: differing_bytes={differing} max_delta={max_delta} alpha_mismatches={alpha_mismatches}"
        );
    }

    #[test]
    fn fixed_cpu_fixture_and_away_from_zero_rounding() {
        // 50% between RGB [150,90,80] and [52,82,132] is [101,86,106].
        let source = Image::new(1, 1, vec![150, 90, 80, 77]).unwrap();
        let repair = Image::new(1, 1, vec![52, 82, 132, 0]).unwrap();
        let mask = [255u8];
        let expected = foot_blue_vein(&source, 50.0, &mask, &repair, 1).unwrap();
        assert_eq!(expected.pixel(0, 0), [101, 86, 106, 77]);

        // 0.5 rounds away from zero, not to even.
        let source = Image::new(1, 1, vec![0, 1, 2, 91]).unwrap();
        let repair = Image::new(1, 1, vec![1, 2, 3, 0]).unwrap();
        let expected = foot_blue_vein(&source, 50.0, &mask, &repair, 1).unwrap();
        assert_eq!(expected.pixel(0, 0), [1, 2, 3, 91]);

        let Some(gpu) = gpu() else { return };
        let (actual, dispatches) = gpu
            .apply(&Job {
                source: &source,
                producer: Some(PF44297_PRODUCER),
                model_mask: Some(&mask),
                model_repair: Some(&repair),
                ui: 50.0,
                feather_radius: 1,
                roi: Roi::full(1, 1),
            })
            .expect("real CUDA PF44297 dispatch");
        assert_byte_exact(&expected, &actual, "rounding fixture");
        assert_eq!(dispatches.len(), 1);
        assert_eq!(dispatches[0].kernel, KERNEL);
        eprintln!(
            "PF44297 kernel={} device={} cc={:?} driver={} ptx_isa={:?} grid={:?} block={:?}",
            dispatches[0].kernel,
            dispatches[0].device,
            dispatches[0].compute_cap,
            dispatches[0].driver_version,
            dispatches[0].ptx_isa,
            dispatches[0].grid,
            dispatches[0].block,
        );
    }

    #[test]
    fn random_mask_repair_and_feather_are_byte_exact() {
        let Some(gpu) = gpu() else { return };
        let mut total_bytes = 0usize;
        for (width, height) in [(1u32, 1u32), (17, 13), (64, 48), (320, 200)] {
            let source = image(width, height, width as u64 * 101 + height as u64);
            let repair = image(width, height, width as u64 * 211 + height as u64);
            let model_mask = mask(width, height, width as u64 * 307 + height as u64);
            for degree in [1.0f32, 37.0, 50.0, 100.0] {
                for feather_radius in [1u32, 2, 5] {
                    let expected =
                        foot_blue_vein(&source, degree, &model_mask, &repair, feather_radius)
                            .unwrap();
                    let (actual, dispatches) = gpu
                        .apply(&Job {
                            source: &source,
                            producer: Some(PF44297_PRODUCER),
                            model_mask: Some(&model_mask),
                            model_repair: Some(&repair),
                            ui: degree,
                            feather_radius,
                            roi: Roi::full(width, height),
                        })
                        .expect("real CUDA PF44297 dispatch");
                    assert_byte_exact(
                        &expected,
                        &actual,
                        &format!("{width}x{height} degree={degree} radius={feather_radius}"),
                    );
                    assert_eq!(dispatches.len(), 1);
                    assert_eq!(dispatches[0].kernel, KERNEL);
                    total_bytes += expected.pixels.len();
                }
            }
        }
        eprintln!("PF44297 random parity bytes={total_bytes} differing_bytes=0");
    }

    #[test]
    fn zero_ui_has_no_dispatch_and_preserves_every_byte() {
        let Some(gpu) = gpu() else { return };
        let source = image(17, 13, 29);
        let repair = image(17, 13, 31);
        let model_mask = mask(17, 13, 37);
        let (actual, dispatches) = gpu
            .apply(&Job {
                source: &source,
                producer: Some(PF44297_PRODUCER),
                model_mask: Some(&model_mask),
                model_repair: Some(&repair),
                ui: 0.0,
                feather_radius: 2,
                roi: Roi::full(17, 13),
            })
            .unwrap();
        assert_eq!(actual, source);
        assert!(dispatches.is_empty());
    }

    #[test]
    fn refusals_happen_before_device_work() {
        let Some(gpu) = gpu() else { return };
        let source = Image::solid(3, 3, [10, 20, 30, 40]);
        let repair = Image::solid(3, 3, [100, 110, 120, 0]);
        let mask = vec![255; 9];
        let test = |job: Job<'_>, expected: Refusal| {
            assert_eq!(gpu.apply(&job).err(), Some(CudaError::Refused(expected)));
        };

        test(
            Job {
                source: &source,
                producer: None,
                model_mask: Some(&mask),
                model_repair: Some(&repair),
                ui: 100.0,
                feather_radius: 1,
                roi: Roi::full(3, 3),
            },
            Refusal::MissingInput("PF44297_MODEL_PRODUCER"),
        );
        test(
            Job {
                source: &source,
                producer: Some(PF44297_PRODUCER),
                model_mask: None,
                model_repair: Some(&repair),
                ui: 100.0,
                feather_radius: 1,
                roi: Roi::full(3, 3),
            },
            Refusal::MissingInput("PF44297_MODEL_MASK"),
        );
        test(
            Job {
                source: &source,
                producer: Some(PF44297_PRODUCER),
                model_mask: Some(&mask),
                model_repair: None,
                ui: 100.0,
                feather_radius: 1,
                roi: Roi::full(3, 3),
            },
            Refusal::MissingInput("PF44297_MODEL_REPAIR"),
        );
        test(
            Job {
                source: &source,
                producer: Some(PF44297_PRODUCER),
                model_mask: Some(&mask),
                model_repair: Some(&repair),
                ui: 100.0,
                feather_radius: 1,
                roi: Roi {
                    x: 1,
                    y: 1,
                    width: 2,
                    height: 2,
                },
            },
            Refusal::ShapeMismatch("PF44297_ROI"),
        );
    }
}
