#[cfg(windows)]
mod hlf_gpu {
    use aurora_portrait::cuda::CudaError;
    use aurora_portrait::cuda_hlf::{BODY_KERNEL, FACE_KERNEL, Hlf};
    use aurora_portrait::kernels::skin_adjust::{
        AdjustLut, BodyHlfUniforms, Roi, body_hlf_apply, hlf_face_apply,
    };
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

        fn unit(&mut self) -> f32 {
            self.byte() as f32 / 255.0
        }
    }

    fn gpu() -> Option<Hlf> {
        match Hlf::open() {
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

    fn plane1(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..len).map(|_| rng.unit()).collect()
    }

    fn plane3(len: usize, seed: u64) -> Vec<f32> {
        plane1(len * 3, seed)
    }

    fn assert_float_bits(expected: &[f32], actual: &[f32], label: &str) {
        assert_eq!(actual.len(), expected.len(), "{label}: length");
        let differing = expected
            .iter()
            .zip(actual)
            .filter(|(left, right)| left.to_bits() != right.to_bits())
            .count();
        assert_eq!(differing, 0, "{label}: {differing} float words differ");
    }

    fn assert_byte_exact(expected: &Image, actual: &Image, label: &str) {
        assert_eq!(
            (actual.width, actual.height),
            (expected.width, expected.height),
            "{label}: shape"
        );
        let differing = expected
            .pixels
            .iter()
            .zip(&actual.pixels)
            .filter(|(left, right)| left != right)
            .count();
        assert_eq!(differing, 0, "{label}: {differing} bytes differ");
    }

    #[test]
    fn authority_fixture_matches_face_math() {
        let source = [0.5f32; 3];
        let median = [0.4f32; 3];
        let gauss = [0.6f32; 3];
        let expected = hlf_face_apply(
            &source,
            &median,
            &gauss,
            &[1.0],
            &[1.0],
            1,
            1,
            Roi {
                x: 0,
                y: 0,
                width: 1,
                height: 1,
            },
            100.0,
        )
        .unwrap();
        assert!((expected[0] - 0.526_0).abs() < 1e-4);
    }

    #[test]
    fn authority_fixture_matches_body_math_and_identity_skips_dispatch() {
        let source = Image::new(2, 1, vec![128, 128, 128, 42, 128, 128, 128, 42]).unwrap();
        let roi = Roi {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        };
        let median = [0.4f32; 6];
        let gauss = [0.6f32; 6];
        let face = [1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let skin = [1.0f32, 1.0];
        let uniforms = BodyHlfUniforms {
            spotless_degree: 0.0,
            retouch_degree: 1.0,
            spotless_acne_threshold: 0.0,
            spotless_high_freq_factor: 1.0,
            spotless_low_freq_alpha: 0.0,
            retouch_high_freq_factor: 0.0,
            diff_min_range: -0.2,
            diff_max_range: 0.2,
        };
        let expected = body_hlf_apply(
            &source, roi, &median, &gauss, &face, &skin, 100.0, uniforms, None,
        )
        .unwrap();
        assert_eq!(expected.pixel(0, 0), [134, 134, 134, 42]);
        assert_eq!(expected.pixel(1, 0), [128, 128, 128, 42]);

        let Some(gpu) = gpu() else { return };
        let (actual, dispatches) = gpu
            .body_apply(
                &source, roi, &median, &gauss, &face, &skin, 100.0, uniforms, None,
            )
            .unwrap();
        assert_eq!(dispatches.len(), 1);
        assert_byte_exact(&expected, &actual, "authority fixture");

        let (identity, dispatches) = gpu
            .body_apply(
                &source, roi, &median, &gauss, &face, &skin, 0.0, uniforms, None,
            )
            .unwrap();
        assert_eq!(identity, source);
        assert!(dispatches.is_empty());
    }

    #[test]
    fn pf30021_matches_cpu_oracle_byte_for_byte() {
        let Some(gpu) = gpu() else { return };
        for (width, height, roi) in [
            (
                1u32,
                1u32,
                Roi {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                },
            ),
            (
                17,
                13,
                Roi {
                    x: 2,
                    y: 3,
                    width: 7,
                    height: 5,
                },
            ),
            (
                37,
                19,
                Roi {
                    x: 5,
                    y: 4,
                    width: 13,
                    height: 9,
                },
            ),
        ] {
            let pixels = width as usize * height as usize;
            let source = plane3(pixels, width as u64 * 101 + height as u64);
            let median = plane3(pixels, width as u64 * 103 + height as u64);
            let gauss = plane3(pixels, width as u64 * 107 + height as u64);
            let face = plane1(pixels, width as u64 * 109 + height as u64);
            let skin = plane1(pixels, width as u64 * 113 + height as u64);
            let expected = hlf_face_apply(
                &source, &median, &gauss, &face, &skin, width, height, roi, 73.0,
            )
            .unwrap();
            let (actual, dispatches) = gpu
                .face_apply(
                    &source, &median, &gauss, &face, &skin, width, height, roi, 73.0,
                )
                .expect("real CUDA PF30021 dispatch");
            assert_eq!(dispatches.len(), 1);
            assert_eq!(dispatches[0].kernel, FACE_KERNEL);
            assert_eq!(
                dispatches[0].grid,
                (width.div_ceil(16), height.div_ceil(16))
            );
            if width == 1 {
                eprintln!(
                    "DEVICE={} CC={}.{} DRIVER={} PTX_ISA={}.{} PTX_BYTES={}",
                    dispatches[0].device,
                    dispatches[0].compute_cap.0,
                    dispatches[0].compute_cap.1,
                    dispatches[0].driver_version,
                    dispatches[0].ptx_isa.0,
                    dispatches[0].ptx_isa.1,
                    dispatches[0].ptx_bytes
                );
            }
            assert_float_bits(&expected, &actual, &format!("{width}x{height}"));
        }
    }

    fn body_uniforms() -> BodyHlfUniforms {
        BodyHlfUniforms {
            spotless_degree: 0.0,
            retouch_degree: 0.73,
            spotless_acne_threshold: 0.0,
            spotless_high_freq_factor: 1.0,
            spotless_low_freq_alpha: 0.0,
            retouch_high_freq_factor: 0.08,
            diff_min_range: -0.2,
            diff_max_range: 0.2,
        }
    }

    fn vendor_adjust_lut() -> &'static [u8] {
        include_bytes!("fixtures/pf30022/vendor_hlfreq_adjust_lut_256x1_l8.bin")
    }

    #[test]
    fn pf30022_retouch_matches_cpu_oracle_and_preserves_outside_roi() {
        let Some(gpu) = gpu() else { return };
        for (width, height, roi) in [
            (
                1u32,
                1u32,
                Roi {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                },
            ),
            (
                19,
                17,
                Roi {
                    x: 3,
                    y: 2,
                    width: 9,
                    height: 7,
                },
            ),
            (
                37,
                19,
                Roi {
                    x: 23,
                    y: 11,
                    width: 11,
                    height: 7,
                },
            ),
        ] {
            let pixels = width as usize * height as usize;
            let source = image(width, height, width as u64 * 131 + height as u64);
            let median = plane3(pixels, width as u64 * 137 + height as u64);
            let gauss = plane3(pixels, width as u64 * 139 + height as u64);
            let face = plane1(pixels * 4, width as u64 * 149 + height as u64);
            let skin = plane1(pixels, width as u64 * 151 + height as u64);
            let uniforms = body_uniforms();
            let expected = body_hlf_apply(
                &source, roi, &median, &gauss, &face, &skin, 73.0, uniforms, None,
            )
            .unwrap();
            let (actual, dispatches) = gpu
                .body_apply(
                    &source, roi, &median, &gauss, &face, &skin, 73.0, uniforms, None,
                )
                .expect("real CUDA PF30022 dispatch");
            assert_eq!(dispatches.len(), 1);
            assert_eq!(dispatches[0].kernel, BODY_KERNEL);
            assert_eq!(
                dispatches[0].grid,
                (roi.width.div_ceil(16), roi.height.div_ceil(16))
            );
            assert_byte_exact(&expected, &actual, &format!("{width}x{height}"));
            for y in 0..height {
                for x in 0..width {
                    let in_roi =
                        x >= roi.x && x < roi.x + roi.width && y >= roi.y && y < roi.y + roi.height;
                    if !in_roi {
                        assert_eq!(
                            actual.pixel(x, y),
                            source.pixel(x, y),
                            "outside ROI changed at ({x},{y})"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn pf30022_spotless_vendor_lut_matches_cpu_and_preserves_alpha() {
        let Some(gpu) = gpu() else { return };
        let luma = vendor_adjust_lut();
        assert_eq!(luma.len(), 256);
        assert_eq!(
            [luma[0], luma[63], luma[128], luma[176], luma[255]],
            [0, 1, 128, 252, 255]
        );
        let lut = AdjustLut {
            width: 256,
            height: 1,
            luma,
        };
        for (width, height, roi) in [
            (
                1u32,
                1u32,
                Roi {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1,
                },
            ),
            (
                19,
                17,
                Roi {
                    x: 3,
                    y: 2,
                    width: 9,
                    height: 7,
                },
            ),
        ] {
            let pixels = width as usize * height as usize;
            let source = image(width, height, width as u64 * 163 + height as u64);
            let median = plane3(pixels, width as u64 * 167 + height as u64);
            let gauss = plane3(pixels, width as u64 * 173 + height as u64);
            let face = plane1(pixels * 4, width as u64 * 179 + height as u64);
            let skin = plane1(pixels, width as u64 * 181 + height as u64);
            let mut uniforms = body_uniforms();
            uniforms.spotless_degree = 0.61;
            uniforms.spotless_acne_threshold = 0.02;
            uniforms.spotless_high_freq_factor = 0.8;
            uniforms.spotless_low_freq_alpha = 0.3;
            let expected = body_hlf_apply(
                &source,
                roi,
                &median,
                &gauss,
                &face,
                &skin,
                73.0,
                uniforms,
                Some(&lut),
            )
            .unwrap();
            let (actual, dispatches) = gpu
                .body_apply(
                    &source,
                    roi,
                    &median,
                    &gauss,
                    &face,
                    &skin,
                    73.0,
                    uniforms,
                    Some(&lut),
                )
                .expect("real CUDA PF30022 spotless dispatch");
            assert_eq!(dispatches.len(), 1);
            assert_eq!(dispatches[0].kernel, BODY_KERNEL);
            assert_byte_exact(&expected, &actual, &format!("{width}x{height}"));
            let changed = source
                .pixels
                .chunks_exact(4)
                .zip(actual.pixels.chunks_exact(4))
                .filter(|(left, right)| left[..3] != right[..3])
                .count();
            assert!(changed > 0, "vendor LUT did not exercise spotless");
            for (left, right) in source
                .pixels
                .chunks_exact(4)
                .zip(actual.pixels.chunks_exact(4))
            {
                assert_eq!(left[3], right[3], "alpha changed");
            }
            eprintln!(
                "PF30022_METRIC={width}x{height} KERNEL={} GRID={}x{} CHANGED_RGB_PIXELS={changed} CPU_CUDA_BYTE_DELTA=0 ALPHA_DELTA=0",
                dispatches[0].kernel,
                dispatches[0].grid.0,
                dispatches[0].grid.1
            );
        }
    }

    #[test]
    fn typed_refusals_happen_before_dispatch() {
        let Some(gpu) = gpu() else { return };
        assert_eq!(
            gpu.face_apply(
                &[0.5; 3],
                &[0.4; 3],
                &[0.6; 3],
                &[1.0],
                &[1.0],
                1,
                1,
                Roi {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1
                },
                101.0,
            )
            .err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("ui")))
        );

        let source = Image::solid(1, 1, [128, 128, 128, 255]);
        let mut uniforms = body_uniforms();
        uniforms.spotless_degree = 1.0;
        assert_eq!(
            gpu.body_apply(
                &source,
                Roi {
                    x: 0,
                    y: 0,
                    width: 1,
                    height: 1
                },
                &[0.4; 3],
                &[0.6; 3],
                &[1.0, 0.0, 0.0, 1.0],
                &[1.0],
                50.0,
                uniforms,
                None,
            )
            .err(),
            Some(CudaError::Refused(Refusal::MissingInput("adjust_lut")))
        );
    }
}
