// Real NVIDIA coverage for PF8069 negative FaceSurfaceBlur and PF8070 clarity.

#[cfg(windows)]
mod skin_models_gpu {
    use std::time::{Duration, Instant};

    use aurora_portrait::cuda::CudaError;
    use aurora_portrait::cuda_skin_models::{
        BLEND_KERNEL, BlendUniforms, CLARITY_KERNEL, ClarityUniforms, FilterUniforms, GAUSS_KERNEL,
        SURFACE_KERNEL, SkinModels,
    };
    use aurora_portrait::kernels::skin_models::{clarity_apply, texture_surface_blur};
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

    fn gpu() -> Option<SkinModels> {
        match SkinModels::open() {
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

    fn plane(len: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng::new(seed);
        (0..len).map(|_| rng.byte()).collect()
    }

    fn assert_byte_exact(expected: &Image, actual: &Image, label: &str) {
        assert_eq!(
            (actual.width, actual.height),
            (expected.width, expected.height)
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
    fn ptx_abi_matches_the_c_header() {
        assert_eq!(std::mem::size_of::<FilterUniforms>(), 32);
        assert_eq!(std::mem::size_of::<BlendUniforms>(), 32);
        assert_eq!(std::mem::size_of::<ClarityUniforms>(), 48);
    }

    #[test]
    fn pf8069_negative_branch_matches_the_cpu_oracle() {
        let Some(gpu) = gpu() else { return };
        for (width, height) in [(1u32, 1u32), (17, 13), (37, 19), (64, 37)] {
            let source = image(width, height, width as u64 * 101 + height as u64);
            let pixels = source.pixel_count();
            let mask = plane(pixels, width as u64 * 31 + height as u64);
            let extra = plane(pixels, width as u64 * 43 + height as u64);
            for ui in [-100.0f32, -25.0, -0.1] {
                let expected = texture_surface_blur(&source, &mask, &extra, ui).unwrap();
                let (actual, dispatches) = gpu
                    .texture_apply(&source, &mask, &extra, ui)
                    .expect("real CUDA PF8069 dispatch");
                assert_byte_exact(&expected, &actual, &format!("{width}x{height} ui {ui}"));
                assert_eq!(
                    dispatches.iter().map(|row| row.kernel).collect::<Vec<_>>(),
                    [
                        SURFACE_KERNEL,
                        SURFACE_KERNEL,
                        GAUSS_KERNEL,
                        GAUSS_KERNEL,
                        BLEND_KERNEL
                    ]
                );
                assert!(!dispatches[0].device.is_empty());
                for pixel in 0..pixels {
                    assert_eq!(actual.pixels[pixel * 4 + 3], source.pixels[pixel * 4 + 3]);
                }
            }
        }
    }

    #[test]
    fn pf8069_identity_and_positive_refusal_do_not_launch() {
        let Some(gpu) = gpu() else { return };
        let source = image(17, 13, 7);
        let mask = plane(source.pixel_count(), 11);
        let (actual, dispatches) = gpu.texture_apply(&source, &mask, &mask, 0.0).unwrap();
        assert_eq!(actual, source);
        assert!(dispatches.is_empty());
        assert_eq!(
            gpu.texture_apply(&source, &mask, &mask, 50.0).err(),
            Some(CudaError::Refused(Refusal::NotImplemented(8069)))
        );
    }

    #[test]
    fn pf8070_clarity_matches_the_cpu_oracle_inside_and_preserves_outside() {
        let Some(gpu) = gpu() else { return };
        for (width, height, roi) in [
            (1u32, 1u32, (0u32, 0u32, 1u32, 1u32)),
            (37, 19, (5, 3, 11, 7)),
            (64, 37, (53, 30, 11, 7)),
            (19, 37, (2, 29, 13, 8)),
        ] {
            let source = image(width, height, width as u64 * 151 + height as u64);
            let (roi_x, roi_y, roi_w, roi_h) = roi;
            let base = image(roi_w, roi_h, width as u64 * 163 + height as u64);
            let blur = image(roi_w, roi_h, width as u64 * 173 + height as u64);
            let mask = plane((roi_w * roi_h) as usize, width as u64 * 181 + height as u64);
            for (degree, threshold, factor) in [
                (1.0f32, 0.0f32, 4.68f32),
                (0.37, 0.04, 2.25),
                (0.02, 0.0, 4.68),
            ] {
                let expected =
                    clarity_apply(&source, &base, &blur, &mask, roi, degree, threshold, factor)
                        .unwrap();
                let (actual, dispatches) = gpu
                    .clarity_apply(&source, &base, &blur, &mask, roi, degree, threshold, factor)
                    .expect("real CUDA PF8070 dispatch");
                assert_eq!(dispatches.len(), 1);
                assert_eq!(dispatches[0].kernel, CLARITY_KERNEL);
                assert_eq!(dispatches[0].grid, (roi_w.div_ceil(16), roi_h.div_ceil(16)));
                assert_byte_exact(
                    &expected,
                    &actual,
                    &format!("{width}x{height} roi {roi:?} degree {degree}"),
                );
                for y in 0..height {
                    for x in 0..width {
                        let in_roi =
                            x >= roi_x && x < roi_x + roi_w && y >= roi_y && y < roi_y + roi_h;
                        if !in_roi {
                            let offset = ((y * width + x) * 4) as usize;
                            assert_eq!(
                                &actual.pixels[offset..offset + 4],
                                &source.pixels[offset..offset + 4],
                                "source changed outside ROI at ({x},{y})"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn shape_refusals_match_the_cpu_oracle() {
        let Some(gpu) = gpu() else { return };
        let source = image(4, 4, 13);
        let base = image(2, 2, 17);
        let blur = image(2, 2, 19);
        let mask = plane(4, 23);
        assert_eq!(
            gpu.clarity_apply(
                &source,
                &base,
                &blur,
                &mask[..3],
                (0, 0, 2, 2),
                1.0,
                0.0,
                4.68
            )
            .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("mask_r")))
        );
        assert_eq!(
            gpu.clarity_apply(&source, &base, &blur, &mask, (0, 0, 3, 2), 1.0, 0.0, 4.68)
                .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("roi")))
        );
        assert_eq!(
            gpu.clarity_apply(&source, &base, &blur, &mask, (3, 3, 2, 2), 1.0, 0.0, 4.68)
                .err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("roi")))
        );
    }

    fn percentile(samples: &[Duration], percent: usize) -> Duration {
        let index = ((samples.len() - 1) * percent).div_ceil(100);
        samples[index]
    }

    #[test]
    #[ignore = "manual P50/P95 profile"]
    fn profile_p50_p95() {
        let Some(gpu) = gpu() else { return };
        let source = image(512, 512, 29);
        let mask = vec![255u8; source.pixel_count()];
        let (_, warm) = gpu.texture_apply(&source, &mask, &mask, -30.0).unwrap();
        println!(
            "DEVICE={} CC={}.{} DRIVER={} PTX_ISA={}.{} PTX_BYTES={} KERNEL={}",
            warm[0].device,
            warm[0].compute_cap.0,
            warm[0].compute_cap.1,
            warm[0].driver_version,
            warm[0].ptx_isa.0,
            warm[0].ptx_isa.1,
            warm[0].ptx_bytes,
            warm[0].kernel
        );
        for _ in 0..2 {
            let _ = gpu.texture_apply(&source, &mask, &mask, -30.0).unwrap();
        }
        let mut texture = Vec::new();
        for _ in 0..20 {
            let start = Instant::now();
            let _ = gpu.texture_apply(&source, &mask, &mask, -30.0).unwrap();
            texture.push(start.elapsed());
        }
        texture.sort_unstable();

        let base = image(256, 256, 31);
        let blur = image(256, 256, 37);
        let roi_mask = vec![255u8; 256 * 256];
        let roi = (128, 128, 256, 256);
        for _ in 0..3 {
            let _ = gpu
                .clarity_apply(&source, &base, &blur, &roi_mask, roi, 1.0, 0.0, 4.68)
                .unwrap();
        }
        let mut clarity = Vec::new();
        for _ in 0..30 {
            let start = Instant::now();
            let _ = gpu
                .clarity_apply(&source, &base, &blur, &roi_mask, roi, 1.0, 0.0, 4.68)
                .unwrap();
            clarity.push(start.elapsed());
        }
        clarity.sort_unstable();
        println!(
            "PF8069 512x512 upload+5dispatch+readback P50={:?} P95={:?}",
            percentile(&texture, 50),
            percentile(&texture, 95)
        );
        println!(
            "PF8070 256x256 ROI upload+dispatch+readback P50={:?} P95={:?}",
            percentile(&clarity, 50),
            percentile(&clarity, 95)
        );
    }
}
