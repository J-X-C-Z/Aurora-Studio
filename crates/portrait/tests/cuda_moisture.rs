// Real NVIDIA coverage for PF90002 face and PF90010 body moisturizing.

#[cfg(windows)]
mod moisture_gpu {
    use aurora_pixcake::cuda::CudaError;
    use aurora_pixcake::cuda_moisture::{Moisture, BODY_KERNEL, FACE_KERNEL};
    use aurora_pixcake::kernels::moisture::{self, FaceParams, Lut1D, MaskChannel, SkinMap};
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

        fn unit(&mut self) -> f32 {
            (self.byte() as f32) / 255.0
        }
    }

    fn gpu() -> Option<Moisture> {
        match Moisture::open() {
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

    fn plane(len: usize, seed: u64) -> Vec<u8> {
        let mut rng = Rng::new(seed);
        (0..len).map(|_| rng.byte()).collect()
    }

    fn weights(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..len).map(|_| rng.unit()).collect()
    }

    fn ramp(len: usize) -> Vec<f32> {
        (0..len).map(|index| index as f32 / (len - 1) as f32).collect()
    }

    #[test]
    fn face_stage_matches_the_cpu_oracle() {
        let Some(gpu) = gpu() else { return };
        let hardlight = ramp(64);
        let beauty = ramp(32);
        let skin = ramp(16 * 8);
        let skin_map = SkinMap { width: 16, height: 8, samples: &skin };
        let params = FaceParams { ui: 70.0, pivot: 0.2, gain: 1.3 };
        for (width, height) in [(1u32, 1u32), (17, 13), (32, 24)] {
            let pixels = (width * height) as usize;
            let source = canvas(width, height, width as u64 * 31 + height as u64);
            let mask = plane(pixels * 4, width as u64 * 7 + height as u64);
            let blur = plane(pixels, width as u64 * 11 + height as u64);
            for channel in [MaskChannel::R, MaskChannel::BA, MaskChannel::RGB] {
                let expected = moisture::face_apply(
                    &source,
                    &mask,
                    &blur,
                    &params,
                    channel,
                    &skin_map,
                    &Lut1D { samples: &hardlight },
                    &Lut1D { samples: &beauty },
                )
                .unwrap();
                let (actual, dispatches) = gpu
                    .face_apply(
                        &source,
                        &mask,
                        &blur,
                        &params,
                        channel,
                        &skin_map,
                        &Lut1D { samples: &hardlight },
                        &Lut1D { samples: &beauty },
                    )
                    .expect("real GPU dispatch");
                let differing =
                    expected.pixels.iter().zip(&actual.pixels).filter(|(a, b)| a != b).count();
                assert_eq!(differing, 0, "{width}x{height}: {differing} bytes differ");
                assert_eq!(dispatches.len(), 1);
                assert_eq!(dispatches[0].kernel, FACE_KERNEL);
                for pixel in 0..pixels {
                    assert_eq!(actual.pixels[pixel * 4 + 3], source.pixels[pixel * 4 + 3]);
                }
            }
        }
    }

    #[test]
    fn body_stage_matches_the_cpu_oracle() {
        let Some(gpu) = gpu() else { return };
        let beauty = ramp(48);
        for (width, height) in [(1u32, 1u32), (17, 13)] {
            let pixels = (width * height) as usize;
            let source = canvas(width, height, width as u64 * 37 + height as u64);
            let skin = weights(pixels, width as u64 * 41 + height as u64);
            let dog = weights(pixels, width as u64 * 43 + height as u64);
            for degree in [1.0f32, 0.5, 0.02] {
                let expected = moisture::body_apply(
                    &source, &skin, &dog, &Lut1D { samples: &beauty }, degree, (0, 0, width, height),
                )
                .unwrap();
                let (actual, dispatches) = gpu
                    .body_apply(&source, &skin, &dog, &Lut1D { samples: &beauty }, degree, (0, 0, width, height))
                    .expect("real GPU dispatch");
                let differing =
                    expected.pixels.iter().zip(&actual.pixels).filter(|(a, b)| a != b).count();
                assert_eq!(differing, 0, "{width}x{height} degree {degree}: {differing} bytes differ");
                assert_eq!(dispatches.len(), 1);
                assert_eq!(dispatches[0].kernel, BODY_KERNEL);
            }
        }
    }

    #[test]
    fn partial_body_roi_preserves_outside_and_matches_inside() {
        let Some(gpu) = gpu() else { return };
        let beauty = ramp(48);
        for (width, height, roi) in [
            (37u32, 19u32, (5u32, 3u32, 11u32, 7u32)),
            (64, 37, (53, 30, 11, 7)),
        ] {
            let (x0, y0, w, h) = roi;
            let pixels = (width * height) as usize;
            let source = canvas(width, height, width as u64 * 47 + height as u64);
            let skin = weights(pixels, width as u64 * 53 + height as u64);
            let dog = weights(pixels, width as u64 * 59 + height as u64);
            let expected =
                moisture::body_apply(&source, &skin, &dog, &Lut1D { samples: &beauty }, 1.0, roi)
                    .unwrap();
            let (actual, dispatches) = gpu
                .body_apply(&source, &skin, &dog, &Lut1D { samples: &beauty }, 1.0, roi)
                .expect("real GPU partial-ROI dispatch");

            assert_eq!(dispatches.len(), 1);
            assert_eq!(dispatches[0].kernel, BODY_KERNEL);
            assert_eq!(dispatches[0].grid, (w.div_ceil(16), h.div_ceil(16)));

            let mut outside = 0usize;
            let mut inside = 0usize;
            for y in 0..height {
                for x in 0..width {
                    let offset = ((y * width + x) * 4) as usize;
                    let in_roi = x >= x0 && x < x0 + w && y >= y0 && y < y0 + h;
                    if in_roi {
                        inside += 1;
                        assert_eq!(
                            &actual.pixels[offset..offset + 4],
                            &expected.pixels[offset..offset + 4],
                            "{width}x{height} roi {roi:?}: CPU/CUDA differs inside at ({x},{y})"
                        );
                    } else {
                        outside += 1;
                        assert_eq!(
                            &actual.pixels[offset..offset + 4],
                            &source.pixels[offset..offset + 4],
                            "{width}x{height} roi {roi:?}: source changed outside at ({x},{y})"
                        );
                    }
                }
            }
            assert_eq!(outside + inside, pixels);
            assert_eq!(outside, pixels - (w * h) as usize);
        }
    }

    #[test]
    fn zero_weight_and_degree_gates_are_identity() {
        let Some(gpu) = gpu() else { return };
        let beauty = ramp(16);
        let source = canvas(17, 13, 47);
        let pixels = (17 * 13) as usize;

        // Body: degree below the 0.01 render gate -> no launch, identity.
        let skin = vec![1.0f32; pixels];
        let dog = vec![1.0f32; pixels];
        let (actual, dispatches) = gpu
            .body_apply(&source, &skin, &dog, &Lut1D { samples: &beauty }, 0.005, (0, 0, 17, 13))
            .unwrap();
        assert!(dispatches.is_empty());
        assert_eq!(actual, source);

        // Body: zero weights -> launched but identity.
        let zero = vec![0.0f32; pixels];
        let (actual, dispatches) = gpu
            .body_apply(&source, &zero, &dog, &Lut1D { samples: &beauty }, 1.0, (0, 0, 17, 13))
            .unwrap();
        assert_eq!(dispatches.len(), 1);
        assert_eq!(actual, source);

        // Face: ui 0 -> no launch, identity.
        let hardlight = ramp(16);
        let skin = ramp(4 * 4);
        let skin_map = SkinMap { width: 4, height: 4, samples: &skin };
        let mask = plane(pixels * 4, 53);
        let blur = plane(pixels, 59);
        let (actual, dispatches) = gpu
            .face_apply(
                &source,
                &mask,
                &blur,
                &FaceParams { ui: 0.0, pivot: 0.2, gain: 1.3 },
                MaskChannel::R,
                &skin_map,
                &Lut1D { samples: &hardlight },
                &Lut1D { samples: &beauty },
            )
            .unwrap();
        assert!(dispatches.is_empty());
        assert_eq!(actual, source);
    }

    #[test]
    fn shape_refusals_are_reported() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(4, 4, 61);
        let beauty = ramp(8);
        let hardlight = ramp(8);
        let skin = ramp(4);
        let skin_map = SkinMap { width: 2, height: 2, samples: &skin };
        let params = FaceParams { ui: 50.0, pivot: 0.0, gain: 1.0 };
        assert_eq!(
            gpu.face_apply(
                &source, &[0u8; 63], &[0u8; 16], &params, MaskChannel::R, &skin_map,
                &Lut1D { samples: &hardlight }, &Lut1D { samples: &beauty },
            )
            .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("mask_rgba")))
        );
        assert_eq!(
            gpu.body_apply(&source, &[0.0; 15], &[1.0; 16], &Lut1D { samples: &beauty }, 1.0, (0, 0, 4, 4)).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("weights")))
        );
        assert_eq!(
            gpu.body_apply(&source, &[1.0; 16], &[1.0; 16], &Lut1D { samples: &beauty }, 1.0, (0, 0, 5, 4)).err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("roi")))
        );
        assert_eq!(
            gpu.body_apply(&source, &[1.0; 16], &[1.0; 16], &Lut1D { samples: &beauty }, 1.0, (u32::MAX, 0, 2, 1)).err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("roi")))
        );
    }
}
