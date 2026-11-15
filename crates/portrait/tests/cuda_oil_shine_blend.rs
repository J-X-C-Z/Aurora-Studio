#[cfg(windows)]
mod oil_shine_gpu {
    use aurora_portrait::cuda::CudaError;
    use aurora_portrait::cuda_oil_shine_blend::{ACTIVE_GATE, Blend, Frame, Gate, KERNEL};
    use aurora_portrait::kernels::skin_color::oil_shine_mask_blend;
    use aurora_portrait::{Image, Refusal};

    fn gpu() -> Option<Blend> {
        match Blend::open() {
            Ok(gpu) => Some(gpu),
            Err(CudaError::Unavailable(reason)) => {
                eprintln!("SKIP_NO_NVIDIA_DRIVER: {reason}");
                assert!(
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none()
                        && std::env::var_os("PIXCAKE_CUDA_REQUIRED").is_none(),
                    "CUDA required but unavailable: {reason}"
                );
                None
            }
            Err(broken) => panic!("NVIDIA path present but broken: {broken:?}"),
        }
    }

    fn frame_inputs() -> (Image, Vec<u8>, Vec<u8>) {
        let (width, height) = (31u32, 17u32);
        let pixels = width * height;
        let source = Image::new(
            width,
            height,
            (0..pixels * 4)
                .map(|i| ((i * 37 + 11) % 256) as u8)
                .collect(),
        )
        .unwrap();
        let effect = (0..pixels * 3)
            .map(|i| ((i * 19 + 203) % 256) as u8)
            .collect();
        let mask = (0..pixels).map(|i| ((i * 29 + 7) % 256) as u8).collect();
        (source, effect, mask)
    }

    #[test]
    fn active_degrees_match_the_cpu_reference() {
        let Some(blend) = gpu() else { return };
        let (source, effect, mask) = frame_inputs();
        let active = f32::from_bits(ACTIVE_GATE.to_bits() + 1);
        for degree in [active, 0.25, 0.5, 1.0] {
            let frame = Frame {
                source: &source,
                effect_rgb: Some(&effect),
                mask: Some(&mask),
                degree,
            };
            let expected = oil_shine_mask_blend(&source, &effect, &mask, degree).unwrap();
            let (actual, dispatch) = blend.apply(&frame).expect("real GPU dispatch");
            assert_eq!(actual.pixels, expected.pixels, "degree={degree}");
            let dispatch = dispatch.expect("active degree must dispatch");
            assert_eq!(dispatch.kernel, KERNEL);
            assert!(!dispatch.device.is_empty());
            eprintln!(
                "DISPATCH degree={degree} kernel={} device={} compute_cap={:?} ptx_isa={:?}",
                dispatch.kernel, dispatch.device, dispatch.compute_cap, dispatch.ptx_isa
            );
        }
    }

    #[test]
    fn inactive_gate_is_identity_without_dispatch() {
        let Some(blend) = gpu() else { return };
        let (source, _, _) = frame_inputs();
        for degree in [0.0, 0.0005, ACTIVE_GATE] {
            let frame = Frame {
                source: &source,
                effect_rgb: None,
                mask: None,
                degree,
            };
            let (actual, dispatch) = blend.apply(&frame).unwrap();
            assert_eq!(actual, source, "degree={degree}");
            assert!(dispatch.is_none());
        }
    }

    #[test]
    fn half_ties_use_nearest_even_and_preserve_alpha() {
        let Some(blend) = gpu() else { return };
        let source = Image::new(
            4,
            1,
            vec![0, 1, 255, 77, 1, 2, 254, 88, 0, 1, 255, 0, 1, 2, 254, 255],
        )
        .unwrap();
        let effect = vec![1, 2, 254, 2, 3, 253, 1, 2, 254, 2, 3, 253];
        let mask = vec![255; 4];
        let frame = Frame {
            source: &source,
            effect_rgb: Some(&effect),
            mask: Some(&mask),
            degree: 0.5,
        };
        let expected = oil_shine_mask_blend(&source, &effect, &mask, 0.5).unwrap();
        assert_eq!(&expected.pixels[0..4], &[0, 2, 254, 77]);
        assert_eq!(&expected.pixels[4..8], &[2, 2, 254, 88]);
        let (actual, _) = blend.apply(&frame).unwrap();
        assert_eq!(actual.pixels, expected.pixels);
    }

    #[test]
    fn active_missing_or_bad_producers_are_typed_refusals() {
        let source = Image::solid(1, 1, [1, 2, 3, 4]);
        let effect = [9, 9, 9];
        let mask = [255];
        assert_eq!(
            aurora_portrait::cuda_oil_shine_blend::gate(&Frame {
                source: &source,
                effect_rgb: None,
                mask: Some(&mask),
                degree: 0.5,
            }),
            Err(Refusal::MissingInput("pf90080_effect_rgb_producer"))
        );
        assert_eq!(
            aurora_portrait::cuda_oil_shine_blend::gate(&Frame {
                source: &source,
                effect_rgb: Some(&effect),
                mask: Some(&mask[..0]),
                degree: 0.5,
            }),
            Err(Refusal::ShapeMismatch("mask"))
        );
        assert_eq!(
            aurora_portrait::cuda_oil_shine_blend::gate(&Frame {
                source: &source,
                effect_rgb: Some(&effect[..2]),
                mask: Some(&mask),
                degree: 0.5,
            }),
            Err(Refusal::ShapeMismatch("effect_rgb"))
        );
        assert_eq!(
            aurora_portrait::cuda_oil_shine_blend::gate(&Frame {
                source: &source,
                effect_rgb: Some(&effect),
                mask: Some(&mask),
                degree: f32::NAN,
            }),
            Err(Refusal::ValueOutOfRange("degree"))
        );
        assert_eq!(
            aurora_portrait::cuda_oil_shine_blend::gate(&Frame {
                source: &source,
                effect_rgb: Some(&effect),
                mask: Some(&mask),
                degree: ACTIVE_GATE,
            }),
            Ok(Gate::Identity)
        );
    }
}
