// Real NVIDIA coverage for PF30187's final hand-neutral-gray consumer.

#[cfg(windows)]
mod hand_neutral_gray_gpu {
    use aurora_portrait::cuda::CudaError;
    use aurora_portrait::cuda_hand_neutral_gray::{gate, Blend, Frame, Gate, KERNEL};
    use aurora_portrait::kernels::hair_body::NeutralTarget;
    use aurora_portrait::{Image, Refusal};

    const FIXTURE: &[u8] = aurora_portrait::private_assets::or_empty(aurora_portrait::private_asset!("fixtures/portrait/pf30187/hand_neutral_fixture.bin"));

    struct Fixture<'a> {
        width: u32,
        height: u32,
        source: Image,
        hand_mask: &'a [u8],
        model_alpha: &'a [u8],
        neutral: Image,
        diff: &'a [u8],
    }

    fn fixture() -> Fixture<'static> {
        let width = u32::from_le_bytes(FIXTURE[0..4].try_into().unwrap());
        let height = u32::from_le_bytes(FIXTURE[4..8].try_into().unwrap());
        let pixels = width as usize * height as usize;
        let mut cursor = 8;
        let source_bytes = &FIXTURE[cursor..cursor + pixels * 4];
        cursor += pixels * 4;
        let hand_mask = &FIXTURE[cursor..cursor + pixels];
        cursor += pixels;
        let model_alpha = &FIXTURE[cursor..cursor + pixels];
        cursor += pixels;
        let neutral_bytes = &FIXTURE[cursor..cursor + pixels * 4];
        cursor += pixels * 4;
        let diff = &FIXTURE[cursor..cursor + pixels * 3 * 4];
        assert_eq!(cursor + pixels * 3 * 4, FIXTURE.len());
        Fixture {
            width,
            height,
            source: Image::new(width, height, source_bytes.to_vec()).unwrap(),
            hand_mask,
            model_alpha,
            neutral: Image::new(width, height, neutral_bytes.to_vec()).unwrap(),
            diff,
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

    fn diff_bits(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
            .collect()
    }

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn bgra_fixture_matches_the_cpu_reference() {
        let Some(blend) = gpu() else { return };
        let f = fixture();
        let diff = diff_bits(f.diff);
        for ui in [0.5f32, 17.5, 50.0, 100.0] {
            let frame = Frame {
                source: &f.source,
                ui,
                hand_mask: Some(f.hand_mask),
                model_alpha: Some(f.model_alpha),
                target: Some(NeutralTarget::Bgra(&f.neutral)),
            };
            let expected = aurora_portrait::cuda_hand_neutral_gray::reference(&frame).unwrap();
            let (actual, dispatch) = blend.apply(&frame).expect("real GPU dispatch");
            assert_eq!(actual.pixels, expected.pixels, "ui={ui}");
            let dispatch = dispatch.expect("active UI must dispatch");
            assert_eq!(dispatch.kernel, KERNEL);
            assert!(!dispatch.device.is_empty());

            let diff_frame = Frame {
                source: &f.source,
                ui,
                hand_mask: Some(f.hand_mask),
                model_alpha: Some(f.model_alpha),
                target: Some(NeutralTarget::Diff(&diff)),
            };
            let expected =
                aurora_portrait::cuda_hand_neutral_gray::reference(&diff_frame).unwrap();
            let (actual, _) = blend.apply(&diff_frame).expect("real GPU dispatch");
            assert_eq!(actual.pixels, expected.pixels, "diff ui={ui}");
        }
        assert_eq!((f.width, f.height), (19, 13));
    }

    #[test]
    fn ui_zero_is_identity_without_producer_or_dispatch() {
        let Some(blend) = gpu() else { return };
        let source = Image::solid(2, 2, [1, 2, 3, 4]);
        let frame = Frame {
            source: &source,
            ui: 0.0,
            hand_mask: None,
            model_alpha: None,
            target: None,
        };
        let (actual, dispatch) = blend.apply(&frame).unwrap();
        assert_eq!(actual, source);
        assert!(dispatch.is_none());
    }

    #[test]
    fn active_missing_or_bad_producers_are_typed_refusals() {
        let source = Image::solid(1, 1, [1, 2, 3, 4]);
        let neutral = Image::solid(1, 1, [9, 9, 9, 4]);
        let hand = [255];
        let alpha = [255];
        assert_eq!(
            gate(&Frame {
                source: &source,
                ui: 50.0,
                hand_mask: None,
                model_alpha: Some(&alpha),
                target: Some(NeutralTarget::Bgra(&neutral)),
            }),
            Err(Refusal::MissingInput("pf30187_hand_mask_producer"))
        );
        assert_eq!(
            gate(&Frame {
                source: &source,
                ui: 50.0,
                hand_mask: Some(&hand),
                model_alpha: None,
                target: Some(NeutralTarget::Bgra(&neutral)),
            }),
            Err(Refusal::MissingInput("pf30187_model_alpha_producer"))
        );
        assert_eq!(
            gate(&Frame {
                source: &source,
                ui: 50.0,
                hand_mask: Some(&hand),
                model_alpha: Some(&alpha),
                target: None,
            }),
            Err(Refusal::MissingInput("pf30187_model_neutral_producer"))
        );
        assert_eq!(
            gate(&Frame {
                source: &source,
                ui: 0.0,
                hand_mask: None,
                model_alpha: None,
                target: None,
            }),
            Ok(Gate::Identity)
        );
    }

    #[test]
    fn half_ties_round_away_from_zero() {
        let Some(blend) = gpu() else { return };
        let source = Image::new(4, 1, vec![0, 1, 254, 77, 1, 2, 253, 88, 254, 0, 1, 0, 255, 1, 2, 255])
            .unwrap();
        let neutral = Image::new(4, 1, vec![1, 2, 255, 0, 2, 3, 254, 0, 255, 1, 2, 0, 0, 0, 0, 0])
            .unwrap();
        let mask = [255; 4];
        let alpha = [255; 4];
        let frame = Frame {
            source: &source,
            ui: 50.0,
            hand_mask: Some(&mask),
            model_alpha: Some(&alpha),
            target: Some(NeutralTarget::Bgra(&neutral)),
        };
        let expected = aurora_portrait::cuda_hand_neutral_gray::reference(&frame).unwrap();
        let (actual, _) = blend.apply(&frame).unwrap();
        assert_eq!(actual.pixels, expected.pixels);
        assert_eq!(&actual.pixels[0..4], &[1, 2, 255, 77]);
        assert_eq!(&actual.pixels[4..8], &[2, 3, 254, 88]);
    }
}
