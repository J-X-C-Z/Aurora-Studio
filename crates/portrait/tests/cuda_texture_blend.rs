// Real NVIDIA coverage for PF30009 texture_mask_blend.
// Sixteen parameter combinations, each compared byte-for-byte with the CPU
// entry point `kernels::texture_mask_blend::apply`.

#[cfg(windows)]
mod blend_gpu {
    use aurora_pixcake::cuda::CudaError;
    use aurora_pixcake::cuda_texture_blend::{Blend, KERNEL};
    use aurora_pixcake::kernels::texture_mask_blend::{self, NormalBlend};
    use aurora_pixcake::Image;

    fn gpu() -> Option<Blend> {
        match Blend::open() {
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

    fn original(width: u32, height: u32) -> Image {
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&[
                    (x * 5 + y * 3) as u8,
                    (y * 7 + 11) as u8,
                    (x * 13 + y * 2) as u8,
                    (200u32 - (x + y) % 50) as u8,
                ]);
            }
        }
        Image::new(width, height, pixels).unwrap()
    }

    fn effect(width: u32, height: u32) -> Image {
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&[
                    (250u32 - x * 3) as u8,
                    (x * 9 + y * 4) as u8,
                    (y * 15) as u8,
                    (x + y) as u8,
                ]);
            }
        }
        Image::new(width, height, pixels).unwrap()
    }

    fn mask_plane(width: u32, height: u32) -> Vec<u8> {
        (0..width * height)
            .map(|i| ((i * 29 + 7) % 256) as u8)
            .collect()
    }

    /// UVs that leave [0,1] on purpose, so the clamp path is covered.
    fn uv_plane(width: u32, height: u32) -> Vec<[f32; 2]> {
        (0..width * height)
            .map(|i| {
                let x = i % width;
                let y = i / width;
                [
                    (x as f32 + 0.5) / width as f32 * 1.4 - 0.2,
                    (y as f32 + 0.5) / height as f32 * 0.8 + 0.1,
                ]
            })
            .collect()
    }

    #[test]
    fn sixteen_parameter_combinations_match_the_cpu_lane() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (24u32, 16u32);
        let original = original(width, height);
        let effect = effect(13, 9);
        let mask = mask_plane(width, height);
        let uvs = uv_plane(width, height);

        let mut cases = 0;
        for degree in [0.0f32, 0.5, 0.37, 1.0] {
            for reverse_alpha in [false, true] {
                for uv in [None, Some(uvs.as_slice())] {
                    let params = NormalBlend {
                        degree,
                        reverse_alpha,
                    };
                    let expected =
                        texture_mask_blend::apply(&original, &effect, &mask, uv, params).unwrap();
                    let actual = gpu
                        .apply(&original, &effect, &mask, uv, params)
                        .expect("real GPU dispatch");
                    let differing = expected
                        .pixels
                        .iter()
                        .zip(&actual.pixels)
                        .filter(|(a, b)| a != b)
                        .count();
                    assert_eq!(
                        differing,
                        0,
                        "degree={degree} reverse={reverse_alpha} uv={}: {differing} bytes differ",
                        uv.is_some()
                    );
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 16);
    }

    #[test]
    fn full_frame_matches_the_cpu_lane() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (64u32, 48u32);
        let original = original(width, height);
        let effect = effect(width, height);
        let mask = mask_plane(width, height);
        let expected =
            texture_mask_blend::apply(&original, &effect, &mask, None, NormalBlend::PF30009)
                .unwrap();
        let actual = gpu
            .apply(&original, &effect, &mask, None, NormalBlend::PF30009)
            .expect("real GPU dispatch");
        assert_eq!(actual.pixels, expected.pixels);
        assert_eq!(KERNEL, "texture_mask_blend_kernel");
    }

    /// Every effect resolution must sample identically, including 1x1.
    #[test]
    fn effect_sizes_match_the_cpu_lane() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (21u32, 13u32);
        let original = original(width, height);
        let mask = mask_plane(width, height);
        let uvs = uv_plane(width, height);
        for (ew, eh) in [(1u32, 1u32), (3, 2), (21, 13), (32, 32), (7, 29)] {
            let effect = self::effect(ew, eh);
            for uv in [None, Some(uvs.as_slice())] {
                let expected =
                    texture_mask_blend::apply(&original, &effect, &mask, uv, NormalBlend::PF30009)
                        .unwrap();
                let actual = gpu
                    .apply(&original, &effect, &mask, uv, NormalBlend::PF30009)
                    .expect("real GPU dispatch");
                let differing = expected
                    .pixels
                    .iter()
                    .zip(&actual.pixels)
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(
                    differing, 0,
                    "effect {ew}x{eh} uv={}: {differing} bytes differ",
                    uv.is_some()
                );
            }
        }
    }

    /// Non-finite and misshapen inputs must be refused before any dispatch.
    #[test]
    fn refusals_match_the_cpu_lane() {
        use aurora_pixcake::Refusal;

        let Some(gpu) = gpu() else { return };
        let original = original(8, 6);
        let effect = effect(4, 3);
        let mask = mask_plane(8, 6);
        let pixels = (8 * 6) as usize;

        let mut bad_uv = vec![[0.5f32, 0.5f32]; pixels];
        bad_uv[5] = [f32::NAN, 0.5];
        assert_eq!(
            gpu.apply(&original, &effect, &mask, Some(&bad_uv), NormalBlend::PF30009)
                .err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange(
                "v_bgmasktexcoordOut"
            )))
        );

        assert_eq!(
            gpu.apply(&original, &effect, &mask[..pixels - 1], None, NormalBlend::PF30009)
                .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("s_maskTexture")))
        );
        assert_eq!(
            gpu.apply(&original, &effect, &mask, None, NormalBlend { degree: 2.0, reverse_alpha: false })
                .err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("u_blendDegree")))
        );
        assert_eq!(
            gpu.apply(&original, &effect, &mask, Some(&bad_uv[..pixels - 1]), NormalBlend::PF30009)
                .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch(
                "v_bgmasktexcoordOut"
            )))
        );
    }
}
