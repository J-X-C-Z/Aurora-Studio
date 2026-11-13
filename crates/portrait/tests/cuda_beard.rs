// PF21026 stubble removal on the GPU, byte-for-byte against the CPU entry.

#[cfg(windows)]
mod beard_gpu {
    use aurora_pixcake::cuda::CudaError;
    use aurora_pixcake::cuda_beard::{Beard, KERNEL};
    use aurora_pixcake::kernels::beard;
    use aurora_pixcake::Image;

    fn gpu() -> Option<Beard> {
        match Beard::open() {
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

    fn noise(len: usize, seed: u32) -> Vec<u8> {
        let mut state = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 24) as u8
            })
            .collect()
    }

    #[test]
    fn matches_the_cpu_lane_across_masks_and_strength() {
        let Some(gpu) = gpu() else { return };
        for (width, height) in [(1u32, 1u32), (17, 13), (320, 200)] {
            let pixels = (width * height) as usize;
            let source = Image::new(width, height, noise(pixels * 4, width + 3)).unwrap();
            let repair = Image::new(width, height, noise(pixels * 4, height + 7)).unwrap();
            let mask_a = noise(pixels, width * 5 + 1);
            let mask_b = noise(pixels, height * 9 + 2);
            for strength in [0.0f32, 0.005, 0.01, 0.5, 1.0] {
                let expected = beard::apply(&source, &repair, &mask_a, &mask_b, strength).unwrap();
                let (actual, dispatches) = gpu
                    .apply(&source, &repair, &mask_a, &mask_b, strength)
                    .expect("real GPU dispatch");
                let differing = expected
                    .pixels
                    .iter()
                    .zip(&actual.pixels)
                    .filter(|(a, b)| a != b)
                    .count();
                assert_eq!(differing, 0, "{width}x{height} strength {strength}");
                if strength <= 0.005 {
                    assert!(dispatches.is_empty(), "gate must skip the launch");
                } else {
                    assert_eq!(dispatches[0].kernel, KERNEL);
                }
            }
        }
    }

    #[test]
    fn cascade_masks_and_alpha_are_preserved() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (64u32, 48u32);
        let pixels = (width * height) as usize;
        let source = Image::solid(width, height, [10, 20, 30, 200]);
        let repair = Image::solid(width, height, [200, 210, 220, 9]);
        let zeros = vec![0u8; pixels];
        let full = vec![255u8; pixels];
        let (actual, _) = gpu.apply(&source, &repair, &full, &zeros, 1.0).unwrap();
        assert_eq!(actual, source, "w = min(255, 0) = 0 keeps the source");
        let (actual, _) = gpu.apply(&source, &repair, &full, &full, 1.0).unwrap();
        let expected = beard::apply(&source, &repair, &full, &full, 1.0).unwrap();
        assert_eq!(actual, expected);
        // 64x48 is under the 300 px boundary, so strength is 0.75 here.
        assert_eq!(actual.pixel(0, 0), [153, 163, 173, 200]);
    }
}
