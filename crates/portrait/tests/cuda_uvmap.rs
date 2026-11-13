// Real NVIDIA coverage for PF407 type-143 UV-map apply.
//
// Two oracles: the public `uvmap_color::apply` for byte-exact parity, and the
// public `raster_fragment` taps for float fields the u8 field cannot express
// (negative offsets must survive transport).

#[cfg(windows)]
mod uvmap_gpu {
    use aurora_pixcake::cuda::CudaError;
    use aurora_pixcake::cuda_uvmap::{Uvmap, KERNEL};
    use aurora_pixcake::kernels::uvmap_color::{self, Channel};
    use aurora_pixcake::raster_fragment;
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
    }

    fn gpu() -> Option<Uvmap> {
        match Uvmap::open() {
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

    /// u8 field image plus the float4 plane decoded from it (byte/255).
    fn u8_field(width: u32, height: u32, seed: u64) -> (Image, Vec<[f32; 4]>) {
        let mut rng = Rng::new(seed);
        let bytes: Vec<u8> = (0..width * height * 4).map(|_| rng.byte()).collect();
        let floats = bytes
            .chunks_exact(4)
            .map(|pixel| [pixel[0] as f32 / 255.0, pixel[1] as f32 / 255.0, pixel[2] as f32 / 255.0, pixel[3] as f32 / 255.0])
            .collect();
        (Image::new(width, height, bytes).unwrap(), floats)
    }

    fn parity(
        gpu: &Uvmap,
        source: &Image,
        u8_field: &Image,
        field: &[[f32; 4]],
        channel: Channel,
        catmull: Option<(f32, f32)>,
        label: &str,
    ) {
        let expected = uvmap_color::apply(source, u8_field, channel, catmull).unwrap();
        let (actual, dispatches) = gpu.apply(source, field, channel, catmull).expect("real GPU dispatch");
        let differing = expected.pixels.iter().zip(&actual.pixels).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing,
            0,
            "{label}: {differing} of {} bytes differ",
            expected.pixels.len()
        );
        assert_eq!(dispatches.len(), 1);
        assert_eq!(dispatches[0].kernel, KERNEL);
        assert!(dispatches[0].ptx_bytes > 0);
    }

    #[test]
    fn rg_and_ba_match_the_cpu_lane_on_non_square_canvases() {
        let Some(gpu) = gpu() else { return };
        for (width, height) in [(1u32, 1u32), (17, 13), (5, 3), (32, 24)] {
            let source = canvas(width, height, width as u64 * 31 + height as u64);
            let (field_image, field) = u8_field(width, height, width as u64 * 7 + height as u64);
            for channel in [Channel::Rg, Channel::Ba] {
                parity(&gpu, &source, &field_image, &field, channel, None, "bilinear");
                parity(
                    &gpu,
                    &source,
                    &field_image,
                    &field,
                    channel,
                    Some((width as f32, height as f32)),
                    "catmull",
                );
                parity(&gpu, &source, &field_image, &field, channel, Some((10.0, 6.0)), "catmull fo");
            }
        }
    }

    #[test]
    fn zero_field_is_identity_and_channels_are_isolated() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(7, 5, 11);
        let zero = vec![[0.0f32; 4]; (7 * 5) as usize];
        let (zero_image, _) = u8_field(7, 5, 13);
        let zero_bytes = Image::solid(7, 5, [0, 0, 0, 0]);
        let _ = zero_image;
        let (actual, _) = gpu.apply(&source, &zero, Channel::Rg, None).unwrap();
        assert_eq!(actual, source, "zero offset must copy the source exactly");
        let expected = uvmap_color::apply(&source, &zero_bytes, Channel::Rg, None).unwrap();
        assert_eq!(actual, expected);

        // A pure RG offset must not move the BA path.
        let mut rg_only = vec![[0.0f32; 4]; (7 * 5) as usize];
        for value in rg_only.iter_mut() {
            *value = [0.25, 0.0, 0.0, 0.0];
        }
        let (rg_out, _) = gpu.apply(&source, &rg_only, Channel::Rg, None).unwrap();
        let (ba_out, _) = gpu.apply(&source, &rg_only, Channel::Ba, None).unwrap();
        assert_ne!(rg_out, source);
        assert_eq!(ba_out, source);
    }

    #[test]
    fn negative_offsets_survive_transport() {
        let Some(gpu) = gpu() else { return };
        let width = 9u32;
        let height = 7u32;
        let source = canvas(width, height, 17);
        // -0.15 in x moves the tap left; the u8 field could never carry this.
        let field = vec![[-0.15f32, 0.0, 0.0, 0.0]; (width * height) as usize];
        let (actual, _) = gpu.apply(&source, &field, Channel::Rg, None).unwrap();
        assert_ne!(actual, source, "a negative offset must move the sample");

        let mut expected_pixels = Vec::with_capacity(source.pixels.len());
        for y in 0..height {
            for x in 0..width {
                let u = (x as f32 + 0.5) / width as f32 - 0.15;
                let v = (y as f32 + 0.5) / height as f32;
                let color = raster_fragment::sample_bilinear(&source, u, v);
                expected_pixels.extend_from_slice(&raster_fragment::to_rgba8(color));
            }
        }
        assert_eq!(actual.pixels, expected_pixels);
    }

    #[test]
    fn out_of_range_targets_clamp_like_the_cpu_lane() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(6, 4, 19);
        let pixels = (6 * 4) as usize;
        // Large positive and negative offsets push the target outside [0,1].
        // Offsets are exact k/255 so the u8 field image and the float plane agree.
        let mut field = Vec::with_capacity(pixels);
        for index in 0..pixels {
            field.push(if index % 2 == 0 { [200.0f32 / 255.0, 0.0, 0.0, 0.0] } else { [0.0, 200.0 / 255.0, 0.0, 0.0] });
        }
        let mut bytes = Vec::with_capacity(pixels * 4);
        for value in &field {
            bytes.extend_from_slice(&[
                (value[0] * 255.0) as u8,
                (value[1] * 255.0) as u8,
                0,
                0,
            ]);
        }
        let field_image = Image::new(6, 4, bytes).unwrap();
        let expected = uvmap_color::apply(&source, &field_image, Channel::Rg, None).unwrap();
        let (actual, _) = gpu.apply(&source, &field, Channel::Rg, None).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn shape_and_size_refusals_are_reported() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(4, 4, 23);
        let field = vec![[0.0f32; 4]; 16];
        assert_eq!(
            gpu.apply(&source, &field[..15], Channel::Rg, None).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("uv field")))
        );
        assert_eq!(
            gpu.apply(&source, &field, Channel::Rg, Some((0.0, 4.0))).err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("catmull size")))
        );
        let short = Image::new(4, 3, vec![0; 48]).unwrap();
        assert_eq!(
            gpu.apply(&short, &field[..11], Channel::Rg, None).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("uv field")))
        );
    }

    // n03 P1: the recovered type-143 shader samples `mo` through texture2D and
    // the CPU oracle uses sample_bilinear(field). This matrix pins the GPU field
    // tap to the same bilinear contract.
    #[test]
    fn field_tap_bilinear_matches_cpu_on_non_square_matrix() {
        let Some(gpu) = gpu() else { return };
        let mut mismatches = Vec::new();
        let mut checked = 0usize;
        for width in 2..=48u32 {
            for height in 2..=32u32 {
                let pixels = (width * height) as usize;
                let mut rng = Rng::new(width as u64 * 1009 + height as u64 * 9176);
                let source = Image::new(
                    width,
                    height,
                    (0..pixels * 4).map(|_| rng.byte()).collect(),
                )
                .unwrap();
                let field_bytes: Vec<u8> = (0..pixels * 4).map(|_| rng.byte()).collect();
                let field_image = Image::new(width, height, field_bytes.clone()).unwrap();
                let field: Vec<[f32; 4]> = field_bytes
                    .chunks_exact(4)
                    .map(|p| {
                        [
                            p[0] as f32 / 255.0,
                            p[1] as f32 / 255.0,
                            p[2] as f32 / 255.0,
                            p[3] as f32 / 255.0,
                        ]
                    })
                    .collect();
                for channel in [Channel::Rg, Channel::Ba] {
                    let expected =
                        uvmap_color::apply(&source, &field_image, channel, None).unwrap();
                    let (actual, _) = gpu.apply(&source, &field, channel, None).unwrap();
                    if let Some((pixel, _)) = expected
                        .pixels
                        .chunks_exact(4)
                        .zip(actual.pixels.chunks_exact(4))
                        .enumerate()
                        .find(|(_, (left, right))| left != right)
                    {
                        mismatches.push((width, height, channel, pixel));
                    }
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 2914, "matrix coverage");
        assert!(
            mismatches.is_empty(),
            "{}/{} dimension/channel cases differ: {:?}",
            mismatches.len(),
            checked,
            &mismatches[..mismatches.len().min(12)]
        );
    }
}
