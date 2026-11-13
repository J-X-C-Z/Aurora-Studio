// Shared makeup layer on the GPU, byte-for-byte against the CPU law.

#[cfg(windows)]
mod makeup_gpu {
    use aurora_pixcake::cuda::CudaError;
    use aurora_pixcake::cuda_makeup::{MakeupLayer, KERNEL};
    use aurora_pixcake::kernels::makeup_material::{
        self, BlendMode, ColorMaterialParams, MaskChannel,
    };
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
            self.byte() as f32 / 255.0
        }
    }

    fn gpu() -> Option<MakeupLayer> {
        match MakeupLayer::open() {
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

    fn uv(count: usize, seed: u64, spread: f32) -> Vec<[f32; 2]> {
        let mut rng = Rng::new(seed);
        (0..count)
            .map(|_| [rng.unit() * spread, rng.unit() * spread])
            .collect()
    }

    fn params(mode: BlendMode, mask_channel: MaskChannel, degree: f32, blend_alpha: bool, alpha: u8) -> ColorMaterialParams {
        ColorMaterialParams {
            mix_color: [255, 255, 255, alpha],
            opacity_percent: 100.0,
            degree01: degree,
            blend_mode: mode,
            mask_channel,
            blend_alpha,
        }
    }

    fn parity(
        gpu: &MakeupLayer,
        source: &Image,
        material: &Image,
        mask: &Image,
        mask_uv: &[[f32; 2]],
        material_uv: &[[f32; 2]],
        settings: ColorMaterialParams,
        label: &str,
    ) {
        let expected =
            makeup_material::apply_image_layer_dual(source, material, mask, mask_uv, material_uv, settings)
                .unwrap();
        let (actual, dispatches) = gpu
            .apply_image_layer_dual(source, material, mask, mask_uv, material_uv, settings)
            .expect("real GPU dispatch");
        let differing = expected.pixels.iter().zip(&actual.pixels).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing, 0,
            "{label}: {differing} of {} bytes differ",
            expected.pixels.len()
        );
        assert!(dispatches.is_empty() || dispatches[0].kernel == KERNEL);
    }

    #[test]
    fn four_modes_with_mask_endpoints_and_degree_gates() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (32u32, 24u32);
        let pixels = (width * height) as usize;
        let source = canvas(width, height, 7);
        let material = canvas(8, 8, 11);
        let mask_uv = uv(pixels, 13, 1.0);
        let material_uv = uv(pixels, 17, 1.0);
        for mode in [BlendMode::Normal, BlendMode::Multiply, BlendMode::SoftLight, BlendMode::PinLight] {
            for mask_value in [0u8, 255] {
                for degree in [0.0f32, 1.0] {
                    let mask = Image::solid(16, 16, [mask_value; 4]);
                    parity(
                        &gpu,
                        &source,
                        &material,
                        &mask,
                        &mask_uv,
                        &material_uv,
                        params(mode, MaskChannel::Rgba, degree, false, 255),
                        "modes",
                    );
                }
            }
        }
    }

    #[test]
    fn blend_alpha_and_out_of_range_uv_match() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (17u32, 13u32);
        let pixels = (width * height) as usize;
        let source = canvas(width, height, 19);
        let material = canvas(5, 9, 23);
        let mask = canvas(11, 3, 29);
        // Deliberately outside [0, 1]: the CPU clamps before scaling.
        let mask_uv: Vec<[f32; 2]> = uv(pixels, 31, 2.4)
            .into_iter()
            .map(|point| [point[0] - 1.2, point[1] - 1.2])
            .collect();
        let material_uv: Vec<[f32; 2]> = uv(pixels, 37, 3.0)
            .into_iter()
            .map(|point| [point[0] - 1.5, point[1] - 1.5])
            .collect();
        for blend_alpha in [false, true] {
            for alpha in [0u8, 128, 255] {
                parity(
                    &gpu,
                    &source,
                    &material,
                    &mask,
                    &mask_uv,
                    &material_uv,
                    params(BlendMode::SoftLight, MaskChannel::Rgba, 1.0, blend_alpha, alpha),
                    "blend alpha",
                );
            }
        }
    }

    #[test]
    fn gate_and_refusals_match() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(8, 6, 41);
        let material = canvas(4, 4, 43);
        let mask = canvas(4, 4, 47);
        let pixels = 48usize;
        let mask_uv = uv(pixels, 53, 1.0);
        let material_uv = uv(pixels, 59, 1.0);

        // mask 0 -> factor 0 -> identity, even at full degree.
        let empty = Image::solid(4, 4, [0, 0, 0, 0]);
        let (actual, _) = gpu
            .apply_image_layer_dual(
                &source,
                &material,
                &empty,
                &mask_uv,
                &material_uv,
                params(BlendMode::Normal, MaskChannel::Rgba, 1.0, false, 255),
            )
            .unwrap();
        assert_eq!(actual, source);

        // degree 0 -> mix_alpha 0 -> no launch.
        let (actual, dispatches) = gpu
            .apply_image_layer_dual(
                &source,
                &material,
                &mask,
                &mask_uv,
                &material_uv,
                params(BlendMode::Normal, MaskChannel::Rgba, 0.0, false, 255),
            )
            .unwrap();
        assert!(dispatches.is_empty());
        assert_eq!(actual, source);

        // Unsupported blend mode refuses before any device work.
        assert_eq!(
            gpu.apply_image_layer_dual(
                &source,
                &material,
                &mask,
                &mask_uv,
                &material_uv,
                params(BlendMode::Screen, MaskChannel::Rgba, 1.0, false, 255),
            )
            .err(),
            Some(CudaError::Refused(Refusal::NotImplemented(0x1_0000 + 1)))
        );
        // UV length mismatch.
        assert_eq!(
            gpu.apply_image_layer_dual(
                &source,
                &material,
                &mask,
                &mask_uv[..pixels - 1],
                &material_uv,
                params(BlendMode::Normal, MaskChannel::Rgba, 1.0, false, 255),
            )
            .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("uv")))
        );
    }
}
