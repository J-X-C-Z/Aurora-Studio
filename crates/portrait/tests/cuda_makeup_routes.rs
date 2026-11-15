#[cfg(windows)]
mod makeup_routes {
    use aurora_portrait::cuda::CudaError;
    use aurora_portrait::cuda_makeup::{MakeupLayer, MakeupRoute, RouteInputs, KERNEL, ROUTED_PFS};
    use aurora_portrait::kernels::makeup_material::{
        self, MakeupLayer as CpuMakeupLayer, MaskChannel,
    };
    use aurora_portrait::{Image, Refusal};

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
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
                    "CUDA required but unavailable: {reason}"
                );
                None
            }
            Err(broken) => panic!("NVIDIA path present but broken: {broken:?}"),
        }
    }

    fn canvas(width: u32, height: u32, seed: u64) -> Image {
        let mut rng = Rng::new(seed);
        Image::new(
            width,
            height,
            (0..width * height * 4).map(|_| rng.byte()).collect(),
        )
        .unwrap()
    }

    fn uv(count: usize, seed: u64) -> Vec<[f32; 2]> {
        let mut rng = Rng::new(seed);
        (0..count).map(|_| [rng.unit(), rng.unit()]).collect()
    }

    fn expected_route(pf: u32) -> MakeupRoute {
        match pf {
            100 | 117 => MakeupRoute::Color {
                pf,
                filter_type: 11,
                mask_channel: MaskChannel::RB,
            },
            103 | 30007 => MakeupRoute::Color {
                pf,
                filter_type: 11,
                mask_channel: MaskChannel::RG,
            },
            105 | 118 => MakeupRoute::Color {
                pf,
                filter_type: 11,
                mask_channel: MaskChannel::RG,
            },
            111 => MakeupRoute::ImageUnmasked {
                pf,
                filter_type: 40,
            },
            _ => panic!("unexpected PF{pf}"),
        }
    }

    fn cpu_layer(route: MakeupRoute) -> CpuMakeupLayer {
        CpuMakeupLayer {
            mix_color: [37, 151, 223, 219],
            opacity_percent: 73.0,
            degree01: 0.81,
            filter_type: route.filter_type(),
            mask_channel: route.mask_channel().unwrap_or(MaskChannel::A),
        }
    }

    #[test]
    fn every_route_matches_cpu_with_exact_filter_channel_and_alpha() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (23u32, 17u32);
        let pixels = (width * height) as usize;
        let source = canvas(width, height, 101);
        let material = canvas(9, 7, 103);
        let mask = canvas(8, 6, 107);
        let mask_uv = uv(pixels, 109);
        let material_uv = uv(pixels, 113);

        for pf in ROUTED_PFS {
            let route = expected_route(pf);
            assert_eq!(MakeupRoute::resolve(pf), Ok(route));
            let inputs = RouteInputs {
                mix_color: [37, 151, 223, 219],
                opacity_percent: 73.0,
                degree01: 0.81,
                filter_type: route.filter_type(),
                material: matches!(route, MakeupRoute::ImageUnmasked { .. }).then_some(&material),
                material_uv: matches!(route, MakeupRoute::ImageUnmasked { .. })
                    .then_some(material_uv.as_slice()),
                mask: matches!(route, MakeupRoute::Color { .. }).then_some(&mask),
                mask_uv: matches!(route, MakeupRoute::Color { .. }).then_some(mask_uv.as_slice()),
            };
            let expected = match route {
                MakeupRoute::Color { .. } => makeup_material::color_makeup(
                    &source,
                    &mask,
                    &mask_uv,
                    cpu_layer(route),
                )
                .unwrap(),
                MakeupRoute::ImageUnmasked { .. } => makeup_material::image_makeup_unmasked(
                    &source,
                    &material,
                    &material_uv,
                    cpu_layer(route),
                )
                .unwrap(),
            };
            let (actual, dispatches) = gpu.apply_route(pf, &source, inputs).unwrap();
            let differing = expected
                .pixels
                .iter()
                .zip(&actual.pixels)
                .filter(|(a, b)| a != b)
                .count();
            assert_eq!(
                differing, 0,
                "PF{pf}: {differing} of {} bytes differ",
                expected.pixels.len()
            );
            assert_eq!(actual.pixels.len(), source.pixels.len());
            assert_eq!(actual.pixels.len() % 4, 0);
            for (expected_pixel, actual_pixel) in expected
                .pixels
                .chunks_exact(4)
                .zip(actual.pixels.chunks_exact(4))
            {
                assert_eq!(actual_pixel[3], expected_pixel[3], "PF{pf} alpha");
            }
            let dispatch = dispatches.first().expect("nonzero makeup route dispatch");
            assert_eq!(dispatch.kernel, KERNEL);
            eprintln!(
                "PF{pf} route={route:?} kernel={} device={} cc={:?} ptx={:?}",
                dispatch.kernel, dispatch.device, dispatch.compute_cap, dispatch.ptx_isa
            );
        }
    }

    #[test]
    fn filter_type_opacity_degree_and_missing_inputs_refuse_typed() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(5, 4, 127);
        let material = canvas(3, 2, 131);
        let mask = canvas(4, 3, 137);
        let uv = vec![[0.5, 0.5]; source.pixel_count()];
        let base = RouteInputs {
            mix_color: [1, 2, 3, 255],
            opacity_percent: 100.0,
            degree01: 1.0,
            filter_type: 11,
            material: None,
            material_uv: None,
            mask: Some(&mask),
            mask_uv: Some(&uv),
        };

        for pf in ROUTED_PFS {
            let route = expected_route(pf);
            let mut wrong = base;
            wrong.filter_type = if route.filter_type() == 11 { 40 } else { 11 };
            let mut prepared = wrong;
            if matches!(route, MakeupRoute::ImageUnmasked { .. }) {
                prepared.material = Some(&material);
                prepared.material_uv = Some(&uv);
                prepared.mask = None;
                prepared.mask_uv = None;
            }
            assert_eq!(
                gpu.apply_route(pf, &source, prepared).unwrap_err(),
                CudaError::Refused(Refusal::ValueOutOfRange("makeup FilterType"))
            );

            for opacity in [-0.1, 100.1, f32::NAN] {
                let mut inputs = base;
                inputs.filter_type = route.filter_type();
                inputs.opacity_percent = opacity;
                if matches!(route, MakeupRoute::ImageUnmasked { .. }) {
                    inputs.material = Some(&material);
                    inputs.material_uv = Some(&uv);
                    inputs.mask = None;
                    inputs.mask_uv = None;
                }
                assert_eq!(
                    gpu.apply_route(pf, &source, inputs).unwrap_err(),
                    CudaError::Refused(Refusal::ValueOutOfRange("opacity_percent"))
                );
            }

            for degree in [-0.1, 1.1, f32::NAN] {
                let mut inputs = base;
                inputs.filter_type = route.filter_type();
                inputs.degree01 = degree;
                if matches!(route, MakeupRoute::ImageUnmasked { .. }) {
                    inputs.material = Some(&material);
                    inputs.material_uv = Some(&uv);
                    inputs.mask = None;
                    inputs.mask_uv = None;
                }
                assert_eq!(
                    gpu.apply_route(pf, &source, inputs).unwrap_err(),
                    CudaError::Refused(Refusal::ValueOutOfRange("degree01"))
                );
            }
        }

        let mut missing_mask = base;
        missing_mask.mask = None;
        assert_eq!(
            gpu.apply_route(100, &source, missing_mask).unwrap_err(),
            CudaError::Refused(Refusal::MissingInput("makeup mask"))
        );

        let mut missing_mask_uv = base;
        missing_mask_uv.mask_uv = None;
        assert_eq!(
            gpu.apply_route(103, &source, missing_mask_uv).unwrap_err(),
            CudaError::Refused(Refusal::MissingInput("makeup mask UV"))
        );

        let unmasked = RouteInputs {
            filter_type: 40,
            material: None,
            material_uv: None,
            mask: None,
            mask_uv: None,
            ..base
        };
        assert_eq!(
            gpu.apply_route(111, &source, unmasked).unwrap_err(),
            CudaError::Refused(Refusal::MissingInput("makeup material"))
        );

        assert_eq!(
            gpu.apply_route(
                111,
                &source,
                RouteInputs {
                    material: Some(&material),
                    material_uv: None,
                    ..unmasked
                },
            )
            .unwrap_err(),
            CudaError::Refused(Refusal::MissingInput("makeup material UV"))
        );

        assert_eq!(
            gpu.apply_route(
                111,
                &source,
                RouteInputs {
                    material: Some(&material),
                    material_uv: Some(&uv),
                    mask: Some(&mask),
                    ..unmasked
                },
            )
            .unwrap_err(),
            CudaError::Refused(Refusal::ValueOutOfRange("PF111 mask must be absent"))
        );
    }
}
