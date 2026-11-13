// End-to-end PF407 on the GPU: the CPU strip writes the float offset field, the
// NVIDIA type-143 kernel consumes it, and the result must equal `leg_refine::apply`.
//
// The field is filled into both `.xy` (RG) and `.zw` (BA) so either channel
// selector reaches the same offsets.

#[cfg(windows)]
mod leg_refine_gpu {
    use aurora_pixcake::cuda::CudaError;
    use aurora_pixcake::cuda_uvmap::Uvmap;
    use aurora_pixcake::kernels::leg_refine;
    use aurora_pixcake::kernels::leg_refine_uv::LegColumn;
    use aurora_pixcake::kernels::uvmap_color::Channel;
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

    fn columns(clip: [f32; 5]) -> Vec<LegColumn> {
        (0..5).map(|i| LegColumn { sample_u: i as f32 / 4.0, clip_x: clip[i] }).collect()
    }

    /// The CPU strip's signed offset, mirrored into `.xy` and `.zw`.
    fn float_field(width: u32, height: u32, columns: &[LegColumn]) -> Vec<[f32; 4]> {
        let field = leg_refine::offset_field((width, height), columns).unwrap();
        field.into_iter().map(|offset| [offset[0], offset[1], offset[0], offset[1]]).collect()
    }

    fn parity(
        gpu: &Uvmap,
        source: &Image,
        columns: &[LegColumn],
        channel: Channel,
        catmull: Option<(f32, f32)>,
        label: &str,
    ) {
        let field = float_field(source.width, source.height, columns);
        let expected = leg_refine::apply(source, columns, catmull).unwrap();
        let (actual, dispatches) = gpu.apply(source, &field, channel, catmull).expect("real GPU dispatch");
        let differing = expected.pixels.iter().zip(&actual.pixels).filter(|(a, b)| a != b).count();
        assert_eq!(
            differing,
            0,
            "{label}: {differing} of {} bytes differ",
            expected.pixels.len()
        );
        assert_eq!(dispatches.len(), 1);
    }

    #[test]
    fn end_to_end_field_matches_the_cpu_stage() {
        let Some(gpu) = gpu() else { return };
        let clip = [0.30f32, 0.35, 0.40, 0.45, 0.50];
        for (width, height) in [(1u32, 1u32), (17, 13), (8, 4), (32, 24)] {
            let source = canvas(width, height, width as u64 * 31 + height as u64);
            let columns = columns(clip);
            for channel in [Channel::Rg, Channel::Ba] {
                parity(&gpu, &source, &columns, channel, None, "bilinear");
                parity(&gpu, &source, &columns, channel, Some((width as f32, height as f32)), "catmull");
                parity(&gpu, &source, &columns, channel, Some((10.0, 6.0)), "catmull fo");
            }
        }
    }

    #[test]
    fn negative_offsets_from_the_strip_reach_the_gpu() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(17, 13, 41);
        let columns = columns([0.30f32, 0.35, 0.40, 0.45, 0.50]);
        let field = float_field(17, 13, &columns);
        let negative = field.iter().filter(|value| value[0] < 0.0 || value[1] < 0.0).count();
        assert!(negative > 0, "the strip must emit signed offsets for this fixture");
        assert_eq!(field.iter().filter(|value| value[2..] != value[..2]).count(), 0, "rg and ba must mirror");
        parity(&gpu, &source, &columns, Channel::Rg, None, "negative rg");
        parity(&gpu, &source, &columns, Channel::Ba, None, "negative ba");
        let (actual, _) = gpu
            .apply(&source, &field, Channel::Rg, None)
            .expect("real GPU dispatch");
        assert_ne!(actual, source, "signed offsets must move pixels");
    }

    #[test]
    fn malformed_strip_is_refused_before_launch() {
        let Some(gpu) = gpu() else { return };
        let source = canvas(8, 4, 43);
        let short = columns([0.30f32, 0.35, 0.40, 0.45, 0.50]);
        assert_eq!(
            leg_refine::offset_field((8, 4), &short[..4]),
            Err(Refusal::ShapeMismatch("leg columns"))
        );
        let field = vec![[0.0f32; 4]; 32];
        assert_eq!(
            gpu.apply(&source, &field[..31], Channel::Rg, None).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("uv field")))
        );
        assert_eq!(
            gpu.apply(&source, &field, Channel::Rg, Some((0.0, 4.0))).err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("catmull size")))
        );
    }
}
