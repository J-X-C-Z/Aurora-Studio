#[cfg(feature = "external-fixtures")]
#[path = "support/mod.rs"]
mod external_fixtures;

// Real NVIDIA coverage for PF90005/PF90007 SkinAdjustV2 (`skin_adjust_lut`).
//
// The GPU stage must produce byte-identical pixels to the CPU oracle for the
// same buffers. LUTs here are test-supplied synthetic atlases; no vendor
// material is fabricated or claimed.
//
// No driver or no device prints `SKIP_NO_NVIDIA_DRIVER` and returns; set
// `PIXCAKE_CUDA_REQUIRED=1` to make that a failure. A present-but-broken
// driver panics.

#[cfg(windows)]
mod skin_lut_gpu {

    use aurora_pixcake::cuda::CudaError;
    use aurora_pixcake::cuda_lut::{SkinLut, GATE, KERNEL, LUT_BYTES};
    use aurora_pixcake::kernels::makeup;
    #[cfg(feature = "external-fixtures")]
    use aurora_pixcake::kernels::face_detail;
    use aurora_pixcake::{Image, Refusal};

    #[derive(Clone)]
    struct Rng(u64);

    impl Rng {
        fn new(seed: u64) -> Self {
            Self(seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1))
        }

        fn next_u32(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 32) as u32
        }

        fn byte(&mut self) -> u8 {
            (self.next_u32() >> 24) as u8
        }
    }

    fn gpu() -> Option<SkinLut> {
        match SkinLut::open() {
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

    /// Test-supplied atlas: every byte independent, so every tap is exercised.
    fn pseudo_lut(seed: u64) -> Vec<u8> {
        let mut rng = Rng::new(seed);
        (0..LUT_BYTES).map(|_| rng.byte()).collect()
    }

    /// Test-supplied atlas with structure: byte tracks the atlas x coordinate.
    fn ramp_lut(channel_gain: [u8; 3]) -> Vec<u8> {
        let mut lut = Vec::with_capacity(LUT_BYTES);
        for _y in 0..512 {
            for x in 0..512 {
                for channel in 0..3 {
                    let value = ((x as u32 * channel_gain[channel] as u32) / 511) as u8;
                    lut.push(value);
                }
            }
        }
        lut
    }

    fn job(rng: &mut Rng, width: u32, height: u32) -> (Image, Vec<u8>) {
        let pixels = (width * height) as usize;
        let source =
            Image::new(width, height, (0..pixels * 4).map(|_| rng.byte()).collect()).unwrap();
        let mask = (0..pixels).map(|_| rng.byte()).collect();
        (source, mask)
    }

    #[cfg(feature = "external-fixtures")]

    fn tint_fixture(name: &str) -> Vec<u8> {
        let bytes = super::external_fixtures::read(&format!("pf21015/{name}"));
        assert_eq!(bytes.len(), LUT_BYTES);
        bytes
    }

    fn tint_cpu(source: &Image, mask: &[u8], green: &[u8], magenta: &[u8], ui: f32) -> Image {
        let signed = ui.clamp(-100.0, 100.0) / 100.0;
        let lut = if signed <= 0.0 { green } else { magenta };
        makeup::skin_adjust_lut(source, mask, lut, signed.abs()).unwrap()
    }

    fn tint_parity(
        gpu: &SkinLut,
        source: &Image,
        mask: &[u8],
        green: &[u8],
        magenta: &[u8],
        ui: f32,
    ) -> u8 {
        let expected = tint_cpu(source, mask, green, magenta, ui);
        let (actual, dispatch) = gpu
            .tint(source, mask, green, magenta, ui)
            .expect("real GPU tint dispatch");
        assert_eq!(dispatch.kernel, KERNEL);
        let mut worst = 0u8;
        for (index, (left, right)) in expected.pixels.iter().zip(&actual.pixels).enumerate() {
            let delta = left.abs_diff(*right);
            assert_eq!(
                delta,
                0,
                "{}x{} ui {ui} byte {index} (pixel {}, channel {}): cpu {left} gpu {right}",
                source.width,
                source.height,
                index / 4,
                index % 4
            );
            worst = worst.max(delta);
        }
        worst
    }

    fn parity(gpu: &SkinLut, source: &Image, mask: &[u8], lut: &[u8], mix: f32) -> u8 {
        let expected = makeup::skin_adjust_lut(source, mask, lut, mix).unwrap();
        let (actual, dispatch) = gpu
            .skin_adjust(source, mask, lut, mix, 90005)
            .expect("real GPU dispatch");
        assert_eq!(dispatch.kernel, KERNEL);
        assert!(dispatch.compute_cap.0 >= 8, "PTX needs sm_89 or newer");
        let mut worst = 0u8;
        for (index, (left, right)) in expected.pixels.iter().zip(&actual.pixels).enumerate() {
            let delta = left.abs_diff(*right);
            assert_eq!(
                delta,
                0,
                "{}x{} mix {mix} byte {index} (pixel {}, channel {}): cpu {left} gpu {right}",
                source.width,
                source.height,
                index / 4,
                index % 4
            );
            worst = worst.max(delta);
        }
        assert_eq!(actual.width, source.width);
        assert_eq!(actual.height, source.height);
        worst
    }

    #[test]
    fn lut_stage_is_byte_identical_to_the_cpu_oracle() {
        let Some(gpu) = gpu() else { return };
        let luts = [pseudo_lut(3), ramp_lut([255, 128, 64])];
        for (width, height) in [(1, 1), (17, 13), (37, 19), (64, 64)] {
            let mut rng = Rng::new((width * 1000 + height) as u64);
            let (source, mask) = job(&mut rng, width, height);
            for lut in &luts {
                for mix in [0.001f32, 0.25, 1.0] {
                    assert_eq!(parity(&gpu, &source, &mask, lut, mix), 0);
                }
            }
            // Alpha is never computed by this stage.
            let (actual, _) = gpu
                .skin_adjust(&source, &mask, &luts[0], 0.5, 90005)
                .unwrap();
            for pixel in 0..(width * height) as usize {
                assert_eq!(actual.pixels[pixel * 4 + 3], source.pixels[pixel * 4 + 3]);
            }
        }
    }

    #[test]
    fn the_gate_is_strict_on_both_sides() {
        let Some(gpu) = gpu() else { return };
        let lut = pseudo_lut(9);
        let (source, _) = job(&mut Rng::new(11), 8, 8);

        // mask 255 and mix == gate land exactly on the boundary: copy, no blend.
        let at_gate = vec![255u8; 64];
        let (actual, _) = gpu
            .skin_adjust(&source, &at_gate, &lut, GATE, 90007)
            .unwrap();
        assert_eq!(actual, source, "weight == gate must copy the source pixel");

        let below = vec![255u8; 64];
        let (actual, _) = gpu
            .skin_adjust(&source, &below, &lut, 0.0009, 90007)
            .unwrap();
        assert_eq!(actual, source);

        // Just above the gate the stage must actually move pixels.
        let above = vec![255u8; 64];
        let (actual, _) = gpu.skin_adjust(&source, &above, &lut, 0.01, 90007).unwrap();
        assert_ne!(actual, source);
        assert_eq!(parity(&gpu, &source, &above, &lut, 0.01), 0);
    }

    #[test]
    fn off_axis_channels_and_odd_geometry_stay_aligned() {
        let Some(gpu) = gpu() else { return };
        // 1xN and Nx1 catch row/column swap; 37x19 catches partial blocks.
        let lut = pseudo_lut(21);
        for (width, height) in [(1, 9), (9, 1), (37, 19)] {
            let (source, mask) = job(&mut Rng::new((width * 31 + height) as u64), width, height);
            assert_eq!(parity(&gpu, &source, &mask, &lut, 1.0), 0);
        }
    }

    #[test]
    fn bad_material_is_refused_before_launch() {
        let Some(gpu) = gpu() else { return };
        let (source, mask) = job(&mut Rng::new(5), 4, 4);
        let lut = pseudo_lut(7);
        let short_lut = &lut[..lut.len() - 3];
        assert_eq!(
            gpu.skin_adjust(&source, &mask, short_lut, 0.5, 90005).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("lut_rgb512")))
        );
        assert_eq!(
            gpu.skin_adjust(&source, &mask[..15], &lut, 0.5, 90005)
                .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("mask_r")))
        );
        assert_eq!(
            gpu.skin_adjust(&source, &mask, &lut, f32::NAN, 90005).err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("mix")))
        );
        assert_eq!(
            gpu.skin_adjust(&source, &mask, &lut, 1.5, 90005).err(),
            Some(CudaError::Refused(Refusal::ValueOutOfRange("mix")))
        );
        assert_eq!(
            gpu.skin_adjust(&source, &mask, &lut, 0.5, 90006).err(),
            Some(CudaError::Refused(Refusal::UnknownEffect(90006)))
        );
    }

    #[test]
    #[cfg(feature = "external-fixtures")]
    fn tint_green_magenta_and_full_limits_match_the_fixed_oracle() {
        let Some(gpu) = gpu() else { return };
        let green = tint_fixture("TintGreen.rgb8");
        let magenta = tint_fixture("TintMagenta.rgb8");
        let source = Image::new(1, 1, vec![27, 153, 110, 226]).unwrap();
        let mask = [255u8];
        for ui in [-100.0f32, 100.0] {
            assert_eq!(tint_parity(&gpu, &source, &mask, &green, &magenta, ui), 0);
        }
        let (green_out, dispatch) = gpu.tint(&source, &mask, &green, &magenta, -100.0).unwrap();
        assert_eq!(dispatch.kernel, KERNEL);
        let (magenta_out, _) = gpu.tint(&source, &mask, &green, &magenta, 100.0).unwrap();
        assert_eq!(green_out.pixels, vec![0, 171, 105, 226]);
        assert_eq!(magenta_out.pixels, vec![65, 149, 130, 226]);
        assert_ne!(green_out.pixels, magenta_out.pixels);
    }

    #[test]
    #[cfg(feature = "external-fixtures")]
    fn tint_zero_and_signed_limits_cover_geometry_and_alpha() {
        let Some(gpu) = gpu() else { return };
        let green = tint_fixture("TintGreen.rgb8");
        let magenta = tint_fixture("TintMagenta.rgb8");
        for (width, height) in [(1, 9), (9, 1), (37, 19)] {
            let (source, mask) = job(&mut Rng::new((width * 41 + height) as u64), width, height);
            assert_eq!(tint_parity(&gpu, &source, &mask, &green, &magenta, 0.0), 0);
            let (zero, _) = gpu.tint(&source, &mask, &green, &magenta, 0.0).unwrap();
            assert_eq!(zero, source, "ui 0 must be the identity");
            for ui in [-100.0f32, 100.0] {
                assert_eq!(tint_parity(&gpu, &source, &mask, &green, &magenta, ui), 0);
                let (out, _) = gpu.tint(&source, &mask, &green, &magenta, ui).unwrap();
                for pixel in 0..(width * height) as usize {
                    assert_eq!(out.pixels[pixel * 4 + 3], source.pixels[pixel * 4 + 3]);
                }
            }
        }
    }

    #[test]
    fn tint_gate_is_strict_at_0_001_and_moves_at_0_0022() {
        let Some(gpu) = gpu() else { return };
        let flat = vec![255u8; LUT_BYTES];
        let source = Image::new(1, 1, vec![0, 0, 0, 200]).unwrap();
        let mask = [255u8];
        for ui in [-0.1f32, 0.1] {
            assert_eq!(tint_parity(&gpu, &source, &mask, &flat, &flat, ui), 0);
            let (at_gate, _) = gpu.tint(&source, &mask, &flat, &flat, ui).unwrap();
            assert_eq!(at_gate, source, "|ui|/100 == gate must copy");
        }
        for ui in [-0.22f32, 0.22] {
            assert_eq!(tint_parity(&gpu, &source, &mask, &flat, &flat, ui), 0);
            let (above, _) = gpu.tint(&source, &mask, &flat, &flat, ui).unwrap();
            assert_eq!(above.pixels, vec![1, 1, 1, 200]);
        }
    }

    #[test]
    #[cfg(feature = "external-fixtures")]
    fn tint_refuses_bad_ui_and_bad_atlases() {
        let Some(gpu) = gpu() else { return };
        let green = tint_fixture("TintGreen.rgb8");
        let magenta = tint_fixture("TintMagenta.rgb8");
        let (source, mask) = job(&mut Rng::new(73), 4, 4);
        assert_eq!(
            gpu.tint(&source, &mask, &green[..green.len() - 3], &magenta, 0.0)
                .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("tint_green")))
        );
        assert_eq!(
            gpu.tint(&source, &mask, &green, &magenta[..magenta.len() - 3], 0.0)
                .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("tint_magenta")))
        );
        assert_eq!(
            gpu.tint(&source, &mask[..15], &green, &magenta, 0.0).err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("mask_r")))
        );
        for ui in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                gpu.tint(&source, &mask, &green, &magenta, ui).err(),
                Some(CudaError::Refused(Refusal::ValueOutOfRange("ui")))
            );
        }
    }

    #[test]
    #[cfg(feature = "external-fixtures")]
    fn tint_clamps_ui_outside_the_ui_range() {
        let Some(gpu) = gpu() else { return };
        let green = tint_fixture("TintGreen.rgb8");
        let magenta = tint_fixture("TintMagenta.rgb8");
        let source = Image::new(1, 1, vec![27, 153, 110, 226]).unwrap();
        let mask = [255u8];
        let (full_positive, _) = gpu.tint(&source, &mask, &green, &magenta, 100.0).unwrap();
        let (over_positive, _) = gpu.tint(&source, &mask, &green, &magenta, 101.0).unwrap();
        let (full_negative, _) = gpu.tint(&source, &mask, &green, &magenta, -100.0).unwrap();
        let (over_negative, _) = gpu.tint(&source, &mask, &green, &magenta, -101.0).unwrap();
        assert_eq!(full_positive, over_positive);
        assert_eq!(full_negative, over_negative);
        assert_eq!(
            tint_parity(&gpu, &source, &mask, &green, &magenta, 101.0),
            0
        );
        assert_eq!(
            tint_parity(&gpu, &source, &mask, &green, &magenta, -101.0),
            0
        );
    }

    #[test]
    #[cfg(feature = "external-fixtures")]
    fn tint_fixed_oracle_is_not_skin_tint() {
        let Some(gpu) = gpu() else { return };
        let green = tint_fixture("TintGreen.rgb8");
        let magenta = tint_fixture("TintMagenta.rgb8");
        let source = Image::new(1, 1, vec![0, 17, 153, 200]).unwrap();
        let mask = [255u8];
        let (fixed, _) = gpu.tint(&source, &mask, &green, &magenta, -100.0).unwrap();
        let float = face_detail::skin_tint(&source, &mask, &green, &magenta, -100.0).unwrap();
        assert_eq!(fixed.pixels, vec![0, 32, 150, 200]);
        assert_eq!(float.pixels, vec![0, 33, 150, 200]);
        assert_ne!(
            fixed.pixels, float.pixels,
            "fixed-point CUDA and skin_tint are different oracles"
        );
    }
}
