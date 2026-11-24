// Real NVIDIA coverage for PF90005/PF90007 SkinAdjustV2 (`skin_adjust_lut`).
//
// The GPU stage must produce byte-identical pixels to the CPU oracle for the
// same buffers. LUTs here are test-supplied synthetic atlases; no vendor
// material is fabricated or claimed.
//
// No driver or no device prints `SKIP_NO_NVIDIA_DRIVER` and returns; set
// `PORTRAIT_CUDA_REQUIRED=1` to make that a failure. A present-but-broken
// driver panics.

#[cfg(windows)]
mod skin_lut_gpu {
    use std::path::Path;

    use aurora_portrait::cuda::{driver_metrics, CudaError};
    use aurora_portrait::cuda_lut::{GATE, KERNEL, LUT_BYTES, SkinLut, TEMPERATURE_KERNEL};
    use aurora_portrait::kernels::{face_detail, makeup};
    use aurora_portrait::{Image, Refusal};

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
                    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_none(),
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

    fn tint_fixture(name: &str) -> Vec<u8> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/pf21015")
            .join(name);
        let bytes =
            std::fs::read(&path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
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

    fn temperature_parity(
        gpu: &SkinLut,
        source: &Image,
        mask: &[u8],
        cold: &[u8],
        warm: &[u8],
        ui: f32,
    ) -> u8 {
        let expected = face_detail::skin_temperature(source, mask, cold, warm, ui).unwrap();
        let (actual, dispatch) = gpu
            .temperature(source, mask, cold, warm, ui)
            .expect("real GPU temperature dispatch");
        assert_eq!(dispatch.kernel, TEMPERATURE_KERNEL);
        assert!(dispatch.compute_cap.0 >= 8, "PTX needs sm_89 or newer");
        assert_eq!(actual.width, source.width);
        assert_eq!(actual.height, source.height);
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
        for pixel in 0..(source.width * source.height) as usize {
            assert_eq!(
                actual.pixels[pixel * 4 + 3],
                source.pixels[pixel * 4 + 3],
                "{}x{} ui {ui} alpha pixel {pixel}",
                source.width,
                source.height
            );
        }
        worst
    }

    fn percentile(samples: &[f64], fraction: f64) -> f64 {
        let mut sorted = samples.to_vec();
        sorted.sort_by(f64::total_cmp);
        sorted[((sorted.len() - 1) as f64 * fraction).round() as usize]
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

    #[test]
    fn temperature_is_byte_identical_on_odd_geometry() {
        let Some(gpu) = gpu() else { return };
        let cold = pseudo_lut(101);
        let warm = pseudo_lut(202);
        for (width, height) in [(1, 1), (17, 13), (37, 19)] {
            let (source, mask) = job(&mut Rng::new((width * 1009 + height) as u64), width, height);
            for ui in [-100.0f32, -37.5, 0.0, 43.25, 100.0] {
                assert_eq!(
                    temperature_parity(&gpu, &source, &mask, &cold, &warm, ui),
                    0
                );
            }
        }
    }

    #[test]
    fn temperature_roi_and_alpha_are_preserved() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (23u32, 11u32);
        let (source, _) = job(&mut Rng::new(303), width, height);
        let mut mask = vec![0u8; (width * height) as usize];
        let mut rng = Rng::new(404);
        for y in 2..9 {
            for x in 3..19 {
                mask[(y * width + x) as usize] = rng.byte().max(1);
            }
        }
        let cold = pseudo_lut(505);
        let warm = pseudo_lut(606);
        for ui in [-73.0f32, 67.0] {
            let expected = face_detail::skin_temperature(&source, &mask, &cold, &warm, ui).unwrap();
            assert_eq!(
                temperature_parity(&gpu, &source, &mask, &cold, &warm, ui),
                0
            );
            for y in 0..height {
                for x in 0..width {
                    if mask[(y * width + x) as usize] == 0 {
                        let offset = ((y * width + x) * 4) as usize;
                        assert_eq!(
                            expected.pixels[offset..offset + 4],
                            source.pixels[offset..offset + 4],
                            "ui {ui} changed outside the mask at {x},{y}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn temperature_uses_the_selected_lut_for_each_direction() {
        let Some(gpu) = gpu() else { return };
        let cold = ramp_lut([0, 255, 0]);
        let warm = ramp_lut([255, 0, 128]);
        let source = Image::new(1, 1, vec![27, 153, 110, 226]).unwrap();
        let mask = [255u8];
        assert_eq!(
            temperature_parity(&gpu, &source, &mask, &cold, &warm, -100.0),
            0
        );
        assert_eq!(
            temperature_parity(&gpu, &source, &mask, &cold, &warm, 100.0),
            0
        );
        let (cold_out, _) = gpu
            .temperature(&source, &mask, &cold, &warm, -100.0)
            .unwrap();
        let (warm_out, _) = gpu
            .temperature(&source, &mask, &cold, &warm, 100.0)
            .unwrap();
        assert_ne!(cold_out.pixels, warm_out.pixels);
    }

    #[test]
    fn temperature_cold_warm_luts_are_reused_across_domain_handles() {
        let before_open = driver_metrics();
        let Some(first) = gpu() else { return };
        let cold = pseudo_lut(9_151);
        let warm = pseudo_lut(9_152);
        let (source, mask) = job(&mut Rng::new(9_153), 1, 1);

        let before_cold_warm = driver_metrics();
        first.temperature(&source, &mask, &cold, &warm, -100.0)
            .expect("cold LUT dispatch");
        first.temperature(&source, &mask, &cold, &warm, 100.0)
            .expect("warm LUT dispatch");
        let after_cold_warm = driver_metrics();
        assert_eq!(
            after_cold_warm.skin_lut_uploads - before_cold_warm.skin_lut_uploads,
            2,
            "the first cold/warm pair must upload exactly two atlases"
        );

        let before_reuse = driver_metrics();
        for index in 0..16 {
            let ui = if index % 2 == 0 { -73.0 } else { 67.0 };
            first.temperature(&source, &mask, &cold, &warm, ui)
                .expect("cached LUT dispatch");
        }
        let after_reuse = driver_metrics();
        assert_eq!(
            after_reuse.skin_lut_uploads - before_reuse.skin_lut_uploads,
            0,
            "warm calls must not re-upload either atlas"
        );
        assert_eq!(
            after_reuse.device_allocations - before_reuse.device_allocations,
            0,
            "warm calls must reuse the task buffers"
        );
        assert_eq!(
            after_reuse.context_syncs - before_reuse.context_syncs,
            0,
            "warm calls must not add a context synchronization"
        );

        drop(first);
        let Some(reopened) = gpu() else { return };
        let before_reopen = driver_metrics();
        reopened
            .temperature(&source, &mask, &cold, &warm, -73.0)
            .expect("cold atlases after handle reopen");
        reopened
            .temperature(&source, &mask, &cold, &warm, 67.0)
            .expect("warm atlases after handle reopen");
        let after_reopen = driver_metrics();
        assert_eq!(
            after_reopen.skin_lut_uploads - before_reopen.skin_lut_uploads,
            0,
            "a reopened handle must see the process-domain atlas cache"
        );
        assert_eq!(
            after_reopen.device_allocations - before_reopen.device_allocations,
            0,
            "a reopened handle must reuse the task buffers"
        );
        assert_eq!(
            after_reopen.context_syncs - before_reopen.context_syncs,
            0,
            "a reopened handle must not add a context synchronization"
        );
        assert_eq!(
            after_reopen.ptx_module_load_calls,
            after_reuse.ptx_module_load_calls,
            "the module must not reload on a warm handle reopen"
        );
        eprintln!(
            "PF30096_CACHE module_loads={} first_pair_uploads={} warm_uploads={} warm_allocations={} warm_syncs={} reopen_uploads={} reopen_allocations={} reopen_syncs={}",
            after_reopen.ptx_module_load_calls - before_open.ptx_module_load_calls,
            after_cold_warm.skin_lut_uploads - before_cold_warm.skin_lut_uploads,
            after_reuse.skin_lut_uploads - before_reuse.skin_lut_uploads,
            after_reuse.device_allocations - before_reuse.device_allocations,
            after_reuse.context_syncs - before_reuse.context_syncs,
            after_reopen.skin_lut_uploads - before_reopen.skin_lut_uploads,
            after_reopen.device_allocations - before_reopen.device_allocations,
            after_reopen.context_syncs - before_reopen.context_syncs,
        );
    }

    #[test]
    fn whiten_rosy_chain_matches_sequential_and_halves_full_frame_traffic() {
        let Some(gpu) = gpu() else { return };
        let whiten = pseudo_lut(9_161);
        let ruddy = pseudo_lut(9_162);
        let (source, mask) = job(&mut Rng::new(9_163), 257, 193);

        gpu.skin_adjust(&source, &mask, &whiten, 0.63, 90005)
            .expect("prime whiten LUT");
        gpu.skin_adjust(&source, &mask, &ruddy, 0.41, 90007)
            .expect("prime rosy LUT");

        let before_sequential = driver_metrics();
        let (first, first_dispatch) = gpu
            .skin_adjust(&source, &mask, &whiten, 0.63, 90005)
            .expect("whiten dispatch");
        let (expected, second_dispatch) = gpu
            .skin_adjust(&first, &mask, &ruddy, 0.41, 90007)
            .expect("rosy dispatch");
        let after_sequential = driver_metrics();
        assert_eq!(first_dispatch.kernel, KERNEL);
        assert_eq!(second_dispatch.kernel, KERNEL);

        let before_chain = driver_metrics();
        let (actual, dispatches) = gpu
            .skin_adjust_chain(&source, &mask, &whiten, 0.63, &ruddy, 0.41)
            .expect("whiten/rosy chain");
        let after_chain = driver_metrics();

        let pixels = source.pixel_count();
        let sequential_h2d =
            after_sequential.host_to_device_bytes - before_sequential.host_to_device_bytes;
        let sequential_d2h =
            after_sequential.device_to_host_bytes - before_sequential.device_to_host_bytes;
        let chain_h2d = after_chain.host_to_device_bytes - before_chain.host_to_device_bytes;
        let chain_d2h = after_chain.device_to_host_bytes - before_chain.device_to_host_bytes;

        assert_eq!(actual, expected, "chained kernels must preserve byte parity");
        assert_eq!(dispatches.len(), 2);
        assert!(dispatches.iter().all(|dispatch| dispatch.kernel == KERNEL));
        assert_eq!(sequential_h2d, pixels * 10);
        assert_eq!(sequential_d2h, pixels * 8);
        assert_eq!(chain_h2d, pixels * 5);
        assert_eq!(chain_d2h, pixels * 4);
        assert_eq!(chain_h2d * 2, sequential_h2d);
        assert_eq!(chain_d2h * 2, sequential_d2h);
    }

    #[test]
    #[ignore = "24 MP PF30096 parity and warm timing gate"]
    fn temperature_24mp_parity_and_warm_timing() {
        let Some(gpu) = gpu() else { return };
        let (width, height) = (6_000u32, 4_000u32);
        let pixels = (width * height) as usize;
        let source = Image::new(
            width,
            height,
            (0..pixels * 4)
                .map(|index| ((index as u32).wrapping_mul(2_654_435_761) >> 24) as u8)
                .collect(),
        )
        .expect("24 MP frame");
        let mask = (0..pixels)
            .map(|index| if index % 7 == 0 { 0 } else { 255 })
            .collect::<Vec<_>>();
        let cold = pseudo_lut(9_154);
        let warm = pseudo_lut(9_155);
        let expected = face_detail::skin_temperature(&source, &mask, &cold, &warm, -61.0)
            .expect("CPU 24 MP oracle");
        let (actual, dispatch) = gpu
            .temperature(&source, &mask, &cold, &warm, -61.0)
            .expect("24 MP CUDA dispatch");
        assert_eq!(dispatch.kernel, TEMPERATURE_KERNEL);
        for (index, (left, right)) in expected.pixels.iter().zip(&actual.pixels).enumerate() {
            assert_eq!(
                left, right,
                "24 MP byte {index} (pixel {}, channel {}): cpu {left} gpu {right}",
                index / 4,
                index % 4
            );
        }

        let mut samples = Vec::with_capacity(8);
        for _ in 0..8 {
            let start = std::time::Instant::now();
            gpu.temperature(&source, &mask, &cold, &warm, 61.0)
                .expect("24 MP warm dispatch");
            samples.push(start.elapsed().as_secs_f64() * 1_000.0);
        }
        let metrics = driver_metrics();
        eprintln!(
            "PF30096_24MP cpu_cuda_differing_bytes=0 warm_p50_ms={:.3} warm_p95_ms={:.3} module_loads={} allocations={} syncs={}",
            percentile(&samples, 0.50),
            percentile(&samples, 0.95),
            metrics.ptx_module_load_calls,
            metrics.device_allocations,
            metrics.context_syncs,
        );
    }

    #[test]
    fn temperature_refuses_bad_inputs_before_launch() {
        let Some(gpu) = gpu() else { return };
        let (source, mask) = job(&mut Rng::new(707), 4, 4);
        let cold = pseudo_lut(808);
        let warm = pseudo_lut(909);
        assert_eq!(
            gpu.temperature(&source, &mask, &cold[..cold.len() - 3], &warm, 0.0)
                .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("cold_lut")))
        );
        assert_eq!(
            gpu.temperature(&source, &mask, &cold, &warm[..warm.len() - 3], 0.0)
                .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("warm_lut")))
        );
        assert_eq!(
            gpu.temperature(&source, &mask[..15], &cold, &warm, 0.0)
                .err(),
            Some(CudaError::Refused(Refusal::ShapeMismatch("mask_r")))
        );
        for ui in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -100.1, 100.1] {
            assert_eq!(
                gpu.temperature(&source, &mask, &cold, &warm, ui).err(),
                Some(CudaError::Refused(Refusal::ValueOutOfRange("ui")))
            );
        }
    }
}
