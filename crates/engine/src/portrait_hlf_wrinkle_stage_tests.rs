use std::collections::BTreeMap;

use super::*;
use crate::NormRect;
use crate::wrinkle_legacy::{
    WrinkleFaceRoiContract, WrinkleLegacySkinMaskProducer, WrinkleLegacySkinMaskTexture,
    WrinkleMaskMiniImage, WrinkleMaskMiniImageProducer, WrinkleProducerInputs,
    WrinkleSameJobReceipt,
};
use aurora_portrait::cuda_hlf::{BODY_KERNEL, FACE_KERNEL, GAUSS_KERNEL, MEDIAN_KERNEL};
use aurora_portrait::cuda_wrinkle::{KERNEL as WRINKLE_KERNEL, reference as wrinkle_reference};
use aurora_portrait::kernels::skin_adjust::{
    AdjustLut, BodyHlfUniforms, Roi, StoreMode, WrinkleUniforms, body_hlf_apply, hlf_face_apply,
};

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

fn image(width: u32, height: u32, seed: u64) -> Image {
    let mut rng = Rng::new(seed);
    Image::new(
        width,
        height,
        (0..width * height * 4).map(|_| rng.byte()).collect(),
    )
    .unwrap()
}

fn masks(width: u32, height: u32, seed: u64) -> PortraitMasks {
    let mut rng = Rng::new(seed);
    let pixels = width as usize * height as usize;
    PortraitMasks {
        skin: Some((0..pixels).map(|_| rng.unit()).collect()),
        hair: Some((0..pixels).map(|_| rng.unit()).collect()),
        face: Some((0..pixels).map(|_| rng.unit()).collect()),
        body: Some((0..pixels).map(|_| rng.unit()).collect()),
        width: width as usize,
        height: height as usize,
    }
}

fn extra(pf: u32, degree: f32) -> BTreeMap<String, f32> {
    BTreeMap::from([(format!("pc:{pf}"), degree)])
}

fn active_wrinkle_uniforms() -> WrinkleUniforms {
    WrinkleUniforms {
        legal_alpha: 1.0,
        dark_circle_alpha: 1.0,
        diff_factor: 1.0,
        diff_factor2: 0.0,
        color_threshold_factor: 0.0,
        mix_alpha: 1.0,
        diff_min_threshold: -1.0,
        diff_max_threshold: 1.0,
        high_freq_min_threshold: -1.0,
        high_freq_max_threshold: 1.0,
    }
}

fn body_uniforms() -> BodyHlfUniforms {
    BodyHlfUniforms {
        spotless_degree: 0.0,
        retouch_degree: 1.0,
        spotless_acne_threshold: 0.0,
        spotless_high_freq_factor: 1.0,
        spotless_low_freq_alpha: 0.0,
        retouch_high_freq_factor: 0.0,
        diff_min_range: -0.2,
        diff_max_range: 0.2,
    }
}

fn spotless_body_uniforms() -> BodyHlfUniforms {
    BodyHlfUniforms {
        spotless_degree: 0.61,
        spotless_acne_threshold: 0.02,
        spotless_high_freq_factor: 0.8,
        spotless_low_freq_alpha: 0.3,
        ..body_uniforms()
    }
}

fn vendor_adjust_lut() -> &'static [u8] {
    aurora_portrait::private_asset!("fixtures/portrait/pf30022/vendor_hlfreq_adjust_lut_256x1_l8.bin").unwrap()
}

fn hlf_face_image(current: &Image, rgb: &[f32]) -> Image {
    let mut pixels = current.pixels.clone();
    for (index, rgb) in rgb.chunks_exact(3).enumerate() {
        for channel in 0..3 {
            pixels[index * 4 + channel] = StoreMode::Round.quantize(rgb[channel]);
        }
    }
    Image::new(current.width, current.height, pixels).unwrap()
}

fn required_cuda() -> bool {
    std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_some()
}

fn assert_real_kernel(run: &PortraitEffectRun, expected: &str) {
    match run.backend {
        PortraitBackend::Cuda => {
            let dispatch = run.dispatches.last().expect("CUDA dispatch");
            assert_eq!(dispatch.kernel, expected);
            if run.dispatches.len() > 1 {
                assert_eq!(run.dispatches[0].kernel, MEDIAN_KERNEL);
                assert_eq!(run.dispatches[1].kernel, GAUSS_KERNEL);
            }
            eprintln!(
                "PF{} device={} kernel={} ptx_isa={:?}",
                run.pf,
                dispatch.device,
                dispatch.kernel,
                dispatch.ptx_isa
            );
        }
        PortraitBackend::Cpu => {
            assert!(
                !required_cuda(),
                "PF{} did not dispatch on required CUDA",
                run.pf
            );
            assert!(run.dispatches.is_empty());
        }
        PortraitBackend::Skipped => {
            panic!(
                "PF{} skipped after validated inputs: {:?}",
                run.pf, run.skip_reason
            )
        }
        PortraitBackend::Metal => {
            panic!(
                "PF{} has no Metal route in this stage: {:?}",
                run.pf, run.metal_dispatches
            )
        }
    }
}

fn assert_auto_kernel(run: &PortraitEffectRun, expected: &str) {
    assert_real_kernel(run, expected);
    if run.backend == PortraitBackend::Cuda {
        assert_eq!(run.dispatches.len(), 3);
        assert_eq!(run.dispatches[0].kernel, MEDIAN_KERNEL);
        assert_eq!(run.dispatches[1].kernel, GAUSS_KERNEL);
        assert_eq!(run.dispatches[2].kernel, expected);
    }
}

#[cfg(windows)]
#[test]
fn hlf_preprocess_cuda_matches_cpu_bytes_on_non_square_intervals() {
    use aurora_portrait::cuda_hlf::Hlf;

    let (width, height) = (31u32, 17u32);
    let pixels = width as usize * height as usize;
    let mut rng = Rng::new(0xA009);
    let source: Vec<f32> = (0..pixels * 3).map(|_| rng.unit()).collect();
    let options = HlfAutoProducerOptions {
        median_interval: 2,
        gauss_interval: 3,
    };
    let (cpu_median, cpu_gauss) = hlf_cpu_preprocess(&source, width, height, options);
    let gpu = match Hlf::open() {
        Ok(gpu) => gpu,
        Err(error) if required_cuda() => panic!("CUDA required: {error:?}"),
        Err(_) => return,
    };
    let (cuda_median, cuda_gauss, dispatches) = gpu
        .preprocess_source_rgb(
            &source,
            width,
            height,
            options.median_interval,
            options.gauss_interval,
        )
        .expect("HLF preprocessing dispatch");
    assert_eq!(dispatches.len(), 2);
    assert_eq!(dispatches[0].kernel, MEDIAN_KERNEL);
    assert_eq!(dispatches[1].kernel, GAUSS_KERNEL);
    assert_eq!(cpu_median.len(), cuda_median.len());
    assert_eq!(cpu_gauss.len(), cuda_gauss.len());
    assert!(
        cpu_median
            .iter()
            .zip(&cuda_median)
            .all(|(cpu, cuda)| cpu.to_bits() == cuda.to_bits())
    );
    assert!(
        cpu_gauss
            .iter()
            .zip(&cuda_gauss)
            .all(|(cpu, cuda)| cpu.to_bits() == cuda.to_bits())
    );
}

#[test]
fn auto_hlf_runs_while_missing_wrinkle_bundle_stays_typed() {
    let (width, height) = (7u32, 5u32);
    let masks = masks(width, height, 1);
    let inputs = PortraitJobInputs::from_renderer_job(1, &masks);
    let mut pixels = image(width, height, 2).pixels;
    let original = pixels.clone();
    let extra = BTreeMap::from([
        ("pc:21025".to_string(), 0.5),
        ("pc:21069".to_string(), 0.5),
        ("pc:21070".to_string(), 0.5),
        ("pc:21071".to_string(), 0.5),
        ("pc:30021".to_string(), 0.5),
        ("pc:30022".to_string(), 0.5),
    ]);
    let report = apply_portrait_effects_with_job(
        &mut pixels,
        width,
        height,
        NormRect::FULL,
        &extra,
        &inputs,
    );

    assert_ne!(pixels, original);
    assert_eq!(report.effects.len(), 6);
    for run in report.effects {
        match run.pf {
            30021 => {
                assert!(run.executed);
                assert_auto_kernel(&run, FACE_KERNEL);
            }
            30022 => {
                assert!(run.executed);
                assert_auto_kernel(&run, BODY_KERNEL);
            }
            21025 | 21069 | 21070 | 21071 => {
                assert!(!run.executed);
                assert!(run.dispatches.is_empty());
                assert_eq!(
                    run.refusal,
                    Some(Refusal::MissingInput(
                        "PF21025_PF21069_PF21070_PF21071_WRINKLE_SAME_JOB_INPUTS"
                    ))
                );
            }
            _ => unreachable!(),
        }
    }
}

#[test]
fn auto_hlf_missing_mask_is_still_explicit() {
    let (width, height) = (9u32, 7u32);
    let masks = PortraitMasks::default();
    let mut pixels = image(width, height, 0xA001).pixels;
    let report = apply_portrait_effects_with_job(
        &mut pixels,
        width,
        height,
        NormRect::FULL,
        &BTreeMap::from([("pc:30021".to_string(), 0.5)]),
        &PortraitJobInputs::from_renderer_job(1, &masks),
    );
    assert_eq!(report.effects.len(), 1);
    assert!(!report.effects[0].executed);
    assert_eq!(
        report.effects[0].refusal,
        Some(Refusal::MissingInput("HLF auto face mask"))
    );
}

#[test]
fn auto_hlf_cpu_and_cuda_are_pixel_identical_on_non_square_frame() {
    let (width, height) = (23u32, 15u32);
    let masks = masks(width, height, 0xA002);
    let source = image(width, height, 0xA003);
    let extra = BTreeMap::from([
        ("pc:30021".to_string(), 0.61),
        ("pc:30022".to_string(), 0.61),
    ]);
    let inputs = PortraitJobInputs::from_renderer_job(1, &masks);
    let mut normal = source.pixels.clone();
    let normal_report = apply_portrait_effects_with_job(
        &mut normal,
        width,
        height,
        NormRect::FULL,
        &extra,
        &inputs,
    );
    #[cfg(windows)]
    let mut cpu = source.pixels.clone();
    #[cfg(windows)]
    {
        let _force = super::tests::ForceCudaFailureGuard::new();
        let cpu_report = apply_portrait_effects_with_job(
            &mut cpu,
            width,
            height,
            NormRect::FULL,
            &extra,
            &inputs,
        );
        assert_eq!(cpu_report.effects.len(), 2);
        assert_eq!(normal, cpu);
        for run in &cpu_report.effects {
            assert_eq!(run.backend, PortraitBackend::Cpu);
            assert!(run.dispatches.is_empty());
            assert!(
                run.fallback_reason
                    .as_deref()
                    .is_some_and(|reason| reason.contains("HLF preprocessing CPU fallback"))
            );
        }
    }
    #[cfg(not(windows))]
    {
        let mut cpu = source.pixels.clone();
        let cpu_report = apply_portrait_effects_with_job(
            &mut cpu,
            width,
            height,
            NormRect::FULL,
            &extra,
            &inputs,
        );
        assert_eq!(cpu_report.effects.len(), 2);
        assert_eq!(normal, cpu);
    }
    assert_eq!(normal_report.effects.len(), 2);
    assert!(normal_report.effects.iter().all(|run| run.executed));
    for run in &normal_report.effects {
        match run.pf {
            30021 => assert_auto_kernel(run, FACE_KERNEL),
            30022 => assert_auto_kernel(run, BODY_KERNEL),
            _ => unreachable!(),
        }
    }
}

#[test]
fn auto_hlf_non_square_roi_preserves_alpha_and_outside_pixels() {
    let (width, height) = (29u32, 17u32);
    let masks = masks(width, height, 0xA004);
    let source = image(width, height, 0xA005);
    let region = NormRect {
        x: 0.2,
        y: 0.25,
        width: 0.55,
        height: 0.5,
    };
    let inputs = PortraitJobInputs::from_renderer_job(1, &masks);
    let mut actual = source.pixels.clone();
    let report = apply_portrait_effects_with_job(
        &mut actual,
        width,
        height,
        region,
        &extra(30022, 0.73),
        &inputs,
    );
    assert!(report.effects[0].executed);
    assert_eq!(report.effects[0].refusal, None);
    let roi = hlf_auto_roi(region, width, height).unwrap();
    for y in 0..height {
        for x in 0..width {
            let offset = ((y * width + x) * 4) as usize;
            assert_eq!(actual[offset + 3], source.pixels[offset + 3]);
            if x < roi.x || x >= roi.x + roi.width || y < roi.y || y >= roi.y + roi.height {
                assert_eq!(
                    &actual[offset..offset + 4],
                    &source.pixels[offset..offset + 4],
                    "outside auto ROI at ({x},{y})"
                );
            }
        }
    }
}

#[test]
fn auto_hlf_repeated_calls_are_deterministic() {
    let (width, height) = (13u32, 11u32);
    let masks = masks(width, height, 0xA006);
    let source = image(width, height, 0xA007);
    let inputs = PortraitJobInputs::from_renderer_job(1, &masks);
    #[cfg(windows)]
    let _force = super::tests::ForceCudaFailureGuard::new();
    let mut first = source.pixels.clone();
    let first_report = apply_portrait_effects_with_job(
        &mut first,
        width,
        height,
        NormRect::FULL,
        &extra(30021, 0.5),
        &inputs,
    );
    let mut second = source.pixels.clone();
    let second_report = apply_portrait_effects_with_job(
        &mut second,
        width,
        height,
        NormRect::FULL,
        &extra(30021, 0.5),
        &inputs,
    );
    assert_eq!(first, second);
    assert_eq!(first_report, second_report);
}

#[test]
fn hlf_face_stage_is_full_frame_and_matches_cpu_oracle() {
    let (width, height) = (17u32, 13u32);
    let mut rng = Rng::new(0x30021);
    let current = image(width, height, 3);
    let pixels = width as usize * height as usize;
    let source: Vec<f32> = (0..pixels * 3).map(|_| 0.1 + rng.unit() * 0.8).collect();
    let median: Vec<f32> = (0..pixels * 3).map(|_| 0.1 + rng.unit() * 0.8).collect();
    let gauss: Vec<f32> = (0..pixels * 3).map(|_| 0.1 + rng.unit() * 0.8).collect();
    let face: Vec<f32> = (0..pixels).map(|_| rng.unit()).collect();
    let skin: Vec<f32> = (0..pixels).map(|_| rng.unit()).collect();
    let roi = Roi {
        x: 2,
        y: 3,
        width: 7,
        height: 5,
    };
    let expected_rgb = hlf_face_apply(
        &source, &median, &gauss, &face, &skin, width, height, roi, 73.0,
    )
    .unwrap();
    let expected = hlf_face_image(&current, &expected_rgb);

    let masks = masks(width, height, 4);
    let inputs = PortraitJobInputs {
        hlf_face: Some(HlfFaceInputs {
            width,
            height,
            source_rgb_full_f32: &source,
            median_rgb_full_f32: &median,
            gauss_rgb_full_f32: &gauss,
            face_mask_full_f32: &face,
            skin_mask_full_f32: &skin,
            roi,
        }),
        ..PortraitJobInputs::from_renderer_job(1, &masks)
    };
    let mut actual = current.pixels.clone();
    let report = apply_portrait_effects_with_job(
        &mut actual,
        width,
        height,
        NormRect::FULL,
        &extra(30021, 0.73),
        &inputs,
    );

    assert_eq!(actual, expected.pixels);
    assert!(report.effects[0].executed);
    assert_real_kernel(&report.effects[0], FACE_KERNEL);
}

#[test]
fn hlf_body_stage_preserves_roi_alpha_and_matches_cpu_oracle() {
    let (width, height) = (19u32, 17u32);
    let mut rng = Rng::new(0x30022);
    let current = image(width, height, 5);
    let pixels = width as usize * height as usize;
    let median: Vec<f32> = (0..pixels * 3).map(|_| 0.1 + rng.unit() * 0.8).collect();
    let gauss: Vec<f32> = (0..pixels * 3).map(|_| 0.1 + rng.unit() * 0.8).collect();
    let face: Vec<f32> = (0..pixels * 4).map(|_| rng.unit()).collect();
    let skin: Vec<f32> = (0..pixels).map(|_| rng.unit()).collect();
    let roi = Roi {
        x: 3,
        y: 2,
        width: 9,
        height: 7,
    };
    let uniforms = body_uniforms();
    let expected = body_hlf_apply(
        &current, roi, &median, &gauss, &face, &skin, 73.0, uniforms, None,
    )
    .unwrap();

    let masks = masks(width, height, 6);
    let inputs = PortraitJobInputs {
        hlf_body: Some(HlfBodyInputs {
            width,
            height,
            median_rgb_full_f32: &median,
            gauss_rgb_full_f32: &gauss,
            face_mask_rgba_full_f32: &face,
            skin_mask_full_f32: &skin,
            roi,
            uniforms,
            adjust_lut: None,
        }),
        ..PortraitJobInputs::from_renderer_job(1, &masks)
    };
    let mut actual = current.pixels.clone();
    let report = apply_portrait_effects_with_job(
        &mut actual,
        width,
        height,
        NormRect::FULL,
        &extra(30022, 0.73),
        &inputs,
    );

    assert_eq!(actual, expected.pixels);
    assert!(report.effects[0].executed);
    assert_real_kernel(&report.effects[0], BODY_KERNEL);
    for y in 0..height {
        for x in 0..width {
            let offset = ((y * width + x) * 4) as usize;
            assert_eq!(actual[offset + 3], current.pixels[offset + 3]);
            if x < roi.x || x >= roi.x + roi.width || y < roi.y || y >= roi.y + roi.height {
                assert_eq!(
                    &actual[offset..offset + 4],
                    &current.pixels[offset..offset + 4]
                );
            }
        }
    }
}

#[test]
#[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
fn hlf_body_lut_remains_separate_from_auto_planes() {
    let masks = masks(19, 17, 7);
    let producer = ProducerInputs {
        hlf_body_adjust_lut: Some(HlfBodyAdjustLutProducerInputs {
            width: 256,
            height: 1,
            luma: vendor_adjust_lut().to_vec(),
        }),
        ..ProducerInputs::default()
    };
    let mut inputs = PortraitJobInputs::from_renderer_job(1, &masks);
    inputs.fill_from_producer_inputs(&producer);
    assert!(inputs.hlf_body.is_none());

    let mut pixels = image(19, 17, 8).pixels;
    let report = apply_portrait_effects_with_job(
        &mut pixels,
        19,
        17,
        NormRect::FULL,
        &extra(30022, 0.73),
        &inputs,
    );
    assert_eq!(report.effects.len(), 1);
    assert!(report.effects[0].executed);
    assert_eq!(report.effects[0].refusal, None);
    assert_auto_kernel(&report.effects[0], BODY_KERNEL);
}

#[test]
#[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
fn hlf_body_stage_uses_vendor_lut_at_spotless_degree() {
    let (width, height) = (19u32, 17u32);
    let mut rng = Rng::new(0x3002201);
    let current = image(width, height, 21);
    let pixels = width as usize * height as usize;
    let median: Vec<f32> = (0..pixels * 3).map(|_| rng.unit()).collect();
    let gauss: Vec<f32> = (0..pixels * 3).map(|_| rng.unit()).collect();
    let face: Vec<f32> = (0..pixels * 4).map(|_| rng.unit()).collect();
    let skin: Vec<f32> = (0..pixels).map(|_| rng.unit()).collect();
    let roi = Roi {
        x: 3,
        y: 2,
        width: 9,
        height: 7,
    };
    let uniforms = spotless_body_uniforms();
    let luma = vendor_adjust_lut();
    assert_eq!(luma.len(), 256);
    let lut = AdjustLut {
        width: 256,
        height: 1,
        luma,
    };
    let expected = body_hlf_apply(
        &current,
        roi,
        &median,
        &gauss,
        &face,
        &skin,
        73.0,
        uniforms,
        Some(&lut),
    )
    .unwrap();

    let producer = ProducerInputs {
        hlf_body_adjust_lut: Some(HlfBodyAdjustLutProducerInputs {
            width: lut.width,
            height: lut.height,
            luma: luma.to_vec(),
        }),
        ..ProducerInputs::default()
    };
    let masks = masks(width, height, 22);
    let mut inputs = PortraitJobInputs {
        hlf_body: Some(HlfBodyInputs {
            width,
            height,
            median_rgb_full_f32: &median,
            gauss_rgb_full_f32: &gauss,
            face_mask_rgba_full_f32: &face,
            skin_mask_full_f32: &skin,
            roi,
            uniforms,
            adjust_lut: None,
        }),
        ..PortraitJobInputs::from_renderer_job(1, &masks)
    };
    inputs.fill_from_producer_inputs(&producer);
    assert_eq!(inputs.hlf_body.unwrap().adjust_lut, Some(lut));

    let mut actual = current.pixels.clone();
    let report = apply_portrait_effects_with_job(
        &mut actual,
        width,
        height,
        NormRect::FULL,
        &extra(30022, 0.73),
        &inputs,
    );
    assert_eq!(actual, expected.pixels);
    assert!(report.effects[0].executed);
    assert_real_kernel(&report.effects[0], BODY_KERNEL);
    #[cfg(windows)]
    {
        let _force_cpu = super::tests::ForceCudaFailureGuard::new();
        let mut cpu_actual = current.pixels.clone();
        let cpu_report = apply_portrait_effects_with_job(
            &mut cpu_actual,
            width,
            height,
            NormRect::FULL,
            &extra(30022, 0.73),
            &inputs,
        );
        assert_eq!(cpu_actual, expected.pixels);
        assert_eq!(cpu_report.effects[0].backend, PortraitBackend::Cpu);
        assert!(cpu_report.effects[0].executed);
    }
    eprintln!(
        "PF30022_VENDOR_LUT byte_len={} spotless_degree={} stage_degree=0.73",
        luma.len(),
        uniforms.spotless_degree
    );
}

#[test]
fn wrinkle_stage_uses_trunc_store_for_every_pf() {
    let (width, height) = (37u32, 29u32);
    let mut rng = Rng::new(0x21025);
    let source = image(width, height, 7);
    let base = image(width, height, 8);
    let residual = image(width, height, 9);
    let pixels = width as usize * height as usize;
    let mask: Vec<u8> = (0..pixels * 4).map(|_| rng.byte()).collect();
    let skin: Vec<u8> = (0..pixels).map(|_| rng.byte()).collect();
    let roi = Roi {
        x: 5,
        y: 4,
        width: 23,
        height: 17,
    };
    let uniforms = active_wrinkle_uniforms();

    for pf in [21025, 21069, 21070, 21071] {
        let expected = wrinkle_reference(
            pf, &source, &base, &residual, &mask, &skin, roi, 50.0, uniforms,
        )
        .unwrap();
        let masks = masks(width, height, pf as u64);
        let inputs = PortraitJobInputs {
            wrinkle: Some(WrinkleInputs {
                base_full: &base,
                residual_full: &residual,
                mask_rgba_full: &mask,
                skin_mask_full: &skin,
                roi,
                uniforms,
            }),
            ..PortraitJobInputs::from_renderer_job(1, &masks)
        };
        let mut actual = source.pixels.clone();
        let report = apply_portrait_effects_with_job(
            &mut actual,
            width,
            height,
            NormRect::FULL,
            &extra(pf, 0.5),
            &inputs,
        );
        assert_eq!(actual, expected.pixels, "PF{pf}");
        assert!(report.effects[0].executed);
        assert_real_kernel(&report.effects[0], WRINKLE_KERNEL);

        for y in 0..height {
            for x in 0..width {
                let offset = ((y * width + x) * 4) as usize;
                assert_eq!(actual[offset + 3], source.pixels[offset + 3]);
                if x < roi.x || x >= roi.x + roi.width || y < roi.y || y >= roi.y + roi.height {
                    assert_eq!(
                        &actual[offset..offset + 4],
                        &source.pixels[offset..offset + 4],
                        "PF{pf} outside ROI"
                    );
                }
            }
        }
    }
}

#[test]
fn wrinkle_stage_store_discriminator_is_trunc_not_round() {
    let (width, height) = (1u32, 1u32);
    let source = Image::solid(width, height, [0, 0, 0, 77]);
    let base = Image::solid(width, height, [0, 0, 0, 0]);
    let residual = Image::solid(width, height, [255, 255, 255, 0]);
    let mask = [0, 0, 255, 255];
    let skin = [255];
    let roi = Roi {
        x: 0,
        y: 0,
        width: 1,
        height: 1,
    };
    let uniforms = WrinkleUniforms {
        legal_alpha: 0.5,
        dark_circle_alpha: 0.0,
        ..active_wrinkle_uniforms()
    };
    let masks = masks(width, height, 10);
    let inputs = PortraitJobInputs {
        wrinkle: Some(WrinkleInputs {
            base_full: &base,
            residual_full: &residual,
            mask_rgba_full: &mask,
            skin_mask_full: &skin,
            roi,
            uniforms,
        }),
        ..PortraitJobInputs::from_renderer_job(1, &masks)
    };
    let mut actual = source.pixels.clone();
    let report = apply_portrait_effects_with_job(
        &mut actual,
        width,
        height,
        NormRect::FULL,
        &extra(21025, 1.0),
        &inputs,
    );

    assert_eq!(&actual[..4], &[127, 127, 127, 77]);
    assert_ne!(&actual[..3], &[128, 128, 128]);
    assert_real_kernel(&report.effects[0], WRINKLE_KERNEL);
}

#[test]
fn wrinkle_receipt_producer_reaches_stage_and_cpu_matches_cuda_bytes() {
    let (width, height) = (37u32, 29u32);
    let pixels = width as usize * height as usize;
    let mut rng = Rng::new(0x50_21025);
    let source = image(width, height, 0x50_21025);
    let base = image(width, height, 0x50_21026);
    let residual = image(width, height, 0x50_21027);
    let mask: Vec<u8> = (0..pixels * 4).map(|_| rng.byte()).collect();
    let skin: Vec<u8> = (0..pixels).map(|_| rng.byte()).collect();
    let roi = Roi {
        x: 5,
        y: 4,
        width: 23,
        height: 17,
    };
    let uniforms = active_wrinkle_uniforms();
    let masks = masks(width, height, 0x50_21028);

    for pf in [21025, 21069, 21070, 21071] {
        let params = extra(pf, 0.5);
        let receipt = PortraitJobReceipt::from_renderer_receipt(
            &source.pixels,
            width,
            height,
            NormRect::FULL,
            &params,
            &masks,
        );
        let same_job = WrinkleSameJobReceipt::from_renderer_receipt(&receipt);
        let legacy = WrinkleLegacySkinMaskProducer::from_same_job(
            receipt.job_id(),
            width,
            height,
            WrinkleLegacySkinMaskTexture::from_service_6200(pf, 2, 2, vec![0x50; 2 * 2 * 4])
                .unwrap(),
            skin.clone(),
        )
        .unwrap();
        let roi_contract = WrinkleFaceRoiContract::from_face_record(
            pf,
            true,
            width,
            height,
            [
                roi.x as f32,
                roi.y as f32,
                roi.width as f32,
                roi.height as f32,
            ],
            0.5,
            vec![[16.0, 12.0]; 194],
            vec![[16.0, 12.0]; 56],
        )
        .unwrap();
        let mask_mini = WrinkleMaskMiniImageProducer::from_same_job(
            receipt.job_id(),
            width,
            height,
            WrinkleMaskMiniImage::from_capture(pf, vec![0x51; 1000 * 1000 * 4]).unwrap(),
            roi_contract,
            mask.clone(),
        )
        .unwrap();
        let wrinkle = WrinkleProducerInputs::from_same_job(
            same_job,
            width,
            height,
            base.clone(),
            residual.clone(),
            legacy,
            mask_mini,
            uniforms,
        )
        .unwrap();
        let producer = ProducerInputs {
            wrinkle: Some(wrinkle),
            ..ProducerInputs::default()
        };
        let mut inputs = PortraitJobInputs::from_renderer_receipt(receipt, &masks);
        inputs.fill_from_producer_inputs(&producer);

        let expected = wrinkle_reference(
            pf, &source, &base, &residual, &mask, &skin, roi, 50.0, uniforms,
        )
        .unwrap();
        let mut cuda_or_cpu = source.pixels.clone();
        let run = apply_portrait_effects_with_job(
            &mut cuda_or_cpu,
            width,
            height,
            NormRect::FULL,
            &params,
            &inputs,
        );
        assert_eq!(cuda_or_cpu, expected.pixels, "PF{pf} stage/CPU oracle");
        assert!(run.effects[0].executed, "PF{pf}");
        assert_real_kernel(&run.effects[0], WRINKLE_KERNEL);

        #[cfg(windows)]
        {
            let _force_cpu = super::tests::ForceCudaFailureGuard::new();
            let mut fallback = source.pixels.clone();
            let cpu_run = apply_portrait_effects_with_job(
                &mut fallback,
                width,
                height,
                NormRect::FULL,
                &params,
                &inputs,
            );
            assert_eq!(fallback, cuda_or_cpu, "PF{pf} CPU/CUDA byte diff");
            assert_eq!(cpu_run.effects[0].backend, PortraitBackend::Cpu);
            eprintln!(
                "PF{pf} wrinkle same_job receipt={} cpu_cuda_byte_diff=0 kernel={}",
                receipt.job_id(),
                WRINKLE_KERNEL
            );
        }
    }
}
