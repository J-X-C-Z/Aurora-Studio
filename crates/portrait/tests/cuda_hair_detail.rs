#![cfg(windows)]

use aurora_portrait::cuda::CudaError;
use aurora_portrait::cuda_hair_detail::{
    GLOSS_KERNEL, GlossProducer, HairDetail, Roi, SMOOTH_KERNEL, SmoothParams, SmoothProducer,
};
use aurora_portrait::{Image, Refusal};

fn gpu() -> Option<HairDetail> {
    match HairDetail::open() {
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

fn hex(text: &str) -> Vec<u8> {
    let compact: Vec<u8> = text
        .bytes()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect();
    assert_eq!(compact.len() % 2, 0, "odd hex fixture");
    compact
        .chunks_exact(2)
        .map(|pair| {
            let digit = |byte: u8| match byte {
                b'0'..=b'9' => byte - b'0',
                b'a'..=b'f' => byte - b'a' + 10,
                b'A'..=b'F' => byte - b'A' + 10,
                _ => panic!("bad hex digit"),
            };
            (digit(pair[0]) << 4) | digit(pair[1])
        })
        .collect()
}

fn image(width: u32, height: u32, bytes: Vec<u8>) -> Image {
    Image::new(width, height, bytes).unwrap()
}

fn reference_smooth(
    source: &Image,
    roi: Roi,
    params: SmoothParams,
    producer: SmoothProducer<'_>,
) -> Image {
    let mut output = source.clone();
    for ry in 0..roi.height as usize {
        for rx in 0..roi.width as usize {
            let hair_byte = producer.hair_mask[ry * producer.hair_stride as usize + rx];
            if hair_byte == 0 {
                continue;
            }
            let mix_byte = producer.mix_mask[ry * producer.mix_stride as usize + rx];
            let detail_byte = producer.detail[ry * producer.detail_stride as usize + rx];
            let gate = hair_byte as f32 * 0.003921568859368563;
            let mix_weight = (mix_byte as f32 * 0.003921568859368563) * params.mix_strength * gate;
            let detail_add =
                ((detail_byte as f32 - 127.5) * 4.0) * (gate * params.detail_strength * 0.8);
            let x = roi.x as usize + rx;
            let y = roi.y as usize + ry;
            let source_offset = (y * source.width as usize + x) * 4;
            let blend_offset = ry * producer.blend_stride as usize + rx * 3;
            for channel in 0..3 {
                let src = source.pixels[source_offset + channel] as f32;
                let base = if mix_weight > 0.0 {
                    src * (1.0 - mix_weight)
                        + producer.blend_rgb[blend_offset + channel] as f32 * mix_weight
                } else {
                    src
                };
                output.pixels[source_offset + channel] =
                    (base + detail_add).round_ties_even().clamp(0.0, 255.0) as u8;
            }
        }
    }
    output
}

fn reference_gloss(source: &Image, roi: Roi, scale: f32, producer: GlossProducer<'_>) -> Image {
    let mut output = source.clone();
    for ry in 0..roi.height as usize {
        for rx in 0..roi.width as usize {
            let mask = producer.hair_mask[ry * producer.hair_stride as usize + rx];
            if mask == 0 {
                continue;
            }
            let detail = producer.model_detail[ry * producer.detail_stride as usize + rx];
            let delta =
                ((detail as f32 - 127.5) * 2.0) * mask as f32 * 0.003921568859368563 * scale;
            let x = roi.x as usize + rx;
            let y = roi.y as usize + ry;
            let offset = (y * source.width as usize + x) * 4;
            for channel in 0..3 {
                output.pixels[offset + channel] =
                    (source.pixels[offset + channel] as f32 + delta) as u8;
            }
        }
    }
    output
}

#[test]
#[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
fn pf90139_tie_uses_nearest_even_not_scalar_or_trunc() {
    let Some(gpu) = gpu() else { return };
    let source_bytes = hex(aurora_portrait::private_asset_str!("fixtures/portrait/hair_detail/pf90139_tie_source.rgba8.hex").unwrap());
    let blend = hex(aurora_portrait::private_asset_str!("fixtures/portrait/hair_detail/pf90139_tie_blend.rgb8.hex").unwrap());
    let nearest_even = hex(aurora_portrait::private_asset_str!("fixtures/portrait/hair_detail/pf90139_tie_nearest_even.rgba8.hex").unwrap());
    let scalar = hex(aurora_portrait::private_asset_str!("fixtures/portrait/hair_detail/pf90139_tie_scalar_half_away.rgba8.hex").unwrap());
    let truncated = hex(aurora_portrait::private_asset_str!("fixtures/portrait/hair_detail/pf90139_tie_trunc.rgba8.hex").unwrap());
    let mask = [255u8; 32];
    let detail = [128u8; 32];
    let source = image(32, 1, source_bytes);
    let producer = SmoothProducer {
        ticket: 1,
        hair_mask: &mask,
        mix_mask: &mask,
        blend_rgb: &blend,
        detail: &detail,
        hair_stride: 32,
        mix_stride: 32,
        blend_stride: 96,
        detail_stride: 32,
    };
    let (actual, dispatch) = gpu
        .smooth(
            &source,
            Roi::full(32, 1),
            SmoothParams {
                mix_strength: 0.5,
                detail_strength: 0.0,
            },
            Some(producer),
        )
        .unwrap();
    assert_eq!(actual.pixels, nearest_even);
    assert_ne!(actual.pixels, scalar);
    assert_ne!(actual.pixels, truncated);
    let dispatch = dispatch.unwrap();
    eprintln!(
        "DEVICE={} CC={}.{} DRIVER={} PTX_ISA={}.{} PTX_BYTES={} KERNEL={}",
        dispatch.device,
        dispatch.compute_cap.0,
        dispatch.compute_cap.1,
        dispatch.driver_version,
        dispatch.ptx_isa.0,
        dispatch.ptx_isa.1,
        dispatch.ptx_bytes,
        dispatch.kernel
    );
    assert_eq!(dispatch.kernel, SMOOTH_KERNEL);
}

#[test]
#[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
fn pf90139_roi_and_independent_strides_match_the_fixture() {
    let Some(gpu) = gpu() else { return };
    let source = image(
        5,
        3,
        hex(aurora_portrait::private_asset_str!("fixtures/portrait/hair_detail/pf90139_roi_source.rgba8.hex").unwrap()),
    );
    let hair = hex(aurora_portrait::private_asset_str!("fixtures/portrait/hair_detail/pf90139_roi_hair.hex").unwrap());
    let mix = hex(aurora_portrait::private_asset_str!("fixtures/portrait/hair_detail/pf90139_roi_mix.hex").unwrap());
    let detail = hex(aurora_portrait::private_asset_str!("fixtures/portrait/hair_detail/pf90139_roi_detail.hex").unwrap());
    let blend = hex(aurora_portrait::private_asset_str!("fixtures/portrait/hair_detail/pf90139_roi_blend.rgb8.hex").unwrap());
    let fixture_expected = image(
        5,
        3,
        hex(aurora_portrait::private_asset_str!("fixtures/portrait/hair_detail/pf90139_roi_nearest_even.rgba8.hex").unwrap()),
    );
    let producer = SmoothProducer {
        ticket: 7,
        hair_mask: &hair[8..],
        mix_mask: &mix[10..],
        blend_rgb: &blend[20..],
        detail: &detail[12..],
        hair_stride: 7,
        mix_stride: 9,
        blend_stride: 17,
        detail_stride: 11,
    };
    let roi = Roi {
        x: 1,
        y: 1,
        width: 3,
        height: 1,
    };
    let params = SmoothParams {
        mix_strength: 0.5,
        detail_strength: 0.5,
    };
    let reference = reference_smooth(&source, roi, params, producer);
    assert_eq!(reference, fixture_expected);
    let (actual, _) = gpu.smooth(&source, roi, params, Some(producer)).unwrap();
    assert_eq!(actual, fixture_expected);
}

#[test]
fn pf90139_zero_mix_still_applies_detail() {
    let Some(gpu) = gpu() else { return };
    let source = Image::solid(4, 3, [20, 40, 60, 77]);
    let mask = [255u8; 12];
    let blend = [200u8; 36];
    let detail = [255u8; 12];
    let producer = SmoothProducer {
        ticket: 2,
        hair_mask: &mask,
        mix_mask: &mask,
        blend_rgb: &blend,
        detail: &detail,
        hair_stride: 4,
        mix_stride: 4,
        blend_stride: 12,
        detail_stride: 4,
    };
    let params = SmoothParams {
        mix_strength: 0.0,
        detail_strength: 0.5,
    };
    let (actual, _) = gpu
        .smooth(&source, Roi::full(4, 3), params, Some(producer))
        .unwrap();
    assert_eq!(
        actual.pixels,
        reference_smooth(&source, Roi::full(4, 3), params, producer).pixels
    );
    assert_ne!(actual, source);
}

#[test]
fn pf90160_half_values_truncate_in_both_directions() {
    let Some(gpu) = gpu() else { return };
    let mut pixels = Vec::new();
    for i in 0..32u32 {
        pixels.extend_from_slice(&[(10 + i) as u8, (64 + i) as u8, (128 + i) as u8, 255]);
    }
    let source = image(32, 1, pixels);
    let mask = [255u8; 32];
    let positive = [128u8; 32];
    let negative = [127u8; 32];
    let roi = Roi::full(32, 1);
    for (detail, expected) in [
        (
            positive.as_slice(),
            reference_gloss(
                &source,
                roi,
                0.5,
                GlossProducer {
                    ticket: 1,
                    hair_mask: &mask,
                    model_detail: &positive,
                    hair_stride: 32,
                    detail_stride: 32,
                },
            ),
        ),
        (
            negative.as_slice(),
            reference_gloss(
                &source,
                roi,
                0.5,
                GlossProducer {
                    ticket: 1,
                    hair_mask: &mask,
                    model_detail: &negative,
                    hair_stride: 32,
                    detail_stride: 32,
                },
            ),
        ),
    ] {
        let producer = GlossProducer {
            ticket: 1,
            hair_mask: &mask,
            model_detail: detail,
            hair_stride: 32,
            detail_stride: 32,
        };
        let (actual, dispatch) = gpu.gloss(&source, roi, 0.5, Some(producer)).unwrap();
        assert_eq!(actual, expected);
        assert_eq!(dispatch.unwrap().kernel, GLOSS_KERNEL);
    }
    let positive_out = reference_gloss(
        &source,
        roi,
        0.5,
        GlossProducer {
            ticket: 1,
            hair_mask: &mask,
            model_detail: &positive,
            hair_stride: 32,
            detail_stride: 32,
        },
    );
    assert_eq!(positive_out, source);
    let negative_out = reference_gloss(
        &source,
        roi,
        0.5,
        GlossProducer {
            ticket: 1,
            hair_mask: &mask,
            model_detail: &negative,
            hair_stride: 32,
            detail_stride: 32,
        },
    );
    assert_eq!(negative_out.pixel(0, 0), [9, 63, 127, 255]);
    assert_ne!(negative_out, source);
}

#[test]
fn pf90160_roi_preserves_outside_pixels_and_alpha() {
    let Some(gpu) = gpu() else { return };
    let source = Image::solid(7, 5, [80, 90, 100, 33]);
    let roi = Roi {
        x: 2,
        y: 1,
        width: 3,
        height: 2,
    };
    let mask = [255u8; 6];
    let detail = [200u8; 6];
    let producer = GlossProducer {
        ticket: 3,
        hair_mask: &mask,
        model_detail: &detail,
        hair_stride: 3,
        detail_stride: 3,
    };
    let (actual, _) = gpu.gloss(&source, roi, 0.5, Some(producer)).unwrap();
    assert_eq!(
        actual.pixels,
        reference_gloss(&source, roi, 0.5, producer).pixels
    );
    for y in 0..source.height {
        for x in 0..source.width {
            if !(x >= roi.x && x < roi.x + roi.width && y >= roi.y && y < roi.y + roi.height) {
                assert_eq!(actual.pixel(x, y), source.pixel(x, y));
            }
            assert_eq!(actual.pixel(x, y)[3], source.pixel(x, y)[3]);
        }
    }
}

#[test]
fn missing_producers_and_receipts_refuse_before_launch() {
    let Some(gpu) = gpu() else { return };
    let source = Image::solid(2, 2, [10, 20, 30, 40]);
    let params = SmoothParams {
        mix_strength: 0.5,
        detail_strength: 0.0,
    };
    assert_eq!(
        gpu.smooth(&source, Roi::full(2, 2), params, None).err(),
        Some(CudaError::Refused(Refusal::MissingInput(
            "PF90139_PF90140_SAME_JOB_PRODUCER"
        )))
    );
    assert_eq!(
        gpu.gloss(&source, Roi::full(2, 2), 0.5, None).err(),
        Some(CudaError::Refused(Refusal::MissingInput(
            "PF90160_SAME_JOB_PRODUCER"
        )))
    );

    let mask = [255u8; 4];
    let blend = [0u8; 12];
    let producer = SmoothProducer {
        ticket: 0,
        hair_mask: &mask,
        mix_mask: &mask,
        blend_rgb: &blend,
        detail: &mask,
        hair_stride: 2,
        mix_stride: 2,
        blend_stride: 6,
        detail_stride: 2,
    };
    assert_eq!(
        gpu.smooth(&source, Roi::full(2, 2), params, Some(producer))
            .err(),
        Some(CudaError::Refused(Refusal::MissingInput(
            "PF90139_PF90140_SAME_JOB_RECEIPT"
        )))
    );
}

#[test]
fn identity_paths_do_not_require_a_producer() {
    let Some(gpu) = gpu() else { return };
    let source = Image::solid(2, 2, [10, 20, 30, 40]);
    let (smooth, smooth_dispatch) = gpu
        .smooth(
            &source,
            Roi::full(2, 2),
            SmoothParams {
                mix_strength: 0.0,
                detail_strength: 0.0,
            },
            None,
        )
        .unwrap();
    assert_eq!(smooth, source);
    assert!(smooth_dispatch.is_none());

    let (gloss, gloss_dispatch) = gpu.gloss(&source, Roi::full(2, 2), 0.0, None).unwrap();
    assert_eq!(gloss, source);
    assert!(gloss_dispatch.is_none());
}
