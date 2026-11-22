use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use aurora_engine::{
    NormRect, Pf30025LiveModelOutputs, Pf30025ProducerInputs, Pf30025SameJobReceipt,
    Pf30025SourceSurface, PortraitBackend, PortraitJobInputs, PortraitJobReceipt, PortraitMasks,
    ProducerInputs,
    apply_portrait_effects_with_job,
};
use aurora_portrait::{Image, Refusal};
use aurora_portrait::kernels::face_detail::{
    Rect as ToothRect, ToothRepairProducer, tooth_repair_same_job,
};

const FRAME_WIDTH: u32 = 2400;
const FRAME_HEIGHT: u32 = 1600;
const CROP_X: u32 = 644;
const CROP_Y: u32 = 571;
const CROP_WIDTH: u32 = 1024;
const CROP_HEIGHT: u32 = 768;
const MODEL_ID: u32 = 1_940_369_920;
const DEGREE: f32 = 1.0;

fn fixture_dir() -> PathBuf {
    PathBuf::from(std::env::var_os("PF30025_FIXTURE_DIR").expect("PF30025_FIXTURE_DIR"))
}

fn read_f32le(path: &Path, expected: usize) -> Vec<f32> {
    let bytes = fs::read(path).expect("fixture bytes");
    assert_eq!(bytes.len(), expected * 4, "{}", path.display());
    bytes
        .chunks_exact(4)
        .map(|word| f32::from_le_bytes(word.try_into().unwrap()))
        .collect()
}

fn source_crop() -> Vec<u8> {
    let image = image::open(fixture_dir().join("source_crop_1024x768.png"))
        .expect("source crop")
        .to_rgba8();
    assert_eq!(image.dimensions(), (CROP_WIDTH, CROP_HEIGHT));
    image.into_raw()
}

fn crop_teeth_mask() -> Vec<f32> {
    let bytes = fs::read(fixture_dir().join("teeth_mask_2400x1600.u8")).expect("teeth mask");
    assert_eq!(bytes.len(), (FRAME_WIDTH * FRAME_HEIGHT) as usize);
    let mut out = vec![0.0f32; (CROP_WIDTH * CROP_HEIGHT) as usize];
    for y in 0..CROP_HEIGHT {
        for x in 0..CROP_WIDTH {
            let source = ((y + CROP_Y) * FRAME_WIDTH + x + CROP_X) as usize;
            out[(y * CROP_WIDTH + x) as usize] = bytes[source] as f32 / 255.0;
        }
    }
    out
}

fn receipt_parts(
    source: &[u8],
    repair: &[u8],
    gate: &[f32],
    job_id: u64,
) -> Pf30025SameJobReceipt {
    Pf30025SameJobReceipt {
        job_id,
        model_id: MODEL_ID,
        width: CROP_WIDTH,
        height: CROP_HEIGHT,
        source_stride: CROP_WIDTH * 4,
        source_sha256: aurora_portrait::model_sha256::sha256(source),
        repair_sha256: aurora_portrait::model_sha256::sha256(repair),
        gate_sha256: aurora_portrait::model_sha256::sha256(unsafe {
            std::slice::from_raw_parts(
                gate.as_ptr().cast::<u8>(),
                std::mem::size_of_val(gate),
            )
        }),
    }
}

#[test]
fn offline_same_job_bundle_enters_portrait_stage_and_matches_cpu() {
    let Some(fixture) = std::env::var_os("PF30025_FIXTURE_DIR") else {
        return;
    };
    let _ = fixture;
    let dir = fixture_dir();
    let source = source_crop();
    let repair = read_f32le(
        &dir.join(format!("output_{MODEL_ID}_780929437.f32le")),
        (CROP_WIDTH * CROP_HEIGHT * 3) as usize,
    );
    let gate = read_f32le(
        &dir.join(format!("output_{MODEL_ID}_3204548346.f32le")),
        (CROP_WIDTH * CROP_HEIGHT) as usize,
    );
    let teeth_mask = crop_teeth_mask();
    let pixels = (CROP_WIDTH * CROP_HEIGHT) as usize;
    let repair_planes = [
        &repair[..pixels],
        &repair[pixels..pixels * 2],
        &repair[pixels * 2..],
    ];

    let masks = PortraitMasks::default();
    let params = BTreeMap::from([("pc:30025".to_string(), DEGREE)]);
    let receipt = PortraitJobReceipt::from_renderer_receipt(
        &source,
        CROP_WIDTH,
        CROP_HEIGHT,
        NormRect::FULL,
        &params,
        &masks,
    );
    let source_surface = Pf30025SourceSurface {
        job_id: receipt.job_id(),
        width: CROP_WIDTH,
        height: CROP_HEIGHT,
        row_stride: CROP_WIDTH * 4,
        rgba: &source,
    };
    let model = Pf30025LiveModelOutputs {
        job_id: receipt.job_id(),
        model_id: MODEL_ID,
        source_surface,
        repair_780929437_bgr: repair_planes,
        gate_3204548346: &gate,
    };
    let bundle = Pf30025ProducerInputs::from_live_model(
        model,
        teeth_mask,
        ToothRect {
            x: 333,
            y: 312,
            width: 356,
            height: 143,
        },
    )
    .expect("same-job bundle");
    assert_eq!(
        bundle.receipt,
        receipt_parts(&source, &bundle.repair_780929437_bgra, &bundle.gate_3204548346, receipt.job_id())
    );

    let mut producer = ProducerInputs::default();
    producer.write_pf30025(bundle);
    let mut inputs = PortraitJobInputs::from_renderer_receipt(receipt, &masks);
    inputs.fill_from_producer_inputs(&producer);

    let mut actual = source.clone();
    let report = apply_portrait_effects_with_job(
        &mut actual,
        CROP_WIDTH,
        CROP_HEIGHT,
        NormRect::FULL,
        &params,
        &inputs,
    );
    assert_eq!(report.effects.len(), 1);
    let effect = &report.effects[0];
    assert!(effect.executed, "{:?}", effect.refusal);
    assert!(effect.refusal.is_none());
    if std::env::var_os("PORTRAIT_CUDA_REQUIRED").is_some() {
        assert_eq!(
            effect.backend,
            PortraitBackend::Cuda,
            "{:?}",
            effect.fallback_reason
        );
        assert_eq!(effect.dispatches.len(), 1);
        assert_eq!(
            effect.dispatches[0].kernel,
            aurora_portrait::cuda_pf30025::KERNEL
        );
        assert!(!effect.dispatches[0].device.is_empty());
    }
    if std::env::var("AURORA_PORTRAIT_RENDER_BACKEND")
        .is_ok_and(|value| value.eq_ignore_ascii_case("cpu"))
    {
        assert_eq!(effect.backend, PortraitBackend::Cpu);
        assert!(effect.fallback_reason.is_some());
    }

    let source_image = Image::new(CROP_WIDTH, CROP_HEIGHT, source.clone()).unwrap();
    let pf = producer.pf30025.as_ref().unwrap();
    let expected = tooth_repair_same_job(
        &source_image,
        Some(ToothRepairProducer {
            repair_780929437_bgra: &pf.repair_780929437_bgra,
            gate_3204548346: &pf.gate_3204548346,
            roi: pf.roi,
        }),
        DEGREE * 100.0,
    )
    .expect("CPU oracle");
    let differing_bytes = actual
        .iter()
        .zip(&expected.pixels)
        .filter(|(left, right)| left != right)
        .count();
    assert_eq!(actual, expected.pixels);
    assert_eq!(differing_bytes, 0);
    for pixel in 0..pixels {
        let offset = pixel * 4;
        assert_eq!(actual[offset + 3], source[offset + 3]);
        let x = (pixel % CROP_WIDTH as usize) as i32;
        let y = (pixel / CROP_WIDTH as usize) as i32;
        let in_roi = x >= pf.roi.x
            && y >= pf.roi.y
            && x < pf.roi.x + pf.roi.width
            && y < pf.roi.y + pf.roi.height;
        if !in_roi {
            assert_eq!(&actual[offset..offset + 4], &source[offset..offset + 4]);
        }
    }

    let dispatch = effect.dispatches.first();
    eprintln!(
        "PF30025 offline source_surface={}x{} repair={} gate={} model={} backend={:?} kernel={} device={} cc={:?} driver={} ptx_isa={:?} differing_bytes={} fallback={:?}",
        CROP_WIDTH,
        CROP_HEIGHT,
        pf.repair_780929437_bgra.len(),
        pf.gate_3204548346.len(),
        MODEL_ID,
        effect.backend,
        dispatch.map(|dispatch| dispatch.kernel).unwrap_or(""),
        dispatch.map(|dispatch| dispatch.device.as_str()).unwrap_or(""),
        dispatch.map(|dispatch| dispatch.compute_cap),
        dispatch.map(|dispatch| dispatch.driver_version).unwrap_or_default(),
        dispatch.map(|dispatch| dispatch.ptx_isa),
        differing_bytes,
        effect.fallback_reason,
    );
}

#[test]
fn a_changed_source_surface_is_not_promoted_to_same_job_input() {
    let Some(_) = std::env::var_os("PF30025_FIXTURE_DIR") else {
        return;
    };
    let source = source_crop();
    let gate = vec![0.0f32; (CROP_WIDTH * CROP_HEIGHT) as usize];
    let repair = vec![0u8; (CROP_WIDTH * CROP_HEIGHT * 4) as usize];
    let teeth_mask = vec![0.0f32; (CROP_WIDTH * CROP_HEIGHT) as usize];
    let masks = PortraitMasks::default();
    let params = BTreeMap::from([("pc:30025".to_string(), DEGREE)]);
    let renderer_receipt = PortraitJobReceipt::from_renderer_receipt(
        &source,
        CROP_WIDTH,
        CROP_HEIGHT,
        NormRect::FULL,
        &params,
        &masks,
    );
    let mut receipt = receipt_parts(&source, &repair, &gate, renderer_receipt.job_id());
    receipt.source_sha256[0] ^= 1;
    let mut inputs = PortraitJobInputs::from_renderer_receipt(
        renderer_receipt,
        &masks,
    );
    inputs.pf30025 = Some(aurora_engine::Pf30025Inputs {
        receipt,
        repair_780929437_bgra: &repair,
        gate_3204548346: &gate,
        teeth_mask: &teeth_mask,
        roi: ToothRect {
            x: 0,
            y: 0,
            width: 1,
            height: 1,
        },
    });
    let mut frame = source.clone();
    let report = apply_portrait_effects_with_job(
        &mut frame,
        CROP_WIDTH,
        CROP_HEIGHT,
        NormRect::FULL,
        &params,
        &inputs,
    );
    assert_eq!(frame, source);
    assert_eq!(
        report.effects[0].refusal,
        Some(Refusal::MissingInput("PF30025 source surface"))
    );
    assert!(!report.effects[0].executed);
}
