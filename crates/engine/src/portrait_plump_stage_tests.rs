use super::*;
use std::collections::BTreeMap;

use aurora_portrait::kernels::plump_pipeline::{
    ChannelMap, ColorEncoding, ModelInput, Roi, SessionOutput, Tensor,
};
use aurora_portrait::model_sha256::sha256;
use aurora_portrait::plump_session::{Route, SessionBinding, SessionStamp};

const SIZE: (u32, u32) = (4, 3);
const TICKET: u64 = 0x51_30210;

fn frame() -> Image {
    Image::new(
        SIZE.0,
        SIZE.1,
        (0..SIZE.0 * SIZE.1 * 4)
            .map(|index| ((index * 37 + 11) % 251) as u8)
            .collect(),
    )
    .unwrap()
}

fn binding() -> SessionBinding {
    SessionBinding {
        model_id: 597_383_622,
        route: Route::HighRes4,
        net_sha256: [7; 32],
        input_shape: [1, 4, 2, 2],
        input_tensor: "3071265413",
        output_tensor: "2294125226",
    }
}

fn model_input() -> ModelInput {
    ModelInput {
        shape: [1, 4, 2, 2],
        nchw: (0..16).map(|value| value as f32 / 16.0).collect(),
        digest: [3; 32],
    }
}

fn model_output() -> SessionOutput {
    let mut data = Vec::new();
    for channel in 0..4 {
        for pixel in 0..4 {
            data.push(match channel {
                3 => {
                    if pixel < 2 {
                        0.5
                    } else {
                        2.0
                    }
                }
                _ => (channel as f32 + 1.0) / 4.0,
            });
        }
    }
    SessionOutput {
        stamp: SessionStamp {
            model_id: binding().model_id,
            net_sha256: binding().net_sha256,
            input_digest: [3; 32],
            input_shape: binding().input_shape,
            output_tensor: binding().output_tensor,
        },
        tensor: Tensor {
            shape: [1, 4, 2, 2],
            data,
        },
    }
}

fn final_composite(
    source: &Image,
    requested_pfs: &[u32],
    model_input: &ModelInput,
    model_output: &SessionOutput,
) -> PlumpFinalComposite {
    let mut frame = source.clone();
    for (index, pixel) in frame.pixels.chunks_exact_mut(4).enumerate() {
        pixel[0] = (pixel[0] as usize + index * 3 + 17).min(255) as u8;
        pixel[1] = (pixel[1] as usize + index * 5 + 29).min(255) as u8;
        pixel[2] = (pixel[2] as usize + index * 7 + 31).min(255) as u8;
    }
    PlumpFinalComposite {
        receipt: PlumpCompositeReceipt {
            ticket: TICKET,
            face_id: 2,
            requested_pfs: requested_pfs.to_vec(),
            source_sha256: sha256(&source.pixels),
            model_id: model_output.stamp.model_id,
            net_sha256: model_output.stamp.net_sha256,
            model_input_digest: model_input.digest,
            output_tensor: model_output.stamp.output_tensor,
        },
        frame,
    }
}

fn producer(source: &Image, requested_pfs: &[u32]) -> PlumpProducerInputs {
    let model_input = model_input();
    let model_output = model_output();
    PlumpProducerInputs {
        ticket: TICKET,
        face_id: 2,
        slots: [1.0; 17],
        requested_pfs: requested_pfs.to_vec(),
        full_face_descriptor: requested_pfs.contains(&30_210).then_some(1.0),
        session: binding(),
        final_composite: Some(final_composite(
            source,
            requested_pfs,
            &model_input,
            &model_output,
        )),
        model_input: Some(model_input),
        model_output: Some(model_output),
        channel_map: ChannelMap {
            color: [0, 1, 2],
            split_mask: 3,
        },
        color_encoding: ColorEncoding::Unit,
        roi: Roi {
            x: 0,
            y: 0,
            width: SIZE.0,
            height: SIZE.1,
        },
    }
}

fn bind_ticket(mut producer: PlumpProducerInputs, ticket: u64) -> PlumpProducerInputs {
    producer.ticket = ticket;
    producer
        .final_composite
        .as_mut()
        .expect("fixture composite")
        .receipt
        .ticket = ticket;
    producer
}

fn controls(requested_pfs: &[u32]) -> BTreeMap<String, f32> {
    requested_pfs
        .iter()
        .map(|pf| {
            (
                format!("pc:{pf}"),
                if *pf == 30_210 { 4.0 } else { 50.0 },
            )
        })
        .collect()
}

#[test]
fn all_eight_plump_rows_execute_one_same_job_group() {
    let source = frame();
    let requested = PLUMP_PFS.to_vec();
    let extra = controls(&requested);
    let masks = PortraitMasks::default();
    let receipt = PortraitJobReceipt::from_renderer_receipt(
        &source.pixels,
        SIZE.0,
        SIZE.1,
        NormRect::FULL,
        &extra,
        &masks,
    );
    let requested_producer = bind_ticket(producer(&source, &requested), receipt.job_id());
    let owner = ProducerInputs {
        plump: Some(requested_producer),
        ..Default::default()
    };
    let mut job = PortraitJobInputs::from_renderer_receipt(receipt, &masks);
    job.fill_from_producer_inputs(&owner);
    let mut pixels = source.pixels.clone();
    let report = apply_portrait_effects_with_job(
        &mut pixels,
        SIZE.0,
        SIZE.1,
        NormRect::FULL,
        &extra,
        &job,
    );

    assert_eq!(report.effects.len(), PLUMP_PFS.len());
    for (run, pf) in report.effects.iter().zip(PLUMP_PFS) {
        assert_eq!(run.pf, pf);
        assert!(run.executed && !run.skipped, "PF{pf}: {run:?}");
        assert_eq!(run.backend, PortraitBackend::Cpu);
        assert!(run.refusal.is_none(), "PF{pf}: {run:?}");
    }
    assert_eq!(
        pixels,
        bind_ticket(producer(&source, &requested), receipt.job_id())
            .final_composite
            .unwrap()
            .frame
            .pixels
    );
}

#[test]
fn absent_model_output_or_final_composite_is_typed_refusal() {
    let source = frame();
    let requested = vec![30_191];
    let extra = controls(&requested);
    let masks = PortraitMasks::default();
    let receipt = PortraitJobReceipt::from_renderer_receipt(
        &source.pixels,
        SIZE.0,
        SIZE.1,
        NormRect::FULL,
        &extra,
        &masks,
    );

    let mut missing_output = bind_ticket(producer(&source, &requested), receipt.job_id());
    missing_output.model_output = None;
    let owner = ProducerInputs {
        plump: Some(missing_output),
        ..Default::default()
    };
    let mut job = PortraitJobInputs::from_renderer_receipt(receipt, &masks);
    job.fill_from_producer_inputs(&owner);
    let mut pixels = source.pixels.clone();
    let report = apply_portrait_effects_with_job(
        &mut pixels,
        SIZE.0,
        SIZE.1,
        NormRect::FULL,
        &extra,
        &job,
    );
    assert_eq!(report.effects[0].refusal, Some(Refusal::MissingInput("plump_model_output")));
    assert_eq!(pixels, source.pixels);

    let mut missing_composite = bind_ticket(producer(&source, &requested), receipt.job_id());
    missing_composite.final_composite = None;
    let owner = ProducerInputs {
        plump: Some(missing_composite),
        ..Default::default()
    };
    let mut job = PortraitJobInputs::from_renderer_receipt(receipt, &masks);
    job.fill_from_producer_inputs(&owner);
    let mut pixels = source.pixels.clone();
    let report = apply_portrait_effects_with_job(
        &mut pixels,
        SIZE.0,
        SIZE.1,
        NormRect::FULL,
        &extra,
        &job,
    );
    assert_eq!(
        report.effects[0].refusal,
        Some(Refusal::MissingInput("facial_plumping_final_composite"))
    );
    assert_eq!(pixels, source.pixels);
}

#[test]
fn stale_source_set_and_job_are_refused_before_pixels_change() {
    let source = frame();
    let requested = vec![30_191];
    let extra = controls(&requested);
    let masks = PortraitMasks::default();

    let mut stale = producer(&source, &requested);
    stale.requested_pfs = vec![30_194];
    let owner = ProducerInputs {
        plump: Some(stale),
        ..Default::default()
    };
    let mut job = PortraitJobInputs::from_renderer_job(TICKET, &masks);
    job.fill_from_producer_inputs(&owner);
    let mut pixels = source.pixels.clone();
    let report = apply_portrait_effects_with_job(
        &mut pixels,
        SIZE.0,
        SIZE.1,
        NormRect::FULL,
        &extra,
        &job,
    );
    assert_eq!(
        report.effects[0].refusal,
        Some(Refusal::ValueOutOfRange("plump requested PF set"))
    );
    assert_eq!(pixels, source.pixels);

    let mut stale_frame = producer(&source, &requested);
    stale_frame
        .final_composite
        .as_mut()
        .unwrap()
        .receipt
        .source_sha256 = [9; 32];
    let owner = ProducerInputs {
        plump: Some(stale_frame),
        ..Default::default()
    };
    let mut job = PortraitJobInputs::from_renderer_job(TICKET, &masks);
    job.fill_from_producer_inputs(&owner);
    let mut pixels = source.pixels.clone();
    let report = apply_portrait_effects_with_job(
        &mut pixels,
        SIZE.0,
        SIZE.1,
        NormRect::FULL,
        &extra,
        &job,
    );
    assert_eq!(
        report.effects[0].refusal,
        Some(Refusal::ValueOutOfRange(
            "plump final composite same-frame digest"
        ))
    );
    assert_eq!(pixels, source.pixels);

    let owner = ProducerInputs {
        plump: Some(producer(&source, &requested)),
        ..Default::default()
    };
    let mut job = PortraitJobInputs::from_renderer_job(TICKET + 1, &masks);
    job.fill_from_producer_inputs(&owner);
    let mut pixels = source.pixels.clone();
    let report = apply_portrait_effects_with_job(
        &mut pixels,
        SIZE.0,
        SIZE.1,
        NormRect::FULL,
        &extra,
        &job,
    );
    assert_eq!(
        report.effects[0].refusal,
        Some(Refusal::ValueOutOfRange("plump same-job ticket"))
    );
    assert_eq!(pixels, source.pixels);
}

#[test]
fn zero_ui_control_needs_no_model_or_composite() {
    let source = frame();
    let masks = PortraitMasks::default();
    let extra = BTreeMap::from([("pc:30191".to_string(), 0.0)]);
    let receipt = PortraitJobReceipt::from_renderer_receipt(
        &source.pixels,
        SIZE.0,
        SIZE.1,
        NormRect::FULL,
        &extra,
        &masks,
    );
    let job = PortraitJobInputs::from_renderer_receipt(receipt, &masks);
    let mut pixels = source.pixels.clone();
    let report = apply_portrait_effects_with_job(
        &mut pixels,
        SIZE.0,
        SIZE.1,
        NormRect::FULL,
        &extra,
        &job,
    );
    assert!(report.effects[0].skipped);
    assert!(!report.effects[0].executed);
    assert!(report.effects[0].refusal.is_none());
    assert_eq!(pixels, source.pixels);
}

#[test]
fn foreign_model_output_is_refused_by_session_identity() {
    let source = frame();
    let requested = vec![30_191];
    let extra = controls(&requested);
    let masks = PortraitMasks::default();
    let mut plump = producer(&source, &requested);
    plump
        .model_output
        .as_mut()
        .expect("fixture output")
        .stamp
        .input_digest = [9; 32];
    let owner = ProducerInputs {
        plump: Some(plump),
        ..Default::default()
    };
    let mut job = PortraitJobInputs::from_renderer_job(TICKET, &masks);
    job.fill_from_producer_inputs(&owner);
    let mut pixels = source.pixels.clone();
    let report = apply_portrait_effects_with_job(
        &mut pixels,
        SIZE.0,
        SIZE.1,
        NormRect::FULL,
        &extra,
        &job,
    );
    assert_eq!(
        report.effects[0].refusal,
        Some(Refusal::ValueOutOfRange("model output session"))
    );
    assert_eq!(pixels, source.pixels);
}

#[test]
fn fractional_full_face_ui_is_refused() {
    assert_eq!(
        full_face_descriptor_from_ui(2.5),
        Err(Refusal::ValueOutOfRange("PF30210 UI"))
    );
}
