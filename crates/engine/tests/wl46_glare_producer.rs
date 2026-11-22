use aurora_engine::{
    PF90130_A_PLANAR_BYTES, PF90130_B_PLANAR_BYTES, PF90130_EFFECT_TENSOR_ID,
    PF90130_INPUT_TENSOR_ID, PF90130_MASK_TENSOR_ID, PF90130_MODEL_ID, PF90130_PIXELS,
    PF90130_PLANE_HEIGHT, PF90130_PLANE_WIDTH, Pf90130AlignedGeometry, Pf90130GlareProducer,
    PortraitImage as Image, PortraitRefusal as Refusal, apply_pf90130_producer_stage,
};
use aurora_portrait::kernels::pf90130_chain::{self, PixelRect};

const TICKET: u64 = 0x46_90130;

fn nhwc4(values: impl Fn(usize) -> [f32; 4]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(PF90130_PIXELS * 16);
    for pixel in 0..PF90130_PIXELS {
        for value in values(pixel) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

fn fixture() -> (Pf90130GlareProducer, Vec<u8>, Vec<u8>) {
    let input = nhwc4(|pixel| {
        [
            ((pixel % 251) as f32 / 125.5) - 1.0,
            ((pixel % 127) as f32 / 63.5) - 1.0,
            ((pixel % 61) as f32 / 30.5) - 1.0,
            0.0,
        ]
    });
    let effect = nhwc4(|pixel| {
        [
            ((pixel % 241) as f32 / 120.0) - 1.0,
            ((pixel % 113) as f32 / 56.0) - 1.0,
            ((pixel % 59) as f32 / 29.0) - 1.0,
            0.0,
        ]
    });
    let mask = nhwc4(|pixel| [((pixel % 101) as f32 / 50.0) - 1.0, 0.0, 0.0, 0.0]);
    let producer = Pf90130GlareProducer::from_same_job(
        TICKET,
        PF90130_MODEL_ID,
        PF90130_INPUT_TENSOR_ID,
        &input,
        PF90130_EFFECT_TENSOR_ID,
        &effect,
        PF90130_MASK_TENSOR_ID,
        &mask,
    )
    .unwrap();
    (producer, effect, mask)
}

#[test]
fn named_model_outputs_bind_to_a012_and_b0_planes() {
    let (producer, effect, mask) = fixture();
    let receipt = producer.receipt();
    assert_eq!(receipt.ticket(), TICKET);
    assert_eq!(receipt.model_id(), PF90130_MODEL_ID);
    assert_eq!(receipt.input_tensor_id(), PF90130_INPUT_TENSOR_ID);
    assert_eq!(receipt.effect_tensor_id(), PF90130_EFFECT_TENSOR_ID);
    assert_eq!(receipt.mask_tensor_id(), PF90130_MASK_TENSOR_ID);

    let planes = producer.bridge().unwrap();
    assert_eq!(planes.ticket, TICKET);
    assert_eq!(planes.material_planar.len(), PF90130_A_PLANAR_BYTES);
    assert_eq!(planes.mask_planar.len(), PF90130_B_PLANAR_BYTES);
    assert_eq!(planes.effect_rgb.len(), PF90130_PIXELS * 3);
    assert_eq!(planes.mask.len(), PF90130_PIXELS);
    assert_eq!(
        planes.material_planar_sha256,
        receipt.material_planar_sha256()
    );
    assert_eq!(planes.mask_planar_sha256, receipt.mask_planar_sha256());

    let mut expected_material = Vec::with_capacity(PF90130_A_PLANAR_BYTES);
    for plane in 0..3 {
        for pixel in effect.chunks_exact(16) {
            expected_material.extend_from_slice(&pixel[plane * 4..plane * 4 + 4]);
        }
    }
    let mut expected_mask = Vec::with_capacity(PF90130_B_PLANAR_BYTES);
    for pixel in mask.chunks_exact(16) {
        expected_mask.extend_from_slice(&pixel[..4]);
    }
    assert_eq!(planes.material_planar, expected_material);
    assert_eq!(planes.mask_planar, expected_mask);
}

#[test]
fn unknown_model_and_tensor_ids_are_refused() {
    let (_, effect, mask) = fixture();
    let input = vec![0u8; PF90130_PIXELS * 16];
    let cases = [
        (
            PF90130_MODEL_ID + 1,
            PF90130_INPUT_TENSOR_ID,
            PF90130_EFFECT_TENSOR_ID,
            PF90130_MASK_TENSOR_ID,
            Refusal::ValueOutOfRange("PF90130_MODEL_ID"),
        ),
        (
            PF90130_MODEL_ID,
            PF90130_MASK_TENSOR_ID,
            PF90130_EFFECT_TENSOR_ID,
            PF90130_MASK_TENSOR_ID,
            Refusal::ValueOutOfRange("PF90130_INPUT_TENSOR"),
        ),
        (
            PF90130_MODEL_ID,
            PF90130_INPUT_TENSOR_ID,
            PF90130_MASK_TENSOR_ID,
            PF90130_MASK_TENSOR_ID,
            Refusal::ValueOutOfRange("PF90130_EFFECT_TENSOR"),
        ),
        (
            PF90130_MODEL_ID,
            PF90130_INPUT_TENSOR_ID,
            PF90130_EFFECT_TENSOR_ID,
            PF90130_EFFECT_TENSOR_ID,
            Refusal::ValueOutOfRange("PF90130_MASK_TENSOR"),
        ),
    ];
    for (model, input_id, effect_id, mask_id, expected) in cases {
        assert_eq!(
            Pf90130GlareProducer::from_same_job(
                TICKET, model, input_id, &input, effect_id, &effect, mask_id, &mask,
            ),
            Err(expected)
        );
    }
}

#[test]
fn non_zero_padding_and_short_outputs_are_refused() {
    let (_, effect, mask) = fixture();
    let input = vec![0u8; PF90130_PIXELS * 16];
    let mut bad_effect = effect.clone();
    bad_effect[12] = 1;
    let mut bad_mask = mask;
    bad_mask[4] = 1;
    assert_eq!(
        Pf90130GlareProducer::from_same_job(
            TICKET,
            PF90130_MODEL_ID,
            PF90130_INPUT_TENSOR_ID,
            &input,
            PF90130_EFFECT_TENSOR_ID,
            &bad_effect,
            PF90130_MASK_TENSOR_ID,
            &bad_mask,
        ),
        Err(Refusal::ValueOutOfRange("PF90130_EFFECT_PADDING"))
    );
    assert_eq!(
        Pf90130GlareProducer::from_same_job(
            TICKET,
            PF90130_MODEL_ID,
            PF90130_INPUT_TENSOR_ID,
            &input,
            PF90130_EFFECT_TENSOR_ID,
            &effect,
            PF90130_MASK_TENSOR_ID,
            &[0u8; 16],
        ),
        Err(Refusal::ShapeMismatch("PF90130_MASK_NHWC4"))
    );
}

#[test]
fn producer_ticket_and_raw_outputs_feed_the_wl41_chain() {
    let (producer, _, _) = fixture();
    let source_rgb8 = vec![128u8; PF90130_PIXELS * 3];
    let base_rgb8 = source_rgb8.clone();
    let add_rgb_f32 = vec![0.0f32; PF90130_PIXELS * 3];
    let uv = vec![[0.5f32, 0.5f32]; PF90130_PIXELS];
    let geometry = Pf90130AlignedGeometry {
        source_width: PF90130_PLANE_WIDTH,
        source_height: PF90130_PLANE_HEIGHT,
        source_roi: PixelRect {
            x: 0,
            y: 0,
            width: PF90130_PLANE_WIDTH,
            height: PF90130_PLANE_HEIGHT,
        },
        uv: &uv,
        src_to_dst: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        source_rgb8: &source_rgb8,
        base_rgb8: &base_rgb8,
        add_rgb_f32: &add_rgb_f32,
        strength: 1.0,
    };
    let input = producer.chain_input(&geometry);
    let prepared = pf90130_chain::prepare(&input).unwrap();
    assert_eq!(prepared.ticket, TICKET);
    assert_eq!(
        (prepared.width, prepared.height),
        (PF90130_PLANE_WIDTH, PF90130_PLANE_HEIGHT)
    );
    assert_eq!(prepared.material_nhwc4.len(), PF90130_PIXELS * 16);
    assert_eq!(prepared.mask_nhwc4.len(), PF90130_PIXELS * 16);
}

#[test]
fn residual_writeback_uses_the_same_job_chain_and_preserves_alpha() {
    let (producer, _, _) = fixture();
    let source = Image::solid(
        PF90130_PLANE_WIDTH,
        PF90130_PLANE_HEIGHT,
        [90, 100, 110, 17],
    );
    let mut destination_pixels = Vec::with_capacity(PF90130_PIXELS * 4);
    for pixel in 0..PF90130_PIXELS {
        destination_pixels.extend_from_slice(&[20, 30, 40, (pixel % 251) as u8]);
    }
    let destination = Image::new(
        PF90130_PLANE_WIDTH,
        PF90130_PLANE_HEIGHT,
        destination_pixels,
    )
    .unwrap();
    let source_rgb8 = source
        .pixels
        .chunks_exact(4)
        .flat_map(|pixel| pixel[..3].iter().copied())
        .collect::<Vec<_>>();
    let base_rgb8 = source_rgb8.clone();
    let add_rgb_f32 = vec![0.0f32; PF90130_PIXELS * 3];
    let uv = vec![[0.5f32, 0.5f32]; PF90130_PIXELS];
    let geometry = Pf90130AlignedGeometry {
        source_width: PF90130_PLANE_WIDTH,
        source_height: PF90130_PLANE_HEIGHT,
        source_roi: PixelRect {
            x: 0,
            y: 0,
            width: PF90130_PLANE_WIDTH,
            height: PF90130_PLANE_HEIGHT,
        },
        uv: &uv,
        src_to_dst: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        source_rgb8: &source_rgb8,
        base_rgb8: &base_rgb8,
        add_rgb_f32: &add_rgb_f32,
        strength: 1.0,
    };
    let run = apply_pf90130_producer_stage(
        Some(&producer),
        Some(&source),
        Some(&destination),
        Some(&geometry),
    );
    assert!(run.refusal.is_none(), "{:?}", run.refusal);
    assert!(run.output.is_some());
    let output = run.writeback.expect("open post gate should write");
    assert_eq!(output.ticket, TICKET);
    for (actual, expected) in output
        .rgba8
        .pixels
        .chunks_exact(4)
        .zip(destination.pixels.chunks_exact(4))
    {
        assert_eq!(actual[3], expected[3]);
    }
}

#[test]
fn writeback_entries_refuse_a_non_aligned_frame() {
    let (producer, _, _) = fixture();
    let source = Image::solid(3, 2, [10, 20, 30, 255]);
    let destination = Image::solid(3, 2, [40, 50, 60, 7]);
    let source_rgb8 = [10u8; 18];
    let base_rgb8 = [20u8; 18];
    let add_rgb_f32 = [0.0f32; 18];
    let uv = [[0.5f32, 0.5f32]; 6];
    let geometry = Pf90130AlignedGeometry {
        source_width: 3,
        source_height: 2,
        source_roi: PixelRect {
            x: 0,
            y: 0,
            width: 3,
            height: 2,
        },
        uv: &uv,
        src_to_dst: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        source_rgb8: &source_rgb8,
        base_rgb8: &base_rgb8,
        add_rgb_f32: &add_rgb_f32,
        strength: 1.0,
    };
    assert_eq!(
        producer.run_stage_writeback(&source, &destination, 1.0, None),
        Err(Refusal::ShapeMismatch("PF90130_ALIGNED_FRAME"))
    );
    assert_eq!(
        producer.run_residual_writeback(&source, &destination, &geometry),
        Err(Refusal::ShapeMismatch("PF90130_ALIGNED_FRAME"))
    );
    assert_eq!(
        apply_pf90130_producer_stage(None, None, None, None).refusal,
        Some(Refusal::MissingInput("PF90130_SAME_JOB_RECEIPT"))
    );
    assert_eq!(
        apply_pf90130_producer_stage(Some(&producer), None, None, None).refusal,
        Some(Refusal::MissingInput("PF90130_SOURCE_FRAME"))
    );
    assert_eq!(
        apply_pf90130_producer_stage(Some(&producer), Some(&source), None, None).refusal,
        Some(Refusal::MissingInput("PF90130_DESTINATION_FRAME"))
    );
}
