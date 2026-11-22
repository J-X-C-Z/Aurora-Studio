use aurora_engine::{
    GlareChainFinalWritebackInputs, PortraitImage as Image, PortraitRefusal as Refusal,
    apply_glare_chain_final_writeback,
};
use aurora_portrait::kernels::pf90130_chain::{self, Inputs, PixelRect};
const WIDTH: u32 = 3;
const HEIGHT: u32 = 2;
const PIXELS: usize = WIDTH as usize * HEIGHT as usize;
const TICKET: u64 = 9741;

fn sha256(bytes: &[u8]) -> [u8; 32] {
    aurora_portrait::model_sha256::sha256(bytes)
}

fn nhwc4(values: &[[f32; 4]]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|pixel| pixel.iter().flat_map(|value| value.to_le_bytes()))
        .collect()
}

struct Fixture {
    source: Image,
    destination: Image,
    source_rgb: Vec<u8>,
    base_rgb: Vec<u8>,
    add_rgb: Vec<f32>,
    uv: Vec<[f32; 2]>,
    material: Vec<u8>,
    mask: Vec<u8>,
}

impl Fixture {
    fn new() -> Self {
        let source = Image::new(
            WIDTH,
            HEIGHT,
            vec![
                10, 20, 30, 1, 40, 50, 60, 2, 70, 80, 90, 3, 100, 110, 120, 4, 130, 140, 150, 5,
                160, 170, 180, 6,
            ],
        )
        .unwrap();
        let destination = Image::new(
            WIDTH,
            HEIGHT,
            vec![
                5, 6, 7, 11, 8, 9, 10, 12, 15, 16, 17, 13, 18, 19, 20, 14, 21, 22, 23, 15, 24,
                25, 26, 16,
            ],
        )
        .unwrap();
        let source_rgb = source
            .pixels
            .chunks_exact(4)
            .flat_map(|pixel| pixel[..3].iter().copied())
            .collect::<Vec<_>>();
        let add_rgb = vec![
            1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, -1.0, -2.0, -3.0, 4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let uv = vec![
            [0.0, 0.0],
            [0.5, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
        ];
        let material = nhwc4(&vec![[0.0, 0.0, 0.0, 0.0]; PIXELS]);
        let mask = nhwc4(&vec![[-1.0, 0.0, 0.0, 0.0]; PIXELS]);
        let base_rgb = source_rgb.clone();
        Self {
            source,
            destination,
            source_rgb,
            base_rgb,
            add_rgb,
            uv,
            material,
            mask,
        }
    }

    fn chain_input(&self) -> Inputs<'_> {
        Inputs {
            ticket: Some(TICKET),
            width: WIDTH,
            height: HEIGHT,
            source_width: Some(WIDTH),
            source_height: Some(HEIGHT),
            source_roi: Some(PixelRect {
                x: 0,
                y: 0,
                width: WIDTH,
                height: HEIGHT,
            }),
            roi: Some(PixelRect {
                x: 0,
                y: 0,
                width: WIDTH,
                height: HEIGHT,
            }),
            uv: Some(&self.uv),
            src_to_dst: Some([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            source_rgb8: Some(&self.source_rgb),
            base_rgb8: Some(&self.base_rgb),
            add_rgb_f32: Some(&self.add_rgb),
            material_nhwc4: Some(&self.material),
            mask_nhwc4: Some(&self.mask),
            strength: Some(0.5),
        }
    }

    fn receipts(&self) -> GlareChainFinalWritebackInputs {
        GlareChainFinalWritebackInputs {
            source_sha256: Some(sha256(&self.source.pixels)),
            destination_sha256: Some(sha256(&self.destination.pixels)),
        }
    }
}

#[test]
fn real_chain_post_second_binds_to_the_same_row_major_frame() {
    let fixture = Fixture::new();
    let input = fixture.chain_input();
    let chain = pf90130_chain::run(&input).unwrap();
    let output = apply_glare_chain_final_writeback(
        &fixture.source,
        &fixture.destination,
        &input,
        &chain,
        &fixture.receipts(),
    )
    .unwrap()
    .expect("open post gate should write");

    assert_eq!(output.ticket, TICKET);
    assert_eq!(output.rgba8.pixel(0, 0), [10, 21, 32, 11]);
    assert_eq!(output.rgba8.pixel(1, 0), [8, 9, 10, 12]);
    assert_eq!(output.rgba8.pixel(2, 0), [75, 90, 105, 13]);
    assert_eq!(output.rgba8.pixel(0, 1), [100, 109, 118, 14]);
    assert_eq!(output.rgba8.pixel(1, 1), [132, 142, 153, 15]);
    assert_eq!(output.rgba8.pixel(2, 1), [164, 174, 184, 16]);

    for (source, destination) in output
        .rgba8
        .pixels
        .chunks_exact(4)
        .zip(fixture.destination.pixels.chunks_exact(4))
    {
        assert_eq!(source[3], destination[3]);
    }
}

#[test]
fn binding_refuses_same_pixel_count_with_a_different_frame_shape() {
    let fixture = Fixture::new();
    let input = fixture.chain_input();
    let chain = pf90130_chain::run(&input).unwrap();
    let transposed = Image::new(HEIGHT, WIDTH, fixture.source.pixels.clone()).unwrap();
    assert_eq!(
        apply_glare_chain_final_writeback(
            &transposed,
            &transposed,
            &input,
            &chain,
            &fixture.receipts(),
        ),
        Err(Refusal::ShapeMismatch("PF90130_FINAL_FRAME"))
    );
}

#[test]
fn binding_refuses_a_source_rgb_plane_from_another_frame() {
    let fixture = Fixture::new();
    let input = fixture.chain_input();
    let chain = pf90130_chain::run(&input).unwrap();
    let mut pixels = fixture.source.pixels.clone();
    pixels[0] ^= 1;
    let other_source = Image::new(WIDTH, HEIGHT, pixels).unwrap();
    assert_eq!(
        apply_glare_chain_final_writeback(
            &other_source,
            &fixture.destination,
            &input,
            &chain,
            &GlareChainFinalWritebackInputs {
                source_sha256: Some(sha256(&other_source.pixels)),
                destination_sha256: Some(sha256(&fixture.destination.pixels)),
            },
        ),
        Err(Refusal::ValueOutOfRange("PF90130_CHAIN_SOURCE_FRAME"))
    );
}

#[test]
fn binding_refuses_a_chain_ticket_from_another_job() {
    let fixture = Fixture::new();
    let input = fixture.chain_input();
    let mut chain = pf90130_chain::run(&input).unwrap();
    chain.ticket += 1;
    assert_eq!(
        apply_glare_chain_final_writeback(
            &fixture.source,
            &fixture.destination,
            &input,
            &chain,
            &fixture.receipts(),
        ),
        Err(Refusal::ValueOutOfRange("PF90130_FINAL_RECEIPT"))
    );
}

#[test]
fn binding_refuses_a_partial_destination_roi() {
    let fixture = Fixture::new();
    let input = fixture.chain_input();
    let chain = pf90130_chain::run(&input).unwrap();
    let partial = Inputs {
        roi: Some(PixelRect {
            x: 1,
            y: 0,
            width: 2,
            height: 2,
        }),
        ..input
    };
    assert_eq!(
        apply_glare_chain_final_writeback(
            &fixture.source,
            &fixture.destination,
            &partial,
            &chain,
            &fixture.receipts(),
        ),
        Err(Refusal::ValueOutOfRange("PF90130_ROI"))
    );
}
