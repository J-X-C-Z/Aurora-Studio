use aurora_engine::{
    GlareFinalWritebackInputs, GlarePlanarInputs, GlarePlanarReceipts, GlareWritebackInputs,
    PF90130_A_PLANAR_BYTES, PF90130_B_PLANAR_BYTES, PF90130_PIXELS, PF90130_PLANE_HEIGHT,
    PF90130_PLANE_WIDTH, PortraitImage as Image, PortraitRefusal as Refusal,
    apply_glare_final_writeback, apply_glare_stage_writeback, bridge_glare_planes,
};
use std::path::PathBuf;

fn sha256(bytes: &[u8]) -> [u8; 32] {
    aurora_portrait::model_sha256::sha256(bytes)
}

fn digest(hex: &str) -> [u8; 32] {
    assert_eq!(hex.len(), 64);
    let mut out = [0u8; 32];
    for (index, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[index * 2..index * 2 + 2], 16).unwrap();
    }
    out
}

fn nhwc4(values: &[[f32; 4]]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|pixel| pixel.iter().flat_map(|value| value.to_le_bytes()))
        .collect()
}

fn planar_sha(material: &[[f32; 4]], mask: &[[f32; 4]]) -> ([u8; 32], [u8; 32]) {
    let mut material_planar = Vec::with_capacity(PF90130_A_PLANAR_BYTES);
    let mut mask_planar = Vec::with_capacity(PF90130_B_PLANAR_BYTES);
    for plane in 0..3 {
        for pixel in material {
            material_planar.extend_from_slice(&pixel[plane].to_le_bytes());
        }
    }
    for pixel in mask {
        mask_planar.extend_from_slice(&pixel[0].to_le_bytes());
    }
    (sha256(&material_planar), sha256(&mask_planar))
}

fn fixture_values() -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let material = (0..PF90130_PIXELS)
        .map(|index| {
            [
                ((index % 251) as f32 / 125.5) - 1.0,
                ((index % 127) as f32 / 63.5) - 1.0,
                ((index % 61) as f32 / 30.5) - 1.0,
                0.0,
            ]
        })
        .collect::<Vec<_>>();
    let mask = (0..PF90130_PIXELS)
        .map(|index| [((index % 101) as f32 / 50.5) - 1.0, 0.0, 0.0, 0.0])
        .collect::<Vec<_>>();
    (material, mask)
}

#[test]
fn exact_planar_bridge_and_stage_writeback_preserve_destination_alpha() {
    let (material_values, mask_values) = fixture_values();
    let material = nhwc4(&material_values);
    let mask = nhwc4(&mask_values);
    let (material_planar_sha256, mask_planar_sha256) = planar_sha(&material_values, &mask_values);
    let receipts = GlarePlanarReceipts {
        material_raw_sha256: sha256(&material),
        mask_raw_sha256: sha256(&mask),
        material_planar_sha256,
        mask_planar_sha256,
    };
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
    let output = apply_glare_stage_writeback(
        &source,
        &destination,
        &GlareWritebackInputs {
            control: Some(1.0),
            source_sha256: Some(sha256(&source.pixels)),
            destination_sha256: Some(sha256(&destination.pixels)),
            planes: GlarePlanarInputs {
                ticket: Some(35),
                material_nhwc4: Some(&material),
                mask_nhwc4: Some(&mask),
                receipts: Some(receipts),
            },
            lens_mask: None,
        },
    )
    .unwrap()
    .expect("PF90130 stage should change the frame");

    assert_eq!(output.ticket, 35);
    assert_eq!(output.planes.material_planar_sha256, material_planar_sha256);
    assert_eq!(output.planes.mask_planar_sha256, mask_planar_sha256);
    for (source, destination) in output
        .rgba8
        .pixels
        .chunks_exact(4)
        .zip(destination.pixels.chunks_exact(4))
    {
        assert_eq!(source[3], destination[3]);
    }
}

#[test]
fn final_worker_is_exact_and_preserves_destination_alpha() {
    let source = Image::new(2, 1, vec![10, 20, 30, 1, 100, 100, 100, 2]).unwrap();
    let destination = Image::new(2, 1, vec![40, 50, 60, 7, 70, 80, 90, 8]).unwrap();
    let residual = [0.0, 0.0, 0.0, 0.5, 1.5, 2.5];
    let output = apply_glare_final_writeback(
        &source,
        &destination,
        &GlareFinalWritebackInputs {
            ticket: Some(99),
            residual_ticket: Some(99),
            source_sha256: Some(sha256(&source.pixels)),
            destination_sha256: Some(sha256(&destination.pixels)),
            residual_rgb_f32: Some(&residual),
        },
    )
    .unwrap();
    assert_eq!(output.ticket, 99);
    assert_eq!(output.rgba8.pixel(0, 0), [40, 50, 60, 7]);
    assert_eq!(output.rgba8.pixel(1, 0), [100, 102, 102, 8]);
}

#[test]
fn formal_artifacts_match_the_authority_planar_sha() {
    let Ok(directory) = std::env::var("PF90130_FORMAL_DIR") else {
        return;
    };
    let directory = PathBuf::from(directory);
    if !directory.join("formal_2977284645.bin").is_file()
        || !directory.join("formal_1615971598.bin").is_file()
    {
        return;
    }
    let material =
        std::fs::read(directory.join("formal_2977284645.bin")).expect("formal A artifact");
    let mask = std::fs::read(directory.join("formal_1615971598.bin")).expect("formal B artifact");
    let bundle = bridge_glare_planes(&GlarePlanarInputs {
        ticket: Some(90130),
        material_nhwc4: Some(&material),
        mask_nhwc4: Some(&mask),
        receipts: Some(GlarePlanarReceipts {
            material_raw_sha256: digest(
                "4750CB899E27AD6B21B96A0DA5C65970462D2CA25576329744B237DC0CD47AC9",
            ),
            mask_raw_sha256: digest(
                "CBE73A9FCBE13B24AF3BA80A984C3350EF9756EADB605AD455850AB803A4CFC6",
            ),
            material_planar_sha256: digest(
                "F2FE8D8BD359880BCE0B1B32F600612CAB6430B844DB65F9EB7811C88DB1459D",
            ),
            mask_planar_sha256: digest(
                "45973969FAEBB8F8BB7F3016CADBC15D106A17E48DA2763E3D30A2AC6A7F2244",
            ),
        }),
    })
    .unwrap();
    assert_eq!(bundle.material_planar.len(), PF90130_A_PLANAR_BYTES);
    assert_eq!(bundle.mask_planar.len(), PF90130_B_PLANAR_BYTES);
}

#[test]
fn bridge_refuses_a_changed_planar_receipt() {
    let (material_values, mask_values) = fixture_values();
    let material = nhwc4(&material_values);
    let mask = nhwc4(&mask_values);
    let (material_planar_sha256, mask_planar_sha256) = planar_sha(&material_values, &mask_values);
    let mut receipts = GlarePlanarReceipts {
        material_raw_sha256: sha256(&material),
        mask_raw_sha256: sha256(&mask),
        material_planar_sha256,
        mask_planar_sha256,
    };
    receipts.mask_planar_sha256[0] ^= 1;
    assert_eq!(
        bridge_glare_planes(&GlarePlanarInputs {
            ticket: Some(35),
            material_nhwc4: Some(&material),
            mask_nhwc4: Some(&mask),
            receipts: Some(receipts),
        }),
        Err(Refusal::ValueOutOfRange("PF90130_MASK_PLANAR_SHA256"))
    );
}
