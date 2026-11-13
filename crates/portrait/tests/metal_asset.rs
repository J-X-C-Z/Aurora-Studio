#![cfg(feature = "external-fixtures")]
#[cfg(feature = "external-fixtures")]
#[path = "support/mod.rs"]
mod external_fixtures;

// Real `whiten.png` atlas checks for the Apple whiten kernel.
//
// The LUT golden is decoded offline by
// `work/backend-r11/o2e/decode_lut.py` into RGBA float32. The MSL reads
// `float4 .x/.y/.z`, so the byte order is RGBA, not BGRA.

use aurora_pixcake::metal::{MetalContext, WhitenInput};

const O3B_MSL: &str = include_str!("../native/metal/o3b_masked_lut_whiten.metal");

fn require_macos() {
    if !cfg!(target_os = "macos") {
        panic!("requires macOS and a Metal device; non-macOS is not a pass");
    }
}

fn read_lut() -> Vec<f32> {
    let bytes = external_fixtures::read("teeth/whiten_lut.f32");
    assert_eq!(bytes.len(), 512 * 512 * 4 * 4);
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn atlas(lut: &[f32], x: usize, y: usize) -> [f32; 4] {
    let i = (y * 512 + x) * 4;
    [lut[i], lut[i + 1], lut[i + 2], lut[i + 3]]
}

fn sample_lut(lut: &[f32], color: [f32; 3]) -> [f32; 3] {
    let pr = color[0].clamp(0.0, 1.0) * 63.0;
    let pg = color[1].clamp(0.0, 1.0) * 63.0;
    let pb = color[2].clamp(0.0, 1.0) * 63.0;
    let (ir, ig, ib) = (pr as usize, pg as usize, pb as usize);
    let (fr, fg, fb) = (pr - ir as f32, pg - ig as f32, pb - ib as f32);
    let (jr, jg, jb) = ((ir + 1).min(63), (ig + 1).min(63), (ib + 1).min(63));
    let mut out = [0.0f32; 3];
    for bz in 0..2 {
        let z = if bz == 0 { ib } else { jb };
        let wz = if bz == 0 { 1.0 - fb } else { fb };
        let y_base = (z >> 3) * 64;
        let x_base = (z & 7) * 64;
        for gy in 0..2 {
            let y = y_base + if gy == 0 { ig } else { jg };
            let wy = if gy == 0 { 1.0 - fg } else { fg };
            for gx in 0..2 {
                let x = x_base + if gx == 0 { ir } else { jr };
                let wx = if gx == 0 { 1.0 - fr } else { fr };
                let weight = wz * wy * wx;
                let texel = atlas(lut, x, y);
                for c in 0..3 {
                    out[c] += texel[c] * weight;
                }
            }
        }
    }
    out
}

fn round_half_even(value: f32) -> u8 {
    let scaled = value * 255.0;
    if !(scaled > 0.0) {
        return 0;
    }
    if scaled >= 255.0 {
        return 255;
    }
    let base = scaled.floor();
    let mut rounded = base as u32;
    let fraction = scaled - base;
    if fraction > 0.5 || (fraction == 0.5 && (rounded & 1) != 0) {
        rounded += 1;
    }
    rounded.min(255) as u8
}

fn oracle(source: &[u8], mask: &[u8], lut: &[f32], alpha: f32) -> Vec<u8> {
    let mut out = source.to_vec();
    for (index, px) in out.chunks_exact_mut(4).enumerate() {
        let weight = mask[index] as f32 / 255.0;
        let a = weight * alpha;
        if !(a > 0.001) {
            continue;
        }
        let color = [
            px[2] as f32 / 255.0,
            px[1] as f32 / 255.0,
            px[0] as f32 / 255.0,
        ];
        let mapped = sample_lut(lut, color);
        let mixed = [
            color[0] * (1.0 - a) + mapped[0] * a,
            color[1] * (1.0 - a) + mapped[1] * a,
            color[2] * (1.0 - a) + mapped[2] * a,
        ];
        px[0] = round_half_even(mixed[2]);
        px[1] = round_half_even(mixed[1]);
        px[2] = round_half_even(mixed[0]);
        // BGRA alpha stays untouched.
    }
    out
}

#[test]
fn lut_asset_is_rgba_float_and_o3b_abi_is_rgba() {
    let lut = read_lut();
    assert_eq!(lut.len(), 512 * 512 * 4);
    assert_eq!(atlas(&lut, 0, 0), [0.0, 0.0, 0.0, 1.0]);
    let red = atlas(&lut, 63, 0);
    assert!((red[0] - 243.0 / 255.0).abs() < 1e-6);
    assert!((red[1] - 31.0 / 255.0).abs() < 1e-6);
    assert!((red[2] - 31.0 / 255.0).abs() < 1e-6);
    assert!((atlas(&lut, 0, 63)[1] - 1.0).abs() < 1e-6);
    assert!((atlas(&lut, 64, 0)[2] - 6.0 / 255.0).abs() < 1e-6);

    // The atlas tile with the strongest red sample proves R is slot 0; a BGRA
    // load would put 31/255 there.
    assert!(red[0] > 0.9 && red[2] < 0.2);

    assert!(O3B_MSL.contains("device const float4 *lut_atlas"));
    assert!(O3B_MSL.contains("texel.x, texel.y, texel.z"));
    assert!(O3B_MSL.contains("source_bgra"));
    assert!(O3B_MSL.contains("static_assert(sizeof(O3MaskedWhitenParams) == 48"));
    assert!(O3B_MSL.contains("destination_bgra[index + 0u]"));
}

#[test]
#[ignore = "requires macOS Metal device"]
fn real_atlas_whiten_matches_cpu_oracle_byte_exact() {
    require_macos();
    let lut = read_lut();
    let width = 8u32;
    let height = 8u32;
    let mut source = Vec::with_capacity((width * height * 4) as usize);
    let mut mask = Vec::with_capacity((width * height) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 31 + 7) % 256) as u8;
            let g = ((y * 29 + 11) % 256) as u8;
            let b = (((x * 13 + y * 17) % 256) as u8).wrapping_add(3);
            let a = (40 + x * 5 + y * 3) as u8;
            source.extend_from_slice(&[b, g, r, a]);
            mask.push(if x == 0 && y == 0 {
                0
            } else if x == 0 {
                128
            } else {
                255
            });
        }
    }

    let alpha = 0.75f32;
    let expected = oracle(&source, &mask, &lut, alpha);
    let input = WhitenInput {
        bgra: &source,
        width,
        height,
        stride_bytes: width * 4,
        mask: &mask,
        mask_x: 0,
        mask_y: 0,
        mask_width: width,
        mask_height: height,
        lut: &lut,
        alpha,
        flags: 0,
        ticket: 0x02E0_0001,
    };
    let actual = MetalContext::new()
        .expect("MetalContext")
        .masked_whiten(&input)
        .expect("whiten dispatch");
    assert_eq!(actual.len(), expected.len());
    for (index, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_eq!(a, e, "byte {index} differs: gpu={a} cpu={e}");
    }
    for (index, px) in actual.chunks_exact(4).enumerate() {
        assert_eq!(px[3], source[index * 4 + 3], "alpha changed at {index}");
    }
}
