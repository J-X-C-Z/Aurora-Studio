#[cfg(feature = "external-fixtures")]
#[path = "support/mod.rs"]
mod external_fixtures;

// Real-atlas fixtures for the fixed-point Metal SkinAdjust LUT channel.
//
// Windows runs hash/CPU-oracle checks. The GPU test is `#[ignore]` and panics
// on non-macOS instead of passing.

#[cfg(feature = "external-fixtures")]
use std::fs;
#[cfg(feature = "external-fixtures")]
use std::path::PathBuf;

#[cfg(feature = "external-fixtures")]
use aurora_pixcake::kernels::makeup;
use aurora_pixcake::metal::SkinLutParams;
#[cfg(feature = "external-fixtures")]
use aurora_pixcake::metal::MetalContext;
#[cfg(feature = "external-fixtures")]
use aurora_pixcake::Image;

#[cfg(feature = "external-fixtures")]

fn require_macos() {
    if !cfg!(target_os = "macos") {
        panic!("requires macOS and a Metal device; non-macOS is not a pass");
    }
}

#[cfg(feature = "external-fixtures")]

fn fixture_root() -> PathBuf {
    external_fixtures::root().join("metal_skin")
}

#[cfg(feature = "external-fixtures")]

fn atlas(name: &str) -> Vec<u8> {
    fs::read(fixture_root().join(format!("atlases/{name}.rgb8"))).expect("atlas")
}

#[cfg(feature = "external-fixtures")]

struct GoldenCase {
    atlas: String,
    name: String,
    width: u32,
    height: u32,
    mix: f32,
    source: Vec<u8>,
    mask: Vec<u8>,
    expected: Vec<u8>,
}

#[cfg(feature = "external-fixtures")]

fn cases() -> Vec<GoldenCase> {
    let root = fixture_root();
    let index = fs::read_to_string(root.join("index.tsv")).expect("index.tsv");
    let mut out = Vec::new();
    for line in index.lines().skip(1).filter(|line| !line.trim().is_empty()) {
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 5, "bad row: {line}");
        let dir = root.join("cases").join(fields[1]);
        out.push(GoldenCase {
            atlas: fields[0].to_string(),
            name: fields[1].to_string(),
            width: fields[2].parse().unwrap(),
            height: fields[3].parse().unwrap(),
            mix: fields[4].parse().unwrap(),
            source: fs::read(dir.join("source.rgba8")).unwrap(),
            mask: fs::read(dir.join("mask.u8")).unwrap(),
            expected: fs::read(dir.join("expected.rgba8")).unwrap(),
        });
    }
    out
}

#[cfg(feature = "external-fixtures")]

fn cpu(case: &GoldenCase, lut: &[u8]) -> Image {
    makeup::skin_adjust_lut(
        &Image::new(case.width, case.height, case.source.clone()).unwrap(),
        &case.mask,
        lut,
        case.mix,
    )
    .unwrap()
}

// Minimal SHA-256 for fixture integrity; test-only, no dependency.
#[cfg(feature = "external-fixtures")]
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut message = data.to_vec();
    let bit_len = (data.len() as u64) * 8;
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_be_bytes());
    for chunk in message.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut v = h;
        for i in 0..64 {
            let s1 = v[4].rotate_right(6) ^ v[4].rotate_right(11) ^ v[4].rotate_right(25);
            let ch = (v[4] & v[5]) ^ (!v[4] & v[6]);
            let t1 = v[7]
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = v[0].rotate_right(2) ^ v[0].rotate_right(13) ^ v[0].rotate_right(22);
            let maj = (v[0] & v[1]) ^ (v[0] & v[2]) ^ (v[1] & v[2]);
            let t2 = s0.wrapping_add(maj);
            v[7] = v[6];
            v[6] = v[5];
            v[5] = v[4];
            v[4] = v[3].wrapping_add(t1);
            v[3] = v[2];
            v[2] = v[1];
            v[1] = v[0];
            v[0] = t1.wrapping_add(t2);
        }
        for i in 0..8 {
            h[i] = h[i].wrapping_add(v[i]);
        }
    }
    h.iter().map(|v| format!("{v:08X}")).collect()
}

#[cfg(feature = "external-fixtures")]

fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

#[cfg(feature = "external-fixtures")]

fn float_lut(lut: &[u8], color: [f32; 3]) -> [f32; 3] {
    let pr = clamp01(color[0]) * 63.0;
    let pg = clamp01(color[1]) * 63.0;
    let pb = clamp01(color[2]) * 63.0;
    let (ir, ig, ib) = (pr as usize, pg as usize, pb as usize);
    let (fr, fg, fb) = (pr - ir as f32, pg - ig as f32, pb - ib as f32);
    let (jr, jg, jb) = ((ir + 1).min(63), (ig + 1).min(63), (ib + 1).min(63));
    let mut out = [0.0f32; 3];
    for bz in 0..2 {
        let z = if bz == 0 { ib } else { jb };
        let wz = if bz == 0 { 1.0 - fb } else { fb };
        let yb = (z >> 3) * 64;
        let xb = (z & 7) * 64;
        for gy in 0..2 {
            let y = yb + if gy == 0 { ig } else { jg };
            let wy = if gy == 0 { 1.0 - fg } else { fg };
            for gx in 0..2 {
                let x = xb + if gx == 0 { ir } else { jr };
                let wx = if gx == 0 { 1.0 - fr } else { fr };
                let weight = wz * wy * wx;
                for c in 0..3 {
                    out[c] += (lut[(y * 512 + x) * 3 + c] as f32 / 255.0) * weight;
                }
            }
        }
    }
    out
}

#[cfg(feature = "external-fixtures")]

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

#[cfg(feature = "external-fixtures")]

fn float_byte(source: &[u8], mask: &[u8], lut: &[u8], mix: f32, pixel: usize) -> u8 {
    let weight = (mask[pixel] as f32 / 255.0) * mix;
    if !(weight > 0.001) {
        return source[pixel * 4 + 1];
    }
    let px = &source[pixel * 4..pixel * 4 + 4];
    let color = [
        px[0] as f32 / 255.0,
        px[1] as f32 / 255.0,
        px[2] as f32 / 255.0,
    ];
    let mapped = float_lut(lut, color);
    round_half_even(color[1] + weight * (mapped[1] - color[1]))
}

#[test]
fn params_are_32_bytes() {
    assert_eq!(std::mem::size_of::<SkinLutParams>(), 32);
}

#[test]
#[cfg(feature = "external-fixtures")]
fn fixture_hashes_and_cpu_expected_match() {
    let root = fixture_root();
    for line in fs::read_to_string(root.join("SHA256SUMS")).unwrap().lines() {
        let (expected, relative) = line.split_once("  ").unwrap();
        let data = fs::read(root.join(relative)).unwrap();
        assert_eq!(sha256_hex(&data), expected, "{relative}");
    }
    let whiten = atlas("whiten");
    let ruddy = atlas("ruddy");
    for case in cases() {
        let lut = if case.atlas == "whiten" {
            &whiten
        } else {
            &ruddy
        };
        let out = cpu(&case, lut);
        assert_eq!(out.pixels, case.expected, "{} expected", case.name);
        for i in 0..(case.width * case.height) as usize {
            assert_eq!(out.pixels[i * 4 + 3], case.source[i * 4 + 3]);
        }
    }
}

#[test]
#[cfg(feature = "external-fixtures")]
fn real_whiten_index5_is_fixed_153_not_float_154() {
    let whiten = atlas("whiten");
    let source = vec![
        27, 153, 110, 226, 11, 153, 64, 114, 94, 124, 214, 187, 206, 191, 246, 235,
    ];
    let mask = [84u8, 68, 36, 248];
    let mix = 0.08983356f32;
    let image = Image::new(2, 2, source.clone()).unwrap();
    let fixed = makeup::skin_adjust_lut(&image, &mask, &whiten, mix).unwrap();
    assert_eq!(fixed.pixels[5], 153);
    assert_eq!(float_byte(&source, &mask, &whiten, mix, 1), 154);
}

#[test]
#[ignore = "requires macOS Metal device"]
#[cfg(feature = "external-fixtures")]
fn real_atlas_metal_matches_fixture_expected() {
    require_macos();
    let context = MetalContext::new().expect("MetalContext");
    let whiten = atlas("whiten");
    let ruddy = atlas("ruddy");
    for case in cases() {
        let lut = if case.atlas == "whiten" {
            &whiten
        } else {
            &ruddy
        };
        let image = Image::new(case.width, case.height, case.source.clone()).unwrap();
        let gpu = context
            .skin_lut_fixed(&image, &case.mask, lut, case.mix)
            .expect("skin_lut_fixed dispatch");
        assert_eq!(gpu.pixels, case.expected, "{} GPU mismatch", case.name);
        for i in 0..(case.width * case.height) as usize {
            assert_eq!(
                gpu.pixels[i * 4 + 3],
                case.source[i * 4 + 3],
                "{} alpha",
                case.name
            );
        }
    }
    let source = vec![
        27, 153, 110, 226, 11, 153, 64, 114, 94, 124, 214, 187, 206, 191, 246, 235,
    ];
    let mask = [84u8, 68, 36, 248];
    let image = Image::new(2, 2, source).unwrap();
    let gpu = context
        .skin_lut_fixed(&image, &mask, &whiten, 0.08983356)
        .expect("discriminator dispatch");
    assert_eq!(gpu.pixels[5], 153, "GPU must use fixed-point path");
}
