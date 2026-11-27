//! Zero-dependency Rust generator, exporter, and SHA-256 verifier for the
//! 7 portrait prior/LUT asset categories.
//!
//! Only deterministic mathematical color-science LUTs and parametric geometric
//! priors are generated. Neural model slots are emitted solely as JSON
//! contracts (`requires_real_dataset_training: true`) with zero fabricated weights.

use std::env;
use std::f64::consts::PI;
use std::fs;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// SHA-256 (FIPS 180-4)
// ---------------------------------------------------------------------------

fn sha256(bytes: &[u8]) -> [u8; 32] {
    const H0: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity((bytes.len() + 72) & !63);
    padded.extend_from_slice(bytes);
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut h = H0;
    for block in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (chunk, word) in out.chunks_exact_mut(4).zip(h) {
        chunk.copy_from_slice(&word.to_be_bytes());
    }
    out
}

fn sha256_hex_upper(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789ABCDEF";
    let digest = sha256(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push(DIGITS[(b >> 4) as usize] as char);
        out.push(DIGITS[(b & 0x0f) as usize] as char);
    }
    out
}

// ---------------------------------------------------------------------------
// Minimal deterministic PNG encoder (uncompressed zlib stored blocks)
// ---------------------------------------------------------------------------

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

fn adler32(bytes: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut a = 1u32;
    let mut b = 0u32;
    for &byte in bytes {
        a = (a + byte as u32) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

fn zlib_store(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + 64);
    out.push(0x78);
    out.push(0x01);
    let chunks: Vec<&[u8]> = if data.is_empty() {
        vec![&[]]
    } else {
        data.chunks(65535).collect()
    };
    for (idx, chunk) in chunks.iter().enumerate() {
        let is_last = idx + 1 == chunks.len();
        out.push(if is_last { 0x01 } else { 0x00 });
        let len = chunk.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

fn png_chunk(tag: &[u8; 4], data: &[u8], out: &mut Vec<u8>) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(tag);
    out.extend_from_slice(data);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

fn encode_png_rgba(width: u32, height: u32, rgba: &[u8]) -> Vec<u8> {
    assert_eq!(rgba.len(), (width as usize) * (height as usize) * 4);
    let stride = (width as usize) * 4;
    let mut raw = Vec::with_capacity((stride + 1) * (height as usize));
    for y in 0..height as usize {
        raw.push(0);
        raw.extend_from_slice(&rgba[y * stride..(y + 1) * stride]);
    }
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&width.to_be_bytes());
    ihdr.extend_from_slice(&height.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]);

    let idat = zlib_store(&raw);
    let mut out = Vec::with_capacity(idat.len() + 64);
    out.extend_from_slice(b"\x89PNG\r\n\x1a\n");
    png_chunk(b"IHDR", &ihdr, &mut out);
    png_chunk(b"IDAT", &idat, &mut out);
    png_chunk(b"IEND", &[], &mut out);
    out
}

// ---------------------------------------------------------------------------
// 7 Portrait Prior/LUT Asset Categories
// ---------------------------------------------------------------------------

fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

fn to_u8(x: f64) -> u8 {
    (clamp01(x) * 255.0).round() as u8
}

fn build_hald_lut_512<F: Fn(f64, f64, f64) -> (f64, f64, f64)>(f: F) -> Vec<u8> {
    let mut buf = vec![0u8; 512 * 512 * 4];
    for b in 0..64 {
        let tile_x = (b % 8) * 64;
        let tile_y = (b / 8) * 64;
        let fb = b as f64 / 63.0;
        for g in 0..64 {
            let fg = g as f64 / 63.0;
            let y = tile_y + g;
            for r in 0..64 {
                let fr = r as f64 / 63.0;
                let x = tile_x + r;
                let (nr, ng, nb) = f(fr, fg, fb);
                let idx = (y * 512 + x) * 4;
                buf[idx] = to_u8(nr);
                buf[idx + 1] = to_u8(ng);
                buf[idx + 2] = to_u8(nb);
                buf[idx + 3] = 255;
            }
        }
    }
    encode_png_rgba(512, 512, &buf)
}

#[derive(Clone)]
struct AssetRecord {
    category: &'static str,
    file: String,
    bytes: usize,
    sha256: String,
    asset_kind: &'static str,
    requires_real_dataset_training: bool,
}

fn write_asset(
    root: &Path,
    category: &'static str,
    rel: &str,
    payload: &[u8],
    asset_kind: &'static str,
    requires_training: bool,
    records: &mut Vec<AssetRecord>,
) -> std::io::Result<()> {
    let path = root.join(rel);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&path, payload)?;
    records.push(AssetRecord {
        category,
        file: rel.replace('\\', "/"),
        bytes: payload.len(),
        sha256: sha256_hex_upper(payload),
        asset_kind,
        requires_real_dataset_training: requires_training,
    });
    Ok(())
}

fn generate_all(out_dir: &Path) -> std::io::Result<Vec<AssetRecord>> {
    fs::create_dir_all(out_dir)?;
    let mut records = Vec::new();

    // 1. skin_adjust (6 x 512x512 Hald 3D LUTs)
    let skin_specs: [(&str, fn(f64, f64, f64) -> (f64, f64, f64)); 6] = [
        ("skin-adjust/whiten.png", |r, g, b| (r.powf(0.86), g.powf(0.88), b.powf(0.85))),
        ("skin-adjust/ruddy_new.png", |r, g, b| (clamp01(r * 1.08 + 0.03 * (1.0 - r)), g * 0.96, b * 0.95)),
        ("skin-adjust/TintGreen.png", |r, g, b| (r * 0.94, clamp01(g * 1.06), b * 0.96)),
        ("skin-adjust/TintMagenta.png", |r, g, b| (clamp01(r * 1.05), g * 0.93, clamp01(b * 1.05))),
        ("skin-adjust/cold.png", |r, g, b| (r * 0.92, g * 0.99, clamp01(b * 1.08 + 0.02 * (1.0 - b)))),
        ("skin-adjust/warm.png", |r, g, b| (clamp01(r * 1.08 + 0.02 * (1.0 - r)), clamp01(g * 1.02), b * 0.90)),
    ];
    for (rel, fn_ptr) in skin_specs {
        let png = build_hald_lut_512(fn_ptr);
        write_asset(out_dir, "skin_adjust", rel, &png, "prior_lut", false, &mut records)?;
    }

    // 2. radiance (SkinRadiance priors)
    let mut lut_a = Vec::with_capacity(76800);
    for row in 0..300 {
        let t = row as f64 / 299.0;
        for col in 0..64 {
            let u = col as f64 / 63.0;
            let v = to_u8(clamp01(u + 0.08 * t * (PI * u).sin()));
            lut_a.extend_from_slice(&[v, v, v, 255]);
        }
    }
    write_asset(out_dir, "radiance", "radiance/prior-lut-a.bin", &lut_a, "prior_lut", false, &mut records)?;

    let mut lut_b = Vec::with_capacity(38400);
    for row in 0..150 {
        let t = row as f64 / 149.0;
        for col in 0..64 {
            let u = col as f64 / 63.0;
            let s = u * u * (3.0 - 2.0 * u);
            let v = to_u8(u * (1.0 - 0.15 * t) + s * (0.15 * t));
            lut_b.extend_from_slice(&[v, v, v, 255]);
        }
    }
    write_asset(out_dir, "radiance", "radiance/prior-lut-b.bin", &lut_b, "prior_lut", false, &mut records)?;

    let mut gamma = Vec::with_capacity(65536);
    for i in 0..4096 {
        let x = i as f64 / 4095.0;
        let lin = (if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }) as f32;
        for _ in 0..3 {
            gamma.extend_from_slice(&lin.to_le_bytes());
        }
        gamma.extend_from_slice(&1.0f32.to_le_bytes());
    }
    write_asset(out_dir, "radiance", "radiance/prior-gamma-rgba32f.bin", &gamma, "prior_lut", false, &mut records)?;

    let identity_2x3x3: [f32; 18] = [
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
        1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    ];
    let mut mats = Vec::with_capacity(72);
    for v in identity_2x3x3 {
        mats.extend_from_slice(&v.to_le_bytes());
    }
    write_asset(out_dir, "radiance", "radiance/prior-color-matrices.bin", &mats, "prior_lut", false, &mut records)?;

    // 3. teeth (Whiten, Bright, toothMask)
    let teeth_whiten = build_hald_lut_512(|r, g, b| {
        let luma = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        (
            clamp01(r * 0.78 + luma * 0.22 + 0.03 * (1.0 - luma)),
            clamp01(g * 0.80 + luma * 0.20 + 0.04 * (1.0 - luma)),
            clamp01(b * 0.72 + luma * 0.28 + 0.07 * (1.0 - luma)),
        )
    });
    write_asset(out_dir, "teeth", "teeth/Whiten.png", &teeth_whiten, "prior_lut", false, &mut records)?;
    let teeth_bright = build_hald_lut_512(|r, g, b| (r.powf(0.82), g.powf(0.82), b.powf(0.80)));
    write_asset(out_dir, "teeth", "teeth/Bright.png", &teeth_bright, "prior_lut", false, &mut records)?;

    let (mw, mh) = (111usize, 43usize);
    let mut mask = vec![0u8; mw * mh * 4];
    for y in 0..mh {
        let ny = (y as f64 - (mh as f64 - 1.0) * 0.5) / ((mh as f64 - 1.0) * 0.5);
        for x in 0..mw {
            let nx = (x as f64 - (mw as f64 - 1.0) * 0.5) / ((mw as f64 - 1.0) * 0.5);
            let v = to_u8((1.0 - (nx * nx + ny * ny)).max(0.0));
            let idx = (y * mw + x) * 4;
            mask[idx..idx + 4].copy_from_slice(&[v, v, v, 255]);
        }
    }
    let mask_png = encode_png_rgba(mw as u32, mh as u32, &mask);
    write_asset(out_dir, "teeth", "teeth/toothMask.png", &mask_png, "prior_geometry", true, &mut records)?;

    // 4. lip_luts (256x2 RGBA8 blend & level weight LUTs)
    let (lw, lh) = (256usize, 2usize);
    let mut blend = vec![0u8; lw * lh * 4];
    let mut level = vec![0u8; lw * lh * 4];
    for x in 0..lw {
        let t = x as f64 / 255.0;
        let smooth = t * t * (3.0 - 2.0 * t);
        let i0 = x * 4;
        blend[i0..i0 + 4].copy_from_slice(&[to_u8(smooth), 0, 0, 255]);
        let i1 = (lw + x) * 4;
        let gloss = clamp01(0.5 * smooth + 0.5 * t.powf(1.5));
        blend[i1..i1 + 4].copy_from_slice(&[0, to_u8(gloss), to_u8(gloss), 255]);
        let lvl = clamp01(0.25 + 0.75 * (PI * t).sin());
        level[i0..i0 + 4].copy_from_slice(&[to_u8(lvl), 0, 0, 255]);
        level[i1..i1 + 4].copy_from_slice(&[0, 0, 0, 255]);
    }
    write_asset(out_dir, "lip_luts", "lip_luts/blend_weight_lut.png", &encode_png_rgba(256, 2, &blend), "prior_lut", false, &mut records)?;
    write_asset(out_dir, "lip_luts", "lip_luts/level_weight_lut.png", &encode_png_rgba(256, 2, &level), "prior_lut", false, &mut records)?;

    // 5. smile (parametric 3D mouth-arch basis)
    let (rows, cols) = (4u32, 16u32);
    let (mut mean, mut normal, mut side_l, mut side_r, mut tris) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for r in 0..rows {
        let v = (r as f32 / (rows - 1) as f32) - 0.5;
        for c in 0..cols {
            let u = (c as f32 / (cols - 1) as f32) * 2.0 - 1.0;
            for f in [u, v * 0.25, 0.0f32] {
                mean.extend_from_slice(&f.to_le_bytes());
            }
            for f in [0.0f32, 0.0, 1.0] {
                normal.extend_from_slice(&f.to_le_bytes());
            }
            let lift = (u * u).max(0.0) * 0.15;
            for f in [if u < 0.0 { -0.05 } else { 0.0 }, if u < 0.0 { lift } else { 0.0 }, 0.0f32] {
                side_l.extend_from_slice(&f.to_le_bytes());
            }
            for f in [if u > 0.0 { 0.05 } else { 0.0 }, if u > 0.0 { lift } else { 0.0 }, 0.0f32] {
                side_r.extend_from_slice(&f.to_le_bytes());
            }
        }
    }
    for r in 0..rows - 1 {
        for c in 0..cols - 1 {
            let a = r * cols + c;
            let b = a + 1;
            let c0 = (r + 1) * cols + c;
            let d0 = c0 + 1;
            for idx in [a, b, c0, b, d0, c0] {
                tris.extend_from_slice(&idx.to_le_bytes());
            }
        }
    }
    write_asset(out_dir, "smile", "smile/mean.f32le", &mean, "prior_geometry", true, &mut records)?;
    write_asset(out_dir, "smile", "smile/normal.f32le", &normal, "prior_geometry", true, &mut records)?;
    write_asset(out_dir, "smile", "smile/side_left.f32le", &side_l, "prior_geometry", true, &mut records)?;
    write_asset(out_dir, "smile", "smile/side_right.f32le", &side_r, "prior_geometry", true, &mut records)?;
    write_asset(out_dir, "smile", "smile/triangles.u32le", &tris, "prior_geometry", true, &mut records)?;

    // 6. face_mesh_priors
    for (rel, amp) in [
        ("face_mesh_priors/face_reshape_parts.f32le", 0.08f32),
        ("face_mesh_priors/face_width_parts.f32le", 0.10f32),
        ("face_mesh_priors/forehead_parts.f32le", 0.06f32),
    ] {
        let mut buf = Vec::with_capacity(64 * 8);
        for i in 0..64 {
            let theta = (i as f64 / 64.0) * 2.0 * PI;
            buf.extend_from_slice(&(amp * theta.cos() as f32).to_le_bytes());
            buf.extend_from_slice(&(amp * theta.sin() as f32).to_le_bytes());
        }
        write_asset(out_dir, "face_mesh_priors", rel, &buf, "prior_geometry", true, &mut records)?;
    }

    // 7. neural_weight_slots (contract only, zero fabricated weights)
    let contract = concat!(
        "{\n",
        "  \"schema\": \"aurora-neural-weight-slots/1\",\n",
        "  \"note\": \"Contracts only. No synthetic or fabricated model weights are included.\",\n",
        "  \"slots\": [\n",
        "    {\"id\": \"face_detection\", \"format\": \"onnx\", \"input_shape\": [1, 3, 256, 256], \"requires_real_dataset_training\": true},\n",
        "    {\"id\": \"face_mesh\", \"format\": \"onnx\", \"input_shape\": [1, 3, 192, 192], \"requires_real_dataset_training\": true},\n",
        "    {\"id\": \"face_parsing_v2\", \"format\": \"onnx\", \"input_shape\": [1, 3, 512, 512], \"requires_real_dataset_training\": true},\n",
        "    {\"id\": \"blemish_pf90116\", \"format\": \"onnx\", \"input_shape\": [1, 3, 512, 512], \"requires_real_dataset_training\": true},\n",
        "    {\"id\": \"eye_bag_pf90116\", \"format\": \"onnx\", \"input_shape\": [1, 3, 256, 256], \"requires_real_dataset_training\": true},\n",
        "    {\"id\": \"body_pose\", \"format\": \"onnx\", \"input_shape\": [1, 3, 256, 192], \"requires_real_dataset_training\": true},\n",
        "    {\"id\": \"portrait_matting\", \"format\": \"onnx\", \"input_shape\": [1, 3, 512, 512], \"requires_real_dataset_training\": true}\n",
        "  ]\n",
        "}\n"
    );
    write_asset(
        out_dir,
        "neural_weight_slots",
        "neural_weight_slots/neural_slots_contract.json",
        contract.as_bytes(),
        "neural_weight_slot",
        true,
        &mut records,
    )?;

    // Write manifest.json and SHA256SUMS
    let mut manifest = String::from("{\n  \"schema\": \"aurora-portrait-priors-manifest/1\",\n  \"categories\": 7,\n  \"assets\": [\n");
    let mut sums = String::new();
    for (idx, r) in records.iter().enumerate() {
        let comma = if idx + 1 == records.len() { "" } else { "," };
        manifest.push_str(&format!(
            "    {{\"category\": \"{}\", \"file\": \"{}\", \"bytes\": {}, \"sha256\": \"{}\", \"asset_kind\": \"{}\", \"requires_real_dataset_training\": {}}}{}\n",
            r.category, r.file, r.bytes, r.sha256, r.asset_kind, r.requires_real_dataset_training, comma
        ));
        sums.push_str(&format!("{}  {}\n", r.sha256.to_ascii_lowercase(), r.file));
    }
    manifest.push_str("  ]\n}\n");
    fs::write(out_dir.join("manifest.json"), manifest)?;
    fs::write(out_dir.join("SHA256SUMS"), sums)?;
    Ok(records)
}

fn verify_dir(dir: &Path) -> Result<(), String> {
    let sums = fs::read_to_string(dir.join("SHA256SUMS"))
        .map_err(|e| format!("failed to read SHA256SUMS: {e}"))?;
    let mut count = 0usize;
    for line in sums.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let (expected_hex, rel) = trimmed
            .split_once("  ")
            .ok_or_else(|| format!("malformed SHA256SUMS line: {trimmed}"))?;
        let data = fs::read(dir.join(rel))
            .map_err(|e| format!("missing asset {rel}: {e}"))?;
        let actual_hex = sha256_hex_upper(&data).to_ascii_lowercase();
        if actual_hex != expected_hex.to_ascii_lowercase() {
            return Err(format!("sha256 mismatch for {rel}: expected {expected_hex}, got {actual_hex}"));
        }
        count += 1;
    }
    if count == 0 {
        return Err("SHA256SUMS contained 0 entries".into());
    }
    println!("verified {count} portrait prior/LUT assets in {}", dir.display());
    Ok(())
}

fn run_tests() -> Result<(), String> {
    assert_eq!(
        sha256_hex_upper(b"abc"),
        "BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"
    );
    let tmp = env::temp_dir().join(format!("aurora-portrait-priors-test-{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    let records = generate_all(&tmp).map_err(|e| format!("generate_all failed: {e}"))?;
    assert_eq!(records.len(), 24, "expected 24 assets across 7 categories");
    verify_dir(&tmp)?;

    // Determinism check: re-generate into a second dir and compare SHA-256 of manifest.json
    let tmp2 = env::temp_dir().join(format!("aurora-portrait-priors-test2-{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp2);
    generate_all(&tmp2).map_err(|e| format!("generate_all 2 failed: {e}"))?;
    assert_eq!(
        sha256_hex_upper(&fs::read(tmp.join("SHA256SUMS")).unwrap()),
        sha256_hex_upper(&fs::read(tmp2.join("SHA256SUMS")).unwrap()),
        "generation must be byte-for-byte deterministic"
    );

    // Tamper detection check
    fs::write(tmp.join("lip_luts/blend_weight_lut.png"), b"corrupted").unwrap();
    assert!(verify_dir(&tmp).is_err(), "tampered asset must fail SHA-256 verification");

    let _ = fs::remove_dir_all(&tmp);
    let _ = fs::remove_dir_all(&tmp2);
    println!("all portrait_priors self-tests passed");
    Ok(())
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut out_dir: Option<PathBuf> = None;
    let mut verify_target: Option<PathBuf> = None;
    let mut test_mode = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                i += 1;
                out_dir = args.get(i).map(PathBuf::from);
            }
            "--verify" => {
                i += 1;
                verify_target = args.get(i).map(PathBuf::from);
            }
            "--test" => test_mode = true,
            other => {
                eprintln!("unknown argument: {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    if test_mode {
        if let Err(err) = run_tests() {
            eprintln!("TEST FAILED: {err}");
            std::process::exit(1);
        }
        return;
    }
    if let Some(dir) = out_dir {
        match generate_all(&dir) {
            Ok(records) => println!("generated {} assets across 7 categories in {}", records.len(), dir.display()),
            Err(err) => {
                eprintln!("GENERATE FAILED: {err}");
                std::process::exit(1);
            }
        }
    }
    if let Some(dir) = verify_target {
        if let Err(err) = verify_dir(&dir) {
            eprintln!("VERIFY FAILED: {err}");
            std::process::exit(1);
        }
    }
}
