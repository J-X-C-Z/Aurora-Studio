//! DNG sensor decode: strips or tiles, uncompressed / PackBits / deflate /
//! lossless JPEG, with the DNG active area and colour metadata.

use std::io::Read;

use super::tiff::{Ifd, Tiff};
use super::{cfa_filters, active_area, make_frame, meta, packbits, NativeError, NativeResult};

use crate::raw::BayerFrame;

/// Upper bound for one sensor plane or one decoded tile. A 150MP sensor needs
/// 150_000_000 samples, so this still admits every real DNG while rejecting the
/// multi-gigabyte allocations a corrupt or hostile header could request.
const MAX_SENSOR_SAMPLES: usize = 250_000_000;

/// Extra bytes a deflate tile may expand to beyond its packed sample size.
const DEFLATE_PADDING: usize = 4096;

pub(crate) fn decode(tiff: &Tiff<'_>, ifd0: &Ifd<'_>) -> NativeResult<BayerFrame> {
    let ifd = super::raw_ifd(tiff)?;
    let raw_width = ifd.u32(0x0100).ok_or_else(|| NativeError::corrupt("DNG has no image width"))? as usize;
    let raw_height = ifd.u32(0x0101).ok_or_else(|| NativeError::corrupt("DNG has no image height"))? as usize;
    if raw_width == 0 || raw_height == 0 || raw_width > 100_000 || raw_height > 100_000 {
        return Err(NativeError::corrupt("DNG dimensions are out of range"));
    }
    if raw_width.saturating_mul(raw_height) > MAX_SENSOR_SAMPLES {
        return Err(NativeError::corrupt("DNG sensor plane is implausibly large"));
    }
    let bits = ifd.u16(0x0102).unwrap_or(16).clamp(8, 16) as u8;
    let samples = ifd.u16(0x0115).unwrap_or(1) as usize;
    if samples != 1 {
        return Err(NativeError::unsupported(format!(
            "linear DNG with {samples} samples per pixel"
        )));
    }
    let compression = ifd.u16(0x0103).unwrap_or(1);
    let predictor = ifd.u16(0x013d).unwrap_or(1);
    let (filters, colors) = cfa_filters(&ifd)?;
    let version = dng_version(ifd0);

    let plane = decode_plane(tiff, &ifd, raw_width, raw_height, bits, compression, predictor, version)?;
    let (left, top, width, height) = active_area(&ifd, raw_width, raw_height);

    let black = black_level(&ifd);
    let maximum = white_level(&ifd).unwrap_or_else(|| (1u32 << bits) - 1);
    // DNG colour tags live in IFD0 for some writers and in the raw SubIFD for
    // others; LibRaw accepts either.
    let cam_mul = as_shot_neutral(&ifd).or_else(|| as_shot_neutral(ifd0)).unwrap_or([0.0; 4]);
    let rgb_cam = color_matrix(&ifd, colors as usize)
        .or_else(|| color_matrix(ifd0, colors as usize))
        .unwrap_or([[0.0f32; 4]; 3]);

    make_frame(
        raw_width,
        raw_height,
        left,
        top,
        width,
        height,
        bits,
        plane,
        filters,
        colors,
        black,
        maximum,
        cam_mul,
        rgb_cam,
    )
}

fn dng_version(ifd0: &Ifd<'_>) -> u32 {
    let Some(Ok(bytes)) = ifd0.tag_bytes(0xc612) else {
        return 0;
    };
    let mut version = 0u32;
    for &byte in bytes.iter().take(4) {
        version = (version << 8) | u32::from(byte);
    }
    version
}

#[allow(clippy::too_many_arguments)]
fn decode_plane(
    tiff: &Tiff<'_>,
    ifd: &Ifd<'_>,
    raw_width: usize,
    raw_height: usize,
    bits: u8,
    compression: u16,
    predictor: u16,
    dng_version: u32,
) -> NativeResult<Vec<u16>> {
    let mut out = vec![0u16; raw_width * raw_height];

    if let Some(tile_offsets) = ifd.u32s(0x0144).and_then(|values| values.ok()) {
        let tile_width = ifd.u32(0x0142).unwrap_or(0) as usize;
        let tile_length = ifd.u32(0x0143).unwrap_or(0) as usize;
        let counts = ifd
            .u32s(0x0145)
            .and_then(|values| values.ok())
            .ok_or_else(|| NativeError::corrupt("DNG tiles have no byte counts"))?;
        if tile_width == 0 || tile_length == 0 || counts.len() < tile_offsets.len() {
            return Err(NativeError::corrupt("DNG tile directory is incomplete"));
        }
        let across = raw_width.div_ceil(tile_width);
        let down = raw_height.div_ceil(tile_length);
        if tile_offsets.len() < across * down {
            return Err(NativeError::corrupt("DNG has fewer tiles than its geometry needs"));
        }
        let mut tile = 0usize;
        for tile_y in 0..down {
            for tile_x in 0..across {
                let offset = tile_offsets[tile] as usize;
                let count = counts[tile] as usize;
                tile += 1;
                let raw = tiff
                    .data()
                    .get(offset..offset.saturating_add(count))
                    .ok_or_else(|| NativeError::corrupt("DNG tile offset is outside the file"))?;
                // LibRaw decodes the full tile (the JPEG stream encodes the
                // padded 256-pixel rows) and clips only when copying.
                let decoded = decode_segment(
                    raw,
                    bits,
                    compression,
                    predictor,
                    tile_width,
                    tile_length,
                    tiff.little,
                    dng_version,
                )?;
                place_tile(
                    &mut out,
                    raw_width,
                    raw_height,
                    tile_x * tile_width,
                    tile_y * tile_length,
                    tile_width,
                    tile_length,
                    &decoded,
                );
            }
        }
        return Ok(out);
    }

    let offsets = ifd
        .u32s(0x0111)
        .and_then(|values| values.ok())
        .ok_or_else(|| NativeError::corrupt("DNG has neither strips nor tiles"))?;
    let counts = ifd
        .u32s(0x0117)
        .and_then(|values| values.ok())
        .ok_or_else(|| NativeError::corrupt("DNG strips have no byte counts"))?;
    if counts.len() < offsets.len() {
        return Err(NativeError::corrupt("DNG strip directory is incomplete"));
    }
    let rows_per_strip = ifd.u32(0x0116).unwrap_or(raw_height as u32).max(1) as usize;
    let mut y = 0usize;
    for (strip, &offset) in offsets.iter().enumerate() {
        if y >= raw_height {
            break;
        }
        let height = rows_per_strip.min(raw_height - y);
        let raw = tiff
            .data()
            .get(offset as usize..(offset as usize).saturating_add(counts[strip] as usize))
            .ok_or_else(|| NativeError::corrupt("DNG strip offset is outside the file"))?;
        let decoded = decode_segment(raw, bits, compression, predictor, raw_width, height, tiff.little, dng_version)?;
        place_segment(&mut out, raw_width, 0, y, raw_width, &decoded);
        y += height;
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
fn decode_segment(
    raw: &[u8],
    bits: u8,
    compression: u16,
    predictor: u16,
    width: usize,
    height: usize,
    little: bool,
    dng_version: u32,
) -> NativeResult<Vec<u16>> {
    let expected = width
        .checked_mul(height)
        .filter(|&samples| samples <= MAX_SENSOR_SAMPLES)
        .ok_or_else(|| NativeError::corrupt("DNG segment is implausibly large"))?;
    // Bytes the samples occupy when packed continuously (the minimum any valid
    // strip must provide). `unpack_bits_msb` already enforces this, but computing
    // it here also bounds the decompressed buffers below.
    let packed = ((expected as u64) * u64::from(bits)).div_ceil(8) as usize;
    // Decompressed byte count the unpackers consume: 16-bit samples are exact,
    // narrower depths are read as whole bytes per sample, so use the larger of
    // the two as the bound (never smaller than what `unpack` reads).
    let target = expected.saturating_mul(usize::from(bits).div_ceil(8)).max(packed);
    let samples = match compression {
        1 => {
            // Uncompressed strips store exactly the packed samples (plus at most
            // the final byte's padding); a shorter strip cannot be valid.
            if raw.len() < packed {
                return Err(NativeError::corrupt("DNG strip is shorter than its samples"));
            }
            unpack(raw, bits, expected, little)?
        }
        32773 => {
            // A PackBits stream producing `target` bytes is at most `target`
            // bytes of literals plus one header byte per 128-byte run, plus any
            // writer padding; anything much longer is garbage, not a valid strip.
            let maximum = target.saturating_add(target.div_ceil(128)).saturating_add(DEFLATE_PADDING);
            if raw.len() > maximum {
                return Err(NativeError::corrupt("DNG PackBits strip is longer than its samples"));
            }
            let expanded = packbits::decode(raw, target)?;
            unpack(&expanded, bits, expected, little)?
        }
        8 | 32946 => {
            let limit = target.saturating_add(DEFLATE_PADDING);
            let mut expanded = Vec::new();
            flate2::read::ZlibDecoder::new(raw)
                .take(limit as u64 + 1)
                .read_to_end(&mut expanded)
                .map_err(|error| NativeError::corrupt(format!("DNG deflate: {error}")))?;
            if expanded.len() > limit {
                return Err(NativeError::corrupt(
                    "DNG deflate expands beyond its declared samples",
                ));
            }
            unpack(&expanded, bits, expected, little)?
        }
        7 => {
            // The DNG quirk: 16-bit difference symbols in DNG 1.1+ mean -32768.
            let dng_bug = dng_version == 0 || dng_version >= 0x0101_0000;
            let image = super::ljpeg::decode(raw, dng_bug)?;
            if image.data.len() < expected {
                return Err(NativeError::corrupt(format!(
                    "lossless DNG tile decoded {} samples, expected {expected}",
                    image.data.len()
                )));
            }
            image.data[..expected].to_vec()
        }
        other => {
            return Err(NativeError::unsupported(format!("DNG compression {other}")));
        }
    };
    // LibRaw does not apply the integer DNG predictor (its deflate path is
    // float-only and `packed_dng_load_raw` ignores tag 0x013d), so neither do we:
    // matching the shipped decoder is the acceptance criterion here.
    let _ = predictor;
    Ok(samples)
}

fn unpack(raw: &[u8], bits: u8, expected: usize, little: bool) -> NativeResult<Vec<u16>> {
    if bits == 16 {
        if raw.len() < expected * 2 {
            return Err(NativeError::corrupt("DNG strip is shorter than its samples"));
        }
        return Ok(raw
            .chunks_exact(2)
            .take(expected)
            .map(|chunk| {
                if little {
                    u16::from_le_bytes([chunk[0], chunk[1]])
                } else {
                    u16::from_be_bytes([chunk[0], chunk[1]])
                }
            })
            .collect());
    }
    meta::unpack_bits_msb(raw, bits, expected)
}

#[allow(clippy::too_many_arguments)]
fn place_tile(
    out: &mut [u16],
    raw_width: usize,
    raw_height: usize,
    x0: usize,
    y0: usize,
    tile_width: usize,
    tile_length: usize,
    samples: &[u16],
) {
    for y in 0..tile_length {
        let dest_y = y0 + y;
        if dest_y >= raw_height {
            break;
        }
        for x in 0..tile_width {
            let dest_x = x0 + x;
            if dest_x >= raw_width {
                break;
            }
            if let Some(value) = samples.get(y * tile_width + x) {
                out[dest_y * raw_width + dest_x] = *value;
            }
        }
    }
}

fn place_segment(out: &mut [u16], raw_width: usize, x0: usize, y0: usize, width: usize, samples: &[u16]) {
    for (index, &value) in samples.iter().enumerate() {
        let x = x0 + index % width;
        let y = y0 + index / width;
        if let Some(slot) = out.get_mut(y * raw_width + x) {
            *slot = value;
        }
    }
}

fn black_level(ifd: &Ifd<'_>) -> u32 {
    let mut black = 0u32;
    if let Some(Ok(levels)) = ifd.reals(0xc61a) {
        let repeat = ifd
            .u16s(0xc619)
            .and_then(|values| values.ok())
            .map(|values| (values.first().copied().unwrap_or(1) as usize) * (values.get(1).copied().unwrap_or(1) as usize))
            .unwrap_or(1);
        if levels.len() == 1 && repeat < 2 {
            // LibRaw truncates the stored float (`unsigned(dng_fblack)`).
            black = levels[0].max(0.0) as u32;
        } else {
            // Per-CFA-position black levels live in `cblack`, which `BayerFrame`
            // does not carry; LibRaw reports `black == 0` for this case too.
            black = 0;
        }
    }
    for tag in [0xc61b, 0xc61c] {
        if let Some(Ok(deltas)) = ifd.reals(tag) {
            if !deltas.is_empty() {
                let mean = deltas.iter().sum::<f64>() / deltas.len() as f64;
                black = black.saturating_add(mean.round() as u32);
            }
        }
    }
    black
}

fn white_level(ifd: &Ifd<'_>) -> Option<u32> {
    let levels = ifd.reals(0xc61d).and_then(|values| values.ok())?;
    let value = *levels.first()?;
    (value > 0.0).then(|| value.round() as u32)
}

fn as_shot_neutral(ifd: &Ifd<'_>) -> Option<[f32; 4]> {
    let Ok(values) = ifd.reals(0xc628)? else {
        return None;
    };
    if values.len() < 3 || values[0] == 0.0 {
        return None;
    }
    let mut cam_mul = [0.0f32; 4];
    for index in 0..3 {
        if values[index].abs() > 0.000_1 {
            // LibRaw stores AsShotNeutral as float, then divides in float.
            cam_mul[index] = 1.0f32 / (values[index] as f32);
        }
    }
    // LibRaw zeroes the second-green slot when AsShotNeutral is present.
    cam_mul[3] = 0.0;
    Some(cam_mul)
}

fn color_matrix(ifd: &Ifd<'_>, colors: usize) -> Option<[[f32; 4]; 3]> {
    // LibRaw parses ColorMatrix1 then ColorMatrix2 into the same slot, so the
    // last one present in IFD order wins.
    let entry = ifd
        .entries
        .iter()
        .rev()
        .find(|entry| entry.tag == 0xc621 || entry.tag == 0xc622)?;
    let values = ifd.reals(entry.tag)?.ok()?;
    let channels = if values.len() == 9 {
        3
    } else if values.len() == 12 {
        4
    } else {
        return None;
    };
    let mut cam_xyz = [[0.0f64; 3]; 4];
    for channel in 0..channels {
        for component in 0..3 {
            // LibRaw keeps the parsed ColorMatrix in `float` inside
            // `dng_color`; the float rounding is visible in `rgb_cam`'s last ULPs.
            cam_xyz[channel][component] = values[channel * 3 + component] as f32 as f64;
        }
    }
    Some(meta::cam_xyz_to_rgb_cam(&cam_xyz, colors.min(channels)))
}

#[cfg(test)]
mod tests {
    fn entry(_tiff: &[u8], tag: u16, typ: u16, count: u32, value: [u8; 4]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(tag.to_le_bytes());
        out.extend(typ.to_le_bytes());
        out.extend(count.to_le_bytes());
        out.extend(value);
        out
    }

    /// Builds a little-endian TIFF with IFD0 as the DNG raw IFD.
    pub(crate) fn build_dng(width: u16, height: u16, bits: u16, compression: u16, pixels: &[u16]) -> Vec<u8> {
        let mut raw = Vec::with_capacity(pixels.len() * 2);
        for &value in pixels {
            raw.extend(value.to_le_bytes());
        }
        if compression == 32773 {
            let mut packed = Vec::new();
            for chunk in raw.chunks(128) {
                packed.push((chunk.len() - 1) as u8);
                packed.extend_from_slice(chunk);
            }
            raw = packed;
        } else if compression == 8 {
            use std::io::Write;
            let mut encoder =
                flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
            encoder.write_all(&raw).unwrap();
            raw = encoder.finish().unwrap();
        }
        let mut data = Vec::new();
        data.extend(b"II\x2a\0\x08\0\0\0");
        let strip = data.len() as u32;
        data.extend_from_slice(&raw);
        let mut entries = Vec::new();
        entries.extend(entry(&data, 0x0100, 4, 1, u32::from(width).to_le_bytes()));
        entries.extend(entry(&data, 0x0101, 4, 1, u32::from(height).to_le_bytes()));
        entries.extend(entry(&data, 0x0102, 3, 1, {
            let mut value = [0u8; 4];
            value[..2].copy_from_slice(&bits.to_le_bytes());
            value
        }));
        entries.extend(entry(&data, 0x0103, 3, 1, {
            let mut value = [0u8; 4];
            value[..2].copy_from_slice(&compression.to_le_bytes());
            value
        }));
        entries.extend(entry(&data, 0x0106, 3, 1, {
            let mut value = [0u8; 4];
            value[..2].copy_from_slice(&32803u16.to_le_bytes());
            value
        }));
        entries.extend(entry(&data, 0x0111, 4, 1, strip.to_le_bytes()));
        entries.extend(entry(
            &data,
            0x0115,
            3,
            1,
            {
                let mut value = [0u8; 4];
                value[..2].copy_from_slice(&1u16.to_le_bytes());
                value
            },
        ));
        entries.extend(entry(&data, 0x0116, 4, 1, u32::from(height).to_le_bytes()));
        entries.extend(entry(
            &data,
            0x0117,
            4,
            1,
            (u32::try_from(raw.len()).unwrap()).to_le_bytes(),
        ));
        entries.extend(entry(
            &data,
            0x828d,
            3,
            2,
            {
                let mut value = [0u8; 4];
                value[..2].copy_from_slice(&2u16.to_le_bytes());
                value[2..].copy_from_slice(&2u16.to_le_bytes());
                value
            },
        ));
        // CFAPattern lives inline (4 bytes) and is stored as BYTE.
        entries.extend(entry(&data, 0x828e, 1, 4, [0, 1, 1, 2]));
        // DNGVersion 1.4.0.0 inline.
        entries.extend(entry(&data, 0xc612, 1, 4, [1, 4, 0, 0]));

        let ifd_at = data.len() as u32;
        let entry_count = (entries.len() / 12) as u16;
        data.extend(entry_count.to_le_bytes());
        data.extend_from_slice(&entries);
        data.extend(0u32.to_le_bytes());
        // Header IFD offset.
        data[4..8].copy_from_slice(&ifd_at.to_le_bytes());
        data
    }

    #[test]
    fn decodes_packbits_dng_plane() {
        let pixels: Vec<u16> = (0..8 * 4).map(|index| 200 + index as u16).collect();
        let bytes = build_dng(8, 4, 16, 32773, &pixels);
        let frame = super::super::decode_bytes(&bytes).unwrap();
        assert_eq!(frame.data, pixels);
    }

    #[test]
    fn decodes_deflate_dng_plane() {
        let pixels: Vec<u16> = (0..8 * 4).map(|index| 300 + index as u16).collect();
        let bytes = build_dng(8, 4, 16, 8, &pixels);
        let frame = super::super::decode_bytes(&bytes).unwrap();
        assert_eq!(frame.data, pixels);
    }

    #[test]
    fn decodes_uncompressed_dng_plane() {
        let pixels: Vec<u16> = (0..4 * 3).map(|index| 100 + index as u16).collect();
        let bytes = build_dng(4, 3, 16, 1, &pixels);
        let frame = super::super::decode_bytes(&bytes).unwrap();
        assert_eq!((frame.raw_width, frame.raw_height), (4, 3));
        assert_eq!((frame.width, frame.height), (4, 3));
        assert_eq!(frame.filters, 0xb4b4_b4b4);
        assert_eq!(frame.data, pixels);
        assert_eq!(frame.maximum, 0xffff);
    }

    #[test]
    fn rejects_implausible_sensor_plane() {
        // 20000x20000 = 400M samples, above MAX_SENSOR_SAMPLES, so the header is
        // refused before any plane allocation.
        let bytes = build_dng(20_000, 20_000, 16, 1, &[]);
        let error = super::super::decode_bytes(&bytes).unwrap_err();
        assert!(matches!(error, super::NativeError::Corrupt(_)), "{error}");
    }

    #[test]
    fn rejects_deflate_that_expands_beyond_its_samples() {
        use std::io::Write;
        let mut encoder = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
        // 1 MiB of zeros for a 4x4 tile (32 sample bytes): far past the limit.
        encoder.write_all(&vec![0u8; 1 << 20]).unwrap();
        let bomb = encoder.finish().unwrap();
        let error = super::decode_segment(&bomb, 16, 8, 1, 4, 4, true, 0x0104_0000).unwrap_err();
        assert!(matches!(error, super::NativeError::Corrupt(_)), "{error}");
    }

    #[test]
    fn rejects_oversized_packbits_strip() {
        // A 4x4 16-bit tile needs 32 bytes; a 5000-byte PackBits strip is far
        // longer than any encoding of it could be.
        let error = super::decode_segment(&vec![0u8; 5000], 16, 32773, 1, 4, 4, true, 0x0104_0000)
            .unwrap_err();
        assert!(matches!(error, super::NativeError::Corrupt(_)), "{error}");
    }
}
