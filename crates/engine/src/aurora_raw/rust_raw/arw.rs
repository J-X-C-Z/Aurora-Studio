//! Sony ARW sensor decode: ARW2 block compression and uncompressed planes.
//!
//! ARW2 stores 16 pixels in 16 bytes: two 11-bit anchors plus fourteen 7-bit
//! deltas at a shared shift.  Values are doubled (`curve[pix << 1]`, identity
//! curve) exactly as LibRaw's `sony_arw2_load_raw` does.

use rayon::prelude::*;

use super::tiff::{Ifd, Tiff};
use super::{cfa_filters, make_frame, meta, NativeError, NativeResult};

use crate::raw::BayerFrame;

pub(crate) fn decode(tiff: &Tiff<'_>, ifd0: &Ifd<'_>) -> NativeResult<BayerFrame> {
    let ifd = super::raw_ifd(tiff)?;
    let raw_width = ifd.u32(0x0100).ok_or_else(|| NativeError::corrupt("ARW has no image width"))? as usize;
    let raw_height = ifd.u32(0x0101).ok_or_else(|| NativeError::corrupt("ARW has no image height"))? as usize;
    if raw_width == 0 || raw_height == 0 || raw_width > 100_000 || raw_height > 100_000 {
        return Err(NativeError::corrupt("ARW dimensions are out of range"));
    }
    let mut bits = ifd.u16(0x0102).unwrap_or(14).clamp(8, 16) as u8;
    let compression = ifd.u16(0x0103).unwrap_or(1);
    let (filters, colors) = cfa_filters(&ifd)?;
    let make = ifd0.ascii(0x010f).unwrap_or_default();
    let model = ifd0.ascii(0x0110).unwrap_or_default();
    if filters == 9 {
        return Err(NativeError::unsupported("X-Trans sensor in an ARW container"));
    }
    let offset = ifd
        .u32s(0x0111)
        .and_then(|values| values.ok())
        .and_then(|values| values.first().copied())
        .ok_or_else(|| NativeError::corrupt("ARW has no strip offset"))? as usize;
    let count = ifd
        .u32s(0x0117)
        .and_then(|values| values.ok())
        .and_then(|values| values.first().copied())
        .unwrap_or_else(|| (tiff.data().len().saturating_sub(offset)) as u32) as usize;
    let strip = tiff
        .data()
        .get(offset..offset.saturating_add(count))
        .ok_or_else(|| NativeError::corrupt("ARW strip is outside the file"))?;

    let data = match (compression, count) {
        (32767, bytes) if bytes == raw_width * raw_height => {
            bits = 14;
            let curve = sony_curve(&ifd);
            decode_arw2(strip, raw_width, raw_height, curve.as_deref())?
        }
        (32767, bytes) if bytes >= raw_width * raw_height * 2 => {
            meta::unpack_bits_le16(strip, raw_width * raw_height)?
        }
        (1, _) => {
            // Sony's uncompressed ARW stores the 14 significant bits in 16-bit
            // words (LibRaw's `unpacked_load_raw`), not as packed 14-bit.
            if bits == 16 || count >= raw_width * raw_height * 2 {
                meta::unpack_bits_le16(strip, raw_width * raw_height)
                    .or_else(|_| meta::unpack_bits_msb(strip, 16, raw_width * raw_height))?
            } else {
                meta::unpack_bits_msb(strip, bits, raw_width * raw_height)?
            }
        }
        (32766, _) => return Err(NativeError::unsupported("Sony ARW6 lightweight compression")),
        (6, _) => return Err(NativeError::unsupported("Sony ARW1 compression")),
        (other, _) => return Err(NativeError::unsupported(format!("Sony ARW compression {other}"))),
    };

    let (left, top, width, height) = geometry(&ifd, raw_width, raw_height, &model);
    let black = sony_black(&ifd).or_else(|| black_tag(&ifd)).unwrap_or_else(|| {
        if raw_width > 3888 {
            if compression == 32767 && count == raw_width * raw_height {
                512
            } else {
                128u32 << (u32::from(bits).saturating_sub(12))
            }
        } else {
            0
        }
    });
    let maximum = ifd.u32(0xc61d).unwrap_or_else(|| (1u32 << bits.min(16)) - 1);
    let cam_mul = sony_cam_mul(&ifd);
    let rgb_cam = meta::adobe_matrix(&make, &model)
        .map(|matrix| meta::cam_xyz_to_rgb_cam(&matrix, colors as usize))
        .unwrap_or([[0.0f32; 4]; 3]);

    make_frame(
        raw_width,
        raw_height,
        left,
        top,
        width,
        height,
        bits,
        data,
        filters,
        colors,
        black,
        maximum,
        cam_mul,
        rgb_cam,
    )
}

fn geometry(
    ifd: &Ifd<'_>,
    raw_width: usize,
    raw_height: usize,
    model: &str,
) -> (usize, usize, usize, usize) {
    // LibRaw's Sony width trims from identify.cpp; the active area always starts
    // at column 0 and keeps the full height.
    let mut width = raw_width;
    let mut height = raw_height;
    let upper = model.to_ascii_uppercase();
    let is_7m4 = upper.starts_with("ILCE-7M4")
        || upper.starts_with("ILCE-7CM2")
        || upper.starts_with("ILME-FX2");
    if is_7m4 {
        match (raw_width, raw_height) {
            (7168, 5120) => {
                width = 7028;
                height = 4688;
            }
            (4736, _) => width = 4692,
            (5120, _) => {
                width = 4624;
                height = 3080;
            }
            (3584, _) => {
                width = 3516;
                height = 2344;
            }
            (7040, _) => width = width.saturating_sub(12),
            _ => {}
        }
    }
    match raw_width {
        4288 => width = width.saturating_sub(32),
        4928 if raw_height < 3280 => width = width.saturating_sub(8),
        5504 => width = width.saturating_sub(if raw_height > 3664 { 8 } else { 32 }),
        6048 => {
            width = width.saturating_sub(24);
            if upper.contains("RX1") || upper.contains("A99") {
                width = width.saturating_sub(6);
            }
        }
        7392 => width = width.saturating_sub(30),
        8000 => width = width.saturating_sub(32),
        9600 => width = width.saturating_sub(32),
        _ => {}
    }
    if width > 0 && width <= raw_width && height > 0 && height <= raw_height
        && (width != raw_width || height != raw_height)
    {
        return (0, 0, width, height);
    }
    super::active_area(ifd, raw_width, raw_height)
}

fn sony_black(ifd: &Ifd<'_>) -> Option<u32> {
    let levels = ifd.u16s(0x7310)?.ok()?;
    if levels.is_empty() {
        return None;
    }
    Some(u32::from(*levels.iter().min()?))
}

fn sony_cam_mul(ifd: &Ifd<'_>) -> [f32; 4] {
    let Some(Ok(raw)) = ifd.tag_bytes(0x7313) else {
        return [0.0; 4];
    };
    if raw.len() < 8 {
        return [0.0; 4];
    }
    let mut values = [0i32; 4];
    for (index, chunk) in raw.chunks_exact(2).take(4).enumerate() {
        values[index] = i32::from(i16::from_le_bytes([chunk[0], chunk[1]]));
    }
    // RGGB_2_RGBG: file order R,G,G2,B becomes cam_mul R,G,B,G2.
    [values[0] as f32, values[1] as f32, values[3] as f32, values[2] as f32]
}

/// LibRaw's `curve` built from Sony tag 0x7010: piecewise increments 1/2/4/8.
fn sony_curve(ifd: &Ifd<'_>) -> Option<Vec<u16>> {
    let values = ifd.u16s(0x7010)?.ok()?;
    if values.len() < 4 {
        return None;
    }
    // LibRaw's default sentinel: the last segment ends at 4095.
    let mut stops = [0i32; 6];
    stops[5] = 4095;
    for (index, &value) in values.iter().take(4).enumerate() {
        stops[index + 1] = i32::from((value >> 2) & 0x0fff);
    }
    // ARW2 indexes `curve[pix << 1]` where `pix <= 0x7ff`, so 4096 entries
    // cover every possible lookup while fitting in L1 cache.
    let mut curve: Vec<u16> = (0..4096u16).collect();
    for step in 0..5 {
        let (start, end) = (stops[step], stops[step + 1]);
        let mut at = start + 1;
        while at <= end {
            if at > 0 && (at as usize) < curve.len() {
                curve[at as usize] = curve[(at - 1) as usize].wrapping_add(1 << step);
            }
            at += 1;
        }
    }
    Some(curve)
}

fn black_tag(ifd: &Ifd<'_>) -> Option<u32> {
    let levels = ifd.reals(0xc61a)?.ok()?;
    (levels.len() == 1 && levels[0] > 0.0).then(|| levels[0].round() as u32)
}

fn decode_arw2(
    strip: &[u8],
    raw_width: usize,
    raw_height: usize,
    curve: Option<&[u16]>,
) -> NativeResult<Vec<u16>> {
    if strip.len() < raw_width.saturating_mul(raw_height) {
        return Err(NativeError::corrupt("ARW2 strip is shorter than one byte per pixel"));
    }
    let mut out = vec![0u16; raw_width * raw_height];
    let decode_row = |row: usize, out_row: &mut [u16]| {
        let row_data = &strip[row * raw_width..(row + 1) * raw_width];
        let mut block = [0u16; 16];
        let mut data_at = 0usize;
        let mut col = 0usize;
        while col + 30 < raw_width && data_at + 16 <= row_data.len() {
            let bytes = &row_data[data_at..data_at + 16];
            let packed = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let maximum = packed & 0x7ff;
            let minimum = (packed >> 11) & 0x7ff;
            let maximum_index = ((packed >> 22) & 0x0f) as usize;
            let minimum_index = ((packed >> 26) & 0x0f) as usize;
            let mut shift = 0u32;
            let range = maximum.saturating_sub(minimum);
            while shift < 4 && (0x80u32 << shift) <= range {
                shift += 1;
            }
            let mut bit = 30u32;
            for (index, slot) in block.iter_mut().enumerate() {
                if index == maximum_index {
                    *slot = maximum as u16;
                } else if index == minimum_index {
                    *slot = minimum as u16;
                } else {
                    let at = (bit >> 3) as usize;
                    let offset = bit & 7;
                    // LibRaw reads past the 16-byte block into the next one;
                    // the row buffer is contiguous, so do the same.
                    let pos = data_at + at;
                    let word = u16::from_le_bytes([
                        *row_data.get(pos).unwrap_or(&0),
                        *row_data.get(pos + 1).unwrap_or(&0),
                    ]);
                    let value = (((u32::from(word) >> offset) & 0x7f) << shift) + minimum;
                    *slot = value.min(0x7ff) as u16;
                    bit += 7;
                }
            }
            for &value in &block {
                let index = usize::from(value) << 1;
                let scaled = curve.and_then(|curve| curve.get(index)).copied().unwrap_or(index as u16);
                if let Some(slot) = out_row.get_mut(col) {
                    *slot = scaled;
                }
                col += 2;
            }
            data_at += 16;
            if col & 1 != 0 {
                col -= 1;
            } else {
                col -= 31;
            }
        }
    };
    if raw_height >= 32 {
        out.par_chunks_exact_mut(raw_width)
            .enumerate()
            .for_each(|(row, out_row)| decode_row(row, out_row));
    } else {
        for (row, out_row) in out.chunks_exact_mut(raw_width).enumerate() {
            decode_row(row, out_row);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_a_single_arw2_block() {
        // 16 pixels, all equal to 100: max = min = 100, deltas unused.
        let mut block = [0u8; 16];
        let packed = 100u32 | (100u32 << 11) | (0u32 << 22) | (1u32 << 26);
        block[..4].copy_from_slice(&packed.to_le_bytes());
        let mut row = vec![0u8; 64];
        row[..16].copy_from_slice(&block);
        let mut strip = vec![0u8; 64 * 2];
        strip[..64].copy_from_slice(&row);
        let out = decode_arw2(&strip, 64, 2, None).unwrap();
        let expected: Vec<u16> = (0..32).map(|index| if index % 2 == 0 { 200 } else { 0 }).collect();
        assert_eq!(&out[..32], expected);
    }
}
