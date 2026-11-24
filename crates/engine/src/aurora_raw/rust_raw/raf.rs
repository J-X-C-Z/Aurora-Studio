//! Fujifilm RAF container reader: uncompressed 16-bit and packed 14-bit
//! sensor planes (including the 6x6 X-Trans CFA).
//!
//! RAF starts with a fixed 128-byte header, followed by a little-endian TIFF
//! header whose raw IFD supplies the sensor size, bit depth and data offset.
//! The Fuji directory at header+92 supplies the crop, CFA layout, black and
//! white balance tags.  All offsets and lengths are bounds checked; an
//! unrecognised packing or a layout change returns `Unsupported` rather than
//! producing a plausible-looking but wrong image.

use super::tiff::{Ifd, Tiff, TYPE_LONG, TYPE_SHORT};
use super::{make_frame, meta, NativeError, NativeResult};
use crate::raw::BayerFrame;

const HEADER_SIZE: usize = 128;
const FUJI_DIR_OFFSET: usize = 92;
const DATA_OFFSET: usize = 100;
const SHORT: u16 = TYPE_SHORT;

/// Parameters from the Fuji directory (header+92).
#[derive(Clone, Debug)]
struct FujiInfo {
    /// Raw sensor rectangle in the Fuji directory; may be a superset of TIFF.
    raw_width: usize,
    raw_height: usize,
    /// Active area after Fuji's geometry rules.
    width: usize,
    height: usize,
    left: usize,
    top: usize,
    /// X-Trans CFA in absolute sensor coordinates (0=R, 1=G, 2=B).
    xtrans: Option<[[u8; 6]; 6]>,
    /// Per-channel black levels in RGBG order.
    black_rgbg: [u32; 4],
    /// As-shot multipliers in RGBG order.
    cam_mul_rgbg: [u32; 4],
    /// RAFData generation (non-zero when the size tag is available).
    rafdata_generation: bool,
    /// White level from the Fuji directory, when present.
    maximum: Option<u32>,
}

pub(crate) fn decode(bytes: &[u8]) -> NativeResult<BayerFrame> {
    if !bytes.starts_with(b"FUJIFILMCCD-RAW") || bytes.len() < HEADER_SIZE + 8 {
        return Err(NativeError::corrupt("not a Fujifilm RAF container"));
    }

    let model = ascii_field(bytes, 0x1c, 0x20);
    // The RAF header is big-endian; offset 100 (`data_offset` in dcraw/LibRaw)
    // points at the embedded TIFF header. IFD and strip offsets are relative to
    // that header.
    let tiff_base = read_be_u32(bytes, DATA_OFFSET)? as usize;
    let tiff = Tiff::parse(bytes, tiff_base)?;
    let first_ifd = tiff.u32_at(tiff_base + 4)?;
    let ifd0 = tiff.ifd(first_ifd)?;
    let (ifd, bits) = raw_ifd(&tiff, &ifd0, first_ifd)?;
    let fuji = parse_fuji_dir(bytes, &model)?;

    // The Fuji directory can declare a slightly larger sensor than the TIFF
    // IFD; LibRaw takes the larger of the two before applying crop rules.
    let raw_width = fuji.raw_width.max(ifd.u32(0x0100).unwrap_or(0) as usize);
    let raw_height = fuji.raw_height.max(ifd.u32(0x0101).unwrap_or(0) as usize);
    if !(1..=0xffff).contains(&raw_width) || !(1..=0xffff).contains(&raw_height) {
        return Err(NativeError::corrupt("RAF sensor dimensions are invalid"));
    }
    if bits != 14 && bits != 16 {
        return Err(NativeError::unsupported(format!("RAF sample depth {bits}")));
    }

    let fuji = apply_geometry(fuji, raw_width, raw_height, &model);

    let compression = ifd.u16(0x0103).unwrap_or(1);
    // StripByteCounts decides whether LibRaw's Fuji 14-bit packing applies.
    let data_size = ifd.u32(0x0117).unwrap_or(0) as usize;
    let plane_samples = raw_width.saturating_mul(raw_height);
    let packed_14_size = plane_samples.saturating_mul(7) / 4;
    let raw_16_size = plane_samples.saturating_mul(2);
    let strip_offset = strip_offset(&ifd, &tiff)?;
    let data_offset = strip_offset.unwrap_or(tiff_base);
    let available = bytes.len().saturating_sub(data_offset);
    if data_offset >= bytes.len() {
        return Err(NativeError::corrupt("RAF image data starts outside the file"));
    }

    let data = if compression != 1 {
        // Fuji's entropy-coded RAF (0x014d and friends) is a separate codec that
        // is not implemented here; refuse instead of mis-reading packed pixels.
        return Err(NativeError::unsupported(format!(
            "Fuji RAF compression {compression:#x}"
        )));
    } else if data_size == raw_16_size {
        if available < raw_16_size {
            return Err(NativeError::corrupt("RAF 16-bit plane is truncated"));
        }
        unpack16(bytes, data_offset, plane_samples, tiff.little)?
    } else if data_size == packed_14_size || (data_size == 0 && bits == 14) {
        if available < packed_14_size {
            return Err(NativeError::corrupt("RAF 14-bit plane is truncated"));
        }
        unpack14(bytes, data_offset, raw_width, raw_height)?
    } else if bits == 16 {
        return Err(NativeError::unsupported(
            "RAF 16-bit sensor packing is not the verified 16-in-16 layout",
        ));
    } else {
        return Err(NativeError::unsupported(
            "Fuji compressed RAF payload is not implemented",
        ));
    };

    let filters = if fuji.xtrans.is_some() { 9 } else { 0x9494_9494 };
    let colors = 3u32;
    // LibRaw averages the two readings per channel for the 16-byte tag, then
    // reports the smallest per-channel level as `black`.
    let black = fuji.black_rgbg.iter().copied().min().unwrap_or(0);
    let maximum = fuji.maximum.unwrap_or((1u32 << bits) - 1);
    let cam_mul = fuji.cam_mul_rgbg.map(|value| value as f32);
    let rgb_cam = meta::adobe_matrix("Fujifilm", &model)
        .map(|matrix| meta::cam_xyz_to_rgb_cam(&matrix, colors as usize))
        .unwrap_or([[0.0; 4]; 3]);

    make_frame(
        raw_width,
        raw_height,
        fuji.left,
        fuji.top,
        fuji.width,
        fuji.height,
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

/// Selects the raw IFD and its bit depth.  RAF has a JPEG preview IFD first;
/// the sensor IFD is the one with a single 14/16-bit sample per pixel.
fn raw_ifd<'a>(tiff: &Tiff<'a>, ifd0: &Ifd<'a>, first_ifd: u32) -> NativeResult<(Ifd<'a>, u8)> {
    let mut best: Option<(Ifd<'a>, u8)> = None;
    let mut candidates = vec![ifd0.clone()];
    candidates.extend(ifd0.sub_ifds().into_iter().filter_map(|offset| tiff.ifd(offset).ok()));
    for chain in tiff.chain(first_ifd).unwrap_or_default() {
        candidates.extend(chain.sub_ifds().into_iter().filter_map(|offset| tiff.ifd(offset).ok()));
        candidates.push(chain);
    }
    for ifd in candidates {
        let bits = ifd.u16(0x0102).unwrap_or(0) as u8;
        let samples = ifd.u16(0x0115).unwrap_or(1);
        let compression = ifd.u16(0x0103).unwrap_or(1);
        let width = ifd.u32(0x0100).unwrap_or(0) as usize;
        let height = ifd.u32(0x0101).unwrap_or(0) as usize;
        if samples != 1 || !matches!(bits, 14 | 16) || compression == 7 || width == 0 || height == 0 {
            continue;
        }
        let rank = (width.saturating_mul(height), bits);
        if best.as_ref().is_none_or(|(old, _)| {
            let old_rank = (
                old.u32(0x0100).unwrap_or(0) as usize * old.u32(0x0101).unwrap_or(0) as usize,
                old.u16(0x0102).unwrap_or(0) as u8,
            );
            rank > old_rank
        }) {
            best = Some((ifd, bits));
        }
    }
    best.ok_or_else(|| NativeError::unsupported("RAF has no 14/16-bit single-sample sensor IFD"))
}

fn strip_offset(ifd: &Ifd<'_>, tiff: &Tiff<'_>) -> NativeResult<Option<usize>> {
    let Some(entry) = ifd.entry(0x0111) else {
        return Ok(None);
    };
    if entry.count != 1 {
        return Ok(None);
    }
    let offset = ifd.bytes(entry)?;
    let value = if entry.typ == SHORT && offset.len() >= 2 {
        u16::from_le_bytes([offset[0], offset[1]]) as usize
    } else if entry.typ == TYPE_LONG && offset.len() >= 4 {
        u32::from_le_bytes([offset[0], offset[1], offset[2], offset[3]]) as usize
    } else {
        return Err(NativeError::unsupported(format!("RAF strip offset type {}", entry.typ)));
    };
    // dcraw/LibRaw store StripOffsets relative to the embedded TIFF header.
    tiff.absolute(value).map(Some)
}

fn parse_fuji_dir(bytes: &[u8], _model: &str) -> NativeResult<FujiInfo> {
    let mut info = FujiInfo {
        raw_width: 0,
        raw_height: 0,
        width: 0,
        height: 0,
        left: 0,
        top: 0,
        xtrans: None,
        black_rgbg: [0; 4],
        cam_mul_rgbg: [0; 4],
        rafdata_generation: false,
        maximum: None,
    };
    // Byte 92 holds a big-endian pointer to the Fuji directory; the directory
    // itself is another big-endian block (LibRaw reads it before parse_tiff
    // switches the stream to little-endian).
    let dir_at = read_be_u32(bytes, FUJI_DIR_OFFSET)? as usize;
    let count = read_be_u32(bytes, dir_at)? as usize;
    if count == 0 || count > 255 {
        return Err(NativeError::corrupt("RAF Fuji directory has an invalid entry count"));
    }
    let mut at = dir_at + 4;
    let mut raw_inset = None;
    let mut crop = None;
    for _ in 0..count {
        // The Fuji directory is big-endian, like the RAF header.
        let tag = read_be_u16(bytes, at)?;
        let len = read_be_u16(bytes, at + 2)? as usize;
        let value_at = at + 4;
        let end = value_at.checked_add(len).filter(|&end| end <= bytes.len())
            .ok_or_else(|| NativeError::corrupt("RAF Fuji directory value is truncated"))?;
        match tag {
            0x0100 if len >= 4 => {
                // RawImageFullSize: height, width (big-endian in the Fuji dir).
                info.raw_height = read_be_u16(bytes, value_at)? as usize;
                info.raw_width = read_be_u16(bytes, value_at + 2)? as usize;
            }
            0x0110 if len >= 4 => raw_inset = Some((read_be_u16(bytes, value_at)? as usize, read_be_u16(bytes, value_at + 2)? as usize)),
            0x0111 if len >= 4 => crop = Some((read_be_u16(bytes, value_at)? as usize, read_be_u16(bytes, value_at + 2)? as usize)),
            0x0121 if len >= 4 => {
                info.height = read_be_u16(bytes, value_at)? as usize;
                let mut width = read_be_u16(bytes, value_at + 2)? as usize;
                if width == 4284 {
                    width += 3;
                }
                info.width = width;
            }
            0x0130 if len >= 2 => {
                // FujiLayout: the high bit of byte 0 and bit 3 of byte 1 mark
                // the SuperCCD rotated layouts, which are out of scope here.
                if bytes[value_at] & 0x80 != 0 || bytes[value_at + 1] & 0x08 != 0 {
                    return Err(NativeError::unsupported("RAF SuperCCD rotated layout"));
                }
            }
            0x0131 if len >= 36 => {
                // XTransLayout: bytes are stored in reverse order, values are
                // clamped to 0..=2.
                let mut xtrans = [[0u8; 6]; 6];
                for index in 0..36 {
                    let color = bytes[value_at + index].min(2);
                    let reverse = 35 - index;
                    xtrans[reverse / 6][reverse % 6] = color;
                }
                info.xtrans = Some(xtrans);
            }
            0x2ff0 if len >= 8 => {
                for channel in 0..4 {
                    let grgb = channel ^ 1;
                    info.cam_mul_rgbg[grgb] = read_be_u16(bytes, value_at + channel * 2)? as u32;
                }
                if len >= 16 {
                    for channel in 0..4 {
                        let grgb = channel ^ 1;
                        let extra = read_be_u16(bytes, value_at + 8 + channel * 2)? as u32;
                        info.cam_mul_rgbg[grgb] = (info.cam_mul_rgbg[grgb] + extra) / 2;
                    }
                }
            }
            0x4000 if len == 8 || len == 16 => {
                for channel in 0..4 {
                    let grgb = channel ^ 1;
                    info.black_rgbg[grgb] = read_be_u16(bytes, value_at + channel * 2)? as u32;
                }
                if len == 16 {
                    for channel in 0..4 {
                        let grgb = channel ^ 1;
                        info.black_rgbg[grgb] =
                            (info.black_rgbg[grgb] + read_be_u16(bytes, value_at + 8 + channel * 2)? as u32) / 2;
                    }
                }
            }
            0xc000 if len > 20_000 => info.rafdata_generation = true,
            _ => {}
        }
        at = end;
    }

    if let Some((top, left)) = raw_inset {
        let active = crop.unwrap_or((info.height, info.width));
        info.width = active.1;
        info.height = active.0;
        info.top = top;
        info.left = left;
    }
    Ok(info)
}

/// Applies LibRaw's Fuji crop rules to the parsed directory values.
fn apply_geometry(mut info: FujiInfo, raw_width: usize, raw_height: usize, model: &str) -> FujiInfo {
    if info.width == 0 || info.height == 0 || info.width > raw_width || info.height > raw_height {
        info.width = raw_width;
        info.height = raw_height;
    }
    if info.rafdata_generation {
        let (width, height) = rafdata_geometry(raw_width, model);
        info.width = width;
        if height != 0 {
            info.height = height;
        }
    }
    top_and_left(&mut info, raw_width, raw_height, model);
    // X-Trans frames are cropped in 6-sample steps so the CFA stays aligned.
    if info.xtrans.is_some() {
        let new_top = if info.top % 6 == 0 { info.top } else { (info.top / 6 + 1) * 6 };
        if new_top != info.top {
            info.height = info.height.saturating_sub(new_top - info.top);
            info.top = new_top;
        }
        let new_left = if info.left % 6 == 0 { info.left } else { (info.left / 6 + 1) * 6 };
        if new_left != info.left {
            info.width = info.width.saturating_sub(new_left - info.left);
            info.left = new_left;
        }
    }
    info
}

fn top_and_left(info: &mut FujiInfo, raw_width: usize, raw_height: usize, _model: &str) {
    info.top = ((raw_height - info.height) >> 2) << 1;
    info.left = ((raw_width - info.width) >> 2) << 1;
    // Compressed/uncompressed variants of the 24MP and 16MP X bodies share a
    // sensor but differ in the padding that LibRaw drops.
    if matches!(raw_width, 6048 | 6160 | 4096 | 5120) {
        info.left = 0;
    } else if raw_width == 4992 {
        info.left = 4;
    }
    if raw_width == 7872 {
        info.top = 6;
        info.left = 0;
    } else if raw_width == 6336 {
        info.top = 6;
        info.left = 0;
    } else if raw_width == 6384 {
        info.top = 6;
        info.left = 0;
    } else if raw_width == 8472 {
        info.left = 0;
    } else if matches!(raw_width, 9696 | 11808) {
        info.top = 2;
        info.left = 0;
    }
}

fn rafdata_geometry(raw_width: usize, _model: &str) -> (usize, usize) {
    match raw_width {
        7872 => (7752, 5178),
        6336 => (6264, 4176),
        6384 => (6246, 4170),
        8472 => (raw_width - 192, 0),
        9696 | 11808 => (raw_width - 146, 0),
        _ => (raw_width, 0),
    }
}

fn unpack16(bytes: &[u8], offset: usize, count: usize, little: bool) -> NativeResult<Vec<u16>> {
    let end = offset.checked_add(count.checked_mul(2).ok_or_else(|| NativeError::corrupt("RAF plane overflow"))?)
        .filter(|&end| end <= bytes.len())
        .ok_or_else(|| NativeError::corrupt("RAF 16-bit plane is truncated"))?;
    Ok(bytes[offset..end]
        .chunks_exact(2)
        .map(|pair| if little {
            u16::from_le_bytes([pair[0], pair[1]])
        } else {
            u16::from_be_bytes([pair[0], pair[1]])
        })
        .collect())
}

/// Unpacks Fuji's packed 14-bit rows exactly like LibRaw's
/// `fuji_14bit_load_raw`: rows whose byte length is a multiple of 28 use the
/// 28-byte/16-sample layout, all other rows are 32-bit byte-swapped first and
/// then decoded with the 7-byte/4-sample layout.
fn unpack14(bytes: &[u8], offset: usize, width: usize, height: usize) -> NativeResult<Vec<u16>> {
    let linelen = width.checked_mul(7).ok_or_else(|| NativeError::corrupt("RAF row overflow"))? / 4;
    let pitch = width;
    let mut out = vec![0u16; width * height];
    for row in 0..height {
        let start = row
            .checked_mul(linelen)
            .and_then(|row_start| offset.checked_add(row_start))
            .ok_or_else(|| NativeError::corrupt("RAF 14-bit row overflow"))?;
        let line = bytes
            .get(start..start + linelen)
            .ok_or_else(|| NativeError::corrupt("RAF 14-bit plane is truncated"))?;
        if linelen % 28 != 0 {
            let mut swapped = line.to_vec();
            for word in swapped.chunks_exact_mut(4) {
                word.reverse();
            }
            let (mut sp, mut dp) = (0usize, 0usize);
            while dp + 3 < pitch && sp + 6 < linelen {
                unpack7(&swapped[sp..sp + 7], &mut out[row * width + dp..]);
                sp += 7;
                dp += 4;
            }
        } else {
            let (mut sp, mut dp) = (0usize, 0usize);
            while dp + 15 < pitch && sp + 27 < linelen {
                unpack28(&line[sp..sp + 28], &mut out[row * width + dp..]);
                sp += 28;
                dp += 16;
            }
        }
    }
    Ok(out)
}

#[inline]
fn unpack7(src: &[u8], dest: &mut [u16]) {
    let s = |index: usize| u16::from(src[index]);
    dest[0] = (s(0) << 6) | (s(1) >> 2);
    dest[1] = ((s(1) & 0x03) << 12) | (s(2) << 4) | (s(3) >> 4);
    dest[2] = ((s(3) & 0x0f) << 10) | (s(4) << 2) | (s(5) >> 6);
    dest[3] = ((s(5) & 0x3f) << 8) | s(6);
}

#[inline]
fn unpack28(src: &[u8], dest: &mut [u16]) {
    let s = |index: usize| u16::from(src[index]);
    dest[0] = (s(3) << 6) | (s(2) >> 2);
    dest[1] = ((s(2) & 0x03) << 12) | (s(1) << 4) | (s(0) >> 4);
    dest[2] = ((s(0) & 0x0f) << 10) | (s(7) << 2) | (s(6) >> 6);
    dest[3] = ((s(6) & 0x3f) << 8) | s(5);
    dest[4] = (s(4) << 6) | (s(11) >> 2);
    dest[5] = ((s(11) & 0x03) << 12) | (s(10) << 4) | (s(9) >> 4);
    dest[6] = ((s(9) & 0x0f) << 10) | (s(8) << 2) | (s(15) >> 6);
    dest[7] = ((s(15) & 0x3f) << 8) | s(14);
    dest[8] = (s(13) << 6) | (s(12) >> 2);
    dest[9] = ((s(12) & 0x03) << 12) | (s(19) << 4) | (s(18) >> 4);
    dest[10] = ((s(18) & 0x0f) << 10) | (s(17) << 2) | (s(16) >> 6);
    dest[11] = ((s(16) & 0x3f) << 8) | s(23);
    dest[12] = (s(22) << 6) | (s(21) >> 2);
    dest[13] = ((s(21) & 0x03) << 12) | (s(20) << 4) | (s(27) >> 4);
    dest[14] = ((s(27) & 0x0f) << 10) | (s(26) << 2) | (s(25) >> 6);
    dest[15] = ((s(25) & 0x3f) << 8) | s(24);
}

fn ascii_field(bytes: &[u8], offset: usize, len: usize) -> String {
    let Some(slice) = bytes.get(offset..offset + len) else {
        return String::new();
    };
    let end = slice.iter().position(|&byte| byte == 0).unwrap_or(slice.len());
    String::from_utf8_lossy(&slice[..end]).trim().to_string()
}

fn read_u16(bytes: &[u8], at: usize) -> NativeResult<u16> {
    let pair = bytes.get(at..at + 2).ok_or_else(|| NativeError::corrupt("truncated RAF directory"))?;
    Ok(u16::from_le_bytes([pair[0], pair[1]]))
}

fn read_be_u32(bytes: &[u8], at: usize) -> NativeResult<u32> {
    let quad = bytes.get(at..at + 4).ok_or_else(|| NativeError::corrupt("truncated RAF header"))?;
    Ok(u32::from_be_bytes([quad[0], quad[1], quad[2], quad[3]]))
}

fn read_be_u16(bytes: &[u8], at: usize) -> NativeResult<u16> {
    let pair = bytes.get(at..at + 2).ok_or_else(|| NativeError::corrupt("truncated RAF tag value"))?;
    Ok(u16::from_be_bytes([pair[0], pair[1]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FUJI_AT: usize = 160;
    const TIFF_AT: usize = 1024;
    const IFD_REL: u32 = 16;
    const DATA_REL: u32 = 0x200;

    fn be_size(height: u32, width: u32) -> Vec<u8> {
        let mut value = Vec::new();
        value.extend_from_slice(&(height as u16).to_be_bytes());
        value.extend_from_slice(&(width as u16).to_be_bytes());
        value
    }

    /// Builds a minimal RAF the way the real container is laid out: big-endian
    /// header pointers at 92/100, a packed Fuji directory and an embedded
    /// little-endian TIFF at the raw-data offset.  Synthetic; no camera file.
    fn make_raf(width: u32, height: u32, bits: u16, payload: &[u8], xtrans: bool) -> Vec<u8> {
        let mut bytes = vec![0u8; HEADER_SIZE];
        bytes[..15].copy_from_slice(b"FUJIFILMCCD-RAW");
        bytes[0x1c..0x20].copy_from_slice(b"X-T2");

        let mut tags: Vec<(u16, Vec<u8>)> = vec![
            (0x0100, be_size(height, width)),
            (0x0121, be_size(height, width)),
        ];
        if xtrans {
            let mut pattern = vec![0u8; 36];
            for row in 0..6 {
                for column in 0..6 {
                    pattern[35 - (row * 6 + column)] = ((row + column) % 3) as u8;
                }
            }
            tags.push((0x0131, pattern));
        }
        tags.push((0x4000, {
            let mut value = Vec::new();
            for _ in 0..4 {
                value.extend_from_slice(&16u16.to_be_bytes());
            }
            value
        }));

        let mut fuji = Vec::new();
        fuji.extend_from_slice(&(tags.len() as u32).to_be_bytes());
        for (tag, value) in &tags {
            fuji.extend_from_slice(&tag.to_be_bytes());
            fuji.extend_from_slice(&(value.len() as u16).to_be_bytes());
            fuji.extend_from_slice(value);
        }
        bytes.resize(FUJI_AT + fuji.len(), 0);
        bytes[FUJI_AT..FUJI_AT + fuji.len()].copy_from_slice(&fuji);
        bytes[92..96].copy_from_slice(&(FUJI_AT as u32).to_be_bytes());
        bytes[100..104].copy_from_slice(&(TIFF_AT as u32).to_be_bytes());

        let ifd_at = TIFF_AT + IFD_REL as usize;
        let entries: Vec<(u16, u16, u32, [u8; 4])> = vec![
            (0x0100, 4, 1, width.to_le_bytes()),
            (0x0101, 4, 1, height.to_le_bytes()),
            (0x0102, 3, 1, u32::from(bits).to_le_bytes()),
            (0x0103, 3, 1, 1u32.to_le_bytes()),
            (0x0106, 3, 1, 32803u32.to_le_bytes()),
            (0x0115, 3, 1, 1u32.to_le_bytes()),
            (0x0117, 4, 1, (payload.len() as u32).to_le_bytes()),
            (0x0111, 4, 1, DATA_REL.to_le_bytes()),
        ];
        let ifd_end = ifd_at + 2 + entries.len() * 12 + 4;
        bytes.resize(TIFF_AT.max(ifd_end), 0);
        bytes[TIFF_AT..TIFF_AT + 2].copy_from_slice(b"II");
        bytes[TIFF_AT + 2..TIFF_AT + 4].copy_from_slice(&42u16.to_le_bytes());
        bytes[TIFF_AT + 4..TIFF_AT + 8].copy_from_slice(&IFD_REL.to_le_bytes());
        bytes[ifd_at..ifd_at + 2].copy_from_slice(&(entries.len() as u16).to_le_bytes());
        for (index, (tag, typ, count, value)) in entries.iter().enumerate() {
            let at = ifd_at + 2 + index * 12;
            bytes[at..at + 2].copy_from_slice(&tag.to_le_bytes());
            bytes[at + 2..at + 4].copy_from_slice(&typ.to_le_bytes());
            bytes[at + 4..at + 8].copy_from_slice(&count.to_le_bytes());
            bytes[at + 8..at + 12].copy_from_slice(value);
        }

        let data_at = TIFF_AT + DATA_REL as usize;
        bytes.resize(data_at + payload.len(), 0);
        bytes[data_at..data_at + payload.len()].copy_from_slice(payload);
        bytes
    }

    fn pack7(values: &[u16]) -> [u8; 7] {
        [
            (values[0] >> 6) as u8,
            ((((values[0] & 0x3f) << 2) | (values[1] >> 12)) & 0xff) as u8,
            ((values[1] >> 4) & 0xff) as u8,
            ((((values[1] & 0xf) << 4) | (values[2] >> 10)) & 0xff) as u8,
            ((values[2] >> 2) & 0xff) as u8,
            ((((values[2] & 3) << 6) | (values[3] >> 8)) & 0xff) as u8,
            (values[3] & 0xff) as u8,
        ]
    }

    fn pack28(values: &[u16; 16]) -> [u8; 28] {
        let mut out = [0u8; 28];
        let d = |index: usize| values[index];
        out[3] = (d(0) >> 6) as u8;
        out[2] = ((((d(0) & 0x3f) << 2) | (d(1) >> 12)) & 0xff) as u8;
        out[1] = ((d(1) >> 4) & 0xff) as u8;
        out[0] = ((((d(1) & 0xf) << 4) | (d(2) >> 10)) & 0xff) as u8;
        out[7] = ((d(2) >> 2) & 0xff) as u8;
        out[6] = ((((d(2) & 3) << 6) | (d(3) >> 8)) & 0xff) as u8;
        out[5] = (d(3) & 0xff) as u8;
        out[4] = (d(4) >> 6) as u8;
        out[11] = ((((d(4) & 0x3f) << 2) | (d(5) >> 12)) & 0xff) as u8;
        out[10] = ((d(5) >> 4) & 0xff) as u8;
        out[9] = ((((d(5) & 0xf) << 4) | (d(6) >> 10)) & 0xff) as u8;
        out[15] = ((d(6) >> 2) & 0xff) as u8;
        out[8] = ((((d(6) & 3) << 6) | (d(7) >> 8)) & 0xff) as u8;
        out[14] = (d(7) & 0xff) as u8;
        out[13] = (d(8) >> 6) as u8;
        out[12] = ((((d(8) & 0x3f) << 2) | (d(9) >> 12)) & 0xff) as u8;
        out[19] = ((d(9) >> 4) & 0xff) as u8;
        out[18] = ((((d(9) & 0xf) << 4) | (d(10) >> 10)) & 0xff) as u8;
        out[17] = ((d(10) >> 2) & 0xff) as u8;
        out[16] = ((((d(10) & 3) << 6) | (d(11) >> 8)) & 0xff) as u8;
        out[23] = (d(11) & 0xff) as u8;
        out[22] = (d(12) >> 6) as u8;
        out[21] = ((((d(12) & 0x3f) << 2) | (d(13) >> 12)) & 0xff) as u8;
        out[20] = ((d(13) >> 4) & 0xff) as u8;
        out[27] = ((((d(13) & 0xf) << 4) | (d(14) >> 10)) & 0xff) as u8;
        out[26] = ((d(14) >> 2) & 0xff) as u8;
        out[25] = ((((d(14) & 3) << 6) | (d(15) >> 8)) & 0xff) as u8;
        out[24] = (d(15) & 0xff) as u8;
        out
    }

    #[test]
    fn decodes_synthetic_uncompressed_xtrans_raf() {
        let width = 24usize;
        let height = 12usize;
        let mut payload = vec![0u8; width * height * 2];
        for index in 0..width * height {
            payload[index * 2..index * 2 + 2].copy_from_slice(&(index as u16).to_le_bytes());
        }
        let bytes = make_raf(width as u32, height as u32, 16, &payload, true);
        let frame = decode(&bytes).unwrap();
        assert_eq!((frame.raw_width, frame.raw_height), (width, height));
        assert_eq!(frame.filters, 9);
        assert_eq!(frame.colors, 3);
        assert_eq!(frame.bits, 16);
        assert_eq!(frame.maximum, 0xffff);
        assert_eq!(frame.black, 16);
        assert_eq!(frame.data.len(), width * height);
        assert_eq!(&frame.data[..4], &[0, 1, 2, 3]);
        assert_eq!(frame.data[width * height - 1], (width * height - 1) as u16);
    }

    #[test]
    fn decodes_synthetic_packed_14bit_raf() {
        // width 24 -> 42 bytes per row: LibRaw byte-swaps 32-bit words and then
        // uses the 7-byte/4-sample decoder.
        let width = 24usize;
        let height = 12usize;
        let mut payload = Vec::new();
        for row in 0..height {
            let mut line = Vec::new();
            for group in 0..width / 4 {
                let values = [
                    ((row * 100 + group * 4) as u16) & 0x3fff,
                    ((row * 100 + group * 4 + 1) as u16) & 0x3fff,
                    ((row * 100 + group * 4 + 2) as u16) & 0x3fff,
                    ((row * 100 + group * 4 + 3) as u16) & 0x3fff,
                ];
                line.extend_from_slice(&pack7(&values));
            }
            for word in line.chunks_exact_mut(4) {
                word.reverse();
            }
            payload.extend_from_slice(&line);
        }
        let bytes = make_raf(width as u32, height as u32, 14, &payload, false);
        let frame = decode(&bytes).unwrap();
        assert_eq!(frame.bits, 14);
        assert_eq!(frame.maximum, 0x3fff);
        assert_eq!(frame.filters, 0x9494_9494);
        assert_eq!(&frame.data[..4], &[0, 1, 2, 3]);
        assert_eq!(frame.data[width], 100);
    }

    #[test]
    fn decodes_synthetic_packed_14bit_raf_28byte_groups() {
        // width 32 -> 56 bytes per row, a multiple of 28, so LibRaw uses the
        // 28-byte/16-sample layout without byte swapping.
        let width = 32usize;
        let height = 8usize;
        let mut payload = Vec::new();
        for row in 0..height {
            for group in 0..width / 16 {
                let mut values = [0u16; 16];
                for (index, value) in values.iter_mut().enumerate() {
                    *value = ((row * 200 + group * 16 + index) as u16) & 0x3fff;
                }
                payload.extend_from_slice(&pack28(&values));
            }
        }
        let bytes = make_raf(width as u32, height as u32, 14, &payload, false);
        let frame = decode(&bytes).unwrap();
        assert_eq!(frame.bits, 14);
        assert_eq!(&frame.data[..4], &[0, 1, 2, 3]);
        assert_eq!(frame.data[15], 15);
        assert_eq!(frame.data[16], 16);
        assert_eq!(frame.data[width * 2], 400);
    }

    #[test]
    fn truncating_the_plane_is_an_error() {
        let payload = vec![0u8; 24 * 12 * 2 - 2];
        let bytes = make_raf(24, 12, 16, &payload, false);
        assert!(decode(&bytes).is_err());
    }
}
