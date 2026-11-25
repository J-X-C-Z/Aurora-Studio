//! Nikon NEF sensor decode.
//!
//! Compression 34713 covers three distinct bitstreams: the Z6III-family HE
//! format (decoded by the already-validated `c1_raw_he` port), and the classic
//! lossless/lossy Huffman tables reproduced from LibRaw's `nikon_load_raw`.
//! Uncompressed strips (compression 1) use the generic packed reader.

use super::tiff::{Ifd, Tiff};
use super::{cfa_filters, make_frame, meta, NativeError, NativeResult};

use crate::raw::BayerFrame;

pub(crate) fn decode(tiff: &Tiff<'_>, _ifd0: &Ifd<'_>, bytes: &[u8]) -> NativeResult<BayerFrame> {
    let ifd = super::raw_ifd(tiff)?;
    let raw_width = ifd.u32(0x0100).ok_or_else(|| NativeError::corrupt("NEF has no image width"))? as usize;
    let raw_height = ifd.u32(0x0101).ok_or_else(|| NativeError::corrupt("NEF has no image height"))? as usize;
    if raw_width == 0 || raw_height == 0 || raw_width > 100_000 || raw_height > 100_000 {
        return Err(NativeError::corrupt("NEF dimensions are out of range"));
    }
    let bits = ifd.u16(0x0102).unwrap_or(14).clamp(8, 16) as u8;
    let compression = ifd.u16(0x0103).unwrap_or(1);
    let (filters, colors) = cfa_filters(&ifd)?;
    if filters == 9 {
        return Err(NativeError::unsupported("X-Trans NEF"));
    }
    let strip_offset = ifd
        .u32s(0x0111)
        .and_then(|values| values.ok())
        .and_then(|values| values.first().copied())
        .ok_or_else(|| NativeError::corrupt("NEF has no strip offset"))? as usize;
    let strip_count = ifd
        .u32s(0x0117)
        .and_then(|values| values.ok())
        .and_then(|values| values.first().copied())
        .unwrap_or((bytes.len().saturating_sub(strip_offset)) as u32) as usize;
    let strip = bytes
        .get(strip_offset..strip_offset.saturating_add(strip_count))
        .ok_or_else(|| NativeError::corrupt("NEF strip is outside the file"))?;

    let data = match compression {
        1 => {
            let expected = raw_width * raw_height;
            if bits == 16 {
                meta::unpack_bits_le16(strip, expected).or_else(|_| meta::unpack_bits_msb(strip, 16, expected))?
            } else {
                meta::unpack_bits_msb(strip, bits, expected)?
            }
        }
        34713 => {
            // Z6III HE / HE*: JPEG XS markers followed by the C1 bitstream.
            if strip.starts_with(&[0xff, 0x10, 0xff, 0x50]) {
                decode_he(bytes, raw_width, raw_height)?
            } else {
                decode_classic(bytes, &ifd, raw_width, raw_height, bits)?
            }
        }
        34892 => decode_lossy(bytes, &ifd, raw_width, raw_height, bits)?,
        other => {
            return Err(NativeError::unsupported(format!("NEF compression {other}")));
        }
    };

    let (left, top, width, height) = geometry(bytes, &ifd, raw_width, raw_height);
    let black = crate::aurora_raw::c1_raw_he::nikon_black_levels(bytes)
        .ok()
        .flatten()
        .map(|levels| levels.0.iter().copied().min().unwrap_or(0) as u32)
        .unwrap_or(0)
        .max(ifd.real(0xc61a).unwrap_or(0.0) as u32);
    let maximum = ifd
        .u32(0xc61d)
        .unwrap_or_else(|| (1u32 << bits.min(16)) - 1);
    let cam_mul = cam_mul(bytes);
    let model = _ifd0.ascii(0x0110).unwrap_or_default();
    let make = _ifd0.ascii(0x010f).unwrap_or_default();
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

fn decode_he(bytes: &[u8], raw_width: usize, raw_height: usize) -> NativeResult<Vec<u16>> {
    let (width, samples) = crate::aurora_raw::c1_raw_he::decode_frame(bytes)
        .map_err(|error| NativeError::corrupt(format!("Nikon HE: {error}")))?
        .ok_or_else(|| NativeError::unsupported("Nikon HE variant is not the verified Z6III one"))?;
    if width != raw_width || samples.len() != raw_width * raw_height {
        return Err(NativeError::unsupported(format!(
            "Nikon HE decoded {width}x{} but the IFD declares {raw_width}x{raw_height}",
            samples.len() / width.max(1)
        )));
    }
    Ok(samples)
}

fn cam_mul(bytes: &[u8]) -> [f32; 4] {
    match crate::aurora_raw::c1_raw_he::nikon_as_shot_rgb(bytes).ok().flatten() {
        Some([red, green, blue]) => [red, green, blue, green],
        None => [0.0; 4],
    }
}

fn geometry(_bytes: &[u8], ifd: &Ifd<'_>, raw_width: usize, raw_height: usize) -> (usize, usize, usize, usize) {
    // LibRaw reports the full sensor for the Nikon bodies this front-end covers;
    // the Nikon MakerNote crop (0x0045) is applied later by the C1 pipeline, not
    // folded into `BayerFrame`.
    let area = super::active_area(ifd, raw_width, raw_height);
    if area != (0, 0, raw_width, raw_height) {
        return area;
    }
    (0, 0, raw_width, raw_height)
}

fn maker_meta_offset(bytes: &[u8]) -> Option<usize> {
    let tiff = Tiff::parse(bytes, 0).ok()?;
    let ifd0 = tiff.ifd(tiff.u32_at(4).ok()?).ok()?;
    let exif = ifd0.exif_ifd().and_then(|offset| tiff.ifd(offset).ok())?;
    let note = exif.entry(0x927c)?;
    let note_start = note.value_offset;
    let note_len = note.value_len;
    let slice = bytes.get(note_start..note_start + note_len)?;
    if !slice.starts_with(b"Nikon\0\x02") {
        return None;
    }
    let maker = Tiff::parse(bytes, note_start + 10).ok()?;
    let root = maker.ifd(maker.u32_at(4).ok()?).ok()?;
    for tag in [0x0096u16, 0x008c] {
        if let Some(entry) = root.entry(tag) {
            return Some(entry.value_offset);
        }
    }
    None
}

/// Bytes the classic Huffman stream may read past its declared strip end before
/// the stream is treated as truncated. Valid files end within the final byte;
/// the small tolerance covers a decoder byte-prefetch into trailing padding.
const NEF_PAST_END_TOLERANCE: usize = 8;

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// Exclusive end of the strip; bytes at or past it are padding and read as
    /// zero, and any such read is counted so truncation is reported.
    end: usize,
    buffer: u64,
    buffered: u8,
    past_end: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], pos: usize, end: usize) -> Self {
        Self { data, pos, end: end.min(data.len()), buffer: 0, buffered: 0, past_end: 0 }
    }

    #[inline(always)]
    fn fill_to(&mut self, count: u8) {
        while self.buffered < count {
            let byte = if self.pos < self.end {
                self.data[self.pos]
            } else {
                // Ran past the strip: keep feeding zero bits so the caller can
                // finish and report truncation once, instead of erroring mid-row.
                self.past_end = self.past_end.saturating_add(1);
                0
            };
            self.pos += 1;
            self.buffer = (self.buffer << 8) | u64::from(byte);
            self.buffered += 8;
        }
    }

    #[inline(always)]
    fn get(&mut self, count: u8) -> u32 {
        if count == 0 {
            return 0;
        }
        self.fill_to(count);
        self.buffered -= count;
        ((self.buffer >> self.buffered) & ((1u64 << count) - 1)) as u32
    }
}

const NEF_LUT_BITS: u8 = 10;
const NEF_LUT_SIZE: usize = 1 << NEF_LUT_BITS;

struct Huff {
    counts: [u8; 16],
    first_code: [i32; 17],
    values: Vec<u8>,
    lut: [(u8, u8); NEF_LUT_SIZE],
    index_after_lut: usize,
}

impl Huff {
    fn new(table: &[u8]) -> Self {
        let mut counts = [0u8; 16];
        counts.copy_from_slice(&table[..16]);
        let mut first_code = [0i32; 17];
        let mut code = 0i32;
        for index in 0..16 {
            first_code[index + 1] = code;
            code = (code + i32::from(counts[index])) << 1;
        }
        let total: usize = counts.iter().map(|&count| usize::from(count)).sum();
        let mut values = table[16..].to_vec();
        // A short initializer in LibRaw's table leaves the remaining leaves zero.
        values.resize(total, 0);
        values.truncate(total);
        let mut lut = [(0u8, 0u8); NEF_LUT_SIZE];
        let mut sym_idx = 0usize;
        for length in 1..=usize::from(NEF_LUT_BITS) {
            let count = usize::from(counts[length - 1]);
            let shift = usize::from(NEF_LUT_BITS) - length;
            let base_code = first_code[length] as usize;
            for offset in 0..count {
                let sym = values[sym_idx + offset];
                let prefix = base_code + offset;
                let start = prefix << shift;
                let end = (prefix + 1) << shift;
                if end <= NEF_LUT_SIZE {
                    for slot in &mut lut[start..end] {
                        *slot = (sym, length as u8);
                    }
                }
            }
            sym_idx += count;
        }
        Self { counts, first_code, values, lut, index_after_lut: sym_idx }
    }

    #[inline(always)]
    fn decode(&self, bits: &mut BitReader<'_>) -> Option<u8> {
        bits.fill_to(NEF_LUT_BITS);
        let peeked = ((bits.buffer >> (bits.buffered - NEF_LUT_BITS)) & ((NEF_LUT_SIZE - 1) as u64)) as usize;
        let (sym, len) = self.lut[peeked];
        if len != 0 {
            bits.buffered -= len;
            return Some(sym);
        }
        bits.buffered -= NEF_LUT_BITS;
        let mut code = peeked as i32;
        let mut index = self.index_after_lut;
        for length in (usize::from(NEF_LUT_BITS) + 1)..=16 {
            code = (code << 1) | bits.get(1) as i32;
            let count = i32::from(self.counts[length - 1]);
            let offset = code - self.first_code[length];
            if offset >= 0 && offset < count {
                return self.values.get(index + offset as usize).copied();
            }
            index += count as usize;
        }
        None
    }
}

/// LibRaw's six Nikon Huffman tables (counts followed by leaf values).
const NIKON_TREES: [[u8; 32]; 6] = [
    [0, 1, 5, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 5, 4, 3, 6, 2, 7, 1, 0, 8, 9, 11, 10, 12, 0, 0, 0],
    [0, 1, 5, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0x39, 0x5a, 0x38, 0x27, 0x16, 5, 4, 3, 2, 1, 0, 11, 12, 12, 0, 0],
    [0, 1, 4, 2, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 4, 6, 3, 7, 2, 8, 1, 9, 0, 10, 11, 12, 0, 0, 0],
    [0, 1, 4, 3, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 0, 5, 6, 4, 7, 8, 3, 9, 2, 1, 0, 10, 11, 12, 13, 14, 0],
    [0, 1, 5, 1, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 0, 0, 8, 0x5c, 0x4b, 0x3a, 0x29, 7, 6, 5, 4, 3, 2, 1, 0, 13, 14, 0],
    [0, 1, 4, 2, 2, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 7, 6, 8, 5, 9, 4, 10, 3, 11, 12, 2, 0, 1, 13, 14, 0],
];

fn decode_classic(
    bytes: &[u8],
    ifd: &Ifd<'_>,
    raw_width: usize,
    raw_height: usize,
    bits: u8,
) -> NativeResult<Vec<u16>> {
    let meta = maker_meta_offset(bytes)
        .ok_or_else(|| NativeError::unsupported("NEF has no Nikon MakerNote meta offset"))?;
    let mut at = meta;
    let ver0 = *bytes.get(at).ok_or_else(|| NativeError::corrupt("NEF meta offset is outside the file"))?;
    let ver1 = *bytes.get(at + 1).ok_or_else(|| NativeError::corrupt("NEF meta offset is truncated"))?;
    at += 2;
    if ver0 == 0x49 || ver1 == 0x58 {
        at += 2110;
    }
    let mut tree = 0usize;
    if ver0 == 0x46 {
        tree = 2;
    }
    if bits == 14 {
        tree += 3;
    }
    let mut vpred = [0u16; 4];
    for value in &mut vpred {
        let hi = *bytes.get(at).ok_or_else(|| NativeError::corrupt("NEF vpred is truncated"))?;
        let lo = *bytes.get(at + 1).ok_or_else(|| NativeError::corrupt("NEF vpred is truncated"))?;
        *value = u16::from_le_bytes([hi, lo]);
        at += 2;
    }
    let mut split = 0usize;
    if ver0 == 0x44 && (ver1 == 0x20 || ver1 == 0x40) {
        let split_at = meta + 562;
        split = u16::from_le_bytes([
            *bytes.get(split_at).ok_or_else(|| NativeError::corrupt("NEF split table is truncated"))?,
            *bytes.get(split_at + 1).ok_or_else(|| NativeError::corrupt("NEF split table is truncated"))?,
        ]) as usize;
    }

    let strip_offset = ifd
        .u32s(0x0111)
        .and_then(|values| values.ok())
        .and_then(|values| values.first().copied())
        .ok_or_else(|| NativeError::corrupt("NEF has no strip offset"))? as usize;
    let strip_count = ifd
        .u32s(0x0117)
        .and_then(|values| values.ok())
        .and_then(|values| values.first().copied())
        .unwrap_or((bytes.len().saturating_sub(strip_offset)) as u32) as usize;
    let strip_end = strip_offset.saturating_add(strip_count).min(bytes.len());
    if strip_offset >= strip_end {
        return Err(NativeError::corrupt("NEF strip is empty or outside the file"));
    }
    let mut bits_reader = BitReader::new(bytes, strip_offset, strip_end);
    let mut huff = Huff::new(&NIKON_TREES[tree]);
    let mut data = vec![0u16; raw_width * raw_height];
    let mut minimum = 0i32;
    let mut hpred = [0i32; 2];
    let mut vpred_i = [vpred[0] as i32, vpred[1] as i32, vpred[2] as i32, vpred[3] as i32];

    for row in 0..raw_height {
        if split != 0 && row == split {
            huff = Huff::new(&NIKON_TREES[tree + 1]);
            minimum = 16;
        }
        for col in 0..raw_width {
            let symbol = huff
                .decode(&mut bits_reader)
                .ok_or_else(|| NativeError::corrupt("NEF Huffman stream is corrupt"))?;
            let length = usize::from(symbol & 15);
            let shift = u32::from(symbol >> 4);
            let raw = bits_reader.get((length as u32).saturating_sub(shift) as u8);
            let mut diff = (((raw << 1) + 1) << shift >> 1) as i32;
            if length > 0 && (diff & (1 << (length - 1))) == 0 {
                diff -= (1i32 << length) - i32::from(shift == 0);
            }
            let value = if col < 2 {
                let slot = (row & 1) * 2 + (col & 1);
                vpred_i[slot] = vpred_i[slot].wrapping_add(diff);
                hpred[col] = vpred_i[slot];
                hpred[col]
            } else {
                hpred[col & 1] = hpred[col & 1].wrapping_add(diff);
                hpred[col & 1]
            };
            let value = (value + minimum).clamp(0, 0x3fff) as u16;
            data[row * raw_width + col] = value;
        }
    }
    // The classic stream has no explicit end marker, so a truncated strip would
    // otherwise decode as a full plane of zero-bit symbols. Any read beyond the
    // strip (past the small prefetch tolerance) means the data was short.
    if bits_reader.past_end > NEF_PAST_END_TOLERANCE {
        return Err(NativeError::corrupt(
            "NEF Huffman stream is shorter than its declared strip",
        ));
    }
    Ok(data)
}

fn decode_lossy(
    _bytes: &[u8],
    _ifd: &Ifd<'_>,
    _raw_width: usize,
    _raw_height: usize,
    _bits: u8,
) -> NativeResult<Vec<u16>> {
    // Compression 34892 uses the same Huffman loader with a 12/14-bit table in
    // most bodies, but the table selection depends on the MakerNote version and
    // is not part of the verified stage-1 set.
    Err(NativeError::unsupported("Nikon lossy compression 34892"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_reader_consuming_its_strip_stays_in_bounds() {
        let data = [0x12u8, 0x34, 0x56, 0x78];
        let mut reader = BitReader::new(&data, 0, data.len());
        assert_eq!(
            [reader.get(8), reader.get(8), reader.get(8), reader.get(8)],
            [0x12, 0x34, 0x56, 0x78]
        );
        assert_eq!(reader.past_end, 0);
    }

    #[test]
    fn bit_reader_flags_reads_past_the_strip() {
        let data = [0xffu8, 0xff];
        let mut reader = BitReader::new(&data, 0, data.len());
        // Two bytes fill the buffer; the third byte request crosses the end.
        let _ = reader.get(8);
        let _ = reader.get(8);
        let _ = reader.get(8);
        assert!(reader.past_end >= 1);
    }

    #[test]
    fn huff_lut_matches_canonical_decode_for_all_nikon_trees() {
        for tree in &NIKON_TREES {
            let huff = Huff::new(tree);
            // Re-encode every leaf in the table and verify LUT + slow path
            // recover the exact symbol and consume the exact bit length.
            let mut idx = 0usize;
            for length in 1..=16usize {
                let count = usize::from(huff.counts[length - 1]);
                for offset in 0..count {
                    let expected_sym = huff.values[idx + offset];
                    let code = (huff.first_code[length] + offset as i32) as u32;
                    let padded = code << (32 - length);
                    let bytes = padded.to_be_bytes();
                    let mut reader = BitReader::new(&bytes, 0, bytes.len());
                    assert_eq!(huff.decode(&mut reader), Some(expected_sym));
                    // Verify remaining bits after decoding match the exact bit length.
                    let consumed = reader.pos * 8 - usize::from(reader.buffered);
                    assert_eq!(consumed, length);
                }
                idx += count;
            }
        }
    }
}
