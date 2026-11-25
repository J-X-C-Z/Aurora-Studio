//! Lossless JPEG (SOF3) decoder for DNG and Canon CR2 streams.
//!
//! The bitstream is the JPEG one: marker segments, canonical DC Huffman tables
//! and entropy-coded data with `FF 00` byte stuffing.  Only the lossless frame
//! type is accepted; the predictor follows ITU T.81 and matches LibRaw's
//! `ljpeg_row`, including LibRaw's handling of the first column (a running
//! per-component DC value rather than the left neighbour).

use super::{NativeError, NativeResult};

/// One decoded lossless JPEG frame, interleaved by component.
pub(crate) struct LjpegImage {
    pub width: usize,
    pub height: usize,
    pub components: usize,
    /// Samples are `width * components * height`, row-major.
    pub data: Vec<u16>,
}

const LUT_BITS: u8 = 10;
const LUT_SIZE: usize = 1 << LUT_BITS;

struct HuffTable {
    counts: [u8; 16],
    first_code: [i32; 17],
    symbols: Vec<u8>,
    /// 10-bit fast lookup: `(symbol, code_length)`. `code_length == 0` means
    /// code is longer than 10 bits and falls back to the canonical loop.
    lut: Box<[(u8, u8); LUT_SIZE]>,
    index_after_lut: usize,
}

impl HuffTable {
    fn new(counts: [u8; 16], symbols: Vec<u8>) -> NativeResult<Self> {
        let total: usize = counts.iter().map(|&count| usize::from(count)).sum();
        if total != symbols.len() || total == 0 {
            return Err(NativeError::corrupt("lossless JPEG Huffman table is inconsistent"));
        }
        let mut first_code = [0i32; 17];
        let mut code = 0i32;
        for (index, &count) in counts.iter().enumerate() {
            first_code[index + 1] = code;
            code = (code + i32::from(count)) << 1;
        }
        let mut lut: Box<[(u8, u8); LUT_SIZE]> = vec![(0u8, 0u8); LUT_SIZE]
            .into_boxed_slice()
            .try_into()
            .unwrap();
        let mut sym_idx = 0usize;
        for length in 1..=usize::from(LUT_BITS) {
            let count = usize::from(counts[length - 1]);
            let shift = usize::from(LUT_BITS) - length;
            let base_code = first_code[length] as usize;
            for offset in 0..count {
                let sym = symbols[sym_idx + offset];
                let prefix = base_code + offset;
                let start = prefix << shift;
                let end = (prefix + 1) << shift;
                if end <= LUT_SIZE {
                    for slot in &mut lut[start..end] {
                        *slot = (sym, length as u8);
                    }
                }
            }
            sym_idx += count;
        }
        Ok(Self { counts, first_code, symbols, lut, index_after_lut: sym_idx })
    }

    #[inline(always)]
    fn decode(&self, bits: &mut Bits<'_>) -> NativeResult<u8> {
        bits.fill();
        let peeked = ((bits.buffer >> (bits.buffered - LUT_BITS)) & ((LUT_SIZE - 1) as u64)) as usize;
        let (sym, len) = self.lut[peeked];
        if len != 0 {
            bits.buffered -= len;
            return Ok(sym);
        }
        bits.buffered -= LUT_BITS;
        let mut code = peeked as i32;
        let mut index = self.index_after_lut;
        for length in (usize::from(LUT_BITS) + 1)..=16usize {
            code = (code << 1) | i32::from(bits.get_one());
            let count = i32::from(self.counts[length - 1]);
            let offset = code - self.first_code[length];
            if offset >= 0 && offset < count {
                return self
                    .symbols
                    .get(index + offset as usize)
                    .copied()
                    .ok_or_else(|| NativeError::corrupt("lossless JPEG Huffman symbol out of range"));
            }
            index += count as usize;
        }
        Err(NativeError::corrupt("lossless JPEG Huffman code is longer than 16 bits"))
    }
}

/// JPEG entropy-coded bit reader with `FF 00` unstuffing.
struct Bits<'a> {
    data: &'a [u8],
    pos: usize,
    buffer: u64,
    buffered: u8,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8], pos: usize) -> Self {
        Self { data, pos, buffer: 0, buffered: 0 }
    }

    #[inline(always)]
    fn next_byte(&mut self) -> Option<u8> {
        let byte = *self.data.get(self.pos)?;
        if byte != 0xff {
            self.pos += 1;
            return Some(byte);
        }
        match self.data.get(self.pos + 1) {
            Some(0x00) => {
                self.pos += 2;
                Some(0xff)
            }
            // Any other byte after FF is a marker: the entropy segment is over.
            _ => None,
        }
    }

    #[inline(always)]
    fn fill(&mut self) {
        while self.buffered < 32 {
            // LibRaw's getbits stops at a marker and returns zero bits after it;
            // do the same so the final padding of a legal stream is harmless.
            let byte = self.next_byte().unwrap_or(0);
            self.buffer = (self.buffer << 8) | u64::from(byte);
            self.buffered += 8;
        }
    }

    #[inline(always)]
    fn get_one(&mut self) -> u8 {
        if self.buffered == 0 {
            self.fill();
        }
        self.buffered -= 1;
        ((self.buffer >> self.buffered) & 1) as u8
    }

    #[inline(always)]
    fn get(&mut self, count: u8) -> u32 {
        if count == 0 {
            return 0;
        }
        if self.buffered < count {
            self.fill();
        }
        self.buffered -= count;
        ((self.buffer >> self.buffered) & ((1u64 << count) - 1)) as u32
    }

    fn reset(&mut self) {
        self.buffer = 0;
        self.buffered = 0;
    }

    /// Drops buffered bits and positions after the next RST marker.
    fn skip_restart(&mut self) -> NativeResult<()> {
        self.reset();
        let mut at = self.pos;
        while at + 1 < self.data.len() {
            if self.data[at] == 0xff && (0xd0..=0xd7).contains(&self.data[at + 1]) {
                self.pos = at + 2;
                return Ok(());
            }
            at += 1;
        }
        Err(NativeError::corrupt("lossless JPEG restart marker is missing"))
    }
}

fn marker_at(data: &[u8], pos: usize) -> NativeResult<u8> {
    if data.get(pos) != Some(&0xff) {
        return Err(NativeError::corrupt("expected a JPEG marker"));
    }
    let mut at = pos + 1;
    while data.get(at) == Some(&0xff) {
        at += 1;
    }
    data.get(at).copied().ok_or_else(|| NativeError::corrupt("truncated JPEG marker"))
}

fn segment<'a>(data: &'a [u8], pos: usize, code_len: usize) -> NativeResult<(&'a [u8], usize)> {
    let length_at = pos + code_len;
    let length = data
        .get(length_at..length_at + 2)
        .map(|bytes| u16::from_be_bytes([bytes[0], bytes[1]]) as usize)
        .ok_or_else(|| NativeError::corrupt("truncated JPEG segment length"))?;
    if length < 2 {
        return Err(NativeError::corrupt("invalid JPEG segment length"));
    }
    let start = length_at + 2;
    let end = length_at + length;
    let body = data
        .get(start..end)
        .ok_or_else(|| NativeError::corrupt("truncated JPEG segment"))?;
    Ok((body, end))
}

/// Decodes one lossless JPEG stream.
///
/// `dng_16bit_bug` reproduces LibRaw's DNG quirk: a 16-bit difference symbol in
/// DNG 1.1 and later stands for `-32768` rather than a signed 16-bit value.
#[allow(unused_assignments)]
pub(crate) fn decode(stream: &[u8], dng_16bit_bug: bool) -> NativeResult<LjpegImage> {
    if stream.get(..2) != Some(&[0xff, 0xd8]) {
        return Err(NativeError::corrupt("lossless JPEG has no SOI"));
    }
    let mut pos = 2usize;
    let mut width = 0usize;
    let mut height = 0usize;
    let mut precision = 0u8;
    let mut components = 0usize;
    let mut component_ids: Vec<u8> = Vec::new();
    let mut component_tables: Vec<usize> = Vec::new();
    let mut tables: [Option<HuffTable>; 4] = [None, None, None, None];
    let mut predictor = 1u8;
    let mut point_transform = 0u8;
    let mut restart = 0usize;
    let mut table_codes: [Vec<u8>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut table_counts: [[u8; 16]; 4] = [[0; 16]; 4];
    let mut table_present = [false; 4];

    loop {
        let marker = marker_at(stream, pos)?;
        let code_len = {
            let mut at = pos + 1;
            while stream.get(at) == Some(&0xff) {
                at += 1;
            }
            at + 1 - pos
        };
        match marker {
            0xd8 => {
                pos += code_len;
            }
            0xd9 => return Err(NativeError::corrupt("lossless JPEG ended before SOS")),
            0xc3 => {
                let (body, next) = segment(stream, pos, code_len)?;
                if body.len() < 6 {
                    return Err(NativeError::corrupt("truncated SOF3"));
                }
                precision = body[0];
                height = u16::from_be_bytes([body[1], body[2]]) as usize;
                width = u16::from_be_bytes([body[3], body[4]]) as usize;
                components = usize::from(body[5]);
                if components == 0 || components > 4 || width == 0 || height == 0 {
                    return Err(NativeError::corrupt("invalid SOF3 dimensions"));
                }
                if body.len() < 6 + components * 3 {
                    return Err(NativeError::corrupt("truncated SOF3 components"));
                }
                component_ids.clear();
                component_tables.clear();
                for index in 0..components {
                    let at = 6 + index * 3;
                    component_ids.push(body[at]);
                    // Table assignment happens in SOS; assume table 0 until then.
                    component_tables.push(0);
                }
                pos = next;
            }
            0xc4 => {
                let (body, next) = segment(stream, pos, code_len)?;
                let mut at = 0usize;
                while at < body.len() {
                    let header = body[at];
                    let class = header >> 4;
                    let identifier = usize::from(header & 0x0f);
                    if class != 0 || identifier > 3 {
                        return Err(NativeError::unsupported("lossless JPEG uses an unsupported Huffman table class"));
                    }
                    let counts_at = at + 1;
                    let counts = body
                        .get(counts_at..counts_at + 16)
                        .ok_or_else(|| NativeError::corrupt("truncated DHT counts"))?;
                    let mut counts_array = [0u8; 16];
                    counts_array.copy_from_slice(counts);
                    let total: usize = counts.iter().map(|&value| usize::from(value)).sum();
                    let symbols_at = counts_at + 16;
                    let symbols = body
                        .get(symbols_at..symbols_at + total)
                        .ok_or_else(|| NativeError::corrupt("truncated DHT symbols"))?
                        .to_vec();
                    table_counts[identifier] = counts_array;
                    table_codes[identifier] = symbols;
                    table_present[identifier] = true;
                    at = symbols_at + total;
                }
                pos = next;
            }
            0xdd => {
                let (body, next) = segment(stream, pos, code_len)?;
                if body.len() < 2 {
                    return Err(NativeError::corrupt("truncated DRI"));
                }
                restart = u16::from_be_bytes([body[0], body[1]]) as usize;
                pos = next;
            }
            0xda => {
                let (body, next) = segment(stream, pos, code_len)?;
                if body.len() < 1 {
                    return Err(NativeError::corrupt("truncated SOS"));
                }
                let count = usize::from(body[0]);
                if count != components || body.len() < 1 + count * 2 + 3 {
                    return Err(NativeError::corrupt("SOS components do not match SOF3"));
                }
                for index in 0..count {
                    let identifier = body[1 + index * 2];
                    let table = usize::from(body[2 + index * 2] >> 4);
                    if table > 3 {
                        return Err(NativeError::corrupt("SOS names an invalid Huffman table"));
                    }
                    let component = component_ids
                        .iter()
                        .position(|&id| id == identifier)
                        .ok_or_else(|| NativeError::corrupt("SOS component is missing from SOF3"))?;
                    component_tables[component] = table;
                }
                let tail = 1 + count * 2;
                predictor = body[tail];
                point_transform = body[tail + 2] & 0x0f;
                pos = next;
                break;
            }
            _ => {
                let (_, next) = segment(stream, pos, code_len)?;
                pos = next;
            }
        }
    }

    if precision < 8 || precision > 16 {
        return Err(NativeError::unsupported(format!("lossless JPEG precision {precision}")));
    }
    if point_transform >= precision {
        return Err(NativeError::corrupt("lossless JPEG point transform exceeds precision"));
    }
    for &table in &component_tables {
        if !table_present[table] && !table_present[0] {
            return Err(NativeError::corrupt("lossless JPEG has no Huffman table"));
        }
    }
    for (index, present) in table_present.iter().enumerate() {
        if *present {
            tables[index] = Some(HuffTable::new(table_counts[index], table_codes[index].clone())?);
        }
    }
    // LibRaw copies table 0 into every table a component names but DHT omitted.
    let fallback = tables[0].as_ref().map(|_| 0usize);
    if let Some(zero) = fallback {
        for index in 1..4 {
            if tables[index].is_none() {
                let counts = table_counts[zero];
                let symbols = table_codes[zero].clone();
                tables[index] = Some(HuffTable::new(counts, symbols)?);
            }
        }
    }

    let bits_available = i32::from(precision) - i32::from(point_transform);
    if !(1..=17).contains(&bits_available) {
        return Err(NativeError::corrupt("lossless JPEG precision after point transform is invalid"));
    }
    let base = 1i32 << (bits_available - 1);
    let mut vpred = [base; 4];
    let mut out = vec![0u16; width * height * components];
    let mut bits = Bits::new(stream, pos);
    let clrs = components;
    let row_samples = width * clrs;

    let mut resolved_tables: Vec<&HuffTable> = Vec::with_capacity(clrs);
    for component in 0..clrs {
        let table = tables[component_tables[component]]
            .as_ref()
            .ok_or_else(|| NativeError::corrupt("missing lossless JPEG Huffman table"))?;
        resolved_tables.push(table);
    }
    if !(1..=7).contains(&predictor) && height > 1 && width > 1 {
        return Err(NativeError::unsupported(format!(
            "lossless JPEG predictor {predictor}"
        )));
    }

    for row in 0..height {
        if restart != 0 && (row * width) % restart == 0 {
            vpred = [base; 4];
            if row != 0 {
                bits.skip_restart()?;
            }
        }
        let row_start = row * row_samples;
        let prev_row_start = row.saturating_sub(1) * row_samples;
        for col in 0..width {
            let col_off = col * clrs;
            let left_off = col.saturating_sub(1) * clrs;
            for (component, &table) in resolved_tables.iter().enumerate() {
                let length = table.decode(&mut bits)?;
                let diff = if length == 16 && dng_16bit_bug {
                    -32768
                } else if length == 0 {
                    0
                } else {
                    if length > 16 {
                        return Err(NativeError::corrupt("lossless JPEG difference is wider than 16 bits"));
                    }
                    let raw = bits.get(length) as i32;
                    // Sign-extend the JPEG magnitude (T.81 F.2.2.1).
                    if raw & (1 << (length - 1)) != 0 {
                        raw
                    } else {
                        raw - ((1i32 << length) - 1)
                    }
                };
                let prediction = if col == 0 {
                    // LibRaw keeps a running DC per component for the first column.
                    let before = vpred[component];
                    vpred[component] = before.wrapping_add(diff);
                    before
                } else if row == 0 {
                    i32::from(out[row_start + left_off + component])
                } else {
                    let left = i32::from(out[row_start + left_off + component]);
                    match predictor {
                        1 => left,
                        2 => i32::from(out[prev_row_start + col_off + component]),
                        3 => i32::from(out[prev_row_start + left_off + component]),
                        4 => {
                            let previous = i32::from(out[prev_row_start + col_off + component]);
                            let above_left = i32::from(out[prev_row_start + left_off + component]);
                            left + previous - above_left
                        }
                        5 => {
                            let previous = i32::from(out[prev_row_start + col_off + component]);
                            let above_left = i32::from(out[prev_row_start + left_off + component]);
                            left + ((previous - above_left) >> 1)
                        }
                        6 => {
                            let previous = i32::from(out[prev_row_start + col_off + component]);
                            let above_left = i32::from(out[prev_row_start + left_off + component]);
                            previous + ((left - above_left) >> 1)
                        }
                        7 => {
                            let previous = i32::from(out[prev_row_start + col_off + component]);
                            (left + previous) >> 1
                        }
                        other => {
                            return Err(NativeError::unsupported(format!(
                                "lossless JPEG predictor {other}"
                            )));
                        }
                    }
                };
                let value = prediction.wrapping_add(diff);
                out[row_start + col_off + component] = value as u16;
            }
        }
    }

    Ok(LjpegImage { width, height, components, data: out })
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A tiny canonical Huffman writer for round-trip tests.
    pub(crate) struct Encoder {
        symbols: Vec<u8>,
        /// (bit length, code) per symbol, aligned with `symbols`.
        codes: Vec<(u8, u16)>,
    }

    impl Encoder {
        /// Builds a table whose lengths are given per symbol, in symbol order.
        pub fn from_lengths(symbols: &[u8], lengths: &[u8]) -> Self {
            assert_eq!(symbols.len(), lengths.len());
            let mut counts = [0u8; 16];
            for &length in lengths {
                counts[usize::from(length) - 1] += 1;
            }
            let mut next = [0u16; 17];
            let mut code = 0u16;
            for length in 1..=16 {
                next[length] = code;
                code = (code + u16::from(counts[length - 1])) << 1;
            }
            let mut codes = Vec::with_capacity(symbols.len());
            for &length in lengths {
                let value = next[usize::from(length)];
                next[usize::from(length)] += 1;
                codes.push((length, value));
            }
            Self { symbols: symbols.to_vec(), codes }
        }

        pub fn code_for(&self, symbol: u8) -> (u8, u16) {
            let index = self.symbols.iter().position(|&value| value == symbol).unwrap();
            self.codes[index]
        }
    }

    pub(crate) struct Writer {
        bytes: Vec<u8>,
        accumulator: u64,
        bits: u8,
    }

    impl Writer {
        pub fn new() -> Self {
            Self { bytes: Vec::new(), accumulator: 0, bits: 0 }
        }
        pub fn push(&mut self, value: u32, count: u8) {
            self.accumulator = (self.accumulator << count) | u64::from(value & ((1 << count) - 1));
            self.bits += count;
            while self.bits >= 8 {
                self.bits -= 8;
                let byte = (self.accumulator >> self.bits) as u8;
                self.bytes.push(byte);
                if byte == 0xff {
                    self.bytes.push(0);
                }
            }
        }
        pub fn symbol(&mut self, encoder: &Encoder, symbol: u8) {
            let (length, code) = encoder.code_for(symbol);
            self.push(u32::from(code), length);
        }
        pub fn finish(mut self) -> Vec<u8> {
            if self.bits > 0 {
                self.push(0xff >> (8 - self.bits), 8 - self.bits);
            }
            self.bytes
        }
    }

    fn stream(segment_body: &[u8], marker: u8) -> Vec<u8> {
        let mut out = vec![0xff, marker];
        let length = segment_body.len() + 2;
        out.extend((length as u16).to_be_bytes());
        out.extend_from_slice(segment_body);
        out
    }

    /// Builds a one-component lossless JPEG for a predictor-1 image.
    pub(crate) fn encode_gray(width: usize, height: usize, samples: &[u16]) -> Vec<u8> {
        // Symbols 0..=4 cover the differences in the test images.
        let encoder = Encoder::from_lengths(&[0, 1, 2, 3, 4], &[2, 3, 3, 4, 4]);
        let mut out = vec![0xff, 0xd8];
        out.extend(stream(&[8, 0, height as u8, 0, width as u8, 1, 1, 0x11, 0], 0xc3));
        let mut dht = vec![0x00];
        let mut counts = [0u8; 16];
        for length in [2u8, 3, 3, 4, 4] {
            counts[usize::from(length) - 1] += 1;
        }
        dht.extend(counts);
        dht.extend_from_slice(&[0, 1, 2, 3, 4]);
        out.extend(stream(&dht, 0xc4));
        out.extend(stream(&[1, 1, 0x00, 1, 0, 0], 0xda));
        let mut writer = Writer::new();
        let mut running = 128i32;
        for row in 0..height {
            for col in 0..width {
                let value = i32::from(samples[row * width + col]);
                let prediction = if col == 0 {
                    let before = running;
                    running = value;
                    before
                } else {
                    i32::from(samples[row * width + col - 1])
                };
                let diff = value - prediction;
                let length = magnitude_bits(diff);
                writer.symbol(&encoder, length as u8);
                if length > 0 {
                    let encoded = if diff < 0 { diff + (1 << length) - 1 } else { diff };
                    writer.push(encoded as u32, length as u8);
                }
            }
        }
        out.extend(writer.finish());
        out
    }

    fn magnitude_bits(diff: i32) -> u32 {
        if diff == 0 {
            0
        } else {
            32 - (diff.unsigned_abs()).leading_zeros()
        }
    }

    #[test]
    fn decodes_a_single_component_predictor_one_stream() {
        let width = 8;
        let height = 4;
        let mut samples = vec![128u16; width * height];
        for row in 0..height {
            for col in 1..width {
                let delta = ((row + col) % 3) as i32 - 1;
                samples[row * width + col] =
                    (i32::from(samples[row * width + col - 1]) + delta) as u16;
            }
        }
        let bytes = encode_gray(width, height, &samples);
        let decoded = decode(&bytes, false).unwrap();
        assert_eq!((decoded.width, decoded.height, decoded.components), (width, height, 1));
        assert_eq!(decoded.data, samples);
    }

    #[test]
    fn rejects_a_stream_without_soi() {
        assert!(decode(&[0, 1, 2, 3], false).is_err());
    }
}
