//! Canon CRX (CR3 raw codec), encoding type 0.
//!
//! The codec stores four quadrant planes.  Each plane is a 5/3 wavelet image
//! whose subbands are entropy coded with the same zero-run/predictive coder for
//! every tile.  This module implements the paths Canon uses in the verified
//! CR3 files: `encType == 0`, one or two tile columns, levels 0 or 3, and the
//! old (`0xFF03`) subband headers.  Rounded/partial re-quantisation (encType 1/3)
//! is deliberately rejected instead of guessed.

use super::{NativeError, NativeResult};

pub(crate) const E_HAS_TILES_ON_THE_RIGHT: u8 = 1;
pub(crate) const E_HAS_TILES_ON_THE_LEFT: u8 = 2;
pub(crate) const E_HAS_TILES_ON_THE_BOTTOM: u8 = 4;
pub(crate) const E_HAS_TILES_ON_THE_TOP: u8 = 8;

const JS: [u32; 32] = [
    1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 0x10, 0x10, 0x20, 0x20, 0x40, 0x40, 0x80, 0x80,
    0x100, 0x200, 0x400, 0x800, 0x1000, 0x2000, 0x4000, 0x8000,
];
const J: [u32; 32] = [
    0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 0x0A, 0x0B, 0x0C,
    0x0D, 0x0E, 0x0F,
];
const Q_STEP: [i32; 6] = [0x28, 0x2D, 0x33, 0x39, 0x40, 0x48];
const EX_COEF: [i32; 144] = [
    1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 2, 2, 1,
    0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 2, 2, 1, 0, 0, 1, 1, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2, 2, 1, 1, 0,
    1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1,
];

#[derive(Clone, Debug)]
pub(crate) struct CrxHeader {
    pub width: usize,
    pub height: usize,
    pub tile_width: usize,
    pub tile_height: usize,
    pub bits: u8,
    pub planes: u8,
    pub cfa_layout: u8,
    pub enc_type: u8,
    pub levels: u8,
    pub mdat_header_size: usize,
}

/// Parses the `CMP1` payload (the bytes after the box header).
pub(crate) fn parse_header(payload: &[u8]) -> NativeResult<CrxHeader> {
    if payload.len() < 32 {
        return Err(NativeError::corrupt("CRX CMP1 payload is too short"));
    }
    let version = u16::from_be_bytes([payload[4], payload[5]]);
    let width = u32::from_be_bytes(payload[8..12].try_into().unwrap()) as usize;
    let height = u32::from_be_bytes(payload[12..16].try_into().unwrap()) as usize;
    let tile_width = u32::from_be_bytes(payload[16..20].try_into().unwrap()) as usize;
    let tile_height = u32::from_be_bytes(payload[20..24].try_into().unwrap()) as usize;
    let bits = payload[24];
    let planes = payload[25] >> 4;
    let cfa_layout = payload[25] & 0x0F;
    let enc_type = payload[26] >> 4;
    let levels = payload[26] & 0x0F;
    let mdat_header_size = u32::from_be_bytes(payload[28..32].try_into().unwrap()) as usize;
    if !matches!(version, 0x100 | 0x200) || mdat_header_size == 0 {
        return Err(NativeError::unsupported(format!("CRX version {version:#x}")));
    }
    if enc_type != 0 {
        return Err(NativeError::unsupported(format!("CRX encoding type {enc_type}")));
    }
    if bits == 0 || bits > 14 || planes != 4 || cfa_layout > 3 {
        return Err(NativeError::unsupported("CRX header is not 4-plane 14-bit"));
    }
    if tile_width == 0 || tile_height == 0 || tile_width > width || tile_height > height {
        return Err(NativeError::corrupt("CRX tile geometry is invalid"));
    }
    Ok(CrxHeader {
        width,
        height,
        tile_width,
        tile_height,
        bits,
        planes,
        cfa_layout,
        enc_type,
        levels,
        mdat_header_size,
    })
}

// ---------------------------------------------------------------------------
// Bitstream

struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    end: usize,
    bit: u32,
    left: i32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], start: usize, len: usize) -> Self {
        Self { data, pos: start, end: (start + len).min(data.len()), bit: 0, left: 0 }
    }

    fn word(&mut self) -> Option<(u32, i32)> {
        if self.pos + 4 <= self.end {
            let value = u32::from_be_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
            self.pos += 4;
            Some((value, 32))
        } else if self.pos < self.end {
            let value = u32::from(self.data[self.pos]) << 24;
            self.pos += 1;
            Some((value, 8))
        } else {
            None
        }
    }

    fn zeros(&mut self) -> NativeResult<u32> {
        if self.bit != 0 {
            let nz = 31 - self.bit.leading_zeros() as i32;
            let result = 31 - nz;
            self.bit = if nz == 0 { 0 } else { self.bit << (32 - nz) };
            self.left -= 32 - nz;
            return Ok(result as u32);
        }
        let mut bits = self.left;
        loop {
            let Some((word, got)) = self.word() else {
                // LibRaw's crxBitstreamGetZeros does not fail at the end of a
                // band: the `break` path leaves nextData == 0, so it reports the
                // remaining zero run and leaves the reader exhausted.
                let result = bits + 7;
                self.bit = 0;
                self.left = 0;
                return Ok(result as u32);
            };
            if word != 0 {
                let nz = 31 - word.leading_zeros() as i32;
                // `word` is left-aligned in 32 bits and only `got` bits are
                // valid (32 for a full word, 8 for LibRaw's byte-at-a-time
                // fallback).  crxBitstreamGetZeros reports `nonZeroBit` inside
                // the fetched unit, so the pending-bit count is `nz` minus the
                // padding bits of a short read.
                let result = bits + 31 - nz;
                self.bit = if nz == 0 { 0 } else { word << (32 - nz) };
                self.left = nz + got - 32;
                return Ok(result as u32);
            }
            bits += got;
        }
    }

    fn get(&mut self, bits: u32) -> NativeResult<u32> {
        if bits == 0 {
            return Ok(0);
        }
        if bits > 32 {
            return Err(NativeError::corrupt("CRX bit request is wider than 32 bits"));
        }
        let mut left = self.left;
        let mut data = self.bit;
        if left < bits as i32 {
            if self.pos + 4 <= self.end {
                let word = u32::from_be_bytes(self.data[self.pos..self.pos + 4].try_into().unwrap());
                self.pos += 4;
                let need = bits as i32 - left;
                if !(0..=32).contains(&need) {
                    return Err(NativeError::corrupt("CRX bit window is inconsistent"));
                }
                self.left = 32 - need;
                let result = ((word >> left) | data) >> (32 - bits);
                self.bit = word << need;
                return Ok(result);
            }
            while left < bits as i32 {
                if self.pos >= self.end {
                    // crxBitstreamGetBits breaks out of its byte loop when the
                    // band is exhausted and returns the pending (zero) bits.
                    break;
                }
                left += 8;
                if left > 32 {
                    return Err(NativeError::corrupt("CRX bit window overflow"));
                }
                data |= u32::from(self.data[self.pos]) << (32 - left);
                self.pos += 1;
            }
        }
        let result = data >> (32 - bits);
        self.bit = data << bits;
        self.left = left - bits as i32;
        Ok(result)
    }
}

fn predict_k(previous: i32, bit_code: u32, max_value: i32) -> i32 {
    let shift = previous.clamp(0, 31) as u32;
    let threshold = if (0..32).contains(&previous) { (1u32 << previous) >> 1 } else { 0 };
    let new_value = previous
        - i32::from(bit_code < threshold)
        + i32::from((bit_code >> shift) > 2)
        + i32::from((bit_code >> shift) > 5);
    if max_value == 0 || new_value < max_value { new_value } else { max_value }
}

fn signed_value(code: u32) -> i32 {
    -((code & 1) as i32) ^ ((code >> 1) as i32)
}

// ---------------------------------------------------------------------------
// Entropy-coded line parameter

struct BandParam<'a> {
    bits: BitReader<'a>,
    width: usize,
    height: usize,
    supports_partial: bool,
    previous: Vec<i32>,
    current: Vec<i32>,
    k_history: Vec<i32>,
    s_param: i32,
    k_param: i32,
    cur_line: usize,
    off0: usize,
    off1: usize,
    off2: usize,
}

impl<'a> BandParam<'a> {
    fn new(data: &'a [u8], offset: usize, len: usize, width: usize, height: usize, partial: bool) -> Self {
        Self {
            bits: BitReader::new(data, offset, len),
            width,
            height,
            supports_partial: partial,
            previous: vec![0; width + 4],
            current: vec![0; width + 4],
            k_history: vec![0; width + 4],
            s_param: 0,
            k_param: 0,
            cur_line: 0,
            off0: 0,
            off1: 0,
            off2: 0,
        }
    }

    fn code(&mut self) -> NativeResult<u32> {
        let mut code = self.bits.zeros()?;
        if code >= 41 {
            code = self.bits.get(21)?;
        } else if self.k_param > 0 {
            code = self.bits.get(self.k_param as u32)? | (code << self.k_param);
        }
        Ok(code)
    }

    fn runs(&mut self, limit: usize) -> NativeResult<usize> {
        let mut count = 0usize;
        if self.bits.get(1)? != 0 {
            count = 1;
            while self.bits.get(1)? != 0 {
                count += JS[self.s_param as usize] as usize;
                if count > limit {
                    return Ok(limit);
                }
                if self.s_param < 31 {
                    self.s_param += 1;
                }
                if count == limit {
                    break;
                }
            }
            if count < limit {
                if J[self.s_param as usize] != 0 {
                    count += self.bits.get(J[self.s_param as usize])? as usize;
                }
                if self.s_param > 0 {
                    self.s_param -= 1;
                }
                if count > limit {
                    return Err(NativeError::corrupt("CRX zero run exceeds the line"));
                }
            }
        }
        Ok(count)
    }

    fn decode_symbol_l1(&mut self, median: bool, not_eol: bool) -> NativeResult<()> {
        let o0 = self.off0;
        let o1 = self.off1;
        if median {
            let delta = self.previous[o0 + 1] - self.previous[o0];
            let symb2 = self.current[o1];
            let symb = [delta + symb2, delta + symb2, symb2, self.previous[o0 + 1]];
            let index = ((((self.previous[o0] < self.current[o1]) ^ (delta < 0)) as usize) << 1)
                + ((self.current[o1] < self.previous[o0 + 1]) ^ (delta < 0)) as usize;
            self.current[o1 + 1] = symb[index];
        } else {
            self.current[o1 + 1] = self.previous[o0 + 1];
        }
        let mut code = self.code()?;
        self.current[o1 + 1] += signed_value(code);
        if not_eol {
            let next_delta = (self.previous[o0 + 2] - self.previous[o0 + 1]) << 1;
            code = ((code as i32 + next_delta.abs()) >> 1) as u32;
            self.off0 += 1;
        }
        self.k_param = predict_k(self.k_param, code, 15);
        self.off1 += 1;
        Ok(())
    }

    /// The normal (partial-support) line decoder.
    fn decode_line(&mut self) -> NativeResult<()> {
        self.start_line();
        self.current[0] = self.previous[1];
        let mut length = self.width;
        while length > 1 {
            if self.current[self.off1] != self.previous[self.off0 + 1]
                || self.current[self.off1] != self.previous[self.off0 + 2]
            {
                self.decode_symbol_l1(true, true)?;
            } else {
                let count = self.runs(length)?;
                length -= count;
                self.off0 += count;
                for _ in 0..count {
                    self.current[self.off1 + 1] = self.current[self.off1];
                    self.off1 += 1;
                }
                if length > 0 {
                    self.decode_symbol_l1(false, length > 1)?;
                }
            }
            length -= 1;
        }
        if length == 1 {
            self.decode_symbol_l1(true, false)?;
        }
        self.current[self.off1 + 1] = self.current[self.off1] + 1;
        Ok(())
    }

    fn decode_top(&mut self) -> NativeResult<()> {
        self.start_line();
        self.current[0] = 0;
        let mut length = self.width;
        while length > 1 {
            if self.current[self.off1] != 0 {
                self.current[self.off1 + 1] = self.current[self.off1];
            } else {
                let count = self.runs(length)?;
                length -= count;
                for _ in 0..count {
                    self.current[self.off1 + 1] = self.current[self.off1];
                    self.off1 += 1;
                }
                if length == 0 {
                    break;
                }
                self.current[self.off1 + 1] = 0;
            }
            let code = self.code()?;
            self.current[self.off1 + 1] += signed_value(code);
            self.k_param = predict_k(self.k_param, code, 15);
            self.off1 += 1;
            length -= 1;
        }
        if length == 1 {
            self.current[self.off1 + 1] = self.current[self.off1];
            let code = self.code()?;
            self.current[self.off1 + 1] += signed_value(code);
            self.k_param = predict_k(self.k_param, code, 15);
            self.off1 += 1;
        }
        self.current[self.off1 + 1] = self.current[self.off1] + 1;
        Ok(())
    }

    fn decode_top_noref(&mut self) -> NativeResult<()> {
        self.start_line();
        self.previous.fill(0);
        self.current.fill(0);
        self.k_history.fill(0);
        let mut length = self.width;
        while length > 1 {
            if self.current[self.off1] != 0 {
                let code = self.code()?;
                self.current[self.off1 + 1] = signed_value(code);
                self.k_param = predict_k(self.k_param, code, 15);
            } else {
                let count = self.runs(length)?;
                length -= count;
                for _ in 0..count {
                    self.k_history[self.off2] = 0;
                    self.current[self.off1 + 1] = 0;
                    self.off1 += 1;
                    self.off2 += 1;
                }
                if length == 0 {
                    break;
                }
                let code = self.code()?;
                self.current[self.off1 + 1] = signed_value(code.wrapping_add(1));
                self.k_param = predict_k(self.k_param, code, 15);
            }
            self.k_history[self.off2] = self.k_param;
            self.off2 += 1;
            self.off1 += 1;
            length -= 1;
        }
        if length == 1 {
            let code = self.code()?;
            self.current[self.off1 + 1] = signed_value(code);
            self.k_param = predict_k(self.k_param, code, 15);
            self.k_history[self.off2] = self.k_param;
            self.off1 += 1;
        }
        self.current[self.off1 + 1] = 0;
        Ok(())
    }

    fn decode_line_noref(&mut self) -> NativeResult<()> {
        self.start_line();
        let mut index = 0usize;
        let width = self.width;
        while index < width - 1 {
            if self.previous[index + 1] != 0
                || self.previous[index + 2] != 0
                || self.current[index] != 0
            {
                let code = self.code()?;
                self.current[index + 1] = signed_value(code);
                self.k_param = predict_k(self.k_param, code, 0);
                if self.k_history[index + 1] - self.k_param <= 1 {
                    if self.k_param >= 15 {
                        self.k_param = 15;
                    }
                } else {
                    self.k_param += 1;
                }
            } else {
                let count = self.runs(width - index)?;
                if count > 0 {
                    for step in 0..count {
                        self.current[index + 1 + step] = 0;
                        self.k_history[index + step] = 0;
                    }
                    index += count;
                }
                if index >= width - 1 {
                    if index == width - 1 {
                        let code = self.code()?;
                        self.current[index + 1] = signed_value(code.wrapping_add(1));
                        self.k_param = predict_k(self.k_param, code, 15);
                        self.k_history[index] = self.k_param;
                    }
                    index += 1;
                    continue;
                }
                let code = self.code()?;
                self.current[index + 1] = signed_value(code.wrapping_add(1));
                self.k_param = predict_k(self.k_param, code, 0);
                if self.k_history[index + 1] - self.k_param <= 1 {
                    if self.k_param >= 15 {
                        self.k_param = 15;
                    }
                } else {
                    self.k_param += 1;
                }
            }
            self.k_history[index] = self.k_param;
            index += 1;
        }
        if index == width - 1 {
            let code = self.code()?;
            self.current[index + 1] = signed_value(code);
            self.k_param = predict_k(self.k_param, code, 15);
            self.k_history[index] = self.k_param;
        }
        Ok(())
    }

    fn start_line(&mut self) {
        self.off0 = 0;
        self.off1 = 0;
        self.off2 = 0;
    }

    fn line(&mut self) -> NativeResult<()> {
        if self.cur_line >= self.height {
            return Err(NativeError::corrupt("CRX line exceeds subband height"));
        }
        if self.cur_line == 0 {
            if self.supports_partial {
                self.decode_top()?;
            } else {
                self.decode_top_noref()?;
            }
        } else if self.supports_partial {
            self.decode_line()?;
        } else {
            self.decode_line_noref()?;
        }
        // The freshly decoded line becomes the reference for the next one.
        std::mem::swap(&mut self.previous, &mut self.current);
        self.cur_line += 1;
        Ok(())
    }

    fn line_data(&self) -> &[i32] {
        &self.previous[1..1 + self.width]
    }
}

// ---------------------------------------------------------------------------
// Subbands, wavelets

struct Subband<'a> {
    width: usize,
    height: usize,
    q_param: i32,
    data_size: usize,
    buf: Vec<i32>,
    param: Option<BandParam<'a>>,
}

struct Wavelet {
    width: usize,
    height: usize,
    line: [Vec<i32>; 8],
    s0: Vec<i32>,
    s1: Vec<i32>,
    s2: Vec<i32>,
    s3: Vec<i32>,
    cur_line: usize,
    cur_h: i32,
    flt: usize,
}

impl Wavelet {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            line: std::array::from_fn(|_| vec![0; width]),
            s0: vec![0; width],
            s1: vec![0; width],
            s2: vec![0; width],
            s3: vec![0; width],
            cur_line: 0,
            cur_h: 0,
            flt: 0,
        }
    }
}

fn setup_subband_geometry(tile: &Tile, levels: u8) -> Vec<(usize, usize)> {
    // (width, height)
    let count = 3 * levels as usize + 1;
    let mut bands = vec![(0usize, 0usize); count];
    let mut band_height = tile.height;
    let mut band_width = tile.width;
    let (mut bw_extra, mut bh_extra) = (0usize, 0usize);
    if levels > 0 {
        let row_base = 0x30 * (levels as usize - 1) + 6 * (tile.width & 7);
        let col_base = 0x30 * (levels as usize - 1) + 6 * (tile.height & 7);
        let mut slot = count - 1;
        for level in 0..levels as usize {
            let width_odd = band_width & 1;
            let height_odd = band_height & 1;
            band_width = (width_odd + band_width) >> 1;
            band_height = (height_odd + band_height) >> 1;
            let mut bw0 = 0usize;
            let mut bw1 = 0usize;
            let mut bh0 = 0usize;
            let mut bh1 = 0usize;
            if tile.flags & E_HAS_TILES_ON_THE_RIGHT != 0 {
                bw0 = EX_COEF[row_base + 2 * level] as usize;
                bw1 = EX_COEF[row_base + 2 * level + 1] as usize;
            }
            if tile.flags & E_HAS_TILES_ON_THE_LEFT != 0 {
                bw0 += 1;
            }
            if tile.flags & E_HAS_TILES_ON_THE_BOTTOM != 0 {
                bh0 = EX_COEF[col_base + 2 * level] as usize;
                bh1 = EX_COEF[col_base + 2 * level + 1] as usize;
            }
            if tile.flags & E_HAS_TILES_ON_THE_TOP != 0 {
                bh0 += 1;
            }
            bands[slot] = (band_width + bw0 - width_odd, band_height + bh0 - height_odd);
            bands[slot - 1] = (band_width + bw1, band_height + bh0 - height_odd);
            bands[slot - 2] = (band_width + bw0 - width_odd, band_height + bh1);
            slot -= 3;
        }
        if tile.flags & E_HAS_TILES_ON_THE_RIGHT != 0 {
            bw_extra = EX_COEF[row_base + 2 * levels as usize - 1] as usize;
        }
        if tile.flags & E_HAS_TILES_ON_THE_BOTTOM != 0 {
            bh_extra = EX_COEF[col_base + 2 * levels as usize - 1] as usize;
        }
        band_width += bw_extra;
        band_height += bh_extra;
    }
    bands[0] = (band_width, band_height);
    bands
}

fn decode_with_iq(sub: &mut Subband<'_>) -> NativeResult<()> {
    if sub.data_size == 0 {
        sub.buf.fill(0);
        return Ok(());
    }
    let param = sub.param.as_mut().ok_or_else(|| NativeError::corrupt("CRX subband has no bitstream"))?;
    let line_no = param.cur_line;
    param.line().map_err(|error| NativeError::corrupt(format!(
        "CRX band {}x{} line {line_no} q{}: {error}",
        sub.width, sub.height, sub.q_param
    )))?;
    let line = param.line_data().to_vec();
    sub.buf[..line.len()].copy_from_slice(&line);
    let q = sub.q_param;
    let index = (q % 6) as usize;
    let scale = if q / 6 >= 6 {
        Q_STEP[index] * (1 << (q / 6 + 26))
    } else {
        Q_STEP[index] >> (6 - q / 6)
    };
    if scale != 1 {
        for value in &mut sub.buf {
            *value *= scale;
        }
    }
    Ok(())
}

fn copy_subband_to(wv: &mut Wavelet, slot: usize, sub: &Subband<'_>) {
    let target = match slot {
        0 => &mut wv.s0,
        1 => &mut wv.s1,
        2 => &mut wv.s2,
        _ => &mut wv.s3,
    };
    let len = sub.buf.len().min(target.len());
    target[..len].copy_from_slice(&sub.buf[..len]);
}

fn horizontal53(la: &mut [i32], lb: &mut [i32], wv: &Wavelet, flag: u8) {
    let width = wv.width;
    if width <= 1 {
        la[0] = wv.s0[0];
        lb[0] = wv.s2[0];
        return;
    }
    let (mut i0, mut i1, mut i2, mut i3) = (0usize, 0usize, 0usize, 0usize);
    if flag & E_HAS_TILES_ON_THE_LEFT != 0 {
        la[0] = wv.s0[0] - ((wv.s1[0] + wv.s1[1] + 2) >> 2);
        lb[0] = wv.s2[0] - ((wv.s3[0] + wv.s3[1] + 2) >> 2);
        i1 += 1;
        i3 += 1;
    } else {
        la[0] = wv.s0[0] - ((wv.s1[0] + 1) >> 1);
        lb[0] = wv.s2[0] - ((wv.s3[0] + 1) >> 1);
    }
    i0 += 1;
    i2 += 1;
    let mut i = 0usize;
    while i + 3 < width {
        let delta = wv.s0[i0] - ((wv.s1[i1] + wv.s1[i1 + 1] + 2) >> 2);
        la[i + 1] = wv.s1[i1] + ((delta + la[i]) >> 1);
        la[i + 2] = delta;
        let delta = wv.s2[i2] - ((wv.s3[i3] + wv.s3[i3 + 1] + 2) >> 2);
        lb[i + 1] = wv.s3[i3] + ((delta + lb[i]) >> 1);
        lb[i + 2] = delta;
        i0 += 1;
        i1 += 1;
        i2 += 1;
        i3 += 1;
        i += 2;
    }
    if flag & E_HAS_TILES_ON_THE_RIGHT != 0 {
        let da = wv.s0[i0] - ((wv.s1[i1] + wv.s1[i1 + 1] + 2) >> 2);
        la[i + 1] = wv.s1[i1] + ((da + la[i]) >> 1);
        let db = wv.s2[i2] - ((wv.s3[i3] + wv.s3[i3 + 1] + 2) >> 2);
        lb[i + 1] = wv.s3[i3] + ((db + lb[i]) >> 1);
        if width & 1 != 0 {
            la[i + 2] = da;
            lb[i + 2] = db;
        }
    } else if width & 1 != 0 {
        la[i + 1] = wv.s1[i1] + ((la[i] + wv.s0[i0] - ((wv.s1[i1] + 1) >> 1)) >> 1);
        la[i + 2] = wv.s0[i0] - ((wv.s1[i1] + 1) >> 1);
        lb[i + 1] = wv.s3[i3] + ((lb[i] + wv.s2[i2] - ((wv.s3[i3] + 1) >> 1)) >> 1);
        lb[i + 2] = wv.s2[i2] - ((wv.s3[i3] + 1) >> 1);
    } else {
        la[i + 1] = la[i] + wv.s1[i1];
        lb[i + 1] = lb[i] + wv.s3[i3];
    }
}

fn idwt_get_line(wv: &mut Wavelet) -> Vec<i32> {
    // crxIdwt53FilterGetLine indexes with curH *before* decrementing it.
    let idx = ((wv.flt as i32 - wv.cur_h + 5) % 5 + 3) as usize;
    wv.cur_h -= 1;
    wv.line[idx].clone()
}

fn idwt_decode(
    wv: &mut [Wavelet],
    subs: &mut [Subband<'_>],
    level: usize,
    flag: u8,
) -> NativeResult<()> {
    if wv[level].cur_h != 0 {
        return Ok(());
    }
    let base = 3 * level;
    let height = wv[level].height;
    let cur_line = wv[level].cur_line;
    if height.saturating_sub(3) <= cur_line && flag & E_HAS_TILES_ON_THE_BOTTOM == 0 {
        if height & 1 != 0 {
            if level > 0 {
                idwt_decode(wv, subs, level - 1, flag)?;
            } else {
                decode_with_iq(&mut subs[base])?;
                copy_subband_to(&mut wv[level], 0, &subs[base]);
            }
            decode_with_iq(&mut subs[base + 1])?;
            copy_subband_to(&mut wv[level], 1, &subs[base + 1]);
        }
    } else {
        if level > 0 {
            idwt_decode(wv, subs, level - 1, flag)?;
        } else {
            decode_with_iq(&mut subs[base])?;
            copy_subband_to(&mut wv[level], 0, &subs[base]);
        }
        for offset in 1..=3 {
            decode_with_iq(&mut subs[base + offset])?;
            copy_subband_to(&mut wv[level], offset, &subs[base + offset]);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn idwt_transform(wv: &mut [Wavelet], level: usize, flag: u8) -> NativeResult<()> {
    if wv[level].cur_h != 0 {
        return Ok(());
    }
    let width = wv[level].width;
    if wv[level].cur_line >= wv[level].height.saturating_sub(3) {
        if flag & E_HAS_TILES_ON_THE_BOTTOM == 0 {
            if wv[level].height & 1 != 0 {
                if level > 0 {
                    if wv[level - 1].cur_h == 0 {
                        idwt_transform(wv, level - 1, flag)?;
                    }
                    let line = idwt_get_line(&mut wv[level - 1]);
                    wv[level].s0 = line;
                }
                let (h0, h1, h2) = (wv[level].flt + 3, (wv[level].flt + 1) % 5 + 3, (wv[level].flt + 2) % 5 + 3);
                let mut l0 = std::mem::take(&mut wv[level].line[0]);
                let l1 = std::mem::take(&mut wv[level].line[1]);
                let l2 = std::mem::take(&mut wv[level].line[2]);
                wv[level].line[2] = l1;
                wv[level].line[1] = l2;
                if width <= 1 {
                    l0[0] = wv[level].s0[0];
                } else {
                    let (mut i0, mut i1) = (0usize, 0usize);
                    if flag & E_HAS_TILES_ON_THE_LEFT != 0 {
                        l0[0] = wv[level].s0[0] - ((wv[level].s1[0] + wv[level].s1[1] + 2) >> 2);
                        i1 += 1;
                    } else {
                        l0[0] = wv[level].s0[0] - ((wv[level].s1[0] + 1) >> 1);
                    }
                    i0 += 1;
                    let mut i = 0usize;
                    while i + 3 < width {
                        let delta = wv[level].s0[i0] - ((wv[level].s1[i1] + wv[level].s1[i1 + 1] + 2) >> 2);
                        l0[i + 1] = wv[level].s1[i1] + ((l0[i] + delta) >> 1);
                        l0[i + 2] = delta;
                        i0 += 1;
                        i1 += 1;
                        i += 2;
                    }
                    if flag & E_HAS_TILES_ON_THE_RIGHT != 0 {
                        let delta = wv[level].s0[i0] - ((wv[level].s1[i1] + wv[level].s1[i1 + 1] + 2) >> 2);
                        l0[i + 1] = wv[level].s1[i1] + ((l0[i] + delta) >> 1);
                        if width & 1 != 0 {
                            l0[i + 2] = delta;
                        }
                    } else if width & 1 != 0 {
                        let delta = wv[level].s0[i0] - ((wv[level].s1[i1] + 1) >> 1);
                        l0[i + 1] = wv[level].s1[i1] + ((l0[i] + delta) >> 1);
                        l0[i + 2] = delta;
                    } else {
                        l0[i + 1] = wv[level].s1[i1] + l0[i];
                    }
                }
                let l1 = std::mem::take(&mut wv[level].line[1]);
                let mut h1v = std::mem::take(&mut wv[level].line[h1]);
                let mut h2v = std::mem::take(&mut wv[level].line[h2]);
                for i in 0..width {
                    let delta = l0[i] - ((l1[i] + 1) >> 1);
                    h1v[i] = l1[i] + ((delta + wv[level].line[h0][i]) >> 1);
                    h2v[i] = delta;
                }
                wv[level].line[0] = l0;
                wv[level].line[1] = l1;
                wv[level].line[h1] = h1v;
                wv[level].line[h2] = h2v;
                wv[level].cur_h += 3;
                wv[level].cur_line += 3;
                wv[level].flt = (wv[level].flt + 3) % 5;
            } else {
                let l2 = std::mem::take(&mut wv[level].line[2]);
                let h0 = wv[level].line[wv[level].flt + 3].clone();
                let mut h1 = std::mem::take(&mut wv[level].line[(wv[level].flt + 1) % 5 + 3]);
                wv[level].line[1] = l2.clone();
                wv[level].line[2] = l2.clone();
                for i in 0..width {
                    h1[i] = h0[i] + l2[i];
                }
                wv[level].line[(wv[level].flt + 1) % 5 + 3] = h1;
                wv[level].cur_h += 2;
                wv[level].cur_line += 2;
                wv[level].flt = (wv[level].flt + 2) % 5;
            }
        }
    } else {
        if level > 0 {
            if wv[level - 1].cur_h == 0 {
                idwt_transform(wv, level - 1, flag)?;
            }
            let line = idwt_get_line(&mut wv[level - 1]);
            wv[level].s0 = line;
        }
        let (h0, h1, h2) = (wv[level].flt + 3, (wv[level].flt + 1) % 5 + 3, (wv[level].flt + 2) % 5 + 3);
        let mut l0 = std::mem::take(&mut wv[level].line[0]);
        let mut l1 = std::mem::take(&mut wv[level].line[1]);
        let l2 = std::mem::take(&mut wv[level].line[2]);
        let mut h1v = std::mem::take(&mut wv[level].line[h1]);
        let mut h2v = std::mem::take(&mut wv[level].line[h2]);
        if width <= 1 {
            l0[0] = wv[level].s0[0];
            l1[0] = wv[level].s2[0];
        } else {
            let (mut i0, mut i1, mut i2, mut i3) = (0usize, 0usize, 0usize, 0usize);
            if flag & E_HAS_TILES_ON_THE_LEFT != 0 {
                l0[0] = wv[level].s0[0] - ((wv[level].s1[0] + wv[level].s1[1] + 2) >> 2);
                l1[0] = wv[level].s2[0] - ((wv[level].s3[0] + wv[level].s3[1] + 2) >> 2);
                i1 += 1;
                i3 += 1;
            } else {
                l0[0] = wv[level].s0[0] - ((wv[level].s1[0] + 1) >> 1);
                l1[0] = wv[level].s2[0] - ((wv[level].s3[0] + 1) >> 1);
            }
            i0 += 1;
            i2 += 1;
            let mut i = 0usize;
            while i + 3 < width {
                let delta = wv[level].s0[i0] - ((wv[level].s1[i1] + wv[level].s1[i1 + 1] + 2) >> 2);
                l0[i + 1] = wv[level].s1[i1] + ((delta + l0[i]) >> 1);
                l0[i + 2] = delta;
                let delta = wv[level].s2[i2] - ((wv[level].s3[i3] + wv[level].s3[i3 + 1] + 2) >> 2);
                l1[i + 1] = wv[level].s3[i3] + ((delta + l1[i]) >> 1);
                l1[i + 2] = delta;
                i0 += 1;
                i1 += 1;
                i2 += 1;
                i3 += 1;
                i += 2;
            }
            if flag & E_HAS_TILES_ON_THE_RIGHT != 0 {
                let da = wv[level].s0[i0] - ((wv[level].s1[i1] + wv[level].s1[i1 + 1] + 2) >> 2);
                l0[i + 1] = wv[level].s1[i1] + ((da + l0[i]) >> 1);
                let db = wv[level].s2[i2] - ((wv[level].s3[i3] + wv[level].s3[i3 + 1] + 2) >> 2);
                l1[i + 1] = wv[level].s3[i3] + ((db + l1[i]) >> 1);
                if width & 1 != 0 {
                    l0[i + 2] = da;
                    l1[i + 2] = db;
                }
            } else if width & 1 != 0 {
                let delta = wv[level].s0[i0] - ((wv[level].s1[i1] + 1) >> 1);
                l0[i + 1] = wv[level].s1[i1] + ((delta + l0[i]) >> 1);
                l0[i + 2] = delta;
                let delta = wv[level].s2[i2] - ((wv[level].s3[i3] + 1) >> 1);
                l1[i + 1] = wv[level].s3[i3] + ((delta + l1[i]) >> 1);
                l1[i + 2] = delta;
            } else {
                l0[i + 1] = l0[i] + wv[level].s1[i1];
                l1[i + 1] = l1[i] + wv[level].s3[i3];
            }
        }
        // C++ swaps lineBuf[1] and lineBuf[2] by pointer.  The buffer that
        // becomes line[2] is the same buffer the L-band reconstruction writes
        // (lineBufL1), so the assignment has to happen *after* that
        // reconstruction; a clone taken before it would freeze zeros.
        wv[level].line[1] = l2.clone();
        wv[level].line[2] = l1.clone();
        for i in 0..width {
            // Re-reading lineBuf[1] after the swap yields the old line[2]
            // (our `l2`) for the H-band prediction.
            let delta = l0[i] - ((l2[i] + l1[i] + 2) >> 2);
            h1v[i] = l2[i] + ((delta + wv[level].line[h0][i]) >> 1);
            h2v[i] = delta;
        }
        wv[level].line[0] = l0;
        wv[level].line[h1] = h1v;
        wv[level].line[h2] = h2v;
        if wv[level].cur_line >= wv[level].height.saturating_sub(3) && wv[level].height & 1 != 0 {
            wv[level].cur_h += 3;
            wv[level].cur_line += 3;
            wv[level].flt = (wv[level].flt + 3) % 5;
        } else {
            wv[level].cur_h += 2;
            wv[level].cur_line += 2;
            wv[level].flt = (wv[level].flt + 2) % 5;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
fn idwt_initialize(wv: &mut [Wavelet], subs: &mut [Subband<'_>], levels: usize, flag: u8) -> NativeResult<()> {
    for level in 0..levels {
        let base = 3 * level;
        if level > 0 {
            let line = idwt_get_line(&mut wv[level - 1]);
            wv[level].s0 = line;
        } else {
            decode_with_iq(&mut subs[base])?;
            copy_subband_to(&mut wv[level], 0, &subs[base]);
        }
        let h0 = wv[level].flt + 3;
        if wv[level].height > 1 {
            decode_with_iq(&mut subs[base + 1])?;
            copy_subband_to(&mut wv[level], 1, &subs[base + 1]);
            decode_with_iq(&mut subs[base + 2])?;
            copy_subband_to(&mut wv[level], 2, &subs[base + 2]);
            decode_with_iq(&mut subs[base + 3])?;
            copy_subband_to(&mut wv[level], 3, &subs[base + 3]);
            let width = wv[level].width;
            if flag & E_HAS_TILES_ON_THE_TOP != 0 {
                let mut l0 = std::mem::take(&mut wv[level].line[0]);
                {
                    let mut l1 = std::mem::take(&mut wv[level].line[1]);
                    horizontal53(&mut l0, &mut l1, &wv[level], flag);
                    wv[level].line[1] = l1;
                }
                decode_with_iq(&mut subs[base + 3])?;
                copy_subband_to(&mut wv[level], 3, &subs[base + 3]);
                decode_with_iq(&mut subs[base + 2])?;
                copy_subband_to(&mut wv[level], 2, &subs[base + 2]);
                let mut l2 = std::mem::take(&mut wv[level].line[2]);
                let (s2, s3) = (wv[level].s2.clone(), wv[level].s3.clone());
                let (mut i2, mut i3) = (0usize, 0usize);
                if width <= 1 {
                    l2[0] = s2[0];
                } else {
                    if flag & E_HAS_TILES_ON_THE_LEFT != 0 {
                        l2[0] = s2[0] - ((s3[0] + s3[1] + 2) >> 2);
                        i3 += 1;
                    } else {
                        l2[0] = s2[0] - ((s3[0] + 1) >> 1);
                    }
                    i2 += 1;
                    let mut i = 0usize;
                    while i + 3 < width {
                        let delta = s2[i2] - ((s3[i3] + s3[i3 + 1] + 2) >> 2);
                        l2[i + 1] = s3[i3] + ((l2[i] + delta) >> 1);
                        l2[i + 2] = delta;
                        i2 += 1;
                        i3 += 1;
                        i += 2;
                    }
                    if flag & E_HAS_TILES_ON_THE_RIGHT != 0 {
                        let delta = s2[i2] - ((s3[i3] + s3[i3 + 1] + 2) >> 2);
                        l2[i + 1] = s3[i3] + ((l2[i] + delta) >> 1);
                        if width & 1 != 0 {
                            l2[i + 2] = delta;
                        }
                    } else if width & 1 != 0 {
                        let delta = s2[i2] - ((s3[i3] + 1) >> 1);
                        l2[i + 1] = s3[i3] + ((l2[i] + delta) >> 1);
                        l2[i + 2] = delta;
                    } else {
                        l2[i + 1] = s3[i3] + l2[i];
                    }
                }
                let mut h0v = std::mem::take(&mut wv[level].line[h0]);
                let l1 = wv[level].line[1].clone();
                for i in 0..width {
                    h0v[i] = l0[i] - ((l1[i] + l2[i] + 2) >> 2);
                }
                wv[level].line[0] = l0;
                wv[level].line[2] = l2;
                wv[level].line[h0] = h0v;
            } else {
                let mut l0 = std::mem::take(&mut wv[level].line[0]);
                let mut l2 = std::mem::take(&mut wv[level].line[2]);
                horizontal53(&mut l0, &mut l2, &wv[level], flag);
                wv[level].line[2] = l2.clone();
                let mut h0v = std::mem::take(&mut wv[level].line[h0]);
                for i in 0..width {
                    h0v[i] = l0[i] - ((l2[i] + 1) >> 1);
                }
                wv[level].line[0] = l0;
                wv[level].line[h0] = h0v;
            }
            idwt_decode(wv, subs, level, flag)?;
            idwt_transform(wv, level, flag)?;
        } else {
            decode_with_iq(&mut subs[base + 1])?;
            copy_subband_to(&mut wv[level], 1, &subs[base + 1]);
            let (s0, s1) = (wv[level].s0.clone(), wv[level].s1.clone());
            let mut h0v = std::mem::take(&mut wv[level].line[h0]);
            let width = wv[level].width;
            if width <= 1 {
                h0v[0] = s0[0];
            } else {
                let (mut i0, mut i1) = (0usize, 0usize);
                if flag & E_HAS_TILES_ON_THE_LEFT != 0 {
                    h0v[0] = s0[0] - ((s1[0] + s1[1] + 2) >> 2);
                    i1 += 1;
                } else {
                    h0v[0] = s0[0] - ((s1[0] + 1) >> 1);
                }
                i0 += 1;
                let mut i = 0usize;
                while i + 3 < width {
                    let delta = s0[i0] - ((s1[i1] + s1[i1 + 1] + 2) >> 2);
                    h0v[i + 1] = s1[i1] + ((h0v[i] + delta) >> 1);
                    h0v[i + 2] = delta;
                    i0 += 1;
                    i1 += 1;
                    i += 2;
                }
                if flag & E_HAS_TILES_ON_THE_RIGHT != 0 {
                    let delta = s0[i0] - ((s1[i1] + s1[i1 + 1] + 2) >> 2);
                    h0v[i + 1] = s1[i1] + ((h0v[i] + delta) >> 1);
                    h0v[i + 2] = delta;
                } else if width & 1 != 0 {
                    let delta = s0[i0] - ((s1[i1] + 1) >> 1);
                    h0v[i + 1] = s1[i1] + ((h0v[i] + delta) >> 1);
                    h0v[i + 2] = delta;
                } else {
                    h0v[i + 1] = s1[i1] + h0v[i];
                }
            }
            wv[level].line[h0] = h0v;
            wv[level].cur_line += 1;
            wv[level].cur_h += 1;
            wv[level].flt = (wv[level].flt + 1) % 5;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Container-level structures

pub(crate) struct Tile {
    pub width: usize,
    pub height: usize,
    pub flags: u8,
    pub data_offset: u64,
    pub comps: Vec<Comp>,
}

pub(crate) struct Comp {
    pub data_offset: u64,
    pub supports_partial: bool,
    pub bands: Vec<BandHeader>,
}

pub(crate) struct BandHeader {
    pub data_offset: u64,
    pub data_size: usize,
    pub q_param: i32,
}

/// Parses the mdat header of one CRX sample.
pub(crate) fn parse_mdat_header(
    sample: &[u8],
    header: &CrxHeader,
    plane_width: usize,
    plane_height: usize,
    tile_width: usize,
    tile_height: usize,
) -> NativeResult<Vec<Tile>> {
    let tile_cols = plane_width.div_ceil(tile_width);
    let tile_rows = plane_height.div_ceil(tile_height);
    let mut tiles = Vec::with_capacity(tile_cols * tile_rows);
    let mut pos = 0usize;
    let end = header.mdat_header_size.min(sample.len());
    let mut tile_offset = 0u64;
    for tile_index in 0..tile_cols * tile_rows {
        let (sign, hdr_size) = read_header(sample, &mut pos, end)?;
        if !matches!((sign, hdr_size), (0xFF01, 8) | (0xFF11, 8) | (0xFF11, 16)) {
            return Err(NativeError::corrupt("CRX tile header is invalid"));
        }
        let tail = be16(sample, pos + 10)?;
        if (hdr_size == 8 && tail != 0) || (hdr_size == 16 && tail != 0x4000) {
            return Err(NativeError::corrupt("CRX tile header tail is invalid"));
        }
        if be16(sample, pos + 8)? as usize != tile_index {
            return Err(NativeError::corrupt("CRX tile index is out of order"));
        }
        let tile_size = be32(sample, pos + 4)? as u64;
        let (qp_size, extra_size) = if hdr_size == 16 {
            (be32(sample, pos + 12)? as u64, be16(sample, pos + 16)? as u64)
        } else {
            (0, 0)
        };
        pos += hdr_size + 4;
        let mut comps = Vec::with_capacity(header.planes as usize);
        let mut comp_offset = 0u64;
        for comp_index in 0..header.planes {
            let (sign, hdr_size) = read_header(sample, &mut pos, end)?;
            if sign != 0xFF02 || hdr_size != 8 {
                return Err(NativeError::corrupt("CRX plane header is invalid"));
            }
            if (sample[pos + 8] >> 4) as u8 != comp_index {
                return Err(NativeError::corrupt("CRX plane index is out of order"));
            }
            let comp_size = be32(sample, pos + 4)? as u64;
            let supports_partial = sample[pos + 8] & 8 != 0;
            let rounded = (sample[pos + 8] >> 1) & 3;
            if rounded != 0 {
                return Err(NativeError::unsupported("rounded CRX plane headers"));
            }
            pos += 12;
            let mut bands = Vec::with_capacity(3 * header.levels as usize + 1);
            let mut band_offset = 0u64;
            for band_index in 0..3 * header.levels as usize + 1 {
                let (sign, hdr_size) = read_header(sample, &mut pos, end)?;
                if sign != 0xFF03 || hdr_size != 8 {
                    return Err(NativeError::unsupported("new-style CRX subband header"));
                }
                let band_size = be32(sample, pos + 4)? as u64;
                let bit_data = be32(sample, pos + 8)?;
                if (sample[pos + 8] >> 4) as usize != band_index {
                    return Err(NativeError::corrupt("CRX band index is out of order"));
                }
                let data_size = band_size.saturating_sub((bit_data & 0x7FFFF) as u64) as usize;
                let q_param = ((bit_data >> 19) & 0xFF) as i32;
                bands.push(BandHeader { data_offset: band_offset, data_size, q_param });
                band_offset += band_size;
                pos += 12;
            }
            comps.push(Comp { data_offset: comp_offset, supports_partial, bands });
            comp_offset += comp_size;
        }
        let width = if (tile_index + 1) % tile_cols != 0 {
            tile_width
        } else {
            plane_width - tile_width * (tile_cols - 1)
        };
        let height = if tile_index < (tile_rows - 1) * tile_cols {
            tile_height
        } else {
            plane_height - tile_height * (tile_rows - 1)
        };
        let mut flags = 0u8;
        if tile_cols > 1 {
            if (tile_index + 1) % tile_cols != 0 {
                flags |= E_HAS_TILES_ON_THE_RIGHT;
            }
            if tile_index % tile_cols != 0 {
                flags |= E_HAS_TILES_ON_THE_LEFT;
            }
        }
        if tile_rows > 1 {
            if tile_index < (tile_rows - 1) * tile_cols {
                flags |= E_HAS_TILES_ON_THE_BOTTOM;
            }
            if tile_index >= tile_cols {
                flags |= E_HAS_TILES_ON_THE_TOP;
            }
        }
        tiles.push(Tile {
            width,
            height,
            flags,
            data_offset: tile_offset + qp_size + extra_size,
            comps,
        });
        tile_offset += tile_size + qp_size + extra_size;
    }
    Ok(tiles)
}

fn read_header(sample: &[u8], pos: &mut usize, end: usize) -> NativeResult<(u16, usize)> {
    if *pos + 4 > end {
        return Err(NativeError::corrupt("truncated CRX mdat header"));
    }
    let sign = be16(sample, *pos)?;
    let size = be16(sample, *pos + 2)? as usize;
    Ok((sign, size))
}

fn be16(data: &[u8], at: usize) -> NativeResult<u16> {
    data.get(at..at + 2)
        .map(|bytes| u16::from_be_bytes([bytes[0], bytes[1]]))
        .ok_or_else(|| NativeError::corrupt("truncated CRX header"))
}

fn be32(data: &[u8], at: usize) -> NativeResult<u32> {
    data.get(at..at + 4)
        .map(|bytes| u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .ok_or_else(|| NativeError::corrupt("truncated CRX header"))
}

/// Decodes the four CRX planes into one packed sensor plane.
pub(crate) fn decode_sample(sample: &[u8], header: &CrxHeader) -> NativeResult<(Vec<u16>, usize, usize)> {
    if header.planes != 4 || header.enc_type != 0 {
        return Err(NativeError::unsupported("CRX only supports four-plane encType 0"));
    }
    let plane_width = header.width / 2;
    let plane_height = header.height / 2;
    let tile_width = header.tile_width / 2;
    let tile_height = header.tile_height / 2;
    if plane_width == 0 || plane_height == 0 || tile_width == 0 || tile_height == 0 {
        return Err(NativeError::corrupt("CRX plane geometry is empty"));
    }
    let tiles = parse_mdat_header(sample, header, plane_width, plane_height, tile_width, tile_height)?;
    let raw_width = plane_width * 2;
    let raw_height = plane_height * 2;
    let mut out = vec![0u16; raw_width * raw_height];
    let median = 1i32 << (header.bits - 1);
    let max_value = (1i32 << header.bits) - 1;
    let mdat_start = header.mdat_header_size;

    for plane in 0..4u8 {
        // Plane -> (row offset, column offset) inside the 2x2 CFA cell.
        let (px, py) = match (header.cfa_layout, plane) {
            (0, 0) => (0, 0),
            (0, 1) => (1, 0),
            (0, 2) => (0, 1),
            (0, 3) => (1, 1),
            (1, 0) => (1, 0),
            (1, 1) => (0, 0),
            (1, 2) => (1, 1),
            (1, 3) => (0, 1),
            (2, 0) => (0, 1),
            (2, 1) => (1, 1),
            (2, 2) => (0, 0),
            (2, 3) => (1, 0),
            (3, 0) => (1, 1),
            (3, 1) => (0, 1),
            (3, 2) => (1, 0),
            (3, 3) => (0, 0),
            _ => return Err(NativeError::corrupt("CRX CFA layout is invalid")),
        };
        let mut image_row = 0usize;
        for tile_row in 0..(plane_height.div_ceil(tile_height)) {
            let mut image_col = 0usize;
            for tile_col in 0..(plane_width.div_ceil(tile_width)) {
                let tile = &tiles[tile_row * plane_width.div_ceil(tile_width) + tile_col];
                let comp = &tile.comps[plane as usize];
                let geometry = setup_subband_geometry(tile, header.levels);
                let mut subs = Vec::with_capacity(geometry.len());
                for (index, (width, height)) in geometry.iter().enumerate() {
                    let band = &comp.bands[index];
                    let absolute = mdat_start as u64 + tile.data_offset + comp.data_offset + band.data_offset;
                    let param = if band.data_size > 0 {
                        Some(BandParam::new(
                            sample,
                            absolute as usize,
                            band.data_size,
                            *width,
                            *height,
                            comp.supports_partial && index == 0,
                        ))
                    } else {
                        None
                    };
                    subs.push(Subband {
                        width: *width,
                        height: *height,
                        q_param: band.q_param,
                        data_size: band.data_size,
                        buf: vec![0i32; *width],
                        param,
                    });
                }
                if header.levels == 0 {
                    let width = tile.width;
                    for row in 0..tile.height {
                        decode_with_iq(&mut subs[0])?;
                        let line = subs[0].buf.clone();
                        convert_line(
                            &mut out,
                            raw_width,
                            2 * image_row + 2 * row + py,
                            2 * image_col + px,
                            &line,
                            width,
                            median,
                            max_value,
                            2,
                        );
                    }
                } else {
                    let levels = header.levels as usize;
                    let mut wv = Vec::with_capacity(levels);
                    for level in 0..levels {
                        let (width, height) = if level >= levels - 1 {
                            (tile.width, tile.height)
                        } else {
                            let geometry = &geometry;
                            let w = geometry[3 * level + 5].0;
                            let h = geometry[3 * level + 4].1;
                            (w, h)
                        };
                        wv.push(Wavelet::new(width, height));
                    }
                    idwt_initialize(&mut wv, &mut subs, levels, tile.flags)?;
                    for row in 0..tile.height {
                        idwt_decode(&mut wv, &mut subs, levels - 1, tile.flags)?;
                        idwt_transform(&mut wv, levels - 1, tile.flags)?;
                        let line = idwt_get_line(&mut wv[levels - 1]);
                        convert_line(
                            &mut out,
                            raw_width,
                            2 * image_row + 2 * row + py,
                            2 * image_col + px,
                            &line,
                            tile.width,
                            median,
                            max_value,
                            2,
                        );
                    }
                }
                image_col += tile.width;
            }
            image_row += tiles[tile_row * plane_width.div_ceil(tile_width)].height;
        }
    }
    Ok((out, raw_width, raw_height))
}

#[allow(clippy::too_many_arguments)]
fn convert_line(
    out: &mut [u16],
    raw_width: usize,
    row: usize,
    col: usize,
    line: &[i32],
    width: usize,
    median: i32,
    max_value: i32,
    step: usize,
) {
    for (index, &value) in line.iter().take(width).enumerate() {
        let sample = (median + value).clamp(0, max_value) as u16;
        if let Some(slot) = out.get_mut(row * raw_width + col + index * step) {
            *slot = sample;
        }
    }
}
