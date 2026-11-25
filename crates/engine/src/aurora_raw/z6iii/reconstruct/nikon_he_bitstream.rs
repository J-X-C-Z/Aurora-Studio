//! Nikon HE VCGH precinct coefficient reader for the 6064-wide r206 profile.

use std::io;

pub type DecodeError = io::Error;

const SUBBANDS: [u8; 25] = [
    0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 36, 54, 55, 56, 57, 58, 59, 6, 7, 24, 25, 60, 61,
];
pub(super) const PACKET_BANDS: [&[usize]; 8] = [
    &[0, 1, 2, 3, 4, 5],
    &[6, 7, 8, 9, 10, 11],
    &[12],
    &[13, 14, 15, 16, 17, 18],
    &[19, 20],
    &[21, 22],
    &[25],
    &[23, 24],
];
const GROUPS: [usize; 26] = [
    24, 24, 48, 95, 190, 379, 24, 24, 48, 95, 190, 379, 758, 24, 24, 48, 95, 190, 379, 379,
    379, 379, 379, 379, 379, 758,
];
const MAX_GROUPS: usize = 758;

/// Total number of dequantized i32 coefficients across all 26 subband slots in one precinct.
pub(super) const TOTAL_COEFFS: usize = 24_280;

/// Starting coefficient offset for each of the 26 subband slots (plus sentinel end).
pub(super) const SLOT_OFFSETS: [usize; 27] = {
    let mut offsets = [0usize; 27];
    let mut i = 0;
    while i < 26 {
        offsets[i + 1] = offsets[i] + GROUPS[i] * 4;
        i += 1;
    }
    offsets
};

#[inline(always)]
pub(super) fn slot_slice(buf: &[i32; TOTAL_COEFFS], slot: usize) -> &[i32] {
    &buf[SLOT_OFFSETS[slot]..SLOT_OFFSETS[slot + 1]]
}

#[derive(Clone, Copy, Debug)]
pub struct Weights([(u8, u8); 25]);

impl Weights {
    /// Parses exactly the complete FF14 WGT marker, including its length field.
    pub fn from_wgt(marker: &[u8]) -> Result<Self, DecodeError> {
        if marker.len() != 54 || marker[..4] != [0xff, 0x14, 0, 52] {
            return Err(invalid("expected 54-byte VCGH WGT marker"));
        }
        let mut weights = [(0, 0); 25];
        for (weight, pair) in weights.iter_mut().zip(marker[4..].chunks_exact(2)) {
            *weight = (pair[0], pair[1]);
        }
        Ok(Self(weights))
    }
}

#[derive(Debug)]
pub struct Precinct {
    pub bytes_consumed: usize,
    pub qp: u8,
    pub rp: u8,
    pub packets: [Packet; 8],
}

#[derive(Debug, Default)]
pub struct Packet {
    pub bands: Vec<Band>,
}

#[derive(Debug)]
pub struct Band {
    pub subband: u8,
    pub gtli: u8,
    pub groups: Vec<Group>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Group {
    pub nbits: u8,
    pub coeffs: [i16; 4],
}

struct PredSlot {
    gcli: [u8; MAX_GROUPS],
    gtli: u8,
    empty: bool,
}

struct PredState {
    slots: [PredSlot; 26],
}

impl PredState {
    fn new_boxed() -> Box<Self> {
        // Heap-allocate directly to avoid putting ~20 KB on the stack.
        let mut slots = Vec::with_capacity(26);
        for _ in 0..26 {
            slots.push(PredSlot {
                gcli: [0u8; MAX_GROUPS],
                gtli: 0,
                empty: true,
            });
        }
        let slots: [PredSlot; 26] = match slots.try_into() {
            Ok(arr) => arr,
            Err(_) => unreachable!(),
        };
        Box::new(Self { slots })
    }
}

pub struct Decoder {
    weights: Weights,
    preds: [Box<PredState>; 2],
    prev_idx: usize,
    has_prev: bool,
}

impl Decoder {
    pub fn new(weights: Weights) -> Self {
        Self {
            weights,
            preds: [PredState::new_boxed(), PredState::new_boxed()],
            prev_idx: 0,
            has_prev: false,
        }
    }

    /// Call after consuming a slice header, before decoding that slice's first precinct.
    pub fn reset_slice(&mut self) {
        self.has_prev = false;
    }

    /// Starts at a precinct length field. Caller validates the outer VCGH profile and
    /// bounds `input` to the enclosing payload. On success, advance by `bytes_consumed`.
    /// An error leaves the previous prediction state unchanged.
    pub fn decode_precinct(&mut self, input: &[u8]) -> Result<Precinct, DecodeError> {
        let header = input.get(..12).ok_or_else(truncated)?;
        let bytes_consumed = 12
            + (usize::from(header[0]) << 16)
            + (usize::from(header[1]) << 8)
            + usize::from(header[2]);
        let mut body = input.get(12..bytes_consumed).ok_or_else(truncated)?;
        let (qp, rp) = (header[3], header[4]);
        // shortcut: only the observed significance+vertical profile (25 subbands * 2 bits = 50 ones).
        if header[5] != 0xff
            || header[6] != 0xff
            || header[7] != 0xff
            || header[8] != 0xff
            || header[9] != 0xff
            || header[10] != 0xff
            || (header[11] & 0xc0) != 0xc0
        {
            return Err(unsupported("expected significance+vertical GCLI flags"));
        }

        let cur_idx = 1 - self.prev_idx;
        let (prev_state, cur_state) = if cur_idx == 1 {
            let (left, right) = self.preds.split_at_mut(1);
            (if self.has_prev { Some(&*left[0]) } else { None }, &mut *right[0])
        } else {
            let (left, right) = self.preds.split_at_mut(1);
            (if self.has_prev { Some(&*right[0]) } else { None }, &mut *left[0])
        };

        let mut packets: [Packet; 8] = std::array::from_fn(|_| Packet::default());
        for (packet_index, (packet, slots)) in packets.iter_mut().zip(PACKET_BANDS).enumerate() {
            let mut parse_packet = || -> Result<(), DecodeError> {
                let ph = take(&mut body, 7)?;
                let hdr = u64::from_be_bytes([0, ph[0], ph[1], ph[2], ph[3], ph[4], ph[5], ph[6]]);
                if (hdr >> 55) & 1 != 0 {
                    return Err(unsupported("raw packet"));
                }
                let data_len = ((hdr >> 35) & 0xfffff) as usize;
                let gcli_len = ((hdr >> 15) & 0xfffff) as usize;
                let sign_len = (hdr & 0x7fff) as usize;
                let sig_bits: usize = slots.iter().map(|&slot| GROUPS[slot].div_ceil(8)).sum();
                let mut sig = Bits::new(take(&mut body, sig_bits.div_ceil(8))?);
                let mut gcli = Bits::new(take(&mut body, gcli_len)?);
                let mut data = Bits::new(take(&mut body, data_len)?);
                let mut sign = Bits::new(take(&mut body, sign_len)?);
                for &slot in slots {
                    let weight_slot = if slot == 25 { 12 } else { slot };
                    let (gain, priority) = self.weights.0[weight_slot];
                    let gtli = qp
                        .saturating_sub(gain)
                        .saturating_sub(u8::from(priority < rp))
                        .min(15);
                    let group_count = GROUPS[slot];
                    let pred_info: Option<(&[u8; MAX_GROUPS], bool, u8)> = if slot == 25 {
                        let c12 = &cur_state.slots[12];
                        Some((&c12.gcli, c12.empty, gtli))
                    } else {
                        prev_state.map(|prev| {
                            let p = &prev.slots[if slot == 12 { 25 } else { slot }];
                            if packet_index < 4 {
                                (&p.gcli, false, p.gtli)
                            } else {
                                (&p.gcli, p.empty, gtli)
                            }
                        })
                    };
                    let mut groups = Vec::with_capacity(group_count);
                    let mut cur_gcli = [0u8; MAX_GROUPS];
                    let mut empty = true;
                    let use_pred = match pred_info {
                        Some((_, pred_empty, pred_gtli)) => !(pred_empty && pred_gtli <= gtli),
                        None => false,
                    };
                    let pred_gcli = pred_info.map(|(g, _, _)| g);

                    for start in (0..group_count).step_by(8) {
                        let insignificant = sig.read_bit()? != 0;
                        empty &= insignificant;
                        let end = (start + 8).min(group_count);
                        for index in start..end {
                            let unary = if insignificant { 0 } else { gcli.unary()? };
                            let planes = if use_pred {
                                // SAFETY: use_pred is true only when pred_gcli is Some.
                                let prev_val = pred_gcli.unwrap()[index];
                                predict(gtli, prev_val, unary)?
                            } else {
                                gtli + unary
                            };
                            let nbits = planes - gtli;
                            if nbits > 15 {
                                return Err(invalid("coefficient bit depth exceeds 15"));
                            }
                            cur_gcli[index] = planes;
                            let mut coeffs = [0i16; 4];
                            if nbits > 0 {
                                let mags = data.read_4_bitplanes(nbits)?;
                                for i in 0..4 {
                                    let m = mags[i] as i16;
                                    coeffs[i] = if m != 0 && sign.read_bit()? != 0 { -m } else { m };
                                }
                            }
                            groups.push(Group { nbits, coeffs });
                        }
                    }
                    cur_state.slots[slot].gcli[..group_count]
                        .copy_from_slice(&cur_gcli[..group_count]);
                    cur_state.slots[slot].gtli = gtli;
                    cur_state.slots[slot].empty = empty;
                    packet.bands.push(Band {
                        subband: SUBBANDS[weight_slot],
                        gtli,
                        groups,
                    });
                }
                sig.finish()?;
                gcli.finish()?;
                data.finish()?;
                sign.finish()?;
                Ok(())
            };
            parse_packet().map_err(|err| {
                io::Error::new(err.kind(), format!("packet {packet_index}: {err}"))
            })?;
        }
        self.prev_idx = cur_idx;
        self.has_prev = true;
        Ok(Precinct {
            bytes_consumed,
            qp,
            rp,
            packets,
        })
    }

    /// Zero-allocation fast path: decodes one precinct and dequantizes all 26 subband slots
    /// directly into `out_coeffs`, returning `bytes_consumed`.
    /// An error leaves the previous prediction state unchanged.
    pub(super) fn decode_precinct_dequantized(
        &mut self,
        input: &[u8],
        out_coeffs: &mut [i32; TOTAL_COEFFS],
        reciprocals: &[u32; 15],
    ) -> Result<usize, DecodeError> {
        let header = input.get(..12).ok_or_else(truncated)?;
        let bytes_consumed = 12
            + (usize::from(header[0]) << 16)
            + (usize::from(header[1]) << 8)
            + usize::from(header[2]);
        let mut body = input.get(12..bytes_consumed).ok_or_else(truncated)?;
        let (qp, rp) = (header[3], header[4]);
        if header[5] != 0xff
            || header[6] != 0xff
            || header[7] != 0xff
            || header[8] != 0xff
            || header[9] != 0xff
            || header[10] != 0xff
            || (header[11] & 0xc0) != 0xc0
        {
            return Err(unsupported("expected significance+vertical GCLI flags"));
        }

        let cur_idx = 1 - self.prev_idx;
        let (prev_state, cur_state) = if cur_idx == 1 {
            let (left, right) = self.preds.split_at_mut(1);
            (if self.has_prev { Some(&*left[0]) } else { None }, &mut *right[0])
        } else {
            let (left, right) = self.preds.split_at_mut(1);
            (if self.has_prev { Some(&*right[0]) } else { None }, &mut *left[0])
        };

        for (packet_index, &slots) in PACKET_BANDS.iter().enumerate() {
            let mut parse_packet = || -> Result<(), DecodeError> {
                let ph = take(&mut body, 7)?;
                let hdr = u64::from_be_bytes([0, ph[0], ph[1], ph[2], ph[3], ph[4], ph[5], ph[6]]);
                if (hdr >> 55) & 1 != 0 {
                    return Err(unsupported("raw packet"));
                }
                let data_len = ((hdr >> 35) & 0xfffff) as usize;
                let gcli_len = ((hdr >> 15) & 0xfffff) as usize;
                let sign_len = (hdr & 0x7fff) as usize;
                let sig_bits: usize = slots.iter().map(|&slot| GROUPS[slot].div_ceil(8)).sum();
                let mut sig = Bits::new(take(&mut body, sig_bits.div_ceil(8))?);
                let mut gcli = Bits::new(take(&mut body, gcli_len)?);
                let mut data = Bits::new(take(&mut body, data_len)?);
                let mut sign = Bits::new(take(&mut body, sign_len)?);

                for &slot in slots {
                    let weight_slot = if slot == 25 { 12 } else { slot };
                    let (gain, priority) = self.weights.0[weight_slot];
                    let gtli = qp
                        .saturating_sub(gain)
                        .saturating_sub(u8::from(priority < rp))
                        .min(15);
                    let group_count = GROUPS[slot];
                    let shift = 16 - gtli;

                    // Borrow predictor gcli without aliasing cur_state.slots[slot].
                    // Only slot == 25 reads from cur_state.slots[12], so split_at_mut works cleanly.
                    let (use_pred, pred_slice, cur_slot): (bool, Option<&[u8]>, &mut PredSlot) =
                        if slot == 25 {
                            let (lo, hi) = cur_state.slots.split_at_mut(25);
                            let c12 = &lo[12];
                            let use_p = !(c12.empty && gtli <= gtli); // i.e. !c12.empty
                            (use_p, Some(&c12.gcli[..group_count]), &mut hi[0])
                        } else {
                            let pred = prev_state.map(|prev| {
                                let p = &prev.slots[if slot == 12 { 25 } else { slot }];
                                let (p_empty, p_gtli) = if packet_index < 4 {
                                    (false, p.gtli)
                                } else {
                                    (p.empty, gtli)
                                };
                                (!(p_empty && p_gtli <= gtli), &p.gcli[..group_count])
                            });
                            match pred {
                                Some((use_p, sl)) => (use_p, Some(sl), &mut cur_state.slots[slot]),
                                None => (false, None, &mut cur_state.slots[slot]),
                            }
                        };

                    let out_slot = &mut out_coeffs[SLOT_OFFSETS[slot]..SLOT_OFFSETS[slot + 1]];
                    let cur_gcli = &mut cur_slot.gcli[..group_count];
                    let mut empty = true;

                    for start in (0..group_count).step_by(8) {
                        let insignificant = sig.read_bit()? != 0;
                        empty &= insignificant;
                        let end = (start + 8).min(group_count);
                        for index in start..end {
                            let unary = if insignificant { 0 } else { gcli.unary()? };
                            let planes = if use_pred {
                                predict(gtli, pred_slice.unwrap()[index], unary)?
                            } else {
                                gtli + unary
                            };
                            let nbits = planes - gtli;
                            if nbits > 15 {
                                return Err(invalid("coefficient bit depth exceeds 15"));
                            }
                            cur_gcli[index] = planes;
                            let dst = &mut out_slot[index * 4..index * 4 + 4];
                            if nbits == 0 {
                                dst[0] = 0;
                                dst[1] = 0;
                                dst[2] = 0;
                                dst[3] = 0;
                            } else if nbits == 1 {
                                let p = data.read_4bits()?;
                                let level = ((reciprocals[0] >> shift) << 4) as i32;
                                if sign.bit_count < 4 {
                                    sign.refill();
                                }
                                dst[0] = if (p & 8) != 0 {
                                    if sign.read_bit()? != 0 { -level } else { level }
                                } else {
                                    0
                                };
                                dst[1] = if (p & 4) != 0 {
                                    if sign.read_bit()? != 0 { -level } else { level }
                                } else {
                                    0
                                };
                                dst[2] = if (p & 2) != 0 {
                                    if sign.read_bit()? != 0 { -level } else { level }
                                } else {
                                    0
                                };
                                dst[3] = if (p & 1) != 0 {
                                    if sign.read_bit()? != 0 { -level } else { level }
                                } else {
                                    0
                                };
                            } else {
                                let mags = data.read_4_bitplanes(nbits)?;
                                let recip = reciprocals[usize::from(nbits - 1)];
                                if sign.bit_count < 4 {
                                    sign.refill();
                                }
                                for i in 0..4 {
                                    let m = mags[i];
                                    if m == 0 {
                                        dst[i] = 0;
                                    } else {
                                        let level =
                                            ((u32::from(m).wrapping_mul(recip) >> shift) << 4)
                                                as i32;
                                        dst[i] =
                                            if sign.read_bit()? != 0 { -level } else { level };
                                    }
                                }
                            }
                        }
                    }
                    cur_slot.gtli = gtli;
                    cur_slot.empty = empty;
                }
                sig.finish()?;
                gcli.finish()?;
                data.finish()?;
                sign.finish()?;
                Ok(())
            };
            parse_packet().map_err(|err| {
                io::Error::new(err.kind(), format!("packet {packet_index}: {err}"))
            })?;
        }
        self.prev_idx = cur_idx;
        self.has_prev = true;
        Ok(bytes_consumed)
    }
}

#[inline(always)]
fn predict(gtli: u8, previous: u8, unary: u8) -> Result<u8, DecodeError> {
    if previous > 15 {
        return Err(invalid("previous GCLI exceeds 15"));
    }
    let down = previous.saturating_sub(gtli);
    if unary > 15 - gtli + down {
        return Err(invalid("invalid bounded unary GCLI prediction"));
    }
    let base = previous.max(gtli);
    Ok(if unary > 2 * down {
        base + unary - down
    } else if unary & 1 == 0 {
        base + (unary >> 1)
    } else {
        base - ((unary + 1) >> 1)
    })
}

#[inline(always)]
fn take<'a>(input: &mut &'a [u8], len: usize) -> Result<&'a [u8], DecodeError> {
    let (head, tail) = input.split_at_checked(len).ok_or_else(truncated)?;
    *input = tail;
    Ok(head)
}

/// MSB-first buffered bit reader.
/// Invariant: the lowest `(64 - bit_count)` bits of `bit_buf` are always 0.
struct Bits<'a> {
    bytes: &'a [u8],
    byte_pos: usize,
    bit_buf: u64,
    bit_count: u32,
}

impl<'a> Bits<'a> {
    #[inline(always)]
    fn new(bytes: &'a [u8]) -> Self {
        let mut bits = Self {
            bytes,
            byte_pos: 0,
            bit_buf: 0,
            bit_count: 0,
        };
        bits.refill();
        bits
    }

    #[inline(always)]
    fn refill(&mut self) {
        if self.bit_count <= 32 && self.byte_pos + 4 <= self.bytes.len() {
            let word = u32::from_be_bytes([
                self.bytes[self.byte_pos],
                self.bytes[self.byte_pos + 1],
                self.bytes[self.byte_pos + 2],
                self.bytes[self.byte_pos + 3],
            ]);
            self.bit_buf |= u64::from(word) << (32 - self.bit_count);
            self.byte_pos += 4;
            self.bit_count += 32;
        } else {
            while self.bit_count <= 56 && self.byte_pos < self.bytes.len() {
                self.bit_buf |= u64::from(self.bytes[self.byte_pos]) << (56 - self.bit_count);
                self.byte_pos += 1;
                self.bit_count += 8;
            }
        }
    }

    #[inline(always)]
    fn read_bit(&mut self) -> Result<u32, DecodeError> {
        if self.bit_count == 0 {
            self.refill();
            if self.bit_count == 0 {
                return Err(truncated());
            }
        }
        let bit = (self.bit_buf >> 63) as u32;
        self.bit_buf <<= 1;
        self.bit_count -= 1;
        Ok(bit)
    }

    #[inline(always)]
    fn read_4bits(&mut self) -> Result<u32, DecodeError> {
        if self.bit_count < 4 {
            self.refill();
            if self.bit_count < 4 {
                return Err(truncated());
            }
        }
        let val = (self.bit_buf >> 60) as u32;
        self.bit_buf <<= 4;
        self.bit_count -= 4;
        Ok(val)
    }

    #[inline(always)]
    fn read_4_bitplanes(&mut self, nbits: u8) -> Result<[u16; 4], DecodeError> {
        debug_assert!((1..=15).contains(&nbits));
        let mut c0 = 0u16;
        let mut c1 = 0u16;
        let mut c2 = 0u16;
        let mut c3 = 0u16;

        let mut remaining = nbits;
        while remaining > 0 {
            let chunk = remaining.min(8);
            let need = u32::from(chunk) * 4;
            if self.bit_count < need {
                self.refill();
                if self.bit_count < need {
                    return Err(truncated());
                }
            }
            let mut w = self.bit_buf;
            self.bit_buf <<= need;
            self.bit_count -= need;
            for _ in 0..chunk {
                let p = (w >> 60) as u16;
                w <<= 4;
                c0 = (c0 << 1) | ((p >> 3) & 1);
                c1 = (c1 << 1) | ((p >> 2) & 1);
                c2 = (c2 << 1) | ((p >> 1) & 1);
                c3 = (c3 << 1) | (p & 1);
            }
            remaining -= chunk;
        }
        Ok([c0, c1, c2, c3])
    }

    #[inline(always)]
    fn unary(&mut self) -> Result<u8, DecodeError> {
        if self.bit_count < 32 {
            self.refill();
        }
        let ones = (!self.bit_buf).leading_zeros();
        if ones < self.bit_count && ones <= 31 {
            let consume = ones + 1;
            self.bit_buf <<= consume;
            self.bit_count -= consume;
            return Ok(ones as u8);
        }
        if self.bit_count >= 32 && ones >= 32 {
            return Err(invalid("unterminated or oversized unary GCLI code"));
        }
        Err(truncated())
    }

    #[inline(always)]
    fn finish(&self) -> Result<(), DecodeError> {
        let bits_read = self.byte_pos * 8 - self.bit_count as usize;
        if bits_read.div_ceil(8) != self.bytes.len() {
            return Err(invalid("unconsumed whole byte in packet substream"));
        }
        Ok(())
    }
}

fn invalid(message: &str) -> DecodeError {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn unsupported(message: &str) -> DecodeError {
    io::Error::new(io::ErrorKind::Unsupported, message)
}

fn truncated() -> DecodeError {
    io::Error::new(
        io::ErrorKind::UnexpectedEof,
        "truncated precinct or packet substream",
    )
}
