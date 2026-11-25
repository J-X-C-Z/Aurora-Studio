//! Nikon HE VCGH precinct coefficient reader for the 6064-wide r206 profile.

use std::io;

pub type DecodeError = io::Error;

const SUBBANDS: [u8; 25] = [
    0, 1, 2, 3, 4, 5, 18, 19, 20, 21, 22, 23, 36, 54, 55, 56, 57, 58, 59, 6, 7, 24, 25, 60, 61,
];
const PACKET_BANDS: [&[usize]; 8] = [
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

#[derive(Default)]
struct Prediction {
    gcli: Vec<u8>,
    gtli: u8,
    empty: bool,
}

pub struct Decoder {
    weights: Weights,
    previous: Option<[Prediction; 26]>,
}

impl Decoder {
    pub fn new(weights: Weights) -> Self {
        Self { weights, previous: None }
    }

    /// Call after consuming a slice header, before decoding that slice's first precinct.
    pub fn reset_slice(&mut self) {
        self.previous = None;
    }

    /// Starts at a precinct length field. Caller validates the outer VCGH profile and
    /// bounds `input` to the enclosing payload. On success, advance by `bytes_consumed`.
    /// An error leaves the previous prediction state unchanged.
    pub fn decode_precinct(&mut self, input: &[u8]) -> Result<Precinct, DecodeError> {
        let header = input.get(..12).ok_or_else(truncated)?;
        let bytes_consumed = 12 + (usize::from(header[0]) << 16)
            + (usize::from(header[1]) << 8) + usize::from(header[2]);
        let mut body = input.get(12..bytes_consumed).ok_or_else(truncated)?;
        let (qp, rp) = (header[3], header[4]);
        let mut flags = Bits::new(&header[5..]);
        for _ in SUBBANDS {
            // shortcut: only the observed significance+vertical profile, extend for other GCLI codings.
            if flags.read(2)? != 3 {
                return Err(unsupported("expected significance+vertical GCLI flags"));
            }
        }

        let mut current: [Prediction; 26] = std::array::from_fn(|_| Prediction::default());
        let mut packets: [Packet; 8] = std::array::from_fn(|_| Packet::default());
        for (packet_index, (packet, slots)) in packets.iter_mut().zip(PACKET_BANDS).enumerate() {
            packet.bands.reserve(slots.len());
            let mut parse_packet = || -> Result<(), DecodeError> {
                let mut packet_header = Bits::new(take(&mut body, 7)?);
                if packet_header.read(1)? != 0 {
                    return Err(unsupported("raw packet"));
                }
                let data_len = packet_header.read(20)?;
                let gcli_len = packet_header.read(20)?;
                let sign_len = packet_header.read(15)?;
                let sig_bits: usize = slots.iter().map(|&slot| GROUPS[slot].div_ceil(8)).sum();
                let mut sig = Bits::new(take(&mut body, sig_bits.div_ceil(8))?);
                let mut gcli = Bits::new(take(&mut body, gcli_len)?);
                let mut data = Bits::new(take(&mut body, data_len)?);
                let mut sign = Bits::new(take(&mut body, sign_len)?);
                for &slot in slots {
                    let weight_slot = if slot == 25 { 12 } else { slot };
                    let (gain, priority) = self.weights.0[weight_slot];
                    let gtli = qp.saturating_sub(gain).saturating_sub(u8::from(priority < rp)).min(15);
                    let predictor = if slot == 25 {
                        Some((&current[12], current[12].empty, gtli))
                    } else {
                        self.previous.as_ref().map(|prev| {
                            let prev = &prev[if slot == 12 { 25 } else { slot }];
                            if packet_index < 4 { (prev, false, prev.gtli) } else { (prev, prev.empty, gtli) }
                        })
                    };
                    let mut groups = Vec::with_capacity(GROUPS[slot]);
                    let mut prediction = Prediction {
                        gcli: Vec::with_capacity(GROUPS[slot]), gtli, empty: true,
                    };
                    for start in (0..GROUPS[slot]).step_by(8) {
                        let insignificant = sig.read1()? != 0;
                        prediction.empty &= insignificant;
                        for index in start..(start + 8).min(GROUPS[slot]) {
                            let unary = if insignificant { 0 } else { gcli.unary()? };
                            let planes = match predictor {
                                Some((prev, empty, prev_gtli)) if !(empty && prev_gtli <= gtli) => {
                                    predict(gtli, prev.gcli[index], unary)?
                                }
                                _ => gtli + unary,
                            };
                            let nbits = planes - gtli;
                            if nbits > 15 {
                                return Err(invalid("coefficient bit depth exceeds 15"));
                            }
                            prediction.gcli.push(planes);
                            let mut coeffs = [0i16; 4];
                            for _ in 0..nbits {
                                let plane = data.read4()?;
                                for (index, coeff) in coeffs.iter_mut().enumerate() {
                                    *coeff = (*coeff << 1) | i16::from((plane >> (3 - index)) & 1);
                                }
                            }
                            for coeff in &mut coeffs {
                                if *coeff != 0 && sign.read1()? != 0 {
                                    *coeff = -*coeff;
                                }
                            }
                            groups.push(Group { nbits, coeffs });
                        }
                    }
                    packet.bands.push(Band { subband: SUBBANDS[weight_slot], gtli, groups });
                    current[slot] = prediction;
                }
                for stream in [&sig, &gcli, &data, &sign] {
                    stream.finish()?;
                }
                Ok(())
            };
            parse_packet().map_err(|err| io::Error::new(err.kind(), format!("packet {packet_index}: {err}")))?;
        }
        self.previous = Some(current);
        Ok(Precinct { bytes_consumed, qp, rp, packets })
    }
}

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
    } else if unary % 2 == 0 {
        base + unary / 2
    } else {
        base - unary.div_ceil(2)
    })
}

fn take<'a>(input: &mut &'a [u8], len: usize) -> Result<&'a [u8], DecodeError> {
    let (head, tail) = input.split_at_checked(len).ok_or_else(truncated)?;
    *input = tail;
    Ok(head)
}

struct Bits<'a> {
    bytes: &'a [u8],
    pos: usize,
    next: usize,
    buffer: u64,
    buffered: u32,
}

impl<'a> Bits<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0, next: 0, buffer: 0, buffered: 0 }
    }

    #[inline(always)]
    fn refill(&mut self) {
        while self.buffered <= 56 {
            let Some(&byte) = self.bytes.get(self.next) else {
                break;
            };
            self.buffer |= u64::from(byte) << (56 - self.buffered);
            self.next += 1;
            self.buffered += 8;
        }
    }

    #[inline(always)]
    fn read(&mut self, count: usize) -> Result<usize, DecodeError> {
        if count == 0 {
            return Ok(0);
        }
        if self.buffered < count as u32 {
            self.refill();
            if self.buffered < count as u32 {
                return Err(truncated());
            }
        }
        let value = (self.buffer >> (64 - count)) & ((1u64 << count) - 1);
        self.buffer <<= count;
        self.buffered -= count as u32;
        self.pos += count;
        Ok(value as usize)
    }

    #[inline(always)]
    fn read1(&mut self) -> Result<u8, DecodeError> {
        if self.buffered == 0 {
            self.refill();
            if self.buffered == 0 {
                return Err(truncated());
            }
        }
        let value = (self.buffer >> 63) as u8;
        self.buffer <<= 1;
        self.buffered -= 1;
        self.pos += 1;
        Ok(value)
    }

    #[inline(always)]
    fn read4(&mut self) -> Result<u8, DecodeError> {
        if self.buffered < 4 {
            self.refill();
            if self.buffered < 4 {
                return Err(truncated());
            }
        }
        let value = (self.buffer >> 60) as u8;
        self.buffer <<= 4;
        self.buffered -= 4;
        self.pos += 4;
        Ok(value)
    }

    #[inline(always)]
    fn unary(&mut self) -> Result<u8, DecodeError> {
        for ones in 0..=31 {
            if self.read1()? == 0 {
                return Ok(ones);
            }
        }
        Err(invalid("unterminated or oversized unary GCLI code"))
    }

    fn finish(&self) -> Result<(), DecodeError> {
        if self.pos.div_ceil(8) != self.bytes.len() {
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
    io::Error::new(io::ErrorKind::UnexpectedEof, "truncated precinct or packet substream")
}
