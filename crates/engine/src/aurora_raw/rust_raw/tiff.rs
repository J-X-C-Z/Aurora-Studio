//! Minimal TIFF/EP reader shared by the native RAW front-ends.
//!
//! Only what a RAW container needs: the header, a bounded IFD walk, typed value
//! access and the byte range behind an entry.  Everything returns errors rather
//! than panicking, because every offset here comes straight from the file.

use super::{NativeError, NativeResult};

pub(crate) const TYPE_BYTE: u16 = 1;
pub(crate) const TYPE_ASCII: u16 = 2;
pub(crate) const TYPE_SHORT: u16 = 3;
pub(crate) const TYPE_LONG: u16 = 4;
pub(crate) const TYPE_RATIONAL: u16 = 5;
pub(crate) const TYPE_SBYTE: u16 = 6;
pub(crate) const TYPE_UNDEFINED: u16 = 7;
pub(crate) const TYPE_SSHORT: u16 = 8;
pub(crate) const TYPE_SLONG: u16 = 9;
pub(crate) const TYPE_SRATIONAL: u16 = 10;
pub(crate) const TYPE_FLOAT: u16 = 11;
pub(crate) const TYPE_DOUBLE: u16 = 12;
pub(crate) const TYPE_IFD: u16 = 13;

/// Upper bound on entries in one IFD; no camera writes more than a few hundred.
const MAX_ENTRIES: usize = 4096;
/// Upper bound on following IFDs in one chain, guards against loops.
const MAX_CHAIN: usize = 64;

fn type_size(typ: u16) -> Option<usize> {
    Some(match typ {
        TYPE_BYTE | TYPE_ASCII | TYPE_SBYTE | TYPE_UNDEFINED => 1,
        TYPE_SHORT | TYPE_SSHORT => 2,
        TYPE_LONG | TYPE_SLONG | TYPE_FLOAT | TYPE_IFD => 4,
        TYPE_RATIONAL | TYPE_SRATIONAL | TYPE_DOUBLE => 8,
        _ => return None,
    })
}

/// A TIFF header and the bytes it addresses.
///
/// `base` is the absolute offset of the header inside `data`; IFD offsets are
/// relative to it, which is also how an embedded MakerNote TIFF works.
#[derive(Clone, Copy)]
pub(crate) struct Tiff<'a> {
    data: &'a [u8],
    base: usize,
    pub little: bool,
}

impl<'a> Tiff<'a> {
    /// Parses the TIFF header at `base` (the whole file for RAW containers).
    pub fn parse(data: &'a [u8], base: usize) -> NativeResult<Self> {
        Self::parse_with_magic(data, base, &[42])
    }

    /// Parses a TIFF-like header whose 16-bit version/magic word matches one of
    /// `allowed_magics` (e.g. `0x0055` for Panasonic RW2 or `RO`/`RS` for ORF).
    pub fn parse_with_magic(data: &'a [u8], base: usize, allowed_magics: &[u16]) -> NativeResult<Self> {
        let header = data
            .get(base..base + 8)
            .ok_or_else(|| NativeError::corrupt("truncated TIFF header"))?;
        let little = match &header[..2] {
            b"II" => true,
            b"MM" => false,
            _ => return Err(NativeError::corrupt("not a TIFF file")),
        };
        let magic = if little {
            u16::from_le_bytes([header[2], header[3]])
        } else {
            u16::from_be_bytes([header[2], header[3]])
        };
        if !allowed_magics.contains(&magic) {
            return Err(NativeError::corrupt(format!("TIFF magic {magic:#06x} is not expected")));
        }
        Ok(Self { data, base, little })
    }

    /// The whole byte slice this header addresses.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    pub fn absolute(&self, offset: usize) -> NativeResult<usize> {
        self.base
            .checked_add(offset)
            .filter(|&at| at <= self.data.len())
            .ok_or_else(|| NativeError::corrupt("TIFF offset outside the file"))
    }

    pub fn u16_at(&self, at: usize) -> NativeResult<u16> {
        let bytes = self
            .data
            .get(at..at + 2)
            .ok_or_else(|| NativeError::corrupt("truncated TIFF u16"))?;
        Ok(if self.little {
            u16::from_le_bytes([bytes[0], bytes[1]])
        } else {
            u16::from_be_bytes([bytes[0], bytes[1]])
        })
    }

    pub fn u32_at(&self, at: usize) -> NativeResult<u32> {
        let bytes = self
            .data
            .get(at..at + 4)
            .ok_or_else(|| NativeError::corrupt("truncated TIFF u32"))?;
        Ok(if self.little {
            u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        } else {
            u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
        })
    }

    /// Parses the IFD stored at `offset` (relative to the header).
    pub fn ifd(&self, offset: u32) -> NativeResult<Ifd<'a>> {
        let at = self.absolute(offset as usize)?;
        let count = usize::from(self.u16_at(at)?);
        if count > MAX_ENTRIES {
            return Err(NativeError::corrupt("TIFF IFD has too many entries"));
        }
        let start = at + 2;
        let end = start
            .checked_add(count * 12)
            .filter(|&end| end + 4 <= self.data.len())
            .ok_or_else(|| NativeError::corrupt("truncated TIFF IFD"))?;
        let mut entries = Vec::with_capacity(count);
        for index in 0..count {
            let entry_at = start + index * 12;
            let tag = self.u16_at(entry_at)?;
            let typ = self.u16_at(entry_at + 2)?;
            let value_count = self.u32_at(entry_at + 4)?;
            let unit = type_size(typ)
                .ok_or_else(|| NativeError::corrupt(format!("TIFF tag {tag:#06x} has type {typ}")))?;
            let bytes = (value_count as usize)
                .checked_mul(unit)
                .ok_or_else(|| NativeError::corrupt("TIFF value overflows"))?;
            let value_offset = if bytes <= 4 {
                entry_at + 8
            } else {
                self.absolute(self.u32_at(entry_at + 8)? as usize)?
            };
            entries.push(Entry {
                tag,
                typ,
                count: value_count,
                value_offset,
                value_len: bytes,
            });
        }
        let next = self.u32_at(end)?;
        Ok(Ifd { tiff: *self, offset, entries, next: (next != 0).then_some(next) })
    }

    /// IFD0 and every IFD reachable through its `next` chain.
    pub fn chain(&self, first: u32) -> NativeResult<Vec<Ifd<'a>>> {
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut next = Some(first);
        while let Some(offset) = next {
            if !seen.insert(offset) || out.len() >= MAX_CHAIN {
                break;
            }
            let ifd = self.ifd(offset)?;
            next = ifd.next;
            out.push(ifd);
        }
        Ok(out)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Entry {
    pub tag: u16,
    pub typ: u16,
    pub count: u32,
    /// Absolute offset of the value bytes (the entry itself when inline).
    pub value_offset: usize,
    pub value_len: usize,
}

#[derive(Clone)]
pub(crate) struct Ifd<'a> {
    pub tiff: Tiff<'a>,
    pub offset: u32,
    pub entries: Vec<Entry>,
    pub next: Option<u32>,
}

impl<'a> Ifd<'a> {
    pub fn entry(&self, tag: u16) -> Option<&Entry> {
        self.entries.iter().find(|entry| entry.tag == tag)
    }

    /// The raw value bytes, with bounds checked against the file.
    pub fn bytes(&self, entry: &Entry) -> NativeResult<&'a [u8]> {
        self.tiff
            .data
            .get(entry.value_offset..entry.value_offset + entry.value_len)
            .ok_or_else(|| NativeError::corrupt(format!("TIFF tag {:#06x} value is truncated", entry.tag)))
    }

    pub fn tag_bytes(&self, tag: u16) -> Option<NativeResult<&'a [u8]>> {
        self.entry(tag).map(|entry| self.bytes(entry))
    }

    fn units(&self, tag: u16) -> Option<NativeResult<Vec<u64>>> {
        let entry = self.entry(tag)?;
        Some((|| {
            let raw = self.bytes(entry)?;
            let mut out = Vec::with_capacity(entry.count as usize);
            match entry.typ {
                TYPE_BYTE | TYPE_UNDEFINED | TYPE_ASCII => {
                    out.extend(raw.iter().map(|&value| u64::from(value)))
                }
                TYPE_SBYTE => out.extend(raw.iter().map(|&value| (value as i8 as i64) as u64)),
                TYPE_SHORT => {
                    for chunk in raw.chunks_exact(2) {
                        out.push(u64::from(if self.tiff.little {
                            u16::from_le_bytes([chunk[0], chunk[1]])
                        } else {
                            u16::from_be_bytes([chunk[0], chunk[1]])
                        }));
                    }
                }
                TYPE_SSHORT => {
                    for chunk in raw.chunks_exact(2) {
                        let value = if self.tiff.little {
                            i16::from_le_bytes([chunk[0], chunk[1]])
                        } else {
                            i16::from_be_bytes([chunk[0], chunk[1]])
                        };
                        out.push(value as i64 as u64);
                    }
                }
                TYPE_LONG | TYPE_IFD => {
                    for chunk in raw.chunks_exact(4) {
                        out.push(u64::from(if self.tiff.little {
                            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                        } else {
                            u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                        }));
                    }
                }
                TYPE_SLONG => {
                    for chunk in raw.chunks_exact(4) {
                        let value = if self.tiff.little {
                            i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                        } else {
                            i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                        };
                        out.push(value as i64 as u64);
                    }
                }
                _ => {
                    return Err(NativeError::corrupt(format!(
                        "TIFF tag {} has non-integer type {}",
                        tag, entry.typ
                    )));
                }
            }
            Ok(out)
        })())
    }

    pub fn u32s(&self, tag: u16) -> Option<NativeResult<Vec<u32>>> {
        self.units(tag)
            .map(|values| values.map(|values| values.into_iter().map(|value| value as u32).collect()))
    }

    pub fn u32(&self, tag: u16) -> Option<u32> {
        self.u32s(tag)?.ok()?.first().copied()
    }

    pub fn u16s(&self, tag: u16) -> Option<NativeResult<Vec<u16>>> {
        self.units(tag)
            .map(|values| values.map(|values| values.into_iter().map(|value| value as u16).collect()))
    }

    pub fn u16(&self, tag: u16) -> Option<u16> {
        self.u16s(tag)?.ok()?.first().copied()
    }

    /// RATIONAL/SRATIONAL/FLOAT/DOUBLE values as `f64`.
    pub fn reals(&self, tag: u16) -> Option<NativeResult<Vec<f64>>> {
        let entry = self.entry(tag)?;
        Some((|| {
            let raw = self.bytes(entry)?;
            let little = self.tiff.little;
            let mut out = Vec::with_capacity(entry.count as usize);
            match entry.typ {
                TYPE_RATIONAL | TYPE_SRATIONAL => {
                    for chunk in raw.chunks_exact(8) {
                        let (num, den) = if little {
                            (
                                i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64,
                                i32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]) as i64,
                            )
                        } else {
                            (
                                i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64,
                                i32::from_be_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]) as i64,
                            )
                        };
                        out.push(if den == 0 { 0.0 } else { num as f64 / den as f64 });
                    }
                }
                TYPE_FLOAT => {
                    for chunk in raw.chunks_exact(4) {
                        let bits = if little {
                            u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                        } else {
                            u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                        };
                        out.push(f64::from(f32::from_bits(bits)));
                    }
                }
                TYPE_DOUBLE => {
                    for chunk in raw.chunks_exact(8) {
                        let bits = if little {
                            u64::from_le_bytes(chunk.try_into().unwrap())
                        } else {
                            u64::from_be_bytes(chunk.try_into().unwrap())
                        };
                        out.push(f64::from_bits(bits));
                    }
                }
                TYPE_SHORT | TYPE_LONG | TYPE_BYTE | TYPE_SLONG | TYPE_SSHORT | TYPE_SBYTE => {
                    if let Some(Ok(values)) = self.units(tag) {
                        out.extend(values.into_iter().map(|value| value as i64 as f64));
                    }
                }
                typ => {
                    return Err(NativeError::corrupt(format!(
                        "TIFF tag {} has non-numeric type {typ}",
                        tag
                    )));
                }
            }
            Ok(out)
        })())
    }

    pub fn real(&self, tag: u16) -> Option<f64> {
        self.reals(tag)?.ok()?.first().copied()
    }

    pub fn ascii(&self, tag: u16) -> Option<String> {
        let bytes = self.tag_bytes(tag)?.ok()?;
        let text = bytes.split(|&byte| byte == 0).next().unwrap_or_default();
        Some(String::from_utf8_lossy(text).trim().to_string())
    }

    pub fn sub_ifds(&self) -> Vec<u32> {
        let Some(Ok(values)) = self.u32s(0x014a) else {
            return Vec::new();
        };
        values.into_iter().filter(|&offset| offset != 0).collect()
    }

    pub fn exif_ifd(&self) -> Option<u32> {
        self.u32(0x8769).filter(|&offset| offset != 0)
    }
}
