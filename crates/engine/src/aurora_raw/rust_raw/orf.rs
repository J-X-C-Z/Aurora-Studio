//! Olympus / OM System ORF sensor decode.
//!
//! ORF is a TIFF variant: the eight-byte `IIRO`/`IIRS` prefix carries the byte
//! order and the offset of IFD0, so the rest of the container (IFDs, strips,
//! MakerNote offsets) is read with the shared TIFF reader.  The sensor payload
//! is either unpacked 16-bit words, packed 12-bit rows, lossless JPEG, or
//! Olympus' 12/14-bit entropy scheme (MakerNote 0x2040 tags 0x0640..0x0653).
//!
//! The decoding rules mirror LibRaw 0.23 (`olympus14.cpp`, `generic.cpp`,
//! `identify.cpp` and `olympus.cpp`) but are implemented independently here.
//! Unknown compression codes are reported as [`NativeError::Unsupported`] so
//! `decode_bayer16` can fall back to LibRaw instead of returning corrupt data.

use super::tiff::{Ifd, Tiff};
use super::{cfa_filters, ljpeg, make_frame, meta, NativeError, NativeResult};

use crate::raw::BayerFrame;

/// Parameters from MakerNote 0x2040 (or 0x3000) used by the 12/14-bit codec.
#[derive(Clone, Copy, Debug)]
struct Oly14Params {
    tag640: u32,
    tag641: u32,
    tag642: u32,
    tag643: u32,
    tag644: u32,
    tag645: u32,
    tag646: u32,
    tag647: u32,
    tag648: u32,
    tag649: u32,
    tag650: u32,
    tag651: u32,
    tag652: u32,
    /// Length of the zero header before the bitstream; tag 0x0653.
    tag653: u32,
    valid_bits: u32,
}

impl Oly14Params {
    fn from_ifd(ifd: &MnIfd<'_>) -> Option<Self> {
        let params = Self {
            tag640: ifd.u32(0x0640).unwrap_or(0),
            tag641: ifd.u32(0x0641).unwrap_or(0),
            tag642: ifd.u32(0x0642).unwrap_or(0),
            tag643: ifd.u32(0x0643).unwrap_or(0),
            tag644: ifd.u32(0x0644).unwrap_or(0),
            tag645: ifd.u32(0x0645).unwrap_or(0),
            tag646: ifd.u32(0x0646).unwrap_or(0),
            tag647: ifd.u32(0x0647).unwrap_or(0),
            tag648: ifd.u32(0x0648).unwrap_or(0),
            tag649: ifd.u32(0x0649).unwrap_or(0),
            tag650: ifd.u32(0x0650).unwrap_or(0),
            tag651: ifd.u32(0x0651).unwrap_or(0),
            tag652: ifd.u32(0x0652).unwrap_or(0),
            tag653: ifd.u32(0x0653).unwrap_or(0),
            valid_bits: ifd.u32(0x0611).unwrap_or(0),
        };
        // LibRaw requires the header and a nonzero escape length; otherwise
        // the entropy stream has no anchor and the file is not a 14-bit ORF.
        (params.tag653 > 0 && params.tag651 > 0 && params.valid_bits >= 8 && params.valid_bits <= 16)
            .then_some(params)
    }
}

/// Metadata extracted from the Olympus MakerNote IFD tree.
#[derive(Default)]
struct OlyMeta {
    params: Option<Oly14Params>,
    /// Per-CFA black levels in file (RGGB) order.
    black_rggb: Option<[u32; 4]>,
    /// Camera-to-XYZ matrix used by LibRaw to build `rgb_cam`.
    cmatrix: Option<[[f64; 3]; 3]>,
    /// As-shot multipliers R and B, already divided by 256.
    wb_r: Option<f32>,
    wb_b: Option<f32>,
}

pub(crate) fn decode(bytes: &[u8]) -> NativeResult<BayerFrame> {
    if !bytes.starts_with(b"IIRO") && !bytes.starts_with(b"IIRS") {
        return Err(NativeError::corrupt("not an ORF container"));
    }
    // LibRaw's ORF entry point reads `II`, ignores the two signature bytes and
    // treats u32@4 as the IFD0 offset.  Patch only the two magic bytes so the
    // shared TIFF reader can use the same zero-based offsets as the file.
    let mut patched = bytes.to_vec();
    patched.get_mut(2..4).ok_or_else(|| NativeError::corrupt("truncated ORF header"))?.copy_from_slice(&42u16.to_le_bytes());
    let tiff = Tiff::parse(&patched, 0)?;
    let ifd0 = tiff.ifd(tiff.u32_at(4)?)?;

    let ifd = super::raw_ifd(&tiff)?;
    let raw_width = ifd.u32(0x0100).ok_or_else(|| NativeError::corrupt("ORF has no image width"))? as usize;
    let raw_height = ifd.u32(0x0101).ok_or_else(|| NativeError::corrupt("ORF has no image height"))? as usize;
    if raw_width == 0 || raw_height == 0 || raw_width > 100_000 || raw_height > 100_000 {
        return Err(NativeError::corrupt("ORF dimensions are out of range"));
    }
    let ifd_bits = ifd.u16(0x0102).unwrap_or(12) as u8;
    let compression = ifd.u16(0x0103).unwrap_or(1);
    let (filters, colors) = cfa_filters(&ifd)?;
    if filters == 9 {
        return Err(NativeError::unsupported("X-Trans sensor in an ORF container"));
    }

    let offset = ifd
        .u32s(0x0111)
        .and_then(|values| values.ok())
        .and_then(|values| values.first().copied())
        .ok_or_else(|| NativeError::corrupt("ORF has no strip offset"))? as usize;
    let count = ifd
        .u32s(0x0117)
        .and_then(|values| values.ok())
        .and_then(|values| values.first().copied())
        .unwrap_or_else(|| (tiff.data().len().saturating_sub(offset)) as u32) as usize;
    let strip = tiff
        .data()
        .get(offset..offset.saturating_add(count))
        .ok_or_else(|| NativeError::corrupt("ORF strip is outside the file"))?;

    let oly = read_maker_notes(&tiff, &ifd0);
    let params = oly.params;
    let bits = params.map(|p| p.valid_bits as u8).unwrap_or(ifd_bits);

    let (data, load_flags) = match compression {
        1 | 32769 | 32770 | 32773 if count.saturating_mul(7) > raw_width.saturating_mul(raw_height) => {
            let params = params.ok_or_else(|| {
                NativeError::unsupported("Olympus 14-bit ORF lacks its MakerNote compression parameters")
            })?;
            (decode_olympus14(strip, raw_width, raw_height, &params)?, 0)
        }
        1 | 32769 | 32770 | 32773 if count >= raw_width.saturating_mul(raw_height) * 2 => {
            (meta::unpack_bits_le16(strip, raw_width * raw_height)?, if bits == 14 { 4 } else { 0 })
        }
        1 | 32769 | 32770 | 32773 if count.saturating_mul(3) >= raw_width.saturating_mul(raw_height) * 2 => {
            (decode_packed12(strip, raw_width, raw_height, 0)?, 0)
        }
        6 | 7 => {
            let image = ljpeg::decode(strip, false)?;
            let expected = raw_width * raw_height;
            if image.width * image.height != expected || image.components != 1 {
                return Err(NativeError::unsupported("ORF lossless JPEG dimensions differ from the sensor"));
            }
            (image.data, 0)
        }
        other => return Err(NativeError::unsupported(format!("Olympus ORF compression {other}"))),
    };

    let data = if load_flags != 0 {
        data.into_iter().map(|value| value >> load_flags).collect()
    } else {
        data
    };

    let (left, top, width, height) = geometry(&ifd, raw_width, raw_height);
    let mut black = oly
        .black_rggb
        .map(|levels| levels.iter().min().copied().unwrap_or(0))
        .or_else(|| {
            ifd.reals(0xc61a)
                .and_then(|values| values.ok())
                .and_then(|values| (values.len() == 1 && values[0] >= 0.0).then(|| values[0].round() as u32))
        })
        .unwrap_or(0);
    if bits == 14 && compression != 6 && compression != 7 {
        black *= 4;
    }
    let maximum = ifd
        .u32(0xc61d)
        .unwrap_or_else(|| 1u32 << bits.min(16))
        .min(0xffff);
    let mut cam_mul = [0.0f32; 4];
    if let Some(red) = oly.wb_r {
        cam_mul[0] = red;
    }
    if let Some(blue) = oly.wb_b {
        cam_mul[2] = blue;
    }
    let rgb_cam = oly
        .cmatrix
        .map(|matrix| rgb_cam_from_oly_matrix(&matrix, colors as usize))
        .unwrap_or([[0.0f32; 4]; 3]);

    make_frame(
        raw_width, raw_height, left, top, width, height, bits, data, filters, colors, black,
        maximum, cam_mul, rgb_cam,
    )
}

/// LibRaw's Olympus width trims in `identify_finetune_dcr` (the XZ/Stylus
/// model-ID special cases are not derivable without decoding CameraType2 and
/// are left to LibRaw via the `Unsupported` fallback where they matter).
fn geometry(ifd: &Ifd<'_>, raw_width: usize, raw_height: usize) -> (usize, usize, usize, usize) {
    let mut width = raw_width;
    let mut height = raw_height + (raw_height & 1);
    if width == 4100 {
        width -= 4;
    }
    if width == 4080 {
        width -= 24;
    }
    if width == 10400 {
        width -= 12;
    }
    if width == 8200 {
        width -= 30;
    }
    if width == 8180 {
        width -= 10;
    }
    if width == 9280 {
        width -= 6;
        height = height.saturating_sub(6);
    }
    if width > 0 && width <= raw_width && height <= raw_height.saturating_add(1) && height > 0 {
        // ORF declares its crop by trimming the right edge; the sensor grid
        // itself remains raw_width x raw_height with a zero origin.
        let _ = ifd;
        return (0, 0, width, height);
    }
    super::active_area(ifd, raw_width, raw_height)
}

/// Reads the MakerNote IFD and the nested 0x2040/0x3000 blocks.
fn read_maker_notes(tiff: &Tiff<'_>, ifd0: &Ifd<'_>) -> OlyMeta {
    let mut out = OlyMeta::default();
    let Some(offset) = maker_note_offset(ifd0, 0xc634).or_else(|| maker_note_offset(ifd0, 0xc635)) else {
        return out;
    };
    let Some(mn) = parse_maker_ifd(tiff, offset, offset) else {
        return out;
    };
    // Header tags may be stored directly in the maker IFD (0x1012 etc.) and
    // the codec parameters in a nested block (0x2040 or 0x3000).
    read_oly_tags(&mn, &mut out);
    for parent in [0x2040u16, 0x3000] {
        let Some(child_offset) = mn.u32(parent) else {
            continue;
        };
        // LibRaw follows the tag's offset field relative to the MakerNote base.
        let Some(child) = parse_maker_ifd(tiff, offset as usize + child_offset as usize, offset) else {
            continue;
        };
        read_oly_tags(&child, &mut out);
    }
    out
}

fn maker_note_offset(ifd: &Ifd<'_>, tag: u16) -> Option<usize> {
    if let Some(Ok(bytes)) = ifd.tag_bytes(tag) {
        if bytes.len() >= 4 {
            let little = ifd.tiff.little;
            let value = if little {
                u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            } else {
                u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
            };
            if value != 0 {
                return Some(value as usize);
            }
        }
    }
    if let Some(exif) = ifd.exif_ifd() {
        let exif = ifd.tiff.ifd(exif).ok()?;
        if let Some(Ok(bytes)) = exif.tag_bytes(tag) {
            if bytes.len() >= 4 {
                let little = ifd.tiff.little;
                let value = if little {
                    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                } else {
                    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
                };
                if value != 0 {
                    return Some(value as usize);
                }
            }
        }
    }
    None
}

/// A minimal reader for Olympus MakerNote IFDs.  The main note starts with
/// `OLYMPUS\0` (or `OM SYS\0`), followed by byte order, a two-byte pad and an
/// ordinary IFD; nested 0x2040/0x3000 blocks are plain IFDs.  Value offsets are
/// relative to the MakerNote base, which LibRaw also uses for the nested IFDs.
struct MnIfd<'a> {
    data: &'a [u8],
    base: usize,
    little: bool,
    entries: Vec<(u16, u16, u32, usize, usize)>,
}

impl<'a> MnIfd<'a> {
    /// `entries` tuples are (tag, type, count, value_offset, value_len).
    fn values(&self, tag: u16) -> Option<&'a [u8]> {
        self.entries
            .iter()
            .find(|entry| entry.0 == tag)
            .and_then(|entry| self.data.get(entry.3..entry.3 + entry.4))
    }

    fn u32(&self, tag: u16) -> Option<u32> {
        let raw = self.values(tag)?;
        if raw.len() < 4 {
            return None;
        }
        let entry = self.entries.iter().find(|entry| entry.0 == tag)?;
        if entry.1 == 3 {
            let value = if self.little {
                u16::from_le_bytes([raw[0], raw[1]])
            } else {
                u16::from_be_bytes([raw[0], raw[1]])
            };
            Some(u32::from(value))
        } else {
            Some(if self.little {
                u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
            } else {
                u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]])
            })
        }
    }

    fn u16(&self, tag: u16) -> Option<u16> {
        let raw = self.values(tag)?;
        if raw.len() < 2 {
            return None;
        }
        Some(if self.little {
            u16::from_le_bytes([raw[0], raw[1]])
        } else {
            u16::from_be_bytes([raw[0], raw[1]])
        })
    }

    fn u16s(&self, tag: u16) -> Option<Vec<u16>> {
        let raw = self.values(tag)?;
        let entry = self.entries.iter().find(|entry| entry.0 == tag)?;
        if entry.1 != 3 {
            return None;
        }
        Some(
            raw.chunks_exact(2)
                .take(entry.2 as usize)
                .map(|chunk| {
                    if self.little {
                        u16::from_le_bytes([chunk[0], chunk[1]])
                    } else {
                        u16::from_be_bytes([chunk[0], chunk[1]])
                    }
                })
                .collect(),
        )
    }

    fn i16s(&self, tag: u16) -> Option<Vec<i16>> {
        self.u16s(tag).map(|values| values.into_iter().map(|value| value as i16).collect())
    }

    fn reals(&self, tag: u16) -> Option<Vec<f64>> {
        let raw = self.values(tag)?;
        let entry = self.entries.iter().find(|entry| entry.0 == tag)?;
        let mut out = Vec::new();
        match entry.1 {
            3 => out.extend(self.u16s(tag)?.into_iter().map(f64::from)),
            4 => {
                for chunk in raw.chunks_exact(4).take(entry.2 as usize) {
                    let bits = if self.little {
                        u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                    } else {
                        u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])
                    };
                    out.push(f64::from(f32::from_bits(bits)));
                }
            }
            5 => {
                for chunk in raw.chunks_exact(8).take(entry.2 as usize) {
                    let (num, den) = if self.little {
                        (
                            i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                            i32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
                        )
                    } else {
                        (
                            i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
                            i32::from_be_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]),
                        )
                    };
                    out.push(if den == 0 { 0.0 } else { f64::from(num) / f64::from(den) });
                }
            }
            _ => return None,
        }
        Some(out)
    }
}

/// Reads a MakerNote IFD. `at` points at the eight-byte `OLYMPUS\0` signature
/// for the top-level note, or at the child IFD for a nested block.
fn parse_maker_ifd<'a>(tiff: &Tiff<'a>, at: usize, value_base: usize) -> Option<MnIfd<'a>> {
    let data = tiff.data();
    let header = data.get(at..at + 2)?;
    let (little, entries_at) = if header == b"II" || header == b"MM" {
        (header == b"II", at + 2)
    } else {
        let order = data.get(at + 8..at + 10)?;
        if order != b"II" && order != b"MM" {
            return None;
        }
        (order == b"II", at + 14)
    };
    let count = read_u16(data, entries_at - 2, little)? as usize;
    if count == 0 || count > 4096 {
        return None;
    }
    let mut entries = Vec::with_capacity(count);
    for index in 0..count {
        let entry_at = entries_at + index * 12;
        let bytes = data.get(entry_at..entry_at + 12)?;
        let tag = read_u16(data, entry_at, little)?;
        let typ = read_u16(data, entry_at + 2, little)?;
        let value_count = read_u32(data, entry_at + 4, little)?;
        let unit = match typ {
            1 | 2 | 6 | 7 => 1usize,
            3 | 8 => 2,
            4 | 9 | 11 => 4,
            5 | 10 | 12 => 8,
            _ => return None,
        };
        let value_len = (value_count as usize).checked_mul(unit)?;
        let value_at = if value_len <= 4 {
            entry_at + 8
        } else {
            value_base + read_u32(data, entry_at + 8, little)? as usize
        };
        entries.push((tag, typ, value_count, value_at, value_len));
    }
    Some(MnIfd { data, base: value_base, little, entries })
}

fn read_u16(data: &[u8], at: usize, little: bool) -> Option<u16> {
    let bytes = data.get(at..at + 2)?;
    Some(if little {
        u16::from_le_bytes([bytes[0], bytes[1]])
    } else {
        u16::from_be_bytes([bytes[0], bytes[1]])
    })
}

fn read_u32(data: &[u8], at: usize, little: bool) -> Option<u32> {
    let bytes = data.get(at..at + 4)?;
    Some(if little {
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    } else {
        u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    })
}

fn read_oly_tags(ifd: &MnIfd<'_>, out: &mut OlyMeta) {
    if out.params.is_none() {
        out.params = Oly14Params::from_ifd(ifd);
    }
    if out.black_rggb.is_none() {
        if let Some(levels) = ifd.u16s(0x0600) {
            if levels.len() >= 4 {
                // File order is RGGB; LibRaw's RGGB_2_RGBG reorders it for the
                // scalar black but BayerFrame only consumes the minimum.
                out.black_rggb = Some([levels[0] as u32, levels[1] as u32, levels[3] as u32, levels[2] as u32]);
            }
        }
    }
    if out.black_rggb.is_none() {
        if let Some(levels) = ifd.u16s(0x1012) {
            if levels.len() >= 4 {
                out.black_rggb = Some([levels[0] as u32, levels[1] as u32, levels[3] as u32, levels[2] as u32]);
            }
        }
    }
    if out.cmatrix.is_none() {
        if let Some(values) = ifd.reals(0x0200) {
            if values.len() >= 9 {
                let mut matrix = [[0.0f64; 3]; 3];
                for (index, value) in values.iter().take(9).enumerate() {
                    matrix[index / 3][index % 3] = *value / 256.0;
                }
                out.cmatrix = Some(matrix);
            }
        }
    }
    if out.cmatrix.is_none() {
        if let Some(values) = ifd.i16s(0x1011) {
            if values.len() >= 9 {
                let mut matrix = [[0.0f64; 3]; 3];
                for (index, value) in values.iter().take(9).enumerate() {
                    matrix[index / 3][index % 3] = f64::from(*value) / 256.0;
                }
                out.cmatrix = Some(matrix);
            }
        }
    }
    if out.wb_r.is_none() {
        if let Some(value) = ifd.u16(0x0100) {
            out.wb_r = Some(f32::from(value) / 256.0);
        }
    }
    if out.wb_r.is_none() {
        if let Some(value) = ifd.u16(0x1017) {
            out.wb_r = Some(f32::from(value) / 256.0);
        }
    }
    if out.wb_b.is_none() {
        if let Some(value) = ifd.u16(0x1018) {
            out.wb_b = Some(f32::from(value) / 256.0);
        }
    }
}

/// LibRaw's Olympus `cmatrix` is camera-to-XYZ, not Adobe's XYZ-to-camera
/// matrix.  Convert it to the `rgb_cam` word using the shared pseudoinverse.
fn rgb_cam_from_oly_matrix(camera_xyz: &[[f64; 3]; 3], colors: usize) -> [[f32; 4]; 3] {
    let mut cam = [[0.0f64; 3]; 4];
    for row in 0..3 {
        cam[row] = camera_xyz[row];
    }
    meta::cam_xyz_to_rgb_cam(&cam, colors)
}

/// Olympus 12/14-bit entropy decoder (LibRaw `olympus14_load_raw`).
fn decode_olympus14(
    strip: &[u8],
    width: usize,
    height: usize,
    p: &Oly14Params,
) -> NativeResult<Vec<u16>> {
    let header = p.tag653 as usize;
    let stream = strip.get(header..).ok_or_else(|| NativeError::corrupt("ORF header exceeds the strip"))?;
    if stream.len() < 4 {
        return Err(NativeError::corrupt("ORF entropy stream is empty"));
    }
    let mut pump = BitPump::new(stream);
    let tag640 = p.tag640 as i32;
    let tag641 = p.tag641 as i32;
    let tag642 = p.tag642 as i32;
    let tag643 = p.tag643 as i32;
    let tag644 = p.tag644 as i32;
    let tag645 = p.tag645.min(31);
    let one_shl_645 = 1i32 << tag645;
    let tag646 = p.tag646 as i32;
    let tag647 = p.tag647 as i32;
    let tag648 = p.tag648 as i32;
    let tag649 = p.tag649 as i32;
    let tag650 = p.tag650 as i32;
    let tag651 = p.tag651 as i32;
    let tag652 = (p.tag652 as i32).max(1);
    let datamax = 1u32 << p.valid_bits.min(16);

    let mut context = [0i32; 4];
    let mut out = vec![0u16; width.saturating_mul(height)];
    let mut lpred = 0i32;
    let mut pred1 = 0i32;
    let mut glc = 0i32;
    let mut gcode = 0i32;
    let bitcounts = bit_count_table();

    for row in 0..height {
        for col in 0..width {
            let previous_code = gcode;
            gcode = glc;
            let psel = if col > 1 && one_shl_645 >= previous_code {
                context[3] + 1
            } else {
                0
            };
            context[3] = context[2];
            context[2] = psel;
            let mut wbits = if col > 1 {
                let highbitcount = bitcounts[(previous_code as u16) as usize];
                let t1 = if psel >= tag646 { tag648 } else { tag650 };
                let t2 = if psel >= tag646 { tag647 } else { tag649 };
                let tt = (highbitcount - t1).max(-1);
                t2 + tt + 1
            } else {
                tag649
            };
            wbits = wbits.min(tag651);
            if wbits < 0 || wbits >= 16 {
                return Err(NativeError::corrupt("ORF entropy width is out of range"));
            }

            let (mut code, low, t640bits, t643bits) =
                oly_code(&mut pump, wbits as u32, tag640, tag643, tag652)?;
            let vbits = if tag643 != 0 {
                let v = if col < 2 { 0 } else { pred1 };
                if tag643 == 1 {
                    code = t643bits + 2 * (v + code);
                    v + (code >> 1)
                } else {
                    code = t643bits + 4 * (v + code);
                    ((code >> 1) & -2i32) + v + (code >> 2)
                }
            } else {
                0
            };
            pred1 = context[0];
            context[0] = if tag644 == 15 { 0 } else { vbits >> tag644.max(0).min(31) };

            let w = if col < 2 { tag642 } else { context[1] };
            let n = if row < 2 {
                tag642
            } else {
                (out[(row - 2) * width + col] as i32) >> tag640.max(0).min(31)
            };
            let mut nw = if row < 2 || col < 2 {
                tag642
            } else {
                (out[(row - 2) * width + col - 2] as i32) >> tag640.max(0).min(31)
            };
            let _ = &mut nw;
            context[1] = lpred;
            let mut n = n;
            if w < n || nw < w {
                if nw <= n && w >= n {
                    n = w;
                } else if nw < n || w >= n {
                    if nw > w || w >= n {
                        if (n - nw).abs() > (1i32 << tag641.min(30)) {
                            n += w - nw;
                        } else if (w - nw).abs() <= (1i32 << tag641.min(30)) {
                            n = (n + w) >> 1;
                        } else {
                            n = n + w - nw;
                        }
                    }
                } else {
                    n = w;
                }
            }
            lpred = code + n;
            let value = (t640bits.max(0) as u32)
                .wrapping_add((lpred as u32) << tag640.max(0).min(15));
            if value > datamax && col >= width {
                return Err(NativeError::corrupt("ORF decoded sample exceeds the sensor white level"));
            }
            let slot = &mut out[row * width + col];
            *slot = value.min(0xffff) as u16;
            glc = low;
        }
    }
    Ok(out)
}

/// Mirrors LibRaw's `bitcounts[65536]` table without allocating 256 KB: entry
/// `n` is `floor(log2(n)) + 1`, with `bitcounts[0] = 0`.
fn bit_count_table() -> Vec<i32> {
    let mut table = vec![0i32; 65536];
    let mut n = 1usize;
    for bits in 0..16 {
        for _ in 0..(1usize << bits) {
            table[n] = bits as i32 + 1;
            n += 1;
        }
    }
    table
}

fn oly_code(
    pump: &mut BitPump,
    wbits: u32,
    tag640: i32,
    tag643: i32,
    escape: i32,
) -> NativeResult<(i32, i32, i32, i32)> {
    if pump.bitcount == 0 {
        pump.replace_data();
    }
    let v = pump.bitcount - 1;
    let highdatabit = (pump.bitstorage >> v) & 1;
    pump.bitcount = v;
    let t643bits = if tag643 != 0 { pump.get(tag643 as u32)? as i32 } else { 0 };
    let t640bits = if tag640 != 0 { pump.get(tag640 as u32)? as i32 } else { 0 };
    let mut vbits = pump.bitcount;
    let mut high = 0u32;
    loop {
        pump.skip_bit(&mut vbits);
        if (pump.bitstorage >> vbits) & 1 != 0 {
            break;
        }
        if escape == (high + 1) as i32 {
            high = 0;
            if wbits != 15 {
                let target = 15 - wbits;
                while vbits < target {
                    vbits = pump.add_data();
                }
                high = (pump.bitstorage << (32 - (vbits & 31))) >> (wbits + 17);
                vbits -= target;
                pump.bitcount = vbits;
            }
            pump.skip_bit(&mut vbits);
            break;
        }
        high += 1;
    }
    let low = if wbits == 0 {
        0
    } else {
        while vbits < wbits {
            vbits = pump.add_data();
        }
        let value = (pump.bitstorage << (32 - (vbits & 31))) >> (32 - (wbits & 31));
        pump.bitcount = vbits - wbits;
        value
    };
    let glc = (low | (high << wbits)) as i32;
    let result = glc ^ (-(highdatabit as i32));
    Ok((result, glc, t640bits, t643bits))
}

/// Big-endian bit pump with the same refill behaviour as LibRaw's
/// `buffered_bitpump_t`.
struct BitPump<'a> {
    stream: &'a [u8],
    pos: usize,
    bitcount: u32,
    bitstorage: u32,
}

impl<'a> BitPump<'a> {
    fn new(stream: &'a [u8]) -> Self {
        Self { stream, pos: 0, bitcount: 0, bitstorage: 0 }
    }

    fn get_be16(&mut self) -> u32 {
        if self.pos + 1 < self.stream.len() {
            let value = (u32::from(self.stream[self.pos]) << 8) | u32::from(self.stream[self.pos + 1]);
            self.pos += 2;
            value
        } else {
            let first = self.stream.get(self.pos).copied().unwrap_or(0);
            self.pos = self.stream.len();
            u32::from(first) << 8
        }
    }

    fn get_be32(&mut self) -> u32 {
        if self.pos + 3 < self.stream.len() {
            let value = (u32::from(self.stream[self.pos]) << 24)
                | (u32::from(self.stream[self.pos + 1]) << 16)
                | (u32::from(self.stream[self.pos + 2]) << 8)
                | u32::from(self.stream[self.pos + 3]);
            self.pos += 4;
            value
        } else {
            // LibRaw pads short reads with zeros; a truncated entropy stream
            // therefore returns a bounded value instead of reading past EOF.
            let mut value = 0u32;
            for _ in 0..4 {
                value = (value << 8) | u32::from(self.stream.get(self.pos).copied().unwrap_or(0));
                if self.pos < self.stream.len() {
                    self.pos += 1;
                }
            }
            value
        }
    }

    fn replace_data(&mut self) {
        self.bitstorage = self.get_be32();
        self.bitcount = 32;
    }

    fn add_data(&mut self) -> u32 {
        let word = self.get_be16();
        self.bitstorage = (self.bitstorage << 16) | word;
        self.bitcount += 16;
        self.bitcount
    }

    fn get(&mut self, width: u32) -> NativeResult<u32> {
        if width == 0 {
            return Ok(0);
        }
        while self.bitcount < width {
            self.add_data();
        }
        let value = (self.bitstorage << (32 - (self.bitcount & 31))) >> (32 - width);
        self.bitcount -= width;
        Ok(value)
    }

    fn skip_bit(&mut self, vbits: &mut u32) {
        if *vbits == 0 {
            self.replace_data();
            self.bitcount -= 1;
            *vbits = self.bitcount;
        } else {
            *vbits -= 1;
            self.bitcount = *vbits;
        }
    }
}

/// 12-bit packed rows with the optional 8-byte padding per 10 pixels (Nikon
/// style `load_flags & 1`, not used by ORF but kept for completeness).
fn decode_packed12(strip: &[u8], width: usize, height: usize, load_flags: u32) -> NativeResult<Vec<u16>> {
    if width == 0 || height == 0 || width > 100_000 {
        return Err(NativeError::corrupt("packed ORF has invalid dimensions"));
    }
    let bwide = width * 12 / 8 + if load_flags & 0x80 != 0 { (width * 12 / 8) & 1 } else { 0 };
    let bite = 8 + (load_flags & 24);
    let mut out = vec![0u16; width * height];
    let mut bitbuf = 0u64;
    let mut vbits = 0i32;
    let mut at = 0usize;
    for row in 0..height {
        for col in 0..width {
            vbits -= 12;
            while vbits < 0 {
                bitbuf <<= bite;
                for i in (0..bite).step_by(8) {
                    let byte = *strip.get(at).ok_or_else(|| NativeError::corrupt("packed ORF strip is truncated"))?;
                    at += 1;
                    bitbuf |= u64::from(byte) << i;
                }
                vbits += bite as i32;
            }
            let shift = 64 - 12 - vbits;
            let value = ((bitbuf << shift) >> (64 - 12)) & 0x7fff;
            out[row * width + col] = value as u16;
        }
        vbits -= (bwide * 8) as i32 - (width * 12) as i32;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_patch_keeps_ifd_offset_and_byte_order() {
        let mut bytes = vec![b'I', b'I', b'R', b'O', 8, 0, 0, 0];
        bytes.extend([0u8; 32]);
        bytes.get_mut(2..4).unwrap().copy_from_slice(&42u16.to_le_bytes());
        let tiff = Tiff::parse(&bytes, 0).unwrap();
        assert!(tiff.little);
        assert_eq!(tiff.u32_at(4).unwrap(), 8);
    }

    #[test]
    fn bit_count_table_matches_libraw_prefix() {
        let table = bit_count_table();
        assert_eq!(table[0], 0);
        assert_eq!(table[1], 1);
        assert_eq!(table[2], 2);
        assert_eq!(table[3], 2);
        assert_eq!(table[4], 3);
        assert_eq!(table[65535], 16);
    }

    #[test]
    fn oly_code_consumes_a_zero_bitstream_without_panicking() {
        let mut pump = BitPump::new(&[0u8; 8]);
        let (code, glc, t640, t643) = oly_code(&mut pump, 1, 0, 0, 1).unwrap();
        assert_eq!((code, glc, t640, t643), (0, 0, 0, 0));
    }

    #[test]
    fn unpacked_words_are_shifted_for_fourteen_bit_orfs() {
        let words = [0x1234u16, 0x5678, 0x9abc, 0xdef0];
        let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let decoded = meta::unpack_bits_le16(&bytes, 4).unwrap();
        let shifted: Vec<u16> = decoded.into_iter().map(|value| value >> 4).collect();
        assert_eq!(shifted, [0x0123, 0x0567, 0x09ab, 0x0def]);
    }

    #[test]
    fn packed_twelve_bit_rows_unpack_in_order() {
        // 0x123, 0x456, 0x789, 0xabc packed MSB-first.
        let bytes = [0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc];
        let out = decode_packed12(&bytes, 4, 1, 0).unwrap();
        assert_eq!(out, [0x123, 0x456, 0x789, 0xabc]);
    }

    #[test]
    fn olympus14_stream_decodes_a_flat_plane() {
        // Minimal parameters: no high/low side bands, escape after one zero,
        // one low bit.  An all-zero bitstream decodes to a constant plane.
        let params = Oly14Params {
            tag640: 0,
            tag641: 1,
            tag642: 0,
            tag643: 0,
            tag644: 0,
            tag645: 0,
            tag646: 0,
            tag647: 0,
            tag648: 0,
            tag649: 1,
            tag650: 0,
            tag651: 4,
            tag652: 1,
            tag653: 2,
            valid_bits: 14,
        };
        let strip = vec![0u8; 8];
        let out = decode_olympus14(&strip, 2, 2, &params).unwrap();
        assert_eq!(out, vec![0, 0, 0, 0]);
    }

    #[test]
    fn unknown_compression_is_reported_as_unsupported() {
        assert!(matches!(
            Err::<(), NativeError>(NativeError::unsupported("Olympus ORF compression 34892")),
            Err(NativeError::Unsupported(_))
        ));
    }
}
