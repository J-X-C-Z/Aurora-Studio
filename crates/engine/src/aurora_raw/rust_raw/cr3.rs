//! Canon CR3 container: ISOBMFF (`moov`/`trak`/`stbl`) plus the `CMP1` codec
//! header and the `CTMD` metadata track that carries Canon's ColorData.

use super::crx::{self, CrxHeader};
use super::tiff::{Tiff, TYPE_ASCII};
use super::{make_frame, meta, NativeError, NativeResult};

use crate::raw::BayerFrame;

#[derive(Clone, Copy)]
struct Atom {
    start: usize,
    body: usize,
    end: usize,
    typ: [u8; 4],
}

fn atoms(data: &[u8], start: usize, end: usize) -> Vec<Atom> {
    let mut out = Vec::new();
    let mut pos = start;
    while pos + 8 <= end {
        let size = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        let typ: [u8; 4] = data[pos + 4..pos + 8].try_into().unwrap();
        let (size, header) = match size {
            0 => (end - pos, 8),
            1 => {
                if pos + 16 > end {
                    break;
                }
                (u64::from_be_bytes(data[pos + 8..pos + 16].try_into().unwrap()) as usize, 16)
            }
            value => (value, 8),
        };
        if size < header || pos + size > end {
            break;
        }
        out.push(Atom { start: pos, body: pos + header, end: pos + size, typ });
        pos += size;
    }
    out
}

fn child<'a>(data: &'a [u8], parent: Atom, typ: &[u8; 4]) -> Option<Atom> {
    atoms(data, parent.body, parent.end).into_iter().find(|atom| &atom.typ == typ)
}

fn path<'a>(data: &'a [u8], mut atom: Atom, names: &[&[u8; 4]]) -> Option<Atom> {
    for name in names {
        atom = child(data, atom, name)?;
    }
    Some(atom)
}

#[derive(Default)]
struct Track {
    header: Option<CrxHeader>,
    cmp1: Option<Vec<u8>>,
    cdi1: Option<Vec<u8>>,
    sample_offset: usize,
    sample_size: usize,
}

pub(crate) fn decode(bytes: &[u8]) -> NativeResult<BayerFrame> {
    if bytes.get(8..12) != Some(b"crx ") && bytes.get(8..12) != Some(b"crx\0") {
        // Some writers use the generic `crx ` brand further down `ftyp`; the
        // caller only reaches here for ISOBMFF, so continue anyway.
    }
    let moov = atoms(bytes, 0, bytes.len())
        .into_iter()
        .find(|atom| &atom.typ == b"moov")
        .ok_or_else(|| NativeError::corrupt("CR3 has no moov atom"))?;

    let mut best: Option<Track> = None;
    let mut ctmd: Option<(usize, usize)> = None;
    for trak in atoms(bytes, moov.body, moov.end).into_iter().filter(|atom| &atom.typ == b"trak") {
        let Some(stbl) = path(bytes, trak, &[b"mdia", b"minf", b"stbl"]) else {
            continue;
        };
        let Some(stsd) = child(bytes, stbl, b"stsd") else {
            continue;
        };
        let Some(entry) = atoms(bytes, stsd.body + 8, stsd.end).into_iter().next() else {
            continue;
        };
        let format = entry.typ;
        if &format == b"CTMD" {
            if let Some((offset, size)) = sample_table(bytes, stbl) {
                ctmd = Some((offset, size));
            }
            continue;
        }
        if &format != b"CRAW" {
            continue;
        }
        let mut track = Track::default();
        let entry_bytes = &bytes[entry.start..entry.end];
        if let Some(index) = find_signature(entry_bytes, b"CMP1") {
            let size = u32::from_be_bytes(entry_bytes[index - 4..index].try_into().unwrap()) as usize;
            if index - 4 + size <= entry_bytes.len() {
                track.cmp1 = Some(entry_bytes[index + 4..index - 4 + size].to_vec());
            }
        }
        if let Some(index) = find_signature(entry_bytes, b"CDI1") {
            let size = u32::from_be_bytes(entry_bytes[index - 4..index].try_into().unwrap()) as usize;
            if index - 4 + size <= entry_bytes.len() {
                track.cdi1 = Some(entry_bytes[index + 4..index - 4 + size].to_vec());
            }
        }
        let (offset, size) = match sample_table(bytes, stbl) {
            Some(value) => value,
            None => continue,
        };
        track.sample_offset = offset;
        track.sample_size = size;
        if let Some(payload) = track.cmp1.as_deref() {
            if let Ok(header) = crx::parse_header(payload) {
                track.header = Some(header);
            }
        }
        let area = track.header.as_ref().map(|h| h.width * h.height).unwrap_or(0);
        if best.as_ref().map(|current| current.header.as_ref().map(|h| h.width * h.height).unwrap_or(0)).unwrap_or(0) < area {
            best = Some(track);
        }
    }

    let track = best.ok_or_else(|| NativeError::unsupported("CR3 has no raw CRAW/CMP1 track"))?;
    let header = track.header.clone().ok_or_else(|| NativeError::corrupt("CR3 raw track has no CRX header"))?;
    if track.sample_size < header.mdat_header_size || track.sample_offset + track.sample_size > bytes.len() {
        return Err(NativeError::corrupt("CR3 raw sample is outside the file"));
    }
    let model = canon_model(bytes, moov);
    let (cam_mul, black) = ctmd
        .and_then(|(offset, size)| color_data(bytes, offset, size))
        .unwrap_or(([0.0; 4], 0));

    let sample =&bytes[track.sample_offset..track.sample_offset + track.sample_size];
    let (plane, raw_width, raw_height) = crx::decode_sample(sample, &header)?;

    let (left, top, width, height) = canon_geometry(raw_width, raw_height, track.cdi1.as_deref());
    let (filters, colors) = cfa_filters(header.cfa_layout)?;
    let maximum = (1u32 << header.bits) - 1;
    let rgb_cam = meta::adobe_matrix("Canon", &adobe_model(&model))
        .map(|matrix| meta::cam_xyz_to_rgb_cam(&matrix, colors as usize))
        .unwrap_or([[0.0f32; 4]; 3]);

    make_frame(
        raw_width,
        raw_height,
        left,
        top,
        width,
        height,
        header.bits,
        plane,
        filters,
        colors,
        black,
        maximum,
        cam_mul,
        rgb_cam,
    )
}

fn find_signature(haystack: &[u8], needle: &[u8; 4]) -> Option<usize> {
    haystack.windows(4).position(|window| window == needle).filter(|&index| index >= 4)
}

fn sample_table(bytes: &[u8], stbl: Atom) -> Option<(usize, usize)> {
    let stsz = child(bytes, stbl, b"stsz")?;
    let body = stsz.body;
    if body + 12 > stsz.end {
        return None;
    }
    let size = u32::from_be_bytes(bytes[body + 4..body + 8].try_into().unwrap()) as usize;
    let size = if size == 0 {
        u32::from_be_bytes(bytes[body + 12..body + 16].try_into().unwrap()) as usize
    } else {
        size
    };
    let co = child(bytes, stbl, b"co64").or_else(|| child(bytes, stbl, b"stco"))?;
    let offset = if &co.typ == b"co64" {
        u64::from_be_bytes(bytes[co.body + 8..co.body + 16].try_into().unwrap()) as usize
    } else {
        u32::from_be_bytes(bytes[co.body + 8..co.body + 12].try_into().unwrap()) as usize
    };
    Some((offset, size))
}

fn canon_model(bytes: &[u8], moov: Atom) -> String {
    let Some(uuid) = atoms(bytes, moov.body, moov.end)
        .into_iter()
        .find(|atom| &atom.typ == b"uuid" && atom.body + 16 <= atom.end && bytes[atom.body..atom.body + 16] == CANON_UUID)
    else {
        return String::new();
    };
    for atom in atoms(bytes, uuid.body + 16, uuid.end) {
        if &atom.typ != b"CMT1" {
            continue;
        }
        let Ok(tiff) = Tiff::parse(bytes, atom.body) else {
            continue;
        };
        // The IFD pointer lives at header+4; `u32_at` takes an absolute offset.
        let Ok(ifd_pointer) = tiff.absolute(4) else {
            continue;
        };
        let Ok(root) = tiff.ifd(tiff.u32_at(ifd_pointer).unwrap_or(0)) else {
            continue;
        };
        for tag in [0x0110u16, 0x010f] {
            if let Some(entry) = root.entry(tag) {
                if entry.typ == TYPE_ASCII {
                    if let Ok(raw) = root.bytes(entry) {
                        let text = raw.split(|&value| value == 0).next().unwrap_or_default();
                        let text = String::from_utf8_lossy(text).trim().to_string();
                        if tag == 0x0110 {
                            return text;
                        }
                    }
                }
            }
        }
    }
    String::new()
}

const CANON_UUID: [u8; 16] = [
    0x85, 0xc0, 0xb6, 0x87, 0x82, 0x0f, 0x11, 0xe0, 0x81, 0x11, 0xf4, 0xce, 0x46, 0x2b, 0x6a, 0x48,
];

/// Canon writes the generation as a compact EXIF suffix ("Canon EOS R5m2"),
/// while the Adobe matrix table uses the marketing name ("EOS R5 Mark II").
fn adobe_model(model: &str) -> String {
    let bytes = model.as_bytes();
    if bytes.len() >= 2 && bytes[bytes.len() - 2] == b'm' {
        let roman = match bytes[bytes.len() - 1] {
            b'2' => "II",
            b'3' => "III",
            b'4' => "IV",
            _ => return model.to_string(),
        };
        return format!("{} Mark {roman}", &model[..model.len() - 2]);
    }
    model.to_string()
}

/// Canon's static crop table (identify.cpp): (raw_w, raw_h, left, top).
const CANON_CROP: &[(usize, usize, usize, usize)] = &[
    (5344, 3584, 126, 100),
    (5360, 3516, 158, 51),
    (5568, 3708, 72, 38),
    (5792, 3804, 158, 51),
    (5920, 3950, 122, 80),
    (6096, 4051, 76, 35),
    (6096, 4056, 72, 34),
    (6288, 4056, 264, 36),
    (6384, 4224, 120, 44),
    (6880, 4544, 136, 42),
    (6888, 4546, 146, 48),
    (7128, 4732, 144, 72),
    (8896, 5920, 160, 64),
    (6192, 4152, 160, 120),
    (6192, 4060, 144, 40),
    (6188, 4120, 154, 96),
    (3936, 2612, 154, 96),
    (8480, 5650, 258, 162),
    (5376, 3574, 258, 162),
    (6160, 4144, 128, 120),
    (5928, 3944, 144, 80),
    (7144, 4760, 158, 96),
    (4512, 2994, 160, 96),
];

fn canon_geometry(raw_width: usize, raw_height: usize, cdi1: Option<&[u8]>) -> (usize, usize, usize, usize) {
    for &(width, height, left, top) in CANON_CROP {
        if width == raw_width && height == raw_height {
            return (left, top, raw_width - left, raw_height - top);
        }
    }
    // Fall back to CDI1's ActiveArea with LibRaw's two-pixel alignment.
    if let Some(cdi1) = cdi1 {
        if cdi1.len() >= 60 && &cdi1[8..12] == b"IAD1" {
            let area = &cdi1[52..60];
            let read = |at: usize| u16::from_be_bytes([area[at], area[at + 1]]) as usize;
            let (left, top, right, bottom) = (read(0), read(2), read(4), read(6));
            if right > left && bottom > top {
                let left = left + 2;
                let top = top + 2;
                let width = (right + 1).saturating_sub(left);
                let height = (bottom + 1).saturating_sub(top);
                if left + width <= raw_width && top + height <= raw_height {
                    return (left, top, width, height);
                }
            }
        }
    }
    (0, 0, raw_width, raw_height)
}

fn cfa_filters(layout: u8) -> NativeResult<(u32, u32)> {
    let pattern: [u8; 4] = match layout {
        0 => [0, 1, 1, 2],
        1 => [1, 0, 2, 1],
        2 => [1, 2, 0, 1],
        3 => [2, 1, 1, 0],
        _ => return Err(NativeError::corrupt("CRX CFA layout is invalid")),
    };
    let (filters, colors) = meta::filters_from_cfa(Some((2, 2)), &pattern, None)?;
    Ok((meta::upgrade_filters(filters, colors), colors))
}

/// Reads cam_mul and the scalar black level from the CTMD track's ColorData.
fn color_data(bytes: &[u8], offset: usize, size: usize) -> Option<([f32; 4], u32)> {
    let end = (offset + size).min(bytes.len());
    let mut pos = offset;
    while pos + 6 <= end {
        let item_size = u32::from_le_bytes(bytes.get(pos..pos + 4)?.try_into().ok()?) as usize;
        let item_type = u16::from_le_bytes(bytes.get(pos + 4..pos + 6)?.try_into().ok()?);
        if item_size < 1 || pos + item_size > end {
            return None;
        }
        if matches!(item_type, 7 | 8 | 9) {
            let mut child = pos + 12;
            while child + 8 < pos + item_size {
                let length = u32::from_le_bytes(bytes.get(child..child + 4)?.try_into().ok()?) as usize;
                let tag = u32::from_le_bytes(bytes.get(child + 4..child + 8)?.try_into().ok()?);
                if length < 8 || child + length > pos + item_size {
                    break;
                }
                if tag == 0x927c && length > 8 {
                    if let Some(result) = parse_color_data(bytes, child + 8) {
                        return Some(result);
                    }
                }
                child += length;
            }
        }
        pos += item_size;
    }
    None
}

fn parse_color_data(bytes: &[u8], base: usize) -> Option<([f32; 4], u32)> {
    let tiff = Tiff::parse(bytes, base).ok()?;
    // The IFD pointer lives at header+4; `u32_at` takes an absolute offset.
    let root = tiff.ifd(tiff.u32_at(tiff.absolute(4).ok()?).ok()?).ok()?;
    let entry = root.entry(0x4001)?;
    let raw = root.bytes(entry).ok()?;
    // Canon's table is indexed by the ColorData element count, not byte length.
    let length = entry.count as usize;
    let (cam_offset, black_offset) = match length {
        1816 | 1824 => (0x47usize, 0x47 + 0x2d1),
        1770 | 2024 | 3656 => (0x55, 0x55 + 0x2d1),
        3973 | 3778 | 4528 => (0x69, 0x69 + 0x227),
        _ => return None,
    };
    let read = |index: usize| -> Option<u16> {
        let at = index * 2;
        if at + 2 > raw.len() {
            return None;
        }
        Some(if tiff.little {
            u16::from_le_bytes([raw[at], raw[at + 1]])
        } else {
            u16::from_be_bytes([raw[at], raw[at + 1]])
        })
    };
    let values = [read(cam_offset)?, read(cam_offset + 1)?, read(cam_offset + 2)?, read(cam_offset + 3)?];
    let channels = [read(black_offset)?, read(black_offset + 1)?, read(black_offset + 2)?, read(black_offset + 3)?];
    let black = u32::from(*channels.iter().min()?);
    Some(([values[0] as f32, values[1] as f32, values[3] as f32, values[2] as f32], black))
}
