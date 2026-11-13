//! EXIF metadata reader used by the import pipeline.
//! Covers JPEG (APP1), TIFF and PNG (`eXIf`); WebP and RAW are not implemented,
//! and RAW would need a native decoder.

use crate::catalog::Iptc;
use crate::Exif;

/// EXIF sits in an APP1 / `eXIf` block near the start. A folder scan used to
/// read every whole file, which made opening a folder of 24 MP JPEGs disk-bound.
const HEAD_BYTES: u64 = 1 << 20;

/// Read EXIF from a file. Returns `Exif::default()` when nothing can be read.
pub fn read_exif(path: &std::path::Path) -> crate::Exif {
    read_metadata(path).0
}

/// Read EXIF and IPTC together, one pass over the file head.
pub fn read_metadata(path: &std::path::Path) -> (Exif, Iptc) {
    use std::io::Read;
    // HEIF keeps EXIF in a metadata item, not in an APP1/eXIf block; the
    // container bytes themselves have no TIFF header to scan.
    if matches!(
        path.extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref(),
        Some("heic" | "heif")
    ) {
        if let Some(tiff) = crate::heif::read_exif(path) {
            return (parse_exif(&tiff), Iptc::default());
        }
    }
    let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let Ok(mut file) = std::fs::File::open(path) else { return (Exif::default(), Iptc::default()) };
    let mut head = Vec::with_capacity(len.min(HEAD_BYTES) as usize);
    if file.by_ref().take(HEAD_BYTES).read_to_end(&mut head).is_err() {
        return (Exif::default(), Iptc::default());
    }
    let exif = parse_exif(&head);
    let iptc = crate::iptc::parse_iptc(&head);
    // A file no longer than the cap was read whole: nothing more to find.
    if exif != Exif::default() || iptc != Iptc::default() || len <= HEAD_BYTES {
        return (exif, iptc);
    }
    // Rare: a writer put EXIF/IPTC past the head. Only these files pay for a full read.
    std::fs::read(path)
        .map(|bytes| (parse_exif(&bytes), crate::iptc::parse_iptc(&bytes)))
        .unwrap_or_default()
}

/// Parse EXIF out of an in-memory image. Never panics.
pub fn parse_exif(data: &[u8]) -> Exif {
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        return parse_png(data);
    }
    if data.len() >= 4 && (data.starts_with(b"II\x2a\x00") || data.starts_with(b"MM\x00\x2a")) {
        return parse_tiff(data);
    }
    if let Some(soi) = find_sub(data, &[0xFF, 0xD8]) {
        return parse_jpeg(&data[soi..]);
    }
    Exif::default()
}

/// Convert GPS degrees/minutes/seconds to decimal degrees.
pub fn gps_to_decimal(deg: f64, min: f64, sec: f64, ref_neg: bool) -> f64 {
    let value = deg + min / 60.0 + sec / 3600.0;
    if ref_neg { -value } else { value }
}

fn find_sub(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

// ---------------------------------------------------------------- JPEG

fn parse_jpeg(data: &[u8]) -> Exif {
    let mut i = 2usize; // past SOI
    while i + 1 < data.len() {
        if data[i] != 0xFF {
            i += 1;
            continue;
        }
        let mut j = i + 1;
        while j < data.len() && data[j] == 0xFF {
            j += 1;
        }
        if j >= data.len() {
            break;
        }
        let marker = data[j];
        i = j + 1;
        match marker {
            0xD8 | 0x01 | 0xD0..=0xD7 => continue,
            0xDA => break, // start of scan: entropy coded data follows
            _ => {}
        }
        if i + 1 >= data.len() {
            break;
        }
        let len = u16::from_be_bytes([data[i], data[i + 1]]) as usize;
        if len < 2 {
            break;
        }
        let start = i + 2;
        let end = match start.checked_add(len - 2) {
            Some(e) if e <= data.len() => e,
            _ => break,
        };
        if marker == 0xE1 && end >= start + 6 && &data[start..start + 6] == b"Exif\0\0" {
            return parse_tiff(&data[start + 6..end]);
        }
        i = end;
    }
    Exif::default()
}

// ----------------------------------------------------------------- PNG

fn parse_png(data: &[u8]) -> Exif {
    let mut i = 8usize;
    while i + 8 <= data.len() {
        let len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        let kind = &data[i + 4..i + 8];
        let start = i + 8;
        let end = match start.checked_add(len) {
            Some(e) if e + 4 <= data.len() => e,
            _ => break,
        };
        if kind == b"eXIf" {
            return parse_tiff(&data[start..end]);
        }
        i = end + 4;
    }
    Exif::default()
}

// ---------------------------------------------------------------- TIFF

fn parse_tiff(data: &[u8]) -> Exif {
    if data.len() < 8 {
        return Exif::default();
    }
    let le = match &data[0..2] {
        b"II" => true,
        b"MM" => false,
        _ => return Exif::default(),
    };
    if rd_u16(data, 2, le) != Some(42) {
        return Exif::default();
    }
    let ifd0 = match rd_u32(data, 4, le) {
        Some(v) => v as usize,
        None => return Exif::default(),
    };
    let entries = read_ifd(data, ifd0, le);

    let mut ex = Exif::default();
    apply_ifd0(data, &entries, &mut ex);

    if let Some(off) = sub_ifd_offset(data, &entries, le, 0x8769) {
        let sub = read_ifd(data, off, le);
        apply_exif_ifd(data, &sub, le, &mut ex);
    }
    if let Some(off) = sub_ifd_offset(data, &entries, le, 0x8825) {
        let sub = read_ifd(data, off, le);
        apply_gps(data, &sub, le, &mut ex);
    }
    ex
}

fn sub_ifd_offset(data: &[u8], entries: &[Entry], le: bool, tag: u16) -> Option<usize> {
    let e = entries.iter().find(|e| e.tag == tag)?;
    rd_u32(data, e.off, le).map(|v| v as usize)
}

fn apply_ifd0(data: &[u8], entries: &[Entry], ex: &mut Exif) {
    for e in entries {
        match e.tag {
            0x010F => set_string(&mut ex.camera_make, value_ascii(data, e)),
            0x0110 => set_string(&mut ex.camera_model, value_ascii(data, e)),
            0x0132 => set_string(&mut ex.captured_at, value_ascii(data, e).and_then(|s| exif_datetime(&s))),
            _ => {}
        }
    }
}

fn apply_exif_ifd(data: &[u8], entries: &[Entry], le: bool, ex: &mut Exif) {
    for e in entries {
        match e.tag {
            0x829A => set_f32(&mut ex.shutter, value_rational(data, e, le)),
            0x829D => set_f32(&mut ex.aperture, value_rational(data, e, le)),
            0x8827 => set_u32(&mut ex.iso, value_short(data, e, le)),
            0x920A => set_f32(&mut ex.focal_length_mm, value_rational(data, e, le)),
            0x9201 => set_f32(&mut ex.exposure_bias, value_srational(data, e, le)),
            0x9209 => set_bool(&mut ex.flash, value_short(data, e, le).map(|v| v & 1 != 0)),
            0xA434 => set_string(&mut ex.lens, value_ascii(data, e)),
            0x9003 => {
                if ex.captured_at.is_none() {
                    set_string(&mut ex.captured_at, value_ascii(data, e).and_then(|s| exif_datetime(&s)));
                }
            }
            _ => {}
        }
    }
}

fn apply_gps(data: &[u8], entries: &[Entry], le: bool, ex: &mut Exif) {
    let lat = gps_pair(data, entries, le, 0x0001, 0x0002);
    let lon = gps_pair(data, entries, le, 0x0003, 0x0004);
    if let (Some((lat, lat_neg)), Some((lon, lon_neg))) = (lat, lon) {
        ex.gps = Some((
            gps_to_decimal(lat.0, lat.1, lat.2, lat_neg),
            gps_to_decimal(lon.0, lon.1, lon.2, lon_neg),
        ));
    }
}

fn gps_pair(
    data: &[u8],
    entries: &[Entry],
    le: bool,
    ref_tag: u16,
    value_tag: u16,
) -> Option<((f64, f64, f64), bool)> {
    let value = entries.iter().find(|e| e.tag == value_tag)?;
    let parts = value_rationals(data, value, le, 3)?;
    let neg = entries
        .iter()
        .find(|e| e.tag == ref_tag)
        .and_then(|e| value_ascii(data, e))
        .map(|s| {
            let s = s.trim().to_ascii_uppercase();
            s == "S" || s == "W"
        })
        .unwrap_or(false);
    Some(((parts[0], parts[1], parts[2]), neg))
}

// ------------------------------------------------------------- IFD read

struct Entry {
    tag: u16,
    typ: u16,
    off: usize,
    len: usize,
}

fn type_size(typ: u16) -> usize {
    match typ {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 | 13 => 4,
        5 | 10 | 12 => 8,
        _ => 0,
    }
}

fn read_ifd(data: &[u8], off: usize, le: bool) -> Vec<Entry> {
    let mut out = Vec::new();
    let count = match rd_u16(data, off, le) {
        Some(v) => v as usize,
        None => return out,
    };
    if count > 4096 {
        return out;
    }
    for k in 0..count {
        let eo = match off.checked_add(2 + k * 12) {
            Some(v) => v,
            None => break,
        };
        if eo + 12 > data.len() {
            break;
        }
        let tag = match rd_u16(data, eo, le) {
            Some(v) => v,
            None => break,
        };
        let typ = match rd_u16(data, eo + 2, le) {
            Some(v) => v,
            None => break,
        };
        let count = match rd_u32(data, eo + 4, le) {
            Some(v) => v as usize,
            None => break,
        };
        let size = type_size(typ);
        if size == 0 {
            continue;
        }
        let total = size.saturating_mul(count);
        let (len, start) = if total <= 4 {
            (total, eo + 8)
        } else {
            let v = match rd_u32(data, eo + 8, le) {
                Some(v) => v as usize,
                None => continue,
            };
            (total, v)
        };
        if start.saturating_add(len) > data.len() {
            continue;
        }
        out.push(Entry { tag, typ, off: start, len });
    }
    out
}

fn value_ascii(data: &[u8], e: &Entry) -> Option<String> {
    if e.typ != 2 && e.typ != 7 {
        return None;
    }
    if e.len == 0 {
        return None;
    }
    let slice = &data[e.off..e.off + e.len];
    let text: String = slice
        .iter()
        .take_while(|b| **b != 0)
        .map(|b| *b as char)
        .collect();
    let text = text.trim();
    if text.is_empty() { None } else { Some(text.to_string()) }
}

fn value_short(data: &[u8], e: &Entry, le: bool) -> Option<u32> {
    if e.typ != 3 || e.len < 2 {
        return None;
    }
    rd_u16(data, e.off, le).map(|v| v as u32)
}

fn value_rational(data: &[u8], e: &Entry, le: bool) -> Option<f64> {
    value_rationals(data, e, le, 1).map(|v| v[0])
}

fn value_rationals(data: &[u8], e: &Entry, le: bool, want: usize) -> Option<Vec<f64>> {
    if e.typ != 5 && e.typ != 10 {
        return None;
    }
    let signed = e.typ == 10;
    let mut out = Vec::new();
    for k in 0..want {
        let base = e.off + k * 8;
        if base + 8 > data.len() {
            return None;
        }
        let (num, den) = if signed {
            (
                rd_i32(data, base, le)? as f64,
                rd_i32(data, base + 4, le)? as f64,
            )
        } else {
            (
                rd_u32(data, base, le)? as f64,
                rd_u32(data, base + 4, le)? as f64,
            )
        };
        if den == 0.0 {
            return None;
        }
        out.push(num / den);
    }
    Some(out)
}

fn value_srational(data: &[u8], e: &Entry, le: bool) -> Option<f64> {
    value_rationals(data, e, le, 1).map(|v| v[0])
}

// ------------------------------------------------------------- helpers

fn rd_u16(data: &[u8], off: usize, le: bool) -> Option<u16> {
    if off + 2 > data.len() {
        return None;
    }
    let b = [data[off], data[off + 1]];
    Some(if le { u16::from_le_bytes(b) } else { u16::from_be_bytes(b) })
}

fn rd_u32(data: &[u8], off: usize, le: bool) -> Option<u32> {
    if off + 4 > data.len() {
        return None;
    }
    let b = [data[off], data[off + 1], data[off + 2], data[off + 3]];
    Some(if le { u32::from_le_bytes(b) } else { u32::from_be_bytes(b) })
}

fn rd_i32(data: &[u8], off: usize, le: bool) -> Option<i32> {
    rd_u32(data, off, le).map(|v| v as i32)
}

/// `YYYY:MM:DD HH:MM:SS` to RFC 3339 `YYYY-MM-DDTHH:MM:SS`.
fn exif_datetime(raw: &str) -> Option<String> {
    let b = raw.as_bytes();
    if b.len() < 19 {
        return None;
    }
    let digits = |r: &[u8]| r.iter().all(|c| c.is_ascii_digit());
    if !digits(&b[0..4]) || b[4] != b':' || !digits(&b[5..7]) || b[7] != b':' {
        return None;
    }
    if !digits(&b[8..10]) || b[10] != b' ' || !digits(&b[11..13]) || b[13] != b':' {
        return None;
    }
    if !digits(&b[14..16]) || b[16] != b':' || !digits(&b[17..19]) {
        return None;
    }
    Some(format!(
        "{}-{}-{}T{}:{}:{}",
        &raw[0..4],
        &raw[5..7],
        &raw[8..10],
        &raw[11..13],
        &raw[14..16],
        &raw[17..19]
    ))
}

fn set_string(slot: &mut Option<String>, value: Option<String>) {
    if let Some(v) = value {
        *slot = Some(v);
    }
}

fn set_f32(slot: &mut Option<f32>, value: Option<f64>) {
    if let Some(v) = value {
        *slot = Some(v as f32);
    }
}

fn set_u32(slot: &mut Option<u32>, value: Option<u32>) {
    if let Some(v) = value {
        *slot = Some(v);
    }
}

fn set_bool(slot: &mut Option<bool>, value: Option<bool>) {
    if let Some(v) = value {
        *slot = Some(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    fn temp_file(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("lumen-exif-{}-{name}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }






    #[test]
    fn heif_exif_is_read_through_libheif() {
        if !crate::heif_write::available() {
            return;
        }
        let mut text = b"LumenCam".to_vec();
        text.push(0);
        let mut tiff = Vec::new();
        tiff.extend_from_slice(b"II*\0");
        tiff.extend_from_slice(&8u32.to_le_bytes());
        tiff.extend_from_slice(&1u16.to_le_bytes());
        tiff.extend_from_slice(&0x010Fu16.to_le_bytes());
        tiff.extend_from_slice(&2u16.to_le_bytes());
        tiff.extend_from_slice(&(text.len() as u32).to_le_bytes());
        tiff.extend_from_slice(&26u32.to_le_bytes());
        tiff.extend_from_slice(&0u32.to_le_bytes());
        tiff.extend_from_slice(&text);

        let path =
            std::env::temp_dir().join(format!("lumen-exif-heif-{}.heic", std::process::id()));
        crate::heif_write::encode(
            &path,
            &vec![90u8; 16 * 12 * 4],
            16,
            12,
            80,
            None,
            Some(&tiff),
            None,
        )
        .unwrap();
        let (exif, iptc) = read_metadata(&path);
        assert_eq!(exif.camera_make.as_deref(), Some("LumenCam"));
        assert_eq!(iptc, Iptc::default());
        std::fs::remove_file(&path).unwrap();
    }

    #[cfg(feature = "external-fixtures")]
    include!("../local-fixtures/exif-tests.rs");
}
