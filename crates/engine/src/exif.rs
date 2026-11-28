//! EXIF metadata reader used by the import pipeline.
//! Covers JPEG (APP1), TIFF (IFD0 + ExifIFD + GPS IFD), PNG (`eXIf`) and HEIF.
//! RAW EXIF arrives through the same TIFF walk: NEF/CR2 and friends are TIFF
//! containers. WebP is not implemented.

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
    // Some writers keep Exif tags in IFD0 instead of a separate ExifIFD; the
    // sub-IFD pass below still wins when both carry the same tag.
    apply_exif_ifd(data, &entries, le, &mut ex);

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
            0x8827 => set_u32(&mut ex.iso, value_u32(data, e, le)),
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

/// ISO is SHORT per the spec but LONG in files written by some editors.
fn value_u32(data: &[u8], e: &Entry, le: bool) -> Option<u32> {
    match e.typ {
        3 if e.len >= 2 => rd_u16(data, e.off, le).map(u32::from),
        4 if e.len >= 4 => rd_u32(data, e.off, le),
        _ => None,
    }
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

    /// A JPEG-shaped buffer carrying the real NEF EXIF block as APP1.
    fn jpeg_with_exif() -> Vec<u8> {
        let tiff = std::fs::read(
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/nef-exif-head.bin"),
        )
        .expect("testdata sample");
        let mut out = vec![0xFF, 0xD8, 0xFF, 0xE1];
        out.extend_from_slice(&((2 + 6 + tiff.len()) as u16).to_be_bytes());
        out.extend_from_slice(b"Exif\0\0");
        out.extend_from_slice(&tiff);
        out.extend_from_slice(&[0xFF, 0xD9]);
        out
    }

    fn temp_file(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("portrait-exif-{}-{name}", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn exif_is_found_in_the_file_head_without_reading_the_rest() {
        let exif = jpeg_with_exif();
        let expected = parse_exif(&exif);
        assert!(expected.camera_model.is_some(), "fixture must carry EXIF");
        let mut padded = exif.clone();
        padded.resize(exif.len() + 2 * 1024 * 1024, 0);
        let path = temp_file("head.jpg", &padded);
        assert_eq!(read_exif(&path), expected);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn exif_past_the_head_still_falls_back_to_a_full_read() {
        let exif = jpeg_with_exif();
        let expected = parse_exif(&exif);
        let mut late = vec![0u8; HEAD_BYTES as usize + 4096];
        late.extend_from_slice(&exif);
        let path = temp_file("late.jpg", &late);
        assert_eq!(read_exif(&path), expected);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn real_sample_carries_exif_and_iptc() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/iptc-sample.jpg");
        let (exif, iptc) = read_metadata(&path);
        assert_eq!(exif.camera_make.as_deref(), Some("Sony"));
        assert_eq!(exif.camera_model.as_deref(), Some("ILCE-7M4"));
        assert_eq!(iptc.keywords, ["wedding", "hero & co"]);
    }

    #[test]
    fn heif_exif_is_read_through_libheif() {
        if !crate::heif_write::available() {
            return;
        }
        let mut text = b"PortraitCam".to_vec();
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
            std::env::temp_dir().join(format!("portrait-exif-heif-{}.heic", std::process::id()));
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
        assert_eq!(exif.camera_make.as_deref(), Some("PortraitCam"));
        assert_eq!(iptc, Iptc::default());
        std::fs::remove_file(&path).unwrap();
    }

    // ------------------------------------------------ TIFF fixtures

    struct TiffEnt {
        tag: u16,
        typ: u16,
        count: u32,
        payload: Vec<u8>,
    }

    fn enc16(le: bool, v: u16) -> [u8; 2] {
        if le { v.to_le_bytes() } else { v.to_be_bytes() }
    }

    fn enc32(le: bool, v: u32) -> [u8; 4] {
        if le { v.to_le_bytes() } else { v.to_be_bytes() }
    }

    fn ent(tag: u16, typ: u16, count: u32, payload: Vec<u8>) -> TiffEnt {
        TiffEnt { tag, typ, count, payload }
    }

    fn ascii_ent(tag: u16, text: &str) -> TiffEnt {
        let mut v = text.as_bytes().to_vec();
        v.push(0);
        let n = v.len() as u32;
        ent(tag, 2, n, v)
    }

    fn short_ent(le: bool, tag: u16, v: u16) -> TiffEnt {
        ent(tag, 3, 1, enc16(le, v).to_vec())
    }

    fn long_ent(le: bool, tag: u16, v: u32) -> TiffEnt {
        ent(tag, 4, 1, enc32(le, v).to_vec())
    }

    fn rational_ent(le: bool, tag: u16, n: u32, d: u32) -> TiffEnt {
        let mut v = enc32(le, n).to_vec();
        v.extend(enc32(le, d));
        ent(tag, 5, 1, v)
    }

    fn push_ifd(out: &mut Vec<u8>, le: bool, entries: &[TiffEnt], next: u32) -> usize {
        let at = out.len();
        out.extend(enc16(le, entries.len() as u16));
        let data_at = at + 2 + entries.len() * 12 + 4;
        let mut extra = Vec::new();
        for e in entries {
            out.extend(enc16(le, e.tag));
            out.extend(enc16(le, e.typ));
            out.extend(enc32(le, e.count));
            if e.payload.len() <= 4 {
                let mut inline = e.payload.clone();
                inline.resize(4, 0);
                out.extend(inline);
            } else {
                out.extend(enc32(le, (data_at + extra.len()) as u32));
                extra.extend_from_slice(&e.payload);
            }
        }
        out.extend(enc32(le, next));
        out.extend(extra);
        at
    }

    /// A TIFF whose IFD0 sits `gap` bytes after the header, with an ExifIFD and
    /// a GPS IFD, mirroring the layout Capture One writes (IFD0 at the tail).
    fn build_tiff(le: bool, gap: usize) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend(if le { b"II*\0" } else { b"MM\0*" });
        out.extend([0u8; 4]);
        out.resize(8 + gap, 0);

        let exif_at = push_ifd(
            &mut out,
            le,
            &[
                rational_ent(le, 0x829A, 1, 40),
                rational_ent(le, 0x829D, 4, 1),
                short_ent(le, 0x8827, 5000),
                ascii_ent(0x9003, "2026:09:13 19:26:24"),
                rational_ent(le, 0x920A, 120, 1),
                ascii_ent(0xA434, "NIKKOR Z 24-120mm f/4 S"),
            ],
            0,
        );
        let mut gps_payload = Vec::new();
        for (n, d) in [(22u32, 1u32), (30, 1), (0, 1)] {
            gps_payload.extend(enc32(le, n));
            gps_payload.extend(enc32(le, d));
        }
        let mut lon_payload = Vec::new();
        for (n, d) in [(114u32, 1u32), (10, 1), (0, 1)] {
            lon_payload.extend(enc32(le, n));
            lon_payload.extend(enc32(le, d));
        }
        let gps_at = push_ifd(
            &mut out,
            le,
            &[
                ascii_ent(0x0001, "S"),
                ent(0x0002, 5, 3, gps_payload),
                ascii_ent(0x0003, "E"),
                ent(0x0004, 5, 3, lon_payload),
            ],
            0,
        );
        let ifd0 = push_ifd(
            &mut out,
            le,
            &[
                long_ent(le, 0x0100, 64),
                long_ent(le, 0x0101, 48),
                ascii_ent(0x010F, "NIKON CORPORATION"),
                ascii_ent(0x0110, "NIKON Z6_3"),
                ascii_ent(0x0132, "2026:09:13 19:26:24"),
                long_ent(le, 0x8769, exif_at as u32),
                long_ent(le, 0x8825, gps_at as u32),
            ],
            0,
        );
        out[4..8].copy_from_slice(&enc32(le, ifd0 as u32));
        out
    }

    fn assert_nikon_exif(ex: &Exif) {
        assert_eq!(ex.camera_make.as_deref(), Some("NIKON CORPORATION"));
        assert_eq!(ex.camera_model.as_deref(), Some("NIKON Z6_3"));
        assert_eq!(ex.lens.as_deref(), Some("NIKKOR Z 24-120mm f/4 S"));
        assert_eq!(ex.iso, Some(5000));
        assert_eq!(ex.shutter, Some(0.025));
        assert_eq!(ex.aperture, Some(4.0));
        assert_eq!(ex.focal_length_mm, Some(120.0));
        assert_eq!(ex.captured_at.as_deref(), Some("2026-09-13T19:26:24"));
    }

    #[test]
    fn tiff_exif_reads_ifd0_exif_ifd_and_gps() {
        let bytes = build_tiff(true, 0);
        let ex = parse_exif(&bytes);
        assert_nikon_exif(&ex);
        let (lat, lon) = ex.gps.expect("gps");
        assert!((lat + 22.5).abs() < 1e-6, "lat {lat}");
        assert!((lon - (114.0 + 10.0 / 60.0)).abs() < 1e-6, "lon {lon}");

        let path = temp_file("tiff-fields.tif", &bytes);
        assert_eq!(read_exif(&path), ex);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn tiff_exif_reads_big_endian_and_long_iso() {
        // Big-endian IFD0 with the camera tags, plus ISO as LONG (type 4) in
        // the ExifIFD: both appear in files written by Adobe tools.
        let mut out = Vec::new();
        out.extend(b"MM\0*");
        out.extend([0u8; 4]);
        let exif_at = push_ifd(
            &mut out,
            false,
            &[
                long_ent(false, 0x8827, 640),
                ascii_ent(0xA434, "Sonnar T* 1.8/55 ZA"),
            ],
            0,
        );
        let ifd0 = push_ifd(
            &mut out,
            false,
            &[
                ascii_ent(0x010F, "Adobe Systems Inc."),
                ascii_ent(0x0110, "Tiff File"),
                long_ent(false, 0x8769, exif_at as u32),
            ],
            0,
        );
        out[4..8].copy_from_slice(&enc32(false, ifd0 as u32));

        let ex = parse_exif(&out);
        assert_eq!(ex.camera_make.as_deref(), Some("Adobe Systems Inc."));
        assert_eq!(ex.camera_model.as_deref(), Some("Tiff File"));
        assert_eq!(ex.lens.as_deref(), Some("Sonnar T* 1.8/55 ZA"));
        assert_eq!(ex.iso, Some(640));
    }

    #[test]
    fn tiff_exif_ifd0_past_the_head_is_still_read() {
        let bytes = build_tiff(true, 2 * HEAD_BYTES as usize);
        let path = temp_file("tiff-tail.tif", &bytes);
        let ex = read_exif(&path);
        assert_nikon_exif(&ex);
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn tiff_exif_truncated_or_corrupt_input_never_panics() {
        let full = build_tiff(true, 0);
        for cut in [0usize, 1, 3, 4, 7, 8, 9, 12, 20, 60, full.len() / 2, full.len() - 1] {
            // A partial IFD may yield partial fields; it must never panic or
            // invent values outside the bytes it was given.
            let _ = parse_exif(&full[..cut.min(full.len())]);
        }
        let mut bad_offset = full.clone();
        bad_offset[4..8].copy_from_slice(&enc32(true, u32::MAX));
        assert_eq!(parse_exif(&bad_offset), Exif::default());

        let mut bad_count = full.clone();
        let ifd0 = rd_u32(&full, 4, true).unwrap() as usize;
        bad_count[ifd0..ifd0 + 2].copy_from_slice(&enc16(true, u16::MAX));
        assert_eq!(parse_exif(&bad_count), Exif::default());

        let path = temp_file("tiff-cut.tif", &full[..6]);
        assert_eq!(read_exif(&path), Exif::default());
        std::fs::remove_file(&path).unwrap();
    }
}
