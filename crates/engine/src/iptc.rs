//! IPTC-IIM metadata from the Photoshop APP13 block.
//!
//! Photoshop stores IPTC as the 8BIM resource `0x0404`; each dataset inside it
//! is `0x1C`, record, dataset, big-endian length, value. Only record 2 (the
//! application record that carries captions, credits and keywords) is read.

use crate::catalog::Iptc;

const IPTC_RESOURCE: u16 = 0x0404;

/// Parse IPTC out of a JPEG, or directly out of a Photoshop APP13 payload.
pub fn parse_iptc(data: &[u8]) -> Iptc {
    let mut out = Iptc::default();
    if data.starts_with(b"Photoshop 3.0\0") {
        parse_app13(data, &mut out);
        return out;
    }
    if !data.starts_with(&[0xFF, 0xD8]) {
        return out;
    }

    let mut i = 2usize;
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
            0xDA | 0xD9 => break,
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
        if marker == 0xED {
            parse_app13(&data[start..end], &mut out);
        }
        i = end;
    }
    out
}

fn parse_app13(data: &[u8], out: &mut Iptc) {
    let mut p = if data.starts_with(b"Photoshop 3.0\0") { 14 } else { 0 };
    while p + 12 <= data.len() {
        if &data[p..p + 4] != b"8BIM" {
            break;
        }
        let id = u16::from_be_bytes([data[p + 4], data[p + 5]]);
        p += 6;

        // Pascal name, padded so the field is an even number of bytes.
        let name_len = data[p] as usize;
        p += 1 + name_len;
        if name_len % 2 == 0 {
            p += 1;
        }
        if p + 4 > data.len() {
            break;
        }
        let size = u32::from_be_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]]) as usize;
        p += 4;
        let end = match p.checked_add(size) {
            Some(e) if e <= data.len() => e,
            _ => break,
        };
        if id == IPTC_RESOURCE {
            parse_iim(&data[p..end], out);
        }
        p = end + (size & 1);
    }
}

fn parse_iim(data: &[u8], out: &mut Iptc) {
    let mut p = 0usize;
    while p + 5 <= data.len() {
        if data[p] != 0x1C {
            p += 1;
            continue;
        }
        let record = data[p + 1];
        let dataset = data[p + 2];
        let size = u16::from_be_bytes([data[p + 3], data[p + 4]]) as usize;
        if size & 0x8000 != 0 {
            break; // extended dataset length, not produced for these tags
        }
        p += 5;
        let value = match data.get(p..p + size) {
            Some(v) => v,
            None => break,
        };
        if record == 2 {
            apply(out, dataset, value);
        }
        p += size;
    }
}

fn apply(out: &mut Iptc, dataset: u8, value: &[u8]) {
    let text = decode(value);
    if dataset == 0x19 {
        if !text.is_empty() && !out.keywords.contains(&text) {
            out.keywords.push(text);
        }
        return;
    }
    let slot = match dataset {
        0x05 => &mut out.title,
        0x50 => &mut out.creator,
        0x5A => &mut out.city,
        0x65 => &mut out.country,
        0x67 => &mut out.job_id,
        0x69 => &mut out.headline,
        0x74 => &mut out.copyright,
        0x78 => &mut out.description,
        _ => return,
    };
    if !text.is_empty() {
        *slot = text;
    }
}

/// UTF-8 when the bytes are valid, otherwise Latin-1 (the IIM default).
fn decode(value: &[u8]) -> String {
    let value = match value.iter().rposition(|b| *b != 0) {
        Some(n) => &value[..=n],
        None => return String::new(),
    };
    let text = match std::str::from_utf8(value) {
        Ok(s) => s.to_string(),
        Err(_) => value.iter().map(|b| *b as char).collect(),
    };
    text.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ds(dataset: u8, value: impl AsRef<[u8]>) -> Vec<u8> {
        let value = value.as_ref();
        let mut out = vec![0x1C, 2, dataset];
        out.extend_from_slice(&(value.len() as u16).to_be_bytes());
        out.extend_from_slice(value);
        out
    }

    fn app13(iim: &[u8]) -> Vec<u8> {
        let mut payload = b"Photoshop 3.0\0".to_vec();
        payload.extend_from_slice(b"8BIM");
        payload.extend_from_slice(&IPTC_RESOURCE.to_be_bytes());
        payload.extend_from_slice(&[0, 0]);
        payload.extend_from_slice(&(iim.len() as u32).to_be_bytes());
        payload.extend_from_slice(iim);
        if iim.len() % 2 == 1 {
            payload.push(0);
        }
        payload
    }

    fn jpeg(payload: &[u8]) -> Vec<u8> {
        let mut out = vec![0xFF, 0xD8, 0xFF, 0xED];
        out.extend_from_slice(&((payload.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(payload);
        out.extend_from_slice(&[0xFF, 0xD9]);
        out
    }

    fn all_fields() -> Vec<u8> {
        let mut datasets = Vec::new();
        datasets.extend(ds(0x05, "Golden hour"));
        datasets.extend(ds(0x19, "wedding"));
        datasets.extend(ds(0x19, "outdoor"));
        datasets.extend(ds(0x50, "Ada Lovelace"));
        datasets.extend(ds(0x5A, "Hangzhou"));
        datasets.extend(ds(0x65, "China"));
        datasets.extend(ds(0x67, "JOB-42"));
        datasets.extend(ds(0x69, "Bridge at dusk"));
        datasets.extend(ds(0x74, "(c) 2026 Ada"));
        datasets.extend(ds(0x78, "A couple on the bridge."));
        datasets
    }

    #[test]
    fn reads_every_core_dataset() {
        let got = parse_iptc(&jpeg(&app13(&all_fields())));
        assert_eq!(got.title, "Golden hour");
        assert_eq!(got.creator, "Ada Lovelace");
        assert_eq!(got.city, "Hangzhou");
        assert_eq!(got.country, "China");
        assert_eq!(got.job_id, "JOB-42");
        assert_eq!(got.headline, "Bridge at dusk");
        assert_eq!(got.copyright, "(c) 2026 Ada");
        assert_eq!(got.description, "A couple on the bridge.");
        assert_eq!(got.keywords, ["wedding", "outdoor"]);
    }

    #[test]
    fn jpeg_without_app13_is_default() {
        let mut out = vec![0xFF, 0xD8, 0xFF, 0xD9];
        out.extend_from_slice(&[0u8; 64]);
        assert_eq!(parse_iptc(&out), Iptc::default());
        assert_eq!(parse_iptc(b"not a jpeg"), Iptc::default());
    }

    #[test]
    fn truncated_app13_does_not_panic() {
        let full = jpeg(&app13(&all_fields()));
        for cut in 4..full.len() {
            let _ = parse_iptc(&full[..cut]);
        }
        // Declared size past the block must stop, not read outside it.
        let mut broken = app13(&ds(0x05, "title"));
        let n = broken.len();
        broken[n - 8..n - 4].copy_from_slice(&0xFFFFu32.to_be_bytes());
        assert_eq!(parse_iptc(&jpeg(&broken)), Iptc::default());
    }

    #[test]
    fn utf8_and_latin1_values_decode() {
        let mut datasets = Vec::new();
        datasets.extend(ds(0x05, "西湖的清晨".as_bytes()));
        datasets.extend(ds(0x50, b"caf\xe9"));
        let got = parse_iptc(&jpeg(&app13(&datasets)));
        assert_eq!(got.title, "西湖的清晨");
        assert_eq!(got.creator, "café");
    }

    #[test]
    fn empty_values_are_ignored() {
        let mut datasets = Vec::new();
        datasets.extend(ds(0x05, b""));
        datasets.extend(ds(0x19, b""));
        datasets.extend(ds(0x74, b"\0\0"));
        assert_eq!(parse_iptc(&jpeg(&app13(&datasets))), Iptc::default());
    }

    #[test]
    fn record_one_is_ignored() {
        let mut datasets = vec![0x1C, 1, 0x05, 0, 5];
        datasets.extend_from_slice(b"model");
        datasets.extend(ds(0x05, "title"));
        assert_eq!(parse_iptc(&jpeg(&app13(&datasets))).title, "title");
    }
}
