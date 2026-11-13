//! Minimal read-only metadata probe for Capture One `.comask` mask sidecars.
//!
//! The container starts with `\x04XCon` and contains repeated `MaskIndex`
//! records. Each record names a mask UUID, a numeric `MaskType`, optional
//! `MaskGenerator` JSON, `CompParam` floats and, for gradients, a
//! `LinearGradient` parameter block. Pixel/raster payloads are deliberately not
//! decoded: [`ComaskDocument::decode_pixels`] returns an explicit unsupported
//! error instead of inventing mask pixels.

use serde_json::{Map, Value};
use std::fmt;

const MAGIC: &[u8] = b"\x04XCon";
const MASK_INDEX: &[u8] = b"MaskIndex";
const MASK_TYPE: &[u8] = b"MaskType";
const COMP_PARAM: &[u8] = b"CompParam";
const LINEAR_GRADIENT: &[u8] = b"LinearGradient";
const RAND_SEED: &[u8] = b"RandSeed";

#[derive(Clone, Debug, PartialEq)]
pub struct ComaskDocument {
    /// Known section names found in the container, in probe order.
    pub sections: Vec<String>,
    pub masks: Vec<ComaskMask>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComaskMask {
    /// Four-byte little-endian value following `MaskIndex`.
    pub index: u32,
    /// UUID from `UserInfo` (`ZMASKUUID` in `ZVARIANTLAYER`).
    pub uuid: Option<String>,
    /// Raw numeric `MaskType`; 3 is observed on the linear-gradient sample.
    pub mask_type: Option<i32>,
    /// `MaskGenerator` from the embedded JSON, e.g. `AISkin`.
    pub generator: Option<String>,
    /// Structured form of the same JSON for known `MaskGenerator` records.
    pub generator_options: Option<ComaskGenerator>,
    /// Floats from `CompParam`; the first four bytes are a count/version word.
    pub comp_params: Vec<f32>,
    /// Five floats from `LinearGradient`.
    pub linear_gradient: Option<[f32; 5]>,
    /// Four-byte value following `RandSeed`, when present.
    pub rand_seed: Option<u32>,
    /// Opaque raster/index section spans; pixels are not decoded.
    pub rasters: Vec<ComaskRaster>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RasterKind {
    Map,
    Blk,
    Idx,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ComaskRaster {
    pub kind: RasterKind,
    /// Offset in the whole `.comask` file.
    pub offset: usize,
    /// Section span from its name through the next raster section or mask end.
    pub span_len: usize,
    /// Raw bytes immediately following the section name, for probe output.
    pub header: Vec<u8>,
    /// `Blk`'s nested `Hdr` value, when present.
    pub blocks: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ComaskSkinOptions {
    pub body: bool,
    pub face: bool,
    pub eyes: bool,
    pub eyebrows: bool,
    pub lips: bool,
    pub hair: bool,
    pub iris_and_pupil: bool,
    pub sclera: bool,
    pub clothes: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComaskGenerator {
    pub name: String,
    pub skin: Option<ComaskSkinOptions>,
    pub selected_submask_index: Option<i64>,
    pub submask_name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RasterProbe {
    pub zlib: bool,
    pub gzip: bool,
    pub lz4_frame: bool,
    pub unique_bytes: usize,
    /// Most frequent byte values, highest first.
    pub top: Vec<(u8, usize)>,
}

/// Checks only signatures and a byte histogram; it never decompresses or
/// interprets samples as pixels.
pub fn probe_raster(bytes: &[u8]) -> RasterProbe {
    let zlib = bytes.len() >= 2 && bytes[0] == 0x78 && matches!(bytes[1], 0x01 | 0x9c | 0xda);
    let gzip = bytes.starts_with(&[0x1f, 0x8b]);
    let lz4_frame = bytes.starts_with(&[0x04, 0x22, 0x4d, 0x18]);
    let mut counts = [0usize; 256];
    for &b in bytes {
        counts[b as usize] += 1;
    }
    let unique_bytes = counts.iter().filter(|&&n| n > 0).count();
    let mut top: Vec<(u8, usize)> = counts
        .iter()
        .enumerate()
        .filter_map(|(i, &n)| (n > 0).then_some((i as u8, n)))
        .collect();
    top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    top.truncate(8);
    RasterProbe { zlib, gzip, lz4_frame, unique_bytes, top }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComaskError {
    NotContainer,
    NoMasks,
    Truncated(&'static str),
    BadGeneratorJson,
    UnsupportedRaster,
}

impl fmt::Display for ComaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotContainer => write!(f, "not a Capture One .comask container"),
            Self::NoMasks => write!(f, "no MaskIndex records"),
            Self::Truncated(field) => write!(f, "truncated {field}"),
            Self::BadGeneratorJson => write!(f, "invalid MaskGenerator JSON"),
            Self::UnsupportedRaster => {
                write!(f, "compressed/raster mask pixels are not decoded")
            }
        }
    }
}

impl std::error::Error for ComaskError {}

impl ComaskDocument {
    /// Reads the metadata records without touching the raster payload.
    pub fn parse(bytes: &[u8]) -> Result<Self, ComaskError> {
        if !bytes.starts_with(MAGIC) {
            return Err(ComaskError::NotContainer);
        }
        let starts = find_all(bytes, MASK_INDEX);
        if starts.is_empty() {
            return Err(ComaskError::NoMasks);
        }
        let mut masks = Vec::with_capacity(starts.len());
        for (i, &start) in starts.iter().enumerate() {
            let end = starts.get(i + 1).copied().unwrap_or(bytes.len());
            let segment = &bytes[start..end];
            let index = mask_index(segment)?;
            let uuid = find_uuid(segment);
            let mask_type = mask_type(segment)?;
            let (generator, generator_options) = generator(segment)?;
            let comp_params = comp_params(segment)?;
            let linear_gradient = linear_gradient(segment)?;
            let rand_seed = rand_seed(segment);
            let rasters = raster_spans(segment, start);
            masks.push(ComaskMask {
                index,
                uuid,
                mask_type,
                generator,
                generator_options,
                comp_params,
                linear_gradient,
                rand_seed,
                rasters,
            });
        }
        let sections = SECTIONS
            .iter()
            .filter(|name| find(bytes, name.as_bytes()).is_some())
            .map(|name| (*name).to_owned())
            .collect();
        Ok(Self { sections, masks })
    }

    pub fn mask(&self, index: u32) -> Option<&ComaskMask> {
        self.masks.iter().find(|m| m.index == index)
    }

    /// Deliberate boundary: the raster format is not implemented yet.
    pub fn decode_pixels(&self, _index: u32) -> Result<(), ComaskError> {
        Err(ComaskError::UnsupportedRaster)
    }
}

fn raster_spans(segment: &[u8], base: usize) -> Vec<ComaskRaster> {
    let mut hits: Vec<(usize, RasterKind)> = Vec::new();
    for (name, kind) in [(b"Map".as_slice(), RasterKind::Map), (b"Blk", RasterKind::Blk), (b"Idx", RasterKind::Idx)] {
        if let Some(at) = find(segment, name) {
            hits.push((at, kind));
        }
    }
    hits.sort_by_key(|(at, _)| *at);
    hits.iter()
        .enumerate()
        .map(|(i, &(at, kind))| {
            let end = hits.get(i + 1).map(|(next, _)| *next).unwrap_or(segment.len());
            let header_start = (at + 3).min(end);
            let header_end = (header_start + 8).min(end);
            let blocks = if kind == RasterKind::Blk { block_count(segment, at, end) } else { None };
            ComaskRaster {
                kind,
                offset: base + at,
                span_len: end - at,
                header: segment[header_start..header_end].to_vec(),
                blocks,
            }
        })
        .collect()
}

fn block_count(segment: &[u8], start: usize, end: usize) -> Option<u32> {
    let hdr = find(&segment[start..end], b"Hdr")? + start;
    let at = hdr + b"Hdr".len();
    (segment.get(at..at + 4) == Some(&[0x00, 0x22, 0xff, 0x10]))
        .then(|| u32_at(segment, at + 4))
        .flatten()
}

const SECTIONS: &[&str] = &[
    "TYPE",
    "VER",
    "MaskInfo",
    "MaskIndex",
    "UserInfo",
    "Info",
    "Name",
    "MaskType",
    "CompParam",
    "LinearGradient",
    "RandSeed",
    "Map",
    "Blk",
    "Idx",
    "Hdr",
];

fn find(bytes: &[u8], needle: &[u8]) -> Option<usize> {
    bytes.windows(needle.len()).position(|w| w == needle)
}

fn find_all(bytes: &[u8], needle: &[u8]) -> Vec<usize> {
    let mut out = Vec::new();
    let mut at = 0;
    while let Some(offset) = bytes.get(at..).and_then(|rest| find(rest, needle)) {
        let hit = at + offset;
        out.push(hit);
        at = hit + needle.len();
    }
    out
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    bytes.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn i32_at(bytes: &[u8], at: usize) -> Option<i32> {
    bytes.get(at..at + 4).map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn f32_at(bytes: &[u8], at: usize) -> Option<f32> {
    bytes.get(at..at + 4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn floats_at<const N: usize>(bytes: &[u8], at: usize) -> Option<[f32; N]> {
    let mut out = [0.0; N];
    for (i, value) in out.iter_mut().enumerate() {
        *value = f32_at(bytes, at + i * 4)?;
    }
    Some(out)
}

fn mask_index(segment: &[u8]) -> Result<u32, ComaskError> {
    let at = MASK_INDEX.len();
    if segment.get(at..at + 6) != Some(&[0x00, 0x22, 0xff, 0xff, 0xff, 0x04]) {
        return Err(ComaskError::Truncated("MaskIndex"));
    }
    u32_at(segment, at + 6).ok_or(ComaskError::Truncated("MaskIndex value"))
}

fn mask_type(segment: &[u8]) -> Result<Option<i32>, ComaskError> {
    let Some(at) = find(segment, MASK_TYPE) else { return Ok(None) };
    let at = at + MASK_TYPE.len();
    if segment.get(at..at + 3) != Some(&[0x00, 0x22, 0x04]) {
        return Err(ComaskError::Truncated("MaskType"));
    }
    i32_at(segment, at + 3).map(Some).ok_or(ComaskError::Truncated("MaskType value"))
}

fn comp_params(segment: &[u8]) -> Result<Vec<f32>, ComaskError> {
    let Some(at) = find(segment, COMP_PARAM) else { return Ok(Vec::new()) };
    let at = at + COMP_PARAM.len();
    if segment.get(at..at + 6) != Some(&[0x00, 0x22, 0xff, 0xff, 0xff, 0x10]) {
        return Err(ComaskError::Truncated("CompParam"));
    }
    u32_at(segment, at + 6).ok_or(ComaskError::Truncated("CompParam header"))?;
    let values = floats_at::<3>(segment, at + 10)
        .ok_or(ComaskError::Truncated("CompParam value"))?;
    Ok(values.to_vec())
}

fn linear_gradient(segment: &[u8]) -> Result<Option<[f32; 5]>, ComaskError> {
    let Some(at) = find(segment, LINEAR_GRADIENT) else { return Ok(None) };
    let at = at + LINEAR_GRADIENT.len();
    if segment.get(at..at + 5) != Some(&[0x00, 0x22, 0xff, 0xff, 0x14]) {
        return Err(ComaskError::Truncated("LinearGradient"));
    }
    floats_at::<5>(segment, at + 5)
        .map(Some)
        .ok_or(ComaskError::Truncated("LinearGradient value"))
}

fn rand_seed(segment: &[u8]) -> Option<u32> {
    let at = find(segment, RAND_SEED)? + RAND_SEED.len();
    (segment.get(at..at + 3) == Some(&[0x00, 0x22, 0x08]))
        .then(|| u32_at(segment, at + 3))
        .flatten()
}

fn generator(segment: &[u8]) -> Result<(Option<String>, Option<ComaskGenerator>), ComaskError> {
    let needle = b"{\"MaskGenerator\"";
    let Some(start) = find(segment, needle) else { return Ok((None, None)) };
    let end = balanced_json_end(segment, start).ok_or(ComaskError::BadGeneratorJson)?;
    let value: Value =
        serde_json::from_slice(&segment[start..=end]).map_err(|_| ComaskError::BadGeneratorJson)?;
    let Some(name) = value.get("MaskGenerator").and_then(Value::as_str).map(str::to_owned) else {
        return Ok((None, None));
    };
    let skin = value
        .get("SkinOptions")
        .and_then(Value::as_object)
        .map(|object| ComaskSkinOptions {
            body: bool_field(object, "Body"),
            face: bool_field(object, "Face"),
            eyes: bool_field(object, "Eyes"),
            eyebrows: bool_field(object, "Eyebrows"),
            lips: bool_field(object, "Lips"),
            hair: bool_field(object, "Hair"),
            iris_and_pupil: bool_field(object, "IrisAndPupil"),
            sclera: bool_field(object, "Sclera"),
            clothes: bool_field(object, "Clothes"),
        });
    let selected_submask_index = value.get("SelectedSubmaskIndex").and_then(Value::as_i64);
    let submask_name = value.get("SubmaskName").and_then(Value::as_str).map(str::to_owned);
    Ok((
        Some(name.clone()),
        Some(ComaskGenerator { name, skin, selected_submask_index, submask_name }),
    ))
}

fn bool_field(object: &Map<String, Value>, key: &str) -> bool {
    object.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn balanced_json_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn find_uuid(bytes: &[u8]) -> Option<String> {
    for i in 0..bytes.len().saturating_sub(35) {
        let s = &bytes[i..i + 36];
        if uuid_shape(s) {
            return Some(std::str::from_utf8(s).ok()?.to_owned());
        }
    }
    None
}

fn uuid_shape(s: &[u8]) -> bool {
    if s[8] != b'-' || s[13] != b'-' || s[18] != b'-' || s[23] != b'-' {
        return false;
    }
    s.iter().enumerate().all(|(i, b)| {
        matches!(i, 8 | 13 | 18 | 23) || b.is_ascii_hexdigit()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push(buf: &mut Vec<u8>, bytes: &[u8]) {
        buf.extend_from_slice(bytes);
    }

    fn mask(index: u32, uuid: &str, kind: i32, generator: Option<&str>, gradient: Option<[f32; 5]>) -> Vec<u8> {
        let mut b = Vec::new();
        push(&mut b, b"MaskIndex\x00\x22\xff\xff\xff\x04");
        push(&mut b, &index.to_le_bytes());
        push(&mut b, b"UserInfo\x00\x22\x00\x04Info\x00\x22\x24");
        push(&mut b, uuid.as_bytes());
        if let Some(json) = generator {
            push(&mut b, b"Name\x00\x22");
            b.push(json.len() as u8);
            push(&mut b, json.as_bytes());
        }
        push(&mut b, b"MaskType\x00\x22\x04");
        push(&mut b, &kind.to_le_bytes());
        push(&mut b, b"CompParam\x00\x22\xff\xff\xff\x10");
        push(&mut b, &1u32.to_le_bytes());
        for v in [1.0f32, 0.0, 0.0] {
            push(&mut b, &v.to_le_bytes());
        }
        if let Some(values) = gradient {
            push(&mut b, b"LinearGradient\x00\x22\xff\xff\x14");
            for v in values {
                push(&mut b, &v.to_le_bytes());
            }
        }
        push(&mut b, b"RandSeed\x00\x22\x08\x01\x00\x00\x00");
        b
    }

    fn fixture() -> Vec<u8> {
        let mut b = b"\x04XCon".to_vec();
        push(&mut b, b"TYPE\x00\x0fCaptureOne Mask");
        push(
            &mut b,
            &mask(
                0,
                "00000000-0000-4000-8000-000000000001",
                0,
                Some(
                    "{\"MaskGenerator\":\"AISkin\",\"SkinOptions\":{\"Body\":true,\"Face\":true,\"Eyes\":false,\"Eyebrows\":false,\"Lips\":false,\"Hair\":false,\"IrisAndPupil\":false,\"Sclera\":false,\"Clothes\":false},\"SelectedSubmaskIndex\":null,\"SubmaskName\":null}",
                ),
                None,
            ),
        );
        push(
            &mut b,
            &mask(
                3,
                "00000000-0000-4000-8000-000000000002",
                3,
                None,
                Some([1.0, 2.0, 3.0, 4.0, 0.5]),
            ),
        );
        b
    }

    #[test]
    fn reads_index_uuid_type_generator_and_gradient() {
        let doc = ComaskDocument::parse(&fixture()).unwrap();
        assert!(doc.sections.iter().any(|name| name == "MaskIndex"));
        assert_eq!(doc.masks.len(), 2);
        assert_eq!(doc.masks[0].index, 0);
        assert_eq!(
            doc.masks[0].uuid.as_deref(),
            Some("00000000-0000-4000-8000-000000000001")
        );
        assert_eq!(doc.masks[0].mask_type, Some(0));
        assert_eq!(doc.masks[0].generator.as_deref(), Some("AISkin"));
        let generator = doc.masks[0].generator_options.as_ref().unwrap();
        let skin = generator.skin.as_ref().unwrap();
        assert_eq!((skin.body, skin.face, skin.eyes), (true, true, false));
        assert_eq!(generator.selected_submask_index, None);
        assert_eq!(generator.submask_name, None);
        assert_eq!(doc.masks[0].comp_params, vec![1.0, 0.0, 0.0]);
        assert_eq!(doc.masks[0].rand_seed, Some(1));
        assert_eq!(doc.masks[1].index, 3);
        assert_eq!(doc.masks[1].mask_type, Some(3));
        assert_eq!(doc.masks[1].linear_gradient, Some([1.0, 2.0, 3.0, 4.0, 0.5]));
        assert_eq!(doc.mask(3).unwrap().index, 3);
    }

    #[test]
    fn rejects_non_container_and_truncated_records() {
        assert_eq!(ComaskDocument::parse(b"not a mask"), Err(ComaskError::NotContainer));
        assert_eq!(ComaskDocument::parse(b"\x04XCon"), Err(ComaskError::NoMasks));
        let mut broken = b"\x04XCon".to_vec();
        push(&mut broken, b"MaskIndex\x00\x22\xff\xff\xff\x04");
        push(&mut broken, &[0, 0]);
        assert_eq!(
            ComaskDocument::parse(&broken),
            Err(ComaskError::Truncated("MaskIndex value"))
        );
    }

    #[test]
    fn raster_pixels_are_explicitly_unsupported() {
        let doc = ComaskDocument::parse(&fixture()).unwrap();
        assert_eq!(doc.decode_pixels(0), Err(ComaskError::UnsupportedRaster));
    }

    #[test]
    fn raster_sections_expose_spans_and_block_header() {
        let mut b = b"\x04XCon".to_vec();
        push(&mut b, b"MaskIndex\x00\x22\xff\xff\xff\x04");
        push(&mut b, &0u32.to_le_bytes());
        push(&mut b, b"Map\x00\x00\x41\x1e\x49");
        push(&mut b, &[1u8; 12]);
        push(&mut b, b"Blk\x00\x22\x00\xff\x03Hdr\x00\x22\xff\x10");
        push(&mut b, &0x4DEu32.to_le_bytes());
        push(&mut b, &[0u8; 8]);
        push(&mut b, b"Idx\x00\x22");
        push(&mut b, &[1u8, 2, 3, 4]);
        let doc = ComaskDocument::parse(&b).unwrap();
        let rasters = &doc.masks[0].rasters;
        assert_eq!(rasters.len(), 3);
        assert_eq!(rasters[0].kind, RasterKind::Map);
        assert_eq!(rasters[1].kind, RasterKind::Blk);
        assert_eq!(rasters[1].blocks, Some(0x4DE));
        assert_eq!(rasters[2].kind, RasterKind::Idx);
        assert!(rasters.iter().all(|r| r.span_len > 0));
    }

    #[test]
    fn raster_probe_detects_headers_and_raw_histogram() {
        let raw = probe_raster(&[1, 1, 8, 0, 255]);
        assert!(!raw.zlib && !raw.gzip && !raw.lz4_frame);
        assert_eq!(raw.unique_bytes, 4);
        assert_eq!(raw.top[0], (1, 2));
        let zlib = probe_raster(&[0x78, 0x9c, 0, 0]);
        assert!(zlib.zlib);
        let lz4 = probe_raster(&[0x04, 0x22, 0x4d, 0x18]);
        assert!(lz4.lz4_frame);
    }
}
