//! Native (pure Rust) RAW front-end.
//!
//! The module parses the camera container itself and returns LibRaw's
//! [`crate::raw::BayerFrame`] shape, so everything downstream keeps working
//! unchanged.  It deliberately returns [`NativeError::Unsupported`] for formats
//! it cannot decode bit-exactly; `raw::decode_bayer16` then falls back to
//! LibRaw unless `AURORA_NO_LIBRAW=1` forbids it.
//!
//! Stage 1 formats: DNG (uncompressed, PackBits, deflate, lossless JPEG),
//! Nikon NEF (uncompressed, lossless/lossy Huffman, Z6III-family HE through the
//! existing C1 decoder), Sony ARW (ARW2 block compression and uncompressed) and
//! Canon CR2 (lossless JPEG with the CR2 slice table).

mod arw;
mod cr2;
mod cr3;
mod crx;
mod dng;
mod ljpeg;
mod meta;
mod nef;
mod orf;
mod packbits;
mod raf;
mod rw2;
mod tiff;

use std::path::Path;

use crate::raw::BayerFrame;

pub(crate) type NativeResult<T> = Result<T, NativeError>;

#[derive(Debug)]
pub(crate) enum NativeError {
    /// The container is understood but this decoder cannot reproduce it
    /// bit-exactly (unknown compression, unsupported sensor).
    Unsupported(String),
    /// The file claims a supported layout but its bytes are inconsistent.
    Corrupt(String),
    Io(String),
}

impl NativeError {
    pub(crate) fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported(message.into())
    }
    pub(crate) fn corrupt(message: impl Into<String>) -> Self {
        Self::Corrupt(message.into())
    }
    pub(crate) fn message(&self) -> &str {
        match self {
            Self::Unsupported(message) | Self::Corrupt(message) | Self::Io(message) => message,
        }
    }
}

impl std::fmt::Display for NativeError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.message())
    }
}

impl From<std::io::Error> for NativeError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.to_string())
    }
}

/// Decodes a RAW file into LibRaw's Bayer frame shape.
pub(crate) fn decode(path: &Path) -> NativeResult<BayerFrame> {
    let bytes = std::fs::read(path).map_err(NativeError::from)?;
    decode_bytes(&bytes)
}

pub(crate) fn decode_bytes(bytes: &[u8]) -> NativeResult<BayerFrame> {
    if bytes.get(4..8) == Some(b"ftyp") {
        return cr3::decode(bytes);
    }
    if bytes.starts_with(b"FUJIFILMCCD-RAW") {
        return raf::decode(bytes);
    }
    if bytes.starts_with(b"IIRO") || bytes.starts_with(b"IIRS") {
        return orf::decode(bytes);
    }
    if bytes.get(..4) == Some(b"IIU\0") {
        return rw2::decode(bytes);
    }

    let tiff = tiff::Tiff::parse(bytes, 0)?;
    let ifd0 = tiff.ifd(tiff.u32_at(4)?)?;
    let make = ifd0.ascii(0x010f).unwrap_or_default();
    let make_upper = make.to_ascii_uppercase();

    // CR2 keeps the "CR" signature at byte 8 and is a TIFF otherwise.
    if bytes.get(8..10) == Some(b"CR") {
        return cr2::decode(&tiff, &ifd0, bytes);
    }
    if ifd0.entry(0xc612).is_some() || make_upper.contains("DNG") {
        return dng::decode(&tiff, &ifd0);
    }
    if make_upper.contains("NIKON") {
        return nef::decode(&tiff, &ifd0, bytes);
    }
    if make_upper.contains("SONY") {
        return arw::decode(&tiff, &ifd0);
    }
    if make_upper.contains("CANON") {
        return cr2::decode(&tiff, &ifd0, bytes);
    }
    Err(NativeError::unsupported(format!("no native decoder for make {make:?}")))
}

/// Finds the raw sensor IFD among IFD0's SubIFDs and the `next` chain.
///
/// A raw IFD has NewSubfileType bit 0 clear and a CFA photometric value or a
/// full-resolution image; previews carry bit 0 and/or JPEG compression.
pub(crate) fn raw_ifd<'a>(tiff: &tiff::Tiff<'a>) -> NativeResult<tiff::Ifd<'a>> {
    fn score(ifd: &tiff::Ifd<'_>) -> i64 {
        let new_subfile = ifd.u32(0x00fe).unwrap_or(0);
        if new_subfile & 1 != 0 {
            return i64::MIN / 2;
        }
        let compression = ifd.u16(0x0103).unwrap_or(1);
        let photometric = ifd.u16(0x0106).unwrap_or(2);
        let width = i64::from(ifd.u32(0x0100).unwrap_or(0));
        let height = i64::from(ifd.u32(0x0101).unwrap_or(0));
        let mut value = width * height;
        if photometric == 32803 {
            value += 1_000_000_000_000;
        }
        if compression == 7 || compression == 1 || compression == 32773 || compression == 8 {
            value += 1_000_000;
        }
        value
    }

    fn collect<'a>(tiff: &tiff::Tiff<'a>, offset: u32, depth: u8, out: &mut Vec<tiff::Ifd<'a>>) {
        if depth > 2 || out.len() >= 64 {
            return;
        }
        let Ok(ifd) = tiff.ifd(offset) else {
            return;
        };
        if ifd.entry(0x0100).is_some() || ifd.entry(0x0142).is_some() {
            out.push(ifd.clone());
        }
        for sub in ifd.sub_ifds() {
            collect(tiff, sub, depth + 1, out);
        }
    }

    let mut candidates: Vec<tiff::Ifd<'a>> = Vec::new();
    for chain_ifd in tiff.chain(tiff.u32_at(4)?)? {
        collect(tiff, chain_ifd.offset, 0, &mut candidates);
        for sub in chain_ifd.sub_ifds() {
            collect(tiff, sub, 1, &mut candidates);
        }
    }
    candidates
        .into_iter()
        .max_by_key(score)
        .filter(|ifd| score(ifd) > 0)
        .ok_or_else(|| NativeError::corrupt("no raw sensor IFD found"))
}

/// Reads the CFA pattern of an IFD into LibRaw's filter word.
pub(crate) fn cfa_filters(ifd: &tiff::Ifd<'_>) -> NativeResult<(u32, u32)> {
    let (filters, colors) = cfa_filters_raw(ifd)?;
    Ok((meta::upgrade_filters(filters, colors), colors))
}

fn cfa_filters_raw(ifd: &tiff::Ifd<'_>) -> NativeResult<(u32, u32)> {
    let repeat = ifd.u16s(0x828d).and_then(|values| values.ok()).and_then(|values| {
        (values.len() >= 2).then(|| (values[0], values[1]))
    });
    let pattern = match ifd.tag_bytes(0x828e) {
        Some(Ok(bytes)) => bytes.to_vec(),
        _ => Vec::new(),
    };
    let plane_colors = match ifd.tag_bytes(0xc616) {
        Some(Ok(bytes)) => Some(bytes.to_vec()),
        _ => None,
    };
    if pattern.is_empty() {
        // DNG always declares a pattern; a 2x2 RGGB origin is the only safe
        // assumption for legacy TIFF raws that omit it.
        let (filters, colors) = meta::filters_from_cfa(repeat, &[0, 1, 1, 2], plane_colors.as_deref())?;
        return Ok((filters, colors));
    }
    meta::filters_from_cfa(repeat, &pattern, plane_colors.as_deref())
}

/// Geometry of the active area, derived from DNG ActiveArea when present.
pub(crate) fn active_area(ifd: &tiff::Ifd<'_>, raw_width: usize, raw_height: usize) -> (usize, usize, usize, usize) {
    if let Some(Ok(area)) = ifd.u32s(0xc68d) {
        if area.len() >= 4 {
            let top = area[0] as usize;
            let left = area[1] as usize;
            let bottom = area[2] as usize;
            let right = area[3] as usize;
            if bottom > top && right > left && right <= raw_width && bottom <= raw_height {
                return (left, top, right - left, bottom - top);
            }
        }
    }
    (0, 0, raw_width, raw_height)
}

/// Assembles the LibRaw-compatible frame. `data` must be `raw_width * raw_height`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn make_frame(
    raw_width: usize,
    raw_height: usize,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
    bits: u8,
    data: Vec<u16>,
    filters: u32,
    colors: u32,
    black: u32,
    maximum: u32,
    cam_mul: [f32; 4],
    rgb_cam: [[f32; 4]; 3],
) -> NativeResult<BayerFrame> {
    if raw_width == 0
        || raw_height == 0
        || data.len() != raw_width * raw_height
        || left + width > raw_width
        || top + height > raw_height
        || width == 0
        || height == 0
    {
        return Err(NativeError::corrupt("decoded sensor plane does not fit its declared geometry"));
    }
    Ok(BayerFrame {
        raw_width,
        raw_height,
        width,
        height,
        top_margin: top,
        left_margin: left,
        stride: raw_width * 2,
        bits,
        data,
        filters,
        colors,
        black,
        maximum,
        cam_mul,
        rgb_cam,
    })
}
