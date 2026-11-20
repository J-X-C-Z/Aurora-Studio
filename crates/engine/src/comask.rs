//! Read-only decoder for Capture One `.comask` mask sidecars.
//!
//! The container starts with `\x04XCon` and contains repeated `MaskIndex`
//! records. Each record names a mask UUID, a numeric `MaskType`, optional
//! `MaskGenerator` JSON, `CompParam` floats and, for gradients, a
//! `LinearGradient` parameter block. Masks that store pixels carry a `Map`
//! grid (one byte per 64×64 tile) plus a shared `Blk`/`Idx` block table;
//! [`read_comask_masks`] turns that into full-canvas planes.
//!
//! `ComaskMask::uuid` is the `UserInfo/Info` 36-character UUID that
//! `ZVARIANTLAYER.ZMASKUUID` (or a `ZRETOUCHINGLAYER.*MASKUUID` column) stores;
//! `c1_masks::C1MaskLayer::comask_metadata` performs that join. The numeric
//! `MaskIndex` is only the record ordinal. Repair layers additionally carry a
//! `Cover` block with heal/clone arrows ([`ComaskCover`]). Nothing in this
//! module is wired into the render path.

use serde_json::{Map, Value};
use std::fmt;

use anyhow::{bail, Result as AnyResult};

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
    /// `Cover` block of a repair layer: heal/clone arrows, one per stroke.
    pub cover: Option<ComaskCover>,
    /// Opaque raster/index section spans; pixels are not decoded.
    pub rasters: Vec<ComaskRaster>,
}

/// A `Cover` chunk: `Hdr = u16 version, u16 mask_index, u32 cover_type,
/// u32 count`, followed by `count` pairs of 24-byte `Vec` + `Data`.
#[derive(Clone, Debug, PartialEq)]
pub struct ComaskCover {
    pub version: u16,
    pub mask_index: u16,
    /// `IC_MaskCoverType`: 1 clone, 2 heal.
    pub cover_type: u32,
    pub count: u32,
    pub arrows: Vec<ComaskArrow>,
}

/// One 24-byte `Cover/Vec`: `u32 type, f32 src.x, src.y, f32 dst.x, dst.y,
/// u32 flags`. Source and destination are raw image pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ComaskArrow {
    /// `IC_MaskCoverArrowType`: 1 clone, 2 heal.
    pub arrow_type: u32,
    pub src: [f32; 2],
    pub dst: [f32; 2],
    /// `IC_MaskCoverArrowFlags` bits (`Approximate=2`, `AutoPlacement=8`, ...).
    pub flags: u32,
    /// Byte length of the `Data` chunk that follows this `Vec`.
    pub data_len: usize,
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
    /// `Map` grid or block table is inconsistent (bad cell, block overrun).
    BadMap,
    /// The mask record carries no `Map` grid, so it has no stored pixels.
    NoRaster,
}

impl fmt::Display for ComaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotContainer => write!(f, "not a Capture One .comask container"),
            Self::NoMasks => write!(f, "no MaskIndex records"),
            Self::Truncated(field) => write!(f, "truncated {field}"),
            Self::BadGeneratorJson => write!(f, "invalid MaskGenerator JSON"),
            Self::BadMap => write!(f, "inconsistent .comask Map/block data"),
            Self::NoRaster => write!(f, "mask record stores no Map raster"),
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
            let cover = cover(segment);
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
                cover,
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

    /// Decodes the plane of mask `index` straight from the container bytes.
    ///
    /// `ComaskError::NoRaster` means the mask record carries no `Map` grid
    /// (empty layer or a mask generated from parameters such as a gradient).
    pub fn decode_pixels(&self, bytes: &[u8], index: u32) -> Result<MaskPlane, ComaskError> {
        if self.mask(index).is_none() {
            return Err(ComaskError::NoRaster);
        }
        let planes = decode_masks(bytes, None).map_err(|_| ComaskError::BadMap)?;
        planes.into_iter().find(|p| p.mask_index == index).ok_or(ComaskError::NoRaster)
    }
}

/// One decoded mask, normalized to `0.0 ..= 1.0`.
///
/// Pixels come from the mask's `Map` grid: `8`/`9` cells consume the next block
/// in raster order, `255`/`0` are constant fills and `1` repeats the previous
/// fill. Blocks are 64×64 tiles; the last tile column only covers the pixels
/// that exist in the canvas (6048 = 94.5 × 64 here).
#[derive(Clone, Debug, PartialEq)]
pub struct MaskPlane {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
    /// `MaskIndex` value of the mask record (0..=7 in the sample).
    pub mask_index: u32,
    /// UUID from the record's `UserInfo` (`ZMASKUUID` on the catalog side).
    pub uuid: Option<String>,
    /// `LinearGradient` parameters behind a parameter-only plane, midpoint
    /// included. Pixels still come from `smoothstep`; a midpoint other than
    /// 0.5 is carried but not reproduced (no C1 reference for that ramp yet).
    pub linear_gradient: Option<[f32; 5]>,
    /// Index of this mask's first block inside the shared `Blk` table.
    pub first_block: usize,
    /// Number of `8`/`9` cells, i.e. blocks belonging to this mask.
    pub blocks: usize,
    /// Blocks decoded into the plane (equals `blocks` unless something failed).
    pub decoded_blocks: usize,
    pub failed_blocks: usize,
    /// Fraction of the full canvas above zero.
    pub coverage: f32,
    pub tile_columns: usize,
    pub tile_rows: usize,
}

impl MaskPlane {
    /// 8-bit alpha plane, the form Capture One's `IC_MaskSampleInt` returns.
    pub fn to_u8(&self) -> Vec<u8> {
        self.data.iter().map(|v| (v * 255.0).round() as u8).collect()
    }

    /// Inclusive `(x0, y0, x1, y1)` of non-zero pixels.
    pub fn bbox(&self) -> Option<(usize, usize, usize, usize)> {
        let mut bounds: Option<(usize, usize, usize, usize)> = None;
        for y in 0..self.height {
            for x in 0..self.width {
                if self.data[y * self.width + x] <= 0.0 {
                    continue;
                }
                bounds = Some(match bounds {
                    None => (x, y, x, y),
                    Some((x0, y0, x1, y1)) => (x0.min(x), y0.min(y), x1.max(x), y1.max(y)),
                });
            }
        }
        bounds
    }
}

fn le_u16_at(bytes: &[u8], at: usize) -> Option<u16> {
    bytes.get(at..at + 2).map(|b| u16::from_le_bytes([b[0], b[1]]))
}

fn le_u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    bytes.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn find_from(bytes: &[u8], needle: &[u8], start: usize) -> Option<usize> {
    bytes.get(start..)?.windows(needle.len()).position(|w| w == needle).map(|i| start + i)
}

/// Reads the big-endian base-64 varint used by the XCon chunks. `0xFF` bytes
/// are padding; a first byte below `0x40` (or above `0x7F`) is the whole value,
/// otherwise bit 6 marks the final group.
fn read_varint(bytes: &[u8], mut i: usize) -> Option<(usize, usize)> {
    while *bytes.get(i)? == 0xFF {
        i += 1;
    }
    let first = *bytes.get(i)?;
    i += 1;
    if first < 0x40 || first >= 0x80 {
        return Some((first as usize, i));
    }
    let mut value = (first & 0x3F) as usize;
    for _ in 0..9 {
        let byte = *bytes.get(i)?;
        i += 1;
        value = (value << 6) | (byte & 0x3F) as usize;
        if byte & 0x40 != 0 {
            return Some((value, i));
        }
    }
    None
}

/// Payload of the chunk whose name starts at `at`, skipping the two flag bytes
/// and the size varint.
fn chunk_payload_at<'a>(bytes: &'a [u8], at: usize, name_len: usize) -> Option<(&'a [u8], usize)> {
    let size_at = at.checked_add(name_len)?.checked_add(2)?;
    let (size, start) = read_varint(bytes, size_at)?;
    let end = start.checked_add(size)?;
    Some((bytes.get(start..end)?, end))
}

/// Payload of a `Hdr` field: `Hdr 00 22 FF <len>` followed by `<len>` bytes.
fn hdr_payload(bytes: &[u8], at: usize) -> Option<&[u8]> {
    if bytes.get(at..at + 3)? != b"Hdr" {
        return None;
    }
    let marker = bytes.get(at + 3..at + 7)?;
    if marker[0] != 0x00 || marker[1] != 0x22 || marker[2] != 0xFF {
        return None;
    }
    let len = marker[3] as usize;
    bytes.get(at + 7..at + 7 + len)
}

/// Parses the `Cover` chunk of a heal/clone repair layer.
///
/// The header is `u16 version, u16 mask_index, u32 cover_type, u32 count`;
/// each arrow is a 24-byte `Vec` followed by an opaque `Data` stroke record.
/// A truncated arrow stops the scan but still returns what was decoded.
fn cover(segment: &[u8]) -> Option<ComaskCover> {
    let at = find(segment, b"Cover")?;
    let hdr_at = find_from(segment, b"Hdr", at)?;
    let (hdr, mut cursor) = chunk_payload_at(segment, hdr_at, 3)?;
    if hdr.len() < 12 {
        return None;
    }
    let version = u16::from_le_bytes([hdr[0], hdr[1]]);
    let mask_index = u16::from_le_bytes([hdr[2], hdr[3]]);
    let cover_type = u32_at(hdr, 4)?;
    let count = u32_at(hdr, 8)?;
    let mut arrows = Vec::new();
    while arrows.len() < count as usize {
        let Some(vec_at) = find_from(segment, b"Vec", cursor) else { break };
        let Some((vec, vec_end)) = chunk_payload_at(segment, vec_at, 3) else { break };
        if vec.len() < 24 {
            break;
        }
        let Some(arrow_type) = u32_at(vec, 0) else { break };
        let Some(src) = floats_at::<2>(vec, 4) else { break };
        let Some(dst) = floats_at::<2>(vec, 12) else { break };
        let Some(flags) = u32_at(vec, 20) else { break };
        let data = find_from(segment, b"Data", vec_end)
            .and_then(|data_at| chunk_payload_at(segment, data_at, 4));
        let data_len = data.map(|(payload, _)| payload.len()).unwrap_or(0);
        arrows.push(ComaskArrow { arrow_type, src, dst, flags, data_len });
        cursor = data.map(|(_, end)| end).unwrap_or(vec_end);
    }
    Some(ComaskCover { version, mask_index, cover_type, count, arrows })
}

/// Decodes every mask that stores a `Map` grid into a full-canvas plane.
///
/// Masks without a `Map` (empty layers, parameter-only masks such as a linear
/// gradient) are skipped; use [`ComaskDocument::masks`] to see all records.
pub fn read_comask_masks(bytes: &[u8]) -> AnyResult<Vec<MaskPlane>> {
    decode_masks(bytes, None)
}

/// Decodes only the maps whose `UserInfo` UUID is in `wanted`.
pub fn read_comask_masks_for(bytes: &[u8], wanted: &[&str]) -> AnyResult<Vec<MaskPlane>> {
    decode_masks(bytes, Some(wanted))
}

/// First decoded mask plane (mask record 0 in the sample) — debug entry point.
pub fn read_comask_bytes(bytes: &[u8]) -> AnyResult<MaskPlane> {
    decode_masks(bytes, None)?
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no mask record stores a Map raster"))
}

fn decode_masks(bytes: &[u8], wanted: Option<&[&str]>) -> AnyResult<Vec<MaskPlane>> {
    if !bytes.starts_with(MAGIC) {
        bail!("not a Capture One .comask container");
    }
    let type_at = find_from(bytes, b"CaptureOne Mask", 0)
        .ok_or_else(|| anyhow::anyhow!("missing TYPE=CaptureOne Mask"))?;
    let ver_at = find_from(bytes, b"VER", type_at).ok_or_else(|| anyhow::anyhow!("missing VER"))?;
    bytes.get(ver_at + 3..ver_at + 8).ok_or_else(|| anyhow::anyhow!("truncated VER"))?;

    let mask_info = find_from(bytes, b"MaskInfo", 0).ok_or_else(|| anyhow::anyhow!("missing MaskInfo"))?;
    let info_hdr = find_from(bytes, b"Hdr", mask_info).ok_or_else(|| anyhow::anyhow!("missing MaskInfo.Hdr"))?;
    let info = hdr_payload(bytes, info_hdr).ok_or_else(|| anyhow::anyhow!("truncated MaskInfo.Hdr"))?;
    let width = le_u32_at(info, 4).ok_or_else(|| anyhow::anyhow!("truncated MaskInfo width"))? as usize;
    let height = le_u32_at(info, 8).ok_or_else(|| anyhow::anyhow!("truncated MaskInfo height"))? as usize;
    if width == 0 || height == 0 || width > 100_000 || height > 100_000 {
        bail!("implausible MaskInfo size {width}x{height}");
    }

    let blk_list = comask_blocks(bytes)?;
    let tile_columns = width.div_ceil(64);
    let tile_rows = height.div_ceil(64);
    let starts = find_all(bytes, MASK_INDEX);
    if starts.is_empty() {
        bail!("no MaskIndex records");
    }

    let mut planes = Vec::new();
    for (i, &start) in starts.iter().enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(bytes.len());
        let segment = &bytes[start..end];
        let uuid = find_uuid(segment);
        if let Some(wanted) = wanted {
            if !uuid.as_deref().is_some_and(|u| wanted.contains(&u)) {
                continue;
            }
        }
        let mask_index = le_u32_at(segment, 15).ok_or_else(|| anyhow::anyhow!("truncated MaskIndex value"))?;
        let grid = decode_mask_grid(segment, tile_columns, tile_rows);
        let gradient = linear_gradient(segment).ok().flatten();
        if grid.is_none() && gradient.is_none() {
            continue;
        }
        let (first_block, grid) = grid.unwrap_or((0, Vec::new()));
        let mut plane = MaskPlane {
            width,
            height,
            data: vec![0.0; width * height],
            mask_index,
            uuid,
            linear_gradient: gradient,
            first_block,
            blocks: 0,
            decoded_blocks: 0,
            failed_blocks: 0,
            coverage: 0.0,
            tile_columns,
            tile_rows,
        };
        if let Some(gradient) = gradient {
            if grid.is_empty() {
                fill_gradient(&mut plane, gradient);
            }
        }
        if !grid.is_empty() {
            plane.blocks = grid.iter().filter(|&&v| v == 8 || v == 9).count();
            fill_plane(&mut plane, &grid, &blk_list)?;
        }
        let nonzero = plane.data.iter().filter(|v| **v > 0.0).count();
        plane.coverage = nonzero as f32 / plane.data.len().max(1) as f32;
        planes.push(plane);
    }
    Ok(planes)
}

/// Parameter-only linear gradient: smoothstep of the projected position between
/// the two end points. C1's own 8-bit quantization leaves ±1 LSB on the sample.
fn fill_gradient(plane: &mut MaskPlane, gradient: [f32; 5]) {
    let [x0, y0, x1, y1, _midpoint] = gradient.map(|v| v as f64);
    let (dx, dy) = (x1 - x0, y1 - y0);
    let length2 = dx * dx + dy * dy;
    if length2 <= f64::EPSILON {
        return;
    }
    for y in 0..plane.height {
        let row = y * plane.width;
        for x in 0..plane.width {
            let t = (((x as f64 - x0) * dx + (y as f64 - y0) * dy) / length2).clamp(0.0, 1.0);
            plane.data[row + x] = (t * t * (3.0 - 2.0 * t)) as f32;
        }
    }
}

/// Decodes a `Map` chunk into `cols * rows` tile cells.
///
/// Layout: the first `9` cell marks this mask's first block tile and is followed
/// by a big-endian u16 (index of that block in `Blk`); `0x00` escapes the next
/// byte as a literal cell value.
fn decode_mask_grid(segment: &[u8], cols: usize, rows: usize) -> Option<(usize, Vec<u8>)> {
    let at = find(segment, b"Map\x00\x00")?;
    let (cells, _) = chunk_payload_at(segment, at, 3)?;
    let want = cols.checked_mul(rows)?;
    let mut grid = Vec::with_capacity(want);
    let mut first_block = None;
    let mut i = 0usize;
    while grid.len() < want {
        let b = *cells.get(i)?;
        let value = if b == 0 {
            i += 1;
            *cells.get(i)?
        } else {
            b
        };
        i += 1;
        grid.push(value);
        if value == 9 && first_block.is_none() {
            let hi = *cells.get(i)? as usize;
            let lo = *cells.get(i + 1)? as usize;
            first_block = Some((hi << 8) | lo);
            i += 2;
        }
    }
    Some((first_block?, grid))
}

/// Writes the grid's tiles into `plane.data`.
fn fill_plane(plane: &mut MaskPlane, grid: &[u8], blocks: &[&[u8]]) -> AnyResult<()> {
    let mut block = plane.first_block;
    let mut carry = 0u8;
    for (cell, &value) in grid.iter().enumerate() {
        let (tile_row, tile_col) = (cell / plane.tile_columns, cell % plane.tile_columns);
        let tile: [u8; 4096] = match value {
            8 | 9 => {
                let raw = blocks
                    .get(block)
                    .ok_or_else(|| anyhow::anyhow!("Map references block {block} but Blk has {}", blocks.len()))?;
                block += 1;
                let tile = decode_block(raw)
                    .ok_or_else(|| anyhow::anyhow!("block {} failed to decode", block - 1))?;
                plane.decoded_blocks += 1;
                tile
            }
            255 => {
                carry = 255;
                [255u8; 4096]
            }
            0 => {
                carry = 0;
                [0u8; 4096]
            }
            1 => [carry; 4096],
            other => bail!("unknown Map cell {other} at tile ({tile_row}, {tile_col})"),
        };
        let x0 = tile_col * 64;
        let y0 = tile_row * 64;
        for row in 0..64 {
            let y = y0 + row;
            if y >= plane.height {
                break;
            }
            let n = (plane.width - x0).min(64);
            let dst = y * plane.width + x0;
            for col in 0..n {
                plane.data[dst + col] = tile[row * 64 + col] as f32 / 255.0;
            }
        }
    }
    Ok(())
}

/// Debug/tool helper: reads a `.comask` path and returns its first mask plane.
///
/// Returns `None` on I/O errors or malformed/truncated input. Not wired into
/// `AdjustmentSet.layers`.
pub fn read_comask<P: AsRef<std::path::Path>>(path: P) -> Option<MaskPlane> {
    let bytes = std::fs::read(path).ok()?;
    read_comask_bytes(&bytes).ok()
}

/// Splits the `Blk`/`Idx` table into raw block spans: `[tag u8][payload]`.
///
/// `Idx` holds one u16 payload size (including the tag byte) per block; the
/// payloads follow the size table back-to-back.
pub fn comask_blocks(bytes: &[u8]) -> AnyResult<Vec<&[u8]>> {
    let mask_info = find_from(bytes, b"MaskInfo", 0).ok_or_else(|| anyhow::anyhow!("missing MaskInfo"))?;
    let blk = find_from(bytes, b"Blk", mask_info).ok_or_else(|| anyhow::anyhow!("missing Blk"))?;
    let blk_hdr = find_from(bytes, b"Hdr", blk).ok_or_else(|| anyhow::anyhow!("missing Blk.Hdr"))?;
    let blk_info = hdr_payload(bytes, blk_hdr).ok_or_else(|| anyhow::anyhow!("truncated Blk.Hdr"))?;
    let count = le_u32_at(blk_info, 0).ok_or_else(|| anyhow::anyhow!("truncated block count"))? as usize;
    if count == 0 || count > 1_000_000 {
        bail!("implausible block count {count}");
    }

    let idx = find_from(bytes, b"Idx", blk).ok_or_else(|| anyhow::anyhow!("missing Idx"))?;
    let sizes_at = idx + 7;
    let sizes_end = sizes_at.checked_add(count * 2).ok_or_else(|| anyhow::anyhow!("block table overflow"))?;
    if sizes_end > bytes.len() {
        bail!("truncated block size table");
    }

    let mut sizes = Vec::with_capacity(count);
    for i in 0..count {
        let size = le_u16_at(bytes, sizes_at + i * 2).ok_or_else(|| anyhow::anyhow!("truncated block size"))? as usize;
        sizes.push(size);
    }
    let total: usize = sizes.iter().sum();

    // shortcut: the block payloads sit behind a `dat\0\0` chunk header whose
    // length is not constant (13 bytes in the sample), so probe a few start
    // offsets and take the first where every tag is a known block type. Widen
    // the probe range if a sample with a longer header shows up.
    let mut start = None;
    for shift in 0..=64usize {
        let Some(q) = sizes_end.checked_add(shift) else { break };
        let Some(end) = q.checked_add(total) else { break };
        if end > bytes.len() {
            continue;
        }
        let mut at = q;
        let mut ok = true;
        for &size in &sizes {
            if size == 0 || (bytes[at] & 0x7f) > 2 {
                ok = false;
                break;
            }
            at += size;
        }
        if ok {
            start = Some(q);
            break;
        }
    }
    let start = start.ok_or_else(|| anyhow::anyhow!("no valid block payload start"))?;

    let mut blocks = Vec::with_capacity(count);
    let mut at = start;
    for (i, &size) in sizes.iter().enumerate() {
        let end = at + size;
        let blk = bytes.get(at..end).ok_or_else(|| anyhow::anyhow!("truncated block payload at block {i}"))?;
        blocks.push(blk);
        at = end;
    }
    Ok(blocks)
}

/// One 64×64 alpha tile as stored by a `Blk` record.
///
/// Block layout is `[tag u8][payload]`:
/// * `tag & 0x7f == 0` — 4096 raw bytes;
/// * `tag & 0x7f == 1` — PackBits variant;
/// * `tag & 0x7f == 2` — record stream documented in `COMASK-DECODER-SPEC.md`.
///
/// Anything else (including a truncated record stream) returns `None`; partial
/// tiles are never emitted.
pub fn decode_block(block: &[u8]) -> Option<[u8; 4096]> {
    let (&tag, payload) = block.split_first()?;
    decode_block_payload(tag & 0x7f, payload)
}

/// [`decode_block`] without the leading tag byte.
pub fn decode_block_payload(tag: u8, payload: &[u8]) -> Option<[u8; 4096]> {
    match tag {
        0 => payload.get(..4096)?.try_into().ok(),
        1 => decode_packbits(payload),
        2 => decode_record(payload),
        _ => None,
    }
}

const SYMBOL: [u8; 30] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11, 12, 13, 14,
    15,
];
const LENGTH_MASK: [u8; 30] = [
    0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x3f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f,
    0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x0f, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07, 0x07,
];
/// Symbols 1..=5 store a segment head followed by per-run deltas.
const SEGMENT_DELTA: [u8; 16] = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
/// 16×16 de-block shuffle applied before the prefix sum when the plane bit is set.
const DEBLOCK_ORDER: [usize; 16] = [0, 1, 14, 15, 3, 2, 13, 12, 4, 7, 8, 11, 5, 6, 9, 10];

fn decode_packbits(src: &[u8]) -> Option<[u8; 4096]> {
    let mut out = [0u8; 4096];
    out[0] = *src.first()?;
    let mut i = 1usize;
    let mut o = 1usize;
    while o < 4096 {
        let b = *src.get(i)?;
        i += 1;
        if b < 0x40 {
            // C1 copies `token + 1` bytes and truncates the last run at 4096.
            let n = b as usize + 1;
            let chunk = src.get(i..i + n)?;
            let take = n.min(4096 - o);
            out.get_mut(o..o + take)?.copy_from_slice(&chunk[..take]);
            i += n;
            o += take;
        } else if b <= 0x7f {
            let n = (b - 0x3f) as usize;
            let prev = out[o - 1];
            out.get_mut(o..o + n)?.fill(prev);
            o += n;
        } else {
            break;
        }
    }
    Some(out)
}

fn decode_record(payload: &[u8]) -> Option<[u8; 4096]> {
    let (seed, runs, token_bytes) = parse_runs(payload)?;
    let plane = payload.get(1 + token_bytes..)?;

    let mut count = [0usize; 16];
    count[1] = 4;
    for &(sym, len) in &runs {
        count[sym] += len + SEGMENT_DELTA[sym] as usize;
    }
    let bits1 = count[1] + count[3] + count[5];
    let bits2 = count[2] + count[3];
    let bits4 = count[4] + count[5];

    let mut work = vec![0u8; 0x2400];
    let need = (bits1 + bits2 + bits4 + count[15])
        .max(count[1] + count[3])
        .max(bits1 + bits2 + count[4] + count[5]);
    if need > work.len() {
        return None;
    }
    let mut pos = 0usize;
    pos = unpack_plane(plane, pos, bits1, 1, &mut work, 0)?;
    pos = unpack_plane(plane, pos, bits2, 2, &mut work, bits1)?;
    pos = unpack_plane(plane, pos, bits4, 4, &mut work, bits1 + bits2)?;
    let raw = plane.get(pos..pos + count[15])?;
    work[bits1 + bits2 + bits4..bits1 + bits2 + bits4 + count[15]].copy_from_slice(raw);
    for k in 0..count[3] {
        work[count[1] + k] |= work[bits1 + count[2] + k] << 1;
    }
    for k in 0..count[5] {
        work[count[1] + count[3] + k] |= work[bits1 + bits2 + count[4] + k] << 1;
    }

    // Segment cursors: 1..=5 start after their head, 15 after the raw block.
    // 12/13/14 start at 3/3/4 as fitted to every real block in the sample.
    let mut cursor = [0usize; 16];
    cursor[1] = 4;
    cursor[2] = bits1;
    cursor[3] = count[1];
    cursor[4] = bits1 + bits2;
    cursor[5] = count[1] + count[3];
    cursor[12] = 3;
    cursor[13] = 3;
    cursor[14] = 4;
    cursor[15] = bits1 + bits2 + bits4;

    let mut tile = [0u8; 4096];
    tile[0] = seed;
    let mut o = 1usize;
    for &(sym, len) in &runs {
        match sym {
            0 => {}
            _ if SEGMENT_DELTA[sym] == 0 => {
                let src = work.get(cursor[sym]..cursor[sym] + len)?;
                tile.get_mut(o..o + len)?.copy_from_slice(src);
                cursor[sym] += len;
            }
            _ => {
                let base = *work.get(cursor[sym])?;
                for k in 0..len {
                    tile[o + k] = work.get(cursor[sym] + 1 + k)?.wrapping_sub(base);
                }
                cursor[sym] += 1 + len;
            }
        }
        o += len;
    }

    let transpose = work[0] & 1 != 0;
    let deblock = work[1] != 0;
    if deblock {
        tile = deblock_tile(&tile);
    }
    prefix_sum(&mut tile);
    if transpose {
        transpose_tile(&mut tile);
    }
    Some(tile)
}

/// Parses the token stream: seed byte, `(symbol, run length)` list and the byte
/// count of the token area. Returns `None` when the stream is truncated before
/// the run lengths reach 4095.
fn parse_runs(payload: &[u8]) -> Option<(u8, Vec<(usize, usize)>, usize)> {
    let seed = *payload.first()?;
    let mut i = 0usize;
    let mut total = 0u32;
    let mut runs: Vec<(usize, usize)> = Vec::new();
    loop {
        let t = *payload.get(1 + i)?;
        let (sym, len);
        if t < 0xf0 {
            let slot = (t >> 3) as usize;
            sym = SYMBOL[slot] as usize;
            len = (t & LENGTH_MASK[slot]) as usize + 1;
            i += 1;
        } else {
            sym = (t & 0x0f) as usize;
            let b1 = *payload.get(2 + i)? as usize;
            if b1 < 0x80 {
                len = b1 + 1;
                i += 2;
            } else {
                let b2 = *payload.get(3 + i)? as usize;
                len = (((b1 & 0x3f) << 8) | b2) + 1;
                i += 3;
            }
        }
        runs.push((sym, len));
        total += len as u32;
        if total >= 0xfff {
            break;
        }
    }
    Some((seed, runs, i))
}

fn unpack_plane(plane: &[u8], mut pos: usize, count: usize, bits: usize, out: &mut [u8], off: usize) -> Option<usize> {
    let mask = ((1u16 << bits) - 1) as u8;
    let per_byte = 8 / bits;
    let per_16 = 128 / bits;
    let per_4 = 32 / bits;
    let mut got = 0usize;
    let mut rem = count;
    while rem >= per_16 {
        for j in 0..per_byte {
            for i in 0..16 {
                out[off + got] = (plane.get(pos + i)? >> (j * bits)) & mask;
                got += 1;
            }
        }
        pos += 16;
        rem -= per_16;
    }
    while rem >= per_4 {
        for j in 0..per_byte {
            for i in 0..4 {
                out[off + got] = (plane.get(pos + i)? >> (j * bits)) & mask;
                got += 1;
            }
        }
        pos += 4;
        rem -= per_4;
    }
    while rem > 0 {
        let n = rem.min(per_byte);
        for j in 0..n {
            out[off + got] = (plane.get(pos)? >> (j * bits)) & mask;
            got += 1;
        }
        pos += 1;
        rem -= n;
    }
    Some(pos)
}

fn deblock_tile(src: &[u8; 4096]) -> [u8; 4096] {
    let mut dst = [0u8; 4096];
    for outer in 0..16 {
        let base = outer * 256;
        for esi in (0..16).step_by(4) {
            for t in 0..4 {
                let s = esi + t;
                let from = base + s * 16;
                let to = esi * 4 + t * 4;
                for row in 0..4 {
                    for col in 0..4 {
                        dst[base + row * 0x40 + to + col] = src[from + DEBLOCK_ORDER[row * 4 + col]];
                    }
                }
            }
        }
    }
    dst
}

fn prefix_sum(d: &mut [u8; 4096]) {
    for x in 1..64 {
        d[x] = d[x].wrapping_add(d[x - 1]);
    }
    for y in 1..64 {
        for x in 0..64 {
            d[y * 64 + x] = d[y * 64 + x].wrapping_add(d[(y - 1) * 64 + x]);
        }
    }
}

fn transpose_tile(d: &mut [u8; 4096]) {
    for y in 0..64 {
        for x in y + 1..64 {
            d.swap(y * 64 + x, x * 64 + y);
        }
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

    /// A parameter-only gradient has no Map tiles but still produces a plane.
    #[test]
    fn a_linear_gradient_fills_its_plane_with_a_smooth_ramp() {
        let mut plane = MaskPlane {
            width: 8,
            height: 1,
            data: vec![0.0; 8],
            mask_index: 3,
            uuid: None,
            linear_gradient: Some([0.0, 0.0, 7.0, 0.0, 0.5]),
            first_block: 0,
            blocks: 0,
            decoded_blocks: 0,
            failed_blocks: 0,
            coverage: 0.0,
            tile_columns: 1,
            tile_rows: 1,
        };
        fill_gradient(&mut plane, [0.0, 0.0, 7.0, 0.0, 0.5]);
        assert_eq!(plane.data[0], 0.0);
        assert_eq!(plane.data[7], 1.0);
        assert!(plane.data.windows(2).all(|w| w[0] <= w[1]));
        assert!(plane.data.iter().all(|v| (0.0..=1.0).contains(v)));
    }

    /// A minimal container with one parameter-only 8x1 gradient mask.
    fn gradient_container(midpoint: f32) -> Vec<u8> {
        let mut b = b"\x04XCon".to_vec();
        push(&mut b, b"TYPE\x00\x0fCaptureOne Mask");
        push(&mut b, b"VER\x00\x00\x02\x03\x00");
        push(&mut b, b"MaskInfo\x00\x22\x00\x03Hdr\x00\x22\xff\x18");
        push(&mut b, &1u32.to_le_bytes());
        push(&mut b, &8u32.to_le_bytes());
        push(&mut b, &1u32.to_le_bytes());
        push(&mut b, &[0u8; 12]);
        push(&mut b, b"Blk\x00\x22\x00\xff\x03Hdr\x00\x22\xff\x10");
        push(&mut b, &1u32.to_le_bytes());
        push(&mut b, &[0u8; 12]);
        push(&mut b, b"Idx\x00\x22");
        push(&mut b, &[0u8, 0]);
        push(&mut b, &1u16.to_le_bytes());
        push(&mut b, &[0u8]);
        push(
            &mut b,
            &mask(0, "24391591-4129-4c9f-a9f5-bb4dd4c41289", 3, None, Some([0.0, 0.0, 7.0, 0.0, midpoint])),
        );
        b
    }

    fn plane_bits(plane: &MaskPlane) -> Vec<u32> {
        plane.data.iter().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn gradient_midpoint_is_kept_and_does_not_change_pixels() {
        let half = read_comask_masks(&gradient_container(0.5)).unwrap();
        let quarter = read_comask_masks(&gradient_container(0.25)).unwrap();
        assert_eq!(half.len(), 1);
        assert_eq!(quarter.len(), 1);
        assert_eq!(half[0].linear_gradient, Some([0.0, 0.0, 7.0, 0.0, 0.5]));
        assert_eq!(quarter[0].linear_gradient, Some([0.0, 0.0, 7.0, 0.0, 0.25]));
        assert_eq!(plane_bits(&half[0]), plane_bits(&quarter[0]));
    }

    /// The 0.5 plane must stay bit-identical to the smoothstep-only formula the
    /// decoder used before the midpoint was carried.
    #[test]
    fn half_midpoint_plane_is_bit_identical_to_plain_smoothstep() {
        let planes = read_comask_masks(&gradient_container(0.5)).unwrap();
        for (x, v) in planes[0].data.iter().enumerate() {
            let t = ((x as f64 * 7.0) / 49.0).clamp(0.0, 1.0);
            assert_eq!(v.to_bits(), ((t * t * (3.0 - 2.0 * t)) as f32).to_bits(), "x={x}");
        }
    }

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
                "24391591-4129-4c9f-a9f5-bb4dd4c41289",
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
                "3977aaec-463d-4677-b661-a8f4a385dab5",
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
            Some("24391591-4129-4c9f-a9f5-bb4dd4c41289")
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
    fn decode_pixels_reports_masks_without_a_map() {
        let bytes = fixture();
        let doc = ComaskDocument::parse(&bytes).unwrap();
        assert!(doc.decode_pixels(&bytes, 0).is_err());
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

    /// Token tables, run lengths and the 0xFFF stop from the r19 spec samples.
    #[test]
    fn record_token_stream_matches_spec_samples() {
        let cases: [(&[u8], u8, &[(usize, usize)]); 3] = [
            (
                &[0xc8, 0xe8, 0xf0, 0x8f, 0xfd, 0x02, 0x38],
                200,
                &[(15, 1), (0, 4094)],
            ),
            (
                &[0xc8, 0x0c, 0xeb, 0xf0, 0x8f, 0xed, 0x02, 0x38, 0x00, 0x00, 0x00],
                200,
                &[(0, 13), (15, 4), (0, 4078)],
            ),
            (
                &[0x80, 0xf0, 0x7e, 0xe9, 0xf0, 0x8f, 0x7d, 0x02, 0x80, 0x00],
                128,
                &[(0, 127), (15, 2), (0, 3966)],
            ),
        ];
        for (payload, seed, want) in cases {
            let (got_seed, runs, _) = parse_runs(payload).unwrap();
            assert_eq!(got_seed, seed);
            assert_eq!(runs, want);
            assert_eq!(runs.iter().map(|(_, len)| len).sum::<usize>(), 4095);
        }
    }

    #[test]
    fn record_tiles_place_single_and_double_column() {
        let one = decode_block_payload(2, &[0xc8, 0xe8, 0xf0, 0x8f, 0xfd, 0x02, 0x38]).unwrap();
        for y in 0..64 {
            assert_eq!(one[y * 64], 200, "row {y} column 0");
            assert!(one[y * 64 + 1..(y + 1) * 64].iter().all(|&v| v == 0), "row {y} tail");
        }
        let two = decode_block_payload(
            2,
            &[0xc8, 0x0c, 0xeb, 0xf0, 0x8f, 0xed, 0x02, 0x38, 0x00, 0x00, 0x00],
        )
        .unwrap();
        for y in 0..64 {
            assert_eq!(&two[y * 64..y * 64 + 2], &[200, 200], "row {y} columns 0..2");
            assert!(two[y * 64 + 2..(y + 1) * 64].iter().all(|&v| v == 0));
        }
    }

    #[test]
    fn packbits_variant_decodes_copy_and_repeat() {
        let tile = decode_block_payload(1, &[0x07, 0x01, 10, 11, 0x41, 0x80]).unwrap();
        assert_eq!(&tile[..6], &[7, 10, 11, 11, 11, 0]);
        assert!(tile[5..].iter().all(|&v| v == 0));
    }

    #[test]
    fn packbits_truncates_the_last_literal_at_the_tile_end() {
        let mut src = vec![0u8];
        for _ in 0..63 {
            src.push(0x3F);
            src.extend(std::iter::repeat(7).take(64));
        }
        src.push(0x3F);
        src.extend(std::iter::repeat(9).take(64));
        src.push(0x80);
        let tile = decode_block_payload(1, &src).unwrap();
        assert!(tile[..4033].iter().all(|&v| v == 0 || v == 7));
        assert!(tile[4033..].iter().all(|&v| v == 9));
    }

    #[test]
    fn unpack_plane_expands_1_2_and_4_bit_groups() {
        let mut out = [0u8; 8];
        assert_eq!(unpack_plane(&[0b1011_0110], 0, 4, 1, &mut out, 0), Some(1));
        assert_eq!(&out[..4], &[0, 1, 1, 0]);
        assert_eq!(unpack_plane(&[0b1110_0100], 0, 2, 2, &mut out, 0), Some(1));
        assert_eq!(&out[..2], &[0, 1]);
        assert_eq!(unpack_plane(&[0x21], 0, 2, 4, &mut out, 0), Some(1));
        assert_eq!(&out[..2], &[1, 2]);

        // Exactly one 16-byte block of 2-bit groups exercises the wide loop.
        let plane: Vec<u8> = (0..16).collect();
        let mut out = [0u8; 64];
        assert_eq!(unpack_plane(&plane, 0, 64, 2, &mut out, 0), Some(16));
        for j in 0..4 {
            for i in 0..16 {
                assert_eq!(out[j * 16 + i], ((i >> (j * 2)) & 3) as u8);
            }
        }
    }

    fn cover_fixture(arrow_count: u32, truncated: bool) -> Vec<u8> {
        let mut b = Vec::new();
        push(&mut b, b"Cover\x00\x22\x00");
        push(&mut b, b"Hdr\x00\x22\xff\x0c");
        push(&mut b, &[2, 0, 3, 0]);
        push(&mut b, &2u32.to_le_bytes());
        push(&mut b, &arrow_count.to_le_bytes());
        for (kind, src, dst, flags) in [
            (1u32, [10.5f32, 20.25], [30.5f32, 40.75], 0x18u32),
            (2, [1.0, 2.0], [3.0, 4.0], 0),
        ]
        .into_iter()
        .take(arrow_count as usize)
        {
            push(&mut b, b"Vec\x00\x22\xff\x18");
            push(&mut b, &kind.to_le_bytes());
            push(&mut b, &src[0].to_le_bytes());
            push(&mut b, &src[1].to_le_bytes());
            push(&mut b, &dst[0].to_le_bytes());
            push(&mut b, &dst[1].to_le_bytes());
            push(&mut b, &flags.to_le_bytes());
            if !truncated {
                push(&mut b, b"Data\x00\x22\x03\xaa\xbb\xcc");
            }
        }
        b
    }

    #[test]
    fn cover_parses_clone_and_heal_arrows() {
        let cover = cover(&cover_fixture(2, false)).unwrap();
        assert_eq!((cover.version, cover.mask_index, cover.cover_type, cover.count), (2, 3, 2, 2));
        assert_eq!(cover.arrows.len(), 2);
        let clone = cover.arrows[0];
        assert_eq!(clone.arrow_type, 1);
        assert_eq!((clone.src, clone.dst, clone.flags, clone.data_len), ([10.5, 20.25], [30.5, 40.75], 0x18, 3));
        assert_eq!(cover.arrows[1].arrow_type, 2);
        assert_eq!(cover.arrows[1].data_len, 3);
    }

    #[test]
    fn cover_stops_at_a_truncated_arrow_without_panicking() {
        let parsed = cover(&cover_fixture(2, true)).unwrap();
        assert_eq!(parsed.arrows.len(), 2);
        assert!(parsed.arrows.iter().all(|a| a.data_len == 0));
        let mut cut = cover_fixture(1, false);
        cut.truncate(cut.len() - 20);
        assert_eq!(cover(&cut).map(|c| c.arrows.len()), Some(0));
        assert!(cover(b"Cover\x00\x22\x00Hdr\x00\x22\xff\x0c\x02\x00").is_none());
    }

    const FIXTURE: &[u8] = include_bytes!("../testdata/comask/decode-fixture.comask");
    const FIXTURE_REFS: &[u8] = include_bytes!("../testdata/comask/reference-planes.bin");

    /// Five 192x128 crops of the real sample, decoded against Capture One's
    /// own `IC_MaskSampleInt` planes. No catalog or installation needed.
    #[test]
    fn fixture_planes_match_capture_one_references_byte_for_byte() {
        let planes = read_comask_masks(FIXTURE).unwrap();
        let uuids = [
            "24391591-4129-4c9f-a9f5-bb4dd4c41289",
            "d7ebf169-4d9f-41fe-b264-87f8cae0f083",
            "6b96b29b-326f-4914-a7cd-56ffa01c881e",
            "cf7a832a-6002-4543-aba0-ccdf30d1242b",
            "7c423e46-c999-440c-a71e-09d1139d8a7e",
        ];
        let blocks = [6usize, 6, 5, 6, 6];
        assert_eq!(planes.len(), 5);
        let plane_len = 192 * 128;
        assert_eq!(FIXTURE_REFS.len(), plane_len * planes.len());
        for (i, plane) in planes.iter().enumerate() {
            assert_eq!((plane.width, plane.height), (192, 128));
            assert_eq!(plane.uuid.as_deref(), Some(uuids[i]));
            assert_eq!((plane.blocks, plane.decoded_blocks, plane.failed_blocks), (blocks[i], blocks[i], 0));
            let want = &FIXTURE_REFS[i * plane_len..(i + 1) * plane_len];
            assert_eq!(plane.to_u8(), want, "mask {}", plane.mask_index);
        }
        assert_eq!(planes.iter().map(|p| p.mask_index).collect::<Vec<_>>(), vec![0, 1, 2, 5, 6]);
    }

    #[test]
    fn truncated_fixture_returns_an_error_instead_of_panicking() {
        for cut in [0, 16, 200, 1000, FIXTURE.len() - 10] {
            assert!(read_comask_masks(&FIXTURE[..cut]).is_err(), "cut {cut}");
        }
    }

    #[test]
    fn truncated_record_stream_returns_none() {
        assert!(decode_block_payload(2, &[0x00]).is_none());
        assert!(decode_block_payload(2, &[0x00, 0xe8]).is_none());
        // Runs reach 4095 but the bit-plane area is missing.
        assert!(decode_block_payload(2, &[0xc8, 0xe8, 0xf0, 0x8f, 0xfd]).is_none());
        assert!(decode_block_payload(0, &[0u8; 10]).is_none());
    }

    /// `out/oracle-tiles.bin` holds the `0x15C75F0` output for each 4096-byte
    /// tag-2 payload (`scripts/make_oracle.py`). 51 tag-0 blocks are raw and are
    /// skipped: feeding their pixels to the record decoder is meaningless.
    #[test]
    fn decodes_real_blocks_against_dll_oracle() {
        let Some(bytes) = lane_sample() else { return };
        let oracle_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../out/oracle-tiles.bin");
        let Ok(oracle) = std::fs::read(&oracle_path) else { return };
        let blocks = comask_blocks(&bytes).unwrap();
        assert_eq!(blocks.len(), 1246);
        assert_eq!(oracle.len(), blocks.len() * 4096);
        let mut checked = 0usize;
        let mut bad = Vec::new();
        for (i, block) in blocks.iter().enumerate() {
            if block[0] & 0x7f != 2 {
                continue;
            }
            checked += 1;
            let tile = decode_block(block).unwrap();
            if tile[..] != oracle[i * 4096..(i + 1) * 4096] {
                bad.push(i);
            }
        }
        assert_eq!(checked, 1195);
        assert!(bad.is_empty(), "mismatched blocks: {bad:?}");
    }

    fn lane_sample() -> Option<Vec<u8>> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../samples/_DSC5673.NEF.comask");
        std::fs::read(path).ok()
    }

    #[test]
    fn reads_real_comask_plane_and_bbox() {
        let Some(bytes) = lane_sample() else { return };
        let plane = read_comask_bytes(&bytes).unwrap();
        assert_eq!((plane.width, plane.height), (6048, 4032));
        assert_eq!(plane.mask_index, 0);
        assert_eq!(plane.uuid.as_deref(), Some("24391591-4129-4c9f-a9f5-bb4dd4c41289"));
        assert_eq!((plane.first_block, plane.blocks), (0, 204));
        assert_eq!((plane.decoded_blocks, plane.failed_blocks), (204, 0));
        assert_eq!((plane.tile_columns, plane.tile_rows), (95, 63));

        assert!((plane.coverage - 0.0463250).abs() < 1e-5, "coverage {}", plane.coverage);
        assert!(plane.data.iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(plane.data.iter().any(|v| *v > 0.0));

        let (x0, y0, x1, y1) = plane.bbox().unwrap();
        assert_eq!((x0, y0, x1, y1), (1514, 0, 3083, 2156));
        assert!(x1 < plane.width && y1 < plane.height);
        let bbox_area = (x1 - x0 + 1) * (y1 - y0 + 1);
        assert!(bbox_area < plane.width * plane.height / 4);

        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../samples/_DSC5673.NEF.comask");
        assert!(read_comask(path).is_some());
    }

    #[test]
    fn decodes_every_mask_that_stores_a_map() {
        let Some(bytes) = lane_sample() else { return };
        let planes = read_comask_masks(&bytes).unwrap();
        let shape: Vec<(u32, usize, usize, usize)> =
            planes.iter().map(|p| (p.mask_index, p.first_block, p.blocks, p.decoded_blocks)).collect();
        assert_eq!(
            shape,
            vec![(0, 0, 204, 204), (1, 204, 688, 688), (2, 892, 8, 8), (5, 900, 192, 192), (6, 1092, 154, 154)]
        );
        assert!(planes.iter().all(|p| p.failed_blocks == 0));
        assert_eq!(planes.iter().map(|p| p.blocks).sum::<usize>(), 1246);

        let doc = ComaskDocument::parse(&bytes).unwrap();
        let second = doc.decode_pixels(&bytes, 1).unwrap();
        assert_eq!((second.mask_index, second.blocks), (1, 688));
        assert_eq!(doc.decode_pixels(&bytes, 3), Err(ComaskError::NoRaster));
        assert_eq!(doc.decode_pixels(&bytes, 99), Err(ComaskError::NoRaster));
    }

    fn next_line<'a>(data: &'a [u8], pos: &mut usize) -> Option<&'a [u8]> {
        let rest = data.get(*pos..)?;
        let end = rest.iter().position(|&b| b == b'\n')?;
        *pos += end + 1;
        Some(&rest[..end])
    }

    fn load_pgm(path: &std::path::Path) -> Option<(usize, usize, Vec<u8>)> {
        let data = std::fs::read(path).ok()?;
        let mut pos = 0usize;
        if next_line(&data, &mut pos)? != b"P5" {
            return None;
        }
        let dims = String::from_utf8_lossy(next_line(&data, &mut pos)?).to_string();
        let mut parts = dims.split_whitespace().map(|v| v.parse::<usize>().ok());
        let (w, h) = (parts.next()??, parts.next()??);
        if next_line(&data, &mut pos)? != b"255" {
            return None;
        }
        let pixels = data.get(pos..pos + w * h)?.to_vec();
        Some((w, h, pixels))
    }

    fn reference_dir() -> Option<std::path::PathBuf> {
        let dir = std::env::var_os("COMASK_REF_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../../../win_c1_comask_r14-20261113-1320/references")
            });
        dir.is_dir().then_some(dir)
    }

    /// 0-diff against the Capture One `IC_MaskSampleInt` planes for every mask
    /// that stores pixels. Uses the r14 reference directory; skipped when absent.
    #[test]
    fn real_masks_match_reference_pgm_exactly() {
        let Some(bytes) = lane_sample() else { return };
        let Some(dir) = reference_dir() else { return };
        let planes = read_comask_masks(&bytes).unwrap();
        assert_eq!(planes.len(), 5);
        for plane in &planes {
            let path = dir.join(format!("mask{}.pgm", plane.mask_index));
            let Some((w, h, pixels)) = load_pgm(&path) else { continue };
            assert_eq!((w, h), (plane.width, plane.height));
            let diff = pixels
                .iter()
                .zip(&plane.data)
                .filter(|(want, got)| **want != (**got * 255.0).round() as u8)
                .count();
            assert_eq!(diff, 0, "mask {} differs in {diff} pixels", plane.mask_index);
        }
    }

    #[test]
    fn corrupted_map_and_truncation_return_errors_without_panicking() {
        let Some(bytes) = lane_sample() else { return };
        let map_at = find(&bytes, b"Map\x00\x00").unwrap();
        let first_nine = (map_at + 5..bytes.len()).find(|&i| bytes[i] == 9).unwrap();
        let mut broken = bytes.clone();
        broken[first_nine + 1] = 0xff;
        broken[first_nine + 2] = 0xff;
        assert!(read_comask_bytes(&broken).is_err());
        assert!(read_comask_masks(&broken).is_err());

        let cut = bytes.len() - 40_000;
        assert!(read_comask_bytes(&bytes[..cut]).is_err());
    }

    #[test]
    fn truncated_real_sample_returns_error() {
        let Some(bytes) = lane_sample() else { return };
        for cut in [0, 100, 1_000, 33_000, bytes.len() - 100_000] {
            assert!(read_comask_bytes(&bytes[..cut]).is_err(), "cut {cut} should fail");
        }
    }
}
