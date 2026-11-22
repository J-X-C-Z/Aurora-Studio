//! Reads the dust patches out of a Capture One `.lcc` sidecar.
//!
//! An `.lcc` is a Phase One "XCon" chunk container, the same wrapper as
//! `.comask`. Its first chunk is `XCon` and carries a 4-byte magic plus two
//! version bytes (major 0, minor >= 2); a `TYPE` chunk holding the 14 bytes
//! `CaptureOne LCC` and a `VER` chunk follow. Groups (`Camera`, `Lens`,
//! `LightFalloff`, `SpotRem`, ...) mark a flat run of children that ends with a
//! `0xFE` marker.
//!
//! Each chunk is
//! `varint(name_len) name flags:u8 pack:u8 [varint(payload_len) payload]`, where
//! `flags & 1` means "children follow" instead of a payload. The varint packs
//! 6 bits per byte, big-endian, with bit `0x40` set on the first and last byte.
//!
//! Inside `SpotRem`, `SpotList` is a run of 16-byte records (`x`, `y`, `w`, `h`
//! as `u16`, `offset` as `u32`, `bias` as `i16`, `step` as `u16`) in the
//! container's byte order, and `SpotData` is the per-pixel gain plane those
//! offsets index. This reader returns the records; see [`crate::DustPatch`].

use std::fmt;
use std::path::{Path, PathBuf};

use crate::DustPatch;

/// XCon magic as a little-endian writer stores it; the integer is
/// `0xCAFE_BABE` and the two words are swapped for a big-endian writer.
pub const MAGIC: u32 = 0xCAFE_BABE;
/// Bytes per `SpotList` record.
pub const RECORD_BYTES: usize = 16;

const LCC_TYPE: &[u8] = b"CaptureOne LCC";

/// The group names whose run of children ends at the next `0xFE`.
const GROUPS: [&[u8]; 10] = [
    b"Camera",
    b"RAW",
    b"Lens",
    b"GPS",
    b"Chroma",
    b"LightFalloff",
    b"SpotRem",
    b"SpotRemoval",
    b"ColumnGainAdjust",
    b"VerticalStripeSuppression",
];

/// Why a `.lcc` could not be read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DustLccError {
    /// The first chunk is not `XCon`, or the file is too short for one.
    NotContainer,
    /// The `XCon` magic is unknown or its version is older than 0.2.
    BadHeader,
    /// No `TYPE` chunk, or it is not `CaptureOne LCC`.
    NotLcc,
    /// The `VER` chunk is missing or truncated.
    MissingVersion,
    /// A chunk header, payload or `SpotList` record run is cut short.
    Truncated,
    /// Invalid or incomplete LightFalloff model.
    InvalidLightFalloff,
    /// A LightFalloff encoding whose payload layout has not been verified.
    UnsupportedLightFalloff,
}

impl fmt::Display for DustLccError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotContainer => write!(f, "not a Capture One XCon container"),
            Self::BadHeader => write!(f, "XCon header magic/version is not supported"),
            Self::NotLcc => write!(f, "TYPE is not \"CaptureOne LCC\""),
            Self::MissingVersion => write!(f, "VER chunk is missing"),
            Self::UnsupportedLightFalloff => write!(f, "unsupported LightFalloff encoding"),
            Self::InvalidLightFalloff => write!(f, "invalid LightFalloff model"),
            Self::Truncated => write!(f, "truncated .lcc chunk or record run"),
        }
    }
}

impl std::error::Error for DustLccError {}

/// The dust half of an `.lcc`: its patch records and their `SpotData` plane.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DustLcc {
    pub patches: Vec<DustPatch>,
    pub gain: Vec<u8>,
}

/// One six-byte LightFalloff node, before normalization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LightFalloffNode {
    pub luminance: u16,
    pub dx: i16,
    pub dy: i16,
}

/// The original 12- or 16-byte header and its row-major two-dimensional grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LightFalloff {
    pub header: Vec<u8>,
    pub columns: u16,
    pub rows: u16,
    pub nodes: Vec<LightFalloffNode>,
}

/// High-gain treatment selected by the caller from its runtime settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LightFalloffLimit {
    Hard,
    Soft,
    Uncapped,
}

/// A decoded, enabled Uniform Light correction prepared once per lens stage.
pub struct UniformLight<'a> {
    grid: &'a LightFalloff,
    origin: [f32; 2],
    step: [f32; 2],
    max_level: f32,
    strength: f32,
    limit: LightFalloffLimit,
}

impl<'a> UniformLight<'a> {
    /// Disabled, absent, invalid, or zero-strength profiles yield identity.
    pub fn prepare(
        grid: Option<&'a LightFalloff>,
        enabled: bool,
        strength: f32,
        limit: LightFalloffLimit,
    ) -> Option<Self> {
        if !enabled || !strength.is_finite() || strength <= 0.0 {
            return None;
        }
        let grid = grid?;
        let header = match grid.header.len() {
            12 => &grid.header[..],
            16 => &grid.header[2..14],
            _ => return None,
        };
        let [columns, rows] = [grid.columns, grid.rows].map(usize::from);
        let origin = [4, 6].map(|offset| f32::from(u16_at(header, offset, true)));
        let step = [8, 10].map(|offset| f32::from(u16_at(header, offset, true)));
        if columns < 2 || rows < 2 || step.contains(&0.0) || grid.nodes.len() != columns * rows {
            return None;
        }
        let max_level = f32::from(grid.nodes.iter().map(|node| node.luminance).max()?);
        Some(Self { grid, origin, step, max_level, strength, limit })
    }

    /// Tile origin and pixel are source-space coordinates before inverse scaling.
    pub fn gain(&self, tile_origin: [i32; 2], pixel: [u32; 2], inverse_scale: f32) -> f32 {
        if !inverse_scale.is_finite() || inverse_scale <= 0.0 {
            return 1.0;
        }
        let source = [0, 1].map(|axis| {
            inverse_scale * (tile_origin[axis] as f32 + pixel[axis] as f32)
        });
        self.source_gain(source)
    }

    fn source_gain(&self, source: [f32; 2]) -> f32 {
        let [columns, rows] = [self.grid.columns, self.grid.rows].map(usize::from);
        let [x, y] = [source[0] - self.origin[0], source[1] - self.origin[1]];
        let [step_x, step_y] = self.step;
        let (qx, qy) = (x / step_x, y / step_y);
        if !qx.is_finite() || !qy.is_finite() {
            return 1.0;
        }
        let ix = ((qx + 100.0).trunc() - 100.0).clamp(0.0, (columns - 2) as f32) as usize;
        let iy = ((qy + 100.0).trunc() - 100.0).clamp(0.0, (rows - 2) as f32) as usize;
        // The edge cell is clamped, but its interpolation weights still extrapolate.
        let (tx, ty) = (qx - ix as f32, qy - iy as f32);
        let value = |nx: usize, ny: usize| {
            let node = self.grid.nodes[ny * columns + nx];
            f32::from(node.luminance)
                + f32::from(node.dx) * (x - nx as f32 * step_x) / (2.0 * step_x)
                + f32::from(node.dy) * (y - ny as f32 * step_y) / (2.0 * step_y)
        };
        let top = value(ix, iy) * (1.0 - tx) + value(ix + 1, iy) * tx;
        let bottom = value(ix, iy + 1) * (1.0 - tx) + value(ix + 1, iy + 1) * tx;
        let field = top * (1.0 - ty) + bottom * ty;
        let ev = ((self.max_level - field) / 8192.0) * self.strength;
        let limited = if ev > 3.0 {
            match self.limit {
                LightFalloffLimit::Hard => 3.0,
                LightFalloffLimit::Soft if ev > 5.0 => 4.0,
                LightFalloffLimit::Soft => ev - 0.25 * (ev - 3.0) * (ev - 3.0),
                LightFalloffLimit::Uncapped => ev,
            }
        } else {
            ev
        };
        limited.exp2()
    }

    /// Applies the gain to linear RGB while preserving alpha.
    pub fn apply_rgba(&self, rgba: &mut [f32; 4], tile: [i32; 2], pixel: [u32; 2], scale: f32) {
        let gain = self.gain(tile, pixel, scale);
        rgba[..3].iter_mut().for_each(|channel| *channel *= gain);
    }

    /// Uses the same continuous full-resolution source coordinate as the sampler.
    pub(crate) fn apply_source_rgb(&self, rgb: [f32; 3], source: [f32; 2]) -> [f32; 3] {
        let gain = self.source_gain(source);
        rgb.map(|channel| channel * gain)
    }
}

/// Reads the inline little-endian LightFalloff grid used by the Windows renderer.
pub fn parse_light_falloff(bytes: &[u8]) -> Result<Option<LightFalloff>, DustLccError> {
    let (mut header, mut model) = (None, None);
    parse_chunks(bytes, |group, name, flags, payload, le| {
        if group != Some(b"LightFalloff") {
            return Ok(());
        }
        let target = match name {
            b"REF" => return Err(DustLccError::UnsupportedLightFalloff),
            b"Hdr" => &mut header,
            b"Model" => &mut model,
            _ => return Ok(()),
        };
        if !le || flags & 0xC0 != 0 {
            return Err(DustLccError::UnsupportedLightFalloff);
        }
        if target.replace(payload).is_some() {
            return Err(DustLccError::InvalidLightFalloff);
        }
        Ok(())
    })?;
    let (header, model) = match (header, model) {
        (None, None) => return Ok(None),
        (Some(header), Some(model)) => (header, model),
        _ => return Err(DustLccError::InvalidLightFalloff),
    };
    let grid = match header.len() {
        12 => header,
        16 => &header[2..14],
        _ => return Err(DustLccError::InvalidLightFalloff),
    };
    let (columns, rows) = (u16_at(grid, 0, true), u16_at(grid, 2, true));
    let (step_x, step_y) = (u16_at(grid, 8, true), u16_at(grid, 10, true));
    const NODE_BYTES: usize = 6;
    if columns < 2 || rows < 2 || step_x == 0 || step_y == 0
        || model.len() % NODE_BYTES != 0
        || model.len() / NODE_BYTES != usize::from(columns) * usize::from(rows)
    {
        return Err(DustLccError::InvalidLightFalloff);
    }
    let nodes = model.chunks_exact(NODE_BYTES).map(|node| LightFalloffNode {
        luminance: u16_at(node, 0, true),
        dx: i16_at(node, 2, true),
        dy: i16_at(node, 4, true),
    }).collect();
    Ok(Some(LightFalloff { header: header.to_vec(), columns, rows, nodes }))
}

/// Reads the `SpotRem` group of an `.lcc` byte image.
///
/// A valid profile without that group yields empty fields.
pub fn parse(bytes: &[u8]) -> Result<DustLcc, DustLccError> {
    let mut dust = DustLcc::default();
    parse_chunks(bytes, |group, name, _, payload, le| {
        if matches!(group, Some(b"SpotRem" | b"SpotRemoval")) {
            if name == b"SpotList" {
                dust.patches = decode_records(payload, le)?;
            } else if name == b"SpotData" {
                dust.gain = payload.to_vec();
            }
        }
        Ok(())
    })?;
    Ok(dust)
}

fn parse_chunks<'a>(
    bytes: &'a [u8],
    mut visit: impl FnMut(Option<&[u8]>, &[u8], u8, &'a [u8], bool) -> Result<(), DustLccError>,
) -> Result<(), DustLccError> {
    if !bytes.starts_with(b"\x04XCon") {
        return Err(DustLccError::NotContainer);
    }
    let mut pos = 0usize;
    let mut groups: Vec<&[u8]> = Vec::new();
    let mut first = true;
    let mut endian: Option<bool> = None;
    let mut kind_ok = false;
    let mut version_ok = false;

    while pos < bytes.len() {
        match bytes[pos] {
            0xFF => {
                pos += 1;
                continue;
            }
            0xFE => {
                pos += 1;
                read_varint(bytes, &mut pos)?;
                groups.pop();
                continue;
            }
            _ => {}
        }
        let name_len = read_varint(bytes, &mut pos)?;
        let name = bytes.get(pos..pos + name_len).ok_or(DustLccError::Truncated)?;
        pos += name_len;
        let flags = *bytes.get(pos).ok_or(DustLccError::Truncated)?;
        pos += 2;
        let payload = if flags & 1 == 0 {
            let size = read_varint(bytes, &mut pos)?;
            let body = bytes.get(pos..pos + size).ok_or(DustLccError::Truncated)?;
            pos += size;
            body
        } else {
            &bytes[0..0]
        };

        if first {
            first = false;
            if name != b"XCon" || payload.len() < 6 {
                return Err(DustLccError::NotContainer);
            }
            // A little-endian writer emits `BE BA FE CA`; a big-endian one
            // emits the bytes of the same integer, `CA FE BA BE`.
            endian = Some(match [payload[0], payload[1], payload[2], payload[3]] {
                [0xBE, 0xBA, 0xFE, 0xCA] => true,
                [0xCA, 0xFE, 0xBA, 0xBE] => false,
                _ => return Err(DustLccError::BadHeader),
            });
            if payload[4] != 0 || payload[5] < 2 {
                return Err(DustLccError::BadHeader);
            }
        }

        if name == b"TYPE" {
            if payload != LCC_TYPE {
                return Err(DustLccError::NotLcc);
            }
            kind_ok = true;
        } else if name == b"VER" {
            if payload.len() < 2 {
                return Err(DustLccError::MissingVersion);
            }
            version_ok = true;
        }
        visit(groups.last().copied(), name, flags, payload, endian.unwrap_or(true))?;

        if flags & 1 != 0 || GROUPS.contains(&name) {
            groups.push(name);
        }
    }

    if !kind_ok {
        return Err(DustLccError::NotLcc);
    }
    if !version_ok {
        return Err(DustLccError::MissingVersion);
    }
    Ok(())
}

/// Where Capture One's Session writer puts an LCC: `<raw name>.<lcc id>.lcc`
/// beside the document (`Session.LccFile`). Catalog documents resolve the id
/// through the catalog's sidecar store instead.
pub fn sidecar_path(settings_dir: &Path, raw_file_name: &str, lcc_id: &str) -> PathBuf {
    settings_dir.join(format!("{raw_file_name}.{lcc_id}.lcc"))
}

/// `ZSIDECAR.ZTYPE` for a lens cast calibration (0 is ICC).
pub const SIDECAR_TYPE_LCC: i64 = 1;

/// The `ZSIDECAR` fields that decide where the file lives.
#[derive(Clone, Copy, Debug)]
pub struct SidecarRow<'a> {
    /// `ZNAME`, the string `ZLENSLCC` stores; a bare name, not a path.
    pub name: &'a str,
    /// `ZTYPE`: 0 ICC, 1 LCC.
    pub kind: i64,
}

/// Resolves a catalog's LCC sidecar to its file.
///
/// `SharedSidecarsDirectory.MakeAbsoluteSidecarPath` combines the catalog
/// folder, `Adjustments/LCC`, the `ZSIDECAR.ZNAME` file name and the `.lcc`
/// extension; the same directory is created by `CatalogFolderPaths`.
pub fn resolve_lcc_path(catalog_folder: &Path, row: SidecarRow<'_>) -> Option<PathBuf> {
    if row.kind != SIDECAR_TYPE_LCC
        || row.name.is_empty()
        || row.name == "."
        || row.name == ".."
        || row.name.contains(['/', '\\', ':', '\0'])
    {
        return None;
    }
    let file = if row.name.ends_with(".lcc") { row.name.to_owned() } else { format!("{}.lcc", row.name) };
    Some(catalog_folder.join("Adjustments").join("LCC").join(file))
}

/// A small little-endian `.lcc` with one `SpotRem` group, for tests.
#[cfg(test)]
pub(crate) fn lcc_with(records: &[DustPatch], plane: &[u8]) -> Vec<u8> {
    fn varint(mut value: usize) -> Vec<u8> {
        if value < 0x40 {
            return vec![value as u8];
        }
        let mut groups = Vec::new();
        while value != 0 {
            groups.push((value & 0x3F) as u8);
            value >>= 6;
        }
        groups.reverse();
        let last = groups.len() - 1;
        groups[0] |= 0x40;
        groups[last] |= 0x40;
        groups
    }
    fn chunk(name: &[u8], group: bool, payload: &[u8]) -> Vec<u8> {
        let mut out = varint(name.len());
        out.extend_from_slice(name);
        out.push(u8::from(group));
        out.push(0);
        if !group {
            out.extend_from_slice(&varint(payload.len()));
            out.extend_from_slice(payload);
        }
        out
    }
    let mut list = Vec::new();
    for p in records {
        list.extend_from_slice(&p.x.to_le_bytes());
        list.extend_from_slice(&p.y.to_le_bytes());
        list.extend_from_slice(&p.w.to_le_bytes());
        list.extend_from_slice(&p.h.to_le_bytes());
        list.extend_from_slice(&p.offset.to_le_bytes());
        list.extend_from_slice(&p.bias.to_le_bytes());
        list.extend_from_slice(&p.step.to_le_bytes());
    }
    let mut spot_rem = chunk(b"SpotRem", true, &[]);
    spot_rem.extend(chunk(b"SpotList", false, &list));
    spot_rem.extend(chunk(b"SpotData", false, plane));
    spot_rem.extend_from_slice(&[0xFE, 0x00]);

    let mut out = chunk(b"XCon", false, &[0xBE, 0xBA, 0xFE, 0xCA, 0, 2]);
    out.extend(chunk(b"TYPE", false, LCC_TYPE));
    out.extend(chunk(b"VER", false, &[6, 0]));
    out.extend(spot_rem);
    out
}

fn decode_records(payload: &[u8], le: bool) -> Result<Vec<DustPatch>, DustLccError> {
    if payload.len() % RECORD_BYTES != 0 {
        return Err(DustLccError::Truncated);
    }
    Ok(payload
        .chunks_exact(RECORD_BYTES)
        .map(|r| DustPatch {
            x: u16_at(r, 0, le),
            y: u16_at(r, 2, le),
            w: u16_at(r, 4, le),
            h: u16_at(r, 6, le),
            offset: u32_at(r, 8, le),
            bias: i16_at(r, 12, le),
            step: u16_at(r, 14, le),
        })
        .collect())
}

/// 6-bit big-endian groups; the first and last byte carry `0x40`.
fn read_varint(buf: &[u8], pos: &mut usize) -> Result<usize, DustLccError> {
    let byte = *buf.get(*pos).ok_or(DustLccError::Truncated)?;
    *pos += 1;
    if byte < 0x40 {
        return Ok(byte as usize);
    }
    if byte >= 0x80 {
        return Err(DustLccError::Truncated);
    }
    let mut value = (byte & 0x3F) as usize;
    for _ in 0..9 {
        let byte = *buf.get(*pos).ok_or(DustLccError::Truncated)?;
        *pos += 1;
        if byte >= 0x80 {
            return Err(DustLccError::Truncated);
        }
        value = (value << 6) | (byte & 0x3F) as usize;
        if byte & 0x40 != 0 {
            return Ok(value);
        }
    }
    Err(DustLccError::Truncated)
}

fn u16_at(b: &[u8], i: usize, le: bool) -> u16 {
    let raw = [b[i], b[i + 1]];
    if le { u16::from_le_bytes(raw) } else { u16::from_be_bytes(raw) }
}

fn i16_at(b: &[u8], i: usize, le: bool) -> i16 {
    u16_at(b, i, le) as i16
}

fn u32_at(b: &[u8], i: usize, le: bool) -> u32 {
    let raw = [b[i], b[i + 1], b[i + 2], b[i + 3]];
    if le { u32::from_le_bytes(raw) } else { u32::from_be_bytes(raw) }
}

impl crate::LensCorrection {
    /// Replaces `dust_patches`/`dust_gain` with the fields of `bytes`.
    ///
    /// The `ZLENSDUSTREMOVALENABLED` gate is left alone: a profile loaded while
    /// the gate is off stays inert until the variant turns it on.
    pub fn load_dust_lcc(&mut self, bytes: &[u8]) -> Result<usize, DustLccError> {
        self.uniform_light_grid = parse_light_falloff(bytes).ok().flatten().map(std::sync::Arc::new);
        let dust = parse(bytes)?;
        let count = dust.patches.len();
        self.dust_patches = dust.patches;
        self.dust_gain = dust.gain;
        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AdjustmentSet;

    fn varint(mut value: usize) -> Vec<u8> {
        if value < 0x40 {
            return vec![value as u8];
        }
        let mut groups = Vec::new();
        while value != 0 {
            groups.push((value & 0x3F) as u8);
            value >>= 6;
        }
        groups.reverse();
        let last = groups.len() - 1;
        groups[0] |= 0x40;
        groups[last] |= 0x40;
        groups
    }

    fn chunk(name: &[u8], group: bool, payload: &[u8]) -> Vec<u8> {
        let mut out = varint(name.len());
        out.extend_from_slice(name);
        out.push(u8::from(group));
        out.push(0);
        if !group {
            out.extend_from_slice(&varint(payload.len()));
            out.extend_from_slice(payload);
        }
        out
    }

    fn grouped(name: &[u8], children: &[Vec<u8>]) -> Vec<u8> {
        let mut out = chunk(name, true, &[]);
        for child in children {
            out.extend_from_slice(child);
        }
        out.push(0xFE);
        out.push(0);
        out
    }

    fn record(x: u16, y: u16, w: u16, h: u16, offset: u32, bias: i16, step: u16, le: bool) -> Vec<u8> {
        let pair = |value: u16| if le { value.to_le_bytes() } else { value.to_be_bytes() };
        let quad = |value: u32| if le { value.to_le_bytes() } else { value.to_be_bytes() };
        let mut out = Vec::with_capacity(RECORD_BYTES);
        for value in [x, y, w, h] {
            out.extend_from_slice(&pair(value));
        }
        out.extend_from_slice(&quad(offset));
        out.extend_from_slice(&pair(bias as u16));
        out.extend_from_slice(&pair(step));
        out
    }

    fn profile(le: bool, children: &[Vec<u8>]) -> Vec<u8> {
        let magic: [u8; 6] = if le {
            [0xBE, 0xBA, 0xFE, 0xCA, 0, 2]
        } else {
            [0xCA, 0xFE, 0xBA, 0xBE, 0, 2]
        };
        let mut out = chunk(b"XCon", false, &magic);
        out.extend(chunk(b"TYPE", false, LCC_TYPE));
        out.extend(chunk(b"VER", false, &[6, 0]));
        for child in children {
            out.extend_from_slice(child);
        }
        out
    }

    fn light_falloff_profile(le: bool, header: &[u8], model: &[u8]) -> Vec<u8> {
        profile(le, &[grouped(b"LightFalloff", &[
            chunk(b"Hdr", false, header),
            chunk(b"Model", false, model),
        ])])
    }

    #[test]
    fn reads_legacy_and_current_light_falloff_headers() {
        let legacy: Vec<_> = [3u16, 2, 12, 18, 24, 32].into_iter().flat_map(u16::to_le_bytes).collect();
        let current = [&[16, 0], legacy.as_slice(), &[1, 0]].concat();
        let mut model = [0; 36];
        model[30..].copy_from_slice(&[0xFE, 0xFF, 0xFD, 0xFF, 0x00, 0x80]);
        for header in [&legacy, &current] {
            let bytes = light_falloff_profile(true, header, &model);
            let parsed = parse_light_falloff(&bytes).unwrap().unwrap();
            assert_eq!(&parsed.header, header);
            assert_eq!((parsed.columns, parsed.rows, parsed.nodes.len()), (3, 2, 6));
            assert_eq!(parsed.nodes[0], LightFalloffNode { luminance: 0, dx: 0, dy: 0 });
            assert_eq!(parsed.nodes[5], LightFalloffNode { luminance: 65_534, dx: -3, dy: i16::MIN });
        }
        assert_eq!(parse_light_falloff(&dust_profile(true)), Ok(None));
    }

    #[test]
    fn rejects_bad_grid_and_unsupported_references_without_breaking_dust() {
        let header = [2, 0, 2, 0, 0, 0, 0, 0, 1, 0, 1, 0];
        for (bad_header, len) in [(header, 23), (header, 30), ([0; 12], 24)] {
            let bytes = light_falloff_profile(true, &bad_header, &[0; 30][..len]);
            assert_eq!(parse_light_falloff(&bytes), Err(DustLccError::InvalidLightFalloff));
            assert_eq!(parse(&bytes), Ok(DustLcc::default()));
        }
        let bytes = profile(true, &[grouped(b"LightFalloff", &[
            chunk(b"Hdr", false, &header),
            chunk(b"REF", false, &[1]),
        ])]);
        assert_eq!(parse_light_falloff(&bytes), Err(DustLccError::UnsupportedLightFalloff));
        assert_eq!(parse(&bytes), Ok(DustLcc::default()));
        let bytes = light_falloff_profile(false, &header, &[0; 24]);
        assert_eq!(parse_light_falloff(&bytes), Err(DustLccError::UnsupportedLightFalloff));
    }

    fn dust_profile(le: bool) -> Vec<u8> {
        profile(
            le,
            &[grouped(b"SpotRem", &[
                chunk(b"Hdr", false, &[0; 12]),
                chunk(b"SpotList", false, &[
                    record(10, 20, 30, 40, 7, -3, 9, le),
                    record(1, 2, 3, 4, 5, 6, 7, le),
                ].concat()),
                chunk(b"SpotData", false, &[0; 1200]),
            ])],
        )
    }

    #[test]
    fn parses_spot_list_records() {
        let got = parse(&dust_profile(true)).unwrap();
        assert_eq!(got.patches.len(), 2);
        assert_eq!(got.gain.len(), 1200);
        assert_eq!(
            got.patches[0],
            DustPatch { x: 10, y: 20, w: 30, h: 40, offset: 7, bias: -3, step: 9 }
        );
        assert_eq!(
            got.patches[1],
            DustPatch { x: 1, y: 2, w: 3, h: 4, offset: 5, bias: 6, step: 7 }
        );
    }

    #[test]
    fn big_endian_profile_reads_its_own_byte_order() {
        assert_eq!(parse(&dust_profile(false)).unwrap(), parse(&dust_profile(true)).unwrap());
    }

    #[test]
    fn rejects_non_lcc_and_truncated_input() {
        assert_eq!(parse(b"CaptureOne LCC"), Err(DustLccError::NotContainer));
        let bytes = dust_profile(true);
        assert_eq!(parse(&bytes[..bytes.len() - 8]), Err(DustLccError::Truncated));
        let mut no_type = chunk(b"XCon", false, &[0xBE, 0xBA, 0xFE, 0xCA, 0, 2]);
        no_type.extend(chunk(b"TYPE", false, b"Something Else"));
        no_type.extend(chunk(b"VER", false, &[6, 0]));
        assert_eq!(parse(&no_type), Err(DustLccError::NotLcc));
    }

    #[test]
    fn lcc_patch_brightens_by_the_stored_curve() {
        let bytes = profile(true, &[grouped(b"SpotRem", &[
            chunk(b"SpotList", false, &record(0, 0, 4, 4, 0, 0, 8, true)),
        ])]);

        let mut adj = AdjustmentSet::default();
        adj.lens.dust_removal_enabled = true;
        assert_eq!(adj.lens.load_dust_lcc(&bytes).unwrap(), 1);

        let mut buf = vec![128u8; 16 * 16 * 4];
        crate::mock::detail::post_process(&mut buf, 16, 16, &adj, 1.0);

        let gain = 4096.0 * 2f32.powf(128.0 * 8.0 / 4096.0);
        let want = 128.0 * gain / 4096.0;
        assert!((buf[0] as f32 - want).abs() <= 1.0, "got {} want {want}", buf[0]);
        assert_eq!(buf[(12 * 16 + 12) * 4], 128, "outside the patch is untouched");
    }

    #[test]
    fn profile_without_dust_keeps_the_frame_byte_identical() {
        let bytes = profile(true, &[grouped(b"LightFalloff", &[
            chunk(b"Model", false, &[0; 8]),
        ])]);
        let mut adj = AdjustmentSet::default();
        adj.lens.dust_removal_enabled = true;
        assert_eq!(adj.lens.load_dust_lcc(&bytes).unwrap(), 0);

        let mut buf = vec![90u8; 8 * 8 * 4];
        let before = buf.clone();
        crate::mock::detail::post_process(&mut buf, 8, 8, &adj, 1.0);
        assert_eq!(buf, before);
    }

    fn single_patch_profile(plane: &[u8], bias: i16) -> Vec<u8> {
        profile(true, &[grouped(b"SpotRem", &[
            chunk(b"SpotList", false, &record(0, 0, 4, 4, 0, bias, 8, true)),
            chunk(b"SpotData", false, plane),
        ])])
    }

    #[test]
    fn spot_data_plane_drives_the_gain_per_pixel() {
        let mut plane = [0u8; 16];
        plane[5] = 10;
        let bytes = single_patch_profile(&plane, 128);

        let mut adj = AdjustmentSet::default();
        adj.lens.dust_removal_enabled = true;
        adj.lens.load_dust_lcc(&bytes).unwrap();
        assert_eq!(adj.lens.dust_gain, plane.to_vec());

        let mut buf = vec![128u8; 8 * 8 * 4];
        crate::mock::detail::post_process(&mut buf, 8, 8, &adj, 1.0);

        let pixel = |x: usize, y: usize| buf[(y * 8 + x) * 4] as f32;
        let bright = 128.0 * 2f32.powf(10.0 * 8.0 / 4096.0);
        assert!((pixel(1, 1) - bright).abs() <= 1.0, "got {}", pixel(1, 1));
        assert_eq!(pixel(0, 0), 128.0, "a zero byte is a neutral delta");
        assert_eq!(pixel(4, 0), 128.0, "outside the patch is untouched");
    }

    #[test]
    fn an_all_zero_plane_leaves_the_frame_byte_identical() {
        let bytes = single_patch_profile(&[0u8; 16], 0);
        let mut adj = AdjustmentSet::default();
        adj.lens.dust_removal_enabled = true;
        adj.lens.load_dust_lcc(&bytes).unwrap();

        let mut buf = vec![90u8; 8 * 8 * 4];
        let before = buf.clone();
        crate::mock::detail::post_process(&mut buf, 8, 8, &adj, 1.0);
        assert_eq!(buf, before);
    }

    #[test]
    fn bias_splits_the_byte_into_positive_and_negative_deltas() {
        let mut plane = [0u8; 16];
        plane[0] = 128;
        plane[1] = 129;
        let bytes = single_patch_profile(&plane, 128);

        let mut adj = AdjustmentSet::default();
        adj.lens.dust_removal_enabled = true;
        adj.lens.load_dust_lcc(&bytes).unwrap();

        let mut buf = vec![128u8; 8 * 8 * 4];
        crate::mock::detail::post_process(&mut buf, 8, 8, &adj, 1.0);

        let pixel = |x: usize| buf[x * 4] as f32;
        assert!((pixel(0) - 128.0 * 2f32.powf(128.0 * 8.0 / 4096.0)).abs() <= 1.0);
        assert!(pixel(0) > 128.0, "byte == bias is the largest positive delta");
        assert!((pixel(1) - 128.0 * 2f32.powf(-127.0 * 8.0 / 4096.0)).abs() <= 1.0);
        assert!(pixel(1) < 128.0, "byte > bias flips to a negative delta");
    }

    #[test]
    fn catalog_sidecar_rows_resolve_into_adjustments_lcc() {
        let folder = Path::new("C:/Catalog");
        let expected = folder.join("Adjustments").join("LCC").join("0a1b2c3d.lcc");
        assert_eq!(
            resolve_lcc_path(folder, SidecarRow { name: "0a1b2c3d", kind: SIDECAR_TYPE_LCC }),
            Some(expected)
        );
        assert!(resolve_lcc_path(folder, SidecarRow { name: "prof", kind: 0 }).is_none());
        assert!(resolve_lcc_path(folder, SidecarRow { name: "", kind: 1 }).is_none());
    }
}
