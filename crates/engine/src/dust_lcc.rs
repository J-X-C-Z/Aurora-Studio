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

use std::collections::HashMap;
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

/// Why [`load_light_falloff`] could not return a calibration.
#[derive(Debug)]
pub enum LccLoadError {
    /// The file could not be read.
    Read { path: PathBuf, source: std::io::Error },
    /// The file exists but its XCon/LightFalloff payload is invalid.
    Parse(DustLccError),
    /// The file is a valid `.lcc` but carries no Uniform Light grid.
    NoLightFalloff,
}

impl fmt::Display for LccLoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, source } => write!(f, "reading {}: {source}", path.display()),
            Self::Parse(error) => write!(f, "{error}"),
            Self::NoLightFalloff => write!(f, "the .lcc has no LightFalloff grid"),
        }
    }
}

impl std::error::Error for LccLoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { source, .. } => Some(source),
            Self::Parse(error) => Some(error),
            Self::NoLightFalloff => None,
        }
    }
}

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

/// The kernel's `mode` argument: how the calibrated EV is limited.
///
/// Capture One's own OpenCL consumer (`LensLightLCC`, kernel `z5523e58be8`)
/// switches on `mode`: 0 applies the soft knee, 1 pins at 3 EV, every other
/// value leaves the EV alone. There is **no per-variant column or SDK field**
/// for it — the catalog only stores the gate, the amount and the LCC id — so
/// the only persisted source is the calibration file's `Hdr` flags word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LightFalloffMode {
    #[default]
    Soft,
    Hard,
    Uncapped,
}

impl LightFalloffMode {
    /// `Hdr` flags bit 0 selects the limiter: 1 keeps the soft knee, 0 hard
    /// cuts at 3 EV (`mode = 1 - (flags & 1)`).
    ///
    /// Measured against Capture One's own renderer: a 6 EV calibration whose
    /// flags word is 1 reaches 4 EV at the corner, clearing bit 0 lands on the
    /// 3 EV plateau and matches the flagged file at amount 50.4. Both renders
    /// are identical below 3 EV. `LCC_CREATE_TECHNICAL_CAM` writes the same
    /// flags word, so it is not a second knob.
    pub fn from_lcc_flags(flags: u16) -> Self {
        if flags & 1 == 1 {
            Self::Soft
        } else {
            Self::Hard
        }
    }

    /// The kernel's integer argument.
    pub fn kernel_index(self) -> i32 {
        match self {
            Self::Soft => 0,
            Self::Hard => 1,
            Self::Uncapped => 2,
        }
    }

    /// Anything outside 0..=2 is treated as "not recognised" and falls back to
    /// the soft knee, which is what an ordinary variant gets. The kernel itself
    /// would leave the EV unlimited for such a value; the fallback is a
    /// conservative choice and is logged.
    pub fn from_kernel_index(index: i32) -> Self {
        match index {
            0 => Self::Soft,
            1 => Self::Hard,
            2 => Self::Uncapped,
            other => {
                log::warn!(
                    "uniform light: unknown kernel mode {other}, falling back to the soft cap"
                );
                Self::Soft
            }
        }
    }

    fn limit(self) -> LightFalloffLimit {
        match self {
            Self::Soft => LightFalloffLimit::Soft,
            Self::Hard => LightFalloffLimit::Hard,
            Self::Uncapped => LightFalloffLimit::Uncapped,
        }
    }
}

impl LightFalloff {
    /// The `Hdr` flags word. A 12-byte legacy header has no such field.
    pub fn flags(&self) -> Option<u16> {
        match self.header.len() {
            16 => Some(u16_at(&self.header, 14, true)),
            _ => None,
        }
    }

    /// The limiter this calibration asks for. Legacy headers, and anything
    /// C1 has not written a mode into, keep the soft knee.
    pub fn mode(&self) -> LightFalloffMode {
        self.flags()
            .map(LightFalloffMode::from_lcc_flags)
            .unwrap_or_default()
    }
}

/// Maps Capture One's 0..120 Light Falloff slider onto the kernel's strength.
///
/// The shipped consumer (`LensLightLCC`, kernel `z5523e58be8` inside
/// `ImgCoreDll.dll`) computes `ev = (max_level - field)/8192 * strength` and
/// `gain = exp2(ev)`, so the slider blends **in EV**, not toward identity:
/// `gain(amount) = gain(100) ^ (amount/100)`. A measured render confirms it —
/// amount 50 gives `sqrt` of the full-strength gain at every radius, while a
/// linear gain blend would be ~1.4% brighter at the corner.
///
/// The second half of the result is the cap policy the mode selects, so a
/// caller cannot accidentally pair one mode's strength with another's limit.
pub fn strength_from_amount(amount: f32, mode: LightFalloffMode) -> (f32, LightFalloffLimit) {
    ((amount / 100.0).clamp(0.0, 1.2), mode.limit())
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
    let (mut header_ref, mut model_ref) = (None, None);
    let mut bins: HashMap<u32, &[u8]> = HashMap::new();
    let mut dat: Option<u32> = None;
    parse_chunks(bytes, |groups, name, flags, payload, le| {
        if name == b"DAT" {
            dat = varint_value(payload);
            return Ok(());
        }
        if name == b"BIN" {
            if let Some(id) = dat.take() {
                bins.insert(id, payload);
            }
            return Ok(());
        }
        if !groups.iter().any(|group| *group == b"LightFalloff".as_slice()) {
            return Ok(());
        }
        if !le {
            return Err(DustLccError::UnsupportedLightFalloff);
        }
        let parent = groups.last().copied();
        match name {
            b"Hdr" if parent == Some(b"LightFalloff".as_slice()) && flags & 1 == 0 => {
                if header.replace(payload).is_some() {
                    return Err(DustLccError::InvalidLightFalloff);
                }
            }
            b"Model" if parent == Some(b"LightFalloff".as_slice()) && flags & 1 == 0 => {
                if model.replace(payload).is_some() {
                    return Err(DustLccError::InvalidLightFalloff);
                }
            }
            b"REF" if parent == Some(b"Hdr".as_slice()) => header_ref = varint_value(payload),
            b"REF" if parent == Some(b"Model".as_slice()) => model_ref = varint_value(payload),
            _ => {}
        }
        Ok(())
    })?;
    let header = header.or_else(|| header_ref.and_then(|id| bins.get(&id).copied()));
    let model = model.or_else(|| model_ref.and_then(|id| bins.get(&id).copied()));
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

/// Reads and parses a `.lcc` file. A missing file is [`LccLoadError::Read`],
/// malformed bytes are [`LccLoadError::Parse`], and a valid profile without a
/// Uniform Light grid is [`LccLoadError::NoLightFalloff`].
pub fn load_light_falloff(path: &Path) -> Result<LightFalloff, LccLoadError> {
    let bytes = std::fs::read(path).map_err(|source| LccLoadError::Read { path: path.to_path_buf(), source })?;
    match parse_light_falloff(&bytes) {
        Ok(Some(grid)) => Ok(grid),
        Ok(None) => Err(LccLoadError::NoLightFalloff),
        Err(error) => Err(LccLoadError::Parse(error)),
    }
}

fn encode_varint(mut value: usize) -> Vec<u8> {
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

fn encode_chunk(name: &[u8], group: bool, payload: &[u8]) -> Vec<u8> {
    let mut out = encode_varint(name.len());
    out.extend_from_slice(name);
    out.push(u8::from(group));
    out.push(0);
    if !group {
        out.extend_from_slice(&encode_varint(payload.len()));
        out.extend_from_slice(payload);
    }
    out
}

fn encode_group(name: &[u8], children: &[Vec<u8>]) -> Vec<u8> {
    let mut out = encode_chunk(name, true, &[]);
    for child in children {
        out.extend_from_slice(child);
    }
    out.push(0xFE);
    out.push(0);
    out
}

/// Encodes a decoded LightFalloff grid as a standalone `.lcc` container.
///
/// The output keeps the original 12/16-byte `Hdr` and writes the grid as the
/// same direct `LightFalloff/Hdr` + `LightFalloff/Model` pair the reader accepts.
/// It is deliberately limited to the Uniform Light payload; dust records are
/// not synthesized.
pub fn write_light_falloff(grid: &LightFalloff) -> Result<Vec<u8>, DustLccError> {
    let node_count = usize::from(grid.columns) * usize::from(grid.rows);
    if !matches!(grid.header.len(), 12 | 16)
        || grid.columns < 2
        || grid.rows < 2
        || grid.nodes.len() != node_count
    {
        return Err(DustLccError::InvalidLightFalloff);
    }
    let mut model = Vec::with_capacity(node_count * 6);
    for node in &grid.nodes {
        model.extend_from_slice(&node.luminance.to_le_bytes());
        model.extend_from_slice(&node.dx.to_le_bytes());
        model.extend_from_slice(&node.dy.to_le_bytes());
    }
    let mut out = encode_chunk(b"XCon", false, &[0xBE, 0xBA, 0xFE, 0xCA, 0, 2]);
    out.extend(encode_chunk(b"TYPE", false, LCC_TYPE));
    out.extend(encode_chunk(b"VER", false, &[6, 0]));
    out.extend(encode_group(
        b"LightFalloff",
        &[
            encode_chunk(b"Hdr", false, &grid.header),
            encode_chunk(b"Model", false, &model),
        ],
    ));
    Ok(out)
}

/// Reads the `SpotRem` group of an `.lcc` byte image.
///
/// A valid profile without that group yields empty fields.
pub fn parse(bytes: &[u8]) -> Result<DustLcc, DustLccError> {
    let (mut patches, mut gain) = (None, None);
    let (mut list_ref, mut data_ref) = (None, None);
    let mut bins: HashMap<u32, &[u8]> = HashMap::new();
    let mut dat: Option<u32> = None;
    let mut le = true;
    parse_chunks(bytes, |groups, name, flags, payload, endian| {
        le = endian;
        if name == b"DAT" {
            dat = varint_value(payload);
            return Ok(());
        }
        if name == b"BIN" {
            if let Some(id) = dat.take() {
                bins.insert(id, payload);
            }
            return Ok(());
        }
        if !groups.iter().any(|group| {
            *group == b"SpotRem".as_slice() || *group == b"SpotRemoval".as_slice()
        }) {
            return Ok(());
        }
        let parent = groups.last().copied();
        match name {
            b"SpotList" if flags & 1 == 0 => patches = Some(decode_records(payload, endian)?),
            b"SpotData" if flags & 1 == 0 => gain = Some(payload.to_vec()),
            b"REF" if parent == Some(b"SpotList".as_slice()) => list_ref = varint_value(payload),
            b"REF" if parent == Some(b"SpotData".as_slice()) => data_ref = varint_value(payload),
            _ => {}
        }
        Ok(())
    })?;
    if patches.is_none() {
        if let Some(payload) = list_ref.and_then(|id| bins.get(&id).copied()) {
            patches = Some(decode_records(payload, le)?);
        }
    }
    if gain.is_none() {
        gain = data_ref.and_then(|id| bins.get(&id).map(|payload| payload.to_vec()));
    }
    Ok(DustLcc { patches: patches.unwrap_or_default(), gain: gain.unwrap_or_default() })
}

fn parse_chunks<'a>(
    bytes: &'a [u8],
    mut visit: impl FnMut(&[&'a [u8]], &[u8], u8, &'a [u8], bool) -> Result<(), DustLccError>,
) -> Result<(), DustLccError> {
    if !bytes.starts_with(b"\x04XCon") {
        return Err(DustLccError::NotContainer);
    }
    let mut pos = 0usize;
    let mut groups: Vec<(&'a [u8], usize)> = Vec::new();
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
                let marker = pos;
                pos += 1;
                skip_padding(bytes, &mut pos);
                // `FE` is followed by the byte length of the block it closes,
                // counted from the block's first byte through the `FE`.
                let length = read_varint(bytes, &mut pos)?;
                if length == 0 {
                    groups.pop();
                } else {
                    let start = marker.saturating_sub(length.saturating_sub(1));
                    while groups.last().is_some_and(|(_, section)| *section >= start) {
                        groups.pop();
                    }
                }
                continue;
            }
            _ => {}
        }
        let chunk_start = pos;
        let name_len = read_varint(bytes, &mut pos)?;
        let name = bytes.get(pos..pos + name_len).ok_or(DustLccError::Truncated)?;
        pos += name_len;
        let flags = *bytes.get(pos).ok_or(DustLccError::Truncated)?;
        pos += 2;
        let payload = if flags & 1 == 0 {
            skip_padding(bytes, &mut pos);
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
        let parents: Vec<&[u8]> = groups.iter().map(|(group, _)| *group).collect();
        visit(&parents, name, flags, payload, endian.unwrap_or(true))?;

        if flags & 1 != 0 || GROUPS.contains(&name) {
            groups.push((name, chunk_start));
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

fn skip_padding(bytes: &[u8], pos: &mut usize) {
    while bytes.get(*pos) == Some(&0xFF) {
        *pos += 1;
    }
}

fn varint_value(bytes: &[u8]) -> Option<u32> {
    let mut pos = 0;
    read_varint(bytes, &mut pos).ok().and_then(|value| u32::try_from(value).ok())
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
        let grid = parse_light_falloff(bytes).ok().flatten().map(std::sync::Arc::new);
        let dust = parse(bytes)?;
        self.uniform_light_grid = grid;
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
    fn rejects_bad_grid_and_missing_references_without_breaking_dust() {
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
        assert_eq!(parse_light_falloff(&bytes), Err(DustLccError::InvalidLightFalloff));
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

    /// `real-lightfalloff-24x16.lcc` is a prefix of a profile that Capture One
    /// itself wrote through `ICP_CreateLCCBuffer` from a Sony A7 II ARW. It is
    /// not a hand-built test vector: the LightFalloff `Hdr` is a direct chunk,
    /// the `Model` is an `REF`/`DAT`/`BIN` block, and C1's writer pads the
    /// lengths with `0xFF`.
    /// Read at run time: the fixture is git-ignored, so `include_bytes!` would
    /// make a clean checkout fail to compile.
    fn real_lcc() -> Option<Vec<u8>> {
        crate::testdata::bytes("testdata/real-lightfalloff-24x16.lcc")
    }

    #[test]
    fn real_capture_one_profile_resolves_the_referenced_24x16_grid() {
        let Some(real_lcc) = real_lcc() else { return };
        let grid = parse_light_falloff(&real_lcc).unwrap().unwrap();
        assert_eq!((grid.columns, grid.rows), (24, 16));
        assert_eq!(grid.header.len(), 16);
        assert_eq!(grid.nodes.len(), 24 * 16);
        assert_eq!(grid.nodes[0], LightFalloffNode { luminance: 44_209, dx: 679, dy: 1_239 });
        assert_eq!(grid.nodes[23], LightFalloffNode { luminance: 13_310, dx: -1_307, dy: 420 });
        assert_eq!(
            grid.nodes[23 + 15 * 24],
            LightFalloffNode { luminance: 15_888, dx: -831, dy: -686 }
        );
        assert_eq!(
            grid.nodes[12 + 8 * 24],
            LightFalloffNode { luminance: 33_470, dx: -1_181, dy: -308 }
        );
        assert_eq!(parse(&real_lcc), Ok(DustLcc::default()), "the prefix has no SpotRemoval");
    }

    #[test]
    fn real_capture_one_uniform_light_gains_match_the_reference() {
        let Some(real_lcc) = real_lcc() else { return };
        let grid = parse_light_falloff(&real_lcc).unwrap().unwrap();
        let stage = UniformLight::prepare(Some(&grid), true, 1.0, LightFalloffLimit::Uncapped)
            .expect("a real grid and a positive strength prepare a consumer");
        let gains = [
            ([124, 124], 1.949_046_560),
            ([5_874, 124], 26.623_330_247),
            ([124, 3_874], 1.583_332_076),
            ([5_874, 3_874], 21.405_681_682),
            ([3_124, 2_124], 4.835_555_524),
            ([3_000, 2_000], 4.538_449_314),
            ([1_124, 1_000], 1.228_025_630),
        ];
        for (source, expected) in gains {
            let got = stage.gain(source, [0, 0], 1.0);
            assert!(
                (got - expected).abs() <= expected * 1e-5,
                "gain at {source:?}: got {got}, want {expected}"
            );
            let rgb = stage.apply_source_rgb([0.125, 0.25, 0.75], source.map(|value| value as f32));
            for (channel, value) in rgb.into_iter().enumerate() {
                let want = [0.125, 0.25, 0.75][channel] * expected;
                assert!((value - want).abs() <= want * 1e-5, "{source:?} channel {channel}");
            }
        }
        assert!(UniformLight::prepare(Some(&grid), false, 1.0, LightFalloffLimit::Soft).is_none());
        assert!(UniformLight::prepare(Some(&grid), true, 0.0, LightFalloffLimit::Soft).is_none());
        assert!(UniformLight::prepare(None, true, 1.0, LightFalloffLimit::Soft).is_none());
    }

    /// A 2x2 grid whose top-left corner sits `max_level` above the field, i.e.
    /// `max_level/8192` EV of calibrated correction, with unit node spacing.
    fn pole_grid(max_level: u16, flags: u16) -> LightFalloff {
        let [flag_lo, flag_hi] = flags.to_le_bytes();
        let header =
            [16u8, 0, 2, 0, 2, 0, 0, 0, 0, 0, 1, 0, 1, 0, flag_lo, flag_hi].to_vec();
        let half = max_level / 2;
        LightFalloff {
            header,
            columns: 2,
            rows: 2,
            nodes: vec![
                LightFalloffNode { luminance: 0, dx: 0, dy: 0 },
                LightFalloffNode { luminance: half, dx: 0, dy: 0 },
                LightFalloffNode { luminance: half, dx: 0, dy: 0 },
                LightFalloffNode { luminance: max_level, dx: 0, dy: 0 },
            ],
        }
    }

    /// The 0/50/100 slider positions of the shipped 0..120 parameter. Half
    /// strength is the square root of the full gain because the kernel
    /// multiplies the EV, not the gain.
    #[test]
    fn uniform_light_amount_vectors_blend_the_calibrated_ev() {
        let grid = pole_grid(16_384, 1);
        assert_eq!(
            strength_from_amount(0.0, LightFalloffMode::Soft),
            (0.0, LightFalloffLimit::Soft)
        );
        assert_eq!(
            strength_from_amount(50.0, LightFalloffMode::Soft),
            (0.5, LightFalloffLimit::Soft)
        );
        assert_eq!(
            strength_from_amount(100.0, LightFalloffMode::Soft),
            (1.0, LightFalloffLimit::Soft)
        );
        assert_eq!(
            strength_from_amount(120.0, LightFalloffMode::Hard),
            (1.2, LightFalloffLimit::Hard)
        );
        assert_eq!(
            strength_from_amount(100.0, LightFalloffMode::Uncapped).1,
            LightFalloffLimit::Uncapped
        );
        assert!(
            UniformLight::prepare(
                Some(&grid),
                true,
                strength_from_amount(0.0, grid.mode()).0,
                strength_from_amount(0.0, grid.mode()).1
            )
            .is_none(),
            "amount 0 is identity"
        );
        let half =
            UniformLight::prepare(Some(&grid), true, 0.5, grid.mode().limit())
                .expect("amount 50 opens the gate");
        let full =
            UniformLight::prepare(Some(&grid), true, 1.0, grid.mode().limit())
                .expect("amount 100 opens the gate");
        // Corner EV, then the pixel that reaches it: top-left 2.0, the other
        // two corners 1.0, bottom-right 0.0.
        let cases = [([0u32, 0u32], 2.0f32), ([1, 0], 1.0), ([0, 1], 1.0), ([1, 1], 0.0)];
        for (pixel, ev) in cases {
            let want_full = ev.exp2();
            let want_half = (ev * 0.5).exp2();
            for (stage, want) in [(&full, want_full), (&half, want_half)] {
                let got = stage.gain([0, 0], pixel, 1.0);
                assert!((got - want).abs() <= want * 1e-6, "pixel {pixel:?}: {got} != {want}");
            }
            // A linear-toward-identity blend would be `1 + (g-1)/2`; make sure
            // the two laws are distinguishable at the strongest corner.
            if ev > 0.0 {
                assert!((half.gain([0, 0], pixel, 1.0) - (1.0 + (want_full - 1.0) / 2.0)).abs() > 1e-3);
            }
        }
    }

    /// Soft has three tiers in the shipped kernel: below 3 EV nothing is
    /// limited, 3..5 EV subtracts `0.25*(ev-3)^2`, above 5 EV it pins at 4.
    #[test]
    fn uniform_light_soft_limit_keeps_its_three_tiers() {
        // max_level/8192 is the corner EV; 8192 is one stop.
        for (max_level, ev) in [(16_384u16, 2.0f32), (32_768, 4.0), (40_960, 5.0), (49_152, 6.0)] {
            let grid = pole_grid(max_level, 1);
            let stage = UniformLight::prepare(Some(&grid), true, 1.0, LightFalloffLimit::Soft)
                .expect("soft stage");
            let want_ev = if ev > 3.0 {
                if ev > 5.0 { 4.0 } else { ev - 0.25 * (ev - 3.0) * (ev - 3.0) }
            } else {
                ev
            };
            let got = stage.gain([0, 0], [0, 0], 1.0);
            let want = want_ev.exp2();
            assert!((got - want).abs() <= want * 1e-6, "{ev} EV -> {got}, want {want}");
        }
        let hard_grid = pole_grid(32_768, 0);
        let hard = UniformLight::prepare(Some(&hard_grid), true, 1.0, LightFalloffLimit::Hard)
            .expect("hard stage");
        assert!((hard.gain([0, 0], [0, 0], 1.0) - 8.0).abs() <= 8.0 * 1e-6);
        let uncapped_grid = pole_grid(49_152, 1);
        let uncapped =
            UniformLight::prepare(Some(&uncapped_grid), true, 1.0, LightFalloffLimit::Uncapped)
                .expect("uncapped stage");
        assert!((uncapped.gain([0, 0], [0, 0], 1.0) - 64.0).abs() <= 64.0 * 1e-6);
    }

    /// The only persisted mode input is the calibration's `Hdr` flags word.
    #[test]
    fn uniform_light_mode_comes_from_the_lcc_flags_word() {
        let soft = pole_grid(49_152, 1);
        assert_eq!(soft.flags(), Some(1));
        assert_eq!(soft.mode(), LightFalloffMode::Soft);
        let hard = pole_grid(49_152, 0);
        assert_eq!(hard.flags(), Some(0));
        assert_eq!(hard.mode(), LightFalloffMode::Hard);
        // Legacy 12-byte headers carry no flags word and keep the soft knee.
        let legacy = LightFalloff {
            header: vec![0; 12],
            columns: 2,
            rows: 2,
            nodes: soft.nodes.clone(),
        };
        assert_eq!(legacy.flags(), None);
        assert_eq!(legacy.mode(), LightFalloffMode::Soft);
        // The mode decides both the strength scale and the limiter.
        assert_eq!(
            strength_from_amount(100.0, soft.mode()),
            (1.0, LightFalloffLimit::Soft)
        );
        assert_eq!(
            strength_from_amount(100.0, hard.mode()),
            (1.0, LightFalloffLimit::Hard)
        );
    }

    #[test]
    fn uniform_light_unknown_kernel_modes_fall_back_to_soft() {
        for (index, want) in [
            (0, LightFalloffMode::Soft),
            (1, LightFalloffMode::Hard),
            (2, LightFalloffMode::Uncapped),
        ] {
            assert_eq!(LightFalloffMode::from_kernel_index(index), want);
            assert_eq!(want.kernel_index(), index);
        }
        assert_eq!(LightFalloffMode::from_kernel_index(3), LightFalloffMode::Soft);
        assert_eq!(LightFalloffMode::from_kernel_index(-1), LightFalloffMode::Soft);
    }
}
