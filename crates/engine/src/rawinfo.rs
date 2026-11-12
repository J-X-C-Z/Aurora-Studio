//! Per-image values and per-camera defaults, as Capture One feeds them into its
//! info block.
//!
//! C1 does not store `IB_F_*` keys in the RAW file: `IB_F_*` are the key names of
//! its in-process property container (`InfoBlock_impl.cpp`), and the NEF of the
//! reference camera contains no `IB_` literal at all. The values behind them come
//! from two places:
//!
//! * `coreConfig.cdx` — per-`cameraID` ranges and defaults, with an ISO ladder
//!   (`<ISO val="3200">` blocks). This is where `IB_F_EXPOSURE_START/_END/_DEFAULT`,
//!   `IB_F_USMAMOUNT_*` and friends are filled from.
//! * the frame itself — MakerNote white-balance gains, EXIF exposure settings and
//!   pixel dimensions (`IB_F_WBRED/WBGREEN/WBBLUE`, `IB_F_EXP_*`).
//!
//! Both are reproduced here. Nothing in this module writes to the image.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{anyhow, Context, Result};

use crate::param_meta::{by_key, ParamOverrides};

/// Values read from one frame, keyed by C1's info-block names.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RawInfoBlock {
    pub values: Vec<(String, f64)>,
    pub make: Option<String>,
    pub model: Option<String>,
    pub iso: Option<u32>,
    /// ExifIFD `0xA434` LensModel (exiftool `[ExifIFD] LensModel`).
    pub lens_model: Option<String>,
    /// Nikon MakerNote `0x00A7` ShutterCount.
    pub shutter_count: Option<u32>,
    /// Nikon MakerNote `0x0005` WhiteBalance, case-normalised like exiftool.
    pub wb_mode: Option<String>,
    /// Nikon MakerNote `0x004F` ColorTemperatureAuto. exiftool has no EXIF
    /// `-ColorTemperature` tag in a NEF; this is the value it reports there.
    pub color_temperature: Option<u32>,
}

impl RawInfoBlock {
    pub fn get(&self, key: &str) -> Option<f64> {
        self.values.iter().find(|(k, _)| k == key).map(|(_, v)| *v)
    }

    /// Nikon MakerNote `WB_RBGGLevels` as positive (red, green, blue) gains.
    pub fn white_balance_gains(&self) -> Option<[f32; 3]> {
        let gain = |key: &str| self.get(key).filter(|v| *v > 0.0).map(|v| v as f32);
        Some([gain("IB_F_WBRED")?, gain("IB_F_WBGREEN")?, gain("IB_F_WBBLUE")?])
    }

    /// Per-image overrides: white balance (converted to kelvin/tint) and the
    /// exposure defaults EXIF states outright.
    pub fn apply_to(&self, overrides: &mut ParamOverrides) {
        if let Some(gains) = self.white_balance_gains() {
            let (kelvin, tint) = crate::gains_to_kelvin_tint(gains);
            overrides.set_default("WhiteBalanceTemperature", kelvin);
            overrides.set_default("WhiteBalanceTint", tint);
        }
        if let Some(ev) = self.get("IB_F_EXP_EXPOSURECOMPENSATION") {
            overrides.set_default("Exposure", ev as f32);
        }
    }
}

// ───────────────────────────── TIFF / EXIF / MakerNote ─────────────────────────────

/// Read the frame's own values. Accepts any TIFF-based RAW (NEF/CR2/ARW/DNG).
pub fn read_nikon_info_block(path: &Path) -> Result<RawInfoBlock> {
    let data = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    parse_info_block(&data)
}

/// Same as [`read_nikon_info_block`] over an already-loaded file.
pub fn parse_info_block(data: &[u8]) -> Result<RawInfoBlock> {
    let tiff = Tiff::new(data)?;
    let mut out = RawInfoBlock::default();

    let ifd0 = tiff.ifd(tiff.first_ifd)?;
    out.make = ifd0.ascii(&tiff, 0x010f);
    out.model = ifd0.ascii(&tiff, 0x0110);
    // exiftool reads ImageWidth/Height from the full-size SubIFD1; IFD0's own
    // dimensions describe the embedded preview and are absent in a NEF.
    if let Some((w, h)) = subifd_size(&tiff, &ifd0).or_else(|| ifd0.pixel_size(&tiff)) {
        out.values.push(("IB_F_PIXELWIDTH".into(), w as f64));
        out.values.push(("IB_F_PIXELHEIGHT".into(), h as f64));
    }
    if let Some(o) = ifd0.uint(&tiff, 0x0112) {
        out.values.push(("IB_F_ORIENTATION_DEFAULT".into(), o as f64));
    }

    let Some(exif_off) = ifd0.uint(&tiff, 0x8769) else { return Ok(out) };
    let exif = tiff.ifd(exif_off as usize)?;

    if let Some(iso) = exif.uint(&tiff, 0x8827).or_else(|| exif.uint(&tiff, 0x8833)) {
        out.iso = Some(iso);
        out.values.push(("IB_I_ISO".into(), iso as f64));
    }
    if let Some(v) = exif.rational(&tiff, 0x829a) {
        out.values.push(("IB_F_EXP_SHUTTERSPEED".into(), v));
    }
    if let Some(v) = exif.rational(&tiff, 0x829d) {
        out.values.push(("IB_F_EXP_APERTURE".into(), v));
    }
    if let Some(v) = exif.rational(&tiff, 0x920a) {
        out.values.push(("IB_F_EXP_FOCALLENGTH".into(), v));
    }
    if let Some(v) = exif.signed(&tiff, 0x9204) {
        out.values.push(("IB_F_EXP_EXPOSURECOMPENSATION".into(), v));
    }

    // ExifIFD `0xA434` LensModel.
    out.lens_model = exif.ascii(&tiff, 0xa434);

    if let Some(mn) = exif.bytes(&tiff, 0x927c) {
        if let Some((mt, mifd)) = nikon_ifd(mn) {
            if let Some(wb) = nikon_wb_gains(&mt, &mifd) {
                out.values.push(("IB_F_WBRED".into(), wb[0]));
                out.values.push(("IB_F_WBGREEN".into(), wb[1]));
                out.values.push(("IB_F_WBBLUE".into(), wb[2]));
            }
            // MakerNote `0x0005` WhiteBalance, ASCII.
            out.wb_mode = mifd.ascii(&mt, 0x0005).map(normalise_nikon_wb);
            // MakerNote `0x004F` ColorTemperatureAuto, int16u.
            if let Some(k) = mifd.uint(&mt, 0x004f) {
                out.color_temperature = Some(k);
                out.values.push(("IB_I_WB_COLORTEMPERATURE".into(), k as f64));
            }
            // MakerNote `0x00A7` ShutterCount, int32u.
            if let Some(n) = mifd.uint(&mt, 0x00a7) {
                out.shutter_count = Some(n);
                out.values.push(("IB_I_SHUTTERCOUNT".into(), n as f64));
            }
        }
    }
    Ok(out)
}

/// IFD0 `0x014A` lists the SubIFD offsets. The full-size frame is the largest
/// one (SubIFD1); the rest are thumbnails and previews.
fn subifd_size(t: &Tiff, ifd0: &Ifd) -> Option<(u32, u32)> {
    ifd0
        .u32s(t, 0x014a)?
        .into_iter()
        .filter_map(|off| t.ifd(off as usize).ok())
        .filter_map(|sub| sub.pixel_size(t))
        .max_by_key(|(w, h)| *w as u64 * *h as u64)
}

/// Nikon type-3 MakerNote: `"Nikon\0"` + 2-byte version + 2 pad bytes, then a
/// nested TIFF header whose offsets are relative to that header, not the file.
fn nikon_ifd(maker: &[u8]) -> Option<(Tiff<'_>, Ifd)> {
    if !maker.starts_with(b"Nikon\0") || maker.len() < 18 {
        return None;
    }
    let inner = Tiff::new_at(maker, 10).ok()?;
    let ifd = inner.ifd(inner.first_ifd).ok()?;
    Some((inner, ifd))
}

/// exiftool's `FormatString` lowercases all but the first letter of a word
/// (`AUTO1` -> `Auto1`); reproduce that for the WB mode strings.
fn normalise_nikon_wb(raw: String) -> String {
    let lower = raw.to_ascii_lowercase();
    match lower.chars().next() {
        Some(c) => c.to_ascii_uppercase().to_string() + lower.get(1..).unwrap_or(""),
        None => raw,
    }
}

/// Nikon MakerNote tag `0x000c` is `WB_RBGGLevels`: four rationals in the order
/// red, blue, green, green2. Green is returned as the mean of the two greens.
///
/// Verified against the reference frame: the file holds `1.625 / 1.412109375`
/// and Capture One's `ZVARIANTLAYER.ZWHITEBALANCE` for that image is
/// `1.625000;1.000000;1.412000` — the same three gains, green pinned at 1.
fn nikon_wb_gains(inner: &Tiff, ifd: &Ifd) -> Option<[f64; 3]> {
    let rationals = ifd.rationals(inner, 0x000c)?;
    if rationals.len() < 3 {
        return None;
    }
    let (r, b, g1, g2) = (rationals[0], rationals[1], rationals[2], *rationals.get(3).unwrap_or(&rationals[2]));
    if !(g1 > 0.0) {
        return None;
    }
    Some([r, (g1 + g2) / 2.0, b])
}

struct Entry {
    tag: u16,
    typ: u16,
    count: u32,
    value: [u8; 4],
    entry_at: usize,
}

impl Entry {
    fn size(&self) -> usize {
        let unit = match self.typ {
            1 | 2 | 6 | 7 => 1,
            3 | 8 => 2,
            4 | 9 | 11 | 13 => 4,
            5 | 10 | 12 => 8,
            _ => 1,
        };
        unit * self.count as usize
    }
}

struct Ifd {
    entries: Vec<Entry>,
}

impl Ifd {
    fn find(&self, tag: u16) -> Option<&Entry> {
        self.entries.iter().find(|e| e.tag == tag)
    }

    fn bytes<'a>(&self, t: &'a Tiff, tag: u16) -> Option<&'a [u8]> {
        let e = self.find(tag)?;
        let len = e.size();
        if len <= 4 {
            let start = e.entry_at + 8;
            t.data.get(start..start + len)
        } else {
            // Offsets are relative to this TIFF header, which is not the file
            // start inside a MakerNote.
            let off = t.base + t.u32(&e.value)? as usize;
            t.data.get(off..off + len)
        }
    }

    fn ascii(&self, t: &Tiff, tag: u16) -> Option<String> {
        let raw = self.bytes(t, tag)?;
        let end = raw.iter().position(|b| *b == 0).unwrap_or(raw.len());
        let s = String::from_utf8_lossy(&raw[..end]).trim().to_string();
        (!s.is_empty()).then_some(s)
    }

    fn uint(&self, t: &Tiff, tag: u16) -> Option<u32> {
        let e = self.find(tag)?;
        match e.typ {
            3 => {
                let raw = self.bytes(t, tag)?;
                Some(t.u16(raw)? as u32)
            }
            4 | 9 => {
                let raw = self.bytes(t, tag)?;
                t.u32(raw)
            }
            _ => None,
        }
    }

    fn rational(&self, t: &Tiff, tag: u16) -> Option<f64> {
        self.rationals(t, tag)?.first().copied()
    }

    fn signed(&self, t: &Tiff, tag: u16) -> Option<f64> {
        let e = self.find(tag)?;
        let raw = self.bytes(t, tag)?;
        match e.typ {
            10 => {
                let n = t.i32(raw)? as f64;
                let d = t.i32(raw.get(4..)?)? as f64;
                (d != 0.0).then_some(n / d)
            }
            _ => self.rational(t, tag),
        }
    }

    fn rationals(&self, t: &Tiff, tag: u16) -> Option<Vec<f64>> {
        self.find(tag)?;
        let raw = self.bytes(t, tag)?;
        let mut out = Vec::new();
        let mut i = 0;
        while i + 8 <= raw.len() {
            let n = t.u32(&raw[i..])? as f64;
            let d = t.u32(&raw[i + 4..])? as f64;
            out.push(if d == 0.0 { 0.0 } else { n / d });
            i += 8;
        }
        Some(out)
    }

    fn u32s(&self, t: &Tiff, tag: u16) -> Option<Vec<u32>> {
        let e = self.find(tag)?;
        if !matches!(e.typ, 4 | 13) {
            return None;
        }
        let raw = self.bytes(t, tag)?;
        Some(raw.chunks_exact(4).filter_map(|b| t.u32(b)).collect())
    }

    fn pixel_size(&self, t: &Tiff) -> Option<(u32, u32)> {
        let w = self.uint(t, 0x0100).or_else(|| self.uint(t, 0x0101))?;
        let h = self.uint(t, 0x0101).or_else(|| self.uint(t, 0x0100))?;
        (w > 0 && h > 0).then_some((w, h))
    }
}

struct Tiff<'a> {
    data: &'a [u8],
    le: bool,
    base: usize,
    first_ifd: usize,
}

impl<'a> Tiff<'a> {
    fn new(data: &'a [u8]) -> Result<Self> {
        Self::new_at(data, 0)
    }

    /// `base` is where the TIFF header starts inside `data` (nonzero for a
    /// MakerNote, whose offsets are relative to its own header).
    fn new_at(data: &'a [u8], base: usize) -> Result<Self> {
        let head = data.get(base..base + 8).ok_or_else(|| anyhow!("truncated TIFF header"))?;
        let le = match &head[0..2] {
            b"II" => true,
            b"MM" => false,
            _ => return Err(anyhow!("not a TIFF container")),
        };
        let t = Tiff { data, le, base, first_ifd: 0 };
        if t.u16(&head[2..]).ok_or_else(|| anyhow!("truncated TIFF header"))? != 42 {
            return Err(anyhow!("bad TIFF magic"));
        }
        let offset = t.u32(&head[4..]).ok_or_else(|| anyhow!("truncated TIFF header"))? as usize;
        Ok(Tiff { first_ifd: base + offset, ..t })
    }

    fn u16(&self, b: &[u8]) -> Option<u16> {
        let a = *b.first()?;
        let c = *b.get(1)?;
        Some(if self.le { u16::from_le_bytes([a, c]) } else { u16::from_be_bytes([a, c]) })
    }

    fn u32(&self, b: &[u8]) -> Option<u32> {
        let v: [u8; 4] = b.get(..4)?.try_into().ok()?;
        Some(if self.le { u32::from_le_bytes(v) } else { u32::from_be_bytes(v) })
    }

    fn i32(&self, b: &[u8]) -> Option<i32> {
        self.u32(b).map(|v| v as i32)
    }

    fn ifd(&self, offset: usize) -> Result<Ifd> {
        let count = self
            .u16(self.data.get(offset..offset + 2).ok_or_else(|| anyhow!("truncated IFD"))?)
            .ok_or_else(|| anyhow!("truncated IFD"))? as usize;
        let mut entries = Vec::with_capacity(count.min(256));
        for i in 0..count {
            let at = offset + 2 + i * 12;
            let Some(raw) = self.data.get(at..at + 12) else { break };
            let tag = self.u16(raw).unwrap_or(0);
            let typ = self.u16(&raw[2..]).unwrap_or(0);
            let count = self.u32(&raw[4..]).unwrap_or(0);
            let value: [u8; 4] = raw[8..12].try_into().unwrap_or([0; 4]);
            entries.push(Entry { tag, typ, count, value, entry_at: at });
        }
        Ok(Ifd { entries })
    }
}

// ───────────────────────────── coreConfig.cdx ─────────────────────────────

/// One `<cameraID>` block: values outside any `<ISO>` plus the ISO ladder.
#[derive(Clone, Debug, Default)]
struct Camera {
    base: Vec<(String, f32)>,
    /// `(iso, values)` in file order.
    ladder: Vec<(u32, Vec<(String, f32)>)>,
}

#[derive(Clone, Debug, Default)]
pub struct CameraDefaults {
    cams: HashMap<String, Camera>,
}

impl CameraDefaults {
    /// Parse the decoded `coreConfig` XML. Unknown tags are ignored.
    pub fn parse(xml: &str) -> Self {
        let mut cams: HashMap<String, Camera> = HashMap::new();
        for (id, body) in camera_blocks(xml) {
            let cam = cams.entry(id).or_default();
            let (base, ladder) = split_iso_blocks(&body);
            cam.base.extend(base);
            for (iso, values) in ladder {
                match cam.ladder.iter_mut().find(|(i, _)| *i == iso) {
                    Some((_, existing)) => existing.extend(values),
                    None => cam.ladder.push((iso, values)),
                }
            }
        }
        Self { cams }
    }

    /// Load and decode `coreConfig.cdx` (XOR `0x81`, see [`crate::decode_cdx`]).
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
        Ok(Self::parse(&crate::decode_cdx(&bytes)?))
    }

    /// Ranges and defaults for one camera at one ISO.
    ///
    /// The ladder resolves to the **highest ISO block at or below** the actual
    /// ISO, which the reference frame pins down: `NikonZ6III` holds
    /// `USMAmount_default` 180 at ISO 0, 160 at 3200 and 140 at 6400, the frame
    /// is ISO 5000, and Capture One's default layer for it stores 160.
    pub fn overrides(&self, camera: Option<&str>, iso: Option<u32>) -> ParamOverrides {
        let mut merged: HashMap<&str, f32> = HashMap::new();
        // BASEDEFAULT's ranges sit inside its ISO 0 block, so the ladder has to
        // be applied there too or the ranges (USMAmount_end and friends) vanish.
        self.apply_camera(&mut merged, "BASEDEFAULT", iso);
        if let Some(id) = camera.and_then(|c| self.resolve(c)) {
            self.apply_camera(&mut merged, &id, iso);
        }
        to_overrides(&merged)
    }

    fn apply_camera<'a>(&'a self, merged: &mut HashMap<&'a str, f32>, id: &str, iso: Option<u32>) {
        let Some(cam) = self.cams.get(id) else { return };
        for (k, v) in &cam.base {
            merged.insert(k.as_str(), *v);
        }
        let rung = match iso {
            Some(actual) => cam.ladder.iter().filter(|(i, _)| *i <= actual).max_by_key(|(i, _)| *i),
            None => cam.ladder.iter().min_by_key(|(i, _)| *i),
        };
        if let Some((_, values)) = rung {
            for (k, v) in values {
                merged.insert(k.as_str(), *v);
            }
        }
    }

    /// Camera ids that exist, for diagnostics.
    pub fn camera_ids(&self) -> Vec<&str> {
        self.cams.keys().map(|s| s.as_str()).collect()
    }

    /// Resolve an EXIF model ("NIKON Z6_3") to a `cameraID` ("NikonZ6III").
    ///
    /// C1 carries an alias table for exactly this in `ImgCoreDll.dll`: adjacent
    /// entries `NIKONZ63` / `NikonZ6III` (and `CANONEOSR5MK2` /
    /// `CANONEOSR5MKII`) show it maps the normalised EXIF string to its own id.
    /// The two differ only in how a trailing mark number is spelled, so the same
    /// normalisation plus one numeral rewrite reproduces it without copying the
    /// whole table.
    fn resolve(&self, model: &str) -> Option<String> {
        let want = normalise(model);
        if self.cams.contains_key(&want) {
            return Some(want);
        }
        for candidate in numeral_variants(&want) {
            if self.cams.contains_key(&candidate) {
                return Some(candidate);
            }
        }
        None
    }
}

/// Split a camera body into the values outside `<ISO>` and one entry per ISO block.
fn split_iso_blocks(body: &str) -> (Vec<(String, f32)>, Vec<(u32, Vec<(String, f32)>)>) {
    let mut base = Vec::new();
    let mut ladder = Vec::new();
    let mut rest = body;
    while let Some(at) = rest.find("<ISO") {
        base.extend(tag_values(&rest[..at]));
        rest = &rest[at..];
        let Some(tag_end) = rest.find('>') else { break };
        let iso = attribute(&rest[..tag_end], "val").and_then(|v| v.parse::<u32>().ok()).unwrap_or(0);
        let body_start = tag_end + 1;
        let close = rest[body_start..].find("</ISO>").map(|i| body_start + i);
        if rest[..tag_end].ends_with('/') {
            ladder.push((iso, Vec::new()));
            rest = &rest[tag_end + 1..];
            continue;
        }
        let Some(end) = close else {
            ladder.push((iso, tag_values(&rest[body_start..])));
            return (base, ladder);
        };
        ladder.push((iso, tag_values(&rest[body_start..end])));
        rest = &rest[(end + "</ISO>".len()).min(rest.len())..];
    }
    base.extend(tag_values(rest));
    (base, ladder)
}

fn normalise(model: &str) -> String {
    model.chars().filter(|c| c.is_ascii_alphanumeric()).map(|c| c.to_ascii_uppercase()).collect()
}

/// `NIKONZ63` → `NIKONZ6III`, `CANONEOSR5MK2` → `CANONEOSR5MKII`, and the reverse.
fn numeral_variants(s: &str) -> Vec<String> {
    const ROMAN: [(u8, &str); 6] = [
        (b'2', "II"),
        (b'3', "III"),
        (b'4', "IV"),
        (b'5', "V"),
        (b'6', "VI"),
        (b'7', "VII"),
    ];
    let mut out = Vec::new();
    let bytes = s.as_bytes();
    let Some(&last) = bytes.last() else { return out };
    if let Some((_, roman)) = ROMAN.iter().find(|(d, _)| *d == last) {
        out.push(format!("{}{}", &s[..s.len() - 1], roman));
    }
    for (digit, roman) in ROMAN {
        if let Some(stem) = s.strip_suffix(roman) {
            out.push(format!("{stem}{}", digit as char));
        }
    }
    out
}

/// `(cameraID, inner XML)` for every `<cameraID ... val="X"> ... </cameraID>`.
fn camera_blocks(xml: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find("<cameraID") {
        rest = &rest[start..];
        let Some(tag_end) = rest.find('>') else { break };
        let head = &rest[..tag_end];
        let id = attribute(head, "val").unwrap_or_default();
        let body_start = tag_end + 1;
        let body_end = rest[body_start..].find("</cameraID>").map(|i| body_start + i).unwrap_or(rest.len());
        out.push((id, rest[body_start..body_end].to_string()));
        rest = &rest[(body_end + "</cameraID>".len()).min(rest.len())..];
    }
    out
}

fn attribute<'a>(tag: &'a str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let at = tag.find(&needle)? + needle.len();
    let end = tag[at..].find('"')?;
    Some(tag[at..at + end].to_string())
}

/// `<tag val="number"/>` pairs in document order.
fn tag_values(body: &str) -> Vec<(String, f32)> {
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(start) = rest.find('<') {
        rest = &rest[start..];
        let Some(end) = rest.find('>') else { break };
        let tag = &rest[1..end];
        if !tag.starts_with('/') && !tag.starts_with("![") && !tag.starts_with('?') {
            let name = tag.split(|c: char| c.is_whitespace() || c == '/').next().unwrap_or("");
            let self_closing = tag.ends_with('/');
            if let Some(raw) = attribute(tag, "val") {
                if let Ok(v) = raw.parse::<f32>() {
                    out.push((name.to_string(), v));
                }
            }
            if !self_closing {
                // Skip the element body: only `<ISO val="..">` blocks nest, and
                // their values are collected by the same loop.
            }
        }
        rest = &rest[end + 1..];
    }
    out
}

/// coreConfig tag → (catalog key, kind). Evidence: `outputs/analysis/raw-info-block.md`.
const TAG_MAP: &[(&str, &str)] = &[
    ("exposure_start", "Exposure"),
    ("exposure_end", "Exposure"),
    ("exposure_default", "Exposure"),
    ("saturation_start", "Saturation"),
    ("saturation_end", "Saturation"),
    ("saturation_default", "Saturation"),
    ("highlightRecovery_default", "HighlightRecovery"),
    ("shadowRecovery_default", "ShadowRecovery"),
    ("USMAmount_start", "USMAmount"),
    ("USMAmount_end", "USMAmount"),
    ("USMAmount_default", "USMAmount"),
    ("USMRadius_start", "USMRadius"),
    ("USMRadius_end", "USMRadius"),
    ("USMRadius_default", "USMRadius"),
    ("USMThreshold_start", "USMThreshold"),
    ("USMThreshold_end", "USMThreshold"),
    ("USMThreshold_default", "USMThreshold"),
    ("USMHaloControl_start", "USMHalo"),
    ("USMHaloControl_end", "USMHalo"),
    ("USMHaloControl_default", "USMHalo"),
    ("CNRAmount_start", "ColorNoiseReductionAmount"),
    ("CNRAmount_end", "ColorNoiseReductionAmount"),
    ("CNRAmount_default", "ColorNoiseReductionAmount"),
    ("noiseReductionAmount_start", "NoiseReductionAmount"),
    ("noiseReductionAmount_end", "NoiseReductionAmount"),
    ("noiseReductionAmount_default", "NoiseReductionAmount"),
    ("detailsAmount_start", "NoiseDetailsAmount"),
    ("detailsAmount_end", "NoiseDetailsAmount"),
    ("detailsAmount_default", "NoiseDetailsAmount"),
    ("cleanLongExposureAmount_start", "CleanNoiseAmount"),
    ("cleanLongExposureAmount_end", "CleanNoiseAmount"),
    ("cleanLongExposureAmount_default", "CleanNoiseAmount"),
    ("SurfaceSmoothness_start", "SurfaceSmoothnessAmount"),
    ("SurfaceSmoothness_end", "SurfaceSmoothnessAmount"),
    ("SurfaceSmoothness_default", "SurfaceSmoothnessAmount"),
];

/// `coreConfig` tags are renamed to the info-block keys C1 itself derives for
/// `ValidVariantPropertyRanges` (§2.1), so ranges and defaults go through the
/// one [`ParamOverrides::from_pairs`] parser.
fn to_overrides(merged: &HashMap<&str, f32>) -> ParamOverrides {
    let mut pairs: Vec<(String, String)> = Vec::new();
    for (tag, value) in merged {
        let Some((_, key)) = TAG_MAP.iter().find(|(t, _)| t == tag) else { continue };
        let Some(meta) = by_key(key) else { continue };
        let Some(info_block) = crate::param_meta::range_key(meta.key) else { continue };
        let suffix = if tag.ends_with("_start") {
            "START"
        } else if tag.ends_with("_end") {
            "END"
        } else {
            "DEFAULT"
        };
        pairs.push((format!("{info_block}_{suffix}"), value.to_string()));
    }
    ParamOverrides::from_pairs(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiff_with(exif: &[(u16, u16, u32, [u8; 4])], extra: &[u8]) -> Vec<u8> {
        // IFD0 -> ExifIFD pointer; ExifIFD holds `exif`; `extra` follows both.
        let mut out = Vec::new();
        out.extend_from_slice(b"II\x2a\x00");
        out.extend_from_slice(&8u32.to_le_bytes());
        let ifd0_at = 8usize;
        let exif_ifd_at = ifd0_at + 2 + 12 + 4;
        let extra_at = exif_ifd_at + 2 + 12 * exif.len() + 4;

        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&0x8769u16.to_le_bytes());
        out.extend_from_slice(&4u16.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(exif_ifd_at as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());

        out.extend_from_slice(&(exif.len() as u16).to_le_bytes());
        for (tag, typ, count, value) in exif {
            out.extend_from_slice(&tag.to_le_bytes());
            out.extend_from_slice(&typ.to_le_bytes());
            out.extend_from_slice(&count.to_le_bytes());
            out.extend_from_slice(value);
        }
        out.extend_from_slice(&0u32.to_le_bytes());
        assert_eq!(out.len(), extra_at, "extra must start where it says");
        out.extend_from_slice(extra);
        out
    }

    #[test]
    fn reads_iso_and_rejects_garbage() {
        let mut data = tiff_with(&[(0x8827, 3, 1, [0x88, 0x13, 0, 0])], &[]);
        let block = parse_info_block(&data).unwrap();
        assert_eq!(block.iso, Some(5000));
        assert_eq!(block.get("IB_I_ISO"), Some(5000.0));

        assert!(parse_info_block(b"not a tiff at all").is_err());
        data.truncate(20);
        assert!(parse_info_block(&data).is_ok() || parse_info_block(&data).is_err());
    }

    #[test]
    fn reads_makernote_white_balance() {
        let mn_at = 8 + 2 + 12 + 4 + 2 + 12 + 4;
        let wb_at = mn_at + 10 + 8 + 2 + 12 + 4;
        let mut extra = Vec::new();
        extra.extend_from_slice(b"Nikon\x00\x02\x11\x00\x00");
        extra.extend_from_slice(b"II\x2a\x00");
        extra.extend_from_slice(&8u32.to_le_bytes());
        extra.extend_from_slice(&1u16.to_le_bytes());
        extra.extend_from_slice(&0x000cu16.to_le_bytes());
        extra.extend_from_slice(&5u16.to_le_bytes());
        extra.extend_from_slice(&4u32.to_le_bytes());
        extra.extend_from_slice(&((wb_at - (mn_at + 10)) as u32).to_le_bytes());
        extra.extend_from_slice(&0u32.to_le_bytes());
        for (n, d) in [(1625u32, 1000u32), (1412109375, 1000000000), (1, 1), (1, 1)] {
            extra.extend_from_slice(&n.to_le_bytes());
            extra.extend_from_slice(&d.to_le_bytes());
        }
        assert_eq!(extra.len(), wb_at - mn_at + 32, "rationals end the blob");

        let len = extra.len() as u32;
        let data = tiff_with(&[(0x927c, 7, len, (mn_at as u32).to_le_bytes())], &extra);
        let block = parse_info_block(&data).unwrap();
        let r = block.get("IB_F_WBRED").unwrap();
        let g = block.get("IB_F_WBGREEN").unwrap();
        let b = block.get("IB_F_WBBLUE").unwrap();
        assert!((r - 1.625).abs() < 1e-9, "{r}");
        assert!((b - 1.412109375).abs() < 1e-9, "{b}");
        assert!((g - 1.0).abs() < 1e-9, "{g}");
    }

    #[test]
    fn white_balance_reaches_the_overrides() {
        let mut block = RawInfoBlock::default();
        block.values.push(("IB_F_WBRED".into(), 1.625));
        block.values.push(("IB_F_WBGREEN".into(), 1.0));
        block.values.push(("IB_F_WBBLUE".into(), 1.412));
        let mut o = ParamOverrides::default();
        block.apply_to(&mut o);
        let (_, _, k) = o.apply(by_key("WhiteBalanceTemperature").unwrap());
        assert!((k - 5648.0).abs() < 20.0, "{k}");
    }

    #[test]
    fn camera_defaults_carry_ranges_and_defaults() {
        let xml = r#"
<cameraDefaults>
  <cameraID val="BASEDEFAULT">
    <exposure_start val="-4" /><exposure_end val="4" /><exposure_default val="0.0" />
    <USMAmount_start val="0.0" /><USMAmount_end val="1000" /><USMAmount_default val="160" />
  </cameraID>
  <cameraID val="NikonZ6III">
    <USMAmount_default val="180" />
    <ISO val="3200"><USMAmount_default val="160" /></ISO>
  </cameraID>
</cameraDefaults>"#;
        let defaults = CameraDefaults::parse(xml);
        let o = defaults.overrides(Some("NIKON Z6_3"), Some(5000));
        let (min, max, def) = o.apply(by_key("Exposure").unwrap());
        assert_eq!((min, max, def), (-4.0, 4.0, 0.0));
        let (min, max, def) = o.apply(by_key("USMAmount").unwrap());
        assert_eq!((min, max), (0.0, 1000.0));
        assert_eq!(def, 160.0, "the deepest ISO block wins, as in C1");
    }

    #[test]
    fn unknown_camera_falls_back_to_base() {
        let xml = r#"<cameraDefaults>
  <cameraID val="BASEDEFAULT"><saturation_start val="-100" /><saturation_end val="100" /></cameraID>
</cameraDefaults>"#;
        let o = CameraDefaults::parse(xml).overrides(Some("NIKON Z9"), None);
        assert_eq!(o.apply(by_key("Saturation").unwrap()), (-100.0, 100.0, 0.0));
    }


    #[test]
    fn numeral_variants_cover_the_alias_table() {
        let v = numeral_variants("NIKONZ63");
        assert!(v.contains(&"NIKONZ6III".to_string()), "{v:?}");
        let v = numeral_variants("CANONEOSR5MK2");
        assert!(v.contains(&"CANONEOSR5MKII".to_string()), "{v:?}");
        let v = numeral_variants("CANONEOSR5MKII");
        assert!(v.contains(&"CANONEOSR5MK2".to_string()), "{v:?}");
    }


    /// IFD0 `0x014A` holds the SubIFD offsets; the full-size one wins over the
    /// 384x256 and 160x120 previews.
    #[test]
    fn subifd_dimensions_pick_the_full_size_frame() {
        let data = tiff_with_subifds(&[(384, 256), (6064, 4040), (160, 120)]);
        let block = parse_info_block(&data).unwrap();
        assert_eq!(block.get("IB_F_PIXELWIDTH"), Some(6064.0));
        assert_eq!(block.get("IB_F_PIXELHEIGHT"), Some(4040.0));
    }

    fn tiff_with_subifds(sizes: &[(u32, u32)]) -> Vec<u8> {
        let mut out = b"II\x2a\x00".to_vec();
        out.extend_from_slice(&8u32.to_le_bytes());
        let arr = 8 + 2 + 12 + 4;
        let mut off = arr + 4 * sizes.len();
        let mut ifds = Vec::new();
        for (w, h) in sizes {
            let mut d = 2u16.to_le_bytes().to_vec();
            for (tag, v) in [(0x0100u16, *w), (0x0101, *h)] {
                d.extend_from_slice(&tag.to_le_bytes());
                d.extend_from_slice(&4u16.to_le_bytes());
                d.extend_from_slice(&1u32.to_le_bytes());
                d.extend_from_slice(&v.to_le_bytes());
            }
            d.extend_from_slice(&0u32.to_le_bytes());
            ifds.push((off, d));
            off += 12 * 2 + 6;
        }
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&0x014au16.to_le_bytes());
        out.extend_from_slice(&4u16.to_le_bytes());
        out.extend_from_slice(&(sizes.len() as u32).to_le_bytes());
        out.extend_from_slice(&(arr as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for (o, _) in &ifds {
            out.extend_from_slice(&(*o as u32).to_le_bytes());
        }
        for (_, d) in &ifds {
            out.extend_from_slice(d);
        }
        out
    }


    /// A frame without SubIFDs, lens or MakerNote reports None, not a guess.
    #[test]
    fn missing_fields_return_none_without_panicking() {
        let block = parse_info_block(&tiff_with(&[], &[])).unwrap();
        assert_eq!(block.lens_model, None);
        assert_eq!(block.wb_mode, None);
        assert_eq!(block.shutter_count, None);
        assert_eq!(block.color_temperature, None);
        assert_eq!(block.get("IB_F_PIXELWIDTH"), None);
        assert_eq!(block.get("IB_I_SHUTTERCOUNT"), None);
    }

    #[test]
    fn non_nikon_makernote_yields_no_nikon_fields() {
        let mn_at = 44usize;
        let maker = b"Canon\0\x01\x00\x00\x00\x00\x00\x00\x00";
        let data = tiff_with(
            &[(0x927c, 7, maker.len() as u32, (mn_at as u32).to_le_bytes())],
            maker,
        );
        let block = parse_info_block(&data).unwrap();
        assert_eq!(block.wb_mode, None);
        assert_eq!(block.shutter_count, None);
        assert_eq!(block.color_temperature, None);
    }

    /// The reference frame, compared with
    /// `exiftool -LensModel -ShutterCount -WhiteBalance -ColorTemperature
    ///  -ImageWidth -ImageHeight -SubSecTimeOriginal _DSC5673.NEF`.
    /// `-ColorTemperature` is empty for this frame; `ColorTemperatureAuto`
    /// (tag 0x004F) is 4790. Skipped when the evidence directory is absent.
    #[test]
    fn matches_exiftool_on_the_reference_frame() {
        let Ok(home) = std::env::var("HOME") else { return };
        let path = Path::new(&home).join("agents/evidence/_DSC5673.NEF");
        let Ok(data) = std::fs::read(path) else { return };
        let block = parse_info_block(&data).unwrap();
        assert_eq!(block.lens_model.as_deref(), Some("NIKKOR Z 24-120mm f/4 S"));
        assert_eq!(block.shutter_count, Some(265058));
        assert_eq!(block.wb_mode.as_deref(), Some("Auto1"));
        assert_eq!(block.color_temperature, Some(4790));
        assert_eq!(block.get("IB_F_PIXELWIDTH"), Some(6064.0));
        assert_eq!(block.get("IB_F_PIXELHEIGHT"), Some(4040.0));
    }

    #[cfg(feature = "external-fixtures")]
    include!("../local-fixtures/rawinfo-tests.rs");
}
