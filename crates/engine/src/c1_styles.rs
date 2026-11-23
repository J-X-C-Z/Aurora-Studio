//! Capture One styles (`.costyle`) and presets (`.copreset`) as engine presets.
//!
//! The container is plain UTF-8 XML: `3C 3F 78 6D 6C` (`<?xml`) then
//! `<SL Engine="800">` and one `<E K="Exposure" V="2.355544"/>` per property.
//! There is no zip or archive header to unpack, and the `K` names are Capture
//! One's own property names.
//!
//! A file is mapped onto [`C1VariantSettings`] — the shape the catalog reader
//! fills from `ZVARIANTLAYER` — and converted once by
//! [`c1_model::to_adjustment_set`], so a style and a catalog variant share one
//! conversion.

use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{bail, Context, Result};

/// Stable category for a failed `.costyle` / `.copreset` import.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StyleImportErrorKind {
    /// The bytes are readable but are not a Capture One style.
    NotAStyle,
    /// The file could not be read.
    Io,
    /// The style uses something this build cannot import.
    Unsupported,
    Other,
}

impl StyleImportErrorKind {
    pub fn of(err: &anyhow::Error) -> Self {
        let lower = format!("{err:#}").to_ascii_lowercase();
        if lower.contains("is not a capture one style") || lower.contains("no <sl> root") {
            Self::NotAStyle
        } else if lower.contains("reading ") || lower.contains("os error") || lower.contains("no such file") {
            Self::Io
        } else if lower.contains("unsupported") || lower.contains("not supported") {
            Self::Unsupported
        } else {
            Self::Other
        }
    }
}

use crate::c1_model::{self, C1VariantSettings};
use crate::matchlook::{self, LookComponents, LookPrediction, LookReference, LookToneContext};
use crate::presets::{C1Preset, PresetKind};
use crate::AdjustmentSet;

/// One `.costyle` / `.copreset` mapped onto the engine's adjustments.
#[derive(Clone, Debug)]
pub struct C1Style {
    /// The file's own `Name`, or the file stem when it carries none.
    pub name: String,
    /// Root `<SL Engine="...">` value; 800 when absent.
    pub engine: u32,
    /// Every `K`/`V` pair in file order, including metadata this writer omits.
    pub original: Vec<(String, String)>,
    /// Raw `ColorCorrections` rows, so writing does not re-encode the editor.
    pub color_corrections: Vec<[f32; 18]>,
    pub adjustments: AdjustmentSet,
    /// `<AICGR>`: AI Color Grade reference frames and tool switches.
    pub look: Option<crate::matchlook::LookReference>,
}

/// Reads a style's `<AICGR>` block: 14 `<E K>` switches, `Impact` and the
/// base64 TIFF reference frame(s). Keys are the `AIColorGradingReferenceKey`
/// enum from `P1.AppCore`.
fn parse_aicgr(text: &str) -> Option<(LookReference, std::ops::Range<usize>)> {
    let document = roxmltree::Document::parse(text).ok()?;
    let node = document.descendants().find(|node| node.has_tag_name("AICGR"))?;
    let mut components = crate::matchlook::LookComponents::default();
    let mut bitmaps = None;
    for entry in node.children().filter(|child| child.has_tag_name("E")) {
        let (Some(key), value) = (entry.attribute("K"), entry.attribute("V").unwrap_or("")) else {
            continue;
        };
        let flag = || value.trim() != "0" && !value.trim().eq_ignore_ascii_case("false");
        match key {
            "Bitmaps" => bitmaps = Some(value),
            "Impact" => {
                components.impact = value.trim().parse().unwrap_or(100.0);
            }
            "BasicColorEdits" => components.basic_color_edits = flag(),
            "BlackAndWhite" => components.black_and_white = flag(),
            "Brightness" => components.brightness = flag(),
            "ColorBalanceHMS" => components.color_balance_hms = flag(),
            "ColorBalanceMaster" => components.color_balance_master = flag(),
            "ColorChannelCurves" => components.color_channel_curves = flag(),
            "Contrast" => components.contrast = flag(),
            "Exposure" => components.exposure = flag(),
            "HDRHighlightShadow" => components.hdr_highlight_shadow = flag(),
            "HDRWhiteBlack" => components.hdr_white_black = flag(),
            "LumaCurve" => components.luma_curve = flag(),
            "Saturation" => components.saturation = flag(),
            "TargetLevels" => components.target_levels = flag(),
            "Whitebalance" => components.whitebalance = flag(),
            _ => {}
        }
    }
    let thumbnails = bitmaps.map(decode_look_bitmaps).unwrap_or_default();
    Some((LookReference { components, thumbnails }, node.range()))
}

/// `Bitmaps` is the base64 TIFF written by Capture One, one page per reference.
fn decode_look_bitmaps(encoded: &str) -> Vec<crate::matchlook::LookThumbnail> {
    const MAX_ENCODED_BYTES: usize = 64 * 1024 * 1024;
    let encoded = encoded.trim();
    if encoded.len() > MAX_ENCODED_BYTES {
        log::warn!("style <AICGR> Bitmaps exceeds the encoded size limit");
        return Vec::new();
    }

    use base64::Engine as _;
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(encoded) else {
        log::warn!("style <AICGR> Bitmaps is not base64");
        return Vec::new();
    };
    let mut thumbnails = Vec::new();
    if let Err(error) = decode_look_tiff(bytes, &mut thumbnails) {
        log::warn!("style <AICGR> Bitmaps stopped at frame {}: {error:#}", thumbnails.len() + 1);
    }
    thumbnails
}

fn decode_look_tiff(
    bytes: Vec<u8>,
    thumbnails: &mut Vec<crate::matchlook::LookThumbnail>,
) -> Result<()> {
    use std::io::Cursor;
    use tiff::tags::ByteOrder;

    const MAX_FRAMES: usize = 64;
    const MAX_RGBA_BYTES: u64 = 64 * 1024 * 1024;
    let mut decoder = tiff::decoder::Decoder::new(Cursor::new(bytes))?;
    let order = decoder.byte_order();
    let bytes = decoder.inner().get_ref();
    let (header_offset, pointer_bytes) = match &bytes[..4] {
        b"II*\0" | b"MM\0*" => (4, 4),
        b"II+\0" | b"MM\0+" => (8, 8),
        _ => bail!("invalid TIFF header"),
    };
    let mut remaining = MAX_RGBA_BYTES;
    loop {
        if thumbnails.len() == MAX_FRAMES {
            bail!("too many TIFF reference frames");
        }
        let (width, height) = decoder.dimensions()?;
        let pixels = u64::from(width) * u64::from(height);
        if pixels == 0 || pixels > remaining / 4 {
            bail!("TIFF reference frames exceed the decoded size limit");
        }
        remaining -= pixels * 4;
        let ifd = decoder.ifd_pointer().context("missing TIFF image directory")?.0;
        let header = match (pointer_bytes, order) {
            (4, ByteOrder::LittleEndian) => u64::from(u32::try_from(ifd)?).to_le_bytes(),
            (4, ByteOrder::BigEndian) => (u64::from(u32::try_from(ifd)?) << 32).to_be_bytes(),
            (_, ByteOrder::LittleEndian) => ifd.to_le_bytes(),
            (_, ByteOrder::BigEndian) => ifd.to_be_bytes(),
        };
        let bytes = decoder.inner().get_mut();
        let pointer = header_offset..header_offset + pointer_bytes;
        let mut original = [0; 8];
        original[..pointer_bytes].copy_from_slice(&bytes[pointer.clone()]);
        // image decodes one page: point its header at the current page without moving TIFF data.
        bytes[pointer.clone()].copy_from_slice(&header[..pointer_bytes]);
        let mut reader = image::ImageReader::with_format(Cursor::new(&*bytes), image::ImageFormat::Tiff);
        let mut limits = image::Limits::default();
        limits.max_alloc = Some(MAX_RGBA_BYTES);
        reader.limits(limits);
        let decoded = reader.decode();
        bytes[pointer].copy_from_slice(&original[..pointer_bytes]);

        let rgba = decoded?.into_rgba8();
        thumbnails.push(crate::matchlook::LookThumbnail { width, height, rgba: rgba.into_raw() });
        if !decoder.more_images() {
            return Ok(());
        }
        decoder.next_image()?;
    }
}

/// Status shown when a C1 style asks for a processing branch this engine does
/// not implement. `Photography` is the only branch that renders normally.
pub fn processing_method_warning(method: crate::TechnicalProcessingMethod) -> Option<&'static str> {
    match method {
        crate::TechnicalProcessingMethod::Photography => None,
        crate::TechnicalProcessingMethod::ReproFilmNegatives
        | crate::TechnicalProcessingMethod::StandardFilmNegatives => {
            Some("C1 film negative processing is not implemented; rendering as normal photography")
        }
        crate::TechnicalProcessingMethod::ReproFilmPositives => {
            Some("C1 film positive processing is not implemented; rendering as normal photography")
        }
        crate::TechnicalProcessingMethod::Scientific => {
            Some("C1 scientific processing is not implemented; rendering as normal photography")
        }
    }
}

/// Reads one style or preset file.
///
/// A file without the `<SL>...</SL>` root is not a Capture One style — truncated
/// or another format — and is an error; a style that defines no tool group
/// yields the default [`AdjustmentSet`].
pub fn load_style(path: &Path) -> Result<C1Style> {
    load_style_impl(path, None, matchlook::predict_look_reference, &mut false)
}

/// Loads a style onto an existing target. Explicit style groups are overlaid
/// first, then AICGR tone tools use the resulting values and camera context.
pub fn load_style_for_target(
    path: &Path,
    target: &AdjustmentSet,
    context: &mut LookToneContext<'_>,
) -> Result<C1Style> {
    load_style_for_target_with(
        path,
        target,
        context,
        matchlook::predict_look_reference,
        &mut false,
    )
}

/// [`load_style_for_target`] with an explicit predictor and an outcome flag,
/// so the app path can be tested without the installed Match Look models.
pub(crate) fn load_style_for_target_with(
    path: &Path,
    target: &AdjustmentSet,
    context: &mut LookToneContext<'_>,
    predict: impl FnOnce(&LookReference) -> Option<LookPrediction>,
    look_applied: &mut bool,
) -> Result<C1Style> {
    load_style_impl(path, Some((target, context)), predict, look_applied)
}

fn load_style_impl(
    path: &Path,
    target: Option<(&AdjustmentSet, &mut LookToneContext<'_>)>,
    predict: impl FnOnce(&LookReference) -> Option<LookPrediction>,
    look_applied: &mut bool,
) -> Result<C1Style> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes);
    if !text.trim().is_empty() && !(text.contains("<SL") && text.contains("</SL>")) {
        bail!("{} is not a Capture One style: no <SL> root", path.display());
    }
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default();

    let mut settings = C1VariantSettings::defaults();
    let mut name = None;
    let (mut kelvin, mut tint) = (None, None);
    let parsed_look = parse_aicgr(&text);
    let original = crate::presets::parse_sl(&text);
    let static_entries;
    let entries = if let (true, Some((_, range))) = (target.is_some(), &parsed_look) {
        // AICGR tool switches can share names with numeric style adjustments.
        static_entries = [
            crate::presets::parse_sl(&text[..range.start]),
            crate::presets::parse_sl(&text[range.end..]),
        ].concat();
        &static_entries
    } else {
        &original
    };
    for (key, value) in entries {
        match key.as_str() {
            "Name" => name = Some(value.clone()),
            "WhiteBalanceTemperature" => kelvin = c1_model::parse_scalar(value),
            "WhiteBalanceTint" => tint = c1_model::parse_scalar(value),
            _ => apply(&mut settings, key, value),
        }
    }
    // Temperature and tint are the numbers Capture One's UI shows; the
    // `WhiteBalance` gains beside them are its internal form, so they only stand
    // in when the file carries no temperature.
    if let Some(k) = kelvin {
        settings.white_balance_gains = c1_model::kelvin_tint_to_gains(k, tint.unwrap_or(0.0));
    }

    let look = parsed_look.map(|(reference, _)| reference);
    let mut adjustments = defined_groups(&settings);
    let mut target_context = None;
    if let Some((base, context)) = target {
        adjustments = crate::styles::apply_style(base, &adjustments);
        target_context = Some(context);
    }
    if let Some(reference) = look.as_ref() {
        let outcome = if let Some(context) = target_context {
            predict(reference).map(|prediction| {
                let remaining = LookComponents {
                    exposure: false,
                    whitebalance: false,
                    brightness: false,
                    contrast: false,
                    ..reference.components
                };
                matchlook::apply_look_color(&prediction.color, &remaining, &mut adjustments);
                matchlook::apply_look_bw(&prediction.bw, &remaining, &mut adjustments);
                matchlook::apply_look_tone(&prediction, &reference.components, &mut adjustments, context);
                *look_applied = true;
            })
        } else {
            matchlook::look_from_reference(reference).map(|predicted| {
                adjustments = crate::styles::apply_style(&adjustments, &predicted);
            })
        };
        if outcome.is_none() {
            log::warn!("style {stem}: <AICGR> reference present but the Match Look heads are unavailable");
        }
    }
    Ok(C1Style {
        name: name.unwrap_or_else(|| stem.to_owned()),
        engine: engine_attr(&text).unwrap_or(800),
        original,
        color_corrections: settings.color_corrections,
        adjustments,
        look,
    })
}

fn engine_attr(text: &str) -> Option<u32> {
    let start = text.find("Engine=\"")? + "Engine=\"".len();
    let end = text[start..].find('"')?;
    text[start..start + end].parse().ok()
}

/// Keys this writer can emit. `C1Style.original` is filtered to this set so
/// metadata such as UUID/StyleSource is not written.
fn writable_key(key: &str) -> bool {
    matches!(
        key,
        "Name"
            | "TechnicalProcessingMethod"
            | "Exposure"
            | "Contrast"
            | "Brightness"
            | "Saturation"
            | "WhiteBalance"
            | "WhiteBalanceTemperature"
            | "WhiteBalanceTint"
            | "GradationCurve"
            | "GradationCurveRed"
            | "GradationCurveGreen"
            | "GradationCurveBlue"
            | "GradationCurveY"
            | "USMAmount"
            | "USMRadius"
            | "USMThreshold"
            | "FilmGrainAmount"
            | "FilmGrainGranularity"
            | "FilmGrainType"
            | "FilmCurve"
            | "BwEnabled"
            | "BwRed"
            | "BwYellow"
            | "BwGreen"
            | "BwCyan"
            | "BwBlue"
            | "BwMagenta"
            | "BwLightHue"
            | "BwLightSaturation"
            | "BwDarkHue"
            | "BwDarkSaturation"
            | "ColorCorrections"
    )
}

/// Writes a loaded style as C1 XML. The real file's header and Engine value are
/// preserved; keys outside [`writable_key`] are intentionally omitted.
pub fn write_costyle_text(style: &C1Style) -> String {
    let preset = C1Preset {
        id: style.name.clone(),
        name: style.name.clone(),
        group: String::new(),
        kind: PresetKind::Style,
        adjustments: style.adjustments.clone(),
        unmapped: Vec::new(),
        orig: style
            .original
            .iter()
            .filter(|(key, _)| writable_key(key))
            .cloned()
            .collect(),
        color_corrections: style.color_corrections.clone(),
    };
    let mut text = crate::presets::sl_text(&preset);
    text = text.replacen(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>",
        "<?xml version=\"1.0\"?>",
        1,
    );
    text = text.replacen(
        "<SL Engine=\"800\">",
        &format!("<SL Engine=\"{}\">", style.engine),
        1,
    );
    preserve_unchanged_values(&text, &style.original)
}

/// Capture One does not re-format numbers on write: its own files hold
/// `0.000,0.000;...` and `0.273999989032745,0.025000000372529;...` side by side,
/// so no single formatter can reproduce both. A value that still parses to the
/// same number keeps the file's own spelling; only edited values are formatted
/// canonically.
fn preserve_unchanged_values(text: &str, original: &[(String, String)]) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let Some(key) = attribute(line, "K=\"") else {
            out.push_str(line);
            continue;
        };
        let Some(value) = attribute(line, " V=\"") else {
            out.push_str(line);
            continue;
        };
        let Some(old) = original.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str()) else {
            out.push_str(line);
            continue;
        };
        if !writable_key(key) || !same_number_text(old, value) {
            out.push_str(line);
            continue;
        }
        let start = line.find(" V=\"").expect("value attribute") + 4;
        let end = start + line[start..].find('"').expect("closing quote");
        out.push_str(&line[..start]);
        out.push_str(&escape_attribute(old));
        out.push_str(&line[end..]);
    }
    out
}

/// The `.cos` edits that turn a layer's current `K`/`V` pairs into
/// `adjustments`. Reuses the `.costyle` writer's number spelling; `Name` is
/// skipped because `.cos` layers do not carry it.
pub fn cos_edits(
    tag: &str,
    adjustments: &AdjustmentSet,
    original: &[(String, String)],
) -> Vec<crate::c1_cos::CosEdit> {
    let preset = C1Preset {
        id: String::new(),
        name: String::new(),
        group: String::new(),
        kind: PresetKind::Style,
        adjustments: adjustments.clone(),
        unmapped: Vec::new(),
        orig: original.to_vec(),
        color_corrections: Vec::new(),
    };
    let text = preserve_unchanged_values(&crate::presets::sl_text(&preset), original);
    let mapped = crate::presets::mapped_values(adjustments, &[]);
    crate::presets::parse_sl(&text)
        .into_iter()
        .filter(|(key, _)| key != "Name")
        .filter(|(_, value)| !value.is_empty())
        .filter(|(key, value)| match original.iter().find(|(existing, _)| existing == key) {
            Some((_, existing)) => existing != value,
            None => mapped
                .iter()
                .any(|(mapped, text, is_default)| mapped == key && text == value && !*is_default),
        })
        .map(|(key, value)| crate::c1_cos::CosEdit::new(tag, key, value))
        .collect()
}

/// The text between `prefix` and the next `"` in a generated `<E/>` line.
fn attribute<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let start = line.find(prefix)? + prefix.len();
    let end = start + line[start..].find('"')?;
    Some(&line[start..end])
}

/// True when both strings hold the same numbers in the same shape. Non-numeric
/// values (Name, FilmCurve) have to match exactly. The tolerance absorbs the
/// white-balance `kelvin -> gains -> kelvin` round trip, which moves the fourth
/// decimal of the temperature.
fn same_number_text(a: &str, b: &str) -> bool {
    let left: Vec<&str> = a.split([';', ',']).map(str::trim).collect();
    let right: Vec<&str> = b.split([';', ',']).map(str::trim).collect();
    if left.len() != right.len() {
        return false;
    }
    left.iter().zip(&right).all(|(x, y)| match (x.parse::<f32>(), y.parse::<f32>()) {
        (Ok(x), Ok(y)) => (x - y).abs() <= 1e-6 * x.abs().max(y.abs()).max(1.0),
        _ => x == y,
    })
}

fn escape_attribute(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for c in value.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(c),
        }
    }
    out
}

/// A filesystem-safe `.costyle` file name for a style's display name.
///
/// C1's own files are named after the style (`Fujicolor Pro 160C.costyle`).
/// Characters Windows forbids become `_`, trailing spaces/dots are dropped,
/// reserved device names get a leading `_`, and the extension is added once.
pub fn style_file_name(name: &str) -> String {
    let mut stem: String = name
        .trim()
        .chars()
        .map(|c| match c {
            '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            c if c.is_control() => '_',
            c => c,
        })
        .collect();
    while stem.ends_with([' ', '.']) {
        stem.pop();
    }
    if stem.is_empty() {
        stem = "Untitled".to_owned();
    }
    let upper = stem.to_ascii_uppercase();
    let reserved = ["CON", "PRN", "AUX", "NUL"]
        .contains(&upper.as_str())
        || (upper.len() == 4
            && (upper.starts_with("COM") || upper.starts_with("LPT"))
            && upper.as_bytes()[3].is_ascii_digit());
    if reserved {
        stem.insert(0, '_');
    }
    if stem.to_ascii_lowercase().ends_with(".costyle") {
        stem
    } else {
        format!("{stem}.costyle")
    }
}

/// Writes a loaded style to `path`.
///
/// The text lands in a sibling temporary file first and is renamed into place,
/// so a failed write never leaves a half-written style; an existing file is
/// replaced. Windows cannot rename over an existing file, so the replace falls
/// back to remove + rename there (the only non-atomic window).
pub fn save_costyle(path: &Path, style: &C1Style) -> Result<()> {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let text = write_costyle_text(style);
    let stem = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("style");
    let tmp = path.with_file_name(format!(
        ".{stem}.{}.{}.tmp",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        let mut file =
            std::fs::File::create(&tmp).with_context(|| format!("writing {}", tmp.display()))?;
        file.write_all(text.as_bytes())
            .with_context(|| format!("writing {}", tmp.display()))?;
        file.sync_all()
            .with_context(|| format!("flushing {}", tmp.display()))?;
        drop(file);
        replace_file(&tmp, path)
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result.with_context(|| format!("saving {}", path.display()))
}

fn replace_file(tmp: &Path, path: &Path) -> Result<()> {
    match std::fs::rename(tmp, path) {
        Ok(()) => Ok(()),
        // Windows refuses to rename over an existing file. Removing first is
        // the only std-only way to replace it.
        Err(err) if cfg!(windows) && path.exists() => {
            std::fs::remove_file(path)
                .with_context(|| format!("replacing {}", path.display()))?;
            std::fs::rename(tmp, path)
                .with_context(|| format!("writing {}", path.display()))
        }
        Err(err) => Err(err).with_context(|| format!("writing {}", path.display())),
    }
}

/// Converts the settings and keeps only the tool groups the file defines.
///
/// `to_adjustment_set` fills in Capture One's neutral value for every tool it
/// covers, and some of those are not the engine's defaults (white balance comes
/// back as Custom / 6500 K, moire pattern as 8). A style that says nothing about
/// a tool has to leave it alone, so a group is taken from the converted settings
/// only when the file moved it off neutral.
fn defined_groups(s: &C1VariantSettings) -> AdjustmentSet {
    let full = c1_model::to_adjustment_set(s);
    let neutral = c1_model::to_adjustment_set(&C1VariantSettings::defaults());
    let mut out = AdjustmentSet::default();
    macro_rules! group {
        ($($field:ident),* $(,)?) => {
            $(
                if full.$field != neutral.$field {
                    out.$field = full.$field.clone();
                }
            )*
        };
    }
    group!(
        white_balance,
        exposure,
        hdr,
        levels,
        curves,
        clarity,
        dehaze,
        vignetting,
        color_balance,
        color_editor,
        black_white,
        sharpening,
        noise_reduction,
        film_grain,
        moire,
        lens,
        geometry,
    );
    // Base characteristics: a style can name the film curve, not the camera's
    // input profile, so only those two are taken.
    if full.base.film_curve != neutral.base.film_curve {
        out.base.film_curve = full.base.film_curve.clone();
    }
    if full.base.curve != neutral.base.curve {
        out.base.curve = full.base.curve;
    }
    if full.base.processing_method != neutral.base.processing_method {
        out.base.processing_method = full.base.processing_method;
    }
    out
}

/// Applies one `K`/`V` pair; a key this build does not map is ignored.
fn apply(s: &mut C1VariantSettings, key: &str, value: &str) {
    let n = c1_model::parse_scalar(value);
    match key {
        "Exposure" => s.exposure = n.unwrap_or(s.exposure),
        "Contrast" => s.contrast = n.unwrap_or(s.contrast),
        "Brightness" => s.brightness = n.unwrap_or(s.brightness),
        "Saturation" => s.saturation = n.unwrap_or(s.saturation),
        "WhiteBalance" => {
            if let Some(g) = gains(value) {
                s.white_balance_gains = g;
            }
        }
        "TechnicalProcessingMethod" => {
            if let Some(v) = n {
                s.technical_processing_method = v as i32;
            }
        }
        // `GCurve*` are Capture One's own legacy names for the same curve:
        // `P1.C1.Core` maps them onto `GradationCurve*` when it reads a style.
        "GradationCurve" | "GCurve" => s.gradation_curves[0] = c1_model::parse_curve(value),
        "GradationCurveRed" | "GCurveR" => s.gradation_curves[1] = c1_model::parse_curve(value),
        "GradationCurveGreen" | "GCurveG" => s.gradation_curves[2] = c1_model::parse_curve(value),
        "GradationCurveBlue" | "GCurveB" => s.gradation_curves[3] = c1_model::parse_curve(value),
        "GradationCurveY" => s.gradation_curves[4] = c1_model::parse_curve(value),
        "USMAmount" => s.sharpening.amount = n.unwrap_or(s.sharpening.amount),
        "USMRadius" => s.sharpening.radius = n.unwrap_or(s.sharpening.radius),
        "USMThreshold" => s.sharpening.threshold = n.unwrap_or(s.sharpening.threshold),
        "FilmGrainAmount" => s.film_grain.amount = n.unwrap_or(s.film_grain.amount),
        "FilmGrainGranularity" => s.film_grain.granularity = n.unwrap_or(s.film_grain.granularity),
        "FilmGrainType" => {
            if let Some(v) = n {
                s.film_grain.kind = v as i32;
            }
        }
        "FilmCurve" => s.film_curve = Some(value.trim().to_owned()),
        "BwEnabled" => s.black_white.enabled = n.is_some_and(|v| v != 0.0),
        "BwRed" => bw(s, 0, n),
        "BwYellow" => bw(s, 1, n),
        "BwGreen" => bw(s, 2, n),
        "BwCyan" => bw(s, 3, n),
        "BwBlue" => bw(s, 4, n),
        "BwMagenta" => bw(s, 5, n),
        "BwLightHue" => s.black_white.light_tone_hue = n.unwrap_or(s.black_white.light_tone_hue),
        "BwLightSaturation" => {
            s.black_white.light_tone_saturation = n.unwrap_or(s.black_white.light_tone_saturation)
        }
        "BwDarkHue" => s.black_white.dark_tone_hue = n.unwrap_or(s.black_white.dark_tone_hue),
        "BwDarkSaturation" => {
            s.black_white.dark_tone_saturation = n.unwrap_or(s.black_white.dark_tone_saturation)
        }
        // One `<E/>` per correction group, 18 comma-separated fields.
        "ColorCorrections" => {
            s.color_corrections.extend(c1_model::parse_color_corrections(value))
        }
        _ => {}
    }
}

fn bw(s: &mut C1VariantSettings, channel: usize, value: Option<f32>) {
    if let Some(v) = value {
        s.black_white.sensitivity[channel] = v;
    }
}

fn gains(value: &str) -> Option<[f32; 3]> {
    let v = c1_model::parse_vec(value);
    (v.len() == 3).then(|| [v[0], v[1], v[2]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn cos_edits_only_report_values_that_changed() {
        let original = vec![
            ("Exposure".to_string(), "1.25".to_string()),
            ("Contrast".to_string(), "-8".to_string()),
        ];
        let mut adjustments = AdjustmentSet::default();
        adjustments.exposure.exposure = -0.25;
        adjustments.exposure.contrast = -8.0;

        let edits = cos_edits("DL", &adjustments, &original);
        let exposure: Vec<&crate::c1_cos::CosEdit> =
            edits.iter().filter(|edit| edit.key == "Exposure").collect();
        assert_eq!(exposure.len(), 1);
        assert_eq!(exposure[0].tag, "DL");
        assert_eq!(exposure[0].value, "-0.25");

        // The untouched value keeps the file's own spelling and is not an edit.
        assert!(!edits.iter().any(|edit| edit.key == "Contrast"));
        assert!(!edits.iter().any(|edit| edit.key == "Name"));
    }

    fn testdata(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/c1-styles").join(name)
    }

    /// A scratch file that removes itself when the test ends.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str, body: &[u8]) -> Self {
            // Tests run in parallel; two files with the same name would let one
            // `Drop` delete the other's file mid-read.
            static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir()
                .join(format!("portrait-c1-style-{}-{n}-{name}", std::process::id()));
            std::fs::write(&path, body).expect("temp file");
            Self(path)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    #[test]
    fn a_real_style_maps_its_tools() {
        let style = load_style(&testdata("Fujicolor Pro 160C.costyle")).expect("real style parses");
        assert_eq!(style.name, "Fujicolor Pro 160C");
        let a = &style.adjustments;
        println!(
            "{}\texposure={}\tcontrast={}\tsaturation={}\tkelvin={}\ttint={}",
            style.name,
            a.exposure.exposure,
            a.exposure.contrast,
            a.exposure.saturation,
            a.white_balance.kelvin,
            a.white_balance.tint
        );
        assert!((a.exposure.exposure - 2.355544).abs() < 1e-3);
        assert!((a.exposure.contrast - 24.09511).abs() < 1e-3);
        assert!((a.exposure.brightness - -3.422448).abs() < 1e-3);
        // The temperature survives the round trip through C1's RGB gains.
        assert!((a.white_balance.kelvin - 2445.051).abs() < 0.5, "{}", a.white_balance.kelvin);
        assert!((a.white_balance.tint - -15.3893).abs() < 0.01);
        assert_eq!(a.curves.rgb.points.len(), 3);
        // The `.fcrv` extension is dropped, so the curve is found by stem.
        assert_eq!(a.base.film_curve.as_deref(), Some("PhaseOneIQ180-Linear Response"));
        assert_eq!(a.base.curve, crate::ToneCurvePreset::Linear);
    }

    #[test]
    fn every_real_sample_parses() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/c1-styles");
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .expect("sample folder")
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                matches!(p.extension().and_then(|e| e.to_str()), Some("copreset") | Some("costyle"))
            })
            .collect();
        files.sort();
        assert!(files.len() >= 5, "only {} samples", files.len());
        let mut non_default = 0;
        for file in &files {
            let style = load_style(file).unwrap_or_else(|e| panic!("{}: {e:#}", file.display()));
            let a = &style.adjustments;
            if !a.is_default() {
                non_default += 1;
            }
            println!(
                "{}\texposure={}\tcontrast={}\tsaturation={}",
                style.name, a.exposure.exposure, a.exposure.contrast, a.exposure.saturation
            );
        }
        assert_eq!(non_default, files.len(), "every sample defines something");
        let bw = load_style(&testdata("Color - Panchromatic 1.copreset")).expect("bw preset");
        assert!(bw.adjustments.black_white.enabled);
        assert_eq!(bw.adjustments.black_white.sensitivity[0], -16.26506);
        let blue = load_style(&testdata("Effect - Blue only.copreset")).expect("color editor");
        let advanced = &blue.adjustments.color_editor.advanced;
        assert_eq!(advanced.len(), 1, "kind 0 is an advanced group");
        assert!(advanced[0].shift.saturation < -99.0);
        let band = blue.adjustments.color_editor.advanced_bands[0].1;
        assert!(band.inverted, "field 15 is the inverted flag");
    }

    #[test]
    fn a_broken_file_is_an_error() {
        let garbage = Scratch::new("garbage.costyle", b"PK\x03\x04 not a style");
        assert!(load_style(&garbage.0).is_err());
        let truncated = Scratch::new("truncated.costyle", b"<?xml version=\"1.0\"?>\n<SL><E K=");
        assert!(load_style(&truncated.0).is_err(), "an unterminated root is not a style");
        assert!(load_style(Path::new("no-such-file.costyle")).is_err());
    }

    #[test]
    fn unsupported_keys_are_ignored_without_panic() {
        let plain = Scratch::new(
            "plain.costyle",
            br#"<?xml version="1.0"?>
<SL Engine="800"><E K="Exposure" V="0.5"/></SL>"#,
        );
        let noisy = Scratch::new(
            "noisy.costyle",
            br#"<?xml version="1.0"?>
<SL Engine="1000">
  <E K="Exposure" V="0.5"/>
  <E K="Highlight" V="0.507402;0.792946;0.858029;1.000000"/>
  <E K="Midtone" V="0;0;0;0"/>
  <E K="Shadow" V="0.171981;0.198540;0.238351;0"/>
  <E K="TargetHighlight" V="1;1;1;0.940427"/>
  <E K="CurveAlias" V="0.514,0.583;0.210,0.260"/>
  <E K="ColorBalance" V="1;1;1"/>
  <E K="Flip" V="1"/>
  <E K="UUID" V="{82466223-FC65-4BC5-8965-0CAA0E5B6813}"/>
  <E K="StyleSource" V="Curve"/>
  <E K="NotAThing" V=""/>
</SL>"#,
        );
        let plain = load_style(&plain.0).expect("plain parses");
        let noisy = load_style(&noisy.0).expect("unknown keys do not break the load");
        assert_eq!(noisy.adjustments, plain.adjustments);
    }

    /// `GCurve*` is Capture One's legacy spelling; its own property map turns it
    /// into `GradationCurve*`, so both files must load the same curve.
    #[test]
    fn gcurve_is_the_legacy_name_for_gradation_curve() {
        let legacy = Scratch::new(
            "legacy.costyle",
            br#"<SL Engine="800"><E K="GCurve" V="0.514,0.583;0.210,0.260"/></SL>"#,
        );
        let current = Scratch::new(
            "current.costyle",
            br#"<SL Engine="800"><E K="GradationCurve" V="0.514,0.583;0.210,0.260"/></SL>"#,
        );
        let legacy = load_style(&legacy.0).expect("legacy curve");
        let current = load_style(&current.0).expect("current curve");
        assert_eq!(legacy.adjustments.curves.rgb, current.adjustments.curves.rgb);
        assert_ne!(legacy.adjustments.curves.rgb, crate::Curve::default());
    }

    /// The installed `Presets60/Curve/Contrast Luma.copreset` carries this value.
    #[test]
    fn gradation_curve_y_reaches_the_luma_curve() {
        let y = Scratch::new(
            "luma.costyle",
            br#"<SL Engine="800"><E K="GradationCurveY" V="0.000,0.000;0.104,0.075;0.252,0.254;0.549,0.640;1.000,1.000"/></SL>"#,
        );
        let style = load_style(&y.0).expect("luma curve");
        let a = &style.adjustments;
        assert_eq!(a.curves.luma.points.len(), 5);
        assert!((a.curves.luma.points[3].y - 0.640).abs() < 1e-6);
        assert_eq!(a.curves.rgb, crate::Curve::default(), "Y must not move the RGB curve");
    }

    /// A legacy style writes back under the canonical key and still loads the
    /// same curve, so the alias does not leak into saved files.
    #[test]
    fn a_gcurve_style_round_trips_through_the_writer() {
        let legacy = Scratch::new(
            "legacy.costyle",
            br#"<SL Engine="800"><E K="Name" V="Legacy"/><E K="GCurve" V="0.514,0.583;0.210,0.260"/></SL>"#,
        );
        let loaded = load_style(&legacy.0).expect("legacy curve");
        let text = write_costyle_text(&loaded);
        assert!(!text.contains("GCurve\""), "the writer canonicalises the key");
        assert!(text.contains("GradationCurve\""));
        let again = Scratch::new("again.costyle", text.as_bytes());
        let reloaded = load_style(&again.0).expect("written curve");
        assert_eq!(reloaded.adjustments.curves.rgb, loaded.adjustments.curves.rgb);
    }

    #[test]
    fn an_empty_style_is_the_default_adjustment_set() {
        let empty = Scratch::new("empty.costyle", b"");
        let root_only = Scratch::new(
            "root.costyle",
            br#"<?xml version="1.0"?>
<SL Engine="800"></SL>"#,
        );
        for path in [&empty.0, &root_only.0] {
            let style = load_style(path).expect("empty style parses");
            assert!(style.adjustments.is_default(), "{}", path.display());
            assert_eq!(style.name, path.file_stem().unwrap().to_str().unwrap());
        }
    }

    #[test]
    fn a_style_leaves_the_tools_it_does_not_define_alone() {
        let exposure_only = Scratch::new(
            "exposure.costyle",
            br#"<SL Engine="800"><E K="Exposure" V="-1.000"/><E K="Name" V="One stop down"/></SL>"#,
        );
        let style = load_style(&exposure_only.0).expect("parses");
        assert_eq!(style.adjustments.exposure.exposure, -1.0);
        assert_eq!(style.adjustments.white_balance, AdjustmentSet::default().white_balance);
        assert_eq!(style.adjustments.moire, AdjustmentSet::default().moire);
    }

    #[test]
    fn technical_processing_method_reaches_the_style() {
        let file = Scratch::new(
            "negative.costyle",
            br#"<SL Engine="800"><E K="TechnicalProcessingMethod" V="2"/></SL>"#,
        );
        let style = load_style(&file.0).expect("parses");
        assert_eq!(
            style.adjustments.base.processing_method,
            crate::TechnicalProcessingMethod::ReproFilmNegatives
        );
    }

    #[test]
    fn only_non_photography_processing_methods_warn() {
        assert_eq!(
            processing_method_warning(crate::TechnicalProcessingMethod::Photography),
            None
        );
        for method in [
            crate::TechnicalProcessingMethod::Scientific,
            crate::TechnicalProcessingMethod::ReproFilmNegatives,
            crate::TechnicalProcessingMethod::ReproFilmPositives,
            crate::TechnicalProcessingMethod::StandardFilmNegatives,
        ] {
            assert!(processing_method_warning(method).is_some(), "missing warning for {method:?}");
        }
    }

    #[test]
    #[ignore = "needs the local Capture One style installation"]
    fn installed_styles_report_all_processing_methods() {
        fn collect(dir: &Path, out: &mut Vec<PathBuf>) {
            for entry in std::fs::read_dir(dir).expect("style directory") {
                let path = entry.expect("entry").path();
                if path.is_dir() {
                    collect(&path, out);
                } else if path.extension().and_then(|e| e.to_str()) == Some("costyle") {
                    out.push(path);
                }
            }
        }
        let root = Path::new(r"C:\Program Files\Capture One\Capture One");
        let mut files = Vec::new();
        collect(root, &mut files);
        let mut methods = std::collections::BTreeMap::new();
        for path in &files {
            let style = load_style(path).unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
            *methods.entry(style.adjustments.base.processing_method).or_insert(0usize) += 1;
        }
        assert_eq!(files.len(), 25, "installed style count");
        assert_eq!(
            methods,
            std::collections::BTreeMap::from([
                (crate::TechnicalProcessingMethod::Photography, 3),
                (crate::TechnicalProcessingMethod::ReproFilmNegatives, 11),
                (crate::TechnicalProcessingMethod::ReproFilmPositives, 11),
            ])
        );
    }

    #[test]
    fn written_style_round_trips_mapped_values() {
        let original = load_style(&testdata("Fujicolor Pro 160C.costyle")).expect("source style");
        let text = write_costyle_text(&original);
        assert!(
            text.starts_with("<?xml version=\"1.0\"?>\n<SL Engine=\""),
            "unexpected header: {text}"
        );
        assert!(text.contains(&format!("<SL Engine=\"{}\">", original.engine)), "{text}");
        let file = Scratch::new("written.costyle", b"");
        std::fs::write(&file.0, text.as_bytes()).expect("write temp style");
        let read = load_style(&file.0).expect("reload written style");
        assert_eq!(read.name, original.name);
        assert_eq!(read.engine, original.engine);
        // C1 persists the processing method inside a style, so it must survive.
        assert_eq!(read.adjustments, original.adjustments);
        assert_eq!(read.color_corrections, original.color_corrections);
    }

    #[test]
    fn writer_omits_metadata_and_unmapped_keys() {
        let style = load_style(&testdata("B&W Generic 4.costyle")).expect("source style");
        let text = write_costyle_text(&style);
        for key in ["UUID", "StyleSource", "Flip", "Color_tag_index"] {
            assert!(!text.contains(&format!("K=\"{key}\"")), "{key} leaked into {text}");
        }
        assert!(text.contains("K=\"Name\""), "{text}");
        assert!(text.contains("K=\"Exposure\""), "{text}");
    }

    #[test]
    fn writer_field_diff_against_real_sample() {
        // Layer two: same parameters in, field-level diff out, order ignored.
        let path = testdata("Fujicolor Pro 160C.costyle");
        let source = std::fs::read_to_string(&path).expect("sample text");
        let style = load_style(&path).expect("source style");
        let written = write_costyle_text(&style);
        let before = crate::presets::parse_sl(&source);
        let after = crate::presets::parse_sl(&written);

        let mut dropped = Vec::new();
        let mut changed = Vec::new();
        for (key, old) in &before {
            match after.iter().find(|(k, _)| k == key) {
                None => dropped.push(key.clone()),
                Some((_, new)) => {
                    if old != new {
                        changed.push((key.clone(), old.clone(), new.clone()));
                    }
                }
            }
        }
        for (key, _) in &after {
            assert!(before.iter().any(|(k, _)| k == key), "writer invented {key}");
        }
        assert_eq!(
            dropped,
            vec![
                "ColorBalance",
                "ColorBalanceHighlight",
                "ColorBalanceMidtone",
                "ColorBalanceShadow",
                "Color_tag_index",
                "Flip",
                "Highlight",
                "Midtone",
                "Shadow",
                "UUID",
            ]
        );
        println!("kept {} keys, dropped {} keys", after.len(), dropped.len());
        assert!(changed.is_empty(), "format drift in kept keys: {changed:?}");
    }

    #[test]
    fn writer_reproduces_capture_one_number_spelling() {
        let path = testdata("Fujicolor Pro 160C.costyle");
        let style = load_style(&path).expect("source style");
        let before = crate::presets::parse_sl(&std::fs::read_to_string(&path).expect("sample"));
        let after = crate::presets::parse_sl(&write_costyle_text(&style));
        for key in [
            "GradationCurve",
            "GradationCurveRed",
            "GradationCurveGreen",
            "GradationCurveBlue",
            "WhiteBalanceTemperature",
            "WhiteBalanceTint",
        ] {
            let old = before.iter().find(|(k, _)| k == key).expect("original key").1.clone();
            let new = after.iter().find(|(k, _)| k == key).expect("written key").1.clone();
            assert_eq!(old, new, "{key}");
        }
    }

    #[test]
    fn unchanged_numbers_keep_their_original_spelling() {
        // Integer, zero, negative zero, one, half, and exponent forms.
        assert!(same_number_text("0", "0.000000"));
        assert!(same_number_text("1", "1.000000"));
        assert!(same_number_text("-0", "0"));
        assert!(same_number_text("0.5", "0.500"));
        assert!(same_number_text("1e-3", "0.001"));
        assert!(!same_number_text("0", "0.001"));
        assert!(!same_number_text("1;2", "1"), "shape mismatch is a change");
        assert!(!same_number_text("Camera A", "Camera B"));
    }

    #[test]
    fn edited_values_are_written_canonically() {
        let mut style = load_style(&testdata("Fujicolor Pro 160C.costyle")).expect("source style");
        style.adjustments.exposure.exposure = -1.5;
        let text = write_costyle_text(&style);
        assert!(text.contains("K=\"Exposure\" V=\"-1.5\""), "{text}");
        let file = Scratch::new("edited.costyle", text.as_bytes());
        let read = load_style(&file.0).expect("reload edited style");
        assert_eq!(read.adjustments.exposure.exposure, -1.5);
    }

    #[test]
    fn high_precision_curves_are_not_shortened() {
        let file = Scratch::new(
            "precise.costyle",
            br#"<SL Engine="800"><E K="GradationCurve" V="0,0;0.273999989032745,0.025000000372529;1,1"/></SL>"#,
        );
        let style = load_style(&file.0).expect("parses");
        let text = write_costyle_text(&style);
        assert!(
            text.contains("0.273999989032745,0.025000000372529"),
            "the sample's own precision was reformatted: {text}"
        );
    }

    fn fnv1a<I: Iterator<Item = u8>>(bytes: I) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for byte in bytes {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    /// End to end on one real NEF: the engine writes a style, loads it again,
    /// and the rendered frame must match the frame from the source style byte
    /// for byte. Values in the three source styles come from installed C1 files.
    #[test]
    #[ignore = "needs AURORA_TEST_RAW (a real NEF) and libraw"]
    fn written_styles_render_the_same_frame_as_their_source() {
        let Ok(raw) = std::env::var("AURORA_TEST_RAW") else {
            panic!("set AURORA_TEST_RAW to a real NEF");
        };
        let raw = PathBuf::from(raw);
        let (source, width, height, sixteen) = match crate::raw::decode16(&raw) {
            Some((w, h, rgba)) => {
                (crate::mock::process::Source::from_rgba16(w, h, &rgba), w, h, true)
            }
            None => {
                let img = crate::raw::decode(&raw).expect("decode raw");
                let (w, h) = (img.width(), img.height());
                (crate::mock::process::Source::from_rgba(w, h, img.as_raw()), w, h, false)
            }
        };
        println!("source\t{width}x{height}\tsixteen={sixteen}");
        let neutral = crate::mock::process::render16(
            &source,
            &AdjustmentSet::default(),
            crate::NormRect::FULL,
            1024,
            false,
        );
        let cases = [
            (
                "curve",
                r#"<SL Engine="800">
<E K="Name" V="E2E Curve"/>
<E K="GradationCurve" V="0.000,0.000;0.504,0.450;1.000,1.000"/>
</SL>"#,
            ),
            (
                "wb",
                r#"<SL Engine="800">
<E K="Name" V="E2E WB"/>
<E K="WhiteBalanceTemperature" V="2445.051"/>
<E K="WhiteBalanceTint" V="-15.3893"/>
</SL>"#,
            ),
            (
                "bw",
                r#"<SL Engine="800">
<E K="Name" V="E2E BW"/>
<E K="BwEnabled" V="1"/>
<E K="BwRed" V="-16.26506"/>
<E K="BwYellow" V="12.5"/>
<E K="BwGreen" V="30"/>
<E K="BwCyan" V="-10"/>
<E K="BwBlue" V="-25"/>
<E K="BwMagenta" V="5"/>
</SL>"#,
            ),
        ];
        for (label, text) in cases {
            let original = Scratch::new(&format!("e2e-src-{label}.costyle"), text.as_bytes());
            let source_style = load_style(&original.0).expect("source style parses");
            let written = write_costyle_text(&source_style);
            let copy = Scratch::new(&format!("e2e-out-{label}.costyle"), written.as_bytes());
            let written_style = load_style(&copy.0).expect("written style parses");
            assert_eq!(
                source_style.adjustments, written_style.adjustments,
                "{label}: field values changed on the round trip"
            );
            let a = crate::mock::process::render16(
                &source,
                &crate::styles::resolve(&source_style.adjustments),
                crate::NormRect::FULL,
                1024,
                false,
            );
            let b = crate::mock::process::render16(
                &source,
                &crate::styles::resolve(&written_style.adjustments),
                crate::NormRect::FULL,
                1024,
                false,
            );
            assert_eq!((a.width, a.height), (b.width, b.height), "{label}: frame size");
            assert_eq!(a.rgba, b.rgba, "{label}: rendered frames differ");
            assert_ne!(a.rgba, neutral.rgba, "{label}: the style did not change the frame");
            let hash = fnv1a(a.rgba.iter().flat_map(|v| v.to_le_bytes()));
            println!("{label}\t{}x{}\tfnv1a64={hash:016x}", a.width, a.height);
            if let Ok(dir) = std::env::var("AURORA_E2E_OUT") {
                let pixels: Vec<u8> = a
                    .rgba
                    .chunks_exact(4)
                    .flat_map(|p| [(p[0] >> 8) as u8, (p[1] >> 8) as u8, (p[2] >> 8) as u8, (p[3] >> 8) as u8])
                    .collect();
                let img = image::RgbaImage::from_raw(a.width, a.height, pixels).expect("frame size");
                let out = Path::new(&dir).join(format!("{label}.png"));
                img.save(&out).expect("png export");
                println!("png\t{}", out.display());
            }
        }
    }

    #[test]
    fn style_file_names_are_safe_and_keep_the_extension() {
        assert_eq!(style_file_name("Fujicolor Pro 160C"), "Fujicolor Pro 160C.costyle");
        assert_eq!(style_file_name("  Fuji  "), "Fuji.costyle");
        assert_eq!(style_file_name("A/B:C*?\"<>|"), "A_B_C______.costyle");
        assert_eq!(style_file_name("dots... "), "dots.costyle");
        assert_eq!(style_file_name(""), "Untitled.costyle");
        assert_eq!(style_file_name("CON"), "_CON.costyle");
        assert_eq!(style_file_name("Look.costyle"), "Look.costyle");
        assert_eq!(style_file_name("Look.COSTYLE"), "Look.COSTYLE");
    }

    #[test]
    fn save_replaces_an_existing_style_without_leaving_temp_files() {
        let dir = std::env::temp_dir().join(format!("portrait-costyle-save-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        let path = dir.join("Look.costyle");
        let mut style = load_style(&testdata("Fujicolor Pro 160C.costyle")).expect("source style");
        style.adjustments.exposure.exposure = 1.0;
        save_costyle(&path, &style).expect("first save");
        style.adjustments.exposure.exposure = -2.0;
        save_costyle(&path, &style).expect("overwrite");
        let read = load_style(&path).expect("reload");
        assert_eq!(read.adjustments.exposure.exposure, -2.0);
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .expect("dir")
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temp files left behind: {leftovers:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_save_keeps_the_existing_path() {
        let dir = std::env::temp_dir().join(format!("portrait-costyle-fail-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("temp dir");
        // A directory where the style should go makes the final rename fail.
        let path = dir.join("Taken.costyle");
        std::fs::create_dir(&path).expect("occupied path");
        let style = load_style(&testdata("Fujicolor Pro 160C.costyle")).expect("source style");
        assert!(save_costyle(&path, &style).is_err(), "saving over a directory must fail");
        assert!(path.is_dir(), "the failed save must not remove the existing path");
        let leftovers: Vec<_> = std::fs::read_dir(&dir)
            .expect("dir")
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "temp files left behind: {leftovers:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn style_import_failures_map_to_stable_kinds() {
        use anyhow::anyhow;
        assert_eq!(
            StyleImportErrorKind::of(&anyhow!("foo.costyle is not a Capture One style: no <SL> root")),
            StyleImportErrorKind::NotAStyle
        );
        assert_eq!(
            StyleImportErrorKind::of(&anyhow!("reading C:/x.costyle: No such file (os error 2)")),
            StyleImportErrorKind::Io
        );
        assert_eq!(StyleImportErrorKind::of(&anyhow!("Unsupported version")), StyleImportErrorKind::Unsupported);
        assert_eq!(StyleImportErrorKind::of(&anyhow!("other")), StyleImportErrorKind::Other);
    }

    #[test]
    fn aicgr_import_preserves_tiff_page_order_and_keeps_valid_prefix() {
        use base64::Engine as _;
        use std::io::Cursor;
        use tiff::encoder::{TiffEncoder, colortype::{RGB8, RGBA16}};

        let mut bytes = Vec::new();
        let mut encoder = TiffEncoder::new(Cursor::new(&mut bytes)).unwrap();
        encoder.write_image::<RGB8>(2, 1, &[10, 20, 30, 40, 50, 60]).unwrap();
        encoder.write_image::<RGBA16>(1, 1, &[0, 65535, 0, 65535]).unwrap();

        let encode = |bytes: &[u8]| base64::engine::general_purpose::STANDARD.encode(bytes);
        let text = format!(
            r#"<SL><AICGR><E K="Bitmaps" V="{}"/></AICGR></SL>"#,
            encode(&bytes)
        );
        let thumbnails = parse_aicgr(&text).unwrap().0.thumbnails;
        assert_eq!(thumbnails.len(), 2);
        assert_eq!((thumbnails[0].width, thumbnails[0].height), (2, 1));
        assert_eq!(thumbnails[0].rgba, [10, 20, 30, 255, 40, 50, 60, 255]);
        assert_eq!((thumbnails[1].width, thumbnails[1].height), (1, 1));
        assert_eq!(thumbnails[1].rgba, [0, 255, 0, 255]);

        bytes.pop();
        assert_eq!(decode_look_bitmaps(&encode(&bytes)), thumbnails[..1]);
    }

    struct TestStyleCamera;

    impl matchlook::LookWhiteBalanceContext for TestStyleCamera {
        fn target_temperature(&self, target: &crate::WhiteBalance) -> (f32, f32) {
            let [kelvin, tint, _] = target.gains.expect("test gains");
            (kelvin, tint)
        }

        fn temperature_to_gains(&self, kelvin: f32, tint: f32) -> [f32; 3] {
            [kelvin, tint, 1.0]
        }
    }

    #[test]
    fn aicgr_target_load_combines_existing_tone_and_respects_switches() {
        use base64::Engine as _;
        let mut tiff = Vec::new();
        tiff::encoder::TiffEncoder::new(std::io::Cursor::new(&mut tiff))
            .unwrap().write_image::<tiff::encoder::colortype::RGB8>(1, 1, &[10, 20, 30]).unwrap();
        let bitmaps = base64::engine::general_purpose::STANDARD.encode(tiff);
        let xml = format!(
            r#"<SL><AICGR><E K="Bitmaps" V="{bitmaps}"/><E K="Impact" V="50"/><E K="Exposure" V="1"/><E K="Brightness" V="0"/><E K="Contrast" V="0"/></AICGR></SL>"#,
        );
        let file = Scratch::new("look-target.costyle", xml.as_bytes());
        let mut target = AdjustmentSet::default();
        target.exposure.exposure = 1.0;
        target.exposure.brightness = -30.0;
        target.exposure.contrast = 12.0;
        target.white_balance.gains = Some([7000.0, 10.0, 1.0]);
        let mut context = LookToneContext {
            subtract_baseline: [0.0; 5], add_baseline: [0.0; 5], weights: [Some(1.0); 4],
            auto_white_balance: false, white_balance: &TestStyleCamera,
        };
        let style = load_style_impl(&file.0, Some((&target, &mut context)), |reference| {
            assert_eq!(reference.thumbnails[0].rgba, [10, 20, 30, 255]);
            let mut color = [0.5; matchlook::LOOK_COLOR_VECTOR];
            color[..5].copy_from_slice(&[0.75, 0.5, 0.75, 1.0, 1.0]);
            color[23] = 0.75;
            Some(LookPrediction { color, bw: [0.5; matchlook::LOOK_BW_VECTOR] })
        }, &mut false).expect("load onto target");

        let out = style.adjustments;
        assert_eq!(out.exposure.exposure, 2.0);
        assert_eq!(out.exposure.brightness, -30.0);
        assert_eq!(out.exposure.contrast, 12.0);
        assert_eq!(out.exposure.saturation, 25.0);
        let gains = out.white_balance.gains.expect("converted gains");
        assert!((gains[0] - 7000.0).abs() < 0.01);
        assert_eq!(gains[1], 35.0);
        assert_eq!(context.weights, [Some(0.7), Some(0.7), Some(1.0), Some(1.0)]);
        assert_eq!(target.exposure.exposure, 1.0);
    }

    #[test]
    fn ordinary_target_style_keeps_the_target_when_a_group_is_absent() {
        let file = Scratch::new("static-target.costyle", br#"<SL><E K="Exposure" V="2"/></SL>"#);
        let mut target = AdjustmentSet::default();
        target.white_balance.tint = 15.0;
        let mut context = LookToneContext {
            subtract_baseline: [0.0; 5], add_baseline: [0.0; 5], weights: [None; 4],
            auto_white_balance: false, white_balance: &TestStyleCamera,
        };
        let loaded = load_style_for_target(&file.0, &target, &mut context).unwrap();
        assert_eq!(loaded.adjustments.exposure.exposure, 2.0);
        assert_eq!(loaded.adjustments.white_balance, target.white_balance);
        assert_eq!(context.weights, [None; 4]);
        assert_eq!(load_style(&file.0).unwrap().adjustments.white_balance.tint, 0.0);
    }

    #[test]
    fn aicgr_switches_impact_and_bitmap_are_parsed() {
        use base64::Engine as _;
        use image::ImageEncoder as _;
        let image = image::RgbaImage::from_pixel(4, 4, image::Rgba([10, 20, 30, 255]));
        let mut cursor = std::io::Cursor::new(Vec::new());
        image::codecs::tiff::TiffEncoder::new(&mut cursor)
            .write_image(
                image.as_raw(),
                image.width(),
                image.height(),
                image::ExtendedColorType::Rgba8,
            )
            .expect("tiff");
        let tiff = cursor.into_inner();
        let bitmaps = base64::engine::general_purpose::STANDARD.encode(&tiff);
        let text = format!(
            r#"<SL Engine="800"><AICGR><E K="Bitmaps" V="{bitmaps}"/><E K="Impact" V="70"/><E K="Saturation" V="0"/></AICGR></SL>"#
        );
        let (look, _) = parse_aicgr(&text).expect("AICGR");
        assert_eq!(look.components.impact, 70.0);
        assert!(!look.components.saturation);
        assert!(look.components.exposure);
        assert_eq!(look.thumbnails.len(), 1);
        assert_eq!((look.thumbnails[0].width, look.thumbnails[0].height), (4, 4));
        assert_eq!(look.thumbnails[0].rgba.len(), 4 * 4 * 4);
    }
}
