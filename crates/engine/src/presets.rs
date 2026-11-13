//! Capture One presets (`.copreset`) and styles (`.costyle`), plus the container
//! of its private `P1XFormation` colour transforms.
//!
//! A preset or style is plain XML: `<SL Engine="800"><E K="Exposure" V="-1.000"/>`.
//! The `K` names are Capture One's own property names, so the mapping table below
//! is a rename, not a reconstruction. Values are decimal floats, `;` vectors
//! (`ColorBalance` = r;g;b) and `x,y;x,y` curves, the same encodings as
//! `ZVARIANTLAYER`.
//!
//! `.P1X` is *not* a preset: `ImgCoreDll.dll` calls it `CP1XSerializer` /
//! `POColorTransform::Create_from_P1X` and it holds a colour-transform pipeline
//! (stages `Matrix`, `CLUT`, `CurveSet`, `Lab2XYZ`, …) built for Fuji film
//! simulations. Its asset payload goes through `CBaseCodec`, whose decode-method
//! ids we did not recover, so only the container is read here.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{Context, Result};

use crate::{param_meta, AdjustmentSet, CurvePoint, GrainKind, StyleInfo};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PresetKind {
    Preset,
    Style,
}

/// One `.copreset` / `.costyle` mapped onto the engine's adjustments.
#[derive(Clone, Debug)]
pub struct C1Preset {
    /// File stem, unique within a folder.
    pub id: String,
    pub name: String,
    /// Folder path below the presets or styles root, `/` separated.
    pub group: String,
    pub kind: PresetKind,
    pub adjustments: AdjustmentSet,
    /// Keys this build does not map; kept so nothing is silently dropped.
    pub unmapped: Vec<(String, String)>,
    /// Every `K`/`V` pair of the file this preset was read from, in file
    /// order. Writing a preset follows this list, so metadata keys and values
    /// we cannot regenerate are preserved.
    pub orig: Vec<(String, String)>,
    /// `ColorCorrections` rows as C1 wrote them, so they can be written back
    /// without re-encoding the editor's bands and shifts.
    pub color_corrections: Vec<[f32; 18]>,
}

/// `K`/`V` pairs of one `<E/>` element, in file order.
pub fn parse_sl(text: &str) -> Vec<(String, String)> {
    let text = text.trim_start_matches('\u{feff}');
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find("<E ") {
        rest = &rest[start + 3..];
        let end = rest.find('>').unwrap_or(rest.len());
        let element = &rest[..end];
        rest = &rest[end..];
        let (Some(k), Some(v)) = (attr(element, "K="), attr(element, "V=")) else { continue };
        out.push((k, v));
    }
    out
}

fn attr(element: &str, key: &str) -> Option<String> {
    let start = element.find(key)? + key.len();
    let rest = element[start..].trim_start();
    let rest = rest.strip_prefix('"')?;
    let raw = &rest[..rest.find('"')?];
    // The attribute holds XML entities (`B&amp;W`); hand back the plain text.
    if raw.contains('&') {
        let mut out = String::with_capacity(raw.len());
        let mut rest = raw;
        while let Some(at) = rest.find('&') {
            out.push_str(&rest[..at]);
            rest = &rest[at..];
            let end = rest.find(';').unwrap_or(rest.len());
            let entity = &rest[1..end.min(rest.len())];
            match entity {
                "amp" => out.push('&'),
                "lt" => out.push('<'),
                "gt" => out.push('>'),
                "quot" => out.push('"'),
                "apos" => out.push('\''),
                other => {
                    out.push('&');
                    out.push_str(other);
                    out.push(';');
                }
            }
            rest = &rest[(end + 1).min(rest.len())..];
        }
        out.push_str(rest);
        return Some(out);
    }
    Some(raw.to_owned())
}

/// Maps one `K`/`V` pair onto `set`; `false` when the key is not mapped.
fn apply_key(set: &mut AdjustmentSet, key: &str, value: &str) -> bool {
    let num = || value.trim().parse::<f32>().ok();
    match key {
        "Exposure" => set.exposure.exposure = num().unwrap_or(0.0),
        "Contrast" => set.exposure.contrast = num().unwrap_or(0.0),
        "Brightness" => set.exposure.brightness = num().unwrap_or(0.0),
        "Saturation" => set.exposure.saturation = num().unwrap_or(0.0),
        // C1 stores the temperature itself, not gains: 5000.0009765625 in the
        // bundled presets, which is 5000 K as a float.
        "WhiteBalanceTemperature" => set.white_balance.kelvin = num().unwrap_or(6500.0),
        "WhiteBalanceTint" => set.white_balance.tint = num().unwrap_or(0.0),
        "GradationCurve" => set.curves.rgb = curve(value),
        "GradationCurveRed" => set.curves.red = curve(value),
        "GradationCurveGreen" => set.curves.green = curve(value),
        "GradationCurveBlue" => set.curves.blue = curve(value),
        // The Curve tool's own keys use the same `x,y;x,y` encoding.
        "GCurve" => set.curves.rgb = curve(value),
        "GCurveR" => set.curves.red = curve(value),
        "GCurveG" => set.curves.green = curve(value),
        "GCurveB" => set.curves.blue = curve(value),
        // Name-for-name only: C1's Y curve is not proven to be our luma curve.
        "GradationCurveY" => set.curves.luma = curve(value),
        // The Levels tool: four `;`-separated values, master first, then R/G/B.
        "Shadow" => levels_edge(&mut set.levels, 0, value),
        "Highlight" => levels_edge(&mut set.levels, 1, value),
        "TargetShadow" => levels_edge(&mut set.levels, 2, value),
        "TargetHighlight" => levels_edge(&mut set.levels, 3, value),
        "Midtone" => levels_midtone(&mut set.levels, value),
        // C1 stores the white balance as camera RGB gains beside its UI
        // temperature; the gains only stand in when no temperature is present.
        // shortcut: gains→kelvin uses the engine's 5600 K reference, unverified
        // against Capture One's own conversion; revisit with a C1-exported pair.
        "WhiteBalance" => white_balance_gains(set, value),
        "USMAmount" => set.sharpening.amount = num().unwrap_or(0.0),
        "USMRadius" => set.sharpening.radius = num().unwrap_or(0.0),
        "USMThreshold" => {
            set.sharpening.threshold = param_meta::clamp("USMThreshold", num().unwrap_or(0.0))
        }
        "FilmGrainAmount" => set.film_grain.impact = num().unwrap_or(0.0),
        "FilmGrainGranularity" => set.film_grain.granularity = num().unwrap_or(0.0),
        // shortcut: index taken as our GrainKind order, C1's enum order unproven.
        "FilmGrainType" => {
            if let Some(i) = num() {
                set.film_grain.kind = GrainKind::ALL[(i.max(0.0) as usize).min(GrainKind::ALL.len() - 1)];
            }
        }
        "FilmCurve" => {
            // `curve_lut` is keyed by file stem, so drop the extension.
            let stem = Path::new(value.trim()).file_stem().and_then(|s| s.to_str()).unwrap_or(value.trim());
            set.base.film_curve = Some(stem.to_owned());
        }
        "BwEnabled" => set.black_white.enabled = num().is_some_and(|v| v != 0.0),
        "BwRed" => bw_sensitivity(set, 0, num()),
        "BwYellow" => bw_sensitivity(set, 1, num()),
        "BwGreen" => bw_sensitivity(set, 2, num()),
        "BwCyan" => bw_sensitivity(set, 3, num()),
        "BwBlue" => bw_sensitivity(set, 4, num()),
        "BwMagenta" => bw_sensitivity(set, 5, num()),
        "BwLightHue" => set.black_white.highlight_tone.hue = num().unwrap_or(0.0),
        "BwLightSaturation" => set.black_white.highlight_tone.saturation = num().unwrap_or(0.0),
        "BwDarkHue" => set.black_white.shadow_tone.hue = num().unwrap_or(0.0),
        "BwDarkSaturation" => set.black_white.shadow_tone.saturation = num().unwrap_or(0.0),
        _ => return false,
    }
    true
}

fn bw_sensitivity(set: &mut AdjustmentSet, channel: usize, value: Option<f32>) {
    if let Some(v) = value {
        set.black_white.sensitivity[channel] = param_meta::clamp(param_meta::BW_CHANNELS[channel], v);
    }
}

fn levels_edge(levels: &mut crate::Levels, slot: usize, value: &str) {
    let v: Vec<f32> = value.split(';').filter_map(|x| x.trim().parse().ok()).collect();
    if v.len() != 4 {
        return;
    }
    let channels = [&mut levels.rgb, &mut levels.red, &mut levels.green, &mut levels.blue];
    for (i, channel) in channels.into_iter().enumerate() {
        let x = v[i].clamp(0.0, 1.0);
        match slot {
            0 => channel.input_black = x,
            1 => channel.input_white = x,
            2 => channel.output_black = x,
            _ => channel.output_white = x,
        }
    }
}

fn levels_midtone(levels: &mut crate::Levels, value: &str) {
    let v: Vec<f32> = value.split(';').filter_map(|x| x.trim().parse().ok()).collect();
    if v.len() != 4 {
        return;
    }
    let channels = [&mut levels.rgb, &mut levels.red, &mut levels.green, &mut levels.blue];
    for (i, channel) in channels.into_iter().enumerate() {
        channel.gamma = crate::gamma_from_c1_midtone(v[i]);
    }
}

fn white_balance_gains(set: &mut AdjustmentSet, value: &str) {
    let v: Vec<f32> = value.split(';').filter_map(|x| x.trim().parse().ok()).collect();
    if v.len() != 3 {
        return;
    }
    let (kelvin, tint) = crate::c1_model::gains_to_kelvin_tint([v[0], v[1], v[2]]);
    set.white_balance.mode = crate::WhiteBalanceMode::Custom;
    set.white_balance.kelvin = kelvin.clamp(2000.0, 50000.0);
    set.white_balance.tint = tint.clamp(-50.0, 50.0);
}

fn curve(value: &str) -> crate::Curve {
    let points: Vec<CurvePoint> = value
        .split(';')
        .filter_map(|pair| {
            let (x, y) = pair.split_once(',')?;
            Some(CurvePoint::new(x.trim().parse().ok()?, y.trim().parse().ok()?))
        })
        .collect();
    if points.len() < 2 {
        return crate::Curve::default();
    }
    crate::Curve { points }
}

/// Writes a preset back as Capture One's `.copreset`.
pub fn save_preset(path: &Path, preset: &C1Preset) -> Result<()> {
    write_sl(path, preset)
}

/// Writes a style back as Capture One's `.costyle`. Same format as a preset.
pub fn save_style(path: &Path, preset: &C1Preset) -> Result<()> {
    write_sl(path, preset)
}

fn write_sl(path: &Path, preset: &C1Preset) -> Result<()> {
    let text = sl_text(preset);
    std::fs::write(path, text).with_context(|| format!("writing {}", path.display()))
}

/// The `.copreset` / `.costyle` text for a preset.
///
/// A key is written when the value differs from the engine's default for that
/// field, so a preset that says nothing about a tool keeps saying nothing:
/// reloading gives the same adjustments. Keys this build does not map are
/// written back exactly as they were read, vectors (`;`) and all.
///
/// The original file order is kept, so a saved preset still looks like a
/// Capture One one.
pub fn sl_text(preset: &C1Preset) -> String {
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"utf-8\"?>\n<SL Engine=\"800\">\n");
    let mapped = mapped_values(&preset.adjustments, &preset.color_corrections);
    let mut written: Vec<String> = Vec::new();
    for (key, value) in &preset.orig {
        let text = match key.as_str() {
            "Name" => Some(preset.name.clone()),
            // Capture One's grain-type enum order is not ours, so keep the
            // file's own number rather than write our index back — unless the
            // kind was changed, in which case our index is the best we have.
            "FilmGrainType" => {
                let same = value.parse::<f32>().ok().is_some_and(|n| {
                    let index = (n.max(0.0) as usize).min(GrainKind::ALL.len() - 1);
                    GrainKind::ALL[index] == preset.adjustments.film_grain.kind
                });
                if same {
                    None
                } else {
                    mapped.iter().find(|(k, _, _)| k == key).map(|(_, v, _)| v.clone())
                }
            }
            // The file names the curve file with its extension; keep it.
            "FilmCurve" => {
                let stem = Path::new(value).file_stem().and_then(|s| s.to_str());
                if stem == preset.adjustments.base.film_curve.as_deref() {
                    None
                } else {
                    mapped.iter().find(|(k, _, _)| k == key).map(|(_, v, _)| v.clone())
                }
            }
            // The Curve tool's keys and `GradationCurve*` hold the same curve:
            // write the value back under the file's own key, and keep the
            // equivalent gradation key from being appended beside it.
            "GCurve" | "GCurveR" | "GCurveG" | "GCurveB" => {
                let default = AdjustmentSet::default();
                let (gradation, is_default) = match key.as_str() {
                    "GCurve" => ("GradationCurve", preset.adjustments.curves.rgb == default.curves.rgb),
                    "GCurveR" => ("GradationCurveRed", preset.adjustments.curves.red == default.curves.red),
                    "GCurveG" => ("GradationCurveGreen", preset.adjustments.curves.green == default.curves.green),
                    _ => ("GradationCurveBlue", preset.adjustments.curves.blue == default.curves.blue),
                };
                written.push(gradation.to_owned());
                // An empty `V` means "no curve"; keep it rather than write a
                // neutral one beside it.
                if is_default {
                    None
                } else {
                    mapped.iter().find(|(k, _, _)| k == gradation).map(|(_, v, _)| v.clone())
                }
            }
            _ => mapped.iter().find(|(k, _, _)| k == key).map(|(_, v, _)| v.clone()),
        };
        let text = match text {
            Some(t) => t,
            // Not a key we regenerate: keep the file's own value.
            None => value.clone(),
        };
        push(&mut out, key, &text);
        written.push(key.clone());
    }
    if !written.iter().any(|k| k == "Name") {
        push(&mut out, "Name", &preset.name);
    }
    // Keys the caller added that the file did not have.
    for (key, value) in &preset.unmapped {
        if !written.iter().any(|k| k == key) {
            push(&mut out, key, value);
            written.push(key.clone());
        }
    }
    // Keys the file did not have, but the adjustments now define.
    for (key, value, is_default) in &mapped {
        if !is_default && !written.iter().any(|k| k == key) {
            push(&mut out, key, value);
        }
    }
    out.push_str("</SL>\n");
    out
}

fn push(out: &mut String, key: &str, value: &str) {
    out.push_str("  <E K=\"");
    out.push_str(&escape(key));
    out.push_str("\" V=\"");
    out.push_str(&escape(value));
    out.push_str("\" />\n");
}

fn escape(value: &str) -> String {
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

/// The mapped keys of an adjustment set, in the encoding `parse_sl` reads,
/// each with whether it still holds the engine's default value.
fn mapped_values(a: &AdjustmentSet, corrections: &[[f32; 18]]) -> Vec<(String, String, bool)> {
    let d = AdjustmentSet::default();
    let mut out: Vec<(String, String, bool)> = Vec::new();
    macro_rules! num {
        ($key:expr, $value:expr, $default:expr) => {
            out.push(($key.to_owned(), fmt_f32($value), $value == $default));
        };
    }
    num!("Exposure", a.exposure.exposure, d.exposure.exposure);
    num!("Contrast", a.exposure.contrast, d.exposure.contrast);
    num!("Brightness", a.exposure.brightness, d.exposure.brightness);
    num!("Saturation", a.exposure.saturation, d.exposure.saturation);
    num!("WhiteBalanceTemperature", a.white_balance.kelvin, d.white_balance.kelvin);
    num!("WhiteBalanceTint", a.white_balance.tint, d.white_balance.tint);
    for (key, curve) in [
        ("GradationCurve", &a.curves.rgb),
        ("GradationCurveRed", &a.curves.red),
        ("GradationCurveGreen", &a.curves.green),
        ("GradationCurveBlue", &a.curves.blue),
        ("GradationCurveY", &a.curves.luma),
    ] {
        let is_default = *curve == d.curves.rgb;
        out.push((key.to_owned(), fmt_curve(curve), is_default));
    }
    num!("USMAmount", a.sharpening.amount, d.sharpening.amount);
    num!("USMRadius", a.sharpening.radius, d.sharpening.radius);
    num!("USMThreshold", a.sharpening.threshold, d.sharpening.threshold);
    num!("FilmGrainAmount", a.film_grain.impact, d.film_grain.impact);
    num!("FilmGrainGranularity", a.film_grain.granularity, d.film_grain.granularity);
    if let Some(index) = GrainKind::ALL.iter().position(|k| *k == a.film_grain.kind) {
        out.push((
            "FilmGrainType".to_owned(),
            index.to_string(),
            a.film_grain.kind == d.film_grain.kind,
        ));
    }
    if let Some(stem) = &a.base.film_curve {
        let value = if stem.ends_with(".fcrv") { stem.clone() } else { format!("{stem}.fcrv") };
        out.push(("FilmCurve".to_owned(), value, false));
    }
    out.push((
        "BwEnabled".to_owned(),
        if a.black_white.enabled { "1" } else { "0" }.to_owned(),
        !a.black_white.enabled,
    ));
    for (key, value) in [
        ("BwRed", a.black_white.sensitivity[0]),
        ("BwYellow", a.black_white.sensitivity[1]),
        ("BwGreen", a.black_white.sensitivity[2]),
        ("BwCyan", a.black_white.sensitivity[3]),
        ("BwBlue", a.black_white.sensitivity[4]),
        ("BwMagenta", a.black_white.sensitivity[5]),
    ] {
        out.push((key.to_owned(), fmt_f32(value), value == 0.0));
    }
    num!("BwLightHue", a.black_white.highlight_tone.hue, 0.0);
    num!("BwLightSaturation", a.black_white.highlight_tone.saturation, 0.0);
    num!("BwDarkHue", a.black_white.shadow_tone.hue, 0.0);
    num!("BwDarkSaturation", a.black_white.shadow_tone.saturation, 0.0);
    // The Levels tool keeps one four-value vector per key, master first.
    for (key, slot) in [
        ("Shadow", 0usize),
        ("Highlight", 1),
        ("TargetShadow", 2),
        ("TargetHighlight", 3),
    ] {
        let value = levels_text(&a.levels, slot, false);
        out.push((key.to_owned(), value.clone(), value == levels_text(&d.levels, slot, false)));
    }
    let midtone = levels_text(&a.levels, 0, true);
    out.push(("Midtone".to_owned(), midtone.clone(), midtone == levels_text(&d.levels, 0, true)));
    // shortcut: the rows are written as read; edits made to the converted
    // editor are not re-encoded until one round-trips through C1.
    out.push((
        "ColorCorrections".to_owned(),
        rows_text(corrections),
        corrections.is_empty(),
    ));
    out
}

fn levels_text(levels: &crate::Levels, slot: usize, midtone: bool) -> String {
    let channels = [&levels.rgb, &levels.red, &levels.green, &levels.blue];
    let mut out = Vec::with_capacity(4);
    for channel in channels {
        let value = if midtone {
            crate::c1_midtone_from_gamma(channel.gamma)
        } else {
            match slot {
                0 => channel.input_black,
                1 => channel.input_white,
                2 => channel.output_black,
                _ => channel.output_white,
            }
        };
        out.push(fmt_f32(value));
    }
    out.join(";")
}

fn rows_text(rows: &[[f32; 18]]) -> String {
    rows.iter()
        .map(|row| row.iter().map(|v| fmt_f32(*v)).collect::<Vec<_>>().join(","))
        .collect::<Vec<_>>()
        .join(";")
}

fn fmt_f32(value: f32) -> String {
    if value == 0.0 { "0".to_owned() } else { format!("{value}") }
}

fn fmt_curve(curve: &crate::Curve) -> String {
    curve
        .points
        .iter()
        .map(|p| format!("{},{}", fmt_f32(p.x), fmt_f32(p.y)))
        .collect::<Vec<_>>()
        .join(";")
}

/// Parses one preset file. `None` when it holds no recognisable `K`/`V` pairs.
pub fn preset_from_sl(
    text: &str,
    kind: PresetKind,
    group: &str,
    fallback_name: &str,
) -> Option<C1Preset> {
    let pairs = parse_sl(text);
    if pairs.is_empty() {
        return None;
    }
    let mut adjustments = AdjustmentSet::default();
    let mut name = None;
    let mut unmapped = Vec::new();
    let mut corrections: Vec<[f32; 18]> = Vec::new();
    // Capture One's UI temperature and its internal gains are two forms of the
    // same white balance; the temperature wins when both are present.
    let has_temperature = pairs.iter().any(|(k, _)| k == "WhiteBalanceTemperature");
    for (k, v) in &pairs {
        match k.as_str() {
            "Name" => name = Some(v.clone()),
            // Metadata, not an adjustment: UUID, StyleSource, Flip,
            // TechnicalProcessingMethod, Color_tag_index and the ColorBalance
            // family, whose C1 semantics are still unproven.
            "UUID" | "StyleSource" | "Flip" | "TechnicalProcessingMethod" | "Color_tag_index" => {}
            "WhiteBalance" if has_temperature => {}
            "ColorCorrections" => corrections.extend(crate::c1_model::parse_color_corrections(v)),
            _ => {
                if !apply_key(&mut adjustments, k, v) {
                    unmapped.push((k.clone(), v.clone()));
                }
            }
        }
    }
    adjustments.color_editor = crate::c1_model::color_editor_from_c1(&corrections);
    Some(C1Preset {
        id: fallback_name.to_owned(),
        name: name.unwrap_or_else(|| fallback_name.to_owned()),
        group: group.to_owned(),
        kind,
        adjustments,
        unmapped,
        orig: pairs,
        color_corrections: corrections,
    })
}

/// Directory holding presets and styles: `LUMEN_C1_DIR`, else this machine's install.
pub fn c1_dir() -> PathBuf {
    std::env::var_os("LUMEN_C1_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Program Files\Capture One\Capture One"))
}

static CACHE: Mutex<Option<(PathBuf, Vec<C1Preset>)>> = Mutex::new(None);

/// Every preset and style below `dir`, loaded once per path.
pub fn scan(dir: &Path) -> Vec<C1Preset> {
    let mut guard = CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((cached, list)) = guard.as_ref() {
        if cached == dir {
            return list.clone();
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort_by(|a, b| a.1.group.cmp(&b.1.group).then_with(|| a.1.name.cmp(&b.1.name)));
    let list: Vec<C1Preset> = out.iter().map(|(_, p)| p.clone()).collect();
    *guard = Some((dir.to_owned(), list.clone()));
    list
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<(PathBuf, C1Preset)>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(root, &path, out);
            continue;
        }
        let kind = match path.extension().and_then(|e| e.to_str()) {
            Some("copreset") => PresetKind::Preset,
            Some("costyle") => PresetKind::Style,
            _ => continue,
        };
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else { continue };
        let Ok(bytes) = std::fs::read(&path) else { continue };
        let text = String::from_utf8_lossy(&bytes);
        let group = path.parent().and_then(|p| p.strip_prefix(root).ok()).map(|p| {
            p.components().map(|c| c.as_os_str().to_string_lossy()).collect::<Vec<_>>().join(" / ")
        });
        if let Some(preset) = preset_from_sl(&text, kind, group.as_deref().unwrap_or(""), stem) {
            out.push((path, preset));
        }
    }
}

/// Capture One's installed presets and styles as engine styles, so the panel
/// shows the real list instead of the built-in demo one.
pub fn styles() -> Vec<StyleInfo> {
    style_cache().clone().unwrap_or_default()
}

static STYLES: Mutex<Option<Vec<StyleInfo>>> = Mutex::new(None);

fn style_cache() -> std::sync::MutexGuard<'static, Option<Vec<StyleInfo>>> {
    let mut guard = STYLES.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        let dir = c1_dir();
        let mut found = Vec::new();
        for sub in ["Presets60", "CulturalHeritage/Styles"] {
            found.extend(scan(&dir.join(sub)));
        }
        *guard = Some(
            found
                .into_iter()
                .map(|p| StyleInfo { name: p.name, group: p.group, adjustments: p.adjustments })
                .collect(),
        );
    }
    guard
}

/// One installed style by name. Cached, so `styles::resolve` can call it per frame.
pub fn style_by_name(name: &str) -> Option<StyleInfo> {
    style_cache().as_ref()?.iter().find(|s| s.name == name).cloned()
}

/// One entry of a `P1XFormation` asset table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct P1xAsset {
    pub tag: u8,
    pub flag: u8,
    pub kind: u16,
    pub len: usize,
    pub offset: usize,
}

impl P1xAsset {
    pub fn bytes<'a>(&self, file: &'a [u8]) -> &'a [u8] {
        let end = (self.offset + self.len).min(file.len());
        file.get(self.offset..end).unwrap_or(&[])
    }
}

/// Header of a `.P1X` file: `u32` length, `"P1XFormation\0"`, `u32` version,
/// then 12-byte assets (tag, flag, `u32` length, `u32` offset, `u16` kind).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct P1xContainer {
    pub version: u32,
    pub assets: Vec<P1xAsset>,
}

const P1X_MAGIC: &[u8] = b"P1XFormation\0";
const P1X_ASSETS: usize = 0x1E;
const P1X_ASSET_LEN: usize = 12;

pub fn parse_p1x(bytes: &[u8]) -> Result<P1xContainer> {
    let declared = bytes.get(..4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
    anyhow::ensure!(declared == Some(bytes.len() as u32), "P1X length header does not match the file");
    anyhow::ensure!(bytes.get(4..17) == Some(P1X_MAGIC), "not a P1XFormation file");
    let version = u32::from_be_bytes([bytes[0x11], bytes[0x12], bytes[0x13], bytes[0x14]]);
    let mut assets = Vec::new();
    let mut at = P1X_ASSETS;
    while at + P1X_ASSET_LEN <= bytes.len() {
        let tag = bytes[at];
        if tag != 0xF0 && tag != 0xF1 {
            break;
        }
        let len = u32::from_be_bytes([bytes[at + 2], bytes[at + 3], bytes[at + 4], bytes[at + 5]]) as usize;
        let offset = u32::from_be_bytes([bytes[at + 6], bytes[at + 7], bytes[at + 8], bytes[at + 9]]) as usize;
        let kind = u16::from_be_bytes([bytes[at + 10], bytes[at + 11]]);
        anyhow::ensure!(offset + len <= bytes.len(), "P1X asset {tag:#04x} runs past the end");
        assets.push(P1xAsset { tag, flag: bytes[at + 1], kind, len, offset });
        at += P1X_ASSET_LEN;
    }
    anyhow::ensure!(!assets.is_empty(), "P1X has no asset table");
    Ok(P1xContainer { version, assets })
}

/// Names of the Fuji film simulations shipped as `.P1X`, keyed for a picker.
pub fn p1x_names(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Vec::new() };
    let mut names: Vec<String> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("P1X"))
        .filter_map(|p| p.file_stem().and_then(|s| s.to_str()).map(str::to_owned))
        .collect();
    names.sort();
    names
}

/// Ensures a caller can turn a preset into a style; kept so `styles::apply_style`
/// is exercised against real Capture One data.
pub fn as_style(preset: &C1Preset) -> StyleInfo {
    StyleInfo { name: preset.name.clone(), group: preset.group.clone(), adjustments: preset.adjustments.clone() }
}

/// Loads one file; exposed for callers that already know the path.
pub fn load(path: &Path) -> Result<C1Preset> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let kind = match path.extension().and_then(|e| e.to_str()) {
        Some("costyle") => PresetKind::Style,
        _ => PresetKind::Preset,
    };
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
    preset_from_sl(&text, kind, "", stem).context("preset has no K/V pairs")
}

/// Groups presets by their folder, for a menu.
pub fn grouped(presets: &[C1Preset]) -> HashMap<String, Vec<&C1Preset>> {
    let mut out: HashMap<String, Vec<&C1Preset>> = HashMap::new();
    for p in presets {
        out.entry(p.group.clone()).or_default().push(p);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const EXPOSURE: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<SL Engine="600">
  <E K="Exposure" V="-1.000" />
  <E K="Contrast" V="0.000" />
  <E K="Name" V="Exposure compensation -1 fstop" />
  <E K="UUID" V="{82466223-FC65-4BC5-8965-0CAA0E5B6813}" />
</SL>"#;


    /// A scratch directory that cleans itself up.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let path = std::env::temp_dir().join(format!("lumen-c1-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&path);
            std::fs::create_dir_all(&path).expect("temp dir");
            Self(path)
        }

        fn file(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// Two adjustment sets equal within 1e-4 on every number.
    fn approx_eq(a: &AdjustmentSet, b: &AdjustmentSet) -> bool {
        fn close(a: &Value, b: &Value) -> bool {
            match (a, b) {
                (Value::Number(x), Value::Number(y)) => {
                    let (x, y) = (x.as_f64().unwrap_or(f64::NAN), y.as_f64().unwrap_or(f64::NAN));
                    (x - y).abs() <= 1e-4
                }
                (Value::Object(x), Value::Object(y)) => {
                    x.len() == y.len()
                        && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| close(v, w)))
                }
                (Value::Array(x), Value::Array(y)) => {
                    x.len() == y.len() && x.iter().zip(y).all(|(v, w)| close(v, w))
                }
                (x, y) => x == y,
            }
        }
        close(&serde_json::to_value(a).unwrap(), &serde_json::to_value(b).unwrap())
    }

    fn sorted(pairs: &[(String, String)]) -> Vec<(String, String)> {
        let mut out = pairs.to_vec();
        out.sort();
        out
    }

    /// The keys of a pair list, sorted.
    fn keys(pairs: &[(String, String)]) -> Vec<String> {
        let mut out: Vec<String> = pairs.iter().map(|(k, _)| k.clone()).collect();
        out.sort();
        out
    }


    /// Every `.copreset` / `.costyle` below `dir`, recursively.
    fn collect_presets(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_presets(&path, out);
                continue;
            }
            if matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("copreset") | Some("costyle")
            ) {
                out.push(path);
            }
        }
    }

    #[test]
    fn parses_the_key_value_pairs() {
        let pairs = parse_sl(EXPOSURE);
        assert_eq!(pairs.len(), 4);
        assert_eq!(pairs[0], ("Exposure".to_owned(), "-1.000".to_owned()));
    }





    /// The `.copreset` path and the `ZVARIANTLAYER` path must clamp identically:
    /// both ask `param_meta::clamp`, which reads the one catalog range.
    #[test]
    fn preset_and_c1_row_clamp_out_of_range_values_the_same_way() {
        let text = r#"<?xml version="1.0" encoding="utf-8"?>
<SL Engine="800">
  <E K="USMThreshold" V="30" />
  <E K="BwRed" V="120" />
  <E K="BwBlue" V="-130" />
</SL>"#;
        let p = preset_from_sl(text, PresetKind::Preset, "", "synthetic").expect("preset parses");
        let mut row: std::collections::HashMap<&str, Option<String>> = std::collections::HashMap::new();
        row.insert("ZUSMTHRESHOLD", Some("30".to_string()));
        row.insert("ZBWRED", Some("120".to_string()));
        row.insert("ZBWBLUE", Some("-130".to_string()));
        let set = crate::c1_model::to_adjustment_set(&crate::c1_model::from_layer_row(&row));

        assert_eq!(p.adjustments.sharpening.threshold, 12.0);
        assert_eq!(set.sharpening.threshold, 12.0);
        assert_eq!(p.adjustments.black_white.sensitivity, set.black_white.sensitivity);
        assert_eq!(p.adjustments.black_white.sensitivity[0], 100.0);
        assert_eq!(p.adjustments.black_white.sensitivity[4], -100.0);
    }

    #[test]
    fn preset_keeps_in_range_threshold_and_sensitivity() {
        let text = r#"<?xml version="1.0" encoding="utf-8"?>
<SL Engine="800">
  <E K="USMThreshold" V="12" />
  <E K="BwGreen" V="-42.5" />
  <E K="BwMagenta" V="100" />
</SL>"#;
        let p = preset_from_sl(text, PresetKind::Preset, "", "synthetic").expect("preset parses");
        assert_eq!(p.adjustments.sharpening.threshold, 12.0);
        assert_eq!(p.adjustments.black_white.sensitivity[2], -42.5);
        assert_eq!(p.adjustments.black_white.sensitivity[5], 100.0);
    }

    /// `.lumen` is the serialized `AdjustmentSet`; loading one must hand back the
    /// stored numbers unchanged, in range or not, or a later save would rewrite
    /// the user's document behind their back.
    #[test]
    fn lumen_round_trip_does_not_clamp_stored_values() {
        let mut a = AdjustmentSet::default();
        a.sharpening.threshold = 12.0;
        a.black_white.sensitivity = [100.0, -100.0, 0.0, 0.0, 0.0, 0.0];
        let back: AdjustmentSet = serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
        assert_eq!(back, a);

        // A document written before the preset clamp keeps its stored number.
        let mut legacy = a.clone();
        legacy.sharpening.threshold = 30.0;
        let back: AdjustmentSet =
            serde_json::from_str(&serde_json::to_string(&legacy).unwrap()).unwrap();
        assert_eq!(back.sharpening.threshold, 30.0);
    }


    #[test]
    fn unmapped_keys_are_kept_not_dropped() {
        let text = r#"<SL><E K="ColorBalanceHighlight" V="1.000000;1.000000;1.000000"/><E K="Name" V="x"/></SL>"#;
        let p = preset_from_sl(text, PresetKind::Style, "", "x").expect("parses");
        assert_eq!(p.unmapped, vec![("ColorBalanceHighlight".to_owned(), "1.000000;1.000000;1.000000".to_owned())]);
    }

    #[test]
    fn truncated_or_foreign_xml_yields_nothing() {
        assert!(preset_from_sl("", PresetKind::Preset, "", "x").is_none());
        assert!(preset_from_sl("<SL><E K=", PresetKind::Preset, "", "x").is_none());
        assert!(parse_p1x(b"nope").is_err());
    }






    /// The Curve tool's keys carry the same `x,y;x,y` data as `GradationCurve`;
    /// values from the installed `Mid tones - brighter.copreset`.
    #[test]
    fn gcurve_keys_fill_the_rgb_curve() {
        let text = r#"<SL Engine="600">
  <E K="GCurve" V="0.514,0.583;0.210,0.260" />
  <E K="GCurveR" V="" />
  <E K="GCurveG" V="" />
  <E K="GCurveB" V="" />
  <E K="Name" V="Mid tones - brighter" />
</SL>"#;
        let p = preset_from_sl(text, PresetKind::Preset, "", "x").expect("parses");
        assert!(p.unmapped.is_empty(), "{:?}", p.unmapped);
        let pts = &p.adjustments.curves.rgb.points;
        assert_eq!(pts.len(), 2);
        assert!((pts[0].x - 0.514).abs() < 1e-6 && (pts[0].y - 0.583).abs() < 1e-6);
        assert!((pts[1].x - 0.210).abs() < 1e-6 && (pts[1].y - 0.260).abs() < 1e-6);
        let default = AdjustmentSet::default();
        let out = crate::styles::apply_style(&default, &p.adjustments);
        assert_ne!(out.curves, default.curves, "the preset must move the curve");
    }

    /// Four `;`-separated values, master first then R/G/B. Values from the
    /// installed `Fujicolor Pro 160C.costyle`.
    #[test]
    fn levels_keys_fill_the_scale_from_the_file() {
        let text = r#"<SL Engine="800">
  <E K="Highlight" V="0.507402;0.792946;0.858029;1.000000" />
  <E K="Midtone" V="0.000000;0.000000;0.000000;0.000000" />
  <E K="Shadow" V="0.171981;0.198540;0.238351;0.000000" />
  <E K="TargetHighlight" V="1.000000;1.000000;1.000000;0.940427" />
</SL>"#;
        let p = preset_from_sl(text, PresetKind::Style, "", "x").expect("parses");
        let l = &p.adjustments.levels;
        assert!((l.rgb.input_white - 0.507402).abs() < 1e-5, "{}", l.rgb.input_white);
        assert!((l.red.input_white - 0.792946).abs() < 1e-5);
        assert!((l.blue.input_white - 1.0).abs() < 1e-6);
        assert!((l.rgb.input_black - 0.171981).abs() < 1e-5);
        assert!((l.blue.input_black - 0.0).abs() < 1e-6);
        assert!((l.rgb.gamma - 1.0).abs() < 1e-5);
        assert!((l.blue.output_white - 0.940427).abs() < 1e-5);
        assert!((l.rgb.output_white - 1.0).abs() < 1e-6);
        assert!(p.unmapped.is_empty(), "{:?}", p.unmapped);
    }

    #[test]
    fn levels_and_editor_stay_default_when_the_file_has_none() {
        let text = r#"<SL Engine="800"><E K="Exposure" V="0.5"/></SL>"#;
        let p = preset_from_sl(text, PresetKind::Preset, "", "x").expect("parses");
        let default = AdjustmentSet::default();
        assert_eq!(p.adjustments.levels, default.levels);
        assert_eq!(p.adjustments.color_editor, default.color_editor);
    }

    /// The installed `Effect - Blue only.copreset` row.
    #[test]
    fn color_corrections_build_the_color_editor() {
        let text = r#"<SL Engine="600"><E K="ColorCorrections" V="1,1,0,0.000000,-100.000000,0.000000,58.000000,82.000000,126.000000,-20.000000,20.000000,-53.968250,46.031750,20.000000,0.000000,1,0.0,0.0"/></SL>"#;
        let p = preset_from_sl(text, PresetKind::Preset, "", "x").expect("parses");
        assert!(p.unmapped.is_empty(), "{:?}", p.unmapped);
        let ce = &p.adjustments.color_editor;
        assert_eq!(ce.advanced.len(), 1, "kind 0 is an advanced group");
        assert!(ce.advanced[0].shift.saturation < -99.0);
        assert!(ce.advanced_bands[0].1.inverted, "field 15 is the inverted flag");
        let default = AdjustmentSet::default();
        assert_ne!(ce, &default.color_editor);
        assert_ne!(
            crate::styles::apply_style(&default, &p.adjustments).color_editor,
            default.color_editor
        );
    }

    /// `WhiteBalance` holds camera RGB gains; Capture One's UI temperature is
    /// the same white balance, so it wins when the file carries both.
    #[test]
    fn white_balance_gains_stand_in_only_without_a_temperature() {
        let gains_only = r#"<SL Engine="800"><E K="WhiteBalance" V="0.9022633;1;1.1216389"/></SL>"#;
        let p = preset_from_sl(gains_only, PresetKind::Preset, "", "x").expect("parses");
        assert!(p.unmapped.is_empty(), "{:?}", p.unmapped);
        // The conversion anchors on the engine's 6500 K reference.
        let expected = 6500.0 * 1.1216389 / 0.9022633;
        assert!((p.adjustments.white_balance.kelvin - expected).abs() < 5.0,
            "{}", p.adjustments.white_balance.kelvin);
        assert!(p.adjustments.white_balance.tint.abs() < 5.0);

        let both = r#"<SL Engine="800">
  <E K="WhiteBalance" V="0.9022633;1;1.1216389" />
  <E K="WhiteBalanceTemperature" V="5000.0009765625" />
  <E K="WhiteBalanceTint" V="0" />
</SL>"#;
        let q = preset_from_sl(both, PresetKind::Preset, "", "x").expect("parses");
        assert!((q.adjustments.white_balance.kelvin - 5000.001).abs() < 0.01);
    }

    #[test]
    fn levels_survive_a_save_and_reload() {
        let text = r#"<SL Engine="800">
  <E K="Highlight" V="0.507402;0.792946;0.858029;1.000000" />
  <E K="Midtone" V="0.000000;0.100000;-0.200000;0.000000" />
  <E K="Shadow" V="0.171981;0.198540;0.238351;0.000000" />
  <E K="TargetHighlight" V="1.000000;1.000000;1.000000;0.940427" />
  <E K="TargetShadow" V="0.000000;0.050000;0.000000;0.000000" />
  <E K="Name" V="Levels" />
</SL>"#;
        let p = preset_from_sl(text, PresetKind::Style, "", "Levels").expect("parses");
        let dir = TempDir::new("levels-rt");
        let path = dir.file("Levels.costyle");
        save_style(&path, &p).expect("saves");
        let written = std::fs::read_to_string(&path).expect("readable");
        assert!(written.contains(r#"<E K="Shadow" V="0.171981;0.19854;0.238351;0" />"#), "{written}");
        assert!(written.contains(r#"<E K="TargetHighlight" V="1;1;1;0.940427" />"#), "{written}");
        let back = load(&path).expect("reads back");
        assert_eq!(back.adjustments.levels, p.adjustments.levels);
    }

    #[test]
    fn color_corrections_survive_a_save_and_reload() {
        let text = r#"<SL Engine="600"><E K="ColorCorrections" V="1,1,0,0.000000,-100.000000,0.000000,58.000000,82.000000,126.000000,-20.000000,20.000000,-53.968250,46.031750,20.000000,0.000000,1,0.0,0.0"/></SL>"#;
        let p = preset_from_sl(text, PresetKind::Preset, "", "Blue").expect("parses");
        let dir = TempDir::new("cc-rt");
        let path = dir.file("Blue.copreset");
        save_preset(&path, &p).expect("saves");
        let written = std::fs::read_to_string(&path).expect("readable");
        let row = written
            .lines()
            .find(|l| l.contains(r#"K="ColorCorrections""#))
            .expect("the key is written");
        assert_eq!(row.matches(',').count(), 17, "{row}");
        let back = load(&path).expect("reads back");
        assert_eq!(back.adjustments.color_editor, p.adjustments.color_editor);
    }



    #[test]
    fn unmapped_keys_are_written_back_verbatim() {
        let text = r#"<SL Engine="600">
  <E K="ColorBalanceHighlight" V="1.000000;1.000000;1.000000"/>
  <E K="ColorBalance" V="1,1,0,0.000000,-100.000000"/>
  <E K="Exposure" V="-1.000"/>
  <E K="Name" V="Round trip"/>
  <E K="UUID" V="{82466223-FC65-4BC5-8965-0CAA0E5B6813}"/>
</SL>"#;
        let p = preset_from_sl(text, PresetKind::Style, "", "Round trip").expect("parses");
        let dir = TempDir::new("unmapped");
        let path = dir.file("Round trip.costyle");
        save_style(&path, &p).expect("saves");
        let written = std::fs::read_to_string(&path).expect("readable");
        assert!(written.contains(r#"<E K="ColorBalanceHighlight" V="1.000000;1.000000;1.000000" />"#), "{written}");
        assert!(written.contains(r#"<E K="ColorBalance" V="1,1,0,0.000000,-100.000000" />"#), "{written}");
        let back = load(&path).expect("reads back");
        assert_eq!(sorted(&back.unmapped), sorted(&p.unmapped));
        assert!(back.adjustments.exposure.exposure == -1.0);
    }

    #[test]
    fn xml_entities_are_read_plain_and_written_escaped() {
        let text = r#"<SL Engine="800">
  <E K="Name" V="B&amp;W Generic 4"/>
  <E K="Exposure" V="-1.000"/>
</SL>"#;
        let p = preset_from_sl(text, PresetKind::Style, "", "B&W Generic 4").expect("parses");
        assert_eq!(p.name, "B&W Generic 4");
        let dir = TempDir::new("escape");
        let path = dir.file("B&W Generic 4.costyle");
        save_style(&path, &p).expect("saves");
        let written = std::fs::read_to_string(&path).expect("readable");
        assert!(written.contains(r#"<E K="Name" V="B&amp;W Generic 4" />"#), "{written}");
        assert!(!written.contains("&amp;amp;"), "{written}");
        let back = load(&path).expect("reads back");
        assert_eq!(back.name, "B&W Generic 4");
    }


    #[test]
    #[ignore = "needs this machine's Capture One install"]
    fn every_installed_preset_survives_a_round_trip() {
        let dir = TempDir::new("install");
        let mut checked = 0;
        for sub in ["Presets60", "CulturalHeritage/Styles"] {
            let mut files = Vec::new();
            collect_presets(&c1_dir().join(sub), &mut files);
            for file in files {
                // Six `Camera Focus` files are Apple plists, not presets.
                let Ok(p) = load(&file) else { continue };
                let ext = file.extension().unwrap();
                let out = dir.file(&format!("{checked}.{}", ext.to_str().unwrap()));
                if ext == "costyle" {
                    save_style(&out, &p).expect("saves style");
                } else {
                    save_preset(&out, &p).expect("saves preset");
                }
                let back = load(&out).expect("reads back");
                assert!(
                    approx_eq(&p.adjustments, &back.adjustments),
                    "{}: {:#?}",
                    file.display(),
                    back.adjustments
                );
                assert_eq!(sorted(&back.unmapped), sorted(&p.unmapped), "{}", file.display());
                assert_eq!(keys(&back.orig), keys(&p.orig), "{}", file.display());
                checked += 1;
            }
        }
        assert!(checked >= 100, "only {checked} installed presets were checked");
    }

    #[test]
    fn p1x_container_lists_its_assets() {
        let mut file = Vec::new();
        let asset = b"hello";
        let offset = 0x1Eu32 + 12;
        file.extend_from_slice(&0u32.to_be_bytes());
        file.extend_from_slice(P1X_MAGIC);
        file.extend_from_slice(&0x0001_0000u32.to_be_bytes());
        file.extend_from_slice(&[0u8; 0x1E - 0x15]);
        file.extend_from_slice(&[0xF0, 0x00]);
        file.extend_from_slice(&(asset.len() as u32).to_be_bytes());
        file.extend_from_slice(&offset.to_be_bytes());
        file.extend_from_slice(&0x0001u16.to_be_bytes());
        file.extend_from_slice(asset);
        let len = file.len() as u32;
        file[..4].copy_from_slice(&len.to_be_bytes());
        let c = parse_p1x(&file).expect("synthetic container parses");
        assert_eq!(c.version, 0x0001_0000);
        assert_eq!(c.assets[0].bytes(&file), b"hello");
    }

    #[test]
    #[ignore = "needs this machine's Capture One install"]
    fn real_p1x_parses() {
        let dir = c1_dir().join("Film Curves");
        let name = p1x_names(&dir).into_iter().next().expect("P1X files present");
        let bytes = std::fs::read(dir.join(format!("{name}.P1X"))).expect("readable");
        let c = parse_p1x(&bytes).expect("parses");
        assert!(c.assets.iter().any(|a| a.tag == 0xF1), "colour-transform payload present");
    }

    #[cfg(feature = "external-fixtures")]
    include!("../local-fixtures/presets-tests.rs");
}
