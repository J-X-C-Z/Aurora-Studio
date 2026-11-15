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

use std::path::Path;

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
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes);
    if !text.trim().is_empty() && !(text.contains("<SL") && text.contains("</SL>")) {
        bail!("{} is not a Capture One style: no <SL> root", path.display());
    }
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or_default();

    let mut settings = C1VariantSettings::defaults();
    let mut name = None;
    let (mut kelvin, mut tint) = (None, None);
    let original = crate::presets::parse_sl(&text);
    for (key, value) in &original {
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

    let adjustments = defined_groups(&settings);
    Ok(C1Style {
        name: name.unwrap_or_else(|| stem.to_owned()),
        engine: engine_attr(&text).unwrap_or(800),
        original,
        color_corrections: settings.color_corrections,
        adjustments,
    })
}

fn engine_attr(text: &str) -> Option<u32> {
    let start = text.find("Engine=\"")? + "Engine=\"".len();
    let end = text[start..].find('"')?;
    text[start..start + end].parse().ok()
}

/// Keys this writer can emit. `C1Style.original` is filtered to this set so
/// metadata such as UUID/StyleSource/TechnicalProcessingMethod is not written.
fn writable_key(key: &str) -> bool {
    matches!(
        key,
        "Name"
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
    text
}

/// Writes a loaded style to `path`.
pub fn save_costyle(path: &Path, style: &C1Style) -> Result<()> {
    std::fs::write(path, write_costyle_text(style))
        .with_context(|| format!("writing {}", path.display()))
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
        "GradationCurve" => s.gradation_curves[0] = c1_model::parse_curve(value),
        "GradationCurveRed" => s.gradation_curves[1] = c1_model::parse_curve(value),
        "GradationCurveGreen" => s.gradation_curves[2] = c1_model::parse_curve(value),
        "GradationCurveBlue" => s.gradation_curves[3] = c1_model::parse_curve(value),
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

    fn testdata(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/c1-styles").join(name)
    }

    /// A scratch file that removes itself when the test ends.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str, body: &[u8]) -> Self {
            let path = std::env::temp_dir()
                .join(format!("portrait-c1-style-{}-{name}", std::process::id()));
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
  <E K="GCurve" V="0.514,0.583;0.210,0.260"/>
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
        // TechnicalProcessingMethod is metadata outside the 30 writable keys.
        let mut expected = original.adjustments.clone();
        expected.base.processing_method = crate::TechnicalProcessingMethod::Photography;
        assert_eq!(read.adjustments, expected);
        assert_eq!(read.color_corrections, original.color_corrections);
    }

    #[test]
    fn writer_omits_metadata_and_unmapped_keys() {
        let style = load_style(&testdata("B&W Generic 4.costyle")).expect("source style");
        let text = write_costyle_text(&style);
        for key in ["UUID", "StyleSource", "TechnicalProcessingMethod", "Flip", "Color_tag_index"] {
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

        let numbers = |value: &str| -> Option<Vec<f32>> {
            value
                .split([';', ','])
                .map(|part| part.trim().parse::<f32>().ok())
                .collect()
        };
        let mut dropped = Vec::new();
        let mut changed = Vec::new();
        for (key, old) in &before {
            match after.iter().find(|(k, _)| k == key) {
                None => dropped.push(key.clone()),
                Some((_, new)) => {
                    if old == new {
                        continue;
                    }
                    let same_numbers = matches!(
                        (numbers(old), numbers(new)),
                        (Some(a), Some(b))
                            if a.len() == b.len()
                                && a.iter().zip(&b).all(|(x, y)| (x - y).abs() <= 1e-4 * x.abs().max(1.0))
                    );
                    assert!(same_numbers, "{key}: {old:?} != {new:?}");
                    changed.push((key.clone(), old.clone(), new.clone()));
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
                "TechnicalProcessingMethod",
                "UUID",
            ]
        );
        println!("kept {} keys, dropped {} keys", after.len(), dropped.len());
        for (key, old, new) in &changed {
            println!("format-only {key}: {old} -> {new}");
        }
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
}
