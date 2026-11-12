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

use crate::c1_model::{self, C1VariantSettings};
use crate::AdjustmentSet;

/// One `.costyle` / `.copreset` mapped onto the engine's adjustments.
#[derive(Clone, Debug)]
pub struct C1Style {
    /// The file's own `Name`, or the file stem when it carries none.
    pub name: String,
    pub adjustments: AdjustmentSet,
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
    for (key, value) in crate::presets::parse_sl(&text) {
        match key.as_str() {
            "Name" => name = Some(value),
            "WhiteBalanceTemperature" => kelvin = c1_model::parse_scalar(&value),
            "WhiteBalanceTint" => tint = c1_model::parse_scalar(&value),
            _ => apply(&mut settings, &key, &value),
        }
    }
    // Temperature and tint are the numbers Capture One's UI shows; the
    // `WhiteBalance` gains beside them are its internal form, so they only stand
    // in when the file carries no temperature.
    if let Some(k) = kelvin {
        settings.white_balance_gains = c1_model::kelvin_tint_to_gains(k, tint.unwrap_or(0.0));
    }

    Ok(C1Style {
        name: name.unwrap_or_else(|| stem.to_owned()),
        adjustments: defined_groups(&settings),
    })
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


    /// A scratch file that removes itself when the test ends.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(name: &str, body: &[u8]) -> Self {
            let path = std::env::temp_dir()
                .join(format!("lumen-c1-style-{}-{name}", std::process::id()));
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

    #[cfg(feature = "external-fixtures")]
    include!("../local-fixtures/c1_styles-tests.rs");
}
