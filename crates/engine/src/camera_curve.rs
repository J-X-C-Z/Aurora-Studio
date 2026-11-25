//! Resolve an EXIF camera model to Capture One's default film curve.
use crate::filmcurve::{curve_dir, curve_lut, names};
use crate::CameraDefaults;
use std::path::PathBuf;
use std::sync::OnceLock;

/// EXIF model -> `<body>-Auto` stem, matched with every space removed so that
/// `NIKON Z6_3` and `NIKON Z 6_3` both resolve.
const MODELS: &[(&str, &str)] = &[
    ("NIKONZ5", "NikonZ5"),
    ("NIKONZ52", "NikonZ5II"),
    ("NIKONZ6", "NikonZ6"),
    ("NIKONZ62", "NikonZ6II"),
    ("NIKONZ63", "NikonZ6III"),
    ("NIKONZ7", "NikonZ7"),
    ("NIKONZ72", "NikonZ7II"),
    ("NIKONZ8", "NikonZ8"),
    ("NIKONZ9", "NikonZ9"),
    ("NIKONZ30", "NikonZ30"),
    ("NIKONZ50", "NikonZ50"),
    ("NIKONZ502", "NikonZ50II"),
    ("NIKONZF", "NikonZF"),
    ("NIKONZFC", "NikonZFC"),
    ("NIKONZR", "NikonZR"),
];
/// Normalize EXIF model text: uppercase, collapse spaces, C1 writes mark II as `_2`.
fn normalise(model: &str) -> String {
    model
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_uppercase())
        .collect::<String>()
        .replace("MARK", "MK")
}
/// File stem of the camera's default curve, or `None` when unknown.
pub fn default_curve(model: &str) -> Option<String> {
    if let Some(curve) = configured_curve(model).filter(|curve| available(curve)) {
        return Some(curve);
    }
    if let Some(curve) = match_installed_auto(model) {
        return Some(curve);
    }
    let want = normalise(model);
    let body = MODELS.iter().find(|(m, _)| *m == want).map(|(_, stem)| *stem)?;
    let stem = format!("{body}-Auto");
    if available(&stem) {
        Some(stem)
    } else {
        None
    }
}

fn available(stem: &str) -> bool {
    curve_lut(stem).is_some() || curve_dir().join(format!("{stem}.fcrv")).exists()
}

/// C1's own `defaultFilmCurve` binding from `coreConfig.cdx`.
fn configured_curve(model: &str) -> Option<String> {
    static DEFAULTS: OnceLock<Option<CameraDefaults>> = OnceLock::new();
    DEFAULTS
        .get_or_init(|| CameraDefaults::load(&core_config_path()).ok())
        .as_ref()
        .and_then(|defaults| defaults.default_film_curve(model))
}

fn core_config_path() -> PathBuf {
    crate::bundled_resources::core_config_path()
}

/// Fallback for bodies whose EXIF name is not the `cameraID` spelling.
///
/// A match must be unique. The common Sony `ILCE-*`/`ILCA-*` EXIF names are
/// rewritten to C1's `SonyA*` ids; other vendors match after punctuation,
/// brand and Mark/numeral normalisation.
fn match_installed_auto(model: &str) -> Option<String> {
    let candidates = alias_forms(model);
    let mut hit = None;
    for name in names().into_iter().filter(|name| name.ends_with("-Auto")) {
        let body = name.strip_suffix("-Auto").unwrap_or(&name);
        let body = normalise(body);
        if !candidates.iter().any(|candidate| {
            candidate == &body
                || (candidate.len() >= 4 && body.ends_with(candidate))
                || (body.len() >= 4 && candidate.ends_with(&body))
        }) {
            continue;
        }
        if hit.is_some() {
            return None;
        }
        hit = Some(name);
    }
    hit
}

fn alias_forms(model: &str) -> Vec<String> {
    let packed = normalise(model);
    let mut forms = crate::camera_profile::numeral_candidates(&packed);
    let upper = model.to_ascii_uppercase();
    let upper = upper.strip_prefix("SONY ").unwrap_or(&upper);
    for (prefix, body) in [("ILCE-", "SONY A"), ("ILCA-", "SONY A"), ("SLT-", "SONY A"), ("DSLR-", "SONY A")] {
        if let Some(rest) = upper.strip_prefix(prefix) {
            let rest = normalise(rest);
            forms.push(normalise(&format!("{body}{rest}")));
            forms.push(normalise(&format!("SONY{prefix}{rest}")));
        }
    }
    for (prefix, replacement) in [
        ("FUJIFILM", "FUJI"),
        ("PANASONICDC", "PANASONIC"),
        ("OMDIGITALSOLUTIONS", "OLYMPUS"),
        ("NIKONCORPORATION", "NIKON"),
    ] {
        if let Some(rest) = packed.strip_prefix(prefix) {
            forms.push(format!("{replacement}{rest}"));
        }
    }
    forms.sort();
    forms.dedup();
    forms
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_film_curves;
    use std::path::PathBuf;

    fn test_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata")
    }

    #[test]
    fn both_exif_spellings_reach_the_same_curve() {
        let _serial = crate::filmcurve::test_lock();
        crate::filmcurve::use_dir(test_dir());
        let spaced = default_curve("NIKON Z 6_3");
        let packed = default_curve("NIKON Z6_3");
        assert_eq!(spaced, packed, "{spaced:?} vs {packed:?}");
        assert!(packed.is_some(), "Z6III must resolve when the curve ships");
    }
    #[test]
    fn parses_real_z6_3_curve() {
        let bytes = std::fs::read(test_dir().join("NikonZ6III-Auto.fcrv")).unwrap();
        let curves = parse_film_curves(&bytes).unwrap();
        assert_eq!(curves.len(), 3);
        assert_eq!(curves[0].points.len(), 12);
        assert_eq!(curves[0].name, "Linear response");
        assert!((curves[0].points[0].0 - 0.0).abs() < 1e-6);
        assert!((curves[0].points.last().unwrap().0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn real_z6_3_lut_is_monotonic() {
        let _serial = crate::filmcurve::test_lock();
        crate::filmcurve::use_dir(test_dir());
        let lut = curve_lut("NikonZ6III-Auto").unwrap();
        for pair in lut.windows(2) {
            assert!(pair[1] >= pair[0], "{} > {}", pair[0], pair[1]);
        }
        assert_eq!(lut[0], 0.0);
        assert_eq!(lut[255], 1.0);
    }

    #[test]
    fn maps_z_series_models() {
        let _serial = crate::filmcurve::test_lock();
        crate::filmcurve::use_dir(test_dir());
        assert_eq!(default_curve("NIKONZ6_3").as_deref(), Some("NikonZ6III-Auto"));
        assert_eq!(default_curve("nikon z 7_2").as_deref(), Some("NikonZ7II-Auto"));
        assert_eq!(default_curve("NIKONZ9").as_deref(), Some("NikonZ9-Auto"));
    }

    #[test]
    fn unknown_model_returns_none() {
        let _serial = crate::filmcurve::test_lock();
        crate::filmcurve::use_dir(test_dir());
        assert_eq!(default_curve("ACME CAMERA 9000"), None);
        assert_eq!(default_curve(""), None);
    }

    #[test]
    fn vendor_aliases_reach_their_auto_curves() {
        let dir = PathBuf::from(r"C:\Program Files\Capture One\Capture One\Film Curves");
        if !dir.is_dir() {
            return;
        }
        let _serial = crate::filmcurve::test_lock();
        crate::filmcurve::use_dir(&dir);
        for (model, expected) in [
            ("NIKON Z 6_3", "NikonZ6III-Auto"),
            ("Canon EOS R5", "CanonEOSR5-Auto"),
            ("Canon EOS R5 Mark II", "CanonEOSR5Mk2-Auto"),
            ("Sony ILCE-7M4", "SonyA7M4-Auto"),
            ("Fujifilm X-T4", "FujiXT4-Auto"),
            ("Panasonic DC-GH5", "PanasonicGH5-Auto"),
        ] {
            if !dir.join(format!("{expected}.fcrv")).is_file() {
                continue;
            }
            assert_eq!(default_curve(model).as_deref(), Some(expected), "{model}");
        }
    }

    #[test]
    fn every_configured_auto_curve_routes_through_the_same_resolver() {
        let dir = PathBuf::from(r"C:\Program Files\Capture One\Capture One\Film Curves");
        let Ok(defaults) = CameraDefaults::load(&core_config_path()) else { return };
        if !dir.is_dir() {
            return;
        }
        let _serial = crate::filmcurve::test_lock();
        crate::filmcurve::use_dir(&dir);
        let mut missing = Vec::new();
        for id in defaults.camera_ids().into_iter().filter(|id| *id != "BASEDEFAULT") {
            let Some(expected) = defaults.default_film_curve(id) else { continue };
            if !dir.join(format!("{expected}.fcrv")).is_file() {
                missing.push(format!("{id}: {expected}"));
                continue;
            }
            assert_eq!(default_curve(id).as_deref(), Some(expected.as_str()), "{id}");
        }
        missing.sort();
        assert_eq!(
            missing,
            [
                "MamiyaM39: MamiyaM39-Auto",
                "MamiyaM40: MamiyaM40-Auto",
                "MamiyaM40_Sensor+: MamiyaM40-Auto",
                "MamiyaM60: MamiyaM60-Auto",
                "MamiyaM60_Sensor+: MamiyaM60-Auto",
                "PhaseOneIQ3100MP_00: PhaseOneIQ3100MP_00-Auto",
            ]
        );
    }
}
