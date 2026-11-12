//! Resolve an EXIF camera model to Capture One's default film curve.
use crate::filmcurve::{curve_dir, curve_lut};
/// EXIF model -> `<body>-Auto` stem, matched with every space removed so that
/// `NIKON Z6_3` and `NIKON Z 6_3` both resolve.
const MODELS: &[(&str, &str)] = &[
    ("NIKONZ5", "NikonZ5"),
    ("NIKONZ5_2", "NikonZ5II"),
    ("NIKONZ6", "NikonZ6"),
    ("NIKONZ6_2", "NikonZ6II"),
    ("NIKONZ6_3", "NikonZ6III"),
    ("NIKONZ7", "NikonZ7"),
    ("NIKONZ7_2", "NikonZ7II"),
    ("NIKONZ8", "NikonZ8"),
    ("NIKONZ9", "NikonZ9"),
    ("NIKONZ30", "NikonZ30"),
    ("NIKONZ50", "NikonZ50"),
    ("NIKONZ50_2", "NikonZ50II"),
    ("NIKONZF", "NikonZF"),
    ("NIKONZFC", "NikonZFC"),
    ("NIKONZR", "NikonZR"),
];
/// Normalize EXIF model text: uppercase, collapse spaces, C1 writes mark II as `_2`.
fn normalise(model: &str) -> String {
    model
        .split_whitespace()
        .collect::<String>()
        .to_ascii_uppercase()
        .replace("II", "_2")
}
/// File stem of the camera's default curve, or `None` when unknown.
pub fn default_curve(model: &str) -> Option<String> {
    let want = normalise(model);
    let body = MODELS.iter().find(|(m, _)| *m == want).map(|(_, stem)| *stem)?;
    let stem = format!("{body}-Auto");
    if curve_lut(&stem).is_some() || curve_dir().join(format!("{stem}.fcrv")).exists() {
        Some(stem)
    } else {
        None
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_film_curves;
    use std::path::PathBuf;

    fn test_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("aurora-camera-curves-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut bytes = b"MM".to_vec();
        for word in [6u32, 2, 0, 0, u32::MAX, u32::MAX, 1] { bytes.extend_from_slice(&word.to_be_bytes()); }
        for name in ["Synthetic", "Linear response"] {
            bytes.extend_from_slice(&(name.len() as u32).to_be_bytes());
            bytes.extend_from_slice(name.as_bytes());
        }
        for name in ["NikonZ6III-Auto", "NikonZ7II-Auto", "NikonZ9-Auto"] {
            std::fs::write(dir.join(format!("{name}.fcrv")), &bytes).unwrap();
        }
        dir
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
    fn parses_synthetic_camera_curve() {
        let bytes = std::fs::read(test_dir().join("NikonZ6III-Auto.fcrv")).unwrap();
        let curves = parse_film_curves(&bytes).unwrap();
        assert_eq!(curves.len(), 1);
        assert_eq!(curves[0].points.len(), 2);
        assert_eq!(curves[0].name, "Linear response");
        assert!((curves[0].points[0].0 - 0.0).abs() < 1e-6);
        assert!((curves[0].points.last().unwrap().0 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn synthetic_camera_lut_is_monotonic() {
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
        assert_eq!(default_curve("SONY ILCE-7M4"), None);
        assert_eq!(default_curve(""), None);
    }
}
