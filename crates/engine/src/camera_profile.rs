
/// Model with the brand spelled out and punctuation removed, upper-cased, so
/// "Canon EOS R5" and "CanonEOSR5" compare equal.
fn fingerprint(model: &str) -> String {
    model
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect::<String>()
        .to_ascii_uppercase()
}

/// Map an EXIF camera model to Capture One's ProStandard profile stem.
///
/// The explicit table covers the Nikon names EXIF and C1 spell differently
/// ("NIKON Z 6_3" is "NikonZ6III"). Everything else falls back to an exact,
/// case-insensitive stem match against the installed `-ProStandard` profiles.
/// No match, or more than one, means no camera profile.
pub fn for_camera(model: &str) -> Option<String> {
    let m = fingerprint(model.strip_prefix("NIKON").or_else(|| model.strip_prefix("Nikon")).unwrap_or(model));
    let table = match m.as_str() {
        "Z6_3" | "Z6III" => Some("NikonZ6III-ProStandard"),
        "Z6_2" | "Z6II" => Some("NikonZ6II-ProStandard"),
        "Z6" | "Z6_1" => Some("NikonZ6-ProStandard"),
        "Z7_2" | "Z7II" => Some("NikonZ7II-ProStandard"),
        "Z7" | "Z7_1" => Some("NikonZ7-ProStandard"),
        _ => None,
    };
    if let Some(name) = table {
        return Some(name.to_string());
    }
    pick_exact(&fingerprint(model), crate::icc::profile_names().iter().map(String::as_str))
        .map(str::to_string)
}

/// The single installed `-ProStandard` stem whose model part fingerprints to
/// the same string. Two matches are ambiguous and therefore rejected.
fn pick_exact<'a>(model: &str, stems: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    let mut hit = None;
    for stem in stems {
        let Some(name) = stem.strip_suffix("-ProStandard") else { continue };
        if fingerprint(name) != model {
            continue;
        }
        if hit.is_some() {
            return None;
        }
        hit = Some(stem);
    }
    hit
}

pub fn is_available(name: &str) -> bool {
    crate::icc::profile(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adjust::AdjustmentSet;

    #[test]
    fn maps_known_cameras() {
        assert_eq!(for_camera("NIKON Z6_3").as_deref(), Some("NikonZ6III-ProStandard"));
        assert_eq!(for_camera("NIKON Z 7_2").as_deref(), Some("NikonZ7II-ProStandard"));
        assert_eq!(for_camera("Nikon Z 6_3").as_deref(), Some("NikonZ6III-ProStandard"));
    }

    #[test]
    fn unknown_camera_maps_to_none() {
        assert_eq!(for_camera("Sony ILCE-7M4"), None);
        assert_eq!(for_camera(""), None);
    }

    #[test]
    fn without_a_table_hit_only_an_exact_stem_matches() {
        let stems = [
            "CanonEOSR5-ProStandard",
            "CanonEOSR5X-ProStandard",
            "NikonZ9-Nikon - Standard",
        ];
        assert_eq!(pick_exact("CANONEOSR5", stems.iter().copied()), Some("CanonEOSR5-ProStandard"));
        assert_eq!(pick_exact("CANONEOSR6", stems.iter().copied()), None);
        let dupes = ["CanonEOSR5-ProStandard", "Canon EOS R5-ProStandard"];
        assert_eq!(pick_exact("CANONEOSR5", dupes.iter().copied()), None, "ambiguous matches are refused");
    }

    #[cfg(feature = "external-fixtures")]
    #[test]
    fn real_icm_parses_as_mft2_lut() {
        let profile = crate::icc::profile("NikonZ6III-ProStandard").expect("profile must load");
        assert_eq!(&profile.data_space, b"RGB ");
        assert!(matches!(profile.pcs, crate::icc::Pcs::Lab));
        let lut = profile.a2b.as_ref().expect("A2B0");
        assert_eq!(lut.grid, 33);
        assert!(profile.b2a.is_none());
        println!("description={} grid={} b2a={:?}", profile.description, lut.grid, profile.b2a.is_some());
    }

    #[cfg(feature = "external-fixtures")]
    #[test]
    fn lut_conversion_is_monotone_in_l() {
        let profile = crate::icc::profile("NikonZ6III-ProStandard").expect("profile must load");
        let mut prev_l = f32::MIN;
        for step in 0..=100 {
            let v = step as f32 / 100.0;
            let lin = profile.to_linear_srgb([v, v, v]);
            let l = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
            assert!(l >= prev_l - 1e-6, "luminance regressed at {}", v);
            prev_l = l;
        }
    }

    #[test]
    fn missing_profile_leaves_default_untouched() {
        let mut set = AdjustmentSet::default();
        assert!(set.base.icc_profile.is_none());
        set.base.icc_profile = Some("Explicit".into());
        let mapped = for_camera("unknown");
        assert_eq!(mapped, None);
        assert_eq!(set.base.icc_profile.as_deref(), Some("Explicit"));
    }

    #[test]
    fn installed_prostandard_profiles_match_their_models() {
        for (model, stem) in [
            ("NIKON Z 6_3", "NikonZ6III-ProStandard"),
            ("Canon EOS R5", "CanonEOSR5-ProStandard"),
            ("NIKON Z 9", "NikonZ9-ProStandard"),
        ] {
            if crate::icc::profile(stem).is_none() {
                continue;
            }
            assert_eq!(for_camera(model).as_deref(), Some(stem), "{model} -> {stem}");
        }
        if crate::icc::profile("SonyA7M4-ProStandard").is_some() {
            assert_eq!(
                for_camera("Sony ILCE-7M4"),
                None,
                "a differently spelled model must fall back to the default, not guess"
            );
        }
    }
}
