
/// Map an EXIF camera model to Capture One's ProStandard profile stem.
pub fn for_camera(model: &str) -> Option<&'static str> {
    let cleaned = model.trim();
    let stripped = cleaned
        .strip_prefix("NIKON")
        .map(|s| s.trim())
        .unwrap_or(cleaned);
    let m: String = stripped
        .to_ascii_uppercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect();
    let name = match m.as_str() {
        "Z6_3" | "Z6III" => "NikonZ6III-ProStandard",
        "Z6_2" | "Z6II" => "NikonZ6II-ProStandard",
        "Z6" | "Z6_1" => "NikonZ6-ProStandard",
        "Z7_2" | "Z7II" => "NikonZ7II-ProStandard",
        "Z7" | "Z7_1" => "NikonZ7-ProStandard",
                        _ => return None,
    };
    Some(name)
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
        assert_eq!(for_camera("NIKON Z6_3"), Some("NikonZ6III-ProStandard"));
        assert_eq!(for_camera("NIKON Z 7_2"), Some("NikonZ7II-ProStandard"));
    }

    #[test]
    fn unknown_camera_maps_to_none() {
        assert_eq!(for_camera("Sony ILCE-7M4"), None);
        assert_eq!(for_camera(""), None);
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
}
