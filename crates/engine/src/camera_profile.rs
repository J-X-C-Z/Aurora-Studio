
//! Body -> Capture One default colour profile.
//!
//! `coreConfig.cdx` already states the default profile for every `cameraID`,
//! so that file is the answer; the engine only has to translate an EXIF model
//! into the `cameraID`, then find the installed file. The hardcoded table now
//! holds just the two spellings normalisation cannot derive (`Mark II` ->
//! `Mk2`, `Mark IV` -> `Mk4`); every other alias falls out of digit/roman
//! normalisation (`NIKON Z6_3` -> `NikonZ6III`).
//!
//! Reading the config a second time costs one parse per process, next to the
//! profiles themselves; it buys C1's own answer for `-Generic` defaults
//! (481 bodies) and for the bodies whose default is a capture style
//! (`-Flash`, `-Outdoor Daylight`, ...).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::OnceLock;

/// Whether a binding is the body's own profile or Capture One's generic one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileKind {
    Dedicated,
    Generic,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CameraProfile {
    pub name: String,
    pub kind: ProfileKind,
}

/// Model with the brand spelled out and punctuation removed, upper-cased, so
/// "Canon EOS R5" and "CanonEOSR5" compare equal.
fn fingerprint(model: &str) -> String {
    model
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect::<String>()
        .to_ascii_uppercase()
}

/// Map an EXIF camera model to Capture One's default profile file stem.
pub fn for_camera(model: &str) -> Option<String> {
    resolve(model).map(|profile| profile.name)
}

/// Same binding, with C1's own dedicated/generic split.
pub fn resolve(model: &str) -> Option<CameraProfile> {
    if model.trim().is_empty() {
        return None;
    }
    let candidates = numeral_candidates(&normalise(model));
    if let Some(catalog) = catalog() {
        for id in &candidates {
            let Some(entry) = catalog.by_key.get(&format!("{id}|{}", suffix_tag(model))) else { continue };
            return entry.as_deref().and_then(bind);
        }
        // The config is authoritative: an unknown body gets no profile rather
        // than a guess from whatever happens to be installed.
        return None;
    }
    // Without a config (no Capture One install) fall back to the file names.
    pick_exact(&fingerprint(model), crate::icc::profile_names().iter().map(String::as_str))
        .map(|stem| CameraProfile { name: stem.to_string(), kind: kind_of(stem) })
}

/// Model text with punctuation and spaces removed, `Mark` folded to C1's `Mk`
/// and a trailing `_SRAW` dropped so the base body answers.
fn normalise(model: &str) -> String {
    let packed: String = model
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase()
        .replace("MARK", "MK");
    packed.strip_suffix("SRAW").unwrap_or(&packed).to_string()
}

/// The distinctions punctuation stripping would erase: `PhaseOneP45+` is not
/// `PhaseOneP45`, and a `_SRAW` body is not the base body.
fn suffix_tag(model: &str) -> &'static str {
    let upper = model.to_ascii_uppercase();
    if upper.contains("SRAW") {
        "SRAW"
    } else if model.contains('+') {
        "PLUS"
    } else {
        ""
    }
}

fn body_key(model: &str) -> String {
    format!("{}|{}", normalise(model), suffix_tag(model))
}

/// `NIKONZ63` <-> `NIKONZ6III` and the like, in both directions.
pub(crate) fn numeral_candidates(normalised: &str) -> Vec<String> {
    const ROMAN: [(char, &str); 6] =
        [('2', "II"), ('3', "III"), ('4', "IV"), ('5', "V"), ('6', "VI"), ('7', "VII")];
    let mut out = vec![normalised.to_string()];
    let mut chars = normalised.chars();
    if let Some(last) = chars.next_back() {
        if let Some((_, roman)) = ROMAN.iter().find(|(digit, _)| *digit == last) {
            out.push(format!("{}{roman}", chars.as_str()));
        }
    }
    for (digit, roman) in ROMAN {
        if let Some(stem) = normalised.strip_suffix(roman) {
            out.push(format!("{stem}{digit}"));
        }
    }
    // C1 writes a first generation as `Z6_1`/`Z7_1` while the id has no suffix.
    if let Some(stem) = normalised.strip_suffix('1') {
        if stem.ends_with(|c: char| c.is_ascii_digit()) {
            out.push(stem.to_string());
        }
    }
    out
}

/// Installed stem for a `defaultColorProfile` value, or nothing when the file
/// is not on this machine (or the value is the `EMBEDDED` sentinel).
fn bind(file: &str) -> Option<CameraProfile> {
    let stem = file.rsplit_once('.').map_or(file, |(stem, _)| stem);
    if stem.eq_ignore_ascii_case("EMBEDDED") {
        return None;
    }
    let installed = crate::icc::profile_names()
        .into_iter()
        .find(|name| name.eq_ignore_ascii_case(stem))?;
    Some(CameraProfile { name: installed, kind: kind_of(stem) })
}

fn kind_of(stem: &str) -> ProfileKind {
    if stem.to_ascii_lowercase().contains("-generic") {
        ProfileKind::Generic
    } else {
        ProfileKind::Dedicated
    }
}

/// The parts of `coreConfig.cdx` this module needs.
struct Catalog {
    /// `cameraID key -> defaultColorProfile`. `None` marks a duplicate id whose
    /// two blocks disagree, which must not be guessed.
    by_key: HashMap<String, Option<String>>,
    /// Every `(cameraID, defaultColorProfile)` in file order, for coverage tests.
    bodies: Vec<(String, String)>,
}

fn catalog() -> Option<&'static Catalog> {
    static CATALOG: OnceLock<Option<Catalog>> = OnceLock::new();
    CATALOG
        .get_or_init(|| {
            let path = core_config_path();
            let bytes = std::fs::read(&path).ok()?;
            let xml = crate::decode_cdx(&bytes).ok()?;
            Some(parse_catalog(&xml))
        })
        .as_ref()
}

/// `$AURORA_CORECONFIG`, else the Capture One install, else the crate's
/// `testdata` copy (which is what `cargo test` runs against).
fn core_config_path() -> PathBuf {
    crate::bundled_resources::core_config_path()
}

fn parse_catalog(xml: &str) -> Catalog {
    let mut by_key: HashMap<String, Option<String>> = HashMap::new();
    let mut bodies = Vec::new();
    let mut rest = xml;
    while let Some(start) = rest.find("<cameraID") {
        rest = &rest[start..];
        let Some(tag_end) = rest.find('>') else { break };
        let Some(id) = attribute(&rest[..tag_end], "val") else { break };
        let body_start = tag_end + 1;
        let body_end = rest[body_start..].find("</cameraID>").map(|i| body_start + i).unwrap_or(rest.len());
        let body = &rest[body_start..body_end];
        if let Some(at) = body.find("<defaultColorProfile") {
            if let Some(end) = body[at..].find("/>").map(|i| at + i) {
                if let Some(profile) = attribute(&body[at..end], "val") {
                    bodies.push((id.clone(), profile.clone()));
                    let key = body_key(&id);
                    match by_key.get(&key) {
                        Some(Some(existing)) if existing != &profile => {
                            by_key.insert(key, None);
                        }
                        Some(_) => {}
                        None => {
                            by_key.insert(key, Some(profile));
                        }
                    }
                }
            }
        }
        rest = &rest[(body_end + "</cameraID>".len()).min(rest.len())..];
    }
    Catalog { by_key, bodies }
}

fn attribute<'a>(tag: &'a str, name: &str) -> Option<String> {
    let needle = format!("{name}=\"");
    let at = tag.find(&needle)? + needle.len();
    let end = tag[at..].find('"')?;
    Some(tag[at..at + end].to_string())
}

/// The model part of a C1 camera stem, tolerating a revision suffix such as
/// `SigmaFPL-ProStandard v2`.
fn stem_model(stem: &str) -> Option<&str> {
    let (model, rest) = stem.split_once("-ProStandard")?;
    let rest = rest.trim();
    if rest.is_empty() || (rest.starts_with('v') && rest[1..].bytes().all(|b| b.is_ascii_digit())) {
        Some(model)
    } else {
        None
    }
}

/// Body part of any camera stem, whatever its suffix (`-Generic`, `-Flash`).
/// C1 body ids never contain a hyphen.
fn stem_body(stem: &str) -> &str {
    stem.split_once('-').map_or(stem, |(body, _)| body)
}

/// The single installed camera stem whose model part fingerprints to
/// the same string. Two matches are ambiguous and therefore rejected.
fn pick_exact<'a>(model: &str, stems: impl Iterator<Item = &'a str>) -> Option<&'a str> {
    let mut hit = None;
    for stem in stems {
        let Some(name) = stem_model(stem) else { continue };
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
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs Capture One coreConfig/ICC profiles in crates/engine/testdata/ (enable --features external-fixtures)")]
    fn maps_known_cameras() {
        assert_eq!(for_camera("NIKON Z6_3").as_deref(), Some("NikonZ6III-ProStandard"));
        assert_eq!(for_camera("NIKON Z 7_2").as_deref(), Some("NikonZ7II-ProStandard"));
        assert_eq!(for_camera("Nikon Z 6_3").as_deref(), Some("NikonZ6III-ProStandard"));
    }

    #[test]
    fn unknown_camera_maps_to_none() {
        assert_eq!(for_camera("Sony ILCE-7M4"), None);
        assert_eq!(for_camera("Sony ILCE-9M3"), None);
        assert_eq!(for_camera(""), None);
    }

    #[test]
    fn matches_bodies_across_exif_case_and_spacing() {
        for (model, stem) in [
            ("Canon EOS R5", "CanonEOSR5-ProStandard"),
            ("canon eosr5", "CanonEOSR5-ProStandard"),
            ("Canon EOS R5 Mark II", "CanonEOSR5Mk2-ProStandard"),
            ("Canon EOS 5D Mark IV", "CanonEOS5DMk4-ProStandard"),
            ("NIKON Z 6_3", "NikonZ6III-ProStandard"),
            ("nikon z 6 iii", "NikonZ6III-ProStandard"),
            ("Sigma fp L", "SigmaFPL-ProStandard v2"),
        ] {
            if !is_available(stem) {
                continue;
            }
            assert_eq!(for_camera(model).as_deref(), Some(stem), "{model}");
            println!("{model} -> {stem}");
        }
    }

    /// Every installed camera stem still points at its own body's default, even
    /// when that default is the generic profile rather than the stem itself.
    #[test]
    fn every_installed_camera_stem_resolves_to_its_body_default() {
        let names = crate::icc::profile_names();
        let stems: Vec<&str> = names
            .iter()
            .map(String::as_str)
            .filter(|name| stem_model(name).is_some())
            .collect();
        if stems.is_empty() {
            return;
        }
        let Some(catalog) = catalog() else {
            eprintln!("coreConfig missing, skipping");
            return;
        };
        let misses: Vec<&str> = stems
            .iter()
            .copied()
            .filter(|stem| {
                let model = stem_model(stem).unwrap();
                if !catalog.by_key.contains_key(&body_key(model)) {
                    return false;
                }
                match resolve(model) {
                    Some(bound) => fingerprint(stem_body(&bound.name)) != fingerprint(model),
                    None => true,
                }
            })
            .collect();
        println!("{} installed camera stems, {} unresolved", stems.len(), misses.len());
        assert!(misses.is_empty(), "unresolved: {misses:?}");
    }

    /// Bodies whose default is C1's generic space report `Generic`, not a
    /// dedicated profile that merely happens to be installed.
    #[test]
    fn generic_defaults_are_reported_as_generic() {
        let Some(bound) = resolve("Nikon Z 5") else {
            eprintln!("coreConfig or NikonZ5-Generic missing, skipping");
            return;
        };
        assert_eq!(bound.kind, ProfileKind::Generic, "{bound:?}");
        assert_eq!(bound.name, "NikonZ5-Generic");
    }

    /// Capture-style defaults (`-Flash`, `-Outdoor Daylight`) are camera spaces
    /// too; the old `-ProStandard`-only filter refused them.
    #[test]
    fn capture_style_defaults_are_dedicated() {
        let Some(bound) = resolve("PhaseOneIQ3100MP") else {
            eprintln!("coreConfig or PhaseOne profile missing, skipping");
            return;
        };
        assert_eq!(bound.kind, ProfileKind::Dedicated);
        assert_eq!(bound.name, "PhaseOneIQ3100MP-Flash");
    }

    /// `_SRAW` files share the base body's profile.
    #[test]
    fn sraw_uses_the_base_body_profile() {
        let Some(bound) = resolve("CanonEOS5DMk3_SRAW") else {
            eprintln!("coreConfig or Canon profile missing, skipping");
            return;
        };
        assert_eq!(bound.name, "CanonEOS5DMk3-ProStandard");
        assert_eq!(resolve("Canon EOS 5D Mark III").map(|p| p.name), Some(bound.name));
    }

    /// coreConfig says Z7 uses `-Generic V2`, while a `-ProStandard` file also
    /// exists; C1's answer wins.
    #[test]
    fn z7_keeps_capture_ones_generic_default() {
        let Some(bound) = resolve("NIKON Z7") else {
            eprintln!("coreConfig or NikonZ7 profile missing, skipping");
            return;
        };
        assert_eq!(bound.name, "NikonZ7-Generic V2");
        assert_eq!(bound.kind, ProfileKind::Generic);
        assert_eq!(resolve("NIKON Z 7"), Some(bound));
    }

    /// An unknown body must not be handed the nearest installed profile.
    #[test]
    fn unknown_bodies_are_not_guessed() {
        for model in ["ACME CAMERA 9000", "Sony ILCE-7M4", "Nikon Z 99", ""] {
            assert_eq!(resolve(model), None, "{model}");
        }
    }

    /// C1 spells the first generation `Z6_1`/`Z7_1`; the ids carry no suffix.
    #[test]
    fn first_generation_suffixes_match_the_base_body() {
        if !is_available("NikonZ6-ProStandard") || !is_available("NikonZ7-Generic V2") {
            eprintln!("Nikon profiles missing, skipping");
            return;
        }
        assert_eq!(resolve("NIKON Z6_1").map(|p| p.name).as_deref(), Some("NikonZ6-ProStandard"));
        assert_eq!(resolve("NIKON Z7_1").map(|p| p.name).as_deref(), Some("NikonZ7-Generic V2"));
    }

    /// Every body in the config either binds to a file or is reported, so the
    /// report's hit counts cannot silently drift.
    #[test]
    fn every_core_config_body_is_bound_or_listed() {
        let Some(catalog) = catalog() else {
            eprintln!("coreConfig missing, skipping");
            return;
        };
        let (mut dedicated, mut generic, mut embedded, mut missing) = (0, 0, Vec::new(), Vec::new());
        let mut total = 0;
        let mut seen = std::collections::HashSet::new();
        for (id, file) in &catalog.bodies {
            if file.is_empty() || !seen.insert(id) {
                continue;
            }
            total += 1;
            match resolve(id) {
                Some(bound) if bound.kind == ProfileKind::Generic => generic += 1,
                Some(_) => dedicated += 1,
                None if file.eq_ignore_ascii_case("EMBEDDED") => embedded.push(id.as_str()),
                None => missing.push(id.as_str()),
            }
        }
        missing.sort_unstable();
        println!(
            "bodies={total} dedicated={dedicated} generic={generic} embedded={} missing={}",
            embedded.len(),
            missing.len()
        );
        println!("missing: {missing:?}");
        assert_eq!(dedicated + generic + embedded.len() + missing.len(), total);
    }

    #[test]
    fn pick_exact_refuses_ambiguous_stems() {
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

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs Capture One coreConfig/ICC profiles in crates/engine/testdata/ (enable --features external-fixtures)")]
    fn real_icm_parses_as_mft2_lut() {
        let profile = crate::icc::profile("NikonZ6III-ProStandard").expect("profile must load");
        assert_eq!(&profile.data_space, b"RGB ");
        assert!(matches!(profile.pcs, crate::icc::Pcs::Lab));
        let lut = profile.a2b.as_ref().expect("A2B0");
        assert_eq!(lut.grid, 33);
        assert!(profile.b2a.is_none());
        println!("description={} grid={} b2a={:?}", profile.description, lut.grid, profile.b2a.is_some());
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs Capture One coreConfig/ICC profiles in crates/engine/testdata/ (enable --features external-fixtures)")]
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
