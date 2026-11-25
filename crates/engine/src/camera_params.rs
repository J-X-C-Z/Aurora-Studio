//! Per-camera Capture One parameters extracted from 16.8.6.
//!
//! The embedded pack is intentionally data-only: model ids, mount, pixel pitch,
//! ISO scaling, white level, compression names, noise tables, CCM, profile and
//! curve names, and Capture One's slider defaults.  A LibRaw decoder supplies
//! the Bayer plane; these values drive the stages after it.
//!
//! Source: `ImgCoreDll.dll` SHA-256
//! `CA02B6AC2786E9A100F63221DF7340E5DF210416EB0BD53A4BB1BFC2306A44A4`
//! plus its `coreConfig.cdx`.  The pack is generated evidence, not a runtime
//! read of the Capture One install; see
//! `06_reports/SESSION-c1-parse-cross-brand-params-20261122.md`.

use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

use serde::Deserialize;

/// Extracted from Capture One, so it is not published with the source: it is
/// embedded from `private-assets/c1/` when present. Without it no body has
/// Capture One defaults and callers fall back to their generic paths.
const PACK_JSON: Option<&str> = aurora_portrait::private_asset_str!("c1/camera_params.json");
/// All 714 non-pseudo bodies, with the fields coreConfig can supply for every
/// model and DLL metadata where it could be extracted statically.
const ALL_PACK_JSON: Option<&str> = aurora_portrait::private_asset_str!("c1/camera_params_all.json");
/// What the packs read as in a build without the private assets.
const EMPTY_PACK_JSON: &str = r#"{"source":{"c1_version":"","dll_sha256":"","snapshot_date":""},"cameras":{}}"#;

#[derive(Clone, Debug, Deserialize)]
pub struct CameraPack {
    pub source: PackSource,
    pub cameras: BTreeMap<String, CameraParams>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PackSource {
    pub c1_version: String,
    pub dll_sha256: String,
    pub snapshot_date: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CameraParams {
    pub brand: String,
    pub image_rep_class: String,
    pub model_id: String,
    pub mount: String,
    pub pixel_pitch_um: f32,
    pub iso_norm: serde_json::Value,
    pub white_level: u32,
    pub cfa: String,
    pub compression: Vec<String>,
    pub noise: NoiseModel,
    pub ccm: [[f32; 3]; 3],
    pub profile: String,
    pub curve: String,
    pub defaults: SliderDefaults,
    pub switches: CameraSwitches,
    pub confidence: BTreeMap<String, String>,
    #[serde(default)]
    pub notes: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct NoiseModel {
    pub source: String,
    #[serde(default)]
    pub table_rva: Option<String>,
    pub entries: Vec<NoiseEntry>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct NoiseEntry {
    pub iso: u32,
    #[serde(default)]
    pub floor: Option<f32>,
    #[serde(default)]
    pub poisson: Option<f32>,
    #[serde(default)]
    pub flags: Option<i32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SliderDefaults {
    pub usm_amount: f32,
    pub usm_radius: f32,
    pub usm_threshold: f32,
    /// `antiColorAliasingBlend`; `None` when coreConfig has no node for the body.
    pub aca: Option<f32>,
    pub nr_luminance: f32,
    pub nr_details: f32,
    pub nr_color: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CameraSwitches {
    pub g1g2_adaptive_gain: bool,
    pub black_clean: f32,
    pub gain: f32,
    pub sensor_range_gain: f32,
}

impl CameraParams {
    /// Model id as the number Capture One compares in `0x198`.
    pub fn model_id_u32(&self) -> Option<u32> {
        u32::from_str_radix(self.model_id.trim_start_matches("0x"), 16).ok()
    }

    /// Highest ISO row at or below `iso`; falls back to the first row.
    pub fn noise_at(&self, iso: u32) -> Option<&NoiseEntry> {
        let entries = &self.noise.entries;
        entries.iter().rev().find(|row| row.iso <= iso).or_else(|| entries.first())
    }

}

/// Normalises an EXIF or Capture One body name to letters and digits, upper-cased.
fn fingerprint(model: &str) -> String {
    model
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .collect::<String>()
        .to_ascii_uppercase()
        .replace("MARK", "MK")
}

/// Names that digit/roman folding cannot derive.  Keys are fingerprints.
const ALIASES: &[(&str, &str)] = &[
    ("NIKONZ63", "NikonZ6III"),
    ("NIKONZ62", "NikonZ6II"),
    ("NIKONZ72", "NikonZ7II"),
    ("NIKONZF", "NikonZF"),
    ("NIKONZFC", "NikonZFC"),
    ("NIKONZ5", "NikonZ5"),
    ("NIKONZ50", "NikonZ50"),
    ("NIKOND850", "NikonD850"),
    ("NIKOND6", "NikonD6"),
    ("SONYILCE7M4", "SonyA7M4"),
    ("SONYILCE7RM5", "SonyA7RM5"),
    ("SONYILCE1", "SonyA1"),
    ("SONYILCE7SM3", "SonyA7SM3"),
    ("SONYILCE9M3", "SonyA9M3"),
    ("SONYILCE6700", "SonyA6700"),
    ("ILCE7M4", "SonyA7M4"),
    ("ILCE7RM5", "SonyA7RM5"),
    ("ILCE1", "SonyA1"),
    ("ILCE7SM3", "SonyA7SM3"),
    ("ILCE9M3", "SonyA9M3"),
    ("ILCE6700", "SonyA6700"),
    ("CANONEOSR5", "CanonEOSR5"),
    ("CANONEOSR5MKII", "CanonEOSR5Mk2"),
    ("CANONEOSR6MKII", "CanonEOSR6Mk2"),
    ("CANONEOSR3", "CanonEOSR3"),
    ("CANONEOSR1", "CanonEOSR1"),
    ("CANONEOS5DMKIV", "CanonEOS5DMk4"),
    ("EOSR5", "CanonEOSR5"),
    ("EOSR5MKII", "CanonEOSR5Mk2"),
    ("EOSR6MKII", "CanonEOSR6Mk2"),
    ("EOSR3", "CanonEOSR3"),
    ("EOSR1", "CanonEOSR1"),
    ("EOS5DMKIV", "CanonEOS5DMk4"),
];

fn pack() -> &'static CameraPack {
    static PACK: OnceLock<CameraPack> = OnceLock::new();
    PACK.get_or_init(|| {
        serde_json::from_str(PACK_JSON.unwrap_or(EMPTY_PACK_JSON)).expect("embedded camera_params.json is valid")
    })
}

fn index() -> &'static HashMap<String, &'static str> {
    static INDEX: OnceLock<HashMap<String, &'static str>> = OnceLock::new();
    INDEX.get_or_init(|| {
        let mut map = HashMap::new();
        for (key, cam) in &pack().cameras {
            map.insert(fingerprint(key), key.as_str());
            map.entry(fingerprint(&cam.image_rep_class)).or_insert(key.as_str());
        }
        for (alias, key) in ALIASES {
            map.insert((*alias).to_string(), *key);
        }
        map
    })
}

/// All cameras in the pack, keyed by Capture One's `cameraID`.
pub fn all() -> &'static BTreeMap<String, CameraParams> {
    &pack().cameras
}

/// Look up a body by EXIF model, Capture One id or image-rep class name.
pub fn for_camera(model: &str) -> Option<&'static CameraParams> {
    let key = index().get(&fingerprint(model))?;
    pack().cameras.get(*key)
}

/// The pack's provenance, for diagnostics and the Info panel.
pub fn source() -> &'static PackSource {
    &pack().source
}

#[derive(Clone, Debug, Deserialize)]
pub struct AllCameraPack {
    pub source: AllSource,
    pub cameras: BTreeMap<String, AllCameraParams>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AllSource {
    pub c1_version: String,
    pub dll_sha256: String,
    #[serde(default)]
    pub snapshot_date: Option<String>,
    #[serde(default)]
    pub scope: String,
}

/// A body from the 714-model pack.  Fields are optional because coreConfig
/// carries profile/curve/ISO data for every body, while model ids, CCM and
/// pixel pitch only exist for the classes the DLL exposes.
#[derive(Clone, Debug, Deserialize)]
pub struct AllCameraParams {
    pub brand: String,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub curve: Option<String>,
    #[serde(default)]
    pub profile_file_exists: Option<bool>,
    #[serde(default)]
    pub curve_file_exists: Option<bool>,
    #[serde(default)]
    pub switches: serde_json::Value,
    #[serde(default)]
    pub defaults: Option<serde_json::Value>,
    #[serde(default)]
    pub iso: Vec<AllIsoRow>,
    #[serde(default)]
    pub dll: Option<AllDllInfo>,
    #[serde(default)]
    pub missing: Vec<String>,
    #[serde(default)]
    pub notes: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AllIsoRow {
    pub iso: u32,
    #[serde(default)]
    pub floor: Option<f32>,
    #[serde(default)]
    pub poisson: Option<f32>,
    #[serde(default)]
    pub usm_amount: Option<f32>,
    #[serde(default)]
    pub usm_radius: Option<f32>,
    #[serde(default)]
    pub usm_threshold: Option<f32>,
    #[serde(default)]
    pub aca: Option<f32>,
    #[serde(default)]
    pub cnr: Option<f32>,
    #[serde(default)]
    pub details: Option<f32>,
    #[serde(default)]
    pub clean_long: Option<f32>,
    #[serde(default)]
    pub fine_grain: Option<f32>,
    #[serde(default)]
    pub kind: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct AllDllInfo {
    #[serde(default)]
    pub class: Option<String>,
    #[serde(default)]
    pub vtable: Option<String>,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub mount: Option<String>,
    #[serde(default)]
    pub pixel_pitch_um: Option<f32>,
    #[serde(default)]
    pub white_level: Option<u32>,
    #[serde(default)]
    pub ccm: Option<[[f32; 3]; 3]>,
    #[serde(default)]
    pub noise: Option<serde_json::Value>,
    #[serde(default)]
    pub compression: Option<Vec<String>>,
    #[serde(default)]
    pub iso_norm: Option<serde_json::Value>,
    #[serde(default)]
    pub cfa: Option<String>,
}

impl AllCameraParams {
    /// Highest ISO row at or below `iso`; falls back to the first row.
    pub fn noise_at(&self, iso: u32) -> Option<&AllIsoRow> {
        self.iso.iter().rev().find(|row| row.iso <= iso).or_else(|| self.iso.first())
    }

    pub fn model_id_u32(&self) -> Option<u32> {
        self.dll
            .as_ref()?
            .model_id
            .as_deref()
            .and_then(|id| u32::from_str_radix(id.trim_start_matches("0x"), 16).ok())
    }
}

fn all_pack() -> &'static AllCameraPack {
    static PACK: OnceLock<AllCameraPack> = OnceLock::new();
    PACK.get_or_init(|| {
        serde_json::from_str(ALL_PACK_JSON.unwrap_or(EMPTY_PACK_JSON)).expect("embedded camera_params_all.json is valid")
    })
}

fn all_index() -> &'static HashMap<String, &'static str> {
    static INDEX: OnceLock<HashMap<String, &'static str>> = OnceLock::new();
    INDEX.get_or_init(|| {
        let mut map = HashMap::new();
        for (key, cam) in &all_pack().cameras {
            map.insert(fingerprint(key), key.as_str());
            if let Some(class) = cam.dll.as_ref().and_then(|dll| dll.class.as_deref()) {
                map.entry(fingerprint(class)).or_insert(key.as_str());
            }
        }
        for (alias, key) in ALIASES {
            map.insert((*alias).to_string(), *key);
        }
        map
    })
}

/// All 714 extracted bodies, keyed by Capture One's `cameraID`.
pub fn all_cameras() -> &'static BTreeMap<String, AllCameraParams> {
    &all_pack().cameras
}

/// Look up any extracted body, including the 703 outside the 23-model full pack.
pub fn for_any_camera(model: &str) -> Option<&'static AllCameraParams> {
    let key = all_index().get(&fingerprint(model))?;
    all_pack().cameras.get(*key)
}

pub fn all_source() -> &'static AllSource {
    &all_pack().source
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn pack_covers_every_extracted_body() {
        assert_eq!(all().len(), 23);
        assert_eq!(
            all().values().filter(|c| c.brand == "Nikon").count(),
            11
        );
        assert_eq!(all().values().filter(|c| c.brand == "Sony").count(), 6);
        assert_eq!(all().values().filter(|c| c.brand == "Canon").count(), 6);
        assert_eq!(
            all().values().map(|c| c.noise.entries.len()).sum::<usize>(),
            689
        );
    }

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn exif_names_resolve_to_capture_one_ids() {
        assert_eq!(for_camera("NIKON Z6_3").unwrap().brand, "Nikon");
        assert_eq!(for_camera("Nikon Z 7_2").unwrap().mount, "Nikon Z");
        assert_eq!(for_camera("Sony ILCE-7M4").unwrap().profile, "SonyA7M4-ProStandard.icm");
        assert_eq!(
            for_camera("Canon EOS R5 Mark II").unwrap().profile,
            "CanonEOSR5Mk2-ProStandard.icm"
        );
        assert_eq!(
            for_camera("Canon EOS 5D Mark IV").unwrap().mount,
            "Canon EF"
        );
        assert!(for_camera("Nikon Z6III").is_some());
        assert!(for_camera("not a camera").is_none());
    }

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn noise_and_ccm_values_are_usable() {
        let z6 = for_camera("Nikon Z6III").unwrap();
        assert_eq!(z6.model_id_u32(), Some(0x3000_0057));
        assert_eq!(z6.cfa, "RGGB");
        assert!((z6.pixel_pitch_um - 5.94).abs() < 0.01);
        let iso50 = z6.noise_at(50).unwrap();
        assert_eq!(iso50.iso, 50);
        assert!((iso50.floor.unwrap() - 0.186_008_8).abs() < 1e-6);
        assert!((iso50.poisson.unwrap() - 0.010_489_9).abs() < 1e-6);
        for row in &z6.ccm {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
        let canon = for_camera("Canon EOS R5").unwrap();
        assert!((canon.pixel_pitch_um - 4.4).abs() < 0.01);
        assert_eq!(canon.noise.entries.len(), 30);
    }

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn source_is_the_installed_16_8_6_build() {
        assert_eq!(source().c1_version, "16.8.6.3784");
        assert_eq!(
            source().dll_sha256,
            "CA02B6AC2786E9A100F63221DF7340E5DF210416EB0BD53A4BB1BFC2306A44A4"
        );
    }

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn all_pack_covers_every_extracted_body() {
        assert_eq!(all_cameras().len(), 714);
        assert_eq!(all_cameras().values().filter(|c| c.brand == "Nikon").count(), 89);
        assert_eq!(all_cameras().values().filter(|c| c.brand == "Sony").count(), 104);
        assert_eq!(all_cameras().values().filter(|c| c.brand == "Canon").count(), 134);
        assert!(all_cameras().values().filter(|c| c.brand == "Fuji").count() >= 60);
        assert_eq!(all_source().scope.contains("Nikon 89"), true);
    }

    #[test]
    #[cfg_attr(not(aurora_private_assets), ignore = "needs private-assets/")]
    fn any_camera_lookup_reaches_the_full_pack() {
        let nikon = for_any_camera("Nikon Z6III").unwrap();
        assert_eq!(nikon.model_id_u32(), Some(0x3000_0057));
        let table = nikon.dll.as_ref().unwrap().noise.as_ref().unwrap();
        let entries = table["entries"].as_array().unwrap();
        assert_eq!(entries.len(), 36);
        assert!((entries[0]["floor"].as_f64().unwrap() - 0.186_008_8).abs() < 1e-6);

        let sony = for_any_camera("Sony ILCE-7M4").unwrap();
        assert_eq!(sony.brand, "Sony");
        assert!(sony.iso.len() > 20);

        let canon = for_any_camera("Canon EOS R5 Mark II").unwrap();
        assert_eq!(canon.brand, "Canon");
        assert!(canon.profile.as_deref().unwrap().contains("CanonEOSR5Mk2"));

        let fuji = for_any_camera("FujiXT5").unwrap();
        assert_eq!(fuji.brand, "Fuji");
    }
}
