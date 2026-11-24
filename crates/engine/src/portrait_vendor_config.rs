//! Read-only loader for decoded PixCake vendor `effect.json` presets.
//!
//! The shipped vendor preset tree is container-obfuscated and is not stored in
//! this repository. A decrypted copy is addressed through
//! [`VENDOR_EFFECT_ROOT_ENV`]. This module only parses and indexes that decoded
//! tree; it changes no render behaviour on its own and never writes to a vendor
//! install directory.
//!
//! The loader exists because the engine currently hardcodes one representative
//! `FilterType` / `ROIRect` / `ShelterType` / `DefaultDegree` / `MixColor` /
//! `Opacity` value per PF (or ignores the field entirely). Vendor values are
//! per-preset, and some PFs (for example `PF101`) appear only inside
//! `ControlContent`, never at the entry's top level. Callers should resolve the
//! exact preset row before applying an effect.

use std::collections::BTreeSet;
use std::fmt;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// Environment variable naming the decoded vendor `effect/config` root.
pub const VENDOR_EFFECT_ROOT_ENV: &str = "AURORA_VENDOR_EFFECT_ROOT";

/// Failure while reading or interpreting a decoded vendor preset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VendorConfigError {
    /// The root or a preset file could not be read.
    Io(String),
    /// A preset file was not valid JSON.
    Json(String),
    /// A preset file was valid JSON but not the expected `Effects` document.
    Shape(String),
}

impl fmt::Display for VendorConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(message) => write!(formatter, "vendor config io error: {message}"),
            Self::Json(message) => write!(formatter, "vendor config json error: {message}"),
            Self::Shape(message) => write!(formatter, "vendor config shape error: {message}"),
        }
    }
}

impl std::error::Error for VendorConfigError {}

/// One `ControlContent[]` row: the UI control mapped to a sub-PF and its ranges.
#[derive(Debug, Clone, PartialEq)]
pub struct ControlRange {
    /// Sub-control `ParamFlag` (for example `101` or `120`..`125`).
    pub param_flag: i64,
    /// `RenderParamRanges`, verbatim.
    pub render_param_ranges: Vec<f64>,
    /// `RenderDegreeRanges`, verbatim (the per-control degree remap).
    pub render_degree_ranges: Vec<f64>,
    /// `DefaultDegree` for this control.
    pub default_degree: Option<f64>,
}

/// One effect entry from a decoded vendor preset file.
#[derive(Debug, Clone, PartialEq)]
pub struct VendorEffect {
    /// Preset path relative to the decoded root, using `/` separators.
    pub config_rel_path: String,
    /// `EffectName`, for example `LipStickLutMakeup`.
    pub effect_name: String,
    /// Top-level `ParamFlag`, when the entry carries one.
    pub param_flag: Option<i64>,
    /// `FilterType`, when present.
    pub filter_type: Option<i64>,
    /// `Paths`, verbatim.
    pub paths: Vec<String>,
    /// `ROIRect` as `[x, y, w, h]`, when present.
    pub roi_rect: Option<[i64; 4]>,
    /// `ShelterType`, when present.
    pub shelter_type: Option<i64>,
    /// `MixColor` as `[r, g, b, a]`, when present.
    pub mix_color: Option<[i64; 4]>,
    /// `Opacity`, when present.
    pub opacity: Option<f64>,
    /// Top-level `DefaultDegree`, when present.
    pub default_degree: Option<f64>,
    /// `DegreeRange` as `[min, max]`, when present.
    pub degree_range: Option<[f64; 2]>,
    /// `ParamRange`, verbatim.
    pub param_range: Vec<f64>,
    /// `EffectRange`, verbatim.
    pub effect_range: Vec<f64>,
    /// `ControlContent[]`, in file order.
    pub controls: Vec<ControlRange>,
    raw: Value,
}

impl VendorEffect {
    /// The original JSON object, for fields this loader does not model.
    pub fn raw(&self) -> &Value {
        &self.raw
    }

    /// True when this entry is driven by `pf` at the top level or via a control.
    pub fn has_pf(&self, pf: i64) -> bool {
        self.param_flag == Some(pf) || self.controls.iter().any(|control| control.param_flag == pf)
    }
}

/// All parsed entries, indexed on demand by PF.
#[derive(Debug, Clone, Default)]
pub struct VendorConfigIndex {
    effects: Vec<VendorEffect>,
}

impl VendorConfigIndex {
    /// Wraps already-parsed entries.
    pub fn from_effects(effects: Vec<VendorEffect>) -> Self {
        Self { effects }
    }

    /// Every parsed entry, in walk order.
    pub fn effects(&self) -> &[VendorEffect] {
        &self.effects
    }

    /// Entries driven by `pf`, at the top level or through a control.
    pub fn for_pf(&self, pf: i64) -> impl Iterator<Item = &VendorEffect> + '_ {
        self.effects.iter().filter(move |effect| effect.has_pf(pf))
    }

    /// Every PF reached by at least one entry (top level or control).
    pub fn pf_set(&self) -> BTreeSet<i64> {
        let mut pfs = BTreeSet::new();
        for effect in &self.effects {
            if let Some(pf) = effect.param_flag {
                pfs.insert(pf);
            }
            for control in &effect.controls {
                pfs.insert(control.param_flag);
            }
        }
        pfs
    }

    /// Walks `root` for `effect.json` / `config.json` files and parses them.
    pub fn load(root: &Path) -> Result<Self, VendorConfigError> {
        if !root.is_dir() {
            return Err(VendorConfigError::Io(format!(
                "{}: not a directory",
                root.display()
            )));
        }
        let mut effects = Vec::new();
        walk(root, root, &mut effects)?;
        Ok(Self { effects })
    }
}

/// Reads [`VENDOR_EFFECT_ROOT_ENV`] when it is set and non-empty.
pub fn root_from_env() -> Option<PathBuf> {
    std::env::var_os(VENDOR_EFFECT_ROOT_ENV)
        .map(PathBuf::from)
        .filter(|path| !path.as_os_str().is_empty())
}

/// Parses one preset document. Entries without an `EffectName` are skipped.
pub fn parse_config(
    config_rel_path: &str,
    text: &str,
) -> Result<Vec<VendorEffect>, VendorConfigError> {
    let document: Value = serde_json::from_str(text)
        .map_err(|error| VendorConfigError::Json(format!("{config_rel_path}: {error}")))?;
    let entries = document
        .get("Effects")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            VendorConfigError::Shape(format!("{config_rel_path}: missing Effects array"))
        })?;
    Ok(entries
        .iter()
        .filter_map(|entry| parse_entry(config_rel_path, entry))
        .collect())
}

fn parse_entry(config_rel_path: &str, entry: &Value) -> Option<VendorEffect> {
    let object = entry.as_object()?;
    let effect_name = object.get("EffectName")?.as_str()?.to_owned();
    let controls = object
        .get("ControlContent")
        .and_then(Value::as_array)
        .map(|rows| {
            rows.iter()
                .filter_map(|row| {
                    let row = row.as_object()?;
                    Some(ControlRange {
                        param_flag: row.get("ParamFlag")?.as_i64()?,
                        render_param_ranges: f64_vec(row.get("RenderParamRanges")),
                        render_degree_ranges: f64_vec(row.get("RenderDegreeRanges")),
                        default_degree: as_f64(row.get("DefaultDegree")),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    Some(VendorEffect {
        config_rel_path: config_rel_path.to_owned(),
        effect_name,
        param_flag: as_i64(object.get("ParamFlag")),
        filter_type: as_i64(object.get("FilterType")),
        paths: str_vec(object.get("Paths")),
        roi_rect: i64_quad(object.get("ROIRect")),
        shelter_type: as_i64(object.get("ShelterType")),
        mix_color: i64_quad(object.get("MixColor")),
        opacity: as_f64(object.get("Opacity")),
        default_degree: as_f64(object.get("DefaultDegree")),
        degree_range: f64_pair(object.get("DegreeRange")),
        param_range: f64_vec(object.get("ParamRange")),
        effect_range: f64_vec(object.get("EffectRange")),
        controls,
        raw: entry.clone(),
    })
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<VendorEffect>) -> Result<(), VendorConfigError> {
    let read_dir = std::fs::read_dir(dir)
        .map_err(|error| VendorConfigError::Io(format!("{}: {error}", dir.display())))?;
    let mut paths: Vec<PathBuf> = read_dir.filter_map(Result::ok).map(|entry| entry.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            walk(root, &path, out)?;
            continue;
        }
        let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
        if !(name.eq_ignore_ascii_case("effect.json") || name.eq_ignore_ascii_case("config.json")) {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .map_err(|error| VendorConfigError::Io(format!("{}: {error}", path.display())))?;
        let relative = path
            .strip_prefix(root)
            .map(Path::to_path_buf)
            .unwrap_or_else(|_| path.clone());
        let relative = relative.to_string_lossy().replace('\\', "/");
        out.extend(parse_config(&relative, &text)?);
    }
    Ok(())
}

fn as_i64(value: Option<&Value>) -> Option<i64> {
    value.and_then(Value::as_i64)
}

fn as_f64(value: Option<&Value>) -> Option<f64> {
    value.and_then(Value::as_f64)
}

fn f64_vec(value: Option<&Value>) -> Vec<f64> {
    value
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_f64).collect())
        .unwrap_or_default()
}

fn str_vec(value: Option<&Value>) -> Vec<String> {
    value
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

fn f64_pair(value: Option<&Value>) -> Option<[f64; 2]> {
    let items = value?.as_array()?;
    if items.len() < 2 {
        return None;
    }
    Some([items[0].as_f64()?, items[1].as_f64()?])
}

fn i64_quad(value: Option<&Value>) -> Option<[i64; 4]> {
    let items = value?.as_array()?;
    if items.len() != 4 {
        return None;
    }
    Some([
        items[0].as_i64()?,
        items[1].as_i64()?,
        items[2].as_i64()?,
        items[3].as_i64()?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"{
      "Effects": [
        {"EffectName":"LipStickLutMakeup","ParamFlag":90021,"FilterType":71,
         "Paths":["0000.png","LipsLUT.png"],"ROIRect":[350,800,300,120],
         "ShelterType":0,"Opacity":80,"DefaultDegree":50,"DegreeRange":[0,100],
         "ParamRange":[0.0,0.5,1.0],"EffectRange":[0.0,0.0,1.0]},
        {"EffectName":"ArmRefine","maxHeadSize":0.36,
         "ControlContent":[{"ParamFlag":400,"RenderParamRanges":[0,0.5,1],
         "RenderDegreeRanges":[-0.5,0.0,0.5],"DefaultDegree":0.5}]}
      ]
    }"#;

    #[test]
    fn parses_top_level_and_control_pf() {
        let effects = parse_config("MakeupEnhance/effect.json", SAMPLE).expect("valid sample");
        assert_eq!(effects.len(), 2);
        let index = VendorConfigIndex::from_effects(effects);
        let lip: Vec<&VendorEffect> = index.for_pf(90021).collect();
        assert_eq!(lip.len(), 1);
        assert_eq!(lip[0].effect_name, "LipStickLutMakeup");
        assert_eq!(lip[0].filter_type, Some(71));
        assert_eq!(
            lip[0].paths,
            vec!["0000.png".to_owned(), "LipsLUT.png".to_owned()]
        );
        assert_eq!(lip[0].roi_rect, Some([350, 800, 300, 120]));
        assert_eq!(lip[0].effect_range, vec![0.0, 0.0, 1.0]);
        let arm: Vec<&VendorEffect> = index.for_pf(400).collect();
        assert_eq!(arm.len(), 1);
        assert_eq!(arm[0].controls.len(), 1);
        assert_eq!(
            arm[0].controls[0].render_degree_ranges,
            vec![-0.5, 0.0, 0.5]
        );
        assert!(index.pf_set().contains(&90021));
        assert!(index.pf_set().contains(&400));
    }

    #[test]
    fn rejects_documents_without_effects() {
        assert!(matches!(
            parse_config("x/effect.json", "{}"),
            Err(VendorConfigError::Shape(_))
        ));
        assert!(matches!(
            parse_config("x/effect.json", "not json"),
            Err(VendorConfigError::Json(_))
        ));
    }

    #[test]
    fn entries_without_effect_name_are_skipped() {
        let effects =
            parse_config("x/effect.json", r#"{"Effects":[{"ParamFlag":1}]}"#).expect("valid json");
        assert!(effects.is_empty());
    }

    #[test]
    fn load_rejects_missing_root() {
        let missing = std::env::temp_dir().join("aurora-vendor-config-missing-root");
        assert!(matches!(
            VendorConfigIndex::load(&missing),
            Err(VendorConfigError::Io(_))
        ));
    }
}
