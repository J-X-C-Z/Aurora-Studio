//! Original, editable whole-look recipes. A recipe is one adjustment edit, not
//! an extra PF90033 render pass in addition to its individual effects.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::AdjustmentSet;
use crate::portrait_makeup_wiring::{makeup_asset_root, makeup_palette_for_pf, makeup_presets_at};

const SCHEMA: &str = "aurora.suit.recipe/1";

#[derive(Clone, Debug, PartialEq)]
pub struct MakeupSuit {
    pub id: String,
    pub label: String,
    /// Same normalized degrees, 1-based ordinals and packed RGB24 as the editor.
    pub effects: BTreeMap<String, f32>,
    /// Incomplete looks remain visible but must not be partially applied.
    pub unavailable_reason: Option<String>,
    root: PathBuf,
    /// The file each 1-based ordinal selected when the recipe was displayed.
    materials: BTreeMap<u32, PathBuf>,
}

#[derive(Deserialize)]
struct Label {
    zh: String,
    #[serde(default)]
    en: String,
}
#[derive(Deserialize)]
struct Unbound {
    status: String,
    subclass: String,
    #[serde(default)]
    strength: f64,
}
#[derive(Deserialize)]
struct Recipe {
    schema: String,
    category: String,
    id: String,
    label: Label,
    apply: BTreeMap<String, f64>,
    #[serde(default)]
    unbound: Vec<Unbound>,
}

// Exclusive ownership of these controls lets switching a look clear an absent
// lower lash or sticker without touching face shape, skin, teeth, or unknown keys.
fn suit_key(key: &str) -> Option<(u32, &str)> {
    let key = key.strip_prefix("pc:")?;
    let (id, kind) = key.split_once(':').unwrap_or((key, ""));
    let pf: u32 = id.parse().ok()?;
    if !matches!(
        pf,
        100 | 101 | 103 | 105 | 106 | 107 | 111 | 115 | 116 | 117 | 118 | 30007 | 90032
    ) || id != pf.to_string()
        || !matches!(kind, "" | "style" | "color" | "rgb")
        || key.ends_with(':')
    {
        return None;
    }
    Some((pf, kind))
}

fn valid_value(kind: &str, value: f64) -> bool {
    match kind {
        "" => (0.0..=1.0).contains(&value),
        "style" | "color" => (1.0..=16_777_215.0).contains(&value) && value.fract() == 0.0,
        "rgb" => (0.0..=16_777_215.0).contains(&value) && value.fract() == 0.0,
        _ => false,
    }
}

fn decode(
    recipe: Recipe,
    id: &str,
    root: &Path,
    catalogue: &mut BTreeMap<u32, Vec<PathBuf>>,
) -> Option<MakeupSuit> {
    if recipe.schema != SCHEMA || recipe.category != "Suit" || recipe.id != id || id.is_empty() {
        return None;
    }
    let label = if recipe.label.zh.trim().is_empty() {
        recipe.label.en
    } else {
        recipe.label.zh
    };
    if label.trim().is_empty() || recipe.apply.is_empty() {
        return None;
    }
    let mut materials = BTreeMap::new();
    let mut missing = Vec::new();
    for (k, &v) in &recipe.apply {
        let (pf, kind) = suit_key(k)?;
        if !valid_value(kind, v) || !recipe.apply.contains_key(&format!("pc:{pf}")) {
            return None;
        }
        match kind {
            "style" => {
                let paths = catalogue.entry(pf).or_insert_with(|| {
                    makeup_presets_at(root, pf)
                        .into_iter()
                        .map(|p| p.path)
                        .collect()
                });
                match paths.get(v as usize - 1) {
                    Some(p) => {
                        materials.insert(pf, p.clone());
                    }
                    None => missing.push(k.clone()), // no silent clamping to a different style
                }
            }
            // An explicit RGB overrides the indexed swatch, by design.
            "color"
                if !recipe.apply.contains_key(&format!("pc:{pf}:rgb"))
                    && v as usize > makeup_palette_for_pf(pf).len() =>
            {
                missing.push(k.clone())
            }
            _ => (),
        }
    }
    for entry in recipe.unbound {
        if entry.strength != 0.0 && !matches!(entry.status.as_str(), "not-selected" | "excluded") {
            missing.push(entry.subclass);
        }
    }
    let unavailable_reason = (!missing.is_empty()).then(|| missing.join(", "));
    Some(MakeupSuit {
        id: recipe.id,
        label,
        effects: recipe
            .apply
            .into_iter()
            .map(|(k, v)| (k, v as f32))
            .collect(),
        unavailable_reason,
        root: root.to_owned(),
        materials,
    })
}

/// Discover only the product's own typed recipes; a third-party preset or
/// an unrelated file under Makeups/Suit is not interpreted as an edit command.
pub fn makeup_suits() -> Vec<MakeupSuit> {
    makeup_asset_root().map_or_else(Vec::new, |root| discover(&root))
}

fn discover(root: &Path) -> Vec<MakeupSuit> {
    let Ok(entries) = std::fs::read_dir(root.join("Makeups").join("Suit")) else {
        return Vec::new();
    };
    let mut catalogue = BTreeMap::new();
    let mut suits: Vec<_> = entries
        .flatten()
        .filter_map(|e| {
            if !e.file_type().ok()?.is_dir() {
                return None;
            }
            let id = e.file_name().into_string().ok()?;
            let path = e.path().join("suit.json");
            if std::fs::metadata(&path).ok()?.len() > 256 * 1024 {
                return None;
            }
            let recipe: Recipe = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
            decode(recipe, &id, &root, &mut catalogue)
        })
        .collect();
    suits.sort_by(|a, b| a.id.cmp(&b.id));
    suits
}

/// Validate before changing anything. Wrap this call in one editor undo/redo
/// transaction. A missing resource or unresolved active component is an error,
/// not a request to silently apply the remainder.
pub fn apply_makeup_suit(suit: &MakeupSuit, adj: &mut AdjustmentSet) -> Result<(), String> {
    apply_at(makeup_asset_root().as_deref(), suit, adj)
}

fn apply_at(root: Option<&Path>, suit: &MakeupSuit, adj: &mut AdjustmentSet) -> Result<(), String> {
    if let Some(reason) = &suit.unavailable_reason {
        return Err(reason.clone());
    }
    let root = root
        .filter(|r| *r == suit.root.as_path())
        .ok_or("makeup asset root changed")?;
    for (key, &value) in &suit.effects {
        if let Some((pf, "style")) = suit_key(key) {
            let current = makeup_presets_at(root, pf);
            if !valid_value("style", value as f64)
                || !current
                    .get(value as usize - 1)
                    .is_some_and(|p| suit.materials.get(&pf) == Some(&p.path))
            {
                return Err(format!("{key}: makeup asset catalogue changed"));
            }
        }
    }
    replace_controls(&suit.effects, adj)
}

fn replace_controls(
    effects: &BTreeMap<String, f32>,
    adj: &mut AdjustmentSet,
) -> Result<(), String> {
    if effects.is_empty()
        || effects.iter().any(|(k, &v)| {
            !suit_key(k).is_some_and(|(pf, kind)| {
                valid_value(kind, v as f64) && effects.contains_key(&format!("pc:{pf}"))
            })
        })
    {
        return Err("invalid makeup recipe control".into());
    }
    adj.extra.retain(|k, _| suit_key(k).is_none());
    adj.extra.extend(effects.clone());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn recipe(apply: serde_json::Value, unbound: serde_json::Value) -> Recipe {
        serde_json::from_value(serde_json::json!({"schema":SCHEMA,"category":"Suit","id":"look-1","label":{"zh":"测试"},"apply":apply,"unbound":unbound})).unwrap()
    }

    #[test]
    fn switching_looks_replaces_only_makeup_and_rejects_an_invalid_transaction_atomically() {
        let mut adj = AdjustmentSet::default();
        adj.extra = BTreeMap::from([
            ("pc:115".into(), 0.4),
            ("pc:115:style".into(), 4.0),
            ("pc:108".into(), 0.8),
            ("pc:115:future".into(), 7.0),
        ]);
        let controls = BTreeMap::from([("pc:100".into(), 0.3), ("pc:100:rgb".into(), 0.0)]);
        replace_controls(&controls, &mut adj).unwrap();
        assert_eq!(adj.extra.get("pc:100:rgb"), Some(&0.0));
        assert_eq!(adj.extra.get("pc:108"), Some(&0.8));
        assert!(adj.extra.contains_key("pc:115:future"));
        assert!(!adj.extra.contains_key("pc:115") && !adj.extra.contains_key("pc:115:style"));
        let before = adj.extra.clone();
        assert!(replace_controls(&BTreeMap::from([("pc:109".into(), 0.8)]), &mut adj).is_err());
        assert_eq!(adj.extra, before);
    }

    #[test]
    fn discovered_look_binds_the_correct_eye_material_and_rejects_reordered_ordinals() {
        struct Temp(PathBuf);
        impl Drop for Temp {
            fn drop(&mut self) {
                let _ = std::fs::remove_dir_all(&self.0);
            }
        }
        let tmp = Temp(std::env::temp_dir().join(
            format!("aurora-look-{}-{}",std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()),
        ));
        let cat = tmp.0.join("Makeups/Eyemakeup");
        let style = cat.join("style2");
        let look = tmp.0.join("Makeups/Suit/look-1");
        std::fs::create_dir_all(&style).unwrap();
        std::fs::create_dir_all(&look).unwrap();
        let pixel = image::RgbaImage::from_pixel(1, 1, image::Rgba([60, 70, 80, 255]));
        for n in ["yyz1.png", "yyy1.png"] {
            pixel.save(style.join(n)).unwrap();
        }
        std::fs::write(look.join("suit.json"), serde_json::to_vec(&serde_json::json!({"schema":SCHEMA,"category":"Suit","id":"look-1","label":{"zh":"眼妆"},"apply":{"pc:90032":0.25,"pc:90032:style":1}})).unwrap()).unwrap();
        let suits = discover(&tmp.0);
        assert_eq!(suits.len(), 1);
        let suit = &suits[0];
        assert_eq!(suit.materials[&90032], style.join("yyz1.png"));
        let mut adj = AdjustmentSet::default();
        apply_at(Some(&tmp.0), suit, &mut adj).unwrap();
        assert_eq!(adj.extra["pc:90032:style"], 1.0);
        assert_eq!(adj.extra["pc:90032"], 0.25);
        let earlier = cat.join("style1");
        std::fs::create_dir(&earlier).unwrap();
        pixel.save(earlier.join("yyz.png")).unwrap();
        adj.extra.insert("pc:90032".into(), 0.71);
        let before = adj.extra.clone();
        assert!(apply_at(Some(&tmp.0), suit, &mut adj).is_err());
        assert_eq!(adj.extra, before);
    }

    #[test]
    fn incomplete_recipes_are_not_clamped_or_partially_applied() {
        use serde_json::json;
        let mut cache = BTreeMap::from([(100, vec![PathBuf::from("first.png")])]);
        let spec = recipe(
            json!({"pc:100":0.2,"pc:100:style":2}),
            json!([{"status":"pf-unresolved","subclass":"Catchlights","strength":0.5}]),
        );
        let suit = decode(spec, "look-1", Path::new("."), &mut cache).unwrap();
        let reason = suit.unavailable_reason.as_deref().unwrap();
        assert!(reason.contains("pc:100:style") && reason.contains("Catchlights"));
        let mut adj = AdjustmentSet::default();
        assert!(apply_makeup_suit(&suit, &mut adj).is_err());
        assert!(adj.extra.is_empty());
        for invalid in [
            json!({"pc:100":0.5,"pc:100:rgb":16777216}),
            json!({"pc:100:style":1}),
            json!({"pc:100:":0.2}),
        ] {
            assert!(
                decode(
                    recipe(invalid, json!([])),
                    "look-1",
                    Path::new("."),
                    &mut cache
                )
                .is_none()
            );
        }
    }
}
