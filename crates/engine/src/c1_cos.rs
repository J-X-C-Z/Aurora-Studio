//! Read-only `.cos` (per-image Capture One settings sidecar) parser.
//!
//! Shape, confirmed against C1's own writer and reader (see
//! `04_reference/.../sdk/formats/cos.py`): `<IMG>` holds image keys and one
//! `<VAR>` per variant; inside a variant `<DL>` is the global adjustment layer
//! and `<AL>`/`<LA>`/`<SL>`-style tags are local layers. Every layer is a flat
//! list of `<E K="..." V="..."/>` pairs using the same key vocabulary as
//! `.costyle`, so the style key mapping is reused rather than duplicated.
//!
//! `MaskUUID` (persisted name; the catalog column is `ZVARIANTLAYER.ZMASKUUID`)
//! links a local layer to its `.comask` mask.

use anyhow::{anyhow, bail, Result};

/// One local layer of a variant.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CosLayer {
    pub tag: String,
    pub mask_uuid: Option<String>,
    pub adjustments: crate::AdjustmentSet,
}

/// One `<VAR>` of a `.cos` sidecar.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CosVariant {
    pub uuid: Option<String>,
    pub engine: Option<u64>,
    /// `Lccid` on the global layer: the `.lcc` file is `<raw name>.<id>.lcc`
    /// in the same Settings folder.
    pub lcc_id: Option<String>,
    pub adjustments: crate::AdjustmentSet,
    pub layers: Vec<CosLayer>,
}

/// A parsed `.cos` sidecar.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CosDocument {
    pub version: Option<u32>,
    pub variants: Vec<CosVariant>,
}

impl CosDocument {
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(bytes).map_err(|_| anyhow!("`.cos` is not UTF-8"))?;
        let text = text.trim_start_matches('\u{feff}');
        let document = roxmltree::Document::parse(text)
            .map_err(|error| anyhow!("invalid `.cos` XML: {error}"))?;
        let root = document.root_element();
        if !root.has_tag_name("IMG") {
            bail!("not a `.cos` document: root is `{}`", root.tag_name().name());
        }

        let mut cos = CosDocument::default();
        for child in root.children().filter(|node| node.is_element()) {
            match child.tag_name().name() {
                "E" => {
                    if child.attribute("K") == Some("Version") {
                        cos.version = child.attribute("V").and_then(|v| v.trim().parse().ok());
                    }
                }
                "VAR" => cos.variants.push(variant(child)),
                _ => {}
            }
        }
        Ok(cos)
    }

    /// First variant, which is what a single-image sidecar carries.
    pub fn variant(&self) -> Option<&CosVariant> {
        self.variants.first()
    }
}

fn variant(node: roxmltree::Node) -> CosVariant {
    let mut variant = CosVariant {
        engine: node.attribute("Engine").and_then(|v| v.trim().parse().ok()),
        ..CosVariant::default()
    };
    for child in node.children().filter(|child| child.is_element()) {
        if child.has_tag_name("E") {
            if child.attribute("K") == Some("UUID") {
                variant.uuid = child.attribute("V").map(str::to_owned);
            }
            continue;
        }
        let entries = entries(child);
        let tag = child.tag_name().name().to_owned();
        if tag == "DL" {
            variant.lcc_id = entries
                .iter()
                .find(|(key, _)| key == "Lccid")
                .map(|(_, value)| value.trim().to_owned())
                .filter(|value| !value.is_empty());
            variant.adjustments = adjustments(&entries, variant.engine);
            continue;
        }
        let mask_uuid = entries
            .iter()
            .find(|(key, _)| key == "MaskUUID")
            .map(|(_, value)| value.clone());
        variant.layers.push(CosLayer {
            tag,
            mask_uuid,
            adjustments: adjustments(&entries, variant.engine),
        });
    }
    variant
}

fn entries(node: roxmltree::Node) -> Vec<(String, String)> {
    node.children()
        .filter(|child| child.has_tag_name("E"))
        .filter_map(|child| {
            let key = child.attribute("K")?;
            Some((key.to_owned(), child.attribute("V").unwrap_or("").to_owned()))
        })
        .collect()
}

/// Reuses the `.costyle` key mapping: build an `<SL>` body and parse it.
fn adjustments(entries: &[(String, String)], engine: Option<u64>) -> crate::AdjustmentSet {
    let mut text = String::from("<SL");
    if let Some(engine) = engine {
        text.push_str(&format!(" Engine=\"{engine}\""));
    }
    text.push('>');
    for (key, value) in entries {
        text.push_str(&format!(
            "<E K=\"{}\" V=\"{}\"/>",
            escape(key),
            escape(value)
        ));
    }
    text.push_str("</SL>");
    let preset = crate::presets::preset_from_sl(&text, crate::PresetKind::Style, "", "cos");
    if preset.is_none() {
        log::warn!("`.cos` layer could not be mapped, keeping defaults");
    }
    preset.map(|preset| preset.adjustments).unwrap_or_default()
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn sample(name: &str) -> Vec<u8> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../../../04_reference/legacy-root/逆向工程/Capture One/resources/cos_samples")
            .join(name);
        std::fs::read(path).expect("sample `.cos`")
    }

    #[test]
    fn parses_a_real_sidecar_with_global_and_local_layers() {
        let document = CosDocument::parse(&sample("_sample_layers.cos")).expect("parses");
        assert_eq!(document.version, Some(1530));
        let variant = document.variant().expect("one variant");
        assert_eq!(variant.uuid.as_deref(), Some("DD3D34BB-E4F8-9046-AF66-5549BA913807"));
        assert_eq!(variant.engine, Some(600));
        assert!((variant.adjustments.exposure.exposure - 1.25).abs() < 1e-6);
        assert_eq!(variant.layers.len(), 1);
        assert_eq!(variant.layers[0].tag, "AL");
        assert!((variant.layers[0].adjustments.exposure.exposure + 0.75).abs() < 1e-6);
        assert_eq!(variant.layers[0].mask_uuid, None);
    }

    #[test]
    fn keeps_the_layer_mask_uuid_and_rejects_foreign_xml() {
        let text = br#"<?xml version="1.0"?><IMG><E K="Version" V="1530"/><VAR Engine="600"><E K="UUID" V="DD3D34BB-E4F8-9046-AF66-5549BA913807"/><DL/><AL><E K="MaskUUID" V="24391591-4129-4c9f-a9f5-bb4dd4c41289"/><E K="Exposure" V="-0.5"/></AL></VAR></IMG>"#;
        let document = CosDocument::parse(text).expect("parses");
        let layer = &document.variant().unwrap().layers[0];
        assert_eq!(layer.mask_uuid.as_deref(), Some("24391591-4129-4c9f-a9f5-bb4dd4c41289"));
        assert!((layer.adjustments.exposure.exposure + 0.5).abs() < 1e-6);

        assert!(CosDocument::parse(b"<SL/>").is_err());
    }

    #[test]
    fn keeps_the_global_lcc_id() {
        let text = br#"<IMG><VAR Engine="600"><DL><E K="Lccid" V="LCCID1"/></DL></VAR></IMG>"#;
        let document = CosDocument::parse(text).expect("parses");
        assert_eq!(document.variant().unwrap().lcc_id.as_deref(), Some("LCCID1"));
    }
}
