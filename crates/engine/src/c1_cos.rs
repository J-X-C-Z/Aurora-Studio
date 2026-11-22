//! `.cos` (per-image Capture One settings sidecar) parser and writer.
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
//!
//! [`CosDocument::write`] reproduces the parsed file byte for byte: entry order,
//! key spelling, number formatting and tab indentation all come from the file
//! itself rather than from a serializer. Edits are applied as explicit
//! `(layer, key, value)` replacements through [`CosDocument::write_with`], so
//! the semantic mapping in `presets` stays the only place that formats values.

use anyhow::{anyhow, bail, Result};

/// One local layer of a variant.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CosLayer {
    pub tag: String,
    pub mask_uuid: Option<String>,
    pub adjustments: crate::AdjustmentSet,
    /// The layer's `<E/>` pairs exactly as they appeared, in file order.
    pub entries: Vec<(String, String)>,
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
    /// Variant-level `<E/>` pairs, including `UUID`.
    pub entries: Vec<(String, String)>,
    /// The global `<DL>` layer's `<E/>` pairs.
    pub dl_entries: Vec<(String, String)>,
}

/// A parsed `.cos` sidecar.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CosDocument {
    pub version: Option<u32>,
    pub variants: Vec<CosVariant>,
    /// `<IMG>`-level `<E/>` pairs, in file order.
    pub entries: Vec<(String, String)>,
}

/// One value to replace, or to append when the key is absent.
///
/// `tag` selects the layer: `""` is the `<IMG>` document, `"VAR"` is the
/// variant element, `"DL"` is the global layer, anything else is matched
/// against a local layer tag such as `AL` or `LA`.
#[derive(Clone, Debug, PartialEq)]
pub struct CosEdit {
    pub tag: String,
    pub key: String,
    pub value: String,
}

impl CosEdit {
    pub fn new(
        tag: impl Into<String>,
        key: impl Into<String>,
        value: impl std::fmt::Display,
    ) -> Self {
        Self {
            tag: tag.into(),
            key: key.into(),
            value: value.to_string(),
        }
    }
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
                    let pair = entry(child);
                    if pair.0 == "Version" {
                        cos.version = pair.1.trim().parse().ok();
                    }
                    cos.entries.push(pair);
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

    /// Serialise back to `.cos`, byte-identical to what [`Self::parse`] read.
    pub fn write(&self) -> String {
        self.write_with(&[])
    }

    /// Serialise with `edits` applied. Every other byte is preserved.
    pub fn write_with(&self, edits: &[CosEdit]) -> String {
        let mut out = String::from("<?xml version=\"1.0\"?>\n<IMG>\n");
        emit_entries(&mut out, 1, "", &self.entries, edits);
        for variant in &self.variants {
            match variant.engine {
                Some(engine) => out.push_str(&format!("\t<VAR Engine=\"{engine}\">\n")),
                None => out.push_str("\t<VAR>\n"),
            }
            emit_entries(&mut out, 2, "VAR", &variant.entries, edits);
            emit_layer(&mut out, 2, "DL", &variant.dl_entries, edits);
            for layer in &variant.layers {
                emit_layer(&mut out, 2, &layer.tag, &layer.entries, edits);
            }
            out.push_str("\t</VAR>\n");
        }
        out.push_str("</IMG>\n");
        out
    }
}

fn variant(node: roxmltree::Node) -> CosVariant {
    let mut variant = CosVariant {
        engine: node.attribute("Engine").and_then(|v| v.trim().parse().ok()),
        ..CosVariant::default()
    };
    for child in node.children().filter(|child| child.is_element()) {
        if child.has_tag_name("E") {
            let pair = entry(child);
            if pair.0 == "UUID" {
                variant.uuid = Some(pair.1.clone());
            }
            variant.entries.push(pair);
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
            variant.dl_entries = entries;
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
            entries,
        });
    }
    variant
}

fn entry(node: roxmltree::Node) -> (String, String) {
    (
        node.attribute("K").unwrap_or("").to_owned(),
        node.attribute("V").unwrap_or("").to_owned(),
    )
}

fn entries(node: roxmltree::Node) -> Vec<(String, String)> {
    node.children()
        .filter(|child| child.has_tag_name("E"))
        .map(entry)
        .filter(|(key, _)| !key.is_empty())
        .collect()
}

fn edit_value<'a>(tag: &str, key: &str, edits: &'a [CosEdit]) -> Option<&'a str> {
    edits
        .iter()
        .rev()
        .find(|edit| edit.tag == tag && edit.key == key)
        .map(|edit| edit.value.as_str())
}

fn emit_entries(
    out: &mut String,
    depth: usize,
    tag: &str,
    entries: &[(String, String)],
    edits: &[CosEdit],
) {
    for (key, original) in entries {
        let value = edit_value(tag, key, edits).unwrap_or(original);
        push_entry(out, depth, key, value);
    }
    for edit in edits
        .iter()
        .filter(|edit| edit.tag == tag && !entries.iter().any(|(key, _)| key == &edit.key))
    {
        push_entry(out, depth, &edit.key, &edit.value);
    }
}

fn emit_layer(
    out: &mut String,
    depth: usize,
    tag: &str,
    entries: &[(String, String)],
    edits: &[CosEdit],
) {
    push_indent(out, depth);
    out.push('<');
    out.push_str(tag);
    out.push_str(">\n");
    emit_entries(out, depth + 1, tag, entries, edits);
    push_indent(out, depth);
    out.push_str("</");
    out.push_str(tag);
    out.push_str(">\n");
}

fn push_indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push('\t');
    }
}

fn push_entry(out: &mut String, depth: usize, key: &str, value: &str) {
    push_indent(out, depth);
    out.push_str("<E K=\"");
    out.push_str(&escape(key));
    out.push_str("\" V=\"");
    out.push_str(&escape(value));
    out.push_str("\" />\n");
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
        assert_eq!(
            variant.uuid.as_deref(),
            Some("DD3D34BB-E4F8-9046-AF66-5549BA913807")
        );
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
        assert_eq!(
            layer.mask_uuid.as_deref(),
            Some("24391591-4129-4c9f-a9f5-bb4dd4c41289")
        );
        assert!((layer.adjustments.exposure.exposure + 0.5).abs() < 1e-6);

        assert!(CosDocument::parse(b"<SL/>").is_err());
    }

    #[test]
    fn keeps_the_global_lcc_id() {
        let text = br#"<IMG><VAR Engine="600"><DL><E K="Lccid" V="LCCID1"/></DL></VAR></IMG>"#;
        let document = CosDocument::parse(text).expect("parses");
        assert_eq!(document.variant().unwrap().lcc_id.as_deref(), Some("LCCID1"));
    }

    #[test]
    fn writes_the_real_sidecar_byte_for_byte() {
        let bytes = sample("_sample_layers.cos");
        let document = CosDocument::parse(&bytes).expect("parses");
        let written = document.write();
        assert_eq!(written.as_bytes(), bytes.as_slice());
    }

    #[test]
    fn an_edit_changes_only_the_requested_value() {
        let bytes = sample("_sample_layers.cos");
        let document = CosDocument::parse(&bytes).expect("parses");
        let edited = document.write_with(&[CosEdit::new("DL", "Exposure", "-0.25")]);

        let before: Vec<&str> = std::str::from_utf8(&bytes).unwrap().lines().collect();
        let after: Vec<&str> = edited.lines().collect();
        assert_eq!(before.len(), after.len());
        let changed: Vec<usize> = before
            .iter()
            .zip(&after)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(changed.len(), 1);
        assert!(after[changed[0]].contains("K=\"Exposure\" V=\"-0.25\""));

        let reparsed = CosDocument::parse(edited.as_bytes()).expect("reparses");
        let variant = reparsed.variant().unwrap();
        assert!((variant.adjustments.exposure.exposure + 0.25).abs() < 1e-6);
        assert!((variant.layers[0].adjustments.exposure.exposure + 0.75).abs() < 1e-6);
    }

    #[test]
    fn appends_keys_that_the_file_did_not_have() {
        let bytes = sample("_sample_layers.cos");
        let document = CosDocument::parse(&bytes).expect("parses");
        let edited = document.write_with(&[
            CosEdit::new("DL", "WhiteBalanceKelvin", "5200"),
            CosEdit::new("", "Version", "1531"),
        ]);

        let reparsed = CosDocument::parse(edited.as_bytes()).expect("reparses");
        assert_eq!(reparsed.version, Some(1531));
        assert!(reparsed
            .variant()
            .unwrap()
            .dl_entries
            .iter()
            .any(|(key, value)| key == "WhiteBalanceKelvin" && value == "5200"));
        assert!(edited.contains("\t\t\t<E K=\"WhiteBalanceKelvin\" V=\"5200\" />\n"));
    }
}
