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

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};

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

/// One local layer the caller wants written back, matched by `MaskUUID`.
#[derive(Clone, Debug, PartialEq)]
pub struct CosLayerWrite {
    /// `MaskUUID` of the layer. `None` cannot be matched and is reported.
    pub mask_uuid: Option<String>,
    pub adjustments: crate::AdjustmentSet,
}

/// What a save did, and everything it refused to do.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CosSaveReport {
    /// `(layer, key)` replacements the write applied.
    pub edits: usize,
    /// Tags of the local layers that were written.
    pub layers: Vec<String>,
    /// Layers that could not be written, with the reason.
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum CosSaveOutcome {
    /// Every value already matched; the file was not touched.
    Unchanged(CosSaveReport),
    Written(CosSaveReport),
}

/// Every `MaskUUID` carried by the `.comask` at `path` (empty when it is
/// missing or unreadable).
pub fn comask_uuids(path: &Path) -> HashSet<String> {
    let mut uuids = HashSet::new();
    let Ok(bytes) = std::fs::read(path) else {
        return uuids;
    };
    let Ok(document) = crate::comask::ComaskDocument::parse(&bytes) else {
        return uuids;
    };
    for mask in document.masks {
        if let Some(uuid) = mask.uuid {
            uuids.insert(uuid);
        }
    }
    uuids
}

/// The edits that bring `document` up to date, plus the layers that could not
/// be written. The global `<DL>` layer is always compared.
pub fn plan_edits(
    document: &CosDocument,
    global: &crate::AdjustmentSet,
    layers: &[CosLayerWrite],
    comask: &HashSet<String>,
) -> (Vec<CosEdit>, CosSaveReport) {
    let mut report = CosSaveReport::default();
    let Some(variant) = document.variant() else {
        report
            .warnings
            .push("the file has no <VAR>; nothing was written".to_owned());
        return (Vec::new(), report);
    };
    let mut edits = crate::c1_styles::cos_edits("DL", global, &variant.dl_entries);
    for layer in layers {
        let Some(uuid) = layer.mask_uuid.as_deref().filter(|uuid| !uuid.is_empty()) else {
            report
                .warnings
                .push("a layer has no MaskUUID; it was not written".to_owned());
            continue;
        };
        let Some(cos_layer) = variant
            .layers
            .iter()
            .find(|layer| layer.mask_uuid.as_deref() == Some(uuid))
        else {
            report.warnings.push(format!(
                "layer {uuid}: no local layer carries that MaskUUID in this file; it was not written"
            ));
            continue;
        };
        if !comask.contains(uuid) {
            report.warnings.push(format!(
                "layer {uuid}: no .comask under the settings folder carries that MaskUUID; it was not written"
            ));
            continue;
        }
        let layer_edits = crate::c1_styles::cos_edits(
            &cos_layer.tag,
            &layer.adjustments,
            &cos_layer.entries,
        );
        if layer_edits.is_empty() {
            continue;
        }
        report.layers.push(cos_layer.tag.clone());
        edits.extend(layer_edits);
    }
    report.edits = edits.len();
    (edits, report)
}

/// Writes `global` and the matched local `layers` into the `.cos` at `path`.
///
/// The file is first copied to `<path>.bak`, then replaced through a temporary
/// file in the same folder, so a failed write leaves the original in place.
/// Nothing is written when no value changed.
pub fn save_adjustments(
    path: &Path,
    global: &crate::AdjustmentSet,
    layers: &[CosLayerWrite],
) -> Result<CosSaveOutcome> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let document = CosDocument::parse(&bytes)?;
    let comask = comask_uuids(&path.with_extension("comask"));
    let (edits, report) = plan_edits(&document, global, layers, &comask);
    if edits.is_empty() {
        return Ok(CosSaveOutcome::Unchanged(report));
    }
    let text = document.write_with(&edits);
    let backup = sibling(path, ".bak");
    std::fs::write(&backup, &bytes).with_context(|| format!("backing up {}", backup.display()))?;
    if let Err(error) = replace_atomically(path, text.as_bytes()) {
        // `rename` is the commit point; if it failed the original is still
        // there. Rewrite it anyway so a platform that truncated the target
        // before failing cannot leave a half-written file.
        let _ = std::fs::write(path, &bytes);
        return Err(error).with_context(|| format!("writing {}", path.display()));
    }
    Ok(CosSaveOutcome::Written(report))
}

/// Writes the global layer's `Lccid` pointer.
///
/// `None` writes an empty value, which [`CosVariant::lcc_id`] reads back as
/// absent. The referenced `<raw>.<id>.lcc` sidecar is managed by the caller.
pub fn write_global_lcc_id(path: &Path, lcc_id: Option<&str>) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let document = CosDocument::parse(&bytes)?;
    let variant = document
        .variant()
        .ok_or_else(|| anyhow!("{} has no <VAR>", path.display()))?;
    let value = lcc_id.unwrap_or("");
    if variant.lcc_id.as_deref().unwrap_or("") == value {
        return Ok(());
    }
    let text = document.write_with(&[CosEdit::new("DL", "Lccid", value)]);
    let backup = sibling(path, ".bak");
    std::fs::write(&backup, &bytes).with_context(|| format!("backing up {}", backup.display()))?;
    if let Err(error) = replace_atomically(path, text.as_bytes()) {
        let _ = std::fs::write(path, &bytes);
        return Err(error).with_context(|| format!("writing {}", path.display()));
    }
    Ok(())
}

fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(suffix);
    PathBuf::from(name)
}

fn replace_atomically(path: &Path, bytes: &[u8]) -> Result<()> {
    let temp = sibling(path, &format!(".tmp{}", std::process::id()));
    std::fs::write(&temp, bytes)?;
    std::fs::rename(&temp, path).inspect_err(|_| {
        let _ = std::fs::remove_file(&temp);
    })?;
    Ok(())
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
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
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
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn writes_the_real_sidecar_byte_for_byte() {
        let bytes = sample("_sample_layers.cos");
        let document = CosDocument::parse(&bytes).expect("parses");
        let written = document.write();
        assert_eq!(written.as_bytes(), bytes.as_slice());
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
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
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
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

    fn scratch(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("aurora-cos-save-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("scratch dir");
        dir
    }

    /// One global layer and one local layer that carries `uuid`.
    fn sidecar(uuid: &str, exposure: &str) -> String {
        format!(
            "<?xml version=\"1.0\"?>\n<IMG>\n\t<E K=\"Version\" V=\"1530\" />\n\t<VAR Engine=\"600\">\n\t\t<E K=\"UUID\" V=\"DD3D34BB-E4F8-9046-AF66-5549BA913807\" />\n\t\t<DL>\n\t\t\t<E K=\"Exposure\" V=\"0\" />\n\t\t\t<E K=\"Contrast\" V=\"-8\" />\n\t\t</DL>\n\t\t<AL>\n\t\t\t<E K=\"MaskUUID\" V=\"{uuid}\" />\n\t\t\t<E K=\"Exposure\" V=\"{exposure}\" />\n\t\t</AL>\n\t</VAR>\n</IMG>\n"
        )
    }

    /// The real `.comask` fixture and one `MaskUUID` it carries.
    fn comask_fixture() -> (Vec<u8>, String) {
        let bytes = std::fs::read(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/comask/decode-fixture.comask"),
        )
        .expect("comask fixture");
        let document = crate::comask::ComaskDocument::parse(&bytes).expect("fixture parses");
        let uuid = document
            .masks
            .iter()
            .find_map(|mask| mask.uuid.clone())
            .expect("fixture carries a MaskUUID");
        (bytes, uuid)
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn unchanged_adjustments_leave_the_file_alone() {
        let dir = scratch("unchanged");
        let (comask, uuid) = comask_fixture();
        std::fs::write(dir.join("shot.comask"), &comask).unwrap();
        let path = dir.join("shot.cos");
        let text = sidecar(&uuid, "0");
        std::fs::write(&path, &text).unwrap();

        let document = CosDocument::parse(text.as_bytes()).unwrap();
        let variant = document.variant().unwrap();
        let global = variant.adjustments.clone();
        let layer = CosLayerWrite {
            mask_uuid: Some(uuid),
            adjustments: variant.layers[0].adjustments.clone(),
        };
        let outcome = save_adjustments(&path, &global, &[layer]).unwrap();
        assert_eq!(
            outcome,
            CosSaveOutcome::Unchanged(CosSaveReport::default())
        );
        assert_eq!(std::fs::read(&path).unwrap(), text.as_bytes());
        assert!(!dir.join("shot.cos.bak").exists(), "no backup without a write");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn one_layer_exposure_is_written() {
        let dir = scratch("layer");
        let (comask, uuid) = comask_fixture();
        std::fs::write(dir.join("shot.comask"), &comask).unwrap();
        let path = dir.join("shot.cos");
        let text = sidecar(&uuid, "0");
        std::fs::write(&path, &text).unwrap();

        let document = CosDocument::parse(text.as_bytes()).unwrap();
        let variant = document.variant().unwrap();
        let global = variant.adjustments.clone();
        let mut layer = variant.layers[0].adjustments.clone();
        layer.exposure.exposure = -0.75;
        let outcome = save_adjustments(
            &path,
            &global,
            &[CosLayerWrite { mask_uuid: Some(uuid), adjustments: layer }],
        )
        .unwrap();
        let CosSaveOutcome::Written(report) = outcome else {
            panic!("expected a write");
        };
        assert_eq!(report.layers, vec!["AL".to_owned()]);
        assert!(report.warnings.is_empty(), "{:?}", report.warnings);
        assert_eq!(
            std::fs::read(dir.join("shot.cos.bak")).unwrap(),
            text.as_bytes(),
            "the backup keeps the original bytes"
        );

        let saved = CosDocument::parse(&std::fs::read(&path).unwrap()).unwrap();
        let variant = saved.variant().unwrap();
        assert!((variant.layers[0].adjustments.exposure.exposure + 0.75).abs() < 1e-6);
        assert!(variant.adjustments.exposure.exposure.abs() < 1e-6, "the global layer stays put");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_new_key_is_appended_to_the_global_layer() {
        let dir = scratch("append");
        let path = dir.join("shot.cos");
        let text = sidecar("11111111-1111-1111-1111-111111111111", "0");
        std::fs::write(&path, &text).unwrap();

        let document = CosDocument::parse(text.as_bytes()).unwrap();
        let mut global = document.variant().unwrap().adjustments.clone();
        global.white_balance.kelvin = 5200.0;
        let outcome = save_adjustments(&path, &global, &[]).unwrap();
        assert!(matches!(&outcome, CosSaveOutcome::Written(_)), "{outcome:?}");

        let saved = std::fs::read(&path).unwrap();
        let after = String::from_utf8(saved.clone()).unwrap();
        assert_eq!(after.lines().count(), text.lines().count() + 1, "exactly one key was added");
        let reparsed = CosDocument::parse(&saved).unwrap();
        assert!((reparsed.variant().unwrap().adjustments.white_balance.kelvin - 5200.0).abs() < 1e-3);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_layer_without_a_matching_entry_is_reported() {
        let dir = scratch("missing");
        let path = dir.join("shot.cos");
        let text = sidecar("11111111-1111-1111-1111-111111111111", "0");
        std::fs::write(&path, &text).unwrap();

        let document = CosDocument::parse(text.as_bytes()).unwrap();
        let global = document.variant().unwrap().adjustments.clone();
        let mut moved = document.variant().unwrap().layers[0].adjustments.clone();
        moved.exposure.exposure = -0.75;
        let absent = save_adjustments(
            &path,
            &global,
            &[CosLayerWrite {
                mask_uuid: Some("22222222-2222-2222-2222-222222222222".to_owned()),
                adjustments: moved.clone(),
            }],
        )
        .unwrap();
        let CosSaveOutcome::Unchanged(report) = absent else {
            panic!("expected no write");
        };
        assert!(
            report.warnings.iter().any(|warning| warning.contains("22222222")),
            "{:?}",
            report.warnings
        );
        assert_eq!(std::fs::read(&path).unwrap(), text.as_bytes());

        // The `.cos` has the layer but no `.comask` carries its MaskUUID.
        let no_mask = save_adjustments(
            &path,
            &global,
            &[CosLayerWrite {
                mask_uuid: Some("11111111-1111-1111-1111-111111111111".to_owned()),
                adjustments: moved,
            }],
        )
        .unwrap();
        let CosSaveOutcome::Unchanged(report) = no_mask else {
            panic!("expected no write");
        };
        assert!(
            report.warnings.iter().any(|warning| warning.contains(".comask")),
            "{:?}",
            report.warnings
        );
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    #[cfg_attr(not(feature = "external-fixtures"), ignore = "needs crates/engine/testdata/ (enable --features external-fixtures)")]
    fn a_real_sidecar_round_trips_through_save() {
        let bytes = sample("_sample_layers.cos");
        let dir = scratch("sample");
        let path = dir.join("_sample_layers.cos");
        std::fs::write(&path, &bytes).unwrap();

        let document = CosDocument::parse(&bytes).unwrap();
        let mut global = document.variant().unwrap().adjustments.clone();
        assert!(
            matches!(
                save_adjustments(&path, &global, &[]).unwrap(),
                CosSaveOutcome::Unchanged(_)
            ),
            "reading the file back must not rewrite it"
        );
        assert_eq!(std::fs::read(&path).unwrap(), bytes);

        global.exposure.exposure = -0.25;
        let outcome = save_adjustments(&path, &global, &[]).unwrap();
        let CosSaveOutcome::Written(report) = outcome else {
            panic!("expected a write");
        };
        assert!(report.edits >= 1);

        let saved = std::fs::read(&path).unwrap();
        let before: Vec<&str> = std::str::from_utf8(&bytes).unwrap().lines().collect();
        let after: Vec<&str> = std::str::from_utf8(&saved).unwrap().lines().collect();
        assert_eq!(before.len(), after.len(), "an edit replaces a value, it does not add lines");
        let changed: Vec<usize> = before
            .iter()
            .zip(&after)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(index, _)| index)
            .collect();
        assert_eq!(changed.len(), 1, "one line must change");
        assert!(
            after[changed[0]].contains("K=\"Exposure\" V=\"-0.25\""),
            "{}",
            after[changed[0]]
        );

        let reparsed = CosDocument::parse(&saved).unwrap();
        let variant = reparsed.variant().unwrap();
        assert!((variant.adjustments.exposure.exposure + 0.25).abs() < 1e-6);
        assert!((variant.layers[0].adjustments.exposure.exposure + 0.75).abs() < 1e-6);
        std::fs::remove_dir_all(dir).ok();
    }
}
