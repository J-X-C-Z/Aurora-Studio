//! XMP sidecars: the Portrait rating triplet plus IPTC, GPS and EXIF capture data.
//!
//! The sidecar is `<image stem>.xmp` next to the image. It is written only when
//! a caller asks for it, atomically (`.tmp` then rename), and every property
//! Portrait does not own is copied through byte for byte, so a Capture One or
//! Lightroom packet keeps its `crs:`, `photoshop:` and `tiff:` data.

use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use roxmltree::{Document, Node};
use serde::{Deserialize, Serialize};

const XMP_NS: &str = "http://ns.adobe.com/xap/1.0/";
const DC_NS: &str = "http://purl.org/dc/elements/1.1/";
const PHOTOSHOP_NS: &str = "http://ns.adobe.com/photoshop/1.0/";
const EXIF_NS: &str = "http://ns.adobe.com/exif/1.0/";
const TIFF_NS: &str = "http://ns.adobe.com/tiff/1.0/";
const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
const XML_NS: &str = "http://www.w3.org/XML/1998/namespace";
const X_NS: &str = "adobe:ns:meta/";

/// Metadata Portrait mirrors to XMP.
///
/// Text is kept as the packet spells it: GPS coordinates use XMP's
/// `DDD,MM.mmmN` form and shutter speed its rational form (`"1/250"`), so a
/// value written by another tool survives a read/write cycle unchanged.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct XmpMeta {
    /// -1 (rejected) ..= 5 stars.
    pub rating: i8,
    /// Colour label name, e.g. `"Red"`; empty means no label.
    pub label: String,
    pub keywords: Vec<String>,
    /// `dc:creator`, ordered names.
    pub creator: Vec<String>,
    /// `dc:rights` (`x-default`).
    pub copyright: Option<String>,
    /// `dc:description` (`x-default`).
    pub description: Option<String>,
    /// `photoshop:City`.
    pub city: Option<String>,
    /// `photoshop:Country`.
    pub country: Option<String>,
    /// `exif:GPSLatitude`, e.g. `"31,13.8240N"`.
    pub gps_latitude: Option<String>,
    /// `exif:GPSLongitude`, e.g. `"121,28.4400E"`.
    pub gps_longitude: Option<String>,
    /// `exif:GPSAltitude` in metres; below sea level is negative.
    pub gps_altitude: Option<f64>,
    /// `tiff:Make`.
    pub make: Option<String>,
    /// `tiff:Model`.
    pub model: Option<String>,
    /// `exif:LensModel`.
    pub lens: Option<String>,
    /// `exif:ISOSpeedRatings`.
    pub iso: Option<u32>,
    /// `exif:ExposureTime`, e.g. `"1/250"`.
    pub shutter: Option<String>,
    /// `exif:FNumber`.
    pub aperture: Option<f64>,
    /// `exif:DateTimeOriginal`.
    pub datetime: Option<String>,
}

impl XmpMeta {
    /// `<dir>/<stem>.xmp`, next to the image.
    pub fn sidecar_path(image: &Path) -> PathBuf {
        image.with_extension("xmp")
    }

    /// Reads the sidecar of `image`. `Ok(None)` when the file does not exist.
    pub fn load(image: &Path) -> Result<Option<Self>> {
        let path = Self::sidecar_path(image);
        match fs::read_to_string(&path) {
            Ok(text) => Self::parse(&text).map(Some).with_context(|| format!("parsing {}", path.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
        }
    }

    /// Parses an XMP packet. An empty file is an empty record; malformed XML is
    /// an error, never a panic.
    pub fn parse(text: &str) -> Result<Self> {
        if text.trim().is_empty() {
            return Ok(Self::default());
        }
        let doc = Document::parse(text).map_err(|e| anyhow!("invalid XMP: {e}"))?;
        let altitude = find_property(&doc, EXIF_NS, "GPSAltitude").and_then(|v| v.parse::<f64>().ok());
        let below_sea = find_property(&doc, EXIF_NS, "GPSAltitudeRef").is_some_and(|v| v.trim() == "1");
        Ok(Self {
            rating: find_property(&doc, XMP_NS, "Rating")
                .map(|v| parse_rating(Some(&v)))
                .unwrap_or(0),
            label: find_property(&doc, XMP_NS, "Label").unwrap_or_default(),
            keywords: find_element(&doc, DC_NS, "subject").map(list_items).unwrap_or_default(),
            creator: find_element(&doc, DC_NS, "creator").map(list_items).unwrap_or_default(),
            copyright: find_element(&doc, DC_NS, "rights").and_then(lang_alt),
            description: find_element(&doc, DC_NS, "description").and_then(lang_alt),
            city: find_property(&doc, PHOTOSHOP_NS, "City"),
            country: find_property(&doc, PHOTOSHOP_NS, "Country"),
            gps_latitude: find_property(&doc, EXIF_NS, "GPSLatitude"),
            gps_longitude: find_property(&doc, EXIF_NS, "GPSLongitude"),
            gps_altitude: altitude.map(|v| if below_sea { -v.abs() } else { v.abs() }),
            make: find_property(&doc, TIFF_NS, "Make"),
            model: find_property(&doc, TIFF_NS, "Model"),
            lens: find_property(&doc, EXIF_NS, "LensModel"),
            iso: find_element(&doc, EXIF_NS, "ISOSpeedRatings")
                .map(list_items)
                .and_then(|v| v.first().and_then(|s| s.trim().parse().ok()))
                .or_else(|| find_property(&doc, EXIF_NS, "ISOSpeedRatings").and_then(|v| v.parse().ok())),
            shutter: find_property(&doc, EXIF_NS, "ExposureTime"),
            aperture: find_property(&doc, EXIF_NS, "FNumber").and_then(|v| v.parse().ok()),
            datetime: find_property(&doc, EXIF_NS, "DateTimeOriginal"),
        })
    }

    /// Writes the sidecar of `image`, merging into an existing packet.
    pub fn save(&self, image: &Path) -> Result<PathBuf> {
        let path = Self::sidecar_path(image);
        if in_c1_bundle(&path) {
            return Err(anyhow!("refusing to write inside a Capture One bundle: {}", path.display()));
        }
        let existing = match fs::read_to_string(&path) {
            Ok(text) => Some(text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        // Parse before touching the file: a damaged packet is never overwritten.
        let text = self.render(existing.as_deref())?;
        crate::store::write_atomic(&path, text.as_bytes())?;
        Ok(path)
    }

    /// Merges into `existing`, or builds a fresh packet when there is none.
    pub fn render(&self, existing: Option<&str>) -> Result<String> {
        let text = existing.filter(|t| !t.trim().is_empty());
        match text {
            None => Ok(self.new_packet()),
            Some(text) => self.merge(text),
        }
    }

    fn new_packet(&self) -> String {
        self.packet_with(true)
    }

    /// The packet as embedded in an image file.
    ///
    /// Unlike a sidecar, a field this record does not carry is left out
    /// instead of written as a default, so a keywords-only packet does not
    /// claim a rating.
    pub fn packet(&self) -> String {
        self.packet_with(false)
    }

    fn packet_with(&self, always_rating: bool) -> String {
        let city = non_empty(self.city.as_deref());
        let country = non_empty(self.country.as_deref());
        let gps_latitude = non_empty(self.gps_latitude.as_deref());
        let gps_longitude = non_empty(self.gps_longitude.as_deref());
        let gps_altitude = self.gps_altitude.filter(|v| v.is_finite());
        let make = non_empty(self.make.as_deref());
        let model = non_empty(self.model.as_deref());
        let lens = non_empty(self.lens.as_deref());
        let shutter = non_empty(self.shutter.as_deref());
        let aperture = self.aperture.filter(|v| v.is_finite());
        let datetime = non_empty(self.datetime.as_deref());
        let has_photoshop = city.is_some() || country.is_some();
        let has_tiff = make.is_some() || model.is_some();
        let has_exif = gps_latitude.is_some()
            || gps_longitude.is_some()
            || gps_altitude.is_some()
            || lens.is_some()
            || self.iso.is_some()
            || shutter.is_some()
            || aperture.is_some()
            || datetime.is_some();
        let mut description = String::from(
            "  <rdf:Description rdf:about=\"\"\n\
             \x20   xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\"\n\
             \x20   xmlns:dc=\"http://purl.org/dc/elements/1.1/\"",
        );
        if has_photoshop {
            description.push_str("\n    xmlns:photoshop=\"http://ns.adobe.com/photoshop/1.0/\"");
        }
        if has_exif {
            description.push_str("\n    xmlns:exif=\"http://ns.adobe.com/exif/1.0/\"");
        }
        if has_tiff {
            description.push_str("\n    xmlns:tiff=\"http://ns.adobe.com/tiff/1.0/\"");
        }
        if always_rating || self.rating != 0 {
            push_attr(&mut description, "xmp", "Rating", &self.rating.clamp(-1, 5).to_string());
        }
        if !self.label.is_empty() {
            push_attr(&mut description, "xmp", "Label", &self.label);
        }
        if let Some(value) = city {
            push_attr(&mut description, "photoshop", "City", value);
        }
        if let Some(value) = country {
            push_attr(&mut description, "photoshop", "Country", value);
        }
        if let Some(value) = gps_latitude {
            push_attr(&mut description, "exif", "GPSLatitude", value);
        }
        if let Some(value) = gps_longitude {
            push_attr(&mut description, "exif", "GPSLongitude", value);
        }
        if let Some(value) = gps_altitude {
            push_attr(&mut description, "exif", "GPSAltitude", &value.abs().to_string());
            push_attr(&mut description, "exif", "GPSAltitudeRef", if value < 0.0 { "1" } else { "0" });
        }
        if let Some(value) = make {
            push_attr(&mut description, "tiff", "Make", value);
        }
        if let Some(value) = model {
            push_attr(&mut description, "tiff", "Model", value);
        }
        if let Some(value) = lens {
            push_attr(&mut description, "exif", "LensModel", value);
        }
        if let Some(value) = shutter {
            push_attr(&mut description, "exif", "ExposureTime", value);
        }
        if let Some(value) = aperture {
            push_attr(&mut description, "exif", "FNumber", &value.to_string());
        }
        if let Some(value) = datetime {
            push_attr(&mut description, "exif", "DateTimeOriginal", value);
        }

        let mut children = String::new();
        if !self.keywords.is_empty() {
            children.push_str(&format!("   <dc:subject>{}</dc:subject>\n", bag("rdf", &self.keywords)));
        }
        if !self.creator.is_empty() {
            children.push_str(&format!("   <dc:creator>{}</dc:creator>\n", seq("rdf", &self.creator)));
        }
        if let Some(value) = non_empty(self.copyright.as_deref()) {
            children.push_str(&format!("   <dc:rights>{}</dc:rights>\n", alt("rdf", value)));
        }
        if let Some(value) = non_empty(self.description.as_deref()) {
            children.push_str(&format!("   <dc:description>{}</dc:description>\n", alt("rdf", value)));
        }
        if let Some(value) = self.iso {
            children.push_str(&format!(
                "   <exif:ISOSpeedRatings>{}</exif:ISOSpeedRatings>\n",
                seq("rdf", &[value.to_string()])
            ));
        }
        if children.is_empty() {
            description.push_str("/>\n");
        } else {
            description.push_str(">\n");
            description.push_str(&children);
            description.push_str("  </rdf:Description>\n");
        }
        format!(
            "<?xpacket begin=\"\u{feff}\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>\n\
             <x:xmpmeta xmlns:x=\"{X_NS}\">\n\
             \x20<rdf:RDF xmlns:rdf=\"{RDF_NS}\">\n\
             {description}\
             \x20</rdf:RDF>\n\
             </x:xmpmeta>\n\
             <?xpacket end=\"w\"?>"
        )
    }

    fn merge(&self, text: &str) -> Result<String> {
        let doc = Document::parse(text).map_err(|e| anyhow!("invalid XMP: {e}"))?;
        let desc = pick_description(&doc).ok_or_else(|| anyhow!("XMP has no rdf:Description"))?;
        let start = desc.range().start;
        let tag_close = tag_end(text, start).ok_or_else(|| anyhow!("unterminated rdf:Description"))?;
        let mut tag = text[start..=tag_close].to_string();

        let rdf_prefix = prefix_for(text, RDF_NS).unwrap_or("rdf");
        let close_name = qname(text, start);
        let mut edits = Vec::new();

        let xmp = ns_prefixes(desc, &mut tag, XMP_NS, "xmp", true);
        let needs_dc = !self.keywords.is_empty()
            || !self.creator.is_empty()
            || non_empty(self.copyright.as_deref()).is_some()
            || non_empty(self.description.as_deref()).is_some();
        let dc = ns_prefixes(desc, &mut tag, DC_NS, "dc", needs_dc);
        let needs_photoshop = non_empty(self.city.as_deref()).is_some() || non_empty(self.country.as_deref()).is_some();
        let photoshop = ns_prefixes(desc, &mut tag, PHOTOSHOP_NS, "photoshop", needs_photoshop);
        let needs_tiff = non_empty(self.make.as_deref()).is_some() || non_empty(self.model.as_deref()).is_some();
        let tiff = ns_prefixes(desc, &mut tag, TIFF_NS, "tiff", needs_tiff);
        let altitude = self.gps_altitude.filter(|v| v.is_finite());
        let needs_exif = non_empty(self.gps_latitude.as_deref()).is_some()
            || non_empty(self.gps_longitude.as_deref()).is_some()
            || altitude.is_some()
            || non_empty(self.lens.as_deref()).is_some()
            || self.iso.is_some()
            || non_empty(self.shutter.as_deref()).is_some()
            || self.aperture.filter(|v| v.is_finite()).is_some()
            || non_empty(self.datetime.as_deref()).is_some();
        let exif = ns_prefixes(desc, &mut tag, EXIF_NS, "exif", needs_exif);

        let rating = self.rating.clamp(-1, 5).to_string();
        apply_scalar(desc, text, &mut tag, &mut edits, XMP_NS, "Rating", &xmp, Some(&rating));
        apply_scalar(desc, text, &mut tag, &mut edits, XMP_NS, "Label", &xmp, non_empty(Some(self.label.as_str())));
        apply_scalar(
            desc,
            text,
            &mut tag,
            &mut edits,
            PHOTOSHOP_NS,
            "City",
            &photoshop,
            non_empty(self.city.as_deref()),
        );
        apply_scalar(
            desc,
            text,
            &mut tag,
            &mut edits,
            PHOTOSHOP_NS,
            "Country",
            &photoshop,
            non_empty(self.country.as_deref()),
        );
        apply_scalar(
            desc,
            text,
            &mut tag,
            &mut edits,
            EXIF_NS,
            "GPSLatitude",
            &exif,
            non_empty(self.gps_latitude.as_deref()),
        );
        apply_scalar(
            desc,
            text,
            &mut tag,
            &mut edits,
            EXIF_NS,
            "GPSLongitude",
            &exif,
            non_empty(self.gps_longitude.as_deref()),
        );
        let altitude_value = altitude.map(|v| v.abs().to_string());
        apply_scalar(
            desc,
            text,
            &mut tag,
            &mut edits,
            EXIF_NS,
            "GPSAltitude",
            &exif,
            altitude_value.as_deref(),
        );
        let altitude_ref = altitude.map(|v| if v < 0.0 { "1".to_string() } else { "0".to_string() });
        apply_scalar(
            desc,
            text,
            &mut tag,
            &mut edits,
            EXIF_NS,
            "GPSAltitudeRef",
            &exif,
            altitude_ref.as_deref(),
        );
        apply_scalar(desc, text, &mut tag, &mut edits, TIFF_NS, "Make", &tiff, non_empty(self.make.as_deref()));
        apply_scalar(desc, text, &mut tag, &mut edits, TIFF_NS, "Model", &tiff, non_empty(self.model.as_deref()));
        apply_scalar(desc, text, &mut tag, &mut edits, EXIF_NS, "LensModel", &exif, non_empty(self.lens.as_deref()));
        let shutter = non_empty(self.shutter.as_deref());
        apply_scalar(desc, text, &mut tag, &mut edits, EXIF_NS, "ExposureTime", &exif, shutter);
        let aperture = self.aperture.filter(|v| v.is_finite()).map(|v| v.to_string());
        apply_scalar(desc, text, &mut tag, &mut edits, EXIF_NS, "FNumber", &exif, aperture.as_deref());
        let datetime = non_empty(self.datetime.as_deref());
        apply_scalar(desc, text, &mut tag, &mut edits, EXIF_NS, "DateTimeOriginal", &exif, datetime);

        let dc_prefix = dc.first().map(String::as_str).unwrap_or("dc");
        let mut new_children = String::new();
        let subject = (!self.keywords.is_empty()).then(|| bag(rdf_prefix, &self.keywords));
        if let Some(added) = merge_element(desc, text, &mut edits, DC_NS, "subject", subject.as_deref(), dc_prefix) {
            new_children.push_str(&added);
        }
        let creator = (!self.creator.is_empty()).then(|| seq(rdf_prefix, &self.creator));
        if let Some(added) = merge_element(desc, text, &mut edits, DC_NS, "creator", creator.as_deref(), dc_prefix) {
            new_children.push_str(&added);
        }
        let copyright = non_empty(self.copyright.as_deref()).map(|v| alt(rdf_prefix, v));
        if let Some(added) = merge_element(desc, text, &mut edits, DC_NS, "rights", copyright.as_deref(), dc_prefix) {
            new_children.push_str(&added);
        }
        let description = non_empty(self.description.as_deref()).map(|v| alt(rdf_prefix, v));
        if let Some(added) =
            merge_element(desc, text, &mut edits, DC_NS, "description", description.as_deref(), dc_prefix)
        {
            new_children.push_str(&added);
        }
        let iso_items = self.iso.map(|v| vec![v.to_string()]);
        let iso = iso_items.as_ref().map(|v| seq(rdf_prefix, v));
        if let Some(added) = merge_element(
            desc,
            text,
            &mut edits,
            EXIF_NS,
            "ISOSpeedRatings",
            iso.as_deref(),
            exif.first().map(String::as_str).unwrap_or("exif"),
        ) {
            new_children.push_str(&added);
        }

        if !new_children.is_empty() {
            if tag.trim_end().ends_with("/>") {
                if let Some(stem) = tag.trim_end().strip_suffix("/>") {
                    tag = format!("{stem}>{new_children}\n  </{close_name}>");
                }
            } else {
                let insert_at = desc.range().end.saturating_sub(close_name.len() + 3);
                edits.push(Edit { start: insert_at, end: insert_at, text: format!("{new_children}\n  ") });
            }
        }

        edits.push(Edit { start, end: tag_close + 1, text: tag });
        let mut out = text.to_string();
        edits.sort_by_key(|e| std::cmp::Reverse(e.start));
        for edit in edits {
            out.replace_range(edit.start..edit.end, &edit.text);
        }
        Ok(out)
    }
}

struct Edit {
    start: usize,
    end: usize,
    text: String,
}

fn parse_rating(value: Option<&str>) -> i8 {
    value.and_then(|v| v.trim().parse::<i8>().ok()).unwrap_or(0).clamp(-1, 5)
}

fn find_element<'a, 'i>(doc: &'a Document<'i>, ns: &str, local: &str) -> Option<Node<'a, 'i>> {
    doc.descendants().find(|n| n.has_tag_name((ns, local)))
}

/// First element text or attribute for a property, trimmed.
fn find_property(doc: &Document<'_>, ns: &str, local: &str) -> Option<String> {
    if let Some(node) = find_element(doc, ns, local) {
        if let Some(text) = node.text().map(str::trim).filter(|s| !s.is_empty()) {
            return Some(text.to_string());
        }
    }
    doc.descendants()
        .flat_map(|node| node.attributes())
        .find(|attr| attr.namespace() == Some(ns) && attr.name() == local)
        .map(|attr| attr.value().trim().to_string())
        .filter(|value| !value.is_empty())
}

fn lang_alt(node: Node<'_, '_>) -> Option<String> {
    let items: Vec<Node<'_, '_>> =
        node.descendants().filter(|n| n.has_tag_name((RDF_NS, "li"))).collect();
    items
        .iter()
        .find(|n| n.attribute((XML_NS, "lang")) == Some("x-default"))
        .or_else(|| items.first())
        .and_then(|n| n.text())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
}

fn list_items(node: Node<'_, '_>) -> Vec<String> {
    let items: Vec<String> = node
        .descendants()
        .filter(|n| n.has_tag_name((RDF_NS, "li")))
        .filter_map(|n| n.text())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    match (items.is_empty(), node.text().map(str::trim)) {
        (true, Some(text)) if !text.is_empty() => vec![text.to_string()],
        _ => items,
    }
}

fn non_empty(value: Option<&str>) -> Option<&str> {
    value.filter(|v| !v.trim().is_empty())
}

fn push_attr(out: &mut String, prefix: &str, local: &str, value: &str) {
    out.push_str(&format!("\n    {prefix}:{local}=\"{}\"", escape_attr(value)));
}

fn seq(rdf_prefix: &str, items: &[String]) -> String {
    let mut out = format!("<{rdf_prefix}:Seq>");
    for item in items {
        out.push_str(&format!("<{rdf_prefix}:li>{}</{rdf_prefix}:li>", escape_text(item)));
    }
    out.push_str(&format!("</{rdf_prefix}:Seq>"));
    out
}

fn alt(rdf_prefix: &str, text: &str) -> String {
    format!(
        "<{rdf_prefix}:Alt><{rdf_prefix}:li xml:lang=\"x-default\">{}</{rdf_prefix}:li></{rdf_prefix}:Alt>",
        escape_text(text)
    )
}

/// In-scope prefix for `uri`, adding `fallback` to the start tag when needed.
fn ns_prefixes(desc: Node<'_, '_>, tag: &mut String, uri: &str, fallback: &str, wanted: bool) -> Vec<String> {
    if let Some(prefix) = desc.lookup_prefix(uri) {
        return vec![prefix.to_string()];
    }
    if !wanted {
        return Vec::new();
    }
    ensure_xmlns(tag, fallback, uri);
    vec![fallback.to_string()]
}

fn apply_scalar(
    desc: Node<'_, '_>,
    text: &str,
    tag: &mut String,
    edits: &mut Vec<Edit>,
    ns: &str,
    local: &str,
    prefixes: &[String],
    value: Option<&str>,
) {
    match desc.descendants().find(|n| n.has_tag_name((ns, local))) {
        Some(node) => {
            let name = qname(text, node.range().start);
            let replacement = match value {
                Some(value) => format!("<{name}>{}</{name}>", escape_text(value)),
                None => String::new(),
            };
            edits.push(Edit { start: node.range().start, end: node.range().end, text: replacement });
        }
        None => set_attribute(tag, prefixes, local, value),
    }
}

/// Replaces an element, removes it, or returns the text to add when absent.
fn merge_element(
    desc: Node<'_, '_>,
    text: &str,
    edits: &mut Vec<Edit>,
    ns: &str,
    local: &str,
    inner: Option<&str>,
    prefix: &str,
) -> Option<String> {
    match desc.descendants().find(|n| n.has_tag_name((ns, local))) {
        Some(node) => {
            let name = qname(text, node.range().start);
            let replacement = match inner {
                Some(inner) => format!("<{name}>{inner}</{name}>"),
                None => String::new(),
            };
            edits.push(Edit { start: node.range().start, end: node.range().end, text: replacement });
            None
        }
        None => inner.map(|inner| format!("\n   <{prefix}:{local}>{inner}</{prefix}:{local}>")),
    }
}

/// Prefers the description that already holds an XMP property.
fn pick_description<'a, 'i>(doc: &'a Document<'i>) -> Option<Node<'a, 'i>> {
    let mut first = None;
    for node in doc.descendants().filter(|n| n.has_tag_name((RDF_NS, "Description"))) {
        if first.is_none() {
            first = Some(node);
        }
        let owns = node.attributes().any(|a| a.namespace() == Some(XMP_NS))
            || node.descendants().any(|n| {
                matches!(
                    n.tag_name().namespace(),
                    Some(XMP_NS) | Some(DC_NS) | Some(PHOTOSHOP_NS) | Some(EXIF_NS) | Some(TIFF_NS)
                )
            });
        if owns {
            return Some(node);
        }
    }
    first
}

/// Byte index of the `>` closing the tag that starts at `start`.
fn tag_end(text: &str, start: usize) -> Option<usize> {
    let bytes = text.as_bytes();
    let mut quote = None;
    let mut i = start;
    while i < bytes.len() {
        let b = bytes[i];
        match quote {
            Some(q) if b == q => quote = None,
            Some(_) => {}
            None => match b {
                b'"' | b'\'' => quote = Some(b),
                b'>' => return Some(i),
                _ => {}
            },
        }
        i += 1;
    }
    None
}

/// `ns:local` name of the element starting at `start`.
fn qname(text: &str, start: usize) -> String {
    let rest = text[start..].trim_start_matches('<');
    rest.chars().take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ':')).collect()
}

fn prefix_for<'a>(text: &'a str, uri: &str) -> Option<&'a str> {
    prefixes_for(text, uri).into_iter().next()
}

/// Every prefix bound to `uri` anywhere in the packet.
fn prefixes_for<'a>(text: &'a str, uri: &str) -> Vec<&'a str> {
    let mut found = Vec::new();
    let mut from = 0;
    while let Some(rel) = text[from..].find("xmlns:") {
        let start = from + rel + 6;
        let rest = &text[start..];
        let end = rest.find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))).unwrap_or(rest.len());
        let prefix = &rest[..end];
        let tail = rest[end..].trim_start();
        if let Some(tail) = tail.strip_prefix('=') {
            let tail = tail.trim_start();
            if let Some(quote) = tail.chars().next().filter(|c| matches!(c, '"' | '\'')) {
                let inner = &tail[1..];
                if let Some(close) = inner.find(quote) {
                    if &inner[..close] == uri {
                        found.push(prefix);
                    }
                }
            }
        }
        from = start;
    }
    found
}

/// Adds `xmlns:prefix="uri"` to a start tag when the packet lacks it.
fn ensure_xmlns(tag: &mut String, prefix: &str, uri: &str) {
    if tag.contains(&format!("xmlns:{prefix}=")) {
        return;
    }
    if let Some(at) = insert_point(tag) {
        tag.insert_str(at, &format!(" xmlns:{prefix}=\"{uri}\""));
    }
}

/// Where a new attribute can go: before `>` or before `/>`.
fn insert_point(tag: &str) -> Option<usize> {
    let close = tag.rfind('>')?;
    Some(if tag[..close].trim_end().ends_with('/') { close - 1 } else { close })
}

/// Sets or removes `prefix:local="…"` in a start tag.
fn set_attribute(tag: &mut String, prefixes: &[String], local: &str, value: Option<&str>) {
    for prefix in prefixes {
        let needle = format!("{prefix}:{local}");
        let mut from = 0;
        while let Some(rel) = tag[from..].find(&needle) {
            let at = from + rel;
            let boundary = tag[..at].chars().next_back().is_some_and(|c| c.is_whitespace() || c == '<');
            let tail = &tag[at + needle.len()..];
            let trimmed = tail.trim_start();
            if boundary {
                if let Some(after_eq) = trimmed.strip_prefix('=') {
                    let eq = at + needle.len() + (tail.len() - trimmed.len());
                    let after_eq_trimmed = after_eq.trim_start();
                    let value_start = eq + 1 + (after_eq.len() - after_eq_trimmed.len());
                    let Some(quote) = after_eq_trimmed.chars().next().filter(|c| matches!(c, '"' | '\'')) else { break };
                    let inner = &after_eq_trimmed[1..];
                    let Some(close) = inner.find(quote) else { break };
                    let value_end = value_start + 1 + close;
                    match value {
                        Some(v) => {
                            tag.replace_range(value_start + 1..value_end, &escape_attr(v));
                            return;
                        }
                        None => {
                            let name_start = tag[..at].trim_end().len();
                            tag.replace_range(name_start..value_end + 1, "");
                            return;
                        }
                    }
                }
            }
            from = at + needle.len();
        }
    }
    if let (Some(value), Some(at)) = (value, insert_point(tag)) {
        let prefix = prefixes.first().map(String::as_str).unwrap_or("xmp");
        tag.insert_str(at, &format!(" {prefix}:{local}=\"{}\"", escape_attr(value)));
    }
}

fn bag(rdf_prefix: &str, keywords: &[String]) -> String {
    let mut out = format!("<{rdf_prefix}:Bag>");
    for keyword in keywords {
        out.push_str(&format!("<{rdf_prefix}:li>{}</{rdf_prefix}:li>", escape_text(keyword)));
    }
    out.push_str(&format!("</{rdf_prefix}:Bag>"));
    out
}

fn escape_text(value: &str) -> String {
    value.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

fn escape_attr(value: &str) -> String {
    escape_text(value).replace('"', "&quot;")
}

/// Sidecars never go into a Capture One catalog or session bundle.
fn in_c1_bundle(path: &Path) -> bool {
    path.components().any(|c| match c {
        Component::Normal(name) => {
            let name = name.to_string_lossy().to_ascii_lowercase();
            name.ends_with(".cocatalogdb") || name.ends_with(".cosessiondb") || name.ends_with(".cocatalog")
        }
        _ => false,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static COUNTER: AtomicUsize = AtomicUsize::new(0);

    struct TempFolder(PathBuf);

    impl TempFolder {
        fn new() -> Self {
            let id = COUNTER.fetch_add(1, Ordering::SeqCst);
            let path = std::env::temp_dir().join(format!("portrait-xmp-test-{}-{id}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            TempFolder(path)
        }

        fn image(&self, name: &str) -> PathBuf {
            let path = self.0.join(name);
            fs::write(&path, b"pixels").unwrap();
            path
        }
    }

    impl Drop for TempFolder {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn sample() -> XmpMeta {
        XmpMeta {
            rating: 4,
            label: "Red".into(),
            keywords: vec!["wedding".into(), "hero & co".into()],
            creator: vec!["Ada Lovelace".into(), "Grace Hopper".into()],
            copyright: Some("© 2026 Ada Lovelace".into()),
            description: Some("A <portrait> & test".into()),
            city: Some("Shanghai".into()),
            country: Some("China".into()),
            gps_latitude: Some("31,13.8240N".into()),
            gps_longitude: Some("121,28.4400E".into()),
            gps_altitude: Some(-12.5),
            make: Some("Canon".into()),
            model: Some("EOS R5".into()),
            lens: Some("RF 85mm F1.2 L USM".into()),
            iso: Some(100),
            shutter: Some("1/250".into()),
            aperture: Some(1.2),
            datetime: Some("2026-11-14T03:40:00".into()),
        }
    }

    fn minimal() -> XmpMeta {
        XmpMeta { rating: -1, label: String::new(), keywords: vec!["solo".into()], ..Default::default() }
    }

    #[test]
    fn round_trips_through_disk() {
        let temp = TempFolder::new();
        let image = temp.image("IMG_0001.CR3");

        let written = sample().save(&image).unwrap();
        assert_eq!(written, temp.0.join("IMG_0001.xmp"));
        assert_eq!(XmpMeta::load(&image).unwrap(), Some(sample()));
        assert!(!temp.0.join("IMG_0001.xmp.tmp").exists());
    }

    #[test]
    fn embedded_packet_round_trips_all_fields() {
        assert_eq!(XmpMeta::parse(&sample().packet()).unwrap(), sample());
    }

    #[test]
    fn rereading_and_rewriting_keeps_the_packet_valid() {
        let temp = TempFolder::new();
        let image = temp.image("a.NEF");
        sample().save(&image).unwrap();
        let second = minimal();
        second.save(&image).unwrap();
        assert_eq!(XmpMeta::load(&image).unwrap(), Some(second));
    }

    #[test]
    fn preserves_unknown_properties() {
        let temp = TempFolder::new();
        let image = temp.image("shot.CR3");
        let original = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
    xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
    xmp:Rating="1"
    crs:Exposure2012="+0.75"
    photoshop:City="Shanghai">
   <crs:ToneCurvePV2012><rdf:Seq><rdf:li>0, 0</rdf:li></rdf:Seq></crs:ToneCurvePV2012>
   <xmp:CreatorTool>Capture One</xmp:CreatorTool>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;
        fs::write(XmpMeta::sidecar_path(&image), original).unwrap();

        sample().save(&image).unwrap();
        let text = fs::read_to_string(XmpMeta::sidecar_path(&image)).unwrap();
        assert!(text.contains("crs:Exposure2012=\"+0.75\""), "{text}");
        assert!(text.contains("photoshop:City=\"Shanghai\""), "{text}");
        assert!(text.contains("<crs:ToneCurvePV2012>"), "{text}");
        assert!(text.contains("<xmp:CreatorTool>Capture One</xmp:CreatorTool>"), "{text}");
        assert!(text.contains("xmp:Rating=\"4\""), "{text}");
        assert!(text.contains("xmp:Label=\"Red\""), "{text}");
        assert_eq!(XmpMeta::load(&image).unwrap(), Some(sample()));
    }

    #[test]
    fn missing_file_is_none() {
        let temp = TempFolder::new();
        assert_eq!(XmpMeta::load(&temp.0.join("absent.jpg")).unwrap(), None);
    }

    #[test]
    fn empty_file_reads_as_default() {
        let temp = TempFolder::new();
        let image = temp.image("blank.jpg");
        fs::write(XmpMeta::sidecar_path(&image), "").unwrap();
        assert_eq!(XmpMeta::load(&image).unwrap(), Some(XmpMeta::default()));
        sample().save(&image).unwrap();
        assert_eq!(XmpMeta::load(&image).unwrap(), Some(sample()));
    }

    #[test]
    fn damaged_xml_errors_without_touching_the_file() {
        let temp = TempFolder::new();
        let image = temp.image("broken.jpg");
        let path = XmpMeta::sidecar_path(&image);
        fs::write(&path, "<x:xmpmeta><rdf:RDF>").unwrap();

        assert!(XmpMeta::load(&image).is_err());
        assert!(sample().save(&image).is_err());
        assert_eq!(fs::read_to_string(&path).unwrap(), "<x:xmpmeta><rdf:RDF>");
        assert!(!temp.0.join("broken.xmp.tmp").exists());
    }

    #[test]
    fn reads_any_prefix_and_both_property_forms() {
        let text = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description xmlns:meta="http://ns.adobe.com/xap/1.0/"
      xmlns:d="http://purl.org/dc/elements/1.1/" rdf:about="">
   <meta:Rating>3</meta:Rating>
   <meta:Label>Green</meta:Label>
   <d:subject><rdf:Seq><rdf:li>one</rdf:li><rdf:li>two</rdf:li></rdf:Seq></d:subject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"#;
        assert_eq!(
            XmpMeta::parse(text).unwrap(),
            XmpMeta {
                rating: 3,
                label: "Green".into(),
                keywords: vec!["one".into(), "two".into()],
                ..Default::default()
            }
        );
    }

    #[test]
    fn missing_fields_get_defaults() {
        let text = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about=""/></rdf:RDF></x:xmpmeta>"#;
        assert_eq!(XmpMeta::parse(text).unwrap(), XmpMeta::default());
    }

    #[test]
    fn legacy_three_field_packet_leaves_new_fields_none() {
        let text = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:dc="http://purl.org/dc/elements/1.1/"
      xmp:Rating="3" xmp:Label="Green">
   <dc:subject><rdf:Bag><rdf:li>one</rdf:li></rdf:Bag></dc:subject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"#;
        let meta = XmpMeta::parse(text).unwrap();
        assert_eq!(meta.rating, 3);
        assert_eq!(meta.label, "Green");
        assert_eq!(meta.keywords, vec!["one".to_string()]);
        assert!(meta.creator.is_empty());
        assert!(meta.copyright.is_none());
        assert!(meta.description.is_none());
        assert!(meta.city.is_none());
        assert!(meta.country.is_none());
        assert!(meta.gps_latitude.is_none());
        assert!(meta.gps_longitude.is_none());
        assert!(meta.gps_altitude.is_none());
        assert!(meta.make.is_none());
        assert!(meta.model.is_none());
        assert!(meta.lens.is_none());
        assert!(meta.iso.is_none());
        assert!(meta.shutter.is_none());
        assert!(meta.aperture.is_none());
        assert!(meta.datetime.is_none());
    }

    #[test]
    fn real_image_copy_round_trips_every_field_and_keeps_unknowns() {
        let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/iptc-sample.jpg");
        if !source.is_file() {
            eprintln!("skipping: testdata/iptc-sample.jpg is absent");
            return;
        }
        let temp = TempFolder::new();
        let image = temp.0.join("iptc-sample.jpg");
        fs::copy(&source, &image).unwrap();
        let path = XmpMeta::sidecar_path(&image);
        fs::write(
            &path,
            r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
      crs:Exposure2012="+0.75">
   <crs:ToneCurvePV2012><rdf:Seq><rdf:li>0, 0</rdf:li></rdf:Seq></crs:ToneCurvePV2012>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>"#,
        )
        .unwrap();

        sample().save(&image).unwrap();
        assert_eq!(XmpMeta::load(&image).unwrap(), Some(sample()));

        let text = fs::read_to_string(&path).unwrap();
        assert!(text.contains("crs:Exposure2012=\"+0.75\""), "{text}");
        assert!(text.contains("<crs:ToneCurvePV2012>"), "{text}");
        for declaration in [
            "xmlns:dc=\"http://purl.org/dc/elements/1.1/\"",
            "xmlns:photoshop=\"http://ns.adobe.com/photoshop/1.0/\"",
            "xmlns:exif=\"http://ns.adobe.com/exif/1.0/\"",
            "xmlns:tiff=\"http://ns.adobe.com/tiff/1.0/\"",
        ] {
            assert!(text.contains(declaration), "missing {declaration}: {text}");
        }
    }

    #[test]
    fn invalid_utf8_sidecar_errors_without_touching_the_file() {
        let temp = TempFolder::new();
        let image = temp.image("bad.jpg");
        let path = XmpMeta::sidecar_path(&image);
        let bytes = [0xff, 0xfe, 0x00, 0x3c];
        fs::write(&path, bytes).unwrap();

        assert!(XmpMeta::load(&image).is_err());
        assert!(sample().save(&image).is_err());
        assert_eq!(fs::read(&path).unwrap(), bytes);
        assert!(!temp.0.join("bad.xmp.tmp").exists());
    }

    #[test]
    fn serde_defaults_fields_missing_from_json() {
        let meta: XmpMeta = serde_json::from_str(r#"{"rating":2,"label":"Blue","keywords":["a"]}"#).unwrap();
        assert_eq!(
            meta,
            XmpMeta { rating: 2, label: "Blue".into(), keywords: vec!["a".into()], ..Default::default() }
        );
    }

    #[test]
    fn refuses_c1_bundles() {
        let temp = TempFolder::new();
        let bundle = temp.0.join("Shoot.cocatalogdb");
        fs::create_dir_all(&bundle).unwrap();
        let image = bundle.join("a.CR3");
        fs::write(&image, b"pixels").unwrap();
        assert!(sample().save(&image).is_err());
        assert!(!bundle.join("a.xmp").exists());
    }
}
