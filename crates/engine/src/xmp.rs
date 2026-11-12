//! XMP sidecars: rating, colour label and keywords, nothing else.
//!
//! The sidecar is `<image stem>.xmp` next to the image. It is written only when
//! a caller asks for it, atomically (`.tmp` then rename), and every property
//! Lumen does not own is copied through byte for byte, so a Capture One or
//! Lightroom packet keeps its `crs:`, `photoshop:` and `tiff:` data.

use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use roxmltree::{Document, Node};

const XMP_NS: &str = "http://ns.adobe.com/xap/1.0/";
const DC_NS: &str = "http://purl.org/dc/elements/1.1/";
const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";
const X_NS: &str = "adobe:ns:meta/";

/// The three metadata fields Lumen mirrors to XMP.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct XmpMeta {
    /// -1 (rejected) ..= 5 stars.
    pub rating: i8,
    /// Colour label name, e.g. `"Red"`; empty means no label.
    pub label: String,
    pub keywords: Vec<String>,
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
        let mut meta = Self::default();
        for node in doc.descendants() {
            let ns = node.tag_name().namespace().unwrap_or("");
            match (ns, node.tag_name().name()) {
                (XMP_NS, "Rating") => meta.rating = parse_rating(node.text()),
                (XMP_NS, "Label") => meta.label = node.text().unwrap_or("").trim().to_string(),
                (DC_NS, "subject") => meta.keywords = subject_items(node),
                _ => {}
            }
            for attr in node.attributes() {
                match (attr.namespace().unwrap_or(""), attr.name()) {
                    (XMP_NS, "Rating") => meta.rating = parse_rating(Some(attr.value())),
                    (XMP_NS, "Label") => meta.label = attr.value().trim().to_string(),
                    _ => {}
                }
            }
        }
        Ok(meta)
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
        let mut description = String::from(
            "  <rdf:Description rdf:about=\"\"\n\
             \x20   xmlns:xmp=\"http://ns.adobe.com/xap/1.0/\"\n\
             \x20   xmlns:dc=\"http://purl.org/dc/elements/1.1/\"",
        );
        if always_rating || self.rating != 0 {
            description.push_str(&format!("\n    xmp:Rating=\"{}\"", self.rating.clamp(-1, 5)));
        }
        if !self.label.is_empty() {
            description.push_str(&format!("\n    xmp:Label=\"{}\"", escape_attr(&self.label)));
        }
        if self.keywords.is_empty() {
            description.push_str("/>\n");
        } else {
            description.push_str(">\n   <dc:subject>\n    ");
            description.push_str(&bag("rdf", &self.keywords));
            description.push_str("\n   </dc:subject>\n  </rdf:Description>\n");
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

        let xmp_prefixes = prefixes_for(text, XMP_NS);
        let dc_prefixes = prefixes_for(text, DC_NS);
        if xmp_prefixes.is_empty() {
            ensure_xmlns(&mut tag, "xmp", XMP_NS);
        }
        if dc_prefixes.is_empty() && !self.keywords.is_empty() {
            ensure_xmlns(&mut tag, "dc", DC_NS);
        }
        let xmp = if xmp_prefixes.is_empty() { vec!["xmp"] } else { xmp_prefixes.clone() };
        let dc = if dc_prefixes.is_empty() { vec!["dc"] } else { dc_prefixes.clone() };
        let rdf_prefix = prefix_for(text, RDF_NS).unwrap_or("rdf");
        let close_name = qname(text, start);
        let mut edits = Vec::new();

        match desc.descendants().find(|n| n.has_tag_name((XMP_NS, "Rating"))) {
            Some(node) => {
                let name = qname(text, node.range().start);
                let value = self.rating.clamp(-1, 5);
                edits.push(Edit { start: node.range().start, end: node.range().end, text: format!("<{name}>{value}</{name}>") });
            }
            None => set_attribute(&mut tag, &xmp, "Rating", Some(&self.rating.clamp(-1, 5).to_string())),
        }
        let label = (!self.label.is_empty()).then_some(self.label.as_str());
        match desc.descendants().find(|n| n.has_tag_name((XMP_NS, "Label"))) {
            Some(node) => {
                let name = qname(text, node.range().start);
                let replacement = match label {
                    Some(v) => format!("<{name}>{}</{name}>", escape_text(v)),
                    None => String::new(),
                };
                edits.push(Edit { start: node.range().start, end: node.range().end, text: replacement });
            }
            None => set_attribute(&mut tag, &xmp, "Label", label),
        }

        let subject = desc.descendants().find(|n| n.has_tag_name((DC_NS, "subject")));
        let replacement = (!self.keywords.is_empty()).then(|| bag(rdf_prefix, &self.keywords));
        match (subject, replacement) {
            (Some(node), Some(bag)) => {
                let name = qname(text, node.range().start);
                let dc_name = format!("<{name}>{bag}</{name}>");
                edits.push(Edit { start: node.range().start, end: node.range().end, text: dc_name });
            }
            (Some(node), None) => {
                edits.push(Edit { start: node.range().start, end: node.range().end, text: String::new() });
            }
            (None, Some(bag)) => {
                let element = format!("<{p}:subject>{bag}</{p}:subject>", p = dc[0]);
                if tag.trim_end().ends_with("/>") {
                    if let Some(stem) = tag.trim_end().strip_suffix("/>") {
                        tag = format!("{stem}>\n   {element}\n  </{close_name}>");
                    }
                } else {
                    let insert_at = desc.range().end.saturating_sub(close_name.len() + 3);
                    edits.push(Edit { start: insert_at, end: insert_at, text: format!("\n   {element}\n  ") });
                }
            }
            (None, None) => {}
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

fn subject_items(node: Node<'_, '_>) -> Vec<String> {
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

/// Prefers the description that already holds an XMP property.
fn pick_description<'a, 'i>(doc: &'a Document<'i>) -> Option<Node<'a, 'i>> {
    let mut first = None;
    for node in doc.descendants().filter(|n| n.has_tag_name((RDF_NS, "Description"))) {
        if first.is_none() {
            first = Some(node);
        }
        let owns = node.attributes().any(|a| a.namespace() == Some(XMP_NS))
            || node
                .descendants()
                .any(|n| matches!(n.tag_name().namespace(), Some(XMP_NS) | Some(DC_NS)));
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
fn set_attribute(tag: &mut String, prefixes: &[&str], local: &str, value: Option<&str>) {
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
        let prefix = prefixes.first().copied().unwrap_or("xmp");
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
            let path = std::env::temp_dir().join(format!("lumen-xmp-test-{}-{id}", std::process::id()));
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
        XmpMeta { rating: 4, label: "Red".into(), keywords: vec!["wedding".into(), "hero & co".into()] }
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
    fn rereading_and_rewriting_keeps_the_packet_valid() {
        let temp = TempFolder::new();
        let image = temp.image("a.NEF");
        sample().save(&image).unwrap();
        let second = XmpMeta { rating: -1, label: String::new(), keywords: vec!["solo".into()] };
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
            XmpMeta { rating: 3, label: "Green".into(), keywords: vec!["one".into(), "two".into()] }
        );
    }

    #[test]
    fn missing_fields_get_defaults() {
        let text = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description rdf:about=""/></rdf:RDF></x:xmpmeta>"#;
        assert_eq!(XmpMeta::parse(text).unwrap(), XmpMeta::default());
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
