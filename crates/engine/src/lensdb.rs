//! Lens database from Capture One's native `lensDatabase.cdx`.
//! The file is plain XML obfuscated with a XOR 0x81 byte stream.
//! Spline radii are sensor-plane micrometres measured from the optical axis.
//! The table's maximum radius equals the half-diagonal of the body the profile
//! was shot on, so a consumer must rescale
//! `r = r_table * (capture_half_diagonal_um / r_table_max)` before sampling.

use anyhow::{anyhow, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Zoom / prime classification as reported by the `type` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LensKind {
    Zoom,
    Prime,
    Other,
}

impl LensKind {
    fn from_attr(value: &str) -> Self {
        match value {
            "Zoom" => LensKind::Zoom,
            "Prime" => LensKind::Prime,
            _ => LensKind::Other,
        }
    }
}

/// One geometric-distortion spline: `(radius, shift)` pairs.
#[derive(Debug, Clone)]
pub struct DistortionEntry {
    pub focal_length: f32,
    pub focus_distance: String,
    pub shift: Vec<(f32, f32)>,
}

/// One light-falloff spline: `(radius, ev_loss)` pairs; 0 at the centre,
/// negative toward the corner.
#[derive(Debug, Clone)]
pub struct FalloffEntry {
    pub focal_length: f32,
    pub aperture: f32,
    pub focus_distance: String,
    pub falloff: Vec<(f32, f32)>,
}

/// One chromatic-aberration curve, red and blue lateral shifts.
#[derive(Debug, Clone)]
pub struct ChromaEntry {
    pub focal_length: f32,
    pub aperture: f32,
    pub red_shift: Vec<(f32, f32)>,
    pub blue_shift: Vec<(f32, f32)>,
}

/// Everything we keep from one `<lens>` element.
#[derive(Debug, Clone)]
pub struct LensProfile {
    pub id: String,
    pub full_name: String,
    pub short_name: String,
    pub brand: String,
    pub mount: String,
    pub kind: LensKind,
    pub focal_min: Option<f32>,
    pub focal_max: Option<f32>,
    pub aperture_min: Option<f32>,
    pub aperture_max: Option<f32>,
    pub detect_id: Option<String>,
    /// geometric distortion: (radius_um, shift_um) per focal length
    pub distortion: Vec<DistortionEntry>,
    /// light falloff: (radius_um, ev_loss) per focal/aperture
    pub falloff: Vec<FalloffEntry>,
    /// chromatic aberration per focal/aperture
    pub chroma: Vec<ChromaEntry>,
}

impl LensProfile {
    /// Nearest distortion spline by focal length.
    pub fn distortion_at(&self, focal_length: f32) -> Option<&DistortionEntry> {
        self.distortion
            .iter()
            .min_by(|a, b| {
                let da = (a.focal_length - focal_length).abs();
                let db = (b.focal_length - focal_length).abs();
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Nearest light-falloff spline by focal length, then by aperture.
    pub fn falloff_at(&self, focal_length: f32, aperture: f32) -> Option<&FalloffEntry> {
        self.falloff
            .iter()
            .min_by(|a, b| {
                let ka = (a.focal_length - focal_length).abs() + (a.aperture - aperture).abs() * 10.0;
                let kb = (b.focal_length - focal_length).abs() + (b.aperture - aperture).abs() * 10.0;
                ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Nearest chromatic-aberration curve by focal length, then by aperture.
    pub fn chroma_at(&self, focal_length: f32, aperture: f32) -> Option<&ChromaEntry> {
        self.chroma
            .iter()
            .min_by(|a, b| {
                let ka = (a.focal_length - focal_length).abs() + (a.aperture - aperture).abs() * 10.0;
                let kb = (b.focal_length - focal_length).abs() + (b.aperture - aperture).abs() * 10.0;
                ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

/// Decode the obfuscated `.cdx` payload: every byte is XOR 0x81, then the two
/// leading `~` characters are dropped. Input that is not obfuscated is
/// returned unchanged; a truncated tail is tolerated.
pub fn decode_cdx(bytes: &[u8]) -> Result<String> {
    if bytes.is_empty() {
        return Err(anyhow!("empty lens database file"));
    }
    if bytes.len() >= 2 && bytes[0] == 0xFF && bytes[1] == 0xFF {
        // '~' (0x7E) XOR 0x81 == 0xFF, so an obfuscated file starts with 0xFF.
        let decoded: Vec<u8> = bytes.iter().map(|b| b ^ 0x81).collect();
        let text = String::from_utf8_lossy(&decoded);
        Ok(text.trim_start_matches('~').to_string())
    } else {
        Ok(String::from_utf8_lossy(bytes).into_owned())
    }
}

/// Parse the decoded XML into lens profiles. Only the fields we need are kept.
pub fn parse_lenses(xml: &str) -> Vec<LensProfile> {
    let mut out: Vec<LensProfile> = Vec::new();
    let mut cur: Option<LensProfile> = None;
    let mut geom: Option<DistortionEntry> = None;
    let mut fall: Option<FalloffEntry> = None;
    let mut chroma: Option<ChromaEntry> = None;
    let mut capture: Option<Capture> = None;
    let mut buf = String::new();

    let mut rest = xml;
    while !rest.is_empty() {
        let Some(lt) = rest.find('<') else { break };
        if capture.is_some() {
            buf.push_str(&rest[..lt]);
        }
        rest = &rest[lt..];

        if rest.starts_with("<!--") {
            let end = rest.find("-->").map(|i| i + 3).unwrap_or(rest.len());
            rest = &rest[end..];
            continue;
        }
        if rest.starts_with("<?") {
            let end = rest.find("?>").map(|i| i + 2).unwrap_or(rest.len());
            rest = &rest[end..];
            continue;
        }
        if rest.starts_with("<![CDATA[") {
            let end = rest.find("]]>").map(|i| i + 3).unwrap_or(rest.len());
            rest = &rest[end..];
            continue;
        }

        // Locate the end of this tag, ignoring '>' inside attribute values.
        let (tag, after) = split_tag(rest);
        rest = after;

        let inner = tag
            .strip_prefix('<')
            .and_then(|t| t.strip_suffix('>'))
            .unwrap_or(tag);
        if let Some(close) = inner.strip_prefix('/') {
            let name = close.trim();
            match name {
                "lens" => {
                    if let Some(lens) = cur.take() {
                        out.push(lens);
                    }
                }
                "lensGeometricAberration" => {
                    if let (Some(lens), Some(entry)) = (cur.as_mut(), geom.take()) {
                        lens.distortion.push(entry);
                    }
                }
                "lensLightFalloff" => {
                    if let (Some(lens), Some(entry)) = (cur.as_mut(), fall.take()) {
                        lens.falloff.push(entry);
                    }
                }
                "chromAberCurve" => {
                    if let (Some(lens), Some(entry)) = (cur.as_mut(), chroma.take()) {
                        lens.chroma.push(entry);
                    }
                }
                _ => {}
            }
            if let Some(kind) = capture.take() {
                commit_text(kind, &buf, cur.as_mut(), geom.as_mut(), fall.as_mut(), chroma.as_mut());
                buf.clear();
            }
            continue;
        }

        let self_closing = inner.trim_end().ends_with('/');
        let body = inner.trim_end().trim_end_matches('/');
        let name = body.split_whitespace().next().unwrap_or("");
        let attrs = parse_attrs(body);

        match name {
            "lens" => {
                cur = Some(LensProfile {
                    id: attr(&attrs, "ID").unwrap_or_default(),
                    full_name: String::new(),
                    short_name: String::new(),
                    brand: String::new(),
                    mount: String::new(),
                    kind: attr(&attrs, "type").map(|v| LensKind::from_attr(&v)).unwrap_or(LensKind::Other),
                    focal_min: None,
                    focal_max: None,
                    aperture_min: None,
                    aperture_max: None,
                    detect_id: None,
                    distortion: Vec::new(),
                    falloff: Vec::new(),
                    chroma: Vec::new(),
                });
            }
            "fullName" | "shortName" | "brand" | "mount" | "note" | "fileDetectID" => {
                capture = Some(Capture::Text(text_field(name)));
                buf.clear();
            }
            "focalLength" => {
                if let Some(lens) = cur.as_mut() {
                    lens.focal_min = attr(&attrs, "min").and_then(|v| v.parse().ok());
                    lens.focal_max = attr(&attrs, "max").and_then(|v| v.parse().ok());
                }
            }
            "aperture" => {
                if let Some(lens) = cur.as_mut() {
                    lens.aperture_min = attr(&attrs, "min").and_then(|v| v.parse().ok());
                    lens.aperture_max = attr(&attrs, "max").and_then(|v| v.parse().ok());
                }
            }
            "lensGeometricAberration" => {
                geom = Some(DistortionEntry {
                    focal_length: attr(&attrs, "focalLength").and_then(|v| v.parse().ok()).unwrap_or(0.0),
                    focus_distance: attr(&attrs, "focusDistance").unwrap_or_default(),
                    shift: Vec::new(),
                });
            }
            "lensLightFalloff" => {
                fall = Some(FalloffEntry {
                    focal_length: attr(&attrs, "focalLength").and_then(|v| v.parse().ok()).unwrap_or(0.0),
                    aperture: attr(&attrs, "aperture").and_then(|v| v.parse().ok()).unwrap_or(0.0),
                    focus_distance: attr(&attrs, "focusDistance").unwrap_or_default(),
                    falloff: Vec::new(),
                });
            }
            "chromAberCurve" => {
                chroma = Some(ChromaEntry {
                    focal_length: attr(&attrs, "focalLength").and_then(|v| v.parse().ok()).unwrap_or(0.0),
                    aperture: attr(&attrs, "aperture").and_then(|v| v.parse().ok()).unwrap_or(0.0),
                    red_shift: Vec::new(),
                    blue_shift: Vec::new(),
                });
            }
            "shift" => {
                capture = Some(Capture::Shift);
                buf.clear();
            }
            "falloff" => {
                capture = Some(Capture::Falloff);
                buf.clear();
            }
            "redShift" => {
                capture = Some(Capture::RedShift);
                buf.clear();
            }
            "blueShift" => {
                capture = Some(Capture::BlueShift);
                buf.clear();
            }
            _ => {}
        }

        if self_closing {
            // Nothing to capture inside an empty element.
            if let Some(kind) = capture.take() {
                commit_text(kind, &buf, cur.as_mut(), geom.as_mut(), fall.as_mut(), chroma.as_mut());
                buf.clear();
            }
            // An empty `<lensLightFalloff/>`-style marker (e.g. reset="TRUE")
            // must not leak into the next lens.
            match name {
                "lensGeometricAberration" => geom = None,
                "lensLightFalloff" => fall = None,
                "chromAberCurve" => chroma = None,
                _ => {}
            }
            if name == "lens" {
                cur = None;
            }
        }
    }

    out
}

/// Read a `.cdx` file, decode it and parse the lenses out of it.
pub fn load_lens_database(path: &Path) -> Result<Vec<LensProfile>> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("reading lens database {}", path.display()))?;
    let xml = decode_cdx(&bytes)?;
    Ok(parse_lenses(&xml))
}

/// The installed Capture One lens database, loaded once on first use.
///
/// `LUMEN_LENS_DB` overrides the path so a machine without Capture One can
/// point at a copy; without either the empty database means no calibrated
/// correction is available and the generic formula stays in charge.
pub fn default_database() -> Option<&'static [LensProfile]> {
    static DB: OnceLock<Option<Vec<LensProfile>>> = OnceLock::new();
    DB.get_or_init(|| {
        let path = std::env::var_os("LUMEN_LENS_DB")
            .map(PathBuf::from)
            .or_else(|| Some(PathBuf::from(r"C:\Program Files\Capture One\Capture One\lensDatabase.cdx")))
            .filter(|p| p.is_file())?;
        load_lens_database(&path).ok().filter(|v| !v.is_empty())
    })
    .as_deref()
}

/// Look up a lens by the EXIF-reported make/model/focal length.
pub fn detect_lens<'a>(
    lenses: &'a [LensProfile],
    make: &str,
    model: &str,
    focal_mm: f32,
) -> Option<&'a LensProfile> {
    let make_key = normalize(make);
    let model_key = normalize(model);
    if model_key.is_empty() {
        return None;
    }

    let mut fallback: Option<&LensProfile> = None;
    for lens in lenses {
        let brand_key = normalize(&lens.brand);
        if !make_key.is_empty()
            && !brand_key.is_empty()
            && brand_key != make_key
            && !normalize(&lens.full_name).contains(&make_key)
            && !model_key.contains(&make_key)
        {
            continue;
        }
        let full = normalize(&lens.full_name);
        let short = normalize(&lens.short_name);
        let name_hit = full == model_key
            || short == model_key
            || full.contains(&model_key)
            || short.contains(&model_key)
            || model_key.contains(&short)
            || (lens.detect_id.as_deref().is_some_and(|d| normalize(d).contains(&model_key)));
        if !name_hit {
            continue;
        }
        let in_range = match (lens.focal_min, lens.focal_max) {
            (Some(min), Some(max)) => focal_mm >= min - 0.5 && focal_mm <= max + 0.5,
            _ => true,
        };
        if in_range {
            return Some(lens);
        }
        if fallback.is_none() {
            fallback = Some(lens);
        }
    }
    fallback
}

#[derive(Debug, Clone, Copy)]
enum Capture {
    Text(TextField),
    Shift,
    Falloff,
    RedShift,
    BlueShift,
}

#[derive(Debug, Clone, Copy)]
enum TextField {
    FullName,
    ShortName,
    Brand,
    Mount,
    Note,
    DetectId,
}

fn text_field(name: &str) -> TextField {
    match name {
        "fullName" => TextField::FullName,
        "shortName" => TextField::ShortName,
        "brand" => TextField::Brand,
        "mount" => TextField::Mount,
        "note" => TextField::Note,
        _ => TextField::DetectId,
    }
}

fn commit_text(
    kind: Capture,
    text: &str,
    cur: Option<&mut LensProfile>,
    geom: Option<&mut DistortionEntry>,
    fall: Option<&mut FalloffEntry>,
    chroma: Option<&mut ChromaEntry>,
) {
    let trimmed = text.trim();
    match kind {
        Capture::Text(name) => {
            if let Some(lens) = cur {
                match name {
                    TextField::FullName => lens.full_name = trimmed.to_string(),
                    TextField::ShortName => lens.short_name = trimmed.to_string(),
                    TextField::Brand => lens.brand = trimmed.to_string(),
                    TextField::Mount => lens.mount = trimmed.to_string(),
                    TextField::Note => {}
                    TextField::DetectId => lens.detect_id = Some(trimmed.to_string()),
                }
            }
        }
        Capture::Shift => {
            if let Some(entry) = geom {
                entry.shift = parse_pairs(trimmed);
            }
        }
        Capture::Falloff => {
            if let Some(entry) = fall {
                entry.falloff = parse_pairs(trimmed);
            }
        }
        Capture::RedShift => {
            if let Some(entry) = chroma {
                entry.red_shift = parse_pairs(trimmed);
            }
        }
        Capture::BlueShift => {
            if let Some(entry) = chroma {
                entry.blue_shift = parse_pairs(trimmed);
            }
        }
    }
}

/// Parse a whitespace separated list of numbers into pairs. A trailing odd
/// number is dropped; an empty string yields an empty vector.
fn parse_pairs(text: &str) -> Vec<(f32, f32)> {
    let values: Vec<f32> = text
        .split_whitespace()
        .filter_map(|v| v.parse::<f32>().ok())
        .collect();
    values
        .chunks_exact(2)
        .map(|c| (c[0], c[1]))
        .collect()
}

/// Split a tag into its text and the remainder of the document.
fn split_tag(rest: &str) -> (&str, &str) {
    let bytes = rest.as_bytes();
    let mut quote: Option<u8> = None;
    let mut i = 1;
    while i < bytes.len() {
        let c = bytes[i];
        match quote {
            Some(q) if c == q => quote = None,
            Some(_) => {}
            None if c == b'"' || c == b'\'' => quote = Some(c),
            None if c == b'>' => return (&rest[..=i], &rest[i + 1..]),
            None => {}
        }
        i += 1;
    }
    (rest, "")
}

fn parse_attrs(body: &str) -> Vec<(String, String)> {
    let mut attrs = Vec::new();
    let bytes = body.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        let name_start = i;
        while i < bytes.len() && bytes[i] != b'=' && !(bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if name_start == i {
            i += 1;
            continue;
        }
        let name = body[name_start..i].to_string();
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() || bytes[i] != b'=' {
            attrs.push((name, String::new()));
            continue;
        }
        i += 1;
        while i < bytes.len() && (bytes[i] as char).is_whitespace() {
            i += 1;
        }
        if i >= bytes.len() {
            break;
        }
        let value = if bytes[i] == b'"' || bytes[i] == b'\'' {
            let quote = bytes[i];
            i += 1;
            let start = i;
            while i < bytes.len() && bytes[i] != quote {
                i += 1;
            }
            let v = body[start..i].to_string();
            i += 1;
            v
        } else {
            let start = i;
            while i < bytes.len() && !(bytes[i] as char).is_whitespace() {
                i += 1;
            }
            body[start..i].to_string()
        };
        attrs.push((name, value));
    }
    attrs
}

fn attr(attrs: &[(String, String)], name: &str) -> Option<String> {
    attrs
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.clone())
}

fn normalize(value: &str) -> String {
    value
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"<?xml version="1.0" encoding="utf-8"?>
<lenses xmlns="lensDatabase">
  <lens ID="2BDA368D-EBD9-4E19-9E59-19BDE21D2352" type="Zoom">
    <fullName>Canon EF 11-24mm f/4L USM</fullName>
    <shortName>Canon EF 11-24mm f/4L USM</shortName>
    <creator>Phase One A/S</creator>
    <note>Based on lens 2500000399 on Canon 5DS-R.</note>
    <brand>Canon</brand>
    <mount>Canon EF</mount>
    <fileDetectID type="CANON">N[508,11,24,1,288,128]S[EF11-24mm f/4L USM]</fileDetectID>
    <focalLength dependent="true" min="11.0" max="24.0"/>
    <aperture dependent="true" min="4.0" max="22.0"/>
    <focusDistance dependent="false" min="0.28" max="INF"/>
    <lensGeometricAberration focalLength="12.0" focusDistance="INF"><shift>0 0.0 7298 21.9 13774 43.2</shift></lensGeometricAberration>
    <lensLightFalloff focalLength="11.0" aperture="4.5" focusDistance="INF"><falloff>1770 -0.015 4391 -0.051</falloff></lensLightFalloff>
    <chromAberCurve focalLength="11.0" aperture="9.0"><redShift>0 0.0000 2997 -0.9707</redShift><blueShift>0 0.0000 2997 -1.1000</blueShift></chromAberCurve>
  </lens>
  <lens ID="AAAA1111-2222-3333-4444-555566667777" type="Prime">
    <fullName>Test Prime 50mm f/2</fullName>
    <shortName>Test Prime 50mm</shortName>
    <brand>Test</brand>
    <mount>TestMount</mount>
    <focalLength dependent="false" min="50.0" max="50.0"/>
    <aperture dependent="true" min="2.0" max="16.0"/>
  </lens>
</lenses>"#;

    #[test]
    fn parses_a_lens() {
        let lenses = parse_lenses(SAMPLE);
        assert_eq!(lenses.len(), 2);
        let l = &lenses[0];
        assert_eq!(l.id, "2BDA368D-EBD9-4E19-9E59-19BDE21D2352");
        assert_eq!(l.full_name, "Canon EF 11-24mm f/4L USM");
        assert_eq!(l.brand, "Canon");
        assert_eq!(l.mount, "Canon EF");
        assert_eq!(l.kind, LensKind::Zoom);
        assert_eq!(l.focal_min, Some(11.0));
        assert_eq!(l.focal_max, Some(24.0));
        assert_eq!(l.aperture_min, Some(4.0));
        assert_eq!(l.aperture_max, Some(22.0));
        assert!(l.detect_id.as_deref().unwrap().contains("EF11-24mm"));
        assert_eq!(l.distortion.len(), 1);
        assert_eq!(l.distortion[0].shift.len(), 3);
        assert_eq!(l.distortion[0].shift[1], (7298.0, 21.9));
        assert_eq!(l.falloff.len(), 1);
        assert_eq!(l.falloff[0].falloff.len(), 2);
        assert_eq!(l.chroma.len(), 1);
        assert_eq!(l.chroma[0].red_shift.len(), 2);
        assert_eq!(l.chroma[0].blue_shift.len(), 2);
        assert_eq!(lenses[1].kind, LensKind::Prime);
    }

    #[test]
    fn decode_round_trips() {
        let plain = "<?xml version=\"1.0\"?><lenses/>";
        let obfuscated: Vec<u8> = format!("~~{plain}").bytes().map(|b| b ^ 0x81).collect();
        assert_eq!(decode_cdx(&obfuscated).unwrap(), plain);
    }

    #[test]
    fn decode_passes_through_plain_text() {
        let plain = "<lenses xmlns=\"lensDatabase\"></lenses>";
        assert_eq!(decode_cdx(plain.as_bytes()).unwrap(), plain);
    }

    #[test]
    fn decode_tolerates_truncation() {
        let plain = "<lenses>";
        let mut obfuscated: Vec<u8> = plain.bytes().map(|b| b ^ 0x81).collect();
        obfuscated[0] = 0xFF;
        obfuscated.insert(0, 0xFF);
        // Not an obfuscated header (0xFF 0xFF is), so decode as plain text.
        let decoded = decode_cdx(&obfuscated).unwrap();
        assert!(decoded.contains("lenses"));
    }

    #[test]
    fn odd_pair_count_drops_last() {
        assert_eq!(parse_pairs("1 2 3").len(), 1);
        assert_eq!(parse_pairs("1 2 3 4").len(), 2);
        assert!(parse_pairs("").is_empty());
        assert!(parse_pairs("   ").is_empty());
    }

    #[test]
    fn detects_canon_11_24() {
        let lenses = parse_lenses(SAMPLE);
        let found = detect_lens(&lenses, "Canon", "Canon EF 11-24mm f/4L USM", 12.0).unwrap();
        assert_eq!(found.id, "2BDA368D-EBD9-4E19-9E59-19BDE21D2352");
    }

    #[test]
    fn missing_curves_return_none() {
        let lenses = parse_lenses(SAMPLE);
        let prime = &lenses[1];
        assert!(prime.falloff_at(50.0, 2.0).is_none());
        assert!(prime.chroma_at(50.0, 2.0).is_none());
        assert!(prime.distortion_at(50.0).is_none());
        let zoom = &lenses[0];
        assert_eq!(zoom.falloff_at(11.0, 4.5).unwrap().aperture, 4.5);
        assert_eq!(zoom.chroma_at(11.0, 9.0).unwrap().aperture, 9.0);
    }

    #[test]
    fn empty_xml_yields_no_lenses() {
        assert!(parse_lenses("").is_empty());
        assert!(parse_lenses("<lenses/>").is_empty());
    }

    #[test]
    fn malformed_xml_does_not_panic() {
        let broken = "<lenses><lens ID=\"1\"><fullName>Broken</fullName><shift>1 2 3";
        let _ = parse_lenses(broken);
        let broken2 = "<lenses><lens ID=\"1\"><fullName>Broken</fullName><shift>1 2 3</lens>";
        let lenses = parse_lenses(broken2);
        assert_eq!(lenses.len(), 1);
        assert_eq!(lenses[0].full_name, "Broken");
    }

    #[test]
    fn interpolation_picks_nearest_focal() {
        let lenses = parse_lenses(SAMPLE);
        let zoom = &lenses[0];
        assert_eq!(zoom.distortion_at(11.5).unwrap().focal_length, 12.0);
    }

    #[test]
    fn reset_marker_does_not_leak_into_next_lens() {
        let xml = r#"<lenses>
  <lens ID="A" type="Zoom"><fullName>A</fullName>
    <chromAberCurve reset="TRUE"/>
    <lensLightFalloff focalLength="70.0" aperture="8.0"><falloff>0 0</falloff></lensLightFalloff>
  </lens>
  <lens ID="B" type="Prime"><fullName>B</fullName></lens>
</lenses>"#;
        let lenses = parse_lenses(xml);
        assert_eq!(lenses.len(), 2);
        assert!(lenses[0].chroma.is_empty());
        assert_eq!(lenses[0].falloff.len(), 1);
        assert!(lenses[1].falloff.is_empty());
    }

    /// Lazily parses the real 12.9 MB database from a Capture One install.
    #[test]
    #[ignore]
    fn real_lens_database_summary() {
        let path = std::env::var("C1_LENSDB").unwrap_or_else(|_| {
            r"C:\Program Files\Capture One\Capture One\lensDatabase.cdx".into()
        });
        let lenses = load_lens_database(Path::new(&path)).expect("load lens database");
        println!("lenses: {}", lenses.len());
        let sample = detect_lens(&lenses, "Canon", "Canon EF 11-24mm f/4L USM", 12.0)
            .expect("Canon EF 11-24mm");
        println!(
            "{}: distortion={} falloff={} chroma={}",
            sample.full_name,
            sample.distortion.len(),
            sample.falloff.len(),
            sample.chroma.len()
        );
        if let Some(d) = sample.distortion_at(12.0) {
            println!("distortion@12mm: {} points", d.shift.len());
        }
    }
}
