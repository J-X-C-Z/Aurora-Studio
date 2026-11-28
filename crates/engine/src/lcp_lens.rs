//! Adobe Lens Correction Profile (`.lcp`) parser, interpolator, and matcher.
//!
//! Parses standard Adobe XMP/RDF `.lcp` XML documents using `roxmltree`,
//! extracting camera/lens metadata, Brown-Conrady radial/tangential distortion
//! (`PerspectiveModel`), vignetting (`VignetteModel`), and lateral chromatic
//! aberration (`ChromaticRedGreenModel`, `ChromaticGreenModel`,
//! `ChromaticBlueGreenModel`).
//!
//! Supports focal-length and aperture interpolation across multi-entry profiles
//! as well as scored camera/lens matching via [`LcpIndex`].

use std::cmp::Ordering;
use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use roxmltree::{Document, Node};
use serde::{Deserialize, Serialize};

const ST_CAMERA_NS: &str = "http://ns.adobe.com/photoshop/1.0/camera-profile";
const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

/// Brown-Conrady perspective distortion model from `stCamera:PerspectiveModel`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LcpDistortionModel {
    pub focal_length_x: f64,
    pub focal_length_y: f64,
    pub image_x_center: f64,
    pub image_y_center: f64,
    /// Brown radial distortion coefficients `k1`, `k2`, `k3`.
    pub k1: f64,
    pub k2: f64,
    pub k3: f64,
    /// Optional tangential distortion coefficients `p1`, `p2`.
    pub p1: f64,
    pub p2: f64,
    pub scale_factor: f64,
}

impl Default for LcpDistortionModel {
    fn default() -> Self {
        Self {
            focal_length_x: 1.0,
            focal_length_y: 1.0,
            image_x_center: 0.5,
            image_y_center: 0.5,
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            scale_factor: 1.0,
        }
    }
}

impl LcpDistortionModel {
    /// Evaluates the radial polynomial `1 + k1*r^2 + k2*r^4 + k3*r^6`.
    pub fn radial_factor(&self, r2: f64) -> f64 {
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
    }

    /// Maps a normalized undistorted point `(x, y)` (relative to optical center,
    /// scaled by focal length) to its distorted coordinates using the Brown model.
    pub fn distort_point(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let radial = self.radial_factor(r2);
        let dx_tan = 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let dy_tan = self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (
            self.scale_factor * (x * radial + dx_tan),
            self.scale_factor * (y * radial + dy_tan),
        )
    }

    /// Linearly interpolates between `self` and `other` with factor `t` in `0.0..=1.0`.
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            focal_length_x: lerp_f64(self.focal_length_x, other.focal_length_x, t),
            focal_length_y: lerp_f64(self.focal_length_y, other.focal_length_y, t),
            image_x_center: lerp_f64(self.image_x_center, other.image_x_center, t),
            image_y_center: lerp_f64(self.image_y_center, other.image_y_center, t),
            k1: lerp_f64(self.k1, other.k1, t),
            k2: lerp_f64(self.k2, other.k2, t),
            k3: lerp_f64(self.k3, other.k3, t),
            p1: lerp_f64(self.p1, other.p1, t),
            p2: lerp_f64(self.p2, other.p2, t),
            scale_factor: lerp_f64(self.scale_factor, other.scale_factor, t),
        }
    }
}

/// Radial vignetting model from `stCamera:VignetteModel`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LcpVignetteModel {
    pub focal_length_x: f64,
    pub focal_length_y: f64,
    pub image_x_center: f64,
    pub image_y_center: f64,
    /// Vignette polynomial coefficients `v1`, `v2`, `v3`.
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
}

impl Default for LcpVignetteModel {
    fn default() -> Self {
        Self {
            focal_length_x: 1.0,
            focal_length_y: 1.0,
            image_x_center: 0.5,
            image_y_center: 0.5,
            v1: 0.0,
            v2: 0.0,
            v3: 0.0,
        }
    }
}

impl LcpVignetteModel {
    /// Evaluates the Adobe vignette polynomial `1 + v1*r^2 + v2*r^4 + v3*r^6`.
    pub fn polynomial(&self, r2: f64) -> f64 {
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        1.0 + self.v1 * r2 + self.v2 * r4 + self.v3 * r6
    }

    /// Evaluates the inverse falloff gain (`1 / polynomial(r2)`), clamped to a
    /// safe positive range so corner pixels never divide by zero.
    pub fn correction_gain(&self, r2: f64) -> f64 {
        let falloff = self.polynomial(r2).max(0.05);
        (1.0 / falloff).clamp(0.1, 16.0)
    }

    /// Linearly interpolates between `self` and `other` with factor `t` in `0.0..=1.0`.
    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            focal_length_x: lerp_f64(self.focal_length_x, other.focal_length_x, t),
            focal_length_y: lerp_f64(self.focal_length_y, other.focal_length_y, t),
            image_x_center: lerp_f64(self.image_x_center, other.image_x_center, t),
            image_y_center: lerp_f64(self.image_y_center, other.image_y_center, t),
            v1: lerp_f64(self.v1, other.v1, t),
            v2: lerp_f64(self.v2, other.v2, t),
            v3: lerp_f64(self.v3, other.v3, t),
        }
    }
}

/// Lateral chromatic aberration model combining Green, Red/Green, and Blue/Green terms.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LcpChromaticAberrationModel {
    pub green: Option<LcpDistortionModel>,
    pub red_green: Option<LcpDistortionModel>,
    pub blue_green: Option<LcpDistortionModel>,
}

impl LcpChromaticAberrationModel {
    pub fn is_empty(&self) -> bool {
        self.green.is_none() && self.red_green.is_none() && self.blue_green.is_none()
    }

    pub fn lerp(&self, other: &Self, t: f64) -> Self {
        Self {
            green: lerp_opt_distortion(self.green.as_ref(), other.green.as_ref(), t),
            red_green: lerp_opt_distortion(self.red_green.as_ref(), other.red_green.as_ref(), t),
            blue_green: lerp_opt_distortion(self.blue_green.as_ref(), other.blue_green.as_ref(), t),
        }
    }
}

/// One calibration entry inside an Adobe `.lcp` file (one focal/aperture/distance sample).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LcpEntry {
    pub make: Option<String>,
    pub camera_model: Option<String>,
    pub camera_pretty_name: Option<String>,
    pub lens: Option<String>,
    pub lens_pretty_name: Option<String>,
    pub lens_info: Option<String>,
    pub camera_raw_profile: bool,
    pub sensor_format_factor: Option<f64>,
    /// Focal length in millimetres.
    pub focal_length: f64,
    /// APEX aperture value (`Av`) from `stCamera:ApertureValue`.
    pub aperture_value: Option<f64>,
    /// Derived f-number (`2^(Av/2)`) or explicit `FNumber`.
    pub f_number: f64,
    /// Focus distance in metres (`None` or `f64::INFINITY` for infinity).
    pub focus_distance: Option<f64>,
    pub image_width: Option<u32>,
    pub image_length: Option<u32>,
    pub distortion: Option<LcpDistortionModel>,
    pub vignette: Option<LcpVignetteModel>,
    pub chromatic_aberration: LcpChromaticAberrationModel,
}

/// Parsed Adobe `.lcp` profile containing lens metadata and all calibration entries.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LcpProfile {
    pub make: String,
    pub camera_model: String,
    pub camera_pretty_name: Option<String>,
    pub lens: String,
    pub lens_pretty_name: Option<String>,
    pub lens_info: Option<String>,
    pub camera_raw_profile: bool,
    pub sensor_format_factor: Option<f64>,
    pub entries: Vec<LcpEntry>,
}

/// Interpolated optical parameters for a requested focal length and aperture.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LcpResolvedParams {
    pub focal_length: f64,
    pub f_number: f64,
    pub focus_distance: Option<f64>,
    pub distortion: Option<LcpDistortionModel>,
    pub vignette: Option<LcpVignetteModel>,
    pub chromatic_aberration: LcpChromaticAberrationModel,
}

/// Query for matching a camera/lens against an [`LcpIndex`].
#[derive(Clone, Debug, Default)]
pub struct LcpQuery {
    pub make: Option<String>,
    pub camera_model: Option<String>,
    pub lens: Option<String>,
    pub focal_length: Option<f64>,
    pub f_number: Option<f64>,
    pub focus_distance: Option<f64>,
    pub prefer_raw_profile: Option<bool>,
}

/// Match result returned by [`LcpIndex::match_best`].
#[derive(Clone, Debug)]
pub struct LcpMatch<'a> {
    pub profile: &'a LcpProfile,
    pub score: f64,
    pub resolved: Option<LcpResolvedParams>,
}

/// In-memory index of loaded Adobe `.lcp` profiles for camera/lens lookup.
#[derive(Clone, Debug, Default)]
pub struct LcpIndex {
    profiles: Vec<LcpProfile>,
}

impl LcpProfile {
    /// Loads and parses an `.lcp` file from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading LCP file {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("parsing LCP file {}", path.display()))
    }

    /// Parses an Adobe `.lcp` XML document from a string.
    pub fn parse(xml: &str) -> Result<Self> {
        if xml.trim().is_empty() {
            return Err(anyhow!("empty LCP XML document"));
        }
        let doc = Document::parse(xml).map_err(|e| anyhow!("invalid LCP XML: {e}"))?;

        let mut entries = Vec::new();
        for node in doc.descendants() {
            if is_tag(node, ST_CAMERA_NS, "CameraProfiles") {
                for seq in node.children().filter(|n| is_tag(*n, RDF_NS, "Seq")) {
                    for li in seq.children().filter(|n| is_tag(*n, RDF_NS, "li")) {
                        let target = li
                            .children()
                            .find(|n| is_tag(*n, RDF_NS, "Description"))
                            .unwrap_or(li);
                        if let Some(entry) = parse_entry(target) {
                            entries.push(entry);
                        }
                    }
                }
            }
        }

        // Fallback: single-entry profiles placed directly on <rdf:Description>.
        if entries.is_empty() {
            for node in doc.descendants().filter(|n| is_tag(*n, RDF_NS, "Description")) {
                if has_st_camera_profile_fields(node) {
                    if let Some(entry) = parse_entry(node) {
                        entries.push(entry);
                    }
                }
            }
        }

        if entries.is_empty() {
            return Err(anyhow!("no stCamera calibration entries found in LCP XML"));
        }

        let first = &entries[0];
        let make = entries
            .iter()
            .find_map(|e| e.make.clone())
            .unwrap_or_default();
        let camera_model = entries
            .iter()
            .find_map(|e| e.camera_model.clone())
            .unwrap_or_default();
        let camera_pretty_name = entries.iter().find_map(|e| e.camera_pretty_name.clone());
        let lens = entries
            .iter()
            .find_map(|e| e.lens.clone())
            .unwrap_or_default();
        let lens_pretty_name = entries.iter().find_map(|e| e.lens_pretty_name.clone());
        let lens_info = entries.iter().find_map(|e| e.lens_info.clone());
        let sensor_format_factor = entries.iter().find_map(|e| e.sensor_format_factor);
        let camera_raw_profile = first.camera_raw_profile;

        Ok(Self {
            make,
            camera_model,
            camera_pretty_name,
            lens,
            lens_pretty_name,
            lens_info,
            camera_raw_profile,
            sensor_format_factor,
            entries,
        })
    }

    /// Returns the `(min, max)` focal length range covered by this profile.
    pub fn focal_range(&self) -> Option<(f64, f64)> {
        let mut iter = self.entries.iter().map(|e| e.focal_length).filter(|v| v.is_finite() && *v > 0.0);
        let first = iter.next()?;
        let (mut min_v, mut max_v) = (first, first);
        for v in iter {
            if v < min_v {
                min_v = v;
            }
            if v > max_v {
                max_v = v;
            }
        }
        Some((min_v, max_v))
    }

    /// Interpolates distortion, vignetting, and chromatic aberration for a given
    /// focal length (mm), f-number (e.g. `2.8`), and optional focus distance (m).
    pub fn interpolate(
        &self,
        focal_length: f64,
        f_number: f64,
        focus_distance: Option<f64>,
    ) -> Option<LcpResolvedParams> {
        if self.entries.is_empty() {
            return None;
        }

        let distortion = interpolate_model(
            &self.entries,
            focal_length,
            f_number,
            focus_distance,
            |e| e.distortion.as_ref(),
            |a, b, t| a.lerp(b, t),
        );
        let vignette = interpolate_model(
            &self.entries,
            focal_length,
            f_number,
            focus_distance,
            |e| e.vignette.as_ref(),
            |a, b, t| a.lerp(b, t),
        );
        let ca_entries: Vec<&LcpEntry> = self
            .entries
            .iter()
            .filter(|e| !e.chromatic_aberration.is_empty())
            .collect();
        let chromatic_aberration = if ca_entries.is_empty() {
            LcpChromaticAberrationModel::default()
        } else {
            interpolate_model(
                &self.entries,
                focal_length,
                f_number,
                focus_distance,
                |e| (!e.chromatic_aberration.is_empty()).then_some(&e.chromatic_aberration),
                |a, b, t| a.lerp(b, t),
            )
            .unwrap_or_default()
        };

        Some(LcpResolvedParams {
            focal_length,
            f_number,
            focus_distance,
            distortion,
            vignette,
            chromatic_aberration,
        })
    }
}

impl LcpIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, profile: LcpProfile) {
        self.profiles.push(profile);
    }

    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    pub fn profiles(&self) -> &[LcpProfile] {
        &self.profiles
    }

    /// Loads all `.lcp` files in a directory (non-recursively), skipping unreadable files.
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let mut index = Self::new();
        for entry in fs::read_dir(dir).with_context(|| format!("reading dir {}", dir.display()))? {
            let entry = entry?;
            let path = entry.path();
            if path
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("lcp"))
            {
                if let Ok(profile) = LcpProfile::load(&path) {
                    index.push(profile);
                }
            }
        }
        Ok(index)
    }

    /// Finds the best matching profile for `query` above a minimum relevance threshold.
    pub fn match_best(&self, query: &LcpQuery) -> Option<LcpMatch<'_>> {
        self.search_top(query, 1).into_iter().next()
    }

    /// Returns up to `limit` best matching profiles sorted by descending score.
    pub fn search_top(&self, query: &LcpQuery, limit: usize) -> Vec<LcpMatch<'_>> {
        if limit == 0 || self.profiles.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(&LcpProfile, f64)> = self
            .profiles
            .iter()
            .filter_map(|p| {
                let s = score_profile(p, query);
                (s > 0.0).then_some((p, s))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        scored
            .into_iter()
            .take(limit)
            .map(|(profile, score)| {
                let resolved = query.focal_length.and_then(|fl| {
                    profile.interpolate(fl, query.f_number.unwrap_or(4.0), query.focus_distance)
                });
                LcpMatch {
                    profile,
                    score,
                    resolved,
                }
            })
            .collect()
    }
}

// ───────────────────────────── Parsing helpers ─────────────────────────────

fn is_tag(node: Node<'_, '_>, ns: &str, local: &str) -> bool {
    node.is_element()
        && node.tag_name().name() == local
        && node.tag_name().namespace() == Some(ns)
}

fn has_st_camera_profile_fields(node: Node<'_, '_>) -> bool {
    get_prop(node, ST_CAMERA_NS, "FocalLength").is_some()
        || find_child_model(node, "PerspectiveModel").is_some()
        || find_child_model(node, "VignetteModel").is_some()
}

fn get_prop<'a>(node: Node<'a, 'a>, ns: &str, name: &str) -> Option<String> {
    if let Some(attr) = node
        .attributes()
        .find(|a| a.name() == name && (a.namespace() == Some(ns) || a.namespace().is_none()))
    {
        return Some(attr.value().trim().to_string());
    }
    for child in node.children().filter(|c| is_tag(*c, ns, name)) {
        if let Some(text) = child.text() {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn get_f64(node: Node<'_, '_>, name: &str) -> Option<f64> {
    let raw = get_prop(node, ST_CAMERA_NS, name)?;
    if raw.eq_ignore_ascii_case("infinity") || raw.eq_ignore_ascii_case("inf") {
        return Some(f64::INFINITY);
    }
    raw.parse::<f64>().ok()
}

fn get_bool(node: Node<'_, '_>, name: &str) -> Option<bool> {
    let raw = get_prop(node, ST_CAMERA_NS, name)?;
    match raw.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" => Some(true),
        "false" | "0" | "no" => Some(false),
        _ => None,
    }
}

fn find_child_model<'a>(node: Node<'a, 'a>, name: &str) -> Option<Node<'a, 'a>> {
    let container = node.children().find(|c| is_tag(*c, ST_CAMERA_NS, name))?;
    if let Some(desc) = container
        .children()
        .find(|c| is_tag(*c, RDF_NS, "Description"))
    {
        return Some(desc);
    }
    Some(container)
}

fn parse_distortion_model(node: Node<'_, '_>) -> LcpDistortionModel {
    LcpDistortionModel {
        focal_length_x: get_f64(node, "FocalLengthX").unwrap_or(1.0),
        focal_length_y: get_f64(node, "FocalLengthY").unwrap_or(1.0),
        image_x_center: get_f64(node, "ImageXCenter").unwrap_or(0.5),
        image_y_center: get_f64(node, "ImageYCenter").unwrap_or(0.5),
        k1: get_f64(node, "RadialDistortParam1").unwrap_or(0.0),
        k2: get_f64(node, "RadialDistortParam2").unwrap_or(0.0),
        k3: get_f64(node, "RadialDistortParam3").unwrap_or(0.0),
        p1: get_f64(node, "TangentialDistortParam1").unwrap_or(0.0),
        p2: get_f64(node, "TangentialDistortParam2").unwrap_or(0.0),
        scale_factor: get_f64(node, "ScaleFactor").unwrap_or(1.0),
    }
}

fn parse_vignette_model(node: Node<'_, '_>) -> LcpVignetteModel {
    LcpVignetteModel {
        focal_length_x: get_f64(node, "FocalLengthX").unwrap_or(1.0),
        focal_length_y: get_f64(node, "FocalLengthY").unwrap_or(1.0),
        image_x_center: get_f64(node, "ImageXCenter").unwrap_or(0.5),
        image_y_center: get_f64(node, "ImageYCenter").unwrap_or(0.5),
        v1: get_f64(node, "VignetteModelParam1").unwrap_or(0.0),
        v2: get_f64(node, "VignetteModelParam2").unwrap_or(0.0),
        v3: get_f64(node, "VignetteModelParam3").unwrap_or(0.0),
    }
}

fn parse_entry(node: Node<'_, '_>) -> Option<LcpEntry> {
    let focal_length = get_f64(node, "FocalLength")?;
    let aperture_value = get_f64(node, "ApertureValue");
    let f_number = get_f64(node, "FNumber")
        .or_else(|| aperture_value.map(|av| 2.0_f64.powf(av * 0.5)))
        .unwrap_or(0.0);
    let focus_distance = get_f64(node, "FocusDistance");

    let distortion = find_child_model(node, "PerspectiveModel").map(parse_distortion_model);
    let vignette = find_child_model(node, "VignetteModel").map(parse_vignette_model);
    let chromatic_aberration = LcpChromaticAberrationModel {
        green: find_child_model(node, "ChromaticGreenModel").map(parse_distortion_model),
        red_green: find_child_model(node, "ChromaticRedGreenModel").map(parse_distortion_model),
        blue_green: find_child_model(node, "ChromaticBlueGreenModel").map(parse_distortion_model),
    };

    Some(LcpEntry {
        make: get_prop(node, ST_CAMERA_NS, "Make"),
        camera_model: get_prop(node, ST_CAMERA_NS, "Model"),
        camera_pretty_name: get_prop(node, ST_CAMERA_NS, "CameraPrettyName"),
        lens: get_prop(node, ST_CAMERA_NS, "Lens"),
        lens_pretty_name: get_prop(node, ST_CAMERA_NS, "LensPrettyName"),
        lens_info: get_prop(node, ST_CAMERA_NS, "LensInfo"),
        camera_raw_profile: get_bool(node, "CameraRawProfile").unwrap_or(false),
        sensor_format_factor: get_f64(node, "SensorFormatFactor"),
        focal_length,
        aperture_value,
        f_number,
        focus_distance,
        image_width: get_f64(node, "ImageWidth").map(|v| v.round().max(0.0) as u32),
        image_length: get_f64(node, "ImageLength").map(|v| v.round().max(0.0) as u32),
        distortion,
        vignette,
        chromatic_aberration,
    })
}

// ───────────────────────────── Interpolation helpers ─────────────────────────────

fn lerp_f64(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

fn lerp_opt_distortion(
    a: Option<&LcpDistortionModel>,
    b: Option<&LcpDistortionModel>,
    t: f64,
) -> Option<LcpDistortionModel> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.lerp(y, t)),
        (Some(x), None) => Some(x.clone()),
        (None, Some(y)) => Some(y.clone()),
        (None, None) => None,
    }
}

/// Two-stage bracket interpolation:
/// 1. Group entries carrying model `T` by distinct focal lengths (tolerance 0.05 mm).
/// 2. Within the lower and upper focal-length brackets, interpolate by f-number
///    (and pick closest inverse focus distance when multiple distances share an aperture).
/// 3. Linearly interpolate between the lower and upper focal-length results.
fn interpolate_model<T, FGet, FLerp>(
    entries: &[LcpEntry],
    focal_length: f64,
    f_number: f64,
    focus_distance: Option<f64>,
    get: FGet,
    lerp: FLerp,
) -> Option<T>
where
    T: Clone,
    FGet: Fn(&LcpEntry) -> Option<&T>,
    FLerp: Fn(&T, &T, f64) -> T,
{
    let mut with_model: Vec<(&LcpEntry, &T)> = entries
        .iter()
        .filter_map(|e| get(e).map(|m| (e, m)))
        .collect();
    if with_model.is_empty() {
        return None;
    }

    with_model.sort_by(|a, b| {
        a.0.focal_length
            .partial_cmp(&b.0.focal_length)
            .unwrap_or(Ordering::Equal)
    });

    // Cluster distinct focal lengths.
    let mut groups: Vec<(f64, Vec<(&LcpEntry, &T)>)> = Vec::new();
    for item in with_model {
        let fl = item.0.focal_length;
        if let Some(last) = groups.last_mut() {
            if (last.0 - fl).abs() <= 0.05 {
                last.1.push(item);
                continue;
            }
        }
        groups.push((fl, vec![item]));
    }

    if groups.len() == 1 || focal_length <= groups[0].0 {
        return interpolate_by_aperture(&groups[0].1, f_number, focus_distance, &lerp);
    }
    let last_idx = groups.len() - 1;
    if focal_length >= groups[last_idx].0 {
        return interpolate_by_aperture(&groups[last_idx].1, f_number, focus_distance, &lerp);
    }

    // Find bracketing focal lengths.
    for w in groups.windows(2) {
        let (fl_lo, ref lo_items) = w[0];
        let (fl_hi, ref hi_items) = w[1];
        if focal_length >= fl_lo && focal_length <= fl_hi {
            let lo_val = interpolate_by_aperture(lo_items, f_number, focus_distance, &lerp)?;
            let hi_val = interpolate_by_aperture(hi_items, f_number, focus_distance, &lerp)?;
            let span = (fl_hi - fl_lo).max(1e-6);
            let t = ((focal_length - fl_lo) / span).clamp(0.0, 1.0);
            return Some(lerp(&lo_val, &hi_val, t));
        }
    }

    interpolate_by_aperture(&groups[0].1, f_number, focus_distance, &lerp)
}

fn inv_distance(d: Option<f64>) -> f64 {
    match d {
        Some(v) if v.is_finite() && v > 1e-4 => 1.0 / v,
        _ => 0.0, // infinity or unspecified
    }
}

fn interpolate_by_aperture<T, FLerp>(
    items: &[(&LcpEntry, &T)],
    f_number: f64,
    focus_distance: Option<f64>,
    lerp: &FLerp,
) -> Option<T>
where
    T: Clone,
    FLerp: Fn(&T, &T, f64) -> T,
{
    if items.is_empty() {
        return None;
    }
    if items.len() == 1 {
        return Some(items[0].1.clone());
    }

    let target_inv_d = inv_distance(focus_distance);
    let mut sorted: Vec<(&LcpEntry, &T)> = items.to_vec();
    sorted.sort_by(|a, b| {
        a.0.f_number
            .partial_cmp(&b.0.f_number)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                let da = (inv_distance(a.0.focus_distance) - target_inv_d).abs();
                let db = (inv_distance(b.0.focus_distance) - target_inv_d).abs();
                da.partial_cmp(&db).unwrap_or(Ordering::Equal)
            })
    });

    // Deduplicate identical apertures keeping the closest focus distance.
    let mut by_ap: Vec<(f64, &T)> = Vec::new();
    for (entry, model) in sorted {
        if let Some(last) = by_ap.last() {
            if (last.0 - entry.f_number).abs() <= 0.05 {
                continue;
            }
        }
        by_ap.push((entry.f_number, model));
    }

    if by_ap.len() == 1 || f_number <= by_ap[0].0 {
        return Some(by_ap[0].1.clone());
    }
    let last = by_ap.len() - 1;
    if f_number >= by_ap[last].0 {
        return Some(by_ap[last].1.clone());
    }

    for w in by_ap.windows(2) {
        let (ap_lo, m_lo) = w[0];
        let (ap_hi, m_hi) = w[1];
        if f_number >= ap_lo && f_number <= ap_hi {
            let span = (ap_hi - ap_lo).max(1e-6);
            let t = ((f_number - ap_lo) / span).clamp(0.0, 1.0);
            return Some(lerp(m_lo, m_hi, t));
        }
    }

    Some(by_ap[0].1.clone())
}

// ───────────────────────────── Matching / scoring helpers ─────────────────────────────

fn normalize_tokens(s: &str) -> Vec<String> {
    s.to_ascii_lowercase()
        .split(|c: char| !c.is_ascii_alphanumeric() && c != '.')
        .filter(|tok| !tok.is_empty())
        .map(str::to_string)
        .collect()
}

fn text_similarity(query: &str, candidate: &str) -> f64 {
    let q_norm = query.trim().to_ascii_lowercase();
    let c_norm = candidate.trim().to_ascii_lowercase();
    if q_norm.is_empty() || c_norm.is_empty() {
        return 0.0;
    }
    if q_norm == c_norm {
        return 1.0;
    }
    if c_norm.contains(&q_norm) || q_norm.contains(&c_norm) {
        return 0.85;
    }
    let q_toks = normalize_tokens(&q_norm);
    let c_toks = normalize_tokens(&c_norm);
    if q_toks.is_empty() || c_toks.is_empty() {
        return 0.0;
    }
    let hits = q_toks.iter().filter(|t| c_toks.contains(t)).count();
    let union = (q_toks.len() + c_toks.len() - hits).max(1);
    hits as f64 / union as f64
}

fn score_profile(profile: &LcpProfile, query: &LcpQuery) -> f64 {
    let mut score = 0.0;

    if let Some(ref q_lens) = query.lens {
        let mut best = text_similarity(q_lens, &profile.lens);
        if let Some(ref pretty) = profile.lens_pretty_name {
            best = best.max(text_similarity(q_lens, pretty));
        }
        if best <= 0.0 {
            return 0.0;
        }
        score += best * 60.0;
    }

    if let Some(ref q_model) = query.camera_model {
        let mut best = text_similarity(q_model, &profile.camera_model);
        if let Some(ref pretty) = profile.camera_pretty_name {
            best = best.max(text_similarity(q_model, pretty));
        }
        score += best * 25.0;
    }

    if let Some(ref q_make) = query.make {
        score += text_similarity(q_make, &profile.make) * 10.0;
    }

    if let (Some(q_fl), Some((min_fl, max_fl))) = (query.focal_length, profile.focal_range()) {
        if q_fl >= min_fl - 0.5 && q_fl <= max_fl + 0.5 {
            score += 10.0;
        } else {
            let dist = if q_fl < min_fl { min_fl - q_fl } else { q_fl - max_fl };
            score -= (dist / q_fl.max(1.0)).min(1.0) * 15.0;
        }
    }

    if let Some(prefer_raw) = query.prefer_raw_profile {
        if profile.camera_raw_profile == prefer_raw {
            score += 5.0;
        }
    }

    score.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_ZOOM_LCP: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:stCamera="http://ns.adobe.com/photoshop/1.0/camera-profile">
      <stCamera:CameraProfiles>
        <rdf:Seq>
          <rdf:li rdf:parseType="Resource"
            stCamera:Make="NIKON CORPORATION"
            stCamera:Model="NIKON Z 6_3"
            stCamera:CameraPrettyName="Nikon Z6 III"
            stCamera:Lens="NIKKOR Z 24-70mm f/2.8 S"
            stCamera:LensPrettyName="NIKKOR Z 24-70mm f/2.8 S"
            stCamera:CameraRawProfile="True"
            stCamera:SensorFormatFactor="1.0"
            stCamera:FocalLength="24.0"
            stCamera:ApertureValue="2.970854"
            stCamera:FocusDistance="3.0">
            <stCamera:PerspectiveModel rdf:parseType="Resource"
              stCamera:FocalLengthX="0.6667"
              stCamera:FocalLengthY="0.6667"
              stCamera:ImageXCenter="0.5"
              stCamera:ImageYCenter="0.5"
              stCamera:RadialDistortParam1="-0.080"
              stCamera:RadialDistortParam2="0.020"
              stCamera:RadialDistortParam3="-0.004" />
            <stCamera:VignetteModel rdf:parseType="Resource"
              stCamera:FocalLengthX="0.6667"
              stCamera:FocalLengthY="0.6667"
              stCamera:VignetteModelParam1="-0.60"
              stCamera:VignetteModelParam2="0.20"
              stCamera:VignetteModelParam3="-0.05" />
            <stCamera:ChromaticRedGreenModel rdf:parseType="Resource"
              stCamera:RadialDistortParam1="0.0004"
              stCamera:ScaleFactor="1.0002" />
            <stCamera:ChromaticBlueGreenModel rdf:parseType="Resource"
              stCamera:RadialDistortParam1="-0.0006"
              stCamera:ScaleFactor="0.9998" />
          </rdf:li>
          <rdf:li rdf:parseType="Resource"
            stCamera:Make="NIKON CORPORATION"
            stCamera:Model="NIKON Z 6_3"
            stCamera:Lens="NIKKOR Z 24-70mm f/2.8 S"
            stCamera:CameraRawProfile="True"
            stCamera:FocalLength="24.0"
            stCamera:ApertureValue="5.0"
            stCamera:FocusDistance="3.0">
            <stCamera:PerspectiveModel rdf:parseType="Resource"
              stCamera:RadialDistortParam1="-0.080"
              stCamera:RadialDistortParam2="0.020"
              stCamera:RadialDistortParam3="-0.004" />
            <stCamera:VignetteModel rdf:parseType="Resource"
              stCamera:VignetteModelParam1="-0.20"
              stCamera:VignetteModelParam2="0.06"
              stCamera:VignetteModelParam3="0.0" />
          </rdf:li>
          <rdf:li rdf:parseType="Resource"
            stCamera:Make="NIKON CORPORATION"
            stCamera:Model="NIKON Z 6_3"
            stCamera:Lens="NIKKOR Z 24-70mm f/2.8 S"
            stCamera:CameraRawProfile="True"
            stCamera:FocalLength="70.0"
            stCamera:ApertureValue="2.970854"
            stCamera:FocusDistance="3.0">
            <stCamera:PerspectiveModel rdf:parseType="Resource"
              stCamera:FocalLengthX="1.9444"
              stCamera:FocalLengthY="1.9444"
              stCamera:RadialDistortParam1="0.040"
              stCamera:RadialDistortParam2="-0.010"
              stCamera:RadialDistortParam3="0.002" />
            <stCamera:VignetteModel rdf:parseType="Resource"
              stCamera:VignetteModelParam1="-0.40"
              stCamera:VignetteModelParam2="0.10"
              stCamera:VignetteModelParam3="-0.01" />
            <stCamera:ChromaticRedGreenModel rdf:parseType="Resource"
              stCamera:RadialDistortParam1="-0.0002"
              stCamera:ScaleFactor="0.9999" />
            <stCamera:ChromaticBlueGreenModel rdf:parseType="Resource"
              stCamera:RadialDistortParam1="0.0002"
              stCamera:ScaleFactor="1.0001" />
          </rdf:li>
        </rdf:Seq>
      </stCamera:CameraProfiles>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"#;

    const SAMPLE_PRIME_LCP: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:stCamera="http://ns.adobe.com/photoshop/1.0/camera-profile">
      <stCamera:CameraProfiles>
        <rdf:Seq>
          <rdf:li>
            <rdf:Description>
              <stCamera:Make>SONY</stCamera:Make>
              <stCamera:Model>ILCE-7RM5</stCamera:Model>
              <stCamera:Lens>FE 85mm F1.4 GM</stCamera:Lens>
              <stCamera:CameraRawProfile>True</stCamera:CameraRawProfile>
              <stCamera:FocalLength>85.0</stCamera:FocalLength>
              <stCamera:FNumber>1.4</stCamera:FNumber>
              <stCamera:FocusDistance>Infinity</stCamera:FocusDistance>
              <stCamera:PerspectiveModel>
                <rdf:Description
                  stCamera:RadialDistortParam1="0.015"
                  stCamera:RadialDistortParam2="-0.005"
                  stCamera:RadialDistortParam3="0.001" />
              </stCamera:PerspectiveModel>
            </rdf:Description>
          </rdf:li>
        </rdf:Seq>
      </stCamera:CameraProfiles>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"#;

    #[test]
    fn parses_real_format_lcp_attributes_and_child_elements() {
        let zoom = LcpProfile::parse(SAMPLE_ZOOM_LCP).expect("parse zoom lcp");
        assert_eq!(zoom.make, "NIKON CORPORATION");
        assert_eq!(zoom.camera_model, "NIKON Z 6_3");
        assert_eq!(zoom.lens, "NIKKOR Z 24-70mm f/2.8 S");
        assert!(zoom.camera_raw_profile);
        assert_eq!(zoom.entries.len(), 3);
        assert_eq!(zoom.focal_range(), Some((24.0, 70.0)));
        // APEX 2.970854 ~= f/2.8
        assert!((zoom.entries[0].f_number - 2.8).abs() < 0.01);
        let dist = zoom.entries[0].distortion.as_ref().unwrap();
        assert!((dist.k1 - (-0.080)).abs() < 1e-6);
        assert!((dist.k2 - 0.020).abs() < 1e-6);
        assert!((dist.k3 - (-0.004)).abs() < 1e-6);
        assert!(zoom.entries[0].chromatic_aberration.red_green.is_some());
        assert!(zoom.entries[0].chromatic_aberration.blue_green.is_some());

        let prime = LcpProfile::parse(SAMPLE_PRIME_LCP).expect("parse prime lcp");
        assert_eq!(prime.make, "SONY");
        assert_eq!(prime.lens, "FE 85mm F1.4 GM");
        assert_eq!(prime.entries[0].focus_distance, Some(f64::INFINITY));
        assert!((prime.entries[0].f_number - 1.4).abs() < 1e-6);
    }

    #[test]
    fn interpolates_focal_length_and_aperture() {
        let zoom = LcpProfile::parse(SAMPLE_ZOOM_LCP).expect("parse zoom lcp");

        // Midpoint focal length: 47mm is halfway between 24mm and 70mm.
        let mid = zoom.interpolate(47.0, 2.8, Some(3.0)).expect("interpolate");
        let dist = mid.distortion.as_ref().unwrap();
        // (-0.080 + 0.040) / 2 = -0.020
        assert!((dist.k1 - (-0.020)).abs() < 1e-4, "k1={}", dist.k1);
        assert!((dist.k2 - 0.005).abs() < 1e-4, "k2={}", dist.k2);

        // Aperture interpolation at 24mm between f/2.8 (v1=-0.60) and f/5.656 (Av=5.0, v1=-0.20).
        let f_mid = (2.8 + 2.0_f64.powf(2.5)) * 0.5;
        let vig_mid = zoom
            .interpolate(24.0, f_mid, Some(3.0))
            .unwrap()
            .vignette
            .unwrap();
        assert!((vig_mid.v1 - (-0.40)).abs() < 1e-2, "v1={}", vig_mid.v1);

        // Clamping beyond max focal length returns 70mm entry.
        let clamped = zoom.interpolate(200.0, 2.8, None).unwrap();
        assert!((clamped.distortion.unwrap().k1 - 0.040).abs() < 1e-6);
    }

    #[test]
    fn matches_camera_and_lens_in_index() {
        let mut idx = LcpIndex::new();
        idx.push(LcpProfile::parse(SAMPLE_ZOOM_LCP).unwrap());
        idx.push(LcpProfile::parse(SAMPLE_PRIME_LCP).unwrap());

        let q = LcpQuery {
            make: Some("NIKON CORPORATION".into()),
            camera_model: Some("NIKON Z 6_3".into()),
            lens: Some("NIKKOR Z 24-70mm f/2.8 S".into()),
            focal_length: Some(50.0),
            f_number: Some(2.8),
            focus_distance: None,
            prefer_raw_profile: Some(true),
        };
        let m = idx.match_best(&q).expect("match found");
        assert_eq!(m.profile.lens, "NIKKOR Z 24-70mm f/2.8 S");
        assert!(m.score > 90.0, "score={}", m.score);
        assert!(m.resolved.is_some());

        let q_sony = LcpQuery {
            make: Some("SONY".into()),
            camera_model: Some("ILCE-7RM5".into()),
            lens: Some("FE 85mm F1.4 GM".into()),
            focal_length: Some(85.0),
            ..Default::default()
        };
        let m2 = idx.match_best(&q_sony).expect("match sony");
        assert_eq!(m2.profile.lens, "FE 85mm F1.4 GM");
    }

    #[test]
    fn evaluates_brown_distortion_and_vignette_math() {
        let dist = LcpDistortionModel {
            k1: -0.1,
            k2: 0.02,
            k3: -0.005,
            ..Default::default()
        };
        let r2 = 0.25; // r = 0.5
        let expected_factor = 1.0 - 0.1 * 0.25 + 0.02 * 0.0625 - 0.005 * 0.015625;
        assert!((dist.radial_factor(r2) - expected_factor).abs() < 1e-9);
        let (dx, dy) = dist.distort_point(0.5, 0.0);
        assert!((dx - 0.5 * expected_factor).abs() < 1e-9);
        assert!(dy.abs() < 1e-9);

        let vig = LcpVignetteModel {
            v1: -0.5,
            v2: 0.1,
            v3: 0.0,
            ..Default::default()
        };
        let poly = vig.polynomial(0.5); // 1 - 0.25 + 0.025 = 0.775
        assert!((poly - 0.775).abs() < 1e-9);
        assert!((vig.correction_gain(0.5) - (1.0 / 0.775)).abs() < 1e-9);
    }
}
