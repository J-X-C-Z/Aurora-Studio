//! Adobe Lightroom / Camera Raw `crs:` XMP preset parser and read-only Aurora mapper.
//!
//! Parses `.xmp` preset files (both attribute-encoded and element-encoded `crs:`
//! properties, `<rdf:Seq>` tone curves, `<rdf:Alt>` localized names, `Supports*`
//! flags, and optional preset `Amount` multiplier) using `roxmltree`, and converts
//! them read-only into Aurora's [`AdjustmentSet`].

use std::fs;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use roxmltree::{Document, Node};
use serde::{Deserialize, Serialize};

use crate::adjust::{AdjustmentSet, Curve, CurvePoint, WhiteBalanceMode};

const CRS_NS: &str = "http://ns.adobe.com/camera-raw-settings/1.0/";
const RDF_NS: &str = "http://www.w3.org/1999/02/22-rdf-syntax-ns#";

/// Lightroom `crs:Supports*` capability flags on an XMP preset.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct XmpPresetSupports {
    pub amount: bool,
    pub amount2: bool,
    pub color: bool,
    pub monochrome: bool,
    pub high_dynamic_range: bool,
    pub normal_dynamic_range: bool,
    pub scene_referred: bool,
    pub output_referred: bool,
}

impl Default for XmpPresetSupports {
    fn default() -> Self {
        Self {
            amount: true,
            amount2: false,
            color: true,
            monochrome: true,
            high_dynamic_range: true,
            normal_dynamic_range: true,
            scene_referred: true,
            output_referred: true,
        }
    }
}

/// One HSL channel's raw Lightroom adjustments (`-100 ..= 100`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct XmpHslChannel {
    pub hue: Option<f32>,
    pub saturation: Option<f32>,
    pub luminance: Option<f32>,
}

/// Eight Lightroom HSL channels in Aurora `HueSector::ALL` order:
/// Red, Orange, Yellow, Green, Aqua (Cyan), Blue, Purple, Magenta.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct XmpHsl {
    pub red: XmpHslChannel,
    pub orange: XmpHslChannel,
    pub yellow: XmpHslChannel,
    pub green: XmpHslChannel,
    pub aqua: XmpHslChannel,
    pub blue: XmpHslChannel,
    pub purple: XmpHslChannel,
    pub magenta: XmpHslChannel,
}

impl XmpHsl {
    pub fn channels(&self) -> [XmpHslChannel; 8] {
        [
            self.red,
            self.orange,
            self.yellow,
            self.green,
            self.aqua,
            self.blue,
            self.purple,
            self.magenta,
        ]
    }
}

/// Parsed Lightroom `crs:` tone curves (normalized `0.0 ..= 1.0` control points).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct XmpToneCurves {
    pub name: Option<String>,
    pub rgb: Option<Vec<CurvePoint>>,
    pub red: Option<Vec<CurvePoint>>,
    pub green: Option<Vec<CurvePoint>>,
    pub blue: Option<Vec<CurvePoint>>,
    pub parametric_shadows: Option<f32>,
    pub parametric_darks: Option<f32>,
    pub parametric_lights: Option<f32>,
    pub parametric_highlights: Option<f32>,
}

/// Parsed Lightroom `crs:` XMP develop preset.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct XmpPreset {
    pub name: Option<String>,
    pub group: Option<String>,
    pub preset_type: Option<String>,
    pub uuid: Option<String>,
    pub version: Option<String>,
    pub process_version: Option<String>,
    pub has_settings: Option<bool>,
    /// Optional preset amount multiplier (1.0 = 100%).
    pub amount: Option<f32>,
    pub supports: XmpPresetSupports,

    // White balance
    pub white_balance: Option<String>,
    pub temperature: Option<f32>,
    pub tint: Option<f32>,
    pub incremental_temperature: Option<f32>,
    pub incremental_tint: Option<f32>,

    // Basic tone & presence
    pub exposure: Option<f32>,
    pub contrast: Option<f32>,
    pub highlights: Option<f32>,
    pub shadows: Option<f32>,
    pub whites: Option<f32>,
    pub blacks: Option<f32>,
    pub texture: Option<f32>,
    pub clarity: Option<f32>,
    pub dehaze: Option<f32>,
    pub vibrance: Option<f32>,
    pub saturation: Option<f32>,
    pub convert_to_grayscale: Option<bool>,

    // Color & curves
    pub hsl: XmpHsl,
    pub tone_curves: XmpToneCurves,
}

impl XmpPreset {
    /// Loads and parses a Lightroom `.xmp` preset file from disk.
    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path)
            .with_context(|| format!("reading XMP preset {}", path.display()))?;
        Self::parse(&text).with_context(|| format!("parsing XMP preset {}", path.display()))
    }

    /// Parses a Lightroom `crs:` XMP preset from an XML string.
    /// Empty input returns default preset; malformed XML returns an error.
    pub fn parse(xml: &str) -> Result<Self> {
        if xml.trim().is_empty() {
            return Ok(Self::default());
        }
        let doc = Document::parse(xml).map_err(|e| anyhow!("invalid XMP preset XML: {e}"))?;
        let desc = doc
            .descendants()
            .find(|n| is_tag(*n, RDF_NS, "Description"))
            .ok_or_else(|| anyhow!("missing rdf:Description in XMP preset"))?;

        let mut supports = XmpPresetSupports::default();
        if let Some(v) = get_crs_bool(desc, "SupportsAmount") {
            supports.amount = v;
        }
        if let Some(v) = get_crs_bool(desc, "SupportsAmount2") {
            supports.amount2 = v;
        }
        if let Some(v) = get_crs_bool(desc, "SupportsColor") {
            supports.color = v;
        }
        if let Some(v) = get_crs_bool(desc, "SupportsMonochrome") {
            supports.monochrome = v;
        }
        if let Some(v) = get_crs_bool(desc, "SupportsHighDynamicRange") {
            supports.high_dynamic_range = v;
        }
        if let Some(v) = get_crs_bool(desc, "SupportsNormalDynamicRange") {
            supports.normal_dynamic_range = v;
        }
        if let Some(v) = get_crs_bool(desc, "SupportsSceneReferred") {
            supports.scene_referred = v;
        }
        if let Some(v) = get_crs_bool(desc, "SupportsOutputReferred") {
            supports.output_referred = v;
        }

        let amount = get_crs_f32(desc, "Amount")
            .or_else(|| get_crs_f32(desc, "PresetAmount"))
            .map(|v| if v > 2.5 { v / 100.0 } else { v });

        let hsl = XmpHsl {
            red: parse_hsl_channel(desc, "Red"),
            orange: parse_hsl_channel(desc, "Orange"),
            yellow: parse_hsl_channel(desc, "Yellow"),
            green: parse_hsl_channel(desc, "Green"),
            aqua: parse_hsl_channel(desc, "Aqua"),
            blue: parse_hsl_channel(desc, "Blue"),
            purple: parse_hsl_channel(desc, "Purple"),
            magenta: parse_hsl_channel(desc, "Magenta"),
        };

        let tone_curves = XmpToneCurves {
            name: get_crs_str(desc, "ToneCurveName2012")
                .or_else(|| get_crs_str(desc, "ToneCurveName")),
            rgb: parse_tone_curve(desc, "ToneCurvePV2012")
                .or_else(|| parse_tone_curve(desc, "ToneCurve")),
            red: parse_tone_curve(desc, "ToneCurvePV2012Red"),
            green: parse_tone_curve(desc, "ToneCurvePV2012Green"),
            blue: parse_tone_curve(desc, "ToneCurvePV2012Blue"),
            parametric_shadows: get_crs_f32(desc, "ParametricShadows"),
            parametric_darks: get_crs_f32(desc, "ParametricDarks"),
            parametric_lights: get_crs_f32(desc, "ParametricLights"),
            parametric_highlights: get_crs_f32(desc, "ParametricHighlights"),
        };

        Ok(Self {
            name: get_crs_alt_or_str(desc, "Name"),
            group: get_crs_alt_or_str(desc, "Group"),
            preset_type: get_crs_str(desc, "PresetType"),
            uuid: get_crs_str(desc, "UUID"),
            version: get_crs_str(desc, "Version"),
            process_version: get_crs_str(desc, "ProcessVersion"),
            has_settings: get_crs_bool(desc, "HasSettings"),
            amount,
            supports,
            white_balance: get_crs_str(desc, "WhiteBalance"),
            temperature: get_crs_f32(desc, "Temperature"),
            tint: get_crs_f32(desc, "Tint"),
            incremental_temperature: get_crs_f32(desc, "IncrementalTemperature"),
            incremental_tint: get_crs_f32(desc, "IncrementalTint"),
            exposure: get_crs_f32(desc, "Exposure2012")
                .or_else(|| get_crs_f32(desc, "Exposure")),
            contrast: get_crs_f32(desc, "Contrast2012")
                .or_else(|| get_crs_f32(desc, "Contrast")),
            highlights: get_crs_f32(desc, "Highlights2012"),
            shadows: get_crs_f32(desc, "Shadows2012"),
            whites: get_crs_f32(desc, "Whites2012"),
            blacks: get_crs_f32(desc, "Blacks2012"),
            texture: get_crs_f32(desc, "Texture"),
            clarity: get_crs_f32(desc, "Clarity2012")
                .or_else(|| get_crs_f32(desc, "Clarity")),
            dehaze: get_crs_f32(desc, "Dehaze"),
            vibrance: get_crs_f32(desc, "Vibrance"),
            saturation: get_crs_f32(desc, "Saturation"),
            convert_to_grayscale: get_crs_bool(desc, "ConvertToGrayscale"),
            hsl,
            tone_curves,
        })
    }

    /// Returns the effective preset amount multiplier (`1.0` if unspecified).
    pub fn effective_amount(&self) -> f32 {
        self.amount.unwrap_or(1.0).clamp(0.0, 2.0)
    }

    /// Read-only conversion to Aurora's [`AdjustmentSet`] using `self.effective_amount()`.
    pub fn to_adjustment_set(&self) -> AdjustmentSet {
        self.to_adjustment_set_with_amount(self.effective_amount())
    }

    /// Read-only conversion to Aurora's [`AdjustmentSet`] scaled by `amount_multiplier`
    /// (typically `0.0 ..= 2.0`, where `1.0` is 100%).
    ///
    /// When `supports.amount` is `false`, scalable tone/color deltas keep `1.0` scale
    /// unless `amount_multiplier == 0.0`.
    pub fn to_adjustment_set_with_amount(&self, amount_multiplier: f32) -> AdjustmentSet {
        let raw_scale = amount_multiplier.clamp(0.0, 2.0);
        let scale = if self.supports.amount || raw_scale == 0.0 {
            raw_scale
        } else {
            1.0
        };

        let mut adj = AdjustmentSet::default();

        // White balance
        if let Some(ref wb_name) = self.white_balance {
            if let Some(mode) = WhiteBalanceMode::from_name(wb_name) {
                adj.white_balance.mode = mode;
            }
        }
        if let Some(temp) = self.temperature {
            adj.white_balance.kelvin = temp.clamp(2000.0, 50000.0);
            if self.white_balance.is_none() {
                adj.white_balance.mode = WhiteBalanceMode::Custom;
            }
        } else if let Some(inc_temp) = self.incremental_temperature {
            // Relative temperature delta in Lightroom (-100..=100) shifted around 5600K.
            adj.white_balance.kelvin = (5600.0 + inc_temp * 25.0 * scale).clamp(2000.0, 50000.0);
            adj.white_balance.mode = WhiteBalanceMode::Custom;
        }
        if let Some(tint) = self.tint {
            // Lightroom tint is -150..=150; Aurora tint is -50..=50.
            adj.white_balance.tint = (tint / 3.0).clamp(-50.0, 50.0);
        } else if let Some(inc_tint) = self.incremental_tint {
            adj.white_balance.tint = ((inc_tint / 3.0) * scale).clamp(-50.0, 50.0);
        }

        // Exposure / Contrast / Saturation
        if let Some(v) = self.exposure {
            adj.exposure.exposure = (v * scale).clamp(-4.0, 4.0);
        }
        if let Some(v) = self.contrast {
            // Lightroom contrast is -100..=100; Aurora contrast is -50..=50.
            adj.exposure.contrast = (v * 0.5 * scale).clamp(-50.0, 50.0);
        }
        if let Some(v) = self.saturation {
            adj.exposure.saturation = (v * scale).clamp(-100.0, 100.0);
        }
        if let Some(v) = self.vibrance {
            adj.extra
                .insert("crs:Vibrance".to_string(), (v * scale).clamp(-100.0, 100.0));
        }

        // HDR
        if let Some(v) = self.highlights {
            adj.hdr.highlights = (v * scale).clamp(-100.0, 100.0);
        }
        if let Some(v) = self.shadows {
            adj.hdr.shadows = (v * scale).clamp(-100.0, 100.0);
        }
        if let Some(v) = self.whites {
            adj.hdr.whites = (v * scale).clamp(-100.0, 100.0);
        }
        if let Some(v) = self.blacks {
            adj.hdr.blacks = (v * scale).clamp(-100.0, 100.0);
        }

        // Clarity / Structure / Dehaze
        if let Some(v) = self.clarity {
            adj.clarity.clarity = (v * scale).clamp(-100.0, 100.0);
        }
        if let Some(v) = self.texture {
            adj.clarity.structure = (v * scale).clamp(-100.0, 100.0);
        }
        if let Some(v) = self.dehaze {
            let scaled = (v * scale).clamp(-100.0, 100.0);
            adj.dehaze.amount = scaled.clamp(0.0, 100.0);
            if scaled < 0.0 {
                adj.extra.insert("crs:Dehaze".to_string(), scaled);
            }
        }

        // Grayscale
        if let Some(bw) = self.convert_to_grayscale {
            adj.black_white.enabled = bw;
        }

        // HSL -> ColorEditor.basic (8 sectors in matching order)
        for (i, ch) in self.hsl.channels().iter().enumerate() {
            if let Some(h) = ch.hue {
                // Lightroom Hue is -100..=100; Aurora HslShift.hue is -30..=30 degrees.
                adj.color_editor.basic[i].hue = (h * 0.3 * scale).clamp(-30.0, 30.0);
            }
            if let Some(s) = ch.saturation {
                adj.color_editor.basic[i].saturation = (s * scale).clamp(-100.0, 100.0);
            }
            if let Some(l) = ch.luminance {
                adj.color_editor.basic[i].lightness = (l * scale).clamp(-100.0, 100.0);
            }
        }

        // Tone curves scaled toward identity line y = x
        if let Some(ref pts) = self.tone_curves.rgb {
            adj.curves.rgb = scale_curve(pts, scale);
        }
        if let Some(ref pts) = self.tone_curves.red {
            adj.curves.red = scale_curve(pts, scale);
        }
        if let Some(ref pts) = self.tone_curves.green {
            adj.curves.green = scale_curve(pts, scale);
        }
        if let Some(ref pts) = self.tone_curves.blue {
            adj.curves.blue = scale_curve(pts, scale);
        }

        adj
    }
}

fn scale_curve(points: &[CurvePoint], scale: f32) -> Curve {
    let scaled: Vec<CurvePoint> = points
        .iter()
        .map(|p| {
            let y = (p.x + (p.y - p.x) * scale).clamp(0.0, 1.0);
            CurvePoint::new(p.x.clamp(0.0, 1.0), y)
        })
        .collect();
    Curve { points: scaled }
}

fn is_tag(node: Node<'_, '_>, ns: &str, local: &str) -> bool {
    node.is_element()
        && node.tag_name().name() == local
        && node.tag_name().namespace() == Some(ns)
}

fn get_crs_str(desc: Node<'_, '_>, name: &str) -> Option<String> {
    if let Some(attr) = desc
        .attributes()
        .find(|a| a.name() == name && a.namespace() == Some(CRS_NS))
    {
        return Some(attr.value().trim().to_string());
    }
    for child in desc.children().filter(|c| is_tag(*c, CRS_NS, name)) {
        if let Some(text) = child.text() {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
    }
    None
}

fn get_crs_alt_or_str(desc: Node<'_, '_>, name: &str) -> Option<String> {
    if let Some(child) = desc.children().find(|c| is_tag(*c, CRS_NS, name)) {
        for alt in child.children().filter(|c| is_tag(*c, RDF_NS, "Alt")) {
            if let Some(li) = alt.children().find(|c| is_tag(*c, RDF_NS, "li")) {
                if let Some(text) = li.text() {
                    let trimmed = text.trim();
                    if !trimmed.is_empty() {
                        return Some(trimmed.to_string());
                    }
                }
            }
        }
    }
    get_crs_str(desc, name)
}

fn get_crs_f32(desc: Node<'_, '_>, name: &str) -> Option<f32> {
    get_crs_str(desc, name)?.parse::<f32>().ok()
}

fn get_crs_bool(desc: Node<'_, '_>, name: &str) -> Option<bool> {
    let raw = get_crs_str(desc, name)?;
    match raw.to_ascii_lowercase().as_str() {
        "true" | "1" => Some(true),
        "false" | "0" => Some(false),
        _ => None,
    }
}

fn parse_hsl_channel(desc: Node<'_, '_>, color: &str) -> XmpHslChannel {
    XmpHslChannel {
        hue: get_crs_f32(desc, &format!("HueAdjustment{color}")),
        saturation: get_crs_f32(desc, &format!("SaturationAdjustment{color}")),
        luminance: get_crs_f32(desc, &format!("LuminanceAdjustment{color}")),
    }
}

fn parse_tone_curve(desc: Node<'_, '_>, name: &str) -> Option<Vec<CurvePoint>> {
    let elem = desc.children().find(|c| is_tag(*c, CRS_NS, name))?;
    let seq = elem.children().find(|c| is_tag(*c, RDF_NS, "Seq"))?;
    let mut pts = Vec::new();
    for li in seq.children().filter(|c| is_tag(*c, RDF_NS, "li")) {
        let text = li.text()?.trim();
        let mut parts = text.split(',');
        let x: f32 = parts.next()?.trim().parse().ok()?;
        let y: f32 = parts.next()?.trim().parse().ok()?;
        pts.push(CurvePoint::new((x / 255.0).clamp(0.0, 1.0), (y / 255.0).clamp(0.0, 1.0)));
    }
    (!pts.is_empty()).then_some(pts)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_LR_PRESET: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
      crs:PresetType="Normal"
      crs:Version="16.0"
      crs:ProcessVersion="15.4"
      crs:HasSettings="True"
      crs:SupportsAmount="True"
      crs:SupportsColor="True"
      crs:SupportsMonochrome="False"
      crs:SupportsHighDynamicRange="True"
      crs:WhiteBalance="Custom"
      crs:Temperature="6200"
      crs:Tint="+12"
      crs:Exposure2012="+0.80"
      crs:Contrast2012="+20"
      crs:Highlights2012="-40"
      crs:Shadows2012="+30"
      crs:Whites2012="+10"
      crs:Blacks2012="-15"
      crs:Texture="+15"
      crs:Clarity2012="+24"
      crs:Dehaze="+18"
      crs:Vibrance="+16"
      crs:Saturation="-10"
      crs:HueAdjustmentOrange="-10"
      crs:SaturationAdjustmentOrange="+20"
      crs:LuminanceAdjustmentOrange="+12"
      crs:HueAdjustmentBlue="+30"
      crs:ToneCurveName2012="Custom">
      <crs:Name>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">Warm Film Portrait</rdf:li>
        </rdf:Alt>
      </crs:Name>
      <crs:Group>
        <rdf:Alt>
          <rdf:li xml:lang="x-default">User Presets</rdf:li>
        </rdf:Alt>
      </crs:Group>
      <crs:ToneCurvePV2012>
        <rdf:Seq>
          <rdf:li>0, 15</rdf:li>
          <rdf:li>64, 54</rdf:li>
          <rdf:li>192, 206</rdf:li>
          <rdf:li>255, 245</rdf:li>
        </rdf:Seq>
      </crs:ToneCurvePV2012>
    </rdf:Description>
  </rdf:RDF>
</x:xmpmeta>"#;

    #[test]
    fn parses_typical_xmp_preset_and_maps_to_aurora() {
        let preset = XmpPreset::parse(SAMPLE_LR_PRESET).expect("parse preset");
        assert_eq!(preset.name.as_deref(), Some("Warm Film Portrait"));
        assert_eq!(preset.group.as_deref(), Some("User Presets"));
        assert!(preset.supports.amount);
        assert!(!preset.supports.monochrome);
        assert_eq!(preset.exposure, Some(0.8));
        assert_eq!(preset.highlights, Some(-40.0));
        assert_eq!(preset.tone_curves.rgb.as_ref().map(Vec::len), Some(4));

        let adj = preset.to_adjustment_set();
        assert_eq!(adj.white_balance.mode, WhiteBalanceMode::Custom);
        assert!((adj.white_balance.kelvin - 6200.0).abs() < 1e-4);
        assert!((adj.white_balance.tint - 4.0).abs() < 1e-4); // 12 / 3
        assert!((adj.exposure.exposure - 0.8).abs() < 1e-4);
        assert!((adj.exposure.contrast - 10.0).abs() < 1e-4); // 20 * 0.5
        assert!((adj.exposure.saturation - (-10.0)).abs() < 1e-4);
        assert_eq!(adj.extra.get("crs:Vibrance").copied(), Some(16.0));
        assert!((adj.hdr.highlights - (-40.0)).abs() < 1e-4);
        assert!((adj.hdr.shadows - 30.0).abs() < 1e-4);
        assert!((adj.clarity.clarity - 24.0).abs() < 1e-4);
        assert!((adj.clarity.structure - 15.0).abs() < 1e-4);
        assert!((adj.dehaze.amount - 18.0).abs() < 1e-4);
        // Orange is index 1, Blue is index 5
        assert!((adj.color_editor.basic[1].hue - (-3.0)).abs() < 1e-4);
        assert!((adj.color_editor.basic[1].saturation - 20.0).abs() < 1e-4);
        assert!((adj.color_editor.basic[1].lightness - 12.0).abs() < 1e-4);
        assert!((adj.color_editor.basic[5].hue - 9.0).abs() < 1e-4);
        assert_eq!(adj.curves.rgb.points.len(), 4);
    }

    #[test]
    fn scales_parameters_and_curves_by_amount() {
        let preset = XmpPreset::parse(SAMPLE_LR_PRESET).expect("parse preset");

        let half = preset.to_adjustment_set_with_amount(0.5);
        assert!((half.exposure.exposure - 0.4).abs() < 1e-4);
        assert!((half.exposure.contrast - 5.0).abs() < 1e-4);
        assert!((half.hdr.highlights - (-20.0)).abs() < 1e-4);
        assert!((half.color_editor.basic[1].saturation - 10.0).abs() < 1e-4);
        // Curve first point (0, 15/255) scaled by 0.5 -> (0, 7.5/255)
        assert!((half.curves.rgb.points[0].y - (7.5 / 255.0)).abs() < 1e-4);

        let zero = preset.to_adjustment_set_with_amount(0.0);
        assert_eq!(zero.exposure.exposure, 0.0);
        assert_eq!(zero.hdr.shadows, 0.0);
        assert!(zero.curves.rgb.is_identity());
    }

    #[test]
    fn handles_missing_fields_and_errors_safely() {
        let empty = XmpPreset::parse("   ").expect("empty ok");
        assert_eq!(empty.to_adjustment_set(), AdjustmentSet::default());

        let minimal = r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
          <rdf:Description rdf:about="" xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" crs:SupportsAmount="False" crs:Exposure2012="1.0"/>
        </rdf:RDF>"#;
        let p = XmpPreset::parse(minimal).expect("minimal ok");
        assert!(!p.supports.amount);
        // Because SupportsAmount=False, non-zero multiplier keeps 1.0
        let adj = p.to_adjustment_set_with_amount(0.5);
        assert!((adj.exposure.exposure - 1.0).abs() < 1e-4);
        assert_eq!(adj.hdr.highlights, 0.0);

        assert!(XmpPreset::parse("<not-valid-xml").is_err());
    }
}
