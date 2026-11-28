//! Adobe Lightroom / Camera Raw plain-text `.lcp` lens profile parser and
//! multi-dimensional (focal length, focus distance, aperture) interpolator.
//!
//! Uses `roxmltree` for zero-copy XML parsing and evaluates Brown's radial
//! distortion model `r' = r * (1 + k1*r^2 + k2*r^4 + k3*r^6)` and radial
//! vignette polynomial `V = 1 + v1*r^2 + v2*r^4 + v3*r^6`.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct LcpNode {
    pub focal_length: f32,
    pub focus_distance: f32,
    pub aperture: f32,
    pub image_width: f32,
    pub image_length: f32,
    pub focal_length_x: f32,
    pub focal_length_y: f32,
    pub radial_k: [f32; 3],
    pub vignette_v: [f32; 3],
    pub ca_rg_k: [f32; 3],
    pub ca_by_k: [f32; 3],
}

impl Default for LcpNode {
    fn default() -> Self {
        Self {
            focal_length: 50.0,
            focus_distance: 10.0,
            aperture: 4.0,
            image_width: 6000.0,
            image_length: 4000.0,
            focal_length_x: 1.0,
            focal_length_y: 1.0,
            radial_k: [0.0; 3],
            vignette_v: [0.0; 3],
            ca_rg_k: [0.0; 3],
            ca_by_k: [0.0; 3],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct LcpProfile {
    pub name: String,
    pub make: String,
    pub model: String,
    pub lens: String,
    pub nodes: Vec<LcpNode>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LcpSolution {
    pub radial_k: [f32; 3],
    pub vignette_v: [f32; 3],
    pub ca_rg_k: [f32; 3],
    pub ca_by_k: [f32; 3],
    pub focal_length_x: f32,
}

impl LcpProfile {
    /// Parses a plain-text `.lcp` XML/XMP file.
    pub fn parse_str(xml: &str) -> Result<Self> {
        let doc = roxmltree::Document::parse(xml).context("parsing LCP XML")?;
        let mut name = String::new();
        let mut make = String::new();
        let mut model = String::new();
        let mut lens = String::new();
        let mut nodes = Vec::new();

        for node in doc.descendants() {
            if !node.is_element() {
                continue;
            }
            if node.tag_name().name() == "Description" {
                let mut parsed = LcpNode::default();
                let mut has_lens_metrics = false;

                for attr in node.attributes() {
                    let k = attr.name();
                    let v = attr.value().trim();
                    match k {
                        "ProfileName" if name.is_empty() => name = v.to_string(),
                        "Make" if make.is_empty() => make = v.to_string(),
                        "Model" if model.is_empty() => model = v.to_string(),
                        "Lens" if lens.is_empty() => lens = v.to_string(),
                        "FocalLength" => {
                            if let Ok(x) = v.parse::<f32>() {
                                parsed.focal_length = x;
                                has_lens_metrics = true;
                            }
                        }
                        "FocusDistance" => {
                            if let Ok(x) = v.parse::<f32>() {
                                parsed.focus_distance = x.max(0.01);
                            }
                        }
                        "ApertureValue" => {
                            if let Ok(x) = v.parse::<f32>() {
                                parsed.aperture = x.max(0.5);
                            }
                        }
                        "ImageWidth" => {
                            if let Ok(x) = v.parse::<f32>() {
                                parsed.image_width = x.max(1.0);
                            }
                        }
                        "ImageLength" => {
                            if let Ok(x) = v.parse::<f32>() {
                                parsed.image_length = x.max(1.0);
                            }
                        }
                        "FocalLengthX" => {
                            if let Ok(x) = v.parse::<f32>() {
                                parsed.focal_length_x = x.max(1e-4);
                            }
                        }
                        "FocalLengthY" => {
                            if let Ok(x) = v.parse::<f32>() {
                                parsed.focal_length_y = x.max(1e-4);
                            }
                        }
                        "RadialDistortParam1" => {
                            if let Ok(x) = v.parse::<f32>() {
                                parsed.radial_k[0] = x;
                                has_lens_metrics = true;
                            }
                        }
                        "RadialDistortParam2" => {
                            if let Ok(x) = v.parse::<f32>() {
                                parsed.radial_k[1] = x;
                            }
                        }
                        "RadialDistortParam3" => {
                            if let Ok(x) = v.parse::<f32>() {
                                parsed.radial_k[2] = x;
                            }
                        }
                        "VignetteModelParam1" => {
                            if let Ok(x) = v.parse::<f32>() {
                                parsed.vignette_v[0] = x;
                                has_lens_metrics = true;
                            }
                        }
                        "VignetteModelParam2" => {
                            if let Ok(x) = v.parse::<f32>() {
                                parsed.vignette_v[1] = x;
                            }
                        }
                        "VignetteModelParam3" => {
                            if let Ok(x) = v.parse::<f32>() {
                                parsed.vignette_v[2] = x;
                            }
                        }
                        _ => {}
                    }
                }

                // Also inspect direct child elements (some LCPs store PerspectiveModel / VignetteModel as child tags).
                for child in node.children().filter(|n| n.is_element()) {
                    let cname = child.tag_name().name();
                    for attr in child.attributes() {
                        if let Ok(val) = attr.value().trim().parse::<f32>() {
                            match (cname, attr.name()) {
                                (_, "RadialDistortParam1") => {
                                    parsed.radial_k[0] = val;
                                    has_lens_metrics = true;
                                }
                                (_, "RadialDistortParam2") => parsed.radial_k[1] = val,
                                (_, "RadialDistortParam3") => parsed.radial_k[2] = val,
                                (_, "VignetteModelParam1") => {
                                    parsed.vignette_v[0] = val;
                                    has_lens_metrics = true;
                                }
                                (_, "VignetteModelParam2") => parsed.vignette_v[1] = val,
                                (_, "VignetteModelParam3") => parsed.vignette_v[2] = val,
                                (_, "FocalLengthX") => parsed.focal_length_x = val.max(1e-4),
                                (_, "FocalLengthY") => parsed.focal_length_y = val.max(1e-4),
                                _ => {}
                            }
                        }
                    }
                }

                if has_lens_metrics {
                    nodes.push(parsed);
                }
            }
        }

        Ok(Self {
            name,
            make,
            model,
            lens,
            nodes,
        })
    }

    pub fn load(path: &Path) -> Result<Self> {
        let text = fs::read_to_string(path).with_context(|| format!("reading LCP {}", path.display()))?;
        Self::parse_str(&text)
    }

    /// Interpolates distortion, vignette and CA coefficients in log-space over
    /// `(focal_mm, focus_m, aperture_f)` using inverse-distance weighting.
    pub fn interpolate(&self, focal_mm: f32, focus_m: f32, aperture_f: f32) -> Option<LcpSolution> {
        if self.nodes.is_empty() {
            return None;
        }
        if self.nodes.len() == 1 {
            let n = &self.nodes[0];
            return Some(LcpSolution {
                radial_k: n.radial_k,
                vignette_v: n.vignette_v,
                ca_rg_k: n.ca_rg_k,
                ca_by_k: n.ca_by_k,
                focal_length_x: n.focal_length_x,
            });
        }

        let lf = focal_mm.max(1.0).ln();
        let ld = focus_m.max(0.05).ln();
        let la = aperture_f.max(0.7).ln();

        let mut w_sum = 0.0f32;
        let mut rk = [0.0f32; 3];
        let mut vv = [0.0f32; 3];
        let mut rg = [0.0f32; 3];
        let mut by = [0.0f32; 3];
        let mut flx = 0.0f32;

        for node in &self.nodes {
            let df = (node.focal_length.max(1.0).ln() - lf) * 4.0;
            let dd = (node.focus_distance.max(0.05).ln() - ld) * 0.6;
            let da = (node.aperture.max(0.7).ln() - la) * 1.2;
            let dist2 = df * df + dd * dd + da * da;
            if dist2 < 1e-8 {
                return Some(LcpSolution {
                    radial_k: node.radial_k,
                    vignette_v: node.vignette_v,
                    ca_rg_k: node.ca_rg_k,
                    ca_by_k: node.ca_by_k,
                    focal_length_x: node.focal_length_x,
                });
            }
            let w = 1.0 / (dist2 + 1e-4);
            w_sum += w;
            for i in 0..3 {
                rk[i] += node.radial_k[i] * w;
                vv[i] += node.vignette_v[i] * w;
                rg[i] += node.ca_rg_k[i] * w;
                by[i] += node.ca_by_k[i] * w;
            }
            flx += node.focal_length_x * w;
        }

        let inv = 1.0 / w_sum.max(1e-8);
        Some(LcpSolution {
            radial_k: rk.map(|v| v * inv),
            vignette_v: vv.map(|v| v * inv),
            ca_rg_k: rg.map(|v| v * inv),
            ca_by_k: by.map(|v| v * inv),
            focal_length_x: flx * inv,
        })
    }
}

/// Brown radial distortion multiplier `1 + k1*r^2 + k2*r^4 + k3*r^6` where `r`
/// is normalized by `(FocalLengthX * ImageWidth)`.
#[inline]
pub fn brown_radial_scale(r: f32, k: [f32; 3]) -> f32 {
    let r2 = r * r;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    1.0 + k[0] * r2 + k[1] * r4 + k[2] * r6
}

/// Radial vignette gain `1 / max(1 + v1*r^2 + v2*r^4 + v3*r^6, 0.25)` for falloff recovery.
#[inline]
pub fn lcp_vignette_gain(r: f32, v: [f32; 3]) -> f32 {
    let r2 = r * r;
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let falloff = (1.0 + v[0] * r2 + v[1] * r4 + v[2] * r6).max(0.25);
    1.0 / falloff
}

/// Finds the best matching `.lcp` file under `dir` for `(make, lens)`.
pub fn find_lcp_in_dir(dir: &Path, make: &str, lens: &str) -> Option<(PathBuf, LcpProfile)> {
    let entries = fs::read_dir(dir).ok()?;
    let make_l = make.trim().to_ascii_lowercase();
    let lens_l = lens.trim().to_ascii_lowercase();
    for entry in entries.flatten() {
        let p = entry.path();
        if p.extension().and_then(|e| e.to_str()).is_some_and(|e| e.eq_ignore_ascii_case("lcp")) {
            if let Ok(prof) = LcpProfile::load(&p) {
                let p_make = prof.make.to_ascii_lowercase();
                let p_lens = prof.lens.to_ascii_lowercase();
                if (!make_l.is_empty() && !p_make.is_empty() && p_make.contains(&make_l))
                    && (!lens_l.is_empty() && (p_lens.contains(&lens_l) || lens_l.contains(&p_lens)))
                {
                    return Some((p, prof));
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_LCP: &str = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description
      xmlns:stCamera="http://ns.adobe.com/photoshop/1.0/camera-profile"
      stCamera:ProfileName="Nikon Z 24-70mm f/2.8 S"
      stCamera:Make="NIKON CORPORATION"
      stCamera:Model="NIKON Z 6_3"
      stCamera:Lens="NIKKOR Z 24-70mm f/2.8 S"
      stCamera:FocalLength="24.0"
      stCamera:FocusDistance="3.0"
      stCamera:ApertureValue="2.8"
      stCamera:ImageWidth="6000"
      stCamera:ImageLength="4000"
      stCamera:FocalLengthX="0.67"
      stCamera:FocalLengthY="0.67"
      stCamera:RadialDistortParam1="-0.12"
      stCamera:RadialDistortParam2="0.03"
      stCamera:RadialDistortParam3="-0.005"
      stCamera:VignetteModelParam1="-0.35"
      stCamera:VignetteModelParam2="0.10"
      stCamera:VignetteModelParam3="0.0" />
    <rdf:Description
      xmlns:stCamera="http://ns.adobe.com/photoshop/1.0/camera-profile"
      stCamera:FocalLength="70.0"
      stCamera:FocusDistance="3.0"
      stCamera:ApertureValue="2.8"
      stCamera:FocalLengthX="1.95"
      stCamera:FocalLengthY="1.95"
      stCamera:RadialDistortParam1="0.08"
      stCamera:RadialDistortParam2="-0.01"
      stCamera:RadialDistortParam3="0.0"
      stCamera:VignetteModelParam1="-0.18"
      stCamera:VignetteModelParam2="0.04"
      stCamera:VignetteModelParam3="0.0" />
  </rdf:RDF>
</x:xmpmeta>"#;

    #[test]
    fn parses_lcp_and_interpolates_across_focal_lengths() {
        let prof = LcpProfile::parse_str(SAMPLE_LCP).unwrap();
        assert_eq!(prof.nodes.len(), 2);
        assert_eq!(prof.lens, "NIKKOR Z 24-70mm f/2.8 S");

        let wide = prof.interpolate(24.0, 3.0, 2.8).unwrap();
        assert!((wide.radial_k[0] - (-0.12)).abs() < 1e-4);

        let mid = prof.interpolate(45.0, 3.0, 2.8).unwrap();
        assert!(mid.radial_k[0] > -0.12 && mid.radial_k[0] < 0.08, "{:?}", mid.radial_k);
        assert!(lcp_vignette_gain(0.5, wide.vignette_v) > 1.0);
        assert!(brown_radial_scale(0.5, wide.radial_k) < 1.0);
    }
}
