//! Lens correction built from Capture One's own calibration tables.
//!
//! The tables in `lensDatabase.cdx` store radii in micrometres on the sensor
//! plane of the body the profile was measured on. A capture pixel is therefore
//! converted with
//!
//! `r_um = (r_px / image_half_diagonal_px) * sensor_half_diagonal_um`
//!
//! and every table is sampled at `t = r_um / sensor_half_diagonal_um`, i.e. at
//! the fraction of the capture's own half diagonal. With no explicit sensor the
//! mount's half diagonal from the lens database analysis is used (`Nikon F`
//! 21,633 µm), falling back to the profile's largest table radius.
//!
//! Distortion is handled as a radial term only, sampled from the measured
//! spline rather than fitted to a Brown-Conrady model.

use crate::LensProfile;
use serde::Deserialize;
use rayon::prelude::*;

/// Half diagonals of the sensors the profiles were measured on, in micrometres.
///
/// Full-frame and the mounts that share it use 21,633 µm (36 × 24 mm,
/// diagonal 43.267 mm). The values with an explicit source come from
/// `lensdb-format.md`: Canon EF 21,932, Fujifilm G 27,453, Micro Four Thirds
/// 11,504 µm. The rest are the nominal sensor sizes of those mounts.
pub fn mount_half_diagonal_um(mount: &str) -> Option<f32> {
    let value = match mount.trim().to_ascii_lowercase().as_str() {
        "canon ef" => 21_932.0,
        "nikon f" | "nikon z" | "sony e" | "leica m" | "leica l" | "leica sl" => 21_633.0,
        "canon ef-s" | "nikon f dx" | "fujifilm x" | "sony e aps-c" => 14_145.0,
        "fujifilm g" => 27_453.0,
        "micro four thirds" => 11_504.0,
        "hasselblad xcd" | "hasselblad x" => 27_390.0,
        "leica s" => 27_020.0,
        "phaseone 645df" | "mamiya 645" => 27_700.0,
        _ => return None,
    };
    Some(value)
}

/// Some bodies record CA shifts in reference-image pixels and the database does
/// not state that image's resolution.
///
/// shortcut: CA shifts assume a 3600 px reference half diagonal, revisit when a
/// profile or sidecar records the measuring resolution.
pub const REFERENCE_HALF_DIAGONAL_PX: f32 = 3600.0;

/// Radius-normalized correction curves: `t` is the fraction of the half
/// diagonal, distortion and CA values are fractions of the same half diagonal,
/// falloff values stay in EV (negative means the lens lost light).
#[derive(Debug, Clone, Default)]
pub struct LensCorrector {
    pub distortion: Vec<(f32, f32)>,
    pub falloff: Vec<(f32, f32)>,
    pub red: Vec<(f32, f32)>,
    pub blue: Vec<(f32, f32)>,
}

#[derive(Debug, Clone, Deserialize)]
struct BuiltinLens {
    model: String,
    mount: String,
    distortion: Vec<BuiltinDistortion>,
    vignetting: Vec<BuiltinVignetting>,
    ca: Vec<BuiltinChroma>,
}

#[derive(Debug, Clone, Deserialize)]
struct BuiltinDistortion {
    focal_length: f32,
    focus_distance: String,
    points: Vec<(f32, f32)>,
}

#[derive(Debug, Clone, Deserialize)]
struct BuiltinVignetting {
    focal_length: f32,
    aperture: f32,
    focus_distance: String,
    points: Vec<(f32, f32)>,
}

#[derive(Debug, Clone, Deserialize)]
struct BuiltinChroma {
    focal_length: f32,
    aperture: f32,
    red_points: Vec<(f32, f32)>,
    blue_points: Vec<(f32, f32)>,
}

fn builtin_lenses() -> Option<&'static Vec<BuiltinLens>> {
    static DB: std::sync::OnceLock<Option<Vec<BuiltinLens>>> = std::sync::OnceLock::new();
    DB.get_or_init(|| {
        // Private calibration stays local; public builds can use the installed lens database.
        let path = std::env::var_os("LUMEN_LENS_DB").map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("testdata/lens_correct.json"));
        match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice(&bytes) {
                Ok(db) => Some(db),
                Err(err) => { log::warn!("lens calibration {}: {err}", path.display()); None }
            },
            Err(_) => None,
        }
    }).as_ref()
}

fn builtin_profile(model: &str) -> Option<&'static BuiltinLens> {
    let model = model.trim().to_ascii_lowercase();
    builtin_lenses()?.iter().find(|l| l.model.to_ascii_lowercase() == model)
}

impl LensCorrector {
    fn from_builtin(
        lens: &BuiltinLens,
        focal_mm: f32,
        aperture: f32,
        focus_m: Option<f32>,
        sensor_half_diag_um: f32,
    ) -> Self {
        let reference_um = if sensor_half_diag_um > 0.0 {
            sensor_half_diag_um
        } else {
            mount_half_diagonal_um(&lens.mount).unwrap_or_else(|| {
                lens.distortion.iter().flat_map(|e| e.points.iter().map(|p| p.0))
                    .chain(lens.vignetting.iter().flat_map(|e| e.points.iter().map(|p| p.0)))
                    .chain(lens.ca.iter().flat_map(|e| e.red_points.iter().map(|p| p.0)))
                    .chain(lens.ca.iter().flat_map(|e| e.blue_points.iter().map(|p| p.0)))
                    .fold(0.0, f32::max)
            })
        };
        if !(reference_um > 0.0) {
            return Self::identity();
        }
        let distortion = lens.distortion.iter()
            .min_by(|a, b| table_score(a.focal_length, 0.0, &a.focus_distance, focal_mm, 0.0, focus_m)
                .partial_cmp(&table_score(b.focal_length, 0.0, &b.focus_distance, focal_mm, 0.0, focus_m)).unwrap_or(std::cmp::Ordering::Equal))
            .map(|e| e.points.iter().map(|&(r, v)| (r / reference_um, v / reference_um)).collect())
            .unwrap_or_default();
        let falloff = lens.vignetting.iter()
            .min_by(|a, b| table_score(a.focal_length, a.aperture, &a.focus_distance, focal_mm, aperture, focus_m)
                .partial_cmp(&table_score(b.focal_length, b.aperture, &b.focus_distance, focal_mm, aperture, focus_m)).unwrap_or(std::cmp::Ordering::Equal))
            .map(|e| e.points.iter().map(|&(r, v)| (r / reference_um, v)).collect())
            .unwrap_or_default();
        let ca = lens.ca.iter()
            .min_by(|a, b| table_score(a.focal_length, a.aperture, "", focal_mm, aperture, focus_m)
                .partial_cmp(&table_score(b.focal_length, b.aperture, "", focal_mm, aperture, focus_m)).unwrap_or(std::cmp::Ordering::Equal));
        let (red, blue) = ca.map(|e| {
            let scale = |points: &[(f32, f32)]| -> Vec<(f32, f32)> {
                points.iter().map(|&(r, px)| (r / reference_um, px / REFERENCE_HALF_DIAGONAL_PX)).collect()
            };
            (scale(&e.red_points), scale(&e.blue_points))
        }).unwrap_or_default();
        Self { distortion, falloff, red, blue }
    }

    /// Build the correction from the compact built-in lens database subset.
    pub fn from_builtin_model(
        model: &str,
        focal_mm: f32,
        aperture: f32,
        focus_m: Option<f32>,
        sensor_half_diag_um: f32,
    ) -> Self {
        let Some(lens) = builtin_profile(model) else { return Self::identity(); };
        Self::from_builtin(lens, focal_mm, aperture, focus_m, sensor_half_diag_um)
    }
}

impl LensCorrector {
    pub fn identity() -> Self {
        Self::default()
    }

    pub fn is_identity(&self) -> bool {
        !has_signal(&self.distortion) && !has_signal(&self.falloff) && !has_signal(&self.red) && !has_signal(&self.blue)
    }

    /// Build the normalized curves for one shot.
    ///
    /// `focus_m` is the focus distance in metres (`None` for unknown/INF).
    /// `sensor_half_diag_um` comes from the capture sensor; pass 0 to use the
    /// mount or the profile's own reference radius.
    pub fn from_profile(
        profile: &LensProfile,
        focal_mm: f32,
        aperture: f32,
        focus_m: Option<f32>,
        sensor_half_diag_um: f32,
    ) -> Self {
        let reference_um = if sensor_half_diag_um > 0.0 {
            sensor_half_diag_um
        } else {
            mount_half_diagonal_um(&profile.mount).unwrap_or_else(|| profile_reference_um(profile))
        };
        if !(reference_um > 0.0) {
            return Self::identity();
        }

        let distortion = profile
            .distortion
            .iter()
            .min_by(|a, b| {
                distortion_score(a, focal_mm, focus_m)
                    .partial_cmp(&distortion_score(b, focal_mm, focus_m))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|entry| {
                entry
                    .shift
                    .iter()
                    .map(|&(r, dr)| (r / reference_um, dr / reference_um))
                    .collect()
            })
            .unwrap_or_default();

        let falloff = profile
            .falloff
            .iter()
            .min_by(|a, b| {
                table_score(a.focal_length, a.aperture, &a.focus_distance, focal_mm, aperture, focus_m)
                    .partial_cmp(&table_score(b.focal_length, b.aperture, &b.focus_distance, focal_mm, aperture, focus_m))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|entry| entry.falloff.iter().map(|&(r, ev)| (r / reference_um, ev)).collect())
            .unwrap_or_default();

        let ca = profile
            .chroma
            .iter()
            .min_by(|a, b| {
                table_score(a.focal_length, a.aperture, "", focal_mm, aperture, focus_m)
                    .partial_cmp(&table_score(b.focal_length, b.aperture, "", focal_mm, aperture, focus_m))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        let (red, blue) = ca
            .map(|entry| {
                let scale = |points: &[(f32, f32)]| -> Vec<(f32, f32)> {
                    points
                        .iter()
                        .map(|&(r, px)| (r / reference_um, px / REFERENCE_HALF_DIAGONAL_PX))
                        .collect()
                };
                (scale(&entry.red_shift), scale(&entry.blue_shift))
            })
            .unwrap_or_default();

        Self { distortion, falloff, red, blue }
    }

    /// Look a lens up by model name and build its curves. An unknown lens, or a
    /// profile without calibration data, yields the identity.
    pub fn for_model(
        lenses: &[LensProfile],
        model: &str,
        focal_mm: f32,
        aperture: f32,
        focus_m: Option<f32>,
        sensor_half_diag_um: f32,
    ) -> Self {
        if let Some(lens) = builtin_profile(model) {
            Self::from_builtin(lens, focal_mm, aperture, focus_m, sensor_half_diag_um)
        } else {
            crate::lensdb::detect_lens(lenses, "", model, focal_mm)
                .map(|profile| Self::from_profile(profile, focal_mm, aperture, focus_m, sensor_half_diag_um))
                .unwrap_or_else(Self::identity)
        }
    }

    /// Largest vignetting loss in the table, in EV, as a positive correction.
    pub fn max_vignette_ev(&self) -> f32 {
        self.falloff.iter().map(|&(_, ev)| -ev).fold(0.0, f32::max)
    }

    /// Largest radial displacement inside the frame, in pixels.
    pub fn max_distortion_px(&self, image_half_diag_px: f32) -> f32 {
        let mut worst = 0.0f32;
        for step in 0..=1024 {
            let t = step as f32 / 1024.0;
            worst = worst.max(sample(&self.distortion, t).abs());
        }
        worst * image_half_diag_px
    }

    /// Apply distortion, lateral CA and vignetting to a tightly packed RGBA8
    /// buffer. Pixels that map outside the source become transparent.
    pub fn apply_rgba(&self, buf: &mut [u8], width: usize, height: usize) {
        if self.is_identity() || width == 0 || height == 0 || buf.len() != width * height * 4 {
            return;
        }
        let src = buf.to_vec();
        let cx = width as f32 * 0.5;
        let cy = height as f32 * 0.5;
        let half_diag = ((width * width + height * height) as f32).sqrt() * 0.5;
        let has_red = has_signal(&self.red);
        let has_blue = has_signal(&self.blue);
        let has_falloff = has_signal(&self.falloff);
        buf.par_chunks_mut(width * 4).enumerate().for_each(|(y, row)| {
            for x in 0..width {
                let i = x * 4;
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cy;
                let r_px = (dx * dx + dy * dy).sqrt();
                let t = r_px / half_diag;
                // The table says how far the lens pushed a point outward, so
                // the source for an output pixel sits at `t - shift`.
                let t_src = t - sample(&self.distortion, t);
                let k = if r_px > 1e-4 { t_src / t } else { 1.0 };
                let sx = cx + dx * k;
                let sy = cy + dy * k;
                if sx < 0.0 || sy < 0.0 || sx > width as f32 || sy > height as f32 {
                    row[i..i + 4].fill(0);
                    continue;
                }
                let mut px = sample_bilinear(&src, width, height, sx, sy);
                if has_red && r_px > 1e-4 {
                    let k = (t - sample(&self.red, t)) / t;
                    px[0] = sample_bilinear(&src, width, height, cx + dx * k, cy + dy * k)[0];
                }
                if has_blue && r_px > 1e-4 {
                    let k = (t - sample(&self.blue, t)) / t;
                    px[2] = sample_bilinear(&src, width, height, cx + dx * k, cy + dy * k)[2];
                }
                if has_falloff {
                    let gain = 2f32.powf(-sample(&self.falloff, t));
                    px[0] *= gain;
                    px[1] *= gain;
                    px[2] *= gain;
                }
                for c in 0..3 {
                    row[i + c] = px[c].clamp(0.0, 255.0).round() as u8;
                }
                row[i + 3] = px[3].clamp(0.0, 255.0).round() as u8;
            }
        });
    }
}

/// Linear interpolation between table points, clamped outside the table.
pub fn sample(table: &[(f32, f32)], t: f32) -> f32 {
    let Some((&(r0, v0), rest)) = table.split_first() else { return 0.0 };
    if !t.is_finite() || t <= r0 {
        return v0;
    }
    let mut prev = (r0, v0);
    for &(r, v) in rest {
        if t <= r {
            let span = r - prev.0;
            return if span <= 0.0 { v } else { prev.1 + (v - prev.1) * (t - prev.0) / span };
        }
        prev = (r, v);
    }
    prev.1
}

fn has_signal(table: &[(f32, f32)]) -> bool {
    table.iter().any(|&(_, v)| v.abs() > 1e-9)
}

fn profile_reference_um(profile: &LensProfile) -> f32 {
    profile
        .distortion
        .iter()
        .flat_map(|e| e.shift.iter().map(|p| p.0))
        .chain(profile.falloff.iter().flat_map(|e| e.falloff.iter().map(|p| p.0)))
        .chain(profile.chroma.iter().flat_map(|e| e.red_shift.iter().map(|p| p.0)))
        .fold(0.0, f32::max)
}

fn distortion_score(entry: &crate::DistortionEntry, focal: f32, focus: Option<f32>) -> f32 {
    (entry.focal_length - focal).abs() * 100.0 + focus_penalty(&entry.focus_distance, focus)
}

fn table_score(
    entry_focal: f32,
    entry_aperture: f32,
    entry_focus: &str,
    focal: f32,
    aperture: f32,
    focus: Option<f32>,
) -> f32 {
    (entry_focal - focal).abs() * 100.0
        + (entry_aperture - aperture).abs() * 10.0
        + focus_penalty(entry_focus, focus)
}

fn focus_penalty(entry: &str, focus: Option<f32>) -> f32 {
    let Some(shot) = focus else { return 0.0 };
    let entry = entry.trim();
    if entry.eq_ignore_ascii_case("INF") {
        return if shot >= 10.0 { 0.0 } else { 1.0 };
    }
    match entry.parse::<f32>() {
        Ok(value) if shot < 10.0 => (value - shot).abs().min(1.0),
        Ok(_) => 1.0,
        Err(_) => 0.0,
    }
}

fn sample_bilinear(src: &[u8], width: usize, height: usize, x: f32, y: f32) -> [f32; 4] {
    if !x.is_finite() || !y.is_finite() || x < 0.0 || y < 0.0 || x > width as f32 || y > height as f32 {
        return [0.0; 4];
    }
    let fx = (x - 0.5).clamp(0.0, width as f32 - 1.0);
    let fy = (y - 0.5).clamp(0.0, height as f32 - 1.0);
    let x0 = fx.floor() as usize;
    let y0 = fy.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let wx = fx - x0 as f32;
    let wy = fy - y0 as f32;
    let at = |px: usize, py: usize, c: usize| src[(py * width + px) * 4 + c] as f32;
    let mut out = [0.0; 4];
    for c in 0..4 {
        let top = at(x0, y0, c) * (1.0 - wx) + at(x1, y0, c) * wx;
        let bottom = at(x0, y1, c) * (1.0 - wx) + at(x1, y1, c) * wx;
        out[c] = top * (1.0 - wy) + bottom * wy;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChromaEntry, DistortionEntry, FalloffEntry, LensKind};

    const HALF_UM: f32 = 21_633.0;

    fn profile(distortion: Vec<(f32, f32)>, falloff: Vec<(f32, f32)>, red: Vec<(f32, f32)>, blue: Vec<(f32, f32)>) -> LensProfile {
        LensProfile {
            id: "TEST".into(),
            full_name: "Nikon NIKKOR Z 24-120mm f/4 S".into(),
            short_name: "NIKKOR Z 24-120mm f/4 S".into(),
            brand: "Nikon".into(),
            mount: "Nikon Z".into(),
            kind: LensKind::Zoom,
            focal_min: Some(24.0),
            focal_max: Some(120.0),
            aperture_min: Some(4.0),
            aperture_max: Some(22.0),
            detect_id: None,
            distortion: vec![DistortionEntry {
                focal_length: 24.0,
                focus_distance: "INF".into(),
                shift: distortion,
            }],
            falloff: vec![FalloffEntry {
                focal_length: 24.0,
                aperture: 4.0,
                focus_distance: "INF".into(),
                falloff,
            }],
            chroma: vec![ChromaEntry {
                focal_length: 24.0,
                aperture: 4.0,
                red_shift: red,
                blue_shift: blue,
            }],
        }
    }

    fn gray(width: usize, height: usize, value: u8) -> Vec<u8> {
        [value, value, value, 255].repeat(width * height)
    }

    #[test]
    fn neutral_curves_leave_pixels_untouched() {
        let p = profile(
            vec![(0.0, 0.0), (HALF_UM, 0.0)],
            vec![(0.0, 0.0), (HALF_UM, 0.0)],
            vec![(0.0, 0.0), (HALF_UM, 0.0)],
            vec![(0.0, 0.0), (HALF_UM, 0.0)],
        );
        let c = LensCorrector::from_profile(&p, 24.0, 4.0, None, HALF_UM);
        assert!(c.is_identity());
        let mut buf = gray(32, 24, 77);
        let before = buf.clone();
        c.apply_rgba(&mut buf, 32, 24);
        assert_eq!(buf, before);
    }

    #[test]
    fn vignetting_keeps_centre_and_brightens_corners() {
        let p = profile(vec![], vec![(0.0, 0.0), (HALF_UM, -1.0)], vec![], vec![]);
        let c = LensCorrector::from_profile(&p, 24.0, 4.0, None, HALF_UM);
        assert!((c.max_vignette_ev() - 1.0).abs() < 1e-6);
        let mut buf = gray(64, 64, 100);
        c.apply_rgba(&mut buf, 64, 64);
        let centre = buf[(32 * 64 + 32) * 4];
        let corner = buf[0];
        assert!((centre as i32 - 100).abs() <= 1, "centre changed: {centre}");
        assert!((corner as i32 - 200).abs() <= 2, "corner gain wrong: {corner}");
    }

    fn render_distorted(ideal: &[u8], w: usize, h: usize, table: &[(f32, f32)]) -> Vec<u8> {
        let cx = w as f32 * 0.5;
        let cy = h as f32 * 0.5;
        let half = ((w * w + h * h) as f32).sqrt() * 0.5;
        let mut out = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 + 0.5 - cx;
                let dy = y as f32 + 0.5 - cy;
                let t = (dx * dx + dy * dy).sqrt() / half;
                // Forward warp: solve r_i + shift(r_i) = r_d so the correction
                // below is the exact inverse.
                let mut t_src = t;
                for _ in 0..4 {
                    t_src = t + sample(table, t_src);
                }
                let k = if t > 1e-4 { t_src / t } else { 1.0 };
                let px = sample_bilinear(ideal, w, h, cx + dx * k, cy + dy * k);
                for c in 0..4 {
                    out[(y * w + x) * 4 + c] = px[c].round() as u8;
                }
            }
        }
        out
    }

    #[test]
    fn distortion_straightens_a_synthetic_grid() {
        let (w, h) = (257, 257);
        let mut ideal = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let on_line = [w / 4, w / 2, 3 * w / 4].iter().any(|&l| x.abs_diff(l) <= 1);
                let i = (y * w + x) * 4;
                if on_line {
                    ideal[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
                } else {
                    ideal[i + 3] = 255;
                }
            }
        }
        let table_um = [(0.0, 0.0), (0.5 * HALF_UM, -0.03 * HALF_UM), (HALF_UM, -0.15 * HALF_UM)];
        let table_t: Vec<(f32, f32)> = table_um.iter().map(|&(r, v)| (r / HALF_UM, v / HALF_UM)).collect();
        let distorted = render_distorted(&ideal, w, h, &table_t);
        let p = profile(table_um.to_vec(), vec![], vec![], vec![]);
        let c = LensCorrector::from_profile(&p, 24.0, 4.0, None, HALF_UM);
        let mut corrected = distorted.clone();
        c.apply_rgba(&mut corrected, w, h);

        let line_x = |buf: &[u8], y: usize| -> f32 {
            let mut sum = 0.0;
            let mut weight = 0.0;
            for x in 0..w / 2 - 8 {
                let v = buf[(y * w + x) * 4] as f32;
                sum += x as f32 * v;
                weight += v;
            }
            sum / weight.max(1.0)
        };
        let bow = |buf: &[u8]| -> f32 {
            let mid = line_x(buf, h / 2);
            (20..h - 20).map(|y| (line_x(buf, y) - mid).abs()).fold(0.0, f32::max)
        };
        assert!(bow(&distorted) > 2.0, "synthetic grid did not bend: {}", bow(&distorted));
        assert!(bow(&corrected) < 0.5, "grid still bent after correction: {}", bow(&corrected));

        let mut diff = 0.0f64;
        for i in 0..w * h {
            for c in 0..3 {
                diff += (corrected[i * 4 + c] as f64 - ideal[i * 4 + c] as f64).abs();
            }
        }
        let mean = diff / (w * h * 3) as f64;
        assert!(mean < 3.0, "mean channel error {mean}");
    }

    #[test]
    fn chromatic_aberration_scales_the_right_directions() {
        let (w, h) = (129, 129);
        let half = ((w * w + h * h) as f32).sqrt() * 0.5;
        let mut buf = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 + 0.5 - w as f32 * 0.5;
                let dy = y as f32 + 0.5 - h as f32 * 0.5;
                let v = (255.0 * (dx * dx + dy * dy).sqrt() / half).clamp(0.0, 255.0) as u8;
                let i = (y * w + x) * 4;
                buf[i] = v;
                buf[i + 1] = v;
                buf[i + 2] = v;
                buf[i + 3] = 255;
            }
        }
        let before = buf.clone();
        // +360 px of reference shift = +0.1 of the half diagonal.
        let p = profile(
            vec![],
            vec![],
            vec![(0.0, 360.0), (HALF_UM, 360.0)],
            vec![(0.0, -360.0), (HALF_UM, -360.0)],
        );
        let c = LensCorrector::from_profile(&p, 24.0, 4.0, None, HALF_UM);
        c.apply_rgba(&mut buf, w, h);
        let probe = ((h / 2 + 20) * w + w / 2 + 20) * 4;
        assert!(buf[probe] < before[probe], "positive red shift must sample inward");
        assert!(buf[probe + 2] > before[probe + 2], "negative blue shift must sample outward");
    }

    #[test]
    fn out_of_frame_samples_are_transparent_not_panics() {
        let p = profile(vec![(0.0, 0.0), (HALF_UM, -0.5 * HALF_UM)], vec![], vec![], vec![]);
        let c = LensCorrector::from_profile(&p, 24.0, 4.0, None, HALF_UM);
        let mut buf = gray(48, 48, 200);
        c.apply_rgba(&mut buf, 48, 48);
        assert_eq!(buf.len(), 48 * 48 * 4);
        assert_eq!(buf[3], 0, "corner should be outside the source");
    }

    #[test]
    fn unknown_lens_returns_identity() {
        let db = vec![profile(vec![(0.0, 0.0), (HALF_UM, -1000.0)], vec![], vec![], vec![])];
        let c = LensCorrector::for_model(&db, "Unknown 10mm", 24.0, 4.0, None, HALF_UM);
        assert!(c.is_identity());
        let mut buf = gray(16, 16, 42);
        let before = buf.clone();
        c.apply_rgba(&mut buf, 16, 16);
        assert_eq!(buf, before);
    }

    #[test]
    fn selects_the_nearest_focal_aperture_and_focus() {
        let mut p = profile(vec![(0.0, 0.0), (HALF_UM, -1000.0)], vec![], vec![], vec![]);
        p.falloff = vec![
            FalloffEntry {
                focal_length: 24.0,
                aperture: 4.0,
                focus_distance: "INF".into(),
                falloff: vec![(HALF_UM, -0.5)],
            },
            FalloffEntry {
                focal_length: 24.0,
                aperture: 8.0,
                focus_distance: "INF".into(),
                falloff: vec![(HALF_UM, -0.1)],
            },
            FalloffEntry {
                focal_length: 70.0,
                aperture: 4.0,
                focus_distance: "INF".into(),
                falloff: vec![(HALF_UM, -0.9)],
            },
            FalloffEntry {
                focal_length: 24.0,
                aperture: 4.0,
                focus_distance: "0.45".into(),
                falloff: vec![(HALF_UM, -0.7)],
            },
        ];
        let near = LensCorrector::from_profile(&p, 25.0, 4.0, Some(0.5), HALF_UM);
        assert!((near.max_vignette_ev() - 0.7).abs() < 1e-6);
        let far = LensCorrector::from_profile(&p, 25.0, 4.0, Some(100.0), HALF_UM);
        assert!((far.max_vignette_ev() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn micrometre_to_pixel_conversion_is_explicit() {
        let p = profile(vec![(0.0, 0.0), (HALF_UM, -0.1 * HALF_UM)], vec![], vec![], vec![]);
        let c = LensCorrector::from_profile(&p, 24.0, 4.0, None, HALF_UM);
        assert!((c.max_distortion_px(1000.0) - 100.0).abs() < 1e-3);
        let half = ((6000f32 * 6000.0 + 4000.0 * 4000.0).sqrt()) * 0.5;
        assert!((c.max_distortion_px(half) - 360.6).abs() < 0.5);
    }


    #[test]
    fn builtin_unknown_lens_is_identity() {
        let c = LensCorrector::from_builtin_model("Nikon NIKKOR Z 24-120mm f/4 S", 24.0, 4.0, None, HALF_UM);
        assert!(c.is_identity());
        let mut buf = gray(16, 16, 42);
        let before = buf.clone();
        c.apply_rgba(&mut buf, 16, 16);
        assert_eq!(buf, before);
    }


    /// Real database check; set `LUMEN_LENS_DB` to a decoded `lensDatabase.cdx`.
    #[test]
    #[ignore]
    fn real_nikkor_24_120_reports_its_limits() {
        let db = crate::lensdb::default_database().expect("set LUMEN_LENS_DB");
        let p = db
            .iter()
            .find(|p| p.full_name.contains("24-120mm f/4G"))
            .expect("24-120 profile");
        let c = LensCorrector::from_profile(p, 24.0, 4.0, Some(1e6), 0.0);
        let half = ((6000f32 * 6000.0 + 4000.0 * 4000.0).sqrt()) * 0.5;
        println!(
            "{} @24mm f/4: max vignetting {:.3} EV, max distortion {:.1} px at 6000x4000",
            p.full_name,
            c.max_vignette_ev(),
            c.max_distortion_px(half)
        );
        assert!(c.max_vignette_ev() > 0.5);
        assert!(c.max_distortion_px(half) > 10.0);
    }

    #[cfg(feature = "external-fixtures")]
    include!("../local-fixtures/lens_correct-tests.rs");
}

