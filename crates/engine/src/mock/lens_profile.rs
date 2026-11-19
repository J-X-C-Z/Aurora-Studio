//! Capture One's own per-lens calibration, normalized so one table serves every
//! capture sensor.
//!
//! `lensDatabase.cdx` stores `(r_um, Δr_um)` on the sensor plane of the body the
//! profile was measured on. Both are divided by that body's half diagonal
//! (`R_ref`), never by the last node's radius: a table that stops short of the
//! corner would otherwise inflate Δr by `R_ref / r_last`. A pixel at `t` of the
//! capture's half diagonal then reads the table at the same `t`.

use crate::{Exif, LensCorrection, LensProfile, ManufacturerCurves};

/// The EXIF facts that pick one table entry out of a profile.
#[derive(Clone, Debug)]
pub(crate) struct LensShot {
    pub(crate) lens: String,
    pub(crate) focal_mm: f32,
    pub(crate) aperture: f32,
    /// Mount of the camera that took the frame; the manufacturer profile's
    /// curves are normalized against its half diagonal.
    pub(crate) mount: Option<&'static str>,
}

impl LensShot {
    pub(crate) fn from_exif(exif: &Exif) -> Option<Self> {
        let lens = exif.lens.clone()?;
        let mount = exif.camera_model.as_deref().and_then(crate::lensdb::mount_for_camera);
        Some(Self {
            lens,
            focal_mm: exif.focal_length_mm?,
            aperture: exif.aperture.unwrap_or(0.0),
            mount,
        })
    }
}

/// Radius-normalized curves: `t` is the fraction of the reference half diagonal
/// (1.0 is the corner), distortion and CA shifts are fractions of the same half
/// diagonal, light falloff stays in EV.
#[derive(Clone, Debug, Default)]
pub(crate) struct LensWarp {
    pub(crate) distortion: Vec<(f32, f32)>,
    pub(crate) falloff: Vec<(f32, f32)>,
    pub(crate) red: Vec<(f32, f32)>,
    pub(crate) blue: Vec<(f32, f32)>,
}

/// The lens table's CA shifts are in reference-image pixels and the table does
/// not record that image's resolution; a 24 MP full-frame half diagonal is the
/// typical measuring body.
///
/// shortcut: CA shifts assume a 3600 px reference half diagonal, revisit if a
/// profile ever states its own pixel pitch.
const REFERENCE_HALF_DIAGONAL_PX: f32 = 3600.0;

/// Gate for the Manufacturer Profile's CA table; see [`LensWarp::from_catalog`].
const APPLY_CATALOG_CA: bool = false;

impl LensWarp {
    pub(crate) fn new(profile: &LensProfile, focal: f32, aperture: f32) -> Self {
        let reference_um = reference_um(profile);
        if !(reference_um > 0.0) {
            return Self::default();
        }
        let distortion = profile
            .distortion_at(focal)
            .map(|e| normalize(&e.shift, reference_um))
            .unwrap_or_default();
        let falloff = profile
            .falloff_at(focal, aperture)
            .map(|e| e.falloff.iter().map(|&(r, ev)| (r / reference_um, ev)).collect())
            .unwrap_or_default();
        let (red, blue) = profile
            .chroma_at(focal, aperture)
            .map(|e| {
                let px = |points: &[(f32, f32)]| -> Vec<(f32, f32)> {
                    normalize(points, reference_um)
                        .into_iter()
                        .map(|(t, v)| (t, v / REFERENCE_HALF_DIAGONAL_PX))
                        .collect()
                };
                (px(&e.red_shift), px(&e.blue_shift))
            })
            .unwrap_or_default();
        Self { distortion, falloff, red, blue }
    }

    /// The variant's own curves for Capture One's built-in Manufacturer
    /// Profile: `(r_um, Δr_um)` distortion and `(r_um, shift_um)` CA, measured
    /// on the capture's sensor. Both are normalized by that sensor's half
    /// diagonal `hypot(W, H) / 2`, which the mount names; without a mount
    /// there is no scale to trust, so no warp is invented.
    pub(crate) fn from_catalog(curves: &ManufacturerCurves, mount: Option<&str>) -> Self {
        let Some(reference_um) = mount.and_then(crate::mount_half_diagonal_um) else {
            return Self::default();
        };
        Self {
            distortion: normalize(&curves.distortion, reference_um),
            falloff: Vec::new(),
            // shortcut: the catalog's CA table is (r_um, shift_um), but its
            // sign convention is not the one C1's per-channel lens stage uses
            // (`p_040754496` is a demosaic block curve). Applying it
            // unvalidated added visible fringing on high-frequency detail, so
            // the table is parsed and carried but only applied once a real
            // C1 render confirms the direction.
            red: if APPLY_CATALOG_CA { normalize(&curves.ca_red, reference_um) } else { Vec::new() },
            blue: if APPLY_CATALOG_CA { normalize(&curves.ca_blue, reference_um) } else { Vec::new() },
        }
    }

    /// Linear interpolation, clamped to the end values. Used for falloff, whose
    /// EV curve Capture One also evaluates as a plain LUT.
    pub(crate) fn sample(table: &[(f32, f32)], t: f32) -> f32 {
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

    /// Monotone cubic Hermite through `(r, Δr)` nodes, clamped to the end
    /// values. Capture One builds the same shape-preserving curve from the
    /// table's nodes (`0x1428130`, monotone tangents) and evaluates it on every
    /// pixel; a linear read cuts the corners off wide-angle tables.
    pub(crate) fn sample_curve(table: &[(f32, f32)], t: f32) -> f32 {
        let Some(&(x0, y0)) = table.first() else { return 0.0 };
        if !t.is_finite() || t <= x0 {
            return y0;
        }
        let Some(&(x_last, y_last)) = table.last() else { return y0 };
        if t >= x_last {
            return y_last;
        }
        let i = table.partition_point(|&(x, _)| x <= t).saturating_sub(1);
        let (xa, ya) = table[i];
        let (xb, yb) = table[i + 1];
        let h = xb - xa;
        if h <= 1e-4 {
            return ya;
        }
        let s = (t - xa) / h;
        let s2 = s * s;
        let s3 = s2 * s;
        let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
        let h10 = s3 - 2.0 * s2 + s;
        let h01 = -2.0 * s3 + 3.0 * s2;
        let h11 = s3 - s2;
        let (m0, m1) = (tangent(table, i), tangent(table, i + 1));
        ya * h00 + h * m0 * h10 + yb * h01 + h * m1 * h11
    }
}

/// Half diagonal of the sensor the profile was measured on, in micrometres.
///
/// The mount names it; for a mount we do not know, the largest table radius is
/// the best available stand-in.
///
/// shortcut: cropType 101 (Scaled) rescales the curve by
/// `max(W/curveW, H/curveH)` for captures whose sensor differs from the
/// measuring body; the profile does not carry `curveW/curveH`, so that scale is
/// not applied. Revisit when the lens database parse keeps the curve canvas.
fn reference_um(profile: &LensProfile) -> f32 {
    crate::mount_half_diagonal_um(&profile.mount).unwrap_or_else(|| {
        profile
            .distortion
            .iter()
            .flat_map(|e| e.shift.iter().map(|p| p.0))
            .chain(profile.falloff.iter().flat_map(|e| e.falloff.iter().map(|p| p.0)))
            .chain(profile.chroma.iter().flat_map(|e| e.red_shift.iter().map(|p| p.0)))
            .fold(0.0f32, f32::max)
    })
}

/// Fritsch–Butland tangent at node `i`, zero at a sign change so the curve
/// cannot overshoot a local extremum.
fn tangent(table: &[(f32, f32)], i: usize) -> f32 {
    let last = table.len().saturating_sub(1);
    if table.len() < 2 {
        return 0.0;
    }
    if i == 0 {
        let (xa, ya) = table[0];
        let (xb, yb) = table[1];
        return if xb - xa > 1e-4 { (yb - ya) / (xb - xa) } else { 0.0 };
    }
    if i >= last {
        let (xa, ya) = table[last - 1];
        let (xb, yb) = table[last];
        return if xb - xa > 1e-4 { (yb - ya) / (xb - xa) } else { 0.0 };
    }
    let (xa, ya) = table[i - 1];
    let (xb, yb) = table[i];
    let (xc, yc) = table[i + 1];
    let (h0, h1) = (xb - xa, xc - xb);
    if h0 <= 1e-4 || h1 <= 1e-4 {
        return 0.0;
    }
    let (d0, d1) = ((yb - ya) / h0, (yc - yb) / h1);
    if d0 * d1 <= 0.0 {
        return 0.0;
    }
    let (w1, w2) = (2.0 * h1 + h0, h1 + 2.0 * h0);
    (w1 + w2) / (w1 / d0 + w2 / d1)
}

/// `t = r / R_ref` (0 at the centre, 1 at the reference corner) and the value
/// in fractions of that same reference.
fn normalize(points: &[(f32, f32)], reference_um: f32) -> Vec<(f32, f32)> {
    points.iter().map(|&(r, v)| (r / reference_um, v / reference_um)).collect()
}

/// The profile EXIF matching picks for this capture, resolved once per source.
pub(crate) fn auto_warp(shot: &LensShot) -> Option<LensWarp> {
    let db = crate::lensdb::default_database()?;
    let profile = crate::lensdb::detect_lens(db, "", &shot.lens, shot.focal_mm)?;
    Some(LensWarp::new(profile, shot.focal_mm, shot.aperture))
}

/// The profile the user chose by name or GUID; a plain string match, no scan cost.
pub(crate) fn named_warp(name: &str, shot: &LensShot) -> Option<LensWarp> {
    let db = crate::lensdb::default_database()?;
    let profile = db.iter().find(|p| p.full_name == name || p.short_name == name || p.id == name)?;
    Some(LensWarp::new(profile, shot.focal_mm, shot.aperture))
}

/// Which warp the current lens settings ask for: the named profile when the user
/// picked one, otherwise the EXIF match already resolved for this source.
pub(crate) fn warp_for(
    lens: &LensCorrection,
    shot: Option<&LensShot>,
    auto: Option<&LensWarp>,
) -> Option<LensWarp> {
    match lens.profile.as_deref() {
        Some(name) if crate::lensdb::builtin_profile(name).is_some() => lens
            .manufacturer
            .as_ref()
            .map(|curves| {
                let mount = lens.mount.as_deref().or_else(|| shot.and_then(|s| s.mount));
                LensWarp::from_catalog(curves, mount)
            })
            .filter(|warp| !warp.distortion.is_empty() || !warp.red.is_empty()),
        Some(name) => shot.and_then(|shot| named_warp(name, shot)),
        None => auto.cloned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChromaEntry, DistortionEntry, FalloffEntry, LensKind, ManufacturerCurves};

    /// Real Canon EF 11-24 values at 12 mm: the table is not monotone.
    const EF_11_24_12MM: &[(f32, f32)] = &[
        (0.0, 0.0),
        (7298.0, 21.9),
        (13774.0, 43.2),
        (17622.0, -35.8),
        (19075.0, -103.2),
        (20293.0, -151.7),
        (21178.0, -83.5),
        (21580.0, 34.5),
    ];

    fn profile() -> LensProfile {
        LensProfile {
            id: "TEST".into(),
            full_name: "Test 11-24mm".into(),
            short_name: "Test 11-24".into(),
            brand: "Canon".into(),
            mount: "Canon EF".into(),
            kind: LensKind::Zoom,
            focal_min: Some(11.0),
            focal_max: Some(24.0),
            aperture_min: Some(4.0),
            aperture_max: Some(22.0),
            detect_id: None,
            distortion: vec![DistortionEntry {
                focal_length: 12.0,
                focus_distance: "INF".into(),
                shift: EF_11_24_12MM.to_vec(),
            }],
            falloff: vec![FalloffEntry {
                focal_length: 12.0,
                aperture: 4.0,
                focus_distance: "INF".into(),
                falloff: vec![(0.0, 0.0), (21633.0, -1.5)],
            }],
            chroma: vec![ChromaEntry {
                focal_length: 12.0,
                aperture: 4.0,
                red_shift: vec![(0.0, 0.0), (21633.0, 3.6)],
                blue_shift: vec![(0.0, 0.0), (21633.0, -2.4)],
            }],
        }
    }

    #[test]
    fn sample_interpolates_between_points() {
        let table = [(0.0, 0.0), (0.5, 10.0), (1.0, 20.0)];
        assert!((LensWarp::sample(&table, 0.25) - 5.0).abs() < 1e-4);
        assert!((LensWarp::sample(&table, 0.75) - 15.0).abs() < 1e-4);
    }

    #[test]
    fn sample_clamps_past_the_last_point() {
        let table = [(0.0, 0.0), (0.5, 10.0)];
        assert_eq!(LensWarp::sample(&table, 9.0), 10.0);
        assert_eq!(LensWarp::sample(&table, -1.0), 0.0);
        assert_eq!(LensWarp::sample(&[], 0.5), 0.0);
    }

    #[test]
    fn non_monotone_table_is_interpolated_as_given() {
        let warp = LensWarp::new(&profile(), 12.0, 4.0);
        // Halfway between 43.2 and -35.8, not a fitted curve.
        let mid = (13774.0 + 17622.0) * 0.5;
        let t = mid / 21932.0;
        let expected = 43.2 / 21932.0 + (-35.8 / 21932.0 - 43.2 / 21932.0) * 0.5;
        assert!((LensWarp::sample(&warp.distortion, t) - expected).abs() < 1e-6);
    }

    #[test]
    fn warp_normalizes_against_the_measuring_bodys_half_diagonal() {
        let warp = LensWarp::new(&profile(), 12.0, 4.0);
        let last = warp.distortion.last().copied().unwrap();
        // Canon EF R_ref = 21,932 µm, not the last node at 21,580 µm.
        assert!((last.0 - 21580.0 / 21932.0).abs() < 1e-6);
        assert!((last.1 - 34.5 / 21932.0).abs() < 1e-6);
        let falloff = warp.falloff.last().copied().unwrap();
        assert!((falloff.0 - 21633.0 / 21932.0).abs() < 1e-6, "got {}", falloff.0);
        assert!((falloff.1 - -1.5).abs() < 1e-6);
    }

    #[test]
    fn hermite_matches_a_straight_line_and_stays_between_nodes() {
        let line = [(0.0, 0.0), (0.5, 5.0), (1.0, 10.0)];
        for step in 0..=20 {
            let t = step as f32 / 20.0;
            assert!((LensWarp::sample_curve(&line, t) - 10.0 * t).abs() < 1e-4);
        }
        // A non-monotone table must not overshoot its own nodes.
        let bump = [(0.0, 0.0), (0.5, 10.0), (1.0, 0.0)];
        for step in 0..=100 {
            let v = LensWarp::sample_curve(&bump, step as f32 / 100.0);
            assert!((0.0..=10.0).contains(&v), "{v} at {}", step as f32 / 100.0);
        }
        assert_eq!(LensWarp::sample_curve(&line, -1.0), 0.0);
        assert_eq!(LensWarp::sample_curve(&line, 2.0), 10.0);
        assert_eq!(LensWarp::sample_curve(&[], 0.5), 0.0);
    }

    /// Needs the real database: `AURORA_LENS_DB=<path to decoded lensDatabase.cdx>`.
    #[test]
    #[ignore]
    fn real_database_builds_a_warp() {
        let shot = LensShot {
            lens: "Canon EF 11-24mm f/4L USM".into(),
            focal_mm: 12.0,
            aperture: 4.0,
            mount: Some("canon ef"),
        };
        let lens = LensCorrection { use_profile: true, ..Default::default() };
        let start = std::time::Instant::now();
        let auto = auto_warp(&shot);
        println!("first resolve: {:.3} ms", start.elapsed().as_secs_f64() * 1000.0);
        let start = std::time::Instant::now();
        for _ in 0..100 {
            std::hint::black_box(warp_for(&lens, Some(&shot), auto.as_ref()));
        }
        println!("warp_for: {:.3} ms per call", start.elapsed().as_secs_f64() * 10.0);
        let warp = warp_for(&lens, Some(&shot), auto.as_ref()).expect("EF 11-24 warp");
        println!(
            "distortion={} falloff={} red={} blue={} corner={:?}",
            warp.distortion.len(),
            warp.falloff.len(),
            warp.red.len(),
            warp.blue.len(),
            warp.distortion.last()
        );
        assert!(warp.distortion.len() >= 4);
        assert!(warp.falloff.len() >= 2);
    }

    fn catalog_curves() -> ManufacturerCurves {
        ManufacturerCurves {
            distortion: vec![(0.0, 0.0), (14097.2, -526.0558), (23742.6, 146.6017)],
            ca_red: vec![(0.0, 0.0), (13355.2, -3.9107), (23742.6, -7.4068)],
            ca_blue: vec![(0.0, 0.0), (13355.2, -2.0419), (23742.6, -4.7671)],
        }
    }

    #[test]
    fn catalog_curves_use_the_sensors_half_diagonal() {
        let warp = LensWarp::from_catalog(&catalog_curves(), Some("nikon z"));
        let (t, dr) = warp.distortion[1];
        assert!((t - 14097.2 / 21633.0).abs() < 1e-6);
        assert!((dr + 526.0558 / 21633.0).abs() < 1e-9);
        // The CA table is parsed but gated: its sign is unverified.
        assert!(warp.red.is_empty() && warp.blue.is_empty());
        assert!((catalog_curves().ca_red[2].1 + 7.4068).abs() < 1e-9);
        // No mount means no trustworthy scale: do not invent one.
        assert!(LensWarp::from_catalog(&catalog_curves(), None).distortion.is_empty());
        assert!(LensWarp::from_catalog(&catalog_curves(), Some("unknown mount")).distortion.is_empty());
    }

    #[test]
    fn manufacturer_guid_uses_the_variants_curves_but_unknown_guids_stay_empty() {
        let shot = LensShot {
            lens: "NIKKOR Z 24-120mm f/4 S".into(),
            focal_mm: 120.0,
            aperture: 4.0,
            mount: Some("nikon z"),
        };
        let mut lens = LensCorrection {
            use_profile: true,
            distortion: 100.0,
            manufacturer: Some(catalog_curves()),
            ..Default::default()
        };
        lens.profile = Some(crate::lensdb::MANUFACTURER_PROFILE_ID.to_string());
        assert!(warp_for(&lens, Some(&shot), None).is_some_and(|w| !w.distortion.is_empty()));
        // The mount on the adjustment alone is enough: no EXIF shot needed.
        lens.mount = Some("nikon z".into());
        assert!(warp_for(&lens, None, None).is_some_and(|w| !w.distortion.is_empty()));

        lens.profile = Some("6E1F8E5B-0000-0000-0000-000000000000".to_string());
        assert!(warp_for(&lens, Some(&shot), None).is_none());
    }
}
