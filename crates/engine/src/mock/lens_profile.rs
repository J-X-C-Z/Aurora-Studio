//! Capture One's own per-lens calibration, normalized so one table serves every
//! capture sensor.
//!
//! `lensDatabase.cdx` stores each curve against the half diagonal of the body it
//! was measured on, so both the radius and the amount are divided by that table's
//! largest radius here. A pixel at `t` of the capture's half diagonal then reads
//! the table at the same `t` regardless of sensor size.

use crate::{Exif, LensCorrection, LensProfile};

/// The EXIF facts that pick one table entry out of a profile.
#[derive(Clone, Debug)]
pub(crate) struct LensShot {
    pub(crate) lens: String,
    pub(crate) focal_mm: f32,
    pub(crate) aperture: f32,
}

impl LensShot {
    pub(crate) fn from_exif(exif: &Exif) -> Option<Self> {
        let lens = exif.lens.clone()?;
        Some(Self { lens, focal_mm: exif.focal_length_mm?, aperture: exif.aperture.unwrap_or(0.0) })
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

impl LensWarp {
    pub(crate) fn new(profile: &LensProfile, focal: f32, aperture: f32) -> Self {
        // One reference radius for every table, so distortion and falloff agree
        // on where the corner is.
        let reference_um = profile
            .distortion
            .iter()
            .flat_map(|e| e.shift.iter().map(|p| p.0))
            .chain(profile.falloff.iter().flat_map(|e| e.falloff.iter().map(|p| p.0)))
            .chain(profile.chroma.iter().flat_map(|e| e.red_shift.iter().map(|p| p.0)))
            .fold(0.0f32, f32::max);
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

    /// Linear interpolation, clamped to the end values. The tables are not
    /// monotone, so this interpolates them as given rather than fitting a model.
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
}

/// `t = r / reference_um` (0 at the centre, 1 at the reference corner) and the
/// value in fractions of that same reference.
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
pub(crate) fn warp_for(lens: &LensCorrection, shot: &LensShot, auto: Option<&LensWarp>) -> Option<LensWarp> {
    match lens.profile.as_deref() {
        Some(name) => named_warp(name, shot),
        None => auto.cloned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChromaEntry, DistortionEntry, FalloffEntry, LensKind};

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
        let t = mid / 21633.0;
        let expected = 43.2 / 21633.0 + (-35.8 / 21633.0 - 43.2 / 21633.0) * 0.5;
        assert!((LensWarp::sample(&warp.distortion, t) - expected).abs() < 1e-6);
    }

    #[test]
    fn warp_normalizes_against_the_profiles_reference_radius() {
        let warp = LensWarp::new(&profile(), 12.0, 4.0);
        let last = warp.distortion.last().copied().unwrap();
        assert!((last.0 - 21580.0 / 21633.0).abs() < 1e-6);
        assert!((last.1 - 34.5 / 21633.0).abs() < 1e-6);
        let falloff = warp.falloff.last().copied().unwrap();
        assert!((falloff.0 - 1.0).abs() < 1e-6, "falloff ends at the corner, got {}", falloff.0);
        assert!((falloff.1 - -1.5).abs() < 1e-6);
    }

    /// Needs the real database: `LUMEN_LENS_DB=<path to decoded lensDatabase.cdx>`.
    #[test]
    #[ignore]
    fn real_database_builds_a_warp() {
        let shot = LensShot { lens: "Canon EF 11-24mm f/4L USM".into(), focal_mm: 12.0, aperture: 4.0 };
        let lens = LensCorrection { use_profile: true, ..Default::default() };
        let start = std::time::Instant::now();
        let auto = auto_warp(&shot);
        println!("first resolve: {:.3} ms", start.elapsed().as_secs_f64() * 1000.0);
        let start = std::time::Instant::now();
        for _ in 0..100 {
            std::hint::black_box(warp_for(&lens, &shot, auto.as_ref()));
        }
        println!("warp_for: {:.3} ms per call", start.elapsed().as_secs_f64() * 10.0);
        let warp = warp_for(&lens, &shot, auto.as_ref()).expect("EF 11-24 warp");
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
}
