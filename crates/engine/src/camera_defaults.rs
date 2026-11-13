//! Per-frame camera defaults for the parameter sliders.
//!
//! Capture One fills its `IB_F_*` container from two sources: `coreConfig.cdx`
//! (per-camera ranges and ISO ladder, [`crate::CameraDefaults`]) and the frame
//! itself. This module covers the second half: values that can be derived from
//! the EXIF exposure triangle and the Nikon white-balance gains.
//!
//! Exposure is **scene luminance**, not a correction: for a metered frame the
//! camera has already turned the scene into N/t/ISO, so `EV100` is the same
//! number whatever the settings and C1's `Exposure` slider stays at 0. That is
//! what C1's `coreConfig.cdx` says (`exposure_default val="0.0"`), and the
//! reference Z6_3 frame stores 0 even though EXIF says -4/3 EV compensation.
//! [`ev100`] is therefore reported (and used to reject a nonsense triple), not
//! copied into the slider.
use crate::param_meta::ParamOverrides;
use crate::rawinfo::RawInfoBlock;

/// Reference ISO of the EV scale, by definition (`EV100`).
pub const EV_REFERENCE_ISO: f32 = 100.0;

/// `EV100 = log2(N^2 / t) - log2(ISO / 100)`.
///
/// Assumptions: `t` is the EXIF exposure time in seconds, `N` the f-number
/// (not T-stop), ISO is the arithmetic EXIF value, and no ND filter is in the
/// path. For the reference frame (1/40 s, f/4, ISO 5000) this gives 3.68,
/// matching exiftool's composite `LightValue: 3.7`.
pub fn ev100(info: &RawInfoBlock) -> Option<f32> {
    let t = info.get("IB_F_EXP_SHUTTERSPEED")? as f32;
    let n = info.get("IB_F_EXP_APERTURE")? as f32;
    let iso = info.iso.or_else(|| info.get("IB_I_ISO").map(|v| v as u32))? as f32;
    if !(t.is_finite() && n.is_finite() && iso.is_finite()) || t <= 0.0 || n <= 0.0 || iso <= 0.0 {
        return None;
    }
    Some((n * n / t).log2() - (iso / EV_REFERENCE_ISO).log2())
}

/// White balance from the Nikon `WB_RBGGLevels` gains, through the same
/// log-ratio model the engine uses for `.copreset` white balance
/// ([`crate::gains_to_kelvin_tint`]). Reusing it keeps the info block and the
/// preset path on one definition of kelvin/tint.
fn apply_white_balance(info: &RawInfoBlock, o: &mut ParamOverrides) {
    let gain = |key: &str| info.get(key).filter(|v| *v > 0.0).map(|v| v as f32);
    if let (Some(r), Some(g), Some(b)) = (
        gain("IB_F_WBRED"),
        gain("IB_F_WBGREEN"),
        gain("IB_F_WBBLUE"),
    ) {
        let (kelvin, tint) = crate::gains_to_kelvin_tint([r, g, b]);
        o.set_default("WhiteBalanceTemperature", kelvin);
        o.set_default("WhiteBalanceTint", tint);
    }
}

/// Unsharp-mask amount ladder, copied from `coreConfig.cdx` for the Z6_3:
/// 180 below ISO 3200, 160 at 3200, 140 at 6400, 120 at 25600, 100 at 51200.
fn sharpening_for(iso: u32) -> f32 {
    match iso {
        0..=3199 => 180.0,
        3200..=6399 => 160.0,
        6400..=25599 => 140.0,
        25600..=51199 => 120.0,
        _ => 100.0,
    }
}

/// Luminance noise reduction. C1's `BASEDEFAULT` is 50; this keeps 50 at the
/// reference ISO band and fades towards 20 in clean light and 80 at the top,
/// roughly one step per ISO doubling of the sharpening ladder.
fn noise_reduction_for(iso: u32) -> f32 {
    match iso {
        0..=400 => 20.0,
        401..=1600 => 35.0,
        1601..=6400 => 50.0,
        6401..=25600 => 65.0,
        _ => 80.0,
    }
}

/// Detail retention moves opposite to noise reduction: C1's default is 50, and
/// heavy NR would otherwise erase texture at high ISO.
fn noise_details_for(iso: u32) -> f32 {
    match iso {
        0..=6400 => 50.0,
        6401..=25600 => 40.0,
        _ => 25.0,
    }
}

/// Chroma noise reduction; C1's default is 50 over a 0..100 range.
fn color_noise_reduction_for(iso: u32) -> f32 {
    match iso {
        0..=1600 => 40.0,
        1601..=12800 => 50.0,
        _ => 60.0,
    }
}

/// Per-image overrides derived from one RAW info block.
///
/// Exposure range is `coreConfig`'s `BASEDEFAULT` (-4..+4) and its default is
/// 0 once the exposure triangle parses. Sharpening and noise defaults follow
/// the ISO ladders above; ranges are C1's (USM 0..1000, the noise tools 0..100).
pub fn overrides_for(info: &RawInfoBlock) -> ParamOverrides {
    let mut o = ParamOverrides::default();
    if let Some(ev) = ev100(info) {
        // Reject physically impossible triples rather than emitting a default
        // for a mis-parsed block.
        if ev.is_finite() && (-10.0..=25.0).contains(&ev) {
            o.set_range("Exposure", -4.0, 4.0);
            o.set_default("Exposure", 0.0);
        }
    }
    apply_white_balance(info, &mut o);
    let iso = info.iso.or_else(|| info.get("IB_I_ISO").map(|v| v as u32));
    if let Some(iso) = iso {
        o.set_range("USMAmount", 0.0, 1000.0);
        o.set_default("USMAmount", sharpening_for(iso));
        o.set_range("USMRadius", 0.2, 2.5);
        o.set_default("USMRadius", 0.8);
        o.set_range("USMThreshold", 0.0, 12.0);
        o.set_default("USMThreshold", 1.0);
        o.set_range("NoiseReductionAmount", 0.0, 100.0);
        o.set_default("NoiseReductionAmount", noise_reduction_for(iso));
        o.set_range("NoiseDetailsAmount", 0.0, 100.0);
        o.set_default("NoiseDetailsAmount", noise_details_for(iso));
        o.set_range("ColorNoiseReductionAmount", 0.0, 100.0);
        o.set_default("ColorNoiseReductionAmount", color_noise_reduction_for(iso));
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::param_meta::by_key;
    use std::path::Path;

    fn block(pairs: &[(&str, f64)]) -> RawInfoBlock {
        RawInfoBlock {
            values: pairs.iter().map(|(k, v)| ((*k).to_string(), *v)).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn ev_formula_matches_the_standard_definition() {
        // 1/125 s, f/8, ISO 100 -> log2(64 * 125) = log2(8000) = 12.9658.
        let b = block(&[
            ("IB_F_EXP_SHUTTERSPEED", 1.0 / 125.0),
            ("IB_F_EXP_APERTURE", 8.0),
            ("IB_I_ISO", 100.0),
        ]);
        let ev = ev100(&b).unwrap();
        assert!((ev - 8000f32.log2()).abs() < 1e-3, "{ev}");
        // One stop of ISO is one stop of scene light.
        let b200 = block(&[
            ("IB_F_EXP_SHUTTERSPEED", 1.0 / 125.0),
            ("IB_F_EXP_APERTURE", 8.0),
            ("IB_I_ISO", 200.0),
        ]);
        assert!((ev100(&b200).unwrap() - (ev - 1.0)).abs() < 1e-4);
        // Missing or impossible values produce no EV.
        assert_eq!(
            ev100(&block(&[("IB_F_EXP_SHUTTERSPEED", 0.0), ("IB_F_EXP_APERTURE", 4.0)])),
            None
        );
    }


    #[test]
    fn missing_fields_fall_back_to_catalog_defaults() {
        let o = overrides_for(&RawInfoBlock::default());
        assert!(o.is_empty(), "{o:?}");
        let m = by_key("USMAmount").unwrap();
        assert_eq!(o.apply(m), (m.min, m.max, m.default));
    }

    #[test]
    fn extreme_iso_clamps_the_ladders() {
        let low = overrides_for(&block(&[("IB_I_ISO", 50.0)]));
        assert_eq!(low.apply(by_key("USMAmount").unwrap()).2, 180.0);
        assert_eq!(low.apply(by_key("NoiseReductionAmount").unwrap()).2, 20.0);
        let high = overrides_for(&block(&[("IB_I_ISO", 409_600.0)]));
        assert_eq!(high.apply(by_key("USMAmount").unwrap()).2, 100.0);
        assert_eq!(high.apply(by_key("NoiseReductionAmount").unwrap()).2, 80.0);
        assert_eq!(high.apply(by_key("NoiseDetailsAmount").unwrap()).2, 25.0);
    }

    #[test]
    fn overrides_are_idempotent() {
        let mut b = block(&[
            ("IB_F_EXP_SHUTTERSPEED", 0.025),
            ("IB_F_EXP_APERTURE", 4.0),
            ("IB_F_WBRED", 1.625),
            ("IB_F_WBGREEN", 1.0),
            ("IB_F_WBBLUE", 1.412),
        ]);
        b.iso = Some(5000);
        assert_eq!(overrides_for(&b), overrides_for(&b));
    }



    #[cfg(feature = "external-fixtures")]
    include!("../local-fixtures/camera_defaults-tests.rs");
}
