//! ISO-adaptive defaults for [`NoiseReduction`], mirroring Capture One's habit of
//! adding noise reduction as the sensor's ISO rises instead of starting at a
//! fixed 50.
//!
//! Amounts are piecewise linear between the anchors below and hold at both
//! ends. `details` stays at Capture One's neutral 50: it narrows the luminance
//! tolerance rather than adding strength, so it does not follow ISO.
use crate::NoiseReduction;

/// The four values of [`crate::adjust::NoiseReduction`] for one ISO.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NoiseDefaults {
    /// 0 ..= 100, as in [`NoiseReduction::luminance`].
    pub luminance: f32,
    pub details: f32,
    pub color: f32,
    pub single_pixel: f32,
}

/// ISO 100 and 400 are off; the curve starts at 800, is decided by 3200 and
/// saturates at 12800. `(iso, luminance, color, single_pixel)`.
const CURVE: [(u32, f32, f32, f32); 7] = [
    (100, 0.0, 0.0, 0.0),
    (400, 0.0, 0.0, 0.0),
    (800, 15.0, 20.0, 0.0),
    (1600, 30.0, 35.0, 0.0),
    (3200, 45.0, 50.0, 10.0),
    (6400, 60.0, 65.0, 15.0),
    (12800, 75.0, 80.0, 20.0),
];

const DETAILS: f32 = 50.0;

/// Defaults for a camera ISO. Values outside the curve's range are clamped to
/// its ends, so ISO 0 and ISO 100 are identical.
pub fn for_iso(iso: u32) -> NoiseDefaults {
    let last = CURVE[CURVE.len() - 1];
    if iso <= CURVE[0].0 {
        return NoiseDefaults {
            luminance: 0.0,
            details: DETAILS,
            color: 0.0,
            single_pixel: 0.0,
        };
    }
    if iso >= last.0 {
        return NoiseDefaults {
            luminance: last.1,
            details: DETAILS,
            color: last.2,
            single_pixel: last.3,
        };
    }
    let i = CURVE.iter().position(|(at, ..)| *at >= iso).expect("iso is between anchors");
    let (lo_iso, lo_lum, lo_color, lo_single) = CURVE[i - 1];
    let (hi_iso, hi_lum, hi_color, hi_single) = CURVE[i];
    let t = (iso - lo_iso) as f32 / (hi_iso - lo_iso) as f32;
    let (lum, color, single) = (
        lo_lum + (hi_lum - lo_lum) * t,
        lo_color + (hi_color - lo_color) * t,
        lo_single + (hi_single - lo_single) * t,
    );
    NoiseDefaults { luminance: lum, details: DETAILS, color, single_pixel: single }
}

impl From<NoiseDefaults> for NoiseReduction {
    fn from(d: NoiseDefaults) -> Self {
        Self {
            luminance: d.luminance,
            details: d.details,
            color: d.color,
            single_pixel: d.single_pixel,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AdjustmentSet;

    #[test]
    fn anchors_match_the_table() {
        for &(iso, lum, color, single) in &CURVE {
            let d = for_iso(iso);
            assert_eq!((d.luminance, d.color, d.single_pixel), (lum, color, single), "ISO {iso}");
            assert_eq!(d.details, 50.0);
        }
    }

    #[test]
    fn monotonic_from_zero_to_high_iso() {
        let mut prev = for_iso(0);
        for iso in (50..=25600).step_by(50) {
            let d = for_iso(iso);
            assert!(d.luminance >= prev.luminance, "luminance drops at ISO {iso}");
            assert!(d.color >= prev.color, "color drops at ISO {iso}");
            assert!(d.single_pixel >= prev.single_pixel, "single pixel drops at ISO {iso}");
            prev = d;
        }
    }

    #[test]
    fn boundaries_clamp_and_base_iso_is_off() {
        assert_eq!(for_iso(0), for_iso(50));
        assert_eq!(for_iso(0), for_iso(100));
        assert_eq!(for_iso(100).luminance, 0.0);
        assert_eq!(for_iso(100).color, 0.0);
        assert_eq!(for_iso(25600), for_iso(12800));
        assert_eq!(for_iso(25600).luminance, 75.0);
    }

    #[test]
    fn fits_noise_reduction_and_has_no_nan() {
        for iso in [0, 50, 100, 799, 800, 3200, 12800, 25600, u32::MAX] {
            let d = for_iso(iso);
            for v in [d.luminance, d.details, d.color, d.single_pixel] {
                assert!(v.is_finite() && (0.0..=100.0).contains(&v), "ISO {iso}: {v}");
            }
            let nr: NoiseReduction = d.into();
            let set = AdjustmentSet { noise_reduction: nr, ..Default::default() };
            assert_eq!(set, set.clone(), "fields must match AdjustmentSet::default types");
        }
    }
}
