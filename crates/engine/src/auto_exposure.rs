//! Suggestions for the Exposure and High Dynamic Range "Auto" buttons.
//!
//! The input is an 8-bit sRGB frame, normally the 256 px probe the engine
//! already renders for auto adjustments. Everything here is a pure function of
//! those pixels; the caller owns the current [`crate::AdjustmentSet`].

use crate::{param_meta, HighDynamicRange};

/// The median this engine aims for, matching the previous auto-exposure probe.
const TARGET_MEDIAN: f32 = 0.46;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Suggestions {
    pub exposure: f32,
    pub contrast: f32,
    pub brightness: f32,
    pub saturation: f32,
    pub hdr: HighDynamicRange,
}

/// Median, 1%/99% points, channel means and highlight clipping of `rgba`, then
/// the slider moves they imply. Empty or short frames return neutral values.
pub fn analyze(rgba: &[u8], width: u32, height: u32) -> Suggestions {
    let pixels = width as usize * height as usize;
    if pixels == 0 || rgba.len() < pixels * 4 {
        return Suggestions::default();
    }

    let mut luma = [0u64; 256];
    let mut sums = [0u64; 3];
    let mut chroma = 0.0f64;
    let mut clipped = 0u64;
    let mut crushed = 0u64;
    let mut counted = 0u64;

    for px in rgba.chunks_exact(4).take(pixels) {
        if px[3] == 0 {
            continue;
        }
        let (r, g, b) = (px[0], px[1], px[2]);
        sums[0] += r as u64;
        sums[1] += g as u64;
        sums[2] += b as u64;
        let y = (0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32).round().clamp(0.0, 255.0) as u8;
        luma[y as usize] += 1;
        let hi = r.max(g).max(b);
        if hi > 0 {
            chroma += (hi - r.min(g).min(b)) as f64 / hi as f64;
        }
        if r >= 250 || g >= 250 || b >= 250 {
            clipped += 1;
        }
        if y <= 5 {
            crushed += 1;
        }
        counted += 1;
    }
    if counted == 0 {
        return Suggestions::default();
    }

    let total = counted;
    let median = percentile(&luma, total, 0.5) as f32 / 255.0;
    let p01 = percentile(&luma, total, 0.01) as f32 / 255.0;
    let p99 = percentile(&luma, total, 0.99) as f32 / 255.0;
    let clip_high = clipped as f32 / total as f32;
    let clip_low = crushed as f32 / total as f32;
    let spread = p99 - p01;
    let mean_chroma = (chroma / total as f64) as f32;

    let exposure = (TARGET_MEDIAN / median.max(0.02)).log2() * 0.75;
    let contrast = if spread < 0.03 { 0.0 } else { (0.5 - spread) * 80.0 };
    let brightness = if (median - TARGET_MEDIAN).abs() < 0.01 { 0.0 } else { (TARGET_MEDIAN - median) * 60.0 };
    let saturation = if mean_chroma < 0.02 { 0.0 } else { (0.25 - mean_chroma) * 80.0 };

    Suggestions {
        exposure: clamp_key("Exposure", exposure),
        contrast: clamp_key("Contrast", contrast),
        brightness: clamp_key("Brightness", brightness),
        saturation: clamp_key("Saturation", saturation),
        hdr: HighDynamicRange {
            highlights: clamp_key("HighlightRecovery", -(clip_high * 250.0)),
            shadows: clamp_key("ShadowRecovery", ((0.06 - p01) / 0.06).max(0.0) * 60.0),
            whites: clamp_key("WhiteRecovery", -(((p99 - 0.9) / 0.1).max(0.0) * 60.0)),
            blacks: clamp_key("BlackRecovery", clip_low * 250.0 - clip_high * 100.0),
        },
    }
}

/// The luma value at `fraction` of the counted pixels.
fn percentile(hist: &[u64; 256], total: u64, fraction: f32) -> u8 {
    let target = (total as f32 * fraction).ceil() as u64;
    let mut seen = 0u64;
    for (value, count) in hist.iter().enumerate() {
        seen += count;
        if seen >= target.max(1) {
            return value as u8;
        }
    }
    255
}

/// Clamp to the catalog range and snap to its UI step. Unknown keys and NaN
/// fall back to neutral.
fn clamp_key(key: &'static str, value: f32) -> f32 {
    let Some(meta) = param_meta::CATALOG.iter().find(|m| m.key == key) else {
        return 0.0;
    };
    if !value.is_finite() {
        return meta.default;
    }
    let value = value.clamp(meta.min, meta.max);
    if meta.step > 0.0 {
        ((value / meta.step).round() * meta.step).clamp(meta.min, meta.max)
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(value: u8, w: u32, h: u32) -> Vec<u8> {
        [value, value, value, 255].repeat((w * h) as usize)
    }

    fn assert_neutral(s: &Suggestions) {
        assert_eq!(s, &Suggestions::default());
    }

    #[test]
    fn neutral_gray_is_unchanged() {
        let s = analyze(&solid(118, 32, 32), 32, 32);
        assert_neutral(&s);
    }

    #[test]
    fn dark_frame_is_lifted() {
        let s = analyze(&solid(30, 32, 32), 32, 32);
        assert!(s.exposure > 0.5, "{s:?}");
        assert!(s.brightness > 0.0, "{s:?}");
    }

    #[test]
    fn bright_frame_is_pulled_down() {
        let s = analyze(&solid(230, 32, 32), 32, 32);
        assert!(s.exposure < -0.3, "{s:?}");
        assert!(s.hdr.highlights < 0.0 || s.hdr.whites < 0.0, "{s:?}");
    }

    #[test]
    fn clipped_highlights_are_recovered() {
        let mut rgba = solid(128, 32, 32);
        for px in rgba.chunks_exact_mut(4).take(16 * 32) {
            px[0] = 255;
            px[1] = 255;
            px[2] = 255;
        }
        let s = analyze(&rgba, 32, 32);
        assert!(s.hdr.highlights < 0.0, "{s:?}");
        assert!(s.hdr.whites < 0.0, "{s:?}");
        assert!(s.hdr.blacks <= 0.0, "{s:?}");
    }

    #[test]
    fn black_white_frames_stay_finite() {
        for value in [0u8, 255] {
            let s = analyze(&solid(value, 16, 16), 16, 16);
            for v in [
                s.exposure,
                s.contrast,
                s.brightness,
                s.saturation,
                s.hdr.highlights,
                s.hdr.shadows,
                s.hdr.whites,
                s.hdr.blacks,
            ] {
                assert!(v.is_finite(), "{v} is not finite for {value}");
            }
        }
    }

    #[test]
    fn every_field_stays_in_catalog_range() {
        let frames = [solid(0, 8, 8), solid(118, 8, 8), solid(255, 8, 8), solid(200, 8, 8)];
        for rgba in frames {
            let s = analyze(&rgba, 8, 8);
            let rows = [
                ("Exposure", s.exposure),
                ("Contrast", s.contrast),
                ("Brightness", s.brightness),
                ("Saturation", s.saturation),
                ("HighlightRecovery", s.hdr.highlights),
                ("ShadowRecovery", s.hdr.shadows),
                ("WhiteRecovery", s.hdr.whites),
                ("BlackRecovery", s.hdr.blacks),
            ];
            for (key, value) in rows {
                let meta = param_meta::CATALOG.iter().find(|m| m.key == key).unwrap();
                assert!(value >= meta.min && value <= meta.max, "{key}={value} outside {}..{}", meta.min, meta.max);
            }
        }
    }

    #[test]
    fn empty_or_short_frames_are_safe() {
        assert_neutral(&analyze(&[], 0, 0));
        assert_neutral(&analyze(&[], 10, 10));
        assert_neutral(&analyze(&[0; 9], 10, 10));
    }

    #[cfg(feature = "external-fixtures")]
    #[test]
    #[ignore = "requires an explicit LUMEN_TEST_RAW input"]
    fn report_real_nef() {
        use std::path::Path;

        let input = std::env::var_os("LUMEN_TEST_RAW").expect("set LUMEN_TEST_RAW to a local test RAW");
        let path = Path::new(&input);
        let image = crate::raw::decode(path).unwrap();
        let src = crate::Source::from_rgba(image.width(), image.height(), image.as_raw());

        let before = crate::render(&src, &crate::AdjustmentSet::default(), crate::NormRect::FULL, 256, false, None);
        let s = analyze(&before.rgba, before.width, before.height);
        let mut adj = crate::AdjustmentSet::default();
        adj.exposure.exposure = s.exposure;
        adj.exposure.contrast = s.contrast;
        adj.exposure.brightness = s.brightness;
        adj.exposure.saturation = s.saturation;
        adj.hdr = s.hdr.clone();
        let after = crate::render(&src, &adj, crate::NormRect::FULL, 256, false, None);

        let stats = |rgba: &[u8]| {
            let mut hist = [0u64; 256];
            let mut clipped = 0u64;
            let mut n = 0u64;
            for px in rgba.chunks_exact(4) {
                let y = (0.2126 * px[0] as f32 + 0.7152 * px[1] as f32 + 0.0722 * px[2] as f32).round() as usize;
                hist[y.min(255)] += 1;
                if px[0] >= 250 || px[1] >= 250 || px[2] >= 250 {
                    clipped += 1;
                }
                n += 1;
            }
            let mut seen = 0;
            let mut median = 1.0f32;
            for (v, c) in hist.iter().enumerate() {
                seen += c;
                if seen >= n / 2 {
                    median = v as f32 / 255.0;
                    break;
                }
            }
            (median, clipped as f32 / n as f32)
        };
        let (bm, bc) = stats(&before.rgba);
        let (am, ac) = stats(&after.rgba);
        println!("NEF before: median={bm:.3} clip={bc:.3}");
        println!("NEF after:  median={am:.3} clip={ac:.3}  {s:?}");
        assert!(am.is_finite() && ac.is_finite());
    }
}
