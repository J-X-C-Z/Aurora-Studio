//! Highlight recovery and shadow lift for the linear-light development pipeline.
//!
//! `linear` is interleaved linear RGB, values may exceed 1.0. `highlight` and
//! `shadow` use the same -100..=100 units as `HighDynamicRange`.
//!
//! shortcut: clipped pixels are rebuilt from the average channel ratio of nearby
//! unclipped pixels in slice order, with no knowledge of row width or true peak
//! (capped at 4.0). Fine detail inside clipped highlights stays flat; upgrade to
//! a 2D weighted neighbourhood plus a reliability-guided peak estimate if needed.

const KNEE: f32 = 0.75;
const CLIP: f32 = 0.995;
const PEAK_CAP: f32 = 4.0;
const RECONSTRUCT_LOOKBEHIND: usize = 8;

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn max3(c: [f32; 3]) -> f32 {
    c[0].max(c[1]).max(c[2])
}

/// Soft knee from `KNEE` to 1.0, approached asymptotically. Monotonic in `x`.
fn shoulder(x: f32, amount: f32) -> f32 {
    if x <= KNEE {
        return x;
    }
    let over = (x - KNEE) / (1.0 - KNEE);
    let soft = KNEE + (1.0 - KNEE) * over / (1.0 + over);
    x + amount.clamp(0.0, 1.0) * (soft - x)
}

/// Channel proportions of an unclipped pixel, max channel normalized to 1.
fn ratios(c: [f32; 3]) -> Option<[f32; 3]> {
    let m = max3(c);
    if m <= 1e-4 || !m.is_finite() {
        return None;
    }
    Some([c[0].max(0.0) / m, c[1].max(0.0) / m, c[2].max(0.0) / m])
}

/// Estimate a clipped pixel's hue and peak from unclipped neighbours.
#[cfg(test)]
fn reconstruct(src: &[f32], px: usize) -> Option<(f32, [f32; 3])> {
    let index = px / 3;
    let count = src.len() / 3;
    reconstruct_with(index, count, |j| {
        let p = j * 3;
        [src[p], src[p + 1], src[p + 2]]
    })
}

fn reconstruct_with<F>(index: usize, count: usize, mut original: F) -> Option<(f32, [f32; 3])>
where
    F: FnMut(usize) -> [f32; 3],
{
    let lo = index.saturating_sub(RECONSTRUCT_LOOKBEHIND);
    let hi = (index + RECONSTRUCT_LOOKBEHIND + 1).min(count);
    let mut acc = [0.0f32; 3];
    let mut used = 0u32;
    for j in lo..hi {
        if j == index {
            continue;
        }
        let c = original(j);
        if c.iter().all(|v| v.is_finite()) && max3(c) < CLIP {
            if let Some(r) = ratios(c) {
                for k in 0..3 {
                    acc[k] += r[k];
                }
                used += 1;
            }
        }
    }
    if used == 0 {
        return None;
    }
    let m = max3(acc);
    if m <= 1e-6 {
        return None;
    }
    let ratio = [acc[0] / m, acc[1] / m, acc[2] / m];
    let own = original(index);
    // Each unclipped channel bounds the true peak through its recovered ratio.
    let mut peak = max3(own);
    for k in 0..3 {
        if own[k].is_finite() && own[k] < CLIP {
            peak = peak.max(own[k] / ratio[k].max(0.02));
        }
    }
    Some((peak.min(PEAK_CAP), ratio))
}

/// Compress highlights and lift shadows in place, preserving channel ratios.
pub fn recover(linear: &mut [f32], highlight: f32, shadow: f32) {
    let h = if highlight.is_finite() { (highlight / 100.0).clamp(0.0, 1.0) } else { 0.0 };
    let s = if shadow.is_finite() { (shadow / 100.0).clamp(0.0, 1.0) } else { 0.0 };
    if h == 0.0 && s == 0.0 {
        return;
    }
    let count = linear.len() / 3;
    let mut history = [[0.0f32; 3]; RECONSTRUCT_LOOKBEHIND];
    for index in 0..count {
        let px = index * 3;
        let original = [linear[px], linear[px + 1], linear[px + 2]];
        let mut c = original;
        if !c.iter().all(|v| v.is_finite()) {
            linear[px..px + 3].fill(0.0);
            history[index % RECONSTRUCT_LOOKBEHIND] = original;
            continue;
        }
        // Shadow lift: scale all channels, weighted to deep tones only.
        if s > 0.0 {
            let l = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]).max(0.0);
            let w = (1.0 - smoothstep(0.0, 0.6, l)).powi(2);
            let gain = 1.0 + s * 1.5 * w;
            c = [c[0] * gain, c[1] * gain, c[2] * gain];
        }
        // Highlights: proportional shoulder, and neighbour-ratio rebuild when clipped.
        if h > 0.0 {
            let clipped = c.iter().any(|v| *v >= CLIP);
            if clipped {
                if let Some((peak, ratio)) =
                    reconstruct_with(index, count, |j| {
                        if j < index {
                            history[j % RECONSTRUCT_LOOKBEHIND]
                        } else {
                            let p = j * 3;
                            [linear[p], linear[p + 1], linear[p + 2]]
                        }
                    })
                {
                    let y = shoulder(peak, h);
                    c = [y * ratio[0], y * ratio[1], y * ratio[2]];
                } else {
                    let m = max3(c);
                    let k = shoulder(m, h) / m.max(1e-6);
                    c = [c[0] * k, c[1] * k, c[2] * k];
                }
            } else {
                let m = max3(c);
                if m > KNEE {
                    let k = shoulder(m, h) / m;
                    c = [c[0] * k, c[1] * k, c[2] * k];
                }
            }
        }
        linear[px..px + 3].copy_from_slice(&c);
        history[index % RECONSTRUCT_LOOKBEHIND] = original;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const HUE: [f32; 3] = [1.0, 0.5, 0.25];

    #[test]
    fn identity_when_both_amounts_are_zero() {
        let original = [0.0, 0.2, 0.8, 2.0, 0.5, 0.4];
        let mut data = original;
        recover(&mut data, 0.0, 0.0);
        assert_eq!(data, original);
    }

    fn recover_full_snapshot_reference(linear: &mut [f32], highlight: f32, shadow: f32) {
        let h = if highlight.is_finite() {
            (highlight / 100.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let s = if shadow.is_finite() {
            (shadow / 100.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if h == 0.0 && s == 0.0 {
            return;
        }
        let src = linear.to_vec();
        for (index, out) in linear.chunks_exact_mut(3).enumerate() {
            let px = index * 3;
            let mut c = [src[px], src[px + 1], src[px + 2]];
            if !c.iter().all(|v| v.is_finite()) {
                out.fill(0.0);
                continue;
            }
            if s > 0.0 {
                let l = (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]).max(0.0);
                let w = (1.0 - smoothstep(0.0, 0.6, l)).powi(2);
                let gain = 1.0 + s * 1.5 * w;
                c = [c[0] * gain, c[1] * gain, c[2] * gain];
            }
            if h > 0.0 {
                let clipped = c.iter().any(|v| *v >= CLIP);
                if clipped {
                    if let Some((peak, ratio)) = reconstruct(&src, px) {
                        let y = shoulder(peak, h);
                        c = [y * ratio[0], y * ratio[1], y * ratio[2]];
                    } else {
                        let m = max3(c);
                        let k = shoulder(m, h) / m.max(1e-6);
                        c = [c[0] * k, c[1] * k, c[2] * k];
                    }
                } else {
                    let m = max3(c);
                    if m > KNEE {
                        let k = shoulder(m, h) / m;
                        c = [c[0] * k, c[1] * k, c[2] * k];
                    }
                }
            }
            out.copy_from_slice(&c);
        }
    }

    #[test]
    fn ring_history_matches_full_snapshot_reference() {
        let mut data = Vec::new();
        for i in 0..257usize {
            let x = i as f32 * 0.137;
            let mut px = [
                (x * 0.73).sin() * 1.6 + 0.2,
                (x * 0.41).cos() * 1.3 + 0.3,
                (x * 0.91).sin() * 0.9 + 0.1,
            ];
            if i % 11 == 0 {
                px[0] = 1.25;
            }
            if i % 17 == 0 {
                px[1] = 2.5;
            }
            if i % 23 == 0 {
                px[2] = 1.75;
            }
            data.extend_from_slice(&px);
        }
        data.extend_from_slice(&[42.0, -13.0]);

        let mut expected = data.clone();
        recover_full_snapshot_reference(&mut expected, 83.0, 41.0);
        recover(&mut data, 83.0, 41.0);

        assert_eq!(data, expected);
    }

    #[test]
    fn output_is_monotonic_in_input() {
        let mut previous = -1.0;
        for v in [0.05, 0.2, 0.5, 0.75, 0.9, 1.5, 2.0, 4.0] {
            let mut data = [v, v, v];
            recover(&mut data, 80.0, 0.0);
            let out = data[0];
            assert!(out >= previous - 1e-6, "v={v} out={out} prev={previous}");
            previous = out;
        }
    }

    #[test]
    fn extreme_parameters_stay_finite() {
        let mut data = [0.0, 0.1, 0.9, 1.0, 0.4, 0.2, 3.0, 0.5, 0.1];
        for (h, s) in [(1e9, -1e9), (f32::NAN, 1e9), (f32::INFINITY, f32::NEG_INFINITY)] {
            recover(&mut data, h, s);
            assert!(data.iter().all(|v| v.is_finite()), "h={h} s={s}");
        }
    }

    #[test]
    fn empty_and_trailing_slices_are_safe() {
        recover(&mut [], 100.0, 100.0);
        let mut data = [0.2, 0.4, 0.6, 42.0];
        recover(&mut data, 100.0, 100.0);
        assert_eq!(data[3], 42.0);
    }

    #[test]
    fn shoulder_preserves_channel_ratios() {
        let mut data = [0.6, 0.3, 0.15, 1.5, 0.75, 0.375];
        recover(&mut data, 90.0, 0.0);
        for px in data.chunks_exact(3) {
            assert!((px[1] / px[0] - HUE[1]).abs() < 1e-3, "green ratio {:?}", px);
            assert!((px[2] / px[0] - HUE[2]).abs() < 1e-3, "blue ratio {:?}", px);
        }
    }

    #[test]
    fn clipped_rebuild_does_not_go_magenta_or_green() {
        let mut data = [0.8, 0.32, 0.16, 1.0, 0.4, 0.2];
        recover(&mut data, 100.0, 0.0);
        let px = &data[3..6];
        assert!(px[0] > px[1] && px[1] > px[2], "channel order lost: {:?}", px);
        assert!((px[1] / px[0] - 0.4).abs() < 5e-2, "green drift: {:?}", px);
        assert!((px[2] / px[0] - 0.2).abs() < 5e-2, "blue drift: {:?}", px);
    }

    #[test]
    fn shadow_only_lifts_deep_tones() {
        let mut dark = [0.05f32, 0.05, 0.05];
        recover(&mut dark, 0.0, 100.0);
        assert!(dark[0] > 0.05);
        let mut bright = [0.9f32, 0.9, 0.9];
        recover(&mut bright, 0.0, 100.0);
        assert_eq!(bright, [0.9, 0.9, 0.9]);
    }
}
