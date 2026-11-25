//! Film Grain pixels, ported from C1's `Grains` CL kernels.
//!
//! v1 (the default compile-time variant) multiplies the image by a per-pixel
//! noise plane; v8 (the composite kernel, host method 3 for Soft/Harsh)
//! composites an erf-Gaussian-blurred noise plane through the 257-entry
//! visibility LUT. Both planes come from the same integer hash family, with
//! every constant copied from `grains.cl`.
//!
//! This stage is a pure pixel operation that depends only on adjustment data
//! ([`crate::FilmGrain`]), so it lives in `processing` rather than `mock`.
//! `crate::mock` re-exports it so pre-existing call sites keep compiling while
//! the production renderer is migrated out of the mock module.

use rayon::prelude::*;

use crate::FilmGrain;

/// v8 weight divisor (`op+0x2aa`).
const LUT_SCALE: f32 = 1611.0;

/// Noise amplitude at Amount = 100 for the v1 path (FineGrain, SilverRich),
/// as a fraction of the pixel value.
///
/// Calibrated against Capture One 16.8.6.3784: a flat 32896 field with
/// `filmFineGrain`, Amount 100, any Granularity renders with a standard
/// deviation of 2089 codes out of 65535 (8.13 of 255 in 8-bit). Our per-pixel
/// plane has sigma 1, so the gain is 8.13 / (255 * 1.0) / 0.5 = 0.0637 — the
/// extra half because the plane multiplies `1 + weight * plane * gain` and the
/// visible deviation is measured on the pixel value.
pub(crate) const GRAIN_GAIN_V1: f32 = 0.0637;

/// Standard deviation of a value plane.
fn plane_sigma(plane: &[f32]) -> f32 {
    if plane.is_empty() {
        return 0.0;
    }
    let mean = plane.iter().sum::<f32>() / plane.len() as f32;
    let var = plane.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / plane.len() as f32;
    var.sqrt()
}

impl crate::GrainKind {
    /// Native v8 deviation at Amount 100 and `w = 1`, in 8-bit codes.
    ///
    /// Capture One on a 512x512 flat 32896 field renders 558.8 / 727.4 /
    /// 1036.5 / 1469.2 codes of 65535 for Soft / Cubic / Tabular / Harsh at
    /// Granularity 50. Dividing by 257 and by the visibility weight at code
    /// 128 (`880 / 1611`) gives the amplitudes below; the slopes come from the
    /// same three Granularity samples (Soft 293.2→1071.3, Cubic 371.9→1469.2,
    /// Tabular 454.2→2257.5, Harsh 767.0→2817.4 over g 0..100).
    pub(crate) fn v8_amplitude(self, granularity: f32) -> f32 {
        use crate::GrainKind::*;
        let (at_50, slope) = match self {
            SoftGrain => (7.96, 0.0187),
            CubicGrain => (10.36, 0.0198),
            TabularGrain => (14.78, 0.0231),
            HarshGrain => (20.94, 0.0188),
            FineGrain | SilverRich => (1.0, 0.01875),
        };
        let g = if granularity.is_finite() { granularity.clamp(0.0, 100.0) } else { 50.0 };
        at_50 * 2f32.powf(slope * (g - 50.0))
    }
}

/// `z94048668db` — v8's visibility LUT, 257 entries as dumped from the kernel.
/// Only indices 0..=255 are read (`p >> 8`); 256 is the array sentinel.
pub(crate) const V8_LUT: [u16; 257] = [
    32, 509, 684, 810, 908, 990, 1060, 1120, 1173, 1220, 1262, 1300, 1334, 1364, 1392, 1417,
    1440, 1461, 1479, 1496, 1512, 1526, 1538, 1550, 1560, 1569, 1577, 1584, 1590, 1595, 1599, 1603,
    1606, 1608, 1610, 1611, 1611, 1611, 1611, 1610, 1608, 1606, 1604, 1601, 1598, 1595, 1591, 1587,
    1582, 1578, 1573, 1567, 1562, 1556, 1550, 1544, 1537, 1531, 1524, 1517, 1510, 1503, 1495, 1487,
    1480, 1472, 1464, 1456, 1447, 1439, 1430, 1422, 1413, 1404, 1396, 1387, 1378, 1369, 1359, 1350,
    1341, 1332, 1322, 1313, 1303, 1294, 1284, 1275, 1265, 1256, 1246, 1236, 1227, 1217, 1207, 1197,
    1188, 1178, 1168, 1158, 1149, 1139, 1129, 1119, 1110, 1100, 1090, 1080, 1071, 1061, 1051, 1041,
    1032, 1022, 1012, 1003, 993, 984, 974, 965, 955, 945, 936, 927, 917, 908, 898, 889,
    880, 871, 861, 852, 843, 834, 825, 816, 807, 798, 789, 780, 771, 762, 753, 744,
    736, 727, 718, 710, 701, 693, 684, 676, 667, 659, 651, 642, 634, 626, 618, 610,
    602, 594, 586, 578, 570, 562, 554, 547, 539, 531, 524, 516, 509, 501, 494, 486,
    479, 472, 464, 457, 450, 443, 436, 429, 422, 415, 408, 401, 394, 388, 381, 374,
    368, 361, 354, 348, 341, 335, 329, 322, 316, 310, 304, 298, 292, 285, 279, 273,
    268, 262, 256, 250, 244, 239, 233, 227, 222, 216, 211, 205, 200, 194, 189, 184,
    179, 173, 168, 163, 158, 153, 148, 143, 138, 133, 128, 124, 119, 114, 109, 105,
    100, 96, 91, 87, 82, 78, 73, 69, 65, 60, 56, 52, 48, 44, 40, 36,
    32,
];

/// `z3a31cb915d` — the three-shift mixer every value-noise path starts from.
fn mix(mut x: u32) -> u32 {
    x = x.wrapping_sub(x << 6);
    x ^= x >> 17;
    x = x.wrapping_sub(x << 9);
    x ^= x << 4;
    x = x.wrapping_sub(x << 3);
    x ^= x << 10;
    x ^ (x >> 15)
}

/// `zcf6e5d12e2` — decorrelates the two axes before the field hashes.
fn axis(mut x: u32) -> u32 {
    x ^= x >> 4;
    x = x.wrapping_mul(0x27d4_eb2d);
    x ^ (x >> 15)
}

/// `zcb05287515` — 36969 multiply-with-carry.
fn mwc_36969(x: u32) -> u32 {
    36969u32.wrapping_mul(x & 0xFFFF).wrapping_add(x >> 16)
}

/// `zd0efa19033` — 18000 multiply-with-carry.
fn mwc_18000(x: u32) -> u32 {
    18000u32.wrapping_mul(x & 0xFFFF).wrapping_add(x >> 16)
}

/// `z24baacaf5e` — the LCG that seeds the second axis.
fn lcg(x: u32) -> u32 {
    69069u32.wrapping_mul(x).wrapping_add(1234567)
}

/// `z306365a747` — two rounds of three 10-bit fields, centred and scaled so one
/// unit is roughly one standard deviation. `(int)(64 * sqrt(2))` truncates to 90.
fn gaussian(x: i32, y: i32) -> f32 {
    let mut a = mix((x as u32).wrapping_add(axis(y as u32)));
    let mut b = mix((y as u32).wrapping_add(lcg(x as u32)));
    let mut sum: i32 = 0;
    for _ in 0..2 {
        a = mwc_36969(a);
        b = mwc_18000(b);
        let packed = a.wrapping_shl(16).wrapping_add(b);
        sum += (packed & 0x3FF) as i32;
        sum += ((packed >> 10) & 0x3FF) as i32;
        sum += ((packed >> 20) & 0x3FF) as i32;
    }
    sum -= 6 << 9;
    sum *= 90;
    sum += (a & 0x3F) as i32 - 32;
    sum as f32 * (1.0 / 65536.0)
}

/// Per-image origin offset. C1 hashes a document string with FNV-1a and adds
/// the result to the sample coordinates; a fixed string keeps the pattern
/// stable across renders, which is all the pixel path needs.
pub(crate) fn noise_offsets() -> (i32, i32) {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0100_0000_01b3;
    let mut h = OFFSET;
    for b in b"Aurora Studio grain" {
        h = (h ^ *b as u64).wrapping_mul(PRIME);
    }
    ((mix(h as u32) & 0xFFFF) as i32, (mix((h >> 32) as u32) & 0xFFFF) as i32)
}

/// v0/v7 luma: `(R + B + 2*max(R,G,B)) / 4`, in the 0..=65535 domain.
pub(crate) fn luma_rb_2max(r: u32, g: u32, b: u32) -> u32 {
    (r + b + 2 * r.max(g).max(b)) / 4
}

/// v8: `w = min(LUT[p >> 8] / 1611, 1)`. Index 256 is the sentinel, never read.
pub(crate) fn visibility(luma16: u32) -> f32 {
    (V8_LUT[(luma16 >> 8).min(255) as usize] as f32 / LUT_SCALE).min(1.0)
}

/// v3's noise amplitude: `n * sqrt(luma)`, so grain vanishes at black and the
/// amplitude doubles when luminance quadruples.
pub(crate) fn v3_amplitude(luma: f32) -> f32 {
    luma.max(0.0).sqrt()
}

/// v3's `type >= 2` attenuation: `luma / (luma + sqrt(luma) * s1 * threshold)`.
pub(crate) fn v3_binomial_attenuation(luma: f32, s1: f32, threshold: f32) -> f32 {
    let root = luma.max(0.0).sqrt();
    let denominator = luma + root * s1 * threshold;
    if luma > 0.0 && denominator > 0.0 { luma / denominator } else { 0.0 }
}

/// v7's brightness-compare composite. C1 compares the grain against
/// `luma = (R+B+2max)/4`; above it the grain adds a delta, below it scales the
/// pixel. `amt` and `scale` are the `uint2` kernel arguments, `1024` is fixed.
pub(crate) fn v7_composite(rgb: [u16; 3], grain: u16, amt: u32, scale: u32) -> [u16; 3] {
    let (r, g, b) = (rgb[0] as u32, rgb[1] as u32, rgb[2] as u32);
    let luma = luma_rb_2max(r, g, b);
    let grain = grain as u32;
    let out = if grain >= luma {
        let delta = (grain - luma) * scale / 1024;
        [r + delta, g + delta, b + delta]
    } else {
        let k = grain * amt / luma + amt;
        [r * k / 1024, g * k / 1024, b * k / 1024]
    };
    out.map(|v| v.min(65535) as u16)
}

/// Abramowitz & Stegun 7.1.26, |error| < 1.5e-7. The source calls libm `erf`.
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = ((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_7) * t - 0.284_496_74) * t
        + 0.254_829_6)
        * t;
    sign * (1.0 - poly * (-x * x).exp())
}

/// C1's `0x17507e0` blur kernel: `tap[i] = erf((i+0.5)/(s*sqrt2)) - erf((i-0.5)/(s*sqrt2))`
/// for `i = -r..=r`, `r = min(8, ceil(3*s))`, normalised to sum 1.
pub(crate) fn erf_taps(sigma: f32) -> Vec<f32> {
    let r = (3.0 * sigma).ceil().min(8.0) as i32;
    let inv = 1.0 / (sigma * std::f32::consts::SQRT_2);
    let mut taps: Vec<f32> = (-r..=r)
        .map(|i| erf((i as f32 + 0.5) * inv) - erf((i as f32 - 0.5) * inv))
        .collect();
    let sum: f32 = taps.iter().sum();
    for tap in &mut taps {
        *tap /= sum;
    }
    taps
}

/// Separable erf-Gaussian blur with clamped edges.
fn blur(plane: &mut [f32], w: usize, h: usize, sigma: f32) {
    let taps = erf_taps(sigma);
    let r = (taps.len() / 2) as isize;
    let mut tmp = vec![0.0f32; plane.len()];
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, tap) in taps.iter().enumerate() {
                let sx = (x as isize + k as isize - r).clamp(0, w as isize - 1) as usize;
                acc += plane[y * w + sx] * tap;
            }
            row[x] = acc;
        }
    });
    plane.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let mut acc = 0.0;
            for (k, tap) in taps.iter().enumerate() {
                let sy = (y as isize + k as isize - r).clamp(0, h as isize - 1) as usize;
                acc += tmp[sy * w + x] * tap;
            }
            row[x] = acc;
        }
    });
}

/// Applies the grain plane to an RGBA8 buffer.
///
/// FineGrain/SilverRich take the v1 path: the image is multiplied by the raw
/// per-pixel noise plane. Soft/Harsh take the v8 path: the plane is blurred
/// with [`FilmGrain::sigma_px`] and weighted by the visibility LUT, so grain
/// stays in the midtones. Cubic/Tabular share v8 here; C1's Poisson (method 1)
/// and pure-CPU (method 2) variants are not ported.
pub(crate) fn stage(buf: &mut [u8], w: usize, h: usize, adj: &FilmGrain, scale: f32) {
    let amount = (adj.impact / 100.0).clamp(0.0, 1.0);
    if amount <= 0.0 || w == 0 || h == 0 || buf.len() < w * h * 4 {
        return;
    }
    let (ox, oy) = noise_offsets();
    let mut plane: Vec<f32> =
        (0..w * h).map(|i| gaussian((i % w) as i32 + ox, (i / w) as i32 + oy)).collect();
    let composite = adj.kind.uses_erf_blur();
    let zoom = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
    let sigma = adj.sigma_px() * zoom;
    if composite && sigma > 0.0 {
        blur(&mut plane, w, h, sigma);
    }
    let mut gain = amount * GRAIN_GAIN_V1;
    if composite {
        // v8 composites `1 + w * plane` where `w` is the visibility weight, and
        // the plane arrives at unit sigma: Capture One's visible deviation is
        // proportional to the blur radius (Soft 293/559/1071 at g 0/50/100 is
        // the 2^(0.01875 g) ratio), so the amplitude is folded in here instead
        // of a fixed gain on a blur-weakened plane.
        let sd = plane_sigma(&plane).max(1e-6);
        gain = amount * adj.kind.v8_amplitude(adj.granularity) / (sd * 257.0);
    }
    buf.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
        for x in 0..w {
            let i = x * 4;
            let weight = if composite {
                let (r, g, b) = (row[i] as u32 * 257, row[i + 1] as u32 * 257, row[i + 2] as u32 * 257);
                visibility(luma_rb_2max(r, g, b))
            } else {
                1.0
            };
            let k = 1.0 + weight * plane[y * w + x] * gain;
            for c in 0..3 {
                row[i + c] = (row[i + c] as f32 * k).clamp(0.0, 255.0).round() as u8;
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GrainKind;

    fn flat(size: usize, value: u8) -> Vec<u8> {
        (0..size * size).flat_map(|_| [value, value, value, 255]).collect()
    }

    fn grain(kind: GrainKind, impact: f32, granularity: f32) -> FilmGrain {
        FilmGrain { kind, impact, granularity, density: 0.0 }
    }

    #[test]
    fn lut_is_the_source_table_and_index_256_is_a_sentinel() {
        assert_eq!(V8_LUT.len(), 257);
        assert_eq!(V8_LUT[0], 32);
        assert_eq!(&V8_LUT[35..=38], &[1611, 1611, 1611, 1611]);
        assert_eq!(V8_LUT[255], 36);
        assert_eq!(V8_LUT[256], 32, "the 257th entry is never read");
        assert_eq!(visibility(65535), 36.0 / LUT_SCALE, "255 is the last index");
        assert_eq!(visibility(65535 + 257), 36.0 / LUT_SCALE, "luma is clamped to 255");
    }

    #[test]
    fn visibility_follows_the_curve() {
        assert!(visibility(0) < 0.03, "black is nearly clean");
        assert_eq!(visibility(36 * 256), 1.0, "peak is clamped to 1");
        assert!(visibility(128 * 256) < visibility(64 * 256));
        assert!(visibility(255 * 256) < 0.03, "white is nearly clean");
    }

    #[test]
    fn luma_uses_red_blue_and_twice_the_max() {
        assert_eq!(luma_rb_2max(65535, 0, 0), 65535 * 3 / 4, "red is R plus the 2*max term");
        assert_eq!(luma_rb_2max(0, 65535, 0), 65535 / 2, "green is the 2*max term");
        assert_eq!(luma_rb_2max(0, 0, 65535), 65535 * 3 / 4);
        assert_eq!(luma_rb_2max(1000, 2000, 3000), (1000 + 3000 + 2 * 3000) / 4);
    }

    #[test]
    fn v3_amplitude_and_attenuation_match_the_kernel() {
        assert_eq!(v3_amplitude(0.0), 0.0);
        assert!((v3_amplitude(0.25) - 0.5).abs() < 1e-6);
        assert!((v3_amplitude(0.0625) - 0.25).abs() < 1e-6);
        let luma = 0.16f32;
        let expected = luma / (luma + luma.sqrt() * 0.5 * 0.2);
        assert!((v3_binomial_attenuation(luma, 0.5, 0.2) - expected).abs() < 1e-6);
        assert_eq!(v3_binomial_attenuation(0.0, 0.5, 0.2), 0.0);
    }

    #[test]
    fn v7_composite_adds_above_luma_and_scales_below_it() {
        // Bright pixel under dark grain: scale branch, k = 100*100/1000 + 100.
        let scaled = v7_composite([1000, 500, 250], 100, 100, 1000);
        let luma = luma_rb_2max(1000, 500, 250);
        let k = 100 * 100 / luma + 100;
        assert_eq!(scaled, [1000 * k / 1024, 500 * k / 1024, 250 * k / 1024].map(|v| v as u16));

        // Dark pixel under bright grain: add branch, delta = (grain-luma)*scale/1024.
        let grown = v7_composite([100, 100, 100], 4000, 100, 2000);
        let luma = luma_rb_2max(100, 100, 100);
        let delta = (4000 - luma) * 2000 / 1024;
        assert_eq!(grown, [100 + delta, 100 + delta, 100 + delta].map(|v| v as u16));
        assert_eq!(v7_composite([0, 0, 0], 0, 100, 1000), [0, 0, 0]);
    }

    #[test]
    fn gaussian_field_is_centred_and_roughly_unit_scale() {
        let (ox, oy) = noise_offsets();
        let samples: Vec<f32> =
            (0..64).flat_map(|y| (0..64).map(move |x| gaussian(x + ox, y + oy))).collect();
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let var = samples.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 0.1, "mean {mean}");
        assert!((var.sqrt() - 1.0).abs() < 0.2, "std {}", var.sqrt());
        assert_eq!(gaussian(17, 29), gaussian(17, 29), "deterministic");
        assert_ne!(gaussian(17, 29), gaussian(18, 29));
    }

    #[test]
    fn erf_taps_are_normalised_and_widen_with_sigma() {
        for sigma in [0.3f32, 0.5, 0.738, 1.414, 1.7] {
            let taps = erf_taps(sigma);
            let sum: f32 = taps.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "sigma {sigma}: {sum}");
            assert_eq!((taps.len() - 1) / 2, (3.0 * sigma).ceil().min(8.0) as usize);
        }
        assert_eq!(erf_taps(4.0).len(), 17, "radius is capped at 8");
    }

    #[test]
    fn amount_zero_is_identity() {
        let base = flat(16, 120);
        let mut buf = base.clone();
        for kind in GrainKind::ALL {
            let mut adj = grain(kind, 0.0, 50.0);
            adj.density = 100.0;
            stage(&mut buf, 16, 16, &adj, 1.0);
        }
        assert_eq!(buf, base);
    }

    /// FineGrain at Amount 100 on a flat mid-grey field, measured from Capture
    /// One: standard deviation 2089 codes of 65535 = 8.13 of 255.
    #[test]
    fn fine_grain_matches_the_native_deviation() {
        let sd = |buf: &[u8]| {
            let v: Vec<f32> = buf.chunks(4).map(|p| p[0] as f32).collect();
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
        };
        let mut full = flat(128, 128);
        stage(&mut full, 128, 128, &grain(GrainKind::FineGrain, 100.0, 50.0), 1.0);
        let s100 = sd(&full);
        assert!((s100 - 8.13).abs() < 0.4, "amount 100 std {s100}");
        let mut half = flat(128, 128);
        stage(&mut half, 128, 128, &grain(GrainKind::FineGrain, 50.0, 50.0), 1.0);
        let s50 = sd(&half);
        assert!((s50 - s100 * 0.5).abs() < 0.1, "amount 50 std {s50}");
        // Granularity does not touch the per-pixel path.
        let mut other = flat(128, 128);
        stage(&mut other, 128, 128, &grain(GrainKind::FineGrain, 100.0, 100.0), 1.0);
        assert!((sd(&other) - s100).abs() < 1e-6);
    }

    /// v8 amplitude must grow with Granularity the way Capture One does:
    /// 293.2/558.8/1071.3 (Soft), 371.9/727.4/1469.2 (Cubic),
    /// 454.2/1036.5/2257.5 (Tabular), 767.0/1469.2/2817.4 (Harsh) at g 0/50/100.
    #[test]
    fn v8_granularity_ratios_match_the_native_renders() {
        let sd = |buf: &[u8]| {
            let v: Vec<f32> = buf.chunks(4).map(|p| p[0] as f32).collect();
            let mean = v.iter().sum::<f32>() / v.len() as f32;
            (v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32).sqrt()
        };
        let cases = [
            (GrainKind::SoftGrain, [0.525f32, 1.0, 1.917]),
            (GrainKind::CubicGrain, [0.511, 1.0, 2.020]),
            (GrainKind::TabularGrain, [0.438, 1.0, 2.178]),
            (GrainKind::HarshGrain, [0.522, 1.0, 1.917]),
        ];
        for (kind, want) in cases {
            let mut got = [0.0f32; 3];
            for (i, g) in [0.0f32, 50.0, 100.0].into_iter().enumerate() {
                let mut buf = flat(96, 128);
                stage(&mut buf, 96, 96, &grain(kind, 100.0, g), 1.0);
                got[i] = sd(&buf);
            }
            for i in 0..3 {
                let ratio = got[i] / got[1];
                assert!(
                    (ratio - want[i]).abs() < want[i] * 0.10,
                    "{kind:?} g{}: ratio {ratio} vs {}",
                    [0, 50, 100][i],
                    want[i]
                );
            }
            // The slider still scales the v8 path linearly.
            let mut half = flat(96, 128);
            stage(&mut half, 96, 96, &grain(kind, 50.0, 50.0), 1.0);
            assert!((sd(&half) - got[1] * 0.5).abs() < got[1] * 0.05, "{kind:?} amount 50");
        }
    }

    #[test]
    fn grain_is_monochrome_and_deterministic() {
        let mut a = flat(32, 128);
        let mut b = a.clone();
        stage(&mut a, 32, 32, &grain(GrainKind::FineGrain, 100.0, 50.0), 1.0);
        stage(&mut b, 32, 32, &grain(GrainKind::FineGrain, 100.0, 50.0), 1.0);
        assert_eq!(a, b);
        assert_ne!(a, flat(32, 128), "grain must change the frame");
        for px in a.chunks_exact(4) {
            assert_eq!(px[0], px[1], "one noise scalar feeds all three channels");
            assert_eq!(px[1], px[2]);
            assert_eq!(px[3], 255, "alpha is untouched");
        }
    }

    #[test]
    fn v8_keeps_black_and_white_clean_while_v1_does_not() {
        let (mut black, mut white) = (flat(32, 0), flat(32, 255));
        stage(&mut black, 32, 32, &grain(GrainKind::SoftGrain, 100.0, 50.0), 1.0);
        stage(&mut white, 32, 32, &grain(GrainKind::SoftGrain, 100.0, 50.0), 1.0);
        assert_eq!(black, flat(32, 0), "v8 weight at black is ~0.02");
        assert!(
            white.chunks_exact(4).all(|px| px[0] >= 253 && px[1] >= 253 && px[2] >= 253),
            "white stays within rounding of clean"
        );

        // v1 multiplies, so black stays black; a midtone shows the noise.
        let mut v1 = flat(32, 64);
        stage(&mut v1, 32, 32, &grain(GrainKind::FineGrain, 100.0, 50.0), 1.0);
        assert_ne!(v1, flat(32, 64), "v1 has no visibility weighting");
    }

    #[test]
    fn v8_blur_depends_on_granularity_and_scale() {
        let (w, h) = (64usize, 64usize);
        let run = |k: GrainKind, g: f32, scale: f32| {
            let mut buf = flat(w, 128);
            stage(&mut buf, w, h, &grain(k, 100.0, g), scale);
            buf
        };
        let fine = run(GrainKind::SoftGrain, 0.0, 1.0);
        let coarse = run(GrainKind::SoftGrain, 100.0, 1.0);
        assert_ne!(fine, coarse, "sigma follows granularity");
        assert_eq!(coarse, run(GrainKind::SoftGrain, 100.0, 1.0));
        assert_ne!(coarse, run(GrainKind::SoftGrain, 100.0, 0.5), "preview scale shrinks sigma");
    }

    #[test]
    fn fine_grain_skips_the_erf_blur() {
        let run = |g: f32| {
            let mut buf = flat(48, 110);
            stage(&mut buf, 48, 48, &grain(GrainKind::FineGrain, 100.0, g), 1.0);
            buf
        };
        assert_eq!(run(0.0), run(100.0), "FineGrain sigma is 0, granularity never blurs");
    }

    #[test]
    fn density_only_changes_fine_grain_pixels() {
        for kind in GrainKind::ALL {
            let (mut low, mut high) = (flat(32, 120), flat(32, 120));
            let mut a = grain(kind, 80.0, 60.0);
            let mut b = a.clone();
            a.density = 5.0;
            b.density = 95.0;
            stage(&mut low, 32, 32, &a, 1.0);
            stage(&mut high, 32, 32, &b, 1.0);
            if kind == GrainKind::FineGrain {
                assert!(a.density_is_effective(), "FineGrain reads density");
            } else {
                assert_eq!(low, high, "{kind:?} must ignore density");
            }
        }
    }

    /// C1's jump table has six rows; anything past the end (catalogs carry
    /// values above 9) falls through to the type-3 row, not to FineGrain.
    #[test]
    fn grain_type_past_the_table_falls_back_to_type_three() {
        assert_eq!(crate::GrainKind::from_c1_type(3), GrainKind::CubicGrain);
        for value in [6, 9, 10, 42, i32::MAX] {
            assert_eq!(crate::GrainKind::from_c1_type(value), GrainKind::CubicGrain, "type {value}");
        }
    }

    #[test]
    fn grain_kinds_split_into_the_v1_and_v8_paths() {
        let mut output = Vec::new();
        for kind in GrainKind::ALL {
            let mut buf = flat(32, 110);
            stage(&mut buf, 32, 32, &grain(kind, 80.0, 60.0), 1.0);
            output.push(buf);
        }
        assert_ne!(output[0], output[2], "FineGrain and Soft differ");
    }
}
