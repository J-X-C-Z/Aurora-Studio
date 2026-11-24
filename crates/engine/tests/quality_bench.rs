//! Synthetic self-tests for the quality bench metrics.
//!
//! The default run is hermetic: no network, no external files. Real RAW and
//! real-photo checks live behind the `external-fixtures` feature and `#[ignore]`
//! (see the bottom of this file and `tests/README-quality-bench.md`).

#[path = "support/quality_metrics.rs"]
mod metrics;

use metrics::{
    ciede2000, colorchecker, diff_rgba8, diff_rgba16, false_color, median_patch_rgb, mtf50,
    shadow_snr, srgb8_to_lab, ColorCheckerConfig, LoadedImage, Rgba16Image, Rgba8Image, Roi,
    D65_WHITE,
};

// ---------------------------------------------------------------------------
// Synthetic image helpers
// ---------------------------------------------------------------------------

/// Abramowitz & Stegun 7.1.26; |error| < 1.5e-7.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t
        - 0.284_496_736)
        * t
        + 0.254_829_592)
        * t;
    sign * (1.0 - poly * (-x * x).exp())
}

/// A Gaussian-blurred straight edge, sampled at pixel centres.
fn gaussian_edge_image(size: u32, sigma: f64, angle_rad: f64) -> Rgba8Image {
    let mut pixels = Vec::with_capacity((size * size * 4) as usize);
    let center = (size as f64 - 1.0) * 0.5;
    let (sin, cos) = angle_rad.sin_cos();
    let norm = (2.0f64).sqrt() * sigma;
    for y in 0..size {
        for x in 0..size {
            let distance =
                (x as f64 - center) * cos + (y as f64 - center) * sin;
            let value = 0.5 * (1.0 + erf(distance / norm));
            let q = (value * 255.0).round().clamp(0.0, 255.0) as u8;
            pixels.extend_from_slice(&[q, q, q, 255]);
        }
    }
    Rgba8Image::new(size, size, pixels)
}

/// Deterministic SplitMix64, so the noise tests are reproducible.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// Box-Muller, standard normal.
    fn gaussian(&mut self) -> f64 {
        let u1 = self.uniform().max(f64::MIN_POSITIVE);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Gray image whose *linear* luminance is `mean + sigma * N(0, 1)`.
fn noisy_gray_image(width: u32, height: u32, mean: f64, sigma: f64, seed: u64) -> Rgba8Image {
    let mut rng = Rng::new(seed);
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..width * height {
        let linear = (mean + sigma * rng.gaussian()).clamp(0.0, 1.0);
        let encoded =
            (metrics::linear_to_srgb(linear) * 255.0).round().clamp(0.0, 255.0) as u8;
        pixels.extend_from_slice(&[encoded, encoded, encoded, 255]);
    }
    Rgba8Image::new(width, height, pixels)
}

fn solid_rgba(width: u32, height: u32, rgb: [u8; 3]) -> Rgba8Image {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..width * height {
        pixels.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
    }
    Rgba8Image::new(width, height, pixels)
}

fn relative_error(measured: f64, expected: f64) -> f64 {
    (measured - expected).abs() / expected.abs()
}

// ---------------------------------------------------------------------------
// Colour: sRGB -> Lab and CIEDE2000
// ---------------------------------------------------------------------------

#[test]
fn srgb_lab_primaries_match_standard_d65() {
    // The sRGB matrix coefficients are rounded, so the D65 white row sums and
    // the reference white differ in the 7th decimal; 1e-4 is far below a
    // visible or reportable error.
    let white = srgb8_to_lab([255, 255, 255]);
    eprintln!(
        "white Lab = [{:.9}, {:.9}, {:.9}]",
        white[0], white[1], white[2]
    );
    assert!((white[0] - 100.0).abs() < 1e-4, "white L = {}", white[0]);
    assert!(white[1].abs() < 1e-4, "white a = {}", white[1]);
    assert!(white[2].abs() < 1e-4, "white b = {}", white[2]);

    // IEC 61966-2-1 / D65 reference values.
    let cases: [([u8; 3], [f64; 3]); 3] = [
        ([255, 0, 0], [53.2408, 80.0925, 67.2032]),
        ([0, 255, 0], [87.7347, -86.1827, 83.1793]),
        ([0, 0, 255], [32.2970, 79.1875, -107.8602]),
    ];
    for (rgb, expected) in cases {
        let lab = srgb8_to_lab(rgb);
        for (i, &want) in expected.iter().enumerate() {
            assert!(
                (lab[i] - want).abs() < 0.05,
                "rgb {:?} Lab {} = {:.4}, want {:.4}",
                rgb,
                i,
                lab[i],
                want
            );
        }
    }

    // Neutral grays must land exactly on the a = b = 0 axis.
    for value in [0u8, 16, 64, 128, 200, 254] {
        let lab = srgb8_to_lab([value, value, value]);
        assert!(lab[1].abs() < 1e-4, "gray {value} a = {}", lab[1]);
        assert!(lab[2].abs() < 1e-4, "gray {value} b = {}", lab[2]);
    }
    assert_eq!(D65_WHITE, [0.950_47, 1.0, 1.088_83]);
}

#[test]
fn ciede2000_matches_sharma_reference_pairs() {
    // Sharma, Wu & Dalal (2005), supplementary table, first rows.
    let cases: [([f64; 3], [f64; 3], f64); 7] = [
        ([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485], 2.0425),
        ([50.0, 3.1571, -77.2803], [50.0, 0.0, -82.7485], 2.8615),
        ([50.0, 2.8361, -74.0200], [50.0, 0.0, -82.7485], 3.4412),
        ([50.0, -1.3802, -84.2814], [50.0, 0.0, -82.7485], 1.0000),
        ([50.0, -1.1848, -84.8006], [50.0, 0.0, -82.7485], 1.0000),
        ([50.0, -0.9009, -85.5211], [50.0, 0.0, -82.7485], 1.0000),
        ([50.0, 0.0, 0.0], [50.0, -1.0, 2.0], 2.3669),
    ];
    for (lab1, lab2, expected) in cases {
        let got = ciede2000(lab1, lab2);
        assert!(
            (got - expected).abs() < 5e-4,
            "ΔE2000({lab1:?}, {lab2:?}) = {got:.6}, want {expected:.4}"
        );
        // Symmetry.
        let reverse = ciede2000(lab2, lab1);
        assert!((got - reverse).abs() < 1e-9);
    }
}

#[test]
fn patch_median_ignores_outliers() {
    let mut img = solid_rgba(40, 40, [50, 50, 50]);
    // 9x9 patch of the expected colour around (20, 20).
    for y in 16..=24 {
        for x in 16..=24 {
            let i = ((y * 40 + x) * 4) as usize;
            img.pixels[i] = 200;
            img.pixels[i + 1] = 100;
            img.pixels[i + 2] = 50;
        }
    }
    // Outliers inside the sampling square must not move the median.
    for (x, y) in [(18u32, 18u32), (22, 22), (20, 18)] {
        let i = ((y * 40 + x) * 4) as usize;
        img.pixels[i] = 255;
        img.pixels[i + 1] = 255;
        img.pixels[i + 2] = 255;
    }

    let (center, rgb) = median_patch_rgb(&img, 20.0, 20.0, 4).unwrap();
    assert_eq!(center, [20, 20]);
    assert_eq!(rgb, [200, 100, 50]);

    let cfg = ColorCheckerConfig {
        patches: vec![[20.0, 20.0]],
        expected: Some(vec![srgb8_to_lab([200, 100, 50])]),
        patch_radius: Some(4),
    };
    let report = colorchecker(&img, &cfg).unwrap();
    eprintln!(
        "patch median RGB={:?} mean ΔE={:.3e} max ΔE={:.3e}",
        rgb,
        report.mean_delta_e.unwrap(),
        report.max_delta_e.unwrap()
    );
    assert_eq!(report.patches.len(), 1);
    assert!(report.mean_delta_e.unwrap() < 1e-9);
    assert!(report.max_delta_e.unwrap() < 1e-9);

    // Coordinates outside the image are rejected, not silently clamped.
    assert!(median_patch_rgb(&img, 100.0, 20.0, 4).is_err());
}

// ---------------------------------------------------------------------------
// Shadow SNR
// ---------------------------------------------------------------------------

#[test]
fn shadow_snr_recovers_known_sigma() {
    let mean = 0.18;
    let sigma = 0.02;
    let img = noisy_gray_image(128, 128, mean, sigma, 0x51a7_c0de);
    let report = shadow_snr(&img, Roi::new(0, 0, 128, 128)).unwrap();

    // 8-bit quantisation at this level adds well under 1 % to sigma.
    assert!(
        relative_error(report.mean, mean) < 0.03,
        "mean = {:.6}, want {mean}",
        report.mean
    );
    assert!(
        relative_error(report.std, sigma) < 0.05,
        "std = {:.6}, want {sigma}",
        report.std
    );
    let expected_db = 20.0 * (mean / sigma).log10();
    let got_db = report.snr_db.unwrap();
    eprintln!(
        "shadow snr mean={:.6} (want {mean}) std={:.6} (want {sigma}) snr={got_db:.4} dB (want {expected_db:.4})",
        report.mean, report.std
    );
    assert!(
        relative_error(got_db, expected_db) < 0.05,
        "SNR = {got_db:.4} dB, want {expected_db:.4} dB"
    );

    // A perfectly flat ROI has no finite SNR.
    let flat = solid_rgba(16, 16, [128, 128, 128]);
    assert!(shadow_snr(&flat, Roi::new(0, 0, 16, 16)).unwrap().snr_db.is_none());
}

// ---------------------------------------------------------------------------
// MTF50
// ---------------------------------------------------------------------------

fn check_mtf50(sigma: f64) {
    let size = 160;
    let img = gaussian_edge_image(size, sigma, 5.0f64.to_radians());
    let roi = Roi::new(16, 16, 128, 128);
    let report = mtf50(&img, roi).unwrap();
    // Gaussian MTF = exp(-2 pi^2 sigma^2 f^2) => MTF50 = 0.18739 / sigma.
    let expected = 0.187_39 / sigma;
    let error = relative_error(report.mtf50_cycles_per_px, expected);
    eprintln!(
        "sigma={sigma:.2}px mtf50={:.5} c/px (theory {:.5}, error {:.2}%, angle {:.2} deg, lsf {})",
        report.mtf50_cycles_per_px,
        expected,
        error * 100.0,
        report.edge_normal_deg,
        report.lsf_len
    );
    assert!(
        error <= 0.12,
        "sigma={sigma}: MTF50 {:.5} c/px differs from theory {:.5} by {:.2}% (limit 12%)",
        report.mtf50_cycles_per_px,
        expected,
        error * 100.0
    );
    // The slanted edge should be recovered near 5 degrees.
    assert!(
        (report.edge_normal_deg - 5.0).abs() < 2.0,
        "edge normal {:.3} deg, want ~5",
        report.edge_normal_deg
    );
}

#[test]
fn mtf50_recovers_gaussian_sigma_1_0() {
    check_mtf50(1.0);
}

#[test]
fn mtf50_recovers_gaussian_sigma_0_5() {
    check_mtf50(0.5);
}

#[test]
fn mtf50_rejects_flat_roi() {
    let flat = solid_rgba(64, 64, [128, 128, 128]);
    assert!(mtf50(&flat, Roi::new(0, 0, 64, 64)).is_err());
}

// ---------------------------------------------------------------------------
// False colour
// ---------------------------------------------------------------------------

#[test]
fn false_color_is_zero_on_neutral_gray() {
    let mut pixels = Vec::new();
    for y in 0..64u32 {
        for x in 0..64u32 {
            let v = ((x * 4 + y) % 256) as u8;
            pixels.extend_from_slice(&[v, v, v, 255]);
        }
    }
    let img = Rgba8Image::new(64, 64, pixels);
    let report = false_color(&img, Roi::new(0, 0, 64, 64), 8.0).unwrap();
    eprintln!(
        "neutral chroma ratio={} P95={:.3e} P99={:.3e} max={:.3e}",
        report.ratio, report.p95, report.p99, report.max
    );
    assert_eq!(report.ratio, 0.0, "neutral gray must have no false colour");
    // Rounding of the sRGB matrix leaves a ~2e-5 residual chroma at the
    // neutral axis; the reported ratio is still exactly zero.
    assert!(report.max < 1e-3, "neutral gray max chroma = {}", report.max);
    assert!(report.p95 < 1e-3, "neutral gray P95 chroma = {}", report.p95);
    assert!(report.p99 < 1e-3, "neutral gray P99 chroma = {}", report.p99);
}

#[test]
fn false_color_flags_colored_half() {
    let mut img = solid_rgba(64, 64, [128, 128, 128]);
    for y in 0..64 {
        for x in 32..64 {
            let i = ((y * 64 + x) * 4) as usize;
            img.pixels[i] = 255;
            img.pixels[i + 1] = 0;
            img.pixels[i + 2] = 0;
        }
    }
    let report = false_color(&img, Roi::new(0, 0, 64, 64), 8.0).unwrap();
    eprintln!(
        "red-half ratio={:.6} P95={:.2} P99={:.2} max={:.2}",
        report.ratio, report.p95, report.p99, report.max
    );
    assert!(
        (report.ratio - 0.5).abs() < 1e-9,
        "ratio = {}, want 0.5",
        report.ratio
    );
    assert!(report.p99 > 8.0);
    assert!(report.max > 8.0);
}

// ---------------------------------------------------------------------------
// CPU vs OpenCL difference
// ---------------------------------------------------------------------------

#[test]
fn diff_rgba8_reports_expected_statistics() {
    // Half of the 256 channel samples differ by 3 levels, alpha included, so
    // every reported fraction is exactly 0.5.
    let a = Rgba8Image::new(8, 8, vec![0u8; 8 * 8 * 4]);
    let mut b_pixels = vec![0u8; 8 * 8 * 4];
    for y in 0..8 {
        for x in 4..8 {
            let i = (y * 8 + x) * 4;
            for c in 0..4 {
                b_pixels[i + c] = 3;
            }
        }
    }
    let b = Rgba8Image::new(8, 8, b_pixels);
    let stats = diff_rgba8(&a, &b).unwrap();
    eprintln!(
        "diff8 max={:.3} mean={:.4} P99={:.3} over1={:.3} over2={:.3}",
        stats.max, stats.mean, stats.p99, stats.over_1_255, stats.over_2_255
    );
    assert_eq!(stats.samples, 256);
    assert!((stats.max - 3.0).abs() < 1e-9);
    assert!((stats.mean - 1.5).abs() < 1e-9);
    assert!((stats.over_1_255 - 0.5).abs() < 1e-9);
    assert!((stats.over_2_255 - 0.5).abs() < 1e-9);
    assert!((stats.p99 - 3.0).abs() < 1e-9);

    let identical = diff_rgba8(&a, &a).unwrap();
    assert_eq!(identical.max, 0.0);
    assert_eq!(identical.over_1_255, 0.0);
    assert_eq!(identical.over_2_255, 0.0);

    assert!(diff_rgba8(&a, &solid_rgba(8, 9, [0, 0, 0])).is_err());
}

#[test]
fn diff_rgba16_normalises_to_eight_bit_levels() {
    let a = Rgba16Image::new(8, 8, vec![0; 8 * 8 * 4]);
    let mut pixels = vec![0u16; 8 * 8 * 4];
    for y in 0..8usize {
        for x in 4..8usize {
            let i = (y * 8 + x) * 4;
            for c in 0..4 {
                pixels[i + c] = 3 * 257;
            }
        }
    }
    let b = Rgba16Image::new(8, 8, pixels);
    let stats = diff_rgba16(&a, &b).unwrap();
    eprintln!(
        "diff16 max={:.4} mean={:.4} P99={:.3} over1={:.3} over2={:.3}",
        stats.max, stats.mean, stats.p99, stats.over_1_255, stats.over_2_255
    );
    assert!((stats.max - 3.0).abs() < 1e-6);
    assert!((stats.mean - 1.5).abs() < 1e-6);
    assert!((stats.over_1_255 - 0.5).abs() < 1e-9);
    assert!((stats.over_2_255 - 0.5).abs() < 1e-9);
}

#[test]
fn diff_supports_mixed_bit_depth() {
    let rgba8 = solid_rgba(4, 4, [10, 20, 30]);
    let rgba16 = Rgba16Image::new(
        4,
        4,
        rgba8.pixels.iter().map(|&v| v as u16 * 257).collect(),
    );
    let stats = LoadedImage::Rgba8(rgba8)
        .diff(&LoadedImage::Rgba16(rgba16))
        .unwrap();
    assert_eq!(stats.max, 0.0);
    assert_eq!(stats.over_1_255, 0.0);
}

// ---------------------------------------------------------------------------
// RAW cache: cold decode vs cache hit vs second cold decode
// ---------------------------------------------------------------------------

#[cfg(feature = "external-fixtures")]
fn unique_cache_dir(tag: &str) -> std::path::PathBuf {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!(
        "aurora-quality-{}-{tag}-{nanos}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[cfg(feature = "external-fixtures")]
#[test]
#[ignore = "需要 AURORA_QUALITY_FIXTURES 指向含真实 RAW 的 manifest"]
fn raw_cold_decode_matches_cache_and_second_cold_decode() {
    let manifest_path = std::env::var("AURORA_QUALITY_FIXTURES")
        .expect("请设置 AURORA_QUALITY_FIXTURES 指向 manifest JSON");
    let manifest =
        metrics::load_manifest(std::path::Path::new(&manifest_path)).expect("读取 manifest");
    let sample = manifest
        .samples
        .iter()
        .find(|s| s.raw.is_some())
        .expect("manifest 中没有带 raw 字段的样本");
    let raw = std::path::PathBuf::from(sample.raw.as_ref().unwrap());
    assert!(raw.is_file(), "RAW 不存在: {}", raw.display());
    let long_side = sample.raw_long_side.unwrap_or(2560);

    let cache_a = unique_cache_dir("a");
    let cache_b = unique_cache_dir("b");

    // First call: cold decode, writes the cache entry.
    let cold = aurora_engine::raw_cache::get_or_decode(&raw, &cache_a, long_side)
        .expect("首次冷解码");
    // Second call with the same directory: cache hit.
    let hit = aurora_engine::raw_cache::get_or_decode(&raw, &cache_a, long_side)
        .expect("缓存命中的解码");
    assert_eq!(cold.width, hit.width);
    assert_eq!(cold.height, hit.height);
    assert_eq!(cold.pixels, hit.pixels, "缓存命中必须与冷解码逐像素一致");

    // Second cold decode into another empty directory: must be deterministic.
    let cold_again = aurora_engine::raw_cache::get_or_decode(&raw, &cache_b, long_side)
        .expect("第二次冷解码");
    assert_eq!(cold.width, cold_again.width);
    assert_eq!(cold.height, cold_again.height);
    assert_eq!(
        cold.pixels, cold_again.pixels,
        "两个空缓存目录的冷解码必须逐像素一致"
    );

    let _ = std::fs::remove_dir_all(&cache_a);
    let _ = std::fs::remove_dir_all(&cache_b);
}
