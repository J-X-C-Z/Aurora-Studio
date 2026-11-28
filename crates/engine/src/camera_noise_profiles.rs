//! Per-camera RAW linear-domain noise model and banding lookup tables.
//!
//! Purpose: give demosaic / linear denoise a camera-aware starting point for
//! the photon-transfer slope, the read-noise floor and the dual-conversion-gain
//! switch, instead of one global constant for every body.
//!
//! `std` only. One `String` is allocated per [`lookup_profile`] call (the
//! normalised needle); everything else is static.
//!
//! ## Noise model
//!
//! Signal is normalised linear light in `0.0..=1.0`, where `1.0` is the
//! saturation level **of the ISO being asked about** — i.e. the caller has
//! already removed the black level and divided by the per-ISO white level.
//! Noise variance at signal `s` is:
//!
//! ```text
//! sigma^2(s) = shot_slope * s + read_floor^2        s in 0..=1
//! ```
//!
//! * `shot_slope` — photon-transfer term: variance per unit signal. Dominates
//!   above roughly 1 % of saturation.
//! * `read_floor` — read noise expressed as an RMS *fraction of saturation*
//!   (so it is not squared inside the table tuple). Dominates in deep shadow.
//!
//! Both columns are monotone non-decreasing in ISO. Interpolation is
//! log-linear in ISO (`x` axis is `ln(iso)`, `y` axis is linear in the value).
//! ISO below the first knot is clamped; ISO above the last knot is
//! extrapolated monotonically by the ISO ratio, then clamped to a sane
//! magnitude. [`interpolate_noise`] never returns NaN, infinity, zero or a
//! negative number, for any `u32` ISO.
//!
//! ## ⚠ 数据来源 / Data provenance
//!
//! 下表中的每一个数字都是**工程估算起点**：按公开的满阱容量量级
//! (full well ≈ 2e4..6e4 e-) 与读出噪声量级 (≈2..3 e- RMS) 推出，再叠加
//! 一个单调的偏离因子。它们**不是**任何机型、任何第三方数据库（Photons to
//! Photos / DxO / 厂方白皮书）的实测值，也没有做过真机平场/暗场标定，更不
//! 是原厂管线参数。`dual_gain_iso` 取的是常见资料里的切换点量级，同样待
//! 实拍确认。
//!
//! These numbers are placeholders with the right order of magnitude and the
//! right monotonicity, nothing more. Replace `iso_noise_knots` with fitted
//! values from real flat-field / dark-frame captures before shipping any
//! quality claim, and record the calibration run in `06_reports/`.
//!
//! Until then: never describe this table as a manufacturer measurement.

/// Static description of one camera body's RAW noise behaviour.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraRawProfile {
    /// EXIF `Model` spellings. Compared case-insensitively with every
    /// whitespace and control character removed.
    pub model_aliases: &'static [&'static str],
    /// Lowest non-extended ISO of the body.
    pub native_base_iso: u32,
    /// ISO at which the sensor switches to its high-conversion-gain mode.
    pub dual_gain_iso: Option<u32>,
    /// `(iso, shot_slope, read_floor)` in ascending ISO order. Every body
    /// carries an exact knot at its `dual_gain_iso` so callers can read the
    /// switch point out of the table instead of special-casing it.
    pub iso_noise_knots: &'static [(u32, f32, f32)],
    /// Native RAW container depth used by the default shooting mode.
    pub default_bit_depth: u8,
    /// X-Trans CFA (affects demosaic, not the noise numbers).
    pub is_xtrans: bool,
}

/// All supported bodies, in the order Nikon / Sony / Canon / Fujifilm.
///
/// 数值为工程估算，见文件头 "数据来源" 一节。
pub const CAMERA_PROFILES: &[CameraRawProfile] = &[
    // ---------------------------------------------------------------- Nikon
    CameraRawProfile {
        model_aliases: &["NIKON Z 6_3", "NIKON Z6_3", "Z6III", "NIKON Z6III"],
        native_base_iso: 100,
        dual_gain_iso: Some(800),
        iso_noise_knots: &[
            (100, 1.8181818e-05, 4.0000000e-05),
            (200, 3.6363636e-05, 8.4000000e-05),
            (400, 7.2727273e-05, 1.7600000e-04),
            (800, 1.4545455e-04, 3.6800000e-04),
            (1600, 2.9090909e-04, 7.6800000e-04),
            (3200, 5.8181818e-04, 1.6000000e-03),
            (6400, 1.1636364e-03, 3.3280000e-03),
            (12800, 2.3272727e-03, 6.9120000e-03),
            (25600, 4.6545455e-03, 1.3824000e-02),
            (51200, 9.3090909e-03, 2.7648000e-02),
            (102400, 1.8618182e-02, 5.5296000e-02),
        ],
        default_bit_depth: 14,
        is_xtrans: false,
    },
    CameraRawProfile {
        model_aliases: &["NIKON Z 6_2", "NIKON Z6_2", "Z6II", "NIKON Z6II"],
        native_base_iso: 100,
        dual_gain_iso: Some(800),
        iso_noise_knots: &[
            (100, 2.0000000e-05, 5.2000000e-05),
            (200, 4.0000000e-05, 1.0920000e-04),
            (400, 8.0000000e-05, 2.2880000e-04),
            (800, 1.6000000e-04, 4.7840000e-04),
            (1600, 3.2000000e-04, 9.9840000e-04),
            (3200, 6.4000000e-04, 2.0800000e-03),
            (6400, 1.2800000e-03, 4.3264000e-03),
            (12800, 2.5600000e-03, 8.9856000e-03),
            (25600, 5.1200000e-03, 1.7971200e-02),
            (51200, 1.0240000e-02, 3.5942400e-02),
            (102400, 2.0480000e-02, 7.1884800e-02),
        ],
        default_bit_depth: 14,
        is_xtrans: false,
    },
    CameraRawProfile {
        model_aliases: &["NIKON Z 7_2", "NIKON Z7_2", "Z7II", "NIKON Z7II"],
        native_base_iso: 64,
        dual_gain_iso: Some(400),
        iso_noise_knots: &[
            (64, 2.5000000e-05, 6.2500000e-05),
            (128, 5.0000000e-05, 1.3125000e-04),
            (256, 1.0000000e-04, 2.7500000e-04),
            (400, 1.5625000e-04, 4.4226282e-04),
            (512, 2.0000000e-04, 5.7500000e-04),
            (1024, 4.0000000e-04, 1.2000000e-03),
            (2048, 8.0000000e-04, 2.5000000e-03),
            (4096, 1.6000000e-03, 5.2000000e-03),
            (8192, 3.2000000e-03, 1.0800000e-02),
            (16384, 6.4000000e-03, 2.1600000e-02),
            (32768, 1.2800000e-02, 4.3200000e-02),
            (65536, 2.5600000e-02, 8.6400000e-02),
            (131072, 5.1200000e-02, 1.7280000e-01),
        ],
        default_bit_depth: 14,
        is_xtrans: false,
    },
    CameraRawProfile {
        model_aliases: &["NIKON Z 8", "NIKON Z8", "Z8"],
        native_base_iso: 64,
        dual_gain_iso: Some(500),
        iso_noise_knots: &[
            (64, 2.5000000e-05, 6.0000000e-05),
            (128, 5.0000000e-05, 1.2600000e-04),
            (256, 1.0000000e-04, 2.6400000e-04),
            (500, 1.9531250e-04, 5.3826057e-04),
            (512, 2.0000000e-04, 5.5200000e-04),
            (1024, 4.0000000e-04, 1.1520000e-03),
            (2048, 8.0000000e-04, 2.4000000e-03),
            (4096, 1.6000000e-03, 4.9920000e-03),
            (8192, 3.2000000e-03, 1.0368000e-02),
            (16384, 6.4000000e-03, 2.0736000e-02),
            (32768, 1.2800000e-02, 4.1472000e-02),
            (65536, 2.5600000e-02, 8.2944000e-02),
            (131072, 5.1200000e-02, 1.6588800e-01),
        ],
        default_bit_depth: 14,
        is_xtrans: false,
    },
    CameraRawProfile {
        model_aliases: &["NIKON Z 9", "NIKON Z9", "Z9"],
        native_base_iso: 64,
        dual_gain_iso: Some(500),
        iso_noise_knots: &[
            (64, 2.5000000e-05, 5.7500000e-05),
            (128, 5.0000000e-05, 1.2075000e-04),
            (256, 1.0000000e-04, 2.5300000e-04),
            (500, 1.9531250e-04, 5.1583305e-04),
            (512, 2.0000000e-04, 5.2900000e-04),
            (1024, 4.0000000e-04, 1.1040000e-03),
            (2048, 8.0000000e-04, 2.3000000e-03),
            (4096, 1.6000000e-03, 4.7840000e-03),
            (8192, 3.2000000e-03, 9.9360000e-03),
            (16384, 6.4000000e-03, 1.9872000e-02),
            (32768, 1.2800000e-02, 3.9744000e-02),
            (65536, 2.5600000e-02, 7.9488000e-02),
            (131072, 5.1200000e-02, 1.5897600e-01),
        ],
        default_bit_depth: 14,
        is_xtrans: false,
    },
    // ----------------------------------------------------------------- Sony
    CameraRawProfile {
        model_aliases: &["ILCE-7M4", "SONY ILCE-7M4", "A7M4", "A7 IV"],
        native_base_iso: 100,
        dual_gain_iso: Some(800),
        iso_noise_knots: &[
            (100, 2.2222222e-05, 5.3333333e-05),
            (200, 4.4444444e-05, 1.1200000e-04),
            (400, 8.8888889e-05, 2.3466667e-04),
            (800, 1.7777778e-04, 4.9066667e-04),
            (1600, 3.5555556e-04, 1.0240000e-03),
            (3200, 7.1111111e-04, 2.1333333e-03),
            (6400, 1.4222222e-03, 4.4373333e-03),
            (12800, 2.8444444e-03, 9.2160000e-03),
            (25600, 5.6888889e-03, 1.8432000e-02),
            (51200, 1.1377778e-02, 3.6864000e-02),
            (102400, 2.2755556e-02, 7.3728000e-02),
        ],
        default_bit_depth: 14,
        is_xtrans: false,
    },
    CameraRawProfile {
        model_aliases: &["ILCE-7RM5", "SONY ILCE-7RM5", "A7RM5", "A7R V"],
        native_base_iso: 100,
        dual_gain_iso: Some(800),
        iso_noise_knots: &[
            (100, 3.3333333e-05, 7.3333333e-05),
            (200, 6.6666667e-05, 1.5400000e-04),
            (400, 1.3333333e-04, 3.2266667e-04),
            (800, 2.6666667e-04, 6.7466667e-04),
            (1600, 5.3333333e-04, 1.4080000e-03),
            (3200, 1.0666667e-03, 2.9333333e-03),
            (6400, 2.1333333e-03, 6.1013333e-03),
            (12800, 4.2666667e-03, 1.2672000e-02),
            (25600, 8.5333333e-03, 2.5344000e-02),
            (51200, 1.7066667e-02, 5.0688000e-02),
            (102400, 3.4133333e-02, 1.0137600e-01),
        ],
        default_bit_depth: 14,
        is_xtrans: false,
    },
    CameraRawProfile {
        model_aliases: &["ILCE-7CM2", "SONY ILCE-7CM2", "A7CM2", "A7C II"],
        native_base_iso: 100,
        dual_gain_iso: Some(800),
        iso_noise_knots: &[
            (100, 2.2222222e-05, 5.7777778e-05),
            (200, 4.4444444e-05, 1.2133333e-04),
            (400, 8.8888889e-05, 2.5422222e-04),
            (800, 1.7777778e-04, 5.3155556e-04),
            (1600, 3.5555556e-04, 1.1093333e-03),
            (3200, 7.1111111e-04, 2.3111111e-03),
            (6400, 1.4222222e-03, 4.8071111e-03),
            (12800, 2.8444444e-03, 9.9840000e-03),
            (25600, 5.6888889e-03, 1.9968000e-02),
            (51200, 1.1377778e-02, 3.9936000e-02),
            (102400, 2.2755556e-02, 7.9872000e-02),
        ],
        default_bit_depth: 14,
        is_xtrans: false,
    },
    CameraRawProfile {
        model_aliases: &["ILCE-1", "SONY ILCE-1", "A1"],
        native_base_iso: 100,
        dual_gain_iso: Some(800),
        iso_noise_knots: &[
            (100, 2.8571429e-05, 5.7142857e-05),
            (200, 5.7142857e-05, 1.2000000e-04),
            (400, 1.1428571e-04, 2.5142857e-04),
            (800, 2.2857143e-04, 5.2571429e-04),
            (1600, 4.5714286e-04, 1.0971429e-03),
            (3200, 9.1428571e-04, 2.2857143e-03),
            (6400, 1.8285714e-03, 4.7542857e-03),
            (12800, 3.6571429e-03, 9.8742857e-03),
            (25600, 7.3142857e-03, 1.9748571e-02),
            (51200, 1.4628571e-02, 3.9497143e-02),
            (102400, 2.9257143e-02, 7.8994286e-02),
        ],
        default_bit_depth: 14,
        is_xtrans: false,
    },
    // ---------------------------------------------------------------- Canon
    CameraRawProfile {
        model_aliases: &["Canon EOS R5", "EOS R5", "EOSR5"],
        native_base_iso: 100,
        dual_gain_iso: Some(400),
        iso_noise_knots: &[
            (100, 2.8571429e-05, 7.4285714e-05),
            (200, 5.7142857e-05, 1.5600000e-04),
            (400, 1.1428571e-04, 3.2685714e-04),
            (800, 2.2857143e-04, 6.8342857e-04),
            (1600, 4.5714286e-04, 1.4262857e-03),
            (3200, 9.1428571e-04, 2.9714286e-03),
            (6400, 1.8285714e-03, 6.1805714e-03),
            (12800, 3.6571429e-03, 1.2836571e-02),
            (25600, 7.3142857e-03, 2.5673143e-02),
            (51200, 1.4628571e-02, 5.1346286e-02),
            (102400, 2.9257143e-02, 1.0269257e-01),
        ],
        default_bit_depth: 14,
        is_xtrans: false,
    },
    CameraRawProfile {
        model_aliases: &[
            "Canon EOS R6m2",
            "Canon EOS R6 Mark II",
            "EOS R6m2",
            "EOS R6 Mark II",
            "EOS R6 II",
        ],
        native_base_iso: 100,
        dual_gain_iso: Some(400),
        iso_noise_knots: &[
            (100, 1.8181818e-05, 5.0909091e-05),
            (200, 3.6363636e-05, 1.0690909e-04),
            (400, 7.2727273e-05, 2.2400000e-04),
            (800, 1.4545455e-04, 4.6836364e-04),
            (1600, 2.9090909e-04, 9.7745455e-04),
            (3200, 5.8181818e-04, 2.0363636e-03),
            (6400, 1.1636364e-03, 4.2356364e-03),
            (12800, 2.3272727e-03, 8.7970909e-03),
            (25600, 4.6545455e-03, 1.7594182e-02),
            (51200, 9.3090909e-03, 3.5188364e-02),
            (102400, 1.8618182e-02, 7.0376727e-02),
        ],
        default_bit_depth: 14,
        is_xtrans: false,
    },
    CameraRawProfile {
        model_aliases: &["Canon EOS R3", "EOS R3", "EOSR3"],
        native_base_iso: 100,
        dual_gain_iso: Some(800),
        iso_noise_knots: &[
            (100, 1.6666667e-05, 4.1666667e-05),
            (200, 3.3333333e-05, 8.7500000e-05),
            (400, 6.6666667e-05, 1.8333333e-04),
            (800, 1.3333333e-04, 3.8333333e-04),
            (1600, 2.6666667e-04, 8.0000000e-04),
            (3200, 5.3333333e-04, 1.6666667e-03),
            (6400, 1.0666667e-03, 3.4666667e-03),
            (12800, 2.1333333e-03, 7.2000000e-03),
            (25600, 4.2666667e-03, 1.4400000e-02),
            (51200, 8.5333333e-03, 2.8800000e-02),
            (102400, 1.7066667e-02, 5.7600000e-02),
        ],
        default_bit_depth: 14,
        is_xtrans: false,
    },
    // ------------------------------------------------------------- Fujifilm
    CameraRawProfile {
        model_aliases: &["X-T5", "FUJIFILM X-T5", "FUJIFILM XT5"],
        native_base_iso: 125,
        dual_gain_iso: Some(500),
        iso_noise_knots: &[
            (125, 5.0000000e-05, 1.3000000e-04),
            (250, 1.0000000e-04, 2.7300000e-04),
            (500, 2.0000000e-04, 5.7200000e-04),
            (1000, 4.0000000e-04, 1.1960000e-03),
            (2000, 8.0000000e-04, 2.4960000e-03),
            (4000, 1.6000000e-03, 5.2000000e-03),
            (8000, 3.2000000e-03, 1.0816000e-02),
            (16000, 6.4000000e-03, 2.2464000e-02),
            (32000, 1.2800000e-02, 4.4928000e-02),
            (64000, 2.5600000e-02, 8.9856000e-02),
            (128000, 5.1200000e-02, 1.7971200e-01),
        ],
        default_bit_depth: 14,
        is_xtrans: true,
    },
    CameraRawProfile {
        model_aliases: &["X-H2", "FUJIFILM X-H2", "FUJIFILM XH2"],
        native_base_iso: 125,
        dual_gain_iso: Some(500),
        iso_noise_knots: &[
            (125, 5.0000000e-05, 1.2000000e-04),
            (250, 1.0000000e-04, 2.5200000e-04),
            (500, 2.0000000e-04, 5.2800000e-04),
            (1000, 4.0000000e-04, 1.1040000e-03),
            (2000, 8.0000000e-04, 2.3040000e-03),
            (4000, 1.6000000e-03, 4.8000000e-03),
            (8000, 3.2000000e-03, 9.9840000e-03),
            (16000, 6.4000000e-03, 2.0736000e-02),
            (32000, 1.2800000e-02, 4.1472000e-02),
            (64000, 2.5600000e-02, 8.2944000e-02),
            (128000, 5.1200000e-02, 1.6588800e-01),
        ],
        default_bit_depth: 14,
        is_xtrans: false,
    },
    CameraRawProfile {
        model_aliases: &["X-S20", "FUJIFILM X-S20", "FUJIFILM XS20"],
        native_base_iso: 160,
        dual_gain_iso: Some(500),
        iso_noise_knots: &[
            (160, 3.8461538e-05, 1.1538462e-04),
            (320, 7.6923077e-05, 2.4230769e-04),
            (500, 1.2019231e-04, 3.9021375e-04),
            (640, 1.5384615e-04, 5.0769231e-04),
            (1280, 3.0769231e-04, 1.0615385e-03),
            (2560, 6.1538462e-04, 2.2153846e-03),
            (5120, 1.2307692e-03, 4.6153846e-03),
            (10240, 2.4615385e-03, 9.6000000e-03),
            (20480, 4.9230769e-03, 1.9938462e-02),
            (40960, 9.8461538e-03, 3.9876923e-02),
            (81920, 1.9692308e-02, 7.9753846e-02),
            (163840, 3.9384615e-02, 1.5950769e-01),
        ],
        default_bit_depth: 14,
        is_xtrans: true,
    },
];

/// Smallest value the model will ever return, so callers may divide by it
/// without a special case.
const MIN_NOISE: f32 = 1.0e-7;

/// Upper sanity clamp for extrapolation far beyond the last knot. A value this
/// large is meaningless for a denoiser; it exists only to keep the contract
/// "finite, positive" for any `u32` ISO.
const MAX_NOISE: f32 = 1.0e4;

/// Lower-case, strip whitespace and control characters. EXIF writers pad
/// `Model` with spaces and sometimes NULs.
fn normalize_model(model: &str) -> String {
    let mut out = String::with_capacity(model.len());
    for ch in model.chars() {
        if ch.is_whitespace() || ch.is_control() {
            continue;
        }
        out.push(ch.to_ascii_lowercase());
    }
    out
}

/// Compare an alias against an already-normalised needle without allocating.
fn alias_matches(alias: &str, needle: &str) -> bool {
    let mut want = needle.chars();
    for ch in alias.chars() {
        if ch.is_whitespace() || ch.is_control() {
            continue;
        }
        match want.next() {
            Some(expected) if expected == ch.to_ascii_lowercase() => {}
            _ => return false,
        }
    }
    want.next().is_none()
}

/// Resolve an EXIF `Model` string to a profile.
///
/// Matching is exact after normalisation: case-insensitive, whitespace and
/// control characters ignored. `"NIKON Z 6_3"`, `"nikon z6_3"` and
/// `" NIKON  Z 6_3 "` all resolve; `"NIKON Z 6"` does not (it is a different
/// body and deliberately has no alias here).
///
/// Cold path: allocates one `String` per call.
pub fn lookup_profile(camera_model: &str) -> Option<&'static CameraRawProfile> {
    let needle = normalize_model(camera_model);
    if needle.is_empty() {
        return None;
    }
    CAMERA_PROFILES
        .iter()
        .find(|profile| profile.model_aliases.iter().any(|alias| alias_matches(alias, &needle)))
}

/// `(shot_slope, read_floor)` for one camera model at one ISO, or `None` when
/// the model is unknown. See [`interpolate_noise`].
pub fn lookup_noise(camera_model: &str, iso: u32) -> Option<[f32; 2]> {
    lookup_profile(camera_model).map(|profile| interpolate_noise(profile, iso))
}

/// Interpolate `(shot_slope, read_floor)` for an already-resolved profile.
///
/// * Inside the table: linear in `ln(iso)`, linear in the value.
/// * `iso` at or below the first knot: clamped to the first knot. There is no
///   meaningful extrapolation towards ISO 0.
/// * `iso` at or above the last knot: scaled by `iso / last_iso`. Both terms
///   are ISO-referred, so they grow with the ISO ratio; this is monotone in
///   ISO and cannot overflow `f32` for any `u32` input.
///
/// The result is always finite, strictly positive and non-decreasing in `iso`
/// for a well-formed (ascending, positive) table.
pub fn interpolate_noise(profile: &CameraRawProfile, iso: u32) -> [f32; 2] {
    let knots = profile.iso_noise_knots;

    // Defensive: never panic in a lookup that feeds a denoiser.
    let Some(first) = knots.first() else {
        return [MIN_NOISE, MIN_NOISE];
    };
    let last = knots[knots.len() - 1];

    // ISO 0 is not a camera setting; treating it as ISO 1 keeps `ln` finite.
    let iso = iso.max(1);

    if iso <= first.0 {
        return [sanitize(first.1), sanitize(first.2)];
    }
    if iso >= last.0 {
        let ratio = iso as f32 / last.0 as f32;
        return [sanitize(last.1 * ratio), sanitize(last.2 * ratio)];
    }

    let bracket = knots.windows(2).find(|w| iso >= w[0].0 && iso <= w[1].0);
    let (lo, hi) = match bracket {
        // A malformed, non-ascending table lands here: fall back to the ends
        // instead of returning garbage.
        Some(w) => (w[0], w[1]),
        None => (*first, last),
    };

    let t = if hi.0 == lo.0 {
        0.0
    } else {
        let num = (iso as f32 / lo.0 as f32).ln();
        let den = (hi.0 as f32 / lo.0 as f32).ln();
        if den.is_finite() && den != 0.0 { (num / den).clamp(0.0, 1.0) } else { 0.0 }
    };

    let shot = lo.1 + (hi.1 - lo.1) * t;
    let read = lo.2 + (hi.2 - lo.2) * t;
    [sanitize(shot), sanitize(read)]
}

/// Variance of the linear-domain noise at `signal` (normalised, `0.0..=1.0`,
/// `1.0` = saturation of the requested ISO).
///
/// This is the model the table encodes: `shot_slope * s + read_floor^2`.
pub fn noise_variance(profile: &CameraRawProfile, iso: u32, signal: f32) -> f32 {
    let [shot, read] = interpolate_noise(profile, iso);
    let s = signal.clamp(0.0, 1.0);
    shot * s + read * read
}

/// Standard deviation of the linear-domain noise, in the same normalised
/// units as [`noise_variance`].
pub fn noise_sigma(profile: &CameraRawProfile, iso: u32, signal: f32) -> f32 {
    noise_variance(profile, iso, signal).sqrt()
}

fn sanitize(value: f32) -> f32 {
    if value.is_finite() && value > MIN_NOISE {
        value.min(MAX_NOISE)
    } else {
        MIN_NOISE
    }
}

// ---------------------------------------------------------------------------
// Banding (row noise / column striping) rules
// ---------------------------------------------------------------------------

/// One banding rule. `from_iso` opens a range; the last rule whose `from_iso`
/// is `<= iso` wins.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BandingRule {
    /// Applies to every ISO at or above this value.
    pub from_iso: u32,
    /// Banding amplitude as a fraction of `read_floor`.
    ///
    /// `0.0` means **uncalibrated**: no measured banding model exists for this
    /// family yet, so a consumer must not synthesise a correction from it.
    pub amplitude: f32,
    /// Vertical striping period in pixels (PDAF row / column artefacts).
    /// `0.0` means none known.
    pub period_px: f32,
}

/// Placeholder rule set. Every entry is uncalibrated `amplitude == 0.0`.
///
/// 这是刻意留空的：没有真机暗场与 PDAF 行统计之前，凭空写一个带幅度的
/// 规则会让降噪在无辜图像上挖出条纹。表的存在只是为了固定接口位置，等
/// 标定数据到位后直接替换本数组。
static BANDING_RULES: [BandingRule; 1] =
    [BandingRule { from_iso: 1, amplitude: 0.0, period_px: 0.0 }];

/// Banding rules for a known camera model, or `None` for an unknown one.
pub fn lookup_banding(camera_model: &str) -> Option<&'static [BandingRule]> {
    lookup_profile(camera_model).map(|_| &BANDING_RULES[..])
}

/// The single banding rule in force at `iso`, or `None` for an unknown model.
pub fn resolve_banding(camera_model: &str, iso: u32) -> Option<BandingRule> {
    let rules = lookup_banding(camera_model)?;
    let iso = iso.max(1);
    let mut chosen = None;
    for rule in rules {
        if iso >= rule.from_iso {
            chosen = Some(*rule);
        }
    }
    chosen.or_else(|| rules.first().copied())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// One canonical EXIF `Model` string per supported body, in table order.
    const CANONICAL: [&str; 15] = [
        "NIKON Z 6_3",
        "NIKON Z 6_2",
        "NIKON Z 7_2",
        "NIKON Z 8",
        "NIKON Z 9",
        "ILCE-7M4",
        "ILCE-7RM5",
        "ILCE-7CM2",
        "ILCE-1",
        "Canon EOS R5",
        "Canon EOS R6m2",
        "Canon EOS R3",
        "X-T5",
        "X-H2",
        "X-S20",
    ];

    /// A dense ISO ladder that crosses every 1/3-stop a user is likely to see.
    const ISO_SWEEP: [u32; 34] = [
        50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600, 2000,
        2500, 3200, 4000, 5000, 6400, 8000, 10000, 12800, 16000, 20000, 25600, 32000, 40000,
        51200, 64000, 80000, 102400,
    ];

    fn same_profile(input: &str, canon: &str) -> bool {
        match (lookup_profile(input), lookup_profile(canon)) {
            (Some(a), Some(b)) => a.model_aliases == b.model_aliases,
            _ => false,
        }
    }

    fn assert_close(actual: f32, expected: f32, ctx: &str) {
        let scale = actual.abs().max(expected.abs()).max(1.0e-12);
        assert!(
            (actual - expected).abs() / scale < 1.0e-6,
            "{ctx}: {actual:e} != {expected:e}"
        );
    }

    #[test]
    fn table_covers_every_target_body() {
        assert_eq!(CAMERA_PROFILES.len(), 15, "profile count changed");
        for (profile, canon) in CAMERA_PROFILES.iter().zip(CANONICAL) {
            assert!(
                profile.model_aliases.contains(&canon),
                "{canon} is missing from its own alias list"
            );
            assert!(lookup_profile(canon).is_some(), "{canon} does not resolve");
            assert!(profile.native_base_iso >= 50, "{canon}: implausible base ISO");
            assert!(
                profile.default_bit_depth == 12 || profile.default_bit_depth == 14,
                "{canon}: unexpected bit depth"
            );
            assert!(
                profile.iso_noise_knots.len() >= 8,
                "{canon}: too few noise knots to interpolate"
            );
        }
    }

    #[test]
    fn every_alias_resolves_to_its_own_body() {
        for profile in CAMERA_PROFILES {
            for alias in profile.model_aliases {
                let found = lookup_profile(alias)
                    .unwrap_or_else(|| panic!("alias {alias:?} does not resolve"));
                assert_eq!(
                    found.model_aliases, profile.model_aliases,
                    "alias {alias:?} resolved to the wrong body"
                );
            }
        }
    }

    #[test]
    fn matching_tolerates_case_and_whitespace() {
        assert!(same_profile("  nIkOn   z 6_3  ", "NIKON Z 6_3"));
        assert!(same_profile("NIKONZ6_3\t", "NIKON Z 6_3"));
        assert!(same_profile("\nnikon z 8\n", "NIKON Z 8"));
        assert!(same_profile("ILCE-7M4 ", "ILCE-7M4"));
        assert!(same_profile("SONY  ILCE-7M4", "ILCE-7M4"));
        assert!(same_profile(" canon eos r6mark ii ", "Canon EOS R6m2"));
        assert!(same_profile("fujifilm x-t5", "X-T5"));
        assert!(same_profile(" FUJIFILM  X-S20 ", "X-S20"));
        assert!(same_profile("Canon EOS R6m2\u{0}", "Canon EOS R6m2"));
    }

    #[test]
    fn unknown_or_ambiguous_models_return_none() {
        assert!(lookup_profile("").is_none());
        assert!(lookup_profile("   \t\n").is_none());
        assert!(lookup_profile("NIKON Z 6").is_none(), "Z6 is not a listed body");
        assert!(lookup_profile("SONY ILCE-9").is_none());
        assert!(lookup_profile("Canon EOS R").is_none());
        assert!(lookup_profile("X-T4").is_none());
        assert!(lookup_profile("X-H2S").is_none());
        assert!(lookup_noise("NIKON Z 6", 800).is_none());
    }

    #[test]
    fn knots_are_ascending_and_strictly_positive() {
        for profile in CAMERA_PROFILES {
            let id = profile.model_aliases[0];
            let knots = profile.iso_noise_knots;
            for pair in knots.windows(2) {
                assert!(
                    pair[0].0 < pair[1].0,
                    "{id}: knots not ascending at ISO {}",
                    pair[0].0
                );
                assert!(
                    pair[0].1 <= pair[1].1 && pair[0].2 <= pair[1].2,
                    "{id}: noise decreases between ISO {} and {}",
                    pair[0].0,
                    pair[1].0
                );
            }
            for knot in knots {
                assert!(knot.1.is_finite() && knot.1 > 0.0, "{id}: bad shot_slope at {}", knot.0);
                assert!(knot.2.is_finite() && knot.2 > 0.0, "{id}: bad read_floor at {}", knot.0);
            }
            assert!(
                knots[knots.len() - 1].0 >= 102400,
                "{id}: table must reach ISO 102400"
            );
        }
    }

    #[test]
    fn every_body_declares_its_dual_gain_knot() {
        for profile in CAMERA_PROFILES {
            let id = profile.model_aliases[0];
            let dual = profile
                .dual_gain_iso
                .unwrap_or_else(|| panic!("{id}: no dual gain switch declared"));
            assert!(dual > profile.native_base_iso, "{id}: dual gain must sit above base ISO");

            let knots = profile.iso_noise_knots;
            let idx = knots
                .iter()
                .position(|k| k.0 == dual)
                .unwrap_or_else(|| panic!("{id}: dual gain ISO {dual} is not a knot"));
            assert!(
                idx > 0 && idx + 1 < knots.len(),
                "{id}: dual gain must be an interior knot"
            );

            // The switch is readable straight out of the table.
            let [shot, read] = interpolate_noise(profile, dual);
            assert_close(shot, knots[idx].1, &format!("{id} shot@dual"));
            assert_close(read, knots[idx].2, &format!("{id} read@dual"));

            // And the curve keeps rising across it — no inversion, so a caller
            // may pick either side without a discontinuity in ordering.
            assert!(
                knots[idx - 1].1 < knots[idx].1 && knots[idx].1 < knots[idx + 1].1,
                "{id}: shot_slope not increasing through the switch"
            );
            assert!(
                knots[idx - 1].2 < knots[idx].2 && knots[idx].2 < knots[idx + 1].2,
                "{id}: read_floor not increasing through the switch"
            );
        }
    }

    #[test]
    fn knots_are_reproduced_exactly_by_the_interpolator() {
        for profile in CAMERA_PROFILES {
            let id = profile.model_aliases[0];
            for knot in profile.iso_noise_knots {
                let [shot, read] = interpolate_noise(profile, knot.0);
                assert_close(shot, knot.1, &format!("{id} shot@{}", knot.0));
                assert_close(read, knot.2, &format!("{id} read@{}", knot.0));
            }
        }
    }

    #[test]
    fn extreme_isos_stay_finite_and_positive() {
        for profile in CAMERA_PROFILES {
            let id = profile.model_aliases[0];
            for iso in [50u32, 102400] {
                let [shot, read] = interpolate_noise(profile, iso);
                assert!(shot.is_finite() && shot > 0.0, "{id}: shot at ISO {iso} = {shot}");
                assert!(read.is_finite() && read > 0.0, "{id}: read at ISO {iso} = {read}");
            }
            // Sanity: one stop at 102400 must be far noisier than ISO 50.
            let [low_shot, low_read] = interpolate_noise(profile, 50);
            let [high_shot, high_read] = interpolate_noise(profile, 102400);
            assert!(high_shot > low_shot * 5.0, "{id}: shot noise barely grew");
            assert!(high_read > low_read * 5.0, "{id}: read noise barely grew");
        }
    }

    #[test]
    fn noise_is_monotone_across_the_iso_sweep() {
        for profile in CAMERA_PROFILES {
            let id = profile.model_aliases[0];
            let mut prev = interpolate_noise(profile, ISO_SWEEP[0]);
            for iso in ISO_SWEEP.iter().copied().skip(1) {
                let cur = interpolate_noise(profile, iso);
                assert!(
                    cur[0] >= prev[0] && cur[1] >= prev[1],
                    "{id}: noise dropped from ISO {} to {iso}",
                    iso
                );
                prev = cur;
            }
            assert!(
                prev[0] > interpolate_noise(profile, ISO_SWEEP[0])[0],
                "{id}: shot noise not strictly rising over the sweep"
            );
        }
    }

    #[test]
    fn absurd_isos_stay_inside_the_contract() {
        for profile in CAMERA_PROFILES {
            let id = profile.model_aliases[0];
            for iso in [0u32, 1, 3, 4_000_000, u32::MAX] {
                let [shot, read] = interpolate_noise(profile, iso);
                assert!(
                    shot.is_finite() && shot > 0.0 && shot <= MAX_NOISE,
                    "{id}: shot at ISO {iso} = {shot}"
                );
                assert!(
                    read.is_finite() && read > 0.0 && read <= MAX_NOISE,
                    "{id}: read at ISO {iso} = {read}"
                );
            }
        }
    }

    #[test]
    fn variance_model_is_positive_and_matches_sigma() {
        for profile in CAMERA_PROFILES {
            let id = profile.model_aliases[0];
            for iso in [50u32, 800, 102400] {
                let v0 = noise_variance(profile, iso, 0.0);
                let v1 = noise_variance(profile, iso, 1.0);
                assert!(v0.is_finite() && v0 > 0.0, "{id}: black variance at ISO {iso}");
                assert!(v1.is_finite() && v1 > v0, "{id}: variance must grow with signal");
                let sigma = noise_sigma(profile, iso, 0.5);
                let expected = noise_variance(profile, iso, 0.5).sqrt();
                assert_close(sigma, expected, &format!("{id} sigma@ISO{iso}"));
            }
        }
    }

    #[test]
    fn xtrans_flags_are_correct() {
        let xtrans: Vec<&str> = CAMERA_PROFILES
            .iter()
            .filter(|p| p.is_xtrans)
            .map(|p| p.model_aliases[0])
            .collect();
        assert_eq!(xtrans, ["X-T5", "X-S20"], "X-H2 is a Bayer body");
    }

    #[test]
    fn banding_table_is_an_uncalibrated_placeholder() {
        for model in CANONICAL {
            let rule = resolve_banding(model, 6400).expect("known body has a banding rule");
            assert_eq!(rule.amplitude, 0.0, "{model}: banding must stay uncalibrated for now");
            assert_eq!(rule.period_px, 0.0, "{model}: no measured striping period");
        }
        assert!(resolve_banding("NIKON Z 6", 6400).is_none());
        assert!(lookup_banding("SONY ILCE-9").is_none());
    }
}
