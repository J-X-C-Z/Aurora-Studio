//! Shared image-quality metrics for Aurora Studio's quality benches.
//!
//! This module is compiled into both `tests/quality_bench.rs` (synthetic
//! self-tests, always runnable) and `examples/quality_bench.rs` (external
//! fixture CLI). It therefore must not depend on test-only helpers and must
//! compile with the default feature set.
//!
//! Conventions:
//! * colours are sRGB 8-bit unless a function says otherwise;
//! * ROIs are `[x, y, width, height]` in pixels, with the origin at the
//!   top-left of the image;
//! * MTF frequencies are cycles per pixel (c/px);
//! * differences between CPU and GPU images are reported on the 8-bit level
//!   scale (0 ..= 255), whichever bit depth the files use.
//!
//! The module is deliberately dependency-light: `image` for decoding, `serde`
//! for the fixture manifest and the JSON report.

#![allow(dead_code)]

use std::path::Path;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// sRGB transfer, XYZ and Lab (D65)
// ---------------------------------------------------------------------------

/// sRGB electro-optical transfer function (EOTF), input in `0.0 ..= 1.0`.
pub fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Inverse sRGB transfer function, input in linear `0.0 ..= 1.0`.
pub fn linear_to_srgb(c: f64) -> f64 {
    if c <= 0.003_130_8 {
        12.92 * c
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

pub fn srgb8_to_linear(c: u8) -> f64 {
    srgb_to_linear(c as f64 / 255.0)
}

/// D65 reference white used by sRGB and by the CIELAB conversion below.
pub const D65_WHITE: [f64; 3] = [0.950_47, 1.0, 1.088_83];

/// Linear sRGB -> CIE XYZ (D65, IEC 61966-2-1 matrix).
pub fn linear_rgb_to_xyz(rgb: [f64; 3]) -> [f64; 3] {
    let [r, g, b] = rgb;
    [
        0.412_456_4 * r + 0.357_576_1 * g + 0.180_437_5 * b,
        0.212_672_9 * r + 0.715_152_2 * g + 0.072_175_0 * b,
        0.019_333_9 * r + 0.119_192_0 * g + 0.950_304_1 * b,
    ]
}

fn lab_f(t: f64) -> f64 {
    // (6/29)^3 = 0.008856...; below it the curve is linear.
    const DELTA: f64 = 6.0 / 29.0;
    if t > DELTA * DELTA * DELTA {
        t.cbrt()
    } else {
        t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
    }
}

/// CIE XYZ (D65) -> CIELAB.
pub fn xyz_to_lab(xyz: [f64; 3]) -> [f64; 3] {
    let fx = lab_f(xyz[0] / D65_WHITE[0]);
    let fy = lab_f(xyz[1] / D65_WHITE[1]);
    let fz = lab_f(xyz[2] / D65_WHITE[2]);
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// sRGB 8-bit -> CIELAB (D65).
pub fn srgb8_to_lab(rgb: [u8; 3]) -> [f64; 3] {
    let linear = [
        srgb8_to_linear(rgb[0]),
        srgb8_to_linear(rgb[1]),
        srgb8_to_linear(rgb[2]),
    ];
    xyz_to_lab(linear_rgb_to_xyz(linear))
}

fn hue_deg(b: f64, ap: f64) -> f64 {
    if ap == 0.0 && b == 0.0 {
        0.0
    } else {
        b.atan2(ap).to_degrees().rem_euclid(360.0)
    }
}

/// CIEDE2000 colour difference with `kL = kC = kH = 1`
/// (Sharma, Wu & Dalal, 2005).
pub fn ciede2000(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    // 25^7
    const POW25_7: f64 = 6_103_515_625.0;

    let [l1, a1, b1] = lab1;
    let [l2, a2, b2] = lab2;

    let c1 = (a1 * a1 + b1 * b1).sqrt();
    let c2 = (a2 * a2 + b2 * b2).sqrt();
    let c_bar = 0.5 * (c1 + c2);
    let c_bar7 = c_bar.powi(7);
    let g = 0.5 * (1.0 - (c_bar7 / (c_bar7 + POW25_7)).sqrt());

    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;
    let c1p = (a1p * a1p + b1 * b1).sqrt();
    let c2p = (a2p * a2p + b2 * b2).sqrt();
    let h1p = hue_deg(b1, a1p);
    let h2p = hue_deg(b2, a2p);

    let dlp = l2 - l1;
    let dcp = c2p - c1p;

    let dhp = if c1p * c2p == 0.0 {
        0.0
    } else {
        let diff = h2p - h1p;
        if diff.abs() <= 180.0 {
            diff
        } else if diff > 180.0 {
            diff - 360.0
        } else {
            diff + 360.0
        }
    };
    let dh_big = 2.0 * (c1p * c2p).sqrt() * (dhp.to_radians() * 0.5).sin();

    let l_bar_p = 0.5 * (l1 + l2);
    let c_bar_p = 0.5 * (c1p + c2p);
    let h_bar_p = if c1p * c2p == 0.0 {
        h1p + h2p
    } else {
        let sum = h1p + h2p;
        if (h1p - h2p).abs() <= 180.0 {
            0.5 * sum
        } else if sum < 360.0 {
            0.5 * (sum + 360.0)
        } else {
            0.5 * (sum - 360.0)
        }
    };

    let t = 1.0 - 0.17 * (h_bar_p - 30.0).to_radians().cos()
        + 0.24 * (2.0 * h_bar_p).to_radians().cos()
        + 0.32 * (3.0 * h_bar_p + 6.0).to_radians().cos()
        - 0.20 * (4.0 * h_bar_p - 63.0).to_radians().cos();

    let d_theta = 30.0 * (-(((h_bar_p - 275.0) / 25.0).powi(2))).exp();
    let rc = 2.0 * (c_bar_p.powi(7) / (c_bar_p.powi(7) + POW25_7)).sqrt();
    let l50 = (l_bar_p - 50.0).powi(2);
    let sl = 1.0 + (0.015 * l50) / (20.0 + l50).sqrt();
    let sc = 1.0 + 0.045 * c_bar_p;
    let sh = 1.0 + 0.015 * c_bar_p * t;
    let rt = -(2.0 * d_theta.to_radians()).sin() * rc;

    let tl = dlp / sl;
    let tc = dcp / sc;
    let th = dh_big / sh;
    (tl * tl + tc * tc + th * th + rt * tc * th).sqrt()
}

// ---------------------------------------------------------------------------
// Image containers and ROI
// ---------------------------------------------------------------------------

/// Straight-alpha RGBA8 pixels, row-major.
#[derive(Clone, Debug)]
pub struct Rgba8Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl Rgba8Image {
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Self {
        debug_assert_eq!(pixels.len(), width as usize * height as usize * 4);
        Self {
            width,
            height,
            pixels,
        }
    }

    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let i = (y as usize * self.width as usize + x as usize) * 4;
        [
            self.pixels[i],
            self.pixels[i + 1],
            self.pixels[i + 2],
            self.pixels[i + 3],
        ]
    }

    #[inline]
    pub fn rgb(&self, x: u32, y: u32) -> [u8; 3] {
        let p = self.pixel(x, y);
        [p[0], p[1], p[2]]
    }

    /// Rec.709 / sRGB luma of the encoded (non-linear) values.
    #[inline]
    pub fn luma(&self, x: u32, y: u32) -> f64 {
        let [r, g, b] = self.rgb(x, y);
        0.2126 * r as f64 + 0.7152 * g as f64 + 0.0722 * b as f64
    }
}

/// Straight-alpha RGBA16 pixels, row-major.
#[derive(Clone, Debug)]
pub struct Rgba16Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u16>,
}

impl Rgba16Image {
    pub fn new(width: u32, height: u32, pixels: Vec<u16>) -> Self {
        debug_assert_eq!(pixels.len(), width as usize * height as usize * 4);
        Self {
            width,
            height,
            pixels,
        }
    }
}

/// Pixel rectangle `[x, y, width, height]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Roi {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

impl From<[u32; 4]> for Roi {
    fn from([x, y, w, h]: [u32; 4]) -> Self {
        Self { x, y, w, h }
    }
}

impl From<Roi> for [u32; 4] {
    fn from(r: Roi) -> Self {
        [r.x, r.y, r.w, r.h]
    }
}

impl Roi {
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    /// Validates the ROI against an image size.
    pub fn checked(self, width: u32, height: u32) -> Result<Self, String> {
        if self.w == 0 || self.h == 0 {
            return Err("ROI 宽高必须大于 0".to_string());
        }
        if self.x as u64 + self.w as u64 > width as u64
            || self.y as u64 + self.h as u64 > height as u64
        {
            return Err(format!(
                "ROI [{}, {}, {}, {}] 超出图像 {}x{}",
                self.x, self.y, self.w, self.h, width, height
            ));
        }
        Ok(self)
    }
}

// ---------------------------------------------------------------------------
// Image loading
// ---------------------------------------------------------------------------

/// An image loaded from disk, preserving 16-bit precision when present.
#[derive(Clone, Debug)]
pub enum LoadedImage {
    Rgba8(Rgba8Image),
    Rgba16(Rgba16Image),
}

impl LoadedImage {
    pub fn load(path: &Path) -> Result<Self, String> {
        let dynamic = image::open(path)
            .map_err(|err| format!("无法读取图像 {}: {err}", path.display()))?;
        if matches!(
            dynamic.color(),
            image::ColorType::L16
                | image::ColorType::La16
                | image::ColorType::Rgb16
                | image::ColorType::Rgba16
        ) {
            let buffer = dynamic.to_rgba16();
            let (w, h) = (buffer.width(), buffer.height());
            Ok(Self::Rgba16(Rgba16Image::new(w, h, buffer.into_raw())))
        } else {
            let buffer = dynamic.to_rgba8();
            let (w, h) = (buffer.width(), buffer.height());
            Ok(Self::Rgba8(Rgba8Image::new(w, h, buffer.into_raw())))
        }
    }

    pub fn width(&self) -> u32 {
        match self {
            Self::Rgba8(i) => i.width,
            Self::Rgba16(i) => i.width,
        }
    }

    pub fn height(&self) -> u32 {
        match self {
            Self::Rgba8(i) => i.height,
            Self::Rgba16(i) => i.height,
        }
    }

    /// 8-bit view for the colour metrics (16-bit input is quantised).
    pub fn to_rgba8(&self) -> Rgba8Image {
        match self {
            Self::Rgba8(i) => i.clone(),
            Self::Rgba16(i) => {
                let pixels = i.pixels.iter().map(|&v| (v >> 8) as u8).collect();
                Rgba8Image::new(i.width, i.height, pixels)
            }
        }
    }

    /// Compares two images. Mixed bit depths are compared after up-converting
    /// the 8-bit side to 16-bit (`v * 257`), so the result keeps the intent.
    pub fn diff(&self, other: &Self) -> Result<DiffStats, String> {
        match (self, other) {
            (Self::Rgba8(a), Self::Rgba8(b)) => diff_rgba8(a, b),
            (Self::Rgba16(a), Self::Rgba16(b)) => diff_rgba16(a, b),
            (Self::Rgba8(a), Self::Rgba16(b)) => diff_rgba16(&upconvert(&a.pixels, a.width, a.height), b),
            (Self::Rgba16(a), Self::Rgba8(b)) => diff_rgba16(a, &upconvert(&b.pixels, b.width, b.height)),
        }
    }
}

fn upconvert(pixels: &[u8], width: u32, height: u32) -> Rgba16Image {
    let pixels = pixels.iter().map(|&v| v as u16 * 257).collect();
    Rgba16Image::new(width, height, pixels)
}

// ---------------------------------------------------------------------------
// ColorChecker
// ---------------------------------------------------------------------------

pub const DEFAULT_PATCH_RADIUS: u32 = 4;

/// ColorChecker section of a fixture manifest.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ColorCheckerConfig {
    /// Patch centres `[x, y]` in the sample image.
    pub patches: Vec<[f64; 2]>,
    /// Optional expected CIELAB values, one per patch.
    #[serde(default)]
    pub expected: Option<Vec<[f64; 3]>>,
    /// Half-size of the median sampling square, in pixels (default 4).
    #[serde(default)]
    pub patch_radius: Option<u32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PatchDelta {
    pub index: usize,
    pub center: [u32; 2],
    pub median_rgb: [u8; 3],
    pub measured_lab: [f64; 3],
    pub expected_lab: Option<[f64; 3]>,
    pub delta_e: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ColorCheckerReport {
    pub patches: Vec<PatchDelta>,
    pub mean_delta_e: Option<f64>,
    pub max_delta_e: Option<f64>,
}

/// Median RGB of a square of pixels centred on `(cx, cy)`.
pub fn median_patch_rgb(
    img: &Rgba8Image,
    cx: f64,
    cy: f64,
    radius: u32,
) -> Result<([u32; 2], [u8; 3]), String> {
    if img.width == 0 || img.height == 0 {
        return Err("空图像".to_string());
    }
    let center_x = cx.round() as i64;
    let center_y = cy.round() as i64;
    if center_x < 0 || center_y < 0 || center_x >= img.width as i64 || center_y >= img.height as i64 {
        return Err(format!("patch 中心 ({cx}, {cy}) 超出图像"));
    }
    let x0 = (center_x - radius as i64).clamp(0, img.width as i64 - 1) as u32;
    let x1 = (center_x + radius as i64).clamp(0, img.width as i64 - 1) as u32;
    let y0 = (center_y - radius as i64).clamp(0, img.height as i64 - 1) as u32;
    let y1 = (center_y + radius as i64).clamp(0, img.height as i64 - 1) as u32;

    let mut channels: [Vec<u8>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    for y in y0..=y1 {
        for x in x0..=x1 {
            let rgb = img.rgb(x, y);
            for (dst, v) in channels.iter_mut().zip(rgb) {
                dst.push(v);
            }
        }
    }
    if channels[0].is_empty() {
        return Err("patch 采样区域为空".to_string());
    }
    let median = |values: &mut Vec<u8>| -> u8 {
        values.sort_unstable();
        let n = values.len();
        if n % 2 == 1 {
            values[n / 2]
        } else {
            ((values[n / 2 - 1] as u32 + values[n / 2] as u32 + 1) / 2) as u8
        }
    };
    let rgb = [
        median(&mut channels[0]),
        median(&mut channels[1]),
        median(&mut channels[2]),
    ];
    Ok(([center_x as u32, center_y as u32], rgb))
}

/// Measures every patch and, when expected Lab values are supplied, the
/// mean and maximum CIEDE2000 over the patches.
pub fn colorchecker(
    img: &Rgba8Image,
    cfg: &ColorCheckerConfig,
) -> Result<ColorCheckerReport, String> {
    if cfg.patches.is_empty() {
        return Err("colorchecker.patches 为空".to_string());
    }
    if let Some(expected) = &cfg.expected {
        if expected.len() != cfg.patches.len() {
            return Err(format!(
                "colorchecker.expected 有 {} 项，但 patches 有 {} 项",
                expected.len(),
                cfg.patches.len()
            ));
        }
    }
    let radius = cfg.patch_radius.unwrap_or(DEFAULT_PATCH_RADIUS);
    let mut patches = Vec::with_capacity(cfg.patches.len());
    let mut sum = 0.0;
    let mut max: Option<f64> = None;
    let mut measured = 0usize;

    for (index, center) in cfg.patches.iter().enumerate() {
        let ([cx, cy], rgb) = median_patch_rgb(img, center[0], center[1], radius)?;
        let lab = srgb8_to_lab(rgb);
        let expected = cfg.expected.as_ref().map(|v| v[index]);
        let delta_e = expected.map(|e| ciede2000(lab, e));
        if let Some(de) = delta_e {
            sum += de;
            measured += 1;
            max = Some(max.map_or(de, |m| m.max(de)));
        }
        patches.push(PatchDelta {
            index,
            center: [cx, cy],
            median_rgb: rgb,
            measured_lab: lab,
            expected_lab: expected,
            delta_e,
        });
    }

    Ok(ColorCheckerReport {
        patches,
        mean_delta_e: (measured > 0).then(|| sum / measured as f64),
        max_delta_e: max,
    })
}

// ---------------------------------------------------------------------------
// Shadow SNR
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
pub struct SnrReport {
    /// Mean linear Rec.709 luminance inside the ROI.
    pub mean: f64,
    /// Population standard deviation of the linear luminance.
    pub std: f64,
    /// `20 * log10(mean / std)`. `None` when the ROI is perfectly flat.
    pub snr_db: Option<f64>,
    pub pixels: usize,
}

/// Linear Rec.709 luminance of one pixel.
#[inline]
pub fn rec709_linear_luminance(img: &Rgba8Image, x: u32, y: u32) -> f64 {
    let [r, g, b] = img.rgb(x, y);
    0.2126 * srgb8_to_linear(r) + 0.7152 * srgb8_to_linear(g) + 0.0722 * srgb8_to_linear(b)
}

/// Shadow-region SNR on linear luminance.
pub fn shadow_snr(img: &Rgba8Image, roi: Roi) -> Result<SnrReport, String> {
    let roi = roi.checked(img.width, img.height)?;
    let n = roi.w as usize * roi.h as usize;
    if n < 2 {
        return Err("SNR ROI 至少需要 2 个像素".to_string());
    }
    let mut sum = 0.0;
    for y in roi.y..roi.y + roi.h {
        for x in roi.x..roi.x + roi.w {
            sum += rec709_linear_luminance(img, x, y);
        }
    }
    let mean = sum / n as f64;
    // Two-pass variance: `E[x^2] - mean^2` cancels catastrophically on a flat
    // ROI and would turn zero noise into a finite SNR.
    let mut sum_sq = 0.0;
    for y in roi.y..roi.y + roi.h {
        for x in roi.x..roi.x + roi.w {
            let d = rec709_linear_luminance(img, x, y) - mean;
            sum_sq += d * d;
        }
    }
    let std = (sum_sq / n as f64).max(0.0).sqrt();
    let snr_db = if std > 1e-12 {
        Some(20.0 * (mean / std).log10())
    } else {
        None
    };
    Ok(SnrReport {
        mean,
        std,
        snr_db,
        pixels: n,
    })
}

// ---------------------------------------------------------------------------
// MTF50 (ISO 12233-style slanted edge)
// ---------------------------------------------------------------------------

/// Projection / ESF bin width in pixels.
pub const MTF_BIN_PX: f64 = 0.25;

#[derive(Clone, Debug, Serialize)]
pub struct MtfReport {
    /// Spatial frequency where MTF falls to 0.5, in cycles per pixel.
    pub mtf50_cycles_per_px: f64,
    /// Estimated edge-normal direction, degrees in `0 .. 180`.
    pub edge_normal_deg: f64,
    /// Number of LSF samples used by the DFT.
    pub lsf_len: usize,
    /// Projection bin width in pixels.
    pub esf_step_px: f64,
}

/// Separable `[1 4 6 4 1] / 16` blur; edges of the buffer are left at zero and
/// callers only read the interior.
fn smooth5(src: &[f64], width: usize, height: usize) -> Vec<f64> {
    let mut tmp = vec![0.0f64; src.len()];
    for y in 0..height {
        for x in 2..width.saturating_sub(2) {
            let i = y * width + x;
            tmp[i] = (src[i - 2] + 4.0 * src[i - 1] + 6.0 * src[i] + 4.0 * src[i + 1]
                + src[i + 2])
                / 16.0;
        }
    }
    let mut out = vec![0.0f64; src.len()];
    for y in 2..height.saturating_sub(2) {
        for x in 0..width {
            let i = y * width + x;
            out[i] = (tmp[i - 2 * width]
                + 4.0 * tmp[i - width]
                + 6.0 * tmp[i]
                + 4.0 * tmp[i + width]
                + tmp[i + 2 * width])
                / 16.0;
        }
    }
    out
}

/// ISO 12233-style MTF50 from a slanted edge.
///
/// The edge normal is estimated with a structure tensor over the ROI. Pixels
/// are projected onto that normal axis and binned at [`MTF_BIN_PX`]; the ESF is
/// differentiated into an LSF, Hamming-windowed, and transformed with a naive
/// DFT. The first descending 0.5 crossing is found by linear interpolation.
pub fn mtf50(img: &Rgba8Image, roi: Roi) -> Result<MtfReport, String> {
    let roi = roi.checked(img.width, img.height)?;
    // The orientation pass needs 4 px of margin on every side (5-point
    // gradient after a 5-tap pre-smoothing pass).
    if roi.w < 12 || roi.h < 12 {
        return Err(format!("MTF ROI 至少需要 12x12，当前 {}x{}", roi.w, roi.h));
    }
    let (rw, rh) = (roi.w as usize, roi.h as usize);

    // Encoded luma. A synthetic edge is generated and measured in the same
    // domain, so no linearisation is applied here (linearising would distort
    // the Gaussian profile of the synthetic test).
    let mut gray = vec![0.0f64; rw * rh];
    for y in 0..rh {
        for x in 0..rw {
            gray[y * rw + x] = img.luma(roi.x + x as u32, roi.y + y as u32);
        }
    }

    // The orientation estimate is sensitive to the small g_y component of a
    // near-vertical edge, so suppress 8-bit quantisation noise first with an
    // isotropic separable [1 4 6 4 1]/16 blur. Being isotropic it does not
    // rotate the edge.
    let smooth = smooth5(&gray, rw, rh);

    // Structure tensor of 5-point gradients. A plain 2-point central
    // difference has an O(h^2) truncation error proportional to f''' cos^3,
    // which biases the estimated normal of a sharp edge; the 5-point stencil
    // is O(h^4) and its leading error is odd across the edge, so it cancels.
    let mut jxx = 0.0;
    let mut jyy = 0.0;
    let mut jxy = 0.0;
    for y in 4..rh - 4 {
        for x in 4..rw - 4 {
            let gx = (-smooth[y * rw + x + 2] + 8.0 * smooth[y * rw + x + 1]
                - 8.0 * smooth[y * rw + x - 1]
                + smooth[y * rw + x - 2])
                / 12.0;
            let gy = (-smooth[(y + 2) * rw + x] + 8.0 * smooth[(y + 1) * rw + x]
                - 8.0 * smooth[(y - 1) * rw + x]
                + smooth[(y - 2) * rw + x])
                / 12.0;
            jxx += gx * gx;
            jyy += gy * gy;
            jxy += gx * gy;
        }
    }
    if jxx + jyy < 1e-9 {
        return Err("ROI 内没有可用的边缘梯度".to_string());
    }
    let theta = 0.5 * (2.0 * jxy).atan2(jxx - jyy);
    let (sin, cos) = theta.sin_cos();
    let cx = (rw as f64 - 1.0) * 0.5;
    let cy = (rh as f64 - 1.0) * 0.5;

    // Projection range over the interior pixels.
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;
    for y in 2..rh - 2 {
        for x in 2..rw - 2 {
            let t = (x as f64 - cx) * cos + (y as f64 - cy) * sin;
            t_min = t_min.min(t);
            t_max = t_max.max(t);
        }
    }
    if !t_min.is_finite() || !t_max.is_finite() || t_max - t_min < 2.0 {
        return Err("边缘投影范围过小".to_string());
    }

    let n_bins = (((t_max - t_min) / MTF_BIN_PX).ceil() as usize) + 1;
    let mut sums = vec![0.0f64; n_bins];
    let mut counts = vec![0u32; n_bins];
    for y in 2..rh - 2 {
        for x in 2..rw - 2 {
            let t = (x as f64 - cx) * cos + (y as f64 - cy) * sin;
            let bin = (((t - t_min) / MTF_BIN_PX).floor() as usize).min(n_bins - 1);
            sums[bin] += gray[y * rw + x];
            counts[bin] += 1;
        }
    }

    let first = counts.iter().position(|&c| c > 0).ok_or("ESF 投影为空")?;
    let last = counts
        .iter()
        .rposition(|&c| c > 0)
        .ok_or("ESF 投影为空")?;
    if last - first + 1 < 16 {
        return Err("ESF 有效区间过短".to_string());
    }

    // Mean intensity per bin, filling interior gaps by linear interpolation.
    let means: Vec<Option<f64>> = (0..n_bins)
        .map(|i| (counts[i] > 0).then(|| sums[i] / counts[i] as f64))
        .collect();
    let mut esf = Vec::with_capacity(last - first + 1);
    let mut prev: Option<(usize, f64)> = None;
    for i in first..=last {
        if let Some(v) = means[i] {
            prev = Some((i, v));
            esf.push(v);
        } else {
            let (pi, pv) = prev.ok_or("ESF 起始 bin 无数据")?;
            let mut j = i + 1;
            while j <= last && means[j].is_none() {
                j += 1;
            }
            if j > last {
                return Err("ESF 末尾 bin 无数据".to_string());
            }
            if j - pi > 16 {
                return Err("ESF 中间空洞过大，无法可靠分箱".to_string());
            }
            let nv = means[j].unwrap();
            let frac = (i - pi) as f64 / (j - pi) as f64;
            esf.push(pv + (nv - pv) * frac);
        }
    }

    // Drop one pixel at each end so the finite difference never sees the ROI
    // boundary.
    let trim = 4usize.min(esf.len() / 8);
    if esf.len() <= 2 * trim + 8 {
        return Err("ESF 有效长度不足".to_string());
    }
    let esf = &esf[trim..esf.len() - trim];

    // Differentiate ESF -> LSF.
    let mut lsf: Vec<f64> = esf
        .windows(3)
        .map(|w| (w[2] - w[0]) / (2.0 * MTF_BIN_PX))
        .collect();

    // The gradient sign (and thus the LSF sign) depends on which side is
    // brighter; MTF only cares about magnitude.
    let peak = lsf
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
        .map(|(i, _)| i)
        .ok_or("LSF 为空")?;
    if lsf[peak] < 0.0 {
        for v in &mut lsf {
            *v = -*v;
        }
    }
    let half = peak.min(lsf.len() - 1 - peak).min(1024);
    if half < 8 {
        return Err("LSF 峰值过于靠边，ROI 需要包含完整边缘".to_string());
    }

    // Hamming window centred on the LSF peak.
    let len = 2 * half + 1;
    let mut windowed = vec![0.0f64; len];
    for (i, out) in windowed.iter_mut().enumerate() {
        let w = 0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (len - 1) as f64).cos();
        *out = lsf[peak - half + i] * w;
    }

    let dc: f64 = windowed.iter().sum();
    if dc.abs() < 1e-12 {
        return Err("LSF 直流分量为 0".to_string());
    }

    // Naive DFT, one-sided magnitude, normalised to MTF(0) = 1.
    let n = len as f64;
    let mut mtf = Vec::with_capacity(len / 2 + 1);
    for k in 0..=len / 2 {
        let mut re = 0.0;
        let mut im = 0.0;
        for (i, v) in windowed.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * (k * i) as f64 / n;
            re += v * angle.cos();
            im -= v * angle.sin();
        }
        mtf.push((re * re + im * im).sqrt() / dc.abs());
    }

    for k in 1..mtf.len() {
        if mtf[k] < 0.5 {
            let prev = mtf[k - 1];
            let cur = mtf[k];
            let frac = if (prev - cur).abs() > 1e-12 {
                (prev - 0.5) / (prev - cur)
            } else {
                0.0
            };
            let bin = (k - 1) as f64 + frac;
            let freq = bin / (len as f64 * MTF_BIN_PX);
            return Ok(MtfReport {
                mtf50_cycles_per_px: freq,
                edge_normal_deg: theta.to_degrees().rem_euclid(180.0),
                lsf_len: len,
                esf_step_px: MTF_BIN_PX,
            });
        }
    }
    Err("MTF 在 Nyquist 之前没有降到 0.5".to_string())
}

// ---------------------------------------------------------------------------
// False colour
// ---------------------------------------------------------------------------

pub const DEFAULT_FALSE_COLOR_CHROMA: f64 = 8.0;

#[derive(Clone, Debug, Serialize)]
pub struct FalseColorReport {
    /// Share of ROI pixels whose Lab chroma exceeds `threshold`.
    pub ratio: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
    pub threshold: f64,
    pub pixels: usize,
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let pos = (p / 100.0).clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as f64;
        sorted[lo] + (sorted[hi] - sorted[lo]) * frac
    }
}

/// Lab-chroma false-colour share inside an ROI.
pub fn false_color(
    img: &Rgba8Image,
    roi: Roi,
    threshold: f64,
) -> Result<FalseColorReport, String> {
    let roi = roi.checked(img.width, img.height)?;
    let n = roi.w as usize * roi.h as usize;
    let mut chroma = Vec::with_capacity(n);
    let mut over = 0usize;
    let mut max = 0.0f64;
    for y in roi.y..roi.y + roi.h {
        for x in roi.x..roi.x + roi.w {
            let lab = srgb8_to_lab(img.rgb(x, y));
            let c = (lab[1] * lab[1] + lab[2] * lab[2]).sqrt();
            if c > threshold {
                over += 1;
            }
            max = max.max(c);
            chroma.push(c);
        }
    }
    chroma.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(FalseColorReport {
        ratio: over as f64 / n as f64,
        p95: percentile(&chroma, 95.0),
        p99: percentile(&chroma, 99.0),
        max,
        threshold,
        pixels: n,
    })
}

// ---------------------------------------------------------------------------
// CPU vs OpenCL difference
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Serialize)]
pub struct DiffStats {
    /// Maximum absolute difference on the 8-bit level scale (0 ..= 255).
    pub max: f64,
    pub mean: f64,
    /// P99 from a 0.25-level histogram.
    pub p99: f64,
    /// Share of channel samples with a difference strictly above 1/255.
    pub over_1_255: f64,
    /// Share of channel samples with a difference strictly above 2/255.
    pub over_2_255: f64,
    /// Number of channel samples compared (pixels * 4).
    pub samples: usize,
}

fn summarize_diff(hist: &[u64], sum: f64, max: f64, over_1: u64, over_2: u64, samples: usize) -> DiffStats {
    let total: u64 = hist.iter().sum();
    let p99 = if total == 0 {
        0.0
    } else {
        let target = 0.99 * total as f64;
        let mut cum = 0u64;
        let mut value = 0.0;
        for (i, &c) in hist.iter().enumerate() {
            cum += c;
            if cum as f64 >= target {
                value = i as f64 / 4.0;
                break;
            }
        }
        value
    };
    let denom = samples.max(1) as f64;
    DiffStats {
        max,
        mean: if samples > 0 { sum / denom } else { 0.0 },
        p99,
        over_1_255: over_1 as f64 / denom,
        over_2_255: over_2 as f64 / denom,
        samples,
    }
}

fn push_diff(hist: &mut [u64], sum: &mut f64, max: &mut f64, over_1: &mut u64, over_2: &mut u64, d: f64) {
    *sum += d;
    if d > *max {
        *max = d;
    }
    if d > 1.0 {
        *over_1 += 1;
    }
    if d > 2.0 {
        *over_2 += 1;
    }
    let bin = ((d * 4.0).round() as usize).min(hist.len() - 1);
    hist[bin] += 1;
}

pub fn diff_rgba8(a: &Rgba8Image, b: &Rgba8Image) -> Result<DiffStats, String> {
    if a.width != b.width || a.height != b.height {
        return Err(format!(
            "图像尺寸不同: {}x{} vs {}x{}",
            a.width, a.height, b.width, b.height
        ));
    }
    let samples = a.pixels.len();
    let mut hist = vec![0u64; 1021];
    let mut sum = 0.0;
    let mut max = 0.0;
    let mut over_1 = 0u64;
    let mut over_2 = 0u64;
    for (&x, &y) in a.pixels.iter().zip(&b.pixels) {
        let d = x.abs_diff(y) as f64;
        push_diff(&mut hist, &mut sum, &mut max, &mut over_1, &mut over_2, d);
    }
    Ok(summarize_diff(&hist, sum, max, over_1, over_2, samples))
}

pub fn diff_rgba16(a: &Rgba16Image, b: &Rgba16Image) -> Result<DiffStats, String> {
    if a.width != b.width || a.height != b.height {
        return Err(format!(
            "图像尺寸不同: {}x{} vs {}x{}",
            a.width, a.height, b.width, b.height
        ));
    }
    let samples = a.pixels.len();
    let mut hist = vec![0u64; 1021];
    let mut sum = 0.0;
    let mut max = 0.0;
    let mut over_1 = 0u64;
    let mut over_2 = 0u64;
    let scale = 255.0 / 65535.0;
    for (&x, &y) in a.pixels.iter().zip(&b.pixels) {
        let d = x.abs_diff(y) as f64 * scale;
        push_diff(&mut hist, &mut sum, &mut max, &mut over_1, &mut over_2, d);
    }
    Ok(summarize_diff(&hist, sum, max, over_1, over_2, samples))
}

// ---------------------------------------------------------------------------
// External fixture manifest
// ---------------------------------------------------------------------------

pub const DEFAULT_COLORCHECKER_THRESHOLD: f64 = 2.0;

/// Top-level fixture manifest. Samples stay outside the repository; this file
/// only references them by absolute path.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Manifest {
    #[serde(default)]
    pub colorchecker_threshold: Option<f64>,
    #[serde(default)]
    pub false_color_chroma: Option<f64>,
    #[serde(default)]
    pub samples: Vec<Sample>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Sample {
    pub name: String,
    pub image: String,
    #[serde(default)]
    pub reference: Option<String>,
    #[serde(default)]
    pub candidate: Option<String>,
    #[serde(default)]
    pub colorchecker: Option<ColorCheckerConfig>,
    /// `[x, y, width, height]` in pixels.
    #[serde(default)]
    pub shadow_roi: Option<[u32; 4]>,
    /// `[x, y, width, height]` in pixels.
    #[serde(default)]
    pub edge_roi: Option<[u32; 4]>,
    /// `[x, y, width, height]` in pixels.
    #[serde(default)]
    pub neutral_roi: Option<[u32; 4]>,
    /// Absolute path to a real RAW file for the cold-decode consistency test.
    #[serde(default)]
    pub raw: Option<String>,
    /// Long side passed to `raw_cache::get_or_decode`.
    #[serde(default)]
    pub raw_long_side: Option<u32>,
}

pub fn load_manifest(path: &Path) -> Result<Manifest, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|err| format!("无法读取 manifest {}: {err}", path.display()))?;
    serde_json::from_str(&text)
        .map_err(|err| format!("manifest JSON 解析失败 {}: {err}", path.display()))
}
