//! Byte-level parity harness for "reference path vs alternate path" renders.
//!
//! Roadmap R3 §15 acceptance infrastructure. Every acceleration patch that
//! claims "identical output" must point at a case here, and the case has to
//! say *where* and *how much* the two paths differ. A bare `bool`, a mean or a
//! perceptual score are not accepted as the decision: §15 forbids a large local
//! difference being averaged away. This file therefore reports, for both 8-bit
//! and 16-bit buffers:
//!
//! * maximum channel error (native levels, plus the 8-bit equivalent for u16),
//! * P99 of the channel-error distribution,
//! * the share of pixels with any channel difference,
//! * the first differing pixel in raster order (x, y, channel, delta),
//! * the worst pixel (x, y, channel, delta).
//!
//! Float buffers get the same shape: a caller-supplied tolerance decides pass
//! or fail, but the max error, P99 and the first/worst location are always
//! printed. Mean and P99 never decide a pass/fail on their own.
//!
//! Conventions
//! -----------
//! * `reference` is the CPU / scalar / unaccelerated side; `candidate` is the
//!   accelerated or alternate side. The stats do not care about the order, the
//!   names only keep the failure messages readable.
//! * Synthetic images use SplitMix64 with a fixed seed; nothing reads the clock,
//!   the network or a GPU. No external fixture is needed.
//! * The engine's `Source::from_rgba16` treats samples as **linear light**
//!   (`sample / 65535`), so the 16-bit generators below emit linear values.
//!   `Source::from_rgba` (sRGB decode) is crate-private, which is why the
//!   synthetic sources enter through the public 16-bit constructor.
//! * Every output-critical crossover in the develop/clarity/detail pipeline is
//!   gated on a pixel count (2,073,600 px for develop, see
//!   `develop_gpu::CROSSOVER_TABLE`); all cases here stay far below it, so the
//!   runs are CPU-only on a machine that has a CUDA device.
//!
//! Relation to the other tests: `tests/support/quality_metrics.rs` owns the
//! aggregate metrics (max/mean/P99, over-1/255 and over-2/255 channel shares)
//! and the external-fixture manifest; `tests/quality_bench.rs` owns the
//! calibration self-tests. Neither carries a first-error coordinate nor a
//! per-pixel error share, and both work on their own `Rgba8Image`/`Rgba16Image`
//! wrappers. This file does not re-implement CIEDE2000, MTF, Lab or the
//! manifest; it only adds the location-aware byte diff those tests lack.
//!
//! Baseline: engine HEAD `32e8aae` (decode_gate_units 6 -> 16). Every
//! `file:line` cited below refers to that commit.

use std::panic::{catch_unwind, set_hook, take_hook};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use aurora_engine::{
    render, render16, AdjustmentSet, FileEngine, ImageEngine, ImageId, MockEngine, NormRect,
    PreviewRequest, RenderQuality, Rendered, Rendered16, Source,
};

// ---------------------------------------------------------------------------
// Synthetic images (deterministic, fixed seed, no clock)
// ---------------------------------------------------------------------------

/// SplitMix64; the same generator `quality_bench.rs` uses, so the noise cases
/// are reproducible across files and machines.
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
}

#[derive(Clone)]
struct Rgba16 {
    width: u32,
    height: u32,
    pixels: Vec<u16>,
}

impl Rgba16 {
    fn of_rendered(rendered: &Rendered16) -> Self {
        Self {
            width: rendered.width,
            height: rendered.height,
            pixels: rendered.rgba.clone(),
        }
    }
}

/// Owned 8-bit RGBA frame, either a preview (`RgbaImage`) or a develop render
/// (`Rendered`); `assert_parity_u8` compares two of these.
#[derive(Clone)]
struct Rgba8 {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

impl Rgba8 {
    fn of_engine(image: &aurora_engine::RgbaImage) -> Self {
        Self {
            width: image.width,
            height: image.height,
            pixels: image.pixels.as_ref().clone(),
        }
    }

    fn of_preview(frame: &aurora_engine::PreviewFrame) -> Self {
        let mut img = Self::of_engine(&frame.image);
        // BGRA is a storage convention; compare the displayed channels in the same order.
        if frame.bgra {
            for pixel in img.pixels.chunks_exact_mut(4) {
                pixel.swap(0, 2);
            }
        }
        img
    }

    fn of_rendered(rendered: &Rendered) -> Self {
        Self {
            width: rendered.width,
            height: rendered.height,
            pixels: rendered.rgba.clone(),
        }
    }
}

/// Horizontal + vertical ramp in sRGB 8-bit codes; the 8-bit half of the diff
/// harness is exercised on it by the self-check below.
fn gradient8(width: u32, height: u32) -> Rgba8 {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    let dx = (width.max(2) - 1) as u32;
    let dy = (height.max(2) - 1) as u32;
    for y in 0..height {
        for x in 0..width {
            let r = (x * 255 / dx) as u8;
            let g = (y * 255 / dy) as u8;
            let b = 255 - r;
            pixels.extend_from_slice(&[r, g, b, 255]);
        }
    }
    Rgba8 { width, height, pixels }
}

/// Linear-light 16-bit solid colour.
fn solid16(width: u32, height: u32, rgb: [u16; 3]) -> Vec<u16> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..width * height {
        pixels.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 65535]);
    }
    pixels
}

/// Linear-light 16-bit ramp; odd widths are intentional.
fn gradient16(width: u32, height: u32) -> Vec<u16> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    let dx = (width.max(2) - 1) as u32;
    let dy = (height.max(2) - 1) as u32;
    for y in 0..height {
        for x in 0..width {
            let r = (x * 65535 / dx) as u16;
            let g = (y * 65535 / dy) as u16;
            let b = 65535 - r;
            pixels.extend_from_slice(&[r, g, b, 65535]);
        }
    }
    pixels
}

/// Linear-light 16-bit checkerboard.
fn checker16(width: u32, height: u32, cell: u32) -> Vec<u16> {
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    let cell = cell.max(1);
    for y in 0..height {
        for x in 0..width {
            let on = (x / cell + y / cell) % 2 == 0;
            let v: u16 = if on { 60000 } else { 4000 };
            pixels.extend_from_slice(&[v, v, v, 65535]);
        }
    }
    pixels
}

/// Linear-light 16-bit single-pixel spikes.
fn spike_noise16(width: u32, height: u32, seed: u64) -> Vec<u16> {
    let mut rng = Rng::new(seed);
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for _ in 0..width * height {
        let spike = rng.uniform() < 0.01;
        let v: u16 = if spike {
            if rng.uniform() < 0.5 { 0 } else { 65535 }
        } else {
            32000
        };
        pixels.extend_from_slice(&[v, v, v, 65535]);
    }
    pixels
}

/// Replaces the one-pixel border's alpha with 0. The engine's `Source` ingress
/// keeps linear RGB only (`from_rgba16` chunks over RGBA but reads p[0..3]),
/// so this documents that transparency is flattened, not carried.
fn with_transparent_border(pixels: &mut [u16], width: u32, height: u32) {
    for y in 0..height {
        for x in 0..width {
            if x == 0 || y == 0 || x + 1 == width || y + 1 == height {
                let i = ((y * width + x) * 4) as usize;
                pixels[i + 3] = 0;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Byte diff statistics
// ---------------------------------------------------------------------------

trait Level: Copy {
    fn level(self) -> u64;
}

impl Level for u8 {
    fn level(self) -> u64 {
        self as u64
    }
}

impl Level for u16 {
    fn level(self) -> u64 {
        self as u64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ErrorLoc {
    x: u32,
    y: u32,
    channel: u8,
    delta: u64,
}

#[derive(Debug)]
struct ParityStats {
    width: u32,
    height: u32,
    /// Native levels of the compared buffers: 255 for u8, 65535 for u16.
    levels: u64,
    /// Channel samples compared (`pixels * 4`).
    samples: usize,
    max_error: u64,
    mean_error: f64,
    p99_error: u64,
    /// Pixels with at least one differing channel, over all pixels.
    error_pixel_ratio: f64,
    first_error: Option<ErrorLoc>,
    worst_error: Option<ErrorLoc>,
}

impl ParityStats {
    fn eight_bit(&self, level: u64) -> f64 {
        level as f64 * 255.0 / self.levels as f64
    }

    fn report(&self) -> String {
        let loc = |error: &Option<ErrorLoc>| match error {
            Some(error) => format!(
                "({}, {}) channel {} delta {}",
                error.x, error.y, error.channel, error.delta
            ),
            None => "none".to_string(),
        };
        format!(
            "  image {}x{}, {} channel samples at {} levels\n  \
             max channel error: {} (8-bit equivalent {:.3})\n  \
             mean: {:.4}   p99: {} (8-bit equivalent {:.3})\n  \
             pixels differing: {:.4}% of the frame\n  \
             first error: {}\n  \
             worst pixel: {}\n  \
             (mean and p99 are context; the pass/fail gate is max + location)",
            self.width,
            self.height,
            self.samples,
            self.levels,
            self.max_error,
            self.eight_bit(self.max_error),
            self.mean_error,
            self.p99_error,
            self.eight_bit(self.p99_error),
            self.error_pixel_ratio * 100.0,
            loc(&self.first_error),
            loc(&self.worst_error),
        )
    }
}

fn diff_levels<T: Level>(
    reference: &[T],
    candidate: &[T],
    width: u32,
    height: u32,
    levels: u64,
) -> Result<ParityStats, String> {
    if width == 0 || height == 0 {
        return Err(format!("empty frame {width}x{height}"));
    }
    if reference.len() != candidate.len() {
        return Err(format!(
            "buffer lengths differ: {} vs {}",
            reference.len(),
            candidate.len()
        ));
    }
    let total_pixels = width as usize * height as usize;
    let samples = total_pixels * 4;
    if reference.len() != samples {
        return Err(format!(
            "buffer length {} does not match {width}x{height}x4 = {samples}",
            reference.len()
        ));
    }

    let mut hist = vec![0u64; levels as usize + 1];
    let mut sum = 0u64;
    let mut max = 0u64;
    let mut first_error: Option<ErrorLoc> = None;
    let mut worst_error: Option<ErrorLoc> = None;
    let mut differing_pixels = 0u64;
    let row = width as usize;

    for pixel in 0..total_pixels {
        let mut differs = false;
        for (channel, (a, b)) in reference[pixel * 4..pixel * 4 + 4]
            .iter()
            .zip(&candidate[pixel * 4..pixel * 4 + 4])
            .enumerate()
        {
            let delta = a.level().abs_diff(b.level());
            if delta == 0 {
                continue;
            }
            differs = true;
            sum += delta;
            hist[delta as usize] += 1;
            let loc = ErrorLoc {
                x: (pixel % row) as u32,
                y: (pixel / row) as u32,
                channel: channel as u8,
                delta,
            };
            if delta > max {
                max = delta;
                worst_error = Some(loc);
            }
            if first_error.is_none() {
                first_error = Some(loc);
            }
        }
        if differs {
            differing_pixels += 1;
        }
    }

    let target = 0.99 * samples as f64;
    let mut cumulative = 0u64;
    let mut p99 = 0u64;
    for (level, count) in hist.iter().enumerate() {
        cumulative += count;
        if cumulative as f64 >= target {
            p99 = level as u64;
            break;
        }
    }

    Ok(ParityStats {
        width,
        height,
        levels,
        samples,
        max_error: max,
        mean_error: sum as f64 / samples as f64,
        p99_error: p99,
        error_pixel_ratio: differing_pixels as f64 / total_pixels as f64,
        first_error,
        worst_error,
    })
}

fn diff8(reference: &Rgba8, candidate: &Rgba8) -> Result<ParityStats, String> {
    diff_levels(&reference.pixels, &candidate.pixels, reference.width, reference.height, 255)
}

fn diff16(reference: &Rgba16, candidate: &Rgba16) -> Result<ParityStats, String> {
    diff_levels(
        &reference.pixels,
        &candidate.pixels,
        reference.width,
        reference.height,
        65535,
    )
}

/// Fails with every statistic when any channel differs by more than
/// `tolerance` levels. `tolerance = 0` is the byte-exact contract.
fn assert_parity_u8(label: &str, reference: &Rgba8, candidate: &Rgba8, tolerance: u8) {
    match diff8(reference, candidate) {
        Ok(stats) if stats.max_error == 0 => {
            eprintln!("{label}: byte-identical ({} channel samples)", stats.samples);
        }
        Ok(stats) if stats.max_error <= u64::from(tolerance) => {
            eprintln!(
                "{label}: within tolerance {tolerance} level(s)\n{}",
                stats.report()
            );
        }
        Ok(stats) => panic!(
            "{label}: parity broken, tolerance {tolerance} level(s)\n{}",
            stats.report()
        ),
        Err(error) => panic!("{label}: {error}"),
    }
}

/// 16-bit flavour; the tolerance is in native 16-bit levels.
fn assert_parity_u16(label: &str, reference: &Rgba16, candidate: &Rgba16, tolerance: u16) {
    match diff16(reference, candidate) {
        Ok(stats) if stats.max_error == 0 => {
            eprintln!("{label}: byte-identical ({} channel samples)", stats.samples);
        }
        Ok(stats) if stats.max_error <= u64::from(tolerance) => {
            eprintln!(
                "{label}: within tolerance {tolerance} level(s)\n{}",
                stats.report()
            );
        }
        Ok(stats) => panic!(
            "{label}: parity broken, tolerance {tolerance} level(s)\n{}",
            stats.report()
        ),
        Err(error) => panic!("{label}: {error}"),
    }
}

// ---------------------------------------------------------------------------
// Float diff statistics (tolerance allowed, tail still reported)
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct RgbaF32 {
    width: u32,
    height: u32,
    pixels: Vec<f32>,
}

#[derive(Clone, Copy, Debug)]
struct FloatLoc {
    x: u32,
    y: u32,
    channel: u8,
    delta: f64,
}

#[derive(Debug)]
struct FloatParityStats {
    width: u32,
    height: u32,
    samples: usize,
    max_error: f64,
    mean_error: f64,
    p99_error: f64,
    error_pixel_ratio: f64,
    first_error: Option<FloatLoc>,
}

impl FloatParityStats {
    fn report(&self) -> String {
        let loc = match self.first_error {
            Some(error) => format!(
                "({}, {}) channel {} delta {:.8}",
                error.x, error.y, error.channel, error.delta
            ),
            None => "none".to_string(),
        };
        format!(
            "  image {}x{}, {} float channel samples\n  \
             max |error|: {:.8}\n  mean |error|: {:.8}   p99 |error|: {:.8}\n  \
             pixels differing: {:.4}% of the frame\n  \
             first error: {}\n  \
             (the long tail, not the mean, decides whether a tolerance is honest)",
            self.width,
            self.height,
            self.samples,
            self.max_error,
            self.mean_error,
            self.p99_error,
            self.error_pixel_ratio * 100.0,
            loc,
        )
    }
}

fn diff_f32(reference: &RgbaF32, candidate: &RgbaF32) -> Result<FloatParityStats, String> {
    if reference.width != candidate.width || reference.height != candidate.height {
        return Err(format!(
            "image sizes differ: {}x{} vs {}x{}",
            reference.width, reference.height, candidate.width, candidate.height
        ));
    }
    let total_pixels = reference.width as usize * reference.height as usize;
    let samples = total_pixels * 4;
    if reference.pixels.len() != samples || candidate.pixels.len() != samples {
        return Err(format!(
            "buffer lengths differ from {}x{}x4 = {samples}",
            reference.width, reference.height
        ));
    }

    let mut errors = Vec::with_capacity(samples);
    let mut sum = 0.0f64;
    let mut max = 0.0f64;
    let mut first_error: Option<FloatLoc> = None;
    let mut differing_pixels = 0u64;
    let row = reference.width as usize;

    for pixel in 0..total_pixels {
        let mut differs = false;
        for channel in 0..4 {
            let index = pixel * 4 + channel;
            let delta = (f64::from(reference.pixels[index]) - f64::from(candidate.pixels[index]))
                .abs();
            errors.push(delta);
            if delta > 0.0 {
                differs = true;
                sum += delta;
                if delta > max {
                    max = delta;
                }
                if first_error.is_none() {
                    first_error = Some(FloatLoc {
                        x: (pixel % row) as u32,
                        y: (pixel / row) as u32,
                        channel: channel as u8,
                        delta,
                    });
                }
            }
        }
        if differs {
            differing_pixels += 1;
        }
    }

    errors.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p99 = if errors.is_empty() {
        0.0
    } else {
        let index = ((0.99 * (errors.len() - 1) as f64) as usize).min(errors.len() - 1);
        errors[index]
    };

    Ok(FloatParityStats {
        width: reference.width,
        height: reference.height,
        samples,
        max_error: max,
        mean_error: sum / samples as f64,
        p99_error: p99,
        error_pixel_ratio: differing_pixels as f64 / total_pixels as f64,
        first_error,
    })
}

fn assert_parity_f32(label: &str, reference: &RgbaF32, candidate: &RgbaF32, tolerance: f64) {
    match diff_f32(reference, candidate) {
        Ok(stats) if stats.max_error == 0.0 => {
            eprintln!("{label}: bit-identical ({} float samples)", stats.samples);
        }
        Ok(stats) if stats.max_error <= tolerance => {
            eprintln!(
                "{label}: within tolerance {tolerance:.8}\n{}",
                stats.report()
            );
        }
        Ok(stats) => panic!(
            "{label}: parity broken, tolerance {tolerance:.8}\n{}",
            stats.report()
        ),
        Err(error) => panic!("{label}: {error}"),
    }
}

// ---------------------------------------------------------------------------
// Engine helpers
// ---------------------------------------------------------------------------

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Per-test scratch folder, removed on drop. Same pattern as
/// `tests/export_faults_bo4.rs`; no external fixture is involved.
struct Temp(PathBuf);

impl Temp {
    fn new(tag: &str) -> Self {
        let id = COUNTER.fetch_add(1, Ordering::SeqCst);
        let path = std::env::temp_dir().join(format!(
            "aurora-quality-parity-{tag}-{}-{id}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A 33x17 PNG: odd width (not 8-aligned) and structure at every frequency, so
/// the decode + develop + detail path has edges to act on.
fn write_parity_png(path: &Path) {
    let (width, height) = (33u32, 17u32);
    let mut pixels = vec![0u8; (width * height * 4) as usize];
    for y in 0..height {
        for x in 0..width {
            let i = ((y * width + x) * 4) as usize;
            pixels[i] = ((x * 8) % 256) as u8;
            pixels[i + 1] = ((y * 16) % 256) as u8;
            pixels[i + 2] = (((x + y) * 7) % 256) as u8;
            pixels[i + 3] = 255;
        }
    }
    let image = image::RgbaImage::from_raw(width, height, pixels).expect("matching buffer size");
    image.save(path).expect("write PNG fixture");
}

fn preview_request(
    image: ImageId,
    adjustments: &AdjustmentSet,
    max_size: u32,
    quality: RenderQuality,
    bypass: bool,
) -> PreviewRequest {
    PreviewRequest {
        image,
        adjustments: adjustments.clone(),
        region: NormRect::FULL,
        max_size,
        quality,
        bypass,
        mask_overlay: None,
        generation: 0,
        display_profile: None,
        proof_profile: None,
        proof_paper_white: 1.0,
        proof_black_point: 0.0,
        rendering_intent: None,
        black_point_compensation: false,
    }
}

// ---------------------------------------------------------------------------
// Self-check: the ruler itself
// ---------------------------------------------------------------------------

#[test]
fn parity_statistics_locate_a_known_single_pixel_error() {
    // 8-bit: one channel of one pixel is 9 levels low.
    let base = gradient8(64, 48);
    let mut damaged = base.clone();
    let index = ((20 * 64 + 33) * 4) as usize + 2;
    damaged.pixels[index] = damaged.pixels[index].saturating_sub(9);
    let stats = diff8(&base, &damaged).expect("same size");
    assert_eq!(stats.max_error, 9, "max error must be the injected delta");
    assert_eq!(
        stats.first_error.map(|e| (e.x, e.y, e.channel)),
        Some((33, 20, 2)),
        "first error must name the injected location"
    );
    assert_eq!(stats.worst_error.map(|e| e.delta), Some(9));
    assert!(
        (stats.error_pixel_ratio - 1.0 / (64.0 * 48.0)).abs() < 1e-12,
        "one damaged pixel over 3072 must be reported as the pixel share"
    );
    assert_eq!(
        stats.p99_error, 0,
        "one channel out of 12288 samples must stay below the 99th percentile"
    );
    // Identity must be exactly zero.
    let identical = diff8(&base, &base).expect("same size");
    assert_eq!(identical.max_error, 0);
    assert!(identical.first_error.is_none());

    // 16-bit: same check in native levels, plus the 8-bit equivalence.
    let base16 = gradient16(97, 65);
    let mut damaged16 = Rgba16 {
        width: 97,
        height: 65,
        pixels: base16.clone(),
    };
    let index16 = ((11 * 97 + 5) * 4) as usize + 1;
    damaged16.pixels[index16] = damaged16.pixels[index16].saturating_sub(257 * 9);
    let reference16 = Rgba16 {
        width: 97,
        height: 65,
        pixels: base16,
    };
    let stats16 = diff16(&reference16, &damaged16).expect("same size");
    assert_eq!(stats16.max_error, 2313);
    assert!(
        (stats16.eight_bit(stats16.max_error) - 9.0).abs() < 1e-9,
        "u16 delta 2313 must report as 9 8-bit levels"
    );
    assert_eq!(
        stats16.first_error.map(|e| (e.x, e.y, e.channel)),
        Some((5, 11, 1))
    );

    // Float: tolerance is honoured, but max and P99 are both reported.
    let float_reference = RgbaF32 {
        width: 16,
        height: 16,
        pixels: vec![0.5; 16 * 16 * 4],
    };
    let mut float_candidate = float_reference.clone();
    let float_index = ((3 * 16 + 7) * 4) as usize;
    float_candidate.pixels[float_index] += 0.02;
    let float_stats = diff_f32(&float_reference, &float_candidate).expect("same size");
    assert!((float_stats.max_error - 0.02).abs() < 1e-6);
    assert!((float_stats.p99_error - 0.0).abs() < 1e-9);
    assert_eq!(
        float_stats.first_error.map(|e| (e.x, e.y, e.channel)),
        Some((7, 3, 0))
    );

    // A tolerance below the real error must fail (the harness must be able to
    // say no). Panic output is silenced only for this window; no other test is
    // expected to panic concurrently, and the hook is restored immediately.
    let previous_hook = take_hook();
    set_hook(Box::new(|_| {}));
    let u8_failed = catch_unwind(|| assert_parity_u8("self-check negative", &base, &damaged, 0));
    let u16_failed = catch_unwind(|| {
        assert_parity_u16("self-check negative u16", &reference16, &damaged16, 100)
    });
    let f32_failed = catch_unwind(|| {
        assert_parity_f32("self-check negative f32", &float_reference, &float_candidate, 0.005)
    });
    set_hook(previous_hook);
    assert!(u8_failed.is_err(), "a 9-level error with tolerance 0 must fail");
    assert!(u16_failed.is_err(), "a 2313-level error with tolerance 100 must fail");
    assert!(f32_failed.is_err(), "a 0.02 error with tolerance 0.005 must fail");
}

// ---------------------------------------------------------------------------
// Case 1: Draft vs Final at the same size and parameters (MockEngine)
// ---------------------------------------------------------------------------

/// The engine-level invariant behind roadmap §1 item 2: `RenderQuality` must
/// not change pixels when the output size is the same. `MockEngine` never
/// enables the half-size `fast` branch, so this case is green today and must
/// stay green; the real `FileEngine` red line is the `#[ignore]`d case below.
#[test]
fn mock_engine_draft_and_final_quality_render_identically() {
    let mut adjustments = AdjustmentSet::default();
    adjustments.exposure.exposure = 0.5;
    adjustments.exposure.saturation = 15.0;
    adjustments.clarity.clarity = 25.0;

    // Two engines, so neither request is served from the other's preview cache.
    let final_engine = MockEngine::new();
    let draft_engine = MockEngine::new();
    let final_image = final_engine.images()[0].id;
    let draft_image = draft_engine.images()[0].id;

    let final_request = preview_request(final_image, &adjustments, 256, RenderQuality::Final, false);
    let draft_request = preview_request(draft_image, &adjustments, 256, RenderQuality::Draft, false);

    let final_frame = final_engine.render_preview(&final_request).unwrap();
    let draft_frame = draft_engine.render_preview(&draft_request).unwrap();

    assert_eq!(
        (final_frame.image.width, final_frame.image.height),
        (draft_frame.image.width, draft_frame.image.height),
        "Draft and Final must keep the same output size"
    );
    assert_parity_u8(
        "Draft vs Final (MockEngine, same size and parameters)",
        &Rgba8::of_engine(&final_frame.image),
        &Rgba8::of_engine(&draft_frame.image),
        0,
    );

    // The trait documents `render_preview_fast(req, false) == render_preview(req)`.
    // MockEngine ignores the flag entirely (`let fast = false`), and this pins
    // that down so a future mock change cannot hide a size/pixel jump.
    let fast_frame = draft_engine.render_preview_fast(&draft_request, true).unwrap();
    assert_parity_u8(
        "fast flag vs slow path (MockEngine)",
        &Rgba8::of_engine(&draft_frame.image),
        &Rgba8::of_engine(&fast_frame.image),
        0,
    );
}

// ---------------------------------------------------------------------------
// Case 2: FileEngine decode + render is deterministic across cold/warm paths
// ---------------------------------------------------------------------------

/// Two independent `FileEngine`s over identical bytes, plus a warm cache hit in
/// one of them: all three renders must be byte-identical. This is the
/// "cold sampling path vs cached path" parity every acceleration patch relies
/// on, and it needs no external fixture (the PNG is generated here).
#[test]
fn file_engine_cold_and_warm_previews_render_identically() {
    let temp_a = Temp::new("cold-a");
    let temp_b = Temp::new("cold-b");
    write_parity_png(&temp_a.path().join("frame.png"));
    write_parity_png(&temp_b.path().join("frame.png"));

    let engine_a = FileEngine::open(temp_a.path()).unwrap();
    let engine_b = FileEngine::open(temp_b.path()).unwrap();

    let mut adjustments = AdjustmentSet::default();
    adjustments.exposure.exposure = 0.3;
    adjustments.exposure.contrast = 12.0;

    let request_a = preview_request(engine_a.images()[0].id, &adjustments, 128, RenderQuality::Final, false);
    let request_b = preview_request(engine_b.images()[0].id, &adjustments, 128, RenderQuality::Final, false);

    let cold_a = engine_a.render_preview(&request_a).unwrap();
    let cold_b = engine_b.render_preview(&request_b).unwrap();
    // Same parameters and image in engine A again: the second request is served
    // from the preview cache (a different code path than the cold render).
    let warm_a = engine_a.render_preview(&request_a).unwrap();

    assert_eq!(
        (cold_a.image.width, cold_a.image.height),
        (cold_b.image.width, cold_b.image.height)
    );
    assert_parity_u8(
        "two cold FileEngine renders over identical bytes",
        &Rgba8::of_engine(&cold_a.image),
        &Rgba8::of_engine(&cold_b.image),
        0,
    );
    assert!(
        std::sync::Arc::ptr_eq(&cold_a.image.pixels, &warm_a.image.pixels),
        "the third request should have hit the preview cache"
    );
    assert_parity_u8(
        "cold render vs warm cache hit",
        &Rgba8::of_engine(&cold_a.image),
        &Rgba8::of_engine(&warm_a.image),
        0,
    );
}

// ---------------------------------------------------------------------------
// Case 3: the two white-balance encodings must agree byte for byte
// ---------------------------------------------------------------------------

/// Real, device-free "second path" switch in the CPU pipeline: the import can
/// persist Capture One's exact `white_balance.gains`, and `render_wb_gains`
/// then uses them verbatim instead of the Kelvin/Tint approximation. At the
/// 5600 K / 0 tint default the approximation is exactly `[1.0, 1.0, 1.0]`
/// (`kelvin_rgb(5600) / kelvin_rgb(5600)`), so the exact-gains encoding must
/// render the same bytes. Any drift here means one of the two encodings is
/// being applied twice or skipped.
#[test]
fn white_balance_exact_gains_and_kelvin_tint_agree_byte_for_byte() {
    let source_pixels = gradient16(64, 48);
    let via_kelvin = Source::from_rgba16(64, 48, &source_pixels);
    let via_gains = Source::from_rgba16(64, 48, &source_pixels);

    let default_encoding = AdjustmentSet::default();
    let mut exact_gains = AdjustmentSet::default();
    exact_gains.white_balance.gains = Some([1.0, 1.0, 1.0]);

    let kelvin_render = render(&via_kelvin, &default_encoding, NormRect::FULL, 64, false, None);
    let gains_render = render(&via_gains, &exact_gains, NormRect::FULL, 64, false, None);

    assert_parity_u8(
        "Kelvin/Tint identity vs exact-gains identity (CPU develop path)",
        &Rgba8::of_rendered(&kelvin_render),
        &Rgba8::of_rendered(&gains_render),
        0,
    );
}

// ---------------------------------------------------------------------------
// Case 4: the synthetic geometry matrix renders deterministically
// ---------------------------------------------------------------------------

/// Exercises the generator set end to end: solid, ramp, high-frequency
/// checkerboard, single-pixel spikes, 1x1, 2x2, a non-8-aligned width and a
/// transparent border. Two independently built `Source`s carry the same bytes,
/// so the sample-tile cache cannot make the comparison trivial; both renders
/// must still be byte-identical.
#[test]
fn synthetic_geometry_matrix_renders_deterministically() {
    let cases: Vec<(&str, u32, u32, Vec<u16>)> = vec![
        ("solid", 64, 48, solid16(64, 48, [12000, 30000, 50000])),
        ("gradient_odd_width", 97, 65, gradient16(97, 65)),
        ("checker_high_frequency", 40, 40, checker16(40, 40, 2)),
        ("single_pixel_spikes", 64, 64, spike_noise16(64, 64, 0xC0FF_EE00)),
        ("one_by_one", 1, 1, solid16(1, 1, [50000, 2500, 2500])),
        ("two_by_two", 2, 2, solid16(2, 2, [1000, 2000, 3000])),
    ];

    for (name, width, height, pixels) in &cases {
        let first = Source::from_rgba16(*width, *height, pixels);
        let second = Source::from_rgba16(*width, *height, pixels);
        let a = render(&first, &AdjustmentSet::default(), NormRect::FULL, 48, false, None);
        let b = render(&second, &AdjustmentSet::default(), NormRect::FULL, 48, false, None);
        assert_eq!((a.width, a.height), (b.width, b.height), "{name}: size");
        assert_parity_u8(name, &Rgba8::of_rendered(&a), &Rgba8::of_rendered(&b), 0);
    }

    // Transparent border: the source alpha is flattened at ingress and the
    // renderer always emits opaque frames. Pinned here so a parity failure can
    // never be blamed on alpha handling the engine does not claim to do.
    let mut border = gradient16(32, 32);
    with_transparent_border(&mut border, 32, 32);
    let source = Source::from_rgba16(32, 32, &border);
    let out = render(&source, &AdjustmentSet::default(), NormRect::FULL, 32, false, None);
    assert!(
        out.rgba.chunks_exact(4).all(|pixel| pixel[3] == 255),
        "the CPU renderer must emit opaque alpha regardless of source alpha"
    );
}

// ---------------------------------------------------------------------------
// Case 5: the 16-bit export path is deterministic and reports native levels
// ---------------------------------------------------------------------------

/// Uses `render16` (the wide/export path: no sample-tile cache, a separate
/// 16-bit buffer) with the default detail adjustments, so the f32 detail path
/// runs. Two independent sources must agree byte for byte in native 16-bit
/// levels; this is also the case that proves the u16 half of the harness is
/// wired into a real engine call.
#[test]
fn wide_export_path_renders_deterministically() {
    let pixels = gradient16(64, 48);
    let mut adjustments = AdjustmentSet::default();
    adjustments.exposure.exposure = 0.5;

    let first = Source::from_rgba16(64, 48, &pixels);
    let second = Source::from_rgba16(64, 48, &pixels);
    let a = render16(&first, &adjustments, NormRect::FULL, 64, false);
    let b = render16(&second, &adjustments, NormRect::FULL, 64, false);

    assert_eq!((a.width, a.height), (b.width, b.height));
    assert_parity_u16(
        "wide export path rendered twice (CPU)",
        &Rgba16::of_rendered(&a),
        &Rgba16::of_rendered(&b),
        0,
    );
}

// ---------------------------------------------------------------------------
// Case 6 (red line): FileEngine Draft must equal Final at the same size
// ---------------------------------------------------------------------------

/// A fast/interactive request must not substitute half-resolution pixels at the engine boundary.
#[test]
fn file_engine_draft_preview_matches_final_byte_for_byte() {
    let temp = Temp::new("draft-vs-final");
    write_parity_png(&temp.path().join("frame.png"));
    let engine = FileEngine::open(temp.path()).unwrap();

    let mut adjustments = AdjustmentSet::default();
    adjustments.exposure.exposure = 0.3;
    adjustments.clarity.clarity = 20.0;

    let image = engine.images()[0].id;
    let final_request = preview_request(image, &adjustments, 256, RenderQuality::Final, false);
    let draft_request = preview_request(image, &adjustments, 256, RenderQuality::Draft, false);

    let final_frame = engine.render_preview(&final_request).unwrap();
    let draft_frame = engine.render_preview_fast(&draft_request, true).unwrap();

    assert_eq!(
        (final_frame.image.width, final_frame.image.height),
        (draft_frame.image.width, draft_frame.image.height),
        "Draft and Final must at least agree on the output size"
    );
    assert_parity_u8(
        "FileEngine Draft (fast) vs Final at the same size and parameters",
        &Rgba8::of_preview(&final_frame),
        &Rgba8::of_preview(&draft_frame),
        0,
    );
}

/// Exercise the same scoped FileEngine entry points as the viewer on a real RAW.
/// Only the copy enters the writable catalog. GPU-specific assertions are opt-in.
#[test]
#[ignore = "requires AURORA_TEST_RAW pointing to a real full-resolution RAW and its decoder"]
fn real_raw_scoped_draft_matches_final_display_pixels() {
    use aurora_engine::{viewer_maps, ScopeWants};

    let raw = PathBuf::from(std::env::var_os("AURORA_TEST_RAW").expect("set AURORA_TEST_RAW"));
    let temp = Temp::new("scoped-raw");
    let mut copy = temp.path().join("control");
    copy.set_extension(raw.extension().expect("RAW extension"));
    std::fs::copy(&raw, copy).expect("copy fixture to an isolated catalog");
    let require_bins = std::env::var("AURORA_TEST_REQUIRE_SCOPE_GPU").as_deref() == Ok("1");
    let engine = FileEngine::open(temp.path()).unwrap();
    let images = engine.images();
    assert_eq!(images.len(), 1);
    let mut previous = None;
    for (step, ev) in [0.30_f32, 1.1085948].into_iter().enumerate() {
        let mut adj = AdjustmentSet::default();
        adj.exposure.exposure = ev;
        let mut req = preview_request(images[0].id, &adj, 2137, RenderQuality::Draft, false);
        req.generation = 2 * step as u64 + 1;
        let d = engine.render_preview_scoped(&req, ScopeWants::WAVEFORM).unwrap();
        assert_eq!(d.generation, req.generation);
        req.quality = RenderQuality::Final;
        req.generation += 1;
        let f = engine.render_preview_scoped(&req, ScopeWants::WAVEFORM).unwrap();
        assert_eq!(f.generation, req.generation);
        let (draft, final_frame) = (Rgba8::of_preview(&d), Rgba8::of_preview(&f));
        assert_eq!(draft.width.max(draft.height), 2137);
        assert_eq!(d.histogram, f.histogram, "histograms must describe the same pixels");
        assert_parity_u8("real RAW full-quality drag vs final", &final_frame, &draft, 0);
        if let Some(prev) = previous {
            assert!(final_frame.pixels != prev, "distinct exposures must not reuse stale pixels");
        }
        if require_bins {
            let device = d.scopes.as_ref().expect("a device waveform was explicitly requested");
            let cpu = viewer_maps(&draft.pixels, draft.width as usize, draft.height as usize, false, ScopeWants::WAVEFORM);
            assert!(device.sampled() > 0);
            assert_eq!(device.sampled(), cpu.sampled());
            assert!(device.wave_bins() == cpu.wave_bins(), "device waveform differs from CPU on displayed pixels");
            eprintln!("[real-raw-waveform] bins={} sampled={} all equal", device.wave_bins().len(), device.sampled());
        }
        eprintln!("[real-raw-preview] ev={ev} size={}x{} compared={} bytes; max_diff=0", draft.width, draft.height, draft.pixels.len());
        previous = Some(final_frame.pixels);
    }
}

// ---------------------------------------------------------------------------
// Case 7: the sampled-tile cache branch vs direct per-pixel sampling
// ---------------------------------------------------------------------------

// `mock/process.rs:2999` picks between two sampling implementations for a
// halo frame: cached 256-px tiles when the padded frame fits
// `AURORA_SAMPLE_TILE_BUDGET_MB`, and direct per-pixel sampling otherwise.
// The budget is a process-wide `OnceLock` (`process.rs:252-263`), so the two
// branches have to run in two child processes of this same test binary and be
// compared byte for byte. The GPU parity tests in `mock/process.rs:9780-9805`
// use the same child-process protocol for the same reason.
const TILE_SWITCH_TEST: &str = "sampled_tile_cache_branch_matches_direct_sampling";
const TILE_CHILD_ENV: &str = "AURORA_TILE_PARITY_CHILD";
const TILE_OUT_ENV: &str = "AURORA_TILE_PARITY_OUT";

fn tile_switch_child() {
    let out_path = std::env::var(TILE_OUT_ENV).expect("child output path");
    // A sub-frame region enables the halo path (`process.rs:2918-2920`); the
    // default noise reduction + sharpening give it a real radius
    // (`detail.rs:77-88`). At max_size 480 the padded frame is several MiB of
    // `SourceSample`, so a 1 MiB budget falls through to direct sampling while
    // the default 192 MiB (or 512 MiB here) budget keeps the tiled branch.
    let pixels = checker16(97, 65, 2);
    let src = Source::from_rgba16(97, 65, &pixels);
    let mut adjustments = AdjustmentSet::default();
    // Pinned, not inherited from the defaults: single-pixel NR alone forces
    // `use_f32_detail` (process.rs:2718-2722) and a non-zero halo
    // (detail.rs:77-88), so the sampling-branch test never silently degrades
    // into the halo-free band path if the shipped defaults change.
    adjustments.noise_reduction.single_pixel = 100.0;
    adjustments.sharpening.amount = 250.0;
    let region = NormRect { x: 0.05, y: 0.07, width: 0.6, height: 0.55 };
    let rendered = render(&src, &adjustments, region, 480, false, None);
    // `geometry_for` returns round(rw * max_size / max(rw, rh)) = 480x295 at
    // HEAD. With the default NR/sharpening halo (about 77 px at this scale)
    // the padded work frame is roughly 4.5 MiB of `SourceSample`, comfortably
    // above the 1 MiB budget and far below 512 MiB.
    let mut dump = Vec::with_capacity(8 + rendered.rgba.len());
    dump.extend_from_slice(&rendered.width.to_le_bytes());
    dump.extend_from_slice(&rendered.height.to_le_bytes());
    dump.extend_from_slice(&rendered.rgba);
    std::fs::write(&out_path, dump).expect("write tile parity dump");
}

fn run_tile_switch_child(budget_mb: &str, path: &Path) {
    let exe = std::env::current_exe().expect("test executable");
    let output = std::process::Command::new(exe)
        .args(["--exact", TILE_SWITCH_TEST, "--nocapture"])
        .env(TILE_CHILD_ENV, "1")
        .env(TILE_OUT_ENV, path)
        .env("AURORA_SAMPLE_TILE_BUDGET_MB", budget_mb)
        // Both children must take the same CPU reference path for every other
        // stage; only the sampling branch may differ.
        .env("AURORA_DEVELOP_GPU", "0")
        .env("AURORA_DETAIL_GPU", "0")
        .env("AURORA_NR_GPU", "0")
        .env("AURORA_CLARITY_GPU", "0")
        .env("AURORA_PIXEL_VIGNETTE_GPU", "0")
        .env("AURORA_PIXEL_GPU", "0")
        .output()
        .expect("spawn tile parity child");
    assert!(
        output.status.success(),
        "tile parity child (budget {budget_mb} MiB) failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn read_tile_dump(path: &Path) -> Rgba8 {
    let bytes = std::fs::read(path).expect("tile parity dump");
    assert!(bytes.len() >= 8, "tile parity dump is truncated");
    let width = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
    let height = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    Rgba8 { width, height, pixels: bytes[8..].to_vec() }
}

/// `mock/process.rs:417-419` says the tiled sampler is "bit-identical" to
/// direct sampling. This is that claim as a test: the same halo render under a
/// 512 MiB budget (tiles) and a 1 MiB budget (direct) must agree on every
/// channel. Pure CPU - no GPU, fixture, network or clock - so it runs on CI.
#[test]
fn sampled_tile_cache_branch_matches_direct_sampling() {
    if std::env::var_os(TILE_CHILD_ENV).is_some() {
        tile_switch_child();
        return;
    }
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    let tiled_path = dir.join(format!("aurora-tile-parity-{pid}-tiled.bin"));
    let direct_path = dir.join(format!("aurora-tile-parity-{pid}-direct.bin"));
    run_tile_switch_child("512", &tiled_path);
    let tiled = read_tile_dump(&tiled_path);
    let _ = std::fs::remove_file(&tiled_path);
    run_tile_switch_child("1", &direct_path);
    let direct = read_tile_dump(&direct_path);
    let _ = std::fs::remove_file(&direct_path);

    assert_eq!(
        (tiled.width, tiled.height),
        (direct.width, direct.height),
        "the sampling branch must not change the output size"
    );
    assert!(
        tiled.width > 0 && tiled.height > 0 && !tiled.pixels.is_empty(),
        "the child produced an empty frame"
    );
    assert_parity_u8(
        "sampled-tile branch vs direct sampling (AURORA_SAMPLE_TILE_BUDGET_MB 512 vs 1)",
        &tiled,
        &direct,
        0,
    );
}
