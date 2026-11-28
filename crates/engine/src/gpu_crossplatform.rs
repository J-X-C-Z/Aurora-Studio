//! Cross-platform GPU path for the develop, demosaic and detail stages.
//!
//! Three stages that the engine needs on every platform, with one portable
//! implementation per stage and a CPU oracle that each GPU path is checked
//! against:
//!
//! 1. **RAW Bayer demosaic** — Malvar-He-Cutler (ICASSP 2004) gradient-corrected
//!    5x5 linear interpolation over RGGB / BGGR / GRBG / GBRG phases, mirrored
//!    borders, RGBA f32 linear output.
//! 2. **Develop** — one pass of white balance gains, exposure, highlight/shadow
//!    tone toe, contrast, saturation, linear-to-sRGB and RGBA8 packing, with the
//!    alpha channel carried through untouched.
//! 3. **Detail** — separable 5-tap gaussian sharpen/clarity plus a 256-bin
//!    R/G/B/luma histogram reduced through `atomicAdd`.
//!
//! Shaders live beside this module:
//!
//! ```text
//! crates/engine/native/wgsl/raw_demosaic.wgsl
//! crates/engine/native/wgsl/develop_stage.wgsl
//! crates/engine/native/wgsl/detail_histogram.wgsl
//! crates/engine/native/metal/raw_demosaic.metal
//! crates/engine/native/metal/develop_stage.metal
//! ```
//!
//! [`CrossPlatformGpuEngine`](crate::gpu_crossplatform::CrossPlatformGpuEngine)
//! executes the WGSL twins through `wgpu` 30 (Vulkan / Direct3D 12 / Metal) and
//! never silently replaces itself with the CPU reference: under
//! `ExecutionMode::Strict` a missing adapter, a failed device request or a
//! device error is returned to the caller. Choosing a CPU
//! fallback is a policy decision for `contracts::compute::select_backend`, not
//! something this module does behind the caller's back.
//!
//! # Verification status
//!
//! * The CPU references are covered by synthetic tests that run everywhere.
//! * The GPU paths are covered by GPU-vs-CPU parity tests (demosaic < 1e-4,
//!   develop <= 1 LSB, histogram exact) that run only where an adapter exists;
//!   on a machine without one they print `SKIP` and do not claim a pass.
//! * The Metal sources are not compiled by any Windows or Linux build. They are
//!   `IMPLEMENTED`, not `COMPILED`, until a macOS device test has run.
//! * Passing parity on synthetic planes is not evidence against a real NEF, a
//!   real Capture One render, or any vendor demosaic.

use std::fmt;

use crate::processing::color::{contrast, linear_to_srgb, luma};

/// Bins per histogram channel.
pub const HISTOGRAM_BINS_PER_CHANNEL: usize = 256;
/// Histogram channels: red, green, blue, luma.
pub const HISTOGRAM_CHANNELS: usize = 4;
/// Total length of a histogram buffer.
pub const HISTOGRAM_LEN: usize = HISTOGRAM_BINS_PER_CHANNEL * HISTOGRAM_CHANNELS;

/// Byte size of the develop uniform block. Byte-identical in the WGSL and MSL
/// twins: `vec4` + five `f32` + three pad.
pub const DEVELOP_UNIFORM_BYTES: usize = 48;
/// Byte size of the demosaic uniform block.
pub const DEMOSAIC_UNIFORM_BYTES: usize = 16;
/// Byte size of the detail uniform block.
pub const DETAIL_UNIFORM_BYTES: usize = 16;

// ---------------------------------------------------------------------------
// CFA phase
// ---------------------------------------------------------------------------

/// The 2x2 colour-filter phase of a Bayer sensor.
///
/// The packed nibble order matches the shaders:
/// nibble 0 = `(x even, y even)`, 1 = `(x odd, y even)`,
/// 2 = `(x even, y odd)`, 3 = `(x odd, y odd)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CfaPattern {
    /// Red at the origin. `0x2110`.
    Rggb,
    /// Blue at the origin. `0x0112`.
    Bggr,
    /// Green then red on the first row. `0x1201`.
    Grbg,
    /// Green then blue on the first row. `0x1021`.
    Gbrg,
}

impl CfaPattern {
    /// Value passed to the shaders in `DemosaicParams.pattern`.
    pub const fn as_u32(self) -> u32 {
        match self {
            Self::Rggb => 0,
            Self::Bggr => 1,
            Self::Grbg => 2,
            Self::Gbrg => 3,
        }
    }

    /// Parse the shader-side enumerator.
    pub const fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Rggb),
            1 => Some(Self::Bggr),
            2 => Some(Self::Grbg),
            3 => Some(Self::Gbrg),
            _ => None,
        }
    }

    /// The packed 2x2 phase, low nibble first.
    pub const fn packed(self) -> u32 {
        match self {
            Self::Rggb => 0x2110,
            Self::Bggr => 0x0112,
            Self::Grbg => 0x1201,
            Self::Gbrg => 0x1021,
        }
    }

    /// Colour code at `(x, y)`: 0 = red, 1 = green, 2 = blue.
    pub const fn code_at(self, x: u32, y: u32) -> u32 {
        let index = (y & 1) * 2 + (x & 1);
        (self.packed() >> (index * 4)) & 0xf
    }
}

// ---------------------------------------------------------------------------
// Develop parameters
// ---------------------------------------------------------------------------

/// Parameters of the fused develop pass.
///
/// Every value is normalised the way the CPU renderer stores it, so a caller
/// can hand over `AdjustmentSet` fields without a second scaling step:
/// `highlights` / `shadows` are the slider value divided by 100 and clamped to
/// `-1..1`, `contrast` is `contrast / 50 * 0.8`, and `saturation` is a
/// multiplier around BT.709 luma with `1.0` neutral.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DevelopParams {
    /// Per-channel white-balance gains, normally green-normalised.
    pub wb: [f32; 3],
    /// Exposure in stops. The shader multiplies by `2^exposure_ev`.
    pub exposure_ev: f32,
    /// Highlight recovery in `-1..1` (slider value / 100).
    pub highlights: f32,
    /// Shadow lift in `-1..1` (slider value / 100).
    pub shadows: f32,
    /// Contrast amount in `-1..1` (already `contrast / 50 * 0.8`).
    pub contrast: f32,
    /// Saturation multiplier, `1.0` neutral.
    pub saturation: f32,
}

impl Default for DevelopParams {
    fn default() -> Self {
        Self {
            wb: [1.0, 1.0, 1.0],
            exposure_ev: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            contrast: 0.0,
            saturation: 1.0,
        }
    }
}

impl DevelopParams {
    /// `2^EV`, the multiplier the shader applies.
    pub fn exposure_scale(&self) -> f32 {
        self.exposure_ev.exp2()
    }

    /// Whether every parameter is finite, so the shaders and the reference
    /// cannot disagree on a NaN.
    pub fn is_finite(&self) -> bool {
        self.wb.iter().all(|v| v.is_finite())
            && [self.exposure_ev, self.highlights, self.shadows, self.contrast, self.saturation]
                .iter()
                .all(|v| v.is_finite())
    }

    fn toning(&self) -> (f32, f32) {
        (self.highlights.clamp(-1.0, 1.0), self.shadows.clamp(-1.0, 1.0))
    }
}

/// Everything that can go wrong on the cross-platform GPU path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuEngineError {
    /// No adapter was reported for the requested backend set.
    NoAdapter {
        /// The backend mask that was requested, for the diagnostic.
        backends: String,
    },
    /// The adapter was found but `request_device` failed.
    DeviceRequest(String),
    /// The host passed a frame that does not match its own dimensions.
    InvalidInput(&'static str),
    /// Dispatch, mapping or readback failed.
    Execution(String),
}

impl fmt::Display for GpuEngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter { backends } => {
                write!(f, "no cross-platform GPU adapter for backends {backends}")
            }
            Self::DeviceRequest(detail) => write!(f, "GPU device request failed: {detail}"),
            Self::InvalidInput(detail) => write!(f, "invalid input: {detail}"),
            Self::Execution(detail) => write!(f, "GPU execution failed: {detail}"),
        }
    }
}

impl std::error::Error for GpuEngineError {}

// ---------------------------------------------------------------------------
// CPU reference oracle — demosaic
// ---------------------------------------------------------------------------

/// Mirror an index at a frame edge: `-1 -> 1`, `n -> n-2`, then clamp.
///
/// One reflection covers the +/-2 taps of a 5x5 kernel for any axis of two or
/// more pixels; a one-pixel axis pins to 0. The shaders use this exact rule.
pub fn mirror_index(i: isize, n: isize) -> usize {
    if n <= 1 {
        return 0;
    }
    let mut v = i;
    if v < 0 {
        v = -v;
    }
    if v >= n {
        v = 2 * n - 2 - v;
    }
    v.clamp(0, n - 1) as usize
}

/// Malvar-He-Cutler demosaic, the CPU oracle for the WGSL and MSL twins.
///
/// `bayer` is one linear sample per photosite, row-major, already
/// black-subtracted and normalised by the caller. The result is RGBA f32 with
/// alpha `1.0`, `width * height * 4` values.
///
/// Only the interpolation is reproduced here: black level, white level and
/// highlight recovery belong to the stage before this one.
pub fn demosaic_bayer_cpu_ref(
    bayer: &[f32],
    width: usize,
    height: usize,
    pattern: CfaPattern,
) -> Vec<f32> {
    assert_eq!(
        bayer.len(),
        width * height,
        "bayer plane must hold width * height samples"
    );
    let mut out = vec![0.0f32; width * height * 4];
    if width == 0 || height == 0 {
        return out;
    }
    let w = width as isize;
    let h = height as isize;
    let tap = |x: isize, y: isize| -> f32 {
        let xx = mirror_index(x, w);
        let yy = mirror_index(y, h);
        bayer[yy * width + xx]
    };
    let tap_ref: &dyn Fn(isize, isize) -> f32 = &tap;

    for y in 0..height {
        for x in 0..width {
            let (xi, yi) = (x as isize, y as isize);
            let code = pattern.code_at(x as u32, y as u32);
            let center = bayer[y * width + x];
            let (r, g, b) = if code == 1 {
                // Green site: the row neighbours carry one colour and the
                // column neighbours the other; only the CFA phase says which.
                let across = rb_at_g_row(tap_ref, xi, yi);
                let down = rb_at_g_col(tap_ref, xi, yi);
                if pattern.code_at(x as u32 + 1, y as u32) == 0 {
                    (across, center, down)
                } else {
                    (down, center, across)
                }
            } else if code == 0 {
                (center, g_at_rb(tap_ref, xi, yi), rb_at_opposite(tap_ref, xi, yi))
            } else {
                (rb_at_opposite(tap_ref, xi, yi), g_at_rb(tap_ref, xi, yi), center)
            };
            let p = (y * width + x) * 4;
            out[p] = r;
            out[p + 1] = g;
            out[p + 2] = b;
            out[p + 3] = 1.0;
        }
    }
    out
}

fn g_at_rb(tap: &dyn Fn(isize, isize) -> f32, x: isize, y: isize) -> f32 {
    let c = tap(x, y);
    let n = tap(x, y - 1);
    let s = tap(x, y + 1);
    let e = tap(x + 1, y);
    let west = tap(x - 1, y);
    let n2 = tap(x, y - 2);
    let s2 = tap(x, y + 2);
    let e2 = tap(x + 2, y);
    let w2 = tap(x - 2, y);
    (4.0 * c + 2.0 * (n + s + e + west) - (n2 + s2 + e2 + w2)) * 0.125
}

fn rb_at_g_row(tap: &dyn Fn(isize, isize) -> f32, x: isize, y: isize) -> f32 {
    let c = tap(x, y);
    let e = tap(x + 1, y);
    let west = tap(x - 1, y);
    let e2 = tap(x + 2, y);
    let w2 = tap(x - 2, y);
    let nw = tap(x - 1, y - 1);
    let ne = tap(x + 1, y - 1);
    let sw = tap(x - 1, y + 1);
    let se = tap(x + 1, y + 1);
    let n2 = tap(x, y - 2);
    let s2 = tap(x, y + 2);
    (5.0 * c + 4.0 * (e + west) - (e2 + w2) - (nw + ne + sw + se) + 0.5 * (n2 + s2)) * 0.125
}

fn rb_at_g_col(tap: &dyn Fn(isize, isize) -> f32, x: isize, y: isize) -> f32 {
    let c = tap(x, y);
    let n = tap(x, y - 1);
    let s = tap(x, y + 1);
    let n2 = tap(x, y - 2);
    let s2 = tap(x, y + 2);
    let nw = tap(x - 1, y - 1);
    let ne = tap(x + 1, y - 1);
    let sw = tap(x - 1, y + 1);
    let se = tap(x + 1, y + 1);
    let e2 = tap(x + 2, y);
    let w2 = tap(x - 2, y);
    (5.0 * c + 4.0 * (n + s) - (n2 + s2) - (nw + ne + sw + se) + 0.5 * (e2 + w2)) * 0.125
}

fn rb_at_opposite(tap: &dyn Fn(isize, isize) -> f32, x: isize, y: isize) -> f32 {
    let c = tap(x, y);
    let nw = tap(x - 1, y - 1);
    let ne = tap(x + 1, y - 1);
    let sw = tap(x - 1, y + 1);
    let se = tap(x + 1, y + 1);
    let n2 = tap(x, y - 2);
    let s2 = tap(x, y + 2);
    let e2 = tap(x + 2, y);
    let w2 = tap(x - 2, y);
    (6.0 * c + 2.0 * (nw + ne + sw + se) - 1.5 * (n2 + s2 + e2 + w2)) * 0.125
}

// ---------------------------------------------------------------------------
// CPU reference oracle — develop
// ---------------------------------------------------------------------------

/// `CImgOpLocalToneMappingV8`'s hard toe, the same shape as
/// `processing::tone::hdr_toe` (kernel `z2714f67010`).
///
/// Duplicated rather than imported because that function is private to
/// `processing::tone`; if its constants change, this copy has to change too.
fn hdr_toe(x: f32, k: f32) -> f32 {
    let u = x * (2.0 * 0.9);
    let v = u.abs() + 1.0;
    let r = (u * v) / (v + u * u) * 0.5 + 0.5;
    let r = r.max(u - r + 1.0) * (1.0 / (2.0 * k));
    let s = r * r;
    let num = (1.0 + r + s) * (r + s * s);
    num / (1.0 + num) * k
}

/// One channel-neutral highlight/shadow gain from the luma of `c`.
fn hdr_gain(c: [f32; 3], highlights: f32, shadows: f32) -> f32 {
    if highlights == 0.0 && shadows == 0.0 {
        return 1.0;
    }
    let x = luma(c);
    let mut b = 1.0;
    if highlights != 0.0 {
        let t = hdr_toe(x, 1.0);
        let exponent = if highlights < 0.0 { -t } else { t };
        let m = highlights.abs();
        b = exponent.exp2() * m + (1.0 - m);
    }
    let mut a = 1.0;
    if shadows != 0.0 {
        let t = hdr_toe(-x, 1.0);
        let exponent = if shadows < 0.0 { -t } else { t };
        let m = shadows.abs();
        a = exponent.exp2() * m + (1.0 - m);
    }
    a / b
}

/// `clamp(v, 0, 1) * 255 + 0.5`, truncated: the byte rule shared by the shader,
/// this reference and the CUDA develop kernel.
#[inline]
pub fn quantize_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Fused develop pass, the CPU oracle for `develop_stage.wgsl`.
///
/// `src` is scene-linear RGBA f32 (`w * h * 4` values, alpha in `0..1`); the
/// result is packed RGBA8 of the same length. The order of operations is the
/// order the shader executes, so the two agree to within the platform's
/// `pow` accuracy:
///
/// ```text
/// wb -> exposure -> luma-derived highlight/shadow gain -> contrast
///    -> saturation -> linear_to_srgb -> quantise
/// ```
pub fn develop_cpu_ref(src: &[f32], params: &DevelopParams) -> Vec<u8> {
    assert_eq!(src.len() % 4, 0, "develop input must be RGBA");
    let exposure = params.exposure_scale();
    let (highlights, shadows) = params.toning();
    let mut out = Vec::with_capacity(src.len());
    for px in src.chunks_exact(4) {
        let mut c = [
            px[0] * params.wb[0] * exposure,
            px[1] * params.wb[1] * exposure,
            px[2] * params.wb[2] * exposure,
        ];
        let gain = hdr_gain(c, highlights, shadows);
        c = [c[0] * gain, c[1] * gain, c[2] * gain];
        c = [
            contrast(c[0], params.contrast),
            contrast(c[1], params.contrast),
            contrast(c[2], params.contrast),
        ];
        let l = luma(c);
        let s = params.saturation;
        c = [l + (c[0] - l) * s, l + (c[1] - l) * s, l + (c[2] - l) * s];
        out.push(quantize_u8(linear_to_srgb(c[0])));
        out.push(quantize_u8(linear_to_srgb(c[1])));
        out.push(quantize_u8(linear_to_srgb(c[2])));
        out.push(quantize_u8(px[3]));
    }
    out
}

// ---------------------------------------------------------------------------
// CPU reference oracle — detail + histogram
// ---------------------------------------------------------------------------

/// The 5-tap gaussian the detail stage uses: `[1 4 6 4 1] / 16`.
const GAUSSIAN_5: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];

#[inline]
fn luma8(r: u8, g: u8, b: u8) -> f32 {
    0.2126 * f32::from(r) + 0.7152 * f32::from(g) + 0.0722 * f32::from(b)
}

fn clamp_index(i: isize, n: isize) -> usize {
    i.clamp(0, n - 1) as usize
}

/// 256-bin R/G/B/luma histogram of an RGBA8 frame.
///
/// Layout: `[0, 256)` red, `[256, 512)` green, `[512, 768)` blue,
/// `[768, 1024)` luma. The luma bin is the truncated BT.709 luma of the *byte*
/// values, which is the definition `quantize_rgb3_u8_hist_kernel` in
/// `native/cuda/detail_gpu.cu` uses.
pub fn histogram_cpu_ref(rgba: &[u8]) -> Vec<u32> {
    let mut hist = vec![0u32; HISTOGRAM_LEN];
    for px in rgba.chunks_exact(4) {
        hist[px[0] as usize] += 1;
        hist[256 + px[1] as usize] += 1;
        hist[512 + px[2] as usize] += 1;
        let l = luma8(px[0], px[1], px[2]).clamp(0.0, 255.0);
        hist[768 + l as usize] += 1;
    }
    hist
}

/// Separable 5-tap gaussian, clamp-to-edge, one axis at a time.
fn gaussian_axis(src: &[f32], w: usize, h: usize, horizontal: bool) -> Vec<f32> {
    let (wi, hi) = (w as isize, h as isize);
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, weight) in GAUSSIAN_5.iter().enumerate() {
                let offset = k as isize - 2;
                let (sx, sy) = if horizontal {
                    (clamp_index(x as isize + offset, wi), y)
                } else {
                    (x, clamp_index(y as isize + offset, hi))
                };
                acc += *weight * src[sy * w + sx];
            }
            out[y * w + x] = acc * 0.0625;
        }
    }
    out
}

/// Detail (high-frequency boost) plus histogram, the CPU oracle for
/// `detail_histogram.wgsl`.
///
/// The boost is the one `mock/detail.rs` applies: `luma - low_pass` scaled by
/// `amount`, carried to RGB as a single gain so hue and saturation survive,
/// with the same `0.05` dead band that stops the stage rewriting pixels it did
/// not move.
pub fn detail_histogram_cpu_ref(
    rgba: &[u8],
    width: usize,
    height: usize,
    amount: f32,
) -> (Vec<u8>, Vec<u32>) {
    assert_eq!(
        rgba.len(),
        width * height * 4,
        "detail input must hold width * height RGBA pixels"
    );
    let mut out = rgba.to_vec();
    if width == 0 || height == 0 {
        return (out, vec![0u32; HISTOGRAM_LEN]);
    }
    let mut plane = vec![0.0f32; width * height];
    for (i, px) in rgba.chunks_exact(4).enumerate() {
        plane[i] = luma8(px[0], px[1], px[2]);
    }
    let horizontal = gaussian_axis(&plane, width, height, true);
    let low = gaussian_axis(&horizontal, width, height, false);

    for (i, px) in out.chunks_exact_mut(4).enumerate() {
        let current = plane[i];
        let target = (current + (current - low[i]) * amount).clamp(0.0, 255.0);
        if (target - current).abs() < 0.05 {
            continue;
        }
        let factor = if current > 1.0 { target / current } else { 1.0 };
        px[0] = (f32::from(px[0]) * factor).round().clamp(0.0, 255.0) as u8;
        px[1] = (f32::from(px[1]) * factor).round().clamp(0.0, 255.0) as u8;
        px[2] = (f32::from(px[2]) * factor).round().clamp(0.0, 255.0) as u8;
    }
    let hist = histogram_cpu_ref(&out);
    (out, hist)
}

// ---------------------------------------------------------------------------
// GPU engine
// ---------------------------------------------------------------------------

#[cfg(feature = "wgpu-backend")]
mod gpu {
    use std::sync::mpsc;
    use std::time::Duration;

    use aurora_portrait::autotune::{
        CandidateSpec, ExecutionMode, ParamBound, VerificationStatus,
    };
    use aurora_portrait::device::{Backend as PortraitBackend, Precision as PortraitPrecision};

    use crate::contracts::compute::{
        BackendCapability, BackendKind, Precision as ContractPrecision,
    };

    use super::{
        CfaPattern, DevelopParams, GpuEngineError, DEMOSAIC_UNIFORM_BYTES, DEVELOP_UNIFORM_BYTES,
        DETAIL_UNIFORM_BYTES, HISTOGRAM_LEN,
    };

    const RAW_DEMOSAIC_WGSL: &str = include_str!("../native/wgsl/raw_demosaic.wgsl");
    const DEVELOP_WGSL: &str = include_str!("../native/wgsl/develop_stage.wgsl");
    const DETAIL_WGSL: &str = include_str!("../native/wgsl/detail_histogram.wgsl");

    /// Timeout on a buffer readback. A device that never answers is an error,
    /// not a hang.
    const READBACK_TIMEOUT: Duration = Duration::from_secs(60);

    /// Cross-thread wake-up for the small `block_on` below, so construction and
    /// adapter enumeration do not need an async runtime dependency.
    struct ThreadWaker(std::thread::Thread);

    impl std::task::Wake for ThreadWaker {
        fn wake(self: std::sync::Arc<Self>) {
            self.0.unpark();
        }
        fn wake_by_ref(self: &std::sync::Arc<Self>) {
            self.0.unpark();
        }
    }

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        use std::task::{Context, Poll, Waker};
        let waker = Waker::from(std::sync::Arc::new(ThreadWaker(std::thread::current())));
        let mut context = Context::from_waker(&waker);
        let mut future = std::pin::pin!(future);
        loop {
            match future.as_mut().poll(&mut context) {
                Poll::Ready(value) => return value,
                Poll::Pending => std::thread::park(),
            }
        }
    }

    /// One compiled kernel: the layout its bind group must follow and the
    /// pipeline itself.
    struct GpuPipeline {
        layout: wgpu::BindGroupLayout,
        pipeline: wgpu::ComputePipeline,
    }

    struct PoolEntry {
        buffer: wgpu::Buffer,
        size: u64,
        usage: wgpu::BufferUsages,
        busy: bool,
    }

    /// Buffer use across calls, reported so a caller can tell reuse from
    /// reallocation instead of guessing.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct BufferReuseStats {
        /// Buffers created since the engine was built.
        pub allocations: u64,
        /// Acquisitions satisfied by a resident buffer.
        pub reuses: u64,
        /// Buffers currently held for reuse.
        pub resident_buffers: usize,
        /// Bytes those buffers hold.
        pub resident_bytes: u64,
    }

    /// What the adapter reported, in the engine's own vocabulary.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct GpuAdapterReport {
        /// Adapter name from the driver.
        pub device_name: String,
        /// Driver name, when reported.
        pub driver: String,
        /// wgpu backend that produced the adapter.
        pub wgpu_backend: String,
        /// `true` when the adapter is a software rasteriser.
        pub software: bool,
    }

    /// Real GPU execution of the demosaic, develop and detail stages.
    ///
    /// The engine is single-device and dispatch is serialised; it owns one
    /// `wgpu::Device` and one buffer pool. It never runs the CPU reference on
    /// the caller's behalf: a missing device is an error the caller resolves
    /// through `contracts::compute::select_backend`.
    pub struct CrossPlatformGpuEngine {
        device: wgpu::Device,
        queue: wgpu::Queue,
        info: wgpu::AdapterInfo,
        mode: ExecutionMode,
        demosaic: GpuPipeline,
        develop: GpuPipeline,
        detail_layout: wgpu::BindGroupLayout,
        detail_blur_h: wgpu::ComputePipeline,
        detail_blur_v: wgpu::ComputePipeline,
        detail_sharpen: wgpu::ComputePipeline,
        pool: Vec<PoolEntry>,
        allocations: u64,
        reuses: u64,
    }

    impl CrossPlatformGpuEngine {
        /// Native backends this module targets: Vulkan, Direct3D 12, Metal.
        pub const NATIVE_BACKENDS: wgpu::Backends =
            wgpu::Backends::VULKAN.union(wgpu::Backends::DX12).union(wgpu::Backends::METAL);

        /// Build an engine over the native backends, hardware adapters only.
        ///
        /// Under [`ExecutionMode::Strict`] every failure — no adapter, no
        /// device, no matching limits — is returned. There is no CPU
        /// substitution at any point.
        pub fn new(mode: ExecutionMode) -> Result<Self, GpuEngineError> {
            Self::new_with_options(mode, Self::NATIVE_BACKENDS, false)
        }

        /// [`Self::new`] with an explicit backend mask and an opt-in to software
        /// adapters (WARP / lavapipe), which are real devices but far slower.
        pub fn new_with_options(
            mode: ExecutionMode,
            backends: wgpu::Backends,
            allow_software: bool,
        ) -> Result<Self, GpuEngineError> {
            let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
            descriptor.backends = backends;
            let instance = wgpu::Instance::new(descriptor);

            let adapter = Self::request_adapter(&instance, backends, allow_software)?;
            let info = adapter.get_info();
            // Request exactly what the adapter advertises: the default 128 MiB
            // storage binding cap cannot hold a full-frame f32 plane.
            let limits = adapter.limits();
            let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("aurora-crossplatform-gpu"),
                required_features: wgpu::Features::empty(),
                required_limits: limits,
                ..Default::default()
            }))
            .map_err(|error| GpuEngineError::DeviceRequest(format!("{error:?}")))?;

            Self::from_device(device, queue, info, mode)
        }

        fn request_adapter(
            instance: &wgpu::Instance,
            backends: wgpu::Backends,
            allow_software: bool,
        ) -> Result<wgpu::Adapter, GpuEngineError> {
            let hardware = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            }));
            match hardware {
                Ok(adapter) => Ok(adapter),
                Err(first) if allow_software => {
                    block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::LowPower,
                        force_fallback_adapter: true,
                        compatible_surface: None,
                        ..Default::default()
                    }))
                    .map_err(|_| GpuEngineError::NoAdapter {
                        backends: format!("{backends:?} ({first:?})"),
                    })
                }
                Err(error) => Err(GpuEngineError::NoAdapter {
                    backends: format!("{backends:?} ({error:?})"),
                }),
            }
        }

        /// Probe the machine without keeping a device, for capability reports.
        ///
        /// Returns one [`BackendCapability`] per adapter that answered. An empty
        /// vector means no adapter was reported; it does not mean the CPU
        /// reference is unavailable.
        pub fn probe_capabilities(backends: wgpu::Backends) -> Vec<BackendCapability> {
            let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
            descriptor.backends = backends;
            let instance = wgpu::Instance::new(descriptor);
            block_on(instance.enumerate_adapters(backends))
                .into_iter()
                .filter_map(|adapter| {
                    let info = adapter.get_info();
                    let kind = map_wgpu_backend(info.backend)?;
                    Some(BackendCapability {
                        kind,
                        available: true,
                        device: Some(info.name).filter(|name| !name.is_empty()),
                        compute_capability: None,
                        precision: ContractPrecision::F32,
                    })
                })
                .collect()
        }

        /// Report the adapter this engine is running on.
        pub fn adapter_report(&self) -> GpuAdapterReport {
            GpuAdapterReport {
                device_name: self.info.name.clone(),
                driver: self.info.driver.clone(),
                wgpu_backend: format!("{:?}", self.info.backend),
                software: self.info.device_type == wgpu::DeviceType::Cpu,
            }
        }

        /// Policy this engine was built under.
        pub fn mode(&self) -> ExecutionMode {
            self.mode
        }

        /// The capability contract for the running adapter, when the wgpu
        /// backend maps onto one of the engine's backend kinds.
        pub fn capability(&self) -> Option<BackendCapability> {
            let kind = map_wgpu_backend(self.info.backend)?;
            Some(BackendCapability {
                kind,
                available: true,
                device: Some(self.info.name.clone()).filter(|name| !name.is_empty()),
                compute_capability: None,
                precision: ContractPrecision::F32,
            })
        }

        /// The AutoTune candidate descriptors this engine can execute.
        ///
        /// The status is deliberately [`VerificationStatus::Implemented`]: the
        /// parity tests exist and run where a device exists, but nothing here
        /// can promote a build to `DEVICE_OK` without that run.
        pub fn candidate_specs(&self) -> Vec<CandidateSpec> {
            let Some(backend) = map_portrait_backend(self.info.backend) else {
                return Vec::new();
            };
            vec![
                CandidateSpec {
                    id: "xp-demosaic-mhc5",
                    algorithm_id: "raw_demosaic",
                    backend,
                    kernel_entry: "demosaic_bayer",
                    precision: PortraitPrecision::Fp32,
                    params: DEMOSAIC_BOUNDS,
                    max_shared_mem_bytes: 0,
                    max_threads_per_group: 64,
                    status: VerificationStatus::Implemented,
                },
                CandidateSpec {
                    id: "xp-develop-fused",
                    algorithm_id: "develop_stage",
                    backend,
                    kernel_entry: "develop_stage",
                    precision: PortraitPrecision::Fp32,
                    params: DEVELOP_BOUNDS,
                    max_shared_mem_bytes: 0,
                    max_threads_per_group: 256,
                    status: VerificationStatus::Implemented,
                },
                CandidateSpec {
                    id: "xp-detail-histogram",
                    algorithm_id: "detail_histogram",
                    backend,
                    kernel_entry: "sharpen_hist",
                    precision: PortraitPrecision::Fp32,
                    params: DETAIL_BOUNDS,
                    max_shared_mem_bytes: 4096,
                    max_threads_per_group: 256,
                    status: VerificationStatus::Implemented,
                },
            ]
        }

        /// Buffer reuse so far.
        pub fn buffer_stats(&self) -> BufferReuseStats {
            BufferReuseStats {
                allocations: self.allocations,
                reuses: self.reuses,
                resident_buffers: self.pool.len(),
                resident_bytes: self.pool.iter().map(|entry| entry.size).sum(),
            }
        }

        fn from_device(
            device: wgpu::Device,
            queue: wgpu::Queue,
            info: wgpu::AdapterInfo,
            mode: ExecutionMode,
        ) -> Result<Self, GpuEngineError> {
            let demosaic_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora-raw-demosaic"),
                source: wgpu::ShaderSource::Wgsl(RAW_DEMOSAIC_WGSL.into()),
            });
            let develop_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora-develop"),
                source: wgpu::ShaderSource::Wgsl(DEVELOP_WGSL.into()),
            });
            let detail_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("aurora-detail-histogram"),
                source: wgpu::ShaderSource::Wgsl(DETAIL_WGSL.into()),
            });

            let demosaic = build_pipeline(
                &device,
                "aurora-demosaic",
                &demosaic_module,
                "demosaic_bayer",
                &[
                    (0, BindingKind::Uniform),
                    (1, BindingKind::StorageRead),
                    (2, BindingKind::StorageWrite),
                ],
            );
            let develop = build_pipeline(
                &device,
                "aurora-develop",
                &develop_module,
                "develop_stage",
                &[
                    (0, BindingKind::Uniform),
                    (1, BindingKind::StorageRead),
                    (2, BindingKind::StorageWrite),
                ],
            );
            // The detail module's three entry points share one layout because
            // every pass is handed the same bind group: unused bindings are
            // simply not read.
            let detail_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("aurora-detail-layout"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1, true),
                    storage_entry(2, false),
                    storage_entry(3, false),
                    storage_entry(4, false),
                    storage_entry(5, false),
                ],
            });
            let detail_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("aurora-detail-pl"),
                    bind_group_layouts: &[Some(&detail_layout)],
                    immediate_size: 0,
                });
            let detail_blur_h = compile_entry(
                &device,
                "aurora-detail-blur-h",
                &detail_pipeline_layout,
                &detail_module,
                "blur_h",
            );
            let detail_blur_v = compile_entry(
                &device,
                "aurora-detail-blur-v",
                &detail_pipeline_layout,
                &detail_module,
                "blur_v",
            );
            let detail_sharpen = compile_entry(
                &device,
                "aurora-detail-sharpen",
                &detail_pipeline_layout,
                &detail_module,
                "sharpen_hist",
            );

            Ok(Self {
                device,
                queue,
                info,
                mode,
                demosaic,
                develop,
                detail_layout,
                detail_blur_h,
                detail_blur_v,
                detail_sharpen,
                pool: Vec::new(),
                allocations: 0,
                reuses: 0,
            })
        }

        /// Demosaic a linear Bayer plane on the device.
        ///
        /// Same contract as [`super::demosaic_bayer_cpu_ref`]: one sample per
        /// photosite, RGBA f32 out.
        pub fn demosaic(
            &mut self,
            bayer: &[f32],
            width: u32,
            height: u32,
            pattern: CfaPattern,
        ) -> Result<Vec<f32>, GpuEngineError> {
            if width == 0 || height == 0 {
                return Err(GpuEngineError::InvalidInput("empty frame"));
            }
            let samples = width as usize * height as usize;
            if bayer.len() != samples {
                return Err(GpuEngineError::InvalidInput(
                    "bayer plane length does not match width * height",
                ));
            }
            let input_bytes = (samples * 4) as u64;
            let output_bytes = (samples * 16) as u64;

            let input = self.acquire(input_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
            let output = self.acquire(output_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
            let readback = self.acquire(output_bytes, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            let parameters = self.acquire(
                DEMOSAIC_UNIFORM_BYTES as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );

            self.queue.write_buffer(&input, 0, &f32s_to_bytes(bayer));
            self.queue
                .write_buffer(&parameters, 0, &demosaic_uniform(width, height, pattern));

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("aurora-demosaic-bg"),
                layout: &self.demosaic.layout,
                entries: &[
                    bind(0, &parameters),
                    bind(1, &input),
                    bind(2, &output),
                ],
            });

            let mut encoder = self.encoder("aurora-demosaic");
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("aurora-demosaic-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.demosaic.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
            self.submit(encoder)?;

            let bytes = self.read_back(&readback, output_bytes)?;
            self.release_all();
            Ok(bytes_to_f32s(&bytes))
        }

        /// Run the fused develop pass on the device.
        ///
        /// Same contract as [`super::develop_cpu_ref`].
        pub fn develop(
            &mut self,
            src: &[f32],
            params: &DevelopParams,
        ) -> Result<Vec<u8>, GpuEngineError> {
            if src.is_empty() || src.len() % 4 != 0 {
                return Err(GpuEngineError::InvalidInput(
                    "develop input must be a non-empty RGBA plane",
                ));
            }
            if !params.is_finite() {
                return Err(GpuEngineError::InvalidInput("develop parameters must be finite"));
            }
            let pixels = (src.len() / 4) as u64;
            let input_bytes = (src.len() * 4) as u64;
            let output_bytes = pixels * 4;

            let input = self.acquire(input_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
            let output = self.acquire(output_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
            let readback = self.acquire(output_bytes, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            let parameters = self.acquire(
                DEVELOP_UNIFORM_BYTES as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );

            self.queue.write_buffer(&input, 0, &f32s_to_bytes(src));
            self.queue
                .write_buffer(&parameters, 0, &develop_uniform(params));

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("aurora-develop-bg"),
                layout: &self.develop.layout,
                entries: &[
                    bind(0, &parameters),
                    bind(1, &input),
                    bind(2, &output),
                ],
            });

            let mut encoder = self.encoder("aurora-develop");
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("aurora-develop-pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.develop.pipeline);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(pixels.div_ceil(256) as u32, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &readback, 0, output_bytes);
            self.submit(encoder)?;

            let bytes = self.read_back(&readback, output_bytes)?;
            self.release_all();
            Ok(bytes)
        }

        /// Sharpen and histogram an RGBA8 frame on the device.
        ///
        /// Same contract as [`super::detail_histogram_cpu_ref`], including the
        /// 1024-entry bin layout.
        pub fn detail_histogram(
            &mut self,
            rgba: &[u8],
            width: u32,
            height: u32,
            amount: f32,
        ) -> Result<(Vec<u8>, Vec<u32>), GpuEngineError> {
            if width == 0 || height == 0 {
                return Err(GpuEngineError::InvalidInput("empty frame"));
            }
            let pixels = width as usize * height as usize;
            if rgba.len() != pixels * 4 {
                return Err(GpuEngineError::InvalidInput(
                    "detail input length does not match width * height * 4",
                ));
            }
            if !amount.is_finite() {
                return Err(GpuEngineError::InvalidInput("detail amount must be finite"));
            }
            let rgba_bytes = (pixels * 4) as u64;
            let plane_bytes = (pixels * 4) as u64;
            let hist_bytes = (HISTOGRAM_LEN * 4) as u64;

            let input = self.acquire(rgba_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
            let luma_h = self.acquire(plane_bytes, wgpu::BufferUsages::STORAGE);
            let luma_v = self.acquire(plane_bytes, wgpu::BufferUsages::STORAGE);
            let output = self.acquire(rgba_bytes, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
            let hist = self.acquire(
                hist_bytes,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            );
            let image_readback = self.acquire(rgba_bytes, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            let hist_readback = self.acquire(hist_bytes, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
            let parameters = self.acquire(
                DETAIL_UNIFORM_BYTES as u64,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            );

            self.queue.write_buffer(&input, 0, rgba);
            self.queue
                .write_buffer(&parameters, 0, &detail_uniform(width, height, amount));
            // The histogram accumulates with atomics, so it must start at zero.
            self.queue.write_buffer(&hist, 0, &vec![0u8; hist_bytes as usize]);

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("aurora-detail-bg"),
                layout: &self.detail_layout,
                entries: &[
                    bind(0, &parameters),
                    bind(1, &input),
                    bind(2, &luma_h),
                    bind(3, &luma_v),
                    bind(4, &output),
                    bind(5, &hist),
                ],
            });

            let mut encoder = self.encoder("aurora-detail");
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("aurora-detail-blur-h"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.detail_blur_h);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("aurora-detail-blur-v"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.detail_blur_v);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("aurora-detail-sharpen"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.detail_sharpen);
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups((pixels as u32).div_ceil(256), 1, 1);
            }
            encoder.copy_buffer_to_buffer(&output, 0, &image_readback, 0, rgba_bytes);
            encoder.copy_buffer_to_buffer(&hist, 0, &hist_readback, 0, hist_bytes);
            self.submit(encoder)?;

            let image = self.read_back(&image_readback, rgba_bytes)?;
            let bins = self.read_back(&hist_readback, hist_bytes)?;
            self.release_all();
            Ok((image, bytes_to_u32s(&bins)))
        }

        fn encoder(&self, label: &str) -> wgpu::CommandEncoder {
            self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some(label),
            })
        }

        fn submit(&self, encoder: wgpu::CommandEncoder) -> Result<(), GpuEngineError> {
            self.queue.submit(Some(encoder.finish()));
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|error| GpuEngineError::Execution(format!("poll: {error:?}")))?;
            Ok(())
        }

        fn read_back(&self, buffer: &wgpu::Buffer, len: u64) -> Result<Vec<u8>, GpuEngineError> {
            let slice = buffer.slice(0..len);
            let (sender, receiver) = mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result.is_ok());
            });
            self.device
                .poll(wgpu::PollType::wait_indefinitely())
                .map_err(|error| GpuEngineError::Execution(format!("poll: {error:?}")))?;
            let mapped = receiver
                .recv_timeout(READBACK_TIMEOUT)
                .map_err(|_| GpuEngineError::Execution("readback timed out".into()))?;
            if !mapped {
                return Err(GpuEngineError::Execution("buffer mapping failed".into()));
            }
            let view = slice
                .get_mapped_range()
                .map_err(|error| GpuEngineError::Execution(format!("map: {error}")))?;
            let bytes = view.to_vec();
            drop(view);
            buffer.unmap();
            Ok(bytes)
        }

        fn acquire(&mut self, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
            let size = size.max(4).div_ceil(4) * 4;
            if let Some(entry) = self
                .pool
                .iter_mut()
                .find(|entry| !entry.busy && entry.usage == usage && entry.size >= size)
            {
                entry.busy = true;
                self.reuses += 1;
                return entry.buffer.clone();
            }
            let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("aurora-crossplatform"),
                size,
                usage,
                mapped_at_creation: false,
            });
            self.allocations += 1;
            self.pool.push(PoolEntry {
                buffer: buffer.clone(),
                size,
                usage,
                busy: true,
            });
            buffer
        }

        fn release_all(&mut self) {
            for entry in &mut self.pool {
                entry.busy = false;
            }
        }
    }

    #[derive(Clone, Copy)]
    enum BindingKind {
        Uniform,
        StorageRead,
        StorageWrite,
    }

    fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn bind(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
        wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: None,
            }),
        }
    }

    fn build_pipeline(
        device: &wgpu::Device,
        label: &str,
        module: &wgpu::ShaderModule,
        entry_point: &str,
        bindings: &[(u32, BindingKind)],
    ) -> GpuPipeline {
        let entries: Vec<wgpu::BindGroupLayoutEntry> = bindings
            .iter()
            .map(|(index, kind)| match kind {
                BindingKind::Uniform => uniform_entry(*index),
                BindingKind::StorageRead => storage_entry(*index, true),
                BindingKind::StorageWrite => storage_entry(*index, false),
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(label),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = compile_entry(device, label, &pipeline_layout, module, entry_point);
        GpuPipeline { layout, pipeline }
    }

    fn compile_entry(
        device: &wgpu::Device,
        label: &str,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        entry_point: &str,
    ) -> wgpu::ComputePipeline {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    const DEMOSAIC_BOUNDS: &[ParamBound] = &[ParamBound::DiscreteU32 {
        name: "workgroup",
        values: &[64],
    }];
    const DEVELOP_BOUNDS: &[ParamBound] = &[ParamBound::DiscreteU32 {
        name: "workgroup",
        values: &[256],
    }];
    const DETAIL_BOUNDS: &[ParamBound] = &[
        ParamBound::DiscreteU32 {
            name: "workgroup",
            values: &[256],
        },
        ParamBound::U32Range {
            name: "histogram_bins",
            min: 256,
            max: 1024,
            step: 256,
        },
    ];

    fn map_wgpu_backend(backend: wgpu::Backend) -> Option<BackendKind> {
        match backend {
            wgpu::Backend::Vulkan => Some(BackendKind::Vulkan),
            wgpu::Backend::Metal => Some(BackendKind::Metal),
            wgpu::Backend::Dx12 => Some(BackendKind::Dx12),
            _ => None,
        }
    }

    fn map_portrait_backend(backend: wgpu::Backend) -> Option<PortraitBackend> {
        match backend {
            wgpu::Backend::Vulkan => Some(PortraitBackend::Vulkan),
            wgpu::Backend::Metal => Some(PortraitBackend::Metal),
            wgpu::Backend::Dx12 => Some(PortraitBackend::Dx12),
            _ => None,
        }
    }

    // --- parameter blocks, byte-identical to the shader structs -------------
    //
    // Each mirror struct exists only so the size assertion fails the build if
    // the hand-written byte offsets below ever disagree with the layout the
    // shaders declare. They are never instantiated.

    #[allow(dead_code)]
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct DemosaicUniform {
        size: [u32; 2],
        pattern: u32,
        pad: u32,
    }

    const _: () = assert!(std::mem::size_of::<DemosaicUniform>() == DEMOSAIC_UNIFORM_BYTES);

    fn demosaic_uniform(width: u32, height: u32, pattern: CfaPattern) -> [u8; DEMOSAIC_UNIFORM_BYTES] {
        let mut out = [0u8; DEMOSAIC_UNIFORM_BYTES];
        put_u32(&mut out, 0, width);
        put_u32(&mut out, 4, height);
        put_u32(&mut out, 8, pattern.as_u32());
        out
    }

    #[allow(dead_code)]
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct DevelopUniform {
        wb: [f32; 4],
        exposure: f32,
        highlights: f32,
        shadows: f32,
        contrast: f32,
        saturation: f32,
        pad: [f32; 3],
    }

    const _: () = assert!(std::mem::size_of::<DevelopUniform>() == DEVELOP_UNIFORM_BYTES);

    fn develop_uniform(params: &DevelopParams) -> [u8; DEVELOP_UNIFORM_BYTES] {
        let (highlights, shadows) = (params.highlights.clamp(-1.0, 1.0), params.shadows.clamp(-1.0, 1.0));
        let mut out = [0u8; DEVELOP_UNIFORM_BYTES];
        for (index, gain) in params.wb.iter().enumerate() {
            put_f32(&mut out, index * 4, *gain);
        }
        put_f32(&mut out, 16, params.exposure_scale());
        put_f32(&mut out, 20, highlights);
        put_f32(&mut out, 24, shadows);
        put_f32(&mut out, 28, params.contrast);
        put_f32(&mut out, 32, params.saturation);
        out
    }

    #[allow(dead_code)]
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct DetailUniform {
        size: [u32; 2],
        amount: f32,
        pad: f32,
    }

    const _: () = assert!(std::mem::size_of::<DetailUniform>() == DETAIL_UNIFORM_BYTES);

    fn detail_uniform(width: u32, height: u32, amount: f32) -> [u8; DETAIL_UNIFORM_BYTES] {
        let mut out = [0u8; DETAIL_UNIFORM_BYTES];
        put_u32(&mut out, 0, width);
        put_u32(&mut out, 4, height);
        put_f32(&mut out, 8, amount);
        out
    }

    fn put_u32(out: &mut [u8], offset: usize, value: u32) {
        out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn put_f32(out: &mut [u8], offset: usize, value: f32) {
        out[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    // --- host <-> device element conversion ---------------------------------

    fn f32s_to_bytes(values: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 4);
        for value in values {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    fn bytes_to_f32s(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    fn bytes_to_u32s(bytes: &[u8]) -> Vec<u32> {
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }
}

#[cfg(feature = "wgpu-backend")]
pub use gpu::{
    BufferReuseStats, CrossPlatformGpuEngine, GpuAdapterReport,
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic LCG so every fixture is reproducible without external data.
    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32) / ((1u64 << 24) as f32)
        }
    }

    #[test]
    fn cfa_phases_match_the_shader_table() {
        // Packed nibbles and per-pixel codes are the shader's lookup, and the
        // green site's "which neighbour is red" decision depends on it.
        for (pattern, origin, right, down, diagonal) in [
            (CfaPattern::Rggb, 0, 1, 1, 2),
            (CfaPattern::Bggr, 2, 1, 1, 0),
            (CfaPattern::Grbg, 1, 0, 2, 1),
            (CfaPattern::Gbrg, 1, 2, 0, 1),
        ] {
            assert_eq!(pattern.code_at(0, 0), origin, "{pattern:?} origin");
            assert_eq!(pattern.code_at(1, 0), right, "{pattern:?} right");
            assert_eq!(pattern.code_at(0, 1), down, "{pattern:?} down");
            assert_eq!(pattern.code_at(1, 1), diagonal, "{pattern:?} diagonal");
            assert_eq!(CfaPattern::from_u32(pattern.as_u32()), Some(pattern));
        }
        assert_eq!(CfaPattern::from_u32(4), None);
    }

    #[test]
    fn mirror_keeps_the_5x5_kernel_inside_the_frame() {
        assert_eq!(mirror_index(-1, 4), 1);
        assert_eq!(mirror_index(-2, 4), 2);
        assert_eq!(mirror_index(4, 4), 2);
        assert_eq!(mirror_index(5, 4), 1);
        assert_eq!(mirror_index(0, 1), 0);
        assert_eq!(mirror_index(-2, 1), 0);
        assert_eq!(mirror_index(2, 1), 0);
        assert_eq!(mirror_index(0, 2), 0);
        assert_eq!(mirror_index(2, 2), 0);
        assert_eq!(mirror_index(3, 2), 0);
    }

    #[test]
    fn demosaic_preserves_a_flat_field_for_every_phase() {
        // Every MHC filter sums to 8/8, so a constant plane must come back
        // unchanged in all three channels at every pixel, edges included.
        for pattern in [CfaPattern::Rggb, CfaPattern::Bggr, CfaPattern::Grbg, CfaPattern::Gbrg] {
            for (w, h) in [(1usize, 1usize), (1, 7), (9, 1), (5, 5), (16, 11)] {
                let plane = vec![0.42f32; w * h];
                let rgba = demosaic_bayer_cpu_ref(&plane, w, h, pattern);
                assert_eq!(rgba.len(), w * h * 4);
                for (i, pixel) in rgba.chunks_exact(4).enumerate() {
                    for channel in 0..3 {
                        assert!(
                            (pixel[channel] - 0.42).abs() < 1e-5,
                            "{pattern:?} {w}x{h} pixel {i} channel {channel} = {}",
                            pixel[channel]
                        );
                    }
                    assert_eq!(pixel[3], 1.0);
                }
            }
        }
    }

    #[test]
    fn demosaic_recovers_colours_at_a_smooth_colour_patch() {
        // A plane whose R/G/B photosites carry 0.25/0.50/0.75 must come back as
        // roughly those values away from the border.
        let (w, h) = (16usize, 16usize);
        for pattern in [CfaPattern::Rggb, CfaPattern::Bggr, CfaPattern::Grbg, CfaPattern::Gbrg] {
            let mut plane = vec![0.0f32; w * h];
            for y in 0..h {
                for x in 0..w {
                    plane[y * w + x] = match pattern.code_at(x as u32, y as u32) {
                        0 => 0.25,
                        2 => 0.75,
                        _ => 0.50,
                    };
                }
            }
            let rgba = demosaic_bayer_cpu_ref(&plane, w, h, pattern);
            for y in 2..h - 2 {
                for x in 2..w - 2 {
                    let p = (y * w + x) * 4;
                    assert!((rgba[p] - 0.25).abs() < 1e-5, "{pattern:?} R at {x},{y}");
                    assert!((rgba[p + 1] - 0.50).abs() < 1e-5, "{pattern:?} G at {x},{y}");
                    assert!((rgba[p + 2] - 0.75).abs() < 1e-5, "{pattern:?} B at {x},{y}");
                }
            }
        }
    }

    #[test]
    fn demosaic_edge_pixels_mirror_instead_of_reading_black() {
        // A one-pixel bright border on a black field: mirrored borders make the
        // corner average the border with itself, never the frame exterior.
        let (w, h) = (8usize, 8usize);
        let mut plane = vec![0.0f32; w * h];
        for x in 0..w {
            plane[x] = 1.0;
        }
        let rgba = demosaic_bayer_cpu_ref(&plane, w, h, CfaPattern::Rggb);
        for (i, value) in rgba.iter().enumerate() {
            assert!(value.is_finite(), "value {i} is not finite");
            // The largest absolute weight sum of a 5x5 MHC filter is 2.5, so a
            // [0, 1] input cannot leave [-2.5, 2.5].
            assert!((-2.5..=2.5).contains(value), "value {i} = {value} is out of range");
        }
        // The second row still sees the bright row, so its green is not black.
        let second_row = (w + 3) * 4;
        assert!(rgba[second_row + 1] > 0.0);
    }

    #[test]
    fn develop_neutral_parameters_are_just_the_transfer_curve() {
        let params = DevelopParams::default();
        let src = [0.0f32, 0.0031308, 0.5, 1.0, -0.2, 2.0, 0.18, 0.5];
        let out = develop_cpu_ref(&src, &params);
        assert_eq!(out.len(), 8);
        assert_eq!(out[0], 0); // black stays black
        assert_eq!(out[3], quantize_u8(1.0));
        assert_eq!(out[7], quantize_u8(0.5)); // alpha carried through
        // Highlight headroom above 1.0 clips, as linear_to_srgb + clamp does.
        assert_eq!(out[5], 255);
        // The mid grey lands where sRGB puts 0.5.
        assert_eq!(out[2], quantize_u8(linear_to_srgb(0.5)));
        assert_eq!(out[6], quantize_u8(linear_to_srgb(0.18)));
    }

    #[test]
    fn develop_alpha_is_never_touched_by_colour_terms() {
        let params = DevelopParams {
            wb: [1.7, 1.0, 0.6],
            exposure_ev: 0.8,
            highlights: -0.4,
            shadows: 0.3,
            contrast: 0.5,
            saturation: 1.4,
        };
        let mut src = Vec::new();
        for alpha in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            src.extend_from_slice(&[0.2, 0.35, 0.1, alpha]);
        }
        let out = develop_cpu_ref(&src, &params);
        for (index, alpha) in [0.0f32, 0.25, 0.5, 0.75, 1.0].iter().enumerate() {
            assert_eq!(out[index * 4 + 3], quantize_u8(*alpha));
        }
        // Far more than a byte of colour must have actually moved.
        assert_ne!(out[0], quantize_u8(linear_to_srgb(0.2)));
    }

    #[test]
    fn develop_toning_is_channel_neutral() {
        // One luma-derived gain for all three channels keeps a neutral grey
        // neutral, which is the whole point of the C1 tone shape.
        let params = DevelopParams {
            highlights: -0.6,
            shadows: 0.5,
            ..Default::default()
        };
        let out = develop_cpu_ref(&[0.18, 0.18, 0.18, 1.0], &params);
        assert_eq!(out[0], out[1]);
        assert_eq!(out[1], out[2]);
    }

    #[test]
    fn histogram_bins_follow_the_documented_layout() {
        let rgba = [
            0u8, 0, 0, 255, // black
            255, 255, 255, 0, // white, alpha 0 is still counted
            10, 20, 30, 128,
        ];
        let hist = histogram_cpu_ref(&rgba);
        assert_eq!(hist.len(), HISTOGRAM_LEN);
        assert_eq!(hist[0], 1);
        assert_eq!(hist[255], 1);
        assert_eq!(hist[256], 1);
        assert_eq!(hist[256 + 20], 1);
        assert_eq!(hist[512 + 30], 1);
        assert_eq!(hist[768], 1); // black
        assert_eq!(hist[768 + 255], 1); // white
        let luma = luma8(10, 20, 30) as usize;
        assert_eq!(hist[768 + luma], 1);
        // Every pixel contributes exactly one count per channel.
        let total: u32 = hist.iter().sum();
        assert_eq!(total, 3 * 4);
    }

    #[test]
    fn detail_with_zero_amount_is_the_identity() {
        let (w, h) = (13usize, 9usize);
        let mut random = lcg(0x1234_5678_9abc_def0);
        let rgba: Vec<u8> = (0..w * h * 4).map(|_| (random() * 255.0) as u8).collect();
        let (out, hist) = detail_histogram_cpu_ref(&rgba, w, h, 0.0);
        assert_eq!(out, rgba, "zero amount must not rewrite pixels");
        assert_eq!(hist, histogram_cpu_ref(&rgba));
    }

    #[test]
    fn detail_boost_raises_local_contrast_and_keeps_alpha() {
        // A dark/light checker under a flat surround: the boost must widen the
        // step and leave alpha alone.
        let (w, h) = (9usize, 9usize);
        let mut rgba = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let p = (y * w + x) * 4;
                let value = if (x / 3 + y / 3) % 2 == 0 { 90 } else { 150 };
                rgba[p] = value;
                rgba[p + 1] = value;
                rgba[p + 2] = value;
                rgba[p + 3] = 200;
            }
        }
        let (out, hist) = detail_histogram_cpu_ref(&rgba, w, h, 1.0);
        // The checker changes value between x = 5 (90) and x = 6 (150).
        let step_in = rgba[(4 * w + 5) * 4] as i32 - rgba[(4 * w + 6) * 4] as i32;
        let step_out = out[(4 * w + 5) * 4] as i32 - out[(4 * w + 6) * 4] as i32;
        assert!(
            step_out.abs() >= step_in.abs(),
            "boost must not flatten local contrast ({step_in} -> {step_out})"
        );
        for (index, px) in out.chunks_exact(4).enumerate() {
            assert_eq!(px[3], 200, "alpha changed at pixel {index}");
        }
        assert_eq!(hist.iter().sum::<u32>(), 4 * (w * h) as u32);
    }
}

#[cfg(all(test, feature = "wgpu-backend"))]
mod gpu_tests {
    use super::*;
    use aurora_portrait::autotune::ExecutionMode;

    /// Build an engine, or report that this machine has none. A skipped test is
    /// not a pass; the eprintln makes that visible in the log.
    fn engine() -> Option<CrossPlatformGpuEngine> {
        match CrossPlatformGpuEngine::new(ExecutionMode::Strict) {
            Ok(engine) => {
                let report = engine.adapter_report();
                eprintln!(
                    "cross-platform GPU: {} via {}{}",
                    report.device_name,
                    report.wgpu_backend,
                    if report.software { " (software)" } else { "" }
                );
                Some(engine)
            }
            Err(error) => {
                eprintln!("SKIP: no cross-platform GPU device ({error})");
                None
            }
        }
    }

    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut state = seed;
        move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32) / ((1u64 << 24) as f32)
        }
    }

    fn bayer_fixture(w: usize, h: usize, pattern: CfaPattern) -> Vec<f32> {
        // A smooth two-axis gradient plus a little noise: enough structure for
        // every kernel term to contribute, no hard edges to mask an error.
        let mut random = lcg(0x0bad_c0de_1234_5678);
        let mut plane = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let base = 0.15 + 0.6 * (x as f32 / w as f32) + 0.2 * (y as f32 / h as f32);
                let tint = match pattern.code_at(x as u32, y as u32) {
                    0 => 0.85,
                    2 => 1.15,
                    _ => 1.0,
                };
                plane[y * w + x] = (base * tint + 0.02 * (random() - 0.5)).clamp(0.0, 1.5);
            }
        }
        plane
    }

    #[test]
    fn gpu_demosaic_matches_the_cpu_reference() {
        let Some(mut engine) = engine() else { return };
        for pattern in [CfaPattern::Rggb, CfaPattern::Bggr, CfaPattern::Grbg, CfaPattern::Gbrg] {
            for (w, h) in [(1usize, 1usize), (9, 1), (1, 9), (37, 23), (64, 33)] {
                let plane = bayer_fixture(w, h, pattern);
                let expected = demosaic_bayer_cpu_ref(&plane, w, h, pattern);
                let actual = engine
                    .demosaic(&plane, w as u32, h as u32, pattern)
                    .expect("gpu demosaic");
                assert_eq!(actual.len(), expected.len());
                for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
                    assert!(
                        (a - e).abs() < 1e-4,
                        "{pattern:?} {w}x{h} value {index}: gpu {a} vs cpu {e}"
                    );
                }
            }
        }
        // A second identical call must be served from the pool, not reallocated.
        let stats = engine.buffer_stats();
        assert!(stats.reuses > 0, "buffer pool never reused a buffer: {stats:?}");
    }

    #[test]
    fn gpu_develop_matches_the_cpu_reference_within_one_lsb() {
        let Some(mut engine) = engine() else { return };
        let params = DevelopParams {
            wb: [1.65, 1.0, 0.72],
            exposure_ev: 0.35,
            highlights: -0.35,
            shadows: 0.25,
            contrast: 0.4,
            saturation: 1.25,
        };
        let mut random = lcg(0xdead_beef_0bad_f00d);
        let mut src = Vec::new();
        for _ in 0..(48 * 27) {
            let a = random() * 1.2;
            src.extend_from_slice(&[a, a * 0.8 + 0.02, a * 0.55 + 0.01, random()]);
        }
        let expected = develop_cpu_ref(&src, &params);
        let actual = engine.develop(&src, &params).expect("gpu develop");
        assert_eq!(actual.len(), expected.len());
        for (index, (a, e)) in actual.iter().zip(&expected).enumerate() {
            let delta = (*a as i32 - *e as i32).abs();
            assert!(delta <= 1, "byte {index}: gpu {a} vs cpu {e} differ by {delta}");
        }
        // Alpha must be byte-identical, not merely within one LSB.
        for (index, px) in actual.chunks_exact(4).enumerate() {
            assert_eq!(px[3], expected[index * 4 + 3], "alpha at pixel {index}");
        }
    }

    #[test]
    fn gpu_histogram_matches_the_cpu_binning_exactly() {
        let Some(mut engine) = engine() else { return };
        let (w, h) = (41usize, 29usize);
        let mut random = lcg(0x5eed_5eed_5eed_5eed);
        let mut rgba = vec![0u8; w * h * 4];
        for (index, px) in rgba.chunks_exact_mut(4).enumerate() {
            let value = (random() * 255.0) as u8;
            px[0] = value;
            px[1] = ((index * 7) % 256) as u8;
            px[2] = ((index * 29 + 11) % 256) as u8;
            px[3] = ((index * 3) % 256) as u8;
        }

        let (cpu_image, _) = detail_histogram_cpu_ref(&rgba, w, h, 0.8);
        let (gpu_image, gpu_hist) = engine
            .detail_histogram(&rgba, w as u32, h as u32, 0.8)
            .expect("gpu detail");

        // Zero difference in every one of the 1024 bins: the shader's bucketing
        // and the workgroup-atomic reduction must reproduce the CPU definition
        // exactly for the bytes the device produced. This is the claim the CPU
        // histogram cannot make for different bytes, which is why the sharpened
        // image is checked separately below.
        assert_eq!(
            gpu_hist,
            histogram_cpu_ref(&gpu_image),
            "atomicAdd histogram differs from the CPU binning of the same bytes"
        );
        assert_eq!(gpu_hist.len(), HISTOGRAM_LEN);
        assert_eq!(
            gpu_hist.iter().sum::<u32>(),
            4 * (w * h) as u32,
            "every pixel must contribute exactly one count per channel"
        );

        // The sharpened bytes are one fused-multiply-add away from the CPU at
        // worst; anything larger is a real disagreement.
        assert_eq!(gpu_image.len(), cpu_image.len());
        for (index, (a, e)) in gpu_image.iter().zip(&cpu_image).enumerate() {
            let delta = (*a as i32 - *e as i32).abs();
            assert!(delta <= 1, "byte {index}: gpu {a} vs cpu {e} differ by {delta}");
        }
    }

    #[test]
    fn strict_mode_refuses_instead_of_running_the_cpu_reference() {
        // A backend mask with no compiled backend cannot produce an adapter, so
        // a Strict build must fail rather than hand back CPU results.
        let result = CrossPlatformGpuEngine::new_with_options(
            ExecutionMode::Strict,
            wgpu::Backends::NOOP,
            false,
        );
        match result {
            Err(GpuEngineError::NoAdapter { .. }) => {}
            Err(other) => panic!("expected NoAdapter, got {other:?}"),
            Ok(_) => panic!("a Strict engine must not be built without an adapter"),
        }
    }

    #[test]
    fn candidate_specs_describe_the_three_kernels() {
        let Some(engine) = engine() else { return };
        let specs = engine.candidate_specs();
        assert_eq!(specs.len(), 3);
        let entries: Vec<&str> = specs.iter().map(|spec| spec.kernel_entry).collect();
        assert!(entries.contains(&"demosaic_bayer"));
        assert!(entries.contains(&"develop_stage"));
        assert!(entries.contains(&"sharpen_hist"));
        for spec in &specs {
            assert_eq!(spec.precision, aurora_portrait::device::Precision::Fp32);
            assert!(spec.fits_budget(&aurora_portrait::autotune::ResourceBudget::conservative_mobile()));
        }
        assert!(engine.capability().is_some());
        assert_eq!(engine.mode(), ExecutionMode::Strict);
    }
}
