//! C1 blemish gain map: 16-bit linear frame -> 450 map -> upsample -> multiply.
//!
//! Producer recovered from `CImgCore::CalculateBlemishRemovalMap`
//! (`0x153EAE0`), consumer from `CImgOpBlemishRemoval::RemoveBlemishes`
//! (`0x16B2DD0`).  The consumer loop at `0x16B36F0..0x16B3994` bilinearly
//! samples the float4 map, uses lane 3 as confidence, multiplies the amount and
//! protection mask, then writes `src_c * (gain_c * w + 1 - w)`.

use aurora_portrait::Refusal;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU8, Ordering};

pub const MAP_LONG_SIDE: f32 = 450.0;
pub const TILE_STEP: u32 = 214;
pub const TILE_INPUT: u32 = 278;
pub const HALO: u32 = 32;
pub const GAIN_FLOOR: f32 = 1.0 / 256.0;
pub const GAIN_MIN: f32 = 0.5;
pub const GAIN_MAX: f32 = 2.5;
pub const CONF_FLOOR: f32 = 1.0e-6;
pub const LUMA_R: f32 = 0.298_999_995;
pub const LUMA_G: f32 = 0.587_000_012;
pub const LUMA_B: f32 = 0.114;

/// `AURORA_BLEMISH_GAINMAP=1` enables the 16-bit gain-map path.  Any other
/// value (including unset) keeps the existing bytes untouched.
pub fn enabled() -> bool {
    std::env::var("AURORA_BLEMISH_GAINMAP").ok().as_deref() == Some("1")
}

/// `AURORA_BLEMISH_GPU=0` forces the CPU bytes; `=1` forces the CUDA host
/// bridge. Unset is auto and stays on the CPU lane: the measured pageable
/// bridge is 42-53 ms against 22-43 ms CPU at 24 MP, and it is slower per
/// megapixel at preview sizes too. Callers that already hold pinned or
/// device-resident frames use [`crate::blemish_gainmap_cuda`] directly.
pub fn gpu_enabled() -> bool {
    std::env::var("AURORA_BLEMISH_GPU").ok().as_deref() != Some("0")
}

pub fn gpu_forced() -> bool {
    std::env::var("AURORA_BLEMISH_GPU").ok().as_deref() == Some("1")
}

/// No automatic crossover exists for the pageable bridge on this host.
pub const GPU_AUTO_MAX_PIXELS: usize = 0;

pub fn gpu_auto(_pixels: usize) -> bool {
    gpu_enabled() && gpu_forced()
}

/// 0 = not reported, 1 = CUDA active, 2 = CPU fallback, 3 = auto skipped.
static GPU_STATE: AtomicU8 = AtomicU8::new(0);

fn report_gpu(state: u8, message: impl FnOnce() -> String) {
    if GPU_STATE
        .compare_exchange(0, state, Ordering::Relaxed, Ordering::Relaxed)
        .is_ok()
    {
        eprintln!("{}", message());
    }
}

/// Shipped model path. `AURORA_BLEMISH_MODEL` overrides it for tests/fixtures.
pub fn model_path() -> PathBuf {
    crate::bundled_resources::blemish_model_path()
}

pub fn model_available_at(path: &Path) -> bool {
    path.is_file()
}

pub fn model_available() -> bool {
    model_available_at(&model_path())
}

/// Existing blemish parameter, in C1's 0..=100 slider units.  The PF table
/// maps `IB_F_BLEMISH_REMOVAL_DEFAULT` to this key; absent means C1 default 0.
pub fn amount_from_adjustments(adjust: &crate::AdjustmentSet) -> f32 {
    (adjust.extra.get("BlemishRemovalAmount").copied().unwrap_or(0.0) / 100.0)
        .clamp(0.0, 1.0)
}

/// `ZRETOUCHING...BLEMISHREMOVALAMOUNT` is parsed by `c1_retouch` as 0..=100.
pub fn amount_from_retouch_layer(amount_percent: Option<f32>) -> f32 {
    (amount_percent.unwrap_or(0.0) / 100.0).clamp(0.0, 1.0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    pub x: u32,
    pub y: u32,
    pub valid_w: u32,
    pub valid_h: u32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GainMap {
    pub width: u32,
    pub height: u32,
    /// RGB = per-channel gain, A = confidence.
    pub rgba: Vec<f32>,
}

pub fn map_dims(src_w: u32, src_h: u32) -> (u32, u32) {
    let long = src_w.max(src_h) as f32;
    let scale = MAP_LONG_SIDE / long.max(1.0);
    (
        ((src_w as f32) * scale).round().max(1.0) as u32,
        ((src_h as f32) * scale).round().max(1.0) as u32,
    )
}

pub fn tiles(width: u32, height: u32) -> Vec<Tile> {
    let nx = width.div_ceil(TILE_STEP);
    let ny = height.div_ceil(TILE_STEP);
    let max_x = width.saturating_sub(TILE_STEP);
    let max_y = height.saturating_sub(TILE_STEP);
    let mut out = Vec::with_capacity((nx * ny) as usize);
    for ty in 0..ny {
        for tx in 0..nx {
            let x = (tx * TILE_STEP).min(max_x);
            let y = (ty * TILE_STEP).min(max_y);
            out.push(Tile {
                x,
                y,
                valid_w: TILE_STEP.min(width - x),
                valid_h: TILE_STEP.min(height - y),
            });
        }
    }
    out
}

fn clamp_i(v: i32, lo: i32, hi: i32) -> i32 {
    v.max(lo).min(hi)
}

fn model_input(src: &[f32], width: u32, height: u32, tile: Tile) -> Vec<f32> {
    let mut out = vec![0.0f32; (TILE_INPUT * TILE_INPUT * 3) as usize];
    let x0 = tile.x as i32 - HALO as i32;
    let y0 = tile.y as i32 - HALO as i32;
    for j in 0..TILE_INPUT {
        let sy = clamp_i(y0 + j as i32, 0, height as i32 - 1) as usize;
        for i in 0..TILE_INPUT {
            let sx = clamp_i(x0 + i as i32, 0, width as i32 - 1) as usize;
            let si = (sy * width as usize + sx) * 3;
            let di = ((j * TILE_INPUT + i) * 3) as usize;
            out[di..di + 3].copy_from_slice(&src[si..si + 3]);
        }
    }
    out
}

/// C1 `Super_16u_C3R` (0x1029710): two-pass exact area coverage.
pub fn downsample_super_16u(src: &[u16], sw: u32, sh: u32, dw: u32, dh: u32) -> Vec<u16> {
    let mut out = vec![0u16; dw as usize * dh as usize * 3];
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return out;
    }
    if sw <= dw && sh <= dh {
        for dy in 0..dh {
            let sy = ((dy as u64 * sh as u64) / dh as u64).min(sh as u64 - 1) as usize;
            for dx in 0..dw {
                let sx = ((dx as u64 * sw as u64) / dw as u64).min(sw as u64 - 1) as usize;
                let si = (sy * sw as usize + sx) * 3;
                let di = ((dy as usize * dw as usize) + dx as usize) * 3;
                out[di..di + 3].copy_from_slice(&src[si..si + 3]);
            }
        }
        return out;
    }
    let step_x = sw as f64 / dw as f64;
    let step_y = sh as f64 / dh as f64;
    let norm = (step_x * step_y) as f32;
    let mut row = vec![0.0f32; sw as usize * 3];
    for dy in 0..dh {
        let y0 = dy as f64 * step_y;
        let y1 = y0 + step_y;
        row.fill(0.0);
        for j in y0.floor() as i64..y1.ceil() as i64 {
            if j < 0 || j >= sh as i64 {
                continue;
            }
            let wy = (y1.min((j + 1) as f64) - y0.max(j as f64)).max(0.0);
            if wy == 0.0 {
                continue;
            }
            let w = wy as f32;
            let base = j as usize * sw as usize * 3;
            for x in 0..sw as usize * 3 {
                row[x] += src[base + x] as f32 * w;
            }
        }
        for dx in 0..dw {
            let x0 = dx as f64 * step_x;
            let x1 = x0 + step_x;
            for c in 0..3 {
                let mut acc = 0.0f32;
                for i in x0.floor() as i64..x1.ceil() as i64 {
                    if i < 0 || i >= sw as i64 {
                        continue;
                    }
                    let wx = (x1.min((i + 1) as f64) - x0.max(i as f64)).max(0.0);
                    if wx != 0.0 {
                        acc += row[i as usize * 3 + c] * (wx as f32);
                    }
                }
                out[((dy as usize * dw as usize) + dx as usize) * 3 + c] =
                    (acc / norm + 0.5).clamp(0.0, 65535.0) as u16;
            }
        }
    }
    out
}

pub fn auto_normalise_scale(rgb: &[f32]) -> Option<f32> {
    if rgb.is_empty() {
        return None;
    }
    let mut sum = 0.0f64;
    for p in rgb.chunks_exact(3) {
        sum += (p[0] + 2.0 * p[1] + p[2]) as f64;
    }
    let avg = sum / (rgb.len() as f64 / 3.0 * 4.0);
    (avg > 0.0).then_some((0.5 / avg) as f32)
}

fn luma_match(model: &[f32], src: &[f32], out: &mut [f32]) {
    for (i, (m, s)) in model.chunks_exact(3).zip(src.chunks_exact(3)).enumerate() {
        let lm = LUMA_R * m[0] + LUMA_G * m[1] + LUMA_B * m[2];
        let ls = LUMA_R * s[0] + LUMA_G * s[1] + LUMA_B * s[2];
        let k = if lm > 0.0 { ls / lm } else { 1.0 };
        for c in 0..3 {
            out[i * 3 + c] = m[c] * k;
        }
    }
}

fn gain_rgb(repaired: &[f32], source: &[f32], mask: &[u8], out: &mut [f32]) {
    for (i, ((r, s), &m)) in repaired
        .chunks_exact(3)
        .zip(source.chunks_exact(3))
        .zip(mask)
        .enumerate()
    {
        let w = m as f32 / 255.0;
        for c in 0..3 {
            let a = r[c].max(GAIN_FLOOR);
            let b = s[c].max(GAIN_FLOOR);
            let g = ((a / b) * (a / b)).clamp(GAIN_MIN, GAIN_MAX);
            out[i * 3 + c] = 1.0 - w * (1.0 - g);
        }
    }
}

fn confidence(gain: &[f32], width: u32, height: u32, out: &mut [f32]) {
    let n = width as usize * height as usize;
    let mut a = vec![0.0f32; n];
    for (i, p) in gain.chunks_exact(3).enumerate() {
        a[i] = p[0].abs().max(p[1].abs()).max(p[2].abs()).max(CONF_FLOOR);
    }
    let mut dil = vec![0.0f32; n];
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut m = CONF_FLOOR;
            for dy in -3..=3 {
                let sy = clamp_i(y + dy, 0, height as i32 - 1) as usize;
                for dx in -3..=3 {
                    let sx = clamp_i(x + dx, 0, width as i32 - 1) as usize;
                    m = m.max(a[sy * width as usize + sx]);
                }
            }
            dil[y as usize * width as usize + x as usize] = m;
        }
    }
    for y in 0..height as i32 {
        for x in 0..width as i32 {
            let mut sum = 0.0f32;
            for dy in -1..=1 {
                let sy = clamp_i(y + dy, 0, height as i32 - 1) as usize;
                for dx in -1..=1 {
                    let sx = clamp_i(x + dx, 0, width as i32 - 1) as usize;
                    sum += dil[sy * width as usize + sx];
                }
            }
            out[y as usize * width as usize + x as usize] = sum / 9.0;
        }
    }
}

pub trait ModelRunner {
    fn run(&mut self, input: &[f32], out: &mut [f32]) -> Result<(), String>;
}

pub use ModelRunner as GainMapModelRunner;

/// Production ONNX runner.  It goes through the existing `aimask` runtime, so
/// provider order is `TensorRT -> CUDA -> DirectML -> CPU` with
/// `AURORA_AI_PROVIDER` able to pin one EP.  The selected provider is printed
/// once at session open and exposed by [`Self::provider`].
pub struct OrtGainMapRunner {
    session: crate::aimask::Session,
    provider: &'static str,
}

impl OrtGainMapRunner {
    pub fn open() -> Result<Self, Refusal> {
        let runtime =
            crate::aimask::Runtime::get().ok_or(Refusal::MissingInput("ONNX Runtime"))?;
        let session = runtime
            .open_session(&model_path())
            .map_err(|_| Refusal::MissingInput("blemishRemoval.onnx session"))?;
        let provider = session.provider().label();
        eprintln!("blemish gain map ORT provider: {provider}");
        Ok(Self { session, provider })
    }

    pub fn provider(&self) -> &'static str {
        self.provider
    }
}

impl ModelRunner for OrtGainMapRunner {
    fn run(&mut self, input: &[f32], out: &mut [f32]) -> Result<(), String> {
        let runtime =
            crate::aimask::Runtime::get().ok_or_else(|| "ONNX Runtime unavailable".to_string())?;
        let mut input = input.to_vec();
        let planes = runtime
            .run(&self.session, &mut input, &[1, 3, 278, 278])
            .map_err(|err| format!("{err:#}"))?;
        let plane = planes
            .first()
            .ok_or_else(|| "blemish model returned no output".to_string())?;
        let n = (TILE_STEP * TILE_STEP) as usize;
        if plane.len() < n * 3 {
            return Err(format!("blemish model output has {} values", plane.len()));
        }
        for c in 0..3 {
            for i in 0..n {
                out[i * 3 + c] = plane[c * n + i];
            }
        }
        Ok(())
    }
}

/// Full producer for a 16-bit linear frame.  `mask` is the 450-map treatment
/// mask (255 = correct); the model is the shipped blemishRemoval.onnx.
pub fn produce_gain_map_16u(
    src: &[u16],
    sw: u32,
    sh: u32,
    mask: &[u8],
    runner: &mut impl ModelRunner,
) -> Result<GainMap, Refusal> {
    if !model_available() {
        return Err(Refusal::MissingInput("blemishRemoval.onnx"));
    }
    if sw == 0 || sh == 0 || src.len() != sw as usize * sh as usize * 3 {
        return Err(Refusal::ShapeMismatch("blemish gain map frame"));
    }
    let (mw, mh) = map_dims(sw, sh);
    let n = mw as usize * mh as usize;
    if mask.len() != n {
        return Err(Refusal::ShapeMismatch("blemish gain map mask"));
    }
    let mut small = {
        let q = downsample_super_16u(src, sw, sh, mw, mh);
        q.iter().map(|v| *v as f32 / 65535.0).collect::<Vec<_>>()
    };
    let scale = auto_normalise_scale(&small).ok_or(Refusal::ValueOutOfRange("auto normalise"))?;
    for v in &mut small {
        *v *= scale;
    }
    let ts = tiles(mw, mh);
    let mut canvas = vec![0.0f32; n * 3];
    let mut model_out = vec![0.0f32; (TILE_STEP * TILE_STEP * 3) as usize];
    for &t in &ts {
        runner
            .run(&model_input(&small, mw, mh, t), &mut model_out)
            .map_err(|_| Refusal::NotImplemented(0))?;
        for y in 0..t.valid_h {
            let dst = ((t.y + y) as usize * mw as usize + t.x as usize) * 3;
            let src_row = y as usize * TILE_STEP as usize * 3;
            canvas[dst..dst + t.valid_w as usize * 3]
                .copy_from_slice(&model_out[src_row..src_row + t.valid_w as usize * 3]);
        }
    }
    let mut repaired = vec![0.0f32; n * 3];
    luma_match(&canvas, &small, &mut repaired);
    let mut gain = vec![0.0f32; n * 3];
    gain_rgb(&repaired, &small, mask, &mut gain);
    let mut conf = vec![0.0f32; n];
    confidence(&gain, mw, mh, &mut conf);
    let mut rgba = vec![0.0f32; n * 4];
    for i in 0..n {
        rgba[i * 4..i * 4 + 3].copy_from_slice(&gain[i * 3..i * 3 + 3]);
        rgba[i * 4 + 3] = conf[i];
    }
    Ok(GainMap { width: mw, height: mh, rgba })
}

/// Product entry: uses the shared ORT runtime and its GPU-first EP order.
pub fn produce_gain_map_16u_ort(
    src: &[u16],
    sw: u32,
    sh: u32,
    mask: &[u8],
) -> Result<GainMap, Refusal> {
    let mut runner = OrtGainMapRunner::open()?;
    produce_gain_map_16u(src, sw, sh, mask, &mut runner)
}

#[derive(Clone, Copy, Debug)]
pub struct SpotlessGainmapInputs<'a> {
    pub rgb16: &'a [u16],
    pub width: u32,
    pub height: u32,
    pub gain_map: &'a GainMap,
    /// 0..=1, C1 `[r12+0x18]`.
    pub amount: f32,
    /// Full-res protection mask (255 = protected).
    pub protection_mask: Option<&'a [u8]>,
}

/// C1 consumer `CImgOpBlemishRemoval::RemoveBlemishes` (`0x16B2DD0`).
///
/// Bilinear map sampling, `w = clamp(A / max(1-amount, 0.01), 0, 1)`, amount
/// lerp toward 1 above 0.25, then `(255-mask)/255`; per channel
/// `out = src * (gain*w + 1-w)`.
pub fn apply_gainmap_16u(input: &SpotlessGainmapInputs<'_>) -> Result<Vec<u16>, Refusal> {
    apply_gainmap_16u_gated(input, enabled())
}

pub(crate) fn validate_gainmap_input(
    input: &SpotlessGainmapInputs<'_>,
) -> Result<(usize, usize), Refusal> {
    let w = input.width as usize;
    let h = input.height as usize;
    if w == 0 || h == 0 || input.rgb16.len() != w * h * 3 {
        return Err(Refusal::ShapeMismatch("blemish gain map frame"));
    }
    if !input.amount.is_finite() || !(0.0..=1.0).contains(&input.amount) {
        return Err(Refusal::ValueOutOfRange("blemish gain map amount"));
    }
    if let Some(mask) = input.protection_mask {
        if mask.len() != w * h {
            return Err(Refusal::ShapeMismatch("blemish protection mask"));
        }
    }
    let map = input.gain_map;
    if map.width == 0
        || map.height == 0
        || map.rgba.len() != map.width as usize * map.height as usize * 4
    {
        return Err(Refusal::ShapeMismatch("blemish gain map"));
    }
    Ok((w, h))
}

pub fn apply_gainmap_16u_gated(
    input: &SpotlessGainmapInputs<'_>,
    gate: bool,
) -> Result<Vec<u16>, Refusal> {
    if !gate {
        return apply_gainmap_16u_mode(input, false, false);
    }
    let pixels = (input.width as u64 * input.height as u64) as usize;
    let try_gpu = gpu_auto(pixels);
    if !try_gpu && gpu_enabled() && !gpu_forced() {
        report_gpu(3, || {
            format!(
                "blemish gain map CUDA auto-skip for {pixels} px (pageable bridge slower than CPU); set AURORA_BLEMISH_GPU=1 to force"
            )
        });
    }
    apply_gainmap_16u_mode(input, true, try_gpu)
}

/// Explicit CPU fast lane; this is the byte-identity path used when
/// `AURORA_BLEMISH_GPU=0`.
pub fn apply_gainmap_16u_cpu(
    input: &SpotlessGainmapInputs<'_>,
) -> Result<Vec<u16>, Refusal> {
    apply_gainmap_16u_mode(input, true, false)
}

/// `try_gpu = false` is the byte-identity CPU path used by tests and by
/// `AURORA_BLEMISH_GPU=0`.
pub(crate) fn apply_gainmap_16u_mode(
    input: &SpotlessGainmapInputs<'_>,
    gate: bool,
    try_gpu: bool,
) -> Result<Vec<u16>, Refusal> {
    if !gate {
        return Ok(input.rgb16.to_vec());
    }
    let (w, h) = validate_gainmap_input(input)?;
    // C1 skips the processor at amount 0 (`0x16B2F20`), so this must be an
    // exact no-op before any map sampling or rounding.
    if input.amount == 0.0 {
        return Ok(input.rgb16.to_vec());
    }
    if try_gpu {
        match crate::blemish_gainmap_cuda::apply_16u(input) {
            Ok((out, dispatch)) => {
                report_gpu(1, || {
                    format!(
                        "blemish gain map CUDA: {} {}x{} on {} (ptx {} B)",
                        dispatch.kernel,
                        dispatch.grid.0,
                        dispatch.grid.1,
                        dispatch.device,
                        dispatch.ptx_bytes
                    )
                });
                return Ok(out);
            }
            Err(error) => report_gpu(2, || {
                format!("blemish gain map CUDA unavailable, using CPU: {error:?}")
            }),
        }
    }
    let rows = upsample_map_rows(input.gain_map, w);
    Ok(apply_rows_parallel(input, w, h, &rows))
}

/// Current scalar oracle.  Kept as the byte-identity reference for tests and
/// for the one-row/small-image path.
pub fn apply_gainmap_16u_scalar(
    input: &SpotlessGainmapInputs<'_>,
) -> Result<Vec<u16>, Refusal> {
    let (w, h) = validate_gainmap_input(input)?;
    if input.amount == 0.0 {
        return Ok(input.rgb16.to_vec());
    }
    let map = input.gain_map;
    let sx = map.width as f32 / w as f32;
    let sy = map.height as f32 / h as f32;
    let amount_scale = 1.0 / (1.0 - input.amount).max(0.01);
    let mut out = vec![0u16; w * h * 3];
    for y in 0..h {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, map.height as f32 - 1.0);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(map.height as usize - 1);
        let ty = fy - y0 as f32;
        for x in 0..w {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, map.width as f32 - 1.0);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(map.width as usize - 1);
            let tx = fx - x0 as f32;
            let at = |xx: usize, yy: usize| {
                let i = (yy * map.width as usize + xx) * 4;
                [map.rgba[i], map.rgba[i + 1], map.rgba[i + 2], map.rgba[i + 3]]
            };
            let a = at(x0, y0);
            let b = at(x1, y0);
            let c = at(x0, y1);
            let d = at(x1, y1);
            let mut g = [0.0f32; 3];
            for k in 0..3 {
                let top = a[k] + (b[k] - a[k]) * tx;
                let bottom = c[k] + (d[k] - c[k]) * tx;
                g[k] = top + (bottom - top) * ty;
            }
            let conf = (a[3] + (b[3] - a[3]) * tx)
                + ((c[3] + (d[3] - c[3]) * tx) - (a[3] + (b[3] - a[3]) * tx)) * ty;
            let mut weight = (conf * amount_scale).clamp(0.0, 1.0);
            if input.amount > 0.25 {
                let t = (4.0 * (input.amount - 0.75)).clamp(0.0, 1.0);
                weight = weight * (1.0 - t) + t;
            }
            if let Some(mask) = input.protection_mask {
                weight *= (255 - mask[y * w + x] as u32) as f32 / 255.0;
            }
            for k in 0..3 {
                let src = input.rgb16[(y * w + x) * 3 + k] as f32 / 65535.0;
                let v = src * (g[k] * weight + (1.0 - weight));
                out[(y * w + x) * 3 + k] = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
            }
        }
    }
    Ok(out)
}

/// 8-bit twin of [`SpotlessGainmapInputs`]. Same map, same amount/mask law;
/// the frame and output are `u8` and the unit interval is 0..=255.
#[derive(Clone, Copy, Debug)]
pub struct SpotlessGainmapInputs8<'a> {
    pub rgb8: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub gain_map: &'a GainMap,
    pub amount: f32,
    pub protection_mask: Option<&'a [u8]>,
}

pub(crate) fn validate_gainmap_input8(
    input: &SpotlessGainmapInputs8<'_>,
) -> Result<(usize, usize), Refusal> {
    let w = input.width as usize;
    let h = input.height as usize;
    if w == 0 || h == 0 || input.rgb8.len() != w * h * 3 {
        return Err(Refusal::ShapeMismatch("blemish gain map frame"));
    }
    if !input.amount.is_finite() || !(0.0..=1.0).contains(&input.amount) {
        return Err(Refusal::ValueOutOfRange("blemish gain map amount"));
    }
    if let Some(mask) = input.protection_mask {
        if mask.len() != w * h {
            return Err(Refusal::ShapeMismatch("blemish protection mask"));
        }
    }
    let map = input.gain_map;
    if map.width == 0
        || map.height == 0
        || map.rgba.len() != map.width as usize * map.height as usize * 4
    {
        return Err(Refusal::ShapeMismatch("blemish gain map"));
    }
    Ok((w, h))
}

/// 8-bit consumer, gated by the same `AURORA_BLEMISH_GAINMAP=1` switch.
pub fn apply_gainmap_8u(input: &SpotlessGainmapInputs8<'_>) -> Result<Vec<u8>, Refusal> {
    if !enabled() {
        return apply_gainmap_8u_mode(input, false, false);
    }
    let pixels = (input.width as u64 * input.height as u64) as usize;
    let try_gpu = gpu_auto(pixels);
    if !try_gpu && gpu_enabled() && !gpu_forced() {
        report_gpu(3, || {
            format!(
                "blemish gain map CUDA 8u auto-skip for {pixels} px (pageable bridge slower than CPU); set AURORA_BLEMISH_GPU=1 to force"
            )
        });
    }
    apply_gainmap_8u_mode(input, true, try_gpu)
}

/// Explicit 8-bit CPU fast lane.
pub fn apply_gainmap_8u_cpu(input: &SpotlessGainmapInputs8<'_>) -> Result<Vec<u8>, Refusal> {
    apply_gainmap_8u_mode(input, true, false)
}

pub(crate) fn apply_gainmap_8u_mode(
    input: &SpotlessGainmapInputs8<'_>,
    gate: bool,
    try_gpu: bool,
) -> Result<Vec<u8>, Refusal> {
    if !gate {
        return Ok(input.rgb8.to_vec());
    }
    let (w, h) = validate_gainmap_input8(input)?;
    if input.amount == 0.0 {
        return Ok(input.rgb8.to_vec());
    }
    if try_gpu {
        match crate::blemish_gainmap_cuda::apply_8u(
            input.rgb8,
            input.width,
            input.height,
            input.gain_map,
            input.amount,
            input.protection_mask,
        ) {
            Ok((out, dispatch)) => {
                report_gpu(1, || {
                    format!(
                        "blemish gain map CUDA 8u: {} {}x{} on {}",
                        dispatch.kernel, dispatch.grid.0, dispatch.grid.1, dispatch.device
                    )
                });
                return Ok(out);
            }
            Err(error) => report_gpu(2, || {
                format!("blemish gain map CUDA 8u unavailable, using CPU: {error:?}")
            }),
        }
    }
    let rows = upsample_map_rows(input.gain_map, w);
    Ok(apply_rows_parallel_8u(input, w, h, &rows))
}

/// 8-bit scalar oracle, byte-for-byte the law the CUDA kernel implements.
pub fn apply_gainmap_8u_scalar(
    input: &SpotlessGainmapInputs8<'_>,
) -> Result<Vec<u8>, Refusal> {
    let (w, h) = validate_gainmap_input8(input)?;
    if input.amount == 0.0 {
        return Ok(input.rgb8.to_vec());
    }
    let map = input.gain_map;
    let sx = map.width as f32 / w as f32;
    let sy = map.height as f32 / h as f32;
    let amount_scale = 1.0 / (1.0 - input.amount).max(0.01);
    let mut out = vec![0u8; w * h * 3];
    for y in 0..h {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, map.height as f32 - 1.0);
        let y0 = fy.floor() as usize;
        let y1 = (y0 + 1).min(map.height as usize - 1);
        let ty = fy - y0 as f32;
        for x in 0..w {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, map.width as f32 - 1.0);
            let x0 = fx.floor() as usize;
            let x1 = (x0 + 1).min(map.width as usize - 1);
            let tx = fx - x0 as f32;
            let at = |xx: usize, yy: usize| {
                let i = (yy * map.width as usize + xx) * 4;
                [map.rgba[i], map.rgba[i + 1], map.rgba[i + 2], map.rgba[i + 3]]
            };
            let a = at(x0, y0);
            let b = at(x1, y0);
            let c = at(x0, y1);
            let d = at(x1, y1);
            let mut g = [0.0f32; 3];
            for k in 0..3 {
                let top = a[k] + (b[k] - a[k]) * tx;
                let bottom = c[k] + (d[k] - c[k]) * tx;
                g[k] = top + (bottom - top) * ty;
            }
            let top_c = a[3] + (b[3] - a[3]) * tx;
            let bottom_c = c[3] + (d[3] - c[3]) * tx;
            let conf = top_c + (bottom_c - top_c) * ty;
            let mut weight = (conf * amount_scale).clamp(0.0, 1.0);
            if input.amount > 0.25 {
                let t = (4.0 * (input.amount - 0.75)).clamp(0.0, 1.0);
                weight = weight * (1.0 - t) + t;
            }
            if let Some(mask) = input.protection_mask {
                weight *= (255 - mask[y * w + x] as u32) as f32 / 255.0;
            }
            for k in 0..3 {
                let src = input.rgb8[(y * w + x) * 3 + k] as f32 / 255.0;
                let v = src * (g[k] * weight + (1.0 - weight));
                out[(y * w + x) * 3 + k] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
    Ok(out)
}

/// Horizontal bilinear pass: one row of `[R,G,B,A]` per map row at output
/// width.  Every value is the exact expression the scalar oracle uses.
pub fn upsample_map_rows(map: &GainMap, w: usize) -> Vec<f32> {
    let mw = map.width as usize;
    let mh = map.height as usize;
    let sx = map.width as f32 / w as f32;
    let mut x0 = vec![0usize; w];
    let mut x1 = vec![0usize; w];
    let mut tx = vec![0.0f32; w];
    for x in 0..w {
        let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, map.width as f32 - 1.0);
        let i0 = fx.floor() as usize;
        x0[x] = i0;
        x1[x] = (i0 + 1).min(mw - 1);
        tx[x] = fx - i0 as f32;
    }
    let mut rows = vec![0.0f32; mh * w * 4];
    rows.par_chunks_mut(w * 4).enumerate().for_each(|(my, row)| {
        let base = my * mw * 4;
        for x in 0..w {
            let a = base + x0[x] * 4;
            let b = base + x1[x] * 4;
            let t = tx[x];
            let d = x * 4;
            for c in 0..4 {
                let av = map.rgba[a + c];
                row[d + c] = av + (map.rgba[b + c] - av) * t;
            }
        }
    });
    rows
}

/// Vertical pass + amount/confidence/mask + per-channel multiply.  Rows are
/// independent, so rayon cannot change any pixel's arithmetic.
pub fn apply_rows_parallel(
    input: &SpotlessGainmapInputs<'_>,
    w: usize,
    h: usize,
    rows: &[f32],
) -> Vec<u16> {
    let map_h = input.gain_map.height as usize;
    let sy = map_h as f32 / h as f32;
    let amount_scale = 1.0 / (1.0 - input.amount).max(0.01);
    let curve_t = if input.amount > 0.25 {
        Some((4.0 * (input.amount - 0.75)).clamp(0.0, 1.0))
    } else {
        None
    };
    let src_lut: Vec<f32> = (0..=65535u32).map(|v| v as f32 / 65535.0).collect();
    let mask_lut: [f32; 256] =
        std::array::from_fn(|m| (255 - m as u32) as f32 / 255.0);
    let mut out = vec![0u16; w * h * 3];
    out.par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, out_row)| {
            let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, map_h as f32 - 1.0);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(map_h - 1);
            let ty = fy - y0 as f32;
            let top = &rows[y0 * w * 4..(y0 + 1) * w * 4];
            let bottom = &rows[y1 * w * 4..(y1 + 1) * w * 4];
            let src_row = &input.rgb16[y * w * 3..(y + 1) * w * 3];
            let mask_row = input
                .protection_mask
                .as_ref()
                .map(|mask| &mask[y * w..(y + 1) * w]);
            for x in 0..w {
                let i = x * 4;
                let mut g = [0.0f32; 3];
                for k in 0..3 {
                    let a = top[i + k];
                    let b = bottom[i + k];
                    g[k] = a + (b - a) * ty;
                }
                let a = top[i + 3];
                let b = bottom[i + 3];
                let conf = a + (b - a) * ty;
                let mut weight = (conf * amount_scale).clamp(0.0, 1.0);
                if let Some(t) = curve_t {
                    weight = weight * (1.0 - t) + t;
                }
                if let Some(row) = mask_row {
                    weight *= mask_lut[row[x] as usize];
                }
                for k in 0..3 {
                    let src = src_lut[src_row[x * 3 + k] as usize];
                    let v = src * (g[k] * weight + (1.0 - weight));
                    out_row[x * 3 + k] = (v.clamp(0.0, 1.0) * 65535.0 + 0.5) as u16;
                }
            }
        });
    out
}

/// 8-bit row pass, the same expressions as [`apply_rows_parallel`].
pub fn apply_rows_parallel_8u(
    input: &SpotlessGainmapInputs8<'_>,
    w: usize,
    h: usize,
    rows: &[f32],
) -> Vec<u8> {
    let map_h = input.gain_map.height as usize;
    let sy = map_h as f32 / h as f32;
    let amount_scale = 1.0 / (1.0 - input.amount).max(0.01);
    let curve_t = if input.amount > 0.25 {
        Some((4.0 * (input.amount - 0.75)).clamp(0.0, 1.0))
    } else {
        None
    };
    let src_lut: Vec<f32> = (0..=255u32).map(|v| v as f32 / 255.0).collect();
    let mask_lut: [f32; 256] = std::array::from_fn(|m| (255 - m as u32) as f32 / 255.0);
    let mut out = vec![0u8; w * h * 3];
    out.par_chunks_mut(w * 3)
        .enumerate()
        .for_each(|(y, out_row)| {
            let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, map_h as f32 - 1.0);
            let y0 = fy.floor() as usize;
            let y1 = (y0 + 1).min(map_h - 1);
            let ty = fy - y0 as f32;
            let top = &rows[y0 * w * 4..(y0 + 1) * w * 4];
            let bottom = &rows[y1 * w * 4..(y1 + 1) * w * 4];
            let src_row = &input.rgb8[y * w * 3..(y + 1) * w * 3];
            let mask_row = input
                .protection_mask
                .as_ref()
                .map(|mask| &mask[y * w..(y + 1) * w]);
            for x in 0..w {
                let i = x * 4;
                let mut g = [0.0f32; 3];
                for k in 0..3 {
                    let a = top[i + k];
                    let b = bottom[i + k];
                    g[k] = a + (b - a) * ty;
                }
                let a = top[i + 3];
                let b = bottom[i + 3];
                let conf = a + (b - a) * ty;
                let mut weight = (conf * amount_scale).clamp(0.0, 1.0);
                if let Some(t) = curve_t {
                    weight = weight * (1.0 - t) + t;
                }
                if let Some(row) = mask_row {
                    weight *= mask_lut[row[x] as usize];
                }
                for k in 0..3 {
                    let src = src_lut[src_row[x * 3 + k] as usize];
                    let v = src * (g[k] * weight + (1.0 - weight));
                    out_row[x * 3 + k] = (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                }
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_map(gain: [f32; 3], conf: f32) -> GainMap {
        GainMap {
            width: 2,
            height: 2,
            rgba: (0..4).flat_map(|_| [gain[0], gain[1], gain[2], conf]).collect(),
        }
    }

    #[test]
    fn disabled_is_byte_identity() {
        let src: Vec<u16> = (0..48).map(|i| (i * 1301) as u16).collect();
        let map = flat_map([2.0, 0.5, 1.0], 1.0);
        let out = apply_gainmap_16u_gated(
            &SpotlessGainmapInputs {
                rgb16: &src,
                width: 4,
                height: 4,
                gain_map: &map,
                amount: 0.5,
                protection_mask: None,
            },
            false,
        )
        .unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn flat_gain_is_per_channel() {
        let src = vec![10000u16; 16 * 3];
        let map = flat_map([2.0, 0.5, 1.0], 1.0);
        let out = apply_gainmap_16u_gated(
            &SpotlessGainmapInputs {
                rgb16: &src,
                width: 4,
                height: 4,
                gain_map: &map,
                amount: 1.0,
                protection_mask: None,
            },
            true,
        )
        .unwrap();
        assert_eq!(&out[0..3], &[20000, 5000, 10000]);
    }

    #[test]
    fn confidence_zero_leaves_source() {
        let src = vec![12345u16; 16 * 3];
        let map = flat_map([2.0, 0.5, 1.0], 0.0);
        let out = apply_gainmap_16u_gated(
            &SpotlessGainmapInputs {
                rgb16: &src,
                width: 4,
                height: 4,
                gain_map: &map,
                amount: 0.5,
                protection_mask: None,
            },
            true,
        )
        .unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn protection_mask_255_leaves_source() {
        let src = vec![12345u16; 16 * 3];
        let mask = vec![255u8; 16];
        let map = flat_map([2.0, 0.5, 1.0], 1.0);
        let out = apply_gainmap_16u_gated(
            &SpotlessGainmapInputs {
                rgb16: &src,
                width: 4,
                height: 4,
                gain_map: &map,
                amount: 1.0,
                protection_mask: Some(&mask),
            },
            true,
        )
        .unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn amount_zero_is_exact_noop() {
        let src = vec![23456u16; 16 * 3];
        let map = flat_map([2.0, 0.5, 1.0], 1.0);
        let out = apply_gainmap_16u_gated(
            &SpotlessGainmapInputs {
                rgb16: &src,
                width: 4,
                height: 4,
                gain_map: &map,
                amount: 0.0,
                protection_mask: None,
            },
            true,
        )
        .unwrap();
        assert_eq!(out, src);
    }

    #[test]
    fn amount_uses_existing_parameter_keys() {
        let mut adjust = crate::AdjustmentSet::default();
        assert_eq!(amount_from_adjustments(&adjust), 0.0);
        adjust.extra.insert("BlemishRemovalAmount".into(), 51.0989);
        assert!((amount_from_adjustments(&adjust) - 0.510989).abs() < 1e-6);
        assert!((amount_from_retouch_layer(Some(51.0989)) - 0.510989).abs() < 1e-6);
        assert_eq!(amount_from_retouch_layer(None), 0.0);
    }

    #[test]
    fn missing_model_path_is_detected() {
        assert!(!model_available_at(Path::new(
            r"C:\definitely-missing\blemishRemoval.onnx"
        )));
    }

    #[test]
    fn parallel_matches_scalar_on_boundaries() {
        let w = 37usize;
        let h = 29usize;
        let src: Vec<u16> = (0..w * h * 3)
            .map(|i| ((i * 7919) % 65536) as u16)
            .collect();
        let map = GainMap {
            width: 7,
            height: 5,
            rgba: (0..7 * 5 * 4)
                .map(|i| 0.5 + (i % 17) as f32 * 0.09)
                .collect(),
        };
        let input = SpotlessGainmapInputs {
            rgb16: &src,
            width: w as u32,
            height: h as u32,
            gain_map: &map,
            amount: 0.5,
            protection_mask: None,
        };
        let scalar = apply_gainmap_16u_scalar(&input).unwrap();
        let fast = apply_gainmap_16u_gated(&input, true).unwrap();
        assert_eq!(scalar, fast);
    }

    #[test]
    fn single_thread_matches_parallel() {
        let w = 64usize;
        let h = 48usize;
        let src: Vec<u16> = (0..w * h * 3).map(|i| (i * 31 % 65535) as u16).collect();
        let map = GainMap {
            width: 9,
            height: 6,
            rgba: (0..9 * 6 * 4).map(|i| 0.75 + (i % 13) as f32 * 0.05).collect(),
        };
        let mask = vec![128u8; w * h];
        let input = SpotlessGainmapInputs {
            rgb16: &src,
            width: w as u32,
            height: h as u32,
            gain_map: &map,
            amount: 0.75,
            protection_mask: Some(&mask),
        };
        let parallel = apply_gainmap_16u_gated(&input, true).unwrap();
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| apply_gainmap_16u_gated(&input, true).unwrap());
        assert_eq!(parallel, serial);
    }

    #[test]
    fn golden_9868_matches_scalar() {
        let dir = std::path::Path::new(
            r"C:\Users\Jxcz2\Documents\Codex\2026-11-10\capture-one-rust-gpui-thread-01a1bf2a\90_work\lanes\c1-parse\c1_blemish_model_r280\harness\fixtures\nef_9868",
        );
        if !dir.join("engine_9868_blemish.u16").is_file() {
            return;
        }
        let read_u16 = |name: &str| -> Vec<u16> {
            std::fs::read(dir.join(name))
                .unwrap()
                .chunks_exact(2)
                .map(|p| u16::from_le_bytes([p[0], p[1]]))
                .collect()
        };
        let src = read_u16("s1_linear_9868.u16");
        let golden = read_u16("engine_9868_blemish.u16");
        let rgba: Vec<f32> = std::fs::read(dir.join("gainmap_9868_450x300.f32"))
            .unwrap()
            .chunks_exact(4)
            .map(|p| f32::from_le_bytes([p[0], p[1], p[2], p[3]]))
            .collect();
        let map = GainMap { width: 450, height: 300, rgba };
        let input = SpotlessGainmapInputs {
            rgb16: &src,
            width: 6048,
            height: 4032,
            gain_map: &map,
            amount: 0.5,
            protection_mask: None,
        };
        let cpu = apply_gainmap_16u_mode(&input, true, false).unwrap();
        assert_eq!(cpu, golden);
    }

    #[test]
    fn golden_9868_gpu_matches_cpu() {
        let dir = std::path::Path::new(
            r"C:\Users\Jxcz2\Documents\Codex\2026-11-10\capture-one-rust-gpui-thread-01a1bf2a\90_work\lanes\c1-parse\c1_blemish_model_r280\harness\fixtures\nef_9868",
        );
        if !dir.join("engine_9868_blemish.u16").is_file() {
            return;
        }
        let read_u16 = |name: &str| -> Vec<u16> {
            std::fs::read(dir.join(name))
                .unwrap()
                .chunks_exact(2)
                .map(|p| u16::from_le_bytes([p[0], p[1]]))
                .collect()
        };
        let src = read_u16("s1_linear_9868.u16");
        let golden = read_u16("engine_9868_blemish.u16");
        let rgba: Vec<f32> = std::fs::read(dir.join("gainmap_9868_450x300.f32"))
            .unwrap()
            .chunks_exact(4)
            .map(|p| f32::from_le_bytes([p[0], p[1], p[2], p[3]]))
            .collect();
        let map = GainMap { width: 450, height: 300, rgba };
        let input = SpotlessGainmapInputs {
            rgb16: &src,
            width: 6048,
            height: 4032,
            gain_map: &map,
            amount: 0.5,
            protection_mask: None,
        };
        let (gpu, dispatch) = match crate::blemish_gainmap_cuda::apply_16u(&input) {
            Ok(value) => value,
            Err(error) => {
                eprintln!("SKIP golden_9868_gpu_matches_cpu: {error:?}");
                return;
            }
        };
        eprintln!(
            "GPU golden: {} grid={:?} block={:?} device={}",
            dispatch.kernel, dispatch.grid, dispatch.block, dispatch.device
        );
        assert!(dispatch.kernel.contains("16u"));
        assert_eq!(gpu, golden);
        let cpu = apply_gainmap_16u_scalar(&input).unwrap();
        assert_eq!(gpu, cpu);
        let forced = apply_gainmap_16u_mode(&input, true, true).unwrap();
        assert_eq!(forced, golden);
    }

    #[test]
    fn gpu_off_matches_golden_without_upload() {
        let src = vec![23456u16; 64 * 48 * 3];
        let map = GainMap {
            width: 9,
            height: 6,
            rgba: (0..9 * 6 * 4).map(|i| 0.75 + (i % 13) as f32 * 0.05).collect(),
        };
        let mask = vec![90u8; 64 * 48];
        let input = SpotlessGainmapInputs {
            rgb16: &src,
            width: 64,
            height: 48,
            gain_map: &map,
            amount: 0.75,
            protection_mask: Some(&mask),
        };
        let cpu = apply_gainmap_16u_scalar(&input).unwrap();
        let gated = apply_gainmap_16u_mode(&input, true, false).unwrap();
        assert_eq!(cpu, gated);
    }

    #[test]
    fn eight_bit_parallel_matches_scalar() {
        let w = 37usize;
        let h = 29usize;
        let src: Vec<u8> = (0..w * h * 3).map(|i| (i * 197 % 256) as u8).collect();
        let map = GainMap {
            width: 7,
            height: 5,
            rgba: (0..7 * 5 * 4)
                .map(|i| 0.5 + (i % 17) as f32 * 0.09)
                .collect(),
        };
        let mask = vec![160u8; w * h];
        let input = SpotlessGainmapInputs8 {
            rgb8: &src,
            width: w as u32,
            height: h as u32,
            gain_map: &map,
            amount: 0.5,
            protection_mask: Some(&mask),
        };
        let scalar = apply_gainmap_8u_scalar(&input).unwrap();
        let fast = apply_gainmap_8u_mode(&input, true, false).unwrap();
        assert_eq!(scalar, fast);
    }

    #[test]
    fn eight_bit_gpu_matches_cpu() {
        let w = 512usize;
        let h = 384usize;
        let src: Vec<u8> = (0..w * h * 3).map(|i| (i * 91 % 256) as u8).collect();
        let map = GainMap {
            width: 450,
            height: 300,
            rgba: (0..450 * 300 * 4)
                .map(|i| 0.5 + (i % 29) as f32 * 0.05)
                .collect(),
        };
        let input = SpotlessGainmapInputs8 {
            rgb8: &src,
            width: w as u32,
            height: h as u32,
            gain_map: &map,
            amount: 0.5,
            protection_mask: None,
        };
        let (gpu, dispatch) =
            match crate::blemish_gainmap_cuda::apply_8u(&src, w as u32, h as u32, &map, 0.5, None)
            {
                Ok(value) => value,
                Err(error) => {
                    eprintln!("SKIP eight_bit_gpu_matches_cpu: {error:?}");
                    return;
                }
            };
        eprintln!(
            "GPU 8u: {} grid={:?} device={}",
            dispatch.kernel, dispatch.grid, dispatch.device
        );
        assert!(dispatch.kernel.contains("8u"));
        assert_eq!(gpu, apply_gainmap_8u_scalar(&input).unwrap());
    }
}
