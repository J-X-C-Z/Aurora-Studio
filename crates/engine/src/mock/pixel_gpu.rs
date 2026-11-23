//! Optional CUDA path for the three light per-pixel operators: the manual
//! vignette, the LCC Uniform Light gain and film grain.
//!
//! Every kernel is a twin of a CPU reference in this file or in
//! [`super::grain`], compiled with `-fmad=false` so the float order matches.
//! The CPU path stays authoritative: `AURORA_PIXEL_GPU=0` (or the per-operator
//! switches) restores it, and a missing driver or PTX image falls back to it.

use std::ffi::c_void;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::{Mutex, OnceLock};
use std::sync::Arc;

use aurora_portrait::cuda::{CudaError, DeviceBuffer, Nvidia};

use crate::dust_lcc::{GpuLccGrid, LightFalloffLimit};
use crate::{FilmGrain, VignetteCenter, VignetteMethod};

const PTX: &str = include_str!("../../native/cuda/pixel_gpu.ptx");
const BLOCK: (u32, u32) = (16, 16);
/// Retain one 24-32 MP working set (three RGBA-sized planes plus a node table).
const MAX_RETAINED_BYTES: usize = 768 * 1024 * 1024;
/// Gain planes are reused across renders: only a parameter that moves the
/// plane itself misses the cache. Two 24 MP planes per operator fit here.
const MAX_CACHED_BYTES: usize = 192 * 1024 * 1024;

/// One of the three operators, for the per-operator fallback switches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Op {
    Vignette,
    Lcc,
    Grain,
}

impl Op {
    fn env_name(self) -> &'static str {
        match self {
            Self::Vignette => "AURORA_PIXEL_VIGNETTE_GPU",
            Self::Lcc => "AURORA_PIXEL_LCC_GPU",
            Self::Grain => "AURORA_PIXEL_GRAIN_GPU",
        }
    }
}

/// `0` / `false` means off; any other value means on.
fn env_flag(name: &str) -> Option<bool> {
    let value = std::env::var_os(name)?;
    let value = value.to_string_lossy();
    Some(!(value == "0" || value.eq_ignore_ascii_case("false")))
}

/// The master switch defaults on; a per-operator switch can still turn one
/// operator off, and `AURORA_PIXEL_GPU=0` overrides all three.
pub(crate) fn enabled(op: Op) -> bool {
    if env_flag("AURORA_PIXEL_GPU") == Some(false) {
        return false;
    }
    env_flag(op.env_name()).unwrap_or(true)
}

/// Counts planes that actually came back from the device; the render tests use
/// it to prove the live path is wired, not just callable.
#[cfg(test)]
pub(crate) static PLANE_HITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// ───────────────────────────── CPU references ─────────────────────────────

/// `CImgOpLensLight`'s manual branch: `exp2(clamp(amount * rho2, -3, 3))`.
/// `rho2` is `(r/r_ref)^2` with the frame corner at 1.0, so `amount` EV lands
/// on the corner and nothing on the centre. This is the same expression the
/// per-pixel render path uses; the kernel mirrors it.
pub(crate) fn vignette_gain(
    x: f32,
    y: f32,
    method: VignetteMethod,
    amount: f32,
    aspect: f32,
    center: [f32; 2],
) -> f32 {
    if amount == 0.0 || !amount.is_finite() {
        return 1.0;
    }
    let (dx, dy) = (x - center[0], y - center[1]);
    let a2 = if aspect.is_finite() && aspect > 0.0 { aspect * aspect } else { 1.0 };
    let rho2 = match method {
        VignetteMethod::EllipticOnCrop => 2.0 * (dx * dx + dy * dy),
        VignetteMethod::Circular | VignetteMethod::CircularCrop => {
            4.0 * (dx * dx * a2 + dy * dy) / (a2 + 1.0)
        }
    };
    (amount * rho2).clamp(-3.0, 3.0).exp2()
}

/// Resolves `IC_Vignetting.center` into normalised frame coordinates, exactly
/// like `vignette_center` in `mock/process.rs`.
pub(crate) fn vignette_center(
    center: VignetteCenter,
    x: f32,
    y: f32,
    optic_center: Option<[f32; 2]>,
    size: [f32; 2],
) -> [f32; 2] {
    let (w, h) = (size[0].max(1.0), size[1].max(1.0));
    let resolved = match center {
        VignetteCenter::Cam | VignetteCenter::Crop => [0.5, 0.5],
        VignetteCenter::Lens => optic_center.map(|p| [p[0] / w, p[1] / h]).unwrap_or([0.5, 0.5]),
        VignetteCenter::Custom => [x / w, y / h],
    };
    resolved.map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.5 })
}

/// Bilinear lookup into an integer-centred plane, matching the LCC grid's own
/// clamped-cell extrapolation at the borders.
pub(crate) fn sample_plane(plane: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    if w < 2 || h < 2 || plane.len() < w * h || !x.is_finite() || !y.is_finite() {
        return 1.0;
    }
    let fx = x.floor();
    let fy = y.floor();
    let ix = (fx.max(0.0) as usize).min(w - 2);
    let iy = (fy.max(0.0) as usize).min(h - 2);
    let tx = x - ix as f32;
    let ty = y - iy as f32;
    let at = |px: usize, py: usize| plane[py * w + px];
    let top = at(ix, iy) * (1.0 - tx) + at(ix + 1, iy) * tx;
    let bottom = at(ix, iy + 1) * (1.0 - tx) + at(ix + 1, iy + 1) * tx;
    top * (1.0 - ty) + bottom * ty
}

// ─────────────────────────────── CUDA pool ────────────────────────────────

struct SendBuffer(DeviceBuffer<'static>);

// The driver table each buffer borrows is boxed inside `Pool` and outlives it;
// `State` serialises every use, so a raw handle never moves concurrently.
unsafe impl Send for SendBuffer {}

struct Pool {
    gpu: Box<Nvidia>,
    rgba: SendBuffer,
    plane: SendBuffer,
    tmp: SendBuffer,
    taps: SendBuffer,
    nodes: SendBuffer,
    stats: SendBuffer,
    bytes: usize,
    node_bytes: usize,
}

fn alloc_static(gpu: &Nvidia, bytes: usize) -> Result<DeviceBuffer<'static>, CudaError> {
    let buffer = gpu.alloc_buffer(bytes)?;
    Ok(unsafe { std::mem::transmute::<DeviceBuffer<'_>, DeviceBuffer<'static>>(buffer) })
}

impl Pool {
    fn open(bytes: usize, node_bytes: usize) -> Result<Self, CudaError> {
        let gpu = Box::new(Nvidia::open_ptx(PTX)?);
        Ok(Self {
            rgba: SendBuffer(alloc_static(&gpu, bytes)?),
            plane: SendBuffer(alloc_static(&gpu, bytes)?),
            tmp: SendBuffer(alloc_static(&gpu, bytes)?),
            taps: SendBuffer(alloc_static(&gpu, 68)?),
            nodes: SendBuffer(alloc_static(&gpu, node_bytes.max(12))?),
            stats: SendBuffer(alloc_static(&gpu, 4)?),
            gpu,
            bytes,
            node_bytes,
        })
    }

    fn ensure(&mut self, bytes: usize, node_bytes: usize) -> Result<(), CudaError> {
        if bytes > self.bytes {
            self.rgba = SendBuffer(alloc_static(&self.gpu, bytes)?);
            self.plane = SendBuffer(alloc_static(&self.gpu, bytes)?);
            self.tmp = SendBuffer(alloc_static(&self.gpu, bytes)?);
            self.bytes = bytes;
        }
        if node_bytes > self.node_bytes {
            self.nodes = SendBuffer(alloc_static(&self.gpu, node_bytes)?);
            self.node_bytes = node_bytes;
        }
        Ok(())
    }

}

enum State {
    NotTried,
    Ready(Pool),
    Unavailable,
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::NotTried))
}

/// Runs `body` against a ready pool, opening it lazily. A refusal disables the
/// GPU path for the process instead of retrying the driver on every frame.
fn with_pool<T>(
    bytes: usize,
    node_bytes: usize,
    body: impl FnOnce(&Pool) -> Result<T, CudaError>,
) -> Option<T> {
    if bytes.saturating_mul(3).saturating_add(node_bytes) > MAX_RETAINED_BYTES {
        return None;
    }
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    match &mut *state {
        State::Unavailable => None,
        State::NotTried => match Pool::open(bytes, node_bytes) {
            Ok(mut pool) => {
                let result = pool
                    .ensure(bytes, node_bytes)
                    .and_then(|()| body(&pool));
                match result {
                    Ok(value) => {
                        *state = State::Ready(pool);
                        Some(value)
                    }
                    Err(error) => {
                        log::warn!("pixel_gpu: {error:?}; staying on CPU");
                        *state = State::Unavailable;
                        None
                    }
                }
            }
            Err(error) => {
                log::warn!("pixel_gpu: {error:?}; staying on CPU");
                *state = State::Unavailable;
                None
            }
        },
        State::Ready(pool) => match pool
            .ensure(bytes, node_bytes)
            .and_then(|()| body(pool))
        {
            Ok(value) => Some(value),
            Err(error) => {
                log::warn!("pixel_gpu: {error:?}; staying on CPU");
                None
            }
        },
    }
}

fn ptr<T>(value: &mut T) -> *mut c_void {
    value as *mut T as *mut c_void
}

fn launch2<const N: usize>(
    gpu: &Nvidia,
    kernel: &'static str,
    w: usize,
    h: usize,
    mut args: [*mut c_void; N],
) -> Result<(), CudaError> {
    let grid = (((w as u32) + 15) / 16, ((h as u32) + 15) / 16);
    gpu.launch_raw(kernel, grid, BLOCK, &mut args)?;
    Ok(())
}

// ─────────────────────────── vignette gain plane ──────────────────────────

struct PlaneCache {
    vignette: Vec<(u64, Arc<Vec<f32>>)>,
    lcc: Vec<(u64, Arc<Vec<f32>>)>,
}

fn cache() -> &'static Mutex<PlaneCache> {
    static CACHE: OnceLock<Mutex<PlaneCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(PlaneCache { vignette: Vec::new(), lcc: Vec::new() }))
}

fn hashed(parts: impl FnOnce(&mut DefaultHasher)) -> u64 {
    let mut hasher = DefaultHasher::new();
    parts(&mut hasher);
    hasher.finish()
}

fn hash_f32(hasher: &mut DefaultHasher, value: f32) {
    value.to_bits().hash(hasher);
}

fn cache_lookup(
    lane: fn(&mut PlaneCache) -> &mut Vec<(u64, Arc<Vec<f32>>)>,
    key: u64,
) -> Option<Arc<Vec<f32>>> {
    let mut cache = cache().lock().unwrap_or_else(|error| error.into_inner());
    let entries = lane(&mut cache);
    let index = entries.iter().position(|(cached, _)| *cached == key)?;
    let entry = entries.remove(index);
    let value = Arc::clone(&entry.1);
    entries.push(entry);
    Some(value)
}

fn cache_store(
    lane: fn(&mut PlaneCache) -> &mut Vec<(u64, Arc<Vec<f32>>)>,
    key: u64,
    value: Arc<Vec<f32>>,
) {
    let mut cache = cache().lock().unwrap_or_else(|error| error.into_inner());
    let entries = lane(&mut cache);
    if entries.iter().any(|(cached, _)| *cached == key) {
        return;
    }
    let bytes = value.len() * 4;
    let mut total: usize = entries.iter().map(|(_, plane)| plane.len() * 4).sum();
    while !entries.is_empty() && (entries.len() >= 2 || total + bytes > MAX_CACHED_BYTES) {
        let removed = entries.remove(0);
        total -= removed.1.len() * 4;
    }
    if bytes <= MAX_CACHED_BYTES {
        entries.push((key, value));
    }
}

/// Cached vignette plane. The whole-operator cost is `exp2` per pixel; the
/// cache exists because a slider drag that does not touch vignetting should not
/// recompute (or re-copy) the plane.
pub(crate) fn vignette_plane_cached(
    width: usize,
    height: usize,
    region: [f32; 4],
    method: VignetteMethod,
    amount: f32,
    aspect: f32,
    center: [f32; 2],
) -> Option<Arc<Vec<f32>>> {
    let key = hashed(|hasher| {
        width.hash(hasher);
        height.hash(hasher);
        region.iter().for_each(|value| hash_f32(hasher, *value));
        method.hash(hasher);
        hash_f32(hasher, amount);
        hash_f32(hasher, aspect);
        center.iter().for_each(|value| hash_f32(hasher, *value));
    });
    if let Some(plane) = cache_lookup(|cache| &mut cache.vignette, key) {
        return Some(plane);
    }
    let plane = Arc::new(vignette_plane(width, height, region, method, amount, aspect, center)?);
    cache_store(|cache| &mut cache.vignette, key, Arc::clone(&plane));
    Some(plane)
}

/// Cached LCC plane, keyed by the grid contents and the slider strength.
pub(crate) fn lcc_plane_cached(grid: &GpuLccGrid, width: usize, height: usize) -> Option<Arc<Vec<f32>>> {
    let key = hashed(|hasher| {
        width.hash(hasher);
        height.hash(hasher);
        grid.columns.hash(hasher);
        grid.rows.hash(hasher);
        grid.origin.iter().chain(grid.step.iter()).for_each(|value| hash_f32(hasher, *value));
        hash_f32(hasher, grid.max_level);
        hash_f32(hasher, grid.strength);
        (grid.limit as u8).hash(hasher);
        grid.luminance.iter().for_each(|value| hash_f32(hasher, *value));
        grid.slope_x.iter().for_each(|value| hash_f32(hasher, *value));
        grid.slope_y.iter().for_each(|value| hash_f32(hasher, *value));
    });
    if let Some(plane) = cache_lookup(|cache| &mut cache.lcc, key) {
        return Some(plane);
    }
    let plane = Arc::new(lcc_plane(grid, width, height)?);
    cache_store(|cache| &mut cache.lcc, key, Arc::clone(&plane));
    Some(plane)
}

/// Vignette gain for every output pixel, `region` = `[x, y, width, height]` in
/// normalised frame coordinates. `None` means "use the CPU formula".
pub(crate) fn vignette_plane(
    width: usize,
    height: usize,
    region: [f32; 4],
    method: VignetteMethod,
    amount: f32,
    aspect: f32,
    center: [f32; 2],
) -> Option<Vec<f32>> {
    if width == 0 || height == 0 || amount == 0.0 || !amount.is_finite() {
        return None;
    }
    if !enabled(Op::Vignette) {
        return None;
    }
    let bytes = width.checked_mul(height)?.checked_mul(4)?;
    let method = match method {
        VignetteMethod::EllipticOnCrop => 0,
        VignetteMethod::Circular | VignetteMethod::CircularCrop => 1,
    } as i32;
    let mut out = vec![0.0f32; width * height];
    let out_bytes =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };
    let done = with_pool(bytes, 0, |pool| {
        let gpu = pool.gpu.as_ref();
        let (mut w_i, mut h_i) = (width as i32, height as i32);
        let (mut ox, mut sx) = (region[0], region[2]);
        let (mut oy, mut sy) = (region[1], region[3]);
        let (mut amount, mut aspect) = (amount, aspect);
        let (mut cx, mut cy) = (center[0], center[1]);
        let mut method = method;
        let mut plane = pool.plane.0.handle();
        launch2(
            gpu,
            "vignette_gain_kernel",
            width,
            height,
            [
                ptr(&mut plane),
                ptr(&mut w_i),
                ptr(&mut h_i),
                ptr(&mut ox),
                ptr(&mut sx),
                ptr(&mut oy),
                ptr(&mut sy),
                ptr(&mut amount),
                ptr(&mut aspect),
                ptr(&mut cx),
                ptr(&mut cy),
                ptr(&mut method),
            ],
        )?;
        gpu.download_buffer(&pool.plane.0, out_bytes)
    });
    done.map(|()| {
        #[cfg(test)]
        PLANE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        out
    })
}

// ──────────────────────────── LCC gain plane ──────────────────────────────

/// One texel per source pixel of `exp2(ev)` from the LCC grid. `None` means
/// "use `UniformLight::source_gain`".
pub(crate) fn lcc_plane(grid: &GpuLccGrid, width: usize, height: usize) -> Option<Vec<f32>> {
    if width < 2 || height < 2 || grid.columns < 2 || grid.rows < 2 {
        return None;
    }
    if !enabled(Op::Lcc) {
        return None;
    }
    let bytes = width.checked_mul(height)?.checked_mul(4)?;
    let nodes = grid.columns.checked_mul(grid.rows)?;
    let node_bytes = nodes.checked_mul(12)?;
    if grid.luminance.len() < nodes || grid.slope_x.len() < nodes || grid.slope_y.len() < nodes {
        return None;
    }
    let mut out = vec![0.0f32; width * height];
    let out_bytes =
        unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4) };
    let node_bytes_slice = |values: &[f32]| unsafe {
        std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4)
    };
    let done = with_pool(bytes, node_bytes, |pool| {
        let gpu = pool.gpu.as_ref();
        // Three arrays in one node buffer: luminance, then slope_x, then slope_y.
        let node_handle = pool.nodes.0.handle();
        let mut packed = Vec::with_capacity(nodes * 3);
        packed.extend_from_slice(&grid.luminance[..nodes]);
        packed.extend_from_slice(&grid.slope_x[..nodes]);
        packed.extend_from_slice(&grid.slope_y[..nodes]);
        gpu.upload_buffer(&pool.nodes.0, node_bytes_slice(&packed))?;
        let (mut columns, mut rows) = (grid.columns as i32, grid.rows as i32);
        let (mut ox, mut oy) = (grid.origin[0], grid.origin[1]);
        let (mut sx, mut sy) = (grid.step[0], grid.step[1]);
        let (mut max_level, mut strength) = (grid.max_level, grid.strength);
        let mut limit = match grid.limit {
            LightFalloffLimit::Soft => 0,
            LightFalloffLimit::Hard => 1,
            LightFalloffLimit::Uncapped => 2,
        };
        // C is `(const float*, ...)`; the driver still wants addresses of the
        // argument values, so keep them in locals.
        let mut lum = node_handle;
        let mut slope_x = node_handle + nodes as u64 * 4;
        let mut slope_y = node_handle + nodes as u64 * 8;
        let mut plane = pool.plane.0.handle();
        let (mut w_i, mut h_i) = (width as i32, height as i32);
        launch2(
            gpu,
            "lcc_gain_kernel",
            width,
            height,
            [
                ptr(&mut plane),
                ptr(&mut w_i),
                ptr(&mut h_i),
                ptr(&mut columns),
                ptr(&mut rows),
                ptr(&mut ox),
                ptr(&mut oy),
                ptr(&mut sx),
                ptr(&mut sy),
                ptr(&mut max_level),
                ptr(&mut strength),
                ptr(&mut limit),
                ptr(&mut lum),
                ptr(&mut slope_x),
                ptr(&mut slope_y),
            ],
        )?;
        gpu.download_buffer(&pool.plane.0, out_bytes)
    });
    done.map(|()| {
        #[cfg(test)]
        PLANE_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        out
    })
}

// ──────────────────────────────── grain ───────────────────────────────────

/// Applies the grain plane on the GPU. `false` means the caller must run the
/// CPU stage (disabled, no driver, short frame, or an inactive amount).
pub(crate) fn grain_stage(
    rgba: &mut [u8],
    w: usize,
    h: usize,
    adj: &FilmGrain,
    scale: f32,
) -> bool {
    let amount = (adj.impact / 100.0).clamp(0.0, 1.0);
    if amount <= 0.0 || w == 0 || h == 0 || rgba.len() < w * h * 4 {
        return false;
    }
    if !enabled(Op::Grain) {
        return false;
    }
    let Some(bytes) = w.checked_mul(h).and_then(|n| n.checked_mul(4)) else {
        return false;
    };
    let composite = adj.kind.uses_erf_blur();
    let zoom = if scale.is_finite() && scale > 0.0 { scale } else { 1.0 };
    let sigma = if adj.impact.is_finite() && adj.impact > 0.0 { adj.sigma_px() * zoom } else { 0.0 };
    let (radius, taps) = if composite && sigma > 0.0 {
        let taps = super::grain::erf_taps(sigma);
        ((taps.len() / 2) as i32, taps)
    } else {
        (0, Vec::new())
    };
    let node_bytes = (h * 8).max(12);
    let (ox, oy) = super::grain::noise_offsets();

    with_pool(bytes, node_bytes, |pool| {
        let gpu = pool.gpu.as_ref();
        let rgba_handle = pool.rgba.0.handle();
        let tap_handle = pool.taps.0.handle();
        gpu.upload_buffer(&pool.rgba.0, rgba)?;
        if !taps.is_empty() {
            let tap_slice = unsafe {
                std::slice::from_raw_parts(taps.as_ptr() as *const u8, taps.len() * 4)
            };
            gpu.upload_buffer(&pool.taps.0, tap_slice)?;
        }
        let (mut w_i, mut h_i) = (w as i32, h as i32);
        let (mut ox, mut oy) = (ox, oy);
        let mut rgba_ptr = rgba_handle;
        let mut plane = pool.plane.0.handle();
        let mut tmp = pool.tmp.0.handle();
        if !composite {
            let mut gain = amount * super::grain::GRAIN_GAIN_V1;
            launch2(gpu, "grain_v1_kernel", w, h, [
                ptr(&mut rgba_ptr),
                ptr(&mut w_i),
                ptr(&mut h_i),
                ptr(&mut gain),
                ptr(&mut ox),
                ptr(&mut oy),
            ])?;
            gpu.download_buffer(&pool.rgba.0, rgba)
        } else {
            launch2(
                gpu,
                "grain_plane_kernel",
                w,
                h,
                [ptr(&mut plane), ptr(&mut w_i), ptr(&mut h_i), ptr(&mut ox), ptr(&mut oy)],
            )?;
            if !taps.is_empty() {
                let mut r = radius;
                let mut tap_ptr = tap_handle;
                launch2(gpu, "grain_blur_h_kernel", w, h, [
                    ptr(&mut plane),
                    ptr(&mut tmp),
                    ptr(&mut w_i),
                    ptr(&mut h_i),
                    ptr(&mut r),
                    ptr(&mut tap_ptr),
                ])?;
                launch2(gpu, "grain_blur_v_kernel", w, h, [
                    ptr(&mut tmp),
                    ptr(&mut plane),
                    ptr(&mut w_i),
                    ptr(&mut h_i),
                    ptr(&mut r),
                    ptr(&mut tap_ptr),
                ])?;
            }
            // Row totals, then one block over the rows: deterministic order.
            let stats = pool.nodes.0.handle();
            let mut row_sum = stats;
            let mut row_sq = stats + h as u64 * 4;
            gpu.launch_raw("grain_row_stats_kernel", (h as u32, 1), (256, 1), &mut [
                ptr(&mut plane),
                ptr(&mut row_sum),
                ptr(&mut row_sq),
                ptr(&mut w_i),
                ptr(&mut h_i),
            ])?;
            let mut inv_n = 1.0f32 / (w * h) as f32;
            let mut sd_out = pool.stats.0.handle();
            gpu.launch_raw("grain_reduce_stats_kernel", (1, 1), (256, 1), &mut [
                ptr(&mut row_sum),
                ptr(&mut row_sq),
                ptr(&mut h_i),
                ptr(&mut inv_n),
                ptr(&mut sd_out),
            ])?;
            let mut sd_bytes = [0u8; 4];
            gpu.download_buffer(&pool.stats.0, &mut sd_bytes)?;
            let sd = f32::from_le_bytes(sd_bytes).max(1e-6);
            let mut gain = amount * adj.kind.v8_amplitude(adj.granularity) / (sd * 257.0);
            launch2(gpu, "grain_v8_kernel", w, h, [
                ptr(&mut rgba_ptr),
                ptr(&mut plane),
                ptr(&mut w_i),
                ptr(&mut h_i),
                ptr(&mut gain),
            ])?;
            gpu.download_buffer(&pool.rgba.0, rgba)
        }
    })
    .is_some()
}

// ─────────────────────────────── tests ────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dust_lcc::{parse_light_falloff, LightFalloffLimit, UniformLight};
    use crate::GrainKind;
    use std::time::Instant;

    const REAL_LCC: &[u8] = include_bytes!("../../testdata/real-lightfalloff-24x16.lcc");

    /// Max per-channel difference and the number of channels that differ by
    /// more than one 8-bit code.
    fn diff8(gpu: &[u8], cpu: &[u8]) -> (u32, usize, usize) {
        let mut max = 0u32;
        let mut over_one = 0usize;
        let mut differing = 0usize;
        for (&a, &b) in gpu.iter().zip(cpu.iter()) {
            let d = a.abs_diff(b) as u32;
            max = max.max(d);
            if d > 0 {
                differing += 1;
            }
            if d > 1 {
                over_one += 1;
            }
        }
        (max, differing, over_one)
    }

    fn gradient_rgba(w: usize, h: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(w * h * 4);
        for y in 0..h {
            for x in 0..w {
                let v = ((x * 251 + y * 97) % 256) as u8;
                out.extend_from_slice(&[v, v.wrapping_add(37), v.wrapping_mul(3), 255]);
            }
        }
        out
    }

    /// Encode a linear channel the way `develop_line_with` does.
    fn encode(v: f32) -> u8 {
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
    }

    #[test]
    fn vignette_plane_matches_the_cpu_formula() {
        let (w, h) = (321usize, 217usize);
        let region = [0.125f32, 0.0625, 0.75, 0.875];
        let cases = [
            (VignetteMethod::EllipticOnCrop, 2.0f32, [0.5, 0.5], 1.5f32),
            (VignetteMethod::Circular, -1.75, [0.5, 0.5], 1.5),
            (VignetteMethod::CircularCrop, 3.0, [0.0, 1.0], 0.8),
            (VignetteMethod::EllipticOnCrop, 0.4, [0.35, 0.7], 2.0),
        ];
        let src = gradient_rgba(w, h);
        let mut worst = 0u32;
        for (method, amount, center, aspect) in cases {
            let Some(plane) = vignette_plane(w, h, region, method, amount, aspect, center) else {
                panic!("CUDA vignette plane unavailable");
            };
            let mut gpu = src.clone();
            let mut cpu = src.clone();
            for y in 0..h {
                let fy = region[1] + (y as f32 + 0.5) / h as f32 * region[3];
                for x in 0..w {
                    let fx = region[0] + (x as f32 + 0.5) / w as f32 * region[2];
                    let g = plane[y * w + x];
                    let c = vignette_gain(fx, fy, method, amount, aspect, center);
                    for k in 0..3 {
                        let i = (y * w + x) * 4 + k;
                        let linear = src[i] as f32 / 255.0;
                        gpu[i] = encode(linear * g);
                        cpu[i] = encode(linear * c);
                    }
                }
            }
            let (max, differing, over_one) = diff8(&gpu, &cpu);
            worst = worst.max(max);
            println!(
                "vignette {method:?} amount {amount} center {center:?}: max {max}, differing {differing}, over-1 {over_one}"
            );
        }
        assert!(worst <= 1, "vignette GPU/CPU difference {worst}/255");
    }

    fn lcc_stage<'a>(grid: &'a crate::dust_lcc::LightFalloff, strength: f32, limit: LightFalloffLimit) -> UniformLight<'a> {
        UniformLight::prepare(Some(grid), true, strength, limit).expect("stage")
    }

    #[test]
    fn lcc_plane_matches_the_cpu_grid() {
        let grid = parse_light_falloff(REAL_LCC).unwrap().unwrap();
        let (w, h) = (512usize, 341usize);
        let cases = [
            (1.0f32, LightFalloffLimit::Uncapped),
            (0.5, LightFalloffLimit::Soft),
            (1.2, LightFalloffLimit::Hard),
        ];
        let src = gradient_rgba(w, h);
        let mut worst = 0u32;
        for (strength, limit) in cases {
            let stage = lcc_stage(&grid, strength, limit);
            let Some(plane) = lcc_plane(&stage.gpu_grid(), w, h) else {
                panic!("CUDA LCC plane unavailable");
            };
            let mut gpu = src.clone();
            let mut cpu = src.clone();
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 4;
                    let point = [x as f32, y as f32];
                    let g = sample_plane(&plane, w, h, point[0], point[1]);
                    let c = stage.source_gain(point);
                    for k in 0..3 {
                        let linear = src[i + k] as f32 / 255.0;
                        gpu[i + k] = encode(linear * g);
                        cpu[i + k] = encode(linear * c);
                    }
                }
            }
            let (max, differing, over_one) = diff8(&gpu, &cpu);
            worst = worst.max(max);
            println!(
                "lcc strength {strength} {limit:?}: max {max}, differing {differing}, over-1 {over_one}"
            );
        }
        assert!(worst <= 1, "LCC GPU/CPU difference {worst}/255");
    }

    #[test]
    fn grain_matches_the_cpu_stage() {
        let (w, h) = (384usize, 257usize);
        let src = gradient_rgba(w, h);
        let cases = [
            (GrainKind::FineGrain, 100.0f32, 50.0f32),
            (GrainKind::SilverRich, 70.0, 20.0),
            (GrainKind::SoftGrain, 100.0, 0.0),
            (GrainKind::SoftGrain, 90.0, 100.0),
            (GrainKind::HarshGrain, 80.0, 60.0),
        ];
        let mut worst = 0u32;
        for (kind, impact, granularity) in cases {
            let adj = FilmGrain { kind, impact, granularity, density: 0.0 };
            let mut gpu = src.clone();
            let mut cpu = src.clone();
            assert!(grain_stage(&mut gpu, w, h, &adj, 1.0), "CUDA grain unavailable");
            super::super::grain::stage(&mut cpu, w, h, &adj, 1.0);
            let (max, differing, over_one) = diff8(&gpu, &cpu);
            worst = worst.max(max);
            println!(
                "grain {kind:?} amount {impact} granularity {granularity}: max {max}, differing {differing}, over-1 {over_one}"
            );
        }
        assert!(worst <= 1, "grain GPU/CPU difference {worst}/255");
    }

    fn timed(iters: usize, mut run: impl FnMut()) -> f64 {
        let started = Instant::now();
        for _ in 0..iters {
            run();
        }
        started.elapsed().as_secs_f64() * 1000.0 / iters as f64
    }

    fn sync(gpu: &Nvidia, pool: &Pool) -> Result<(), CudaError> {
        let mut sink = [0u8; 4];
        gpu.download_buffer(&pool.stats.0, &mut sink)
    }

    /// Queue `iters` launches and wait for the device before stopping the
    /// clock, so the result is execution time rather than enqueue time.
    fn timed_synced(
        gpu: &Nvidia,
        pool: &Pool,
        iters: usize,
        mut run: impl FnMut(),
    ) -> f64 {
        let started = Instant::now();
        for _ in 0..iters {
            run();
        }
        sync(gpu, pool).expect("sync");
        started.elapsed().as_secs_f64() * 1000.0 / iters as f64
    }

    /// Kernel-only 24 MP timings. Uploads and the final download are outside
    /// the loop, so the numbers are device execution time.
    #[test]
    #[ignore = "manual 24MP CUDA benchmark"]
    fn bench_24mp() {
        let (w, h) = (6000usize, 4000usize);
        let iters = 20;
        let bytes = w * h * 4;
        let grid = parse_light_falloff(REAL_LCC).unwrap().unwrap();
        let stage = lcc_stage(&grid, 1.0, LightFalloffLimit::Soft);
        let gpu_grid = stage.gpu_grid();
        let nodes = gpu_grid.columns * gpu_grid.rows;
        let node_bytes = nodes * 12;
        let grain = FilmGrain {
            kind: GrainKind::SoftGrain,
            impact: 100.0,
            granularity: 50.0,
            density: 0.0,
        };
        let rgba = gradient_rgba(w, h);

        let mut results = Vec::new();
        with_pool(bytes, node_bytes.max(h * 8), |pool| {
            let gpu = pool.gpu.as_ref();
            let (mut w_i, mut h_i) = (w as i32, h as i32);
            let mut plane = pool.plane.0.handle();

            // Vignette kernel.
            let (mut vox, mut vsx) = (0.0f32, 1.0f32);
            let (mut voy, mut vsy) = (0.0f32, 1.0f32);
            let (mut amount, mut aspect) = (2.0f32, 1.5f32);
            let (mut cx, mut cy) = (0.5f32, 0.5f32);
            let mut method = 0i32;
            let mut plane = plane;
            let ms = timed_synced(gpu, pool, iters, || {
                launch2(gpu, "vignette_gain_kernel", w, h, [
                    super::ptr(&mut plane), super::ptr(&mut w_i), super::ptr(&mut h_i),
                    super::ptr(&mut vox), super::ptr(&mut vsx), super::ptr(&mut voy),
                    super::ptr(&mut vsy), super::ptr(&mut amount), super::ptr(&mut aspect),
                    super::ptr(&mut cx), super::ptr(&mut cy), super::ptr(&mut method),
                ]).unwrap();
            });
            results.push(("vignette", ms));

            // LCC kernel.
            let mut packed = Vec::with_capacity(nodes * 3);
            packed.extend_from_slice(&gpu_grid.luminance);
            packed.extend_from_slice(&gpu_grid.slope_x);
            packed.extend_from_slice(&gpu_grid.slope_y);
            let raw = unsafe {
                std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4)
            };
            gpu.upload_buffer(&pool.nodes.0, raw).unwrap();
            let node_handle = pool.nodes.0.handle();
            let (mut columns, mut rows) = (gpu_grid.columns as i32, gpu_grid.rows as i32);
            let (mut gox, mut goy) = (gpu_grid.origin[0], gpu_grid.origin[1]);
            let (mut gsx, mut gsy) = (gpu_grid.step[0], gpu_grid.step[1]);
            let (mut max_level, mut strength) = (gpu_grid.max_level, gpu_grid.strength);
            let mut limit = 0i32;
            let mut lum = node_handle;
            let (mut slope_x, mut slope_y) =
                (node_handle + nodes as u64 * 4, node_handle + nodes as u64 * 8);
            let mut plane2 = plane;
            let ms = timed_synced(gpu, pool, iters, || {
                launch2(gpu, "lcc_gain_kernel", w, h, [
                    super::ptr(&mut plane2), super::ptr(&mut w_i), super::ptr(&mut h_i),
                    super::ptr(&mut columns), super::ptr(&mut rows), super::ptr(&mut gox),
                    super::ptr(&mut goy), super::ptr(&mut gsx), super::ptr(&mut gsy),
                    super::ptr(&mut max_level), super::ptr(&mut strength), super::ptr(&mut limit),
                    super::ptr(&mut lum), super::ptr(&mut slope_x), super::ptr(&mut slope_y),
                ]).unwrap();
            });
            results.push(("lcc", ms));

            // Grain chain: hash, blur h, blur v, stats, apply.
            gpu.upload_buffer(&pool.rgba.0, &rgba).unwrap();
            let sigma = grain.sigma_px();
            let taps = super::super::grain::erf_taps(sigma);
            let tap_raw = unsafe {
                std::slice::from_raw_parts(taps.as_ptr() as *const u8, taps.len() * 4)
            };
            gpu.upload_buffer(&pool.taps.0, tap_raw).unwrap();
            let mut rgba_handle = pool.rgba.0.handle();
            let mut tmp = pool.tmp.0.handle();
            let mut tap_handle = pool.taps.0.handle();
            let mut r = (taps.len() / 2) as i32;
            let (ox, oy) = super::super::grain::noise_offsets();
            let (mut nox, mut noy) = (ox, oy);
            let stats = pool.nodes.0.handle();
            let mut row_sum = stats;
            let mut row_sq = stats + h as u64 * 4;
            let mut sd_out = pool.stats.0.handle();
            let mut inv_n = 1.0f32 / (w * h) as f32;
            let mut vgain = 0.1f32;
            let ms = timed_synced(gpu, pool, iters, || {
                launch2(gpu, "grain_plane_kernel", w, h, [
                    super::ptr(&mut plane), super::ptr(&mut w_i), super::ptr(&mut h_i),
                    super::ptr(&mut nox), super::ptr(&mut noy),
                ]).unwrap();
                launch2(gpu, "grain_blur_h_kernel", w, h, [
                    super::ptr(&mut plane), super::ptr(&mut tmp), super::ptr(&mut w_i),
                    super::ptr(&mut h_i), super::ptr(&mut r), super::ptr(&mut tap_handle),
                ]).unwrap();
                launch2(gpu, "grain_blur_v_kernel", w, h, [
                    super::ptr(&mut tmp), super::ptr(&mut plane), super::ptr(&mut w_i),
                    super::ptr(&mut h_i), super::ptr(&mut r), super::ptr(&mut tap_handle),
                ]).unwrap();
                gpu.launch_raw("grain_row_stats_kernel", (h as u32, 1), (256, 1), &mut [
                    super::ptr(&mut plane), super::ptr(&mut row_sum),
                    super::ptr(&mut row_sq), super::ptr(&mut w_i), super::ptr(&mut h_i),
                ]).unwrap();
                gpu.launch_raw("grain_reduce_stats_kernel", (1, 1), (256, 1), &mut [
                    super::ptr(&mut row_sum), super::ptr(&mut row_sq),
                    super::ptr(&mut h_i), super::ptr(&mut inv_n), super::ptr(&mut sd_out),
                ]).unwrap();
                launch2(gpu, "grain_v8_kernel", w, h, [
                    super::ptr(&mut rgba_handle), super::ptr(&mut plane), super::ptr(&mut w_i),
                    super::ptr(&mut h_i), super::ptr(&mut vgain),
                ]).unwrap();
            });
            results.push(("grain(v8)", ms));

            // Chained: vignette, lcc, grain in one queue.
            let ms = timed_synced(gpu, pool, iters, || {
                launch2(gpu, "vignette_gain_kernel", w, h, [
                    super::ptr(&mut plane), super::ptr(&mut w_i), super::ptr(&mut h_i),
                    super::ptr(&mut vox), super::ptr(&mut vsx), super::ptr(&mut voy),
                    super::ptr(&mut vsy), super::ptr(&mut amount), super::ptr(&mut aspect),
                    super::ptr(&mut cx), super::ptr(&mut cy), super::ptr(&mut method),
                ]).unwrap();
                launch2(gpu, "lcc_gain_kernel", w, h, [
                    super::ptr(&mut plane2), super::ptr(&mut w_i), super::ptr(&mut h_i),
                    super::ptr(&mut columns), super::ptr(&mut rows), super::ptr(&mut gox),
                    super::ptr(&mut goy), super::ptr(&mut gsx), super::ptr(&mut gsy),
                    super::ptr(&mut max_level), super::ptr(&mut strength), super::ptr(&mut limit),
                    super::ptr(&mut lum), super::ptr(&mut slope_x), super::ptr(&mut slope_y),
                ]).unwrap();
                launch2(gpu, "grain_plane_kernel", w, h, [
                    super::ptr(&mut plane), super::ptr(&mut w_i), super::ptr(&mut h_i),
                    super::ptr(&mut nox), super::ptr(&mut noy),
                ]).unwrap();
                launch2(gpu, "grain_blur_h_kernel", w, h, [
                    super::ptr(&mut plane), super::ptr(&mut tmp), super::ptr(&mut w_i),
                    super::ptr(&mut h_i), super::ptr(&mut r), super::ptr(&mut tap_handle),
                ]).unwrap();
                launch2(gpu, "grain_blur_v_kernel", w, h, [
                    super::ptr(&mut tmp), super::ptr(&mut plane), super::ptr(&mut w_i),
                    super::ptr(&mut h_i), super::ptr(&mut r), super::ptr(&mut tap_handle),
                ]).unwrap();
                gpu.launch_raw("grain_row_stats_kernel", (h as u32, 1), (256, 1), &mut [
                    super::ptr(&mut plane), super::ptr(&mut row_sum),
                    super::ptr(&mut row_sq), super::ptr(&mut w_i), super::ptr(&mut h_i),
                ]).unwrap();
                gpu.launch_raw("grain_reduce_stats_kernel", (1, 1), (256, 1), &mut [
                    super::ptr(&mut row_sum), super::ptr(&mut row_sq),
                    super::ptr(&mut h_i), super::ptr(&mut inv_n), super::ptr(&mut sd_out),
                ]).unwrap();
                launch2(gpu, "grain_v8_kernel", w, h, [
                    super::ptr(&mut rgba_handle), super::ptr(&mut plane), super::ptr(&mut w_i),
                    super::ptr(&mut h_i), super::ptr(&mut vgain),
                ]).unwrap();
            });
            results.push(("chain", ms));
            Ok(())
        })
        .expect("24MP pool");

        for (name, ms) in results {
            println!("{name}: {ms:.2} ms / 24MP");
        }
        let end_to_end = {
            let ms = |run: &mut dyn FnMut()| timed(3, || run());
            let mut plane = || {
                let _ = vignette_plane(
                    w, h, [0.0, 0.0, 1.0, 1.0],
                    VignetteMethod::EllipticOnCrop, 2.0, 1.5, [0.5, 0.5],
                );
            };
            let mut lcc = || {
                let _ = lcc_plane(&gpu_grid, w, h);
            };
            let mut grain_run = || {
                let mut buf = rgba.clone();
                grain_stage(&mut buf, w, h, &grain, 1.0);
            };
            (
                ms(&mut plane),
                ms(&mut lcc),
                ms(&mut grain_run),
            )
        };
        println!(
            "end-to-end vignette {:.2} ms, lcc {:.2} ms, grain {:.2} ms (incl. copies)",
            end_to_end.0, end_to_end.1, end_to_end.2
        );
        // Warm-cache path: a slider that does not move the gain planes.
        let _ = vignette_plane_cached(
            w, h, [0.0, 0.0, 1.0, 1.0],
            VignetteMethod::EllipticOnCrop, 2.0, 1.5, [0.5, 0.5],
        );
        let _ = lcc_plane_cached(&gpu_grid, w, h);
        let warm_vignette = timed(5, || {
            let _ = vignette_plane_cached(
                w, h, [0.0, 0.0, 1.0, 1.0],
                VignetteMethod::EllipticOnCrop, 2.0, 1.5, [0.5, 0.5],
            );
        });
        let warm_lcc = timed(5, || {
            let _ = lcc_plane_cached(&gpu_grid, w, h);
        });
        println!(
            "cached-hit vignette {warm_vignette:.3} ms, lcc {warm_lcc:.3} ms (no copy)"
        );
        let cpu_vignette = timed(1, || {
            let mut sink = 0.0f32;
            for y in 0..h {
                let fy = (y as f32 + 0.5) / h as f32;
                for x in 0..w {
                    let fx = (x as f32 + 0.5) / w as f32;
                    sink += vignette_gain(
                        fx, fy, VignetteMethod::EllipticOnCrop, 2.0, 1.5, [0.5, 0.5],
                    );
                }
            }
            std::hint::black_box(sink);
        });
        let cpu_lcc = timed(1, || {
            let mut sink = 0.0f32;
            for y in 0..h {
                for x in 0..w {
                    sink += stage.source_gain([x as f32, y as f32]);
                }
            }
            std::hint::black_box(sink);
        });
        let mut cpu_grain_buf = rgba.clone();
        let cpu_grain = timed(1, || {
            super::super::grain::stage(&mut cpu_grain_buf, w, h, &grain, 1.0);
        });
        println!(
            "CPU single-thread vignette {cpu_vignette:.2} ms, lcc {cpu_lcc:.2} ms; rayon grain {cpu_grain:.2} ms"
        );
    }

}
