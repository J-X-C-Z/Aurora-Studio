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

use aurora_portrait::cuda::DeviceBuffer;

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

/// Device slots this stage uses. `Frame` is the RGBA frame, `Scratch` the gain
/// plane, `Aux0` the grain blur temporary, `Aux2` the LCC node table or grain
/// row totals. The tiny tap array and the 4-byte statistics word are owned
/// buffers because they are uploaded/read at whole-buffer granularity.
struct SendBuffer(DeviceBuffer<'static>);

// The driver table the buffer borrows lives inside `Pool.gpu`, which outlives
// it; `State` serialises every use.
unsafe impl Send for SendBuffer {}

struct Pool {
    gpu: Gpu,
    taps: SendBuffer,
    stats: SendBuffer,
    taps_bytes: usize,
}

impl Pool {
    fn open(bytes: usize, node_bytes: usize, taps_bytes: usize) -> Result<Self, Failure> {
        let mut gpu = Gpu::open("pixel", PTX)?;
        for slot in [Slot::Frame, Slot::Scratch, Slot::Aux0] {
            gpu.reserve(slot, bytes, MAX_RETAINED_BYTES)?;
        }
        gpu.reserve(Slot::Aux2, node_bytes.max(12), MAX_RETAINED_BYTES)?;
        let taps_bytes = taps_bytes.max(68);
        let taps = SendBuffer(gpu_runtime::static_buffer(&gpu, taps_bytes)?);
        let stats = SendBuffer(gpu_runtime::static_buffer(&gpu, 4)?);
        Ok(Self { gpu, taps, stats, taps_bytes })
    }

    fn ensure(&mut self, bytes: usize, node_bytes: usize, taps_bytes: usize) -> Result<(), Failure> {
        for slot in [Slot::Frame, Slot::Scratch, Slot::Aux0] {
            self.gpu.reserve(slot, bytes, MAX_RETAINED_BYTES)?;
        }
        self.gpu.reserve(Slot::Aux2, node_bytes.max(12), MAX_RETAINED_BYTES)?;
        if taps_bytes > self.taps_bytes {
            self.taps = SendBuffer(gpu_runtime::static_buffer(&self.gpu, taps_bytes)?);
            self.taps_bytes = taps_bytes;
        }
        Ok(())
    }

    fn taps(&self) -> u64 {
        self.taps.0.handle()
    }

    fn stats(&self) -> u64 {
        self.stats.0.handle()
    }
}

enum State {
    NotTried,
    Ready(Pool),
    Unavailable,
}

/// Test-only snapshot of the pixel stage's runtime counters and kernels.
#[cfg(test)]
pub(crate) fn stage_snapshot() -> Option<(
    String,
    (i32, i32),
    super::gpu_runtime::Counters,
    Vec<(&'static str, u64)>,
)> {
    let state = state().lock().unwrap_or_else(|error| error.into_inner());
    let State::Ready(pool) = &*state else { return None };
    let (device, cc) = pool.gpu.device().unwrap_or(("unknown", (0, 0)));
    Some((
        device.to_owned(),
        cc,
        pool.gpu.counters(),
        pool.gpu.kernels().iter().map(|entry| (entry.kernel, entry.count)).collect(),
    ))
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
    taps_bytes: usize,
    body: impl FnOnce(&mut Pool) -> Result<T, Failure>,
) -> Option<T> {
    if bytes.saturating_mul(3).saturating_add(node_bytes) > MAX_RETAINED_BYTES {
        super::gpu_runtime::note_fallback(
            "pixel",
            &Failure::TooLarge {
                need: bytes.saturating_mul(3).saturating_add(node_bytes),
                cap: MAX_RETAINED_BYTES,
            },
        );
        return None;
    }
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    match &mut *state {
        State::Unavailable => None,
        State::NotTried => match Pool::open(bytes, node_bytes, taps_bytes) {
            Ok(mut pool) => {
                let result = pool
                    .ensure(bytes, node_bytes, taps_bytes)
                    .and_then(|()| body(&mut pool));
                match result {
                    Ok(value) => {
                        *state = State::Ready(pool);
                        Some(value)
                    }
                    Err(failure) => {
                        log::warn!("pixel_gpu: {failure:?}; staying on CPU");
                        super::gpu_runtime::note_fallback("pixel", &failure);
                        *state = State::Unavailable;
                        None
                    }
                }
            }
            Err(failure) => {
                log::warn!("pixel_gpu: {failure:?}; staying on CPU");
                super::gpu_runtime::note_fallback("pixel", &failure);
                *state = State::Unavailable;
                None
            }
        },
        State::Ready(pool) => match pool
            .ensure(bytes, node_bytes, taps_bytes)
            .and_then(|()| body(pool))
        {
            Ok(value) => Some(value),
            Err(failure) => {
                log::warn!("pixel_gpu: {failure:?}; staying on CPU");
                super::gpu_runtime::note_fallback("pixel", &failure);
                None
            }
        },
    }
}

use super::gpu_runtime::{self, Failure, Gpu, Slot};
use super::gpu_runtime::{grid2, ptr};

fn launch2<const N: usize>(
    gpu: &mut Gpu,
    kernel: &'static str,
    w: usize,
    h: usize,
    mut args: [*mut c_void; N],
) -> Result<(), Failure> {
    gpu.launch(kernel, grid2(w, h, BLOCK), BLOCK, &mut args)
}

/// Launches `vignette_gain_kernel` into `plane`. Shared by the production
/// entry point and the P-08 benchmark so both measure the same kernel.
#[allow(clippy::too_many_arguments)]
fn launch_vignette(
    gpu: &mut Gpu,
    plane: u64,
    width: usize,
    height: usize,
    region: [f32; 4],
    method: i32,
    amount: f32,
    aspect: f32,
    center: [f32; 2],
) -> Result<(), Failure> {
    let (mut w_i, mut h_i) = (width as i32, height as i32);
    let (mut ox, mut sx) = (region[0], region[2]);
    let (mut oy, mut sy) = (region[1], region[3]);
    let (mut amount, mut aspect) = (amount, aspect);
    let (mut cx, mut cy) = (center[0], center[1]);
    let mut method = method;
    let mut plane = plane;
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
    )
}

/// Packs the three grid arrays into the node buffer layout the kernel expects.
fn pack_lcc_nodes(grid: &GpuLccGrid, nodes: usize) -> Vec<f32> {
    let mut packed = Vec::with_capacity(nodes * 3);
    packed.extend_from_slice(&grid.luminance[..nodes]);
    packed.extend_from_slice(&grid.slope_x[..nodes]);
    packed.extend_from_slice(&grid.slope_y[..nodes]);
    packed
}

/// Launches `lcc_gain_kernel` from a node buffer holding luminance, slope_x,
/// slope_y back to back.
fn launch_lcc(
    gpu: &mut Gpu,
    plane: u64,
    node_handle: u64,
    grid: &GpuLccGrid,
    nodes: usize,
    width: usize,
    height: usize,
) -> Result<(), Failure> {
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
    let mut plane = plane;
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
    )
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

/// Production switch for the "device plane, CPU consume" shape.
///
/// P-08 measured this path end to end (generation + upload + kernel + download
/// + CPU consume) against the CPU `exp2` formula: it is 1.3–3.1× slower at
/// every tested size (256²…45 MP) because the D2H copy of a 4 B/pixel plane
/// dominates a very cheap per-pixel formula. The shipped default therefore
/// declines; `AURORA_PIXEL_VIGNETTE_GPU=1` (plus the master switch) restores it
/// for parity probes and for a future device-resident develop (P-07), where the
/// plane never comes back to host.
fn vignette_plane_shipped() -> bool {
    match std::env::var_os("AURORA_PIXEL_VIGNETTE_GPU") {
        Some(value) => !(value == "0" || value.eq_ignore_ascii_case("false")),
        None => false,
    }
}

/// Cached vignette plane. The whole-operator cost is `exp2` per pixel; the
/// cache exists because a slider drag that does not touch vignetting should not
/// recompute (or re-copy) the plane.
///
/// A cache hit is always served: the measured warm path is at or below the CPU
/// formula (0.32–1.05×). Only a cold generation is gated by
/// [`vignette_plane_shipped`].
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
    if !vignette_plane_shipped() {
        super::gpu_runtime::note_fallback(
            "pixel_vignette",
            &super::gpu_runtime::Failure::Disabled("vignette plane end-to-end"),
        );
        return None;
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
    let done = with_pool(bytes, 0, 0, |pool| {
        let plane = pool
            .gpu
            .handle(Slot::Scratch)
            .ok_or(Failure::Invalid("plane slot"))?;
        launch_vignette(&mut pool.gpu, plane, width, height, region, method, amount, aspect, center)?;
        pool.gpu.download(Slot::Scratch, out_bytes)
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
    let done = with_pool(bytes, node_bytes, 0, |pool| {
        // Three arrays in one node buffer: luminance, then slope_x, then slope_y.
        let packed = pack_lcc_nodes(grid, nodes);
        pool.gpu.upload(Slot::Aux2, node_bytes_slice(&packed))?;
        let plane = pool
            .gpu
            .handle(Slot::Scratch)
            .ok_or(Failure::Invalid("plane slot"))?;
        let nodes_handle = pool
            .gpu
            .handle(Slot::Aux2)
            .ok_or(Failure::Invalid("node slot"))?;
        launch_lcc(
            &mut pool.gpu,
            plane,
            nodes_handle,
            grid,
            nodes,
            width,
            height,
        )?;
        pool.gpu.download(Slot::Scratch, out_bytes)
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

    with_pool(bytes, node_bytes, taps.len() * 4, |pool| {
        let rgba_handle = pool
            .gpu
            .handle(Slot::Frame)
            .ok_or(Failure::Invalid("frame slot"))?;
        let tap_handle = pool.taps();
        let stats_handle = pool.stats();
        pool.gpu.upload(Slot::Frame, rgba)?;
        if !taps.is_empty() {
            let tap_slice = unsafe {
                std::slice::from_raw_parts(taps.as_ptr() as *const u8, taps.len() * 4)
            };
            pool.gpu.upload_to(&pool.taps.0, tap_slice)?;
        }
        let (mut w_i, mut h_i) = (w as i32, h as i32);
        let (mut ox, mut oy) = (ox, oy);
        let mut rgba_ptr = rgba_handle;
        let mut plane = pool
            .gpu
            .handle(Slot::Scratch)
            .ok_or(Failure::Invalid("plane slot"))?;
        let mut tmp = pool
            .gpu
            .handle(Slot::Aux0)
            .ok_or(Failure::Invalid("tmp slot"))?;
        if !composite {
            let mut gain = amount * super::grain::GRAIN_GAIN_V1;
            launch2(&mut pool.gpu, "grain_v1_kernel", w, h, [
                ptr(&mut rgba_ptr),
                ptr(&mut w_i),
                ptr(&mut h_i),
                ptr(&mut gain),
                ptr(&mut ox),
                ptr(&mut oy),
            ])?;
            pool.gpu.download(Slot::Frame, rgba)
        } else {
            launch2(
                &mut pool.gpu,
                "grain_plane_kernel",
                w,
                h,
                [ptr(&mut plane), ptr(&mut w_i), ptr(&mut h_i), ptr(&mut ox), ptr(&mut oy)],
            )?;
            if !taps.is_empty() {
                let mut r = radius;
                let mut tap_ptr = tap_handle;
                launch2(&mut pool.gpu, "grain_blur_h_kernel", w, h, [
                    ptr(&mut plane),
                    ptr(&mut tmp),
                    ptr(&mut w_i),
                    ptr(&mut h_i),
                    ptr(&mut r),
                    ptr(&mut tap_ptr),
                ])?;
                launch2(&mut pool.gpu, "grain_blur_v_kernel", w, h, [
                    ptr(&mut tmp),
                    ptr(&mut plane),
                    ptr(&mut w_i),
                    ptr(&mut h_i),
                    ptr(&mut r),
                    ptr(&mut tap_ptr),
                ])?;
            }
            // Row totals, then one block over the rows: deterministic order.
            let row_base = pool
                .gpu
                .handle(Slot::Aux2)
                .ok_or(Failure::Invalid("row totals slot"))?;
            let mut row_sum = row_base;
            let mut row_sq = row_base + h as u64 * 4;
            pool.gpu.launch(
                "grain_row_stats_kernel",
                (h as u32, 1),
                (256, 1),
                &mut [
                    ptr(&mut plane),
                    ptr(&mut row_sum),
                    ptr(&mut row_sq),
                    ptr(&mut w_i),
                    ptr(&mut h_i),
                ],
            )?;
            let mut inv_n = 1.0f32 / (w * h) as f32;
            let mut sd_out = stats_handle;
            pool.gpu.launch(
                "grain_reduce_stats_kernel",
                (1, 1),
                (256, 1),
                &mut [
                    ptr(&mut row_sum),
                    ptr(&mut row_sq),
                    ptr(&mut h_i),
                    ptr(&mut inv_n),
                    ptr(&mut sd_out),
                ],
            )?;
            let mut sd_bytes = [0u8; 4];
            pool.gpu.download_from(&pool.stats.0, &mut sd_bytes)?;
            let sd = f32::from_le_bytes(sd_bytes).max(1e-6);
            let mut gain = amount * adj.kind.v8_amplitude(adj.granularity) / (sd * 257.0);
            launch2(&mut pool.gpu, "grain_v8_kernel", w, h, [
                ptr(&mut rgba_ptr),
                ptr(&mut plane),
                ptr(&mut w_i),
                ptr(&mut h_i),
                ptr(&mut gain),
            ])?;
            pool.gpu.download(Slot::Frame, rgba)
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

    fn real_lcc() -> Option<Vec<u8>> {
        crate::testdata::bytes("testdata/real-lightfalloff-24x16.lcc")
    }

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
        let Some(real_lcc) = real_lcc() else { return; };
        let grid = parse_light_falloff(&real_lcc).unwrap().unwrap();
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

    // ────────────────── P-08: five-segment end-to-end benchmark ──────────────────

    /// 256², 1080p, 4K, 24 MP, 45 MP.
    const P08_SIZES: [(usize, usize); 5] = [
        (256, 256),
        (1920, 1080),
        (3840, 2160),
        (6048, 4032),
        (8192, 5464),
    ];

    fn p08_frame(w: usize, h: usize) -> Vec<[f32; 3]> {
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let v = ((x * 13 + y * 7) % 251) as f32 / 255.0;
                out.push([v, (v * 0.9 + 0.05).min(1.0), (v * 0.8 + 0.1).min(1.0)]);
            }
        }
        out
    }

    fn xs_table(w: usize, region: [f32; 4]) -> Vec<f32> {
        (0..w)
            .map(|x| region[0] + (x as f32 + 0.5) / w as f32 * region[2])
            .collect()
    }

    fn ms(started: std::time::Instant) -> f64 {
        started.elapsed().as_secs_f64() * 1000.0
    }

    fn p50_p95(mut values: Vec<f64>) -> (f64, f64) {
        values.sort_by(f64::total_cmp);
        let pick = |q: f64| {
            let index = ((values.len() as f64 - 1.0) * q).round() as usize;
            values[index.min(values.len() - 1)]
        };
        (pick(0.5), pick(0.95))
    }

    /// The shipped default declines the cold vignette plane and says so; the
    /// kernel itself stays callable (parity probes, future device develop).
    #[test]
    fn shipped_vignette_plane_declines_and_reports() {
        if std::env::var_os("AURORA_PIXEL_VIGNETTE_GPU").is_some() {
            eprintln!("skipping: AURORA_PIXEL_VIGNETTE_GPU is set");
            return;
        }
        super::super::gpu_runtime::reset_fallback();
        let (w, h) = (97usize, 61usize);
        let plane = vignette_plane_cached(
            w, h, [0.0, 0.0, 1.0, 1.0],
            VignetteMethod::EllipticOnCrop, 2.0, 1.5, [0.5, 0.5],
        );
        assert!(plane.is_none(), "shipped default must stay on the CPU formula");
        let report = super::super::gpu_runtime::fallback_snapshot();
        assert_eq!(
            report.last.iter().find(|(name, _)| *name == "pixel_vignette").unwrap().1,
            "disabled"
        );
        // The direct entry point is not gated: parity/tests can still ask.
        assert!(vignette_plane(
            w, h, [0.0, 0.0, 1.0, 1.0],
            VignetteMethod::EllipticOnCrop, 2.0, 1.5, [0.5, 0.5],
        )
        .is_some());
        super::super::gpu_runtime::reset_fallback();
    }

    /// Real develop-loop consumption shape: formula inline, three multiplies.
    fn cpu_vignette_fused(
        frame: &mut [[f32; 3]],
        w: usize,
        h: usize,
        region: [f32; 4],
        xs: &[f32],
        method: VignetteMethod,
        amount: f32,
        aspect: f32,
        center: [f32; 2],
    ) -> f64 {
        use rayon::prelude::*;
        let started = Instant::now();
        frame.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
            let fy = region[1] + (y as f32 + 0.5) / h as f32 * region[3];
            for (x, px) in line.iter_mut().enumerate() {
                let g = vignette_gain(xs[x], fy, method, amount, aspect, center);
                *px = [px[0] * g, px[1] * g, px[2] * g];
            }
        });
        ms(started)
    }

    fn cpu_vignette_plane(
        w: usize,
        h: usize,
        region: [f32; 4],
        xs: &[f32],
        method: VignetteMethod,
        amount: f32,
        aspect: f32,
        center: [f32; 2],
    ) -> (Vec<f32>, f64) {
        use rayon::prelude::*;
        let started = Instant::now();
        let mut plane = vec![0.0f32; w * h];
        plane.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
            let fy = region[1] + (y as f32 + 0.5) / h as f32 * region[3];
            for (x, value) in line.iter_mut().enumerate() {
                *value = vignette_gain(xs[x], fy, method, amount, aspect, center);
            }
        });
        (plane, ms(started))
    }

    fn consume_plane(frame: &mut [[f32; 3]], plane: &[f32], w: usize) -> f64 {
        use rayon::prelude::*;
        let started = Instant::now();
        frame
            .par_chunks_mut(w)
            .enumerate()
            .for_each(|(y, line)| {
                let base = y * w;
                for (x, px) in line.iter_mut().enumerate() {
                    let g = plane[base + x];
                    *px = [px[0] * g, px[1] * g, px[2] * g];
                }
            });
        ms(started)
    }

    fn consume_lcc_plane(frame: &mut [[f32; 3]], plane: &[f32], w: usize, h: usize) -> f64 {
        use rayon::prelude::*;
        let started = Instant::now();
        frame.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
            for (x, px) in line.iter_mut().enumerate() {
                let g = sample_plane(plane, w, h, x as f32, y as f32);
                *px = [px[0] * g, px[1] * g, px[2] * g];
            }
        });
        ms(started)
    }

    fn cpu_lcc_fused(
        frame: &mut [[f32; 3]],
        w: usize,
        stage: &UniformLight<'_>,
    ) -> f64 {
        use rayon::prelude::*;
        let started = Instant::now();
        frame.par_chunks_mut(w).enumerate().for_each(|(y, line)| {
            for (x, px) in line.iter_mut().enumerate() {
                let g = stage.source_gain([x as f32, y as f32]);
                *px = [px[0] * g, px[1] * g, px[2] * g];
            }
        });
        ms(started)
    }

    /// Device-side segments: kernel-only time (queued launches + one 4 B sync),
    /// then a measured plane download. Upload is measured by the caller.
    #[cfg(windows)]
    #[allow(clippy::type_complexity)]
    fn gpu_vignette_segments(
        w: usize,
        h: usize,
        region: [f32; 4],
        method: VignetteMethod,
        amount: f32,
        aspect: f32,
        center: [f32; 2],
        iters: usize,
    ) -> Option<(f64, f64, Vec<f32>)> {
        let code = match method {
            VignetteMethod::EllipticOnCrop => 0,
            VignetteMethod::Circular | VignetteMethod::CircularCrop => 1,
        };
        let bytes = w * h * 4;
        with_pool(bytes, 0, 0, |pool| {
            let mut sink = [0u8; 4];
            let idle = {
                let started = Instant::now();
                pool.gpu.download_from(&pool.stats.0, &mut sink)?;
                ms(started)
            };
            let plane = pool
            .gpu
            .handle(Slot::Scratch)
            .ok_or(Failure::Invalid("plane slot"))?;
            let started = Instant::now();
            for _ in 0..iters {
                launch_vignette(&mut pool.gpu, plane, w, h, region, code, amount, aspect, center)?;
            }
            pool.gpu.download_from(&pool.stats.0, &mut sink)?;
            let kernel = (ms(started) / iters as f64 - idle).max(0.0);
            launch_vignette(&mut pool.gpu, plane, w, h, region, code, amount, aspect, center)?;
            pool.gpu.download_from(&pool.stats.0, &mut sink)?;
            let mut out = vec![0.0f32; w * h];
            let out_bytes = unsafe {
                std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4)
            };
            let started = Instant::now();
            pool.gpu.download(Slot::Scratch, out_bytes)?;
            let download = ms(started);
            Ok((kernel, download, out))
        })
    }

    #[cfg(windows)]
    #[allow(clippy::type_complexity)]
    fn gpu_lcc_segments(
        grid: &GpuLccGrid,
        w: usize,
        h: usize,
        iters: usize,
    ) -> Option<(f64, f64, f64, Vec<f32>)> {
        let bytes = w * h * 4;
        let nodes = grid.columns * grid.rows;
        let node_bytes = nodes * 12;
        with_pool(bytes, node_bytes, 0, |pool| {
            let packed = pack_lcc_nodes(grid, nodes);
            let raw = unsafe {
                std::slice::from_raw_parts(packed.as_ptr() as *const u8, packed.len() * 4)
            };
            let started = Instant::now();
            pool.gpu.upload(Slot::Aux2, raw)?;
            let upload = ms(started);
            let mut sink = [0u8; 4];
            let idle = {
                let started = Instant::now();
                pool.gpu.download_from(&pool.stats.0, &mut sink)?;
                ms(started)
            };
            let plane = pool
            .gpu
            .handle(Slot::Scratch)
            .ok_or(Failure::Invalid("plane slot"))?;
            let nodes_handle = pool
            .gpu
            .handle(Slot::Aux2)
            .ok_or(Failure::Invalid("node slot"))?;
            let started = Instant::now();
            for _ in 0..iters {
                launch_lcc(&mut pool.gpu, plane, nodes_handle, grid, nodes, w, h)?;
            }
            pool.gpu.download_from(&pool.stats.0, &mut sink)?;
            let kernel = (ms(started) / iters as f64 - idle).max(0.0);
            launch_lcc(&mut pool.gpu, plane, nodes_handle, grid, nodes, w, h)?;
            pool.gpu.download_from(&pool.stats.0, &mut sink)?;
            let mut out = vec![0.0f32; w * h];
            let out_bytes = unsafe {
                std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out.len() * 4)
            };
            let started = Instant::now();
            pool.gpu.download(Slot::Scratch, out_bytes)?;
            let download = ms(started);
            Ok((upload, kernel, download, out))
        })
    }

    /// P-08: vignette/LCC gain plane, five segments, cold parameter change,
    /// CPU formula vs GPU plane, 256²…45 MP. Release-mode numbers only.
    ///
    /// `cargo test -p aurora-engine --release --lib bench_p08_gain_plane \
    ///  -- --ignored --nocapture --test-threads=1`
    #[cfg(windows)]
    #[test]
    #[ignore = "P-08 end-to-end gain-plane benchmark; run in Release with --ignored"]
    fn bench_p08_gain_plane() {
        use crate::testdata;
        // Measure the GPU path itself, independent of the shipped default.
        unsafe { std::env::set_var("AURORA_PIXEL_VIGNETTE_GPU", "1") };
        let region = [0.0f32, 0.0, 1.0, 1.0];
        let (method, amount, aspect, center) =
            (VignetteMethod::EllipticOnCrop, 2.0f32, 1.5f32, [0.5f32, 0.5f32]);
        let lcc_source = testdata::bytes("testdata/real-lightfalloff-24x16.lcc");
        let lcc_grid = lcc_source
            .as_deref()
            .and_then(|bytes| parse_light_falloff(bytes).ok().flatten());
        let lcc_stage = lcc_grid.as_ref().and_then(|grid| {
            UniformLight::prepare(Some(grid), true, 1.0, LightFalloffLimit::Soft)
        });
        let runs = 5usize;
        let metrics = aurora_portrait::cuda::driver_metrics();
        println!(
            "P08 start runs={runs} kernel_loads={} h2d={} d2h={}",
            metrics.ptx_module_load_calls, metrics.host_to_device_bytes, metrics.device_to_host_bytes
        );
        for (w, h) in P08_SIZES {
            let base = p08_frame(w, h);
            let xs = xs_table(w, region);
            let px = (w * h) as f64 / 1e6;
            // Warm the pool, the module and the plane cache so p50 is a
            // steady-state number (the first call pays JIT + first-touch).
            let _ = vignette_plane(w, h, region, method, amount, aspect, center);
            let _ = vignette_plane_cached(w, h, region, method, amount, aspect, center);

            // CPU: fused formula (production shape) and split generation/consume.
            let mut cpu_fused = Vec::new();
            let mut cpu_gen = Vec::new();
            let mut cpu_cons = Vec::new();
            for _ in 0..runs {
                let mut frame = base.clone();
                cpu_fused.push(cpu_vignette_fused(
                    &mut frame, w, h, region, &xs, method, amount, aspect, center,
                ));
                let (plane, gen_ms) =
                    cpu_vignette_plane(w, h, region, &xs, method, amount, aspect, center);
                cpu_gen.push(gen_ms);
                let mut frame = base.clone();
                cpu_cons.push(consume_plane(&mut frame, &plane, w));
            }
            let (cpu_fused_p50, cpu_fused_p95) = p50_p95(cpu_fused);
            let (cpu_gen_p50, _) = p50_p95(cpu_gen);
            let (cpu_cons_p50, _) = p50_p95(cpu_cons);

            // GPU: cold plane call (alloc + kernel + sync D2H) + consume.
            let mut gpu_e2e = Vec::new();
            let mut gpu_kernel = 0.0;
            let mut gpu_download = 0.0;
            let mut gpu_cons = 0.0;
            let mut gpu_plane_call = 0.0;
            for _ in 0..runs {
                let mut frame = base.clone();
                let started = Instant::now();
                let plane = vignette_plane(w, h, region, method, amount, aspect, center)
                    .expect("CUDA vignette plane");
                gpu_plane_call = ms(started);
                gpu_cons = consume_plane(&mut frame, &plane, w);
                gpu_e2e.push(gpu_plane_call + gpu_cons);
            }
            if let Some((kernel, download, _plane)) =
                gpu_vignette_segments(w, h, region, method, amount, aspect, center, 8)
            {
                gpu_kernel = kernel;
                gpu_download = download;
            }
            let (gpu_p50, gpu_p95) = p50_p95(gpu_e2e);
            // Warm cache: a slider that does not touch the plane.
            let mut warm_e2e = Vec::new();
            let mut warm_hit = 0.0;
            for _ in 0..runs {
                let mut frame = base.clone();
                let started = Instant::now();
                let plane =
                    vignette_plane_cached(w, h, region, method, amount, aspect, center)
                        .expect("cached vignette plane");
                warm_hit = ms(started);
                let consumed = consume_plane(&mut frame, &plane, w);
                warm_e2e.push(warm_hit + consumed);
            }
            let (warm_p50, _) = p50_p95(warm_e2e);
            println!(
                "P08 vignette {w}x{h} ({px:.1} MP) cpu_fused p50={cpu_fused_p50:.2} p95={cpu_fused_p95:.2} | \
cpu_gen={cpu_gen_p50:.2} consume={cpu_cons_p50:.2} | gpu kernel={gpu_kernel:.2} \
download={gpu_download:.2} consume={gpu_cons:.2} planecall={gpu_plane_call:.2} \
e2e p50={gpu_p50:.2} p95={gpu_p95:.2} | warm_hit={warm_hit:.3} warm_e2e p50={warm_p50:.2} | \
h2d=0 d2h={} | gpu/cpu={:.2} warm/cpu={:.2}",
                w * h * 4,
                gpu_p50 / cpu_fused_p50,
                warm_p50 / cpu_fused_p50,
            );

            // LCC: plane is per source pixel, so 1:1 frame == source size.
            if let Some(stage) = lcc_stage.as_ref() {
                let gpu_grid = stage.gpu_grid();
                let _ = lcc_plane(&gpu_grid, w, h);
                let _ = lcc_plane_cached(&gpu_grid, w, h);
                let mut cpu_fused = Vec::new();
                for _ in 0..runs {
                    let mut frame = base.clone();
                    cpu_fused.push(cpu_lcc_fused(&mut frame, w, stage));
                }
                let (cpu_p50, cpu_p95) = p50_p95(cpu_fused);
                let mut gpu_e2e = Vec::new();
                let mut gpu_cons = 0.0;
                let mut gpu_call = 0.0;
                for _ in 0..runs {
                    let mut frame = base.clone();
                    let started = Instant::now();
                    let plane = lcc_plane(&gpu_grid, w, h).expect("CUDA LCC plane");
                    gpu_call = ms(started);
                    gpu_cons = consume_lcc_plane(&mut frame, &plane, w, h);
                    gpu_e2e.push(gpu_call + gpu_cons);
                }
                let segments = gpu_lcc_segments(&gpu_grid, w, h, 8);
                let (upload, kernel, download) = segments
                    .as_ref()
                    .map(|(u, k, d, _)| (*u, *k, *d))
                    .unwrap_or((0.0, 0.0, 0.0));
                let (gpu_p50, gpu_p95) = p50_p95(gpu_e2e);
                let mut warm_e2e = Vec::new();
                let mut warm_hit = 0.0;
                for _ in 0..runs {
                    let mut frame = base.clone();
                    let started = Instant::now();
                    let plane =
                        lcc_plane_cached(&gpu_grid, w, h).expect("cached LCC plane");
                    warm_hit = ms(started);
                    let consumed = consume_lcc_plane(&mut frame, &plane, w, h);
                    warm_e2e.push(warm_hit + consumed);
                }
                let (warm_p50, _) = p50_p95(warm_e2e);
                println!(
                    "P08 lcc {w}x{h} ({px:.1} MP) cpu_fused p50={cpu_p50:.2} p95={cpu_p95:.2} | \
gpu upload={upload:.2} kernel={kernel:.2} download={download:.2} consume={gpu_cons:.2} \
planecall={gpu_call:.2} e2e p50={gpu_p50:.2} p95={gpu_p95:.2} | warm_hit={warm_hit:.3} \
warm_e2e p50={warm_p50:.2} | h2d={} d2h={} | gpu/cpu={:.2} warm/cpu={:.2}",
                    gpu_grid.columns * gpu_grid.rows * 12,
                    w * h * 4,
                    gpu_p50 / cpu_p50,
                    warm_p50 / cpu_p50,
                );
            }
        }

        // Preview shape: a 24 MP LCC plane consumed at 1080p output vs the CPU
        // grid formula at 1080p (what `linear_edits` really runs today).
        if let Some(stage) = lcc_stage.as_ref() {
            let gpu_grid = stage.gpu_grid();
            let (sw, sh) = (6048usize, 4032usize);
            let (ow, oh) = (1920usize, 1080usize);
            let out = p08_frame(ow, oh);
            let _ = lcc_plane(&gpu_grid, sw, sh);
            let _ = lcc_plane_cached(&gpu_grid, sw, sh);
            let mut cpu = Vec::new();
            for _ in 0..5 {
                let mut frame = out.clone();
                cpu.push(cpu_lcc_fused(&mut frame, ow, stage));
            }
            let mut gpu = Vec::new();
            for _ in 0..5 {
                let mut frame = out.clone();
                let started = Instant::now();
                let plane = lcc_plane(&gpu_grid, sw, sh).expect("CUDA LCC plane");
                let produced = ms(started);
                let consumed = consume_lcc_plane(&mut frame, &plane, ow, oh);
                gpu.push(produced + consumed);
            }
            let mut warm = Vec::new();
            for _ in 0..5 {
                let mut frame = out.clone();
                let started = Instant::now();
                let plane = lcc_plane_cached(&gpu_grid, sw, sh).expect("cached LCC plane");
                let hit = ms(started);
                let consumed = consume_lcc_plane(&mut frame, &plane, ow, oh);
                warm.push(hit + consumed);
            }
            let (cpu_p50, _) = p50_p95(cpu);
            let (gpu_p50, _) = p50_p95(gpu);
            let (warm_p50, _) = p50_p95(warm);
            println!(
                "P08 lcc preview 24MP plane -> 1080p out: cpu_grid p50={cpu_p50:.2} gpu p50={gpu_p50:.2} \
warm p50={warm_p50:.2} cold_ratio={:.2} warm_ratio={:.2}",
                gpu_p50 / cpu_p50,
                warm_p50 / cpu_p50
            );
        }

        if let Some((device, cc, counters, kernels)) = stage_snapshot() {
            println!(
                "P08 runtime device={device} cc={cc:?} uploads={} downloads={} h2d_bytes={} d2h_bytes={} launches={} syncs={} kernels={kernels:?}",
                counters.uploads,
                counters.downloads,
                counters.h2d_bytes,
                counters.d2h_bytes,
                counters.launches,
                counters.syncs,
            );
        }
    }
}
