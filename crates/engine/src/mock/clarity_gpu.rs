//! Optional CUDA path for Clarity and Structure.
//!
//! The kernels mirror `mock/clarity.rs`: the two low-frequency planes are
//! block-averaged and bilinearly upsampled on the device, the per-pixel gain is
//! applied there, and Structure's 5x5 box is a separable device blur. Every
//! entry point returns `false` when it cannot run, and the caller then takes
//! the CPU stage. `AURORA_CLARITY_GPU=0` disables the module outright.
//!
//! The grid (or frame luma) still comes from the caller: sampling the RAW
//! pipeline needs the source tap plus `develop_pre_curve`, which stays on the
//! CPU. Only the operator geometry moves to the device.
//!
//! Device handles come from the shared [`super::gpu_runtime::Gpu`]: one module,
//! grow-only `Frame`/`Scratch` slots, accounted transfers and a reported
//! fallback reason.

use std::ffi::c_void;

use super::gpu_runtime::{self, Failure, Gpu, Slot};
use super::clarity;
use crate::ClarityMethod;

const PTX: &str = include_str!("../../native/cuda/clarity_gpu.ptx");
const BLOCK: (u32, u32) = (16, 16);
const PREWARM_BYTES: usize = 1400 * 933 * 20;
/// Keep one 24-32 MP frame resident so slider refinements do not reallocate;
/// anything larger is allocated per call and released.
const MAX_RETAINED_BYTES: usize = 768 * 1024 * 1024;
const STRUCTURE_RADIUS: usize = 2;

struct Pool {
    gpu: Gpu,
}

impl Pool {
    fn open(frame: usize, scratch: usize) -> Result<Self, Failure> {
        let mut gpu = Gpu::open("clarity", PTX)?;
        reserve(&mut gpu, frame, scratch, MAX_RETAINED_BYTES)?;
        Ok(Self { gpu })
    }

    fn ensure(&mut self, frame: usize, scratch: usize) -> Result<(), Failure> {
        reserve(&mut self.gpu, frame, scratch, MAX_RETAINED_BYTES)
    }
}

fn reserve(
    gpu: &mut Gpu,
    frame: usize,
    scratch: usize,
    cap: usize,
) -> Result<(), Failure> {
    gpu.reserve(Slot::Frame, frame, cap)?;
    gpu.reserve(Slot::Scratch, scratch, cap)?;
    Ok(())
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

use std::sync::{Mutex, OnceLock};

/// Defaults to on; `AURORA_CLARITY_GPU=0` forces the CPU stage.
pub(crate) fn enabled() -> bool {
    match std::env::var_os("AURORA_CLARITY_GPU") {
        Some(value) => value != "0",
        None => true,
    }
}

pub(crate) fn prewarm() -> bool {
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    match &mut *state {
        State::Ready(_) => true,
        State::Unavailable => false,
        State::NotTried => match Pool::open(PREWARM_BYTES, PREWARM_BYTES) {
            Ok(pool) => {
                *state = State::Ready(pool);
                log::info!("CUDA clarity path ready");
                true
            }
            Err(failure) => {
                log::warn!("clarity_gpu: {failure:?}; staying on CPU");
                gpu_runtime::note_fallback("clarity", &failure);
                *state = State::Unavailable;
                false
            }
        },
    }
}

/// Runs `body` with a persistent frame/scratch pair, or with throw-away buffers
/// when the request exceeds [`MAX_RETAINED_BYTES`].
fn with_pool<T>(
    frame: usize,
    scratch: usize,
    body: impl FnOnce(&mut Gpu) -> Result<T, Failure>,
) -> Result<T, Failure> {
    let result = if frame.saturating_add(scratch) > MAX_RETAINED_BYTES {
        let mut gpu = Gpu::open("clarity", PTX)?;
        reserve(&mut gpu, frame, scratch, usize::MAX)?;
        body(&mut gpu)
    } else {
        let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
        match &mut *state {
            State::Unavailable => Err(Failure::Unavailable("clarity_gpu")),
            State::NotTried => {
                let mut pool = Pool::open(frame, scratch)?;
                pool.ensure(frame, scratch)?;
                let result = body(&mut pool.gpu);
                *state = State::Ready(pool);
                result
            }
            State::Ready(pool) => {
                pool.ensure(frame, scratch)?;
                body(&mut pool.gpu)
            }
        }
    };
    if let Err(failure) = &result {
        log::warn!("clarity_gpu: {failure:?}; staying on CPU");
        gpu_runtime::note_fallback("clarity", failure);
    }
    result
}

fn ptr<T>(value: &mut T) -> *mut c_void {
    value as *mut T as *mut c_void
}

fn launch2<const N: usize>(
    gpu: &mut Gpu,
    kernel: &'static str,
    w: usize,
    h: usize,
    mut args: [*mut c_void; N],
) -> Result<(), Failure> {
    gpu.launch(kernel, gpu_runtime::grid2(w, h, BLOCK), BLOCK, &mut args)
}

fn plane_rows(cols: usize, gw: usize, gh: usize) -> usize {
    ((cols as f32 * gh as f32 / gw as f32).round() as usize).clamp(1, gh)
}

fn plane_cols(gw: usize, gh: usize, step: usize, w: usize, h: usize) -> (usize, usize, usize, usize) {
    let fine = clarity::cols_for(gw, step, clarity::fine_cell(w));
    let coarse = clarity::cols_for(gw, step, clarity::coarse_cell(w, h));
    (fine, plane_rows(fine, gw, gh), coarse, plane_rows(coarse, gw, gh))
}

/// Bytes of the two upsampled planes plus their small block-average cells.
fn planes_bytes(gw: usize, gh: usize, w: usize, h: usize, step: usize) -> usize {
    let (fc, fr, cc, cr) = plane_cols(gw, gh, step, w, h);
    gw * gh * 4 * 2 + fr * fc * 4 + cr * cc * 4
}

/// Builds both low-frequency planes in `scratch` from the grid at `scratch[0]`.
///
/// Returns `(fine_offset, coarse_offset)` relative to the scratch buffer.
#[allow(clippy::too_many_arguments)]
fn build_planes(
    gpu: &mut Gpu,
    scratch_base: u64,
    grid_off: usize,
    gw: usize,
    gh: usize,
    w: usize,
    h: usize,
    step: usize,
    compensate: i32,
) -> Result<(usize, usize), Failure> {
    let gbytes = gw * gh * 4;
    let (fine_cols, fine_rows, coarse_cols, coarse_rows) = plane_cols(gw, gh, step, w, h);
    let fine_plane = grid_off + gbytes;
    let small_fine = fine_plane + gbytes;
    let coarse_plane = small_fine + fine_rows * fine_cols * 4;
    let small_coarse = coarse_plane + gbytes;

    let mut gw_i = gw as i32;
    let mut gh_i = gh as i32;
    let mut fine_cols_i = fine_cols as i32;
    let mut fine_rows_i = fine_rows as i32;
    let mut coarse_cols_i = coarse_cols as i32;
    let mut coarse_rows_i = coarse_rows as i32;
    let mut compensate_i = compensate;
    let mut p_grid = scratch_base + grid_off as u64;
    let mut p_small = scratch_base + small_fine as u64;
    let mut p_plane = scratch_base + fine_plane as u64;
    gpu.launch(
        "clarity_blockavg_kernel",
        (fine_cols.div_ceil(16) as u32, fine_rows.div_ceil(16) as u32),
        BLOCK,
        &mut [
            ptr(&mut p_grid),
            ptr(&mut p_small),
            ptr(&mut gw_i),
            ptr(&mut gh_i),
            ptr(&mut fine_cols_i),
            ptr(&mut fine_rows_i),
            ptr(&mut compensate_i),
        ],
    )?;
    launch2(
        gpu,
        "clarity_upsample_kernel",
        gw,
        gh,
        [
            ptr(&mut p_small),
            ptr(&mut p_plane),
            ptr(&mut gw_i),
            ptr(&mut gh_i),
            ptr(&mut fine_cols_i),
            ptr(&mut fine_rows_i),
        ],
    )?;

    let mut p_small = scratch_base + small_coarse as u64;
    let mut p_plane = scratch_base + coarse_plane as u64;
    gpu.launch(
        "clarity_blockavg_kernel",
        (coarse_cols.div_ceil(16) as u32, coarse_rows.div_ceil(16) as u32),
        BLOCK,
        &mut [
            ptr(&mut p_grid),
            ptr(&mut p_small),
            ptr(&mut gw_i),
            ptr(&mut gh_i),
            ptr(&mut coarse_cols_i),
            ptr(&mut coarse_rows_i),
            ptr(&mut compensate_i),
        ],
    )?;
    launch2(
        gpu,
        "clarity_upsample_kernel",
        gw,
        gh,
        [
            ptr(&mut p_small),
            ptr(&mut p_plane),
            ptr(&mut gw_i),
            ptr(&mut gh_i),
            ptr(&mut coarse_cols_i),
            ptr(&mut coarse_rows_i),
        ],
    )?;
    Ok((fine_plane, coarse_plane))
}

/// Applies Clarity to a pre-curve f32 RGB frame using the CPU-sampled grid.
///
/// `rgb` is the `Vec<[f32; 3]>` layout the develop loop writes. The grid is
/// uploaded as-is, so the device planes are the same block averages the CPU
/// `PreCurve` holds.
pub(crate) fn apply_precurve(
    rgb: &mut [[f32; 3]],
    w: usize,
    h: usize,
    grid: &clarity::Grid,
    amount: f32,
    method: ClarityMethod,
) -> bool {
    precurve_runs(rgb, w, h, grid, amount, method, 1).is_ok()
}

/// Kernel-only timing for the pre-curve gain (upload/build/download excluded).
#[cfg(test)]
pub(crate) fn bench_precurve_kernel(
    rgb: &mut [[f32; 3]],
    w: usize,
    h: usize,
    grid: &clarity::Grid,
    amount: f32,
    method: ClarityMethod,
) -> Option<f64> {
    precurve_runs(rgb, w, h, grid, amount, method, 16).ok()
}

fn precurve_runs(
    rgb: &mut [[f32; 3]],
    w: usize,
    h: usize,
    grid: &clarity::Grid,
    amount: f32,
    method: ClarityMethod,
    runs: usize,
) -> Result<f64, Failure> {
    let pixels = w * h;
    if pixels == 0 || rgb.len() < pixels || grid.raw.len() < grid.gw * grid.gh {
        return Err(Failure::Invalid("precurve frame"));
    }
    let grid_bytes = grid.gw * grid.gh * 4;
    let scratch = grid_bytes + planes_bytes(grid.gw, grid.gh, grid.w, grid.h, grid.step);
    let frame = pixels * 12;
    let (k_big, k_fine, chroma) = clarity::PreCurve::params(amount, method);
    let grid_raw = &grid.raw;
    let grid_u8 = unsafe {
        std::slice::from_raw_parts(grid_raw.as_ptr() as *const u8, grid_raw.len() * 4)
    };
    let rgb_u8 = unsafe {
        std::slice::from_raw_parts_mut(rgb.as_mut_ptr() as *mut u8, pixels * 12)
    };
    with_pool(frame, scratch, |gpu| {
        gpu.upload(Slot::Scratch, grid_u8)?;
        let scratch_base = gpu.handle(Slot::Scratch).ok_or(Failure::Invalid("scratch slot"))?;
        let frame_handle = gpu.handle(Slot::Frame).ok_or(Failure::Invalid("frame slot"))?;
        let (fine_off, coarse_off) =
            build_planes(gpu, scratch_base, 0, grid.gw, grid.gh, grid.w, grid.h, grid.step, 2)?;
        gpu.upload(Slot::Frame, rgb_u8)?;
        let mut p_rgb = frame_handle;
        let mut p_fine = scratch_base + fine_off as u64;
        let mut p_coarse = scratch_base + coarse_off as u64;
        let mut w_i = w as i32;
        let mut h_i = h as i32;
        let mut gw_i = grid.gw as i32;
        let mut gh_i = grid.gh as i32;
        let mut step_f = grid.step as f32;
        let mut k_big_f = k_big;
        let mut k_fine_f = k_fine;
        let mut chroma_f = chroma;
        let start = std::time::Instant::now();
        for _ in 0..runs.max(1) {
            launch2(
                gpu,
                "clarity_gain_f32_kernel",
                w,
                h,
                [
                    ptr(&mut p_rgb),
                    ptr(&mut p_fine),
                    ptr(&mut p_coarse),
                    ptr(&mut w_i),
                    ptr(&mut h_i),
                    ptr(&mut gw_i),
                    ptr(&mut gh_i),
                    ptr(&mut step_f),
                    ptr(&mut k_big_f),
                    ptr(&mut k_fine_f),
                    ptr(&mut chroma_f),
                ],
            )?;
        }
        gpu.download(Slot::Frame, rgb_u8)?;
        Ok(start.elapsed().as_secs_f64() * 1000.0 / runs.max(1) as f64)
    })
}

/// Structure on an 8-bit RGBA frame. `false` means the caller must run the CPU
/// stage.
pub(crate) fn apply_structure_u8(rgba: &mut [u8], w: usize, h: usize, structure: f32) -> bool {
    structure_u8_runs(rgba, w, h, structure, 1).is_ok()
}

/// Kernel-only timing for Structure (upload/download excluded).
#[cfg(test)]
pub(crate) fn bench_structure_kernel(
    rgba: &mut [u8],
    w: usize,
    h: usize,
    structure: f32,
) -> Option<f64> {
    structure_u8_runs(rgba, w, h, structure, 16).ok()
}

fn structure_u8_runs(
    rgba: &mut [u8],
    w: usize,
    h: usize,
    structure: f32,
    runs: usize,
) -> Result<f64, Failure> {
    let pixels = w * h;
    if structure.abs() < 1e-4 || pixels == 0 || rgba.len() < pixels * 4 {
        return Err(Failure::Invalid("structure frame"));
    }
    let frame = pixels * 4;
    let scratch = pixels * 12;
    let k = 0.0028 * structure;
    with_pool(frame, scratch, |gpu| {
        gpu.upload(Slot::Frame, rgba)?;
        let frame_handle = gpu.handle(Slot::Frame).ok_or(Failure::Invalid("frame slot"))?;
        let scratch_base = gpu.handle(Slot::Scratch).ok_or(Failure::Invalid("scratch slot"))?;
        let mut p_rgba = frame_handle;
        let mut p_y = scratch_base;
        let mut p_tmp = scratch_base + (pixels * 4) as u64;
        let mut p_low = scratch_base + (pixels * 8) as u64;
        let mut w_i = w as i32;
        let mut h_i = h as i32;
        let mut r_i = STRUCTURE_RADIUS as i32;
        let mut k_f = k;
        let start = std::time::Instant::now();
        for _ in 0..runs.max(1) {
            launch2(
                gpu,
                "clarity_luma8_kernel",
                w,
                h,
                [ptr(&mut p_rgba), ptr(&mut p_y), ptr(&mut w_i), ptr(&mut h_i)],
            )?;
            launch2(
                gpu,
                "clarity_box_h_kernel",
                w,
                h,
                [ptr(&mut p_y), ptr(&mut p_tmp), ptr(&mut w_i), ptr(&mut h_i), ptr(&mut r_i)],
            )?;
            launch2(
                gpu,
                "clarity_box_v_kernel",
                w,
                h,
                [ptr(&mut p_tmp), ptr(&mut p_low), ptr(&mut w_i), ptr(&mut h_i), ptr(&mut r_i)],
            )?;
            launch2(
                gpu,
                "clarity_structure8_kernel",
                w,
                h,
                [
                    ptr(&mut p_rgba),
                    ptr(&mut p_y),
                    ptr(&mut p_low),
                    ptr(&mut w_i),
                    ptr(&mut h_i),
                    ptr(&mut k_f),
                ],
            )?;
        }
        gpu.download(Slot::Frame, rgba)?;
        Ok(start.elapsed().as_secs_f64() * 1000.0 / runs.max(1) as f64)
    })
}

/// Structure on a 16-bit linear RGBA frame; the 8-bit preview twin is written
/// as well, matching `clarity::shade_frame`.
pub(crate) fn apply_structure_wide(
    rgba8: &mut [u8],
    rgba16: &mut [u16],
    w: usize,
    h: usize,
    structure: f32,
) -> bool {
    let pixels = w * h;
    if structure.abs() < 1e-4
        || pixels == 0
        || rgba8.len() < pixels * 4
        || rgba16.len() < pixels * 4
    {
        return false;
    }
    let frame = pixels * 8;
    let scratch = pixels * 16;
    let k = 0.0028 * structure;
    let wide_bytes = unsafe {
        std::slice::from_raw_parts_mut(rgba16.as_mut_ptr() as *mut u8, pixels * 8)
    };
    with_pool(frame, scratch, |gpu| {
        gpu.upload(Slot::Frame, wide_bytes)?;
        gpu.upload(Slot::Scratch, rgba8)?;
        let frame_handle = gpu.handle(Slot::Frame).ok_or(Failure::Invalid("frame slot"))?;
        let scratch_base = gpu.handle(Slot::Scratch).ok_or(Failure::Invalid("scratch slot"))?;
        let mut p_wide = frame_handle;
        let mut p_rgba = scratch_base;
        let mut p_y = scratch_base + (pixels * 4) as u64;
        let mut p_tmp = scratch_base + (pixels * 8) as u64;
        let mut p_low = scratch_base + (pixels * 12) as u64;
        let mut w_i = w as i32;
        let mut h_i = h as i32;
        let mut r_i = STRUCTURE_RADIUS as i32;
        launch2(
            gpu,
            "clarity_luma16_kernel",
            w,
            h,
            [ptr(&mut p_wide), ptr(&mut p_y), ptr(&mut w_i), ptr(&mut h_i)],
        )?;
        launch2(
            gpu,
            "clarity_box_h_kernel",
            w,
            h,
            [ptr(&mut p_y), ptr(&mut p_tmp), ptr(&mut w_i), ptr(&mut h_i), ptr(&mut r_i)],
        )?;
        launch2(
            gpu,
            "clarity_box_v_kernel",
            w,
            h,
            [ptr(&mut p_tmp), ptr(&mut p_low), ptr(&mut w_i), ptr(&mut h_i), ptr(&mut r_i)],
        )?;
        let mut k_f = k;
        launch2(
            gpu,
            "clarity_structure16_kernel",
            w,
            h,
            [
                ptr(&mut p_wide),
                ptr(&mut p_rgba),
                ptr(&mut p_y),
                ptr(&mut p_low),
                ptr(&mut w_i),
                ptr(&mut h_i),
                ptr(&mut k_f),
            ],
        )?;
        gpu.download(Slot::Frame, wide_bytes)?;
        gpu.download(Slot::Scratch, rgba8)?;
        Ok(())
    })
    .is_ok()
}

/// Frame-level Clarity on an 8-bit RGBA frame. This is the standalone operator
/// twin of `clarity::legacy_apply`, used for parity probes and for callers that
/// already hold a display-referred frame.
pub(crate) fn apply_frame_u8(
    rgba: &mut [u8],
    w: usize,
    h: usize,
    amount: f32,
    method: ClarityMethod,
) -> bool {
    frame_u8_runs(rgba, w, h, amount, method, 1).is_ok()
}

#[cfg(test)]
pub(crate) fn bench_frame_u8_kernel(
    rgba: &mut [u8],
    w: usize,
    h: usize,
    amount: f32,
    method: ClarityMethod,
) -> Option<f64> {
    frame_u8_runs(rgba, w, h, amount, method, 8).ok()
}

fn frame_u8_runs(
    rgba: &mut [u8],
    w: usize,
    h: usize,
    amount: f32,
    method: ClarityMethod,
    runs: usize,
) -> Result<f64, Failure> {
    let pixels = w * h;
    if amount.abs() < 1e-4 || pixels == 0 || rgba.len() < pixels * 4 {
        return Err(Failure::Invalid("frame clarity"));
    }
    let frame = pixels * 4;
    let scratch = pixels * 4 + planes_bytes(w, h, w, h, 1);
    let (k_big, k_fine, chroma) = clarity::PreCurve::params(amount, method);
    with_pool(frame, scratch, |gpu| {
        gpu.upload(Slot::Frame, rgba)?;
        let frame_handle = gpu.handle(Slot::Frame).ok_or(Failure::Invalid("frame slot"))?;
        let scratch_base = gpu.handle(Slot::Scratch).ok_or(Failure::Invalid("scratch slot"))?;
        let mut p_rgba = frame_handle;
        let mut p_grid = scratch_base;
        let mut w_i = w as i32;
        let mut h_i = h as i32;
        launch2(
            gpu,
            "clarity_luma8_kernel",
            w,
            h,
            [ptr(&mut p_rgba), ptr(&mut p_grid), ptr(&mut w_i), ptr(&mut h_i)],
        )?;
        let (fine_off, coarse_off) =
            build_planes(gpu, scratch_base, 0, w, h, w, h, 1, 0)?;
        let mut p_fine = scratch_base + fine_off as u64;
        let mut p_coarse = scratch_base + coarse_off as u64;
        let mut gw_i = w as i32;
        let mut gh_i = h as i32;
        let mut step_f = 1.0f32;
        let mut k_big_f = k_big;
        let mut k_fine_f = k_fine;
        let mut chroma_f = chroma;
        let start = std::time::Instant::now();
        for _ in 0..runs.max(1) {
            launch2(
                gpu,
                "clarity_gain_rgba8_kernel",
                w,
                h,
                [
                    ptr(&mut p_rgba),
                    ptr(&mut p_fine),
                    ptr(&mut p_coarse),
                    ptr(&mut w_i),
                    ptr(&mut h_i),
                    ptr(&mut gw_i),
                    ptr(&mut gh_i),
                    ptr(&mut step_f),
                    ptr(&mut k_big_f),
                    ptr(&mut k_fine_f),
                    ptr(&mut chroma_f),
                ],
            )?;
        }
        gpu.download(Slot::Frame, rgba)?;
        Ok(start.elapsed().as_secs_f64() * 1000.0 / runs.max(1) as f64)
    })
}

/// Frame-level Clarity on a 16-bit linear RGBA frame plus its 8-bit twin.
pub(crate) fn apply_frame_wide(
    rgba8: &mut [u8],
    rgba16: &mut [u16],
    w: usize,
    h: usize,
    amount: f32,
    method: ClarityMethod,
) -> bool {
    frame_wide_runs(rgba8, rgba16, w, h, amount, method, 1).is_ok()
}

#[cfg(test)]
pub(crate) fn bench_frame_wide_kernel(
    rgba8: &mut [u8],
    rgba16: &mut [u16],
    w: usize,
    h: usize,
    amount: f32,
    method: ClarityMethod,
) -> Option<f64> {
    frame_wide_runs(rgba8, rgba16, w, h, amount, method, 8).ok()
}

fn frame_wide_runs(
    rgba8: &mut [u8],
    rgba16: &mut [u16],
    w: usize,
    h: usize,
    amount: f32,
    method: ClarityMethod,
    runs: usize,
) -> Result<f64, Failure> {
    let pixels = w * h;
    if amount.abs() < 1e-4
        || pixels == 0
        || rgba8.len() < pixels * 4
        || rgba16.len() < pixels * 4
    {
        return Err(Failure::Invalid("frame clarity 16"));
    }
    let frame = pixels * 8;
    let scratch = pixels * 12 + planes_bytes(w, h, w, h, 1);
    let (k_big, k_fine, chroma) = clarity::PreCurve::params(amount, method);
    let wide_bytes = unsafe {
        std::slice::from_raw_parts_mut(rgba16.as_mut_ptr() as *mut u8, pixels * 8)
    };
    with_pool(frame, scratch, |gpu| {
        gpu.upload(Slot::Frame, wide_bytes)?;
        gpu.upload(Slot::Scratch, rgba8)?;
        let frame_handle = gpu.handle(Slot::Frame).ok_or(Failure::Invalid("frame slot"))?;
        let scratch_base = gpu.handle(Slot::Scratch).ok_or(Failure::Invalid("scratch slot"))?;
        let mut p_wide = frame_handle;
        let mut p_rgba = scratch_base;
        let mut p_grid = scratch_base + (pixels * 4) as u64;
        let mut w_i = w as i32;
        let mut h_i = h as i32;
        launch2(
            gpu,
            "clarity_luma16_kernel",
            w,
            h,
            [ptr(&mut p_wide), ptr(&mut p_grid), ptr(&mut w_i), ptr(&mut h_i)],
        )?;
        let grid_off = pixels * 4;
        let (fine_off, coarse_off) =
            build_planes(gpu, scratch_base, grid_off, w, h, w, h, 1, 2)?;
        let mut p_fine = scratch_base + fine_off as u64;
        let mut p_coarse = scratch_base + coarse_off as u64;
        let mut gw_i = w as i32;
        let mut gh_i = h as i32;
        let mut step_f = 1.0f32;
        let mut k_big_f = k_big;
        let mut k_fine_f = k_fine;
        let mut chroma_f = chroma;
        let start = std::time::Instant::now();
        for _ in 0..runs.max(1) {
            launch2(
                gpu,
                "clarity_gain_rgba16_kernel",
                w,
                h,
                [
                    ptr(&mut p_wide),
                    ptr(&mut p_rgba),
                    ptr(&mut p_fine),
                    ptr(&mut p_coarse),
                    ptr(&mut w_i),
                    ptr(&mut h_i),
                    ptr(&mut gw_i),
                    ptr(&mut gh_i),
                    ptr(&mut step_f),
                    ptr(&mut k_big_f),
                    ptr(&mut k_fine_f),
                    ptr(&mut chroma_f),
                ],
            )?;
        }
        gpu.download(Slot::Frame, wide_bytes)?;
        gpu.download(Slot::Scratch, rgba8)?;
        Ok(start.elapsed().as_secs_f64() * 1000.0 / runs.max(1) as f64)
    })
}

/// One H2D and one D2H of `bytes`, for separating kernel time from PCIe time
/// in the benchmark. Test-only.
#[cfg(test)]
pub(crate) fn transfer_bench(bytes: usize) -> Option<(f64, f64)> {
    let mut gpu = Gpu::open("clarity", PTX).ok()?;
    gpu.reserve(Slot::Frame, bytes, usize::MAX).ok()?;
    // Touch both sides and warm the path once: a cold `vec!` would measure
    // page faults, not PCIe.
    let data = vec![1u8; bytes];
    let mut out = vec![2u8; bytes];
    gpu.upload(Slot::Frame, &data).ok()?;
    gpu.download(Slot::Frame, &mut out).ok()?;
    let start = std::time::Instant::now();
    gpu.upload(Slot::Frame, &data).ok()?;
    let upload = start.elapsed().as_secs_f64() * 1000.0;
    let start = std::time::Instant::now();
    gpu.download(Slot::Frame, &mut out).ok()?;
    let download = start.elapsed().as_secs_f64() * 1000.0;
    Some((upload, download))
}
