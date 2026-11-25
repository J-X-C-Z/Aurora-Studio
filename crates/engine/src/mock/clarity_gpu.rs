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
//!
//! P-07 step 2: the pre-curve frame and its grid/planes are device-resident
//! [`FrameHandle`]s charged to the shared VRAM ledger (see
//! [`super::gpu_runtime::VramLedger`]). A repeated call with a byte-identical
//! grid skips the grid H2D and the four plane kernels; the frame H2D/D2H stay,
//! because the develop tone chain between stages runs on the CPU.

use std::ffi::c_void;

use super::gpu_runtime::{self, Failure, FrameFormat, FrameHandle, FrameLayout, Gpu, Slot};
use super::clarity;
use crate::ClarityMethod;

const PTX: &str = include_str!("../../native/cuda/clarity_gpu.ptx");
const BLOCK: (u32, u32) = (16, 16);
const STRUCTURE_RADIUS: usize = 2;
/// Reclaim priority: interactive Clarity goes before detail/pixel but after
/// develop (which owns the process main path).
const RECLAIM_PRIORITY: u8 = 20;

/// The retained cap is now the shared ledger budget, not a private 768 MiB
/// constant: one allocation cannot exceed what the whole engine may hold.
fn retained_cap() -> usize {
    gpu_runtime::vram_budget()
}

/// Identity of the grid a resident [`ResidentPlanes`] was built from. Planes are
/// a pure function of the grid, the geometry and the compensation step, so a
/// byte-identical grid under the same key produces byte-identical planes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlaneKey {
    gw: usize,
    gh: usize,
    w: usize,
    h: usize,
    step: usize,
    compensate: i32,
}

/// Device-resident grid plus the two low-frequency planes built from it.
///
/// This is what P-07 actually removes today: a repeated Clarity call with an
/// unchanged grid (a clarity-amount-only slider move) skips the grid H2D and
/// the four block-average/upsample kernels. The frame H2D/D2H still happen
/// every render because the develop tone chain runs on the CPU between stages.
struct ResidentPlanes {
    key: PlaneKey,
    buffer: FrameHandle,
    fine_off: usize,
    coarse_off: usize,
    /// Exact host mirror of the uploaded grid, used for the reuse test.
    host_grid: Vec<f32>,
}

impl ResidentPlanes {
    fn bytes(&self) -> usize {
        self.buffer.bytes()
    }
}

/// Device-resident Clarity frame for one layout.
///
/// `FrameHandle` keeps the f32 frame (12 B/px) in the ledger with an explicit
/// generation instead of an anonymous slot offset, so a later cross-stage fusion
/// can pass the same device pointer on. Its resident planes hang off the same
/// owner and are released with it.
struct ClarityFrame {
    handle: FrameHandle,
}

struct Pool {
    gpu: Gpu,
    frame: Option<ClarityFrame>,
    planes: Option<ResidentPlanes>,
}

impl Pool {
    fn open(frame: usize, scratch: usize) -> Result<Self, Failure> {
        let mut gpu = Gpu::open("clarity", PTX)?;
        reserve(&mut gpu, frame, scratch, retained_cap())?;
        Ok(Self { gpu, frame: None, planes: None })
    }

    /// Grows the code-owned slots. It never touches the resident planes: those
    /// live in their own `FrameHandle`, so the structure / 8-bit paths writing
    /// `Slot::Scratch` cannot alias them.
    fn ensure(&mut self, frame: usize, scratch: usize) -> Result<(), Failure> {
        reserve(&mut self.gpu, frame, scratch, retained_cap())
    }

    fn resident_bytes(&self) -> usize {
        self.gpu.resident_bytes()
            + self.planes.as_ref().map_or(0, ResidentPlanes::bytes)
            + self.frame.as_ref().map_or(0, |frame| frame.handle.lease_bytes())
    }

    /// Ensures the f32 frame handle matches `(w, h, bytes)`; a new layout drops
    /// the old handle first so its bytes return before the new admission.
    fn ensure_frame(&mut self, w: usize, h: usize, bytes: usize) -> Result<(), Failure> {
        let matches = self.frame.as_ref().is_some_and(|frame| {
            let layout = frame.handle.layout();
            layout.format == FrameFormat::F32Rgb12
                && layout.width as usize == w
                && layout.height as usize == h
                && layout.bytes == bytes
        });
        if matches {
            return Ok(());
        }
        self.frame = None;
        let layout = FrameLayout {
            width: w as u32,
            height: h as u32,
            format: FrameFormat::F32Rgb12,
            bytes,
            generation: 0,
            // M1 defaults: `alloc` stamps generation/identity/residency; the
            // remaining descriptor fields keep their pre-M1 meaning.
            ..FrameLayout::default()
        };
        self.frame = Some(ClarityFrame { handle: self.gpu.frame(layout)? });
        Ok(())
    }

    fn frame_handle(&self) -> Result<u64, Failure> {
        self.frame
            .as_ref()
            .map(|frame| frame.handle.handle())
            .ok_or(Failure::Invalid("clarity frame handle"))
    }

    fn planes_handle(&self) -> Result<u64, Failure> {
        self.planes
            .as_ref()
            .map(|planes| planes.buffer.handle())
            .ok_or(Failure::Invalid("clarity planes handle"))
    }

    fn upload_frame(&mut self, data: &[u8]) -> Result<(), Failure> {
        match self.frame.as_ref() {
            Some(frame) => self.gpu.upload_frame(&frame.handle, data),
            None => Err(Failure::Invalid("clarity frame handle")),
        }
    }

    fn download_frame(&mut self, data: &mut [u8]) -> Result<(), Failure> {
        match self.frame.as_ref() {
            Some(frame) => self.gpu.download_frame(&frame.handle, data),
            None => Err(Failure::Invalid("clarity frame handle")),
        }
    }

    /// Ensures the grid and its two planes are resident. Returns
    /// `(fine_off, coarse_off)` relative to the plane buffer and whether the
    /// resident copy was reused.
    fn ensure_planes(
        &mut self,
        grid: &clarity::Grid,
        compensate: i32,
    ) -> Result<(usize, usize, bool), Failure> {
        let step = grid.step;
        let key = PlaneKey {
            gw: grid.gw,
            gh: grid.gh,
            w: grid.w,
            h: grid.h,
            step,
            compensate,
        };
        if let Some(planes) = self.planes.as_ref() {
            if planes.key == key && planes.host_grid.as_slice() == grid.raw.as_slice() {
                return Ok((planes.fine_off, planes.coarse_off, true));
            }
        }
        // Drop the old residency before admitting the new one.
        self.planes = None;
        let grid_bytes = grid.gw * grid.gh * 4;
        let bytes = grid_bytes + planes_bytes(grid.gw, grid.gh, grid.w, grid.h, step);
        let layout = FrameLayout {
            width: grid.w as u32,
            height: grid.h as u32,
            format: FrameFormat::F32Planes,
            bytes,
            generation: 0,
            // M1 defaults: `alloc` stamps generation/identity/residency; the
            // remaining descriptor fields keep their pre-M1 meaning.
            ..FrameLayout::default()
        };
        let buffer = self.gpu.frame(layout)?;
        let grid_raw = &grid.raw;
        let grid_u8 = unsafe {
            std::slice::from_raw_parts(grid_raw.as_ptr() as *const u8, grid_raw.len() * 4)
        };
        self.gpu.upload_frame(&buffer, grid_u8)?;
        let base = buffer.handle();
        let (fine_off, coarse_off) = build_planes(
            &mut self.gpu,
            base,
            0,
            grid.gw,
            grid.gh,
            grid.w,
            grid.h,
            step,
            compensate,
        )?;
        self.planes = Some(ResidentPlanes {
            key,
            buffer,
            fine_off,
            coarse_off,
            host_grid: grid.raw.clone(),
        });
        Ok((fine_off, coarse_off, false))
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
    register_reclaimer();
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    match &mut *state {
        State::Ready(_) => true,
        State::Unavailable => false,
        State::NotTried => match Pool::open(0, 0) {
            Ok(pool) => {
                *state = State::Ready(pool);
                log::info!("CUDA clarity path ready");
                true
            }
            Err(failure) => {
                log::warn!("clarity_gpu: {failure:?}; staying on CPU");
                gpu_runtime::note_fallback("clarity", &failure);
                if !matches!(failure, Failure::TooLarge { .. }) {
                    *state = State::Unavailable;
                }
                false
            }
        },
    }
}

/// Test-only view of this stage's counters, real kernels and resident bytes.
#[cfg(test)]
pub(crate) fn stage_snapshot() -> Option<(gpu_runtime::Counters, Vec<(&'static str, u64)>, usize)> {
    let state = state().lock().unwrap_or_else(|error| error.into_inner());
    let State::Ready(pool) = &*state else { return None };
    let kernels = pool
        .gpu
        .kernels()
        .iter()
        .map(|entry| (entry.kernel, entry.count))
        .collect();
    Some((pool.gpu.counters(), kernels, pool.resident_bytes()))
}

/// Test-only: the device the Clarity kernels actually ran on.
#[cfg(test)]
pub(crate) fn stage_device() -> Option<String> {
    let state = state().lock().unwrap_or_else(|error| error.into_inner());
    let State::Ready(pool) = &*state else { return None };
    pool.gpu.device().map(|(name, _)| name.to_owned())
}

/// Test-only: drops the retained pool so the residency tests start clean.
#[cfg(test)]
pub(crate) fn release_retained_for_test() -> usize {
    release_retained()
}

/// Drops this stage's retained device buffers so another stage can use the
/// budget. Uses `try_lock` only: a busy stage is in use and must not be touched,
/// and a reclaim hook must never block behind a render.
fn release_retained() -> usize {
    let mut state = match state().try_lock() {
        Ok(state) => state,
        Err(_) => return 0,
    };
    let freed = match &*state {
        State::Ready(pool) => pool.resident_bytes(),
        _ => 0,
    };
    if freed > 0 {
        *state = State::NotTried;
    }
    freed
}

fn register_reclaimer() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        gpu_runtime::vram_register_reclaimer("clarity", RECLAIM_PRIORITY, release_retained)
    });
}

/// Runs `body` with a persistent pool whose frame and planes are
/// device-resident `FrameHandle`s.
///
/// Admission is one non-allocating check against the shared ledger before any
/// frame-sized work exists; a refusal (or a lost device) leaves the caller on
/// the CPU path with nothing allocated.
fn with_pool<T>(
    frame: usize,
    scratch: usize,
    admission: usize,
    body: impl FnOnce(&mut Pool) -> Result<T, Failure>,
) -> Result<T, Failure> {
    register_reclaimer();
    let result = if !gpu_runtime::vram_can_allocate("clarity", admission) {
        Err(Failure::TooLarge { need: admission, cap: retained_cap() })
    } else {
        let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
        match &mut *state {
            State::Unavailable => Err(Failure::Unavailable("clarity_gpu")),
            State::NotTried => match Pool::open(frame, scratch) {
                Ok(mut pool) => {
                    let result = pool.ensure(frame, scratch).and_then(|()| body(&mut pool));
                    if result.is_ok() {
                        *state = State::Ready(pool);
                    }
                    result
                }
                Err(failure) => {
                    // A budget refusal is transient: retry next render instead
                    // of disabling Clarity for the process.
                    if !matches!(failure, Failure::TooLarge { .. }) {
                        *state = State::Unavailable;
                    }
                    Err(failure)
                }
            },
            State::Ready(pool) => pool.ensure(frame, scratch).and_then(|()| body(pool)),
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
    // The frame and the grid/planes are device-resident `FrameHandle`s, so the
    // pool's own slots stay unused here (0 bytes) and nothing is duplicated.
    let frame = pixels * 12;
    let resident = frame
        + grid.gw * grid.gh * 4
        + planes_bytes(grid.gw, grid.gh, grid.w, grid.h, grid.step);
    let (k_big, k_fine, chroma) = clarity::PreCurve::params(amount, method);
    let rgb_u8 = unsafe {
        std::slice::from_raw_parts_mut(rgb.as_mut_ptr() as *mut u8, pixels * 12)
    };
    let result = with_pool(0, 0, resident, |pool| {
        pool.ensure_frame(w, h, frame)?;
        let (fine_off, coarse_off, reused) = pool.ensure_planes(grid, 2)?;
        if reused {
            // Grid H2D plus blockavg x2 and upsample x2 were skipped.
            pool.gpu.note_skipped_upload();
            pool.gpu.note_skipped_plane_builds(4);
        }
        let mut p_rgb = pool.frame_handle()?;
        let planes = pool.planes_handle()?;
        let mut p_fine = planes + fine_off as u64;
        let mut p_coarse = planes + coarse_off as u64;
        pool.upload_frame(rgb_u8)?;
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
                &mut pool.gpu,
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
        pool.download_frame(rgb_u8)?;
        pool.gpu.note_fused_frame();
        Ok(start.elapsed().as_secs_f64() * 1000.0 / runs.max(1) as f64)
    });
    // One throttled `AURORA_PERF {"event":"gpu_vram",...}` line when replaying.
    gpu_runtime::vram_maybe_emit_perf("clarity");
    result
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
    with_pool(frame, scratch, frame.saturating_add(scratch), |pool| {
        let gpu = &mut pool.gpu;
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
    with_pool(frame, scratch, frame.saturating_add(scratch), |pool| {
        let gpu = &mut pool.gpu;
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
    with_pool(frame, scratch, frame.saturating_add(scratch), |pool| {
        let gpu = &mut pool.gpu;
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
    with_pool(frame, scratch, frame.saturating_add(scratch), |pool| {
        let gpu = &mut pool.gpu;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic pre-curve luma on a `step`-spaced grid.
    fn chart_grid(w: usize, h: usize, step: usize) -> clarity::Grid {
        clarity::Grid::sample(w, h, step, |gx, gy| {
            let x = (gx * step) as f32 / w as f32;
            let y = (gy * step) as f32 / h as f32;
            Some(0.18 + 0.64 * (0.5 + 0.5 * (x * 9.0 + 0.3).sin() * (y * 7.0 + 0.7).cos()))
        })
        .expect("grid")
    }

    fn chart_frame(w: usize, h: usize) -> Vec<[f32; 3]> {
        (0..w * h)
            .map(|i| {
                let x = (i % w) as f32 / w as f32;
                let y = (i / w) as f32 / h as f32;
                [0.15 + 0.6 * x, 0.2 + 0.55 * y, 0.3 + 0.4 * (x * y)]
            })
            .collect()
    }

    fn code8(v: f32) -> i32 {
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as i32
    }

    fn max_drift(cpu: &[[f32; 3]], planes: &clarity::PreCurve, frame: &[[f32; 3]], w: usize, h: usize) -> i32 {
        let mut max = 0;
        for y in 0..h {
            for x in 0..w {
                let expected = planes.apply(cpu[y * w + x], x, y);
                for k in 0..3 {
                    max = max.max((code8(expected[k]) - code8(frame[y * w + x][k])).abs());
                }
            }
        }
        max
    }

    /// A device `FrameHandle` must reproduce the CPU `PreCurve::apply` planes
    /// within the Clarity budget (the kernels mirror the CPU arithmetic).
    #[test]
    fn precurve_frame_handle_matches_the_cpu_planes() {
        let (w, h, step) = (192usize, 160usize, 8usize);
        let grid = chart_grid(w, h, step);
        let src = chart_frame(w, h);
        let (amount, method) = (80.0f32, ClarityMethod::Neutral);
        let mut gpu = src.clone();
        if !apply_precurve(&mut gpu, w, h, &grid, amount, method) {
            eprintln!("skipping precurve parity: CUDA clarity unavailable");
            return;
        }
        let planes = grid.planes(amount, method).expect("cpu planes");
        let max = max_drift(&src, &planes, &gpu, w, h);
        eprintln!("precurve frame handle vs CPU planes: max {max} code8");
        assert!(max <= 1, "frame handle drifted {max} code8 from the CPU planes");
    }

    /// A second call with a byte-identical grid must reuse the resident planes:
    /// no grid H2D and no block-average/upsample kernels, with the same pixels.
    #[test]
    fn repeated_clarity_reuses_the_resident_grid() {
        let (w, h, step) = (160usize, 128usize, 8usize);
        let grid = chart_grid(w, h, step);
        let src = chart_frame(w, h);
        let mut first = src.clone();
        if !apply_precurve(&mut first, w, h, &grid, 60.0, ClarityMethod::Punch) {
            eprintln!("skipping residency reuse: CUDA clarity unavailable");
            return;
        }
        let opened = stage_snapshot().expect("clarity pool");
        let blockavg = |snapshot: &(gpu_runtime::Counters, Vec<(&'static str, u64)>, usize)| {
            snapshot
                .1
                .iter()
                .filter(|(kernel, _)| *kernel == "clarity_blockavg_kernel")
                .map(|(_, count)| *count)
                .sum::<u64>()
        };
        let opened_blockavg = blockavg(&opened);

        let mut second = src.clone();
        assert!(apply_precurve(&mut second, w, h, &grid, 35.0, ClarityMethod::Punch));
        let reused = stage_snapshot().expect("clarity pool");
        assert!(
            reused.0.skipped_uploads > opened.0.skipped_uploads,
            "grid upload was not skipped for an identical grid"
        );
        assert!(
            reused.0.skipped_plane_builds >= opened.0.skipped_plane_builds + 4,
            "block-average/upsample kernels ran again for an identical grid"
        );
        assert_eq!(blockavg(&reused), opened_blockavg, "planes were rebuilt anyway");

        let planes = grid.planes(35.0, ClarityMethod::Punch).expect("cpu planes");
        let max = max_drift(&src, &planes, &second, w, h);
        assert!(max <= 1, "reused planes drifted {max} code8");

        // A different grid must invalidate the resident planes again.
        let changed = clarity::Grid {
            w: grid.w,
            h: grid.h,
            gw: grid.gw,
            gh: grid.gh,
            step: grid.step,
            raw: grid.raw.iter().map(|value| (value * 1.01).min(1.0)).collect(),
        };
        let mut third = src.clone();
        assert!(apply_precurve(&mut third, w, h, &changed, 35.0, ClarityMethod::Punch));
        let rebuilt = stage_snapshot().expect("clarity pool");
        assert!(blockavg(&rebuilt) > opened_blockavg, "a new grid did not rebuild the planes");
    }

    /// Switching images replaces the handle instead of stacking buffers, and a
    /// released pool gives every ledger byte back.
    #[test]
    fn image_switching_bounds_residency_and_release_frees_it() {
        release_retained_for_test();
        let baseline = gpu_runtime::vram_stats().resident_bytes;
        let layouts = [(96usize, 72usize), (160, 120), (96, 72)];
        let mut last = 0usize;
        for (w, h) in layouts {
            let grid = chart_grid(w, h, 8);
            let mut frame = chart_frame(w, h);
            if !apply_precurve(&mut frame, w, h, &grid, 50.0, ClarityMethod::Neutral) {
                eprintln!("skipping residency bounds: CUDA clarity unavailable");
                return;
            }
            last = stage_snapshot().map_or(0, |snapshot| snapshot.2);
        }
        let resident = gpu_runtime::vram_stats().resident_bytes;
        assert!(resident > baseline, "the resident pool was not charged to the ledger");
        assert_eq!(
            resident - baseline,
            last,
            "residency stacked old image buffers instead of replacing them"
        );
        let freed = release_retained_for_test();
        assert_eq!(freed, last, "the pool reported bytes the ledger did not hold");
        assert_eq!(gpu_runtime::vram_stats().resident_bytes, baseline, "release leaked bytes");
    }

    /// Real-device 24 MP / 45 MP residency probe for P-07 step 2. Ignored by
    /// default: it needs a CUDA device and more than 1 GiB of host memory.
    ///
    /// Not an end-to-end preview benchmark: it measures the Clarity segment
    /// (frame H2D, grid/plane residency, gain kernel, frame D2H) on a synthetic
    /// frame, not the IN-P Release baseline.
    #[test]
    #[ignore = "needs a CUDA device; run with --ignored --nocapture --test-threads=1"]
    fn bench_p07_clarity_residency() {
        for (w, h) in [(6048usize, 4032usize), (8192usize, 5464usize)] {
            let src = chart_frame(w, h);
            let grid_a = chart_grid(w, h, 8);
            // Warm-up only: first-touch of the host frame, module load and the
            // first FrameHandle allocation must not be charged to either row.
            let mut warmup = src.clone();
            if !apply_precurve(&mut warmup, w, h, &grid_a, 50.0, ClarityMethod::Neutral) {
                eprintln!("bench_p07: CUDA clarity unavailable, {w}x{h} skipped");
                return;
            }
            drop(warmup);
            // A different grid content forces the rebuild path; repeating it
            // then measures the resident-plane reuse with the same host pages.
            let grid_b = clarity::Grid {
                w: grid_a.w,
                h: grid_a.h,
                gw: grid_a.gw,
                gh: grid_a.gh,
                step: grid_a.step,
                raw: grid_a.raw.iter().map(|value| (value * 1.013).min(1.0)).collect(),
            };
            let before = stage_snapshot().expect("clarity pool").0;
            let mut frame = src.clone();
            let started = std::time::Instant::now();
            assert!(apply_precurve(&mut frame, w, h, &grid_b, 60.0, ClarityMethod::Neutral));
            let rebuild_ms = started.elapsed().as_secs_f64() * 1000.0;
            let rebuilt = stage_snapshot().expect("clarity pool").0;
            let mut frame = src.clone();
            let started = std::time::Instant::now();
            assert!(apply_precurve(&mut frame, w, h, &grid_b, 45.0, ClarityMethod::Neutral));
            let reuse_ms = started.elapsed().as_secs_f64() * 1000.0;
            let reused = stage_snapshot().expect("clarity pool").0;
            let stats = gpu_runtime::vram_stats();
            let device = stage_device().unwrap_or_else(|| "unknown".to_string());
            println!(
                "P07 clarity {w}x{h} device={device}: rebuild-grid {rebuild_ms:.2} ms / reuse {reuse_ms:.2} ms | \
h2d rebuild {} B reuse {} B | d2h rebuild {} B reuse {} B | skipped_uploads {} plane_builds {} | \
resident {:.1} MiB peak {:.1} MiB evictions {}",
                rebuilt.h2d_bytes - before.h2d_bytes,
                reused.h2d_bytes - rebuilt.h2d_bytes,
                rebuilt.d2h_bytes - before.d2h_bytes,
                reused.d2h_bytes - rebuilt.d2h_bytes,
                reused.skipped_uploads - before.skipped_uploads,
                reused.skipped_plane_builds - before.skipped_plane_builds,
                stats.resident_bytes as f64 / (1024.0 * 1024.0),
                stats.peak_bytes as f64 / (1024.0 * 1024.0),
                stats.eviction_count,
            );
        }
    }
}
