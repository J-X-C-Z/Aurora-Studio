//! Optional CUDA path for the detail stages.
//!
//! It mirrors `stage_luminance_nr`, `stage_color_nr`, `stage_moire` and
//! `stage_sharpening` byte-for-byte; the CPU path stays the default and is used
//! whenever the driver, PTX or `AURORA_DETAIL_GPU` is absent.
//!
//! Device memory comes from the shared [`super::gpu_runtime::Gpu`]: one
//! `Frame` slot sized from the stages that actually run, one LUT slot,
//! grow-only and accounted through the shared VRAM ledger (`VramLease`).

use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

use super::gpu_runtime::{self, Failure, Gpu, Slot};

use crate::{FilmGrain, LensCorrection, Moire, NoiseReduction, Sharpening};

const PTX: &str = include_str!("../../native/cuda/detail_gpu.ptx");
const BLOCK: (u32, u32) = (16, 16);
const PREWARM_BYTES: usize = 1400 * 933 * 4;
/// Planes a frame can touch: RGBA, luma, blur and one scratch plane. Colour NR
/// rewrites the finished luma/blur planes with chroma, and lens keeps its
/// source in the scratch plane, so four is the production maximum.
const MAX_PLANES: usize = 4;
/// One 32 MP RGBA frame: the retention boundary is the frame, not
/// `bytes × planes`, so plane sizing never raises the largest frame kept
/// resident.
const MAX_RETAINED_FRAME_BYTES: usize = 32 * 1024 * 1024 * 4;
/// A retained 32 MP frame at four planes: the grow-only slot's hard cap.
const MAX_RETAINED_BYTES: usize = MAX_RETAINED_FRAME_BYTES * MAX_PLANES;
/// Retained cap is the shared ledger budget, not a private 768 MiB constant.
fn retained_cap() -> usize {
    gpu_runtime::vram_budget()
}
/// Reclaimed before clarity but after develop.
const RECLAIM_PRIORITY: u8 = 10;
/// Device bytes a `pixels`-frame at `planes` active planes reserves: the RGBA
/// frame times the planes, plus the 1025-entry luma LUT. `None` on overflow.
fn retained_bytes(pixels: usize, planes: usize) -> Option<usize> {
    pixels
        .checked_mul(4)?
        .checked_mul(planes)?
        .checked_add(1025 * 4)
}
/// Whether a frame is small enough to keep a grow-only pool for. The boundary
/// is the frame itself: a 45 MP frame takes the CPU fallback even when only two
/// planes are active, so subtracting planes cannot pin a larger frame.
fn retained_frame(bytes: usize) -> bool {
    bytes != 0 && bytes <= MAX_RETAINED_FRAME_BYTES
}

struct DetailPool {
    gpu: Gpu,
}

impl DetailPool {
    fn open(bytes: usize, planes: usize) -> Result<Self, Failure> {
        let mut gpu = Gpu::open("detail", PTX)?;
        reserve(&mut gpu, bytes, planes, retained_cap())?;
        Ok(Self { gpu })
    }

    fn ensure(&mut self, bytes: usize, planes: usize) -> Result<(), Failure> {
        reserve(&mut self.gpu, bytes, planes, retained_cap())
    }
}

/// Grows the grow-only `Frame` slot to `bytes × planes` and the LUT slot. Both
/// reservations become `VramLease`s in the shared ledger; a refused growth is a
/// `TooLarge` the caller answers by staying on the CPU.
fn reserve(gpu: &mut Gpu, bytes: usize, planes: usize, cap: usize) -> Result<(), Failure> {
    gpu.reserve(Slot::Frame, bytes.saturating_mul(planes), cap)?;
    gpu.reserve(Slot::Aux0, 1025 * 4, cap)?;
    Ok(())
}

enum State {
    NotTried,
    Ready(DetailPool),
    Unavailable,
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::NotTried))
}

pub(crate) fn enabled() -> bool {
    // Legacy 8-bit CUDA PTX kernels (`luminance_nr_kernel`, `color_nr_kernel`,
    // `sharpening_kernel`) predate the center-weighted edge-preserving luma NR,
    // C1 ColorNoise2012, and separable Gaussian linear USM; keep opt-in only.
    match std::env::var_os("AURORA_DETAIL_GPU") {
        Some(value) => value == "1",
        None => false,
    }
}

/// Drops this stage's retained pools so another stage can use the budget.
/// `try_lock` only: a busy stage is in use and must not be touched.
fn release_retained() -> usize {
    let mut state = match state().try_lock() {
        Ok(state) => state,
        Err(_) => return 0,
    };
    let freed = match &*state {
        State::Ready(pool) => pool.gpu.resident_bytes(),
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
        gpu_runtime::vram_register_reclaimer("detail", RECLAIM_PRIORITY, release_retained)
    });
}

pub(crate) fn prewarm() -> bool {
    register_reclaimer();
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    match &mut *state {
        State::Ready(_) => true,
        State::Unavailable => false,
        State::NotTried => match DetailPool::open(PREWARM_BYTES, MAX_PLANES) {
            Ok(pool) => {
                *state = State::Ready(pool);
                log::info!("CUDA detail path ready");
                true
            }
            Err(failure) => {
                log::warn!("detail_gpu: {failure:?}; staying on CPU");
                gpu_runtime::note_fallback("detail", &failure);
                *state = State::Unavailable;
                false
            }
        },
    }
}

/// Runs the NE + colour NR + moire + sharpening + lens stages on the GPU.
/// `false` means the caller must use the CPU stages (no driver, no context, or
/// an invalid frame).
pub(crate) fn post_process(
    rgba: &mut [u8],
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    moire: &Moire,
    sh: &Sharpening,
    lens: &LensCorrection,
    scale: f32,
) -> bool {
    let Some(pixels) = w.checked_mul(h) else { return false };
    let bytes = pixels.saturating_mul(4);
    if bytes == 0 || rgba.len() != bytes {
        return false;
    }
    let planes = active_planes(nr, moire, sh, lens);
    if planes == 0 {
        // Every stage the GPU path implements is off; the caller already ran
        // the CPU-only stages, so the frame is finished.
        return true;
    }
    // Non-allocating admission against the shared ledger before any device
    // memory exists; a refusal is a CPU fallback, not a broken stage. Above the
    // retained frame boundary the pool would be oversized after one call, so
    // the frame goes to the CPU the same way.
    if !retained_frame(bytes) {
        return false;
    }
    let Some(need) = retained_bytes(pixels, planes) else { return false };
    if need > retained_cap() || !gpu_runtime::vram_can_allocate("detail", need) {
        gpu_runtime::note_fallback(
            "detail",
            &Failure::TooLarge { need, cap: retained_cap() },
        );
        return false;
    }
    register_reclaimer();
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    let result = match &mut *state {
        State::Unavailable => Err(Failure::Unavailable("detail_gpu")),
        State::NotTried => match DetailPool::open(bytes, planes) {
            Ok(mut pool) => {
                let result = pool
                    .ensure(bytes, planes)
                    .and_then(|()| post_process_pool(&mut pool.gpu, rgba, w, h, nr, moire, sh, lens, scale));
                if result.is_ok() {
                    *state = State::Ready(pool);
                }
                result
            }
            Err(failure) => {
                // Baseline semantics: a failed first open disables the stage
                // for the process instead of retrying the driver every frame.
                *state = State::Unavailable;
                Err(failure)
            }
        },
        State::Ready(pool) => pool
            .ensure(bytes, planes)
            .and_then(|()| post_process_pool(&mut pool.gpu, rgba, w, h, nr, moire, sh, lens, scale)),
    };
    match result {
        Ok(()) => true,
        Err(failure) => {
            log::warn!("detail_gpu: {failure:?}; staying on CPU");
            gpu_runtime::note_fallback("detail", &failure);
            false
        }
    }
}

/// Same as [`post_process`] but on a caller-owned context (benchmarks/tests).
pub(crate) fn post_process_with(
    gpu: &mut Gpu,
    rgba: &mut [u8],
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    moire: &Moire,
    sh: &Sharpening,
    lens: &LensCorrection,
    scale: f32,
) -> Result<(), Failure> {
    let Some(pixels) = w.checked_mul(h) else { return Ok(()) };
    if pixels == 0 || rgba.len() != pixels.saturating_mul(4) {
        return Ok(());
    }
    let planes = active_planes(nr, moire, sh, lens);
    if planes == 0 {
        return Ok(());
    }
    reserve(gpu, pixels * 4, planes, usize::MAX)?;
    post_process_pool(gpu, rgba, w, h, nr, moire, sh, lens, scale)
}

/// Planes the active stages touch. Mirrors [`post_process_pool`]'s stage
/// flags; pure so the sizing is testable without a GPU.
fn active_planes(
    nr: &NoiseReduction,
    moire: &Moire,
    sh: &Sharpening,
    lens: &LensCorrection,
) -> usize {
    plane_count(
        nr.luminance > 0.0 || nr.single_pixel > 0.0,
        nr.color > 0.0,
        moire.amount > 0.0,
        sh.amount > 0.0,
        lens.distortion != 0.0
            || lens.light_falloff != 0.0
            || lens.chromatic_aberration
            || lens.purple_fringing > 0.0,
    )
}

/// Luma-only stages (NE/moire) build luma plus one blur/scratch plane; colour
/// NR, sharpening and lens also run box blurs against a second plane.
fn plane_count(lum: bool, color: bool, moire: bool, sharp: bool, lens: bool) -> usize {
    if color || sharp || lens {
        4
    } else if lum || moire {
        2
    } else {
        0
    }
}

/// Bench entry point: reserves the slots it needs on `gpu`.
#[cfg(test)]
pub(crate) fn post_process_bench(
    gpu: &mut Gpu,
    rgba: &mut [u8],
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    moire: &Moire,
    sh: &Sharpening,
    lens: &LensCorrection,
    scale: f32,
) -> Result<(), Failure> {
    post_process_with(gpu, rgba, w, h, nr, moire, sh, lens, scale)
}

/// Uploads `rgba` into `Slot::Frame` and runs whichever detail kernels are
/// active. `Ok(None)` means nothing was active (or the shape is invalid): no
/// upload happened and the slot does not hold this frame. `Ok(Some(base))` is
/// the device handle of the post-detail RGBA8 frame; every kernel writes back
/// to `p_rgba = base`, so the result is always at the slot base.
fn launch_detail_chain(
    gpu: &mut Gpu,
    rgba: &mut [u8],
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    moire: &Moire,
    sh: &Sharpening,
    lens: &LensCorrection,
    scale: f32,
) -> Result<Option<u64>, Failure> {
    let lum_active = nr.luminance > 0.0 || nr.single_pixel > 0.0;
    let color_active = nr.color > 0.0;
    let moire_active = moire.amount > 0.0;
    let sharp_active = sh.amount > 0.0;
    let lens_active = lens.distortion != 0.0
        || lens.light_falloff != 0.0
        || lens.chromatic_aberration
        || lens.purple_fringing > 0.0;
    if !lum_active && !color_active && !moire_active && !sharp_active && !lens_active {
        return Ok(None);
    }
    let planes = plane_count(lum_active, color_active, moire_active, sharp_active, lens_active);
    let Some(pixels) = w.checked_mul(h) else { return Ok(None) };
    if pixels == 0 || rgba.len() != pixels.saturating_mul(4) {
        return Ok(None);
    }
    let bytes = pixels * 4;
    if gpu.slot_bytes(Slot::Frame) < bytes.saturating_mul(planes) {
        return Err(Failure::Invalid("detail frame slot smaller than active planes"));
    }

    let base = gpu.handle(Slot::Frame).ok_or(Failure::Invalid("detail frame slot"))?;
    let mut p_rgba = base;
    let mut p_luma = base + bytes as u64;
    let mut p_blur = base + (bytes * 2) as u64;
    let mut p_tmp = base + (bytes * 3) as u64;
    gpu.upload(Slot::Frame, rgba)?;

    let (mut w_i, mut h_i) = (w as i32, h as i32);

    if lum_active {
        launch2(
            gpu,
            "luma_plane_kernel",
            w,
            h,
            [ptr(&mut p_rgba), ptr(&mut p_luma), ptr(&mut w_i), ptr(&mut h_i)],
        )?;

        let strength = (nr.luminance / 100.0).clamp(0.0, 1.0);
        let details = (nr.details / 100.0).clamp(0.0, 1.0);
        let sigma = ((4.0 + 60.0 * strength) / (1.0 + 3.0 * details)).max(0.5);
        let two_sigma2 = 2.0 * sigma * sigma;
        let lut: Vec<f32> = (0..1025)
            .map(|i| {
                let d = i as f32 * 0.25;
                (-(d * d) / two_sigma2).exp()
            })
            .collect();
        let lut_bytes = unsafe {
            std::slice::from_raw_parts(lut.as_ptr() as *const u8, lut.len() * 4)
        };
        gpu.upload(Slot::Aux0, lut_bytes)?;

        let mut p_lut = gpu.handle(Slot::Aux0).ok_or(Failure::Invalid("detail lut slot"))?;
        let mut strength_f = strength;
        let mut single_f = (nr.single_pixel / 100.0).clamp(0.0, 1.0);
        launch2(
            gpu,
            "luminance_nr_kernel",
            w,
            h,
            [
                ptr(&mut p_rgba),
                ptr(&mut p_luma),
                ptr(&mut p_lut),
                ptr(&mut w_i),
                ptr(&mut h_i),
                ptr(&mut strength_f),
                ptr(&mut single_f),
            ],
        )?;
    }

    if color_active {
        // Chroma rewrites the finished luma and blur planes; moire and
        // sharpening rebuild those planes before reading them again.
        let mut p_cb = p_luma;
        let mut p_cr = p_blur;
        let strength = (nr.color / 100.0).clamp(0.0, 1.0);
        let mut radius = (1.0 + 2.0 * strength).round() as i32;
        let mut n_i = pixels as i32;
        launch1(
            gpu,
            "ycbcr_kernel",
            pixels,
            [ptr(&mut p_rgba), ptr(&mut p_cb), ptr(&mut p_cr), ptr(&mut n_i)],
        )?;
        launch2(gpu, "box_h_kernel", w, h, [
            ptr(&mut p_cb), ptr(&mut p_tmp), ptr(&mut w_i), ptr(&mut h_i), ptr(&mut radius),
        ])?;
        launch2(gpu, "box_v_kernel", w, h, [
            ptr(&mut p_tmp), ptr(&mut p_cb), ptr(&mut w_i), ptr(&mut h_i), ptr(&mut radius),
        ])?;
        launch2(gpu, "box_h_kernel", w, h, [
            ptr(&mut p_cr), ptr(&mut p_tmp), ptr(&mut w_i), ptr(&mut h_i), ptr(&mut radius),
        ])?;
        launch2(gpu, "box_v_kernel", w, h, [
            ptr(&mut p_tmp), ptr(&mut p_cr), ptr(&mut w_i), ptr(&mut h_i), ptr(&mut radius),
        ])?;
        launch1(
            gpu,
            "ycbcr_back_kernel",
            pixels,
            [ptr(&mut p_rgba), ptr(&mut p_cb), ptr(&mut p_cr), ptr(&mut n_i)],
        )?;
    }

    if moire_active {
        launch2(gpu, "luma_plane_kernel", w, h, [
            ptr(&mut p_rgba), ptr(&mut p_luma), ptr(&mut w_i), ptr(&mut h_i),
        ])?;
        let amount = (moire.amount / 100.0).clamp(0.0, 1.0);
        let pattern = (moire.pattern / 100.0).clamp(0.0, 1.0);
        let mut weight = amount * (0.25 + 0.75 * pattern);
        launch2(gpu, "moire_kernel", w, h, [
            ptr(&mut p_rgba),
            ptr(&mut p_luma),
            ptr(&mut w_i),
            ptr(&mut h_i),
            ptr(&mut weight),
        ])?;
    }

    if sharp_active {
        launch2(gpu, "luma_plane_kernel", w, h, [
            ptr(&mut p_rgba), ptr(&mut p_luma), ptr(&mut w_i), ptr(&mut h_i),
        ])?;
        let mut radius = ((sh.radius.max(0.1)) * scale).round().max(1.0) as i32;
        launch2(gpu, "box_h_kernel", w, h, [
            ptr(&mut p_luma), ptr(&mut p_blur), ptr(&mut w_i), ptr(&mut h_i), ptr(&mut radius),
        ])?;
        launch2(gpu, "box_v_kernel", w, h, [
            ptr(&mut p_blur), ptr(&mut p_tmp), ptr(&mut w_i), ptr(&mut h_i), ptr(&mut radius),
        ])?;
        for _ in 1..3 {
            launch2(gpu, "box_h_kernel", w, h, [
                ptr(&mut p_tmp), ptr(&mut p_blur), ptr(&mut w_i), ptr(&mut h_i), ptr(&mut radius),
            ])?;
            launch2(gpu, "box_v_kernel", w, h, [
                ptr(&mut p_blur), ptr(&mut p_tmp), ptr(&mut w_i), ptr(&mut h_i), ptr(&mut radius),
            ])?;
        }
        let mut gain = (sh.amount / 100.0) * 0.75;
        let mut threshold = sh.threshold.max(0.0);
        let mut halo = (sh.halo_suppression / 100.0).clamp(0.0, 1.0);
        launch2(gpu, "sharpen_kernel", w, h, [
            ptr(&mut p_rgba),
            ptr(&mut p_luma),
            ptr(&mut p_tmp),
            ptr(&mut w_i),
            ptr(&mut h_i),
            ptr(&mut gain),
            ptr(&mut threshold),
            ptr(&mut halo),
        ])?;
    }

    if lens_active {
        let mut n_i = pixels as i32;
        launch1(gpu, "copy_rgba_kernel", pixels, [ptr(&mut p_rgba), ptr(&mut p_tmp), ptr(&mut n_i)])?;
        let mut distortion = (lens.distortion / 100.0).clamp(-1.0, 1.0);
        let mut falloff = (lens.light_falloff / 100.0).clamp(0.0, 1.0);
        let mut purple = (lens.purple_fringing / 100.0).clamp(0.0, 1.0);
        let mut ca = i32::from(lens.chromatic_aberration);
        let mut hide = i32::from(lens.hide_distorted_areas);
        launch2(gpu, "lens_kernel", w, h, [
            ptr(&mut p_tmp),
            ptr(&mut p_rgba),
            ptr(&mut w_i),
            ptr(&mut h_i),
            ptr(&mut distortion),
            ptr(&mut falloff),
            ptr(&mut ca),
            ptr(&mut hide),
            ptr(&mut purple),
        ])?;
    }

    Ok(Some(base))
}

/// Classic two-copy entry point: run the chain, then download the frame.
pub(crate) fn post_process_pool(
    gpu: &mut Gpu,
    rgba: &mut [u8],
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    moire: &Moire,
    sh: &Sharpening,
    lens: &LensCorrection,
    scale: f32,
) -> Result<(), Failure> {
    if launch_detail_chain(gpu, rgba, w, h, nr, moire, sh, lens, scale)?.is_some() {
        gpu.download(Slot::Frame, rgba)?;
    }
    Ok(())
}

/// A detail frame left resident on the device for the grain stage.
///
/// Holding the stage mutex is the handoff pin: `release_retained` reclaims with
/// `try_lock`, so while this guard exists the reclaim hook cannot drop the
/// pool. The frame's bytes therefore stay on `detail`'s existing
/// [`gpu_runtime::VramLease`]: the handoff takes no second lease and releases
/// nothing early, so the ledger never double-bills or prematurely frees the
/// frame.
pub(crate) struct DetailResident {
    state: MutexGuard<'static, State>,
    handle: u64,
    pixels: usize,
}

impl DetailResident {
    /// Device handle of the post-detail RGBA8 frame (the `Slot::Frame` base the
    /// detail kernels wrote; grain reads and rewrites the same bytes).
    pub(crate) fn handle(&self) -> u64 {
        self.handle
    }

    /// Row-major RGBA8 bytes of the frame.
    pub(crate) fn bytes(&self) -> usize {
        self.pixels.saturating_mul(4)
    }

    /// Downloads the resident frame into `rgba`. The grain rollback path uses
    /// this so the host never keeps a partially applied stage.
    pub(crate) fn download(&mut self, rgba: &mut [u8]) -> Result<(), Failure> {
        match &mut *self.state {
            State::Ready(pool) => pool.gpu.download(Slot::Frame, rgba),
            State::NotTried | State::Unavailable => Err(Failure::Unavailable("detail_gpu")),
        }
    }

    /// Records one completed handoff: one frame-sized H2D (grain's upload) and
    /// one frame-sized D2H (detail's download) were made unnecessary.
    pub(crate) fn note_handoff_complete(&mut self) {
        let bytes = self.bytes();
        RESIDENT_SKIPPED_H2D.fetch_add(1, Ordering::Relaxed);
        RESIDENT_SKIPPED_H2D_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        RESIDENT_SKIPPED_D2H.fetch_add(1, Ordering::Relaxed);
        RESIDENT_SKIPPED_D2H_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
        if let State::Ready(pool) = &mut *self.state {
            pool.gpu.note_skipped_download();
        }
        log::debug!(
            "detail_gpu: resident handoff kept {bytes} B on the device (skipped one H2D and one D2H)"
        );
    }
}

/// Process-wide transfers the resident detail->grain handoff made unnecessary.
static RESIDENT_SKIPPED_H2D: AtomicU64 = AtomicU64::new(0);
static RESIDENT_SKIPPED_H2D_BYTES: AtomicU64 = AtomicU64::new(0);
static RESIDENT_SKIPPED_D2H: AtomicU64 = AtomicU64::new(0);
static RESIDENT_SKIPPED_D2H_BYTES: AtomicU64 = AtomicU64::new(0);

/// `(skipped H2D count, skipped H2D bytes, skipped D2H count, skipped D2H
/// bytes)`. Only fully completed handoffs advance these counters, so a non-zero
/// result is the acceptance proof that the reuse actually happened.
pub(crate) fn resident_savings() -> (u64, u64, u64, u64) {
    (
        RESIDENT_SKIPPED_H2D.load(Ordering::Relaxed),
        RESIDENT_SKIPPED_H2D_BYTES.load(Ordering::Relaxed),
        RESIDENT_SKIPPED_D2H.load(Ordering::Relaxed),
        RESIDENT_SKIPPED_D2H_BYTES.load(Ordering::Relaxed),
    )
}

/// Same reservation and state contract as [`post_process`], but on success the
/// post-detail frame stays in `Slot::Frame` and is returned instead of being
/// downloaded. `Ok(None)` means the chain had nothing active, so no upload
/// happened and `rgba` already is the post-detail frame.
pub(crate) fn post_process_resident(
    rgba: &mut [u8],
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    moire: &Moire,
    sh: &Sharpening,
    lens: &LensCorrection,
    scale: f32,
) -> Result<Option<DetailResident>, Failure> {
    let lum_active = nr.luminance > 0.0 || nr.single_pixel > 0.0;
    let color_active = nr.color > 0.0;
    let moire_active = moire.amount > 0.0;
    let sharp_active = sh.amount > 0.0;
    let lens_active = lens.distortion != 0.0
        || lens.light_falloff != 0.0
        || lens.chromatic_aberration
        || lens.purple_fringing > 0.0;
    if !lum_active && !color_active && !moire_active && !sharp_active && !lens_active {
        return Ok(None);
    }
    let planes = plane_count(lum_active, color_active, moire_active, sharp_active, lens_active);
    let Some(pixels) = w.checked_mul(h) else {
        return Err(Failure::Invalid("detail frame"));
    };
    let bytes = pixels.saturating_mul(4);
    if bytes == 0 || rgba.len() != bytes {
        return Err(Failure::Invalid("detail frame"));
    }
    let Some(need) = retained_bytes(pixels, planes) else {
        return Err(Failure::Invalid("detail frame"));
    };
    if need > retained_cap() || !gpu_runtime::vram_can_allocate("detail", need) {
        let failure = Failure::TooLarge { need, cap: retained_cap() };
        gpu_runtime::note_fallback("detail", &failure);
        return Err(failure);
    }
    register_reclaimer();
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    let launched = match &mut *state {
        State::Unavailable => Err(Failure::Unavailable("detail_gpu")),
        State::NotTried => match DetailPool::open(bytes, planes) {
            Ok(mut pool) => {
                let result = pool.ensure(bytes, planes).and_then(|()| {
                    launch_detail_chain(&mut pool.gpu, rgba, w, h, nr, moire, sh, lens, scale)
                });
                if result.is_ok() {
                    *state = State::Ready(pool);
                }
                result
            }
            Err(failure) => {
                *state = State::Unavailable;
                Err(failure)
            }
        },
        State::Ready(pool) => pool.ensure(bytes, planes).and_then(|()| {
            launch_detail_chain(&mut pool.gpu, rgba, w, h, nr, moire, sh, lens, scale)
        }),
    };
    match launched {
        Ok(handle) => Ok(handle.map(|handle| DetailResident { state, handle, pixels })),
        Err(failure) => {
            log::warn!("detail_gpu: {failure:?}; staying on CPU");
            gpu_runtime::note_fallback("detail", &failure);
            Err(failure)
        }
    }
}

/// How [`super::detail::post_process_impl`] continues after the resident
/// detail->grain attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ResidentGrain {
    /// Detail and grain both ran on the device and the final frame was
    /// downloaded; the caller is done.
    Applied,
    /// Detail is applied on the host and grain still has to run: either the
    /// chain was inactive (nothing to upload) or grain refused/failed and the
    /// post-detail frame was restored. The caller skips a second detail attempt
    /// and lets the normal grain tail run.
    DetailRestored,
    /// The frame never reached the host (detail failed, or the device was lost
    /// before the download). `rgba` holds the pre-detail bytes, so the caller
    /// reruns the whole detail path on the CPU: exactly what today's code does
    /// when the detail attempt or its download fails.
    CpuEverything,
}

/// Detail plus grain in one handoff. Only called for a plan whose CPU-split
/// flags are all clear (see `DetailStagePlan::resident_to_grain`), so no host
/// rewrite can sit between the detail device frame and grain.
pub(crate) fn post_process_resident_grain(
    rgba: &mut [u8],
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    moire: &Moire,
    sh: &Sharpening,
    lens: &LensCorrection,
    scale: f32,
    grain: &FilmGrain,
) -> ResidentGrain {
    let mut resident = match post_process_resident(rgba, w, h, nr, moire, sh, lens, scale) {
        Ok(Some(resident)) => resident,
        // No detail operator touched the frame: `rgba` already is the
        // post-detail frame, so the tail can run grain on it.
        Ok(None) => return ResidentGrain::DetailRestored,
        // Same failure contract as `post_process`: the host frame is untouched,
        // so rerun everything on the CPU.
        Err(_) => return ResidentGrain::CpuEverything,
    };
    match super::pixel_gpu::grain_stage_resident(&mut resident, rgba, w, h, grain, scale) {
        super::pixel_gpu::GrainResident::Applied => ResidentGrain::Applied,
        super::pixel_gpu::GrainResident::Restored => ResidentGrain::DetailRestored,
        super::pixel_gpu::GrainResident::RestoreFailed => ResidentGrain::CpuEverything,
    }
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

fn launch1<const N: usize>(
    gpu: &mut Gpu,
    kernel: &'static str,
    n: usize,
    mut args: [*mut c_void; N],
) -> Result<(), Failure> {
    gpu.launch(kernel, gpu_runtime::grid1(n, 256), (256, 1), &mut args)
}

#[cfg(test)]
pub(crate) fn open_for_bench() -> Option<Gpu> {
    Gpu::open("detail", PTX).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nr(luminance: f32, color: f32) -> NoiseReduction {
        NoiseReduction {
            luminance,
            details: 0.0,
            color,
            single_pixel: 0.0,
        }
    }

    fn sharp(amount: f32) -> Sharpening {
        Sharpening { amount, ..Default::default() }
    }

    #[test]
    fn active_plane_count_covers_every_stage_without_overallocation() {
        let off_nr = nr(0.0, 0.0);
        let off_moire = Moire::default();
        let off_sharp = sharp(0.0);
        let off_lens = LensCorrection::default();

        assert_eq!(plane_count(false, false, false, false, false), 0);
        assert_eq!(plane_count(true, false, false, false, false), 2);
        assert_eq!(plane_count(false, false, true, false, false), 2);
        assert_eq!(plane_count(false, true, false, false, false), 4);
        assert_eq!(plane_count(false, false, false, true, false), 4);
        assert_eq!(plane_count(false, false, false, false, true), 4);
        assert_eq!(plane_count(true, true, true, true, true), 4);

        assert_eq!(active_planes(&off_nr, &off_moire, &off_sharp, &off_lens), 0);
        assert_eq!(active_planes(&nr(50.0, 0.0), &off_moire, &off_sharp, &off_lens), 2);
        let single = NoiseReduction { single_pixel: 50.0, ..nr(0.0, 0.0) };
        assert_eq!(active_planes(&single, &off_moire, &off_sharp, &off_lens), 2);
        assert_eq!(
            active_planes(&off_nr, &Moire { amount: 50.0, ..Default::default() }, &off_sharp, &off_lens),
            2
        );
        assert_eq!(active_planes(&nr(0.0, 50.0), &off_moire, &off_sharp, &off_lens), 4);
        assert_eq!(active_planes(&off_nr, &off_moire, &sharp(50.0), &off_lens), 4);
        assert_eq!(
            active_planes(
                &off_nr,
                &off_moire,
                &off_sharp,
                &LensCorrection { distortion: 50.0, ..Default::default() },
            ),
            4,
            "lens reads the scratch plane, so it needs four planes"
        );
    }

    #[test]
    fn plane_sizing_preserves_the_32_mp_retention_boundary() {
        assert_eq!(MAX_RETAINED_FRAME_BYTES, 128 * 1024 * 1024);
        assert_eq!(MAX_RETAINED_BYTES, 512 * 1024 * 1024);
        assert_eq!(MAX_RETAINED_FRAME_BYTES.saturating_mul(MAX_PLANES), MAX_RETAINED_BYTES);

        // The boundary is the frame, not bytes*planes: 24 MP and exactly
        // 32 MP stay resident; one pixel over 32 MP takes the CPU fallback.
        assert!(retained_frame(24_000_000 * 4));
        assert!(retained_frame(MAX_RETAINED_FRAME_BYTES));
        assert!(!retained_frame(MAX_RETAINED_FRAME_BYTES + 4));

        // A 45 MP frame is not retained even when only two planes run; sizing
        // by `bytes * planes` alone would have pinned 363 MB of it.
        let frame_45mp = 8256usize * 5504 * 4;
        assert!(!retained_frame(frame_45mp));
        assert!(frame_45mp.saturating_mul(2) < MAX_RETAINED_BYTES);
    }

    #[test]
    fn retained_bytes_tracks_the_planes_actually_active() {
        let pixels = 1400 * 933;
        // Four planes (colour NR active) and two planes (luma only) differ.
        let four = retained_bytes(pixels, 4).expect("no overflow");
        let two = retained_bytes(pixels, 2).expect("no overflow");
        assert_eq!(four, pixels * 4 * 4 + 1025 * 4);
        assert_eq!(two, pixels * 4 * 2 + 1025 * 4);
        assert_eq!(retained_bytes(pixels, 0), Some(1025 * 4));

        // The old six-plane formula was bytes * 6; a 32 MP four-plane frame
        // must not reserve more than the old shape did.
        let pixels_32mp = MAX_RETAINED_FRAME_BYTES / 4;
        assert!(retained_bytes(pixels_32mp, 4).unwrap() < pixels_32mp * 4 * 6 + 1025 * 4);

        // Checked arithmetic: a huge frame returns None instead of wrapping.
        assert!(retained_bytes(usize::MAX / 8, 4).is_none());
    }
}
