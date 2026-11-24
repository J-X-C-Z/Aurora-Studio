//! Optional CUDA path for the detail stages.
//!
//! It mirrors `stage_luminance_nr`, `stage_color_nr`, `stage_moire` and
//! `stage_sharpening` byte-for-byte; the CPU path stays the default and is used
//! whenever the driver, PTX or `AURORA_DETAIL_GPU` is absent.
//!
//! Device memory comes from the shared [`super::gpu_runtime::Gpu`]: one
//! six-plane `Frame` slot, one LUT slot, grow-only and accounted.

use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use super::gpu_runtime::{self, Failure, Gpu, Slot};

use crate::{LensCorrection, Moire, NoiseReduction, Sharpening};

const PTX: &str = include_str!("../../native/cuda/detail_gpu.ptx");
const BLOCK: (u32, u32) = (16, 16);
const PREWARM_BYTES: usize = 1400 * 933 * 4;
/// Retained cap is the shared ledger budget, not a private 768 MiB constant.
fn retained_cap() -> usize {
    gpu_runtime::vram_budget()
}
/// Reclaimed before clarity but after develop.
const RECLAIM_PRIORITY: u8 = 10;
/// Six RGBA planes plus the 1025-entry luma LUT.
fn retained_bytes(pixels: usize) -> Option<usize> {
    pixels.checked_mul(24)?.checked_add(1025 * 4)
}

struct DetailPool {
    gpu: Gpu,
}

impl DetailPool {
    fn open(bytes: usize) -> Result<Self, Failure> {
        let mut gpu = Gpu::open("detail", PTX)?;
        reserve(&mut gpu, bytes, retained_cap())?;
        Ok(Self { gpu })
    }

    fn ensure(&mut self, bytes: usize) -> Result<(), Failure> {
        reserve(&mut self.gpu, bytes, retained_cap())
    }
}

fn reserve(gpu: &mut Gpu, bytes: usize, cap: usize) -> Result<(), Failure> {
    gpu.reserve(Slot::Frame, bytes.saturating_mul(6), cap)?;
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
        State::NotTried => match DetailPool::open(PREWARM_BYTES) {
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
    // Non-allocating admission against the shared ledger before any device
    // memory exists; a refusal is a CPU fallback, not a broken stage.
    let Some(need) = retained_bytes(pixels) else { return false };
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
        State::NotTried => match DetailPool::open(bytes) {
            Ok(mut pool) => {
                let result = pool
                    .ensure(bytes)
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
            .ensure(bytes)
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
    reserve(gpu, pixels * 4, usize::MAX)?;
    post_process_pool(gpu, rgba, w, h, nr, moire, sh, lens, scale)
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
    let lum_active = nr.luminance > 0.0 || nr.single_pixel > 0.0;
    let color_active = nr.color > 0.0;
    let moire_active = moire.amount > 0.0;
    let sharp_active = sh.amount > 0.0;
    let lens_active = lens.distortion != 0.0
        || lens.light_falloff != 0.0
        || lens.chromatic_aberration
        || lens.purple_fringing > 0.0;
    if !lum_active && !color_active && !moire_active && !sharp_active && !lens_active {
        return Ok(());
    }
    let Some(pixels) = w.checked_mul(h) else { return Ok(()) };
    if pixels == 0 || rgba.len() != pixels.saturating_mul(4) {
        return Ok(());
    }
    let bytes = pixels * 4;

    let base = gpu.handle(Slot::Frame).ok_or(Failure::Invalid("detail frame slot"))?;
    let mut p_rgba = base;
    let mut p_luma = base + bytes as u64;
    let mut p_blur = base + (bytes * 2) as u64;
    let mut p_tmp = base + (bytes * 3) as u64;
    let mut p_cb = base + (bytes * 4) as u64;
    let mut p_cr = base + (bytes * 5) as u64;
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

    gpu.download(Slot::Frame, rgba)?;
    Ok(())
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
