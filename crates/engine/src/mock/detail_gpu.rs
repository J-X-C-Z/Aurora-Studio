//! Optional CUDA path for the detail stages.
//!
//! It mirrors `stage_luminance_nr`, `stage_color_nr`, `stage_moire` and
//! `stage_sharpening` byte-for-byte; the CPU path stays the default and is used
//! whenever the driver, PTX or `AURORA_DETAIL_GPU` is absent.

use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};
use aurora_portrait::cuda::{CudaError, DeviceBuffer, Nvidia};

use crate::{LensCorrection, Moire, NoiseReduction, Sharpening};

const PTX: &str = include_str!("../../native/cuda/detail_gpu.ptx");
const BLOCK: (u32, u32) = (16, 16);
const PREWARM_BYTES: usize = 1400 * 933 * 4;
// Keep one 24-32 MP render resident so slider refinements do not reallocate the
// six-plane pool. Larger frames release after each call instead of pinning
// unbounded GPU memory.
const MAX_RETAINED_BYTES: usize = 768 * 1024 * 1024;

struct DetailPool {
    gpu: Box<Nvidia>,
    pool: SendBuffer,
    lut: SendBuffer,
    bytes: usize,
}

impl DetailPool {
    fn open(bytes: usize) -> Result<Self, CudaError> {
        let gpu = Box::new(Nvidia::open_ptx(PTX)?);
        Ok(Self {
            pool: SendBuffer(alloc_static(&gpu, bytes * 6)?),
            lut: SendBuffer(alloc_static(&gpu, 1025 * 4)?),
            gpu,
            bytes,
        })
    }

    fn ensure(&mut self, bytes: usize) -> Result<(), CudaError> {
        if bytes > self.bytes {
            self.pool = SendBuffer(alloc_static(&self.gpu, bytes * 6)?);
            self.bytes = bytes;
        }
        Ok(())
    }
}

struct SendBuffer(DeviceBuffer<'static>);

// `DeviceBuffer` only needs the driver table to move; the owning `Nvidia`
// object stays boxed and `State` serializes every use.
unsafe impl Send for SendBuffer {}

/// The boxed driver table never moves, and `DetailPool` owns it for its whole
/// allocation lifetime, so the buffers can safely outlive the short borrow.
fn alloc_static(gpu: &Nvidia, bytes: usize) -> Result<DeviceBuffer<'static>, CudaError> {
    let buffer = gpu.alloc_buffer(bytes)?;
    Ok(unsafe { std::mem::transmute::<DeviceBuffer<'_>, DeviceBuffer<'static>>(buffer) })
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

pub(crate) fn prewarm() -> bool {
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
            Err(err) => {
                log::warn!("detail_gpu: {err:?}; staying on CPU");
                *state = State::Unavailable;
                false
            }
        },
    }
}

/// Runs the NR + colour NR + moire + sharpening stages on the GPU. `false` means the
/// caller must use the CPU stages (no driver, no context, or an invalid frame).
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
    if bytes == 0 || rgba.len() != bytes || bytes.saturating_mul(6) > MAX_RETAINED_BYTES {
        return post_process_alloc(
            rgba, w, h, nr, moire, sh, lens, scale,
        );
    }
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    match &mut *state {
        State::Unavailable => false,
        State::NotTried => {
            let mut pool = match DetailPool::open(bytes) {
                Ok(pool) => pool,
                Err(err) => {
                    log::warn!("detail_gpu: {err:?}; staying on CPU");
                    *state = State::Unavailable;
                    return false;
                }
            };
            let result = pool
                .ensure(bytes)
                .and_then(|()| post_process_pool(
                    pool.gpu.as_ref(),
                    rgba,
                    w,
                    h,
                    nr,
                    moire,
                    sh,
                    lens,
                    scale,
                    &pool.pool.0,
                    Some(&pool.lut.0),
                ));
            match result {
                Ok(()) => {
                    *state = State::Ready(pool);
                    true
                }
                Err(err) => {
                    log::warn!("detail_gpu: {err:?}; staying on CPU");
                    false
                }
            }
        }
        State::Ready(pool) => {
            let result = pool
                .ensure(bytes)
                .and_then(|()| post_process_pool(
                    pool.gpu.as_ref(),
                    rgba,
                    w,
                    h,
                    nr,
                    moire,
                    sh,
                    lens,
                    scale,
                    &pool.pool.0,
                    Some(&pool.lut.0),
                ));
            match result {
                Ok(()) => true,
                Err(err) => {
                    log::warn!("detail_gpu: {err:?}; staying on CPU");
                    false
                }
            }
        }
    }
}

fn post_process_alloc(
    rgba: &mut [u8],
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    moire: &Moire,
    sh: &Sharpening,
    lens: &LensCorrection,
    scale: f32,
) -> bool {
    match Nvidia::open_ptx(PTX) {
        Ok(gpu) => match post_process_with(&gpu, rgba, w, h, nr, moire, sh, lens, scale) {
            Ok(()) => true,
            Err(err) => {
                log::warn!("detail_gpu: {err:?}; staying on CPU");
                false
            }
        },
        Err(err) => {
            log::warn!("detail_gpu: {err:?}; staying on CPU");
            false
        }
    }
}

pub(crate) fn post_process_with(
    gpu: &Nvidia,
    rgba: &mut [u8],
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    moire: &Moire,
    sh: &Sharpening,
    lens: &LensCorrection,
    scale: f32,
) -> Result<(), CudaError> {
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
    let pool = gpu.alloc_buffer(bytes * 6)?;
    let lut = lum_active.then(|| gpu.alloc_buffer(1025 * 4)).transpose()?;
    post_process_pool(gpu, rgba, w, h, nr, moire, sh, lens, scale, &pool, lut.as_ref())
}

pub(crate) fn post_process_pool(
    gpu: &Nvidia,
    rgba: &mut [u8],
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    moire: &Moire,
    sh: &Sharpening,
    lens: &LensCorrection,
    scale: f32,
    pool: &DeviceBuffer<'_>,
    lut_buf: Option<&DeviceBuffer<'_>>,
) -> Result<(), CudaError> {
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

    let base = pool.handle();
    let mut p_rgba = base;
    let mut p_luma = base + bytes as u64;
    let mut p_blur = base + (bytes * 2) as u64;
    let mut p_tmp = base + (bytes * 3) as u64;
    let mut p_cb = base + (bytes * 4) as u64;
    let mut p_cr = base + (bytes * 5) as u64;
    gpu.upload_buffer(pool, rgba)?;

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
        let Some(lut_buf) = lut_buf else {
            return Err(CudaError::Unavailable("detail lut buffer"));
        };
        let lut_bytes = unsafe {
            std::slice::from_raw_parts(lut.as_ptr() as *const u8, lut.len() * 4)
        };
        gpu.upload_buffer(lut_buf, lut_bytes)?;

        let mut p_lut = lut_buf.handle();
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

    gpu.download_buffer(pool, rgba)?;
    Ok(())
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

fn launch1<const N: usize>(
    gpu: &Nvidia,
    kernel: &'static str,
    n: usize,
    mut args: [*mut c_void; N],
) -> Result<(), CudaError> {
    let grid = (((n as u32) + 255) / 256, 1);
    gpu.launch_raw(kernel, grid, (256, 1), &mut args)?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn open_for_bench() -> Option<Nvidia> {
    Nvidia::open_ptx(PTX).ok()
}
