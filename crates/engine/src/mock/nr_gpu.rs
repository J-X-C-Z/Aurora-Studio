//! Optional CUDA path for the two production noise stages.
//!
//! `stage_luminance_nr_f32` (C1 `LumaNoise2018`) and `stage_color_nr_f32`
//! (C1 `ColorNoise2012` / `LABRangeFilter2012`) are the heaviest full-frame
//! CPU stages of the display-sRGB float core. The kernels here mirror their
//! arithmetic; `AURORA_NR_GPU=0` keeps the CPU stages and leaves every byte
//! untouched. Any driver, module or launch failure also falls back to CPU.

use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use aurora_portrait::cuda::{CudaError, DeviceBuffer, Nvidia, Stream};

use crate::NoiseReduction;

const PTX: &str = include_str!("../../native/cuda/nr_gpu.ptx");
const BLOCK: (u32, u32) = (32, 8);
/// One 24 MP frame needs ~32 bytes per pixel; 36 MP stays inside this cap.
const MAX_POOL_BYTES: usize = 1536 * 1024 * 1024;
const PREWARM: (usize, usize) = (1400, 933);

struct Layout {
    n: usize,
    rgb: usize,
    pad: usize,
    plane: usize,
    total: usize,
}

fn layout(w: usize, h: usize) -> Option<Layout> {
    let n = w.checked_mul(h)?;
    let rgb = n.checked_mul(12)?;
    let pad = w
        .checked_add(2)?
        .checked_mul(h.checked_add(2)?)?
        .checked_mul(4)?;
    let plane = n.checked_mul(4)?;
    // The colour pass reuses the luma pad as its L plane, so only four extra
    // planes (U, V, horizontal U, horizontal V) follow it.
    let total = rgb.checked_add(pad)?.checked_add(plane.checked_mul(4)?)?;
    Some(Layout {
        n,
        rgb,
        pad,
        plane,
        total,
    })
}

struct SendBuffer(DeviceBuffer<'static>);

// `DeviceBuffer` only needs the driver table to move; the owning `Nvidia`
// object stays boxed and `State` serializes every use.
unsafe impl Send for SendBuffer {}

struct SendStream(Stream<'static>);

unsafe impl Send for SendStream {}

/// The boxed driver table never moves, and `NrPool` owns it for its whole
/// allocation lifetime, so the buffers can safely outlive the short borrow.
fn alloc_static(gpu: &Nvidia, bytes: usize) -> Result<DeviceBuffer<'static>, CudaError> {
    let buffer = gpu.alloc_buffer(bytes)?;
    Ok(unsafe { std::mem::transmute::<DeviceBuffer<'_>, DeviceBuffer<'static>>(buffer) })
}

struct NrPool {
    gpu: Box<Nvidia>,
    buffer: SendBuffer,
    lut: SendBuffer,
    stream: Option<SendStream>,
    lut_sigma: Option<f32>,
    bytes: usize,
}

impl NrPool {
    fn open(w: usize, h: usize) -> Result<Self, CudaError> {
        let l = layout(w, h).ok_or(CudaError::Unavailable("nr frame size"))?;
        let gpu = Box::new(Nvidia::open_ptx(PTX)?);
        let stream = gpu
            .create_stream()
            .ok()
            .map(|stream| unsafe {
                SendStream(std::mem::transmute::<Stream<'_>, Stream<'static>>(stream))
            });
        Ok(Self {
            buffer: SendBuffer(alloc_static(&gpu, l.total)?),
            lut: SendBuffer(alloc_static(&gpu, 1025 * 4)?),
            stream,
            lut_sigma: None,
            gpu,
            bytes: l.total,
        })
    }

    fn ensure(&mut self, total: usize) -> Result<(), CudaError> {
        if total > self.bytes {
            self.buffer = SendBuffer(alloc_static(&self.gpu, total)?);
            self.bytes = total;
        }
        Ok(())
    }
}

enum State {
    NotTried,
    Ready(NrPool),
    Unavailable,
}

fn state() -> &'static Mutex<State> {
    static STATE: OnceLock<Mutex<State>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(State::NotTried))
}

/// GPU noise reduction is the shipped default; `AURORA_NR_GPU=0` restores the
/// exact CPU path. Unit tests default to CPU so their byte assertions stay
/// deterministic unless they opt in explicitly.
pub(crate) fn enabled() -> bool {
    match std::env::var_os("AURORA_NR_GPU") {
        Some(value) => value != "0",
        None => !cfg!(test),
    }
}

pub(crate) fn prewarm() -> bool {
    if !enabled() {
        return false;
    }
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    match &mut *state {
        State::Ready(_) => true,
        State::Unavailable => false,
        State::NotTried => match NrPool::open(PREWARM.0, PREWARM.1) {
            Ok(pool) => {
                *state = State::Ready(pool);
                log::info!("CUDA noise-reduction path ready");
                true
            }
            Err(err) => {
                log::warn!("nr_gpu: {err:?}; staying on CPU");
                *state = State::Unavailable;
                false
            }
        },
    }
}

/// Runs the luma and/or colour stage on the GPU. `false` means the caller must
/// run the CPU stages instead.
pub(crate) fn run_f32(
    rgb: &mut [[f32; 3]],
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    scale: f32,
) -> bool {
    let luma = nr.luminance > 0.0;
    let color = nr.color > 0.0;
    if (!luma && !color) || !enabled() {
        return false;
    }
    let Some(l) = layout(w, h) else { return false };
    if rgb.len() != l.n || l.total > MAX_POOL_BYTES {
        return false;
    }
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    match &mut *state {
        State::Unavailable => false,
        State::NotTried => {
            let mut pool = match NrPool::open(w, h) {
                Ok(pool) => pool,
                Err(err) => {
                    log::warn!("nr_gpu: {err:?}; staying on CPU");
                    *state = State::Unavailable;
                    return false;
                }
            };
            let result = pool
                .ensure(l.total)
                .and_then(|()| run_pool(&mut pool, rgb, w, h, nr, scale, luma, color));
            match result {
                Ok(()) => {
                    *state = State::Ready(pool);
                    true
                }
                Err(err) => {
                    log::warn!("nr_gpu: {err:?}; staying on CPU");
                    false
                }
            }
        }
        State::Ready(pool) => {
            let result = pool
                .ensure(l.total)
                .and_then(|()| run_pool(pool, rgb, w, h, nr, scale, luma, color));
            match result {
                Ok(()) => true,
                Err(err) => {
                    log::warn!("nr_gpu: {err:?}; staying on CPU");
                    false
                }
            }
        }
    }
}

fn run_pool(
    pool: &mut NrPool,
    rgb: &mut [[f32; 3]],
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    scale: f32,
    luma: bool,
    color: bool,
) -> Result<(), CudaError> {
    ensure_lut(pool, nr, luma)?;
    let gpu = pool.gpu.as_ref();
    let l = layout(w, h).ok_or(CudaError::Unavailable("nr frame size"))?;
    let host = unsafe { std::slice::from_raw_parts_mut(rgb.as_mut_ptr() as *mut u8, l.n * 12) };
    if let Some(stream) = pool.stream.as_ref().map(|stream| &stream.0)
        && let Ok(mut registration) = gpu.register_host(host)
    {
        // Pinned in place: the DMA engine reads and writes the caller's frame
        // directly, with no 290 MB host-to-host staging copy.
        gpu.upload_async(&pool.buffer.0, registration.as_slice(), stream)?;
        launch_stages(pool, w, h, nr, scale, luma, color, Some(stream))?;
        gpu.download_async(&pool.buffer.0, registration.as_mut_slice(), stream)?;
        gpu.stream_sync(stream)?;
        return Ok(());
    }
    gpu.upload_buffer(&pool.buffer.0, host)?;
    launch_stages(pool, w, h, nr, scale, luma, color, None)?;
    gpu.download_buffer(&pool.buffer.0, host)?;
    Ok(())
}

/// Uploads the luma range LUT only when the slider-derived sigma changed; a
/// synchronous 4 KB copy mid-frame would otherwise drain the async pipeline.
fn ensure_lut(
    pool: &mut NrPool,
    nr: &NoiseReduction,
    luma: bool,
) -> Result<(), CudaError> {
    if !luma {
        return Ok(());
    }
    let strength = (nr.luminance / 100.0).clamp(0.0, 1.0);
    let details = (nr.details / 100.0).clamp(0.0, 1.0);
    let sigma = ((2.5 + 26.0 * strength) / (1.0 + 3.5 * details)).max(0.5);
    if pool.lut_sigma == Some(sigma) {
        return Ok(());
    }
    let two_sigma2 = 2.0 * sigma * sigma;
    let table: Vec<f32> = (0..1025)
        .map(|i| {
            let d = i as f32 * 0.25;
            (-(d * d) / two_sigma2).exp()
        })
        .collect();
    let table_bytes =
        unsafe { std::slice::from_raw_parts(table.as_ptr() as *const u8, table.len() * 4) };
    pool.gpu.upload_buffer(&pool.lut.0, table_bytes)?;
    pool.lut_sigma = Some(sigma);
    Ok(())
}

fn launch_stages(
    pool: &NrPool,
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    scale: f32,
    luma: bool,
    color: bool,
    stream: Option<&Stream<'_>>,
) -> Result<(), CudaError> {
    let gpu = pool.gpu.as_ref();
    let l = layout(w, h).ok_or(CudaError::Unavailable("nr frame size"))?;
    let base = pool.buffer.0.handle();
    let mut p_rgb = base;
    let mut p_pad = base + l.rgb as u64;
    // `p_l` aliases the luma pad: the luma pass has already finished.
    let mut p_l = p_pad;
    let mut p_u = p_pad + l.pad as u64;
    let mut p_v = p_u + l.plane as u64;
    let mut p_hu = p_v + l.plane as u64;
    let mut p_hv = p_hu + l.plane as u64;
    let (mut w_i, mut h_i) = (w as i32, h as i32);
    let mut n_i = l.n as i32;

    if luma {
        let strength = (nr.luminance / 100.0).clamp(0.0, 1.0);
        let mut p_lut = pool.lut.0.handle();
        let mut strength_f = strength;

        launch2(
            gpu,
            "nr_luma_pad_kernel",
            w + 2,
            h + 2,
            [
                ptr(&mut p_rgb),
                ptr(&mut p_pad),
                ptr(&mut w_i),
                ptr(&mut h_i),
            ],
            stream,
        )?;
        launch2(
            gpu,
            "nr_luma_apply_kernel",
            w,
            h,
            [
                ptr(&mut p_rgb),
                ptr(&mut p_pad),
                ptr(&mut p_lut),
                ptr(&mut w_i),
                ptr(&mut h_i),
                ptr(&mut strength_f),
            ],
            stream,
        )?;
    }

    if color {
        let strength = (nr.color / 100.0).clamp(0.0, 1.0);
        let scale = if scale.is_finite() && scale > 0.0 {
            scale
        } else {
            1.0
        };
        let scale_p = 0.35 + 1.3 * strength;
        let mut pz = 0.401_832_58 * scale_p;
        let mut pw = 2.968_700_4 * scale_p;
        let mut pz_chroma = 2_400.0 + 18_000.0 * strength * strength;
        let mut pw_chroma = 12.0 + 36.0 * strength;
        let mut step = (4.0 * scale).round().clamp(1.0, 8.0) as i32;
        let half_taps = (7.0 * (0.4 + 1.2 * strength)).round().clamp(2.0, 11.0) as i32;
        let mut radius = half_taps * step;

        launch1(
            gpu,
            "nr_color_encode_kernel",
            l.n,
            [
                ptr(&mut p_rgb),
                ptr(&mut p_l),
                ptr(&mut p_u),
                ptr(&mut p_v),
                ptr(&mut n_i),
            ],
            stream,
        )?;
        launch2(
            gpu,
            "nr_color_h_kernel",
            w,
            h,
            [
                ptr(&mut p_l),
                ptr(&mut p_u),
                ptr(&mut p_v),
                ptr(&mut p_hu),
                ptr(&mut p_hv),
                ptr(&mut w_i),
                ptr(&mut h_i),
                ptr(&mut radius),
                ptr(&mut step),
                ptr(&mut pz),
                ptr(&mut pw),
                ptr(&mut pz_chroma),
                ptr(&mut pw_chroma),
            ],
            stream,
        )?;
        launch2(
            gpu,
            "nr_color_v_kernel",
            w,
            h,
            [
                ptr(&mut p_l),
                ptr(&mut p_hu),
                ptr(&mut p_hv),
                ptr(&mut p_rgb),
                ptr(&mut w_i),
                ptr(&mut h_i),
                ptr(&mut radius),
                ptr(&mut step),
                ptr(&mut pz),
                ptr(&mut pw),
                ptr(&mut pz_chroma),
                ptr(&mut pw_chroma),
            ],
            stream,
        )?;
    }

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
    stream: Option<&Stream<'_>>,
) -> Result<(), CudaError> {
    let grid = (
        ((w as u32) + BLOCK.0 - 1) / BLOCK.0,
        ((h as u32) + BLOCK.1 - 1) / BLOCK.1,
    );
    match stream {
        Some(stream) => gpu.launch_raw_stream(kernel, grid, BLOCK, &mut args, stream)?,
        None => gpu.launch_raw(kernel, grid, BLOCK, &mut args)?,
    };
    Ok(())
}

fn launch1<const N: usize>(
    gpu: &Nvidia,
    kernel: &'static str,
    n: usize,
    mut args: [*mut c_void; N],
    stream: Option<&Stream<'_>>,
) -> Result<(), CudaError> {
    let grid = (((n as u32) + 255) / 256, 1);
    match stream {
        Some(stream) => gpu.launch_raw_stream(kernel, grid, (256, 1), &mut args, stream)?,
        None => gpu.launch_raw(kernel, grid, (256, 1), &mut args)?,
    };
    Ok(())
}

/// Device-only time: `iters` passes are queued, then one readback (the only
/// synchronisation this driver wrapper exposes) drains them; the calibrated
/// idle readback is subtracted, so the result is kernel execution time.
#[cfg(test)]
fn device_time(
    rgb: &mut [[f32; 3]],
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    scale: f32,
    luma: bool,
    color: bool,
    iters: usize,
) -> Option<f64> {
    use std::time::Instant;
    let l = layout(w, h)?;
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    let State::Ready(pool) = &mut *state else {
        return None;
    };
    pool.ensure(l.total).ok()?;
    ensure_lut(pool, nr, luma).ok()?;
    let bytes = unsafe { std::slice::from_raw_parts(rgb.as_ptr() as *const u8, l.n * 12) };
    pool.gpu.upload_buffer(&pool.buffer.0, bytes).ok()?;
    launch_stages(pool, w, h, nr, scale, luma, color, None).ok()?;
    let out = unsafe { std::slice::from_raw_parts_mut(rgb.as_mut_ptr() as *mut u8, l.n * 12) };
    pool.gpu.download_buffer(&pool.buffer.0, out).ok()?;
    // Calibrate an idle readback of the same size.
    let start = Instant::now();
    pool.gpu.download_buffer(&pool.buffer.0, out).ok()?;
    let idle = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    for _ in 0..iters.max(1) {
        launch_stages(pool, w, h, nr, scale, luma, color, None).ok()?;
    }
    pool.gpu.download_buffer(&pool.buffer.0, out).ok()?;
    let per_pass = start.elapsed().as_secs_f64() * 1000.0 / iters.max(1) as f64;
    Some((per_pass - idle / iters.max(1) as f64).max(0.0))
}

/// Upload and download cost for one 24 MP frame, measured separately.
#[cfg(test)]
fn io_time(rgb: &mut [[f32; 3]], w: usize, h: usize) -> Option<(f64, f64)> {
    use std::time::Instant;
    let l = layout(w, h)?;
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    let State::Ready(pool) = &mut *state else {
        return None;
    };
    pool.ensure(l.total).ok()?;
    let bytes = unsafe { std::slice::from_raw_parts(rgb.as_ptr() as *const u8, l.n * 12) };
    let start = Instant::now();
    pool.gpu.upload_buffer(&pool.buffer.0, bytes).ok()?;
    let up = start.elapsed().as_secs_f64() * 1000.0;
    let out = unsafe { std::slice::from_raw_parts_mut(rgb.as_mut_ptr() as *mut u8, l.n * 12) };
    let start = Instant::now();
    pool.gpu.download_buffer(&pool.buffer.0, out).ok()?;
    let down = start.elapsed().as_secs_f64() * 1000.0;
    Some((up, down))
}

/// Pinned/async staging cost split for one frame.
#[cfg(test)]
fn staging_time(rgb: &mut [[f32; 3]], w: usize, h: usize) -> Option<(f64, f64, f64, f64)> {
    use std::time::Instant;
    let l = layout(w, h)?;
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    let State::Ready(pool) = &mut *state else {
        return None;
    };
    pool.ensure(l.total).ok()?;
    let gpu = pool.gpu.as_ref();
    let stream = pool.stream.as_ref().map(|stream| &stream.0)?;
    let host = unsafe { std::slice::from_raw_parts_mut(rgb.as_mut_ptr() as *mut u8, l.n * 12) };
    let start = Instant::now();
    let mut registration = gpu.register_host(host).ok()?;
    let register = start.elapsed().as_secs_f64() * 1000.0;
    // First touch faults the newly pinned pages; measure the steady state.
    gpu.upload_async(&pool.buffer.0, registration.as_slice(), stream).ok()?;
    gpu.stream_sync(stream).ok()?;
    let start = Instant::now();
    gpu.upload_async(&pool.buffer.0, registration.as_slice(), stream).ok()?;
    gpu.stream_sync(stream).ok()?;
    let up = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    gpu.download_async(&pool.buffer.0, registration.as_mut_slice(), stream).ok()?;
    gpu.stream_sync(stream).ok()?;
    let down = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    drop(registration);
    let unregister = start.elapsed().as_secs_f64() * 1000.0;
    Some((register, up, down, unregister))
}

/// `cuMemAllocHost` staging: allocation, host copy, async up/down.
#[cfg(test)]
fn pinned_time(rgb: &mut [[f32; 3]], w: usize, h: usize) -> Option<(f64, f64, f64, f64)> {
    use std::time::Instant;
    let l = layout(w, h)?;
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    let State::Ready(pool) = &mut *state else {
        return None;
    };
    pool.ensure(l.total).ok()?;
    let gpu = pool.gpu.as_ref();
    let stream = pool.stream.as_ref().map(|stream| &stream.0)?;
    let start = Instant::now();
    let mut pinned = gpu.alloc_pinned(l.n * 12).ok()?;
    let alloc = start.elapsed().as_secs_f64() * 1000.0;
    let host = unsafe { std::slice::from_raw_parts(rgb.as_ptr() as *const u8, l.n * 12) };
    let start = Instant::now();
    pinned.as_mut_slice().copy_from_slice(host);
    let host_in = start.elapsed().as_secs_f64() * 1000.0;
    gpu.upload_async(&pool.buffer.0, pinned.as_slice(), stream).ok()?;
    gpu.stream_sync(stream).ok()?;
    let start = Instant::now();
    gpu.upload_async(&pool.buffer.0, pinned.as_slice(), stream).ok()?;
    gpu.stream_sync(stream).ok()?;
    let up = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    gpu.download_async(&pool.buffer.0, pinned.as_mut_slice(), stream).ok()?;
    gpu.stream_sync(stream).ok()?;
    let down = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    let out = unsafe { std::slice::from_raw_parts_mut(rgb.as_mut_ptr() as *mut u8, l.n * 12) };
    out.copy_from_slice(pinned.as_slice());
    let host_out = start.elapsed().as_secs_f64() * 1000.0;
    Some((alloc, host_in + host_out, up, down))
}

/// Whether H2D and D2H actually overlap on two streams.
#[cfg(test)]
fn bidirectional_time(rgb: &mut [[f32; 3]], w: usize, h: usize) -> Option<(f64, f64)> {
    use std::time::Instant;
    let l = layout(w, h)?;
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    let State::Ready(pool) = &mut *state else {
        return None;
    };
    pool.ensure(l.total).ok()?;
    let gpu = pool.gpu.as_ref();
    let up_stream = pool.stream.as_ref().map(|stream| &stream.0)?;
    let down_stream = gpu.create_stream().ok()?;
    let host = unsafe { std::slice::from_raw_parts_mut(rgb.as_mut_ptr() as *mut u8, l.n * 12) };
    let mut registration = gpu.register_host(host).ok()?;
    let mut sink = vec![0u8; l.n * 12];
    let mut sink_registration = gpu.register_host(&mut sink).ok()?;
    // Distinct source and destination buffers so neither direction waits on a
    // read-after-write hazard inside the same allocation.
    let temp_up = gpu.alloc_buffer(l.n * 12).ok()?;
    let temp_down = gpu.alloc_buffer(l.n * 12).ok()?;
    gpu.upload_async(&temp_up, registration.as_slice(), up_stream).ok()?;
    gpu.stream_sync(up_stream).ok()?;
    let start = Instant::now();
    gpu.upload_async(&temp_up, registration.as_slice(), up_stream).ok()?;
    gpu.download_async(&temp_down, sink_registration.as_mut_slice(), &down_stream).ok()?;
    gpu.stream_sync(up_stream).ok()?;
    gpu.stream_sync(&down_stream).ok()?;
    let both = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    gpu.upload_async(&temp_up, registration.as_slice(), up_stream).ok()?;
    gpu.stream_sync(up_stream).ok()?;
    let up = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    gpu.download_async(&temp_down, sink_registration.as_mut_slice(), &down_stream).ok()?;
    gpu.stream_sync(&down_stream).ok()?;
    let down = start.elapsed().as_secs_f64() * 1000.0;
    Some((both, up + down))
}

/// Step split of the real `run_pool` path: register, upload, kernels,
/// download, unregister.
#[cfg(test)]
fn split_run(
    rgb: &mut [[f32; 3]],
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    scale: f32,
) -> Option<(f64, f64, f64, f64, f64)> {
    use std::time::Instant;
    let l = layout(w, h)?;
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    let State::Ready(pool) = &mut *state else {
        return None;
    };
    pool.ensure(l.total).ok()?;
    ensure_lut(pool, nr, true).ok()?;
    let gpu = pool.gpu.as_ref();
    let stream = pool.stream.as_ref().map(|stream| &stream.0)?;
    let host = unsafe { std::slice::from_raw_parts_mut(rgb.as_mut_ptr() as *mut u8, l.n * 12) };
    let start = Instant::now();
    let mut registration = gpu.register_host(host).ok()?;
    let register = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    gpu.upload_async(&pool.buffer.0, registration.as_slice(), stream).ok()?;
    gpu.stream_sync(stream).ok()?;
    let up = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    launch_stages(pool, w, h, nr, scale, true, true, Some(stream)).ok()?;
    let kernels = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    gpu.download_async(&pool.buffer.0, registration.as_mut_slice(), stream).ok()?;
    gpu.stream_sync(stream).ok()?;
    let down = start.elapsed().as_secs_f64() * 1000.0;
    let start = Instant::now();
    drop(registration);
    let unregister = start.elapsed().as_secs_f64() * 1000.0;
    Some((register, up, kernels, down, unregister))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::detail;

    fn available() -> bool {
        if !prewarm() {
            eprintln!("skipping: CUDA noise-reduction device unavailable");
            return false;
        }
        true
    }

    fn photo_like(w: usize, h: usize) -> Vec<[f32; 3]> {
        let mut out = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                let base = ((x / 37) as f32 * 0.21 + (y / 29) as f32 * 0.13) % 0.8 + 0.1;
                let n = detail::hash_noise(x as u32, y as u32, 11);
                let m = detail::hash_noise(x as u32 / 3, y as u32 / 3, 5);
                out.push([
                    (base + n * 0.12).clamp(0.0, 1.0),
                    (base * 0.96 + n * 0.10 + m * 0.04).clamp(0.0, 1.0),
                    (base * 0.9 + n * 0.08 + m * 0.06).clamp(0.0, 1.0),
                ]);
            }
        }
        out
    }

    fn code8(v: f32) -> i32 {
        (v.clamp(0.0, 1.0) * 255.0 + 0.5) as i32
    }

    fn compare(cpu: &[[f32; 3]], gpu: &[[f32; 3]]) -> (i32, f64) {
        let (mut max, mut sum, mut count) = (0i32, 0f64, 0usize);
        for (a, b) in cpu.iter().zip(gpu.iter()) {
            for k in 0..3 {
                let d = (code8(a[k]) - code8(b[k])).abs();
                max = max.max(d);
                sum += d as f64;
                count += 1;
            }
        }
        (max, sum / count as f64)
    }

    #[test]
    fn nr_gpu_matches_the_cpu_stages() {
        if !available() {
            return;
        }
        let (w, h) = (512usize, 384usize);
        let base = photo_like(w, h);
        let cases = [
            (
                "luma",
                NoiseReduction {
                    luminance: 50.0,
                    details: 50.0,
                    color: 0.0,
                    single_pixel: 0.0,
                },
            ),
            (
                "color",
                NoiseReduction {
                    luminance: 0.0,
                    details: 0.0,
                    color: 50.0,
                    single_pixel: 0.0,
                },
            ),
            (
                "both",
                NoiseReduction {
                    luminance: 50.0,
                    details: 50.0,
                    color: 50.0,
                    single_pixel: 0.0,
                },
            ),
        ];
        for (label, nr) in cases {
            let mut cpu = base.clone();
            if nr.luminance > 0.0 {
                detail::stage_luminance_nr_f32(&mut cpu, w, h, &nr);
            }
            if nr.color > 0.0 {
                detail::stage_color_nr_f32(&mut cpu, w, h, &nr, 1.0);
            }
            let mut gpu = base.clone();
            assert!(run_f32(&mut gpu, w, h, &nr, 1.0), "CUDA NR must run");
            let (max, mean) = compare(&cpu, &gpu);
            println!("NR_PARITY {label} max_code8={max} mean_code8={mean:.4}");
            assert!(max <= 1, "{label}: max code8 {max}");
            assert!(mean <= 0.1, "{label}: mean code8 {mean}");
        }
    }

    #[test]
    #[ignore = "timing benchmark, run explicitly with --ignored"]
    fn bench_nr_gpu_24mp() {
        use std::time::Instant;
        if !available() {
            return;
        }
        let (w, h) = (6048usize, 4032usize);
        let pixels = (w * h) as f64;
        let base = photo_like(w, h);
        let luma_nr = NoiseReduction {
            luminance: 50.0,
            details: 50.0,
            color: 0.0,
            single_pixel: 0.0,
        };
        let color_nr = NoiseReduction {
            luminance: 0.0,
            details: 0.0,
            color: 50.0,
            single_pixel: 0.0,
        };
        let both_nr = NoiseReduction {
            luminance: 50.0,
            details: 50.0,
            color: 50.0,
            single_pixel: 0.0,
        };

        let mut buf = base.clone();
        let t = Instant::now();
        detail::stage_luminance_nr_f32(&mut buf, w, h, &luma_nr);
        let cpu_luma = t.elapsed().as_secs_f64() * 1000.0;
        let mut buf = base.clone();
        let t = Instant::now();
        detail::stage_color_nr_f32(&mut buf, w, h, &color_nr, 1.0);
        let cpu_color = t.elapsed().as_secs_f64() * 1000.0;

        let mut buf = base.clone();
        let t = Instant::now();
        assert!(run_f32(&mut buf, w, h, &both_nr, 1.0));
        let gpu_both = t.elapsed().as_secs_f64() * 1000.0;
        // Same allocation reused (the viewer's "adjust and re-render" case)
        // versus a fresh Vec per frame (what the pipeline does today).
        let mut reused = base.clone();
        let mut reused_ms = Vec::new();
        for _ in 0..5 {
            let t = Instant::now();
            assert!(run_f32(&mut reused, w, h, &both_nr, 1.0));
            reused_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        reused_ms.sort_by(f64::total_cmp);
        let mut fresh_ms = Vec::new();
        for _ in 0..3 {
            let mut fresh = base.clone();
            let t = Instant::now();
            assert!(run_f32(&mut fresh, w, h, &both_nr, 1.0));
            fresh_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        fresh_ms.sort_by(f64::total_cmp);
        let mut buf = base.clone();
        let t = Instant::now();
        assert!(run_f32(&mut buf, w, h, &luma_nr, 1.0));
        let gpu_luma = t.elapsed().as_secs_f64() * 1000.0;
        let mut buf = base.clone();
        let t = Instant::now();
        assert!(run_f32(&mut buf, w, h, &color_nr, 1.0));
        let gpu_color = t.elapsed().as_secs_f64() * 1000.0;
        let dev_luma = device_time(&mut buf, w, h, &luma_nr, 1.0, true, false, 10).unwrap_or(-1.0);
        let dev_color =
            device_time(&mut buf, w, h, &color_nr, 1.0, false, true, 10).unwrap_or(-1.0);
        let dev_both = device_time(&mut buf, w, h, &both_nr, 1.0, true, true, 10).unwrap_or(-1.0);
        let (up, down) = io_time(&mut buf, w, h).unwrap_or((-1.0, -1.0));
        let staging = staging_time(&mut buf, w, h);
        let (reg_ms, async_up, async_down, unreg_ms) = staging.unwrap_or((-1.0, -1.0, -1.0, -1.0));
        let pinned = pinned_time(&mut buf, w, h);
        let (pin_alloc, pin_host, pin_up, pin_down) = pinned.unwrap_or((-1.0, -1.0, -1.0, -1.0));
        let (bidir, seq) = bidirectional_time(&mut buf, w, h).unwrap_or((-1.0, -1.0));
        let split = split_run(&mut buf, w, h, &both_nr, 1.0);
        let (s_reg, s_up, s_k, s_down, s_unreg) = split.unwrap_or((-1.0, -1.0, -1.0, -1.0, -1.0));

        println!(
            "NR_GPU_24MP cpu_luma_ms={cpu_luma:.2} cpu_color_ms={cpu_color:.2} \
cpu_total_ms={:.2} gpu_luma_ms={gpu_luma:.2} gpu_color_ms={gpu_color:.2} \
gpu_total_ms={gpu_both:.2} dev_luma_ms={dev_luma:.2} dev_color_ms={dev_color:.2} \
dev_total_ms={dev_both:.2} up_ms={up:.2} down_ms={down:.2} reg_ms={reg_ms:.2} \
async_up_ms={async_up:.2} async_down_ms={async_down:.2} unreg_ms={unreg_ms:.2} \
pin_alloc_ms={pin_alloc:.2} pin_host_ms={pin_host:.2} pin_up_ms={pin_up:.2} \
pin_down_ms={pin_down:.2} bidir_ms={bidir:.2} seq_ms={seq:.2} mpx={:.1}",
            cpu_luma + cpu_color,
            pixels / gpu_both / 1000.0
        );
        println!(
            "NR_GPU_REUSE reused_p50_ms={:.2} fresh_p50_ms={:.2}",
            reused_ms[2], fresh_ms[1]
        );
        println!(
            "NR_GPU_SPLIT reg={s_reg:.2} up={s_up:.2} kernels={s_k:.2} down={s_down:.2} \
unreg={s_unreg:.2} sum={:.2}",
            s_reg + s_up + s_k + s_down + s_unreg
        );
    }

    /// Real 24 MP NEF: parity and timing on decoded photo pixels.
    /// `AURORA_NR_NEF=<path> cargo test -p aurora-engine --lib
    /// nr_gpu_matches_cpu_on_real_nef -- --ignored --nocapture`.
    #[test]
    #[ignore = "needs a real NEF"]
    fn nr_gpu_matches_cpu_on_real_nef() {
        use std::path::Path;
        use std::time::Instant;
        if !available() {
            return;
        }
        let path = std::env::var("AURORA_NR_NEF")
            .unwrap_or_else(|_| r"C:\Users\Jxcz2\Desktop\_DSC5673.NEF".to_owned());
        let image = crate::raw::decode(Path::new(&path)).expect("decode NEF");
        let (w, h) = image.dimensions();
        let raw = image.into_raw();
        let base: Vec<[f32; 3]> = raw
            .chunks_exact(4)
            .map(|p| {
                [
                    p[0] as f32 / 255.0,
                    p[1] as f32 / 255.0,
                    p[2] as f32 / 255.0,
                ]
            })
            .collect();
        let (w, h) = (w as usize, h as usize);
        let pixels = (w * h) as f64;
        let cases = [
            (
                "luma",
                NoiseReduction {
                    luminance: 50.0,
                    details: 50.0,
                    color: 0.0,
                    single_pixel: 0.0,
                },
            ),
            (
                "color",
                NoiseReduction {
                    luminance: 0.0,
                    details: 0.0,
                    color: 50.0,
                    single_pixel: 0.0,
                },
            ),
            (
                "both",
                NoiseReduction {
                    luminance: 50.0,
                    details: 50.0,
                    color: 50.0,
                    single_pixel: 0.0,
                },
            ),
        ];
        for (label, nr) in cases {
            let mut cpu = base.clone();
            let t = Instant::now();
            if nr.luminance > 0.0 {
                detail::stage_luminance_nr_f32(&mut cpu, w, h, &nr);
            }
            if nr.color > 0.0 {
                detail::stage_color_nr_f32(&mut cpu, w, h, &nr, 1.0);
            }
            let cpu_ms = t.elapsed().as_secs_f64() * 1000.0;
            let mut gpu = base.clone();
            let t = Instant::now();
            assert!(run_f32(&mut gpu, w, h, &nr, 1.0), "CUDA NR must run");
            let gpu_ms = t.elapsed().as_secs_f64() * 1000.0;
            let (max, mean) = compare(&cpu, &gpu);
            println!(
                "NR_REAL {label} w={w} h={h} mpx={:.1} max_code8={max} mean_code8={mean:.4} \
cpu_ms={cpu_ms:.1} gpu_ms={gpu_ms:.1}",
                pixels / 1e6
            );
            assert!(max <= 1, "{label}: max code8 {max}");
            assert!(mean <= 0.1, "{label}: mean code8 {mean}");
        }
    }
}
