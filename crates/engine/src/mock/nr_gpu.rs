//! Optional CUDA path for the two production noise stages.
//!
//! `stage_luminance_nr_f32` (C1 `LumaNoise2018`) and `stage_color_nr_f32`
//! (C1 `ColorNoise2012` / `LABRangeFilter2012`) are the heaviest full-frame
//! CPU stages of the display-sRGB float core. The kernels here mirror their
//! arithmetic; `AURORA_NR_GPU=0` keeps the CPU stages and leaves every byte
//! untouched. Any driver, module or launch failure also falls back to CPU.
//!
//! This is the only stage that uses the shared non-blocking stream and pins the
//! caller's frame in place; both come from [`super::gpu_runtime::Gpu`].

use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use super::gpu_runtime::{self, Failure, Gpu, Slot};

use crate::{NoiseReduction};

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

struct NrPool {
    gpu: Gpu,
    lut_sigma: Option<f32>,
    bytes: usize,
}

impl NrPool {
    fn open(w: usize, h: usize) -> Result<Self, Failure> {
        let l = layout(w, h).ok_or(Failure::Invalid("nr frame size"))?;
        let mut gpu = Gpu::open("nr", PTX)?.with_stream();
        gpu.reserve(Slot::Frame, l.total, MAX_POOL_BYTES)?;
        gpu.reserve(Slot::Aux0, 1025 * 4, MAX_POOL_BYTES)?;
        Ok(Self {
            gpu,
            lut_sigma: None,
            bytes: l.total,
        })
    }

    fn ensure(&mut self, total: usize) -> Result<(), Failure> {
        if total > self.bytes {
            self.gpu.reserve(Slot::Frame, total, MAX_POOL_BYTES)?;
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
            Err(failure) => {
                log::warn!("nr_gpu: {failure:?}; staying on CPU");
                gpu_runtime::note_fallback("nr", &failure);
                *state = State::Unavailable;
                false
            }
        },
    }
}

/// Runs the luma and/or colour noise reduction kernels in place on a
/// device-resident `[f32; 3]` frame (`p_rgb_dev`), enqueuing every kernel onto
/// `stream` with zero host-device pixel transfers and zero mid-frame sync.
pub(crate) fn run_resident_on_stream(
    p_rgb_dev: u64,
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    scale: f32,
    stream: &aurora_portrait::cuda::Stream<'_>,
) -> bool {
    let luma = nr.luminance > 0.0;
    let color = nr.color > 0.0;
    if (!luma && !color) || !enabled() {
        return true;
    }
    let Some(l) = layout(w, h) else { return false };
    let scratch_bytes = l.pad.saturating_add(l.plane.saturating_mul(4));
    if scratch_bytes == 0 || scratch_bytes > MAX_POOL_BYTES {
        return false;
    }
    let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
    let result = match &mut *state {
        State::Unavailable => return false,
        State::NotTried => match NrPool::open(w, h) {
            Ok(mut pool) => {
                let res = pool
                    .gpu
                    .reserve(Slot::Scratch, scratch_bytes, MAX_POOL_BYTES)
                    .and_then(|scratch_base| {
                        ensure_lut(&mut pool, nr, luma)?;
                        launch_stages_at(
                            &mut pool.gpu,
                            Some(stream),
                            p_rgb_dev,
                            scratch_base,
                            w,
                            h,
                            nr,
                            scale,
                            luma,
                            color,
                        )
                    });
                if res.is_ok() {
                    *state = State::Ready(pool);
                }
                res
            }
            Err(failure) => {
                log::warn!("nr_gpu: {failure:?}; staying on CPU");
                gpu_runtime::note_fallback("nr", &failure);
                *state = State::Unavailable;
                return false;
            }
        },
        State::Ready(pool) => pool
            .gpu
            .reserve(Slot::Scratch, scratch_bytes, MAX_POOL_BYTES)
            .and_then(|scratch_base| {
                ensure_lut(pool, nr, luma)?;
                launch_stages_at(
                    &mut pool.gpu,
                    Some(stream),
                    p_rgb_dev,
                    scratch_base,
                    w,
                    h,
                    nr,
                    scale,
                    luma,
                    color,
                )
            }),
    };
    match result {
        Ok(()) => true,
        Err(failure) => {
            log::warn!("nr_gpu resident: {failure:?}; falling back");
            gpu_runtime::note_fallback("nr", &failure);
            false
        }
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
    let result = match &mut *state {
        State::Unavailable => return false,
        State::NotTried => match NrPool::open(w, h) {
            Ok(mut pool) => {
                let result = pool
                    .ensure(l.total)
                    .and_then(|()| run_pool(&mut pool, rgb, w, h, nr, scale, luma, color));
                if result.is_ok() {
                    *state = State::Ready(pool);
                }
                result
            }
            Err(failure) => {
                log::warn!("nr_gpu: {failure:?}; staying on CPU");
                gpu_runtime::note_fallback("nr", &failure);
                *state = State::Unavailable;
                return false;
            }
        },
        State::Ready(pool) => pool
            .ensure(l.total)
            .and_then(|()| run_pool(pool, rgb, w, h, nr, scale, luma, color)),
    };
    match result {
        Ok(()) => true,
        Err(failure) => {
            log::warn!("nr_gpu: {failure:?}; staying on CPU");
            gpu_runtime::note_fallback("nr", &failure);
            false
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
) -> Result<(), Failure> {
    ensure_lut(pool, nr, luma)?;
    let l = layout(w, h).ok_or(Failure::Invalid("nr frame size"))?;
    let host = unsafe { std::slice::from_raw_parts_mut(rgb.as_mut_ptr() as *mut u8, l.n * 12) };
    if pool.gpu.stream().is_some()
        && let Ok(mut registration) = pool.gpu.register_host(host)
    {
        // Pinned in place: the DMA engine reads and writes the caller's frame
        // directly, with no 290 MB host-to-host staging copy.
        pool.gpu.upload_async(Slot::Frame, registration.as_slice())?;
        launch_stages(&mut pool.gpu, w, h, nr, scale, luma, color)?;
        pool.gpu.download_async(Slot::Frame, registration.as_mut_slice())?;
        pool.gpu.sync()?;
        return Ok(());
    }
    pool.gpu.upload(Slot::Frame, host)?;
    launch_stages(&mut pool.gpu, w, h, nr, scale, luma, color)?;
    pool.gpu.download(Slot::Frame, host)?;
    Ok(())
}

/// Uploads the luma range LUT only when the slider-derived sigma changed; a
/// synchronous 4 KB copy mid-frame would otherwise drain the async pipeline.
fn ensure_lut(
    pool: &mut NrPool,
    nr: &NoiseReduction,
    luma: bool,
) -> Result<(), Failure> {
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
    pool.gpu.upload(Slot::Aux0, table_bytes)?;
    pool.lut_sigma = Some(sigma);
    Ok(())
}

fn launch_stages(
    gpu: &mut Gpu,
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    scale: f32,
    luma: bool,
    color: bool,
) -> Result<(), Failure> {
    let l = layout(w, h).ok_or(Failure::Invalid("nr frame size"))?;
    let base = gpu.handle(Slot::Frame).ok_or(Failure::Invalid("nr frame slot"))?;
    launch_stages_at(gpu, None, base, base + l.rgb as u64, w, h, nr, scale, luma, color)
}

#[allow(clippy::too_many_arguments)]
fn launch_stages_at(
    gpu: &mut Gpu,
    external_stream: Option<&aurora_portrait::cuda::Stream<'_>>,
    p_rgb_in: u64,
    scratch_base: u64,
    w: usize,
    h: usize,
    nr: &NoiseReduction,
    scale: f32,
    luma: bool,
    color: bool,
) -> Result<(), Failure> {
    let l = layout(w, h).ok_or(Failure::Invalid("nr frame size"))?;
    let mut p_rgb = p_rgb_in;
    let mut p_pad = scratch_base;
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
        let mut p_lut = gpu.handle(Slot::Aux0).ok_or(Failure::Invalid("nr lut slot"))?;
        let mut strength_f = strength;

        launch2_on(
            gpu,
            external_stream,
            "nr_luma_pad_kernel",
            w + 2,
            h + 2,
            [
                ptr(&mut p_rgb),
                ptr(&mut p_pad),
                ptr(&mut w_i),
                ptr(&mut h_i),
            ],
        )?;
        launch2_on(
            gpu,
            external_stream,
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

        launch1_on(
            gpu,
            external_stream,
            "nr_color_encode_kernel",
            l.n,
            [
                ptr(&mut p_rgb),
                ptr(&mut p_l),
                ptr(&mut p_u),
                ptr(&mut p_v),
                ptr(&mut n_i),
            ],
        )?;
        launch2_on(
            gpu,
            external_stream,
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
        )?;
        launch2_on(
            gpu,
            external_stream,
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
        )?;
    }

    Ok(())
}

fn ptr<T>(value: &mut T) -> *mut c_void {
    value as *mut T as *mut c_void
}

fn launch2_on<const N: usize>(
    gpu: &mut Gpu,
    stream: Option<&aurora_portrait::cuda::Stream<'_>>,
    kernel: &'static str,
    w: usize,
    h: usize,
    mut args: [*mut c_void; N],
) -> Result<(), Failure> {
    let grid = gpu_runtime::grid2(w, h, BLOCK);
    match stream {
        Some(stream) => gpu.launch_on_stream(stream, kernel, grid, BLOCK, &mut args),
        None => gpu.launch(kernel, grid, BLOCK, &mut args),
    }
}

fn launch1_on<const N: usize>(
    gpu: &mut Gpu,
    stream: Option<&aurora_portrait::cuda::Stream<'_>>,
    kernel: &'static str,
    n: usize,
    mut args: [*mut c_void; N],
) -> Result<(), Failure> {
    let grid = gpu_runtime::grid1(n, 256);
    match stream {
        Some(stream) => gpu.launch_on_stream(stream, kernel, grid, (256, 1), &mut args),
        None => gpu.launch(kernel, grid, (256, 1), &mut args),
    }
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

    /// Device-only time: `iters` passes are queued, then one 4-byte readback
    /// drains them; the calibrated idle readback is subtracted.
    fn device_time(pool: &mut NrPool, w: usize, h: usize, nr: &NoiseReduction, luma: bool, color: bool, iters: usize) -> Option<f64> {
        use std::time::Instant;
        ensure_lut(pool, nr, luma).ok()?;
        let mut sink = [0u8; 4];
        let idle = {
            let started = Instant::now();
            pool.gpu.download(Slot::Aux0, &mut sink).ok()?;
            started.elapsed().as_secs_f64() * 1000.0
        };
        let started = Instant::now();
        for _ in 0..iters.max(1) {
            launch_stages(&mut pool.gpu, w, h, nr, 1.0, luma, color).ok()?;
        }
        pool.gpu.download(Slot::Aux0, &mut sink).ok()?;
        let per_pass = started.elapsed().as_secs_f64() * 1000.0 / iters.max(1) as f64;
        Some((per_pass - idle / iters.max(1) as f64).max(0.0))
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
        // Same allocation reused (the viewer's "adjust and re-render" case).
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

        let mut state = state().lock().unwrap_or_else(|error| error.into_inner());
        let State::Ready(pool) = &mut *state else { return };
        let dev_luma = device_time(pool, w, h, &luma_nr, true, false, 10).unwrap_or(-1.0);
        let dev_color = device_time(pool, w, h, &color_nr, false, true, 10).unwrap_or(-1.0);
        let dev_both = device_time(pool, w, h, &both_nr, true, true, 10).unwrap_or(-1.0);
        let counters = pool.gpu.counters();
        let kernels: Vec<&'static str> = pool.gpu.kernels().iter().map(|k| k.kernel).collect();

        println!(
            "NR_GPU_24MP cpu_luma_ms={cpu_luma:.2} cpu_color_ms={cpu_color:.2} \
cpu_total_ms={:.2} gpu_total_ms={gpu_both:.2} dev_luma_ms={dev_luma:.2} \
dev_color_ms={dev_color:.2} dev_total_ms={dev_both:.2} mpx={:.1} \
h2d_bytes={} d2h_bytes={} launches={} kernels={kernels:?}",
            cpu_luma + cpu_color,
            pixels / gpu_both / 1000.0,
            counters.h2d_bytes,
            counters.d2h_bytes,
            counters.launches,
        );
        println!(
            "NR_GPU_REUSE reused_p50_ms={:.2} fresh_p50_ms={:.2}",
            reused_ms[2], fresh_ms[1]
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
