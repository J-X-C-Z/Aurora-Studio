//! Product-level Metal parity for PF90005 / PF90007 (SkinAdjust fixed LUT).
//!
//! These tests drive the real engine entry point
//! (`apply_portrait_effects_with_report_and_inputs`), not the library-level
//! host, so they prove the Metal branch is reachable from product code and that
//! its pixels equal the unchanged product CPU oracle
//! (`kernels::makeup::skin_adjust_lut`).
//!
//! All input is synthetic: a deterministic RGBA frame plus a synthetic skin
//! plane. No external model, photo or fallback mask is used.
//!
//! The Metal route is `#[cfg(target_os = "macos")]`, so on other platforms
//! every test is `ignore`d rather than passing vacuously.

use std::collections::BTreeMap;
use std::sync::{Mutex, MutexGuard};

use aurora_engine::{
    apply_portrait_effects_with_report_and_inputs, NormRect, PortraitBackend, PortraitJobInputs,
    PortraitJobReceipt, PortraitMasks,
};
use aurora_portrait::Image;

const WHITEN_PNG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/portrait/skin-adjust/whiten.png"
));
const RUDDY_PNG: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../assets/portrait/skin-adjust/ruddy_new.png"
));

/// Serializes tests that read/write the process-global `AURORA_*` dev switches.
/// Integration tests share one process, so this is required for correctness
/// (and matches the documented `--test-threads=1` verification command).
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK.lock().unwrap_or_else(|poison| poison.into_inner())
}

/// Restores the previous `AURORA_PORTRAIT_RENDER_BACKEND` value on drop.
struct RenderBackendGuard {
    previous: Option<std::ffi::OsString>,
}

impl RenderBackendGuard {
    fn force_cpu() -> Self {
        let previous = std::env::var_os("AURORA_PORTRAIT_RENDER_BACKEND");
        // Edition 2024: mutating process env is unsafe. The ENV_LOCK guard
        // ensures no test in this binary observes the value concurrently.
        unsafe {
            std::env::set_var("AURORA_PORTRAIT_RENDER_BACKEND", "cpu");
        }
        Self { previous }
    }
}

impl Drop for RenderBackendGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => unsafe {
                std::env::set_var("AURORA_PORTRAIT_RENDER_BACKEND", value);
            },
            None => unsafe {
                std::env::remove_var("AURORA_PORTRAIT_RENDER_BACKEND");
            },
        }
    }
}

/// Restores/removes `AURORA_PORTRAIT_FORCE_CPU` on drop.
struct ForceCpuGuard {
    previous: Option<std::ffi::OsString>,
}

impl ForceCpuGuard {
    fn set() -> Self {
        let previous = std::env::var_os("AURORA_PORTRAIT_FORCE_CPU");
        unsafe {
            std::env::set_var("AURORA_PORTRAIT_FORCE_CPU", "1");
        }
        Self { previous }
    }
}

impl Drop for ForceCpuGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(value) => unsafe {
                std::env::set_var("AURORA_PORTRAIT_FORCE_CPU", value);
            },
            None => unsafe {
                std::env::remove_var("AURORA_PORTRAIT_FORCE_CPU");
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReceiptView {
    kernel: String,
    grid: (u32, u32),
    threads: (u32, u32),
    dispatch_count: u32,
}

/// What the public report exposes, copied out without naming the (public but
/// module-private) receipt element type.
#[derive(Debug)]
struct RunView {
    bytes: Vec<u8>,
    backend: PortraitBackend,
    receipts: Vec<ReceiptView>,
    cpu_dispatches: usize,
    fallback_reason: Option<String>,
    skipped: bool,
    executed: bool,
}

fn run_engine(
    pf: u32,
    degree: f32,
    width: u32,
    height: u32,
    frame: &[u8],
    skin: Option<Vec<f32>>,
) -> RunView {
    let masks = PortraitMasks {
        skin,
        hair: None,
        face: None,
        body: None,
        width: width as usize,
        height: height as usize,
    };
    let params = BTreeMap::from([(format!("pc:{pf}"), degree)]);
    let receipt =
        PortraitJobReceipt::from_renderer_receipt(frame, width, height, NormRect::FULL, &params, &masks);
    let inputs = PortraitJobInputs::from_renderer_receipt(receipt, &masks);

    let mut bytes = frame.to_vec();
    let report = apply_portrait_effects_with_report_and_inputs(
        &mut bytes,
        width,
        height,
        NormRect::FULL,
        &params,
        &inputs,
    );
    assert_eq!(report.effects.len(), 1, "PF{pf} must produce one row");
    let effect = &report.effects[0];
    let receipts = effect
        .metal_dispatches
        .iter()
        .map(|dispatch| ReceiptView {
            kernel: dispatch.kernel.to_string(),
            grid: dispatch.grid,
            threads: dispatch.threads,
            dispatch_count: dispatch.dispatch_count,
        })
        .collect();
    RunView {
        bytes,
        backend: effect.backend,
        receipts,
        cpu_dispatches: effect.dispatches.len(),
        fallback_reason: effect.fallback_reason.clone(),
        skipped: effect.skipped,
        executed: effect.executed,
    }
}

fn decode_lut(png: &[u8], name: &str) -> Vec<u8> {
    let decoded = image::load_from_memory(png)
        .unwrap_or_else(|error| panic!("decode {name}: {error}"))
        .into_rgb8();
    assert_eq!(
        (decoded.width(), decoded.height()),
        (512, 512),
        "{name} must be a 512x512 atlas"
    );
    let bytes = decoded.into_raw();
    assert_eq!(bytes.len(), 512 * 512 * 3, "{name} atlas bytes");
    bytes
}

fn synthetic_frame(width: u32, height: u32) -> Vec<u8> {
    let pixels = width as usize * height as usize;
    let mut out = vec![0u8; pixels * 4];
    for (index, pixel) in out.chunks_exact_mut(4).enumerate() {
        let x = (index as u32) % width;
        let y = (index as u32) / width;
        pixel[0] = ((x * 53 + y * 17 + 11) % 256) as u8;
        pixel[1] = ((x * 29 + y * 61 + 97) % 256) as u8;
        pixel[2] = ((x * 97 + y * 13 + 191) % 256) as u8;
        // Non-opaque alpha proves the alpha channel is copied from source.
        pixel[3] = 255 - ((x + y) % 17) as u8;
    }
    out
}

fn synthetic_mask_plane(width: u32, height: u32) -> Vec<f32> {
    let pixels = width as usize * height as usize;
    (0..pixels)
        .map(|index| ((index * 37 + 13) % 256) as f32 / 255.0)
        .collect()
}

/// Exact replica of `PortraitMasks::mask_u8` for a plane whose dimensions equal
/// the output frame and `NormRect::FULL`: the bilinear sample lands on the
/// source pixel center, so only the quantization remains.
fn mask_u8_from_plane(plane: &[f32]) -> Vec<u8> {
    plane
        .iter()
        .map(|value| (value.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
        .collect()
}

const SIZES: &[(u32, u32)] = &[(1, 1), (17, 13), (64, 37), (37, 64)];
const DEGREES: &[f32] = &[0.0, 0.5, 1.0];

#[test]
#[cfg_attr(not(target_os = "macos"), ignore = "macOS Metal route only")]
fn metal_skin_lut_matches_cpu_byte_for_byte() {
    let _env = env_lock();
    let luts = [
        (90005u32, decode_lut(WHITEN_PNG, "whiten.png")),
        (90007u32, decode_lut(RUDDY_PNG, "ruddy_new.png")),
    ];

    for (pf, lut) in &luts {
        for &(width, height) in SIZES {
            let frame = synthetic_frame(width, height);
            let plane = synthetic_mask_plane(width, height);
            let mask = mask_u8_from_plane(&plane);
            let source = Image::new(width, height, frame.clone()).expect("synthetic frame");

            for &degree in DEGREES {
                // Unchanged product CPU oracle (the exact function
                // `apply_skin_whiten`/`apply_skin_ruddy` call).
                let cpu = aurora_portrait::kernels::makeup::skin_adjust_lut(
                    &source,
                    &mask,
                    lut,
                    degree.clamp(0.0, 1.0),
                )
                .expect("CPU oracle");

                // CPU-only engine run with Metal forced off.
                let forced_cpu = {
                    let _force = RenderBackendGuard::force_cpu();
                    run_engine(*pf, degree, width, height, &frame, Some(plane.clone()))
                };
                assert_eq!(
                    forced_cpu.backend,
                    PortraitBackend::Cpu,
                    "PF{pf} {width}x{height} degree={degree} forced-CPU backend"
                );
                assert!(
                    forced_cpu.receipts.is_empty(),
                    "PF{pf} forced CPU must not report Metal"
                );
                assert_eq!(forced_cpu.cpu_dispatches, 0);
                assert_eq!(
                    forced_cpu.bytes, cpu.pixels,
                    "PF{pf} {width}x{height} degree={degree} engine CPU vs product oracle"
                );

                // Metal engine run.
                let metal = run_engine(*pf, degree, width, height, &frame, Some(plane.clone()));
                assert_eq!(
                    metal.backend,
                    PortraitBackend::Metal,
                    "PF{pf} {width}x{height} degree={degree} backend; fallback={:?}",
                    metal.fallback_reason
                );
                assert!(
                    !metal.receipts.is_empty(),
                    "PF{pf} Metal row must carry a non-empty dispatch"
                );
                let receipt = &metal.receipts[0];
                assert_eq!(receipt.kernel, "skin_lut_fixed");
                assert_eq!(
                    receipt.grid,
                    (width.div_ceil(16), height.div_ceil(16)),
                    "PF{pf} {width}x{height} threadgroups"
                );
                assert_eq!(receipt.threads, (16, 16));
                assert!(receipt.dispatch_count > 0);
                assert_eq!(metal.cpu_dispatches, 0, "Metal row must not claim CUDA");
                assert!(
                    metal.fallback_reason.is_none(),
                    "Metal success must not record a fallback: {:?}",
                    metal.fallback_reason
                );
                let max_byte_diff = metal
                    .bytes
                    .iter()
                    .zip(&cpu.pixels)
                    .map(|(metal, cpu)| metal.abs_diff(*cpu))
                    .max()
                    .unwrap_or(0);
                eprintln!(
                    "PF{pf} {width}x{height} degree={degree} backend={:?} kernel={} grid={:?} threads={:?} dispatch_count={} max_byte_diff={max_byte_diff}",
                    metal.backend,
                    receipt.kernel,
                    receipt.grid,
                    receipt.threads,
                    receipt.dispatch_count
                );
                assert_eq!(
                    max_byte_diff, 0,
                    "PF{pf} {width}x{height} degree={degree} Metal/CPU byte parity (max diff {max_byte_diff})"
                );

                // Determinism: identical run yields identical bytes/receipt.
                let again = run_engine(*pf, degree, width, height, &frame, Some(plane.clone()));
                assert_eq!(
                    again.bytes, metal.bytes,
                    "PF{pf} {width}x{height} degree={degree} repeat is deterministic"
                );
                assert_eq!(again.backend, PortraitBackend::Metal);
                assert_eq!(
                    again.receipts, metal.receipts,
                    "PF{pf} repeat receipt is deterministic"
                );
            }
        }
    }
}

#[test]
#[cfg_attr(not(target_os = "macos"), ignore = "macOS Metal route only")]
fn all_zero_skin_mask_is_identity() {
    let _env = env_lock();
    for pf in [90005u32, 90007] {
        for &(width, height) in &[(1u32, 1u32), (17, 13), (64, 37)] {
            let frame = synthetic_frame(width, height);
            let plane = vec![0.0f32; width as usize * height as usize];
            let run = run_engine(pf, 1.0, width, height, &frame, Some(plane));
            assert_eq!(
                run.bytes, frame,
                "PF{pf} {width}x{height} all-zero mask must be identity"
            );
            assert_eq!(run.backend, PortraitBackend::Metal);
            assert!(!run.receipts.is_empty());
        }
    }
}

#[test]
#[cfg_attr(not(target_os = "macos"), ignore = "macOS Metal route only")]
fn missing_skin_mask_leaves_cpu_bytes_unchanged() {
    let _env = env_lock();
    for pf in [90005u32, 90007] {
        let (width, height) = (17u32, 13u32);
        let frame = synthetic_frame(width, height);
        let run = run_engine(pf, 0.9, width, height, &frame, None);
        assert_eq!(run.bytes, frame, "PF{pf} missing mask must not alter bytes");
        assert_eq!(run.backend, PortraitBackend::Skipped);
        assert!(run.receipts.is_empty());
        assert!(run.skipped);
        assert!(!run.executed);
    }
}

/// Regression: the CPU lane is exactly the pre-change product function and is
/// unaffected by adding the Metal branch. Both existing dev switches that mean
/// "do not use the GPU" must reach it with an honest fallback reason.
#[test]
#[cfg_attr(not(target_os = "macos"), ignore = "macOS Metal route only")]
fn forced_cpu_fallback_is_byte_identical_and_reasoned() {
    let _env = env_lock();
    let luts = [
        (90005u32, decode_lut(WHITEN_PNG, "whiten.png")),
        (90007u32, decode_lut(RUDDY_PNG, "ruddy_new.png")),
    ];
    let (width, height) = (64u32, 37u32);
    let frame = synthetic_frame(width, height);
    let plane = synthetic_mask_plane(width, height);
    let mask = mask_u8_from_plane(&plane);
    let source = Image::new(width, height, frame.clone()).expect("synthetic frame");

    for (pf, lut) in &luts {
        for &degree in &[0.25f32, 0.75, 1.0] {
            let expected = aurora_portrait::kernels::makeup::skin_adjust_lut(
                &source,
                &mask,
                lut,
                degree.clamp(0.0, 1.0),
            )
            .expect("CPU oracle")
            .pixels;

            let via_render_backend = {
                let _force = RenderBackendGuard::force_cpu();
                run_engine(*pf, degree, width, height, &frame, Some(plane.clone()))
            };
            assert_eq!(via_render_backend.backend, PortraitBackend::Cpu);
            assert_eq!(via_render_backend.bytes, expected, "PF{pf} degree={degree}");
            assert!(via_render_backend.receipts.is_empty());
            assert!(
                via_render_backend
                    .fallback_reason
                    .as_deref()
                    .is_some_and(|reason| reason.contains("AURORA_PORTRAIT_RENDER_BACKEND=cpu")),
                "PF{pf} fallback must name the switch: {:?}",
                via_render_backend.fallback_reason
            );

            let via_force_cpu = {
                let _force = ForceCpuGuard::set();
                run_engine(*pf, degree, width, height, &frame, Some(plane.clone()))
            };
            assert_eq!(via_force_cpu.backend, PortraitBackend::Cpu);
            assert_eq!(via_force_cpu.bytes, expected, "PF{pf} degree={degree}");
            assert!(via_force_cpu.receipts.is_empty());
            assert!(
                via_force_cpu
                    .fallback_reason
                    .as_deref()
                    .is_some_and(|reason| reason.contains("AURORA_PORTRAIT_FORCE_CPU")),
                "PF{pf} fallback must name the switch: {:?}",
                via_force_cpu.fallback_reason
            );
        }
    }
}

/// The Metal bridge only owns the SkinAdjust LUT family. PF21024 (P15) must
/// never be reported as Metal even though the library host lists it.
#[test]
#[cfg_attr(not(target_os = "macos"), ignore = "macOS Metal route only")]
fn p15_21024_is_not_claimed_by_metal() {
    let _env = env_lock();
    let (width, height) = (17u32, 13u32);
    let frame = synthetic_frame(width, height);
    let plane = synthetic_mask_plane(width, height);
    let run = run_engine(21024, 0.5, width, height, &frame, Some(plane));
    assert_ne!(
        run.backend,
        PortraitBackend::Metal,
        "PF21024 must not be routed through Metal: {:?}",
        run.receipts
    );
    assert!(run.receipts.is_empty());
}
