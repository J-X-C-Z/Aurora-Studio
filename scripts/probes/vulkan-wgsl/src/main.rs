//! Vulkan compute probe — standalone fixture, not part of the engine build.
//!
//! WHAT THIS IS
//!   WGSL compute kernels, compiled to SPIR-V with `naga` (WGSL stays the single
//!   source of truth) and submitted to real Vulkan hardware through `ash`. Two
//!   measurable claims:
//!     1. integer RGBA8 stage fusion — two stages, one dispatch, byte-exact
//!        against a CPU reference that runs them serially with per-stage 8-bit
//!        quantization, with the intermediate buffer poisoned to prove the fused
//!        path never touches it;
//!     2. a small FP32 weighted matmul (Gemm + bias) within 1e-4 of an FP32 CPU
//!        reference, plus an FMA reference so a difference can be attributed to
//!        driver contraction instead of a wrong kernel.
//!
//! WHAT THIS IS NOT
//!   Not a production compute backend, not a renderer, not a benchmark: the
//!   timings printed are single-case submit+wait+readback wall clock on a warm
//!   device and are not comparable across APIs or targets. Every buffer is
//!   host-visible coherent (map/unmap, no staging copies), which is the shape of
//!   a fixture, not of a shipping backend.
//!
//! DEVICE SELECTION
//!   The process loads only `vulkan-1.dll` and enumerates only Vulkan devices.
//!   By default every non-software device is measured; `C_PROBE_VENDORS` narrows
//!   that to a vendor list for reproducing a specific hardware matrix. Software
//!   adapters (deviceType == CPU, e.g. lavapipe) are enumerated, printed and
//!   rejected as measurement targets.
//!
//! USAGE
//!   <probe> [out_dir] [tag]      writes per-device JSON, inventory + summary
//!   C_PROBE_ONLY_INVENTORY=1     enumerate only, measure nothing
//!   C_PROBE_VENDORS=0x10de,...   measure only these vendor ids (absent -> fail)
//!
//! Exit codes: 0 pass, 1 failure / nothing measured, 2 shader compile error,
//! 3 Vulkan unavailable or enumeration failed.

mod refs;
mod shader;
mod vk;

use refs::{
    diff_rgba8, make_fixture, max_abs_f32, max_abs_f32_f64, FusionParams, MatMulCase, Rgba8Diff,
    LEVELS, MATMUL_SHAPES, MID_POISON, PARAM_WORDS, SIZES,
};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use vk::{DeviceInfo, Gpu, VkCtx};

const FUSION_WGSL: &str = include_str!("../shaders/fusion.wgsl");
const MATMUL_WGSL: &str = include_str!("../shaders/matmul.wgsl");

// ---------------------------------------------------------------------------
// result records
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct Window {
    submits: u64,
    dispatches: u64,
    downloads: u64,
    upload_bytes: u64,
    download_bytes: u64,
}

fn win_of(c: vk::Counters) -> Window {
    Window {
        submits: c.submits,
        dispatches: c.dispatches,
        downloads: c.downloads,
        upload_bytes: c.upload_bytes,
        download_bytes: c.download_bytes,
    }
}

struct FusionCase {
    width: u32,
    height: u32,
    level: u32,
    mode: &'static str,
    win: Window,
    exact: bool,
    diff: Rgba8Diff,
    mid_state: &'static str,
    ref_invariant: bool,
    elapsed_us: u128,
}

struct MatMulResult {
    m: usize,
    k: usize,
    n: usize,
    win: Window,
    max_abs_f32: f32,
    max_abs_fma: f32,
    max_abs_f64: f64,
    exact_vs_f32_ref: bool,
    elapsed_us: u128,
}

struct Unsupported {
    vendor: u32,
    label: String,
    reason: String,
}

struct DeviceRun {
    info: DeviceInfo,
    status: &'static str,
    error: Option<String>,
    fusion: Vec<FusionCase>,
    matmul: Vec<MatMulResult>,
    queue_family: u32,
}

// ---------------------------------------------------------------------------
// fusion execution
// ---------------------------------------------------------------------------

/// One fusion case. `elapsed_us` is wall clock from command-buffer begin to the
/// first readback of the result (submit + fence wait + one map/copy); it is a
/// per-case figure, not a benchmark sample.
struct FusionRun {
    dst: Vec<u32>,
    mid: Vec<u32>,
    win: Window,
    elapsed_us: u128,
}

/// Buffers are per-case locals: they free themselves on drop, so a run never
/// accumulates allocations across the 6x3x2 case matrix.
fn run_fusion(
    gpu: &mut Gpu,
    pipe: &[ash::vk::Pipeline],
    prm: &FusionParams,
    src: &[u32],
    fused: bool,
) -> Result<FusionRun, String> {
    let n = src.len();
    let bytes = (n * 4) as u64;
    let usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER;

    let src_buf = gpu.create_buf(bytes, usage)?;
    let dst_buf = gpu.create_buf(bytes, usage)?;
    let mid_buf = gpu.create_buf(bytes, usage)?;
    let prm_buf = gpu.create_buf((PARAM_WORDS * 4).max(64) as u64, usage)?;

    gpu.upload_u32(&src_buf, src)?;
    gpu.upload_u32(&mid_buf, &vec![MID_POISON; n])?;
    gpu.upload_u32(&prm_buf, &prm.to_words())?;

    let set = gpu.alloc_set(&[&src_buf, &dst_buf, &mid_buf, &prm_buf, &prm_buf])?;
    let groups = (
        prm.width.div_ceil(8),
        prm.height.div_ceil(8),
    );

    let base = gpu.snapshot();
    let t0 = Instant::now();
    gpu.begin()?;
    gpu.barrier_host_to_compute();
    if fused {
        gpu.dispatch(pipe[0], set, groups);
    } else {
        gpu.dispatch(pipe[0], set, groups);
        gpu.barrier_compute_to_compute();
        gpu.dispatch(pipe[1], set, groups);
    }
    gpu.submit_and_wait()?;
    let dst = gpu.download_u32(&dst_buf, n)?;
    // Measured window ends here: begin -> submit -> fence -> first readback.
    let elapsed_us = t0.elapsed().as_micros();
    let win = win_of(gpu.since(&base));

    // Verification read of the intermediate image, deliberately outside the
    // measured window (it is not part of the kernel's work).
    let mid = gpu.download_u32(&mid_buf, n)?;

    Ok(FusionRun { dst, mid, win, elapsed_us })
}

fn matmul_run(
    gpu: &mut Gpu,
    pipe: ash::vk::Pipeline,
    case: &MatMulCase,
) -> Result<(Vec<f32>, Window, u128), String> {
    let usage = ash::vk::BufferUsageFlags::STORAGE_BUFFER;
    let a_buf = gpu.create_buf((case.m * case.k * 4) as u64, usage)?;
    let b_buf = gpu.create_buf((case.k * case.n * 4) as u64, usage)?;
    let c_buf = gpu.create_buf((case.m * case.n * 4) as u64, usage)?;
    let dims_buf = gpu.create_buf(16, usage)?;
    let bias_buf = gpu.create_buf((case.n * 4) as u64, usage)?;

    gpu.upload_f32(&a_buf, &case.a)?;
    gpu.upload_f32(&b_buf, &case.b)?;
    gpu.upload_f32(&bias_buf, &case.bias)?;
    gpu.upload_u32(
        &dims_buf,
        &[case.m as u32, case.k as u32, case.n as u32, 0],
    )?;

    let set = gpu.alloc_set(&[&a_buf, &b_buf, &c_buf, &dims_buf, &bias_buf])?;
    let groups = (case.m.div_ceil(8) as u32, case.n.div_ceil(8) as u32);

    let base = gpu.snapshot();
    let t0 = Instant::now();
    gpu.begin()?;
    gpu.barrier_host_to_compute();
    gpu.dispatch(pipe, set, groups);
    gpu.submit_and_wait()?;
    let c = gpu.download_f32(&c_buf, case.m * case.n)?;
    let elapsed_us = t0.elapsed().as_micros();
    let win = win_of(gpu.since(&base));
    Ok((c, win, elapsed_us))
}

// ---------------------------------------------------------------------------
// per-device driver
// ---------------------------------------------------------------------------

fn run_device(
    ctx: &VkCtx,
    pd: ash::vk::PhysicalDevice,
    info: DeviceInfo,
    fusion_spv: &[u32],
    matmul_spv: &[u32],
) -> DeviceRun {
    let mut run = DeviceRun {
        info: info.clone(),
        status: "error",
        error: None,
        fusion: Vec::new(),
        matmul: Vec::new(),
        queue_family: 0,
    };

    let attempt = (|| -> Result<(), String> {
        let mut gpu = Gpu::open(ctx, pd, info.clone())?;
        run.queue_family = gpu.queue_family();
        // One pipeline per WGSL entry point, reused by every case. Released at
        // the end of this block; an early error leaves them to vkDestroyDevice.
        let p_fused = gpu.make_pipeline(fusion_spv, "op_ab")?;
        let p_op_a = gpu.make_pipeline(fusion_spv, "op_a")?;
        let p_op_b = gpu.make_pipeline(fusion_spv, "op_b")?;
        let p_matmul = gpu.make_pipeline(matmul_spv, "matmul_bias")?;

        println!(
            "\n--- device {}:{} {} [{}] api {} driver {} (driverVersion 0x{:08x})",
            info.vendor_hex(),
            info.device_hex(),
            info.name,
            info.type_str(),
            info.api_str(),
            info.driver_version_human,
            info.driver_version
        );
        println!(
            "    compute queue family {} | maxComputeWorkGroupInvocations {} | \
             workgroup {}x{}x{} | maxStorageBufferRange {} B | push constants {} B",
            gpu.queue_family(),
            info.max_compute_invocations,
            info.max_compute_workgroup[0],
            info.max_compute_workgroup[1],
            info.max_compute_workgroup[2],
            info.max_storage_buffer_range,
            info.max_push_constants
        );
        println!(
            "    shaderFloat16={} shaderFloat64={} shaderInt8={}",
            info.shader_float16, info.shader_float64, info.shader_int8
        );

        for &(w, h) in SIZES {
            for &level in LEVELS {
                let prm = FusionParams::for_level(w, h, level);
                let src = make_fixture(w, h);
                let (mid_ref, dst_ref) = refs::reference_serial(&src, &prm);
                let dst_fused_ref = refs::reference_fused(&src, &prm);
                let ref_invariant = dst_ref == dst_fused_ref;

                for fused in [true, false] {
                    let chain: [ash::vk::Pipeline; 2] = if fused {
                        [p_fused, p_op_b]
                    } else {
                        [p_op_a, p_op_b]
                    };
                    let out = run_fusion(&mut gpu, &chain, &prm, &src, fused)?;
                    let diff = diff_rgba8(&out.dst, &dst_ref);
                    let mid_state = if fused {
                        if out.mid.iter().all(|&v| v == MID_POISON) {
                            "poison_intact"
                        } else {
                            "poison_overwritten"
                        }
                    } else if out.mid == mid_ref {
                        "matches_reference_intermediate"
                    } else {
                        "intermediate_mismatch"
                    };
                    let case = FusionCase {
                        width: w,
                        height: h,
                        level,
                        mode: if fused { "fused_1_dispatch" } else { "serial_2_dispatch" },
                        win: out.win.clone(),
                        exact: diff.exact(),
                        diff,
                        mid_state,
                        ref_invariant,
                        elapsed_us: out.elapsed_us,
                    };
                    println!(
                        "    {:<19} {:>3}x{:<3} L{:<3} exact={:<5} mismatched_px={:<3} \
                         rgba_max={:?} submits={} dispatches={} downloads={} mid={} {:>7} us",
                        case.mode,
                        case.width,
                        case.height,
                        case.level,
                        case.exact,
                        case.diff.mismatch_pixels,
                        case.diff.max_abs,
                        case.win.submits,
                        case.win.dispatches,
                        case.win.downloads,
                        case.mid_state,
                        case.elapsed_us
                    );
                    if let Some((i, got, want)) = case.diff.first_mismatch {
                        println!("        first mismatch at px {i}: got 0x{got:08x} want 0x{want:08x}");
                    }
                    run.fusion.push(case);
                }
            }
        }

        for &(m, k, n) in MATMUL_SHAPES {
            let case = MatMulCase::new(m, k, n);
            let (got, win, elapsed_us) = matmul_run(&mut gpu, p_matmul, &case)?;
            let want_f32 = case.reference_f32();
            let want_fma = case.reference_fma_f32();
            let want_f64 = case.reference_f64();
            let r = MatMulResult {
                m,
                k,
                n,
                win,
                max_abs_f32: max_abs_f32(&got, &want_f32),
                max_abs_fma: max_abs_f32(&got, &want_fma),
                max_abs_f64: max_abs_f32_f64(&got, &want_f64),
                exact_vs_f32_ref: got == want_f32,
                elapsed_us,
            };
            println!(
                "    matmul_bias {}x{}x{} max_abs(f32 ref)={:.3e} max_abs(fma ref)={:.3e} \
                 max_abs(f64 ref)={:.3e} exact_vs_f32_ref={} submits={} dispatches={} downloads={} {:>7} us",
                r.m,
                r.k,
                r.n,
                r.max_abs_f32,
                r.max_abs_fma,
                r.max_abs_f64,
                r.exact_vs_f32_ref,
                r.win.submits,
                r.win.dispatches,
                r.win.downloads,
                r.elapsed_us
            );
            run.matmul.push(r);
        }

        for p in [p_fused, p_op_a, p_op_b, p_matmul] {
            gpu.destroy_pipeline(p);
        }
        run.status = "ok";
        Ok(())
    })();

    if let Err(e) = attempt {
        run.status = "error";
        run.error = Some(e.clone());
        println!("    DEVICE ERROR: {e}");
    }
    run
}

// ---------------------------------------------------------------------------
// JSON output (hand-rolled: no serde in this crate on purpose)
// ---------------------------------------------------------------------------

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o
}

fn device_json(run: &DeviceRun) -> String {
    let i = &run.info;
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"vendor_id\": \"{}\",\n", i.vendor_hex()));
    s.push_str(&format!("  \"device_id\": \"{}\",\n", i.device_hex()));
    s.push_str(&format!("  \"vendor_id_dec\": {},\n", i.vendor_id));
    s.push_str(&format!("  \"device_id_dec\": {},\n", i.device_id));
    s.push_str(&format!("  \"name\": \"{}\",\n", esc(&i.name)));
    s.push_str(&format!("  \"device_type\": \"{}\",\n", i.type_str()));
    s.push_str(&format!("  \"api_version\": \"{}\",\n", i.api_str()));
    s.push_str(&format!("  \"driver_version_raw\": {},\n", i.driver_version));
    s.push_str(&format!(
        "  \"driver_version\": \"{}\",\n",
        esc(&i.driver_version_human)
    ));
    s.push_str(&format!("  \"backend\": \"vulkan\",\n"));
    s.push_str(&format!("  \"compute_queue_family\": {},\n", run.queue_family));
    s.push_str(&format!(
        "  \"limits\": {{\"maxComputeWorkGroupInvocations\": {}, \"maxComputeWorkGroupSize\": [{}, {}, {}], \"maxStorageBufferRange\": {}, \"maxPushConstantsSize\": {}}},\n",
        i.max_compute_invocations,
        i.max_compute_workgroup[0],
        i.max_compute_workgroup[1],
        i.max_compute_workgroup[2],
        i.max_storage_buffer_range,
        i.max_push_constants
    ));
    s.push_str(&format!(
        "  \"features\": {{\"shaderFloat16\": {}, \"shaderFloat64\": {}, \"shaderInt8\": {}}},\n",
        i.shader_float16, i.shader_float64, i.shader_int8
    ));
    s.push_str(&format!("  \"status\": \"{}\",\n", run.status));
    s.push_str(&format!(
        "  \"error\": {},\n",
        match &run.error {
            Some(e) => format!("\"{}\"", esc(e)),
            None => "null".to_string(),
        }
    ));

    s.push_str("  \"fusion_cases\": [\n");
    for (idx, c) in run.fusion.iter().enumerate() {
        s.push_str(&format!(
            "    {{\"width\": {}, \"height\": {}, \"level\": {}, \"mode\": \"{}\", \"submits\": {}, \"dispatches\": {}, \"downloads\": {}, \"upload_bytes\": {}, \"download_bytes\": {}, \"exact\": {}, \"mismatched_pixels\": {}, \"max_abs_per_channel_rgba\": [{}, {}, {}, {}], \"intermediate\": \"{}\", \"cpu_fused_equals_cpu_serial\": {}, \"elapsed_us\": {}}}{}\n",
            c.width,
            c.height,
            c.level,
            c.mode,
            c.win.submits,
            c.win.dispatches,
            c.win.downloads,
            c.win.upload_bytes,
            c.win.download_bytes,
            c.exact,
            c.diff.mismatch_pixels,
            c.diff.max_abs[0],
            c.diff.max_abs[1],
            c.diff.max_abs[2],
            c.diff.max_abs[3],
            c.mid_state,
            c.ref_invariant,
            c.elapsed_us,
            if idx + 1 == run.fusion.len() { "" } else { "," }
        ));
    }
    s.push_str("  ],\n");

    s.push_str("  \"matmul_cases\": [\n");
    for (idx, m) in run.matmul.iter().enumerate() {
        s.push_str(&format!(
            "    {{\"m\": {}, \"k\": {}, \"n\": {}, \"submits\": {}, \"dispatches\": {}, \"downloads\": {}, \"max_abs_vs_f32_ref\": {:.9}, \"max_abs_vs_fma_ref\": {:.9}, \"max_abs_vs_f64_ref\": {:.9}, \"exact_vs_f32_ref\": {}, \"contract_lt_1e-4\": {}, \"elapsed_us\": {}}}{}\n",
            m.m,
            m.k,
            m.n,
            m.win.submits,
            m.win.dispatches,
            m.win.downloads,
            m.max_abs_f32,
            m.max_abs_fma,
            m.max_abs_f64,
            m.exact_vs_f32_ref,
            m.max_abs_f32 < 1e-4,
            m.elapsed_us,
            if idx + 1 == run.matmul.len() { "" } else { "," }
        ));
    }
    s.push_str("  ]\n}\n");
    s
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

fn main() {
    let out_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("logs"));
    let _ = fs::create_dir_all(&out_dir);
    let tag = std::env::args().nth(2).unwrap_or_else(|| "run".to_string());

    println!("=== aurora-vulkan-wgsl-probe: WGSL compute kernels on a Vulkan queue ===");
    println!("crate      : {} (standalone fixture, own workspace)", env!("CARGO_PKG_NAME"));
    println!("scope      : two integer RGBA8 stages fused into 1 dispatch + one small FP32 matmul");
    println!("scope note : self-checking fixture; not a production backend and not a benchmark");
    println!("out dir    : {}", out_dir.display());

    let fusion = match shader::compile("fusion.wgsl", FUSION_WGSL) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FATAL shader: {e}");
            std::process::exit(2);
        }
    };
    let matmul = match shader::compile("matmul.wgsl", MATMUL_WGSL) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FATAL shader: {e}");
            std::process::exit(2);
        }
    };
    println!(
        "\nWGSL -> SPIR-V : fusion.wgsl {} B, entry points {:?}, storage bindings {}",
        fusion.bytes, fusion.entry_points, fusion.storage_bindings
    );
    println!(
        "WGSL -> SPIR-V : matmul.wgsl {} B, entry points {:?}, storage bindings {}",
        matmul.bytes, matmul.entry_points, matmul.storage_bindings
    );

    let ctx = match VkCtx::new() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FATAL vulkan: {e}");
            let partial = format!(
                "{{\"status\": \"vulkan_unavailable\", \"error\": \"{}\"}}\n",
                esc(&e)
            );
            let _ = fs::write(out_dir.join(format!("partial-{tag}.json")), partial);
            std::process::exit(3);
        }
    };

    let enumerated = match vk::enumerate(&ctx) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("FATAL enumerate: {e}");
            std::process::exit(3);
        }
    };

    println!("\n=== Vulkan device inventory (only Vulkan is enumerated by this process) ===");
    for (_, info) in &enumerated {
        println!(
            "  [{}] {}:{} {} | type={} | api {} | driver {} | software_adapter={}",
            info.index,
            info.vendor_hex(),
            info.device_hex(),
            info.name,
            info.type_str(),
            info.api_str(),
            info.driver_version_human,
            info.is_software
        );
    }

    let mut runs: Vec<DeviceRun> = Vec::new();
    let mut unsupported: Vec<Unsupported> = Vec::new();
    let mut rejected: Vec<(String, String, &'static str)> = Vec::new();

    let only_inventory = std::env::var("C_PROBE_ONLY_INVENTORY")
        .map(|v| v == "1")
        .unwrap_or(false);
    // Default: measure every non-software device this machine exposes, so the
    // probe carries no vendor or hardware-model assumption. C_PROBE_VENDORS
    // narrows it to reproduce a specific hardware matrix; a vendor listed there
    // but absent becomes a recorded failure instead of a silent skip.
    let requested_vendors: Option<Vec<u32>> = std::env::var("C_PROBE_VENDORS").ok().map(|raw| {
        let mut ids = Vec::new();
        for part in raw.split(',') {
            let t = part.trim();
            if t.is_empty() {
                continue;
            }
            let hex = t.trim_start_matches("0x").trim_start_matches("0X");
            match u32::from_str_radix(hex, 16) {
                Ok(id) => ids.push(id),
                Err(_) => println!("  ignoring unparsable C_PROBE_VENDORS entry {t:?}"),
            }
        }
        ids
    });
    if only_inventory {
        println!("  C_PROBE_ONLY_INVENTORY=1: enumeration only, no measurement");
    }
    if let Some(list) = &requested_vendors {
        println!(
            "  C_PROBE_VENDORS: measurement restricted to {:?}",
            list.iter().map(|v| format!("0x{v:04x}")).collect::<Vec<_>>()
        );
    }

    let mut targets: Vec<(ash::vk::PhysicalDevice, DeviceInfo)> = Vec::new();
    for (pd, info) in &enumerated {
        if info.is_software {
            rejected.push((
                info.vendor_hex(),
                info.name.clone(),
                "software adapter (deviceType == CPU): enumerated, never measured",
            ));
            continue;
        }
        match &requested_vendors {
            Some(list) if !list.contains(&info.vendor_id) => rejected.push((
                info.vendor_hex(),
                info.name.clone(),
                "vendor not listed in C_PROBE_VENDORS",
            )),
            _ => targets.push((*pd, info.clone())),
        }
    }

    if let Some(list) = &requested_vendors {
        for vendor in list {
            if targets.iter().any(|(_, info)| info.vendor_id == *vendor) {
                continue;
            }
            let reason =
                format!("no non-software Vulkan device with vendor_id 0x{vendor:04x} was enumerated");
            println!("\n=== requested vendor 0x{vendor:04x}: UNSUPPORTED — {reason} ===");
            unsupported.push(Unsupported {
                vendor: *vendor,
                label: format!("requested 0x{vendor:04x}"),
                reason,
            });
        }
    }
    if targets.is_empty() {
        println!("\n=== no measurement target: no matching non-software Vulkan device ===");
    }

    for (pd, info) in &targets {
        if only_inventory {
            println!(
                "\n=== {}:{} {} present, measurement skipped ===",
                info.vendor_hex(),
                info.device_hex(),
                info.name
            );
            continue;
        }
        println!(
            "\n=== measuring {}:{} {} [{}] ===",
            info.vendor_hex(),
            info.device_hex(),
            info.name,
            info.type_str()
        );
        runs.push(run_device(&ctx, *pd, info.clone(), &fusion.words, &matmul.words));
    }

    // --- summary -----------------------------------------------------------
    let mut fusion_total = 0usize;
    let mut fusion_exact = 0usize;
    let mut fused_dispatches = 0u64;
    let mut serial_dispatches = 0u64;
    let mut fused_cases = 0usize;
    let mut serial_cases = 0usize;
    let mut matmul_total = 0usize;
    let mut matmul_ok = 0usize;
    let mut pixels_total = 0u64;
    let mut poison_ok = 0usize;
    let mut fused_cases_n = 0usize;
    let mut ref_invariant_ok = 0usize;

    for r in &runs {
        for c in &r.fusion {
            fusion_total += 1;
            if c.exact {
                fusion_exact += 1;
            }
            if c.ref_invariant {
                ref_invariant_ok += 1;
            }
            pixels_total += (c.width as u64) * (c.height as u64);
            if c.mode == "fused_1_dispatch" {
                fused_cases += 1;
                fused_cases_n += 1;
                fused_dispatches += c.win.dispatches;
                if c.mid_state == "poison_intact" {
                    poison_ok += 1;
                }
            } else {
                serial_cases += 1;
                serial_dispatches += c.win.dispatches;
            }
        }
        for m in &r.matmul {
            matmul_total += 1;
            if m.max_abs_f32 < 1e-4 {
                matmul_ok += 1;
            }
        }
    }
    let _ = (fused_cases, serial_cases);

    println!("\n=== summary ===");
    println!(
        "  devices measured (real hardware, Vulkan): {}",
        runs.iter().filter(|r| r.status == "ok").count()
    );
    for u in &unsupported {
        println!("  requested vendor 0x{:04x} unsupported: {}", u.vendor, u.reason);
    }
    for (v, name, why) in &rejected {
        println!("  rejected {v} {name}: {why}");
    }
    println!(
        "  fusion cases: {fusion_exact}/{fusion_total} byte-exact vs CPU serial reference \
         ({} pixels compared per channel R,G,B,A)",
        pixels_total
    );
    println!(
        "  fused dispatch accounting: {fused_dispatches} dispatches over {fused_cases_n} fused cases \
         (should equal the case count); serial: {serial_dispatches} over {} cases (2x)",
        fusion_total - fused_cases_n
    );
    println!(
        "  fused intermediate buffer untouched (poison intact): {poison_ok}/{fused_cases_n}"
    );
    println!("  CPU fused == CPU serial reference (both baselines agree): {ref_invariant_ok}/{fusion_total}");
    println!("  matmul cases inside 1e-4: {matmul_ok}/{matmul_total}");

    // An enumeration-only run measured nothing on purpose, so it is not a failure.
    let any_failure = (!only_inventory && runs.is_empty())
        || runs.iter().any(|r| r.status != "ok")
        || fusion_exact != fusion_total
        || matmul_ok != matmul_total
        || poison_ok != fused_cases_n
        || ref_invariant_ok != fusion_total
        || !unsupported.is_empty();

    // --- write artifacts ---------------------------------------------------
    write_json(&out_dir, &tag, &ctx, &enumerated, &runs, &unsupported, &rejected, any_failure);

    println!("\n  verdict: {}", if any_failure { "FAIL / PARTIAL (see above)" } else { "PASS" });
    println!("  logs written under: {}", out_dir.display());
    if any_failure {
        std::process::exit(1);
    }
}

#[allow(clippy::too_many_arguments)]
fn write_json(
    out_dir: &Path,
    tag: &str,
    ctx: &VkCtx,
    enumerated: &[(ash::vk::PhysicalDevice, DeviceInfo)],
    runs: &[DeviceRun],
    unsupported: &[Unsupported],
    rejected: &[(String, String, &'static str)],
    failed: bool,
) {
    for run in runs {
        let path = out_dir.join(format!(
            "device-{}-{}.json",
            run.info.vendor_hex().trim_start_matches("0x"),
            run.info.device_hex().trim_start_matches("0x")
        ));
        let _ = fs::write(&path, device_json(run));
        println!("  wrote {}", path.display());
    }

    let mut inv = String::from("{\n  \"backend\": \"vulkan\",\n  \"devices\": [\n");
    for (idx, (_, info)) in enumerated.iter().enumerate() {
        inv.push_str(&format!(
            "    {{\"index\": {}, \"vendor_id\": \"{}\", \"device_id\": \"{}\", \"name\": \"{}\", \"device_type\": \"{}\", \"api_version\": \"{}\", \"driver_version\": \"{}\", \"software_adapter\": {}}}{}\n",
            info.index,
            info.vendor_hex(),
            info.device_hex(),
            esc(&info.name),
            info.type_str(),
            info.api_str(),
            esc(&info.driver_version_human),
            info.is_software,
            if idx + 1 == enumerated.len() { "" } else { "," }
        ));
    }
    inv.push_str("  ]\n}\n");
    let _ = fs::write(out_dir.join("inventory.json"), inv);

    let mut s = String::new();
    s.push_str("{\n");
    s.push_str(&format!("  \"tag\": \"{}\",\n", esc(tag)));
    s.push_str(&format!(
        "  \"vulkan_loader_api\": \"{}\",\n",
        ctx.loader_api_str()
    ));
    s.push_str(&format!("  \"failed\": {},\n", failed));
    s.push_str("  \"sizes\": [");
    for (i, (w, h)) in SIZES.iter().enumerate() {
        s.push_str(&format!("[{w}, {h}]{}", if i + 1 == SIZES.len() { "" } else { ", " }));
    }
    s.push_str("],\n");
    s.push_str(&format!("  \"levels\": {:?},\n", LEVELS));
    s.push_str("  \"limitations\": [\n");
    let limitations = [
        "host-visible coherent memory for every buffer (map/unmap instead of staging copies); this is a fixture, not the residency strategy of a shipping backend",
        "correctness rests on the CPU-reference comparison, not on validation-layer output: no layer is required and none is enabled unless the caller installs one",
        "elapsed_us is submit+wait+readback wall clock for one case on a warm device; it is not a benchmark and does not compare across APIs or targets",
        "one dispatch per workgroup grid, no tiling: buffer sizes here are kilobytes, nothing about this shape extrapolates to full-resolution frames",
    ];
    for (i, l) in limitations.iter().enumerate() {
        s.push_str(&format!(
            "    \"{}\"{}\n",
            esc(l),
            if i + 1 == limitations.len() { "" } else { "," }
        ));
    }
    s.push_str("  ],\n");
    s.push_str("  \"unsupported\": [");
    for (i, u) in unsupported.iter().enumerate() {
        s.push_str(&format!(
            "{{\"vendor_id\": \"0x{:04x}\", \"label\": \"{}\", \"reason\": \"{}\"}}{}",
            u.vendor,
            esc(&u.label),
            esc(&u.reason),
            if i + 1 == unsupported.len() { "" } else { ", " }
        ));
    }
    s.push_str("],\n");
    s.push_str("  \"rejected\": [");
    for (i, (v, name, why)) in rejected.iter().enumerate() {
        s.push_str(&format!(
            "{{\"vendor_id\": \"{v}\", \"name\": \"{}\", \"reason\": \"{}\"}}{}",
            esc(name),
            esc(why),
            if i + 1 == rejected.len() { "" } else { ", " }
        ));
    }
    s.push_str("],\n");
    s.push_str("  \"devices\": [\n");
    for (idx, run) in runs.iter().enumerate() {
        s.push_str(&format!(
            "    {{\"vendor_id\": \"{}\", \"device_id\": \"{}\", \"name\": \"{}\", \"status\": \"{}\", \"error\": {}, \"fusion_cases\": {}, \"matmul_cases\": {}}}{}\n",
            run.info.vendor_hex(),
            run.info.device_hex(),
            esc(&run.info.name),
            run.status,
            match &run.error {
                Some(e) => format!("\"{}\"", esc(e)),
                None => "null".to_string(),
            },
            run.fusion.len(),
            run.matmul.len(),
            if idx + 1 == runs.len() { "" } else { "," }
        ));
    }
    s.push_str("  ]\n}\n");
    let p = out_dir.join("summary.json");
    let _ = fs::write(&p, s);
    println!("  wrote {}", p.display());
}
