//! Synthetic microbenchmark for the macOS Metal 2-kernel chain.
//!
//! **Scope:** this measures `aurora_engine::skin_whiten_metal` — a library
//! harness (3x3 box blur -> masked 3D-LUT whiten), not the product preview
//! pipeline. It says nothing about UI frame rate, RAW decoding, portrait
//! staging or Windows. Every number printed is measured on this machine.
//!
//! Usage (Release, Apple Silicon):
//!
//! ```text
//! cargo run -p aurora-engine --release --locked --target aarch64-apple-darwin \
//!     --example metal_chain_bench
//! ```
//!
//! Optional environment:
//! * `AURORA_METAL_BENCH_ITERS`  — warm iterations (default 40, minimum 30).
//! * `AURORA_METAL_BENCH_JSON`   — append the JSON lines to this file.
//!
//! Output: one JSON line per case on stdout (plus one `environment` and one
//! `process_warmup` line), then a Markdown table. On non-macOS the binary
//! prints why it is not applicable and exits 0.
//!
//! Timing definitions used by every case:
//! * **cold first call** — the first `run_skin_whiten` at that size/degree
//!   after the process-level warmup. It may include the first buffer
//!   allocation for a new size class, but not MSL compilation (that is
//!   reported once as `process_warmup`).
//! * **warm sample** — wall time of the full library call: uploads, encode,
//!   commit, the single `waitUntilCompleted` and the single CPU readback.
//! * **percentiles** — nearest-rank on the sorted warm samples:
//!   `index = ceil(p/100 * n) - 1`, so with `n = 40`, P99 is the maximum.

// The case JSON object has many stable fields; serde_json's `json!` expands
// recursively and needs more than the default 128.
#![recursion_limit = "256"]

#[cfg(target_os = "macos")]
mod macos {
    use std::hint::black_box;
    use std::io::Write;
    use std::time::Instant;

    use aurora_engine::metal_runtime::MetalCounters;
    use aurora_engine::skin_whiten_metal::{
        SkinWhitenInput, cpu_reference, run_skin_whiten, synthetic_test_lut,
    };

    const SIZES: [(u32, u32); 3] = [(37, 23), (1920, 1080), (3840, 2160)];
    const DEGREES: [f32; 2] = [0.5, 1.0];
    const DEFAULT_WARM_ITERS: usize = 40;
    const MIN_WARM_ITERS: usize = 30;

    pub fn run() -> Result<(), String> {
        let runtime = aurora_engine::metal_runtime::MetalRuntime::shared()
            .map_err(|error| format!("Metal runtime unavailable: {error}"))?;
        let probe = runtime.probe_info();

        let warm_iters = std::env::var("AURORA_METAL_BENCH_ITERS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(DEFAULT_WARM_ITERS)
            .max(MIN_WARM_ITERS);

        let mut sink = JsonSink::open(std::env::var_os("AURORA_METAL_BENCH_JSON"))?;

        let environment = serde_json::json!({
            "event": "environment",
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "device_name": probe.name,
            "unified_memory": probe.unified,
            "recommended_working_set_bytes": probe.recommended_working_set,
            "max_buffer_len_bytes": probe.max_buffer_len,
            "warm_iterations": warm_iters,
            "sizes": SIZES,
            "degrees": DEGREES,
        });
        sink.emit(&environment)?;
        println!(
            "metal_chain_bench: device=\"{}\" unified={} warm_iterations={}",
            probe.name, probe.unified, warm_iters
        );

        let lut = synthetic_test_lut();

        // Process warmup: the first Evercall compiles both MSL libraries and
        // creates the pipelines. Report it separately so per-case "cold" does
        // not silently include one-time shader compilation.
        {
            let warm = make_case_input(8, 8);
            let input = SkinWhitenInput {
                width: warm.width,
                height: warm.height,
                bgra: &warm.bgra,
                mask: &warm.mask,
                lut: &lut,
                degree: 0.5,
            };
            let before = runtime.counters();
            let started = Instant::now();
            let result = run_skin_whiten(runtime, &input)
                .map_err(|error| format!("process warmup failed: {error}"))?;
            let ms = started.elapsed().as_secs_f64() * 1000.0;
            black_box(&result.pixels);
            let after = runtime.counters();
            let line = serde_json::json!({
                "event": "process_warmup",
                "ms": ms,
                "pipelines_compiled_delta": after.pipelines_compiled - before.pipelines_compiled,
                "buffer_allocations_delta": after.buffer_allocations - before.buffer_allocations,
            });
            sink.emit(&line)?;
            println!(
                "process warmup (first call, includes MSL compilation): {ms:.3} ms, pipelines_compiled={}",
                after.pipelines_compiled - before.pipelines_compiled
            );
        }

        let mut rows: Vec<CaseRow> = Vec::new();
        for (width, height) in SIZES {
            let case = make_case_input(width, height);
            for degree in DEGREES {
                let row = run_case(runtime, &case, &lut, degree, warm_iters, &mut sink)?;
                rows.push(row);
            }
        }

        print_markdown(&rows, &probe.name, warm_iters);
        Ok(())
    }

    struct CaseInput {
        width: u32,
        height: u32,
        bgra: Vec<u8>,
        mask: Vec<u8>,
    }

    /// Deterministic synthetic input. The same bytes feed GPU and CPU, so the
    /// comparison tests the chain and not an asset. No product/vendor data.
    fn make_case_input(width: u32, height: u32) -> CaseInput {
        let pixels = width as usize * height as usize;
        let mut bgra = vec![0u8; pixels * 4];
        let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ (pixels as u64);
        for pixel in bgra.chunks_exact_mut(4) {
            // xorshift64; deterministic across runs and platforms.
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            pixel[0] = (state >> 16) as u8; // B
            pixel[1] = (state >> 8) as u8; // G
            pixel[2] = state as u8; // R
            pixel[3] = 255; // A
        }
        let mut mask = vec![0u8; pixels];
        for (index, value) in mask.iter_mut().enumerate() {
            *value = ((index.wrapping_mul(97) + 13) % 256) as u8;
        }
        CaseInput {
            width,
            height,
            bgra,
            mask,
        }
    }

    #[derive(Clone, Copy, Default)]
    struct Percentiles {
        min: f64,
        p50: f64,
        p95: f64,
        p99: f64,
        max: f64,
        mean: f64,
    }

    fn summarize(samples_ms: &[f64]) -> Percentiles {
        let mut sorted = samples_ms.to_vec();
        sorted.sort_by(f64::total_cmp);
        let n = sorted.len();
        let rank = |p: f64| -> f64 {
            let index = ((p / 100.0) * n as f64).ceil() as usize;
            sorted[index.saturating_sub(1).min(n - 1)]
        };
        let mean = sorted.iter().sum::<f64>() / n as f64;
        Percentiles {
            min: sorted[0],
            p50: rank(50.0),
            p95: rank(95.0),
            p99: rank(99.0),
            max: sorted[n - 1],
            mean,
        }
    }

    struct CaseRow {
        width: u32,
        height: u32,
        degree: f32,
        gpu_cold_ms: f64,
        gpu: Percentiles,
        cpu_cold_ms: f64,
        cpu: Percentiles,
        max_diff: [u8; 4],
        counters_delta: CountersDelta,
        runs_in_case: u64,
        counters_after: MetalCounters,
        footprint_before: Option<u64>,
        footprint_after: Option<u64>,
    }

    struct CountersDelta {
        command_buffers: u64,
        encoders: u64,
        dispatches: u64,
        waits: u64,
        readbacks: u64,
        buffer_allocations: u64,
        buffer_reuses: u64,
    }

    impl CountersDelta {
        fn between(before: MetalCounters, after: MetalCounters) -> Self {
            Self {
                command_buffers: after.command_buffers - before.command_buffers,
                encoders: after.compute_encoders - before.compute_encoders,
                dispatches: after.dispatches - before.dispatches,
                waits: after.waits - before.waits,
                readbacks: after.readbacks - before.readbacks,
                buffer_allocations: after.buffer_allocations - before.buffer_allocations,
                buffer_reuses: after.buffer_reuses - before.buffer_reuses,
            }
        }

        fn per_run(&self, runs: u64) -> [u64; 5] {
            [
                self.command_buffers / runs,
                self.encoders / runs,
                self.dispatches / runs,
                self.waits / runs,
                self.readbacks / runs,
            ]
        }
    }

    fn run_case(
        runtime: &'static aurora_engine::metal_runtime::MetalRuntime,
        case: &CaseInput,
        lut: &[f32],
        degree: f32,
        warm_iters: usize,
        sink: &mut JsonSink,
    ) -> Result<CaseRow, String> {
        let input = SkinWhitenInput {
            width: case.width,
            height: case.height,
            bgra: &case.bgra,
            mask: &case.mask,
            lut,
            degree,
        };
        let runs = 1 + warm_iters as u64;

        let footprint_before = process_footprint_bytes();
        let counters_before = runtime.counters();

        // Cold GPU call (buffer allocation for this size, pipelines warm).
        let started = Instant::now();
        let cold_gpu = run_skin_whiten(runtime, &input)
            .map_err(|error| format!("GPU cold run failed at {}x{}: {error}", case.width, case.height))?;
        let gpu_cold_ms = started.elapsed().as_secs_f64() * 1000.0;
        let cold_bytes = cold_gpu.pixels;
        black_box(&cold_bytes);

        // Cold CPU reference (no compile/allocation warmup concept; first call).
        let started = Instant::now();
        let cpu_bytes = cpu_reference(&input)
            .map_err(|error| format!("CPU cold run failed at {}x{}: {error}", case.width, case.height))?;
        let cpu_cold_ms = started.elapsed().as_secs_f64() * 1000.0;
        black_box(&cpu_bytes);

        let max_diff = max_channel_diff(&cold_bytes, &cpu_bytes);

        // Warm GPU loop.
        let mut gpu_samples = Vec::with_capacity(warm_iters);
        let mut gpu_output_stable = true;
        for _ in 0..warm_iters {
            let started = Instant::now();
            let result = run_skin_whiten(runtime, &input)
                .map_err(|error| format!("GPU warm run failed at {}x{}: {error}", case.width, case.height))?;
            gpu_samples.push(started.elapsed().as_secs_f64() * 1000.0);
            gpu_output_stable &= result.pixels == cold_bytes;
            black_box(&result.pixels);
        }

        // Warm CPU loop (same count; the reference has no compile step).
        let mut cpu_samples = Vec::with_capacity(warm_iters);
        for _ in 0..warm_iters {
            let started = Instant::now();
            let bytes = cpu_reference(&input)
                .map_err(|error| format!("CPU warm run failed at {}x{}: {error}", case.width, case.height))?;
            cpu_samples.push(started.elapsed().as_secs_f64() * 1000.0);
            black_box(&bytes);
        }

        let footprint_after = process_footprint_bytes();
        let counters_after = runtime.counters();
        let counters_delta = CountersDelta::between(counters_before, counters_after);
        let gpu = summarize(&gpu_samples);
        let cpu = summarize(&cpu_samples);
        let per_run = counters_delta.per_run(runs);
        let cpu_over_gpu = cpu.p50 / gpu.p50;

        let line = serde_json::json!({
            "event": "case",
            "width": case.width,
            "height": case.height,
            "pixels": case.width as u64 * case.height as u64,
            "degree": degree,
            "warm_iterations": warm_iters,
            "runs_in_case": runs,
            "gpu_cold_first_call_ms": gpu_cold_ms,
            "gpu_warm_min_ms": gpu.min,
            "gpu_warm_mean_ms": gpu.mean,
            "gpu_warm_p50_ms": gpu.p50,
            "gpu_warm_p95_ms": gpu.p95,
            "gpu_warm_p99_ms": gpu.p99,
            "gpu_warm_max_ms": gpu.max,
            "cpu_cold_first_call_ms": cpu_cold_ms,
            "cpu_warm_min_ms": cpu.min,
            "cpu_warm_mean_ms": cpu.mean,
            "cpu_warm_p50_ms": cpu.p50,
            "cpu_warm_p95_ms": cpu.p95,
            "cpu_warm_p99_ms": cpu.p99,
            "cpu_warm_max_ms": cpu.max,
            "cpu_over_gpu_p50_ratio": cpu_over_gpu,
            "max_abs_diff_b": max_diff[0],
            "max_abs_diff_g": max_diff[1],
            "max_abs_diff_r": max_diff[2],
            "max_abs_diff_a": max_diff[3],
            "gpu_output_stable": gpu_output_stable,
            "counters_delta_command_buffers": counters_delta.command_buffers,
            "counters_delta_compute_encoders": counters_delta.encoders,
            "counters_delta_dispatches": counters_delta.dispatches,
            "counters_delta_waits": counters_delta.waits,
            "counters_delta_readbacks": counters_delta.readbacks,
            "counters_delta_buffer_allocations": counters_delta.buffer_allocations,
            "counters_delta_buffer_reuses": counters_delta.buffer_reuses,
            "per_run_command_buffers": per_run[0],
            "per_run_compute_encoders": per_run[1],
            "per_run_dispatches": per_run[2],
            "per_run_waits": per_run[3],
            "per_run_readbacks": per_run[4],
            "live_bytes_after": counters_after.live_bytes,
            "pooled_bytes_after": counters_after.pooled_bytes,
            "peak_live_bytes_cumulative": counters_after.peak_live_bytes,
            "process_footprint_before_bytes": footprint_before,
            "process_footprint_after_bytes": footprint_after,
            "process_footprint_delta_bytes": match (footprint_before, footprint_after) {
                (Some(before), Some(after)) => Some(after as i64 - before as i64),
                _ => None,
            },
        });
        sink.emit(&line)?;

        Ok(CaseRow {
            width: case.width,
            height: case.height,
            degree,
            gpu_cold_ms,
            gpu,
            cpu_cold_ms,
            cpu,
            max_diff,
            counters_delta,
            runs_in_case: runs,
            counters_after,
            footprint_before,
            footprint_after,
        })
    }

    /// Max absolute byte difference per packed BGRA channel (`0=B,1=G,2=R,3=A`).
    fn max_channel_diff(gpu: &[u8], cpu: &[u8]) -> [u8; 4] {
        let mut max = [0u8; 4];
        let compared = gpu.len().min(cpu.len());
        for index in 0..compared {
            let diff = gpu[index].abs_diff(cpu[index]);
            let channel = index % 4;
            max[channel] = max[channel].max(diff);
        }
        // A length mismatch is itself a correctness failure; saturate.
        if gpu.len() != cpu.len() {
            max = [u8::MAX; 4];
        }
        max
    }

    struct JsonSink {
        file: Option<std::fs::File>,
    }

    impl JsonSink {
        fn open(path: Option<std::ffi::OsString>) -> Result<Self, String> {
            let Some(path) = path else {
                return Ok(Self { file: None });
            };
            let file = std::fs::File::create(std::path::Path::new(&path))
                .map_err(|error| format!("cannot create JSON output {path:?}: {error}"))?;
            Ok(Self { file: Some(file) })
        }

        fn emit(&mut self, value: &serde_json::Value) -> Result<(), String> {
            let line = serde_json::to_string(value).map_err(|error| error.to_string())?;
            println!("{line}");
            if let Some(file) = self.file.as_mut() {
                writeln!(file, "{line}")
                    .map_err(|error| format!("cannot write JSON output: {error}"))?;
            }
            Ok(())
        }
    }

    fn print_markdown(rows: &[CaseRow], device: &str, warm_iters: usize) {
        println!();
        println!("## Metal 2-kernel chain microbenchmark (synthetic, library-level)");
        println!();
        println!(
            "Device: {device}. Warm iterations per case: {warm_iters}. \
             Percentiles are nearest-rank on the sorted samples (P99 = max at n={warm_iters})."
        );
        println!();
        println!(
            "| size | degree | GPU cold ms | GPU P50 | GPU P95 | GPU P99 | CPU cold ms | CPU P50 | CPU P95 | CPU P99 | CPU/GPU P50 | max diff B/G/R/A | CB/enc/disp/wait/read per run | allocs | live B | pooled B | footprint before→after MiB |"
        );
        println!(
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        );
        for row in rows {
            let per_run = row.counters_delta.per_run(row.runs_in_case);
            let footprint = match (row.footprint_before, row.footprint_after) {
                (Some(before), Some(after)) => format!(
                    "{:.1}→{:.1}",
                    before as f64 / (1024.0 * 1024.0),
                    after as f64 / (1024.0 * 1024.0)
                ),
                _ => "n/a".to_string(),
            };
            println!(
                "| {}x{} | {:.1} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.2}x | {}/{}/{}/{} | {}/{}/{}/{}/{} | {} | {} | {} | {} |",
                row.width,
                row.height,
                row.degree,
                row.gpu_cold_ms,
                row.gpu.p50,
                row.gpu.p95,
                row.gpu.p99,
                row.cpu_cold_ms,
                row.cpu.p50,
                row.cpu.p95,
                row.cpu.p99,
                row.cpu.p50 / row.gpu.p50,
                row.max_diff[0],
                row.max_diff[1],
                row.max_diff[2],
                row.max_diff[3],
                per_run[0],
                per_run[1],
                per_run[2],
                per_run[3],
                per_run[4],
                row.counters_delta.buffer_allocations,
                row.counters_after.live_bytes,
                row.counters_after.pooled_bytes,
                footprint,
            );
        }
        println!();
        println!(
            "Note: `cold` = first call at that size/degree in an already-warmed process \
             (MSL compilation is reported once as `process_warmup`). \
             `allocs` is the count of new MTLBuffer objects during the case; \
             pooled bytes are retained by the runtime's buffer pool. \
             These are library-chain microbenchmarks, not product preview timings."
        );
    }

    // --- Mach process footprint (same fields as crates/app/src/perf.rs) -----

    const KERN_SUCCESS: i32 = 0;
    const TASK_VM_INFO: u32 = 22;

    /// Prefix of `struct task_vm_info` through `ledger_phys_footprint_peak`.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    #[allow(dead_code)]
    struct TaskVmInfoPrefix {
        virtual_size: u64,
        region_count: i32,
        page_size: i32,
        resident_size: u64,
        resident_size_peak: u64,
        device: u64,
        device_peak: u64,
        internal: u64,
        internal_peak: u64,
        external: u64,
        external_peak: u64,
        reusable: u64,
        reusable_peak: u64,
        purgeable_volatile_pmap: u64,
        purgeable_volatile_resident: u64,
        purgeable_volatile_virtual: u64,
        compressed: u64,
        compressed_peak: u64,
        compressed_lifetime: u64,
        phys_footprint: u64,
        min_address: u64,
        max_address: u64,
        ledger_phys_footprint_peak: i64,
    }
    const _: () = assert!(std::mem::size_of::<TaskVmInfoPrefix>() == 176);
    const TASK_VM_INFO_PREFIX_COUNT: u32 =
        (std::mem::size_of::<TaskVmInfoPrefix>() / std::mem::size_of::<u32>()) as u32;
    const UNITS_THROUGH_PHYS_FOOTPRINT: u32 = 38;

    unsafe extern "C" {
        fn mach_task_self() -> u32;
        fn task_info(
            target_task: u32,
            flavor: u32,
            task_info_out: *mut i32,
            task_info_out_cnt: *mut u32,
        ) -> i32;
    }

    /// Kernel-reported physical footprint, the metric Activity Monitor shows.
    fn process_footprint_bytes() -> Option<u64> {
        let mut info = TaskVmInfoPrefix::default();
        let mut count = TASK_VM_INFO_PREFIX_COUNT;
        // SAFETY: `info` is a repr(C) prefix of `task_vm_info`, `count` is its
        // size in natural_t units, and the kernel writes at most `count` units.
        let result = unsafe {
            task_info(
                mach_task_self(),
                TASK_VM_INFO,
                (&mut info as *mut TaskVmInfoPrefix).cast::<i32>(),
                &mut count,
            )
        };
        (result == KERN_SUCCESS && count >= UNITS_THROUGH_PHYS_FOOTPRINT && info.phys_footprint > 0)
            .then_some(info.phys_footprint)
    }
}

#[cfg(target_os = "macos")]
fn main() {
    if let Err(error) = macos::run() {
        eprintln!("metal_chain_bench: {error}");
        std::process::exit(1);
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!(
        "metal_chain_bench: not applicable on {} — this benchmark measures the macOS \
         Metal 2-kernel chain and must run on an Apple Silicon Mac. Exiting 0.",
        std::env::consts::OS
    );
}
