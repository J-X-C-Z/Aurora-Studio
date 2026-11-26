//! The `AURORA_GPU_BACKEND` switch has to hold for the startup warmup.
//!
//! This is an integration test on purpose: a unit test shares process state with
//! 2 300 other tests, and the claim here is about the *process* — "on this host,
//! after this switch, the CUDA domain never loaded a module". A fresh test binary
//! gives exactly one process in which the CUDA module-load counter starts at zero
//! and every phase is ordered by this test.
//!
//! Phases, in order:
//! 1. `AURORA_GPU_BACKEND=cpu` — no device warmer runs and the driver's module
//!    counter stays at its initial value, on a host that has a real NVIDIA GPU.
//! 2. `AURORA_GPU_BACKEND=opencl` — same, and the report says the OpenCL device
//!    is cold instead of pretending it was warmed.
//! 3. `AURORA_RAW_PREWARM=0` — the switch wins over an explicit `cuda` request.
//! 4. `AURORA_GPU_BACKEND=cuda` — the same host now really loads the embedded
//!    PTX modules, and a second call loads nothing.
//!
//! Without an NVIDIA device (CI) phase 4 reports a failure reason instead of
//! warming; the earlier phases still prove the switch, because "zero module
//! loads" is what they assert.

#![cfg(windows)]

use aurora_engine::aurora_raw::prewarm;

fn set_env(key: &str, value: Option<&str>) {
    // SAFETY: this test binary runs one test, so no other thread reads the
    // environment while it is being changed.
    unsafe {
        match value {
            Some(value) => std::env::set_var(key, value),
            None => std::env::remove_var(key),
        }
    }
}

fn cuda_module_loads() -> usize {
    aurora_portrait::cuda::driver_metrics().ptx_module_load_calls
}

#[test]
fn the_backend_switch_controls_every_cuda_module_load() {
    // A clean environment is part of the fixture: an inherited value would make
    // phase 4 meaningless.
    set_env("AURORA_RAW_PREWARM", None);
    set_env("AURORA_GPU_BACKEND", None);
    assert_eq!(cuda_module_loads(), 0, "the process must start with no CUDA module loaded");

    // Phase 1: the CPU request must not touch CUDA, and this host has a GPU, so
    // "the probe would have found one" is not an excuse.
    set_env("AURORA_GPU_BACKEND", Some("cpu"));
    let cpu = prewarm::prewarm_devices();
    assert_eq!(cpu.backend, Some(aurora_engine::aurora_raw::gpu_probe::BackendKind::Cpu), "{}", cpu.summary());
    assert!(cpu.devices.is_empty(), "{}", cpu.summary());
    assert!(!cpu.is_warm(), "{}", cpu.summary());
    assert_eq!(cpu.cuda_module_loads, 0, "{}", cpu.summary());
    assert_eq!(cuda_module_loads(), 0, "AURORA_GPU_BACKEND=cpu loaded a CUDA module");

    // Phase 2: every GPU backend this build has no warmer for is reported cold,
    // and none of them falls back to the CUDA warmer.
    for (spelling, expected) in [
        ("metal", aurora_engine::aurora_raw::gpu_probe::BackendKind::Metal),
        ("opencl", aurora_engine::aurora_raw::gpu_probe::BackendKind::OpenCl),
    ] {
        set_env("AURORA_GPU_BACKEND", Some(spelling));
        let report = prewarm::prewarm_devices();
        assert_eq!(report.backend, Some(expected), "{spelling}: {}", report.summary());
        assert!(report.devices.is_empty(), "{spelling}: {}", report.summary());
        assert!(!report.is_warm(), "{spelling}: a cold device is not warm: {}", report.summary());
        assert!(
            report
                .skipped
                .as_deref()
                .is_some_and(|reason| reason.contains("no warmer registered")),
            "{spelling}: {}",
            report.summary()
        );
        assert_eq!(report.cuda_module_loads, 0, "{spelling}: {}", report.summary());
        assert_eq!(
            cuda_module_loads(),
            0,
            "AURORA_GPU_BACKEND={spelling} loaded a CUDA module"
        );
    }

    // A real PSO can be reused only if both transient command lists share the live device.
    set_env("AURORA_GPU_BACKEND",Some("dx12"));
    let first=prewarm::prewarm_devices();
    assert_eq!(first.devices.len(),1,"{}",first.summary());
    assert_eq!(first.devices[0].name,"dx12-inference-psos");
    eprintln!("{}",first.summary());
    if first.is_warm() {
        let warm=prewarm::prewarm_devices();eprintln!("{}",warm.summary());
        assert_eq!(warm.devices[0].outcome,prewarm::DeviceOutcome::Warmed{loaded:0,cached:4});
    } else {
        assert!(matches!(&first.devices[0].outcome,prewarm::DeviceOutcome::Failed(s) if !s.is_empty()));
    }
    assert_eq!(cuda_module_loads(),0,"DX12 warmup invoked CUDA");

    // Phase 3: the switch is the last word, even for an explicit cuda request.
    set_env("AURORA_GPU_BACKEND", Some("cuda"));
    set_env("AURORA_RAW_PREWARM", Some("0"));
    let disabled = prewarm::prewarm_devices();
    assert_eq!(disabled.skipped.as_deref(), Some("AURORA_RAW_PREWARM=0"), "{}", disabled.summary());
    assert!(disabled.devices.is_empty(), "{}", disabled.summary());
    assert_eq!(cuda_module_loads(), 0, "AURORA_RAW_PREWARM=0 still loaded a CUDA module");

    // Phase 4: the same process, the same host, the switch removed.
    set_env("AURORA_RAW_PREWARM", None);
    let cuda = prewarm::prewarm_devices();
    assert_eq!(
        cuda.backend,
        Some(aurora_engine::aurora_raw::gpu_probe::BackendKind::Cuda),
        "an explicit request is dispatched, not pre-judged: {}",
        cuda.summary()
    );
    assert_eq!(cuda.devices.len(), 1, "{}", cuda.summary());
    assert_eq!(cuda.devices[0].name, "cuda-ptx", "{}", cuda.summary());
    eprintln!("{}", cuda.summary());
    if cuda.is_warm() {
        assert_eq!(
            cuda.cuda_module_loads,
            prewarm::ptx_images().len(),
            "the first CUDA warmup must load every planned PTX image once: {}",
            cuda.summary()
        );
        assert_eq!(
            cuda_module_loads(),
            prewarm::ptx_images().len(),
            "the process counter must agree with the report"
        );
        let second = prewarm::prewarm_devices();
        assert_eq!(second.cuda_module_loads, 0, "a warm second call loaded a module: {}", second.summary());
        assert!(second.is_warm(), "{}", second.summary());
    } else {
        // No NVIDIA device here: the failure must be reported, not hidden.
        eprintln!("no CUDA device on this host: {}", cuda.summary());
        assert_eq!(cuda.cuda_module_loads, 0, "{}", cuda.summary());
        assert_eq!(cuda_module_loads(), 0, "a failed CUDA warmup must not load a module");
    }

    // Phase 5: auto must follow the real probe. On this host that resolves CUDA
    // and hits the warm cache; on a host without NVIDIA it must not invent one.
    set_env("AURORA_GPU_BACKEND", None);
    let loads_before = cuda_module_loads();
    let auto = prewarm::prewarm_devices();
    assert_eq!(auto.request, "auto", "{}", auto.summary());
    assert!(auto.backend.is_some(), "auto always resolves a backend: {}", auto.summary());
    if auto.backend == Some(aurora_engine::aurora_raw::gpu_probe::BackendKind::Cuda) {
        assert!(auto.is_warm(), "{}", auto.summary());
        assert_eq!(auto.cuda_module_loads, 0, "auto hit the warm cache: {}", auto.summary());
    } else {
        assert_eq!(cuda_module_loads(), loads_before, "a non-CUDA auto resolution loaded a module");
    }
    assert_eq!(cuda_module_loads(), loads_before, "{}", auto.summary());
}
