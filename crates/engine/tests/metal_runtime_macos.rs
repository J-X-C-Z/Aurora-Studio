//! M2 Metal runtime evidence.
//!
//! On macOS these tests open the real system device, run the two-kernel
//! (box blur -> masked 3D-LUT whiten) chain in one command buffer, and compare
//! the readback against the portrait crate's CPU `masked_lut_blend`.
//!
//! On other platforms the test binary still compiles, but every Metal test is
//! `#[ignore]`d with an explicit "skipped" reason; nothing is reported as a
//! passing GPU run.

#[cfg(target_os = "macos")]
mod macos {
    use aurora_engine::metal_runtime::MetalRuntime;
    use aurora_engine::skin_whiten_metal::{
        SkinWhitenInput, cpu_reference, run_skin_whiten, synthetic_test_lut,
    };

    #[test]
    fn metal_probe_reports_a_real_device() {
        let runtime = MetalRuntime::shared().expect("Metal runtime must open on macOS");
        let probe = runtime.probe_info();
        eprintln!(
            "metal probe: name={:?} unified={} recommended_working_set={} max_buffer_len={} registry_id={}",
            probe.name, probe.unified, probe.recommended_working_set, probe.max_buffer_len, probe.registry_id
        );
        assert!(!probe.name.is_empty(), "device must report a name");
        assert!(probe.max_buffer_len >= 1 << 20, "max buffer length looks bogus");
        assert!(
            probe.recommended_working_set > 0,
            "recommended working set must be non-zero"
        );
        assert_ne!(probe.registry_id, 0, "registry id must identify the device");
    }

    #[test]
    fn metal_two_kernel_chain_matches_cpu_reference() {
        // Non-square, odd dimensions exercise bounds checks and row stride.
        let width = 37u32;
        let height = 23u32;
        let pixels = (width as usize) * (height as usize);

        let mut bgra = vec![0u8; pixels * 4];
        for y in 0..height {
            for x in 0..width {
                let index = ((y * width + x) as usize) * 4;
                bgra[index] = ((x * 5 + y * 3) % 256) as u8;
                bgra[index + 1] = ((x * 3 + y * 7 + 40) % 256) as u8;
                bgra[index + 2] = ((x * 11 + y * 2 + 90) % 256) as u8;
                bgra[index + 3] = ((x * 2 + y * 5 + 150) % 256) as u8;
            }
        }
        let mask: Vec<u8> = (0..pixels)
            .map(|pixel| ((pixel * 37 + 11) % 256) as u8)
            .collect();
        let lut = synthetic_test_lut();
        let degree = 0.75f32;
        let input = SkinWhitenInput {
            width,
            height,
            bgra: &bgra,
            mask: &mask,
            lut: &lut,
            degree,
        };

        let runtime = MetalRuntime::shared().expect("Metal runtime must open on macOS");
        let before = runtime.counters();
        let device = run_skin_whiten(runtime, &input).expect("two-kernel chain must run");
        let after = runtime.counters();
        let cpu = cpu_reference(&input).expect("CPU reference must run");

        eprintln!(
            "metal pipelines: blur tew={} max_total={} groups={:?} threads={:?}; whiten tew={} max_total={} groups={:?} threads={:?}",
            device.blur_limits.thread_execution_width,
            device.blur_limits.max_total_threads,
            device.blur_grid.groups,
            device.blur_grid.threads,
            device.whiten_limits.thread_execution_width,
            device.whiten_limits.max_total_threads,
            device.whiten_grid.groups,
            device.whiten_grid.threads,
        );
        assert!(
            device.blur_grid.threads_total()
                <= u64::from(device.blur_limits.max_total_threads)
        );
        assert!(
            device.whiten_grid.threads_total()
                <= u64::from(device.whiten_limits.max_total_threads)
        );

        assert_eq!(device.pixels.len(), cpu.len());
        assert!(device.generation > 0, "frame must carry a real generation");

        // Documented tolerance: the MSL is compiled with fast math disabled and
        // `#pragma clang fp contract(off)`, but trilinear LUT accumulation can
        // still differ by one quantization step. Alpha is copied from the
        // identically-computed integer blur and must match exactly.
        let mut max_per_channel = [0i32; 4];
        for (index, (&gpu, &cpu)) in device.pixels.iter().zip(cpu.iter()).enumerate() {
            let difference = i32::from(gpu) - i32::from(cpu);
            let slot = index % 4;
            max_per_channel[slot] = max_per_channel[slot].max(difference.abs());
        }
        eprintln!(
            "metal parity: {}x{} pixels={} degree={} per-channel max |gpu-cpu| = B:{} G:{} R:{} A:{}",
            width, height, pixels, degree,
            max_per_channel[0], max_per_channel[1], max_per_channel[2], max_per_channel[3]
        );
        assert!(
            max_per_channel[0] <= 1
                && max_per_channel[1] <= 1
                && max_per_channel[2] <= 1,
            "RGB differs by more than the documented 1 LSB tolerance: {max_per_channel:?}"
        );
        assert_eq!(
            max_per_channel[3], 0,
            "alpha must be bit-identical (integer blur + byte copy)"
        );

        // Batch contract: one committed command buffer, two encoders/dispatches,
        // one wait and one readback. `pipeline_limits` compiles the two
        // pipelines before dispatch, so the cache is populated exactly once.
        eprintln!(
            "metal counters delta: command_buffers={} encoders={} dispatches={} waits={} readbacks={} allocations={} reuses={} live_bytes_after={} pooled_bytes_after={}",
            after.command_buffers - before.command_buffers,
            after.compute_encoders - before.compute_encoders,
            after.dispatches - before.dispatches,
            after.waits - before.waits,
            after.readbacks - before.readbacks,
            after.buffer_allocations - before.buffer_allocations,
            after.buffer_reuses - before.buffer_reuses,
            after.live_bytes,
            after.pooled_bytes,
        );
        assert_eq!(after.command_buffers - before.command_buffers, 1);
        assert_eq!(after.compute_encoders - before.compute_encoders, 2);
        assert_eq!(after.dispatches - before.dispatches, 2);
        assert_eq!(after.waits - before.waits, 1);
        assert_eq!(after.readbacks - before.readbacks, 1);
        assert_eq!(after.pipelines_compiled - before.pipelines_compiled, 2);
        assert_eq!(after.buffer_allocations - before.buffer_allocations, 7);
        assert_eq!(after.buffer_reuses - before.buffer_reuses, 0);
        assert_eq!(after.live_bytes, 0, "read_into must return every buffer");
        assert!(after.pooled_bytes > 0, "pool must retain completed buffers");
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
#[ignore = "skipped: the Metal runtime only exists on target_os = macos"]
fn metal_runtime_tests_are_skipped_off_macos() {
    eprintln!(
        "SKIPPED: Metal runtime tests require target_os = macos; host is {}-{}",
        std::env::consts::OS,
        std::env::consts::ARCH
    );
}
