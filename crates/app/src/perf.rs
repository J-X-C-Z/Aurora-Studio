//! Opt-in diagnostics and a real-window slider replay for performance checks.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use gpui_kit::{AnyWindowHandle, App, Entity};
use aurora_engine::RenderQuality;

use crate::state::{AppState, ToolTab};

#[derive(Clone, Copy)]
pub(crate) struct Stamp {
    pub id: u64,
    pub quality: RenderQuality,
    pub requested_at: Instant,
}

impl Stamp {
    pub fn new(id: u64, quality: RenderQuality, requested_at: Instant) -> Self {
        Self { id, quality, requested_at }
    }
}

pub(crate) fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("AURORA_PERF_REPLAY").is_some())
}

pub(crate) fn record(mut data: serde_json::Value) {
    if enabled() {
        static START: OnceLock<Instant> = OnceLock::new();
        data["t_ms"] = (START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0).into();
        eprintln!("AURORA_PERF {data}");
    }
}

/// One prepared preview frame, with the display-side stages split out.
///
/// Engine-internal stages (`source_lookup_ms` … `histogram_ms`) are not exposed
/// by `PreviewFrame` yet; they are recorded as `null` until S2/S3/S6 add the
/// per-branch timings. Cache hit/miss, source kind and the fast flag come from
/// the engine's own `engine_preview` event, matched by `id`.
pub(crate) struct PreparedTiming {
    pub id: u64,
    pub image_id: u64,
    pub quality: String,
    pub width: u32,
    pub height: u32,
    pub display_reused: bool,
    pub queue_ms: f64,
    pub engine_ms: f64,
    pub engine_wall_ms: f64,
    pub analysis_ms: f64,
    pub overlay_ms: f64,
    pub scope_ms: f64,
    pub display_copy_ms: f64,
    pub display_channel_convert_ms: f64,
    pub render_image_create_ms: f64,
    pub prepare_ms: f64,
    pub total_frame_ms: f64,
    pub ram_before_bytes: Option<u64>,
    pub ram_after_bytes: Option<u64>,
    pub peak_ram_bytes: Option<u64>,
}

pub(crate) fn record_prepared(t: PreparedTiming) {
    if !enabled() {
        return;
    }
    record(serde_json::json!({
        "event": "prepared",
        "id": t.id,
        "image_id": t.image_id,
        "quality": t.quality,
        "fast": false,
        "width": t.width,
        "height": t.height,
        "display_reused": t.display_reused,
        // Kept for existing parsers; `queue_wait_ms` is the same value.
        "queue_ms": t.queue_ms,
        "queue_wait_ms": t.queue_ms,
        "engine_ms": t.engine_ms,
        "engine_wall_ms": t.engine_wall_ms,
        "source_lookup_ms": null,
        "raw_decode_ms": null,
        "source_build_ms": null,
        "sampling_ms": null,
        "plan_build_ms": null,
        "develop_ms": null,
        "detail_ms": null,
        "gpu_upload_ms": null,
        "gpu_kernel_ms": null,
        "gpu_download_ms": null,
        "icc_ms": null,
        "histogram_ms": null,
        "backend": null,
        "fallback": null,
        "cache_hit": null,
        "overlay_ms": t.overlay_ms,
        "scope_ms": t.scope_ms,
        "analysis_ms": t.analysis_ms,
        "display_copy_ms": t.display_copy_ms,
        "display_channel_convert_ms": t.display_channel_convert_ms,
        "render_image_create_ms": t.render_image_create_ms,
        "prepare_ms": t.prepare_ms,
        "total_frame_ms": t.total_frame_ms,
        "ram_before_bytes": t.ram_before_bytes,
        "ram_after_bytes": t.ram_after_bytes,
        "peak_ram_bytes": t.peak_ram_bytes,
    }));
}

#[cfg(target_os = "windows")]
mod memory {
    use std::mem::size_of;

    #[repr(C)]
    #[derive(Default)]
    struct ProcessMemoryCounters {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn K32GetProcessMemoryInfo(
            process: *mut core::ffi::c_void,
            counters: *mut ProcessMemoryCounters,
            cb: u32,
        ) -> i32;
        fn GetCurrentProcess() -> *mut core::ffi::c_void;
    }

    fn counters() -> Option<ProcessMemoryCounters> {
        let mut counters = ProcessMemoryCounters {
            cb: size_of::<ProcessMemoryCounters>() as u32,
            ..ProcessMemoryCounters::default()
        };
        // SAFETY: the struct matches PROCESS_MEMORY_COUNTERS and `cb` is its size.
        let ok = unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
        (ok != 0).then_some(counters)
    }

    pub(super) fn working_set_bytes() -> Option<u64> {
        counters().map(|c| c.working_set_size as u64)
    }

    pub(super) fn peak_working_set_bytes() -> Option<u64> {
        counters().map(|c| c.peak_working_set_size as u64)
    }
}

#[cfg(target_os = "macos")]
mod memory {
    //! macOS process memory via Mach `task_info`.
    //!
    //! `working_set_bytes` reports the kernel's **physical footprint**
    //! (`task_vm_info::phys_footprint`), which is the figure Activity Monitor
    //! and `footprint(1)` show: dirty + compressed pages attributed to this
    //! task, not file-backed clean pages. If a kernel is too old to return the
    //! rev1 `phys_footprint` field we fall back to `resident_size` from
    //! `MACH_TASK_BASIC_INFO`.
    //!
    //! `peak_working_set_bytes` returns an **OS-provided high-water mark**, never
    //! a sampled/synthesized value: `task_vm_info::ledger_phys_footprint_peak`
    //! (rev3) when available, otherwise `resident_size_max` from
    //! `MACH_TASK_BASIC_INFO`. The two are different metrics (peak footprint vs
    //! peak resident), so a mixed fallback is reported as-is rather than
    //! pretending they are identical.
    //!
    //! The structs and constants are declared locally because `crates/app` does
    //! not depend on `libc`; field order and padding must match
    //! `<mach/task_info.h>` exactly. Compile-time `size_of` assertions guard the
    //! two layouts this module reads.

    use std::mem::size_of;

    /// `kern_return_t`; `KERN_SUCCESS` is 0.
    type KernReturn = i32;
    const KERN_SUCCESS: KernReturn = 0;
    /// `natural_t` / `mach_msg_type_number_t` unit used by `task_info` counts.
    type NaturalT = u32;

    /// `MACH_TASK_BASIC_INFO` flavor (always 64-bit basic info).
    const MACH_TASK_BASIC_INFO: NaturalT = 20;
    /// `TASK_VM_INFO` flavor.
    const TASK_VM_INFO: NaturalT = 22;

    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    #[allow(dead_code)] // Fields after resident_size_max exist only for ABI layout.
    struct TimeValue {
        seconds: i32,
        microseconds: i32,
    }

    /// `struct mach_task_basic_info` from `<mach/task_info.h>`.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    #[allow(dead_code)] // Timers/policy are ABI padding for this module's purpose.
    struct MachTaskBasicInfo {
        virtual_size: u64,
        resident_size: u64,
        resident_size_max: u64,
        user_time: TimeValue,
        system_time: TimeValue,
        policy: i32,
        suspend_count: i32,
    }
    const _: () = assert!(size_of::<MachTaskBasicInfo>() == 48);

    /// Prefix of `struct task_vm_info` through `ledger_phys_footprint_peak`
    /// (rev3). Later revisions append fields this module does not read, so the
    /// requested count stops here and works on every macOS the project targets.
    ///
    /// Offsets that matter: `phys_footprint` @144 B (rev1),
    /// `ledger_phys_footprint_peak` @168 B (rev3).
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    #[allow(dead_code)] // The full prefix is declared to keep the ABI offsets exact.
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
    const _: () = assert!(size_of::<TaskVmInfoPrefix>() == 176);

    /// `task_info` counts in `natural_t` units (4 bytes on all Apple targets).
    const MACH_TASK_BASIC_INFO_COUNT: u32 =
        (size_of::<MachTaskBasicInfo>() / size_of::<NaturalT>()) as u32;
    const TASK_VM_INFO_PREFIX_COUNT: u32 =
        (size_of::<TaskVmInfoPrefix>() / size_of::<NaturalT>()) as u32;
    /// Units needed before `phys_footprint` is valid.
    const UNITS_THROUGH_PHYS_FOOTPRINT: u32 = 38;
    /// Units needed before `ledger_phys_footprint_peak` is valid.
    const UNITS_THROUGH_LEDGER_PEAK: u32 = 44;

    // `mach_task_self()` and `task_info` live in libsystem_kernel, which every
    // macOS binary links by default; no `#[link]` attribute is needed.
    unsafe extern "C" {
        fn mach_task_self() -> u32;
        /// `task_info(task_name_t, task_flavor_t, task_info_t, *mut u32)`.
        /// `task_info_t` is `*mut integer_t` (i32), whatever the struct behind it.
        fn task_info(
            target_task: u32,
            flavor: NaturalT,
            task_info_out: *mut i32,
            task_info_out_cnt: *mut u32,
        ) -> KernReturn;
    }

    fn task_basic_info() -> Option<MachTaskBasicInfo> {
        let mut info = MachTaskBasicInfo::default();
        let mut count = MACH_TASK_BASIC_INFO_COUNT;
        // SAFETY: `info` is a `repr(C)` struct matching `mach_task_basic_info`,
        // `count` is its size in `natural_t` units, and the kernel writes at
        // most `count` units into it. Casting the struct pointer to
        // `*mut i32` matches `task_info_t`; `mach_task_self()` takes no input.
        let result = unsafe {
            task_info(
                mach_task_self(),
                MACH_TASK_BASIC_INFO,
                (&mut info as *mut MachTaskBasicInfo).cast::<i32>(),
                &mut count,
            )
        };
        (result == KERN_SUCCESS).then_some(info)
    }

    /// Returns the populated prefix and the unit count the kernel actually
    /// wrote, so callers can tell whether rev1/rev3 fields are valid.
    fn task_vm_info() -> Option<(TaskVmInfoPrefix, u32)> {
        let mut info = TaskVmInfoPrefix::default();
        let mut count = TASK_VM_INFO_PREFIX_COUNT;
        // SAFETY: same contract as `task_basic_info`; `info` is a `repr(C)`
        // prefix of `task_vm_info` and `count` is its size in natural_t units.
        let result = unsafe {
            task_info(
                mach_task_self(),
                TASK_VM_INFO,
                (&mut info as *mut TaskVmInfoPrefix).cast::<i32>(),
                &mut count,
            )
        };
        (result == KERN_SUCCESS).then_some((info, count))
    }

    pub(super) fn working_set_bytes() -> Option<u64> {
        if let Some((info, units)) = task_vm_info()
            && units >= UNITS_THROUGH_PHYS_FOOTPRINT
            && info.phys_footprint > 0
        {
            return Some(info.phys_footprint);
        }
        task_basic_info()
            .map(|info| info.resident_size)
            .filter(|bytes| *bytes > 0)
    }

    pub(super) fn peak_working_set_bytes() -> Option<u64> {
        if let Some((info, units)) = task_vm_info()
            && units >= UNITS_THROUGH_LEDGER_PEAK
            && info.ledger_phys_footprint_peak > 0
        {
            return Some(info.ledger_phys_footprint_peak as u64);
        }
        task_basic_info()
            .map(|info| info.resident_size_max)
            .filter(|bytes| *bytes > 0)
    }
}

/// Current process working set (Windows working set / macOS physical
/// footprint), or `None` where the platform is unsupported.
#[cfg(any(target_os = "windows", target_os = "macos"))]
pub(crate) fn working_set_bytes() -> Option<u64> {
    memory::working_set_bytes()
}

/// OS-reported peak process working set, or `None` when unavailable. This is
/// never a sampled or synthesized peak.
#[cfg(any(target_os = "windows", target_os = "macos"))]
pub(crate) fn peak_working_set_bytes() -> Option<u64> {
    memory::peak_working_set_bytes()
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
pub(crate) fn working_set_bytes() -> Option<u64> {
    None
}

#[cfg(not(any(target_os = "windows", target_os = "macos")))]
pub(crate) fn peak_working_set_bytes() -> Option<u64> {
    None
}

pub(crate) fn replay(state: Entity<AppState>, window: AnyWindowHandle, cx: &mut App) {
    if !enabled() {
        return;
    }
    cx.spawn(async move |cx| {
        state.update(cx, |s, cx| {
            s.tool_tab = ToolTab::Grading;
            cx.notify();
        });
        for _ in 0..300 {
            cx.background_executor().timer(Duration::from_millis(20)).await;
            let ready = state.read_with(cx, |s, _| s.preview.frame.is_some() && !s.preview.rendering && s.thumbs.len() == s.images.len());
            if ready {
                break;
            }
        }
        window.update(cx, |_, window, cx| {
            window.activate_window();
            record(serde_json::json!({"event": "window", "active": window.is_window_active(),
                "gpu": format!("{:?}", window.gpu_specs()), "power_mode": format!("{:?}", cx.power_mode())}));
        }).expect("replay window closed");
        record(serde_json::json!({"event": "drag_start"}));
        for step in 0..120 {
            cx.background_executor().timer(Duration::from_millis(8)).await;
            let value = 1.12 * (step as f32 * 0.12).sin();
            state.update(cx, |s, cx| {
                record(serde_json::json!({"event": "input", "step": step}));
                s.edit_live("Exposure", |a| a.exposure.exposure = value, cx);
            });
        }
        let released_at = Instant::now();
        state.update(cx, |s, cx| s.end_gesture(cx));
        record(serde_json::json!({"event": "commit", "elapsed_ms": released_at.elapsed().as_secs_f64() * 1000.0}));
        record(serde_json::json!({"event": "drag_end"}));
        for _ in 0..300 {
            cx.background_executor().timer(Duration::from_millis(20)).await;
            if state.read_with(cx, |s, _| !s.preview.rendering) {
                break;
            }
        }
        cx.background_executor().timer(Duration::from_millis(150)).await;
        state.read_with(cx, |s, _| record(serde_json::json!({
            "event": "replay_done", "exposure": s.adj().exposure.exposure,
            "undo_steps": s.current.and_then(|id| s.edits.get(&id)).map_or(0, |e| e.undo.len()),
            "rendering": s.preview.rendering,
            "expected_exposure": 1.12f32 * (119.0f32 * 0.12).sin(),
            "final_quality": s.preview.frame.as_ref().map(|f| format!("{:?}", f.stamp.quality)),
            "latest_frame": s.preview.frame.as_ref().is_some_and(|f| f.stamp.id == s.preview.generation),
        })));
        cx.update(|cx| cx.quit());
    }).detach();
}

#[cfg(all(test, target_os = "windows"))]
mod tests {
    #[test]
    fn process_memory_counters_are_readable() {
        let working = super::working_set_bytes().expect("working set");
        let peak = super::peak_working_set_bytes().expect("peak working set");
        assert!(working > 0);
        assert!(peak >= working);
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    #[test]
    fn mach_task_info_reports_a_footprint_and_peak() {
        let working = super::working_set_bytes().expect("phys_footprint");
        let peak = super::peak_working_set_bytes().expect("peak footprint");
        // A live test process always owns at least a few pages.
        assert!(working > 0, "phys_footprint must be positive, got {working}");
        assert!(peak > 0, "peak footprint must be positive, got {peak}");
        eprintln!("perf memory: working={working} B peak={peak} B");
    }
}
