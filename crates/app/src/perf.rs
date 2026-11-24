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

/// Current process working set, or `None` where the platform is unsupported.
#[cfg(target_os = "windows")]
pub(crate) fn working_set_bytes() -> Option<u64> {
    memory::working_set_bytes()
}

#[cfg(target_os = "windows")]
pub(crate) fn peak_working_set_bytes() -> Option<u64> {
    memory::peak_working_set_bytes()
}

#[cfg(not(target_os = "windows"))]
pub(crate) fn working_set_bytes() -> Option<u64> {
    None
}

#[cfg(not(target_os = "windows"))]
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
