//! Opt-in diagnostics and a real-window slider replay for performance checks.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use gpui_kit::{AnyWindowHandle, App, Entity};
use lumen_engine::RenderQuality;

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
    *ENABLED.get_or_init(|| std::env::var_os("LUMEN_PERF_REPLAY").is_some())
}

pub(crate) fn record(mut data: serde_json::Value) {
    if enabled() {
        static START: OnceLock<Instant> = OnceLock::new();
        data["t_ms"] = (START.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0).into();
        eprintln!("LUMEN_PERF {data}");
    }
}

pub(crate) fn replay(state: Entity<AppState>, window: AnyWindowHandle, cx: &mut App) {
    if !enabled() {
        return;
    }
    cx.spawn(async move |cx| {
        state.update(cx, |s, cx| {
            s.tool_tab = ToolTab::Exposure;
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
