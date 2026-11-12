//! Slideshow: the visible images one after another, full screen, advancing on
//! a timer. The images are the regular previews, rendered fitted to the window.

use std::time::Duration;

use gpui_kit::{Context, Window};

use super::{AppState, RenderQuality, Slideshow};
use crate::i18n::t;

static RUNS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl AppState {
    pub fn start_slideshow(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.slideshow.is_some() || self.visible_images().is_empty() {
            return;
        }
        // One image at a time: a multiple selection would show side by side.
        let first = self.current.unwrap_or(self.visible_images()[0].id);
        self.select(first, false, false, cx);
        let was_fullscreen = window.is_fullscreen();
        if !was_fullscreen {
            window.toggle_fullscreen();
        }
        let run = RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.slideshow = Some(Slideshow { playing: true, interval: self.settings.slideshow_interval.max(1.0), run, was_fullscreen });
        self.set_zoom(None, None, cx);
        self.request_render(RenderQuality::Final, cx);
        self.schedule_slide(run, cx);
        cx.notify();
    }

    fn schedule_slide(&mut self, run: u64, cx: &mut Context<Self>) {
        let Some(interval) = self.slideshow.as_ref().map(|s| s.interval) else { return };
        cx.spawn(async move |this, cx| {
            cx.background_executor().timer(Duration::from_secs_f32(interval)).await;
            let _ = this.update(cx, |s, cx| {
                let Some(show) = s.slideshow.as_ref().filter(|show| show.run == run) else { return };
                if show.playing {
                    s.slide(1, cx);
                }
                s.schedule_slide(run, cx);
            });
        })
        .detach();
    }

    /// Next (`1`) or previous (`-1`) slide, wrapping around.
    pub fn slide(&mut self, delta: isize, cx: &mut Context<Self>) {
        let ids: Vec<_> = self.visible_images().iter().map(|i| i.id).collect();
        if ids.is_empty() {
            return;
        }
        let at = self.current.and_then(|c| ids.iter().position(|i| *i == c)).unwrap_or(0) as isize;
        let next = ids[(at + delta).rem_euclid(ids.len() as isize) as usize];
        self.select(next, false, false, cx);
    }

    pub fn toggle_slideshow_play(&mut self, cx: &mut Context<Self>) {
        if let Some(show) = self.slideshow.as_mut() {
            show.playing = !show.playing;
            let msg = if show.playing { t("Playing", "播放") } else { t("Paused", "已暂停") };
            self.notify(msg, cx);
        }
    }

    pub fn set_slideshow_interval(&mut self, secs: f32, cx: &mut Context<Self>) {
        self.settings.slideshow_interval = secs;
        self.settings.save();
        // Restart the timer so the new pace applies from now.
        if let Some(show) = self.slideshow.as_mut() {
            show.interval = secs;
            show.run = RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let run = show.run;
            self.schedule_slide(run, cx);
        }
        cx.notify();
    }

    pub fn stop_slideshow(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(show) = self.slideshow.take() else { return };
        if !show.was_fullscreen && window.is_fullscreen() {
            window.toggle_fullscreen();
        }
        self.request_render(RenderQuality::Final, cx);
        cx.notify();
    }
}
