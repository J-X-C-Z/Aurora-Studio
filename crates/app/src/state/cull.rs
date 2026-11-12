//! Culling: one image at a time, full screen, rated from the keyboard as in
//! Lightroom's loupe culling and Capture One's Cull view. Ratings are the
//! catalog's own (`ImageInfo::rating`, -1 rejected); nothing else changes.

use gpui_kit::{Context, Window};
use lumen_engine::ImageId;

use super::{AppState, RenderQuality};
use crate::i18n::t;

#[derive(Clone, Debug)]
pub struct Cull {
    /// After a rating or reject, move on to the next image.
    pub auto_advance: bool,
    /// Step only between images that have no rating yet.
    pub unrated_only: bool,
    pub(crate) was_fullscreen: bool,
}

impl AppState {
    pub fn start_cull(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        if self.cull.is_some() || self.slideshow.is_some() || self.visible_images().is_empty() {
            return;
        }
        let first = self.current.unwrap_or(self.visible_images()[0].id);
        self.select(first, false, false, cx);
        let was_fullscreen = window.is_fullscreen();
        if !was_fullscreen {
            window.toggle_fullscreen();
        }
        self.cull = Some(Cull { auto_advance: self.settings.cull_auto_advance, unrated_only: false, was_fullscreen });
        self.set_zoom(None, None, cx);
        self.request_render(RenderQuality::Final, cx);
        cx.notify();
    }

    pub fn stop_cull(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let Some(cull) = self.cull.take() else { return };
        if !cull.was_fullscreen && window.is_fullscreen() {
            window.toggle_fullscreen();
        }
        self.set_zoom(None, None, cx);
        self.request_render(RenderQuality::Final, cx);
        cx.notify();
    }

    /// A slideshow or culling has the screen: the image alone, no tool overlays.
    pub fn presenting(&self) -> bool {
        self.slideshow.is_some() || self.cull.is_some()
    }

    /// The images culling steps through, in browser order.
    pub fn cull_images(&self) -> Vec<(ImageId, i8)> {
        self.visible_images().iter().map(|i| (i.id, i.rating)).collect()
    }

    /// Next (`1`) or previous (`-1`) image, skipping rated ones when asked.
    /// Stops at the ends instead of wrapping, so the last image is not passed by accident.
    pub fn cull_step(&mut self, delta: isize, cx: &mut Context<Self>) {
        let Some(cull) = self.cull.clone() else { return };
        let ids = self.cull_images();
        let Some(at) = self.current.and_then(|c| ids.iter().position(|(i, _)| *i == c)) else { return };
        let mut i = at as isize + delta;
        while (0..ids.len() as isize).contains(&i) {
            let (id, rating) = ids[i as usize];
            if !cull.unrated_only || rating == 0 {
                self.select(id, false, false, cx);
                if self.zoom.scale.is_some() {
                    self.set_zoom(None, None, cx);
                }
                return;
            }
            i += delta;
        }
        let msg = match (cull.unrated_only, delta > 0) {
            (true, _) => t("No more unrated images this way", "这个方向没有未评级的图像了"),
            (false, true) => t("Last image", "已是最后一张"),
            (false, false) => t("First image", "已是第一张"),
        };
        self.notify(msg, cx);
    }

    /// Rates (0 ..= 5) or rejects (-1) the image on screen; a second press of the
    /// same key clears it. Moves on when auto-advance is on.
    pub fn cull_rate(&mut self, rating: i8, cx: &mut Context<Self>) {
        let Some(id) = self.current else { return };
        let old = self.info(id).map(|i| i.rating).unwrap_or(0);
        let new = if old == rating && rating != 0 { 0 } else { rating };
        self.selection = [id].into_iter().collect();
        self.set_rating(new, cx);
        if self.cull.as_ref().is_some_and(|c| c.auto_advance) && new != 0 {
            self.cull_step(1, cx);
        }
    }

    pub fn toggle_cull_auto_advance(&mut self, cx: &mut Context<Self>) {
        let Some(cull) = self.cull.as_mut() else { return };
        cull.auto_advance = !cull.auto_advance;
        self.settings.cull_auto_advance = cull.auto_advance;
        self.settings.save();
        let msg = if cull.auto_advance { t("Auto-advance on", "自动前进：开") } else { t("Auto-advance off", "自动前进：关") };
        self.notify(msg, cx);
    }

    pub fn toggle_cull_unrated(&mut self, cx: &mut Context<Self>) {
        let Some(cull) = self.cull.as_mut() else { return };
        cull.unrated_only = !cull.unrated_only;
        let on = cull.unrated_only;
        let msg = if on { t("Stepping through unrated images only", "只在未评级的图像间切换") } else { t("Stepping through all images", "在所有图像间切换") };
        self.notify(msg, cx);
    }

    /// Z: fit, or 100 % at the center to check focus.
    pub fn toggle_cull_zoom(&mut self, cx: &mut Context<Self>) {
        if self.zoom.scale.is_some() {
            self.set_zoom(None, None, cx);
        } else {
            let sf = self.scale_factor;
            self.set_zoom(Some(sf), Some((0.5, 0.5)), cx);
        }
    }
}
