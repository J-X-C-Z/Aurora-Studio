//! Engine analyses run on the selection: closed-eye check, auto rotation and
//! Match Look. Each runs in the background, applies what it found as ordinary
//! undoable edits, and reports a one-line summary, including what failed and why.

use lumen_engine::{AdjustmentSet, EyeReport, ImageId};
use gpui_kit::{AppContext as _, Context};

use super::AppState;
use crate::i18n::t;

/// `auto_rotate`'s answer as clockwise quarter turns. The engine returns degrees
/// in a `u16`, preserving all four orientations.
fn quarter_turns(degrees: u16) -> Option<u8> {
    match degrees {
        0 => Some(0),
        90 => Some(1),
        180 => Some(2),
        270 => Some(3),
        _ => None,
    }
}

/// The part of a Match Look result that moves the target: white balance,
/// exposure, saturation and RGB levels. Everything else the target keeps.
fn overlay_look(dst: &mut AdjustmentSet, look: &AdjustmentSet) {
    dst.white_balance = look.white_balance.clone();
    dst.exposure.exposure = look.exposure.exposure;
    dst.exposure.saturation = look.exposure.saturation;
    dst.levels.rgb = look.levels.rgb.clone();
}

impl AppState {
    /// One image's change with its own undo step, for results that differ per image.
    pub(crate) fn edit_image(&mut self, id: ImageId, label: impl Into<gpui_kit::SharedString>, f: impl FnOnce(&mut AdjustmentSet), cx: &mut Context<Self>) {
        let edit = self.edits.entry(id).or_default();
        let before = edit.adj.clone();
        f(&mut edit.adj);
        if before != edit.adj {
            edit.undo.push((label.into(), before));
            edit.redo.clear();
            self.after_commit(id, cx);
        }
    }

    fn selected_ids(&self) -> Vec<ImageId> {
        let mut ids: Vec<ImageId> = self.images.iter().map(|i| i.id).filter(|id| self.selection.contains(id)).collect();
        if ids.is_empty() {
            ids.extend(self.current);
        }
        ids
    }

    fn start_analysis(&mut self, what: &str, count: usize, cx: &mut Context<Self>) -> bool {
        if self.analysing {
            self.notify(t("Another analysis is still running", "另一项分析仍在进行"), cx);
            return false;
        }
        if count == 0 {
            return false;
        }
        self.analysing = true;
        self.notify(format!("{what} · {count} {}", t("images…", "张…")), cx);
        true
    }

    /// Capture One's closed-eye check on the selection. Results stay on the
    /// thumbnails for this session; nothing is changed or rejected.
    pub fn check_closed_eyes(&mut self, cx: &mut Context<Self>) {
        let ids = self.selected_ids();
        if !self.start_analysis(t("Checking closed eyes", "正在检查闭眼"), ids.len(), cx) {
            return;
        }
        self.run_eye_check(ids, true, cx);
    }

    /// `verbose`: report even when every eye is open (a manual check); a capture
    /// only speaks up when it finds something.
    pub(crate) fn run_eye_check(&mut self, ids: Vec<ImageId>, verbose: bool, cx: &mut Context<Self>) {
        let engine = self.engine.clone();
        cx.spawn(async move |this, cx| {
            let results: Vec<(ImageId, anyhow::Result<EyeReport>)> =
                cx.background_spawn(async move { ids.into_iter().map(|id| (id, engine.eye_report(id))).collect() }).await;
            this.update(cx, |this, cx| {
                if verbose {
                    this.analysing = false;
                }
                let (mut closed, mut faces, mut first_err) = (0, 0, None);
                for (id, r) in results {
                    match r {
                        Ok(report) => {
                            faces += (report.evaluated > 0) as usize;
                            closed += (report.closed > 0) as usize;
                            this.eyes.insert(id, report);
                        }
                        Err(err) => {
                            first_err.get_or_insert(err.to_string());
                        }
                    }
                }
                let msg = match first_err {
                    Some(err) if faces == 0 => Some(format!("{}: {err}", t("Closed-eye check unavailable", "闭眼检查不可用"))),
                    _ if closed > 0 => Some(format!("{closed} {}", t("image(s) with closed eyes, marked on the thumbnails", "张有闭眼，已在缩略图上标出"))),
                    _ if verbose && faces == 0 => Some(t("No eyes found to check", "未找到可检查的眼睛").to_string()),
                    _ if verbose => Some(t("No closed eyes found", "未发现闭眼").to_string()),
                    _ => None,
                };
                if let Some(msg) = msg {
                    this.notify(msg, cx);
                }
                cx.notify();
            })
            .ok();
        })
        .detach();
    }

    /// Asks the engine which way each selected image is up and turns the ones
    /// that need it, one undo step per image.
    pub fn auto_rotate_selection(&mut self, cx: &mut Context<Self>) {
        let ids = self.selected_ids();
        if !self.start_analysis(t("Detecting orientation", "正在检测方向"), ids.len(), cx) {
            return;
        }
        let engine = self.engine.clone();
        cx.spawn(async move |this, cx| {
            let results: Vec<(ImageId, anyhow::Result<u16>)> =
                cx.background_spawn(async move { ids.into_iter().map(|id| (id, engine.auto_rotate(id))).collect() }).await;
            this.update(cx, |this, cx| {
                this.analysing = false;
                let (mut turned, mut upright, mut first_err) = (0, 0, None);
                for (id, r) in results {
                    match r.map(quarter_turns) {
                        Ok(Some(0)) => upright += 1,
                        Ok(Some(q)) => {
                            this.edit_image(id, t("Auto rotate", "自动旋转"), |a| a.geometry.orientation = (a.geometry.orientation + q) % 4, cx);
                            turned += 1;
                        }
                        Ok(None) => {
                            first_err.get_or_insert(t("unexpected angle from the engine", "引擎返回了意外的角度").to_string());
                        }
                        Err(err) => {
                            first_err.get_or_insert(err.to_string());
                        }
                    }
                }
                let mut msg = format!("{} {turned} · {} {upright}", t("Rotated", "已旋转"), t("already upright", "无需旋转"));
                if let Some(err) = first_err {
                    msg = if turned + upright == 0 { format!("{}: {err}", t("Auto rotation unavailable", "自动旋转不可用")) } else { format!("{msg} · {err}") };
                }
                this.preview.before_dirty = true;
                this.request_render(super::RenderQuality::Final, cx);
                this.notify(msg, cx);
            })
            .ok();
        })
        .detach();
    }

    /// Match Look: the pinned compare variant (or else the current image) is the
    /// reference; every other selected image moves toward it.
    pub fn match_look_selection(&mut self, cx: &mut Context<Self>) {
        let Some(reference) = self.view.compare_variant.filter(|id| self.info(*id).is_some()).or(self.current) else { return };
        let targets: Vec<ImageId> = self.selected_ids().into_iter().filter(|id| *id != reference).collect();
        if targets.is_empty() {
            self.notify(t("Select the images to match, with the reference as the current image or pinned compare variant", "请选择要匹配的图像，并把参考图设为当前图像或对比变体"), cx);
            return;
        }
        if !self.start_analysis(t("Matching look", "正在匹配风格"), targets.len(), cx) {
            return;
        }
        let engine = self.engine.clone();
        cx.spawn(async move |this, cx| {
            let results: Vec<(ImageId, anyhow::Result<lumen_engine::MatchLook>)> =
                cx.background_spawn(async move { targets.into_iter().map(|id| (id, engine.match_look(reference, id))).collect() }).await;
            this.update(cx, |this, cx| {
                this.analysing = false;
                let (mut matched, mut residual, mut first_err) = (0, 0.0f32, None);
                for (id, r) in results {
                    match r {
                        Ok(look) => {
                            this.edit_image(id, t("Match look", "匹配风格"), |a| overlay_look(a, &look.adjustments), cx);
                            residual = residual.max(look.report.residual);
                            matched += 1;
                        }
                        Err(err) => {
                            first_err.get_or_insert(err.to_string());
                        }
                    }
                }
                let mut msg = match matched {
                    0 => t("Match look failed", "风格匹配失败").to_string(),
                    n => format!("{} {n} · {} {:.0}%", t("Matched", "已匹配"), t("largest remaining difference", "最大剩余差异"), residual * 100.0),
                };
                if let Some(err) = first_err {
                    msg = format!("{msg}: {err}");
                }
                this.preview.before_dirty = true;
                this.request_render(super::RenderQuality::Final, cx);
                this.notify(msg, cx);
            })
            .ok();
        })
        .detach();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_angles_map_to_quarter_turns() {
        assert_eq!([0, 90, 180, 270].map(quarter_turns), [Some(0), Some(1), Some(2), Some(3)]);
        assert_eq!(quarter_turns(45), None);
    }

    #[test]
    fn match_look_only_moves_its_tools() {
        let mut dst = AdjustmentSet::default();
        dst.exposure.contrast = 12.0;
        let mut look = AdjustmentSet::default();
        look.exposure.exposure = 0.7;
        look.exposure.contrast = -40.0;
        look.white_balance.kelvin = 6100.0;
        overlay_look(&mut dst, &look);
        assert_eq!(dst.exposure.exposure, 0.7);
        assert_eq!(dst.exposure.contrast, 12.0);
        assert_eq!(dst.white_balance.kelvin, 6100.0);
    }
}
