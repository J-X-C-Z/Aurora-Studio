//! Actions whose interface is built ahead of the engine (`crate::planned`):
//! importing Capture One styles, merging HDR and panoramas, cancelling an
//! export. Each runs off the UI thread and reports what happened, including
//! "not connected yet" while the engine lacks the call.

use std::path::PathBuf;

use gpui_kit::{AppContext as _, Context};
use lumen_engine::{ImageId, JobStatus};

use super::{AppState, Edit};
use crate::i18n::t;
use crate::planned::{self, HdrOptions, PanoramaOptions};

/// Which merge the dialog runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MergeKind {
    Hdr,
    Panorama,
}

impl MergeKind {
    pub fn label(self) -> &'static str {
        match self {
            Self::Hdr => t("Merge to HDR", "合并为 HDR"),
            Self::Panorama => t("Merge to Panorama", "合并为全景"),
        }
    }
}

impl AppState {
    /// Reads Capture One `.costyle` / `.copreset` files into the styles list.
    pub fn import_styles(&mut self, paths: Vec<PathBuf>, cx: &mut Context<Self>) {
        if paths.is_empty() {
            return;
        }
        let engine = self.engine.clone();
        cx.spawn(async move |this, cx| {
            let results = cx.background_spawn(async move { paths.iter().map(|p| planned::import_style(engine.as_ref(), p)).collect::<Vec<_>>() }).await;
            this.update(cx, |this, cx| {
                let mut added = 0;
                let mut first_err = None;
                for r in results {
                    match r {
                        Ok(style) => {
                            this.styles.retain(|s| s.name != style.name);
                            this.styles.push(style);
                            added += 1;
                        }
                        Err(err) => {
                            first_err.get_or_insert(err.to_string());
                        }
                    }
                }
                let msg = match first_err {
                    Some(err) if added == 0 => format!("{}: {err}", t("Style import failed", "风格导入失败")),
                    Some(err) => format!("{} {added} · {err}", t("Imported", "已导入")),
                    None => format!("{} {added} {}", t("Imported", "已导入"), t("styles", "个风格")),
                };
                this.notify(msg, cx);
            })
            .ok();
        })
        .detach();
    }

    /// Merges the selected images into one new image, cataloged next to them.
    pub fn merge_selection(&mut self, kind: MergeKind, hdr: HdrOptions, pano: PanoramaOptions, cx: &mut Context<Self>) {
        let ids: Vec<ImageId> = self.images.iter().map(|i| i.id).filter(|id| self.selection.contains(id)).collect();
        if ids.len() < 2 {
            self.notify(t("Select at least two images to merge", "请至少选择两张图像进行合并"), cx);
            return;
        }
        let engine = self.engine.clone();
        self.notify(format!("{} · {} {}", kind.label(), ids.len(), t("images…", "张…")), cx);
        cx.spawn(async move |this, cx| {
            let result = cx
                .background_spawn(async move {
                    match kind {
                        MergeKind::Hdr => planned::merge_hdr(engine.as_ref(), &ids, &hdr),
                        MergeKind::Panorama => planned::merge_panorama(engine.as_ref(), &ids, &pano),
                    }
                })
                .await;
            this.update(cx, |this, cx| match result {
                Ok(info) => {
                    let id = info.id;
                    this.edits.insert(id, Edit::default());
                    this.images.push(info);
                    this.queue_thumbs(vec![id], cx);
                    this.select(id, false, false, cx);
                    this.notify(format!("{} · {}", kind.label(), t("done", "完成")), cx);
                }
                Err(err) => this.notify(format!("{}: {err}", kind.label()), cx),
            })
            .ok();
        })
        .detach();
    }

    pub fn cancel_job(&mut self, job: u64, cx: &mut Context<Self>) {
        match planned::cancel_job(self.engine.as_ref(), job) {
            Ok(()) => {
                if let Some(j) = self.jobs.iter_mut().find(|j| j.id == job) {
                    j.status = JobStatus::Failed { message: t("Cancelled", "已取消").into() };
                }
                cx.notify();
            }
            Err(err) => self.notify(err.to_string(), cx),
        }
    }
}
