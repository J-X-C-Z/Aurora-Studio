//! The Export window: export variants through a recipe, or copy originals with an
//! optional adjustments sidecar.

use std::path::PathBuf;

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Disableable as _, Icon, Sizable as _, WindowExt as _,
    button::{Button, ButtonVariants as _},
    h_flex, v_flex,
};
use gpui_kit::{
    App, AppContext as _, Context, Entity, IntoElement, ParentElement, PathPromptOptions, Render, SharedString, Styled,
    Window, div, prelude::FluentBuilder as _, px,
};
use lumen_engine::{ExportFormat, ImageId};

use crate::i18n::t;
use crate::state::AppState;
use crate::theme::pal;
use crate::widgets::{ParamSlider, Segmented, caption, row, select, switch_row, text_field};

pub struct ExportDialog {
    state: Entity<AppState>,
    originals: bool,
    recipe: usize,
    format: Option<ExportFormat>,
    quality: u8,
    dest: Option<PathBuf>,
    naming: String,
    by_date: bool,
    sidecar: bool,
    working: bool,
}

impl ExportDialog {
    fn pick_dest(&mut self, cx: &mut Context<Self>) {
        let rx = cx.prompt_for_paths(PathPromptOptions { files: false, directories: true, multiple: false, prompt: None });
        cx.spawn(async move |this, cx| {
            if let Ok(Ok(Some(paths))) = rx.await
                && let Some(dir) = paths.into_iter().next()
            {
                this.update(cx, |d, cx| {
                    d.dest = Some(dir);
                    cx.notify();
                })
                .ok();
            }
        })
        .detach();
    }

    fn run(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let state = self.state.clone();
        if !self.originals {
            let Some(mut recipe) = state.read(cx).recipes.get(self.recipe).cloned() else { return };
            if let Some(f) = self.format {
                recipe.format = f;
            }
            recipe.quality = self.quality;
            recipe.destination = self.dest.clone().or(recipe.destination);
            recipe.file_name_format = self.naming.clone();
            state.update(cx, |s, cx| s.process(vec![recipe], cx));
            window.close_dialog(cx);
            return;
        }
        let Some(dest) = self.dest.clone() else { return };
        let dest = if self.by_date { dest.join(crate::ui::import::today()) } else { dest };
        let sidecar = self.sidecar;
        let jobs: Vec<(ImageId, PathBuf, Option<String>)> = {
            let s = state.read(cx);
            s.selection
                .iter()
                .filter_map(|id| s.info(*id).map(|i| (*id, i.path.clone(), sidecar.then(|| serde_json::to_string_pretty(&s.edits.get(id).map(|e| e.adj.clone()).unwrap_or_default()).unwrap_or_default()))))
                .collect()
        };
        self.working = true;
        cx.notify();
        let handle = window.window_handle();
        cx.spawn(async move |this, cx| {
            let (done, errors) = cx
                .background_spawn(async move {
                    let mut done = 0usize;
                    let mut errors = Vec::new();
                    if let Err(e) = std::fs::create_dir_all(&dest) {
                        errors.push(e.to_string());
                        return (done, errors);
                    }
                    for (_, src, json) in jobs {
                        let name = src.file_name().unwrap_or_default().to_owned();
                        match std::fs::copy(&src, dest.join(&name)) {
                            Ok(_) => {
                                done += 1;
                                if let Some(json) = json {
                                    let side = dest.join(format!("{}.lumen.json", name.to_string_lossy()));
                                    if let Err(e) = std::fs::write(side, json) {
                                        errors.push(e.to_string());
                                    }
                                }
                            }
                            Err(e) => errors.push(format!("{}: {e}", src.display())),
                        }
                    }
                    (done, errors)
                })
                .await;
            cx.update(|cx| {
                state.update(cx, |s, cx| {
                    let mut msg = format!("{done} {}", t("originals exported", "个原始文件已导出"));
                    if let Some(first) = errors.first() {
                        msg = format!("{msg} · {} {} ({first})", errors.len(), t("failed", "个失败"));
                    }
                    s.notify(msg, cx)
                })
            });
            this.update(cx, |d, cx| {
                d.working = false;
                cx.notify();
            })
            .ok();
            handle.update(cx, |_, window, cx| window.close_dialog(cx)).ok();
        })
        .detach();
    }
}

impl Render for ExportDialog {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let p = pal(cx).clone();
        let this = cx.entity();
        let s = self.state.read(cx);
        let n = s.selection.len();
        let recipes: Vec<(usize, SharedString)> = s.recipes.iter().enumerate().map(|(i, r)| (i, r.name.clone().into())).collect();
        let recipe = s.recipes.get(self.recipe).cloned();
        let format = self.format.or(recipe.as_ref().map(|r| r.format)).unwrap_or_default();
        let dest_label = self
            .dest
            .as_ref()
            .map(|d| d.display().to_string())
            .or_else(|| recipe.as_ref().and_then(|r| r.destination.as_ref()).map(|d| d.display().to_string()))
            .unwrap_or_else(|| t("Output folder next to originals", "原图旁的 Output 文件夹").to_string());
        let can_run = n > 0 && !self.working && (!self.originals || self.dest.is_some());

        let body = if self.originals {
            v_flex()
                .child(caption(t("Originals", "原始文件"), cx))
                .child(div().px_3().text_xs().text_color(p.text_muted).child(t(
                    "Copies the original files unchanged. Adjustments can travel along as a sidecar.",
                    "原样复制原始文件，可附带一个记录调整的旁车文件。",
                )))
                .child(switch_row("exp-bydate", t("Subfolder by date", "按日期建子文件夹"), self.by_date, {
                    let this = this.clone();
                    move |v, _, cx| {
                        this.update(cx, |d, cx| {
                            d.by_date = v;
                            cx.notify();
                        })
                    }
                }, cx))
                .child(switch_row("exp-sidecar", t("Include adjustments (.lumen.json)", "附带调整（.lumen.json）"), self.sidecar, {
                    let this = this.clone();
                    move |v, _, cx| {
                        this.update(cx, |d, cx| {
                            d.sidecar = v;
                            cx.notify();
                        })
                    }
                }, cx))
        } else {
            let this2 = this.clone();
            let this3 = this.clone();
            let this4 = this.clone();
            let naming = self.naming.clone();
            v_flex()
                .child(caption(t("Recipe", "配方"), cx))
                .child(row(t("Based on", "基于"), select("exp-recipe", recipes, self.recipe, move |i, _, cx| {
                    this2.update(cx, |d, cx| {
                        d.recipe = i;
                        d.format = None;
                        if let Some(r) = d.state.read(cx).recipes.get(i) {
                            d.quality = r.quality;
                            d.naming = r.file_name_format.clone();
                        }
                        cx.notify();
                    })
                }), cx))
                .child(row(t("Format", "格式"), select("exp-format", ExportFormat::ALL.iter().map(|f| (*f, crate::contract::format_label(*f).into())).collect(), format, move |f, _, cx| {
                    this3.update(cx, |d, cx| {
                        d.format = Some(f);
                        cx.notify();
                    })
                }), cx))
                .when(format.has_quality(), |d| {
                    d.child(ParamSlider::new("exp-quality", t("Quality", "质量"), self.quality as f32, 1.0, 100.0).default_value(90.0).step(1.0).on_change(move |v, _, cx| {
                        this4.update(cx, |d, cx| {
                            d.quality = v.round() as u8;
                            cx.notify();
                        })
                    }))
                })
                .child(row(t("File name", "文件名"), text_field("exp-naming", naming, "[Image Name]", {
                    let this = this.clone();
                    move |v, _, cx| this.update(cx, |d, _| d.naming = v)
                }, window, cx), cx))
        };

        v_flex()
            .gap_3()
            .child(
                div().w(px(320.0)).child(Segmented::new("exp-mode", if self.originals { 1 } else { 0 }).text(t("Export Variants", "导出变体")).text(t("Export Originals", "导出原始文件")).on_select({
                    let this = this.clone();
                    move |i, _, cx| {
                        this.update(cx, |d, cx| {
                            d.originals = i == 1;
                            cx.notify();
                        })
                    }
                })),
            )
            .child(body)
            .child(caption(t("Location", "位置"), cx))
            .child(
                h_flex()
                    .px_3()
                    .gap_2()
                    .child(Icon::new(IconName::Folder).small().text_color(p.text_muted))
                    .child(div().flex_1().text_xs().text_color(p.text_muted).overflow_hidden().whitespace_nowrap().text_ellipsis().child(dest_label))
                    .child(Button::new("exp-dest").xsmall().outline().label(t("Choose…", "选择…")).on_click(cx.listener(|d, _, _, cx| d.pick_dest(cx)))),
            )
            .child(
                h_flex()
                    .pt_2()
                    .justify_end()
                    .gap_2()
                    .child(Button::new("exp-cancel").outline().label(t("Cancel", "取消")).on_click(|_, window, cx| window.close_dialog(cx)))
                    .child(
                        Button::new("exp-run")
                            .primary()
                            .loading(self.working)
                            .disabled(!can_run)
                            .icon(Icon::new(IconName::Upload))
                            .label(format!("{} {n} {}", t("Export", "导出"), if self.originals { t("originals", "个原始文件") } else { t("variants", "个变体") }))
                            .on_click(cx.listener(|d, _, window, cx| d.run(window, cx))),
                    ),
            )
    }
}

pub fn open(state: &Entity<AppState>, window: &mut Window, cx: &mut App) {
    let (quality, naming) = state.read(cx).recipes.first().map(|r| (r.quality, r.file_name_format.clone())).unwrap_or((90, "[Image Name]".into()));
    let view = cx.new(|_| ExportDialog {
        state: state.clone(),
        originals: false,
        recipe: 0,
        format: None,
        quality,
        dest: None,
        naming,
        by_date: false,
        sidecar: true,
        working: false,
    });
    window.open_dialog(cx, move |dialog, _, _| dialog.title(t("Export", "导出")).w(px(520.0)).child(view.clone()));
}
