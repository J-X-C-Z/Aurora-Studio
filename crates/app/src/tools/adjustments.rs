//! Adjustments tab: styles and presets, the selective adjustments clipboard, and history.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _,
    button::{Button, ButtonVariants as _},
    checkbox::Checkbox,
    h_flex, v_flex,
};
use gpui_kit::{
    AnyElement, App, Entity, InteractiveElement, IntoElement, ParentElement, SharedString, StatefulInteractiveElement,
    Styled, Window, div, prelude::FluentBuilder as _, px,
};

use crate::i18n::t;
use crate::state::{AppState, ClipGroup, HoverLook};
use crate::theme::pal;
use crate::widgets::*;

pub fn render(state: &Entity<AppState>, _window: &mut Window, cx: &mut App) -> AnyElement {
    v_flex()
        .w_full()
        .child(user_styles_panel(state, cx))
        .child(styles_panel(state, cx))
        .child(clipboard_panel(state, cx))
        .child(history_panel(state, cx))
        .into_any_element()
}

/// Styles the user saved from their own adjustments: apply, rename, delete.
fn user_styles_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let styles = s.settings.user_styles.clone();
    let applied = s.adj().style.clone();
    let n = s.selection.len().max(1);
    let st_save = state.clone();
    let mut panel = ToolPanel::new(state, "user-styles", t("User Styles", "用户风格"))
        .badge(styles.len().to_string())
        .child(
            h_flex().px_3().pb_1().child(
                Button::new("style-save-new")
                    .xsmall()
                    .outline()
                    .icon(Icon::new(IconName::Plus))
                    .label(t("Save Current as Style…", "将当前调整存储为风格…"))
                    .on_click(move |_, window, cx| crate::ui::naming::save_style(&st_save, window, cx)),
            ),
        );
    if styles.is_empty() {
        panel = panel.child(div().px_3().pb_2().text_xs().text_color(p.text_subtle).child(t(
            "Saved styles appear here and apply to every selected image.",
            "存储的风格会列在这里，可套用到所有所选图像。",
        )));
    }
    for (i, style) in styles.into_iter().enumerate() {
        let on = applied.as_deref() == Some(style.name.as_str());
        let (st_apply, st_ren, st_del) = (state.clone(), state.clone(), state.clone());
        let tools = style.groups.iter().filter_map(|g| ClipGroup::from_id(g)).map(|g| g.label()).collect::<Vec<_>>().join("、");
        let name = style.name.clone();
        panel = panel.child(
            h_flex()
                .id(SharedString::from(format!("ustyle-{i}")))
                .group("ustyle")
                .h(px(26.0))
                .mx_2()
                .px_2()
                .gap_2()
                .rounded(px(5.0))
                .cursor_pointer()
                .when(on, |d| d.bg(p.accent_soft))
                .hover(|d| d.bg(p.hover))
                .tooltip(move |window, cx| gpui_kit::component::tooltip::Tooltip::new(tools.clone()).build(window, cx))
                .on_hover({
                    let st = state.clone();
                    move |h, _, cx| {
                        let look = h.then_some(HoverLook::User(i));
                        st.update(cx, |s, cx| s.preview_look(look, cx))
                    }
                })
                .on_click(move |_, _, cx| st_apply.update(cx, |s, cx| s.apply_user_style(i, cx)))
                .child(Icon::new(IconName::Bookmark).xsmall().text_color(if on { p.accent } else { p.text_subtle }))
                .child(div().flex_1().min_w_0().text_xs().text_color(p.text).overflow_hidden().whitespace_nowrap().text_ellipsis().child(style.name.clone()))
                .when(on, |d| d.child(Icon::new(IconName::Check).xsmall().text_color(p.accent)))
                .child(
                    h_flex()
                        .gap_0p5()
                        .invisible()
                        .group_hover("ustyle", |d| d.visible())
                        .child(Button::new(SharedString::from(format!("ustyle-ren-{i}"))).xsmall().ghost().icon(Icon::new(IconName::Pencil)).tooltip(t("Rename", "重命名")).on_click(
                            move |_, window, cx| {
                                cx.stop_propagation();
                                let st = st_ren.clone();
                                crate::ui::naming::prompt(t("Rename Style", "重命名风格"), "", name.clone(), t("Style name", "风格名称"), window, cx, move |n, cx| {
                                    st.update(cx, |s, cx| s.rename_user_style(i, n, cx))
                                });
                            },
                        ))
                        .child(Button::new(SharedString::from(format!("ustyle-del-{i}"))).xsmall().ghost().icon(Icon::new(IconName::Trash)).tooltip(t("Delete", "删除")).on_click(
                            move |_, _, cx| {
                                cx.stop_propagation();
                                st_del.update(cx, |s, cx| s.delete_user_style(i, cx))
                            },
                        )),
                ),
        );
    }
    if n > 1 {
        panel = panel.child(div().px_3().pt_1().text_xs().text_color(p.text_subtle).child(format!("{} {n} {}", t("Applies to", "将套用到"), t("selected images", "张所选图像"))));
    }
    panel
}

fn styles_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let styles = s.styles.clone();
    let applied = s.adj().style.clone();
    let stack = s.panels.stack_styles;
    let mut groups: Vec<String> = styles.iter().map(|s| s.group.clone()).collect();
    groups.dedup();
    let st = state.clone();
    let st_clear = state.clone();
    let st_import = state.clone();
    let import_status = crate::contract::STYLE_IMPORT;
    let mut panel = ToolPanel::new(state, "styles", t("Styles and Presets", "风格与预设"))
        .modified(applied.is_some())
        .on_reset(move |_, cx| st_clear.update(cx, |s, cx| s.edit(t("Remove style", "移除风格"), |a| a.style = None, cx)))
        // Capture One `.costyle` / `.copreset` files; built ahead of the engine (`planned::import_style`).
        .child(
            h_flex().px_3().pb_1().gap_2().items_center().child(
                Button::new("style-import")
                    .xsmall()
                    .outline()
                    .icon(Icon::new(IconName::ImageDown))
                    .label(t("Import Capture One Styles…", "导入 Capture One 风格…"))
                    .on_click(move |_, _, cx| {
                        let st = st_import.clone();
                        let rx = cx.prompt_for_paths(gpui_kit::PathPromptOptions { files: true, directories: false, multiple: true, prompt: None });
                        cx.spawn(async move |cx| {
                            if let Ok(Ok(Some(paths))) = rx.await {
                                st.update(cx, |s, cx| s.import_styles(paths, cx));
                            }
                        })
                        .detach();
                    }),
            )
            .when(!crate::planned::STYLE_IMPORT_READY, |d| d.child(div().text_xs().text_color(p.warning).child(import_status.chip().unwrap_or_default()))),
        )
        .child(switch_row("stack-styles", t("Stack styles", "叠加风格"), stack, move |v, _, cx| {
            st.update(cx, |s, cx| {
                s.panels.stack_styles = v;
                cx.notify();
            })
        }, cx));
    for g in groups {
        panel = panel.child(caption(crate::i18n::builtin(&g), cx));
        for style in styles.iter().filter(|s| s.group == g) {
            let style = style.clone();
            let on = applied.as_deref() == Some(style.name.as_str());
            let st = state.clone();
            panel = panel.child(
                h_flex()
                    .id(SharedString::from(format!("style-{}", style.name)))
                    .h(px(26.0))
                    .mx_2()
                    .px_2()
                    .gap_2()
                    .rounded(px(5.0))
                    .cursor_pointer()
                    .when(on, |d| d.bg(p.accent_soft))
                    .hover(|d| d.bg(p.hover))
                    .child(Icon::new(IconName::Palette).xsmall().text_color(if on { p.accent } else { p.text_subtle }))
                    .child(div().flex_1().text_xs().text_color(p.text).child(style.name.clone()))
                    .when(on, |d| d.child(Icon::new(IconName::Check).xsmall().text_color(p.accent)))
                    .on_hover({
                        let (st, style) = (state.clone(), style.clone());
                        move |h, _, cx| {
                            let look = h.then(|| HoverLook::Style(style.clone(), stack));
                            st.update(cx, |s, cx| s.preview_look(look, cx))
                        }
                    })
                    .on_click(move |_, _, cx| {
                        st.update(cx, |s, cx| {
                            let stack = s.panels.stack_styles;
                            s.apply_style(&style, stack, cx)
                        })
                    }),
            );
        }
    }
    panel
}

fn clipboard_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let groups = s.clip_groups.clone();
    let clip = s.clipboard.clone();
    let source = clip.clone().unwrap_or_else(|| s.adj().clone());
    let (st_all, st_none, st_copy, st_apply) = (state.clone(), state.clone(), state.clone(), state.clone());
    let n = s.selection.len();
    ToolPanel::new(state, "clipboard", t("Adjustments Clipboard", "调整剪贴板"))
        .badge(if clip.is_some() { t("Copied", "已拷贝") } else { t("Empty", "空") })
        .child(
            h_flex()
                .px_3()
                .pb_1()
                .gap_1p5()
                .child(Button::new("clip-copy").xsmall().outline().icon(Icon::new(IconName::Copy)).label(t("Copy", "拷贝")).on_click(move |_, _, cx| st_copy.update(cx, |s, cx| s.copy_adjustments(cx))))
                .child(
                    Button::new("clip-apply")
                        .xsmall()
                        .primary()
                        .icon(Icon::new(IconName::ClipboardPaste))
                        .label(format!("{} ({n})", t("Apply", "应用")))
                        .on_click(move |_, _, cx| st_apply.update(cx, |s, cx| s.apply_adjustments(cx))),
                )
                .child(div().flex_1())
                .child(Button::new("clip-all").xsmall().ghost().label(t("All", "全选")).on_click(move |_, _, cx| {
                    st_all.update(cx, |s, cx| {
                        s.clip_groups = ClipGroup::ALL.iter().copied().collect();
                        cx.notify();
                    })
                }))
                .child(Button::new("clip-none").xsmall().ghost().label(t("None", "全不选")).on_click(move |_, _, cx| {
                    st_none.update(cx, |s, cx| {
                        s.clip_groups.clear();
                        cx.notify();
                    })
                })),
        )
        .children(ClipGroup::ALL.iter().map(|g| {
            let g = *g;
            let st = state.clone();
            let set = g.is_set(&source);
            h_flex()
                .h(px(22.0))
                .px_3()
                .gap_2()
                .child(
                    Checkbox::new(SharedString::from(format!("clip-{g:?}")))
                        .xsmall()
                        .checked(groups.contains(&g))
                        .label(g.label())
                        .on_change(move |v: &bool, _, cx| {
                            let v = *v;
                            st.update(cx, |s, cx| {
                                if v {
                                    s.clip_groups.insert(g);
                                } else {
                                    s.clip_groups.remove(&g);
                                }
                                cx.notify();
                            })
                        }),
                )
                .child(div().flex_1())
                .when(set, |d| d.child(div().size(px(5.0)).rounded_full().bg(p.accent)))
        }))
}

fn history_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let (undo, redo) = s.history();
    let current = undo.len();
    let mut rows: Vec<(usize, SharedString, bool)> = Vec::new();
    rows.push((0, t("Original", "原始").into(), false));
    for (i, label) in undo.iter().enumerate() {
        rows.push((i + 1, label.clone(), false));
    }
    for (i, label) in redo.iter().rev().enumerate() {
        rows.push((current + i + 1, label.clone(), true));
    }
    ToolPanel::new(state, "history", t("History", "历史记录")).badge(format!("{current}")).children(rows.into_iter().rev().map(|(pos, label, future)| {
        let st = state.clone();
        let here = pos == current;
        h_flex()
            .id(SharedString::from(format!("hist-{pos}")))
            .h(px(24.0))
            .mx_2()
            .px_2()
            .gap_2()
            .rounded(px(4.0))
            .cursor_pointer()
            .when(here, |d| d.bg(p.accent_soft))
            .hover(|d| d.bg(p.hover))
            .child(div().size(px(6.0)).rounded_full().bg(if here { p.accent } else if future { p.text_disabled } else { p.text_subtle }))
            .child(div().flex_1().text_xs().text_color(if future { p.text_disabled } else { p.text }).child(label))
            .on_click(move |_, _, cx| st.update(cx, |s, cx| s.jump_history(pos, cx)))
    }))
}
