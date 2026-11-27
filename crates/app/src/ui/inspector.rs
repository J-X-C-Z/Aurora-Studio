//! The tool tab strip and the scrolling column of tool panels for the selected tab.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _, h_flex,
    menu::{ContextMenuExt as _, PopupMenuItem},
    scroll::ScrollableElement as _,
    tooltip::Tooltip,
    v_flex,
};
use gpui_kit::AnimationExt as _;
use gpui_kit::AppContext as _;
use gpui_kit::{
    App, Entity, InteractiveElement, IntoElement, ParentElement, SharedString, StatefulInteractiveElement, Styled,
    Window, div, prelude::FluentBuilder as _, px,
};

use crate::i18n::t;
use crate::motion;
use crate::state::{AppState, Scope, ToolTab};
use crate::theme::{metrics, pal};
use crate::state::panel_order::DropAt;
use crate::tools;
use crate::widgets::panel::DraggedPanel;

const TAB_H: f32 = 40.0;
/// Tab height and strip width when the tabs show their names under the icons.
const TAB_H_LABELED: f32 = 50.0;
const STRIP_LABELED: f32 = 60.0;

/// A tool tab being dragged to a new place in the strip; also its drag image.
#[derive(Clone, Copy)]
pub struct DraggedTab(pub ToolTab);

impl gpui_kit::Render for DraggedTab {
    fn render(&mut self, _: &mut Window, cx: &mut gpui_kit::Context<Self>) -> impl IntoElement {
        let p = pal(cx).clone();
        h_flex()
            .gap_1p5()
            .px_2()
            .py_1()
            .rounded(px(6.0))
            .bg(p.raised)
            .border_1()
            .border_color(p.edge)
            .text_xs()
            .text_color(p.text)
            .child(Icon::new(self.0.icon()).xsmall())
            .child(self.0.label())
    }
}

pub fn tool_strip(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> impl IntoElement + use<> {
    let p = pal(cx).clone();
    let current = state.read(cx).tool_tab;
    let tabs = state.read(cx).visible_tabs();
    let right = state.read(cx).prefs.view.inspector_right;
    let labels = state.read(cx).prefs.settings.workspace.tab_labels;
    let names: Vec<SharedString> = tabs.iter().map(|t| state.read(cx).tab_name(*t)).collect();
    let tab_h = if labels { TAB_H_LABELED } else { TAB_H };
    let panel_drop = if cx.has_active_drag() { state.read(cx).panel_drag.at } else { DropAt::Nowhere };
    let accent = p.accent;
    let index = tabs.iter().position(|t| *t == current).unwrap_or(0) as f32;
    let y = motion::value(motion::key("tool-strip", "indicator"), index * tab_h, motion::GLIDE, window, cx);

    let st_menu = state.clone();
    v_flex()
        .id("tool-strip")
        .w(px(if labels { STRIP_LABELED } else { metrics::TOOL_STRIP }))
        .h_full()
        .flex_shrink_0()
        .bg(p.chrome)
        .when(right, |d| d.border_l_1())
        .when(!right, |d| d.border_r_1())
        .border_color(p.divider)
        // Right-click: choose which tool tabs this workspace shows, make and
        // name tabs of the user's own, and whether the strip shows names.
        .context_menu(move |mut menu, _, cx| {
            let s = st_menu.read(cx);
            for tab in s.tab_order() {
                let shown = !s.tab_hidden(tab);
                let st = st_menu.clone();
                menu = menu.item(PopupMenuItem::new(s.tab_name(tab)).checked(shown).on_click(move |_, _, cx| st.update(cx, |s, cx| s.set_tab_hidden(tab, shown, cx))));
            }
            let current = s.tool_tab;
            let labels = s.prefs.settings.workspace.tab_labels;
            let (st_new, st_labels) = (st_menu.clone(), st_menu.clone());
            menu = menu.separator().item(
                PopupMenuItem::new(t("New Tab…", "新建页签…"))
                    .on_click(move |_, window, cx| crate::ui::workspaces::new_tab_dialog(&st_new, Vec::new(), window, cx)),
            );
            if let ToolTab::User(key) = current {
                let name = s.tab_name(current);
                let (st_rename, st_delete) = (st_menu.clone(), st_menu.clone());
                menu = menu
                    .item(
                        PopupMenuItem::new(format!("{} “{name}”…", t("Rename", "重命名")))
                            .on_click(move |_, window, cx| crate::ui::workspaces::rename_tab_dialog(&st_rename, key, window, cx)),
                    )
                    .item(
                        PopupMenuItem::new(format!("{} “{name}”", t("Delete", "删除")))
                            .on_click(move |_, _, cx| st_delete.update(cx, |s, cx| s.delete_user_tab(key, cx))),
                    );
            }
            menu.separator()
                .item(
                    PopupMenuItem::new(t("Show Tab Names", "显示页签名称"))
                        .checked(labels)
                        .on_click(move |_, _, cx| st_labels.update(cx, |s, cx| s.set_tab_labels(!labels, cx))),
                )
                .item(PopupMenuItem::new(t("Tools on the Other Side", "工具栏换边")).on_click(|_, window, cx| window.dispatch_action(Box::new(crate::actions::ToggleInspectorSide), cx)))
                .item(PopupMenuItem::new(t("Save Workspace…", "存储工作区…")).on_click(|_, window, cx| window.dispatch_action(Box::new(crate::actions::SaveWorkspace), cx)))
        })
        .pt_1()
        .child(
            div()
                .relative()
                .child(
                    div()
                        .absolute()
                        .left(px(4.0))
                        .right(px(4.0))
                        .top(px(y + 3.0))
                        .h(px(tab_h - 6.0))
                        .rounded(px(6.0))
                        .bg(p.accent_soft),
                )
                .child(
                    div()
                        .absolute()
                        .when(!right, |d| d.left_0().rounded_r(px(2.0)))
                        .when(right, |d| d.right_0().rounded_l(px(2.0)))
                        .top(px(y + 10.0))
                        .h(px(tab_h - 20.0))
                        .w(px(3.0))
                        .bg(p.accent),
                )
                .children(tabs.iter().enumerate().map(|(i, tab)| {
                    let tab = *tab;
                    let selected = tab == current;
                    let st = state.clone();
                    // Ctrl+1…9 follow the order the strip shows.
                    let tip: SharedString = if i < 9 {
                        let key = if cfg!(target_os = "macos") { "⌘" } else { "Ctrl+" };
                        format!("{}  ({key}{})", names[i], i + 1).into()
                    } else {
                        names[i].clone()
                    };
                    let takes_panels = matches!(tab, ToolTab::Quick | ToolTab::User(_));
                    let drop_here = panel_drop == DropAt::Tab(tab);
                    let name = names[i].clone();
                    div()
                        .id(SharedString::from(format!("tab-{tab:?}")))
                        .h(px(tab_h))
                        .w_full()
                        .flex()
                        .items_center()
                        .justify_center()
                        .cursor_pointer()
                        .text_color(if selected { p.text } else { p.text_subtle })
                        .hover(|s| s.text_color(p.text))
                        .group(SharedString::from(format!("tab-g-{tab:?}")))
                        .tooltip(move |window, cx| Tooltip::new(tip.clone()).build(window, cx))
                        .on_click(move |_, _, cx| {
                            st.update(cx, |s, cx| {
                                s.tool_tab = tab;
                                cx.notify();
                            })
                        })
                        // Drag a tab onto another to put it in front of that one.
                        .on_drag(DraggedTab(tab), {
                            let st = state.clone();
                            move |d, _, _, cx| {
                                st.update(cx, |s, cx| s.start_drag(None, cx));
                                let tab = d.0;
                                cx.new(|_| DraggedTab(tab))
                            }
                        })
                        .drag_over::<DraggedTab>(move |d, _, _, _| d.border_t_2().border_color(accent))
                        .on_drop({
                            let st = state.clone();
                            move |d: &DraggedTab, _, cx| {
                                let moved = d.0;
                                st.update(cx, |s, cx| s.move_tab(moved, Some(tab), cx))
                            }
                        })
                        // A tool panel dropped on the Quick tab or a tab of the
                        // user's own goes on that tab, as in Capture One.
                        .when(takes_panels, |d| {
                            let (st, st_drop) = (state.clone(), state.clone());
                            // Hover styles do not apply during a drag, so the
                            // highlight follows the recorded drop place instead.
                            d.on_drag_move::<DraggedPanel>(move |e, _, cx| {
                                let at = DropAt::Tab(tab);
                                let dragged = e.drag(cx).id;
                                if e.bounds.contains(&e.event.position) && crate::tools::tab_panel(dragged).is_some() {
                                    st.update(cx, |s, cx| s.hover_panel_drop(dragged, at, cx));
                                } else {
                                    st.update(cx, |s, cx| s.leave_panel_drop(|a| a == at, cx));
                                }
                            })
                            .on_drop(move |d: &DraggedPanel, _, cx| {
                                st_drop.update(cx, |s, cx| {
                                    s.start_drag(None, cx);
                                    s.add_tab_panel(tab, d.id, cx)
                                })
                            })
                            .when(drop_here, |d| d.bg(p.accent_soft).text_color(p.accent))
                        })
                        .child(
                            // A soft pill under the pointer, so the hit area reads as a button.
                            v_flex()
                                .size(px(TAB_H - 8.0))
                                .when(labels, |d| d.w(px(STRIP_LABELED - 6.0)).h(px(TAB_H_LABELED - 6.0)).gap_0p5())
                                .items_center()
                                .justify_center()
                                .rounded(px(6.0))
                                .when(!selected, |d| d.group_hover(SharedString::from(format!("tab-g-{tab:?}")), |s| s.bg(p.hover)))
                                .child(Icon::new(tab.icon()).size(px(if labels { 16.0 } else { 18.0 })))
                                .when(labels, |d| {
                                    d.child(
                                        div()
                                            .max_w(px(STRIP_LABELED - 10.0))
                                            .overflow_hidden()
                                            .whitespace_nowrap()
                                            .text_ellipsis()
                                            .text_size(px(10.0))
                                            .line_height(px(12.0))
                                            .child(name),
                                    )
                                }),
                        )
                        // Tabs cascade in when a workspace changes the set.
                        .with_animation(
                            SharedString::from(format!("tab-in-{tab:?}")),
                            gpui_kit::Animation::new(std::time::Duration::from_millis((160.0 + 25.0 * i as f32) as u64)),
                            |el, t| el.opacity(t),
                        )
                })),
        )
        .child(div().flex_1())
        .child(
            div()
                .id("strip-prefs")
                .h(px(TAB_H))
                .flex()
                .items_center()
                .justify_center()
                .cursor_pointer()
                .text_color(p.text_subtle)
                .hover(|s| s.text_color(p.text))
                .tooltip(|window, cx| Tooltip::new(t("Preferences", "首选项")).build(window, cx))
                .on_click(|_, window, cx| window.dispatch_action(Box::new(crate::actions::ShowPreferences), cx))
                .child(Icon::new(IconName::Settings).size(px(17.0))),
        )
}

pub fn inspector(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> impl IntoElement + use<> {
    let p = pal(cx).clone();
    let s = state.read(cx);
    let tab = s.tool_tab;
    let scope = s.scope();
    let layer_name = s.active_layer_ref().map(|l| l.name.clone());
    let layered = matches!(tab, ToolTab::Grading | ToolTab::Details | ToolTab::Quick | ToolTab::User(_));
    let tab_name = s.tab_name(tab);

    let content = tools::render_tab(tab, state, window, cx);
    let st = state.clone();

    v_flex()
        .size_full()
        .bg(p.panel)
        .child(
            h_flex()
                .h(px(32.0))
                .flex_shrink_0()
                .px_3()
                .gap_2()
                .bg(p.chrome)
                .border_b_1()
                .border_color(p.divider)
                .child(div().text_sm().font_weight(gpui_kit::FontWeight::SEMIBOLD).text_color(p.text).child(tab_name))
                .child(div().flex_1())
                .when(layered, |this| {
                    this.child(
                        div()
                            .id("scope-chip")
                            .flex()
                            .items_center()
                            .gap_1()
                            .px_2()
                            .h(px(20.0))
                            .rounded(px(10.0))
                            .text_xs()
                            .when(scope == Scope::Global, |d| d.bg(p.well).text_color(p.text_muted))
                            .when(scope != Scope::Global, |d| d.bg(p.accent_soft).text_color(p.accent).cursor_pointer())
                            .child(Icon::new(IconName::Layers).xsmall())
                            .child(layer_name.clone().unwrap_or_else(|| t("Base", "基础").into()))
                            .when(scope != Scope::Global, |d| {
                                d.child(Icon::new(IconName::X).xsmall())
                                    .tooltip(|window, cx| Tooltip::new(t("Back to the base adjustments", "返回基础调整")).build(window, cx))
                                    .on_click(move |_, _, cx| {
                                        st.update(cx, |s, cx| {
                                            s.editor.active_layer = None;
                                            s.editor.active_node = None;
                                            cx.notify();
                                        })
                                    })
                            }),
                    )
                }),
        )
        .child(
            div()
                .id("inspector-scroll")
                .flex_1()
                .min_h_0()
                .overflow_y_scrollbar()
                .child(motion::enter(SharedString::from(format!("tab-content-{tab:?}")), v_flex().w_full().pb_8().child(content), 8.0)),
        )
}
