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
use crate::tools;

const TAB_H: f32 = 40.0;

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
    let right = state.read(cx).view.inspector_right;
    let accent = p.accent;
    let index = tabs.iter().position(|t| *t == current).unwrap_or(0) as f32;
    let y = motion::value(motion::key("tool-strip", "indicator"), index * TAB_H, motion::GLIDE, window, cx);

    let st_menu = state.clone();
    v_flex()
        .id("tool-strip")
        .w(px(metrics::TOOL_STRIP))
        .h_full()
        .flex_shrink_0()
        .bg(p.chrome)
        .when(right, |d| d.border_l_1())
        .when(!right, |d| d.border_r_1())
        .border_color(p.divider)
        // Right-click: choose which tool tabs this workspace shows.
        .context_menu(move |mut menu, _, cx| {
            let s = st_menu.read(cx);
            for tab in s.tab_order() {
                let shown = !s.tab_hidden(tab);
                let st = st_menu.clone();
                menu = menu.item(PopupMenuItem::new(tab.label()).checked(shown).on_click(move |_, _, cx| st.update(cx, |s, cx| s.set_tab_hidden(tab, shown, cx))));
            }
            menu.separator()
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
                        .h(px(TAB_H - 6.0))
                        .rounded(px(6.0))
                        .bg(p.accent_soft),
                )
                .child(
                    div()
                        .absolute()
                        .when(!right, |d| d.left_0().rounded_r(px(2.0)))
                        .when(right, |d| d.right_0().rounded_l(px(2.0)))
                        .top(px(y + 10.0))
                        .h(px(TAB_H - 20.0))
                        .w(px(3.0))
                        .bg(p.accent),
                )
                .children(tabs.iter().enumerate().map(|(i, tab)| {
                    let tab = *tab;
                    let selected = tab == current;
                    let st = state.clone();
                    let tip: SharedString = match tab.shortcut() {
                        Some(k) => format!("{}  ({})", tab.label(), crate::platform::keys(k)).into(),
                        None => tab.label().into(),
                    };
                    div()
                        .id(SharedString::from(format!("tab-{tab:?}")))
                        .h(px(TAB_H))
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
                        .on_drag(DraggedTab(tab), |d, _, _, cx| {
                            let tab = d.0;
                            cx.new(|_| DraggedTab(tab))
                        })
                        .drag_over::<DraggedTab>(move |d, _, _, _| d.border_t_2().border_color(accent))
                        .on_drop({
                            let st = state.clone();
                            move |d: &DraggedTab, _, cx| {
                                let moved = d.0;
                                st.update(cx, |s, cx| s.move_tab(moved, Some(tab), cx))
                            }
                        })
                        .child(
                            // A soft pill under the pointer, so the hit area reads as a button.
                            div()
                                .size(px(TAB_H - 8.0))
                                .flex()
                                .items_center()
                                .justify_center()
                                .rounded(px(6.0))
                                .when(!selected, |d| d.group_hover(SharedString::from(format!("tab-g-{tab:?}")), |s| s.bg(p.hover)))
                                .child(Icon::new(tab.icon()).size(px(18.0))),
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
    let layered = matches!(tab, ToolTab::Color | ToolTab::Exposure | ToolTab::Details | ToolTab::Layers | ToolTab::Quick);

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
                .child(div().text_sm().font_weight(gpui_kit::FontWeight::SEMIBOLD).text_color(p.text).child(tab.label()))
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
                                            s.active_layer = None;
                                            s.active_node = None;
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
