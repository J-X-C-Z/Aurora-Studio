//! A collapsible tool panel with a header of tool actions, as in Capture One's
//! tool tabs. Opening and closing animate the measured height with a spring.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::base::MotionReveal;
use gpui_kit::component::{
    Icon, Sizable as _, h_flex,
    menu::{ContextMenuExt as _, PopupMenuItem},
    tooltip::Tooltip,
    v_flex,
};
use gpui_kit::{
    AnyElement, App, Entity, InteractiveElement, IntoElement, ParentElement, RenderOnce, SharedString,
    StatefulInteractiveElement, Styled, Window, div, prelude::FluentBuilder as _, px, radians,
};

use crate::motion;
use crate::state::AppState;
use crate::theme::{metrics, pal};

type Handler = Rc<dyn Fn(&mut Window, &mut App)>;

pub struct HeaderAction {
    id: &'static str,
    icon: IconName,
    tip: SharedString,
    active: bool,
    on_click: Handler,
}

impl HeaderAction {
    pub fn new(id: &'static str, icon: IconName, tip: impl Into<SharedString>, f: impl Fn(&mut Window, &mut App) + 'static) -> Self {
        Self { id, icon, tip: tip.into(), active: false, on_click: Rc::new(f) }
    }

    pub fn active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }
}

#[derive(IntoElement)]
pub struct ToolPanel {
    state: Entity<AppState>,
    id: &'static str,
    title: SharedString,
    badge: Option<SharedString>,
    status: crate::contract::Status,
    force_open: bool,
    modified: bool,
    actions: Vec<HeaderAction>,
    content: Vec<AnyElement>,
    padded: bool,
}

impl ToolPanel {
    pub fn new(state: &Entity<AppState>, id: &'static str, title: impl Into<SharedString>) -> Self {
        Self {
            state: state.clone(),
            id,
            title: title.into(),
            badge: None,
            status: crate::contract::Status::Ready,
            force_open: false,
            modified: false,
            actions: Vec::new(),
            content: Vec::new(),
            padded: true,
        }
    }

    /// Marks that the tool differs from its defaults; shows a dot in the header.
    pub fn modified(mut self, modified: bool) -> Self {
        self.modified = modified;
        self
    }

    pub fn badge(mut self, badge: impl Into<SharedString>) -> Self {
        self.badge = Some(badge.into());
        self
    }

    /// Shows how far the engine implements this tool (`catalog.md`), with the
    /// gap as a tooltip.
    pub fn status(mut self, status: crate::contract::Status) -> Self {
        self.status = status;
        self
    }

    /// Shows the content regardless of the remembered collapsed state, e.g. while searching.
    pub fn force_open(mut self, force: bool) -> Self {
        self.force_open = force;
        self
    }

    pub fn action(mut self, action: HeaderAction) -> Self {
        self.actions.push(action);
        self
    }

    /// Adds the standard reset action.
    pub fn on_reset(self, f: impl Fn(&mut Window, &mut App) + 'static) -> Self {
        self.action(HeaderAction::new("reset", IconName::RotateCcw, crate::i18n::t("Reset", "复位"), f))
    }

    /// Adds the standard auto action.
    pub fn on_auto(self, f: impl Fn(&mut Window, &mut App) + 'static) -> Self {
        self.action(HeaderAction::new("auto", IconName::Sparkles, crate::i18n::t("Auto", "自动"), f))
    }

    pub fn child(mut self, child: impl IntoElement) -> Self {
        self.content.push(child.into_any_element());
        self
    }

    pub fn children(mut self, children: impl IntoIterator<Item = impl IntoElement>) -> Self {
        self.content.extend(children.into_iter().map(|c| c.into_any_element()));
        self
    }
}

impl RenderOnce for ToolPanel {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let p = pal(cx).clone();
        let open = self.force_open || !self.state.read(cx).panels.collapsed.contains(self.id);
        let progress = motion::toggle(motion::key(self.id, "open"), open, motion::PANEL, window, cx);
        let id = self.id;
        let state = self.state.clone();
        let dot = motion::toggle(motion::key(id, "dot"), self.modified, motion::FLOW, window, cx);

        let header = h_flex()
            .id(SharedString::from(format!("panel-{id}-header")))
            .h(px(metrics::panel_header()))
            .px_2()
            .gap_1p5()
            .bg(p.raised)
            .border_t_1()
            .border_color(p.edge)
            .cursor_pointer()
            .hover(|s| s.bg(p.hover))
            .on_click(move |e: &gpui_kit::ClickEvent, _, cx| {
                // Alt-click keeps only this panel open, as in Capture One.
                let solo = e.modifiers().alt && crate::tools::QUICK_PANELS.contains(&id);
                state.update(cx, |s, cx| if solo { s.solo_panel(id, cx) } else { s.toggle_panel(id, cx) })
            })
            .child(
                Icon::new(IconName::ChevronRight)
                    .xsmall()
                    .text_color(p.text_subtle)
                    .rotate(radians(std::f32::consts::FRAC_PI_2 * progress)),
            )
            .child(
                div()
                    .flex_1()
                    .text_xs()
                    .font_weight(gpui_kit::FontWeight::SEMIBOLD)
                    .text_color(p.text)
                    .child(self.title.clone()),
            )
            .when(dot > 0.02, |this| {
                // The "changed" dot pops in and out instead of blinking.
                this.child(div().size(px(5.0)).flex().items_center().justify_center().child(div().size(px(5.0 * dot.min(1.2))).rounded_full().bg(p.accent.opacity(dot.min(1.0)))))
            })
            .when_some(self.badge.clone(), |this, b| {
                this.child(div().px_1().rounded(px(3.0)).bg(p.well).text_xs().text_color(p.text_muted).child(b))
            })
            .when_some(self.status.chip().zip(self.status.note()), |this, (chip, note)| {
                let color = match self.status {
                    crate::contract::Status::Partial(..) => p.warning,
                    crate::contract::Status::Pending(..) => p.accent,
                    _ => p.text_subtle,
                };
                this.child(
                    div()
                        .id(SharedString::from(format!("panel-{id}-status")))
                        .px_1()
                        .rounded(px(3.0))
                        .border_1()
                        .border_color(color.opacity(0.45))
                        .text_size(px(10.0))
                        .text_color(color)
                        .child(chip)
                        .tooltip(move |window, cx| Tooltip::new(note).build(window, cx)),
                )
            })
            .children(self.actions.into_iter().map(|a| {
                let f = a.on_click.clone();
                let tip = a.tip.clone();
                div()
                    .id(SharedString::from(format!("panel-{id}-{}", a.id)))
                    .size(px(20.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .rounded(px(3.0))
                    .text_color(if a.active { p.accent } else { p.text_subtle })
                    .hover(|s| s.bg(p.pressed).text_color(p.text))
                    .tooltip(move |window, cx| Tooltip::new(tip.clone()).build(window, cx))
                    .on_click(move |_, window, cx| {
                        cx.stop_propagation();
                        f(window, cx)
                    })
                    .child(Icon::new(a.icon).xsmall())
            }));

        // Right-click: put this panel on the Quick tab, or arrange it there.
        let header = if crate::tools::QUICK_PANELS.contains(&id) {
            let st = self.state.clone();
            header
                .context_menu(move |menu, _, cx| {
                    let s = st.read(cx);
                    let quick = s.in_quick(id);
                    let on_quick = s.tool_tab == crate::state::ToolTab::Quick;
                    let (a, b, c) = (st.clone(), st.clone(), st.clone());
                    let (d, e, f) = (st.clone(), st.clone(), st.clone());
                    let menu = menu
                        .item(PopupMenuItem::new(crate::i18n::t("Show Only This Panel  (Alt-click)", "仅展开此面板（Alt+单击）")).on_click(move |_, _, cx| d.update(cx, |s, cx| s.solo_panel(id, cx))))
                        .item(PopupMenuItem::new(crate::i18n::t("Expand All Panels", "展开全部面板")).on_click(move |_, _, cx| e.update(cx, |s, cx| s.set_all_panels(true, cx))))
                        .item(PopupMenuItem::new(crate::i18n::t("Collapse All Panels", "折叠全部面板")).on_click(move |_, _, cx| f.update(cx, |s, cx| s.set_all_panels(false, cx))))
                        .separator();
                    let menu = menu.item(
                        PopupMenuItem::new(if quick { crate::i18n::t("Remove from Quick Tab", "从快速页签移除") } else { crate::i18n::t("Add to Quick Tab", "添加到快速页签") })
                            .on_click(move |_, _, cx| a.update(cx, |s, cx| s.toggle_quick(id, cx))),
                    );
                    if quick && on_quick {
                        menu.separator()
                            .item(PopupMenuItem::new(crate::i18n::t("Move Up", "上移")).on_click(move |_, _, cx| b.update(cx, |s, cx| s.move_quick(id, -1, cx))))
                            .item(PopupMenuItem::new(crate::i18n::t("Move Down", "下移")).on_click(move |_, _, cx| c.update(cx, |s, cx| s.move_quick(id, 1, cx))))
                    } else {
                        menu
                    }
                })
                .into_any_element()
        } else {
            header.into_any_element()
        };

        let body = v_flex()
            .w_full()
            .when(self.padded, |this| this.py_1p5())
            .bg(p.panel)
            .children(self.content)
            .into_any_element();

        v_flex()
            .w_full()
            .border_b_1()
            .border_color(p.divider)
            .child(header)
            .child(MotionReveal::new(SharedString::from(format!("panel-{id}-reveal")), progress, body))
    }
}

/// A labeled row for non-slider controls inside a panel.
pub fn row(label: impl Into<SharedString>, control: impl IntoElement, cx: &App) -> AnyElement {
    let p = pal(cx);
    h_flex()
        .min_h(px(metrics::row() + 2.0))
        .px_3()
        .gap_2()
        .child(div().w(px(metrics::LABEL_W)).flex_shrink_0().text_xs().text_color(p.text_muted).child(label.into()))
        .child(div().flex_1().min_w_0().child(control))
        .into_any_element()
}

/// A small caption separating groups inside a panel.
pub fn caption(text: impl Into<SharedString>, cx: &App) -> AnyElement {
    let p = pal(cx);
    h_flex()
        .px_3()
        .pt_2()
        .pb_0p5()
        .gap_2()
        .text_xs()
        .text_color(p.text_subtle)
        .child(div().flex_shrink_0().child(text.into()))
        .child(div().flex_1().h(px(1.0)).bg(p.edge))
        .into_any_element()
}
