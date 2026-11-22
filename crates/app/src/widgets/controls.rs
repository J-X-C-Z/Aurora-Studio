//! Small shared controls: segmented choices with a gliding indicator, dropdown
//! selects, icon buttons and switch rows.

use std::rc::Rc;

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _,
    button::Button,
    h_flex,
    menu::{DropdownMenu as _, PopupMenuItem},
    switch::Switch,
    tooltip::Tooltip,
};
use gpui_kit::{
    AnyElement, App, InteractiveElement, IntoElement, ParentElement, RenderOnce, SharedString, StatefulInteractiveElement,
    Styled, Window, div, prelude::FluentBuilder as _, px, relative,
};

use crate::motion;
use crate::theme::{metrics, pal};

type OnIndex = Rc<dyn Fn(usize, &mut Window, &mut App)>;

pub enum SegmentContent {
    Text(SharedString),
    /// A colored dot, used for channel pickers.
    Swatch(gpui_kit::Hsla, SharedString),
}

/// Equal-width segments with an indicator that springs to the selection.
#[derive(IntoElement)]
pub struct Segmented {
    id: SharedString,
    items: Vec<SegmentContent>,
    selected: usize,
    on_select: Option<OnIndex>,
    height: f32,
}

impl Segmented {
    pub fn new(id: impl Into<SharedString>, selected: usize) -> Self {
        Self { id: id.into(), items: Vec::new(), selected, on_select: None, height: 22.0 }
    }

    pub fn text(mut self, label: impl Into<SharedString>) -> Self {
        self.items.push(SegmentContent::Text(label.into()));
        self
    }


    pub fn swatch(mut self, color: gpui_kit::Hsla, tip: impl Into<SharedString>) -> Self {
        self.items.push(SegmentContent::Swatch(color, tip.into()));
        self
    }

    pub fn height(mut self, h: f32) -> Self {
        self.height = h;
        self
    }

    pub fn on_select(mut self, f: impl Fn(usize, &mut Window, &mut App) + 'static) -> Self {
        self.on_select = Some(Rc::new(f));
        self
    }
}

impl RenderOnce for Segmented {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let p = pal(cx).clone();
        let n = self.items.len().max(1);
        let pos = motion::value(motion::key(&self.id, "indicator"), self.selected as f32, motion::GLIDE, window, cx);
        let id = self.id.clone();
        div()
            .id(id.clone())
            .relative()
            .h(px(self.height))
            .w_full()
            .p(px(2.0))
            .rounded(px(5.0))
            .bg(p.well)
            .child(
                div()
                    .absolute()
                    .top(px(2.0))
                    .bottom(px(2.0))
                    .left(relative(pos / n as f32))
                    .w(relative(1.0 / n as f32))
                    .px(px(2.0))
                    .child(div().size_full().rounded(px(4.0)).bg(p.pressed).shadow_sm()),
            )
            .child(h_flex().relative().size_full().children(self.items.into_iter().enumerate().map(|(i, item)| {
                let on_select = self.on_select.clone();
                let selected = i == self.selected;
                let color = if selected { p.text } else { p.text_muted };
                let (content, tip): (AnyElement, Option<SharedString>) = match item {
                    SegmentContent::Text(t) => (div().text_xs().child(t).into_any_element(), None),
                    SegmentContent::Swatch(c, tip) => {
                        (div().size(px(8.0)).rounded_full().bg(c).into_any_element(), Some(tip))
                    }
                };
                div()
                    .id(SharedString::from(format!("{id}-{i}")))
                    .flex_1()
                    .h_full()
                    .flex()
                    .items_center()
                    .justify_center()
                    .text_color(color)
                    .cursor_pointer()
                    .hover(|s| s.text_color(p.text))
                    .when_some(tip, |this, tip| this.tooltip(move |w, cx| Tooltip::new(tip.clone()).build(w, cx)))
                    .on_click(move |_, window, cx| {
                        if let Some(f) = &on_select {
                            f(i, window, cx);
                        }
                    })
                    .child(content)
            })))
    }
}

/// A compact dropdown showing the current choice.
pub fn select<T: Copy + PartialEq + 'static>(
    id: impl Into<SharedString>,
    options: Vec<(T, SharedString)>,
    current: T,
    on_select: impl Fn(T, &mut Window, &mut App) + 'static,
) -> AnyElement {
    let label = options.iter().find(|(v, _)| *v == current).map(|(_, l)| l.clone()).unwrap_or_default();
    let on_select = Rc::new(on_select);
    let id: SharedString = id.into();
    Button::new(id)
        .label(label)
        .xsmall()
        .outline()
        .w_full()
        .dropdown_caret(true)
        .dropdown_menu(move |mut menu, _, _| {
            for (value, label) in &options {
                let value = *value;
                let f = on_select.clone();
                menu = menu.item(
                    PopupMenuItem::new(label.clone()).checked(value == current).on_click(move |_, window, cx| f(value, window, cx)),
                );
            }
            menu.max_h(px(420.0)).scrollable(true)
        })
        .into_any_element()
}

/// A square icon button for toolbars and panel rows.
#[derive(IntoElement)]
pub struct IconBtn {
    id: SharedString,
    icon: IconName,
    tip: SharedString,
    shortcut: Option<SharedString>,
    active: bool,
    disabled: bool,
    size: f32,
    on_click: Option<Rc<dyn Fn(&mut Window, &mut App)>>,
}

impl IconBtn {
    pub fn new(id: impl Into<SharedString>, icon: IconName, tip: impl Into<SharedString>) -> Self {
        Self { id: id.into(), icon, tip: tip.into(), shortcut: None, active: false, disabled: false, size: 26.0, on_click: None }
    }

    pub fn active(mut self, active: bool) -> Self {
        self.active = active;
        self
    }

    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    pub fn size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    pub fn shortcut(mut self, key: impl Into<SharedString>) -> Self {
        self.shortcut = Some(key.into());
        self
    }

    pub fn on_click(mut self, f: impl Fn(&mut Window, &mut App) + 'static) -> Self {
        self.on_click = Some(Rc::new(f));
        self
    }
}

impl RenderOnce for IconBtn {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let p = pal(cx).clone();
        let lit = motion::toggle(motion::key(&self.id, "lit"), self.active, motion::QUICK, window, cx);
        let tip = match &self.shortcut {
            Some(k) => SharedString::from(format!("{}  ({k})", self.tip)),
            None => self.tip.clone(),
        };
        let on_click = self.on_click.clone();
        div()
            .id(self.id.clone())
            .size(px(self.size))
            .flex()
            .items_center()
            .justify_center()
            .rounded(px(5.0))
            .bg(p.accent.opacity(0.22 * lit))
            .text_color(if self.disabled {
                p.text_disabled
            } else if self.active {
                p.accent
            } else {
                p.text_muted
            })
            .when(!self.disabled, |this| {
                this.cursor_pointer()
                    .hover(|s| s.bg(p.hover).text_color(p.text))
                    .active(|s| s.bg(p.pressed))
                    .on_click(move |_, window, cx| {
                        if let Some(f) = &on_click {
                            f(window, cx);
                        }
                    })
            })
            .tooltip(move |window, cx| Tooltip::new(tip.clone()).build(window, cx))
            .child(Icon::new(self.icon).small())
    }
}

/// A switch on the right of a label, sized to the inspector rows.
pub fn switch_row(
    id: impl Into<SharedString>,
    label: impl Into<SharedString>,
    checked: bool,
    on_change: impl Fn(bool, &mut Window, &mut App) + 'static,
    cx: &App,
) -> AnyElement {
    let p = pal(cx);
    let id: SharedString = id.into();
    h_flex()
        .h(px(metrics::row() + 2.0))
        .px_3()
        .child(div().flex_1().text_xs().text_color(p.text_muted).child(label.into()))
        .child(Switch::new(id).xsmall().checked(checked).on_change(move |v: &bool, window, cx| on_change(*v, window, cx)))
        .into_any_element()
}
