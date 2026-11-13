//! The parameter slider used throughout the inspector.
//!
//! * Click anywhere on the track to jump, drag to set.
//! * Hold Shift or Alt while dragging for ten times finer control.
//! * Double-click the label or knob to reset to the default.
//! * Click the number to type an exact value: Enter or clicking away applies
//!   it, Esc leaves it unchanged.
//! * Ctrl+scroll over the slider nudges by one step (Shift: ten steps); plain
//!   scrolling keeps scrolling the inspector.
//! * The knob glides when the value changes from elsewhere (reset, auto, undo),
//!   and stays pinned to the pointer while dragging.

use std::rc::Rc;

use gpui_kit::base::ElementExt as _;
use gpui_kit::component::{Sizable as _, WindowExt as _, input::{Input, InputEvent, InputState}};
use gpui_kit::{
    App, AppContext as _, Bounds, ClickEvent, Context, Entity, Hsla, InteractiveElement, IntoElement, MouseButton,
    MouseDownEvent, ParentElement, Pixels, Render, RenderOnce, ScrollDelta, ScrollEvent, SharedString,
    StatefulInteractiveElement, Styled, Window, div, linear_color_stop, linear_gradient, prelude::FluentBuilder as _,
    px, relative,
};

use crate::motion;
use crate::state::{AppState, Binding};
use crate::theme::{hexa, metrics, pal};

#[derive(Clone)]
pub enum Track {
    Plain,
    /// Evenly spaced color stops painted under the track.
    Gradient(Vec<Hsla>),
}

#[derive(Clone, Copy)]
pub enum Scale {
    Linear,
    /// value = min + span · t^γ, for ranges like color temperature.
    Power(f32),
}

impl Scale {
    fn to_t(self, v: f32, min: f32, max: f32) -> f32 {
        let t = ((v - min) / (max - min)).clamp(0.0, 1.0);
        match self {
            Scale::Linear => t,
            Scale::Power(g) => t.powf(1.0 / g),
        }
    }

    fn to_value(self, t: f32, min: f32, max: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        let t = match self {
            Scale::Linear => t,
            Scale::Power(g) => t.powf(g),
        };
        min + (max - min) * t
    }
}

type OnChange = Rc<dyn Fn(f32, &mut Window, &mut App)>;
type OnCommit = Rc<dyn Fn(&mut Window, &mut App)>;

/// Drag payload; only the slider whose id matches reacts.
#[derive(Clone)]
struct SliderDrag {
    id: SharedString,
}

struct DragGhost;

impl Render for DragGhost {
    fn render(&mut self, _: &mut Window, _: &mut Context<Self>) -> impl IntoElement {
        div()
    }
}

/// Per-slider retained interaction state.
#[derive(Default)]
struct SliderState {
    bounds: Bounds<Pixels>,
    hovered: bool,
    dragging: bool,
    /// Pointer x and value when a fine-adjust drag started.
    anchor: Option<(f32, f32)>,
    /// The number is being typed.
    editing: bool,
    /// A typed value waiting for the next render, which has the live callbacks.
    pending: Option<f32>,
    input: Option<Entity<InputState>>,
    _input_sub: Option<gpui_kit::Subscription>,
}

/// Reads a typed number, tolerating a unit, a leading plus and a decimal comma.
fn parse_number(text: &str) -> Option<f32> {
    let cleaned: String = text
        .trim()
        .replace(',', ".")
        .replace('−', "-")
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | '+'))
        .collect();
    cleaned.trim_start_matches('+').parse::<f32>().ok().filter(|v| v.is_finite())
}

/// Gives the keyboard back to the workspace so shortcuts work again.
fn release_focus(window: &mut Window, cx: &mut App) {
    if window.has_active_dialog(cx) {
        return;
    }
    if let Some(focus) = cx.try_global::<crate::ui::WorkspaceFocus>().map(|f| f.0.clone()) {
        window.focus(&focus, cx);
    }
}

impl SliderState {
    fn begin_edit(&mut self, text: String, window: &mut Window, cx: &mut Context<Self>) {
        if self.input.is_none() {
            let input = cx.new(|cx| InputState::new(window, cx));
            let sub = cx.subscribe_in(&input, window, |s: &mut Self, input, e: &InputEvent, window, cx| {
                if !matches!(e, InputEvent::PressEnter { .. }) || !s.editing {
                    return;
                }
                s.editing = false;
                s.pending = parse_number(&input.read(cx).value());
                cx.notify();
                release_focus(window, cx);
            });
            self.input = Some(input);
            self._input_sub = Some(sub);
        }
        self.editing = true;
        if let Some(input) = self.input.clone() {
            input.update(cx, |i, cx| {
                i.set_value(text, window, cx);
                i.focus(window, cx);
                i.select_all(window, cx);
            });
        }
        cx.notify();
    }
}

#[derive(IntoElement)]
pub struct ParamSlider {
    id: SharedString,
    label: SharedString,
    value: f32,
    min: f32,
    max: f32,
    default: f32,
    origin: Option<f32>,
    decimals: usize,
    unit: &'static str,
    step: f32,
    scale: Scale,
    track: Track,
    disabled: bool,
    compact_label: bool,
    /// Why the engine does not render this parameter yet; shown as a small
    /// marker beside the label instead of lengthening the label.
    flag: Option<SharedString>,
    on_change: Option<OnChange>,
    on_commit: Option<OnCommit>,
}

impl ParamSlider {
    pub fn new(id: impl Into<SharedString>, label: impl Into<SharedString>, value: f32, min: f32, max: f32) -> Self {
        Self {
            id: id.into(),
            label: label.into(),
            value,
            min,
            max,
            default: if min < 0.0 && max > 0.0 { 0.0 } else { min },
            origin: None,
            decimals: 0,
            unit: "",
            step: (max - min) / 200.0,
            scale: Scale::Linear,
            track: Track::Plain,
            disabled: false,
            compact_label: false,
            flag: None,
            on_change: None,
            on_commit: None,
        }
    }

    pub fn range(mut self, min: f32, max: f32) -> Self {
        self.min = min;
        self.max = max;
        self
    }

    pub fn label(mut self, label: impl Into<SharedString>) -> Self {
        self.label = label.into();
        self
    }

    pub fn default_value(mut self, v: f32) -> Self {
        self.default = v;
        self
    }

    /// Where the fill starts; defaults to zero for bipolar ranges and `min` otherwise.
    pub fn origin(mut self, v: f32) -> Self {
        self.origin = Some(v);
        self
    }

    pub fn decimals(mut self, d: usize) -> Self {
        self.decimals = d;
        self
    }

    pub fn unit(mut self, unit: &'static str) -> Self {
        self.unit = unit;
        self
    }

    pub fn step(mut self, step: f32) -> Self {
        self.step = step;
        self
    }

    pub fn scale(mut self, scale: Scale) -> Self {
        self.scale = scale;
        self
    }

    pub fn track(mut self, track: Track) -> Self {
        self.track = track;
        self
    }

    pub fn disabled(mut self, disabled: bool) -> Self {
        self.disabled = disabled;
        self
    }

    /// Marks the parameter as stored but not rendered yet, with the reason.
    pub fn flag(mut self, status: crate::contract::Status) -> Self {
        self.flag = status.note().or(status.chip()).map(SharedString::from);
        self
    }

    pub fn compact_label(mut self) -> Self {
        self.compact_label = true;
        self
    }

    pub fn on_change(mut self, f: impl Fn(f32, &mut Window, &mut App) + 'static) -> Self {
        self.on_change = Some(Rc::new(f));
        self
    }

    pub fn on_commit(mut self, f: impl Fn(&mut Window, &mut App) + 'static) -> Self {
        self.on_commit = Some(Rc::new(f));
        self
    }

    fn format(value: f32, decimals: usize, unit: &str) -> String {
        // Avoid "-0" for values that round to zero.
        let v = if value.abs() < 0.5 * 10f32.powi(-(decimals as i32)) { 0.0 } else { value };
        format!("{v:.decimals$}{unit}")
    }
}

/// Builds a slider bound to the current image's adjustments.
pub fn param(
    state: &Entity<AppState>,
    id: impl Into<SharedString>,
    label: impl Into<SharedString>,
    binding: Binding,
    min: f32,
    max: f32,
    cx: &App,
) -> ParamSlider {
    let label: SharedString = label.into();
    let value = (binding.get)(state.read(cx).adj());
    let set = binding.set.clone();
    let change_state = state.clone();
    let commit_state = state.clone();
    let undo_label = label.clone();
    ParamSlider::new(id, label, value, min, max)
        .disabled(state.read(cx).current.is_none())
        .on_change(move |v, _, cx| {
            let set = set.clone();
            let undo_label = undo_label.clone();
            change_state.update(cx, |s, cx| s.edit_live(undo_label, move |a| set(a, v), cx));
        })
        .on_commit(move |_, cx| commit_state.update(cx, |s, cx| s.end_gesture(cx)))
}

impl RenderOnce for ParamSlider {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let p = pal(cx).clone();
        let id = self.id.clone();
        let st: Entity<SliderState> = window.use_keyed_state(motion::key(&id, "state"), cx, |_, _| SliderState::default());
        // Clicking away from a number being typed applies it, like Enter. (Esc
        // clears `editing` first, so it never gets here.)
        let left = {
            let s = st.read(cx);
            s.editing && s.input.as_ref().is_some_and(|i| !gpui_kit::Focusable::focus_handle(i.read(cx), cx).is_focused(window))
        };
        if left {
            st.update(cx, |s, cx| {
                s.editing = false;
                s.pending = s.input.as_ref().and_then(|i| parse_number(&i.read(cx).value()));
            });
        }
        let (hovered, dragging, editing, typed) = {
            let s = st.read(cx);
            (s.hovered, s.dragging, s.editing, s.pending)
        };
        let field = st.read(cx).input.clone().filter(|_| editing);
        let (min, max, scale) = (self.min, self.max, self.scale);
        let t_target = scale.to_t(self.value, min, max);
        let t = motion::follow(motion::key(&id, "t"), t_target, motion::GLIDE, !dragging, window, cx);
        let origin = self.origin.unwrap_or(if min < 0.0 && max > 0.0 { 0.0 } else { min });
        let t_origin = scale.to_t(origin, min, max);
        let knob = motion::value(
            motion::key(&id, "knob"),
            if crate::theme::touch() { if dragging { 22.0 } else { 18.0 } } else if dragging { 13.0 } else if hovered { 12.0 } else { 10.0 },
            motion::QUICK,
            window,
            cx,
        );
        let emphasis = motion::toggle(motion::key(&id, "emph"), hovered || dragging, motion::QUICK, window, cx);
        let changed = (self.value - self.default).abs() > 1e-6;

        let on_change = self.on_change.clone();
        let on_commit = self.on_commit.clone();
        let default = self.default;

        let set_from_x = {
            let on_change = on_change.clone();
            move |x: Pixels, bounds: Bounds<Pixels>, window: &mut Window, cx: &mut App| {
                let w = f32::from(bounds.size.width).max(1.0);
                let t = (f32::from(x - bounds.origin.x) / w).clamp(0.0, 1.0);
                if let Some(f) = &on_change {
                    f(scale.to_value(t, min, max), window, cx);
                }
            }
        };

        let reset = {
            let on_change = on_change.clone();
            let on_commit = on_commit.clone();
            move |window: &mut Window, cx: &mut App| {
                if let Some(f) = &on_change {
                    f(default, window, cx);
                }
                if let Some(f) = &on_commit {
                    f(window, cx);
                }
            }
        };

        // A number typed into the value field: apply it once this frame is done.
        if let Some(v) = typed {
            st.update(cx, |s, _| s.pending = None);
            let (on_change, on_commit) = (on_change.clone(), on_commit.clone());
            let v = v.clamp(min, max);
            if (v - self.value).abs() > 1e-6 {
                window.defer(cx, move |window, cx| {
                    if let Some(f) = &on_change {
                        f(v, window, cx);
                    }
                    if let Some(f) = &on_commit {
                        f(window, cx);
                    }
                });
            }
        }

        let fill_color = if self.disabled { p.text_disabled } else { p.accent.opacity(0.75 + 0.25 * emphasis) };
        let (fill_l, fill_r) = (t.min(t_origin), t.max(t_origin));

        let track_bg = match &self.track {
            Track::Plain => div().size_full().rounded_full().bg(p.well).into_any_element(),
            Track::Gradient(stops) => {
                let n = stops.len().max(2) - 1;
                gpui_kit::component::h_flex()
                    .size_full()
                    .rounded_full()
                    .overflow_hidden()
                    .opacity(0.85)
                    .children((0..n).map(|i| {
                        div().h_full().flex_1().bg(linear_gradient(
                            90.0,
                            linear_color_stop(stops[i], 0.0),
                            linear_color_stop(stops[(i + 1).min(stops.len() - 1)], 1.0),
                        ))
                    }))
                    .into_any_element()
            }
        };
        let plain = matches!(self.track, Track::Plain);

        let track = div()
            .id(SharedString::from(format!("{id}-track")))
            .relative()
            .flex_1()
            .h_full()
            .flex()
            .items_center()
            .when(!self.disabled, |this| this.cursor_ew_resize())
            .child(
                div()
                    .relative()
                    .w_full()
                    .h(px(if plain { 3.0 } else { 4.0 } + emphasis))
                    .child(track_bg)
                    .when(plain, |this| {
                        this.child(
                            div()
                                .absolute()
                                .top_0()
                                .bottom_0()
                                .left(relative(fill_l))
                                .w(relative((fill_r - fill_l).max(0.0)))
                                .rounded_full()
                                .bg(fill_color),
                        )
                    })
                    // Default tick.
                    .when(t_origin > 0.001 && t_origin < 0.999, |this| {
                        this.child(
                            div()
                                .absolute()
                                .top(px(-2.0))
                                .bottom(px(-2.0))
                                .left(relative(t_origin))
                                .w(px(1.0))
                                .bg(p.text_subtle.opacity(0.6)),
                        )
                    })
                    .child(
                        div()
                            .absolute()
                            .top(px(-(knob - 3.0 - emphasis) / 2.0))
                            .left(relative(t))
                            .ml(px(-knob / 2.0))
                            .size(px(knob))
                            .rounded_full()
                            .bg(if self.disabled { p.text_disabled } else { p.thumb_knob })
                            .border_1()
                            .border_color(hexa(0x000000, 0.35))
                            .shadow_sm(),
                    ),
            )
            .on_prepaint({
                let st = st.clone();
                move |bounds, _, cx| {
                    if st.read(cx).bounds != bounds {
                        st.update(cx, |s, _| s.bounds = bounds);
                    }
                }
            })
            .when(!self.disabled, |this| {
                this.on_mouse_down(MouseButton::Left, {
                    let st = st.clone();
                    let set_from_x = set_from_x.clone();
                    let reset = reset.clone();
                    move |e: &MouseDownEvent, window, cx| {
                        cx.stop_propagation();
                        if e.click_count >= 2 {
                            reset(window, cx);
                            return;
                        }
                        let bounds = st.read(cx).bounds;
                        let fine = e.modifiers.shift || e.modifiers.alt;
                        st.update(cx, |s, cx| {
                            s.dragging = true;
                            s.anchor = None;
                            cx.notify();
                        });
                        if !fine {
                            set_from_x(e.position.x, bounds, window, cx);
                        }
                    }
                })
                .on_drag(SliderDrag { id: id.clone() }, |_, _, _, cx| cx.new(|_| DragGhost))
                .on_drag_move::<SliderDrag>({
                    let st = st.clone();
                    let id = id.clone();
                    let on_change = on_change.clone();
                    let current = self.value;
                    move |e, window, cx| {
                        if e.drag(cx).id != id {
                            return;
                        }
                        let bounds = st.read(cx).bounds;
                        let fine = e.event.modifiers.shift || e.event.modifiers.alt;
                        let x = f32::from(e.event.position.x);
                        if fine {
                            let anchor = st.read(cx).anchor;
                            let (ax, av) = match anchor {
                                Some(a) => a,
                                None => {
                                    st.update(cx, |s, _| s.anchor = Some((x, current)));
                                    (x, current)
                                }
                            };
                            let w = f32::from(bounds.size.width).max(1.0);
                            let t0 = scale.to_t(av, min, max);
                            let t = (t0 + (x - ax) / w / 10.0).clamp(0.0, 1.0);
                            if let Some(f) = &on_change {
                                f(scale.to_value(t, min, max), window, cx);
                            }
                        } else {
                            if st.read(cx).anchor.is_some() {
                                st.update(cx, |s, _| s.anchor = None);
                            }
                            set_from_x(e.event.position.x, bounds, window, cx);
                        }
                    }
                })
                .on_mouse_up(MouseButton::Left, {
                    let st = st.clone();
                    let on_commit = on_commit.clone();
                    move |_, window, cx| {
                        st.update(cx, |s, cx| {
                            s.dragging = false;
                            s.anchor = None;
                            cx.notify();
                        });
                        if let Some(f) = &on_commit {
                            f(window, cx);
                        }
                    }
                })
                .on_mouse_up_out(MouseButton::Left, {
                    let st = st.clone();
                    let on_commit = on_commit.clone();
                    move |_, window, cx| {
                        if st.read(cx).dragging {
                            st.update(cx, |s, cx| {
                                s.dragging = false;
                                s.anchor = None;
                                cx.notify();
                            });
                            if let Some(f) = &on_commit {
                                f(window, cx);
                            }
                        }
                    }
                })
            });

        let label_w = if self.compact_label { 64.0 } else { metrics::LABEL_W };
        let value_text = Self::format(self.value, self.decimals, self.unit);
        let step = self.step;
        let value = self.value;

        div()
            .id(id.clone())
            .h(px(metrics::row()))
            .w_full()
            .flex()
            .items_center()
            .gap_2()
            .px_3()
            .on_hover({
                let st = st.clone();
                move |h, _, cx| {
                    let h = *h;
                    st.update(cx, |s, cx| {
                        if s.hovered != h {
                            s.hovered = h;
                            cx.notify();
                        }
                    })
                }
            })
            .when(!self.disabled, |this| {
                this.on_scroll({
                    let on_change = on_change.clone();
                    let on_commit = on_commit.clone();
                    move |e: &ScrollEvent, window, cx| {
                        if !(e.modifiers.control || e.modifiers.platform) {
                            return;
                        }
                        let dy = match e.delta {
                            ScrollDelta::Lines(d) => d.y,
                            ScrollDelta::Pixels(d) => f32::from(d.y) / 20.0,
                        };
                        if dy.abs() < 0.01 {
                            return;
                        }
                        cx.stop_propagation();
                        let mult = if e.modifiers.shift { 10.0 } else { 1.0 };
                        let next = (value + dy.signum() * step * mult).clamp(min, max);
                        if let Some(f) = &on_change {
                            f(next, window, cx);
                        }
                        if let Some(f) = &on_commit {
                            f(window, cx);
                        }
                    }
                })
            })
            .child(
                div()
                    .id(SharedString::from(format!("{id}-label")))
                    .w(px(label_w))
                    .flex_shrink_0()
                    .flex()
                    .items_center()
                    .gap_1()
                    .text_xs()
                    .text_color(if self.disabled {
                        p.text_disabled
                    } else if changed {
                        p.text
                    } else {
                        p.text_muted
                    })
                    .child(div().min_w_0().overflow_hidden().whitespace_nowrap().text_ellipsis().child(self.label.clone()))
                    .when_some(self.flag.clone(), |this, note| {
                        this.child(
                            div()
                                .id(SharedString::from(format!("{id}-flag")))
                                .flex_shrink_0()
                                .size(px(6.0))
                                .rounded_full()
                                .border_1()
                                .border_color(p.warning)
                                .tooltip(move |window, cx| gpui_kit::component::tooltip::Tooltip::new(note.clone()).build(window, cx)),
                        )
                    })
                    .when(!self.disabled, |this| {
                        this.on_click({
                            let reset = reset.clone();
                            move |e: &ClickEvent, window, cx| {
                                if e.click_count() >= 2 {
                                    reset(window, cx);
                                }
                            }
                        })
                    }),
            )
            .child(track)
            .child(match field {
                Some(input) => div()
                    .w(px(metrics::VALUE_W))
                    .flex_shrink_0()
                    .h(px(20.0))
                    .rounded(px(3.0))
                    .bg(p.well)
                    .border_1()
                    .border_color(p.accent)
                    .overflow_hidden()
                    .text_xs()
                    .capture_key_down({
                        let st = st.clone();
                        move |e: &gpui_kit::KeyDownEvent, window, cx| {
                            if e.keystroke.key == "escape" {
                                cx.stop_propagation();
                                st.update(cx, |s, cx| {
                                    s.editing = false;
                                    cx.notify();
                                });
                                release_focus(window, cx);
                            }
                        }
                    })
                    .child(Input::new(&input).xsmall().appearance(false))
                    .into_any_element(),
                None => div()
                    .id(SharedString::from(format!("{id}-value")))
                    .w(px(metrics::VALUE_W))
                    .flex_shrink_0()
                    .h(px(18.0))
                    .px_1()
                    .flex()
                    .items_center()
                    .justify_end()
                    .rounded(px(3.0))
                    .bg(p.well.opacity(0.6 + 0.4 * emphasis))
                    .text_xs()
                    .text_color(if changed { p.text } else { p.text_muted })
                    .when(!self.disabled, |this| {
                        let st = st.clone();
                        let text = format!("{:.*}", self.decimals, self.value);
                        this.cursor_text().hover(|s| s.bg(p.hover)).on_click(move |_, window, cx| {
                            cx.stop_propagation();
                            let text = text.clone();
                            st.update(cx, |s, cx| s.begin_edit(text, window, cx));
                        })
                    })
                    .child(value_text)
                    .into_any_element(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::parse_number;

    #[test]
    fn typed_numbers_tolerate_units_and_signs() {
        assert_eq!(parse_number("0.35"), Some(0.35));
        assert_eq!(parse_number(" +1,5 EV"), Some(1.5));
        assert_eq!(parse_number("-20"), Some(-20.0));
        assert_eq!(parse_number("5600 K"), Some(5600.0));
        assert_eq!(parse_number(""), None);
        assert_eq!(parse_number("abc"), None);
    }
}
