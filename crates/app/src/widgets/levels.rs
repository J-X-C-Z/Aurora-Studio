//! Levels: input black / midtone / white handles over the histogram, and output
//! black / white handles below. The midtone handle sits at `0.5^γ` between the
//! black and white points, so dragging it right darkens the image as in Capture One.

use std::rc::Rc;

use gpui_kit::{
    App, Entity, IntoElement, ParentElement, RenderOnce, SharedString, Styled, Window, canvas, div, linear_color_stop,
    linear_gradient, px,
};
use lumen_engine::LevelsChannel;

use super::paint;
use super::surface::{Pointer, Surface};
use crate::motion;
use crate::theme::{hexa, pal};

type OnChange = Rc<dyn Fn(LevelsChannel, &mut Window, &mut App)>;
type OnCommit = Rc<dyn Fn(&mut Window, &mut App)>;

#[derive(Clone, Copy, PartialEq, Default)]
enum Handle {
    #[default]
    None,
    Black,
    Mid,
    White,
    OutBlack,
    OutWhite,
}

#[derive(IntoElement)]
pub struct LevelsView {
    id: SharedString,
    value: LevelsChannel,
    histogram: Vec<f32>,
    tint: gpui_kit::Hsla,
    on_change: Option<OnChange>,
    on_commit: Option<OnCommit>,
}

impl LevelsView {
    pub fn new(id: impl Into<SharedString>, value: LevelsChannel, histogram: Vec<f32>, tint: gpui_kit::Hsla) -> Self {
        Self { id: id.into(), value, histogram, tint, on_change: None, on_commit: None }
    }

    pub fn on_change(mut self, f: impl Fn(LevelsChannel, &mut Window, &mut App) + 'static) -> Self {
        self.on_change = Some(Rc::new(f));
        self
    }

    pub fn on_commit(mut self, f: impl Fn(&mut Window, &mut App) + 'static) -> Self {
        self.on_commit = Some(Rc::new(f));
        self
    }
}

fn mid_pos(v: &LevelsChannel) -> f32 {
    v.input_black + (v.input_white - v.input_black) * 0.5f32.powf(v.gamma)
}

fn handle_marker(x: f32, fill: gpui_kit::Hsla, outline: gpui_kit::Hsla) -> impl IntoElement + use<> {
    div()
        .absolute()
        .top(px(0.0))
        .left(gpui_kit::relative(x))
        .ml(px(-5.0))
        .w(px(10.0))
        .h(px(10.0))
        .child(
            canvas(
                |_, _, _| (),
                move |b, _, window, _| {
                    let top = gpui_kit::point(b.origin.x + b.size.width / 2.0, b.origin.y);
                    let l = gpui_kit::point(b.origin.x, b.origin.y + b.size.height);
                    let r = gpui_kit::point(b.origin.x + b.size.width, b.origin.y + b.size.height);
                    paint::polygon(window, &[top, r, l], outline);
                    let inset = px(1.6);
                    let top = gpui_kit::point(top.x, top.y + inset * 1.6);
                    let l = gpui_kit::point(l.x + inset * 1.4, l.y - inset);
                    let r = gpui_kit::point(r.x - inset * 1.4, r.y - inset);
                    paint::polygon(window, &[top, r, l], fill);
                },
            )
            .size_full(),
        )
}

impl RenderOnce for LevelsView {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let p = pal(cx).clone();
        let v = self.value.clone();
        let active: Entity<Handle> = window.use_keyed_state(motion::key(&self.id, "handle"), cx, |_, _| Handle::None);
        // Switching channel or a new preview frame reshapes the histogram in place.
        let hist = motion::morph(motion::key(&self.id, "hist"), self.histogram.clone(), window, cx);
        let tint = self.tint;
        // Handles glide on reset, undo, auto and channel switches, and follow
        // the pointer exactly while one of them is held.
        let held = *active.read(cx) != Handle::None
            || window.use_keyed_state(motion::key(&self.id, "out-handle"), cx, |_, _| Handle::None).read(cx) != &Handle::None;
        let marks = motion::morph_follow(
            motion::key(&self.id, "marks"),
            vec![v.input_black, mid_pos(&v), v.input_white, v.output_black, v.output_white],
            !held,
            window,
            cx,
        );

        let pick_input = move |ptr: &Pointer, v: &LevelsChannel| {
            let x = ptr.pos.0;
            [(Handle::Black, v.input_black), (Handle::Mid, mid_pos(v)), (Handle::White, v.input_white)]
                .into_iter()
                .min_by(|a, b| (a.1 - x).abs().partial_cmp(&(b.1 - x).abs()).unwrap_or(std::cmp::Ordering::Equal))
                .map(|h| h.0)
                .unwrap_or(Handle::None)
        };

        let apply = move |h: Handle, x: f32, v: &LevelsChannel| -> LevelsChannel {
            let mut n = v.clone();
            let x = x.clamp(0.0, 1.0);
            match h {
                Handle::Black => n.input_black = x.min(v.input_white - 0.02),
                Handle::White => n.input_white = x.max(v.input_black + 0.02),
                Handle::Mid => {
                    let span = (v.input_white - v.input_black).max(0.02);
                    let m = ((x - v.input_black) / span).clamp(0.02, 0.98);
                    n.gamma = (m.ln() / 0.5f32.ln()).clamp(0.1, 9.99);
                }
                Handle::OutBlack => n.output_black = x.min(v.output_white - 0.02),
                Handle::OutWhite => n.output_white = x.max(v.output_black + 0.02),
                Handle::None => {}
            }
            n
        };

        let input_surface = {
            let id = SharedString::from(format!("{}-in", self.id));
            let (f, c, active2, active3) = (self.on_change.clone(), self.on_commit.clone(), active.clone(), active.clone());
            let v1 = v.clone();
            let v2 = v.clone();
            Surface::new(id)
                .w_full()
                .h(px(86.0))
                .rounded(px(4.0))
                .bg(p.well)
                .overflow_hidden()
                .cursor_ew_resize()
                .on_down({
                    let f = f.clone();
                    let c = c.clone();
                    move |ptr, window, cx| {
                        let h = pick_input(&ptr, &v1);
                        if ptr.clicks >= 2 {
                            let mut n = v1.clone();
                            match h {
                                Handle::Black => n.input_black = 0.0,
                                Handle::White => n.input_white = 1.0,
                                _ => n.gamma = 1.0,
                            }
                            if let Some(f) = &f {
                                f(n, window, cx);
                            }
                            if let Some(c) = &c {
                                c(window, cx);
                            }
                            return;
                        }
                        active.update(cx, |a, _| *a = h);
                    }
                })
                .on_drag({
                    let f = f.clone();
                    move |ptr, window, cx| {
                        let h = *active2.read(cx);
                        if let Some(f) = &f {
                            f(apply(h, ptr.pos.0, &v2), window, cx);
                        }
                    }
                })
                .on_up(move |window, cx| {
                    active3.update(cx, |a, _| *a = Handle::None);
                    if let Some(c) = &c {
                        c(window, cx);
                    }
                })
                .child(
                    canvas(
                        |_, _, _| (),
                        move |bounds, _, window, _| {
                            if !hist.is_empty() {
                                paint::fill_area(window, bounds, &hist, tint.opacity(0.45));
                            }
                        },
                    )
                    .absolute()
                    .size_full(),
                )
        };

        let output_surface = {
            let id = SharedString::from(format!("{}-out", self.id));
            let f = self.on_change.clone();
            let c = self.on_commit.clone();
            let v1 = v.clone();
            let v2 = v.clone();
            let active = window.use_keyed_state(motion::key(&self.id, "out-handle"), cx, |_, _| Handle::None);
            let (a1, a2, a3) = (active.clone(), active.clone(), active);
            Surface::new(id)
                .w_full()
                .h(px(20.0))
                .cursor_ew_resize()
                .on_down(move |ptr, _, cx| {
                    let h = if (ptr.pos.0 - v1.output_black).abs() < (ptr.pos.0 - v1.output_white).abs() {
                        Handle::OutBlack
                    } else {
                        Handle::OutWhite
                    };
                    a1.update(cx, |a, _| *a = h);
                })
                .on_drag(move |ptr, window, cx| {
                    let h = *a2.read(cx);
                    if let Some(f) = &f {
                        f(apply(h, ptr.pos.0, &v2), window, cx);
                    }
                })
                .on_up(move |window, cx| {
                    a3.update(cx, |a, _| *a = Handle::None);
                    if let Some(c) = &c {
                        c(window, cx);
                    }
                })
                .child(
                    div()
                        .absolute()
                        .top(px(0.0))
                        .left_0()
                        .right_0()
                        .h(px(6.0))
                        .rounded(px(2.0))
                        .bg(linear_gradient(90.0, linear_color_stop(hexa(0x000000, 1.0), 0.0), linear_color_stop(hexa(0xffffff, 1.0), 1.0))),
                )
                .child(div().absolute().top(px(8.0)).left_0().right_0().h(px(10.0)).child(handle_marker(marks[3], hexa(0x000000, 1.0), p.text_muted)).child(handle_marker(marks[4], hexa(0xffffff, 1.0), p.text_muted)))
        };

        gpui_kit::component::v_flex()
            .px_3()
            .gap_1()
            .child(input_surface)
            .child(
                div()
                    .relative()
                    .h(px(10.0))
                    .w_full()
                    .child(handle_marker(marks[0], hexa(0x000000, 1.0), p.text_muted))
                    .child(handle_marker(marks[1], hexa(0x808080, 1.0), p.text_muted))
                    .child(handle_marker(marks[2], hexa(0xffffff, 1.0), p.text_muted)),
            )
            .child(div().h(px(4.0)))
            .child(output_surface)
            .child(
                gpui_kit::component::h_flex()
                    .justify_between()
                    .text_xs()
                    .text_color(p.text_subtle)
                    .child(format!("{:.0}", v.input_black * 255.0))
                    .child(format!("γ {:.2}", v.gamma))
                    .child(format!("{:.0}", v.input_white * 255.0)),
            )
    }
}
