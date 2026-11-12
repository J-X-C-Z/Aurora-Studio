//! Tone curve editor. Click to add a point, drag to move it, double-click or drag
//! it out of the box to remove it. The drawn curve uses the same interpolation as
//! the engine contract ([`lumen_engine::Curve::eval`]).

use std::rc::Rc;

use gpui_kit::{
    App, Entity, Hsla, IntoElement, ParentElement, RenderOnce, SharedString, Styled, Window, canvas, div,
    prelude::FluentBuilder as _, px,
};
use lumen_engine::{Curve, CurvePoint};

use super::paint;
use super::surface::{Pointer, Surface, SurfaceState};
use crate::motion;
use crate::theme::{hexa, pal};

type OnChange = Rc<dyn Fn(Curve, &mut Window, &mut App)>;
type OnCommit = Rc<dyn Fn(&mut Window, &mut App)>;

#[derive(Default)]
struct DragState {
    index: Option<usize>,
    removing: bool,
}

#[derive(IntoElement)]
pub struct CurveEditor {
    id: SharedString,
    curve: Curve,
    color: Hsla,
    histogram: Vec<f32>,
    on_change: Option<OnChange>,
    on_commit: Option<OnCommit>,
}

impl CurveEditor {
    pub fn new(id: impl Into<SharedString>, curve: Curve, color: Hsla) -> Self {
        Self { id: id.into(), curve, color, histogram: Vec::new(), on_change: None, on_commit: None }
    }

    pub fn histogram(mut self, normalized: Vec<f32>) -> Self {
        self.histogram = normalized;
        self
    }

    pub fn on_change(mut self, f: impl Fn(Curve, &mut Window, &mut App) + 'static) -> Self {
        self.on_change = Some(Rc::new(f));
        self
    }

    pub fn on_commit(mut self, f: impl Fn(&mut Window, &mut App) + 'static) -> Self {
        self.on_commit = Some(Rc::new(f));
        self
    }
}

const HIT: f32 = 9.0;

fn nearest(curve: &Curve, p: &Pointer) -> Option<usize> {
    let w = f32::from(p.bounds.size.width);
    let h = f32::from(p.bounds.size.height);
    curve
        .points
        .iter()
        .enumerate()
        .map(|(i, cp)| {
            let dx = (cp.x - p.pos.0) * w;
            let dy = (1.0 - cp.y - p.pos.1) * h;
            (i, (dx * dx + dy * dy).sqrt())
        })
        .filter(|(_, d)| *d < HIT)
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
}

impl RenderOnce for CurveEditor {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let p = pal(cx).clone();
        let drag: Entity<DragState> = window.use_keyed_state(motion::key(&self.id, "drag"), cx, |_, _| DragState::default());
        let surface_state: Entity<SurfaceState> = Surface::state(&self.id, window, cx);
        let hover = surface_state.read(cx).hover;
        let dragging = drag.read(cx).index;
        let removing = drag.read(cx).removing;

        let curve = self.curve.clone();
        let color = self.color;
        // Keyed apart from the editor's id (which names the channel), so the
        // backdrop morphs from one channel's histogram to the next.
        let hist = motion::morph(motion::key("curve-editor", "hist"), self.histogram.clone(), window, cx);
        let edge = p.edge;
        let point_fill = p.panel;
        // Reset, undo, auto and presets glide the curve to its new shape; a
        // point being dragged follows the pointer exactly.
        let travel = dragging.is_none();
        let samples = motion::morph_follow(
            motion::key("curve-editor", "line"),
            (0..=96).map(|i| curve.eval(i as f32 / 96.0)).collect(),
            travel,
            window,
            cx,
        );
        let points = motion::morph_follow(
            motion::key("curve-editor", "points"),
            curve.points.iter().flat_map(|q| [q.x, q.y]).collect(),
            travel,
            window,
            cx,
        );

        let readout = hover.map(|(x, _)| {
            let x = x.clamp(0.0, 1.0);
            format!("{:.0} → {:.0}", x * 255.0, curve.eval(x) * 255.0)
        });

        let on_down = {
            let drag = drag.clone();
            let on_change = self.on_change.clone();
            let on_commit = self.on_commit.clone();
            let curve = self.curve.clone();
            move |ptr: Pointer, window: &mut Window, cx: &mut App| {
                let mut c = curve.clone();
                let hit = nearest(&c, &ptr);
                if ptr.clicks >= 2 {
                    if let Some(i) = hit
                        && i != 0
                        && i != c.points.len() - 1
                    {
                        c.points.remove(i);
                        if let Some(f) = &on_change {
                            f(c, window, cx);
                        }
                        if let Some(f) = &on_commit {
                            f(window, cx);
                        }
                    }
                    drag.update(cx, |d, _| d.index = None);
                    return;
                }
                let index = match hit {
                    Some(i) => i,
                    None => {
                        let x = ptr.pos.0.clamp(0.001, 0.999);
                        let y = (1.0 - ptr.pos.1).clamp(0.0, 1.0);
                        let at = c.points.iter().position(|q| q.x > x).unwrap_or(c.points.len());
                        c.points.insert(at, CurvePoint::new(x, y));
                        if let Some(f) = &on_change {
                            f(c, window, cx);
                        }
                        at
                    }
                };
                drag.update(cx, |d, cx| {
                    d.index = Some(index);
                    d.removing = false;
                    cx.notify();
                });
            }
        };

        let on_move = {
            let drag = drag.clone();
            let on_change = self.on_change.clone();
            let curve = self.curve.clone();
            move |ptr: Pointer, window: &mut Window, cx: &mut App| {
                let Some(i) = drag.read(cx).index else { return };
                let mut c = curve.clone();
                if i >= c.points.len() {
                    return;
                }
                let last = c.points.len() - 1;
                let lo = if i == 0 { 0.0 } else { c.points[i - 1].x + 0.01 };
                let hi = if i == last { 1.0 } else { c.points[i + 1].x - 0.01 };
                let outside = ptr.pos.1 < -0.12 || ptr.pos.1 > 1.12 || ptr.pos.0 < -0.12 || ptr.pos.0 > 1.12;
                let removable = i != 0 && i != last;
                drag.update(cx, |d, cx| {
                    if d.removing != (outside && removable) {
                        d.removing = outside && removable;
                        cx.notify();
                    }
                });
                c.points[i] = CurvePoint::new(ptr.pos.0.clamp(lo, hi.max(lo)), (1.0 - ptr.pos.1).clamp(0.0, 1.0));
                if let Some(f) = &on_change {
                    f(c, window, cx);
                }
            }
        };

        let on_up = {
            let drag = drag.clone();
            let on_change = self.on_change.clone();
            let on_commit = self.on_commit.clone();
            let curve = self.curve.clone();
            move |window: &mut Window, cx: &mut App| {
                let (index, removing) = {
                    let d = drag.read(cx);
                    (d.index, d.removing)
                };
                if let (Some(i), true) = (index, removing) {
                    let mut c = curve.clone();
                    if i < c.points.len() {
                        c.points.remove(i);
                        if let Some(f) = &on_change {
                            f(c, window, cx);
                        }
                    }
                }
                drag.update(cx, |d, cx| {
                    d.index = None;
                    d.removing = false;
                    cx.notify();
                });
                if let Some(f) = &on_commit {
                    f(window, cx);
                }
            }
        };

        Surface::new(self.id.clone())
            .w_full()
            .h(px(196.0))
            .rounded(px(4.0))
            .bg(p.well)
            .overflow_hidden()
            .cursor_crosshair()
            .on_down(on_down)
            .on_drag(on_move)
            .on_up(on_up)
            .child(
                canvas(
                    |_, _, _| (),
                    move |bounds, _, window, _| {
                        let (x0, y0) = (bounds.origin.x, bounds.origin.y);
                        let (w, h) = (bounds.size.width, bounds.size.height);
                        for i in 1..4 {
                            let f = i as f32 / 4.0;
                            paint::segment(window, gpui_kit::point(x0 + w * f, y0), gpui_kit::point(x0 + w * f, y0 + h), 1.0, edge);
                            paint::segment(window, gpui_kit::point(x0, y0 + h * f), gpui_kit::point(x0 + w, y0 + h * f), 1.0, edge);
                        }
                        if !hist.is_empty() {
                            paint::fill_area(window, bounds, &hist, hexa(0xffffff, 0.06));
                        }
                        paint::stroke_line(window, bounds, &[(0.0, 0.0), (1.0, 1.0)], 1.0, hexa(0xffffff, 0.12));
                        let line: Vec<(f32, f32)> = samples.iter().enumerate().map(|(i, y)| (i as f32 / 96.0, *y)).collect();
                        paint::stroke_line(window, bounds, &line, 1.8, color);
                        for (i, cp) in points.chunks_exact(2).enumerate() {
                            let c = gpui_kit::point(x0 + w * cp[0], y0 + h * (1.0 - cp[1]));
                            let active = dragging == Some(i);
                            let r = if active { 5.5 } else { 4.0 };
                            let ring = if active && removing { hexa(0xe5605a, 1.0) } else { color };
                            let square = [
                                gpui_kit::point(c.x - px(r), c.y - px(r)),
                                gpui_kit::point(c.x + px(r), c.y - px(r)),
                                gpui_kit::point(c.x + px(r), c.y + px(r)),
                                gpui_kit::point(c.x - px(r), c.y + px(r)),
                            ];
                            paint::polygon(window, &square, ring);
                            let inner = r - 1.5;
                            let square = [
                                gpui_kit::point(c.x - px(inner), c.y - px(inner)),
                                gpui_kit::point(c.x + px(inner), c.y - px(inner)),
                                gpui_kit::point(c.x + px(inner), c.y + px(inner)),
                                gpui_kit::point(c.x - px(inner), c.y + px(inner)),
                            ];
                            paint::polygon(window, &square, if active { ring } else { point_fill });
                        }
                    },
                )
                .absolute()
                .size_full(),
            )
            .when_some(readout, |this, text| {
                this.child(
                    div()
                        .absolute()
                        .top(px(4.0))
                        .left(px(6.0))
                        .text_xs()
                        .text_color(p.text_muted)
                        .child(text),
                )
            })
    }
}
