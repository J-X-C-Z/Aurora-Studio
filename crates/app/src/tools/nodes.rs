//! The node editor, after DaVinci Resolve's: the image flows from the input on
//! the left through corrector nodes to the output on the right. Each corrector
//! is a layer with its own grade and mask; parallel and layer mixers join
//! branches. Drag a node to move it, drag from an output to an input to wire
//! it, drag a wire off an input to cut it, drag the background to pan and
//! Ctrl+scroll to zoom. Selecting a node points every grading tab at it.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Disableable as _, Icon, Sizable as _,
    button::{Button, ButtonVariants as _},
    h_flex,
    menu::{ContextMenuExt as _, DropdownMenu as _, PopupMenuItem},
    tooltip::Tooltip,
};
use gpui_kit::{
    AnyElement, App, Entity, Hsla, InteractiveElement, IntoElement, ParentElement, PathBuilder, Pixels, Point, ScrollDelta, SharedString,
    StatefulInteractiveElement, Styled, Window, canvas, div, point, prelude::FluentBuilder as _, px,
};
use lumen_engine::{GraphNode, LayerKind, NodeGraph, NodeKind};

use crate::actions;
use crate::i18n::t;
use crate::motion;
use crate::state::{AppState, GraphExt as _, NewNode};
use crate::theme::{hexa, pal};
use crate::widgets::surface::{Pointer, Surface};

const INPUT: u64 = NodeGraph::INPUT;
const OUTPUT: u64 = NodeGraph::OUTPUT;
/// Spacing of a mixer's inputs, in editor units.
const PORT_GAP: f32 = 15.0;
/// How close the pointer must be to a port to grab it, in pixels.
const PORT_HIT: f32 = 9.0;

/// What a drag in the editor is doing.
#[derive(Clone, Copy, Debug)]
enum Drag {
    Move { id: u64, grab: [f32; 2], moved: bool },
    /// A wire from `from`'s output to the pointer; `cut` is the input it was
    /// pulled from, which loses its wire unless it is dropped back.
    Wire { from: u64, at: [f32; 2], cut: Option<(u64, u32)> },
    /// A wire from an empty input, looking for an output.
    WireBack { to: u64, input: u32, at: [f32; 2] },
    Pan { from: [f32; 2], start: [f32; 2] },
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Hit {
    Out(u64),
    In(u64, u32),
    Node(u64),
}

#[derive(Default)]
struct View {
    pan: [f32; 2],
    zoom: f32,
    drag: Option<Drag>,
    hover: Option<Hit>,
    /// The editor's bounds at the last layout, in window pixels.
    bounds: gpui_kit::Bounds<Pixels>,
    /// The image and graph shape last fitted, so a new node or image refits.
    fitted: Option<(u64, usize, usize, u32)>,
}

/// A node's box in editor units.
#[derive(Clone, Copy, Debug)]
struct Shape {
    c: [f32; 2],
    w: f32,
    h: f32,
    inputs: u32,
    outputs: bool,
}

impl Shape {
    fn of(g: &NodeGraph, n: &GraphNode, c: [f32; 2]) -> Self {
        let inputs = g.input_count(n.id);
        let (w, h) = match n.kind {
            NodeKind::Layer => (78.0, 44.0),
            NodeKind::Input | NodeKind::Output => (54.0, 26.0),
            NodeKind::Parallel | NodeKind::LayerMixer => (28.0, (inputs as f32 * PORT_GAP + 4.0).max(30.0)),
        };
        Self { c, w, h, inputs, outputs: n.kind != NodeKind::Output }
    }

    fn input(&self, i: u32) -> [f32; 2] {
        let off = (i as f32 - (self.inputs as f32 - 1.0) / 2.0) * PORT_GAP;
        [self.c[0] - self.w / 2.0, self.c[1] + off]
    }

    fn output(&self) -> [f32; 2] {
        [self.c[0] + self.w / 2.0, self.c[1]]
    }

    fn contains(&self, p: [f32; 2]) -> bool {
        (p[0] - self.c[0]).abs() <= self.w / 2.0 && (p[1] - self.c[1]).abs() <= self.h / 2.0
    }
}

/// Editor units ↔ pixels within the editor.
#[derive(Clone, Copy, Debug)]
struct Xf {
    pan: [f32; 2],
    zoom: f32,
}

impl Xf {
    fn to_px(self, p: [f32; 2]) -> [f32; 2] {
        [p[0] * self.zoom + self.pan[0], p[1] * self.zoom + self.pan[1]]
    }

    fn to_units(self, p: [f32; 2]) -> [f32; 2] {
        [(p[0] - self.pan[0]) / self.zoom, (p[1] - self.pan[1]) / self.zoom]
    }
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

/// What lies under `p` (pixels), ports before bodies, top nodes first.
fn hit(g: &NodeGraph, xf: Xf, p: [f32; 2]) -> Option<Hit> {
    let shapes: Vec<(u64, Shape)> = g.nodes.iter().map(|n| (n.id, Shape::of(g, n, n.pos))).collect();
    for (id, s) in shapes.iter().rev() {
        if s.outputs && dist(xf.to_px(s.output()), p) <= PORT_HIT {
            return Some(Hit::Out(*id));
        }
        for i in 0..s.inputs {
            if dist(xf.to_px(s.input(i)), p) <= PORT_HIT {
                return Some(Hit::In(*id, i));
            }
        }
    }
    let u = xf.to_units(p);
    shapes.iter().rev().find(|(_, s)| s.contains(u)).map(|(id, _)| Hit::Node(*id))
}

/// Fits the graph into a `w`×`h` editor, never above 100 %.
fn fit(g: &NodeGraph, w: f32, h: f32) -> Xf {
    let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
    for n in &g.nodes {
        let s = Shape::of(g, n, n.pos);
        x0 = x0.min(s.c[0] - s.w / 2.0);
        x1 = x1.max(s.c[0] + s.w / 2.0);
        y0 = y0.min(s.c[1] - s.h / 2.0);
        y1 = y1.max(s.c[1] + s.h / 2.0);
    }
    let zoom = ((w - 40.0) / (x1 - x0).max(1.0)).min((h - 40.0) / (y1 - y0).max(1.0)).clamp(0.45, 1.0);
    Xf { zoom, pan: [w / 2.0 - (x0 + x1) / 2.0 * zoom, h / 2.0 - (y0 + y1) / 2.0 * zoom] }
}

/// A wire from an output to an input, as an S-curve.
fn wire(a: [f32; 2], b: [f32; 2]) -> Vec<[f32; 2]> {
    let dx = ((b[0] - a[0]).abs() / 2.0).max(24.0);
    let (c1, c2) = ([a[0] + dx, a[1]], [b[0] - dx, b[1]]);
    (0..=24)
        .map(|i| {
            let t = i as f32 / 24.0;
            let u = 1.0 - t;
            let w = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
            [w[0] * a[0] + w[1] * c1[0] + w[2] * c2[0] + w[3] * b[0], w[0] * a[1] + w[1] * c1[1] + w[2] * c2[1] + w[3] * b[1]]
        })
        .collect()
}

fn stroke(window: &mut Window, origin: Point<Pixels>, pts: &[[f32; 2]], width: f32, color: Hsla) {
    let mut b = PathBuilder::stroke(px(width));
    for (i, p) in pts.iter().enumerate() {
        let q = point(origin.x + px(p[0]), origin.y + px(p[1]));
        if i == 0 { b.move_to(q) } else { b.line_to(q) }
    }
    if let Ok(path) = b.build() {
        window.paint_path(path, color);
    }
}

fn dot(window: &mut Window, origin: Point<Pixels>, c: [f32; 2], r: f32, color: Hsla) {
    let pts: Vec<Point<Pixels>> = (0..14)
        .map(|i| {
            let a = i as f32 / 14.0 * std::f32::consts::TAU;
            point(origin.x + px(c[0] + r * a.cos()), origin.y + px(c[1] + r * a.sin()))
        })
        .collect();
    let mut b = PathBuilder::fill();
    b.add_polygon(&pts, true);
    if let Ok(path) = b.build() {
        window.paint_path(path, color);
    }
}

fn on_down(state: &Entity<AppState>, view: &Entity<View>, ptr: Pointer, window: &mut Window, cx: &mut App) {
    let size = [f32::from(ptr.bounds.size.width), f32::from(ptr.bounds.size.height)];
    let p = [ptr.pos.0 * size[0], ptr.pos.1 * size[1]];
    let g = state.read(cx).node_graph();
    let xf = { let v = view.read(cx); Xf { pan: v.pan, zoom: v.zoom } };
    let drag = match hit(&g, xf, p) {
        Some(Hit::Out(id)) => Drag::Wire { from: id, at: p, cut: None },
        Some(Hit::In(id, i)) => match g.source(id, i) {
            Some(from) => Drag::Wire { from, at: p, cut: Some((id, i)) },
            None => Drag::WireBack { to: id, input: i, at: p },
        },
        Some(Hit::Node(id)) => {
            state.update(cx, |s, cx| s.select_node(id, cx));
            let is_layer = state.read(cx).adj().layers.iter().any(|l| l.id == id);
            if ptr.clicks >= 2 && is_layer {
                let name = state.read(cx).adj().layers.iter().find(|l| l.id == id).map(|l| l.name.clone()).unwrap_or_default();
                super::layers::rename(state, id, name, window, cx);
                return;
            }
            let c = xf.to_px(g.pos(id));
            Drag::Move { id, grab: [p[0] - c[0], p[1] - c[1]], moved: false }
        }
        None => Drag::Pan { from: xf.pan, start: p },
    };
    view.update(cx, |v, cx| {
        v.drag = Some(drag);
        cx.notify();
    });
}

fn on_move(state: &Entity<AppState>, view: &Entity<View>, ptr: Pointer, cx: &mut App) {
    let size = [f32::from(ptr.bounds.size.width), f32::from(ptr.bounds.size.height)];
    let p = [ptr.pos.0 * size[0], ptr.pos.1 * size[1]];
    let (drag, xf) = { let v = view.read(cx); (v.drag, Xf { pan: v.pan, zoom: v.zoom }) };
    match drag {
        Some(Drag::Move { id, grab, moved }) => {
            let start = xf.to_px(state.read(cx).node_graph().pos(id));
            if !moved && dist([p[0] - grab[0], p[1] - grab[1]], start) < 3.0 {
                return;
            }
            let pos = xf.to_units([p[0] - grab[0], p[1] - grab[1]]);
            state.update(cx, |s, cx| s.move_node(id, pos, cx));
            view.update(cx, |v, _| v.drag = Some(Drag::Move { id, grab, moved: true }));
        }
        Some(Drag::Wire { from, cut, .. }) => view.update(cx, |v, cx| {
            v.drag = Some(Drag::Wire { from, at: p, cut });
            cx.notify();
        }),
        Some(Drag::WireBack { to, input, .. }) => view.update(cx, |v, cx| {
            v.drag = Some(Drag::WireBack { to, input, at: p });
            cx.notify();
        }),
        Some(Drag::Pan { from, start }) => view.update(cx, |v, cx| {
            v.pan = [from[0] + p[0] - start[0], from[1] + p[1] - start[1]];
            cx.notify();
        }),
        None => {}
    }
}

fn on_up(state: &Entity<AppState>, view: &Entity<View>, cx: &mut App) {
    let (drag, xf) = { let v = view.read(cx); (v.drag, Xf { pan: v.pan, zoom: v.zoom }) };
    view.update(cx, |v, cx| {
        v.drag = None;
        cx.notify();
    });
    let g = state.read(cx).node_graph();
    match drag {
        Some(Drag::Move { moved: true, .. }) => state.update(cx, |s, cx| s.end_gesture(cx)),
        Some(Drag::Wire { from, at, cut }) => {
            // Dropped on a node body: its first free input, or its first.
            let target = match hit(&g, xf, at) {
                Some(Hit::In(id, i)) => Some((id, i)),
                Some(Hit::Node(id)) if id != INPUT => Some((id, (0..g.input_count(id)).find(|i| g.source(id, *i).is_none()).unwrap_or(0))),
                _ => None,
            };
            if target == cut {
                return;
            }
            state.update(cx, |s, cx| match (target, cut) {
                (Some((to, i)), Some((cto, ci))) => {
                    if from != to && !g.reaches(to, from) {
                        s.edit_graph(t("Move wire", "移动连线"), false, |g, _| {
                            g.unlink(cto, ci);
                            g.link(from, to, i);
                        }, cx);
                    }
                }
                (Some((to, i)), None) => s.connect_nodes(from, to, i, cx),
                (None, Some((cto, ci))) => s.disconnect_input(cto, ci, cx),
                (None, None) => {}
            });
        }
        Some(Drag::WireBack { to, input, at }) => {
            if let Some(Hit::Out(from) | Hit::Node(from)) = hit(&g, xf, at)
                && from != OUTPUT
            {
                state.update(cx, |s, cx| s.connect_nodes(from, to, input, cx));
            }
        }
        _ => {}
    }
}

fn node_body(
    state: &Entity<AppState>,
    g: &NodeGraph,
    n: &GraphNode,
    s: Shape,
    xf: Xf,
    number: Option<usize>,
    selected: bool,
    cx: &App,
) -> AnyElement {
    let p = pal(cx).clone();
    let tl = xf.to_px([s.c[0] - s.w / 2.0, s.c[1] - s.h / 2.0]);
    let (w, h) = (s.w * xf.zoom, s.h * xf.zoom);
    let base = div()
        .absolute()
        .left(px(tl[0]))
        .top(px(tl[1]))
        .w(px(w))
        .h(px(h))
        .border_1()
        .border_color(if selected { p.accent } else { p.divider })
        .when(selected, |d| d.border_2().shadow_md())
        .bg(p.raised);
    let label = |zh: &'static str, en: &'static str| div().text_xs().text_color(p.text_muted).child(t(en, zh));
    match n.kind {
        NodeKind::Input | NodeKind::Output => base
            .rounded(px(h / 2.0))
            .flex()
            .items_center()
            .justify_center()
            .gap_1()
            .child(Icon::new(if n.kind == NodeKind::Input { IconName::LogIn } else { IconName::LogOut }).xsmall().text_color(p.text_muted))
            .when(xf.zoom > 0.6, |d| d.child(if n.kind == NodeKind::Input { label("基础", "Base") } else { label("输出", "Out") }))
            .into_any_element(),
        NodeKind::Parallel | NodeKind::LayerMixer => {
            let parallel = n.kind == NodeKind::Parallel;
            base.id(SharedString::from(format!("node-mix-{}", n.id)))
                .rounded(px(7.0))
                .flex()
                .items_center()
                .justify_center()
                .child(Icon::new(if parallel { IconName::Blend } else { IconName::Layers2 }).xsmall().text_color(p.text_muted))
                .tooltip(move |window, cx| {
                    Tooltip::new(if parallel { t("Parallel mixer", "并行混合器") } else { t("Layer mixer · lowest input on top", "图层混合器 · 越靠下的输入越在上层") }).build(window, cx)
                })
                .into_any_element()
        }
        NodeKind::Layer => {
            let Some(layer) = state.read(cx).adj().layers.iter().find(|l| l.id == n.id).cloned() else { return div().into_any_element() };
            let live = number.is_some();
            let icon = match layer.kind {
                LayerKind::Adjustment => IconName::SlidersHorizontal,
                LayerKind::Heal => IconName::Bandage,
                LayerKind::Clone => IconName::Stamp,
                LayerKind::Fill => IconName::PaintBucket,
            };
            let masked = !layer.mask.shapes.is_empty();
            let graded = layer.adjustments != Default::default();
            let _ = g;
            base.rounded(px(6.0))
                .overflow_hidden()
                .when(!layer.enabled || !live, |d| d.opacity(0.45))
                .when(!live, |d| d.border_color(p.text_subtle))
                .child(
                    // The grade strip: lit once the node changes the image.
                    div().absolute().top_0().left_0().right_0().h(px(3.0)).bg(if graded && layer.enabled { p.accent } else { p.well }),
                )
                .child(
                    div()
                        .size_full()
                        .pt(px(4.0))
                        .px(px(6.0))
                        .flex()
                        .flex_col()
                        .justify_center()
                        .gap(px(1.0))
                        .child(
                            h_flex()
                                .gap_1()
                                .text_xs()
                                .text_color(if selected { p.accent } else { p.text_muted })
                                .child(number.map(|i| format!("{:02}", i)).unwrap_or_else(|| "—".into()))
                                .child(div().flex_1())
                                .when(masked, |d| d.child(Icon::new(if layer.mask.invert { IconName::CircleDashed } else { IconName::CircleDot }).xsmall()))
                                .when(!layer.enabled, |d| d.child(Icon::new(IconName::PowerOff).xsmall()))
                                .child(Icon::new(icon).xsmall()),
                        )
                        .when(xf.zoom > 0.6, |d| {
                            d.child(div().text_xs().text_color(if layer.enabled { p.text } else { p.text_subtle }).truncate().child(layer.name.clone()))
                        }),
                )
                .into_any_element()
        }
    }
}

/// The editor itself, `height` pixels tall; `id` keeps its view (pan, zoom)
/// apart from other copies of it.
pub fn editor(state: &Entity<AppState>, id: &'static str, height: f32, window: &mut Window, cx: &mut App) -> AnyElement {
    let p = pal(cx).clone();
    let sid = SharedString::from(id);
    let view = window.use_keyed_state(motion::key(id, "nodes"), cx, |_, _| View { zoom: 1.0, ..Default::default() });
    let bounds = view.read(cx).bounds;
    let size = [f32::from(bounds.size.width), f32::from(bounds.size.height)];
    let s = state.read(cx);
    let g = s.node_graph();
    let selected = s.selected_node();
    let order = g.render_order();
    let image = s.current.map(|c| c.0).unwrap_or(0);

    // Refit when the image or the graph's shape changes.
    let shape_key = (image, g.nodes.len(), g.links.len(), size[0] as u32);
    if size[0] > 1.0 && view.read(cx).fitted != Some(shape_key) {
        let f = fit(&g, size[0], size[1]);
        view.update(cx, |v, _| {
            v.fitted = Some(shape_key);
            v.pan = f.pan;
            v.zoom = f.zoom;
        });
    }
    let v = view.read(cx);
    let drag = v.drag;
    let hover = v.hover;
    let panning = matches!(drag, Some(Drag::Pan { .. }));
    let target = Xf { pan: v.pan, zoom: v.zoom };
    // Pan and zoom glide when refitting; a hand drag tracks the pointer.
    let xf = Xf {
        pan: [
            motion::follow(motion::key(id, "pan-x"), target.pan[0], motion::VIEW, !panning, window, cx),
            motion::follow(motion::key(id, "pan-y"), target.pan[1], motion::VIEW, !panning, window, cx),
        ],
        zoom: motion::follow(motion::key(id, "zoom"), target.zoom, motion::VIEW, true, window, cx),
    };

    // Nodes glide to new places; the one being dragged follows the pointer.
    let moving = match drag {
        Some(Drag::Move { id, moved: true, .. }) => Some(id),
        _ => None,
    };
    let shapes: Vec<(GraphNode, Shape)> = g
        .nodes
        .iter()
        .map(|n| {
            let travel = moving != Some(n.id);
            let c = [
                motion::follow(motion::key(format!("{id}-{}", n.id), "x"), n.pos[0], motion::FLOW, travel, window, cx),
                motion::follow(motion::key(format!("{id}-{}", n.id), "y"), n.pos[1], motion::FLOW, travel, window, cx),
            ];
            (n.clone(), Shape::of(&g, n, c))
        })
        .collect();
    let shape_of = |nid: u64| shapes.iter().find(|(n, _)| n.id == nid).map(|(_, s)| *s);

    let live_nodes: Vec<u64> = g.nodes.iter().map(|n| n.id).filter(|n| g.reaches(INPUT, *n) && g.reaches(*n, OUTPUT)).collect();
    let mut wires: Vec<(Vec<[f32; 2]>, bool)> = g
        .links
        .iter()
        .filter(|l| !matches!(drag, Some(Drag::Wire { cut: Some(c), .. }) if c == (l.to, l.input)))
        .filter_map(|l| {
            let (a, b) = (shape_of(l.from)?, shape_of(l.to)?);
            let live = live_nodes.contains(&l.from) && live_nodes.contains(&l.to);
            Some((wire(xf.to_px(a.output()), xf.to_px(b.input(l.input))), live))
        })
        .collect();
    let pending = match drag {
        Some(Drag::Wire { from, at, .. }) => shape_of(from).map(|s| wire(xf.to_px(s.output()), at)),
        Some(Drag::WireBack { to, input, at }) => shape_of(to).map(|s| wire(at, xf.to_px(s.input(input)))),
        _ => None,
    };
    let mut ports: Vec<([f32; 2], bool, bool)> = Vec::new();
    for (n, s) in &shapes {
        if s.outputs {
            ports.push((xf.to_px(s.output()), true, hover == Some(Hit::Out(n.id))));
        }
        for i in 0..s.inputs {
            ports.push((xf.to_px(s.input(i)), g.source(n.id, i).is_some(), hover == Some(Hit::In(n.id, i))));
        }
    }
    let (wire_live, wire_dead, port_on, port_off, grid) = (hexa(0x7fc98f, 0.9), hexa(0xffffff, 0.22), hexa(0x7fc98f, 1.0), hexa(0xffffff, 0.35), hexa(0xffffff, 0.035));
    let accent = p.accent;
    let zoom = xf.zoom;
    let pan = xf.pan;
    let pending_wire = pending.clone();
    wires.sort_by_key(|(_, live)| *live);

    let bodies: Vec<AnyElement> = shapes
        .iter()
        .map(|(n, s)| {
            let number = order.iter().position(|i| *i == n.id).map(|i| i + 1);
            motion::enter(SharedString::from(format!("{id}-node-{}", n.id)), div().absolute().size_full().child(node_body(state, &g, n, *s, xf, number, selected == n.id, cx)), 0.0)
                .into_any_element()
        })
        .collect();

    let (st_down, st_move, st_up, st_hover) = (state.clone(), state.clone(), state.clone(), state.clone());
    let (v_down, v_move, v_up, v_hover, v_wheel) = (view.clone(), view.clone(), view.clone(), view.clone(), view.clone());
    let cursor_grab = matches!(drag, Some(Drag::Pan { .. } | Drag::Move { .. }));
    let surface = Surface::new(sid.clone())
        .size_full()
        .when(cursor_grab, |d| d.cursor_grabbing())
        .when(!cursor_grab && hover.is_some(), |d| d.cursor_pointer())
        .on_down(move |ptr, window, cx| on_down(&st_down, &v_down, ptr, window, cx))
        .on_drag(move |ptr, _, cx| on_move(&st_move, &v_move, ptr, cx))
        .on_up(move |_, cx| on_up(&st_up, &v_up, cx))
        .on_hover_move(move |ptr, _, cx| {
            let size = [f32::from(ptr.bounds.size.width), f32::from(ptr.bounds.size.height)];
            let p = [ptr.pos.0 * size[0], ptr.pos.1 * size[1]];
            let g = st_hover.read(cx).node_graph();
            let xf = { let v = v_hover.read(cx); Xf { pan: v.pan, zoom: v.zoom } };
            let h = hit(&g, xf, p);
            if v_hover.read(cx).hover != h {
                v_hover.update(cx, |v, cx| {
                    v.hover = h;
                    cx.notify();
                });
            }
        })
        .child(
            canvas(
                {
                    let view = view.clone();
                    move |b, _, cx| {
                        // Fit once the editor's size is known, and after it changes.
                        if view.read(cx).bounds != b {
                            view.update(cx, |v, cx| {
                                v.bounds = b;
                                cx.notify();
                            });
                        }
                    }
                },
                move |b, _, window, _| {
                    let o = b.origin;
                    let (w, h) = (f32::from(b.size.width), f32::from(b.size.height));
                    // A dot grid that pans and zooms with the nodes.
                    let step = 18.0 * zoom;
                    if step > 6.0 {
                        let mut y = pan[1].rem_euclid(step);
                        while y < h {
                            let mut x = pan[0].rem_euclid(step);
                            while x < w {
                                dot(window, o, [x, y], 0.8, grid);
                                x += step;
                            }
                            y += step;
                        }
                    }
                    for (pts, live) in &wires {
                        stroke(window, o, pts, if *live { 1.6 } else { 1.2 }, if *live { wire_live } else { wire_dead });
                    }
                    if let Some(pts) = &pending_wire {
                        stroke(window, o, pts, 1.8, accent);
                    }
                    for (c, linked, hot) in &ports {
                        let r = if *hot { 5.0 } else { 3.6 };
                        dot(window, o, *c, r + 1.2, hexa(0x000000, 0.6));
                        dot(window, o, *c, r, if *hot { accent } else if *linked { port_on } else { port_off });
                    }
                },
            )
            .absolute()
            .size_full(),
        )
        .children(bodies);

    let st_menu = state.clone();
    div()
        .id(SharedString::from(format!("{id}-frame")))
        .relative()
        .w_full()
        .h(px(height))
        .rounded(px(5.0))
        .bg(p.well)
        .overflow_hidden()
        .on_scroll(move |e, _, cx| {
            // Ctrl or Alt + wheel zooms around the pointer; plain scroll leaves
            // the panel scrolling.
            if !(e.modifiers.control || e.modifiers.alt || e.modifiers.platform) {
                return;
            }
            cx.stop_propagation();
            let dy = match e.delta {
                ScrollDelta::Lines(d) => d.y,
                ScrollDelta::Pixels(d) => f32::from(d.y) / 40.0,
            };
            let origin = bounds.origin;
            let p = [f32::from(e.position.x - origin.x), f32::from(e.position.y - origin.y)];
            v_wheel.update(cx, |v, cx| {
                let zoom = (v.zoom * 1.12f32.powf(dy)).clamp(0.3, 2.0);
                let u = Xf { pan: v.pan, zoom: v.zoom }.to_units(p);
                v.pan = [p[0] - u[0] * zoom, p[1] - u[1] * zoom];
                v.zoom = zoom;
                cx.notify();
            });
        })
        .on_mouse_down(gpui_kit::MouseButton::Right, {
            let (st, view) = (state.clone(), view.clone());
            move |e, _, cx| {
                // The context menu acts on the node under the pointer.
                let v = view.read(cx);
                let p = [f32::from(e.position.x - v.bounds.origin.x), f32::from(e.position.y - v.bounds.origin.y)];
                let xf = Xf { pan: v.pan, zoom: v.zoom };
                let g = st.read(cx).node_graph();
                if let Some(Hit::Node(id) | Hit::Out(id) | Hit::In(id, _)) = hit(&g, xf, p) {
                    st.update(cx, |s, cx| s.select_node(id, cx));
                }
            }
        })
        .child(surface)
        .child(
            div()
                .absolute()
                .bottom(px(4.0))
                .right(px(6.0))
                .text_xs()
                .text_color(p.text_subtle)
                .child(format!("{:.0}%", xf.zoom * 100.0)),
        )
        .context_menu(move |menu, _, cx| {
            let s = st_menu.read(cx);
            let sel = s.selected_node();
            let is_layer = s.adj().layers.iter().any(|l| l.id == sel);
            let enabled = s.active_layer_ref().is_some_and(|l| l.enabled);
            let st = st_menu.clone();
            let st2 = st_menu.clone();
            let st3 = st_menu.clone();
            let mut m = menu
                .menu(NewNode::Serial.label(), Box::new(actions::AddSerialNode))
                .menu(NewNode::Parallel.label(), Box::new(actions::AddParallelNode))
                .menu(NewNode::Layer.label(), Box::new(actions::AddLayerNode))
                .menu(NewNode::Outside.label(), Box::new(actions::AddOutsideNode));
            if is_layer {
                m = m
                    .separator()
                    .menu(if enabled { t("Disable Node", "停用节点") } else { t("Enable Node", "启用节点") }, Box::new(actions::ToggleNode))
                    .item(PopupMenuItem::new(t("Reset Node Grade", "重置节点调色")).on_click(move |_, _, cx| st.update(cx, |s, cx| s.reset_node_grade(sel, cx))))
                    .item(PopupMenuItem::new(t("Rename…", "重命名…")).on_click(move |_, window, cx| {
                        let name = st2.read(cx).active_layer_ref().map(|l| l.name.clone()).unwrap_or_default();
                        super::layers::rename(&st2, sel, name, window, cx);
                    }));
            }
            if sel != INPUT && sel != OUTPUT {
                m = m.item(PopupMenuItem::new(t("Delete Node", "删除节点")).on_click(move |_, _, cx| st3.update(cx, |s, cx| s.delete_node(sel, cx))));
            }
            m.separator().menu(t("Tidy Nodes", "整理节点"), Box::new(actions::TidyNodes)).menu(t("Dock Node Editor", "停靠节点编辑器"), Box::new(actions::ToggleNodeDock))
        })
        .into_any_element()
}

/// The row of node commands above the editor.
pub fn toolbar(state: &Entity<AppState>, docked: bool, cx: &App) -> impl IntoElement + use<> {
    let p = pal(cx).clone();
    let has_image = state.read(cx).current.is_some();
    let add = |how: NewNode| {
        let st = state.clone();
        Button::new(SharedString::from(format!("node-add-{how:?}")))
            .xsmall()
            .ghost()
            .label(how.short())
            .disabled(!has_image)
            .tooltip(format!("{} ({})", how.label(), how.keys()))
            .on_click(move |_, _, cx| st.update(cx, |s, cx| s.add_node(how, cx)))
    };
    let st_kind = state.clone();
    let kinds = Button::new("node-add-kind").xsmall().ghost().icon(Icon::new(IconName::Plus)).disabled(!has_image).dropdown_menu(move |menu, _, _| {
        let item = |label: &'static str, kind: LayerKind| {
            let st = st_kind.clone();
            PopupMenuItem::new(label).on_click(move |_, _, cx| st.update(cx, |s, cx| s.add_node_of(NewNode::Serial, kind, cx)))
        };
        menu.item(item(t("Heal Node", "修复节点"), LayerKind::Heal)).item(item(t("Clone Node", "仿制节点"), LayerKind::Clone)).item(item(t("Fill Node", "填色节点"), LayerKind::Fill))
    });
    h_flex()
        .px_2()
        .pb_1p5()
        .gap_0p5()
        .child(add(NewNode::Serial))
        .child(add(NewNode::Parallel))
        .child(add(NewNode::Layer))
        .child(add(NewNode::Outside))
        .child(kinds)
        .child(div().flex_1())
        .child(
            Button::new("node-dock")
                .xsmall()
                .ghost()
                .icon(Icon::new(if docked { IconName::PanelBottom } else { IconName::Maximize }))
                .tooltip(if docked { t("Back to the Nodes tab (Ctrl+Shift+N)", "放回节点页（Ctrl+Shift+N）") } else { t("Dock under the viewer (Ctrl+Shift+N)", "停靠到查看器下方（Ctrl+Shift+N）") })
                .on_click(|_, window, cx| window.dispatch_action(Box::new(actions::ToggleNodeDock), cx)),
        )
        .text_color(p.text_muted)
}
