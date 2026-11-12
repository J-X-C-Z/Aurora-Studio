//! Everything drawn over the image: crop, grid, proof margin, straighten guide,
//! mask shapes, spots and the color-picker readout.

use gpui_kit::{
    App, Entity, Hsla, IntoElement, ParentElement, PathBuilder, Styled, Window, canvas, div,
    prelude::FluentBuilder as _, px,
};
use lumen_engine::*;

use super::Placement;
use crate::i18n::t;
use crate::state::{AppState, CursorTool};
use crate::theme::hexa;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Handle {
    Move,
    New,
    N,
    S,
    E,
    W,
    NE,
    NW,
    SE,
    SW,
}

#[derive(Clone, Debug)]
pub struct CropDrag {
    handle: Handle,
    start_rect: NormRect,
    start: (f32, f32),
}

impl CropDrag {
    pub fn begin(rect: NormRect, f: (f32, f32), pl: &Placement) -> Self {
        let tol_x = 12.0 / (pl.fw * pl.s);
        let tol_y = 12.0 / (pl.fh * pl.s);
        let near = |a: f32, b: f32, tol: f32| (a - b).abs() < tol;
        let inside_x = f.0 > rect.x - tol_x && f.0 < rect.x + rect.width + tol_x;
        let inside_y = f.1 > rect.y - tol_y && f.1 < rect.y + rect.height + tol_y;
        let l = near(f.0, rect.x, tol_x) && inside_y;
        let r = near(f.0, rect.x + rect.width, tol_x) && inside_y;
        let tp = near(f.1, rect.y, tol_y) && inside_x;
        let b = near(f.1, rect.y + rect.height, tol_y) && inside_x;
        let handle = match (l, r, tp, b) {
            (true, _, true, _) => Handle::NW,
            (_, true, true, _) => Handle::NE,
            (true, _, _, true) => Handle::SW,
            (_, true, _, true) => Handle::SE,
            (true, ..) => Handle::W,
            (_, true, ..) => Handle::E,
            (_, _, true, _) => Handle::N,
            (.., true) => Handle::S,
            // A drag inside an uncropped frame draws a new crop, as in Capture One.
            _ if inside_x && inside_y && rect != NormRect::FULL => Handle::Move,
            _ => Handle::New,
        };
        Self { handle, start_rect: rect, start: f }
    }

    /// The crop for a pointer at `f`. `ratio` is width/height in normalized units.
    pub fn update(&self, f: (f32, f32), ratio: Option<f32>, square: bool) -> NormRect {
        let r0 = self.start_rect;
        let (dx, dy) = (f.0 - self.start.0, f.1 - self.start.1);
        let (mut x0, mut y0, mut x1, mut y1) = (r0.x, r0.y, r0.x + r0.width, r0.y + r0.height);
        match self.handle {
            Handle::Move => {
                let x = (r0.x + dx).clamp(0.0, 1.0 - r0.width);
                let y = (r0.y + dy).clamp(0.0, 1.0 - r0.height);
                return NormRect { x, y, ..r0 };
            }
            Handle::New => {
                x0 = self.start.0.min(f.0);
                x1 = self.start.0.max(f.0);
                y0 = self.start.1.min(f.1);
                y1 = self.start.1.max(f.1);
            }
            Handle::N => y0 += dy,
            Handle::S => y1 += dy,
            Handle::W => x0 += dx,
            Handle::E => x1 += dx,
            Handle::NW => {
                x0 += dx;
                y0 += dy
            }
            Handle::NE => {
                x1 += dx;
                y0 += dy
            }
            Handle::SW => {
                x0 += dx;
                y1 += dy
            }
            Handle::SE => {
                x1 += dx;
                y1 += dy
            }
        }
        let (x0, x1) = (x0.clamp(0.0, 1.0), x1.clamp(0.0, 1.0));
        let (y0, y1) = (y0.clamp(0.0, 1.0), y1.clamp(0.0, 1.0));
        let mut rect = NormRect { x: x0.min(x1), y: y0.min(y1), width: (x1 - x0).abs().max(0.02), height: (y1 - y0).abs().max(0.02) };
        let ratio = ratio.or(square.then_some(1.0));
        if let Some(k) = ratio {
            // Keep the dragged edge, derive the other dimension, and stay inside the frame.
            let horizontal_edge = matches!(self.handle, Handle::N | Handle::S);
            if horizontal_edge {
                rect.width = (rect.height * k).min(1.0);
                rect.height = rect.width / k;
            } else {
                rect.height = (rect.width / k).min(1.0);
                rect.width = rect.height * k;
            }
            if matches!(self.handle, Handle::NW | Handle::NE | Handle::N) {
                rect.y = (y1.max(y0) - rect.height).max(0.0);
            }
            if matches!(self.handle, Handle::NW | Handle::SW | Handle::W) {
                rect.x = (x1.max(x0) - rect.width).max(0.0);
            }
            rect.x = rect.x.min(1.0 - rect.width);
            rect.y = rect.y.min(1.0 - rect.height);
        }
        rect
    }
}

fn stroke_path(window: &mut Window, pts: &[(f32, f32)], origin: gpui_kit::Point<gpui_kit::Pixels>, width: f32, color: Hsla, closed: bool) {
    if pts.len() < 2 {
        return;
    }
    let mut b = PathBuilder::stroke(px(width));
    for (i, (x, y)) in pts.iter().enumerate() {
        let p = gpui_kit::point(origin.x + px(*x), origin.y + px(*y));
        if i == 0 { b.move_to(p) } else { b.line_to(p) }
    }
    if closed {
        b.close();
    }
    if let Ok(path) = b.build() {
        window.paint_path(path, color);
    }
}

fn ellipse(c: (f32, f32), rx: f32, ry: f32, rot: f32) -> Vec<(f32, f32)> {
    let (s, co) = rot.to_radians().sin_cos();
    (0..=64)
        .map(|i| {
            let a = i as f32 / 64.0 * std::f32::consts::TAU;
            let (x, y) = (a.cos() * rx, a.sin() * ry);
            (c.0 + x * co - y * s, c.1 + x * s + y * co)
        })
        .collect()
}

pub fn render(
    state: &Entity<AppState>,
    pl: &Placement,
    guide: Option<((f32, f32), (f32, f32))>,
    _window: &mut Window,
    cx: &mut App,
) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let pl = *pl;
    let adj = s.adj();
    let cursor = s.cursor;
    let view = s.view.clone();
    let crop = adj.geometry.crop;
    let in_crop = cursor == CursorTool::Crop;
    let shown_crop = if in_crop { NormRect::FULL } else { crop };
    let (fl, ft, fw, fh) = pl.frame_rect();
    let frame_px = s.frame_size();
    let src_crop_px = (frame_px.0 * crop.width, frame_px.1 * crop.height);

    // Oriented-image point → viewport point.
    let img_to_view = move |p: [f32; 2]| {
        pl.to_view(((p[0] - shown_crop.x) / shown_crop.width.max(1e-4), (p[1] - shown_crop.y) / shown_crop.height.max(1e-4)))
    };

    let layer = s.active_layer_ref().cloned();
    let notes: Vec<Vec<(f32, f32)>> = s.current.and_then(|id| s.annotations.get(&id).cloned()).unwrap_or_default();
    let keystone_guides = cursor == CursorTool::Keystone;
    let ks = adj.geometry.keystone.clone();
    let show_shapes = cursor.is_mask_tool() || s.panels.mask_overlay;
    let spots = if cursor == CursorTool::Spot { adj.spots.clone() } else { Vec::new() };
    let picker = s.picker_point.zip(s.sampled_color).filter(|_| cursor == CursorTool::ColorPicker);
    let image_w_view = fw / shown_crop.width.max(1e-4);
    let brush_r = s.panels.brush_size * image_w_view.max(fh / shown_crop.height.max(1e-4));

    let (cl, ct) = pl.to_view((crop.x, crop.y));
    let (cw, ch) = (crop.width * fw, crop.height * fh);
    let dim = hexa(0x000000, s.settings.crop_mask);
    let crop_guide = s.settings.crop_guide;
    let divisions = s.settings.grid_divisions.max(2) as usize;

    div()
        .absolute()
        .size_full()
        // Proof margin: dim everything outside the frame a little more.
        .when(view.proof, |this| {
            this.child(div().absolute().left(px(fl - 12.0)).top(px(ft - 12.0)).w(px(fw + 24.0)).h(px(fh + 24.0)).border(px(12.0)).border_color(hexa(0xffffff, 0.9)))
        })
        .when(view.grid && !in_crop, |this| {
            this.child(
                canvas(
                    |_, _, _| (),
                    move |b, _, window, _| {
                        let c = hexa(0xffffff, 0.28);
                        for i in 1..divisions {
                            let x = fl + fw * i as f32 / divisions as f32;
                            let y = ft + fh * i as f32 / divisions as f32;
                            stroke_path(window, &[(x, ft), (x, ft + fh)], b.origin, 1.0, c, false);
                            stroke_path(window, &[(fl, y), (fl + fw, y)], b.origin, 1.0, c, false);
                        }
                    },
                )
                .absolute()
                .size_full(),
            )
        })
        .when(in_crop, |this| {
            this.child(div().absolute().left(px(fl)).top(px(ft)).w(px(fw)).h(px(ct - ft)).bg(dim))
                .child(div().absolute().left(px(fl)).top(px(ct + ch)).w(px(fw)).h(px(ft + fh - ct - ch)).bg(dim))
                .child(div().absolute().left(px(fl)).top(px(ct)).w(px(cl - fl)).h(px(ch)).bg(dim))
                .child(div().absolute().left(px(cl + cw)).top(px(ct)).w(px(fl + fw - cl - cw)).h(px(ch)).bg(dim))
                .child(div().absolute().left(px(cl)).top(px(ct)).w(px(cw)).h(px(ch)).border_1().border_color(hexa(0xffffff, 0.9)))
                .child(
                    canvas(
                        |_, _, _| (),
                        move |b, _, window, _| {
                            let c = hexa(0xffffff, 0.35);
                            let lines: Vec<f32> = match crop_guide {
                                crate::state::CropGuide::Thirds => vec![1.0 / 3.0, 2.0 / 3.0],
                                crate::state::CropGuide::Golden => vec![0.382, 0.618],
                                crate::state::CropGuide::Grid => (1..6).map(|i| i as f32 / 6.0).collect(),
                                _ => Vec::new(),
                            };
                            for f in lines {
                                let x = cl + cw * f;
                                let y = ct + ch * f;
                                stroke_path(window, &[(x, ct), (x, ct + ch)], b.origin, 1.0, c, false);
                                stroke_path(window, &[(cl, y), (cl + cw, y)], b.origin, 1.0, c, false);
                            }
                            if crop_guide == crate::state::CropGuide::Diagonal {
                                stroke_path(window, &[(cl, ct), (cl + cw, ct + ch)], b.origin, 1.0, c, false);
                                stroke_path(window, &[(cl + cw, ct), (cl, ct + ch)], b.origin, 1.0, c, false);
                            }
                            let k = 16.0f32.min(cw / 3.0).min(ch / 3.0);
                            let w = hexa(0xffffff, 1.0);
                            for (x, y, sx, sy) in [(cl, ct, 1.0, 1.0), (cl + cw, ct, -1.0, 1.0), (cl, ct + ch, 1.0, -1.0), (cl + cw, ct + ch, -1.0, -1.0)] {
                                stroke_path(window, &[(x, y + k * sy), (x, y), (x + k * sx, y)], b.origin, 3.0, w, false);
                            }
                            for (x, y, horizontal) in [(cl + cw / 2.0, ct, true), (cl + cw / 2.0, ct + ch, true), (cl, ct + ch / 2.0, false), (cl + cw, ct + ch / 2.0, false)] {
                                if horizontal {
                                    stroke_path(window, &[(x - k / 2.0, y), (x + k / 2.0, y)], b.origin, 3.0, w, false);
                                } else {
                                    stroke_path(window, &[(x, y - k / 2.0), (x, y + k / 2.0)], b.origin, 3.0, w, false);
                                }
                            }
                        },
                    )
                    .absolute()
                    .size_full(),
                )
                .child(
                    div()
                        .absolute()
                        .left(px(cl + 6.0))
                        .top(px(ct + 6.0))
                        .px_1p5()
                        .py_0p5()
                        .rounded(px(3.0))
                        .bg(hexa(0x000000, 0.6))
                        .text_xs()
                        .text_color(hexa(0xffffff, 0.9))
                        .child(format!("{:.0} × {:.0} px", src_crop_px.0, src_crop_px.1)),
                )
        })
        .when_some(guide, |this, (a, b2)| {
            let a = pl.to_view(a);
            let b2 = pl.to_view(b2);
            this.child(
                canvas(
                    |_, _, _| (),
                    move |b, _, window, _| {
                        stroke_path(window, &[a, b2], b.origin, 2.0, hexa(0xffd36b, 1.0), false);
                    },
                )
                .absolute()
                .size_full(),
            )
            .child(
                div()
                    .absolute()
                    .left(px(b2.0 + 10.0))
                    .top(px(b2.1 + 10.0))
                    .px_1p5()
                    .rounded(px(3.0))
                    .bg(hexa(0x000000, 0.6))
                    .text_xs()
                    .text_color(hexa(0xffffff, 0.9))
                    .child(t("Release to straighten", "松开以拉直")),
            )
        })
        .when(show_shapes && layer.is_some(), |this| {
            let layer = layer.clone().unwrap_or_default();
            this.child(
                canvas(
                    |_, _, _| (),
                    move |b, _, window, _| {
                        let line = hexa(0xffffff, 0.9);
                        let shadow = hexa(0x000000, 0.5);
                        for shape in &layer.mask.shapes {
                            match shape {
                                MaskShape::Radial(g) => {
                                    let c = img_to_view(g.center);
                                    let e = img_to_view([g.center[0] + g.radius[0], g.center[1] + g.radius[1]]);
                                    let (rx, ry) = ((e.0 - c.0).abs(), (e.1 - c.1).abs());
                                    let inner = 1.0 - g.feather / 100.0;
                                    stroke_path(window, &ellipse(c, rx, ry, g.rotation), b.origin, 2.5, shadow, true);
                                    stroke_path(window, &ellipse(c, rx, ry, g.rotation), b.origin, 1.5, line, true);
                                    stroke_path(window, &ellipse(c, rx * inner, ry * inner, g.rotation), b.origin, 1.0, hexa(0xffffff, 0.5), true);
                                }
                                MaskShape::Linear(g) => {
                                    let a = img_to_view(g.start);
                                    let z = img_to_view(g.end);
                                    let d = (z.0 - a.0, z.1 - a.1);
                                    let n = {
                                        let l = (d.0 * d.0 + d.1 * d.1).sqrt().max(1e-3);
                                        (-d.1 / l * 2000.0, d.0 / l * 2000.0)
                                    };
                                    for (pt, alpha) in [(a, 0.95), ((a.0 + d.0 / 2.0, a.1 + d.1 / 2.0), 0.5), (z, 0.95)] {
                                        stroke_path(window, &[(pt.0 - n.0, pt.1 - n.1), (pt.0 + n.0, pt.1 + n.1)], b.origin, 1.5, hexa(0xffffff, alpha), false);
                                    }
                                }
                                _ => {}
                            }
                        }
                    },
                )
                .absolute()
                .size_full(),
            )
        })
        .when(!notes.is_empty(), |this| {
            this.child(
                canvas(
                    |_, _, _| (),
                    move |b, _, window, _| {
                        for stroke in &notes {
                            let pts: Vec<(f32, f32)> = stroke.iter().map(|f| pl.to_view(*f)).collect();
                            if pts.len() == 1 {
                                let c = pts[0];
                                stroke_path(window, &ellipse(c, 2.0, 2.0, 0.0), b.origin, 3.0, hexa(0xffd36b, 1.0), true);
                            }
                            stroke_path(window, &pts, b.origin, 3.0, hexa(0x000000, 0.4), false);
                            stroke_path(window, &pts, b.origin, 2.0, hexa(0xffd36b, 1.0), false);
                        }
                    },
                )
                .absolute()
                .size_full(),
            )
        })
        .when(keystone_guides, |this| {
            // A perspective grid that leans with the current correction.
            let lean_v = ks.vertical / 100.0 * 0.12;
            let lean_h = ks.horizontal / 100.0 * 0.12;
            this.child(
                canvas(
                    |_, _, _| (),
                    move |b, _, window, _| {
                        let c = hexa(0x6ad1ff, 0.75);
                        for i in 0..=6 {
                            let u = i as f32 / 6.0;
                            let top = (fl + fw * (u + (u - 0.5) * lean_v), ft);
                            let bottom = (fl + fw * (u - (u - 0.5) * lean_v), ft + fh);
                            stroke_path(window, &[top, bottom], b.origin, 1.0, c, false);
                            let left = (fl, ft + fh * (u + (u - 0.5) * lean_h));
                            let right = (fl + fw, ft + fh * (u - (u - 0.5) * lean_h));
                            stroke_path(window, &[left, right], b.origin, 1.0, c, false);
                        }
                    },
                )
                .absolute()
                .size_full(),
            )
            .child(
                div()
                    .absolute()
                    .left(px(fl + 8.0))
                    .top(px(ft + 8.0))
                    .px_1p5()
                    .rounded(px(3.0))
                    .bg(hexa(0x000000, 0.6))
                    .text_xs()
                    .text_color(hexa(0xffffff, 0.9))
                    .child(format!("{}  V {:.0}  H {:.0}", t("Drag to correct perspective", "拖动校正透视"), ks.vertical, ks.horizontal)),
            )
        })
        .children(spots.into_iter().map(|sp| {
            let c = img_to_view([sp.x, sp.y]);
            let r = sp.radius * image_w_view;
            div()
                .absolute()
                .left(px(c.0 - r))
                .top(px(c.1 - r))
                .size(px(r * 2.0))
                .rounded_full()
                .border_2()
                .border_color(hexa(0xffffff, 0.9))
        }))
        .when(cursor.is_mask_tool() || cursor == CursorTool::Spot, |this| {
            // Brush size ring follows the pointer via the viewer's hover state.
            this.child(BrushRing { radius: brush_r })
        })
        .when_some(picker, |this, (pt, c)| {
            let v = pl.to_view((pt[0], pt[1]));
            let swatch = gpui_kit::Rgba { r: c[0], g: c[1], b: c[2], a: 1.0 };
            this.child(
                div()
                    .absolute()
                    .left(px(v.0 + 12.0))
                    .top(px(v.1 + 12.0))
                    .flex()
                    .items_center()
                    .gap_1p5()
                    .px_2()
                    .py_1()
                    .rounded(px(4.0))
                    .bg(hexa(0x000000, 0.7))
                    .text_xs()
                    .text_color(hexa(0xffffff, 0.9))
                    .child(div().size(px(14.0)).rounded(px(3.0)).bg(swatch).border_1().border_color(hexa(0xffffff, 0.6)))
                    .child(format!("R {:.0}  G {:.0}  B {:.0}", c[0] * 255.0, c[1] * 255.0, c[2] * 255.0)),
            )
        })
}

#[derive(gpui_kit::IntoElement)]
struct BrushRing {
    radius: f32,
}

impl gpui_kit::RenderOnce for BrushRing {
    fn render(self, window: &mut Window, _cx: &mut App) -> impl IntoElement {
        let pos = window.mouse_position();
        let r = self.radius.max(3.0);
        canvas(
            |_, _, _| (),
            move |b, _, window, _| {
                if !b.contains(&pos) {
                    return;
                }
                let c = (f32::from(pos.x - b.origin.x), f32::from(pos.y - b.origin.y));
                stroke_path(window, &ellipse(c, r, r, 0.0), b.origin, 2.5, hexa(0x000000, 0.5), true);
                stroke_path(window, &ellipse(c, r, r, 0.0), b.origin, 1.2, hexa(0xffffff, 0.9), true);
            },
        )
        .absolute()
        .size_full()
    }
}
