//! The image viewer: zoom and pan with springs, before/after comparison, overlays,
//! and every cursor tool's pointer interaction.

mod multi;
mod overlays;

use gpui_kit::assets::IconName;
use gpui_kit::component::{Icon, Sizable as _, button::ButtonVariants as _, h_flex, spinner::Spinner, v_flex};
use gpui_kit::{
    App, Context, Entity, InteractiveElement, IntoElement, ObjectFit, ParentElement, Pixels, Render,
    ScrollDelta, SharedString, StatefulInteractiveElement as _, Styled, StyledImage as _, Subscription, Window, div, img, prelude::FluentBuilder as _, px,
};
use lumen_engine::*;

use crate::i18n::t;
use crate::motion;
use crate::state::{AppState, CompareMode, CursorTool, Scope, SpeedParam};
use crate::theme::{hexa, pal};
use crate::widgets::surface::{Pointer, Surface};

/// Where the developed frame sits in the viewport this frame, in logical pixels.
#[derive(Clone, Copy, Debug, Default)]
pub struct Placement {
    /// Viewport size.
    pub vw: f32,
    pub vh: f32,
    /// Logical pixels per frame pixel.
    pub s: f32,
    /// Frame point at the viewport center.
    pub cx: f32,
    pub cy: f32,
    /// Frame size in frame pixels.
    pub fw: f32,
    pub fh: f32,
}

impl Placement {
    /// Viewport point of a normalized frame point.
    pub fn to_view(&self, f: (f32, f32)) -> (f32, f32) {
        (self.vw / 2.0 + (f.0 - self.cx) * self.fw * self.s, self.vh / 2.0 + (f.1 - self.cy) * self.fh * self.s)
    }

    /// Normalized frame point under a viewport point.
    pub fn to_frame(&self, v: (f32, f32)) -> (f32, f32) {
        (self.cx + (v.0 - self.vw / 2.0) / (self.fw * self.s), self.cy + (v.1 - self.vh / 2.0) / (self.fh * self.s))
    }

    /// The whole frame's rectangle in the viewport: (left, top, width, height).
    pub fn frame_rect(&self) -> (f32, f32, f32, f32) {
        let (l, t) = self.to_view((0.0, 0.0));
        (l, t, self.fw * self.s, self.fh * self.s)
    }
}

#[derive(Clone, Debug)]
enum Drag {
    Pan { start: (f32, f32), center: (f32, f32) },
    Split,
    Crop(overlays::CropDrag),
    Straighten { from: (f32, f32), to: (f32, f32) },
    Brush,
    Linear,
    Radial { center: (f32, f32) },
    Keystone { start: (f32, f32), vertical: f32, horizontal: f32 },
    Annotate,
    /// Speed Edit drag: pointer x and the value at mouse down.
    Speed { start: f32, value: f32 },
}

pub struct Viewer {
    state: Entity<AppState>,
    placement: Placement,
    drag: Option<Drag>,
    /// Straighten line in frame space while dragging.
    pub(crate) guide: Option<((f32, f32), (f32, f32))>,
    hover: Option<(f32, f32)>,
    /// The loupe is pressed: it grows and magnifies further.
    loupe_down: bool,
    last_frame: u64,
    /// The last Speed Edit parameter, kept while its readout fades out.
    speed_shown: Option<SpeedParam>,
    _sub: Subscription,
}

impl Viewer {
    pub fn new(state: &Entity<AppState>, cx: &mut Context<Self>) -> Self {
        Self {
            state: state.clone(),
            placement: Placement::default(),
            drag: None,
            guide: None,
            hover: None,
            loupe_down: false,
            last_frame: 0,
            speed_shown: None,
            _sub: cx.observe(state, |_, _, cx| cx.notify()),
        }
    }

    fn local(&self, p: &Pointer) -> (f32, f32) {
        (f32::from(p.point.x - p.bounds.origin.x), f32::from(p.point.y - p.bounds.origin.y))
    }

    /// Frame point → oriented-image point (masks and spots live there).
    fn to_image(&self, f: (f32, f32), cx: &App) -> [f32; 2] {
        let s = self.state.read(cx);
        let crop = if s.cursor == CursorTool::Crop { NormRect::FULL } else { s.adj().geometry.crop };
        [crop.x + f.0 * crop.width, crop.y + f.1 * crop.height]
    }

    fn on_down(&mut self, p: Pointer, window: &mut Window, cx: &mut Context<Self>) {
        let v = self.local(&p);
        let f = self.placement.to_frame(v);
        let (cursor, compare, split, zoomed) = {
            let s = self.state.read(cx);
            (s.cursor, s.view.compare, s.view.split, s.zoom.scale.is_some())
        };
        // While a Speed Edit key is held, dragging adjusts its parameter.
        if let Some(param) = self.state.read(cx).speed.as_ref().map(|s| s.param) {
            let value = self.state.read(cx).speed_value(param);
            self.drag = Some(Drag::Speed { start: v.0, value });
            return;
        }
        // The before/after divider wins wherever it is.
        if compare == CompareMode::Split && (v.0 - split * self.placement.vw).abs() < if crate::theme::touch() { 22.0 } else { 8.0 } {
            self.drag = Some(Drag::Split);
            return;
        }
        if p.clicks >= 2 && matches!(cursor, CursorTool::Select | CursorTool::Pan) {
            self.toggle_zoom(f, cx);
            return;
        }
        let state = self.state.clone();
        match cursor {
            CursorTool::Select | CursorTool::Pan => {
                if zoomed || cursor == CursorTool::Pan {
                    let center = state.read(cx).zoom.center;
                    self.drag = Some(Drag::Pan { start: v, center });
                }
            }
            CursorTool::Zoom => {
                let current = state.read(cx).effective_scale();
                let next = if p.modifiers.alt { current / 2.0 } else { current * 2.0 };
                let fit = state.read(cx).fit_scale();
                state.update(cx, |s, cx| {
                    if next <= fit * 1.01 { s.set_zoom(None, None, cx) } else { s.set_zoom(Some(next), Some(f), cx) }
                });
            }
            CursorTool::Crop => {
                let crop = state.read(cx).adj().geometry.crop;
                self.drag = Some(Drag::Crop(overlays::CropDrag::begin(crop, f, &self.placement)));
            }
            CursorTool::Straighten => {
                self.drag = Some(Drag::Straighten { from: f, to: f });
                self.guide = Some((f, f));
            }
            CursorTool::Spot => {
                let at = self.to_image(f, cx);
                state.update(cx, |s, cx| {
                    let size = s.panels.brush_size;
                    s.edit(t("Spot removal", "污点去除"), |a| {
                        let id = a.spots.iter().map(|s| s.id).max().unwrap_or(0) + 1;
                        a.spots.push(Spot { id, kind: SpotKind::Spot, x: at[0], y: at[1], radius: size * 0.5 });
                    }, cx)
                });
            }
            CursorTool::Brush | CursorTool::Eraser => {
                let at = self.to_image(f, cx);
                let erase = cursor == CursorTool::Eraser;
                state.update(cx, |s, cx| {
                    let Scope::Layer(id) = s.scope() else { return };
                    let pr = s.panels.clone();
                    s.edit_layer(t("Brush", "画笔"), id, true, |l| {
                        crate::state::add_shape(&mut l.mask, MaskShape::Brush(BrushStroke {
                            op: if erase { MaskOp::Subtract } else { MaskOp::Add },
                            points: vec![[at[0], at[1], 1.0]],
                            size: pr.brush_size,
                            hardness: pr.brush_hardness,
                            opacity: pr.brush_opacity,
                            flow: pr.brush_flow,
                            auto_mask: pr.brush_auto_mask,
                        }))
                    }, cx)
                });
                self.drag = Some(Drag::Brush);
            }
            CursorTool::LinearGradient => {
                self.drag = Some(Drag::Linear);
                let at = self.to_image(f, cx);
                state.update(cx, |s, cx| {
                    let Scope::Layer(id) = s.scope() else { return };
                    s.edit_layer(t("Linear gradient", "线性渐变"), id, true, |l| {
                        crate::state::add_shape(&mut l.mask, MaskShape::Linear(LinearGradient { op: MaskOp::Add, start: at, end: [at[0], at[1] + 0.001] }))
                    }, cx)
                });
            }
            CursorTool::RadialGradient => {
                self.drag = Some(Drag::Radial { center: f });
                let at = self.to_image(f, cx);
                state.update(cx, |s, cx| {
                    let Scope::Layer(id) = s.scope() else { return };
                    s.edit_layer(t("Radial gradient", "径向渐变"), id, true, |l| {
                        crate::state::add_shape(&mut l.mask, MaskShape::Radial(RadialGradient {
                            op: MaskOp::Add,
                            center: at,
                            radius: [0.01, 0.01],
                            rotation: 0.0,
                            feather: 60.0,
                            inverted: false,
                        }))
                    }, cx)
                });
            }
            CursorTool::ColorPicker => {
                let at = [f.0, f.1];
                state.update(cx, |s, cx| {
                    let Some(id) = s.current else { return };
                    let adj = crate::state::flattened(s.adj());
                    match s.engine.sample_color(id, &adj, at) {
                        Ok(c) => {
                            s.sampled_color = Some(c);
                            s.picker_point = Some(at);
                            if s.panels.pick_to_range {
                                s.range_from_sample(c, cx);
                            }
                        }
                        Err(err) => s.notify(err.to_string(), cx),
                    }
                    cx.notify();
                });
            }
            CursorTool::WhiteBalance => {
                let at = self.to_image(f, cx);
                state.update(cx, |s, cx| {
                    let Some(id) = s.current else { return };
                    match s.engine.pick_white_balance(id, at) {
                        Ok((k, tint)) => s.edit(t("Pick white balance", "白平衡取样"), |a| {
                            a.white_balance.mode = WhiteBalanceMode::Custom;
                            a.white_balance.kelvin = k;
                            a.white_balance.tint = tint;
                        }, cx),
                        Err(err) => s.notify(err.to_string(), cx),
                    }
                });
            }
            CursorTool::Keystone => {
                let k = state.read(cx).adj().geometry.keystone.clone();
                self.drag = Some(Drag::Keystone { start: v, vertical: k.vertical, horizontal: k.horizontal });
            }
            CursorTool::Loupe => {
                self.loupe_down = true;
                cx.notify();
            }
            CursorTool::Annotate => {
                state.update(cx, |s, cx| {
                    if let Some(id) = s.current {
                        s.annotations.entry(id).or_default().push(vec![f]);
                        cx.notify();
                    }
                });
                self.drag = Some(Drag::Annotate);
            }
        }
        let _ = window;
    }

    fn on_drag(&mut self, p: Pointer, _window: &mut Window, cx: &mut Context<Self>) {
        let v = self.local(&p);
        let f = self.placement.to_frame(v);
        let Some(drag) = self.drag.clone() else { return };
        let state = self.state.clone();
        match drag {
            Drag::Speed { start, value } => {
                state.update(cx, |s, cx| {
                    let Some(param) = s.speed.as_ref().map(|s| s.param) else { return };
                    let (min, max, _) = s.speed_meta(param);
                    // The whole range over 400 px; Shift for fine control.
                    let k = if p.modifiers.shift { 1600.0 } else { 400.0 };
                    s.speed_set(value + (v.0 - start) / k * (max - min), cx)
                });
            }
            Drag::Pan { start, center } => {
                let pl = self.placement;
                let dx = -(v.0 - start.0) / (pl.fw * pl.s);
                let dy = -(v.1 - start.1) / (pl.fh * pl.s);
                state.update(cx, |s, cx| {
                    let delta = (center.0 + dx - s.zoom.center.0, center.1 + dy - s.zoom.center.1);
                    s.pan_by(delta.0, delta.1, cx)
                });
            }
            Drag::Split => {
                let x = (v.0 / self.placement.vw).clamp(0.02, 0.98);
                state.update(cx, |s, cx| {
                    s.view.split = x;
                    cx.notify();
                });
            }
            Drag::Crop(d) => {
                let (aspect, orig) = {
                    let s = state.read(cx);
                    let (fw, fh) = s.frame_size();
                    (s.adj().geometry.aspect, fw / fh)
                };
                let ratio = aspect.ratio(orig).map(|r| r / orig);
                let rect = d.update(f, ratio, p.modifiers.shift);
                state.update(cx, |s, cx| s.edit_live(t("Crop", "裁剪"), |a| a.geometry.crop = rect, cx));
            }
            Drag::Straighten { from, .. } => {
                self.drag = Some(Drag::Straighten { from, to: f });
                self.guide = Some((from, f));
                cx.notify();
            }
            Drag::Brush => {
                let at = self.to_image(f, cx);
                state.update(cx, |s, cx| {
                    let Scope::Layer(id) = s.scope() else { return };
                    s.edit_layer(t("Brush", "画笔"), id, true, |l| {
                        if let Some(MaskShape::Brush(stroke)) = l.mask.shapes.last_mut() {
                            // Fill the gap since the last sample so fast strokes stay continuous.
                            let spacing = (stroke.size * 0.2).max(1e-4);
                            let last = stroke.points.last().copied().unwrap_or([at[0], at[1], 1.0]);
                            let d = ((last[0] - at[0]).powi(2) + (last[1] - at[1]).powi(2)).sqrt();
                            let steps = (d / spacing).floor() as usize;
                            for i in 1..=steps.min(512) {
                                let k = i as f32 / steps as f32;
                                stroke.points.push([last[0] + (at[0] - last[0]) * k, last[1] + (at[1] - last[1]) * k, 1.0]);
                            }
                        }
                    }, cx)
                });
            }
            Drag::Linear => {
                let at = self.to_image(f, cx);
                state.update(cx, |s, cx| {
                    let Scope::Layer(id) = s.scope() else { return };
                    s.edit_layer(t("Linear gradient", "线性渐变"), id, true, |l| {
                        if let Some(MaskShape::Linear(g)) = l.mask.shapes.last_mut() {
                            g.end = at;
                        }
                    }, cx)
                });
            }
            Drag::Keystone { start, vertical, horizontal } => {
                let pl = self.placement;
                let dv = (v.1 - start.1) / pl.vh.max(1.0) * 200.0;
                let dh = (v.0 - start.0) / pl.vw.max(1.0) * 200.0;
                state.update(cx, |s, cx| {
                    s.edit_live(t("Keystone", "梯形校正"), |a| {
                        a.geometry.keystone.vertical = (vertical + dv).clamp(-100.0, 100.0);
                        a.geometry.keystone.horizontal = (horizontal + dh).clamp(-100.0, 100.0);
                    }, cx)
                });
            }
            Drag::Annotate => {
                state.update(cx, |s, cx| {
                    if let Some(id) = s.current
                        && let Some(stroke) = s.annotations.entry(id).or_default().last_mut()
                    {
                        stroke.push(f);
                        cx.notify();
                    }
                });
            }
            Drag::Radial { center } => {
                let c = self.to_image(center, cx);
                let at = self.to_image(f, cx);
                state.update(cx, |s, cx| {
                    let Scope::Layer(id) = s.scope() else { return };
                    s.edit_layer(t("Radial gradient", "径向渐变"), id, true, |l| {
                        if let Some(MaskShape::Radial(g)) = l.mask.shapes.last_mut() {
                            g.radius = [(at[0] - c[0]).abs().max(0.005), (at[1] - c[1]).abs().max(0.005)];
                        }
                    }, cx)
                });
            }
        }
    }

    fn on_up(&mut self, _window: &mut Window, cx: &mut Context<Self>) {
        self.loupe_down = false;
        let drag = self.drag.take();
        if let Some(Drag::Straighten { from, to }) = drag {
            self.guide = None;
            let pl = self.placement;
            let dx = (to.0 - from.0) * pl.fw;
            let dy = (to.1 - from.1) * pl.fh;
            if dx.abs() + dy.abs() > 4.0 {
                let mut angle = dy.atan2(dx).to_degrees();
                // Snap to whichever axis the line is closer to.
                if angle.abs() > 45.0 && angle.abs() < 135.0 {
                    angle -= 90.0 * angle.signum();
                } else if angle.abs() >= 135.0 {
                    angle -= 180.0 * angle.signum();
                }
                self.state.update(cx, |s, cx| {
                    s.edit(t("Straighten", "拉直"), |a| a.geometry.rotation = (a.geometry.rotation - angle).clamp(-45.0, 45.0), cx)
                });
            }
        }
        self.state.update(cx, |s, cx| s.end_gesture(cx));
        cx.notify();
    }

    fn toggle_zoom(&mut self, f: (f32, f32), cx: &mut Context<Self>) {
        self.state.update(cx, |s, cx| {
            if s.zoom.scale.is_some() {
                s.set_zoom(None, None, cx)
            } else {
                let sf = s.scale_factor;
                s.set_zoom(Some(sf), Some(f), cx)
            }
        });
    }

    /// Scales the view by `factor`, keeping frame point `f` where it is on
    /// screen; at or below the fitted size the view goes back to fit.
    fn zoom_about(&mut self, f: (f32, f32), factor: f32, cx: &mut Context<Self>) {
        self.state.update(cx, |s, cx| {
            let current = s.effective_scale();
            let next = current * factor;
            if next <= s.fit_scale() {
                s.set_zoom(None, None, cx)
            } else {
                let k = current / next;
                let c = if s.zoom.scale.is_none() { (0.5, 0.5) } else { s.zoom.center };
                s.set_zoom(Some(next), Some((f.0 + (c.0 - f.0) * k, f.1 + (c.1 - f.1) * k)), cx)
            }
        });
    }

    /// Two fingers on a touch screen, or a trackpad pinch.
    fn on_pinch(&mut self, e: &gpui_kit::PinchEvent, cx: &mut Context<Self>) {
        let origin = self.surface_origin(cx);
        let v = (f32::from(e.position.x - origin.x), f32::from(e.position.y - origin.y));
        self.zoom_about(self.placement.to_frame(v), (1.0 + e.delta).clamp(0.5, 2.0), cx);
    }

    fn on_wheel(&mut self, e: &gpui_kit::ScrollEvent, origin: gpui_kit::Point<Pixels>, cx: &mut Context<Self>) {
        if self.state.read(cx).speed.is_some() {
            // One notch is 1/100 of the range; scrolling up raises the value.
            let steps = match e.delta {
                ScrollDelta::Lines(d) => d.y,
                ScrollDelta::Pixels(d) => f32::from(d.y) / 40.0,
            };
            let steps = if e.modifiers.shift { steps / 4.0 } else { steps };
            self.state.update(cx, |s, cx| s.speed_nudge(steps, cx));
            return;
        }
        let delta = match e.delta {
            ScrollDelta::Lines(d) => (d.x * 40.0, d.y * 40.0),
            ScrollDelta::Pixels(d) => (f32::from(d.x), f32::from(d.y)),
        };
        let v = (f32::from(e.position.x - origin.x), f32::from(e.position.y - origin.y));
        let pl = self.placement;
        if e.modifiers.control || e.modifiers.platform {
            self.zoom_about(pl.to_frame(v), (1.0 + delta.1 / 400.0).clamp(0.5, 2.0), cx);
        } else {
            self.state.update(cx, |s, cx| {
                if s.zoom.scale.is_some() {
                    s.pan_by(-delta.0 / (pl.fw * pl.s), -delta.1 / (pl.fh * pl.s), cx);
                }
            });
        }
    }
}

/// Opacity and image of the composition overlay, bottom right while it shows.
fn overlay_controls(state: &Entity<AppState>, opacity: f32, p: &crate::theme::Palette) -> impl IntoElement + use<> {
    let chip = |v: f32| {
        let st = state.clone();
        let on = (opacity - v).abs() < 0.05;
        div()
            .id(SharedString::from(format!("overlay-op-{v}")))
            .px_1p5()
            .rounded(px(3.0))
            .cursor_pointer()
            .text_color(hexa(0xffffff, if on { 0.95 } else { 0.55 }))
            .bg(hexa(0xffffff, if on { 0.16 } else { 0.0 }))
            .hover(|d| d.bg(hexa(0xffffff, 0.12)))
            .on_click(move |_, _, cx| {
                st.update(cx, |s, cx| {
                    s.settings.overlay_opacity = v;
                    s.settings.save();
                    cx.notify();
                })
            })
            .child(format!("{:.0}%", v * 100.0))
    };
    let st_hide = state.clone();
    div().absolute().bottom(px(12.0)).right(px(12.0)).child(
        h_flex()
            .gap_1()
            .px_2()
            .py_1()
            .rounded(px(6.0))
            .bg(hexa(0x141416, 0.82))
            .text_xs()
            .text_color(hexa(0xffffff, 0.8))
            .child(Icon::new(IconName::Layers2).xsmall().text_color(p.accent))
            .child(t("Overlay", "叠加图"))
            .child(chip(0.25))
            .child(chip(0.5))
            .child(chip(0.75))
            .child(chip(1.0))
            .child(
                div()
                    .id("overlay-choose")
                    .px_1()
                    .cursor_pointer()
                    .hover(|d| d.text_color(gpui_kit::white()))
                    .child(Icon::new(IconName::FolderOpen).xsmall())
                    .on_click(|_, window, cx| window.dispatch_action(Box::new(crate::actions::ChooseOverlay), cx)),
            )
            .child(
                div()
                    .id("overlay-hide")
                    .px_1()
                    .cursor_pointer()
                    .hover(|d| d.text_color(gpui_kit::white()))
                    .child(Icon::new(IconName::X).xsmall())
                    .on_click(move |_, _, cx| {
                        st_hide.update(cx, |s, cx| {
                            s.view.overlay = false;
                            cx.notify();
                        })
                    }),
            ),
    )
}

impl Render for Viewer {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        if crate::perf::enabled() {
            if let Some(frame) = &s.preview.frame {
                if frame.stamp.id != self.last_frame {
                    self.last_frame = frame.stamp.id;
                    let stamp = frame.stamp;
                    crate::perf::record(serde_json::json!({"event": "viewer", "id": stamp.id, "input_to_viewer_ms": stamp.requested_at.elapsed().as_secs_f64() * 1000.0}));
                    window.on_next_frame(move |_, _| crate::perf::record(serde_json::json!({
                        "event": "next_frame", "id": stamp.id, "quality": format!("{:?}", stamp.quality),
                        "input_to_next_frame_ms": stamp.requested_at.elapsed().as_secs_f64() * 1000.0,
                    })));
                }
            }
        }
        let gray = s.view.viewer_gray;
        // A slideshow or culling shows the image alone on black.
        let showing = self.state.read(cx).presenting();
        let bg = if showing { gpui_kit::black() } else { gpui_kit::hsla(0.0, 0.0, gray, 1.0) };
        let multi_ids = s.multi_images();
        if !multi_ids.is_empty() {
            return div()
                .id("viewer")
                .relative()
                .size_full()
                .overflow_hidden()
                .bg(bg)
                .child(Surface::new("viewer-surface-multi").absolute().size_full().on_prepaint_viewport(self.state.clone()))
                .child(multi::render(&self.state, &multi_ids, window, cx))
                .into_any_element();
        }
        let s = self.state.read(cx);
        let sf = s.scale_factor.max(0.5);
        let (fw, fh) = s.frame_size();
        let target_s = s.effective_scale() / sf;
        let target_c = if s.zoom.scale.is_none() { (0.5, 0.5) } else { s.zoom.center };
        let panning = matches!(self.drag, Some(Drag::Pan { .. }));
        let image_key = s.current.map(|i| i.0).unwrap_or(0);
        // Springs keyed per image so switching images snaps instead of gliding across.
        let anim_s = motion::follow(motion::key(format!("viewer-{image_key}"), "scale"), target_s, motion::VIEW, !panning, window, cx);
        let anim_cx = motion::follow(motion::key(format!("viewer-{image_key}"), "cx"), target_c.0, motion::VIEW, !panning, window, cx);
        let anim_cy = motion::follow(motion::key(format!("viewer-{image_key}"), "cy"), target_c.1, motion::VIEW, !panning, window, cx);
        // Exposure warning and focus mask fade with their toggles.
        let (warn_on, focus_on) = {
            let v = &self.state.read(cx).view;
            (v.exposure_warning, v.focus_mask)
        };
        let warn_t = motion::toggle(motion::key("viewer", "warn"), warn_on, motion::GLIDE, window, cx);
        let focus_t = motion::toggle(motion::key("viewer", "focus"), focus_on, motion::GLIDE, window, cx);
        let s = self.state.read(cx);
        self.placement = Placement { vw: s.viewport.0, vh: s.viewport.1, s: anim_s, cx: anim_cx, cy: anim_cy, fw, fh };
        let pl = self.placement;

        let frame_el = |frame: &crate::state::Frame| {
            let (l, t) = pl.to_view((frame.region.x, frame.region.y));
            let w = frame.region.width * fw * pl.s;
            let h = frame.region.height * fh * pl.s;
            let layer = |image: std::sync::Arc<gpui_kit::RenderImage>| img(image).absolute().size_full().object_fit(ObjectFit::Fill);
            div()
                .absolute()
                .left(px(l))
                .top(px(t))
                .w(px(w))
                .h(px(h))
                .child(layer(frame.image.clone()))
                .when_some(frame.warning.clone().filter(|_| warn_t > 0.01), |d, o| d.child(layer(o).opacity(warn_t)))
                .when_some(frame.focus.clone().filter(|_| focus_t > 0.01), |d, o| d.child(layer(o).opacity(focus_t)))
        };

        let has_image = s.current.is_some();
        let compare = s.view.compare;
        let split = s.view.split;
        let rendering = s.preview.rendering;
        let render_error = s.preview.error.as_ref().filter(|(id, _)| Some(*id) == s.current).map(|(_, e)| e.clone());
        let source_note = if s.current.is_some_and(|id| s.library_offline.contains(&id)) {
            Some(t("Original offline · cached preview; adjustments become visible when reconnected", "原片离线 · 正在显示缓存预览，调整将在原片连接后显示").into())
        } else { s.source_note.as_ref().filter(|(id, _)| Some(*id) == s.current).map(|(_, n)| n.clone()) };
        // Soft proofing uses the output profile of the selected process recipe, as Capture One does.
        let soft_proof = s.view.soft_proof.then(|| {
            let profile = s.recipes.get(s.selected_recipe).map(|r| r.icc_profile.clone()).filter(|p| !p.is_empty()).unwrap_or_else(|| "sRGB".into());
            (profile, crate::planned::PROOF_READY)
        });
        let after = s.preview.frame.as_ref().filter(|f| Some(f.image_id) == s.current).map(frame_el);
        let before = s.preview.before.as_ref().filter(|f| Some(f.image_id) == s.current).map(frame_el);
        let (rl, rt, rw, rh) = pl.frame_rect();
        let info = s.current_info().cloned();
        let cursor = s.cursor;
        let show_info = s.view.show_info_overlay && !s.presenting();
        let annotating = cursor == CursorTool::Annotate;
        let notes = s.current.and_then(|id| s.annotations.get(&id)).map_or(0, |n| n.len());

        let this = cx.entity();
        let (e1, e2, e3, e4) = (this.clone(), this.clone(), this.clone(), this.clone());
        let surface_id: SharedString = "viewer-surface".into();

        let cursor_style = match cursor {
            CursorTool::Pan => gpui_kit::CursorStyle::OpenHand,
            CursorTool::Select if s.zoom.scale.is_some() => gpui_kit::CursorStyle::OpenHand,
            CursorTool::Zoom => gpui_kit::CursorStyle::Arrow,
            CursorTool::Select => gpui_kit::CursorStyle::Arrow,
            _ => gpui_kit::CursorStyle::Crosshair,
        };

        let overlays = overlays::render(&self.state, &pl, self.guide, window, cx);
        // Composition overlay: a transparent image laid over the frame.
        let (overlay_on, overlay_path, overlay_opacity, showing) = {
            let s = self.state.read(cx);
            (s.view.overlay && s.settings.overlay_image.is_some(), s.settings.overlay_image.clone(), s.settings.overlay_opacity, s.presenting())
        };
        let overlay_t = motion::toggle(motion::key("viewer", "overlay"), overlay_on && !showing, motion::GLIDE, window, cx);
        let overlay_img = overlay_path.filter(|_| overlay_t > 0.01).map(|path| {
            img(path).absolute().left(px(rl)).top(px(rt)).w(px(rw)).h(px(rh)).object_fit(ObjectFit::Contain).opacity(overlay_opacity * overlay_t)
        });
        let overlay_bar = (overlay_on && !showing).then(|| overlay_controls(&self.state, overlay_opacity, &p));
        let loupe = self.loupe(window, cx);
        let hud = self.speed_hud(window, cx);
        let hint = {
            let s = self.state.read(cx);
            let layer = s.active_layer_ref().map(|l| l.name.to_string());
            let tool = s.cursor;
            tool_hint(tool, layer.as_deref()).filter(|_| has_image && !showing).map(|text| (tool, text))
        };

        div()
            .id("viewer")
            .relative()
            .size_full()
            .overflow_hidden()
            .bg(bg)
            .child(
                Surface::new(surface_id)
                    .absolute()
                    .size_full()
                    .cursor(cursor_style)
                    .on_down(move |p, window, cx| e1.update(cx, |v, cx| v.on_down(p, window, cx)))
                    .on_drag(move |p, window, cx| e2.update(cx, |v, cx| v.on_drag(p, window, cx)))
                    .on_up(move |window, cx| e3.update(cx, |v, cx| v.on_up(window, cx)))
                    .on_hover_move({
                        let e = this.clone();
                        move |p, _, cx| {
                            e.update(cx, |v, cx| {
                                v.hover = Some((f32::from(p.point.x - p.bounds.origin.x), f32::from(p.point.y - p.bounds.origin.y)));
                                if v.state.read(cx).cursor == CursorTool::Loupe {
                                    cx.notify();
                                }
                            })
                        }
                    })
                    .on_prepaint_viewport(self.state.clone())
                    .when(has_image, |this| {
                        this.child(
                            // Drop shadow under the image.
                            div().absolute().left(px(rl)).top(px(rt)).w(px(rw)).h(px(rh)).shadow_lg(),
                        )
                        .when_some(after, |this, a| {
                            this.child(motion::enter(SharedString::from(format!("viewer-img-{image_key}")), div().absolute().size_full().child(a), 0.0))
                        })
                        .when(compare == CompareMode::Split, |this| {
                            this.when_some(before, |this, b| {
                                this.child(div().absolute().top_0().left_0().bottom_0().w(px(split * pl.vw)).overflow_hidden().child(b))
                            })
                            .child(div().absolute().top_0().bottom_0().left(px(split * pl.vw - 1.0)).w(px(2.0)).bg(hexa(0xffffff, 0.85)))
                            .child(
                                div()
                                    .absolute()
                                    .top(px(pl.vh / 2.0 - 14.0))
                                    .left(px(split * pl.vw - 14.0))
                                    .size(px(28.0))
                                    .rounded_full()
                                    .bg(hexa(0xffffff, 0.92))
                                    .shadow_md()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .text_color(hexa(0x1c1c1d, 1.0))
                                    .child(Icon::new(IconName::ChevronsLeftRight).xsmall()),
                            )
                            .child(label_chip(t("Before", "调整前"), true, &p))
                            .child(label_chip(t("After", "调整后"), false, &p))
                        })
                        .child(overlays)
                        .when_some(overlay_img, |this, o| this.child(o))
                    }),
            )
            .when_some(overlay_bar, |this, bar| this.child(bar))
            .on_pinch({
                let this = this.clone();
                move |e, _, cx| this.update(cx, |v, cx| v.on_pinch(e, cx))
            })
            .on_scroll(move |e, _, cx| {
                // The viewer fills its parent, so the event's window position is offset by the viewer origin.
                e4.update(cx, |v, cx| {
                    let origin = v.surface_origin(cx);
                    v.on_wheel(e, origin, cx)
                })
            })
            .when(!has_image, |this| {
                this.child(
                    v_flex()
                        .size_full()
                        .items_center()
                        .justify_center()
                        .gap_2()
                        .text_color(p.text_subtle)
                        .child(Icon::new(IconName::ImagePlus).size(px(40.0)))
                        .child(div().text_sm().child(t("No image selected", "未选择图像")))
                        .child(div().text_xs().child(t("Pick one in the browser, or bring new photos in", "在浏览器中选一张，或导入新的照片")))
                        .child(
                            gpui_kit::component::button::Button::new("viewer-empty-import")
                                .small()
                                .outline()
                                .icon(Icon::new(IconName::ImageDown))
                                .label(format!("{}  {}", t("Import…", "导入…"), crate::platform::keys("ctrl-shift-i")))
                                .on_click(|_, window, cx| window.dispatch_action(Box::new(crate::actions::Import), cx)),
                        ),
                )
            })
            .when_some(soft_proof, |this, (profile, ready)| {
                this.child(
                    h_flex()
                        .absolute()
                        .top(px(10.0))
                        .left_1_2()
                        .gap_1p5()
                        .px_2()
                        .py_1()
                        .rounded(px(5.0))
                        .bg(hexa(0x000000, 0.7))
                        .text_xs()
                        .text_color(p.text)
                        .child(Icon::new(IconName::Printer).xsmall())
                        .child(format!("{} · {profile}", t("Soft proof", "软打样")))
                        .when(!ready, |d| d.child(div().text_color(p.warning).child(crate::contract::SOFT_PROOF.chip().unwrap_or_default()))),
                )
            })
            // The engine shows a stand-in for the full data (e.g. the embedded JPEG).
            .when_some(source_note, |this, note| {
                this.child(
                    h_flex()
                        .absolute()
                        .bottom(px(10.0))
                        .right(px(12.0))
                        .max_w(px(420.0))
                        .gap_1p5()
                        .px_2()
                        .py_1()
                        .rounded(px(5.0))
                        .bg(hexa(0x000000, 0.7))
                        .text_xs()
                        .text_color(p.warning)
                        .child(Icon::new(IconName::TriangleAlert).xsmall())
                        .child(div().child(note)),
                )
            })
            .when_some(render_error, |this, err| {
                // The last frame shown may be stale; say so instead of leaving it unexplained.
                this.child(
                    h_flex()
                        .absolute()
                        .top(px(10.0))
                        .left(px(12.0))
                        .max_w(px(560.0))
                        .gap_2()
                        .px_2()
                        .py_1()
                        .rounded(px(5.0))
                        .bg(hexa(0x000000, 0.7))
                        .text_xs()
                        .text_color(p.text)
                        .child(Icon::new(IconName::TriangleAlert).xsmall().text_color(p.danger))
                        .child(div().child(format!("{}: {err}", t("Preview could not be rendered", "预览渲染失败")))),
                )
            })
            .when(rendering, |this| {
                this.child(
                    div()
                        .absolute()
                        .top(px(10.0))
                        .right(px(12.0))
                        .p_1()
                        .rounded_full()
                        .bg(hexa(0x000000, 0.45))
                        .child(Spinner::new().xsmall().color(p.text_muted)),
                )
            })
            .when_some(info.filter(|_| has_image && show_info), |this, info| {
                let e = &info.exif;
                let exif = [
                    e.iso.map(|i| format!("ISO {i}")),
                    e.shutter.map(|s| if s >= 1.0 { format!("{s:.0}s") } else { format!("1/{:.0}", 1.0 / s) }),
                    e.aperture.map(|a| format!("f/{a:.1}")),
                    e.focal_length_mm.map(|f| format!("{f:.0}mm")),
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join("  ");
                let stars = if info.rating < 0 { "✕".to_string() } else { "★".repeat(info.rating as usize) };
                this.child(
                    h_flex()
                        .absolute()
                        .bottom(px(10.0))
                        .left(px(12.0))
                        .gap_3()
                        .px_2()
                        .py_0p5()
                        .rounded(px(4.0))
                        .bg(hexa(0x000000, 0.45))
                        .text_xs()
                        .text_color(hexa(0xffffff, 0.85))
                        .child(info.file_name.clone())
                        .when(info.variant > 1, |d| d.child(format!("{} {}", t("Variant", "变体"), info.variant)))
                        .when(!exif.is_empty(), |d| d.child(div().text_color(hexa(0xffffff, 0.6)).child(exif)))
                        .when(!stars.is_empty(), |d| d.child(div().text_color(p.warning).child(stars))),
                )
            })
            .when(annotating, |this| {
                let st = self.state.clone();
                this.child(
                    h_flex()
                        .absolute()
                        .top(px(10.0))
                        .left(px(12.0))
                        .gap_2()
                        .pl_2()
                        .pr_1()
                        .py_0p5()
                        .rounded(px(6.0))
                        .bg(hexa(0x000000, 0.6))
                        .text_xs()
                        .text_color(hexa(0xffffff, 0.85))
                        .child(Icon::new(IconName::PenLine).xsmall())
                        .child(format!("{} · {notes} {}", t("Draw notes on the image", "在图像上绘制标注"), t("notes", "条标注")))
                        .child(gpui_kit::component::button::Button::new("notes-clear").xsmall().ghost().label(t("Clear", "清除")).on_click(move |_, _, cx| {
                            st.update(cx, |s, cx| {
                                if let Some(id) = s.current {
                                    s.annotations.remove(&id);
                                }
                                cx.notify();
                            })
                        })),
                )
            })
            .when_some(hint, |this, (tool, text)| {
                let st = self.state.clone();
                this.child(
                    div().absolute().left_0().right_0().bottom(px(10.0)).flex().justify_center().child(motion::enter(
                        SharedString::from(format!("tool-hint-{tool:?}")),
                        h_flex()
                            .max_w(px(760.0))
                            .gap_2()
                            .pl_2()
                            .pr_1()
                            .py_0p5()
                            .rounded(px(6.0))
                            .bg(hexa(0x000000, 0.62))
                            .text_xs()
                            .text_color(hexa(0xffffff, 0.86))
                            .child(Icon::new(tool.icon()).xsmall().text_color(p.accent))
                            .child(div().font_weight(gpui_kit::FontWeight::SEMIBOLD).child(tool.label()))
                            .child(div().text_color(hexa(0xffffff, 0.66)).child(text))
                            .child(
                                gpui_kit::component::button::Button::new("tool-hint-done")
                                    .xsmall()
                                    .ghost()
                                    .label(format!("{}  V", t("Done", "完成")))
                                    .on_click(move |_, _, cx| st.update(cx, |s, cx| s.set_cursor(CursorTool::Select, cx))),
                            ),
                        6.0,
                    )),
                )
            })
            .when_some(loupe, |this, l| this.child(l))
            .when_some(hud, |this, h| this.child(h))
            .into_any_element()
    }
}

impl Viewer {
    /// The loupe: a magnified square that trails the pointer a few milliseconds
    /// behind, pops open when the tool is picked and grows while pressed.
    /// The Speed Edit readout: which parameter the held key adjusts, its value
    /// and where it sits in its range. Fades in once the key is clearly held.
    fn speed_hud(&mut self, window: &mut Window, cx: &mut Context<Self>) -> Option<gpui_kit::AnyElement> {
        let (live, visible) = {
            let s = self.state.read(cx);
            (s.speed.as_ref().map(|e| e.param), s.speed.as_ref().is_some_and(|e| e.visible()))
        };
        if let Some(param) = live {
            self.speed_shown = Some(param);
        }
        let shown = motion::toggle(motion::key("speed-hud", "in"), visible, motion::QUICK, window, cx);
        let param = self.speed_shown?;
        if shown < 0.01 {
            return None;
        }
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let (min, max, decimals) = s.speed_meta(param);
        let value = s.speed_value(param);
        let t_val = motion::value(motion::key("speed-hud", "t"), (value - min) / (max - min).max(1e-6), motion::QUICK, window, cx);
        let origin = if min < 0.0 && max > 0.0 { -min / (max - min) } else { 0.0 };
        let (lo, hi) = (origin.min(t_val), origin.max(t_val));
        let unit = if param == SpeedParam::Exposure { " EV" } else { "" };
        let text = if param == SpeedParam::Exposure { format!("{value:+.2}{unit}") } else { format!("{value:.decimals$}") };
        const BAR: f32 = 140.0;
        Some(
            div()
                .absolute()
                .top(px(18.0 - 6.0 * (1.0 - shown)))
                .left_0()
                .right_0()
                .flex()
                .justify_center()
                .opacity(shown)
                .child(
                    h_flex()
                        .gap_2()
                        .px_3()
                        .py_1p5()
                        .rounded(px(8.0))
                        .bg(hexa(0x141416, 0.86))
                        .border_1()
                        .border_color(hexa(0xffffff, 0.08))
                        .shadow_lg()
                        .text_sm()
                        .text_color(hexa(0xffffff, 0.92))
                        .child(
                            div()
                                .px_1p5()
                                .rounded(px(3.0))
                                .bg(hexa(0xffffff, 0.12))
                                .text_xs()
                                .child(param.key().to_uppercase()),
                        )
                        .child(param.label())
                        .child(
                            div()
                                .relative()
                                .w(px(BAR))
                                .h(px(4.0))
                                .rounded_full()
                                .bg(hexa(0xffffff, 0.14))
                                .child(div().absolute().top_0().h_full().rounded_full().bg(p.accent).left(px(lo * BAR)).w(px((hi - lo) * BAR)))
                                .child(div().absolute().top(px(-3.0)).left(px(t_val * BAR - 1.0)).w(px(2.0)).h(px(10.0)).rounded_full().bg(gpui_kit::white())),
                        )
                        .child(div().w(px(64.0)).flex().justify_end().child(text)),
                )
                .into_any_element(),
        )
    }

    fn loupe(&self, window: &mut Window, cx: &mut Context<Self>) -> Option<gpui_kit::AnyElement> {
        let s = self.state.read(cx);
        let on = s.cursor == CursorTool::Loupe && s.current.is_some() && self.hover.is_some();
        let pl = self.placement;
        let src = s.preview.frame.as_ref().filter(|f| Some(f.image_id) == s.current).map(|f| {
            let (l, t) = pl.to_view((f.region.x, f.region.y));
            (f.image.clone(), l, t, f.region.width * pl.fw * pl.s, f.region.height * pl.fh * pl.s)
        });
        let shown = s.effective_scale() / s.scale_factor.max(0.5);
        let open = motion::toggle(motion::key("loupe", "open"), on, motion::FLOW, window, cx);
        let size = motion::value(motion::key("loupe", "size"), if self.loupe_down { 300.0 } else { 200.0 }, motion::FLOW, window, cx);
        let mag = motion::value(motion::key("loupe", "mag"), if self.loupe_down { 4.0 } else { 2.0 }, motion::GLIDE, window, cx);
        let (hx, hy) = self.hover.unwrap_or((pl.vw / 2.0, pl.vh / 2.0));
        let x = motion::value(motion::key("loupe", "x"), hx, motion::QUICK, window, cx);
        let y = motion::value(motion::key("loupe", "y"), hy, motion::QUICK, window, cx);
        if open < 0.01 {
            return None;
        }
        let (image, l, t, w, h) = src?;
        let sz = size * (0.7 + 0.3 * open);
        let p = pal(cx);
        Some(
            div()
                .absolute()
                .left(px(x - sz / 2.0))
                .top(px(y - sz / 2.0))
                .size(px(sz))
                .opacity(open)
                // GPUI clips children to rectangles, so the loupe is a square with
                // only a slight rounding the clip can follow.
                .rounded(px(4.0))
                .overflow_hidden()
                .border_2()
                .border_color(hexa(0xffffff, 0.9))
                .shadow_lg()
                .bg(hexa(0x000000, 0.9))
                .child(
                    img(image)
                        .absolute()
                        .left(px(sz / 2.0 - (x - l) * mag))
                        .top(px(sz / 2.0 - (y - t) * mag))
                        .w(px(w * mag))
                        .h(px(h * mag))
                        .object_fit(ObjectFit::Fill),
                )
                .child(div().absolute().left(px(sz / 2.0 - 6.0)).top(px(sz / 2.0)).w(px(12.0)).h(px(1.0)).bg(hexa(0xffffff, 0.6)))
                .child(div().absolute().left(px(sz / 2.0)).top(px(sz / 2.0 - 6.0)).w(px(1.0)).h(px(12.0)).bg(hexa(0xffffff, 0.6)))
                .child(
                    div()
                        .absolute()
                        .bottom(px(sz * 0.12))
                        .left(px(sz / 2.0 - 26.0))
                        .w(px(52.0))
                        .flex()
                        .justify_center()
                        .child(
                            div().px_1p5().rounded(px(8.0)).bg(hexa(0x000000, 0.6)).text_xs().text_color(p.text).child(format!("{:.0}%", shown * mag * 100.0)),
                        ),
                )
                .into_any_element(),
        )
    }
}

/// What the active cursor tool does with a click or drag, shown under the image
/// so a tool never has to be discovered by trial.
fn tool_hint(tool: CursorTool, layer: Option<&str>) -> Option<String> {
    let on_layer = |what: &str| match layer {
        Some(name) => format!("{what} · {} “{name}”", t("on layer", "作用于图层")),
        None => what.to_string(),
    };
    Some(match tool {
        CursorTool::Select | CursorTool::Annotate => return None,
        CursorTool::Pan => t("Drag to pan · double-click toggles 100%", "拖动平移 · 双击在 100% 与适合窗口间切换").to_string(),
        CursorTool::Zoom => format!("{} · {}{}", t("Click to zoom in", "单击放大"), crate::platform::alt(), t("-click to zoom out", "+单击缩小")),
        CursorTool::Crop => t("Drag to draw a crop · drag its edges or corners to adjust · drag inside to move", "拖动画出裁剪框 · 拖动边或角调整 · 在框内拖动可移动").to_string(),
        CursorTool::Straighten => t("Drag along a line that should be level or upright", "沿着应当水平或垂直的线拖动").to_string(),
        CursorTool::Keystone => t("Drag up or down for vertical, left or right for horizontal correction", "上下拖动校正垂直透视 · 左右拖动校正水平透视").to_string(),
        CursorTool::Spot => t("Click a blemish to remove it", "单击要去除的污点").to_string(),
        CursorTool::Brush => on_layer(t("Paint to add to the mask", "涂抹以绘制蒙版")),
        CursorTool::Eraser => on_layer(t("Paint to erase the mask", "涂抹以擦除蒙版")),
        CursorTool::LinearGradient => on_layer(t("Drag from full effect to none", "从完全生效处拖向不生效处")),
        CursorTool::RadialGradient => on_layer(t("Drag outward from the center", "从中心向外拖动")),
        CursorTool::ColorPicker => t("Click a color to sample it", "单击图像中的颜色取样").to_string(),
        CursorTool::WhiteBalance => t("Click something that should be neutral gray", "单击应为中性灰的区域").to_string(),
        CursorTool::Loupe => t("Move to inspect · hold the button to magnify more", "移动查看细节 · 按住鼠标进一步放大").to_string(),
    })
}

fn label_chip(text: &'static str, left: bool, p: &crate::theme::Palette) -> impl IntoElement + use<> {
    div()
        .absolute()
        .top(px(10.0))
        .when(left, |d| d.left(px(12.0)))
        .when(!left, |d| d.right(px(44.0)))
        .px_2()
        .py_0p5()
        .rounded(px(4.0))
        .bg(hexa(0x000000, 0.5))
        .text_xs()
        .text_color(p.text)
        .child(text)
}

impl Viewer {
    fn surface_origin(&self, cx: &App) -> gpui_kit::Point<Pixels> {
        self.state.read(cx).viewport_origin
    }
}

trait ViewportReport {
    fn on_prepaint_viewport(self, state: Entity<AppState>) -> Self;
}

impl ViewportReport for Surface {
    fn on_prepaint_viewport(self, state: Entity<AppState>) -> Self {
        use gpui_kit::base::ElementExt as _;
        self.on_prepaint(move |bounds, window, cx| {
            let sf = window.scale_factor();
            let changed = {
                let s = state.read(cx);
                s.viewport_origin != bounds.origin
                    || (f32::from(bounds.size.width) - s.viewport.0).abs() > 0.5
                    || (f32::from(bounds.size.height) - s.viewport.1).abs() > 0.5
                    || s.scale_factor != sf
            };
            if changed {
                state.update(cx, |s, cx| {
                    s.viewport_origin = bounds.origin;
                    s.set_viewport((bounds.size.width, bounds.size.height), sf, cx);
                });
            }
        })
    }
}
