//! Color balance wheel: drag the puck to choose a hue (angle) and strength (radius).
//! Double-click resets it. The wheel image is generated once and shared.

use std::rc::Rc;
use std::sync::{Arc, OnceLock};

use gpui_kit::{
    App, IntoElement, ObjectFit, ParentElement, RenderImage, RenderOnce, SharedString, Styled, StyledImage as _, Window,
    div, img, prelude::FluentBuilder as _, px, relative,
};
use lumen_engine::ColorWheel;

use super::surface::{Pointer, Surface};
use crate::motion;
use crate::theme::{hexa, hue_color, pal};

fn wheel_image() -> Arc<RenderImage> {
    static IMG: OnceLock<Arc<RenderImage>> = OnceLock::new();
    IMG.get_or_init(|| {
        let n = 256u32;
        let mut px_data = vec![0u8; (n * n * 4) as usize];
        for y in 0..n {
            for x in 0..n {
                let dx = (x as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                let dy = (y as f32 + 0.5) / n as f32 * 2.0 - 1.0;
                let r = (dx * dx + dy * dy).sqrt();
                let alpha = ((1.0 - r) * n as f32 / 2.0).clamp(0.0, 1.0);
                let hue = (-dy).atan2(dx).to_degrees().rem_euclid(360.0);
                let c = hsv(hue, (r * 0.85).min(0.85), 0.62 + (1.0 - r) * 0.08);
                let i = ((y * n + x) * 4) as usize;
                // BGRA for GPUI.
                px_data[i] = (c[2] * 255.0) as u8;
                px_data[i + 1] = (c[1] * 255.0) as u8;
                px_data[i + 2] = (c[0] * 255.0) as u8;
                px_data[i + 3] = (alpha * 255.0) as u8;
            }
        }
        let buf = image::RgbaImage::from_raw(n, n, px_data).expect("wheel buffer size");
        Arc::new(RenderImage::new(vec![image::Frame::new(buf)]))
    })
    .clone()
}

fn hsv(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = match hp as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m]
}

type OnChange = Rc<dyn Fn(ColorWheel, &mut Window, &mut App)>;
type OnCommit = Rc<dyn Fn(&mut Window, &mut App)>;

#[derive(IntoElement)]
pub struct ColorWheelView {
    id: SharedString,
    value: ColorWheel,
    size: f32,
    title: Option<SharedString>,
    on_change: Option<OnChange>,
    on_commit: Option<OnCommit>,
}

impl ColorWheelView {
    pub fn new(id: impl Into<SharedString>, value: ColorWheel) -> Self {
        Self { id: id.into(), value, size: 120.0, title: None, on_change: None, on_commit: None }
    }

    pub fn size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    pub fn title(mut self, t: impl Into<SharedString>) -> Self {
        self.title = Some(t.into());
        self
    }

    pub fn on_change(mut self, f: impl Fn(ColorWheel, &mut Window, &mut App) + 'static) -> Self {
        self.on_change = Some(Rc::new(f));
        self
    }

    pub fn on_commit(mut self, f: impl Fn(&mut Window, &mut App) + 'static) -> Self {
        self.on_commit = Some(Rc::new(f));
        self
    }
}

fn to_wheel(p: &Pointer, current: ColorWheel) -> ColorWheel {
    let dx = p.pos.0 * 2.0 - 1.0;
    let dy = p.pos.1 * 2.0 - 1.0;
    let r = (dx * dx + dy * dy).sqrt().min(1.0);
    ColorWheel { hue: (-dy).atan2(dx).to_degrees().rem_euclid(360.0), saturation: r, lightness: current.lightness }
}

impl RenderOnce for ColorWheelView {
    fn render(self, window: &mut Window, cx: &mut App) -> impl IntoElement {
        let p = pal(cx).clone();
        let v = self.value;
        let (tx, ty) = {
            let a = v.hue.to_radians();
            (0.5 + a.cos() * v.saturation * 0.5, 0.5 - a.sin() * v.saturation * 0.5)
        };
        let surface = Surface::state(&self.id, window, cx);
        let dragging = surface.read(cx).dragging;
        let x = motion::follow(motion::key(&self.id, "x"), tx, motion::GLIDE, !dragging, window, cx);
        let y = motion::follow(motion::key(&self.id, "y"), ty, motion::GLIDE, !dragging, window, cx);
        let puck = motion::value(motion::key(&self.id, "puck"), if dragging { 14.0 } else { 11.0 }, motion::QUICK, window, cx);
        let size = self.size;
        let changed = v.saturation > 0.001;

        let on_down = {
            let f = self.on_change.clone();
            let c = self.on_commit.clone();
            move |ptr: Pointer, window: &mut Window, cx: &mut App| {
                if ptr.clicks >= 2 {
                    if let Some(f) = &f {
                        f(ColorWheel { lightness: v.lightness, ..Default::default() }, window, cx);
                    }
                    if let Some(c) = &c {
                        c(window, cx);
                    }
                    return;
                }
                if let Some(f) = &f {
                    f(to_wheel(&ptr, v), window, cx);
                }
            }
        };
        let on_move = {
            let f = self.on_change.clone();
            move |ptr: Pointer, window: &mut Window, cx: &mut App| {
                if let Some(f) = &f {
                    f(to_wheel(&ptr, v), window, cx);
                }
            }
        };
        let on_up = {
            let c = self.on_commit.clone();
            move |window: &mut Window, cx: &mut App| {
                if let Some(c) = &c {
                    c(window, cx);
                }
            }
        };

        let swatch = hue_color(v.hue, v.saturation.min(1.0) * 0.8, 0.55);
        gpui_kit::component::v_flex()
            .items_center()
            .gap_1()
            .when_some(self.title.clone(), |this, t| {
                this.child(div().text_xs().text_color(if changed { p.text } else { p.text_muted }).child(t))
            })
            .child(
                Surface::new(self.id.clone())
                    .size(px(size))
                    .rounded_full()
                    .cursor_crosshair()
                    .on_down(on_down)
                    .on_drag(on_move)
                    .on_up(on_up)
                    .child(img(wheel_image()).absolute().size_full().object_fit(ObjectFit::Fill))
                    .child(
                        div()
                            .absolute()
                            .top(relative(0.5))
                            .left(relative(0.5))
                            .ml(px(-0.5))
                            .mt(px(-4.0))
                            .w(px(1.0))
                            .h(px(8.0))
                            .bg(hexa(0x000000, 0.35)),
                    )
                    .child(
                        div()
                            .absolute()
                            .top(relative(0.5))
                            .left(relative(0.5))
                            .ml(px(-4.0))
                            .mt(px(-0.5))
                            .w(px(8.0))
                            .h(px(1.0))
                            .bg(hexa(0x000000, 0.35)),
                    )
                    .child(
                        div()
                            .absolute()
                            .left(relative(x))
                            .top(relative(y))
                            .ml(px(-puck / 2.0))
                            .mt(px(-puck / 2.0))
                            .size(px(puck))
                            .rounded_full()
                            .border_2()
                            .border_color(gpui_kit::white())
                            .bg(swatch)
                            .shadow_md(),
                    ),
            )
    }
}
