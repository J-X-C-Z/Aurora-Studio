//! The Live View window: the camera's live feed with focus, overlays and capture.

use std::sync::Arc;
use std::time::Duration;

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Disableable as _, Icon, TitleBar,
    button::{Button, ButtonVariants as _},
    h_flex, v_flex,
};
use gpui_kit::{
    AppContext as _, Context, Entity, IntoElement, ObjectFit, ParentElement, Render, RenderImage, Styled, StyledImage as _,
    Subscription, Task, Window, WindowBounds, WindowOptions, canvas, div, img, prelude::FluentBuilder as _, px, size,
};
use lumen_engine::CameraStatus;

use crate::i18n::t;
use crate::state::AppState;
use crate::state::render::to_render_image;
use crate::theme::{hexa, pal};
use crate::widgets::{IconBtn, paint};

pub struct LiveView {
    state: Entity<AppState>,
    frame: Option<Arc<RenderImage>>,
    supported: bool,
    grid: bool,
    paused: bool,
    fps: f32,
    _poll: Task<()>,
    _sub: Subscription,
}

impl LiveView {
    fn new(state: Entity<AppState>, cx: &mut Context<Self>) -> Self {
        let poll = cx.spawn(async move |this, cx| {
            loop {
                let started = std::time::Instant::now();
                let Some((engine, paused)) = this.update(cx, |v, cx| (v.state.read(cx).engine.clone(), v.paused)).ok() else { break };
                if !paused {
                    let frame = cx.background_spawn(async move { engine.live_view_frame(1280).map(|f| to_render_image(&f)) }).await;
                    let alive = this
                        .update(cx, |v, cx| {
                            v.supported = frame.is_some();
                            if let Some(f) = frame {
                                if let Some(old) = v.frame.replace(f) {
                                    cx.drop_image(old, None);
                                }
                                let dt = started.elapsed().as_secs_f32().max(1e-3);
                                v.fps = v.fps * 0.8 + (1.0 / dt.max(1.0 / 60.0)).min(60.0) * 0.2;
                            }
                            cx.notify();
                        })
                        .is_ok();
                    if !alive {
                        break;
                    }
                }
                cx.background_executor().timer(Duration::from_millis(66)).await;
            }
        });
        Self {
            _sub: cx.observe(&state, |_, _, cx| cx.notify()),
            state,
            frame: None,
            supported: true,
            grid: true,
            paused: false,
            fps: 0.0,
            _poll: poll,
        }
    }
}

impl Render for LiveView {
    fn render(&mut self, _: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let p = pal(cx).clone();
        let s = self.state.read(cx);
        let camera = match &s.camera {
            CameraStatus::Connected { model, .. } => Some(model.clone()),
            CameraStatus::Disconnected => None,
        };
        let capturing = s.capturing;
        let settings = s.camera_settings.clone();
        let this = cx.entity();
        let (e1, e2) = (this.clone(), this.clone());
        let st = self.state.clone();
        let focus = |steps: i32, icon: IconName, tip: &'static str| {
            let st = st.clone();
            IconBtn::new(format!("lv-focus-{steps}"), icon, tip).on_click(move |_, cx| {
                let engine = st.read(cx).engine.clone();
                engine.live_view_focus(steps);
            })
        };
        let grid = self.grid;

        v_flex()
            .size_full()
            .bg(p.chrome)
            .text_color(p.text)
            .child(TitleBar::new().child(div().text_xs().text_color(p.text_muted).child(format!("{} — {}", t("Live View", "实时取景"), camera.clone().unwrap_or_else(|| t("No camera", "无相机").into())))))
            .child(
                div()
                    .relative()
                    .flex_1()
                    .min_h_0()
                    .bg(hexa(0x000000, 1.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .when_some(self.frame.clone().filter(|_| self.supported), |d, f| d.child(img(f).size_full().object_fit(ObjectFit::Contain)))
                    .when(!self.supported || camera.is_none(), |d| {
                        d.child(
                            v_flex()
                                .items_center()
                                .gap_2()
                                .text_color(p.text_subtle)
                                .child(Icon::new(IconName::VideoOff).size(px(40.0)))
                                .child(div().text_sm().child(if camera.is_none() {
                                    t("Connect a camera to use Live View", "连接相机后可使用实时取景")
                                } else {
                                    t("This camera or engine does not provide a live feed", "当前相机或引擎不提供实时画面")
                                })),
                        )
                    })
                    .when(grid && self.supported && self.frame.is_some(), |d| {
                        d.child(
                            canvas(
                                |_, _, _| (),
                                |b, _, window, _| {
                                    for i in 1..3 {
                                        let x = b.origin.x + b.size.width * (i as f32 / 3.0);
                                        let y = b.origin.y + b.size.height * (i as f32 / 3.0);
                                        paint::segment(window, gpui_kit::point(x, b.origin.y), gpui_kit::point(x, b.origin.y + b.size.height), 1.0, hexa(0xffffff, 0.35));
                                        paint::segment(window, gpui_kit::point(b.origin.x, y), gpui_kit::point(b.origin.x + b.size.width, y), 1.0, hexa(0xffffff, 0.35));
                                    }
                                },
                            )
                            .absolute()
                            .size_full(),
                        )
                    }),
            )
            .child(
                h_flex()
                    .h(px(52.0))
                    .px_3()
                    .gap_2()
                    .border_t_1()
                    .border_color(p.divider)
                    .child(focus(-10, IconName::ChevronsLeft, t("Focus nearer (large step)", "向近对焦（大步）")))
                    .child(focus(-1, IconName::ChevronLeft, t("Focus nearer", "向近对焦")))
                    .child(focus(1, IconName::ChevronRight, t("Focus farther", "向远对焦")))
                    .child(focus(10, IconName::ChevronsRight, t("Focus farther (large step)", "向远对焦（大步）")))
                    .child(div().w(px(1.0)).h(px(22.0)).bg(p.divider))
                    .child(IconBtn::new("lv-grid", IconName::Grid3x3, t("Grid", "网格")).active(grid).on_click(move |_, cx| {
                        e1.update(cx, |v, cx| {
                            v.grid = !v.grid;
                            cx.notify();
                        })
                    }))
                    .child(IconBtn::new("lv-pause", if self.paused { IconName::Play } else { IconName::Pause }, t("Pause feed", "暂停画面")).active(self.paused).on_click(move |_, cx| {
                        e2.update(cx, |v, cx| {
                            v.paused = !v.paused;
                            cx.notify();
                        })
                    }))
                    .when_some(settings, |d, c| d.child(div().text_xs().text_color(p.text_muted).child(format!("{}   {}   {}   {:+.1} EV", c.iso, c.shutter, c.aperture, c.exposure_comp))))
                    .child(div().flex_1())
                    .when(self.supported && self.frame.is_some(), |d| d.child(div().text_xs().text_color(p.text_subtle).child(format!("{:.0} fps", self.fps))))
                    .child(
                        Button::new("lv-capture")
                            .primary()
                            .loading(capturing)
                            .disabled(camera.is_none())
                            .icon(Icon::new(IconName::Aperture))
                            .label(t("Capture", "拍摄"))
                            .on_click({
                                let st = self.state.clone();
                                move |_, _, cx| st.update(cx, |s, cx| s.capture(cx))
                            }),
                    ),
            )
    }
}

pub fn open(state: &Entity<AppState>, cx: &mut gpui_kit::App) {
    let state = state.clone();
    let options = WindowOptions {
        window_bounds: Some(WindowBounds::centered(size(px(1000.0), px(720.0)), cx)),
        window_min_size: Some(size(px(560.0), px(400.0))),
        ..TitleBar::window_options()
    };
    if let Err(err) = gpui_kit::open_window(options, cx, move |window, cx| {
        window.set_window_title(t("Live View", "实时取景"));
        cx.new(|cx| LiveView::new(state, cx))
    }) {
        log::error!("live view: window failed to open: {err}");
    }
}
