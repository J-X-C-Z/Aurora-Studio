//! A second viewer window for another screen: the current image on black with
//! its name and rating, following whatever the main window shows. It mirrors
//! the main viewer's latest preview, so a zoomed view is mirrored zoomed.

use gpui_kit::assets::IconName;
use gpui_kit::component::{Icon, h_flex};
use gpui_kit::{
    AppContext as _, Context, Entity, FocusHandle, InteractiveElement, IntoElement, KeyDownEvent, ObjectFit, ParentElement, Render, SharedString, Styled, StyledImage as _, Subscription, Window,
    div, img, prelude::FluentBuilder as _, px,
};

use crate::i18n::t;
use crate::motion;
use crate::state::AppState;
use crate::theme::hexa;

pub struct SecondViewer {
    state: Entity<AppState>,
    focus: FocusHandle,
    _sub: Subscription,
}

impl SecondViewer {
    fn on_key(&mut self, e: &KeyDownEvent, window: &mut Window, cx: &mut Context<Self>) {
        let k = &e.keystroke;
        if k.modifiers.modified() {
            return;
        }
        match k.key.as_str() {
            "left" | "up" => self.state.update(cx, |s, cx| s.step(-1, cx)),
            "right" | "down" | "space" => self.state.update(cx, |s, cx| s.step(1, cx)),
            "escape" => window.remove_window(),
            d if d.len() == 1 && ("0"..="5").contains(&d) => {
                let rating = d.parse().unwrap_or(0);
                self.state.update(cx, |s, cx| s.set_rating(rating, cx));
            }
            _ => return,
        }
        cx.stop_propagation();
    }
}

impl Render for SecondViewer {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let focus = self.focus.clone();
        let view = window.viewport_size();
        let area = gpui_kit::size((f32::from(view.width) - 48.0).max(1.0), (f32::from(view.height) - 80.0).max(1.0));
        let s = self.state.read(cx);
        let current = s.current;
        // The preview once it is of this image; its thumbnail until then.
        let (picture, frame_id) = match s.preview.frame.as_ref().filter(|f| Some(f.image_id) == current) {
            Some(f) => (Some(f.image.clone()), f.image_id.0),
            None => (current.and_then(|id| s.thumbs.get(&id).cloned()), 0),
        };
        let info = s.current_info().cloned();
        let ids: Vec<_> = s.visible_images().iter().map(|i| i.id).collect();
        let at = current.and_then(|c| ids.iter().position(|i| *i == c)).map(|i| i + 1).unwrap_or(0);
        let name = info.as_ref().map(|i| i.file_name.clone()).unwrap_or_default();
        let rating = info.as_ref().map(|i| i.rating).unwrap_or(0);

        div()
            .size_full()
            .relative()
            .bg(gpui_kit::black())
            .track_focus(&focus)
            .on_key_down(cx.listener(Self::on_key))
            .when_some(picture, |d, pic| {
                // Fitted by hand: `ObjectFit::Contain` on a full-size image crops
                // portrait frames instead of letterboxing them.
                let px_size = pic.size(0);
                let (iw, ih) = (px_size.width.0.max(1) as f32, px_size.height.0.max(1) as f32);
                let fit = (area.width / iw).min(area.height / ih);
                let (w, h) = (iw * fit, ih * fit);
                d.child(motion::enter(
                    SharedString::from(format!("mirror-{}-{frame_id}", current.map(|c| c.0).unwrap_or(0))),
                    div().absolute().top(px(24.0)).bottom(px(56.0)).left(px(24.0)).right(px(24.0)).flex().items_center().justify_center().child(
                        img(pic).w(px(w)).h(px(h)).object_fit(ObjectFit::Fill),
                    ),
                    0.0,
                ))
            })
            .when(current.is_none(), |d| {
                d.flex().items_center().justify_center().text_color(hexa(0xffffff, 0.4)).child(t("No image selected", "未选择图像"))
            })
            .when(current.is_some(), |d| {
                d.child(
                    h_flex()
                        .absolute()
                        .bottom(px(16.0))
                        .left_0()
                        .right_0()
                        .justify_center()
                        .gap_3()
                        .text_sm()
                        .text_color(hexa(0xffffff, 0.75))
                        .child(name)
                        .child(
                            h_flex().gap(px(2.0)).children((1..=5).map(|i| {
                                Icon::new(if rating < 0 { IconName::X } else { IconName::Star })
                                    .size(px(11.0))
                                    .text_color(if rating < 0 { hexa(0xe5605a, 0.9) } else if i <= rating { hexa(0xf2c14e, 1.0) } else { hexa(0xffffff, 0.2) })
                            })),
                        )
                        .child(div().text_color(hexa(0xffffff, 0.45)).child(format!("{at} / {}", ids.len()))),
                )
            })
    }
}

pub fn open(state: &Entity<AppState>, cx: &mut gpui_kit::App) {
    let options = gpui_kit::WindowOptions {
        window_bounds: Some(gpui_kit::WindowBounds::centered(gpui_kit::size(px(1200.0), px(800.0)), cx)),
        ..Default::default()
    };
    let state = state.clone();
    let opened = gpui_kit::open_window(options, cx, move |window, cx| {
        window.set_window_title(t("Aurora Studio — Second Viewer", "Aurora Studio — 第二查看器"));
        let view = cx.new(|cx| SecondViewer { _sub: cx.observe(&state, |_, _, cx| cx.notify()), focus: cx.focus_handle(), state: state.clone() });
        window.focus(&view.read(cx).focus.clone(), cx);
        view
    });
    if let Err(e) = opened {
        log::error!("second viewer: {e}");
    }
}
