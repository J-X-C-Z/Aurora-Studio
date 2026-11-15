//! The slideshow screen: the viewer alone on black, each new image fading up
//! as its preview arrives, with quiet controls along the bottom that come
//! forward under the pointer. Space plays or pauses, ← → step, Esc ends.

use gpui_kit::assets::IconName;
use gpui_kit::component::{Icon, Sizable as _, h_flex};
use gpui_kit::{
    AnyElement, App, Entity, InteractiveElement, IntoElement, ParentElement, SharedString, StatefulInteractiveElement, Styled, Window, div, px,
};

use super::viewer::Viewer;
use crate::i18n::t;
use crate::motion;
use crate::state::AppState;
use crate::theme::hexa;

fn control(id: &'static str, icon: IconName, tip: &'static str, on: impl Fn(&mut Window, &mut App) + 'static) -> impl IntoElement {
    div()
        .id(id)
        .size(px(34.0))
        .rounded_full()
        .flex()
        .items_center()
        .justify_center()
        .text_color(hexa(0xffffff, 0.9))
        .cursor_pointer()
        .hover(|d| d.bg(hexa(0xffffff, 0.14)))
        .tooltip(move |window, cx| gpui_kit::component::tooltip::Tooltip::new(tip).build(window, cx))
        .on_click(move |_, window, cx| on(window, cx))
        .child(Icon::new(icon).small())
}

pub fn render(state: &Entity<AppState>, viewer: &Entity<Viewer>, _window: &mut Window, cx: &mut App) -> AnyElement {
    let s = state.read(cx);
    let Some(show) = s.slideshow.clone() else { return div().into_any_element() };
    let ids: Vec<_> = s.visible_images().iter().map(|i| i.id).collect();
    let at = s.current.and_then(|c| ids.iter().position(|i| *i == c)).map(|i| i + 1).unwrap_or(0);
    let name = s.current_info().map(|i| i.file_name.clone()).unwrap_or_default();
    // Fade each image up when its own preview arrives, not when it is selected,
    // so the previous picture never flashes at full strength.
    let frame_id = s.preview.frame.as_ref().map(|f| f.image_id.0).unwrap_or(0);

    let (st_prev, st_play, st_next, st_exit) = (state.clone(), state.clone(), state.clone(), state.clone());
    let pace = |secs: f32| {
        let st = state.clone();
        let on = (show.interval - secs).abs() < 0.1;
        div()
            .id(SharedString::from(format!("show-pace-{secs}")))
            .px_2()
            .py_0p5()
            .rounded(px(10.0))
            .text_xs()
            .cursor_pointer()
            .text_color(hexa(0xffffff, if on { 0.95 } else { 0.55 }))
            .bg(hexa(0xffffff, if on { 0.16 } else { 0.0 }))
            .hover(|d| d.bg(hexa(0xffffff, 0.12)))
            .on_click(move |_, _, cx| st.update(cx, |s, cx| s.set_slideshow_interval(secs, cx)))
            .child(format!("{secs:.0} s"))
    };

    div()
        .id("slideshow")
        .size_full()
        .relative()
        .bg(gpui_kit::black())
        .child(motion::enter(SharedString::from(format!("slide-{frame_id}")), div().absolute().size_full().child(viewer.clone()), 0.0))
        .child(
            div().absolute().bottom(px(28.0)).left_0().right_0().flex().justify_center().child(
                h_flex()
                    .id("slideshow-controls")
                    .gap_1()
                    .px_3()
                    .py_1p5()
                    .rounded(px(24.0))
                    .bg(hexa(0x000000, 0.55))
                    .border_1()
                    .border_color(hexa(0xffffff, 0.08))
                    .opacity(0.35)
                    .hover(|d| d.opacity(1.0))
                    .child(control("show-prev", IconName::ChevronLeft, t("Previous (←)", "上一张（←）"), move |_, cx| st_prev.update(cx, |s, cx| s.slide(-1, cx))))
                    .child(control(
                        "show-play",
                        if show.playing { IconName::Pause } else { IconName::Play },
                        t("Play / pause (Space)", "播放 / 暂停（空格）"),
                        move |_, cx| st_play.update(cx, |s, cx| s.toggle_slideshow_play(cx)),
                    ))
                    .child(control("show-next", IconName::ChevronRight, t("Next (→)", "下一张（→）"), move |_, cx| st_next.update(cx, |s, cx| s.slide(1, cx))))
                    .child(div().w(px(1.0)).h(px(18.0)).mx_1().bg(hexa(0xffffff, 0.15)))
                    .child(pace(3.0))
                    .child(pace(5.0))
                    .child(pace(8.0))
                    .child(div().w(px(1.0)).h(px(18.0)).mx_1().bg(hexa(0xffffff, 0.15)))
                    .child(div().px_2().text_xs().text_color(hexa(0xffffff, 0.75)).child(format!("{at} / {}  {name}", ids.len())))
                    .child(control("show-exit", IconName::X, t("End slideshow (Esc)", "结束放映（Esc）"), move |window, cx| {
                        st_exit.update(cx, |s, cx| s.stop_slideshow(window, cx))
                    })),
            ),
        )
        .into_any_element()
}
