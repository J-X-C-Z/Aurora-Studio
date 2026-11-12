//! The culling screen: the viewer alone on black with the image's rating along
//! the top and its neighbors along the bottom. Keys: 1–5 rate (again to clear),
//! X or - reject, 0 clear, ← → step, A auto-advance, U unrated only, Z fit/100 %,
//! Esc ends.

use gpui_kit::assets::IconName;
use gpui_kit::component::{Icon, Sizable as _, h_flex, v_flex};
use gpui_kit::{
    AnyElement, App, Entity, InteractiveElement, IntoElement, ObjectFit, ParentElement, SharedString, StatefulInteractiveElement, Styled,
    StyledImage as _, Window, div, img, prelude::FluentBuilder as _, px,
};

use super::viewer::Viewer;
use crate::i18n::t;
use crate::state::AppState;
use crate::theme::{hexa, pal};

/// Neighbors shown on each side of the current image.
const AROUND: usize = 4;

fn pill() -> gpui_kit::Div {
    h_flex().gap_2().px_3().py_1p5().rounded(px(18.0)).bg(hexa(0x000000, 0.6)).border_1().border_color(hexa(0xffffff, 0.08))
}

fn toggle(id: &'static str, label: &'static str, key: &'static str, on: bool, f: impl Fn(&mut Window, &mut App) + 'static) -> impl IntoElement {
    div()
        .id(id)
        .px_2()
        .py_0p5()
        .rounded(px(10.0))
        .text_xs()
        .cursor_pointer()
        .text_color(hexa(0xffffff, if on { 0.95 } else { 0.55 }))
        .bg(hexa(0xffffff, if on { 0.16 } else { 0.0 }))
        .hover(|d| d.bg(hexa(0xffffff, 0.12)))
        .on_click(move |_, window, cx| f(window, cx))
        .child(format!("{label} ({key})"))
}

pub fn render(state: &Entity<AppState>, viewer: &Entity<Viewer>, _window: &mut Window, cx: &mut App) -> AnyElement {
    let s = state.read(cx);
    let Some(cull) = s.cull.clone() else { return div().into_any_element() };
    let p = pal(cx).clone();
    let images = s.cull_images();
    let at = s.current.and_then(|c| images.iter().position(|(i, _)| *i == c)).unwrap_or(0);
    let info = s.current_info().cloned();
    let rating = info.as_ref().map(|i| i.rating).unwrap_or(0);
    let name = info.as_ref().map(|i| i.file_name.clone()).unwrap_or_default();
    let rated = images.iter().filter(|(_, r)| *r != 0).count();
    let rejected = images.iter().filter(|(_, r)| *r < 0).count();

    let stars = (1..=5).map(|n| {
        let st = state.clone();
        div()
            .id(SharedString::from(format!("cull-star-{n}")))
            .cursor_pointer()
            .text_color(if rating >= n { p.warning } else { hexa(0xffffff, 0.3) })
            .on_click(move |_, _, cx| st.update(cx, |s, cx| s.cull_rate(n, cx)))
            .child(Icon::new(IconName::Star).small())
    });
    let st_rej = state.clone();
    let reject = div()
        .id("cull-reject")
        .ml_1()
        .px_1p5()
        .rounded(px(4.0))
        .cursor_pointer()
        .text_xs()
        .text_color(if rating < 0 { gpui_kit::white() } else { hexa(0xffffff, 0.5) })
        .bg(if rating < 0 { p.danger } else { hexa(0xffffff, 0.0) })
        .hover(|d| d.bg(hexa(0xffffff, 0.12)))
        .on_click(move |_, _, cx| st_rej.update(cx, |s, cx| s.cull_rate(-1, cx)))
        .child(t("Reject", "拒绝"));

    let (st_a, st_u, st_exit) = (state.clone(), state.clone(), state.clone());
    let top = div().absolute().top(px(18.0)).left_0().right_0().flex().justify_center().child(
        pill()
            .items_center()
            .child(div().text_xs().text_color(hexa(0xffffff, 0.8)).child(format!("{} / {}  {name}", at + 1, images.len())))
            .child(div().w(px(1.0)).h(px(16.0)).bg(hexa(0xffffff, 0.15)))
            .child(h_flex().gap_0p5().children(stars).child(reject))
            .child(div().w(px(1.0)).h(px(16.0)).bg(hexa(0xffffff, 0.15)))
            .child(toggle("cull-auto", t("Auto-advance", "自动前进"), "A", cull.auto_advance, move |_, cx| st_a.update(cx, |s, cx| s.toggle_cull_auto_advance(cx))))
            .child(toggle("cull-unrated", t("Unrated only", "仅未评级"), "U", cull.unrated_only, move |_, cx| st_u.update(cx, |s, cx| s.toggle_cull_unrated(cx))))
            .child(div().text_xs().text_color(hexa(0xffffff, 0.55)).child(format!("{} {rated} · {} {rejected}", t("rated", "已评"), t("rejected", "拒绝"))))
            .child(
                div()
                    .id("cull-exit")
                    .size(px(26.0))
                    .rounded_full()
                    .flex()
                    .items_center()
                    .justify_center()
                    .cursor_pointer()
                    .text_color(hexa(0xffffff, 0.9))
                    .hover(|d| d.bg(hexa(0xffffff, 0.14)))
                    .on_click(move |_, window, cx| st_exit.update(cx, |s, cx| s.stop_cull(window, cx)))
                    .child(Icon::new(IconName::X).small()),
            ),
    );

    // Neighbors with their ratings, so a run of picks and rejects reads at a glance.
    let lo = at.saturating_sub(AROUND);
    let hi = (at + AROUND + 1).min(images.len());
    let strip = images[lo..hi].iter().map(|(id, r)| {
        let (id, r) = (*id, *r);
        let current = Some(id) == s.current;
        let st = state.clone();
        v_flex()
            .id(SharedString::from(format!("cull-thumb-{}", id.0)))
            .items_center()
            .gap_0p5()
            .cursor_pointer()
            .opacity(if r < 0 { 0.4 } else { 1.0 })
            .on_click(move |_, _, cx| st.update(cx, |s, cx| s.select(id, false, false, cx)))
            .child(
                div()
                    .size(px(64.0))
                    .rounded(px(4.0))
                    .border_2()
                    .border_color(if current { gpui_kit::white() } else { hexa(0xffffff, 0.0) })
                    .bg(hexa(0xffffff, 0.06))
                    .overflow_hidden()
                    .when_some(s.thumbs.get(&id).cloned(), |d, thumb| d.child(img(thumb).size_full().object_fit(ObjectFit::Contain))),
            )
            .child(div().text_xs().text_color(if r < 0 { p.danger } else { hexa(0xffffff, 0.7) }).child(match r {
                r if r < 0 => "✕".to_string(),
                0 => "·".to_string(),
                r => "★".repeat(r as usize),
            }))
    });
    let bottom = div().absolute().bottom(px(18.0)).left_0().right_0().flex().flex_col().items_center().gap_2().child(pill().children(strip)).child(
        div().text_xs().text_color(hexa(0xffffff, 0.45)).child(t(
            "1–5 rate · X reject · 0 clear · ← → step · Z fit / 100 % · Esc exit",
            "1–5 评级 · X 拒绝 · 0 清除 · ← → 切换 · Z 适合 / 100% · Esc 退出",
        )),
    );

    div()
        .id("cull")
        .size_full()
        .relative()
        .bg(gpui_kit::black())
        .child(div().absolute().size_full().child(viewer.clone()))
        .child(top)
        .child(bottom)
        .into_any_element()
}
