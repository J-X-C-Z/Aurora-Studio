//! Multiple view: every selected image side by side, the pinned compare variant
//! first. Tiles glide to their new place when the selection changes and new
//! ones fade in, so the eye can follow which image went where.

use gpui_kit::assets::IconName;
use gpui_kit::component::{Icon, Sizable as _, h_flex};
use gpui_kit::{
    AnyElement, App, Entity, InteractiveElement, IntoElement, ObjectFit, ParentElement, SharedString, StatefulInteractiveElement, Styled,
    StyledImage as _, Window, div, img, prelude::FluentBuilder as _, px,
};
use lumen_engine::ImageId;

use crate::i18n::t;
use crate::motion;
use crate::state::AppState;
use crate::theme::{hexa, pal};

const PAD: f32 = 14.0;
const GAP: f32 = 10.0;

/// Columns that give `n` tiles of aspect `aspect` the most area in `vw × vh`.
fn columns(n: usize, vw: f32, vh: f32, aspect: f32) -> usize {
    (1..=n)
        .max_by(|a, b| {
            let area = |cols: usize| {
                let rows = n.div_ceil(cols);
                let cw = (vw - 2.0 * PAD - GAP * (cols as f32 - 1.0)) / cols as f32;
                let ch = (vh - 2.0 * PAD - GAP * (rows as f32 - 1.0)) / rows as f32;
                let w = cw.min(ch * aspect).max(0.0);
                w * w / aspect
            };
            area(*a).total_cmp(&area(*b))
        })
        .unwrap_or(1)
}

pub fn render(state: &Entity<AppState>, ids: &[ImageId], window: &mut Window, cx: &mut App) -> AnyElement {
    let p = pal(cx).clone();
    let s = state.read(cx);
    let (vw, vh) = s.viewport;
    let n = ids.len();
    let aspect = ids.iter().filter_map(|id| s.info(*id)).map(|i| i.aspect()).sum::<f32>() / n.max(1) as f32;
    let cols = columns(n, vw, vh, aspect.max(0.2));
    let rows = n.div_ceil(cols);
    let cw = (vw - 2.0 * PAD - GAP * (cols as f32 - 1.0)) / cols as f32;
    let ch = (vh - 2.0 * PAD - GAP * (rows as f32 - 1.0)) / rows as f32;
    // Center the last, possibly short, row.
    let pinned = s.view.compare_variant;
    let current = s.current;

    let tiles: Vec<_> = ids
        .iter()
        .enumerate()
        .map(|(i, id)| {
            let (row, col) = (i / cols, i % cols);
            let in_row = if row == rows - 1 { n - row * cols } else { cols };
            let row_w = in_row as f32 * cw + (in_row as f32 - 1.0) * GAP;
            let x0 = (vw - row_w) / 2.0;
            (*id, x0 + col as f32 * (cw + GAP), PAD + row as f32 * (ch + GAP))
        })
        .collect();

    // Explicit size: an id'd absolute `size_full` root lays out empty here.
    let mut root = div().absolute().left_0().top_0().w(px(vw)).h(px(vh));
    for (id, tx, ty) in tiles {
        let key = format!("tile-{}", id.0);
        let x = motion::value(motion::key(&key, "x"), tx, motion::FLOW, window, cx);
        let y = motion::value(motion::key(&key, "y"), ty, motion::FLOW, window, cx);
        let w = motion::value(motion::key(&key, "w"), cw, motion::FLOW, window, cx);
        let h = motion::value(motion::key(&key, "h"), ch, motion::FLOW, window, cx);
        let is_current = current == Some(id);
        let is_pinned = pinned == Some(id);
        let ring = motion::toggle(motion::key(&key, "ring"), is_current, motion::QUICK, window, cx);

        let s = state.read(cx);
        let picture = if is_current {
            s.preview.frame.as_ref().filter(|f| f.image_id == id).map(|f| f.image.clone()).or_else(|| s.thumbs.get(&id).cloned())
        } else {
            s.thumbs.get(&id).cloned()
        };
        let info = s.info(id).cloned();
        let name = info.as_ref().map(|i| i.file_name.clone()).unwrap_or_default();
        let stars = info.as_ref().map(|i| if i.rating < 0 { "✕".to_string() } else { "★".repeat(i.rating.max(0) as usize) }).unwrap_or_default();

        let (st_click, st_pin) = (state.clone(), state.clone());
        let tile = div()
            .id(SharedString::from(key.clone()))
            .absolute()
            .left(px(x))
            .top(px(y))
            .w(px(w))
            .h(px(h))
            .rounded(px(5.0))
            .overflow_hidden()
            .bg(hexa(0x000000, 0.18))
            .border_2()
            .border_color(if is_pinned { p.warning } else { p.accent.opacity(ring) })
            .shadow_md()
            .cursor_pointer()
            .on_click(move |e, _, cx| {
                st_click.update(cx, |s, cx| {
                    if e.click_count() >= 2 {
                        // Double-click opens the image alone.
                        s.view.multi = false;
                        s.select(id, false, false, cx);
                    } else {
                        s.set_current(id, cx);
                    }
                })
            })
            .when_some(picture, |d, pic| d.child(img(pic).absolute().size_full().object_fit(ObjectFit::Contain)))
            .child(
                h_flex()
                    .absolute()
                    .bottom(px(6.0))
                    .left(px(6.0))
                    .gap_2()
                    .px_1p5()
                    .py_0p5()
                    .rounded(px(4.0))
                    .bg(hexa(0x000000, 0.55))
                    .text_xs()
                    .text_color(hexa(0xffffff, 0.88))
                    .when(is_pinned, |d| d.child(div().text_color(p.warning).child(t("Compare", "对比"))))
                    .child(name)
                    .when(!stars.is_empty(), |d| d.child(div().text_color(p.warning).child(stars))),
            )
            .child(
                div()
                    .id(SharedString::from(format!("{key}-pin")))
                    .absolute()
                    .top(px(6.0))
                    .right(px(6.0))
                    .size(px(24.0))
                    .rounded(px(4.0))
                    .flex()
                    .items_center()
                    .justify_center()
                    .bg(hexa(0x000000, if is_pinned { 0.6 } else { 0.35 }))
                    .text_color(if is_pinned { p.warning } else { hexa(0xffffff, 0.7) })
                    .hover(|d| d.bg(hexa(0x000000, 0.7)).text_color(hexa(0xffffff, 1.0)))
                    .tooltip(move |window, cx| {
                        gpui_kit::component::tooltip::Tooltip::new(if is_pinned { t("Clear compare variant", "取消对比变体") } else { t("Set as compare variant", "设为对比变体") })
                            .build(window, cx)
                    })
                    .on_click(move |_, _, cx| {
                        cx.stop_propagation();
                        st_pin.update(cx, |s, cx| {
                            s.view.compare_variant = if is_pinned { None } else { Some(id) };
                            cx.notify();
                        })
                    })
                    .child(Icon::new(if is_pinned { IconName::PinOff } else { IconName::Pin }).xsmall()),
            );
        root = root.child(motion::enter(SharedString::from(format!("{key}-in")), tile, 10.0));
    }
    root.into_any_element()
}
