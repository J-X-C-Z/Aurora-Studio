//! Status bar: selection, the latest message, render time, zoom and the engine.

use gpui_kit::assets::IconName;
use gpui_kit::component::{Icon, Sizable as _, h_flex};
use gpui_kit::{App, Entity, IntoElement, ParentElement, SharedString, Styled, Window, div, prelude::FluentBuilder as _, px};

use crate::i18n::t;
use crate::motion;
use crate::state::AppState;
use crate::theme::{metrics, pal};

pub fn status_bar(state: &Entity<AppState>, notice: Option<SharedString>, serial: u64, _window: &mut Window, cx: &App) -> impl IntoElement + use<> {
    let p = pal(cx).clone();
    let s = state.read(cx);
    let total = s.images.len();
    let selected = s.selection.len();
    let zoom = (s.effective_scale() / s.scale_factor.max(0.5) * 100.0).round();
    let ms = s.preview.elapsed_ms;
    let scope = s.active_layer_ref().map(|l| l.name.clone());
    let cursor = s.cursor.label();
    let dims = s.current_info().map(|i| format!("{} × {}", i.width, i.height));
    // Behind the welcome screen nothing is open yet: only messages show.
    let welcome = s.welcome;

    h_flex()
        .h(px(metrics::STATUS_BAR))
        .flex_shrink_0()
        .px_3()
        .gap_4()
        .bg(p.chrome)
        .border_t_1()
        .border_color(p.divider)
        .text_xs()
        .text_color(p.text_subtle)
        .when(!welcome, |d| {
            d.child(format!("{total} {}", t("images", "张图像")))
                .when(selected > 1, |d| d.child(format!("{selected} {}", t("selected", "已选"))))
                .when_some(dims, |d, x| d.child(x))
                .child(h_flex().gap_1().child(Icon::new(IconName::MousePointer2).xsmall()).child(cursor))
        })
        .when_some(scope.filter(|_| !welcome), |d, name| d.child(h_flex().gap_1().text_color(p.accent).child(Icon::new(IconName::Layers).xsmall()).child(name)))
        .child(
            div().flex_1().flex().justify_center().when_some(notice, |d, n| {
                d.child(motion::enter(SharedString::from(format!("notice-{serial}")), div().text_color(p.text_muted).child(n), 4.0))
            }),
        )
        .when(!welcome, |d| d.child(format!("{zoom:.0}%")).child(format!("{ms:.0} ms")))
        .when(!welcome, |d| d.child(match &s.source {
            // The demo is procedural: say so instead of naming an engine.
            crate::state::Source::Demo => h_flex()
                .gap_1()
                .text_color(p.warning)
                .child(Icon::new(IconName::Sparkles).xsmall())
                .child(t("Demo images · generated, not real photos", "演示图像 · 程序生成，非真实照片"))
                .into_any_element(),
            _ => div()
                .text_color(p.text_disabled)
                .child(s.engine_name.clone())
                .into_any_element(),
        }))
}
