//! Controls for the portrait algorithm catalogue
//! (`docs/frontend-contract/portrait-algorithms.json`): an amount slider, a
//! toggle, a mode menu, a color swatch row or an asset picker, chosen by the
//! parameter's [`Control`]. Values live in `AdjustmentSet::extra` under the
//! parameter's store key (`pc:<id>`, colors as `pc:<id>:r|g|b|a`).

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Disableable as _, Icon, Sizable as _,
    button::{Button, ButtonVariants as _},
    h_flex, v_flex,
};
use gpui_kit::{
    AnyElement, App, Entity, InteractiveElement, IntoElement, ParentElement, SharedString, StatefulInteractiveElement, Styled, Window, div,
    prelude::FluentBuilder as _, px, rgba,
};

use super::{row, select, switch_row};
use crate::contract::{Control, PortraitParam};
use crate::i18n::t;
use crate::state::{AppState, Binding, extra_binding};
use crate::theme::pal;

/// One catalogue parameter as a row: its control, a tooltip with id, key,
/// confidence and evidence, and the effects it is shared with. Child
/// parameters are indented under their parent.
pub fn param_portrait(state: &Entity<AppState>, param: &'static PortraitParam, cx: &App) -> AnyElement {
    let p = pal(cx).clone();
    let row = control(state, param, cx);
    let shared = (!param.shared.is_empty()).then(|| param.shared.join(" · "));
    let tip = format!("#{} {} · {} {:.2}\n{}", param.id, param.key, t("confidence", "置信度"), param.confidence, param.evidence);
    v_flex()
        .id(SharedString::from(format!("pc-row-{}", param.id)))
        .tooltip(move |window, cx| gpui_kit::component::tooltip::Tooltip::new(tip.clone()).build(window, cx))
        .when(param.child, |d| d.pl_3().border_l_2().border_color(p.divider).ml_3())
        .child(row)
        .when_some(shared, |d, effects| {
            d.child(
                div()
                    .px_3()
                    .pb_1()
                    .text_size(px(10.0))
                    .text_color(p.text_subtle)
                    .child(format!("{} {effects}", t("Shared by", "共用于"))),
            )
        })
        .into_any_element()
}

/// Binds a portrait amount stored as 0 ..= 1 to a 0 ..= 100 slider.
fn amount_binding(key: String, default: f32) -> Binding {
    let inner = extra_binding(key, default);
    let (get, set) = (inner.get.clone(), inner.set.clone());
    Binding::new(move |a| get(a) * 100.0, move |a, v| set(a, v / 100.0))
}

fn control(state: &Entity<AppState>, param: &'static PortraitParam, cx: &App) -> AnyElement {
    let key = param.store_key();
    let id = SharedString::from(format!("pc-{}", param.id));
    let label = param.label();
    let current = state.read(cx).adj().extra.get(&key).copied().unwrap_or(param.default);
    match &param.control {
        Control::Amount => super::param(state, id, label, amount_binding(key, param.default), 0.0, 100.0, cx)
            .default_value(param.default * 100.0)
            .into_any_element(),
        Control::Toggle => {
            let st = state.clone();
            let default = param.default;
            switch_row(id, label, current > 0.5, move |v, _, cx| {
                let key = key.clone();
                let label = t("Portrait", "人像");
                st.update(cx, |s, cx| {
                    s.edit(label, |a| {
                        let v = if v { 1.0 } else { 0.0 };
                        if (v - default).abs() < 1e-6 {
                            a.extra.remove(&key);
                        } else {
                            a.extra.insert(key, v);
                        }
                    }, cx)
                })
            }, cx)
        }
        Control::Mode => {
            let st = state.clone();
            let default = param.default;
            let count = (default as usize + 1).max(4);
            let options = (0..count).map(|i| (i, SharedString::from(format!("{} {}", t("Mode", "模式"), i + 1)))).collect();
            row(label, select(id, options, current.round().max(0.0) as usize, move |i, _, cx| {
                let key = key.clone();
                st.update(cx, |s, cx| {
                    s.edit(t("Portrait", "人像"), |a| {
                        if (i as f32 - default).abs() < 1e-6 {
                            a.extra.remove(&key);
                        } else {
                            a.extra.insert(key, i as f32);
                        }
                    }, cx)
                })
            }), cx)
        }
        Control::Color => row(label, swatches(state, param, cx), cx),
        Control::Asset(default) => {
            let p = pal(cx);
            let name = default.clone().unwrap_or_else(|| t("None", "无").to_string());
            row(
                label,
                h_flex()
                    .gap_1()
                    .child(div().flex_1().min_w_0().text_xs().text_color(p.text_subtle).overflow_hidden().whitespace_nowrap().text_ellipsis().child(name))
                    .child(
                        Button::new(id)
                            .xsmall()
                            .ghost()
                            .icon(Icon::new(IconName::FolderOpen))
                            .disabled(true)
                            .tooltip(t("The material library is not connected yet", "素材库后端尚未接入")),
                    ),
                cx,
            )
        }
    }
}

const SWATCHES: [u32; 8] = [0xc0392b, 0xe0607e, 0xd98c5f, 0x8e5a3c, 0x3b2a20, 0x7a4e9a, 0x2e6fb0, 0x1d1d1d];

/// A row of preset colors for an RGBA parameter, stored as `pc:<id>:r|g|b|a`.
fn swatches(state: &Entity<AppState>, param: &'static PortraitParam, cx: &App) -> AnyElement {
    let p = pal(cx).clone();
    let base = param.store_key();
    let extra = &state.read(cx).adj().extra;
    let ch = |c: &str, d: f32| extra.get(&format!("{base}:{c}")).copied().unwrap_or(d);
    let cur = [ch("r", param.default_color[0]), ch("g", param.default_color[1]), ch("b", param.default_color[2]), ch("a", param.default_color[3])];
    let set = |rgba: Option<[f32; 4]>| {
        let st = state.clone();
        let base = base.clone();
        move |_: &gpui_kit::ClickEvent, _: &mut Window, cx: &mut App| {
            let base = base.clone();
            st.update(cx, |s, cx| {
                s.edit(t("Portrait color", "人像颜色"), |a| {
                    for (i, c) in ["r", "g", "b", "a"].iter().enumerate() {
                        let k = format!("{base}:{c}");
                        match rgba {
                            Some(v) => {
                                a.extra.insert(k, v[i]);
                            }
                            None => {
                                a.extra.remove(&k);
                            }
                        }
                    }
                }, cx)
            })
        }
    };
    let none = cur[3] <= 0.0;
    h_flex()
        .gap_1()
        .child(
            div()
                .id(SharedString::from(format!("sw-{}-none", param.id)))
                .size(px(14.0))
                .rounded(px(3.0))
                .border_1()
                .border_color(if none { p.accent } else { p.edge })
                .bg(p.well)
                .cursor_pointer()
                .child(div().absolute().w(px(18.0)).h(px(1.0)).bg(p.danger).top(px(6.0)).left(px(-2.0)))
                .on_click(set(None)),
        )
        .children(SWATCHES.iter().enumerate().map(|(i, hex)| {
            let rgba = [((hex >> 16) & 0xff) as f32 / 255.0, ((hex >> 8) & 0xff) as f32 / 255.0, (hex & 0xff) as f32 / 255.0, 1.0];
            let selected = !none && (0..3).all(|c| (cur[c] - rgba[c]).abs() < 0.01);
            div()
                .id(SharedString::from(format!("sw-{}-{i}", param.id)))
                .size(px(14.0))
                .rounded(px(3.0))
                .border_1()
                .border_color(if selected { p.text } else { p.edge })
                .bg(rgba_u32(*hex))
                .cursor_pointer()
                .on_click(set(Some(rgba)))
        }))
        .into_any_element()
}

fn rgba_u32(hex: u32) -> gpui_kit::Rgba {
    rgba((hex << 8) | 0xff)
}
