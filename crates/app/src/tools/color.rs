//! Color tab: base characteristics, white balance, color balance, color editor and
//! black & white. White balance, color balance and the color editor follow the
//! selected layer.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _,
    button::{Button, ButtonVariants as _},
    h_flex, v_flex,
};
use gpui_kit::{
    AnyElement, App, Entity, InteractiveElement, IntoElement, ParentElement, SharedString, StatefulInteractiveElement,
    Styled, Window, div, prelude::FluentBuilder as _, px,
};
use lumen_engine::*;

use crate::i18n::t;
use crate::state::{AppState, Binding, CursorTool, Scope, read_scoped, with_scoped};
use crate::theme::{hex, hue_color, pal};
use crate::widgets::*;
use crate::{global, scoped};

pub fn render(state: &Entity<AppState>, _window: &mut Window, cx: &mut App) -> AnyElement {
    let scope = state.read(cx).scope();
    v_flex()
        .w_full()
        .when(scope == Scope::Global, |d| d.child(base_panel(state, cx)))
        .child(white_balance_panel(state, scope, cx))
        .child(color_balance_panel(state, scope, cx))
        .child(color_editor_panel(state, scope, cx))
        .when(scope == Scope::Global, |d| d.child(black_white_panel(state, cx)))
        .child(readout_panel(state, cx))
        .into_any_element()
}

/// One panel of this tab by id, for the Quick tab.
pub fn panel(id: &str, state: &Entity<AppState>, _window: &mut Window, cx: &mut App) -> Option<AnyElement> {
    let scope = state.read(cx).scope();
    Some(match id {
        "base" if scope == Scope::Global => base_panel(state, cx).into_any_element(),
        "wb" => white_balance_panel(state, scope, cx).into_any_element(),
        "color-balance" => color_balance_panel(state, scope, cx).into_any_element(),
        "color-editor" => color_editor_panel(state, scope, cx).into_any_element(),
        "bw" if scope == Scope::Global => black_white_panel(state, cx).into_any_element(),
        "readout" => readout_panel(state, cx).into_any_element(),
        _ => return None,
    })
}

pub fn hue_stops() -> Vec<gpui_kit::Hsla> {
    (0..=6).map(|i| hue_color(i as f32 * 60.0, 0.75, 0.55)).collect()
}

fn base_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let base = s.adj().base.clone();
    let profiles = s.icc_profiles.clone();
    let (st, st2, st3) = (state.clone(), state.clone(), state.clone());
    let mut icc: Vec<(Option<usize>, SharedString)> = vec![(None, t("Default for camera", "相机默认").into())];
    icc.extend(profiles.iter().enumerate().map(|(i, p)| (Some(i), p.clone().into())));
    let current_icc = base.icc_profile.as_ref().and_then(|n| profiles.iter().position(|p| p == n));
    let curves = crate::planned::film_curves(s.engine.as_ref());
    let curve_names = [
        t("Auto", "自动"),
        t("Film Standard", "胶片标准"),
        t("Film Extra Shadow", "胶片增强阴影"),
        t("Film High Contrast", "胶片高对比"),
        t("Linear Response", "线性响应"),
    ];
    let p = pal(cx).clone();
    ToolPanel::new(state, "base", t("Base Characteristics", "基础特性"))
        .modified(base != BaseCharacteristics::default())
        .on_reset(move |_, cx| st.update(cx, |s, cx| s.edit(t("Reset base", "复位基础特性"), |a| a.base = Default::default(), cx)))
        .child(row(
            t("ICC Profile", "ICC 配置"),
            select("icc", icc, current_icc, move |i, _, cx| {
                st2.update(cx, |s, cx| {
                    let name = i.and_then(|i| s.icc_profiles.get(i).cloned());
                    s.edit(t("ICC profile", "ICC 配置"), |a| a.base.icc_profile = name, cx)
                })
            }),
            cx,
        ))
        .child(row(
            t("Curve", "曲线"),
            select(
                "base-curve",
                ToneCurvePreset::ALL.iter().zip(curve_names).map(|(c, n)| (*c, n.into())).collect(),
                base.curve,
                move |c, _, cx| st3.update(cx, |s, cx| s.edit(t("Base curve", "基础曲线"), |a| a.base.curve = c, cx)),
            ),
            cx,
        ))
        .child(row(t("Film Curve", "胶片曲线"), film_curve_select(state, &base, curves), cx))
        .child(row(t("Engine", "引擎"), div().text_xs().text_color(p.text_muted).child(base.engine_version.replacen("Lumen", "Aurora", 1)), cx))
}

/// The film curve the engine applies before gamma: none, the one it set for the
/// camera, or any the engine lists (`planned::film_curves`, empty until the
/// engine offers the list).
fn film_curve_select(state: &Entity<AppState>, base: &BaseCharacteristics, mut names: Vec<String>) -> AnyElement {
    if let Some(name) = base.film_curve.as_ref().filter(|n| !names.contains(n)) {
        names.insert(0, name.clone());
    }
    let current = base.film_curve.as_ref().and_then(|n| names.iter().position(|x| x == n)).map_or(0, |i| i + 1);
    let mut options: Vec<(usize, SharedString)> = vec![(0, t("None (standard gamma)", "无（标准伽马）").into())];
    options.extend(names.iter().enumerate().map(|(i, n)| (i + 1, n.clone().into())));
    let st = state.clone();
    select("base-film-curve", options, current, move |i, _, cx| {
        let name = i.checked_sub(1).and_then(|i| names.get(i).cloned());
        st.update(cx, |s, cx| s.edit(t("Film curve", "胶片曲线"), |a| a.base.film_curve = name, cx))
    })
}

pub(crate) fn wb_binding(scope: Scope, kelvin: bool) -> Binding {
    let get = move |a: &AdjustmentSet| {
        let wb = match scope {
            Scope::Global => Some(&a.white_balance),
            Scope::Layer(id) => a.layers.iter().find(|l| l.id == id).and_then(|l| l.adjustments.white_balance.as_ref()),
        };
        let wb = wb.cloned().unwrap_or_else(|| a.white_balance.clone());
        if kelvin { wb.kelvin } else { wb.tint }
    };
    let set = move |a: &mut AdjustmentSet, v: f32| {
        let base = a.white_balance.clone();
        let wb = match scope {
            Scope::Global => &mut a.white_balance,
            Scope::Layer(id) => match a.layers.iter_mut().find(|l| l.id == id) {
                Some(l) => l.adjustments.white_balance.get_or_insert(base),
                None => return,
            },
        };
        wb.mode = WhiteBalanceMode::Custom;
        if kelvin { wb.kelvin = v } else { wb.tint = v }
    };
    Binding::new(get, set)
}

fn wb_mode_name(m: WhiteBalanceMode) -> &'static str {
    match m {
        WhiteBalanceMode::Shot => t("Shot", "拍摄"),
        WhiteBalanceMode::Auto => t("Auto", "自动"),
        WhiteBalanceMode::Daylight => t("Daylight", "日光"),
        WhiteBalanceMode::Cloudy => t("Cloudy", "阴天"),
        WhiteBalanceMode::Shade => t("Shade", "阴影"),
        WhiteBalanceMode::Tungsten => t("Tungsten", "钨丝灯"),
        WhiteBalanceMode::Fluorescent => t("Fluorescent", "荧光灯"),
        WhiteBalanceMode::Flash => t("Flash", "闪光灯"),
        WhiteBalanceMode::Custom => t("Custom", "自定义"),
    }
}

fn white_balance_panel(state: &Entity<AppState>, scope: Scope, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let wb = match scope {
        Scope::Global => s.adj().white_balance.clone(),
        Scope::Layer(id) => s
            .adj()
            .layers
            .iter()
            .find(|l| l.id == id)
            .and_then(|l| l.adjustments.white_balance.clone())
            .unwrap_or_else(|| s.adj().white_balance.clone()),
    };
    let picking = s.cursor == CursorTool::WhiteBalance;
    let (st, st2, st3, st4) = (state.clone(), state.clone(), state.clone(), state.clone());
    let skin = [t("None", "无"), t("Light", "浅肤色"), t("Medium", "中等肤色"), t("Dark", "深肤色")];
    ToolPanel::new(state, "wb", t("White Balance", "白平衡"))
        .modified(wb != WhiteBalance::default())
        .on_auto(move |_, cx| st4.update(cx, |s, cx| s.auto_adjust(AutoTarget::WhiteBalance, cx)))
        .action(
            HeaderAction::new("pick", IconName::Pipette, t("Pick white balance (W)", "白平衡取样 (W)"), move |_, cx| {
                st3.update(cx, |s, cx| {
                    let next = if s.cursor == CursorTool::WhiteBalance { CursorTool::Select } else { CursorTool::WhiteBalance };
                    s.set_cursor(next, cx)
                })
            })
            .active(picking),
        )
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                let scope = s.scope();
                s.edit(t("Reset white balance", "复位白平衡"), |a| match scope {
                    Scope::Global => a.white_balance = Default::default(),
                    Scope::Layer(id) => {
                        if let Some(l) = a.layers.iter_mut().find(|l| l.id == id) {
                            l.adjustments.white_balance = None;
                        }
                    }
                }, cx)
            })
        })
        .child(row(
            t("Mode", "模式"),
            select(
                "wb-mode",
                WhiteBalanceMode::ALL.iter().map(|m| (*m, wb_mode_name(*m).into())).collect(),
                wb.mode,
                move |m, _, cx| {
                    st2.update(cx, |s, cx| {
                        if m == WhiteBalanceMode::Auto {
                            s.auto_adjust(AutoTarget::WhiteBalance, cx);
                            return;
                        }
                        let scope = s.scope();
                        s.edit(t("White balance mode", "白平衡模式"), |a| {
                            let base = a.white_balance.clone();
                            let wb = match scope {
                                Scope::Global => &mut a.white_balance,
                                Scope::Layer(id) => match a.layers.iter_mut().find(|l| l.id == id) {
                                    Some(l) => l.adjustments.white_balance.get_or_insert(base),
                                    None => return,
                                },
                            };
                            wb.mode = m;
                            if let Some((k, tint)) = m.preset() {
                                wb.kelvin = k;
                                wb.tint = tint;
                            } else if m == WhiteBalanceMode::Shot {
                                wb.kelvin = 5600.0;
                                wb.tint = 0.0;
                            }
                        }, cx)
                    })
                },
            ),
            cx,
        ))
        .child(
            param_c1(state, "wb-kelvin", "WhiteBalanceTemperature", wb_binding(scope, true), cx)
                .origin(5600.0)
                .scale(Scale::Power(1.6))
                .track(Track::Gradient(vec![hex(0x4a7bd8), hex(0xd8d8d8), hex(0xe8b040)])),
        )
        .child(
            param_c1(state, "wb-tint", "WhiteBalanceTint", wb_binding(scope, false), cx)
                .track(Track::Gradient(vec![hex(0x3fae58), hex(0xd8d8d8), hex(0xc048b8)])),
        )
        .child(row(
            t("Skin Tone", "肤色"),
            select("wb-skin", (0..4).map(|i| (i, skin[i].into())).collect(), wb.skin_tone.as_deref().map_or(0, |s| match s {
                "Light" => 1,
                "Medium" => 2,
                "Dark" => 3,
                _ => 0,
            }), {
                let st = state.clone();
                move |i, _, cx| {
                    st.update(cx, |s, cx| {
                        s.edit(t("Skin tone", "肤色"), |a| a.white_balance.skin_tone = [None, Some("Light"), Some("Medium"), Some("Dark")][i].map(String::from), cx)
                    })
                }
            }),
            cx,
        ))
}

fn wheel_for(
    state: &Entity<AppState>,
    id: &'static str,
    title: &'static str,
    value: ColorWheel,
    size: f32,
    pick: fn(&mut ColorBalance) -> &mut ColorWheel,
) -> impl IntoElement + use<> {
    let (st, st2) = (state.clone(), state.clone());
    ColorWheelView::new(id, value)
        .title(title)
        .size(size)
        .on_change(move |w, _, cx| {
            st.update(cx, |s, cx| {
                let scope = s.scope();
                s.edit_live(t("Color balance", "色彩平衡"), |a| {
                    with_scoped(a, scope, |a| &mut a.color_balance, |l| &mut l.color_balance, |cb| *pick(cb) = w)
                }, cx)
            })
        })
        .on_commit(move |_, cx| st2.update(cx, |s, cx| s.end_gesture(cx)))
}

fn color_balance_panel(state: &Entity<AppState>, scope: Scope, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let cb = read_scoped(s.adj(), scope, |a| &a.color_balance, |l| &l.color_balance).cloned().unwrap_or_default();
    let mode = s.panels.color_balance_mode;
    let (st, st2) = (state.clone(), state.clone());
    let modes = [
        ColorBalanceMode::ThreeWay,
        ColorBalanceMode::Master,
        ColorBalanceMode::Shadow,
        ColorBalanceMode::Midtone,
        ColorBalanceMode::Highlight,
    ];
    let idx = modes.iter().position(|m| *m == mode).unwrap_or(0);
    let body: AnyElement = match mode {
        ColorBalanceMode::ThreeWay => v_flex()
            .px_2()
            .gap_1()
            .child(
                h_flex()
                    .justify_between()
                    .child(wheel_for(state, "cb-shadow", t("Shadow", "阴影"), cb.shadow, 84.0, |c| &mut c.shadow))
                    .child(wheel_for(state, "cb-mid", t("Midtone", "中间调"), cb.midtone, 84.0, |c| &mut c.midtone))
                    .child(wheel_for(state, "cb-high", t("Highlight", "高光"), cb.highlight, 84.0, |c| &mut c.highlight)),
            )
            .child(param(state, "cb-shadow-l", t("Shadow L", "阴影亮度"), scoped!(scope, color_balance.shadow.lightness), -1.0, 1.0, cx).decimals(2))
            .child(param(state, "cb-mid-l", t("Midtone L", "中间调亮度"), scoped!(scope, color_balance.midtone.lightness), -1.0, 1.0, cx).decimals(2))
            .child(param(state, "cb-high-l", t("Highlight L", "高光亮度"), scoped!(scope, color_balance.highlight.lightness), -1.0, 1.0, cx).decimals(2))
            .into_any_element(),
        single => {
            let (value, pick, b_h, b_s, b_l): (ColorWheel, fn(&mut ColorBalance) -> &mut ColorWheel, Binding, Binding, Binding) = match single {
                ColorBalanceMode::Shadow => (
                    cb.shadow,
                    |c| &mut c.shadow,
                    scoped!(scope, color_balance.shadow.hue),
                    scoped!(scope, color_balance.shadow.saturation),
                    scoped!(scope, color_balance.shadow.lightness),
                ),
                ColorBalanceMode::Midtone => (
                    cb.midtone,
                    |c| &mut c.midtone,
                    scoped!(scope, color_balance.midtone.hue),
                    scoped!(scope, color_balance.midtone.saturation),
                    scoped!(scope, color_balance.midtone.lightness),
                ),
                ColorBalanceMode::Highlight => (
                    cb.highlight,
                    |c| &mut c.highlight,
                    scoped!(scope, color_balance.highlight.hue),
                    scoped!(scope, color_balance.highlight.saturation),
                    scoped!(scope, color_balance.highlight.lightness),
                ),
                _ => (
                    cb.master,
                    |c| &mut c.master,
                    scoped!(scope, color_balance.master.hue),
                    scoped!(scope, color_balance.master.saturation),
                    scoped!(scope, color_balance.master.lightness),
                ),
            };
            v_flex()
                .gap_1()
                .child(h_flex().justify_center().child(wheel_for(state, "cb-single", "", value, 150.0, pick)))
                .child(param(state, "cb-h", t("Hue", "色相"), b_h, 0.0, 360.0, cx).unit("°").track(Track::Gradient(hue_stops())))
                .child(param(state, "cb-s", t("Saturation", "饱和度"), b_s, 0.0, 1.0, cx).decimals(2))
                .child(param(state, "cb-l", t("Lightness", "亮度"), b_l, -1.0, 1.0, cx).decimals(2))
                .into_any_element()
        }
    };
    ToolPanel::new(state, "color-balance", t("Color Balance", "色彩平衡"))
        .modified(cb != ColorBalance::default())
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                let scope = s.scope();
                s.edit(t("Reset color balance", "复位色彩平衡"), |a| {
                    with_scoped(a, scope, |a| &mut a.color_balance, |l| &mut l.color_balance, |c| *c = Default::default())
                }, cx)
            })
        })
        .child(
            div().px_3().pb_2().child(
                Segmented::new("cb-mode", idx)
                    .text(t("3-Way", "三路"))
                    .text(t("Master", "主"))
                    .text(t("Shadow", "阴影"))
                    .text(t("Mid", "中间"))
                    .text(t("High", "高光"))
                    .on_select(move |i, _, cx| {
                        st2.update(cx, |s, cx| {
                            s.panels.color_balance_mode = modes[i];
                            cx.notify();
                        })
                    }),
            ),
        )
        .child(body)
}

fn sector_name(s: HueSector) -> &'static str {
    match s {
        HueSector::Red => t("Red", "红"),
        HueSector::Orange => t("Orange", "橙"),
        HueSector::Yellow => t("Yellow", "黄"),
        HueSector::Green => t("Green", "绿"),
        HueSector::Cyan => t("Cyan", "青"),
        HueSector::Blue => t("Blue", "蓝"),
        HueSector::Purple => t("Purple", "紫"),
        HueSector::Magenta => t("Magenta", "洋红"),
    }
}

/// Hue bar with the selected range's window drawn over it.
pub(crate) fn hue_range_bar(range: &ColorRange, cx: &App) -> impl IntoElement + use<> {
    let p = pal(cx).clone();
    let stops = hue_stops();
    let left = ((range.hue - range.width / 2.0).rem_euclid(360.0)) / 360.0;
    let width = (range.width / 360.0).min(1.0);
    let feather = (range.smoothness / 360.0).min(0.5);
    div()
        .mx_3()
        .h(px(18.0))
        .relative()
        .rounded(px(3.0))
        .overflow_hidden()
        .child(
            h_flex().absolute().size_full().children((0..6).map(|i| {
                div().h_full().flex_1().bg(gpui_kit::linear_gradient(
                    90.0,
                    gpui_kit::linear_color_stop(stops[i], 0.0),
                    gpui_kit::linear_color_stop(stops[i + 1], 1.0),
                ))
            })),
        )
        .child(div().absolute().top_0().bottom_0().left(gpui_kit::relative((left - feather).max(0.0))).w(gpui_kit::relative(width + feather * 2.0)).bg(gpui_kit::hsla(0.0, 0.0, 1.0, 0.15)))
        .child(
            div()
                .absolute()
                .top_0()
                .bottom_0()
                .left(gpui_kit::relative(left))
                .w(gpui_kit::relative(width))
                .border_2()
                .border_color(p.text),
        )
}

fn color_editor_panel(state: &Entity<AppState>, scope: Scope, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let ce = read_scoped(s.adj(), scope, |a| &a.color_editor, |l| &l.color_editor).cloned().unwrap_or_default();
    let mode = s.panels.color_editor_mode;
    let sector = s.panels.selected_sector.min(7);
    let selected_range = s.panels.selected_color_range;
    let p = pal(cx).clone();
    let (st, st2) = (state.clone(), state.clone());

    let body: AnyElement = match mode {
        0 => {
            let chips = h_flex().px_3().pb_2().gap_1().children(HueSector::ALL.iter().enumerate().map(|(i, sec)| {
                let st = state.clone();
                let changed = ce.basic[i] != HslShift::default();
                div()
                    .id(SharedString::from(format!("sector-{i}")))
                    .flex_1()
                    .h(px(22.0))
                    .rounded(px(4.0))
                    .bg(hue_color(sec.hue(), 0.7, 0.5))
                    .border_2()
                    .border_color(if i == sector { p.text } else { gpui_kit::transparent_black() })
                    .cursor_pointer()
                    .flex()
                    .items_end()
                    .justify_center()
                    .when(changed, |d| d.child(div().mb(px(2.0)).size(px(4.0)).rounded_full().bg(gpui_kit::white())))
                    .on_click(move |_, _, cx| {
                        st.update(cx, |s, cx| {
                            s.panels.selected_sector = i;
                            cx.notify();
                        })
                    })
            }));
            let name = sector_name(HueSector::ALL[sector]);
            let st_pick = state.clone();
            let chips = chips.child(
                IconBtn::new("ce-basic-pick", IconName::Pipette, t("Pick a color in the image", "在图像中取样颜色")).size(22.0).on_click(move |_, cx| {
                    st_pick.update(cx, |s, cx| {
                        s.panels.pick_to_range = true;
                        s.set_cursor(CursorTool::ColorPicker, cx)
                    })
                }),
            );
            v_flex()
                .child(chips)
                .child(param_c1(state, format!("ce-h-{sector}"), "BasicHue", scoped!(scope, color_editor.basic[sector].hue), cx).label(format!("{name} · {}", t("Hue", "色相"))))
                .child(param_c1(state, format!("ce-s-{sector}"), "BasicSaturation", scoped!(scope, color_editor.basic[sector].saturation), cx).label(format!("{name} · {}", t("Saturation", "饱和度"))))
                .child(param_c1(state, format!("ce-l-{sector}"), "BasicLightness", scoped!(scope, color_editor.basic[sector].lightness), cx).label(format!("{name} · {}", t("Lightness", "亮度"))).range(-50.0, 50.0))
                .child(band_controls(state, scope, sector, &ce.bands[sector], cx))
                .into_any_element()
        }
        _ => {
            let skin = mode == 2;
            let list = if skin { ce.skin.clone() } else { ce.advanced.clone() };
            let sel_idx = selected_range.and_then(|id| list.iter().position(|r| r.id == id)).or(if list.is_empty() { None } else { Some(0) });
            let add = {
                let st = state.clone();
                move |_: &gpui_kit::ClickEvent, _: &mut Window, cx: &mut App| {
                    st.update(cx, |s, cx| {
                        let scope = s.scope();
                        let id = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_micros() as u64).unwrap_or(1);
                        s.edit(t("Add color range", "添加色彩范围"), |a| {
                            with_scoped(a, scope, |a| &mut a.color_editor, |l| &mut l.color_editor, |ce| {
                                let r = ColorRange {
                                    id,
                                    hue: if skin { 25.0 } else { 200.0 },
                                    width: if skin { 30.0 } else { 40.0 },
                                    uniformity: skin.then(SkinUniformity::default),
                                    ..Default::default()
                                };
                                if skin { ce.skin.push(r) } else { ce.advanced.push(r) }
                            })
                        }, cx);
                        s.panels.selected_color_range = Some(id);
                    })
                }
            };
            let rows = v_flex().px_3().gap_1().children(list.iter().enumerate().map(|(i, r)| {
                let st = state.clone();
                let selected = Some(i) == sel_idx;
                let id = r.id;
                h_flex()
                    .id(SharedString::from(format!("range-{id}")))
                    .h(px(24.0))
                    .px_2()
                    .gap_2()
                    .rounded(px(4.0))
                    .cursor_pointer()
                    .when(selected, |d| d.bg(p.accent_soft))
                    .hover(|d| d.bg(p.hover))
                    .child(div().size(px(12.0)).rounded(px(3.0)).bg(hue_color(r.hue, 0.7, 0.5)))
                    .child(div().flex_1().text_xs().text_color(p.text).child(format!("{} {}  ·  {:.0}°", if skin { t("Skin", "肤色") } else { t("Range", "范围") }, i + 1, r.hue)))
                    .on_click(move |_, _, cx| {
                        st.update(cx, |s, cx| {
                            s.panels.selected_color_range = Some(id);
                            cx.notify();
                        })
                    })
            }));
            let mut col = v_flex()
                .gap_1()
                .child(rows)
                .child(
                    h_flex()
                        .px_3()
                        .gap_1p5()
                        .child(Button::new("ce-add").xsmall().outline().icon(Icon::new(IconName::Plus)).label(t("Add range", "添加范围")).on_click(add))
                        .child(
                            Button::new("ce-pick").xsmall().ghost().icon(Icon::new(IconName::Pipette)).label(t("Pick", "取样")).on_click({
                                let st = state.clone();
                                move |_, _, cx| {
                                    st.update(cx, |s, cx| {
                                        s.panels.pick_to_range = true;
                                        s.set_cursor(CursorTool::ColorPicker, cx)
                                    })
                                }
                            }),
                        ),
                );
            if let Some(i) = sel_idx {
                let r = list[i].clone();
                let range_binding = |field: fn(&mut ColorRange) -> &mut f32, read: fn(&ColorRange) -> f32| {
                    let id = r.id;
                    Binding::new(
                        move |a: &AdjustmentSet| {
                            let ce = match scope {
                                Scope::Global => Some(&a.color_editor),
                                Scope::Layer(lid) => a.layers.iter().find(|l| l.id == lid).map(|l| &l.adjustments.color_editor),
                            };
                            ce.and_then(|ce| ce.advanced.iter().chain(ce.skin.iter()).find(|r| r.id == id)).map(read).unwrap_or_default()
                        },
                        move |a: &mut AdjustmentSet, v: f32| {
                            with_scoped(a, scope, |a| &mut a.color_editor, |l| &mut l.color_editor, |ce| {
                                if let Some(r) = ce.advanced.iter_mut().chain(ce.skin.iter_mut()).find(|r| r.id == id) {
                                    *field(r) = v;
                                }
                            })
                        },
                    )
                };
                let rid = r.id;
                col = col
                    .child(div().h(px(6.0)))
                    .child(hue_range_bar(&r, cx))
                    .child(param(state, format!("cr-hue-{rid}"), t("Center", "中心"), range_binding(|r| &mut r.hue, |r| r.hue), 0.0, 360.0, cx).unit("°").track(Track::Gradient(hue_stops())))
                    .child(param(state, format!("cr-w-{rid}"), t("Range", "范围"), range_binding(|r| &mut r.width, |r| r.width), 5.0, 180.0, cx).unit("°").default_value(40.0))
                    .child(param(state, format!("cr-sm-{rid}"), t("Smoothness", "平滑度"), range_binding(|r| &mut r.smoothness, |r| r.smoothness), 0.0, 60.0, cx).default_value(15.0))
                    .child(param(state, format!("cr-ms-{rid}"), t("Min. Saturation", "最低饱和度"), range_binding(|r| &mut r.min_saturation, |r| r.min_saturation), 0.0, 1.0, cx).decimals(2))
                    .child(caption(t("Adjust", "调整"), cx))
                    .child(param(state, format!("cr-sh-{rid}"), t("Hue", "色相"), range_binding(|r| &mut r.shift.hue, |r| r.shift.hue), -30.0, 30.0, cx).unit("°"))
                    .child(param(state, format!("cr-ss-{rid}"), t("Saturation", "饱和度"), range_binding(|r| &mut r.shift.saturation, |r| r.shift.saturation), -100.0, 100.0, cx))
                    .child(param(state, format!("cr-sl-{rid}"), t("Lightness", "亮度"), range_binding(|r| &mut r.shift.lightness, |r| r.shift.lightness), -100.0, 100.0, cx));
                if skin {
                    col = col
                        .child(caption(t("Uniformity", "均匀度"), cx))
                        .child(param(state, format!("cu-h-{rid}"), t("Hue", "色相"), range_binding(|r| &mut r.uniformity.get_or_insert_default().hue, |r| r.uniformity.map_or(0.0, |u| u.hue)), 0.0, 100.0, cx))
                        .child(param(state, format!("cu-s-{rid}"), t("Saturation", "饱和度"), range_binding(|r| &mut r.uniformity.get_or_insert_default().saturation, |r| r.uniformity.map_or(0.0, |u| u.saturation)), 0.0, 100.0, cx))
                        .child(param(state, format!("cu-l-{rid}"), t("Lightness", "亮度"), range_binding(|r| &mut r.uniformity.get_or_insert_default().lightness, |r| r.uniformity.map_or(0.0, |u| u.lightness)), 0.0, 100.0, cx));
                }
                let st_del = state.clone();
                col = col.child(div().px_3().pt_1().child(Button::new("cr-del").xsmall().ghost().icon(Icon::new(IconName::Trash)).label(t("Delete range", "删除范围")).on_click(move |_, _, cx| {
                    st_del.update(cx, |s, cx| {
                        let scope = s.scope();
                        s.edit(t("Delete color range", "删除色彩范围"), |a| {
                            with_scoped(a, scope, |a| &mut a.color_editor, |l| &mut l.color_editor, |ce| {
                                ce.advanced.retain(|r| r.id != rid);
                                ce.skin.retain(|r| r.id != rid);
                            })
                        }, cx)
                    })
                })));
            } else {
                col = col.child(div().px_3().py_2().text_xs().text_color(p.text_subtle).child(t(
                    "Add a range, or pick a color in the viewer.",
                    "添加一个范围，或在查看器中取样颜色。",
                )));
            }
            col.into_any_element()
        }
    };

    ToolPanel::new(state, "color-editor", t("Color Editor", "色彩编辑器"))
        .status(crate::contract::COLOR_EDITOR)
        .modified(ce != ColorEditor::default())
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                let scope = s.scope();
                s.edit(t("Reset color editor", "复位色彩编辑器"), |a| {
                    with_scoped(a, scope, |a| &mut a.color_editor, |l| &mut l.color_editor, |c| *c = Default::default())
                }, cx)
            })
        })
        .child(
            div().px_3().pb_2().child(
                Segmented::new("ce-mode", mode as usize)
                    .text(t("Basic", "基本"))
                    .text(t("Advanced", "高级"))
                    .text(t("Skin Tone", "肤色"))
                    .on_select(move |i, _, cx| {
                        st2.update(cx, |s, cx| {
                            s.panels.color_editor_mode = i as u8;
                            s.panels.selected_color_range = None;
                            cx.notify();
                        })
                    }),
            ),
        )
        .child(body)
}

fn black_white_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let bw = state.read(cx).adj().black_white.clone();
    let (st, st2) = (state.clone(), state.clone());
    let hues = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0];
    let mut panel = ToolPanel::new(state, "bw", t("Black & White", "黑白"))
        .modified(bw != BlackWhite::default())
        .on_reset(move |_, cx| st.update(cx, |s, cx| s.edit(t("Reset black & white", "复位黑白"), |a| a.black_white = Default::default(), cx)))
        .child(switch_row("bw-enable", t("Enable Black & White", "启用黑白"), bw.enabled, move |v, _, cx| {
            st2.update(cx, |s, cx| s.edit(t("Black & white", "黑白"), |a| a.black_white.enabled = v, cx))
        }, cx))
        .child(caption(t("Color Sensitivity", "色彩敏感度"), cx));
    for i in 0..6 {
        panel = panel.child(
            param_c1(state, format!("bw-{i}"), ["BwRed", "BwYellow", "BwGreen", "BwCyan", "BwBlue", "BwMagenta"][i], global!(black_white.sensitivity[i]), cx)
                .disabled(!bw.enabled)
                .track(Track::Gradient(vec![hex(0x202020), hue_color(hues[i], 0.75, 0.5), hex(0xf0f0f0)])),
        );
    }
    panel
        .child(caption(t("Split Tones · Highlights", "分离色调 · 高光"), cx))
        .child(param_c1(state, "bw-hh", "BwLightHue", global!(black_white.highlight_tone.hue), cx).track(Track::Gradient(hue_stops())))
        .child(param_c1(state, "bw-hs", "BwLightSaturation", global!(black_white.highlight_tone.saturation), cx))
        .child(caption(t("Split Tones · Shadows", "分离色调 · 阴影"), cx))
        .child(param_c1(state, "bw-sh", "BwDarkHue", global!(black_white.shadow_tone.hue), cx).track(Track::Gradient(hue_stops())))
        .child(param_c1(state, "bw-ss", "BwDarkSaturation", global!(black_white.shadow_tone.saturation), cx))
}

fn readout_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let c = s.sampled_color;
    let picking = s.cursor == CursorTool::ColorPicker;
    let st = state.clone();
    ToolPanel::new(state, "readout", t("Color Readout", "颜色读数"))
        .action(
            HeaderAction::new("pick", IconName::Pipette, t("Color picker (I)", "颜色取样 (I)"), move |_, cx| {
                st.update(cx, |s, cx| {
                    let next = if s.cursor == CursorTool::ColorPicker { CursorTool::Select } else { CursorTool::ColorPicker };
                    s.set_cursor(next, cx)
                })
            })
            .active(picking),
        )
        .child(match c {
            Some(c) => h_flex()
                .px_3()
                .gap_3()
                .child(div().size(px(36.0)).rounded(px(4.0)).border_1().border_color(p.edge).bg(gpui_kit::Rgba { r: c[0], g: c[1], b: c[2], a: 1.0 }))
                .child(
                    v_flex()
                        .text_xs()
                        .text_color(p.text_muted)
                        .child(format!("R {:>3.0}   G {:>3.0}   B {:>3.0}", c[0] * 255.0, c[1] * 255.0, c[2] * 255.0))
                        .child(format!("{}  {:.0}%", t("Luminance", "亮度"), (0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]) * 100.0)),
                )
                .into_any_element(),
            None => div()
                .px_3()
                .text_xs()
                .text_color(p.text_subtle)
                .child(h_flex().gap_1().child(Icon::new(IconName::Pipette).xsmall()).child(t("Click the image with the color picker", "使用颜色取样工具单击图像")))
                .into_any_element(),
        })
}

/// The selected sector's band: which hues and saturations it takes in, how
/// softly its edges fall off, and how strongly it pulls colors together.
fn band_controls(state: &Entity<AppState>, scope: Scope, sector: usize, band: &ColorBand, cx: &App) -> impl IntoElement + use<> {
    let st = state.clone();
    v_flex()
        .child(caption(t("Band", "色带"), cx))
        .child(param(state, format!("cb-hl-{sector}"), t("Hue range, low", "色相范围下限"), scoped!(scope, color_editor.bands[sector].hue_low), -90.0, 0.0, cx).unit("°").default_value(-25.0))
        .child(param(state, format!("cb-hh-{sector}"), t("Hue range, high", "色相范围上限"), scoped!(scope, color_editor.bands[sector].hue_high), 0.0, 90.0, cx).unit("°").default_value(25.0))
        .child(param(state, format!("cb-sl-{sector}"), t("Saturation range, low", "饱和度范围下限"), scoped!(scope, color_editor.bands[sector].sat_low), -100.0, 0.0, cx).default_value(-100.0))
        .child(param(state, format!("cb-sh-{sector}"), t("Saturation range, high", "饱和度范围上限"), scoped!(scope, color_editor.bands[sector].sat_high), 0.0, 100.0, cx).default_value(100.0))
        .child(param(state, format!("cb-sm-{sector}"), t("Smoothness", "平滑度"), scoped!(scope, color_editor.bands[sector].smoothness), 1.0, 29.1, cx).default_value(10.0).decimals(1))
        .child(param(state, format!("cb-uh-{sector}"), t("Uniformity · Hue", "均匀度 · 色相"), scoped!(scope, color_editor.bands[sector].homogeneity[0]), 0.0, 100.0, cx))
        .child(param(state, format!("cb-us-{sector}"), t("Uniformity · Saturation", "均匀度 · 饱和度"), scoped!(scope, color_editor.bands[sector].homogeneity[1]), 0.0, 100.0, cx))
        .child(param(state, format!("cb-ul-{sector}"), t("Uniformity · Lightness", "均匀度 · 亮度"), scoped!(scope, color_editor.bands[sector].homogeneity[2]), 0.0, 100.0, cx))
        .child(switch_row(format!("cb-inv-{sector}"), t("Invert band", "反转色带"), band.inverted, move |v, _, cx| {
            st.update(cx, |s, cx| {
                let scope = s.scope();
                s.edit(t("Invert band", "反转色带"), |a| {
                    with_scoped(a, scope, |a| &mut a.color_editor, |l| &mut l.color_editor, |ce| ce.bands[sector].inverted = v)
                }, cx)
            })
        }, cx))
}
