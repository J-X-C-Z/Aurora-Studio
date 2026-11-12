//! Exposure tab: histogram, exposure, high dynamic range, levels, curve, clarity,
//! dehaze and vignetting. Everything except vignetting follows the selected layer.

use gpui_kit::{AnyElement, App, Entity, IntoElement, ParentElement, Styled, Window, div, prelude::FluentBuilder as _, px};
use lumen_engine::*;

use crate::i18n::t;
use crate::state::{AppState, Scope, read_scoped, with_scoped};
use crate::theme::{hex, pal};
use crate::widgets::*;
use crate::{global, scoped};

pub fn render(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    let s = state.read(cx);
    let scope = s.scope();
    div()
        .w_full()
        .child(histogram_panel(state, cx))
        .child(scopes_panel(state, cx))
        .child(exposure_panel(state, scope, cx))
        .child(hdr_panel(state, scope, cx))
        .child(levels_panel(state, scope, window, cx))
        .child(curve_panel(state, scope, cx))
        .child(clarity_panel(state, scope, cx))
        .child(dehaze_panel(state, scope, cx))
        .when(scope == Scope::Global, |this| this.child(vignetting_panel(state, cx)))
        .into_any_element()
}

/// One panel of this tab by id, for the Quick tab.
pub fn panel(id: &str, state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> Option<AnyElement> {
    let scope = state.read(cx).scope();
    Some(match id {
        "histogram" => histogram_panel(state, cx).into_any_element(),
        "scopes" => scopes_panel(state, cx).into_any_element(),
        "exposure" => exposure_panel(state, scope, cx).into_any_element(),
        "hdr" => hdr_panel(state, scope, cx).into_any_element(),
        "levels" => levels_panel(state, scope, window, cx).into_any_element(),
        "curve" => curve_panel(state, scope, cx).into_any_element(),
        "clarity" => clarity_panel(state, scope, cx).into_any_element(),
        "dehaze" => dehaze_panel(state, scope, cx).into_any_element(),
        "vignetting" if scope == Scope::Global => vignetting_panel(state, cx).into_any_element(),
        _ => return None,
    })
}

/// Waveform, RGB parade and vectorscope of the preview. Off by default so a
/// frame costs nothing extra until someone looks.
fn scopes_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let kind = s.panels.scope_kind;
    let image = s.preview.scope.clone().filter(|_| kind != crate::state::ScopeKind::Off);
    let p = pal(cx).clone();
    let st = state.clone();
    let index = crate::state::ScopeKind::ALL.iter().position(|k| *k == kind).unwrap_or(0);
    ToolPanel::new(state, "scopes", t("Scopes", "示波器"))
        .child(
            div().px_3().pb_1().child(
                Segmented::new("scope-kind", index)
                    .text(t("Off", "关"))
                    .text(t("Waveform", "波形"))
                    .text(t("Parade", "分量"))
                    .text(t("Vector", "矢量"))
                    .on_select(move |i, _, cx| {
                        st.update(cx, |s, cx| {
                            s.panels.scope_kind = crate::state::ScopeKind::ALL[i];
                            s.request_render(lumen_engine::RenderQuality::Final, cx);
                            cx.notify();
                        })
                    }),
            ),
        )
        .when_some(image, |d, img| {
            let square = kind == crate::state::ScopeKind::Vectorscope;
            d.child(
                div().px_3().pb_2().flex().justify_center().child(
                    gpui_kit::img(img)
                        .when(square, |i| i.size(px(180.0)))
                        .when(!square, |i| i.w_full().h(px(140.0)))
                        .rounded(px(4.0))
                        .border_1()
                        .border_color(p.divider),
                ),
            )
        })
}

pub fn histogram_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let mode = match s.panels.histogram_mode {
        1 => HistogramMode::Luma,
        2 => HistogramMode::Red,
        3 => HistogramMode::Green,
        4 => HistogramMode::Blue,
        _ => HistogramMode::Rgb,
    };
    let h = s.preview.histogram.clone();
    let info = s.current_info().cloned();
    let p = pal(cx).clone();
    let st = state.clone();
    ToolPanel::new(state, "histogram", t("Histogram", "直方图"))
        .child(
            div()
                .px_3()
                .child(HistogramView::new("main-histogram", h).mode(mode).height(100.0))
                .child(
                    gpui_kit::component::h_flex()
                        .mt_1p5()
                        .gap_2()
                        .child(
                            div().w(px(150.0)).child(
                                Segmented::new("hist-mode", s.panels.histogram_mode as usize)
                                    .text("RGB")
                                    .text(t("Luma", "亮度"))
                                    .swatch(hex(0xff4a4a), "R")
                                    .swatch(hex(0x4ade6a), "G")
                                    .swatch(hex(0x4a8aff), "B")
                                    .height(20.0)
                                    .on_select(move |i, _, cx| {
                                        st.update(cx, |s, cx| {
                                            s.panels.histogram_mode = i as u8;
                                            cx.notify();
                                        })
                                    }),
                            ),
                        )
                        .child(div().flex_1())
                        .child(div().text_xs().text_color(p.text_subtle).child(info.map_or(String::new(), |i| {
                            let e = &i.exif;
                            format!(
                                "ISO {}  {}  f/{:.1}",
                                e.iso.unwrap_or(0),
                                e.shutter.map_or("-".into(), |s| if s >= 1.0 { format!("{s:.0}s") } else { format!("1/{:.0}", 1.0 / s) }),
                                e.aperture.unwrap_or(0.0)
                            )
                        }))),
                ),
        )
}

fn exposure_panel(state: &Entity<AppState>, scope: Scope, cx: &App) -> impl IntoElement + use<> {
    let e = read_scoped(state.read(cx).adj(), scope, |a| &a.exposure, |l| &l.exposure).cloned().unwrap_or_default();
    let modified = e != Exposure::default();
    let st = state.clone();
    let st2 = state.clone();
    ToolPanel::new(state, "exposure", t("Exposure", "曝光"))
        .modified(modified)
        .on_auto(move |_, cx| st2.update(cx, |s, cx| s.auto_adjust(AutoTarget::Exposure, cx)))
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                let scope = s.scope();
                s.edit(t("Reset exposure", "复位曝光"), |a| with_scoped(a, scope, |a| &mut a.exposure, |l| &mut l.exposure, |e| *e = Exposure::default()), cx)
            })
        })
        .child(param_c1(state, "exp-exposure", "Exposure", scoped!(scope, exposure.exposure), cx))
        .child(param_c1(state, "exp-contrast", "Contrast", scoped!(scope, exposure.contrast), cx))
        .child(param_c1(state, "exp-brightness", "Brightness", scoped!(scope, exposure.brightness), cx))
        .child(
            param_c1(state, "exp-saturation", "Saturation", scoped!(scope, exposure.saturation), cx)
                .track(Track::Gradient(vec![hex(0x808080), hex(0x9a8a7a), hex(0xd0603a)])),
        )
}

fn hdr_panel(state: &Entity<AppState>, scope: Scope, cx: &App) -> impl IntoElement + use<> {
    let h = read_scoped(state.read(cx).adj(), scope, |a| &a.hdr, |l| &l.hdr).cloned().unwrap_or_default();
    let st = state.clone();
    let st2 = state.clone();
    ToolPanel::new(state, "hdr", t("High Dynamic Range", "高动态范围"))
        .modified(h != HighDynamicRange::default())
        .on_auto(move |_, cx| st2.update(cx, |s, cx| s.auto_adjust(AutoTarget::HighDynamicRange, cx)))
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                let scope = s.scope();
                s.edit(t("Reset HDR", "复位高动态范围"), |a| with_scoped(a, scope, |a| &mut a.hdr, |l| &mut l.hdr, |h| *h = Default::default()), cx)
            })
        })
        .child(param_c1(state, "hdr-highlights", "HighlightRecovery", scoped!(scope, hdr.highlights), cx))
        .child(param_c1(state, "hdr-shadows", "ShadowRecovery", scoped!(scope, hdr.shadows), cx))
        .child(param_c1(state, "hdr-whites", "WhiteRecovery", scoped!(scope, hdr.whites), cx))
        .child(param_c1(state, "hdr-blacks", "BlackRecovery", scoped!(scope, hdr.blacks), cx))
}

fn channel_color(c: ToneChannel) -> gpui_kit::Hsla {
    match c {
        ToneChannel::Rgb | ToneChannel::Luma => hex(0xe6e6ea),
        ToneChannel::Red => hex(0xff5b5b),
        ToneChannel::Green => hex(0x55d672),
        ToneChannel::Blue => hex(0x5b95ff),
    }
}

fn channel_bins(h: &Histogram, c: ToneChannel) -> Vec<f32> {
    let bins = match c {
        ToneChannel::Red => &h.red,
        ToneChannel::Green => &h.green,
        ToneChannel::Blue => &h.blue,
        ToneChannel::Rgb | ToneChannel::Luma => &h.luma,
    };
    crate::widgets::paint::normalize_bins(bins, 96)
}

fn levels_panel(state: &Entity<AppState>, scope: Scope, _window: &mut Window, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let channel = s.panels.levels_channel;
    let levels = read_scoped(s.adj(), scope, |a| &a.levels, |l| &l.levels).cloned().unwrap_or_default();
    let value = levels.channel(channel).clone();
    let hist = channel_bins(&s.preview.histogram, channel);
    let (st, st2, st3, st4) = (state.clone(), state.clone(), state.clone(), state.clone());
    let idx = ToneChannel::LEVELS.iter().position(|c| *c == channel).unwrap_or(0);
    ToolPanel::new(state, "levels", t("Levels", "色阶"))
        .modified(levels != Levels::default())
        .on_auto(move |_, cx| st4.update(cx, |s, cx| s.auto_adjust(AutoTarget::Levels, cx)))
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                let scope = s.scope();
                s.edit(t("Reset levels", "复位色阶"), |a| with_scoped(a, scope, |a| &mut a.levels, |l| &mut l.levels, |l| *l = Default::default()), cx)
            })
        })
        .child(
            div().px_3().pb_1p5().child(
                Segmented::new("levels-channel", idx)
                    .text("RGB")
                    .swatch(hex(0xff5b5b), t("Red", "红"))
                    .swatch(hex(0x55d672), t("Green", "绿"))
                    .swatch(hex(0x5b95ff), t("Blue", "蓝"))
                    .on_select(move |i, _, cx| {
                        st2.update(cx, |s, cx| {
                            s.panels.levels_channel = ToneChannel::LEVELS[i];
                            cx.notify();
                        })
                    }),
            ),
        )
        .child(
            LevelsView::new("levels", value, hist, channel_color(channel))
                .on_change(move |v, _, cx| {
                    st3.update(cx, |s, cx| {
                        let scope = s.scope();
                        let channel = s.panels.levels_channel;
                        s.edit_live(t("Levels", "色阶"), |a| {
                            with_scoped(a, scope, |a| &mut a.levels, |l| &mut l.levels, |l| *l.channel_mut(channel) = v)
                        }, cx)
                    })
                })
                .on_commit({
                    let st = state.clone();
                    move |_, cx| st.update(cx, |s, cx| s.end_gesture(cx))
                }),
        )
}

fn curve_panel(state: &Entity<AppState>, scope: Scope, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let channel = s.panels.curve_channel;
    let curves = read_scoped(s.adj(), scope, |a| &a.curves, |l| &l.curves).cloned().unwrap_or_default();
    let curve = curves.channel(channel).clone();
    let hist = channel_bins(&s.preview.histogram, channel);
    let idx = ToneChannel::CURVES.iter().position(|c| *c == channel).unwrap_or(0);
    let (st, st2, st3) = (state.clone(), state.clone(), state.clone());
    let p = pal(cx).clone();
    ToolPanel::new(state, "curve", t("Curve", "曲线"))
        .modified(curves != Curves::default())
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                let scope = s.scope();
                let channel = s.panels.curve_channel;
                s.edit(t("Reset curve", "复位曲线"), |a| {
                    with_scoped(a, scope, |a| &mut a.curves, |l| &mut l.curves, |c| *c.channel_mut(channel) = Curve::default())
                }, cx)
            })
        })
        .child(
            div().px_3().pb_1p5().child(
                Segmented::new("curve-channel", idx)
                    .text("RGB")
                    .text(t("Luma", "亮度"))
                    .swatch(hex(0xff5b5b), t("Red", "红"))
                    .swatch(hex(0x55d672), t("Green", "绿"))
                    .swatch(hex(0x5b95ff), t("Blue", "蓝"))
                    .on_select(move |i, _, cx| {
                        st2.update(cx, |s, cx| {
                            s.panels.curve_channel = ToneChannel::CURVES[i];
                            cx.notify();
                        })
                    }),
            ),
        )
        .child(
            div().px_3().child(
                CurveEditor::new(format!("curve-{channel:?}"), curve, channel_color(channel))
                    .histogram(hist)
                    .on_change(move |c, _, cx| {
                        st3.update(cx, |s, cx| {
                            let scope = s.scope();
                            let channel = s.panels.curve_channel;
                            s.edit_live(t("Curve", "曲线"), |a| {
                                with_scoped(a, scope, |a| &mut a.curves, |l| &mut l.curves, |cv| *cv.channel_mut(channel) = c)
                            }, cx)
                        })
                    })
                    .on_commit({
                        let st = state.clone();
                        move |_, cx| st.update(cx, |s, cx| s.end_gesture(cx))
                    }),
            ),
        )
        .child(
            div()
                .px_3()
                .pt_1()
                .text_xs()
                .text_color(p.text_subtle)
                .child(t("Click to add · drag out or double-click to remove", "单击添加点 · 拖出或双击删除")),
        )
}

fn clarity_panel(state: &Entity<AppState>, scope: Scope, cx: &App) -> impl IntoElement + use<> {
    let c = read_scoped(state.read(cx).adj(), scope, |a| &a.clarity, |l| &l.clarity).cloned().unwrap_or_default();
    let (st, st2) = (state.clone(), state.clone());
    let method_names = [t("Natural", "自然"), t("Punch", "强烈"), t("Neutral", "中性"), t("Classic", "经典")];
    ToolPanel::new(state, "clarity", t("Clarity", "清晰度"))
        .modified(c != Clarity::default())
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                let scope = s.scope();
                s.edit(t("Reset clarity", "复位清晰度"), |a| with_scoped(a, scope, |a| &mut a.clarity, |l| &mut l.clarity, |c| *c = Default::default()), cx)
            })
        })
        .child(row(
            t("Method", "方法"),
            select(
                "clarity-method",
                ClarityMethod::ALL.iter().zip(method_names).map(|(m, n)| (*m, n.into())).collect(),
                c.method,
                move |m, _, cx| {
                    st2.update(cx, |s, cx| {
                        let scope = s.scope();
                        s.edit(t("Clarity method", "清晰度方法"), |a| with_scoped(a, scope, |a| &mut a.clarity, |l| &mut l.clarity, |c| c.method = m), cx)
                    })
                },
            ),
            cx,
        ))
        .child(param_c1(state, "clarity-clarity", "Clarity", scoped!(scope, clarity.clarity), cx).range(-100.0, 100.0).default_value(0.0))
        .child(param_c1(state, "clarity-structure", "ClarityStructure", scoped!(scope, clarity.structure), cx).range(-100.0, 100.0).default_value(0.0))
}

fn dehaze_panel(state: &Entity<AppState>, scope: Scope, cx: &App) -> impl IntoElement + use<> {
    let d = read_scoped(state.read(cx).adj(), scope, |a| &a.dehaze, |l| &l.dehaze).cloned().unwrap_or_default();
    let st = state.clone();
    ToolPanel::new(state, "dehaze", t("Dehaze", "去雾"))
        .modified(d != Dehaze::default())
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                let scope = s.scope();
                s.edit(t("Reset dehaze", "复位去雾"), |a| with_scoped(a, scope, |a| &mut a.dehaze, |l| &mut l.dehaze, |d| *d = Default::default()), cx)
            })
        })
        .child(param_c1(state, "dehaze-amount", "DehazeAmount", scoped!(scope, dehaze.amount), cx))
}

fn vignetting_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let v = state.read(cx).adj().vignetting.clone();
    let (st, st2) = (state.clone(), state.clone());
    let names = [t("Elliptic on Crop", "裁剪椭圆"), t("Circular", "圆形"), t("Circular Crop", "圆形裁剪")];
    ToolPanel::new(state, "vignetting", t("Vignetting", "暗角"))
        .modified(v != Vignetting::default())
        .on_reset(move |_, cx| st.update(cx, |s, cx| s.edit(t("Reset vignetting", "复位暗角"), |a| a.vignetting = Default::default(), cx)))
        .child(param_c1(state, "vig-amount", "LensVignettingAmount", global!(vignetting.amount), cx))
        .child(row(
            t("Method", "方法"),
            select(
                "vig-method",
                VignetteMethod::ALL.iter().zip(names).map(|(m, n)| (*m, n.into())).collect(),
                v.method,
                move |m, _, cx| st2.update(cx, |s, cx| s.edit(t("Vignetting method", "暗角方法"), |a| a.vignetting.method = m, cx)),
            ),
            cx,
        ))
}
