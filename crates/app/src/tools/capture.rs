//! Capture tab: tethered camera status and settings, capture, and next-capture
//! naming, location and adjustments.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _,
    button::{Button, ButtonVariants as _},
    h_flex, v_flex,
};
use gpui_kit::{AnyElement, App, Entity, IntoElement, ParentElement, SharedString, Styled, Window, div, px, relative};
use lumen_engine::{CameraStatus, TetherStatus};

use crate::i18n::t;
use crate::state::AppState;
use crate::theme::pal;
use crate::widgets::*;

pub fn render(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    v_flex()
        .w_full()
        .child(camera_panel(state, cx))
        .child(settings_panel(state, cx))
        .child(naming_panel(state, window, cx))
        .child(adjust_panel(state, cx))
        .into_any_element()
}

fn camera_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let capturing = s.capturing;
    let probing = s.tether_probing;
    let st = state.clone();
    let recheck = {
        let st = state.clone();
        move |id: &'static str| {
            let st = st.clone();
            Button::new(id).outline().small().loading(probing).icon(Icon::new(IconName::RefreshCw)).label(t("Check again", "重新检测")).on_click(move |_, _, cx| st.update(cx, |s, cx| s.probe_tether(cx)))
        }
    };
    let body: AnyElement = match (s.tether.clone(), s.camera.clone()) {
        (Some(TetherStatus::Available { lib_version }), camera) => {
            // libgphoto2 opened a camera; model, battery and card come from
            // `camera_status` only when the engine reads them.
            let (model, detail) = match camera {
                CameraStatus::Connected { model, battery, shots_remaining } => (model, Some((battery, shots_remaining))),
                CameraStatus::Disconnected => (t("Camera", "相机").to_string(), None),
            };
            v_flex()
                .px_3()
                .gap_2()
                .child(
                    h_flex()
                        .gap_2()
                        .child(div().size(px(32.0)).rounded(px(6.0)).bg(p.well).flex().items_center().justify_center().child(Icon::new(IconName::Camera).text_color(p.success)))
                        .child(
                            v_flex()
                                .child(div().text_sm().text_color(p.text).child(model))
                                .child(div().text_xs().text_color(p.success).child(format!("{} · libgphoto2 {lib_version}", t("Connected via USB", "已通过 USB 连接")))),
                        ),
                )
                .children(detail.map(|(battery, shots_remaining)| {
                    let used = 1.0 - (shots_remaining as f32 / 2000.0).min(1.0);
                    v_flex()
                        .gap_2()
                        .child(
                            h_flex()
                                .gap_2()
                                .text_xs()
                                .text_color(p.text_muted)
                                .child(Icon::new(IconName::BatteryMedium).xsmall())
                                .child(
                                    div().flex_1().h(px(6.0)).rounded_full().bg(p.well).child(
                                        div().h_full().w(relative(battery)).rounded_full().bg(if battery < 0.2 { p.danger } else if battery < 0.4 { p.warning } else { p.success }),
                                    ),
                                )
                                .child(format!("{:.0}%", battery * 100.0)),
                        )
                        .child(ely_core::data_display::Meter::new("card-meter", t("Card", "存储卡"), used).detail(format!("{shots_remaining} {}", t("shots left", "张剩余"))))
                }))
                .child(
                    Button::new("capture")
                        .primary()
                        .w_full()
                        .loading(capturing)
                        .icon(Icon::new(IconName::Aperture))
                        .label(t("Capture", "拍摄"))
                        .on_click(move |_, _, cx| st.update(cx, |s, cx| s.capture(cx))),
                )
                .child(Button::new("live-view").outline().w_full().small().icon(Icon::new(IconName::Video)).label(t("Live View", "实时取景")).on_click(|_, window, cx| window.dispatch_action(Box::new(crate::actions::LiveView), cx)))
                .into_any_element()
        }
        (tether, _) => v_flex()
            .px_3()
            .py_3()
            .items_center()
            .gap_1()
            .text_color(p.text_subtle)
            .child(Icon::new(IconName::CameraOff).size(px(28.0)))
            .child(div().text_xs().text_color(p.text_muted).child(match tether {
                None => t("Looking for a camera…", "正在检测相机…"),
                Some(TetherStatus::NoLibrary) => t("Tethering unavailable", "联机拍摄不可用"),
                _ => t("No camera connected", "未连接相机"),
            }))
            .child(div().text_xs().text_center().child(s.tether_note()))
            .child(h_flex().pt_1().child(recheck("tether-recheck")))
            .into_any_element(),
    };
    ToolPanel::new(state, "camera", t("Camera", "相机")).status(crate::contract::TETHER).child(body)
}

fn settings_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let settings = s.camera_settings.clone().filter(|_| s.camera_ready());
    let tile = |label: &'static str, value: String| {
        v_flex()
            .flex_1()
            .p_2()
            .rounded(px(5.0))
            .bg(p.well)
            .child(div().text_xs().text_color(p.text_subtle).child(label))
            .child(div().text_sm().text_color(p.text).child(value))
    };
    ToolPanel::new(state, "camera-settings", t("Camera Settings", "相机设置")).status(crate::contract::TETHER).child(match settings {
        Some(c) => v_flex()
            .px_3()
            .gap_1p5()
            .child(h_flex().gap_1p5().child(tile(t("ISO", "ISO"), c.iso)).child(tile(t("Shutter", "快门"), c.shutter)).child(tile(t("Aperture", "光圈"), c.aperture)))
            .child(
                h_flex()
                    .gap_1p5()
                    .child(tile(t("Exp. Comp.", "曝光补偿"), format!("{:+.1}", c.exposure_comp)))
                    .child(tile(t("White Bal.", "白平衡"), c.white_balance))
                    .child(tile(t("Drive", "驱动"), c.drive_mode)),
            )
            .into_any_element(),
        None => div().px_3().text_xs().text_color(p.text_subtle).child(t("Settings appear when a camera is connected.", "连接相机后显示设置。")).into_any_element(),
    })
}

fn sample_name(format: &str, counter: u32) -> String {
    format.replace("[Job]", "Session").replace("[Counter]", &format!("{counter:04}")).replace("[Date]", "2026-11-10").replace("[Camera]", "IQ4")
}

fn naming_panel(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let pr = s.panels.clone();
    let (st, st2, st3) = (state.clone(), state.clone(), state.clone());
    let format_field = text_field("cap-format", pr.capture_format.clone(), "[Job]_[Counter]", move |v, _, cx| {
        st.update(cx, |s, cx| {
            s.panels.capture_format = v;
            cx.notify();
        })
    }, window, cx);
    let folder_field = text_field("cap-folder", pr.capture_folder.clone(), "Capture", move |v, _, cx| {
        st2.update(cx, |s, cx| {
            s.panels.capture_folder = v;
            cx.notify();
        })
    }, window, cx);
    let tokens = ["[Job]", "[Counter]", "[Date]", "[Camera]"];
    ToolPanel::new(state, "capture-naming", t("Next Capture Naming", "下次拍摄命名"))
        .child(row(t("Format", "格式"), format_field, cx))
        .child(h_flex().px_3().py_1().gap_1().children(tokens.iter().map(|tok| {
            let st = st3.clone();
            let tok = *tok;
            Button::new(SharedString::from(format!("tok-{tok}"))).xsmall().ghost().label(tok).on_click(move |_, _, cx| {
                st.update(cx, |s, cx| {
                    s.panels.capture_format.push_str(tok);
                    cx.notify();
                })
            })
        })))
        .child(row(t("Sample", "示例"), div().text_xs().text_color(p.accent).child(format!("{}.IIQ", sample_name(&pr.capture_format, pr.capture_counter))), cx))
        .child(
            ParamSlider::new("cap-counter", t("Counter", "计数器"), pr.capture_counter as f32, 1.0, 9999.0).step(1.0).on_change({
                let st = state.clone();
                move |v, _, cx| {
                    st.update(cx, |s, cx| {
                        s.panels.capture_counter = v.round() as u32;
                        cx.notify();
                    })
                }
            }),
        )
        .child(caption(t("Next Capture Location", "下次拍摄位置"), cx))
        .child(row(t("Folder", "文件夹"), folder_field, cx))
}

fn adjust_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let mode = s.panels.capture_adjust;
    let st = state.clone();
    ToolPanel::new(state, "capture-adjust", t("Next Capture Adjustments", "下次拍摄调整")).child(
        div().px_3().child(
            Segmented::new("cap-adj", mode as usize)
                .text(t("Defaults", "默认"))
                .text(t("Copy last", "沿用上一张"))
                .text(t("Style", "风格"))
                .on_select(move |i, _, cx| {
                    st.update(cx, |s, cx| {
                        s.panels.capture_adjust = i as u8;
                        cx.notify();
                    })
                }),
        ),
    )
}
