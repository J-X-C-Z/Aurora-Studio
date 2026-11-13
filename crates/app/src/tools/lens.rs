//! Lens tab: lens correction, rotation & flip, crop and keystone.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _,
    button::{Button, ButtonVariants as _},
    h_flex, v_flex,
};
use gpui_kit::{AnyElement, App, Entity, IntoElement, ParentElement, SharedString, Styled, Window, div};
use lumen_engine::*;

use crate::global;
use crate::i18n::t;
use crate::state::{AppState, CursorTool};
use crate::theme::pal;
use crate::widgets::*;

pub fn render(state: &Entity<AppState>, _window: &mut Window, cx: &mut App) -> AnyElement {
    v_flex()
        .w_full()
        .child(lens_panel(state, cx))
        .child(rotation_panel(state, cx))
        .child(crop_panel(state, cx))
        .child(keystone_panel(state, cx))
        .into_any_element()
}

/// One panel of this tab by id, for the Quick tab.
pub fn panel(id: &str, state: &Entity<AppState>, _window: &mut Window, cx: &mut App) -> Option<AnyElement> {
    Some(match id {
        "lens" => lens_panel(state, cx).into_any_element(),
        "rotation" => rotation_panel(state, cx).into_any_element(),
        "crop" => crop_panel(state, cx).into_any_element(),
        "keystone" => keystone_panel(state, cx).into_any_element(),
        _ => return None,
    })
}

fn lens_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let lens = s.adj().lens.clone();
    let profiles = s.lens_profiles.clone();
    let current = lens.profile.as_ref().and_then(|n| profiles.iter().position(|p| p == n));
    let mut options: Vec<(Option<usize>, SharedString)> = vec![(None, t("Generic", "通用").into())];
    options.extend(profiles.iter().enumerate().map(|(i, p)| (Some(i), p.clone().into())));
    let (st, st2, st3, st4) = (state.clone(), state.clone(), state.clone(), state.clone());
    let lens_name = s.current_info().and_then(|i| i.exif.lens.clone()).unwrap_or_default();
    let p = pal(cx).clone();
    ToolPanel::new(state, "lens", t("Lens Correction", "镜头校正"))
        .status(crate::contract::LENS_CORRECTION)
        .modified(lens != LensCorrection::default())
        .on_reset(move |_, cx| st.update(cx, |s, cx| s.edit(t("Reset lens correction", "复位镜头校正"), |a| a.lens = Default::default(), cx)))
        .child(row(
            t("Profile", "配置文件"),
            select("lens-profile", options, current, move |i, _, cx| {
                st2.update(cx, |s, cx| {
                    let name = i.and_then(|i| s.lens_profiles.get(i).cloned());
                    s.edit(t("Lens profile", "镜头配置文件"), |a| a.lens.profile = name, cx)
                })
            }),
            cx,
        ))
        .child(row(t("Lens", "镜头"), div().text_xs().text_color(p.text_subtle).child(lens_name), cx))
        // Per-lens calibration from the lens database instead of the generic formula.
        .child(switch_row("lens-db", t("Use lens calibration", "使用镜头库校准"), lens.use_profile, {
            let st = state.clone();
            move |v, _, cx| st.update(cx, |s, cx| s.edit(t("Lens calibration", "镜头库校准"), |a| a.lens.use_profile = v, cx))
        }, cx))
        .child(param_c1(state, "lens-dist", "LensGeometricAberrationAmount", global!(lens.distortion), cx))
        .child(param_c1(state, "lens-sharp", "LensRadialBlurAmount", global!(lens.sharpness_falloff), cx))
        .child(param_c1(state, "lens-light", "LensLccLightFalloffAmount", global!(lens.light_falloff), cx))
        .child(param_c1(state, "lens-fringe", "PurpleFringingAmount", global!(lens.purple_fringing), cx).label(t("Purple Fringing", "紫边")))
        .child(switch_row("lens-ca", t("Chromatic Aberration", "色差"), lens.chromatic_aberration, move |v, _, cx| {
            st3.update(cx, |s, cx| s.edit(t("Chromatic aberration", "色差"), |a| a.lens.chromatic_aberration = v, cx))
        }, cx))
        .child(switch_row("lens-hide", t("Hide Distorted Areas", "隐藏畸变区域"), lens.hide_distorted_areas, move |v, _, cx| {
            st4.update(cx, |s, cx| s.edit(t("Hide distorted areas", "隐藏畸变区域"), |a| a.lens.hide_distorted_areas = v, cx))
        }, cx))
        .child(caption(t("Movement", "移轴"), cx))
        .child(param(state, "lens-sx", t("Shift X", "水平偏移"), global!(lens.shift_x), -100.0, 100.0, cx))
        .child(param(state, "lens-sy", t("Shift Y", "垂直偏移"), global!(lens.shift_y), -100.0, 100.0, cx))
}

fn rotation_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let g = s.adj().geometry.clone();
    let straightening = s.cursor == CursorTool::Straighten;
    let st = state.clone();
    let btn = |id: &'static str, icon: IconName, tip: &'static str, f: fn(&mut Geometry)| {
        let st = state.clone();
        IconBtn::new(id, icon, tip).on_click(move |_, cx| st.update(cx, |s, cx| s.edit(tip, |a| f(&mut a.geometry), cx)))
    };
    let st_tool = state.clone();
    ToolPanel::new(state, "rotation", t("Rotation & Flip", "旋转与翻转"))
        .modified(g.rotation != 0.0 || g.orientation != 0 || g.flip_horizontal || g.flip_vertical)
        .on_auto({
            let st = state.clone();
            move |_, cx| st.update(cx, |s, cx| s.auto_adjust(AutoTarget::Rotation, cx))
        })
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                s.edit(t("Reset rotation", "复位旋转"), |a| {
                    a.geometry.rotation = 0.0;
                    a.geometry.orientation = 0;
                    a.geometry.flip_horizontal = false;
                    a.geometry.flip_vertical = false;
                }, cx)
            })
        })
        .child(param_c1(state, "rot-angle", "RotationAdjustment", global!(geometry.rotation), cx))
        .child(
            h_flex()
                .px_3()
                .pt_1()
                .gap_1()
                .child(btn("rot-ccw", IconName::RotateCcwSquare, t("Rotate left", "向左旋转"), |g| g.orientation = (g.orientation + 3) % 4))
                .child(btn("rot-cw", IconName::RotateCwSquare, t("Rotate right", "向右旋转"), |g| g.orientation = (g.orientation + 1) % 4))
                .child(btn("flip-h", IconName::FlipHorizontal2, t("Flip horizontal", "水平翻转"), |g| g.flip_horizontal = !g.flip_horizontal))
                .child(btn("flip-v", IconName::FlipVertical2, t("Flip vertical", "垂直翻转"), |g| g.flip_vertical = !g.flip_vertical))
                .child(div().flex_1())
                .child(
                    Button::new("straighten-tool")
                        .xsmall()
                        .when_selected(straightening)
                        .icon(Icon::new(IconName::Ruler))
                        .label(t("Straighten", "拉直"))
                        .on_click(move |_, _, cx| {
                            st_tool.update(cx, |s, cx| {
                                let next = if s.cursor == CursorTool::Straighten { CursorTool::Select } else { CursorTool::Straighten };
                                s.set_cursor(next, cx)
                            })
                        }),
                ),
        )
}

trait SelectedExt {
    fn when_selected(self, on: bool) -> Self;
}

impl SelectedExt for Button {
    fn when_selected(self, on: bool) -> Self {
        if on { self.primary() } else { self.outline() }
    }
}

/// Sets the crop ratio and fits the crop to it around its center. `original`
/// is the oriented image's width over height.
pub(crate) fn set_aspect(adj: &mut AdjustmentSet, a: AspectRatio, original: f32) {
    adj.geometry.aspect = a;
    if let Some(r) = a.ratio(original) {
        let r = if adj.geometry.portrait { 1.0 / r } else { r };
        let k = r / original;
        let c = adj.geometry.crop;
        let (cx0, cy0) = (c.x + c.width / 2.0, c.y + c.height / 2.0);
        let (mut w, mut h) = (c.width, c.width / k);
        if h > c.height {
            h = c.height;
            w = h * k;
        }
        adj.geometry.crop = NormRect {
            x: (cx0 - w / 2.0).clamp(0.0, 1.0 - w),
            y: (cy0 - h / 2.0).clamp(0.0, 1.0 - h),
            width: w,
            height: h,
        };
    }
}

pub(crate) fn aspect_name(a: AspectRatio) -> &'static str {
    match a {
        AspectRatio::Unconstrained => t("Unconstrained", "不受约束"),
        AspectRatio::Original => t("Original", "原始比例"),
        AspectRatio::Square => "1 : 1",
        AspectRatio::R4x5 => "4 : 5",
        AspectRatio::R5x7 => "5 : 7",
        AspectRatio::R2x3 => "2 : 3",
        AspectRatio::R3x4 => "3 : 4",
        AspectRatio::R16x9 => "16 : 9",
        AspectRatio::R65x24 => "65 : 24",
    }
}

fn crop_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let g = s.adj().geometry.clone();
    let cropping = s.cursor == CursorTool::Crop;
    let info = s.current_info().cloned();
    let p = pal(cx).clone();
    let (st, st2, st3) = (state.clone(), state.clone(), state.clone());
    let (ow, oh) = info.map(|i| if g.orientation % 2 == 1 { (i.height as f32, i.width as f32) } else { (i.width as f32, i.height as f32) }).unwrap_or((1.0, 1.0));
    let original = ow / oh;
    let crop = g.crop;
    ToolPanel::new(state, "crop", t("Crop", "裁剪"))
        .modified(g.crop != NormRect::FULL || g.aspect != AspectRatio::Unconstrained)
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                s.edit(t("Reset crop", "复位裁剪"), |a| {
                    a.geometry.crop = NormRect::FULL;
                    a.geometry.aspect = AspectRatio::Unconstrained;
                }, cx)
            })
        })
        .child(row(
            t("Ratio", "比例"),
            select("crop-ratio", AspectRatio::ALL.iter().map(|a| (*a, aspect_name(*a).into())).collect(), g.aspect, move |a, _, cx| {
                st2.update(cx, |s, cx| {
                    s.edit(t("Crop ratio", "裁剪比例"), |adj| {
                        set_aspect(adj, a, original);
                    }, cx)
                })
            }),
            cx,
        ))
        .child(
            v_flex()
                .px_3()
                .py_1()
                .gap_0p5()
                .text_xs()
                .text_color(p.text_muted)
                .child(format!("X {:.0}   Y {:.0}", crop.x * ow, crop.y * oh))
                .child(format!("{} {:.0} × {:.0} px", t("Size", "尺寸"), crop.width * ow, crop.height * oh)),
        )
        .child(
            h_flex()
                .px_3()
                .gap_1p5()
                .child(
                    Button::new("crop-tool")
                        .xsmall()
                        .when_selected(cropping)
                        .icon(Icon::new(IconName::Crop))
                        .label(if cropping { t("Done", "完成") } else { t("Crop tool", "裁剪工具") })
                        .on_click(move |_, _, cx| {
                            st3.update(cx, |s, cx| {
                                let next = if s.cursor == CursorTool::Crop { CursorTool::Select } else { CursorTool::Crop };
                                s.set_cursor(next, cx)
                            })
                        }),
                )
                .child(
                    Button::new("crop-orient").xsmall().outline().icon(Icon::new(IconName::RectangleVertical)).label(t("Portrait/Landscape", "横/竖")).on_click({
                        let st = state.clone();
                        move |_, _, cx| {
                            st.update(cx, |s, cx| {
                                s.edit(t("Crop orientation", "裁剪方向"), |a| {
                                    a.geometry.portrait = !a.geometry.portrait;
                                    let c = a.geometry.crop;
                                    let (w, h) = (c.height.min(1.0), c.width.min(1.0));
                                    a.geometry.crop = NormRect { x: (c.x + c.width / 2.0 - w / 2.0).clamp(0.0, 1.0 - w), y: (c.y + c.height / 2.0 - h / 2.0).clamp(0.0, 1.0 - h), width: w, height: h };
                                }, cx)
                            })
                        }
                    }),
                ),
        )
}

fn keystone_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let k = s.adj().geometry.keystone.clone();
    let active = s.cursor == CursorTool::Keystone;
    let (st, st2, st3) = (state.clone(), state.clone(), state.clone());
    ToolPanel::new(state, "keystone", t("Keystone", "梯形校正"))
        .status(crate::contract::KEYSTONE)
        .modified(k != Keystone { amount: 100.0, ..Default::default() })
        .on_auto(move |_, cx| st3.update(cx, |s, cx| s.auto_adjust(AutoTarget::Keystone, cx)))
        .on_reset(move |_, cx| st.update(cx, |s, cx| s.edit(t("Reset keystone", "复位梯形校正"), |a| {
            a.geometry.keystone = Keystone { amount: 100.0, ..Default::default() };
            a.extra.remove("KeystoneSkew");
        }, cx)))
        .child(param_c1(state, "ks-v", "KeystoneV", global!(geometry.keystone.vertical), cx).range(-100.0, 100.0).decimals(0))
        .child(param_c1(state, "ks-h", "KeystoneH", global!(geometry.keystone.horizontal), cx).range(-100.0, 100.0).decimals(0))
        .child(param(state, "ks-amount", t("Amount", "数量"), global!(geometry.keystone.amount), 0.0, 100.0, cx).default_value(100.0))
        .child(param_c1(state, "ks-aspect", "KeystoneAspect", global!(geometry.keystone.aspect), cx).range(-50.0, 50.0).decimals(0))
        .child(
            param_c1(state, "ks-skew", "KeystoneSkew", crate::state::extra_binding("KeystoneSkew", 0.0), cx)
                .label(t("Skew", "偏斜"))
                .flag(crate::contract::RETOUCH),
        )
        .child(div().px_3().pt_1().child(
            Button::new("ks-tool").xsmall().when_selected(active).icon(Icon::new(IconName::Grid2x2Check)).label(t("Keystone guides", "梯形参考线")).on_click(move |_, _, cx| {
                st2.update(cx, |s, cx| {
                    let next = if s.cursor == CursorTool::Keystone { CursorTool::Select } else { CursorTool::Keystone };
                    s.set_cursor(next, cx)
                })
            }),
        ))
}
