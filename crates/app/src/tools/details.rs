//! Details tab: navigator, sharpening, noise reduction, film grain, moiré and spot removal.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _,
    button::{Button, ButtonVariants as _},
    h_flex, v_flex,
};
use gpui_kit::{
    AnyElement, App, Entity, InteractiveElement, IntoElement, ObjectFit, ParentElement, SharedString, Styled, StyledImage as _, Window, div, img, prelude::FluentBuilder as _, px, relative,
};
use lumen_engine::*;

use crate::i18n::t;
use crate::state::{AppState, Binding, CursorTool, Scope};
use crate::theme::{hexa, pal};
use crate::widgets::surface::Surface;
use crate::widgets::*;
use crate::{global, scoped};

pub fn render(state: &Entity<AppState>, _window: &mut Window, cx: &mut App) -> AnyElement {
    let scope = state.read(cx).scope();
    v_flex()
        .w_full()
        .child(navigator_panel(state, cx))
        .child(sharpening_panel(state, scope, cx))
        .child(noise_panel(state, scope, cx))
        .child(moire_panel(state, scope, cx))
        .when(scope == Scope::Global, |d| d.child(grain_panel(state, cx)).child(lens_blur_panel(state, cx)).child(spots_panel(state, cx)))
        .into_any_element()
}

/// One panel of this tab by id, for the Quick tab.
pub fn panel(id: &str, state: &Entity<AppState>, _window: &mut Window, cx: &mut App) -> Option<AnyElement> {
    let scope = state.read(cx).scope();
    Some(match id {
        "navigator" => navigator_panel(state, cx).into_any_element(),
        "sharpening" => sharpening_panel(state, scope, cx).into_any_element(),
        "noise" => noise_panel(state, scope, cx).into_any_element(),
        "moire" => moire_panel(state, scope, cx).into_any_element(),
        "grain" if scope == Scope::Global => grain_panel(state, cx).into_any_element(),
        "lens-blur" if scope == Scope::Global => lens_blur_panel(state, cx).into_any_element(),
        "spots" if scope == Scope::Global => spots_panel(state, cx).into_any_element(),
        _ => return None,
    })
}

/// Binds a field of an optional local structure, creating it from the background's copy on first write.
pub(crate) fn opt_binding<T: Clone + Default + 'static>(
    scope: Scope,
    global: fn(&mut AdjustmentSet) -> &mut T,
    local: fn(&mut LocalAdjustments) -> &mut Option<T>,
    read_global: fn(&AdjustmentSet) -> &T,
    read_local: fn(&LocalAdjustments) -> &Option<T>,
    field: fn(&mut T) -> &mut f32,
    read: fn(&T) -> f32,
) -> Binding {
    Binding::new(
        move |a: &AdjustmentSet| match scope {
            Scope::Global => read(read_global(a)),
            Scope::Layer(id) => a
                .layers
                .iter()
                .find(|l| l.id == id)
                .and_then(|l| read_local(&l.adjustments).as_ref())
                .map(read)
                .unwrap_or_else(|| read(read_global(a))),
        },
        move |a: &mut AdjustmentSet, v: f32| match scope {
            Scope::Global => *field(global(a)) = v,
            Scope::Layer(id) => {
                let base = global(a).clone();
                if let Some(l) = a.layers.iter_mut().find(|l| l.id == id) {
                    *field(local(&mut l.adjustments).get_or_insert(base)) = v;
                }
            }
        },
    )
}

fn navigator_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let p = pal(cx).clone();
    let thumb = s.current.and_then(|id| s.thumbs.get(&id).cloned());
    let region = s.visible_region();
    let zoomed = s.zoom.scale.is_some();
    let (fw, fh) = s.frame_size();
    let aspect = fw / fh.max(1.0);
    let w = 270.0f32;
    let h = (w / aspect).min(200.0);
    let w = h * aspect;
    let (st, st2) = (state.clone(), state.clone());
    let zoom_pct = (s.effective_scale() / s.scale_factor.max(0.5) * 100.0).round();
    let st3 = state.clone();
    ToolPanel::new(state, "navigator", t("Navigator", "导航器")).badge(format!("{zoom_pct:.0}%")).child(
        v_flex()
            .items_center()
            .gap_1p5()
            .child(
                Surface::new("navigator")
                    .w(px(w))
                    .h(px(h))
                    .bg(p.well)
                    .cursor_pointer()
                    .when_some(thumb, |d, t| d.child(img(t).absolute().size_full().object_fit(ObjectFit::Fill)))
                    .when(zoomed, |d| {
                        d.child(div().absolute().size_full().bg(hexa(0x000000, 0.35)))
                            .child(
                                div()
                                    .absolute()
                                    .left(relative(region.x))
                                    .top(relative(region.y))
                                    .w(relative(region.width))
                                    .h(relative(region.height))
                                    .border_2()
                                    .border_color(p.accent)
                                    .bg(hexa(0xffffff, 0.08)),
                            )
                    })
                    .on_down(move |ptr, _, cx| {
                        st.update(cx, |s, cx| {
                            if s.zoom.scale.is_some() {
                                let z = s.zoom.scale;
                                s.set_zoom(z, Some((ptr.pos.0.clamp(0.0, 1.0), ptr.pos.1.clamp(0.0, 1.0))), cx);
                            }
                        })
                    })
                    .on_drag(move |ptr, _, cx| {
                        st2.update(cx, |s, cx| {
                            let c = s.zoom.center;
                            s.pan_by(ptr.pos.0.clamp(0.0, 1.0) - c.0, ptr.pos.1.clamp(0.0, 1.0) - c.1, cx)
                        })
                    }),
            )
            .child(
                h_flex().gap_1().children([(None, t("Fit", "适合")), (Some(0.5), "50%"), (Some(1.0), "100%"), (Some(2.0), "200%")].into_iter().map(move |(z, label)| {
                    let st = st3.clone();
                    Button::new(SharedString::from(format!("nav-{label}"))).xsmall().ghost().label(label).on_click(move |_, _, cx| {
                        st.update(cx, |s, cx| {
                            let sf = s.scale_factor;
                            let c = s.zoom.center;
                            s.set_zoom(z.map(|z| z * sf), Some(c), cx)
                        })
                    })
                })),
            ),
    )
}

fn sharpening_panel(state: &Entity<AppState>, scope: Scope, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let sh = match scope {
        Scope::Global => s.adj().sharpening.clone(),
        Scope::Layer(id) => s.adj().layers.iter().find(|l| l.id == id).and_then(|l| l.adjustments.sharpening.clone()).unwrap_or_else(|| s.adj().sharpening.clone()),
    };
    let st = state.clone();
    let b = |field: fn(&mut Sharpening) -> &mut f32, read: fn(&Sharpening) -> f32| {
        opt_binding(scope, |a| &mut a.sharpening, |l| &mut l.sharpening, |a| &a.sharpening, |l| &l.sharpening, field, read)
    };
    ToolPanel::new(state, "sharpening", t("Sharpening", "锐化"))
        .modified(sh != Sharpening::default())
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                let scope = s.scope();
                s.edit(t("Reset sharpening", "复位锐化"), |a| match scope {
                    Scope::Global => a.sharpening = Default::default(),
                    Scope::Layer(id) => {
                        if let Some(l) = a.layers.iter_mut().find(|l| l.id == id) {
                            l.adjustments.sharpening = None
                        }
                    }
                }, cx)
            })
        })
        .child(param_c1(state, "sh-amount", "USMAmount", b(|s| &mut s.amount, |s| s.amount), cx))
        .child(param_c1(state, "sh-radius", "USMRadius", b(|s| &mut s.radius, |s| s.radius), cx))
        .child(param_c1(state, "sh-threshold", "USMThreshold", b(|s| &mut s.threshold, |s| s.threshold), cx))
        .child(param_c1(state, "sh-halo", "USMHalo", b(|s| &mut s.halo_suppression, |s| s.halo_suppression), cx))
}

fn noise_panel(state: &Entity<AppState>, scope: Scope, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let nr = match scope {
        Scope::Global => s.adj().noise_reduction.clone(),
        Scope::Layer(id) => s.adj().layers.iter().find(|l| l.id == id).and_then(|l| l.adjustments.noise_reduction.clone()).unwrap_or_else(|| s.adj().noise_reduction.clone()),
    };
    let st = state.clone();
    let b = |field: fn(&mut NoiseReduction) -> &mut f32, read: fn(&NoiseReduction) -> f32| {
        opt_binding(scope, |a| &mut a.noise_reduction, |l| &mut l.noise_reduction, |a| &a.noise_reduction, |l| &l.noise_reduction, field, read)
    };
    ToolPanel::new(state, "noise", t("Noise Reduction", "降噪"))
        .modified(nr != NoiseReduction::default())
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                let scope = s.scope();
                s.edit(t("Reset noise reduction", "复位降噪"), |a| match scope {
                    Scope::Global => {
                        a.noise_reduction = Default::default();
                        a.extra.remove("SurfaceSmoothnessAmount");
                    }
                    Scope::Layer(id) => {
                        if let Some(l) = a.layers.iter_mut().find(|l| l.id == id) {
                            l.adjustments.noise_reduction = None
                        }
                    }
                }, cx)
            })
        })
        .child(param_c1(state, "nr-lum", "NoiseReductionAmount", b(|n| &mut n.luminance, |n| n.luminance), cx))
        .child(param_c1(state, "nr-details", "NoiseDetailsAmount", b(|n| &mut n.details, |n| n.details), cx))
        .when(scope == Scope::Global, |d| {
            d.child(
                param_c1(state, "nr-surface", "SurfaceSmoothnessAmount", crate::state::extra_binding("SurfaceSmoothnessAmount", 0.0), cx)
                    .label(t("Surface", "表面"))
                    .flag(crate::contract::SURFACE_NOISE),
            )
        })
        .child(param_c1(state, "nr-color", "ColorNoiseReductionAmount", b(|n| &mut n.color, |n| n.color), cx))
        .child(param_c1(state, "nr-pixel", "CleanNoiseAmount", b(|n| &mut n.single_pixel, |n| n.single_pixel), cx))
        // Capture One's AI denoise and banding reduction: stored under their C1
        // keys in `extra` so the engine can pick them up as soon as it renders them.
        .when(scope == Scope::Global, |d| {
            d.child(
                param(state, "nr-ai", t("AI Denoise", "AI 降噪"), crate::state::extra_binding("AIDenoiseAmount", 0.0), 0.0, 100.0, cx).flag(crate::contract::AI_DENOISE),
            )
            .child(
                param(state, "nr-banding", t("Banding Reduction", "条纹抑制"), crate::state::extra_binding("BandingReductionAmount", 0.0), 0.0, 100.0, cx).flag(crate::contract::BANDING),
            )
        })
}

fn moire_panel(state: &Entity<AppState>, scope: Scope, cx: &App) -> impl IntoElement + use<> {
    let m = crate::state::read_scoped(state.read(cx).adj(), scope, |a| &a.moire, |l| &l.moire).cloned().unwrap_or_default();
    let st = state.clone();
    ToolPanel::new(state, "moire", t("Moiré", "摩尔纹"))
        .status(crate::contract::MOIRE)
        .modified(m != Moire::default())
        .on_reset(move |_, cx| {
            st.update(cx, |s, cx| {
                let scope = s.scope();
                s.edit(t("Reset moiré", "复位摩尔纹"), |a| crate::state::with_scoped(a, scope, |a| &mut a.moire, |l| &mut l.moire, |m| *m = Default::default()), cx)
            })
        })
        .child(param_c1(state, "moire-amount", "MoireAmount", scoped!(scope, moire.amount), cx))
        .child(param_c1(state, "moire-pattern", "MoirePattern", scoped!(scope, moire.pattern), cx))
}

fn grain_name(g: GrainKind) -> &'static str {
    match g {
        GrainKind::FineGrain => t("Fine Grain", "细颗粒"),
        GrainKind::SilverRich => t("Silver Rich", "富银"),
        GrainKind::SoftGrain => t("Soft Grain", "柔和颗粒"),
        GrainKind::CubicGrain => t("Cubic Grain", "立方颗粒"),
        GrainKind::TabularGrain => t("Tabular Grain", "扁平颗粒"),
        GrainKind::HarshGrain => t("Harsh Grain", "粗糙颗粒"),
    }
}

/// Lightroom-style lens blur, built ahead of the engine. The values live in
/// `AdjustmentSet::extra` under `aurora:LensBlur*` keys (engines ignore keys they
/// do not render); the request in `engine-requests.md` names them.
const BOKEH: [(&str, &str); 5] = [("Circle", "圆形"), ("Bubble", "泡泡"), ("5 blade", "五叶"), ("Ring", "环形"), ("Cat eye", "猫眼")];

fn lens_blur_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let extra = &s.adj().extra;
    let on = extra.get("aurora:LensBlurAmount").is_some_and(|v| *v > 0.0);
    let bokeh = extra.get("aurora:LensBlurBokeh").copied().unwrap_or(0.0) as usize;
    let modified = extra.keys().any(|k| k.starts_with("aurora:LensBlur"));
    let (st, st2) = (state.clone(), state.clone());
    let key = |k: &str| crate::state::extra_binding(k.to_string(), 0.0);
    ToolPanel::new(state, "lens-blur", t("Lens Blur", "镜头模糊"))
        .status(crate::contract::LENS_BLUR)
        .modified(modified)
        .on_reset(move |_, cx| st.update(cx, |s, cx| s.edit(t("Reset lens blur", "复位镜头模糊"), |a| a.extra.retain(|k, _| !k.starts_with("aurora:LensBlur")), cx)))
        .child(param(state, "lb-amount", t("Blur amount", "模糊量"), key("aurora:LensBlurAmount"), 0.0, 100.0, cx))
        .child(param(state, "lb-near", t("Focal range, near", "焦点范围 · 近"), key("aurora:LensBlurFocusNear"), 0.0, 100.0, cx).disabled(!on))
        .child(param(state, "lb-far", t("Focal range, far", "焦点范围 · 远"), crate::state::extra_binding("aurora:LensBlurFocusFar", 100.0), 0.0, 100.0, cx).default_value(100.0).disabled(!on))
        .child(param(state, "lb-boost", t("Boost highlights", "高光增强"), key("aurora:LensBlurBoost"), 0.0, 100.0, cx).disabled(!on))
        .child(row(
            t("Bokeh", "焦外形状"),
            select("lb-bokeh", BOKEH.iter().enumerate().map(|(i, (en, zh))| (i, SharedString::from(t(en, zh)))).collect(), bokeh.min(BOKEH.len() - 1), move |i, _, cx| {
                st2.update(cx, |s, cx| {
                    s.edit(t("Bokeh", "焦外形状"), |a| {
                        if i == 0 { a.extra.remove("aurora:LensBlurBokeh"); } else { a.extra.insert("aurora:LensBlurBokeh".into(), i as f32); }
                    }, cx)
                })
            }),
            cx,
        ))
}

fn grain_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let g = state.read(cx).adj().film_grain.clone();
    let (st, st2) = (state.clone(), state.clone());
    ToolPanel::new(state, "grain", t("Film Grain", "胶片颗粒"))
        .modified(g != FilmGrain::default())
        .on_reset(move |_, cx| st.update(cx, |s, cx| s.edit(t("Reset grain", "复位颗粒"), |a| a.film_grain = Default::default(), cx)))
        .child(row(
            t("Type", "类型"),
            select("grain-type", GrainKind::ALL.iter().map(|k| (*k, grain_name(*k).into())).collect(), g.kind, move |k, _, cx| {
                st2.update(cx, |s, cx| s.edit(t("Grain type", "颗粒类型"), |a| a.film_grain.kind = k, cx))
            }),
            cx,
        ))
        .child(param_c1(state, "grain-impact", "FilmGrainAmount", global!(film_grain.impact), cx))
        .child(param_c1(state, "grain-gran", "FilmGrainGranularity", global!(film_grain.granularity), cx))
}

fn spots_panel(state: &Entity<AppState>, cx: &App) -> impl IntoElement + use<> {
    let s = state.read(cx);
    let spots = s.adj().spots.clone();
    let active = s.cursor == CursorTool::Spot;
    let p = pal(cx).clone();
    let (st, st2) = (state.clone(), state.clone());
    let size = s.panels.brush_size;
    ToolPanel::new(state, "spots", t("Spot Removal", "污点去除"))
        .status(crate::contract::SPOTS)
        .modified(!spots.is_empty())
        .badge(format!("{}", spots.len()))
        .on_reset(move |_, cx| st.update(cx, |s, cx| s.edit(t("Remove all spots", "移除所有污点"), |a| a.spots.clear(), cx)))
        .child(
            h_flex().px_3().pb_1().gap_1p5().child(
                Button::new("spot-tool")
                    .xsmall()
                    .map(|b| if active { b.primary() } else { b.outline() })
                    .icon(Icon::new(IconName::Bandage))
                    .label(if active { t("Click image to add", "单击图像添加") } else { t("Heal & clone tool", "修复与仿制工具") })
                    .on_click(move |_, _, cx| {
                        st2.update(cx, |s, cx| {
                            let next = if s.cursor == CursorTool::Spot { CursorTool::Select } else { CursorTool::Spot };
                            s.set_cursor(next, cx)
                        })
                    }),
            ),
        )
        .child(
            ParamSlider::new("spot-size", t("Size", "大小"), size * 100.0, 0.5, 20.0).decimals(1).unit("%").default_value(5.0).on_change({
                let st = state.clone();
                move |v, _, cx| {
                    st.update(cx, |s, cx| {
                        s.panels.brush_size = v / 100.0;
                        cx.notify();
                    })
                }
            }),
        )
        .children(spots.iter().enumerate().map(|(i, sp)| {
            let id = sp.id;
            let kind = sp.kind;
            let st = state.clone();
            let st_k = state.clone();
            h_flex()
                .id(SharedString::from(format!("spot-{id}")))
                .h(px(24.0))
                .px_3()
                .gap_2()
                .text_xs()
                .text_color(p.text_muted)
                .hover(|d| d.bg(p.hover))
                .child(Icon::new(IconName::CircleDot).xsmall())
                .child(div().flex_1().child(format!("{} {}", t("Spot", "污点"), i + 1)))
                .child(
                    Button::new(SharedString::from(format!("spot-kind-{id}")))
                        .xsmall()
                        .ghost()
                        .label(if kind == SpotKind::Dust { t("Dust", "灰尘") } else { t("Spot", "斑点") })
                        .on_click(move |_, _, cx| {
                            st_k.update(cx, |s, cx| {
                                s.edit(t("Spot type", "污点类型"), |a| {
                                    if let Some(sp) = a.spots.iter_mut().find(|s| s.id == id) {
                                        sp.kind = if sp.kind == SpotKind::Dust { SpotKind::Spot } else { SpotKind::Dust };
                                    }
                                }, cx)
                            })
                        }),
                )
                .child(IconBtn::new(SharedString::from(format!("spot-del-{id}")), IconName::X, t("Remove", "移除")).size(20.0).on_click(move |_, cx| {
                    st.update(cx, |s, cx| s.edit(t("Remove spot", "移除污点"), |a| a.spots.retain(|s| s.id != id), cx))
                }))
        }))
}
