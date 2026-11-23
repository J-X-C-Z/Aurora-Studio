//! Portrait tab: Capture One's Retouch tools (face & skin, hair, teeth, eyes) and
//! the portrait algorithm catalogue from `docs/frontend-contract/`.
//!
//! Values are stored in `AdjustmentSet::extra`, so they follow undo, copy/paste,
//! styles and the sidecar. Effects with available same-job inputs render; the
//! rest stay marked as needing face geometry or model output.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _,
    button::Button,
    h_flex, v_flex,
};
use gpui_kit::{AnyElement, App, Entity, IntoElement, ParentElement, Styled, Window, div, px};

use crate::contract::{self, BANDS, Band, PortraitParam, Status};
use crate::i18n::t;
use crate::state::{AppState, Binding, extra_binding};
use crate::theme::pal;
use crate::widgets::*;

const RETOUCH_GROUPS: [(&str, &str, &str); 4] = [
    ("RetouchFaceSkin", "Face & Skin", "面部与皮肤"),
    ("RetouchEyes", "Eyes", "眼睛"),
    ("RetouchHair", "Hair", "头发"),
    ("RetouchTeeth", "Teeth", "牙齿"),
];

/// Effects whose CPU/CUDA kernels have their same-job inputs wired. Values are
/// stored as 0..=1 in `AdjustmentSet::extra`; the sliders show 0..=100.
const WIRED_SKIN_EFFECTS: &[(&str, &str, &str, f32)] = &[
    ("pc:21015", "Skin hue", "肤色色相", 0.5),
    ("pc:30096", "Skin temperature", "色调", 0.5),
    ("pc:90005", "Whitening", "美白", 0.0),
    ("pc:90007", "Rosy", "红润", 0.0),
    ("pc:8068", "Skin gloss", "皮肤光泽", 0.5),
    ("pc:8069", "Skin texture", "肤质", 0.5),
    ("pc:8070", "Skin clarity", "清晰度", 0.0),
    ("pc:30021", "Face smooth (HLF)", "中性灰/高低频磨皮", 0.0),
    ("pc:30022", "Body smooth (HLF)", "全身磨皮", 0.0),
];

/// Teeth effects wired to canonical FaceParsingV2 (`fbe.onnx`) on CUDA.
const WIRED_TEETH_EFFECTS: &[(&str, &str, &str, f32)] = &[
    ("pc:108", "Teeth whitening", "美牙/牙齿美白", 0.0),
];

/// Face geometry reshape effects wired to real 194->414 landmarks + CUDA mesh raster.
/// Stored as -100..=100 in `AdjustmentSet::extra`.
const WIRED_FACE_GEOMETRY_EFFECTS: &[(&str, &str, &str)] = &[
    ("pc:30092", "Face width (L)", "窄脸（左）"),
    ("pc:30093", "Face width (R)", "窄脸（右）"),
    ("pc:242", "Sharp chin (L)", "尖下巴（左）"),
    ("pc:243", "Sharp chin (R)", "尖下巴（右）"),
    ("pc:8023", "Jawline (L)", "下颌角（左）"),
    ("pc:8024", "Jawline (R)", "下颌角（右）"),
    ("pc:8027", "Cheekbone (L)", "颧骨（左）"),
    ("pc:8028", "Cheekbone (R)", "颧骨（右）"),
    ("pc:8029", "Temple (L)", "太阳穴（左）"),
    ("pc:8030", "Temple (R)", "太阳穴（右）"),
    ("pc:30080", "Forehead width (L)", "额头宽窄（左）"),
    ("pc:30081", "Forehead width (R)", "额头宽窄（右）"),
    ("pc:30082", "Forehead height (L)", "额头高低（左）"),
    ("pc:30083", "Forehead height (R)", "额头高低（右）"),
];

const WIRED_TOTAL: usize =
    WIRED_SKIN_EFFECTS.len() + WIRED_TEETH_EFFECTS.len() + WIRED_FACE_GEOMETRY_EFFECTS.len();

fn portrait_amount_binding(key: &'static str, default: f32) -> Binding {
    let inner = extra_binding(key, default);
    let (get, set) = (inner.get.clone(), inner.set.clone());
    Binding::new(move |a| get(a) * 100.0, move |a, v| set(a, v / 100.0))
}

pub fn render(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    let query = state.read(cx).panels.portrait_query.trim().to_lowercase();
    let p = pal(cx).clone();
    let st = state.clone();
    let search = text_field("portrait-search", state.read(cx).panels.portrait_query.clone(), t("Search parameters…", "搜索参数…"), move |q, _, cx| {
        st.update(cx, |s, cx| {
            s.panels.portrait_query = q;
            cx.notify();
        })
    }, window, cx);

    let mut col = v_flex()
        .w_full()
        .child(
            v_flex()
                .px_3()
                .py_2()
                .gap_1p5()
                .border_b_1()
                .border_color(p.divider)
                .child(
                    h_flex()
                        .gap_1p5()
                        .child(Icon::new(IconName::Info).xsmall().text_color(p.accent))
                        .items_start()
                        .child(div().flex_1().min_w_0().text_xs().text_color(p.text_muted).child(t(
                            "Parameters follow the frontend contract and are saved with the variant. Skin hue, whitening, rosy, gloss, texture, clarity and radiance render; effects needing face geometry or model output stay marked.",
                            "参数按前端契约解析，可调并随变体保存；肤色色相、美白、红润、光泽、肤质、清晰度和透亮已接入，需要面部几何或模型输出的效果会标记缺输入。",
                        ))),
                )
                .child(div().text_xs().text_color(p.text_subtle).child(format!(
                    "{} {}",
                    WIRED_TOTAL,
                    t(
                        "effects render live on CUDA/CPU (skin, teeth, 414-pt face reshape); the rest need extra models",
                        "个效果已实时接入 CUDA/CPU 渲染（皮肤、美牙、414点面部精修）；其余待接入专用模型"
                    ),
                )))
                .child(h_flex().gap_1p5().child(Icon::new(IconName::Search).xsmall().text_color(p.text_subtle)).child(div().flex_1().child(search))),
        );

    // Until an algorithm renders, the parameters stay folded behind one card so
    // the tab does not look like a wall of broken sliders. A search still finds them.
    let open = state.read(cx).panels.portrait_open;
    if !open && query.is_empty() {
        let s = state.read(cx);
        let retouch: Vec<&'static str> =
            s.engine.param_metadata().iter().filter(|m| RETOUCH_GROUPS.iter().any(|(g, ..)| m.group == *g)).map(|m| m.key).collect();
        let total = retouch.len() + contract::BANDS.iter().map(|b| b.params().count()).sum::<usize>();
        let set = s.adj().extra.keys().filter(|k| k.starts_with("pc:") || retouch.contains(&k.as_str())).count();
        let st = state.clone();
        return col
            .child(
                v_flex()
                    .m_3()
                    .p_3()
                    .gap_2()
                    .rounded(px(6.0))
                    .bg(p.well)
                    .child(div().text_sm().text_color(p.text).child(t("Portrait tools are partially rendered", "人像工具已部分接入渲染")))
                    .child(div().text_xs().text_color(p.text_muted).child(format!(
                        "{} {} · {} {}",
                        total,
                        t("parameters parsed", "个参数已解析"),
                        set,
                        t("set on this image", "个已在本图设置"),
                    )))
                    .child(
                        Button::new("portrait-open")
                            .outline()
                            .small()
                            .label(t("Show the parameters anyway", "仍然显示参数"))
                            .on_click(move |_, _, cx| st.update(cx, |s, cx| {
                                s.panels.portrait_open = true;
                                cx.notify();
                            })),
                    ),
            )
            .into_any_element();
    }

    col = col.child(wired_face_geometry_panel(state, &query, cx));
    for (group, en, zh) in RETOUCH_GROUPS {
        col = col.child(retouch_panel(state, group, t(en, zh), &query, cx));
    }
    col = col.child(caption(t("Portrait algorithms", "人像算法"), cx));
    for band in BANDS.iter() {
        col = col.child(band_panel(state, band, &query, cx));
    }
    col.into_any_element()
}

fn matches(query: &str, label: &str, key: &str, id: Option<i64>) -> bool {
    query.is_empty()
        || label.to_lowercase().contains(query)
        || key.to_lowercase().contains(query)
        || id.is_some_and(|id| id.to_string() == query)
}

fn wired_face_geometry_panel(state: &Entity<AppState>, query: &str, cx: &App) -> AnyElement {
    let s = state.read(cx);
    let shown: Vec<_> = WIRED_FACE_GEOMETRY_EFFECTS
        .iter()
        .filter(|(key, en, zh)| matches(query, t(*en, *zh), *key, None))
        .collect();
    if shown.is_empty() && !query.is_empty() {
        return div().into_any_element();
    }
    let keys: Vec<&'static str> = WIRED_FACE_GEOMETRY_EFFECTS.iter().map(|(k, ..)| *k).collect();
    let modified = keys.iter().any(|k| s.adj().extra.contains_key(*k));
    let st = state.clone();
    ToolPanel::new(state, "retouch-face-geometry", t("Face Reshape (CUDA)", "面部精修（实时）"))
        .status(Status::Ready)
        .modified(modified)
        .on_reset(move |_, cx| {
            let keys = keys.clone();
            st.update(cx, |s, cx| {
                s.edit(t("Reset face reshape", "复位面部精修"), |a| {
                    for k in &keys {
                        a.extra.remove(*k);
                    }
                }, cx)
            })
        })
        .children(shown.into_iter().map(|(key, en, zh)| {
            param(
                state,
                format!("fg-{key}"),
                t(*en, *zh),
                extra_binding(*key, 0.0),
                -100.0,
                100.0,
                cx,
            )
            .default_value(0.0)
        }))
        .into_any_element()
}

fn retouch_panel(state: &Entity<AppState>, group: &'static str, title: &'static str, query: &str, cx: &App) -> AnyElement {
    let s = state.read(cx);
    let metas: Vec<_> = s.engine.param_metadata().iter().filter(|m| m.group == group).collect();
    let wired: Vec<_> = match group {
        "RetouchFaceSkin" => WIRED_SKIN_EFFECTS
            .iter()
            .filter(|(key, en, zh, _)| matches(query, t(*en, *zh), *key, None))
            .collect(),
        "RetouchTeeth" => WIRED_TEETH_EFFECTS
            .iter()
            .filter(|(key, en, zh, _)| matches(query, t(*en, *zh), *key, None))
            .collect(),
        _ => Vec::new(),
    };
    let mut keys: Vec<&'static str> = metas.iter().map(|m| m.key).collect();
    keys.extend(wired.iter().map(|(key, ..)| *key));
    let modified = keys.iter().any(|k| s.adj().extra.contains_key(*k));
    let shown: Vec<_> = metas.into_iter().filter(|m| matches(query, t(m.label, m.label_zh), m.key, None)).collect();
    if shown.is_empty() && wired.is_empty() && !query.is_empty() {
        return div().into_any_element();
    }
    let st = state.clone();
    let (id, status): (&'static str, Status) = match group {
        "RetouchFaceSkin" => (
            "retouch-face",
            Status::Partial(
                "9 skin effects render live on CUDA/CPU; C1 legacy sliders await binding",
                "9 项皮肤效果已实时渲染；其余旧版滑块待绑定",
            ),
        ),
        "RetouchEyes" => ("retouch-eyes", contract::RETOUCH),
        "RetouchHair" => ("retouch-hair", contract::RETOUCH),
        _ => (
            "retouch-teeth",
            Status::Partial(
                "PF108 teeth whitening renders live via FaceParsingV2 on CUDA",
                "美牙（PF108）已通过 FaceParsingV2 实时渲染",
            ),
        ),
    };
    ToolPanel::new(state, id, title)
        .status(status)
        .modified(modified)
        .on_reset(move |_, cx| {
            let keys = keys.clone();
            st.update(cx, |s, cx| {
                s.edit(t("Reset retouch", "复位修饰"), |a| {
                    for k in &keys {
                        a.extra.remove(*k);
                    }
                }, cx)
            })
        })
        .children(shown.into_iter().map(|m| param_c1(state, format!("rt-{}", m.key), m.key, extra_binding(m.key, m.default), cx)))
        .children(wired.into_iter().map(|(key, en, zh, default)| {
            param(
                state,
                format!("rt-{key}"),
                t(*en, *zh),
                portrait_amount_binding(*key, *default),
                0.0,
                100.0,
                cx,
            )
            .default_value(*default * 100.0)
        }))
        .into_any_element()
}

fn band_panel(state: &Entity<AppState>, band: &'static Band, query: &str, cx: &App) -> AnyElement {
    let s = state.read(cx);
    let params: Vec<&'static PortraitParam> = band.params().collect();
    let keys: Vec<String> = params.iter().map(|p| p.store_key()).collect();
    let set = s.adj().extra.keys().filter(|k| keys.iter().any(|p| *k == p || k.starts_with(&format!("{p}:")))).count();
    let shown: Vec<_> = params.iter().copied().filter(|p| matches(query, &p.label(), &p.key, Some(p.id))).collect();
    if shown.is_empty() && !query.is_empty() {
        return div().into_any_element();
    }
    // While searching, show hits even in a closed band; otherwise build rows only when open.
    let open = !query.is_empty() || !s.panels.collapsed.contains(band.id);
    let st = state.clone();
    let mut panel = ToolPanel::new(state, band.id, band.title())
        .status(Status::Partial("Skin effects render; face geometry and model-dependent effects need live inputs", "皮肤类效果已接入；面部几何与依赖模型输出的效果需要实时输入"))
        .badge(if set > 0 { format!("{set}/{}", params.len()) } else { format!("{}", params.len()) })
        .modified(set > 0)
        .force_open(!query.is_empty())
        .on_reset(move |_, cx| {
            let keys = keys.clone();
            st.update(cx, |s, cx| {
                s.edit(t("Reset portrait", "复位人像"), |a| a.extra.retain(|k, _| !keys.iter().any(|p| k == p || k.starts_with(&format!("{p}:")))), cx)
            })
        });
    if open {
        panel = panel.children(shown.into_iter().map(|param| param_portrait(state, param, cx)));
        let unresolved = band.unresolved();
        if !unresolved.is_empty() && query.is_empty() {
            let p = pal(cx);
            let ids = unresolved.iter().map(|id| format!("#{id}")).collect::<Vec<_>>().join(" ");
            panel = panel.child(
                h_flex()
                    .px_3()
                    .py_1()
                    .gap_1p5()
                    .items_start()
                    .text_size(px(10.0))
                    .text_color(p.text_disabled)
                    .child(Icon::new(IconName::CircleDashed).xsmall())
                    .child(div().flex_1().min_w_0().child(format!(
                        "{} {} · {ids}",
                        unresolved.len(),
                        t("unresolved, inferred from the id band only; no control", "个参数未解析，仅按 ID 段推断，不提供控件"),
                    ))),
            );
        }
    }
    panel.into_any_element()
}

