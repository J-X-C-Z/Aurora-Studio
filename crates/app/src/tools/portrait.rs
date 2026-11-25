//! Portrait tab: Capture One's Retouch tools (face & skin, hair, teeth, eyes) and
//! the portrait algorithm catalogue from `docs/frontend-contract/`.
//!
//! Values are stored in `AdjustmentSet::extra`, so they follow undo, copy/paste,
//! styles and the sidecar. Effects with available same-job inputs render; the
//! rest stay marked as needing face geometry or model output.
//!
//! The tab opens on the controls the renderer uses ("Live"); the parameter
//! catalogue, which mostly only stores values, is one switch away ("All") and
//! always reachable through search.

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _,
    h_flex, v_flex,
};
use gpui_kit::{
    AnyElement, App, Entity, InteractiveElement, IntoElement, ParentElement, SharedString, StatefulInteractiveElement, Styled, Window,
    div, prelude::FluentBuilder as _, px,
};

use crate::contract::{self, BANDS, Band, PortraitParam, Status};
use crate::evidence;
use crate::i18n::t;
use crate::state::{AppState, Binding, extra_binding};
use crate::theme::pal;
use crate::widgets::*;

/// Capture One's legacy retouch groups: engine group, panel id, title.
const RETOUCH_GROUPS: [(&str, &str, &str, &str); 4] = [
    ("RetouchFaceSkin", "retouch-face", "Face & Skin", "面部与皮肤"),
    ("RetouchEyes", "retouch-eyes", "Eyes", "眼睛"),
    ("RetouchHair", "retouch-hair", "Hair", "头发"),
    ("RetouchTeeth", "retouch-teeth", "Teeth", "牙齿"),
];

/// How the live skin, wrinkle, eye, hair and teeth sliders are grouped in their panel.
const SKIN_SECTIONS: [(&str, &str, &[&str]); 6] = [
    ("Skin tone & Lip", "肤色与唇妆", &["pc:21015", "pc:30096", "pc:90005", "pc:90007", "pc:90004", "pc:21105", "pc:90086", "pc:90021"]),
    ("Texture", "质感", &["pc:8068", "pc:8069", "pc:8070"]),
    ("Smoothing & Neutral Grey", "磨皮与中性灰", &["pc:30021", "pc:30022", "pc:90002", "pc:90010", "pc:30013", "pc:30014"]),
    ("Wrinkles", "祛皱", &["pc:90023", "pc:90024", "pc:21024", "pc:21025", "pc:21069", "pc:21070", "pc:21071"]),
    ("Eyes & Hair", "眼部与头发", &["pc:100004", "pc:101", "pc:30009"]),
    ("Teeth", "牙齿", &["pc:108", "pc:109", "pc:110"]),
];

/// Names of the left/right pairs in `WIRED_FACE_GEOMETRY_EFFECTS`, in its order.
const FACE_PAIRS: [(&str, &str); 20] = [
    ("Face width", "窄脸"),
    ("V-face", "V 脸"),
    ("Sharp chin", "尖下巴"),
    ("Jawline", "下颌角"),
    ("Cheekbone", "颧骨"),
    ("Temple", "太阳穴"),
    ("Forehead width", "额头宽窄"),
    ("Forehead height", "额头高低"),
    ("Eye width", "眼宽"),
    ("Eye height", "眼高"),
    ("Eye distance", "眼距"),
    ("Eye tilt", "眼角倾斜"),
    ("Pupil size", "眼瞳大小"),
    ("Lower eyelid", "眼睑下至"),
    ("Brow thickness", "眉粗细"),
    ("Brow distance", "眉距"),
    ("Brow shape", "眉形"),
    ("Brow position", "眉位"),
    ("Brow head length", "眉头长短"),
    ("Brow tail length", "眉尾长短"),
];

/// Effects whose CPU/CUDA kernels have their same-job inputs wired. Values are
/// stored as 0..=1 in `AdjustmentSet::extra`; the sliders show 0..=100.
const WIRED_SKIN_EFFECTS: &[(&str, &str, &str, f32)] = &[
    ("pc:21015", "Skin hue", "肤色色相", 0.5),
    ("pc:30096", "Skin temperature", "色调", 0.5),
    ("pc:90005", "Whitening", "美白", 0.0),
    ("pc:90007", "Rosy", "红润", 0.0),
    ("pc:90004", "Skin radiance", "皮肤透亮", 0.0),
    ("pc:8068", "Skin gloss", "皮肤光泽", 0.5),
    ("pc:8069", "Skin texture", "肤质", 0.5),
    ("pc:8070", "Skin clarity", "清晰度", 0.0),
    ("pc:30021", "Face smooth (HLF)", "中性灰/高低频磨皮", 0.0),
    ("pc:30022", "Body smooth (HLF)", "全身磨皮", 0.0),
    ("pc:30013", "Neutral grey flat", "中性灰平", 0.0),
    ("pc:30014", "Neutral grey 3D", "中性灰 3D", 0.0),
    ("pc:90002", "Moisture smooth (face)", "水润磨皮（脸部）", 0.0),
    ("pc:90010", "Moisture smooth (body)", "水润磨皮（身体）", 0.0),
    ("pc:21105", "AI body skin tone unify", "AI 身体肤色统一", 0.0),
    ("pc:90086", "Body skin tone unify", "肤色统一（身体）", 0.0),
    ("pc:90023", "Eye wrinkles", "祛眼纹", 0.0),
    ("pc:90024", "Forehead wrinkles", "祛抬头纹", 0.0),
    ("pc:21024", "Glabella wrinkles", "祛眉间纹", 0.0),
    ("pc:21025", "Puppet wrinkles", "祛木偶纹", 0.0),
    ("pc:21069", "Cheek wrinkles", "祛面颊纹", 0.0),
    ("pc:21070", "Nose back wrinkles", "祛鼻背纹", 0.0),
    ("pc:21071", "Perioral wrinkles", "祛口周纹", 0.0),
    ("pc:90021", "Lip makeup enhance", "唇妆增强", 0.5),
];

/// Eye effects wired to FaceParsingV2 / face-input pipeline.
const WIRED_EYES_EFFECTS: &[(&str, &str, &str, f32)] = &[
    ("pc:100004", "Red blood removal", "去红血丝", 0.0),
    ("pc:101", "Silkworm", "卧蚕", 0.0),
];

/// Hair effects wired to FaceParsingV2 / same-job producer inputs.
const WIRED_HAIR_EFFECTS: &[(&str, &str, &str, f32)] = &[
    ("pc:30009", "Hair color unify", "统一发色", 0.0),
];

/// Teeth effects wired to canonical FaceParsingV2 (`fbe.onnx`) on CUDA.
const WIRED_TEETH_EFFECTS: &[(&str, &str, &str, f32)] = &[
    ("pc:108", "Teeth whitening", "美牙/牙齿美白", 0.0),
    ("pc:109", "Teeth brightness", "牙齿亮白", 0.0),
    ("pc:110", "Teeth de-yellow", "牙齿去黄", 0.0),
];

/// Face geometry reshape effects wired to real 194->414 landmarks + CUDA mesh raster.
/// Stored as -100..=100 in `AdjustmentSet::extra`.
const WIRED_FACE_GEOMETRY_EFFECTS: &[(&str, &str, &str)] = &[
    ("pc:30092", "Face width (L)", "窄脸（左）"),
    ("pc:30093", "Face width (R)", "窄脸（右）"),
    ("pc:247", "V-face (L)", "V 脸（左）"),
    ("pc:248", "V-face (R)", "V 脸（右）"),
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
    ("pc:8019", "Eye width (L)", "眼宽（左）"),
    ("pc:8020", "Eye width (R)", "眼宽（右）"),
    ("pc:8013", "Eye height (L)", "眼高（左）"),
    ("pc:8014", "Eye height (R)", "眼高（右）"),
    ("pc:8015", "Eye distance (L)", "眼距（左）"),
    ("pc:8016", "Eye distance (R)", "眼距（右）"),
    ("pc:8017", "Eye tilt (L)", "眼角倾斜（左）"),
    ("pc:8018", "Eye tilt (R)", "眼角倾斜（右）"),
    ("pc:8011", "Pupil size (L)", "眼瞳大小（左）"),
    ("pc:8012", "Pupil size (R)", "眼瞳大小（右）"),
    ("pc:30073", "Lower eyelid (L)", "眼睑下至（左）"),
    ("pc:30079", "Lower eyelid (R)", "眼睑下至（右）"),
    ("pc:8001", "Brow thickness (L)", "眉粗细（左）"),
    ("pc:8002", "Brow thickness (R)", "眉粗细（右）"),
    ("pc:8005", "Brow distance (L)", "眉距（左）"),
    ("pc:8006", "Brow distance (R)", "眉距（右）"),
    ("pc:8009", "Brow shape (L)", "眉形（左）"),
    ("pc:8010", "Brow shape (R)", "眉形（右）"),
    ("pc:8021", "Brow position (L)", "眉位（左）"),
    ("pc:8022", "Brow position (R)", "眉位（右）"),
    ("pc:44124", "Brow head length (L)", "眉头长短（左）"),
    ("pc:48019", "Brow head length (R)", "眉头长短（右）"),
    ("pc:41922", "Brow tail length (L)", "眉尾长短（左）"),
    ("pc:49221", "Brow tail length (R)", "眉尾长短（右）"),
];

const WIRED_FACE_SINGLE_EFFECTS: &[(&str, &str, &str)] = &[
    ("pc:202", "Hairline", "发际线"),
    ("pc:232", "Skull top enhance", "高颅顶"),
    ("pc:8033", "Smile mouth corner", "微笑嘴角"),
];

/// Facial plumpness effects wired to `attach_plump_from_extra`.
/// Stored as -100..=100 in `AdjustmentSet::extra`.
const WIRED_PLUMP_EFFECTS: &[(&str, &str, &str)] = &[
    ("pc:30210", "Full face plump", "全脸丰盈"),
    ("pc:30191", "Forehead plump", "额头丰盈"),
    ("pc:30194", "Mid-face plump", "面中丰盈"),
    ("pc:30197", "Cheek plump", "面颊丰盈"),
    ("pc:30198", "Nasal base plump", "鼻基底丰盈"),
    ("pc:30203", "Brow arch plump", "眉弓丰盈"),
    ("pc:30204", "Chin plump", "下巴丰盈"),
    ("pc:30207", "Mouth corner lift", "嘴角上扬"),
];

const WIRED_TOTAL: usize = WIRED_SKIN_EFFECTS.len()
    + WIRED_EYES_EFFECTS.len()
    + WIRED_HAIR_EFFECTS.len()
    + WIRED_TEETH_EFFECTS.len()
    + WIRED_FACE_GEOMETRY_EFFECTS.len()
    + WIRED_FACE_SINGLE_EFFECTS.len()
    + WIRED_PLUMP_EFFECTS.len();

pub(crate) fn has_face_reshape(adjustments: &aurora_engine::AdjustmentSet) -> bool {
    WIRED_FACE_GEOMETRY_EFFECTS
        .iter()
        .chain(WIRED_FACE_SINGLE_EFFECTS)
        .chain(WIRED_PLUMP_EFFECTS)
        .any(|(key, ..)| {
            adjustments.extra.get(*key).is_some_and(|amount| amount.abs() > 1.0e-6)
        })
}

pub(crate) fn has_live_portrait_effect(adjustments: &aurora_engine::AdjustmentSet) -> bool {
    has_face_reshape(adjustments)
        || WIRED_SKIN_EFFECTS
            .iter()
            .chain(WIRED_EYES_EFFECTS)
            .chain(WIRED_HAIR_EFFECTS)
            .chain(WIRED_TEETH_EFFECTS)
            .any(|(key, _, _, neutral)| {
                adjustments.extra.get(*key).is_some_and(|amount| (amount - neutral).abs() > 1.0e-6)
            })
}

#[cfg(test)]
#[test]
fn every_live_skin_and_teeth_effect_has_one_section() {
    let listed: Vec<&str> = SKIN_SECTIONS.iter().flat_map(|(_, _, keys)| keys.iter().copied()).collect();
    // `SKIN_SECTIONS` also groups the wired eye and hair effects, so they are
    // part of the same one-section-per-live-effect check.
    let live: Vec<&str> = WIRED_SKIN_EFFECTS
        .iter()
        .chain(WIRED_EYES_EFFECTS)
        .chain(WIRED_HAIR_EFFECTS)
        .chain(WIRED_TEETH_EFFECTS)
        .map(|e| e.0)
        .collect();
    assert_eq!(listed.len(), live.len());
    assert!(live.iter().all(|k| listed.iter().filter(|l| *l == k).count() == 1));
}

#[cfg(test)]
#[test]
fn face_pairs_line_up_left_then_right() {
    assert_eq!(WIRED_FACE_GEOMETRY_EFFECTS.len(), FACE_PAIRS.len() * 2);
    for (sides, (en, _)) in WIRED_FACE_GEOMETRY_EFFECTS.chunks(2).zip(FACE_PAIRS) {
        assert_eq!(sides[0].1, format!("{en} (L)"));
        assert_eq!(sides[1].1, format!("{en} (R)"));
    }
}

#[cfg(test)]
#[test]
fn live_portrait_backing_excludes_neutral_values() {
    let mut adjustments = aurora_engine::AdjustmentSet::default();
    adjustments.extra.insert("pc:8069".into(), 0.5);
    adjustments.extra.insert("pc:90021".into(), 0.5);
    adjustments.extra.insert("pc:242".into(), 0.0);
    assert!(!has_live_portrait_effect(&adjustments));
    adjustments.extra.insert("pc:90005".into(), 0.5);
    assert!(has_live_portrait_effect(&adjustments));
    adjustments.extra.remove("pc:90005");
    adjustments.extra.insert("pc:242".into(), -25.0);
    assert!(has_live_portrait_effect(&adjustments));
    adjustments.extra.remove("pc:242");
    adjustments.extra.insert("pc:30210".into(), 20.0);
    assert!(has_live_portrait_effect(&adjustments));
    adjustments.extra.remove("pc:30210");
    adjustments.extra.insert("pc:90023".into(), 0.3);
    assert!(has_live_portrait_effect(&adjustments));
}

/// Catalogue bands must not duplicate a control that already has a dedicated
/// wired panel: both rows would use the same `pc:<id>` key and reset together.
fn is_wired_key(key: &str) -> bool {
    WIRED_SKIN_EFFECTS.iter().any(|row| row.0 == key)
        || WIRED_EYES_EFFECTS.iter().any(|row| row.0 == key)
        || WIRED_HAIR_EFFECTS.iter().any(|row| row.0 == key)
        || WIRED_TEETH_EFFECTS.iter().any(|row| row.0 == key)
        || WIRED_FACE_GEOMETRY_EFFECTS.iter().any(|row| row.0 == key)
        || WIRED_FACE_SINGLE_EFFECTS.iter().any(|row| row.0 == key)
        || WIRED_PLUMP_EFFECTS.iter().any(|row| row.0 == key)
}

fn portrait_amount_binding(key: &'static str, default: f32) -> Binding {
    let inner = extra_binding(key, default);
    let (get, set) = (inner.get.clone(), inner.set.clone());
    Binding::new(move |a| get(a) * 100.0, move |a, v| set(a, v / 100.0))
}

pub fn render(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    let s = state.read(cx);
    let raw_query = s.prefs.panels.portrait_query.clone();
    let query = raw_query.trim().to_lowercase();
    let show_all = s.prefs.panels.portrait_open;
    let p = pal(cx).clone();

    let retouch: Vec<&'static str> =
        s.engine.param_metadata().iter().filter(|m| RETOUCH_GROUPS.iter().any(|(g, ..)| m.group == *g)).map(|m| m.key).collect();
    let stored_total = retouch.len() + BANDS.iter().map(|b| b.params().filter(|p| !is_wired_key(&p.store_key())).count()).sum::<usize>();
    let stored_set = s.adj().extra.keys().filter(|k| !is_wired_key(k) && (k.starts_with("pc:") || retouch.contains(&k.as_str()))).count();

    let st = state.clone();
    let search = text_field("portrait-search", raw_query, t("Search all parameters…", "搜索全部参数…"), move |q, _, cx| {
        st.update(cx, |s, cx| {
            s.prefs.panels.portrait_query = q;
            cx.notify();
        })
    }, window, cx);
    let about = format!(
        "{}\n\n{} {}… · {} {}\n{}",
        t(
            "Each effect shows its seven-level evidence state. Only effects with evidence at ev5 (main preview) may be called available; stored parameters alone do not change the preview.",
            "每个效果都显示七级证据状态；只有达到 ev5（应用主预览接入）才可标为可用。仅保存参数不会改变预览。",
        ),
        t("Evidence registry", "证据表"),
        &evidence::REGISTRY_SHA256[..12],
        evidence::all().count(),
        t("registered effects", "个已登记效果"),
        evidence::REGISTRY_SOURCE,
    );
    let st = state.clone();
    let note: SharedString = if !query.is_empty() {
        t("Searching every parameter, stored-only ones included", "正在搜索全部参数，包括仅保存的参数").into()
    } else if show_all {
        t("Parameters not marked green are stored only and do not change the preview", "未标绿的参数仅保存，不会改变预览").into()
    } else {
        format!("{stored_total} {}", t("stored-only parameters hidden", "个仅保存的参数已隐藏")).into()
    };
    let header = v_flex()
        .px_3()
        .py_2()
        .gap_1p5()
        .border_b_1()
        .border_color(p.divider)
        .child(
            h_flex()
                .gap_1p5()
                .child(Icon::new(IconName::Search).xsmall().text_color(p.text_subtle))
                .child(div().flex_1().min_w_0().child(search))
                .child(
                    div()
                        .id("portrait-about")
                        .flex_shrink_0()
                        .text_color(p.text_subtle)
                        .hover(|s| s.text_color(p.text))
                        .child(Icon::new(IconName::Info).xsmall())
                        .tooltip(move |window, cx| gpui_kit::component::tooltip::Tooltip::new(about.clone()).build(window, cx)),
                ),
        )
        .child(
            Segmented::new("portrait-scope", usize::from(show_all))
                .text(format!("{} · {WIRED_TOTAL}", t("Live", "实时")))
                .text(format!("{} · {}", t("All", "全部"), WIRED_TOTAL + stored_total))
                .on_select(move |i, _, cx| {
                    st.update(cx, |s, cx| {
                        s.prefs.panels.portrait_open = i == 1;
                        cx.notify();
                    })
                }),
        )
        .child(
            h_flex()
                .gap_1()
                .text_size(px(10.0))
                .text_color(p.text_subtle)
                .child(div().flex_1().min_w_0().child(note))
                // Values hidden behind "Live" still apply; say so.
                .when(!show_all && query.is_empty() && stored_set > 0, |d| {
                    d.child(div().flex_shrink_0().text_color(p.warning).child(format!("{stored_set} {}", t("set on this image", "个已在本图设置"))))
                }),
        );

    let live = PanelStack::new(state)
        .head(header)
        .panel("portrait-skin", skin_panel(state, &query, cx))
        .panel("retouch-face-geometry", wired_face_geometry_panel(state, &query, cx))
        .panel("retouch-face-plump", wired_plump_panel(state, &query, cx));
    if !show_all && query.is_empty() {
        return live.render(cx);
    }
    let live_ids = live.ids();
    let mut stored = PanelStack::new(state)
        .also_resets(live_ids)
        .head(caption(t("Parameter catalogue", "参数目录"), cx));
    for (group, id, en, zh) in RETOUCH_GROUPS {
        stored = stored.panel(id, retouch_panel(state, group, id, t(en, zh), &query, cx));
    }
    for band in BANDS.iter() {
        stored = stored.panel(band.id, band_panel(state, band, &query, cx));
    }
    v_flex().w_full().child(live.inner().render(cx)).child(stored.render(cx)).into_any_element()
}

fn matches(query: &str, label: &str, key: &str, id: Option<i64>) -> bool {
    query.is_empty()
        || label.to_lowercase().contains(query)
        || key.to_lowercase().contains(query)
        || id.is_some_and(|id| id.to_string() == query)
}

/// One live slider for a wired skin or teeth effect.
fn live_slider(state: &Entity<AppState>, key: &'static str, en: &'static str, zh: &'static str, default: f32, cx: &App) -> AnyElement {
    let mut slider = param(state, format!("rt-{key}"), t(en, zh), portrait_amount_binding(key, default), 0.0, 100.0, cx);
    if key == "pc:8069" {
        slider = slider
            .label(t("Skin texture (smoothing)", "肤质（柔化）"))
            .range(0.0, 50.0)
            .flag(Status::Partial(
                "Lower values smooth texture; 50 is neutral. Texture enhancement is not available yet.",
                "向左调低可柔化肤质，50 为中性；增强纹理尚未接入。",
            ));
    }
    slider.default_value(default * 100.0).into_any_element()
}

/// The skin, wrinkle, eye, hair and teeth effects the renderer uses, grouped by what they change.
fn skin_panel(state: &Entity<AppState>, query: &str, cx: &App) -> AnyElement {
    let s = state.read(cx);
    let effects: Vec<_> = WIRED_SKIN_EFFECTS
        .iter()
        .chain(WIRED_EYES_EFFECTS)
        .chain(WIRED_HAIR_EFFECTS)
        .chain(WIRED_TEETH_EFFECTS)
        .collect();
    let keys: Vec<&'static str> = effects.iter().map(|e| e.0).collect();
    let modified = effects.iter().any(|(k, _, _, neutral)| s.adj().extra.get(*k).is_some_and(|v| (v - neutral).abs() > 1.0e-6));
    let mut rows: Vec<AnyElement> = Vec::new();
    for (en, zh, section) in SKIN_SECTIONS {
        let hits: Vec<_> = section
            .iter()
            .filter_map(|k| effects.iter().find(|e| e.0 == *k))
            .filter(|(key, en, zh, _)| matches(query, t(en, zh), key, None))
            .collect();
        if hits.is_empty() {
            continue;
        }
        rows.push(caption(t(en, zh), cx));
        rows.extend(hits.into_iter().map(|(key, en, zh, default)| live_slider(state, key, en, zh, *default, cx)));
    }
    if rows.is_empty() {
        return div().into_any_element();
    }
    let st = state.clone();
    ToolPanel::new(state, "portrait-skin", t("Skin & Teeth", "皮肤与牙齿"))
        .status(Status::Partial(
            "Skin effects need a suitable input; teeth whitening (PF108) renders through FaceParsingV2 on CUDA",
            "皮肤效果需要有效输入；美牙（PF108）通过 FaceParsingV2 在 CUDA 上实时渲染",
        ))
        .modified(modified)
        .on_reset(move |_, cx| {
            let keys = keys.clone();
            st.update(cx, |s, cx| {
                s.edit(t("Reset skin & teeth", "复位皮肤与牙齿"), |a| {
                    for k in &keys {
                        a.extra.remove(*k);
                    }
                }, cx)
            })
        })
        .children(rows)
        .into_any_element()
}

/// Face reshape as left/right pairs. Linked (the default), one slider moves
/// both sides; unlinked, each side has its own slider under the pair's name.
fn wired_face_geometry_panel(state: &Entity<AppState>, query: &str, cx: &App) -> AnyElement {
    let s = state.read(cx);
    let linked = s.prefs.panels.portrait_link_lr;
    let pairs: Vec<_> = WIRED_FACE_GEOMETRY_EFFECTS
        .chunks(2)
        .zip(FACE_PAIRS)
        .filter(|(sides, (en, zh))| matches(query, t(en, zh), "", None) || sides.iter().any(|(key, en, zh)| matches(query, t(en, zh), key, None)))
        .collect();
    let singles: Vec<_> = WIRED_FACE_SINGLE_EFFECTS
        .iter()
        .filter(|(key, en, zh)| matches(query, t(en, zh), key, None))
        .collect();
    if pairs.is_empty() && singles.is_empty() {
        return div().into_any_element();
    }
    let keys: Vec<&'static str> = WIRED_FACE_GEOMETRY_EFFECTS
        .iter()
        .chain(WIRED_FACE_SINGLE_EFFECTS)
        .map(|(k, ..)| *k)
        .collect();
    let modified = keys.iter().any(|k| s.adj().extra.contains_key(*k));
    let mut rows: Vec<AnyElement> = Vec::new();
    for (sides, (en, zh)) in pairs {
        let (left, right) = (sides[0].0, sides[1].0);
        if linked {
            let (l, r) = (extra_binding(left, 0.0), extra_binding(right, 0.0));
            let (lget, rget, lset, rset) = (l.get.clone(), r.get.clone(), l.set.clone(), r.set.clone());
            let both = Binding::new(move |a| (lget(a) + rget(a)) * 0.5, move |a, v| {
                lset(a, v);
                rset(a, v);
            });
            rows.push(param(state, format!("fg-{left}-lr"), t(en, zh), both, -100.0, 100.0, cx).default_value(0.0).into_any_element());
        } else {
            rows.push(caption(t(en, zh), cx));
            rows.push(param(state, format!("fg-{left}"), t("Left", "左"), extra_binding(left, 0.0), -100.0, 100.0, cx).default_value(0.0).into_any_element());
            rows.push(param(state, format!("fg-{right}"), t("Right", "右"), extra_binding(right, 0.0), -100.0, 100.0, cx).default_value(0.0).into_any_element());
        }
    }
    for (key, en, zh) in singles {
        rows.push(param(state, format!("fg-{key}"), t(en, zh), extra_binding(*key, 0.0), -100.0, 100.0, cx).default_value(0.0).into_any_element());
    }
    let (st, st_link) = (state.clone(), state.clone());
    ToolPanel::new(state, "retouch-face-geometry", t("Face Reshape", "面部精修"))
        .status(Status::Partial(
            "Face reshape requires usable landmarks on this photo",
            "面部精修需要在当前照片中检测到可用的面部关键点",
        ))
        .modified(modified)
        .action(
            HeaderAction::new(
                "link",
                if linked { IconName::Link } else { IconName::Unlink },
                if linked { t("Left and right move together — click to adjust each side", "左右联动 · 单击分别调整左右") } else { t("Sides adjust separately — click to link them", "左右分别调整 · 单击联动") },
                move |_, cx| st_link.update(cx, |s, cx| {
                    s.prefs.panels.portrait_link_lr = !s.prefs.panels.portrait_link_lr;
                    cx.notify();
                }),
            )
            .active(linked),
        )
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
        .children(rows)
        .into_any_element()
}

fn wired_plump_panel(state: &Entity<AppState>, query: &str, cx: &App) -> AnyElement {
    let s = state.read(cx);
    let shown: Vec<_> = WIRED_PLUMP_EFFECTS
        .iter()
        .filter(|(key, en, zh)| matches(query, t(*en, *zh), *key, None))
        .collect();
    if shown.is_empty() && !query.is_empty() {
        return div().into_any_element();
    }
    let keys: Vec<&'static str> = WIRED_PLUMP_EFFECTS.iter().map(|(k, ..)| *k).collect();
    let modified = keys.iter().any(|k| s.adj().extra.contains_key(*k));
    let st = state.clone();
    ToolPanel::new(state, "retouch-face-plump", t("Facial Plumpness", "面部丰盈"))
        .status(Status::Partial(
            "Facial plumpness requires face landmarks and plump ONNX models",
            "面部丰盈需要检测到面部关键点并加载丰盈模型",
        ))
        .modified(modified)
        .on_reset(move |_, cx| {
            let keys = keys.clone();
            st.update(cx, |s, cx| {
                s.edit(t("Reset plumpness", "复位面部丰盈"), |a| {
                    for k in &keys {
                        a.extra.remove(*k);
                    }
                }, cx)
            })
        })
        .children(shown.into_iter().map(|(key, en, zh)| {
            param(
                state,
                format!("fp-{key}"),
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

/// Capture One's legacy retouch sliders for one group. They are stored with the
/// variant; the live skin and teeth effects have their own panel.
fn retouch_panel(state: &Entity<AppState>, group: &'static str, id: &'static str, title: &'static str, query: &str, cx: &App) -> AnyElement {
    let s = state.read(cx);
    let metas: Vec<_> = s.engine.param_metadata().iter().filter(|m| m.group == group).collect();
    let keys: Vec<&'static str> = metas.iter().map(|m| m.key).collect();
    let modified = keys.iter().any(|k| s.adj().extra.contains_key(*k));
    let shown: Vec<_> = metas.into_iter().filter(|m| matches(query, t(m.label, m.label_zh), m.key, None)).collect();
    if shown.is_empty() {
        return div().into_any_element();
    }
    let st = state.clone();
    ToolPanel::new(state, id, title)
        .status(contract::RETOUCH)
        .badge(shown.len().to_string())
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
        .into_any_element()
}

fn band_panel(state: &Entity<AppState>, band: &'static Band, query: &str, cx: &App) -> AnyElement {
    let s = state.read(cx);
    let params: Vec<&'static PortraitParam> = band
        .params()
        .filter(|param| !is_wired_key(&param.store_key()))
        .collect();
    let keys: Vec<String> = params.iter().map(|p| p.store_key()).collect();
    let set = s.adj().extra.keys().filter(|k| keys.iter().any(|p| *k == p || k.starts_with(&format!("{p}:")))).count();
    let shown: Vec<_> = params.iter().copied().filter(|p| matches(query, &p.label(), &p.key, Some(p.id))).collect();
    let unlisted: Vec<_> = evidence::without_control(band.lo, band.hi)
        .into_iter()
        .filter(|effect| matches(query, &effect.display_label(), effect.key, Some(i64::from(effect.pf))))
        .collect();
    if shown.is_empty() && unlisted.is_empty() {
        return div().into_any_element();
    }
    // While searching, show hits even in a closed band; otherwise build rows only when open.
    let open = !query.is_empty() || !s.prefs.panels.collapsed.contains(band.id);
    let st = state.clone();
    let mut panel = ToolPanel::new(state, band.id, band.title())
        .status(band_status(band))
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
        if !unlisted.is_empty() {
            // Effects the registry knows but that have no control: listed, never dropped
            // (e.g. PF30096 sediao, PF90021 lipMakeupEnhance, PF44297).
            panel = panel.child(caption(t("Registered, no control", "已登记 · 无参数控件"), cx)).children(registry_rows(band, unlisted, cx));
        }
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

/// 段内证据摘要：达到 ev5 的项也只标“部分”，因为 ev6/ev7 仍无证据；
/// 其余一律“未接入”。
fn band_status(band: &'static Band) -> Status {
    let (wired, _total) = evidence::preview_count(band.lo, band.hi);
    if wired > 0 {
        Status::Partial(
            "Only ev5 (main preview) effects are presented as available; vendor parity and real-photo acceptance are unverified",
            "只有达到 ev5（应用主预览接入）的效果才标为可用；原厂对拍与真实照片验收未验证",
        )
    } else {
        Status::Missing(
            "No effect in this group has ev5 (main preview) evidence; parameters are parsed and stored only",
            "本组没有效果达到 ev5（应用主预览接入）；参数仅解析并保存",
        )
    }
}

/// 能力表里有登记、但没有参数控件可显示的效果，列在所属段面板底部。它们不能被省略：
/// 例如 PF30096（sediao）、PF90021（lipMakeupEnhance）与 PF44297（祛青筋）。
fn registry_rows(band: &'static Band, unlisted: Vec<&'static evidence::Effect>, cx: &App) -> Vec<AnyElement> {
    let p = pal(cx).clone();
    unlisted
        .into_iter()
        .map(|effect| {
            let label = effect.display_label();
            let detail = effect.state_line();
            let tip = effect.tooltip();
            h_flex()
                .id(SharedString::from(format!("reg-{}-{}", band.evidence_id, effect.pf)))
                .px_3()
                .py_0p5()
                .gap_2()
                .items_start()
                .text_size(px(10.0))
                .text_color(if effect.preview_connected() { p.success } else { p.text_muted })
                .child(div().flex_1().min_w_0().child(label))
                .child(div().flex_shrink_0().child(format!("PF{}", effect.pf)))
                .child(div().flex_shrink_0().child(detail))
                .tooltip(move |window, cx| gpui_kit::component::tooltip::Tooltip::new(tip.clone()).build(window, cx))
                .into_any_element()
        })
        .collect()
}
