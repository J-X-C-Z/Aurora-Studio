//! Portrait tab: Capture One's Retouch tools (face & skin, hair, teeth, eyes) and
//! the portrait algorithm catalogue from `docs/frontend-contract/`.
//!
//! Values are stored in `AdjustmentSet::extra`, so they follow undo, copy/paste,
//! styles and the sidecar.
//!
//! Whether a row may be touched is not decided here. The engine answers
//! `portrait_capability_snapshot(image)` in the background and
//! [`crate::state::portrait_capability`] caches it per image and per registry
//! revision; this file only reads that cache. **Only `Available` is
//! interactive** — every other state (missing model / asset / same-frame input,
//! no backend, stored only, descriptor absent, snapshot not ready yet) is a
//! disabled row that says exactly what is missing. Registration, compilation,
//! `RuntimeProbeContext::fully_ready()` and `has_live_portrait_effect()` are
//! never treated as availability.
//!
//! The tab opens on the controls the renderer uses ("Live"): one panel per
//! capability category (skin, face reshape, plumpness, expression, teeth, eyes,
//! makeup + total makeup, hair, body/hands/feet), listing every descriptor the
//! registry knows. The parameter catalogue ("All") lists all 147 registered main
//! effects grouped into the main categories plus the long tail, keeps the
//! contract parameters that have no registry row, and counts the two teeth
//! sub-controls (PF109/PF110) once under PF108 instead of as extra main items.

#[path = "portrait/makeup_picker.rs"]
mod makeup_picker;

use gpui_kit::assets::IconName;
use gpui_kit::component::{
    Icon, Sizable as _,
    h_flex, v_flex,
};
use gpui_kit::{
    AnyElement, App, Entity, InteractiveElement, IntoElement, ParentElement, SharedString, StatefulInteractiveElement, Styled, Window,
    div, prelude::FluentBuilder as _, px,
};

use aurora_engine::portrait_registry::{descriptor_by_pf, EffectCategory};

use crate::contract::{self, BANDS, Band, PortraitParam, Status};
use crate::evidence;
use crate::i18n::t;
use crate::state::portrait_capability::{self, CapabilityStatus, SliderSpec};
use crate::state::{AppState, Binding, extra_binding};
use crate::theme::pal;
use crate::widgets::*;

/// 该 PF 在当前图片上是否可交互：只有新鲜快照里的 `Available` 才算。
///
/// 集成点：子妆容组件（`portrait/makeup_picker.rs`）对子妆容行、总妆面板都用它
/// 决定款式/色板是否出现，未就绪一律 `false`。
pub(super) fn row_available(s: &AppState, pf: u32) -> bool {
    portrait_capability::row_available(s, pf)
}

/// Capture One's legacy retouch groups: engine group, panel id, title.
const RETOUCH_GROUPS: [(&str, &str, &str, &str); 4] = [
    ("RetouchFaceSkin", "retouch-face", "Face & Skin", "面部与皮肤"),
    ("RetouchEyes", "retouch-eyes", "Eyes", "眼睛"),
    ("RetouchHair", "retouch-hair", "Hair", "头发"),
    ("RetouchTeeth", "retouch-teeth", "Teeth", "牙齿"),
];

/// How the live skin adjustment sliders are grouped in their panel.
const SKIN_SECTIONS: [(&str, &str, &[&str]); 4] = [
    ("Skin tone", "肤色", &["pc:21015", "pc:30096", "pc:90005", "pc:90007", "pc:90004", "pc:21105", "pc:90086"]),
    ("Texture", "质感", &["pc:8068", "pc:8069", "pc:8070"]),
    ("Smoothing & Neutral Grey", "磨皮与中性灰", &["pc:30021", "pc:30022", "pc:90002", "pc:90010", "pc:30013", "pc:30014"]),
    ("Wrinkles", "祛皱", &["pc:90023", "pc:90024", "pc:21024", "pc:21025", "pc:21069", "pc:21070", "pc:21071"]),
];

/// How the live makeup sliders are grouped in their panel.
const MAKEUP_SECTIONS: [(&str, &str, &[&str]); 5] = [
    ("Stereo contour", "立体修容", &["pc:8", "pc:9"]),
    ("Brows & Eyes", "眉眼妆", &["pc:100", "pc:117", "pc:30007", "pc:103", "pc:105", "pc:118", "pc:111"]),
    ("Blusher & Contour", "腮红与修容", &["pc:106", "pc:116"]),
    ("Lips", "唇妆", &["pc:107", "pc:90032", "pc:90021"]),
    ("Sticker", "贴纸", &["pc:115"]),
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
];

/// Expression management effects wired to same-job smile warp (`assets/portrait/smile/`).
const WIRED_EXPRESSION_EFFECTS: &[(&str, &str, &str, f32)] = &[
    ("pc:8033", "Smile mouth corner", "微笑嘴角", 0.0),
];

/// Makeup effects wired to `B3_MAKEUP_ROWS` (`attach_makeup_bundle`) and `PF90021`.
const WIRED_MAKEUP_EFFECTS: &[(&str, &str, &str, f32)] = &[
    ("pc:8", "Stereo shadow", "立体阴影", 0.0),
    ("pc:9", "Stereo highlight", "立体高光", 0.0),
    ("pc:100", "Eyebrow (L)", "眉妆（左）", 0.0),
    ("pc:117", "Eyebrow (R)", "眉妆（右）", 0.0),
    ("pc:30007", "Eyeshadow", "眼影", 0.0),
    ("pc:103", "Upper eyeliner", "上眼线", 0.0),
    ("pc:105", "Upper eyelash", "上睫毛", 0.0),
    ("pc:118", "Lower eyelash", "下睫毛", 0.0),
    ("pc:111", "Contact lenses", "美瞳", 0.0),
    ("pc:106", "Blusher", "腮红", 0.0),
    ("pc:116", "Contour", "修容", 0.0),
    ("pc:107", "Lipstick", "口红", 0.0),
    ("pc:90032", "Lip gloss", "唇彩", 0.0),
    ("pc:90021", "Lip makeup enhance", "唇妆增强", 0.5),
    ("pc:115", "Face decoration", "面部贴纸", 0.0),
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
    ("pc:49221", "Brow tail length (L)", "眉尾长短（左）"),
    ("pc:41922", "Brow tail length (R)", "眉尾长短（右）"),
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

/// 牙齿子控：同一效果的亮白/去黄，不单独计入 147 个主项。
const TEETH_SUB_CONTROLS: [(u32, &[u32]); 1] = [(108, &[109, 110])];

/// 长尾分类：已登记、但基本没有渲染接线的分组。
const LONG_TAIL_GROUPS: [&str; 5] = ["pc-hands", "pc-feet", "pc-clothes", "pc-scene", "pc-other"];

fn is_sub_control(pf: u32) -> bool {
    TEETH_SUB_CONTROLS.iter().any(|(_, subs)| subs.contains(&pf))
}

fn sub_controls_of(pf: u32) -> &'static [u32] {
    TEETH_SUB_CONTROLS
        .iter()
        .find(|(main, _)| *main == pf)
        .map(|(_, subs)| *subs)
        .unwrap_or(&[])
}

/// 一个分类下的主项（不含挂在上面的子控）。
fn main_effects(category: EffectCategory) -> Vec<u32> {
    portrait_capability::pfs_in(category).into_iter().filter(|pf| !is_sub_control(*pf)).collect()
}

/// 「全部」视图里一个 PF 归属的分组 id（与 `contract::BANDS` 一致）。
fn group_of(pf: u32) -> &'static str {
    if is_sub_control(pf) {
        return "pc-teeth";
    }
    let band = contract::band_index_for_id(i64::from(pf));
    BANDS[band].id
}

/// 某分组里的 147 个主项之一；子控单独走 [`sub_items_in`]。
///
/// 事实源是能力描述符表（`registry`），不是证据表：证据表是 332 条历史登记，
/// 既可能缺新加的描述符，也会把只登记过、没有描述符的 id 混进来。这里按 registry
/// 的 PF 集筛选，分组只借 `group_of` 的归属。
fn main_items_in(group: &str) -> Vec<u32> {
    let mut items: Vec<u32> = aurora_engine::portrait_registry::WIRED_EFFECT_DESCRIPTORS
        .iter()
        .map(|descriptor| descriptor.id.pf)
        .filter(|pf| !is_sub_control(*pf) && group_of(*pf) == group)
        .collect();
    items.sort_unstable();
    items
}

/// 挂在主项下面的子控（目前只有牙齿 PF109/PF110）。
fn sub_items_in(group: &str) -> Vec<u32> {
    let mut items: Vec<u32> = TEETH_SUB_CONTROLS
        .iter()
        .filter(|_| group == "pc-teeth")
        .flat_map(|(_, subs)| subs.iter().copied())
        .collect();
    items.sort_unstable();
    items
}

fn is_long_tail(group: &str) -> bool {
    LONG_TAIL_GROUPS.contains(&group)
}

fn is_registered_pf(pf: i64) -> bool {
    evidence::for_pf(pf).is_some()
}

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
            .chain(WIRED_EXPRESSION_EFFECTS)
            .chain(WIRED_TEETH_EFFECTS)
            .chain(WIRED_EYES_EFFECTS)
            .chain(WIRED_MAKEUP_EFFECTS)
            .chain(WIRED_HAIR_EFFECTS)
            .any(|(key, _, _, neutral)| {
                adjustments.extra.get(*key).is_some_and(|amount| (amount - neutral).abs() > 1.0e-6)
            })
}

#[cfg(test)]
#[test]
fn every_live_skin_and_teeth_effect_has_one_section() {
    let listed_skin: Vec<&str> = SKIN_SECTIONS.iter().flat_map(|(_, _, keys)| keys.iter().copied()).collect();
    let live_skin: Vec<&str> = WIRED_SKIN_EFFECTS.iter().map(|e| e.0).collect();
    assert_eq!(listed_skin.len(), live_skin.len());
    assert!(live_skin.iter().all(|k| listed_skin.iter().filter(|l| *l == k).count() == 1));

    let listed_makeup: Vec<&str> = MAKEUP_SECTIONS.iter().flat_map(|(_, _, keys)| keys.iter().copied()).collect();
    let live_makeup: Vec<&str> = WIRED_MAKEUP_EFFECTS.iter().map(|e| e.0).collect();
    assert_eq!(listed_makeup.len(), live_makeup.len());
    assert!(live_makeup.iter().all(|k| listed_makeup.iter().filter(|l| *l == k).count() == 1));
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
        || WIRED_EXPRESSION_EFFECTS.iter().any(|row| row.0 == key)
        || WIRED_TEETH_EFFECTS.iter().any(|row| row.0 == key)
        || WIRED_EYES_EFFECTS.iter().any(|row| row.0 == key)
        || WIRED_MAKEUP_EFFECTS.iter().any(|row| row.0 == key)
        || WIRED_HAIR_EFFECTS.iter().any(|row| row.0 == key)
        || WIRED_FACE_GEOMETRY_EFFECTS.iter().any(|row| row.0 == key)
        || WIRED_FACE_SINGLE_EFFECTS.iter().any(|row| row.0 == key)
        || WIRED_PLUMP_EFFECTS.iter().any(|row| row.0 == key)
}

// ─────────────────────── 行原语：能力门控 ───────────────────────

/// 已登记效果的本地化标签：描述符优先，其次能力表，最后 PF 号。
fn effect_label(pf: u32) -> SharedString {
    if let Some(descriptor) = descriptor_by_pf(pf) {
        return t(descriptor.label_en, descriptor.label_zh).into();
    }
    if let Some(effect) = evidence::for_pf(i64::from(pf)) {
        return effect.display_label().into();
    }
    format!("PF{pf}").into()
}

/// 分节的本地化标题；`Eye makeup` 只属于 PF90032。
fn section_zh(section: &'static str) -> &'static str {
    match section {
        "Skin tone" => "肤色",
        "Blemish" => "祛瑕疵",
        "Texture" => "质感",
        "Smoothing & Neutral Grey" => "磨皮与中性灰",
        "Wrinkles" => "祛皱",
        "Face pairs" => "左右配对",
        "Face single" => "单项",
        "Facial Plumpness" => "面部丰盈",
        "Expression" => "表情管理",
        "Teeth" => "牙齿",
        "Eyes" => "眼睛",
        "Eye makeup" => "眼妆",
        "Stereo contour" => "立体修容",
        "Brows & Eyes" => "眉眼妆",
        "Blusher & Contour" => "腮红与修容",
        "Lips" => "唇妆",
        "Sticker" => "贴纸",
        "Suit makeup" => "成套妆容",
        "Hair" => "头发",
        "Body" => "全身美型",
        "Body detail" => "身体细节",
        "Hands" => "手部",
        "Feet" => "脚部",
        other => other,
    }
}

/// 界面值 ↔ `extra` 存储值的绑定；单位、默认值和离散步进都来自 [`SliderSpec`]，
/// 脸型百分比/归一化的判定只在 `state::portrait_capability` 里做一次。
fn spec_binding(pf: u32, spec: SliderSpec) -> Binding {
    let store_key = descriptor_by_pf(pf)
        .map(|descriptor| descriptor.id.store_key.to_owned())
        .unwrap_or_else(|| format!("pc:{pf}"));
    let inner = extra_binding(store_key, spec.stored_default);
    let (get, set) = (inner.get.clone(), inner.set.clone());
    Binding::new(
        move |a| portrait_capability::ui_value(pf, &spec, get(a)),
        move |a, v| set(a, spec.to_stored(v)),
    )
}

fn effect_store_key(pf: u32) -> String {
    descriptor_by_pf(pf)
        .map(|descriptor| descriptor.id.store_key.to_owned())
        .unwrap_or_else(|| format!("pc:{pf}"))
}

/// 该效果在本图是否偏离中性默认（面板头上的小圆点）。
fn effect_is_modified(state: &AppState, pf: u32) -> bool {
    let Some(spec) = portrait_capability::slider_spec(pf) else {
        return false;
    };
    state
        .adj()
        .extra
        .get(&effect_store_key(pf))
        .is_some_and(|value| (value - spec.stored_default).abs() > 1.0e-6)
}

fn matches_effect(query: &str, label: &str, pf: u32) -> bool {
    query.is_empty()
        || label.to_lowercase().contains(query)
        || format!("pc:{pf}").contains(query)
        || query.strip_prefix("pf").is_some_and(|number| number == pf.to_string())
        || pf.to_string() == query
}

/// 不可交互时的原因行：写确切原因（缺模型/资产/同帧输入/后端/仅保存/未就绪），
/// 不用「部分可用」一类含混说法。可用的行不画这一行。
fn reason_line(text: String, cx: &App) -> AnyElement {
    let p = pal(cx).clone();
    h_flex()
        .px_3()
        .pb_1()
        .gap_1()
        .items_start()
        .text_size(px(10.0))
        .text_color(p.text_muted)
        .child(div().size(px(4.0)).rounded_full().mt(px(4.0)).bg(p.text_disabled))
        .child(div().flex_1().min_w_0().child(text))
        .into_any_element()
}

fn row_tooltip(pf: u32, status: CapabilityStatus<'_>) -> String {
    let mut out = format!("PF{pf}");
    if let Some(descriptor) = descriptor_by_pf(pf) {
        out.push_str(&format!(" · {}", descriptor.section));
        if !descriptor.dependencies.is_empty() {
            out.push_str(&format!("\n{} {}", t("Inputs", "输入"), portrait_capability::inputs_line(descriptor.dependencies)));
        }
        out.push_str(&format!("\n{} {}", t("Stored unit", "存储单位"), descriptor.range.unit.as_str()));
    }
    out.push('\n');
    out.push_str(&status.line());
    if let Some(effect) = evidence::for_pf(i64::from(pf)) {
        out.push('\n');
        out.push_str(&effect.tooltip());
    }
    out
}

fn is_makeup_row(pf: u32) -> bool {
    // PF90032 是眼妆；PF101（卧蚕）分类在眼部，但它同样有真实的材质款式网格
    // （`makeup_presets_for_pf(101)`，`attach_pf101_inputs_with_style` 消费序数），
    // 所以也挂款式条，不然 24 套材质在界面上无处可选。
    pf == 90032 || pf == 101 || portrait_capability::category_of(pf) == Some(EffectCategory::Makeup)
}

/// 一行已登记效果：`Available` 才可交互；其它状态禁用并写原因，绝不发 edit。
fn effect_row(state: &Entity<AppState>, pf: u32, cx: &App) -> AnyElement {
    let (available, reason, tip, spec) = {
        let s = state.read(cx);
        let status = s.capability_status(pf);
        (
            status.is_available(),
            status.line(),
            row_tooltip(pf, status),
            portrait_capability::slider_spec(pf),
        )
    };
    let p = pal(cx).clone();
    let mut rows: Vec<AnyElement> = Vec::new();
    match spec {
        Some(spec) => {
            let mut slider = param(state, format!("pc-{pf}"), effect_label(pf), spec_binding(pf, spec), spec.min, spec.max, cx)
                .default_value(spec.default)
                .unit(spec.unit)
                .decimals(spec.decimals)
                .disabled(!available);
            if let Some(step) = spec.step {
                slider = slider.step(step);
            }
            rows.push(slider.into_any_element());
        }
        None => {
            // 没有尺寸/单位可靠的控件：只列标签与状态，不猜一个滑杆出来。
            rows.push(
                h_flex()
                    .id(SharedString::from(format!("prow-{pf}")))
                    .px_3()
                    .py_1()
                    .gap_2()
                    .items_center()
                    .child(div().flex_1().min_w_0().text_size(px(11.0)).text_color(p.text_muted).child(effect_label(pf)))
                    .child(div().flex_shrink_0().text_size(px(10.0)).text_color(p.text_disabled).child(format!("PF{pf}")))
                    .into_any_element(),
            );
        }
    }
    if !available {
        rows.push(reason_line(reason, cx));
    }
    // 子妆容行：款式与色板。旧的小色条已经搬进这个组件，这里不再重复画。
    if is_makeup_row(pf) && state.read(cx).capability_status(pf).can_probe_selection() {
        if let Some(key) = descriptor_by_pf(pf).map(|descriptor| descriptor.id.store_key) {
            if let Some(bar) = makeup_picker::style_bar(state, key, cx) {
                rows.push(bar);
            }
        }
    }
    v_flex()
        .id(SharedString::from(format!("prow-wrap-{pf}")))
        .w_full()
        .tooltip(move |window, cx| gpui_kit::component::tooltip::Tooltip::new(tip.clone()).build(window, cx))
        .children(rows)
        .into_any_element()
}

/// 挂在主项下的子控（牙齿 PF109/PF110）：缩进、不单独计入 147 个主项。
fn sub_control_row(state: &Entity<AppState>, pf: u32, cx: &App) -> AnyElement {
    let p = pal(cx).clone();
    v_flex()
        .w_full()
        .pl_3()
        .border_l_1()
        .border_color(p.divider)
        .child(effect_row(state, pf, cx))
        .into_any_element()
}

pub fn render(state: &Entity<AppState>, window: &mut Window, cx: &mut App) -> AnyElement {
    // 只有打开人像页才要能力快照。请求放到这一帧结束后排队执行：绘制路径
    // 绝不同步调用引擎，也不在绘制中途改自己的状态；缓存缺失时行一律禁用。
    if state.read(cx).portrait_capability_needed() {
        let st = state.clone();
        window.defer(cx, move |_, cx| {
            st.update(cx, |s, cx| {
                s.request_portrait_capability(portrait_capability::CapabilityDetail::Auto, cx)
            });
        });
    }
    let s = state.read(cx);
    let raw_query = s.prefs.panels.portrait_query.clone();
    let query = raw_query.trim().to_lowercase();
    let show_all = s.prefs.panels.portrait_open;
    let p = pal(cx).clone();

    let registered_total = evidence::all().count();
    let retouch: Vec<&'static str> =
        s.engine.param_metadata().iter().filter(|m| RETOUCH_GROUPS.iter().any(|(g, ..)| m.group == *g)).map(|m| m.key).collect();
    let stored_total = retouch.len() + BANDS.iter().map(|b| b.params().filter(|p| !is_registered_pf(p.id) && !is_wired_key(&p.store_key())).count()).sum::<usize>();
    let stored_set = s.adj().extra.keys().filter(|k| !is_wired_key(k) && (k.starts_with("pc:") || retouch.contains(&k.as_str()))).count();
    // 真实分母是 147 个主项：引擎描述符里去掉牙齿子控 PF109/PF110。
    // 149（含两个子控）和 332（证据表历史登记）都不是这个面板的分母。
    let caps = s
        .portrait_caps
        .snapshot(s.library.current, s.capability_key())
        .map(|snapshot| {
            let main = || snapshot.effects.iter().filter(|entry| !is_sub_control(entry.pf));
            (
                main().filter(|entry| entry.availability.is_available()).count(),
                main().count(),
            )
        });
    let caps_pending = caps.is_none();
    let caps_label = match caps {
        Some((available, total)) => format!("{available}/{total}"),
        None => format!("—/{registered_total}"),
    };
    let main_label = registered_total.to_string();

    let st = state.clone();
    let search = text_field("portrait-search", raw_query, t("Search all parameters…", "搜索全部参数…"), move |q, _, cx| {
        st.update(cx, |s, cx| {
            s.prefs.panels.portrait_query = q;
            cx.notify();
        })
    }, window, cx);
    let about = format!(
        "{}\n\n{}\n\n{} {}… · {} {}\n{}",
        t(
            "Whether a control may be touched is decided by the engine's portrait capability snapshot for this photo, fetched in the background. Only Available is interactive; missing model / asset / same-frame input, no backend, stored only and 'not ready yet' are named on the row and stay disabled. Evidence level, registration and compilation never make an effect available.",
            "控件能否操作由引擎为当前照片返回的人像能力快照决定，后台异步获取。只有 Available 可交互；缺模型/资产/同帧输入、无可用后端、仅保存、尚未就绪都会写在行上并保持禁用。证据等级、登记与编译都不能让效果变成可用。",
        ),
        t(
            "Each effect also shows its seven-level evidence state. Only ev5 (main preview) is evidence of wiring; stored parameters alone do not change the preview.",
            "每个效果另外显示七级证据状态；只有 ev5（应用主预览接入）才是接线证据。仅保存参数不会改变预览。",
        ),
        t("Evidence registry", "证据表"),
        &evidence::REGISTRY_SHA256[..12],
        registered_total,
        t("registered effects", "个已登记效果"),
        evidence::REGISTRY_SOURCE,
    );
    let note: SharedString = if !query.is_empty() {
        t("Searching every parameter, stored-only ones included", "正在搜索全部参数，包括仅保存的参数").into()
    } else if show_all {
        t("Unavailable rows say what is missing and stay disabled", "不可用的行会写明缺什么，并保持禁用").into()
    } else {
        format!("{stored_total} {}", t("stored-only parameters hidden", "个仅保存的参数已隐藏")).into()
    };
    // 手动刷新：模型/资产刚装上、换了计算后端时用。缓存按图 + 资料版本，
    // 这里只清当前图。
    let st_refresh = state.clone();
    // 分段控件自己的句柄：搜索框的闭包已经把 `st` 移走了。
    let st_scope = state.clone();
    let refresh = div()
        .id("portrait-refresh-caps")
        .flex_shrink_0()
        .text_color(p.text_subtle)
        .hover(|s| s.text_color(p.text))
        .child(Icon::new(IconName::RotateCcw).xsmall())
        .tooltip(move |window, cx| {
            gpui_kit::component::tooltip::Tooltip::new(
                t(
                    "Re-read this photo's portrait capability from the engine",
                    "重新向引擎读取当前照片的人像能力",
                )
                .to_string(),
            )
            .build(window, cx)
        })
        .on_click(move |_, _, cx| st_refresh.update(cx, |s, cx| s.refresh_portrait_capability(cx)));
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
                .child(refresh)
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
                .text(format!("{} {caps_label}", t("Live", "实时")))
                .text(format!("{} {main_label}", t("All", "全部")))
                .on_select(move |i, _, cx| {
                    st_scope.update(cx, |s, cx| {
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
                .child(div().flex_shrink_0().when(caps_pending, |d| d.text_color(p.warning)).child(match caps {
                    Some((available, total)) => format!("{} {available}/{total}", t("available now", "当前可用")),
                    None => t("capability snapshot not ready", "能力快照未就绪").to_string(),
                }))
                // Values hidden behind "Live" still apply; say so.
                .when(!show_all && query.is_empty() && stored_set > 0, |d| {
                    d.child(div().flex_shrink_0().text_color(p.warning).child(format!("{stored_set} {}", t("set on this image", "个已在本图设置"))))
                }),
        );

    if !show_all && query.is_empty() {
        return PanelStack::new(state)
        .head(header)
        .panel("portrait-skin", category_panel(state, "portrait-skin", EffectCategory::Skin, &query, cx))
        .panel("retouch-face-geometry", face_geometry_panel(state, &query, cx))
        .panel("retouch-face-plump", category_panel(state, "retouch-face-plump", EffectCategory::FacePlump, &query, cx))
        .panel("portrait-expression", category_panel(state, "portrait-expression", EffectCategory::Expression, &query, cx))
        .panel("portrait-teeth", category_panel(state, "portrait-teeth", EffectCategory::Teeth, &query, cx))
        .panel("portrait-eyes", category_panel(state, "portrait-eyes", EffectCategory::Eyes, &query, cx))
        .panel("portrait-makeup", makeup_panel(state, &query, cx))
        .panel("portrait-makeup-suit", makeup_suit_panel(state, cx))
        .panel("portrait-hair", category_panel(state, "portrait-hair", EffectCategory::Hair, &query, cx))
        .panel("portrait-body", category_panel(state, "portrait-body", EffectCategory::Body, &query, cx)).render(cx);
    }
    let mut stored = PanelStack::new(state).head(header);
    if query.is_empty() { stored = stored.panel("portrait-makeup-suit", makeup_suit_panel(state, cx)); }
    for (group, id, en, zh) in RETOUCH_GROUPS {
        stored = stored.panel(id, retouch_panel(state, group, id, t(en, zh), &query, cx));
    }
    // 「全部」：14 个主项/长尾分组，合起来正好是 147 个已登记主项；
    // PF109/PF110 挂在 PF108 下面，不重复计入。
    for band in BANDS.iter() {
        stored = stored.panel(band.id, group_panel(state, band, &query, cx));
    }
    stored.render(cx)
}

fn matches(query: &str, label: &str, key: &str, id: Option<i64>) -> bool {
    query.is_empty()
        || label.to_lowercase().contains(query)
        || key.to_lowercase().contains(query)
        || id.is_some_and(|id| id.to_string() == query)
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

/// 「实时」面板：一个分类下的全部已登记效果，按描述符分节。只有能力快照里
/// `Available` 的行可交互；其它状态在行上写明缺什么并保持禁用。
fn category_panel(
    state: &Entity<AppState>,
    id: &'static str,
    category: EffectCategory,
    query: &str,
    cx: &App,
) -> AnyElement {
    let pfs = main_effects(category);
    // 先取数据（读锁短），再建元素。
    let (items, modified) = {
        let s = state.read(cx);
        let mut items: Vec<(Option<&'static str>, u32, &'static [u32])> = Vec::new();
        let mut modified = false;
        let mut section: Option<&'static str> = None;
        for pf in pfs.iter().copied() {
            if !matches_effect(query, &effect_label(pf), pf) && !sub_controls_of(pf).iter().any(|&sub| matches_effect(query, &effect_label(sub), sub)) {
                continue;
            }
            let current = portrait_capability::section_of(pf);
            let starts_section = current != section;
            section = current;
            let subs = sub_controls_of(pf);
            modified |= effect_is_modified(&*s, pf);
            modified |= subs.iter().any(|sub| effect_is_modified(&*s, *sub));
            items.push((if starts_section { current } else { None }, pf, subs));
        }
        (items, modified)
    };
    if items.is_empty() {
        return div().into_any_element();
    }
    let link_brows = category == EffectCategory::Makeup && state.read(cx).prefs.panels.portrait_link_lr
        && [100, 117].iter().all(|pf| items.iter().any(|(_, item, _)| item == pf));
    let mut rows: Vec<AnyElement> = Vec::new();
    for (section, pf, subs) in items {
        if let Some(section) = section {
            rows.push(caption(t(section, section_zh(section)), cx));
        }
        if link_brows && pf == 117 { continue; }
        if link_brows && pf == 100 { rows.push(brow_pair(state, cx)); }
        else { rows.push(effect_row(state, pf, cx)); }
        for sub in subs {
            rows.push(sub_control_row(state, *sub, cx));
        }
    }
    let keys: Vec<String> = pfs.iter().flat_map(|&pf| std::iter::once(pf).chain(sub_controls_of(pf).iter().copied())).map(effect_store_key).collect();
    let title = t(category.display_name_en(), category.display_name_zh());
    let st = state.clone();
    ToolPanel::new(state, id, title)
        .status(Status::Partial(
            "Rows stay disabled until the engine reports this photo's models, assets and same-frame inputs as available",
            "引擎为该照片报告模型、资产与同帧输入就绪前，各行保持禁用",
        ))
        .badge(pfs.len().to_string())
        .modified(modified)
        .on_reset(move |_, cx| {
            let keys = keys.clone();
            st.update(cx, |s, cx| {
                s.edit(format!("{} {title}", t("Reset", "复位")), |a| {
                    a.extra.retain(|k, _| !keys.iter().any(|row| k == row || k.strip_prefix(row).is_some_and(|suffix| suffix.starts_with(':'))));
                }, cx)
            })
        })
        .children(rows)
        .into_any_element()
}

/// 一对左右分量的标签基数：`"V-face (L)"` → `("V-face", false)`。
fn pair_base_en(label: &'static str) -> Option<(&'static str, bool)> {
    if let Some(base) = label.strip_suffix(" (L)") {
        Some((base, false))
    } else {
        label.strip_suffix(" (R)").map(|base| (base, true))
    }
}

/// 只有眉妆是真正的左右对；上下睫毛不是。
#[cfg(test)]
#[test]
fn only_the_brow_rows_are_a_left_right_pair() {
    let label = |key: &str| {
        WIRED_MAKEUP_EFFECTS
            .iter()
            .find(|effect| effect.0 == key)
            .map(|effect| effect.1)
    };
    // 眉妆：左/右一对，联动时一条滑杆写两个键。
    assert_eq!(label("pc:100"), Some("Eyebrow (L)"));
    assert_eq!(label("pc:117"), Some("Eyebrow (R)"));
    assert!(pair_base_en("Eyebrow (L)").is_some());
    assert!(pair_base_en("Eyebrow (R)").is_some());
    // 上/下睫毛：同一条效果的两个分量，但不是左右对，永远各自独立。
    assert_eq!(label("pc:105"), Some("Upper eyelash"));
    assert_eq!(label("pc:118"), Some("Lower eyelash"));
    assert!(pair_base_en("Upper eyelash").is_none());
    assert!(pair_base_en("Lower eyelash").is_none());
    assert!(pair_base_zh("上睫毛").is_none());
    assert!(pair_base_zh("下睫毛").is_none());
}

/// PF101（卧蚕）分类在眼部，但同样有真实款式网格。
#[cfg(test)]
#[test]
fn pf101_renders_a_style_grid_like_a_makeup_row() {
    assert!(is_makeup_row(101), "PF101 has material presets to pick");
    assert!(is_makeup_row(90032), "the eye image makeup row");
    assert!(!is_makeup_row(108), "teeth are not a makeup row");
}

fn pair_base_zh(label: &'static str) -> Option<(&'static str, bool)> {
    if let Some(base) = label.strip_suffix("（左）") {
        Some((base, false))
    } else {
        label.strip_suffix("（右）").map(|base| (base, true))
    }
}

/// 联动绑定：读两侧平均，写两侧同一个值（存储单位由 `spec` 决定）。
fn linked_binding(left: u32, right: u32, spec: SliderSpec) -> Binding {
    let l = spec_binding(left, spec);
    let r = spec_binding(right, spec);
    let (lget, lset) = (l.get.clone(), l.set.clone());
    let (rget, rset) = (r.get.clone(), r.set.clone());
    Binding::new(
        move |a| (lget(a) + rget(a)) * 0.5,
        move |a, v| {
            lset(a, v);
            rset(a, v);
        },
    )
}

/// 联动的一对：一条滑杆写两侧；两侧都 `Available` 才可交互。
fn linked_row(state: &Entity<AppState>, label: SharedString, left: u32, right: u32, cx: &App) -> AnyElement {
    let (available, reason, spec) = {
        let s = state.read(cx);
        let left_status = s.capability_status(left);
        let right_status = s.capability_status(right);
        // 两侧都可用才能联动写：用的是对外公开的 `row_available`，与子妆容组件同一判据。
        let available = row_available(&*s, left) && row_available(&*s, right);
        let reason = if left_status.is_available() {
            right_status.line()
        } else {
            left_status.line()
        };
        (available, reason, portrait_capability::slider_spec(left))
    };
    let Some(spec) = spec else {
        return effect_row(state, left, cx);
    };
    let slider = param(
        state,
        format!("fg-{left}-lr"),
        label,
        linked_binding(left, right, spec),
        spec.min,
        spec.max,
        cx,
    )
    .default_value(spec.default)
    .unit(spec.unit)
    .decimals(spec.decimals)
    .disabled(!available);
    v_flex()
        .id(SharedString::from(format!("fg-pair-{left}")))
        .w_full()
        .child(slider)
        .when(!available, |this| this.child(reason_line(reason, cx)))
        .into_any_element()
}

/// 面部精修：左右配对 + 单项。联动（默认）时一条滑杆写两侧，分开时各一条。
fn face_geometry_panel(state: &Entity<AppState>, query: &str, cx: &App) -> AnyElement {
    let linked = state.read(cx).prefs.panels.portrait_link_lr;
    let pfs = main_effects(EffectCategory::FaceGeometry);
    let descriptors: Vec<_> = pfs.iter().filter_map(|pf| descriptor_by_pf(*pf)).collect();
    let mut pairs: Vec<(&'static str, &'static str, u32, u32)> = Vec::new();
    let mut singles: Vec<u32> = Vec::new();
    let mut index = 0;
    while index < descriptors.len() {
        let current = descriptors[index];
        if let (Some((base_en, false)), Some(next)) = (pair_base_en(current.label_en), descriptors.get(index + 1)) {
            if matches!(pair_base_en(next.label_en), Some((next_base, true)) if next_base == base_en) {
                let base_zh = pair_base_zh(current.label_zh).map(|(base, _)| base).unwrap_or(current.label_zh);
                pairs.push((base_en, base_zh, current.id.pf, next.id.pf));
                index += 2;
                continue;
            }
        }
        singles.push(current.id.pf);
        index += 1;
    }
    let shown_pairs: Vec<_> = pairs
        .iter()
        .copied()
        .filter(|(en, zh, left, right)| {
            matches_effect(query, t(*en, *zh), *left)
                || matches_effect(query, t(*en, *zh), *right)
                || matches_effect(query, &effect_label(*left), *left)
                || matches_effect(query, &effect_label(*right), *right)
        })
        .collect();
    let shown_singles: Vec<u32> = singles
        .iter()
        .copied()
        .filter(|pf| matches_effect(query, &effect_label(*pf), *pf))
        .collect();
    if shown_pairs.is_empty() && shown_singles.is_empty() {
        return div().into_any_element();
    }
    let modified = {
        let s = state.read(cx);
        pfs.iter().any(|pf| effect_is_modified(&*s, *pf))
    };
    let mut rows: Vec<AnyElement> = Vec::new();
    for (en, zh, left, right) in shown_pairs {
        if linked {
            rows.push(linked_row(state, t(en, zh).into(), left, right, cx));
        } else {
            rows.push(caption(t(en, zh), cx));
            rows.push(effect_row(state, left, cx));
            rows.push(effect_row(state, right, cx));
        }
    }
    for pf in shown_singles {
        rows.push(effect_row(state, pf, cx));
    }
    let keys: Vec<String> = pfs.iter().map(|pf| effect_store_key(*pf)).collect();
    let (st, st_link) = (state.clone(), state.clone());
    ToolPanel::new(state, "retouch-face-geometry", t("Face Reshape", "面部精修"))
        .status(Status::Partial(
            "Face reshape needs this photo's landmarks; linked pairs move both sides, and both must be available",
            "面部精修需要当前照片的面部关键点；联动时同时写两侧，两侧都可用才可操作",
        ))
        .badge(pfs.len().to_string())
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
                    for key in &keys {
                        a.extra.remove(key);
                    }
                }, cx)
            })
        })
        .children(rows)
        .into_any_element()
}

/// 妆容：子妆容行（款式/色板由 `makeup_picker::style_bar` 画）。
fn brow_pair(state: &Entity<AppState>, cx: &App) -> AnyElement {
    let mut rows = vec![linked_row(state, t("Eyebrow", "眉妆").into(), 100, 117, cx)];
    for (pf, key) in [(100, "pc:100"), (117, "pc:117")] {
        if state.read(cx).capability_status(pf).can_probe_selection() {
            if let Some(bar) = makeup_picker::style_bar(state, key, cx) { rows.push(bar); }
        }
    }
    v_flex().w_full().children(rows).into_any_element()
}

fn makeup_panel(state: &Entity<AppState>, query: &str, cx: &App) -> AnyElement {
    category_panel(state, "portrait-makeup", EffectCategory::Makeup, query, cx)
}

/// 总妆：成套妆容面板由 `makeup_picker::suit_picker` 提供。
fn makeup_suit_panel(state: &Entity<AppState>, cx: &App) -> AnyElement {
    ToolPanel::new(state, "portrait-makeup-suit", t("Total makeup", "总妆"))
        .status(Status::Partial(
            "A suit needs the makeup asset root and this photo's masks; a row stays disabled until the engine reports it available",
            "成套妆容需要妆容素材目录与当前照片的遮罩；引擎报告可用前，各行保持禁用",
        ))
        .children(makeup_picker::suit_picker(state, cx))
        .into_any_element()
}

/// 「全部」里一个分组要画的行，先在读锁里算好。
struct GroupRows {
    main: Vec<u32>,
    subs: Vec<u32>,
    catalogue: Vec<&'static PortraitParam>,
    modified: bool,
    set: usize,
}

/// 「全部」里的一个分组，按 147 个已登记主项来列：
///
/// * 主项（含挂在 PF108 下的 PF109/PF110 子控，不重复计数）；
/// * 该段里没有登记进能力表的契约参数，仍然用原来的参数目录控件；
/// * 只按 ID 段推断、没有控件的未解析参数，照旧列出来。
fn group_panel(state: &Entity<AppState>, band: &'static Band, query: &str, cx: &App) -> AnyElement {
    let main = main_items_in(band.id);
    let subs = sub_items_in(band.id);
    let catalogue: Vec<&'static PortraitParam> = band
        .params()
        .filter(|param| !is_registered_pf(param.id) && !is_wired_key(&param.store_key()))
        .collect();
    let rows = {
        let s = state.read(cx);
        let main = main
            .iter()
            .copied()
            .filter(|pf| matches_effect(query, &effect_label(*pf), *pf) || sub_controls_of(*pf).iter().any(|&sub| matches_effect(query, &effect_label(sub), sub)))
            .collect::<Vec<_>>();
        let subs = subs
            .iter()
            .copied()
            .filter(|pf| matches_effect(query, &effect_label(*pf), *pf))
            .collect::<Vec<_>>();
        let catalogue = catalogue
            .iter()
            .copied()
            .filter(|param| matches(query, &param.label(), &param.key, Some(param.id)))
            .collect::<Vec<_>>();
        let keys: Vec<String> = main.iter().chain(subs.iter()).map(|pf| effect_store_key(*pf)).collect();
        let catalogue_keys: Vec<String> = catalogue.iter().map(|param| param.store_key()).collect();
        let set = s
            .adj()
            .extra
            .keys()
            .filter(|key| {
                keys.iter()
                    .chain(catalogue_keys.iter())
                    .any(|stored| *key == stored || key.starts_with(&format!("{stored}:")))
            })
            .count();
        let modified = main.iter().chain(subs.iter()).any(|pf| effect_is_modified(&*s, *pf));
        GroupRows { main: main.to_vec(), subs, catalogue, modified, set }
    };
    if rows.main.is_empty() && rows.subs.is_empty() && rows.catalogue.is_empty() {
        return div().into_any_element();
    }
    // 搜索时展开命中的分组；否则沿用折叠记忆。
    let open = !query.is_empty() || !state.read(cx).prefs.panels.collapsed.contains(band.id);
    let total = main.len();
    let mut keys: Vec<String> = main.iter().chain(subs.iter()).map(|pf| effect_store_key(*pf)).collect();
    keys.extend(rows.catalogue.iter().map(|param| param.store_key()));
    let st = state.clone();
    let measured = main.iter().filter(|&&pf| row_available(state.read(cx), pf)).count();
    let status = if measured == total && measured > 0 { Status::Ready }
        else if measured > 0 { Status::Partial("See availability on each row", "各行分别显示当前照片的可用性") }
        else { Status::Pending("No effect proven available for this photo", "本图当前没有已证可用的效果") };
    let badge = if total == 0 {
        // 这个段里没有已登记主项（例如脚部/服装）：显示契约参数条数。
        rows.catalogue.len().to_string()
    } else if subs.is_empty() {
        format!("{total}")
    } else {
        format!("{total} +{}", subs.len())
    };
    let mut panel = ToolPanel::new(state, band.id, band.title())
        .status(status)
        .badge(if rows.set > 0 && total > 0 { format!("{}/{}", rows.set, total) } else { badge })
        .modified(rows.modified || rows.set > 0)
        .force_open(!query.is_empty())
        .on_reset(move |_, cx| {
            let keys = keys.clone();
            st.update(cx, |s, cx| {
                s.edit(t("Reset portrait", "复位人像"), |a| {
                    a.extra.retain(|key, _| {
                        !keys
                            .iter()
                            .any(|stored| key == stored || key.starts_with(&format!("{stored}:")))
                    })
                }, cx)
            })
        });
    if open {
        let mut children: Vec<AnyElement> = Vec::new();
        if !rows.main.is_empty() {
            let link_brows = state.read(cx).prefs.panels.portrait_link_lr;
            children.push(caption(t("Registered effects", "已登记效果"), cx));
            for pf in &rows.main {
                // 眉妆左右是真正的一对：PF100（左）/PF117（右）。跟面部精修共用同一个
                // 联动开关（不另立第二套全局选择）：联动时一条滑杆一次 edit 写两个键，
                // 且两侧都 Available 才可操作；上下睫毛 PF105/PF118 不是左右对，永远两条
                // 独立行，不参与联动。
                if *pf == 117 && rows.main.contains(&100) && link_brows {
                    continue;
                }
                if *pf == 100 && rows.main.contains(&117) && link_brows {
                    children.push(brow_pair(state, cx));
                    continue;
                }
                children.push(effect_row(state, *pf, cx));
                for sub in sub_controls_of(*pf) {
                    if rows.subs.contains(sub) {
                        children.push(sub_control_row(state, *sub, cx));
                    }
                }
            }
        }
        if !rows.catalogue.is_empty() {
            children.push(caption(t("Contract parameters — stored only", "契约参数 · 仅保存"), cx));
            for param in rows.catalogue.iter().copied() {
                children.push(h_flex().px_3().py_1().gap_2().text_size(px(11.0)).text_color(pal(cx).text_disabled)
                    .child(div().flex_1().child(param.label())).child(format!("#{} · {}", param.id, t("stored only", "仅保存"))).into_any_element());
            }
        }
        let unresolved = band.unresolved();
        if !unresolved.is_empty() && query.is_empty() {
            let p = pal(cx);
            let ids = unresolved.iter().map(|id| format!("#{id}")).collect::<Vec<_>>().join(" ");
            children.push(
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
                    )))
                    .into_any_element(),
            );
        }
        panel = panel.children(children);
    }
    panel.into_any_element()
}

#[cfg(test)]
#[test]
fn all_groups_cover_every_registered_effect_exactly_once() {
    let mut seen: Vec<u32> = Vec::new();
    for band in BANDS.iter() {
        seen.extend(main_items_in(band.id));
    }
    let registered: Vec<u32> = evidence::all().map(|effect| effect.pf).collect();
    assert_eq!(registered.len(), 147, "the capability table has 147 kind=effect rows");
    assert_eq!(seen.len(), registered.len(), "every main item is listed exactly once");
    let mut sorted = seen.clone();
    sorted.sort_unstable();
    sorted.dedup();
    assert_eq!(sorted.len(), seen.len(), "no group lists a main item twice");
    for pf in &registered {
        assert!(sorted.contains(pf), "PF{pf} is registered but no group lists it");
    }
    // 子控不属于 147 个主项。
    for (_, subs) in TEETH_SUB_CONTROLS {
        for sub in subs {
            assert!(!registered.contains(sub), "PF{sub} must not be counted as a main item");
            assert!(!sorted.contains(sub), "PF{sub} is a sub control, not a main item");
        }
    }
}

#[cfg(test)]
#[test]
fn teeth_sub_controls_are_nested_under_pf108_only() {
    assert_eq!(sub_controls_of(108).to_vec(), vec![109, 110]);
    assert_eq!(sub_items_in("pc-teeth"), vec![109, 110]);
    assert_eq!(main_items_in("pc-teeth"), vec![108, 30025, 30026]);
    assert!(is_sub_control(109) && is_sub_control(110) && !is_sub_control(108));
    for band in BANDS.iter().filter(|band| band.id != "pc-teeth") {
        assert!(sub_items_in(band.id).is_empty(), "only the teeth group nests sub controls");
    }
}

#[cfg(test)]
#[test]
fn every_live_category_lists_all_its_descriptors_once() {
    let mut total = 0usize;
    for category in EffectCategory::ALL {
        let listed = main_effects(category);
        let mut seen: Vec<u32> = Vec::new();
        for pf in &listed {
            assert!(!seen.contains(pf), "PF{pf} is listed twice under {category:?}");
            seen.push(*pf);
            assert_eq!(portrait_capability::category_of(*pf), Some(category), "PF{pf} is in the wrong live panel");
        }
        total += listed.len();
    }
    // 「实时」列出全部描述符：147 个主项 + PF108 下挂的 2 个子控。
    let sub_rows: usize = TEETH_SUB_CONTROLS.iter().map(|(_, subs)| subs.len()).sum();
    assert_eq!(
        total + sub_rows,
        aurora_engine::portrait_registry::WIRED_EFFECT_DESCRIPTORS.len(),
        "Live lists every registered descriptor exactly once"
    );
    assert_eq!(
        aurora_engine::portrait_registry::WIRED_EFFECT_DESCRIPTORS.len(),
        149,
        "capability descriptors = 147 main effects + 2 teeth sub controls"
    );
}

#[cfg(test)]
#[test]
fn pf90032_lives_in_eyes_eye_makeup_not_in_lips() {
    assert!(main_effects(EffectCategory::Eyes).contains(&90032));
    assert!(!main_effects(EffectCategory::Makeup).contains(&90032));
    assert_eq!(group_of(90032), "pc-eyes");
    assert_eq!(group_of(30210), "pc-skin");
    assert_eq!(group_of(400), "pc-body");
    assert_eq!(group_of(30030), "pc-hands");
    assert_eq!(group_of(44297), "pc-feet");
    for pf in [5, 101, 90116, 90130] { assert_eq!(group_of(pf), "pc-eyes"); }
    assert_eq!(group_of(30007), "pc-makeup");
    assert_eq!(group_of(30096), "pc-skin");
    assert!(is_long_tail("pc-hands") && !is_long_tail("pc-body"));
}
