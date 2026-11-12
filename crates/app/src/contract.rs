//! The frontend contract in `docs/frontend-contract/`.
//!
//! * Slider metadata (`params.json`) is served by the engine as
//!   [`lumen_engine::ParamMeta`]; [`crate::widgets::param_c1`] turns a key into a slider.
//! * Feature status (`catalog.md`) is mirrored by the [`Status`] constants below,
//!   which tool panels show as a badge.
//! * The portrait parameter catalogue (`portrait-algorithms.json`) is embedded and
//!   parsed once; only resolved parameters get a control, unresolved ones are
//!   listed greyed out.

use std::sync::LazyLock;

use serde::Deserialize;

use crate::i18n::t;

// ───────────────────────────── Feature status ─────────────────────────────

/// How far the engine implements a feature, per `catalog.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    /// Adjustable, previews live and exports.
    Ready,
    /// Works with a known gap: (English, Chinese) description of the gap.
    Partial(&'static str, &'static str),
    /// Parameters are defined and saved with the variant, but nothing renders them yet.
    Pending(&'static str, &'static str),
    /// The backend is not connected; the control is disabled.
    Missing(&'static str, &'static str),
}

impl Status {
    pub fn chip(self) -> Option<&'static str> {
        match self {
            Status::Ready => None,
            Status::Partial(..) => Some(t("Partial", "部分")),
            Status::Pending(..) => Some(t("Not rendered", "未接入")),
            Status::Missing(..) => Some(t("Unavailable", "后端未接")),
        }
    }

    pub fn note(self) -> Option<&'static str> {
        match self {
            Status::Ready => None,
            Status::Partial(en, zh) | Status::Pending(en, zh) | Status::Missing(en, zh) => Some(t(en, zh)),
        }
    }
}

pub const COLOR_EDITOR: Status = Status::Partial("Uses 5 of the 18 range fields", "18 个范围字段只用到 5 个");
pub const MOIRE: Status = Status::Partial("3×3 low-pass, no frequency-domain filter", "3×3 低通，无频域处理");
pub const LENS_CORRECTION: Status = Status::Partial("Distortion, aberration and purple fringing direction are assumed", "畸变/色差/紫边方向为假设");
pub const KEYSTONE: Status = Status::Partial("Direction assumed, not verified against Capture One", "方向为假设，未与 C1 对拍");
pub const LOCAL_ADJUSTMENTS: Status = Status::Partial("Copied shapes carry zero weight", "复制形状权重为 0");
/// The node graph is stored with the variant; engines still render the layer list in series.
pub const NODE_GRAPH: Status = Status::Partial(
    "Parallel and layer mixers render their branches in series; export still includes disconnected nodes",
    "并行与图层混合器暂按串联渲染分支；导出暂仍包含未连接的节点",
);
pub const SPOTS: Status = Status::Partial("Healing is approximate", "修复为近似实现");
pub const C1_CATALOG: Status = Status::Partial("Collections and WAL only", "仅集合与 WAL");
pub const AI_MASK: Status = Status::Partial("Subject, background and skin only, and only where the segmentation models and ONNX Runtime are installed", "仅主体、背景与皮肤，且需本机装有分割模型与 ONNX Runtime");
pub const TETHER: Status = Status::Partial("Linux libgphoto2 / Windows WIA; capture and live view depend on the camera driver; settings and focus are not read yet", "Linux libgphoto2 / Windows WIA；拍摄和取景取决于相机驱动，设置与对焦尚未读取");
/// The monitor profile is chosen and saved; the engine does not convert previews to it yet.
pub const DISPLAY_PROFILE: Status = Status::Missing(
    "The engine does not convert previews to a monitor profile yet; previews are shown as sRGB",
    "引擎尚未按显示器配置文件转换预览，目前按 sRGB 显示",
);
/// Parsed Capture One parameters the engine stores but does not render yet.
pub const AI_DENOISE: Status = Status::Pending("Saved with the variant; the engine has no AI denoise yet", "随变体保存，引擎尚无 AI 降噪");
pub const BANDING: Status = Status::Pending("Saved with the variant; the engine has no banding reduction yet", "随变体保存，引擎尚无条纹抑制");
/// Interface built ahead of the engine (see `planned`).
pub const MERGE: Status = Status::Missing("The engine cannot merge images yet", "引擎尚不支持合并图像");
pub const STYLE_IMPORT: Status = Status::Partial("Imported styles are available for this session", "导入风格在当前会话可用");
pub const SESSIONS: Status = Status::Partial("Session import is read-only; XMP write-back and .P1X are not connected yet", "Session 支持只读导入；XMP 写回与 .P1X 尚未接入");
pub const PRINT: Status = Status::Missing("Printing is not connected to a backend yet", "打印后端尚未接入");
pub const RETOUCH: Status = Status::Pending("Saved with the variant; the engine does not render it yet", "随变体保存，渲染管线尚未实现");
pub const SURFACE_NOISE: Status = Status::Pending("Surface smoothness has no engine field yet", "表面平滑尚无引擎字段");

/// Export formats the engine has an encoder for (`ExportFormat::is_supported`).
pub fn format_ready(format: lumen_engine::ExportFormat) -> bool {
    format.is_supported()
}

/// The label a format selector shows. Formats without an encoder stay listed so
/// recipes can be set up ahead of the engine, but say that exporting fails.
pub fn format_label(format: lumen_engine::ExportFormat) -> String {
    let ext = format.extension().to_uppercase();
    if format_ready(format) { ext } else { format!("{ext} · {}", t("not connected, export fails", "未接入，导出会失败")) }
}

/// Soft proofing: chosen and shown, but the engine does not convert previews yet.
pub const SOFT_PROOF: Status = Status::Missing("The engine does not render the proof profile yet; the preview is unchanged", "引擎尚未按校样配置文件渲染，预览不变");
/// Lens blur: stored with the variant; the engine has no depth map or blur yet.
pub const LENS_BLUR: Status = Status::Pending("Saved with the variant; the engine has no depth estimation or lens blur yet", "随变体保存，引擎尚无景深估计与镜头模糊");

/// The unit suffix a slider shows for a `ParamMeta::unit`.
pub fn unit_suffix(unit: &str) -> &'static str {
    match unit {
        "EV" => " EV",
        "K" => " K",
        "%" => "%",
        "deg" => "°",
        "px" => " px",
        _ => "",
    }
}

/// The key under which a contract parameter without a typed field is stored in
/// `AdjustmentSet::extra`.
pub fn portrait_key(id: i64) -> String {
    format!("pc:{id}")
}

// ───────────────────────────── Portrait catalogue ─────────────────────────────

#[derive(Deserialize)]
struct RawDoc {
    newly_resolved: Vec<RawParam>,
    ambiguous: std::collections::BTreeMap<String, RawParam>,
}

#[derive(Deserialize)]
struct UnresolvedDoc {
    unresolved: Vec<UnresolvedParam>,
}

#[derive(Deserialize)]
struct UnresolvedParam {
    id: i64,
}

/// Ids the contract could only guess at from their id band. They get no
/// control (`catalog.md`); the panel lists them greyed out so the gap is visible.
pub static UNRESOLVED: LazyLock<Vec<i64>> = LazyLock::new(|| {
    serde_json::from_str::<UnresolvedDoc>(include_str!("../../../docs/frontend-contract/portrait-algorithms.json"))
        .map(|d| {
            let mut ids: Vec<i64> = d.unresolved.into_iter().map(|p| p.id).collect();
            ids.sort_unstable();
            ids
        })
        .unwrap_or_default()
});

#[derive(Deserialize)]
struct RawParam {
    id: i64,
    default: serde_json::Value,
    string_enum: Option<String>,
    ui_key: Option<String>,
    ui_label: Option<String>,
    confidence: f32,
    evidence: String,
    #[serde(default)]
    config_effects: Vec<String>,
}

/// The control a portrait parameter gets.
#[derive(Clone, Debug, PartialEq)]
pub enum Control {
    /// 0 ..= 1, shown as 0 ..= 100.
    Amount,
    /// 0 or 1.
    Toggle,
    /// A small integer choice whose options the contract does not name.
    Mode,
    /// An RGBA color.
    Color,
    /// A material / asset path; only the contract's default value is known.
    Asset(Option<String>),
}

#[derive(Clone, Debug)]
pub struct PortraitParam {
    pub id: i64,
    pub key: String,
    pub label_en: String,
    pub label_zh: String,
    pub default: f32,
    pub default_color: [f32; 4],
    pub control: Control,
    pub confidence: f32,
    pub evidence: String,
    /// Effects that share this parameter (`shared_effect_param`).
    pub shared: Vec<String>,
    /// Nested under another parameter of the same band (`spotless` → `spotless_acne`).
    pub child: bool,
}

impl PortraitParam {
    pub fn label(&self) -> String {
        match crate::i18n::locale() {
            crate::i18n::Locale::En => self.label_en.clone(),
            crate::i18n::Locale::ZhCn => self.label_zh.clone(),
        }
    }

    pub fn store_key(&self) -> String {
        portrait_key(self.id)
    }
}

/// One ID band of `catalog.md`'s portrait table.
pub struct Band {
    pub id: &'static str,
    pub en: &'static str,
    pub zh: &'static str,
    pub lo: i64,
    pub hi: i64,
}

pub const BANDS: [Band; 8] = [
    Band { id: "pc-blemish", en: "Blemish Removal", zh: "瑕疵去除", lo: 0, hi: 999 },
    Band { id: "pc-face", en: "Face Reshaping", zh: "面部重塑", lo: 21000, hi: 21999 },
    Band { id: "pc-skin", en: "Skin, Makeup & Body", zh: "皮肤/妆容/身体", lo: 30000, hi: 39999 },
    Band { id: "pc-makeup", en: "Makeup Details", zh: "妆容细节", lo: 40000, hi: 49999 },
    Band { id: "pc-smart", en: "Smart Recognition", zh: "智能识别", lo: 80000, hi: 89999 },
    Band { id: "pc-material", en: "Materials, Looks & Sky", zh: "素材/样式/天空替换", lo: 90000, hi: 90999 },
    Band { id: "pc-placement", en: "Placement & Edges", zh: "素材定位/边缘调整", lo: 100000, hi: 899999 },
    Band { id: "pc-local", en: "Local Blemish Removal", zh: "局部去瑕疵", lo: 900000, hi: i64::MAX },
];

impl Band {
    pub fn title(&self) -> &'static str {
        t(self.en, self.zh)
    }

    pub fn params(&self) -> impl Iterator<Item = &'static PortraitParam> + use<'_> {
        PORTRAIT.iter().filter(move |p| p.id >= self.lo && p.id <= self.hi)
    }

    /// Unresolved ids in this band.
    pub fn unresolved(&self) -> Vec<i64> {
        UNRESOLVED.iter().copied().filter(|id| *id >= self.lo && *id <= self.hi).collect()
    }
}

/// Every resolved portrait parameter, sorted by id. Unresolved parameters
/// (`unresolved` in the JSON) get no control; see [`UNRESOLVED`].
pub static PORTRAIT: LazyLock<Vec<PortraitParam>> = LazyLock::new(|| {
    let doc: RawDoc = match serde_json::from_str(include_str!("../../../docs/frontend-contract/portrait-algorithms.json")) {
        Ok(doc) => doc,
        Err(e) => {
            log::error!("portrait-algorithms.json: {e}");
            return Vec::new();
        }
    };
    let mut raw: Vec<RawParam> = doc.newly_resolved;
    raw.extend(doc.ambiguous.into_values());
    raw.sort_by_key(|p| p.id);
    raw.dedup_by_key(|p| p.id);

    let keys: Vec<(i64, String)> = raw.iter().filter_map(|p| p.ui_key.clone().map(|k| (p.id, k))).collect();
    raw.into_iter()
        .map(|p| {
            let key = p.ui_key.clone().unwrap_or_else(|| p.string_enum.clone().map(|s| s.replace('/', "_")).unwrap_or_else(|| format!("param{}", p.id)));
            let band = BANDS.iter().position(|b| p.id >= b.lo && p.id <= b.hi);
            let child = keys.iter().any(|(id, k)| {
                *id != p.id && BANDS.iter().position(|b| *id >= b.lo && *id <= b.hi) == band && key.len() > k.len() + 1 && key.starts_with(k.as_str()) && key.as_bytes()[k.len()] == b'_'
            });
            let lower = key.to_ascii_lowercase();
            let control = match &p.default {
                serde_json::Value::Array(_) => Control::Color,
                serde_json::Value::Null => Control::Asset(p.string_enum.clone()),
                _ if lower.ends_with("path") || lower.contains("material") || lower.ends_with("_path") => Control::Asset(p.string_enum.clone()),
                _ if lower.contains("switch") || lower.contains("enable") || key.starts_with("eyeSymmetry") && !lower.contains("combobox") => Control::Toggle,
                serde_json::Value::Number(n) if n.is_i64() => Control::Mode,
                _ => Control::Amount,
            };
            let default = p.default.as_f64().unwrap_or(0.0) as f32;
            let mut default_color = [0.0; 4];
            if let serde_json::Value::Array(v) = &p.default {
                for (i, c) in v.iter().take(4).enumerate() {
                    default_color[i] = c.as_f64().unwrap_or(0.0) as f32;
                }
            }
            let (en, zh) = label_for(&key);
            let side = side_of(&key);
            let label_zh = p.ui_label.clone().or(zh.map(str::to_owned)).unwrap_or_else(|| humanize(&key));
            let label_en = en.map(str::to_owned).unwrap_or_else(|| humanize(&key));
            let (label_en, label_zh) = match side {
                Some((se, sz)) if !label_zh.contains(sz) => (format!("{label_en} ({se})"), format!("{label_zh}（{sz}）")),
                _ => (label_en, label_zh),
            };
            PortraitParam {
                id: p.id,
                key,
                label_en,
                label_zh,
                default,
                default_color,
                control,
                confidence: p.confidence,
                evidence: p.evidence,
                shared: p.config_effects,
                child,
            }
        })
        .collect()
});

fn side_of(key: &str) -> Option<(&'static str, &'static str)> {
    if key.ends_with("_left") || key.contains("Left") {
        Some(("left", "左"))
    } else if key.ends_with("_right") || key.contains("Right") {
        Some(("right", "右"))
    } else {
        None
    }
}

/// `removeGlabellaLines` → `Remove glabella lines`.
fn humanize(key: &str) -> String {
    let mut out = String::new();
    for (i, ch) in key.chars().enumerate() {
        if ch == '_' {
            out.push(' ');
        } else if ch.is_ascii_uppercase() && i > 0 && !out.ends_with(' ') {
            out.push(' ');
            out.push(ch.to_ascii_lowercase());
        } else if i == 0 {
            out.push(ch.to_ascii_uppercase());
        } else {
            out.push(ch.to_ascii_lowercase());
        }
    }
    out
}

/// Labels for parameters the vendor UI registry left unnamed, and English
/// labels for the pinyin keys.
fn label_for(key: &str) -> (Option<&'static str>, Option<&'static str>) {
    let base = key.trim_end_matches("_left").trim_end_matches("_right");
    let hit = LABELS.iter().find(|(k, _, _)| *k == key).or_else(|| LABELS.iter().find(|(k, _, _)| *k == base));
    match hit {
        Some((_, en, zh)) => (Some(*en), Some(*zh)),
        None => (None, None),
    }
}

const LABELS: &[(&str, &str, &str)] = &[
    ("spotless", "Blemish removal", "祛瑕疵"),
    ("spotless_acne", "Acne", "祛痘"),
    ("spotless_spot", "Spots", "祛斑"),
    ("spotless_texture", "Keep texture", "质感保留"),
    ("lightenPouch", "Lighten eye bags", "淡化眼袋"),
    ("eyebrowDegree", "Eyebrows", "眉毛"),
    ("lipstickDegree", "Lipstick", "口红"),
    ("teethWhitening_brightness", "Brightness", "明度"),
    ("teethWhitening_deYellow", "De-yellow", "祛黄"),
    ("faceDecorationDegree", "Face decoration", "脸部装饰"),
    ("mLip", "M lip", "M唇"),
    ("faceShapePreset_degree", "Face shape preset", "脸型预设程度"),
    ("skinTexture", "Skin texture", "肤质"),
    ("tianchongMode", "Fill mode", "填充模式"),
    ("quxiaci_dimian", "Clean ground", "地面去瑕疵"),
    ("quxiaci_beijing", "Clean background", "背景去瑕疵"),
    ("kouxiangbaoliu", "Mouth keep", "口型保留"),
    ("skinColorHue", "Skin hue", "肤色色相"),
    ("aiRemoveTattoo", "Remove tattoos", "AI 祛纹身"),
    ("removeStubble_cleanShave", "Clean shave", "祛胡青"),
    ("removeStubble_stubble", "Stubble", "祛胡渣"),
    ("removeGlabellaLines", "Glabella lines", "祛川字纹"),
    ("removeMarionetteLines", "Marionette lines", "祛木偶纹"),
    ("removeStubble", "Remove stubble", "祛胡须"),
    ("yinyingButouming", "Shadow opacity", "阴影不透明度"),
    ("yinyingXuhua", "Shadow feather", "阴影虚化"),
    ("jianbian", "Gradient", "渐变"),
    ("xuhuaFanwei", "Feather range", "虚化范围"),
    ("yinyingBlendMode", "Shadow blend mode", "阴影混合模式"),
    ("yinyingSucaiPath", "Shadow material", "阴影素材"),
    ("aiYinyingSwitch", "AI shadow", "AI 阴影"),
    ("aiYinyingChengdu", "AI shadow amount", "AI 阴影程度"),
    ("guangyingjunyun_chengdu", "Even lighting amount", "光影均匀程度"),
    ("aiYinyingBlendMode", "AI shadow blend mode", "AI 阴影混合模式"),
    ("guangyingjunyun_mingdu", "Even lighting brightness", "光影均匀明度"),
    ("removeTearTrough", "Nasolabial folds", "祛法令纹"),
    ("guangyingjunyun", "Even lighting", "光影均匀"),
    ("redEyeRemoval", "Red eye", "祛红眼"),
    ("CLOTHES_BLEMISH_REMOVAL", "Clothes blemishes", "衣物去瑕疵"),
    ("CLOTHES_EDGE_SMOOTH_PROTECT_THING", "Protect objects", "保护物体"),
    ("removeCheekWrinkles", "Cheek wrinkles", "祛脸颊纹"),
    ("removeNasalWrinkles", "Nasal wrinkles", "祛鼻纹"),
    ("removePerioralWrinkles", "Perioral wrinkles", "祛口周纹"),
    ("removeSpill", "Remove color spill", "去溢色"),
    ("smileDegree_closeMouth", "Smile (closed mouth)", "微笑（闭嘴）"),
    ("irisEditingSwitch", "AI eye adjustment", "AI眼球调整"),
    ("irisEditingDirectSwitch", "Look at camera", "直视镜头"),
    ("irisEditingLeftHorizontal", "Horizontal", "水平"),
    ("irisEditingLeftVertical", "Vertical", "垂直"),
    ("irisEditingLeftScale", "Size", "大小"),
    ("irisEditingRightHorizontal", "Horizontal", "水平"),
    ("irisEditingRightVertical", "Vertical", "垂直"),
    ("irisEditingRightScale", "Size", "大小"),
    ("removeNoseFlaw", "Nose flaws", "祛鼻部瑕疵"),
    ("skinUnifyColorPicker", "Skin unify picker", "肤色统一取色"),
    ("CLOTHES_SWEAT_STAIN_REMOVAL", "Sweat stains", "祛汗渍"),
    ("blusherColor", "Blusher color", "腮红颜色"),
    ("eyelashColor", "Eyelash color", "睫毛颜色"),
    ("eyebrowColor", "Eyebrow color", "眉毛颜色"),
    ("eyelinerColor", "Eyeliner color", "眼线颜色"),
    ("hairColorUnify", "Unify hair color", "发色统一"),
    ("neutralGrayStereo", "Dodge & burn", "中性灰立体"),
    ("neutralGrayStereo_facial", "Features", "五官"),
    ("neutralGrayStereo_contour", "Contour", "轮廓"),
    ("poreEnhance", "Pores", "毛孔增强"),
    ("chunsebgquchuanbang", "Solid backdrop seams", "纯色背景去穿帮"),
    ("freqSplit_face", "Frequency separation (face)", "高低频（脸部）"),
    ("freqSplit_body", "Frequency separation (body)", "高低频（身体）"),
    ("removeDoubleEyelids", "Eyelid tape", "祛双眼皮贴"),
    ("teethBlemishRemoval", "Teeth blemishes", "牙齿去瑕疵"),
    ("teethAlignment", "Teeth alignment", "牙齿矫正"),
    ("teethBlemishRemoval_edge", "Fill edges", "边缘补齐"),
    ("faceShapePreset_path", "Face shape preset", "脸型预设"),
    ("aiBodyReshape_bodySmooth", "Smooth body line", "身型流畅"),
    ("removeVein", "Veins", "祛青筋"),
    ("freqSplit_face_high", "High frequency (face)", "高频增强（脸部）"),
    ("freqSplit_face_low", "Low frequency (face)", "低频平整（脸部）"),
    ("freqSplit_body_high", "High frequency (body)", "高频增强（身体）"),
    ("freqSplit_body_low", "Low frequency (body)", "低频平整（身体）"),
    ("lipstickColor", "Lipstick color", "口红颜色"),
    ("skinToneUnifyMulti", "Unify skin (group)", "多人肤色统一"),
    ("smileMaterialPath", "Smile material", "微笑素材"),
    ("sediao", "Tone", "色调"),
    ("handNeutralGray", "Hand dodge & burn", "手部中性灰"),
    ("removeDoubleChin_degree", "Amount", "祛除程度"),
    ("removeDoubleChin_shadow", "Shadow", "阴影程度"),
    ("handReshape", "Hand reshape", "手部美型"),
    ("foreheadPlump", "Forehead fill", "额头饱满"),
    ("foreheadPlump_flatten", "Smooth", "平整程度"),
    ("foreheadPlump_brighten", "Brighten", "提亮程度"),
    ("midFacePlump", "Mid-face fill", "中庭饱满"),
    ("midFacePlump_flatten", "Smooth", "平整程度"),
    ("midFacePlump_brighten", "Brighten", "提亮程度"),
    ("cheekPlump", "Cheek fill", "苹果肌饱满"),
    ("nasalBasePlump", "Nasal base fill", "鼻基底饱满"),
    ("nasalBasePlump_flatten", "Smooth", "平整程度"),
    ("nasalBasePlump_brighten", "Brighten", "提亮程度"),
    ("eyeSocketPlump", "Eye socket fill", "眼窝程度"),
    ("geisonPlump", "Temple fill", "太阳穴饱满"),
    ("chinPlump", "Chin fill", "下巴饱满"),
    ("chinPlump_flatten", "Smooth", "平整程度"),
    ("chinPlump_brighten", "Brighten", "提亮程度"),
    ("mouthAngulusPlump", "Mouth corner fill", "嘴角饱满"),
    ("jawlineEnhance", "Jawline", "下颌线增强"),
    ("fullFacePlump", "Full face fill", "全脸饱满"),
    ("facialLiftEyeLiquifyDegree", "Liquify", "液化程度"),
    ("removeSweat", "Sweat", "祛汗"),
    ("facialLiftMouthDegree", "Mouth lift", "嘴部提拉"),
    ("facialLiftMidFaceDegree", "Mid-face lift", "中庭提拉"),
    ("mouthJawReshapeContour", "Contour", "轮廓塑形"),
    ("facialLiftMidFaceSmoothDegree", "Smooth", "平整程度"),
    ("facialLiftMouthSmoothDegree", "Smooth", "平整程度"),
    ("facialLiftMidFaceLiquifyDegree", "Liquify", "液化程度"),
    ("facialLiftEyeDegree", "Eye lift", "眼部提拉"),
    ("facialLiftContourDegree", "Contour", "轮廓"),
    ("shuipingFanzhuan", "Flip horizontal", "水平翻转"),
    ("facialLiftEyeSmoothDegree", "Smooth", "平整程度"),
    ("mouthJawReshape", "Mouth & jaw reshape", "嘴部下颌重塑"),
    ("tiankongJianbian", "Sky gradient", "天空渐变"),
    ("mouthJawReshapeShadow", "Light correction", "光影矫正"),
    ("facialLiftMouthLiquifyDegree", "Liquify", "液化程度"),
    ("xuanzhuanJiaodu", "Rotation", "旋转角度"),
    ("facialLiftTotalDegree", "Full face lift", "全脸提拉"),
    ("makeupRemovalEnable", "Remove makeup", "卸妆"),
    ("makeupRemovalContourDegree", "Contour amount", "修容卸除程度"),
    ("makeupRemovalEyeDegree", "Eye makeup amount", "眼妆卸除程度"),
    ("makeupRemovalFullFaceDegree", "Full face amount", "全脸卸妆程度"),
    ("makeupRemovalFullFaceEnable", "Full face", "全脸卸妆"),
    ("removeBlueVeins", "Blue veins", "祛青筋"),
    ("eyeholeCornerSwitch", "Eye corners", "眼角"),
    ("skinToneSelect_path", "Skin tone", "肤色选择"),
    ("skinToneSelect_degree", "Skin tone amount", "肤色程度"),
    ("skinToneWarmth", "Skin warmth", "肤色冷暖"),
    ("moistureSmoothSkin", "Smooth skin", "水润磨皮"),
    ("skinColorUniform", "Even skin color", "肤色均匀"),
    ("skinWhitening", "Whitening", "美白"),
    ("skinRosy", "Rosy", "红润"),
    ("removeStretchMark", "Stretch marks", "祛妊娠纹"),
    ("moistureSmoothSkin_body", "Smooth skin (body)", "身体磨皮"),
    ("removeNevus", "Moles", "祛痣"),
    ("removeNeckLine", "Neck lines", "祛颈纹"),
    ("bodySpotless", "Body blemishes", "身体去瑕疵"),
    ("removeEyeLines", "Eye lines", "祛眼纹"),
    ("removeForeheadLines", "Forehead lines", "祛抬头纹"),
    ("suitMakeupDegree", "Makeup look amount", "主题妆容程度"),
    ("removeDoubleChin", "Double chin", "祛双下巴"),
    ("chunsebgquxiaci", "Solid backdrop cleanup", "纯色背景去瑕疵"),
    ("qusecaiduanceng", "Remove color banding", "去色彩断层"),
    ("teethRepair", "Teeth repair", "牙齿修复"),
    ("aiBodyReshape", "AI body reshape", "AI 美型"),
    ("bianyuantiaozheng", "Edge adjustment", "边缘调整"),
    ("bianyuanGuodu", "Edge transition", "边缘过渡"),
    ("sewen", "Temperature", "色温"),
    ("baohedu", "Saturation", "饱和度"),
    ("liangdu", "Brightness", "亮度"),
    ("xuhua", "Feather", "虚化"),
    ("jingwuSecai", "Scene color", "景物色彩"),
    ("renwuSecai", "Subject color", "人物色彩"),
    ("jingwuFanshe", "Scene reflection", "景物反射"),
    ("jingwuShuixiaoXuhua", "Water reflection blur", "水面倒影虚化"),
    ("butouming", "Opacity", "不透明度"),
    ("quxiaci_chengdu", "Cleanup amount", "去瑕疵程度"),
    ("removeOilShine", "Oil shine", "祛油光"),
    ("skinToneUnifyFace", "Unify skin (face)", "面部肤色统一"),
    ("skinToneUnifyBody", "Unify skin (body)", "身体肤色统一"),
    ("hairPartFill", "Hair parting fill", "发缝填充"),
    ("removeStrayHair", "Stray hair", "祛碎发"),
    ("bodySpotless_baby", "Baby blemishes", "宝宝去瑕疵"),
    ("bgReplacePath", "Background", "背景替换"),
    ("skyReplacePath", "Sky", "天空替换"),
    ("chuizhiWeizhi", "Vertical position", "垂直位置"),
    ("shuipingWeizhi", "Horizontal position", "水平位置"),
    ("removeEyeBag", "Eye bags", "祛眼袋"),
    ("lipLineCorrection", "Lip line", "唇线矫正"),
    ("glassesGlareRemoval", "Glasses glare", "祛眼镜反光"),
    ("removeWhiteHair", "White hair", "祛白发"),
    ("hairColorSelect_path", "Hair color", "发色选择"),
    ("daxiao", "Size", "大小"),
    ("hairColorSelect_degree", "Hair color amount", "发色程度"),
    ("hairSmooth", "Smooth hair", "头发柔顺"),
    ("CLOTHES_EDGE_SMOOTH", "Smooth clothing edges", "衣物边缘平滑"),
    ("hairGloss", "Hair gloss", "头发光泽"),
    ("eyeSymmetry_size_combobox", "Size reference", "大小参照"),
    ("bgColor", "Background color", "背景颜色"),
    ("bgBlendMode", "Background blend mode", "背景混合模式"),
    ("redBloodRemoval", "Bloodshot eyes", "祛红血丝"),
    ("eyeWhitePurify", "Whiten eyes", "眼白净化"),
    ("smileDegree_openMouth", "Smile (open mouth)", "微笑（张嘴）"),
    ("repairBreast", "Chest repair", "胸部修复"),
    ("makeupProtection", "Protect makeup", "妆容保护"),
    ("arcProtection", "Protect curves", "弧线保护"),
    ("REMOVE_CLOTHES_CREASE_TOP", "Top creases", "上衣去褶皱"),
    ("REMOVE_CLOTHES_CREASE_TOP_COARSE", "Coarse", "粗褶皱"),
    ("REMOVE_CLOTHES_CREASE_TOP_DETAIL", "Fine", "细褶皱"),
    ("REMOVE_CLOTHES_CREASE_BOTTOM", "Bottom creases", "下装去褶皱"),
    ("REMOVE_CLOTHES_CREASE_BOTTOM_COARSE", "Coarse", "粗褶皱"),
    ("REMOVE_CLOTHES_CREASE_BOTTOM_DETAIL", "Fine", "细褶皱"),
    ("SkinTexture_00", "Skin texture material", "肤质素材"),
];
