//! Slider metadata for every Capture One tool that has a numeric slider.
//!
//! Evidence for every row is `outputs/analysis/c1-tool-metadata.md` §2.1
//! (`RangeDefinitions.GetDefaultPropertyRange`) and §3 (per-tool XAML and
//! `ToolDefaults`). Rows whose range only exists in a RAW info block are marked
//! [`MetaSource::RawImage`]: the numbers here are the neutral fallback, and the
//! real ones must come from [`ParamOverrides`].
//!
//! Curves and Color Balance are absent on purpose: the report gives them no
//! numeric min/max (§3.5, §3.6). `AIDenoiseAmount` and `BandingReductionAmount`
//! are absent for the same reason (§2.1: no static range).
//!
//! `LensLccLightFalloffAmount` appears once, under `LensCorrection`; the LCC
//! tool binds the same property (§3.24b).

use std::collections::HashMap;

use serde::Serialize;

/// Where the static numbers came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum MetaSource {
    /// Range and default both have a hardcoded value in C1 (§2.1 / §2.2).
    Hardcoded,
    /// C1 reads the range and default from the image's RAW info block; the
    /// static row is only a neutral fallback until [`ParamOverrides`] arrives.
    RawImage,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ParamMeta {
    /// C1 variant property id, the same key `.copreset` files use.
    pub key: &'static str,
    pub label: &'static str,
    pub label_zh: &'static str,
    pub min: f32,
    pub max: f32,
    pub default: f32,
    /// `""`, `"EV"`, `"K"`, `"%"`, `"deg"`, or `"px"`.
    pub unit: &'static str,
    /// UI increment from the tool's XAML `Increment`, or the
    /// `SpeedEditIncrementConstants` value when the XAML sets none (§2.3).
    pub step: f32,
    /// `VariantPropertyDecimalPlaces` (§2.3); 0 when C1 shows integers.
    pub decimals: u8,
    /// Inspector tool id, e.g. `Exposure` or `RetouchEyes`.
    pub group: &'static str,
    pub source: MetaSource,
}

#[allow(clippy::too_many_arguments)]
const fn meta(
    key: &'static str,
    label: &'static str,
    label_zh: &'static str,
    min: f32,
    max: f32,
    default: f32,
    unit: &'static str,
    step: f32,
    decimals: u8,
    group: &'static str,
    source: MetaSource,
) -> ParamMeta {
    ParamMeta { key, label, label_zh, min, max, default, unit, step, decimals, group, source }
}

use MetaSource::{Hardcoded, RawImage};

/// Every slider the engine knows a range for.
///
/// `USM*` defaults come from the camera baseline in `coreConfig.cdx` (§3.12),
/// which is still per camera, hence [`RawImage`].
///
/// `step` / `decimals` come from the XAML `Increment` and
/// `VariantPropertyDecimalPlaces` (§2.3, §3); `0.0` / `0` means no evidence was
/// found and the row is marked `未证明` inline.
pub static CATALOG: &[ParamMeta] = &[
    // §3.1 Exposure
    meta("Exposure", "Exposure", "曝光", -2.5, 2.5, 0.0, "EV", 0.1, 2, "Exposure", RawImage),
    meta("Contrast", "Contrast", "对比度", -50.0, 50.0, 0.0, "", 0.1, 0, "Exposure", RawImage),
    meta("Brightness", "Brightness", "亮度", -50.0, 50.0, 0.0, "", 0.1, 0, "Exposure", RawImage),
    meta("Saturation", "Saturation", "饱和度", -100.0, 100.0, 0.0, "", 0.1, 0, "Exposure", RawImage),
    // §3.2 White Balance; kelvin default is proven per image, not in C#
    meta("WhiteBalanceTemperature", "Temperature", "色温", 800.0, 14000.0, 5600.0, "K", 25.0, 0, "WhiteBalance", RawImage),
    meta("WhiteBalanceTint", "Tint", "色调", -50.0, 50.0, 0.0, "", 0.1, 1, "WhiteBalance", RawImage),
    // §3.3 High Dynamic Range
    meta("HighlightRecovery", "Highlights", "高光", -100.0, 100.0, 0.0, "", 0.1, 0, "HighDynamicRange", RawImage),
    meta("ShadowRecovery", "Shadows", "阴影", -100.0, 100.0, 0.0, "", 0.1, 0, "HighDynamicRange", RawImage),
    meta("WhiteRecovery", "Whites", "白色", -100.0, 100.0, 0.0, "", 0.1, 0, "HighDynamicRange", RawImage),
    meta("BlackRecovery", "Blacks", "黑色", -100.0, 100.0, 0.0, "", 0.1, 0, "HighDynamicRange", RawImage),
    // §3.4 Levels, normalized 0..1 per channel; gamma is a UI concept only
    meta("ShadowRGB", "Shadow", "输入阴影", 0.0, 1.0, 0.0, "", 0.000390625, 0, "Levels", Hardcoded),
    meta("MidtoneRGB", "Midtone", "输入中间调", -1.0, 1.0, 0.0, "", 0.0025, 2, "Levels", Hardcoded),
    meta("HighlightRGB", "Highlight", "输入高光", 0.0, 1.0, 1.0, "", 0.000390625, 0, "Levels", Hardcoded),
    // 未证明: no Increment in the report.
    meta("TargetShadowRGB", "Target Shadow", "目标阴影", 0.0, 1.0, 0.0, "", 0.0, 0, "Levels", Hardcoded),
    meta("TargetHighlightRGB", "Target Highlight", "目标高光", 0.0, 1.0, 1.0, "", 0.0, 0, "Levels", Hardcoded),
    // §3.7 Black & White
    // 未证明: §3.7 lists no Increment/DecimalPlaces for these sliders.
    meta("BwRed", "Red", "红", -100.0, 100.0, 0.0, "", 0.0, 0, "BlackWhite", Hardcoded),
    meta("BwYellow", "Yellow", "黄", -100.0, 100.0, 0.0, "", 0.0, 0, "BlackWhite", Hardcoded),
    meta("BwGreen", "Green", "绿", -100.0, 100.0, 0.0, "", 0.0, 0, "BlackWhite", Hardcoded),
    meta("BwCyan", "Cyan", "青", -100.0, 100.0, 0.0, "", 0.0, 0, "BlackWhite", Hardcoded),
    meta("BwBlue", "Blue", "蓝", -100.0, 100.0, 0.0, "", 0.0, 0, "BlackWhite", Hardcoded),
    meta("BwMagenta", "Magenta", "品红", -100.0, 100.0, 0.0, "", 0.0, 0, "BlackWhite", Hardcoded),
    meta("BwLightHue", "Light Hue", "高光色相", 0.0, 360.0, 0.0, "deg", 0.0, 0, "BlackWhite", Hardcoded),
    meta("BwLightSaturation", "Light Saturation", "高光饱和度", 0.0, 100.0, 0.0, "", 0.0, 0, "BlackWhite", Hardcoded),
    meta("BwDarkHue", "Dark Hue", "阴影色相", 0.0, 360.0, 0.0, "deg", 0.0, 0, "BlackWhite", Hardcoded),
    meta("BwDarkSaturation", "Dark Saturation", "阴影饱和度", 0.0, 100.0, 0.0, "", 0.0, 0, "BlackWhite", Hardcoded),
    // §3.8 Color Editor; code `ToolDefaults.ColorEditor` wins over the XAML 1..30
    // 未证明: §3.8 lists no Increment/DecimalPlaces.
    meta("BasicSmoothness", "Smoothness", "平滑度", 1.0, 29.1, 1.0, "", 0.0, 0, "ColorEditor", Hardcoded),
    meta("BasicHue", "Hue", "色相", -30.0, 30.0, 0.0, "", 0.0, 0, "ColorEditor", Hardcoded),
    meta("BasicSaturation", "Saturation", "饱和度", -80.0, 80.0, 0.0, "", 0.0, 0, "ColorEditor", Hardcoded),
    meta("BasicLightness", "Lightness", "明度", -5.0, 5.0, 0.0, "", 0.0, 0, "ColorEditor", Hardcoded),
    // §3.9 Clarity
    meta("Clarity", "Clarity", "清晰度", 0.0, 1.0, 0.0, "", 0.1, 0, "Clarity", RawImage),
    meta("ClarityStructure", "Structure", "结构", 0.0, 1.0, 0.0, "", 0.1, 0, "Clarity", RawImage),
    // §3.10 Dehaze
    meta("DehazeAmount", "Amount", "强度", -100.0, 100.0, 0.0, "", 0.1, 0, "Dehaze", RawImage),
    // §3.11 Vignetting
    meta("LensVignettingAmount", "Amount", "强度", -4.0, 4.0, 0.0, "", 0.1, 2, "Vignetting", RawImage),
    // §3.12 Sharpening; defaults are the `coreConfig.cdx` camera baseline
    meta("USMAmount", "Amount", "锐化量", 0.0, 500.0, 160.0, "", 1.0, 0, "Sharpening", RawImage),
    meta("USMRadius", "Radius", "半径", 0.2, 2.5, 0.8, "px", 0.1, 1, "Sharpening", RawImage),
    meta("USMThreshold", "Threshold", "阈值", 0.0, 12.0, 1.0, "", 0.1, 1, "Sharpening", RawImage),
    meta("USMHalo", "Halo Suppression", "光晕抑制", 0.0, 100.0, 0.0, "", 1.0, 0, "Sharpening", RawImage),
    // §3.13 Noise Reduction
    meta("NoiseReductionAmount", "Luminance", "亮度", 0.0, 100.0, 0.0, "", 1.0, 0, "Noise", RawImage),
    meta("NoiseDetailsAmount", "Details", "细节", 0.0, 100.0, 0.0, "", 1.0, 0, "Noise", RawImage),
    meta("SurfaceSmoothnessAmount", "Surface", "表面", 0.0, 100.0, 0.0, "", 1.0, 0, "Noise", RawImage),
    meta("ColorNoiseReductionAmount", "Color", "颜色", 0.0, 100.0, 50.0, "", 1.0, 0, "Noise", RawImage),
    meta("CleanNoiseAmount", "Single Pixel", "单像素", 0.0, 100.0, 0.0, "", 1.0, 0, "Noise", RawImage),
    // §3.14 Film Grain
    meta("FilmGrainAmount", "Amount", "强度", 0.0, 100.0, 0.0, "", 1.0, 0, "FilmGrain", RawImage),
    meta("FilmGrainGranularity", "Granularity", "颗粒度", 0.0, 100.0, 0.0, "", 1.0, 0, "FilmGrain", RawImage),
    // §3.15 Keystone; KeystoneTiltX/Y are raw point sets, not sliders
    meta("KeystoneV", "Vertical", "垂直", -5.5, 5.5, 0.0, "", 1.0, 2, "Keystone", RawImage),
    meta("KeystoneH", "Horizontal", "水平", -5.5, 5.5, 0.0, "", 1.0, 2, "Keystone", RawImage),
    meta("KeystoneSkew", "Skew", "偏斜", -45.0, 45.0, 0.0, "deg", 0.1, 2, "Keystone", RawImage),
    meta("KeystoneAspect", "Aspect", "宽高比", -0.5, 0.5, 0.0, "", 0.1, 1, "Keystone", RawImage),
    // §3.16 Rotation
    meta("RotationAdjustment", "Angle", "角度", -45.0, 45.0, 0.0, "deg", 0.01, 2, "Rotation", Hardcoded),
    // §3.17 + §3.24b Lens Correction / LCC
    meta("LensGeometricAberrationAmount", "Distortion", "畸变", 0.0, 120.0, 0.0, "", 1.0, 0, "LensCorrection", RawImage),
    meta("LensRadialBlurAmount", "Sharpness Falloff", "锐度衰减", 0.0, 120.0, 0.0, "", 1.0, 0, "LensCorrection", RawImage),
    meta("LensLccLightFalloffAmount", "Light Falloff", "光照衰减", 0.0, 120.0, 0.0, "", 1.0, 0, "LensCorrection", RawImage),
    // §3.18 Moire
    // 未证明: §3.18 lists no Increment/DecimalPlaces.
    meta("MoireAmount", "Amount", "强度", 0.0, 100.0, 0.0, "", 0.0, 0, "Moire", RawImage),
    meta("MoirePattern", "Pattern", "图案", 4.0, 80.0, 4.0, "", 0.0, 0, "Moire", RawImage),
    // §3.19 Purple Fringing
    // 未证明: §3.19 lists no Increment/DecimalPlaces.
    meta("PurpleFringingAmount", "Amount", "强度", -100.0, 100.0, 0.0, "", 0.0, 0, "PurpleFringing", RawImage),
    // §3.20 Retouch Face Skin
    meta("RetouchingOpacity", "Opacity", "不透明度", 0.0, 100.0, 100.0, "%", 1.0, 0, "RetouchFaceSkin", RawImage),
    meta("BlemishRemovalAmount", "Blemish Removal", "瑕疵", 0.0, 100.0, 0.0, "", 1.0, 0, "RetouchFaceSkin", RawImage),
    meta("DarkCirclesReductionAmount", "Dark Circles", "黑眼圈", 0.0, 100.0, 0.0, "", 1.0, 0, "RetouchFaceSkin", RawImage),
    meta("SkinEveningAmount", "Even Skin Tone", "均匀肤色", 0.0, 100.0, 0.0, "", 1.0, 0, "RetouchFaceSkin", RawImage),
    meta("SkinEveningTexture", "Skin Texture", "纹理", -100.0, 100.0, 0.0, "", 1.0, 0, "RetouchFaceSkin", RawImage),
    meta("FaceSculptingContouring", "Contouring", "轮廓", 0.0, 100.0, 0.0, "", 1.0, 0, "RetouchFaceSkin", RawImage),
    // §3.21 Retouch Hair
    meta("StrayHairRemovalInnerAmount", "Inner", "内侧", 0.0, 100.0, 0.0, "", 1.0, 0, "RetouchHair", RawImage),
    meta("StrayHairRemovalOuterAmount", "Outer", "外侧", 0.0, 100.0, 0.0, "", 1.0, 0, "RetouchHair", RawImage),
    // §3.22 Retouch Teeth
    meta("TeethWhiteningBrightness", "Brightness", "亮度", 0.0, 100.0, 0.0, "", 1.0, 0, "RetouchTeeth", RawImage),
    meta("TeethWhiteningDesaturation", "Desaturation", "去饱和", 0.0, 100.0, 0.0, "", 1.0, 0, "RetouchTeeth", RawImage),
    // §3.23 Retouch Eyes; C1's VariantProperties lists EyeBrightRightIrisAmount twice
    meta("EyeBrightBothIrisAmount", "Iris (both)", "双眼虹膜", 0.0, 100.0, 0.0, "", 1.0, 0, "RetouchEyes", RawImage),
    meta("EyeBrightBothScleraAmount", "Sclera (both)", "双眼巩膜", 0.0, 100.0, 0.0, "", 1.0, 0, "RetouchEyes", RawImage),
    meta("EyeBrightLeftIrisAmount", "Iris (left)", "左虹膜偏移", -30.0, 30.0, 0.0, "", 1.0, 0, "RetouchEyes", RawImage),
    meta("EyeBrightLeftScleraAmount", "Sclera (left)", "左巩膜偏移", -30.0, 30.0, 0.0, "", 1.0, 0, "RetouchEyes", RawImage),
    meta("EyeBrightRightIrisAmount", "Iris (right)", "右虹膜偏移", -30.0, 30.0, 0.0, "", 1.0, 0, "RetouchEyes", RawImage),
    meta("EyeBrightRightScleraAmount", "Sclera (right)", "右巩膜偏移", -30.0, 30.0, 0.0, "", 1.0, 0, "RetouchEyes", RawImage),
    // §3.24 Local adjustment layer opacity; local-only range
    meta("LayerOpacity", "Opacity", "不透明度", 0.0, 100.0, 100.0, "%", 1.0, 0, "LocalAdjustments", Hardcoded),
];

pub fn catalog() -> &'static [ParamMeta] {
    CATALOG
}

pub fn by_key(key: &str) -> Option<&'static ParamMeta> {
    CATALOG.iter().find(|m| m.key == key)
}

pub fn by_group(group: &str) -> Vec<&'static ParamMeta> {
    CATALOG.iter().filter(|m| m.group == group).collect()
}

/// Per-image range and default overrides read from the RAW info block.
///
/// C1 derives `_START` / `_END` from the range key in
/// [`VARIANT_RANGES`] (§2.1): a single-valued key means the symmetric range
/// `[-v, +v]`, `_START`/`_END` pair an asymmetric one. `IB_*_DEFAULT` keys set
/// the default (§2.2).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParamOverrides {
    defaults: HashMap<&'static str, f32>,
    ranges: HashMap<&'static str, (f32, f32)>,
}

/// `P1.C1.Core.Ranges.RangeDefinitions.ValidVariantPropertyRanges`, verbatim.
///
/// Evidence: `work/dec/P1.C1.Core.Ranges.RangeDefinitions.decompiled.cs:9-50`.
/// The report calls it 41 entries (§2.1) but the dictionary literal has 40
/// rows; the two test-only entries the report may have counted are not here.
/// `KeystoneAmount`, `LensLightFallOffAmount`, `FilmGrainDensity`,
/// `AIDenoiseAmount` and `BandingReductionAmount` are absent from [`CATALOG`],
/// so their ranges are stored but never applied.
pub const VARIANT_RANGES: &[(&str, &str)] = &[
    ("Exposure", "IB_F_EXPOSURE"),
    ("Contrast", "IB_F_CONTRAST"),
    ("Brightness", "IB_F_BRIGHTNESS"),
    ("Saturation", "IB_F_SATURATION"),
    ("ShadowRecovery", "IB_F_SHADOWRECOVERY"),
    ("HighlightRecovery", "IB_F_HIGHLIGHTRECOVERY"),
    ("WhiteRecovery", "IB_F_WHITERECOVERY"),
    ("BlackRecovery", "IB_F_BLACKRECOVERY"),
    ("Clarity", "IB_F_CLARITY_AMOUNT"),
    ("ClarityStructure", "IB_F_STRUCTURE_AMOUNT"),
    ("DehazeAmount", "IB_F_DEHAZEAMOUNT"),
    ("WhiteBalanceTemperature", "IB_F_KELVIN"),
    ("WhiteBalanceTint", "IB_F_TINT"),
    ("LensLightFallOffAmount", "IB_F_LENS_LIGHTFALLOFF"),
    ("LensRadialBlurAmount", "IB_F_LENS_SHARPNESSFALLOFF"),
    ("LensGeometricAberrationAmount", "IB_F_LENS_DISTORTION"),
    ("LensVignettingAmount", "IB_F_LENS_VIGNETTING"),
    ("LensLccLightFalloffAmount", "IB_F_LCC_LIGHTFALLOFF"),
    ("PurpleFringingAmount", "IB_F_PURPLEFRINGING"),
    ("MoireAmount", "IB_F_MOIRE_AMOUNT"),
    ("MoirePattern", "IB_F_MOIRE_RADIUS"),
    ("KeystoneH", "IB_F_KEYSTONE_H"),
    ("KeystoneV", "IB_F_KEYSTONE_V"),
    ("KeystoneAmount", "IB_F_KEYSTONE_AMOUNT"),
    ("KeystoneAspect", "IB_F_KEYSTONE_ASPECT"),
    ("KeystoneSkew", "IB_F_KEYSTONE_SKEW"),
    ("FilmGrainAmount", "IB_F_FILMGRAIN_AMOUNT"),
    ("FilmGrainDensity", "IB_F_FILMGRAIN_DENSITY"),
    ("FilmGrainGranularity", "IB_F_FILMGRAIN_GRANULARITY"),
    ("UnsharpMaskAmount", "IB_F_USMAMOUNT"),
    ("UnsharpMaskRadius", "IB_F_USMRADIUS"),
    ("UnsharpMaskThreshold", "IB_F_USMTHRESHOLD"),
    ("UnsharpMaskHalo", "IB_F_USMHALOCONTROL"),
    ("NoiseReductionAmount", "IB_F_NOISEREDUCTIONAMOUNT"),
    ("NoiseDetailsAmount", "IB_F_DETAILSAMOUNT"),
    ("AIDenoiseAmount", "IB_F_AIDENOISEAMOUNT"),
    ("CleanNoiseAmount", "IB_F_CLEANLONGEXPOSUREAMOUNT"),
    ("SurfaceSmoothnessAmount", "IB_F_SURFACESMOOTHNESSAMOUNT"),
    ("ColorNoiseReductionAmount", "IB_F_CNRAMOUNT"),
    ("BandingReductionAmount", "IB_F_BANDINGREDUCTIONAMOUNT"),
];

/// [`CATALOG`] keeps C1's short `.copreset` ids for sharpening while the range
/// table uses the long property names; map back to the catalog id.
fn catalog_key(property: &'static str) -> &'static str {
    match property {
        "UnsharpMaskAmount" => "USMAmount",
        "UnsharpMaskRadius" => "USMRadius",
        "UnsharpMaskThreshold" => "USMThreshold",
        "UnsharpMaskHalo" => "USMHalo",
        other => other,
    }
}

/// Info-block range key for a C1 property, e.g. `"Exposure" -> "IB_F_EXPOSURE"`.
/// Accepts either the range-table name or the [`CATALOG`] alias.
pub fn range_key(property: &str) -> Option<&'static str> {
    VARIANT_RANGES
        .iter()
        .find(|(p, _)| *p == property || catalog_key(*p) == property)
        .map(|(_, k)| *k)
}

/// `IB_*_DEFAULT` -> C1 property id. Only the keys whose property is in
/// [`CATALOG`]; the source table is §2.2 of the report.
const DEFAULT_KEYS: &[(&str, &str)] = &[
    ("IB_F_EXPOSURE_DEFAULT", "Exposure"),
    ("IB_F_BRIGHTNESS_DEFAULT", "Brightness"),
    ("IB_F_CONTRAST_DEFAULT", "Contrast"),
    ("IB_F_SATURATION_DEFAULT", "Saturation"),
    ("IB_F_HIGHLIGHTRECOVERY_DEFAULT", "HighlightRecovery"),
    ("IB_F_SHADOWRECOVERY_DEFAULT", "ShadowRecovery"),
    ("IB_F_WHITERECOVERY_DEFAULT", "WhiteRecovery"),
    ("IB_F_BLACKRECOVERY_DEFAULT", "BlackRecovery"),
    ("IB_F_TARGETLEVELSHADOWX_DEFAULT", "TargetShadowRGB"),
    ("IB_F_TARGETLEVELHIGHLIGHTX_DEFAULT", "TargetHighlightRGB"),
    ("IB_F_CLARITY_AMOUNT_DEFAULT", "Clarity"),
    ("IB_F_STRUCTURE_AMOUNT_DEFAULT", "ClarityStructure"),
    ("IB_F_DEHAZEAMOUNT_DEFAULT", "DehazeAmount"),
    ("IB_F_PURPLEFRINGING_DEFAULT", "PurpleFringingAmount"),
    ("IB_F_KEYSTONE_AMOUNT_DEFAULT", "KeystoneV"),
    ("IB_F_KEYSTONE_ASPECT_DEFAULT", "KeystoneAspect"),
    ("IB_F_KEYSTONE_SKEW_DEFAULT", "KeystoneSkew"),
    ("IB_F_USMAMOUNT_DEFAULT", "USMAmount"),
    ("IB_F_USMRADIUS_DEFAULT", "USMRadius"),
    ("IB_F_USMTHRESHOLD_DEFAULT", "USMThreshold"),
    ("IB_F_USMHALOCONTROL_DEFAULT", "USMHalo"),
    ("IB_F_CNRAMOUNT_DEFAULT", "ColorNoiseReductionAmount"),
    ("IB_F_CLEANLONGEXPOSUREAMOUNT_DEFAULT", "CleanNoiseAmount"),
    ("IB_F_NOISEREDUCTIONAMOUNT_DEFAULT", "NoiseReductionAmount"),
    ("IB_F_FILMGRAIN_GRANULARITY_DEFAULT", "FilmGrainGranularity"),
    ("IB_F_FILMGRAIN_AMOUNT_DEFAULT", "FilmGrainAmount"),
    ("IB_F_LENS_DISTORTION_DEFAULT", "LensGeometricAberrationAmount"),
    ("IB_F_LENS_LIGHTFALLOFF_DEFAULT", "LensLccLightFalloffAmount"),
    ("IB_F_OPACITY_DEFAULT", "RetouchingOpacity"),
    ("IB_F_BLEMISH_REMOVAL_DEFAULT", "BlemishRemovalAmount"),
    ("IB_F_DARK_CIRCLES_DEFAULT", "DarkCirclesReductionAmount"),
    ("IB_F_COUNTOURING_DEFAULT", "FaceSculptingContouring"),
    ("IB_F_EVEN_SKIN_DEFAULT", "SkinEveningAmount"),
    ("IB_F_SMOOTH_SKIN_TEXTURE_DEFAULT", "SkinEveningTexture"),
    ("IB_F_TEETH_DESATURATION_DEFAULT", "TeethWhiteningDesaturation"),
    ("IB_F_TEETH_BRIGHTNESS_DEFAULT", "TeethWhiteningBrightness"),
    ("IB_F_HAIR_REMOVAL_INNER_AMOUNT_DEFAULT", "StrayHairRemovalInnerAmount"),
    ("IB_F_HAIR_REMOVAL_OUTER_AMOUNT_DEFAULT", "StrayHairRemovalOuterAmount"),
    // `detailsAmount_* -> IB_F_DETAILSAMOUNT_*` and
    // `SurfaceSmoothness_* -> IB_F_SURFACESMOOTHNESSAMOUNT_*` per
    // `outputs/analysis/raw-info-block.md:25,27`.
    ("IB_F_DETAILSAMOUNT_DEFAULT", "NoiseDetailsAmount"),
    ("IB_F_SURFACESMOOTHNESSAMOUNT_DEFAULT", "SurfaceSmoothnessAmount"),
];

impl ParamOverrides {
    /// Builds overrides from already-decoded `key -> value` info block pairs.
    /// Unknown keys are ignored, so a superset is fine.
    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut defaults = HashMap::new();
        let mut starts: HashMap<&'static str, f32> = HashMap::new();
        let mut ends: HashMap<&'static str, f32> = HashMap::new();
        let mut ranges: HashMap<&'static str, (f32, f32)> = HashMap::new();
        for (key, value) in pairs {
            let key = key.as_ref();
            let Ok(value) = value.as_ref().trim().parse::<f32>() else { continue };
            if let Some((_, prop)) = DEFAULT_KEYS.iter().find(|(k, _)| *k == key) {
                defaults.insert(*prop, value);
                continue;
            }
            // `_LOCAL_*` keys end in `_START`/`_END` too, but their base never
            // matches a range key, so local ranges are ignored.
            let (base, side) = match key.strip_suffix("_START") {
                Some(base) => (base, 1),
                None => match key.strip_suffix("_END") {
                    Some(base) => (base, 2),
                    None => (key, 3),
                },
            };
            let Some((prop, _)) = VARIANT_RANGES.iter().find(|(_, k)| *k == base) else { continue };
            let prop = catalog_key(*prop);
            match side {
                1 => {
                    starts.insert(prop, value);
                }
                2 => {
                    ends.insert(prop, value);
                }
                _ => {
                    ranges.insert(prop, (-value.abs(), value.abs()));
                }
            }
        }
        for (prop, min) in starts {
            if let Some(max) = ends.remove(prop) {
                ranges.insert(prop, (min, max));
            }
        }
        let mut out = Self { defaults, ranges };
        out.ranges.retain(|_, (min, max)| min.is_finite() && max.is_finite() && min < max);
        out
    }

    /// Reads a whole TIFF-based RAW, taking the per-frame info-block values out
    /// of EXIF and the Nikon MakerNote; a plain `KEY=VALUE` text block is also
    /// accepted. See [`crate::rawinfo`] for why the file carries no `IB_*` text.
    pub fn from_raw_info_block(bytes: &[u8]) -> Self {
        if let Ok(block) = crate::parse_info_block(bytes) {
            if !block.values.is_empty() {
                let mut overrides = Self::from_pairs(block.values.iter().map(|(k, v)| (k.as_str(), v.to_string())));
                block.apply_to(&mut overrides);
                return overrides;
            }
        }
        let text = String::from_utf8_lossy(bytes);
        Self::from_pairs(text.lines().filter_map(|line| line.split_once('=')))
    }

    /// Explicit range override, for a future RAW range reader.
    pub fn set_range(&mut self, key: &'static str, min: f32, max: f32) {
        if min.is_finite() && max.is_finite() && min < max {
            self.ranges.insert(key, (min, max));
        }
    }

    /// Explicit default override, for a RAW or camera-default reader.
    pub fn set_default(&mut self, key: &'static str, value: f32) {
        if value.is_finite() {
            self.defaults.insert(key, value);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.defaults.is_empty() && self.ranges.is_empty()
    }

    /// `(min, max, default)` for `meta`, with the image's own values taking
    /// precedence; the default is clamped into the resulting range.
    pub fn apply(&self, meta: &ParamMeta) -> (f32, f32, f32) {
        let (mut min, mut max) = self.ranges.get(meta.key).copied().unwrap_or((meta.min, meta.max));
        if !(min < max) {
            (min, max) = (meta.min, meta.max);
        }
        let default = self.defaults.get(meta.key).copied().unwrap_or(meta.default).clamp(min, max);
        (min, max, default)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_unique_and_non_empty() {
        let mut seen = std::collections::HashSet::new();
        for m in catalog() {
            assert!(!m.key.is_empty() && !m.group.is_empty(), "{m:?}");
            assert!(seen.insert(m.key), "duplicate key {}", m.key);
        }
    }

    #[test]
    fn ranges_are_ordered_and_defaults_inside() {
        for m in catalog() {
            assert!(m.min < m.max, "{}: {}..{}", m.key, m.min, m.max);
            assert!(
                m.default >= m.min && m.default <= m.max,
                "{}: default {} outside {}..{}",
                m.key,
                m.default,
                m.min,
                m.max
            );
        }
    }

    #[test]
    fn by_group_filters_one_tool() {
        let eyes = by_group("RetouchEyes");
        assert_eq!(eyes.len(), 6);
        assert!(eyes.iter().all(|m| m.group == "RetouchEyes"));
        assert!(by_group("NoSuchTool").is_empty());
    }

    #[test]
    fn by_key_hits_and_misses() {
        let m = by_key("USMAmount").expect("USMAmount");
        assert_eq!((m.min, m.max, m.default), (0.0, 500.0, 160.0));
        assert_eq!(m.label_zh, "锐化量");
        assert!(by_key("usmamount").is_none());
        assert!(by_key("").is_none());
    }

    #[test]
    fn override_replaces_default_within_range() {
        let o = ParamOverrides::from_pairs([("IB_F_USMAMOUNT_DEFAULT", "240")]);
        let (min, max, default) = o.apply(by_key("USMAmount").unwrap());
        assert_eq!((min, max, default), (0.0, 500.0, 240.0));
        // Unrelated keys keep the static numbers.
        let (_, _, untouched) = o.apply(by_key("Exposure").unwrap());
        assert_eq!(untouched, 0.0);
    }

    #[test]
    fn override_default_is_clamped_and_unknown_keys_ignored() {
        let o = ParamOverrides::from_pairs([
            ("IB_F_USMAMOUNT_DEFAULT", "9999"),
            ("IB_F_UNKNOWN_DEFAULT", "5"),
            ("IB_F_CONTRAST_DEFAULT", "not-a-number"),
        ]);
        let (_, max, default) = o.apply(by_key("USMAmount").unwrap());
        assert_eq!((max, default), (500.0, 500.0));
        assert_eq!(o.apply(by_key("Contrast").unwrap()).2, 0.0);
    }

    #[test]
    fn range_override_wins_and_stays_valid() {
        let mut o = ParamOverrides::default();
        assert!(o.is_empty());
        o.set_range("Exposure", -5.0, 5.0);
        assert_eq!(o.apply(by_key("Exposure").unwrap()), (-5.0, 5.0, 0.0));
        // An inverted range is rejected, leaving the static one.
        o.set_range("Contrast", 50.0, -50.0);
        assert_eq!(o.apply(by_key("Contrast").unwrap()), (-50.0, 50.0, 0.0));
    }

    #[test]
    fn reads_info_block_lines() {
        let block = b"IB_F_KELVIN_DEFAULT 0\nIB_F_EXPOSURE_DEFAULT=0.35\njunk\n";
        let o = ParamOverrides::from_raw_info_block(block);
        assert_eq!(o.apply(by_key("Exposure").unwrap()).2, 0.35);
        assert!(o.is_empty() == false);
    }

    /// The report says 41 (§2.1); the decompiled dictionary literal has 40 rows.
    #[test]
    fn variant_range_table_matches_the_decompiled_dictionary() {
        assert_eq!(VARIANT_RANGES.len(), 40);
        let mut props = std::collections::HashSet::new();
        for (property, info_block_key) in VARIANT_RANGES {
            assert!(info_block_key.starts_with("IB_"), "{info_block_key}");
            assert!(props.insert(*property), "duplicate {property}");
        }
        assert_eq!(range_key("Exposure"), Some("IB_F_EXPOSURE"));
        assert_eq!(range_key("NoSuchProperty"), None);
    }

    #[test]
    fn start_and_end_pairs_set_an_asymmetric_range() {
        let o = ParamOverrides::from_pairs([
            ("IB_F_EXPOSURE_START", "-4"),
            ("IB_F_EXPOSURE_END", "4"),
            ("IB_F_USMAMOUNT_START", "0"),
            ("IB_F_USMAMOUNT_END", "1000"),
        ]);
        assert_eq!(o.apply(by_key("Exposure").unwrap()), (-4.0, 4.0, 0.0));
        assert_eq!(o.apply(by_key("USMAmount").unwrap()), (0.0, 1000.0, 160.0));
        // A lone `_START` is not a range.
        let o = ParamOverrides::from_pairs([("IB_F_CONTRAST_START", "-70")]);
        assert_eq!(o.apply(by_key("Contrast").unwrap()), (-50.0, 50.0, 0.0));
    }

    /// §2.1: a single-valued key means `[-v, +v]`.
    #[test]
    fn single_valued_key_is_a_symmetric_range() {
        let o = ParamOverrides::from_pairs([("IB_F_MOIRE_AMOUNT", "50")]);
        assert_eq!(o.apply(by_key("MoireAmount").unwrap()), (-50.0, 50.0, 0.0));
        let o = ParamOverrides::from_pairs([("IB_F_EXPOSURE_LOCAL_START", "-9")]);
        assert_eq!(o.apply(by_key("Exposure").unwrap()), (-2.5, 2.5, 0.0));
    }

    #[test]
    fn step_and_decimals_carry_the_proven_values() {
        let m = by_key("Exposure").unwrap();
        assert_eq!((m.step, m.decimals), (0.1, 2));
        let m = by_key("USMRadius").unwrap();
        assert_eq!((m.step, m.decimals), (0.1, 1));
        let m = by_key("WhiteBalanceTemperature").unwrap();
        assert_eq!((m.step, m.decimals), (25.0, 0));
        // No evidence: kept at 0 and commented `未证明`.
        let m = by_key("MoireAmount").unwrap();
        assert_eq!((m.step, m.decimals), (0.0, 0));
    }
}
