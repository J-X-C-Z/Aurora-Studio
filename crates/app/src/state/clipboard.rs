//! Selective copy and apply of adjustments, per tool, as in Capture One's
//! adjustments clipboard.

use lumen_engine::AdjustmentSet;

use crate::i18n::t;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ClipGroup {
    Base,
    WhiteBalance,
    Exposure,
    Hdr,
    Levels,
    Curves,
    Clarity,
    Dehaze,
    Vignetting,
    ColorBalance,
    ColorEditor,
    BlackWhite,
    Sharpening,
    NoiseReduction,
    FilmGrain,
    Moire,
    Lens,
    Rotation,
    Crop,
    Keystone,
    Spots,
    Layers,
    Style,
}

impl ClipGroup {
    pub const ALL: [ClipGroup; 23] = [
        ClipGroup::Base,
        ClipGroup::WhiteBalance,
        ClipGroup::Exposure,
        ClipGroup::Hdr,
        ClipGroup::Levels,
        ClipGroup::Curves,
        ClipGroup::Clarity,
        ClipGroup::Dehaze,
        ClipGroup::Vignetting,
        ClipGroup::ColorBalance,
        ClipGroup::ColorEditor,
        ClipGroup::BlackWhite,
        ClipGroup::Sharpening,
        ClipGroup::NoiseReduction,
        ClipGroup::FilmGrain,
        ClipGroup::Moire,
        ClipGroup::Lens,
        ClipGroup::Rotation,
        ClipGroup::Crop,
        ClipGroup::Keystone,
        ClipGroup::Spots,
        ClipGroup::Layers,
        ClipGroup::Style,
    ];

    /// Stable name for settings files.
    pub fn id(self) -> String {
        format!("{self:?}")
    }

    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|g| g.id() == id)
    }

    /// Groups that describe composition rather than look; off by default when applying.
    pub fn is_composition(self) -> bool {
        matches!(self, ClipGroup::Rotation | ClipGroup::Crop | ClipGroup::Keystone | ClipGroup::Spots)
    }

    pub fn label(self) -> &'static str {
        match self {
            ClipGroup::Base => t("Base Characteristics", "基础特性"),
            ClipGroup::WhiteBalance => t("White Balance", "白平衡"),
            ClipGroup::Exposure => t("Exposure", "曝光"),
            ClipGroup::Hdr => t("High Dynamic Range", "高动态范围"),
            ClipGroup::Levels => t("Levels", "色阶"),
            ClipGroup::Curves => t("Curve", "曲线"),
            ClipGroup::Clarity => t("Clarity", "清晰度"),
            ClipGroup::Dehaze => t("Dehaze", "去雾"),
            ClipGroup::Vignetting => t("Vignetting", "暗角"),
            ClipGroup::ColorBalance => t("Color Balance", "色彩平衡"),
            ClipGroup::ColorEditor => t("Color Editor", "色彩编辑器"),
            ClipGroup::BlackWhite => t("Black & White", "黑白"),
            ClipGroup::Sharpening => t("Sharpening", "锐化"),
            ClipGroup::NoiseReduction => t("Noise Reduction", "降噪"),
            ClipGroup::FilmGrain => t("Film Grain", "胶片颗粒"),
            ClipGroup::Moire => t("Moiré", "摩尔纹"),
            ClipGroup::Lens => t("Lens Correction", "镜头校正"),
            ClipGroup::Rotation => t("Rotation & Flip", "旋转与翻转"),
            ClipGroup::Crop => t("Crop", "裁剪"),
            ClipGroup::Keystone => t("Keystone", "梯形校正"),
            ClipGroup::Spots => t("Spot Removal", "污点去除"),
            ClipGroup::Layers => t("Layers", "图层"),
            ClipGroup::Style => t("Style", "风格"),
        }
    }

    /// Whether `a` differs from the defaults in this group.
    pub fn is_set(self, a: &AdjustmentSet) -> bool {
        let d = AdjustmentSet::default();
        let mut probe = d.clone();
        self.copy(&mut probe, a);
        probe != d
    }

    /// Copies this group's values from `src` into `dst`.
    pub fn copy(self, dst: &mut AdjustmentSet, src: &AdjustmentSet) {
        match self {
            ClipGroup::Base => dst.base = src.base.clone(),
            ClipGroup::WhiteBalance => dst.white_balance = src.white_balance.clone(),
            ClipGroup::Exposure => dst.exposure = src.exposure.clone(),
            ClipGroup::Hdr => dst.hdr = src.hdr.clone(),
            ClipGroup::Levels => dst.levels = src.levels.clone(),
            ClipGroup::Curves => dst.curves = src.curves.clone(),
            ClipGroup::Clarity => dst.clarity = src.clarity.clone(),
            ClipGroup::Dehaze => dst.dehaze = src.dehaze.clone(),
            ClipGroup::Vignetting => dst.vignetting = src.vignetting.clone(),
            ClipGroup::ColorBalance => dst.color_balance = src.color_balance.clone(),
            ClipGroup::ColorEditor => dst.color_editor = src.color_editor.clone(),
            ClipGroup::BlackWhite => dst.black_white = src.black_white.clone(),
            ClipGroup::Sharpening => dst.sharpening = src.sharpening.clone(),
            ClipGroup::NoiseReduction => dst.noise_reduction = src.noise_reduction.clone(),
            ClipGroup::FilmGrain => dst.film_grain = src.film_grain.clone(),
            ClipGroup::Moire => dst.moire = src.moire.clone(),
            ClipGroup::Lens => dst.lens = src.lens.clone(),
            ClipGroup::Rotation => {
                dst.geometry.rotation = src.geometry.rotation;
                dst.geometry.orientation = src.geometry.orientation;
                dst.geometry.flip_horizontal = src.geometry.flip_horizontal;
                dst.geometry.flip_vertical = src.geometry.flip_vertical;
            }
            ClipGroup::Crop => {
                dst.geometry.crop = src.geometry.crop;
                dst.geometry.aspect = src.geometry.aspect;
                dst.geometry.portrait = src.geometry.portrait;
            }
            ClipGroup::Keystone => dst.geometry.keystone = src.geometry.keystone.clone(),
            ClipGroup::Spots => dst.spots = src.spots.clone(),
            ClipGroup::Layers => {
                dst.layers = src.layers.clone();
                dst.nodes = src.nodes.clone();
            }
            ClipGroup::Style => dst.style = src.style.clone(),
        }
    }
}
