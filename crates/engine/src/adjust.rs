//! The complete, serializable description of every adjustment a variant carries.
//!
//! Every value is in the unit the interface shows, so an engine and the interface
//! agree without conversion tables. Ranges are documented on each field; the
//! interface clamps to them before handing a set to the engine.

use serde::{Deserialize, Serialize};

/// All adjustments applied to one image variant: the background layer's
/// global settings plus the local adjustment layers stacked above it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AdjustmentSet {
    pub base: BaseCharacteristics,
    pub white_balance: WhiteBalance,
    pub exposure: Exposure,
    pub hdr: HighDynamicRange,
    pub levels: Levels,
    pub curves: Curves,
    pub clarity: Clarity,
    pub dehaze: Dehaze,
    pub vignetting: Vignetting,
    pub color_balance: ColorBalance,
    /// Capture One's `ColorBalance*` RGB multipliers, per tonal zone. Kept
    /// beside the wheels so older documents (no field) read as `None`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_balance_gains: Option<ColorBalanceGains>,
    pub color_editor: ColorEditor,
    pub black_white: BlackWhite,
    pub sharpening: Sharpening,
    pub noise_reduction: NoiseReduction,
    pub film_grain: FilmGrain,
    pub moire: Moire,
    pub lens: LensCorrection,
    pub geometry: Geometry,
    pub spots: Vec<Spot>,
    /// Capture One Retouch Faces: one entry per `ZRETOUCHINGLAYER` row, each
    /// carrying its own masks. Empty for every image that has none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub retouch: Vec<RetouchFace>,
    pub layers: Vec<crate::Layer>,
    /// Node editor topology; absent in older saved adjustments.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nodes: Option<crate::NodeGraph>,
    /// Style applied last, by name; the engine owns what a style contains.
    pub style: Option<String>,
    /// Contract parameters with no typed field yet, keyed by C1 property key
    /// (`BlemishRemovalAmount`, see `param_meta`) or portrait parameter id
    /// (`pc:21093`, see `docs/frontend-contract/portrait-algorithms.json`).
    /// Engines ignore keys they do not render.
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub extra: std::collections::BTreeMap<String, f32>,
    /// Ephemeral producer bytes for this exact render job. Never persisted.
    #[serde(skip)]
    pub portrait_producer_inputs:
        Option<std::sync::Arc<crate::portrait_stage::ProducerInputs>>,
}

impl AdjustmentSet {
    /// True when nothing differs from a freshly imported image.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

/// One Capture One retouch face: the sliders of one `ZRETOUCHINGLAYER` row and
/// the masks that place them.
///
/// Amounts are the stored slider values in `0..=100` (EyeBright offsets may be
/// negative). `impact` is the row's `ZOPACITY`, which C1 multiplies into every
/// operation's amount; 100 is the variant default. Masks are keyed by C1's
/// `IC_RetouchingMaskType`: 1 SkinEvening, 2 Sculpting, 8 Teeth, 4 L.Iris,
/// 5 L.Sclera, 6 R.Iris, 7 R.Sclera.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RetouchFace {
    /// Face rectangle `[x, y, w, h]`, normalized in the oriented, uncropped
    /// image. Only its long side is used: it scales the skin-evening radius.
    pub face: [f32; 4],
    /// `ZOPACITY` in percent; 100 when the row stores no override.
    pub impact: f32,
    pub skin_amount: f32,
    /// Signed, `-100..=100`: negative adds texture, positive smooths.
    pub skin_texture: f32,
    pub sculpt: f32,
    pub teeth_brightness: f32,
    pub teeth_desaturation: f32,
    pub iris_left: f32,
    pub iris_right: f32,
    pub iris_both: f32,
    pub sclera_left: f32,
    pub sclera_right: f32,
    pub sclera_both: f32,
    #[serde(skip_serializing_if = "std::collections::BTreeMap::is_empty")]
    pub masks: std::collections::BTreeMap<u8, crate::RasterMask>,
}

impl Default for RetouchFace {
    fn default() -> Self {
        Self {
            face: [0.0; 4],
            impact: 100.0,
            skin_amount: 0.0,
            skin_texture: 0.0,
            sculpt: 0.0,
            teeth_brightness: 0.0,
            teeth_desaturation: 0.0,
            iris_left: 0.0,
            iris_right: 0.0,
            iris_both: 0.0,
            sclera_left: 0.0,
            sclera_right: 0.0,
            sclera_both: 0.0,
            masks: std::collections::BTreeMap::new(),
        }
    }
}

// ───────────────────────────── Base characteristics ─────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BaseCharacteristics {
    /// Camera input profile name; `None` means the engine's profile for the camera.
    pub icc_profile: Option<String>,
    pub curve: ToneCurvePreset,
    /// Capture One Base Characteristics "Mode" (`TechnicalProcessingMethod`).
    /// This is a processing branch, not a tone curve.
    #[serde(default)]
    pub processing_method: TechnicalProcessingMethod,
    /// Capture One `.fcrv` file name (`ZFILMCURVE`), applied to linear values
    /// before gamma. Keyed on the file name: the curve's embedded name does not
    /// match its content in Capture One's own files.
    pub film_curve: Option<String>,
    pub engine_version: String,
}

impl Default for BaseCharacteristics {
    fn default() -> Self {
        Self {
            icc_profile: None,
            curve: ToneCurvePreset::Auto,
            processing_method: TechnicalProcessingMethod::Photography,
            film_curve: None,
            engine_version: "Portrait 1".into(),
        }
    }
}

/// Capture One's Base Characteristics mode. Values match
/// `P1.AppCore.TechnicalProcessingMethod`; 0 is ordinary photography.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TechnicalProcessingMethod {
    #[default]
    Photography = 0,
    Scientific = 1,
    ReproFilmNegatives = 2,
    ReproFilmPositives = 3,
    StandardFilmNegatives = 4,
}

impl TechnicalProcessingMethod {
    pub const ALL: [Self; 5] = [
        Self::Photography,
        Self::Scientific,
        Self::ReproFilmNegatives,
        Self::ReproFilmPositives,
        Self::StandardFilmNegatives,
    ];

    pub fn from_code(code: i32) -> Self {
        match code {
            1 => Self::Scientific,
            2 => Self::ReproFilmNegatives,
            3 => Self::ReproFilmPositives,
            4 => Self::StandardFilmNegatives,
            _ => Self::Photography,
        }
    }

    pub fn to_c1(self) -> i32 {
        match self {
            Self::Photography => 0,
            Self::Scientific => 1,
            Self::ReproFilmNegatives => 2,
            Self::ReproFilmPositives => 3,
            Self::StandardFilmNegatives => 4,
        }
    }

    /// C1 and ImageCore share the same 0..4 numbering. The managed bridge enum
    /// (`IC_TechnicalProcessingMethod`) only names 0..3, but the value crosses
    /// the boundary as a plain int, and the chain gates on 4 directly
    /// (`0x183BC19` accepts 2|4, `0x16E85C0` requires 4), so do not fold.
    pub fn imgcore_value(self) -> i32 {
        match self {
            Self::Photography => 0,
            Self::Scientific => 1,
            Self::ReproFilmNegatives => 2,
            Self::ReproFilmPositives => 3,
            Self::StandardFilmNegatives => 4,
        }
    }

    pub fn is_film_negative(self) -> bool {
        matches!(self, Self::ReproFilmNegatives | Self::StandardFilmNegatives)
    }

    /// The negative branch inserts `FilmNegativeConversion`/`ClampNegative` and
    /// skips the normal exposure/levels pair before `ArtEffects`.
    pub fn inserts_film_negative_conversion(self) -> bool {
        self.is_film_negative()
    }

    /// 1/2/3 insert `ColorBalance` unconditionally; 4 uses
    /// `ColorBalanceFilmNegative`; 0 only when the variant flag is set.
    pub fn forces_color_balance(self) -> bool {
        matches!(
            self,
            Self::Scientific | Self::ReproFilmNegatives | Self::ReproFilmPositives
        )
    }

    /// The Base Characteristics combo in this build offers 0/3/2/4, never 1.
    pub fn offered_in_base_characteristics(self) -> bool {
        !matches!(self, Self::Scientific)
    }
}

/// Numeric C1 engine generation, stored at `settings+0x10C90`. The `.costyle`
/// attribute `<SL Engine="...">` is the same counter: `0x320` is 800, the value
/// an absent attribute stands for.
///
/// Every constant is a literal `cmp dword ptr [<settings> + 0x10C90], imm` in
/// `ImgCoreDll.dll` (image base `0x180000000`):
///   `0x320` 800  - RVA `0x1839A15`
///   `0x44C` 1100 - RVA `0x183BD40` (negative chain), also `0x154E500`, `0x15508C8`
///   `0x514` 1300 - RVA `0x183C6C8`, `0x16E8480`, `0x1868559`
/// 0x1100 is *not* a threshold: it is the `+0x11000` field offset the chain
/// builder also reads. The three real boundaries are the ones above.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum EngineGeneration {
    /// `< 0x44C` (1100): the legacy standalone `ColorBalance` op stays in.
    #[default]
    Before1100,
    /// `0x44C ..= 0x514` (1100 through 1300, both inclusive).
    V1100ToV1300,
    /// `> 0x514` (1300).
    After1300,
}

impl EngineGeneration {
    pub const RAW_800: u32 = 0x320;
    pub const RAW_1100: u32 = 0x44C;
    pub const RAW_1300: u32 = 0x514;

    pub fn from_raw(version: u32) -> Self {
        if version > Self::RAW_1300 {
            Self::After1300
        } else if version >= Self::RAW_1100 {
            Self::V1100ToV1300
        } else {
            Self::Before1100
        }
    }

    /// `ColorBalance` is appended when the document predates 1100.
    pub fn keeps_legacy_color_balance(self) -> bool {
        matches!(self, Self::Before1100)
    }

    /// `ColorBalanceFilmNegative` stops being built after 1300 (`0x16E85C0`).
    pub fn builds_color_balance_film_negative(self) -> bool {
        !matches!(self, Self::After1300)
    }
}

/// The negative-only stages Capture One's chain builder appends. Mirrors the
/// builder at RVA `0x183BB80`; per-stage conditions are in
/// `out/negative-chain.tsv`.
///
/// Parameter provenance (`out/negative-chain.tsv`, column `params_read`):
/// `FilmNegativeConversion` reads three channel levels (`settings+0x8A8/AC/B0`,
/// the film-base triple), a density scalar (`+0x8FC`) and the white-balance
/// gains; `ColorBalanceFilmNegative` reads 16 per-layer gain triples and 16
/// mask planes; `UndoFilmcurve` reads a LUT vector.
///
/// **None of those is a managed column.** `ZVARIANTLAYER` has no film-base,
/// density, mask or LUT column, `ZDOCUMENTSETTING` only holds capture/UI keys,
/// and no installed `.costyle` carries such a key, so the parameters come from
/// live image state at render time. The field below therefore stays `None`
/// whenever the answer depends on them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NegativeChain {
    /// `FilmNegativeConversion`: reciprocal inversion with base subtraction.
    pub film_negative_conversion: bool,
    /// `ClampNegative`: `max(x, 0)`.
    pub clamp_negative: bool,
    /// `ColorBalanceFilmNegative`: per-layer masked RGB gain. `None` means the
    /// catalog cannot decide it: `0x16E85C0` also requires non-neutral gains.
    pub color_balance_film_negative: Option<bool>,
}

impl Default for NegativeChain {
    fn default() -> Self {
        Self {
            film_negative_conversion: false,
            clamp_negative: false,
            color_balance_film_negative: Some(false),
        }
    }
}

impl NegativeChain {
    /// True when the block cannot change a pixel: both fixed stages are off and
    /// the colour-balance stage is off or undecidable from catalog data.
    pub fn is_inert(self) -> bool {
        !self.film_negative_conversion
            && !self.clamp_negative
            && !matches!(self.color_balance_film_negative, Some(true))
    }
}

impl BaseCharacteristics {
    /// The `settings+0x11008` switch. Neither `.cop` nor the catalog carries a
    /// column for it, so it is derived from the mode; the binary only ever
    /// clears it (`0x1855D06`), never fills it from a stored value.
    pub fn negative_switch(&self) -> bool {
        self.processing_method.inserts_film_negative_conversion()
    }

    /// Stage gate for the negative block. `generation` is the document's engine
    /// version. `color_balance_film_negative` is `None` for method 4 inside the
    /// 1300 window because `0x16E85C0` also tests the per-layer gains, which
    /// have no catalog column; only the renderer can settle it.
    pub fn negative_chain(&self, generation: EngineGeneration) -> NegativeChain {
        let block = self.negative_switch();
        let color_balance = if block
            && self.processing_method == TechnicalProcessingMethod::StandardFilmNegatives
            && generation.builds_color_balance_film_negative()
        {
            None
        } else {
            Some(false)
        };
        NegativeChain {
            film_negative_conversion: block,
            clamp_negative: block,
            color_balance_film_negative: color_balance,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToneCurvePreset {
    #[default]
    Auto,
    DngStandard,
    DngToneMapped,
    FilmStandard,
    FilmExtraShadow,
    FilmHighContrast,
    Linear,
}

impl ToneCurvePreset {
    pub const ALL: [Self; 7] = [
        Self::Auto,
        Self::DngStandard,
        Self::DngToneMapped,
        Self::FilmStandard,
        Self::FilmExtraShadow,
        Self::FilmHighContrast,
        Self::Linear,
    ];

    /// The family behind a `.fcrv` filename. `Linear Scientific` is a Linear
    /// curve; DNG files use their own names.
    pub fn from_film_curve(name: &str) -> Self {
        if name.contains("DNG Tone Mapped") {
            Self::DngToneMapped
        } else if name.contains("DNG Standard") {
            Self::DngStandard
        } else if name.contains("ExtraShadow") {
            Self::FilmExtraShadow
        } else if name.contains("HighContrast") {
            Self::FilmHighContrast
        } else if name.contains("Film Standard") || name.contains("FilmStandard") {
            Self::FilmStandard
        } else if name.contains("Linear") {
            Self::Linear
        } else {
            Self::Auto
        }
    }
}

// ───────────────────────────── White balance ─────────────────────────────

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WhiteBalanceMode {
    #[default]
    Shot,
    Auto,
    Daylight,
    Cloudy,
    Shade,
    Tungsten,
    Fluorescent,
    Flash,
    Custom,
}

impl WhiteBalanceMode {
    pub const ALL: [Self; 9] = [
        Self::Shot,
        Self::Auto,
        Self::Daylight,
        Self::Cloudy,
        Self::Shade,
        Self::Tungsten,
        Self::Fluorescent,
        Self::Flash,
        Self::Custom,
    ];

    /// Nominal (kelvin, tint) for preset modes; `None` for modes the engine measures.
    pub fn preset(self) -> Option<(f32, f32)> {
        match self {
            Self::Daylight => Some((5200.0, 0.0)),
            Self::Cloudy => Some((6000.0, 0.0)),
            Self::Shade => Some((7000.0, 0.0)),
            Self::Tungsten => Some((3200.0, 0.0)),
            Self::Fluorescent => Some((4000.0, 0.0)),
            Self::Flash => Some((5500.0, 0.0)),
            Self::Shot | Self::Auto | Self::Custom => None,
        }
    }

    /// Capture One's preset names, ignoring case, spaces and separators.
    pub fn from_name(name: &str) -> Option<Self> {
        let key: String = name.chars().filter(char::is_ascii_alphanumeric).collect();
        Some(match key.to_ascii_lowercase().as_str() {
            "asshot" | "shot" => Self::Shot,
            "auto" => Self::Auto,
            "daylight" => Self::Daylight,
            "cloudy" => Self::Cloudy,
            "shade" => Self::Shade,
            "tungsten" => Self::Tungsten,
            "fluorescent" => Self::Fluorescent,
            "flash" => Self::Flash,
            "custom" => Self::Custom,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WhiteBalance {
    pub mode: WhiteBalanceMode,
    /// 2000 ..= 50000 K.
    pub kelvin: f32,
    /// -50 ..= 50.
    pub tint: f32,
    /// Skin tone target picked from the skin-tone presets; engine-defined name.
    pub skin_tone: Option<String>,
    /// Exact C1 `ZWHITEBALANCE` gains when the import had them. Rendering uses
    /// these verbatim; K/T is then only the display pair. Any K/T edit clears
    /// this so the sliders win again.
    pub gains: Option<[f32; 3]>,
}

impl Default for WhiteBalance {
    fn default() -> Self {
        Self {
            mode: WhiteBalanceMode::Shot,
            kelvin: 5600.0,
            tint: 0.0,
            skin_tone: None,
            gains: None,
        }
    }
}

/// Gray-world estimate from the frame mean, the same warm/green split the
/// white-balance picker uses on a sampled pixel.
pub fn auto_from_mean(rgb: [f32; 3]) -> (f32, f32) {
    let sum = (rgb[0] + rgb[1] + rgb[2]).max(1e-3);
    let warm = (rgb[0] - rgb[2]) / sum;
    let green = (rgb[1] - (rgb[0] + rgb[2]) / 2.0) / sum;
    ((5600.0 - warm * 9000.0).clamp(2000.0, 15000.0), (green * 120.0).clamp(-50.0, 50.0))
}

impl WhiteBalance {
    /// Kelvin and tint this adjustment renders with. `Shot` inverts the frame's
    /// MakerNote gains (falling back to `Auto` when the import had none), `Auto`
    /// estimates a gray world from `mean_rgb`, presets use their nominal pair,
    /// and `Custom` keeps the sliders.
    pub fn resolved(&self, exif_gains: Option<[f32; 3]>, mean_rgb: [f32; 3]) -> (f32, f32) {
        match self.mode {
            WhiteBalanceMode::Shot => self
                .gains
                .filter(|g| g.iter().all(|v| v.is_finite() && *v > 0.0))
                .filter(|g| *g != [1.0, 1.0, 1.0])
                .or(exif_gains.filter(|g| g.iter().all(|v| v.is_finite() && *v > 0.0)))
                .map(crate::c1_model::gains_to_kelvin_tint)
                .unwrap_or_else(|| auto_from_mean(mean_rgb)),
            WhiteBalanceMode::Auto => auto_from_mean(mean_rgb),
            WhiteBalanceMode::Custom => (self.kelvin, self.tint),
            mode => mode.preset().unwrap_or((self.kelvin, self.tint)),
        }
    }
}

// ───────────────────────────── Exposure / HDR / Levels ─────────────────────────────

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Exposure {
    /// -4 ..= 4 EV.
    pub exposure: f32,
    /// -50 ..= 50.
    pub contrast: f32,
    /// -50 ..= 50.
    pub brightness: f32,
    /// -100 ..= 100.
    pub saturation: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HighDynamicRange {
    /// -100 ..= 100 for all four.
    pub highlights: f32,
    pub shadows: f32,
    pub whites: f32,
    pub blacks: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToneChannel {
    #[default]
    Rgb,
    Luma,
    Red,
    Green,
    Blue,
}

impl ToneChannel {
    pub const LEVELS: [Self; 4] = [Self::Rgb, Self::Red, Self::Green, Self::Blue];
    pub const CURVES: [Self; 5] = [Self::Rgb, Self::Luma, Self::Red, Self::Green, Self::Blue];
}

/// One channel's levels, all values normalized to `0.0 ..= 1.0` except `gamma`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LevelsChannel {
    pub input_black: f32,
    pub input_white: f32,
    /// Midtone gamma, 0.1 ..= 10, 1.0 neutral. The input that maps to mid-output
    /// is `0.5^gamma`, so gamma above 1 lifts midtones.
    pub gamma: f32,
    pub output_black: f32,
    pub output_white: f32,
}

impl Default for LevelsChannel {
    fn default() -> Self {
        Self { input_black: 0.0, input_white: 1.0, gamma: 1.0, output_black: 0.0, output_white: 1.0 }
    }
}

impl LevelsChannel {
    /// Maps a display-referred value through input points, midtone gamma, then
    /// output points. Both ends are clipped to the output range.
    pub fn map(&self, v: f32) -> f32 {
        self.map_output(self.map_input(v))
    }

    /// Input points and midtone gamma only; the output pair is left at identity.
    pub fn map_input(&self, v: f32) -> f32 {
        let span = (self.input_white - self.input_black).max(1e-3);
        let t = ((v - self.input_black) / span).clamp(0.0, 1.0);
        t.powf(1.0 / self.gamma.clamp(0.1, 10.0))
    }

    /// Output points only. Capture One's target levels are a separate stage
    /// after the gradation curves, so the pipeline applies this half there.
    pub fn map_output(&self, t: f32) -> f32 {
        self.output_black + t * (self.output_white - self.output_black)
    }
}

/// Capture One's Levels midtone is a signed marker position in `-1.0 ..= 1.0`,
/// not a gamma: 0 is neutral and the marker sits at `(1 - m) / 2` of the input
/// range (see `LevelsToolViewModel.MidtoneForSlider`). That point maps to 50%
/// output, so `gamma = -log2((1 - m) / 2) = 1 - log2(1 - m)`.
pub fn gamma_from_c1_midtone(midtone: f32) -> f32 {
    let position = ((1.0 - midtone) * 0.5).clamp(1e-4, 1.0 - 1e-4);
    (-position.log2()).clamp(0.1, 10.0)
}

/// Inverse of [`gamma_from_c1_midtone`], for writing the UI value back.
pub fn c1_midtone_from_gamma(gamma: f32) -> f32 {
    1.0 - 2f32.powf(1.0 - gamma.clamp(0.1, 10.0))
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Levels {
    pub rgb: LevelsChannel,
    pub red: LevelsChannel,
    pub green: LevelsChannel,
    pub blue: LevelsChannel,
}

impl Levels {
    /// Applies the per-channel levels first and the RGB master last, the order
    /// Capture One and Photoshop use for composite vs. single-channel edits.
    pub fn apply(&self, rgb: [f32; 3]) -> [f32; 3] {
        self.apply_output(self.apply_input(rgb))
    }

    /// The input points and midtone gamma, i.e. everything before the curves.
    pub fn apply_input(&self, rgb: [f32; 3]) -> [f32; 3] {
        [
            self.rgb.map_input(self.red.map_input(rgb[0])),
            self.rgb.map_input(self.green.map_input(rgb[1])),
            self.rgb.map_input(self.blue.map_input(rgb[2])),
        ]
    }

    /// Capture One's target levels (stage 18, after `GradationCurves`):
    /// `ApplyTargetLevelsCommon` reads only the `.x` scalar of `targetShadow`
    /// and `targetHighlight` and applies one RGB-master affine,
    /// `out = in * (highlight - shadow) + shadow`. The stored r/g/b
    /// components have no read point in the native op.
    pub fn apply_output(&self, rgb: [f32; 3]) -> [f32; 3] {
        let (shadow, highlight) = (self.rgb.output_black, self.rgb.output_white);
        let gain = highlight - shadow;
        [
            rgb[0] * gain + shadow,
            rgb[1] * gain + shadow,
            rgb[2] * gain + shadow,
        ]
    }

    pub fn channel(&self, channel: ToneChannel) -> &LevelsChannel {
        match channel {
            ToneChannel::Red => &self.red,
            ToneChannel::Green => &self.green,
            ToneChannel::Blue => &self.blue,
            ToneChannel::Rgb | ToneChannel::Luma => &self.rgb,
        }
    }

    pub fn channel_mut(&mut self, channel: ToneChannel) -> &mut LevelsChannel {
        match channel {
            ToneChannel::Red => &mut self.red,
            ToneChannel::Green => &mut self.green,
            ToneChannel::Blue => &mut self.blue,
            ToneChannel::Rgb | ToneChannel::Luma => &mut self.rgb,
        }
    }
}

// ───────────────────────────── Curves ─────────────────────────────

/// A control point in normalized `0.0 ..= 1.0` input/output space.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CurvePoint {
    pub x: f32,
    pub y: f32,
}

impl CurvePoint {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// A tone curve through control points in normalized input/output space.
///
/// Capture One stores `ZGRADATIONCURVE*` as `in,out;in,out`, both components
/// normalized `0.0 ..= 1.0` (`c_zeroToOneCurve` runs `PointF(0,0)` to
/// `PointF(1,1)`). Points are `(input, output)`, not pixel coordinates.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Curve {
    pub points: Vec<CurvePoint>,
}

impl Default for Curve {
    fn default() -> Self {
        Self { points: vec![CurvePoint::new(0.0, 0.0), CurvePoint::new(1.0, 1.0)] }
    }
}

impl Curve {
    pub fn is_identity(&self) -> bool {
        self.normalized().iter().all(|p| (p.x - p.y).abs() < 1e-4)
    }

    /// The control points as the engine consumes them: finite, inside the unit
    /// square, sorted by input, duplicates removed, endpoints `(0,0)`/`(1,1)`
    /// filled in, and outputs forced non-decreasing. C1's Gradation Curve is
    /// monotone, so a point dragged below the previous one becomes a plateau
    /// instead of inverting tonal order.
    fn normalized(&self) -> Vec<CurvePoint> {
        let mut pts: Vec<CurvePoint> = self
            .points
            .iter()
            .filter(|p| p.x.is_finite() && p.y.is_finite())
            .map(|p| CurvePoint::new(p.x.clamp(0.0, 1.0), p.y.clamp(0.0, 1.0)))
            .collect();
        pts.sort_by(|a, b| a.x.total_cmp(&b.x));
        pts.dedup_by(|a, b| (a.x - b.x).abs() < 1e-6);
        if pts.first().is_none_or(|p| p.x > 1e-6) {
            pts.insert(0, CurvePoint::new(0.0, 0.0));
        }
        if pts.last().is_none_or(|p| p.x < 1.0 - 1e-6) {
            pts.push(CurvePoint::new(1.0, 1.0));
        }
        let mut prev = pts[0].y;
        for p in pts.iter_mut().skip(1) {
            p.y = p.y.max(prev);
            prev = p.y;
        }
        pts
    }

    /// Evaluates the curve with monotone cubic (Fritsch–Carlson) interpolation.
    /// The interface and the mock engine both use this, so what the curve editor
    /// draws is what the engine applies.
    pub fn eval(&self, x: f32) -> f32 {
        let pts = self.normalized();
        let p = pts.as_slice();
        if x <= p[0].x {
            return p[0].y;
        }
        if x >= p[p.len() - 1].x {
            return p[p.len() - 1].y;
        }
        let n = p.len();
        let mut delta = Vec::with_capacity(n - 1);
        for i in 0..n - 1 {
            let dx = (p[i + 1].x - p[i].x).max(1e-6);
            delta.push((p[i + 1].y - p[i].y) / dx);
        }
        let mut m = vec![0.0f32; n];
        m[0] = delta[0];
        m[n - 1] = delta[n - 2];
        for i in 1..n - 1 {
            m[i] = if delta[i - 1] * delta[i] <= 0.0 { 0.0 } else { (delta[i - 1] + delta[i]) / 2.0 };
        }
        for i in 0..n - 1 {
            if delta[i].abs() < 1e-6 {
                m[i] = 0.0;
                m[i + 1] = 0.0;
                continue;
            }
            let a = m[i] / delta[i];
            let b = m[i + 1] / delta[i];
            let s = a * a + b * b;
            if s > 9.0 {
                let t = 3.0 / s.sqrt();
                m[i] = t * a * delta[i];
                m[i + 1] = t * b * delta[i];
            }
        }
        let i = p.windows(2).position(|w| x >= w[0].x && x <= w[1].x).unwrap_or(n - 2);
        let h = (p[i + 1].x - p[i].x).max(1e-6);
        let t = (x - p[i].x) / h;
        let t2 = t * t;
        let t3 = t2 * t;
        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;
        (h00 * p[i].y + h10 * h * m[i] + h01 * p[i + 1].y + h11 * h * m[i + 1]).clamp(0.0, 1.0)
    }

    /// A 256-entry lookup table, piecewise-linearly interpolated from the
    /// normalized control points at `i / 255`.
    pub fn lut(&self) -> [f32; 256] {
        let p = self.normalized();
        let mut out = [0.0; 256];
        let mut seg = 0;
        for (i, v) in out.iter_mut().enumerate() {
            let x = i as f32 / 255.0;
            while seg + 2 < p.len() && x > p[seg + 1].x {
                seg += 1;
            }
            let a = p[seg];
            let b = p[seg + 1];
            let t = ((x - a.x) / (b.x - a.x).max(1e-6)).clamp(0.0, 1.0);
            *v = (a.y + (b.y - a.y) * t).clamp(0.0, 1.0);
        }
        out
    }
}

/// The five Gradation Curves. The engine applies them in this order:
/// `rgb` to all three channels, then `red`/`green`/`blue` to their channel,
/// then `luma` as a luminance ratio (hue and saturation stay put).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Curves {
    pub rgb: Curve,
    pub luma: Curve,
    pub red: Curve,
    pub green: Curve,
    pub blue: Curve,
}

impl Curves {
    pub fn channel(&self, channel: ToneChannel) -> &Curve {
        match channel {
            ToneChannel::Rgb => &self.rgb,
            ToneChannel::Luma => &self.luma,
            ToneChannel::Red => &self.red,
            ToneChannel::Green => &self.green,
            ToneChannel::Blue => &self.blue,
        }
    }

    pub fn channel_mut(&mut self, channel: ToneChannel) -> &mut Curve {
        match channel {
            ToneChannel::Rgb => &mut self.rgb,
            ToneChannel::Luma => &mut self.luma,
            ToneChannel::Red => &mut self.red,
            ToneChannel::Green => &mut self.green,
            ToneChannel::Blue => &mut self.blue,
        }
    }
}

// ───────────────────────────── Clarity / Dehaze / Vignetting ─────────────────────────────

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ClarityMethod {
    #[default]
    Natural,
    Punch,
    Neutral,
    Classic,
}

impl ClarityMethod {
    pub const ALL: [Self; 4] = [Self::Natural, Self::Punch, Self::Neutral, Self::Classic];

    /// `ZCLARITYMETHOD` stores the `IC_ClarityMethod` enum, whose order is not
    /// this enum's declaration order. 3 is Contrast in C1, shown as Natural.
    pub fn from_c1(value: i32) -> Self {
        match value {
            0 => Self::Neutral,
            1 => Self::Punch,
            2 => Self::Classic,
            _ => Self::Natural,
        }
    }

    /// The inverse of [`Self::from_c1`], for writing a C1 row back.
    pub fn to_c1(self) -> i32 {
        match self {
            Self::Neutral => 0,
            Self::Punch => 1,
            Self::Classic => 2,
            Self::Natural => 3,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Clarity {
    pub method: ClarityMethod,
    /// -100 ..= 100.
    pub clarity: f32,
    /// -100 ..= 100.
    pub structure: f32,
}

impl Clarity {
    /// Capture One's Clarity2012 amount uniform: `*0.012`, negative `*0.003`,
    /// method 3 negatives doubled, clamped at -1.
    pub fn c1_amount(&self) -> f32 {
        let mut a = self.clarity * 0.012;
        if a < 0.0 {
            a = self.clarity * 0.003;
            if matches!(self.method, ClarityMethod::Natural) {
                a *= 2.0;
            }
        }
        a.max(-1.0)
    }

    /// Capture One's Structure2012 amount uniform: negative `*0.01`, else `*0.02`.
    pub fn c1_structure(&self) -> f32 {
        let a = if self.structure < 0.0 { self.structure * 0.01 } else { self.structure * 0.02 };
        a.clamp(-1.0, 1.0)
    }
}

/// Clarity2012 pyramid radius for a positive amount.
pub fn c1_clarity_radius_big(width: u32, height: u32, zoom: f32) -> u32 {
    (0.125 * width.min(height) as f32 * zoom).floor().max(8.0) as u32
}

/// Clarity2012 radius for a negative amount without an auxiliary buffer.
pub fn c1_clarity_radius_small(width: u32, zoom: f32) -> u32 {
    (100.0 * width as f32 * zoom / 11976.0).floor().max(8.0) as u32
}

/// The low-pass runs on a 1/8-resolution copy.
pub fn c1_clarity_lowres_radius(radius: u32) -> u32 {
    radius / 8
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Dehaze {
    /// 0 ..= 100.
    pub amount: f32,
    /// Haze colour the engine removes, linear sRGB. `ZDEHAZECOLOR`, or the
    /// colour `c1_model` resolved for the auto (`0;0;0`) sentinel.
    pub shadow_tone: [f32; 3],
    /// `ZDEHAZELEVELLOW`: per-channel black point, 0 neutral.
    pub level_low: [f32; 3],
    /// `ZDEHAZELEVELHIGH`: per-channel white point, 1 neutral.
    pub level_high: [f32; 3],
}

impl Default for Dehaze {
    fn default() -> Self {
        Self {
            amount: 0.0,
            shadow_tone: [0.0; 3],
            level_low: [0.0; 3],
            level_high: [1.0; 3],
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VignetteMethod {
    #[default]
    EllipticOnCrop,
    Circular,
    CircularCrop,
}

impl VignetteMethod {
    pub const ALL: [Self; 3] = [Self::EllipticOnCrop, Self::Circular, Self::CircularCrop];
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Vignetting {
    /// -4 ..= 4 EV.
    pub amount: f32,
    pub method: VignetteMethod,
}

// ───────────────────────────── Color ─────────────────────────────

/// One color wheel: a hue angle, the distance from neutral, and a lightness offset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorWheel {
    /// 0 ..= 360 degrees.
    pub hue: f32,
    /// 0 ..= 1, the radius on the wheel.
    pub saturation: f32,
    /// -1 ..= 1.
    pub lightness: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorBalanceMode {
    #[default]
    ThreeWay,
    Master,
    Shadow,
    Midtone,
    Highlight,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorBalance {
    pub master: ColorWheel,
    pub shadow: ColorWheel,
    pub midtone: ColorWheel,
    pub highlight: ColorWheel,
}

/// Capture One's `ColorBalance*` triples: per-zone RGB multipliers, 1.0 neutral.
///
/// They are applied after the [`ColorWheel`] stage. The zone order matches the
/// `.copreset` keys and the `ZCOLORBALANCE*` columns: master, shadow, midtone,
/// highlight.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorBalanceGains {
    pub master: [f32; 3],
    pub shadow: [f32; 3],
    pub midtone: [f32; 3],
    pub highlight: [f32; 3],
}

impl Default for ColorBalanceGains {
    fn default() -> Self {
        Self {
            master: [1.0; 3],
            shadow: [1.0; 3],
            midtone: [1.0; 3],
            highlight: [1.0; 3],
        }
    }
}

impl ColorBalanceGains {
    pub fn is_neutral(&self) -> bool {
        *self == Self::default()
    }

    /// `0 = master, 1 = shadow, 2 = midtone, 3 = highlight`.
    pub fn zone(&self, index: usize) -> [f32; 3] {
        match index {
            0 => self.master,
            1 => self.shadow,
            2 => self.midtone,
            _ => self.highlight,
        }
    }

    pub fn set_zone(&mut self, index: usize, value: [f32; 3]) {
        match index {
            0 => self.master = value,
            1 => self.shadow = value,
            2 => self.midtone = value,
            _ => self.highlight = value,
        }
    }
}

/// The eight hue sectors of the basic color editor.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HueSector {
    Red,
    Orange,
    Yellow,
    Green,
    Cyan,
    Blue,
    Purple,
    Magenta,
}

impl HueSector {
    pub const ALL: [Self; 8] = [
        Self::Red,
        Self::Orange,
        Self::Yellow,
        Self::Green,
        Self::Cyan,
        Self::Blue,
        Self::Purple,
        Self::Magenta,
    ];

    /// Center hue in degrees.
    pub fn hue(self) -> f32 {
        match self {
            Self::Red => 0.0,
            Self::Orange => 30.0,
            Self::Yellow => 60.0,
            Self::Green => 120.0,
            Self::Cyan => 180.0,
            Self::Blue => 225.0,
            Self::Purple => 270.0,
            Self::Magenta => 315.0,
        }
    }

    /// The sector whose center hue is closest to `hue`.
    pub fn nearest(hue: f32) -> Self {
        let distance = |a: f32, b: f32| {
            let d = (a - b).rem_euclid(360.0);
            d.min(360.0 - d)
        };
        *Self::ALL
            .iter()
            .min_by(|a, b| distance(a.hue(), hue).total_cmp(&distance(b.hue(), hue)))
            .unwrap_or(&Self::Red)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HslShift {
    /// -30 ..= 30 degrees.
    pub hue: f32,
    /// -100 ..= 100.
    pub saturation: f32,
    /// -100 ..= 100.
    pub lightness: f32,
}

/// The hue/saturation band of one basic color-editor sector, decoded from a
/// `ZCOLORCORRECTIONS` group (`center`, hue/sat low+high, smoothness,
/// homogeneity, inverted, enabled).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorBand {
    /// Hue of the selected color, degrees.
    pub center_hue: f32,
    /// Saturation of the selected color, 0 ..= 1.
    pub center_saturation: f32,
    /// Value (HSV V) of the selected color, 0 ..= 1.
    pub center_value: f32,
    /// Band edges relative to [`Self::center_hue`], degrees; low <= 0 <= high.
    pub hue_low: f32,
    pub hue_high: f32,
    /// Band edges relative to the selected saturation, percentage points.
    pub sat_low: f32,
    pub sat_high: f32,
    /// Band edges on HSV V, absolute 0 ..= 1. C1 stores no value band, so the
    /// full range keeps imported bands byte-identical.
    pub value_low: f32,
    pub value_high: f32,
    /// Feather half-width of the band edge, degrees (C1: 1 ..= 29.1).
    pub smoothness: f32,
    /// Pull toward the selected hue/saturation/lightness, 0 ..= 100.
    pub homogeneity: [f32; 3],
    pub inverted: bool,
    pub enabled: bool,
}

impl Default for ColorBand {
    fn default() -> Self {
        Self {
            center_hue: 0.0,
            center_saturation: 1.0,
            center_value: 1.0,
            hue_low: -25.0,
            hue_high: 25.0,
            sat_low: -100.0,
            sat_high: 100.0,
            value_low: 0.0,
            value_high: 1.0,
            smoothness: 10.0,
            homogeneity: [0.0; 3],
            inverted: false,
            enabled: true,
        }
    }
}

/// A hue range picked in the advanced color editor.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorRange {
    pub id: u64,
    /// Center hue in degrees.
    pub hue: f32,
    /// Full width of the selected slice in degrees.
    pub width: f32,
    /// Feather in degrees on each side.
    pub smoothness: f32,
    /// Minimum chroma included, 0 ..= 1.
    pub min_saturation: f32,
    pub shift: HslShift,
    /// Skin-tone selections add a uniformity pass.
    pub uniformity: Option<SkinUniformity>,
}

impl Default for ColorRange {
    fn default() -> Self {
        Self {
            id: 0,
            hue: 30.0,
            width: 40.0,
            smoothness: 15.0,
            min_saturation: 0.05,
            shift: HslShift::default(),
            uniformity: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SkinUniformity {
    /// 0 ..= 100 for all three.
    pub hue: f32,
    pub saturation: f32,
    pub lightness: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ColorEditor {
    /// Basic mode: one shift per sector, indexed in [`HueSector::ALL`] order.
    pub basic: [HslShift; 8],
    /// Basic mode: the hue/saturation band each shift applies to.
    pub bands: [ColorBand; 8],
    /// The ninth C1 group, a correction that covers the whole image.
    pub global: ColorBand,
    /// HSL shift of the global group.
    pub global_shift: HslShift,
    /// Full band geometry for [`Self::advanced`] and [`Self::skin`] ranges,
    /// keyed by [`ColorRange::id`].
    pub advanced_bands: Vec<(u64, ColorBand)>,
    pub advanced: Vec<ColorRange>,
    pub skin: Vec<ColorRange>,
    /// C1's MaxSmooth groups (`kind=3`). C1 renders them with the basic
    /// `pie_smooth` window, but they cannot live in [`Self::basic`]: that is a
    /// fixed eight-slot array and a real layer already fills all eight slots.
    pub max_smooth: Vec<ColorRange>,
}

impl Default for ColorEditor {
    fn default() -> Self {
        Self {
            basic: [HslShift::default(); 8],
            bands: std::array::from_fn(|i| ColorBand {
                center_hue: HueSector::ALL[i].hue(),
                ..ColorBand::default()
            }),
            global: ColorBand { smoothness: 0.0, ..ColorBand::default() },
            global_shift: HslShift::default(),
            advanced_bands: Vec::new(),
            advanced: Vec::new(),
            skin: Vec::new(),
            max_smooth: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SplitTone {
    /// 0 ..= 360.
    pub hue: f32,
    /// 0 ..= 100.
    pub saturation: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BlackWhite {
    pub enabled: bool,
    /// Sensitivity per channel, -100 ..= 100: red, yellow, green, cyan, blue, magenta.
    pub sensitivity: [f32; 6],
    pub highlight_tone: SplitTone,
    pub shadow_tone: SplitTone,
}

impl BlackWhite {
    /// The six floats in C1's internal `s_bwColorSensitivityProperties` order
    /// (`settings+0x10b7c`): blue, yellow, cyan, green, red, magenta.
    pub fn c1_runtime_sensitivity(&self) -> [f32; 6] {
        let s = &self.sensitivity;
        [s[4], s[1], s[3], s[2], s[0], s[5]]
    }
}

// ───────────────────────────── Details ─────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sharpening {
    /// 0 ..= 1000.
    pub amount: f32,
    /// 0.2 ..= 2.5 px.
    pub radius: f32,
    /// 0 ..= 12.
    pub threshold: f32,
    /// 0 ..= 100.
    pub halo_suppression: f32,
}

impl Default for Sharpening {
    fn default() -> Self {
        // Capture One's default layer carries ZUSMAMOUNT=160, ZUSMRADIUS=0.8,
        // ZUSMTHRESHOLD=1.0 — a camera-matched baseline, not a user edit.
        Self { amount: 160.0, radius: 0.8, threshold: 1.0, halo_suppression: 0.0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct NoiseReduction {
    /// 0 ..= 100 for the first three.
    pub luminance: f32,
    pub details: f32,
    pub color: f32,
    /// 0 ..= 100.
    pub single_pixel: f32,
}

impl Default for NoiseReduction {
    fn default() -> Self {
        // ZNOISEREDUCTIONAMOUNT=50, ZNOISEDETAILSAMOUNT=50, ZAIDENOISEAMOUNT=50.
        Self { luminance: 50.0, details: 50.0, color: 50.0, single_pixel: 0.0 }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GrainKind {
    #[default]
    FineGrain,
    SilverRich,
    SoftGrain,
    CubicGrain,
    TabularGrain,
    HarshGrain,
}

impl GrainKind {
    pub const ALL: [Self; 6] = [
        Self::FineGrain,
        Self::SilverRich,
        Self::SoftGrain,
        Self::CubicGrain,
        Self::TabularGrain,
        Self::HarshGrain,
    ];

    /// C1 runs the erf Gaussian blur for Soft/Cubic/Tabular/Harsh only.
    /// FineGrain and SilverRich are per-pixel noise and never build that kernel.
    pub fn uses_erf_blur(self) -> bool {
        !matches!(self, Self::FineGrain | Self::SilverRich)
    }

    /// Blur-scale multiplier C1 applies on top of the shared sigma curve:
    /// Tabular 0.8, Harsh 1.2, everything else 1.0.
    pub fn sigma_k(self) -> f32 {
        match self {
            Self::TabularGrain => 0.8,
            Self::HarshGrain => 1.2,
            _ => 1.0,
        }
    }

    /// `ZFILMGRAINTYPE`. C1's jump table has six entries; every other value
    /// (including the >9 some catalogs carry) falls through to type 3.
    pub fn from_c1_type(value: i32) -> Self {
        match value {
            0 => Self::FineGrain,
            1 => Self::SilverRich,
            2 => Self::SoftGrain,
            3 => Self::CubicGrain,
            4 => Self::TabularGrain,
            5 => Self::HarshGrain,
            _ => Self::CubicGrain,
        }
    }
}

fn density_is_zero(value: &f32) -> bool {
    *value == 0.0
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FilmGrain {
    pub kind: GrainKind,
    /// `ZFILMGRAINAMOUNT`, 0 ..= 100.
    pub impact: f32,
    /// `ZFILMGRAINGRANULARITY`, 0 ..= 100.
    pub granularity: f32,
    /// `ZFILMGRAINDENSITY`, 0 ..= 100. C1 reads it only for
    /// [`GrainKind::FineGrain`]; the other five kinds recompute the internal
    /// density from `granularity`, so this slider is ignored there.
    /// See [`Self::density_is_effective`].
    #[serde(skip_serializing_if = "density_is_zero")]
    pub density: f32,
}

impl Default for FilmGrain {
    fn default() -> Self {
        Self { kind: GrainKind::FineGrain, impact: 0.0, granularity: 50.0, density: 0.0 }
    }
}

impl FilmGrain {
    /// Whether C1 actually feeds `density` into the grain generator.
    pub fn density_is_effective(&self) -> bool {
        self.kind == GrainKind::FineGrain
    }

    /// C1's post-grain erf-blur sigma in image pixels.
    ///
    /// `sigma = k * (Amount / 100) * 2^(0.01875 * Granularity - 1.375)`, with
    /// `k` from [`GrainKind::sigma_k`]. FineGrain and SilverRich do not run the
    /// blur, and Amount <= 0 disables grain, so both cases return 0.
    pub fn sigma_px(&self) -> f32 {
        if !self.kind.uses_erf_blur() || !self.impact.is_finite() || self.impact <= 0.0 {
            return 0.0;
        }
        let amount = (self.impact / 100.0).clamp(0.0, 1.0);
        let granularity =
            if self.granularity.is_finite() { self.granularity.clamp(0.0, 100.0) } else { 0.0 };
        self.kind.sigma_k() * amount * 2f32.powf(0.01875 * granularity - 1.375)
    }

    /// Visible grain diameter in pixels (Gaussian FWHM = 2.355 sigma).
    pub fn diameter_px(&self) -> f32 {
        2.355 * self.sigma_px()
    }

    /// The internal density C1 actually feeds to the generator. FineGrain uses
    /// the slider; the other five recompute it from granularity, so moving the
    /// Density slider changes nothing there.
    pub fn effective_density(&self) -> f32 {
        let g = self.granularity.clamp(0.0, 100.0);
        match self.kind {
            GrainKind::FineGrain => self.density.clamp(0.0, 100.0),
            GrainKind::SilverRich => (0.5625 * g + 43.75).trunc(),
            GrainKind::SoftGrain | GrainKind::TabularGrain => (0.375 * g + 37.5).trunc(),
            GrainKind::CubicGrain => (0.5625 * g + 18.75).trunc(),
            GrainKind::HarshGrain => (0.375 * g + 62.5).trunc(),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Moire {
    /// `ZMOIRE` component 0, 0 ..= 100.
    pub amount: f32,
    /// `ZMOIRE` component 1. C1 calls this `IB_F_MOIRE_RADIUS`: the filter
    /// support in image pixels, not a 0..100 strength. The engine clamps it to
    /// the kernel's 40-sample grid.
    pub pattern: f32,
}

// ───────────────────────────── Lens / geometry ─────────────────────────────

/// `ZLENSCHROMABRORIGIN`: where the two-curve lateral CA data came from.
///
/// `None` means no active source, so the stored curves are not applied; C1's
/// `ChromaticAberrationOrigin` uses the same 0/1/2 order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CaOrigin {
    #[default]
    None,
    LensDb,
    Custom,
}

/// `ZLENSCHROMAFRINGING`: six per-channel lateral-fringing curves.
///
/// Each curve is `(radius, value)` control points. The radius is normalized to
/// the frame's half diagonal by the profile reader; `shift` is a radial offset
/// in reference pixels, `ra` a dimensionless radial gain, and `strength` the
/// blend weight (0 leaves the pixels alone). C1 stores the six curves for both
/// channels in `IC_LensChromaFringing`; no kernel that consumes all six has
/// been recovered, so `ra` and `strength` follow the names only (grade C).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CaFringing {
    pub red_shift: Vec<(f32, f32)>,
    pub red_ra: Vec<(f32, f32)>,
    pub red_strength: Vec<(f32, f32)>,
    pub blue_shift: Vec<(f32, f32)>,
    pub blue_ra: Vec<(f32, f32)>,
    pub blue_strength: Vec<(f32, f32)>,
}

impl CaFringing {
    /// True when none of the six curves carries a point.
    pub fn is_empty(&self) -> bool {
        [
            &self.red_shift,
            &self.red_ra,
            &self.red_strength,
            &self.blue_shift,
            &self.blue_ra,
            &self.blue_strength,
        ]
        .iter()
        .all(|curve| curve.is_empty())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LensCorrection {
    /// Lens profile name; `None` is the engine's generic profile.
    pub profile: Option<String>,
    /// Use Capture One's built-in per-lens calibration (distortion, light
    /// falloff and chromatic aberration from `lensDatabase.cdx`) instead of the
    /// generic formula. Off by default so an untouched variant renders as before.
    pub use_profile: bool,
    /// 0 ..= 100 for the correction amounts.
    pub distortion: f32,
    /// `ZLENSRADIALBLUR`'s amount, 0 ..= 120; below 0.5 the op is identity.
    pub sharpness_falloff: f32,
    /// `ZLENSRADIALBLUR`'s two falloff curves and its `valid` flag.
    pub radial_blur: RadialBlur,
    pub light_falloff: f32,
    /// `ZLENSLCCLIGHTFALLOFFAMOUNT`: Lens Calibration "Uniform Light", 0 ..= 120.
    pub uniform_light: f32,
    /// `ZLENSLCCLIGHTFALLOFF` gates `uniform_light`; off leaves the frame alone.
    pub uniform_light_enabled: bool,
    /// Rehydrated from the Capture One calibration sidecar.
    #[serde(skip)]
    pub uniform_light_grid: Option<std::sync::Arc<crate::dust_lcc::LightFalloff>>,
    /// `ZLENSDUSTREMOVALENABLED`: apply the LCC dust patches.
    pub dust_removal_enabled: bool,
    /// Rectangles read from the LCC's `SpotRemoval.SpotData`.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dust_patches: Vec<crate::DustPatch>,
    /// The LCC's `SpotData` plane: one byte per pixel inside each patch.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub dust_gain: Vec<u8>,
    pub chromatic_aberration: bool,
    /// C1's red/blue CA splines, normalized to the first radius.
    pub ca_red_points: Vec<(f32, f32)>,
    pub ca_blue_points: Vec<(f32, f32)>,
    /// `ZLENSCHROMABRORIGIN`: which spline set supplied the two-curve CA data.
    pub ca_origin: CaOrigin,
    /// `ZLENSCHROMAFRINGING`: the profile's six lateral-fringing curves.
    pub ca_fringing: CaFringing,
    pub purple_fringing: f32,
    /// `ZLENSDIFFRACTIONCORRECTIONENABLED`: aperture-dependent Airy blur.
    pub diffraction_enabled: bool,
    /// f-number for [`Self::diffraction_enabled`]; `0` means the capture has none.
    pub aperture: f32,
    pub hide_distorted_areas: bool,
    /// -100 ..= 100, a manual movement/shift of the optical center.
    pub shift_x: f32,
    pub shift_y: f32,
    /// `ZLENSTILT`, degrees; the renderer clamps to +-45.
    pub tilt: f32,
    /// Curves Capture One's built-in "Manufacturer Profile" (type 15) keeps on
    /// the variant instead of `lensDatabase.cdx`, in sensor-plane micrometres.
    pub manufacturer: Option<ManufacturerCurves>,
    /// Camera mount, so the manufacturer curves have a half diagonal to
    /// normalize against.
    pub mount: Option<String>,
}

/// Capture One's sharpness-falloff curves, `ZLENSRADIALBLUR` fields 1 and 2.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RadialBlur {
    pub radial: Vec<(f32, f32)>,
    pub transverse: Vec<(f32, f32)>,
    pub valid: bool,
}

/// `(r_um, Δr_um)` radial curves carried by a Manufacturer Profile variant.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ManufacturerCurves {
    pub distortion: Vec<(f32, f32)>,
    pub ca_red: Vec<(f32, f32)>,
    pub ca_blue: Vec<(f32, f32)>,
}

impl Default for LensCorrection {
    fn default() -> Self {
        Self {
            profile: None,
            use_profile: false,
            distortion: 0.0,
            sharpness_falloff: 0.0,
            radial_blur: RadialBlur::default(),
            light_falloff: 0.0,
            uniform_light: 0.0,
            uniform_light_enabled: false,
            uniform_light_grid: None,
            dust_removal_enabled: false,
            dust_patches: Vec::new(),
            dust_gain: Vec::new(),
            chromatic_aberration: false,
            ca_red_points: Vec::new(),
            ca_blue_points: Vec::new(),
            ca_origin: CaOrigin::None,
            ca_fringing: CaFringing::default(),
            purple_fringing: 0.0,
            diffraction_enabled: false,
            aperture: 0.0,
            hide_distorted_areas: true,
            shift_x: 0.0,
            shift_y: 0.0,
            tilt: 0.0,
            manufacturer: None,
            mount: None,
        }
    }
}

/// A normalized rectangle in image space, `0.0 ..= 1.0` on both axes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NormRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl NormRect {
    pub const FULL: Self = Self { x: 0.0, y: 0.0, width: 1.0, height: 1.0 };
}

impl Default for NormRect {
    fn default() -> Self {
        Self::FULL
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AspectRatio {
    #[default]
    Unconstrained,
    Original,
    Square,
    R4x5,
    R5x7,
    R2x3,
    R3x4,
    R16x9,
    R65x24,
}

impl AspectRatio {
    pub const ALL: [Self; 9] = [
        Self::Unconstrained,
        Self::Original,
        Self::Square,
        Self::R4x5,
        Self::R5x7,
        Self::R2x3,
        Self::R3x4,
        Self::R16x9,
        Self::R65x24,
    ];

    /// Width over height for landscape orientation, `None` when free.
    pub fn ratio(self, original: f32) -> Option<f32> {
        match self {
            Self::Unconstrained => None,
            Self::Original => Some(original),
            Self::Square => Some(1.0),
            Self::R4x5 => Some(5.0 / 4.0),
            Self::R5x7 => Some(7.0 / 5.0),
            Self::R2x3 => Some(3.0 / 2.0),
            Self::R3x4 => Some(4.0 / 3.0),
            Self::R16x9 => Some(16.0 / 9.0),
            Self::R65x24 => Some(65.0 / 24.0),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Keystone {
    /// Slider values, -100 ..= 100 for both axes. These are C1's H/V frame;
    /// [`Keystone::set_hv`] converts them to the stored tiltX/tiltY.
    pub vertical: f32,
    pub horizontal: f32,
    /// 0 ..= 100, how strongly the correction is applied.
    pub amount: f32,
    /// -50 ..= 50, compensates the stretch the correction introduces.
    pub aspect: f32,
    /// `ZGEOMETRYKEYSTONESKEW`, degrees, -45 ..= 45.
    pub skew: f32,
    /// `ZGEOMETRYKEYSTONEFOCALLENGTH`: focal mm / sensor diagonal mm. `0`
    /// means unknown; C1 then uses `1` as the tilt scale.
    pub focal_length: f32,
    /// `IC_KeystoneRaw.flags` (`IC_ProcessSettings+0xfa94`). Bit0 makes
    /// `SetGeoCfg` force `c = 0`, i.e. `UpdateGeo` takes its unnormalised
    /// branch. No C1 code writes the field and it has no `.cos` key, so it
    /// stays at `IB_I_KEYSTONE_FLAG_DEFAULT` (0).
    pub flags: u32,
}

impl Keystone {
    /// C1's `c` from `SetGeoCfg`: `fLen > 0 ? fLen : 1`, forced to `0` by
    /// `flags` bit0. `c = 0` only drops the `fLen` multiply (and the
    /// `1/sqrt(1+L²)` cosine compensation, which this weak perspective does
    /// not model), so the scale consumed here is `1`.
    pub fn tilt_scale(&self) -> f32 {
        if self.flags & 1 != 0 {
            return 1.0;
        }
        let f = self.focal_length;
        if f.is_finite() && f > 0.0 { f } else { 1.0 }
    }

    /// Stored tilt → H/V sliders (`IC_KeystoneGet`). The storage frame is the
    /// image frame; rotation and the H/V flip sign sit between the two.
    pub fn hv(&self, rotation_deg: f32, flip_h: bool, flip_v: bool) -> (f32, f32) {
        let r = if rotation_deg.is_finite() { rotation_deg.to_radians() } else { 0.0 };
        let (s, c) = r.sin_cos();
        let h = self.horizontal * c - self.vertical * s;
        let v = self.horizontal * s + self.vertical * c;
        (if flip_v { -h } else { h }, if flip_h { -v } else { v })
    }

    /// H/V sliders → stored tilt (`IC_KeystoneSet`, the inverse of [`Self::hv`]).
    pub fn set_hv(&mut self, h: f32, v: f32, rotation_deg: f32, flip_h: bool, flip_v: bool) {
        let r = if rotation_deg.is_finite() { rotation_deg.to_radians() } else { 0.0 };
        let (s, c) = r.sin_cos();
        let (h, v) = (if flip_v { -h } else { h }, if flip_h { -v } else { v });
        self.horizontal = h * c + v * s;
        self.vertical = v * c - h * s;
    }

    /// Stored tilt after the `fLen` scale, i.e. the vector the matrix consumes.
    /// Camera data writes `tilt = 100·tan(pitch)/fLen`, so this cancels back to
    /// `100·tan(pitch)` and the correction is independent of the lens.
    pub fn scaled_tilt(&self) -> (f32, f32) {
        let c = self.tilt_scale();
        (self.horizontal * c, self.vertical * c)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Geometry {
    /// Fine rotation, -45 ..= 45 degrees.
    pub rotation: f32,
    /// Quarter turns clockwise, 0 ..= 3.
    pub orientation: u8,
    pub flip_horizontal: bool,
    pub flip_vertical: bool,
    /// `ZCROPOUTSIDEIMAGE`: C1's `AllowCropOutsideOfImage` gate. When set the
    /// stored crop rectangle may extend past the image bounds and is not
    /// clamped; the out-of-image pixels are the renderer's fill.
    pub crop_outside_image: bool,
    pub crop: NormRect,
    /// `ZCANVASSIZE`: the pixel frame `crop` is normalized against. `[0, 0]`
    /// means the canvas is the oriented image.
    pub canvas: [f32; 2],
    pub aspect: AspectRatio,
    pub portrait: bool,
    pub keystone: Keystone,
}

impl Default for Geometry {
    fn default() -> Self {
        Self {
            rotation: 0.0,
            orientation: 0,
            flip_horizontal: false,
            flip_vertical: false,
            crop_outside_image: false,
            crop: NormRect::FULL,
            canvas: [0.0, 0.0],
            aspect: AspectRatio::Unconstrained,
            portrait: false,
            keystone: Keystone { amount: 100.0, ..Default::default() },
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpotKind {
    #[default]
    Dust,
    Spot,
    /// Copy a source patch instead of synthesising pixels. `src_offset` is the
    /// source point minus the spot centre, in pixels of the oriented, uncropped
    /// source image (the same frame the viewer draws).
    Clone { src_offset: (i32, i32) },
}

/// A spot-removal circle in normalized image space.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Spot {
    pub id: u64,
    pub kind: SpotKind,
    pub x: f32,
    pub y: f32,
    /// Radius as a fraction of the image's long side.
    pub radius: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    const NOMINALS: [(WhiteBalanceMode, f32); 6] = [
        (WhiteBalanceMode::Daylight, 5200.0),
        (WhiteBalanceMode::Cloudy, 6000.0),
        (WhiteBalanceMode::Shade, 7000.0),
        (WhiteBalanceMode::Tungsten, 3200.0),
        (WhiteBalanceMode::Fluorescent, 4000.0),
        (WhiteBalanceMode::Flash, 5500.0),
    ];

    #[test]
    fn preset_kelvins_match_capture_one_and_tint_is_neutral() {
        for (mode, nominal) in NOMINALS {
            let (kelvin, tint) = mode.preset().unwrap_or_else(|| panic!("{mode:?} is a preset"));
            assert!((kelvin - nominal).abs() <= nominal * 0.02, "{mode:?}: {kelvin} vs {nominal}");
            assert_eq!(tint, 0.0, "{mode:?}");
        }
        assert_eq!(WhiteBalanceMode::Shot.preset(), None);
        assert_eq!(WhiteBalanceMode::Auto.preset(), None);
    }

    #[test]
    fn as_shot_inverts_exif_gains() {
        let wb = WhiteBalance { mode: WhiteBalanceMode::Shot, ..Default::default() };
        let (kelvin, tint) = wb.resolved(Some([1.625, 1.0, 1.412]), [0.5; 3]);
        assert!((kelvin - 4779.55).abs() < 2.0, "{kelvin}");
        assert!((tint + 4.4875).abs() < 0.05, "{tint}");
    }

    #[test]
    fn as_shot_without_exif_uses_auto() {
        let wb = WhiteBalance { mode: WhiteBalanceMode::Shot, ..Default::default() };
        let mean = [0.55, 0.5, 0.45];
        assert_eq!(wb.resolved(None, mean), auto_from_mean(mean));
        assert_eq!(wb.resolved(Some([0.0, 1.0, 1.0]), mean), auto_from_mean(mean));
        assert_eq!(wb.resolved(Some([f32::NAN, 1.0, 1.0]), mean), auto_from_mean(mean));
    }

    #[test]
    fn auto_is_a_gray_world_estimate() {
        assert_eq!(auto_from_mean([0.5; 3]), (5600.0, 0.0));
        let warm = [0.6, 0.5, 0.4];
        assert!(auto_from_mean(warm).0 < 5600.0);
        let green = [0.5, 0.55, 0.5];
        assert!(auto_from_mean(green).1 > 0.0);
        let wb = WhiteBalance {
            mode: WhiteBalanceMode::Auto,
            kelvin: 1234.0,
            tint: 9.0,
            skin_tone: None,
            gains: None,
        };
        assert_eq!(wb.resolved(None, warm), auto_from_mean(warm));
    }

    #[test]
    fn unknown_preset_name_does_not_panic() {
        assert_eq!(WhiteBalanceMode::from_name("As Shot"), Some(WhiteBalanceMode::Shot));
        assert_eq!(WhiteBalanceMode::from_name("fluorescent"), Some(WhiteBalanceMode::Fluorescent));
        assert_eq!(WhiteBalanceMode::from_name("Cloudy "), Some(WhiteBalanceMode::Cloudy));
        assert_eq!(WhiteBalanceMode::from_name("Klingon 9000"), None);
        let wb = WhiteBalance { mode: WhiteBalanceMode::from_name("Klingon 9000").unwrap_or_default(), ..Default::default() };
        assert_eq!(wb.resolved(None, [0.5; 3]), auto_from_mean([0.5; 3]));
    }

    /// The reference Z6_3 frame: C1 stores gains `1.625;1.000;1.412`, which
    /// resolve to its 4779.6 K / -4.5 tint.
    #[test]
    fn as_shot_reads_the_real_nef_gains() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("testdata/nef-exif-head.bin");
        let block = crate::read_nikon_info_block(&path).expect("sample header parses");
        let gains = block.white_balance_gains().expect("MakerNote WB gains");
        let wb = WhiteBalance { mode: WhiteBalanceMode::Shot, ..Default::default() };
        let (kelvin, tint) = wb.resolved(Some(gains), [0.5; 3]);
        assert!((kelvin - 4779.55).abs() < 2.0, "{kelvin}");
        assert!((tint + 4.4875).abs() < 0.05, "{tint}");
    }

    fn s_curve() -> Curve {
        Curve {
            points: vec![
                CurvePoint::new(0.0, 0.0),
                CurvePoint::new(0.25, 0.10),
                CurvePoint::new(0.75, 0.90),
                CurvePoint::new(1.0, 1.0),
            ],
        }
    }

    fn lut_get(lut: &[f32; 256], x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0) * 255.0;
        let i = (x as usize).min(254);
        let t = x - i as f32;
        lut[i] + (lut[i + 1] - lut[i]) * t
    }

    fn luma(c: [f32; 3]) -> f32 {
        0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    }

    /// The order `mock/process.rs` uses: RGB, then red/green/blue, then luma.
    fn apply_stack(curves: &Curves, mut g: [f32; 3]) -> [f32; 3] {
        let rgb = curves.rgb.lut();
        g = g.map(|v| lut_get(&rgb, v));
        for (i, curve) in [&curves.red, &curves.green, &curves.blue].into_iter().enumerate() {
            let lut = curve.lut();
            g[i] = lut_get(&lut, g[i]);
        }
        let lut = curves.luma.lut();
        let l = luma(g).max(1e-4);
        let k = lut_get(&lut, l) / l;
        [g[0] * k, g[1] * k, g[2] * k]
    }

    #[test]
    fn identity_curve_lut_is_identity() {
        assert!(Curve::default().is_identity());
        let lut = Curve::default().lut();
        for (i, v) in lut.iter().enumerate() {
            assert!((v - i as f32 / 255.0).abs() < 1e-6, "{i}: {v}");
        }
        let mid = Curve { points: vec![CurvePoint::new(0.5, 0.5)] };
        assert!((mid.lut()[128] - 128.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn single_point_lift_raises_midtones() {
        let curve = Curve {
            points: vec![
                CurvePoint::new(0.0, 0.0),
                CurvePoint::new(0.5, 0.7),
                CurvePoint::new(1.0, 1.0),
            ],
        };
        let lut = curve.lut();
        assert_eq!(lut[0], 0.0);
        assert_eq!(lut[255], 1.0);
        assert!((lut[128] - 0.7).abs() < 0.002, "{}", lut[128]);
        assert!(lut[64] > 64.0 / 255.0);
        assert!(lut.windows(2).all(|w| w[1] >= w[0] - 1e-6));
    }

    #[test]
    fn s_curve_steepens_tones() {
        let lut = s_curve().lut();
        assert!(lut[64] < 64.0 / 255.0);
        assert!(lut[192] > 192.0 / 255.0);
        assert!(lut.windows(2).all(|w| w[1] >= w[0] - 1e-6));
        let codes: Vec<u8> = [0, 64, 128, 192, 255].map(|i| (lut[i] * 255.0).round() as u8).to_vec();
        assert_eq!(codes, vec![0, 26, 128, 230, 255]);
    }

    #[test]
    fn non_monotonic_points_are_clamped() {
        let curve = Curve {
            points: vec![
                CurvePoint::new(0.0, 0.0),
                CurvePoint::new(0.3, 0.8),
                CurvePoint::new(0.6, 0.5),
                CurvePoint::new(1.0, 1.0),
            ],
        };
        let lut = curve.lut();
        assert!(lut.windows(2).all(|w| w[1] >= w[0] - 1e-6));
        assert!((lut[153] - 0.8).abs() < 1e-5, "{}", lut[153]);
    }

    #[test]
    fn missing_endpoints_are_completed() {
        let lone = Curve { points: vec![CurvePoint::new(0.5, 0.8)] };
        let lut = lone.lut();
        assert_eq!(lut[0], 0.0);
        assert_eq!(lut[255], 1.0);
        assert!((lut[128] - 0.8).abs() < 0.002, "{}", lut[128]);
        let ends = Curve {
            points: vec![CurvePoint::new(0.2, 0.3), CurvePoint::new(0.8, 0.7)],
        };
        let lut = ends.lut();
        assert_eq!(lut[0], 0.0);
        assert_eq!(lut[255], 1.0);
        assert!((lut_get(&lut, 0.2) - 0.3).abs() < 1e-4);
        assert!((lut_get(&lut, 0.8) - 0.7).abs() < 1e-4);
    }

    #[test]
    fn luma_curve_changes_luminance_not_hue() {
        let before = [0.2, 0.3, 0.35];
        let lut = s_curve().lut();
        let l = luma(before);
        let k = lut_get(&lut, l) / l;
        let after = before.map(|v| v * k);
        assert!((luma(after) - lut_get(&lut, l)).abs() < 1e-5);
        assert!((luma(after) - l).abs() > 0.05);
        for i in 0..3 {
            assert!((after[i] / after[1] - before[i] / before[1]).abs() < 1e-5);
        }
    }

    #[test]
    fn five_curves_stack_rgb_channel_then_luma() {
        let curves = Curves {
            rgb: Curve {
                points: vec![
                    CurvePoint::new(0.0, 0.0),
                    CurvePoint::new(0.5, 0.8),
                    CurvePoint::new(1.0, 1.0),
                ],
            },
            red: Curve {
                points: vec![
                    CurvePoint::new(0.0, 0.0),
                    CurvePoint::new(0.5, 0.2),
                    CurvePoint::new(1.0, 1.0),
                ],
            },
            ..Default::default()
        };
        let got = apply_stack(&curves, [0.5; 3]);
        // RGB first gives 0.8; red then gives 0.68. Red first would give 0.32.
        assert!((got[0] - 0.68).abs() < 0.002, "{got:?}");
        assert!((got[1] - 0.8).abs() < 0.002, "{got:?}");

        let curves = Curves {
            red: Curve {
                points: vec![
                    CurvePoint::new(0.0, 0.0),
                    CurvePoint::new(0.5, 0.8),
                    CurvePoint::new(1.0, 1.0),
                ],
            },
            luma: s_curve(),
            ..Default::default()
        };
        let got = apply_stack(&curves, [0.5; 3]);
        let r = lut_get(&curves.red.lut(), 0.5);
        let l = luma([r, 0.5, 0.5]);
        let k = lut_get(&curves.luma.lut(), l) / l;
        assert!((got[0] - r * k).abs() < 1e-5, "{got:?}");
        assert!((got[1] - 0.5 * k).abs() < 1e-5, "{got:?}");
    }

    /// Documents written before `color_balance_gains` existed carry no field and
    /// must read back as `None`; a default document must not start writing it.
    #[test]
    fn old_documents_read_color_balance_gains_as_none() {
        let a: AdjustmentSet =
            serde_json::from_str(r#"{"exposure":{"exposure":0.5}}"#).expect("old document");
        assert_eq!(a.exposure.exposure, 0.5);
        assert_eq!(a.color_balance_gains, None);
        let text = serde_json::to_string(&AdjustmentSet::default()).expect("serializes");
        assert!(!text.contains("color_balance_gains"), "{text}");
    }

    #[test]
    fn levels_identity_is_identity() {
        let levels = Levels::default();
        for v in [0.0, 0.125, 0.5, 0.75, 1.0] {
            assert!(close(LevelsChannel::default().map(v), v));
        }
        assert_eq!(levels.apply([0.25, 0.5, 0.75]), [0.25, 0.5, 0.75]);
    }

    #[test]
    fn levels_input_black_lifts_shadows() {
        let channel = LevelsChannel { input_black: 0.1, ..LevelsChannel::default() };
        assert_eq!(channel.map(0.0), 0.0);
        assert_eq!(channel.map(0.1), 0.0);
        assert!(close(channel.map(0.55), 0.5));
        assert!(close(channel.map(1.0), 1.0));
    }

    #[test]
    fn levels_output_white_compresses_highlights() {
        let channel = LevelsChannel { output_white: 0.8, ..LevelsChannel::default() };
        assert_eq!(channel.map(0.0), 0.0);
        assert!(close(channel.map(0.5), 0.4));
        assert!(close(channel.map(1.0), 0.8));
    }

    #[test]
    fn levels_gamma_direction_and_points_combine() {
        let brighten = LevelsChannel { gamma: 2.0, ..LevelsChannel::default() };
        let darken = LevelsChannel { gamma: 0.5, ..LevelsChannel::default() };
        assert!(close(brighten.map(0.25), 0.5));
        assert!(close(darken.map(0.25), 0.0625));

        let combined = LevelsChannel {
            input_black: 0.1,
            input_white: 0.9,
            gamma: 2.0,
            output_black: 0.05,
            output_white: 0.95,
        };
        assert!(close(combined.map(0.1), 0.05));
        assert!(close(combined.map(0.3), 0.5));
        assert!(close(combined.map(0.9), 0.95));
    }

    #[test]
    fn levels_channels_do_not_leak() {
        let levels = Levels {
            red: LevelsChannel { input_black: 0.2, ..LevelsChannel::default() },
            ..Levels::default()
        };
        let out = levels.apply([0.5, 0.5, 0.5]);
        assert!(close(out[0], 0.375));
        assert_eq!(out[1], 0.5);
        assert_eq!(out[2], 0.5);
    }

    #[test]
    fn levels_apply_single_channels_before_rgb() {
        let levels = Levels {
            rgb: LevelsChannel { input_black: 0.25, ..LevelsChannel::default() },
            red: LevelsChannel { gamma: 2.0, ..LevelsChannel::default() },
            ..Levels::default()
        };
        // Red first: 0.5^0.5 = 0.7071, then RGB: (0.7071 - 0.25) / 0.75.
        assert!(close(levels.apply([0.5, 0.5, 0.5])[0], 0.609476));
        // RGB first would give 0.57735, so this pins the order.
        assert!(!close(levels.apply([0.5, 0.5, 0.5])[0], 0.577350));
    }

    #[test]
    fn c1_midtone_maps_to_gamma() {
        assert!(close(gamma_from_c1_midtone(0.0), 1.0));
        assert!(close(gamma_from_c1_midtone(0.5), 2.0));
        assert!(close(gamma_from_c1_midtone(-0.5), 0.4150375));
        let m = c1_midtone_from_gamma(1.2);
        assert!(close(m, 0.129449));
        assert!(close(gamma_from_c1_midtone(m), 1.2));
    }

    #[test]
    fn c1_clarity_method_numbers_follow_ic_enum_order() {
        assert_eq!(ClarityMethod::from_c1(0), ClarityMethod::Neutral);
        assert_eq!(ClarityMethod::from_c1(1), ClarityMethod::Punch);
        assert_eq!(ClarityMethod::from_c1(2), ClarityMethod::Classic);
        assert_eq!(ClarityMethod::from_c1(3), ClarityMethod::Natural);
        assert_eq!(ClarityMethod::from_c1(9), ClarityMethod::Natural);
        for method in ClarityMethod::ALL {
            assert_eq!(ClarityMethod::from_c1(method.to_c1()), method);
        }
    }

    #[test]
    fn c1_clarity_amount_and_radius_match_the_host_formula() {
        let neutral = Clarity { method: ClarityMethod::Neutral, clarity: 50.0, structure: 0.0 };
        assert!(close(neutral.c1_amount(), 0.6));
        let natural = Clarity { method: ClarityMethod::Natural, clarity: -100.0, structure: 0.0 };
        assert!(close(natural.c1_amount(), -0.6));
        let punch = Clarity { method: ClarityMethod::Punch, clarity: -100.0, structure: 0.0 };
        assert!(close(punch.c1_amount(), -0.3));
        assert!(close(Clarity { clarity: 0.0, ..Clarity::default() }.c1_amount(), 0.0));
        assert!(close(Clarity { structure: 100.0, ..Clarity::default() }.c1_structure(), 1.0));
        assert!(close(Clarity { structure: -100.0, ..Clarity::default() }.c1_structure(), -1.0));

        assert_eq!(c1_clarity_radius_big(6000, 4000, 1.0), 500);
        assert_eq!(c1_clarity_radius_big(6000, 4000, 0.01), 8);
        assert_eq!(c1_clarity_radius_small(6000, 1.0), 50);
        assert_eq!(c1_clarity_radius_small(6000, 0.01), 8);
        assert_eq!(c1_clarity_lowres_radius(500), 62);
    }

    #[test]
    fn c1_bw_runtime_order_is_blue_yellow_cyan_green_red_magenta() {
        let bw = BlackWhite { sensitivity: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], ..BlackWhite::default() };
        assert_eq!(bw.c1_runtime_sensitivity(), [50.0, 20.0, 40.0, 30.0, 10.0, 60.0]);
    }

    #[test]
    fn keystone_hv_roundtrips_through_rotation_and_flip() {
        for rot in [0.0, 90.0, 270.0, 45.0, -12.5] {
            for (fh, fv) in [(false, false), (true, false), (false, true), (true, true)] {
                let stored = Keystone { horizontal: 30.0, vertical: -20.0, ..Keystone::default() };
                let (h, v) = stored.hv(rot, fh, fv);
                let mut back = Keystone::default();
                back.set_hv(h, v, rot, fh, fv);
                assert!((back.horizontal - stored.horizontal).abs() < 1e-4, "{rot} {fh} {fv} h");
                assert!((back.vertical - stored.vertical).abs() < 1e-4, "{rot} {fh} {fv} v");
            }
        }
        // The 90° sample from the UI report: h = -tiltY, v = tiltX, then flip_h negates v.
        let stored = Keystone { horizontal: 30.0, vertical: -20.0, ..Keystone::default() };
        for (got, want) in [
            (stored.hv(90.0, false, false), (20.0, 30.0)),
            (stored.hv(90.0, true, false), (20.0, -30.0)),
            (stored.hv(90.0, false, true), (-20.0, 30.0)),
        ] {
            assert!((got.0 - want.0).abs() < 1e-4 && (got.1 - want.1).abs() < 1e-4, "{got:?} != {want:?}");
        }
    }

    #[test]
    fn focal_length_scale_is_c1_fallback_one() {
        assert_eq!(Keystone { focal_length: 0.0, ..Keystone::default() }.tilt_scale(), 1.0);
        assert_eq!(Keystone { focal_length: -0.3, ..Keystone::default() }.tilt_scale(), 1.0);
        assert_eq!(Keystone { focal_length: 0.35, ..Keystone::default() }.tilt_scale(), 0.35);
    }

    /// `flags` bit0 (`IC_ProcessSettings+0xfa94`) makes `SetGeoCfg` force `c=0`,
    /// which drops the `fLen` factor in `UpdateGeo`.
    #[test]
    fn flags_bit0_drops_the_focal_length_scale() {
        let k = Keystone { focal_length: 0.35, flags: 1, ..Keystone::default() };
        assert_eq!(k.tilt_scale(), 1.0);
        assert_eq!(Keystone { flags: 2, focal_length: 0.35, ..Keystone::default() }.tilt_scale(), 0.35);
        assert_eq!(Keystone::default().tilt_scale(), 1.0);
    }

    #[test]
    fn levels_reference_curve_samples_c1_ui() {
        // C1 UI: input 10 / 240, midtone 0.13 (= gamma 1.2), output 0 / 255.
        let channel = LevelsChannel {
            input_black: 10.0 / 255.0,
            input_white: 240.0 / 255.0,
            gamma: 1.2,
            ..LevelsChannel::default()
        };
        let samples = [(0.0, 0.0), (32.0, 0.1414), (64.0, 0.2989), (128.0, 0.5734), (192.0, 0.8228), (240.0, 1.0)];
        for (input, expected) in samples {
            assert!(close(channel.map(input / 255.0), expected), "{input} -> {}", channel.map(input / 255.0));
        }
    }

    fn grain(kind: GrainKind, impact: f32, granularity: f32) -> FilmGrain {
        FilmGrain { kind, impact, granularity, density: 0.0 }
    }

    #[test]
    fn grain_sigma_and_diameter_follow_the_c1_curve() {
        for (granularity, sigma, diameter) in
            [(0.0, 0.386, 0.91), (50.0, 0.738, 1.74), (100.0, 1.414, 3.33)]
        {
            let g = grain(GrainKind::SoftGrain, 100.0, granularity);
            assert!((g.sigma_px() - sigma).abs() < 0.002, "g={granularity}: {}", g.sigma_px());
            assert!(
                (g.diameter_px() - diameter).abs() < 0.01,
                "g={granularity}: {}",
                g.diameter_px()
            );
        }
    }

    #[test]
    fn grain_type_scales_sigma_and_fine_grain_skips_the_blur() {
        let unit = grain(GrainKind::SoftGrain, 100.0, 50.0).sigma_px();
        assert!(close(grain(GrainKind::CubicGrain, 100.0, 50.0).sigma_px(), unit));
        assert!(close(grain(GrainKind::TabularGrain, 100.0, 50.0).sigma_px(), unit * 0.8));
        assert!(close(grain(GrainKind::HarshGrain, 100.0, 50.0).sigma_px(), unit * 1.2));
        for kind in [GrainKind::FineGrain, GrainKind::SilverRich] {
            assert_eq!(grain(kind, 100.0, 50.0).sigma_px(), 0.0, "{kind:?}");
            assert_eq!(grain(kind, 100.0, 50.0).diameter_px(), 0.0, "{kind:?}");
        }
    }

    #[test]
    fn grain_amount_sets_sigma_and_zero_turns_it_off() {
        let full = grain(GrainKind::SoftGrain, 100.0, 50.0).sigma_px();
        assert!(close(grain(GrainKind::SoftGrain, 50.0, 50.0).sigma_px(), full * 0.5));
        assert_eq!(grain(GrainKind::SoftGrain, 0.0, 50.0).sigma_px(), 0.0);
    }

    #[test]
    fn grain_density_only_applies_to_fine_grain() {
        let mut g = FilmGrain { density: 70.0, ..FilmGrain::default() };
        assert!(g.density_is_effective());
        for kind in [
            GrainKind::SilverRich,
            GrainKind::SoftGrain,
            GrainKind::CubicGrain,
            GrainKind::TabularGrain,
            GrainKind::HarshGrain,
        ] {
            g.kind = kind;
            assert!(!g.density_is_effective(), "{kind:?}");
        }
        // Density is not part of the sigma curve; changing it must not move the blur.
        let mut soft = grain(GrainKind::SoftGrain, 100.0, 50.0);
        let sigma = soft.sigma_px();
        soft.density = 100.0;
        assert_eq!(soft.sigma_px(), sigma);
    }

    #[test]
    fn default_grain_is_a_no_op_and_keeps_serialization_stable() {
        let g = FilmGrain::default();
        assert_eq!(g.sigma_px(), 0.0);
        assert_eq!(g.diameter_px(), 0.0);
        let text = serde_json::to_string(&AdjustmentSet::default()).expect("serializes");
        assert!(!text.contains("\"density\""), "{text}");
    }
}
