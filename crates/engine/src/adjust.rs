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
    pub color_editor: ColorEditor,
    pub black_white: BlackWhite,
    pub sharpening: Sharpening,
    pub noise_reduction: NoiseReduction,
    pub film_grain: FilmGrain,
    pub moire: Moire,
    pub lens: LensCorrection,
    pub geometry: Geometry,
    pub spots: Vec<Spot>,
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
}

impl AdjustmentSet {
    /// True when nothing differs from a freshly imported image.
    pub fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

// ───────────────────────────── Base characteristics ─────────────────────────────

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BaseCharacteristics {
    /// Camera input profile name; `None` means the engine's profile for the camera.
    pub icc_profile: Option<String>,
    pub curve: ToneCurvePreset,
    /// Capture One `.fcrv` file name (`ZFILMCURVE`), applied to linear values
    /// before gamma. Keyed on the file name: the curve's embedded name does not
    /// match its content in Capture One's own files.
    pub film_curve: Option<String>,
    pub engine_version: String,
}

impl Default for BaseCharacteristics {
    fn default() -> Self {
        Self { icc_profile: None, curve: ToneCurvePreset::Auto, film_curve: None, engine_version: "Lumen 1".into() }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ToneCurvePreset {
    #[default]
    Auto,
    FilmStandard,
    FilmExtraShadow,
    FilmHighContrast,
    Linear,
}

impl ToneCurvePreset {
    pub const ALL: [Self; 5] =
        [Self::Auto, Self::FilmStandard, Self::FilmExtraShadow, Self::FilmHighContrast, Self::Linear];
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
}

impl Default for WhiteBalance {
    fn default() -> Self {
        Self { mode: WhiteBalanceMode::Shot, kelvin: 5600.0, tint: 0.0, skin_tone: None }
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
            WhiteBalanceMode::Shot => exif_gains
                .filter(|g| g.iter().all(|v| v.is_finite() && *v > 0.0))
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
        let span = (self.input_white - self.input_black).max(1e-3);
        let t = ((v - self.input_black) / span).clamp(0.0, 1.0);
        let t = t.powf(1.0 / self.gamma.clamp(0.1, 10.0));
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
        [
            self.rgb.map(self.red.map(rgb[0])),
            self.rgb.map(self.green.map(rgb[1])),
            self.rgb.map(self.blue.map(rgb[2])),
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

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Dehaze {
    /// 0 ..= 100.
    pub amount: f32,
    /// Haze color the engine removes, linear sRGB.
    pub shadow_tone: [f32; 3],
}

impl Default for Dehaze {
    fn default() -> Self {
        Self { amount: 0.0, shadow_tone: [0.0, 0.0, 0.0] }
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
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FilmGrain {
    pub kind: GrainKind,
    /// 0 ..= 100.
    pub impact: f32,
    /// 0 ..= 100.
    pub granularity: f32,
}

impl Default for FilmGrain {
    fn default() -> Self {
        Self { kind: GrainKind::FineGrain, impact: 0.0, granularity: 50.0 }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Moire {
    /// 0 ..= 100.
    pub amount: f32,
    /// 0 ..= 100.
    pub pattern: f32,
}

// ───────────────────────────── Lens / geometry ─────────────────────────────

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
    pub sharpness_falloff: f32,
    pub light_falloff: f32,
    pub chromatic_aberration: bool,
    pub purple_fringing: f32,
    pub hide_distorted_areas: bool,
    /// -100 ..= 100, a manual movement/shift of the optical center.
    pub shift_x: f32,
    pub shift_y: f32,
}

impl Default for LensCorrection {
    fn default() -> Self {
        Self {
            profile: None,
            use_profile: false,
            distortion: 0.0,
            sharpness_falloff: 0.0,
            light_falloff: 0.0,
            chromatic_aberration: false,
            purple_fringing: 0.0,
            hide_distorted_areas: true,
            shift_x: 0.0,
            shift_y: 0.0,
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
    /// -100 ..= 100 for both axes.
    pub vertical: f32,
    pub horizontal: f32,
    /// 0 ..= 100, how strongly the correction is applied.
    pub amount: f32,
    /// -50 ..= 50, compensates the stretch the correction introduces.
    pub aspect: f32,
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
    pub crop: NormRect,
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
            crop: NormRect::FULL,
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
        assert!((kelvin - 5648.0).abs() < 20.0, "{kelvin}");
        assert!((tint + 41.5).abs() < 1.0, "{tint}");
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
        let wb = WhiteBalance { mode: WhiteBalanceMode::Auto, kelvin: 1234.0, tint: 9.0, skin_tone: None };
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

    #[cfg(feature = "external-fixtures")]
    include!("../local-fixtures/adjust-tests.rs");
}
