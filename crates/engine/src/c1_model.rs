//! Capture One's own adjustment model, decoded from the `ZVARIANTLAYER` table.
//! Data source: the real C1 catalog `Capture One.cocatalogdb`, table `ZVARIANTLAYER` (163 columns).
//! This decodes only the *input parameters* Capture One stores for a variant layer.
//! It is **not** Capture One's rendering operator: the maths that turns these values
//! into pixels still has to be re-derived (or measured) separately.

use std::collections::HashMap;

use crate::adjust::{
    AspectRatio, BlackWhite, Clarity, ClarityMethod, ColorBalance, ColorBand, ColorEditor,
    ColorRange, ColorWheel, Curve, CurvePoint, Curves, Dehaze, Exposure, FilmGrain, Geometry,
    GrainKind, HighDynamicRange, HslShift, HueSector, Keystone, Levels, LevelsChannel,
    LensCorrection, Moire, NoiseReduction, NormRect, Sharpening, SkinUniformity, SplitTone,
    ToneCurvePreset, VignetteMethod, Vignetting, WhiteBalance, WhiteBalanceMode,
    gamma_from_c1_midtone,
};

// ─────────────────────────────────────────────────────────────────────────────
// Column table
// ─────────────────────────────────────────────────────────────────────────────

/// How a `ZVARIANTLAYER` value is encoded as text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum C1Encoding {
    /// One number (`-0.6492753624916077`). Text columns use this too; their value
    /// is kept as a string and never goes through [`parse_scalar`].
    Scalar,
    /// Semicolon-separated components (`1.625000;1.000000;1.412000`).
    Semicolon,
    /// Semicolon-separated `in,out` pairs (`0.000,0.000;1.000,1.000`).
    CommaPairs,
    /// Pipe-separated composite fields (`0.000000|1|0|1`).
    Pipe,
    /// Semicolon-separated groups of comma-separated values (`9 × 18` for
    /// `ZCOLORCORRECTIONS`).
    SemicolonComma,
}

/// One column of `ZVARIANTLAYER` and how its value is stored.
#[derive(Clone, Copy, Debug)]
pub struct C1Column {
    pub name: &'static str,
    pub encoding: C1Encoding,
    /// Number of components in one group: 1 for scalars, 3 for `R;G;B`, 4 for the
    /// four-channel levels vectors, 18 for one `ZCOLORCORRECTIONS` group, and the
    /// number of `|`-separated fields for [`C1Encoding::Pipe`].
    pub components: usize,
}

/// Every `ZVARIANTLAYER` column this model understands.
///
/// The first block is the full set of columns whose semantics *and* encoding were
/// confirmed against the real catalog (`zvariantlayer-map.json`, confidence >= 0.55).
/// The second block holds the columns this model carries but whose semantics are
/// inferred; each one is marked in the comment above its group.
pub static C1_COLUMNS: &[C1Column] = &[
    // ── identity / bookkeeping (confidence >= 0.55) ───────────────────────────
    C1Column { name: "Z_ENT", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "Z_PK", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZINDEX", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZMETADATA", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZVARIANT", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZENABLED", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZOPACITY", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZNAME", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZMASKUUID", encoding: C1Encoding::Scalar, components: 1 },
    // ── exposure / tone (confidence >= 0.55) ─────────────────────────────────
    C1Column { name: "ZEXPOSURE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZCONTRAST", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBRIGHTNESS", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZSATURATION", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZCLARITY", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZCLARITYSTRUCTURE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZCLARITYMETHOD", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZHIGHLIGHTRECOVERY", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZSHADOWRECOVERY", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZWHITERECOVERY", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBLACKRECOVERY", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZDEHAZEAMOUNT", encoding: C1Encoding::Scalar, components: 1 },
    // ── detail (confidence >= 0.55) ──────────────────────────────────────────
    C1Column { name: "ZUSMAMOUNT", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZUSMRADIUS", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZUSMTHRESHOLD", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZUSMHALOCONTROL", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZNOISEREDUCTIONAMOUNT", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZNOISEDETAILSAMOUNT", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZFILMGRAINGRANULARITY", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZFILMGRAINTYPE", encoding: C1Encoding::Scalar, components: 1 },
    // ── black & white (confidence >= 0.55) ───────────────────────────────────
    C1Column { name: "ZBWENABLED", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBWRED", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBWYELLOW", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBWGREEN", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBWCYAN", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBWBLUE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBWMAGENTA", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBWDARKTONEHUE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBWDARKTONESATURATION", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBWLIGHTTONEHUE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBWLIGHTTONESATURATION", encoding: C1Encoding::Scalar, components: 1 },
    // ── geometry (confidence >= 0.55) ────────────────────────────────────────
    C1Column { name: "ZCROP", encoding: C1Encoding::Semicolon, components: 4 },
    C1Column { name: "ZCROPOUTSIDEIMAGE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZCROPASPECTRATIONAME", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZGEOMETRYKEYSTONEAMOUNT", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZGEOMETRYKEYSTONEASPECT", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZGEOMETRYKEYSTONETILTX", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZGEOMETRYKEYSTONETILTY", encoding: C1Encoding::Scalar, components: 1 },
    // ── lens / base (confidence >= 0.55) ─────────────────────────────────────
    C1Column { name: "ZLENSCHROMABRENABLED", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSIDGUID", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSREVISIONIDGUID", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSAPERTURE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSFOCALLENGTH", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZICCPROFILE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZFILMCURVE", encoding: C1Encoding::Scalar, components: 1 },
    // ── inferred columns this model still carries ───────────────────────────
    // White balance: R;G;B gains, G is always 1.
    C1Column { name: "ZWHITEBALANCE", encoding: C1Encoding::Semicolon, components: 3 },
    // Levels: four channels (rgb, red, green, blue) of black / white points.
    C1Column { name: "ZLEVELSSHADOW", encoding: C1Encoding::Semicolon, components: 4 },
    C1Column { name: "ZLEVELSHIGHLIGHT", encoding: C1Encoding::Semicolon, components: 4 },
    C1Column { name: "ZDEHAZECOLOR", encoding: C1Encoding::Semicolon, components: 3 },
    C1Column { name: "ZMOIRE", encoding: C1Encoding::Semicolon, components: 2 },
    C1Column { name: "ZCANVASSIZE", encoding: C1Encoding::Semicolon, components: 2 },
    C1Column { name: "ZCOLORBALANCE", encoding: C1Encoding::Semicolon, components: 3 },
    C1Column { name: "ZCOLORBALANCESHADOW", encoding: C1Encoding::Semicolon, components: 3 },
    C1Column { name: "ZCOLORBALANCEMIDTONE", encoding: C1Encoding::Semicolon, components: 3 },
    C1Column { name: "ZCOLORBALANCEHIGHLIGHT", encoding: C1Encoding::Semicolon, components: 3 },
    C1Column { name: "ZLENSSHIFTXY", encoding: C1Encoding::Semicolon, components: 2 },
    C1Column { name: "ZGRADATIONCURVE", encoding: C1Encoding::CommaPairs, components: 2 },
    C1Column { name: "ZGRADATIONCURVEBLUE", encoding: C1Encoding::CommaPairs, components: 2 },
    C1Column { name: "ZGRADATIONCURVEGREEN", encoding: C1Encoding::CommaPairs, components: 2 },
    C1Column { name: "ZGRADATIONCURVERED", encoding: C1Encoding::CommaPairs, components: 2 },
    C1Column { name: "ZGRADATIONCURVEY", encoding: C1Encoding::CommaPairs, components: 2 },
    C1Column { name: "ZLENSVIGNETTING", encoding: C1Encoding::Pipe, components: 4 },
    C1Column { name: "ZLENSDISTORTION", encoding: C1Encoding::Pipe, components: 5 },
    C1Column { name: "ZLENSLIGHTFALLOFF", encoding: C1Encoding::Pipe, components: 3 },
    C1Column { name: "ZLENSDEFAULTCHROMABR", encoding: C1Encoding::Pipe, components: 2 },
    C1Column { name: "ZLENSOPTICCENTER", encoding: C1Encoding::Pipe, components: 2 },
    C1Column { name: "ZCOLORCORRECTIONS", encoding: C1Encoding::SemicolonComma, components: 18 },
    C1Column { name: "ZROTATION", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZGEOMETRYCANVASCLIPPING", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZFILMGRAINAMOUNT", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZFILMGRAINDENSITY", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZAIDENOISEAMOUNT", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZCNRAMOUNT", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZDENOISEMETHOD", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZCLEANLONGEXPOSUREAMOUNT", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZLENSPURPLEFRINGINGAMOUNT", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZSURFACESMOOTHNESSAMOUNT", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBRUSHSIZE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBRUSHHARDNESS", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBRUSHOPACITY", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBRUSHFLOW", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBRUSHAIRBRUSH", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBRUSHUSEPENPRESSURE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZBRUSHAUTOMASK", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZERASERSIZE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZERASERHARDNESS", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZERASEROPACITY", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZERASERFLOW", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZERASERAIRBRUSH", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZERASERUSEPENPRESSURE", encoding: C1Encoding::Scalar, components: 1 },
    C1Column { name: "ZERASERAUTOMASK", encoding: C1Encoding::Scalar, components: 1 },
];

/// Looks a column up by name.
pub fn column(name: &str) -> Option<&'static C1Column> {
    C1_COLUMNS.iter().find(|c| c.name == name)
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsers
// ─────────────────────────────────────────────────────────────────────────────

/// A multi-segment spline profile as C1 stores lens calibration data:
/// `x,y;x,y;…|param|param|…`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SplineProfile {
    /// One spline per `|`-separated field that holds `x,y` pairs.
    pub splines: Vec<Vec<(f32, f32)>>,
    /// The remaining `|`-separated fields, read as plain numbers.
    pub params: Vec<f32>,
}

impl SplineProfile {
    /// The first spline's control points, if any.
    pub fn points(&self) -> &[(f32, f32)] {
        self.splines.first().map(|s| s.as_slice()).unwrap_or(&[])
    }

    /// The scalar parameter at `index`, if present.
    pub fn param(&self, index: usize) -> Option<f32> {
        self.params.get(index).copied()
    }
}

/// Strips the quoting and whitespace C1 wraps its values in.
pub(crate) fn clean(raw: &str) -> &str {
    raw.trim().trim_matches('\'').trim()
}

/// Parses one number. Empty strings, non-numbers and NaN all yield `None`.
pub fn parse_scalar(raw: &str) -> Option<f32> {
    let t = clean(raw);
    if t.is_empty() {
        return None;
    }
    let v: f32 = t.parse().ok()?;
    if v.is_nan() {
        None
    } else {
        Some(v)
    }
}

/// Parses a semicolon-separated vector. Unparsable components are skipped, so the
/// result may be shorter than the source.
pub fn parse_vec(raw: &str) -> Vec<f32> {
    clean(raw).split(';').filter_map(parse_scalar).collect()
}

/// Parses `in,out;in,out` control points. Pairs that are not two numbers are skipped.
pub fn parse_curve(raw: &str) -> Vec<(f32, f32)> {
    let mut out = Vec::new();
    for pair in clean(raw).split(';') {
        let mut it = pair.split(',');
        let (Some(x), Some(y)) = (it.next().and_then(parse_scalar), it.next().and_then(parse_scalar))
        else {
            continue;
        };
        out.push((x, y));
    }
    out
}

/// Parses a pipe-separated lens profile. Fields that hold `x,y` pairs become
/// splines, the rest become scalar parameters.
pub fn parse_spline(raw: &str) -> SplineProfile {
    let mut profile = SplineProfile::default();
    for field in clean(raw).split('|') {
        if field.contains(',') {
            let points = parse_curve(field);
            if !points.is_empty() {
                profile.splines.push(points);
            }
        } else if let Some(v) = parse_scalar(field) {
            profile.params.push(v);
        }
    }
    profile
}

/// Parses `ZCOLORCORRECTIONS`: 9 groups of 18 comma-separated numbers.
pub fn parse_color_corrections(raw: &str) -> Vec<[f32; 18]> {
    clean(raw)
        .split(';')
        .filter(|g| !g.trim().is_empty())
        .map(|group| {
            let mut out = [0.0f32; 18];
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = group.split(',').nth(i).and_then(parse_scalar).unwrap_or(0.0);
            }
            out
        })
        .collect()
}

/// One decoded `ZCOLORCORRECTIONS` group. Field order (see the evidence table in
/// the module docs): `local, enabled, kind, hue, saturation, lightness,
/// R, G, B, hue_low, hue_high, sat_low, sat_high, smoothness, hom_hue,
/// inverted, hom_sat, hom_light`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ColorCorrection {
    /// Field 0: set for local layers, clear for the trailing global group.
    pub local: bool,
    /// Field 1.
    pub enabled: bool,
    /// Field 2: 0 = advanced, 1 = basic, 2 = skin, 3 = max smooth.
    pub kind: u8,
    /// Field 3, degrees.
    pub hue: f32,
    /// Field 4.
    pub saturation: f32,
    /// Field 5, stored at 1/10 of the UI slider (C1 UI shows -50 ..= 50).
    pub lightness: f32,
    /// Fields 6-8: the selected color, 0 ..= 255 as stored.
    pub rgb: [f32; 3],
    /// Field 9, degrees (negative).
    pub hue_low: f32,
    /// Field 10, degrees (positive).
    pub hue_high: f32,
    /// Field 11, percentage points relative to the selected saturation.
    pub sat_low: f32,
    /// Field 12, percentage points.
    pub sat_high: f32,
    /// Field 13, degrees (C1: 1 ..= 29.1).
    pub smoothness: f32,
    /// Fields 14, 16, 17: homogeneity hue / saturation / lightness.
    pub homogeneity: [f32; 3],
    /// Field 15.
    pub inverted: bool,
}

impl ColorCorrection {
    pub fn from_row(v: &[f32; 18]) -> Self {
        Self {
            local: v[0] != 0.0,
            enabled: v[1] != 0.0,
            kind: v[2] as u8,
            hue: v[3],
            saturation: v[4],
            lightness: v[5],
            rgb: [v[6], v[7], v[8]],
            hue_low: v[9],
            hue_high: v[10],
            sat_low: v[11],
            sat_high: v[12],
            smoothness: v[13],
            homogeneity: [v[14], v[16], v[17]],
            inverted: v[15] != 0.0,
        }
    }
}

/// Hue and saturation of the group's selected color.
pub fn selected_hue_saturation(rgb: [f32; 3]) -> (f32, f32) {
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    let saturation = if max < 1e-6 { 0.0 } else { (max - min) / max };
    (rgb_to_hue(rgb[0], rgb[1], rgb[2]), saturation)
}

/// Positional read of a semicolon vector: missing or broken components fall back
/// to `default` so a short vector never shifts the ones after it.
fn components(raw: Option<&str>, count: usize, default: f32) -> Vec<f32> {
    let mut out = vec![default; count];
    if let Some(text) = raw {
        for (i, slot) in out.iter_mut().enumerate() {
            *slot = text.split(';').nth(i).and_then(parse_scalar).unwrap_or(default);
        }
    }
    out
}

fn fixed(raw: Option<&str>, default: f32) -> f32 {
    raw.and_then(parse_scalar).unwrap_or(default)
}

pub(crate) fn flag(raw: Option<&str>, default: bool) -> bool {
    match raw.map(parse_scalar) {
        Some(Some(v)) => v != 0.0,
        _ => default,
    }
}

pub(crate) fn text(raw: Option<&str>) -> Option<String> {
    let t = clean(raw.unwrap_or(""));
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

/// File name without its extension; C1 stores `name.icm` / `name.fcrv`.
fn stem_of(name: &str) -> String {
    std::path::Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(name)
        .to_string()
}

// ─────────────────────────────────────────────────────────────────────────────
// Settings
// ─────────────────────────────────────────────────────────────────────────────

/// The kind of row a `ZVARIANTLAYER` entry is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum C1EntityKind {
    /// `Z_ENT` 17: a delta layer that only stores what was changed.
    Adjustment,
    /// `Z_ENT` 18: the default layer, a full snapshot.
    #[default]
    DefaultLayer,
    /// `Z_ENT` 20: the combined layer, a full merged snapshot.
    Combined,
    /// `Z_ENT` 23: a local / brush mask layer.
    LocalMask,
}

impl C1EntityKind {
    /// Maps a `Z_ENT` value onto the kind.
    pub fn from_ent(ent: i64) -> Option<Self> {
        match ent {
            17 => Some(Self::Adjustment),
            18 => Some(Self::DefaultLayer),
            20 => Some(Self::Combined),
            23 => Some(Self::LocalMask),
            _ => None,
        }
    }
}

/// White balance, colour balance, geometry, lens and detail settings as C1 stores them.
#[derive(Clone, Debug, PartialEq)]
pub struct C1VariantSettings {
    pub entity: C1EntityKind,
    pub index: i64,
    pub enabled: bool,
    pub name: Option<String>,
    pub opacity: f32,
    pub mask_uuid: Option<String>,

    pub exposure: f32,
    pub contrast: f32,
    pub brightness: f32,
    pub saturation: f32,
    /// R;G;B gains, G is always 1 in C1's own data.
    pub white_balance_gains: [f32; 3],

    /// Black point per channel (rgb, red, green, blue), `0.0 ..= 1.0`.
    pub levels_shadow: [f32; 4],
    /// White point per channel (rgb, red, green, blue), `0.0 ..= 1.0`.
    pub levels_highlight: [f32; 4],
    /// Midtone marker per channel (rgb, red, green, blue), `-1.0 ..= 1.0`,
    /// 0 neutral; converted to gamma when building the adjustment set.
    pub levels_midtone: [f32; 4],
    /// Output black point per channel (rgb, red, green, blue), `0.0 ..= 1.0`.
    pub levels_target_shadow: [f32; 4],
    /// Output white point per channel (rgb, red, green, blue), `0.0 ..= 1.0`.
    pub levels_target_highlight: [f32; 4],

    pub clarity: f32,
    pub clarity_structure: f32,
    pub clarity_method: i32,

    pub highlights_recovery: f32,
    pub shadows_recovery: f32,
    pub whites_recovery: f32,
    pub blacks_recovery: f32,

    pub sharpening: C1Sharpening,
    pub noise_reduction: C1NoiseReduction,
    pub film_grain: C1FilmGrain,
    pub moire: [f32; 2],
    pub dehaze_amount: f32,
    pub dehaze_color: [f32; 3],

    pub black_white: C1BlackWhite,
    /// Master, shadow, midtone and highlight colour balance, three components each.
    pub color_balance: [[f32; 3]; 4],
    /// The nine `ZCOLORCORRECTIONS` groups.
    pub color_corrections: Vec<[f32; 18]>,
    /// Gradation curves: rgb, red, green, blue, yellow.
    pub gradation_curves: [Vec<(f32, f32)>; 5],

    /// Crop rectangle in pixels: `x1;y1;x2;y2`.
    pub crop_px: [f32; 4],
    pub crop_outside_image: bool,
    pub crop_aspect_ratio: Option<String>,
    pub rotation: f32,
    pub keystone_amount: f32,
    pub keystone_aspect: f32,
    pub keystone_tilt_x: f32,
    pub keystone_tilt_y: f32,
    pub geometry_canvas_clipping: i32,
    /// Canvas size in pixels, `w;h`; `0` when C1 did not store it on the layer.
    pub canvas_size: [f32; 2],

    pub lens: C1Lens,
    pub vignetting: [f32; 4],
    pub icc_profile: Option<String>,
    pub film_curve: Option<String>,

    pub brush: C1Brush,
    pub eraser: C1Eraser,
    pub surface_smoothness: f32,
}

/// C1's sharpening triple.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C1Sharpening {
    pub amount: f32,
    pub radius: f32,
    pub threshold: f32,
    pub halo_suppression: f32,
}

impl Default for C1Sharpening {
    fn default() -> Self {
        Self { amount: 0.0, radius: 0.8, threshold: 1.0, halo_suppression: 0.0 }
    }
}

/// C1's noise reduction amounts.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C1NoiseReduction {
    pub luminance: f32,
    pub details: f32,
    pub color: f32,
    pub method: i32,
    /// `ZCLEANLONGEXPOSUREAMOUNT`, kept separate from `color`.
    pub long_exposure: f32,
    /// `ZCNRAMOUNT`; C1's split between this and `color` is unproven.
    pub cnr: f32,
}

impl Default for C1NoiseReduction {
    fn default() -> Self {
        Self { luminance: 50.0, details: 50.0, color: 50.0, method: 0, long_exposure: 0.0, cnr: 0.0 }
    }
}

/// C1's film grain settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C1FilmGrain {
    pub amount: f32,
    pub density: f32,
    pub granularity: f32,
    pub kind: i32,
}

impl Default for C1FilmGrain {
    fn default() -> Self {
        Self { amount: 0.0, density: 0.0, granularity: 50.0, kind: 0 }
    }
}

/// C1's black & white settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C1BlackWhite {
    pub enabled: bool,
    /// red, yellow, green, cyan, blue, magenta.
    pub sensitivity: [f32; 6],
    pub dark_tone_hue: f32,
    pub dark_tone_saturation: f32,
    pub light_tone_hue: f32,
    pub light_tone_saturation: f32,
}

impl Default for C1BlackWhite {
    fn default() -> Self {
        Self {
            enabled: false,
            sensitivity: [0.0; 6],
            dark_tone_hue: 0.0,
            dark_tone_saturation: 0.0,
            light_tone_hue: 0.0,
            light_tone_saturation: 0.0,
        }
    }
}

/// C1's lens calibration block.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct C1Lens {
    /// `ZLENSDISTORTION`: radial spline plus amount / flag / amount / centre.
    pub distortion: SplineProfile,
    /// `ZLENSLIGHTFALLOFF`: radial spline plus amount / flag.
    pub light_falloff: SplineProfile,
    /// `ZLENSDEFAULTCHROMABR`: two radial splines (red and blue).
    pub chromatic_aberration: SplineProfile,
    pub chromatic_aberration_enabled: bool,
    pub profile: Option<String>,
    pub revision_profile: Option<String>,
    pub aperture: f32,
    pub focal_length: f32,
    /// Manual optical centre shift, `x;y` in percent.
    pub shift_xy: [f32; 2],
    pub purple_fringing: f32,
}

/// C1's brush tool settings (local mask layers).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C1Brush {
    pub size: f32,
    pub hardness: f32,
    pub opacity: f32,
    pub flow: f32,
    pub hair_brush: bool,
    pub use_pen_pressure: bool,
    pub auto_mask: bool,
}

impl Default for C1Brush {
    fn default() -> Self {
        Self {
            size: 100.0,
            hardness: 80.0,
            opacity: 100.0,
            flow: 100.0,
            hair_brush: false,
            use_pen_pressure: false,
            auto_mask: false,
        }
    }
}

/// C1's eraser tool settings (local mask layers).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct C1Eraser {
    pub size: f32,
    pub hardness: f32,
    pub opacity: f32,
    pub flow: f32,
    pub air_brush: bool,
    pub use_pen_pressure: bool,
    pub auto_mask: bool,
}

impl Default for C1Eraser {
    fn default() -> Self {
        Self {
            size: 100.0,
            hardness: 80.0,
            opacity: 100.0,
            flow: 100.0,
            air_brush: false,
            use_pen_pressure: false,
            auto_mask: false,
        }
    }
}

impl C1VariantSettings {
    /// All-neutral settings: what an untouched C1 variant layer stores.
    pub fn defaults() -> Self {
        Self {
            entity: C1EntityKind::DefaultLayer,
            index: 0,
            enabled: true,
            name: None,
            opacity: 100.0,
            mask_uuid: None,

            exposure: 0.0,
            contrast: 0.0,
            brightness: 0.0,
            saturation: 0.0,
            white_balance_gains: [1.0, 1.0, 1.0],

            levels_shadow: [0.0; 4],
            levels_highlight: [1.0; 4],
            levels_midtone: [0.0; 4],
            levels_target_shadow: [0.0; 4],
            levels_target_highlight: [1.0; 4],

            clarity: 0.0,
            clarity_structure: 0.0,
            clarity_method: 3,

            highlights_recovery: 0.0,
            shadows_recovery: 0.0,
            whites_recovery: 0.0,
            blacks_recovery: 0.0,

            sharpening: C1Sharpening::default(),
            noise_reduction: C1NoiseReduction::default(),
            film_grain: C1FilmGrain::default(),
            moire: [0.0, 8.0],
            dehaze_amount: 0.0,
            dehaze_color: [0.0, 0.0, 0.0],

            black_white: C1BlackWhite::default(),
            color_balance: [[1.0, 1.0, 1.0]; 4],
            color_corrections: Vec::new(),
            gradation_curves: [Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()],

            crop_px: [0.0; 4],
            crop_outside_image: false,
            crop_aspect_ratio: None,
            rotation: 0.0,
            keystone_amount: 100.0,
            keystone_aspect: 0.0,
            keystone_tilt_x: 0.0,
            keystone_tilt_y: 0.0,
            geometry_canvas_clipping: 2,
            canvas_size: [0.0, 0.0],

            lens: C1Lens {
                distortion: SplineProfile::default(),
                light_falloff: SplineProfile::default(),
                chromatic_aberration: SplineProfile::default(),
                chromatic_aberration_enabled: false,
                profile: None,
                revision_profile: None,
                aperture: 4.0,
                focal_length: 120.0,
                shift_xy: [0.0, 0.0],
                purple_fringing: 0.0,
            },
            vignetting: [0.0, 1.0, 0.0, 1.0],
            icc_profile: None,
            film_curve: None,

            brush: C1Brush::default(),
            eraser: C1Eraser::default(),
            surface_smoothness: 0.0,
        }
    }
}

/// Builds the settings from one row of `ZVARIANTLAYER`.
///
/// Columns that are absent or `NULL` keep C1's own neutral value, so a delta row
/// (`Z_ENT` 17) can be read with the same function as a full snapshot.
pub fn from_layer_row(row: &HashMap<&str, Option<String>>) -> C1VariantSettings {
    let get = |key: &str| -> Option<&str> { row.get(key).and_then(|v| v.as_deref()).map(clean) };
    let mut s = C1VariantSettings::defaults();

    s.entity = get("Z_ENT")
        .and_then(parse_scalar)
        .map(|v| v as i64)
        .and_then(C1EntityKind::from_ent)
        .unwrap_or(s.entity);
    s.index = get("ZINDEX").and_then(parse_scalar).map(|v| v as i64).unwrap_or(s.index);
    s.enabled = flag(get("ZENABLED"), s.enabled);
    s.opacity = fixed(get("ZOPACITY"), s.opacity);
    s.name = text(get("ZNAME"));
    s.mask_uuid = text(get("ZMASKUUID"));

    s.exposure = fixed(get("ZEXPOSURE"), s.exposure);
    s.contrast = fixed(get("ZCONTRAST"), s.contrast);
    s.brightness = fixed(get("ZBRIGHTNESS"), s.brightness);
    s.saturation = fixed(get("ZSATURATION"), s.saturation);

    let gains = components(get("ZWHITEBALANCE"), 3, 1.0);
    s.white_balance_gains = [gains[0], gains[1], gains[2]];

    let shadow = components(get("ZLEVELSSHADOW"), 4, 0.0);
    s.levels_shadow = [shadow[0], shadow[1], shadow[2], shadow[3]];
    let highlight = components(get("ZLEVELSHIGHLIGHT"), 4, 1.0);
    s.levels_highlight = [highlight[0], highlight[1], highlight[2], highlight[3]];
    let midtone = components(get("ZMIDTONE"), 4, 0.0);
    s.levels_midtone = [midtone[0], midtone[1], midtone[2], midtone[3]];
    let target_shadow = components(get("ZLEVELSTARGETSHADOW"), 4, 0.0);
    s.levels_target_shadow = [target_shadow[0], target_shadow[1], target_shadow[2], target_shadow[3]];
    let target_highlight = components(get("ZLEVELSTARGETHIGHLIGHT"), 4, 1.0);
    s.levels_target_highlight =
        [target_highlight[0], target_highlight[1], target_highlight[2], target_highlight[3]];

    s.clarity = fixed(get("ZCLARITY"), s.clarity);
    s.clarity_structure = fixed(get("ZCLARITYSTRUCTURE"), s.clarity_structure);
    s.clarity_method = fixed(get("ZCLARITYMETHOD"), s.clarity_method as f32) as i32;

    s.highlights_recovery = fixed(get("ZHIGHLIGHTRECOVERY"), s.highlights_recovery);
    s.shadows_recovery = fixed(get("ZSHADOWRECOVERY"), s.shadows_recovery);
    s.whites_recovery = fixed(get("ZWHITERECOVERY"), s.whites_recovery);
    s.blacks_recovery = fixed(get("ZBLACKRECOVERY"), s.blacks_recovery);

    s.sharpening.amount = fixed(get("ZUSMAMOUNT"), s.sharpening.amount);
    s.sharpening.radius = fixed(get("ZUSMRADIUS"), s.sharpening.radius);
    s.sharpening.threshold = fixed(get("ZUSMTHRESHOLD"), s.sharpening.threshold);
    s.sharpening.halo_suppression = fixed(get("ZUSMHALOCONTROL"), s.sharpening.halo_suppression);

    s.noise_reduction.luminance = fixed(get("ZNOISEREDUCTIONAMOUNT"), s.noise_reduction.luminance);
    s.noise_reduction.details = fixed(get("ZNOISEDETAILSAMOUNT"), s.noise_reduction.details);
    s.noise_reduction.color = fixed(get("ZAIDENOISEAMOUNT"), s.noise_reduction.color);
    s.noise_reduction.method = fixed(get("ZDENOISEMETHOD"), s.noise_reduction.method as f32) as i32;
    s.noise_reduction.long_exposure =
        fixed(get("ZCLEANLONGEXPOSUREAMOUNT"), s.noise_reduction.long_exposure);
    s.noise_reduction.cnr = fixed(get("ZCNRAMOUNT"), s.noise_reduction.cnr);

    s.film_grain.amount = fixed(get("ZFILMGRAINAMOUNT"), s.film_grain.amount);
    s.film_grain.density = fixed(get("ZFILMGRAINDENSITY"), s.film_grain.density);
    s.film_grain.granularity = fixed(get("ZFILMGRAINGRANULARITY"), s.film_grain.granularity);
    s.film_grain.kind = fixed(get("ZFILMGRAINTYPE"), s.film_grain.kind as f32) as i32;

    let moire = components(get("ZMOIRE"), 2, 0.0);
    s.moire = [moire[0], if moire[1] == 0.0 { 8.0 } else { moire[1] }];

    s.dehaze_amount = fixed(get("ZDEHAZEAMOUNT"), s.dehaze_amount);
    let dehaze = components(get("ZDEHAZECOLOR"), 3, 0.0);
    s.dehaze_color = [dehaze[0], dehaze[1], dehaze[2]];

    s.black_white.enabled = flag(get("ZBWENABLED"), s.black_white.enabled);
    s.black_white.sensitivity = [
        fixed(get("ZBWRED"), 0.0),
        fixed(get("ZBWYELLOW"), 0.0),
        fixed(get("ZBWGREEN"), 0.0),
        fixed(get("ZBWCYAN"), 0.0),
        fixed(get("ZBWBLUE"), 0.0),
        fixed(get("ZBWMAGENTA"), 0.0),
    ];
    s.black_white.dark_tone_hue = fixed(get("ZBWDARKTONEHUE"), s.black_white.dark_tone_hue);
    s.black_white.dark_tone_saturation =
        fixed(get("ZBWDARKTONESATURATION"), s.black_white.dark_tone_saturation);
    s.black_white.light_tone_hue = fixed(get("ZBWLIGHTTONEHUE"), s.black_white.light_tone_hue);
    s.black_white.light_tone_saturation =
        fixed(get("ZBWLIGHTTONESATURATION"), s.black_white.light_tone_saturation);

    for (slot, key) in s.color_balance.iter_mut().zip([
        "ZCOLORBALANCE",
        "ZCOLORBALANCESHADOW",
        "ZCOLORBALANCEMIDTONE",
        "ZCOLORBALANCEHIGHLIGHT",
    ]) {
        let v = components(get(key), 3, 1.0);
        *slot = [v[0], v[1], v[2]];
    }

    if let Some(raw) = get("ZCOLORCORRECTIONS") {
        s.color_corrections = parse_color_corrections(raw);
    }

    for (slot, key) in s.gradation_curves.iter_mut().zip([
        "ZGRADATIONCURVE",
        "ZGRADATIONCURVERED",
        "ZGRADATIONCURVEGREEN",
        "ZGRADATIONCURVEBLUE",
        "ZGRADATIONCURVEY",
    ]) {
        if let Some(raw) = get(key) {
            *slot = parse_curve(raw);
        }
    }

    let crop = components(get("ZCROP"), 4, 0.0);
    s.crop_px = [crop[0], crop[1], crop[2], crop[3]];
    s.crop_outside_image = flag(get("ZCROPOUTSIDEIMAGE"), s.crop_outside_image);
    s.crop_aspect_ratio = text(get("ZCROPASPECTRATIONAME"));
    s.rotation = fixed(get("ZROTATION"), s.rotation);
    s.keystone_amount = fixed(get("ZGEOMETRYKEYSTONEAMOUNT"), s.keystone_amount);
    s.keystone_aspect = fixed(get("ZGEOMETRYKEYSTONEASPECT"), s.keystone_aspect);
    s.keystone_tilt_x = fixed(get("ZGEOMETRYKEYSTONETILTX"), s.keystone_tilt_x);
    s.keystone_tilt_y = fixed(get("ZGEOMETRYKEYSTONETILTY"), s.keystone_tilt_y);
    s.geometry_canvas_clipping =
        fixed(get("ZGEOMETRYCANVASCLIPPING"), s.geometry_canvas_clipping as f32) as i32;
    let canvas = components(get("ZCANVASSIZE"), 2, 0.0);
    s.canvas_size = [canvas[0], canvas[1]];

    s.lens.distortion = get("ZLENSDISTORTION").map(parse_spline).unwrap_or_default();
    s.lens.light_falloff = get("ZLENSLIGHTFALLOFF").map(parse_spline).unwrap_or_default();
    s.lens.chromatic_aberration = get("ZLENSDEFAULTCHROMABR").map(parse_spline).unwrap_or_default();
    s.lens.chromatic_aberration_enabled =
        flag(get("ZLENSCHROMABRENABLED"), s.lens.chromatic_aberration_enabled);
    s.lens.profile = text(get("ZLENSIDGUID"));
    s.lens.revision_profile = text(get("ZLENSREVISIONIDGUID"));
    s.lens.aperture = fixed(get("ZLENSAPERTURE"), s.lens.aperture);
    s.lens.focal_length = fixed(get("ZLENSFOCALLENGTH"), s.lens.focal_length);
    let shift = components(get("ZLENSSHIFTXY"), 2, 0.0);
    s.lens.shift_xy = [shift[0], shift[1]];
    s.lens.purple_fringing = fixed(get("ZLENSPURPLEFRINGINGAMOUNT"), s.lens.purple_fringing);

    let vig = components(get("ZLENSVIGNETTING"), 4, 0.0);
    s.vignetting = [vig[0], vig[1], vig[2], vig[3]];

    s.icc_profile = text(get("ZICCPROFILE"));
    s.film_curve = text(get("ZFILMCURVE"));

    s.brush.size = fixed(get("ZBRUSHSIZE"), s.brush.size);
    s.brush.hardness = fixed(get("ZBRUSHHARDNESS"), s.brush.hardness);
    s.brush.opacity = fixed(get("ZBRUSHOPACITY"), s.brush.opacity);
    s.brush.flow = fixed(get("ZBRUSHFLOW"), s.brush.flow);
    s.brush.hair_brush = flag(get("ZBRUSHAIRBRUSH"), s.brush.hair_brush);
    s.brush.use_pen_pressure = flag(get("ZBRUSHUSEPENPRESSURE"), s.brush.use_pen_pressure);
    s.brush.auto_mask = flag(get("ZBRUSHAUTOMASK"), s.brush.auto_mask);

    s.eraser.size = fixed(get("ZERASERSIZE"), s.eraser.size);
    s.eraser.hardness = fixed(get("ZERASERHARDNESS"), s.eraser.hardness);
    s.eraser.opacity = fixed(get("ZERASEROPACITY"), s.eraser.opacity);
    s.eraser.flow = fixed(get("ZERASERFLOW"), s.eraser.flow);
    s.eraser.air_brush = flag(get("ZERASERAIRBRUSH"), s.eraser.air_brush);
    s.eraser.use_pen_pressure = flag(get("ZERASERUSEPENPRESSURE"), s.eraser.use_pen_pressure);
    s.eraser.auto_mask = flag(get("ZERASERAUTOMASK"), s.eraser.auto_mask);

    s.surface_smoothness = fixed(get("ZSURFACESMOOTHNESSAMOUNT"), s.surface_smoothness);
    s
}

// ─────────────────────────────────────────────────────────────────────────────
// White balance conversion
// ─────────────────────────────────────────────────────────────────────────────

/// Reference temperature the gain model is anchored at, in Kelvin.
const WB_REFERENCE_KELVIN: f32 = 6500.0;
/// Scale from log-gain to the -50 ..= 50 tint axis.
const WB_TINT_SCALE: f32 = 100.0;

/// Approximates the (kelvin, tint) pair behind C1's RGB gains.
///
/// C1 stores relative channel gains with green pinned at 1; this inverts that
/// with a log-ratio model, not C1's real colour transform: the kelvin axis is
/// `6500 * exp(-(ln r - ln b))` and the tint axis measures how far green sits
/// below the red/blue average. Expect it to be close, not identical, to C1.
pub fn gains_to_kelvin_tint(gains: [f32; 3]) -> (f32, f32) {
    let (r, g, b) = (gains[0], gains[1], gains[2]);
    if !(g > 0.0) || !r.is_finite() || !b.is_finite() {
        return (WB_REFERENCE_KELVIN, 0.0);
    }
    let lr = (r / g).ln();
    let lb = (b / g).ln();
    let kelvin = WB_REFERENCE_KELVIN * (-(lr - lb)).exp();
    let tint = -0.5 * (lr + lb) * WB_TINT_SCALE;
    (kelvin, tint)
}

/// Approximates RGB gains for a (kelvin, tint) pair; the inverse of
/// [`gains_to_kelvin_tint`] under the same log-ratio model.
pub fn kelvin_tint_to_gains(kelvin: f32, tint: f32) -> [f32; 3] {
    let kelvin = kelvin.clamp(1000.0, 40000.0);
    let d = -(kelvin / WB_REFERENCE_KELVIN).ln() / 2.0;
    let m = -tint / WB_TINT_SCALE;
    [(m + d).exp(), 1.0, (m - d).exp()]
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversion to the engine's AdjustmentSet
// ─────────────────────────────────────────────────────────────────────────────

fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    if v.is_nan() {
        lo
    } else {
        v.clamp(lo, hi)
    }
}

fn curve_from(points: &[(f32, f32)]) -> Curve {
    if points.is_empty() {
        return Curve::default();
    }
    Curve { points: points.iter().map(|(x, y)| CurvePoint::new(*x, *y)).collect() }
}

fn rgb_to_hue(r: f32, g: f32, b: f32) -> f32 {
    let (max, min) = (r.max(g).max(b), r.min(g).min(b));
    let d = max - min;
    if d.abs() < 1e-6 {
        return 0.0;
    }
    let h = if max == r {
        60.0 * (((g - b) / d) % 6.0)
    } else if max == g {
        60.0 * ((b - r) / d + 2.0)
    } else {
        60.0 * ((r - g) / d + 4.0)
    };
    (h + 360.0).rem_euclid(360.0)
}

/// Maps C1's stored settings onto the engine's adjustment contract.
/// C1's `ZCOLORCORRECTIONS` / `ColorCorrections` rows as the engine's editor.
///
/// C1 stores its eight basic slices in descending hue order, not in the
/// engine's Red..Magenta order, so each group is placed by its own color.
/// `local` separates the global group from slices; `kind` keeps advanced and
/// skin groups out of the eight basic slots.
pub fn color_editor_from_c1(rows: &[[f32; 18]]) -> ColorEditor {
    let mut editor = ColorEditor::default();
    let mut basic_used = [false; 8];
    for (index, row) in rows.iter().enumerate() {
        let cc = ColorCorrection::from_row(row);
        let (center_hue, center_saturation) = selected_hue_saturation(cc.rgb);
        let center_value = (cc.rgb[0].max(cc.rgb[1]).max(cc.rgb[2]) / 255.0).clamp(0.0, 1.0);
        let band = ColorBand {
            center_hue,
            center_saturation,
            center_value,
            hue_low: cc.hue_low,
            hue_high: cc.hue_high,
            sat_low: cc.sat_low,
            sat_high: cc.sat_high,
            smoothness: clamp(cc.smoothness, 0.0, 60.0),
            homogeneity: cc.homogeneity,
            inverted: cc.inverted,
            enabled: cc.enabled,
        };
        let shift = HslShift {
            hue: clamp(cc.hue, -30.0, 30.0),
            saturation: clamp(cc.saturation, -100.0, 100.0),
            // C1 stores ValueChange at 1/10 of the UI slider.
            lightness: clamp(cc.lightness * 10.0, -100.0, 100.0),
        };

        if !cc.local {
            editor.global = band;
            editor.global_shift = shift;
            continue;
        }

        if cc.kind == 1 {
            let slot = HueSector::nearest(center_hue) as usize;
            if !basic_used[slot] {
                editor.basic[slot] = shift;
                editor.bands[slot] = band;
                basic_used[slot] = true;
            }
        } else {
            let id = index as u64 + 1;
            let skin = cc.kind == 2;
            let range = ColorRange {
                id,
                hue: center_hue,
                width: (cc.hue_high - cc.hue_low).abs().max(1.0),
                smoothness: clamp(cc.smoothness, 0.0, 60.0),
                min_saturation: ((center_saturation * 100.0 + cc.sat_low) / 100.0).clamp(0.0, 1.0),
                shift,
                uniformity: skin.then(|| SkinUniformity {
                    hue: cc.homogeneity[0],
                    saturation: cc.homogeneity[1],
                    lightness: cc.homogeneity[2],
                }),
            };
            if skin {
                editor.skin.push(range);
            } else {
                editor.advanced.push(range);
            }
            editor.advanced_bands.push((id, band));
        }
    }
    editor
}

pub fn to_adjustment_set(s: &C1VariantSettings) -> crate::AdjustmentSet {
    let mut set = crate::AdjustmentSet::default();

    set.exposure = Exposure {
        exposure: clamp(s.exposure, -4.0, 4.0),
        contrast: clamp(s.contrast, -50.0, 50.0),
        brightness: clamp(s.brightness, -100.0, 100.0),
        saturation: clamp(s.saturation, -100.0, 100.0),
    };

    let (kelvin, tint) = gains_to_kelvin_tint(s.white_balance_gains);
    set.white_balance = WhiteBalance {
        mode: WhiteBalanceMode::Custom,
        kelvin: clamp(kelvin, 2000.0, 50000.0),
        tint: clamp(tint, -50.0, 50.0),
        skin_tone: None,
    };

    set.hdr = HighDynamicRange {
        highlights: clamp(s.highlights_recovery, -100.0, 100.0),
        shadows: clamp(s.shadows_recovery, -100.0, 100.0),
        whites: clamp(s.whites_recovery, -100.0, 100.0),
        blacks: clamp(s.blacks_recovery, -100.0, 100.0),
    };

    let channel = |i: usize| LevelsChannel {
        input_black: clamp(s.levels_shadow[i], 0.0, 1.0),
        input_white: clamp(s.levels_highlight[i], 0.0, 1.0),
        gamma: gamma_from_c1_midtone(s.levels_midtone[i]),
        output_black: clamp(s.levels_target_shadow[i], 0.0, 1.0),
        output_white: clamp(s.levels_target_highlight[i], 0.0, 1.0),
    };
    set.levels = Levels {
        rgb: channel(0),
        red: channel(1),
        green: channel(2),
        blue: channel(3),
    };

    set.curves = Curves {
        rgb: curve_from(&s.gradation_curves[0]),
        red: curve_from(&s.gradation_curves[1]),
        green: curve_from(&s.gradation_curves[2]),
        blue: curve_from(&s.gradation_curves[3]),
        // ZGRADATIONCURVEY has no counterpart: the engine's luma curve stays empty.
        luma: Curve::default(),
    };

    set.clarity = Clarity {
        // C1's ZCLARITYMETHOD numbering is not proven to match the engine's enum.
        method: match s.clarity_method {
            0 => ClarityMethod::Natural,
            1 => ClarityMethod::Punch,
            2 => ClarityMethod::Neutral,
            3 => ClarityMethod::Classic,
            _ => ClarityMethod::Natural,
        },
        clarity: clamp(s.clarity, -100.0, 100.0),
        structure: clamp(s.clarity_structure, -100.0, 100.0),
    };

    set.dehaze = Dehaze {
        amount: clamp(s.dehaze_amount, 0.0, 100.0),
        shadow_tone: [s.dehaze_color[0], s.dehaze_color[1], s.dehaze_color[2]],
    };

    set.vignetting = Vignetting {
        amount: clamp(s.vignetting[0], -4.0, 4.0),
        method: VignetteMethod::EllipticOnCrop,
    };

    // shortcut: C1's three numbers per colour-balance wheel are unproven — they may
    // be RGB multipliers, not hue/saturation/lightness. Feeding the neutral 1;1;1
    // through the wheel turns every catalog image red, so the mapping stays off
    // until a sample with a non-neutral balance proves the encoding.
    let _ = &s.color_balance;

    set.color_editor = color_editor_from_c1(&s.color_corrections);

    set.black_white = BlackWhite {
        enabled: s.black_white.enabled,
        sensitivity: std::array::from_fn(|i| {
            crate::param_meta::clamp(crate::param_meta::BW_CHANNELS[i], s.black_white.sensitivity[i])
        }),
        highlight_tone: SplitTone {
            hue: clamp(s.black_white.light_tone_hue, 0.0, 360.0),
            saturation: clamp(s.black_white.light_tone_saturation, 0.0, 100.0),
        },
        shadow_tone: SplitTone {
            hue: clamp(s.black_white.dark_tone_hue, 0.0, 360.0),
            saturation: clamp(s.black_white.dark_tone_saturation, 0.0, 100.0),
        },
    };

    set.sharpening = Sharpening {
        amount: clamp(s.sharpening.amount, 0.0, 1000.0),
        radius: clamp(s.sharpening.radius, 0.2, 2.5),
        // The catalog row is C1's own 0..12; the old engine cap of 10 clipped
        // legal 11–12 values.
        threshold: crate::param_meta::clamp("USMThreshold", s.sharpening.threshold),
        halo_suppression: clamp(s.sharpening.halo_suppression, 0.0, 100.0),
    };

    set.noise_reduction = NoiseReduction {
        luminance: clamp(s.noise_reduction.luminance, 0.0, 100.0),
        details: clamp(s.noise_reduction.details, 0.0, 100.0),
        color: clamp(s.noise_reduction.color, 0.0, 100.0),
        // ZCLEANLONGEXPOSUREAMOUNT has no field of its own; it is dropped here.
        single_pixel: 0.0,
    };

    set.film_grain = FilmGrain {
        kind: match s.film_grain.kind {
            0 => GrainKind::FineGrain,
            1 => GrainKind::SilverRich,
            2 => GrainKind::SoftGrain,
            3 => GrainKind::CubicGrain,
            4 => GrainKind::TabularGrain,
            5 => GrainKind::HarshGrain,
            _ => GrainKind::FineGrain,
        },
        impact: clamp(s.film_grain.amount, 0.0, 100.0),
        granularity: clamp(s.film_grain.granularity, 0.0, 100.0),
    };

    set.moire = Moire {
        amount: clamp(s.moire[0], 0.0, 100.0),
        pattern: clamp(s.moire[1], 0.0, 100.0),
    };

    set.lens = LensCorrection {
        profile: s.lens.profile.clone(),
        // shortcut: C1 stores a lens-profile GUID we cannot resolve to its LCP data,
        // so enabling a warp from a guessed profile only crops the frame.
        // A variant that names a lens GUID asks for that lens's correction;
        // a GUID the database does not know leaves the frame unchanged.
        use_profile: s.lens.profile.is_some(),
        distortion: clamp(s.lens.distortion.param(0).unwrap_or(0.0), 0.0, 100.0),
        // ZLENSLIGHTFALLOFF covers light falloff only; the engine's separate
        // sharpness-falloff amount has no C1 column and stays at zero.
        sharpness_falloff: 0.0,
        light_falloff: clamp(s.lens.light_falloff.param(0).unwrap_or(0.0), 0.0, 100.0),
        chromatic_aberration: s.lens.chromatic_aberration_enabled,
        purple_fringing: clamp(s.lens.purple_fringing, 0.0, 100.0),
        hide_distorted_areas: true,
        shift_x: clamp(s.lens.shift_xy[0], -100.0, 100.0),
        shift_y: clamp(s.lens.shift_xy[1], -100.0, 100.0),
    };

    // A variant without ZCROP stores all four values as 0; that is "no crop",
    // not a zero-sized rectangle, even though the canvas size is present.
    let has_crop = s.crop_px[2] > s.crop_px[0] && s.crop_px[3] > s.crop_px[1];
    let crop = if has_crop && s.canvas_size[0] > 0.0 && s.canvas_size[1] > 0.0 {
        let (w, h) = (s.canvas_size[0], s.canvas_size[1]);
        NormRect {
            x: clamp(s.crop_px[0] / w, 0.0, 1.0),
            y: clamp(s.crop_px[1] / h, 0.0, 1.0),
            width: clamp((s.crop_px[2] - s.crop_px[0]) / w, 0.0, 1.0),
            height: clamp((s.crop_px[3] - s.crop_px[1]) / h, 0.0, 1.0),
        }
    } else {
        // Without the canvas size the pixel crop cannot be normalized; C1 keeps it
        // on ZVARIANT.ZCANVASSIZE, so the engine keeps a full frame here.
        NormRect::FULL
    };

    set.geometry = Geometry {
        rotation: split_rotation(s.rotation).0,
        orientation: split_rotation(s.rotation).1,
        flip_horizontal: false,
        flip_vertical: false,
        crop,
        aspect: match s.crop_aspect_ratio.as_deref() {
            Some("Original") => AspectRatio::Original,
            Some("Square") => AspectRatio::Square,
            Some("4x5") => AspectRatio::R4x5,
            Some("5x7") => AspectRatio::R5x7,
            Some("2x3") => AspectRatio::R2x3,
            Some("3x4") => AspectRatio::R3x4,
            Some("16x9") => AspectRatio::R16x9,
            Some("65x24") => AspectRatio::R65x24,
            _ => AspectRatio::Unconstrained,
        },
        portrait: false,
        keystone: Keystone {
            vertical: clamp(s.keystone_tilt_y, -100.0, 100.0),
            horizontal: clamp(s.keystone_tilt_x, -100.0, 100.0),
            amount: clamp(s.keystone_amount, 0.0, 100.0),
            aspect: clamp(s.keystone_aspect, -50.0, 50.0),
        },
    };

    // C1 stores file names (`NikonZ6-ProStandard.icm`, `NikonZ6-Auto.fcrv`),
    // but the renderer looks profiles and curves up by stem, so the extension
    // has to go or the camera profile and film curve are silently dropped.
    set.base.icc_profile = s.icc_profile.clone().filter(|p| p != "EMBEDDED").map(|p| stem_of(&p));
    set.base.film_curve = s.film_curve.clone().map(|p| stem_of(&p));
    set.base.curve = match s.film_curve.as_deref() {
        Some(name) if name.contains("ExtraShadow") => ToneCurvePreset::FilmExtraShadow,
        Some(name) if name.contains("HighContrast") => ToneCurvePreset::FilmHighContrast,
        Some(name) if name.contains("Standard") => ToneCurvePreset::FilmStandard,
        Some(name) if name.contains("Linear") => ToneCurvePreset::Linear,
        Some(name) if name.contains("Auto") => ToneCurvePreset::Auto,
        _ => ToneCurvePreset::Auto,
    };

    set
}

/// Capture One stores the display rotation in `ZROTATION`: quarter turns
/// (0/90/180/270), not the fine straighten angle. A value that is not a
/// multiple of 90 is treated as straighten degrees; C1's unit for those is
/// unproven, so they are clamped like any other fine rotation.
fn split_rotation(deg: f32) -> (f32, u8) {
    if !deg.is_finite() {
        return (0.0, 0);
    }
    let turns = (deg / 90.0).round();
    if (deg - turns * 90.0).abs() < 1e-3 {
        (0.0, turns.rem_euclid(4.0) as u8)
    } else {
        (clamp(deg, -45.0, 45.0), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(pairs: &[(&'static str, Option<&'static str>)]) -> HashMap<&'static str, Option<String>> {
        pairs.iter().map(|(k, v)| (*k, v.map(|s| s.to_string()))).collect()
    }

    #[test]
    fn a_quarter_turn_becomes_orientation_not_fine_rotation() {
        let s = from_layer_row(&row(&[("ZROTATION", Some("270"))]));
        let set = to_adjustment_set(&s);
        assert_eq!(set.geometry.orientation, 3);
        assert_eq!(set.geometry.rotation, 0.0);
    }

    #[test]
    fn a_straighten_angle_stays_fine_rotation() {
        let s = from_layer_row(&row(&[("ZROTATION", Some("12.5"))]));
        let set = to_adjustment_set(&s);
        assert_eq!(set.geometry.rotation, 12.5);
        assert_eq!(set.geometry.orientation, 0);
    }

    #[test]
    fn a_variant_without_a_crop_keeps_the_full_frame() {
        let mut s = from_layer_row(&row(&[]));
        s.canvas_size = [6000.0, 4000.0];
        let set = to_adjustment_set(&s);
        assert_eq!(set.geometry.crop, crate::NormRect::FULL);
    }

    #[test]
    fn column_table_covers_the_high_confidence_set() {
        for name in [
            "Z_ENT",
            "ZEXPOSURE",
            "ZCONTRAST",
            "ZBRIGHTNESS",
            "ZSATURATION",
            "ZCLARITY",
            "ZCLARITYSTRUCTURE",
            "ZCLARITYMETHOD",
            "ZUSMAMOUNT",
            "ZUSMRADIUS",
            "ZUSMTHRESHOLD",
            "ZUSMHALOCONTROL",
            "ZNOISEREDUCTIONAMOUNT",
            "ZNOISEDETAILSAMOUNT",
            "ZCROP",
            "ZCROPASPECTRATIONAME",
            "ZGEOMETRYKEYSTONEAMOUNT",
            "ZGEOMETRYKEYSTONEASPECT",
            "ZGEOMETRYKEYSTONETILTX",
            "ZGEOMETRYKEYSTONETILTY",
            "ZHIGHLIGHTRECOVERY",
            "ZSHADOWRECOVERY",
            "ZWHITERECOVERY",
            "ZBLACKRECOVERY",
            "ZDEHAZEAMOUNT",
            "ZFILMGRAINGRANULARITY",
            "ZFILMGRAINTYPE",
            "ZBWENABLED",
            "ZLENSCHROMABRENABLED",
            "ZLENSIDGUID",
            "ZICCPROFILE",
            "ZFILMCURVE",
            "ZOPACITY",
            "ZNAME",
            "ZMASKUUID",
        ] {
            assert!(column(name).is_some(), "missing column {name}");
        }
        assert!(C1_COLUMNS.len() >= 56);
        assert_eq!(column("ZWHITEBALANCE").unwrap().encoding, C1Encoding::Semicolon);
        assert_eq!(column("ZWHITEBALANCE").unwrap().components, 3);
        assert_eq!(column("ZCOLORCORRECTIONS").unwrap().encoding, C1Encoding::SemicolonComma);
        assert_eq!(column("ZGRADATIONCURVE").unwrap().encoding, C1Encoding::CommaPairs);
        assert_eq!(column("ZLENSVIGNETTING").unwrap().encoding, C1Encoding::Pipe);
    }

    #[test]
    fn parses_real_scalar_value() {
        assert_eq!(parse_scalar("-0.6492753624916077"), Some(-0.64927536));
        assert_eq!(parse_scalar("3.768115997314453"), Some(3.768116));
    }

    #[test]
    fn parses_real_semicolon_vector() {
        let v = parse_vec("1.625000;1.000000;1.412000");
        assert_eq!(v.len(), 3);
        assert!((v[0] - 1.625).abs() < 1e-6);
        assert!((v[1] - 1.0).abs() < 1e-6);
        assert!((v[2] - 1.412).abs() < 1e-6);
    }

    #[test]
    fn parses_real_curve_pairs() {
        let c = parse_curve("0.000,0.000;1.000,1.000");
        assert_eq!(c.len(), 2);
        assert!((c[1].0 - 1.0).abs() < 1e-6 && (c[1].1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn parses_real_pipe_spline() {
        let raw = "0.0,0.0000;742.0,-43.3392;23742.6,146.6017|100.0|1|100|-1.00000";
        let p = parse_spline(raw);
        assert_eq!(p.splines.len(), 1);
        assert_eq!(p.splines[0].len(), 3);
        assert_eq!(p.params.len(), 4);
        assert!((p.param(0).unwrap() - 100.0).abs() < 1e-6);
        assert!((p.param(3).unwrap() + 1.0).abs() < 1e-6);
    }

    #[test]
    fn parses_real_color_corrections() {
        let raw = "1,1,1,0,0,0,255,0,168.909881591797,-7.11462593078613,7.11462593078613,-100,100,15,0,0,0,0;\
                   1,1,1,0,0,0,170.93928527832,0,255,-17.8928756713867,17.8928756713867,-100,100,15,0,0,0,0";
        let groups = parse_color_corrections(raw);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].len(), 18);
        assert!((groups[0][6] - 255.0).abs() < 1e-6);
        assert!((groups[1][8] - 255.0).abs() < 1e-6);
    }

    /// Real catalog groups, in the order C1 stores them (descending hue).
    const REAL_COLOR_CORRECTIONS: &str = "\
1,1,1,0,0,0,255,0,168.909881591797,-7.11462593078613,7.11462593078613,-100,100,15,0,0,0,0;\
1,1,1,0,0,0,170.93928527832,0,255,-17.8928756713867,17.8928756713867,-100,100,15,0,0,0,0;\
1,1,1,0,0,0,0,64.2907180786133,255,-22.4273700714111,22.4273700714111,-100,100,15,0,0,0,0;\
1,1,1,0,0,0,0,255,236.14192199707,-11.8545951843262,11.8545951843262,-100,100,15,0,0,0,0;\
1,1,1,0,0,0,45.498851776123,255,0,-39.3857116699219,39.3857116699219,-100,100,15,0,0,0,0;\
1,1,1,0,0,0,255,211.99267578125,0,-5,5,-100,100,15,0,0,0,0;\
1,1,1,0,0,0,255,101.301910400391,0,-6.01692199707031,6.01692199707031,-100,100,15,0,0,0,0;\
1,1,1,0,0,0,255,0,31.4708557128906,-10.195873260498,10.195873260498,-100,100,15,0,0,0,0;\
0,1,1,0,0,0,0,128,0,-2,2,-100,100,0,0,0,0,0";

    #[test]
    fn decodes_every_field_of_a_real_group() {
        let groups = parse_color_corrections(REAL_COLOR_CORRECTIONS);
        let cc = ColorCorrection::from_row(&groups[0]);
        assert!(cc.local && cc.enabled);
        assert_eq!(cc.kind, 1, "group 0 is a basic slice");
        assert_eq!((cc.hue, cc.saturation, cc.lightness), (0.0, 0.0, 0.0));
        assert!((cc.rgb[0] - 255.0).abs() < 1e-6);
        assert!((cc.rgb[1] - 0.0).abs() < 1e-6);
        assert!((cc.rgb[2] - 168.90988).abs() < 1e-3);
        assert!((cc.hue_low + 7.1146259).abs() < 1e-5);
        assert!((cc.hue_high - 7.1146259).abs() < 1e-5);
        assert_eq!((cc.sat_low, cc.sat_high), (-100.0, 100.0));
        assert!((cc.smoothness - 15.0).abs() < 1e-6);
        assert_eq!(cc.homogeneity, [0.0; 3]);
        assert!(!cc.inverted);
    }

    #[test]
    fn sectors_are_placed_by_their_color_not_their_index() {
        let s = from_layer_row(&row(&[("ZCOLORCORRECTIONS", Some(REAL_COLOR_CORRECTIONS))]));
        let editor = &to_adjustment_set(&s).color_editor;
        // C1 stores magenta first and red last; the engine keeps Red..Magenta.
        assert!((editor.bands[HueSector::Magenta as usize].center_hue - 320.3).abs() < 1.0);
        assert!((editor.bands[HueSector::Red as usize].center_hue - 352.6).abs() < 1.0);
        assert!((editor.bands[HueSector::Green as usize].center_hue - 109.3).abs() < 1.0);
        assert!(editor.bands.iter().all(|b| (b.smoothness - 15.0).abs() < 1e-6));

        // The trailing group is the global correction, not a ninth sector.
        assert!((editor.global.center_hue - 120.0).abs() < 1e-3);
        assert_eq!(editor.global.smoothness, 0.0);
    }

    #[test]
    fn a_slice_shift_lands_on_its_own_sector() {
        // Give the stored red slice (group 7) a value on every axis.
        let mut raw = String::new();
        for (i, group) in REAL_COLOR_CORRECTIONS.split(';').enumerate() {
            if i == 7 {
                raw.push_str("1,1,1,12,-40,-3,255,0,31.4708557128906,-10.195873260498,10.195873260498,-100,100,15,0,0,0,0");
            } else {
                raw.push_str(group);
            }
            raw.push(';');
        }
        let raw: &'static str = Box::leak(raw.into_boxed_str());
        let s = from_layer_row(&row(&[("ZCOLORCORRECTIONS", Some(raw))]));
        let editor = &to_adjustment_set(&s).color_editor;
        let red = editor.basic[HueSector::Red as usize];
        assert!((red.hue - 12.0).abs() < 1e-6);
        assert!((red.saturation + 40.0).abs() < 1e-6);
        assert!((red.lightness + 30.0).abs() < 1e-6, "stored -3 is -30 on the UI slider");
        assert_eq!(editor.basic[HueSector::Orange as usize], HslShift::default());
        assert!((editor.bands[HueSector::Red as usize].center_hue - 352.6).abs() < 1.0);
    }

    #[test]
    fn kind_dispatch_keeps_advanced_and_skin_out_of_basic_slots() {
        let rgb = [
            [255.0, 0.0, 0.0],
            [255.0, 128.0, 0.0],
            [255.0, 255.0, 0.0],
            [0.0, 255.0, 0.0],
            [0.0, 255.0, 255.0],
            [0.0, 0.0, 255.0],
            [128.0, 0.0, 255.0],
            [255.0, 0.0, 255.0],
        ];
        let mut groups: Vec<String> = rgb
            .iter()
            .map(|c| {
                format!(
                    "1,1,1,0,-10,0,{},{},{},-5,5,-100,100,3,0,0,0,0",
                    c[0], c[1], c[2]
                )
            })
            .collect();
        groups.push("1,1,0,0,-90,0,0,0,255,-10,10,-100,100,4,0,0,0,0".into());
        groups.push("1,1,2,0,-20,0,200,120,80,-8,8,-100,100,6,80,0,70,60".into());
        groups.push("1,1,3,0,10,0,0,255,0,-9,9,-100,100,29.1,0,0,0,0".into());
        groups.push("0,1,1,5,0,0,255,0,0,-2,2,-100,100,0,0,0,0,0".into());
        let raw: &'static str = Box::leak(groups.join(";").into_boxed_str());
        let s = from_layer_row(&row(&[("ZCOLORCORRECTIONS", Some(raw))]));
        let ce = &to_adjustment_set(&s).color_editor;

        assert_eq!(ce.basic.iter().filter(|s| s.saturation != 0.0).count(), 8);
        assert_eq!(ce.basic[HueSector::Blue as usize].saturation, -10.0);
        assert_eq!(ce.advanced.len(), 2, "kind 0 and kind 3 are advanced");
        assert_eq!(ce.skin.len(), 1, "kind 2 is skin");
        assert_eq!(ce.advanced_bands.len(), 3);
        assert_eq!(ce.skin[0].uniformity.expect("skin uniformity").lightness, 60.0);
        assert!((ce.global_shift.hue - 5.0).abs() < 1e-6);
    }

    #[test]
    fn local_flag_finds_the_global_group_at_any_position() {
        let mut groups = vec!["0,1,1,5,0,0,255,0,0,-2,2,-100,100,0,0,0,0,0".to_string()];
        let rgb = [
            [255.0, 0.0, 0.0],
            [255.0, 128.0, 0.0],
            [255.0, 255.0, 0.0],
            [0.0, 255.0, 0.0],
            [0.0, 255.0, 255.0],
            [0.0, 0.0, 255.0],
            [128.0, 0.0, 255.0],
            [255.0, 0.0, 255.0],
        ];
        for c in rgb {
            groups.push(format!(
                "1,1,1,0,-10,0,{},{},{},-5,5,-100,100,3,0,0,0,0",
                c[0], c[1], c[2]
            ));
        }
        let raw: &'static str = Box::leak(groups.join(";").into_boxed_str());
        let s = from_layer_row(&row(&[("ZCOLORCORRECTIONS", Some(raw))]));
        let ce = &to_adjustment_set(&s).color_editor;
        assert!((ce.global_shift.hue - 5.0).abs() < 1e-6);
        assert_eq!(ce.basic.iter().filter(|s| s.saturation != 0.0).count(), 8);
        assert_eq!(ce.basic[HueSector::Red as usize].saturation, -10.0);
    }

    #[test]
    fn empty_and_broken_values_are_safe() {
        assert_eq!(parse_scalar(""), None);
        assert_eq!(parse_scalar("NaN"), None);
        assert_eq!(parse_scalar("'abc'"), None);
        assert!(parse_vec("").is_empty());
        assert!(parse_curve("").is_empty());
        assert!(parse_spline("").splines.is_empty());
        assert!(parse_color_corrections("").is_empty());
        let s = from_layer_row(&row(&[("ZEXPOSURE", Some(""))]));
        assert_eq!(s.exposure, 0.0);
    }

    #[test]
    fn lens_profile_guid_enables_calibration_only_when_present() {
        let present = to_adjustment_set(&from_layer_row(&row(&[
            ("ZLENSIDGUID", Some("EF11-24-GUID")),
        ])));
        assert!(present.lens.use_profile);
        assert_eq!(present.lens.profile.as_deref(), Some("EF11-24-GUID"));

        let missing = to_adjustment_set(&from_layer_row(&row(&[])));
        assert!(!missing.lens.use_profile);
        assert_eq!(missing.lens.profile, None);
    }

    #[test]
    fn missing_components_fall_back_to_defaults() {
        let s = from_layer_row(&row(&[("ZWHITEBALANCE", Some("1.625000"))]));
        assert!((s.white_balance_gains[0] - 1.625).abs() < 1e-6);
        assert!((s.white_balance_gains[1] - 1.0).abs() < 1e-6);
        assert!((s.white_balance_gains[2] - 1.0).abs() < 1e-6);
        let s = from_layer_row(&row(&[("ZLEVELSSHADOW", Some("0.5"))]));
        assert!((s.levels_shadow[0] - 0.5).abs() < 1e-6);
        assert_eq!(s.levels_shadow[3], 0.0);
    }

    #[test]
    fn levels_midtone_is_converted_to_gamma() {
        let s = from_layer_row(&row(&[
            ("ZMIDTONE", Some("0.000;0.500;0.000;0.000")),
            ("ZLEVELSTARGETSHADOW", Some("0.000;0.100;0.000;0.000")),
            ("ZLEVELSTARGETHIGHLIGHT", Some("1.000;0.900;1.000;1.000")),
        ]));
        assert_eq!(s.levels_midtone, [0.0, 0.5, 0.0, 0.0]);
        let adj = to_adjustment_set(&s);
        assert!((adj.levels.rgb.gamma - 1.0).abs() < 1e-4);
        assert!((adj.levels.red.gamma - 2.0).abs() < 1e-4);
        assert!((adj.levels.green.gamma - 1.0).abs() < 1e-4);
        assert!((adj.levels.red.output_black - 0.1).abs() < 1e-6);
        assert!((adj.levels.red.output_white - 0.9).abs() < 1e-6);
        assert_eq!(adj.levels.rgb.output_black, 0.0);
        assert_eq!(adj.levels.rgb.output_white, 1.0);
    }

    #[test]
    fn reads_the_real_sample_row() {
        let s = from_layer_row(&row(&[
            ("Z_ENT", Some("17")),
            ("ZEXPOSURE", Some("-0.6492753624916077")),
            ("ZCONTRAST", Some("3.768115997314453")),
            ("ZBRIGHTNESS", Some("15.977961540222168")),
            ("ZSATURATION", Some("-20.936634063720703")),
            ("ZWHITEBALANCE", Some("1.625000;1.000000;1.412000")),
            ("ZLEVELSSHADOW", Some("0.000;0.000;0.000;0.000")),
            ("ZLEVELSHIGHLIGHT", Some("1.000;1.000;1.000;1.000")),
            (
                "ZCROP",
                Some("2525.8076171875;2348.0876464844;3328.8486328125;4993.2734375000"),
            ),
        ]));
        assert_eq!(s.entity, C1EntityKind::Adjustment);
        assert!((s.exposure + 0.6492753624916077).abs() < 1e-6);
        assert!((s.contrast - 3.768115997314453).abs() < 1e-6);
        assert!((s.brightness - 15.977961540222168).abs() < 1e-5);
        assert!((s.saturation + 20.936634063720703).abs() < 1e-5);
        assert!((s.white_balance_gains[0] - 1.625).abs() < 1e-6);
        assert!((s.white_balance_gains[1] - 1.0).abs() < 1e-6);
        assert!((s.white_balance_gains[2] - 1.412).abs() < 1e-6);
        assert_eq!(s.levels_shadow, [0.0; 4]);
        assert_eq!(s.levels_highlight, [1.0; 4]);
        assert!((s.crop_px[0] - 2525.8076171875).abs() < 1e-3);
        assert!((s.crop_px[3] - 4993.2734375).abs() < 1e-3);
    }

    #[test]
    fn defaults_are_neutral() {
        let s = C1VariantSettings::defaults();
        assert_eq!(s.exposure, 0.0);
        assert_eq!(s.white_balance_gains, [1.0, 1.0, 1.0]);
        assert_eq!(s.levels_shadow, [0.0; 4]);
        assert_eq!(s.levels_highlight, [1.0; 4]);
        assert_eq!(s.sharpening.radius, 0.8);
        assert_eq!(s.sharpening.threshold, 1.0);
    }

    #[test]
    fn conversion_keeps_exposure_in_range() {
        let s = from_layer_row(&row(&[
            ("ZEXPOSURE", Some("-0.6492753624916077")),
            ("ZCONTRAST", Some("3.768115997314453")),
            ("ZBRIGHTNESS", Some("15.977961540222168")),
            ("ZSATURATION", Some("-20.936634063720703")),
        ]));
        let set = to_adjustment_set(&s);
        assert!((-4.0..=4.0).contains(&set.exposure.exposure));
        assert!((-50.0..=50.0).contains(&set.exposure.contrast));
        assert!((-100.0..=100.0).contains(&set.exposure.brightness));
        assert!((-100.0..=100.0).contains(&set.exposure.saturation));
        assert!((set.exposure.exposure + 0.6492753).abs() < 1e-5);
        assert!((set.exposure.contrast - 3.768116).abs() < 1e-5);
        assert_eq!(set.white_balance.mode, WhiteBalanceMode::Custom);
        assert!((2000.0..=50000.0).contains(&set.white_balance.kelvin));
        assert!((-50.0..=50.0).contains(&set.white_balance.tint));
    }

    #[test]
    fn gains_and_kelvin_tint_round_trip() {
        let gains = [1.625, 1.0, 1.412];
        let (kelvin, tint) = gains_to_kelvin_tint(gains);
        assert!((kelvin - 5648.0).abs() < 60.0, "kelvin {kelvin}");
        assert!((tint + 41.5).abs() < 3.0, "tint {tint}");
        let back = kelvin_tint_to_gains(kelvin, tint);
        for (a, b) in gains.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn neutral_gains_map_to_the_reference_temperature() {
        let (kelvin, tint) = gains_to_kelvin_tint([1.0, 1.0, 1.0]);
        assert!((kelvin - 6500.0).abs() < 1.0);
        assert!(tint.abs() < 1e-3);
    }

    #[test]
    fn crop_is_normalized_when_the_canvas_is_known() {
        let mut s = C1VariantSettings::defaults();
        s.canvas_size = [6000.0, 4000.0];
        s.crop_px = [1500.0, 1000.0, 4500.0, 3000.0];
        let set = to_adjustment_set(&s);
        assert!((set.geometry.crop.x - 0.25).abs() < 1e-5);
        assert!((set.geometry.crop.width - 0.5).abs() < 1e-5);
    }

    #[test]
    fn brush_layer_columns_are_read() {
        let s = from_layer_row(&row(&[
            ("Z_ENT", Some("23")),
            ("ZNAME", Some("Skin Mask 1")),
            ("ZOPACITY", Some("100.0")),
            ("ZMASKUUID", Some("24391591-4129-4c9f-a9f5-bb4dd4c41289")),
            ("ZBRUSHSIZE", Some("12.96581745147705")),
            ("ZBRUSHHARDNESS", Some("80.0")),
            ("ZERASEROPACITY", Some("100.0")),
        ]));
        assert_eq!(s.entity, C1EntityKind::LocalMask);
        assert_eq!(s.name.as_deref(), Some("Skin Mask 1"));
        assert_eq!(s.mask_uuid.as_deref(), Some("24391591-4129-4c9f-a9f5-bb4dd4c41289"));
        assert!((s.brush.size - 12.96581745147705).abs() < 1e-4);
        assert_eq!(s.brush.hardness, 80.0);
        assert_eq!(s.eraser.opacity, 100.0);
    }

    #[test]
    fn moire_and_black_white_reach_the_adjustment_set() {
        let s = from_layer_row(&row(&[
            ("ZMOIRE", Some("0.000;8")),
            ("ZBWENABLED", Some("1")),
            ("ZBWRED", Some("12.5")),
            ("ZBWDARKTONEHUE", Some("210.0")),
        ]));
        let set = to_adjustment_set(&s);
        assert_eq!(set.moire.amount, 0.0);
        assert_eq!(set.moire.pattern, 8.0);
        assert!(set.black_white.enabled);
        assert!((set.black_white.sensitivity[0] - 12.5).abs() < 1e-5);
        assert!((set.black_white.shadow_tone.hue - 210.0).abs() < 1e-4);
    }

    #[test]
    fn usm_threshold_keeps_the_c1_bound_and_clamps_above_it() {
        let at_bound = to_adjustment_set(&from_layer_row(&row(&[("ZUSMTHRESHOLD", Some("12"))])));
        assert_eq!(at_bound.sharpening.threshold, 12.0);
        let over = to_adjustment_set(&from_layer_row(&row(&[("ZUSMTHRESHOLD", Some("30"))])));
        assert_eq!(over.sharpening.threshold, 12.0);
    }

    #[test]
    fn black_white_sensitivity_clamps_each_channel_to_plus_minus_100() {
        let s = from_layer_row(&row(&[
            ("ZBWRED", Some("120")),
            ("ZBWYELLOW", Some("-130")),
            ("ZBWGREEN", Some("100")),
            ("ZBWMAGENTA", Some("-100")),
        ]));
        let set = to_adjustment_set(&s);
        assert_eq!(set.black_white.sensitivity, [100.0, -100.0, 100.0, 0.0, 0.0, -100.0]);
    }

    /// `ZFILMCURVE` names a file, so the renderer needs the name itself, not
    /// just the preset the name happens to resemble.
    #[test]
    fn film_curve_name_reaches_the_adjustment_set() {
        let s = from_layer_row(&row(&[("ZFILMCURVE", Some("NikonC1-Look Portrait"))]));
        let set = to_adjustment_set(&s);
        assert_eq!(set.base.film_curve.as_deref(), Some("NikonC1-Look Portrait"));
        assert_eq!(set.base.curve, ToneCurvePreset::Auto, "an unrecognised name keeps the default preset");
    }

    /// C1 stores file names; the renderer looks both up by stem, so the
    /// extension has to be dropped or the camera profile never applies.
    #[test]
    fn file_names_lose_their_extension() {
        let s = from_layer_row(&row(&[
            ("ZICCPROFILE", Some("NikonZ6-ProStandard.icm")),
            ("ZFILMCURVE", Some("NikonZ6-Auto.fcrv")),
        ]));
        let set = to_adjustment_set(&s);
        assert_eq!(set.base.icc_profile.as_deref(), Some("NikonZ6-ProStandard"));
        assert_eq!(set.base.film_curve.as_deref(), Some("NikonZ6-Auto"));
        assert_eq!(set.base.curve, ToneCurvePreset::Auto);
    }

    #[test]
    fn embedded_icc_means_no_camera_profile() {
        let s = from_layer_row(&row(&[("ZICCPROFILE", Some("EMBEDDED"))]));
        let set = to_adjustment_set(&s);
        assert_eq!(set.base.icc_profile, None);
    }
}
